#!/usr/bin/env python3
"""Annotate cross-model universal CLT triplets by max-activating positions.

This is the first guarded version of the R2 "universal primitives" pivot.  It
does not require DSSP or Swiss-Prot residue annotations.  Instead, it extracts
per-position activations for the already discovered three-model triplets and
tests cheap labels first: amino-acid identity, coarse residue chemistry,
sequence source, and normalized sequence position.

The output is intended as an A-0/A-1 pilot.  If the cheap labels show coherent
enrichment, the same script can be extended with structural/per-residue labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.analysis.circuit_discovery import load_trained_clt
from src.models.model_loader import load_model


AA = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AILMFWVY")
POLAR = set("STNQCY")
ACIDIC = set("DE")
BASIC = set("KRH")
SPECIAL = set("GP")


def clean_sequence(seq: str) -> str:
    return "".join(c for c in (seq or "").upper() if c in AA)


def parse_model_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"bad --model-spec {spec!r}; expected model=checkpoint_dir")
    name, ckpt = spec.split("=", 1)
    return name.strip(), ckpt.strip()


def descend_json(data, field: str):
    cur = data
    for token in field.split("."):
        if not token:
            continue
        if isinstance(cur, dict):
            cur = cur[token]
        else:
            raise KeyError(f"cannot descend into {field!r}")
    return cur


def read_json_records(path: Path, field: str, max_sequences: int, max_length: int) -> list[dict]:
    data = json.loads(path.read_text())
    rows = descend_json(data, field)
    out = []
    for i, item in enumerate(rows):
        if isinstance(item, str):
            rec = {"id": f"seq_{i:06d}", "source": "unknown", "sequence": item, "meta": {}}
        elif isinstance(item, dict):
            seq = item.get("sequence") or item.get("seq") or item.get("protein_sequence")
            rec = {
                "id": str(item.get("id") or item.get("name") or f"seq_{i:06d}"),
                "source": str(item.get("source") or item.get("label") or item.get("class") or "unknown"),
                "sequence": seq,
                "meta": item.get("meta", {}),
            }
        else:
            continue
        seq = clean_sequence(rec["sequence"])[:max_length]
        if seq:
            rec["sequence"] = seq
            out.append(rec)
        if len(out) >= max_sequences:
            break
    return out


def read_fasta(path: Path, max_sequences: int, max_length: int) -> list[dict]:
    out = []
    cur_id = None
    cur = []
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur:
                    seq = clean_sequence("".join(cur))[:max_length]
                    if seq:
                        out.append({"id": cur_id or f"seq_{len(out):06d}", "source": "fasta", "sequence": seq, "meta": {}})
                    if len(out) >= max_sequences:
                        return out
                cur_id = line[1:].split()[0]
                cur = []
            else:
                cur.append(line)
        if cur and len(out) < max_sequences:
            seq = clean_sequence("".join(cur))[:max_length]
            if seq:
                out.append({"id": cur_id or f"seq_{len(out):06d}", "source": "fasta", "sequence": seq, "meta": {}})
    return out


def load_sequences(args) -> list[dict]:
    if args.json:
        return read_json_records(args.json, args.json_field, args.max_sequences, args.max_length)
    if args.fasta:
        return read_fasta(args.fasta, args.max_sequences, args.max_length)
    raise ValueError("provide --json or --fasta")


def read_triplets(path: Path, max_triplets: int | None = None) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    triplets = []
    for rank, row in enumerate(rows, 1):
        rec = {
            "triplet_id": f"T{rank:03d}",
            "rank": rank,
            "anchor_layer": int(row.get("anchor_layer", 0)),
            "min_abs_corr": float(row.get("min_abs_corr", "nan")),
            "mean_abs_corr": float(row.get("mean_abs_corr", "nan")),
            "features": {},
        }
        for key, value in row.items():
            if key.endswith("_feature"):
                model = key[: -len("_feature")]
                layer_key = f"{model}_layer"
                if layer_key not in row:
                    continue
                rec["features"][model] = {
                    "layer": int(row[layer_key]),
                    "feature": int(value),
                }
        if len(rec["features"]) >= 2:
            triplets.append(rec)
        if max_triplets and len(triplets) >= max_triplets:
            break
    return triplets


def residue_class(aa: str) -> str:
    if aa in HYDROPHOBIC:
        return "hydrophobic"
    if aa in ACIDIC:
        return "acidic"
    if aa in BASIC:
        return "basic"
    if aa in POLAR:
        return "polar"
    if aa in SPECIAL:
        return "gly_pro"
    return "other"


def position_bin(pos: int, length: int) -> str:
    if length <= 0:
        return "unknown"
    x = (pos + 0.5) / length
    if x < 0.25:
        return "N_quarter"
    if x < 0.75:
        return "middle"
    return "C_quarter"


def token_residue_spans(tokenizer, input_ids: torch.Tensor, sequence: str) -> list[list[int]]:
    """Best-effort mapping from model tokens to residue indices.

    Protein tokenizers in this repo are not guaranteed to be single-residue.
    We decode each token, keep amino-acid characters, and greedily align them
    to the sequence.  Empty/special tokens map to an empty span.
    """
    ids = input_ids.detach().cpu().view(-1).tolist()
    spans: list[list[int]] = []
    cursor = 0
    for tok_id in ids:
        piece = tokenizer.decode([tok_id], skip_special_tokens=True)
        letters = clean_sequence(piece)
        if not letters:
            spans.append([])
            continue
        found = sequence.find(letters, cursor)
        if found < 0 and len(letters) == 1:
            # Some tokenizers decode with whitespace artifacts; allow a short
            # local search before giving up.
            for j in range(cursor, min(len(sequence), cursor + 8)):
                if sequence[j] == letters:
                    found = j
                    break
        if found < 0:
            spans.append([])
            continue
        span = list(range(found, min(len(sequence), found + len(letters))))
        spans.append(span)
        cursor = max(cursor, found + len(letters))
    return spans


def add_token_values_to_residues(
    token_values: np.ndarray,
    spans: list[list[int]],
    length: int,
) -> np.ndarray:
    arr = np.full(length, np.nan, dtype=np.float32)
    for token_idx, span in enumerate(spans[: len(token_values)]):
        if not span:
            continue
        val = float(token_values[token_idx])
        for pos in span:
            if 0 <= pos < length:
                if np.isnan(arr[pos]) or val > arr[pos]:
                    arr[pos] = val
    return arr


@torch.no_grad()
def collect_model_values(
    model_name: str,
    checkpoint_dir: str,
    sequences: list[dict],
    triplets: list[dict],
    device: str,
) -> dict[str, list[np.ndarray]]:
    needed_by_layer: dict[int, set[int]] = defaultdict(set)
    for t in triplets:
        spec = t["features"].get(model_name)
        if spec:
            needed_by_layer[int(spec["layer"])].add(int(spec["feature"]))
    if not needed_by_layer:
        return {}

    print(f"\n[{model_name}] loading model and CLT")
    pm = load_model(model_name, device=device)
    clt = load_trained_clt(checkpoint_dir, device=device)
    out = {t["triplet_id"]: [] for t in triplets if model_name in t["features"]}

    t0 = time.time()
    for i, rec in enumerate(sequences):
        seq = rec["sequence"]
        input_ids = pm.tokenize(seq)
        spans = token_residue_spans(pm.tokenizer, input_ids, seq)
        cache = pm.get_activations(input_ids)
        feats = clt.encode([x.float() for x in cache.resid_pre])

        for t in triplets:
            spec = t["features"].get(model_name)
            if not spec:
                continue
            layer = int(spec["layer"])
            feat_idx = int(spec["feature"])
            vals = feats[layer][0, :, feat_idx].detach().float().cpu().numpy()
            out[t["triplet_id"]].append(add_token_values_to_residues(vals, spans, len(seq)))

        if (i + 1) % 10 == 0 or i == len(sequences) - 1:
            print(f"  {model_name}: {i+1}/{len(sequences)} sequences ({time.time()-t0:.1f}s)")

    del pm
    del clt
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def mutual_information(labels: list[str], binary: list[int]) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    joint = Counter(zip(labels, binary))
    lc = Counter(labels)
    bc = Counter(binary)
    mi = 0.0
    for (label, b), count in joint.items():
        pxy = count / n
        px = lc[label] / n
        py = bc[b] / n
        if pxy > 0 and px > 0 and py > 0:
            mi += pxy * math.log(pxy / (px * py))
    return float(mi)


def log2_enrichment(top_counts: Counter, bg_counts: Counter, key: str) -> float:
    top_n = sum(top_counts.values())
    bg_n = sum(bg_counts.values())
    top_p = (top_counts[key] + 0.5) / (top_n + 0.5 * max(len(bg_counts), 1))
    bg_p = (bg_counts[key] + 0.5) / (bg_n + 0.5 * max(len(bg_counts), 1))
    return float(math.log2(top_p / bg_p))


def interpret_triplet(top_class: str, top_aa: str, class_l2e: float, aa_l2e: float, mi_best: float) -> str:
    if mi_best < 0.02 and class_l2e < 0.75 and aa_l2e < 1.0:
        return "uninterpreted_low_enrichment"
    if top_class == "hydrophobic":
        return "hydrophobic-composition primitive"
    if top_class in {"acidic", "basic"}:
        return f"{top_class}-charged primitive"
    if top_class == "gly_pro":
        return "glycine/proline flexibility primitive"
    if top_aa == "C":
        return "cysteine/disulfide-context primitive"
    if top_class == "polar":
        return "polar-residue primitive"
    return f"{top_aa}-enriched sequence-context primitive"


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def analyze(args, sequences: list[dict], triplets: list[dict], model_values: dict[str, dict[str, list[np.ndarray]]]) -> dict:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bg_aa = Counter()
    bg_class = Counter()
    bg_source = Counter()
    bg_pos = Counter()
    all_labels = []
    for rec in sequences:
        seq = rec["sequence"]
        for pos, aa in enumerate(seq):
            bg_aa[aa] += 1
            bg_class[residue_class(aa)] += 1
            bg_source[rec["source"]] += 1
            bg_pos[position_bin(pos, len(seq))] += 1
            all_labels.append((aa, residue_class(aa), rec["source"], position_bin(pos, len(seq))))

    per_triplet_rows = []
    aa_rows = []
    enrich_rows = []
    mi_rows = []
    interp_lines = [
        "# Universal Primitive Annotation Pilot",
        "",
        f"- Sequences: {len(sequences)}",
        f"- Triplets analyzed: {len(triplets)}",
        f"- Top positions per triplet: {args.top_positions}",
        "- Labels used: amino-acid identity, coarse residue chemistry, source, sequence-position bin",
        "- Structural labels: not used in this pilot",
        "",
        "| Triplet | Min abs(r) | Top class | Top AA | Best MI (nats) | Interpretation |",
        "|---|---:|---|---|---:|---|",
    ]

    for t in triplets:
        tid = t["triplet_id"]
        per_model_z = []
        for model_name in model_values:
            vals = model_values[model_name].get(tid)
            if not vals:
                continue
            flat = np.concatenate([x[~np.isnan(x)] for x in vals if np.isfinite(x).any()])
            if flat.size == 0:
                continue
            mu = float(flat.mean())
            sd = float(flat.std() + 1e-6)
            per_model_z.append([(x - mu) / sd for x in vals])
        if not per_model_z:
            continue

        events = []
        labels_aa = []
        labels_class = []
        labels_source = []
        labels_pos = []
        consensus_values = []
        for seq_idx, rec in enumerate(sequences):
            seq = rec["sequence"]
            for pos, aa in enumerate(seq):
                zs = []
                for model_arrays in per_model_z:
                    if seq_idx >= len(model_arrays):
                        continue
                    arr = model_arrays[seq_idx]
                    if pos < len(arr) and not np.isnan(arr[pos]):
                        zs.append(float(arr[pos]))
                if not zs:
                    continue
                value = float(np.mean(zs))
                labels_aa.append(aa)
                labels_class.append(residue_class(aa))
                labels_source.append(rec["source"])
                labels_pos.append(position_bin(pos, len(seq)))
                consensus_values.append(value)
                events.append({
                    "triplet_id": tid,
                    "rank": t["rank"],
                    "seq_idx": seq_idx,
                    "seq_id": rec["id"],
                    "source": rec["source"],
                    "position_0based": pos,
                    "aa": aa,
                    "residue_class": residue_class(aa),
                    "position_bin": position_bin(pos, len(seq)),
                    "consensus_z": value,
                })

        if not events:
            continue
        order = np.argsort(-np.asarray(consensus_values, dtype=np.float64))
        top_mask = np.zeros(len(consensus_values), dtype=np.int8)
        top_mask[order[: min(args.top_positions, len(order))]] = 1
        binary = top_mask.tolist()
        mi_aa = mutual_information(labels_aa, binary)
        mi_class = mutual_information(labels_class, binary)
        mi_source = mutual_information(labels_source, binary)
        mi_pos = mutual_information(labels_pos, binary)
        best_label, best_mi = max(
            [("aa", mi_aa), ("residue_class", mi_class), ("source", mi_source), ("position_bin", mi_pos)],
            key=lambda x: x[1],
        )
        mi_rows.append({
            "triplet_id": tid,
            "rank": t["rank"],
            "mi_aa": f"{mi_aa:.6g}",
            "mi_residue_class": f"{mi_class:.6g}",
            "mi_source": f"{mi_source:.6g}",
            "mi_position_bin": f"{mi_pos:.6g}",
            "best_label": best_label,
            "best_mi": f"{best_mi:.6g}",
            "n_positions_scored": len(consensus_values),
            "n_top_positions": int(sum(binary)),
        })

        events.sort(key=lambda r: r["consensus_z"], reverse=True)
        top_events = events[: args.top_positions]
        top_aa = Counter(e["aa"] for e in top_events)
        top_class = Counter(e["residue_class"] for e in top_events)
        top_source = Counter(e["source"] for e in top_events)
        top_pos = Counter(e["position_bin"] for e in top_events)
        for e in top_events:
            per_triplet_rows.append({
                **e,
                "consensus_z": f"{e['consensus_z']:.6g}",
            })

        for aa in sorted(bg_aa):
            aa_rows.append({
                "triplet_id": tid,
                "rank": t["rank"],
                "aa": aa,
                "top_count": top_aa[aa],
                "background_count": bg_aa[aa],
                "log2_enrichment": f"{log2_enrichment(top_aa, bg_aa, aa):.6g}",
            })
        for key in sorted(bg_class):
            enrich_rows.append({
                "triplet_id": tid,
                "rank": t["rank"],
                "label_type": "residue_class",
                "label": key,
                "top_count": top_class[key],
                "background_count": bg_class[key],
                "log2_enrichment": f"{log2_enrichment(top_class, bg_class, key):.6g}",
            })
        for key in sorted(bg_source):
            enrich_rows.append({
                "triplet_id": tid,
                "rank": t["rank"],
                "label_type": "source",
                "label": key,
                "top_count": top_source[key],
                "background_count": bg_source[key],
                "log2_enrichment": f"{log2_enrichment(top_source, bg_source, key):.6g}",
            })
        for key in sorted(bg_pos):
            enrich_rows.append({
                "triplet_id": tid,
                "rank": t["rank"],
                "label_type": "position_bin",
                "label": key,
                "top_count": top_pos[key],
                "background_count": bg_pos[key],
                "log2_enrichment": f"{log2_enrichment(top_pos, bg_pos, key):.6g}",
            })

        best_class = max(bg_class, key=lambda k: log2_enrichment(top_class, bg_class, k))
        best_aa_key = max(bg_aa, key=lambda k: log2_enrichment(top_aa, bg_aa, k))
        class_l2e = log2_enrichment(top_class, bg_class, best_class)
        aa_l2e = log2_enrichment(top_aa, bg_aa, best_aa_key)
        interpretation = interpret_triplet(best_class, best_aa_key, class_l2e, aa_l2e, best_mi)
        interp_lines.append(
            f"| {tid} | {t['min_abs_corr']:.4f} | {best_class} ({class_l2e:+.2f}) | "
            f"{best_aa_key} ({aa_l2e:+.2f}) | {best_mi:.4f} | {interpretation} |"
        )

    write_tsv(
        args.out_dir / "per_triplet_max_act.tsv",
        per_triplet_rows,
        ["triplet_id", "rank", "seq_idx", "seq_id", "source", "position_0based", "aa", "residue_class", "position_bin", "consensus_z"],
    )
    with (args.out_dir / "per_triplet_max_act.jsonl").open("w") as f:
        for row in per_triplet_rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    write_tsv(
        args.out_dir / "aa_composition.tsv",
        aa_rows,
        ["triplet_id", "rank", "aa", "top_count", "background_count", "log2_enrichment"],
    )
    write_tsv(
        args.out_dir / "simple_enrichment.tsv",
        enrich_rows,
        ["triplet_id", "rank", "label_type", "label", "top_count", "background_count", "log2_enrichment"],
    )
    write_tsv(
        args.out_dir / "mutual_information.tsv",
        mi_rows,
        [
            "triplet_id",
            "rank",
            "mi_aa",
            "mi_residue_class",
            "mi_source",
            "mi_position_bin",
            "best_label",
            "best_mi",
            "n_positions_scored",
            "n_top_positions",
        ],
    )
    (args.out_dir / "interpretation.md").write_text("\n".join(interp_lines) + "\n")
    summary = {
        "task": "R2 universal primitive annotation pilot",
        "status": "completed",
        "n_sequences": len(sequences),
        "n_triplets": len(triplets),
        "top_positions": args.top_positions,
        "outputs": {
            "per_triplet_max_act": str(args.out_dir / "per_triplet_max_act.jsonl"),
            "aa_composition": str(args.out_dir / "aa_composition.tsv"),
            "simple_enrichment": str(args.out_dir / "simple_enrichment.tsv"),
            "mutual_information": str(args.out_dir / "mutual_information.tsv"),
            "interpretation": str(args.out_dir / "interpretation.md"),
        },
        "resource_status": {
            "cheap_sequence_labels": "used",
            "dssp": "not_used_in_pilot",
            "solvent_accessibility": "not_used_in_pilot",
            "swiss_prot_per_residue": "not_used_in_pilot",
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplets", type=Path, required=True)
    ap.add_argument("--model-spec", action="append", required=True)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--json-field", default="records")
    ap.add_argument("--fasta", type=Path, default=None)
    ap.add_argument("--max-sequences", type=int, default=200)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-triplets", type=int, default=10)
    ap.add_argument("--top-positions", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    specs = [parse_model_spec(s) for s in args.model_spec]
    sequences = load_sequences(args)
    triplets = read_triplets(args.triplets, args.max_triplets)
    if not sequences:
        raise ValueError("no sequences loaded")
    if not triplets:
        raise ValueError("no triplets loaded")
    print("=" * 72)
    print("R2 universal primitive annotation pilot")
    print("=" * 72)
    print(f"Sequences: {len(sequences)}")
    print(f"Triplets: {len(triplets)}")

    model_values = {}
    for model_name, ckpt in specs:
        vals = collect_model_values(model_name, ckpt, sequences, triplets, args.device)
        if vals:
            model_values[model_name] = vals
    summary = analyze(args, sequences, triplets, model_values)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
