#!/usr/bin/env python3
"""Swiss-Prot anchored rich-label annotation for universal triplets.

This is the N-1 experiment from OPUS_NEXT_20260513.md.  It repeats the
top-firing-position analysis on a cohort where Pfam and Swiss-Prot residue
labels are dense, then computes mutual information between triplet firing and
rich biological labels.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


u29 = load_module(SCRIPT_DIR / "29_universal_primitive_annotation.py", "universal_primitive_annotation_29")


def load_pfam_intervals(path: Path) -> dict[str, list[tuple[int, int, str]]]:
    out: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                out[row["uniprot"]].append((int(row["start"]), int(row["end"]), row["pfam_id"]))
            except (KeyError, ValueError):
                continue
    return dict(out)


def load_swissprot(path: Path) -> list:
    # The cache was pickled with Research1's package layout
    # (`src.data.swissprot_parser`).  This script also imports Research2's
    # `src.*`, so temporarily swap the `src` module namespace only while
    # unpickling.
    candidate_roots = [
        REPO / "r1_encoder_interpretability_benchmark",
        REPO / "Research1",
        Path("/oss-pvc/zhk_zip/biocc/Research1"),
        Path("/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research1"),
    ]
    try:
        r1_path = str(next(p for p in candidate_roots if (p / "src/data/swissprot_parser.py").exists()))
    except StopIteration as exc:
        roots = ", ".join(str(p) for p in candidate_roots)
        raise FileNotFoundError(f"cannot unpickle Swiss-Prot cache; missing Research1 src.data in: {roots}") from exc
    saved_path = list(sys.path)
    saved_src_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "src" or name.startswith("src.")
    }
    for name in list(saved_src_modules):
        sys.modules.pop(name, None)
    sys.path.insert(0, r1_path)
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    finally:
        for name in [x for x in sys.modules if x == "src" or x.startswith("src.")]:
            sys.modules.pop(name, None)
        sys.modules.update(saved_src_modules)
        sys.path[:] = saved_path


def feature_labels(features: list[tuple[int, int, str, str, str]], pos_1based: int) -> dict[str, str]:
    cats = []
    types = []
    secondary = []
    functional = []
    domain = []
    ptm = []
    topology = []
    region = []
    for start, end, feat_type, desc, category in features:
        if int(start) <= pos_1based <= int(end):
            cats.append(category)
            types.append(feat_type)
            if category == "secondary_structure":
                secondary.append(feat_type)
            elif category == "functional":
                functional.append(feat_type if not desc else f"{feat_type}:{desc}")
            elif category == "domain":
                domain.append(feat_type if not desc else f"{feat_type}:{desc}")
            elif category == "ptm":
                ptm.append(feat_type)
            elif category == "topology":
                topology.append(feat_type)
            elif category == "region":
                region.append(feat_type if not desc else f"{feat_type}:{desc}")
    return {
        "swiss_category": "+".join(sorted(set(cats))) or "none",
        "swiss_feature_type": "+".join(sorted(set(types))) or "none",
        "secondary_structure": "+".join(sorted(set(secondary))) or "none",
        "functional_label": "+".join(sorted(set(functional))) or "none",
        "domain_label": "+".join(sorted(set(domain))) or "none",
        "ptm_label": "+".join(sorted(set(ptm))) or "none",
        "topology_label": "+".join(sorted(set(topology))) or "none",
        "region_label": "+".join(sorted(set(region))) or "none",
    }


def pfam_at(intervals: list[tuple[int, int, str]], pos_1based: int) -> str:
    hits = sorted({pfam for start, end, pfam in intervals if start <= pos_1based <= end})
    return "+".join(hits) if hits else "none"


def dominant_pfam(intervals: list[tuple[int, int, str]]) -> str:
    counts = Counter()
    for start, end, pfam in intervals:
        counts[pfam] += max(1, int(end) - int(start) + 1)
    return counts.most_common(1)[0][0] if counts else "none"


def choose_cohort(
    swissprot_cache: Path,
    pfam_residue: Path,
    max_sequences: int,
    min_len: int,
    max_len: int,
    seed: int,
) -> tuple[list[dict], dict, dict]:
    anns = load_swissprot(swissprot_cache)
    pfam = load_pfam_intervals(pfam_residue)
    by_pfam: dict[str, list] = defaultdict(list)
    ann_by_acc = {}
    for ann in anns:
        acc = getattr(ann, "accession", "")
        seq = getattr(ann, "sequence", "")
        if not acc or not seq or not (min_len <= len(seq) <= max_len):
            continue
        intervals = pfam.get(acc, [])
        if not intervals:
            continue
        dom = dominant_pfam(intervals)
        by_pfam[dom].append(ann)
        ann_by_acc[acc] = ann

    rng = np.random.default_rng(seed)
    for xs in by_pfam.values():
        rng.shuffle(xs)
    selected = []
    seen = set()
    # Prefer a broad, stratified cohort: one pass over Pfam families, repeated
    # until the requested size is reached.
    pfams = [p for p, xs in Counter({p: len(xs) for p, xs in by_pfam.items()}).items()]
    pfams = sorted(by_pfam, key=lambda p: len(by_pfam[p]), reverse=True)
    offset = 0
    while len(selected) < max_sequences:
        added = 0
        for pf in pfams:
            xs = by_pfam[pf]
            if offset >= len(xs):
                continue
            ann = xs[offset]
            acc = ann.accession
            if acc in seen:
                continue
            selected.append(
                {
                    "id": acc,
                    "source": "swissprot",
                    "sequence": ann.sequence,
                    "dominant_pfam": pf,
                    "organism": getattr(ann, "organism", ""),
                }
            )
            seen.add(acc)
            added += 1
            if len(selected) >= max_sequences:
                break
        if added == 0:
            break
        offset += 1

    meta = {
        "n_swissprot_total": len(anns),
        "n_length_pfam_eligible": sum(len(xs) for xs in by_pfam.values()),
        "n_pfam_families_eligible": len(by_pfam),
        "n_selected": len(selected),
        "top_selected_pfams": dict(Counter(r["dominant_pfam"] for r in selected).most_common(20)),
        "sampling": "round-robin over dominant Pfam families, sorted by family size",
    }
    return selected, ann_by_acc, pfam, meta


def encode_labels(labels: list[str]) -> tuple[np.ndarray, int]:
    label_to_i = {label: i for i, label in enumerate(sorted(set(labels)))}
    return np.asarray([label_to_i[x] for x in labels], dtype=np.int32), len(label_to_i)


def mutual_information_encoded(codes: np.ndarray, binary: np.ndarray, n_labels: int) -> float:
    n = int(codes.size)
    if n == 0 or n_labels == 0:
        return 0.0
    joint = np.bincount(codes * 2 + binary.astype(np.int32), minlength=n_labels * 2).reshape(n_labels, 2)
    total = float(joint.sum())
    if total <= 0:
        return 0.0
    pxy = joint / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px @ py
    mask = (pxy > 0) & (denom > 0)
    return float(np.sum(pxy[mask] * np.log(pxy[mask] / denom[mask])))


def bootstrap_mi(labels: list[str], binary: list[int], n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    binary_arr = np.asarray(binary, dtype=np.int8)
    codes, n_labels = encode_labels(labels)
    vals = []
    n = len(labels)
    if n == 0:
        return {"ci95": [0.0, 0.0], "ci99": [0.0, 0.0], "n_boot": 0}
    sample_n = min(n, 20000)
    for _ in range(n_boot):
        idx = rng.integers(0, n, sample_n)
        vals.append(mutual_information_encoded(codes[idx], binary_arr[idx], n_labels))
    return {
        "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
        "ci99": [float(np.percentile(vals, 0.5)), float(np.percentile(vals, 99.5))],
        "n_boot": n_boot,
        "bootstrap_sample_n": int(sample_n),
    }


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def consensus_values_for_triplet(model_values: dict, tid: str, n_sequences: int) -> list[np.ndarray]:
    per_model_z = []
    for vals_by_tid in model_values.values():
        vals = vals_by_tid.get(tid)
        if not vals:
            continue
        flat_parts = [x[np.isfinite(x)] for x in vals if np.isfinite(x).any()]
        if not flat_parts:
            continue
        flat = np.concatenate(flat_parts)
        mu = float(flat.mean())
        sd = float(flat.std() + 1e-6)
        per_model_z.append([(x - mu) / sd for x in vals])
    consensus = []
    for seq_idx in range(n_sequences):
        arrs = []
        for model_arrays in per_model_z:
            if seq_idx < len(model_arrays):
                arrs.append(model_arrays[seq_idx])
        if not arrs:
            consensus.append(np.zeros(0, dtype=np.float32))
            continue
        max_len = max(len(a) for a in arrs)
        vals = np.full((len(arrs), max_len), np.nan, dtype=np.float32)
        for i, arr in enumerate(arrs):
            vals[i, : len(arr)] = arr
        consensus.append(np.nanmean(vals, axis=0))
    return consensus


def run_analysis(args, sequences, ann_by_acc, pfam_intervals, triplets, model_values):
    label_keys = [
        "pfam_family",
        "swiss_category",
        "swiss_feature_type",
        "secondary_structure",
        "functional_label",
        "domain_label",
        "ptm_label",
        "topology_label",
        "region_label",
        "dominant_pfam",
        "aa",
    ]
    rich_keys = [k for k in label_keys if k != "aa"]
    per_pos_rows = []
    mi_rows = []
    interp_rows = []

    for t in triplets:
        tid = t["triplet_id"]
        consensus = consensus_values_for_triplet(model_values, tid, len(sequences))
        events = []
        labels_by_key = {k: [] for k in label_keys}
        values = []
        for seq_idx, rec in enumerate(sequences):
            acc = rec["id"]
            seq = rec["sequence"]
            ann = ann_by_acc.get(acc)
            features = getattr(ann, "features", []) if ann is not None else []
            intervals = pfam_intervals.get(acc, [])
            arr = consensus[seq_idx]
            n = min(len(seq), len(arr))
            for pos0 in range(n):
                if not np.isfinite(arr[pos0]):
                    continue
                pos1 = pos0 + 1
                fl = feature_labels(features, pos1)
                label_record = {
                    "aa": seq[pos0],
                    "pfam_family": pfam_at(intervals, pos1),
                    "dominant_pfam": rec.get("dominant_pfam", "none"),
                    **fl,
                }
                for k in label_keys:
                    labels_by_key[k].append(label_record[k])
                values.append(float(arr[pos0]))
                events.append((seq_idx, pos0, float(arr[pos0]), label_record))

        if not events:
            continue
        order = np.argsort(-np.asarray(values, dtype=np.float64))
        top_n = min(args.top_positions, len(order))
        binary = np.zeros(len(values), dtype=np.int8)
        binary[order[:top_n]] = 1

        key_results = []
        for key in label_keys:
            codes, n_labels = encode_labels(labels_by_key[key])
            mi = mutual_information_encoded(codes, binary, n_labels)
            boot = bootstrap_mi(labels_by_key[key], binary.tolist(), args.n_boot, args.seed + t["rank"] + len(key))
            key_results.append((key, mi, boot))
            mi_rows.append(
                {
                    "triplet_id": tid,
                    "rank": t["rank"],
                    "label": key,
                    "mi_nats": f"{mi:.6g}",
                    "ci95_low": f"{boot['ci95'][0]:.6g}",
                    "ci95_high": f"{boot['ci95'][1]:.6g}",
                    "ci99_low": f"{boot['ci99'][0]:.6g}",
                    "ci99_high": f"{boot['ci99'][1]:.6g}",
                    "n_positions": len(values),
                    "n_top": int(top_n),
                    "top_label_counts": json.dumps(
                        Counter(np.asarray(labels_by_key[key], dtype=object)[order[:top_n]].tolist()).most_common(10)
                    ),
                }
            )

        print(
            f"  analyzed {tid} ({t['rank']}/{len(triplets)}): "
            f"best_rich={max([x for x in key_results if x[0] in rich_keys], key=lambda x: x[1])[0]} "
            f"MI={max([x for x in key_results if x[0] in rich_keys], key=lambda x: x[1])[1]:.4g}",
            flush=True,
        )

        best_key, best_mi, best_boot = max(key_results, key=lambda x: x[1])
        best_rich_key, best_rich_mi, best_rich_boot = max(
            [x for x in key_results if x[0] in rich_keys], key=lambda x: x[1]
        )
        gate = "interpretable" if best_rich_mi >= args.mi_gate else "below_gate"
        interp_rows.append(
            {
                "triplet_id": tid,
                "rank": t["rank"],
                "min_abs_corr": f"{t['min_abs_corr']:.6g}",
                "best_label": best_key,
                "best_mi_nats": f"{best_mi:.6g}",
                "best_rich_label": best_rich_key,
                "best_rich_mi_nats": f"{best_rich_mi:.6g}",
                "best_rich_ci95": f"[{best_rich_boot['ci95'][0]:.4g}, {best_rich_boot['ci95'][1]:.4g}]",
                "gate": gate,
            }
        )

        for event_idx in order[:top_n]:
            seq_idx, pos0, value, label_record = events[int(event_idx)]
            rec = sequences[seq_idx]
            per_pos_rows.append(
                {
                    "triplet_id": tid,
                    "rank": t["rank"],
                    "seq_idx": seq_idx,
                    "accession": rec["id"],
                    "position_1based": pos0 + 1,
                    "aa": label_record["aa"],
                    "consensus_z": f"{value:.6g}",
                    **{k: label_record[k] for k in rich_keys},
                }
            )

    n_interpretable = sum(1 for r in interp_rows if r["gate"] == "interpretable")
    if n_interpretable >= 25:
        outcome = "PASS"
    elif n_interpretable >= 10:
        outcome = "PARTIAL"
    else:
        outcome = "FAIL"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(
        args.out_dir / "per_triplet_max_act_rich.tsv",
        per_pos_rows,
        [
            "triplet_id",
            "rank",
            "seq_idx",
            "accession",
            "position_1based",
            "aa",
            "consensus_z",
            *rich_keys,
        ],
    )
    write_tsv(
        args.out_dir / "rich_label_mi.tsv",
        mi_rows,
        [
            "triplet_id",
            "rank",
            "label",
            "mi_nats",
            "ci95_low",
            "ci95_high",
            "ci99_low",
            "ci99_high",
            "n_positions",
            "n_top",
            "top_label_counts",
        ],
    )
    write_tsv(
        args.out_dir / "interpretation_gate.tsv",
        interp_rows,
        [
            "triplet_id",
            "rank",
            "min_abs_corr",
            "best_label",
            "best_mi_nats",
            "best_rich_label",
            "best_rich_mi_nats",
            "best_rich_ci95",
            "gate",
        ],
    )

    lines = [
        "# Swiss-Prot Triplet Annotation",
        "",
        f"- Cohort size: {len(sequences)}",
        f"- Triplets: {len(triplets)}",
        f"- Top positions per triplet: {args.top_positions}",
        f"- MI gate: {args.mi_gate} nats",
        f"- Interpretable triplets: {n_interpretable} / {len(triplets)}",
        f"- Outcome: {outcome}",
        "",
        "| Triplet | Best rich label | Best rich MI | 95% CI | Gate |",
        "|---|---|---:|---:|---|",
    ]
    for r in interp_rows:
        lines.append(
            f"| {r['triplet_id']} | {r['best_rich_label']} | {r['best_rich_mi_nats']} | "
            f"{r['best_rich_ci95']} | {r['gate']} |"
        )
    (args.out_dir / "interpretation.md").write_text("\n".join(lines) + "\n")

    summary = {
        "task": "N-1 Swiss-Prot anchored triplet annotation",
        "status": "completed",
        "cohort_size": len(sequences),
        "n_triplets": len(triplets),
        "top_positions": args.top_positions,
        "mi_gate": args.mi_gate,
        "n_interpretable": n_interpretable,
        "outcome": outcome,
        "outputs": {
            "interpretation": str(args.out_dir / "interpretation.md"),
            "interpretation_gate": str(args.out_dir / "interpretation_gate.tsv"),
            "rich_label_mi": str(args.out_dir / "rich_label_mi.tsv"),
            "per_triplet_max_act_rich": str(args.out_dir / "per_triplet_max_act_rich.tsv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplets", type=Path, required=True)
    ap.add_argument("--swissprot-cache", type=Path, default=REPO / "data" / "processed" / "swissprot_all_max1022.pkl")
    ap.add_argument("--pfam-residue", type=Path, default=REPO / "data" / "interpro" / "pfam_residue.tsv")
    ap.add_argument("--max-sequences", type=int, default=500)
    ap.add_argument("--min-len", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--max-triplets", type=int, default=38)
    ap.add_argument("--top-positions", type=int, default=100)
    ap.add_argument("--mi-gate", type=float, default=0.1)
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260513)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model-spec", action="append", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    print("=" * 72)
    print("N-1 Swiss-Prot anchored triplet annotation")
    print("=" * 72)
    t0 = time.time()
    sequences, ann_by_acc, pfam, cohort_meta = choose_cohort(
        args.swissprot_cache, args.pfam_residue, args.max_sequences, args.min_len, args.max_len, args.seed
    )
    if not sequences:
        raise ValueError("no Swiss-Prot sequences selected")
    triplets = u29.read_triplets(args.triplets, args.max_triplets)
    print(json.dumps(cohort_meta, indent=2))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cohort.json").write_text(json.dumps({"meta": cohort_meta, "records": sequences}, indent=2) + "\n")

    specs = [u29.parse_model_spec(s) for s in args.model_spec]
    model_values = {}
    for model_name, ckpt in specs:
        vals = u29.collect_model_values(model_name, ckpt, sequences, triplets, args.device)
        if vals:
            model_values[model_name] = vals
    summary = run_analysis(args, sequences, ann_by_acc, pfam, triplets, model_values)
    summary["runtime_seconds"] = time.time() - t0
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
