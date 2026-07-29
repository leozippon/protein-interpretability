#!/usr/bin/env python3
"""Causal ablation test for R2 N-terminal attention-sink triplets.

This script ablates the model-specific CLT feature for T011/T018/T023 during
teacher-forced forward passes and compares the effect to T025 plus same-layer
random controls. The intervention is the same TopK-aware, same-layer MLP-output
patch used by the existing steering code; attention changes are therefore an
indirect downstream readout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

R2 = Path(__file__).resolve().parents[1]
REPO = R2.parent
if str(R2) not in sys.path:
    sys.path.insert(0, str(R2))

from src.analysis.circuit_discovery import load_trained_clt
from src.models.model_loader import load_model


AA = set("ACDEFGHIKLMNPQRSTVWY")
TARGET_TRIPLETS = {"T011", "T018", "T023"}
SPECIFICITY_TRIPLETS = {"T025"}


def clean_sequence(seq: str) -> str:
    return "".join(c for c in (seq or "").upper() if c in AA)


def parse_model_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"bad --model-spec {spec!r}; expected model=checkpoint_dir")
    name, ckpt = spec.split("=", 1)
    return name.strip(), ckpt.strip()


def read_triplets(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out = []
    for rank, row in enumerate(rows, start=1):
        tid = f"T{rank:03d}"
        rec = {
            "triplet_id": tid,
            "rank": rank,
            "anchor_layer": int(row["anchor_layer"]),
            "min_abs_corr": float(row["min_abs_corr"]),
            "mean_abs_corr": float(row["mean_abs_corr"]),
            "features": {},
        }
        for key, value in row.items():
            if not key.endswith("_feature"):
                continue
            model = key[: -len("_feature")]
            layer_key = f"{model}_layer"
            if layer_key in row:
                rec["features"][model] = {"layer": int(row[layer_key]), "feature": int(value)}
        out.append(rec)
    return out


def read_cohort(path: Path, source: str, max_sequences: int, max_length: int) -> list[dict]:
    data = json.loads(path.read_text())
    rows = data.get("records", data if isinstance(data, list) else [])
    out = []
    for i, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        src = str(item.get("source", "unknown"))
        if source != "all" and src != source:
            continue
        seq = clean_sequence(item.get("sequence", ""))[:max_length]
        if not seq:
            continue
        out.append({
            "id": str(item.get("id") or f"seq_{i:06d}"),
            "source": src,
            "sequence": seq,
            "starts_m": int(seq.startswith("M")),
        })
        if len(out) >= max_sequences:
            break
    return out


def token_residue_spans(tokenizer, input_ids: torch.Tensor, sequence: str) -> list[list[int]]:
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


def shifted_nll(logits: torch.Tensor, input_ids: torch.Tensor) -> np.ndarray:
    ids = input_ids
    if ids.dim() == 2:
        ids = ids[0]
    logits = logits[0]
    out = torch.full((ids.shape[0],), float("nan"), device=logits.device, dtype=torch.float32)
    if ids.shape[0] < 2:
        return out.detach().cpu().numpy()
    losses = F.cross_entropy(logits[:-1].float(), ids[1:].long(), reduction="none")
    out[1:] = losses
    return out.detach().cpu().numpy()


def mean_or_nan(vals: list[float]) -> float:
    vals = [float(x) for x in vals if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def summarize_delta_nll(delta: np.ndarray, spans: list[list[int]]) -> dict:
    bins = defaultdict(list)
    for tok_idx, span in enumerate(spans):
        if tok_idx >= len(delta) or not span or not math.isfinite(float(delta[tok_idx])):
            continue
        pos0 = min(span)
        val = float(delta[tok_idx])
        bins["all"].append(val)
        if pos0 <= 1:
            bins["first2"].append(val)
        if 1 <= pos0 <= 9:
            bins["pos2_10"].append(val)
        if 2 <= pos0 <= 9:
            bins["pos3_10"].append(val)
        if pos0 >= 10:
            bins["pos11_plus"].append(val)
    out = {}
    for key in ["all", "first2", "pos2_10", "pos3_10", "pos11_plus"]:
        out[f"delta_nll_{key}"] = mean_or_nan(bins[key])
        out[f"n_tokens_{key}"] = len(bins[key])
    out["specificity_pos2_10_minus_11plus"] = (
        out["delta_nll_pos2_10"] - out["delta_nll_pos11_plus"]
        if math.isfinite(out["delta_nll_pos2_10"]) and math.isfinite(out["delta_nll_pos11_plus"])
        else float("nan")
    )
    return out


def token_mask_for_residues(spans: list[list[int]], predicate) -> np.ndarray:
    mask = np.zeros(len(spans), dtype=bool)
    for i, span in enumerate(spans):
        if span and any(predicate(pos0) for pos0 in span):
            mask[i] = True
    return mask


def attention_received(attentions, spans: list[list[int]], intervention_layer: int) -> np.ndarray | None:
    if attentions is None:
        return None
    layer_indices = list(range(intervention_layer + 1, len(attentions)))
    if not layer_indices:
        layer_indices = [min(intervention_layer, len(attentions) - 1)]
    recvs = []
    for li in layer_indices:
        attn = attentions[li]
        if attn is None:
            continue
        # batch x heads x query x key -> mean attention received by key token
        recv = attn[0].detach().float().mean(dim=0).mean(dim=0).cpu().numpy()
        if len(recv) >= len(spans):
            recvs.append(recv[: len(spans)])
    if not recvs:
        return None
    return np.mean(np.stack(recvs, axis=0), axis=0)


def attention_metrics(intact_attn, ablated_attn, spans: list[list[int]], layer: int) -> dict:
    a = attention_received(intact_attn, spans, layer)
    b = attention_received(ablated_attn, spans, layer)
    if a is None or b is None:
        return {
            "attention_first2_intact": float("nan"),
            "attention_first2_ablated": float("nan"),
            "attention_first2_delta": float("nan"),
            "attention_cosine_distance": float("nan"),
        }
    n = min(len(a), len(b), len(spans))
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    mapped = np.asarray([bool(s) for s in spans[:n]], dtype=bool)
    first2 = token_mask_for_residues(spans[:n], lambda p: p <= 1)
    if mapped.any():
        a_norm = a[mapped] / max(float(a[mapped].sum()), 1e-12)
        b_norm = b[mapped] / max(float(b[mapped].sum()), 1e-12)
        denom = max(float(np.linalg.norm(a_norm) * np.linalg.norm(b_norm)), 1e-12)
        cosine_distance = 1.0 - float(np.dot(a_norm, b_norm) / denom)
    else:
        cosine_distance = float("nan")
    first2_intact = float(a[first2].sum()) if first2.any() else float("nan")
    first2_ablated = float(b[first2].sum()) if first2.any() else float("nan")
    return {
        "attention_first2_intact": first2_intact,
        "attention_first2_ablated": first2_ablated,
        "attention_first2_delta": first2_ablated - first2_intact
        if math.isfinite(first2_intact) and math.isfinite(first2_ablated)
        else float("nan"),
        "attention_cosine_distance": cosine_distance,
    }


def feature_activity_metrics(trace: np.ndarray | None, spans: list[list[int]]) -> dict:
    if trace is None:
        return {
            "feature_first2_max": float("nan"),
            "feature_first2_active": 0,
            "feature_all_max": float("nan"),
        }
    n = min(len(trace), len(spans))
    first2 = token_mask_for_residues(spans[:n], lambda p: p <= 1)
    vals = np.asarray(trace[:n], dtype=np.float32)
    first2_vals = vals[first2]
    first2_max = float(np.nanmax(first2_vals)) if first2_vals.size else float("nan")
    all_max = float(np.nanmax(vals)) if vals.size else float("nan")
    return {
        "feature_first2_max": first2_max,
        "feature_first2_active": int(math.isfinite(first2_max) and first2_max > 0),
        "feature_all_max": all_max,
    }


def make_conditions(triplets: list[dict], model_name: str, d_clt: int, n_random: int, seed: int) -> list[dict]:
    wanted = TARGET_TRIPLETS | SPECIFICITY_TRIPLETS
    by_tid = {t["triplet_id"]: t for t in triplets if t["triplet_id"] in wanted and model_name in t["features"]}
    rng = random.Random(seed + sum(ord(c) for c in model_name))
    conditions = []
    excluded_by_layer = defaultdict(set)
    for t in triplets:
        if model_name in t["features"]:
            spec = t["features"][model_name]
            excluded_by_layer[int(spec["layer"])].add(int(spec["feature"]))
    for tid in sorted(by_tid):
        spec = by_tid[tid]["features"][model_name]
        kind = "target" if tid in TARGET_TRIPLETS else "specificity"
        conditions.append({
            "condition": tid,
            "condition_kind": kind,
            "triplet_id": tid,
            "layer": int(spec["layer"]),
            "feature": int(spec["feature"]),
            "control_for": "",
        })
    for tid in sorted(TARGET_TRIPLETS):
        if tid not in by_tid:
            continue
        spec = by_tid[tid]["features"][model_name]
        layer = int(spec["layer"])
        excluded = set(excluded_by_layer[layer])
        for j in range(n_random):
            feat = rng.randrange(d_clt)
            tries = 0
            while feat in excluded and tries < 10000:
                feat = rng.randrange(d_clt)
                tries += 1
            excluded.add(feat)
            conditions.append({
                "condition": f"RND_{tid}_{j}",
                "condition_kind": "random",
                "triplet_id": "",
                "layer": layer,
                "feature": int(feat),
                "control_for": tid,
            })
    return conditions


def run_forward(pm, clt, input_ids, condition: dict | None, multiplier: float):
    trace = {"feature_values": None}
    hooks = []
    if condition is not None:
        layer = int(condition["layer"])
        feat_idx = int(condition["feature"])

        def hook_fn(module, inputs, output):
            resid = inputs[0].float()
            pre_act = torch.einsum("bsd,fd->bsf", resid, clt.W_enc[layer]) + clt.b_enc[layer]
            pre_act = torch.relu(pre_act)
            k = min(clt.k, pre_act.shape[-1])
            vals, idx = pre_act.topk(k, dim=-1)
            sparse_unsteered = torch.zeros_like(pre_act)
            sparse_unsteered.scatter_(-1, idx, vals)
            if trace["feature_values"] is None:
                trace["feature_values"] = sparse_unsteered[0, :, feat_idx].detach().float().cpu().numpy()
            steered_pre = pre_act.clone()
            steered_pre[:, :, feat_idx] = steered_pre[:, :, feat_idx] * multiplier
            vals2, idx2 = steered_pre.topk(k, dim=-1)
            sparse_steered = torch.zeros_like(steered_pre)
            sparse_steered.scatter_(-1, idx2, vals2)
            dec0 = clt.W_dec[layer][:, 0, :]
            explained_unsteered = torch.einsum("bsf,fd->bsd", sparse_unsteered, dec0)
            explained_steered = torch.einsum("bsf,fd->bsd", sparse_steered, dec0)
            delta = explained_steered - explained_unsteered
            return output + delta.to(output.dtype)

        block = pm._get_block(layer)
        mlp = pm._get_mlp(block)
        hooks.append(mlp.register_forward_hook(hook_fn))
    try:
        with torch.no_grad():
            out = pm.model(input_ids, output_attentions=True, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return out, trace["feature_values"]


def existing_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            keys.add((row["model"], row["condition"], row["sequence_id"]))
    return keys


def write_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = [
        "model", "condition", "condition_kind", "control_for", "triplet_id",
        "layer", "feature", "sequence_id", "source", "starts_m", "seq_len",
        "n_tokens", "feature_first2_active", "feature_first2_max", "feature_all_max",
        "delta_nll_all", "delta_nll_first2", "delta_nll_pos2_10",
        "delta_nll_pos3_10", "delta_nll_pos11_plus",
        "specificity_pos2_10_minus_11plus",
        "n_tokens_all", "n_tokens_first2", "n_tokens_pos2_10",
        "n_tokens_pos3_10", "n_tokens_pos11_plus",
        "attention_first2_intact", "attention_first2_ablated",
        "attention_first2_delta", "attention_cosine_distance",
    ]
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        clean = {}
        for k in fields:
            v = row.get(k, "")
            if isinstance(v, float):
                clean[k] = f"{v:.8g}" if math.isfinite(v) else "nan"
            else:
                clean[k] = v
        writer.writerow(clean)


def run(args) -> None:
    os.environ.setdefault("R2_MODEL_BASE_DIR", "/gpfs/jiaotongdamoxing/zhk_zip/models")
    records = read_cohort(args.cohort, args.source, args.max_sequences, args.max_length)
    if not records:
        raise ValueError("no records loaded")
    triplets = read_triplets(args.triplets)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "per_sequence_metrics.tsv"
    done = set() if args.force else existing_keys(metrics_path)

    for model_name, ckpt in [parse_model_spec(s) for s in args.model_spec]:
        print(f"\n=== {model_name} ===", flush=True)
        pm = load_model(model_name, device=args.device)
        clt = load_trained_clt(ckpt, device=args.device)
        conditions = make_conditions(triplets, model_name, clt.d_clt, args.n_random_controls, args.seed)
        (args.out_dir / f"{model_name}_conditions.json").write_text(json.dumps(conditions, indent=2) + "\n")
        t0 = time.time()
        for i, rec in enumerate(records):
            pending = [c for c in conditions if (model_name, c["condition"], rec["id"]) not in done]
            if not pending:
                continue
            seq = rec["sequence"]
            input_ids = pm.tokenize(seq)
            spans = token_residue_spans(pm.tokenizer, input_ids, seq)
            intact_out, _ = run_forward(pm, clt, input_ids, None, 1.0)
            intact_nll = shifted_nll(intact_out.logits, input_ids)
            for cond in pending:
                ablated_out, feature_trace = run_forward(pm, clt, input_ids, cond, args.multiplier)
                ablated_nll = shifted_nll(ablated_out.logits, input_ids)
                delta = ablated_nll - intact_nll
                row = {
                    "model": model_name,
                    "condition": cond["condition"],
                    "condition_kind": cond["condition_kind"],
                    "control_for": cond["control_for"],
                    "triplet_id": cond["triplet_id"],
                    "layer": cond["layer"],
                    "feature": cond["feature"],
                    "sequence_id": rec["id"],
                    "source": rec["source"],
                    "starts_m": rec["starts_m"],
                    "seq_len": len(seq),
                    "n_tokens": int(input_ids.shape[-1]),
                }
                row.update(feature_activity_metrics(feature_trace, spans))
                row.update(summarize_delta_nll(delta, spans))
                row.update(attention_metrics(intact_out.attentions, ablated_out.attentions, spans, int(cond["layer"])))
                write_row(metrics_path, row)
                done.add((model_name, cond["condition"], rec["id"]))
            if (i + 1) % args.report_every == 0 or i == len(records) - 1:
                print(f"{model_name}: {i+1}/{len(records)} sequences ({time.time()-t0:.1f}s)", flush=True)
        del pm
        del clt
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    summarize(args)


def numeric(row: dict, key: str) -> float:
    try:
        x = float(row.get(key, "nan"))
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def bootstrap_ci(vals: list[float], seed: int, n_boot: int = 1000) -> tuple[float, float]:
    vals = np.asarray([v for v in vals if math.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        means.append(float(rng.choice(vals, size=vals.size, replace=True).mean()))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(args) -> None:
    metrics_path = args.out_dir / "per_sequence_metrics.tsv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    with metrics_path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["condition"], row["condition_kind"], row.get("control_for", ""))].append(row)

    summary_rows = []
    for (model, condition, kind, control_for), items in sorted(groups.items()):
        rec = {
            "model": model,
            "condition": condition,
            "condition_kind": kind,
            "control_for": control_for,
            "n_sequences": len(items),
        }
        for key in [
            "delta_nll_first2", "delta_nll_pos2_10", "delta_nll_pos3_10",
            "delta_nll_pos11_plus", "specificity_pos2_10_minus_11plus",
            "attention_first2_delta", "attention_cosine_distance",
            "feature_first2_active",
        ]:
            vals = [numeric(r, key) for r in items]
            vals = [v for v in vals if math.isfinite(v)]
            rec[f"mean_{key}"] = float(np.mean(vals)) if vals else float("nan")
            if key.startswith("delta_nll") or key.startswith("specificity"):
                lo, hi = bootstrap_ci(vals, args.seed + len(summary_rows))
                rec[f"{key}_ci_low"] = lo
                rec[f"{key}_ci_high"] = hi
        summary_rows.append(rec)

    out_tsv = args.out_dir / "condition_summary.tsv"
    fields = list(summary_rows[0].keys()) if summary_rows else ["model", "condition"]
    with out_tsv.open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    target_rows = [r for r in summary_rows if r["condition_kind"] == "target"]
    random_rows = [r for r in summary_rows if r["condition_kind"] == "random"]
    random_by_model = defaultdict(list)
    for r in random_rows:
        random_by_model[r["model"]].append(r)

    gate_hits = []
    for r in target_rows:
        rand_specs = [
            x["mean_specificity_pos2_10_minus_11plus"]
            for x in random_by_model.get(r["model"], [])
            if math.isfinite(x["mean_specificity_pos2_10_minus_11plus"])
        ]
        rand_mean = float(np.mean(rand_specs)) if rand_specs else 0.0
        spec = r["mean_specificity_pos2_10_minus_11plus"]
        attn_cos = r["mean_attention_cosine_distance"]
        hit = (
            math.isfinite(spec)
            and spec > rand_mean + args.gate_specificity_margin
            and math.isfinite(attn_cos)
            and attn_cos > args.gate_attention_cosine
        )
        gate_hits.append({**r, "random_specificity_mean": rand_mean, "gate_hit": hit})

    hits_by_triplet = defaultdict(int)
    for r in gate_hits:
        if r["gate_hit"]:
            hits_by_triplet[r["condition"]] += 1
    n_triplets_hit = sum(1 for tid in TARGET_TRIPLETS if hits_by_triplet.get(tid, 0) >= 2)
    if n_triplets_hit >= 2:
        outcome = "PASS"
    elif any(r["gate_hit"] for r in gate_hits):
        outcome = "PARTIAL"
    else:
        outcome = "FAIL"

    payload = {
        "task": "R2 attention-sink causal ablation",
        "outcome": outcome,
        "n_per_sequence_rows": len(rows),
        "target_gate_hits": gate_hits,
        "gate": {
            "pass_rule": "at least two target triplets have hits in at least two models",
            "specificity_margin_over_random": args.gate_specificity_margin,
            "attention_cosine_min": args.gate_attention_cosine,
        },
        "interpretation": (
            "PASS supports a causal role for the N-terminal attention-sink features. "
            "PARTIAL supports a narrower model- or triplet-specific effect. FAIL means "
            "the current evidence should remain correlation-only."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Attention-Sink Causal Ablation",
        "",
        f"- Per-sequence rows: {len(rows)}",
        f"- Outcome: {outcome}",
        "",
        "## Target Conditions",
        "",
        "| Model | Triplet | n | dNLL pos2-10 | dNLL 11+ | specificity | attention cosine | gate |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in gate_hits:
        lines.append(
            f"| {r['model']} | {r['condition']} | {r['n_sequences']} | "
            f"{r['mean_delta_nll_pos2_10']:.4f} | {r['mean_delta_nll_pos11_plus']:.4f} | "
            f"{r['mean_specificity_pos2_10_minus_11plus']:.4f} | "
            f"{r['mean_attention_cosine_distance']:.4f} | {r['gate_hit']} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- The intervention is a TopK-aware CLT same-layer MLP-output patch, not a direct attention-head patch.",
        "- `first2_fraction=1.00` in the upstream characterization refers to saved top-firing rows, not all cohort proteins.",
        "- Matched random controls are same-layer random CLT features; T025 is retained as a non-N-terminal specificity control.",
        "",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(lines))
    print(f"Wrote {args.out_dir / 'summary.md'}")
    print(f"Outcome: {outcome}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "summarize"], default="run")
    ap.add_argument("--triplets", type=Path, default=R2 / "results/circuit_analysis/universal_atlas_balanced200_wide_triplets_20260512.tsv")
    ap.add_argument("--cohort", type=Path, default=R2 / "results/circuit_analysis/triplet_characterization_20260515_nperm2000/cohort.json")
    ap.add_argument("--source", default="swissprot_n1")
    ap.add_argument("--max-sequences", type=int, default=200)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--n-random-controls", type=int, default=2)
    ap.add_argument("--multiplier", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=20260517)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report-every", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--gate-specificity-margin", type=float, default=0.02)
    ap.add_argument("--gate-attention-cosine", type=float, default=0.005)
    ap.add_argument("--out-dir", type=Path, default=R2 / "results/circuit_analysis/attention_sink_causal_ablation_20260517")
    ap.add_argument("--model-spec", action="append", default=[
        "protgpt2=/oss-pvc/zhk_zip/outputs/research2/clt_weights/protgpt2_v2/step_200000",
        "zymctrl=/oss-pvc/zhk_zip/outputs/research2/clt_weights/zymctrl_v2/step_200000",
        "progen2-medium=/gpfs/jiaotongdamoxing/zhk_zip/biocc/paper_r2_nature_mi/results/final_checkpoints/r2_clt_progen2_medium_rerun_20260403/clt_weights/progen2-medium/step_100000",
    ])
    args = ap.parse_args()
    if args.mode == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
