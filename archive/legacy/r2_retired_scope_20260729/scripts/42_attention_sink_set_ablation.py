#!/usr/bin/env python3
"""Multi-head sink-set ablation for R2.

The single-head E-1 ablation failed. This final rescue test asks whether the
N-terminal sink is distributed over a set of high sink-mass heads. It ablates
the top N-terminal sink heads as a set and compares against same-layer random
head sets of equal size.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
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


R2 = Path(__file__).resolve().parents[1]
if str(R2) not in sys.path:
    sys.path.insert(0, str(R2))

BASE_PATH = Path(__file__).with_name("41_attention_head_sink_ablation.py")
spec = importlib.util.spec_from_file_location("head_ablation_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

from src.analysis.circuit_discovery import load_trained_clt
from src.models.model_loader import load_model


TARGET_TRIPLETS = ["T011", "T018", "T023"]


def parse_model_spec(spec_text: str) -> tuple[str, str]:
    return base.parse_model_spec(spec_text)


def group_selected_heads(head_rows: list[dict], top_k: int, min_sink_mass: float, min_corr: float) -> list[dict]:
    by_head: dict[tuple[int, int], dict] = {}
    for row in head_rows:
        if row["triplet_id"] not in TARGET_TRIPLETS:
            continue
        key = (int(row["head_layer"]), int(row["head"]))
        rec = by_head.setdefault(
            key,
            {
                "model": row["model"],
                "head_layer": int(row["head_layer"]),
                "head": int(row["head"]),
                "n_heads": int(row["n_heads"]),
                "mean_first2_mass": float(row["mean_first2_mass"]),
                "max_corr_with_target": float("-inf"),
                "paired_triplets": [],
            },
        )
        r = float(row.get("corr_with_feature_first2", float("nan")))
        if math.isfinite(r):
            rec["max_corr_with_target"] = max(rec["max_corr_with_target"], r)
        rec["paired_triplets"].append(row["triplet_id"])
    candidates = []
    for rec in by_head.values():
        corr = rec["max_corr_with_target"]
        if not math.isfinite(corr):
            corr = float("nan")
        rec["max_corr_with_target"] = corr
        rec["selection_score"] = rec["mean_first2_mass"] + (max(corr, 0.0) if math.isfinite(corr) else 0.0)
        if rec["mean_first2_mass"] >= min_sink_mass and (not math.isfinite(corr) or corr >= min_corr):
            candidates.append(rec)
    candidates.sort(key=lambda r: (r["selection_score"], r["mean_first2_mass"]), reverse=True)
    return candidates[:top_k]


def sample_same_layer_controls(selected: list[dict], all_heads: list[dict], n_controls: int, seed: int) -> list[list[dict]]:
    rng = random.Random(seed)
    selected_keys = {(int(h["head_layer"]), int(h["head"])) for h in selected}
    n_heads_by_layer = {}
    for h in all_heads:
        n_heads_by_layer[int(h["head_layer"])] = int(h["n_heads"])
    controls = []
    for _ in range(n_controls):
        used = set(selected_keys)
        ctrl = []
        for h in selected:
            layer = int(h["head_layer"])
            n_heads = n_heads_by_layer[layer]
            head = rng.randrange(n_heads)
            tries = 0
            while (layer, head) in used and tries < 2000:
                head = rng.randrange(n_heads)
                tries += 1
            used.add((layer, head))
            ctrl.append(
                {
                    "model": h["model"],
                    "head_layer": layer,
                    "head": head,
                    "n_heads": n_heads,
                    "mean_first2_mass": float("nan"),
                    "max_corr_with_target": float("nan"),
                    "selection_score": float("nan"),
                    "paired_triplets": [],
                }
            )
        controls.append(ctrl)
    return controls


def make_multi_head_hook(heads: list[int], n_heads: int, multiplier: float):
    heads = sorted(set(int(h) for h in heads))

    def hook(_module, inputs):
        x = inputs[0]
        if not torch.is_tensor(x) or x.shape[-1] % n_heads != 0:
            return inputs
        y = x.clone()
        head_dim = x.shape[-1] // n_heads
        for head in heads:
            start = int(head) * head_dim
            y[..., start : start + head_dim] = y[..., start : start + head_dim] * multiplier
        return (y,) + tuple(inputs[1:])

    return hook


def register_set_hooks(pm, condition: dict, multiplier: float):
    hooks = []
    if condition["condition_kind"] == "sham":
        return hooks
    by_layer = defaultdict(list)
    n_heads_by_layer = {}
    for h in condition["heads"]:
        by_layer[int(h["head_layer"])].append(int(h["head"]))
        n_heads_by_layer[int(h["head_layer"])] = int(h["n_heads"])
    for layer, heads in by_layer.items():
        module = base.attn_projection_module(pm, layer)
        hooks.append(module.register_forward_pre_hook(make_multi_head_hook(heads, n_heads_by_layer[layer], multiplier)))
    return hooks


def forward_with_set(pm, input_ids: torch.Tensor, condition: dict, multiplier: float):
    hooks = register_set_hooks(pm, condition, multiplier)
    try:
        with torch.no_grad():
            return pm.model(input_ids, output_attentions=True, use_cache=False)
    finally:
        for h in hooks:
            h.remove()


def triplet_first2_activations(pm, clt, input_ids: torch.Tensor, sequence: str, specs: dict, condition: dict, multiplier: float) -> dict[str, float]:
    hooks = register_set_hooks(pm, condition, multiplier)
    try:
        spans = base.token_residue_spans(pm.tokenizer, input_ids, sequence)
        mask = base.first2_token_mask(spans)
        cache = pm.get_activations(input_ids)
        feats = clt.encode([x.float() for x in cache.resid_pre])
        out = {}
        for tid, spec in specs.items():
            vals = feats[int(spec["layer"])][0, :, int(spec["feature"])].detach().float().cpu().numpy()
            n = min(len(vals), len(mask))
            out[tid] = float(np.nanmax(vals[:n][mask[:n]])) if n and mask[:n].any() else float("nan")
        return out
    finally:
        for h in hooks:
            h.remove()


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def append_row(path: Path, row: dict, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        clean = {}
        for k in fields:
            v = row.get(k, "")
            if isinstance(v, float):
                clean[k] = f"{v:.8g}" if math.isfinite(v) else "nan"
            elif isinstance(v, (list, dict)):
                clean[k] = json.dumps(v, sort_keys=True)
            else:
                clean[k] = v
        w.writerow(clean)


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
    means = [float(rng.choice(vals, size=vals.size, replace=True).mean()) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(out_dir: Path, seed: int) -> dict:
    path = out_dir / "per_sequence_metrics.tsv"
    with path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["condition"], r["condition_kind"])].append(r)
    summary_rows = []
    for (model, condition, kind), items in sorted(groups.items()):
        rec = {
            "model": model,
            "condition": condition,
            "condition_kind": kind,
            "n_sequences": len(items),
            "n_heads_ablated": int(items[0].get("n_heads_ablated", 0)),
            "head_set": items[0].get("head_set", ""),
        }
        for key in [
            "delta_nll_pos2_10",
            "delta_nll_pos11_plus",
            "specificity_pos2_10_minus_11plus",
            "attention_first2_delta",
            "attention_cosine_distance",
            "mean_feature_drop_fraction",
            "min_feature_drop_fraction",
        ]:
            vals = [numeric(r, key) for r in items]
            vals = [v for v in vals if math.isfinite(v)]
            rec[f"mean_{key}"] = float(np.mean(vals)) if vals else float("nan")
            lo, hi = bootstrap_ci(vals, seed + len(summary_rows))
            rec[f"{key}_ci_low"] = lo
            rec[f"{key}_ci_high"] = hi
        summary_rows.append(rec)
    write_tsv(out_dir / "condition_summary.tsv", summary_rows, list(summary_rows[0].keys()))

    random_by_model = defaultdict(list)
    for r in summary_rows:
        if r["condition_kind"] == "random_set":
            random_by_model[r["model"]].append(r)
    gate_rows = []
    for r in summary_rows:
        if r["condition_kind"] != "sink_set":
            continue
        rand = random_by_model.get(r["model"], [])
        rand_nll = float(np.mean([x["mean_delta_nll_pos2_10"] for x in rand])) if rand else float("nan")
        rand_drop = float(np.mean([x["mean_mean_feature_drop_fraction"] for x in rand])) if rand else float("nan")
        nll_margin = r["mean_delta_nll_pos2_10"] - rand_nll if math.isfinite(rand_nll) else float("nan")
        drop_margin = r["mean_mean_feature_drop_fraction"] - rand_drop if math.isfinite(rand_drop) else float("nan")
        exploratory = (
            r["mean_delta_nll_pos2_10"] >= 0.05
            and r["mean_mean_feature_drop_fraction"] >= 0.10
            and (not math.isfinite(nll_margin) or nll_margin >= 0.025)
        )
        strict = r["mean_delta_nll_pos2_10"] >= 0.5 and r["mean_mean_feature_drop_fraction"] >= 0.5
        gate_rows.append({**r, "random_mean_delta_nll_pos2_10": rand_nll, "random_mean_feature_drop": rand_drop, "nll_margin": nll_margin, "feature_drop_margin": drop_margin, "exploratory_gate_hit": exploratory, "strict_gate_hit": strict})
    strict_pass = sum(bool(r["strict_gate_hit"]) for r in gate_rows) >= 2
    exploratory_pass = sum(bool(r["exploratory_gate_hit"]) for r in gate_rows) >= 2
    out = {
        "task": "R2 multi-head N-terminal sink-set ablation",
        "status": "completed",
        "n_rows": len(rows),
        "strict_pass": strict_pass,
        "exploratory_pass": exploratory_pass,
        "gate_rows": gate_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2))

    md = [
        "# Multi-Head Sink-Set Ablation",
        "",
        f"- Per-sequence rows: {len(rows)}",
        f"- Strict gate: {'PASS' if strict_pass else 'FAIL'}",
        f"- Exploratory gate: {'PASS' if exploratory_pass else 'FAIL'}",
        "",
        "| Model | heads | dNLL pos2-10 | random dNLL | feature drop | random drop | attention dFirst2 | strict | exploratory |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in gate_rows:
        md.append(
            f"| {r['model']} | {r['n_heads_ablated']} | {r['mean_delta_nll_pos2_10']:.4f} | "
            f"{r['random_mean_delta_nll_pos2_10']:.4f} | {r['mean_mean_feature_drop_fraction']:.4f} | "
            f"{r['random_mean_feature_drop']:.4f} | {r['mean_attention_first2_delta']:.4f} | "
            f"{r['strict_gate_hit']} | {r['exploratory_gate_hit']} |"
        )
    md += [
        "",
        "Notes:",
        "- The strict gate preserves the original Nat-Methods-grade effect-size expectation.",
        "- The exploratory gate asks whether a distributed sink-set effect exists at all.",
        "- If both fail, the causal sink mechanism should be dropped.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(md))
    return out


def run(args) -> None:
    os.environ.setdefault("R2_MODEL_BASE_DIR", "/gpfs/jiaotongdamoxing/zhk_zip/models")
    records = base.read_cohort(args.cohort, args.source, args.max_sequences, args.max_length)
    triplets = base.read_triplets(args.triplets)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metric_fields = [
        "model", "condition", "condition_kind", "sequence_id", "source", "starts_m",
        "seq_len", "n_tokens", "n_heads_ablated", "head_set", "delta_nll_all",
        "delta_nll_first2", "delta_nll_pos2_10", "delta_nll_pos3_10",
        "delta_nll_pos11_plus", "specificity_pos2_10_minus_11plus",
        "attention_first2_delta", "attention_cosine_distance",
        "feature_drop_T011", "feature_drop_T018", "feature_drop_T023",
        "mean_feature_drop_fraction", "min_feature_drop_fraction",
    ]
    metrics_path = args.out_dir / "per_sequence_metrics.tsv"
    if args.force and metrics_path.exists():
        metrics_path.unlink()

    for model_name, ckpt in [parse_model_spec(s) for s in args.model_spec]:
        print(f"\n=== {model_name} ===", flush=True)
        pm = load_model(model_name, device=args.device)
        clt = load_trained_clt(ckpt, device=args.device)
        head_rows, _best = base.identify_heads(pm, clt, records, triplets, model_name, args)
        write_tsv(
            args.out_dir / f"{model_name}_head_discovery.tsv",
            head_rows,
            ["model", "triplet_id", "head_layer", "head", "n_heads", "mean_first2_mass", "corr_with_feature_first2", "selection_score"],
        )
        selected = group_selected_heads(head_rows, args.top_k_heads, args.min_sink_mass, args.min_target_corr)
        controls = sample_same_layer_controls(selected, head_rows, args.n_random_sets, args.seed)
        conditions = [
            {"condition": "SINK_SET", "condition_kind": "sink_set", "heads": selected},
            *[
                {"condition": f"RANDOM_SET_{i}", "condition_kind": "random_set", "heads": c}
                for i, c in enumerate(controls)
            ],
            {"condition": "SHAM", "condition_kind": "sham", "heads": []},
        ]
        (args.out_dir / f"{model_name}_conditions.json").write_text(json.dumps(conditions, indent=2) + "\n")
        specs = {
            t["triplet_id"]: t["features"][model_name]
            for t in triplets
            if t["triplet_id"] in TARGET_TRIPLETS and model_name in t["features"]
        }
        t0 = time.time()
        for i, rec in enumerate(records):
            seq = rec["sequence"]
            input_ids = pm.tokenize(seq)
            spans = base.token_residue_spans(pm.tokenizer, input_ids, seq)
            intact = forward_with_set(pm, input_ids, {"condition_kind": "sham", "heads": []}, 1.0)
            intact_nll = base.shifted_nll(intact.logits, input_ids)
            intact_feats = triplet_first2_activations(pm, clt, input_ids, seq, specs, {"condition_kind": "sham", "heads": []}, 1.0)
            for cond in conditions:
                ablated = forward_with_set(pm, input_ids, cond, args.multiplier)
                delta = base.shifted_nll(ablated.logits, input_ids) - intact_nll
                ablated_feats = triplet_first2_activations(pm, clt, input_ids, seq, specs, cond, args.multiplier)
                drops = {}
                for tid in TARGET_TRIPLETS:
                    v0 = intact_feats.get(tid, float("nan"))
                    v1 = ablated_feats.get(tid, float("nan"))
                    drops[tid] = (v0 - v1) / max(abs(v0), 1e-8) if math.isfinite(v0) and math.isfinite(v1) else float("nan")
                valid_drops = [v for v in drops.values() if math.isfinite(v)]
                row = {
                    "model": model_name,
                    "condition": cond["condition"],
                    "condition_kind": cond["condition_kind"],
                    "sequence_id": rec["id"],
                    "source": rec["source"],
                    "starts_m": rec["starts_m"],
                    "seq_len": len(seq),
                    "n_tokens": int(input_ids.shape[-1]),
                    "n_heads_ablated": len(cond["heads"]),
                    "head_set": ";".join(f"L{h['head_layer']}H{h['head']}" for h in cond["heads"]),
                    "feature_drop_T011": drops.get("T011", float("nan")),
                    "feature_drop_T018": drops.get("T018", float("nan")),
                    "feature_drop_T023": drops.get("T023", float("nan")),
                    "mean_feature_drop_fraction": float(np.mean(valid_drops)) if valid_drops else float("nan"),
                    "min_feature_drop_fraction": float(np.min(valid_drops)) if valid_drops else float("nan"),
                }
                row.update(base.summarize_delta_nll(delta, spans))
                row.update(base.attention_delta_metrics(intact.attentions, ablated.attentions, spans, 0))
                append_row(metrics_path, row, metric_fields)
            if (i + 1) % args.report_every == 0 or i == len(records) - 1:
                print(f"  set ablation {model_name}: {i+1}/{len(records)} ({time.time()-t0:.1f}s)", flush=True)
        del pm
        del clt
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    summarize(args.out_dir, args.seed)
    print(f"Wrote {args.out_dir / 'summary.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplets", type=Path, default=R2 / "results" / "circuit_analysis" / "universal_atlas_balanced200_wide_triplets_20260512.tsv")
    ap.add_argument("--cohort", type=Path, default=R2 / "results" / "circuit_analysis" / "triplet_characterization_20260515_nperm2000" / "cohort.json")
    ap.add_argument("--source", default="swissprot_n1")
    ap.add_argument("--max-sequences", type=int, default=200)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--head-discovery-sequences", type=int, default=100)
    ap.add_argument("--top-k-heads", type=int, default=8)
    ap.add_argument("--min-sink-mass", type=float, default=0.5)
    ap.add_argument("--min-target-corr", type=float, default=-1.0)
    ap.add_argument("--n-random-sets", type=int, default=3)
    ap.add_argument("--multiplier", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--report-every", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=R2 / "results" / "circuit_analysis" / "attention_sink_set_ablation_20260518")
    ap.add_argument(
        "--model-spec",
        action="append",
        default=[
            "protgpt2=/oss-pvc/zhk_zip/outputs/research2/clt_weights/protgpt2_v2/step_200000",
            "zymctrl=/oss-pvc/zhk_zip/outputs/research2/clt_weights/zymctrl_v2/step_200000",
            "progen2-medium=/gpfs/jiaotongdamoxing/zhk_zip/biocc/paper_r2_nature_mi/results/final_checkpoints/r2_clt_progen2_medium_rerun_20260403/clt_weights/progen2-medium/step_100000",
        ],
    )
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
