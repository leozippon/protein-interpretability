#!/usr/bin/env python
"""Diagnostics for the R1-D ProteinGym SAE follow-up result.

The full follow-up run stores one record per ProteinGym assay. This script
does not require GPU access; it audits those assay-level records for the
failure modes that matter most before deciding whether to rerun per-mutant
diagnostics:

  * SAE direction: does negating the SAE score become consistently better?
  * Ensemble utility: how often does the z-sum ensemble beat LLR alone?
  * Assay-size effects: are large MAVE assays dominating the story?
  * Outliers: which assays most help or hurt the ensemble?
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Iterable

import numpy as np


def finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def mean_ci(values: Iterable[float], n_boot: int, seed: int) -> dict:
    arr = np.array([float(v) for v in values if finite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "ci95": [None, None]}
    out = {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
    }
    if arr.size == 1 or n_boot <= 0:
        out["ci95"] = [float(arr.mean()), float(arr.mean())]
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = arr[idx].mean(axis=1)
    out["ci95"] = [
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    ]
    return out


def assay_bin(n_mutants: int) -> str:
    if n_mutants < 500:
        return "<500"
    if n_mutants < 2_000:
        return "500-2k"
    if n_mutants < 10_000:
        return "2k-10k"
    if n_mutants < 100_000:
        return "10k-100k"
    return ">=100k"


def summarize_records(records: list[dict], n_boot: int, seed: int) -> dict:
    usable = [
        r for r in records
        if finite(r.get("spearman_llr"))
        and finite(r.get("spearman_sae"))
        and finite(r.get("spearman_ensemble"))
    ]
    for r in usable:
        r["_llr"] = float(r["spearman_llr"])
        r["_sae"] = float(r["spearman_sae"])
        r["_ensemble"] = float(r["spearman_ensemble"])
        r["_ensemble_signed"] = (
            float(r["spearman_ensemble_signed"])
            if finite(r.get("spearman_ensemble_signed")) else float("nan")
        )
        r["_ensemble_classifier"] = (
            float(r["spearman_ensemble_classifier"])
            if finite(r.get("spearman_ensemble_classifier")) else float("nan")
        )
        r["_n"] = int(r.get("n_mutants") or 0)
        r["_ensemble_minus_llr"] = r["_ensemble"] - r["_llr"]
        r["_sae_minus_llr"] = r["_sae"] - r["_llr"]
        r["_neg_sae"] = -r["_sae"]

    by_bin = defaultdict(list)
    for r in usable:
        by_bin[assay_bin(r["_n"])].append(r)

    def frac(predicate) -> float | None:
        if not usable:
            return None
        return float(sum(1 for r in usable if predicate(r)) / len(usable))

    def compact(r: dict) -> dict:
        return {
            "name": r.get("name"),
            "n_mutants": r.get("n_mutants"),
            "llr": r["_llr"],
            "sae": r["_sae"],
            "ensemble": r["_ensemble"],
            "ensemble_minus_llr": r["_ensemble_minus_llr"],
        }

    summary = {
        "n_records_total": len(records),
        "n_records_usable": len(usable),
        "rho": {
            "llr": mean_ci((r["_llr"] for r in usable), n_boot, seed),
            "sae": mean_ci((r["_sae"] for r in usable), n_boot, seed + 1),
            "negated_sae": mean_ci((r["_neg_sae"] for r in usable), n_boot, seed + 2),
            "ensemble": mean_ci((r["_ensemble"] for r in usable), n_boot, seed + 3),
            "ensemble_signed": mean_ci((r["_ensemble_signed"] for r in usable), n_boot, seed + 6),
            "ensemble_classifier": mean_ci((r["_ensemble_classifier"] for r in usable), n_boot, seed + 7),
        },
        "deltas": {
            "ensemble_minus_llr": mean_ci(
                (r["_ensemble_minus_llr"] for r in usable), n_boot, seed + 4
            ),
            "sae_minus_llr": mean_ci(
                (r["_sae_minus_llr"] for r in usable), n_boot, seed + 5
            ),
        },
        "win_rates": {
            "sae_gt_llr": frac(lambda r: r["_sae"] > r["_llr"]),
            "negated_sae_gt_llr": frac(lambda r: r["_neg_sae"] > r["_llr"]),
            "ensemble_gt_llr": frac(lambda r: r["_ensemble"] > r["_llr"]),
            "ensemble_signed_gt_llr": frac(
                lambda r: finite(r["_ensemble_signed"]) and r["_ensemble_signed"] > r["_llr"]
            ),
            "ensemble_classifier_gt_llr": frac(
                lambda r: finite(r["_ensemble_classifier"]) and r["_ensemble_classifier"] > r["_llr"]
            ),
            "ensemble_gt_sae": frac(lambda r: r["_ensemble"] > r["_sae"]),
        },
        "by_assay_size": {},
        "best_ensemble_vs_llr": [
            compact(r) for r in sorted(
                usable, key=lambda x: x["_ensemble_minus_llr"], reverse=True
            )[:10]
        ],
        "worst_ensemble_vs_llr": [
            compact(r) for r in sorted(usable, key=lambda x: x["_ensemble_minus_llr"])[:10]
        ],
        "most_negative_sae": [
            compact(r) for r in sorted(usable, key=lambda x: x["_sae"])[:10]
        ],
        "most_positive_sae": [
            compact(r) for r in sorted(usable, key=lambda x: x["_sae"], reverse=True)[:10]
        ],
    }

    for key in ["<500", "500-2k", "2k-10k", "10k-100k", ">=100k"]:
        rows = by_bin.get(key, [])
        summary["by_assay_size"][key] = {
            "n": len(rows),
            "llr": mean_ci((r["_llr"] for r in rows), n_boot, seed),
            "sae": mean_ci((r["_sae"] for r in rows), n_boot, seed + 1),
            "negated_sae": mean_ci((r["_neg_sae"] for r in rows), n_boot, seed + 2),
            "ensemble": mean_ci((r["_ensemble"] for r in rows), n_boot, seed + 3),
            "ensemble_signed": mean_ci((r["_ensemble_signed"] for r in rows), n_boot, seed + 6),
            "ensemble_classifier": mean_ci((r["_ensemble_classifier"] for r in rows), n_boot, seed + 7),
            "ensemble_minus_llr": mean_ci(
                (r["_ensemble_minus_llr"] for r in rows), n_boot, seed + 4
            ),
        }

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="r1_encoder_interpretability_benchmark/results/variant_effect/proteingym_benchmark_sae_latest.json",
    )
    ap.add_argument(
        "--out",
        default="r1_encoder_interpretability_benchmark/results/variant_effect/proteingym_benchmark_sae_diagnostics.json",
    )
    ap.add_argument("--bootstrap", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    records = data.get("per_assay", [])
    out = {
        "input": args.input,
        "source_summary": data.get("summary", {}),
        "diagnostics": summarize_records(records, args.bootstrap, args.seed),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    diag = out["diagnostics"]
    print(f"Saved: {args.out}")
    print(f"Usable assays: {diag['n_records_usable']}/{diag['n_records_total']}")
    print(f"Mean LLR rho: {diag['rho']['llr']['mean']:.4f}")
    print(f"Mean SAE rho: {diag['rho']['sae']['mean']:.4f}")
    print(f"Mean negated-SAE rho: {diag['rho']['negated_sae']['mean']:.4f}")
    print(f"Mean ensemble rho: {diag['rho']['ensemble']['mean']:.4f}")
    print(
        "Mean ensemble-minus-LLR: "
        f"{diag['deltas']['ensemble_minus_llr']['mean']:.4f} "
        f"CI={diag['deltas']['ensemble_minus_llr']['ci95']}"
    )
    print(f"Ensemble beats LLR: {diag['win_rates']['ensemble_gt_llr']:.3f}")


if __name__ == "__main__":
    main()
