#!/usr/bin/env python
"""Summarize real lysozyme vs random UniRef50 calibration for R2 EC metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from scipy.stats import fisher_exact, mannwhitneyu
except Exception:  # pragma: no cover
    fisher_exact = None
    mannwhitneyu = None


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "r2_interpretability_transfer" / "results" / "ec_metrics"


def fmt(x: float) -> str:
    return "nan" if x is None or not math.isfinite(float(x)) else f"{float(x):.3f}"


def binary_d(p_real: float, p_random: float) -> float:
    sd = math.sqrt((p_real * (1 - p_real) + p_random * (1 - p_random)) / 2)
    if sd == 0:
        return math.inf if p_real != p_random else 0.0
    return (p_real - p_random) / sd


def cohen_d(real: list[float], random: list[float]) -> float:
    x = np.asarray(real, dtype=np.float64)
    y = np.asarray(random, dtype=np.float64)
    if len(x) < 2 or len(y) < 2:
        return math.nan
    var = ((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1)) / (len(x) + len(y) - 2)
    if var <= 0:
        return math.inf if float(np.mean(x)) != float(np.mean(y)) else 0.0
    return float((np.mean(x) - np.mean(y)) / math.sqrt(var))


def fisher(success_real: int, total_real: int, success_random: int, total_random: int) -> dict:
    table = [[success_real, total_real - success_real], [success_random, total_random - success_random]]
    if fisher_exact is None:
        return {"table": table, "alternative": "greater", "odds_ratio": math.nan, "p": math.nan}
    odds, p = fisher_exact(table, alternative="greater")
    return {"table": table, "alternative": "greater", "odds_ratio": float(odds), "p": float(p)}


def u_test(real: list[float], random: list[float]) -> dict:
    if mannwhitneyu is None or not real or not random:
        return {"alternative": "greater", "u": math.nan, "p": math.nan}
    stat = mannwhitneyu(real, random, alternative="greater")
    return {"alternative": "greater", "u": float(stat.statistic), "p": float(stat.pvalue)}


def foldseek_scores(foldseek: dict, source: str) -> list[float]:
    set_data = foldseek.get("sets", {}).get(source, {})
    n = int(set_data.get("n_query_pdbs", 0))
    scores = [
        float(r["alntmscore"])
        for r in set_data.get("summary", {}).get("top_scores", [])
        if math.isfinite(float(r.get("alntmscore", math.nan)))
    ]
    if len(scores) < n:
        scores.extend([0.0] * (n - len(scores)))
    return scores


def esmfold_scores(path: Path, key: str) -> list[float]:
    data = json.loads(path.read_text())
    vals = []
    for row in data.get("per_sequence", []):
        if row.get(key) is not None:
            vals.append(float(row[key]))
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pfam", type=Path, default=OUT_DIR / "pfam_calibration_lysozyme_20260507.json")
    ap.add_argument("--clean", type=Path, default=OUT_DIR / "clean_calibration_lysozyme_20260507.json")
    ap.add_argument("--real-esmfold", type=Path, default=OUT_DIR / "calibration_real_lysozyme_esmfold_20260507.json")
    ap.add_argument("--random-esmfold", type=Path, default=OUT_DIR / "calibration_random_uniref50_esmfold_20260507.json")
    ap.add_argument("--foldseek", type=Path, default=OUT_DIR / "foldseek_calibration_lysozyme_20260507.json")
    ap.add_argument("--out-json", type=Path, default=OUT_DIR / "ec_metric_calibration_summary_20260507.json")
    args = ap.parse_args()

    pfam = json.loads(args.pfam.read_text())
    clean = json.loads(args.clean.read_text())
    foldseek = json.loads(args.foldseek.read_text())
    p_real = pfam["by_source"]["real_lysozyme"]
    p_rand = pfam["by_source"]["random_uniref50"]
    c_real = clean["by_source"]["real_lysozyme"]
    c_rand = clean["by_source"]["random_uniref50"]

    real_plddt = esmfold_scores(args.real_esmfold, "mean_plddt")
    rand_plddt = esmfold_scores(args.random_esmfold, "mean_plddt")
    real_conf = esmfold_scores(args.real_esmfold, "frac_confident")
    rand_conf = esmfold_scores(args.random_esmfold, "frac_confident")
    real_tm = foldseek_scores(foldseek, "real_lysozyme")
    rand_tm = foldseek_scores(foldseek, "random_uniref50")

    clean_exact_real = round(float(c_real["exact_target_rate"]) * int(c_real["n"]))
    clean_exact_rand = round(float(c_rand["exact_target_rate"]) * int(c_rand["n"]))
    clean_prefix_real = round(float(c_real["glycosidase_prefix_rate"]) * int(c_real["n"]))
    clean_prefix_rand = round(float(c_rand["glycosidase_prefix_rate"]) * int(c_rand["n"]))

    rows = [
        {
            "metric": "Pfam lysozyme-like hit",
            "real_n": int(p_real["n"]),
            "random_n": int(p_rand["n"]),
            "real_mean": float(p_real["lysozyme_like_hit_rate"]),
            "random_mean": float(p_rand["lysozyme_like_hit_rate"]),
            "effect_size_d": binary_d(float(p_real["lysozyme_like_hit_rate"]), float(p_rand["lysozyme_like_hit_rate"])),
            "test": fisher(int(p_real["n_with_lysozyme_like_hit"]), int(p_real["n"]), int(p_rand["n_with_lysozyme_like_hit"]), int(p_rand["n"])),
        },
        {
            "metric": "CLEAN exact 3.2.1.17",
            "real_n": int(c_real["n"]),
            "random_n": int(c_rand["n"]),
            "real_mean": float(c_real["exact_target_rate"]),
            "random_mean": float(c_rand["exact_target_rate"]),
            "effect_size_d": binary_d(float(c_real["exact_target_rate"]), float(c_rand["exact_target_rate"])),
            "test": fisher(clean_exact_real, int(c_real["n"]), clean_exact_rand, int(c_rand["n"])),
        },
        {
            "metric": "CLEAN 3.2.1.x prefix",
            "real_n": int(c_real["n"]),
            "random_n": int(c_rand["n"]),
            "real_mean": float(c_real["glycosidase_prefix_rate"]),
            "random_mean": float(c_rand["glycosidase_prefix_rate"]),
            "effect_size_d": binary_d(float(c_real["glycosidase_prefix_rate"]), float(c_rand["glycosidase_prefix_rate"])),
            "test": fisher(clean_prefix_real, int(c_real["n"]), clean_prefix_rand, int(c_rand["n"])),
        },
        {
            "metric": "ESMFold mean pLDDT",
            "real_n": len(real_plddt),
            "random_n": len(rand_plddt),
            "real_mean": float(np.mean(real_plddt)) if real_plddt else math.nan,
            "random_mean": float(np.mean(rand_plddt)) if rand_plddt else math.nan,
            "effect_size_d": cohen_d(real_plddt, rand_plddt),
            "test": u_test(real_plddt, rand_plddt),
        },
        {
            "metric": "ESMFold confident fraction",
            "real_n": len(real_conf),
            "random_n": len(rand_conf),
            "real_mean": float(np.mean(real_conf)) if real_conf else math.nan,
            "random_mean": float(np.mean(rand_conf)) if rand_conf else math.nan,
            "effect_size_d": cohen_d(real_conf, rand_conf),
            "test": u_test(real_conf, rand_conf),
        },
        {
            "metric": "Foldseek top TM",
            "real_n": len(real_tm),
            "random_n": len(rand_tm),
            "real_mean": float(np.mean(real_tm)) if real_tm else math.nan,
            "random_mean": float(np.mean(rand_tm)) if rand_tm else math.nan,
            "effect_size_d": cohen_d(real_tm, rand_tm),
            "test": u_test(real_tm, rand_tm),
        },
    ]
    summary = {
        "task": "R2 T2-C EC metric real-vs-random calibration summary",
        "status": "completed",
        "inputs": {
            "pfam": str(args.pfam),
            "clean": str(args.clean),
            "real_esmfold": str(args.real_esmfold),
            "random_esmfold": str(args.random_esmfold),
            "foldseek": str(args.foldseek),
        },
        "rows": rows,
        "interpretation": (
            "This calibration tests whether external EC/structure metrics distinguish real lysozymes from length-matched random UniRef50 sequences. "
            "It is a metric-validity control, not a steering-success estimate."
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# R2 EC Metric Real-vs-Random Calibration (2026-05-07)\n"]
    lines.append("| Metric | Real n | Random n | Real mean | Random mean | Effect size d | p |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['metric']} | {row['real_n']} | {row['random_n']} | "
            f"{fmt(row['real_mean'])} | {fmt(row['random_mean'])} | "
            f"{fmt(row['effect_size_d'])} | {fmt(row['test'].get('p', math.nan))} |"
        )
    lines.append("\n" + summary["interpretation"] + "\n")
    args.out_json.with_suffix(".md").write_text("\n".join(lines))
    print(f"Saved {args.out_json}")
    print(f"Saved {args.out_json.with_suffix('.md')}")


if __name__ == "__main__":
    main()
