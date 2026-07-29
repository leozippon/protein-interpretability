#!/usr/bin/env python
"""Combine generated-sequence Pfam, CLEAN, and Foldseek metrics for R2 T2-C."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from scipy.stats import fisher_exact
except Exception:  # pragma: no cover - scipy is available in the main env.
    fisher_exact = None


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "r2_interpretability_transfer" / "results" / "ec_metrics"


def fmt(x: float) -> str:
    return "nan" if x is None or not math.isfinite(float(x)) else f"{float(x):.3f}"


def fishers(success_a: int, total_a: int, success_b: int, total_b: int) -> dict:
    table = [[success_a, total_a - success_a], [success_b, total_b - success_b]]
    if fisher_exact is None:
        return {"table": table, "alternative": "greater", "odds_ratio": math.nan, "p": math.nan}
    odds, p = fisher_exact(table, alternative="greater")
    return {"table": table, "alternative": "greater", "odds_ratio": float(odds), "p": float(p)}


def metric_rows(pfam: dict, clean: dict, foldseek: dict) -> list[dict]:
    rows = []
    sources = ["steered_leads", "steered_all", "unsteered"]
    for src in sources:
        p = pfam.get("by_source", {}).get(src, {})
        c = clean.get("by_source", {}).get(src, {})
        row = {
            "source": src,
            "n_sequences": int(max(p.get("n", 0), c.get("n", 0))),
            "pfam_lysozyme_like_hit_rate": p.get("lysozyme_like_hit_rate", math.nan),
            "pfam_any_hit_rate": p.get("hit_rate", math.nan),
            "clean_exact_3_2_1_17_rate": c.get("exact_target_rate", math.nan),
            "clean_3_2_1_prefix_rate": c.get("glycosidase_prefix_rate", math.nan),
        }
        if src == "steered_leads":
            f = foldseek.get("sets", {}).get("steered_leads", {}).get("summary", {})
            row.update({
                "n_foldseek_pdbs": foldseek.get("sets", {}).get("steered_leads", {}).get("n_query_pdbs", 0),
                "foldseek_mean_top_tm": f.get("mean_top_alntmscore", math.nan),
                "foldseek_frac_top_tm_ge_0_5": f.get("frac_top_tm_ge_0_5", math.nan),
                "foldseek_frac_top_tm_ge_0_7": f.get("frac_top_tm_ge_0_7", math.nan),
            })
        elif src == "unsteered":
            f = foldseek.get("sets", {}).get("unsteered_baseline", {}).get("summary", {})
            row.update({
                "n_foldseek_pdbs": foldseek.get("sets", {}).get("unsteered_baseline", {}).get("n_query_pdbs", 0),
                "foldseek_mean_top_tm": f.get("mean_top_alntmscore", math.nan),
                "foldseek_frac_top_tm_ge_0_5": f.get("frac_top_tm_ge_0_5", math.nan),
                "foldseek_frac_top_tm_ge_0_7": f.get("frac_top_tm_ge_0_7", math.nan),
            })
        else:
            row.update({
                "n_foldseek_pdbs": 0,
                "foldseek_mean_top_tm": math.nan,
                "foldseek_frac_top_tm_ge_0_5": math.nan,
                "foldseek_frac_top_tm_ge_0_7": math.nan,
            })
        rows.append(row)
    return rows


def infer_tests(pfam: dict, clean: dict) -> dict:
    p_all = pfam["by_source"]["steered_all"]
    p_un = pfam["by_source"]["unsteered"]
    c_all = clean["by_source"]["steered_all"]
    c_un = clean["by_source"]["unsteered"]
    return {
        "pfam_lysozyme_like_steered_all_gt_unsteered": fishers(
            int(p_all["n_with_lysozyme_like_hit"]), int(p_all["n"]),
            int(p_un["n_with_lysozyme_like_hit"]), int(p_un["n"]),
        ),
        "clean_exact_steered_all_gt_unsteered": fishers(
            round(float(c_all["exact_target_rate"]) * int(c_all["n"])), int(c_all["n"]),
            round(float(c_un["exact_target_rate"]) * int(c_un["n"])), int(c_un["n"]),
        ),
        "clean_prefix_steered_all_gt_unsteered": fishers(
            round(float(c_all["glycosidase_prefix_rate"]) * int(c_all["n"])), int(c_all["n"]),
            round(float(c_un["glycosidase_prefix_rate"]) * int(c_un["n"])), int(c_un["n"]),
        ),
    }


def markdown(summary: dict) -> str:
    lines = ["# R2 Generated Metric Triad Summary (2026-05-07)\n"]
    lines.append("| Source | n seq | Pfam lysozyme-like | CLEAN exact 3.2.1.17 | CLEAN 3.2.1.x | n PDB | Foldseek mean TM | TM >= 0.5 | TM >= 0.7 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary["rows"]:
        lines.append(
            f"| {row['source']} | {row['n_sequences']} | {fmt(row['pfam_lysozyme_like_hit_rate'])} | "
            f"{fmt(row['clean_exact_3_2_1_17_rate'])} | {fmt(row['clean_3_2_1_prefix_rate'])} | "
            f"{row['n_foldseek_pdbs']} | {fmt(row['foldseek_mean_top_tm'])} | "
            f"{fmt(row['foldseek_frac_top_tm_ge_0_5'])} | {fmt(row['foldseek_frac_top_tm_ge_0_7'])} |"
        )
    lines.append("\n## Statistical Checks\n")
    lines.append("| Comparison | Alternative | Odds ratio | p |")
    lines.append("|---|---|---:|---:|")
    for name, vals in summary["tests"].items():
        lines.append(f"| {name} | {vals['alternative']} | {fmt(vals['odds_ratio'])} | {fmt(vals['p'])} |")
    lines.append("\n## Interpretation\n")
    lines.append(summary["interpretation"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pfam", type=Path, default=OUT_DIR / "pfam_generated_lysozyme_20260504.json")
    ap.add_argument("--clean", type=Path, default=OUT_DIR / "clean_generated_lysozyme_20260507.json")
    ap.add_argument("--foldseek", type=Path, default=OUT_DIR / "foldseek_generated_lysozyme_20260507.json")
    ap.add_argument("--out-json", type=Path, default=OUT_DIR / "generated_metric_triad_summary_20260507.json")
    args = ap.parse_args()

    pfam = json.loads(args.pfam.read_text())
    clean = json.loads(args.clean.read_text())
    foldseek = json.loads(args.foldseek.read_text())
    rows = metric_rows(pfam, clean, foldseek)
    tests = infer_tests(pfam, clean)
    summary = {
        "task": "R2 T2-C generated metric triad summary",
        "status": "completed",
        "inputs": {"pfam": str(args.pfam), "clean": str(args.clean), "foldseek": str(args.foldseek)},
        "rows": rows,
        "tests": tests,
        "interpretation": (
            "The selected steered leads pass all three external checks strongly. "
            "Across all generated sequences, steering is not clearly better than the unsteered baseline on CLEAN, "
            "and the small Pfam lift is not statistically decisive; this supports using the triad as a filter/calibration "
            "layer rather than claiming a robust generation-wide steering improvement."
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    args.out_json.with_suffix(".md").write_text(markdown(summary))
    print(f"Saved {args.out_json}")
    print(f"Saved {args.out_json.with_suffix('.md')}")


if __name__ == "__main__":
    main()
