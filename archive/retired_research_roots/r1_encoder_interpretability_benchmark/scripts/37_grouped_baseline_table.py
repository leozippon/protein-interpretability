#!/usr/bin/env python3
"""Build a gene-grouped pathogenicity baseline table without third-party deps."""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "r1_encoder_interpretability_benchmark/results/variant_effect"
ENSEMBLE_TSV = OUT_DIR / "alphamissense_sae_ensemble_predictions_20260511.tsv"
EXTERNAL_TSV = OUT_DIR / "external_baselines_available_scores_20260507.tsv"
OUT_JSON = OUT_DIR / "grouped_pathogenicity_baselines_20260511.json"
OUT_MD = OUT_DIR / "grouped_pathogenicity_baselines_20260511.md"


def parse_float(value: str) -> float:
    value = (value or "").strip()
    if not value or value.lower() == "nan":
        return math.nan
    return float(value)


def auc(y: list[int], score: list[float]) -> float:
    pairs = [(s, yi) for yi, s in zip(y, score) if math.isfinite(s)]
    if not pairs:
        return math.nan
    n_pos = sum(yi for _, yi in pairs)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    pairs.sort(key=lambda x: x[0])
    rank_sum = 0.0
    rank = 1
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        rank_sum += avg_rank * sum(yi for _, yi in pairs[i:j])
        rank += j - i
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def group_bootstrap_ci(rows: list[dict], score_key: str, n_boot: int = 2000, seed: int = 42) -> list[float]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if math.isfinite(row.get(score_key, math.nan)):
            groups[row["gene"]].append(row)
    keys = sorted(groups)
    rng = random.Random(seed)
    vals: list[float] = []
    for _ in range(n_boot):
        sampled = []
        for _ in keys:
            sampled.extend(groups[rng.choice(keys)])
        y = [int(r["label"]) for r in sampled]
        s = [float(r[score_key]) for r in sampled]
        val = auc(y, s)
        if math.isfinite(val):
            vals.append(val)
    if not vals:
        return [math.nan, math.nan]
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return [lo, hi]


def load_rows() -> list[dict]:
    rows: dict[tuple[str, str], dict] = {}
    with ENSEMBLE_TSV.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = (row["gene"].upper(), row["variant"])
            rows[key] = {
                "gene": row["gene"].upper(),
                "variant": row["variant"],
                "label": int(row["label"]),
                "AlphaMissense": parse_float(row["am_pathogenicity"]),
                "SAE-LR group-CV": parse_float(row["sae_lr_groupcv"]),
                "AM+SAE stack": parse_float(row["am_sae_stack_groupcv"]),
                "AM+SAE z-sum": parse_float(row["am_sae_zsum_groupcv"]),
            }
    with EXTERNAL_TSV.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("cohort") != "ClinVar2000":
                continue
            key = (row["gene"].upper(), row["variant"])
            if key not in rows:
                continue
            rows[key]["gMVP"] = parse_float(row.get("gmvp_rankscore", ""))
            rows[key]["ESM-1v"] = parse_float(row.get("esm1v_ensemble_pathogenicity", ""))
    return list(rows.values())


def summarize(rows: list[dict], method: str, seed_offset: int) -> dict:
    sub = [r for r in rows if math.isfinite(r.get(method, math.nan))]
    y = [int(r["label"]) for r in sub]
    s = [float(r[method]) for r in sub]
    ci = group_bootstrap_ci(sub, method, seed=42 + seed_offset)
    return {
        "method": method,
        "n": len(sub),
        "n_genes": len({r["gene"] for r in sub}),
        "auc": auc(y, s),
        "ci95_gene_bootstrap": ci,
    }


def main() -> None:
    rows = load_rows()
    methods = [
        "AlphaMissense",
        "gMVP",
        "ESM-1v",
        "SAE-LR group-CV",
        "AM+SAE stack",
        "AM+SAE z-sum",
    ]
    summary = {
        "date": "2026-05-11",
        "cohort": "ClinVar2000",
        "group_unit": "gene_symbol",
        "note": "SAE and AM+SAE rows use gene-grouped predictions from F-T1-3; external scalar predictors are evaluated directly with gene-bootstrap CIs.",
        "methods": [summarize(rows, method, i) for i, method in enumerate(methods)],
    }

    import json

    OUT_JSON.write_text(json.dumps(summary, indent=2))

    lines = [
        "# Gene-Grouped Pathogenicity Baselines",
        "",
        "Date: 2026-05-11",
        "",
        "Cohort: ClinVar2000",
        "",
        "Group unit: gene symbol",
        "",
        "SAE and AM+SAE rows use gene-grouped predictions from F-T1-3. External scalar predictors are evaluated directly and use gene-bootstrap confidence intervals.",
        "",
        "| Method | n | genes | AUC | gene-bootstrap 95% CI |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summary["methods"]:
        ci = row["ci95_gene_bootstrap"]
        lines.append(
            f"| {row['method']} | {row['n']} | {row['n_genes']} | {row['auc']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- AlphaMissense and gMVP remain the strongest scalar pathogenicity predictors in this table.",
        "- SAE-LR group-CV is substantially weaker as a scalar predictor.",
        "- AM+SAE ensembles do not improve over AlphaMissense and should not be used as a headline claim.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
