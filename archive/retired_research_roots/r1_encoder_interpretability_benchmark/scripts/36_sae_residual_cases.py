#!/usr/bin/env python3
"""Build a lightweight SAE residual case-study index.

This script intentionally avoids external Python dependencies. It turns the
existing SAE-vs-external residual candidate table into a reviewer-facing index
with computational patterns and clear evidence boundaries. It is not a manual
biological validation step.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "r1_encoder_interpretability_benchmark/results/variant_effect/available_baseline_sae_residual_cases_20260507.tsv"
OUT_TSV = ROOT / "r1_encoder_interpretability_benchmark/results/variant_effect/sae_residual_case_index_20260511.tsv"
OUT_MD = ROOT / "r1_encoder_interpretability_benchmark/results/variant_effect/sae_residual_case_index_20260511.md"


def parse_float(value: str) -> float:
    value = (value or "").strip()
    if not value or value.lower() == "nan":
        return math.nan
    return float(value)


def label_name(value: str) -> str:
    return "pathogenic" if str(value).strip() == "1" else "benign"


def pattern_for(row: dict[str, str]) -> tuple[str, str, str]:
    label = int(row["label"])
    sae = parse_float(row["SAE-LR"])
    ext = parse_float(row["sae_lr_pred_from_external"])
    residual = parse_float(row["sae_lr_residual_vs_external"])
    am = parse_float(row.get("am_pathogenicity", "nan"))
    gmvp = parse_float(row.get("gmvp_rankscore", "nan"))

    if label == 1 and sae >= 0.8 and ext <= 0.4:
        pattern = "SAE pathogenic rescue candidate"
        interpretation = "SAE is high while the external-score proxy is low; prioritize for annotation review as a possible SAE-specific pathogenic signal."
        priority = "high"
    elif label == 0 and sae <= 0.2 and ext >= 0.6:
        pattern = "SAE benign rescue candidate"
        interpretation = "SAE is low while the external-score proxy is high; prioritize for annotation review as a possible SAE-specific benign signal."
        priority = "high"
    elif label == 0 and sae >= 0.8 and ext <= 0.4:
        pattern = "SAE pathogenic false-positive candidate"
        interpretation = "SAE is high for a benign label while external predictors are lower; useful for finding disruption features that are not clinically pathogenic."
        priority = "medium"
    elif label == 1 and sae <= 0.2 and ext >= 0.6:
        pattern = "SAE pathogenic false-negative candidate"
        interpretation = "SAE is low for a pathogenic label while external predictors are higher; useful for finding pathogenic mechanisms missed by the SAE perturbation head."
        priority = "medium"
    elif abs(residual) >= 0.5:
        pattern = "large SAE-external disagreement"
        interpretation = "SAE and external predictors disagree strongly; inspect top firing features and protein annotations before making a mechanistic claim."
        priority = "medium"
    else:
        pattern = "moderate SAE-external disagreement"
        interpretation = "Candidate retained from the residual list but needs lower priority manual triage."
        priority = "low"

    if not math.isnan(am) and not math.isnan(gmvp) and abs(am - gmvp) >= 0.5:
        interpretation += " AlphaMissense and gMVP also disagree, so this may be a generally unstable scalar-predictor case."

    return pattern, interpretation, priority


def main() -> None:
    with INPUT.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    output_rows: list[dict[str, str]] = []
    for i, row in enumerate(rows, start=1):
        pattern, interpretation, priority = pattern_for(row)
        out = {
            "rank": str(i),
            "cohort": row["cohort"],
            "gene": row["gene"],
            "variant": row["variant"],
            "label": label_name(row["label"]),
            "mechanism": row["mechanism"],
            "sae_lr": row["SAE-LR"],
            "external_proxy": row["sae_lr_pred_from_external"],
            "sae_external_residual": row["sae_lr_residual_vs_external"],
            "alphamissense": row.get("am_pathogenicity", ""),
            "gmvp": row.get("gmvp_rankscore", ""),
            "esm1v": row.get("esm1v_ensemble_pathogenicity", ""),
            "pattern": pattern,
            "manual_review_priority": priority,
            "interpretation_stub": interpretation,
        }
        output_rows.append(out)

    fieldnames = list(output_rows[0].keys()) if output_rows else []
    with OUT_TSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    pattern_counts = Counter(row["pattern"] for row in output_rows)
    priority_counts = Counter(row["manual_review_priority"] for row in output_rows)
    label_counts = Counter(row["label"] for row in output_rows)

    lines = [
        "# SAE Residual Case-Study Index",
        "",
        "Date: 2026-05-11",
        "",
        "Input: `r1_encoder_interpretability_benchmark/results/variant_effect/available_baseline_sae_residual_cases_20260507.tsv`",
        "",
        "This is a computational triage index, not manual biological validation. It supports the R1 residual-diagnostic framing by identifying variants where SAE-LR disagrees with the external-score proxy.",
        "",
        "## Summary",
        "",
        f"- Candidate cases: {len(output_rows)}",
        f"- Labels: {dict(label_counts)}",
        f"- Manual review priority: {dict(priority_counts)}",
        "",
        "Pattern counts:",
        "",
    ]
    for pattern, count in pattern_counts.most_common():
        lines.append(f"- {pattern}: {count}")

    lines += [
        "",
        "## Top 30 Cases",
        "",
        "| rank | gene | variant | label | pattern | SAE-LR | external proxy | residual | priority |",
        "|---:|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in output_rows[:30]:
        lines.append(
            "| {rank} | {gene} | {variant} | {label} | {pattern} | {sae_lr:.4f} | {external:.4f} | {residual:.4f} | {priority} |".format(
                rank=row["rank"],
                gene=row["gene"],
                variant=row["variant"],
                label=row["label"],
                pattern=row["pattern"],
                sae_lr=parse_float(row["sae_lr"]),
                external=parse_float(row["external_proxy"]),
                residual=parse_float(row["sae_external_residual"]),
                priority=row["manual_review_priority"],
            )
        )

    lines += [
        "",
        "## Interpretation Boundary",
        "",
        "- These cases are useful for figure planning and manual annotation review.",
        "- They do not prove mechanism by themselves because top firing features still require protein-specific residue review.",
        "- The manuscript should describe them as residual-diagnostic examples unless manually curated biological evidence is added.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))

    print(f"Wrote {OUT_TSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
