#!/usr/bin/env python3
"""Audit IndelMissense v1.1 coordinate/resource compatibility."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
RECORDS = REPO / "data" / "indelmissense" / "v1.1_coordinates" / "records.tsv"
OUT_DIR = REPO / "r0_shared_interpretability_framework" / "results" / "v0_20260515" / "encoder"


def has_vcf(row: dict[str, str], build: str) -> bool:
    prefix = f"{build}_"
    return all(
        row.get(prefix + field, "")
        for field in [
            "chromosome",
            "position_vcf",
            "reference_allele_vcf",
            "alternate_allele_vcf",
        ]
    )


def summarize(records_path: Path) -> dict[str, Any]:
    variant_classes: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    grch37_len_pairs: Counter[tuple[int, int]] = Counter()
    grch38_len_pairs: Counter[tuple[int, int]] = Counter()
    grch37_vcf = 0
    grch38_vcf = 0
    grch37_snv = 0
    grch38_snv = 0
    grch38_mnv = 0

    with records_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    for row in rows:
        variant_classes[row.get("variant_class", "")] += 1
        labels[row.get("label", "")] += 1
        split_counts[row.get("split", "")] += 1

        if has_vcf(row, "grch37"):
            grch37_vcf += 1
            ref = row["grch37_reference_allele_vcf"]
            alt = row["grch37_alternate_allele_vcf"]
            grch37_len_pairs[(len(ref), len(alt))] += 1
            if len(ref) == len(alt) == 1:
                grch37_snv += 1

        if has_vcf(row, "grch38"):
            grch38_vcf += 1
            ref = row["grch38_reference_allele_vcf"]
            alt = row["grch38_alternate_allele_vcf"]
            grch38_len_pairs[(len(ref), len(alt))] += 1
            if len(ref) == len(alt) == 1:
                grch38_snv += 1
            elif len(ref) == len(alt):
                grch38_mnv += 1

    n = len(rows)
    return {
        "dataset": "IndelMissense_v1.1_coordinates",
        "records": str(records_path.relative_to(REPO)),
        "n_records": n,
        "variant_class_counts": dict(sorted(variant_classes.items())),
        "label_counts": dict(sorted(labels.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "grch37_vcf_records": grch37_vcf,
        "grch38_vcf_records": grch38_vcf,
        "grch37_vcf_coverage": grch37_vcf / n if n else 0.0,
        "grch38_vcf_coverage": grch38_vcf / n if n else 0.0,
        "grch37_exact_snv_compatible_records": grch37_snv,
        "grch38_exact_snv_compatible_records": grch38_snv,
        "grch38_mnv_compatible_records": grch38_mnv,
        "grch38_exact_snv_compatible_fraction_of_vcf": grch38_snv / grch38_vcf if grch38_vcf else 0.0,
        "top_grch37_ref_alt_length_pairs": [
            {"ref_len": ref_len, "alt_len": alt_len, "n": count}
            for (ref_len, alt_len), count in grch37_len_pairs.most_common(12)
        ],
        "top_grch38_ref_alt_length_pairs": [
            {"ref_len": ref_len, "alt_len": alt_len, "n": count}
            for (ref_len, alt_len), count in grch38_len_pairs.most_common(12)
        ],
        "resource_implication": (
            "Exact dbNSFP/REVEL matching is expected to have very low or zero "
            "coverage because this coordinate-augmented IndelMissense table is "
            "dominated by protein indels/delins rather than single-nucleotide "
            "missense substitutions. Treat dbNSFP/REVEL as a resource coverage "
            "gate for this dataset; use indel-aware CADD or protein-level "
            "baselines as primary comparators when available."
        ),
    }


def write_summary_md(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# IndelMissense v1.1 Resource Preflight",
        "",
        f"- Records: {payload['n_records']}",
        f"- GRCh37 VCF records: {payload['grch37_vcf_records']} ({payload['grch37_vcf_coverage']:.4f})",
        f"- GRCh38 VCF records: {payload['grch38_vcf_records']} ({payload['grch38_vcf_coverage']:.4f})",
        f"- GRCh38 exact SNV-compatible records: {payload['grch38_exact_snv_compatible_records']}",
        f"- GRCh38 exact SNV-compatible fraction among VCF records: {payload['grch38_exact_snv_compatible_fraction_of_vcf']:.4f}",
        "",
        "## Resource Implication",
        "",
        payload["resource_implication"],
        "",
        "## Top GRCh38 REF/ALT Length Pairs",
        "",
        "| ref_len | alt_len | n |",
        "|---:|---:|---:|",
    ]
    for row in payload["top_grch38_ref_alt_length_pairs"]:
        lines.append(f"| {row['ref_len']} | {row['alt_len']} | {row['n']} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    if not RECORDS.exists():
        raise FileNotFoundError(RECORDS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = summarize(RECORDS)
    (OUT_DIR / "indelmissense_v11_resource_preflight.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    write_summary_md(payload, OUT_DIR / "indelmissense_v11_resource_preflight.md")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
