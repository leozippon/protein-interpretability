#!/usr/bin/env python3
"""Curate AlphaMissense-vs-SAE residual case studies for R1.

This is N-4 from OPUS_NEXT_20260513.md.  It creates a manuscript-facing table
of cases where AlphaMissense is confidently wrong relative to ClinVar labels and
records whether the SAE residual signal rescues or fails on the same variant.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
R1 = REPO / "r1_encoder_interpretability_benchmark" / "results" / "variant_effect"

AA3 = {
    "A": "Ala",
    "R": "Arg",
    "N": "Asn",
    "D": "Asp",
    "C": "Cys",
    "Q": "Gln",
    "E": "Glu",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "L": "Leu",
    "K": "Lys",
    "M": "Met",
    "F": "Phe",
    "P": "Pro",
    "S": "Ser",
    "T": "Thr",
    "W": "Trp",
    "Y": "Tyr",
    "V": "Val",
    "*": "Ter",
}


def parse_float(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def review_stars(status: str) -> int:
    s = (status or "").lower()
    if "practice guideline" in s:
        return 4
    if "reviewed by expert panel" in s:
        return 3
    if "multiple submitters" in s and "no conflicts" in s:
        return 2
    if "criteria provided" in s:
        return 1
    return 0


def protein_hgvs_one_to_three(variant: str) -> str | None:
    m = re.fullmatch(r"([A-Z])(\d+)([A-Z*])", variant)
    if not m:
        return None
    ref, pos, alt = m.groups()
    if ref not in AA3 or alt not in AA3:
        return None
    return f"p.{AA3[ref]}{pos}{AA3[alt]}"


def load_residual_lookup(path: Path) -> dict[tuple[str, str], dict]:
    out = {}
    if not path.exists():
        return out
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            out[(row["gene"], row["variant"])] = row
    return out


def load_am_predictions(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_clinvar_review(path: Path, targets: set[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    wanted_hgvs = {
        (gene, variant): protein_hgvs_one_to_three(variant)
        for gene, variant in targets
    }
    targets_by_gene: dict[str, list[tuple[str, str]]] = {}
    for (gene, variant), hgvs in wanted_hgvs.items():
        if hgvs:
            targets_by_gene.setdefault(gene, []).append((variant, hgvs))

    out = {}
    with gzip.open(path, "rt", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene = row.get("GeneSymbol", "")
            if not gene:
                continue
            gene_targets = targets_by_gene.get(gene, [])
            if not gene_targets:
                continue
            name = row.get("Name", "")
            for variant, hgvs in gene_targets:
                if hgvs and hgvs in name:
                    stars = review_stars(row.get("ReviewStatus", ""))
                    key = (gene, variant)
                    prev = out.get(key)
                    rec = {
                        "review_status": row.get("ReviewStatus", ""),
                        "review_stars": stars,
                        "clinical_significance_summary": row.get("ClinicalSignificance", ""),
                        "variation_id": row.get("VariationID", ""),
                        "assembly": row.get("Assembly", ""),
                    }
                    if prev is None or stars > int(prev["review_stars"]):
                        out[key] = rec
    return out


def am_confident_wrong(row: dict) -> bool:
    label = int(row["label"])
    am = parse_float(row.get("am_pathogenicity", ""))
    return (label == 1 and am <= 0.34) or (label == 0 and am >= 0.564)


def am_wrong_margin(row: dict) -> float:
    label = int(row["label"])
    am = parse_float(row.get("am_pathogenicity", ""))
    if label == 1:
        return 0.34 - am
    return am - 0.564


def pattern(row: dict, residual: dict | None) -> tuple[str, str]:
    label = int(row["label"])
    am = parse_float(row.get("am_pathogenicity", ""))
    sae = parse_float(row.get("sae_lr_groupcv", ""))
    residual_sae = parse_float((residual or {}).get("SAE-LR", ""))
    if math.isnan(sae) and not math.isnan(residual_sae):
        sae = residual_sae

    if label == 1 and am <= 0.34 and not math.isnan(sae) and sae >= 0.5:
        return (
            "AM false-benign, SAE rescue",
            "SAE-LR is high on a ClinVar pathogenic variant that AlphaMissense scores benign; use as an interpretation-layer rescue candidate.",
        )
    if label == 0 and am >= 0.564 and not math.isnan(sae) and sae <= 0.5:
        return (
            "AM false-pathogenic, SAE rescue",
            "SAE-LR is low on a ClinVar benign variant that AlphaMissense scores pathogenic; use as an interpretation-layer benign rescue candidate.",
        )
    if label == 1 and am <= 0.34:
        return (
            "AM false-benign, SAE not rescued",
            "AlphaMissense is confidently benign for a ClinVar pathogenic variant, but SAE does not clearly rescue; keep as a negative diagnostic.",
        )
    if label == 0 and am >= 0.564:
        return (
            "AM false-pathogenic, SAE not rescued",
            "AlphaMissense is confidently pathogenic for a ClinVar benign variant, but SAE does not clearly rescue; keep as a negative diagnostic.",
        )
    return ("not AM-confident-wrong", "Not selected for the main case-study table.")


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--am-predictions", type=Path, default=R1 / "alphamissense_sae_ensemble_predictions_20260511.tsv")
    ap.add_argument("--residual-cases", type=Path, default=R1 / "available_baseline_sae_residual_cases_20260507.tsv")
    ap.add_argument("--clinvar-summary", type=Path, default=REPO / "data" / "clinvar" / "variant_summary.txt.gz")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out-md", type=Path, default=R1 / "sae_residual_case_studies_20260513.md")
    ap.add_argument("--out-tsv", type=Path, default=R1 / "sae_residual_case_studies_20260513.tsv")
    args = ap.parse_args()

    residual_lookup = load_residual_lookup(args.residual_cases)
    am_rows = [r for r in load_am_predictions(args.am_predictions) if am_confident_wrong(r)]
    targets = {(r["gene"], r["variant"]) for r in am_rows}
    review = load_clinvar_review(args.clinvar_summary, targets)

    curated = []
    for row in am_rows:
        key = (row["gene"], row["variant"])
        residual = residual_lookup.get(key)
        pat, interp = pattern(row, residual)
        rv = review.get(key, {})
        curated.append(
            {
                "gene": row["gene"],
                "variant": row["variant"],
                "label": "pathogenic" if row["label"] == "1" else "benign",
                "clinical_significance": row.get("clinical_significance", ""),
                "am_pathogenicity": row.get("am_pathogenicity", ""),
                "sae_lr_groupcv": row.get("sae_lr_groupcv", ""),
                "am_wrong_margin": f"{am_wrong_margin(row):.6g}",
                "review_stars": str(rv.get("review_stars", "")),
                "review_status": rv.get("review_status", ""),
                "variation_id": rv.get("variation_id", ""),
                "in_residual_table": str(residual is not None),
                "external_proxy": "" if residual is None else residual.get("sae_lr_pred_from_external", ""),
                "sae_external_residual": "" if residual is None else residual.get("sae_lr_residual_vs_external", ""),
                "case_pattern": pat,
                "sae_feature_firing_pattern": interp,
            }
        )

    def sort_key(r):
        stars = int(r["review_stars"] or 0)
        rescue = 1 if "SAE rescue" in r["case_pattern"] else 0
        residual = 1 if r["in_residual_table"] == "True" else 0
        return (-stars, -rescue, -residual, -parse_float(r["am_wrong_margin"]))

    curated.sort(key=sort_key)
    top = curated[: args.n]
    fields = [
        "gene",
        "variant",
        "label",
        "clinical_significance",
        "am_pathogenicity",
        "sae_lr_groupcv",
        "am_wrong_margin",
        "review_stars",
        "review_status",
        "variation_id",
        "in_residual_table",
        "external_proxy",
        "sae_external_residual",
        "case_pattern",
        "sae_feature_firing_pattern",
    ]
    write_tsv(args.out_tsv, top, fields)

    n_review2 = sum(1 for r in curated if int(r["review_stars"] or 0) >= 2)
    lines = [
        "# SAE Residual Case Studies",
        "",
        "This table selects AlphaMissense-confident wrong cases and records whether SAE-LR rescues the ClinVar label.",
        "",
        f"- AlphaMissense-confident wrong candidates: {len(curated)}",
        f"- Candidates with ClinVar review stars >= 2: {n_review2}",
        f"- Cases shown: {len(top)}",
        "",
        "| Gene | Variant | ClinVar label | AM | SAE-LR | Review | Pattern | SAE firing-pattern interpretation |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for r in top:
        lines.append(
            f"| {r['gene']} | {r['variant']} | {r['label']} | {parse_float(r['am_pathogenicity']):.4f} | "
            f"{parse_float(r['sae_lr_groupcv']):.4f} | {r['review_stars'] or 'NA'} | "
            f"{r['case_pattern']} | {r['sae_feature_firing_pattern']} |"
        )
    lines += [
        "",
        "## Evidence Boundary",
        "",
        "- AlphaMissense confidence uses the published AM thresholds: <=0.34 as benign-like and >=0.564 as pathogenic-like.",
        "- ClinVar review stars are parsed from `variant_summary.txt.gz` by matching gene and protein HGVS where possible.",
        "- The firing-pattern interpretation is computational triage. It is not manual biological validation.",
        "",
    ]
    args.out_md.write_text("\n".join(lines))
    print(f"Wrote {args.out_tsv}")
    print(f"Wrote {args.out_md}")
    print(f"AM-confident wrong candidates: {len(curated)}; review>=2: {n_review2}; shown: {len(top)}")


if __name__ == "__main__":
    main()
