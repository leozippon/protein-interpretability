#!/usr/bin/env python
"""Score the staged AlphaMissense table on the local R1 cohorts.

The staged AlphaMissense amino-acid substitution file is keyed by
``uniprot_id`` and ``protein_variant``. The local perturbation-signature pickle
uses tuple keys ``(uniprot_id, position, ref, alt)``, so no genome-coordinate
lift-over is needed for the current ClinVar and cancer-gene holdout cohorts.

Outputs:
  - r1_encoder_interpretability_benchmark/results/variant_effect/alphamissense_baseline_20260504.json
  - r1_encoder_interpretability_benchmark/results/variant_effect/alphamissense_baseline_20260504.md
  - r1_encoder_interpretability_benchmark/results/variant_effect/alphamissense_matched_scores_20260504.tsv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "r1_encoder_interpretability_benchmark" / "results" / "variant_effect"
DEFAULT_AM = REPO / "external_resources" / "baselines" / "alphamissense" / "AlphaMissense_aa_substitutions.tsv.gz"


def clinvar_binary_label(clin_sig: str) -> int | None:
    sig = (clin_sig or "").lower()
    is_path = "pathogenic" in sig and "benign" not in sig and "conflicting" not in sig
    is_benign = "benign" in sig and "pathogenic" not in sig
    if is_path:
        return 1
    if is_benign:
        return 0
    return None


def auc_ci(y: Iterable[int], score: Iterable[float], n_boot: int, seed: int = 0) -> dict:
    y = np.asarray(list(y), dtype=np.int32)
    score = np.asarray(list(score), dtype=np.float64)
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
    out = {"n": int(len(y)), "n_pathogenic": int(y.sum()), "n_benign": int((1 - y).sum())}
    if len(np.unique(y)) < 2:
        out.update({"auc": math.nan, "ci95": [math.nan, math.nan]})
        return out
    out["auc"] = float(roc_auc_score(y, score))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(float(roc_auc_score(y[idx], score[idx])))
    out["ci95"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [math.nan, math.nan]
    return out


def load_clinvar_rows(signatures_path: Path) -> tuple[list[dict], dict[tuple[str, str], tuple[str, str]]]:
    with signatures_path.open("rb") as f:
        signatures = pickle.load(f)
    rows = []
    by_gene_variant = {}
    for key, var_sigs in signatures.items():
        uniprot_id, pos, ref, alt = key
        s0 = var_sigs[0]
        label = clinvar_binary_label(s0.get("clinical_significance", ""))
        if label is None:
            continue
        protein_variant = s0.get("variant_str") or f"{ref}{pos}{alt}"
        gene = str(s0.get("gene", "")).upper()
        row = {
            "cohort": "ClinVar2000",
            "gene": gene,
            "variant": protein_variant,
            "uniprot_id": str(uniprot_id),
            "protein_variant": protein_variant,
            "label": int(label),
            "clinical_significance": s0.get("clinical_significance", ""),
        }
        rows.append(row)
        by_gene_variant[(gene, protein_variant)] = (str(uniprot_id), protein_variant)
    return rows, by_gene_variant


def load_cancer_rows(cancer_path: Path, lookup: dict[tuple[str, str], tuple[str, str]]) -> list[dict]:
    with cancer_path.open() as f:
        data = json.load(f)
    rows = []
    for r in data.get("predictions", []):
        key = (str(r["gene"]).upper(), str(r["variant"]))
        if key not in lookup:
            continue
        uniprot_id, protein_variant = lookup[key]
        rows.append({
            "cohort": "CancerHoldout101",
            "gene": key[0],
            "variant": key[1],
            "uniprot_id": uniprot_id,
            "protein_variant": protein_variant,
            "label": int(r["label"]),
            "role": r.get("role", ""),
            "score_sae": float(r.get("score_sae", math.nan)),
            "score_llr": float(r.get("score_llr", math.nan)),
            "score_sae_plus_llr": float(r.get("score_sae_plus_llr", math.nan)),
        })
    return rows


def load_needed_alphamissense(am_path: Path, targets: set[tuple[str, str]]) -> dict[tuple[str, str], tuple[float, str]]:
    found = {}
    with gzip.open(am_path, "rt") as f:
        reader = csv.reader((line for line in f if line and not line.startswith("#")), delimiter="\t")
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        for row in reader:
            key = (row[idx["uniprot_id"]], row[idx["protein_variant"]])
            if key not in targets:
                continue
            found[key] = (float(row[idx["am_pathogenicity"]]), row[idx["am_class"]])
            if len(found) == len(targets):
                break
    return found


def cohort_eval(rows: list[dict], am: dict[tuple[str, str], tuple[float, str]], n_boot: int) -> dict:
    matched = []
    for r in rows:
        key = (r["uniprot_id"], r["protein_variant"])
        if key not in am:
            continue
        score, cls = am[key]
        rr = dict(r)
        rr["am_pathogenicity"] = score
        rr["am_class"] = cls
        matched.append(rr)
    result = auc_ci([r["label"] for r in matched], [r["am_pathogenicity"] for r in matched], n_boot=n_boot)
    result["n_total"] = len(rows)
    result["n_matched"] = len(matched)
    result["match_rate"] = len(matched) / max(len(rows), 1)
    result["class_counts"] = {c: sum(1 for r in matched if r["am_class"] == c) for c in sorted({r["am_class"] for r in matched})}
    return result, matched


def markdown(summary: dict) -> str:
    lines = ["# AlphaMissense Baseline (2026-05-04)\n"]
    lines.append("Scores come from the staged `AlphaMissense_aa_substitutions.tsv.gz` table and are matched by `(uniprot_id, protein_variant)`.\n")
    lines.append("| Cohort | Matched | Match Rate | AUC | 95% CI |")
    lines.append("|---|---:|---:|---:|---|")
    for cohort, vals in summary["cohorts"].items():
        ci = vals["ci95"]
        lines.append(f"| {cohort} | {vals['n_matched']}/{vals['n_total']} | {vals['match_rate']:.3f} | {vals['auc']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
    lines.append("\n## Comparison To Existing Local Scores\n")
    lines.append("| Cohort | Method | AUC |")
    lines.append("|---|---|---:|")
    for row in summary["comparison_rows"]:
        lines.append(f"| {row['cohort']} | {row['method']} | {row['auc']:.4f} |")
    lines.append("\n## Decision\n")
    lines.append(summary["decision"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphamissense", type=Path, default=DEFAULT_AM)
    ap.add_argument("--signatures", type=Path, default=OUT_DIR / "scaled_perturbation_signatures.pkl")
    ap.add_argument("--cancer", type=Path, default=OUT_DIR / "cancer_holdout.json")
    ap.add_argument("--readiness", type=Path, default=OUT_DIR / "baseline_headtohead_readiness_20260504.json")
    ap.add_argument("--out-json", type=Path, default=OUT_DIR / "alphamissense_baseline_20260504.json")
    ap.add_argument("--out-md", type=Path, default=OUT_DIR / "alphamissense_baseline_20260504.md")
    ap.add_argument("--out-tsv", type=Path, default=OUT_DIR / "alphamissense_matched_scores_20260504.tsv")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    args = ap.parse_args()

    clinvar_rows, lookup = load_clinvar_rows(args.signatures)
    cancer_rows = load_cancer_rows(args.cancer, lookup)
    all_rows = clinvar_rows + cancer_rows
    targets = {(r["uniprot_id"], r["protein_variant"]) for r in all_rows}
    am = load_needed_alphamissense(args.alphamissense, targets)

    cohorts = {}
    matched_all = []
    for name, rows in [("ClinVar2000", clinvar_rows), ("CancerHoldout101", cancer_rows)]:
        stats, matched = cohort_eval(rows, am, args.n_bootstrap)
        cohorts[name] = stats
        matched_all.extend(matched)

    comparison_rows = []
    if args.readiness.exists():
        with args.readiness.open() as f:
            ready = json.load(f)
        for cohort, methods in ready.get("pathogenicity", {}).items():
            for method, vals in methods.items():
                if isinstance(vals, dict) and "auc" in vals:
                    comparison_rows.append({"cohort": cohort, "method": method, "auc": float(vals["auc"])})
    for cohort, vals in cohorts.items():
        comparison_rows.append({"cohort": cohort, "method": "AlphaMissense", "auc": float(vals["auc"])})
    comparison_rows.sort(key=lambda r: (r["cohort"], r["method"]))

    missing = ["PrimateAI-3D", "gMVP", "ESM-1v"]
    decision = (
        "AlphaMissense is now staged and scored for the local R1 pathogenicity cohorts. "
        "T1-A remains incomplete because " + ", ".join(missing) +
        " are still missing and all competitors still need a unified reviewer-facing calibration table."
    )
    summary = {
        "task": "T1-A AlphaMissense baseline",
        "status": "alphamissense_scored_other_competitors_missing",
        "alphamissense_file": str(args.alphamissense.relative_to(REPO) if args.alphamissense.is_absolute() and args.alphamissense.is_relative_to(REPO) else args.alphamissense),
        "n_targets": len(targets),
        "n_targets_found": len(am),
        "cohorts": cohorts,
        "comparison_rows": comparison_rows,
        "remaining_missing_competitors": missing,
        "decision": decision,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    args.out_md.write_text(markdown(summary))
    with args.out_tsv.open("w", newline="") as f:
        fields = ["cohort", "gene", "variant", "uniprot_id", "protein_variant", "label", "am_pathogenicity", "am_class"]
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for r in matched_all:
            writer.writerow(r)
    print(f"Saved {args.out_json}")
    print(f"Saved {args.out_md}")
    print(f"Saved {args.out_tsv}")
    for cohort, vals in cohorts.items():
        print(f"{cohort}: matched {vals['n_matched']}/{vals['n_total']} AUC={vals['auc']:.4f}")


if __name__ == "__main__":
    main()
