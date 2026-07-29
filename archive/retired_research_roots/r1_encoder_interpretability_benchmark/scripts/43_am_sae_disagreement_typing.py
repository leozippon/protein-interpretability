#!/usr/bin/env python3
"""Systematically type AlphaMissense-vs-SAE disagreement cases.

This is R1-Add-2 from OPUS_NEXT_20260516.md. It turns the residual case-study
idea into a statistical audit: among ClinVar variants where AlphaMissense and
SAE-LR make opposite binary calls, ask whether the SAE-correct or AM-correct
side is enriched for residue-context classes from Swiss-Prot annotations.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import fisher_exact


REPO = Path(__file__).resolve().parents[2]
R1 = REPO / "r1_encoder_interpretability_benchmark"
OUT = R1 / "results" / "variant_effect"
DEFAULT_PRED = OUT / "alphamissense_sae_ensemble_predictions_20260511.tsv"
DEFAULT_CLINVAR = REPO / "data/clinvar/variant_summary.txt.gz"
DEFAULT_IDMAP = REPO / "data/swissprot/HUMAN_9606_idmapping.dat.gz"
DEFAULT_SWISSPROT_CACHE = REPO / "data/processed/swissprot_all_max1022.pkl"
DEFAULT_OUT_DIR = OUT / "am_sae_disagreement_typing_20260516"


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


def parse_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


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


def parse_variant(variant: str) -> tuple[str, int, str] | None:
    m = re.fullmatch(r"([A-Z])(\d+)([A-Z*])", variant or "")
    if not m:
        return None
    ref, pos, alt = m.groups()
    return ref, int(pos), alt


def protein_hgvs_one_to_three(variant: str) -> str | None:
    parsed = parse_variant(variant)
    if not parsed:
        return None
    ref, pos, alt = parsed
    if ref not in AA3 or alt not in AA3:
        return None
    return f"p.{AA3[ref]}{pos}{AA3[alt]}"


def load_predictions(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_gene_to_uniprot(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    with gzip.open(path, "rt", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            acc, kind, value = parts
            if kind in {"Gene_Name", "Gene_Synonym", "Gene_ORFName"} and "-" not in acc:
                key = value.upper()
                if acc not in out[key]:
                    out[key].append(acc)
    return out


def load_swissprot_annotations(path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(R1))
    with path.open("rb") as f:
        anns = pickle.load(f)
    return {ann.accession: ann for ann in anns}


def load_clinvar_review(path: Path, targets: set[tuple[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    wanted = {(gene, variant): protein_hgvs_one_to_three(variant) for gene, variant in targets}
    by_gene: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for (gene, variant), hgvs in wanted.items():
        if hgvs:
            by_gene[gene].append((variant, hgvs))

    out: dict[tuple[str, str], dict[str, Any]] = {}
    with gzip.open(path, "rt", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene = row.get("GeneSymbol", "").upper()
            if gene not in by_gene:
                continue
            name = row.get("Name", "")
            for variant, hgvs in by_gene[gene]:
                if hgvs not in name:
                    continue
                stars = review_stars(row.get("ReviewStatus", ""))
                key = (gene, variant)
                rec = {
                    "review_status": row.get("ReviewStatus", ""),
                    "review_stars": stars,
                    "clinical_significance_summary": row.get("ClinicalSignificance", ""),
                    "variation_id": row.get("VariationID", ""),
                }
                prev = out.get(key)
                if prev is None or stars > int(prev.get("review_stars", 0)):
                    out[key] = rec
    return out


def classify_context(ann: Any | None, pos: int | None, window: int) -> tuple[set[str], str]:
    if ann is None or pos is None:
        return {"no_swissprot_mapping"}, ""
    labels: set[str] = set()
    hits = []
    seq_len = len(getattr(ann, "sequence", "") or "")
    if pos <= 5:
        labels.add("N_terminal_5")
    if seq_len and pos >= seq_len - 4:
        labels.add("C_terminal_5")
    for start, end, feat_type, desc, category in getattr(ann, "features", []):
        text = f"{feat_type} {desc} {category}".lower()
        exact = start <= pos <= end
        near = start - window <= pos <= end + window
        if exact:
            hits.append(f"{category}:{feat_type}:{start}-{end}:{desc}".strip(":"))
        if exact and category == "domain":
            labels.add("domain")
        if exact and category == "region":
            labels.add("region")
        if exact and category == "topology":
            labels.add("topology")
        if exact and category == "secondary_structure":
            labels.add("secondary_structure")
        if near and category == "ptm":
            labels.add("PTM_adjacent")
        if exact and category == "functional":
            if any(k in text for k in ["active", "catalytic", "binding", "metal", "site", "motif", "dna", "nucleotide"]):
                labels.add("functional_site")
        if exact and any(k in text for k in ["binding", "interface", "dimer", "oligomer", "complex"]):
            labels.add("binding_or_interface")
        if exact and any(k in text for k in ["active", "catalytic"]):
            labels.add("active_or_catalytic")
        if exact and any(k in text for k in ["signal peptide", "transmembrane", "topological domain"]):
            labels.add("topology")
    if not labels:
        labels.add("no_local_annotation")
    return labels, " | ".join(hits[:5])


def bh(rows: list[dict[str, Any]]) -> None:
    indexed = sorted(enumerate(rows), key=lambda item: float(item[1]["p_value"]))
    m = len(indexed)
    qvals = [1.0] * m
    prev = 1.0
    for rank_from_end, (idx, row) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        q = min(prev, float(row["p_value"]) * m / rank)
        qvals[idx] = q
        prev = q
    for row, q in zip(rows, qvals):
        row["q_value"] = q


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, default=DEFAULT_PRED)
    ap.add_argument("--clinvar-summary", type=Path, default=DEFAULT_CLINVAR)
    ap.add_argument("--idmapping", type=Path, default=DEFAULT_IDMAP)
    ap.add_argument("--swissprot-cache", type=Path, default=DEFAULT_SWISSPROT_CACHE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--am-benign-threshold", type=float, default=0.34)
    ap.add_argument("--am-pathogenic-threshold", type=float, default=0.564)
    ap.add_argument("--sae-threshold", type=float, default=0.5)
    ap.add_argument("--min-review-stars", type=int, default=2)
    ap.add_argument("--ptm-window", type=int, default=3)
    args = ap.parse_args()

    pred_rows = load_predictions(args.predictions)
    targets = {(r["gene"].upper(), r["variant"]) for r in pred_rows}
    review = load_clinvar_review(args.clinvar_summary, targets)
    gene_to_uniprot = load_gene_to_uniprot(args.idmapping)
    ann_by_acc = load_swissprot_annotations(args.swissprot_cache)

    typed = []
    for row in pred_rows:
        gene = row["gene"].upper()
        variant = row["variant"]
        label = int(row["label"])
        am = parse_float(row.get("am_pathogenicity"))
        sae = parse_float(row.get("sae_lr_groupcv"))
        if math.isnan(am) or math.isnan(sae):
            continue
        if am <= args.am_benign_threshold:
            am_pred = 0
        elif am >= args.am_pathogenic_threshold:
            am_pred = 1
        else:
            continue
        sae_pred = 1 if sae >= args.sae_threshold else 0
        if am_pred == sae_pred:
            continue
        parsed = parse_variant(variant)
        pos = parsed[1] if parsed else None
        ann = None
        selected_acc = ""
        for acc in gene_to_uniprot.get(gene, []):
            cand = ann_by_acc.get(acc)
            if cand is not None and pos is not None and pos <= len(cand.sequence):
                ann = cand
                selected_acc = acc
                break
        if ann is None:
            for acc in gene_to_uniprot.get(gene, []):
                cand = ann_by_acc.get(acc)
                if cand is not None:
                    ann = cand
                    selected_acc = acc
                    break
        labels, local_hits = classify_context(ann, pos, args.ptm_window)
        rv = review.get((gene, variant), {})
        outcome = "SAE_right_AM_wrong" if sae_pred == label and am_pred != label else "AM_right_SAE_wrong"
        typed.append(
            {
                "gene": gene,
                "variant": variant,
                "label": label,
                "am_pathogenicity": am,
                "sae_lr_groupcv": sae,
                "am_pred": am_pred,
                "sae_pred": sae_pred,
                "outcome": outcome,
                "review_stars": int(rv.get("review_stars", 0) or 0),
                "review_status": rv.get("review_status", ""),
                "variation_id": rv.get("variation_id", ""),
                "uniprot_id": selected_acc,
                "position": pos if pos is not None else "",
                "context_labels": ";".join(sorted(labels)),
                "local_swissprot_hits": local_hits,
            }
        )

    stat_rows = [r for r in typed if int(r["review_stars"]) >= args.min_review_stars]
    all_labels = sorted({lab for r in stat_rows for lab in str(r["context_labels"]).split(";") if lab})
    tests = []
    for outcome in ["SAE_right_AM_wrong", "AM_right_SAE_wrong"]:
        outcome_set = [r for r in stat_rows if r["outcome"] == outcome]
        other_set = [r for r in stat_rows if r["outcome"] != outcome]
        for lab in all_labels:
            a = sum(lab in r["context_labels"].split(";") for r in outcome_set)
            b = len(outcome_set) - a
            c = sum(lab in r["context_labels"].split(";") for r in other_set)
            d = len(other_set) - c
            if a + c == 0:
                continue
            odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            tests.append(
                {
                    "outcome": outcome,
                    "context_label": lab,
                    "outcome_with_label": a,
                    "outcome_total": len(outcome_set),
                    "outcome_fraction": a / len(outcome_set) if outcome_set else 0.0,
                    "other_with_label": c,
                    "other_total": len(other_set),
                    "other_fraction": c / len(other_set) if other_set else 0.0,
                    "odds_ratio": odds if math.isfinite(odds) else "inf",
                    "p_value": p,
                }
            )
    if tests:
        bh(tests)
    for row in tests:
        row["significant_q05"] = bool(float(row["q_value"]) < 0.05)
        row["significant_q01"] = bool(float(row["q_value"]) < 0.01)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(
        args.out_dir / "typed_disagreements.tsv",
        typed,
        [
            "gene",
            "variant",
            "label",
            "am_pathogenicity",
            "sae_lr_groupcv",
            "am_pred",
            "sae_pred",
            "outcome",
            "review_stars",
            "review_status",
            "variation_id",
            "uniprot_id",
            "position",
            "context_labels",
            "local_swissprot_hits",
        ],
    )
    write_tsv(
        args.out_dir / "context_enrichment.tsv",
        tests,
        [
            "outcome",
            "context_label",
            "outcome_with_label",
            "outcome_total",
            "outcome_fraction",
            "other_with_label",
            "other_total",
            "other_fraction",
            "odds_ratio",
            "p_value",
            "q_value",
            "significant_q05",
            "significant_q01",
        ],
    )

    outcome_counts = Counter(r["outcome"] for r in typed)
    outcome_counts_review = Counter(r["outcome"] for r in stat_rows)
    sig01 = [r for r in tests if r.get("significant_q01")]
    sig05 = [r for r in tests if r.get("significant_q05")]
    pass_sae = any(r["outcome"] == "SAE_right_AM_wrong" and r.get("significant_q01") for r in tests)
    pass_am = any(r["outcome"] == "AM_right_SAE_wrong" and r.get("significant_q01") for r in tests)
    acceptance = pass_sae and pass_am

    summary = {
        "task": "R1-Add-2 AM-vs-SAE disagreement typing",
        "status": "completed",
        "n_prediction_rows": len(pred_rows),
        "n_disagreements": len(typed),
        "n_disagreements_review_ge_min": len(stat_rows),
        "min_review_stars": args.min_review_stars,
        "outcome_counts_all": dict(outcome_counts),
        "outcome_counts_review_filtered": dict(outcome_counts_review),
        "n_context_tests": len(tests),
        "n_significant_q05": len(sig05),
        "n_significant_q01": len(sig01),
        "acceptance_gate": "At least one BH q<0.01 context for SAE-right/AM-wrong and at least one for AM-right/SAE-wrong.",
        "acceptance_pass": acceptance,
        "outputs": {
            "typed_disagreements": str(args.out_dir / "typed_disagreements.tsv"),
            "context_enrichment": str(args.out_dir / "context_enrichment.tsv"),
            "summary_md": str(args.out_dir / "summary.md"),
        },
    }
    with (args.out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    tests_sorted = sorted(tests, key=lambda r: float(r["q_value"]))
    md = [
        "# AM-vs-SAE Disagreement Typing",
        "",
        "This analysis tests whether AlphaMissense and SAE-LR errors occupy different residue-context classes.",
        "",
        f"- Prediction rows: {len(pred_rows)}",
        f"- AM/SAE opposite-direction disagreements: {len(typed)}",
        f"- Review-filtered disagreements (stars >= {args.min_review_stars}): {len(stat_rows)}",
        f"- Outcome counts after review filter: {dict(outcome_counts_review)}",
        f"- Context enrichment tests: {len(tests)}",
        f"- Significant at BH q < 0.05: {len(sig05)}",
        f"- Significant at BH q < 0.01: {len(sig01)}",
        f"- Acceptance gate: {'PASS' if acceptance else 'FAIL'}",
        "",
        "## Top Context Tests",
        "",
        "| Outcome | context | outcome frac | other frac | odds ratio | q |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in tests_sorted[:12]:
        odds = r["odds_ratio"]
        odds_s = f"{float(odds):.3g}" if odds != "inf" else "inf"
        md.append(
            f"| {r['outcome']} | {r['context_label']} | {float(r['outcome_fraction']):.3f} | "
            f"{float(r['other_fraction']):.3f} | {odds_s} | {float(r['q_value']):.3g} |"
        )
    md += [
        "",
        "## Interpretation",
        "",
        "- PASS means the R1 manuscript can state that SAE-LR and AlphaMissense have statistically typed, bidirectional blind spots.",
        "- FAIL means the 30 residual case studies should remain illustrative triage examples rather than a systematic context-class claim.",
        "- Context labels are derived from staged Swiss-Prot residue annotations and simple terminal-position proxies; they are not manual structural validation.",
        "",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(md))
    print(f"Wrote {args.out_dir / 'typed_disagreements.tsv'}")
    print(f"Wrote {args.out_dir / 'context_enrichment.tsv'}")
    print(f"Wrote {args.out_dir / 'summary.md'}")
    print(f"Acceptance: {'PASS' if acceptance else 'FAIL'}")


if __name__ == "__main__":
    main()
