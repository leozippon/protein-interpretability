#!/usr/bin/env python3
"""Extend AlphaMissense-vs-SAE disagreement typing with protein-level contexts.

This is the CPU-only R1-Save-3 diagnostic from OPUS_R1_RESCUE_20260518.md.
It preserves the original residue-context table and adds staged protein-level
covariates that do not require new external approvals: protein length,
gnomAD constraint, existing mechanism labels, Pfam family-size proxy, ESM2
LLR confidence, and BLOSUM62 substitution conservativeness.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import fisher_exact


REPO = Path(__file__).resolve().parents[2]
R1 = REPO / "r1_encoder_interpretability_benchmark"
DEFAULT_TYPED = R1 / "results" / "variant_effect" / "am_sae_disagreement_typing_20260516" / "typed_disagreements.tsv"
DEFAULT_SWISSPROT = REPO / "data" / "processed" / "swissprot_all_max1022.pkl"
DEFAULT_GNOMAD = REPO / "data" / "gnomad" / "gnomad.v4.1.constraint_metrics.tsv"
DEFAULT_MECH = R1 / "results" / "variant_effect" / "gene_level_mechanism_20260512.json"
DEFAULT_PFAM = REPO / "data" / "interpro" / "pfam_residue.tsv"
DEFAULT_LLR = R1 / "results" / "variant_effect" / "esm2_per_variant_llr.json"
DEFAULT_OUT_DIR = R1 / "results" / "variant_effect" / "extended_disagreement_typing_20260518"


BLOSUM62 = {
    "A": {"A": 4, "R": -1, "N": -2, "D": -2, "C": 0, "Q": -1, "E": -1, "G": 0, "H": -2, "I": -1, "L": -1, "K": -1, "M": -1, "F": -2, "P": -1, "S": 1, "T": 0, "W": -3, "Y": -2, "V": 0},
    "R": {"A": -1, "R": 5, "N": 0, "D": -2, "C": -3, "Q": 1, "E": 0, "G": -2, "H": 0, "I": -3, "L": -2, "K": 2, "M": -1, "F": -3, "P": -2, "S": -1, "T": -1, "W": -3, "Y": -2, "V": -3},
    "N": {"A": -2, "R": 0, "N": 6, "D": 1, "C": -3, "Q": 0, "E": 0, "G": 0, "H": 1, "I": -3, "L": -3, "K": 0, "M": -2, "F": -3, "P": -2, "S": 1, "T": 0, "W": -4, "Y": -2, "V": -3},
    "D": {"A": -2, "R": -2, "N": 1, "D": 6, "C": -3, "Q": 0, "E": 2, "G": -1, "H": -1, "I": -3, "L": -4, "K": -1, "M": -3, "F": -3, "P": -1, "S": 0, "T": -1, "W": -4, "Y": -3, "V": -3},
    "C": {"A": 0, "R": -3, "N": -3, "D": -3, "C": 9, "Q": -3, "E": -4, "G": -3, "H": -3, "I": -1, "L": -1, "K": -3, "M": -1, "F": -2, "P": -3, "S": -1, "T": -1, "W": -2, "Y": -2, "V": -1},
    "Q": {"A": -1, "R": 1, "N": 0, "D": 0, "C": -3, "Q": 5, "E": 2, "G": -2, "H": 0, "I": -3, "L": -2, "K": 1, "M": 0, "F": -3, "P": -1, "S": 0, "T": -1, "W": -2, "Y": -1, "V": -2},
    "E": {"A": -1, "R": 0, "N": 0, "D": 2, "C": -4, "Q": 2, "E": 5, "G": -2, "H": 0, "I": -3, "L": -3, "K": 1, "M": -2, "F": -3, "P": -1, "S": 0, "T": -1, "W": -3, "Y": -2, "V": -2},
    "G": {"A": 0, "R": -2, "N": 0, "D": -1, "C": -3, "Q": -2, "E": -2, "G": 6, "H": -2, "I": -4, "L": -4, "K": -2, "M": -3, "F": -3, "P": -2, "S": 0, "T": -2, "W": -2, "Y": -3, "V": -3},
    "H": {"A": -2, "R": 0, "N": 1, "D": -1, "C": -3, "Q": 0, "E": 0, "G": -2, "H": 8, "I": -3, "L": -3, "K": -1, "M": -2, "F": -1, "P": -2, "S": -1, "T": -2, "W": -2, "Y": 2, "V": -3},
    "I": {"A": -1, "R": -3, "N": -3, "D": -3, "C": -1, "Q": -3, "E": -3, "G": -4, "H": -3, "I": 4, "L": 2, "K": -3, "M": 1, "F": 0, "P": -3, "S": -2, "T": -1, "W": -3, "Y": -1, "V": 3},
    "L": {"A": -1, "R": -2, "N": -3, "D": -4, "C": -1, "Q": -2, "E": -3, "G": -4, "H": -3, "I": 2, "L": 4, "K": -2, "M": 2, "F": 0, "P": -3, "S": -2, "T": -1, "W": -2, "Y": -1, "V": 1},
    "K": {"A": -1, "R": 2, "N": 0, "D": -1, "C": -3, "Q": 1, "E": 1, "G": -2, "H": -1, "I": -3, "L": -2, "K": 5, "M": -1, "F": -3, "P": -1, "S": 0, "T": -1, "W": -3, "Y": -2, "V": -2},
    "M": {"A": -1, "R": -1, "N": -2, "D": -3, "C": -1, "Q": 0, "E": -2, "G": -3, "H": -2, "I": 1, "L": 2, "K": -1, "M": 5, "F": 0, "P": -2, "S": -1, "T": -1, "W": -1, "Y": -1, "V": 1},
    "F": {"A": -2, "R": -3, "N": -3, "D": -3, "C": -2, "Q": -3, "E": -3, "G": -3, "H": -1, "I": 0, "L": 0, "K": -3, "M": 0, "F": 6, "P": -4, "S": -2, "T": -2, "W": 1, "Y": 3, "V": -1},
    "P": {"A": -1, "R": -2, "N": -2, "D": -1, "C": -3, "Q": -1, "E": -1, "G": -2, "H": -2, "I": -3, "L": -3, "K": -1, "M": -2, "F": -4, "P": 7, "S": -1, "T": -1, "W": -4, "Y": -3, "V": -2},
    "S": {"A": 1, "R": -1, "N": 1, "D": 0, "C": -1, "Q": 0, "E": 0, "G": 0, "H": -1, "I": -2, "L": -2, "K": 0, "M": -1, "F": -2, "P": -1, "S": 4, "T": 1, "W": -3, "Y": -2, "V": -2},
    "T": {"A": 0, "R": -1, "N": 0, "D": -1, "C": -1, "Q": -1, "E": -1, "G": -2, "H": -2, "I": -1, "L": -1, "K": -1, "M": -1, "F": -2, "P": -1, "S": 1, "T": 5, "W": -2, "Y": -2, "V": 0},
    "W": {"A": -3, "R": -3, "N": -4, "D": -4, "C": -2, "Q": -2, "E": -3, "G": -2, "H": -2, "I": -3, "L": -2, "K": -3, "M": -1, "F": 1, "P": -4, "S": -3, "T": -2, "W": 11, "Y": 2, "V": -3},
    "Y": {"A": -2, "R": -2, "N": -2, "D": -3, "C": -2, "Q": -1, "E": -2, "G": -3, "H": 2, "I": -1, "L": -1, "K": -2, "M": -1, "F": 3, "P": -3, "S": -2, "T": -2, "W": 2, "Y": 7, "V": -1},
    "V": {"A": 0, "R": -3, "N": -3, "D": -3, "C": -1, "Q": -2, "E": -2, "G": -3, "H": -3, "I": 3, "L": 1, "K": -2, "M": 1, "F": -1, "P": -2, "S": -2, "T": 0, "W": -3, "Y": -1, "V": 4},
}


def parse_float(x: Any) -> float:
    try:
        if x in {"", "NA", None}:
            return float("nan")
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def parse_variant(v: str) -> tuple[str, int, str] | None:
    if not v or len(v) < 3:
        return None
    ref = v[0]
    alt = v[-1]
    mid = v[1:-1]
    if not mid.isdigit():
        return None
    return ref, int(mid), alt


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_swissprot_lengths(path: Path) -> dict[str, int]:
    sys.path.insert(0, str(R1))
    with path.open("rb") as f:
        anns = pickle.load(f)
    return {ann.accession: len(getattr(ann, "sequence", "") or "") for ann in anns}


def load_gnomad(path: Path) -> dict[str, dict[str, float]]:
    best: dict[str, tuple[int, dict[str, float]]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene = (row.get("gene") or "").upper()
            if not gene:
                continue
            priority = int(str(row.get("mane_select", "")).lower() == "true") * 2 + int(
                str(row.get("canonical", "")).lower() == "true"
            )
            rec = {
                "pLI": parse_float(row.get("lof.pLI")),
                "loeuf_upper": parse_float(row.get("lof.oe_ci.upper")),
                "mis_z": parse_float(row.get("mis.z_score")),
            }
            prev = best.get(gene)
            if prev is None or priority > prev[0]:
                best[gene] = (priority, rec)
    return {gene: rec for gene, (_, rec) in best.items()}


def load_mechanism(path: Path) -> dict[str, str]:
    obj = json.loads(path.read_text())
    return {str(r.get("gene", "")).upper(): str(r.get("label", "")) for r in obj.get("gene_rows", [])}


def load_pfam(path: Path) -> tuple[dict[str, list[tuple[int, int, str]]], dict[str, int]]:
    intervals: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    members: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            acc = row["uniprot"]
            pfam = row["pfam_id"]
            try:
                start = int(row["start"])
                end = int(row["end"])
            except ValueError:
                continue
            intervals[acc].append((start, end, pfam))
            members[pfam].add(acc)
    sizes = {pfam: len(accs) for pfam, accs in members.items()}
    return intervals, sizes


def load_llr(path: Path) -> dict[tuple[str, str], float]:
    rows = json.loads(path.read_text())
    out = {}
    for row in rows:
        gene = str(row.get("gene", "")).upper()
        variant = str(row.get("variant", ""))
        out[(gene, variant)] = parse_float(row.get("llr"))
    return out


def quantiles(values: list[float]) -> tuple[float, float, float]:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return float("nan"), float("nan"), float("nan")

    def q(p: float) -> float:
        idx = min(len(clean) - 1, max(0, round((len(clean) - 1) * p)))
        return clean[idx]

    return q(0.25), q(0.50), q(0.75)


def qbucket(value: float, cuts: tuple[float, float, float], prefix: str) -> str:
    if not math.isfinite(value):
        return f"{prefix}_missing"
    q1, q2, q3 = cuts
    if value <= q1:
        return f"{prefix}_q1_low"
    if value <= q2:
        return f"{prefix}_q2"
    if value <= q3:
        return f"{prefix}_q3"
    return f"{prefix}_q4_high"


def pli_bucket(v: float) -> str:
    if not math.isfinite(v):
        return "pLI_missing"
    if v < 0.1:
        return "pLI_low_lt0.1"
    if v <= 0.9:
        return "pLI_mid_0.1_0.9"
    return "pLI_high_gt0.9"


def loeuf_bucket(v: float) -> str:
    if not math.isfinite(v):
        return "LOEUF_missing"
    if v <= 0.35:
        return "LOEUF_constrained_le0.35"
    if v <= 0.7:
        return "LOEUF_mid_0.35_0.7"
    return "LOEUF_unconstrained_gt0.7"


def blosum_bucket(variant: str) -> tuple[str, str]:
    parsed = parse_variant(variant)
    if not parsed:
        return "BLOSUM_missing", ""
    ref, _, alt = parsed
    if alt == "*":
        return "BLOSUM_stop", ""
    score = BLOSUM62.get(ref, {}).get(alt)
    if score is None:
        return "BLOSUM_missing", ""
    if score >= 1:
        bucket = "BLOSUM_conservative_ge1"
    elif score == 0:
        bucket = "BLOSUM_neutral_0"
    else:
        bucket = "BLOSUM_radical_lt0"
    return bucket, str(score)


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
    ap.add_argument("--typed", type=Path, default=DEFAULT_TYPED)
    ap.add_argument("--swissprot-cache", type=Path, default=DEFAULT_SWISSPROT)
    ap.add_argument("--gnomad", type=Path, default=DEFAULT_GNOMAD)
    ap.add_argument("--mechanism", type=Path, default=DEFAULT_MECH)
    ap.add_argument("--pfam", type=Path, default=DEFAULT_PFAM)
    ap.add_argument("--llr", type=Path, default=DEFAULT_LLR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--min-review-stars", type=int, default=2)
    args = ap.parse_args()

    typed = load_tsv(args.typed)
    lengths = load_swissprot_lengths(args.swissprot_cache)
    gnomad = load_gnomad(args.gnomad)
    mechanism = load_mechanism(args.mechanism)
    pfam_intervals, pfam_sizes = load_pfam(args.pfam)
    llr = load_llr(args.llr)

    length_cuts = quantiles([lengths.get(r.get("uniprot_id", ""), float("nan")) for r in typed])
    llr_abs_cuts = quantiles([abs(llr.get((r["gene"].upper(), r["variant"]), float("nan"))) for r in typed])
    pfam_size_cuts = quantiles(list(pfam_sizes.values()))

    enriched = []
    all_new_labels = set()
    for row in typed:
        gene = row["gene"].upper()
        variant = row["variant"]
        acc = row.get("uniprot_id", "")
        pos = int(row["position"]) if str(row.get("position", "")).isdigit() else None
        labels = set(filter(None, str(row.get("context_labels", "")).split(";")))

        new_labels = set()
        plen = lengths.get(acc, float("nan"))
        new_labels.add(qbucket(plen, length_cuts, "protein_length"))

        gc = gnomad.get(gene, {})
        new_labels.add(pli_bucket(gc.get("pLI", float("nan"))))
        new_labels.add(loeuf_bucket(gc.get("loeuf_upper", float("nan"))))
        mis_z = gc.get("mis_z", float("nan"))
        if math.isfinite(mis_z):
            if mis_z >= 3:
                new_labels.add("missense_z_high_ge3")
            elif mis_z <= 0:
                new_labels.add("missense_z_low_le0")
            else:
                new_labels.add("missense_z_mid_0_3")
        else:
            new_labels.add("missense_z_missing")

        mech = mechanism.get(gene)
        new_labels.add(f"mechanism_{mech}" if mech else "mechanism_missing")

        pfams = []
        if pos is not None:
            pfams = [pfam for start, end, pfam in pfam_intervals.get(acc, []) if start <= pos <= end]
        if pfams:
            max_size = max(pfam_sizes.get(pfam, 0) for pfam in pfams)
            new_labels.add(qbucket(float(max_size), pfam_size_cuts, "pfam_family_size"))
        else:
            max_size = float("nan")
            new_labels.add("pfam_no_variant_domain")

        llr_value = llr.get((gene, variant), float("nan"))
        new_labels.add(qbucket(abs(llr_value), llr_abs_cuts, "esm2_llr_abs_confidence"))
        if math.isfinite(llr_value):
            if llr_value <= -3:
                new_labels.add("esm2_llr_strong_deleterious_le-3")
            elif llr_value < 0:
                new_labels.add("esm2_llr_mild_deleterious")
            else:
                new_labels.add("esm2_llr_tolerated_ge0")
        else:
            new_labels.add("esm2_llr_missing")

        blosum_label, blosum_score = blosum_bucket(variant)
        new_labels.add(blosum_label)

        labels |= new_labels
        all_new_labels |= new_labels
        out = dict(row)
        out.update(
            {
                "protein_length": "" if not math.isfinite(plen) else int(plen),
                "gnomad_pLI": gc.get("pLI", ""),
                "gnomad_loeuf_upper": gc.get("loeuf_upper", ""),
                "gnomad_mis_z": gc.get("mis_z", ""),
                "gene_mechanism": mech or "",
                "pfam_variant_domains": ";".join(pfams),
                "pfam_max_family_size": "" if not math.isfinite(max_size) else int(max_size),
                "esm2_llr": "" if not math.isfinite(llr_value) else llr_value,
                "blosum62": blosum_score,
                "new_context_labels": ";".join(sorted(new_labels)),
                "all_context_labels": ";".join(sorted(labels)),
            }
        )
        enriched.append(out)

    stat_rows = [r for r in enriched if int(r["review_stars"]) >= args.min_review_stars]
    all_labels = sorted({lab for r in stat_rows for lab in str(r["all_context_labels"]).split(";") if lab})
    tests = []
    for outcome in ["SAE_right_AM_wrong", "AM_right_SAE_wrong"]:
        outcome_set = [r for r in stat_rows if r["outcome"] == outcome]
        other_set = [r for r in stat_rows if r["outcome"] != outcome]
        for lab in all_labels:
            a = sum(lab in r["all_context_labels"].split(";") for r in outcome_set)
            b = len(outcome_set) - a
            c = sum(lab in r["all_context_labels"].split(";") for r in other_set)
            d = len(other_set) - c
            if a + c == 0:
                continue
            odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            tests.append(
                {
                    "outcome": outcome,
                    "context_label": lab,
                    "context_source": "new_protein_context" if lab in all_new_labels else "original_residue_context",
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
    enriched_fields = list(enriched[0].keys()) if enriched else []
    write_tsv(args.out_dir / "typed_disagreements_extended.tsv", enriched, enriched_fields)
    test_fields = [
        "outcome",
        "context_label",
        "context_source",
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
    ]
    write_tsv(args.out_dir / "extended_context_enrichment.tsv", tests, test_fields)

    outcome_counts = Counter(r["outcome"] for r in stat_rows)
    sig05 = [r for r in tests if r["significant_q05"]]
    sig01 = [r for r in tests if r["significant_q01"]]
    sae_new_hits = [
        r
        for r in tests
        if r["outcome"] == "SAE_right_AM_wrong"
        and r["context_source"] == "new_protein_context"
        and r["significant_q05"]
    ]
    acceptance = bool(sae_new_hits)
    top = sorted(tests, key=lambda r: float(r["q_value"]))

    summary = {
        "task": "R1-Save-3 extended AM-vs-SAE disagreement typing",
        "status": "completed",
        "n_disagreements": len(enriched),
        "n_review_filtered": len(stat_rows),
        "min_review_stars": args.min_review_stars,
        "outcome_counts_review_filtered": dict(outcome_counts),
        "n_context_tests": len(tests),
        "n_new_context_labels": len(all_new_labels),
        "n_significant_q05": len(sig05),
        "n_significant_q01": len(sig01),
        "sae_right_new_context_q05_hits": sae_new_hits,
        "acceptance_gate": "At least one new protein-level context has BH q<0.05 positive enrichment for SAE_right_AM_wrong.",
        "acceptance_pass": acceptance,
        "unavailable_opus_contexts": [
            "MSA depth quartile: no cached per-protein MSA depth table yet",
            "IUPred disorder fraction: no staged IUPred output yet",
        ],
        "outputs": {
            "typed_disagreements_extended": str(args.out_dir / "typed_disagreements_extended.tsv"),
            "extended_context_enrichment": str(args.out_dir / "extended_context_enrichment.tsv"),
            "summary_md": str(args.out_dir / "summary.md"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    md = [
        "# Extended AM-vs-SAE Disagreement Typing",
        "",
        "This CPU-only diagnostic extends the residue-context disagreement audit with staged protein-level contexts.",
        "",
        f"- Disagreements: {len(enriched)}",
        f"- Review-filtered disagreements (stars >= {args.min_review_stars}): {len(stat_rows)}",
        f"- Outcome counts after review filter: {dict(outcome_counts)}",
        f"- Context tests: {len(tests)}",
        f"- New protein-context labels: {len(all_new_labels)}",
        f"- Significant at BH q < 0.05: {len(sig05)}",
        f"- Significant at BH q < 0.01: {len(sig01)}",
        f"- Acceptance gate: {'PASS' if acceptance else 'FAIL'}",
        "",
        "## Top Context Tests",
        "",
        "| Outcome | context | source | outcome frac | other frac | odds ratio | q |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in top[:20]:
        odds = r["odds_ratio"]
        odds_s = f"{float(odds):.3g}" if odds != "inf" else "inf"
        md.append(
            f"| {r['outcome']} | {r['context_label']} | {r['context_source']} | "
            f"{float(r['outcome_fraction']):.3f} | {float(r['other_fraction']):.3f} | "
            f"{odds_s} | {float(r['q_value']):.3g} |"
        )
    md += [
        "",
        "## Resource Gaps",
        "",
        "- MSA depth quartile was not included because no cached per-protein MSA depth table is staged yet.",
        "- IUPred disorder was not included because no staged IUPred residue-disorder output is available yet.",
        "",
        "## Interpretation",
        "",
        "- PASS would support a defensible SAE-complementary disagreement context.",
        "- FAIL means the AM-vs-SAE disagreements remain illustrative rather than statistically typed.",
        "",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(md))
    print(f"Wrote {args.out_dir / 'summary.md'}")
    print(f"Acceptance: {'PASS' if acceptance else 'FAIL'}")


if __name__ == "__main__":
    main()
