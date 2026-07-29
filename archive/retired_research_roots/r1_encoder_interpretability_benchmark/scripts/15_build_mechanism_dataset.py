#!/usr/bin/env python
"""Build LOF / GOF / DN / Neomorphic mechanism labels for ClinVar variants.

The distinguishing R1 claim is: ESM-2 LLR predicts pathogenicity but cannot
distinguish mechanism classes. SAE features — because they are functionally
labeled — can. This script produces the labeled dataset that lets us train
and evaluate a mechanism classifier.

Label sources (in priority order, fall through if not available):
  1. Gerasimavicius et al. 2022 supplementary table — explicit per-variant
     LOF/GOF/DN labels (~1300 variants)
  2. Badonyi & Marsh 2025 non-LOF table (if downloaded)
  3. Curated gene-level dominant mechanism fallback (tumor-suppressor → LOF,
     oncogene → GOF, collagen/HBB-like → DN)

Output: `r1_encoder_interpretability_benchmark/results/variant_effect/variant_mechanisms.tsv` with
columns `gene`, `variant`, `mechanism`, `source`.

Usage:
    python r1_encoder_interpretability_benchmark/scripts/15_build_mechanism_dataset.py \
        --gerasimavicius data/mechanism/gerasimavicius2022_TableS1.tsv \
        --signatures r1_encoder_interpretability_benchmark/results/variant_effect/scaled_perturbation_signatures.pkl
"""

import argparse
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Gene-level dominant mechanism (used as fallback when per-variant labels
# are unavailable). Curated from the ClinGen Dosage Sensitivity map and
# widely-used oncogene/tumor-suppressor lists.
GENE_DOMINANT_MECHANISM = {
    # Tumor suppressors — primarily LOF
    "TP53": "LOF",
    "PTEN": "LOF",
    "BRCA1": "LOF",
    "BRCA2": "LOF",
    "APC": "LOF",
    "NF1": "LOF",
    "NF2": "LOF",
    "RB1": "LOF",
    "VHL": "LOF",
    "MLH1": "LOF",
    "MSH2": "LOF",
    "MSH6": "LOF",
    "PMS2": "LOF",
    "STK11": "LOF",
    "CDKN2A": "LOF",
    "SMAD4": "LOF",
    "FBN1": "LOF",      # Marfan-associated, classical LOF/haploinsufficient
    "CFTR": "LOF",
    "DMD": "LOF",
    "ATM": "LOF",
    # Oncogenes — primarily GOF
    "KRAS": "GOF",
    "HRAS": "GOF",
    "NRAS": "GOF",
    "BRAF": "GOF",
    "EGFR": "GOF",
    "PIK3CA": "GOF",
    "MYC": "GOF",
    "AKT1": "GOF",
    "FLT3": "GOF",
    "JAK2": "GOF",
    # Dominant-negative canonical examples
    "COL1A1": "DN",
    "COL1A2": "DN",
    "COL3A1": "DN",
    "COL2A1": "DN",
    "HBB": "DN",        # many HBB variants act dominant-negative on tetramer
    "KCNQ1": "DN",      # channel tetramerization, many dominant-negative forms
    "FGFR3": "GOF",
    "RUNX1": "LOF",
    "GATA3": "LOF",
    "LMNA": "DN",
}


def normalize_mechanism(mech: str) -> str | None:
    """Normalize mechanism labels from different supplement tables."""
    raw = mech.strip().upper().replace("-", "_").replace(" ", "_")
    mapping = {
        "LOF": "LOF",
        "OTHER_LOF": "LOF",
        "HI": "LOF",
        "HAPLOINSUFFICIENCY": "LOF",
        "AR": "LOF",
        "GOF": "GOF",
        "DN": "DN",
        "DNF": "DN",
        "DOMINANT_NEGATIVE": "DN",
        "DOMINANTNEGATIVE": "DN",
        "NEOMORPHIC": "NEOMORPHIC",
        "NEOMORPH": "NEOMORPHIC",
    }
    return mapping.get(raw)


def load_gerasimavicius(path: str | None) -> dict[tuple[str, str], str]:
    """Parse Gerasimavicius et al. 2022 Table S1 if available.

    Expected columns (tab-separated): gene, variant, mechanism
    mechanism ∈ {LOF, GOF, DN, DNF, Neomorphic}

    Returns:
        Dict mapping (gene, variant) → mechanism. variant in "A123B" format.
    """
    if not path or not os.path.exists(path):
        print(f"  Gerasimavicius table not found at {path}; skipping.")
        return {}
    labels: dict[tuple[str, str], str] = {}
    with open(path) as f:
        header = f.readline().strip().lower().split("\t")
        try:
            gi = header.index("gene")
            vi = header.index("variant")
            mi = header.index("mechanism")
        except ValueError:
            print(f"  Header malformed (need gene/variant/mechanism): {header}")
            return {}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < max(gi, vi, mi) + 1:
                continue
            gene = parts[gi].strip().upper()
            variant = parts[vi].strip().replace("p.", "")
            mech = normalize_mechanism(parts[mi])
            if mech:
                labels[(gene, variant)] = mech
    print(f"  Loaded {len(labels)} Gerasimavicius per-variant labels.")
    return labels


def load_badonyi(path: str | None) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    """Parse Badonyi & Marsh 2025 mechanism dataset.

    Supports:
      1. per-variant TSV: gene, variant, mechanism
      2. gene-level CSV/TSV: gene, class
    """
    if not path or not os.path.exists(path):
        return {}, {}
    with open(path, encoding="latin1") as f:
        sample = f.readline()
        delim = "\t" if sample.count("\t") >= sample.count(",") else ","
        header = sample.strip().lower().split(delim)
        lines = f.readlines()

    per_variant: dict[tuple[str, str], str] = {}
    gene_level: dict[str, str] = {}

    if {"gene", "variant", "mechanism"}.issubset(header):
        gi = header.index("gene")
        vi = header.index("variant")
        mi = header.index("mechanism")
        for line in lines:
            parts = line.rstrip("\n").split(delim)
            if len(parts) < max(gi, vi, mi) + 1:
                continue
            gene = parts[gi].strip().upper()
            variant = parts[vi].strip().replace("p.", "")
            mech = normalize_mechanism(parts[mi])
            if mech:
                per_variant[(gene, variant)] = mech
        print(f"  Loaded {len(per_variant)} Badonyi per-variant labels.")
        return per_variant, gene_level

    if {"gene", "class"}.issubset(header):
        gi = header.index("gene")
        ci = header.index("class")
        for line in lines:
            parts = line.rstrip("\n").split(delim)
            if len(parts) < max(gi, ci) + 1:
                continue
            gene = parts[gi].strip().upper()
            mech = normalize_mechanism(parts[ci])
            if mech:
                gene_level[gene] = mech
        print(f"  Loaded {len(gene_level)} Badonyi gene-level labels.")
        return per_variant, gene_level

    print(f"  Badonyi table header not recognized: {header}")
    return per_variant, gene_level


def collect_clinvar_variants(signatures_path: str) -> list[dict]:
    """Read the 2000-variant ClinVar set used throughout R1."""
    with open(signatures_path, "rb") as f:
        sigs = pickle.load(f)

    variants = []
    for key, var_sigs in sigs.items():
        s0 = var_sigs[0]
        clin = s0.get("clinical_significance", "").lower()
        is_path = ("pathogenic" in clin and "benign" not in clin and
                   "conflicting" not in clin)
        is_benign = "benign" in clin and "pathogenic" not in clin
        if not (is_path or is_benign):
            continue
        variants.append({
            "gene": s0["gene"].upper(),
            "variant": s0["variant_str"],
            "clinical_significance": s0.get("clinical_significance", ""),
            "is_path": is_path,
        })
    return variants


def assign_mechanism(variants, per_variant_labels, badonyi_gene_labels,
                     gene_fallback) -> list[dict]:
    """Attach mechanism label to each variant using priority: per-variant → gene."""
    out = []
    counts = Counter()
    for v in variants:
        gene = v["gene"]
        variant = v["variant"]
        key = (gene, variant)
        if key in per_variant_labels:
            mech = per_variant_labels[key]
            src = "per_variant"
        elif gene in badonyi_gene_labels and v["is_path"]:
            mech = badonyi_gene_labels[gene]
            src = "badonyi_gene"
        elif gene in gene_fallback and v["is_path"]:
            # Only assign gene-level mechanism to PATHOGENIC variants; benign
            # variants in a LOF gene are not LOF by default.
            mech = gene_fallback[gene]
            src = "gene_fallback"
        else:
            mech = "UNLABELED"
            src = "none"
        counts[mech] += 1
        out.append({**v, "mechanism": mech, "source": src})

    print("  Mechanism label distribution:")
    for m, n in counts.most_common():
        print(f"    {m:<12s} {n}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--signatures",
        default="r1_encoder_interpretability_benchmark/results/variant_effect/scaled_perturbation_signatures.pkl",
        help="Scaled perturbation signatures pickle (source of the 2000 variants)",
    )
    ap.add_argument(
        "--gerasimavicius",
        default="data/mechanism/gerasimavicius2022_TableS1.tsv",
        help="Path to Gerasimavicius 2022 per-variant mechanism TSV",
    )
    ap.add_argument(
        "--badonyi",
        default="data/mechanism/badonyi2025_table_S1.csv",
        help="Path to Badonyi 2025 mechanism table (per-variant TSV or gene-level CSV)",
    )
    ap.add_argument(
        "--out",
        default="r1_encoder_interpretability_benchmark/results/variant_effect/variant_mechanisms.tsv",
    )
    args = ap.parse_args()

    print("=" * 70)
    print("  Building LOF/GOF/DN/Neomorphic mechanism dataset")
    print("=" * 70)

    print(f"\n[1/4] Reading variant list from {args.signatures}...")
    variants = collect_clinvar_variants(args.signatures)
    print(f"  {len(variants)} variants (pathogenic + benign)")

    print(f"\n[2/4] Loading per-variant labels...")
    per_variant = {}
    per_variant.update(load_gerasimavicius(args.gerasimavicius))
    # Badonyi entries override Gerasimavicius ties (more recent curation)
    badonyi_per_variant, badonyi_gene = load_badonyi(args.badonyi)
    per_variant.update(badonyi_per_variant)

    print(f"\n[3/4] Assigning mechanisms (priority: per_variant → gene)...")
    labeled = assign_mechanism(
        variants,
        per_variant,
        badonyi_gene,
        GENE_DOMINANT_MECHANISM,
    )

    print(f"\n[4/4] Writing {args.out}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("gene\tvariant\tclinical_significance\tis_pathogenic\tmechanism\tsource\n")
        for v in labeled:
            f.write(f"{v['gene']}\t{v['variant']}\t{v['clinical_significance']}"
                    f"\t{int(v['is_path'])}\t{v['mechanism']}\t{v['source']}\n")

    # Also write a summary JSON
    summary = {
        "n_total": len(labeled),
        "counts": Counter(v["mechanism"] for v in labeled),
        "source_counts": Counter(v["source"] for v in labeled),
        "per_gene": Counter(v["gene"] for v in labeled),
    }
    summary["counts"] = dict(summary["counts"])
    summary["source_counts"] = dict(summary["source_counts"])
    summary["per_gene"] = dict(summary["per_gene"].most_common(30))

    summary_path = args.out.replace(".tsv", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Wrote {args.out}")
    print(f"  Wrote {summary_path}")


if __name__ == "__main__":
    main()
