#!/usr/bin/env python
"""Case study analysis of SAE variant perturbation signatures.

Selects well-known disease variants and shows how SAE features
reveal the biological mechanism of pathogenicity. This provides
the interpretability showcase for the paper.

Key case studies:
  - TP53: Tumor suppressor with known LOF (DNA binding domain mutations)
  - KRAS: Oncogene with known GOF (GTPase-activating mutations)
  - PTEN: Tumor suppressor with LOF (phosphatase domain)
  - KCNQ1: Ion channel with both LOF and GOF (Long QT vs Short QT)
  - HBB: Hemoglobin with diverse mechanisms (sickle cell = neomorphic)

For each variant, the script shows:
  1. Which SAE features are most perturbed
  2. What biological annotations those features align with
  3. How the perturbation pattern differs between pathogenic and benign
  4. Whether the SAE tells us something the LLR can't

No GPU required — operates on saved data.

Usage:
    python scripts/10_case_study_analysis.py
"""

import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


CASE_STUDY_GENES = ["TP53", "KRAS", "PTEN", "KCNQ1", "HBB"]


def load_data():
    """Load all required data."""
    # Perturbation signatures
    with open("results/variant_effect/scaled_perturbation_signatures.pkl", "rb") as f:
        sigs = pickle.load(f)

    # Annotation results
    annotation_results = {}
    for layer in [19, 23, 27, 31, 35]:
        pkl = f"results/annotation_alignment/ours_3B_l{layer}_step500000.pkl"
        if os.path.exists(pkl):
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            annotation_results[layer] = data["results"]

    # ESM-2 LLR results
    llr_data = {}
    llr_path = "results/variant_effect/esm2_baseline_comparison.json"
    if os.path.exists(llr_path):
        with open(llr_path) as f:
            llr_data = json.load(f)

    return sigs, annotation_results, llr_data


def get_top_perturbed_features(delta_local, annotation_results_layer, top_n=10):
    """Get the most perturbed features and their annotations."""
    abs_delta = np.abs(delta_local)
    top_idx = np.argsort(abs_delta)[::-1][:top_n]

    results = []
    for idx in top_idx:
        feat = annotation_results_layer[idx]
        results.append({
            "feature_idx": int(idx),
            "delta": float(delta_local[idx]),
            "abs_delta": float(abs_delta[idx]),
            "annotation": feat.best_annotation if feat.alive else "DEAD",
            "f1": float(feat.best_f1) if feat.alive else 0.0,
            "classification": feat.classification,
        })
    return results


def analyze_gene(gene, sigs, annotation_results):
    """Detailed perturbation analysis for one gene."""
    # Find all variants for this gene
    gene_variants = []
    for key, var_sigs in sigs.items():
        s0 = var_sigs[0]
        if s0["gene"] != gene:
            continue

        clin_sig = s0.get("clinical_significance", "")
        sig_lower = clin_sig.lower()
        is_path = "pathogenic" in sig_lower and "benign" not in sig_lower and "conflicting" not in sig_lower
        is_benign = "benign" in sig_lower and "pathogenic" not in sig_lower

        if not (is_path or is_benign):
            continue

        gene_variants.append({
            "key": key,
            "variant": s0["variant_str"],
            "label": "PATH" if is_path else "BENIGN",
            "sigs": var_sigs,
        })

    if not gene_variants:
        return None

    print(f"\n{'='*70}")
    print(f"  CASE STUDY: {gene}")
    print(f"{'='*70}")
    print(f"  Variants: {len(gene_variants)} "
          f"({sum(1 for v in gene_variants if v['label']=='PATH')} pathogenic, "
          f"{sum(1 for v in gene_variants if v['label']=='BENIGN')} benign)")

    # For each variant, show top perturbed features at layers 31 and 35
    analysis_layers = [31, 35]
    for var in gene_variants:
        sig_by_layer = {s["layer"]: s for s in var["sigs"]}
        raw_pert = sum(float(np.abs(s["delta_local"]).sum()) for s in var["sigs"])

        print(f"\n  --- {gene} {var['variant']} ({var['label']}) ---")
        print(f"  Total perturbation: {raw_pert:.1f}")

        for layer in analysis_layers:
            if layer not in sig_by_layer or layer not in annotation_results:
                continue

            s = sig_by_layer[layer]
            top_feats = get_top_perturbed_features(
                s["delta_local"], annotation_results[layer], top_n=5)

            print(f"\n  Layer {layer} — Top 5 most perturbed features:")
            for i, feat in enumerate(top_feats):
                direction = "+" if feat["delta"] > 0 else "-"
                ann_str = feat["annotation"] if feat["annotation"] else "NOVEL"
                f1_str = f" (F1={feat['f1']:.3f})" if feat["f1"] > 0.1 else ""
                print(f"    {i+1}. Feature {feat['feature_idx']:5d}: "
                      f"delta={direction}{feat['abs_delta']:.3f}  "
                      f"[{feat['classification']}] {ann_str}{f1_str}")

    # Compare pathogenic vs benign: which annotation categories are differentially disrupted?
    print(f"\n  --- Annotation Category Disruption Comparison ---")

    cat_disruptions_path = defaultdict(list)
    cat_disruptions_benign = defaultdict(list)

    for var in gene_variants:
        cat_pert = defaultdict(float)
        for s in var["sigs"]:
            layer = s["layer"]
            if layer not in annotation_results:
                continue
            delta = np.abs(s["delta_local"])
            ann_res = annotation_results[layer]
            for i in range(len(delta)):
                r = ann_res[i]
                if r.alive and r.best_f1 > 0.1 and r.best_annotation:
                    cat = r.best_annotation.split("/")[0] if "/" in r.best_annotation else r.best_annotation
                    cat_pert[cat] += float(delta[i] * r.best_f1)

        target = cat_disruptions_path if var["label"] == "PATH" else cat_disruptions_benign
        for cat, val in cat_pert.items():
            target[cat].append(val)

    all_cats = sorted(set(list(cat_disruptions_path.keys()) + list(cat_disruptions_benign.keys())))
    print(f"  {'Category':<20s} {'Path mean':>10s} {'Ben mean':>10s} {'Ratio':>8s}")
    for cat in all_cats:
        p_vals = cat_disruptions_path.get(cat, [0])
        b_vals = cat_disruptions_benign.get(cat, [0])
        pm = np.mean(p_vals) if p_vals else 0
        bm = np.mean(b_vals) if b_vals else 0
        ratio = pm / max(bm, 1e-6)
        if pm + bm > 0.01:
            print(f"  {cat:<20s} {pm:10.3f} {bm:10.3f} {ratio:8.2f}")

    # Per-layer perturbation profile comparison
    print(f"\n  --- Per-Layer Perturbation Profile ---")
    layers = [19, 23, 27, 31, 35]
    print(f"  {'Variant':<12s} {'Label':<7s}", end="")
    for l in layers:
        print(f" {'L'+str(l):>8s}", end="")
    print(f" {'Total':>8s}")

    for var in gene_variants:
        sig_by_layer = {s["layer"]: s for s in var["sigs"]}
        print(f"  {var['variant']:<12s} {var['label']:<7s}", end="")
        total = 0
        for l in layers:
            if l in sig_by_layer:
                p = float(np.abs(sig_by_layer[l]["delta_local"]).sum())
                print(f" {p:8.1f}", end="")
                total += p
            else:
                print(f" {'--':>8s}", end="")
        print(f" {total:8.1f}")

    return {
        "gene": gene,
        "n_pathogenic": sum(1 for v in gene_variants if v["label"] == "PATH"),
        "n_benign": sum(1 for v in gene_variants if v["label"] == "BENIGN"),
        "variants": [{
            "variant": v["variant"],
            "label": v["label"],
            "total_perturbation": sum(
                float(np.abs(s["delta_local"]).sum()) for s in v["sigs"]),
        } for v in gene_variants],
    }


def main():
    print("=" * 70)
    print("  Case Study Analysis: SAE Variant Perturbation Signatures")
    print("=" * 70)

    sigs, annotation_results, llr_data = load_data()

    case_studies = {}
    for gene in CASE_STUDY_GENES:
        result = analyze_gene(gene, sigs, annotation_results)
        if result:
            case_studies[gene] = result

    # Save
    out_dir = "results/variant_effect"
    os.makedirs(out_dir, exist_ok=True)

    # Convert to JSON-serializable
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return super().default(obj)

    with open(os.path.join(out_dir, "case_studies.json"), "w") as f:
        json.dump(case_studies, f, indent=2, cls=NumpyEncoder)

    print(f"\n  Saved to {out_dir}/case_studies.json")


if __name__ == "__main__":
    main()
