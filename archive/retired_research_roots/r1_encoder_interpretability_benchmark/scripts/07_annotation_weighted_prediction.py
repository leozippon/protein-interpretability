#!/usr/bin/env python
"""Annotation-weighted variant effect prediction.

Key insight from EXP-007: raw perturbation magnitude (AUC=0.547) doesn't
separate pathogenic from benign because it weights all 16384 SAE features
equally. Most features are NOVEL (no known biological alignment), so
signal from biologically meaningful features is drowned out by noise.

This script re-scores existing perturbation signatures using annotation
F1 weights from EXP-006. A feature aligned with "active site" (F1=0.8)
contributes 0.8× its perturbation; a novel feature (F1<0.05) contributes
nearly nothing.

Multiple scoring strategies are compared:
  1. Raw (baseline): sum |delta_local| across all features — EXP-007 baseline
  2. F1-weighted: sum F1_i * |delta_local_i| — weight by best annotation F1
  3. Category-weighted: biological categories have different importance
     (functional > domain > topology > chain)
  4. Cross-layer profile: concatenate per-layer scores into a feature vector,
     then train a simple logistic regression classifier
  5. Top-K perturbation: only consider top-K most perturbed features per layer,
     weighted by F1

No GPU required — operates on saved perturbation signatures and annotation results.

Usage:
    python scripts/07_annotation_weighted_prediction.py
"""

import json
import os
import pickle
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Biological category importance weights
# Rationale: disrupting a functional site or PTM is more likely pathogenic
# than disrupting a chain boundary or generic region
CATEGORY_WEIGHTS = {
    "functional": 5.0,   # active sites, binding sites, catalytic residues
    "ptm": 4.0,          # post-translational modifications
    "domain": 3.0,       # protein domains (kinase, SH3, etc.)
    "region": 2.0,       # functional regions
    "topology": 1.5,     # transmembrane, signal peptide
    "secondary_structure": 1.0,
    "chain": 0.5,        # chain annotations (least specific)
}


def load_annotation_weights(annotation_pkl_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load per-feature annotation F1 scores and category weights.

    Returns:
        f1_weights: (d_sae,) array of best F1 scores per feature
        cat_weights: (d_sae,) array of category-weighted F1 scores
        categories: list of best annotation category per feature
    """
    with open(annotation_pkl_path, 'rb') as f:
        data = pickle.load(f)
    results = data['results']
    d_sae = len(results)

    f1_weights = np.zeros(d_sae, dtype=np.float32)
    cat_weights = np.zeros(d_sae, dtype=np.float32)
    categories = [""] * d_sae

    for r in results:
        if not r.alive:
            continue
        f1_weights[r.feature_idx] = r.best_f1

        # Extract category
        if r.best_annotation and '/' in r.best_annotation:
            cat = r.best_annotation.split('/')[0]
        else:
            cat = r.best_annotation or ""
        categories[r.feature_idx] = cat

        cat_w = CATEGORY_WEIGHTS.get(cat, 1.0)
        cat_weights[r.feature_idx] = r.best_f1 * cat_w

    return f1_weights, cat_weights, categories


def score_variant_raw(sigs: list[dict]) -> float:
    """Baseline: sum of |delta_local| across layers."""
    return sum(float(np.abs(s['delta_local']).sum()) for s in sigs)


def score_variant_f1_weighted(sigs: list[dict],
                               f1_weights_by_layer: dict[int, np.ndarray]) -> float:
    """F1-weighted: sum F1_i * |delta_local_i| across features and layers."""
    total = 0.0
    for s in sigs:
        layer = s['layer']
        if layer not in f1_weights_by_layer:
            continue
        w = f1_weights_by_layer[layer]
        delta = np.abs(s['delta_local'])
        total += float((w * delta).sum())
    return total


def score_variant_cat_weighted(sigs: list[dict],
                                cat_weights_by_layer: dict[int, np.ndarray]) -> float:
    """Category-weighted: F1 * category_importance * |delta_local|."""
    total = 0.0
    for s in sigs:
        layer = s['layer']
        if layer not in cat_weights_by_layer:
            continue
        w = cat_weights_by_layer[layer]
        delta = np.abs(s['delta_local'])
        total += float((w * delta).sum())
    return total


def score_variant_topk_f1(sigs: list[dict],
                           f1_weights_by_layer: dict[int, np.ndarray],
                           top_k: int = 50) -> float:
    """Top-K F1-weighted: only consider the K most perturbed features per layer,
    then weight by F1. This focuses on the most disrupted features."""
    total = 0.0
    for s in sigs:
        layer = s['layer']
        if layer not in f1_weights_by_layer:
            continue
        w = f1_weights_by_layer[layer]
        delta = np.abs(s['delta_local'])

        # Top-K most perturbed features
        top_indices = np.argsort(delta)[-top_k:]
        total += float((w[top_indices] * delta[top_indices]).sum())
    return total


def score_variant_known_only(sigs: list[dict],
                              f1_weights_by_layer: dict[int, np.ndarray],
                              threshold: float = 0.2) -> float:
    """Only count perturbation of features with F1 > threshold (KNOWN + PARTIAL)."""
    total = 0.0
    for s in sigs:
        layer = s['layer']
        if layer not in f1_weights_by_layer:
            continue
        w = f1_weights_by_layer[layer]
        delta = np.abs(s['delta_local'])
        mask = w > threshold
        total += float(delta[mask].sum())
    return total


def score_variant_functional_disruption(sigs: list[dict],
                                         f1_weights_by_layer: dict[int, np.ndarray],
                                         categories_by_layer: dict[int, list[str]]) -> float:
    """Score focusing on disruption of functional/ptm/domain features only."""
    important_cats = {"functional", "ptm", "domain"}
    total = 0.0
    for s in sigs:
        layer = s['layer']
        if layer not in f1_weights_by_layer:
            continue
        w = f1_weights_by_layer[layer]
        cats = categories_by_layer[layer]
        delta = np.abs(s['delta_local'])

        for i in range(len(delta)):
            if cats[i] in important_cats and w[i] > 0.1:
                total += float(delta[i] * w[i])
    return total


def extract_cross_layer_features(sigs: list[dict],
                                  f1_weights_by_layer: dict[int, np.ndarray],
                                  cat_weights_by_layer: dict[int, np.ndarray],
                                  layers: list[int]) -> np.ndarray:
    """Extract a rich feature vector for logistic regression.

    Per layer: raw_pert, f1_weighted_pert, cat_weighted_pert,
              n_known_perturbed, n_novel_perturbed, max_f1_perturbed,
              mean_delta_known, mean_delta_novel, top10_f1_pert, functional_pert
    """
    sig_by_layer = {s['layer']: s for s in sigs}
    features = []

    for layer in layers:
        if layer not in sig_by_layer:
            features.extend([0.0] * 12)
            continue

        s = sig_by_layer[layer]
        delta = np.abs(s['delta_local'])
        f1w = f1_weights_by_layer.get(layer, np.zeros_like(delta))
        cw = cat_weights_by_layer.get(layer, np.zeros_like(delta))

        known_mask = f1w > 0.2
        novel_mask = f1w < 0.05
        perturbed_mask = delta > 0.1  # significantly perturbed

        # Raw perturbation
        raw_pert = float(delta.sum())
        # F1-weighted
        f1_pert = float((f1w * delta).sum())
        # Category-weighted
        cat_pert = float((cw * delta).sum())

        # Number of known/novel features that are perturbed
        n_known_pert = int((known_mask & perturbed_mask).sum())
        n_novel_pert = int((novel_mask & perturbed_mask).sum())

        # Max F1 among perturbed features
        if perturbed_mask.any():
            max_f1_pert = float(f1w[perturbed_mask].max())
        else:
            max_f1_pert = 0.0

        # Mean perturbation of known vs novel features
        mean_delta_known = float(delta[known_mask].mean()) if known_mask.any() else 0.0
        mean_delta_novel = float(delta[novel_mask].mean()) if novel_mask.any() else 0.0

        # Top-10 F1-weighted perturbation
        top10_idx = np.argsort(f1w * delta)[-10:]
        top10_f1_pert = float((f1w[top10_idx] * delta[top10_idx]).sum())

        # Ratio: known perturbation / total perturbation
        known_pert_frac = f1_pert / max(raw_pert, 1e-6)

        # Perturbation asymmetry: are features gained or lost?
        delta_signed = s['delta_local']
        gain = float(delta_signed[delta_signed > 0].sum())
        loss = float(abs(delta_signed[delta_signed < 0].sum()))
        asymmetry = (gain - loss) / max(gain + loss, 1e-6)

        features.extend([
            raw_pert, f1_pert, cat_pert,
            n_known_pert, n_novel_pert, max_f1_pert,
            mean_delta_known, mean_delta_novel,
            top10_f1_pert, known_pert_frac,
            asymmetry,
            float(s.get('total_perturbation', raw_pert)),
        ])

    return np.array(features, dtype=np.float32)


def main():
    print("=" * 70)
    print("  Annotation-Weighted Variant Effect Prediction")
    print("=" * 70)

    # --- Load annotation weights for each layer ---
    print("\n[1/3] Loading annotation weights...")

    layers = [23, 27, 31, 35]
    f1_weights_by_layer = {}
    cat_weights_by_layer = {}
    categories_by_layer = {}

    for layer in layers:
        pkl_path = f"results/annotation_alignment/ours_3B_l{layer}_step500000.pkl"
        if not os.path.exists(pkl_path):
            print(f"  Layer {layer}: annotation file not found, skipping")
            continue
        f1w, cw, cats = load_annotation_weights(pkl_path)
        f1_weights_by_layer[layer] = f1w
        cat_weights_by_layer[layer] = cw
        categories_by_layer[layer] = cats

        n_known = int((f1w > 0.5).sum())
        n_partial = int(((f1w > 0.2) & (f1w <= 0.5)).sum())
        n_novel = int((f1w <= 0.2).sum())
        print(f"  Layer {layer}: {n_known} KNOWN, {n_partial} PARTIAL, {n_novel} NOVEL")

    # --- Load perturbation signatures ---
    print("\n[2/3] Loading perturbation signatures...")
    with open("results/variant_effect/perturbation_signatures.pkl", 'rb') as f:
        sig_data = pickle.load(f)

    # Load variant predictions for clinical significance labels
    with open("results/variant_effect/variant_predictions.json") as f:
        predictions = json.load(f)

    # Build lookup from variant key to clinical significance
    clin_sig_map = {}
    for p in predictions:
        key = (p['uniprot_id'], int(p['variant'][1:-1]),
               p['variant'][0], p['variant'][-1])
        clin_sig_map[key] = p['clinical_significance']

    print(f"  {len(sig_data)} variants with perturbation signatures")
    print(f"  {len(predictions)} variant predictions loaded")

    # --- Score variants with each strategy ---
    print("\n[3/3] Computing annotation-weighted scores...")

    strategies = {
        "raw": lambda sigs: score_variant_raw(sigs),
        "f1_weighted": lambda sigs: score_variant_f1_weighted(sigs, f1_weights_by_layer),
        "cat_weighted": lambda sigs: score_variant_cat_weighted(sigs, cat_weights_by_layer),
        "topk50_f1": lambda sigs: score_variant_topk_f1(sigs, f1_weights_by_layer, 50),
        "topk20_f1": lambda sigs: score_variant_topk_f1(sigs, f1_weights_by_layer, 20),
        "known_only": lambda sigs: score_variant_known_only(sigs, f1_weights_by_layer, 0.2),
        "functional": lambda sigs: score_variant_functional_disruption(
            sigs, f1_weights_by_layer, categories_by_layer),
    }

    # Compute scores for each variant under each strategy
    variant_records = []
    for key, sigs in sig_data.items():
        uniprot_id, pos, wt_aa, mut_aa = key
        clin_sig = clin_sig_map.get(key, "")

        is_pathogenic = ('pathogenic' in clin_sig.lower()
                         and 'benign' not in clin_sig.lower()
                         and 'conflicting' not in clin_sig.lower())
        is_benign = ('benign' in clin_sig.lower()
                     and 'pathogenic' not in clin_sig.lower())

        if not (is_pathogenic or is_benign):
            continue

        label = 1 if is_pathogenic else 0

        scores = {}
        for name, scorer in strategies.items():
            scores[name] = scorer(sigs)

        # Cross-layer features for ML classifier
        ml_features = extract_cross_layer_features(
            sigs, f1_weights_by_layer, cat_weights_by_layer, layers)

        variant_records.append({
            "key": key,
            "label": label,
            "clin_sig": clin_sig,
            "scores": scores,
            "ml_features": ml_features,
        })

    n_path = sum(1 for r in variant_records if r["label"] == 1)
    n_benign = sum(1 for r in variant_records if r["label"] == 0)
    print(f"  {len(variant_records)} variants: {n_path} pathogenic, {n_benign} benign")

    # --- Compute AUC-ROC for each strategy ---
    print("\n" + "=" * 70)
    print("  RESULTS: AUC-ROC for Pathogenicity Prediction")
    print("=" * 70)

    from sklearn.metrics import roc_auc_score, average_precision_score

    labels = np.array([r["label"] for r in variant_records])

    auc_results = {}
    for name in strategies:
        scores = np.array([r["scores"][name] for r in variant_records])
        if scores.std() < 1e-10:
            print(f"  {name:20s}: constant scores, skipping")
            continue
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        auc_results[name] = {"auc": auc, "ap": ap}
        print(f"  {name:20s}: AUC={auc:.4f}  AP={ap:.4f}")

    # --- Cross-layer logistic regression ---
    print("\n  --- Cross-Layer Logistic Regression (5-fold CV) ---")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler

    X = np.stack([r["ml_features"] for r in variant_records])
    y = labels

    # Remove constant features
    feature_std = X.std(axis=0)
    good_features = feature_std > 1e-10
    X_clean = X[:, good_features]
    print(f"  Features: {X.shape[1]} total, {good_features.sum()} non-constant")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')

    y_prob = cross_val_predict(lr, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    auc_lr = roc_auc_score(y, y_prob)
    ap_lr = average_precision_score(y, y_prob)
    accuracy = (y_pred == y).mean()

    auc_results["logistic_regression_cv"] = {"auc": auc_lr, "ap": ap_lr, "accuracy": accuracy}
    print(f"  Logistic Regression:  AUC={auc_lr:.4f}  AP={ap_lr:.4f}  Acc={accuracy:.4f}")

    # Feature importance from full model
    lr_full = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
    lr_full.fit(X_scaled, y)
    coefs = np.zeros(X.shape[1])
    coefs[good_features] = lr_full.coef_[0]

    # Name the features
    feature_names = []
    for layer in layers:
        for feat_name in ["raw_pert", "f1_pert", "cat_pert",
                          "n_known_pert", "n_novel_pert", "max_f1_pert",
                          "mean_delta_known", "mean_delta_novel",
                          "top10_f1_pert", "known_pert_frac",
                          "asymmetry", "total_pert"]:
            feature_names.append(f"L{layer}_{feat_name}")

    print(f"\n  Top 10 most important features (by |coefficient|):")
    top_idx = np.argsort(np.abs(coefs))[::-1][:10]
    for i, idx in enumerate(top_idx):
        print(f"    {i+1}. {feature_names[idx]:30s} coef={coefs[idx]:+.4f}")

    # --- Comparison table ---
    print(f"\n{'='*70}")
    print(f"  COMPARISON: Raw vs Annotation-Weighted")
    print(f"{'='*70}")
    print(f"  {'Strategy':<25s} {'AUC':>8s} {'AP':>8s} {'Improvement':>12s}")
    print(f"  {'-'*55}")
    baseline_auc = auc_results.get("raw", {}).get("auc", 0.547)
    for name in ["raw", "f1_weighted", "cat_weighted", "topk50_f1",
                 "topk20_f1", "known_only", "functional",
                 "logistic_regression_cv"]:
        if name not in auc_results:
            continue
        r = auc_results[name]
        improvement = r["auc"] - baseline_auc
        sign = "+" if improvement > 0 else ""
        extra = f"  Acc={r['accuracy']:.3f}" if "accuracy" in r else ""
        print(f"  {name:<25s} {r['auc']:8.4f} {r['ap']:8.4f} {sign}{improvement:11.4f}{extra}")

    # --- Per-category analysis ---
    print(f"\n{'='*70}")
    print(f"  PER-CATEGORY PERTURBATION ANALYSIS")
    print(f"{'='*70}")

    # For each annotation category, compute mean perturbation of associated features
    for layer in layers:
        if layer not in categories_by_layer:
            continue
        cats = categories_by_layer[layer]
        f1w = f1_weights_by_layer[layer]

        cat_pert_path = {cat: [] for cat in CATEGORY_WEIGHTS}
        cat_pert_benign = {cat: [] for cat in CATEGORY_WEIGHTS}

        for rec in variant_records:
            sig = next((s for s in sig_data[rec["key"]] if s['layer'] == layer), None)
            if sig is None:
                continue
            delta = np.abs(sig['delta_local'])
            for cat in CATEGORY_WEIGHTS:
                mask = np.array([c == cat and f1w[i] > 0.1
                                 for i, c in enumerate(cats)])
                if mask.sum() == 0:
                    continue
                pert = float(delta[mask].sum())
                if rec["label"] == 1:
                    cat_pert_path[cat].append(pert)
                else:
                    cat_pert_benign[cat].append(pert)

        print(f"\n  Layer {layer}:")
        print(f"    {'Category':<20s} {'Path mean':>10s} {'Benign mean':>12s} {'Ratio':>8s} {'N_features':>12s}")
        for cat in ["functional", "ptm", "domain", "region", "topology", "chain"]:
            p = cat_pert_path.get(cat, [])
            b = cat_pert_benign.get(cat, [])
            if not p or not b:
                continue
            pm, bm = np.mean(p), np.mean(b)
            ratio = pm / max(bm, 1e-6)
            n_feat = sum(1 for i, c in enumerate(cats) if c == cat and f1w[i] > 0.1)
            print(f"    {cat:<20s} {pm:10.2f} {bm:12.2f} {ratio:8.3f} {n_feat:12d}")

    # --- Save results ---
    out_dir = "results/variant_effect"
    os.makedirs(out_dir, exist_ok=True)

    output = {
        "auc_results": {k: {kk: round(vv, 4) for kk, vv in v.items()}
                        for k, v in auc_results.items()},
        "baseline_auc": round(baseline_auc, 4),
        "best_strategy": max(auc_results, key=lambda k: auc_results[k]["auc"]),
        "best_auc": round(max(r["auc"] for r in auc_results.values()), 4),
        "n_pathogenic": n_path,
        "n_benign": n_benign,
        "layers": layers,
        "feature_importance": [
            {"feature": feature_names[idx], "coefficient": round(float(coefs[idx]), 4)}
            for idx in top_idx
        ],
    }

    with open(os.path.join(out_dir, "annotation_weighted_results.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to {out_dir}/annotation_weighted_results.json")
    print()


if __name__ == "__main__":
    main()
