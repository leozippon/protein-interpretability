#!/usr/bin/env python
"""Annotation-selected feature variant classifier.

Lessons from script 11: naive full 163k-dim vectors OVERFIT (AUC=0.747)
with only 2000 samples. The 60-dim summary stats beat it (0.836) because
annotation weights encode strong priors.

This script takes the middle ground:
  - Select ONLY annotated features (F1 > threshold) per layer
  - Optionally multiply each selected feature by its F1 score
  - Use the resulting "interpretable feature vector" (~100-500 dims)
  - Train L2 LR, MLP, and an ensemble with LLR

Also adds:
  - Per-variant LLR matching (reads esm2_per_variant_llr.json if present)
  - Multiple feature selection strategies (F1 > 0.1, 0.2, 0.3)

Goal: Leverage biological priors while using richer signal than 12 stats.

Usage:
    python scripts/12_annotated_feature_classifier.py
"""

import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = "r1_encoder_interpretability_benchmark/results/variant_effect"
LAYERS = [19, 23, 27, 31, 35]
CATEGORY_WEIGHTS = {
    "functional": 5.0,
    "ptm": 4.0,
    "domain": 3.0,
    "region": 2.0,
    "topology": 1.5,
    "secondary_structure": 1.0,
    "chain": 0.5,
}


def load_annotation_metadata():
    """Load per-feature F1 and category info for each layer."""
    meta_by_layer = {}
    for layer in LAYERS:
        pkl = f"r1_encoder_interpretability_benchmark/results/annotation_alignment/ours_3B_l{layer}_step500000.pkl"
        if not os.path.exists(pkl):
            print(f"  Missing annotation file: {pkl}")
            continue
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        results = data["results"]
        d_sae = len(results)
        f1 = np.zeros(d_sae, dtype=np.float32)
        cat_w = np.zeros(d_sae, dtype=np.float32)
        for r in results:
            if not r.alive:
                continue
            f1[r.feature_idx] = r.best_f1 or 0.0
            cat = r.best_annotation.split("/")[0] if r.best_annotation and "/" in r.best_annotation else (r.best_annotation or "")
            cat_w[r.feature_idx] = CATEGORY_WEIGHTS.get(cat, 1.0)
        meta_by_layer[layer] = {"f1": f1, "cat_w": cat_w, "d_sae": d_sae}
        n_known = int((f1 > 0.2).sum())
        n_useful = int((f1 > 0.1).sum())
        print(f"  L{layer}: {n_known} known (F1>0.2), {n_useful} useful (F1>0.1)")
    return meta_by_layer


def build_feature_vectors(meta_by_layer, f1_threshold=0.1, weighted=True):
    """Assemble feature vectors using only annotated features.

    For each layer:
      - Select features with F1 > threshold
      - Compute (delta_local, delta_global, signed_delta_local) for those features
      - Optionally weight by F1 * category_importance

    Returns:
        X: (n_variants, total_selected_features * 3) float32 array
        y: (n_variants,) labels
        meta: list of dicts
    """
    print(f"\n[1/4] Loading perturbation signatures...", flush=True)
    with open(f"{OUT_DIR}/scaled_perturbation_signatures.pkl", "rb") as f:
        sigs = pickle.load(f)

    # Determine selected feature indices per layer
    selected = {}
    for layer, m in meta_by_layer.items():
        mask = m["f1"] >= f1_threshold
        idx = np.where(mask)[0]
        selected[layer] = idx
        print(f"  L{layer}: {len(idx)} features selected (F1 >= {f1_threshold})")

    # Build matrix
    print(f"\n[2/4] Building matrices...", flush=True)
    X_rows = []
    y_rows = []
    meta_rows = []

    for key, var_sigs in sigs.items():
        s0 = var_sigs[0]
        clin_sig = s0.get("clinical_significance", "")
        sig_lower = clin_sig.lower()
        is_path = ("pathogenic" in sig_lower and
                   "benign" not in sig_lower and
                   "conflicting" not in sig_lower)
        is_benign = ("benign" in sig_lower and
                     "pathogenic" not in sig_lower)
        if not (is_path or is_benign):
            continue

        sig_by_layer = {s["layer"]: s for s in var_sigs}
        feature_parts = []
        for layer in LAYERS:
            idx = selected.get(layer, np.array([], dtype=np.int64))
            if len(idx) == 0 or layer not in sig_by_layer:
                feature_parts.append(np.zeros(len(idx) * 3, dtype=np.float32))
                continue
            s = sig_by_layer[layer]
            dl = s["delta_local"][idx].astype(np.float32)
            dg = s["delta_global"][idx].astype(np.float32)
            if weighted:
                w = meta_by_layer[layer]["f1"][idx] * meta_by_layer[layer]["cat_w"][idx]
                dl_w = dl * w
                dg_w = dg * w
                # Magnitude, signed magnitude, and weighted magnitude
                feature_parts.append(np.concatenate([np.abs(dl), np.abs(dg), dl_w]))
            else:
                feature_parts.append(np.concatenate([np.abs(dl), np.abs(dg), dl]))

        X_rows.append(np.concatenate(feature_parts))
        y_rows.append(1 if is_path else 0)
        meta_rows.append({
            "gene": s0["gene"],
            "variant": s0["variant_str"],
            "clinical_significance": clin_sig,
        })

    X = np.stack(X_rows).astype(np.float32)
    y = np.array(y_rows, dtype=np.int32)
    print(f"  X.shape={X.shape}, y.shape={y.shape}", flush=True)
    print(f"  Pathogenic: {int(y.sum())}, Benign: {int((1-y).sum())}", flush=True)
    return X, y, meta_rows


def run_classifiers(X, y):
    """Try several classifiers with 5-fold CV."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score

    results = {}

    # Remove constant columns
    good = X.std(0) > 1e-8
    X = X[:, good]
    print(f"\n[3/4] After constant removal: X.shape={X.shape}", flush=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
        probs = cross_val_predict(clf, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(y, probs))
        acc = float(((probs > 0.5).astype(int) == y).mean())
        results[f"lr_C{C}"] = {"auc": auc, "acc": acc, "scores": probs.tolist()}
        print(f"  LR C={C}: AUC={auc:.4f}  Acc={acc:.4f}", flush=True)

    # XGBoost
    try:
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            tree_method="hist",
            n_jobs=4,
            eval_metric="logloss",
            verbosity=0,
        )
        probs = cross_val_predict(clf, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(y, probs))
        acc = float(((probs > 0.5).astype(int) == y).mean())
        results["xgb"] = {"auc": auc, "acc": acc, "scores": probs.tolist()}
        print(f"  XGBoost: AUC={auc:.4f}  Acc={acc:.4f}", flush=True)
    except ImportError:
        pass

    return results


def run_ensemble_with_llr(scores_sae, y, meta):
    """Ensemble SAE scores with LLR via 5-fold CV logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score, accuracy_score

    per_var_path = f"{OUT_DIR}/esm2_per_variant_llr.json"
    if not os.path.exists(per_var_path):
        print(f"  No {per_var_path} — cannot ensemble with LLR", flush=True)
        return {}

    with open(per_var_path) as f:
        pv = json.load(f)
    pv_lookup = {(r["gene"], r["variant"]): r["llr"] for r in pv}

    llr = np.zeros(len(meta))
    matched = 0
    for i, m in enumerate(meta):
        k = (m["gene"], m["variant"])
        if k in pv_lookup:
            llr[i] = -pv_lookup[k]
            matched += 1

    print(f"\n  Matched {matched}/{len(meta)} LLR scores", flush=True)
    if matched < len(meta) // 2:
        return {}

    X_comb = np.stack([scores_sae, llr], axis=1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    probs = cross_val_predict(clf, X_comb, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, probs))
    acc = float(((probs > 0.5).astype(int) == y).mean())

    llr_auc = float(roc_auc_score(y, llr))
    sae_auc = float(roc_auc_score(y, scores_sae))

    return {
        "llr_alone": {"auc": llr_auc},
        "sae_alone": {"auc": sae_auc},
        "ensemble": {"auc": auc, "acc": acc, "scores": probs.tolist()},
    }


def main():
    print("=" * 70)
    print("  Annotation-Selected Feature Classifier")
    print("=" * 70)

    print("\nLoading annotation metadata...", flush=True)
    meta_by_layer = load_annotation_metadata()

    all_results = {}

    for threshold in [0.1, 0.2, 0.3]:
        print(f"\n{'='*70}")
        print(f"  Feature selection threshold: F1 >= {threshold}")
        print(f"{'='*70}")

        X, y, meta = build_feature_vectors(meta_by_layer, f1_threshold=threshold,
                                            weighted=True)
        if X.shape[1] == 0:
            print("  No features selected — skipping")
            continue

        res = run_classifiers(X, y)
        for k, v in res.items():
            all_results[f"t{threshold}_{k}"] = v

        # Use best model for ensemble
        best_key = max(res.keys(), key=lambda k: res[k]["auc"])
        best_scores = np.array(res[best_key]["scores"])
        print(f"\n  Best at threshold {threshold}: {best_key} AUC={res[best_key]['auc']:.4f}")

        if threshold == 0.2:
            ens = run_ensemble_with_llr(best_scores, y, meta)
            for k, v in ens.items():
                all_results[f"t{threshold}_{k}"] = v

    # Save
    summary = {}
    for k, v in all_results.items():
        v2 = {kk: vv for kk, vv in v.items() if kk != "scores"}
        summary[k] = v2

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/annotated_feature_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    # Sort by AUC
    sorted_items = sorted(summary.items(), key=lambda x: x[1].get("auc", 0),
                          reverse=True)
    for k, v in sorted_items:
        print(f"  {k:<35s} AUC={v.get('auc', 0):.4f}")

    print(f"\n  Saved to {OUT_DIR}/annotated_feature_results.json")


if __name__ == "__main__":
    main()
