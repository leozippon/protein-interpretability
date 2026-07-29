#!/usr/bin/env python
"""Annotation-selected feature LR + ESM-2 LLR ensemble (fast).

Script 12 was slow because XGBoost on 52k features was intractable.
This script does only the winning step: L2 logistic regression on
annotation-selected features (F1 >= 0.1) + ensemble with per-variant
LLR, 5-fold CV. No XGBoost, no threshold sweep.

Output: ensemble_annotated_llr.json with AUCs for:
  - SAE annotation-selected LR alone
  - LLR alone
  - Logistic ensemble LR(SAE) + LLR

Usage:
    python r1_encoder_interpretability_benchmark/scripts/14_ensemble_annotated_llr.py
"""

import json
import os
import pickle
import sys

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
F1_THRESHOLD = 0.1
LR_C = 0.1


def load_annotation_metadata():
    meta_by_layer = {}
    for layer in LAYERS:
        pkl = f"r1_encoder_interpretability_benchmark/results/annotation_alignment/ours_3B_l{layer}_step500000.pkl"
        if not os.path.exists(pkl):
            print(f"  Missing annotation: {pkl}")
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
    return meta_by_layer


def build_feature_vectors(meta_by_layer):
    with open(f"{OUT_DIR}/scaled_perturbation_signatures.pkl", "rb") as f:
        sigs = pickle.load(f)

    selected = {}
    for layer, m in meta_by_layer.items():
        idx = np.where(m["f1"] >= F1_THRESHOLD)[0]
        selected[layer] = idx

    X_rows = []
    y_rows = []
    meta_rows = []
    for key, var_sigs in sigs.items():
        s0 = var_sigs[0]
        clin = s0.get("clinical_significance", "").lower()
        is_path = ("pathogenic" in clin and "benign" not in clin and
                   "conflicting" not in clin)
        is_benign = "benign" in clin and "pathogenic" not in clin
        if not (is_path or is_benign):
            continue

        sig_by_layer = {s["layer"]: s for s in var_sigs}
        parts = []
        for layer in LAYERS:
            idx = selected.get(layer, np.array([], dtype=np.int64))
            if len(idx) == 0 or layer not in sig_by_layer:
                parts.append(np.zeros(len(idx) * 3, dtype=np.float32))
                continue
            s = sig_by_layer[layer]
            dl = s["delta_local"][idx].astype(np.float32)
            dg = s["delta_global"][idx].astype(np.float32)
            w = meta_by_layer[layer]["f1"][idx] * meta_by_layer[layer]["cat_w"][idx]
            parts.append(np.concatenate([np.abs(dl), np.abs(dg), dl * w]))

        X_rows.append(np.concatenate(parts))
        y_rows.append(1 if is_path else 0)
        meta_rows.append({
            "gene": s0["gene"],
            "variant": s0["variant_str"],
            "clinical_significance": s0.get("clinical_significance", ""),
        })

    X = np.stack(X_rows).astype(np.float32)
    y = np.array(y_rows, dtype=np.int32)
    return X, y, meta_rows


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    print("=" * 70)
    print("  Annotation LR + LLR Ensemble")
    print("=" * 70)

    meta_by_layer = load_annotation_metadata()
    for layer, m in meta_by_layer.items():
        n = int((m["f1"] >= F1_THRESHOLD).sum())
        print(f"  L{layer}: {n} features selected (F1 >= {F1_THRESHOLD})")

    print("\nBuilding feature matrix...", flush=True)
    X, y, meta = build_feature_vectors(meta_by_layer)
    print(f"  X={X.shape}  y={y.shape}  path={int(y.sum())} benign={int((1-y).sum())}")

    good = X.std(0) > 1e-8
    X = X[:, good]
    print(f"  After constant removal: {X.shape}")
    X_scaled = StandardScaler().fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\nRunning LR C={LR_C} (SAE only)...", flush=True)
    clf = LogisticRegression(C=LR_C, max_iter=1000, solver="lbfgs")
    sae_probs = cross_val_predict(clf, X_scaled, y, cv=cv,
                                   method="predict_proba")[:, 1]
    sae_auc = float(roc_auc_score(y, sae_probs))
    print(f"  SAE alone AUC={sae_auc:.4f}")

    with open(f"{OUT_DIR}/esm2_per_variant_llr.json") as f:
        pv = json.load(f)
    pv_lookup = {(r["gene"], r["variant"]): r["llr"] for r in pv}

    llr = np.zeros(len(meta))
    matched = 0
    for i, m in enumerate(meta):
        k = (m["gene"], m["variant"])
        if k in pv_lookup:
            llr[i] = -pv_lookup[k]
            matched += 1
    print(f"  Matched {matched}/{len(meta)} LLR scores")

    llr_auc = float(roc_auc_score(y, llr))
    print(f"  LLR alone AUC={llr_auc:.4f}")

    X_comb = np.stack([sae_probs, llr], axis=1)
    clf2 = LogisticRegression(max_iter=1000)
    ens_probs = cross_val_predict(clf2, X_comb, y, cv=cv,
                                   method="predict_proba")[:, 1]
    ens_auc = float(roc_auc_score(y, ens_probs))
    print(f"  LR(SAE) + LLR ensemble AUC={ens_auc:.4f}")

    # Also try raw ensemble (equal weight z-scored)
    sae_z = (sae_probs - sae_probs.mean()) / (sae_probs.std() + 1e-8)
    llr_z = (llr - llr.mean()) / (llr.std() + 1e-8)
    simple = sae_z + llr_z
    simple_auc = float(roc_auc_score(y, simple))
    print(f"  Simple z-sum ensemble AUC={simple_auc:.4f}")

    summary = {
        "sae_annotated_lr_C0.1": {"auc": sae_auc, "n": int(len(y))},
        "llr_alone": {"auc": llr_auc, "matched": matched},
        "logistic_ensemble": {"auc": ens_auc},
        "simple_z_ensemble": {"auc": simple_auc},
        "config": {
            "f1_threshold": F1_THRESHOLD,
            "lr_C": LR_C,
            "layers": LAYERS,
            "n_features": int(X.shape[1]),
            "n_variants": int(len(y)),
        },
    }

    with open(f"{OUT_DIR}/ensemble_annotated_llr.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + "=" * 70)
    print(f"  Saved: {OUT_DIR}/ensemble_annotated_llr.json")
    for k, v in summary.items():
        if "auc" in v:
            print(f"    {k:<30s} AUC={v['auc']:.4f}")


if __name__ == "__main__":
    main()
