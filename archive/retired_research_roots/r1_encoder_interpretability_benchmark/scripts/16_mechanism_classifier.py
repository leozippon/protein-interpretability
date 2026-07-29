#!/usr/bin/env python
"""R1 mechanism classifier: LOF vs GOF vs DN vs Neomorphic from SAE features.

Distinguishing claim of R1: LLR predicts pathogenicity but cannot predict
mechanism class. SAE features can, because they are functionally labeled.

This script:
  1. Loads annotation-selected SAE features + per-variant LLR
  2. Loads mechanism labels produced by 15_build_mechanism_dataset.py
  3. Trains multinomial LR on SAE features only, LLR only, and SAE+LLR
  4. Reports per-class ROC AUC, macro-F1, confusion matrix
  5. Bootstrap CIs and a permutation test against the LLR baseline
  6. Extracts top SAE features per mechanism class for interpretation

Usage:
    python r1_encoder_interpretability_benchmark/scripts/16_mechanism_classifier.py \
        --mechanisms r1_encoder_interpretability_benchmark/results/variant_effect/variant_mechanisms.tsv
"""

import argparse
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

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


def load_annotation_metadata():
    meta_by_layer = {}
    for layer in LAYERS:
        pkl = f"r1_encoder_interpretability_benchmark/results/annotation_alignment/ours_3B_l{layer}_step500000.pkl"
        if not os.path.exists(pkl):
            continue
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        results = data["results"]
        d_sae = len(results)
        f1 = np.zeros(d_sae, dtype=np.float32)
        cat_w = np.zeros(d_sae, dtype=np.float32)
        best_ann = [""] * d_sae
        for r in results:
            if not r.alive:
                continue
            f1[r.feature_idx] = r.best_f1 or 0.0
            ann = r.best_annotation or ""
            cat = ann.split("/")[0] if "/" in ann else ann
            cat_w[r.feature_idx] = CATEGORY_WEIGHTS.get(cat, 1.0)
            best_ann[r.feature_idx] = ann
        meta_by_layer[layer] = {
            "f1": f1, "cat_w": cat_w, "d_sae": d_sae, "ann": best_ann,
        }
    return meta_by_layer


def load_mechanism_labels(path: str) -> dict[tuple[str, str], str]:
    labels = {}
    with open(path) as f:
        header = f.readline().strip().split("\t")
        gi = header.index("gene")
        vi = header.index("variant")
        mi = header.index("mechanism")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < max(gi, vi, mi) + 1:
                continue
            labels[(parts[gi], parts[vi])] = parts[mi]
    return labels


def build_features(meta_by_layer, mech_labels, keep_classes):
    """Assemble features + labels for variants in `keep_classes`."""
    with open(f"{OUT_DIR}/scaled_perturbation_signatures.pkl", "rb") as f:
        sigs = pickle.load(f)

    selected = {l: np.where(m["f1"] >= F1_THRESHOLD)[0]
                for l, m in meta_by_layer.items()}

    with open(f"{OUT_DIR}/esm2_per_variant_llr.json") as f:
        pv = json.load(f)
    llr_lookup = {(r["gene"], r["variant"]): r["llr"] for r in pv}

    X_sae_rows = []
    llr_rows = []
    y_rows = []
    group_rows = []
    feat_src_rows = []  # layer, idx for each column

    # Build column metadata once
    col_meta = []
    for layer in LAYERS:
        idx = selected.get(layer, np.array([], dtype=np.int64))
        for f_idx in idx:
            col_meta.append((layer, int(f_idx), "abs_local"))
        for f_idx in idx:
            col_meta.append((layer, int(f_idx), "abs_global"))
        for f_idx in idx:
            col_meta.append((layer, int(f_idx), "weighted_local"))

    for key, var_sigs in sigs.items():
        s0 = var_sigs[0]
        gene = s0["gene"].upper()
        variant = s0["variant_str"]
        mech = mech_labels.get((gene, variant))
        if mech is None or mech not in keep_classes:
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

        X_sae_rows.append(np.concatenate(parts))
        llr_val = -llr_lookup.get((gene, variant), 0.0)
        llr_rows.append(llr_val)
        y_rows.append(mech)
        group_rows.append(gene)

    X_sae = np.stack(X_sae_rows).astype(np.float32) if X_sae_rows else np.zeros((0, 0), dtype=np.float32)
    llr = np.array(llr_rows, dtype=np.float32)
    y = np.array(y_rows)
    groups = np.array(group_rows)
    return X_sae, llr, y, groups, col_meta


def make_cv(y, groups=None, n_splits=5):
    if groups is None:
        from sklearn.model_selection import StratifiedKFold
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    try:
        from sklearn.model_selection import StratifiedGroupKFold
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    except ImportError:
        from sklearn.model_selection import GroupKFold
        return GroupKFold(n_splits=n_splits)


def evaluate_multiclass(X, y, classes, n_splits=5, C=0.1, label="", groups=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (roc_auc_score, f1_score,
                                  confusion_matrix, accuracy_score)
    from sklearn.preprocessing import StandardScaler

    cv = make_cv(y, groups=groups, n_splits=n_splits)
    all_probs = np.zeros((len(y), len(classes)), dtype=np.float32)
    all_pred = np.empty(len(y), dtype=object)

    split_iter = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
    for fold, (tr, te) in enumerate(split_iter):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])
        clf = LogisticRegression(
            C=C, solver="lbfgs",
            max_iter=3000, class_weight="balanced",
        )
        clf.fit(X_tr, y[tr])
        probs = clf.predict_proba(X_te)
        fold_probs = np.zeros((len(te), len(classes)), dtype=np.float32)
        for ci, c in enumerate(classes):
            if c in clf.classes_:
                fold_probs[:, ci] = probs[:, list(clf.classes_).index(c)]
        all_probs[te] = fold_probs
        all_pred[te] = clf.predict(X_te)

    # Per-class one-vs-rest ROC AUC
    per_class_auc = {}
    for i, c in enumerate(classes):
        y_bin = (y == c).astype(int)
        if y_bin.sum() > 0 and y_bin.sum() < len(y):
            per_class_auc[c] = float(roc_auc_score(y_bin, all_probs[:, i]))
        else:
            per_class_auc[c] = float("nan")
    macro_auc = float(np.nanmean(list(per_class_auc.values())))
    macro_f1 = float(f1_score(y, all_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y, all_pred))
    cm = confusion_matrix(y, all_pred, labels=classes).tolist()

    print(f"\n  [{label}] per-class OVR AUC:")
    for c, a in per_class_auc.items():
        print(f"    {c:<12s} AUC={a:.4f}")
    print(f"    macro AUC={macro_auc:.4f}  macro F1={macro_f1:.4f}  acc={acc:.4f}")

    return {
        "per_class_auc": per_class_auc,
        "macro_auc": macro_auc,
        "macro_f1": macro_f1,
        "accuracy": acc,
        "confusion_matrix": cm,
        "classes": classes,
        "cv_unit": "protein" if groups is not None else "variant",
        "n_groups": int(len(set(groups.tolist()))) if groups is not None else None,
        "probs": all_probs,
        "pred": all_pred.tolist(),
    }


def bootstrap_auc_diff(y_true, probs_a, probs_b, classes, n_boot=1000):
    """Permutation bootstrap: is macro-AUC of A significantly > B?"""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)

    def macro_auc(y_local, probs):
        vals = []
        for i, c in enumerate(classes):
            y_bin = (y_local == c).astype(int)
            if 0 < y_bin.sum() < len(y_bin):
                vals.append(roc_auc_score(y_bin, probs[:, i]))
        return float(np.mean(vals))

    obs = macro_auc(y_true, probs_a) - macro_auc(y_true, probs_b)
    boot = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_boot = y_true[idx]
        boot.append(macro_auc(y_boot, probs_a[idx]) - macro_auc(y_boot, probs_b[idx]))
    boot = np.array(boot)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    p = float((boot <= 0).mean())
    return {"obs_diff": obs, "bootstrap_ci95": ci, "one_sided_p": p}


def top_features_per_class(X, y, classes, col_meta, meta_by_layer, k=20):
    """Train a single LR and extract highest-weight features per class."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)
    clf = LogisticRegression(
        C=0.1, solver="lbfgs",
        max_iter=3000, class_weight="balanced",
    )
    clf.fit(Xn, y)

    out = {}
    for i, c in enumerate(classes):
        idx = list(clf.classes_).index(c)
        coefs = clf.coef_[idx]
        top = np.argsort(-coefs)[:k]
        rows = []
        for col in top:
            layer, feat_idx, kind = col_meta[col]
            ann = meta_by_layer.get(layer, {}).get("ann", [""])[feat_idx] if feat_idx < len(meta_by_layer.get(layer, {}).get("ann", [])) else ""
            rows.append({
                "layer": layer, "feature": feat_idx,
                "kind": kind, "coef": float(coefs[col]),
                "annotation": ann,
            })
        out[c] = rows
    return out


def main():
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mechanisms",
        default=f"{OUT_DIR}/variant_mechanisms.tsv",
    )
    ap.add_argument("--out", default=f"{OUT_DIR}/mechanism_classifier_results.json")
    ap.add_argument("--min-per-class", type=int, default=15)
    args = ap.parse_args()

    print("=" * 70)
    print("  R1 Mechanism Classifier — LOF vs GOF vs DN vs Neomorphic")
    print("=" * 70)

    print("\n[1/5] Loading annotation metadata...")
    meta_by_layer = load_annotation_metadata()

    print(f"\n[2/5] Reading mechanism labels from {args.mechanisms}...")
    mech_labels = load_mechanism_labels(args.mechanisms)
    class_counts = Counter(mech_labels.values())
    print(f"  Raw class counts: {dict(class_counts)}")

    keep_classes = [c for c, n in class_counts.items()
                    if n >= args.min_per_class and c != "UNLABELED"]
    keep_classes = sorted(keep_classes)
    print(f"  Classes retained (≥{args.min_per_class} examples): {keep_classes}")

    print(f"\n[3/5] Building feature matrices...")
    X_sae, llr, y, groups, col_meta = build_features(meta_by_layer, mech_labels, keep_classes)
    print(f"  X_sae={X_sae.shape}  LLR={llr.shape}  y={Counter(y)}  proteins={len(set(groups.tolist()))}")

    if X_sae.shape[0] < 50:
        print("  Too few labeled variants to fit multinomial LR (need >= 50).")
        print("  Run 15_build_mechanism_dataset.py with a mechanism TSV first.")
        return

    # Drop constant columns
    good = X_sae.std(0) > 1e-8
    X_sae = X_sae[:, good]
    col_meta = [col_meta[i] for i, g in enumerate(good) if g]
    print(f"  After constant removal: X_sae={X_sae.shape}")

    print(f"\n[4/5] Evaluating three classifiers with variant-level 5-fold CV...")
    res_sae = evaluate_multiclass(X_sae, y, keep_classes, label="SAE only")
    res_llr = evaluate_multiclass(llr.reshape(-1, 1), y, keep_classes, label="LLR only")
    res_comb = evaluate_multiclass(
        np.concatenate([X_sae, llr.reshape(-1, 1)], axis=1),
        y, keep_classes, label="SAE + LLR",
    )

    boot = bootstrap_auc_diff(y, res_sae["probs"], res_llr["probs"], keep_classes)
    print(f"\n  Bootstrap test (SAE vs LLR macro-AUC):")
    print(f"    Δ macro-AUC = {boot['obs_diff']:+.4f} "
          f"(95% CI [{boot['bootstrap_ci95'][0]:+.4f}, "
          f"{boot['bootstrap_ci95'][1]:+.4f}], p={boot['one_sided_p']:.4f})")

    print(f"\n[4b/5] Evaluating three classifiers with protein-level 5-fold CV...")
    res_sae_prot = evaluate_multiclass(
        X_sae, y, keep_classes, label="SAE only / protein CV", groups=groups,
    )
    res_llr_prot = evaluate_multiclass(
        llr.reshape(-1, 1), y, keep_classes, label="LLR only / protein CV",
        groups=groups,
    )
    res_comb_prot = evaluate_multiclass(
        np.concatenate([X_sae, llr.reshape(-1, 1)], axis=1),
        y, keep_classes, label="SAE + LLR / protein CV", groups=groups,
    )
    boot_prot = bootstrap_auc_diff(
        y, res_sae_prot["probs"], res_llr_prot["probs"], keep_classes
    )
    print(f"\n  Protein-level bootstrap test (SAE vs LLR macro-AUC):")
    print(f"    Δ macro-AUC = {boot_prot['obs_diff']:+.4f} "
          f"(95% CI [{boot_prot['bootstrap_ci95'][0]:+.4f}, "
          f"{boot_prot['bootstrap_ci95'][1]:+.4f}], p={boot_prot['one_sided_p']:.4f})")

    print(f"\n[5/5] Top SAE features per mechanism class...")
    top_feats = top_features_per_class(
        X_sae, y, keep_classes, col_meta, meta_by_layer, k=10,
    )
    for c in keep_classes:
        print(f"\n  === {c} top features ===")
        for r in top_feats[c][:5]:
            print(f"    L{r['layer']} F{r['feature']:>5d} coef={r['coef']:+.3f} "
                  f"[{r['kind']}] {r['annotation']}")

    summary = {
        "classes": keep_classes,
        "n_per_class": {c: int((y == c).sum()) for c in keep_classes},
        "n_proteins": int(len(set(groups.tolist()))),
        "sae_only": {k: v for k, v in res_sae.items() if k not in {"probs", "pred"}},
        "llr_only": {k: v for k, v in res_llr.items() if k not in {"probs", "pred"}},
        "sae_plus_llr": {k: v for k, v in res_comb.items() if k not in {"probs", "pred"}},
        "bootstrap_sae_vs_llr": boot,
        "variant_level_cv": {
            "sae_only": {k: v for k, v in res_sae.items() if k not in {"probs", "pred"}},
            "llr_only": {k: v for k, v in res_llr.items() if k not in {"probs", "pred"}},
            "sae_plus_llr": {k: v for k, v in res_comb.items() if k not in {"probs", "pred"}},
            "bootstrap_sae_vs_llr": boot,
        },
        "protein_level_cv": {
            "sae_only": {k: v for k, v in res_sae_prot.items() if k not in {"probs", "pred"}},
            "llr_only": {k: v for k, v in res_llr_prot.items() if k not in {"probs", "pred"}},
            "sae_plus_llr": {k: v for k, v in res_comb_prot.items() if k not in {"probs", "pred"}},
            "bootstrap_sae_vs_llr": boot_prot,
        },
        "top_features_per_class": top_feats,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"  Saved: {args.out}")
    print(f"  Summary:")
    print(f"    SAE only   macro-AUC = {res_sae['macro_auc']:.4f}")
    print(f"    LLR only   macro-AUC = {res_llr['macro_auc']:.4f}")
    print(f"    SAE + LLR  macro-AUC = {res_comb['macro_auc']:.4f}")
    print(f"    SAE only   protein-CV macro-AUC = {res_sae_prot['macro_auc']:.4f}")
    print(f"    LLR only   protein-CV macro-AUC = {res_llr_prot['macro_auc']:.4f}")
    print(f"    SAE + LLR  protein-CV macro-AUC = {res_comb_prot['macro_auc']:.4f}")


if __name__ == "__main__":
    main()
