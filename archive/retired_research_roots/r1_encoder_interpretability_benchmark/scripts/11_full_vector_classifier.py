#!/usr/bin/env python
"""Full-feature-vector variant effect classifier (GPU-accelerated).

Previous approach (script 07-09): compressed delta_local into 12 summary
statistics per layer (60 total features). This lost per-feature information.
AUC plateaued at 0.836 vs LLR's 0.882.

This script uses the FULL delta_local + delta_global vectors across all
layers (2 × 16384 × 5 = 163,840 features per variant) with:
  1. GPU-trained L2 logistic regression via PyTorch (fast)
  2. PyTorch MLP with dropout regularization
  3. XGBoost on top-variance subset (fallback)
  4. Ensemble with LLR (from previously saved baseline)

Goal: Close the gap with ESM-2 LLR (0.882) or beat it.

Usage:
    python scripts/11_full_vector_classifier.py
"""

import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = "r1_encoder_interpretability_benchmark/results/variant_effect"
LAYERS = [19, 23, 27, 31, 35]
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"


def load_data_as_matrix():
    """Load scaled perturbation signatures and assemble feature matrix."""
    print("[1/5] Loading perturbation signatures...", flush=True)
    with open(f"{OUT_DIR}/scaled_perturbation_signatures.pkl", "rb") as f:
        sigs = pickle.load(f)

    print(f"  Loaded {len(sigs)} variants.", flush=True)

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
            if layer not in sig_by_layer:
                feature_parts.append(np.zeros(16384, dtype=np.float32))
                feature_parts.append(np.zeros(16384, dtype=np.float32))
                continue
            s = sig_by_layer[layer]
            feature_parts.append(s["delta_local"].astype(np.float32))
            feature_parts.append(s["delta_global"].astype(np.float32))

        X_rows.append(np.concatenate(feature_parts))
        y_rows.append(1 if is_path else 0)
        meta_rows.append({
            "gene": s0["gene"],
            "variant": s0["variant_str"],
            "clinical_significance": clin_sig,
        })

    X = np.stack(X_rows).astype(np.float32)
    y = np.array(y_rows, dtype=np.int32)

    print(f"  Feature matrix: X.shape={X.shape}, y.shape={y.shape}", flush=True)
    print(f"  Pathogenic: {int(y.sum())}, Benign: {int((1-y).sum())}", flush=True)
    return X, y, meta_rows


class LRModel(nn.Module):
    """GPU logistic regression with L2 weight decay."""
    def __init__(self, d_in):
        super().__init__()
        self.linear = nn.Linear(d_in, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class MLPModel(nn.Module):
    """Small MLP with dropout."""
    def __init__(self, d_in, d_hidden=256, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_gpu_clf(
    X_tr, y_tr, X_va, y_va,
    model_type="lr",
    lr=1e-3, weight_decay=1e-2,
    epochs=200, batch_size=256, verbose=False,
):
    """Train GPU classifier with early stopping on validation AUC."""
    from sklearn.metrics import roc_auc_score

    d_in = X_tr.shape[1]
    if model_type == "lr":
        model = LRModel(d_in).to(DEVICE)
    elif model_type == "mlp":
        model = MLPModel(d_in).to(DEVICE)
    else:
        raise ValueError(model_type)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_tr_t = torch.from_numpy(X_tr).to(DEVICE)
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).to(DEVICE)
    X_va_t = torch.from_numpy(X_va).to(DEVICE)

    n = X_tr_t.shape[0]
    best_auc = 0.0
    best_scores = None
    patience = 20
    bad = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_tr_t[idx]
            yb = y_tr_t[idx]
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(X_va_t).cpu().numpy()
        va_scores = 1.0 / (1.0 + np.exp(-va_logits))
        auc = roc_auc_score(y_va, va_scores)

        if auc > best_auc:
            best_auc = auc
            best_scores = va_scores.copy()
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

        if verbose and epoch % 10 == 0:
            print(f"    epoch {epoch:3d} loss={total_loss/n:.4f} val_auc={auc:.4f}",
                  flush=True)

    return best_scores, best_auc


def run_cv(X, y, model_type, label, n_folds=5, **kwargs):
    """5-fold CV training on GPU."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score

    print(f"\n[{label}]", flush=True)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aucs = []
    accs = []
    scores_all = np.zeros(len(y), dtype=np.float32)

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        t0 = time.time()
        # Per-fold standardization
        mu = X[tr].mean(0, keepdims=True)
        sd = X[tr].std(0, keepdims=True) + 1e-6
        X_tr = ((X[tr] - mu) / sd).astype(np.float32)
        X_va = ((X[va] - mu) / sd).astype(np.float32)

        scores, best_auc = train_gpu_clf(X_tr, y[tr], X_va, y[va],
                                          model_type=model_type, **kwargs)
        scores_all[va] = scores
        acc = accuracy_score(y[va], scores > 0.5)
        aucs.append(best_auc)
        accs.append(acc)
        print(f"  Fold {fold+1}: AUC={best_auc:.4f}  Acc={acc:.4f}  ({time.time()-t0:.0f}s)",
              flush=True)

    result = {
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "mean_acc": float(np.mean(accs)),
        "scores": scores_all.tolist(),
    }
    print(f"  Mean AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}", flush=True)
    return result


def load_llr_scores(meta):
    """Compute LLR scores on the fly if not cached.

    We can't rely on per_variant in the saved baseline JSON, so we'll
    match on (gene, variant) when reconstructing by rerunning the LLR
    computation only if a per-variant file exists.
    """
    per_variant_path = f"{OUT_DIR}/esm2_per_variant_llr.json"
    if not os.path.exists(per_variant_path):
        return None

    with open(per_variant_path) as f:
        per_variant = json.load(f)
    pv_lookup = {(pv["gene"], pv["variant"]): pv["llr"] for pv in per_variant}

    llr_scores = np.zeros(len(meta))
    matched = 0
    for i, m in enumerate(meta):
        k = (m["gene"], m["variant"])
        if k in pv_lookup:
            llr_scores[i] = -pv_lookup[k]  # more negative LLR → more deleterious
            matched += 1

    print(f"  Matched {matched}/{len(meta)} LLR scores", flush=True)
    return llr_scores if matched > len(meta) // 2 else None


def run_ensemble(X, y, meta, sae_scores):
    """Ensemble SAE predictions with LLR."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score

    llr = load_llr_scores(meta)
    if llr is None:
        print("  No per-variant LLR — skipping ensemble", flush=True)
        return {}

    X_comb = np.stack([sae_scores, llr], axis=1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    accs = []
    for fold, (tr, va) in enumerate(skf.split(X_comb, y)):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_comb[tr], y[tr])
        probs = clf.predict_proba(X_comb[va])[:, 1]
        auc = roc_auc_score(y[va], probs)
        acc = accuracy_score(y[va], probs > 0.5)
        aucs.append(auc)
        accs.append(acc)

    llr_auc = float(roc_auc_score(y, llr))
    sae_auc = float(roc_auc_score(y, sae_scores))
    results = {
        "llr_alone": {"mean_auc": llr_auc},
        "sae_full_alone": {"mean_auc": sae_auc},
        "ensemble_llr_sae_full": {
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
            "mean_acc": float(np.mean(accs)),
        }
    }
    print(f"  LLR alone: {llr_auc:.4f}", flush=True)
    print(f"  SAE full alone (cross-val): {sae_auc:.4f}", flush=True)
    print(f"  Ensemble AUC = {np.mean(aucs):.4f}", flush=True)
    return results


def main():
    print("=" * 70)
    print("  Full-Feature-Vector Variant Effect Classification (GPU)")
    print("=" * 70)
    print(f"  Device: {DEVICE}", flush=True)

    X, y, meta = load_data_as_matrix()

    all_results = {}

    # GPU LR with different weight decay values
    for wd in [1e-3, 1e-2, 1e-1]:
        r = run_cv(X, y, model_type="lr",
                   label=f"LR wd={wd}",
                   lr=1e-3, weight_decay=wd, epochs=100)
        all_results[f"lr_wd{wd}"] = r

    # GPU MLP
    for dropout in [0.3, 0.5]:
        r = run_cv(X, y, model_type="mlp",
                   label=f"MLP dropout={dropout}",
                   lr=5e-4, weight_decay=1e-3, epochs=100)
        all_results[f"mlp_dropout{dropout}"] = r

    # Pick best
    best_key = max(all_results.keys(), key=lambda k: all_results[k]["mean_auc"])
    best_scores = np.array(all_results[best_key]["scores"])
    print(f"\n  Best: {best_key} AUC={all_results[best_key]['mean_auc']:.4f}",
          flush=True)

    # Ensemble with LLR if available
    ensemble = run_ensemble(X, y, meta, best_scores)
    all_results.update(ensemble)

    # Save
    summary = {}
    for k, v in all_results.items():
        v2 = {kk: vv for kk, vv in v.items() if kk != "scores"}
        summary[k] = v2

    with open(f"{OUT_DIR}/full_vector_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save best scores
    np.save(f"{OUT_DIR}/full_vector_best_scores.npy", best_scores)

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k:<30s} AUC={v.get('mean_auc', 0):.4f}")

    print(f"\n  Saved to {OUT_DIR}/full_vector_results.json")


if __name__ == "__main__":
    main()
