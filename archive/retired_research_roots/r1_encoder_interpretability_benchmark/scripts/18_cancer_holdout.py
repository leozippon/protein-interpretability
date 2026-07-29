#!/usr/bin/env python
"""Held-out cancer validation for R1.

The original R1-E TODO called for training on ClinVar and testing on an
independent cancer variant set such as COSMIC. In the current repo, no COSMIC
table is staged, so this script implements the strongest available default:

  1. Build pathogenic-vs-benign features from the existing R1 perturbation
     signatures and ESM-2 LLR scores.
  2. Train on non-cancer ClinVar missense variants.
  3. Test on held-out ClinVar variants from curated cancer genes
     (oncogenes / tumor suppressors).

If an external cancer table is staged later, this script can be extended to
score that cohort; the current implementation makes the data gap explicit in
the output JSON.

Usage:
    python r1_encoder_interpretability_benchmark/scripts/18_cancer_holdout.py \
        --out r1_encoder_interpretability_benchmark/results/variant_effect/cancer_holdout.json
"""

import argparse
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


LAYERS = [19, 23, 27, 31, 35]
OUT_DIR = "r1_encoder_interpretability_benchmark/results/variant_effect"

ONCOGENES = {
    "AKT1", "ALK", "BRAF", "EGFR", "ERBB2", "FGFR3", "FLT3", "HRAS",
    "IDH1", "IDH2", "JAK2", "KIT", "KRAS", "MET", "MYC", "NRAS",
    "PIK3CA", "RET",
}
TUMOR_SUPPRESSORS = {
    "APC", "ARID1A", "ATM", "BAP1", "BRCA1", "BRCA2", "CDKN2A", "CHEK2",
    "FBXW7", "MLH1", "MSH2", "MSH6", "NF1", "NF2", "PALB2", "PMS2",
    "PTEN", "RB1", "RUNX1", "SMAD4", "SMARCB1", "STK11", "TP53",
    "TSC1", "TSC2", "VHL",
}


def clinvar_binary_label(clin_sig: str) -> int | None:
    sig = (clin_sig or "").lower()
    is_path = ("pathogenic" in sig and "benign" not in sig and
               "conflicting" not in sig)
    is_benign = ("benign" in sig and "pathogenic" not in sig)
    if is_path:
        return 1
    if is_benign:
        return 0
    return None


def cancer_role(gene: str) -> str:
    gene = gene.upper()
    if gene in ONCOGENES:
        return "oncogene"
    if gene in TUMOR_SUPPRESSORS:
        return "tumor_suppressor"
    return "other"


def load_llr_lookup(path: str) -> dict[tuple[str, str], float]:
    with open(path) as f:
        rows = json.load(f)
    return {(r["gene"].upper(), r["variant"]): float(r["llr"]) for r in rows}


def summarize_signature(sig: dict) -> list[float]:
    delta_local = np.asarray(sig["delta_local"], dtype=np.float32)
    delta_global = np.asarray(sig["delta_global"], dtype=np.float32)
    wt_count = float(sig.get("wt_active_count", 0))
    mut_count = float(sig.get("mut_active_count", 0))
    return [
        float(sig.get("n_ablated", 0)),
        float(sig.get("n_amplified", 0)),
        float(sig.get("n_novel", 0)),
        float(sig.get("total_perturbation", float(np.abs(delta_local).sum()))),
        wt_count,
        mut_count,
        float(mut_count / max(wt_count, 1.0)),
        float(np.abs(delta_local).mean()) if delta_local.size else 0.0,
        float(np.abs(delta_global).mean()) if delta_global.size else 0.0,
        float(np.abs(delta_local).max()) if delta_local.size else 0.0,
    ]


def build_dataset(signatures_path: str, llr_path: str) -> list[dict]:
    with open(signatures_path, "rb") as f:
        signatures = pickle.load(f)
    llr_lookup = load_llr_lookup(llr_path)

    rows = []
    for _, var_sigs in signatures.items():
        s0 = var_sigs[0]
        label = clinvar_binary_label(s0.get("clinical_significance", ""))
        if label is None:
            continue

        sig_by_layer = {int(s["layer"]): s for s in var_sigs}
        summary_feats = []
        for layer in LAYERS:
            if layer in sig_by_layer:
                summary_feats.extend(summarize_signature(sig_by_layer[layer]))
            else:
                summary_feats.extend([0.0] * 10)

        gene = s0["gene"].upper()
        variant = s0["variant_str"]
        rows.append({
            "gene": gene,
            "variant": variant,
            "clinical_significance": s0.get("clinical_significance", ""),
            "label": int(label),
            "role": cancer_role(gene),
            "sae_summary": summary_feats,
            "llr": -float(llr_lookup.get((gene, variant), 0.0)),
        })
    return rows


def compute_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def evaluate_split(train_rows: list[dict], test_rows: list[dict]) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    if not train_rows:
        raise ValueError("train set is empty")
    if not test_rows:
        raise ValueError("test set is empty")

    X_train = np.array([r["sae_summary"] for r in train_rows], dtype=np.float32)
    X_test = np.array([r["sae_summary"] for r in test_rows], dtype=np.float32)
    llr_train = np.array([r["llr"] for r in train_rows], dtype=np.float32)
    llr_test = np.array([r["llr"] for r in test_rows], dtype=np.float32)
    y_train = np.array([r["label"] for r in train_rows], dtype=np.int32)
    y_test = np.array([r["label"] for r in test_rows], dtype=np.int32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sae_lr = LogisticRegression(
        C=0.1,
        solver="lbfgs",
        max_iter=3000,
        class_weight="balanced",
    )
    sae_lr.fit(X_train, y_train)
    sae_scores = sae_lr.predict_proba(X_test)[:, 1]

    comb_lr = LogisticRegression(
        C=0.1,
        solver="lbfgs",
        max_iter=3000,
        class_weight="balanced",
    )
    Xc_train = np.concatenate([X_train, llr_train[:, None]], axis=1)
    Xc_test = np.concatenate([X_test, llr_test[:, None]], axis=1)
    comb_lr.fit(Xc_train, y_train)
    comb_scores = comb_lr.predict_proba(Xc_test)[:, 1]

    by_role = {}
    for role in ["oncogene", "tumor_suppressor"]:
        idx = [i for i, r in enumerate(test_rows) if r["role"] == role]
        if not idx:
            continue
        sub_y = y_test[idx]
        by_role[role] = {
            "n": len(idx),
            "n_pathogenic": int(sub_y.sum()),
            "n_benign": int((1 - sub_y).sum()),
            "auc_sae": compute_auc(sub_y, sae_scores[idx]),
            "auc_llr": compute_auc(sub_y, llr_test[idx]),
            "auc_sae_plus_llr": compute_auc(sub_y, comb_scores[idx]),
        }

    return {
        "train": {
            "n": len(train_rows),
            "n_pathogenic": int(y_train.sum()),
            "n_benign": int((1 - y_train).sum()),
        },
        "test": {
            "n": len(test_rows),
            "n_pathogenic": int(y_test.sum()),
            "n_benign": int((1 - y_test).sum()),
        },
        "overall": {
            "auc_sae": compute_auc(y_test, sae_scores),
            "auc_llr": compute_auc(y_test, llr_test),
            "auc_sae_plus_llr": compute_auc(y_test, comb_scores),
            "acc_sae_plus_llr": float(
                accuracy_score(y_test, (comb_scores >= 0.5).astype(np.int32))
            ),
        },
        "by_role": by_role,
        "predictions": [
            {
                "gene": r["gene"],
                "variant": r["variant"],
                "role": r["role"],
                "label": r["label"],
                "score_sae": float(sae_scores[i]),
                "score_llr": float(llr_test[i]),
                "score_sae_plus_llr": float(comb_scores[i]),
            }
            for i, r in enumerate(test_rows)
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--signatures",
        default=f"{OUT_DIR}/scaled_perturbation_signatures.pkl",
    )
    ap.add_argument(
        "--llr-json",
        default=f"{OUT_DIR}/esm2_per_variant_llr.json",
    )
    ap.add_argument(
        "--out",
        default=f"{OUT_DIR}/cancer_holdout.json",
    )
    args = ap.parse_args()

    print("=" * 70)
    print("  R1-E Held-out cancer validation")
    print("=" * 70)

    rows = build_dataset(args.signatures, args.llr_json)
    role_counts = Counter(r["role"] for r in rows)
    print(f"  Total labeled variants: {len(rows)}")
    print(f"  Role counts: {dict(role_counts)}")

    train_rows = [r for r in rows if r["role"] == "other"]
    test_rows = [r for r in rows if r["role"] != "other"]
    result = evaluate_split(train_rows, test_rows)
    result["config"] = {
        "layers": LAYERS,
        "holdout_mode": "clinvar_cancer_gene_holdout",
        "external_cancer_dataset_present": False,
        "note": (
            "No COSMIC/external cancer table was staged locally; using held-out "
            "ClinVar variants from curated cancer genes instead."
        ),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Train n={result['train']['n']}  Test n={result['test']['n']}")
    print(
        "  Overall AUCs: "
        f"SAE={result['overall']['auc_sae']:.4f}  "
        f"LLR={result['overall']['auc_llr']:.4f}  "
        f"SAE+LLR={result['overall']['auc_sae_plus_llr']:.4f}"
    )
    for role, metrics in result["by_role"].items():
        print(
            f"  {role}: n={metrics['n']}  "
            f"AUC(SAE+LLR)={metrics['auc_sae_plus_llr']:.4f}"
        )
    print(f"\n  Saved: {args.out}")


if __name__ == "__main__":
    main()
