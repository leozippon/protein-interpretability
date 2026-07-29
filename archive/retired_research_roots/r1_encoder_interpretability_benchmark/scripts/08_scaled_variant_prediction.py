#!/usr/bin/env python
"""Scaled variant prediction with all 5 layers including layer 19.

Extends EXP-007/EXP-008:
  - Adds layer 19 (best annotated: 146 KNOWN features)
  - Increases variant count to 2000 (1000 pathogenic + 1000 benign)
  - Computes both raw perturbation signatures and annotation-weighted scores
  - Runs cross-layer logistic regression with 5-fold CV

Usage:
    python scripts/08_scaled_variant_prediction.py --gpu 2 --num-variants 2000
"""

import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, EsmModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.feature_annotation import load_our_sae
from src.analysis.variant_effect import (
    compute_perturbation_signature,
    create_mutant_sequence,
    MissenseVariant,
)


CHECKPOINT_DIRS = {
    19: "results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights/layer_19/step_500000",
    23: "results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights/layer_23/step_500000",
    27: "results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights/layer_27/step_500000",
    31: "results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights/layer_31/step_500000",
    35: "results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights/layer_35/step_500000",
}

CATEGORY_WEIGHTS = {
    "functional": 5.0,
    "ptm": 4.0,
    "domain": 3.0,
    "region": 2.0,
    "topology": 1.5,
    "secondary_structure": 1.0,
    "chain": 0.5,
}


def load_annotation_weights(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    results = data['results']
    d_sae = len(results)
    f1w = np.zeros(d_sae, dtype=np.float32)
    cw = np.zeros(d_sae, dtype=np.float32)
    cats = [""] * d_sae
    for r in results:
        if not r.alive:
            continue
        f1w[r.feature_idx] = r.best_f1
        cat = r.best_annotation.split('/')[0] if r.best_annotation and '/' in r.best_annotation else (r.best_annotation or "")
        cats[r.feature_idx] = cat
        cw[r.feature_idx] = r.best_f1 * CATEGORY_WEIGHTS.get(cat, 1.0)
    return f1w, cw, cats


def load_variants_balanced(clinvar_path, seq_map, idmapping_path, n_each=1000):
    """Load balanced pathogenic/benign variants from ClinVar."""
    import gzip, re
    gene_to_uniprot = {}
    if idmapping_path and os.path.exists(idmapping_path):
        opener = gzip.open if idmapping_path.endswith('.gz') else open
        with opener(idmapping_path, 'rt') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3 and parts[1] == 'Gene_Name':
                    if parts[0] in seq_map:
                        gene_to_uniprot[parts[2]] = parts[0]
        print(f"  Gene-UniProt map: {len(gene_to_uniprot)} entries")

    hgvs_pattern = re.compile(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})')
    aa3to1 = {
        'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
        'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
        'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
        'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    }

    pathogenic, benign = [], []
    seen = set()

    opener = gzip.open if clinvar_path.endswith('.gz') else open
    with opener(clinvar_path, 'rt') as f:
        f.readline()  # skip header
        for line in f:
            if len(pathogenic) >= n_each * 5 and len(benign) >= n_each * 5:
                break
            fields = line.strip().split('\t')
            if len(fields) < 8:
                continue
            if fields[1] != 'single nucleotide variant':
                continue

            name, gene, clin_sig = fields[2], fields[4], fields[6]
            sig_lower = clin_sig.lower()
            match = hgvs_pattern.search(name)
            if not match:
                continue
            wt_aa3, pos_str, mut_aa3 = match.groups()
            wt_aa = aa3to1.get(wt_aa3)
            mut_aa = aa3to1.get(mut_aa3)
            if not wt_aa or not mut_aa:
                continue

            pos = int(pos_str)
            uniprot_id = gene_to_uniprot.get(gene, "")
            if not uniprot_id:
                continue

            seq = seq_map.get(uniprot_id, "")
            if not seq or pos > len(seq) or seq[pos - 1] != wt_aa:
                continue
            if len(seq) > 1022:
                continue  # ESM-2 limit

            key = (uniprot_id, pos, wt_aa, mut_aa)
            if key in seen:
                continue
            seen.add(key)

            v = MissenseVariant(
                gene=gene, uniprot_id=uniprot_id, position=pos,
                wt_residue=wt_aa, mut_residue=mut_aa,
                clinical_significance=clin_sig, source="clinvar",
            )

            is_path = 'pathogenic' in sig_lower and 'benign' not in sig_lower and 'conflicting' not in sig_lower
            is_benign = 'benign' in sig_lower and 'pathogenic' not in sig_lower

            if is_path:
                pathogenic.append(v)
            elif is_benign:
                benign.append(v)

    print(f"  Found {len(pathogenic)} pathogenic, {len(benign)} benign (pre-balance)")

    rng = np.random.RandomState(42)
    actual_n = min(len(pathogenic), len(benign), n_each)
    if len(pathogenic) > actual_n:
        idx = rng.choice(len(pathogenic), actual_n, replace=False)
        pathogenic = [pathogenic[i] for i in idx]
    if len(benign) > actual_n:
        idx = rng.choice(len(benign), actual_n, replace=False)
        benign = [benign[i] for i in idx]

    variants = pathogenic + benign
    rng.shuffle(variants)
    return variants, actual_n


def extract_features(sigs, f1w_by_layer, cw_by_layer, cats_by_layer, layers):
    """Extract cross-layer feature vector for LR classifier."""
    sig_by_layer = {s['layer']: s for s in sigs}
    features = []
    for layer in layers:
        if layer not in sig_by_layer:
            features.extend([0.0] * 12)
            continue
        s = sig_by_layer[layer]
        delta = np.abs(s['delta_local'])
        f1w = f1w_by_layer.get(layer, np.zeros_like(delta))
        cw = cw_by_layer.get(layer, np.zeros_like(delta))
        known = f1w > 0.2
        novel = f1w < 0.05
        perturbed = delta > 0.1

        raw_pert = float(delta.sum())
        f1_pert = float((f1w * delta).sum())
        cat_pert = float((cw * delta).sum())
        n_known_p = int((known & perturbed).sum())
        n_novel_p = int((novel & perturbed).sum())
        max_f1_p = float(f1w[perturbed].max()) if perturbed.any() else 0.0
        mean_dk = float(delta[known].mean()) if known.any() else 0.0
        mean_dn = float(delta[novel].mean()) if novel.any() else 0.0
        top10 = np.argsort(f1w * delta)[-10:]
        top10_pert = float((f1w[top10] * delta[top10]).sum())
        known_frac = f1_pert / max(raw_pert, 1e-6)
        ds = s['delta_local']
        gain = float(ds[ds > 0].sum())
        loss = float(abs(ds[ds < 0].sum()))
        asym = (gain - loss) / max(gain + loss, 1e-6)

        features.extend([
            raw_pert, f1_pert, cat_pert,
            n_known_p, n_novel_p, max_f1_p,
            mean_dk, mean_dn, top10_pert, known_frac,
            asym, float(s.get('total_perturbation', raw_pert)),
        ])
    return np.array(features, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--num-variants", type=int, default=2000)
    parser.add_argument("--layers", type=str, default="19,23,27,31,35")
    parser.add_argument("--esm-model", type=str,
                        default="/Data/public/esm2_t36_3B_UR50D")
    parser.add_argument("--swissprot-cache", type=str,
                        default="../data/processed/swissprot_all_max1022.pkl")
    parser.add_argument("--clinvar-path", type=str,
                        default="../data/clinvar/variant_summary.txt.gz")
    parser.add_argument("--idmapping-path", type=str,
                        default="../data/swissprot/HUMAN_9606_idmapping.dat.gz")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"
    layers = [int(x) for x in args.layers.split(",")]

    print("=" * 70)
    print("  Scaled Variant Prediction (5 layers, larger sample)")
    print("=" * 70)
    print(f"  GPU: {args.gpu}, Layers: {layers}")
    print(f"  Target variants: {args.num_variants}")

    # --- Load annotation weights ---
    print("\n[1/5] Loading annotation weights...")
    f1w_by_layer, cw_by_layer, cats_by_layer = {}, {}, {}
    for layer in layers:
        pkl = f"results/annotation_alignment/ours_3B_l{layer}_step500000.pkl"
        if os.path.exists(pkl):
            f1w, cw, cats = load_annotation_weights(pkl)
            f1w_by_layer[layer] = f1w
            cw_by_layer[layer] = cw
            cats_by_layer[layer] = cats
            n_known = int((f1w > 0.5).sum())
            n_partial = int(((f1w > 0.2) & (f1w <= 0.5)).sum())
            print(f"  Layer {layer}: {n_known} KNOWN, {n_partial} PARTIAL")

    # --- Load sequences ---
    print("\n[2/5] Loading protein sequences...")
    t0 = time.time()
    with open(args.swissprot_cache, 'rb') as f:
        annotations = pickle.load(f)
    seq_map = {ann.accession: ann.sequence for ann in annotations
               if 'Homo sapiens' in ann.organism}
    print(f"  {len(seq_map)} human proteins ({time.time()-t0:.1f}s)")

    # --- Load variants ---
    print("\n[3/5] Loading ClinVar variants...")
    n_each = args.num_variants // 2
    variants, actual_n = load_variants_balanced(
        args.clinvar_path, seq_map, args.idmapping_path, n_each)
    print(f"  Balanced: {actual_n} pathogenic + {actual_n} benign = {len(variants)}")

    # --- Load ESM-2 + compute perturbation signatures ---
    print("\n[4/5] Loading ESM-2-3B and computing perturbation signatures...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
    esm_model = EsmModel.from_pretrained(args.esm_model, torch_dtype=torch.float16)
    esm_model.to(device).eval()
    print(f"  ESM-2-3B loaded ({time.time()-t0:.1f}s)")

    all_sigs = {}  # key -> [sig_dict, ...]

    for layer in layers:
        ckpt = CHECKPOINT_DIRS.get(layer)
        if not ckpt or not os.path.exists(os.path.join(ckpt, "sae.pt")):
            print(f"  Layer {layer}: checkpoint not found, skipping")
            continue

        print(f"\n  --- Layer {layer} ---")
        t_layer = time.time()
        sae = load_our_sae(ckpt, device=device)
        n_ok, n_fail = 0, 0

        for vi, var in enumerate(variants):
            wt_seq = seq_map.get(var.uniprot_id, "")
            if not wt_seq:
                continue
            mut_seq = create_mutant_sequence(wt_seq, var.position, var.wt_residue, var.mut_residue)
            if not mut_seq:
                n_fail += 1
                continue

            try:
                sig = compute_perturbation_signature(
                    wt_seq, mut_seq, var, esm_model, tokenizer, sae,
                    layer, device=device)
                key = (var.uniprot_id, var.position, var.wt_residue, var.mut_residue)
                all_sigs.setdefault(key, []).append({
                    "layer": sig.layer,
                    "n_ablated": sig.n_ablated,
                    "n_amplified": sig.n_amplified,
                    "n_novel": sig.n_novel,
                    "total_perturbation": sig.total_perturbation,
                    "wt_active_count": sig.wt_active_count,
                    "mut_active_count": sig.mut_active_count,
                    "delta_local": sig.delta_local,
                    "delta_global": sig.delta_global,
                    "clinical_significance": var.clinical_significance,
                    "gene": var.gene,
                    "variant_str": f"{var.wt_residue}{var.position}{var.mut_residue}",
                })
                n_ok += 1
            except Exception as e:
                n_fail += 1
                if n_fail <= 3:
                    print(f"    Failed: {var.gene} {var.wt_residue}{var.position}{var.mut_residue}: {e}")

            if (vi + 1) % 100 == 0:
                elapsed = time.time() - t_layer
                rate = n_ok / max(elapsed, 1)
                print(f"    {vi+1}/{len(variants)}: {n_ok} ok, {n_fail} fail ({rate:.1f} var/s)")

        print(f"  Layer {layer}: {n_ok} ok, {n_fail} fail ({time.time()-t_layer:.1f}s)")
        del sae
        torch.cuda.empty_cache()

    # --- Analysis ---
    print(f"\n[5/5] Running annotation-weighted analysis...")

    # Build records
    records = []
    for key, sigs in all_sigs.items():
        s0 = sigs[0]
        clin_sig = s0['clinical_significance']
        sig_lower = clin_sig.lower()
        is_path = 'pathogenic' in sig_lower and 'benign' not in sig_lower and 'conflicting' not in sig_lower
        is_benign = 'benign' in sig_lower and 'pathogenic' not in sig_lower
        if not (is_path or is_benign):
            continue

        label = 1 if is_path else 0
        feats = extract_features(sigs, f1w_by_layer, cw_by_layer, cats_by_layer, layers)

        # Raw and functional scores
        raw_score = sum(float(np.abs(s['delta_local']).sum()) for s in sigs)
        func_score = 0.0
        for s in sigs:
            layer = s['layer']
            if layer not in f1w_by_layer:
                continue
            fw = f1w_by_layer[layer]
            cats = cats_by_layer[layer]
            delta = np.abs(s['delta_local'])
            for i in range(len(delta)):
                if cats[i] in ("functional", "ptm", "domain") and fw[i] > 0.1:
                    func_score += float(delta[i] * fw[i])

        records.append({
            "key": key,
            "gene": s0['gene'],
            "variant": s0['variant_str'],
            "label": label,
            "clin_sig": clin_sig,
            "raw_score": raw_score,
            "func_score": func_score,
            "ml_features": feats,
            "n_layers": len(sigs),
        })

    n_p = sum(1 for r in records if r['label'] == 1)
    n_b = sum(1 for r in records if r['label'] == 0)
    print(f"  {len(records)} variants: {n_p} pathogenic, {n_b} benign")

    labels = np.array([r['label'] for r in records])

    # AUCs
    raw_scores = np.array([r['raw_score'] for r in records])
    func_scores = np.array([r['func_score'] for r in records])
    auc_raw = roc_auc_score(labels, raw_scores)
    ap_raw = average_precision_score(labels, raw_scores)
    auc_func = roc_auc_score(labels, func_scores)
    ap_func = average_precision_score(labels, func_scores)

    print(f"\n  Raw perturbation:        AUC={auc_raw:.4f}  AP={ap_raw:.4f}")
    print(f"  Functional disruption:   AUC={auc_func:.4f}  AP={ap_func:.4f}")

    # Cross-layer LR
    X = np.stack([r['ml_features'] for r in records])
    y = labels
    good = X.std(axis=0) > 1e-10
    X_clean = X[:, good]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
    y_prob = cross_val_predict(lr, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    auc_lr = roc_auc_score(y, y_prob)
    ap_lr = average_precision_score(y, y_prob)
    acc = (y_pred == y).mean()

    print(f"  Cross-layer LR (5-fold): AUC={auc_lr:.4f}  AP={ap_lr:.4f}  Acc={acc:.4f}")

    # Feature importance
    feature_names = []
    for layer in layers:
        for fn in ["raw_pert", "f1_pert", "cat_pert", "n_known_pert", "n_novel_pert",
                    "max_f1_pert", "mean_delta_known", "mean_delta_novel",
                    "top10_f1_pert", "known_pert_frac", "asymmetry", "total_pert"]:
            feature_names.append(f"L{layer}_{fn}")

    lr_full = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
    lr_full.fit(X_scaled, y)
    coefs = np.zeros(X.shape[1])
    coefs[good] = lr_full.coef_[0]

    print(f"\n  Top 10 features:")
    top_idx = np.argsort(np.abs(coefs))[::-1][:10]
    for i, idx in enumerate(top_idx):
        print(f"    {i+1}. {feature_names[idx]:30s} coef={coefs[idx]:+.4f}")

    # Comparison table
    print(f"\n{'='*70}")
    print(f"  COMPARISON (n={len(records)}, {len(layers)} layers)")
    print(f"{'='*70}")
    print(f"  {'Method':<30s} {'AUC':>8s} {'AP':>8s}")
    print(f"  {'-'*48}")
    print(f"  {'Raw perturbation':<30s} {auc_raw:8.4f} {ap_raw:8.4f}")
    print(f"  {'Functional disruption':<30s} {auc_func:8.4f} {ap_func:8.4f}")
    print(f"  {'Cross-layer LR (5-fold CV)':<30s} {auc_lr:8.4f} {ap_lr:8.4f}  Acc={acc:.3f}")

    # Save
    out_dir = "results/variant_effect"
    os.makedirs(out_dir, exist_ok=True)

    output = {
        "n_variants": len(records),
        "n_pathogenic": n_p,
        "n_benign": n_b,
        "layers": layers,
        "auc_raw": round(auc_raw, 4),
        "auc_functional": round(auc_func, 4),
        "auc_lr_cv": round(auc_lr, 4),
        "ap_lr_cv": round(ap_lr, 4),
        "accuracy_lr_cv": round(acc, 4),
        "feature_importance": [
            {"feature": feature_names[idx], "coefficient": round(float(coefs[idx]), 4)}
            for idx in top_idx
        ],
    }
    with open(os.path.join(out_dir, "scaled_prediction_results.json"), "w") as f:
        json.dump(output, f, indent=2)

    # Save signatures for future use
    with open(os.path.join(out_dir, "scaled_perturbation_signatures.pkl"), "wb") as f:
        pickle.dump(all_sigs, f)

    print(f"\n  Saved to {out_dir}/scaled_prediction_results.json")


if __name__ == "__main__":
    main()
