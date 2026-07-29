#!/usr/bin/env python
"""ESM-2 log-likelihood ratio baseline for variant pathogenicity prediction.

Computes the standard zero-shot variant effect prediction score:
  score = log P(mut_aa | context) - log P(wt_aa | context)

where context is the wild-type sequence with the mutation position masked.

This is the established baseline (Meier et al. 2021, Brandes et al. 2023)
that our SAE-based annotation-weighted approach must be compared against.

Also computes:
  - ESM-2 embedding cosine similarity between WT and mutant
  - Combined SAE + LLR score

Usage:
    python scripts/09_esm2_baseline_comparison.py --gpu 2 --num-variants 2000
"""

import argparse
import gzip
import json
import os
import pickle
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, EsmForMaskedLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.variant_effect import MissenseVariant, create_mutant_sequence


def load_variants_balanced(clinvar_path, seq_map, idmapping_path, n_each=1000):
    """Same balanced loading as 08_scaled_variant_prediction.py."""
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
        f.readline()
        for line in f:
            if len(pathogenic) >= n_each * 5 and len(benign) >= n_each * 5:
                break
            fields = line.strip().split('\t')
            if len(fields) < 8 or fields[1] != 'single nucleotide variant':
                continue
            name, gene, clin_sig = fields[2], fields[4], fields[6]
            sig_lower = clin_sig.lower()
            match = hgvs_pattern.search(name)
            if not match:
                continue
            wt_aa3, pos_str, mut_aa3 = match.groups()
            wt_aa, mut_aa = aa3to1.get(wt_aa3), aa3to1.get(mut_aa3)
            if not wt_aa or not mut_aa:
                continue
            pos = int(pos_str)
            uniprot_id = gene_to_uniprot.get(gene, "")
            if not uniprot_id:
                continue
            seq = seq_map.get(uniprot_id, "")
            if not seq or pos > len(seq) or seq[pos - 1] != wt_aa or len(seq) > 1022:
                continue
            key = (uniprot_id, pos, wt_aa, mut_aa)
            if key in seen:
                continue
            seen.add(key)
            v = MissenseVariant(gene=gene, uniprot_id=uniprot_id, position=pos,
                                wt_residue=wt_aa, mut_residue=mut_aa,
                                clinical_significance=clin_sig, source="clinvar")
            is_path = 'pathogenic' in sig_lower and 'benign' not in sig_lower and 'conflicting' not in sig_lower
            is_benign = 'benign' in sig_lower and 'pathogenic' not in sig_lower
            if is_path:
                pathogenic.append(v)
            elif is_benign:
                benign.append(v)

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


@torch.no_grad()
def compute_masked_marginal_llr(
    sequence: str,
    position: int,  # 1-indexed
    wt_aa: str,
    mut_aa: str,
    model,
    tokenizer,
    device: str,
) -> float:
    """Compute masked marginal log-likelihood ratio.

    score = log P(mut_aa | context_masked) - log P(wt_aa | context_masked)

    Negative score = model prefers WT → variant is likely deleterious.
    """
    # ESM-2 tokenizes amino acids as single tokens after spacing
    spaced = " ".join(list(sequence))
    encoded = tokenizer(spaced, return_tensors="pt", truncation=True,
                        max_length=1024).to(device)

    # Mask the mutation position
    # ESM-2 tokens: [CLS, AA1, AA2, ..., AAN, EOS]
    # Position is 1-indexed, so token index = position (CLS at 0)
    mask_idx = position  # 1-indexed position maps to token index `position`
    input_ids = encoded["input_ids"].clone()
    input_ids[0, mask_idx] = tokenizer.mask_token_id

    with torch.amp.autocast("cuda", dtype=torch.float16):
        outputs = model(input_ids=input_ids, attention_mask=encoded["attention_mask"])

    logits = outputs.logits[0, mask_idx].float()  # (vocab_size,)
    log_probs = F.log_softmax(logits, dim=-1)

    # Get token IDs for WT and mutant amino acids
    wt_token = tokenizer.convert_tokens_to_ids(wt_aa)
    mut_token = tokenizer.convert_tokens_to_ids(mut_aa)

    llr = float(log_probs[mut_token] - log_probs[wt_token])
    return llr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--num-variants", type=int, default=2000)
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

    print("=" * 70)
    print("  ESM-2 Baseline Comparison for Variant Effect Prediction")
    print("=" * 70)

    # Load sequences
    print("\n[1/4] Loading protein sequences...")
    with open(args.swissprot_cache, 'rb') as f:
        annotations = pickle.load(f)
    seq_map = {ann.accession: ann.sequence for ann in annotations
               if 'Homo sapiens' in ann.organism}
    print(f"  {len(seq_map)} human proteins")

    # Load variants (same balanced sampling as script 08)
    print("\n[2/4] Loading ClinVar variants...")
    n_each = args.num_variants // 2
    variants, actual_n = load_variants_balanced(
        args.clinvar_path, seq_map, args.idmapping_path, n_each)
    print(f"  Balanced: {actual_n} pathogenic + {actual_n} benign = {len(variants)}")

    # Load ESM-2 for masked LM
    print("\n[3/4] Loading ESM-2-3B (masked LM head)...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
    model = EsmForMaskedLM.from_pretrained(args.esm_model, torch_dtype=torch.float16)
    model.to(device).eval()
    print(f"  Loaded ({time.time()-t0:.1f}s)")

    # Compute LLR for each variant
    print(f"\n[4/4] Computing masked marginal log-likelihood ratios...")
    t0 = time.time()

    results = []
    n_ok, n_fail = 0, 0

    for vi, var in enumerate(variants):
        seq = seq_map.get(var.uniprot_id, "")
        if not seq:
            n_fail += 1
            continue

        try:
            llr = compute_masked_marginal_llr(
                seq, var.position, var.wt_residue, var.mut_residue,
                model, tokenizer, device)

            is_path = ('pathogenic' in var.clinical_significance.lower()
                       and 'benign' not in var.clinical_significance.lower()
                       and 'conflicting' not in var.clinical_significance.lower())

            results.append({
                "key": (var.uniprot_id, var.position, var.wt_residue, var.mut_residue),
                "gene": var.gene,
                "variant": f"{var.wt_residue}{var.position}{var.mut_residue}",
                "label": 1 if is_path else 0,
                "llr": llr,
                "clin_sig": var.clinical_significance,
            })
            n_ok += 1
        except Exception as e:
            n_fail += 1
            if n_fail <= 3:
                print(f"  Failed: {var.gene} {var.wt_residue}{var.position}{var.mut_residue}: {e}")

        if (vi + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = n_ok / max(elapsed, 1)
            print(f"  {vi+1}/{len(variants)}: {n_ok} ok, {n_fail} fail ({rate:.1f} var/s)")

    elapsed = time.time() - t0
    print(f"  Done: {n_ok} ok, {n_fail} fail ({elapsed:.1f}s)")

    # Compute AUC-ROC
    labels = np.array([r["label"] for r in results])
    # Negative LLR = model prefers WT = deleterious → use -LLR as pathogenicity score
    llr_scores = np.array([-r["llr"] for r in results])

    auc_llr = roc_auc_score(labels, llr_scores)
    ap_llr = average_precision_score(labels, llr_scores)

    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  ESM-2 Masked Marginal LLR:  AUC={auc_llr:.4f}  AP={ap_llr:.4f}")

    # Load our SAE results for comparison
    sae_results_path = "results/variant_effect/scaled_prediction_results.json"
    if os.path.exists(sae_results_path):
        with open(sae_results_path) as f:
            sae_results = json.load(f)
        print(f"\n  SAE Annotation-Weighted LR:  AUC={sae_results['auc_lr_cv']:.4f}  "
              f"AP={sae_results['ap_lr_cv']:.4f}")
        print(f"  SAE Raw Perturbation:       AUC={sae_results['auc_raw']:.4f}")
        print(f"  SAE Functional Disruption:  AUC={sae_results['auc_functional']:.4f}")

    # Try combining LLR + SAE features
    # Load SAE perturbation signatures to combine
    sae_sigs_path = "results/variant_effect/scaled_perturbation_signatures.pkl"
    if os.path.exists(sae_sigs_path):
        print(f"\n  --- Combined LLR + SAE Classifier ---")
        with open(sae_sigs_path, 'rb') as f:
            sae_sigs = pickle.load(f)

        # Build combined feature matrix
        # Match variants between LLR results and SAE results
        llr_map = {r["key"]: r["llr"] for r in results}

        # Load annotation weights
        layers = [19, 23, 27, 31, 35]
        f1w_by_layer = {}
        cw_by_layer = {}
        CATEGORY_WEIGHTS = {
            "functional": 5.0, "ptm": 4.0, "domain": 3.0,
            "region": 2.0, "topology": 1.5, "secondary_structure": 1.0, "chain": 0.5,
        }
        for layer in layers:
            pkl = f"results/annotation_alignment/ours_3B_l{layer}_step500000.pkl"
            if os.path.exists(pkl):
                with open(pkl, 'rb') as f:
                    data = pickle.load(f)
                rr = data['results']
                d_sae = len(rr)
                f1w = np.zeros(d_sae, dtype=np.float32)
                cw = np.zeros(d_sae, dtype=np.float32)
                for r in rr:
                    if r.alive:
                        f1w[r.feature_idx] = r.best_f1
                        cat = r.best_annotation.split('/')[0] if r.best_annotation and '/' in r.best_annotation else (r.best_annotation or "")
                        cw[r.feature_idx] = r.best_f1 * CATEGORY_WEIGHTS.get(cat, 1.0)
                f1w_by_layer[layer] = f1w
                cw_by_layer[layer] = cw

        # Build combined features for matched variants
        combined_records = []
        for key, sigs in sae_sigs.items():
            if key not in llr_map:
                continue
            llr_val = llr_map[key]
            s0 = sigs[0]
            clin_sig = s0.get('clinical_significance', '')
            sig_lower = clin_sig.lower()
            is_path = 'pathogenic' in sig_lower and 'benign' not in sig_lower and 'conflicting' not in sig_lower
            is_benign = 'benign' in sig_lower and 'pathogenic' not in sig_lower
            if not (is_path or is_benign):
                continue

            # SAE features (same as script 08)
            sig_by_layer = {s['layer']: s for s in sigs}
            sae_feats = []
            for layer in layers:
                if layer not in sig_by_layer:
                    sae_feats.extend([0.0] * 12)
                    continue
                s = sig_by_layer[layer]
                delta = np.abs(s['delta_local'])
                f1w = f1w_by_layer.get(layer, np.zeros_like(delta))
                cw_arr = cw_by_layer.get(layer, np.zeros_like(delta))
                known = f1w > 0.2
                novel = f1w < 0.05
                perturbed = delta > 0.1
                raw_pert = float(delta.sum())
                f1_pert = float((f1w * delta).sum())
                cat_pert = float((cw_arr * delta).sum())
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
                sae_feats.extend([raw_pert, f1_pert, cat_pert, n_known_p, n_novel_p,
                                  max_f1_p, mean_dk, mean_dn, top10_pert, known_frac,
                                  asym, float(s.get('total_perturbation', raw_pert))])

            combined_records.append({
                "label": 1 if is_path else 0,
                "llr": llr_val,
                "sae_feats": np.array(sae_feats, dtype=np.float32),
            })

        if len(combined_records) > 100:
            y = np.array([r["label"] for r in combined_records])
            X_sae = np.stack([r["sae_feats"] for r in combined_records])
            X_llr = np.array([r["llr"] for r in combined_records]).reshape(-1, 1)
            X_combined = np.hstack([X_sae, X_llr])

            good = X_combined.std(axis=0) > 1e-10
            X_clean = X_combined[:, good]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
            y_prob = cross_val_predict(lr, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
            auc_combined = roc_auc_score(y, y_prob)
            ap_combined = average_precision_score(y, y_prob)
            acc_combined = ((y_prob > 0.5).astype(int) == y).mean()

            # Also SAE-only on matched set
            X_sae_clean = X_sae[:, X_sae.std(axis=0) > 1e-10]
            X_sae_scaled = StandardScaler().fit_transform(X_sae_clean)
            y_prob_sae = cross_val_predict(lr, X_sae_scaled, y, cv=cv, method='predict_proba')[:, 1]
            auc_sae_matched = roc_auc_score(y, y_prob_sae)

            # LLR-only on matched set
            auc_llr_matched = roc_auc_score(y, -np.array([r["llr"] for r in combined_records]))

            print(f"  Matched variants: {len(combined_records)} "
                  f"({sum(r['label']==1 for r in combined_records)} path, "
                  f"{sum(r['label']==0 for r in combined_records)} benign)")
            print(f"  LLR only (matched):     AUC={auc_llr_matched:.4f}")
            print(f"  SAE only (matched):     AUC={auc_sae_matched:.4f}")
            print(f"  Combined LLR+SAE:       AUC={auc_combined:.4f}  "
                  f"AP={ap_combined:.4f}  Acc={acc_combined:.4f}")

    # Final comparison table
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"  {'Method':<35s} {'AUC':>8s} {'AP':>8s}")
    print(f"  {'-'*53}")
    print(f"  {'ESM-2 LLR (zero-shot)':<35s} {auc_llr:8.4f} {ap_llr:8.4f}")
    if os.path.exists(sae_results_path):
        print(f"  {'SAE Raw Perturbation':<35s} {sae_results['auc_raw']:8.4f}")
        print(f"  {'SAE Functional Disruption':<35s} {sae_results['auc_functional']:8.4f}")
        print(f"  {'SAE Cross-Layer LR (5-fold)':<35s} {sae_results['auc_lr_cv']:8.4f} "
              f"{sae_results['ap_lr_cv']:8.4f}")
    if 'auc_combined' in dir() or 'auc_combined' in locals():
        print(f"  {'Combined LLR+SAE (5-fold)':<35s} {auc_combined:8.4f} {ap_combined:8.4f}")

    # Save results
    out_dir = "results/variant_effect"
    os.makedirs(out_dir, exist_ok=True)
    output = {
        "esm2_llr_auc": round(auc_llr, 4),
        "esm2_llr_ap": round(ap_llr, 4),
        "n_variants": len(results),
        "n_pathogenic": int(labels.sum()),
        "n_benign": int((1 - labels).sum()),
    }
    if 'auc_combined' in locals():
        output["combined_auc"] = round(auc_combined, 4)
        output["combined_ap"] = round(ap_combined, 4)
        output["combined_accuracy"] = round(acc_combined, 4)
        output["n_matched"] = len(combined_records)
        output["sae_matched_auc"] = round(auc_sae_matched, 4)
        output["llr_matched_auc"] = round(auc_llr_matched, 4)

    with open(os.path.join(out_dir, "esm2_baseline_comparison.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {out_dir}/esm2_baseline_comparison.json")

    # Also save per-variant LLR scores for downstream ensembles
    per_variant = [
        {
            "gene": r["gene"],
            "variant": r["variant"],
            "label": int(r["label"]),
            "llr": float(r["llr"]),
            "clin_sig": r["clin_sig"],
        }
        for r in results
    ]
    with open(os.path.join(out_dir, "esm2_per_variant_llr.json"), "w") as f:
        json.dump(per_variant, f, indent=2)
    print(f"  Saved per-variant LLR to {out_dir}/esm2_per_variant_llr.json")


if __name__ == "__main__":
    main()
