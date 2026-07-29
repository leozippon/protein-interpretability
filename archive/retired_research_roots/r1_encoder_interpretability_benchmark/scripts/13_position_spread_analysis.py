#!/usr/bin/env python
"""Position-spread analysis: how perturbations propagate along the sequence.

The scaled_perturbation_signatures.pkl contains delta_local (d_sae at the
mutation position) and delta_global (averaged |delta| across all positions).
These summarize position 0 and all-positions-averaged. We are missing the
MIDDLE: how the effect spreads near the mutation site.

This script recomputes a richer set of position-aware features on a random
sample of variants, to test whether the spread of perturbation carries
pathogenicity signal beyond what delta_local and delta_global capture.

Key new features per layer:
  - Window L1 at ±5 residues (neighborhood effect)
  - Decay rate: how fast |delta| drops with distance from mutation
  - Peak position: where the largest |delta| occurs (should be 0 for local)
  - Spread half-width: how far from mutation does |delta| fall to half

Usage:
    python scripts/13_position_spread_analysis.py --gpu 2 --n-variants 500
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


OUT_DIR = "r1_encoder_interpretability_benchmark/results/variant_effect"
LAYERS = [19, 27, 35]  # Reduced set for speed


def load_balanced_variants(seq_map, n_each=250):
    """Load balanced variants same as other scripts."""
    from src.analysis.variant_effect import MissenseVariant

    gene_to_uniprot = {}
    idmapping_path = "data/swissprot/HUMAN_9606_idmapping.dat.gz"
    with gzip.open(idmapping_path, "rt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and parts[1] == "Gene_Name":
                if parts[0] in seq_map:
                    gene_to_uniprot[parts[2]] = parts[0]

    hgvs_pattern = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")
    aa3to1 = {
        "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
        "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
        "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
        "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    }

    pathogenic, benign = [], []
    seen = set()
    with gzip.open("data/clinvar/variant_summary.txt.gz", "rt") as f:
        f.readline()
        for line in f:
            if len(pathogenic) >= n_each * 3 and len(benign) >= n_each * 3:
                break
            fields = line.strip().split("\t")
            if len(fields) < 8 or fields[1] != "single nucleotide variant":
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
            v = MissenseVariant(
                gene=gene, uniprot_id=uniprot_id, position=pos,
                wt_residue=wt_aa, mut_residue=mut_aa,
                clinical_significance=clin_sig, source="clinvar")
            is_path = ("pathogenic" in sig_lower and
                       "benign" not in sig_lower and
                       "conflicting" not in sig_lower)
            is_benign = ("benign" in sig_lower and
                         "pathogenic" not in sig_lower)
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


def compute_position_features(wt_seq, mut_seq, variant, esm_model, tokenizer,
                               sae, layer, device):
    """Compute position-resolved SAE features for WT and mutant, returning
    a window of delta vectors around the mutation."""
    from transformers import AutoTokenizer, EsmModel

    def get_features(sequence):
        spaced = " ".join(list(sequence))
        encoded = tokenizer(spaced, return_tensors="pt", truncation=True,
                             max_length=1024).to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = esm_model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_hidden_states=True,
            )
        layer_acts = outputs.hidden_states[layer + 1].float()
        seq_len = len(sequence)
        aa_acts = layer_acts[0, 1:seq_len + 1, :]
        output = sae(aa_acts)
        return output.f.cpu().numpy()

    with torch.no_grad():
        f_wt = get_features(wt_seq)
        f_mut = get_features(mut_seq)

    pos_idx = variant.position - 1
    min_len = min(f_wt.shape[0], f_mut.shape[0])
    delta_abs = np.abs(f_mut[:min_len] - f_wt[:min_len])  # (seq_len, d_sae)
    per_pos_l1 = delta_abs.sum(axis=1)  # (seq_len,)

    # Position features
    feats = {}

    # Window sum at ±5, ±10, ±20
    for W in [5, 10, 20]:
        lo = max(0, pos_idx - W)
        hi = min(min_len, pos_idx + W + 1)
        feats[f"window_l1_{W}"] = float(per_pos_l1[lo:hi].sum())

    # Fraction of total in ±5 window
    total_l1 = float(per_pos_l1.sum())
    if total_l1 > 0:
        lo = max(0, pos_idx - 5)
        hi = min(min_len, pos_idx + 6)
        feats["local_fraction"] = float(per_pos_l1[lo:hi].sum()) / total_l1
    else:
        feats["local_fraction"] = 0.0

    # Decay rate: how fast does |delta| drop from the mutation site?
    distances = np.arange(min_len) - pos_idx
    nonzero = per_pos_l1 > 1e-6
    if nonzero.sum() >= 3 and per_pos_l1[pos_idx] > 1e-6:
        # Use exp decay model: l1(d) = l1(0) * exp(-rate * |d|)
        log_l1 = np.log(per_pos_l1[nonzero] + 1e-10)
        abs_d = np.abs(distances[nonzero])
        if np.ptp(abs_d) > 0:
            from numpy.polynomial import polynomial as P
            slope, _ = np.polyfit(abs_d, log_l1, 1)
            feats["decay_rate"] = float(-slope)
        else:
            feats["decay_rate"] = 0.0
    else:
        feats["decay_rate"] = 0.0

    # Peak position offset
    peak_idx = int(np.argmax(per_pos_l1))
    feats["peak_offset"] = float(peak_idx - pos_idx)
    feats["peak_value"] = float(per_pos_l1[peak_idx])
    feats["mut_site_value"] = float(per_pos_l1[pos_idx])
    feats["peak_is_mut_site"] = 1.0 if peak_idx == pos_idx else 0.0

    # Spread half-width
    threshold = per_pos_l1[pos_idx] / 2
    if threshold > 1e-6:
        wide = np.where(per_pos_l1 >= threshold)[0]
        if len(wide) > 0:
            spread = float(max(wide) - min(wide))
        else:
            spread = 0.0
    else:
        spread = 0.0
    feats["spread_halfwidth"] = spread

    # Total L1 / global mean
    feats["total_l1"] = total_l1
    feats["mean_l1"] = float(per_pos_l1.mean())

    # N positions with appreciable perturbation
    feats["n_perturbed_positions"] = int((per_pos_l1 > 0.1).sum())

    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--n-variants", type=int, default=500)
    parser.add_argument("--esm-model", type=str,
                        default="/Data/public/esm2_t36_3B_UR50D")
    parser.add_argument("--swissprot-cache", type=str,
                        default="data/processed/swissprot_all_max1022.pkl")
    args = parser.parse_args()

    from transformers import AutoTokenizer, EsmModel
    from src.analysis.feature_annotation import load_our_sae
    from src.analysis.variant_effect import create_mutant_sequence

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    CHECKPOINT_DIRS = {
        l: f"r1_encoder_interpretability_benchmark/results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights/layer_{l}/step_500000"
        for l in LAYERS
    }

    print("=" * 70)
    print("  Position-Spread Analysis")
    print("=" * 70)
    print(f"  Layers: {LAYERS}, Variants: {args.n_variants}")

    print("\n[1/4] Loading sequences...", flush=True)
    with open(args.swissprot_cache, "rb") as f:
        annotations = pickle.load(f)
    seq_map = {ann.accession: ann.sequence for ann in annotations
               if "Homo sapiens" in ann.organism}

    print("\n[2/4] Loading variants...", flush=True)
    n_each = args.n_variants // 2
    variants, actual_n = load_balanced_variants(seq_map, n_each=n_each)
    print(f"  {actual_n} path + {actual_n} benign = {len(variants)}")

    print("\n[3/4] Loading ESM-2-3B...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
    esm_model = EsmModel.from_pretrained(args.esm_model, torch_dtype=torch.float16)
    esm_model.to(device).eval()

    print("\n[4/4] Computing position features per layer...", flush=True)

    records = {}  # key -> {gene, variant, label, layer_feats: {layer: {...}}}
    for layer in LAYERS:
        ckpt = CHECKPOINT_DIRS[layer]
        print(f"\n  --- Layer {layer} ---", flush=True)
        t0 = time.time()
        sae = load_our_sae(ckpt, device=device)

        n_ok = 0
        for vi, var in enumerate(variants):
            wt_seq = seq_map.get(var.uniprot_id, "")
            if not wt_seq:
                continue
            mut_seq = create_mutant_sequence(wt_seq, var.position, var.wt_residue,
                                              var.mut_residue)
            if not mut_seq:
                continue

            try:
                feats = compute_position_features(
                    wt_seq, mut_seq, var, esm_model, tokenizer, sae,
                    layer, device)
            except Exception as e:
                if vi < 3:
                    print(f"  Failed: {var.gene} {var.wt_residue}{var.position}{var.mut_residue}: {e}")
                continue

            key = (var.uniprot_id, var.position, var.wt_residue, var.mut_residue)
            if key not in records:
                sig_lower = var.clinical_significance.lower()
                is_path = ("pathogenic" in sig_lower and
                           "benign" not in sig_lower and
                           "conflicting" not in sig_lower)
                records[key] = {
                    "gene": var.gene,
                    "variant": f"{var.wt_residue}{var.position}{var.mut_residue}",
                    "label": 1 if is_path else 0,
                    "layer_feats": {},
                }
            records[key]["layer_feats"][layer] = feats
            n_ok += 1
            if (vi + 1) % 100 == 0:
                print(f"    {vi+1}/{len(variants)}: {n_ok} ok "
                      f"({time.time()-t0:.0f}s)", flush=True)

        print(f"  Layer {layer}: {n_ok} ok ({time.time()-t0:.1f}s)", flush=True)
        del sae
        torch.cuda.empty_cache()

    # Build classifier input
    print("\n=== Classification ===", flush=True)
    records_list = [r for r in records.values()
                    if len(r["layer_feats"]) == len(LAYERS)]
    print(f"  {len(records_list)} complete records")

    feat_names = ["window_l1_5", "window_l1_10", "window_l1_20",
                  "local_fraction", "decay_rate", "peak_offset",
                  "peak_value", "mut_site_value", "peak_is_mut_site",
                  "spread_halfwidth", "total_l1", "mean_l1",
                  "n_perturbed_positions"]

    X = []
    y = []
    meta = []
    for r in records_list:
        feats = []
        for layer in LAYERS:
            for name in feat_names:
                feats.append(r["layer_feats"][layer].get(name, 0.0))
        X.append(feats)
        y.append(r["label"])
        meta.append({"gene": r["gene"], "variant": r["variant"]})

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    n_p = int(y.sum())
    n_b = int((1 - y).sum())
    print(f"  Feature matrix: {X.shape}, {n_p} path, {n_b} benign")

    # Run CV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score

    good = X.std(0) > 1e-8
    X_clean = X[:, good]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for C in [0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
        probs = cross_val_predict(clf, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, probs)
        ap = average_precision_score(y, probs)
        acc = ((probs > 0.5).astype(int) == y).mean()
        results[f"lr_C{C}"] = {
            "auc": float(auc), "ap": float(ap), "acc": float(acc),
            "scores": probs.tolist(),
        }
        print(f"  LR C={C}: AUC={auc:.4f} AP={ap:.4f} Acc={acc:.4f}", flush=True)

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {
        k: {kk: vv for kk, vv in v.items() if kk != "scores"}
        for k, v in results.items()
    }
    summary["n_variants"] = len(records_list)
    summary["n_pathogenic"] = n_p
    summary["n_benign"] = n_b
    summary["feat_names"] = feat_names
    summary["layers"] = LAYERS

    with open(f"{OUT_DIR}/position_spread_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-variant records for downstream use
    with open(f"{OUT_DIR}/position_spread_records.pkl", "wb") as f:
        pickle.dump(records_list, f)

    print(f"\n  Saved to {OUT_DIR}/position_spread_results.json")


if __name__ == "__main__":
    main()
