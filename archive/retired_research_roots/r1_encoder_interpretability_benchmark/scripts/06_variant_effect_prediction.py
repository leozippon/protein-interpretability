#!/usr/bin/env python
"""Variant effect prediction using multi-layer SAE perturbation signatures.

Core R1 experiment: compute WT vs mutant SAE feature perturbation for
ClinVar missense variants, then classify variant mechanisms (LOF/GOF/DN).

Usage:
    python scripts/06_variant_effect_prediction.py --gpu 4 --num-variants 500
    python scripts/06_variant_effect_prediction.py --gpu 4 --layers 23,27,31,35
"""

import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

from src.analysis.variant_effect import (
    MissenseVariant,
    PerturbationSignature,
    compute_perturbation_signature,
    classify_mechanism,
    compute_perturbation_features,
    create_mutant_sequence,
    load_clinvar_missense,
)
from src.analysis.feature_annotation import load_our_sae


# Checkpoint paths (same as 05_analyze_all_layers.py)
CHECKPOINT_DIRS = {
    19: "results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights/layer_19/step_500000",
    23: "results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights/layer_23/step_500000",
    27: "results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights/layer_27/step_500000",
    31: "results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights/layer_31/step_500000",
    35: "results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights/layer_35/step_500000",
}


def load_human_sequences_and_gene_map(swissprot_cache: str):
    """Load human protein sequences and gene-to-accession mapping."""
    with open(swissprot_cache, 'rb') as f:
        annotations = pickle.load(f)

    seq_map = {}      # accession -> sequence
    gene_map = {}     # gene_symbol -> accession (first match)

    for ann in annotations:
        if 'Homo sapiens' not in ann.organism:
            continue
        seq_map[ann.accession] = ann.sequence

        # Extract gene names from chain features
        for start, end, feat_type, desc, category in ann.features:
            if feat_type == 'chain' and desc:
                # Not ideal but chain descriptions often contain the protein name
                pass

    return seq_map, annotations


def load_variants_direct(
    variant_summary_path: str,
    seq_map: dict[str, str],
    idmapping_path: str | None = None,
    max_variants: int = 1000,
) -> list[MissenseVariant]:
    """Load ClinVar missense variants mapped to UniProt sequences.

    Uses the UniProt ID mapping file to connect gene symbols to accessions.
    """
    import gzip
    import re

    # Build gene -> uniprot mapping from ID mapping file
    gene_to_uniprot = {}
    if idmapping_path and os.path.exists(idmapping_path):
        print(f"  Loading ID mapping from {idmapping_path}...")
        opener = gzip.open if idmapping_path.endswith('.gz') else open
        with opener(idmapping_path, 'rt') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3 and parts[1] == 'Gene_Name':
                    acc = parts[0]
                    gene = parts[2]
                    if acc in seq_map:
                        gene_to_uniprot[gene] = acc
        print(f"  Mapped {len(gene_to_uniprot)} genes to UniProt accessions")

    # Parse ClinVar
    hgvs_pattern = re.compile(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})')
    aa3to1 = {
        'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
        'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
        'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
        'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    }

    variants = []
    seen = set()

    opener = gzip.open if variant_summary_path.endswith('.gz') else open
    with opener(variant_summary_path, 'rt') as f:
        header = f.readline()
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 8:
                continue

            var_type = fields[1]
            name = fields[2]
            gene = fields[4]
            clin_sig = fields[6]

            if var_type != 'single nucleotide variant':
                continue

            # Only pathogenic/likely pathogenic or benign for clear signal
            sig_lower = clin_sig.lower()
            if not any(x in sig_lower for x in ['pathogenic', 'benign']):
                continue

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

            # Verify the WT residue matches
            seq = seq_map.get(uniprot_id, "")
            if not seq or pos > len(seq) or seq[pos - 1] != wt_aa:
                continue

            key = (uniprot_id, pos, wt_aa, mut_aa)
            if key in seen:
                continue
            seen.add(key)

            variants.append(MissenseVariant(
                gene=gene,
                uniprot_id=uniprot_id,
                position=pos,
                wt_residue=wt_aa,
                mut_residue=mut_aa,
                clinical_significance=clin_sig,
                source="clinvar",
            ))

            if len(variants) >= max_variants:
                break

    return variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--num-variants", type=int, default=500)
    parser.add_argument("--layers", type=str, default="23,27,31,35",
                        help="SAE layers for perturbation analysis")
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
    print("  Variant Effect Prediction via SAE Perturbation Signatures")
    print("=" * 70)
    print(f"  GPU: {args.gpu}")
    print(f"  Layers: {layers}")
    print(f"  Max variants: {args.num_variants}")
    print()

    # --- Step 1: Load protein sequences ---
    print("[1/5] Loading human protein sequences...")
    t0 = time.time()
    seq_map, all_annotations = load_human_sequences_and_gene_map(args.swissprot_cache)
    human_seqs = {acc: ann.sequence for ann in all_annotations
                  if 'Homo sapiens' in ann.organism
                  for acc in [ann.accession]}
    print(f"  {len(human_seqs)} human proteins loaded ({time.time()-t0:.1f}s)")

    # --- Step 2: Load ClinVar variants ---
    print("\n[2/5] Loading ClinVar missense variants...")
    t0 = time.time()
    # Load all mapped variants (no limit), then balance
    all_variants = load_variants_direct(
        variant_summary_path=args.clinvar_path,
        seq_map=human_seqs,
        idmapping_path=args.idmapping_path,
        max_variants=100000,  # load many to find enough benign
    )

    # Split by clear pathogenic vs clear benign
    pathogenic = [v for v in all_variants
                  if 'pathogenic' in v.clinical_significance.lower()
                  and 'benign' not in v.clinical_significance.lower()
                  and 'conflicting' not in v.clinical_significance.lower()]
    benign = [v for v in all_variants
              if 'benign' in v.clinical_significance.lower()
              and 'pathogenic' not in v.clinical_significance.lower()]

    print(f"  {len(all_variants)} total mapped variants")
    print(f"  {len(pathogenic)} pathogenic, {len(benign)} benign")

    # Balance and sample
    n_each = min(len(pathogenic), len(benign), args.num_variants // 2)
    rng = np.random.RandomState(42)
    if len(pathogenic) > n_each:
        idx = rng.choice(len(pathogenic), n_each, replace=False)
        pathogenic = [pathogenic[i] for i in idx]
    if len(benign) > n_each:
        idx = rng.choice(len(benign), n_each, replace=False)
        benign = [benign[i] for i in idx]

    variants = pathogenic + benign
    rng.shuffle(variants)

    print(f"  Balanced to {len(pathogenic)} pathogenic + {len(benign)} benign = {len(variants)}")
    t_load = time.time() - t0
    print(f"  ({t_load:.1f}s)")

    # Clinical significance breakdown
    sig_counts = Counter(v.clinical_significance for v in variants)
    for sig, count in sig_counts.most_common(10):
        print(f"    {sig}: {count}")

    if len(variants) == 0:
        print("  ERROR: No variants found. Check ID mapping path.")
        return

    # --- Step 3: Load ESM-2-3B ---
    print("\n[3/5] Loading ESM-2-3B...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
    esm_model = EsmModel.from_pretrained(args.esm_model, torch_dtype=torch.float16)
    esm_model.to(device).eval()
    print(f"  ESM-2-3B loaded ({time.time()-t0:.1f}s)")

    # --- Step 4: Compute perturbation signatures ---
    print(f"\n[4/5] Computing perturbation signatures across {len(layers)} layers...")

    all_signatures = {}  # variant_key -> list[PerturbationSignature]
    failed_variants = []

    for layer in layers:
        if layer not in CHECKPOINT_DIRS:
            print(f"  Skipping layer {layer}: no checkpoint")
            continue

        ckpt_path = CHECKPOINT_DIRS[layer]
        if not os.path.exists(os.path.join(ckpt_path, "sae.pt")):
            print(f"  Skipping layer {layer}: checkpoint not found")
            continue

        print(f"\n  --- Layer {layer} ({layer/36*100:.0f}% depth) ---")
        t0 = time.time()
        sae = load_our_sae(ckpt_path, device=device)
        print(f"  SAE loaded: d_sae={sae.d_sae}, k={sae.k}")

        n_computed = 0
        n_failed = 0

        for vi, variant in enumerate(variants):
            wt_seq = human_seqs.get(variant.uniprot_id, "")
            if not wt_seq:
                continue

            mut_seq = create_mutant_sequence(
                wt_seq, variant.position, variant.wt_residue, variant.mut_residue
            )
            if not mut_seq:
                n_failed += 1
                continue

            # Skip very long sequences (ESM-2 limit)
            if len(wt_seq) > 1022:
                continue

            try:
                sig = compute_perturbation_signature(
                    wt_sequence=wt_seq,
                    mut_sequence=mut_seq,
                    variant=variant,
                    esm_model=esm_model,
                    tokenizer=tokenizer,
                    sae=sae,
                    layer=layer,
                    device=device,
                )

                key = (variant.uniprot_id, variant.position,
                       variant.wt_residue, variant.mut_residue)
                if key not in all_signatures:
                    all_signatures[key] = []
                all_signatures[key].append(sig)
                n_computed += 1

            except Exception as e:
                n_failed += 1
                if n_failed <= 3:
                    print(f"    Failed: {variant.gene} {variant.wt_residue}{variant.position}"
                          f"{variant.mut_residue}: {e}")

            if (vi + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"    {vi+1}/{len(variants)} variants, "
                      f"{n_computed} computed, {elapsed:.1f}s")

        elapsed = time.time() - t0
        print(f"  Layer {layer}: {n_computed} computed, {n_failed} failed ({elapsed:.1f}s)")

        del sae
        torch.cuda.empty_cache()

    # --- Step 5: Classify mechanisms and analyze ---
    print(f"\n[5/5] Analyzing perturbation patterns...")

    results = []
    pathogenic_results = []
    benign_results = []

    for key, sigs in all_signatures.items():
        variant = sigs[0].variant

        # Classify mechanism
        mechanism = classify_mechanism(sigs)

        # Compute feature vector
        features = compute_perturbation_features(sigs)

        # Aggregate statistics
        total_ablated = sum(s.n_ablated for s in sigs)
        total_amplified = sum(s.n_amplified for s in sigs)
        total_novel = sum(s.n_novel for s in sigs)
        total_perturbation = sum(s.total_perturbation for s in sigs)
        mean_wt_active = np.mean([s.wt_active_count for s in sigs])
        mean_mut_active = np.mean([s.mut_active_count for s in sigs])

        result = {
            "gene": variant.gene,
            "uniprot_id": variant.uniprot_id,
            "variant": f"{variant.wt_residue}{variant.position}{variant.mut_residue}",
            "clinical_significance": variant.clinical_significance,
            "predicted_mechanism": mechanism,
            "n_ablated": total_ablated,
            "n_amplified": total_amplified,
            "n_novel": total_novel,
            "total_perturbation": round(total_perturbation, 2),
            "mean_wt_active": round(mean_wt_active, 1),
            "mean_mut_active": round(mean_mut_active, 1),
            "activity_ratio": round(mean_mut_active / max(mean_wt_active, 1), 3),
            "layers_analyzed": len(sigs),
        }
        results.append(result)

        is_pathogenic = any(x in variant.clinical_significance.lower()
                           for x in ['pathogenic'])
        is_benign = 'benign' in variant.clinical_significance.lower()

        if is_pathogenic:
            pathogenic_results.append(result)
        elif is_benign:
            benign_results.append(result)

    # Report
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Total variants analyzed: {len(results)}")
    print(f"  Pathogenic variants: {len(pathogenic_results)}")
    print(f"  Benign variants: {len(benign_results)}")
    print()

    # Mechanism distribution
    mech_counts = Counter(r["predicted_mechanism"] for r in results)
    print(f"  Mechanism predictions (all variants):")
    for mech, count in mech_counts.most_common():
        print(f"    {mech}: {count} ({count/len(results)*100:.1f}%)")

    path_mech = Counter(r["predicted_mechanism"] for r in pathogenic_results)
    benign_mech = Counter(r["predicted_mechanism"] for r in benign_results)

    print(f"\n  Pathogenic variant mechanisms:")
    for mech, count in path_mech.most_common():
        print(f"    {mech}: {count} ({count/len(pathogenic_results)*100:.1f}%)")

    print(f"\n  Benign variant mechanisms:")
    for mech, count in benign_mech.most_common():
        print(f"    {mech}: {count} ({count/len(benign_results)*100:.1f}%)")

    # Perturbation magnitude comparison
    if pathogenic_results and benign_results:
        path_pert = [r["total_perturbation"] for r in pathogenic_results]
        benign_pert = [r["total_perturbation"] for r in benign_results]

        print(f"\n  Perturbation magnitude (pathogenic vs benign):")
        print(f"    Pathogenic: mean={np.mean(path_pert):.2f}, "
              f"median={np.median(path_pert):.2f}, "
              f"std={np.std(path_pert):.2f}")
        print(f"    Benign:     mean={np.mean(benign_pert):.2f}, "
              f"median={np.median(benign_pert):.2f}, "
              f"std={np.std(benign_pert):.2f}")

        # Effect size
        if np.std(path_pert) + np.std(benign_pert) > 0:
            cohens_d = (np.mean(path_pert) - np.mean(benign_pert)) / (
                np.sqrt((np.std(path_pert)**2 + np.std(benign_pert)**2) / 2)
            )
            print(f"    Cohen's d: {cohens_d:.3f}")

        # AUC-ROC for pathogenicity detection via perturbation magnitude
        from sklearn.metrics import roc_auc_score
        labels = [1] * len(pathogenic_results) + [0] * len(benign_results)
        scores = path_pert + benign_pert
        auc = roc_auc_score(labels, scores)
        print(f"    AUC-ROC (perturbation → pathogenicity): {auc:.3f}")

    # Activity ratio comparison
    if pathogenic_results and benign_results:
        path_ratio = [r["activity_ratio"] for r in pathogenic_results]
        benign_ratio = [r["activity_ratio"] for r in benign_results]
        print(f"\n  Activity ratio (mut/WT features):")
        print(f"    Pathogenic: mean={np.mean(path_ratio):.3f}, "
              f"std={np.std(path_ratio):.3f}")
        print(f"    Benign:     mean={np.mean(benign_ratio):.3f}, "
              f"std={np.std(benign_ratio):.3f}")

    # Top examples
    print(f"\n  Top 10 most perturbing pathogenic variants:")
    top_path = sorted(pathogenic_results, key=lambda r: -r["total_perturbation"])[:10]
    for r in top_path:
        print(f"    {r['gene']:10s} {r['variant']:10s} "
              f"pert={r['total_perturbation']:8.2f} "
              f"ablated={r['n_ablated']:3d} amp={r['n_amplified']:3d} "
              f"novel={r['n_novel']:3d} -> {r['predicted_mechanism']}")

    print(f"\n  Top 10 least perturbing benign variants:")
    top_benign = sorted(benign_results, key=lambda r: r["total_perturbation"])[:10]
    for r in top_benign:
        print(f"    {r['gene']:10s} {r['variant']:10s} "
              f"pert={r['total_perturbation']:8.2f} "
              f"ablated={r['n_ablated']:3d} amp={r['n_amplified']:3d} "
              f"novel={r['n_novel']:3d} -> {r['predicted_mechanism']}")

    # --- Save results ---
    out_dir = "results/variant_effect"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "variant_predictions.json"), "w") as f:
        json.dump(results, f, indent=2)

    summary = {
        "total_variants": len(results),
        "pathogenic_count": len(pathogenic_results),
        "benign_count": len(benign_results),
        "layers_used": layers,
        "mechanism_distribution": dict(mech_counts),
        "pathogenic_mechanisms": dict(path_mech),
        "benign_mechanisms": dict(benign_mech),
    }
    if pathogenic_results and benign_results:
        summary["pathogenic_perturbation_mean"] = round(np.mean(path_pert), 2)
        summary["benign_perturbation_mean"] = round(np.mean(benign_pert), 2)
        summary["auc_perturbation_pathogenicity"] = round(auc, 4)

    with open(os.path.join(out_dir, "variant_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save raw signatures for downstream analysis
    with open(os.path.join(out_dir, "perturbation_signatures.pkl"), "wb") as f:
        # Convert to serializable format
        sig_data = {}
        for key, sigs in all_signatures.items():
            sig_data[key] = [{
                "layer": s.layer,
                "n_ablated": s.n_ablated,
                "n_amplified": s.n_amplified,
                "n_novel": s.n_novel,
                "total_perturbation": s.total_perturbation,
                "wt_active_count": s.wt_active_count,
                "mut_active_count": s.mut_active_count,
                "delta_local": s.delta_local,
                "delta_global": s.delta_global,
            } for s in sigs]
        pickle.dump(sig_data, f)

    print(f"\n  Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
