#!/usr/bin/env python
"""Test the feature-annotation alignment pipeline on InterProt 650M SAEs.

This script:
1. Loads a subset of Swiss-Prot annotations (cached)
2. Loads ESM-2-650M + InterProt SAE (layer 24)
3. Runs ESM-2 → SAE inference to get per-residue feature activations
4. Computes F1 between each SAE feature and each annotation type
5. Reports KNOWN/PARTIAL/NOVEL classification distribution

Usage:
    python scripts/03_test_annotation_pipeline.py [--gpu 2] [--num-proteins 200]
"""

import argparse
import os
import pickle
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

from src.analysis.feature_annotation import (
    compute_feature_annotation_f1,
    extract_sae_activations,
    load_interprot_sae,
    summarize_results,
)
from src.data.swissprot_parser import ProteinAnnotation, parse_and_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2, help="GPU index to use")
    parser.add_argument("--num-proteins", type=int, default=200, help="Number of proteins to test")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--sae-layer", type=int, default=24, help="InterProt SAE layer (1-indexed)")
    parser.add_argument("--sae-path", type=str,
                        default="/Data/public/InterProt-ESM2-SAEs/esm2_plm1280_l24_sae4096.safetensors")
    parser.add_argument("--esm-model", type=str,
                        default="/Data/public/esm2_t33_650M_UR50D")
    parser.add_argument("--swissprot-xml", type=str,
                        default="data/swissprot/uniprot_sprot.xml.gz")
    parser.add_argument("--cache-path", type=str,
                        default="data/processed/swissprot_all_max1022.pkl")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    print(f"=== Feature-Annotation Alignment Pipeline Test ===")
    print(f"GPU: {args.gpu}")
    print(f"SAE: InterProt layer {args.sae_layer} (d=4096)")
    print(f"ESM model: {args.esm_model}")
    print(f"Num proteins: {args.num_proteins}")
    print()

    # Step 1: Load annotations
    print("[1/4] Loading Swiss-Prot annotations...")
    t0 = time.time()

    # Try full cache first, then fall back to test cache or parse fresh
    if os.path.exists(args.cache_path):
        with open(args.cache_path, "rb") as f:
            all_annotations = pickle.load(f)
        print(f"  Loaded {len(all_annotations):,} proteins from cache")
    else:
        print(f"  Full cache not found, parsing fresh (max {args.num_proteins} proteins)...")
        all_annotations = parse_and_cache(
            args.swissprot_xml,
            cache_path=f"data/processed/swissprot_test_{args.num_proteins}.pkl",
            max_seq_len=1022,
            max_proteins=args.num_proteins,
        )

    # Filter to proteins with annotations and select subset
    annotated = [a for a in all_annotations if a.features]
    if len(annotated) > args.num_proteins:
        # Sample deterministically
        rng = np.random.RandomState(42)
        indices = rng.choice(len(annotated), args.num_proteins, replace=False)
        annotations = [annotated[i] for i in sorted(indices)]
    else:
        annotations = annotated[:args.num_proteins]

    sequences = [a.sequence for a in annotations]
    print(f"  Selected {len(annotations)} annotated proteins")
    print(f"  Total residues: {sum(len(s) for s in sequences):,}")
    print(f"  Time: {time.time() - t0:.1f}s")
    print()

    # Step 2: Load models
    print("[2/4] Loading ESM-2-650M + InterProt SAE...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
    esm_model = EsmModel.from_pretrained(args.esm_model, torch_dtype=torch.float16)
    esm_model.to(device).eval()

    sae = load_interprot_sae(args.sae_path, d_model=1280, d_sae=4096, device=device)
    print(f"  ESM-2 loaded to {device}")
    print(f"  SAE loaded: {sum(p.numel() for p in sae.parameters()):,} params")
    print(f"  Time: {time.time() - t0:.1f}s")
    print()

    # Step 3: Extract SAE activations
    # InterProt layer 24 = ESM-2 layer index 23 (0-indexed)
    target_layer = args.sae_layer - 1
    print(f"[3/4] Running ESM-2 → SAE inference (layer {args.sae_layer}, idx {target_layer})...")
    t0 = time.time()

    sae_activations = extract_sae_activations(
        sequences=sequences,
        esm_model=esm_model,
        tokenizer=tokenizer,
        sae=sae,
        target_layer=target_layer,
        device=device,
        batch_size=args.batch_size,
        sae_type="interprot",
    )

    total_residues = sum(a.shape[0] for a in sae_activations)
    print(f"  Processed {len(sae_activations)} proteins, {total_residues:,} residues")
    print(f"  Activation shapes: {sae_activations[0].shape} (first protein)")
    print(f"  Nonzero features per residue (mean): {np.mean([np.count_nonzero(a, axis=1).mean() for a in sae_activations]):.1f}")
    print(f"  Time: {time.time() - t0:.1f}s")
    print()

    # Free GPU memory
    del esm_model, sae
    torch.cuda.empty_cache()

    # Step 4: Compute F1 alignment
    print("[4/4] Computing feature-annotation F1...")
    t0 = time.time()

    results = compute_feature_annotation_f1(
        annotations=annotations,
        sae_activations=sae_activations,
        d_sae=4096,
        threshold_percentile=95.0,
        min_positive_residues=50,
    )

    summary = summarize_results(results)
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s")
    print()

    # Report results
    print("=" * 60)
    print("RESULTS: InterProt 650M Layer 24 SAE (d=4096)")
    print("=" * 60)
    print(f"Total features: {summary['total_features']}")
    print(f"Alive features: {summary['alive_features']}")
    print(f"Dead features:  {summary['dead_features']}")
    print()

    print("Classification (alive features):")
    for cls, count in sorted(summary["classification"].items()):
        pct = summary["classification_pct"][cls]
        print(f"  {cls:8s}: {count:4d} ({pct:5.1f}%)")
    print()
    print(f"Mean best F1 (alive): {summary['mean_best_f1_alive']:.4f}")
    print()

    print("Top 20 KNOWN features (F1 > 0.5):")
    for feat_idx, ann_name, f1 in summary["top_known_features"]:
        print(f"  Feature {feat_idx:4d}: F1={f1}  ← {ann_name}")
    print()

    # Save results
    os.makedirs("results/annotation_alignment", exist_ok=True)
    out_path = "results/annotation_alignment/interprot_l24_test.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"results": results, "summary": summary}, f)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
