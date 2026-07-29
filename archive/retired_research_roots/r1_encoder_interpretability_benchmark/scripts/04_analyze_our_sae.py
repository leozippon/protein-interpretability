#!/usr/bin/env python
"""Annotation alignment analysis for our trained ESM-2-3B SAEs.

Runs the feature-annotation F1 pipeline on our SAE checkpoints and compares
with the InterProt 650M baseline.

Usage:
    python scripts/04_analyze_our_sae.py --gpu 1 --num-proteins 1000
    python scripts/04_analyze_our_sae.py --gpu 1 --layer 3 --step 500000
"""

import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

from src.analysis.feature_annotation import (
    compute_feature_annotation_f1,
    extract_sae_activations,
    load_our_sae,
    summarize_results,
)
from src.data.swissprot_parser import ProteinAnnotation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--num-proteins", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--layer", type=int, default=3, help="ESM-2 layer (0-indexed)")
    parser.add_argument("--step", type=int, default=500000, help="Checkpoint step")
    parser.add_argument("--checkpoint-root", type=str, default="results/sae_weights",
                        help="Root containing layer_{layer}/step_{step} SAE checkpoints")
    parser.add_argument("--esm-model", type=str,
                        default="/Data/public/esm2_t36_3B_UR50D")
    parser.add_argument("--cache-path", type=str,
                        default="data/processed/swissprot_all_max1022.pkl")
    parser.add_argument("--save-firing-positions", action="store_true",
                        help="Store top thresholded residue positions/examples per feature")
    parser.add_argument("--max-firing-positions-per-feature", type=int, default=200)
    parser.add_argument("--out-prefix", type=str, default=None,
                        help="Optional output prefix; default is ours_3B_l{layer}_step{step}")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    checkpoint_path = os.path.join(
        args.checkpoint_root, f"layer_{args.layer}", f"step_{args.step}"
    )

    print("=" * 70)
    print("  Annotation Alignment: ESM-2-3B SAE")
    print("=" * 70)
    print(f"  GPU: {args.gpu}")
    print(f"  Layer: {args.layer}, Step: {args.step}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Num proteins: {args.num_proteins}")
    print()

    # --- Step 1: Load annotations ---
    print("[1/4] Loading Swiss-Prot annotations...")
    t0 = time.time()

    with open(args.cache_path, "rb") as f:
        all_annotations = pickle.load(f)
    print(f"  Loaded {len(all_annotations):,} proteins from cache")

    annotated = [a for a in all_annotations if a.features]
    rng = np.random.RandomState(42)
    indices = rng.choice(len(annotated), min(args.num_proteins, len(annotated)), replace=False)
    annotations = [annotated[i] for i in sorted(indices)]
    sequences = [a.sequence for a in annotations]

    total_residues = sum(len(s) for s in sequences)
    print(f"  Selected {len(annotations)} annotated proteins")
    print(f"  Total residues: {total_residues:,}")
    print(f"  Time: {time.time() - t0:.1f}s")
    print()

    # --- Step 2: Load ESM-2-3B + our SAE ---
    print("[2/4] Loading ESM-2-3B + our SAE...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
    esm_model = EsmModel.from_pretrained(args.esm_model, torch_dtype=torch.float16)
    esm_model.to(device).eval()

    sae = load_our_sae(checkpoint_path, device=device)
    d_sae = sae.d_sae

    print(f"  ESM-2-3B loaded to {device}")
    print(f"  SAE: d_sae={d_sae}, k={sae.k}, params={sum(p.numel() for p in sae.parameters()):,}")
    print(f"  Time: {time.time() - t0:.1f}s")
    print()

    # --- Step 3: Extract SAE activations ---
    print(f"[3/4] Running ESM-2-3B → SAE inference (layer {args.layer})...")
    t0 = time.time()

    sae_activations = extract_sae_activations(
        sequences=sequences,
        esm_model=esm_model,
        tokenizer=tokenizer,
        sae=sae,
        target_layer=args.layer,
        device=device,
        batch_size=args.batch_size,
        sae_type="ours",
    )

    actual_residues = sum(a.shape[0] for a in sae_activations)
    nz_per_residue = np.mean([np.count_nonzero(a, axis=1).mean() for a in sae_activations])
    print(f"  Processed {len(sae_activations)} proteins, {actual_residues:,} residues")
    print(f"  Nonzero features per residue (mean): {nz_per_residue:.1f}")
    print(f"  Time: {time.time() - t0:.1f}s")
    print()

    # Free GPU memory
    del esm_model, sae
    torch.cuda.empty_cache()

    # --- Step 4: Compute F1 alignment ---
    print("[4/4] Computing feature-annotation F1...")
    t0 = time.time()

    results = compute_feature_annotation_f1(
        annotations=annotations,
        sae_activations=sae_activations,
        d_sae=d_sae,
        threshold_percentile=95.0,
        min_positive_residues=50,
        save_firing_positions=args.save_firing_positions,
        max_firing_positions_per_feature=args.max_firing_positions_per_feature,
    )

    summary = summarize_results(results)
    elapsed = time.time() - t0
    print(f"  F1 computation time: {elapsed:.1f}s")
    print()

    # --- Report ---
    print("=" * 70)
    print(f"  RESULTS: ESM-2-3B Layer {args.layer} SAE (d={d_sae}, step {args.step})")
    print("=" * 70)
    print(f"  Total features:  {summary['total_features']}")
    print(f"  Alive features:  {summary['alive_features']}")
    print(f"  Dead features:   {summary['dead_features']}")
    dead_pct = summary['dead_features'] / summary['total_features'] * 100
    alive_pct = summary['alive_features'] / summary['total_features'] * 100
    print(f"  Alive%: {alive_pct:.1f}%, Dead%: {dead_pct:.1f}%")
    print()

    print("  Classification (alive features):")
    for cls in ["KNOWN", "PARTIAL", "NOVEL"]:
        count = summary["classification"].get(cls, 0)
        pct = summary["classification_pct"].get(cls, 0)
        print(f"    {cls:8s}: {count:5d} ({pct:5.1f}%)")
    print()
    print(f"  Mean best F1 (alive): {summary['mean_best_f1_alive']:.4f}")
    print()

    print("  Top 20 KNOWN features (F1 > 0.5):")
    for feat_idx, ann_name, f1 in summary["top_known_features"]:
        print(f"    Feature {feat_idx:5d}: F1={f1}  <- {ann_name}")
    print()

    # --- Comparison with InterProt baseline (if available) ---
    interprot_path = "results/annotation_alignment/interprot_l24_test.pkl"
    if os.path.exists(interprot_path):
        with open(interprot_path, "rb") as f:
            interprot_data = pickle.load(f)
        ip_summary = interprot_data["summary"]

        print("=" * 70)
        print("  COMPARISON: Our 3B SAE vs InterProt 650M (layer 24)")
        print("=" * 70)
        print(f"  {'Metric':<30s} {'InterProt 650M':>15s} {'Ours 3B L{}'.format(args.layer):>15s}")
        print(f"  {'-'*30} {'-'*15} {'-'*15}")
        print(f"  {'Total features':<30s} {ip_summary['total_features']:>15d} {summary['total_features']:>15d}")
        print(f"  {'Alive features':<30s} {ip_summary['alive_features']:>15d} {summary['alive_features']:>15d}")
        print(f"  {'Dead features':<30s} {ip_summary['dead_features']:>15d} {summary['dead_features']:>15d}")

        for cls in ["KNOWN", "PARTIAL", "NOVEL"]:
            ip_count = ip_summary["classification"].get(cls, 0)
            our_count = summary["classification"].get(cls, 0)
            ip_pct = ip_summary["classification_pct"].get(cls, 0)
            our_pct = summary["classification_pct"].get(cls, 0)
            print(f"  {cls + ' (count / %)':<30s} {ip_count:>7d} / {ip_pct:4.1f}% {our_count:>7d} / {our_pct:4.1f}%")

        print(f"  {'Mean best F1 (alive)':<30s} {ip_summary['mean_best_f1_alive']:>15.4f} {summary['mean_best_f1_alive']:>15.4f}")
        print()

    # --- Save results ---
    out_dir = "results/annotation_alignment"
    os.makedirs(out_dir, exist_ok=True)
    prefix = args.out_prefix or f"ours_3B_l{args.layer}_step{args.step}"
    out_path = os.path.join(out_dir, f"{prefix}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"results": results, "summary": summary, "args": vars(args)}, f)
    print(f"  Results saved to {out_path}")

    # Also save a human-readable JSON summary
    json_path = os.path.join(out_dir, f"{prefix}_summary.json")
    json_summary = {
        "model": "ESM-2-3B",
        "layer": args.layer,
        "step": args.step,
        "d_sae": d_sae,
        "num_proteins": len(annotations),
        "total_residues": actual_residues,
        "nonzero_per_residue": round(nz_per_residue, 1),
        "total_features": summary["total_features"],
        "alive_features": summary["alive_features"],
        "dead_features": summary["dead_features"],
        "classification": summary["classification"],
        "classification_pct": {k: round(v, 2) for k, v in summary["classification_pct"].items()},
        "mean_best_f1_alive": round(summary["mean_best_f1_alive"], 4),
        "top_known": summary["top_known_features"],
        "save_firing_positions": args.save_firing_positions,
        "max_firing_positions_per_feature": args.max_firing_positions_per_feature,
        "n_features_with_firing_positions": sum(
            1 for r in results if getattr(r, "firing_positions", None)
        ),
    }
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2)
    print(f"  JSON summary saved to {json_path}")


if __name__ == "__main__":
    main()
