#!/usr/bin/env python
"""Multi-layer annotation alignment analysis for ESM-2-3B SAEs.

Runs feature-annotation F1 on all available trained SAE checkpoints,
producing a cross-layer comparison showing how feature interpretability
changes with network depth.

Usage:
    python scripts/05_analyze_all_layers.py --gpu 1 --num-proteins 2000
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


# Checkpoint locations for each layer
CHECKPOINT_DIRS = {
    # Local L20-trained
    3:  "results/sae_weights/layer_3/step_30000",
    7:  "results/sae_weights/layer_7/step_30000",
    # H200-trained (full 500K steps)
    19: "results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights/layer_19/step_500000",
    23: "results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights/layer_23/step_500000",
    27: "results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights/layer_27/step_500000",
    31: "results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights/layer_31/step_500000",
    35: "results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights/layer_35/step_500000",
}


def analyze_layer(
    layer: int,
    checkpoint_path: str,
    annotations: list,
    sequences: list[str],
    esm_model,
    tokenizer,
    device: str,
    batch_size: int,
    out_dir: str,
) -> dict:
    """Run annotation alignment for a single layer."""
    print(f"\n{'='*70}")
    print(f"  Layer {layer} ({layer/36*100:.0f}% depth)")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*70}")

    # Load SAE
    t0 = time.time()
    sae = load_our_sae(checkpoint_path, device=device)
    d_sae = sae.d_sae
    print(f"  SAE loaded: d_sae={d_sae}, k={sae.k} ({time.time()-t0:.1f}s)")

    # Extract activations
    t0 = time.time()
    sae_activations = extract_sae_activations(
        sequences=sequences,
        esm_model=esm_model,
        tokenizer=tokenizer,
        sae=sae,
        target_layer=layer,
        device=device,
        batch_size=batch_size,
        sae_type="ours",
    )
    actual_residues = sum(a.shape[0] for a in sae_activations)
    nz_per_residue = np.mean([np.count_nonzero(a, axis=1).mean() for a in sae_activations])
    print(f"  Activations extracted: {actual_residues:,} residues, "
          f"{nz_per_residue:.1f} nonzero/residue ({time.time()-t0:.1f}s)")

    # Free SAE GPU memory
    del sae
    torch.cuda.empty_cache()

    # Compute F1
    t0 = time.time()
    results = compute_feature_annotation_f1(
        annotations=annotations,
        sae_activations=sae_activations,
        d_sae=d_sae,
        threshold_percentile=95.0,
        min_positive_residues=50,
    )
    summary = summarize_results(results)
    print(f"  F1 computed ({time.time()-t0:.1f}s)")

    # Print results
    alive = summary["alive_features"]
    dead = summary["dead_features"]
    total = summary["total_features"]
    print(f"\n  Alive: {alive}/{total} ({alive/total*100:.1f}%)")
    print(f"  Dead:  {dead}/{total} ({dead/total*100:.1f}%)")
    for cls in ["KNOWN", "PARTIAL", "NOVEL"]:
        count = summary["classification"].get(cls, 0)
        pct = summary["classification_pct"].get(cls, 0)
        print(f"  {cls:8s}: {count:5d} ({pct:5.1f}%)")
    print(f"  Mean best F1 (alive): {summary['mean_best_f1_alive']:.4f}")

    if summary["top_known_features"]:
        print(f"\n  Top KNOWN features:")
        for feat_idx, ann_name, f1 in summary["top_known_features"][:10]:
            print(f"    Feature {feat_idx:5d}: F1={f1}  <- {ann_name}")

    # Save
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, f"ours_3B_l{layer}_step500000.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({"results": results, "summary": summary}, f)

    json_summary = {
        "model": "ESM-2-3B",
        "layer": layer,
        "depth_pct": round(layer / 36 * 100, 1),
        "d_sae": d_sae,
        "num_proteins": len(annotations),
        "total_residues": actual_residues,
        "nonzero_per_residue": round(nz_per_residue, 1),
        "total_features": total,
        "alive_features": alive,
        "dead_features": dead,
        "alive_pct": round(alive / total * 100, 1),
        "classification": summary["classification"],
        "classification_pct": {k: round(v, 2) for k, v in summary["classification_pct"].items()},
        "mean_best_f1_alive": round(summary["mean_best_f1_alive"], 4),
        "top_known": summary["top_known_features"][:20],
    }
    json_path = os.path.join(out_dir, f"ours_3B_l{layer}_step500000_summary.json")
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2)

    return json_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--num-proteins", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--layers", type=str, default="19,23,27,31,35",
                        help="Comma-separated layers to analyze")
    parser.add_argument("--esm-model", type=str,
                        default="/Data/public/esm2_t36_3B_UR50D")
    parser.add_argument("--cache-path", type=str,
                        default="../data/processed/swissprot_all_max1022.pkl")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    layers = [int(x) for x in args.layers.split(",")]

    print("=" * 70)
    print("  Multi-Layer Annotation Alignment: ESM-2-3B SAEs")
    print("=" * 70)
    print(f"  GPU: {args.gpu}")
    print(f"  Layers: {layers}")
    print(f"  Num proteins: {args.num_proteins}")
    print()

    # Load annotations (shared across all layers)
    print("[1] Loading Swiss-Prot annotations...")
    t0 = time.time()
    with open(args.cache_path, "rb") as f:
        all_annotations = pickle.load(f)
    annotated = [a for a in all_annotations if a.features]
    rng = np.random.RandomState(42)
    indices = rng.choice(len(annotated), min(args.num_proteins, len(annotated)), replace=False)
    annotations = [annotated[i] for i in sorted(indices)]
    sequences = [a.sequence for a in annotations]
    total_residues = sum(len(s) for s in sequences)
    print(f"  {len(annotations)} proteins, {total_residues:,} residues ({time.time()-t0:.1f}s)")

    # Load ESM-2-3B (shared across all layers)
    print("\n[2] Loading ESM-2-3B...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
    esm_model = EsmModel.from_pretrained(args.esm_model, torch_dtype=torch.float16)
    esm_model.to(device).eval()
    print(f"  ESM-2-3B loaded ({time.time()-t0:.1f}s)")

    # Analyze each layer
    out_dir = "results/annotation_alignment"
    all_summaries = {}

    for layer in layers:
        if layer not in CHECKPOINT_DIRS:
            print(f"\n  Skipping layer {layer}: no checkpoint found")
            continue
        ckpt = CHECKPOINT_DIRS[layer]
        if not os.path.exists(os.path.join(ckpt, "sae.pt")):
            print(f"\n  Skipping layer {layer}: checkpoint not found at {ckpt}")
            continue

        summary = analyze_layer(
            layer=layer,
            checkpoint_path=ckpt,
            annotations=annotations,
            sequences=sequences,
            esm_model=esm_model,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.batch_size,
            out_dir=out_dir,
        )
        all_summaries[layer] = summary

    # Free ESM-2
    del esm_model
    torch.cuda.empty_cache()

    # Cross-layer comparison table
    print("\n\n" + "=" * 90)
    print("  CROSS-LAYER COMPARISON")
    print("=" * 90)
    print(f"  {'Layer':<8s} {'Depth%':<8s} {'Alive%':<8s} {'KNOWN':<8s} {'PARTIAL':<10s} "
          f"{'NOVEL':<8s} {'Mean F1':<10s} {'Top Annotation'}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*30}")

    for layer in sorted(all_summaries.keys()):
        s = all_summaries[layer]
        top_ann = s["top_known"][0][1] if s["top_known"] else "—"
        top_f1 = s["top_known"][0][2] if s["top_known"] else "—"
        print(
            f"  {layer:<8d} {s['depth_pct']:<8.1f} {s['alive_pct']:<8.1f} "
            f"{s['classification'].get('KNOWN', 0):<8d} "
            f"{s['classification'].get('PARTIAL', 0):<10d} "
            f"{s['classification'].get('NOVEL', 0):<8d} "
            f"{s['mean_best_f1_alive']:<10.4f} "
            f"{top_ann[:30]}"
        )

    print()

    # Save cross-layer summary
    cross_layer_path = os.path.join(out_dir, "cross_layer_summary.json")
    with open(cross_layer_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"  Cross-layer summary saved to {cross_layer_path}")

    # Analysis of annotation type distribution across layers
    print("\n" + "=" * 70)
    print("  ANNOTATION TYPE PROGRESSION ACROSS DEPTH")
    print("=" * 70)

    for layer in sorted(all_summaries.keys()):
        s = all_summaries[layer]
        if not s["top_known"]:
            print(f"\n  Layer {layer}: No KNOWN features")
            continue
        print(f"\n  Layer {layer} ({s['depth_pct']:.0f}% depth) — {s['classification'].get('KNOWN',0)} KNOWN features:")

        # Categorize KNOWN features by annotation type
        categories = {}
        for feat_idx, ann_name, f1 in s["top_known"]:
            cat = ann_name.split("/")[0] if "/" in ann_name else ann_name
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((feat_idx, ann_name, f1))

        for cat, feats in sorted(categories.items(), key=lambda x: -len(x[1])):
            print(f"    {cat}: {len(feats)} features")
            for feat_idx, ann_name, f1 in feats[:3]:
                print(f"      Feature {feat_idx}: F1={f1} <- {ann_name}")


if __name__ == "__main__":
    main()
