#!/usr/bin/env python
"""Circuit analysis and feature interpretation for trained CLTs.

Performs three analyses:
1. Feature interpretation: find max-activating sequences and amino acid preferences
2. Circuit comparison: compare feature activations across different EC numbers (ZymCTRL)
3. Steering validation: generate with feature amplification/ablation

Usage:
    python scripts/06_circuit_analysis.py --gpu 4 --model zymctrl
    python scripts/06_circuit_analysis.py --gpu 4 --model protgpt2
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Must set before importing model_loader
os.environ["R2_MODEL_BASE_DIR"] = "/Data/public/models_R2"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml

from src.models.model_loader import load_model
from src.training.clt_trainer import CLTForTraining
from src.analysis.circuit_discovery import (
    load_trained_clt,
    compute_feature_attribution,
    compare_circuits,
    steer_generation,
)


CHECKPOINT_MAP = {
    "protgpt2": "results/final_checkpoints/r2_clt_protgpt2_rerun_20260403/clt_weights/protgpt2/step_100000",
    "zymctrl": "results/final_checkpoints/r2_clt_zymctrl_rerun_20260403/clt_weights/zymctrl/step_100000",
    "progen2-medium": "results/final_checkpoints/r2_clt_progen2_medium_rerun_20260403/clt_weights/progen2-medium/step_100000",
}

# EC numbers for ZymCTRL circuit comparison
# Chosen to span different enzyme classes and well-studied enzymes
EC_PROMPTS = {
    "lysozyme":       "3.2.1.17",    # Glycosidase (lysozyme)
    "trypsin":        "3.4.21.4",    # Serine protease
    "ADH":            "1.1.1.1",     # Alcohol dehydrogenase (oxidoreductase)
    "catalase":       "1.11.1.6",    # Catalase (peroxidase)
    "DNA_polymerase": "2.7.7.7",     # DNA polymerase (transferase)
    "lipase":         "3.1.1.3",     # Lipase (hydrolase)
    "kinase":         "2.7.11.1",    # Protein kinase
    "carbonic_anh":   "4.2.1.1",     # Carbonic anhydrase (lyase)
}


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def load_eval_sequences(fasta_path: str, num_seqs: int = 500, max_len: int = 256,
                        skip: int = 200000) -> list[str]:
    """Load held-out protein sequences for feature interpretation."""
    sequences = []
    current_seq = []
    skipped = 0
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    seq = "".join(current_seq)
                    if len(seq) <= max_len:
                        skipped += 1
                        if skipped > skip:
                            sequences.append(seq)
                            if len(sequences) >= num_seqs:
                                break
                current_seq = []
            else:
                current_seq.append(line)
    return sequences


@torch.no_grad()
def analyze_feature_preferences(
    protein_model,
    clt: CLTForTraining,
    sequences: list[str],
    device: str,
    top_k_features: int = 50,
    batch_size: int = 2,
) -> dict:
    """Analyze what each CLT feature responds to.

    For each alive feature, compute:
    - Which amino acids it preferentially fires on
    - Mean/max activation across sequences
    - Example sequences where it's maximally active
    """
    tokenizer = protein_model.tokenizer
    n_layers = clt.n_layers
    d_clt = clt.d_clt

    # Track per-feature statistics
    feature_aa_counts = torch.zeros(n_layers, d_clt, 20, device=device)  # AA preferences
    feature_fire_counts = torch.zeros(n_layers, d_clt, device=device)
    feature_max_activation = torch.zeros(n_layers, d_clt, device=device)
    feature_sum_activation = torch.zeros(n_layers, d_clt, device=device)

    # Max-activating examples (tracked for top features only)
    max_examples = defaultdict(list)  # (layer, feat_idx) -> [(activation, seq_idx, pos)]

    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

    for si in range(0, len(sequences), batch_size):
        batch_seqs = sequences[si:si + batch_size]
        tokens = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        input_ids = tokens["input_ids"].to(device)

        cache = protein_model.get_activations(input_ids)
        resid_pre = [x.float() for x in cache.resid_pre]

        features = clt.encode(resid_pre)

        for bi, seq in enumerate(batch_seqs):
            for l in range(n_layers):
                feat = features[l][bi]  # (seq_len, d_clt)
                # Only look at positions that correspond to actual AAs
                seq_len = min(len(seq), feat.shape[0])

                for pos in range(seq_len):
                    aa = seq[pos]
                    aa_idx = aa_to_idx.get(aa, -1)
                    if aa_idx < 0:
                        continue

                    active = feat[pos] > 0
                    if not active.any():
                        continue

                    active_indices = active.nonzero(as_tuple=True)[0]
                    active_vals = feat[pos, active_indices]

                    feature_fire_counts[l, active_indices] += 1
                    feature_aa_counts[l, active_indices, aa_idx] += 1
                    feature_sum_activation[l, active_indices] += active_vals

                    # Update max activations
                    for fi, val in zip(active_indices.tolist(), active_vals.tolist()):
                        if val > feature_max_activation[l, fi].item():
                            feature_max_activation[l, fi] = val

        if (si // batch_size + 1) % 25 == 0:
            print(f"    Processed {si + len(batch_seqs)}/{len(sequences)} sequences")

    # Compute AA preferences (normalized by overall AA frequency)
    total_aa_counts = feature_aa_counts.sum(dim=1)  # (n_layers, 20)
    total_per_layer = total_aa_counts.sum(dim=1, keepdim=True).clamp(min=1)  # (n_layers, 1)
    aa_freq = total_aa_counts / total_per_layer  # (n_layers, 20) - background frequency

    # Per-feature AA preference (enrichment over background)
    feature_aa_freq = feature_aa_counts / feature_fire_counts.unsqueeze(-1).clamp(min=1)
    aa_enrichment = feature_aa_freq / aa_freq.unsqueeze(1).clamp(min=1e-6)  # (n_layers, d_clt, 20)

    # Find most informative features per layer
    results = {"layers": {}}

    for l in range(n_layers):
        alive_mask = feature_fire_counts[l] > 10  # fired at least 10 times
        n_alive = int(alive_mask.sum())

        if n_alive == 0:
            results["layers"][l] = {"n_alive": 0, "features": []}
            continue

        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        # Sort by total fire count (most active features)
        counts = feature_fire_counts[l, alive_indices]
        sorted_idx = counts.argsort(descending=True)
        top_indices = alive_indices[sorted_idx[:top_k_features]]

        layer_features = []
        for fi in top_indices.tolist():
            # Top amino acid preferences
            enrichments = aa_enrichment[l, fi].cpu().numpy()
            top_aa_idx = np.argsort(enrichments)[::-1][:5]
            top_aas = [(AMINO_ACIDS[i], float(enrichments[i])) for i in top_aa_idx]

            mean_act = (feature_sum_activation[l, fi] /
                        feature_fire_counts[l, fi].clamp(min=1)).item()

            layer_features.append({
                "feature_idx": fi,
                "fire_count": int(feature_fire_counts[l, fi].item()),
                "max_activation": float(feature_max_activation[l, fi].item()),
                "mean_activation": round(mean_act, 4),
                "top_aa_preferences": top_aas,
                "dominant_aa": top_aas[0][0] if top_aas else "?",
                "enrichment_score": round(float(enrichments[top_aa_idx[0]]), 2),
            })

        results["layers"][l] = {
            "n_alive": n_alive,
            "features": layer_features,
        }

    return results


@torch.no_grad()
def run_ec_circuit_comparison(
    protein_model,
    clt: CLTForTraining,
    ec_prompts: dict[str, str],
    device: str,
) -> dict:
    """Compare CLT circuits across different EC numbers in ZymCTRL.

    For each pair of EC numbers, identifies which features differentiate them.
    """
    print("\n  Running EC number circuit comparison...")

    # Get features for each EC prompt
    ec_features = {}
    for name, ec in ec_prompts.items():
        ids = protein_model.tokenize(ec)
        cache = protein_model.get_activations(ids)
        resid_pre = [x.float() for x in cache.resid_pre]
        features = clt.encode(resid_pre)

        # Mean feature activation across all positions
        mean_feats = [f.mean(dim=(0, 1)).cpu().numpy() for f in features]
        ec_features[name] = mean_feats
        print(f"    {name} ({ec}): extracted features")

    # Pairwise comparisons
    comparisons = []
    ec_names = list(ec_prompts.keys())
    for i in range(len(ec_names)):
        for j in range(i + 1, len(ec_names)):
            name_a, name_b = ec_names[i], ec_names[j]
            feats_a = ec_features[name_a]
            feats_b = ec_features[name_b]

            # Find most differential features across all layers
            diff_scores = []
            for l in range(clt.n_layers):
                diff = np.abs(feats_a[l] - feats_b[l])
                for fi in range(clt.d_clt):
                    if diff[fi] > 0.01:
                        diff_scores.append((l, fi, float(diff[fi]),
                                            float(feats_a[l][fi]),
                                            float(feats_b[l][fi])))

            # Sort by difference magnitude
            diff_scores.sort(key=lambda x: -x[2])
            top_diff = diff_scores[:20]

            comparisons.append({
                "ec_a": f"{name_a} ({ec_prompts[name_a]})",
                "ec_b": f"{name_b} ({ec_prompts[name_b]})",
                "n_differential_features": len(diff_scores),
                "top_differential": [
                    {
                        "layer": l,
                        "feature": fi,
                        "diff": round(d, 4),
                        f"activation_{name_a}": round(a, 4),
                        f"activation_{name_b}": round(b, 4),
                    }
                    for l, fi, d, a, b in top_diff
                ],
            })

    # Feature clustering: which features are shared vs specific
    # A feature is "EC-specific" if it fires for one EC but not others
    shared_features = defaultdict(int)  # (layer, feat) -> count of ECs
    specific_features = defaultdict(list)  # ec_name -> [(layer, feat, activation)]

    for name in ec_names:
        for l in range(clt.n_layers):
            active = ec_features[name][l] > 0.01
            for fi in np.where(active)[0]:
                shared_features[(l, int(fi))] += 1

    for name in ec_names:
        for l in range(clt.n_layers):
            for fi in range(clt.d_clt):
                val = ec_features[name][l][fi]
                if val > 0.01 and shared_features[(l, fi)] == 1:
                    specific_features[name].append(
                        (l, fi, float(val))
                    )

    # Sort specific features by activation
    for name in specific_features:
        specific_features[name].sort(key=lambda x: -x[2])

    return {
        "pairwise_comparisons": comparisons[:10],  # Top 10 pairs
        "shared_feature_distribution": {
            f"shared_by_{k}_ecs": sum(1 for v in shared_features.values() if v == k)
            for k in range(1, len(ec_names) + 1)
        },
        "ec_specific_features": {
            name: [{"layer": l, "feature": fi, "activation": round(a, 4)}
                   for l, fi, a in feats[:10]]
            for name, feats in specific_features.items()
        },
    }


@torch.no_grad()
def run_steering_experiments(
    protein_model,
    clt: CLTForTraining,
    device: str,
    model_name: str,
) -> dict:
    """Run steering experiments: ablate/amplify features and measure effect.

    For ZymCTRL: steer between enzyme classes.
    For unconditional models: steer based on top features.
    """
    print("\n  Running steering experiments...")

    results = []

    if model_name == "zymctrl":
        # Steer from one EC to another by finding differential features
        ec_a = "3.2.1.17"  # lysozyme
        ec_b = "1.1.1.1"   # ADH

        # Find features specific to EC A
        ids_a = protein_model.tokenize(ec_a)
        ids_b = protein_model.tokenize(ec_b)
        cache_a = protein_model.get_activations(ids_a)
        cache_b = protein_model.get_activations(ids_b)
        feats_a = clt.encode([x.float() for x in cache_a.resid_pre])
        feats_b = clt.encode([x.float() for x in cache_b.resid_pre])

        # Find top differential features
        diff_features = []
        for l in range(clt.n_layers):
            mean_a = feats_a[l].mean(dim=(0, 1))
            mean_b = feats_b[l].mean(dim=(0, 1))
            diff = mean_a - mean_b
            for fi in range(clt.d_clt):
                if abs(diff[fi].item()) > 0.1:
                    diff_features.append((l, fi, diff[fi].item()))

        diff_features.sort(key=lambda x: -abs(x[2]))
        top_diff = diff_features[:5]

        # Generate baseline from EC A
        baseline = protein_model.generate(ec_a, max_new_tokens=50, temperature=0.8)
        print(f"    Baseline ({ec_a}): {baseline[:80]}...")

        # Steer: amplify top features that differentiate EC A from EC B
        if top_diff:
            interventions = [(l, fi, 3.0) for l, fi, _ in top_diff if _ > 0]
            if interventions:
                try:
                    steered = steer_generation(
                        protein_model, clt, ec_a,
                        interventions=interventions,
                        max_new_tokens=50, temperature=0.8,
                    )
                    print(f"    Steered (amplify {ec_a} features): {steered[:80]}...")
                    results.append({
                        "experiment": f"amplify_{ec_a}_features",
                        "prompt": ec_a,
                        "interventions": [(l, fi, 3.0) for l, fi, _ in top_diff[:5] if _ > 0],
                        "baseline": baseline[:200],
                        "steered": steered[:200],
                    })
                except Exception as e:
                    print(f"    Steering failed: {e}")

            # Steer: suppress to make it more like EC B
            suppressions = [(l, fi, 0.0) for l, fi, _ in top_diff if _ > 0]
            if suppressions:
                try:
                    suppressed = steer_generation(
                        protein_model, clt, ec_a,
                        interventions=suppressions,
                        max_new_tokens=50, temperature=0.8,
                    )
                    print(f"    Steered (suppress {ec_a} features): {suppressed[:80]}...")
                    results.append({
                        "experiment": f"suppress_{ec_a}_features",
                        "prompt": ec_a,
                        "interventions": suppressions[:5],
                        "baseline": baseline[:200],
                        "steered": suppressed[:200],
                    })
                except Exception as e:
                    print(f"    Steering failed: {e}")
    else:
        # Unconditional model: generate baseline, then steer
        seed_seq = "M"
        baseline = protein_model.generate(seed_seq, max_new_tokens=50, temperature=0.8)
        print(f"    Baseline: {baseline[:80]}...")

        # Find most active features in baseline generation
        ids = protein_model.tokenize(seed_seq)
        cache = protein_model.get_activations(ids)
        features = clt.encode([x.float() for x in cache.resid_pre])

        # Get top active features
        active_features = []
        for l in range(clt.n_layers):
            mean_act = features[l].mean(dim=(0, 1))
            for fi in range(clt.d_clt):
                if mean_act[fi] > 0.1:
                    active_features.append((l, fi, mean_act[fi].item()))
        active_features.sort(key=lambda x: -x[2])

        # Ablate top features
        if active_features:
            ablations = [(l, fi, 0.0) for l, fi, _ in active_features[:5]]
            try:
                ablated = steer_generation(
                    protein_model, clt, seed_seq,
                    interventions=ablations,
                    max_new_tokens=50, temperature=0.8,
                )
                print(f"    Ablated (top 5): {ablated[:80]}...")
                results.append({
                    "experiment": "ablate_top5",
                    "baseline": baseline[:200],
                    "steered": ablated[:200],
                    "interventions": ablations,
                })
            except Exception as e:
                print(f"    Ablation failed: {e}")

    return {"steering_results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--model", type=str, default="zymctrl",
                        choices=["protgpt2", "zymctrl", "progen2-medium"])
    parser.add_argument("--num-seqs", type=int, default=300,
                        help="Sequences for feature interpretation")
    parser.add_argument("--fasta", type=str,
                        default="/Data/lzp/BioInterpretebility-CC/data/uniref50/uniref50.fasta")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    print("=" * 70)
    print(f"  Circuit Analysis: {args.model}")
    print("=" * 70)

    ckpt_dir = CHECKPOINT_MAP[args.model]
    print(f"  Checkpoint: {ckpt_dir}")
    print()

    # Load model and CLT
    print("[1] Loading model and CLT...")
    t0 = time.time()
    protein_model = load_model(args.model, device=device, dtype=torch.float16)
    clt = load_trained_clt(ckpt_dir, device=device)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  n_layers={clt.n_layers}, d_model={clt.d_model}, d_clt={clt.d_clt}")

    all_results = {"model": args.model}

    # Phase 1: Feature interpretation
    print(f"\n[2] Feature interpretation ({args.num_seqs} sequences)...")
    t0 = time.time()
    sequences = load_eval_sequences(args.fasta, num_seqs=args.num_seqs)
    feature_results = analyze_feature_preferences(
        protein_model, clt, sequences, device,
        top_k_features=30, batch_size=2,
    )
    print(f"  Feature analysis done ({time.time()-t0:.1f}s)")

    # Summary
    total_alive = sum(v["n_alive"] for v in feature_results["layers"].values())
    print(f"\n  Total alive features across all layers: {total_alive}")

    # Show interesting features from middle layers
    print(f"\n  Sample features from key layers:")
    for l in [clt.n_layers // 4, clt.n_layers // 2, 3 * clt.n_layers // 4]:
        if l < clt.n_layers and feature_results["layers"][l]["features"]:
            feats = feature_results["layers"][l]["features"][:5]
            print(f"\n    Layer {l} ({l/clt.n_layers*100:.0f}% depth, "
                  f"{feature_results['layers'][l]['n_alive']} alive):")
            for feat in feats:
                top_aa = feat["top_aa_preferences"][:3]
                aa_str = ", ".join(f"{aa}({e:.1f}x)" for aa, e in top_aa)
                print(f"      Feature {feat['feature_idx']:4d}: "
                      f"fires={feat['fire_count']:5d}, "
                      f"mean_act={feat['mean_activation']:.3f}, "
                      f"top_AA=[{aa_str}]")

    all_results["feature_interpretation"] = feature_results

    # Phase 2: EC comparison (ZymCTRL only)
    if args.model == "zymctrl":
        print(f"\n[3] EC number circuit comparison...")
        t0 = time.time()
        ec_results = run_ec_circuit_comparison(
            protein_model, clt, EC_PROMPTS, device
        )
        print(f"  EC comparison done ({time.time()-t0:.1f}s)")

        # Summary
        print(f"\n  Feature sharing across EC numbers:")
        for k, v in ec_results["shared_feature_distribution"].items():
            print(f"    {k}: {v} features")

        print(f"\n  EC-specific features (top per EC):")
        for ec_name, feats in ec_results["ec_specific_features"].items():
            if feats:
                print(f"    {ec_name}: {len(feats)} specific features, "
                      f"top at layer {feats[0]['layer']} feat {feats[0]['feature']}")

        # Show top pairwise differences
        if ec_results["pairwise_comparisons"]:
            comp = ec_results["pairwise_comparisons"][0]
            print(f"\n  Top comparison: {comp['ec_a']} vs {comp['ec_b']}")
            print(f"    {comp['n_differential_features']} differential features")
            for d in comp["top_differential"][:5]:
                print(f"      Layer {d['layer']}, Feature {d['feature']}: diff={d['diff']:.4f}")

        all_results["ec_comparison"] = ec_results

    # Phase 3: Steering experiments
    print(f"\n[{'4' if args.model == 'zymctrl' else '3'}] Steering experiments...")
    t0 = time.time()
    steering_results = run_steering_experiments(
        protein_model, clt, device, args.model
    )
    print(f"  Steering done ({time.time()-t0:.1f}s)")
    all_results["steering"] = steering_results

    # Save results
    out_dir = f"results/circuit_analysis/{args.model}"
    os.makedirs(out_dir, exist_ok=True)

    # Save feature interpretation (can be large)
    with open(os.path.join(out_dir, "feature_interpretation.json"), "w") as f:
        json.dump(feature_results, f, indent=2)

    # Save circuit comparison and steering
    summary = {
        "model": args.model,
        "n_layers": clt.n_layers,
        "d_clt": clt.d_clt,
        "total_alive_features": total_alive,
    }
    if "ec_comparison" in all_results:
        summary["ec_comparison"] = all_results["ec_comparison"]
    if "steering" in all_results:
        summary["steering"] = all_results["steering"]

    with open(os.path.join(out_dir, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Results saved to {out_dir}/")
    print("  Done!")


if __name__ == "__main__":
    main()
