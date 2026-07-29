#!/usr/bin/env python
"""Layer-by-layer EC number specificity analysis for ZymCTRL CLT.

Analyzes how enzyme class information flows through the ZymCTRL transformer:
  - Which layers encode the most EC-specific features?
  - Does EC specificity increase with depth (early = universal, deep = specific)?
  - How does feature specialization relate to the model's generation process?

This reveals the computational structure: ZymCTRL likely encodes
universal protein generation rules in early layers and enzyme-class-specific
sequence patterns in deep layers.

No GPU required — operates on saved circuit analysis data, or optionally
re-runs feature extraction with --recompute flag.

Usage:
    python scripts/07_layer_ec_specificity.py
    python scripts/07_layer_ec_specificity.py --recompute --gpu 4
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

# Must set before importing model_loader
os.environ["R2_MODEL_BASE_DIR"] = "/Data/public/models_R2"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


EC_PROMPTS = {
    "lysozyme":       "3.2.1.17",
    "trypsin":        "3.4.21.4",
    "ADH":            "1.1.1.1",
    "catalase":       "1.11.1.6",
    "DNA_polymerase": "2.7.7.7",
    "lipase":         "3.1.1.3",
    "kinase":         "2.7.11.1",
    "carbonic_anh":   "4.2.1.1",
}


def compute_ec_features(gpu: int):
    """Recompute EC features from model (requires GPU)."""
    import torch
    from src.models.model_loader import load_model
    from src.training.clt_trainer import CLTForTraining
    from src.analysis.circuit_discovery import load_trained_clt

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = "cuda:0"

    ckpt_path = "results/final_checkpoints/r2_clt_zymctrl_rerun_20260403/clt_weights/zymctrl/step_100000"
    protein_model = load_model("zymctrl", device=device)
    clt = load_trained_clt(ckpt_path, device=device)

    ec_features = {}
    with torch.no_grad():
        for name, ec in EC_PROMPTS.items():
            ids = protein_model.tokenize(ec)
            cache = protein_model.get_activations(ids)
            resid_pre = [x.float() for x in cache.resid_pre]
            features = clt.encode(resid_pre)

            # Per-layer mean features, plus per-position for richer analysis
            mean_feats = [f.mean(dim=(0, 1)).cpu().numpy() for f in features]
            # Also get per-position for position analysis
            pos_feats = [f[0].cpu().numpy() for f in features]  # (seq_len, d_clt)
            ec_features[name] = {
                "mean": mean_feats,
                "per_position": pos_feats,
                "n_layers": len(mean_feats),
                "d_clt": mean_feats[0].shape[0],
            }
            print(f"  {name} ({ec}): extracted features, prompt len={pos_feats[0].shape[0]}")

    return ec_features


def analyze_from_summary(summary_path: str, n_layers: int = 36, d_clt: int = 4096):
    """Analyze using saved analysis_summary.json data."""
    with open(summary_path) as f:
        data = json.load(f)

    # We have shared_feature_distribution and ec_specific_features
    # but need per-layer breakdown. Let's extract what we can.
    ec_comparison = data["ec_comparison"]

    # From pairwise comparisons, extract layer distribution
    layer_diff_counts = defaultdict(int)
    for comp in ec_comparison["pairwise_comparisons"]:
        for feat in comp["top_differential"]:
            layer_diff_counts[feat["layer"]] += 1

    # From EC-specific features, extract layer distribution
    layer_specific_counts = defaultdict(int)
    for ec_name, feats in ec_comparison["ec_specific_features"].items():
        for feat in feats:
            layer_specific_counts[feat["layer"]] += 1

    return {
        "layer_diff_counts": dict(layer_diff_counts),
        "layer_specific_counts": dict(layer_specific_counts),
    }


def run_full_analysis(ec_features: dict):
    """Run comprehensive layer-by-layer EC specificity analysis."""
    ec_names = list(ec_features.keys())
    n_ecs = len(ec_names)
    first_ec = ec_features[ec_names[0]]
    n_layers = first_ec["n_layers"]
    d_clt = first_ec["d_clt"]

    print(f"\n{'='*70}")
    print(f"  Layer-by-Layer EC Specificity Analysis")
    print(f"{'='*70}")
    print(f"  Model: ZymCTRL (36L, 1280d)")
    print(f"  CLT: d_clt={d_clt}, 8 EC numbers")
    print(f"  EC classes: {', '.join(ec_names)}")

    # === Analysis 1: Per-layer feature activity and specificity ===
    print(f"\n  --- Per-Layer Feature Statistics ---")
    print(f"  {'Layer':>5s} {'Alive':>6s} {'EC-specific':>12s} {'Universal':>10s} "
          f"{'Specificity':>12s} {'Mean Act':>10s}")
    print(f"  {'-'*60}")

    layer_stats = []
    for l in range(n_layers):
        # Build feature matrix: (n_ecs, d_clt)
        feat_matrix = np.stack([ec_features[name]["mean"][l] for name in ec_names])

        # Alive = fires for at least one EC
        alive_mask = feat_matrix.max(axis=0) > 0.01
        n_alive = int(alive_mask.sum())

        # Universal = fires for all ECs
        universal_mask = (feat_matrix > 0.01).all(axis=0)
        n_universal = int(universal_mask.sum())

        # EC-specific = fires for exactly one EC
        ec_counts = (feat_matrix > 0.01).sum(axis=0)
        specific_mask = ec_counts == 1
        n_specific = int(specific_mask.sum())

        # Specificity score: coefficient of variation across ECs
        # High CV = different ECs activate different features = high specificity
        if n_alive > 0:
            alive_feats = feat_matrix[:, alive_mask]
            cv_per_feat = alive_feats.std(axis=0) / (alive_feats.mean(axis=0) + 1e-6)
            mean_specificity = float(cv_per_feat.mean())
        else:
            mean_specificity = 0.0

        mean_act = float(feat_matrix[:, alive_mask].mean()) if alive_mask.any() else 0.0

        print(f"  {l:5d} {n_alive:6d} {n_specific:12d} {n_universal:10d} "
              f"{mean_specificity:12.3f} {mean_act:10.4f}")

        layer_stats.append({
            "layer": l,
            "n_alive": n_alive,
            "n_specific": n_specific,
            "n_universal": n_universal,
            "specificity_cv": round(mean_specificity, 4),
            "mean_activation": round(mean_act, 6),
        })

    # === Analysis 2: Layer groups ===
    print(f"\n  --- Layer Group Summary ---")
    groups = {
        "Early (0-8)": list(range(0, 9)),
        "Middle (9-17)": list(range(9, 18)),
        "Late-mid (18-26)": list(range(18, 27)),
        "Deep (27-35)": list(range(27, 36)),
    }

    print(f"  {'Group':<20s} {'Mean Alive':>10s} {'Mean Specific':>14s} "
          f"{'Mean Universal':>14s} {'Mean Spec CV':>12s}")
    print(f"  {'-'*72}")
    for gname, glayers in groups.items():
        stats = [layer_stats[l] for l in glayers]
        print(f"  {gname:<20s} "
              f"{np.mean([s['n_alive'] for s in stats]):10.0f} "
              f"{np.mean([s['n_specific'] for s in stats]):14.0f} "
              f"{np.mean([s['n_universal'] for s in stats]):14.0f} "
              f"{np.mean([s['specificity_cv'] for s in stats]):12.3f}")

    # === Analysis 3: EC pairwise similarity by layer ===
    print(f"\n  --- EC Pairwise Cosine Similarity by Layer Group ---")

    # Compute mean cosine similarity for each layer group
    for gname, glayers in groups.items():
        sim_matrix = np.zeros((n_ecs, n_ecs))
        for i in range(n_ecs):
            for j in range(i + 1, n_ecs):
                sims = []
                for l in glayers:
                    a = ec_features[ec_names[i]]["mean"][l]
                    b = ec_features[ec_names[j]]["mean"][l]
                    norm = np.linalg.norm(a) * np.linalg.norm(b)
                    if norm > 0:
                        sims.append(float(np.dot(a, b) / norm))
                if sims:
                    sim_matrix[i, j] = sim_matrix[j, i] = np.mean(sims)

        mean_sim = sim_matrix[np.triu_indices(n_ecs, k=1)].mean()
        min_sim = sim_matrix[np.triu_indices(n_ecs, k=1)].min()
        max_sim = sim_matrix[np.triu_indices(n_ecs, k=1)].max()
        print(f"  {gname:<20s}: mean={mean_sim:.4f}, min={min_sim:.4f}, max={max_sim:.4f}")

    # === Analysis 4: Most discriminating layers ===
    print(f"\n  --- Per-Layer Discrimination Power (8-way classification) ---")
    # For each layer, compute how well features separate the 8 EC classes
    # Use total pairwise L2 distance as a simple metric
    print(f"  {'Layer':>5s} {'Total L2':>10s} {'Max pair L2':>12s} "
          f"{'Min pair L2':>12s} {'Discrimination':>14s}")
    print(f"  {'-'*55}")

    layer_discrimination = []
    for l in range(n_layers):
        feat_matrix = np.stack([ec_features[name]["mean"][l] for name in ec_names])
        total_l2 = 0.0
        pair_l2s = []
        for i in range(n_ecs):
            for j in range(i + 1, n_ecs):
                d = np.linalg.norm(feat_matrix[i] - feat_matrix[j])
                total_l2 += d
                pair_l2s.append(d)

        n_pairs = len(pair_l2s)
        mean_l2 = total_l2 / max(n_pairs, 1)
        max_l2 = max(pair_l2s) if pair_l2s else 0
        min_l2 = min(pair_l2s) if pair_l2s else 0

        # Print only every 3rd layer for readability
        if l % 3 == 0 or l == n_layers - 1:
            print(f"  {l:5d} {total_l2:10.2f} {max_l2:12.4f} "
                  f"{min_l2:12.4f} {mean_l2:14.4f}")

        layer_discrimination.append({
            "layer": l,
            "total_l2": round(total_l2, 4),
            "mean_l2": round(mean_l2, 4),
            "max_l2": round(max_l2, 4),
            "min_l2": round(min_l2, 4),
        })

    # Find peak discrimination layers
    sorted_by_disc = sorted(layer_discrimination, key=lambda x: -x["total_l2"])
    print(f"\n  Top 5 most discriminating layers:")
    for i, s in enumerate(sorted_by_disc[:5]):
        print(f"    {i+1}. Layer {s['layer']}: total L2={s['total_l2']:.2f}")

    # === Analysis 5: Position-dependent specificity ===
    print(f"\n  --- EC Specificity by Position (ZymCTRL prompt tokens) ---")
    # ZymCTRL EC prompts are tokenized differently. Let's look at how
    # features evolve across positions in the prompt.

    # For the shortest EC prompt (e.g., "1.1.1.1" = ADH), analyze per-position
    ref_ec = "ADH"
    ref_feats = ec_features[ref_ec]["per_position"]  # list of (seq_len, d_clt)
    prompt_len = ref_feats[0].shape[0]
    print(f"  Reference: {ref_ec} (prompt length={prompt_len} tokens)")

    for l in [0, 9, 18, 27, 35]:
        if l >= n_layers:
            continue
        feats = ref_feats[l]  # (prompt_len, d_clt)
        n_active_per_pos = (feats > 0.01).sum(axis=1)
        print(f"    Layer {l:2d}: active features per position = "
              f"{', '.join(str(int(x)) for x in n_active_per_pos)}")

    # === Save results ===
    out_dir = "results/circuit_analysis/zymctrl"
    os.makedirs(out_dir, exist_ok=True)

    output = {
        "layer_stats": layer_stats,
        "layer_discrimination": layer_discrimination,
        "top_discriminating_layers": [s["layer"] for s in sorted_by_disc[:5]],
        "layer_group_summary": {},
    }

    for gname, glayers in groups.items():
        stats = [layer_stats[l] for l in glayers]
        output["layer_group_summary"][gname] = {
            "mean_alive": round(float(np.mean([s["n_alive"] for s in stats])), 1),
            "mean_specific": round(float(np.mean([s["n_specific"] for s in stats])), 1),
            "mean_universal": round(float(np.mean([s["n_universal"] for s in stats])), 1),
            "mean_specificity_cv": round(float(np.mean([s["specificity_cv"] for s in stats])), 4),
        }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(os.path.join(out_dir, "layer_ec_specificity.json"), "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"\n  Results saved to {out_dir}/layer_ec_specificity.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute features from model (requires GPU)")
    parser.add_argument("--gpu", type=int, default=4)
    args = parser.parse_args()

    if args.recompute:
        ec_features = compute_ec_features(args.gpu)
        # Save for future use
        import pickle
        out_path = "results/circuit_analysis/zymctrl/ec_features.pkl"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(ec_features, f)
        print(f"  Saved EC features to {out_path}")
    else:
        # Try to load saved features
        pkl_path = "results/circuit_analysis/zymctrl/ec_features.pkl"
        if os.path.exists(pkl_path):
            import pickle
            with open(pkl_path, "rb") as f:
                ec_features = pickle.load(f)
            print(f"  Loaded EC features from {pkl_path}")
        else:
            print(f"  No saved features found. Running with --recompute --gpu <N>")
            return

    run_full_analysis(ec_features)


if __name__ == "__main__":
    main()
