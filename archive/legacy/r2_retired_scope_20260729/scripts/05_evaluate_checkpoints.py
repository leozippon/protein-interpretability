#!/usr/bin/env python
"""Evaluate trained CLT checkpoints: dead features, FVU, feature statistics.

Loads each rerun checkpoint, runs forward pass on held-out sequences,
and reports quality metrics.

Usage:
    python scripts/05_evaluate_checkpoints.py --gpu 3
    python scripts/05_evaluate_checkpoints.py --gpu 3 --models protgpt2,zymctrl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must set before importing model_loader (reads env at import time)
os.environ.setdefault("R2_MODEL_BASE_DIR", "/Data/public/models_R2")

import numpy as np
import torch
import yaml

from src.models.model_loader import load_model
from src.training.clt_trainer import (
    CLTForTraining,
    _load_sequence_manifest,
    _sha256_file,
    _validate_attention_mask,
    _valid_token_rows,
)


CHECKPOINT_MAP = {
    "protgpt2": "results/final_checkpoints/r2_clt_protgpt2_rerun_20260403/clt_weights/protgpt2/step_100000",
    "zymctrl": "results/final_checkpoints/r2_clt_zymctrl_rerun_20260403/clt_weights/zymctrl/step_100000",
    "progen2-medium": "results/final_checkpoints/r2_clt_progen2_medium_rerun_20260403/clt_weights/progen2-medium/step_100000",
    "progen2-xlarge": "results/final_checkpoints/r2_clt_progen2_xlarge_rerun_20260403/clt_weights/progen2-xlarge/step_100000",
}


def load_clt_from_checkpoint(ckpt_dir: str, device: str) -> tuple[CLTForTraining, dict]:
    """Load CLT and its config from checkpoint."""
    ckpt_path = Path(ckpt_dir)
    with open(ckpt_path / "config.yaml") as f:
        config = yaml.safe_load(f)

    state = torch.load(ckpt_path / "clt.pt", map_location=device)
    n_layers = state["W_enc"].shape[0]
    d_model = state["W_enc"].shape[2]

    clt_cfg = config.get("clt", {})
    clt = CLTForTraining(
        n_layers=n_layers,
        d_model=d_model,
        d_clt=clt_cfg["d_clt"],
        k=clt_cfg["k"],
        window=clt_cfg.get("window", 8),
    )
    clt.load_state_dict(state)
    clt.to(device).eval()
    return clt, config


def load_eval_sequences(fasta_path: str, num_seqs: int = 500, max_len: int = 256,
                        skip: int = 200000) -> list[str]:
    """Load held-out protein sequences for evaluation."""
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
def evaluate_clt(
    protein_model,
    clt: CLTForTraining,
    sequences: list[str],
    device: str,
    batch_size: int = 2,
) -> dict:
    """Evaluate CLT on held-out sequences.

    Returns comprehensive quality metrics:
    - FVU per layer and mean
    - Dead feature fraction per layer and mean
    - L0 (active features per token)
    - Feature activation statistics
    - Cross-layer feature sharing
    """
    tokenizer = protein_model.tokenizer

    feature_fire_counts = torch.zeros(clt.n_layers, clt.d_clt, device=device)
    feature_activation_sums = torch.zeros(clt.n_layers, clt.d_clt, device=device)
    total_tokens = 0
    total_active = 0.0
    squared_error = [0.0] * clt.n_layers
    target_sum = [0.0] * clt.n_layers
    target_squared_sum = [0.0] * clt.n_layers
    target_elements = [0] * clt.n_layers
    mlp_norm_sum = [0.0] * clt.n_layers
    mlp_norm_count = [0] * clt.n_layers

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        tokens = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        batch_tokens = int(attention_mask.sum().item())
        total_tokens += batch_tokens

        cache = protein_model.get_activations(input_ids, attention_mask)
        resid_pre = [x.float() for x in cache.resid_pre]
        mlp_out = [x.float() for x in cache.mlp_out]
        valid_mask = _validate_attention_mask(attention_mask, resid_pre[0])

        # Per-layer metrics
        features = clt.encode(resid_pre)
        mlp_hat = clt.decode(features)

        for l in range(clt.n_layers):
            # FVU per layer
            diff = _valid_token_rows(mlp_hat[l] - mlp_out[l], valid_mask)
            target = _valid_token_rows(mlp_out[l], valid_mask)
            flat_diff = diff.reshape(-1).double()
            flat_target = target.reshape(-1).double()
            squared_error[l] += float(flat_diff.square().sum().item())
            target_sum[l] += float(flat_target.sum().item())
            target_squared_sum[l] += float(flat_target.square().sum().item())
            target_elements[l] += int(flat_target.numel())

            # MLP output norms
            norms = target.norm(dim=-1)
            mlp_norm_sum[l] += float(norms.sum().item())
            mlp_norm_count[l] += int(norms.numel())

            # Feature firing counts
            active = _valid_token_rows((features[l] > 0).float(), valid_mask)
            valid_features = _valid_token_rows(features[l], valid_mask)
            total_active += float(active.sum().item())
            fire_count = active.reshape(-1, clt.d_clt).sum(dim=0)
            feature_fire_counts[l] += fire_count
            feature_activation_sums[l] += valid_features.reshape(
                -1, clt.d_clt
            ).sum(dim=0)

    # Compute dead features from fire counts
    dead_threshold = total_tokens * 0.001  # fired on < 0.1% of tokens
    dead_per_layer = []
    alive_per_layer = []
    for l in range(clt.n_layers):
        dead_frac = (feature_fire_counts[l] < dead_threshold).float().mean().item()
        dead_per_layer.append(dead_frac)
        alive_per_layer.append(1.0 - dead_frac)

    # Feature statistics
    mean_activation_per_feature = feature_activation_sums / feature_fire_counts.clamp(min=1)

    fvu_per_layer = []
    for l in range(clt.n_layers):
        n = target_elements[l]
        if n < 2:
            raise ValueError(f"layer {l} has fewer than two held-out target elements")
        mse = squared_error[l] / n
        variance = (
            target_squared_sum[l] - target_sum[l] ** 2 / n
        ) / (n - 1)
        fvu_per_layer.append(mse / (variance + 1e-8))
    l0_mean = total_active / (total_tokens * clt.n_layers)

    # Cross-layer feature utilization
    features_ever_active = (feature_fire_counts > 0).float()  # (n_layers, d_clt)

    # Find features that fire across many layers (shared representations)
    # This doesn't apply directly since each layer has its own encoder,
    # but we can check if the same feature index is used across layers

    results = {
        "total_eval_tokens": total_tokens,
        "num_sequences": len(sequences),
        "n_layers": clt.n_layers,
        "d_clt": clt.d_clt,
        "k": clt.k,
        "fvu_mean": float(np.mean(fvu_per_layer)),
        "l0_mean": float(l0_mean),
        "dead_mean": float(np.mean(dead_per_layer)),
        "alive_mean": float(np.mean(alive_per_layer)),
        "dead_per_layer": dead_per_layer,
        "alive_per_layer": alive_per_layer,
        "fvu_per_layer": [float(x) for x in fvu_per_layer],
        "mlp_norm_per_layer": [
            mlp_norm_sum[l] / mlp_norm_count[l] for l in range(clt.n_layers)
        ],
        "alive_features_per_layer": [int(x.item()) for x in features_ever_active.sum(dim=1)],
        "mean_activation_global": float(mean_activation_per_feature[features_ever_active.bool()].mean().item()),
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--models", type=str, default="protgpt2,zymctrl,progen2-medium",
                        help="Comma-separated model names")
    parser.add_argument("--num-seqs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--fasta", type=str,
                        default="/Data/lzp/BioInterpretebility-CC/data/uniref50/uniref50.fasta")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Immutable cohort JSONL; preferred for confirmatory evaluation")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="Override checkpoint as model=/absolute/path (repeatable)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    models_to_eval = [m.strip() for m in args.models.split(",")]

    print("=" * 70)
    print("  CLT Checkpoint Evaluation")
    print("=" * 70)
    print(f"  GPU: {args.gpu}")
    print(f"  Models: {models_to_eval}")
    print(f"  Eval sequences: {args.num_seqs}")
    print()

    checkpoint_map = dict(CHECKPOINT_MAP)
    for value in args.checkpoint:
        if "=" not in value:
            parser.error("--checkpoint must have form model=/path")
        name, path = value.split("=", 1)
        checkpoint_map[name] = path

    # Load eval sequences
    print("Loading held-out sequences...")
    if args.manifest is not None:
        sequences = _load_sequence_manifest(args.manifest, args.split, args.num_seqs)
        cohort_metadata = {
            "path": str(args.manifest.resolve()),
            "split": args.split,
            "sha256": _sha256_file(args.manifest),
        }
    else:
        sequences = load_eval_sequences(args.fasta, num_seqs=args.num_seqs)
        cohort_metadata = {
            "path": str(Path(args.fasta).resolve()),
            "split": "legacy_skip_200000",
            "sha256": None,
        }
    print(f"  Loaded {len(sequences)} sequences")
    print()

    all_results = {}
    for model_name in models_to_eval:
        ckpt_dir = checkpoint_map.get(model_name)
        if not ckpt_dir or not os.path.exists(os.path.join(ckpt_dir, "clt.pt")):
            print(f"\n  Skipping {model_name}: checkpoint not found")
            continue

        print(f"\n{'='*70}")
        print(f"  Evaluating: {model_name}")
        print(f"  Checkpoint: {ckpt_dir}")
        print(f"{'='*70}")

        t0 = time.time()

        # Load protein model
        print(f"  Loading protein model...")
        protein_model = load_model(model_name, device=device, dtype=torch.float16)

        # Load CLT
        print(f"  Loading CLT checkpoint...")
        clt, config = load_clt_from_checkpoint(ckpt_dir, device)
        print(f"  CLT: n_layers={clt.n_layers}, d_model={clt.d_model}, "
              f"d_clt={clt.d_clt}, k={clt.k}, window={clt.window}")

        # Evaluate
        print(f"  Running evaluation on {len(sequences)} sequences...")
        results = evaluate_clt(protein_model, clt, sequences, device, args.batch_size)
        elapsed = time.time() - t0

        # Report
        print(f"\n  --- Results for {model_name} ({elapsed:.1f}s) ---")
        print(f"  FVU (mean):     {results['fvu_mean']:.4f} ({(1-results['fvu_mean'])*100:.1f}% variance explained)")
        print(f"  L0 (mean):      {results['l0_mean']:.1f}")
        print(f"  Dead (mean):    {results['dead_mean']:.3f} ({results['dead_mean']*100:.1f}%)")
        print(f"  Alive (mean):   {results['alive_mean']:.3f} ({results['alive_mean']*100:.1f}%)")
        print(f"  Alive features: {int(results['alive_mean'] * results['d_clt'])}/{results['d_clt']}")

        # Per-layer breakdown (sample layers)
        n = clt.n_layers
        sample_layers = [0, n//6, n//3, n//2, 2*n//3, 5*n//6, n-1]
        print(f"\n  Per-layer breakdown (sampled):")
        print(f"  {'Layer':<8s} {'FVU':<10s} {'Dead%':<10s} {'Alive':<10s} {'MLP norm':<10s}")
        for l in sample_layers:
            if l < n:
                print(f"  {l:<8d} {results['fvu_per_layer'][l]:<10.4f} "
                      f"{results['dead_per_layer'][l]*100:<10.1f} "
                      f"{results['alive_features_per_layer'][l]:<10d} "
                      f"{results['mlp_norm_per_layer'][l]:<10.2f}")

        all_results[model_name] = results
        all_results[model_name]["checkpoint"] = {
            "path": str(Path(ckpt_dir).resolve()),
            "clt_sha256": _sha256_file(Path(ckpt_dir) / "clt.pt"),
            "config_sha256": _sha256_file(Path(ckpt_dir) / "config.yaml"),
        }

        # Free memory
        del protein_model, clt
        torch.cuda.empty_cache()

    payload = {
        "schema_version": "r2_masked_clt_evaluation_v1",
        "cohort": cohort_metadata,
        "results": all_results,
    }
    if not all_results:
        raise RuntimeError("no requested checkpoints were evaluated")

    # Cross-model comparison
    print("\n\n" + "=" * 70)
    print("  CROSS-MODEL COMPARISON")
    print("=" * 70)
    print(f"  {'Model':<20s} {'FVU':<10s} {'Var Expl':<10s} {'Dead%':<10s} {'L0':<8s} {'Layers':<8s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for name in models_to_eval:
        if name in all_results:
            r = all_results[name]
            print(f"  {name:<20s} {r['fvu_mean']:<10.4f} "
                  f"{(1-r['fvu_mean'])*100:<10.1f}% "
                  f"{r['dead_mean']*100:<10.1f} "
                  f"{r['l0_mean']:<8.1f} {r['n_layers']:<8d}")

    # Save results
    out_path = args.output or Path("results/checkpoint_evaluation/rerun_evaluation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
