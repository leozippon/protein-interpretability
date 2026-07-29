#!/usr/bin/env python
"""Cross-model circuit conservation for R2.

Given a shared sequence set and multiple CLT checkpoints, this script asks:
do different generator models activate analogous sparse features on the same
biological sequences?

Implementation:
  1. Load sequences from FASTA or a JSON artifact (e.g. case-study leads).
  2. For each model+checkpoint pair, run the model and CLT to obtain per-layer
     mean feature activations for every sequence.
  3. Align requested layers by relative depth across models.
  4. Compare models using:
     - layer-level linear CKA on per-sequence feature matrices
     - greedy feature correspondence via activation-profile correlation

The feature-matching step is intentionally model-agnostic: it compares feature
activation profiles across the same sequence cohort, so models with different
hidden widths can still be compared.

Usage:
    python r2_interpretability_transfer/scripts/15_cross_model_conservation.py \
        --model-spec zymctrl=r2_interpretability_transfer/results/final_checkpoints/.../step_100000 \
        --model-spec progen2-medium=r2_interpretability_transfer/results/final_checkpoints/.../step_100000 \
        --json r2_interpretability_transfer/results/drug_design/ec_lysozyme_leads.json \
        --json-field leads \
        --out r2_interpretability_transfer/results/circuit_analysis/cross_model_conservation.json
"""

import argparse
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import yaml

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.analysis.circuit_discovery import load_trained_clt
from src.models.model_loader import load_model


AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")


def parse_model_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"bad --model-spec {spec!r}; expected name=checkpoint_dir")
    name, ckpt = spec.split("=", 1)
    return name.strip(), ckpt.strip()


def clean_sequence(seq: str) -> str:
    return "".join(c for c in seq.upper() if c in AA_ALPHABET)


def read_fasta(path: str, max_sequences: int, max_length: int) -> list[str]:
    out = []
    cur = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur:
                    seq = clean_sequence("".join(cur))[:max_length]
                    if seq:
                        out.append(seq)
                    if len(out) >= max_sequences:
                        return out
                cur = []
            else:
                cur.append(line)
        if cur and len(out) < max_sequences:
            seq = clean_sequence("".join(cur))[:max_length]
            if seq:
                out.append(seq)
    return out


def read_json_sequences(path: str, field: str, max_sequences: int,
                        max_length: int) -> list[str]:
    with open(path) as f:
        data = json.load(f)

    cur = data
    for token in field.split("."):
        if not token:
            continue
        if isinstance(cur, dict):
            cur = cur[token]
        else:
            raise KeyError(f"cannot descend into field {field!r}")

    seqs = []
    if isinstance(cur, list):
        for item in cur:
            seq = None
            if isinstance(item, str):
                seq = item
            elif isinstance(item, dict):
                for key in ("sequence", "steered", "unsteered"):
                    if isinstance(item.get(key), str):
                        seq = item[key]
                        break
            if seq:
                seq = clean_sequence(seq)[:max_length]
                if seq:
                    seqs.append(seq)
            if len(seqs) >= max_sequences:
                break
    return seqs


def load_sequences(args) -> list[str]:
    if args.fasta:
        return read_fasta(args.fasta, args.max_sequences, args.max_length)
    if args.json:
        return read_json_sequences(
            args.json, args.json_field, args.max_sequences, args.max_length
        )
    raise ValueError("provide --fasta or --json")


def checkpoint_n_layers(checkpoint_dir: str) -> int:
    state = torch.load(
        os.path.join(checkpoint_dir, "clt.pt"),
        map_location="cpu",
        weights_only=False,
    )
    return int(state["W_enc"].shape[0])


def map_anchor_layers(anchor_layers: list[int], anchor_n_layers: int,
                      target_n_layers: int) -> list[int]:
    if target_n_layers == anchor_n_layers:
        return list(anchor_layers)
    mapped = []
    for layer in anchor_layers:
        rel = layer / max(anchor_n_layers - 1, 1)
        mapped.append(int(round(rel * max(target_n_layers - 1, 1))))
    return mapped


def center_gram(x: np.ndarray) -> np.ndarray:
    k = x @ x.T
    n = k.shape[0]
    h = np.eye(n, dtype=np.float32) - np.ones((n, n), dtype=np.float32) / n
    return h @ k @ h


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[0] < 2 or y.shape[0] < 2:
        return float("nan")
    kx = center_gram(x.astype(np.float32))
    ky = center_gram(y.astype(np.float32))
    num = float((kx * ky).sum())
    den = float(np.linalg.norm(kx) * np.linalg.norm(ky) + 1e-8)
    return num / den


def greedy_feature_matches(x: np.ndarray, y: np.ndarray, top_k: int,
                           pool_size: int) -> list[dict]:
    if x.shape[0] < 3 or y.shape[0] < 3:
        return []
    x_var = x.var(axis=0)
    y_var = y.var(axis=0)
    x_idx = np.argsort(-x_var)[:min(pool_size, x.shape[1])]
    y_idx = np.argsort(-y_var)[:min(pool_size, y.shape[1])]
    xa = x[:, x_idx]
    yb = y[:, y_idx]
    xa = (xa - xa.mean(axis=0)) / (xa.std(axis=0) + 1e-6)
    yb = (yb - yb.mean(axis=0)) / (yb.std(axis=0) + 1e-6)
    # xa/yb are normalized with np.std(..., ddof=0), so the matching
    # Pearson correlation uses an n denominator. Using n-1 here inflates
    # small cohorts above 1.0, which makes the conservation metric invalid.
    corr = (xa.T @ yb) / max(xa.shape[0], 1)
    corr = np.clip(corr, -1.0, 1.0)
    score = np.abs(corr).copy()
    matches = []
    for _ in range(min(top_k, score.shape[0], score.shape[1])):
        flat = int(np.argmax(score))
        i, j = np.unravel_index(flat, score.shape)
        if not np.isfinite(score[i, j]):
            break
        matches.append({
            "feature_a": int(x_idx[i]),
            "feature_b": int(y_idx[j]),
            "corr": float(corr[i, j]),
            "abs_corr": float(abs(corr[i, j])),
        })
        score[i, :] = -np.inf
        score[:, j] = -np.inf
    return matches


@torch.no_grad()
def encode_sequences(model_name: str, checkpoint_dir: str, sequences: list[str],
                     anchor_layers: list[int], anchor_n_layers: int, device: str):
    pm = load_model(model_name, device=device)
    clt = load_trained_clt(checkpoint_dir, device=device)
    mapped_layers = map_anchor_layers(anchor_layers, anchor_n_layers, clt.n_layers)

    per_layer = {layer: [] for layer in mapped_layers}
    t0 = time.time()
    for i, seq in enumerate(sequences):
        ids = pm.tokenize(seq)
        cache = pm.get_activations(ids)
        feats = clt.encode([x.float() for x in cache.resid_pre])
        for layer in mapped_layers:
            per_layer[layer].append(feats[layer][0].mean(dim=0).cpu().numpy())
        if (i + 1) % 5 == 0 or i == len(sequences) - 1:
            print(f"    {model_name}: {i+1}/{len(sequences)} ({time.time()-t0:.1f}s)")

    matrices = {
        layer: np.stack(rows).astype(np.float32) if rows else np.zeros((0, clt.d_clt), dtype=np.float32)
        for layer, rows in per_layer.items()
    }
    del pm
    del clt
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return mapped_layers, matrices


def infer_model_name_from_config(checkpoint_dir: str) -> str:
    with open(os.path.join(checkpoint_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    return cfg["model"]["name"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-spec", action="append", required=True,
                    help="Repeated: model_name=checkpoint_dir")
    ap.add_argument("--fasta", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--json-field", default="leads")
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 12, 30])
    ap.add_argument("--max-sequences", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--top-feature-pairs", type=int, default=10)
    ap.add_argument("--feature-pool-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--out",
        default="r2_interpretability_transfer/results/circuit_analysis/cross_model_conservation.json",
    )
    args = ap.parse_args()

    print("=" * 70)
    print("  R2-H Cross-model circuit conservation")
    print("=" * 70)

    specs = [parse_model_spec(s) for s in args.model_spec]
    if len(specs) < 2:
        raise ValueError("need at least two --model-spec entries")

    sequences = load_sequences(args)
    if not sequences:
        raise ValueError("no sequences loaded from the provided input")
    print(f"  Loaded {len(sequences)} sequences")

    model_info = []
    anchor_name, anchor_ckpt = specs[0]
    anchor_name = anchor_name or infer_model_name_from_config(anchor_ckpt)
    anchor_n_layers = checkpoint_n_layers(anchor_ckpt)
    for name, ckpt in specs:
        true_name = name or infer_model_name_from_config(ckpt)
        n_layers = checkpoint_n_layers(ckpt)
        mapped_layers = map_anchor_layers(args.layers, anchor_n_layers, n_layers)
        model_info.append({
            "model": true_name,
            "checkpoint_dir": ckpt,
            "n_layers": n_layers,
            "mapped_layers": mapped_layers,
        })

    if args.dry_run:
        out = {
            "config": {
                "layers": args.layers,
                "max_sequences": len(sequences),
                "max_length": args.max_length,
                "device": args.device,
                "dry_run": True,
            },
            "models": model_info,
            "input": {
                "source": args.fasta or args.json,
                "source_type": "fasta" if args.fasta else "json",
                "json_field": args.json_field if args.json else None,
                "n_sequences": len(sequences),
                "lengths": [len(s) for s in sequences[:10]],
            },
        }
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Dry run complete. Saved: {args.out}")
        return

    encoded = {}
    for info in model_info:
        print(f"\n  Loading {info['model']}...")
        mapped_layers, matrices = encode_sequences(
            info["model"],
            info["checkpoint_dir"],
            sequences,
            args.layers,
            anchor_n_layers,
            args.device,
        )
        info["mapped_layers"] = mapped_layers
        info["layer_stats"] = {
            str(layer): {
                "shape": list(matrices[layer].shape),
                "mean_alive_fraction": float((matrices[layer] > 0).mean()),
                "mean_activation": float(matrices[layer].mean()),
            }
            for layer in mapped_layers
        }
        encoded[info["model"]] = matrices

    pairwise = []
    for a, b in combinations(model_info, 2):
        pair_rec = {
            "model_a": a["model"],
            "model_b": b["model"],
            "layers": [],
        }
        for anchor_layer, la, lb in zip(args.layers, a["mapped_layers"], b["mapped_layers"]):
            xa = encoded[a["model"]][la]
            yb = encoded[b["model"]][lb]
            matches = greedy_feature_matches(
                xa, yb, args.top_feature_pairs, args.feature_pool_size
            )
            pair_rec["layers"].append({
                "anchor_layer": int(anchor_layer),
                "layer_a": int(la),
                "layer_b": int(lb),
                "cka": float(linear_cka(xa, yb)),
                "top_feature_matches": matches,
                "mean_abs_match_corr": float(np.mean([m["abs_corr"] for m in matches])) if matches else float("nan"),
            })
        pairwise.append(pair_rec)

    out = {
        "config": {
            "layers": args.layers,
            "max_sequences": len(sequences),
            "max_length": args.max_length,
            "top_feature_pairs": args.top_feature_pairs,
            "feature_pool_size": args.feature_pool_size,
            "device": args.device,
        },
        "input": {
            "source": args.fasta or args.json,
            "source_type": "fasta" if args.fasta else "json",
            "json_field": args.json_field if args.json else None,
            "n_sequences": len(sequences),
        },
        "models": model_info,
        "pairwise": pairwise,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {args.out}")


if __name__ == "__main__":
    main()
