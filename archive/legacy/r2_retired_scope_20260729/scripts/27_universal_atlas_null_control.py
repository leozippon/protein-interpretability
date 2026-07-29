#!/usr/bin/env python3
"""Permutation null control for cross-model universal CLT features."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cm = load_module(SCRIPT_DIR / "15_cross_model_conservation.py", "cross_model_conservation_15")


def node(model: str, layer: int, feature: int) -> tuple[str, int, int]:
    return (model, int(layer), int(feature))


def count_triplets(data: dict, threshold: float) -> int:
    models = [m["model"] for m in data.get("models", [])]
    if len(models) != 3:
        return 0

    edges_by_layer: dict[int, dict[frozenset, float]] = {}
    nodes_by_layer_model: dict[tuple[int, str], set[tuple[str, int, int]]] = {}
    for pair in data.get("pairwise", []):
        ma, mb = pair["model_a"], pair["model_b"]
        for layer in pair.get("layers", []):
            anchor = int(layer["anchor_layer"])
            la, lb = int(layer["layer_a"]), int(layer["layer_b"])
            edges = edges_by_layer.setdefault(anchor, {})
            for match in layer.get("top_feature_matches", []):
                corr = float(match.get("abs_corr", 0.0))
                if corr < threshold:
                    continue
                na = node(ma, la, int(match["feature_a"]))
                nb = node(mb, lb, int(match["feature_b"]))
                edges[frozenset((na, nb))] = corr
                nodes_by_layer_model.setdefault((anchor, ma), set()).add(na)
                nodes_by_layer_model.setdefault((anchor, mb), set()).add(nb)

    n = 0
    for anchor, edges in edges_by_layer.items():
        model_nodes = [sorted(nodes_by_layer_model.get((anchor, m), set())) for m in models]
        if any(not xs for xs in model_nodes):
            continue
        for combo in product(*model_nodes):
            if all(frozenset((a, b)) in edges for a, b in combinations(combo, 2)):
                n += 1
    return n


def pairwise_from_encoded(model_info: list[dict], encoded: dict, anchor_layers: list[int],
                          top_k: int, pool_size: int, rng: np.random.Generator | None) -> list[dict]:
    pairwise = []
    for a, b in combinations(model_info, 2):
        pair_rec = {"model_a": a["model"], "model_b": b["model"], "layers": []}
        for anchor_layer, la, lb in zip(anchor_layers, a["mapped_layers"], b["mapped_layers"]):
            xa = encoded[a["model"]][la]
            yb = encoded[b["model"]][lb]
            if rng is not None:
                yb = yb[rng.permutation(yb.shape[0])]
            matches = cm.greedy_feature_matches(xa, yb, top_k, pool_size)
            pair_rec["layers"].append({
                "anchor_layer": int(anchor_layer),
                "layer_a": int(la),
                "layer_b": int(lb),
                "cka": float(cm.linear_cka(xa, yb)),
                "top_feature_matches": matches,
                "mean_abs_match_corr": float(np.mean([m["abs_corr"] for m in matches])) if matches else float("nan"),
            })
        pairwise.append(pair_rec)
    return pairwise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-spec", action="append", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--json-field", default="records")
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 12, 30])
    ap.add_argument("--max-sequences", type=int, default=200)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--top-feature-pairs", type=int, default=300)
    ap.add_argument("--feature-pool-size", type=int, default=4096)
    ap.add_argument("--n-null", type=int, default=3)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.90, 0.95, 0.98])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    specs = [cm.parse_model_spec(s) for s in args.model_spec]
    sequences = cm.read_json_sequences(args.json, args.json_field, args.max_sequences, args.max_length)
    if not sequences:
        raise ValueError("no sequences loaded")

    anchor_name, anchor_ckpt = specs[0]
    anchor_n_layers = cm.checkpoint_n_layers(anchor_ckpt)
    model_info = []
    encoded = {}
    for name, ckpt in specs:
        true_name = name or cm.infer_model_name_from_config(ckpt)
        n_layers = cm.checkpoint_n_layers(ckpt)
        mapped_layers = cm.map_anchor_layers(args.layers, anchor_n_layers, n_layers)
        info = {
            "model": true_name,
            "checkpoint_dir": ckpt,
            "n_layers": n_layers,
            "mapped_layers": mapped_layers,
        }
        model_info.append(info)
        print(f"Encoding {true_name}", flush=True)
        mapped, matrices = cm.encode_sequences(true_name, ckpt, sequences, args.layers, anchor_n_layers, args.device)
        info["mapped_layers"] = mapped
        encoded[true_name] = matrices

    observed_pairwise = pairwise_from_encoded(
        model_info, encoded, args.layers, args.top_feature_pairs, args.feature_pool_size, rng=None
    )
    observed = {
        "config": vars(args),
        "input": {"source": args.json, "n_sequences": len(sequences)},
        "models": model_info,
        "pairwise": observed_pairwise,
    }

    rng = np.random.default_rng(args.seed)
    null_runs = []
    for i in range(args.n_null):
        print(f"Null replicate {i + 1}/{args.n_null}", flush=True)
        pairwise = pairwise_from_encoded(
            model_info, encoded, args.layers, args.top_feature_pairs, args.feature_pool_size, rng=rng
        )
        payload = {"models": model_info, "pairwise": pairwise}
        null_runs.append({
            "replicate": i,
            "triplet_counts": {str(t): count_triplets(payload, t) for t in args.thresholds},
            "mean_pairwise_cka": float(np.nanmean([
                layer["cka"] for pair in pairwise for layer in pair.get("layers", [])
            ])),
            "mean_abs_match_corr": float(np.nanmean([
                layer["mean_abs_match_corr"] for pair in pairwise for layer in pair.get("layers", [])
            ])),
        })

    result = {
        "task": "F-T2-2 universal atlas permutation null control",
        "input": {"source": args.json, "n_sequences": len(sequences)},
        "config": {
            "layers": args.layers,
            "top_feature_pairs": args.top_feature_pairs,
            "feature_pool_size": args.feature_pool_size,
            "n_null": args.n_null,
            "thresholds": args.thresholds,
            "seed": args.seed,
        },
        "observed_triplet_counts": {str(t): count_triplets(observed, t) for t in args.thresholds},
        "null_runs": null_runs,
    }
    result["null_summary"] = {}
    for t in args.thresholds:
        vals = [r["triplet_counts"][str(t)] for r in null_runs]
        result["null_summary"][str(t)] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "max": int(np.max(vals)),
            "ci99_percentile": [
                float(np.percentile(vals, 0.5)),
                float(np.percentile(vals, 99.5)),
            ],
            "values": vals,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    md = out.with_suffix(".md")
    lines = [
        "# Universal Atlas Null Control",
        "",
        f"- Sequences: {len(sequences)}",
        f"- Null replicates: {args.n_null}",
        f"- Top feature pairs: {args.top_feature_pairs}",
        f"- Feature pool size: {args.feature_pool_size}",
        "",
        "| Threshold | Observed triplets | Null mean | Null std | Null 99% interval | Null max | Null values |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for t in args.thresholds:
        key = str(t)
        ns = result["null_summary"][key]
        lines.append(
            f"| {t:.2f} | {result['observed_triplet_counts'][key]} | "
            f"{ns['mean']:.2f} | {ns['std']:.2f} | "
            f"[{ns['ci99_percentile'][0]:.2f}, {ns['ci99_percentile'][1]:.2f}] | "
            f"{ns['max']} | {ns['values']} |"
        )
    lines.append("")
    md.write_text("\n".join(lines))
    print(f"Wrote {out}")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
