#!/usr/bin/env python3
"""Summarize cross-model CLT feature-conservation artifacts.

The upstream conservation script writes pairwise feature matches. This helper
turns that JSON into a compact table and, when three models are present,
extracts exact feature triplets supported by all three pairwise matches.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations, product
from pathlib import Path


def node(model: str, layer: int, feature: int) -> tuple[str, int, int]:
    return (model, int(layer), int(feature))


def fmt(x: float) -> str:
    if x is None or not math.isfinite(float(x)):
        return "nan"
    return f"{float(x):.4f}"


def summarize_pairs(data: dict, threshold: float) -> list[dict]:
    rows = []
    for pair in data.get("pairwise", []):
        for layer in pair.get("layers", []):
            matches = layer.get("top_feature_matches", [])
            vals = [float(m.get("abs_corr", 0.0)) for m in matches]
            rows.append({
                "model_a": pair["model_a"],
                "model_b": pair["model_b"],
                "anchor_layer": int(layer["anchor_layer"]),
                "layer_a": int(layer["layer_a"]),
                "layer_b": int(layer["layer_b"]),
                "cka": float(layer.get("cka", math.nan)),
                "mean_abs_match_corr": float(layer.get("mean_abs_match_corr", math.nan)),
                "n_matches": len(matches),
                f"n_abs_corr_ge_{threshold:.2f}": sum(v >= threshold for v in vals),
                "max_abs_corr": max(vals) if vals else math.nan,
            })
    return rows


def extract_triplets(data: dict, threshold: float) -> list[dict]:
    models = [m["model"] for m in data.get("models", [])]
    if len(models) != 3:
        return []

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

    triplets = []
    for anchor, edges in edges_by_layer.items():
        model_nodes = [sorted(nodes_by_layer_model.get((anchor, m), set())) for m in models]
        if any(not xs for xs in model_nodes):
            continue
        for combo in product(*model_nodes):
            pair_keys = [frozenset((a, b)) for a, b in combinations(combo, 2)]
            if not all(k in edges for k in pair_keys):
                continue
            vals = [edges[k] for k in pair_keys]
            rec = {
                "anchor_layer": anchor,
                "min_abs_corr": min(vals),
                "mean_abs_corr": sum(vals) / len(vals),
            }
            for m, n in zip(models, combo):
                rec[f"{m}_layer"] = n[1]
                rec[f"{m}_feature"] = n[2]
            triplets.append(rec)
    triplets.sort(key=lambda r: (r["min_abs_corr"], r["mean_abs_corr"]), reverse=True)
    return triplets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-tsv", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.90)
    args = ap.parse_args()

    data = json.loads(args.input.read_text())
    pair_rows = summarize_pairs(data, args.threshold)
    triplets = extract_triplets(data, args.threshold)

    summary = {
        "input": str(args.input),
        "threshold": args.threshold,
        "n_models": len(data.get("models", [])),
        "models": [m.get("model") for m in data.get("models", [])],
        "n_sequences": data.get("input", {}).get("n_sequences"),
        "pairwise_rows": pair_rows,
        "n_universal_triplets": len(triplets),
        "top_universal_triplets": triplets[:50],
        "interpretation": (
            "This summarizes exact three-model feature triplets from the top "
            "pairwise matches. It is a resource-ready pilot summary, not the "
            "final 10k-sequence universal atlas gate."
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2))

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w", newline="") as f:
        fieldnames = sorted(triplets[0]) if triplets else ["anchor_layer", "min_abs_corr", "mean_abs_corr"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in triplets:
            writer.writerow(row)

    lines = [
        "# R2 Universal Feature Atlas Pilot Summary",
        "",
        f"- Input: `{args.input}`",
        f"- Models: {', '.join(summary['models'])}",
        f"- Shared sequences: {summary['n_sequences']}",
        f"- Correlation threshold: abs(r) >= {args.threshold:.2f}",
        f"- Exact three-model universal triplets: {len(triplets)}",
        "",
        "## Pairwise Layer Summary",
        "",
        "| Models | Anchor layer | CKA | Mean match abs(r) | Matches | Matches above threshold | Max abs(r) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    threshold_key = f"n_abs_corr_ge_{args.threshold:.2f}"
    for row in pair_rows:
        lines.append(
            f"| {row['model_a']} vs {row['model_b']} | {row['anchor_layer']} | "
            f"{fmt(row['cka'])} | {fmt(row['mean_abs_match_corr'])} | {row['n_matches']} | "
            f"{row[threshold_key]} | {fmt(row['max_abs_corr'])} |"
        )
    lines += ["", "## Top Universal Triplets", ""]
    if triplets:
        header_keys = [k for k in triplets[0] if k not in {"min_abs_corr", "mean_abs_corr"}]
        lines.append("| Min abs(r) | Mean abs(r) | " + " | ".join(header_keys) + " |")
        lines.append("|---:|---:|" + "|".join(["---:"] * len(header_keys)) + "|")
        for row in triplets[:20]:
            vals = [str(row[k]) for k in header_keys]
            lines.append(f"| {fmt(row['min_abs_corr'])} | {fmt(row['mean_abs_corr'])} | " + " | ".join(vals) + " |")
    else:
        lines.append("No exact three-model triplets were recovered from the top pairwise matches at this threshold.")
    lines += [
        "",
        "Interpretation: " + summary["interpretation"],
        "",
    ]
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
