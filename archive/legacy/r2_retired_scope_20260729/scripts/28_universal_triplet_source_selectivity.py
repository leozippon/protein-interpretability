#!/usr/bin/env python3
"""Annotate universal feature triplets by real-vs-random source selectivity."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
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


def roc_auc_binary(y_true: np.ndarray, score: np.ndarray) -> float | None:
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and score[order[j]] == score[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    rank_sum_pos = float(ranks[y_true == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load_records(path: str, field: str, max_sequences: int, max_length: int) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    cur = data
    for token in field.split("."):
        if token:
            cur = cur[token]
    out = []
    for row in cur[:max_sequences]:
        seq = cm.clean_sequence(row["sequence"])[:max_length]
        if seq:
            out.append({**row, "sequence": seq})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-spec", action="append", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--json-field", default="records")
    ap.add_argument("--triplets", required=True)
    ap.add_argument("--max-sequences", type=int, default=200)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-tsv", required=True)
    args = ap.parse_args()

    records = load_records(args.json, args.json_field, args.max_sequences, args.max_length)
    sequences = [r["sequence"] for r in records]
    y = np.asarray([1 if r.get("source") == "real_lysozyme" else 0 for r in records], dtype=np.int32)

    triplets = list(csv.DictReader(open(args.triplets), delimiter="\t"))
    specs = [cm.parse_model_spec(s) for s in args.model_spec]
    anchor_name, anchor_ckpt = specs[0]
    anchor_n_layers = cm.checkpoint_n_layers(anchor_ckpt)

    anchor_layers = sorted({int(row["anchor_layer"]) for row in triplets})
    model_info = {}
    encoded = {}
    needed_layers: dict[str, set[int]] = {}
    for row in triplets:
        for model in ["protgpt2", "zymctrl", "progen2-medium"]:
            needed_layers.setdefault(model, set()).add(int(row[f"{model}_layer"]))

    for name, ckpt in specs:
        true_name = name or cm.infer_model_name_from_config(ckpt)
        n_layers = cm.checkpoint_n_layers(ckpt)
        model_info[true_name] = {
            "checkpoint_dir": ckpt,
            "n_layers": n_layers,
            "needed_layers": sorted(needed_layers.get(true_name, [])),
            "anchor_layers": anchor_layers,
        }
        print(
            f"Encoding {true_name} anchor layers {anchor_layers} "
            f"-> target layers {model_info[true_name]['needed_layers']}",
            flush=True,
        )
        _mapped, matrices = cm.encode_sequences(
            true_name,
            ckpt,
            sequences,
            anchor_layers,
            anchor_n_layers,
            args.device,
        )
        encoded[true_name] = matrices

    annotated = []
    for i, row in enumerate(triplets, start=1):
        model_scores = {}
        aucs = {}
        effects = {}
        for model in ["protgpt2", "zymctrl", "progen2-medium"]:
            layer = int(row[f"{model}_layer"])
            feature = int(row[f"{model}_feature"])
            score = encoded[model][layer][:, feature].astype(np.float64)
            model_scores[model] = score
            auc = roc_auc_binary(y, score)
            aucs[model] = auc
            effects[model] = float(score[y == 1].mean() - score[y == 0].mean())
        consensus = np.mean(np.stack([model_scores[m] for m in ["protgpt2", "zymctrl", "progen2-medium"]]), axis=0)
        consensus_auc = roc_auc_binary(y, consensus)
        consensus_effect = float(consensus[y == 1].mean() - consensus[y == 0].mean())
        annotated.append({
            **row,
            "rank": i,
            "consensus_auc_real_vs_random": consensus_auc,
            "consensus_effect_real_minus_random": consensus_effect,
            "protgpt2_auc": aucs["protgpt2"],
            "zymctrl_auc": aucs["zymctrl"],
            "progen2_medium_auc": aucs["progen2-medium"],
            "protgpt2_effect": effects["protgpt2"],
            "zymctrl_effect": effects["zymctrl"],
            "progen2_medium_effect": effects["progen2-medium"],
            "interpretation": "lysozyme-associated" if consensus_auc and consensus_auc >= 0.80 else ("random-associated" if consensus_auc and consensus_auc <= 0.20 else "weak/source-mixed"),
        })

    annotated.sort(key=lambda r: abs(float(r["consensus_auc_real_vs_random"]) - 0.5), reverse=True)
    counts = {}
    for row in annotated:
        counts[row["interpretation"]] = counts.get(row["interpretation"], 0) + 1

    summary = {
        "task": "F-T2-2 universal triplet source-selectivity annotation",
        "n_sequences": len(records),
        "n_real": int(y.sum()),
        "n_random": int((1 - y).sum()),
        "n_triplets": len(annotated),
        "interpretation_counts": counts,
        "top_triplets": annotated[:20],
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    with open(args.out_tsv, "w", newline="") as f:
        fields = list(annotated[0].keys()) if annotated else ["rank"]
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(annotated)

    lines = [
        "# Universal Triplet Source Selectivity",
        "",
        f"- Sequences: {len(records)} (real={int(y.sum())}, random={int((1-y).sum())})",
        f"- Triplets annotated: {len(annotated)}",
        f"- Interpretation counts: {counts}",
        "",
        "| Rank | Anchor layer | Min abs(r) | Consensus AUC | Consensus effect | Interpretation | Feature triplet |",
        "|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in annotated[:20]:
        trip = (
            f"P:{row['protgpt2_layer']}/{row['protgpt2_feature']}; "
            f"Z:{row['zymctrl_layer']}/{row['zymctrl_feature']}; "
            f"G:{row['progen2-medium_layer']}/{row['progen2-medium_feature']}"
        )
        lines.append(
            f"| {row['rank']} | {row['anchor_layer']} | {float(row['min_abs_corr']):.4f} | "
            f"{float(row['consensus_auc_real_vs_random']):.4f} | "
            f"{float(row['consensus_effect_real_minus_random']):.4g} | "
            f"{row['interpretation']} | {trip} |"
        )
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_tsv}")


if __name__ == "__main__":
    main()
