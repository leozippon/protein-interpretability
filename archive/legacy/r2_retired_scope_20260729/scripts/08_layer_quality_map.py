#!/usr/bin/env python
"""Layer-by-layer CLT quality map for all trained models.

Codex raised a critical point: "high dead-feature rates (especially in
ProtGPT2 and ZymCTRL)" and "moderate reconstruction quality" make circuit
findings preliminary. This script quantifies exactly which layers are
usable for circuit analysis and steering, and which are not.

Output is a layer quality map for each model showing:
  - Per-layer alive feature count
  - Per-layer FVU (reconstruction fidelity)
  - "Usable" flag (alive >= 20% AND FVU < 0.5)
  - Recommended layers for circuit tracing
  - Recommended layers for steering interventions

This feeds directly into the R2 paper's layer selection strategy and
reality-checks our "three-phase architecture" claim about ZymCTRL.
A layer with 90% dead features CANNOT support reliable steering
regardless of its role in raw model activations.

Usage:
    python scripts/08_layer_quality_map.py
"""

import json
import os

import numpy as np


EVAL_PATH = "r2_interpretability_transfer/results/checkpoint_evaluation/rerun_evaluation.json"
OUT_PATH = "r2_interpretability_transfer/results/checkpoint_evaluation/layer_quality_map.json"

ALIVE_THRESHOLD = 0.20    # >= 20% alive
FVU_THRESHOLD = 0.50       # reconstruction at least marginal


def analyze_model(name: str, d: dict) -> dict:
    """Analyze layer quality for one model."""
    alive = np.array(d["alive_per_layer"])
    fvu = np.array(d["fvu_per_layer"])
    n_layers = d["n_layers"]
    d_clt = d["d_clt"]

    # Usability mask
    usable = (alive >= ALIVE_THRESHOLD) & (fvu < FVU_THRESHOLD)
    usable_layers = np.where(usable)[0].tolist()

    # Quality score = alive rate * (1 - min(fvu, 1.0))
    quality = alive * (1 - np.clip(fvu, 0, 1))

    # Best layers for circuit tracing (top-5 by quality)
    top_quality = np.argsort(quality)[::-1][:5].tolist()

    # Layer regions (early / mid / deep)
    early_end = n_layers // 3
    mid_end = 2 * n_layers // 3

    early_usable = [l for l in usable_layers if l < early_end]
    mid_usable = [l for l in usable_layers if early_end <= l < mid_end]
    deep_usable = [l for l in usable_layers if l >= mid_end]

    # Contiguous usable runs
    runs = []
    if usable_layers:
        cur_start = usable_layers[0]
        prev = usable_layers[0]
        for l in usable_layers[1:]:
            if l == prev + 1:
                prev = l
            else:
                runs.append((cur_start, prev))
                cur_start = l
                prev = l
        runs.append((cur_start, prev))

    # Assessment
    if len(usable_layers) < n_layers // 4:
        assessment = "POOR"
    elif len(usable_layers) < n_layers // 2:
        assessment = "MODERATE"
    else:
        assessment = "GOOD"

    return {
        "n_layers": n_layers,
        "d_clt": d_clt,
        "k": d["k"],
        "overall_fvu": float(d["fvu_mean"]),
        "overall_dead_pct": float(d["dead_mean"] * 100),
        "n_usable": len(usable_layers),
        "usable_layers": usable_layers,
        "usable_early": early_usable,
        "usable_mid": mid_usable,
        "usable_deep": deep_usable,
        "usable_runs": [{"start": s, "end": e, "length": e - s + 1} for s, e in runs],
        "top_quality_layers": top_quality,
        "alive_per_layer": alive.tolist(),
        "fvu_per_layer": fvu.tolist(),
        "quality_per_layer": quality.tolist(),
        "assessment": assessment,
    }


def print_layer_map(name: str, analysis: dict):
    """Pretty-print layer quality map."""
    print(f"\n{'='*78}")
    print(f"  {name.upper()}  ({analysis['assessment']})")
    print(f"{'='*78}")
    print(f"  n_layers={analysis['n_layers']}  d_clt={analysis['d_clt']}  k={analysis['k']}")
    print(f"  Overall: FVU={analysis['overall_fvu']:.3f}  "
          f"Dead={analysis['overall_dead_pct']:.1f}%")
    print(f"  Usable layers (alive>=20%, FVU<0.5): "
          f"{analysis['n_usable']}/{analysis['n_layers']}")

    if analysis["usable_runs"]:
        print(f"  Contiguous usable regions:")
        for r in analysis["usable_runs"]:
            print(f"    L{r['start']}-L{r['end']}  ({r['length']} layers)")

    print(f"  Top-5 quality layers:", analysis["top_quality_layers"])
    print(f"  Early-usable: {analysis['usable_early']}")
    print(f"  Mid-usable:   {analysis['usable_mid']}")
    print(f"  Deep-usable:  {analysis['usable_deep']}")

    # Per-layer table
    print(f"\n  Layer-by-layer:")
    print(f"  {'L':>3} {'alive%':>8} {'FVU':>7} {'quality':>8} {'usable':>7}")
    for l in range(analysis["n_layers"]):
        a = analysis["alive_per_layer"][l]
        f = analysis["fvu_per_layer"][l]
        q = analysis["quality_per_layer"][l]
        u = "Y" if l in analysis["usable_layers"] else "-"
        marker = " <-- TOP" if l in analysis["top_quality_layers"][:3] else ""
        print(f"  {l:>3} {a*100:>7.1f}% {f:>7.3f} {q:>8.3f} {u:>7s}{marker}")


def recommend_circuit_layers(analysis: dict) -> dict:
    """Recommend which layers to use for different R2 circuit tasks."""
    usable = analysis["usable_layers"]
    n_layers = analysis["n_layers"]

    if not usable:
        return {
            "can_trace_circuits": False,
            "reason": "No usable layers — CLT training insufficient",
        }

    # For EC/condition recognition: prefer earliest usable
    recognition = [l for l in usable if l < n_layers // 3]
    if not recognition:
        recognition = [usable[0]]

    # For sequence generation rules: prefer mid
    generation = [l for l in usable if n_layers // 3 <= l < 2 * n_layers // 3]
    if not generation:
        generation = [usable[len(usable) // 2]]

    # For output steering: prefer deep
    steering = [l for l in usable if l >= 2 * n_layers // 3]
    if not steering:
        # If no deep layers usable, we CANNOT steer reliably
        return {
            "can_trace_circuits": True,
            "can_steer_output": False,
            "reason": "No deep layers usable — steering unreliable",
            "recognition_layers": recognition,
            "generation_layers": generation,
        }

    return {
        "can_trace_circuits": True,
        "can_steer_output": True,
        "recognition_layers": recognition,
        "generation_layers": generation,
        "steering_layers": steering,
    }


def main():
    print("=" * 78)
    print("  R2 CLT Layer Quality Map")
    print("=" * 78)

    if not os.path.exists(EVAL_PATH):
        print(f"Missing: {EVAL_PATH}")
        return

    with open(EVAL_PATH) as f:
        data = json.load(f)

    layer_maps = {}
    recommendations = {}
    for name, d in data.items():
        a = analyze_model(name, d)
        layer_maps[name] = a
        print_layer_map(name, a)
        recommendations[name] = recommend_circuit_layers(a)

    # Print recommendations summary
    print(f"\n{'='*78}")
    print("  R2 Paper Recommendations (per model)")
    print(f"{'='*78}")
    for name, rec in recommendations.items():
        print(f"\n  [{name}]")
        for k, v in rec.items():
            print(f"    {k}: {v}")

    # Save
    output = {
        "layer_maps": layer_maps,
        "recommendations": recommendations,
        "thresholds": {
            "alive_threshold": ALIVE_THRESHOLD,
            "fvu_threshold": FVU_THRESHOLD,
        },
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
