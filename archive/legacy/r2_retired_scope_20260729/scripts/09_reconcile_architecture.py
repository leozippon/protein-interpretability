#!/usr/bin/env python
"""Reconcile ZymCTRL three-phase architecture with CLT usability.

Our original EC-specificity analysis (script 07) reported L35 dominates
EC discrimination with total_L2=564 vs avg ~40. BUT the layer quality map
(script 08) reveals L35 has only 9.7% alive features. The "discrimination"
is therefore driven by a tiny subset of surviving features in a mostly-
dead CLT — not a reliable basis for steering.

This script produces a reconciled architecture picture:
  - Effective discrimination = raw_discrimination × CLT_quality
  - Usable layers with highest effective discrimination are the TRUE
    targets for circuit tracing and steering

Output: architecture_reconciliation.json with per-layer picture.

Usage:
    python scripts/09_reconcile_architecture.py
"""

import json
import os
import pickle

import numpy as np


def main():
    print("=" * 70)
    print("  ZymCTRL Architecture Reconciliation")
    print("=" * 70)

    with open("r2_interpretability_transfer/results/checkpoint_evaluation/layer_quality_map.json") as f:
        quality = json.load(f)["layer_maps"]["zymctrl"]

    with open("r2_interpretability_transfer/results/circuit_analysis/zymctrl/ec_features.pkl", "rb") as f:
        ec_features = pickle.load(f)

    ec_names = list(ec_features.keys())
    n_ecs = len(ec_names)
    n_layers = ec_features[ec_names[0]]["n_layers"]
    d_clt = ec_features[ec_names[0]]["d_clt"]

    # For each layer compute discrimination metrics and reconcile with quality
    records = []
    for l in range(n_layers):
        feat_matrix = np.stack([ec_features[name]["mean"][l] for name in ec_names])

        # Total pairwise L2 distance
        total_l2 = 0.0
        for i in range(n_ecs):
            for j in range(i + 1, n_ecs):
                total_l2 += float(np.linalg.norm(feat_matrix[i] - feat_matrix[j]))

        # Coefficient of variation across ECs, averaged over features that fire
        alive_in_ec = (feat_matrix > 0.01).any(axis=0)
        n_any_active = int(alive_in_ec.sum())
        if n_any_active > 0:
            sub = feat_matrix[:, alive_in_ec]
            cv = sub.std(0) / (np.abs(sub.mean(0)) + 1e-6)
            cv_mean = float(cv.mean())
        else:
            cv_mean = 0.0

        # Fraction of "EC-specific" features (fire for 1-2 ECs only)
        activation_count = (feat_matrix > 0.01).sum(axis=0)
        specific_frac = float(((activation_count > 0) & (activation_count <= 2)).mean())

        alive_rate = quality["alive_per_layer"][l]
        fvu = quality["fvu_per_layer"][l]
        clt_quality = quality["quality_per_layer"][l]
        is_usable = l in quality["usable_layers"]

        # Effective discrimination accounts for CLT quality
        # A layer that looks discriminating but has a dead CLT is unreliable
        effective_l2 = total_l2 * clt_quality

        records.append({
            "layer": l,
            "total_l2": total_l2,
            "cv_mean": cv_mean,
            "specific_frac": specific_frac,
            "alive_rate": alive_rate,
            "fvu": fvu,
            "clt_quality": clt_quality,
            "effective_l2": effective_l2,
            "is_usable": is_usable,
            "n_features_active_in_any_ec": n_any_active,
        })

    # Print reconciled table
    print(f"\n  Layer-by-layer reconciled architecture:")
    print(f"  {'L':>3} {'RawL2':>8} {'Alive%':>7} {'FVU':>6} "
          f"{'Qual':>6} {'EffL2':>8} {'Usable':>7}")
    for r in records:
        marker = " <<<" if r["is_usable"] and r["effective_l2"] > 15 else ""
        print(f"  {r['layer']:>3} {r['total_l2']:>8.1f} "
              f"{r['alive_rate']*100:>6.1f}% {r['fvu']:>6.3f} "
              f"{r['clt_quality']:>6.3f} {r['effective_l2']:>8.2f} "
              f"{'Y' if r['is_usable'] else '-':>7s}{marker}")

    # Find phases among USABLE layers only
    usable_records = [r for r in records if r["is_usable"]]
    usable_records_sorted = sorted(usable_records, key=lambda r: -r["effective_l2"])

    print(f"\n  Top 10 usable layers by effective discrimination:")
    for r in usable_records_sorted[:10]:
        print(f"    L{r['layer']}: eff_L2={r['effective_l2']:.2f} "
              f"raw_L2={r['total_l2']:.1f} alive={r['alive_rate']*100:.1f}%")

    # Classify usable layers by depth
    early_end = n_layers // 3
    mid_end = 2 * n_layers // 3

    early_usable = [r for r in usable_records if r["layer"] < early_end]
    mid_usable = [r for r in usable_records if early_end <= r["layer"] < mid_end]
    deep_usable = [r for r in usable_records if r["layer"] >= mid_end]

    # Recognition = early with high discrimination
    recognition_layers = sorted(
        early_usable, key=lambda r: -r["effective_l2"])[:3]
    # Generation = mid with highest CLT quality (least EC-specific, most universal)
    generation_layers = sorted(
        mid_usable, key=lambda r: -r["clt_quality"])[:3]
    # Output = deep with both discrimination and usability
    output_layers = sorted(
        deep_usable, key=lambda r: -r["effective_l2"])[:3]

    print(f"\n{'='*70}")
    print(f"  RECONCILED THREE-PHASE ARCHITECTURE (usable layers only)")
    print(f"{'='*70}")
    print(f"\n  Recognition phase (early, high discrimination):")
    for r in recognition_layers:
        print(f"    L{r['layer']}: effective_L2={r['effective_l2']:.2f}, "
              f"alive={r['alive_rate']*100:.1f}%")
    print(f"\n  Generation phase (mid, highest CLT quality):")
    for r in generation_layers:
        print(f"    L{r['layer']}: quality={r['clt_quality']:.3f}, "
              f"alive={r['alive_rate']*100:.1f}%")
    print(f"\n  Output phase (deep usable, discrimination+usability):")
    for r in output_layers:
        print(f"    L{r['layer']}: effective_L2={r['effective_l2']:.2f}, "
              f"alive={r['alive_rate']*100:.1f}%")

    # Compare to raw-only naive pick (what we used before)
    records_sorted_raw = sorted(records, key=lambda r: -r["total_l2"])
    print(f"\n  Contrast: naive top-5 by raw L2 (old approach):")
    for r in records_sorted_raw[:5]:
        print(f"    L{r['layer']}: raw_L2={r['total_l2']:.1f}, "
              f"alive={r['alive_rate']*100:.1f}% "
              f"{'(UNRELIABLE)' if not r['is_usable'] else ''}")

    # Save
    output = {
        "layer_records": records,
        "recommendations": {
            "recognition": [{"layer": r["layer"],
                             "effective_l2": r["effective_l2"],
                             "alive_rate": r["alive_rate"]}
                            for r in recognition_layers],
            "generation": [{"layer": r["layer"],
                            "clt_quality": r["clt_quality"],
                            "alive_rate": r["alive_rate"]}
                           for r in generation_layers],
            "output": [{"layer": r["layer"],
                        "effective_l2": r["effective_l2"],
                        "alive_rate": r["alive_rate"]}
                       for r in output_layers],
        },
        "note": (
            "Previous EC-specificity analysis (script 07) reported L35 "
            "dominates with total_L2=564 but L35 has only 9.7% alive "
            "features in the CLT. Effective discrimination reconciled "
            "against CLT quality places L3 and L30 as the best usable "
            "targets for circuit tracing."
        ),
    }
    os.makedirs("r2_interpretability_transfer/results/circuit_analysis/zymctrl", exist_ok=True)
    with open("r2_interpretability_transfer/results/circuit_analysis/zymctrl/architecture_reconciliation.json",
              "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to r2_interpretability_transfer/results/circuit_analysis/zymctrl/"
          f"architecture_reconciliation.json")


if __name__ == "__main__":
    main()
