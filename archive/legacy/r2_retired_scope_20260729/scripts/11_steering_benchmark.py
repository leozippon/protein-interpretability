#!/usr/bin/env python
"""EC-class steering benchmark with statistical evaluation (R2-D).

For each ZymCTRL EC class, we compare unsteered generation vs steered
generation (boosting discriminating features at the reconciled usable
layers L3 / L12 / L30). The outcome metric is EC-class purity — the
fraction of generated sequences that match the intended EC class according
to an external motif-based classifier.

Pipeline:
  1. For each EC class C ∈ {8 classes tested in EXP-R2-004}:
     a. Select top-discriminating features for C at L3, L12, L30 from
        `ec_features.pkl` (produced by 07_layer_ec_specificity.py)
     b. Generate N unsteered sequences with the EC-conditioned prompt
     c. Generate N steered sequences amplifying those features
     d. Score each sequence with a catalytic-motif HMM (external) OR
        our own feature-based heuristic
  2. Bootstrap CI for the steering effect size
  3. Permutation test for statistical significance

Output:
  `r2_interpretability_transfer/results/steering_benchmark/zymctrl_purity.json`

Usage:
    python r2_interpretability_transfer/scripts/11_steering_benchmark.py \
        --clt r2_interpretability_transfer/results/final_checkpoints/r2_clt_zymctrl_rerun_20260403/step_100000 \
        --ec-features r2_interpretability_transfer/results/circuit_analysis/zymctrl/ec_features.pkl \
        --n 100 --out r2_interpretability_transfer/results/steering_benchmark/zymctrl_purity.json
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.analysis.circuit_discovery import load_trained_clt, steer_generation
from src.models.model_loader import load_model


AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")

# Steering layers per phase (reconciled in EXP-R2-006)
STEERING_LAYERS = [3, 12, 30]
EC_PROMPTS = {
    "lysozyme": "3.2.1.17",
    "trypsin": "3.4.21.4",
    "ADH": "1.1.1.1",
    "catalase": "1.11.1.6",
    "DNA_polymerase": "2.7.7.7",
    "lipase": "3.1.1.3",
    "kinase": "2.7.11.1",
    "carbonic_anh": "4.2.1.1",
}


def pick_top_features(ec_features: dict, ec_name: str, layer: int, k: int = 5):
    """Pick top k most class-distinctive features for an EC class at a layer.

    Class-distinctiveness = z-score of feature's mean activation on this EC
    class relative to the across-class average.
    """
    ec_names = list(ec_features.keys())
    per_class = np.stack([ec_features[n]["mean"][layer] for n in ec_names])  # (n_ec, d_clt)
    across_mean = per_class.mean(axis=0)
    across_std = per_class.std(axis=0) + 1e-6
    idx = ec_names.index(ec_name)
    z = (per_class[idx] - across_mean) / across_std
    top = np.argsort(-z)[:k]
    return [(int(f), float(z[f])) for f in top]


def pick_direct_effect_features(direct_effect: dict, ec_name: str, layer: int, k: int = 5):
    """Pick exactly k positive direct-effect features or fail the cell."""
    if ec_name not in direct_effect.get("per_ec", {}):
        raise KeyError(f"{ec_name} missing from direct-effect feature file")
    rows = direct_effect["per_ec"][ec_name]["layers"].get(str(layer), [])
    if not rows:
        raise KeyError(f"layer {layer} missing for {ec_name} in direct-effect feature file")
    positive = [r for r in rows if float(r.get("direct_effect", 0.0)) > 0]
    if len(positive) < k:
        raise ValueError(
            f"{ec_name} layer {layer} has {len(positive)} positive candidates; "
            f"requires {k}. Refusing opposite-sign fallback."
        )
    picked = positive[:k]
    return [(int(r["feature"]), float(r.get("direct_effect", 0.0))) for r in picked]


def ec_purity_score(sequence: str, ec_name: str) -> float:
    """Heuristic EC purity score based on target-family motifs.

    This is a placeholder for an external HMM/PRAI-based classifier. It
    returns a score in [0, 1]. The benchmark passes class names such as
    ``lysozyme`` and ``ADH`` rather than raw EC numbers, so keep explicit
    class-name rules here instead of relying only on EC top-level prefixes.

    For a real benchmark, swap in a Pfam/HMMER scan or a trained classifier
    such as PRAI (https://github.com/bioinfomaticsCSU/PRAI) by implementing
    `run_external_classifier`.
    """
    import re
    seq = sequence.upper()
    seq = "".join(c for c in seq if c in AA_ALPHABET)
    if not seq:
        return 0.0
    ec_key = ec_name.lower()
    ec_top = ec_name.split(".")[0] if "." in ec_name else ""
    signals: list[float] = []

    if ec_key == "lysozyme":
        # Muramidases commonly rely on a Glu/Asp catalytic pair; allow
        # flexible spacing because generated sequences are unaligned.
        signals.append(1.0 if re.search(r"E.{8,35}D", seq) else 0.0)
        signals.append(1.0 if re.search(r"[ST]..[DE]", seq) else 0.0)
        signals.append(min(sum(seq.count(a) for a in "FWY") / max(len(seq), 1) * 12, 1.0))
    elif ec_key == "trypsin":
        # Serine protease proxy: catalytic H/D/S residues plus Gly-rich loop.
        signals.append(1.0 if ("H" in seq and "D" in seq and "S" in seq) else 0.0)
        signals.append(1.0 if re.search(r"G.[SG]G", seq) else 0.0)
        signals.append(1.0 if re.search(r"[ST]G", seq) else 0.0)
    elif ec_key == "adh":
        # Alcohol dehydrogenases: Rossmann-like nucleotide binding and
        # Zn-binding Cys/His enrichment in many families.
        signals.append(1.0 if re.search(r"G.G..G", seq[:180]) else 0.0)
        signals.append(min((seq.count("C") + seq.count("H")) / max(len(seq), 1) * 18, 1.0))
        signals.append(1.0 if re.search(r"[ST]..[KR]", seq[:120]) else 0.0)
    elif ec_key == "catalase":
        # Catalases contain conserved histidine/asparagine/arginine motifs
        # around the heme active site; use loose proxies for generated text.
        signals.append(1.0 if re.search(r"H.{4,35}[NR]", seq) else 0.0)
        signals.append(1.0 if re.search(r"R.{2,20}Y", seq) else 0.0)
        signals.append(min(seq.count("H") / max(len(seq), 1) * 25, 1.0))
    elif ec_key == "dna_polymerase":
        # Polymerases often expose acidic metal-binding motifs.
        signals.append(1.0 if re.search(r"D.{1,4}D", seq) else 0.0)
        signals.append(1.0 if re.search(r"[DE].{2,8}[DE].{2,8}[DE]", seq) else 0.0)
        signals.append(min((seq.count("D") + seq.count("E")) / max(len(seq), 1) * 8, 1.0))
    elif ec_key == "lipase":
        # Lipase/esterase GXSXG nucleophile elbow plus catalytic H/D/E.
        signals.append(1.0 if re.search(r"G.[ST].G", seq) else 0.0)
        signals.append(1.0 if ("H" in seq and ("D" in seq or "E" in seq)) else 0.0)
        signals.append(min(seq.count("G") / max(len(seq), 1) * 10, 1.0))
    elif ec_key == "kinase":
        # Protein kinase proxies: glycine-rich loop, VAIK-like lysine, HRD/DFG.
        signals.append(1.0 if re.search(r"G.G..G", seq[:120]) else 0.0)
        signals.append(1.0 if re.search(r"[VIL]A[IVL]K", seq) else 0.0)
        signals.append(1.0 if ("HRD" in seq or "DFG" in seq) else 0.0)
    elif ec_key == "carbonic_anh":
        # Carbonic anhydrases coordinate zinc with histidines.
        signals.append(1.0 if re.search(r"H.{2,25}H.{2,25}H", seq) else 0.0)
        signals.append(min(seq.count("H") / max(len(seq), 1) * 30, 1.0))
        signals.append(1.0 if re.search(r"[DE].{1,8}H", seq) else 0.0)
    elif ec_top == "1":
        # Rossmann-fold NAD binding
        signals.append(1.0 if re.search(r"G.G..G", seq[:150]) else 0.0)
        signals.append(min(seq.count("C") / max(len(seq), 1) * 30, 1.0))
    elif ec_top == "2":
        # Glycine-rich P-loop common in kinases / methyltransferases
        signals.append(1.0 if re.search(r"G.{2,4}G.G", seq[:100]) else 0.0)
        signals.append(min(seq.count("S") / max(len(seq), 1) * 20, 1.0))
    elif ec_top == "3":
        # Classical Ser-His-Asp catalytic triad for serine hydrolases
        has_ser = "S" in seq
        has_his = "H" in seq
        has_asp = "D" in seq
        signals.append(1.0 if (has_ser and has_his and has_asp) else 0.0)
        # Zinc-binding motif for metallohydrolases
        signals.append(1.0 if re.search(r"H..H", seq) else 0.0)
    elif ec_top == "4":
        # PLP-binding K for transaminases / aldolases
        k_frac = seq.count("K") / max(len(seq), 1)
        signals.append(min(k_frac * 25, 1.0))
    elif ec_top == "5":
        # Isomerase signature: rich in small residues, proline-pyramids
        signals.append(min(seq.count("P") / max(len(seq), 1) * 25, 1.0))
    elif ec_top == "6":
        # Ligase ATP-binding Walker-A motif
        signals.append(1.0 if re.search(r"G.{4}GK[ST]", seq) else 0.0)
    else:
        signals.append(0.0)

    return float(np.mean(signals)) if signals else 0.0


def run_external_classifier(sequence: str, ec_name: str) -> float | None:
    """Slot for swapping in a real EC classifier.

    Returns None if no external classifier is configured; callers then
    fall back to ec_purity_score.
    """
    return None


def generate_batch(pm, clt, prompt, interventions, n, max_new_tokens,
                   temperature, seed_base):
    records = []
    for i in range(n):
        seed = seed_base + i
        torch.manual_seed(seed)
        seq = steer_generation(
            pm, clt, prompt, interventions,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
        records.append({"seed": seed, "sequence": seq})
    return records


def paired_bootstrap_diff(scores_a: np.ndarray, scores_b: np.ndarray,
                          n_boot: int = 10000) -> dict:
    """Paired bootstrap interval and paired sign-randomization p-value."""
    if scores_a.shape != scores_b.shape:
        raise ValueError(f"paired score shapes differ: {scores_a.shape} vs {scores_b.shape}")
    rng = np.random.default_rng(0)
    differences = scores_a - scores_b
    obs = float(differences.mean())
    n = len(differences)
    boot = np.empty(n_boot, dtype=np.float32)
    for i in range(n_boot):
        indices = rng.integers(0, n, n)
        boot[i] = differences[indices].mean()
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    perm_diff = np.empty(n_boot, dtype=np.float32)
    for i in range(n_boot):
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)
        perm_diff[i] = (differences * signs).mean()
    p = float((1 + np.sum(np.abs(perm_diff) >= abs(obs))) / (n_boot + 1))
    return {"obs_diff": obs, "ci95": ci, "paired_sign_randomization_p": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="zymctrl")
    ap.add_argument("--clt", required=True)
    ap.add_argument("--ec-features", required=True,
                    help="Path to ec_features.pkl produced by 07_layer_ec_specificity.py")
    ap.add_argument("--direct-effect-features", default=None,
                    help="Optional T2-A direct_effect_features_v2.pkl; if set, use direct-effect features for steering.")
    ap.add_argument("--layers", type=int, nargs="+", default=STEERING_LAYERS)
    ap.add_argument("--features-per-layer", type=int, default=5)
    ap.add_argument("--multiplier", type=float, default=2.5)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed-base", type=int, default=1000,
                    help="First seed for paired unsteered/steered random streams")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ec-classes", nargs="+", default=None,
                    help="Optional subset of EC classes to evaluate")
    args = ap.parse_args()

    print("=" * 70)
    print(f"  Steering benchmark — {args.model}")
    print("=" * 70)

    with open(args.ec_features, "rb") as f:
        ec_features = pickle.load(f)
    direct_effect = None
    if args.direct_effect_features:
        with open(args.direct_effect_features, "rb") as f:
            direct_effect = pickle.load(f)
        print(f"  Loaded direct-effect features: {args.direct_effect_features}")
    ec_names = list(ec_features.keys())
    if args.ec_classes:
        ec_names = [e for e in ec_names if e in args.ec_classes]
    print(f"  EC classes: {ec_names}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Loading {args.model}...")
    pm = load_model(args.model, device=device)

    print(f"  Loading CLT...")
    clt = load_trained_clt(args.clt, device=device)

    results_per_class = {}
    overall_start = time.time()

    for ec_index, ec in enumerate(ec_names):
        print(f"\n  === EC {ec} ===")
        # Build interventions: top features at each steering layer
        interventions = []
        for l in args.layers:
            if l >= clt.n_layers:
                continue
            if direct_effect is not None:
                top = pick_direct_effect_features(
                    direct_effect, ec, l, k=args.features_per_layer
                )
            else:
                top = pick_top_features(ec_features, ec, l, k=args.features_per_layer)
            print(f"    L{l} top features: {top[:3]}")
            for feat_idx, z in top:
                interventions.append((l, feat_idx, args.multiplier))

        # Build ZymCTRL EC-conditioned prompt. ec_features.pkl uses human
        # class names as keys, while ZymCTRL generation expects EC numbers.
        prompt_ec = EC_PROMPTS.get(ec, ec)
        prompt = f"{prompt_ec}<sep><start>"

        print(f"    Generating {args.n} unsteered sequences...")
        t0 = time.time()
        class_seed_base = args.seed_base + ec_index * args.n
        unsteered = generate_batch(
            pm, clt, prompt, [], args.n, args.max_new_tokens,
            args.temperature, seed_base=class_seed_base,
        )
        print(f"      done ({time.time()-t0:.1f}s)")

        print(f"    Generating {args.n} steered sequences...")
        t0 = time.time()
        steered = generate_batch(
            pm, clt, prompt, interventions, args.n,
            args.max_new_tokens, args.temperature, seed_base=class_seed_base,
        )
        print(f"      done ({time.time()-t0:.1f}s)")

        # Score with external classifier if available, else heuristic
        def score(seq):
            v = run_external_classifier(seq, ec)
            return v if v is not None else ec_purity_score(seq, ec)

        scores_un = np.array([score(r["sequence"]) for r in unsteered], dtype=np.float32)
        scores_st = np.array([score(r["sequence"]) for r in steered], dtype=np.float32)

        stats = paired_bootstrap_diff(scores_st, scores_un)

        print(f"    purity unsteered = {scores_un.mean():.3f} ± {scores_un.std():.3f}")
        print(f"    purity steered   = {scores_st.mean():.3f} ± {scores_st.std():.3f}")
        print(f"    Δ = {stats['obs_diff']:+.3f} "
              f"(95% CI [{stats['ci95'][0]:+.3f}, {stats['ci95'][1]:+.3f}], "
              f"paired p={stats['paired_sign_randomization_p']:.4f})")

        results_per_class[ec] = {
            "prompt": prompt,
            "interventions": [list(x) for x in interventions],
            "seed_base": class_seed_base,
            "mean_unsteered": float(scores_un.mean()),
            "std_unsteered": float(scores_un.std()),
            "mean_steered": float(scores_st.mean()),
            "std_steered": float(scores_st.std()),
            "statistics": stats,
            "unsteered_records": unsteered,
            "steered_records": steered,
        }

    summary = {
        "model": args.model,
        "clt": args.clt,
        "n_per_condition": args.n,
        "layers": args.layers,
        "features_per_layer": args.features_per_layer,
        "multiplier": args.multiplier,
        "sampling": {
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "seed_base": args.seed_base,
            "paired_random_streams": True,
        },
        "feature_source": "direct_effect" if direct_effect is not None else "ec_zscore",
        "direct_effect_features": args.direct_effect_features,
        "direct_effect_selection": {
            "positive_only": direct_effect is not None,
            "opposite_sign_fallback": False,
            "source_samples_per_ec": direct_effect.get("samples_per_ec") if direct_effect else None,
            "source_cohort": direct_effect.get("cohort_source") if direct_effect else None,
        },
        "endpoint_status": "heuristic_only_not_validated_ec_classifier",
        "per_class": results_per_class,
        "elapsed_s": round(time.time() - overall_start, 1),
    }

    # Aggregate: how many classes show steering effect at p < 0.05?
    sig = sum(1 for r in results_per_class.values()
              if r["statistics"]["paired_sign_randomization_p"] < 0.05
              and r["statistics"]["obs_diff"] > 0)
    summary["n_classes_significant_positive"] = sig
    summary["n_classes"] = len(results_per_class)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print("\n" + "=" * 70)
    print(f"  Saved: {args.out}")
    print(f"  Significant positive steering: "
          f"{sig}/{len(results_per_class)} classes")


if __name__ == "__main__":
    main()
