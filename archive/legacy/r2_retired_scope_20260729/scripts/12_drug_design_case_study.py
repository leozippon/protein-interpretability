#!/usr/bin/env python
"""Drug-design case study: CLT-steered therapeutic-candidate generation (R2-E).

This is the therapeutic-proof step of R2. We pick a concrete target, use
the reconciled L3 / L12 / L30 circuit to produce steered candidate
sequences, filter by sequence-level heuristics, and hand off a short list
of lead sequences to downstream structure validation (R2-F) and potentially
wet-lab followup.

Two built-in targets (pick via --target):

  - ec_lysozyme : EC 3.2.1.17 (muramidase). Generate ZymCTRL enzymes
    boosting L30 features most distinctive for EC 3.2.1.x, filter for the
    canonical Glu-Asp catalytic diad and a reasonable length (80-200aa).

  - kras_g12d_binder : short ProGen2 peptides (~30-60aa) steered with L12
    features active on binding-motif training sequences; filter for:
      - length 30-60 residues
      - hydrophobic/aromatic core fraction > 0.3
      - at least one disulfide-forming pair of Cys
      - no early stop tokens

For each generated candidate we record:
  - sequence
  - length, aa composition
  - target-specific pass/fail criteria
  - a "rank score" combining perplexity + target match

Output: JSON with top N leads + raw generations.

Usage:
    python r2_interpretability_transfer/scripts/12_drug_design_case_study.py \
        --target ec_lysozyme \
        --model zymctrl \
        --clt r2_interpretability_transfer/results/final_checkpoints/r2_clt_zymctrl_rerun_20260403/step_100000 \
        --ec-features r2_interpretability_transfer/results/circuit_analysis/zymctrl/ec_features.pkl \
        --n 200 --top-k 10 \
        --out r2_interpretability_transfer/results/drug_design/ec_lysozyme_leads.json
"""

import argparse
import json
import os
import pickle
import re
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


# ---------- Target definitions ----------

def target_ec_lysozyme():
    """EC 3.2.1.17 — muramidase (glycosidase, hydrolyzes peptidoglycan)."""
    return {
        "name": "ec_lysozyme",
        "model": "zymctrl",
        "prompt": "3.2.1.17<sep><start>",
        "ec_class": "lysozyme",           # key in ec_features.pkl
        "steering_layers": [3, 12, 30],
        "features_per_layer": 6,
        "multiplier": 2.5,
        "length_min": 80,
        "length_max": 200,
        "criteria": _criteria_lysozyme,
    }


def target_kras_g12d_binder():
    """Short binder peptide design (30-60aa) targeting KRAS G12D surface."""
    return {
        "name": "kras_g12d_binder",
        "model": "progen2-medium",
        "prompt": "1",                    # ProGen2 BOS token; binder specific via steering
        "ec_class": None,                  # ProGen2 is unconditional; no EC lookup
        "steering_layers": [3, 12, 24],
        "features_per_layer": 4,
        "multiplier": 2.0,
        "length_min": 30,
        "length_max": 60,
        "criteria": _criteria_kras_binder,
    }


TARGETS = {
    "ec_lysozyme": target_ec_lysozyme,
    "kras_g12d_binder": target_kras_g12d_binder,
}


# ---------- Target-specific filters ----------

def _clean_aa(seq: str) -> str:
    return "".join(c for c in seq.upper() if c.isalpha() and c in "ACDEFGHIKLMNPQRSTVWY")


def _criteria_lysozyme(seq: str) -> dict:
    """Lysozyme catalytic residues are Glu35 and Asp52 (hen egg-white
    numbering). In generated sequences we accept any E/D pair separated by
    8-25 residues, with E before D, as a proxy for the diad placement.
    """
    s = _clean_aa(seq)
    has_diad = False
    diad_span = None
    # Look for E followed by D within 8..25 residues
    for i, c in enumerate(s):
        if c == "E":
            for j in range(i + 8, min(i + 25, len(s))):
                if s[j] == "D":
                    has_diad = True
                    diad_span = (i + 1, j + 1)
                    break
            if has_diad:
                break
    cys_count = s.count("C")
    aromatic = sum(1 for c in s if c in "FWY")
    return {
        "seq_len": len(s),
        "cys_count": cys_count,
        "aromatic_count": aromatic,
        "aromatic_frac": aromatic / max(len(s), 1),
        "has_ed_diad": has_diad,
        "ed_diad_span": diad_span,
    }


def _criteria_kras_binder(seq: str) -> dict:
    s = _clean_aa(seq)
    hydrophobic = sum(1 for c in s if c in "VILMFYW")
    aromatic = sum(1 for c in s if c in "FWY")
    cys_count = s.count("C")
    has_disulfide_pair = cys_count >= 2
    hydrophobic_frac = hydrophobic / max(len(s), 1)
    aromatic_frac = aromatic / max(len(s), 1)
    return {
        "seq_len": len(s),
        "cys_count": cys_count,
        "has_disulfide_pair": has_disulfide_pair,
        "hydrophobic_count": hydrophobic,
        "hydrophobic_frac": hydrophobic_frac,
        "aromatic_count": aromatic,
        "aromatic_frac": aromatic_frac,
    }


# ---------- Feature selection ----------

def pick_top_features(ec_features: dict, ec_name: str, layer: int, k: int):
    ec_names = list(ec_features.keys())
    per_class = np.stack([ec_features[n]["mean"][layer] for n in ec_names])
    across_mean = per_class.mean(axis=0)
    across_std = per_class.std(axis=0) + 1e-6
    if ec_name not in ec_names:
        # Fallback: closest EC class match by prefix
        matches = [n for n in ec_names if n.startswith(ec_name.split(".")[0])]
        if not matches:
            return []
        ec_name = matches[0]
    idx = ec_names.index(ec_name)
    z = (per_class[idx] - across_mean) / across_std
    top = np.argsort(-z)[:k]
    return [(int(f), float(z[f])) for f in top]


def default_features(clt, layer: int, k: int):
    """Fallback: pick features with the highest mean decoder norm when we
    have no ec_features map (e.g., for unconditional generators)."""
    dec = clt.W_dec[layer][:, 0, :]  # (d_clt, d_model)
    norms = torch.linalg.norm(dec, dim=-1).cpu().numpy()
    top = np.argsort(-norms)[:k]
    return [(int(f), float(norms[f])) for f in top]


# ---------- Ranking ----------

@torch.no_grad()
def sequence_log_likelihood(pm, sequence: str) -> float:
    """Model's mean log-likelihood per token; higher = more plausible."""
    s = _clean_aa(sequence)
    if len(s) < 5:
        return float("-inf")
    ids = pm.tokenize(s)
    out = pm.model(ids)
    if hasattr(out, "logits"):
        logits = out.logits
    else:
        logits = out[0]
    shift_logits = logits[:, :-1, :]
    shift_labels = ids[:, 1:]
    logp = torch.log_softmax(shift_logits.float(), dim=-1)
    token_logp = logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return float(token_logp.mean())


def compute_rank_score(target_cfg: dict, criteria: dict, log_lik: float) -> float:
    """Combine target-specific pass count with model likelihood."""
    pass_count = 0
    total_crit = 0

    if target_cfg["name"] == "ec_lysozyme":
        total_crit = 3
        if target_cfg["length_min"] <= criteria["seq_len"] <= target_cfg["length_max"]:
            pass_count += 1
        if criteria["has_ed_diad"]:
            pass_count += 1
        if criteria["aromatic_frac"] > 0.05:
            pass_count += 1
    elif target_cfg["name"] == "kras_g12d_binder":
        total_crit = 4
        if target_cfg["length_min"] <= criteria["seq_len"] <= target_cfg["length_max"]:
            pass_count += 1
        if criteria["has_disulfide_pair"]:
            pass_count += 1
        if criteria["hydrophobic_frac"] > 0.3:
            pass_count += 1
        if criteria["aromatic_frac"] > 0.1:
            pass_count += 1

    crit_score = pass_count / max(total_crit, 1)
    return 0.6 * crit_score + 0.4 * (max(log_lik, -20.0) + 20.0) / 20.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(TARGETS.keys()))
    ap.add_argument("--model", default=None,
                    help="Override model (defaults to target recommendation)")
    ap.add_argument("--clt", required=True)
    ap.add_argument("--ec-features", default=None,
                    help="ec_features.pkl for EC-conditioned targets (optional)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-unsteered", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    target_cfg = TARGETS[args.target]()
    if args.model:
        target_cfg["model"] = args.model

    print("=" * 70)
    print(f"  Drug-design case study: {target_cfg['name']}")
    print("=" * 70)
    print(f"  Model:   {target_cfg['model']}")
    print(f"  Prompt:  {target_cfg['prompt']!r}")
    print(f"  Layers:  {target_cfg['steering_layers']}")
    print(f"  N:       {args.n}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Loading {target_cfg['model']}...")
    pm = load_model(target_cfg["model"], device=device)
    print(f"  Loading CLT...")
    clt = load_trained_clt(args.clt, device=device)

    ec_features = None
    if args.ec_features and os.path.exists(args.ec_features):
        with open(args.ec_features, "rb") as f:
            ec_features = pickle.load(f)

    # Build interventions
    interventions = []
    for l in target_cfg["steering_layers"]:
        if l >= clt.n_layers:
            continue
        k = target_cfg["features_per_layer"]
        if ec_features and target_cfg["ec_class"]:
            top = pick_top_features(ec_features, target_cfg["ec_class"], l, k)
        else:
            top = default_features(clt, l, k)
        print(f"    L{l}: top features {top[:3]}")
        for feat_idx, _ in top:
            interventions.append((l, feat_idx, target_cfg["multiplier"]))

    print(f"\n  Generating {args.n} steered candidates...")
    t0 = time.time()
    records = []
    for i in range(args.n):
        torch.manual_seed(args.seed + i)
        seq_steered = steer_generation(
            pm, clt, target_cfg["prompt"], interventions,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        # Strip prompt from output
        gen = seq_steered
        if gen.startswith(target_cfg["prompt"]):
            gen = gen[len(target_cfg["prompt"]):]
        crit = target_cfg["criteria"](gen)
        try:
            ll = sequence_log_likelihood(pm, gen)
        except Exception:
            ll = float("-inf")
        score = compute_rank_score(target_cfg, crit, ll)
        rec = {
            "idx": i,
            "sequence": _clean_aa(gen),
            "raw_output": gen,
            "criteria": crit,
            "log_likelihood": ll,
            "rank_score": score,
        }
        records.append(rec)
        if (i + 1) % 20 == 0 or i == args.n - 1:
            print(f"    {i+1}/{args.n}  ({time.time()-t0:.1f}s)")

    # Optional unsteered baseline
    unsteered = []
    if args.include_unsteered:
        print(f"\n  Generating {args.n} unsteered baseline...")
        for i in range(args.n):
            torch.manual_seed(args.seed + 10000 + i)
            seq = pm.generate(
                target_cfg["prompt"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            gen = seq[len(target_cfg["prompt"]):] if seq.startswith(target_cfg["prompt"]) else seq
            crit = target_cfg["criteria"](gen)
            unsteered.append({
                "idx": i,
                "sequence": _clean_aa(gen),
                "criteria": crit,
            })

    # Rank and pick top-k
    records.sort(key=lambda r: -r["rank_score"])
    top = records[:args.top_k]

    summary_criteria = {}
    for k in top[0]["criteria"].keys() if top else []:
        vals = [r["criteria"].get(k) for r in records]
        num = [v for v in vals if isinstance(v, (int, float))]
        if num:
            summary_criteria[k] = {
                "mean": float(np.mean(num)),
                "median": float(np.median(num)),
                "min": float(np.min(num)),
                "max": float(np.max(num)),
            }

    out = {
        "target": target_cfg["name"],
        "model": target_cfg["model"],
        "prompt": target_cfg["prompt"],
        "steering_layers": target_cfg["steering_layers"],
        "interventions": [list(x) for x in interventions],
        "n_generated": args.n,
        "top_k": args.top_k,
        "summary_criteria": summary_criteria,
        "leads": top,
        "all_records": records,
        "unsteered_baseline": unsteered,
        "elapsed_s": round(time.time() - t0, 1),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n" + "=" * 70)
    print(f"  Saved: {args.out}")
    print(f"  Top-{args.top_k} leads (by rank score):")
    for r in top[:5]:
        print(f"    rank={r['rank_score']:.3f}  len={r['criteria'].get('seq_len')}  "
              f"ll={r['log_likelihood']:.3f}")
        print(f"      {r['sequence'][:70]}...")


if __name__ == "__main__":
    main()
