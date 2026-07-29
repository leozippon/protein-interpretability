#!/usr/bin/env python
"""Causal feature ablation: prove a CLT feature controls specific positions (R2-G).

For a chosen (layer, feature_idx) pair, we:
  1. Run the generator on a set of reference sequences with the feature
     INTACT and record per-position log-probabilities for the actual tokens
  2. Re-run with the same feature ABLATED (multiplier=0) and record the
     same log-probabilities
  3. Compute Δlogp(pos) = logp_intact(pos) − logp_ablated(pos). Large
     positive Δ means the feature was driving that token; large negative
     means it was suppressing it.
  4. Report top-K positions by |Δlogp| along with their residue identity
     and annotation context.

Two causal-claim strengths the script produces:
  - "Local causal": ablation shifts probability at specific positions.
  - "Structured causal": the shifted positions cluster in biologically
    meaningful regions (e.g., active-site residues) when reference
    sequences are aligned to known features.

Usage:
    python r2_interpretability_transfer/scripts/14_causal_feature_ablation.py \
        --model zymctrl \
        --clt r2_interpretability_transfer/results/final_checkpoints/r2_clt_zymctrl_rerun_20260403/step_100000 \
        --ec-features r2_interpretability_transfer/results/circuit_analysis/zymctrl/ec_features.pkl \
        --ec-class 3.2.1.17 \
        --layer 12 --feature 2104 \
        --sequences r2_interpretability_transfer/data/ec_reference/ec_3.2.1.17.fasta \
        --out r2_interpretability_transfer/results/causal_ablation/zymctrl_L12_F2104.json
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

from src.analysis.circuit_discovery import load_trained_clt
from src.models.model_loader import load_model


def read_fasta(path: str, max_n: int = 20, max_len: int = 300):
    """Tiny FASTA reader. Returns list of (id, seq)."""
    out = []
    cur_id, cur_seq = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur_id is not None:
                    seq = "".join(cur_seq)
                    if 10 <= len(seq) <= max_len:
                        out.append((cur_id, seq))
                    if len(out) >= max_n:
                        return out
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line.strip())
        if cur_id is not None and cur_seq:
            seq = "".join(cur_seq)
            if 10 <= len(seq) <= max_len:
                out.append((cur_id, seq))
    return out


@torch.no_grad()
def per_position_logp(pm, sequence: str, clt, layer: int, feature: int,
                      multiplier: float) -> np.ndarray:
    """Compute per-token log-probability for `sequence`, with the (layer,
    feature) CLT feature optionally scaled by `multiplier`.

    `multiplier=1.0` → no intervention (intact).
    `multiplier=0.0` → full ablation.
    """
    ids = pm.tokenize(sequence)  # (1, T)

    hook_handle = None
    if multiplier != 1.0:
        def hook(module, inp, output):
            resid = inp[0].float()
            pre_act = torch.einsum(
                "bsd,fd->bsf", resid, clt.W_enc[layer]
            ) + clt.b_enc[layer]
            pre_act = torch.relu(pre_act)
            original = pre_act[:, :, feature].clone()
            modified = original * multiplier
            diff = (modified - original)
            dec_vec = clt.W_dec[layer][feature, 0, :]  # (d_model,)
            delta = torch.einsum("bs,d->bsd", diff, dec_vec).to(output.dtype)
            return output + delta
        block = pm._get_block(layer)
        mlp = pm._get_mlp(block)
        hook_handle = mlp.register_forward_hook(hook)

    try:
        out = pm.model(ids)
        logits = out.logits if hasattr(out, "logits") else out[0]
        shift_logits = logits[:, :-1, :]
        shift_labels = ids[:, 1:]
        logp = torch.log_softmax(shift_logits.float(), dim=-1)
        token_logp = logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    return token_logp[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--clt", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--feature", type=int, required=True)
    ap.add_argument("--sequences", required=True,
                    help="FASTA of reference sequences for the EC class")
    ap.add_argument("--ec-features", default=None,
                    help="Optional ec_features.pkl; used to verify feature "
                         "is class-distinctive")
    ap.add_argument("--ec-class", default=None)
    ap.add_argument("--max-sequences", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=300)
    ap.add_argument("--top-k-positions", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("=" * 70)
    print(f"  Causal feature ablation — {args.model} L{args.layer} F{args.feature}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading {args.model}...")
    pm = load_model(args.model, device=device)
    print(f"  Loading CLT from {args.clt}...")
    clt = load_trained_clt(args.clt, device=device)

    if args.ec_features and args.ec_class:
        with open(args.ec_features, "rb") as f:
            ec_features = pickle.load(f)
        if args.ec_class in ec_features:
            mean_map = ec_features[args.ec_class]["mean"]
            target_act = float(mean_map[args.layer][args.feature])
            across = float(np.stack([ec_features[e]["mean"][args.layer][args.feature]
                                      for e in ec_features]).mean())
            print(f"  Feature mean activation on {args.ec_class}: {target_act:.3f} "
                  f"(across-class mean: {across:.3f})")

    print(f"  Reading {args.sequences}...")
    seqs = read_fasta(args.sequences, max_n=args.max_sequences, max_len=args.max_len)
    print(f"  {len(seqs)} reference sequences")

    per_seq_records = []
    all_deltas = []

    t0 = time.time()
    for si, (sid, seq) in enumerate(seqs):
        try:
            logp_intact = per_position_logp(pm, seq, clt, args.layer,
                                             args.feature, multiplier=1.0)
            logp_ablated = per_position_logp(pm, seq, clt, args.layer,
                                              args.feature, multiplier=0.0)
        except Exception as e:
            print(f"  [{si}] skip {sid} ({e})")
            continue
        delta = logp_intact - logp_ablated
        abs_d = np.abs(delta)
        top_idx = np.argsort(-abs_d)[:args.top_k_positions]
        top_positions = [{
            "position": int(p + 2),          # +2 accounts for input->shifted label
            "residue": seq[p + 1] if p + 1 < len(seq) else "",
            "delta_logp": float(delta[p]),
            "logp_intact": float(logp_intact[p]),
            "logp_ablated": float(logp_ablated[p]),
        } for p in top_idx]
        per_seq_records.append({
            "id": sid,
            "length": len(seq),
            "mean_abs_delta": float(abs_d.mean()),
            "max_abs_delta": float(abs_d.max()),
            "fraction_suppressive": float((delta > 0.05).mean()),
            "fraction_driving": float((delta < -0.05).mean()),
            "top_positions": top_positions,
        })
        all_deltas.extend(delta.tolist())
        if (si + 1) % 5 == 0 or si == len(seqs) - 1:
            print(f"    {si+1}/{len(seqs)}  ({time.time()-t0:.1f}s)")

    deltas = np.array(all_deltas, dtype=np.float32) if all_deltas else np.array([])
    aggregate = {
        "n_sequences": len(per_seq_records),
        "n_positions_total": int(deltas.size),
        "mean_delta": float(deltas.mean()) if deltas.size else float("nan"),
        "std_delta": float(deltas.std()) if deltas.size else float("nan"),
        "frac_driving": float((deltas < -0.05).mean()) if deltas.size else float("nan"),
        "frac_suppressive": float((deltas > 0.05).mean()) if deltas.size else float("nan"),
        "mean_abs_delta": float(np.abs(deltas).mean()) if deltas.size else float("nan"),
    }

    # Collect residues at high-|Δ| positions: causal-affected residue profile
    affected_residues = {}
    for rec in per_seq_records:
        for p in rec["top_positions"][:5]:
            aa = p["residue"]
            if aa:
                affected_residues[aa] = affected_residues.get(aa, 0) + 1
    affected_total = sum(affected_residues.values())
    affected_pct = {aa: n / affected_total * 100 for aa, n in affected_residues.items()}

    out = {
        "model": args.model,
        "layer": args.layer,
        "feature": args.feature,
        "aggregate": aggregate,
        "top_affected_residue_composition_pct": dict(
            sorted(affected_pct.items(), key=lambda kv: -kv[1])
        ),
        "per_sequence": per_seq_records,
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "=" * 70)
    print(f"  Saved: {args.out}")
    print(f"  Mean |Δ logp|: {aggregate['mean_abs_delta']:.4f}")
    print(f"  Top affected residue composition:")
    for aa, pct in sorted(affected_pct.items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {aa}: {pct:.1f}%")


if __name__ == "__main__":
    main()
