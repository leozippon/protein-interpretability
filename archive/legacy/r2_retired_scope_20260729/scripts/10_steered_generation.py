#!/usr/bin/env python
"""Steered generation CLI for CLT-based protein design (R2-C).

Wraps `src/analysis/circuit_discovery.steer_generation` in a script-level
interface so we can run large batches of steered + unsteered generations
for benchmarks (R2-D) and case studies (R2-E).

Intervention syntax:
    --interventions "L12:F2104:x2.0,L30:F45:x0.0"
  Each comma-separated token is "L<layer>:F<feat>:<op><value>", where
  <op> is `x` (multiply) or `=` (set absolute). Examples:
    L3:F128:x2.5   -> amplify L3 feature 128 by 2.5×
    L30:F2104:x0.0 -> ablate L30 feature 2104

Usage:
    python r2_interpretability_transfer/scripts/10_steered_generation.py \
        --model zymctrl \
        --clt r2_interpretability_transfer/results/final_checkpoints/r2_clt_zymctrl_rerun_20260403/step_100000 \
        --prompt "3.2.1.17<sep><start>" \
        --interventions "L30:F2104:x2.0" \
        --n 20 --max-new-tokens 200 \
        --out r2_interpretability_transfer/results/steered_generation/zymctrl_ec_3_2_1_17_boost.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.analysis.circuit_discovery import load_trained_clt, steer_generation
from src.models.model_loader import load_model


def parse_interventions(spec: str) -> list[tuple[int, int, float]]:
    """Parse 'L12:F2104:x2.0,L30:F45:x0.0' into [(layer, feat, multiplier)]."""
    if not spec:
        return []
    out = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 3:
            raise ValueError(f"bad intervention token: {token}")
        layer = int(parts[0].lstrip("L"))
        feat = int(parts[1].lstrip("F"))
        op = parts[2]
        if op.startswith("x"):
            mult = float(op[1:])
        elif op.startswith("="):
            raise NotImplementedError("absolute clamp (=) not yet supported; use x<value>")
        else:
            mult = float(op)
        out.append((layer, feat, mult))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Model name from MODEL_REGISTRY (e.g., zymctrl)")
    ap.add_argument("--clt", required=True,
                    help="Path to CLT checkpoint directory (with clt.pt + config.yaml)")
    ap.add_argument("--prompt", required=True,
                    help="Starting prompt (e.g., an EC number for ZymCTRL)")
    ap.add_argument("--interventions", default="",
                    help="Comma-separated list of L<layer>:F<feat>:x<mult>")
    ap.add_argument("--n", type=int, default=20, help="Number of generations")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-unsteered", action="store_true",
                    help="Also generate matched unsteered sequences for comparison")
    args = ap.parse_args()

    print("=" * 70)
    print(f"  Steered generation — {args.model}")
    print("=" * 70)

    interventions = parse_interventions(args.interventions)
    print(f"  Interventions: {interventions}")
    print(f"  Prompt: {args.prompt!r}")
    print(f"  N generations: {args.n}")

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Loading {args.model}...")
    pm = load_model(args.model, device=device)

    print(f"  Loading CLT from {args.clt}...")
    clt = load_trained_clt(args.clt, device=device)
    print(f"    n_layers={clt.n_layers}  d_model={clt.d_model}  "
          f"d_clt={clt.d_clt}  k={clt.k}")

    records = []
    t0 = time.time()
    for i in range(args.n):
        torch.manual_seed(args.seed + i)
        seq_steered = steer_generation(
            pm, clt,
            prompt=args.prompt,
            interventions=interventions,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        record = {
            "idx": i,
            "steered": seq_steered,
        }
        if args.include_unsteered:
            torch.manual_seed(args.seed + i)
            seq_unsteered = pm.generate(
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            record["unsteered"] = seq_unsteered
        records.append(record)

        if (i + 1) % 5 == 0 or i == args.n - 1:
            elapsed = time.time() - t0
            print(f"    {i+1}/{args.n} done ({elapsed:.1f}s)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "model": args.model,
            "clt": args.clt,
            "prompt": args.prompt,
            "interventions": [list(x) for x in interventions],
            "n": args.n,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "records": records,
        }, f, indent=2)
    print(f"\n  Saved: {args.out}")


if __name__ == "__main__":
    main()
