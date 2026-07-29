#!/usr/bin/env python3
"""Experiment 46 (PROTOCOL §6.2): oracle-direction controllability test.

Asks whether the generator is steerable by an *oracle* class direction taken
straight from the residual stream, where prior CLT-*feature* steering failed.

For each EC class we take the oracle direction = mean(R_raw[L] | class) -
mean(R_raw[L] | other) from the cached decoder representations (script 44), add
alpha * unit_direction to the MLP output at layer L during ZymCTRL generation,
and measure class purity with the same class-specific scorer used by
11_steering_benchmark.py. A permutation test on (steered - unsteered) purity
feeds the §6.2 controllability gate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args():
    pkg = Path(__file__).resolve().parent.parent
    base = pkg / "results/representation_audit_20260604"
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=base / "cache")
    ap.add_argument("--out-dir", type=Path, default=base / "steering")
    ap.add_argument("--model-base-dir", default=os.environ.get("R2_MODEL_BASE_DIR", "/Data/public/models_R2"))
    ap.add_argument("--model", default="zymctrl")
    ap.add_argument("--clt-ckpt", default=None, help="CLT checkpoint (for the feature-steering contrast arm)")
    ap.add_argument("--inject-layer", type=int, default=3, help="recognition layer L for the oracle direction")
    ap.add_argument("--alpha", type=float, default=8.0, help="injection strength (units of direction norm)")
    ap.add_argument("--ec-prompts", type=Path, default=None,
                    help="JSON {ec_topclass: prompt}; default = lysozyme worked example")
    ap.add_argument("--n-gen", type=int, default=40, help="sequences per arm per class")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


DEFAULT_EC_PROMPTS = {
    "lysozyme": "3.2.1.17<sep><start>",
    "trypsin": "3.4.21.4<sep><start>",
    "ADH": "1.1.1.1<sep><start>",
    "catalase": "1.11.1.6<sep><start>",
    "DNA_polymerase": "2.7.7.7<sep><start>",
    "lipase": "3.1.1.3<sep><start>",
    "kinase": "2.7.11.1<sep><start>",
    "carbonic_anh": "4.2.1.1<sep><start>",
}


def permutation_p(steered_hits, unsteered_hits, n_perm=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.array(steered_hits, float), np.array(unsteered_hits, float)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    na = len(a)
    null = []
    for _ in range(n_perm):
        rng.shuffle(pool)
        null.append(pool[:na].mean() - pool[na:].mean())
    return float(obs), float((np.sum(np.array(null) >= obs) + 1) / (n_perm + 1))


def main():
    args = parse_args()
    os.environ["R2_MODEL_BASE_DIR"] = args.model_base_dir
    import torch
    import recoverability_audit as ra
    from src.models.model_loader import load_model

    steer11 = ra.load_module(Path(__file__).resolve().parent / "11_steering_benchmark.py", "ra_steering11")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ec_prompts = json.loads(args.ec_prompts.read_text()) if args.ec_prompts else DEFAULT_EC_PROMPTS

    # ---- oracle directions from cached decoder representations ----
    dec = np.load(args.cache_dir / f"decoder_{args.model}.npz", allow_pickle=True)
    dec_labels = dec["labels"].astype(str)
    L = args.inject_layer
    raw = dec[f"raw_L{L}"]

    pm = load_model(args.model, device=args.device)
    results = {"model": args.model, "inject_layer": L, "alpha": args.alpha,
               "purity_judge": "11_steering_benchmark.ec_purity_score",
               "per_class": {}, "models": {}}

    for ec_class, prompt in ec_prompts.items():
        mask = dec_labels == str(ec_class)
        if mask.sum() < 5:
            results["per_class"][ec_class] = {"status": "skipped", "reason": "too few reference seqs for direction"}
            continue
        direction = raw[mask].mean(0) - raw[~mask].mean(0)
        direction = torch.tensor(direction / (np.linalg.norm(direction) + 1e-8), device=args.device)

        def oracle_hook(module, inp, out):
            return out + (args.alpha * direction).to(out.dtype)

        block = pm._get_block(L)
        mlp = pm._get_mlp(block)

        unsteered, steered = [], []
        for j in range(args.n_gen):
            torch.manual_seed(args.seed + j)
            unsteered.append(pm.generate(prompt, max_new_tokens=args.max_new_tokens,
                                         temperature=args.temperature, top_k=50))
        h = mlp.register_forward_hook(oracle_hook)
        try:
            for j in range(args.n_gen):
                torch.manual_seed(args.seed + 10000 + j)
                steered.append(pm.generate(prompt, max_new_tokens=args.max_new_tokens,
                                           temperature=args.temperature, top_k=50))
        finally:
            h.remove()

        hu = [float(steer11.ec_purity_score(s, ec_class)) for s in unsteered]
        hs = [float(steer11.ec_purity_score(s, ec_class)) for s in steered]
        obs, p = permutation_p(hs, hu, seed=args.seed)
        results["per_class"][ec_class] = {
            "prompt": prompt, "n_gen": args.n_gen,
            "unsteered_purity": float(np.mean(hu)) if hu else 0.0,
            "steered_purity": float(np.mean(hs)) if hs else 0.0,
            "delta": obs, "perm_p": p, "significant": bool(p < 0.05 and obs >= 0.05),
        }
        print(f"  EC class {ec_class}: steered {results['per_class'][ec_class]['steered_purity']:.2f} "
              f"vs unsteered {results['per_class'][ec_class]['unsteered_purity']:.2f} "
              f"(Δ={obs:.2f}, p={p:.3f})", flush=True)

    n_sig = sum(1 for v in results["per_class"].values() if v.get("significant"))
    n_classes = sum(1 for v in results["per_class"].values() if v.get("status") != "skipped")
    if n_classes < 8:
        verdict = "insufficient_classes"
    else:
        verdict = "controllable" if n_sig >= 3 else "distributed_or_robust"
    results["models"][args.model] = {"n_significant": n_sig, "n_classes": n_classes, "verdict": verdict}
    results["gate"] = "PROTOCOL §6.2: controllable iff Δ≥0.05 & p<0.05 on ≥3/8 EC classes."

    (args.out_dir / "oracle_steering.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    print(f"[46] verdict={verdict} ({n_sig}/{n_classes} classes) -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
