#!/usr/bin/env python
"""Sanity-check whether CLT steering hooks actually move model logits.

The previous R2-G causal-ablation output was exactly zero, which is a plumbing
failure until proven otherwise. This diagnostic attaches the same style of MLP
forward hook used by steering/ablation, chooses an active CLT feature on a
reference sequence, and compares teacher-forced next-token logits under:

  - no hook
  - hook with multiplier=1.0
  - hook with multiplier=10.0
  - hook with multiplier=0.0

Acceptance for a healthy hook:
  - hook hit count > 0
  - multiplier=1.0 is identical to no-hook up to FP tolerance
  - multiplier=10.0 causes a measurable logit shift
  - multiplier=0.0 causes a measurable logit shift
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.analysis.circuit_discovery import load_trained_clt
from src.models.model_loader import load_model


DEFAULT_ZYMCTRL_CLT = "/oss-pvc/zhk_zip/outputs/research2/clt_weights/zymctrl_v2/step_200000"
DEFAULT_PROGEN2_CLT = (
    "/gpfs/jiaotongdamoxing/zhk_zip/biocc/paper_r2_nature_mi/results/final_checkpoints/"
    "r2_clt_progen2_medium_rerun_20260403/clt_weights/progen2-medium/step_100000"
)
DEFAULT_SEQUENCE = (
    "MTRLSAPARALVRQLIDREGYRQPAYVCPAGQLTLGYGHTRAAARTPAEGTLPRPLDTVAALDLLARDLDAALRCVADAVDPALPAGEFDALVSLAFNIGAGAFARSTLLRRLNAGDAAA"
)


def logits_for_ids(pm, ids: torch.Tensor) -> torch.Tensor:
    out = pm.model(ids)
    return (out.logits if hasattr(out, "logits") else out[0]).detach().float()


def pick_active_feature(pm, clt, ids: torch.Tensor, preferred_layers: list[int]) -> tuple[int, int, float]:
    cache = pm.get_activations(ids)
    feats = clt.encode([x.float() for x in cache.resid_pre])
    candidates = []
    for layer in preferred_layers:
        if 0 <= layer < clt.n_layers:
            layer_feats = feats[layer]
            val, idx = layer_feats.reshape(-1, clt.d_clt).max(dim=0)
            best_val, feat_idx = val.max(dim=0)
            candidates.append((float(best_val.item()), int(layer), int(feat_idx.item())))
    if not candidates:
        raise ValueError(f"no valid preferred layers in {preferred_layers}")
    value, layer, feature = max(candidates, key=lambda x: x[0])
    if value <= 0:
        raise RuntimeError("no active CLT feature found on the reference sequence")
    return layer, feature, value


def run_with_hook(pm, clt, ids: torch.Tensor, layer: int, feature: int,
                  multiplier: float) -> tuple[torch.Tensor, dict]:
    stats = {
        "hits": 0,
        "input_norms": [],
        "output_norms": [],
        "feature_pre_act_max": [],
    }

    def hook(module, inp, output):
        resid = inp[0].float()
        pre_act = torch.einsum("bsd,fd->bsf", resid, clt.W_enc[layer]) + clt.b_enc[layer]
        pre_act = torch.relu(pre_act)
        original = pre_act[:, :, feature]
        diff = original * multiplier - original
        dec_vec = clt.W_dec[layer][feature, 0, :]
        delta = torch.einsum("bs,d->bsd", diff, dec_vec).to(output.dtype)
        stats["hits"] += 1
        stats["input_norms"].append(float(resid.norm().detach().cpu()))
        stats["output_norms"].append(float(output.float().norm().detach().cpu()))
        stats["feature_pre_act_max"].append(float(original.max().detach().cpu()))
        return output + delta

    block = pm._get_block(layer)
    mlp = pm._get_mlp(block)
    handle = mlp.register_forward_hook(hook)
    try:
        logits = logits_for_ids(pm, ids)
    finally:
        handle.remove()
    return logits, stats


def summarize_model(model_name: str, clt_path: str, prompt: str, sequence: str,
                    layers: list[int], device: str) -> dict:
    pm = load_model(model_name, device=device)
    clt = load_trained_clt(clt_path, device=device)

    text = prompt + sequence
    ids = pm.tokenize(text)
    if ids.shape[1] < 4:
        raise RuntimeError(f"tokenized input too short for {model_name}: {ids.shape}")

    layer, feature, active_value = pick_active_feature(pm, clt, ids, layers)
    print(f"  {model_name}: using L{layer} F{feature} active_value={active_value:.4f}")

    base = logits_for_ids(pm, ids)
    same, same_stats = run_with_hook(pm, clt, ids, layer, feature, 1.0)
    amp, amp_stats = run_with_hook(pm, clt, ids, layer, feature, 10.0)
    abl, abl_stats = run_with_hook(pm, clt, ids, layer, feature, 0.0)

    same_delta = float((same - base).abs().max().cpu())
    amp_delta = float((amp - base).abs().max().cpu())
    abl_delta = float((abl - base).abs().max().cpu())
    result = {
        "model": model_name,
        "clt": clt_path,
        "input_tokens": int(ids.shape[1]),
        "layer": layer,
        "feature": feature,
        "active_value": active_value,
        "multiplier_1_max_abs_delta": same_delta,
        "multiplier_10_max_abs_delta": amp_delta,
        "multiplier_0_max_abs_delta": abl_delta,
        "stats_multiplier_1": same_stats,
        "stats_multiplier_10": amp_stats,
        "stats_multiplier_0": abl_stats,
        "assertions": {
            "hook_fires": amp_stats["hits"] > 0 and abl_stats["hits"] > 0,
            "multiplier_1_identical": same_delta < 1e-5,
            "multiplier_10_moves_logits": amp_delta > 1.0,
            "multiplier_0_moves_logits": abl_delta > 0.05,
        },
    }
    result["passed"] = all(result["assertions"].values())

    del pm
    del clt
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zymctrl-clt", default=DEFAULT_ZYMCTRL_CLT)
    ap.add_argument("--progen2-clt", default=DEFAULT_PROGEN2_CLT)
    ap.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    ap.add_argument("--zymctrl-prompt", default="3.2.1.17<sep><start>")
    ap.add_argument("--progen2-prompt", default="")
    ap.add_argument("--zymctrl-layers", type=int, nargs="+", default=[3, 12, 30])
    ap.add_argument("--progen2-layers", type=int, nargs="+", default=[2, 9, 22])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", nargs="+", default=["zymctrl", "progen2-medium"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = []
    print("=" * 70)
    print("  R2 hook sanity diagnostic")
    print("=" * 70)
    if "zymctrl" in args.models:
        results.append(summarize_model(
            "zymctrl", args.zymctrl_clt, args.zymctrl_prompt, args.sequence,
            args.zymctrl_layers, args.device,
        ))
    if "progen2-medium" in args.models:
        results.append(summarize_model(
            "progen2-medium", args.progen2_clt, args.progen2_prompt, args.sequence,
            args.progen2_layers, args.device,
        ))

    out = {
        "diagnostic": "00_hook_sanity",
        "results": results,
        "passed": all(r["passed"] for r in results),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  Saved: {args.out}")
    for r in results:
        print(
            f"  {r['model']}: passed={r['passed']} "
            f"delta@1={r['multiplier_1_max_abs_delta']:.3e} "
            f"delta@10={r['multiplier_10_max_abs_delta']:.3f} "
            f"delta@0={r['multiplier_0_max_abs_delta']:.3f}"
        )
        print(f"    assertions={r['assertions']}")
    if not out["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
