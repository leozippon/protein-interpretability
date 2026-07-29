#!/usr/bin/env python
"""Direct-effect CLT feature selection for ZymCTRL EC conditioning.

This script ranks CLT features by an attribution-style direct-effect score:

    feature_activation * grad(EC-conditioned log-likelihood) * decoder_vector

For a source layer L and feature f, the score sums the feature's contribution
through every CLT decoder target in the trained window. This is a stricter
feature-selection criterion than mean-activation z-scores because a feature
must be active and point in a direction that changes the model likelihood.

The main output is a pickle containing `top_indices` with shape
`(n_ec_classes, n_layers, top_k)`, plus richer per-feature records.
"""

from __future__ import annotations

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

DEFAULT_REFERENCE_SEQUENCE = (
    "MTRLSAPARALVRQLIDREGYRQPAYVCPAGQLTLGYGHTRAAARTPAEGTLPRPLDTVAALDLLARDLDAALRCVADAVDPALPAGEFDALVSLAFNIGAGAFARSTLLRRLNAGDAAA"
)


def zymctrl_prompt(ec_name: str) -> str:
    ec = EC_PROMPTS.get(ec_name, ec_name)
    return f"{ec}<sep><start>"


def load_ec_class_order(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing ec_features pickle: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"bad ec_features pickle: expected non-empty dict at {path}")
    return list(data.keys())


def read_fasta(path: str, max_sequences: int, max_length: int) -> list[str]:
    seqs: list[str] = []
    cur: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seq = "".join(cur)[:max_length]
                    if seq:
                        seqs.append(seq)
                    cur = []
                    if len(seqs) >= max_sequences:
                        break
            else:
                cur.append(line)
        if cur and len(seqs) < max_sequences:
            seq = "".join(cur)[:max_length]
            if seq:
                seqs.append(seq)
    return seqs


def build_texts(ec_name: str, args) -> list[tuple[str, str]]:
    prompt = zymctrl_prompt(ec_name)
    seqs: list[str]
    if args.reference_fasta:
        seqs = read_fasta(args.reference_fasta, args.samples_per_ec, args.max_sequence_length)
        if not seqs:
            raise ValueError(f"no sequences loaded from {args.reference_fasta}")
    else:
        seqs = [args.reference_sequence[: args.max_sequence_length]]
    return [(prompt, prompt + seq) for seq in seqs[: args.samples_per_ec]]


def capture_forward(pm, input_ids: torch.Tensor):
    """Run a forward pass while turning each MLP output into a grad leaf."""
    n_layers = pm.n_layers
    resid_pre: list[torch.Tensor | None] = [None] * n_layers
    mlp_leafs: list[torch.Tensor | None] = [None] * n_layers
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, inp, output):
            resid_pre[layer_idx] = inp[0].detach().float()
            leaf = output.detach().requires_grad_(True)
            leaf.retain_grad()
            mlp_leafs[layer_idx] = leaf
            return leaf

        return hook

    for layer_idx in range(n_layers):
        block = pm._get_block(layer_idx)
        mlp = pm._get_mlp(block)
        handles.append(mlp.register_forward_hook(make_hook(layer_idx)))

    try:
        out = pm.model(input_ids)
        logits = out.logits if hasattr(out, "logits") else out[0]
    finally:
        for h in handles:
            h.remove()

    missing = [i for i, x in enumerate(resid_pre) if x is None]
    if missing:
        raise RuntimeError(f"MLP hooks did not fire for layers: {missing}")
    return logits, [x for x in resid_pre if x is not None], [x for x in mlp_leafs if x is not None]


def teacher_forced_logp(logits: torch.Tensor, labels: torch.Tensor,
                        target_start: int) -> torch.Tensor:
    """Mean log-probability from shifted position `target_start` onward."""
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    logp = torch.log_softmax(shift_logits, dim=-1)
    tok = logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    if target_start > 0:
        tok = tok[:, target_start:]
    if tok.numel() == 0:
        raise ValueError("empty target region for log-likelihood")
    return tok.mean()


def score_one_text(pm, clt, prompt: str, text: str, score_region: str,
                   device: str) -> tuple[list[torch.Tensor], float, int]:
    input_ids = pm.tokenize(text).to(device)
    prompt_len = pm.tokenize(prompt).shape[1]
    target_start = max(prompt_len - 1, 0) if score_region == "sequence" else 0

    pm.model.zero_grad(set_to_none=True)
    logits, resid_pre, mlp_leafs = capture_forward(pm, input_ids)
    objective = teacher_forced_logp(logits, input_ids, target_start)
    objective.backward()

    grad_mlp = []
    for layer_idx, leaf in enumerate(mlp_leafs):
        if leaf.grad is None:
            raise RuntimeError(f"missing MLP gradient for layer {layer_idx}")
        grad_mlp.append(leaf.grad.detach().float())

    with torch.no_grad():
        features = clt.encode(resid_pre)
        layer_scores = []
        for src_layer in range(clt.n_layers):
            score = torch.zeros(clt.d_clt, device=device, dtype=torch.float32)
            feat = features[src_layer].float()
            n_targets = min(clt.window, clt.n_layers - src_layer)
            for offset in range(n_targets):
                target_layer = src_layer + offset
                dec = clt.W_dec[src_layer][:, offset, :].float()
                grad = grad_mlp[target_layer]
                score = score + torch.einsum("bsf,bsd,fd->f", feat, grad, dec)
            layer_scores.append(score.detach().cpu())

    n_tokens = int(input_ids.shape[1])
    del logits, resid_pre, mlp_leafs, grad_mlp, features
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return layer_scores, float(objective.detach().cpu()), n_tokens


def top_records(scores: torch.Tensor, top_k: int) -> tuple[np.ndarray, list[dict]]:
    abs_scores = scores.abs()
    vals, idx = torch.topk(abs_scores, k=min(top_k, scores.numel()))
    idx_np = idx.cpu().numpy().astype(np.int64)
    rows = []
    for rank, (feature_idx, abs_value) in enumerate(zip(idx_np.tolist(), vals.cpu().tolist()), start=1):
        signed = float(scores[feature_idx].item())
        rows.append({
            "rank": rank,
            "feature": int(feature_idx),
            "direct_effect": signed,
            "abs_direct_effect": float(abs_value),
            "sign": "positive" if signed >= 0 else "negative",
        })
    return idx_np, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="zymctrl")
    ap.add_argument("--clt", required=True)
    ap.add_argument("--ec-features", required=True)
    ap.add_argument("--ec-classes", nargs="+", default=None)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--samples-per-ec", type=int, default=1)
    ap.add_argument("--reference-sequence", default=DEFAULT_REFERENCE_SEQUENCE)
    ap.add_argument("--reference-fasta", default=None)
    ap.add_argument("--max-sequence-length", type=int, default=256)
    ap.add_argument("--score-region", choices=["sequence", "full"], default="sequence")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    ec_classes = load_ec_class_order(args.ec_features)
    if args.ec_classes:
        missing = [x for x in args.ec_classes if x not in ec_classes]
        if missing:
            raise ValueError(f"requested EC classes missing from ec_features: {missing}")
        ec_classes = args.ec_classes

    device = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    print("=" * 70)
    print("  R2 T2-A Direct-effect feature selection")
    print("=" * 70)
    print(f"  model={args.model}  device={device}")
    print(f"  clt={args.clt}")
    print(f"  ec_classes={ec_classes}")
    print(f"  top_k={args.top_k} samples_per_ec={args.samples_per_ec}")

    pm = load_model(args.model, device=device)
    pm.model.config.use_cache = False
    for p in pm.model.parameters():
        p.requires_grad_(False)
    clt = load_trained_clt(args.clt, device=device)

    top_indices = np.full((len(ec_classes), clt.n_layers, args.top_k), -1, dtype=np.int64)
    top_scores = np.zeros((len(ec_classes), clt.n_layers, args.top_k), dtype=np.float32)
    per_ec: dict[str, dict] = {}

    for ec_i, ec_name in enumerate(ec_classes):
        texts = build_texts(ec_name, args)
        accum = [torch.zeros(clt.d_clt, dtype=torch.float32) for _ in range(clt.n_layers)]
        logps = []
        n_tokens = []
        print(f"\n  EC {ec_name}: {len(texts)} text(s)")
        for sample_i, (prompt, text) in enumerate(texts, start=1):
            scores, logp, ntok = score_one_text(pm, clt, prompt, text, args.score_region, device)
            logps.append(logp)
            n_tokens.append(ntok)
            for layer_idx, score in enumerate(scores):
                accum[layer_idx] += score / max(len(texts), 1)
            print(f"    sample {sample_i}/{len(texts)}: tokens={ntok} mean_logp={logp:.4f}")

        layer_records = {}
        for layer_idx, score in enumerate(accum):
            idx_np, rows = top_records(score, args.top_k)
            top_indices[ec_i, layer_idx, : len(idx_np)] = idx_np
            top_scores[ec_i, layer_idx, : len(idx_np)] = np.array(
                [r["direct_effect"] for r in rows], dtype=np.float32
            )
            layer_records[str(layer_idx)] = rows

        # Print compact layer check for the steering-relevant layers.
        for layer_idx in [3, 12, 30]:
            if layer_idx < clt.n_layers:
                head = layer_records[str(layer_idx)][:3]
                print(f"    L{layer_idx} top3: {[(r['feature'], round(r['direct_effect'], 4)) for r in head]}")

        per_ec[ec_name] = {
            "prompt": zymctrl_prompt(ec_name),
            "n_texts": len(texts),
            "mean_logp": float(np.mean(logps)),
            "tokens": n_tokens,
            "layers": layer_records,
        }

    output = {
        "task": "T2-A direct-effect feature selection",
        "model": args.model,
        "clt": args.clt,
        "ec_features": args.ec_features,
        "ec_classes": ec_classes,
        "top_k": args.top_k,
        "score_region": args.score_region,
        "samples_per_ec": args.samples_per_ec,
        "cohort_source": args.reference_fasta or "default_reference_sequence",
        "top_indices": top_indices,
        "top_scores": top_scores,
        "per_ec": per_ec,
        "shape": list(top_indices.shape),
        "elapsed_s": time.time() - t0,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(output, f)

    summary_out = args.summary_out or os.path.splitext(args.out)[0] + "_summary.json"
    summary = {
        "task": output["task"],
        "model": args.model,
        "clt": args.clt,
        "ec_classes": ec_classes,
        "shape": list(top_indices.shape),
        "top_k": args.top_k,
        "score_region": args.score_region,
        "samples_per_ec": args.samples_per_ec,
        "cohort_source": output["cohort_source"],
        "elapsed_s": output["elapsed_s"],
        "steering_layer_top_features": {
            ec: {
                str(layer): per_ec[ec]["layers"][str(layer)][:10]
                for layer in [3, 12, 30] if str(layer) in per_ec[ec]["layers"]
            }
            for ec in ec_classes
        },
    }
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  Saved pickle: {args.out}")
    print(f"  Saved summary: {summary_out}")
    print(f"  top_indices shape: {top_indices.shape}")
    print(f"  elapsed_s={output['elapsed_s']:.1f}")


if __name__ == "__main__":
    main()
