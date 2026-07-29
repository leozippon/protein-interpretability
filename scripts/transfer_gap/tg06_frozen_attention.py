"""TG-06: what does freezing the attention pattern cost, in text versus protein?

Attribution graphs hold attention patterns fixed and explain only the pathway
through features. Anthropic states this openly as a limitation ("we do not
explain how attention patterns are computed"). Its *price* is a measurable
quantity: how much of the model's context-derived information depends on the
attention pattern being a function of this particular input.

Three conditions, identical tokens and weights throughout:

    clean       patterns computed from the true sequence
    self        patterns recaptured from the true sequence and re-injected
                (an exactness check on the injection path)
    transplant  patterns captured from a different sequence of the same length
    uniform     flat causal attention

Only the GPT-2-family arms are covered: injection needs a known attention
implementation, and GPT-2-large / ProtGPT2 / ZymCTRL share one, at matched
depth and width.

**Corrections against the 2026-07-24 run.** Rendering and cohort now come from
`src.transfer.arms` via `tg_common`, so the ProtGPT2 row -- clean CE 7.696 -- is
retracted and remeasured.

A second defect was silent rather than wrong. This script overrides
`transformers.models.gpt2.eager_attention_forward`, which reaches ProtGPT2 and
ZymCTRL because both are GPT-2, and does *not* reach ProGen2, whose modelling
code ships with the checkpoint. On an arm the patch does not reach, capture
returns no patterns, injection injects nothing, and `self` and `transplant` both
equal `clean` -- so the exactness check passes with an error of exactly zero and
the transplant cost reads 0.000 rather than raising. The capture is now asserted
to have produced one pattern per layer before anything is injected.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.gpt2 import modeling_gpt2

from tg_common import (
    DEFAULT_COHORT_SEED,
    REPO,
    cohort_for,
    cohort_provenance,
    load_arm,
    write_json,
)

LN2 = math.log(2.0)
_ORIGINAL_EAGER = modeling_gpt2.eager_attention_forward
_STATE: dict = {"mode": "clean", "captured": {}, "inject": {}}


def _patched_eager(module, query, key, value, attention_mask, head_mask=None, **kwargs):
    """Capture or override the post-softmax attention pattern of one layer."""
    output, weights = _ORIGINAL_EAGER(
        module, query, key, value, attention_mask, head_mask=head_mask, **kwargs
    )
    idx = module.layer_idx
    if _STATE["mode"] == "capture":
        _STATE["captured"][idx] = weights.detach().clone()
        return output, weights
    if _STATE["mode"] == "inject":
        pattern = _STATE["inject"][idx].to(weights.dtype)
        if pattern.shape != weights.shape:
            raise ValueError(f"layer {idx}: pattern {pattern.shape} vs {weights.shape}")
        return torch.matmul(pattern, value).transpose(1, 2), pattern
    return output, weights


def uniform_causal(shape, device, dtype):
    _, _, q_len, k_len = shape
    mask = torch.tril(torch.ones(q_len, k_len, device=device, dtype=torch.float32))
    mask = mask / mask.sum(-1, keepdim=True)
    return mask.expand(shape).to(dtype)


@torch.no_grad()
def cross_entropy(arm, ids) -> float:
    logits = arm.model(input_ids=ids).logits
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return float(nll.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-seq", type=int, default=200)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--res-min", type=int, default=400)
    ap.add_argument("--res-max", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=DEFAULT_COHORT_SEED)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arm = load_arm(args.arm, device=args.device, attn_implementation="eager")
    if not hasattr(arm.model, "transformer"):
        raise SystemExit(f"{arm.name}: unsupported attention implementation")
    modeling_gpt2.eager_attention_forward = _patched_eager

    cohort = cohort_for(arm, args.n_seq * 10, args.res_min, args.res_max, seed=args.seed)
    texts = cohort.input_strings(arm)
    blocks = []
    for text in texts:
        ids = arm.tokenizer(text, return_tensors=None)["input_ids"]
        if len(ids) >= args.window:
            blocks.append(ids[: args.window])
        if len(blocks) >= args.n_seq * 2:
            break
    if len(blocks) < 2 * args.n_seq:
        raise RuntimeError(f"{arm.name}: only {len(blocks)} usable windows")

    rng = np.random.default_rng(args.seed)
    donors = rng.permutation(args.n_seq) + args.n_seq  # disjoint donor pool

    results = {k: [] for k in ("clean", "self", "transplant", "uniform")}
    try:
        for i in range(args.n_seq):
            own = torch.tensor([blocks[i]], dtype=torch.long).to(arm.device)
            donor = torch.tensor([blocks[donors[i]]], dtype=torch.long).to(arm.device)

            _STATE["mode"] = "clean"
            results["clean"].append(cross_entropy(arm, own))

            _STATE["mode"], _STATE["captured"] = "capture", {}
            cross_entropy(arm, own)
            own_patterns = _STATE["captured"]
            # An empty or partial capture means the monkeypatch did not reach
            # this model's attention. Left unchecked it makes injection a no-op,
            # which reads downstream as a perfect injection and a zero transplant
            # cost rather than as an unsupported arm.
            if sorted(own_patterns) != list(range(arm.n_layer)):
                raise RuntimeError(
                    f"{arm.name}: captured attention patterns for "
                    f"{sorted(own_patterns)} of {arm.n_layer} layers; this arm's "
                    "attention does not route through "
                    "transformers.models.gpt2.eager_attention_forward, so patterns "
                    "cannot be injected and every condition would equal clean"
                )

            _STATE["mode"], _STATE["captured"] = "capture", {}
            cross_entropy(arm, donor)
            donor_patterns = _STATE["captured"]

            _STATE["mode"], _STATE["inject"] = "inject", own_patterns
            results["self"].append(cross_entropy(arm, own))

            _STATE["inject"] = donor_patterns
            results["transplant"].append(cross_entropy(arm, own))

            sample = next(iter(own_patterns.values()))
            flat = uniform_causal(sample.shape, sample.device, sample.dtype)
            _STATE["inject"] = {k: flat for k in own_patterns}
            results["uniform"].append(cross_entropy(arm, own))
            _STATE["mode"] = "clean"
    finally:
        modeling_gpt2.eager_attention_forward = _ORIGINAL_EAGER

    mean = {k: float(np.mean(v)) for k, v in results.items()}
    injection_error = abs(mean["self"] - mean["clean"])
    if injection_error > 5e-3:
        raise RuntimeError(f"pattern re-injection is not exact: {injection_error:.5f}")

    payload = dict(
        arm=arm.name,
        modality=arm.modality,
        seed=args.seed,
        n_sequences=args.n_seq,
        window=args.window,
        cohort=cohort_provenance(cohort, arm),
        ce_nats=mean,
        injection_exactness_nats=injection_error,
        transplant_cost_nats=mean["transplant"] - mean["clean"],
        transplant_cost_bits=(mean["transplant"] - mean["clean"]) / LN2,
        uniform_cost_nats=mean["uniform"] - mean["clean"],
        uniform_cost_bits=(mean["uniform"] - mean["clean"]) / LN2,
        sem_transplant_nats=float(
            np.std(np.asarray(results["transplant"]) - np.asarray(results["clean"]), ddof=1)
            / math.sqrt(args.n_seq)
        ),
    )
    out = Path(args.out) if args.out else (
        REPO / "results/transfer_gap_20260729_corrected/tg06"
    )
    write_json(out / f"{arm.name}.json", payload)
    print(f"  ce {mean}")
    print(f"  transplant cost {payload['transplant_cost_bits']:.4f} bits/token")
    print(f"  uniform    cost {payload['uniform_cost_bits']:.4f} bits/token")


if __name__ == "__main__":
    main()
