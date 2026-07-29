"""TG-10: how much causal headroom does the P0-2b estimand actually have?

P0-2b replaced a single layer's MLP output at relative depth 0.5 and required
loss recovered >= 0.80 against a training-target-mean ablation baseline. Two of
three models produced denominators too small to score at all (ProtGPT2 0.0338,
ZymCTRL 0.0158 nats/token) and the third cleared the guard with only 0.0614.

Before concluding that the dictionaries are at fault, the estimand itself has to
be checked: in a 36-layer residual network, one MLP's contribution to the final
logits is intrinsically small, so a >= 0.80 recovery gate on that quantity may
be unattainable for *any* model. That is testable, and the test P0-2b lacked is
a matched text positive control.

**Superseded** by `scripts/transfer/03_estimand_power.py`, which measures the
same estimand over the current panel with bootstrap guard-pass fractions and is
the source of the L1 anchor (`mlp_single@d0.50@cohort_mean` costing GPT-2-large
0.0225 nats/token, guard pass fraction 0.0). This script is retained because it
is the cheapest way to re-derive the scope ladder on a new arm, and is corrected
rather than left wrong.

Corrections against the 2026-07-24 run: rendering and cohort now come from
`src.transfer.arms` via `tg_common`, so the ProtGPT2 row -- clean CE 10.117 and a
*negative* single-MLP delta of -0.0692 nats/token, which is the signature of a
model being scored off its own distribution -- is retracted and remeasured. The
ablation means were also estimated from the first 8 batches while the ablation
itself ran over the whole cohort; they are now estimated over the same cohort
they are applied to.

This script measures the causal footprint of mean-ablating, with no dictionary
involved:

    mlp_single    one MLP output, swept over depth
    mlp_window8   the eight-layer window a windowed CLT decodes into
    mlp_all       every MLP output in the model
    resid_block   the whole block output at one depth (the TG-03 estimand)

Reported as CE delta and KL against clean, in nats/token, with the P0-2b
denominator guards applied. If GPT-2-large also fails the guard at mlp_single,
the gate is mis-specified rather than the protein dictionaries being uniquely
bad.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from tg_common import (
    DEFAULT_COHORT_SEED,
    REPO,
    analysis_layer,
    cohort_for,
    cohort_provenance,
    load_arm,
    tokenize_batch,
    write_json,
)

CE_GUARD = 0.05   # P0-2b minimum mean-ablation CE delta, nats/token
KL_GUARD = 0.01   # P0-2b minimum mean-ablation KL, nats/token


@torch.no_grad()
def mlp_output_means(arm, texts, max_len, batch, cap_batches=None):
    """Per-layer mean of the MLP output, the P0-2b ablation target.

    ``cap_batches`` defaults to the whole cohort. It used to default to 8, so the
    replacement constant was a mean over the first 32 sequences while the
    ablation it defined ran over 120 -- a cohort-mean baseline estimated on a
    quarter of the cohort it is the mean of.
    """
    sums = [torch.zeros(arm.d_model, dtype=torch.float64) for _ in range(arm.n_layer)]
    count = 0
    store: dict[int, torch.Tensor] = {}
    handles = [
        arm.blocks()[i].mlp.register_forward_hook(
            lambda _m, _i, out, i=i: store.__setitem__(
                i, out[0] if isinstance(out, tuple) else out
            )
        )
        for i in range(arm.n_layer)
    ]
    try:
        for b, start in enumerate(range(0, len(texts), batch)):
            if cap_batches is not None and b >= cap_batches:
                break
            ids, mask = tokenize_batch(arm, texts[start : start + batch], max_len)
            ids, mask = ids.to(arm.device), mask.to(arm.device)
            arm.model(input_ids=ids, attention_mask=mask)
            keep = mask.bool()
            for i in range(arm.n_layer):
                sums[i] += store[i][keep].double().sum(0).cpu()
            count += int(keep.sum())
            store.clear()
    finally:
        for h in handles:
            h.remove()
    if count == 0:
        raise RuntimeError(f"{arm.name}: no tokens for the ablation mean")
    return [(s / count).float() for s in sums]


@torch.no_grad()
def scored_pass(arm, texts, max_len, batch, ablate_layers, means, resid_layer=None):
    """CE and KL against a clean reference under a given ablation scope."""
    handles = []
    if resid_layer is not None:
        mean = means["resid"][resid_layer].to(arm.device)

        def resid_hook(_m, _i, out, mean=mean):
            h = out[0] if isinstance(out, tuple) else out
            new = mean.to(h.dtype).expand_as(h)
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

        handles.append(arm.blocks()[resid_layer].register_forward_hook(resid_hook))
    for layer in ablate_layers:
        mean = means["mlp"][layer].to(arm.device)

        def mlp_hook(_m, _i, out, mean=mean):
            h = out[0] if isinstance(out, tuple) else out
            new = mean.to(h.dtype).expand_as(h)
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

        handles.append(arm.blocks()[layer].mlp.register_forward_hook(mlp_hook))

    ce_total, kl_total, n = 0.0, 0.0, 0
    try:
        for start in range(0, len(texts), batch):
            ids, mask = tokenize_batch(arm, texts[start : start + batch], max_len)
            ids, mask = ids.to(arm.device), mask.to(arm.device)
            valid = (mask[:, 1:] * mask[:, :-1]).bool()
            logp = F.log_softmax(
                arm.model(input_ids=ids, attention_mask=mask).logits[:, :-1].float(),
                dim=-1,
            )
            nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[valid]
            ce_total += float(nll.sum())
            n += int(valid.sum())
            if handles:
                ref = REFERENCE["logp"][start]
                kl = (ref.exp() * (ref - logp.cpu())).sum(-1)[valid.cpu()]
                kl_total += float(kl.sum())
            else:
                REFERENCE["logp"][start] = logp.cpu()
    finally:
        for h in handles:
            h.remove()
    return ce_total / n, (kl_total / n if handles else 0.0)


REFERENCE: dict = {"logp": {}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-seq", type=int, default=120)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--res-min", type=int, default=64)
    ap.add_argument("--res-max", type=int, default=246)
    ap.add_argument("--seed", type=int, default=DEFAULT_COHORT_SEED)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arm = load_arm(args.arm, device=args.device)
    cohort = cohort_for(arm, args.n_seq, args.res_min, args.res_max, seed=args.seed)
    texts = cohort.input_strings(arm)

    means = {"mlp": mlp_output_means(arm, texts, args.max_len, args.batch)}
    # block-output means for the residual-stream reference scope, over the same
    # cohort the ablation is scored on
    store: dict[int, torch.Tensor] = {}
    sums = [torch.zeros(arm.d_model, dtype=torch.float64) for _ in range(arm.n_layer)]
    count = 0
    handles = [
        arm.blocks()[i].register_forward_hook(
            lambda _m, _i, out, i=i: store.__setitem__(
                i, out[0] if isinstance(out, tuple) else out
            )
        )
        for i in range(arm.n_layer)
    ]
    with torch.no_grad():
        for start in range(0, len(texts), args.batch):
            ids, mask = tokenize_batch(arm, texts[start : start + args.batch], args.max_len)
            ids, mask = ids.to(arm.device), mask.to(arm.device)
            arm.model(input_ids=ids, attention_mask=mask)
            keep = mask.bool()
            for i in range(arm.n_layer):
                sums[i] += store[i][keep].double().sum(0).cpu()
            count += int(keep.sum())
            store.clear()
    for h in handles:
        h.remove()
    means["resid"] = [(s / count).float() for s in sums]

    ce_clean, _ = scored_pass(arm, texts, args.max_len, args.batch, [], means)
    print(f"[{arm.name}] clean CE {ce_clean:.4f} nats/token", flush=True)

    mid = analysis_layer(arm.n_layer, 0.5)
    scopes = []
    for frac in (0.15, 0.33, 0.50, 0.67, 0.85):
        layer = analysis_layer(arm.n_layer, frac)
        scopes.append((f"mlp_single_d{frac:.2f}", [layer], None))
    window = list(range(mid, min(mid + 8, arm.n_layer)))
    scopes.append((f"mlp_window8_from_{mid}", window, None))
    scopes.append(("mlp_all", list(range(arm.n_layer)), None))
    scopes.append(("resid_block_d0.50", [], mid))

    rows = []
    for name, layers, resid in scopes:
        ce, kl = scored_pass(arm, texts, args.max_len, args.batch, layers, means, resid)
        row = dict(
            scope=name,
            n_layers_ablated=len(layers) if resid is None else 1,
            ce_delta_nats=ce - ce_clean,
            kl_nats=kl,
            passes_p0_2b_ce_guard=(ce - ce_clean) >= CE_GUARD,
            passes_p0_2b_kl_guard=kl >= KL_GUARD,
        )
        rows.append(row)
        flag = "OK " if row["passes_p0_2b_ce_guard"] else "GUARD-FAIL"
        print(f"  {name:26s} dCE={row['ce_delta_nats']:+.4f}  KL={kl:.4f}  {flag}",
              flush=True)

    single = next(r for r in rows if r["scope"] == "mlp_single_d0.50")
    payload = dict(
        arm=arm.name,
        modality=arm.modality,
        n_layer=arm.n_layer,
        layer_depth_half=mid,
        n_sequences=len(texts),
        seed=args.seed,
        cohort=cohort_provenance(cohort, arm),
        ce_clean_nats=ce_clean,
        p0_2b_ce_guard=CE_GUARD,
        p0_2b_kl_guard=KL_GUARD,
        single_mlp_headroom_nats=single["ce_delta_nats"],
        single_mlp_passes_guard=single["passes_p0_2b_ce_guard"],
        scopes=rows,
    )
    out = Path(args.out) if args.out else (
        REPO / "results/transfer_gap_20260729_corrected/tg10"
    )
    write_json(out / f"{arm.name}.json", payload)


if __name__ == "__main__":
    main()
