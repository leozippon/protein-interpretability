"""TG-08: is the protein dictionary deficit budget-limited or structural?

The preliminary screen (TG-03) trained one dictionary per arm at one budget:
5M activations, x8 expansion, k=32, 6,000 steps. At that point GPT-2-large
recovered 0.968 of its cross-entropy and the protein arms 0.34-0.70. That single
point cannot distinguish

    (a) protein decoders need more dictionary budget, from
    (b) protein residual streams are structurally harder to replace,

and the whole round-2 plan branches on the answer. This script collects
activations once per arm and then trains a grid over training tokens, sparsity
and expansion, so the comparison is a *curve* rather than a point.

Reported per configuration: FVU, dead fraction, absolute cross-entropy increase,
loss recovered against mean-ablation, and KL(clean || spliced), which unlike
loss recovered has no arm-specific denominator. Dictionaries are saved so
downstream analyses do not have to retrain.

**Corrections against the 2026-07-25 run.** Rendering and cohort now come from
`src.transfer.arms` via `tg_common`, so every ProtGPT2 row is retracted and
remeasured. `loss_recovered` was computed here with no denominator guard at all
-- not even the `> 1e-9` test TG-03 had -- so a configuration whose mean-ablation
headroom happened to be small produced a large number rather than a refusal. It
now goes through the shared guard. KL is unaffected and is the quantity to prefer
when the guard refuses, which is the reason it was measured.
"""

from __future__ import annotations

import argparse
import math
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
    loss_recovered,
    tokenize_batch,
    write_json,
)
from tg03_matched_sae import TopKSAE, collect

DEFAULT_OUT = (
    REPO / "results/transfer_gap_20260729_corrected/tg08"
)


def train(acts, d_sae, k, steps, batch, lr, device, seed, log_every=2000):
    torch.manual_seed(seed)
    d_model = acts.shape[1]
    mean = acts.mean(0)
    scale = (acts - mean).norm(dim=1).mean() / math.sqrt(d_model)
    norm = ((acts - mean) / scale).half()
    sae = TopKSAE(d_model, d_sae, k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        idx = torch.randint(0, norm.shape[0], (batch,), generator=g)
        x = norm[idx].to(device, non_blocking=True).float()
        recon, _ = sae(x)
        loss = (recon - x).pow(2).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            sae.decoder.weight.data /= sae.decoder.weight.data.norm(dim=0, keepdim=True)
        opt.step()
        sched.step()
        if step % log_every == 0:
            print(f"      step {step:6d} mse {loss.item():.3f}", flush=True)
    return sae, mean.to(device), scale.to(device)


@torch.no_grad()
def evaluate(arm, eval_acts, eval_texts, layer, sae, mean, scale, max_len, batch):
    mean_c, scale_c = mean.cpu(), scale.cpu()
    centre = (eval_acts - mean_c) / scale_c
    grand = centre.mean(0).to(arm.device)
    num = den = 0.0
    fires = torch.zeros(sae.decoder.weight.shape[1])
    for start in range(0, centre.shape[0], 16384):
        chunk = centre[start : start + 16384].to(arm.device)
        recon, z = sae(chunk)
        num += float((recon - chunk).pow(2).sum())
        den += float((chunk - grand).pow(2).sum())
        fires += (z > 0).float().sum(0).cpu()
    fvu = num / den
    dead = float((fires == 0).float().mean())

    # splice: clean / sae / mean-ablate, plus KL(clean || spliced)
    state = {"mode": "clean"}

    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if state["mode"] == "clean":
            return out
        if state["mode"] == "mean_ablate":
            new = mean.to(h.dtype).expand_as(h)
        else:
            new = (sae((h.float() - mean) / scale)[0] * scale + mean).to(h.dtype)
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

    handle = arm.blocks()[layer].register_forward_hook(hook)
    totals = {m: 0.0 for m in ("clean", "sae", "mean_ablate")}
    counts = {m: 0 for m in totals}
    kl_total, kl_count = 0.0, 0
    try:
        for start in range(0, len(eval_texts), batch):
            ids, mask = tokenize_batch(arm, eval_texts[start : start + batch], max_len)
            ids, mask = ids.to(arm.device), mask.to(arm.device)
            valid = (mask[:, 1:] * mask[:, :-1]).bool()
            target = ids[:, 1:]
            ref = None
            for mode in totals:
                state["mode"] = mode
                logp = F.log_softmax(
                    arm.model(input_ids=ids, attention_mask=mask).logits[:, :-1].float(),
                    dim=-1,
                )
                nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)[valid]
                totals[mode] += float(nll.sum())
                counts[mode] += int(valid.sum())
                if mode == "clean":
                    ref = logp
                elif mode == "sae":
                    kl = (ref.exp() * (ref - logp)).sum(-1)[valid]
                    kl_total += float(kl.sum())
                    kl_count += int(valid.sum())
    finally:
        handle.remove()
    ce = {m: totals[m] / counts[m] for m in totals}
    return dict(
        fvu=fvu,
        dead_fraction=dead,
        ce_clean_nats=ce["clean"],
        ce_sae_nats=ce["sae"],
        ce_mean_ablated_nats=ce["mean_ablate"],
        ce_delta_nats=ce["sae"] - ce["clean"],
        kl_clean_to_spliced_nats=kl_total / kl_count,
        **loss_recovered(ce["clean"], ce["mean_ablate"], ce["sae"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layer-frac", type=float, default=0.5)
    ap.add_argument("--pool-tokens", type=int, default=12_000_000)
    ap.add_argument("--eval-tokens", type=int, default=300_000)
    ap.add_argument("--eval-seqs", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--fwd-batch", type=int, default=8)
    ap.add_argument("--sae-batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=DEFAULT_COHORT_SEED)
    ap.add_argument("--res-min", type=int, default=120)
    ap.add_argument("--res-max", type=int, default=1000)
    ap.add_argument("--save-dictionaries", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_root = Path(args.out) if args.out else DEFAULT_OUT
    arm = load_arm(args.arm, device=args.device)
    layer = analysis_layer(arm.n_layer, args.layer_frac)

    probe = cohort_for(arm, 200, args.res_min, args.res_max,
                       seed=args.seed).input_strings(arm)
    per_seq = sum(
        min(len(arm.tokenizer(t, return_tensors=None)["input_ids"]), args.max_len)
        for t in probe
    ) / len(probe)
    n_train_seq = int(args.pool_tokens / per_seq * 1.3) + 500
    n_eval_seq = int(args.eval_tokens / per_seq * 1.3) + 200
    print(f"[{arm.name}] layer {layer}/{arm.n_layer}  {per_seq:.1f} tok/seq  "
          f"-> {n_train_seq} train sequences", flush=True)

    train_cohort = cohort_for(arm, n_train_seq, args.res_min, args.res_max, seed=args.seed)
    eval_cohort = cohort_for(
        arm, n_eval_seq, args.res_min, args.res_max, skip=n_train_seq, seed=args.seed
    )
    train_texts = train_cohort.input_strings(arm)
    eval_texts = eval_cohort.input_strings(arm)
    pool = collect(arm, train_texts, layer, args.max_len, args.fwd_batch, args.pool_tokens)[0]
    eval_acts = collect(
        arm, eval_texts, layer, args.max_len, args.fwd_batch, args.eval_tokens
    )[0]
    print(f"  pool {tuple(pool.shape)}  eval {tuple(eval_acts.shape)}", flush=True)

    # Four axes, each varied alone from the centre point so that data volume and
    # optimisation compute are not conflated (the preliminary screen scaled them
    # together and could not tell them apart).
    full = min(8_000_000, pool.shape[0])
    centre = (full, 8000, 32, 8)
    grid = [
        centre,
        (full // 16, 8000, 32, 8),   # data axis, steps held fixed
        (full // 4, 8000, 32, 8),
        (full, 2000, 32, 8),         # compute axis, data held fixed
        (full, 24000, 32, 8),
        (full, 8000, 16, 8),         # sparsity axis
        (full, 8000, 64, 8),
        (full, 8000, 128, 8),
        (full, 8000, 32, 4),         # width axis
        (full, 8000, 32, 16),
    ]
    rows = []
    for tokens, steps, k, expansion in grid:
        tokens = min(tokens, pool.shape[0])
        d_sae = expansion * arm.d_model
        tag = f"t{tokens // 1000}k_s{steps}_k{k}_x{expansion}"
        print(f"  -- {tag}: d_sae={d_sae} steps={steps}", flush=True)
        sae, mean, scale = train(
            pool[:tokens], d_sae, k, steps, args.sae_batch, args.lr,
            arm.device, args.seed,
        )
        metrics = evaluate(
            arm, eval_acts, eval_texts[: args.eval_seqs], layer, sae, mean, scale,
            args.max_len, args.fwd_batch,
        )
        rows.append(dict(train_tokens=tokens, k=k, expansion=expansion,
                         d_sae=d_sae, steps=steps, **metrics))
        recovered = metrics["loss_recovered"]
        print(f"     fvu={metrics['fvu']:.4f} dead={metrics['dead_fraction']:.3f} "
              f"dCE={metrics['ce_delta_nats']:.4f} "
              f"LR={'refused' if recovered is None else f'{recovered:.4f}'} "
              f"KL={metrics['kl_clean_to_spliced_nats']:.4f}", flush=True)
        if args.save_dictionaries:
            path = out_root / "dictionaries" / f"{arm.name}_L{layer}_{tag}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": sae.state_dict(), "mean": mean.cpu(),
                        "scale": scale.cpu(), "k": k, "d_sae": d_sae,
                        "layer": layer, "arm": arm.name}, path)
        del sae
        torch.cuda.empty_cache()

    write_json(
        out_root / f"{arm.name}.json",
        dict(arm=arm.name, modality=arm.modality, layer=layer, n_layer=arm.n_layer,
             d_model=arm.d_model, pool_tokens=int(pool.shape[0]),
             eval_tokens=int(eval_acts.shape[0]), tokens_per_sequence=per_seq,
             seed=args.seed, sweep=rows,
             train_cohort=cohort_provenance(train_cohort, arm),
             eval_cohort=cohort_provenance(eval_cohort, arm)),
    )


if __name__ == "__main__":
    main()
