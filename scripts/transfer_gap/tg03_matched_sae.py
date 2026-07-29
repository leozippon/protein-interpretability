"""TG-03: matched sparse-dictionary fidelity, text versus protein.

Trains an identical TopK sparse autoencoder on the layer-L residual stream of
GPT-2-large and of the protein decoders, matched on activation count, hook site,
relative depth, expansion, sparsity, optimizer, steps and evaluation budget. It
then reports the only fidelity number that matters for circuit tracing: how much
of the model's cross-entropy survives when the dictionary reconstruction is
spliced back into the forward pass.

It also asks how much of each learned feature is a function of information that
is *locally available at its own position* (current token identity, position),
which is the ceiling on what a per-position dictionary can report sharply.

**Three corrections against the 2026-07-24 run, whose ProtGPT2 row is retracted.**
ProtGPT2 is now rendered in the FASTA stream it was pretrained on rather than as
one unwrapped line (worth 1.78 nats/token on this cohort, TG-00); cohorts are a
seeded permutation of the eligible corpus rather than its first block; and
``loss_recovered`` refuses to divide by a mean-ablation headroom below a declared
floor rather than reporting a ratio against one. Cross-entropies are additionally
reported over residue-bearing positions alone, because a native rendering puts
FASTA separators and conditioning tags into the scored stream.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
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
    symbol_position_mask,
    tokenize_batch,
    write_json,
)


class TopKSAE(torch.nn.Module):
    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.k = k
        self.pre_bias = torch.nn.Parameter(torch.zeros(d_model))
        self.encoder = torch.nn.Linear(d_model, d_sae)
        self.decoder = torch.nn.Linear(d_sae, d_model, bias=False)
        w = torch.randn(d_sae, d_model) / math.sqrt(d_model)
        self.decoder.weight.data = (w / w.norm(dim=1, keepdim=True)).T.contiguous()
        self.encoder.weight.data = self.decoder.weight.data.T.clone()
        self.encoder.bias.data.zero_()

    def encode(self, x):
        pre = self.encoder(x - self.pre_bias)
        values, index = torch.topk(pre, self.k, dim=-1)
        z = torch.zeros_like(pre).scatter_(-1, index, F.relu(values))
        return z

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z) + self.pre_bias, z


@torch.no_grad()
def collect(arm, texts, layer, max_len, batch, cap):
    """Residual stream after block `layer`, plus token id and position."""
    acts, toks, poss = [], [], []
    total = 0
    store = {}

    def hook(_module, _inp, out):
        store["h"] = out[0] if isinstance(out, tuple) else out

    handle = arm.blocks()[layer].register_forward_hook(hook)
    try:
        for start in range(0, len(texts), batch):
            ids, mask = tokenize_batch(arm, texts[start : start + batch], max_len)
            ids, mask = ids.to(arm.device), mask.to(arm.device)
            arm.model(input_ids=ids, attention_mask=mask)
            h = store.pop("h")
            keep = mask.bool()
            acts.append(h[keep].float().cpu())
            toks.append(ids[keep].cpu())
            poss.append(
                torch.arange(ids.shape[1], device=arm.device)
                .expand_as(ids)[keep]
                .cpu()
            )
            total += int(keep.sum())
            if total >= cap:
                break
    finally:
        handle.remove()
    a = torch.cat(acts)[:cap]
    if a.shape[0] < cap:
        raise RuntimeError(
            f"{arm.name}: cohort exhausted at {a.shape[0]}/{cap} activations; "
            "the matched budget would be violated"
        )
    if not torch.isfinite(a).all():
        raise FloatingPointError(f"non-finite activations for {arm.name}")
    return a, torch.cat(toks)[:cap], torch.cat(poss)[:cap]


def train_sae(acts, d_sae, k, steps, batch, lr, device, seed):
    torch.manual_seed(seed)
    d_model = acts.shape[1]
    mean = acts.mean(0)
    scale = (acts - mean).norm(dim=1).mean() / math.sqrt(d_model)
    # The activation buffer stays in host memory; only minibatches are moved.
    # A multi-million-token buffer does not fit alongside the frozen model.
    norm = ((acts - mean) / scale).half()

    sae = TopKSAE(d_model, d_sae, k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    g = torch.Generator(device="cpu").manual_seed(seed)
    fires = torch.zeros(d_sae, device=device)
    for step in range(steps):
        idx = torch.randint(0, norm.shape[0], (batch,), generator=g)
        x = norm[idx].to(device, non_blocking=True).float()
        recon, z = sae(x)
        loss = (recon - x).pow(2).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():  # unit-norm decoder columns, standard for TopK SAEs
            sae.decoder.weight.data /= sae.decoder.weight.data.norm(dim=0, keepdim=True)
        opt.step()
        sched.step()
        fires += (z > 0).float().sum(0)
        if step % 500 == 0:
            print(f"    step {step:5d} mse {loss.item():.4f}", flush=True)
    return sae, mean.to(device), scale.to(device)


@torch.no_grad()
def splice_eval(arm, texts, layer, sae, mean, scale, max_len, batch):
    """Cross-entropy with the residual stream replaced by three variants.

    Every mode is scored twice: over all valid positions, and over the positions
    whose target token carries at least one alphabet symbol. Under a native
    rendering the two differ -- ProtGPT2's FASTA wrap contributes a newline every
    60 residues -- and a fidelity claim about protein computation should be
    checkable on the residue-bearing subset rather than resting on separators.
    """
    modes = ("clean", "sae", "mean_ablate")
    totals = {m: {"all": 0.0, "symbol": 0.0} for m in modes}
    counts = {m: {"all": 0, "symbol": 0} for m in modes}
    state = {"mode": "clean"}

    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if state["mode"] == "clean":
            return out
        if state["mode"] == "mean_ablate":
            new = mean.to(h.dtype).expand_as(h)
        else:
            x = (h.float() - mean) / scale
            new = (sae(x)[0] * scale + mean).to(h.dtype)
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

    handle = arm.blocks()[layer].register_forward_hook(hook)
    try:
        for start in range(0, len(texts), batch):
            ids, mask = tokenize_batch(arm, texts[start : start + batch], max_len)
            ids, mask = ids.to(arm.device), mask.to(arm.device)
            target = ids[:, 1:]
            valid = (mask[:, 1:] * mask[:, :-1]).bool()
            symbol = symbol_position_mask(arm, target) & valid
            for mode in modes:
                state["mode"] = mode
                logits = arm.model(input_ids=ids, attention_mask=mask).logits
                logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
                nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                totals[mode]["all"] += float(nll[valid].sum())
                counts[mode]["all"] += int(valid.sum())
                totals[mode]["symbol"] += float(nll[symbol].sum())
                counts[mode]["symbol"] += int(symbol.sum())
    finally:
        handle.remove()
    if counts["clean"]["symbol"] == 0:
        raise RuntimeError(f"{arm.name}: no residue-bearing scored positions")
    return {
        m: {k: totals[m][k] / counts[m][k] for k in ("all", "symbol")} for m in modes
    } | {"n_scored": counts["clean"]}


def local_explainability(z, toks, poss, n_features, seed):
    """Variance of each feature explained by current-token identity / position.

    A per-position dictionary can only be a sharp function of what is visible at
    its own position. This is the fraction of feature variance that is.
    """
    rng = np.random.default_rng(seed)
    pick = rng.choice(z.shape[1], size=min(n_features, z.shape[1]), replace=False)
    tok = toks.numpy()
    pos = np.clip(poss.numpy() // 16, 0, 23)
    out_tok, out_pos = [], []
    for f in pick:
        a = z[:, f].numpy().astype(np.float64)
        if a.std() < 1e-9:
            continue
        total = a.var()
        for codes, sink in ((tok, out_tok), (pos, out_pos)):
            uniq, inv = np.unique(codes, return_inverse=True)
            sums = np.bincount(inv, weights=a, minlength=uniq.size)
            n = np.bincount(inv, minlength=uniq.size).astype(np.float64)
            keep = n >= 20
            if keep.sum() < 2:
                continue
            grand = sums[keep].sum() / n[keep].sum()
            between = (n[keep] * (sums[keep] / n[keep] - grand) ** 2).sum() / n[keep].sum()
            # unbiased-ish: subtract the expected between-group variance under
            # random assignment, (g-1)/N * total
            bias = (keep.sum() - 1) / n[keep].sum() * total
            sink.append(max(0.0, float((between - bias) / total)))
    return out_tok, out_pos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layer-frac", type=float, default=0.5)
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--train-tokens", type=int, default=4_000_000)
    ap.add_argument("--eval-tokens", type=int, default=400_000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--sae-batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--fwd-batch", type=int, default=8)
    ap.add_argument("--splice-seqs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=DEFAULT_COHORT_SEED)
    ap.add_argument("--res-min", type=int, default=120)
    ap.add_argument("--res-max", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arm = load_arm(args.arm, device=args.device)
    layer = analysis_layer(arm.n_layer, args.layer_frac)
    d_sae = args.expansion * arm.d_model

    # Cohort size is set from a measured tokens-per-sequence probe. ProtGPT2's
    # multi-residue BPE yields roughly a third as many tokens per protein as the
    # residue-level arms, so a fixed sequence count would silently break the
    # matched activation budget. The probe is drawn from the same permutation as
    # the training cohort, so it measures the ratio on the records that will
    # actually be used.
    probe_texts = cohort_for(arm, 200, args.res_min, args.res_max,
                             seed=args.seed).input_strings(arm)
    per_seq = sum(
        min(len(arm.tokenizer(t, return_tensors=None)["input_ids"]), args.max_len)
        for t in probe_texts
    ) / len(probe_texts)
    n_train_seq = int(args.train_tokens / per_seq * 1.25) + 200
    n_eval_seq = int(args.eval_tokens / per_seq * 1.25) + 100
    print(f"  {per_seq:.1f} tokens/sequence -> {n_train_seq} train sequences")
    train_cohort = cohort_for(arm, n_train_seq, args.res_min, args.res_max, seed=args.seed)
    eval_cohort = cohort_for(
        arm, n_eval_seq, args.res_min, args.res_max, skip=n_train_seq, seed=args.seed
    )
    train_texts = train_cohort.input_strings(arm)
    eval_texts = eval_cohort.input_strings(arm)

    print(f"[{arm.name}] layer {layer}/{arm.n_layer}  d_sae={d_sae}  k={args.k}")
    train_acts, _, _ = collect(
        arm, train_texts, layer, args.max_len, args.fwd_batch, args.train_tokens
    )
    print(f"  collected {train_acts.shape[0]} train activations")
    sae, mean, scale = train_sae(
        train_acts, d_sae, args.k, args.steps, args.sae_batch, args.lr,
        arm.device, args.seed,
    )
    del train_acts
    torch.cuda.empty_cache()

    eval_acts, eval_toks, eval_poss = collect(
        arm, eval_texts, layer, args.max_len, args.fwd_batch, args.eval_tokens
    )
    with torch.no_grad():
        mean_c, scale_c = mean.cpu(), scale.cpu()
        centre = ((eval_acts - mean_c) / scale_c)
        grand = centre.mean(0)
        num = den = 0.0
        fires = torch.zeros(d_sae)
        z_rows = []
        for start in range(0, centre.shape[0], 16384):
            chunk = centre[start : start + 16384].to(arm.device)
            recon, z = sae(chunk)
            num += float((recon - chunk).pow(2).sum())
            den += float((chunk - grand.to(arm.device)).pow(2).sum())
            fires += (z > 0).float().sum(0).cpu()
            z_rows.append(z.cpu())
        fvu = num / den
        z_cpu = torch.cat(z_rows)
        alive = fires > 0
        fire_rate = fires / centre.shape[0]
        del centre, z_rows
    torch.cuda.empty_cache()

    ce = splice_eval(
        arm, eval_texts[: args.splice_seqs], layer, sae, mean, scale,
        args.max_len, args.fwd_batch,
    )
    recovered = loss_recovered(
        ce["clean"]["all"], ce["mean_ablate"]["all"], ce["sae"]["all"]
    )
    recovered_symbol = loss_recovered(
        ce["clean"]["symbol"], ce["mean_ablate"]["symbol"], ce["sae"]["symbol"]
    )

    tok_r2, pos_r2 = local_explainability(z_cpu, eval_toks, eval_poss, 512, args.seed)

    payload = dict(
        arm=arm.name,
        modality=arm.modality,
        layer=layer,
        n_layer=arm.n_layer,
        d_model=arm.d_model,
        d_sae=d_sae,
        k=args.k,
        seed=args.seed,
        train_tokens=int(args.train_tokens),
        tokens_per_sequence=per_seq,
        eval_tokens=int(eval_acts.shape[0]),
        steps=args.steps,
        train_cohort=cohort_provenance(train_cohort, arm),
        eval_cohort=cohort_provenance(eval_cohort, arm),
        fvu=fvu,
        variance_explained=1.0 - fvu,
        dead_fraction=float((~alive).float().mean()),
        frac_features_below_1e4_firing=float((fire_rate < 1e-4).float().mean()),
        ce_clean_nats=ce["clean"]["all"],
        ce_sae_spliced_nats=ce["sae"]["all"],
        ce_mean_ablated_nats=ce["mean_ablate"]["all"],
        ce_delta_nats=ce["sae"]["all"] - ce["clean"]["all"],
        ce_clean_nats_symbol_positions=ce["clean"]["symbol"],
        ce_sae_spliced_nats_symbol_positions=ce["sae"]["symbol"],
        ce_mean_ablated_nats_symbol_positions=ce["mean_ablate"]["symbol"],
        n_scored_positions=ce["n_scored"]["all"],
        n_scored_symbol_positions=ce["n_scored"]["symbol"],
        symbol_position_share=ce["n_scored"]["symbol"] / ce["n_scored"]["all"],
        loss_recovered=recovered["loss_recovered"],
        loss_recovered_symbol_positions=recovered_symbol["loss_recovered"],
        ablation_headroom_nats=recovered["ablation_headroom_nats"],
        denominator_valid=recovered["denominator_valid"],
        denominator_floor_nats=recovered["denominator_floor_nats"],
        denominator_refusal=recovered["denominator_refusal"],
        feature_variance_explained_by_current_token=dict(
            n=len(tok_r2),
            mean=float(np.mean(tok_r2)),
            median=float(np.median(tok_r2)),
            frac_above_0p5=float(np.mean(np.asarray(tok_r2) > 0.5)),
        ),
        feature_variance_explained_by_position=dict(
            n=len(pos_r2),
            mean=float(np.mean(pos_r2)),
            median=float(np.median(pos_r2)),
            frac_above_0p5=float(np.mean(np.asarray(pos_r2) > 0.5)),
        ),
    )
    out = Path(args.out) if args.out else (
        REPO / "results/transfer_gap_20260729_corrected/tg03"
    )
    write_json(out / f"{arm.name}_L{layer}_k{args.k}.json", payload)
    for key in ("fvu", "dead_fraction", "symbol_position_share", "ce_clean_nats",
                "ce_sae_spliced_nats", "ce_mean_ablated_nats", "ablation_headroom_nats",
                "denominator_valid", "loss_recovered", "loss_recovered_symbol_positions"):
        print(f"  {key:34s} {payload[key]}")
    print(f"  token-explained feature var (mean) {payload['feature_variance_explained_by_current_token']['mean']:.3f}")


if __name__ == "__main__":
    main()
