"""TG-01: how much output-proximal computation is there to explain?

Matched measurement of the predictive-information budget of a text decoder and
three protein decoders. Everything a circuit-tracing method can explain about
next-symbol prediction is bounded by the information the model actually commits
per symbol relative to a context-free baseline, and by how much of that
information requires long context.

Outputs one JSON per arm under results/transfer_gap_20260729_corrected/tg01/.

**Superseded in part** by `scripts/transfer/01_cohort_power.py`, which computes
context information over the current eleven-arm panel with the held-out estimator.
What survives here and is not there: the order-of-magnitude truncation curve, the
symbol-level Markov ladder, and the concentration statistics on the per-token
information gain.

Corrections against the 2026-07-24 run:

*Rendering and cohort.* Both now come from `src.transfer.arms` through
`tg_common`: per-arm native input format, seeded-permutation cohorts.

*Estimator.* The context-free baseline was the **plug-in** entropy of the
estimation cohort, `H(p-hat)` evaluated on the same sample that fitted `p-hat`.
That is downward-biased by an amount that scales with vocabulary size, which is
precisely the axis this panel varies: EXP-R2-059 measured the bias at +1.291 nats
for ProtGPT2 and +0.301 for GPT-2-large against +0.007 and +0.014 for the
residue-level arms. Using it would inflate the two 50k-vocabulary arms and no
others. The baseline is now the **held-out** cross-entropy of that same unigram
model on the disjoint evaluation cohort; the plug-in value is retained beside it
so the bias is visible rather than assumed.

*Denominators.* The long-range fractions divide by `baseline - NLL(longest
context)`, which on ZymCTRL was 0.386 nats and produced a "fraction" of 1.0197.
That denominator is now floored and the fractions refused below it.

*Conditioned arms.* A truncation window cuts the conditioning prefix off an
EC-conditioned arm, so its short-context points are not that model's behaviour
under short context, they are that model unconditioned. ZymCTRL read 21.3
nats/token at context 1 for this reason. The curve is still reported, flagged,
and the fractions derived from it are refused.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tg_common import (
    AA20,
    DEFAULT_COHORT_SEED,
    REPO,
    Arm,
    cohort_for,
    cohort_provenance,
    load_arm,
    tokenize_batch,
    write_json,
)
from src.transfer.arms import PANEL

LN2 = math.log(2.0)
CONTEXTS = [1, 2, 4, 8, 16, 32, 64, 128]

#: Floor on ``baseline - NLL(longest context)``, the denominator of every
#: long-range fraction below. Same floor and same reason as the mean-ablation
#: headroom in TG-03/TG-07: a ratio against a few tenths of a nat is not a weak
#: measurement of the share of information that is long-range, it is arithmetic
#: on noise.
MIN_INFORMATION_RANGE_NATS = 0.5

#: Input formats that carry conditioning the truncation windows would destroy.
_CONDITIONED_FORMATS = frozenset({"ec_conditioned"})


@torch.no_grad()
def full_context_pass(arm: Arm, texts: list[str], max_len: int, batch: int):
    """Per-token NLL, predictive entropy, top-1 hit, token id, position."""
    rows = []
    for start in range(0, len(texts), batch):
        chunk = texts[start : start + batch]
        ids, mask = tokenize_batch(arm, chunk, max_len)
        ids, mask = ids.to(arm.device), mask.to(arm.device)
        logits = arm.model(input_ids=ids, attention_mask=mask).logits.float()
        logp = F.log_softmax(logits[:, :-1], dim=-1)
        target = ids[:, 1:]
        valid = (mask[:, 1:] * mask[:, :-1]).bool()
        nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        ent = -(logp.exp() * logp).sum(-1)
        hit = logp.argmax(-1).eq(target)
        pos = torch.arange(1, ids.shape[1], device=arm.device).expand_as(target)
        seq_len = mask.sum(1, keepdim=True).expand_as(target)
        for tensor, name in ((nll, "nll"), (ent, "ent")):
            if not torch.isfinite(tensor[valid]).all():
                raise FloatingPointError(f"non-finite {name} in {arm.name}")
        rows.append(
            np.stack(
                [
                    nll[valid].cpu().numpy(),
                    ent[valid].cpu().numpy(),
                    hit[valid].float().cpu().numpy(),
                    target[valid].cpu().numpy().astype(np.float64),
                    pos[valid].cpu().numpy().astype(np.float64),
                    seq_len[valid].cpu().numpy().astype(np.float64),
                ],
                axis=1,
            )
        )
    return np.concatenate(rows, axis=0)


@torch.no_grad()
def truncation_curve(arm: Arm, texts: list[str], max_len: int, n_query: int,
                     batch: int, seed: int):
    """NLL at sampled query positions as a function of visible context length."""
    rng = np.random.default_rng(seed)
    windows = []  # (token id list up to and including query, target)
    for text in texts:
        ids = arm.tokenizer(text, return_tensors=None)["input_ids"][:max_len]
        if len(ids) < CONTEXTS[-1] + 3:
            continue
        lo = CONTEXTS[-1]
        hi = len(ids) - 1
        picks = rng.choice(np.arange(lo, hi), size=min(n_query, hi - lo), replace=False)
        for q in picks:
            windows.append((ids, int(q)))
    if len(windows) < 200:
        raise RuntimeError(f"{arm.name}: only {len(windows)} query positions")

    # Only the final logit row is needed. The 50k-vocabulary arms must use the
    # trimmed-head path or the (batch, ctx, vocab) tensor dominates memory; the
    # local ProGen2 implementation has no such argument but a 32-token vocab,
    # so the full tensor is negligible there.
    import inspect

    trimmed = "logits_to_keep" in inspect.signature(arm.model.forward).parameters
    if not trimmed and arm.model.config.vocab_size > 1024:
        raise RuntimeError(f"{arm.name}: large vocab without logits_to_keep support")

    out = {}
    for ctx in CONTEXTS:
        nlls = []
        for start in range(0, len(windows), batch):
            chunk = windows[start : start + batch]
            block = torch.tensor(
                [ids[q - ctx : q + 1] for ids, q in chunk], dtype=torch.long
            ).to(arm.device)
            kwargs = {"logits_to_keep": 1} if trimmed else {}
            logits = arm.model(input_ids=block[:, :-1], **kwargs).logits
            logp = F.log_softmax(logits[:, -1].float(), dim=-1)
            nlls.append(-logp.gather(-1, block[:, -1:]).squeeze(-1).cpu().numpy())
        out[ctx] = float(np.concatenate(nlls).mean())
    return out, len(windows)


def unigram_model(token_ids: np.ndarray, vocab: int) -> np.ndarray:
    """Add-one-smoothed unigram distribution fitted on ``token_ids``.

    Smoothing is not cosmetic here. The held-out evaluation cohort contains
    tokens the estimation cohort never produced, and an unsmoothed model assigns
    them probability zero, which makes the held-out cross-entropy infinite for
    exactly the large-vocabulary arms whose baseline this is meant to fix.
    """

    counts = np.bincount(token_ids.astype(np.int64), minlength=vocab).astype(np.float64)
    return (counts + 1.0) / (counts.sum() + vocab)


def plug_in_entropy(token_ids: np.ndarray, vocab: int) -> float:
    """``H(p-hat)`` on the sample that fitted ``p-hat``. Reported, never used.

    Kept so that the bias it carries is a measured number in the artefact rather
    than a claim in a docstring. It must not be substituted for the held-out
    cross-entropy: the substitution is limitation L12 and it moved a published
    decomposition by up to 1.02 nats.
    """

    counts = np.bincount(token_ids.astype(np.int64), minlength=vocab).astype(np.float64)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def ratio_or_refusal(numerator: float, denominator: float, floor: float):
    """A share, or ``None`` when its denominator is too small to be one."""

    return numerator / denominator if denominator >= floor else None


def markov_bits_per_symbol(train: list[str], test: list[str], order: int,
                           alphabet: str) -> float:
    """Held-out cross-entropy (bits/symbol) of an order-k Markov baseline.

    Tokenizer-independent, so ProtGPT2 (BPE) and the residue-level protein
    models are scored on the same axis.
    """
    index = {c: i for i, c in enumerate(alphabet)}
    size = len(alphabet)
    shape = (size,) * order + (size,)
    counts = np.ones(shape, dtype=np.float64)  # Laplace
    for seq in train:
        idx = [index[c] for c in seq if c in index]
        for i in range(order, len(idx)):
            counts[tuple(idx[i - order : i]) + (idx[i],)] += 1.0
    probs = counts / counts.sum(axis=-1, keepdims=True)
    total, n = 0.0, 0
    for seq in test:
        idx = [index[c] for c in seq if c in index]
        for i in range(order, len(idx)):
            total -= math.log2(probs[tuple(idx[i - order : i]) + (idx[i],)])
            n += 1
    if n == 0:
        raise RuntimeError("empty Markov evaluation set")
    return total / n


def gini(x: np.ndarray) -> float:
    y = np.sort(np.clip(x, 0, None))
    n = y.size
    if y.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * y).sum()) / (n * y.sum()) - (n + 1) / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-seq", type=int, default=400)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--trunc-batch", type=int, default=96)
    ap.add_argument("--n-query", type=int, default=6)
    ap.add_argument("--seed", type=int, default=DEFAULT_COHORT_SEED)
    ap.add_argument("--res-min", type=int, default=400)
    ap.add_argument("--res-max", type=int, default=1000)
    ap.add_argument("--n-unigram", type=int, default=8000,
                    help="disjoint cohort size for the unigram/Markov baselines")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arm = load_arm(args.arm, device=args.device)

    cohort = cohort_for(arm, args.n_seq, args.res_min, args.res_max, seed=args.seed)
    texts, raw = cohort.input_strings(arm), cohort.records
    symbol_name = "character" if arm.modality == "text" else "residue"

    rows = full_context_pass(arm, texts, args.max_len, args.batch)
    nll, ent, hit, tok, pos, seq_len = (rows[:, i] for i in range(6))

    vocab = arm.model.config.vocab_size

    # The unigram baseline is fitted on a much larger disjoint cohort and then
    # scored *held out* on the evaluation cohort. Fitting and evaluating on one
    # sample is the plug-in estimator, whose bias tracks vocabulary size and so
    # falls almost entirely on the two arms whose relative position matters.
    base_cohort = cohort_for(
        arm, args.n_unigram, args.res_min, args.res_max, skip=args.n_seq, seed=args.seed
    )
    base_texts, base_raw = base_cohort.input_strings(arm), base_cohort.records
    base_tokens = np.concatenate(
        [
            np.asarray(arm.tokenizer(t, return_tensors=None)["input_ids"][: args.max_len],
                       dtype=np.int64)
            for t in base_texts
        ]
    )
    p_uni = unigram_model(base_tokens, vocab)
    logp_uni = np.log(p_uni[tok.astype(np.int64)])
    h_uni = float((-logp_uni).mean())          # held-out cross-entropy: the baseline
    h_uni_plug_in = plug_in_entropy(base_tokens, vocab)  # reported for its bias only
    gain = (-logp_uni) - nll  # nats, positive = model beats unigram

    curve, n_windows = truncation_curve(
        arm, texts, args.max_len, args.n_query, args.trunc_batch, args.seed
    )

    # Tokenizer expansion, measured over exactly the scored window: symbols are
    # counted after truncation, otherwise long sequences inflate the ratio.
    n_tokens, n_symbols = 0, 0
    for text in texts:
        ids = arm.tokenizer(text, return_tensors=None)["input_ids"][: args.max_len]
        decoded = arm.tokenizer.decode(ids)
        n_tokens += len(ids)
        if arm.modality == "protein":
            n_symbols += sum(1 for c in decoded if c in AA20)
        else:
            n_symbols += len(decoded)
    symbols_per_token = n_symbols / n_tokens

    order = np.argsort(-gain)
    top10 = order[: max(1, int(0.1 * gain.size))]
    pos_bin = np.clip((pos / np.maximum(seq_len - 1, 1) * 20).astype(int), 0, 19)

    # Symbol-level (tokenizer-independent) reference ladder. For proteins this
    # is the 20-letter residue alphabet, which is the axis on which BPE and
    # residue-level protein models can be compared to each other at all.
    if arm.modality == "protein":
        markov = {
            f"order{k}_bits_per_residue": markov_bits_per_symbol(
                base_raw[:4000], raw, k, AA20
            )
            for k in (0, 1, 2)
        }
    else:
        markov = {}

    # Every long-range share divides by the information the model extracts at its
    # longest context relative to the context-free baseline. Compute it once,
    # guard it once, and refuse every derived share together when it is too small
    # -- reporting three of them separately would let a reader take the one that
    # happened to look sane.
    information_range = h_uni - curve[CONTEXTS[-1]]
    conditioned = PANEL[arm.name].input_format in _CONDITIONED_FORMATS
    range_valid = information_range >= MIN_INFORMATION_RANGE_NATS and not conditioned

    def share(numerator: float):
        if not range_valid:
            return None
        return ratio_or_refusal(numerator, information_range, MIN_INFORMATION_RANGE_NATS)

    payload = dict(
        arm=arm.name,
        modality=arm.modality,
        symbol=symbol_name,
        seed=args.seed,
        n_sequences=len(texts),
        n_scored_tokens=int(gain.size),
        vocab_size=int(vocab),
        symbols_per_token=symbols_per_token,
        max_len=args.max_len,
        # Read from the panel, not from an argument. This used to be
        # `args.protein_source`, whose default was the literal "swissprot" and
        # which nothing consumed: `tg_common.cohort_for` dispatches on
        # `PANEL[arm.name].source`. Every ZymCTRL artefact therefore asserted a
        # corpus it was not drawn from -- `results/transfer_gap_20260724/tg01/
        # zymctrl.json` records `protein_source: "swissprot"` against an actual
        # source of `zymctrl_ec`, the EC-labelled FASTA whose conditioning tag is
        # separately priced at 1.73 nats.
        protein_source=(
            PANEL[arm.name].evaluation_cohort_source if arm.modality == "protein" else None
        ),
        cohort=cohort_provenance(cohort, arm),
        baseline_cohort=cohort_provenance(base_cohort, arm),
        # --- core budget, nats/token unless noted
        symbol_level_markov_baselines=markov,
        unigram_entropy_nats=h_uni,
        unigram_estimator="held_out_cross_entropy",
        unigram_plug_in_entropy_nats=h_uni_plug_in,
        unigram_plug_in_bias_nats=h_uni - h_uni_plug_in,
        model_nll_nats=float(nll.mean()),
        model_pred_entropy_nats=float(ent.mean()),
        top1_accuracy=float(hit.mean()),
        info_gain_over_unigram_nats=float(gain.mean()),
        info_gain_over_unigram_bits=float(gain.mean() / LN2),
        info_gain_bits_per_symbol=float(gain.mean() / LN2 / symbols_per_token),
        unigram_entropy_bits_per_symbol=float(h_uni / LN2 / symbols_per_token),
        model_nll_bits_per_symbol=float(nll.mean() / LN2 / symbols_per_token),
        fraction_of_unigram_entropy_explained=float(gain.mean() / h_uni),
        # --- concentration of that information
        gain_gini=gini(gain),
        gain_top_decile_share=(
            float(gain[top10].sum() / gain.sum()) if gain.sum() > 1e-6 else None
        ),
        gain_quantiles_nats={
            str(q): float(np.quantile(gain, q))
            for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
        },
        frac_tokens_gain_below_0p1_nats=float((gain < 0.1).mean()),
        frac_tokens_model_worse_than_unigram=float((gain < 0).mean()),
        # --- where in the sequence the information sits
        gain_by_position_bin_nats=[
            float(gain[pos_bin == b].mean()) for b in range(20)
        ],
        gain_first_token_nats=float(gain[pos == 1].mean()),
        gain_first_five_tokens_nats=float(gain[pos <= 5].mean()),
        # --- how much of it needs long context
        truncation_curve_nll_nats={str(k): v for k, v in curve.items()},
        n_truncation_windows=n_windows,
        truncation_strips_conditioning=conditioned,
        information_range_nats=information_range,
        information_range_valid=range_valid,
        information_range_floor_nats=MIN_INFORMATION_RANGE_NATS,
        information_range_refusal=(
            None
            if range_valid
            else (
                "truncation removes this arm's conditioning prefix, so its "
                "short-context points are the unconditioned model"
                if conditioned
                else "baseline minus longest-context NLL is below the floor"
            )
        ),
        # referenced to the longest truncation, not to the full pass: the two
        # use different query-position distributions and are not comparable
        long_range_fraction_beyond_8=share(curve[8] - curve[CONTEXTS[-1]]),
        local_fraction_within_8=share(h_uni - curve[8]),
        long_range_fraction_beyond_32=share(curve[32] - curve[CONTEXTS[-1]]),
        long_range_bits_beyond_8=float((curve[8] - curve[CONTEXTS[-1]]) / LN2),
        markov_order2_bits_per_symbol=markov.get("order2_bits_per_residue"),
        long_range_bits_beyond_32=float((curve[32] - curve[CONTEXTS[-1]]) / LN2),
    )
    out = Path(args.out) if args.out else (
        REPO / "results/transfer_gap_20260729_corrected/tg01"
    )
    write_json(out / f"{arm.name}.json", payload)
    for key in (
        "unigram_entropy_nats",
        "unigram_plug_in_entropy_nats",
        "unigram_plug_in_bias_nats",
        "model_nll_nats",
        "info_gain_over_unigram_bits",
        "info_gain_bits_per_symbol",
        "top1_accuracy",
        "gain_top_decile_share",
        "information_range_nats",
        "long_range_fraction_beyond_8",
    ):
        value = payload[key]
        print(f"  {key:42s} " + ("None" if value is None else f"{value:.4f}"))


if __name__ == "__main__":
    main()
