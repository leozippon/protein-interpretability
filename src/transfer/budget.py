"""Stage-1 substrate measurement: is an evaluation cohort measurable at all?

Every normalised interpretability score - loss recovered, KL recovered, a
mutual-information gate - is a ratio whose denominator is the information the
model actually commits on the cohort relative to a context-free baseline. If
that denominator is small the ratio is noise; if it is negative the arm is
off-distribution and the ratio is meaningless.

The motivating history is worth stating precisely, because the first version of
this docstring quoted figures that were themselves measurement artefacts and
have since been retracted. What actually happened: a production qualification on
a 64-246 residue EC-labelled cohort appeared to show ProGen2-medium starved at
0.099 nats/token and ProtGPT2 off-distribution at -1.73 nats/token. Neither
survived. The ProGen2 figure came from comparing against a *different* cohort
(plain rather than EC-labelled Swiss-Prot); its true value there is 1.24-1.52.
The ProtGPT2 figure came from rendering it as one unwrapped line when it was
pretrained on FASTA-formatted UniRef50, worth 1.42 nats/token; measured across
four cohorts under the correct rendering it is +0.21 to +3.89 and has never
reproduced as negative.

The lesson is not that the power check is unnecessary - it is that the power
check must be applied to a cohort the arm is actually native to, and with an
unbiased baseline. A plug-in unigram estimate computed on the scored tokens
understates entropy by 0.75 nats for a 50k-vocabulary text arm and 1.65 nats for
a 50k-vocabulary protein arm while leaving residue-level arms untouched, which
inflates every share computed from it by up to 74 per cent.

This module measures that power figure *before* any scientific gate is applied
so that a starved arm is reported as unmeasurable on the cohort rather than as
a failed scientific hypothesis.

Per-token quantities are tokenizer-dependent and are therefore not comparable
across arms: ProtGPT2 uses multi-residue BPE while ZymCTRL and ProGen2-medium
are residue-level. The held-out residue Markov ladder is the only
tokenizer-independent axis on which the protein arms can be compared to each
other, so it is reported alongside every per-token figure.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import transformers

from .arms import (
    AA20,
    Arm,
    Cohort,
    conditioning_boundary_ids,
    symbols_per_token,
    tokenize_batch,
)
from .scoring import sequence_target_mask, target_rule
from .statistics import mean_interval

LN2 = math.log(2.0)

#: Default visible-context lengths for the truncation curve. Powers of two span
#: the range over which a protein decoder's local statistics saturate.
DEFAULT_CONTEXT_LENGTHS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)

#: Minimum context-derived information for an arm to be measurable on a cohort.
#: Below this the denominator of any recovery ratio is comparable to its own
#: sampling error, so the arm carries no usable signal.
MIN_CONTEXT_INFORMATION_NATS = 0.30

MEASURABLE = "measurable"
UNMEASURABLE = "unmeasurable_on_this_cohort"

#: The same two verdicts, qualified by the estimator that produced them.
#:
#: :func:`arm_power` decides measurability against a *plug-in* unigram baseline
#: unless the caller supplies a held-out one, and ``pathways.UNIGRAM_ESTIMATORS``
#: declares the plug-in to be "an explicit opt-in diagnostic" whose bias "grows
#: with vocabulary against sample size" -- roughly +0.003 nats at 32 symbols
#: against +1.65 at 50257 pieces on a cohort of this size. That differential is
#: several times :data:`MIN_CONTEXT_INFORMATION_NATS`, so which arms pass a
#: plug-in verdict is set mostly by vocabulary size.
#:
#: ``01_cohort_power.py`` recomputes the verdict from the held-out estimator
#: whenever a reference cohort is given, so no published verdict came from the
#: plug-in. The defect is that nothing stopped a *caller* inheriting one: the
#: estimator was recorded in a different field from the verdict, and a reader
#: who took ``measurability`` alone could not tell which they had. The verdict
#: now carries its estimator in its own value when that estimator is the
#: diagnostic one, which is the only form a reader cannot separate them in.
MEASURABLE_PLUG_IN = "measurable_against_plug_in_baseline"
UNMEASURABLE_PLUG_IN = "unmeasurable_on_this_cohort_against_plug_in_baseline"


@dataclass(frozen=True)
class ScoredTokens:
    """Next-token targets that belong to the cohort's own content.

    Padding, the first token of every sequence, and - for EC-conditioned
    ZymCTRL inputs - the conditioning prefix and the terminator are excluded,
    so the unigram baseline and the model cross-entropy are estimated on
    exactly the same token multiset.
    """

    target_ids: np.ndarray
    nll_nats: np.ndarray
    sequence_index: np.ndarray

    def __post_init__(self) -> None:
        if self.target_ids.ndim != 1:
            raise ValueError("scored-token arrays must be one-dimensional")
        if self.nll_nats.shape != self.target_ids.shape:
            raise ValueError("scored-token NLL does not align with targets")
        if self.sequence_index.shape != self.target_ids.shape:
            raise ValueError("scored-token sequence index does not align with targets")
        if self.target_ids.size == 0:
            raise ValueError("no scored tokens were produced")

    def __len__(self) -> int:
        return int(self.target_ids.size)


@torch.no_grad()
def scored_tokens(
    arm: Arm,
    input_strings: Sequence[str],
    *,
    max_len: int,
    batch_size: int,
) -> ScoredTokens:
    """Per-token clean negative log-likelihood over the cohort's scored targets."""

    if not input_strings:
        raise ValueError(f"{arm.name}: empty cohort")
    if max_len < 2:
        raise ValueError("max_len must admit at least one next-token target")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    rule = target_rule(arm.spec.input_format)
    start_id, end_id = conditioning_boundary_ids(arm)
    conditioned = start_id is not None

    targets: list[np.ndarray] = []
    losses: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for offset in range(0, len(input_strings), batch_size):
        chunk = list(input_strings[offset : offset + batch_size])
        ids, mask = tokenize_batch(arm, chunk, max_len)
        if conditioned:
            complete = (ids == end_id).sum(dim=1).eq(1) & (ids == start_id).sum(dim=1).eq(1)
            if not bool(complete.all()):
                raise ValueError(
                    f"{arm.name}: max_len={max_len} truncates the EC-conditioned prompt "
                    "before its <end> boundary; the scored window would be undefined"
                )
        ids = ids.to(arm.device)
        mask = mask.to(arm.device)
        logits = arm.model(input_ids=ids, attention_mask=mask).logits
        logprobs = F.log_softmax(logits[:, :-1].float(), dim=-1)
        target = ids[:, 1:]
        nll = -logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        keep = sequence_target_mask(
            ids,
            mask,
            rule=rule,
            start_token_id=start_id,
            end_token_id=end_id,
        )
        selected = nll[keep]
        if not bool(torch.isfinite(selected).all()):
            raise FloatingPointError(f"{arm.name}: non-finite clean NLL")
        row = torch.arange(ids.shape[0], device=arm.device).unsqueeze(1).expand_as(target)
        targets.append(target[keep].cpu().numpy().astype(np.int64))
        losses.append(selected.cpu().numpy().astype(np.float64))
        indices.append((row[keep] + offset).cpu().numpy().astype(np.int64))
    return ScoredTokens(
        target_ids=np.concatenate(targets),
        nll_nats=np.concatenate(losses),
        sequence_index=np.concatenate(indices),
    )


def scored_symbols_per_token(arm: Arm, scored: ScoredTokens) -> float:
    """Tokenizer expansion over exactly the tokens the cross-entropy was taken on.

    ``arms.symbols_per_token`` measures the expansion of the *whole rendered
    string*: every token the tokenizer produced, against every alphabet symbol
    it decodes to. That is the right quantity for describing a rendering and the
    wrong one for converting a per-token cross-entropy into bits per symbol,
    because ``clean_ce`` and the unigram baseline are taken over
    :class:`ScoredTokens` -- which for an EC-conditioned arm excludes the EC
    tag, ``<sep>``, ``<start>`` and ``<end>``.

    Those prompt tokens raise the token count and contribute no AA20 symbols, so
    the whole-string expansion is *understated* for ZymCTRL alone, and a
    per-symbol figure divides by it. Every ``*_bits_per_symbol`` field was
    therefore inflated for exactly one arm, on the axis this module declares to
    be the cross-arm comparable one. ProtGPT2 is unaffected: its newlines are
    scored like any other token, so its scored window and its rendered string
    are the same multiset apart from the first token.

    Computed here rather than in ``arms`` because ``arms.symbols_per_token`` has
    a legitimate second caller and a legitimate second meaning; this is a third
    quantity, not a correction to that one, and both are reported.
    """

    decoded = arm.tokenizer.decode([int(value) for value in scored.target_ids])
    symbols = (
        sum(1 for character in decoded if character in AA20)
        if arm.modality == "protein"
        else len(decoded)
    )
    if symbols < 1:
        raise RuntimeError(
            f"{arm.name}: the scored targets decode to no alphabet symbols, so a "
            "per-symbol conversion is undefined"
        )
    return symbols / len(scored)


def unigram_entropy_nats(target_ids: np.ndarray, vocab_size: int) -> float:
    """Plug-in entropy of the cohort's own scored-token distribution.

    This is the context-free baseline an arm has to beat. Estimating it on the
    cohort itself rather than on a held-out corpus is deliberate: it is the
    tightest baseline available to that arm on that cohort, so the resulting
    context-information figure is a conservative lower bound.
    """

    if target_ids.ndim != 1 or target_ids.size == 0:
        raise ValueError("target_ids must be a non-empty one-dimensional array")
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    if int(target_ids.min()) < 0 or int(target_ids.max()) >= vocab_size:
        raise ValueError("target_ids fall outside the declared vocabulary")
    counts = np.bincount(target_ids, minlength=vocab_size).astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def miller_madow_entropy_nats(target_ids: np.ndarray, vocab_size: int) -> float:
    """Miller-Madow corrected entropy, in nats.

    The plug-in estimator is biased downwards, severely so for the 50k-token
    arms on a cohort of this size. The correction is reported as a diagnostic
    because it moves the baseline, and therefore the headline figure, upwards.
    """

    plugin = unigram_entropy_nats(target_ids, vocab_size)
    observed = int(np.unique(target_ids).size)
    return plugin + (observed - 1) / (2.0 * target_ids.size)


def markov_cross_entropy_bits(
    train_sequences: Sequence[str],
    test_sequences: Sequence[str],
    *,
    order: int,
    alphabet: str = AA20,
) -> float:
    """Held-out order-``k`` Markov cross-entropy in bits per symbol.

    Tokenizer-independent by construction, which is the only way a
    multi-residue-BPE arm and a residue-level arm can be placed on one axis.
    Train and test sets must be disjoint or the number is a training fit.
    """

    if order < 0:
        raise ValueError("Markov order must be non-negative")
    if len(set(alphabet)) != len(alphabet) or not alphabet:
        raise ValueError("alphabet must be non-empty and free of duplicates")
    if not train_sequences or not test_sequences:
        raise ValueError("both a training and a test sequence set are required")
    shared = set(train_sequences) & set(test_sequences)
    if shared:
        raise ValueError(
            f"Markov train and test sets share {len(shared)} sequences; "
            "held-out cross-entropy requires disjoint sets"
        )

    index = {symbol: position for position, symbol in enumerate(alphabet)}
    size = len(alphabet)
    counts = np.ones((size,) * order + (size,), dtype=np.float64)
    for sequence in train_sequences:
        encoded = [index[symbol] for symbol in sequence if symbol in index]
        for position in range(order, len(encoded)):
            counts[tuple(encoded[position - order : position]) + (encoded[position],)] += 1.0
    probabilities = counts / counts.sum(axis=-1, keepdims=True)

    total = 0.0
    scored = 0
    for sequence in test_sequences:
        encoded = [index[symbol] for symbol in sequence if symbol in index]
        for position in range(order, len(encoded)):
            probability = probabilities[
                tuple(encoded[position - order : position]) + (encoded[position],)
            ]
            total -= math.log2(probability)
            scored += 1
    if scored == 0:
        raise ValueError("the Markov test set contains no scorable symbols")
    return total / scored


#: Substring length at which a shared exact match between a training and a test
#: protein is evidence of family membership rather than of chance. Over a
#: twenty-letter alphabet a specific 30-mer has probability 20**-30; a corpus of
#: 10**10 residues expects none by chance, so every shared 30-mer is homology.
NEAR_DUPLICATE_KMER = 30


def near_duplicate_fraction(
    train_sequences: Sequence[str], test_sequences: Sequence[str], *, k: int = NEAR_DUPLICATE_KMER
) -> dict[str, Any]:
    """Share of test sequences that have a ``k``-residue exact match in the train set.

    :func:`markov_cross_entropy_bits` enforces "held out" by exact string
    identity, which catches byte-identical records and nothing else -- while the
    hazard this programme has actually declared for these corpora is *near-clonal
    family grouping*. Two sequences differing at one residue share essentially
    every order-1 and order-2 statistic, so an order-2 Markov model fitted on one
    is not held out with respect to the other in any sense that matters, and this
    ladder is described as "the only tokenizer-independent axis on which the
    protein arms can be compared".

    Exact set disjointness cannot be strengthened into clustering here without
    importing an alignment tool this module has no business owning, so the
    remaining leak is *measured* rather than removed: a shared 30-mer is a fact a
    reader can weigh, and a ladder computed on a train/test split where most test
    sequences have one is a ladder to distrust. One pass and one set, so it costs
    nothing next to the ladder it annotates.
    """

    if k < 2:
        raise ValueError("the near-duplicate substring must be at least two residues")
    seen: set[str] = set()
    for sequence in train_sequences:
        seen.update(sequence[i : i + k] for i in range(len(sequence) - k + 1))
    scorable = [sequence for sequence in test_sequences if len(sequence) >= k]
    matched = sum(
        1
        for sequence in scorable
        if any(sequence[i : i + k] in seen for i in range(len(sequence) - k + 1))
    )
    return {
        "kmer": int(k),
        "n_test_sequences_scorable": len(scorable),
        "n_test_sequences_with_shared_kmer": matched,
        "fraction_test_sequences_with_shared_kmer": (
            matched / len(scorable) if scorable else None
        ),
        "interpretation": (
            "disjointness is enforced by exact string identity only; a shared "
            f"{k}-mer over a twenty-letter alphabet is homology rather than chance, "
            "so this fraction bounds how much of the 'held-out' ladder is fitted on "
            "the test set's own protein families"
        ),
    }


def markov_baselines(
    train_sequences: Sequence[str],
    test_sequences: Sequence[str],
    *,
    orders: Sequence[int] = (0, 1, 2),
    alphabet: str = AA20,
) -> dict[str, Any]:
    """The tokenizer-independent residue ladder for a protein cohort."""

    if not orders:
        raise ValueError("at least one Markov order is required")
    return {
        "alphabet_size": len(alphabet),
        "n_train_sequences": len(train_sequences),
        "n_test_sequences": len(test_sequences),
        "cross_entropy_bits_per_residue": {
            f"order{order}": markov_cross_entropy_bits(
                train_sequences, test_sequences, order=order, alphabet=alphabet
            )
            for order in orders
        },
        "maximum_bits_per_residue": math.log2(len(alphabet)),
        # Travels with the ladder, because a reader who is handed only the
        # cross-entropies cannot tell a held-out corpus from a near-clonal one.
        "held_out_strictness": near_duplicate_fraction(train_sequences, test_sequences),
    }


def _supports_trimmed_logits(arm: Arm) -> bool:
    """Whether this build of ``transformers`` lets the arm trim its logit head.

    The answer is a property of the installed library, not of the measurement:
    transformers 4.57.3 gives ``GPT2LMHeadModel.forward`` an explicit
    ``logits_to_keep`` parameter and 4.52.4 does not, and the ProGen2 remote code
    has never had one. It is read from the signature rather than tried, because
    both signatures also accept ``**kwargs``, so an unsupported build would take
    the argument and silently return the full tensor.

    Trimming is not numerically inert. It moves the unembedding matmul from
    ``(batch, tokens, d)`` to ``(batch, 1, d)``, which selects a different cuBLAS
    kernel; measured on gpt2-large in bfloat16 the last-position logits differ by
    up to 0.25 and the resulting per-token NLL by up to 0.12 nats, with a mean
    shift of order 1e-3 nats. Which path ran is therefore recorded by
    :func:`truncation_curve` rather than left to the host.
    """

    return "logits_to_keep" in inspect.signature(arm.model.forward).parameters


@torch.no_grad()
def truncation_curve(
    arm: Arm,
    input_strings: Sequence[str],
    *,
    max_len: int,
    context_lengths: Sequence[int] = DEFAULT_CONTEXT_LENGTHS,
    queries_per_sequence: int = 6,
    batch_size: int = 64,
    seed: int,
    min_windows: int = 200,
) -> dict[str, Any]:
    """Clean NLL at sampled query positions as a function of visible context.

    A cohort can be starved either because the model is off-distribution or
    because the content is locally predictable and nothing beyond a short window
    matters. The two look identical in a single cross-entropy number and are
    told apart by how the NLL moves as context is added.
    """

    lengths = sorted({int(value) for value in context_lengths})
    if not lengths or lengths[0] < 1:
        raise ValueError("context lengths must be positive integers")
    if queries_per_sequence < 1 or batch_size < 1 or min_windows < 1:
        raise ValueError("queries_per_sequence, batch_size and min_windows must be positive")
    longest = lengths[-1]
    if max_len <= longest + 1:
        raise ValueError("max_len must exceed the longest requested context by at least two")

    if not _supports_trimmed_logits(arm) and int(arm.model.config.vocab_size) > 1024:
        raise RuntimeError(
            f"{arm.name}: vocabulary of {arm.model.config.vocab_size} without "
            "logits_to_keep support; the full logit tensor would dominate memory. "
            f"transformers {transformers.__version__} does not expose "
            "logits_to_keep on this architecture's forward; 4.57.3 does for the "
            "GPT-2 family. Relaxing the guard is not a free port: the untrimmed "
            "path is a different unembedding kernel and would not reproduce a "
            "curve measured on the trimmed one"
        )

    generator = np.random.default_rng(seed)
    windows: list[tuple[list[int], int]] = []
    for text in input_strings:
        ids = arm.tokenizer(text, return_tensors=None)["input_ids"][:max_len]
        first, last = longest, len(ids) - 1
        if last <= first:
            continue
        picks = generator.choice(
            np.arange(first, last), size=min(queries_per_sequence, last - first), replace=False
        )
        windows.extend((ids, int(query)) for query in picks)
    if len(windows) < min_windows:
        raise RuntimeError(
            f"{arm.name}: only {len(windows)} truncation windows survive a "
            f"{longest}-token context requirement; need {min_windows}"
        )

    trimmed = _supports_trimmed_logits(arm)
    curve: dict[int, float] = {}
    for context in lengths:
        batches: list[np.ndarray] = []
        for offset in range(0, len(windows), batch_size):
            chunk = windows[offset : offset + batch_size]
            block = torch.tensor(
                [ids[query - context : query + 1] for ids, query in chunk], dtype=torch.long
            ).to(arm.device)
            extra = {"logits_to_keep": 1} if trimmed else {}
            logits = arm.model(input_ids=block[:, :-1], **extra).logits
            logprobs = F.log_softmax(logits[:, -1].float(), dim=-1)
            nll = -logprobs.gather(-1, block[:, -1:]).squeeze(-1)
            if not bool(torch.isfinite(nll).all()):
                raise FloatingPointError(f"{arm.name}: non-finite truncated NLL")
            batches.append(nll.cpu().numpy().astype(np.float64))
        curve[context] = float(np.concatenate(batches).mean())

    shortest = lengths[0]
    span = curve[shortest] - curve[longest]
    # ``span > 1e-9`` collapsed two different answers into one ``None``. A
    # negative span -- NLL *rising* as context is added -- is a finding: it says
    # the arm is off-distribution on this cohort in a way a single cross-entropy
    # cannot show, and it is one of the two readings this curve exists to
    # separate. A span at zero says the curve is flat and the fraction is
    # genuinely undefined. Reported apart, so a reader does not have to guess
    # which of the two produced the null.
    span_status = (
        "negative_nll_rises_with_context"
        if span < -1e-9
        else "flat_no_context_effect"
        if abs(span) <= 1e-9
        else "positive"
    )
    return {
        "n_windows": len(windows),
        "seed": int(seed),
        # Provenance, not a result: the trimmed and untrimmed unembedding paths
        # are the same quantity through different kernels, and a curve compared
        # across hosts has to say which one produced it.
        "logits_to_keep_used": bool(trimmed),
        "transformers_version": transformers.__version__,
        "context_lengths": lengths,
        "nll_nats_by_context": {str(context): value for context, value in curve.items()},
        "nll_reduction_shortest_to_longest_nats": span,
        "nll_reduction_status": span_status,
        "fraction_of_reduction_beyond_context_8": (
            (curve[8] - curve[longest]) / span if 8 in curve and span > 1e-9 else None
        ),
        "fraction_of_reduction_beyond_context_32": (
            (curve[32] - curve[longest]) / span if 32 in curve and span > 1e-9 else None
        ),
        "fraction_undefined_because": (
            None
            if span > 1e-9
            else (
                "the context span is not positive, so there is no reduction to take "
                "a fraction of; see nll_reduction_status for which case this is"
            )
        ),
    }


def power_status(context_information_nats: float, threshold_nats: float) -> tuple[str, str]:
    """Map a measured power figure onto a verdict and a measurability status.

    A FAIL here is a statement about the cohort, not about the model or the
    interpretability method: below the threshold no downstream ratio computed on
    this arm can be interpreted, so the arm must be excluded rather than
    reported as a negative result.
    """

    if not math.isfinite(context_information_nats):
        raise ValueError("context information must be finite")
    if not math.isfinite(threshold_nats) or threshold_nats <= 0.0:
        raise ValueError("threshold must be finite and positive")
    if context_information_nats >= threshold_nats:
        return "PASS", MEASURABLE
    return "FAIL", UNMEASURABLE


def arm_power(
    arm: Arm,
    cohort: Cohort,
    *,
    max_len: int,
    batch_size: int,
    minimum_context_information_nats: float = MIN_CONTEXT_INFORMATION_NATS,
    held_out_unigram_nats: float | None = None,
) -> dict[str, Any]:
    """The headline power figure for one arm on one frozen cohort.

    ``held_out_unigram_nats`` makes the estimator behind the verdict explicit at
    the call. Supply it and the baseline, the context information, every
    per-symbol view and the verdict all come from the held-out estimator, and
    the record says so. Omit it and everything is computed against the in-cohort
    plug-in exactly as before -- but the ``measurability`` verdict then names the
    estimator in its own value, so it cannot be quoted as a cross-arm one, and
    the plug-in verdict is additionally published under an estimator-qualified
    key that no caller overwrites. See :data:`MEASURABLE_PLUG_IN`.
    """

    inputs = cohort.input_strings(arm)
    scored = scored_tokens(arm, inputs, max_len=max_len, batch_size=batch_size)
    vocab = int(arm.model.config.vocab_size)
    plug_in = unigram_entropy_nats(scored.target_ids, vocab)
    baseline_mm = miller_madow_entropy_nats(scored.target_ids, vocab)
    clean_ce = float(scored.nll_nats.mean())
    held_out = None if held_out_unigram_nats is None else float(held_out_unigram_nats)
    if held_out is not None and not math.isfinite(held_out):
        raise ValueError("the held-out unigram baseline must be finite")
    baseline = plug_in if held_out is None else held_out
    context_information = baseline - clean_ce
    verdict, status = power_status(context_information, minimum_context_information_nats)
    # Always the plug-in verdict, under a name that says so. It is what
    # ``power_verdict`` is when no held-out baseline was supplied, and it stays
    # true when a caller replaces ``power_verdict`` afterwards, so it can never
    # go stale the way an "estimator" label beside a recomputed verdict would.
    plug_in_verdict, plug_in_status = power_status(
        plug_in - clean_ce, minimum_context_information_nats
    )
    if held_out is None:
        status = MEASURABLE_PLUG_IN if status == MEASURABLE else UNMEASURABLE_PLUG_IN

    order = np.argsort(scored.sequence_index, kind="mergesort")
    grouped = np.split(
        scored.nll_nats[order],
        np.unique(scored.sequence_index[order], return_index=True)[1][1:],
    )
    per_sequence_ce = [float(block.mean()) for block in grouped]
    rendered_expansion = symbols_per_token(arm, inputs, max_len)
    # The per-symbol conversion divides a per-*scored-token* quantity, so it has
    # to divide by the expansion of the scored window and not of the rendered
    # string. Both are reported: the rendered figure describes the rendering and
    # several artefacts quote it, the scored figure is the one the bits-per-
    # symbol axis is built on.
    expansion = scored_symbols_per_token(arm, scored)

    return {
        "arm": arm.name,
        "modality": arm.modality,
        "input_format": arm.spec.input_format,
        "tokenisation": arm.spec.tokenisation,
        "vocab_size": vocab,
        "max_len": int(max_len),
        "n_sequences": len(inputs),
        "n_scored_tokens": len(scored),
        "n_distinct_scored_tokens": int(np.unique(scored.target_ids).size),
        "symbols_per_token": expansion,
        "symbols_per_token_rendered_string": rendered_expansion,
        "symbols_per_token_basis": (
            "scored next-token targets only; for an EC-conditioned arm the "
            "rendered-string figure counts prompt tokens that carry no alphabet "
            "symbol and so understates the expansion, which inflates every "
            "bits-per-symbol figure for that arm alone"
        ),
        # The baseline here is the in-cohort plug-in estimator. It is biased
        # downwards, and the bias grows with vocabulary against sample size: on a
        # cohort of this size roughly +0.003 nats for a 32-symbol arm against
        # +1.65 for a 50257-piece one. That is conservative for the measurability
        # gate, which is what this function exists to decide -- an understated
        # denominator can only exclude an arm, never admit one -- but it is
        # differential across the panel and therefore not a cross-arm quantity.
        # ``src.transfer.prediction_addressed.cohort_power_held_out`` is the
        # held-out estimator a cross-arm number must use. Named in the record so
        # a reader does not have to know which function produced it.
        "unigram_estimator": (
            "cohort_plug_in" if held_out is None else "held_out_disjoint"
        ),
        "unigram_estimator_bias": (
            "plug-in on the scored targets; biased downwards by an amount that "
            "grows with vocabulary size, so context information is understated "
            "and understated unequally across the panel"
            if held_out is None
            else "held-out cross-entropy supplied by the caller"
        ),
        "cross_arm_comparable": held_out is not None,
        "unigram_entropy_on_cohort_nats": plug_in,
        "unigram_entropy_used_for_verdict_nats": baseline,
        "unigram_entropy_miller_madow_nats": baseline_mm,
        "clean_ce_nats": clean_ce,
        "context_information_nats": context_information,
        "context_information_miller_madow_nats": baseline_mm - clean_ce,
        "context_information_bits_per_symbol": context_information / LN2 / expansion,
        "clean_ce_bits_per_symbol": clean_ce / LN2 / expansion,
        "unigram_entropy_bits_per_symbol": baseline / LN2 / expansion,
        # ``clean_ce_nats`` above is token-weighted: a 2000-residue sequence
        # contributes thirty times what a 64-residue one does, which is the
        # convention ``scoring.per_sequence_scores`` states and defends. This
        # interval is over an unweighted mean of per-sequence cross-entropies.
        # On a 64-2000 residue cohort those are not the same estimand, so the
        # interval is not an interval *for* the number printed beside it, and
        # ``mean_interval``'s own ``mean`` key is the only place the difference
        # was visible. The sequence-weighted point estimate is now published so
        # that the pair reads as a point estimate and its interval.
        "clean_ce_nats_sequence_weighted": float(np.mean(per_sequence_ce)),
        "clean_ce_weighting": "token",
        "per_sequence_clean_ce_interval": {
            **mean_interval(per_sequence_ce),
            "estimand": "unweighted mean over sequences of the per-sequence clean CE",
            "is_an_interval_for_clean_ce_nats": False,
        },
        # The same shift applied to every endpoint: this interval is the one
        # above translated by a constant baseline, so it carries the spread of
        # the cross-entropy and none of the uncertainty in the baseline it is
        # measured against, despite the baseline being an estimate too.
        "per_sequence_context_information_interval": {
            **mean_interval([baseline - value for value in per_sequence_ce]),
            "estimand": (
                "unweighted mean over sequences of (fixed baseline - per-sequence "
                "clean CE)"
            ),
            "is_an_interval_for_context_information_nats": False,
            "baseline_uncertainty_included": False,
        },
        "minimum_context_information_nats": float(minimum_context_information_nats),
        "power_verdict": verdict,
        "measurability": status,
        # Estimator-qualified and never overwritten, so an artefact always holds
        # one verdict whose provenance is unambiguous even after a caller
        # recomputes ``power_verdict`` from a better baseline.
        "power_verdict_plug_in": plug_in_verdict,
        "measurability_plug_in": plug_in_status,
    }
