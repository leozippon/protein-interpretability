"""Prediction-addressed attention (PAA) / copy-suppression census.

A head is *prediction-addressed* at a scored position ``q`` if its attention
concentrates on earlier occurrences of the token the model is about to predict,
beyond what position alone explains, and *suppressive* if removing that
attention raises the predicted token's margin.  The query source is what
separates this from induction: induction addresses by the identity of the
*current* token, PAA by the identity of the *predicted* token.

Everything here is a gate instrument.  It exists to answer whether a
sparse, causally confirmable suppressive head population exists in the text
control at all, and whether the protein arms could ever supply matched
instances.  A negative answer is a complete answer and is reported as one.

Three design commitments the rest of this module is built around.

**Decoy correction, not raw attention.**  ``A_h[q, k*]`` alone would rank
positional heads: a head that attends uniformly over the last sixty-four tokens
scores highly on any antecedent that happens to sit there.  Every score in this
module is ``A_h[q, k*]`` minus the mean over four decoy keys drawn from the same
distance bin, so the positional component cancels within the instance.

**Hard exclusion of the induction target set.**  In a repeat region the
induction target and the PAA antecedent coincide, and a census that admitted
those instances would share scored positions with the induction census it is
supposed to be independent of.  Instances whose nearest antecedent ``k*``
satisfies ``t_{k*-1} == t_q`` are discarded, and the discard rate is reported.

**Renormalising knockout.**  The causal statistic removes attention *before* the
softmax, by adding a large negative number to the pre-softmax scores of the
antecedent keys for one head at one layer.  The remaining keys therefore
renormalise, which is the intervention "this head did not read the antecedent"
rather than "this head read a shorter row".  A multiplicative post-softmax head
mask is available in the same models and would not renormalise; it is
deliberately not used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.transfer.arms import (
    Arm,
    Cohort,
    conditioning_boundary_ids,
    symbols_per_token,
)
from src.transfer.budget import scored_tokens
from src.transfer.circuits import RepeatProbe, content_bounds, n_head
from src.transfer.pathways import assert_disjoint, disjoint_unigram_cross_entropy_nats
from src.transfer.scoring import target_rule

#: Distance bins, in tokens, used both to draw decoys and to coarsen the
#: matching covariate.  Doubling bins because dependency distance is read on a
#: log scale everywhere else in this programme and a linear bin would put a
#: distance-3 and a distance-300 antecedent in comparable cells.
DISTANCE_BIN_EDGES: tuple[tuple[int, int], ...] = (
    (2, 4),
    (5, 8),
    (9, 16),
    (17, 32),
    (33, 64),
    (65, 128),
    (129, 256),
    (257, 1024),
)

#: Absolute bins on the corruption effect, expressed as the change in
#: ``log p(X)`` when the antecedent token is replaced.  Absolute rather than
#: within-arm quantile bins: quantile bins would make any two arms match by
#: construction, which is the opposite of what a matching feasibility gate is
#: for.
CORRUPTION_BIN_EDGES: tuple[float, ...] = (-math.inf, -2.0, -1.0, -0.5, -0.1, 0.1, math.inf)

#: Confidence bins on ``p(X | t_<=q)``.
CONFIDENCE_BIN_EDGES: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0 + 1e-9)

#: Pre-softmax score added to a knocked-out key.  Large enough to zero the
#: softmax weight in bfloat16, small enough not to produce ``-inf - inf``.
KNOCKOUT_LOGIT = -1.0e30

#: Largest logit movement an *all-zero* knockout mask may produce before the
#: causal statistic is refused, in logits.
#:
#: The knockout works by adding to the additive attention mask the attention
#: module already receives.  If a build passes no such mask -- or passes it
#: somewhere this hook does not see -- the injected tensor becomes the mask
#: rather than an addition to it, and the model's own causal masking is replaced
#: by a matrix of zeros.  Every subsequent number would then be a difference
#: between two non-causal forward passes: entirely well-formed, entirely wrong.
#: The zero-mask pass is the control for that, and a control that is measured and
#: reported but never compared against anything is not a control.  1e-2 is loose
#: against bfloat16 logits of magnitude ~20 and tight against the ~10-logit
#: movement dropping causal masking produces.
ZERO_MASK_TOLERANCE_LOGITS = 1.0e-2


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def distance_bin(distance: int) -> int:
    """Index of the doubling bin holding ``distance``; raises outside the range."""

    for index, (low, high) in enumerate(DISTANCE_BIN_EDGES):
        if low <= distance <= high:
            return index
    raise ValueError(f"distance {distance} lies outside the declared bins")


def _bin_index(value: float, edges: Sequence[float]) -> int:
    for index in range(len(edges) - 1):
        if edges[index] <= value < edges[index + 1]:
            return index
    raise ValueError(f"value {value} lies outside the declared bin edges {tuple(edges)}")


# ------------------------------------------------------------------- Gate 0


def scored_target_counts(arm: Arm, strings: Sequence[str], *, max_len: int) -> np.ndarray:
    """Next-token-target counts over exactly the multiset ``scored_tokens`` scores.

    Applies :func:`src.transfer.scoring.target_rule` without a forward pass, so
    that a held-out reference corpus and the scored cohort are counted over the
    same kind of token. Counting the reference over a different span --
    including ZymCTRL's EC tag, say -- would fit the context-free baseline on a
    distribution the model is never scored against.

    The span arithmetic is open-coded rather than routed through
    :func:`src.transfer.scoring.sequence_target_mask` because this path has no
    attention mask and no batch: it walks one untruncated id list at a time. The
    *rule* and the *boundary ids* still come from the shared declarations, which
    is where the two used to be able to drift apart.
    """

    if max_len < 2:
        raise ValueError("max_len must admit at least one next-token target")
    vocab = int(arm.model.config.vocab_size)
    counts = np.zeros(vocab, dtype=np.int64)
    conditioned = target_rule(arm.spec.input_format) == "between_boundaries"
    start_id, end_id = conditioning_boundary_ids(arm)
    for text in strings:
        ids = arm.tokenizer(text, return_tensors=None)["input_ids"][:max_len]
        if len(ids) < 2:
            continue
        if conditioned:
            if ids.count(start_id) != 1 or ids.count(end_id) != 1:
                raise ValueError(f"{arm.name}: row lacks exactly one <start>/<end> pair")
            targets = ids[ids.index(start_id) + 1 : ids.index(end_id)]
        else:
            targets = ids[1:]
        if not targets:
            continue
        array = np.asarray(targets, dtype=np.int64)
        if array.min() < 0 or array.max() >= vocab:
            raise ValueError(f"{arm.name}: token id outside the declared vocabulary")
        counts += np.bincount(array, minlength=vocab)
    if counts.sum() < 1:
        raise RuntimeError(f"{arm.name}: reference corpus yields no scored targets")
    return counts


def cohort_power_held_out(
    arm: Arm,
    cohort: Cohort,
    reference: Cohort,
    *,
    max_len: int,
    batch_size: int,
    threshold_nats: float,
) -> dict[str, Any]:
    """Context-derived information with a held-out unigram baseline.

    The plug-in estimator this replaces is biased downwards by up to 1.02 nats
    on a 50k-vocabulary arm and by 0.01 on a residue-level one, so it inflates
    exactly the arms whose relative position a modality reading depends on.
    """

    if threshold_nats <= 0:
        raise ValueError("the power threshold must be positive")
    assert_disjoint(cohort, reference)
    inputs = cohort.input_strings(arm)
    scored = scored_tokens(arm, inputs, max_len=max_len, batch_size=batch_size)
    vocab = int(arm.model.config.vocab_size)
    target_counts = np.bincount(scored.target_ids, minlength=vocab).astype(np.int64)
    reference_counts = scored_target_counts(arm, reference.input_strings(arm), max_len=max_len)
    held_out = disjoint_unigram_cross_entropy_nats(reference_counts, target_counts)
    plug_in_probabilities = target_counts[target_counts > 0] / target_counts.sum()
    plug_in = float(-(plug_in_probabilities * np.log(plug_in_probabilities)).sum())
    clean_ce = float(scored.nll_nats.mean())

    order = np.argsort(scored.sequence_index, kind="mergesort")
    boundaries = np.unique(scored.sequence_index[order], return_index=True)[1][1:]
    per_sequence_ce = [float(block.mean()) for block in np.split(scored.nll_nats[order], boundaries)]

    context = held_out - clean_ce
    return {
        "arm": arm.name,
        "modality": arm.modality,
        "input_format": arm.spec.input_format,
        "tokenisation": arm.spec.tokenisation,
        "vocab_size": vocab,
        "cohort": cohort.name,
        "cohort_digest": cohort.digest,
        "reference_digest": reference.digest,
        "reference_sequences": len(reference),
        "n_sequences": len(inputs),
        "n_scored_tokens": len(scored),
        "max_len": int(max_len),
        "symbols_per_token": _finite(symbols_per_token(arm, inputs, max_len), "expansion"),
        "unigram_held_out_nats": _finite(held_out, "held-out unigram"),
        "unigram_plug_in_nats": _finite(plug_in, "plug-in unigram"),
        "plug_in_bias_nats": _finite(held_out - plug_in, "plug-in bias"),
        "clean_ce_nats": _finite(clean_ce, "clean CE"),
        "context_information_nats": _finite(context, "context information"),
        "per_sequence_context_information_sd": _finite(
            float(np.std([held_out - value for value in per_sequence_ce], ddof=1)),
            "per-sequence context information sd",
        ),
        "threshold_nats": float(threshold_nats),
        "verdict": "PASS" if context >= threshold_nats else "FAIL",
    }


# ------------------------------------------------------- instance construction


def tokenised_rows(
    arm: Arm, strings: Sequence[str], *, width: int
) -> tuple[list[list[int]], int]:
    """Equal-length token rows and the first index that carries modality content.

    Rows shorter than ``width`` are dropped rather than padded.  A padded row
    would change which keys the softmax normalises over, and every statistic
    below is an attention weight; the scaffolding prefix is skipped because a
    protein arm's control token is not a token whose repetition means anything.
    """

    if width < 8:
        raise ValueError("rows must be long enough to carry a dependency")
    rows: list[list[int]] = []
    lows: set[int] = set()
    for text in strings:
        ids = arm.tokenizer(text, return_tensors=None)["input_ids"]
        if len(ids) < width:
            continue
        row = [int(value) for value in ids[:width]]
        low, _ = content_bounds(arm, row, width)
        lows.add(int(low))
        rows.append(row)
    if not rows:
        raise RuntimeError(f"{arm.name}: no cohort record reached {width} tokens")
    if len(lows) != 1:
        raise ValueError(f"{arm.name}: inconsistent content offsets {sorted(lows)}")
    return rows, lows.pop()


@dataclass(frozen=True)
class InstancePool:
    """Candidate PAA instances over one cohort, with their matching covariates.

    Arrays are parallel and one entry long per instance.  ``sequence`` indexes
    the cohort, which is the cluster for every bootstrap in this module: two
    instances from one document are not independent draws.
    """

    arm: str
    sequence: np.ndarray
    query: np.ndarray
    antecedent: np.ndarray
    predicted_token: np.ndarray
    confidence: np.ndarray
    distance: np.ndarray
    unigram_percentile: np.ndarray
    decoys: np.ndarray
    clean_logit_target: np.ndarray
    clean_logit_runner_up: np.ndarray
    cascade: dict[str, int]

    def __post_init__(self) -> None:
        n = self.sequence.size
        for name in (
            "query",
            "antecedent",
            "predicted_token",
            "confidence",
            "distance",
            "unigram_percentile",
            "clean_logit_target",
            "clean_logit_runner_up",
        ):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"instance field {name} does not align with the pool")
        if self.decoys.ndim != 2 or self.decoys.shape[0] != n:
            raise ValueError("decoy array does not align with the pool")
        if n < 1:
            raise RuntimeError(f"{self.arm}: no PAA instances survived construction")

    def __len__(self) -> int:
        return int(self.sequence.size)

    @property
    def m_gap(self) -> np.ndarray:
        return self.clean_logit_target - self.clean_logit_runner_up


def unigram_percentiles(counts: np.ndarray) -> np.ndarray:
    """Cumulative unigram mass at or below each token's own probability.

    A rank would not be comparable between a 50,257-piece vocabulary and a
    32-symbol one.  Cumulative mass is: it answers "what share of the stream is
    at least this rare", which means the same thing on both sides.
    """

    array = np.asarray(counts, dtype=np.float64)
    if array.ndim != 1 or array.sum() <= 0:
        raise ValueError("unigram counts must be a non-empty vector")
    probabilities = array / array.sum()
    order = np.argsort(probabilities, kind="mergesort")
    cumulative = np.cumsum(probabilities[order])
    percentile = np.empty_like(probabilities)
    percentile[order] = cumulative
    return percentile


@torch.no_grad()
def _batched_top_predictions(
    arm: Arm,
    ids: torch.Tensor,
    mask: torch.Tensor,
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    logits = arm.model(input_ids=ids, attention_mask=mask).logits.float()
    top = logits.topk(top_k, dim=-1)
    probabilities = F.softmax(logits, dim=-1)
    top_probabilities = probabilities.gather(-1, top.indices)
    return (
        top.indices.cpu().numpy().astype(np.int64),
        top_probabilities.cpu().numpy().astype(np.float64),
        logits,
    )


def build_instance_pool(
    arm: Arm,
    rows: Sequence[Sequence[int]],
    *,
    unigram_counts: np.ndarray,
    query_min: int,
    top_k: int,
    candidate_depth: int,
    min_confidence: float,
    n_decoys: int,
    seed: int,
    batch_size: int,
    ban_depth: int | None = None,
    max_per_sequence: int | None = None,
) -> InstancePool:
    """Select ``(q, X, k*)`` instances and their decoys over pre-tokenised rows.

    ``rows`` are equal-length token rows, one per cohort sequence, already
    truncated and free of padding.  Requiring equal lengths is not a
    convenience: a padded row changes which keys a softmax normalises over, and
    every score below is an attention weight.

    ``ban_depth`` is how many of the model's own top candidates at ``q`` a decoy
    token may not be.  It defaults to ``top_k`` because that is what the design
    specifies, and it is exposed because that specification is unsatisfiable on
    a residue-level arm: banning twenty candidates over a twenty-symbol alphabet
    bans the alphabet, and every decoy pool comes back empty.  Making the depth
    a declared parameter is what turns that into a measured obstacle rather than
    an empty result table.
    """

    if not rows:
        raise ValueError(f"{arm.name}: no rows supplied")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{arm.name}: instance construction requires equal-length rows")
    if query_min < 2 or query_min >= width:
        raise ValueError("query_min must leave a context and a scored suffix")
    if candidate_depth < 1 or candidate_depth > top_k:
        raise ValueError("candidate_depth must lie within the recorded top-k")
    if n_decoys < 1:
        raise ValueError("at least one decoy is required")
    ban = top_k if ban_depth is None else ban_depth
    if ban < 1 or ban > top_k:
        raise ValueError("ban_depth must lie within the recorded top-k")

    percentile = unigram_percentiles(unigram_counts)
    rng = np.random.default_rng(seed)
    pad = arm.tokenizer.pad_token_id
    if pad is None:
        raise ValueError(f"{arm.name}: tokenizer has no pad token")

    sequences: list[int] = []
    queries: list[int] = []
    antecedents: list[int] = []
    predicted: list[int] = []
    confidences: list[float] = []
    distances: list[int] = []
    percentiles: list[float] = []
    decoy_rows: list[np.ndarray] = []
    target_logits: list[float] = []
    runner_up_logits: list[float] = []

    cascade = {
        "positions_scored": 0,
        "positions_with_eligible_candidate": 0,
        "candidates_discarded_by_induction_target": 0,
        "candidates_discarded_by_distance_range": 0,
        "candidates_discarded_by_empty_decoy_pool": 0,
        "instances_retained": 0,
        "decoys_drawn_with_replacement": 0,
        "ban_depth": int(ban),
    }

    for begin in range(0, len(rows), batch_size):
        chunk = rows[begin : begin + batch_size]
        ids = torch.tensor(np.asarray(chunk, dtype=np.int64), dtype=torch.long, device=arm.device)
        mask = torch.ones_like(ids)
        top_ids, top_probabilities, logits = _batched_top_predictions(
            arm, ids, mask, top_k=top_k
        )
        for row_index, row in enumerate(chunk):
            sequence_index = begin + row_index
            tokens = np.asarray(row, dtype=np.int64)
            kept_here = 0
            for q in range(query_min, width):
                cascade["positions_scored"] += 1
                if max_per_sequence is not None and kept_here >= max_per_sequence:
                    continue
                banned = set(int(value) for value in top_ids[row_index, q, :ban])
                chosen: tuple[int, int, float] | None = None
                induction_blocked = False
                distance_blocked = False
                for depth in range(candidate_depth):
                    token = int(top_ids[row_index, q, depth])
                    probability = float(top_probabilities[row_index, q, depth])
                    if probability < min_confidence:
                        break
                    matches = np.flatnonzero(tokens[:q] == token)
                    if matches.size == 0:
                        continue
                    star = int(matches[-1])
                    if star >= 1 and int(tokens[star - 1]) == int(tokens[q]):
                        induction_blocked = True
                        continue
                    span = q - star
                    if span < DISTANCE_BIN_EDGES[0][0] or span > DISTANCE_BIN_EDGES[-1][1]:
                        distance_blocked = True
                        continue
                    chosen = (token, star, probability)
                    break
                if chosen is None:
                    if induction_blocked:
                        cascade["candidates_discarded_by_induction_target"] += 1
                    elif distance_blocked:
                        cascade["candidates_discarded_by_distance_range"] += 1
                    continue
                cascade["positions_with_eligible_candidate"] += 1
                token, star, probability = chosen
                span = q - star
                low, high = DISTANCE_BIN_EDGES[distance_bin(span)]
                lowest = max(0, q - high)
                highest = q - low
                if highest < lowest:
                    cascade["candidates_discarded_by_empty_decoy_pool"] += 1
                    continue
                window = np.arange(lowest, highest + 1)
                predecessor_ok = np.ones(window.size, dtype=bool)
                nonzero = window >= 1
                predecessor_ok[nonzero] = tokens[window[nonzero] - 1] != int(tokens[q])
                eligible = window[
                    (window != star)
                    & (tokens[window] != token)
                    & (~np.isin(tokens[window], list(banned)))
                    & predecessor_ok
                ]
                if eligible.size < 1:
                    cascade["candidates_discarded_by_empty_decoy_pool"] += 1
                    continue
                if eligible.size >= n_decoys:
                    draw = rng.choice(eligible, size=n_decoys, replace=False)
                else:
                    draw = rng.choice(eligible, size=n_decoys, replace=True)
                    cascade["decoys_drawn_with_replacement"] += 1
                row_logits = logits[row_index, q]
                target_logit = float(row_logits[token])
                masked = row_logits.clone()
                masked[token] = float("-inf")
                runner_up = float(masked.max())

                sequences.append(sequence_index)
                queries.append(q)
                antecedents.append(star)
                predicted.append(token)
                confidences.append(probability)
                distances.append(span)
                percentiles.append(float(percentile[token]))
                decoy_rows.append(np.sort(draw.astype(np.int64)))
                target_logits.append(target_logit)
                runner_up_logits.append(runner_up)
                cascade["instances_retained"] += 1
                kept_here += 1
        del logits
        torch.cuda.empty_cache()

    return InstancePool(
        arm=arm.name,
        sequence=np.asarray(sequences, dtype=np.int64),
        query=np.asarray(queries, dtype=np.int64),
        antecedent=np.asarray(antecedents, dtype=np.int64),
        predicted_token=np.asarray(predicted, dtype=np.int64),
        confidence=np.asarray(confidences, dtype=np.float64),
        distance=np.asarray(distances, dtype=np.int64),
        unigram_percentile=np.asarray(percentiles, dtype=np.float64),
        decoys=np.stack(decoy_rows) if decoy_rows else np.zeros((0, n_decoys), dtype=np.int64),
        clean_logit_target=np.asarray(target_logits, dtype=np.float64),
        clean_logit_runner_up=np.asarray(runner_up_logits, dtype=np.float64),
        cascade=cascade,
    )


# --------------------------------------------------------------- attention taps


def attention_module(arm: Arm, layer: int) -> torch.nn.Module:
    """The attention submodule whose forward returns ``(output, weights)``."""

    return arm.attention(layer)


class _WeightTap:
    """Forward hook returning the eager attention pattern of one layer.

    ``output_attentions=True`` would materialise every layer's pattern at once;
    at 36 layers, 20 heads and 512 tokens that is 377 MiB per sequence held
    simultaneously.  Tapping layer by layer and reducing inside the hook keeps
    the peak at one layer.
    """

    def __init__(self, consume: Callable[[int, torch.Tensor], None], layer: int) -> None:
        self.consume = consume
        self.layer = layer

    def __call__(self, module, args, output):
        if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
            raise RuntimeError(
                "attention module returned no weights; load the arm with "
                "attn_implementation='eager'"
            )
        self.consume(self.layer, output[1])
        return None


@torch.no_grad()
def paa_attention_scores(
    arm: Arm,
    rows: Sequence[Sequence[int]],
    pool: InstancePool,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Per-sequence, per-head decoy-corrected attention onto the antecedent.

    Returns per-sequence matrices rather than pooled means.  A pooled mean
    cannot be cluster-bootstrapped afterwards, and a census that emits only
    pooled means is not re-analysable, which is a failure this programme has
    already recorded once.
    """

    arm.require("circuits")
    arm.require_eager_attention("the prediction-addressed attention census")
    heads = n_head(arm)
    layers = arm.n_layer
    n_sequences = len(rows)
    device = arm.device

    antecedent_sum = torch.zeros((n_sequences, layers, heads), dtype=torch.float64, device=device)
    decoy_sum = torch.zeros_like(antecedent_sum)
    sink_sum = torch.zeros_like(antecedent_sum)
    counts = np.zeros(n_sequences, dtype=np.int64)

    state: dict[str, Any] = {}

    def consume(layer: int, weights: torch.Tensor) -> None:
        rows_index = state["rows"]
        queries = state["queries"]
        keys = state["keys"]
        decoys = state["decoys"]
        sequence_index = state["sequences"]
        pattern = weights.float()
        antecedent = pattern[rows_index, :, queries, keys]
        n, n_decoy = decoys.shape
        flat_rows = rows_index.unsqueeze(1).expand(n, n_decoy).reshape(-1)
        flat_queries = queries.unsqueeze(1).expand(n, n_decoy).reshape(-1)
        decoy_values = pattern[flat_rows, :, flat_queries, decoys.reshape(-1)]
        decoy_mean = decoy_values.view(n, n_decoy, heads).mean(dim=1)
        sink = pattern[rows_index, :, queries, 0]
        antecedent_sum[:, layer].index_add_(0, sequence_index, antecedent.double())
        decoy_sum[:, layer].index_add_(0, sequence_index, decoy_mean.double())
        sink_sum[:, layer].index_add_(0, sequence_index, sink.double())

    handles = [
        attention_module(arm, layer).register_forward_hook(_WeightTap(consume, layer))
        for layer in range(layers)
    ]
    try:
        for begin in range(0, n_sequences, batch_size):
            end = min(begin + batch_size, n_sequences)
            selected = np.flatnonzero((pool.sequence >= begin) & (pool.sequence < end))
            if selected.size == 0:
                continue
            chunk = rows[begin:end]
            ids = torch.tensor(
                np.asarray(chunk, dtype=np.int64), dtype=torch.long, device=device
            )
            mask = torch.ones_like(ids)
            state["rows"] = torch.tensor(
                pool.sequence[selected] - begin, dtype=torch.long, device=device
            )
            state["sequences"] = torch.tensor(
                pool.sequence[selected], dtype=torch.long, device=device
            )
            state["queries"] = torch.tensor(pool.query[selected], dtype=torch.long, device=device)
            state["keys"] = torch.tensor(
                pool.antecedent[selected], dtype=torch.long, device=device
            )
            state["decoys"] = torch.tensor(
                pool.decoys[selected], dtype=torch.long, device=device
            )
            arm.model(input_ids=ids, attention_mask=mask, use_cache=False)
            np.add.at(counts, pool.sequence[selected], 1)
    finally:
        for handle in handles:
            handle.remove()

    active = counts > 0
    if not active.any():
        raise RuntimeError(f"{arm.name}: no sequence contributed a PAA instance")
    divisor = torch.tensor(
        np.where(active, counts, 1).astype(np.float64), device=device
    ).view(-1, 1, 1)
    antecedent_mean = (antecedent_sum / divisor).cpu().numpy()
    decoy_mean = (decoy_sum / divisor).cpu().numpy()
    sink_mean = (sink_sum / divisor).cpu().numpy()
    # A sequence that contributed no instance has an accumulated sum of exactly
    # zero, and dividing by the placeholder 1 leaves it there. Zero is a legal
    # decoy-corrected attention and 1 - 0 is a legal non-sink mass, so an
    # unweighted mean over all rows -- which is what a reader who missed
    # ``active_sequences`` would take -- would silently pull ``paa_specific``
    # toward zero and ``non_sink_mass`` toward one, in proportion to how many
    # sequences supplied nothing. NaN there makes that mistake raise.
    # ``cluster_bootstrap`` is unaffected: it drops zero-weight clusters before
    # touching the values.
    inactive = ~active
    for block in (antecedent_mean, decoy_mean, sink_mean):
        block[inactive] = np.nan
    return {
        "instances_per_sequence": counts,
        "active_sequences": active,
        "antecedent_attention": antecedent_mean,
        "decoy_attention": decoy_mean,
        "paa_specific": antecedent_mean - decoy_mean,
        "non_sink_mass": 1.0 - sink_mean,
    }


# ---------------------------------------------------------- knockout machinery


class _AntecedentKnockout:
    """Pre-softmax removal of chosen keys for one head at one layer.

    Implemented by adding to the additive attention mask the module already
    receives, so the softmax renormalises over the surviving keys.  The mask is
    per-head, which the eager kernel supports because it broadcasts a
    ``(batch, 1, q, k)`` mask against ``(batch, head, q, k)`` scores and
    therefore also accepts the full per-head shape.
    """

    def __init__(self, head: int, mask: torch.Tensor) -> None:
        if mask.ndim != 4:
            raise ValueError("knockout mask must be (batch, head, query, key)")
        self.head = head
        self.mask = mask

    def __call__(self, module, args, kwargs):
        existing = kwargs.get("attention_mask")
        if existing is None:
            kwargs["attention_mask"] = self.mask
        else:
            kwargs["attention_mask"] = existing + self.mask
        return args, kwargs


def build_knockout_mask(
    *,
    batch: int,
    heads: int,
    width: int,
    head: int,
    query_positions: torch.Tensor,
    key_sets: Sequence[Sequence[int]],
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive mask removing ``key_sets[b]`` from row ``query_positions[b]``."""

    if len(key_sets) != batch or query_positions.shape != (batch,):
        raise ValueError("knockout specification does not align with the batch")
    mask = torch.zeros((batch, heads, width, width), device=device, dtype=dtype)
    for row, keys in enumerate(key_sets):
        if not keys:
            raise ValueError(f"row {row} has an empty antecedent set")
        index = torch.tensor(list(keys), dtype=torch.long, device=device)
        mask[row, head, int(query_positions[row]), index] = KNOCKOUT_LOGIT
    return mask


def antecedent_sets(
    rows: Sequence[Sequence[int]], pool: InstancePool, selected: np.ndarray
) -> list[list[int]]:
    """All earlier occurrences of ``X`` for each selected instance."""

    sets: list[list[int]] = []
    for index in selected:
        tokens = np.asarray(rows[int(pool.sequence[index])], dtype=np.int64)
        q = int(pool.query[index])
        token = int(pool.predicted_token[index])
        positions = np.flatnonzero(tokens[:q] == token)
        if positions.size == 0:
            raise RuntimeError("instance has no antecedent; pool construction is inconsistent")
        sets.append([int(value) for value in positions])
    return sets


@torch.no_grad()
def knockout_effects(
    arm: Arm,
    rows: Sequence[Sequence[int]],
    pool: InstancePool,
    selected: np.ndarray,
    heads: Sequence[tuple[int, int]],
    *,
    batch_size: int,
    zero_mask_tolerance: float = ZERO_MASK_TOLERANCE_LOGITS,
) -> dict[str, np.ndarray]:
    """Change in M-gap and in ``p(X)`` when a head stops reading the antecedents.

    A positive M-gap change means the head was *suppressing* ``X``: removing its
    read of the antecedent raised the margin between ``X`` and its strongest
    competitor.

    An all-zero mask is injected first and its effect on the logits is required
    to be below ``zero_mask_tolerance``. That pass establishes that the injection
    path is additive and inert when it carries nothing; without it, a build whose
    attention module receives no additive mask would have its causal masking
    *replaced* by the injected tensor, and every knockout effect would be a
    difference between two non-causal forward passes.
    """

    arm.require("circuits")
    arm.require_eager_attention("the antecedent-knockout causal statistic")
    if zero_mask_tolerance <= 0:
        raise ValueError("the zero-mask tolerance must be positive")
    if selected.size == 0:
        raise ValueError("no instances selected for the causal statistic")
    width = len(rows[0])
    n_heads = n_head(arm)
    dtype = next(arm.model.parameters()).dtype
    delta_gap = np.zeros((len(heads), selected.size), dtype=np.float64)
    delta_probability = np.zeros_like(delta_gap)
    antecedent_mass = np.zeros_like(delta_gap)
    exactness = 0.0
    layers_needed = sorted({layer for layer, _ in heads})

    for begin in range(0, selected.size, batch_size):
        block = selected[begin : begin + batch_size]
        sequence_ids = pool.sequence[block]
        ids = torch.tensor(
            np.asarray([rows[int(index)] for index in sequence_ids], dtype=np.int64),
            dtype=torch.long,
            device=arm.device,
        )
        mask = torch.ones_like(ids)
        queries = torch.tensor(pool.query[block], dtype=torch.long, device=arm.device)
        targets = torch.tensor(
            pool.predicted_token[block], dtype=torch.long, device=arm.device
        )
        keys = antecedent_sets(rows, pool, block)
        rows_index = torch.arange(ids.shape[0], device=arm.device)
        key_mask = torch.zeros((ids.shape[0], width), dtype=torch.bool, device=arm.device)
        for row, keys_row in enumerate(keys):
            key_mask[row, torch.tensor(keys_row, dtype=torch.long, device=arm.device)] = True

        # The pooled causal mean cannot distinguish "no suppressive head" from
        # "a head that suppresses hard on a sparse subset of instances".  The
        # per-instance antecedent mass captured here is what lets the second be
        # read off the same run.
        captured: dict[int, torch.Tensor] = {}

        def consume(layer: int, weights: torch.Tensor, _store=captured, _mask=key_mask) -> None:
            row_attention = weights.float()[torch.arange(weights.shape[0], device=weights.device), :, queries]
            _store[layer] = (row_attention * _mask.unsqueeze(1)).sum(dim=-1)

        taps = [
            attention_module(arm, layer).register_forward_hook(_WeightTap(consume, layer))
            for layer in layers_needed
        ]
        try:
            clean_logits = arm.model(input_ids=ids, attention_mask=mask).logits[
                rows_index, queries
            ].float()
        finally:
            for tap in taps:
                tap.remove()
        for position, (layer, head) in enumerate(heads):
            antecedent_mass[position, begin : begin + block.size] = (
                captured[layer][:, head].cpu().numpy()
            )
        captured.clear()
        clean_gap, clean_p = _margin_and_probability(clean_logits, targets)

        zero_mask = torch.zeros(
            (ids.shape[0], n_heads, width, width), device=arm.device, dtype=dtype
        )
        handle = attention_module(arm, 0).register_forward_pre_hook(
            _AntecedentKnockout(0, zero_mask), with_kwargs=True
        )
        try:
            null_logits = arm.model(input_ids=ids, attention_mask=mask).logits[
                rows_index, queries
            ].float()
        finally:
            handle.remove()
        exactness = max(exactness, float((null_logits - clean_logits).abs().max()))
        if exactness > zero_mask_tolerance:
            raise RuntimeError(
                f"{arm.name}: injecting an all-zero attention mask moved the logits by "
                f"{exactness:.4g}, above the {zero_mask_tolerance:.4g} tolerance. The "
                "knockout adds to an existing additive mask; if none is present the "
                "injection replaces the model's causal mask instead, and every "
                "knockout effect below would be a difference between two non-causal "
                "forward passes"
            )
        del zero_mask, null_logits

        for position, (layer, head) in enumerate(heads):
            knock = build_knockout_mask(
                batch=ids.shape[0],
                heads=n_heads,
                width=width,
                head=head,
                query_positions=queries,
                key_sets=keys,
                device=arm.device,
                dtype=dtype,
            )
            handle = attention_module(arm, layer).register_forward_pre_hook(
                _AntecedentKnockout(head, knock), with_kwargs=True
            )
            try:
                logits = arm.model(input_ids=ids, attention_mask=mask).logits[
                    rows_index, queries
                ].float()
            finally:
                handle.remove()
            gap, probability = _margin_and_probability(logits, targets)
            delta_gap[position, begin : begin + block.size] = (
                (gap - clean_gap).cpu().numpy()
            )
            delta_probability[position, begin : begin + block.size] = (
                (probability - clean_p).cpu().numpy()
            )
            del knock, logits
        torch.cuda.empty_cache()

    return {
        "delta_m_gap": delta_gap,
        "delta_probability": delta_probability,
        "antecedent_attention_mass": antecedent_mass,
        "zero_mask_max_logit_difference": exactness,
        "zero_mask_tolerance_logits": float(zero_mask_tolerance),
    }


def _margin_and_probability(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(logits.shape[0], device=logits.device)
    target_logit = logits[rows, targets]
    masked = logits.clone()
    masked[rows, targets] = float("-inf")
    runner_up = masked.max(dim=-1).values
    probability = F.softmax(logits, dim=-1)[rows, targets]
    return target_logit - runner_up, probability


# ------------------------------------------------------ query-source intervention


class _ResidualNudge:
    """Adds a fixed vector at one position of one layer's block input."""

    def __init__(self, positions: torch.Tensor, delta: torch.Tensor) -> None:
        if positions.ndim != 1 or delta.ndim != 2 or delta.shape[0] != positions.shape[0]:
            raise ValueError("nudge positions and deltas do not align")
        self.positions = positions
        self.delta = delta

    def __call__(self, module, args, kwargs):
        hidden = args[0]
        if hidden.ndim != 3 or hidden.shape[0] != self.positions.shape[0]:
            raise ValueError("residual nudge does not align with the batch")
        patched = hidden.clone()
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        patched[rows, self.positions] = (
            patched[rows, self.positions] + self.delta.to(patched.dtype)
        )
        return (patched,) + tuple(args[1:]), kwargs


def unembedding_rows(arm: Arm) -> torch.Tensor:
    """Rows of the output embedding, one per vocabulary item."""

    head = arm.model.get_output_embeddings()
    if head is None or not hasattr(head, "weight"):
        raise TypeError(f"{arm.name}: no output embedding to read unembedding rows from")
    return head.weight.detach()


@torch.no_grad()
def query_source_intervention(
    arm: Arm,
    rows: Sequence[Sequence[int]],
    pool: InstancePool,
    selected: np.ndarray,
    heads: Sequence[tuple[int, int]],
    substitutes: np.ndarray,
    *,
    alphas: Sequence[float],
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """Attention onto ``A_q(X)`` as the *predicted* token is steered away from X.

    The residual stream at ``q`` receives ``alpha * ||x_q|| * (u_Y - u_X)``
    normalised to unit direction, before the layer holding the head being read.
    The token at ``q`` is untouched, so a head addressed by the current token's
    identity has no reason to move and a head addressed by the prediction does.

    A seeded random direction of identical norm is run alongside every alpha.
    Without it the measurement cannot separate "this head was addressed by the
    prediction" from "a perturbation of this size disrupts attention from ``q``
    however it is pointed", and at ``alpha = 2`` the perturbation is twice the
    residual norm, which disrupts a great deal.
    """

    arm.require("circuits")
    arm.require_eager_attention("the query-source intervention")
    if substitutes.shape != selected.shape:
        raise ValueError("one substitute token is required per selected instance")
    unembedding = unembedding_rows(arm).float()
    layers = sorted({layer for layer, _ in heads})
    by_layer = {layer: [head for candidate, head in heads if candidate == layer] for layer in layers}
    width = len(rows[0])

    results: dict[str, Any] = {"alphas": [float(value) for value in alphas], "heads": {}}
    manipulation: dict[str, list[float]] = {}

    for layer in layers:
        captured: dict[str, torch.Tensor] = {}

        def consume(_layer: int, weights: torch.Tensor) -> None:
            captured["weights"] = weights.float()

        tap = attention_module(arm, layer).register_forward_hook(_WeightTap(consume, layer))
        try:
            per_alpha_mass = np.zeros((len(alphas), len(by_layer[layer]), selected.size))
            control_mass = np.zeros_like(per_alpha_mass)
            per_alpha_target = np.zeros((len(alphas), selected.size))
            per_alpha_substitute = np.zeros((len(alphas), selected.size))
            for begin in range(0, selected.size, batch_size):
                block = selected[begin : begin + batch_size]
                sequence_ids = pool.sequence[block]
                ids = torch.tensor(
                    np.asarray([rows[int(index)] for index in sequence_ids], dtype=np.int64),
                    dtype=torch.long,
                    device=arm.device,
                )
                mask = torch.ones_like(ids)
                queries = torch.tensor(pool.query[block], dtype=torch.long, device=arm.device)
                targets = torch.tensor(
                    pool.predicted_token[block], dtype=torch.long, device=arm.device
                )
                replacements = torch.tensor(
                    substitutes[begin : begin + block.size], dtype=torch.long, device=arm.device
                )
                key_sets = antecedent_sets(rows, pool, block)
                key_mask = torch.zeros(
                    (ids.shape[0], width), dtype=torch.bool, device=arm.device
                )
                for row, keys in enumerate(key_sets):
                    key_mask[row, torch.tensor(keys, dtype=torch.long, device=arm.device)] = True
                rows_index = torch.arange(ids.shape[0], device=arm.device)

                hidden = _block_input(arm, layer, ids, mask)
                reference = hidden[rows_index, queries].float()
                direction = unembedding[replacements] - unembedding[targets]
                norm = direction.norm(dim=-1, keepdim=True)
                if float(norm.min()) <= 0:
                    raise ValueError("substitute and target unembedding rows coincide")
                direction = direction / norm
                scale = reference.norm(dim=-1, keepdim=True)
                generator = torch.Generator(device="cpu").manual_seed(seed + begin + layer)
                random_direction = torch.randn(
                    direction.shape, generator=generator, dtype=torch.float32
                ).to(direction.device)
                random_direction = random_direction / random_direction.norm(
                    dim=-1, keepdim=True
                )

                for alpha_index, alpha in enumerate(alphas):
                    for label, unit in (
                        ("prediction", direction),
                        ("random", random_direction),
                    ):
                        delta = float(alpha) * scale * unit
                        handle = arm.blocks()[layer].register_forward_pre_hook(
                            _ResidualNudge(queries, delta), with_kwargs=True
                        )
                        try:
                            logits = arm.model(input_ids=ids, attention_mask=mask).logits
                        finally:
                            handle.remove()
                        pattern = captured["weights"]
                        row_attention = pattern[rows_index, :, queries]
                        mass = (row_attention * key_mask.unsqueeze(1)).sum(dim=-1)
                        destination = (
                            per_alpha_mass if label == "prediction" else control_mass
                        )
                        for head_index, head in enumerate(by_layer[layer]):
                            destination[
                                alpha_index, head_index, begin : begin + block.size
                            ] = mass[:, head].cpu().numpy()
                        if label == "prediction":
                            probabilities = F.softmax(
                                logits[rows_index, queries].float(), dim=-1
                            )
                            per_alpha_target[alpha_index, begin : begin + block.size] = (
                                probabilities[rows_index, targets].cpu().numpy()
                            )
                            per_alpha_substitute[
                                alpha_index, begin : begin + block.size
                            ] = probabilities[rows_index, replacements].cpu().numpy()
                        del logits, pattern
                torch.cuda.empty_cache()
        finally:
            tap.remove()

        for head_index, head in enumerate(by_layer[layer]):
            results["heads"][f"L{layer}H{head}"] = {
                "layer": int(layer),
                "head": int(head),
                "antecedent_mass_by_alpha": [
                    _finite(float(per_alpha_mass[index, head_index].mean()), "antecedent mass")
                    for index in range(len(alphas))
                ],
                "antecedent_mass_by_alpha_random_control": [
                    _finite(float(control_mass[index, head_index].mean()), "control mass")
                    for index in range(len(alphas))
                ],
            }
        manipulation[f"layer_{layer}"] = {
            "p_target_by_alpha": [
                _finite(float(per_alpha_target[index].mean()), "p target")
                for index in range(len(alphas))
            ],
            "p_substitute_by_alpha": [
                _finite(float(per_alpha_substitute[index].mean()), "p substitute")
                for index in range(len(alphas))
            ],
        }
    results["manipulation_check"] = manipulation
    return results


@torch.no_grad()
def _block_input(
    arm: Arm, layer: int, ids: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}

    def hook(module, args, kwargs):
        captured["hidden"] = args[0].detach()
        return None

    handle = arm.blocks()[layer].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        arm.model(input_ids=ids, attention_mask=mask)
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError(f"{arm.name}: block {layer} was never entered")
    return captured["hidden"]


# ---------------------------------------------- decoy-corrected induction census


@torch.no_grad()
def decoy_corrected_prefix_matching(
    arm: Arm,
    probes: Sequence[RepeatProbe],
    *,
    batch_size: int,
    n_decoys: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Prefix matching with and without the same decoy subtraction PAA uses.

    The induction headline this programme reports is an uncorrected attention
    weight.  If subtracting position-matched decoys moves it materially, that is
    a finding about the induction census and has to be read before any PAA
    number is.
    """

    arm.require("circuits")
    arm.require_eager_attention("the decoy-corrected prefix-matching census")
    if not probes:
        raise ValueError("no probes supplied")
    heads = n_head(arm)
    layers = arm.n_layer
    rng = np.random.default_rng(seed)

    raw = np.zeros((len(probes), layers, heads), dtype=np.float64)
    corrected = np.zeros_like(raw)
    state: dict[str, Any] = {}

    def consume(layer: int, weights: torch.Tensor) -> None:
        pattern = weights.float()
        rows_index = state["rows"]
        queries = state["queries"]
        keys = state["keys"]
        decoys = state["decoys"]
        probe_index = state["probes"]
        target = pattern[rows_index, :, queries, keys]
        n, n_decoy = decoys.shape
        flat_rows = rows_index.unsqueeze(1).expand(n, n_decoy).reshape(-1)
        flat_queries = queries.unsqueeze(1).expand(n, n_decoy).reshape(-1)
        decoy_values = pattern[flat_rows, :, flat_queries, decoys.reshape(-1)]
        decoy_mean = decoy_values.view(n, n_decoy, heads).mean(dim=1)
        raw_block = torch.zeros((len(probes), heads), dtype=torch.float64, device=pattern.device)
        corrected_block = torch.zeros_like(raw_block)
        raw_block.index_add_(0, probe_index, target.double())
        corrected_block.index_add_(0, probe_index, (target - decoy_mean).double())
        raw[:, layer] += raw_block.cpu().numpy()
        corrected[:, layer] += corrected_block.cpu().numpy()

    handles = [
        attention_module(arm, layer).register_forward_hook(_WeightTap(consume, layer))
        for layer in range(layers)
    ]
    counts = np.zeros(len(probes), dtype=np.int64)
    try:
        for begin in range(0, len(probes), batch_size):
            chunk = probes[begin : begin + batch_size]
            width = max(len(probe.input_ids) for probe in chunk)
            if any(len(probe.input_ids) != width for probe in chunk):
                raise ValueError("repeat probes in a batch must share a length")
            ids = torch.tensor(
                np.asarray([probe.input_ids for probe in chunk], dtype=np.int64),
                dtype=torch.long,
                device=arm.device,
            )
            mask = torch.ones_like(ids)
            rows_list: list[int] = []
            probe_list: list[int] = []
            query_list: list[int] = []
            key_list: list[int] = []
            decoy_list: list[np.ndarray] = []
            for row, probe in enumerate(chunk):
                tokens = np.asarray(probe.input_ids, dtype=np.int64)
                for query, key in zip(probe.query_positions, probe.key_positions):
                    span = query - key
                    low, high = DISTANCE_BIN_EDGES[distance_bin(span)]
                    window = np.arange(max(0, query - high), query - low + 1)
                    if window.size == 0:
                        continue
                    predecessor_ok = np.ones(window.size, dtype=bool)
                    nonzero = window >= 1
                    predecessor_ok[nonzero] = tokens[window[nonzero] - 1] != int(tokens[query])
                    eligible = window[
                        (window != key) & (tokens[window] != int(tokens[key])) & predecessor_ok
                    ]
                    if eligible.size < 1:
                        continue
                    draw = rng.choice(
                        eligible, size=n_decoys, replace=eligible.size < n_decoys
                    )
                    rows_list.append(row)
                    probe_list.append(begin + row)
                    query_list.append(int(query))
                    key_list.append(int(key))
                    decoy_list.append(np.sort(draw.astype(np.int64)))
                    counts[begin + row] += 1
            if not rows_list:
                continue
            state["rows"] = torch.tensor(rows_list, dtype=torch.long, device=arm.device)
            state["probes"] = torch.tensor(probe_list, dtype=torch.long, device=arm.device)
            state["queries"] = torch.tensor(query_list, dtype=torch.long, device=arm.device)
            state["keys"] = torch.tensor(key_list, dtype=torch.long, device=arm.device)
            state["decoys"] = torch.tensor(
                np.stack(decoy_list), dtype=torch.long, device=arm.device
            )
            arm.model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    if not (counts > 0).all():
        raise RuntimeError(f"{arm.name}: a probe contributed no scored query positions")
    divisor = counts.astype(np.float64).reshape(-1, 1, 1)
    return {
        "per_probe_prefix_matching": raw / divisor,
        "per_probe_prefix_matching_decoy_corrected": corrected / divisor,
        "scored_positions_per_probe": counts,
    }


# ------------------------------------------------------------------ statistics


def cluster_bootstrap(
    per_cluster: np.ndarray,
    weights: np.ndarray,
    *,
    replicates: int,
    seed: int,
    alpha: float = 0.05,
    chunk: int = 100,
) -> dict[str, np.ndarray]:
    """Weighted mean over clusters with a percentile interval over cluster draws.

    ``per_cluster`` is ``(n_clusters, n_statistics)``.  Resampling clusters --
    sequences -- rather than instances is the whole point: instances inside one
    document are not independent, and an instance-level interval would be
    narrow for the wrong reason.
    """

    if per_cluster.ndim != 2 or weights.ndim != 1:
        raise ValueError("cluster bootstrap expects a (cluster, statistic) matrix")
    if per_cluster.shape[0] != weights.shape[0]:
        raise ValueError("weights do not align with clusters")
    if replicates < 100:
        raise ValueError("a percentile interval needs at least 100 replicates")
    keep = weights > 0
    values = per_cluster[keep]
    mass = weights[keep].astype(np.float64)
    if values.shape[0] < 2:
        raise ValueError("cluster bootstrap needs at least two non-empty clusters")
    rng = np.random.default_rng(seed)
    point = (values * mass[:, None]).sum(axis=0) / mass.sum()
    draws = np.empty((replicates, values.shape[1]), dtype=np.float64)
    n = values.shape[0]
    for begin in range(0, replicates, chunk):
        size = min(chunk, replicates - begin)
        index = rng.integers(0, n, size=(size, n))
        block_weights = mass[index]
        block_values = values[index]
        draws[begin : begin + size] = (block_values * block_weights[:, :, None]).sum(
            axis=1
        ) / block_weights.sum(axis=1)[:, None]
    return {
        "mean": point,
        "q_low": np.quantile(draws, alpha / 2.0, axis=0),
        "q_high": np.quantile(draws, 1.0 - alpha / 2.0, axis=0),
        "fraction_positive": (draws > 0).mean(axis=0),
    }


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for index in range(1, values.size + 1):
        if index == values.size or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


def _residualise(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones_like(x), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coefficients


def partial_spearman(
    first: np.ndarray, second: np.ndarray, control: np.ndarray
) -> dict[str, float]:
    """Spearman correlation between two head statistics, partialling out a third.

    The control is each head's total non-sink attention mass.  Without it a
    correlation between two attention-weight statistics is partly a correlation
    between two measurements of how much attention the head places anywhere at
    all.
    """

    for name, array in (("first", first), ("second", second), ("control", control)):
        if array.ndim != 1 or array.size != first.size:
            raise ValueError(f"{name} does not align with the head population")
    ranks = [_ranks(array.astype(np.float64)) for array in (first, second, control)]
    residual_first = _residualise(ranks[0], ranks[2])
    residual_second = _residualise(ranks[1], ranks[2])
    raw = float(np.corrcoef(ranks[0], ranks[1])[0, 1])
    partial = float(np.corrcoef(residual_first, residual_second)[0, 1])
    return {
        "spearman": _finite(raw, "spearman"),
        "partial_spearman": _finite(partial, "partial spearman"),
        "n_heads": int(first.size),
    }


def top_set_jaccard(first: np.ndarray, second: np.ndarray, *, count: int) -> dict[str, Any]:
    """Overlap of the two head rankings' top ``count`` members."""

    if first.shape != second.shape:
        raise ValueError("head matrices must share a shape")
    if count < 1 or count > first.size:
        raise ValueError("invalid top-set size")
    left = set(int(index) for index in np.argsort(first, axis=None)[::-1][:count])
    right = set(int(index) for index in np.argsort(second, axis=None)[::-1][:count])
    union = left | right
    return {
        "count": int(count),
        "intersection": len(left & right),
        "jaccard": _finite(len(left & right) / len(union), "jaccard"),
    }


def flat_head_index(matrix: np.ndarray, flat: int) -> tuple[int, int]:
    layer, head = np.unravel_index(int(flat), matrix.shape)
    return int(layer), int(head)


# -------------------------------------------------------------- CEM matching


def coarsened_cells(
    pool: InstancePool,
    corruption: np.ndarray,
    *,
    gates: Sequence[str],
) -> np.ndarray:
    """Coarsened cell label per instance under a prefix of the matching gates."""

    known = ("distance", "unigram_percentile", "confidence", "corruption")
    for gate in gates:
        if gate not in known:
            raise ValueError(f"unknown matching gate {gate!r}")
    if corruption.shape != (len(pool),):
        raise ValueError("corruption effects do not align with the pool")
    columns: list[np.ndarray] = []
    if "distance" in gates:
        columns.append(np.asarray([distance_bin(int(value)) for value in pool.distance]))
    if "unigram_percentile" in gates:
        columns.append(np.clip((pool.unigram_percentile * 10).astype(np.int64), 0, 9))
    if "confidence" in gates:
        columns.append(
            np.asarray(
                [_bin_index(float(value), CONFIDENCE_BIN_EDGES) for value in pool.confidence]
            )
        )
    if "corruption" in gates:
        columns.append(
            np.asarray(
                [_bin_index(float(value), CORRUPTION_BIN_EDGES) for value in corruption]
            )
        )
    if not columns:
        raise ValueError("at least one matching gate is required")
    stacked = np.column_stack(columns)
    return np.asarray(
        ["|".join(str(int(value)) for value in row) for row in stacked], dtype=object
    )


@torch.no_grad()
def corruption_effects(
    arm: Arm,
    rows: Sequence[Sequence[int]],
    pool: InstancePool,
    selected: np.ndarray,
    *,
    unigram_counts: np.ndarray,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Change in ``log p(X)`` at ``q`` when the antecedent token is replaced.

    The replacement is drawn from the arm's own unigram distribution, so the
    corrupted context stays inside the model's token distribution and the effect
    measured is the loss of that particular antecedent rather than the arrival
    of an impossible symbol.
    """

    if selected.size == 0:
        raise ValueError("no instances selected for corruption")
    rng = np.random.default_rng(seed)
    probabilities = unigram_counts.astype(np.float64)
    support = np.flatnonzero(probabilities > 0)
    probabilities = probabilities[support] / probabilities[support].sum()
    effects = np.zeros(selected.size, dtype=np.float64)

    for begin in range(0, selected.size, batch_size):
        block = selected[begin : begin + batch_size]
        base = np.asarray([rows[int(pool.sequence[index])] for index in block], dtype=np.int64)
        ids = torch.tensor(base, dtype=torch.long, device=arm.device)
        mask = torch.ones_like(ids)
        queries = torch.tensor(pool.query[block], dtype=torch.long, device=arm.device)
        targets = torch.tensor(pool.predicted_token[block], dtype=torch.long, device=arm.device)
        rows_index = torch.arange(ids.shape[0], device=arm.device)
        clean = F.log_softmax(
            arm.model(input_ids=ids, attention_mask=mask).logits[rows_index, queries].float(),
            dim=-1,
        )[rows_index, targets]

        corrupted = base.copy()
        for row, index in enumerate(block):
            original = int(pool.predicted_token[index])
            for _ in range(64):
                candidate = int(support[rng.choice(support.size, p=probabilities)])
                if candidate != original:
                    corrupted[row, int(pool.antecedent[index])] = candidate
                    break
            else:
                raise RuntimeError("could not draw a replacement token")
        corrupted_ids = torch.tensor(corrupted, dtype=torch.long, device=arm.device)
        dirty = F.log_softmax(
            arm.model(input_ids=corrupted_ids, attention_mask=mask)
            .logits[rows_index, queries]
            .float(),
            dim=-1,
        )[rows_index, targets]
        effects[begin : begin + block.size] = (dirty - clean).cpu().numpy()
        torch.cuda.empty_cache()
    return effects


__all__ = [
    "CONFIDENCE_BIN_EDGES",
    "CORRUPTION_BIN_EDGES",
    "DISTANCE_BIN_EDGES",
    "InstancePool",
    "build_instance_pool",
    "build_knockout_mask",
    "cluster_bootstrap",
    "coarsened_cells",
    "cohort_power_held_out",
    "corruption_effects",
    "decoy_corrected_prefix_matching",
    "distance_bin",
    "flat_head_index",
    "knockout_effects",
    "paa_attention_scores",
    "partial_spearman",
    "query_source_intervention",
    "scored_target_counts",
    "tokenised_rows",
    "top_set_jaccard",
    "unigram_percentiles",
]
