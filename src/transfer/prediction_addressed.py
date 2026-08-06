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

import inspect
import math
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Callable, Sequence

import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F

# Relative, like every other module in this package. These five were the only
# absolute ``src.transfer.*`` imports left inside ``src/transfer``, and an
# absolute import here resolves through whatever ``src`` happens to be on
# ``sys.path``: a caller that puts the repository root on the path and a caller
# that puts ``src`` on the path get two distinct module objects for the same
# file, with two copies of every module constant. Constants such as
# ``LAPLACE_SMOOTHING`` and ``KNOCKOUT_LOGIT`` are compared and monkeypatched by
# identity, so that duplication is not cosmetic.
from .arms import (
    Arm,
    Cohort,
    conditioning_boundary_ids,
    symbols_per_token,
)
from .budget import scored_tokens
from .circuits import RepeatProbe, content_bounds, layout_token_ids, n_head
from .pathways import (
    LAPLACE_SMOOTHING,
    assert_disjoint,
    disjoint_unigram_cross_entropy_nats,
    smoothing_diagnostics,
)
from .scoring import target_rule
from .statistics import (
    MINIMUM_BOOTSTRAP_UNITS,
    bootstrap_unit_floor,
    depth_controlled_rank_correlation,
)

#: Minimum bootstrap draws that must fall below the requested lower percentile
#: for the resulting bound to be an estimate rather than an order statistic of
#: the extreme draws.  Ten is the point at which the Monte-Carlo spread of the
#: bound across seeds drops below the smallest effect this census publishes; the
#: guard using it is in :func:`cluster_bootstrap`, with the measurement that
#: motivated it.
MINIMUM_DRAWS_IN_TAIL = 10.0

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

#: Largest attention mass a knocked-out head may still place on the antecedent
#: keys it was supposed to stop reading.
#:
#: The zero-mask control below establishes that the injection path is additive.
#: It says nothing about whether the *non*-zero mask lands on the right head or
#: the right keys, and nothing in the causal statistic reveals that either: a
#: mask that broadcasts across heads, or one whose head index is off by one,
#: produces a well-formed ``delta_m_gap`` of approximately zero for every head,
#: and the module then reports no causally confirmable suppressive head
#: population -- which this module's own docstring calls "a complete answer".
#: A negative result that an inert instrument produces identically is not a
#: result. Adding ``KNOCKOUT_LOGIT`` pre-softmax drives the surviving weight to
#: exactly zero in every dtype the panel runs, so the residual mass is a
#: manipulation check with no tolerance budget to spend; 1e-6 is float noise.
KNOCKOUT_RESIDUAL_MASS = 1.0e-6

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
    smoothing: float = LAPLACE_SMOOTHING,
) -> dict[str, Any]:
    """Context-derived information with a held-out unigram baseline.

    The plug-in estimator this replaces is biased downwards by up to 1.02 nats
    on a 50k-vocabulary arm and by 0.01 on a residue-level one, so it inflates
    exactly the arms whose relative position a modality reading depends on.

    The held-out estimator carries a smaller bias of its own, upwards and on the
    same vocabulary axis, from its additive smoothing: +0.224 nats at
    ``V = 50257`` against +0.0001 at ``V = 32`` on a 100k-token reference. It is
    not removed -- there is no unsmoothed held-out unigram, since one unseen
    target token makes the estimate infinite -- so it is measured and published
    beside the headline, together with the same baseline recomputed over a
    sweep of the constant. ``smoothing`` is exposed for that sweep;
    :data:`~src.transfer.pathways.LAPLACE_SMOOTHING` records why the default is
    unchanged.
    """

    if threshold_nats <= 0:
        raise ValueError("the power threshold must be positive")
    assert_disjoint(cohort, reference)
    inputs = cohort.input_strings(arm)
    scored = scored_tokens(arm, inputs, max_len=max_len, batch_size=batch_size)
    vocab = int(arm.model.config.vocab_size)
    target_counts = np.bincount(scored.target_ids, minlength=vocab).astype(np.int64)
    reference_counts = scored_target_counts(arm, reference.input_strings(arm), max_len=max_len)
    held_out = disjoint_unigram_cross_entropy_nats(
        reference_counts, target_counts, smoothing=smoothing
    )
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
        # The smoothing constant is a second, opposite bias on the same
        # vocabulary axis as the plug-in bias reported above it. Reporting one
        # and not the other is how a corrected estimator inherits the shape of
        # the error it corrected.
        "unigram_smoothing": smoothing_diagnostics(
            reference_counts, target_counts, smoothing=smoothing
        ),
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
    #: First token index that carries modality content; everything below it is
    #: format scaffolding. Carried on the pool rather than passed alongside it
    #: because :func:`antecedent_sets` -- which decides which keys the causal
    #: knockout removes -- takes only ``rows`` and a pool, and searched from
    #: index 0. ``tokenised_rows`` has always computed this bound and used it to
    #: exclude scaffolding from the *queries*, and nothing excluded it from the
    #: antecedents or the decoys, so ``k*`` could land on ProtGPT2's
    #: end-of-text/newline wrapping or ZymCTRL's EC tag, ``<sep>`` and
    #: ``<start>``. Those tokens recur in every row by construction, so the
    #: protein-side instance population could be dominated by FASTA formatting
    #: while the text side had no equivalent. Defaults to zero so that a pool
    #: built by hand is unchanged.
    content_low: int = 0

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
    if (
        array.ndim != 1
        or not np.isfinite(array).all()
        or (array < 0).any()
        or array.sum() <= 0
    ):
        raise ValueError("unigram counts must be a non-empty vector")
    unique_counts, inverse = np.unique(array, return_inverse=True)
    mass_by_count = unique_counts * np.bincount(inverse, minlength=unique_counts.size)
    cumulative_mass = np.cumsum(mass_by_count) / array.sum()
    return cumulative_mass[inverse]


def a1_candidate_pool_verdict(n_instances: int, minimum: int) -> str:
    """``PASS`` or ``FAIL`` for the A1 candidate-pool gate.

    One declaration because the rule has two consumers that must not drift: the
    census stage writes the verdict into its artefact, and the causal stage has
    to refuse a pool that fails it.  Until now only the writer existed.  The
    consequence was measured rather than supposed -- of the 61 retained census
    artefacts, **eight carry ``verdict: FAIL`` and six of those have a
    ``causal.json`` produced from them in the same invocation**: ProtGPT2 ban 3 /
    n=200 at 9,958-10,040 instances and dialogpt-small ban 3 at 13,315-13,544,
    both against a gate of 20,000.  Those six are the ProtGPT2 and dialogpt rows
    of the published D2.c table, and EXP-R2-116 recorded quoting them "without
    reading the gate verdict that sits in the artefact beside the number".

    The design had predicted the failure before the runs happened -- the audit's
    D2.c step 2 states that ProtGPT2 yields ~50 instances per sequence, so n=200
    cannot reach 20,000 -- which is precisely why a verdict nothing reads is
    worse than no verdict at all: it looks like the check was made.

    An operator who wants a smaller pool lowers ``--a1-minimum`` explicitly, and
    that choice is recorded in the run's ``settings``.  Evidence-discipline rule
    2's shape applies: an under-powered arm is reported as unmeasurable at this
    configuration, not as having produced a number.
    """

    return "PASS" if int(n_instances) >= int(minimum) else "FAIL"


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

    # Derived here from the same declaration ``tokenised_rows`` uses, rather
    # than accepted as an argument, so that every existing caller gets the
    # scaffolding exclusion without having to remember to pass it. The rows
    # arrive equal-length and from one arm, so one bound serves them all; that
    # they agree is checked rather than assumed, because a row whose scaffolding
    # is a different length is a rendering fault and not something to average
    # over. Position 0 is excluded on top of it: it is the attention sink, which
    # this module measures separately and must not draw keys from.
    bounds = {content_bounds(arm, row, width)[0] for row in rows}
    if len(bounds) != 1:
        raise ValueError(f"{arm.name}: inconsistent content offsets {sorted(bounds)}")
    content_low = int(bounds.pop())
    key_floor = max(1, content_low)
    if query_min < key_floor:
        raise ValueError(
            f"{arm.name}: query_min={query_min} reaches into the {content_low}-token "
            "format scaffolding, which carries no repeatable content"
        )

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

    # Resolved once over the tokens this cohort actually contains, rather than
    # over the whole vocabulary: decoding 50257 pieces per arm to find two of
    # them is the kind of cost that gets a guard removed later.
    layout_tokens = layout_token_ids(arm, {int(t) for row in rows for t in row})
    cascade_layout_vocabulary = sorted(layout_tokens)
    # Materialised once for the decoy filter below, which is vectorised over the
    # window and cannot use the set membership the candidate loop uses.
    layout_key_ids = np.asarray(cascade_layout_vocabulary, dtype=np.int64)

    cascade = {
        "positions_scored": 0,
        "positions_with_eligible_candidate": 0,
        "candidates_discarded_by_induction_target": 0,
        "candidates_discarded_by_distance_range": 0,
        "candidates_discarded_by_induction_and_distance": 0,
        "candidates_discarded_by_layout_token": 0,
        "positions_with_no_antecedent_candidate": 0,
        "candidates_discarded_by_empty_decoy_pool": 0,
        "instances_retained": 0,
        "instances_dropped_by_per_sequence_cap": 0,
        "decoys_drawn_with_replacement": 0,
        "ban_depth": int(ban),
        "per_sequence_cap": max_per_sequence,
        "per_sequence_cap_selection": "seeded uniform draw over the row's eligible queries",
        "content_low": content_low,
        "key_floor": key_floor,
        "layout_tokens_excluded_from_candidates": cascade_layout_vocabulary,
        "layout_tokens_excluded_from_decoys": cascade_layout_vocabulary,
        "layout_reason": (
            "a predicted line break is a prediction about the record's layout "
            "rather than its sequence, and only ProtGPT2's rendering emits them; "
            "the judgement is circuits.layout_token_ids, which fit_unigram has "
            "used since the module was written. The same vocabulary is barred "
            "from the decoy pool, because the decoy mean is the subtrahend of "
            "every head's paa_specific score and an anomalous key inflates it -- "
            "the argument key_floor_reason makes for the attention sink"
        ),
        "key_floor_reason": (
            "antecedents and decoys are drawn at or above max(1, content_low): "
            "position 0 is the attention sink, and positions below content_low are "
            "format scaffolding whose repetition is a property of the rendering "
            "rather than of the sequence"
        ),
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
            # Every eligible query in the row is built first and the cap is
            # applied afterwards, by a seeded draw. Taking the first
            # ``max_per_sequence`` was a "first N" selection inside a function
            # that already holds an ``rng``, and it is not a neutral one:
            # earlier queries have less context, so capping from the front
            # shifts both the query-position distribution and the
            # antecedent-distance distribution -- and distance is one of the
            # matching covariates the CEM gate balances on. Appendix B rule 1 is
            # about corpora, but the mechanism is the same and this programme
            # has manufactured an effect with it three times. The cap is unused
            # by the campaign today, which is why this is a latent defect rather
            # than a correction; with the cap unset the draw order and the RNG
            # stream are byte-for-byte what they were.
            row_instances: list[dict[str, Any]] = []
            for q in range(query_min, width):
                cascade["positions_scored"] += 1
                banned = set(int(value) for value in top_ids[row_index, q, :ban])
                chosen: tuple[int, int, float] | None = None
                induction_blocked = False
                distance_blocked = False
                layout_blocked = False
                for depth in range(candidate_depth):
                    token = int(top_ids[row_index, q, depth])
                    probability = float(top_probabilities[row_index, q, depth])
                    if probability < min_confidence:
                        break
                    # A predicted line break is a prediction about the record's
                    # layout, not about its sequence, and only one arm's
                    # rendering has them. Admitting them made ProtGPT2's census
                    # 28-33% FASTA wraps carrying more than the whole
                    # clean-margin mass, and its copy-suppression correlation
                    # reported the opposite sign from the residues (EXP-R2-116).
                    # The judgement is `circuits.layout_token_ids`, imported
                    # rather than repeated: `fit_unigram` has excluded these
                    # since the module was written, for the same reason.
                    if token in layout_tokens:
                        layout_blocked = True
                        continue
                    # Searched from the content floor, not from index 0: an
                    # antecedent inside the format scaffolding is a repetition
                    # of the rendering, not of the sequence.
                    matches = np.flatnonzero(tokens[key_floor:q] == token) + key_floor
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
                    # The cascade has to close: `positions_scored` must equal the
                    # sum of its exits, or a reader cannot tell an excluded
                    # position from an unaccounted one. It did not. A position
                    # blocked by *both* rules was charged to induction alone, and
                    # a position where no candidate ever reached the antecedent
                    # test -- nothing above `min_confidence`, or no earlier
                    # occurrence of any top-k token -- was charged to nothing at
                    # all: 15,536 of gpt2-large's 96,000 scored positions, 16.2%,
                    # fell into no category.
                    if induction_blocked and distance_blocked:
                        cascade["candidates_discarded_by_induction_and_distance"] += 1
                    elif induction_blocked:
                        cascade["candidates_discarded_by_induction_target"] += 1
                    elif distance_blocked:
                        cascade["candidates_discarded_by_distance_range"] += 1
                    elif layout_blocked:
                        cascade["candidates_discarded_by_layout_token"] += 1
                    else:
                        cascade["positions_with_no_antecedent_candidate"] += 1
                    continue
                cascade["positions_with_eligible_candidate"] += 1
                token, star, probability = chosen
                span = q - star
                low, high = DISTANCE_BIN_EDGES[distance_bin(span)]
                # Position 0 is the attention sink and is not a key like any
                # other. This module measures ``non_sink_mass = 1 - A[q, 0]``
                # twenty lines away, so it already knows that; the decoy window
                # started at ``max(0, q - high)`` anyway and therefore admitted
                # the sink as a decoy for every instance whose distance bin
                # reached back to the start of the row. Appendix B rule 9 is the
                # same point on the other axis. The consequence is not a wash:
                # the sink carries far more mass than an ordinary key, so an
                # instance whose decoys include it gets a large subtrahend and
                # its ``paa_specific`` can change sign -- and *which* instances
                # those are is decided by the distance bin, which is a matching
                # covariate. Sink-inclusive decoys therefore put a covariate
                # into the statistic the covariate is supposed to balance.
                lowest = max(key_floor, q - high)
                highest = q - low
                if highest < lowest:
                    cascade["candidates_discarded_by_empty_decoy_pool"] += 1
                    continue
                window = np.arange(lowest, highest + 1)
                predecessor_ok = np.ones(window.size, dtype=bool)
                nonzero = window >= 1
                predecessor_ok[nonzero] = tokens[window[nonzero] - 1] != int(tokens[q])
                # A layout token is barred from the decoy pool for the reason the
                # paragraph above bars position 0, and the reason is not that it
                # is rare. ``paa_specific_matched`` is
                # ``antecedent_set_attention - decoy_mean * n_keys``, so the decoy
                # draw is the *subtrahend* of every head's score: a decoy that
                # carries anomalous attention inflates the baseline subtracted
                # from whichever heads attend to it. Excluding a predicted line
                # break (EXP-R2-118) cleaned the numerator and left this alone.
                # Measured on the repaired pools: 4.30% of ProtGPT2's decoy keys
                # and 2.74% of gpt2-large's are the wrap, reaching 16.1% and 9.2%
                # of instances -- so the arm whose rendering has layout tokens
                # takes the larger baseline distortion, which is the asymmetry
                # EXP-R2-116 was written about.
                eligible = window[
                    (window != star)
                    & (tokens[window] != token)
                    & (~np.isin(tokens[window], list(banned)))
                    & (~np.isin(tokens[window], layout_key_ids))
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

                row_instances.append(
                    {
                        "query": q,
                        "antecedent": star,
                        "predicted": token,
                        "confidence": probability,
                        "distance": span,
                        "percentile": float(percentile[token]),
                        "decoys": np.sort(draw.astype(np.int64)),
                        "target_logit": target_logit,
                        "runner_up": runner_up,
                    }
                )
            if max_per_sequence is not None and len(row_instances) > max_per_sequence:
                keep = sorted(
                    int(index)
                    for index in rng.choice(
                        len(row_instances), size=max_per_sequence, replace=False
                    )
                )
                cascade["instances_dropped_by_per_sequence_cap"] += (
                    len(row_instances) - max_per_sequence
                )
                row_instances = [row_instances[index] for index in keep]
            for instance in row_instances:
                sequences.append(sequence_index)
                queries.append(instance["query"])
                antecedents.append(instance["antecedent"])
                predicted.append(instance["predicted"])
                confidences.append(instance["confidence"])
                distances.append(instance["distance"])
                percentiles.append(instance["percentile"])
                decoy_rows.append(instance["decoys"])
                target_logits.append(instance["target_logit"])
                runner_up_logits.append(instance["runner_up"])
                cascade["instances_retained"] += 1
        del logits
        torch.cuda.empty_cache()

    # Every scored position leaves through exactly one exit. Checked rather than
    # trusted, because the previous accounting silently lost 16.2% of gpt2-large's
    # positions and a rate quoted against a partial denominator reads exactly like
    # a rate quoted against a whole one.
    exits = (
        "positions_with_eligible_candidate",
        "candidates_discarded_by_induction_target",
        "candidates_discarded_by_distance_range",
        "candidates_discarded_by_induction_and_distance",
        "candidates_discarded_by_layout_token",
        "positions_with_no_antecedent_candidate",
    )
    accounted = sum(cascade[name] for name in exits)
    if accounted != cascade["positions_scored"]:
        raise RuntimeError(
            f"{arm.name}: the instance cascade does not close -- "
            f"{cascade['positions_scored']} positions scored against {accounted} "
            f"accounted for across {list(exits)}"
        )
    cascade["cascade_closes_over"] = " + ".join(exits)

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
        content_low=content_low,
        clean_logit_target=np.asarray(target_logits, dtype=np.float64),
        clean_logit_runner_up=np.asarray(runner_up_logits, dtype=np.float64),
        cascade=cascade,
    )


# --------------------------------------------------------------- attention taps


#: Architectures whose pre-softmax attention this module can both read and
#: override.
#:
#: Not the same set as ``circuits._CIRCUIT_ARCHITECTURES`` and deliberately so.
#: That declaration answers "can a per-head OV circuit be rebuilt from the
#: weights"; this one answers "can the attention pattern be tapped and can a key
#: be removed from it before the softmax", which is all this census does. The
#: sets happen to overlap on four architectures and differ on ``t5_decoder``,
#: whose sublayer decomposition is not commensurate (``arms._DECOMPOSABLE``)
#: while its pattern and its additive term are both addressable -- see
#: :class:`_RelativeBiasKnockout` for what "addressable" costs there.
#:
#: ``scripts/transfer/panel_contract.py`` mirrors this rather than the circuit
#: declaration, so a stage cannot be scheduled on an arm whose knockout this
#: module has no path for.
PAA_ARCHITECTURES = frozenset({"gpt2", "progen", "llama", "qwen2", "t5_decoder"})


def attention_module(arm: Arm, layer: int) -> torch.nn.Module:
    """The attention submodule whose forward computes the pattern.

    ``Arm.attention_pattern_module`` rather than ``Arm.attention``: this module
    reads and overrides a pattern and never decomposes a block into sublayers, so
    the decomposition claim the latter accessor carries is not one it needs. The
    two resolve the same module for every arm that has both.
    """

    return arm.attention_pattern_module(layer)


class _RequestAttentionWeights:
    """Pre-hook asking one attention module to return its pattern.

    Whether an eager attention module emits its pattern unconditionally is an
    architecture detail, not a property of the kernel.  GPT-2's returns
    ``(output, weights)`` on every call; ProGen2 ships its own modelling code,
    whose attention returns ``(output, present)`` and appends the weights *only*
    when its forward is asked for them.  Requesting them per module keeps the
    layer-by-layer memory profile :class:`_WeightTap` exists for -- the
    model-level ``output_attentions=True`` materialises all 36 layers at once.
    """

    def __call__(self, module, args, kwargs):
        kwargs["output_attentions"] = True
        return args, kwargs


class _WeightTap:
    """Forward hook returning the eager attention pattern of one layer.

    ``output_attentions=True`` at the model level would materialise every
    layer's pattern at once; at 36 layers, 20 heads and 512 tokens that is
    377 MiB per sequence held simultaneously.  Tapping layer by layer and
    reducing inside the hook keeps the peak at one layer.

    **The pattern is identified by contract rather than by tuple position.**
    Position 1 is the pattern on GPT-2 and the key-value cache on ProGen2, and
    reading it blindly is a silent-wrong-answer path rather than a failure: a
    cache is ``(batch, head, token, d_head)``, which the previous
    ``output[1] is None`` check rejects only because ``use_cache`` happens to be
    off at every call site today.

    The contract is two properties, and the second is not decoration.  A pattern
    has four axes, the arm's head count on axis 1 and square trailing axes --
    which is also, exactly, the shape of a T5 decoder's ``position_bias``, so on
    ByGPT5 the shape test alone finds two candidates and the tap can only refuse.
    What separates them is that a pattern's rows are a probability distribution
    and a position bias's rows are not: the bias carries the causal mask, so its
    masked entries are the dtype minimum and its row sums are large and negative.
    Both properties are required and the tap refuses unless exactly one tensor has
    them -- an output with two distributions of pattern shape is as unreadable as
    an output with none.
    """

    #: Slack on a row sum of one.  Generous because the row may be accumulated in
    #: bfloat16 by the model that produced it: 192 weights each carrying up to
    #: 2^-8 relative error sum to within 0.4% of one, and no position bias comes
    #: anywhere near this band.
    ROW_SUM_TOLERANCE = 1.0e-2

    def __init__(
        self,
        consume: Callable[[int, torch.Tensor], None],
        layer: int,
        *,
        n_heads: int,
    ) -> None:
        self.consume = consume
        self.layer = layer
        self.n_heads = n_heads

    def _is_pattern(self, item: object) -> bool:
        if (
            not isinstance(item, torch.Tensor)
            or item.ndim != 4
            or item.shape[1] != self.n_heads
            or item.shape[-1] != item.shape[-2]
        ):
            return False
        # In float32 whatever the module's own dtype: a bfloat16 accumulation of
        # the row sum would itself be worth several times the tolerance.
        sums = item.detach().float().sum(dim=-1)
        return bool(
            torch.isfinite(sums).all()
            and float((sums - 1.0).abs().max()) <= self.ROW_SUM_TOLERANCE
        )

    def __call__(self, module, args, output):
        items = output if isinstance(output, tuple) else (output,)
        found = [item for item in items if self._is_pattern(item)]
        if len(found) != 1:
            shaped = sum(
                1
                for item in items
                if isinstance(item, torch.Tensor)
                and item.ndim == 4
                and item.shape[1] == self.n_heads
                and item.shape[-1] == item.shape[-2]
            )
            raise RuntimeError(
                f"layer {self.layer}: expected exactly one "
                f"(batch, {self.n_heads}, query, key) tensor whose rows sum to one "
                f"in the attention module's output, found {len(found)} of "
                f"{shaped} with that shape; load the arm with "
                "attn_implementation='eager'"
            )
        self.consume(self.layer, found[0])
        return None


def tap_attention(
    arm: Arm, layer: int, consume: Callable[[int, torch.Tensor], None]
) -> list[Any]:
    """Register the pattern tap for one layer and return its hook handles.

    Single declaration: every read of an attention pattern in this module goes
    through here, so the ``output_attentions`` negotiation and the shape
    contract cannot drift apart between the five call sites that need them.
    The request hook is registered only when the module's forward declares the
    parameter, so an architecture that emits the pattern unconditionally is
    left exactly as it was.
    """

    module = attention_module(arm, layer)
    handles: list[Any] = []
    parameters = inspect.signature(type(module).forward).parameters
    if "output_attentions" in parameters:
        handles.append(
            module.register_forward_pre_hook(
                _RequestAttentionWeights(), with_kwargs=True
            )
        )
    handles.append(
        module.register_forward_hook(_WeightTap(consume, layer, n_heads=n_head(arm)))
    )
    return handles


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

    **Two scores, because the selector and the causal statistic disagree about
    what a key is, and the disagreement is alphabet-size-dependent.**
    ``paa_specific`` scores attention onto ``pool.antecedent`` -- the *nearest*
    earlier occurrence of the predicted token -- against a decoy baseline.
    :func:`knockout_effects` removes *every* earlier occurrence, the set
    :func:`antecedent_sets` returns.  Counted from the shipped pools, that is a
    median of 3 occurrences per instance on gpt2-large against 13 to 17 on
    ProGen2-medium, so a rank correlation between this score and that causal
    effect is attenuated harder on a small alphabet **by construction**, in the
    direction any modality hypothesis would predict.  ``corruption_effects``
    already fixed this exact error for the matching gate and its docstring names
    it: "a conclusion the estimator manufactured out of alphabet size".

    ``paa_specific_matched`` is the score whose key set is the one the knockout
    removes: attention *summed* over the whole antecedent set, against a decoy
    baseline scaled to the same number of keys, so the correction stays a
    positional baseline of matched size rather than a per-key one subtracted from
    a sum.  The sum rather than the mean because the intervention's size scales
    with the total mass it removes, which is what the causal effect responds to.

    Both are returned rather than one replacing the other.  ``paa_specific`` is
    the statistic EXP-R2-059 published and L5/L6 quote, and silently redefining it
    would make those numbers unreproducible; ``paa_specific_matched`` is the one a
    census-to-causal comparison has to use.  The choice is then visible in the
    artefact instead of made in an analysis script -- the same reason this package
    emits two effect scales and two standard-error units elsewhere.
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
    matched_sum = torch.zeros_like(antecedent_sum)
    matched_decoy_sum = torch.zeros_like(antecedent_sum)
    counts = np.zeros(n_sequences, dtype=np.int64)
    key_counts = np.zeros(n_sequences, dtype=np.int64)

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

        # The knockout's key set. Padded with the query's own position and masked
        # out rather than with 0, because 0 is the attention sink and a padding
        # index that lands on it would be read as real mass if the mask were ever
        # dropped.
        padded = state["key_set"]
        key_mask = state["key_set_mask"]
        n_keys = state["key_set_sizes"]
        width = padded.shape[1]
        flat_rows_k = rows_index.unsqueeze(1).expand(n, width).reshape(-1)
        flat_queries_k = queries.unsqueeze(1).expand(n, width).reshape(-1)
        key_values = pattern[flat_rows_k, :, flat_queries_k, padded.reshape(-1)]
        key_values = key_values.view(n, width, heads) * key_mask.unsqueeze(-1)
        matched = key_values.sum(dim=1)
        # A positional baseline of the same size as the set it corrects, not a
        # per-key baseline subtracted from a sum.
        matched_decoy = decoy_mean * n_keys.unsqueeze(-1)
        matched_sum[:, layer].index_add_(0, sequence_index, matched.double())
        matched_decoy_sum[:, layer].index_add_(0, sequence_index, matched_decoy.double())

    handles = [
        handle
        for layer in range(layers)
        for handle in tap_attention(arm, layer, consume)
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
            # One declaration of the knockout's key set, imported rather than
            # re-derived: this is the function `knockout_effects` calls.
            key_sets = antecedent_sets(rows, pool, selected)
            width = max(len(keys) for keys in key_sets)
            padded = np.empty((len(key_sets), width), dtype=np.int64)
            key_mask = np.zeros((len(key_sets), width), dtype=np.float32)
            for index, keys in enumerate(key_sets):
                padded[index, : len(keys)] = keys
                padded[index, len(keys) :] = int(pool.query[selected[index]])
                key_mask[index, : len(keys)] = 1.0
            state["key_set"] = torch.tensor(padded, dtype=torch.long, device=device)
            state["key_set_mask"] = torch.tensor(key_mask, dtype=torch.float32, device=device)
            state["key_set_sizes"] = torch.tensor(
                np.asarray([len(keys) for keys in key_sets], dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )
            arm.model(input_ids=ids, attention_mask=mask, use_cache=False)
            np.add.at(counts, pool.sequence[selected], 1)
            np.add.at(
                key_counts,
                pool.sequence[selected],
                np.asarray([len(keys) for keys in key_sets], dtype=np.int64),
            )
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
    matched_mean = (matched_sum / divisor).cpu().numpy()
    matched_decoy_mean = (matched_decoy_sum / divisor).cpu().numpy()
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
    for block in (
        antecedent_mean,
        decoy_mean,
        sink_mean,
        matched_mean,
        matched_decoy_mean,
    ):
        block[inactive] = np.nan
    return {
        "instances_per_sequence": counts,
        "active_sequences": active,
        "antecedent_attention": antecedent_mean,
        "decoy_attention": decoy_mean,
        "paa_specific": antecedent_mean - decoy_mean,
        "non_sink_mass": 1.0 - sink_mean,
        # The knockout-matched pair. `keys_per_instance` is the quantity that
        # differs by alphabet size -- a median of 3 on gpt2-large against 13 to 17
        # on ProGen2-medium -- and it is emitted so that a reader can see how far
        # the two scores can diverge on this arm rather than having to assume.
        "antecedent_set_attention": matched_mean,
        "decoy_attention_size_matched": matched_decoy_mean,
        "paa_specific_matched": matched_mean - matched_decoy_mean,
        "keys_per_instance": np.where(active, key_counts / np.maximum(counts, 1), np.nan),
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

    def __init__(self, mask: torch.Tensor) -> None:
        if mask.ndim != 4:
            raise ValueError("knockout mask must be (batch, head, query, key)")
        self.mask = mask

    def __call__(self, module, args, kwargs):
        existing = kwargs.get("attention_mask")
        if existing is None:
            kwargs["attention_mask"] = self.mask
        else:
            kwargs["attention_mask"] = existing + self.mask
        return args, kwargs


class _RelativeBiasKnockout:
    """The same removal where the additive term is a *shared* relative position bias.

    A T5-derived decoder has no per-layer additive attention mask to add to.  Its
    attention takes the causal mask under the name ``mask`` and consults it only
    when ``position_bias`` is absent, which is true at layer 0 and nowhere else:
    layer 0 builds ``relative bias + causal mask`` once, returns it, and
    ``T5Stack`` re-reads it from every block's output and hands it to the next.
    Two failure modes follow from that sharing and neither announces itself.
    Adding to ``mask`` above layer 0 changes nothing at all.  Adding to
    ``position_bias`` at layer *L* changes every layer above *L* as well, so a
    one-head knockout would silently run on the rest of the stack -- and the
    residual-mass control could not see it, because the head it re-reads is the
    one that was knocked out correctly.

    The hook therefore holds the clean bias, hands the module ``clean + mask``,
    and puts ``clean`` back into the output the stack propagates.  At layer 0 the
    bias does not exist yet and is built from the module's own ``compute_bias``
    and the causal mask the stack passed in -- reproducing ``T5Attention``'s own
    two lines, not re-deriving the masking.  That reconstruction is exactly what
    :func:`knockout_effects`' all-zero mask control checks: an inexact one moves
    the logits and the control refuses the run.
    """

    def __init__(self, mask: torch.Tensor) -> None:
        if mask.ndim != 4:
            raise ValueError("knockout mask must be (batch, head, query, key)")
        self.mask = mask
        self._clean: torch.Tensor | None = None

    def __call__(self, module, args, kwargs):
        bias = kwargs.get("position_bias")
        if bias is None:
            bias = self._shared_bias(module, args, kwargs)
        self._clean = bias
        kwargs["position_bias"] = bias + self.mask.to(bias.dtype)
        return args, kwargs

    def _shared_bias(self, module, args, kwargs) -> torch.Tensor:
        if not getattr(module, "has_relative_attention_bias", False):
            raise RuntimeError(
                "no position bias reached this attention module and it cannot "
                "compute one, so there is no additive term for the knockout to "
                "add to and the intervention would not happen"
            )
        hidden = args[0] if args else kwargs["hidden_states"]
        length = int(hidden.shape[1])
        bias = module.compute_bias(
            length,
            length,
            device=hidden.device,
            cache_position=kwargs.get("cache_position"),
        )[:, :, -length:, :]
        causal = kwargs.get("mask")
        if causal is not None:
            bias = bias + causal[:, :, :, :length]
        return bias

    def restore(self, module, args, output):
        """Return the *clean* bias to the stack, so the knockout stays at one layer."""

        if self._clean is None:
            raise RuntimeError(
                "the relative-bias knockout's output hook ran without its pre-hook, "
                "so the poisoned bias would propagate to every layer above this one"
            )
        return (output[0], self._clean) + tuple(output[2:])


def install_knockout(arm: Arm, layer: int, mask: torch.Tensor) -> list[Any]:
    """Register the additive pre-softmax knockout for one layer, and return its handles.

    Single declaration of *where* the intervention enters, because the answer is
    per architecture and getting it wrong is not a crash on every architecture it
    could be wrong on.  The four decoders whose attention takes an additive
    ``attention_mask`` share one hook; ``t5_decoder`` does not have one and needs
    :class:`_RelativeBiasKnockout`.
    """

    module = attention_module(arm, layer)
    architecture = arm.spec.architecture
    if architecture not in PAA_ARCHITECTURES:
        raise TypeError(
            f"{arm.name}: no attention knockout is implemented for {architecture!r}; "
            f"implemented: {sorted(PAA_ARCHITECTURES)}"
        )
    if architecture == "t5_decoder":
        hook = _RelativeBiasKnockout(mask)
        return [
            module.register_forward_pre_hook(hook, with_kwargs=True),
            module.register_forward_hook(hook.restore),
        ]
    return [module.register_forward_pre_hook(_AntecedentKnockout(mask), with_kwargs=True)]


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
    """All earlier occurrences of ``X`` for each selected instance.

    Searched from ``pool.content_low``, and never below position 0's successor.
    This function decides which keys the causal knockout removes, so a key it
    returns from inside the format scaffolding is a key the intervention
    silently adds: the head would be stopped from reading ProtGPT2's newline or
    ZymCTRL's ``<sep>`` as well as the antecedent, and the resulting
    ``delta_m_gap`` would be attributed to the antecedent.
    """

    floor = max(1, int(pool.content_low))
    sets: list[list[int]] = []
    for index in selected:
        tokens = np.asarray(rows[int(pool.sequence[index])], dtype=np.int64)
        q = int(pool.query[index])
        token = int(pool.predicted_token[index])
        positions = np.flatnonzero(tokens[floor:q] == token) + floor
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
    residual_mass_tolerance: float = KNOCKOUT_RESIDUAL_MASS,
) -> dict[str, np.ndarray]:
    """Change in M-gap and in ``p(X)`` when a head stops reading the antecedents.

    A positive M-gap change means the head was *suppressing* ``X``: removing its
    read of the antecedent raised the margin between ``X`` and its strongest
    competitor.

    Two controls, pointing in opposite directions, because the two failures they
    catch are opposite.

    *Negative.* An all-zero mask is injected first and its effect on the logits
    is required to be below ``zero_mask_tolerance``. That pass establishes that
    the injection path is additive and inert when it carries nothing; without
    it, a build whose attention module receives no additive mask would have its
    causal masking *replaced* by the injected tensor, and every knockout effect
    would be a difference between two non-causal forward passes.

    *Positive.* The knocked head's own attention onto the antecedent keys is
    re-read **during** each knockout pass and required to have collapsed below
    ``residual_mass_tolerance``. Until this existed the only antecedent mass the
    module measured came from the clean pass -- the taps are removed before any
    knockout runs -- so nothing in the run showed the intervention taking
    effect. A mask that broadcast over heads, or a head index off by one, would
    knock out nothing, return ``delta_m_gap`` near zero for every head, and be
    reported as the absence of a causally confirmable suppressive head
    population, which this module states up front is "a complete answer". Both
    masses are returned, so the collapse lands in the artefact rather than only
    in an assertion.
    """

    arm.require("circuits")
    arm.require_eager_attention("the antecedent-knockout causal statistic")
    if zero_mask_tolerance <= 0:
        raise ValueError("the zero-mask tolerance must be positive")
    if residual_mass_tolerance <= 0:
        raise ValueError("the residual-mass tolerance must be positive")
    if selected.size == 0:
        raise ValueError("no instances selected for the causal statistic")
    width = len(rows[0])
    n_heads = n_head(arm)
    dtype = next(arm.model.parameters()).dtype
    delta_gap = np.zeros((len(heads), selected.size), dtype=np.float64)
    delta_probability = np.zeros_like(delta_gap)
    antecedent_mass = np.zeros_like(delta_gap)
    knocked_mass = np.zeros_like(delta_gap)
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
            handle
            for layer in layers_needed
            for handle in tap_attention(arm, layer, consume)
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
        zero_handles = install_knockout(arm, 0, zero_mask)
        try:
            null_logits = arm.model(input_ids=ids, attention_mask=mask).logits[
                rows_index, queries
            ].float()
        finally:
            for handle in zero_handles:
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
            # The tap rides the knockout pass itself, so what it reports is the
            # head's antecedent mass *after* the intervention rather than
            # before it. Reading it on the clean pass, which is what this
            # function used to do, describes the instance and not the
            # intervention.
            residual: dict[str, torch.Tensor] = {}

            def consume_knocked(
                _layer: int,
                weights: torch.Tensor,
                _store=residual,
                _mask=key_mask,
                _head=head,
                _queries=queries,
            ) -> None:
                row_attention = weights.float()[
                    torch.arange(weights.shape[0], device=weights.device), :, _queries
                ]
                _store["mass"] = (row_attention * _mask.unsqueeze(1)).sum(dim=-1)[:, _head]

            handles = [
                *install_knockout(arm, layer, knock),
                *tap_attention(arm, layer, consume_knocked),
            ]
            try:
                logits = arm.model(input_ids=ids, attention_mask=mask).logits[
                    rows_index, queries
                ].float()
            finally:
                for handle in handles:
                    handle.remove()
            if "mass" not in residual:
                raise RuntimeError(
                    f"{arm.name}: layer {layer} was never entered under knockout, so "
                    "the intervention cannot be confirmed to have run"
                )
            knocked_mass[position, begin : begin + block.size] = (
                residual["mass"].cpu().numpy()
            )
            gap, probability = _margin_and_probability(logits, targets)
            delta_gap[position, begin : begin + block.size] = (
                (gap - clean_gap).cpu().numpy()
            )
            delta_probability[position, begin : begin + block.size] = (
                (probability - clean_p).cpu().numpy()
            )
            del knock, logits
        torch.cuda.empty_cache()

    residual_max = float(knocked_mass.max())
    if residual_max > residual_mass_tolerance:
        worst = int(np.unravel_index(int(knocked_mass.argmax()), knocked_mass.shape)[0])
        layer, head = heads[worst]
        raise RuntimeError(
            f"{arm.name}: after knocking out L{layer}H{head} it still places "
            f"{residual_max:.4g} attention mass on the antecedent keys, above the "
            f"{residual_mass_tolerance:.4g} tolerance. Adding {KNOCKOUT_LOGIT:.1e} "
            "pre-softmax drives the surviving weight to exactly zero, so a residual "
            "mass means the mask reached the wrong head or the wrong keys and every "
            "delta below is the effect of an intervention that did not happen"
        )
    return {
        "delta_m_gap": delta_gap,
        "delta_probability": delta_probability,
        "antecedent_attention_mass": antecedent_mass,
        # Named for when it was measured, because the pair is the control: the
        # clean mass says the head reads the antecedent, the knocked mass says
        # the knockout stopped it. Either alone is consistent with an inert
        # instrument.
        "knocked_antecedent_attention_mass": knocked_mass,
        "knockout_residual_mass_max": residual_max,
        "knockout_residual_mass_tolerance": float(residual_mass_tolerance),
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


#: Why a per-head interval is absent when it is absent. Stated once, so that a
#: null interval is never a null a reader has to explain to themselves.
_CLUSTER_REFUSAL = (
    "fewer sequences contributed instances than the shared percentile-bootstrap "
    "unit floor admits, so no interval is published; the point estimates above "
    "are still means over every instance"
)


def _cluster_interval(
    per_instance: np.ndarray,
    clusters: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any] | None:
    """Sequence-clustered percentile intervals for a per-instance statistic matrix.

    ``per_instance`` is ``(n_instances, n_statistics)`` and ``clusters`` names the
    sequence each instance came from. Instances are averaged within sequence and
    the sequences are resampled, which is the same unit every other interval in
    this module uses. ``None`` when there are too few sequences to bootstrap:
    refusing the interval is the right answer there and aborting the stage is
    not, since the point estimates remain valid.
    """

    if per_instance.ndim != 2 or clusters.shape != (per_instance.shape[0],):
        raise ValueError("per-instance statistics do not align with their clusters")
    unique = np.unique(clusters)
    means = np.stack(
        [per_instance[clusters == cluster].mean(axis=0) for cluster in unique]
    )
    weights = np.asarray(
        [float((clusters == cluster).sum()) for cluster in unique], dtype=np.float64
    )
    if bootstrap_unit_floor(int(unique.size))["degenerate"]:
        return None
    booted = cluster_bootstrap(means, weights, replicates=replicates, seed=seed)
    return {
        "n_clusters": int(unique.size),
        "intervals": [
            [_finite(float(low), "cluster CI low"), _finite(float(high), "cluster CI high")]
            for low, high in zip(booted["q_low"], booted["q_high"])
        ],
    }


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
    bootstrap_replicates: int = 1000,
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

    **Every number here is accompanied by a sequence-clustered interval.** The
    per-instance axis used to be collapsed with ``.mean()`` at the point of
    reporting, so the artefact held one number per head per alpha and could not
    be re-analysed at all -- the rule ``paa_attention_scores``' own docstring
    states, four hundred lines up, is that "a census that emits only pooled
    means is not re-analysable, which is a failure this programme has already
    recorded once", and ``cluster_bootstrap`` sits unused in the same file. The
    resampling unit is the sequence, because two instances from one document are
    not independent draws.
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

        taps = tap_attention(arm, layer, consume)
        try:
            per_alpha_mass = np.zeros((len(alphas), len(by_layer[layer]), selected.size))
            control_mass = np.zeros_like(per_alpha_mass)
            per_alpha_target = np.zeros((len(alphas), selected.size))
            per_alpha_substitute = np.zeros((len(alphas), selected.size))
            # The random control needs its own manipulation check. Recording
            # p(X) only under the prediction nudge means the artefact can say
            # that the prediction direction moved the head's antecedent mass and
            # that a random direction of identical norm moved it too, without
            # being able to say whether the random direction moved p(X) as well
            # -- and at alpha = 2 the perturbation is twice the residual norm,
            # which disrupts a great deal. Appendix B rule 5: an intervention
            # that moves everything needs a control for moving everything, and
            # the control has to be measured on the same axis as the treatment
            # or the comparison is not a comparison.
            control_target = np.zeros_like(per_alpha_target)
            control_substitute = np.zeros_like(per_alpha_substitute)
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
                        probabilities = F.softmax(
                            logits[rows_index, queries].float(), dim=-1
                        )
                        target_store, substitute_store = (
                            (per_alpha_target, per_alpha_substitute)
                            if label == "prediction"
                            else (control_target, control_substitute)
                        )
                        target_store[alpha_index, begin : begin + block.size] = (
                            probabilities[rows_index, targets].cpu().numpy()
                        )
                        substitute_store[alpha_index, begin : begin + block.size] = (
                            probabilities[rows_index, replacements].cpu().numpy()
                        )
                        del logits, pattern, probabilities
                torch.cuda.empty_cache()
        finally:
            for tap in taps:
                tap.remove()

        clusters = pool.sequence[selected]
        for head_index, head in enumerate(by_layer[layer]):
            # One bootstrap per head over a matrix whose columns are the alpha
            # sweep under both conditions, so the prediction column and its
            # control column are resampled on the same sequences and their
            # difference is a paired quantity rather than two marginals.
            columns = np.vstack(
                [per_alpha_mass[:, head_index, :], control_mass[:, head_index, :]]
            ).T
            booted = _cluster_interval(
                columns, clusters, replicates=bootstrap_replicates, seed=seed + layer
            )
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
                "antecedent_mass_ci_by_alpha": (
                    None if booted is None else booted["intervals"][: len(alphas)]
                ),
                "antecedent_mass_ci_by_alpha_random_control": (
                    None if booted is None else booted["intervals"][len(alphas) :]
                ),
                "cluster_bootstrap": (
                    {"available": False, "reason": _CLUSTER_REFUSAL}
                    if booted is None
                    else {
                        "available": True,
                        "unit": "sequence",
                        "n_clusters": booted["n_clusters"],
                        "replicates": int(bootstrap_replicates),
                    }
                ),
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
            "p_target_by_alpha_random_control": [
                _finite(float(control_target[index].mean()), "control p target")
                for index in range(len(alphas))
            ],
            "p_substitute_by_alpha_random_control": [
                _finite(float(control_substitute[index].mean()), "control p substitute")
                for index in range(len(alphas))
            ],
            "reading": (
                "the prediction nudge is specific only if it moves p(X) further "
                "than a random direction of identical norm does; the control rows "
                "are what make that comparison possible"
            ),
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
    """Prefix matching with and without a position-matched decoy subtraction.

    The induction headline this programme reports is an uncorrected attention
    weight.  If subtracting position-matched decoys moves it materially, that is
    a finding about the induction census and has to be read before any PAA
    number is.

    **This is not the identical correction ``build_instance_pool`` applies, and
    it cannot be made identical.**  That function's decoy eligibility carries a
    fourth condition this one does not: a decoy key must not hold one of the
    model's top-``ban`` predicted tokens at the query.  Adding it here would
    require the query-position logits, which this pass does not read -- but the
    reason not to add it is stronger than the cost.  **The ban is
    alphabet-pathological.**  Over a twenty-symbol residue alphabet a top-20 ban
    covers the alphabet, and it is already measured doing so: on ProGen2-medium it
    empties the decoy pool for **93.0%** of eligible positions, which is why
    ``--protein-ban-depths`` exists and why the matching gate reports a relaxed
    depth beside the specified one.  Applying it here would delete the protein
    side of this census to make a text-side definition match.

    So the two corrections share the position window, the self-exclusion, the
    same-token exclusion, the predecessor rule and the sink floor, and differ in
    the prediction ban.  The docstring previously claimed they were "the same
    decoy subtraction PAA uses" and an inline comment asserted the populations had
    to match; both were false, and the difference runs in the direction that makes
    *this* correction the more conservative one -- a decoy here may hold a
    plausible prediction, so its attention is subtracted as if it were positional
    baseline, and the corrected score is if anything over-subtracted.  Recorded as
    an accepted limitation rather than closed, because closing it costs an arm.
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
        handle
        for layer in range(layers)
        for handle in tap_attention(arm, layer, consume)
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
                    # Same sink exclusion as ``build_instance_pool``. The two
                    # populations agree on the window, the self- and same-token
                    # exclusions, the predecessor rule and this floor, and differ
                    # in that function's top-``ban`` prediction ban -- see this
                    # function's docstring for why that difference is accepted
                    # rather than closed.
                    window = np.arange(max(1, query - high), query - low + 1)
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
    minimum_clusters: int = MINIMUM_BOOTSTRAP_UNITS,
) -> dict[str, np.ndarray]:
    """Weighted mean over clusters with a percentile interval over cluster draws.

    ``per_cluster`` is ``(n_clusters, n_statistics)``.  Resampling clusters --
    sequences -- rather than instances is the whole point: instances inside one
    document are not independent, and an instance-level interval would be
    narrow for the wrong reason.

    The unit floor is the same one ``homology.bootstrap_stratum`` applies and is
    declared once, in ``statistics``.  This function guarded only ``n < 2``,
    which admits the entire range where the interval is anti-conservative: at
    three non-empty clusters a nominal 95% percentile interval realises 74%
    coverage and at four it realises 82%.  Refused rather than flagged, unlike
    the stratum bootstrap, because this function returns bare arrays with no
    room to carry a verdict -- every caller reads ``q_low``/``q_high`` straight
    into a report, so a degenerate marker beside them would be optional to
    notice.  The PAA census resamples hundreds of sequences and is nowhere near
    this floor; what the floor stops is a future caller narrowing a cohort until
    it is.

    **``alpha`` is guarded against ``replicates`` for the same reason.**  A
    percentile at ``alpha/2`` is read off ``replicates`` draws, so the tail must
    contain enough of them to be an estimate rather than an order statistic of the
    minimum.  The caller that motivates this is the census's Bonferroni column,
    ``alpha = 0.05 / n_heads``: at the 24 heads it has run, ``alpha/2`` lands at
    sorted index 1.04 of 1000 and the bound already carries a Monte-Carlo spread
    of 0.0028 logits across seeds, comparable to the smallest effect it publishes;
    at the 720 heads an exhaustive census would patch it lands at index 0.035, so
    the "lower bound" is the smallest draw and nothing else.  Refusing states the
    replicate count that would resolve the requested tail, because that is the
    parameter the caller can actually change.

    This **replaces** a flat ``replicates >= 100``, which is the same rule stated
    for one alpha and stated wrongly: at the default ``alpha = 0.05`` a hundred
    replicates put 2.5 draws below the lower percentile, so the constant admitted
    exactly what it was meant to exclude.  One rule, applied to the request the
    caller actually made.
    """

    if per_cluster.ndim != 2 or weights.ndim != 1:
        raise ValueError("cluster bootstrap expects a (cluster, statistic) matrix")
    if per_cluster.shape[0] != weights.shape[0]:
        raise ValueError("weights do not align with clusters")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}")
    keep = weights > 0
    values = per_cluster[keep]
    mass = weights[keep].astype(np.float64)
    floor = bootstrap_unit_floor(values.shape[0], minimum_units=minimum_clusters)
    if floor["degenerate"]:
        raise ValueError(f"cluster bootstrap refused: {floor['degenerate_reason']}")
    # After the unit floor, because a caller who has both problems has the data
    # problem first and the replicate count is the cheaper one to fix.
    draws_in_tail = alpha / 2.0 * replicates
    if draws_in_tail < MINIMUM_DRAWS_IN_TAIL:
        raise ValueError(
            f"alpha={alpha:.3e} puts {draws_in_tail:.2f} of {replicates} draws below "
            f"the lower percentile, under a floor of {MINIMUM_DRAWS_IN_TAIL:.0f}; the "
            "bound would be an order statistic of the extreme draws rather than an "
            "estimate. Use at least "
            f"{int(math.ceil(MINIMUM_DRAWS_IN_TAIL * 2.0 / alpha))} replicates, or a "
            "less extreme alpha"
        )
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
    """Change in ``log p(X)`` at ``q`` when the antecedent tokens are replaced.

    The replacement is drawn from the arm's own unigram distribution, so the
    corrupted context stays inside the model's token distribution and the effect
    measured is the loss of the antecedent rather than the arrival of an
    impossible symbol.

    **Every earlier occurrence is replaced, not just the nearest one, because
    the statistic this gate matches removes every earlier occurrence.**
    ``knockout_effects`` stops a head reading the keys ``antecedent_sets``
    returns, which is all of them; ``coarsened_cells`` then bins instances on
    the corruption effect and calls the result a matched cell. Replacing one
    occurrence out of many measures almost nothing whenever the token recurs,
    and how often it recurs is a property of the *alphabet*: over twenty amino
    acids, X occurs dozens of times in a 512-token protein row, while in English
    X is often a once-occurring content word. So the single-token gate assigned
    nearly every protein instance to the "no effect" bin and spread the text
    instances across the range, and the matching then reported that the arms
    cannot be matched -- a conclusion the estimator manufactured out of
    alphabet size rather than one it measured. This lands directly on the L5
    deliverable, so it is corrected rather than caveated.

    Each occurrence draws its own replacement, so the corrupted row is not a run
    of one repeated symbol -- which would itself be a pattern the model can read.
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
        key_sets = antecedent_sets(rows, pool, block)
        for row, index in enumerate(block):
            original = int(pool.predicted_token[index])
            for position in key_sets[row]:
                for _ in range(64):
                    candidate = int(support[rng.choice(support.size, p=probabilities)])
                    if candidate != original:
                        corrupted[row, position] = candidate
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


#: Census-rank strata the D2.c comparison is published on. The top 5% and the
#: bulk 80% are the cuts EXP-R2-101 decomposed the modality gap into; 20% is the
#: middle cut it also quotes. Swept rather than reported at one cut, per standing
#: rule 17.
CENSUS_RANK_FRACTIONS = (0.05, 0.20, 0.80)

#: Floor on a census-rank stratum. 'The top 1% of heads' is one head on gpt2 and
#: seven on ProtGPT2, so a per-arm proportion is not a comparable population
#: (standing rule 21); below this a stratum correlation is not published.
MINIMUM_STRATUM_HEADS = 8

#: Depth of the retrieval statistic. ``hit@20`` has a grid-dependent ceiling of
#: ``n/20`` and is comparable only within a grid size, which the record states.
CENSUS_RETRIEVAL_K = 20


def census_head_scores(matched_per_sequence: np.ndarray) -> np.ndarray:
    """One census score per head, reduced from the per-sequence matrix.

    **The reduction is the ambiguous step and that is why it lives here.** The
    census stage emits ``paa_specific_matched`` per *sequence*; the causal stage
    emits ``delta_m_gap`` per head; nothing in this repository joined them, so
    every published D2.c correlation was produced by throwaway code -- the
    provenance defect that cost this programme two retracted figures and that
    ``path_patching.causal_census_agreement`` was written to close on the other
    mechanism.  It is not a formality: on the rebuilt ProtGPT2 pool the
    unweighted mean gives rho +0.1806, an instance-weighted mean +0.2017 and a
    per-sequence median +0.2270, while gpt2-large's three agree to 0.005.  The
    ambiguity is largest on the arm that carries the modality claim.

    The unweighted mean over sequences is the convention, for the reason the
    sequence is the bootstrap cluster everywhere else in this module: two
    instances from one record are not independent draws, so weighting a sequence
    by how many instances it happened to contribute would let a long record
    outvote a short one.  It is also, as a matter of fact rather than of taste,
    the reduction the census artefact already publishes -- it reproduces
    ``knockout_matched_score.distribution`` exactly, which is how it was
    identified.
    """

    array = np.asarray(matched_per_sequence, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] < 1:
        raise ValueError(
            "matched score must be (sequences, layers, heads); got "
            f"{array.shape}"
        )
    return np.nanmean(array, axis=0)


def census_causal_agreement(
    matched_per_sequence: np.ndarray,
    per_head: Sequence[Mapping[str, Any]],
    *,
    fractions: Sequence[float] = CENSUS_RANK_FRACTIONS,
    retrieval_k: int = CENSUS_RETRIEVAL_K,
) -> dict[str, Any]:
    """Does the PAA census score order causal importance?  D2.c's statistic.

    The mirror of ``path_patching.causal_census_agreement`` for the second
    mechanism, and it carries the same two preconditions because the same two
    failures are available.  The head set must be the whole grid, or the
    correlation is computed over the heads the census itself selected and
    measures nothing (standing rule 24).  And the all-grid figure is published
    beside its depth control, because the causal readout here is depth-biased for
    the same reason it is in path patching -- the knockout and the M-gap both act
    at the query row -- so an all-grid rank correlation between it and a census
    score confounds the two (EXP-R2-120, EXP-R2-117).

    **Control-flagged heads are kept, and that is a decision the data forces.**
    In an exhaustive configuration the flag is a labelling artefact rather than a
    different population: ``--causal-heads 712 --control-heads 8
    --control-offset 712`` partitions a 36x20 grid into 712 plus 8 with no
    overlap, and the eight sit at census ranks 710 and 713-719 -- the bottom of
    the very ranking being correlated.  Dropping them truncates one end of it.
    Every historical D2.c figure was computed over the 712 and is therefore
    slightly different from what this function returns; the difference is
    measured rather than assumed, and is +0.0180 on ProtGPT2 (+0.1806 over the 712,
    +0.1987 over the grid) and +0.0070 on gpt2-large (+0.4425 against +0.4495) at
    n=600 / ban 3 / draw 20260728.  What the precondition actually
    forbids is a *census-selected* head set, so it is checked on the union.

    The census-rank strata are the reason this is not a single number.  On both
    mechanisms the all-grid statistic is dominated by the heads a census never
    reports, and EXP-R2-101 found the modality gap reversing inside every
    stratum while the aggregate separated -- an aggregate can separate on
    heterogeneity alone.  ``retrieval`` is the statistic that survived that: how
    much of the causal top-k the census top-k actually recovers, which is
    selection-free and has a computable chance level.
    """

    score_grid = census_head_scores(matched_per_sequence)
    n_layer, n_head_ = score_grid.shape
    records = list(per_head)
    covered = {(int(row["layer"]), int(row["head_index"])) for row in records}
    if len(covered) != n_layer * n_head_ or len(records) != len(covered):
        raise ValueError(
            "census-to-causal agreement needs a causal effect for every head in "
            f"the grid exactly once: {len(records)} records covering "
            f"{len(covered)} distinct heads against a {n_layer}x{n_head_} grid of "
            f"{n_layer * n_head_}. A correlation over a census-selected subset "
            "reproduces the census's own ranking and measures nothing "
            "(standing rule 24)"
        )
    census = np.empty(len(records), dtype=np.float64)
    magnitude = np.empty(len(records), dtype=np.float64)
    layer = np.empty(len(records), dtype=np.float64)
    for index, row in enumerate(records):
        census[index] = score_grid[int(row["layer"]), int(row["head_index"])]
        magnitude[index] = abs(float(row["delta_m_gap"]))
        layer[index] = float(row["layer"])

    order = np.argsort(census)[::-1]
    strata: dict[str, Any] = {}
    for fraction in fractions:
        count = max(int(round(fraction * census.size)), MINIMUM_STRATUM_HEADS)
        if count > census.size:
            continue
        top = order[:count]
        strata[f"top_{fraction:g}"] = {
            "n_heads": int(count),
            "fraction_requested": float(fraction),
            "spearman": _finite(
                float(stats.spearmanr(census[top], magnitude[top]).statistic),
                "stratum spearman",
            ),
            "floored_to_minimum": bool(count > round(fraction * census.size)),
        }
    bulk = order[max(int(round(fractions[0] * census.size)), MINIMUM_STRATUM_HEADS) :]

    k = min(int(retrieval_k), census.size)
    causal_top = set(int(i) for i in np.argsort(magnitude)[::-1][:k])
    census_top = set(int(i) for i in order[:k])
    return {
        "n_heads": int(census.size),
        "n_control_flagged_heads_kept": int(
            sum(1 for row in records if row.get("is_control", False))
        ),
        "census_score": "paa_specific_matched, unweighted mean over sequences",
        "spearman_census_vs_causal_magnitude": _finite(
            float(stats.spearmanr(census, magnitude).statistic), "all heads"
        ),
        "depth_controlled": depth_controlled_rank_correlation(
            census, magnitude, layer
        ),
        "rank_split": {
            "split_by": "census_rank",
            "strata": strata,
            "bulk_spearman": _finite(
                float(stats.spearmanr(census[bulk], magnitude[bulk]).statistic),
                "bulk spearman",
            ),
            "note": (
                "split by census rank rather than by causal magnitude, because "
                "splitting on the outcome selects the heads whose effect is large"
            ),
        },
        "retrieval": {
            "k": int(k),
            "hit_at_k": int(len(causal_top & census_top)),
            "chance": float(k * k / census.size),
            "ceiling": int(k),
            "note": (
                "hit@k has a grid-dependent ceiling and a grid-dependent chance "
                "level, so it is comparable only between arms of the same grid size"
            ),
        },
    }
