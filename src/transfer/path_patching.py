"""Path patching: is an induction head's effect on the logits written or mediated?

Three measurements on this panel agree that attention contributes to next-token
prediction more *indirectly* in protein decoders than in text: whole-pathway
ablation credits attention more than direct logit attribution does and the gap is
about three times larger in residue-level protein arms; ProtGPT2's
prefix-matching heads score at or below chance on the OV copying statistic while
GPT-2's copy directly; and activation patching recovers through ``resid_post``
what it cannot recover through ``attn_out`` or ``mlp_out`` separately.  None of
the three separates a direct write to the unembedding from an effect routed
through a later component, because none of them holds the rest of the model
fixed.  Path patching does (Wang, Variengien, Conmy, Shlegeris and Steinhardt,
"Interpretability in the Wild", ICLR 2023, Appendix B; Goldowsky-Dill, MacLeod,
Sato and Arora, "Localizing Model Behavior with Path Patching", arXiv:2304.05969),
and this module is that instrument for the induction heads the census in
:mod:`.circuits` identifies.

Everything below is fixed before any result is read.

**The clean/corrupted pair.**  A case is one (query, key) pair of a natural
repeat probe from :func:`~.circuits.natural_repeat_probes`.  ``q`` is a token in
the second copy; ``k`` is the token that *followed* the aligned earlier position,
which is what a prefix-matching head attends to and what its OV circuit would
copy.  The clean input is the probe truncated to ``[0, q]``; the corrupted input
is identical except that the token at ``k`` is replaced by one other token drawn
from the arm's own unigram support.  Exactly one token differs, and it is the
token the induction mechanism would copy.  This is the same counterfactual shape
as the IOI name swap: the corruption changes *what the circuit should output*
while leaving the rest of the context alone.

**The metric.**  ``L = logit_q[T] - logit_q[T']`` where ``T`` is the clean token
at ``k`` and ``T'`` the corrupted one.  Both tokens are fixed by the pair
construction, never by a model output, so the metric is a genuine
difference-in-differences.  Note that ``T`` need not be the true next token at
``q``: under an approximate-repeat criterion the second copy is a diverged copy,
and the quantity of interest is what the *induction mechanism* promotes, not
whether the sequence happens to be self-consistent there.

**The direction.**  Denoising.  Every patched run starts from the corrupted
input, and restoring an activation to its clean value is what recovers the
metric.  ``recovery(x) = (L_x - L_corrupt) / (L_clean - L_corrupt)``, so
restoring nothing recovers 0 and restoring the whole residual stream at the final
layer recovers 1.  Both are checked by :func:`structural_invariants` and both
must hold or the run fails.

**Eligibility.**  A case enters the average only if ``L_clean - L_corrupt >=
minimum_effect``.  The ratio is otherwise a ratio of noise, and a *signed* floor
rather than an absolute one is used because a case where corrupting the copied
token *raises* the metric is a case where the induction mechanism is not doing
the thing being measured; averaging it in would flip the sign of every recovery
it contributes.  The exclusion rate is reported per arm and is itself a
measurement.

**The sender node.**  One attention head at the read-out position ``q``, patched
at the input of the attention output projection so that exactly that head's write
to the residual stream changes and every other head in the layer is untouched.
Restricting the sender to position ``q`` is deliberate: the read-out is the
next-token logit at ``q``, MLPs are position-wise, so the sender-to-MLP-to-logits
path is position-local, and admitting the sender's writes at other positions
would inflate the mediated share with contributions that can only reach the
read-out through attention.  The cost is that attention-mediated effects sourced
at other positions are outside the measurement, which is recorded rather than
repaired.

**What is held fixed.**  A path-patching run that lets the whole model recompute
is an ablation with extra steps.  For a sender at layer ``L``:

``direct``    every attention module above ``L`` and every MLP from ``L`` on is
              frozen at its corrupted-run value, so the sender's change reaches
              the unembedding through the residual stream and nothing else;
``via_mlp``   attention above ``L`` is frozen, MLPs recompute;
``via_attn``  MLPs from ``L`` on are frozen, attention recomputes;
``total``     nothing is frozen.

``mlp_mediated = via_mlp - direct``, ``attn_mediated = via_attn - direct``, and
``interaction = total - direct - mlp_mediated - attn_mediated`` collects the
higher-order mass that no single-mediator path carries.  Freezing a component
that is not downstream of the sender is a no-op by construction, which is why the
same rule serves GPT-2's sequential block and ProGen2's parallel (GPT-J) block
where the layer-``L`` MLP does not read the layer-``L`` attention output.

**Position locality.**  Because the only perturbation is the sender's write at
``q``, and the model is causally masked and its MLPs are position-wise, every
position other than ``q`` computes identically in the base and patched runs.
Freezing and patching therefore touch column ``q`` only.  That is a theorem, not
an assumption, but an implementation can still break it, so
:func:`structural_invariants` checks it against a freeze applied at every
position.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy import stats

from .arms import Arm
from .circuits import (
    RepeatProbe,
    Unigram,
    final_norm,
    head_dim,
    head_ov_weights,
    inner_decoder,
    n_head,
)
from .statistics import (
    MINIMUM_BOOTSTRAP_UNITS,
    bootstrap_unit_floor,
    depth_controlled_rank_correlation,
)

SCHEMA_VERSION = "r2_transfer_path_patching_v1"

#: The four freezing regimes, in the order they are reported.  ``direct`` is
#: first because every mediated quantity is defined against it.
PATHWAYS: tuple[str, ...] = ("direct", "via_mlp", "via_attn", "total")

#: Derived per-head quantities, all on the same recovered-logit-difference scale.
EFFECTS: tuple[str, ...] = (
    "direct",
    "mlp_mediated",
    "attn_mediated",
    "interaction",
    "mediated",
    "total",
)

#: Minimum ``L_clean - L_corrupt`` in logits for a case to enter an average.
#: Matched to the *value* :func:`~.circuits.activation_patching` already uses; the
#: rule differs and is stated in the module docstring. That module admits a case
#: on ``|denominator| >= floor`` over a top-1-minus-rank-2 metric, this one on a
#: signed ``denominator >= floor`` over a clean-minus-corrupt-token metric, so the
#: two share a number rather than an eligibility rule.
DEFAULT_MINIMUM_EFFECT = 0.25

#: A sender head whose mean total recovery is smaller than this in absolute value
#: is left out of the *fraction* summaries, because ``direct / total`` is not a
#: meaningful decomposition of an effect that is indistinguishable from zero.
#: The head is still reported individually and still enters the effect-scale
#: means; only the ratio is withheld, and the count is reported.
DEFAULT_MIN_HEAD_EFFECT = 0.01

#: The floor is a judgement call, and where a judgement call lands can move a
#: fraction: a high floor keeps only the strongest heads, which are not a random
#: sample of an arm's heads.  Every summary therefore carries the fraction at each
#: of these floors as well as at the declared one, so that the sensitivity is in
#: the artefact rather than in whichever floor happened to be chosen.
MIN_HEAD_EFFECT_LADDER: tuple[float, ...] = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)

#: How small a case's head write may be, as a fraction of the batch's median
#: head write, before its *relative* linearity error stops meaning anything.
#:
#: Relative to the batch rather than absolute, because the quantity it bounds
#: has no scale of its own: residual norms differ by orders of magnitude between
#: a 768-wide and a 2560-wide arm, and between bfloat16 and float32, so any
#: absolute norm that admits every case on one arm excludes every case on
#: another. The floor that was here before -- ``max(scale, 1e-6)`` applied to
#: the *denominator* rather than to the row -- did the opposite of excluding: it
#: rescued a vanishing denominator by replacing it, so an arm whose sender
#: writes almost nothing had every absolute discrepancy divided by 1e-6 and
#: reported as a small relative error.
#:
#: A row a thousand times below its batch's median write is scored against its
#: peers' arithmetic noise, not against its own signal, so it is excluded from
#: the maximum and counted. It cannot exclude a whole batch: the median row
#: always clears a fraction of the median.
_HEAD_WRITE_RELATIVE_FLOOR = 1e-3


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _leading(output: Any) -> torch.Tensor:
    """The ``[batch, token, d_model]`` tensor a block or sublayer returns."""

    tensor = output[0] if isinstance(output, tuple) else output
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
        raise TypeError("expected a [batch, token, d_model] module output")
    return tensor


# ----------------------------------------------------------------- sender set


@dataclass(frozen=True)
class SenderHead:
    """One attention head admitted as a path-patching sender."""

    layer: int
    head: int
    prefix_matching: float
    above_threshold: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer": int(self.layer),
            "head": int(self.head),
            "prefix_matching": _finite(self.prefix_matching, "sender prefix matching"),
            "above_threshold": bool(self.above_threshold),
        }

    @property
    def label(self) -> str:
        return f"L{self.layer}H{self.head}"


def select_senders(
    prefix_matching: np.ndarray,
    *,
    threshold: float,
    fallback_top_k: int,
    max_senders: int | None = None,
    exhaustive: bool = False,
) -> tuple[list[SenderHead], dict[str, Any]]:
    """Heads above the census threshold, or an explicitly flagged top-k if none are.

    Head counts differ across the panel by an order of magnitude and the
    difference is the disputed quantity, so the selection rule has to be one rule
    applied to every arm.  Lowering the threshold for the one arm that has no
    head above it would be exactly the silent per-arm tuning this measurement is
    supposed to adjudicate.  ZymCTRL has zero heads above 0.10 under both repeat
    criteria; it therefore enters on its ``fallback_top_k`` highest-scoring heads
    with ``above_threshold`` false on every one of them and
    ``not_induction_heads_by_panel_criterion`` set on the provenance record, so
    no reader can quote its numbers as induction-head numbers.

    ``exhaustive`` admits **every** head, still ranked by prefix matching.  It
    exists because the two selection rules above make one question unanswerable:
    both admit only heads the census already picked, so any agreement statistic
    computed between the causal ranking and the census is agreement between a set
    and itself.  Audit item D2.b asked for a top-20 Jaccard against the census
    and got 1.0 on all four arms by construction, which carries no information.
    Seeing the census's *misses* requires a causal effect for heads the census
    did not select, and that requires patching heads it scored below threshold.
    This is a far more expensive measurement -- every head rather than the tail --
    and it is opt-in for that reason, not a default.

    ``max_senders`` is refused together with ``exhaustive``, because the two
    compose into exactly the artefact the exhaustive criterion was written to
    make impossible.  Truncation keeps the highest-scoring heads, so an
    exhaustive set cut to its first ``max_senders`` rows is a census-selected set
    carrying the ``exhaustive_all_heads`` label: every head still in it is one the
    census ranked at the top, the causal top-k is drawn from the census's own
    choice again, and ``n_senders_below_threshold`` comes back 0 with nothing in
    the provenance record contradicting the criterion.  That is the D2.b
    circularity restored under a different flag (Appendix B rule 24).
    """

    if prefix_matching.ndim != 2:
        raise ValueError("prefix-matching scores must be a (layer, head) matrix")
    if not np.isfinite(prefix_matching).all():
        raise ValueError("prefix-matching scores contain non-finite values")
    if fallback_top_k < 1:
        raise ValueError("fallback_top_k must be positive")
    if max_senders is not None and max_senders < 1:
        raise ValueError("max_senders must be positive when given")
    if max_senders is not None and exhaustive:
        raise ValueError(
            "max_senders and exhaustive are mutually exclusive: truncation keeps "
            "the highest-scoring heads, so an exhaustive set cut to max_senders "
            "rows is a census-selected set wearing the exhaustive label, and the "
            "causal/census agreement computed on it is circular again (audit "
            "D2.b, Appendix B rule 24)"
        )

    order = np.argsort(prefix_matching, axis=None)[::-1]
    ranked = [
        (int(layer), int(head), float(prefix_matching[layer, head]))
        for layer, head in (
            np.unravel_index(int(flat), prefix_matching.shape) for flat in order
        )
    ]
    n_above = int((prefix_matching >= threshold).sum())
    if exhaustive:
        chosen = ranked
        criterion = "exhaustive_all_heads"
    elif n_above > 0:
        chosen = [row for row in ranked if row[2] >= threshold]
        criterion = "prefix_matching_above_threshold"
    else:
        chosen = ranked[:fallback_top_k]
        criterion = "top_k_no_head_above_threshold"
    truncated = False
    if max_senders is not None and len(chosen) > max_senders:
        chosen = chosen[:max_senders]
        truncated = True
    # ``above_threshold`` is each head's own score against the threshold. Under
    # the two selective criteria that is identical to the set-level answer -- they
    # admit either only heads at or above it, or, when none exist, only heads
    # below it -- so this is not a behaviour change on those paths. Under
    # ``exhaustive`` the set spans both sides of the threshold and only a per-head
    # answer is true, which is the whole point of the criterion: it is the field
    # that tells a reader which causally-ranked heads the census would have missed.
    senders = [
        SenderHead(
            layer=layer,
            head=head,
            prefix_matching=score,
            above_threshold=score >= threshold,
        )
        for layer, head, score in chosen
    ]
    if not senders:
        raise RuntimeError("sender selection produced no heads")
    provenance = {
        "criterion": criterion,
        # The size of the head grid this set was drawn from. Published because
        # ``causal_census_agreement`` needs it: its precondition is that every
        # head in the grid carries a causal effect, and that is a fact about the
        # data, not a flag the caller asserts.
        "n_heads_in_grid": int(prefix_matching.size),
        "threshold": float(threshold),
        "n_above_threshold": n_above,
        "n_senders": len(senders),
        "fallback_top_k": int(fallback_top_k),
        "max_senders": None if max_senders is None else int(max_senders),
        "truncated_to_max_senders": truncated,
        "not_induction_heads_by_panel_criterion": n_above == 0,
        "exhaustive": bool(exhaustive),
        "n_senders_above_threshold": sum(1 for s in senders if s.above_threshold),
        "n_senders_below_threshold": sum(1 for s in senders if not s.above_threshold),
        "heads": [sender.as_dict() for sender in senders],
    }
    return senders, provenance


#: The selection criterion under which two sender sets carry no stability
#: information, because both are the whole head grid and their overlap is 1.0
#: whatever the census did.
EXHAUSTIVE_CRITERION = "exhaustive_all_heads"


def sender_set_overlap(
    left: Sequence[SenderHead],
    right: Sequence[SenderHead],
    *,
    left_criterion: str,
    right_criterion: str,
) -> dict[str, Any]:
    """How much two sender sets agree, and whether the comparison could have failed.

    Under the two selective criteria this is a real check: the exact and the
    approximate census pick heads independently, and a low Jaccard would mean the
    sender set is a property of the repeat criterion rather than of the model.

    Under ``exhaustive_all_heads`` it is not a check at all.  Both sets are the
    entire head grid, so the intersection is the union and the Jaccard is 1.0 by
    construction -- which is what all six arms of EXP-R2-071 published, under a
    docstring that said "stability is shown rather than assumed".  It was
    assumed.  The criteria are therefore taken as arguments and the triviality is
    stated in the record rather than left for a reader to infer from a number
    that cannot come out any other way (Appendix B rule 24).
    """

    for label, criterion in (("left", left_criterion), ("right", right_criterion)):
        if not criterion:
            raise ValueError(f"the {label} sender set must declare its criterion")
    a = {(s.layer, s.head) for s in left}
    b = {(s.layer, s.head) for s in right}
    union = a | b
    if not union:
        raise ValueError("cannot compare two empty sender sets")
    top = 4
    trivial = left_criterion == right_criterion == EXHAUSTIVE_CRITERION
    return {
        "n_left": len(a),
        "n_right": len(b),
        "n_intersection": len(a & b),
        "jaccard": _finite(len(a & b) / len(union), "sender jaccard"),
        "left_criterion": str(left_criterion),
        "right_criterion": str(right_criterion),
        "comparison_is_trivial": bool(trivial),
        "stability_verdict": (
            "not a stability check: both sets are the whole head grid under the "
            "exhaustive criterion, so the Jaccard is 1.0 whatever the two censuses "
            "did and no value other than 1.0 is reachable"
            if trivial
            else "the two censuses selected these sets independently, so a Jaccard "
            "below 1.0 was reachable and the value carries information"
        ),
        "left_top_heads": [s.label for s in list(left)[:top]],
        "right_top_heads": [s.label for s in list(right)[:top]],
        "top_heads_identical_as_set": {s.label for s in list(left)[:top]}
        == {s.label for s in list(right)[:top]},
    }


#: Top-k cuts the causal/census agreement is reported at. Audit item D2.b named
#: k=20; it is swept rather than reported alone because a single k is a threshold
#: and standing rule 8 asks for the ordering to be shown invariant across one.
#:
#: 32 is in the ladder because EXP-R2-071's retraction of its own first reading
#: turns on the rank correlation split at k=32, and a published number that no
#: versioned code path produces is the defect that retraction was written to
#: correct. One ladder serves both the Jaccard and the rank split, so the cut a
#: number is quoted at cannot drift between them.
CAUSAL_AGREEMENT_TOP_K = (5, 10, 20, 32, 40)

def _spearman(
    census_score: np.ndarray, magnitude: np.ndarray, label: str
) -> dict[str, Any]:
    """Spearman rho, or the reason there is none, in one shape.

    Two points always correlate at +/-1 and a constant vector has no ranking at
    all; ``scipy`` answers the first with 1.0 and the second with NaN, and both
    would be published as a correlation.  Both are reachable here: splitting the
    grid by census rank leaves a remainder that a large k can empty, and an arm
    whose census scores are tied across the tail has a constant side.
    """

    if census_score.size != magnitude.size:
        raise ValueError("census scores and causal magnitudes do not align")
    if census_score.size < 3:
        return {
            "rho": None,
            "p_value": None,
            "n": int(census_score.size),
            "withheld_reason": (
                "fewer than three heads: a rank correlation over two points is "
                "+1 or -1 by construction and over fewer is undefined"
            ),
        }
    if float(np.ptp(census_score)) == 0.0 or float(np.ptp(magnitude)) == 0.0:
        return {
            "rho": None,
            "p_value": None,
            "n": int(census_score.size),
            "withheld_reason": (
                "one side is constant across these heads, so it carries no "
                "ordering for the other to agree with"
            ),
        }
    rho, p_value = stats.spearmanr(census_score, magnitude)
    return {
        "rho": _finite(float(rho), f"{label} rho"),
        "p_value": _finite(float(p_value), f"{label} p"),
        "n": int(census_score.size),
    }


def causal_census_agreement(
    per_head: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    n_heads_in_grid: int,
    effect_key: str = "total",
    top_k: Sequence[int] = CAUSAL_AGREEMENT_TOP_K,
) -> dict[str, Any]:
    """Does the causal ranking recover the census ranking, and what does it add?

    This refuses a sender set smaller than the head grid, and the refusal is the
    point.  Audit item D2.b asked for a top-20 Jaccard between the causal ranking
    and the prefix-matching census, and the answer came back 1.0 on all four arms
    -- not because the two agree, but because the sender set *was* the
    census-selected set, so both rankings ranked the same heads and no head the
    census rejected could appear.  Computing this statistic on a selective set
    produces a number that looks like agreement and measures nothing.  The
    precondition is enforced here rather than documented, because the circularity
    is invisible in the output: a Jaccard of 1.0 is exactly what a real agreement
    would also give.

    **The precondition is derived from the data, not accepted from the caller.**
    It used to be a boolean the caller passed, and a boolean is only as true as
    the argument beside it: ``select_senders(..., max_senders=20,
    exhaustive=True)`` truncated the grid to the twenty highest-scoring heads,
    left ``exhaustive`` set, and this function then certified a top-20 Jaccard of
    1.0 over a census-selected set with the guard silent -- bit for bit the D2.b
    artefact.  ``n_heads_in_grid`` is a fact about the model, published by
    ``select_senders`` in its provenance record, and the check is that every head
    in that grid carries a causal effect (Appendix B rule 24).

    The primary statistic is the rank correlation over **all** heads, which needs
    no cut.  The top-k agreement is reported beside it, swept over k, because that
    is the form the gate was written in, and at each k the same rank correlation
    is reported *split* by census rank: over the top-k heads and over the
    remainder.  The split is what EXP-R2-071 had to retract its first reading on
    -- an all-grid rho near zero on the protein arms turned out to be an unordered
    bulk sitting under a top-32 rho of +0.79 to +0.81, higher than either text
    control -- and it had no implementation in this repository until now.
    """

    if n_heads_in_grid < 1:
        raise ValueError("n_heads_in_grid must be positive")
    if not per_head:
        raise ValueError("no per-head results to compare against the census")
    if len(per_head) != n_heads_in_grid:
        raise ValueError(
            "causal/census agreement requires a causal effect for every head in "
            f"the grid: {len(per_head)} of {n_heads_in_grid} heads were patched. "
            "On any smaller set the heads present are the ones a census-derived "
            "rule already chose, so the top-k Jaccard is 1.0 by construction and "
            "the census's misses cannot appear (audit D2.b, Appendix B rule 24)"
        )
    if not top_k or any(k < 1 for k in top_k):
        raise ValueError("top_k cuts must all be positive")

    labels = [str(row["label"]) for row in per_head]
    if len(set(labels)) != len(labels):
        raise ValueError("per-head records contain a duplicate head label")
    n_without_logits = sum(1 for row in per_head if "effects_logits" not in row)
    if n_without_logits:
        raise ValueError(
            "every per-head record must carry effects_logits: the recovery-scale "
            "effect is divided by that arm's own L_clean - L_corrupt, whose mean "
            "differs across this panel by more than an order of magnitude, so a "
            "magnitude read off it is not a magnitude another arm can be compared "
            f"against. {n_without_logits} of {len(per_head)} records carry only "
            "effects"
        )
    n_without_layer = sum(1 for row in per_head if "layer" not in row)
    if n_without_layer:
        raise ValueError(
            "every per-head record must carry layer: the all-grid rank correlation "
            "is taken over a flattened (layer, head) grid against a causal readout "
            "that is depth-biased by construction, and on this panel one arm's "
            "census score falls with depth while every other arm's rises, which is "
            "enough to drive its all-grid value to zero on its own. The correlation "
            f"cannot be read without the covariate. {n_without_layer} of "
            f"{len(per_head)} records carry no layer"
        )
    census_score = np.asarray(
        [float(row["prefix_matching"]) for row in per_head], dtype=np.float64
    )
    effect = np.asarray([float(row["effects"][effect_key]) for row in per_head], dtype=np.float64)
    effect_logits = np.asarray(
        [float(row["effects_logits"][effect_key]) for row in per_head], dtype=np.float64
    )
    if (
        not np.isfinite(census_score).all()
        or not np.isfinite(effect).all()
        or not np.isfinite(effect_logits).all()
    ):
        raise ValueError("census scores and causal effects must all be finite")
    magnitude = np.abs(effect)

    above = census_score >= threshold
    census_order = np.argsort(-census_score, kind="stable")
    causal_order = np.argsort(-magnitude, kind="stable")

    cuts: dict[str, Any] = {}
    for k in top_k:
        cut = min(int(k), len(per_head))
        top_index = census_order[:cut]
        remaining_index = census_order[cut:]
        census_top = {labels[i] for i in top_index}
        causal_top = [labels[i] for i in causal_order[:cut]]
        missed = [labels[i] for i in causal_order[:cut] if not above[i]]
        union = census_top | set(causal_top)
        cuts[repr(int(k))] = {
            "k": int(k),
            "k_effective": cut,
            "truncated_to_head_count": cut != int(k),
            "n_intersection": len(census_top & set(causal_top)),
            "jaccard": _finite(len(census_top & set(causal_top)) / len(union), "causal jaccard"),
            "causal_top_heads": causal_top,
            "n_causal_top_below_census_threshold": len(missed),
            "causal_top_heads_below_census_threshold": missed,
            # Split by *census* rank, not by causal rank: the question is whether
            # the census orders the heads it itself ranks highest, and splitting on
            # the causal magnitude would condition on the outcome.
            "n_remaining": int(remaining_index.size),
            "spearman_top_k": _spearman(
                census_score[top_index], magnitude[top_index], f"top-{cut}"
            ),
            "spearman_remaining": _spearman(
                census_score[remaining_index],
                magnitude[remaining_index],
                f"below-top-{cut}",
            ),
        }
    if len(cuts) != len(top_k):
        raise ValueError("two top-k cuts collided on one key")

    strongest_missed: dict[str, Any] | None = None
    for rank, index in enumerate(causal_order):
        if not above[index]:
            strongest_missed = {
                "label": labels[index],
                "causal_rank": rank,
                "prefix_matching": _finite(float(census_score[index]), "missed head score"),
                f"effect_{effect_key}": _finite(float(effect[index]), "missed head effect"),
                # The figure a cross-arm sentence has to be built on. EXP-R2-071
                # ranked this head's magnitude across four arms on the recovery
                # scale, where each arm divides by its own denominator; on the
                # logit scale the ordering it reported reverses.
                f"effect_{effect_key}_logits": _finite(
                    float(effect_logits[index]), "missed head logit effect"
                ),
            }
            break

    return {
        "effect_key": effect_key,
        "census_threshold": float(threshold),
        "n_heads": len(per_head),
        "n_heads_in_grid": int(n_heads_in_grid),
        "n_above_census_threshold": int(above.sum()),
        "n_below_census_threshold": int((~above).sum()),
        "spearman_census_vs_causal_magnitude": _spearman(
            census_score, magnitude, "all heads"
        ),
        # Published beside the correlation it qualifies rather than in analysis
        # code, because the correlation cannot be read without it: the causal
        # readout is depth-biased by construction and one arm's census score runs
        # the other way (EXP-R2-120, and see `depth_controlled_rank_correlation`).
        "depth_controlled": depth_controlled_rank_correlation(
            census_score,
            magnitude,
            np.asarray([float(row["layer"]) for row in per_head], dtype=np.float64),
        ),
        "rank_split": {
            "split_by": "census_rank",
            "cuts": [int(k) for k in top_k],
            "note": (
                "top_k[k].spearman_top_k and top_k[k].spearman_remaining are the "
                "all-head rank correlation restricted to the k heads the census "
                "scores highest and to the rest of the grid. The split is by census "
                "rank rather than by causal magnitude because splitting on the "
                "outcome would select the heads whose effect is large. It is swept "
                "over the same ladder as the Jaccard (Appendix B rule 17)"
            ),
        },
        "top_k": cuts,
        "strongest_head_below_census_threshold": strongest_missed,
        "effect_scale_note": (
            f"effect_{effect_key} is on the recovery scale: each case is divided by "
            "its own L_clean - L_corrupt before averaging, so the scale is a "
            f"property of this arm and this case set. effect_{effect_key}_logits is "
            "the same quantity un-normalised. Only the logit figure is comparable "
            "across arms; the eligible denominators of this panel span more than an "
            "order of magnitude and a magnitude ordering taken on the recovery scale "
            "can reverse on the logit scale"
        ),
    }


# --------------------------------------------------------------------- cases


@dataclass(frozen=True)
class PathCase:
    """One clean/corrupted pair, truncated so that the read-out is the last token."""

    input_ids: tuple[int, ...]
    corrupt_ids: tuple[int, ...]
    position_q: int
    position_k: int
    token_clean: int
    token_corrupt: int
    probe_index: int
    probe_kind: str

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.corrupt_ids):
            raise ValueError("clean and corrupted inputs must have the same length")
        if self.position_q != len(self.input_ids) - 1:
            raise ValueError("a case must be truncated so that q is the last token")
        if not 0 < self.position_k < self.position_q:
            raise ValueError("a case requires 0 < k < q")
        differing = [
            index
            for index, (left, right) in enumerate(zip(self.input_ids, self.corrupt_ids))
            if left != right
        ]
        if differing != [self.position_k]:
            raise ValueError("the corrupted input must differ at exactly position k")
        if self.input_ids[self.position_k] != self.token_clean:
            raise ValueError("declared clean token does not sit at position k")
        if self.corrupt_ids[self.position_k] != self.token_corrupt:
            raise ValueError("declared corrupted token does not sit at position k")


def build_path_cases(
    arm: Arm,
    probes: Sequence[RepeatProbe],
    unigram: Unigram,
    *,
    n_cases: int,
    cases_per_probe: int,
    max_tokens: int,
    seed: int,
) -> tuple[list[PathCase], dict[str, Any]]:
    """Sample cases across probes, capping the contribution of any one record.

    The cap exists so that a cohort in which one record happens to carry a very
    long repeat cannot supply most of the cases and turn a panel measurement into
    a measurement of that record.  ``max_tokens`` bounds the read-out position:
    the sequence is truncated at ``q``, so a query deep inside a long protein
    would otherwise set the batch width for every case.  Both the number of
    dropped queries and the reason are reported.

    **Probes are visited under a seeded permutation.**  They used to be visited
    in list order and the loop stopped as soon as ``n_cases`` was reached, so at
    the campaign's defaults -- 64 cases, 4 per probe, a 32-record cohort -- only
    the first sixteen probes ever contributed, and those come from a repeat
    cohort that was itself the head of a family-grouped file.  The seed was spent
    entirely on choosing query positions *within* a probe, which is the one place
    the choice did not matter.  The docstring said "sample cases across probes";
    now it does.
    """

    if n_cases < 1 or cases_per_probe < 1 or max_tokens < 8:
        raise ValueError("invalid path-case parameters")
    if not probes:
        raise ValueError("no probes were supplied")
    kinds = {probe.kind for probe in probes}
    if len(kinds) != 1:
        raise ValueError(
            f"{arm.name}: probes mix kinds {sorted(kinds)}; the provenance record "
            "labels the whole case set by one kind and the design crosses criteria, "
            "so a mixed list would be labelled by whichever probe came first"
        )
    support = {int(token) for token in unigram.token_ids}
    rng = np.random.default_rng(seed)
    cases: list[PathCase] = []
    dropped_long = 0
    dropped_support = 0
    candidates = 0
    visiting = [int(index) for index in rng.permutation(len(probes))]
    for probe_index in visiting:
        probe = probes[probe_index]
        eligible: list[int] = []
        for index, (query, key) in enumerate(zip(probe.query_positions, probe.key_positions)):
            candidates += 1
            if query + 1 > max_tokens:
                dropped_long += 1
                continue
            if int(probe.input_ids[key]) not in support:
                dropped_support += 1
                continue
            eligible.append(index)
        if not eligible:
            continue
        take = min(cases_per_probe, len(eligible), n_cases - len(cases))
        picked = rng.choice(np.asarray(eligible), size=take, replace=False)
        for index in sorted(int(value) for value in picked):
            query = int(probe.query_positions[index])
            key = int(probe.key_positions[index])
            clean = [int(token) for token in probe.input_ids[: query + 1]]
            token_clean = clean[key]
            token_corrupt = unigram.sample_other(rng, token_clean)
            corrupt = list(clean)
            corrupt[key] = token_corrupt
            cases.append(
                PathCase(
                    input_ids=tuple(clean),
                    corrupt_ids=tuple(corrupt),
                    position_q=query,
                    position_k=key,
                    token_clean=token_clean,
                    token_corrupt=token_corrupt,
                    probe_index=probe_index,
                    probe_kind=probe.kind,
                )
            )
        if len(cases) >= n_cases:
            break
    if len(cases) < n_cases:
        raise RuntimeError(
            f"{arm.name}: only {len(cases)}/{n_cases} path cases from {len(probes)} probes "
            f"at max_tokens={max_tokens}; {dropped_long} queries sat beyond the token cap "
            f"and {dropped_support} keys held a token outside the unigram support"
        )
    provenance = {
        "n_cases": len(cases),
        "n_probes_used": len({case.probe_index for case in cases}),
        "n_probes_available": len(probes),
        "cases_per_probe": int(cases_per_probe),
        "max_case_tokens": int(max_tokens),
        "probe_visit_order": "seeded_permutation",
        "probe_kind": cases[0].probe_kind,
        "candidate_query_positions": candidates,
        "dropped_beyond_token_cap": dropped_long,
        "dropped_key_outside_unigram_support": dropped_support,
        "mean_case_tokens": _finite(
            float(np.mean([len(case.input_ids) for case in cases])), "case length"
        ),
        "max_case_tokens_observed": int(max(len(case.input_ids) for case in cases)),
        "mean_query_key_distance": _finite(
            float(np.mean([case.position_q - case.position_k for case in cases])),
            "query-key distance",
        ),
        "seed": int(seed),
    }
    return cases, provenance


# ------------------------------------------------------------------- patcher


#: Where each architecture keeps the projection whose *input* is the
#: concatenation of per-head outputs. Declared per architecture rather than found
#: by trying attribute names in turn, for the reason ``arms.py`` states about
#: attention modules: a search resolves a newly admitted architecture to whichever
#: candidate attribute happened to exist on it, and getting this wrong moves the
#: measurement to a different head without changing any shape.
_OUTPUT_PROJECTION_ATTRIBUTE: dict[str, str] = {
    "gpt2": "c_proj",
    "progen": "out_proj",
    "llama": "o_proj",
    "qwen2": "o_proj",
}

#: Architectures whose module layout this file resolves end to end: the trunk and
#: final norm through :func:`~.circuits.inner_decoder` and
#: :func:`~.circuits.final_norm`, and an attention output projection above.
#:
#: The two rotary lineages were added for EXP-R2-079. Until then this module
#: refused them, and the refusal was catalogued as an instrument limit on L22:
#: every text arm carrying that result was GPT-2 architecture, which is the exact
#: configuration that retracted the QK/OV finding. The extension is mechanical
#: rather than conceptual, because the three things a rotary decoder does
#: differently are all downstream of the patch:
#:
#: * **Rotary position embedding** is applied to queries and keys inside the
#:   attention computation. The sender patch replaces the *input* to the output
#:   projection, which is the attention result -- already past RoPE.
#: * **Grouped-query attention** changes which key/value head a query head reads
#:   and does not change the layout of that projection's input, which is the
#:   concatenation of per-*query*-head outputs in every case. ``n_head`` returns
#:   the query count by explicit declaration for this reason.
#: * **RMSNorm** is a drop-in for the final-norm pre-hook, which only reads
#:   ``args[0]``.
SUPPORTED_ARCHITECTURES = frozenset(_OUTPUT_PROJECTION_ATTRIBUTE)


def require_supported_layout(arm: Arm) -> None:
    """Refuse an arm whose module layout this file cannot address.

    ``circuits`` and ``pathway`` are both granted to the rotary text arms, so the
    capability gate alone no longer separates the arms this module can reach from
    the ones it cannot. Refusing here, with the reason, is the panel's own
    convention: an exception at an arbitrary depth of a GPU run is the failure
    mode capability declarations exist to replace.

    **The declared architecture is checked, then verified.** The trunk and the
    final norm are resolved through ``circuits.inner_decoder`` and
    ``circuits.final_norm`` rather than by naming ``model.transformer.ln_f`` here:
    this module held a second copy of that layout decision in six places, and
    Appendix B rule 12 is that the module which decides a layout owns it and every
    other module imports it. Resolving the trunk, the final norm, the block list
    and the layer-zero projection makes admission mean that the path patcher can
    address the declared layout end to end.

    A spec-only ``Arm`` (``model=None``, built by a scheduler to ask the
    question without a checkpoint) cannot be verified, so the declaration is all
    there is and the resolution step is skipped.
    """

    if arm.spec.architecture not in SUPPORTED_ARCHITECTURES:
        raise TypeError(
            f"{arm.name}: path patching is implemented for "
            f"{sorted(SUPPORTED_ARCHITECTURES)} only; {arm.spec.architecture!r} has "
            "no declared attention output projection in this module"
        )
    if arm.model is None:
        return
    # These raise with the architecture and the missing attribute named, which is
    # the diagnostic a layout mismatch needs; catching and re-wrapping them would
    # only restate it less precisely.
    inner_decoder(arm)
    final_norm(arm)
    blocks = arm.blocks()
    if len(blocks) != arm.n_layer:
        raise TypeError(
            f"{arm.name}: declared {arm.n_layer} layers but the block list has "
            f"{len(blocks)}"
        )
    if not hasattr(arm.model, "lm_head"):
        raise TypeError(f"{arm.name}: path patching requires model.lm_head")
    attention_output_projection(arm, 0)


def attention_output_projection(arm: Arm, layer: int) -> torch.nn.Module:
    """The projection whose input is the concatenation of the per-head outputs.

    Patching there rather than at the attention module's output is what makes the
    sender a *head* rather than a layer: head ``h`` owns columns
    ``[h * d_head, (h + 1) * d_head)`` of this input, and
    :func:`~.circuits.head_ov_weights` reads the matching rows of the same
    projection weight.  The correspondence is checked against the live forward
    pass by :func:`structural_invariants`, because getting it wrong would move
    the measurement to a different head without changing any shape.
    """

    attribute = _OUTPUT_PROJECTION_ATTRIBUTE.get(arm.spec.architecture)
    if attribute is None:
        raise TypeError(
            f"{arm.name}: no attention output projection is declared for "
            f"{arm.spec.architecture!r}; implemented: "
            f"{sorted(_OUTPUT_PROJECTION_ATTRIBUTE)}"
        )
    attention = arm.attention(layer)
    if not hasattr(attention, attribute):
        raise TypeError(
            f"{arm.name}: declared {arm.spec.architecture} but layer {layer} has no "
            f"{attribute}"
        )
    return getattr(attention, attribute)


@dataclass
class _Batch:
    """One padded batch of cases with every activation the patcher needs cached."""

    clean_ids: torch.Tensor
    corrupt_ids: torch.Tensor
    mask: torch.Tensor
    rows: torch.Tensor
    q_index: torch.Tensor
    token_clean: torch.Tensor
    token_corrupt: torch.Tensor
    metric_clean: torch.Tensor
    metric_corrupt: torch.Tensor
    denominator: torch.Tensor
    eligible: torch.Tensor
    clean_z: dict[int, torch.Tensor]
    base_z: dict[int, torch.Tensor]
    base_attn: dict[int, torch.Tensor]
    base_mlp: dict[int, torch.Tensor]
    clean_resid_post_final: torch.Tensor
    base_final_residual: torch.Tensor
    n_cases: int
    #: The probe each case came from. Up to ``cases_per_probe`` cases are drawn
    #: from one probe, and they are nested prefixes of one protein sharing almost
    #: their whole context, so the record -- not the case -- is the sampling unit.
    probe_index: torch.Tensor


def _probe_clustered_sem(
    values: torch.Tensor, probes: torch.Tensor, pathway: str
) -> dict[str, Any]:
    """Standard error over probe records rather than over cases, with its own centre.

    Averages within probe first, then takes the standard error over probe means.
    ``None`` when fewer than two probes contributed, because a single record
    supports no interval and a fabricated zero reads as perfect precision.

    ``mean_probe_clustered`` is returned because it is **the estimator this
    standard error belongs to**, and it is not the ``mean`` published beside it.
    That ``mean`` is ``values.mean()``, weighted by how many cases each probe
    contributed; this one weights every probe equally. The two coincide only
    when every probe contributes the same number of eligible cases, which the
    eligibility filter specifically prevents -- a probe whose corrupted token
    barely moves the metric contributes one case, a strong one contributes
    ``cases_per_probe``. On a constructed but entirely ordinary example -- three
    probes contributing 20, 2 and 2 cases at recoveries 0.9, 0.1 and 0.05 -- the
    case-weighted mean is 0.7625 and the clustered standard error is 0.2754
    around a centre of 0.35, so the published mean sits one and a half standard
    errors outside its own interval. Both centres are reported rather than one
    being silently replaced: the case-weighted mean is what several artefacts
    quote, and the clustered pair is the one to read.
    """

    if values.numel() != probes.numel():
        raise ValueError("recovery values and probe indices do not align")
    unique = torch.unique(probes)
    if unique.numel() < 2:
        return {
            "mean_probe_clustered": (
                _finite(float(values.mean()), f"{pathway} clustered mean")
                if values.numel() > 0
                else None
            ),
            "sem_probe_clustered": None,
            "n_probes": int(unique.numel()),
        }
    means = torch.stack([values[probes == probe].mean() for probe in unique])
    return {
        "mean_probe_clustered": _finite(
            float(means.mean()), f"{pathway} clustered mean"
        ),
        "sem_probe_clustered": _finite(
            float(means.std() / math.sqrt(means.numel())), f"{pathway} clustered sem"
        ),
        "n_probes": int(unique.numel()),
    }


class PathPatcher:
    """Runs one arm's path-patching conditions over a fixed set of cases.

    The class exists to hold the caches rather than to add abstraction: every
    condition needs the clean per-head writes, the corrupted-run sublayer outputs
    to freeze at, and the two metric endpoints, and recomputing them per sender
    would dominate the runtime and risk the four conditions being measured
    against three slightly different baselines.
    """

    def __init__(
        self,
        arm: Arm,
        cases: Sequence[PathCase],
        *,
        batch_size: int,
        minimum_effect: float = DEFAULT_MINIMUM_EFFECT,
    ) -> None:
        arm.require("circuits")
        arm.require("pathway")
        # The capability gate does not imply this module's layout: it is granted
        # per measurement family, while admission here means every module this
        # file addresses resolves on this checkpoint. Without the check an arm
        # passes the gate and dies at an arbitrary depth of a GPU run, which
        # ``arms.py`` names as the exact failure capability declarations exist to
        # prevent. The set is now four architectures rather than two, so the check
        # is what distinguishes "declared and resolvable" from "declared".
        require_supported_layout(arm)
        if not cases:
            raise ValueError("no path cases were supplied")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if minimum_effect <= 0:
            raise ValueError("minimum_effect must be positive")
        self.arm = arm
        self.cases = list(cases)
        self.minimum_effect = float(minimum_effect)
        self.n_head = n_head(arm)
        # ``circuits.head_dim`` exists because ``d_model / n_head`` is an
        # inference, not a definition: it raises when the division is inexact and
        # prefers a config that declares ``head_dim``. Dividing here would
        # truncate silently and patch the wrong columns, while
        # ``head_ov_weights`` -- used by the linearity check below -- would be
        # slicing on the other value.
        self.head_dim = head_dim(arm)
        # The invariant is that the heads tile the projection's INPUT, not that
        # they tile the residual stream. Those coincide on every current panel
        # arm, and they are not the same statement: a decoder that declares a
        # ``head_dim`` independent of ``d_model / n_head`` -- which is why
        # ``circuits.head_dim`` prefers the declaration -- has a perfectly
        # patchable output projection whose input is ``n_head * head_dim`` wide,
        # and the residual-stream form of this check would refuse it for a
        # property the patch does not use.
        self.sender_width = self.head_dim * self.n_head
        projection_input = attention_output_projection(arm, 0).weight
        # Conv1D (GPT-2) stores (in, out); nn.Linear stores (out, in). Take the
        # axis that matches one of them rather than assuming a class.
        in_features = getattr(
            attention_output_projection(arm, 0), "in_features", None
        ) or int(projection_input.shape[0])
        if self.sender_width != in_features:
            raise TypeError(
                f"{arm.name}: {self.n_head} heads of width {self.head_dim} sum to "
                f"{self.sender_width} but the attention output projection takes "
                f"{in_features} inputs, so a per-head column slice of it is not defined"
            )
        self.batches = [
            self._prepare(self.cases[begin : begin + batch_size])
            for begin in range(0, len(self.cases), batch_size)
        ]
        self.batch_size = int(batch_size)

    # ------------------------------------------------------------- plumbing

    @torch.no_grad()
    def _readout(self, ids: torch.Tensor, mask: torch.Tensor, batch: _Batch) -> torch.Tensor:
        """Logits at the read-out position only.

        The transformer trunk is called directly and the unembedding applied to a
        single position per row, so the full ``[batch, token, vocab]`` logit
        tensor is never materialised -- at a fifty-thousand-piece vocabulary and
        six hundred tokens that tensor is larger than the model.
        ``structural_invariants`` checks this read-out against the model's own
        logits rather than assuming the trunk is the whole forward pass.
        """

        hidden = inner_decoder(self.arm)(
            input_ids=ids, attention_mask=mask, use_cache=False
        ).last_hidden_state
        return self._unembed(hidden, batch)

    def _unembed(self, hidden: torch.Tensor, batch: _Batch) -> torch.Tensor:
        return self.arm.model.lm_head(hidden[batch.rows, batch.q_index]).float()

    def _metric(self, read: torch.Tensor, batch: _Batch) -> torch.Tensor:
        return read.gather(1, batch.token_clean.unsqueeze(1)).squeeze(1) - read.gather(
            1, batch.token_corrupt.unsqueeze(1)
        ).squeeze(1)

    def _freeze_hook(self, cached: torch.Tensor, batch: _Batch, *, all_positions: bool):
        rows, index = batch.rows, batch.q_index

        def hook(_module, _args, output: Any) -> Any:
            tensor = _leading(output)
            patched = tensor.clone()
            if all_positions:
                patched.copy_(cached.to(patched.dtype))
            else:
                patched[rows, index] = cached.to(patched.dtype)
            if isinstance(output, tuple):
                return (patched,) + tuple(output[1:])
            return patched

        return hook

    def _sender_hook(self, head: int, value: torch.Tensor, batch: _Batch):
        rows, index = batch.rows, batch.q_index
        low = head * self.head_dim
        high = low + self.head_dim

        def hook(_module, args: tuple) -> tuple:
            tensor = args[0]
            # Against the projection's own input width, for the reason stated at
            # ``self.sender_width``: this is the axis the head slice indexes, and
            # it is not the residual width on a decoder that declares its own
            # head_dim.
            if tensor.ndim != 3 or tensor.shape[-1] != self.sender_width:
                raise TypeError("attention output projection received an unexpected input")
            patched = tensor.clone()
            patched[rows, index, low:high] = value.to(patched.dtype)
            return (patched,) + tuple(args[1:])

        return hook

    # -------------------------------------------------------------- caching

    @torch.no_grad()
    def _prepare(self, chunk: Sequence[PathCase]) -> _Batch:
        arm = self.arm
        pad = arm.tokenizer.pad_token_id
        if pad is None:
            raise ValueError(f"{arm.name}: tokenizer has no pad token")
        width = max(len(case.input_ids) for case in chunk)
        device = arm.device
        clean = torch.full((len(chunk), width), int(pad), dtype=torch.long, device=device)
        corrupt = torch.full((len(chunk), width), int(pad), dtype=torch.long, device=device)
        mask = torch.zeros((len(chunk), width), dtype=torch.long, device=device)
        for row, case in enumerate(chunk):
            length = len(case.input_ids)
            clean[row, :length] = torch.tensor(case.input_ids, dtype=torch.long, device=device)
            corrupt[row, :length] = torch.tensor(
                case.corrupt_ids, dtype=torch.long, device=device
            )
            mask[row, :length] = 1
        batch = _Batch(
            clean_ids=clean,
            corrupt_ids=corrupt,
            mask=mask,
            rows=torch.arange(len(chunk), device=device),
            q_index=torch.tensor(
                [case.position_q for case in chunk], dtype=torch.long, device=device
            ),
            token_clean=torch.tensor(
                [case.token_clean for case in chunk], dtype=torch.long, device=device
            ),
            token_corrupt=torch.tensor(
                [case.token_corrupt for case in chunk], dtype=torch.long, device=device
            ),
            metric_clean=torch.zeros(len(chunk), device=device),
            metric_corrupt=torch.zeros(len(chunk), device=device),
            denominator=torch.zeros(len(chunk), device=device),
            eligible=torch.zeros(len(chunk), dtype=torch.bool, device=device),
            clean_z={},
            base_z={},
            base_attn={},
            base_mlp={},
            clean_resid_post_final=torch.zeros(1, device=device),
            base_final_residual=torch.zeros(1, device=device),
            n_cases=len(chunk),
            probe_index=torch.tensor(
                [case.probe_index for case in chunk], dtype=torch.long, device=device
            ),
        )

        clean = self._capture(batch, batch.clean_ids, capture_sublayers=False)
        batch.clean_z = clean["z"]
        batch.clean_resid_post_final = clean["resid_post"]
        batch.metric_clean = self._metric(clean["read"], batch)

        base = self._capture(batch, batch.corrupt_ids, capture_sublayers=True)
        batch.base_z = base["z"]
        batch.base_attn = base["attn"]
        batch.base_mlp = base["mlp"]
        batch.base_final_residual = base["final_residual"]
        batch.metric_corrupt = self._metric(base["read"], batch)

        batch.denominator = batch.metric_clean - batch.metric_corrupt
        batch.eligible = batch.denominator >= self.minimum_effect
        return batch

    @torch.no_grad()
    def _capture(
        self, batch: _Batch, ids: torch.Tensor, *, capture_sublayers: bool
    ) -> dict[str, Any]:
        """One forward pass, caching everything a later condition may need at ``q``."""

        arm = self.arm
        rows, index = batch.rows, batch.q_index
        z_cache: dict[int, torch.Tensor] = {}
        attn_cache: dict[int, torch.Tensor] = {}
        mlp_cache: dict[int, torch.Tensor] = {}
        resid_final: dict[str, torch.Tensor] = {}
        handles = []

        def capture_z(layer: int):
            def hook(_module, args: tuple) -> None:
                z_cache[layer] = args[0][rows, index].detach().clone()

            return hook

        def capture_out(store: dict[int, torch.Tensor], layer: int):
            def hook(_module, _args, output: Any) -> None:
                store[layer] = _leading(output)[rows, index].detach().clone()

            return hook

        def capture_block(layer: int):
            def hook(_module, _args, output: Any) -> None:
                resid_final["resid_post"] = _leading(output)[rows, index].detach().clone()

            return hook

        def capture_ln_f(_module, args: tuple) -> None:
            resid_final["final_residual"] = args[0][rows, index].detach().clone()

        for layer in range(arm.n_layer):
            handles.append(
                attention_output_projection(arm, layer).register_forward_pre_hook(
                    capture_z(layer)
                )
            )
            if capture_sublayers:
                handles.append(
                    arm.attention(layer).register_forward_hook(capture_out(attn_cache, layer))
                )
                handles.append(arm.mlp(layer).register_forward_hook(capture_out(mlp_cache, layer)))
        handles.append(arm.blocks()[arm.n_layer - 1].register_forward_hook(capture_block(arm.n_layer - 1)))
        handles.append(final_norm(arm).register_forward_pre_hook(capture_ln_f))
        try:
            hidden = inner_decoder(arm)(
                input_ids=ids, attention_mask=batch.mask, use_cache=False
            ).last_hidden_state
        finally:
            for handle in handles:
                handle.remove()
        if len(z_cache) != arm.n_layer:
            raise RuntimeError(f"{arm.name}: per-head capture reached {len(z_cache)} layers")
        if capture_sublayers and (
            len(attn_cache) != arm.n_layer or len(mlp_cache) != arm.n_layer
        ):
            raise RuntimeError(f"{arm.name}: sublayer capture is incomplete")
        if set(resid_final) != {"resid_post", "final_residual"}:
            raise RuntimeError(f"{arm.name}: final residual capture failed")
        return {
            "z": z_cache,
            "resid_post": resid_final["resid_post"],
            "attn": attn_cache,
            "mlp": mlp_cache,
            "final_residual": resid_final["final_residual"],
            "read": self._unembed(hidden, batch),
        }

    # ------------------------------------------------------------ conditions

    @torch.no_grad()
    def _run(
        self,
        batch: _Batch,
        *,
        sender: SenderHead | None,
        freeze_attn_from: int | None,
        freeze_mlp_from: int | None,
        only_attn_layer: int | None = None,
        only_mlp_layer: int | None = None,
        patch_source: str = "clean",
        capture_final_residual: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """One patched forward pass, returning the metric and optionally ``ln_f``'s input.

        ``freeze_*_from`` names the first layer whose sublayer output is pinned to
        the corrupted-run value; ``only_*_layer`` exempts a single layer from that
        freeze, which is how a per-layer receiver is isolated.
        """

        arm = self.arm
        # Validated unconditionally and before use: an unknown source silently
        # resolved to the corrupted cache, and the check was skipped entirely
        # whenever ``sender`` was None.
        if patch_source not in {"clean", "base"}:
            raise ValueError(f"unknown patch source {patch_source!r}")
        if only_attn_layer is not None and freeze_attn_from is None:
            raise ValueError("only_attn_layer exempts a layer from a freeze that is not applied")
        if only_mlp_layer is not None and freeze_mlp_from is None:
            raise ValueError("only_mlp_layer exempts a layer from a freeze that is not applied")
        handles = []
        captured: dict[str, torch.Tensor] = {}
        if sender is not None:
            source = batch.clean_z if patch_source == "clean" else batch.base_z
            low = sender.head * self.head_dim
            value = source[sender.layer][:, low : low + self.head_dim]
            handles.append(
                attention_output_projection(arm, sender.layer).register_forward_pre_hook(
                    self._sender_hook(sender.head, value, batch)
                )
            )
        if freeze_attn_from is not None:
            for layer in range(freeze_attn_from, arm.n_layer):
                if layer == only_attn_layer:
                    continue
                handles.append(
                    arm.attention(layer).register_forward_hook(
                        self._freeze_hook(batch.base_attn[layer], batch, all_positions=False)
                    )
                )
        if freeze_mlp_from is not None:
            for layer in range(freeze_mlp_from, arm.n_layer):
                if layer == only_mlp_layer:
                    continue
                handles.append(
                    arm.mlp(layer).register_forward_hook(
                        self._freeze_hook(batch.base_mlp[layer], batch, all_positions=False)
                    )
                )
        if capture_final_residual:

            def capture(_module, args: tuple) -> None:
                captured["final_residual"] = args[0][batch.rows, batch.q_index].detach().clone()

            handles.append(final_norm(arm).register_forward_pre_hook(capture))
        try:
            read = self._readout(batch.corrupt_ids, batch.mask, batch)
        finally:
            for handle in handles:
                handle.remove()
        return self._metric(read, batch), captured.get("final_residual")

    def _recovery(self, metric: torch.Tensor, batch: _Batch) -> torch.Tensor:
        return (metric - batch.metric_corrupt) / batch.denominator

    def _pathway_freeze(self, pathway: str, layer: int) -> tuple[int | None, int | None]:
        """Which sublayers a pathway pins, for a sender at ``layer``.

        Attention is frozen from ``layer + 1`` and MLPs from ``layer`` because a
        sequential block feeds its own MLP from its attention output.  A parallel
        block does not, so pinning its layer-``layer`` MLP is a no-op; the rule is
        stated once here rather than branched on architecture, since freezing a
        component that is not downstream cannot change anything.
        """

        if pathway == "direct":
            return layer + 1, layer
        if pathway == "via_mlp":
            return layer + 1, None
        if pathway == "via_attn":
            return None, layer
        if pathway == "total":
            return None, None
        raise ValueError(f"unknown pathway {pathway!r}")

    # --------------------------------------------------------------- public

    def eligibility(self) -> dict[str, Any]:
        denominators = torch.cat([batch.denominator for batch in self.batches])
        eligible = torch.cat([batch.eligible for batch in self.batches])
        clean = torch.cat([batch.metric_clean for batch in self.batches])
        corrupt = torch.cat([batch.metric_corrupt for batch in self.batches])
        count = int(eligible.sum())
        if count < 1:
            raise RuntimeError(
                f"{self.arm.name}: no case reached the {self.minimum_effect} logit "
                "eligibility floor; corrupting the copied token never moved the metric, "
                "so there is no induction effect here for path patching to decompose"
            )
        return {
            "n_cases": int(denominators.numel()),
            "n_eligible": count,
            "eligible_fraction": _finite(count / denominators.numel(), "eligible fraction"),
            "minimum_effect_logits": self.minimum_effect,
            "eligibility_rule": "signed: L_clean - L_corrupt >= minimum_effect",
            "n_absolute_effect_above_floor": int(
                (denominators.abs() >= self.minimum_effect).sum()
            ),
            "mean_clean_metric": _finite(float(clean.mean()), "clean metric"),
            "mean_corrupt_metric": _finite(float(corrupt.mean()), "corrupt metric"),
            "mean_denominator": _finite(float(denominators.mean()), "denominator"),
            "mean_denominator_eligible": _finite(
                float(denominators[eligible].mean()), "eligible denominator"
            ),
        }

    @torch.no_grad()
    def sender_recoveries(self, sender: SenderHead) -> dict[str, dict[str, Any]]:
        """Mean recovered logit difference per pathway, on both scales.

        ``mean`` is the recovery: each case's restored logit difference divided by
        that case's own ``L_clean - L_corrupt``. ``mean_logits`` is the numerator
        of that ratio, averaged over the same eligible cases and nothing else.

        The second one is published because the first is not comparable across
        arms and reads as though it were. The denominator is a property of the arm
        and the case set -- mean eligible denominators on this panel run from 0.76
        logits on ZymCTRL to 24.10 on gpt2-large, a factor of 32 -- so a head
        carrying recovery 0.30 on the arm with the small denominator is a *weaker*
        write to the logits than a head carrying 0.02 on the arm with the large
        one. EXP-R2-071 published exactly that comparison in the wrong direction.
        The numerator was already computed here and thrown away.
        """

        collected: dict[str, list[torch.Tensor]] = {pathway: [] for pathway in PATHWAYS}
        collected_logits: dict[str, list[torch.Tensor]] = {
            pathway: [] for pathway in PATHWAYS
        }
        keeps: list[torch.Tensor] = []
        for batch in self.batches:
            keeps.append(batch.eligible)
            for pathway in PATHWAYS:
                attn_from, mlp_from = self._pathway_freeze(pathway, sender.layer)
                metric, _ = self._run(
                    batch,
                    sender=sender,
                    freeze_attn_from=attn_from,
                    freeze_mlp_from=mlp_from,
                )
                collected[pathway].append(self._recovery(metric, batch))
                collected_logits[pathway].append(metric - batch.metric_corrupt)
        keep = torch.cat(keeps)
        probes = torch.cat([batch.probe_index for batch in self.batches])[keep]
        result: dict[str, dict[str, Any]] = {}
        for pathway, parts in collected.items():
            values = torch.cat(parts)[keep]
            logits = torch.cat(collected_logits[pathway])[keep]
            # ``build_path_cases`` draws several cases from one probe, and those
            # cases are nested prefixes of one protein sharing nearly their whole
            # context. Dividing by sqrt(n_cases) treats them as independent and
            # understates the interval by roughly the square root of the
            # cases-per-probe ratio. The record is the sampling unit, so the
            # clustered figure is the one to read; the case-level one is kept
            # beside it because several artefacts quote it.
            #
            # ``mean`` and ``sem_probe_clustered`` are not a point estimate and
            # its standard error: ``mean`` weights probes by their case count and
            # the clustered standard error does not. ``mean_probe_clustered``,
            # supplied by ``_probe_clustered_sem``, is the centre that standard
            # error actually describes.
            result[pathway] = {
                "mean": _finite(float(values.mean()), f"{pathway} recovery"),
                "mean_cluster_unit": "case",
                # The same mean before the per-case denominator is applied. Not a
                # rescaling of ``mean`` by one number: each case carries its own
                # denominator, so the two scales can order heads differently.
                "mean_logits": _finite(
                    float(logits.mean()), f"{pathway} logit effect"
                ),
                # ``None``, not 0.0. Nine lines below, ``_probe_clustered_sem``
                # returns None in the same situation and says why: a single
                # observation supports no interval and a fabricated zero reads as
                # perfect precision. This branch published the zero.
                "sem": (
                    _finite(
                        float(values.std() / math.sqrt(values.numel())),
                        f"{pathway} sem",
                    )
                    if values.numel() > 1
                    else None
                ),
                "sem_cluster_unit": "case",
                "median": _finite(float(values.median()), f"{pathway} median"),
                "n": int(values.numel()),
                **_probe_clustered_sem(values, probes, pathway),
            }
        return result

    @torch.no_grad()
    def sender_receiver_profile(self, sender: SenderHead) -> dict[str, Any]:
        """Per-layer receivers: the sender to one component to the unembedding.

        Everything downstream of the receiver is pinned, so each number is the
        two-step path and nothing else.  The per-layer effects do not sum to the
        pathway-level mediated effect, because unfreezing one component at a time
        cannot produce the mass that only appears when several recompute together;
        that shortfall is returned rather than absorbed.
        """

        parts: list[torch.Tensor] = []
        keeps: list[torch.Tensor] = []
        for batch in self.batches:
            keeps.append(batch.eligible)
            metric, _ = self._run(
                batch,
                sender=sender,
                freeze_attn_from=sender.layer + 1,
                freeze_mlp_from=sender.layer,
            )
            parts.append(self._recovery(metric, batch))
        direct = float(torch.cat(parts)[torch.cat(keeps)].mean())
        profile: dict[str, Any] = {"direct": _finite(direct, "profile direct"), "mlp": {}, "attn": {}}
        for kind in ("mlp", "attn"):
            first = sender.layer if kind == "mlp" else sender.layer + 1
            for receiver in range(first, self.arm.n_layer):
                parts: list[torch.Tensor] = []
                keeps: list[torch.Tensor] = []
                for batch in self.batches:
                    keeps.append(batch.eligible)
                    metric, _ = self._run(
                        batch,
                        sender=sender,
                        freeze_attn_from=sender.layer + 1,
                        freeze_mlp_from=sender.layer,
                        only_mlp_layer=receiver if kind == "mlp" else None,
                        only_attn_layer=receiver if kind == "attn" else None,
                    )
                    parts.append(self._recovery(metric, batch))
                values = torch.cat(parts)[torch.cat(keeps)]
                profile[kind][str(receiver)] = _finite(
                    float(values.mean()) - direct, f"{kind} receiver {receiver}"
                )
        return profile


# ------------------------------------------------------------- invariants


def _select_batch_rows(batch: _Batch, rows: torch.Tensor) -> _Batch:
    """A prepared batch restricted to ``rows``, caches included.

    The locality invariant has to cache every sublayer output at every position,
    which is two orders of magnitude more memory than the column-``q`` caches the
    measurement itself uses.  It is a check on the plumbing, not a statistic, so
    it runs on a few rows rather than forcing the whole run to a smaller batch.

    Rows are selected rather than sliced from the front, because the invariant
    needs cases with a usable denominator and eligibility is not a property of
    position in the batch.
    """

    if rows.ndim != 1 or rows.numel() < 1:
        raise ValueError("row selection must be a non-empty index vector")
    if int(rows.max()) >= batch.n_cases or int(rows.min()) < 0:
        raise ValueError(f"row selection falls outside a {batch.n_cases}-case batch")
    count = int(rows.numel())
    width = int(batch.mask.index_select(0, rows).sum(dim=1).max())
    return _Batch(
        clean_ids=batch.clean_ids.index_select(0, rows)[:, :width],
        corrupt_ids=batch.corrupt_ids.index_select(0, rows)[:, :width],
        mask=batch.mask.index_select(0, rows)[:, :width],
        # ``rows`` indexes into the *new* batch, not the old one: every cache
        # below has been re-indexed, so row r of the slice is at position r.
        rows=torch.arange(count, device=batch.rows.device),
        q_index=batch.q_index.index_select(0, rows),
        token_clean=batch.token_clean.index_select(0, rows),
        token_corrupt=batch.token_corrupt.index_select(0, rows),
        metric_clean=batch.metric_clean.index_select(0, rows),
        metric_corrupt=batch.metric_corrupt.index_select(0, rows),
        denominator=batch.denominator.index_select(0, rows),
        eligible=batch.eligible.index_select(0, rows),
        clean_z={
            layer: value.index_select(0, rows) for layer, value in batch.clean_z.items()
        },
        base_z={layer: value.index_select(0, rows) for layer, value in batch.base_z.items()},
        base_attn={
            layer: value.index_select(0, rows) for layer, value in batch.base_attn.items()
        },
        base_mlp={
            layer: value.index_select(0, rows) for layer, value in batch.base_mlp.items()
        },
        clean_resid_post_final=batch.clean_resid_post_final.index_select(0, rows),
        base_final_residual=batch.base_final_residual.index_select(0, rows),
        n_cases=count,
        probe_index=batch.probe_index.index_select(0, rows),
    )


@torch.no_grad()
def structural_invariants(
    patcher: PathPatcher,
    sender: SenderHead,
    *,
    tolerance: float = 1e-3,
    linearity_tolerance: float = 0.02,
    locality_cases: int = 2,
) -> dict[str, Any]:
    """Checks that must hold for the instrument to be a path-patching instrument.

    A path-patching result that does not hold the rest of the model fixed is an
    ablation with extra steps, and nothing in the numbers reveals the difference,
    so the plumbing is checked against facts that are true by construction:

    ``null_patch``            no hook at all recovers exactly 0;
    ``identity_patch``        re-writing the sender's own corrupted value recovers 0;
    ``freeze_only``           pinning every sublayer with no sender patch recovers 0;
    ``freeze_only_perturbed`` pinning every sublayer to a *wrong* value moves the
                              metric, which is what gives ``freeze_only`` force;
    ``resid_final_at_q``      restoring the last block's output at the read-out
                              position recovers exactly 1;
    ``resid_final_all_positions``  the same restoration applied at every position
                              recovers exactly 1, so the read-out really is a
                              function of column ``q`` alone;
    ``position_locality``     the direct condition with sublayers pinned at every
                              position equals the direct condition with them pinned
                              at ``q`` only, which is what licenses the column-``q``
                              caching everywhere else;
    ``readout_matches_model`` the trunk-plus-unembedding read-out equals the
                              model's own logits at the read-out position;
    ``head_write_linearity``  under the direct condition the residual entering the
                              final LayerNorm moves by exactly the patched head's
                              own write, ``(z_clean - z_base) @ W_O[head]``, which
                              is what makes the sender a head and not a layer.
                              Read per case and reported as the worst case, not
                              as a ratio of batch norms -- see below.

    Raised, never returned as a warning: a failure here invalidates every number
    the run would otherwise produce.
    """

    if tolerance <= 0 or linearity_tolerance <= 0:
        raise ValueError("invariant tolerances must be positive")
    if locality_cases < 1:
        raise ValueError("locality_cases must be positive")
    arm = patcher.arm
    batch = patcher.batches[0]
    report: dict[str, Any] = {
        "tolerance": float(tolerance),
        "linearity_tolerance": float(linearity_tolerance),
        "sender": sender.as_dict(),
        # Every recovery-based invariant below averages over *eligible* rows, not
        # over the whole batch. Reporting the batch size made the artefact claim
        # the invariants were checked on eight cases when they may have been
        # checked on one.
        "n_cases_in_batch": batch.n_cases,
        "n_cases_checked": int(batch.eligible.sum()),
        "n_cases_locality_requested": min(locality_cases, batch.n_cases),
    }
    failures: list[str] = []

    def record(name: str, observed: float, expected: float, limit: float) -> None:
        report[name] = {
            "observed": _finite(observed, name),
            "expected": float(expected),
            "absolute_error": _finite(abs(observed - expected), f"{name} error"),
        }
        if abs(observed - expected) > limit:
            failures.append(f"{name}: {observed:.6f} != {expected} (limit {limit})")

    keep = batch.eligible

    def mean_recovery(metric: torch.Tensor) -> float:
        return float(patcher._recovery(metric, batch)[keep].mean())

    metric, _ = patcher._run(batch, sender=None, freeze_attn_from=None, freeze_mlp_from=None)
    record("null_patch", mean_recovery(metric), 0.0, tolerance)

    metric, _ = patcher._run(
        batch,
        sender=sender,
        freeze_attn_from=None,
        freeze_mlp_from=None,
        patch_source="base",
    )
    record("identity_patch", mean_recovery(metric), 0.0, tolerance)

    metric, _ = patcher._run(batch, sender=None, freeze_attn_from=0, freeze_mlp_from=0)
    record("freeze_only", mean_recovery(metric), 0.0, tolerance)

    # ``freeze_only`` on its own passes whether the freezing hooks are exact or
    # do nothing at all, and the docstring used to claim it as "the check that
    # the freezing hooks are exact". It is not. ``_run`` pins every sublayer to
    # ``base_attn``/``base_mlp``, which were captured on the corrupted input --
    # the same input this pass runs on -- so the hook writes each module's own
    # deterministic output back into it. A hook that failed to bind, bound to
    # the wrong module, or wrote into a detached copy would leave the forward
    # pass unchanged and recover exactly 0, which is the number the invariant
    # demands. The failure that hides behind it is not hypothetical: a freeze
    # that silently does not bind makes ``direct``, ``via_mlp``, ``via_attn``
    # and ``total`` the same condition, and the arm then reports 100% direct
    # effect with every invariant green.
    #
    # The positive control pins the same sublayers to zero instead. Zeroing
    # every attention and MLP write into the read-out column from layer 0
    # upwards cannot leave the metric where it was, so a recovery still inside
    # ``tolerance`` means the hooks are not writing where they claim. Requiring
    # movement of at least the same tolerance the null conditions must stay
    # within keeps one number for both directions rather than inventing a
    # second threshold.
    zero_handles = []
    for layer in range(arm.n_layer):
        zero_handles.append(
            arm.attention(layer).register_forward_hook(
                patcher._freeze_hook(
                    torch.zeros_like(batch.base_attn[layer]), batch, all_positions=False
                )
            )
        )
        zero_handles.append(
            arm.mlp(layer).register_forward_hook(
                patcher._freeze_hook(
                    torch.zeros_like(batch.base_mlp[layer]), batch, all_positions=False
                )
            )
        )
    try:
        perturbed_read = patcher._readout(batch.corrupt_ids, batch.mask, batch)
    finally:
        for handle in zero_handles:
            handle.remove()
    perturbed = mean_recovery(patcher._metric(perturbed_read, batch))
    report["freeze_only_perturbed"] = {
        "observed": _finite(perturbed, "freeze_only_perturbed"),
        "required_absolute_movement": float(tolerance),
        "frozen_value": "zero",
        "role": (
            "positive control for freeze_only: the freezing hooks are pinned to a "
            "value that is not the module's own output, so a recovery of zero here "
            "means they are not writing at all"
        ),
    }
    if abs(perturbed) <= tolerance:
        failures.append(
            f"freeze_only_perturbed: {perturbed:.6f} did not move by more than "
            f"{tolerance}; zeroing every sublayer write at the read-out column left "
            "the metric where it was, so the freezing hooks are inert and "
            "freeze_only's pass carries no information"
        )

    final_block = arm.blocks()[arm.n_layer - 1]
    handle = final_block.register_forward_hook(
        patcher._freeze_hook(batch.clean_resid_post_final, batch, all_positions=False)
    )
    try:
        read = patcher._readout(batch.corrupt_ids, batch.mask, batch)
    finally:
        handle.remove()
    record("resid_final_at_q", mean_recovery(patcher._metric(read, batch)), 1.0, tolerance)

    # Take the locality rows from the *eligible* ones rather than from the front
    # of the batch. The previous version fell back to averaging over ineligible
    # rows whenever neither of the first two happened to be eligible, and an
    # ineligible row has a denominator near the floor or at zero -- so a 1e-4
    # float discrepancy between two differently-hooked forward passes became a
    # 1e-2 recovery against a 1e-3 tolerance, and the invariant failed for
    # arithmetic rather than for structure.
    eligible_rows = torch.nonzero(batch.eligible).flatten()
    if eligible_rows.numel() < 1:
        raise RuntimeError(
            f"{arm.name}: no case in the first batch clears the "
            f"{patcher.minimum_effect} eligibility floor, so the locality invariants "
            "have no case with a usable denominator to run on"
        )
    locality_rows = eligible_rows[: min(locality_cases, int(eligible_rows.numel()))]
    small = _select_batch_rows(batch, locality_rows)
    report["n_cases_locality_checked"] = int(small.n_cases)

    def small_recovery(metric: torch.Tensor) -> float:
        return float(patcher._recovery(metric, small).mean())

    full = _capture_all_positions(patcher, small)
    handle = final_block.register_forward_hook(
        patcher._freeze_hook(full["clean_resid_post_final"], small, all_positions=True)
    )
    try:
        read = patcher._readout(small.corrupt_ids, small.mask, small)
    finally:
        handle.remove()
    record(
        "resid_final_all_positions",
        small_recovery(patcher._metric(read, small)),
        1.0,
        tolerance,
    )

    metric_small, _ = patcher._run(
        small,
        sender=sender,
        freeze_attn_from=sender.layer + 1,
        freeze_mlp_from=sender.layer,
    )
    metric_all = _direct_frozen_everywhere(patcher, small, sender, full)
    record("position_locality", small_recovery(metric_all), small_recovery(metric_small), tolerance)

    metric_q, residual_q = patcher._run(
        batch,
        sender=sender,
        freeze_attn_from=sender.layer + 1,
        freeze_mlp_from=sender.layer,
        capture_final_residual=True,
    )
    report["direct_recovery_reference"] = _finite(mean_recovery(metric_q), "direct recovery")

    # Row by row: a full [batch, token, vocab] logit tensor is larger than the
    # model at this vocabulary, and this check exists precisely because the
    # measurement never materialises one.
    errors = []
    for row in range(batch.n_cases):
        ids = batch.clean_ids[row : row + 1]
        mask = batch.mask[row : row + 1]
        position = int(batch.q_index[row])
        reference = arm.model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()[
            0, position
        ]
        hidden = inner_decoder(arm)(
            input_ids=ids, attention_mask=mask, use_cache=False
        ).last_hidden_state
        observed = arm.model.lm_head(hidden[0, position]).float()
        errors.append(
            float((observed - reference).abs().max()) / max(float(reference.abs().max()), 1e-6)
        )
    record("readout_matches_model", max(errors), 0.0, tolerance)

    if residual_q is None:
        raise RuntimeError("the direct condition did not return the final residual")
    _, output_heads = head_ov_weights(arm, sender.layer)
    low = sender.head * patcher.head_dim
    delta_z = (
        batch.clean_z[sender.layer][:, low : low + patcher.head_dim]
        - batch.base_z[sender.layer][:, low : low + patcher.head_dim]
    ).float()
    predicted = batch.base_final_residual.float() + delta_z @ output_heads[sender.head]
    write = predicted - batch.base_final_residual.float()
    error = (residual_q.float() - predicted).norm(dim=-1)
    write_norm = write.norm(dim=-1)
    # Per row, not over the batch. The check used to divide one Frobenius norm by
    # another, which is a case-weighted average error in disguise: one row whose
    # head write is large sets the denominator for every row, so a handful of
    # rows can be entirely wrong and vanish into it. Worked through at tolerance
    # 0.02 over eight rows, seven rows wrong by 100% still passes if the eighth
    # row's write is an order of magnitude larger than theirs. The identity being
    # checked -- that the final residual moves by exactly the patched head's own
    # write -- is a per-case identity, so it is checked per case and the worst
    # case is what has to clear the tolerance.
    #
    # The old ``max(scale, 1e-6)`` floor was the second hole: an arm whose sender
    # genuinely writes almost nothing had its relative error divided by 1e-6, so
    # any absolute discrepancy at all was rescaled to something small. Each row
    # is now divided by its own write, and a row far below its batch's median
    # write is excluded from the maximum and counted rather than being given a
    # denominator it does not have. A sender that writes nothing anywhere is not
    # a sender, and that is a refusal rather than a pass.
    median_write = float(write_norm.median())
    if not median_write > 0.0:
        raise RuntimeError(
            f"{arm.name}: sender {sender.as_dict()} writes nothing into the final "
            "residual on at least half the cases in the batch, so the linearity "
            "identity has no scale to be relative to and the head is not a sender "
            "on these cases"
        )
    usable = write_norm >= _HEAD_WRITE_RELATIVE_FLOOR * median_write
    relative = (error[usable] / write_norm[usable]).float()
    report["head_write_linearity_rows"] = {
        "n_rows": int(write_norm.numel()),
        "n_rows_scored": int(usable.sum()),
        "n_rows_below_write_floor": int(write_norm.numel()) - int(usable.sum()),
        "relative_write_floor": float(_HEAD_WRITE_RELATIVE_FLOOR),
        "median_head_write_norm": _finite(median_write, "median head write"),
        "median_relative_error": _finite(
            float(relative.median()), "head write linearity median"
        ),
        # The statistic this check used to publish, kept so that a run can be
        # compared against the artefacts that quote it and the difference
        # between the two readings is visible rather than inferred.
        "batch_aggregate_relative_error": _finite(
            float(error.norm() / float(write.norm())) if float(write.norm()) > 0 else 0.0,
            "head write linearity aggregate",
        ),
    }
    record(
        "head_write_linearity",
        float(relative.max()),
        0.0,
        linearity_tolerance,
    )

    # ``passed`` was always True: any failure raises on the next line. The report
    # is the evidence; whether it was produced at all is the verdict.
    if failures:
        raise RuntimeError(
            f"{arm.name}: path-patching structural invariants failed: " + "; ".join(failures)
        )
    return report


@torch.no_grad()
def _capture_all_positions(patcher: PathPatcher, batch: _Batch) -> dict[str, Any]:
    """Whole-sequence sublayer caches, used only by the locality invariant."""

    arm = patcher.arm
    attn: dict[int, torch.Tensor] = {}
    mlp: dict[int, torch.Tensor] = {}
    handles = []

    def store(target: dict[int, torch.Tensor], layer: int):
        def hook(_module, _args, output: Any) -> None:
            target[layer] = _leading(output).detach().clone()

        return hook

    for layer in range(arm.n_layer):
        handles.append(arm.attention(layer).register_forward_hook(store(attn, layer)))
        handles.append(arm.mlp(layer).register_forward_hook(store(mlp, layer)))
    try:
        inner_decoder(arm)(
            input_ids=batch.corrupt_ids, attention_mask=batch.mask, use_cache=False
        )
    finally:
        for handle in handles:
            handle.remove()

    clean_final: dict[str, torch.Tensor] = {}

    def store_block(_module, _args, output: Any) -> None:
        clean_final["resid"] = _leading(output).detach().clone()

    handle = arm.blocks()[arm.n_layer - 1].register_forward_hook(store_block)
    try:
        inner_decoder(arm)(
            input_ids=batch.clean_ids, attention_mask=batch.mask, use_cache=False
        )
    finally:
        handle.remove()
    return {"attn": attn, "mlp": mlp, "clean_resid_post_final": clean_final["resid"]}


@torch.no_grad()
def _direct_frozen_everywhere(
    patcher: PathPatcher, batch: _Batch, sender: SenderHead, full: Mapping[str, Any]
) -> torch.Tensor:
    """The direct condition with every downstream sublayer pinned at every position."""

    arm = patcher.arm
    low = sender.head * patcher.head_dim
    value = batch.clean_z[sender.layer][:, low : low + patcher.head_dim]
    handles = [
        attention_output_projection(arm, sender.layer).register_forward_pre_hook(
            patcher._sender_hook(sender.head, value, batch)
        )
    ]
    for layer in range(sender.layer + 1, arm.n_layer):
        handles.append(
            arm.attention(layer).register_forward_hook(
                patcher._freeze_hook(full["attn"][layer], batch, all_positions=True)
            )
        )
    for layer in range(sender.layer, arm.n_layer):
        handles.append(
            arm.mlp(layer).register_forward_hook(
                patcher._freeze_hook(full["mlp"][layer], batch, all_positions=True)
            )
        )
    try:
        read = patcher._readout(batch.corrupt_ids, batch.mask, batch)
    finally:
        for handle in handles:
            handle.remove()
    return patcher._metric(read, batch)


# --------------------------------------------------------------- summaries


#: The two scales :meth:`PathPatcher.sender_recoveries` publishes a mean on.
#: ``mean`` is the recovery, normalised per case by that case's own
#: ``L_clean - L_corrupt``; ``mean_logits`` is the un-normalised numerator.
EFFECT_SCALE_KEYS: tuple[str, ...] = ("mean", "mean_logits")


def sender_effects(
    recoveries: Mapping[str, Mapping[str, Any]], *, key: str = "mean"
) -> dict[str, float]:
    """The five path quantities one sender head implies, on one declared scale.

    ``key="mean"`` gives the recovery scale, which is what every within-arm
    statement is made on.  ``key="mean_logits"`` gives the same decomposition in
    logits, which is the only scale on which two arms' magnitudes are commensurate
    -- the recovery divides by a denominator that is a property of the arm.  The
    decomposition is linear in the pathway means, so it holds identically on both.
    """

    if key not in EFFECT_SCALE_KEYS:
        raise ValueError(f"unknown effect scale {key!r}; scales are {list(EFFECT_SCALE_KEYS)}")
    direct = float(recoveries["direct"][key])
    via_mlp = float(recoveries["via_mlp"][key])
    via_attn = float(recoveries["via_attn"][key])
    total = float(recoveries["total"][key])
    mlp_mediated = via_mlp - direct
    attn_mediated = via_attn - direct
    return {
        "direct": _finite(direct, "direct"),
        "mlp_mediated": _finite(mlp_mediated, "mlp mediated"),
        "attn_mediated": _finite(attn_mediated, "attn mediated"),
        "interaction": _finite(total - direct - mlp_mediated - attn_mediated, "interaction"),
        "mediated": _finite(total - direct, "mediated"),
        "total": _finite(total, "total"),
    }


def summarise_senders(
    per_head: Sequence[Mapping[str, Any]], *, min_head_effect: float = DEFAULT_MIN_HEAD_EFFECT
) -> dict[str, Any]:
    """Per-sender-head means as the primary reading, with the aggregate beside it.

    Head counts differ across the panel by construction -- roughly fifty heads
    above threshold for gpt2-large against fourteen for ProtGPT2 and six for
    ProGen2-medium -- and the head count is itself the disputed quantity, so a
    sum over senders would confound per-head mediation strength with how many
    heads an arm has.  The per-head mean is therefore primary and the sum is
    reported separately and labelled.  The effect-weighted fraction is a third
    reading: it is the ratio of the two sums, which is head-count invariant but
    weights a head by how much it does.
    """

    if not per_head:
        raise ValueError("no per-head results to summarise")
    if min_head_effect <= 0:
        raise ValueError("min_head_effect must be positive")
    effects = {
        key: np.asarray([float(row["effects"][key]) for row in per_head], dtype=np.float64)
        for key in EFFECTS
    }

    def at_floor(floor: float) -> tuple[dict[str, Any], dict[str, Any], int]:
        usable = np.abs(effects["total"]) >= floor
        count = int(usable.sum())
        mean: dict[str, Any] = {}
        median: dict[str, Any] = {}
        for key in EFFECTS:
            if count == 0:
                mean[key] = None
                median[key] = None
                continue
            ratio = effects[key][usable] / effects["total"][usable]
            mean[key] = _finite(float(ratio.mean()), f"{key} fraction mean")
            median[key] = _finite(float(np.median(ratio)), f"{key} fraction median")
        return mean, median, count

    fractions, fraction_median, n_usable = at_floor(min_head_effect)
    usable = np.abs(effects["total"]) >= min_head_effect
    ladder: dict[str, Any] = {}
    for floor in MIN_HEAD_EFFECT_LADDER:
        mean, _, count = at_floor(floor)
        ladder[f"{floor:g}"] = {"n_heads_used": count, "fraction_mean": mean}
    totals = {key: float(value.sum()) for key, value in effects.items()}
    # ``min_head_effect`` is a floor on *one head's* mean total recovery. Guarding
    # a sum over every sender with the same number let a denominator that is near
    # zero by cancellation clear it: an arm with heads of opposing sign -- which
    # ``n_heads_negative_total`` reports as real -- can have large individual
    # effects and a residual total, giving an unbounded ratio that looks like a
    # pathway share. The guard is scaled to the number of heads being summed, and
    # the total absolute effect is published so the cancellation is visible.
    absolute_total = float(np.abs(effects["total"]).sum())
    n_negative_total = int((effects["total"] < 0.0).sum())
    aggregate_floor = min_head_effect * max(len(per_head), 1)
    aggregate_valid = abs(totals["total"]) >= aggregate_floor
    aggregate_fraction = {
        key: (
            _finite(totals[key] / totals["total"], f"{key} weighted fraction")
            if aggregate_valid
            else None
        )
        for key in EFFECTS
    }
    return {
        "n_senders": len(per_head),
        "per_sender_head_mean": {
            key: _finite(float(value.mean()), f"{key} head mean") for key, value in effects.items()
        },
        "per_sender_head_sd": {
            key: _finite(float(value.std(ddof=1)) if value.size > 1 else 0.0, f"{key} head sd")
            for key, value in effects.items()
        },
        "per_sender_head_median": {
            key: _finite(float(np.median(value)), f"{key} head median")
            for key, value in effects.items()
        },
        "per_sender_head_fraction_mean": fractions,
        "per_sender_head_fraction_median": fraction_median,
        "per_sender_head_fraction_by_floor": ladder,
        "fraction_min_head_effect": float(min_head_effect),
        "fraction_n_heads_used": n_usable,
        "fraction_n_heads_withheld": len(per_head) - n_usable,
        # A head whose total effect is negative -- restoring its clean write moves
        # the metric the wrong way -- still has a well-defined decomposition, but
        # its direct and mediated shares carry the opposite sign convention to a
        # positive head's, and averaging the two kinds silently is how a fraction
        # summary becomes uninterpretable. The count is reported so a reader can
        # see how much of the mean is built on them.
        "n_heads_negative_total": int((effects["total"] < 0.0).sum()),
        "n_heads_negative_total_above_floor": int(
            ((effects["total"] < 0.0) & usable).sum()
        ),
        "aggregate_sum_over_senders": {
            key: _finite(value, f"{key} sum") for key, value in totals.items()
        },
        "aggregate_effect_weighted_fraction": aggregate_fraction,
        "aggregate_n_heads": len(per_head),
        "aggregate_absolute_total": _finite(absolute_total, "absolute total"),
        "aggregate_cancellation_ratio": (
            _finite(abs(totals["total"]) / absolute_total, "cancellation ratio")
            if absolute_total > 0.0
            else None
        ),
        "aggregate_floor": _finite(aggregate_floor, "aggregate floor"),
        "aggregate_fraction_valid": bool(aggregate_valid),
        "aggregate_note": (
            "aggregate_sum_over_senders is confounded by how many heads an arm has "
            "and must not be compared across arms; per_sender_head_* is the primary "
            "statistic. aggregate_effect_weighted_fraction is a total-effect "
            "weighted mean of the per-head fractions ONLY when every head's total "
            "has the same sign: with n_heads_negative_total > 0 some weights are "
            "negative, it is not a weighted mean, and it can fall outside the range "
            "of every per-head fraction it appears to average. It is also computed "
            "over all aggregate_n_heads senders, whereas "
            "per_sender_head_fraction_mean is computed over the "
            "fraction_n_heads_used that clear the floor: two different head "
            "populations, reported side by side"
        ),
        "aggregate_is_a_weighted_mean": bool(n_negative_total == 0),
    }


# ------------------------------------------- concentration and reliability
#
# EXP-R2-071's two structural readings -- that protein arms concentrate causal
# effect into far fewer heads than text does, and that their low all-grid rank
# correlation is not attenuation from noisy estimates -- were computed by
# unversioned code that no longer exists. Neither statistic had an implementation
# anywhere in this repository, so neither could be recomputed, checked against a
# new artefact, or tested. One of the published numbers does not reproduce. The
# functions below are those statistics, defined once and imported wherever they
# are quoted (Appendix B rule 12).


def _heads_carrying_half(magnitude: np.ndarray) -> int:
    """How many of the largest heads it takes to carry half the absolute effect."""

    ordered = np.sort(magnitude)[::-1]
    total = float(ordered.sum())
    cumulative = np.cumsum(ordered)
    return int(np.searchsorted(cumulative, 0.5 * total) + 1)


def gini(values: Sequence[float] | np.ndarray, *, label: str = "effect") -> float:
    """Gini coefficient of ``|values|``: 0 for an even grid, near 1 for one head.

    Head-count free, which is the requirement.  Head counts on this panel run
    from 144 to 720 and the count is itself a disputed quantity, so a top-k share
    is not a comparable statistic -- "the top 1% of heads" is one head on GPT-2
    and seven on ProtGPT2 (Appendix B rule 21).  The Gini is invariant both to
    replicating the grid and to rescaling every head by one positive constant.

    An all-zero grid raises rather than returning 0: zero is the value a
    *perfectly even* grid takes, so returning it for a grid with no effect at all
    would publish the strongest possible statement about a measurement that
    found nothing.
    """

    x = np.sort(np.abs(np.asarray(values, dtype=np.float64)))
    if x.size < 2:
        raise ValueError(f"a Gini over fewer than two {label} values is not defined")
    if not np.isfinite(x).all():
        raise ValueError(f"{label} values contain non-finite entries")
    total = float(x.sum())
    if total <= 0.0:
        raise ValueError(
            f"every {label} magnitude is zero, so there is no distribution to "
            "concentrate; 0.0 is what a perfectly even grid returns and would be "
            "read as one"
        )
    index = np.arange(1, x.size + 1, dtype=np.float64)
    return _finite(
        2.0 * float((index * x).sum()) / (x.size * total) - (x.size + 1) / x.size,
        f"{label} gini",
    )


def share_of_grid_carrying_half_effect(
    values: Sequence[float] | np.ndarray, *, label: str = "effect"
) -> float:
    """The fraction of the head grid that carries half the total absolute effect.

    A proportion of the grid rather than a count, so it does not move with the
    head count; the companion to :func:`gini` under Appendix B rule 21, and the
    more readable of the two.
    """

    magnitude = np.abs(np.asarray(values, dtype=np.float64))
    if magnitude.size < 2:
        raise ValueError(f"a share of grid over fewer than two {label} values is not defined")
    if not np.isfinite(magnitude).all():
        raise ValueError(f"{label} values contain non-finite entries")
    if float(magnitude.sum()) <= 0.0:
        raise ValueError(
            f"every {label} magnitude is zero, so no head carries half of nothing"
        )
    return _finite(
        _heads_carrying_half(magnitude) / magnitude.size, f"{label} share of grid"
    )


def effect_concentration(
    per_head: Sequence[Mapping[str, Any]], *, effect_key: str = "total"
) -> dict[str, Any]:
    """How unevenly one condition spreads its causal effect over the head grid."""

    if not per_head:
        raise ValueError("no per-head results to measure concentration over")
    effect = np.asarray(
        [float(row["effects"][effect_key]) for row in per_head], dtype=np.float64
    )
    magnitude = np.abs(effect)
    n_half = _heads_carrying_half(magnitude) if float(magnitude.sum()) > 0.0 else 0
    return {
        "effect_key": effect_key,
        "n_heads": int(effect.size),
        "gini": gini(effect, label=f"{effect_key} effect"),
        "share_of_grid_carrying_half_effect": share_of_grid_carrying_half_effect(
            effect, label=f"{effect_key} effect"
        ),
        "n_heads_carrying_half_effect": int(n_half),
        "total_absolute_effect": _finite(float(magnitude.sum()), "total absolute effect"),
        "n_heads_negative_effect": int((effect < 0.0).sum()),
        "scale_note": (
            "both statistics are computed on |effect| over the whole grid and are "
            "invariant to replicating it, so two arms with different head counts "
            "are comparable (Appendix B rule 21). They are computed on the recovery "
            "scale. A single positive rescaling of every head would leave both "
            "unchanged, but the recovery scale divides each case by its own "
            "denominator rather than the arm by one constant, so the two scales are "
            "not guaranteed to agree exactly; the logit-scale per-head figures are "
            "in effects_logits"
        ),
        "population_note": (
            "this is the concentration of the sender set that was patched. It is a "
            "statement about the head grid only when that set is the head grid, "
            "which is the exhaustive criterion"
        ),
    }


#: The point estimate and standard error that belong to each other, keyed by the
#: sampling unit both are taken over. **The pair, not the standard error alone.**
#: ``sender_recoveries`` says at its own construction site that ``mean`` and
#: ``sem_probe_clustered`` are not an estimate and its standard error -- ``mean``
#: weights each probe by how many cases it contributed and the clustered standard
#: error describes ``mean_probe_clustered``, which weights probes equally. A
#: reliability built from the observed variance of one estimator and the error
#: variance of the other is not a reliability of anything, and it read ZymCTRL's
#: approximate-repeat grid at 0.008 where the paired figure is 0.170.
RELIABILITY_ESTIMATORS: dict[str, tuple[str, str]] = {
    "probe": ("mean_probe_clustered", "sem_probe_clustered"),
    "case": ("mean", "sem"),
}

#: Which of them a reliability figure should be read from. ``_Batch`` declares the
#: probe record as this design's sampling unit -- several cases are drawn from one
#: probe and are nested prefixes of one record sharing nearly their whole context
#: -- so the case-level standard error understates the error variance and the
#: reliability built on it overstates how well the heads are ordered.
PRIMARY_SEM_CLUSTER_UNIT = "probe"


def head_effect_reliability(
    per_head: Sequence[Mapping[str, Any]], *, pathway: str = "total"
) -> dict[str, Any]:
    """How much of the head-to-head spread in causal effect is signal.

    Appendix B rule 25: a low census-to-causal rank correlation is evidence of
    absent signal only once the per-head estimates are shown reliable enough to
    rank.  Classical reliability answers that from two quantities the artefact
    already carries -- the variance of the per-head means, and each head's own
    standard error:

        reliability = (observed variance - mean error variance) / observed variance

    **Which standard error is not a free choice, and neither is the mean beside
    it.** This module declares the probe record as the sampling unit and publishes
    a case-level and a probe-clustered estimator, each with its own centre.
    EXP-R2-071's published range, 0.916 to 0.991 across six arms, is the
    case-level one.  Both pairs are returned, each labelled with its unit, so the
    difference is a visible property of the artefact rather than a choice made in
    a script that no longer exists.

    This function used to take the observed variance from the case-weighted
    ``mean`` under *both* units while varying only the standard error, which is
    the mismatch ``sender_recoveries`` warns against at the point it builds the
    record.  On the exact-repeat case sets the two pairings agree to within 0.006
    and nothing turned on it; on ZymCTRL's approximate-repeat set the mismatched
    pairing read **0.008** against a paired **0.170** on the magnitude scale, and
    **0.189** against **0.443** signed.  That 0.008 reached the audit document as
    "a grid that cannot be ranked at all" (EXP-R2-078).

    **Two scales, because the ranking statistic is the magnitude.**
    ``reliability_signed_effect`` is the exact quantity: the standard error is the
    standard error of the signed mean, so observed variance decomposes into true
    plus error variance on that scale and nowhere else.
    ``reliability_magnitude_ranking`` applies the same formula to ``|effect|``,
    which is what the census-to-causal correlation actually ranks and what
    EXP-R2-071 quoted; folding the estimate at zero makes its error distribution
    non-normal, so that figure is an approximation and is labelled as one.
    """

    if pathway not in PATHWAYS:
        raise ValueError(f"unknown pathway {pathway!r}; pathways are {list(PATHWAYS)}")
    if len(per_head) < 2:
        raise ValueError("reliability needs at least two heads to have a spread")

    by_unit: dict[str, Any] = {}
    for unit, (mean_key, sem_key) in RELIABILITY_ESTIMATORS.items():
        raw = [row["recovery"][pathway][sem_key] for row in per_head]
        centres = [row["recovery"][pathway][mean_key] for row in per_head]
        # A head missing either half of the pair is missing the pair. Withholding
        # on the standard error alone would leave the observed variance taken over
        # a head population the error variance is not taken over, which is the
        # same class of mismatch this function was repaired for.
        n_missing = sum(
            1 for value, centre in zip(raw, centres) if value is None or centre is None
        )
        record: dict[str, Any] = {
            "sem_cluster_unit": unit,
            "sem_key": sem_key,
            "mean_key": mean_key,
            "n_heads": len(per_head),
            "n_heads_without_a_standard_error": n_missing,
        }
        if n_missing:
            # Not "average over the heads that have one". The observed variance is
            # taken over every head; an error variance taken over a subset is an
            # average over a different population, and the ratio of the two is not
            # a reliability.
            record.update(
                {
                    "reliability_signed_effect": None,
                    "reliability_magnitude_ranking": None,
                    "signal_to_noise_mean_effect": None,
                    "withheld_reason": (
                        f"{n_missing} of {len(per_head)} heads carry no complete "
                        f"({mean_key}, {sem_key}) pair, so the mean error variance "
                        "would be an average over a different head population than "
                        "the observed variance"
                    ),
                }
            )
            by_unit[unit] = record
            continue
        effect = np.asarray([float(value) for value in centres], dtype=np.float64)
        if not np.isfinite(effect).all():
            raise ValueError(f"{mean_key} values contain non-finite entries")
        sem = np.asarray([float(value) for value in raw], dtype=np.float64)
        if not np.isfinite(sem).all() or bool((sem < 0.0).any()):
            raise ValueError(f"{sem_key} values must be finite and non-negative")
        error_variance = float(np.mean(sem**2))
        record["mean_standard_error"] = _finite(float(sem.mean()), "mean standard error")
        record["mean_error_variance"] = _finite(error_variance, "mean error variance")
        for scale, values in (
            ("signed_effect", effect),
            ("magnitude_ranking", np.abs(effect)),
        ):
            observed = float(np.var(values, ddof=1))
            record[f"observed_variance_{scale}"] = _finite(observed, f"{scale} variance")
            record[f"reliability_{scale}"] = (
                _finite((observed - error_variance) / observed, f"{scale} reliability")
                if observed > 0.0
                else None
            )
        record["signal_to_noise_mean_effect"] = (
            _finite(float(np.abs(effect).mean() / sem.mean()), "effect snr")
            if float(sem.mean()) > 0.0
            else None
        )
        by_unit[unit] = record

    return {
        "pathway": pathway,
        "primary_sem_cluster_unit": PRIMARY_SEM_CLUSTER_UNIT,
        "by_sem_cluster_unit": by_unit,
        "estimator": (
            "(variance of the per-head means - mean of the squared per-head "
            "standard errors) / variance of the per-head means"
        ),
        "unit_note": (
            "read by_sem_cluster_unit.probe. The case-level figure treats several "
            "nested prefixes of one protein as independent observations, so it "
            "understates the error variance and overstates the reliability; it is "
            "published beside it because it is the figure EXP-R2-071 quoted. Each "
            "unit's observed variance is taken over the centre named in its own "
            "mean_key, which is the estimator its sem_key describes"
        ),
        "scale_note": (
            "reliability_signed_effect is exact: the standard error belongs to the "
            "signed mean. reliability_magnitude_ranking applies the same formula to "
            "|effect|, which is the quantity the census-to-causal correlation ranks, "
            "and is an approximation because folding at zero makes the error "
            "distribution non-normal"
        ),
    }


#: What the resampled head population *is*, per sender criterion. The caveat used
#: to be one fixed string describing a threshold-selected population, which is
#: false under ``exhaustive_all_heads``: there the population is the arm's entire
#: head grid, most of it scored at zero by the census. A provenance string that is
#: wrong for the criterion the run used is worse than none, because it reads as a
#: checked fact.
RESAMPLED_POPULATION: dict[str, str] = {
    "prefix_matching_above_threshold": (
        "every head this arm scores at or above the prefix-matching threshold: its "
        "whole population under that criterion rather than a sample of it, and "
        "selected by a threshold on the census score, so the heads are not "
        "exchangeable either"
    ),
    "top_k_no_head_above_threshold": (
        "no head of this arm reaches the prefix-matching threshold, so the "
        "population is its fallback_top_k highest-scoring heads: a top-k cut of the "
        "grid, selected on the census score, and flagged by "
        "not_induction_heads_by_panel_criterion as not induction heads under the "
        "panel's own criterion"
    ),
    EXHAUSTIVE_CRITERION: (
        "the arm's entire attention-head grid, not a threshold-selected subset. The "
        "population is complete by construction, most of it carries an effect near "
        "zero, and resampling it describes how heterogeneous a whole grid is"
    ),
}


def bootstrap_difference(
    left: Sequence[float],
    right: Sequence[float],
    *,
    resamples: int,
    seed: int,
    left_criterion: str,
    right_criterion: str,
    minimum_units: int = MINIMUM_BOOTSTRAP_UNITS,
) -> dict[str, Any]:
    """Percentile spread of ``mean(left) - mean(right)`` across sender heads.

    **This is not a sampling confidence interval, and the previous docstring's
    justification for calling it one was invalid on its own terms.** An arm's
    induction heads are not a draw from a superpopulation of that checkpoint's
    heads -- they are all of them -- and they were selected by a threshold on
    prefix matching, so they are not exchangeable either. "The uncertainty that
    comes from an arm having a handful of induction heads" is not sampling
    uncertainty: the head count is a fixed, measured property of the checkpoint.
    Resampling them describes how heterogeneous a fixed head population is.

    Meanwhile the one genuine sampling unit in this design -- the probe records,
    which really were drawn from a corpus -- contributes nothing to this interval
    at all. So the interval manufactures variance from a fixed population and
    omits it from the real one, in both directions at once.

    It is kept because head-to-head heterogeneity is worth reporting, and renamed
    so it cannot be quoted as a confidence statement: ``spread_low``/
    ``spread_high`` rather than ``ci_low``/``ci_high``, and
    ``separated_across_heads`` rather than ``excludes_zero``. A two-sided
    bootstrap p-value is returned so that a caller running many of these
    comparisons can apply a multiplicity correction, which nothing here does.

    **The unit floor applies here too, and this function had none.** It guarded
    ``n < 2`` while ``homology.bootstrap_stratum`` refused anything below eight
    for the same statistic and the same reason, and the 2026-07-28 panel summary
    published spreads at ``n_left`` of 3, 4, 5 and 6 -- three of them with
    ``excludes_zero: true``, on the protein side of the controlled pair. A
    percentile spread over four heads trims its own extreme atoms; whether that
    is called a confidence interval or a heterogeneity spread does not change
    what the percentile rule does to it.

    Degenerate is *returned*, not raised, because the head count is a measured
    property of the arm: ProGen2-medium has four induction heads under the
    approximate criterion and that is the finding, not a configuration error.
    The point difference survives -- a mean of four numbers is a mean of four
    numbers -- and only the spread, its separation verdict and its p-value are
    withheld.

    **What is being resampled depends on the criterion, so the caveat does too.**
    Each side declares the criterion its sender set was selected under and the
    record says what that population is.  The fixed caveat this function used to
    publish described a threshold-selected population on every run, including the
    six exhaustive arms of EXP-R2-071 where the population was the whole head
    grid; the two sides may also differ, since an arm with no head above threshold
    enters on a top-k fallback while its comparator does not.
    """

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        raise ValueError("a bootstrap needs at least two heads per side")
    if resamples < 100:
        raise ValueError("too few bootstrap resamples to read a percentile interval")
    for side, criterion in (("left", left_criterion), ("right", right_criterion)):
        if criterion not in RESAMPLED_POPULATION:
            raise ValueError(
                f"the {side} sender set declares criterion {criterion!r}, which has "
                f"no declared population; declared: {sorted(RESAMPLED_POPULATION)}"
            )
    population = {
        "left": {
            "sender_criterion": str(left_criterion),
            "what_is_resampled": RESAMPLED_POPULATION[left_criterion],
        },
        "right": {
            "sender_criterion": str(right_criterion),
            "what_is_resampled": RESAMPLED_POPULATION[right_criterion],
        },
    }
    floor = bootstrap_unit_floor(int(min(a.size, b.size)), minimum_units=minimum_units)
    if floor["degenerate"]:
        return {
            "difference": _finite(float(a.mean() - b.mean()), "bootstrap difference"),
            "spread_low": None,
            "spread_high": None,
            "separated_across_heads": None,
            "p_two_sided_uncorrected": None,
            "resampling_unit": "sender_head",
            "resampled_population": population,
            "is_a_sampling_confidence_interval": False,
            "interval_caveat": (
                "no spread is published: "
                f"{floor['degenerate_reason']}"
            ),
            "multiplicity": "no interval was produced, so there is nothing to correct",
            **floor,
            "n_left": int(a.size),
            "n_right": int(b.size),
            "resamples": int(resamples),
            "seed": int(seed),
        }
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        draws[index] = rng.choice(a, size=a.size, replace=True).mean() - rng.choice(
            b, size=b.size, replace=True
        ).mean()
    low, high = np.percentile(draws, [2.5, 97.5])
    tail = float(min((draws <= 0.0).mean(), (draws >= 0.0).mean()))
    return {
        "difference": _finite(float(a.mean() - b.mean()), "bootstrap difference"),
        "spread_low": _finite(float(low), "bootstrap spread low"),
        "spread_high": _finite(float(high), "bootstrap spread high"),
        "separated_across_heads": bool(low > 0.0 or high < 0.0),
        "p_two_sided_uncorrected": _finite(min(1.0, 2.0 * tail), "bootstrap p"),
        "resampling_unit": "sender_head",
        "resampled_population": population,
        "is_a_sampling_confidence_interval": False,
        "interval_caveat": (
            "each side's heads are that arm's whole population under the criterion "
            "named in resampled_population, not a sample of one; this interval "
            "describes head-to-head heterogeneity within those populations and "
            "carries no uncertainty from the probe records, which are the design's "
            "only real sampling unit"
        ),
        "multiplicity": (
            "p_two_sided_uncorrected is uncorrected; the caller runs this once per "
            "quantity per criterion per arm pair and must record the comparison "
            "count if it is to be corrected"
        ),
        **floor,
        "n_left": int(a.size),
        "n_right": int(b.size),
        "resamples": int(resamples),
        "seed": int(seed),
    }
