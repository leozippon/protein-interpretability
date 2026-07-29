"""What a transfer measurement scores: which depth, which window, which targets.

Four decisions sit upstream of every number this programme reports, and each is
declared exactly once here:

``analysis_layer`` / ``analysis_layers``
    the absolute layer a *relative depth* names on an arm of a given size;
``source_layers_for_target``
    the layers a window of a given width spans, ending at a target layer;
``target_rule`` / ``sequence_target_mask``
    which next-token positions belong to the cohort rather than to the prompt;
``per_sequence_scores`` / ``aggregate_variant``
    how those positions turn into a cross-entropy, a KL and an agreement rate.

Two of these were duplicated before EXP-R2-066, and both duplicates disagreed
with their originals.

**Depth.** ``src/transfer/relational.py`` carried a second depth conversion,
``int(round(fraction * (n_layer - 1)))``. Python's ``round`` is banker's
rounding, so at an exact half it goes to the *even* neighbour while
:func:`analysis_layer`'s ``floor(x + 0.5)`` goes up. On this panel they disagree
for exactly one cell -- the 27-layer ProGen2 arms at relative depth 0.25, where
``0.25 * 26 = 6.5`` is exact in binary: ``analysis_layer`` says layer 7 and the
duplicate said layer 6. ``09_probe_and_erasure.py`` reached the duplicate and
``02``, ``03`` and ``08`` reached the original, so two stages of one campaign
disagreed about what depth 0.25 meant on the two protein arms carrying the
protein-side corpus contrast. See the EXP-R2-066 log entry for the blast radius.

**Targets.** :func:`sequence_target_mask` used to dispatch on the arm's *name*:
it stripped a conditioning prompt for the literal string ``"zymctrl"`` and, for
every other name, discarded the boundary token ids it was handed without a word.
An EC-conditioned arm named anything else would have had its EC tag, separator
and terminator scored as cohort content -- the positions the model predicts most
confidently -- which lowers clean cross-entropy and raises context information,
with no exception anywhere. ``src/transfer/budget.py`` carried a guard against
that (``require_boundary_masking_support``), and ``src/transfer/probes.py``
worked around it by synthesising fake names (``f"{arm.name}_plain"``) to select a
rule. The rule is now a parameter, derived from ``ArmSpec.input_format``, so the
guard and the fake names are both gone and a second EC-conditioned arm is handled
correctly by construction rather than refused by an allow-list.

Vendored from ``src/revision/dictionary_fidelity.py``, whose remaining contents
(windowed-transcoder encode/decode, checkpoint identity, the frozen fidelity
spec) served the retired CLT / dictionary-qualification scope.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ depth grid


def analysis_layer(n_layers: int, fraction: float) -> int:
    """The absolute layer index that relative depth ``fraction`` names.

    Round half up, deliberately: ``round()`` is round-half-to-even and turns an
    exact half into a value that depends on the parity of the neighbour, which is
    not a property anyone intends to compare arms on.
    """

    if n_layers < 1 or not 0 <= fraction <= 1:
        raise ValueError("invalid layer count or analysis-layer fraction")
    return int(math.floor(fraction * (n_layers - 1) + 0.5))


def analysis_layers(n_layers: int, fractions: Sequence[float]) -> list[int]:
    """The sorted, deduplicated absolute layers a depth grid names on this arm."""

    if not fractions:
        raise ValueError("at least one layer fraction is required")
    return sorted({analysis_layer(n_layers, float(value)) for value in fractions})


def source_layers_for_target(
    target_layer: int, *, n_layers: int, window: int
) -> tuple[int, ...]:
    """The layers a width-``window`` source window spans, ending at ``target_layer``.

    Clipped at layer 0 rather than shifted, so a window anchored near the bottom
    of a model is narrower than requested instead of covering layers above its
    anchor. Callers that care record both endpoints in the scope name.
    """

    if not 0 <= target_layer < n_layers or window < 1:
        raise ValueError("invalid target layer or decoder window")
    return tuple(range(max(0, target_layer - window + 1), target_layer + 1))


# --------------------------------------------------------------- scored targets

#: ``all_valid``
#:     every next-token position where both the source and the target token are
#:     real rather than padding.
#: ``between_boundaries``
#:     ``all_valid`` intersected with the span strictly between the conditioning
#:     prompt's ``<start>`` and its ``<end>``, so neither the prompt nor the
#:     terminator is scored as cohort content.
TARGET_RULES = ("all_valid", "between_boundaries")

#: Input formats whose rendering prepends a conditioning prompt that is *not*
#: cohort content. Derived from ``ArmSpec.input_format`` rather than from an arm
#: name, which is what makes a future EC-conditioned arm correct by default.
CONDITIONED_INPUT_FORMATS = frozenset({"ec_conditioned"})


def target_rule(input_format: str, *, ec_conditioning: str = "native") -> str:
    """The masking rule an arm rendered in ``input_format`` is scored under.

    ``ec_conditioning="unconditioned"`` means the rendering carries no tag and so
    no boundaries; it takes the plain rule even on a conditioned arm. That mode
    is off the arm's training distribution and is measured as such (EXP-R2-034
    prices the tag at 1.73 nats), which is a separate fact from how it is scored.
    """

    if input_format in CONDITIONED_INPUT_FORMATS and ec_conditioning != "unconditioned":
        return "between_boundaries"
    return "all_valid"


def sequence_target_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    rule: str,
    start_token_id: int | None = None,
    end_token_id: int | None = None,
) -> torch.Tensor:
    """Return a mask over next-token targets belonging to the scored cohort.

    Column ``q`` of the returned mask governs the prediction of input token
    ``q + 1``, so it has one fewer column than ``input_ids``.
    """

    if rule not in TARGET_RULES:
        raise ValueError(f"unknown target rule {rule!r}; rules are {list(TARGET_RULES)}")
    if (
        input_ids.ndim != 2
        or attention_mask.shape != input_ids.shape
        or input_ids.shape[1] < 2
    ):
        raise ValueError("invalid token or attention-mask shape")
    valid = attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
    if rule == "all_valid":
        if start_token_id is not None or end_token_id is not None:
            raise ValueError(
                "boundary token ids were supplied under the 'all_valid' rule, which "
                "ignores them; pass rule='between_boundaries' or drop the ids"
            )
        return valid
    if start_token_id is None or end_token_id is None:
        raise ValueError("the 'between_boundaries' rule requires start and end token IDs")
    result = torch.zeros_like(valid)
    for row in range(input_ids.shape[0]):
        ids = input_ids[row]
        starts = torch.nonzero(ids == start_token_id, as_tuple=False).flatten()
        ends = torch.nonzero(ids == end_token_id, as_tuple=False).flatten()
        if starts.numel() != 1 or ends.numel() != 1 or ends[0] <= starts[0] + 1:
            raise ValueError("conditioned prompt lacks one valid start/end boundary")
        # Target column q predicts input token q+1. Keep content targets strictly
        # after <start> and strictly before <end>.
        result[row, int(starts[0]) : int(ends[0]) - 1] = True
    return result & valid


# ------------------------------------------------------------------- scoring


def per_sequence_scores(
    clean_logits: torch.Tensor,
    variant_logits: torch.Tensor,
    input_ids: torch.Tensor,
    target_mask: torch.Tensor,
) -> list[dict[str, float | int]]:
    """Per-sequence sums, so the aggregate can be token-weighted downstream.

    Sums rather than means: a per-sequence mean averaged over sequences weights a
    40-residue protein like a 2000-residue one, and the panel's cohorts are not
    length-matched across arms.
    """

    if (
        clean_logits.shape != variant_logits.shape
        or clean_logits.ndim != 3
        or input_ids.shape != clean_logits.shape[:2]
        or target_mask.shape != (input_ids.shape[0], input_ids.shape[1] - 1)
    ):
        raise ValueError("logit, token and target-mask shapes disagree")
    clean_logp = F.log_softmax(clean_logits[:, :-1].float(), dim=-1)
    variant_logp = F.log_softmax(variant_logits[:, :-1].float(), dim=-1)
    clean_p = clean_logp.exp()
    targets = input_ids[:, 1:]
    clean_nll = -clean_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    variant_nll = -variant_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    kl = (clean_p * (clean_logp - variant_logp)).sum(-1)
    agreement = clean_logp.argmax(-1) == variant_logp.argmax(-1)
    rows: list[dict[str, float | int]] = []
    for index in range(input_ids.shape[0]):
        mask = target_mask[index]
        count = int(mask.sum())
        if count < 1:
            raise ValueError("sequence has no scored next-token targets")
        rows.append(
            {
                "token_count": count,
                "clean_nll_sum": float(clean_nll[index][mask].sum()),
                "variant_nll_sum": float(variant_nll[index][mask].sum()),
                "kl_sum": float(kl[index][mask].sum()),
                "argmax_agreement_count": int(agreement[index][mask].sum()),
            }
        )
    return rows


def aggregate_variant(rows: Sequence[Mapping[str, float | int]]) -> dict[str, float]:
    """Token-weighted aggregate over :func:`per_sequence_scores` rows."""

    if not rows:
        raise ValueError("cannot aggregate an empty sequence set")
    tokens = sum(int(row["token_count"]) for row in rows)
    if tokens < 1:
        raise ValueError("aggregate contains no scored targets")
    return {
        "clean_ce_nats": sum(float(row["clean_nll_sum"]) for row in rows) / tokens,
        "variant_ce_nats": sum(float(row["variant_nll_sum"]) for row in rows) / tokens,
        "clean_to_variant_kl_nats": sum(float(row["kl_sum"]) for row in rows) / tokens,
        "argmax_agreement": sum(int(row["argmax_agreement_count"]) for row in rows)
        / tokens,
        "scored_tokens": tokens,
        "sequences": len(rows),
    }
