"""Does a text-derived concept direction causally steer the same checkpoint's protein mode?

D3.g's claim-bearing stage. The criteria are **frozen** in ``docs/EXPERIMENT_LOG.md``
as EXP-R2-213 (A36-0 to A36-7) and this module implements them and nothing else: every
threshold below is quoted from that entry rather than chosen here, and
``scripts/transfer/36_concept_injection.py`` requires each as a never-defaulted flag and
validates it against the frozen value.

**The claim, stated once.** A direction estimated in a checkpoint's TEXT mode, from
curated descriptions masked of the concept's own surface forms, moves that same
checkpoint's PROTEIN mode in the concept-consistent direction; the movement is graded in
the injection coefficient, specific to the injected concept, and visible to an
instrument that is not the model.

**The estimand and its sign, because the sign is the criterion.** Readout A is the
graded NLL shift of A36-3, in the pre-registration's own convention:

    Delta(alpha) = [NLL_bearing(alpha) - NLL_bearing(0)]
                 - [NLL_nonbearing(alpha) - NLL_nonbearing(0)]

in nats per scored token. It is an **NLL shift**, so the predicted direction is
``Delta(alpha) < 0`` for ``alpha > 0`` -- a working concept vector makes the bearing
sequences *cheaper* -- ``Delta(0) = 0`` by construction, and the graded criterion is a
Spearman ``rho(alpha, Delta) <= -0.8``. Nothing here reports a "benefit" with the
opposite sign; :func:`delta_nll_shift` carries the frozen convention and the field is
named for it.

**Why the name masking is the whole design.** Without it a shared concept direction and
a label-token lookup are the same object. C34-1 makes the masking a refusal upstream;
:func:`assert_descriptions_masked` re-reads the surviving text here, before a checkpoint
is loaded, because L15 is the record of what the alternative costs: D3.b's sole positive
protein cell was ZymCTRL's EC conditioning tag and it dissolved to nothing once the tag
was held fixed.

**What is injected, and where.** ``alpha * sigma * u`` at the input of the per-layer
feed-forward -- ``concept_alignment.REPRESENTATION_SITE``, the tensor the direction and
its sigma were estimated on. That correspondence is why the site is not a choice: the
post-attention RMSNorm rescales the vector and applies a learned per-dimension gain, so
a residual-stream write of the same coordinates would not be alpha population-sigmas
along the concept direction at the site sigma was measured, and the graded criterion
would lose its units. The cost is locality, stated rather than hidden: the write is
consumed by one block's feed-forward.

**Which positions.** Every content position, which is the mask
``mode_representations`` pooled the representation over (in protein mode the token run
that spells the sequence, so the instruction prefix, the delimiters and the
beginning-of-sequence token stay out). A position-level write is not on offer, and that
is L31 rather than a simplification: this family writes residues into an unmodified
LLaMA-2 SentencePiece vocabulary, single substitutions leave mutant and wild type
token-aligned on only 47.0-54.5% of instances on the panel's BPE arm, and the survivors
are the BPE-stable subset rather than a random one.

**Hook hygiene is ported from DAS and not re-decided.** :func:`invariants` is A36-0 and
is ``das.invariants`` adapted to a masked multi-position write: a null patch must move
the logits by at most a tolerance, and a large patch must move them. The second is the
one that matters -- a hook that fails to bind passes every null test while silently
measuring an unpatched model -- and its perturbation is a seeded *random* direction
rather than a constant, because the normalisation in front of the site very nearly
annihilates a constant (measured on gpt2-large at layer 9: a uniform +10 across 1280
dimensions moves the logits by 1.9e-5, a random direction of the same norm by 0.62).
Nothing here trains a subspace; distributed alignment search is closed, and what is
reused is that stage's hook discipline, not its search.

**Every threshold, and where it comes from.**

``A36-1`` :data:`COHERENCE_PRIMARY_BOUND` 0.25 nats/token of inflation on the
    **non-bearing** held-out sequences, re-read at :data:`COHERENCE_SENSITIVITY_BOUNDS`.
    Anchored: these checkpoints' protein-mode context information is 0.5505 and 0.5215
    nats/token, so the bound admits damage of at most half the directional signal the
    mode has. If no rung is admissible the readout is **closed** and the bound is never
    widened.
``A36-2`` the same direction in **text** mode must satisfy A36-3(a) and (b) at one
    admissible rung, or the run is **VOID** and the protein side is not read.
``A36-3`` (a) the 95% interval of ``Delta(alpha)`` excludes zero at one admissible
    ``alpha > 0``; (b) ``|Delta| >= 2 x`` the 95th percentile of the norm-matched
    random-direction control at the same alpha and site, over at least
    :data:`MINIMUM_CONTROL_DIRECTIONS` **distinct directions**; (c) Spearman
    ``rho(alpha, Delta) <= -0.8`` over all nine rungs, with the sign reversing below
    zero.
``A36-4`` each concept's own diagonal must exceed the **95th percentile of its own
    row's off-diagonal entries**. The mean diagonal minus mean off-diagonal is reported
    and may not substitute for it -- that substitution is L32 exactly.
``A36-5`` permuted-concept directions must **fail** A36-3; if one passes, the readout is
    void.
``A36-6`` readout B's attainability is read first: at ``alpha = 0`` the generator must
    annotate at a non-zero rate against Pfam-A, or readout B is void.

**One tension between the frozen clauses is recorded rather than resolved here.** A36-5
requires a permuted-label refit to fail A36-3, and A36-3(b)'s bar is set by *isotropic*
random directions. Those are not the same population: a permuted refit lies in the span
of the concept structure plus the representation cloud's own noise, so its component on a
concept direction is systematically larger than an isotropic vector's, which falls only
as one over the root of the width. The two clauses therefore pull against each other, and
the pull is stronger the lower the cloud's effective dimension and the fewer the concepts.
On the known-answer fixture this is measured rather than argued: at three concepts and
width 32 the worst of eight permuted refits carried 0.89 of the planted direction and
passed every clause of A36-3, and at eight concepts and width 128 none of thirty-two
draws passes. Every permuted draw's verdict and its distance from the bar reach the
artefact, because the campaign's margin is not the fixture's.

**Which checkpoints may enter** is not decided here.
``concept_alignment.assert_behavioural_read_permitted`` is the one declaration, and it
refuses ``Llama-2-7b-hf``'s protein mode on EXP-R2-152's measured bound -- reversing its
residues costs it -0.0013 nats, exactly no directional sequence information.
"""

from __future__ import annotations

import gzip
import hashlib
import re
import shutil
import subprocess
import tarfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from scipy import stats
from torch import nn

from . import concept_alignment as ca
from .io import sha256_file
from .replaceable import JointBlockLayout
from .statistics import MINIMUM_BOOTSTRAP_UNITS, paired_group_bootstrap

SCHEMA_VERSION = "r2_transfer_concept_injection_v1"

#: EXP-R2-213 is the pre-registration every threshold in this module is quoted from.
PRE_REGISTRATION = "EXP-R2-213"

#: A36-3's nine rungs, in units of sigma. Frozen: a stage that could choose its own
#: ladder could choose it after seeing which rungs worked.
FROZEN_ALPHA_LADDER: tuple[float, ...] = (-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0)

#: A36-1's primary coherence bound, in nats per scored token of inflation on the
#: non-bearing held-out sequences.
COHERENCE_PRIMARY_BOUND = 0.25

#: A36-1's sensitivity sweep. The verdict is re-read at each, and either the conclusion
#: is invariant across them or the dependence is itself the finding (Appendix B rule 17).
COHERENCE_SENSITIVITY_BOUNDS: tuple[float, ...] = (0.10, 0.25, 0.50)

#: A36-3(c). Negative because Delta is an NLL shift: a working concept vector at
#: alpha > 0 REDUCES the bearing sequences' cross-entropy.
FROZEN_SPEARMAN_CEILING = -0.8

#: A36-3(b)'s multiple of the norm-matched random-direction control's 95th percentile.
FROZEN_RANDOM_NULL_MARGIN = 2.0

#: A36-3(b) and A36-5: distinct *directions*, not repeats of one and not more positions.
MINIMUM_CONTROL_DIRECTIONS = 8

#: A36-4's rule, as a one-element tuple so a second rule cannot arrive without being
#: named. A row mean may not substitute for the percentile.
SPECIFICITY_RULES = ("row_offdiagonal_p95",)

#: Which admissible rung a concept's operating point is, declared rather than picked
#: after the fact: the smallest admissible positive rung whose interval excludes zero in
#: the predicted direction. Choosing the strongest rung instead would be selection on
#: the outcome.
OPERATING_ALPHA_RULES = ("smallest_admissible_positive_significant",)

#: The one site a concept direction may be written at, and it is
#: ``concept_alignment.REPRESENTATION_SITE`` -- the tensor the direction and its sigma
#: were estimated on.
INJECTION_SITE = ca.REPRESENTATION_SITE
INJECTION_SITES = (INJECTION_SITE,)

INJECTION_SITE_NOTE = (
    "the input of the per-layer feed-forward, which is exactly the tensor "
    "concept_alignment.mode_representations pools and concept_vector's sigma is "
    "measured on. The correspondence is the reason there is no second site: the "
    "post-attention RMSNorm rescales the vector and applies a learned per-dimension "
    "gain, so a unit direction after it is not a unit direction before it, and a "
    "residual-stream write of alpha*sigma*u would not be alpha population-sigmas "
    "along the concept direction at the site sigma was measured. What it costs is "
    "locality: the write is consumed by one block's feed-forward"
)

DELTA_DEFINITION = (
    "Delta(alpha) = [NLL_bearing(alpha) - NLL_bearing(0)] - [NLL_nonbearing(alpha) - "
    "NLL_nonbearing(0)], in nats per scored token. It is an NLL SHIFT, so the predicted "
    "direction is Delta < 0 for alpha > 0 and A36-3(c)'s Spearman ceiling is negative. "
    "Delta(0) = 0 by construction"
)

#: Refused outright. The estimand is a cross-entropy difference of order 0.01-0.1 nats
#: between two conditions that differ only by the injected delta, so the rounding does
#: not cancel between them (Appendix B rule 15b).
REFUSED_DTYPES = ("bfloat16", "float16")

REQUIRED_READABLE_MODES = ("text", "protein")


# ------------------------------------------------------------------- refusals


def require_behavioural_modes(checkpoint: Path | str) -> dict[str, Any]:
    """Refuse a checkpoint that may not carry a behavioural read.

    Delegated to ``concept_alignment.assert_behavioural_read_permitted``, the single
    declaration of what may be claimed about a checkpoint's mode; an undeclared
    checkpoint is refused there too, because unqualified is not the same as
    measurable. Both modes are passed to it rather than the protein one alone --
    A36-2 makes the text mode load-bearing -- with the caveat that the declaration
    constrains protein mode only, so the text call is a no-op today and will start
    refusing if a text-mode bound is ever added to it.
    """

    for mode in REQUIRED_READABLE_MODES:
        ca.assert_behavioural_read_permitted(checkpoint, mode)
    return {
        "checkpoint": str(checkpoint),
        "modes": list(REQUIRED_READABLE_MODES),
        "status": ca.protein_mode_behavioural_status(checkpoint),
        "declaration": "src.transfer.concept_alignment.PROTEIN_MODE_BEHAVIOURAL_STATUS",
    }


def require_full_precision(handle: Any) -> str:
    """Refuse half precision, for the reason :data:`REFUSED_DTYPES` records."""

    observed = sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in handle.model.parameters()
            if parameter.is_floating_point()
        }
    )
    refused = [dtype for dtype in observed if dtype in REFUSED_DTYPES]
    if refused:
        raise ValueError(
            f"{getattr(handle, 'name', 'this handle')} is loaded in {refused}. The "
            "estimand is a cross-entropy difference of order 0.01-0.1 nats/token between "
            "two conditions that differ only by the injected delta, so the quantisation "
            "does not cancel between them (Appendix B rule 15b). Load in float32"
        )
    return ",".join(observed)


def assert_descriptions_masked(
    records: Sequence[Mapping[str, Any]], term: str
) -> dict[str, Any]:
    """Refuse a cohort whose masked descriptions still spell the concept's term.

    C34-1's refusal, re-checked here on the surviving text. It asks a different
    question from the cohort stage's own record of what it removed: whether what
    is left still contains the term. A direction estimated from text that does is
    a direction that reads a word rather than a concept, and every downstream
    control is passed by a label-token lookup exactly as it is passed by a
    concept.

    Matching is whole-word and case-insensitive with internal whitespace relaxed,
    so a substring inside a longer word does not raise and a capitalised
    occurrence does.
    """

    needle = str(term).strip()
    if not needle:
        raise ValueError("a concept with an empty term cannot be checked for masking")
    pattern = re.compile(
        r"\b" + r"\s+".join(re.escape(part) for part in needle.split()) + r"\b",
        re.IGNORECASE,
    )
    offenders = [
        str(record.get("accession", index))
        for index, record in enumerate(records)
        if pattern.search(str(record["description_masked"]))
    ]
    if offenders:
        raise ValueError(
            f"{len(offenders)} masked description(s) still carry the term {needle!r}, "
            f"e.g. {offenders[:5]}. A concept direction estimated from this text would "
            "be indistinguishable from a label-token lookup, which is the failure mode "
            "limitation L15 records for D3.b's only positive protein arm (C34-1)"
        )
    return {
        "term": needle,
        "n_checked": len(records),
        "rule": "whole-word, case-insensitive, internal whitespace relaxed, over "
        "description_masked",
    }


# ----------------------------------------------------------------- the direction


@dataclass(frozen=True)
class ConceptDirection:
    """A unit direction and the population scale that makes its coefficient readable."""

    concept: str
    layer: int
    direction: np.ndarray
    sigma: float
    provenance: str

    def __post_init__(self) -> None:
        vector = np.asarray(self.direction, dtype=np.float64)
        if vector.ndim != 1 or vector.size < 2:
            raise ValueError("a concept direction must be a one-dimensional vector")
        if not np.isfinite(vector).all():
            raise ValueError(f"{self.concept}: the direction carries a non-finite entry")
        norm = float(np.linalg.norm(vector))
        if abs(norm - 1.0) > 1e-6:
            raise ValueError(
                f"{self.concept}: the direction has norm {norm:.6f} rather than 1. The "
                "coefficient is in units of sigma along a UNIT direction, so a "
                "non-unit vector silently rescales every rung of the ladder"
            )
        if not np.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError(
                f"{self.concept}: sigma is {self.sigma}, and the population scale along "
                "the direction is the unit the coefficient is measured in"
            )

    @classmethod
    def from_concept_vector(
        cls, vector: Any, *, concept: str, layer: int, provenance: str
    ) -> "ConceptDirection":
        """Wrap ``concept_alignment.ConceptVector`` without re-deriving anything."""

        return cls(
            concept=concept,
            layer=int(layer),
            direction=np.asarray(vector.direction, dtype=np.float64),
            sigma=float(vector.sigma),
            provenance=provenance,
        )

    @property
    def d_model(self) -> int:
        return int(np.asarray(self.direction).size)

    def delta(self, alpha: float, *, device: Any, dtype: torch.dtype) -> torch.Tensor:
        """``alpha * sigma * u`` on the model's device."""

        vector = np.asarray(self.direction, dtype=np.float64) * float(alpha) * self.sigma
        return torch.tensor(vector, device=device, dtype=dtype)

    def record(self) -> dict[str, Any]:
        return {
            "concept": self.concept,
            "layer": int(self.layer),
            "d_model": self.d_model,
            "sigma": float(self.sigma),
            "provenance": self.provenance,
            "site": INJECTION_SITE,
        }


# ------------------------------------------------------------- the intervention


def feed_forward_modules(handle: Any) -> list[nn.Module]:
    """The per-layer feed-forward modules, from the handle's declared block layout.

    Resolved from ``JointReplaceable.layout`` rather than by searching for an
    attribute called ``mlp``: ``replaceable.joint_block_layout`` refuses an
    undeclared architecture precisely so that duck-typing cannot measure a
    different object without saying so.
    """

    layout = handle.layout
    if not isinstance(layout, JointBlockLayout):
        raise TypeError(
            f"{type(handle).__name__} carries no declared JointBlockLayout, so the "
            "site this stage writes at cannot be resolved"
        )
    node: Any = handle.model
    for attribute in layout.layer_list:
        node = getattr(node, attribute, None)
        if node is None:
            raise TypeError(
                f"this checkpoint has no {'.'.join(layout.layer_list)}, so the declared "
                "block layout does not describe it"
            )
    modules: list[nn.Module] = []
    for index, block in enumerate(node):
        module = getattr(block, layout.feed_forward, None)
        if module is None:
            raise TypeError(
                f"layer {index} has no {layout.feed_forward!r}, so the declared block "
                "layout does not describe it"
            )
        modules.append(module)
    return modules


def install_injection(
    handle: Any, *, layer: int, delta: torch.Tensor, position_mask: torch.Tensor
) -> Any:
    """Add ``delta`` to the feed-forward's input at the masked positions of one layer.

    ``position_mask`` is ``[batch, tokens]`` and is the batch's own content mask.
    It is a mask rather than an index because the write is over every content
    position of a right-padded batch whose rows have different spans, and because
    L31 makes a position-level write undefined on part of any protein cohort.

    The handle is returned rather than the call being wrapped, so that the
    caller's ``try/finally`` has the same shape as ``probes._install_hook``'s and
    ``das.patched_logits``'.
    """

    modules = feed_forward_modules(handle)
    if not 0 <= layer < len(modules):
        raise ValueError(f"layer {layer} is outside this backbone's 0..{len(modules) - 1}")
    if position_mask.ndim != 2:
        raise ValueError("the position mask must be [batch, tokens]")
    if delta.ndim != 1:
        raise ValueError("the injected delta must be a single d_model vector")
    if int(position_mask.sum()) < 1:
        raise ValueError("a batch with no content position has nothing to write at")

    def hook(module: nn.Module, args: tuple, kwargs: dict[str, Any]):
        if not args or not isinstance(args[0], torch.Tensor):
            raise TypeError(
                f"layer {layer}'s feed-forward was called without a positional input, "
                "so there is nothing for this intervention to address"
            )
        hidden = args[0]
        if hidden.ndim != 3 or hidden.shape[-1] != delta.shape[0]:
            raise TypeError(
                f"the site at layer {layer} is {tuple(hidden.shape)}, which does not "
                f"admit a {tuple(delta.shape)} write; the direction was estimated at a "
                "different width"
            )
        if position_mask.shape != hidden.shape[:2]:
            raise ValueError(
                f"the position mask {tuple(position_mask.shape)} does not match the "
                f"site {tuple(hidden.shape[:2])}; the batch changed under the hook"
            )
        addend = position_mask.to(hidden.dtype).unsqueeze(-1) * delta.to(hidden.dtype)
        return (hidden + addend,) + tuple(args[1:]), kwargs

    return modules[layer].register_forward_pre_hook(hook, with_kwargs=True)


@dataclass(frozen=True)
class PreparedBatch:
    """One rendered, padded batch and the cohort identities of its rows."""

    tensors: Mapping[str, torch.Tensor]
    record_ids: tuple[str, ...]
    dup_groups: tuple[str, ...]

    def __post_init__(self) -> None:
        rows = int(self.tensors["input_ids"].shape[0])
        if len(self.record_ids) != rows or len(self.dup_groups) != rows:
            raise ValueError("record ids and dup groups must align with the batch rows")
        if int(self.tensors["target_mask"].sum()) < 1:
            raise ValueError("a batch with no scored target cannot be scored")
        if int(self.tensors["content_mask"].sum()) < 1:
            raise ValueError("a batch with no content position cannot be intervened on")


def prepare_batches(
    handle: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    id_field: str = "accession",
) -> tuple[PreparedBatch, ...]:
    """Render and batch the cohort through the handle's own declarations.

    The mode decides the input: sequences in protein mode, masked descriptions in
    text mode. Neither the rendering nor the scored span nor the content mask is
    decided here -- ``JointReplaceable`` owns all three, which is what keeps this
    stage's positions identical to the ones the representation was pooled over.
    """

    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if not records:
        raise ValueError("an empty cohort has nothing to score")
    field = "sequence" if handle.mode == "protein" else "description_masked"
    batches: list[PreparedBatch] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        rendered = handle.render([str(record[field]) for record in chunk])
        tensors = handle.batch(rendered)
        handle.forget_rendered()
        batches.append(
            PreparedBatch(
                tensors=tensors,
                record_ids=tuple(str(record[id_field]) for record in chunk),
                dup_groups=tuple(str(record["dup_group"]) for record in chunk),
            )
        )
    return tuple(batches)


def invariants(
    handle: Any,
    batch: PreparedBatch,
    *,
    layer: int,
    scale: float,
    seed: int = 20260818,
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    """A36-0: refuse to measure unless the write is doing what it claims.

    Ported from ``das.invariants`` and kept in its shape because both checks are
    load-bearing and one is not obvious. The null patch says the intervention is
    the identity when it should be. The **positive** control says the hook is
    bound at all: an unbound hook passes every null invariant while silently
    measuring an unpatched model.

    The perturbation is a seeded *random* direction rather than a constant, for
    the reason ``das.invariants`` documents from measurement -- the normalisation
    in front of this site rescales a constant away, so a constant-vector control
    reports a correctly bound hook as unbound.
    """

    if scale <= 0.0:
        raise ValueError("the invariant's perturbation scale must be positive")
    parameter = next(handle.model.parameters())
    generator = torch.Generator(device="cpu").manual_seed(seed)
    direction = torch.randn(int(handle.width), generator=generator, dtype=torch.float32)
    direction = (direction / direction.norm() * scale).to(
        device=handle.device, dtype=parameter.dtype
    )
    zero = torch.zeros_like(direction)
    mask = batch.tensors["content_mask"]

    with torch.no_grad():
        clean = handle.run(dict(batch.tensors)).logits.float()
        null = _injected_logits(handle, batch, layer=layer, delta=zero).float()
        moved = _injected_logits(handle, batch, layer=layer, delta=direction).float()
    null_gap = float((clean - null).abs().max())
    moved_gap = float((clean - moved).abs().max())
    record = {
        "criterion": "A36-0",
        "layer": int(layer),
        "site": INJECTION_SITE,
        "null_patch_max_logit_gap": null_gap,
        "perturbed_patch_max_logit_gap": moved_gap,
        "perturbation_norm": float(scale),
        "perturbation": "seeded random unit direction scaled to the declared norm; a "
        "constant vector is annihilated by the normalisation in front of this site and "
        "would report a bound hook as unbound",
        "tolerance": float(tolerance),
        "n_written_positions": int(mask.sum()),
    }
    if null_gap > tolerance:
        raise RuntimeError(
            f"writing a zero delta at layer {layer} ({INJECTION_SITE}) moved the logits "
            f"by {null_gap:.3g}, so the intervention is not the identity when it should "
            "be and every measured effect is partly the hook itself"
        )
    if moved_gap <= tolerance:
        raise RuntimeError(
            f"writing a large delta at layer {layer} ({INJECTION_SITE}) moved the logits "
            f"by {moved_gap:.3g}, which is within the null tolerance: the hook is not "
            "bound to the site. A null-only check cannot see this, which is why this "
            "positive control exists"
        )
    return record


def _injected_logits(
    handle: Any, batch: PreparedBatch, *, layer: int, delta: torch.Tensor
) -> torch.Tensor:
    installed = install_injection(
        handle, layer=layer, delta=delta, position_mask=batch.tensors["content_mask"]
    )
    try:
        return handle.run(dict(batch.tensors)).logits
    finally:
        installed.remove()


# ------------------------------------------------------- readout A: the NLL shift


@dataclass(frozen=True)
class ScoredResponse:
    """Per-record cross-entropy and predictive entropy under one condition."""

    record_ids: tuple[str, ...]
    dup_groups: tuple[str, ...]
    nll_per_token: np.ndarray
    entropy_per_token: np.ndarray
    scored_tokens: np.ndarray


@torch.no_grad()
def scored_response(
    handle: Any,
    batches: Sequence[PreparedBatch],
    *,
    layer: int,
    delta: torch.Tensor | None,
) -> ScoredResponse:
    """Per-record NLL and predictive entropy over the scored targets, in nats/token.

    ``delta=None`` is the uninjected reference. The log-softmax is taken in
    float32 whatever the parameters are stored in, which is the convention every
    scored measurement in this repository uses.
    """

    ids: list[str] = []
    groups: list[str] = []
    nll: list[float] = []
    entropy: list[float] = []
    counts: list[int] = []
    for batch in batches:
        installed = (
            None
            if delta is None
            else install_injection(
                handle,
                layer=layer,
                delta=delta,
                position_mask=batch.tensors["content_mask"],
            )
        )
        try:
            logits, targets, target_mask = handle.scored_logits(dict(batch.tensors))
        finally:
            if installed is not None:
                installed.remove()
        logp = torch.log_softmax(logits.float(), dim=-1)
        token_nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        token_entropy = -(logp.exp() * logp).sum(-1)
        for row in range(logits.shape[0]):
            mask = target_mask[row]
            count = int(mask.sum())
            if count < 1:
                raise ValueError(
                    f"{batch.record_ids[row]}: no scored target, so this record "
                    "contributes nothing and the cohort is wrong"
                )
            ids.append(batch.record_ids[row])
            groups.append(batch.dup_groups[row])
            nll.append(float(token_nll[row][mask].mean()))
            entropy.append(float(token_entropy[row][mask].mean()))
            counts.append(count)
    return ScoredResponse(
        record_ids=tuple(ids),
        dup_groups=tuple(groups),
        nll_per_token=np.asarray(nll, dtype=np.float64),
        entropy_per_token=np.asarray(entropy, dtype=np.float64),
        scored_tokens=np.asarray(counts, dtype=np.int64),
    )


def _delta_metric(truth: np.ndarray, predicted: np.ndarray) -> float:
    """A36-3's Delta, in the frozen sign convention: bearing minus non-bearing."""

    bearing = np.asarray(truth).astype(bool)
    if bearing.all() or (~bearing).all():
        return float("nan")
    return float(predicted[bearing].mean() - predicted[~bearing].mean())


def _paired_shift(
    baseline: ScoredResponse, injected: ScoredResponse, bearing: Sequence[bool]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if baseline.record_ids != injected.record_ids:
        raise ValueError(
            "the injected and uninjected passes scored different records, so the "
            "within-record difference is not paired"
        )
    flags = np.asarray(list(bearing), dtype=bool)
    if flags.shape != baseline.nll_per_token.shape:
        raise ValueError("the bearing flags do not align with the scored records")
    if flags.all() or (~flags).all():
        raise ValueError(
            "this cohort carries only one class of this concept, so Delta is undefined"
        )
    return injected.nll_per_token - baseline.nll_per_token, flags, np.asarray(
        baseline.dup_groups
    )


def per_side_group_counts(
    dup_groups: Sequence[str], bearing: Sequence[bool]
) -> dict[str, int]:
    """Distinct near-duplicate groups on each side of a concept.

    Counted per side because EXP-R2-213's floor is per side: eight groups in total
    with one on the bearing side is not eight units of the comparison that is
    being made.
    """

    groups = np.asarray(list(dup_groups))
    flags = np.asarray(list(bearing), dtype=bool)
    return {
        "bearing": int(np.unique(groups[flags]).size),
        "non_bearing": int(np.unique(groups[~flags]).size),
    }


def require_per_side_group_floor(
    dup_groups: Sequence[str], bearing: Sequence[bool], *, label: str
) -> dict[str, int]:
    counts = per_side_group_counts(dup_groups, bearing)
    short = {side: value for side, value in counts.items() if value < MINIMUM_BOOTSTRAP_UNITS}
    if short:
        raise ValueError(
            f"{label}: {short} near-duplicate group(s) below the "
            f"{MINIMUM_BOOTSTRAP_UNITS}-unit floor, which EXP-R2-213 states per SIDE. A "
            "comparison below it is not reported with a wider interval; it is not "
            "reported"
        )
    return counts


def delta_point(
    baseline: ScoredResponse, injected: ScoredResponse, bearing: Sequence[bool]
) -> float:
    """Delta(alpha) alone, with no interval.

    Used for a *control* draw, whose per-draw interval nothing reads: A36-3(b) is a
    percentile over directions, so the draws contribute point values and the
    economy removes no control and no draw (Appendix B rule 37).
    """

    shift, flags, _ = _paired_shift(baseline, injected, bearing)
    return _delta_metric(flags.astype(int), shift)


def delta_nll_shift(
    baseline: ScoredResponse,
    injected: ScoredResponse,
    bearing: Sequence[bool],
    *,
    seed: int,
    n_bootstrap: int,
    label: str = "delta",
) -> dict[str, Any]:
    """A36-3's Delta with its group-bootstrap interval.

    ``paired_group_bootstrap`` is used the way it is used everywhere else -- the
    near-duplicate group is the resampling unit -- with the right-hand prediction
    vector held at zero, so the reported ``difference`` is Delta itself and its
    interval is a percentile interval over group resamples. No resampler is
    declared here; this stage adds none.
    """

    shift, flags, groups = _paired_shift(baseline, injected, bearing)
    counts = require_per_side_group_floor(groups, flags, label=label)
    bootstrap = paired_group_bootstrap(
        flags.astype(int),
        shift,
        np.zeros_like(shift),
        groups,
        _delta_metric,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    return {
        "delta_nats_per_token": bootstrap["difference"],
        "delta_ci95": bootstrap["difference_ci95"],
        # Both readings reach the artefact. A36-3(a) as written says "the 95%
        # interval of Delta(alpha) excludes zero", and the pre-registration states
        # the predicted direction separately (Delta < 0 for alpha > 0). This stage
        # gates on the directional reading, because an interval excluding zero on
        # the WRONG side is an effect in the opposite direction and reporting it as
        # a pass would be indefensible -- and it records the direction-agnostic
        # reading beside it so that the choice is visible rather than silent.
        "excludes_zero_in_predicted_direction": bool(
            bootstrap["difference_ci95"][1] < 0.0
        ),
        "excludes_zero_either_direction": bool(
            bootstrap["difference_ci95"][1] < 0.0 or bootstrap["difference_ci95"][0] > 0.0
        ),
        "n_bearing": int(flags.sum()),
        "n_non_bearing": int((~flags).sum()),
        "n_groups_per_side": counts,
        "mean_shift_bearing": float(shift[flags].mean()),
        "mean_shift_non_bearing": float(shift[~flags].mean()),
        "bootstrap": {
            key: bootstrap[key]
            for key in ("n_bootstrap_requested", "n_finite_draws", "minimum_groups")
        },
        "estimand": DELTA_DEFINITION,
    }


def coherence_record(
    baseline: ScoredResponse, injected: ScoredResponse, bearing: Sequence[bool]
) -> dict[str, Any]:
    """A36-1: what the injection cost the NON-BEARING sequences.

    Non-bearing only, and that is the point: an inflation measured over the whole
    cohort could be lowered by the very effect being claimed, so the coherence
    floor is read on the sequences the concept is not supposed to help.
    """

    shift, flags, _ = _paired_shift(baseline, injected, bearing)
    weights = baseline.scored_tokens.astype(np.float64)
    entropy_shift = injected.entropy_per_token - baseline.entropy_per_token
    return {
        "criterion": "A36-1",
        "non_bearing_nll_inflation_nats_per_token": float(
            np.average(shift[~flags], weights=weights[~flags])
        ),
        "non_bearing_entropy_shift_nats_per_token": float(
            np.average(entropy_shift[~flags], weights=weights[~flags])
        ),
        "bearing_nll_inflation_nats_per_token": float(
            np.average(shift[flags], weights=weights[flags])
        ),
        "baseline_non_bearing_nll_nats_per_token": float(
            np.average(baseline.nll_per_token[~flags], weights=weights[~flags])
        ),
        "n_non_bearing": int((~flags).sum()),
        "rule": "a rung is admissible only where the non-bearing inflation is at most "
        "the declared bound; the bound is never widened after seeing which rungs it "
        "excludes",
    }


def admissible_alphas(
    coherence: Mapping[float, Mapping[str, Any]], *, max_nll_inflation: float
) -> tuple[float, ...]:
    """The rungs A36-1 admits at one bound."""

    if max_nll_inflation <= 0.0:
        raise ValueError("the coherence bound must be a positive nats/token inflation")
    return tuple(
        sorted(
            float(alpha)
            for alpha, record in coherence.items()
            if float(record["non_bearing_nll_inflation_nats_per_token"]) <= max_nll_inflation
        )
    )


def graded_record(deltas: Mapping[float, float]) -> dict[str, Any]:
    """A36-3(c): is Delta graded in alpha over the nine rungs, with a sign reversal?

    Computed over the whole ladder rather than over the admissible subset, because
    that is what the pre-registration states: gradedness is a property of the
    intervention's response and the coherence bound governs where the *effect* may
    be read, not where the response may be described. The admissible-subset
    correlation is reported beside it.
    """

    rungs = sorted(float(alpha) for alpha in deltas)
    if len(rungs) < 3:
        raise ValueError("a dose-response needs at least three rungs")
    values = [float(deltas[alpha]) for alpha in rungs]
    correlation: float | None = None
    p_value: float | None = None
    if len(set(values)) > 1:
        result = stats.spearmanr(rungs, values)
        if np.isfinite(result.statistic):
            correlation = float(result.statistic)
            p_value = float(result.pvalue)
    positive = [alpha for alpha in rungs if alpha > 0.0]
    negative = [alpha for alpha in rungs if alpha < 0.0]
    reversal = bool(
        positive
        and negative
        and max(deltas[alpha] for alpha in positive) < 0.0
        and min(deltas[alpha] for alpha in negative) > 0.0
    )
    steps = [values[index + 1] - values[index] for index in range(len(values) - 1)]
    return {
        "criterion": "A36-3(c)",
        "alphas": rungs,
        "deltas": values,
        "spearman": correlation,
        "spearman_p": p_value,
        "monotone_step_fraction": sum(1 for step in steps if step < 0.0) / len(steps),
        "sign_reverses_below_zero": reversal,
        "delta_at_zero": float(deltas.get(0.0, float("nan")))
        if 0.0 in deltas
        else None,
        "convention": DELTA_DEFINITION,
    }


def random_direction_control(values: Sequence[float]) -> dict[str, Any]:
    """A36-3(b): the norm-matched random-direction control at one alpha and site.

    Reported as a distribution over **distinct directions**, with the 95th
    percentile of ``|Delta|`` as the bar. The absolute value is taken because the
    criterion is written on ``|Delta(alpha)|``: a control direction that happens to
    move the bearing sequences the wrong way is still evidence about how large a
    move an arbitrary direction of this norm produces, and dropping its magnitude
    would make the bar easier.
    """

    sample = np.asarray([float(value) for value in values], dtype=np.float64)
    if sample.size < MINIMUM_CONTROL_DIRECTIONS:
        raise ValueError(
            f"{sample.size} random direction(s) is below the "
            f"{MINIMUM_CONTROL_DIRECTIONS}-direction floor EXP-R2-213 states for "
            "A36-3(b); with a random-direction control the detection floor is set by "
            "direction-to-direction variation, so the count of DIRECTIONS is what has "
            "to be met"
        )
    if not np.isfinite(sample).all():
        raise ValueError("a random-direction control draw is non-finite")
    return {
        "criterion": "A36-3(b)",
        "n_directions": int(sample.size),
        "signed": ca.null_distribution(sample),
        "absolute_p95": float(np.percentile(np.abs(sample), 95.0)),
        "absolute_mean": float(np.abs(sample).mean()),
        "bar_note": "the 95th percentile of |Delta| over distinct norm-matched random "
        "directions at the same alpha and the same site",
    }


def evaluate_a36_3(
    *,
    deltas: Mapping[float, Mapping[str, Any]],
    controls: Mapping[float, Mapping[str, Any]],
    admissible: Sequence[float],
    margin: float,
    spearman_ceiling: float,
) -> dict[str, Any]:
    """A36-3's three conditions, together, on one direction in one mode.

    ``deltas`` is the bootstrapped Delta at every rung and ``controls`` the
    random-direction control at every rung; both are keyed by alpha, because
    A36-3(b) is stated at the same alpha as the effect it bounds.
    """

    positive = [alpha for alpha in sorted(admissible) if alpha > 0.0]
    per_alpha: dict[str, Any] = {}
    for alpha in positive:
        entry = deltas[alpha]
        control = controls[alpha]
        value = float(entry["delta_nats_per_token"])
        bar = margin * float(control["absolute_p95"])
        per_alpha[str(alpha)] = {
            "delta_nats_per_token": value,
            "delta_ci95": [float(bound) for bound in entry["delta_ci95"]],
            "a_excludes_zero": bool(entry["excludes_zero_in_predicted_direction"]),
            "a_excludes_zero_either_direction": bool(
                entry["excludes_zero_either_direction"]
            ),
            "b_control_bar": bar,
            "b_clears_control": bool(abs(value) >= bar) and value < 0.0,
            "b_clears_control_magnitude_only": bool(abs(value) >= bar),
            "both": bool(entry["excludes_zero_in_predicted_direction"])
            and bool(abs(value) >= bar)
            and value < 0.0,
        }
    firing = [alpha for alpha in positive if per_alpha[str(alpha)]["both"]]
    graded = graded_record(
        {alpha: float(entry["delta_nats_per_token"]) for alpha, entry in deltas.items()}
    )
    spearman = graded["spearman"]
    condition_c = spearman is not None and float(spearman) <= float(spearman_ceiling)
    return {
        "criterion": "A36-3",
        "per_alpha": per_alpha,
        "a_and_b_firing_alphas": firing,
        "condition_a": bool(
            any(per_alpha[str(alpha)]["a_excludes_zero"] for alpha in positive)
        ),
        "condition_b": bool(
            any(per_alpha[str(alpha)]["b_clears_control"] for alpha in positive)
        ),
        "condition_a_and_b": bool(firing),
        "condition_c": bool(condition_c),
        "graded": graded,
        "spearman_ceiling": float(spearman_ceiling),
        "random_null_margin": float(margin),
        "passed": bool(firing) and bool(condition_c),
        "reading": "all three of (a), (b) and (c) are required; (a) and (b) must fire at "
        "the SAME admissible positive rung, which is what 'at one admissible alpha' "
        "states. Both are read in the PREDICTED direction -- Delta < 0 at alpha > 0 -- "
        "and each cell also carries the direction-agnostic form, so a reader can see "
        "that an interval excluding zero on the wrong side was not counted as a pass",
    }


def operating_alpha(evaluation: Mapping[str, Any], *, rule: str) -> float | None:
    """The declared rung a concept's specificity and generation are read at.

    The smallest admissible positive rung at which A36-3(a) and (b) both fire.
    Smallest rather than strongest, because picking the strongest would be
    selection on the outcome; declared here so that no rung is chosen after the
    numbers are seen.
    """

    if rule not in OPERATING_ALPHA_RULES:
        raise ValueError(f"unknown operating-alpha rule {rule!r}; declared: {OPERATING_ALPHA_RULES}")
    firing = evaluation["a_and_b_firing_alphas"]
    return float(min(firing)) if firing else None


# ------------------------------------------------------------------- the nulls


def norm_matched_random_directions(
    reference: ConceptDirection, *, n_draws: int, seed: int
) -> tuple[ConceptDirection, ...]:
    """Distinct uniform directions on the sphere, at the reference's sigma.

    Norm matching is exact by construction rather than by rescaling: the concept
    direction is a unit vector and the coefficient carries the scale, so matching
    the norm is matching ``sigma``. Refused below
    :data:`MINIMUM_CONTROL_DIRECTIONS`, because A36-3(b) counts directions.
    """

    if n_draws < MINIMUM_CONTROL_DIRECTIONS:
        raise ValueError(
            f"{n_draws} random direction(s) is below the "
            f"{MINIMUM_CONTROL_DIRECTIONS}-direction floor A36-3(b) states"
        )
    generator = np.random.default_rng(seed)
    draws: list[ConceptDirection] = []
    for index in range(n_draws):
        vector = generator.standard_normal(reference.d_model)
        vector /= np.linalg.norm(vector)
        draws.append(
            ConceptDirection(
                concept=f"{reference.concept}::random_{index:03d}",
                layer=reference.layer,
                direction=vector,
                sigma=reference.sigma,
                provenance="norm-matched uniform random direction",
            )
        )
    return tuple(draws)


def permuted_label_directions(
    representations: np.ndarray,
    labels: Sequence[bool],
    *,
    concept: str,
    layer: int,
    n_draws: int,
    seed: int,
    method: str = "diff_means",
) -> tuple[ConceptDirection, ...]:
    """A36-5's directions: concept vectors fitted to shuffled labels.

    The null a norm-matched random direction cannot supply: it keeps the
    representation cloud, the estimator and the class balance and destroys only
    the correspondence between a description and its concept.
    ``concept_alignment.concept_vector`` is called rather than reimplemented, so
    the null and the estimate are the same computation.
    """

    if n_draws < MINIMUM_CONTROL_DIRECTIONS:
        raise ValueError(
            f"{n_draws} permuted direction(s) is below the "
            f"{MINIMUM_CONTROL_DIRECTIONS}-direction floor; A36-5 asks whether a "
            "permuted direction can pass A36-3, and one draw cannot answer that"
        )
    flags = np.asarray(list(labels), dtype=bool)
    if representations.ndim != 2 or representations.shape[0] != flags.size:
        raise ValueError("representations and labels disagree in shape")
    generator = np.random.default_rng(seed)
    draws: list[ConceptDirection] = []
    for index in range(n_draws):
        shuffled = generator.permutation(flags)
        vector = ca.concept_vector(representations, shuffled, method=method)
        draws.append(
            ConceptDirection.from_concept_vector(
                vector,
                concept=f"{concept}::permuted_{index:03d}",
                layer=layer,
                provenance="concept vector fitted to permuted labels",
            )
        )
    return tuple(draws)


def specificity_matrix(
    cells: Mapping[tuple[str, str], float], concepts: Sequence[str], *, rule: str
) -> dict[str, Any]:
    """A36-4: the full injected x scored matrix, judged per concept and never as a mean.

    A concept is admitted only where its own diagonal exceeds the 95th percentile
    of **its own row's** off-diagonal entries. The comparison is made on the
    concept-consistent effect ``-Delta``, so that "exceeds" means a larger effect
    in the predicted direction: Delta is an NLL shift and a stronger effect is more
    negative, so comparing the raw signed values would invert the criterion. The
    mean diagonal minus mean off-diagonal is reported and may not substitute for
    the per-concept condition -- that substitution is L32 and Appendix B rule 33
    exactly.
    """

    if rule not in SPECIFICITY_RULES:
        raise ValueError(f"unknown specificity rule {rule!r}; declared: {SPECIFICITY_RULES}")
    names = list(concepts)
    if len(set(names)) != len(names) or len(names) < 2:
        raise ValueError(
            "the specificity matrix needs at least two distinct concepts; with one "
            "concept there is no off-diagonal and A36-4 is vacuous"
        )
    missing = [
        (injected, scored)
        for injected in names
        for scored in names
        if (injected, scored) not in cells
    ]
    if missing:
        raise ValueError(f"the specificity matrix is incomplete; missing cells {missing}")
    delta_matrix = [
        [float(cells[(injected, scored)]) for scored in names] for injected in names
    ]
    effect = [[-value for value in row] for row in delta_matrix]
    rows: dict[str, Any] = {}
    for index, injected in enumerate(names):
        off = [effect[index][column] for column in range(len(names)) if column != index]
        diagonal = effect[index][index]
        percentile = float(np.percentile(off, 95.0))
        rows[injected] = {
            "diagonal_effect": diagonal,
            "off_diagonal_effect": off,
            "off_diagonal_p95": percentile,
            "admitted": bool(diagonal > percentile),
        }
    diagonal_values = [effect[index][index] for index in range(len(names))]
    off_values = [
        effect[row][column]
        for row in range(len(names))
        for column in range(len(names))
        if row != column
    ]
    return {
        "criterion": "A36-4",
        "rule": rule,
        "concepts": names,
        "delta_matrix": delta_matrix,
        "effect_matrix": effect,
        "rows": "injected concept",
        "columns": "scored concept",
        "effect_convention": "effect = -Delta, so a larger effect is a larger movement "
        "in the predicted direction; the raw signed Delta matrix is reported beside it",
        "per_concept": rows,
        "admitted_concepts": [name for name in names if rows[name]["admitted"]],
        "reported_but_not_a_criterion": {
            "mean_diagonal_minus_mean_off_diagonal": float(
                np.mean(diagonal_values) - np.mean(off_values)
            ),
            "why": "A36-4 is per concept against its own row's 95th percentile. This "
            "mean is reported because the pre-registration asks for it and it may NOT "
            "substitute for the per-concept condition (L32, Appendix B rule 33)",
        },
    }


# ---------------------------------------------------------------- the verdict

#: STOP-36's four branches plus the pass, each with the reading the
#: pre-registration attaches to it. Named so that no outcome needs a decision
#: after the numbers land.
OUTCOMES = (
    "TRANSFERS",
    "VOID_INSTRUMENT",
    "VOID_PERMUTED_CONTROL_PASSES",
    "NO_ADMISSIBLE_COEFFICIENT_RANGE",
    "MEASURED_NEGATIVE",
    "NULL_NO_CONCEPT_CLEARS_ITS_ROW",
)


def verdict(
    *,
    concept: str,
    text_control: Mapping[str, Any] | None,
    protein: Mapping[str, Any] | None,
    permuted_passes: Sequence[str],
    specificity_row: Mapping[str, Any] | None,
    admissible: Sequence[float],
) -> dict[str, Any]:
    """STOP-36, evaluated in the order the pre-registration fixes it.

    The order is not cosmetic. A36-2 comes first because a protein null measured by
    an instrument that has not been shown to work is uninterpretable rather than
    negative; A36-5 comes next because a permuted direction that passes voids the
    readout whatever the concept direction did; A36-1's empty admissible set comes
    before A36-3 because there is nothing to read; and only then is A36-3 a
    statement about the model.
    """

    gates: dict[str, Any] = {
        "A36-2_text_positive_control": (
            None
            if text_control is None
            else {
                "passed": bool(text_control["condition_a_and_b"]),
                "firing_alphas": list(text_control["a_and_b_firing_alphas"]),
                "rule": "the same direction must satisfy A36-3(a) and (b) in TEXT mode "
                "at one admissible rung; below this the protein side is not read",
            }
        ),
        "A36-5_permuted_control": {
            "passed": not list(permuted_passes),
            "permuted_directions_passing_a36_3": list(permuted_passes),
            "rule": "a permuted-label direction that passes A36-3 means the direction "
            "carries something the label assignment does not",
        },
        "A36-1_admissible_range": {
            "passed": bool([alpha for alpha in admissible if alpha > 0.0]),
            "admissible_alphas": [float(alpha) for alpha in admissible],
        },
        "A36-3_graded_effect": (
            None
            if protein is None
            else {
                "passed": bool(protein["passed"]),
                "condition_a": bool(protein["condition_a"]),
                "condition_b": bool(protein["condition_b"]),
                "condition_c": bool(protein["condition_c"]),
                "spearman": protein["graded"]["spearman"],
                "sign_reverses_below_zero": protein["graded"]["sign_reverses_below_zero"],
            }
        ),
        "A36-4_specificity": (
            None
            if specificity_row is None
            else {
                "passed": bool(specificity_row["admitted"]),
                "diagonal_effect": float(specificity_row["diagonal_effect"]),
                "off_diagonal_p95": float(specificity_row["off_diagonal_p95"]),
            }
        ),
    }
    if gates["A36-2_text_positive_control"] is None or not gates[
        "A36-2_text_positive_control"
    ]["passed"]:
        outcome = "VOID_INSTRUMENT"
    elif not gates["A36-5_permuted_control"]["passed"]:
        outcome = "VOID_PERMUTED_CONTROL_PASSES"
    elif not gates["A36-1_admissible_range"]["passed"]:
        outcome = "NO_ADMISSIBLE_COEFFICIENT_RANGE"
    elif gates["A36-3_graded_effect"] is None or not gates["A36-3_graded_effect"]["passed"]:
        outcome = "MEASURED_NEGATIVE"
    elif gates["A36-4_specificity"] is None or not gates["A36-4_specificity"]["passed"]:
        outcome = "NULL_NO_CONCEPT_CLEARS_ITS_ROW"
    else:
        outcome = "TRANSFERS"
    return {
        "concept": concept,
        "outcome": outcome,
        "gates": gates,
        "pre_registration": PRE_REGISTRATION,
        "reading": {
            "TRANSFERS": "every frozen clause passed at this layer and inside the "
            "admissible coefficient range: a text-derived concept direction has a "
            "graded, concept-specific causal effect on this checkpoint's protein mode",
            "VOID_INSTRUMENT": "the text-mode positive control did not fire, so the "
            "protein side is not read; a protein null here would be uninterpretable "
            "rather than negative",
            "VOID_PERMUTED_CONTROL_PASSES": "a permuted-label direction passed A36-3, so "
            "the readout is measuring something the label assignment does not carry",
            "NO_ADMISSIBLE_COEFFICIENT_RANGE": "the coherence floor admits no positive "
            "rung; the bound is not widened, and this is a statement about the "
            "intervention's dynamic range",
            "MEASURED_NEGATIVE": "the instrument fired on its own text control and the "
            "protein effect did not meet A36-3 at any admissible rung: a measured "
            "negative about the model, at this site and range, registered as a result",
            "NULL_NO_CONCEPT_CLEARS_ITS_ROW": "A36-3 passed but this concept's diagonal "
            "does not exceed its own row's off-diagonal 95th percentile, so the effect "
            "is not specific to the injected concept",
        }[outcome],
    }


# -------------------------------- readout B: an instrument that is not the model


@dataclass(frozen=True)
class HmmerTool:
    """A built HMMER 3.4 installation and the provenance to reproduce it."""

    hmmscan: Path
    hmmpress: Path
    version: str
    tarball: Path
    tarball_sha256: str
    hmmscan_sha256: str

    def record(self) -> dict[str, Any]:
        return {
            "hmmscan": str(self.hmmscan),
            "hmmpress": str(self.hmmpress),
            "version": self.version,
            "tarball": str(self.tarball),
            "tarball_sha256": self.tarball_sha256,
            "hmmscan_sha256": self.hmmscan_sha256,
            "checksum_note": "no publisher checksum is staged beside hmmer-3.4.tar.gz, "
            "unlike diamond-linux64-v2.1.24.tar.gz; the measured digest of the archive "
            "that was built and of the resulting binary are recorded instead of a "
            "digest nobody published being asserted",
        }


def prepare_hmmer(tarball: Path, destination: Path) -> HmmerTool:
    """Build HMMER from the staged source archive, or reuse an existing build.

    ``homology.prepare_diamond``'s shape, with the one difference the staged
    artefact forces: HMMER ships as source, so the tool is *built* rather than
    extracted and the version is read back from the built binary rather than from
    the archive's name. The build goes to a working location outside the
    repository, for the reason that module gives -- a binary and a pressed
    database never enter version control.
    """

    tarball = Path(tarball)
    destination = Path(destination)
    if not tarball.is_file():
        raise FileNotFoundError(f"{tarball} does not exist")
    hmmscan = destination / "bin" / "hmmscan"
    hmmpress = destination / "bin" / "hmmpress"
    if not (hmmscan.is_file() and hmmpress.is_file()):
        build = destination / "build"
        if build.exists():
            shutil.rmtree(build)
        build.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball, "r:gz") as archive:
            archive.extractall(path=build, filter="data")
        roots = [entry for entry in build.iterdir() if entry.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(
                f"{tarball} unpacked to {len(roots)} top-level directories; expected one"
            )
        for command in (
            ["./configure", f"--prefix={destination.resolve()}"],
            ["make", "-j", "8"],
            ["make", "install"],
        ):
            subprocess.run(
                command, cwd=roots[0], check=True, capture_output=True, text=True
            )
    if not (hmmscan.is_file() and hmmpress.is_file()):
        raise RuntimeError(f"the build did not produce {hmmscan} and {hmmpress}")
    completed = subprocess.run(
        [str(hmmscan), "-h"], capture_output=True, text=True, check=True
    )
    match = re.search(r"HMMER ([0-9][0-9.]*)", completed.stdout)
    if match is None:
        raise RuntimeError(f"cannot parse a HMMER version from {completed.stdout[:200]!r}")
    if not match.group(1).startswith("3.4"):
        raise RuntimeError(
            f"the built binary reports HMMER {match.group(1)}; this stage declares 3.4"
        )
    return HmmerTool(
        hmmscan=hmmscan,
        hmmpress=hmmpress,
        version=match.group(1),
        tarball=tarball,
        tarball_sha256=sha256_file(tarball),
        hmmscan_sha256=sha256_file(hmmscan),
    )


@dataclass(frozen=True)
class PfamDatabase:
    """A pressed Pfam-A profile database and what it covers."""

    path: Path
    source_gz: Path
    source_sha256: str
    n_profiles: int

    def record(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "source": str(self.source_gz),
            "source_sha256": self.source_sha256,
            "n_profiles": int(self.n_profiles),
        }


def prepare_pfam(
    archive: Path, checksum_file: Path, destination: Path, *, tool: HmmerTool
) -> PfamDatabase:
    """Verify, decompress and press Pfam-A, or reuse an existing pressed database.

    The published digest is verified rather than assumed, which is what
    ``prepare_diamond`` does and what §0.05 records the cost of skipping: a search
    whose inputs were not the ones named produced a retraction, and the error
    worked in the direction that defeated the hypothesis under test.
    """

    archive = Path(archive)
    checksum_file = Path(checksum_file)
    destination = Path(destination)
    for path in (archive, checksum_file):
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist")
    fields = checksum_file.read_text(encoding="utf-8").split()
    if not fields or len(fields[0]) != 64:
        raise ValueError(f"{checksum_file} does not begin with a sha256 digest")
    expected = fields[0].lower()
    observed = sha256_file(archive)
    if observed != expected:
        raise RuntimeError(
            f"{archive} sha256 {observed} does not match the published {expected}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    profile = destination / "Pfam-A.hmm"
    if not (profile.is_file() and profile.with_suffix(".hmm.h3i").is_file()):
        with gzip.open(archive, "rb") as source, profile.open("wb") as target:
            shutil.copyfileobj(source, target, length=1 << 24)
        subprocess.run(
            [str(tool.hmmpress), "-f", str(profile)],
            check=True,
            capture_output=True,
            text=True,
        )
    n_profiles = 0
    with profile.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("NAME "):
                n_profiles += 1
    if n_profiles < 1:
        raise RuntimeError(f"{profile} carries no profile")
    return PfamDatabase(
        path=profile, source_gz=archive, source_sha256=observed, n_profiles=n_profiles
    )


def write_fasta(path: Path, sequences: Mapping[str, str]) -> Path:
    """One FASTA of the generated sequences, wrapped at 60 residues."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for name, sequence in sequences.items():
            if not sequence:
                raise ValueError(f"{name}: refusing to write an empty sequence")
            handle.write(f">{name}\n")
            for start in range(0, len(sequence), 60):
                handle.write(sequence[start : start + 60] + "\n")
    return path


def run_hmmscan(
    tool: HmmerTool,
    database: PfamDatabase,
    query_fasta: Path,
    output_tbl: Path,
    *,
    evalue: float,
    threads: int,
) -> tuple[list[str], str]:
    """Assign Pfam families to the query sequences; return the command and the log tail.

    The E-value threshold is a parameter rather than a literal because it is the
    one knob that decides how many families a generated sequence appears to carry,
    so it belongs in the artefact. No masking option is passed: HMMER's own null
    model handles composition bias, and §0.05 records what happened the last time
    a masking default silently truncated the evidence this programme reads.
    """

    query_fasta = Path(query_fasta)
    output_tbl = Path(output_tbl)
    if not query_fasta.is_file():
        raise FileNotFoundError(f"{query_fasta} does not exist")
    if evalue <= 0 or threads < 1:
        raise ValueError("invalid hmmscan parameters")
    command = [
        str(tool.hmmscan),
        "--tblout",
        str(output_tbl),
        "--noali",
        "-E",
        repr(evalue),
        "--cpu",
        str(threads),
        str(database.path),
        str(query_fasta),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    if not output_tbl.is_file():
        raise RuntimeError("hmmscan produced no table")
    return command, completed.stdout[-2000:]


def parse_hmmscan_table(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Per-query family hits from ``--tblout``, best-scoring first.

    The accession is compared without its version downstream, because a Pfam
    release bump changes the version of a family that is otherwise the same family
    and a concept's declared referent cannot track that.
    """

    hits: dict[str, list[dict[str, Any]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split()
        if len(fields) < 6:
            raise ValueError(f"malformed hmmscan table row: {line!r}")
        accession = fields[1]
        hits.setdefault(fields[2], []).append(
            {
                "family": fields[0],
                "accession": accession,
                "accession_unversioned": accession.split(".", 1)[0],
                "evalue": float(fields[4]),
                "score": float(fields[5]),
            }
        )
    for entries in hits.values():
        entries.sort(key=lambda entry: entry["evalue"])
    return hits


def annotation_rates(
    hits: Mapping[str, Sequence[Mapping[str, Any]]],
    names: Sequence[str],
    accessions: Sequence[str],
) -> dict[str, Any]:
    """A36-6's two rates: any family at all, and a family the concept declares.

    ``any_family_rate`` is the attainability quantity and is read first: a
    generator whose unperturbed output no annotator recognises cannot show concept
    enrichment, and a null against a zero base rate is indistinguishable from an
    unreachable statistic.
    """

    if not names:
        raise ValueError("no generated sequence to score")
    wanted = {str(value).split(".", 1)[0] for value in accessions}
    annotated = [name for name in names if hits.get(name)]
    carrying = [
        name
        for name in names
        if any(hit["accession_unversioned"] in wanted for hit in hits.get(name, ()))
    ]
    return {
        "n_sequences": len(names),
        "n_with_any_family": len(annotated),
        "any_family_rate": len(annotated) / len(names),
        "n_with_concept_family": len(carrying),
        "concept_family_rate": len(carrying) / len(names),
        "declared_accessions": sorted(wanted),
        "per_sequence_concept_hit": {name: name in set(carrying) for name in names},
    }


def annotation_rate_contrast(
    injected: Mapping[str, Any], baseline: Mapping[str, Any], *, seed: int, n_bootstrap: int
) -> dict[str, Any]:
    """A36-6's interval on the injected minus the alpha=0 concept-family rate.

    The resampling unit is the generated sequence, and that is declared rather
    than assumed: generated output has no near-duplicate groups, so the group
    bootstrap this repository uses everywhere degenerates to a bootstrap over
    sequences here. ``paired_group_bootstrap`` is still the resampler -- one group
    per sequence -- so no second resampler enters the package.
    """

    labels = ["injected"] * int(injected["n_sequences"]) + ["baseline"] * int(
        baseline["n_sequences"]
    )
    values = list(injected["per_sequence_concept_hit"].values()) + list(
        baseline["per_sequence_concept_hit"].values()
    )
    if len(labels) != len(values):
        raise ValueError("the two conditions' sequence counts do not match their labels")
    condition = np.asarray([label == "injected" for label in labels], dtype=int)
    hits = np.asarray([1.0 if value else 0.0 for value in values], dtype=np.float64)
    groups = np.arange(len(values))
    if min(int(injected["n_sequences"]), int(baseline["n_sequences"])) < MINIMUM_BOOTSTRAP_UNITS:
        raise ValueError(
            f"fewer than {MINIMUM_BOOTSTRAP_UNITS} usable sequences on one side; the "
            "interval is not reported wider, it is not reported"
        )
    bootstrap = paired_group_bootstrap(
        condition,
        hits,
        np.zeros_like(hits),
        groups,
        _rate_difference_metric,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )
    return {
        "criterion": "A36-6",
        "rate_injected": float(injected["concept_family_rate"]),
        "rate_baseline": float(baseline["concept_family_rate"]),
        "rate_difference": bootstrap["difference"],
        "rate_difference_ci95": bootstrap["difference_ci95"],
        "excludes_zero": bool(bootstrap["difference_ci95"][0] > 0.0),
        "resampling_unit": "the generated sequence; generated output carries no "
        "near-duplicate groups, so the group bootstrap degenerates to one group per "
        "sequence and that is declared rather than left to a reader",
        "n_bootstrap": bootstrap["n_bootstrap_requested"],
    }


def _rate_difference_metric(truth: np.ndarray, predicted: np.ndarray) -> float:
    injected = np.asarray(truth).astype(bool)
    if injected.all() or (~injected).all():
        return float("nan")
    return float(predicted[injected].mean() - predicted[~injected].mean())


def pfam_referent(
    records: Sequence[Mapping[str, Any]],
    bearing: Sequence[bool],
    *,
    min_bearing_records: int,
) -> tuple[str, ...]:
    """The Pfam families a concept's bearing records carry, as its external referent.

    A36-6 needs a "concept-consistent" family set, and EXP-R2-213's concepts are
    GO terms and EC numbers, neither of which is a Pfam accession. The referent is
    therefore derived from the cohort's own ``pfam`` column, on the **fit** split
    only, so that the evaluation split never defines the target it is scored
    against. A family must appear on at least ``min_bearing_records`` bearing
    records to enter; a concept whose referent comes out empty has readout B
    refused with that reason rather than scored against a mapping invented here.
    """

    if min_bearing_records < 1:
        raise ValueError("a referent family must appear on at least one bearing record")
    flags = np.asarray(list(bearing), dtype=bool)
    if flags.size != len(records):
        raise ValueError("the bearing flags do not align with the records")
    counts: dict[str, int] = {}
    for record, carries in zip(records, flags):
        if not carries:
            continue
        for value in record["pfam"] or ():
            accession = str(value).split(".", 1)[0]
            counts[accession] = counts.get(accession, 0) + 1
    return tuple(
        sorted(
            accession
            for accession, count in counts.items()
            if count >= min_bearing_records
        )
    )


def clean_availability(root: Path) -> dict[str, Any]:
    """Whether CLEAN can be run here, stated rather than worked around.

    CLEAN is an optional second external instrument. Its inference path runs an
    ESM-1b encoder followed by CLEAN's own trained model, and neither weight file
    is one this host can fetch. This returns what is present and what is missing so
    that the artefact records a refusal with its reason instead of an EC prediction
    nothing produced.
    """

    root = Path(root)
    inference = root / "app" / "CLEAN_infer_fasta.py"
    weights = sorted(root.rglob("*.pth")) + sorted(root.rglob("*.pt"))
    cache = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    esm = sorted(cache.glob("esm1b*")) if cache.is_dir() else []
    missing: list[str] = []
    if not inference.is_file():
        missing.append("CLEAN inference entry point")
    if not weights:
        missing.append("CLEAN trained model weights (*.pt/*.pth)")
    if not esm:
        missing.append("ESM-1b encoder weights (esm1b_t33_650M_UR50S)")
    return {
        "runnable": not missing,
        "source_present": inference.is_file(),
        "missing": missing,
        "reason": "CLEAN's EC prediction runs an ESM-1b encoder followed by CLEAN's own "
        "trained model; the staged tree carries the source only and this host has no "
        "route to either weight host, so no EC prediction is produced rather than a "
        "placeholder being written",
    }


# ------------------------------------------------------------------ generation


_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")


def extract_generated_sequence(text: str, *, end_delimiter: str) -> str:
    """The residue run a generated continuation spells, up to the end delimiter.

    Anything that is not a canonical residue ends the sequence, which is stricter
    than stripping the delimiter alone: a decoder that wandered back into prose
    should contribute a short sequence or none, never a sequence with prose folded
    into it.
    """

    residues: list[str] = []
    for character in text.split(end_delimiter, 1)[0]:
        if character in _RESIDUES:
            residues.append(character)
        elif character.isspace():
            continue
        else:
            break
    return "".join(residues)


@torch.no_grad()
def generate_under_injection(
    handle: Any,
    prompt: str,
    *,
    n_sequences: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    layer: int,
    delta: torch.Tensor | None,
) -> list[str]:
    """Sample continuations of one prompt with the injection live throughout.

    The write is at every generated position and, on the prefill pass, at the last
    prompt position only -- which is the position that produces the first
    generated token. That is the same "all content positions, conditioning prompt
    excluded" policy the scored readout uses, expressed against a cache whose
    forward passes are one token wide after the first.
    """

    if n_sequences < 1 or max_new_tokens < 1:
        raise ValueError("generation needs at least one sequence and one new token")
    tokenizer = handle.tokenizer
    ids = torch.tensor(
        [tokenizer(prompt, add_special_tokens=True)["input_ids"]],
        dtype=torch.long,
        device=handle.device,
    )
    prompt_length = int(ids.shape[1])
    installed = None
    if delta is not None:
        modules = feed_forward_modules(handle)
        if not 0 <= layer < len(modules):
            raise ValueError(f"layer {layer} is outside 0..{len(modules) - 1}")
        state = {"calls": 0}

        def hook(module: nn.Module, args: tuple, kwargs: dict[str, Any]):
            hidden = args[0]
            mask = torch.zeros(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
            if state["calls"] == 0:
                mask[:, -1] = True
            else:
                mask[:, :] = True
            state["calls"] += 1
            addend = mask.to(hidden.dtype).unsqueeze(-1) * delta.to(hidden.dtype)
            return (hidden + addend,) + tuple(args[1:]), kwargs

        installed = modules[layer].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        torch.manual_seed(seed)
        outputs = handle.model.generate(
            input_ids=ids.repeat(n_sequences, 1),
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        if installed is not None:
            installed.remove()
    return [tokenizer.decode(row[prompt_length:], skip_special_tokens=True) for row in outputs]


# ------------------------------------------------------- the known-answer model


class _ToyFeedForward(nn.Module):
    """A feed-forward whose response along each planted direction is known.

    ``gain * sum_k (x . u_k) u_k`` plus a small generic term, so writing
    ``alpha * sigma * u_k`` at this module's input raises the residual stream's
    component along ``u_k`` by ``gain * alpha * sigma``. The gain is small on
    purpose: this term compounds across blocks, and at ``gain = 1`` a four-block
    stack multiplies a record's own concept component sixteenfold, which saturates
    the output distribution and leaves an injection nothing to move. Measured on an
    earlier version of this toy: bearing records shifted by 0.01 nats where
    non-bearing ones shifted by 8.3, so every concept's Delta collapsed onto the
    same number.
    """

    def __init__(
        self,
        d_model: int,
        directions: torch.Tensor,
        generator: torch.Generator,
        *,
        gain: float,
    ):
        super().__init__()
        self.register_buffer("directions", directions)
        weight = torch.randn(d_model, d_model, generator=generator) / d_model**0.5
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.gain = float(gain)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        projections = hidden @ self.directions.T
        return self.gain * (projections @ self.directions) + 0.05 * torch.tanh(
            hidden @ self.weight
        )


class _ToyBlock(nn.Module):
    """``residual + mlp(post_attention_layernorm(residual))`` -- the declared layout."""

    def __init__(
        self,
        d_model: int,
        directions: torch.Tensor,
        generator: torch.Generator,
        *,
        gain: float,
    ):
        super().__init__()
        self.post_attention_layernorm = nn.Identity()
        self.mlp = _ToyFeedForward(d_model, directions, generator, gain=gain)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


#: The two states a concept's slice can be in. Every record carries one slice per
#: concept in one of these states, which is what makes the cohort balanced on
#: everything except the memberships themselves -- see :func:`synthetic_records`.
TOY_STATES = ("present", "absent")


class SyntheticConceptModel(nn.Module):
    """A toy decoder carrying planted concept directions, for the known-answer check.

    The construction is the claim under test, made true by hand. Each concept owns
    an orthonormal direction ``u_k`` and, in each mode, a **present** and an
    **absent** token block. The present block's tokens are embedded with
    ``+m * u_k`` and the head adds ``gain_mode(k) * (h . u_k)`` to their logits; the
    absent block's carry neither. So a record whose concept-``k`` slice is present
    has ``h . u_k`` raised over that slice, and writing ``+alpha * sigma * u_k`` at
    the feed-forward's input makes the present block's tokens cheaper still: the
    bearing records gain, because part of their targets are those tokens, and the
    non-bearing records lose the mass that moved. ``Delta`` is therefore negative at
    positive alpha, graded in alpha, sign-reversing below zero, and specific to
    concept ``k``.

    **Three construction details are load-bearing and each was found by the toy
    failing rather than reasoned in advance.**

    The per-token noise is centred *within each token block*, because a block's
    mean noise offset is a property of its tokens rather than of the sample and so
    does not average away with more records; uncentred, the recovered direction kept
    a 0.30 component on each of the other concepts.

    ``head_scale`` shrinks the concept-blind part of the head, which every token's
    logit moves with under any injection. At full scale it is the same size as the
    concept-selective term, and the planted effect is then neither specific nor
    sign-reversing.

    ``mode_gain`` must leave the model *under*-confident about the concept relative
    to the data, or the injection has nothing to buy: if the head already puts
    almost all of the mass on the present block, boosting it further changes the
    bearing records' cross-entropy by nothing while still costing the non-bearing
    ones, which inverts the specificity matrix.

    ``protein_gain=0`` builds the negative cell: the direction is still estimable
    from text-mode representations and still steers text mode, and it must not steer
    protein mode. A pipeline that reports a transfer there is reporting one that
    does not exist, which is the failure this check exists to catch.

    The blocks are exposed at ``model.layers`` with the feed-forward at ``mlp``, so
    :func:`feed_forward_modules` resolves them through the same declared layout a
    real checkpoint goes through.
    """

    def __init__(
        self,
        *,
        d_model: int = 64,
        n_layers: int = 4,
        n_concepts: int = 3,
        tokens_per_block: int = 6,
        embed_strength: float = 1.0,
        # The per-record token-sampling noise, and it is A36-5 that fixes its value
        # rather than realism. A permuted-label refit's direction lies in the span
        # of the planted directions plus this noise, so the noise is what keeps it
        # off the planted direction -- while A36-3(b)'s bar is set by ISOTROPIC
        # random directions, whose component on a planted direction is only one over
        # the square root of the width. The two are therefore in tension, and the
        # measured trade-off on this toy at width 128 and eight concepts is: at 0.4
        # the worst of ~300 permuted refits carries 0.37 of the planted direction
        # against a bar of 0.36 and would pass A36-3; at 0.8 the worst carries 0.31
        # against the same bar and none of them passes, while the true direction
        # still recovers the plant at cosine 0.46 to 0.83 and clears the bar. Above
        # 0.8 the true direction degrades faster than the permuted one and the
        # fixture stops discriminating in the other direction.
        embed_noise: float = 0.8,
        head_scale: float = 0.25,
        mlp_gain: float = 0.1,
        text_gain: float = 1.0,
        protein_gain: float = 1.0,
        seed: int = 20260818,
    ) -> None:
        super().__init__()
        if n_concepts < 2:
            raise ValueError("the specificity matrix needs at least two concepts")
        if tokens_per_block < 2:
            raise ValueError("each token block needs at least two tokens")
        generator = torch.Generator().manual_seed(seed)
        self.d_model = int(d_model)
        self.n_concepts = int(n_concepts)
        self.tokens_per_block = int(tokens_per_block)
        self.pad_token_id = 0
        # pad, then (concept, mode, state) blocks in a declared order.
        self._blocks: dict[tuple[str, int, str], list[int]] = {}
        cursor = 1
        for concept in range(n_concepts):
            for mode in ("protein", "text"):
                for state in TOY_STATES:
                    self._blocks[(mode, concept, state)] = list(
                        range(cursor, cursor + tokens_per_block)
                    )
                    cursor += tokens_per_block
        vocab = cursor

        basis, _ = torch.linalg.qr(torch.randn(d_model, n_concepts, generator=generator))
        directions = basis[:, :n_concepts].T.contiguous()
        self.register_buffer("concept_directions", directions)

        embedding = torch.randn(vocab, d_model, generator=generator) * float(embed_noise)
        for rows in [[0], *self._blocks.values()]:
            embedding[rows] -= embedding[rows].mean(dim=0, keepdim=True)
        for (mode, concept, state), rows in self._blocks.items():
            if state == "present":
                embedding[rows] += float(embed_strength) * directions[concept]
        self.embedding = nn.Parameter(embedding, requires_grad=False)
        self.layers = nn.ModuleList(
            _ToyBlock(d_model, directions, generator, gain=mlp_gain)
            for _ in range(n_layers)
        )
        self.head = nn.Parameter(
            torch.randn(vocab, d_model, generator=generator)
            * float(head_scale)
            / d_model**0.5,
            requires_grad=False,
        )
        selector = torch.zeros(n_concepts, vocab)
        for (mode, concept, state), rows in self._blocks.items():
            if state == "present":
                selector[concept, rows] = float(
                    protein_gain if mode == "protein" else text_gain
                )
        self.register_buffer("concept_selector", selector)
        self.config = SimpleNamespace(hidden_size=int(d_model), vocab_size=int(vocab))
        self.text_gain = float(text_gain)
        self.protein_gain = float(protein_gain)
        self.embed_strength = float(embed_strength)
        self.embed_noise = float(embed_noise)
        self.head_scale = float(head_scale)
        self.mlp_gain = float(mlp_gain)

    def block_tokens(self, mode: str, concept: int, state: str) -> list[int]:
        if state not in TOY_STATES:
            raise ValueError(f"unknown slice state {state!r}; declared: {TOY_STATES}")
        key = (mode, int(concept), state)
        if key not in self._blocks:
            raise ValueError(f"no token block for {key}")
        return list(self._blocks[key])

    def declaration(self) -> dict[str, Any]:
        """What was planted, as it reaches the artefact."""

        return {
            "d_model": self.d_model,
            "n_layers": len(self.layers),
            "n_concepts": self.n_concepts,
            "tokens_per_block": self.tokens_per_block,
            "vocab_size": int(self.config.vocab_size),
            "embed_strength": self.embed_strength,
            "embed_noise": self.embed_noise,
            "head_scale": self.head_scale,
            "mlp_gain": self.mlp_gain,
            "text_head_gain": self.text_gain,
            "protein_head_gain": self.protein_gain,
        }

    def forward(self, input_ids: torch.Tensor, **_: Any) -> Any:
        hidden = self.embedding[input_ids]
        for block in self.layers:
            hidden = block(hidden)
        logits = hidden @ self.head.T + (hidden @ self.concept_directions.T) @ (
            self.concept_selector
        )
        return SimpleNamespace(logits=logits)


class SyntheticJointHandle:
    """The toy model behind the surface :class:`JointReplaceable` presents.

    Only the members this stage uses -- the declared layout, the rendering, the
    batch with its content and target masks, the forward pass and the scored
    logits. It exists so that the known-answer check runs the *same* injection,
    scoring, control and verdict code the real campaign runs; a synthetic path with
    its own scoring loop would certify a computation nobody performs.
    """

    layout = JointBlockLayout(
        layer_list=("layers",),
        feed_forward="mlp",
        pre_feed_forward_norm="post_attention_layernorm",
    )

    def __init__(self, model: SyntheticConceptModel, *, mode: str, device: str = "cpu"):
        if mode not in ("text", "protein"):
            raise ValueError(f"unknown mode {mode!r}")
        self.model = model
        self.mode = mode
        self.name = f"synthetic:{mode}"
        self._device = torch.device(device)
        self.tokenizer = None

    @property
    def width(self) -> int:
        return int(self.model.d_model)

    @property
    def n_layers(self) -> int:
        return len(self.model.layers)

    @property
    def device(self) -> torch.device:
        return self._device

    def render(self, records: Sequence[Any]) -> list[Any]:
        return list(records)

    def forget_rendered(self) -> None:
        return None

    def batch(self, inputs: Sequence[Sequence[int]]) -> dict[str, torch.Tensor]:
        rows = [list(int(value) for value in row) for row in inputs]
        if not rows:
            raise ValueError("an empty batch has nothing to score")
        width = max(len(row) for row in rows)
        if width < 2:
            raise ValueError("a row shorter than two tokens has no scored target")
        input_ids = torch.full((len(rows), width), self.model.pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(rows), width), dtype=torch.long)
        content = torch.zeros((len(rows), width), dtype=torch.bool)
        target = torch.zeros((len(rows), width - 1), dtype=torch.bool)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention[index, : len(row)] = 1
            content[index, : len(row)] = True
            target[index, : len(row) - 1] = True
        return {
            "input_ids": input_ids.to(self._device),
            "attention_mask": attention.to(self._device),
            "content_mask": content.to(self._device),
            "target_mask": target.to(self._device),
        }

    def content_mask(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return batch["content_mask"]

    def run(self, batch: Mapping[str, torch.Tensor]) -> Any:
        return self.model(input_ids=batch["input_ids"])

    def scored_logits(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.run(batch)
        ids = batch["input_ids"]
        return output.logits[..., :-1, :].float(), ids[..., 1:], batch["target_mask"]


def synthetic_records(
    model: SyntheticConceptModel,
    *,
    n_records: int,
    span: int = 6,
    seed: int = 20260818,
) -> tuple[list[dict[str, Any]], list[str]]:
    """A cohort balanced on everything except the memberships themselves.

    Every record carries one ``span``-position slice per concept, in the *present*
    state if it bears that concept and the *absent* state if it does not, and
    membership is drawn independently per concept at probability one half.

    **Three properties, and each was bought by the toy failing without it.**

    *Every record has the same number of positions the injection can act on*,
    whatever it bears. An earlier design filled the remainder with a shared
    background block, so a record bearing more concepts had less background to
    damage; that count asymmetry alone produced an off-diagonal twice the size of
    the diagonal, because a concept's bearers systematically bore more concepts
    than its non-bearers.

    *Membership is independent across concepts*, so an injection along ``u_k``
    moves both sides of every other concept's split equally and the off-diagonal is
    a real zero rather than an artefact of who is in which class. Independence also
    makes the difference of class means recover ``u_k`` alone: an exhaustive design
    in which a concept's non-bearers are exactly the other concepts' bearers gives a
    direction carrying a systematic -0.30 component on each of them.

    *There are enough concepts that a permuted-label refit cannot land on the
    planted direction.* This is A36-5's discriminating power and it is a property of
    the geometry rather than of the seed. A random relabelling's difference of class
    means lies in the span of the planted directions plus sampling noise, so its
    component on any one of them falls as one over the square root of the concept
    count, while the random-direction control's bar falls only with the width. At
    three concepts a permuted refit carried up to 0.89 of the planted direction and
    passed every clause of A36-3 on this toy; the count is therefore a declared
    parameter of the fixture and the measured worst case reaches the artefact.
    """

    if n_records < 2 * MINIMUM_BOOTSTRAP_UNITS:
        raise ValueError(
            f"{n_records} records cannot put {MINIMUM_BOOTSTRAP_UNITS} near-duplicate "
            "groups on both sides of a half-and-half membership draw"
        )
    if span < 2:
        raise ValueError("a slice needs at least two positions")
    generator = np.random.default_rng(seed)
    concepts = [f"concept_{index}" for index in range(model.n_concepts)]
    membership = generator.random((n_records, model.n_concepts)) < 0.5
    counts = membership.sum(axis=0)
    short = {
        concepts[index]: (int(counts[index]), int(n_records - counts[index]))
        for index in range(model.n_concepts)
        if min(counts[index], n_records - counts[index]) < MINIMUM_BOOTSTRAP_UNITS
    }
    if short:
        raise ValueError(
            f"this membership draw leaves {short} (bearing, non-bearing) records, below "
            f"the {MINIMUM_BOOTSTRAP_UNITS}-unit per-side floor; raise n_records"
        )
    records: list[dict[str, Any]] = []
    for row in range(n_records):
        pattern = tuple(
            index for index in range(model.n_concepts) if bool(membership[row, index])
        )
        rows: dict[str, list[int]] = {}
        for mode, field in (("protein", "protein_tokens"), ("text", "text_tokens")):
            tokens: list[int] = []
            for concept in range(model.n_concepts):
                state = "present" if concept in pattern else "absent"
                tokens.extend(
                    int(value)
                    for value in generator.choice(
                        model.block_tokens(mode, concept, state), size=span
                    )
                )
            rows[field] = tokens
        records.append(
            {
                "accession": f"record_{row:04d}",
                "dup_group": f"record_g{row:04d}",
                "concepts": [concepts[index] for index in pattern],
                **rows,
            }
        )
    return records, concepts


def synthetic_batches(
    handle: SyntheticJointHandle,
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> tuple[PreparedBatch, ...]:
    """The toy cohort in the same :class:`PreparedBatch` the real path uses."""

    field = "protein_tokens" if handle.mode == "protein" else "text_tokens"
    batches: list[PreparedBatch] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        tensors = handle.batch(handle.render([record[field] for record in chunk]))
        batches.append(
            PreparedBatch(
                tensors=tensors,
                record_ids=tuple(str(record["accession"]) for record in chunk),
                dup_groups=tuple(str(record["dup_group"]) for record in chunk),
            )
        )
    return tuple(batches)


@torch.no_grad()
def pooled_representations(
    handle: Any, batches: Sequence[PreparedBatch], *, layer: int
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Mean over each row's content positions at the injection site.

    The synthetic path's stand-in for ``concept_alignment.mode_representations``,
    which type-checks its handle against ``JointReplaceable`` and therefore cannot
    take the toy one. It reads the same tensor at the same site and pools it the way
    ``mean_content`` does; the real path calls that function and never this one.
    """

    modules = feed_forward_modules(handle)
    captured: dict[str, torch.Tensor] = {}

    def hook(module: nn.Module, args: tuple, kwargs: dict[str, Any]):
        captured["value"] = args[0].detach()
        return None

    rows: list[np.ndarray] = []
    ids: list[str] = []
    for batch in batches:
        installed = modules[layer].register_forward_pre_hook(hook, with_kwargs=True)
        try:
            handle.run(dict(batch.tensors))
        finally:
            installed.remove()
        hidden = captured.pop("value").float()
        mask = batch.tensors["content_mask"]
        for row in range(hidden.shape[0]):
            rows.append(hidden[row][mask[row]].mean(0).cpu().numpy())
            ids.append(batch.record_ids[row])
    return np.stack(rows).astype(np.float64), tuple(ids)


def bearing_flags(
    records: Sequence[Mapping[str, Any]], concept: str, *, field: str = "concepts"
) -> np.ndarray:
    """Boolean membership of one concept over a synthetic cohort."""

    return np.asarray([concept in record[field] for record in records], dtype=bool)


def digest_of(values: Iterable[str]) -> str:
    """A short stable digest of a declared set, for an artefact basename."""

    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:12]
