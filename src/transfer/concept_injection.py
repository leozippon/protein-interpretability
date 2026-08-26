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
``A36-5`` **as amended (amendment 3):** the effect must exceed the margin times the
    95th percentile of the permuted-label control's ``|Delta|`` over at least eight
    draws -- A36-3(b)'s own rule against a second control population. The frozen text
    voided the readout if any single permuted refit passed A36-3; that is retained as
    a reported diagnostic, per draw and with each draw's distance to its bar.
``A36-6`` readout B's attainability is read first: at ``alpha = 0`` the generator must
    annotate at a non-zero rate against Pfam-A, or readout B is void.

**The population mismatch behind A36-5 is irreducible and is recorded rather than
resolved.** A36-3(b)'s bar comes from *isotropic* random directions; a permuted-label
refit is not isotropic -- it lies in the span of the concept structure plus the
representation cloud's own noise -- so its component on a concept direction is
systematically larger. The two controls therefore bound different things, and amendment 3
does not remove that: it stops the *maximum* of the permuted sample from being the
criterion, which is a separate defect.

The geometry is measured rather than argued, and it is what shows the real run sits in
the safe regime. A permuted refit's component on one concept direction falls as one over
the root of the **concept count**, while the isotropic bar falls as one over the root of
the **width**. At three concepts and width 32 the worst of eight refits carried **0.886**
of the planted direction and passed every clause of A36-3. At eight concepts and width
128, **0 of 32** pass, with the worst reaching **0.617** of its bar against the planted
direction's **1.19**. The real checkpoints are width **4096** with eight or more admitted
concepts, so both curves move the right way there -- but the campaign's margin is not the
fixture's, which is why every permuted draw's distance to its bar reaches the artefact.

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

#: Amendments to that entry which this module implements, each recorded in the
#: artefact so that a reader never has to infer which text a number was produced
#: under. Amendment 3 was decided **before any campaign number existed** -- the only
#: stage-36 numbers in being at the time were this module's own known-answer
#: fixture -- and it makes three changes, two of them relaxations of the frozen
#: text and one an addition:
#:
#: * **A36-5 becomes distributional** (:func:`evaluate_a36_5`). The frozen text
#:   voids the readout if *any* permuted-label refit passes A36-3; the amended rule
#:   requires the real effect to exceed the margin times the 95th percentile of the
#:   permuted control's ``|Delta|`` over at least eight draws. This is a relaxation,
#:   and the reason it is a safe one is that it reuses A36-3(b)'s own rule against a
#:   second control population rather than inventing a second rule: the maximum of a
#:   sample estimates nothing and moves with the draw count, while a percentile
#:   estimates something. The per-draw pass count and each draw's distance to its
#:   bar are retained as reported diagnostics, which is what makes the relaxation
#:   auditable.
#: * **A balanced per-concept evaluation draw is admitted** as a named addition to
#:   the frozen reduction rule (:func:`balanced_evaluation_draw`).
#: * **A36-6's Pfam referent is authorised by name**, derived from the cohort's own
#:   ``pfam`` column on the fit split alone (:func:`pfam_referent`).
PRE_REGISTRATION_AMENDMENTS: tuple[str, ...] = ("amendment 3", "amendment 4")

AMENDMENT_3_NOTE = (
    "EXP-R2-213 amendment 3, decided before any campaign number existed: A36-5 is "
    "distributional rather than any-draw (a RELAXATION of the frozen text, reusing "
    "A36-3(b)'s percentile rule against the permuted population, with the per-draw "
    "pass count and distances retained as diagnostics); a balanced per-concept "
    "evaluation draw is admitted as a named addition to the reduction rule, with a "
    "mandatory second-seed draw-variance measurement; and A36-6's fit-split Pfam "
    "referent is authorised by name"
)

#: EXP-R2-213 amendment 4: the per-side bearing-group cap. Every concept contributes
#: at most this many near-duplicate groups to each side of its own Delta.
#:
#: **It is a criterion improvement and not an economy**, and the reason is A36-4.
#: Within one row of the specificity matrix, cell ``(A, B)`` is Delta for concept
#: ``B`` computed on ``B``'s own subset at ``B``'s own group count -- so the
#: off-diagonal referent set itself mixes precisions, and uncapped a 15-group
#: diagonal could be compared against a percentile taken over 985-group cells. The
#: heterogeneity is therefore *within* row as well as across rows, and the cap
#: equalises the diagonal and every off-diagonal cell of a row simultaneously, which
#: is what makes the row's 95th percentile a like-for-like referent. On the
#: production cohort it collapses the spread from 65.7x to 2.13x.
#:
#: Two things it does **not** do, recorded because either would be read into it.
#: Capping equalises *noise*, not *signal*: A36-4 still compares raw effects, so a
#: concept with a genuinely larger effect still dominates its row, and a capped
#: matrix must never be read as a matrix of t-statistics. And it cannot equalise a
#: concept that never had the groups -- ``go_atp_binding`` carries 15 and reaches no
#: cap at or above 16, so its row is flagged rather than averaged in.
FROZEN_PER_SIDE_CAP = 32

#: Below roughly this many near-duplicate groups per side the *point estimate*
#: becomes draw-sensitive, which is a different failure from a wide interval and is
#: not visible in one. Measured on the known-answer fixture by subsampling groups: at
#: 12 per side a single unlucky subsample moved ``|Delta|`` from about 0.00095 to
#: 0.00075 and ``|Delta|/bar`` from 1.2 to **0.78**, across the decision boundary,
#: while its interval stayed at 28% of ``|Delta|``. At 24 and above the ratio is flat
#: (1.06-1.27, no trend in n). This is why :data:`FROZEN_PER_SIDE_CAP` is 32 and not
#: 16: cap 32 leaves exactly one concept below the threshold, cap 16 would put all
#: seventeen below it. Amendment 3's second-seed draw-variance check is the
#: designated detector for this mode, and it is wired to every concept that sits
#: below this line.
DRAW_STABILITY_GROUP_THRESHOLD = 24

#: What the cap is safe *because of*, recorded so that the reasoning is not later
#: compressed into "power saturates at 32", which is false.
#:
#: The interval does not saturate: its half-width scales as one over the root of the
#: group count with no knee (half-width times root-n measured flat at 0.00047-0.00073
#: on the fixture), so 64 to 128 buys the same 29% that 32 to 64 does. The cap is
#: safe for a different reason. A36-3(a) is not the binding gate -- the interval
#: half-width sits at 8-16% of ``|Delta|`` across every count from 8 to 64, clearing
#: (a) throughout. **A36-3(b) binds, and it compares point estimates**: its bar is
#: the 95th percentile of the random-direction control's ``|Delta|``, which rises only
#: about 22% from 64 groups per side to 8 because it is set mostly by genuine
#: direction-to-direction variation rather than by estimation noise. The decisive
#: ratio ``|Delta|/bar`` accordingly shows no trend in the group count.
CAP_SAFETY_NOTE = (
    "the interval does NOT saturate -- it scales as 1/sqrt(n) with no knee. The cap is "
    "safe because A36-3(a) is not the binding gate (its half-width is 8-16% of |Delta| "
    "throughout) while A36-3(b) binds and compares POINT ESTIMATES, and (b)'s bar is "
    "set mostly by direction-to-direction variation rather than estimation noise, so it "
    "rises only ~22% from 64 groups per side to 8 and |Delta|/bar shows no trend in n. "
    "This must not be remembered as 'power saturates at 32'"
)

#: Why the cap retires the frozen reduction rule instead of inverting it. Cost is
#: ``K x passes x |union|``: one pass scores the union and yields Delta for every
#: concept, so concepts are **equal-cost given the union** and differ only in how much
#: they inflate it. Uncapped that is wildly unequal -- ``ec_transferase`` adds 832
#: marginal records to the union and ``go_membrane`` 435, against tens for the small GO
#: concepts -- which is what made the frozen rule (drop in ascending bearing count)
#: keep the most expensive concepts. Capped at 32 the union is 539 records for all
#: seventeen and a concept's marginal cost is near-uniform, so no reduction is needed
#: and the rule never fires.
COST_MODEL_NOTE = (
    "cost is K x passes x |union|, not a sum over concepts: one pass scores the union "
    "and yields Delta for every concept, so concepts are equal-cost given the union and "
    "differ only in how much they inflate it. That is why the cap retires the frozen "
    "reduction rule rather than inverting it -- at all 17 concepts and cap 32 the rule "
    "never fires"
)

#: How the evaluation population is chosen. ``full_split`` is the frozen behaviour --
#: every record of the declared split. ``balanced_1to1`` is amendment 3's addition:
#: every near-duplicate group a concept bears, plus a seeded equal-size draw of
#: groups it does not. Declared rather than defaulted, so the authorisation is
#: visible at the call site.
EVALUATION_DRAWS = ("full_split", "balanced_1to1")

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


@dataclass(frozen=True)
class ConceptCell:
    """One concept's three-valued membership over the scored records.

    The cohort's rule is three-valued and not two: a record with no annotation of
    the relevant sort is **undefined** for the concept and enters neither side,
    "which is the only honest reading of a curated database where absence of an
    annotation is absence of curation" (``sequence_description.ConceptSpec``). On
    the production cohort that is not a corner case -- ``ec_hydrolase`` has 508
    bearing, 2,065 non-bearing and **1,292 undefined** groups in ``eval`` -- so a
    two-valued reading would fold 1,292 groups of unknown status into the
    non-bearing arm of Delta and silently change the estimand.

    ``included`` additionally carries the evaluation draw: under
    :data:`EVALUATION_DRAWS` ``balanced_1to1`` it is the concept's own balanced
    subset, and under ``full_split`` it is every defined record.
    """

    concept: str
    bearing: np.ndarray
    defined: np.ndarray
    included: np.ndarray

    def __post_init__(self) -> None:
        shapes = {self.bearing.shape, self.defined.shape, self.included.shape}
        if len(shapes) != 1:
            raise ValueError(f"{self.concept}: membership arrays disagree in shape")
        if bool(np.any(self.included & ~self.defined)):
            raise ValueError(
                f"{self.concept}: a record the cohort leaves UNDEFINED for this concept "
                "was included in its evaluation subset; undefined records enter neither "
                "side of Delta"
            )

    @property
    def subset(self) -> np.ndarray:
        return self.included

    def counts(self) -> dict[str, int]:
        return {
            "bearing": int((self.included & self.bearing).sum()),
            "non_bearing": int((self.included & ~self.bearing).sum()),
            "undefined_excluded": int((~self.defined).sum()),
            "defined_not_drawn": int((self.defined & ~self.included).sum()),
        }


def _paired_shift(
    baseline: ScoredResponse,
    injected: ScoredResponse,
    bearing: Sequence[bool],
    *,
    subset: Sequence[bool] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The per-record shift, the bearing flags and the groups, on the scored subset.

    ``subset`` is the concept's own evaluation population: the records the cohort
    defines it on, intersected with the evaluation draw. Rows outside it are dropped
    before anything is computed, so an undefined record cannot reach either arm and
    a balanced draw cannot be diluted by records it did not draw.
    """

    if baseline.record_ids != injected.record_ids:
        raise ValueError(
            "the injected and uninjected passes scored different records, so the "
            "within-record difference is not paired"
        )
    flags = np.asarray(list(bearing), dtype=bool)
    if flags.shape != baseline.nll_per_token.shape:
        raise ValueError("the bearing flags do not align with the scored records")
    keep = (
        np.ones(flags.shape, dtype=bool)
        if subset is None
        else np.asarray(list(subset), dtype=bool)
    )
    if keep.shape != flags.shape:
        raise ValueError("the evaluation subset does not align with the scored records")
    shift = (injected.nll_per_token - baseline.nll_per_token)[keep]
    flags = flags[keep]
    if flags.size == 0 or flags.all() or (~flags).all():
        raise ValueError(
            "this concept's evaluation subset carries only one class, so Delta is "
            "undefined on it"
        )
    return shift, flags, np.asarray(baseline.dup_groups)[keep]


def per_side_group_counts(
    dup_groups: Sequence[str],
    bearing: Sequence[bool],
    *,
    subset: Sequence[bool] | None = None,
) -> dict[str, int]:
    """Distinct near-duplicate groups on each side of a concept.

    Counted per side because EXP-R2-213's floor is per side: eight groups in total
    with one on the bearing side is not eight units of the comparison that is
    being made.
    """

    groups = np.asarray(list(dup_groups))
    flags = np.asarray(list(bearing), dtype=bool)
    if subset is not None:
        keep = np.asarray(list(subset), dtype=bool)
        groups, flags = groups[keep], flags[keep]
    return {
        "bearing": int(np.unique(groups[flags]).size),
        "non_bearing": int(np.unique(groups[~flags]).size),
    }


def require_per_side_group_floor(
    dup_groups: Sequence[str],
    bearing: Sequence[bool],
    *,
    label: str,
    subset: Sequence[bool] | None = None,
) -> dict[str, int]:
    counts = per_side_group_counts(dup_groups, bearing, subset=subset)
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
    baseline: ScoredResponse,
    injected: ScoredResponse,
    bearing: Sequence[bool],
    *,
    subset: Sequence[bool] | None = None,
) -> float:
    """Delta(alpha) alone, with no interval.

    Used for a *control* draw, whose per-draw interval nothing reads: A36-3(b) is a
    percentile over directions, so the draws contribute point values and the
    economy removes no control and no draw (Appendix B rule 37).
    """

    shift, flags, _ = _paired_shift(baseline, injected, bearing, subset=subset)
    return _delta_metric(flags.astype(int), shift)


def delta_nll_shift(
    baseline: ScoredResponse,
    injected: ScoredResponse,
    bearing: Sequence[bool],
    *,
    seed: int,
    n_bootstrap: int,
    label: str = "delta",
    subset: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """A36-3's Delta with its group-bootstrap interval.

    ``paired_group_bootstrap`` is used the way it is used everywhere else -- the
    near-duplicate group is the resampling unit -- with the right-hand prediction
    vector held at zero, so the reported ``difference`` is Delta itself and its
    interval is a percentile interval over group resamples. No resampler is
    declared here; this stage adds none.
    """

    shift, flags, groups = _paired_shift(baseline, injected, bearing, subset=subset)
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
    baseline: ScoredResponse,
    injected: ScoredResponse,
    bearing: Sequence[bool],
    *,
    subset: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """A36-1: what the injection cost the NON-BEARING sequences.

    Non-bearing only, and that is the point: an inflation measured over the whole
    cohort could be lowered by the very effect being claimed, so the coherence
    floor is read on the sequences the concept is not supposed to help.
    """

    shift, flags, _ = _paired_shift(baseline, injected, bearing, subset=subset)
    keep = (
        np.ones(baseline.nll_per_token.shape, dtype=bool)
        if subset is None
        else np.asarray(list(subset), dtype=bool)
    )
    weights = baseline.scored_tokens.astype(np.float64)[keep]
    entropy_shift = (injected.entropy_per_token - baseline.entropy_per_token)[keep]
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
            np.average(baseline.nll_per_token[keep][~flags], weights=weights[~flags])
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


def _control_distribution(
    values: Sequence[float], *, criterion: str, population: str
) -> dict[str, Any]:
    """One control population's ``Delta`` distribution at one alpha and site.

    The 95th percentile of ``|Delta|`` is the bar in both places it is used, and the
    absolute value is taken because the criterion is written on ``|Delta(alpha)|``: a
    control direction that happens to move the bearing sequences the wrong way is
    still evidence about how large a move a direction of this kind produces, and
    dropping its magnitude would make the bar easier.

    One implementation, two named callers, because A36-3(b) and amended A36-5 are the
    same statistic against two different populations -- and stating that is the whole
    argument for the amendment.
    """

    sample = np.asarray([float(value) for value in values], dtype=np.float64)
    if sample.size < MINIMUM_CONTROL_DIRECTIONS:
        raise ValueError(
            f"{sample.size} {population} direction(s) is below the "
            f"{MINIMUM_CONTROL_DIRECTIONS}-direction floor EXP-R2-213 states for "
            f"{criterion}; the detection floor is set by direction-to-direction "
            "variation, so the count of DIRECTIONS is what has to be met"
        )
    if not np.isfinite(sample).all():
        raise ValueError(f"a {population} control draw is non-finite")
    return {
        "criterion": criterion,
        "population": population,
        "n_directions": int(sample.size),
        "signed": ca.null_distribution(sample),
        "absolute_p95": float(np.percentile(np.abs(sample), 95.0)),
        "absolute_mean": float(np.abs(sample).mean()),
        "absolute_max": float(np.abs(sample).max()),
        "bar_note": f"the 95th percentile of |Delta| over distinct {population} "
        "directions at the same alpha and the same site",
    }


def random_direction_control(values: Sequence[float]) -> dict[str, Any]:
    """A36-3(b): the norm-matched random-direction control at one alpha and site."""

    return _control_distribution(
        values, criterion="A36-3(b)", population="norm-matched isotropic random"
    )


def permuted_label_control(values: Sequence[float]) -> dict[str, Any]:
    """A36-5 as amended: the permuted-label control at one alpha and site.

    The population a random direction cannot supply. A permuted refit keeps the
    representation cloud, the estimator and the class balance and destroys only the
    correspondence between a description and its concept, so it lies in the span of
    the concept structure plus the cloud's own noise rather than being isotropic --
    which is exactly why it is a second population and needs its own bar.
    """

    return _control_distribution(
        values, criterion="A36-5", population="permuted-label refit"
    )


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


def evaluate_a36_5(
    *,
    deltas: Mapping[float, Mapping[str, Any]],
    permuted_controls: Mapping[float, Mapping[str, Any]],
    firing_alphas: Sequence[float],
    margin: float,
    per_draw: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """A36-5 as amended: the real effect must clear the permuted control's percentile.

    The frozen text voids the readout if *any* permuted-label refit passes A36-3.
    Amendment 3 replaces that with the criterion A36-3(b) already uses, applied to
    the permuted population: ``|Delta| >= margin x`` the 95th percentile of the
    permuted control's ``|Delta|`` over at least eight draws, at a rung where the
    effect fires. This is a **relaxation** of the frozen text, and what makes it a
    safe one is that it is not a second rule: the maximum of a sample estimates
    nothing and grows with the draw count, so an any-draw void gets strictly harder
    as the control is enlarged, which is a property no control should have. A
    percentile estimates something.

    ``per_draw`` is retained and reported rather than collapsed, because the
    relaxation is only auditable if a reader can see how many draws would have
    voided under the frozen rule and how close each came to its bar.
    """

    rungs = [float(alpha) for alpha in firing_alphas]
    per_alpha: dict[str, Any] = {}
    for alpha in rungs:
        control = permuted_controls[alpha]
        value = float(deltas[alpha]["delta_nats_per_token"])
        bar = float(margin) * float(control["absolute_p95"])
        per_alpha[str(alpha)] = {
            "delta_nats_per_token": value,
            "permuted_control_bar": bar,
            "permuted_control_p95": float(control["absolute_p95"]),
            "n_permuted_directions": int(control["n_directions"]),
            "clears": bool(abs(value) >= bar) and value < 0.0,
        }
    clearing = [alpha for alpha in rungs if per_alpha[str(alpha)]["clears"]]
    vacuous = not rungs
    passing = [
        entry["direction"] for entry in per_draw if entry.get("passes_a36_3")
    ]
    return {
        "criterion": "A36-5",
        "rule": "amended: |Delta| >= margin x the permuted control's 95th percentile of "
        "|Delta| at a rung where the effect fires",
        "amended_by": f"{PRE_REGISTRATION} amendment 3",
        "relaxation": "the frozen text voided the readout if ANY permuted draw passed "
        "A36-3; that criterion gets strictly harder as the control is enlarged, which "
        "is a property no control should have. The per-draw outcomes below are what the "
        "frozen rule would have read",
        "margin": float(margin),
        "per_alpha": per_alpha,
        "clearing_alphas": clearing,
        "vacuous": vacuous,
        "vacuous_note": "A36-3 fired at no admissible positive rung, so there is no "
        "effect for this criterion to be asked about. The verdict reads A36-3 first and "
        "reports a measured negative rather than dressing one as a control failure",
        "passed": bool(clearing) or vacuous,
        "frozen_rule_would_have_voided": bool(passing),
        "permuted_draws_passing_a36_3": passing,
        "n_permuted_draws": len(per_draw),
        "per_draw": list(per_draw),
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


def admitted_concepts(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """The concept ids the cohort stage ADMITTED, read and never derived.

    ``34_sequence_description_cohort.py`` admits a concept under C34-5 by measured
    per-cell group counts in both deciding splits, with the floor's attainability
    read off a curve first, and it records the surviving list. Deriving a concept
    set here instead -- by enumerating the ``ec`` or ``go`` columns, say -- would
    evaluate concepts nobody checked against the floor, at counts nobody measured,
    and would put the campaign outside the pre-registration entirely.
    """

    concepts = manifest.get("concepts")
    if not isinstance(concepts, Mapping) or "admitted" not in concepts:
        raise ValueError(
            "the cohort manifest carries no concepts.admitted list, so the admitted "
            "concept set cannot be read; this stage does not derive one"
        )
    admitted = tuple(str(value) for value in concepts["admitted"])
    if not admitted:
        raise ValueError(
            "the cohort admitted no concept, so there is nothing for this stage to "
            "measure; STOP-34 governs that outcome and stage 36 is not authorised"
        )
    return admitted


def require_admitted_concepts(
    requested: Sequence[str], manifest: Mapping[str, Any], *, source: Path | str
) -> tuple[str, ...]:
    """Refuse any requested concept the cohort did not admit, naming both.

    Refuse rather than warn, and never fall back to a derived set: an interval on a
    concept that failed C34-5 is not a weaker result, it is an unreportable one.
    """

    admitted = admitted_concepts(manifest)
    unknown = [name for name in requested if name not in admitted]
    if unknown:
        raise ValueError(
            f"{unknown} are not admitted concepts of {source}. This stage measures the "
            "concepts 34_sequence_description_cohort.py admitted under C34-5 and "
            "derives no set of its own; the admitted ids are "
            f"{list(admitted)}"
        )
    if len(set(requested)) != len(requested):
        raise ValueError(f"the requested concept set carries a duplicate: {list(requested)}")
    if len(requested) < 2:
        raise ValueError(
            "at least two concepts are required: with one there is no off-diagonal and "
            "A36-4 is vacuous"
        )
    return tuple(requested)


def assert_stage35_handoff(
    handoff: Mapping[str, Any], *, layer: int, site: str, pooling: str, source: Path | str
) -> dict[str, Any]:
    """Refuse a layer, site or pooling that is not the one stage 35 handed off.

    EXP-R2-213 states every criterion at one layer and reads it there (L32), and
    amendment 3 fixes that layer as stage 35's already pre-registered decision
    layer -- so choosing it here creates no new post-hoc freedom, and *checking* it
    is what makes that true rather than asserted. The site and the pooling are
    checked in the same call because a direction is only a direction with respect
    to the tensor and the aggregation it was estimated under.
    """

    if not handoff.get("emitted"):
        raise ValueError(
            f"{source} emitted no causal hand-off, so stage 35 did not authorise a "
            "causal stage on this cell; A35-2 and STOP-35 govern that outcome"
        )
    mismatches = {
        field: (declared, observed)
        for field, declared, observed in (
            ("layer", int(handoff["layer"]), int(layer)),
            ("site", str(handoff["site"]), str(site)),
            ("pooling", str(handoff["pooling"]), str(pooling)),
        )
        if declared != observed
    }
    if mismatches:
        raise ValueError(
            f"this run disagrees with {source}'s causal hand-off on "
            f"{ {k: {'stage35': v[0], 'requested': v[1]} for k, v in mismatches.items()} }. "
            "The layer is stage 35's pre-registered decision layer and the site and "
            "pooling are the ones the directions were estimated under; a mismatch means "
            "the direction being steered is not the direction that was measured"
        )
    return {
        "source": str(source),
        "layer": int(handoff["layer"]),
        "site": str(handoff["site"]),
        "pooling": str(handoff["pooling"]),
        "description_variant": str(handoff.get("description_variant")),
        "checked": ["layer", "site", "pooling"],
    }


def balanced_evaluation_draw(
    dup_groups: Sequence[str],
    cells: Mapping[str, tuple[Sequence[bool], Sequence[bool]]],
    *,
    seed: int,
    cap: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """EXP-R2-213 amendment 3: every bearing group, plus a 1:1 seeded non-bearing draw.

    Authorised as a named addition to the frozen reduction rule. The frozen design
    scores the whole declared split, where a concept's interval is governed by its
    bearing groups while almost all of the compute goes to estimating a non-bearing
    mean to a precision nothing reads -- on the production cohort ``go_atp_binding``
    has 15 bearing groups against 1,359 non-bearing.

    **One seeded permutation of the groups serves every concept**, and each concept
    takes the leading groups of it that are non-bearing *for that concept*, until it
    has as many as it bears. That is what makes the draws overlap heavily rather
    than being independent per concept, so the union stays small; and it is what
    makes the population fixed once -- the same draw is reused across every rung,
    every concept and both checkpoints, so no two reported numbers rest on different
    populations. Undefined records are never drawn: a concept's complement is taken
    from the groups it is *defined and not bearing* on.

    What it costs, stated because it is a real cost the full-split design did not
    carry: the non-bearing arm becomes a *sample*, so it carries draw variance. That
    variance is not assumed small -- the stage measures it against a second,
    independent seed and reports it, on every concept that sits below
    :data:`DRAW_STABILITY_GROUP_THRESHOLD`.

    ``cap`` is amendment 4's per-side bearing-group cap. It is applied to **both**
    sides -- a concept contributes at most ``cap`` bearing groups and the same number
    of non-bearing ones -- and it is taken from the same seeded permutation, so the
    capped selection is reproducible from the recorded seed alone and is identical
    across every rung, every concept and both checkpoints. See
    :data:`FROZEN_PER_SIDE_CAP` for why it is a criterion improvement rather than an
    economy, and :data:`CAP_SAFETY_NOTE` for why it does not weaken the gate that
    binds.
    """

    if cap is not None and cap < MINIMUM_BOOTSTRAP_UNITS:
        raise ValueError(
            f"a per-side cap of {cap} is below the {MINIMUM_BOOTSTRAP_UNITS}-unit "
            "bootstrap floor, so it would refuse every concept it applied to"
        )
    groups = np.asarray([str(value) for value in dup_groups])
    order = np.random.default_rng(seed).permutation(np.unique(groups))
    rank = {group: index for index, group in enumerate(order)}
    included: dict[str, np.ndarray] = {}
    record: dict[str, Any] = {}
    for concept, (bearing, defined) in cells.items():
        flags = np.asarray(list(bearing), dtype=bool)
        known = np.asarray(list(defined), dtype=bool)
        if flags.shape != groups.shape or known.shape != groups.shape:
            raise ValueError(f"{concept}: membership does not align with the groups")
        # Both sides are ordered by the ONE seeded permutation and taken as a prefix,
        # which is what makes the capped selection reproducible from the seed.
        bearing_set = set(np.unique(groups[known & flags]).tolist())
        non_bearing_set = set(np.unique(groups[known & ~flags]).tolist())
        # A near-duplicate group can STRADDLE the boundary -- carry both a bearing and
        # a non-bearing record of the same concept -- and such a group cannot be
        # assigned to a side. It is excluded from both, exactly as an undefined record
        # is: the resampling unit is the group, so a group in both arms would be
        # resampled into both at once and its per-side count would exceed any cap. The
        # cohort stage measures the same quantity as ``groups_on_both_sides``, and this
        # reproduces its counts (11 on go_membrane, 1 to 4 on the other GO concepts,
        # 0 on all six EC concepts); it was found by amendment 4's cap invariant
        # firing on go_rna_binding at 33 groups against a cap of 32.
        straddling = sorted(bearing_set & non_bearing_set, key=lambda g: rank[g])
        bearing_available = sorted(bearing_set - set(straddling), key=lambda g: rank[g])
        non_bearing_available = sorted(
            non_bearing_set - set(straddling), key=lambda g: rank[g]
        )
        take = (
            len(bearing_available)
            if cap is None
            else min(int(cap), len(bearing_available))
        )
        drawn = set(bearing_available[:take]) | set(non_bearing_available[:take])
        keep = np.array([g in drawn for g in groups], dtype=bool) & known
        included[concept] = keep
        counts = per_side_group_counts(groups, flags, subset=keep)
        if cap is not None and max(counts.values()) > int(cap):
            raise RuntimeError(
                f"{concept}: the draw put {counts} groups on a side against a cap of "
                f"{cap}; the cap is an invariant of the selection, not a target"
            )
        short = {
            side: value
            for side, value in counts.items()
            if value < MINIMUM_BOOTSTRAP_UNITS
        }
        if short:
            raise ValueError(
                f"{concept}: the balanced draw at cap {cap} leaves {short} "
                f"near-duplicate group(s), below the {MINIMUM_BOOTSTRAP_UNITS}-unit "
                "per-side floor. Refused rather than reported with a wider interval"
            )
        smaller_side = min(counts.values())
        record[concept] = {
            "bearing_groups": counts["bearing"],
            "non_bearing_groups_drawn": counts["non_bearing"],
            "bearing_groups_available": len(bearing_available),
            "non_bearing_groups_available": len(non_bearing_available),
            "straddling_groups_excluded": len(straddling),
            "capped": bool(cap is not None and len(bearing_available) > int(cap)),
            "undefined_records_excluded": int((~known).sum()),
            "records_scored": int(keep.sum()),
            "smaller_side_groups": smaller_side,
            "above_draw_stability_threshold": bool(
                smaller_side >= DRAW_STABILITY_GROUP_THRESHOLD
            ),
        }
    union = np.zeros(groups.shape, dtype=bool)
    for keep in included.values():
        union |= keep
    below = [
        concept
        for concept, entry in record.items()
        if not entry["above_draw_stability_threshold"]
    ]
    return included, {
        "rule": "balanced_1to1",
        "seed": int(seed),
        "per_side_cap": None if cap is None else int(cap),
        "authorised_by": f"{PRE_REGISTRATION} amendments 3 and 4",
        "per_concept": record,
        "union_records_scored": int(union.sum()),
        "union_groups_scored": int(np.unique(groups[union]).size),
        "draw_stability_threshold_groups": DRAW_STABILITY_GROUP_THRESHOLD,
        "concepts_below_draw_stability_threshold": below,
        "straddling_note": "a near-duplicate group carrying both a bearing and a "
        "non-bearing record of the same concept is excluded from both sides, as an "
        "undefined record is: the resampling unit is the group, so a group in both arms "
        "would be resampled into both at once. The counts reproduce the cohort stage's "
        "own groups_on_both_sides",
        "cap_note": "the cap is applied to BOTH sides from the same seeded "
        "permutation, so the selection is reproducible from the seed alone and is "
        "identical across every rung, every concept and both checkpoints. It equalises "
        "noise and NOT signal, so a capped specificity matrix must never be read as a "
        "matrix of t-statistics",
        "cap_safety_note": CAP_SAFETY_NOTE,
        "cost_model_note": COST_MODEL_NOTE,
        "note": "one seeded permutation of the near-duplicate groups serves every "
        "concept, which is what fixes the population once and keeps the union small; "
        "undefined records are never drawn",
    }


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
    "PERMUTED_CONTROL_NOT_CLEARED",
    "NO_ADMISSIBLE_COEFFICIENT_RANGE",
    "MEASURED_NEGATIVE",
    "NULL_NO_CONCEPT_CLEARS_ITS_ROW",
)


def verdict(
    *,
    concept: str,
    text_control: Mapping[str, Any] | None,
    protein: Mapping[str, Any] | None,
    permuted: Mapping[str, Any] | None,
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
        "A36-5_permuted_control": (
            None
            if permuted is None
            else {
                "passed": bool(permuted["passed"]),
                "clearing_alphas": list(permuted["clearing_alphas"]),
                "frozen_rule_would_have_voided": bool(
                    permuted["frozen_rule_would_have_voided"]
                ),
                "permuted_directions_passing_a36_3": list(
                    permuted["permuted_draws_passing_a36_3"]
                ),
                "rule": permuted["rule"],
                "amended_by": permuted["amended_by"],
            }
        ),
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
    elif (
        gates["A36-5_permuted_control"] is not None
        and not gates["A36-5_permuted_control"]["passed"]
    ):
        outcome = "PERMUTED_CONTROL_NOT_CLEARED"
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
            "PERMUTED_CONTROL_NOT_CLEARED": "the effect does not exceed the margin "
            "times the permuted-label control's 95th percentile at any firing rung, so "
            "it is within what a relabelling of the same cloud produces",
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
    evalue: float | None = None,
    threads: int,
    gathering_threshold: bool = False,
) -> tuple[list[str], str]:
    """Assign Pfam families to the query sequences; return the command and the log tail.

    The E-value threshold is a parameter rather than a literal because it is the
    one knob that decides how many families a generated sequence appears to carry,
    so it belongs in the artefact. ``gathering_threshold`` instead passes
    ``--cut_ga``, Pfam's own curated per-family cut, which is what makes "this
    sequence carries this family" a statement of the release rather than a
    threshold decision taken inside the measurement; the two are mutually
    exclusive and one of them must be given. No masking option is passed: HMMER's
    own null model handles composition bias, and §0.05 records what happened the
    last time a masking default silently truncated the evidence this programme
    reads.
    """

    query_fasta = Path(query_fasta)
    output_tbl = Path(output_tbl)
    if not query_fasta.is_file():
        raise FileNotFoundError(f"{query_fasta} does not exist")
    if threads < 1:
        raise ValueError("invalid hmmscan parameters")
    if gathering_threshold == (evalue is not None):
        raise ValueError(
            "pass exactly one of evalue= and gathering_threshold=True; a run that "
            "declared both would report a cut it did not apply"
        )
    if evalue is not None and evalue <= 0:
        raise ValueError("invalid hmmscan parameters")
    cut = ["--cut_ga"] if gathering_threshold else ["-E", repr(evalue)]
    command = [
        str(tool.hmmscan),
        "--tblout",
        str(output_tbl),
        "--noali",
        *cut,
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
