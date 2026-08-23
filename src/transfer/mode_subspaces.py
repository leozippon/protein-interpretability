"""Which directions one set of weights uses for text and which it uses for protein.

The question
============

A joint language-protein decoder holds **one** set of weights and reads two input
formats. So the question "do text continuation and protein continuation occupy
distinct computational subspaces?" can be asked on this lineage with architecture,
scale, tokenizer and weights all held exactly fixed -- the only axis that moves is
which mode the checkpoint is being read in. Two standalone models control the
first three of those and never the fourth.

This is an **Objective 1/2 measurement of what a model does**, not an Objective-3
knowledge claim, so the recombination ceiling of audit section 7.0 is not the
admission rule here; section 7.0's own closing paragraph says so ("it does not
reach Objective 1 or Objective 2, where a measurement of what a model does is
itself the result"). What section 7.0 *does* govern and this module inherits is
the resampling unit: the near-duplicate group, never the record (L30).

Why this design is available where others are not
=================================================

**No token alignment is required, and that is the whole reason this design can
run on the only qualified joint bridge this programme has.** Limitation L31
measures a multi-residue BPE leaving a single substitution's mutant and wild type
token-aligned on 47.0-54.5% of instances, and -- the load-bearing half -- the
survivors are the BPE-stable subset rather than a random one, so a position-level
intervention on such an arm is both undefined on half the cohort and computed on
a selected remainder. Every design blocked under section 7.0's fourth declined
item (role swap, analogue patch, intra-fragment intervention) is blocked on
exactly that.

This measurement never pairs a position in one run with a position in another
run. It asks which **dimensions** of one layer's write into the residual stream a
mode occupies and which of them it needs, and it compares two *subspaces* of
R^d_model. A subspace comparison has no position index in it. The two modes read
different corpora, of different lengths, in different symbol units, and none of
that has to correspond. That is what makes the ProLLaMA lineage measurable here
while it is refused for a positional intervention on the identical evidence.

What is measured, in four parts
===============================

**1. Occupancy.** Per layer and per mode, the centred covariance of the sampled
activations and the scale-free summaries of its spectrum. This is
:mod:`src.transfer.spectrum`'s measurement, reused rather than re-derived, so an
occupancy figure here and the R1.2 result are the same estimator on the same
statistic. Several statistics are reported and **none of them is "the" answer**:
on one contrast R1.2 reads a 35x multiplier on the participation ratio, 7.7x on
the effective rank and 1.24x on the 99%-variance dimension, so a single-statistic
occupancy claim is a claim about the choice of statistic (:data:`OCCUPANCY_NOTE`).

**2. Necessity, which is the part that matters.** Occupancy is not computation. A
direction a mode's activations happen to vary along is not thereby part of what
the mode computes. So the top-r principal directions of a mode's own covariance
at a layer are **projected out of that layer's residual write** and the damage to
that mode's own next-symbol likelihood is measured, over a ladder of r. The
result is a necessity-ranked spectrum: damage as a function of occupancy rank,
with the per-band increments that say where the necessary directions sit inside
the occupied ones.

**3. Overlap.** Between the two modes' *necessary* subspaces, by a declared
statistic and **always beside its chance level**. The ambient dimension is 4,096
and two random subspaces of high dimension overlap substantially by construction:
two random r-dimensional subspaces of R^d have expected mean squared principal
cosine exactly ``r/d``, which is 0.125 at r = 512. An overlap statistic reported
without that number is uninterpretable, so :func:`subspace_overlap` returns the
closed form and a seeded Monte-Carlo band with every value it computes.

**4. The unigram control, which decides whether any of it means anything.** A
"mode-specific" direction whose ablation only moves the model's unigram output
distribution is a bias term, not a computation. Every damage figure is therefore
decomposed, **per position and therefore exactly**, into the part explained by the
induced shift in the model's own marginal predictive distribution and the
residual:

    -log p(y|x)  =  -log q(y)  +  ( -log [ p(y|x) / q(y) ] )

with ``q`` the model's mean predictive distribution over the measured positions.
Differencing an ablated pass against the clean pass gives
``total = unigram + residual`` at every position, hence in every bootstrap draw.
The second term is the pointwise **context information**, which is the estimand
``21_joint_mode_qualification.py`` reports for these very cells, so the residual
half of the damage is "how much context information the ablation destroyed" in
the same units the modes were qualified in.

``q`` is estimated **held out across near-duplicate groups**, never plug-in:
Appendix B rule 3, and L12's measured plug-in bias of up to +1.02 nats on a
protein arm against +0.009 at residue level is larger than most of the damages
this ladder is asked to resolve. The plug-in value is computed anyway and reported
beside it as the measured bias at this cohort size, and the cohort's own
target-count entropy is reported through
:func:`src.transfer.budget.miller_madow_entropy_nats` as the bias-corrected
context-free reference.

**The headline claim is licensed only by the residual.** If the residual is small
the finding is "the two modes differ in their unigram statistics", which is a much
weaker statement, and :data:`DECISION_RULES` returns ``UNIGRAM_ONLY`` for it
rather than letting a large total damage be read as distinct computation.

A mode with nothing to destroy, and how that is decided
======================================================

Some modes carry too little context-derived signal for an ablation to destroy.
``Llama-2-7b-hf``'s protein mode is the measured case on this lineage: context
information **+0.0843 nats/token** and a reversal cost of **-0.0013 nats/residue**
(EXP-R2-152, re-measured at EXP-R2-174), recorded in
:data:`LOW_SIGNAL_MODE_EVIDENCE`.

That is a **prior expectation about a cohort and never an admission rule.** This
module used to gate on it: a mode's context information was declared on the
command line and compared against a locally declared 0.30-nat floor before any
behavioural quantity was computed, and a mode below it was refused every damage
figure, every cross-mode contrast and every verdict. The floor was underived -- it
is the constant L41 catalogues and EXP-R2-218 retired -- the number it decided was
measured on a different cohort from the one being ablated, and the failure it was
guarding against is one :data:`DECISION_RULES` already tests by measurement.

The rule tests it ex post, on this stage's own cohort, and per layer. A licensed
verdict requires **in every mode of the run** that the residual non-unigram damage
from ablating that mode's own necessary subspace be positive with a paired
group-bootstrap 95% interval excluding zero. A mode with no context-derived signal
cannot satisfy that clause, and the artefact records which way it failed:
``VOID_INSTRUMENT`` where zeroing the whole block write does not damage the mode
and there is therefore no attainable denominator, ``NO_MEASURED_DAMAGE`` where the
ablation establishes no damage to decompose, ``UNIGRAM_ONLY`` where the damage is
mostly the induced shift in the model's own marginal, and ``MIXED`` where the
clauses disagree. None of the four is a necessity claim. The mode's context
information is reported beside them by :func:`cohort_context_information`,
measured on the cohort the damage was measured on rather than declared.

Occupancy was never gated and is not gated now. It is representational: the
activations exist and their covariance is a real object, which is the same
distinction ``32_crosscoder.py`` records for the pre-adaptation checkpoint's
protein mode.

Per layer, never a mean
=======================

Every quantity in the artefact this module feeds is indexed by layer, every
verdict is a per-layer verdict, and :func:`assert_per_layer_fields` -- imported
from :mod:`src.transfer.crosscoder`, which is where that rule has its one
implementation -- refuses a per-site field that was collapsed to a scalar. L32 and
Appendix B rule 33 exist because a criterion stated per unit was instrumented as
a cross-layer mean and returned a verdict its own per-layer vector contradicted.
There is no cross-layer summary here, not even a convenience one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .budget import (
    miller_madow_entropy_nats,
    unigram_entropy_nats,
)
from .crosscoder import assert_per_layer_fields  # noqa: F401  (re-exported guard)
from .statistics import MINIMUM_BOOTSTRAP_UNITS, paired_group_bootstrap

SCHEMA_VERSION = "r2_transfer_mode_subspaces_v1"

#: The entry that froze this stage's criteria before any campaign number existed.
#:
#: This constant read ``UNREGISTERED_IN_THE_AUDIT`` from the day the stage was
#: written until EXP-R2-215 was recorded, and the refusal it carried was real: the
#: artefact said so, and the limitations block said no number might be cited until
#: a log entry existed. **Both halves are kept rather than deleted.** The status
#: now names the entry, and :data:`PRE_REGISTRATION_SCOPE` keeps the rest of what
#: the refusal was protecting -- that a frozen design is not a logged run, and a
#: campaign artefact still has to reach ``docs/EXPERIMENT_LOG.md`` before it is
#: cited. Replacing a refusal with an identifier and nothing else would discharge
#: more than the entry actually discharges.
PRE_REGISTRATION = "EXP-R2-215"
PRE_REGISTRATION_STATUS = f"PRE_REGISTERED_{PRE_REGISTRATION}"

#: What the pre-registration does and does not license, carried into every artefact.
PRE_REGISTRATION_SCOPE = (
    f"{PRE_REGISTRATION} (D2.i) freezes this stage's estimand, its decision rule and "
    "its refusals before any campaign number existed, and it records that this is an "
    "Objective 1/2 measurement which audit section 7.0's recombination ceiling does "
    "not gate. What it does NOT do is admit a result: a run of this stage is citable "
    "only once its own artefact is recorded in docs/EXPERIMENT_LOG.md against this "
    "identifier, and a pre-registration is a statement about the design rather than "
    "about any number produced under it"
)

#: The measured evidence that a mode of this lineage may carry little
#: context-derived signal, carried into the artefact's limitations block. Quoted
#: here once so that what the artefact says and what this module's own reasoning
#: rests on cannot drift apart.
#:
#: **It gates nothing.** A 0.30-nat floor keyed to this reading used to admit or
#: refuse every behavioural cell of a mode before any of them was computed; it was
#: declared locally, never derived, and decided a number measured on a different
#: cohort from the one being ablated. It is retired, and what a mode's ablation
#: can support is now decided where it was always also decided -- ex post, per
#: layer, by the residual-damage clauses of :data:`DECISION_RULES`.
#:
#: The catalogue this evidence was once said to fall outside of does **not** end at
#: L32: it runs past L42, to L43. The mode's own low reading is still a measurement rather
#: than a catalogued defect, but the floor it was refused against is catalogued, at
#: **L41** -- ``budget.MIN_CONTEXT_INFORMATION_NATS = 0.30`` was never derived,
#: identification needs 0.010-0.020 nats and denominator admissibility is a per-arm
#: bound spanning 0.1456-0.9664, and the repair L41 declares is a
#: precision-referenced criterion, a screening floor at 0.05 nats, and the constant
#: kept only as a reporting column.
LOW_SIGNAL_MODE_EVIDENCE = (
    "Llama-2-7b-hf's protein mode reads +0.0843 nats/token of context information "
    "and a reversal cost of -0.0013 nats/residue on this lineage (EXP-R2-152, "
    "re-measured at EXP-R2-174), so an ablation in it may have little to destroy. "
    "That is a property of the mode on that cohort and it decides nothing here: "
    "whether a mode's ablation supports a necessity claim is decided per layer by "
    "this stage's own measured residual damage under the decision rule, and a mode "
    "with no signal reaches NO_MEASURED_DAMAGE, VOID_INSTRUMENT, UNIGRAM_ONLY or "
    "MIXED rather than a verdict. The 0.30-nat floor this stage once refused such a "
    "mode against is catalogued at L41 (EXP-R2-218) and is retired; the catalogue "
    "runs past L42, to L43"
)

#: Why a design that would be blocked by L31 is not blocked here.
NO_ALIGNMENT_NOTE = (
    "nothing in this measurement pairs a position of one run with a position of "
    "another run. It compares SUBSPACES of R^d_model -- which dimensions a mode "
    "occupies and which it needs -- and a subspace comparison carries no position "
    "index. L31's finding that a single substitution leaves a multi-residue-BPE "
    "arm token-aligned on only 47.0-54.5% of instances, with a non-random survivor "
    "set, therefore does not reach it. That is the reason this design is available "
    "on this lineage where a role swap, an analogue patch and an intra-fragment "
    "intervention are all declined (audit section 7.0)"
)

#: Reported occupancy statistics, in the order a reader should see them.
#: :mod:`src.transfer.spectrum` computes every one of them.
OCCUPANCY_STATISTICS: tuple[str, ...] = (
    "participation_ratio",
    "effective_rank",
    "r95",
    "r99",
    "r999",
)

OCCUPANCY_NOTE = (
    "several statistics, and none of them is 'the' occupancy. R1.2 measured the "
    "same contrast at a 35x multiplier on the participation ratio, 7.7x on the "
    "effective rank and 1.24x on the 99%-variance dimension, so a single-statistic "
    "occupancy claim is mostly a claim about the choice of statistic. They "
    "disagree by construction: the participation ratio is dominated by the largest "
    "eigenvalues, the effective rank reads the whole spectrum, and a variance-cut "
    "rank reads one point of the cumulative curve"
)

#: Written in place of the rank statistics when the position budget cannot carry
#: them. A covariance estimated from N samples has rank at most min(N, d), so a
#: spectrum measured below the declared floor reports the sampling budget rather
#: than the data, and a rank-limited spectrum is indistinguishable from a
#: genuinely low-dimensional one in every statistic reported here.
OCCUPANCY_UNDERSAMPLED = "OCCUPANCY_UNDERSAMPLED"

#: Written in place of a cross-mode quantity that needs two modes when one ran.
SINGLE_MODE_RUN = "SINGLE_MODE_NOT_A_CROSS_MODE_COMPARISON"

#: Written in place of an overlap when a mode's necessary subspace is not defined
#: at that layer -- the ladder never reached the necessity fraction, or the full
#: block ablation was not attainable. It is deliberately NOT
#: :data:`SINGLE_MODE_RUN`: one is a statement about this site and this ladder in a
#: run that had both modes, the other about a run that named one, and spelling them
#: the same way would let a reader take a short ladder for a missing comparison.
NO_NECESSARY_SUBSPACE = "NO_NECESSARY_SUBSPACE_AT_THIS_LAYER"

#: What ``-mean log[p(y|x)/q(y)]`` with the model's OWN marginal is, and what it is
#: not. EXP-R2-152's context information is a corpus unigram entropy minus the
#: model's cross-entropy; this one replaces the corpus unigram with the model's own
#: mean predictive distribution. The two answer different questions -- how much
#: better the model is than the corpus's own symbol frequencies, against how much
#: of the model's own skill is contextual -- and the second is generally the larger.
#: They are not interchangeable. This module holds no threshold on either any
#: more, but ``21_joint_mode_qualification.py`` still gates a mode on the
#: corpus-referenced one, so the prohibition stands where a threshold does: neither
#: may be substituted for the other at that gate.
MODEL_MARGINAL_CONTEXT_INFORMATION_NOTE = (
    "context information against the MODEL'S OWN held-out marginal, not against a "
    "corpus unigram. EXP-R2-152's figure -- and the quantity "
    "21_joint_mode_qualification.py still gates a mode on -- is the corpus unigram "
    "entropy minus the model's cross-entropy; this one is -mean log[p(y|x)/q(y)] "
    "with q the model's mean predictive distribution, which is what the residual "
    "half of every damage figure here is a difference of. The two are different "
    "estimands. Nothing in this stage takes a threshold on either, and neither may "
    "be substituted for the other where one does. The corpus-referenced quantity is "
    "reported beside this one by cohort_context_information, computed on this "
    "stage's own cohort"
)

MEAN_SQUARED_COSINE = "mean_squared_cosine"
FIRST_PRINCIPAL_ANGLE_COSINE = "first_principal_angle_cosine"

#: The overlap statistics this module computes. Both are reported on every
#: comparison; ``--overlap-statistic`` declares which one the decision rule reads,
#: because a rule that could pick either after seeing both is not a rule.
OVERLAP_STATISTICS: tuple[str, ...] = (MEAN_SQUARED_COSINE, FIRST_PRINCIPAL_ANGLE_COSINE)

OVERLAP_STATISTIC_DEFINITIONS: dict[str, str] = {
    MEAN_SQUARED_COSINE: (
        "sum of the squared cosines of the principal angles between the two "
        "subspaces, divided by min(r_a, r_b) -- equivalently ||A^T B||_F^2 / "
        "min(r_a, r_b) for orthonormal A, B. 1 when one subspace contains the "
        "other, 0 when they are orthogonal. Its chance level is EXACT: for a fixed "
        "A and a Haar-random B, E||A^T B||_F^2 = r_a r_b / d, so the statistic's "
        "expectation under chance is max(r_a, r_b) / d"
    ),
    FIRST_PRINCIPAL_ANGLE_COSINE: (
        "the largest singular value of A^T B: the cosine of the SMALLEST principal "
        "angle, i.e. how well the single best-aligned direction pair does. It has "
        "no closed-form chance level and is reported against a seeded Monte-Carlo "
        "band only. It is the statistic to read when a claim is about one shared "
        "direction rather than about a shared subspace"
    ),
}


@dataclass(frozen=True)
class DecisionRule:
    """One frozen bundle of thresholds, selected by name and never by flag.

    The thresholds are constants of a *named rule* rather than command-line
    numbers for the reason ``36_concept_injection.py`` freezes its ladder in
    :mod:`src.transfer.concept_injection`: a threshold that can be passed can be
    passed again, and a rule whose numbers are chosen at the call site is chosen
    after the numbers exist. ``--decision-rule`` is required and never defaulted;
    what it selects is fixed here.
    """

    name: str
    necessity_fraction: float
    residual_share_floor: float
    overlap_margin: float
    min_positions_per_dimension: int
    logit_tolerance: float
    note: str

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "necessity_fraction": self.necessity_fraction,
            "residual_share_floor": self.residual_share_floor,
            "overlap_margin": self.overlap_margin,
            "min_positions_per_dimension": self.min_positions_per_dimension,
            "logit_tolerance": self.logit_tolerance,
            "note": self.note,
        }


DECISION_RULES: dict[str, DecisionRule] = {
    "residual_licensed_v1": DecisionRule(
        name="residual_licensed_v1",
        # The necessary subspace of a mode at a layer is the smallest rung of the
        # ladder whose TOTAL damage reaches this fraction of what zeroing the whole
        # block write costs. The denominator is an attainability check as much as a
        # normaliser (Appendix B rule 2): a layer whose full ablation does not hurt
        # the mode has no necessary subspace to find, and the rule says so instead
        # of dividing by a number near zero.
        necessity_fraction=0.5,
        # Below this share of the damage surviving the unigram decomposition, the
        # finding is about unigram statistics and is reported as such. Half is the
        # point at which the two halves of the decomposition change places.
        residual_share_floor=0.5,
        # How far above its own chance band an overlap has to sit before the two
        # necessary subspaces are called shared, and how far below the top of that
        # band before they are called distinct. Two random subspaces already
        # overlap at r/d, so a 25% margin on the Monte-Carlo band's upper end is
        # the smallest gap that is not reading the band's own width.
        overlap_margin=0.25,
        # Sampled positions per ambient dimension before an occupancy spectrum may
        # be read at all. Ten is `30_activation_spectrum.py`'s declared floor and
        # is restated here as a number of this rule rather than imported from a
        # numbered stage, because it is a threshold of THIS decision and a stage
        # import for one integer would pull three model-loading stages into a
        # library module.
        min_positions_per_dimension=10,
        # Maximum absolute logit movement a null ablation may cause, and the
        # minimum a large random ablation must cause. `src.transfer.das.invariants`
        # uses 1e-3 on the same class of check.
        logit_tolerance=1e-3,
        note=(
            "a layer reads DISTINCT_SUBSPACES only if, in BOTH modes, the residual "
            "(non-unigram) damage from ablating that mode's own necessary subspace "
            "has a paired group-bootstrap 95% interval excluding zero, the residual "
            "share clears the floor, the paired own-minus-other contrast at matched "
            "rank is positive with an interval excluding zero, AND the two necessary "
            "subspaces' overlap sits at or below its own chance band. Anything that "
            "clears the damage clauses but not the unigram floor is UNIGRAM_ONLY; "
            "anything that clears them with an overlap above the chance band and no "
            "asymmetry is SHARED_SUBSPACE; anything else is MIXED and is reported "
            "rather than forced into a verdict"
        ),
    )
}

VERDICTS: tuple[str, ...] = (
    "DISTINCT_SUBSPACES",
    "SHARED_SUBSPACE",
    "UNIGRAM_ONLY",
    "NO_MEASURED_DAMAGE",
    "MIXED",
    "VOID_INSTRUMENT",
)

VERDICT_READINGS: dict[str, str] = {
    "DISTINCT_SUBSPACES": (
        "at this layer each mode needs directions the other does not, the effect "
        "survives the unigram decomposition, and the two necessary subspaces are no "
        "more aligned than two random subspaces of the same dimensions would be"
    ),
    "SHARED_SUBSPACE": (
        "at this layer both modes need a subspace, the effect survives the unigram "
        "decomposition, the two necessary subspaces overlap above their own chance "
        "band and neither mode is damaged more by its own basis than by the other's"
    ),
    "UNIGRAM_ONLY": (
        "the ablation damages the likelihood, and the damage is mostly the induced "
        "shift in the model's marginal output distribution. The finding is that the "
        "modes differ in their unigram statistics, which is much weaker than a "
        "claim about distinct computation and must not be reported as one"
    ),
    "NO_MEASURED_DAMAGE": (
        "ablating a mode's own necessary subspace does not establish damage to that "
        "mode at this layer, in at least one mode, so there is nothing to decompose "
        "and no subspace claim to make. It is a statement about this site at this "
        "cohort size and is not a claim that the subspace is unnecessary"
    ),
    "MIXED": (
        "some clauses passed and some did not; the per-clause record is the result. "
        "No verdict is forced, because the clauses are about different things"
    ),
    "VOID_INSTRUMENT": (
        "the intervention could not be read at this layer -- either the hook "
        "invariants did not hold, or zeroing the whole block write does not damage "
        "the mode, so there is no attainable denominator and a necessary subspace "
        "is undefined here. This is a statement about the site, not about the model"
    ),
}

#: The tensor every quantity here is defined on. Not a flag: occupancy and
#: necessity have to be read on ONE tensor or "occupies but does not need" is a
#: sentence about two different objects, and the block's feed-forward output is
#: the only per-layer tensor
#: :meth:`src.transfer.replaceable.JointReplaceable.block_intercept` can
#: substitute -- its `block_input` partner is the input to a normalisation and can
#: be read but not written. It is also the tensor R2.3's transcoders decode into
#: and the one `30_activation_spectrum.py` reads its primary spectrum at, so an
#: occupancy figure here is commensurable with those.
TENSOR = "block_output"

PROJECTION_NOTE = (
    "the intervention is a PROJECTION, not an addition: the block's feed-forward "
    "output y is replaced by y - (y B) B^T for an orthonormal basis B of the "
    "ablated subspace, so the block still writes into the residual stream and "
    "writes nothing along B. Rank 0 is exactly the identity and rank d_model is "
    "exactly the zero write that "
    "src.transfer.replaceable.JointReplaceable.ablated already performs, so the "
    "ladder is anchored at both ends by objects that exist independently of it"
)

RANDOM_CONTROL_NOTE = (
    "the positive control projects out a SEEDED RANDOM subspace of the ladder's "
    "largest rank, and the rank matters. src.transfer.das.invariants records why a "
    "CONSTANT perturbation is the wrong positive control on this architecture -- "
    "the block's normalisation very nearly annihilates it, and a correctly bound "
    "hook then reports as unbound. A projection is not a constant, but it has the "
    "matching failure at the other end: removing a random rank-r subspace of a "
    "4,096-dimensional write removes about r/4,096 of its energy, so a control run "
    "at a small rank can be genuinely too small to move the logits and would report "
    "a correctly bound hook as unbound for a second reason. It is run at the "
    "largest declared rank"
)

#: What a top-r principal basis is, and is not, identified by.
#:
#: The eigenvectors of a covariance are ordered by eigenvalue, so the top-r
#: subspace is a property of the data only where the r-th and (r+1)-th eigenvalues
#: are separated. Where the spectrum is flat the top-r subspace is set by sampling
#: noise, and two clouds occupying *literally the same span* then recover different
#: top-r subspaces: measured on this module's own fixture with a flat spectrum, two
#: identical 12-dimensional spans returned a top-8 overlap of 0.668, which is
#: exactly the 8/12 two random 8-dimensional subspaces of a shared 12-dimensional
#: space would give. **This bounds every overlap figure this module produces**: an
#: overlap BELOW chance is evidence of distinct subspaces only where the two
#: spectra are separated at the rank being read, and the eigenvalue gap at that
#: rank is reported beside every basis so a reader can see whether it was.
EIGEN_ORDER_IDENTIFIABILITY_NOTE = (
    "a top-r principal basis is identified only where the r-th and (r+1)-th "
    "eigenvalues are separated. On a flat stretch of spectrum the top-r subspace is "
    "chosen by sampling noise, and two clouds with identical spans recover "
    "different top-r subspaces -- 0.668 mean squared cosine between two top-8 bases "
    "of one shared 12-dimensional span on this module's own fixture, which is the "
    "8/12 chance value for that ambient span. The relative eigenvalue gap at each "
    "ablated rank is reported with every basis; a low overlap read at a rank whose "
    "gap is near zero is a statement about the estimator and not about the modes"
)

#: What a paired own-minus-other contrast means at a rung where the ablation has
#: saturated, and why the necessary rank is chosen below one.
#:
#: Once a rung removes essentially everything a mode computes at that layer, the
#: residual damage is at its ceiling and the contrast between two bases stops
#: measuring what the two modes need: it measures the difference between two
#: *estimates* of the same subspace. Measured on this module's own fixture with the
#: two modes' spans planted identical, the two recovered bases read a mean squared
#: cosine of 0.9987 and the paired contrast still excluded zero at the saturated
#: rung, while at a rung below saturation it correctly did not. The necessary rank
#: is therefore the SMALLEST rung reaching the necessity fraction rather than the
#: largest available, and a ladder whose only rungs are far apart cannot supply one.
SATURATED_RUNG_NOTE = (
    "a paired own-minus-other contrast read at a rung where the ablation has "
    "saturated measures the difference between two estimates of one subspace "
    "rather than a difference between two modes: two bases planted identical read "
    "0.9987 mean squared cosine on this module's fixture and the contrast still "
    "excluded zero there. The necessary rank is the smallest rung reaching the "
    "necessity fraction for this reason, and a run whose necessary rank equals its "
    "ladder's top rung should be read as saturated and not as asymmetric"
)

#: The resampling unit, and the reason it is not the record. Audit section 7.0
#: clause 3 and L30: 42.5% of held-out records keep a >=95%-identity relative on
#: this corpus, so a record-level unit resamples near-copies as independent draws.
GROUP_UNIT_NOTE = (
    "the near-duplicate group from src.transfer.near_duplicates.near_duplicate_"
    "groups, never the record. L30 measured 871 of 2,048 held-out records (42.5%) "
    "keeping a >=95%-identity relative, so a record-level resampling unit treats "
    "near-copies as independent draws and reports an interval narrower than the "
    "evidence supports"
)

UNIGRAM_ESTIMATOR_NOTE = (
    "q is the model's OWN mean predictive distribution over the measured "
    "positions, estimated held out across near-duplicate groups: a position in "
    "half A is scored against the q accumulated on half B and vice versa. Appendix "
    "B rule 3 and L12 -- a plug-in unigram carries up to +1.02 nats of bias on a "
    "protein arm against +0.009 at residue level, several times the 0.30-nat "
    "measurability floor these cells were qualified against. The plug-in value is "
    "computed as well and the difference is reported as the measured bias at this "
    "cohort size, and the cohort's own target-count entropy is reported through "
    "miller_madow_entropy_nats as the bias-corrected context-free reference"
)


# ------------------------------------------------------------- declarations


def decision_rule(name: str) -> DecisionRule:
    """The named rule, refusing one nobody declared."""

    if name not in DECISION_RULES:
        raise KeyError(
            f"unknown decision rule {name!r}; declared: {sorted(DECISION_RULES)}. A "
            "rule is a frozen bundle of thresholds selected by name, so an "
            "undeclared name has no thresholds to run under"
        )
    return DECISION_RULES[name]


def parse_rank_ladder(argument: str) -> tuple[int, ...]:
    """``"1,2,4,8"`` into ``(1, 2, 4, 8)``, ascending, unique and positive.

    Rank 0 is not written and cannot be: it is the clean pass, which every rung is
    differenced against, so it is a property of the ladder rather than a rung of
    it. A ladder that named it would make the first rung's damage a difference
    against itself.
    """

    ranks: list[int] = []
    for piece in str(argument).replace(" ", "").split(","):
        if not piece:
            continue
        value = int(piece)
        if value < 1:
            raise ValueError(
                f"rank {value} is not a rung: rank 0 IS the clean pass every rung is "
                "differenced against, and a negative rank is not a subspace"
            )
        ranks.append(value)
    if not ranks:
        raise ValueError("an ablation ladder cannot be empty")
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"{argument!r} names a rank twice")
    if ranks != sorted(ranks):
        raise ValueError(
            f"{argument!r} is not ascending; the necessity spectrum's increments are "
            "differences between consecutive rungs and are meaningless out of order"
        )
    return tuple(ranks)


# --------------------------------------------------------------- subspaces


def principal_basis(covariance: torch.Tensor, rank: int) -> torch.Tensor:
    """The top-``rank`` eigenvectors of a covariance, as an orthonormal ``(d, rank)``.

    float64 in, float32 out: the eigendecomposition is done at the precision
    :class:`src.transfer.spectrum.CovarianceAccumulator` accumulates in, and the
    basis is spent in a forward pass where float32 is the analysis precision.
    """

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            f"expected a square covariance, got shape {tuple(covariance.shape)}"
        )
    d_model = int(covariance.shape[0])
    if not 0 <= rank <= d_model:
        raise ValueError(f"rank {rank} is outside 0..{d_model}")
    if rank == 0:
        return torch.zeros((d_model, 0), dtype=torch.float32, device=covariance.device)
    values, vectors = torch.linalg.eigh(covariance.to(torch.float64))
    order = torch.argsort(values, descending=True)[:rank]
    return vectors[:, order].to(torch.float32).contiguous()


def eigen_gap(eigenvalues: torch.Tensor | np.ndarray, rank: int) -> dict[str, Any]:
    """How separated the ``rank``-th eigenvalue is from the next one down.

    ``(lambda_r - lambda_{r+1}) / lambda_r`` on the descending spectrum. It is the
    number that says whether the top-``rank`` subspace is a property of the data or
    of the sampling: see :data:`EIGEN_ORDER_IDENTIFIABILITY_NOTE`. Reported beside
    every basis rather than checked against a threshold, because there is no cut at
    which a small gap makes a basis wrong -- it makes the *overlap* read on that
    basis uninterpretable, which is a statement a reader has to make with the
    number in front of them.
    """

    values = np.sort(
        np.asarray(
            eigenvalues.detach().cpu().numpy()
            if isinstance(eigenvalues, torch.Tensor)
            else eigenvalues,
            dtype=np.float64,
        ).reshape(-1)
    )[::-1]
    if not 1 <= rank <= values.size:
        raise ValueError(f"rank {rank} is outside 1..{values.size}")
    at_rank = float(values[rank - 1])
    below = float(values[rank]) if rank < values.size else 0.0
    return {
        "rank": int(rank),
        "eigenvalue_at_rank": at_rank,
        "eigenvalue_below_rank": below,
        "relative_gap": (at_rank - below) / at_rank if at_rank > 0.0 else None,
        "note": EIGEN_ORDER_IDENTIFIABILITY_NOTE,
    }


def project_out(activations: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """``x - (x B) B^T``: the same tensor with its component in ``span(B)`` removed.

    A rank-0 basis returns the input bit for bit, which is what makes the null
    ablation a genuine identity rather than an identity up to arithmetic.
    """

    if basis.ndim != 2:
        raise ValueError(f"expected a (d_model, rank) basis, got {tuple(basis.shape)}")
    if activations.shape[-1] != basis.shape[0]:
        raise ValueError(
            f"activations of width {activations.shape[-1]} cannot be projected "
            f"against a basis of width {basis.shape[0]}"
        )
    if basis.shape[1] == 0:
        return activations
    working = activations.to(torch.float32)
    removed = working - (working @ basis) @ basis.T
    return removed.to(activations.dtype)


def random_orthonormal_basis(
    d_model: int,
    rank: int,
    *,
    generator: torch.Generator,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """A Haar-distributed ``rank``-dimensional subspace of ``R^d_model``.

    The QR factor of a Gaussian matrix. Uniform in the span, which is the property
    both the positive control and every chance level below need; the column signs
    are arbitrary and no statistic here reads them.
    """

    if rank < 1 or rank > d_model:
        raise ValueError(f"a random basis of rank {rank} in {d_model} dimensions")
    gaussian = torch.randn(
        (d_model, rank),
        generator=generator,
        device=generator.device,
        dtype=torch.float64,
    )
    factor, _ = torch.linalg.qr(gaussian)
    return factor.to(dtype=torch.float32, device=device or gaussian.device).contiguous()


def _overlap_values(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    """Both declared statistics from one singular-value decomposition."""

    cosines = torch.linalg.svdvals(left.to(torch.float64).T @ right.to(torch.float64))
    smaller = min(int(left.shape[1]), int(right.shape[1]))
    return (
        float((cosines**2).sum() / smaller),
        float(cosines[0]),
    )


def subspace_overlap(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    seed: int,
    chance_draws: int = 64,
) -> dict[str, Any]:
    """Both overlap statistics, each beside the chance level of its own shapes.

    **The chance level is not optional and is not a footnote.** Two random
    ``r``-dimensional subspaces of ``R^d`` have expected mean squared principal
    cosine exactly ``r/d`` -- 0.125 at ``r`` 512 in 4,096 dimensions -- so a raw
    overlap of 0.13 is a statement that the two subspaces are unrelated, and
    reporting it without ``r/d`` invites the opposite reading.

    The closed form is exact: for a fixed orthonormal ``A`` and Haar-random ``B``,
    ``E||A^T B||_F^2 = r_a r_b / d``, so the statistic's expectation is
    ``max(r_a, r_b) / d``. The Monte-Carlo band is drawn by randomising **one**
    side only, which is exactly right rather than a shortcut: the Haar measure is
    orthogonally invariant, so the joint law of two independent random subspaces
    and the law of one fixed subspace against one random one are the same. The
    first principal angle has no closed form and is reported against the band
    alone.
    """

    if left.shape[0] != right.shape[0]:
        raise ValueError("two subspaces of different ambient dimensions do not overlap")
    d_model = int(left.shape[0])
    rank_left, rank_right = int(left.shape[1]), int(right.shape[1])
    if rank_left < 1 or rank_right < 1:
        raise ValueError("an empty subspace has no principal angles")
    if chance_draws < 2:
        raise ValueError("a chance band needs at least two draws")
    mean_squared, first_cosine = _overlap_values(left, right)

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    drawn_mean: list[float] = []
    drawn_first: list[float] = []
    reference = random_orthonormal_basis(
        d_model, rank_left, generator=generator, device="cpu"
    )
    for _ in range(chance_draws):
        other = random_orthonormal_basis(
            d_model, rank_right, generator=generator, device="cpu"
        )
        one, two = _overlap_values(reference, other)
        drawn_mean.append(one)
        drawn_first.append(two)

    def band(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "p2.5": float(np.percentile(array, 2.5)),
            "p97.5": float(np.percentile(array, 97.5)),
        }

    return {
        "d_model": d_model,
        "rank_left": rank_left,
        "rank_right": rank_right,
        MEAN_SQUARED_COSINE: mean_squared,
        FIRST_PRINCIPAL_ANGLE_COSINE: first_cosine,
        "chance": {
            MEAN_SQUARED_COSINE: {
                "closed_form_mean": max(rank_left, rank_right) / d_model,
                "closed_form": "E||A^T B||_F^2 = r_a r_b / d, divided by min(r_a, r_b)",
                **band(drawn_mean),
            },
            FIRST_PRINCIPAL_ANGLE_COSINE: {
                "closed_form_mean": None,
                "closed_form": "none; Monte-Carlo band only",
                **band(drawn_first),
            },
            "n_draws": int(chance_draws),
            "seed": int(seed),
            "draw_note": (
                "one side is fixed and the other randomised, which is exact rather "
                "than a shortcut: the Haar measure is orthogonally invariant, so two "
                "independent random subspaces and one fixed against one random have "
                "the same joint law for every statistic of the principal angles"
            ),
        },
        "definitions": dict(OVERLAP_STATISTIC_DEFINITIONS),
    }


# ------------------------------------------------------------ hook hygiene


def intervention_invariants(
    logits_for_basis: Callable[[torch.Tensor | None], torch.Tensor],
    *,
    d_model: int,
    rank: int,
    layer: int,
    seed: int,
    tolerance: float,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Refuse to measure unless the projection is doing what it claims.

    ``logits_for_basis(None)`` must run the model with **no interceptor bound at
    all**; a basis argument binds one. That distinction is the whole of the first
    check -- comparing a rank-0 hook against a rank-0 hook would compare a function
    with itself, which is what ``src.transfer.das.invariants`` avoids by writing
    ``delta=None`` against ``delta=zeros``.

    Two checks, and the second is the one that matters. A hook that fails to bind
    passes every null test while silently measuring an unpatched model -- which is
    a failure ``src.transfer.path_patching`` has actually recorded -- so a null
    test alone is not evidence that the write happened. See
    :data:`RANDOM_CONTROL_NOTE` for why the positive control is a random subspace
    at the ladder's largest rank rather than a small one or a constant.
    """

    empty = torch.zeros((d_model, 0), dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    large = random_orthonormal_basis(d_model, rank, generator=generator, device=device)
    clean_logits = logits_for_basis(None)
    null_logits = logits_for_basis(empty)
    moved_logits = logits_for_basis(large)
    null_gap = float((clean_logits - null_logits).abs().max())
    moved_gap = float((clean_logits - moved_logits).abs().max())
    record = {
        "layer": int(layer),
        "null_projection_max_logit_gap": null_gap,
        "random_projection_max_logit_gap": moved_gap,
        "random_projection_rank": int(rank),
        "tolerance": float(tolerance),
        "control": RANDOM_CONTROL_NOTE,
        "projection": PROJECTION_NOTE,
    }
    if null_gap > tolerance:
        raise RuntimeError(
            f"layer {layer}: projecting out a rank-0 subspace moved the logits by "
            f"{null_gap:.3g}, so the intervention is not the identity when it must "
            "be and every measured effect is partly the hook itself"
        )
    if moved_gap <= tolerance:
        raise RuntimeError(
            f"layer {layer}: projecting out a random rank-{rank} subspace moved the "
            f"logits by {moved_gap:.3g}, which is inside the null tolerance. The hook "
            "is not bound to the block's feed-forward output. A null-only check "
            "cannot see this, which is why this positive control exists"
        )
    return record


# --------------------------------------------------- damage and its decomposition


@dataclass(frozen=True)
class ScoredPass:
    """One forward pass read at the measured positions.

    ``marginal`` is the *unnormalised* sum of the model's predictive distributions
    over the positions of each held-out half, so the two halves' estimates of
    ``q`` are built in one pass and neither is contaminated by the positions it
    scores.
    """

    label: str
    target_ids: np.ndarray
    nll_nats: np.ndarray
    group_ids: np.ndarray
    half_ids: np.ndarray
    marginal: np.ndarray
    marginal_counts: np.ndarray

    def __post_init__(self) -> None:
        n = self.target_ids.shape[0]
        for name in ("nll_nats", "group_ids", "half_ids"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{self.label}: {name} does not align with the targets")
        if self.marginal.ndim != 2 or self.marginal.shape[0] != 2:
            raise ValueError(f"{self.label}: the marginal must be (2, vocabulary)")
        if self.marginal_counts.shape != (2,):
            raise ValueError(f"{self.label}: the marginal counts must be (2,)")
        if not np.all(self.marginal_counts > 0):
            raise ValueError(
                f"{self.label}: one held-out half carries no positions, so the other "
                "half's targets have no q to be scored against. Draw more groups"
            )
        if not np.isfinite(self.nll_nats).all():
            raise ValueError(f"{self.label}: a non-finite per-position likelihood")

    @property
    def unigram_nll_nats(self) -> np.ndarray:
        """``-log q(y)`` with ``q`` estimated on the *other* half's positions."""

        q = self.marginal / self.marginal_counts[:, None]
        other = 1 - self.half_ids
        taken = q[other, self.target_ids]
        if not np.all(taken > 0.0):
            raise ValueError(
                f"{self.label}: the held-out marginal assigns exactly zero to a target "
                "that occurred, so its cross-entropy is infinite. This is an underflow "
                "in the accumulated predictive distribution, not a measurement"
            )
        return -np.log(taken)

    @property
    def plug_in_unigram_nll_nats(self) -> np.ndarray:
        """The same quantity estimated on the positions it scores.

        Reported only so the difference against the held-out estimator can be
        published as the measured bias at this cohort size (L12).
        """

        q = self.marginal.sum(axis=0) / self.marginal_counts.sum()
        taken = q[self.target_ids]
        if not np.all(taken > 0.0):
            raise ValueError(f"{self.label}: the plug-in marginal underflowed to zero")
        return -np.log(taken)

    @property
    def conditional_nll_nats(self) -> np.ndarray:
        """``-log[p(y|x)/q(y)]``: the negative pointwise context information."""

        return self.nll_nats - self.unigram_nll_nats

    def assert_aligned(self, other: "ScoredPass") -> None:
        """Two passes must be over the same positions in the same order."""

        if not np.array_equal(self.target_ids, other.target_ids):
            raise ValueError(
                f"{self.label} and {other.label} scored different targets, so the "
                "difference between them is not a paired damage"
            )
        if not np.array_equal(self.group_ids, other.group_ids):
            raise ValueError(f"{self.label} and {other.label} disagree on the groups")
        if not np.array_equal(self.half_ids, other.half_ids):
            raise ValueError(f"{self.label} and {other.label} disagree on the halves")


def _mean_of_predictions(_truth: np.ndarray, predictions: np.ndarray) -> float:
    """The metric :func:`src.transfer.statistics.paired_group_bootstrap` scores.

    That resampler is written for two prediction vectors against one truth, and
    the quantity wanted here is the difference of two per-position means. The
    truth vector is carried through as the target ids -- it is the row identity, so
    a mispairing would be visible -- and the metric ignores it. No new resampler is
    added to this package: the group bootstrap has one implementation and this is
    it.
    """

    return float(np.mean(predictions))


def damage_interval(
    ablated: np.ndarray,
    clean: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Mean ``ablated - clean`` over positions, with a paired group interval."""

    return paired_group_bootstrap(
        targets,
        ablated,
        clean,
        groups,
        _mean_of_predictions,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )


def unigram_decomposition(
    clean: ScoredPass,
    ablated: ScoredPass,
    *,
    seed: int,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Split one ablation's damage into its unigram half and its residual half.

    The split is **per position**, so ``total = unigram + residual`` holds exactly
    at every position, in every bootstrap draw, and for every subsample -- it is an
    identity of the logarithm and not a fitted decomposition:

        -log p(y|x)  =  -log q(y)  -  log[ p(y|x) / q(y) ]

    The residual is the second term's difference, which is the context information
    the ablation destroyed. **It is the only half that licenses a claim about
    computation.**
    """

    clean.assert_aligned(ablated)
    total = damage_interval(
        ablated.nll_nats, clean.nll_nats, clean.target_ids, clean.group_ids,
        seed=seed, n_bootstrap=n_bootstrap,
    )
    unigram = damage_interval(
        ablated.unigram_nll_nats, clean.unigram_nll_nats,
        clean.target_ids, clean.group_ids, seed=seed, n_bootstrap=n_bootstrap,
    )
    residual = damage_interval(
        ablated.conditional_nll_nats, clean.conditional_nll_nats,
        clean.target_ids, clean.group_ids, seed=seed, n_bootstrap=n_bootstrap,
    )
    plug_in_damage = float(
        np.mean(ablated.plug_in_unigram_nll_nats - clean.plug_in_unigram_nll_nats)
    )
    total_damage = float(total["difference"])
    residual_damage = float(residual["difference"])
    return {
        "total_damage_nats": total_damage,
        "total": total,
        "unigram_damage_nats": float(unigram["difference"]),
        "unigram": unigram,
        "residual_damage_nats": residual_damage,
        "residual": residual,
        "residual_share": (
            residual_damage / total_damage if abs(total_damage) > 1e-12 else None
        ),
        "decomposition_closes_to_nats": abs(
            total_damage - float(unigram["difference"]) - residual_damage
        ),
        "plug_in_unigram_damage_nats": plug_in_damage,
        "plug_in_minus_held_out_nats": plug_in_damage - float(unigram["difference"]),
        "estimator": UNIGRAM_ESTIMATOR_NOTE,
        "group_unit": GROUP_UNIT_NOTE,
        "n_groups": int(total["n_groups"]),
        "minimum_groups": int(MINIMUM_BOOTSTRAP_UNITS),
    }


def cohort_unigram_reference(target_ids: np.ndarray, vocabulary: int) -> dict[str, Any]:
    """The cohort's own context-free floor, plug-in and bias-corrected.

    The quantity L12 is literally about, computed through
    :mod:`src.transfer.budget` rather than restated. It is a property of the
    measured positions and does not move with an ablation, so it is reported once
    per cell as the reference every damage figure sits against.
    """

    plug_in = unigram_entropy_nats(target_ids, vocabulary)
    corrected = miller_madow_entropy_nats(target_ids, vocabulary)
    return {
        "plug_in_entropy_nats": plug_in,
        "miller_madow_entropy_nats": corrected,
        "miller_madow_correction_nats": corrected - plug_in,
        "n_positions": int(target_ids.size),
        "n_distinct_targets": int(np.unique(target_ids).size),
        "vocabulary": int(vocabulary),
        "note": (
            "the cohort's own target-count entropy, not the model's marginal. "
            "Reported bias-corrected because L12 prices the plug-in estimator at up "
            "to +1.02 nats on a protein arm, and the correction grows with the "
            "vocabulary against the sample"
        ),
    }


COHORT_CONTEXT_INFORMATION_NOTE = (
    "the corpus-unigram-referenced context information -- the cohort's own "
    "target-count entropy minus the model's clean cross-entropy over the same "
    "scored positions -- computed HERE, on the cohort whose damage figures this "
    "artefact reports. It is the estimand EXP-R2-152 reports and it is NOT that "
    "run's number: a different eligible set is a different cohort. It is also not "
    "budget.arm_power's held-out `context_information_nats`, whose reference is a "
    "disjoint smoothed unigram rather than the scored cohort's own counts; each "
    "field here names the reference it used. Nothing in this stage gates on it"
)


def cohort_context_information(
    unigram_reference: Mapping[str, Any], clean_nll_nats: float
) -> dict[str, Any]:
    """One mode's context information against its own cohort's unigram entropy.

    Both operands are already measured on this run -- the reference by
    :func:`cohort_unigram_reference` over the scored targets, the cross-entropy by
    the clean pass over the same positions -- so this quantity is reported rather
    than declared, and it moves with the cohort it describes. That is the whole
    reason it exists: a figure hand-carried from another stage's cohort describes a
    population this artefact never measured.

    Reported at both estimators of the reference, because L12 prices the plug-in
    one at up to +1.02 nats on a protein arm and a single figure would hide which
    estimator produced it.
    """

    clean = float(clean_nll_nats)
    plug_in = float(unigram_reference["plug_in_entropy_nats"])
    corrected = float(unigram_reference["miller_madow_entropy_nats"])
    return {
        "context_information_plug_in_reference_nats": plug_in - clean,
        "context_information_miller_madow_reference_nats": corrected - clean,
        "clean_cross_entropy_nats": clean,
        "note": COHORT_CONTEXT_INFORMATION_NOTE,
    }


def necessary_rank(
    ladder: Sequence[int],
    total_damage_per_rank: Sequence[float],
    full_ablation_damage: float,
    rule: DecisionRule,
) -> dict[str, Any]:
    """The smallest rung reaching ``necessity_fraction`` of the full block ablation.

    The denominator is an attainability check before it is a normaliser (Appendix
    B rule 2). A layer whose whole feed-forward write can be zeroed without
    damaging the mode has no necessary subspace to find at all, and this says so
    instead of dividing by a number near zero and reporting whichever rung the
    noise happened to cross.
    """

    if len(ladder) != len(total_damage_per_rank):
        raise ValueError("the ladder and its damage curve must have equal length")
    attainable = float(full_ablation_damage) > 0.0
    target = rule.necessity_fraction * float(full_ablation_damage)
    reached = None
    if attainable:
        for rank, damage in zip(ladder, total_damage_per_rank):
            if float(damage) >= target:
                reached = int(rank)
                break
    return {
        "necessary_rank": reached,
        "full_ablation_damage_nats": float(full_ablation_damage),
        "target_damage_nats": target if attainable else None,
        "necessity_fraction": rule.necessity_fraction,
        "attainable": attainable,
        "withheld_reason": (
            None
            if attainable
            else "zeroing this layer's whole feed-forward write does not damage this "
            "mode, so there is no attainable denominator and a necessary subspace is "
            "undefined at this site"
        ),
        "ladder_exhausted": bool(attainable and reached is None),
    }


def _excludes_zero(interval: Sequence[float]) -> bool:
    low, high = float(interval[0]), float(interval[1])
    return low > 0.0 or high < 0.0


def _positive_and_excludes_zero(record: Mapping[str, Any]) -> bool:
    return float(record["difference"]) > 0.0 and _excludes_zero(record["difference_ci95"])


def layer_verdict(
    *,
    layer: int,
    modes: Sequence[str],
    own: Mapping[str, Mapping[str, Any]],
    asymmetry: Mapping[str, Mapping[str, Any]],
    overlap: Mapping[str, Any] | str | None,
    attainable: Mapping[str, bool],
    invariants_held: bool,
    rule: DecisionRule,
    statistic: str,
) -> dict[str, Any]:
    """One layer's verdict, from clauses that are each recorded beside it.

    ``own`` maps a mode to the :func:`unigram_decomposition` of ablating that
    mode's own necessary subspace and evaluating that same mode. ``asymmetry``
    maps a mode to the paired own-minus-other residual contrast **inside that
    mode's own positions**, which is the only form the contrast is paired in: two
    modes score different symbols, so a cross-mode difference of two means is not
    a paired quantity and is not used as one here.
    """

    if statistic not in OVERLAP_STATISTICS:
        raise ValueError(f"unknown overlap statistic {statistic!r}")
    clauses: dict[str, Any] = {
        "instrument_invariants_held": bool(invariants_held),
        "full_ablation_attainable": {mode: bool(attainable[mode]) for mode in modes},
    }
    if not invariants_held or not all(attainable[mode] for mode in modes):
        return {
            "layer": int(layer),
            "verdict": "VOID_INSTRUMENT",
            "reading": VERDICT_READINGS["VOID_INSTRUMENT"],
            "clauses": clauses,
            "decision_rule": rule.record(),
        }

    total_significant = {
        mode: _positive_and_excludes_zero(own[mode]["total"]) for mode in modes
    }
    residual_significant = {
        mode: _positive_and_excludes_zero(own[mode]["residual"]) for mode in modes
    }
    shares = {mode: own[mode]["residual_share"] for mode in modes}
    share_clears = {
        mode: shares[mode] is not None and float(shares[mode]) >= rule.residual_share_floor
        for mode in modes
    }
    asymmetric = {
        mode: isinstance(asymmetry[mode], Mapping)
        and _positive_and_excludes_zero(asymmetry[mode])
        for mode in modes
    }
    clauses.update(
        {
            "total_damage_excludes_zero": total_significant,
            "residual_damage_excludes_zero": residual_significant,
            "residual_share": {mode: shares[mode] for mode in modes},
            "residual_share_clears_floor": share_clears,
            "own_minus_other_residual_excludes_zero": asymmetric,
        }
    )

    if isinstance(overlap, Mapping):
        measured = float(overlap[statistic])
        upper = float(overlap["chance"][statistic]["p97.5"])
        above = measured > upper * (1.0 + rule.overlap_margin)
        within = measured <= upper
        clauses["overlap"] = {
            "statistic": statistic,
            "measured": measured,
            "chance_p97.5": upper,
            "chance_closed_form_mean": overlap["chance"][statistic]["closed_form_mean"],
            "margin": rule.overlap_margin,
            "above_chance_band_by_margin": bool(above),
            "within_chance_band": bool(within),
        }
    else:
        above = False
        within = False
        clauses["overlap"] = overlap

    all_total = all(total_significant.values())
    all_residual = all(residual_significant.values())
    all_shares = all(share_clears.values())
    all_asymmetric = all(asymmetric.values())
    no_asymmetry = not any(asymmetric.values())

    # The order matters and each branch answers a different question. There is
    # nothing to decompose before there is damage; the unigram clause is decided
    # before either subspace verdict, because a damage that does not survive the
    # decomposition cannot support a claim about computation whatever the overlap
    # does; and MIXED is the residue rather than a default, because the remaining
    # clauses are about different things and forcing them into one verdict would
    # report a rule's own thresholds as a finding.
    if not all_total:
        verdict = "NO_MEASURED_DAMAGE"
    elif not all_shares:
        verdict = "UNIGRAM_ONLY"
    elif all_residual and all_asymmetric and within:
        verdict = "DISTINCT_SUBSPACES"
    elif all_residual and no_asymmetry and above:
        verdict = "SHARED_SUBSPACE"
    else:
        verdict = "MIXED"
    return {
        "layer": int(layer),
        "verdict": verdict,
        "reading": VERDICT_READINGS[verdict],
        "clauses": clauses,
        "decision_rule": rule.record(),
    }


# ------------------------------------------------------- the known-answer fixture


#: Upper bound on the residual share the synthetic's planted **bias** block may
#: reach, and lower bound on the share its planted **context** block must reach.
#: Measured on the fixture at its declared configuration and set with margin: the
#: bias block reads 0.108 and 0.080 in the two modes and the context block reads
#: 0.969 and 0.930, so a decomposition that had the two halves confused fails by an
#: order of magnitude rather than by a hair.
#:
#: **Neither bound is zero or one, and that is a property of a softmax rather than
#: slack in the fixture.** Removing a constant vector ``c`` from the logits gives
#: ``p_abl(y|x) = p(y|x) e^{-c_y} / Z(x)`` with a normaliser that depends on ``x``,
#: so an exponential tilt of the output prior is never *exactly* a change to the
#: marginal alone: no intervention on a softmax model is exactly unigram-only
#: except the trivial ones. The measured 0.08-0.11 is that second-order term at
#: this fixture's scale, and it is why the decision rule's clause is a share and
#: not a test against zero.
SYNTHETIC_BIAS_RESIDUAL_SHARE_MAX = 0.20
SYNTHETIC_CONTEXT_RESIDUAL_SHARE_MIN = 0.60

#: How close the recovered context-subspace overlap must land to the planted one.
SYNTHETIC_OVERLAP_TOLERANCE = 0.05


@dataclass(frozen=True)
class SyntheticDesign:
    """A planted two-mode geometry whose every answer is known before the run.

    The construction, and what each piece is for:

    * ``context_rank`` directions per mode carry the mode's computation, and
      ``shared`` of them are literally the same directions in both modes. The
      planted mean squared principal cosine between the two context subspaces is
      therefore exactly ``shared / context_rank``, against a chance level of
      ``context_rank / d_model``.
    * ``bias_rank`` further directions per mode carry a **constant** coefficient,
      so they add a fixed vector to the logits at every position. Projecting them
      out moves the marginal output distribution and destroys almost no context
      information -- which is exactly the state the unigram control exists to
      separate from a computation.
    * the readout is ``U_mode @ S_mode^T``, so a logit depends on the activation
      only through its coordinates in the planted subspace, and targets are
      **sampled from the clean model itself**, which makes the clean model the true
      generating distribution and every damage figure non-negative in expectation.

    **What the bias block is and is not.** It validates the arithmetic of the
    decomposition on data where the answer is known. It is not a claim that a
    unigram-only direction in a real transformer looks like a constant offset: the
    block normalisations that make a constant residual perturbation nearly
    invisible (``src.transfer.das.invariants``) have no counterpart here, and this
    fixture has no normalisation at all.
    """

    d_model: int = 96
    context_rank: int = 12
    shared: int = 6
    bias_rank: int = 3
    vocabulary: int = 64
    n_groups: int = 24
    positions_per_group: int = 24
    bias_level: float = 0.6
    noise: float = 0.02
    temperature: float = 0.5
    spectrum_decay: float = 0.85
    seed: int = 20260819

    def __post_init__(self) -> None:
        if not 0 <= self.shared <= self.context_rank:
            raise ValueError("the shared block must fit inside the context block")
        needed = self.shared + 2 * (self.context_rank - self.shared) + 2 * self.bias_rank
        if needed > self.d_model:
            raise ValueError(
                f"the planted geometry needs {needed} orthonormal directions and "
                f"d_model is {self.d_model}"
            )
        if self.n_groups < MINIMUM_BOOTSTRAP_UNITS:
            raise ValueError(
                f"{self.n_groups} groups is below the {MINIMUM_BOOTSTRAP_UNITS}-unit "
                "bootstrap floor this package holds every interval to"
            )
        if self.context_rank < 2:
            raise ValueError("a context block of one direction has no ladder to climb")
        if not 0.0 < self.spectrum_decay <= 1.0:
            raise ValueError("the spectrum decay must lie in (0, 1]")

    @property
    def planted_overlap(self) -> float:
        return self.shared / self.context_rank

    @property
    def chance_overlap(self) -> float:
        return self.context_rank / self.d_model

    @property
    def n_positions(self) -> int:
        return self.n_groups * self.positions_per_group

    def record(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "context_rank": self.context_rank,
            "shared": self.shared,
            "bias_rank": self.bias_rank,
            "vocabulary": self.vocabulary,
            "n_groups": self.n_groups,
            "positions_per_group": self.positions_per_group,
            "n_positions": self.n_positions,
            "bias_level": self.bias_level,
            "noise": self.noise,
            "temperature": self.temperature,
            "spectrum_decay": self.spectrum_decay,
            "seed": self.seed,
            "planted_mean_squared_cosine": self.planted_overlap,
            "chance_mean_squared_cosine": self.chance_overlap,
        }


@dataclass(frozen=True)
class _SyntheticMode:
    context: torch.Tensor
    bias: torch.Tensor
    readout: torch.Tensor
    activations: torch.Tensor
    targets: np.ndarray
    groups: np.ndarray
    halves: np.ndarray


def _build_synthetic_mode(
    design: SyntheticDesign,
    context: torch.Tensor,
    bias: torch.Tensor,
    generator: torch.Generator,
) -> _SyntheticMode:
    readout = torch.randn(
        (design.vocabulary, design.context_rank + design.bias_rank),
        generator=generator,
        dtype=torch.float64,
    )
    # A DECAYING coefficient spectrum, and it is not decoration. A top-r principal
    # basis is identified only up to the ordering of its eigenvalues, so on a block
    # of near-equal variances the top-r subspace is not a property of the data at
    # all: two modes whose occupied spans are literally identical then recover
    # different top-r subspaces, and the overlap statistic reads 8/12 where the
    # truth is 1. That is a real limitation of this method at ranks where the
    # spectrum is flat (:data:`EIGEN_ORDER_IDENTIFIABILITY_NOTE`), it is what a
    # flat fixture measured before this line existed, and a fixture that reproduced
    # it would be validating the instrument against its own weakest regime rather
    # than against a known answer.
    scale = torch.pow(
        torch.tensor(float(design.spectrum_decay), dtype=torch.float64),
        torch.arange(design.context_rank, dtype=torch.float64),
    )
    coefficients = scale * torch.randn(
        (design.n_positions, design.context_rank), generator=generator, dtype=torch.float64
    )
    bias_coefficients = torch.full(
        (design.n_positions, design.bias_rank), float(design.bias_level), dtype=torch.float64
    )
    activations = (
        coefficients @ context.T
        + bias_coefficients @ bias.T
        + design.noise
        * torch.randn(
            (design.n_positions, design.d_model), generator=generator, dtype=torch.float64
        )
    )
    logits = design.temperature * (
        torch.cat([coefficients, bias_coefficients], dim=1) @ readout.T
    )
    targets = torch.multinomial(
        torch.softmax(logits, dim=-1), num_samples=1, generator=generator
    )[:, 0]
    groups = np.repeat(np.arange(design.n_groups), design.positions_per_group)
    halves = (groups % 2).astype(np.int64)
    return _SyntheticMode(
        context=context,
        bias=bias,
        readout=readout,
        activations=activations,
        targets=targets.numpy().astype(np.int64),
        groups=groups,
        halves=halves,
    )


def _synthetic_pass(
    design: SyntheticDesign, mode: _SyntheticMode, basis: torch.Tensor, label: str
) -> ScoredPass:
    """One 'forward pass' of the toy, through the same projection the real path uses."""

    hidden = project_out(mode.activations.to(torch.float32), basis).to(torch.float64)
    full = torch.cat([mode.context, mode.bias], dim=1)
    logits = design.temperature * (hidden @ full) @ mode.readout.T
    log_probabilities = torch.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp().numpy()
    targets = mode.targets
    marginal = np.stack(
        [probabilities[mode.halves == half].sum(axis=0) for half in (0, 1)]
    )
    counts = np.asarray([int((mode.halves == half).sum()) for half in (0, 1)], dtype=np.int64)
    return ScoredPass(
        label=label,
        target_ids=targets,
        nll_nats=-log_probabilities.numpy()[np.arange(targets.size), targets],
        group_ids=mode.groups,
        half_ids=mode.halves,
        marginal=marginal,
        marginal_counts=counts,
    )


def _interleaved_context(
    shared_block: torch.Tensor, exclusive_block: torch.Tensor
) -> torch.Tensor:
    """One mode's context columns, shared and exclusive spread evenly through the order.

    The order is the variance order, because :func:`_build_synthetic_mode` gives
    column ``j`` a coefficient scale of ``spectrum_decay ** j``. Putting the shared
    block first would make the fixture's own *necessary* subspaces shared whatever
    the planted split was -- the top rungs of the ladder would be nothing but the
    shared directions -- so the half-shared design could never produce a partly
    shared necessary subspace and the rule's MIXED outcome would be unreachable on
    it. Spreading the two evenly puts the planted proportion into every prefix of
    the order, which is what makes the top-r overlap read the planted split.
    """

    n_shared = int(shared_block.shape[1])
    n_exclusive = int(exclusive_block.shape[1])
    columns: list[torch.Tensor] = []
    taken_shared = taken_exclusive = 0
    for _ in range(n_shared + n_exclusive):
        take_shared = taken_shared < n_shared and (
            taken_exclusive >= n_exclusive
            or taken_shared * n_exclusive <= taken_exclusive * n_shared
        )
        if take_shared:
            columns.append(shared_block[:, taken_shared : taken_shared + 1])
            taken_shared += 1
        else:
            columns.append(exclusive_block[:, taken_exclusive : taken_exclusive + 1])
            taken_exclusive += 1
    return torch.cat(columns, dim=1)


def _planted_frame(design: SyntheticDesign) -> dict[str, torch.Tensor]:
    """The orthonormal blocks the fixture is built from, in one place."""

    generator = torch.Generator(device="cpu").manual_seed(int(design.seed))
    span = design.shared + 2 * (design.context_rank - design.shared) + 2 * design.bias_rank
    frame, _ = torch.linalg.qr(
        torch.randn((design.d_model, span), generator=generator, dtype=torch.float64)
    )
    exclusive = design.context_rank - design.shared
    cursor = 0

    def take(width: int) -> torch.Tensor:
        nonlocal cursor
        block = frame[:, cursor : cursor + width]
        cursor += width
        return block

    shared_block = take(design.shared)
    text_only = take(exclusive)
    protein_only = take(exclusive)
    blocks = {
        "text_context": _interleaved_context(shared_block, text_only),
        "protein_context": _interleaved_context(shared_block, protein_only),
        "text_bias": take(design.bias_rank),
        "protein_bias": take(design.bias_rank),
    }
    blocks["generator_state"] = generator  # type: ignore[assignment]
    return blocks


def synthetic_certificate(
    design: SyntheticDesign,
    *,
    ladder: Sequence[int],
    rule: DecisionRule,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Run the whole pipeline on planted geometry and report recovered against planted.

    Every function a real run calls is called here on data whose answer exists
    before the run: the covariance accumulation, the principal basis, the
    projection, the per-position unigram decomposition, the paired group interval,
    the overlap statistic with its chance band, the necessary-rank rule and the
    per-layer verdict. **This is the only place any of it is falsifiable**, because
    nothing in a real artefact could contradict a subspace claim, and it is run
    before a campaign rather than after one.

    Six things are checked, each reported with its own verdict beside it:

    1. the recovered context subspaces' overlap lands on the planted value;
    2. the chance level reproduces its closed form, and the measured overlap
       stands clear of it -- so the overlap null does not fire;
    3. the null ablation is a no-op to the last bit;
    4. ablating a mode's own recovered directions damages that mode far more than
       a random subspace of the same rank does -- so the necessity null does not
       fire either;
    5. the unigram decomposition assigns a planted pure-bias block to the unigram
       half and a planted context block to the residual half, which is the clause
       the headline claim rests on;
    6. the decomposition closes: total minus unigram minus residual is zero to
       float precision at every cell.
    """

    blocks = _planted_frame(design)
    generator: torch.Generator = blocks["generator_state"]  # type: ignore[assignment]
    modes = {
        "text": _build_synthetic_mode(
            design, blocks["text_context"], blocks["text_bias"], generator
        ),
        "protein": _build_synthetic_mode(
            design, blocks["protein_context"], blocks["protein_bias"], generator
        ),
    }
    names = ("text", "protein")
    other_of = {"text": "protein", "protein": "text"}
    rungs_available = [int(rank) for rank in ladder if rank <= design.d_model]
    if not rungs_available:
        raise ValueError("no rung of the ladder fits inside the fixture's d_model")

    from .spectrum import CovarianceAccumulator, spectrum_statistics

    occupancy: dict[str, Any] = {}
    recovered: dict[str, torch.Tensor] = {}
    for name in names:
        accumulator = CovarianceAccumulator(n_layers=1, d_model=design.d_model, device="cpu")
        accumulator.update(modes[name].activations[None, :, :])
        covariance = accumulator.covariance_at(0)
        occupancy[name] = spectrum_statistics(
            torch.linalg.eigvalsh(covariance),
            n_samples=design.n_positions,
            d_model=design.d_model,
        )
        recovered[name] = principal_basis(covariance, design.context_rank)

    planted_overlap = subspace_overlap(
        recovered["text"], recovered["protein"], seed=design.seed + 1, chance_draws=64
    )
    recovered_overlap = float(planted_overlap[MEAN_SQUARED_COSINE])
    overlap_error = abs(recovered_overlap - design.planted_overlap)

    empty = torch.zeros((design.d_model, 0), dtype=torch.float32)
    whole_space = torch.eye(design.d_model, dtype=torch.float32)
    control_generator = torch.Generator(device="cpu").manual_seed(int(design.seed) + 2)
    clean = {name: _synthetic_pass(design, modes[name], empty, f"{name}:clean") for name in names}
    full = {
        name: _synthetic_pass(design, modes[name], whole_space, f"{name}:full")
        for name in names
    }

    necessity: dict[str, Any] = {}
    for name in names:
        rungs = []
        for rank in rungs_available:
            own_pass = _synthetic_pass(
                design, modes[name], recovered[name][:, :rank], f"{name}:own:r{rank}"
            )
            other_pass = _synthetic_pass(
                design,
                modes[name],
                recovered[other_of[name]][:, :rank],
                f"{name}:other:r{rank}",
            )
            random_pass = _synthetic_pass(
                design,
                modes[name],
                random_orthonormal_basis(
                    design.d_model, rank, generator=control_generator
                ),
                f"{name}:random:r{rank}",
            )
            rungs.append(
                {
                    "rank": int(rank),
                    "own": unigram_decomposition(
                        clean[name], own_pass, seed=design.seed + rank,
                        n_bootstrap=n_bootstrap,
                    ),
                    "other_mode_basis": unigram_decomposition(
                        clean[name], other_pass, seed=design.seed + rank,
                        n_bootstrap=n_bootstrap,
                    ),
                    "random_control": unigram_decomposition(
                        clean[name], random_pass, seed=design.seed + rank,
                        n_bootstrap=n_bootstrap,
                    ),
                    "own_minus_other_residual": damage_interval(
                        own_pass.conditional_nll_nats,
                        other_pass.conditional_nll_nats,
                        clean[name].target_ids,
                        clean[name].group_ids,
                        seed=design.seed + rank,
                        n_bootstrap=n_bootstrap,
                    ),
                }
            )
        necessity[name] = {
            "null_ablation_damage_nats": float(
                np.mean(
                    _synthetic_pass(design, modes[name], empty, f"{name}:null").nll_nats
                    - clean[name].nll_nats
                )
            ),
            "full_ablation": unigram_decomposition(
                clean[name], full[name], seed=design.seed + 5, n_bootstrap=n_bootstrap
            ),
            "rungs": rungs,
            "cohort_unigram_reference": cohort_unigram_reference(
                clean[name].target_ids, design.vocabulary
            ),
        }
        necessity[name]["necessary_rank"] = necessary_rank(
            [rung["rank"] for rung in rungs],
            [rung["own"]["total_damage_nats"] for rung in rungs],
            necessity[name]["full_ablation"]["total_damage_nats"],
            rule,
        )

    unigram_control: dict[str, Any] = {}
    for name in names:
        bias_pass = _synthetic_pass(
            design, modes[name], modes[name].bias.to(torch.float32), f"{name}:bias"
        )
        context_pass = _synthetic_pass(
            design, modes[name], modes[name].context.to(torch.float32), f"{name}:context"
        )
        unigram_control[name] = {
            "planted_bias_block": unigram_decomposition(
                clean[name], bias_pass, seed=design.seed + 3, n_bootstrap=n_bootstrap
            ),
            "planted_context_block": unigram_decomposition(
                clean[name], context_pass, seed=design.seed + 4, n_bootstrap=n_bootstrap
            ),
        }

    chosen = {
        name: necessity[name]["necessary_rank"]["necessary_rank"] or rungs_available[-1]
        for name in names
    }
    necessary_overlap: dict[str, Any] | str = subspace_overlap(
        recovered["text"][:, : chosen["text"]],
        recovered["protein"][:, : chosen["protein"]],
        seed=design.seed + 6,
        chance_draws=64,
    )
    own_at_rank = {
        name: next(
            rung["own"] for rung in necessity[name]["rungs"] if rung["rank"] == chosen[name]
        )
        for name in names
    }
    asymmetry_at_rank = {
        name: next(
            rung["own_minus_other_residual"]
            for rung in necessity[name]["rungs"]
            if rung["rank"] == chosen[name]
        )
        for name in names
    }
    verdict = layer_verdict(
        layer=0,
        modes=names,
        own=own_at_rank,
        asymmetry=asymmetry_at_rank,
        overlap=necessary_overlap,
        attainable={
            name: necessity[name]["necessary_rank"]["attainable"] for name in names
        },
        invariants_held=True,
        rule=rule,
        statistic=MEAN_SQUARED_COSINE,
    )

    top_rank = rungs_available[-1]
    closure = max(
        float(cell["decomposition_closes_to_nats"])
        for name in names
        for cell in (
            unigram_control[name]["planted_bias_block"],
            unigram_control[name]["planted_context_block"],
            necessity[name]["full_ablation"],
            *[rung["own"] for rung in necessity[name]["rungs"]],
        )
    )
    checks: dict[str, Any] = {
        "overlap_recovers_the_planted_value": {
            "planted": design.planted_overlap,
            "recovered": recovered_overlap,
            "absolute_error": overlap_error,
            "tolerance": SYNTHETIC_OVERLAP_TOLERANCE,
            "passed": overlap_error <= SYNTHETIC_OVERLAP_TOLERANCE,
        },
        "chance_level_reproduces_the_closed_form": {
            "closed_form": planted_overlap["chance"][MEAN_SQUARED_COSINE][
                "closed_form_mean"
            ],
            "monte_carlo_mean": planted_overlap["chance"][MEAN_SQUARED_COSINE]["mean"],
            "passed": abs(
                planted_overlap["chance"][MEAN_SQUARED_COSINE]["closed_form_mean"]
                - planted_overlap["chance"][MEAN_SQUARED_COSINE]["mean"]
            )
            <= 0.25 * design.chance_overlap,
        },
        "overlap_null_does_not_fire": {
            "recovered": recovered_overlap,
            "chance_p97.5": planted_overlap["chance"][MEAN_SQUARED_COSINE]["p97.5"],
            "note": "vacuous by construction when nothing is planted (shared = 0), "
            "where the correct answer IS the chance level; the check is then reported "
            "as not applicable rather than as a pass",
            "applicable": design.shared > 0,
            "passed": (
                recovered_overlap
                > planted_overlap["chance"][MEAN_SQUARED_COSINE]["p97.5"]
                * (1.0 + rule.overlap_margin)
            )
            if design.shared > 0
            else True,
        },
        "null_ablation_is_a_no_op": {
            **{name: necessity[name]["null_ablation_damage_nats"] for name in names},
            "passed": all(
                abs(necessity[name]["null_ablation_damage_nats"]) <= rule.logit_tolerance
                for name in names
            ),
        },
        "decomposition_closes": {
            "max_absolute_residual_nats": closure,
            "tolerance": 1e-9,
            "passed": closure <= 1e-9,
        },
        "own_basis_beats_the_random_control": {},
        "bias_block_reads_as_unigram": {},
        "context_block_reads_as_residual": {},
    }
    for name in names:
        rung = next(r for r in necessity[name]["rungs"] if r["rank"] == top_rank)
        checks["own_basis_beats_the_random_control"][name] = {
            "rank": top_rank,
            "own_total_damage_nats": rung["own"]["total_damage_nats"],
            "random_total_damage_nats": rung["random_control"]["total_damage_nats"],
            "passed": rung["own"]["total_damage_nats"]
            > 2.0 * rung["random_control"]["total_damage_nats"],
        }
        bias_share = unigram_control[name]["planted_bias_block"]["residual_share"]
        context_share = unigram_control[name]["planted_context_block"]["residual_share"]
        checks["bias_block_reads_as_unigram"][name] = {
            "residual_share": bias_share,
            "bound": SYNTHETIC_BIAS_RESIDUAL_SHARE_MAX,
            "passed": bias_share is not None
            and bias_share <= SYNTHETIC_BIAS_RESIDUAL_SHARE_MAX,
        }
        checks["context_block_reads_as_residual"][name] = {
            "residual_share": context_share,
            "bound": SYNTHETIC_CONTEXT_RESIDUAL_SHARE_MIN,
            "passed": context_share is not None
            and context_share >= SYNTHETIC_CONTEXT_RESIDUAL_SHARE_MIN,
        }
    for key in (
        "own_basis_beats_the_random_control",
        "bias_block_reads_as_unigram",
        "context_block_reads_as_residual",
    ):
        checks[key]["passed"] = all(checks[key][name]["passed"] for name in names)
    passed = all(bool(entry["passed"]) for entry in checks.values())
    return {
        "kind": "synthetic_known_answer_check",
        "design": design.record(),
        "ladder": rungs_available,
        "decision_rule": rule.record(),
        "occupancy": occupancy,
        "overlap_of_recovered_subspaces": planted_overlap,
        "overlap_of_necessary_subspaces": necessary_overlap,
        "necessary_rank_used": {name: int(chosen[name]) for name in names},
        "necessity": necessity,
        "unigram_control": unigram_control,
        "verdict": verdict,
        "checks": checks,
        "certificate": "PASSED" if passed else "FAILED",
        "note": (
            "planted geometry, not a model. Every number here is a statement about "
            "this instrument and none of them is a statement about any checkpoint"
        ),
    }


#: The ladder the verdict-attainability corners are run at, declared here rather
#: than taken from a campaign's ``--rank-ladder``.
#:
#: **Because a coarse ladder turns a corner into MIXED, and the reason is worth
#: recording.** With rungs ``(1, 12)`` the fully-shared corner's necessary rank
#: lands on 12, which is the whole planted computation: the ablation saturates,
#: and in that regime the paired own-minus-other contrast resolves the *estimation
#: noise between two bases of the same span* -- the two read a mean squared cosine
#: of 0.9987 and the contrast still excludes zero. That is a true statement about a
#: saturated rung and not about the rule, so the attainability certificate is run
#: at a ladder fine enough to place the necessary rank below saturation, and the
#: hazard is recorded as :data:`SATURATED_RUNG_NOTE` rather than papered over.
SYNTHETIC_ATTAINABILITY_LADDER: tuple[int, ...] = (1, 2, 4, 8, 12)

#: The three planted geometries the verdict must be able to return, and what each
#: must return. Appendix B rule 2 applied to a verdict rather than to a threshold:
#: a decision rule that cannot reach one of its own outcomes on data built to
#: produce it has been decided before the measurement, and the two corner cases are
#: the ones a half-shared fixture would never expose. NO_MEASURED_DAMAGE and
#: VOID_INSTRUMENT are states of the instrument rather than of the geometry and are
#: unreachable on a fixture that plants an effect, so they are exercised directly
#: against :func:`layer_verdict`.
SYNTHETIC_VERDICT_CORNERS: tuple[tuple[str, int | None, str], ...] = (
    ("nothing_shared", 0, "DISTINCT_SUBSPACES"),
    ("half_shared", None, "MIXED"),
    ("fully_shared", -1, "SHARED_SUBSPACE"),
)


def synthetic_verdict_attainability(
    design: SyntheticDesign,
    *,
    rule: DecisionRule,
    ladder: Sequence[int] = SYNTHETIC_ATTAINABILITY_LADDER,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Every verdict the rule can return, on geometry planted to return it.

    ``half_shared`` is the design as declared and must read ``MIXED``: two
    subspaces sharing half their directions are neither distinct nor shared, and a
    rule that called them either would be reporting its own thresholds. The two
    corners move only ``shared`` and change nothing else.
    """

    import dataclasses

    cells: dict[str, Any] = {}
    for name, shared, expected in SYNTHETIC_VERDICT_CORNERS:
        if shared is None:
            variant = design
        else:
            variant = dataclasses.replace(
                design, shared=design.context_rank if shared < 0 else shared
            )
        result = synthetic_certificate(
            variant, ladder=ladder, rule=rule, n_bootstrap=n_bootstrap
        )
        cells[name] = {
            "shared": variant.shared,
            "planted_mean_squared_cosine": variant.planted_overlap,
            "expected_verdict": expected,
            "verdict": result["verdict"]["verdict"],
            "clauses": result["verdict"]["clauses"],
            "certificate": result["certificate"],
            "passed": result["verdict"]["verdict"] == expected
            and result["certificate"] == "PASSED",
        }
    return {
        "cells": cells,
        "ladder": [int(rank) for rank in ladder],
        "ladder_note": (
            "declared by this module and not taken from a campaign's --rank-ladder: "
            "a ladder too coarse to place the necessary rank below saturation turns "
            "the fully-shared corner into MIXED, which is a statement about that "
            "ladder rather than about the rule. See SATURATED_RUNG_NOTE"
        ),
        "saturation": SATURATED_RUNG_NOTE,
        "passed": all(cell["passed"] for cell in cells.values()),
        "rule": (
            "a verdict a planted geometry cannot reach is a verdict the rule decided "
            "in advance. The three GEOMETRY-determined outcomes are exercised here "
            "and the declared design is the one that must return MIXED; the "
            "remaining two -- NO_MEASURED_DAMAGE and VOID_INSTRUMENT -- are states of "
            "the instrument rather than of the planted geometry and no fixture that "
            "plants an effect can reach them, so they are exercised as unit tests of "
            "layer_verdict instead"
        ),
    }
