"""Text-to-protein mechanistic-interpretability transfer measurements.

This package supports one question: when circuit-level interpretability methods
are moved from text decoders to protein decoders, where does the performance go,
and why. It is organised around the four stages such a method depends on.

**Both lists below are exhaustive, and keeping them so is the point of having
them.** Every module in this package appears in exactly one of them and in
``__all__``, and a new module is not finished until it has a line here. The lists
had drifted seventeen modules behind by 2026-08-19, when three new ones were
added and would have read as "the new surface" while ``crosscoder`` and
``differential_reliance`` -- which carry the whole D3.h line -- were still
absent. A curated list that looks complete is worse than either an exhaustive one
or none at all, so this one is exhaustive by rule rather than by habit.

Declarations -- what is measured, on what, and how it is scored. Each of these
is the *only* copy of the decision it carries, which is Appendix B rule 12:

``arms``            the matched model panel, its frozen cohorts, and the one renderer
``joint_modes``     how one joint checkpoint is rendered in each of its two
                    modes, which is that same rule applied twice to one set of
                    weights
``scoring``         the depth grid, the scored-target rule, and per-token aggregation
``statistics``      interval estimation and group-disjoint resampling
``families``        what "family-disjoint" means, which curated label sources may
                    define it, and the split that refuses rather than asserting it
``near_duplicates`` why the unit of independence in a protein corpus is not the
                    record, and the group relation a split is taken over instead
``io``              the one atomic, NaN-rejecting artefact writer

Measurements, by the stage whose transfer they probe:

``budget``               stage 1: how much context-derived information exists
``pathways``             stage 1: which sublayer pathway carries it
``circuits``             stage 1/4: induction heads, DLA, activation patching
``transcoders``          stage 2: cross-layer and per-layer transcoders in plain
                         torch, trained here so that the CLT-versus-PLT claim is
                         gated on a matched pair rather than on released weights
``replaceable``          stage 2: the block a replacement model replaces, on a
                         MoE decoder and on a dense one, so that a replacement
                         result has a text control and a dense control
``routing``              stage 2: whether a sparse-MoE replacement's residual is
                         structured by the routing decision or diffuse
``crosscoder``           stage 2: one dictionary trained jointly over two
                         checkpoints' activations at the same position, and the
                         readout separating shared from model-specific latents
``spectrum``             stage 2: how many dimensions an activation cloud
                         actually occupies where a dictionary is fitted, which
                         ``d_model`` bounds rather than measures
``basis_criteria``       stage 2: whether a fitted basis is adequate at the layer
                         a diff is reported on, as executable code rather than
                         arithmetic done by hand
``lenses``               stage 3: the logit / tuned / Jacobian lens family
``concept_lens``         stage 3: a coarsened lens readout and the null structure
                         without which coarsening on its own looks like early
                         resolution
``probes``               stage 3: decodability and concept erasure
``channels``             stage 3: bits of explanation the annotation channel holds
``relational``           stage 3: whether pair structure is per-position readable
``concept_alignment``    stage 3: whether one checkpoint's two modes place a
                         sequence and its own description in the same
                         neighbourhood, against the baselines that must be
                         beaten first
``path_patching``        stage 4: whether an effect is mediated by a named edge
``prediction_addressed`` stage 4: per-instance causal effect where no cheap screen is valid
``differential_reliance``
                         stage 4: whether ablating one crosscoder latent moves
                         one checkpoint's behaviour more than the other's
``concept_injection``    stage 4: whether a concept direction estimated in text
                         mode causally steers the same checkpoint's protein mode
``das``                  stage 4: distributed alignment search -- does a
                         low-dimensional subspace carry the antecedent; retained
                         as a closed negative-design record
``progen3``              external baseline: ProGen3-112M, loaded so that its
                         released megablocks-packed experts cannot come back
                         silently random
``fitness``              external baseline: zero-shot DMS fitness scoring, and
                         the substitution-matrix floor it has to be read against
``homology``             control: is a measured mechanism memorisation?
``profiles``             control: MODEL - LOOKUP, what a decoder adds to a lookup
                         of its own pretraining corpus
``epistasis``            control: measured pairwise epistasis against a corpus
                         coupling channel, which is the second-order bound the
                         additive channels cannot supply
``designed_referent``    control: the same phenotype estimand on de novo designs,
                         where corpus retrieval is excluded by construction
                         rather than estimated and subtracted
``kmer_background``      control: the corpus fragment statistics a homologue-free
                         referent is still exposed to
``collision_null``       control: each arm's prefix-matching census read against
                         its own collision null, so that a head count is not
                         partly a reading of the tokenizer
``induction_robustness`` control: does the induction result survive threshold and probe choice
``sequence_description`` cohort: genuine sequence-description pairs, and the
                         concept-name mask without which a cross-modal concept
                         alignment is a string match
``scaling``              cross-arm reading: scale, lineage and modality decomposition
``scale_comparison``     cross-arm reading: the paired rung-to-rung arithmetic a
                         descriptive scale ladder is read with, held once so that
                         two campaigns on the same frozen queues cannot drift
                         apart on the operations that make them comparable

**Stage 2, instrument fidelity, once had no module here and now has six, and
both halves of that belong on record.** The line the original note described is
closed and stays closed: B2
returned NO on the variance-behaviour dissociation and C4 was dropped with it
(EXP-R2-062), the production P0-2b qualification whose windowed-transcoder
implementation this package used to import now lives only under ``archive/``,
and what survives from that work is a set of limitations rather than an
instrument -- L1,
L3 and L4 in the audit catalogue -- which are measured by ``pathways`` and
``budget`` with no dictionary at all. ``scoring`` holds the four
scored-measurement declarations that were the only part of the retired package
this one ever reached. What reopened the stage is a different question rather
than a revival of that one: R2.4 and D3.h ask which directions a training stage
changed, which cannot be asked without a dictionary, so ``transcoders``,
``crosscoder``, ``replaceable``, ``routing``, ``spectrum`` and ``basis_criteria``
exist to make the instrument itself measured instead of assumed.

The panel deliberately spans the pre-dictionary toolkit, because a transfer
failure can originate at any stage: the circuit primitives in ``circuits`` need
no dictionary and so isolate substrate differences from instrument differences.
"""

from __future__ import annotations

__all__ = [
    "arms",
    "basis_criteria",
    "budget",
    "channels",
    "circuits",
    "collision_null",
    "concept_alignment",
    "concept_injection",
    "concept_lens",
    "crosscoder",
    "das",
    "designed_referent",
    "differential_reliance",
    "epistasis",
    "families",
    "fitness",
    "homology",
    "induction_robustness",
    "io",
    "joint_modes",
    "kmer_background",
    "lenses",
    "near_duplicates",
    "path_patching",
    "pathways",
    "prediction_addressed",
    "probes",
    "profiles",
    "progen3",
    "relational",
    "replaceable",
    "routing",
    "scale_comparison",
    "scaling",
    "scoring",
    "sequence_description",
    "spectrum",
    "statistics",
    "transcoders",
]
