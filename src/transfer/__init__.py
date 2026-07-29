"""Text-to-protein mechanistic-interpretability transfer measurements.

This package supports one question: when circuit-level interpretability methods
are moved from text decoders to protein decoders, where does the performance go,
and why. It is organised around the four stages such a method depends on.

Declarations -- what is measured, on what, and how it is scored. Each of these
is the *only* copy of the decision it carries, which is Appendix B rule 12:

``arms``       the matched model panel, its frozen cohorts, and the one renderer
``scoring``    the depth grid, the scored-target rule, and per-token aggregation
``statistics`` interval estimation and group-disjoint resampling
``io``         the one atomic, NaN-rejecting artefact writer

Measurements, by the stage whose transfer they probe:

``budget``               stage 1: how much context-derived information exists
``pathways``             stage 1: which sublayer pathway carries it
``circuits``             stage 1/4: induction heads, DLA, activation patching
``path_patching``        stage 4: whether an effect is mediated by a named edge
``prediction_addressed`` stage 4: per-instance causal effect where no cheap screen is valid
``lenses``               stage 3: the logit / tuned / Jacobian lens family
``probes``               stage 3: decodability and concept erasure
``channels``             stage 3: bits of explanation the annotation channel holds
``relational``           stage 3: whether pair structure is per-position readable
``homology``             control: is a measured mechanism memorisation?
``induction_robustness`` control: does the induction result survive threshold and probe choice
``scaling``              cross-arm reading: scale, lineage and modality decomposition

**Stage 2, instrument fidelity, has no module here, and that is a result rather
than an omission.** The dictionary line it would have served is closed: B2
returned NO on the variance-behaviour dissociation and C4 was dropped with it
(EXP-R2-062), and the production P0-2b qualification whose windowed-transcoder
implementation this package used to import now lives only under ``archive/``.
What survives from that work is a set of limitations, not an instrument -- L1,
L3 and L4 in the audit catalogue -- and those are measured by ``pathways`` and
``budget`` with no dictionary at all. ``scoring`` holds the four
scored-measurement declarations that were the only part of the retired package
this one ever reached.

The panel deliberately spans the pre-dictionary toolkit, because a transfer
failure can originate at any stage: the circuit primitives in ``circuits`` need
no dictionary and so isolate substrate differences from instrument differences.
"""

from __future__ import annotations

__all__ = [
    "arms",
    "budget",
    "channels",
    "circuits",
    "homology",
    "induction_robustness",
    "io",
    "lenses",
    "path_patching",
    "pathways",
    "prediction_addressed",
    "probes",
    "relational",
    "scaling",
    "scoring",
    "statistics",
]
