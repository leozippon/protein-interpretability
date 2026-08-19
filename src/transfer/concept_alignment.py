"""Does one joint checkpoint place a protein where its own text mode places the description of it?

D3.g's estimand, its linear ladder, and every baseline the pre-registered
admission bar (§8 item 4) requires to be run *before* any non-linear adapter.
This module owns the measurement; ``scripts/transfer/35_concept_alignment.py``
owns the pre-registration, the cohort and the artefact.

The object
==========

One frozen checkpoint qualified in **both** modes (EXP-R2-152: the ProLLaMA
lineage is the only one there is) is run twice over the same cohort record --
once in protein mode over the sequence, once in text mode over the curated
description -- and each side is reduced to **one vector per record**. A map
fitted on the ``fit`` split alone is then asked whether it carries the protein
vector to the neighbourhood of its own description, and whether a concept
direction declared on the *text* side reads the mapped protein vector correctly.

**Sequence level, and that is a scoping decision rather than a convenience.**
The gate names raw activation equality and unpaired token positions as
non-estimands, and L31 is why the second half of that is not negotiable on this
family: ProLLaMA writes protein into the unmodified LLaMA-2 SentencePiece
vocabulary, which merges residue runs at about 1.536 residues per token, so a
"residue *i*" of the sequence has no token of its own on roughly half the cohort
and the instances that survive are the BPE-stable ones rather than a random
subsample. There is therefore no position correspondence between a residue and a
word, and none is attempted here. Pooling over the mode's own content positions
is the aggregation, and :data:`POOLINGS` is the declared set of ways to do it.

**Which positions are content** comes from
:meth:`src.transfer.replaceable.JointReplaceable.content_mask`, which reads the
declared rendering's own scored span -- in protein mode the token run whose
spellings are exactly the sequence, so the ``Seq=<`` prefix, the closing
delimiter, any instruction context and the beginning-of-sequence token stay out
of the pooled vector; in text mode every non-padding, non-special position.
``src.transfer.scoring.target_rule`` / ``sequence_target_mask`` are the panel's
declaration of the same question and are deliberately *not* used here: their
``between_boundaries`` rule locates a conditioning prompt by its delimiters'
**token ids**, and :mod:`src.transfer.joint_modes` declares ProLLaMA's
delimiters not to be tokens of its vocabulary at all (``SPELLED_SEQUENCE_RUN``),
so the panel rule cannot express this family's content span. The depth grid
*is* taken from ``scoring`` -- :func:`~src.transfer.scoring.analysis_layer` and
:func:`~src.transfer.scoring.analysis_layers` -- because two stages of one
campaign once disagreed about what relative depth 0.25 meant.

The statistic, and why it is one number
=======================================

EXP-R2-213 fixes it: **top-1 retrieval accuracy in excess of chance,
description to sequence, on a common gallery size.** The gallery is common by
construction (:func:`common_gallery`) because a per-query field makes the chance
level an average over fields of different sizes -- after the near-duplicate
exclusion, a record with four copies in the cohort ranks against a smaller field
than a singleton, so its top-1 accuracy is not the same statistic. With one
common size the chance level is the single stated number ``1/gallery_size``.
MRR, the wider recall cut-offs and the whole concept axis are reported beside it
and decide nothing; in particular a concept AUC excess is not commensurable with
a retrieval excess and is never compared against one.

The ladder, in the order the gate fixes
=======================================

Nothing above a rung may be read until the rung below it has been run and
reported, and :func:`assert_ladder_reported` is what enforces it:

``mean``
    a pure mean shift fitted on the ``fit`` split. It can never be the *deciding*
    rung, because it does not read the pairing at all -- so the shuffled-fit
    baseline is the truth under another name for it, and a criterion cannot be
    passed by being vacuous.
``procrustes``
    the orthogonal map, by SVD.
``affine``
    ridge, with the penalty chosen inside the fit fold on group-disjoint folds
    and never on the evaluation split.

``nearest_neighbour`` -- cosine retrieval between the two sides' pooled vectors
with **no fit at all**, well defined because both modes of one checkpoint live in
one activation space -- sits below all of them and is reported. It is not in the
frozen decisive set, and it is not promoted into it: widening a pre-registered
criterion is as much a change to it as softening one.

**No non-linear adapter is implemented here, deliberately.** The gate locks a
bottleneck Adapter MLP behind the linear ladder passing, so building one before
that would be building the thing the pre-registration exists to defer.

What has to be beaten
=====================

:data:`A35_1_BASELINES` is the frozen decisive set, and each of its members must
satisfy **both** conditions on the group-disjoint ``eval`` split **and** on the
``family_holdout`` split: the paired group-bootstrap 95% interval of the
difference excludes zero, **and** the excess over chance is at least the declared
multiple of the baseline's. The two are separate on purpose -- significance alone
is a detection criterion and does not license a comparative claim.

``shuffled_pair``
    the pairing permuted at evaluation, many draws, reported as a distribution.
    Its gallery structure is identical to the truth's by construction, because
    the query side is what is permuted and the gallery -- and therefore the
    chance level -- is untouched.
``shuffled_fit``
    the same permutation applied to the **fit** split, the map refitted on it,
    and the true evaluation pairing scored. This is the one that prices the
    map's own capacity to manufacture a correspondence, which a permutation at
    evaluation time cannot see.
``rank_matched``
    a null of matched difficulty rather than a free one, in the spirit of
    :func:`src.transfer.concept_lens.rank_matched_partitions`: hold fixed the
    profile that makes any answer look good -- here the nuisance variable a pair
    shares, sequence length -- and randomise only the identity. A permutation
    restricted to blocks of the length-rank order leaves a length-driven
    retrieval intact and destroys everything else, and the achieved match is
    measured and reported rather than asserted (:func:`pairing_match_quality`).
``composition`` / ``kmer``
    the protein side replaced by 20-dimensional amino-acid composition, and
    separately by 3-mer frequencies, with the identical subspace, map and
    metric. This prices how much of any apparent alignment is recoverable from
    surface composition with no model in the loop. It is built to be strong
    rather than nominal: D3.b died to a conditioning leak and F12 died to a free
    hydropathy baseline, and this is the same shape of objection.
``description_only``
    the retrieval attainable with **no sequence-derived pairing signal**: the
    stronger of a query-independent ranking of the gallery by its own typicality
    and a ridge from the description representation to log length. It is the same
    task on the same field, which is what makes the effect-size condition
    well-posed for it -- a concept-classification accuracy could not be compared
    against a retrieval excess under one criterion. It prices text-side
    self-information plus gallery structure: a description that says what kind of
    protein it is has already said something about how long it is.

:data:`A35_1B_BASELINE` -- ``bridge_specific`` -- is deliberately **not** in that
set, and amendment 1 is why. It is the **same retrieval task** restricted to the
span of the concepts *defined on both sides*: carried by the curated annotation
and named in the description, which is what the cohort's ``masked_terms``
records. Restricted by the orthogonal projector onto that span rather than by
re-coordinating in the concept directions, which are not orthonormal. The
amendment moved it out of the decisive set on a measured attainability failure
and on a conceptual one: at the limit the original bar was backwards, because if
a genuine cross-modal alignment *is* carried by the concepts declared on both
sides then ``bridge ~ full`` is what the hypothesis predicts. A restriction that
loses nothing is informative in its own right, so the reported quantity is the
ratio ``bridge / full`` on the primary statistic and clause (ii) becomes decisive
against it only where the raw-description arm demonstrates, at the run's own
settings, that the declared margin against it is reachable.

**Attainability comes before control, and the gate is one baseline.** The
raw-description arm -- concept name present -- is read first, and if it does not
clear A35-1's margin over :data:`A35_0_GATE_BASELINE` the whole ladder is void as
a specification defect and the masked arm is not read at all. A bar the positive
control cannot reach is a property of the specification and not a result about
proteins; two of D3.h's criteria were voided for skipping exactly this ordering,
one of them unreachable at any sample size. The gate is ``shuffled_pair`` alone
because that is what EXP-R2-213 names, in the singular, and the distinction it
carries is load-bearing: a raw arm that clears ``shuffled_pair`` and loses to a
3-mer surrogate is a **measured** result on the surface-statistics branch, which
is a statement about the method, while a raw arm that cannot clear
``shuffled_pair`` is a statement about the instrument. Gating A35-0 on the whole
decisive set collapses the two and files the first under the second.

The **pre-adaptation reference** is a separate invocation of the same pipeline
on ``Llama-2-7b-hf`` and is representational only. Its protein mode is
behaviourally unmeasurable on this lineage -- context information +0.084
nats/token and a reversal cost of **-0.0013** nats/residue against the adapted
stage's +0.1442 (EXP-R2-152, re-measured on a second cohort draw at EXP-R2-174)
-- so :data:`PROTEIN_MODE_BEHAVIOURAL_STATUS` marks it, :func:`admission_verdict`
refuses to admit it and returns ``REFERENCE_ONLY``, and
:func:`assert_behavioural_read_permitted` raises rather than hand a causal stage
an intervention target that cannot be read back.

What this module does not do
============================

It states no behavioural quantity of its own. A pooled activation and a map
between two clouds of them are representational, exactly as
``25_model_diffing_baselines.py``'s residuals are; the graded intervention the
gate also requires belongs to the causal stage, which imports
:func:`concept_vector` from here so that the direction it steers along and the
direction measured here are one object.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats

from src.transfer.arms import AA20
from src.transfer.kmer_background import ALPHABET as KMER_ALPHABET
from src.transfer.kmer_background import kmer_index
from src.transfer.replaceable import JointReplaceable
from src.transfer.sequence_description import RECORD_FIELDS, SPLIT_NAMES
from src.transfer.statistics import (
    MINIMUM_BOOTSTRAP_UNITS,
    make_group_splits,
    mean_interval,
)

# ``src.transfer.statistics.paired_group_bootstrap`` is the only resampler this
# stage uses and it is called from the stage script, not wrapped here. A wrapper
# in this package would be a second name in the repository's resampler inventory
# (``tests/test_transfer_audit_invariants.py``), which exists precisely so that a
# resampler cannot arrive without a decision about the eight-unit floor -- and a
# second entry point would have been exactly the thing the Single-Source
# Principle forbids, for a function that resamples nothing of its own.

__all__ = [
    "A35_0_GATE_BASELINE",
    "A35_0_GATE_NOTE",
    "A35_1B_BASELINE",
    "A35_1B_NOTE",
    "A35_1_BASELINES",
    "ALIGNMENT_METHODS",
    "AMENDMENT_1_NOTE",
    "AlignmentMap",
    "COHORT_FIELDS",
    "CONCEPT_VECTOR_METHODS",
    "Cohort",
    "ConceptVector",
    "DECIDING_DESCRIPTION_VARIANT",
    "DEFAULT_RIDGE_GRID",
    "NONLINEAR_ADAPTER_NOTE",
    "POOLINGS",
    "PRE_REGISTRATION",
    "PRE_REGISTRATION_AMENDMENTS",
    "PROTEIN_MODE_BEHAVIOURAL_STATUS",
    "REPRESENTATION_SITE",
    "REPRESENTATION_SITE_NOTE",
    "RETRIEVAL_KS",
    "SPLITS",
    "Subspace",
    "admission_verdict",
    "apply_alignment",
    "apply_subspace",
    "assert_behavioural_read_permitted",
    "assert_ladder_reported",
    "assert_per_layer_only",
    "attainability_gate",
    "baseline_row",
    "bridge_concepts",
    "composition_features",
    "concept_auc",
    "concept_labels",
    "concept_vector",
    "PRIMARY_STATISTIC",
    "declared_concepts",
    "fit_alignment",
    "fit_subspace",
    "kmer_features",
    "auc_metric",
    "common_gallery",
    "load_cohort",
    "masked_term_vocabulary",
    "mean_metric",
    "mode_representations",
    "metrics_from_ranks",
    "null_distribution",
    "pairing_match_quality",
    "protein_mode_behavioural_status",
    "rank_matched_pairing",
    "ranks_from_scores",
    "retrieval_metrics",
    "retrieval_ranks",
    "shuffled_pairing",
    "split_mask",
    "top1_indicators",
]


# --------------------------------------------------------------- declarations

#: The tensor a pooled representation is taken from, in
#: :meth:`src.transfer.replaceable.JointReplaceable.block_intercept`'s own terms:
#: the *input* of the per-layer feed-forward. One declaration rather than a flag,
#: because a stage that could be pointed at either tensor would produce two
#: populations under one name.
REPRESENTATION_SITE = "block_input"

REPRESENTATION_SITE_NOTE = (
    "the input of the per-layer feed-forward, which on this serial block layout "
    "is the residual stream after the attention write, normalised by "
    "pre_feed_forward_norm. It is the same tensor 25_model_diffing_baselines.py "
    "fits its affine and orthogonal maps between and the same one "
    "17_train_transcoder.py calls 'block_input', which is what makes this "
    "stage's linear ladder readable beside theirs. It is NOT the feed-forward "
    "output that 30_activation_spectrum.py measures its spectra at"
)

#: How the content positions of one record become one vector. ``mean_content`` is
#: the declared default of this programme and the one the module docstring
#: argues for; ``last_content`` is the causal-model alternative -- the position
#: that has attended to the whole record -- and is offered so the choice is a
#: pre-registered decision rather than an unstated one.
POOLINGS = ("mean_content", "last_content")

#: The linear ladder, in the order the gate fixes.
ALIGNMENT_METHODS = ("mean", "procrustes", "affine")

#: One method, and the parameter exists so that a second one cannot be added
#: silently: :func:`concept_vector` refuses anything else by name.
CONCEPT_VECTOR_METHODS = ("diff_means",)

RETRIEVAL_KS = (1, 5, 10)

#: The one statistic the pre-registered admission criteria are evaluated on
#: (EXP-R2-213). Declared here so the stage cannot read its verdict off a
#: different number than the one the pre-registration names.
PRIMARY_STATISTIC = "top1_excess"

PRIMARY_STATISTIC_NOTE = (
    "top-1 retrieval accuracy in excess of chance, description -> sequence, on a "
    "COMMON gallery size: every query ranks its own target against the same "
    "number of admissible distractors, so the chance level is the single stated "
    "number 1/gallery_size rather than an average over galleries of different "
    "sizes. MRR and the other cut-offs are reported beside it and decide nothing"
)

#: The pre-registration every threshold in this module is quoted from, and the
#: amendments to it this module implements. Declared here and echoed into the
#: artefact -- 36_concept_injection.py's ``PRE_REGISTRATION_AMENDMENTS`` is the
#: same declaration for the causal stage -- so that a reader never has to infer
#: which text a number was produced under, and so that a recorded amendment the
#: executing code does not implement is a detectable gap rather than a silent one
#: (``tests/test_concept_alignment.py`` checks the declaration against the frozen
#: constants it implies).
PRE_REGISTRATION = "EXP-R2-213"
PRE_REGISTRATION_AMENDMENTS: tuple[str, ...] = ("amendment 1",)

AMENDMENT_1_NOTE = (
    "EXP-R2-213 amendment 1, decided before any cohort existed: A35-1's decisive "
    "set is SIX baselines read on description -> sequence top-1 accuracy in excess "
    "of chance, and bridge_specific is not one of them. It becomes A35-1b, a "
    "reported restriction diagnostic whose interpretable quantity is the "
    "bridge/full ratio, gating only where the raw-description arm demonstrates at "
    "the run's own settings that the declared margin against it is reachable. The "
    "concept-axis AUC and the training-free nearest_neighbour rung are explicitly "
    "non-decisive"
)

#: The baselines EXP-R2-213 requires, each under BOTH of its conditions: the
#: paired group-bootstrap 95% interval of the difference excludes zero, and the
#: excess over chance is at least ``excess_ratio`` times the baseline's. Frozen
#: as a tuple so a missing baseline is a refusal rather than a shorter table.
#: **Six**, per amendment 1; ``bridge_specific`` left this set and is
#: :data:`A35_1B_BASELINE`.
A35_1_BASELINES = (
    "shuffled_pair",
    "shuffled_fit",
    "rank_matched",
    "composition",
    "kmer",
    "description_only",
)

#: A35-1b's baseline: reported always, decisive only where the raw arm shows the
#: margin against it is reachable. It is not a member of :data:`A35_1_BASELINES`
#: and a run that put it back there would be running the pre-amendment criterion.
A35_1B_BASELINE = "bridge_specific"

A35_1B_NOTE = (
    "A35-1b (EXP-R2-213 amendment 1): the concept restriction, reported as the "
    "ratio bridge/full on the primary statistic on BOTH splits, and gating under "
    "clause (ii) only if the raw-description arm demonstrates at the run's own "
    "settings that the declared margin against it is attainable. Where it is not, "
    "clause (ii) is declared non-applicable for this baseline with the measured "
    "reason and the achieved ratio recorded. The word carrying the load in §8's "
    "'beat every applicable baseline' is APPLICABLE, and the amendment makes "
    "applicability a matter of measured attainability rather than assumption: a "
    "baseline the positive control cannot be separated from is not an applicable "
    "baseline, and declaring it one is how a criterion becomes decorative"
)

#: The ONE baseline A35-0's attainability gate is taken on. EXP-R2-213 names it in
#: the singular -- "if the identical ladder on raw descriptions does not clear
#: A35-1's margin over ``shuffled_pair``, the ladder is void as a specification
#: defect" -- and its own branch table gives the two outcomes opposite subjects.
A35_0_GATE_BASELINE = "shuffled_pair"

A35_0_GATE_NOTE = (
    "A35-0 is gated on shuffled_pair ALONE, under A35-1's two clauses, because "
    "that is the baseline EXP-R2-213 names and because the two failures are "
    "statements about different things. A raw arm that cannot clear shuffled_pair "
    "has not aligned anything at all and the ladder is VOID as a specification "
    "defect -- about the instrument, not the modality. A raw arm that clears "
    "shuffled_pair and loses to a composition or 3-mer surrogate is the frozen "
    "SURFACE-STATISTICS branch -- about the method -- and it is a measured "
    "negative read on the masked arm, not a void. Gating this decision on the "
    "whole decisive set reports the second as the first. The full raw-arm baseline "
    "table is reported either way and is what the surface-statistics branch is "
    "read off"
)

#: The cohort's three sides and its record schema, imported from the module that
#: writes them rather than restated here. ``src.transfer.sequence_description``
#: says it in its own words -- "the schema is declared once, here, and both the
#: writer and the reader check against this tuple rather than against each
#: other" -- and a second copy is exactly the drift Appendix B rule 12 exists to
#: stop: a reader whose field list had fallen behind the writer's would accept a
#: cohort missing ``dup_group`` and silently turn the near-duplicate exclusion
#: into a no-op.
SPLITS = SPLIT_NAMES
COHORT_FIELDS = RECORD_FIELDS

#: The description variant the verdict is read on. The raw description is
#: reported beside it, and the pair IS the description-leakage check: a result
#: that needs the concept term present in the text is a result about the term.
DECIDING_DESCRIPTION_VARIANT = "masked"

DEFAULT_RIDGE_GRID = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5)

NONLINEAR_ADAPTER_NOTE = (
    "no non-linear adapter is implemented in this module. §8 item 4 admits a "
    "small bottleneck Adapter MLP only after the linear ladder has been run and "
    "only if non-linearity adds family-disjoint value, so building one before "
    "the ladder reports would defeat the pre-registration it is gated by"
)

#: Whether a checkpoint's protein mode may carry a behavioural claim, keyed by
#: the checkpoint directory's name. Declared with the measurement that decides
#: it, because "unmeasurable" is a reading of a number and not a label: the
#: pre-adaptation checkpoint pays -0.0013 nats/residue to have a protein read
#: backwards, so it does not read sequence directionally at all and nothing
#: behavioural can be attributed to its protein mode.
PROTEIN_MODE_BEHAVIOURAL_STATUS: Mapping[str, Mapping[str, Any]] = {
    "ProLLaMA_Stage_1": {
        "measurable": True,
        "context_information_nats_per_token": 0.5505,
        "reversal_cost_nats_per_residue": 0.1442,
        "evidence": "EXP-R2-152, re-measured on a second cohort draw at EXP-R2-174",
    },
    "ProLLaMA": {
        "measurable": True,
        "context_information_nats_per_token": 0.52,
        "reversal_cost_nats_per_residue": None,
        "evidence": "EXP-R2-152",
    },
    "Llama-2-7b-hf": {
        "measurable": False,
        "context_information_nats_per_token": 0.084,
        "reversal_cost_nats_per_residue": -0.0013,
        "evidence": "EXP-R2-152, re-measured on a second cohort draw at EXP-R2-174",
        "reason": (
            "the pre-adaptation checkpoint's protein mode is behaviourally "
            "unmeasurable on this lineage: it pays -0.0013 nats/residue to have a "
            "sequence reversed, against +0.1442 for the adapted stage, so it does "
            "not read protein directionally. It may serve as a REPRESENTATIONAL "
            "pre-adaptation reference -- the activations exist and are comparable "
            "-- and it may never carry a behavioural claim or seed a causal stage"
        ),
    },
}


# ------------------------------------------------------------------- guards


def _finite_matrix(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _readonly(array: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(array, dtype=np.float64)
    out.flags.writeable = False
    return out


def _digest(*arrays: np.ndarray) -> str:
    """Identity of the data a map was fitted on, so a refit cannot hide."""

    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _unit_rows(values: np.ndarray, name: str) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1)
    if not np.all(norms > 0.0):
        raise ValueError(
            f"{name} carries a zero-norm row, which has no direction and therefore "
            "no cosine similarity to anything"
        )
    return values / norms[:, None]


# -------------------------------------------------------------------- cohort


@dataclass(frozen=True)
class Cohort:
    """The frozen sequence-description cohort, as it reaches this stage.

    Nothing here derives a record, a split or a group. All three are decided by
    ``34_sequence_description_cohort.py`` and are read: a stage that could
    re-split would be a second declaration of what "held out" means, and L30 is
    the record of what a second definition of a held-out set costs on a protein
    corpus.
    """

    path: Path
    records: tuple[Mapping[str, Any], ...]
    manifest: Mapping[str, Any]

    def __len__(self) -> int:
        return len(self.records)

    def counts(self) -> dict[str, int]:
        return {
            split: int(sum(1 for record in self.records if record["split"] == split))
            for split in SPLITS
        }

    def facts(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "n_records": len(self.records),
            "split_counts": self.counts(),
            "n_dup_groups": len({record["dup_group"] for record in self.records}),
            "n_family_groups": len({record["family_group"] for record in self.records}),
            "manifest_keys": sorted(self.manifest),
        }


def load_cohort(path: Path) -> Cohort:
    """Read ``records.jsonl`` and its sibling ``cohort.json``, or refuse.

    ``path`` may name either the directory the cohort stage wrote or the
    ``records.jsonl`` inside it, and which one it is is decided by the suffix
    rather than by whether the directory happens to exist -- a rule that tested
    ``is_dir()`` reported a missing *directory* as a missing *file* and named a
    path with no ``records.jsonl`` in it, which is the one thing this message is
    for. A missing file raises with the path it looked for: this stage has no
    fallback cohort, and synthesising one in the real path would produce an
    artefact indistinguishable from a measured one.
    """

    location = Path(path)
    records_path = location if location.suffix == ".jsonl" else location / "records.jsonl"
    manifest_path = records_path.parent / "cohort.json"
    if not records_path.exists():
        raise FileNotFoundError(
            f"{records_path} does not exist. It is written by "
            "scripts/transfer/34_sequence_description_cohort.py and is not "
            "synthesised here; run that stage or point --cohort at its output"
        )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} does not exist. The manifest carries the split "
            "certificates, the leakage curves and the per-concept counts a result "
            "on this cohort is only readable beside, so a records file without it "
            "is not a cohort"
        )
    records: list[Mapping[str, Any]] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            record = json.loads(text)
            missing = [name for name in COHORT_FIELDS if name not in record]
            if missing:
                raise ValueError(
                    f"{records_path}:{number} is missing {missing}; the cohort schema "
                    f"is {list(COHORT_FIELDS)} and this stage codes against it rather "
                    "than against whatever a record happens to carry"
                )
            if record["split"] not in SPLITS:
                raise ValueError(
                    f"{records_path}:{number} declares split {record['split']!r}; the "
                    f"declared splits are {list(SPLITS)}"
                )
            records.append(record)
    if not records:
        raise ValueError(f"{records_path} carries no records")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cohort = Cohort(path=records_path, records=tuple(records), manifest=manifest)
    _assert_group_disjoint(cohort)
    return cohort


def _assert_group_disjoint(cohort: Cohort) -> None:
    """Re-check the property the cohort stage certified, on the file that arrived.

    Appendix B rule 32: a contract stated in prose gets read as evidence in
    exactly the place the property should have been checked. The cohort's
    manifest asserts group-disjoint splits; this is that assertion evaluated.
    """

    for group_field in ("dup_group", "family_group"):
        owners: dict[Any, set[str]] = {}
        for record in cohort.records:
            owners.setdefault(record[group_field], set()).add(record["split"])
        straddling = sorted(
            str(group) for group, splits in owners.items() if len(splits) > 1
        )
        if group_field == "dup_group" and straddling:
            raise ValueError(
                f"{len(straddling)} dup_group values straddle two splits "
                f"(first: {straddling[:3]}). A near-duplicate group split across "
                "fit and eval is L30's defect, and it acts through the true "
                "pairing alone, so it widens exactly the gap this stage reads"
            )
        if group_field == "family_group":
            holdout = {
                record["family_group"]
                for record in cohort.records
                if record["split"] == "family_holdout"
            }
            seen = {
                record["family_group"]
                for record in cohort.records
                if record["split"] != "family_holdout"
            }
            shared = sorted(str(group) for group in holdout & seen)
            if shared:
                raise ValueError(
                    f"{len(shared)} family_group values appear both in "
                    f"family_holdout and outside it (first: {shared[:3]}); the "
                    "unseen-family reproduction the gate requires is then not on "
                    "an unseen family"
                )


def split_mask(cohort: Cohort, split: str) -> np.ndarray:
    """Boolean row selector for one declared split."""

    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; declared: {list(SPLITS)}")
    mask = np.array([record["split"] == split for record in cohort.records], dtype=bool)
    if not mask.any():
        raise ValueError(f"the cohort carries no records in the {split!r} split")
    return mask


# ------------------------------------------------------------ representations


def _assert_handle(
    checkpoint: JointReplaceable,
    *,
    rendering: str,
    mode: str,
    device: str,
    dtype: str,
) -> dict[str, Any]:
    """Verify the loaded handle against what the run declared it to be.

    ``checkpoint`` is the *loaded* joint checkpoint rather than a path, because
    ``scripts/transfer/21_joint_mode_qualification.py`` owns the one loader this
    programme has for a joint checkpoint (Appendix B rule 12) and a library
    module carrying a second one would be the duplicate that rule exists to
    stop. The declared rendering, mode, device and dtype are therefore checked
    against the object rather than used to build it -- read back, not echoed.
    """

    if not isinstance(checkpoint, JointReplaceable):
        raise TypeError(
            "mode_representations takes the loaded joint checkpoint handle "
            "(src.transfer.replaceable.JointReplaceable), not a path: the joint "
            "checkpoint loader is owned by "
            "scripts/transfer/21_joint_mode_qualification.py and is not "
            "reimplemented here"
        )
    if checkpoint.declaration.name != rendering:
        raise ValueError(
            f"this handle was built for the {checkpoint.declaration.name!r} "
            f"rendering and the run declared {rendering!r}"
        )
    if checkpoint.mode != mode:
        raise ValueError(
            f"this handle is the {checkpoint.mode!r} mode and the run declared "
            f"{mode!r}; one handle is one mode"
        )
    requested = torch.device(device)
    resident = checkpoint.device
    if requested.type != resident.type or (
        requested.index is not None and requested.index != resident.index
    ):
        raise ValueError(f"the model is resident on {resident} and the run declared {device}")
    observed = sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in checkpoint.model.parameters()
            if parameter.is_floating_point()
        }
    )
    if observed != [dtype]:
        raise ValueError(f"the run declared dtype {dtype} and the model carries {observed}")
    return {
        "rendering": rendering,
        "mode": mode,
        "device": str(resident),
        "dtype_observed": observed,
        "checkpoint": str(checkpoint.checkpoint),
        "n_layers": int(checkpoint.n_layers),
        "d_model": int(checkpoint.width),
        "rendering_note": checkpoint.rendering_note,
        "content_positions": checkpoint.scoring_note,
        "site": REPRESENTATION_SITE,
        "site_note": REPRESENTATION_SITE_NOTE,
    }


def _pool(
    activation: torch.Tensor, content: torch.Tensor, *, pooling: str
) -> torch.Tensor:
    """``(batch, tokens, d)`` and a content mask into ``(batch, d)``, in float32."""

    if pooling not in POOLINGS:
        raise ValueError(f"unknown pooling {pooling!r}; declared: {list(POOLINGS)}")
    values = activation.float()
    counts = content.sum(dim=1)
    if int((counts < 1).sum()) > 0:
        raise ValueError(
            "a record contributed no content positions, so it has no pooled "
            "representation; nothing downstream can be computed from it and a "
            "zero vector would look like one"
        )
    if pooling == "mean_content":
        weighted = values * content.unsqueeze(-1).to(values.dtype)
        return weighted.sum(dim=1) / counts.unsqueeze(-1).to(values.dtype)
    positions = torch.arange(content.shape[1], device=content.device)
    last = (positions.unsqueeze(0) * content.long()).argmax(dim=1)
    return values[torch.arange(values.shape[0], device=values.device), last]


@torch.no_grad()
def mode_representations(
    checkpoint: JointReplaceable,
    rendering: str,
    mode: str,
    records: Sequence[str],
    layers: Sequence[int],
    pooling: str,
    device: str,
    batch_size: int,
    dtype: str,
) -> dict[int, np.ndarray]:
    """One pooled vector per record, per declared layer, for one mode.

    ``records`` are the mode's own inputs: sequences in protein mode, strings in
    text mode. The rendering, the scored span and the refusal of a tokenizer that
    cannot carry the declared alphabet all come from
    :mod:`src.transfer.joint_modes` through the handle, so nothing here decides a
    format.

    Pooling happens inside the interceptor, one batch at a time, so the run never
    holds ``(layers, batch, tokens, d_model)``: at 32 layers, 4 records and 1024
    tokens that tensor alone is 2.1 GB, and this stage needs none of it.
    """

    facts = _assert_handle(
        checkpoint, rendering=rendering, mode=mode, device=device, dtype=dtype
    )
    if pooling not in POOLINGS:
        raise ValueError(f"unknown pooling {pooling!r}; declared: {list(POOLINGS)}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    wanted = tuple(sorted({int(layer) for layer in layers}))
    if not wanted:
        raise ValueError("at least one layer must be declared")
    outside = [layer for layer in wanted if not 0 <= layer < facts["n_layers"]]
    if outside:
        raise ValueError(
            f"layers {outside} are outside this backbone's 0..{facts['n_layers'] - 1}"
        )
    inputs = [str(record) for record in records]
    if not inputs:
        raise ValueError("an empty record set has nothing to represent")

    width = facts["d_model"]
    out = {layer: np.zeros((len(inputs), width), dtype=np.float64) for layer in wanted}
    for start in range(0, len(inputs), batch_size):
        chunk = inputs[start : start + batch_size]
        batch = checkpoint.batch(checkpoint.render(list(chunk)))
        checkpoint.forget_rendered()
        content = checkpoint.content_mask(batch)
        seen: set[int] = set()

        def tap(
            layer: int, block_input: torch.Tensor, block_output: torch.Tensor
        ) -> None:
            if layer in wanted:
                seen.add(layer)
                pooled = _pool(block_input, content, pooling=pooling)
                out[layer][start : start + len(chunk)] = (
                    pooled.detach().cpu().numpy().astype(np.float64)
                )
            return None

        with checkpoint.block_intercept(tap):
            checkpoint.run(batch)
        if seen != set(wanted):
            raise RuntimeError(
                f"the forward pass reached layers {sorted(seen)} of the {list(wanted)} "
                "requested, so a declared layer produced no representation"
            )
    for layer, array in out.items():
        if not np.isfinite(array).all():
            raise RuntimeError(
                f"layer {layer} produced a non-finite pooled representation; a "
                "bfloat16 overflow reaches float64 as an infinity and every "
                "downstream interval would be nan"
            )
    return out


# ------------------------------------------------------------------ subspace


@dataclass(frozen=True)
class Subspace:
    """The shared basis both modes are compared in, fitted on the fit split alone.

    One basis rather than two. Both modes of one checkpoint write into one
    activation space, so a per-side basis would make the identity map meaningless
    and would let a "difference" between the two sides be a difference between
    two arbitrary rotations.

    The components come from the fit split's *centred* second moment, which is
    what makes them principal directions, and the projection deliberately keeps
    the origin: a projection that also subtracted the mean would make the
    mean-shift rung of the ladder the identity by construction and silently
    delete a rung the gate requires.
    """

    components: np.ndarray
    mean: np.ndarray
    explained_variance_ratio: tuple[float, ...]
    n_fit_rows: int

    @property
    def n_components(self) -> int:
        return int(self.components.shape[1])

    def record(self) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "d_model": int(self.components.shape[0]),
            "n_fit_rows": int(self.n_fit_rows),
            "explained_variance_ratio_sum": float(sum(self.explained_variance_ratio)),
            "explained_variance_ratio_head": [
                float(value) for value in self.explained_variance_ratio[:8]
            ],
            "note": (
                "one basis for both modes, fitted on the fit split only, from the "
                "centred second moment; the projection keeps the origin so the "
                "mean rung of the ladder is not the identity by construction"
            ),
        }


def fit_subspace(rows: np.ndarray, n_components: int) -> Subspace:
    """Principal directions of the fit split's pooled representations."""

    data = _finite_matrix(rows, "rows")
    limit = min(data.shape)
    if n_components < 1 or n_components > limit:
        raise ValueError(
            f"n_components must lie in 1..{limit} for a {data.shape[0]}x{data.shape[1]} "
            "fit block; a basis wider than the data it is fitted on carries "
            "directions no record ever moved along"
        )
    mean = data.mean(axis=0)
    _, singular, right = np.linalg.svd(data - mean, full_matrices=False)
    variance = singular**2
    total = float(variance.sum())
    if total <= 0.0:
        raise ValueError("the fit block has zero variance, so it spans no subspace")
    return Subspace(
        components=_readonly(right[:n_components].T),
        mean=_readonly(mean.reshape(1, -1)),
        explained_variance_ratio=tuple(
            float(value) for value in (variance[:n_components] / total)
        ),
        n_fit_rows=int(data.shape[0]),
    )


def apply_subspace(subspace: Subspace, rows: np.ndarray) -> np.ndarray:
    """Project into the fitted basis, keeping the origin."""

    data = _finite_matrix(rows, "rows")
    if data.shape[1] != subspace.components.shape[0]:
        raise ValueError(
            f"the basis is over {subspace.components.shape[0]} dimensions and these "
            f"rows carry {data.shape[1]}"
        )
    return data @ subspace.components


# ----------------------------------------------------------------- alignment


@dataclass(frozen=True)
class AlignmentMap:
    """A fitted linear map from one mode's coordinates to the other's.

    ``fit_digest`` is the identity of the arrays the map was fitted on, and it is
    the field that makes "fitted on the fit split, applied to eval" checkable
    rather than asserted. A caller that refits on the evaluation block gets a
    different digest, and the artefact carries both.
    """

    method: str
    weight: np.ndarray
    bias: np.ndarray
    n_fit: int
    fit_digest: str
    penalty: float | None = None
    penalty_selection: Mapping[str, Any] | None = None

    def record(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_fit": int(self.n_fit),
            "fit_digest": self.fit_digest,
            "d_source": int(self.weight.shape[0]),
            "d_target": int(self.weight.shape[1]),
            "free_parameters": int(self.free_parameters()),
            "penalty": None if self.penalty is None else float(self.penalty),
            "penalty_selection": (
                None if self.penalty_selection is None else dict(self.penalty_selection)
            ),
        }

    def free_parameters(self) -> int:
        """The capacity the gate requires to be reported beside every result."""

        d_source, d_target = self.weight.shape
        if self.method == "mean":
            return d_target
        if self.method == "procrustes":
            return d_target * (d_target - 1) // 2 + d_target
        return d_source * d_target + d_target


def _ridge_weight(source: np.ndarray, target: np.ndarray, penalty: float) -> np.ndarray:
    gram = source.T @ source
    gram = gram + penalty * np.eye(gram.shape[0])
    return np.linalg.solve(gram, source.T @ target)


def _held_out_residual(
    source: np.ndarray, target: np.ndarray, weight: np.ndarray
) -> float:
    """Residual as a fraction of the target's own centred energy.

    The same unit ``25_model_diffing_baselines.py`` reads its 0.400 and 0.684 in,
    so a penalty chosen here and a residual reported there mean one thing.
    """

    residual = target - source @ weight
    denominator = float(((target - target.mean(axis=0)) ** 2).sum())
    if denominator <= 0.0:
        raise ValueError("the held-out target block has zero variance to explain")
    return float((residual**2).sum() / denominator)


def _select_penalty(
    source: np.ndarray,
    target: np.ndarray,
    groups: Sequence[Any],
    *,
    grid: Sequence[float],
    seed: int,
    n_splits: int,
) -> tuple[float, dict[str, Any]]:
    """Choose the ridge penalty inside the fit fold, on group-disjoint folds.

    Group-disjoint and not row-disjoint: a penalty selected across a near-
    duplicate boundary is selected on a fold whose held-out half the training
    half already contains, which is L30 one level down from the split itself.
    """

    unique = len(set(map(str, groups)))
    folds = min(int(n_splits), unique)
    if folds < 2:
        raise ValueError(
            f"the fit split carries {unique} distinct groups, so no group-disjoint "
            "fold can be built and the ridge penalty cannot be chosen inside the "
            "fit fold"
        )
    splits = make_group_splits(
        np.arange(source.shape[0]),
        np.asarray([str(value) for value in groups]),
        n_splits=folds,
        seed=seed,
        task_type="regression",
    )
    scores: list[float] = []
    for penalty in grid:
        if penalty <= 0.0:
            raise ValueError("a ridge penalty must be positive; 0 is the unpenalised fit")
        fold_scores = [
            _held_out_residual(
                source[test],
                target[test],
                _ridge_weight(source[train], target[train], float(penalty)),
            )
            for train, test in splits
        ]
        scores.append(float(np.mean(fold_scores)))
    best = int(np.argmin(scores))
    return float(grid[best]), {
        "grid": [float(value) for value in grid],
        "held_out_residual_per_penalty": scores,
        "selected": float(grid[best]),
        "n_folds": int(folds),
        "fold_unit": "group-disjoint over the grouping handed to fit_alignment",
        "criterion": "mean held-out residual as a fraction of the target's centred energy",
    }


def fit_alignment(
    X_src: np.ndarray,
    X_tgt: np.ndarray,
    method: str,
    *,
    groups: Sequence[Any] | None = None,
    seed: int = 0,
    ridge_grid: Sequence[float] = DEFAULT_RIDGE_GRID,
    n_splits: int = 5,
) -> AlignmentMap:
    """Fit one rung of the ladder. **On the fit split, and on nothing else.**

    ``mean``
        the shift that carries one cloud's centre onto the other's.
    ``procrustes``
        the orthogonal map, by SVD of the cross-covariance. Rotation and
        reflection only: no rescaling, so it cannot explain a difference away by
        shrinking one side.
    ``affine``
        ridge, with the penalty chosen inside the fit fold on group-disjoint
        folds. Unpenalised least squares is refused rather than offered: at
        ``d_model`` 4096 against a few hundred fit records it has more free
        parameters than data and would fit any pairing at all, which is the
        reading ``25_model_diffing_baselines.py`` exists to have excluded.
    """

    if method not in ALIGNMENT_METHODS:
        raise ValueError(f"unknown alignment method {method!r}; the ladder is {list(ALIGNMENT_METHODS)}")
    source = _finite_matrix(X_src, "X_src")
    target = _finite_matrix(X_tgt, "X_tgt")
    if source.shape[0] != target.shape[0]:
        raise ValueError(
            f"the two sides carry {source.shape[0]} and {target.shape[0]} rows; an "
            "alignment is fitted on paired rows"
        )
    if source.shape[0] < 2:
        raise ValueError("an alignment needs at least two paired records")
    digest = _digest(source, target)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centred_source = source - source_mean
    centred_target = target - target_mean

    if method == "mean":
        if source.shape[1] != target.shape[1]:
            raise ValueError(
                "a mean shift is only defined between two clouds of one width; "
                f"got {source.shape[1]} and {target.shape[1]}"
            )
        weight = np.eye(source.shape[1])
        bias = target_mean - source_mean
        penalty: float | None = None
        selection: dict[str, Any] | None = None
    elif method == "procrustes":
        if source.shape[1] != target.shape[1]:
            raise ValueError(
                "an orthogonal map is only defined between two clouds of one width; "
                f"got {source.shape[1]} and {target.shape[1]}"
            )
        left, _, right = np.linalg.svd(centred_source.T @ centred_target)
        weight = left @ right
        bias = target_mean - source_mean @ weight
        penalty = None
        selection = None
    else:
        if groups is None:
            raise ValueError(
                "the affine rung chooses its ridge penalty inside the fit fold, so "
                "it needs the fit split's grouping; a penalty chosen without one "
                "is chosen across near-duplicate boundaries"
            )
        if len(groups) != source.shape[0]:
            raise ValueError("groups must align with the fitted rows")
        penalty, selection = _select_penalty(
            centred_source,
            centred_target,
            groups,
            grid=ridge_grid,
            seed=seed,
            n_splits=n_splits,
        )
        weight = _ridge_weight(centred_source, centred_target, penalty)
        bias = target_mean - source_mean @ weight

    return AlignmentMap(
        method=method,
        weight=_readonly(weight),
        bias=_readonly(bias.reshape(-1)),
        n_fit=int(source.shape[0]),
        fit_digest=digest,
        penalty=penalty,
        penalty_selection=selection,
    )


def apply_alignment(map_: AlignmentMap, X: np.ndarray) -> np.ndarray:
    """Carry rows through a fitted map."""

    if not isinstance(map_, AlignmentMap):
        raise TypeError("apply_alignment takes an AlignmentMap")
    data = _finite_matrix(X, "X")
    if data.shape[1] != map_.weight.shape[0]:
        raise ValueError(
            f"this map takes {map_.weight.shape[0]}-dimensional rows and these carry "
            f"{data.shape[1]}"
        )
    return data @ map_.weight + map_.bias


# ----------------------------------------------------------------- retrieval


def common_gallery(
    groups: Sequence[Any], *, gallery_size: int, seed: int
) -> np.ndarray:
    """The ``(n, gallery_size)`` gallery each target is ranked in, drawn once.

    EXP-R2-213 fixes the primary statistic on a **common** gallery size, and the
    reason is that a per-query gallery makes the chance level an average over
    galleries of different sizes rather than a number: after the near-duplicate
    exclusion, a record with four copies in the cohort ranks against a smaller
    field than a singleton, so its top-1 accuracy is not the same statistic.
    Here every target ranks against exactly ``gallery_size - 1`` admissible
    distractors and chance is ``1 / gallery_size`` exactly.

    Column zero of each row is the target itself. The draw is seeded and depends
    on nothing but the grouping, the size and the seed, so **every arm, every
    baseline, every null draw and a separate reference run over the same cohort
    rank against the identical field** -- which is what makes the paired
    bootstrap paired and the reference comparison a comparison.
    """

    labels = np.asarray([str(value) for value in groups])
    n = int(labels.size)
    if gallery_size < 2:
        raise ValueError("a gallery of one has nothing to rank the target against")
    if n < gallery_size:
        raise ValueError(
            f"a gallery of {gallery_size} cannot be drawn from {n} records"
        )
    generator = np.random.default_rng(seed)
    out = np.empty((n, int(gallery_size)), dtype=np.int64)
    for index in range(n):
        admissible = np.flatnonzero(labels != labels[index])
        if admissible.size < gallery_size - 1:
            raise ValueError(
                f"record {index} has {admissible.size} admissible distractors and the "
                f"declared gallery size needs {gallery_size - 1}. Its near-duplicate "
                "group is too large a share of this split for the declared size, and "
                "shrinking the gallery for it alone would give it a different chance "
                "level from every other query"
            )
        out[index, 0] = index
        out[index, 1:] = np.sort(
            generator.choice(admissible, size=int(gallery_size) - 1, replace=False)
        )
    return out


def ranks_from_scores(
    scores: np.ndarray, groups: Sequence[Any], *, gallery: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Rank of each target under an arbitrary score matrix, and its gallery size.

    Separated from :func:`retrieval_ranks` because not every baseline scores by
    cosine similarity: a description-side length prior ranks by agreement between
    a predicted and an observed attribute, and it has to be the *same* retrieval
    task on the *same* gallery or its number is not comparable with the ladder's.

    Ties take the midpoint rank, which is what a fair random tie-break gives in
    expectation; awarding rank 1 to a tie would let two identical scores read as a
    perfect retrieval.
    """

    matrix = _finite_matrix(scores, "scores")
    labels = np.asarray([str(value) for value in groups])
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != labels.size:
        raise ValueError(
            f"a score matrix must be square over the records; got {matrix.shape} "
            f"against {labels.size} groups"
        )
    if gallery is None:
        target = np.diagonal(matrix).copy()
        same_group = labels[None, :] == labels[:, None]
        excluded = same_group & ~np.eye(labels.size, dtype=bool)
        greater = ((matrix > target[:, None]) & ~excluded).sum(axis=1)
        tied = ((matrix == target[:, None]) & ~excluded).sum(axis=1) - 1
        sizes = (~excluded).sum(axis=1).astype(np.float64)
    else:
        index = np.asarray(gallery, dtype=np.int64)
        if index.ndim != 2 or index.shape[0] != labels.size:
            raise ValueError("the gallery index must be one row per record")
        if not np.array_equal(index[:, 0], np.arange(labels.size)):
            raise ValueError(
                "column zero of the gallery index must be each record's own target"
            )
        if np.any(labels[index[:, 1:]] == labels[:, None]):
            raise ValueError(
                "the gallery index carries a distractor from a target's own "
                "near-duplicate group, so a near-copy of the answer is in the field"
            )
        drawn = np.take_along_axis(matrix, index, axis=1)
        target = drawn[:, 0].copy()
        greater = (drawn > target[:, None]).sum(axis=1)
        tied = (drawn == target[:, None]).sum(axis=1) - 1
        sizes = np.full(labels.size, float(index.shape[1]))
    ranks = 1.0 + greater.astype(np.float64) + 0.5 * tied.astype(np.float64)
    return ranks, sizes


def retrieval_ranks(
    X_query: np.ndarray,
    X_gallery: np.ndarray,
    groups: Sequence[Any],
    *,
    gallery: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine retrieval: the rank of each query's own gallery row, and the field size.

    ``groups`` is aligned to the **gallery**, and every gallery row sharing a
    target's group -- other than the target itself -- is removed before ranking.
    Without that, a near-duplicate of the target is a second correct answer scored
    as an error. Because the exclusion and the drawn field are properties of the
    gallery alone, a null that permutes the *query* side leaves both untouched,
    which is what makes the null's chance level identical to the truth's by
    construction rather than by adjustment.
    """

    query = _unit_rows(_finite_matrix(X_query, "X_query"), "X_query")
    field = _unit_rows(_finite_matrix(X_gallery, "X_gallery"), "X_gallery")
    if query.shape != field.shape:
        raise ValueError(
            f"query and gallery blocks must pair row for row; got {query.shape} and "
            f"{field.shape}"
        )
    return ranks_from_scores(query @ field.T, groups, gallery=gallery)


def top1_indicators(ranks: np.ndarray) -> np.ndarray:
    """Per-record top-1 hits, as the vector the paired bootstrap resamples."""

    values = np.asarray(ranks, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 1.0):
        raise ValueError("ranks must be a finite one-dimensional vector of ranks >= 1")
    return (values <= 1.0).astype(np.float64)


def metrics_from_ranks(
    ranks: np.ndarray, sizes: np.ndarray, *, ks: Sequence[int] = RETRIEVAL_KS
) -> dict[str, Any]:
    """The reported retrieval block, with **the chance level of this gallery in it**.

    The chance level is not a constant and is not quoted: it is computed from the
    field each target actually ranked in. A retrieval number published without it
    is unreadable -- D3.d recorded four "perfect" corpus predictions that were a
    Spearman over two points.
    """

    ranks = np.asarray(ranks, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    if ranks.shape != sizes.shape or ranks.ndim != 1:
        raise ValueError("ranks and gallery sizes must be aligned one-dimensional vectors")
    harmonic = np.cumsum(1.0 / np.arange(1, int(sizes.max()) + 1))
    top1 = float(np.mean(ranks <= 1.0))
    chance = float(np.mean(1.0 / sizes))
    common = bool(np.all(sizes == sizes[0]))
    record: dict[str, Any] = {
        "n_queries": int(ranks.size),
        "gallery_size": float(sizes[0]) if common else None,
        "common_gallery_size": common,
        "mean_gallery_size": float(sizes.mean()),
        "min_gallery_size": float(sizes.min()),
        "top1_accuracy": top1,
        "top1_chance": chance,
        "top1_excess": top1 - chance,
        "mrr": float(np.mean(1.0 / ranks)),
        # Sum of 1/r over the gallery, divided by its size: the expected
        # reciprocal rank of a uniformly random ranking.
        "mrr_chance": float(np.mean(harmonic[sizes.astype(int) - 1] / sizes)),
    }
    for k in ks:
        if k < 1:
            raise ValueError("a recall cut-off must be positive")
        record[f"recall_at_{k}"] = float(np.mean(ranks <= k))
        record[f"recall_at_{k}_chance"] = float(np.mean(np.minimum(k, sizes) / sizes))
    return record


def retrieval_metrics(
    X_query: np.ndarray,
    X_gallery: np.ndarray,
    groups: Sequence[Any],
    *,
    ks: Sequence[int] = RETRIEVAL_KS,
    gallery: np.ndarray | None = None,
) -> dict[str, Any]:
    """Cosine retrieval reported with the chance level of its own gallery."""

    ranks, sizes = retrieval_ranks(X_query, X_gallery, groups, gallery=gallery)
    return metrics_from_ranks(ranks, sizes, ks=ks)


# ------------------------------------------------------------------ concepts


@dataclass(frozen=True)
class ConceptVector:
    """A unit direction and the scale of the population along it.

    ``sigma`` is the standard deviation of the whole population's projection onto
    ``direction``, and it is carried beside the direction because a steering
    magnitude in raw activation units is not comparable across layers, modes or
    checkpoints -- the same nats of intervention are a different multiple of the
    cloud at every depth. The causal stage builds its graded intervention as
    multiples of ``sigma``, which is what makes "graded" mean the same thing at
    two layers.
    """

    method: str
    direction: np.ndarray
    sigma: float
    n_positive: int
    n_negative: int
    positive_mean_projection: float
    negative_mean_projection: float

    @property
    def separation_sigma(self) -> float:
        """Class separation in units of the population's own spread."""

        return float(
            (self.positive_mean_projection - self.negative_mean_projection) / self.sigma
        )

    def project(self, X: np.ndarray) -> np.ndarray:
        data = _finite_matrix(X, "X")
        if data.shape[1] != self.direction.size:
            raise ValueError(
                f"this concept vector is {self.direction.size}-dimensional and these "
                f"rows carry {data.shape[1]}"
            )
        return data @ self.direction

    def record(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "dimension": int(self.direction.size),
            "sigma": float(self.sigma),
            "n_positive": int(self.n_positive),
            "n_negative": int(self.n_negative),
            "positive_mean_projection": float(self.positive_mean_projection),
            "negative_mean_projection": float(self.negative_mean_projection),
            "separation_sigma": float(self.separation_sigma),
        }


def concept_vector(
    reps: np.ndarray, labels: Sequence[Any], method: str = "diff_means"
) -> ConceptVector:
    """The direction a declared concept moves a population along.

    ``diff_means`` -- the difference of the two class means, normalised. It is
    the whole of what this module offers, and the ``method`` parameter exists so
    that a second rule cannot arrive without being named: a probe direction and a
    difference of means answer different questions, and a stage that silently
    switched between them would publish two estimands under one name.
    """

    if method not in CONCEPT_VECTOR_METHODS:
        raise ValueError(
            f"unknown concept-vector method {method!r}; declared: "
            f"{list(CONCEPT_VECTOR_METHODS)}"
        )
    data = _finite_matrix(reps, "reps")
    truth = np.asarray(labels)
    if truth.ndim != 1 or truth.shape[0] != data.shape[0]:
        raise ValueError("labels must be a one-dimensional vector aligned with reps")
    positive = truth.astype(bool)
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if n_positive < 2 or n_negative < 2:
        raise ValueError(
            f"a concept direction needs at least two records on each side; got "
            f"{n_positive} positive and {n_negative} negative"
        )
    difference = data[positive].mean(axis=0) - data[~positive].mean(axis=0)
    norm = float(np.linalg.norm(difference))
    if norm <= 0.0:
        raise ValueError(
            "the two class means coincide, so this concept has no direction in "
            "these coordinates"
        )
    direction = difference / norm
    projections = data @ direction
    sigma = float(projections.std(ddof=1))
    if sigma <= 0.0:
        raise ValueError(
            "the population has zero spread along this direction, so a graded "
            "intervention in units of sigma is undefined"
        )
    return ConceptVector(
        method=method,
        direction=_readonly(direction),
        sigma=sigma,
        n_positive=n_positive,
        n_negative=n_negative,
        positive_mean_projection=float(projections[positive].mean()),
        negative_mean_projection=float(projections[~positive].mean()),
    )


def _rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """The AUC in its Mann-Whitney form, with midranks for ties.

    Identical to ``sklearn.metrics.roc_auc_score`` on binary labels -- pinned to
    it by a test rather than assumed -- and the reason it is written out is
    arithmetic that showed up in a profile: sklearn's parameter validation costs
    about 12 ms per call whatever the sample size, and this stage's paired
    bootstrap over a pooled concept set makes thousands of calls per cell, which
    was 53 of every 65 seconds of a cell's runtime. One implementation, used by
    both the reported point estimate and the bootstrap, so the interval and the
    number it surrounds cannot come from two definitions.
    """

    ranks = stats.rankdata(scores)
    n_positive = int(labels.sum())
    n_negative = int(labels.size - n_positive)
    if n_positive < 1 or n_negative < 1:
        raise ValueError("an AUC needs both classes present")
    return float(
        (ranks[labels].sum() - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def concept_auc(scores: Sequence[float], labels: Sequence[Any]) -> float:
    """Ranking agreement between a projection and a declared concept."""

    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels).astype(bool)
    if values.shape != truth.shape or values.ndim != 1:
        raise ValueError("scores and labels must be one-dimensional and aligned")
    if not np.isfinite(values).all():
        raise ValueError("scores contain non-finite values")
    if truth.all() or not truth.any():
        raise ValueError(
            "an AUC needs both classes present; a single-class evaluation block "
            "returns nan, and nan compares false against every gate"
        )
    return _rank_auc(truth, values)


def _terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item]
    raise TypeError(
        f"a concept annotation must be a string, a list of strings or null; got "
        f"{type(value).__name__}"
    )


def concept_labels(
    records: Sequence[Mapping[str, Any]], namespace: str, term: str
) -> np.ndarray:
    """Boolean membership of one concept, read from the cohort's own annotation."""

    return np.asarray(
        [term in _terms(record[namespace]) for record in records], dtype=bool
    )


def declared_concepts(
    records: Sequence[Mapping[str, Any]],
    namespaces: Sequence[str],
    *,
    min_positive: int,
    splits: Sequence[str] = SPLITS,
) -> list[tuple[str, str]]:
    """Concepts with enough records on both sides in **every** named split.

    Both sides in every split, not just the fit one: a concept with no negatives
    in the family-holdout block cannot be scored there, and a stage that dropped
    it at scoring time would report a different concept set per split under one
    name.
    """

    if min_positive < 2:
        raise ValueError("a concept needs at least two records a side to be fitted")
    for namespace in namespaces:
        if namespace not in ("ec", "go", "go_propagated", "pfam", "cath"):
            raise ValueError(f"{namespace!r} is not a concept namespace of this cohort")
    by_split = {
        split: [record for record in records if record["split"] == split]
        for split in splits
    }
    concepts: list[tuple[str, str]] = []
    for namespace in namespaces:
        candidates = sorted(
            {term for record in records for term in _terms(record[namespace])}
        )
        for term in candidates:
            if all(
                int(concept_labels(block, namespace, term).sum()) >= min_positive
                and int((~concept_labels(block, namespace, term)).sum()) >= min_positive
                for block in by_split.values()
            ):
                concepts.append((namespace, term))
    return concepts


def _normalised(forms: Any) -> set[str]:
    """Surface forms as they compare: stripped and case-folded, nothing else.

    ``mask_description`` records the **canonical** form it matched rather than the
    spelling it found, so both sides of the comparison are drawn from one
    vocabulary and an exact match is the right test. Case is folded because the
    producer's term set is assembled from several releases -- UniProt's spelling
    of a GO term beside the ontology's -- which differ in capitalisation and not
    in identity.
    """

    return {form.strip().lower() for form in _terms(forms) if form.strip()}


def masked_term_vocabulary(records: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    """Every surface form the cohort stage actually removed from a description."""

    return frozenset(
        form
        for record in records
        for form in _normalised(record["masked_terms"])
    )


def bridge_concepts(
    records: Sequence[Mapping[str, Any]],
    concepts: Sequence[tuple[str, str]],
    *,
    surface_forms: Mapping[tuple[str, str], Sequence[str]],
) -> list[tuple[str, str]]:
    """Concepts defined on **both** sides: annotated, and named in the description.

    Both halves are decided against one object. A concept's *identifier* -- a GO
    id, an EC number, a Pfam accession -- is not what a curated description calls
    it, and ``masked_terms`` is a vocabulary of names: the strings the cohort
    stage removed from ``description_raw`` to build ``description_masked``. Keying
    this test on the identifier compares two vocabularies that cannot meet, and on
    the production cohort it did exactly that -- of 3,970 distinct masked terms
    none was GO-id-shaped or Pfam-accession-shaped, and the intersection was
    empty for all 1,174 declared concepts.

    So ``surface_forms`` is supplied by the caller from
    :func:`src.transfer.sequence_description.concept_surface_forms`, which is the
    same function the cohort stage derives the masked forms from. A concept the
    cohort declares no surface form for has no declared text side and is
    therefore not a bridge concept; that is a property of the declaration and is
    reported as a count rather than inferred here.
    """

    vocabulary = masked_term_vocabulary(records)
    return [
        key
        for key in concepts
        if vocabulary & _normalised(surface_forms.get(key, ()))
    ]


# ------------------------------------------------------------------ surrogates


def composition_features(sequences: Sequence[str]) -> np.ndarray:
    """Amino-acid composition, twenty dimensions, rows summing to one.

    The cheapest thing that is not the model. If an alignment does not beat this,
    it is a statement about which residues a protein is made of.
    """

    if KMER_ALPHABET != AA20:
        raise RuntimeError(
            "the k-mer alphabet and the canonical residue alphabet disagree, so a "
            "composition column would mean two different residues in two modules"
        )
    out = np.zeros((len(sequences), len(AA20)), dtype=np.float64)
    index = {residue: position for position, residue in enumerate(AA20)}
    for row, sequence in enumerate(sequences):
        if not sequence:
            raise ValueError("an empty sequence has no composition")
        for symbol in sequence:
            position = index.get(symbol)
            if position is None:
                raise ValueError(
                    f"{symbol!r} is outside the canonical alphabet {AA20}; the "
                    "rendering this cohort is measured under is declared over it"
                )
            out[row, position] += 1.0
        out[row] /= len(sequence)
    return out


def kmer_features(sequences: Sequence[str], k: int = 3) -> np.ndarray:
    """k-mer frequencies over the canonical alphabet, rows summing to one.

    Indexed through :func:`src.transfer.kmer_background.kmer_index`, so a column
    of this matrix and a row of the staged corpus background are the same k-mer.
    """

    if k < 1:
        raise ValueError("a k-mer has at least one symbol")
    width = len(AA20) ** k
    out = np.zeros((len(sequences), width), dtype=np.float64)
    for row, sequence in enumerate(sequences):
        if len(sequence) < k:
            raise ValueError(
                f"a {len(sequence)}-residue sequence carries no {k}-mer; the cohort's "
                "own length band is what should have excluded it"
            )
        for start in range(len(sequence) - k + 1):
            out[row, kmer_index(sequence[start : start + k])] += 1.0
        out[row] /= len(sequence) - k + 1
    return out


# ---------------------------------------------------------------------- nulls


def shuffled_pairing(n: int, *, draws: int, seed: int) -> np.ndarray:
    """``(draws, n)`` free permutations of the pairing.

    Plain permutations rather than derangements, and the reason is an exact
    identity rather than a preference. Under a uniform permutation the target of
    query ``i`` is uniform over all ``n`` gallery rows, whose ranks in row ``i``
    are a permutation of ``1..n``, so ``E[1/rank] = H_n/n`` -- **exactly the
    chance level** :func:`retrieval_metrics` reports, whatever the true signal
    is. The ``1/n`` fixed points contribute ``1/n`` of a reciprocal rank of 1 and
    the remaining draws contribute ``(H_n - 1)/n``, and the two sum to the
    identity: the truth the fixed points carry is exactly the deficit the rest
    leaves. So this null estimates the chance level *and* the dispersion of the
    metric around it at this sample size, which the closed form alone cannot give.

    A derangement would break that identity in the flattering direction: it can
    never place the target in the slot the true item occupies, so it sits below
    chance at ``(H_n - 1)/(n - 1)``. :func:`rank_matched_pairing` is deranged
    anyway, because its blocks make the arithmetic different -- at width ``b`` the
    fixed-point rate is ``1/b``, an eighth of the truth at the default width, and
    nothing compensates it -- and because it is not the chance calibration. That
    is why the chance level is reported as its own baseline row beside both nulls.
    """

    if n < 2 or draws < 1:
        raise ValueError("a pairing null needs at least two records and one draw")
    generator = np.random.default_rng(seed)
    return np.stack([generator.permutation(n) for _ in range(draws)])


def rank_matched_pairing(
    values: Sequence[float], *, draws: int, seed: int, block: int
) -> np.ndarray:
    """``(draws, n)`` permutations restricted to blocks of the rank order.

    The difficulty-matched counterpart of :func:`shuffled_pairing`, and the same
    idea :func:`src.transfer.concept_lens.rank_matched_partitions` is built on:
    hold fixed the profile that makes any answer look good and randomise only the
    identity. Here the held profile is the nuisance variable the two sides of a
    true pair share -- sequence length -- so a retrieval that runs on length
    survives this null and only a correspondence beyond it does not.

    **Within a block the permutation is a derangement, and that is not a detail.**
    A free permutation of a width-``b`` block leaves each record with its own
    true partner with probability ``1/b`` -- 12.5% of the records at the default
    width -- so a null built from free block permutations contains an eighth of
    the truth and is not a null of the correspondence at all. Measured on the
    instrument check before this was fixed, the free-block null read a decision
    level of 0.285 against the free permutation's 0.154 on data whose length was
    independent of everything. The fixed-point fraction is reported by
    :func:`pairing_match_quality` so the property is checked rather than claimed.

    The match itself is measured rather than asserted; see the same function.
    """

    ordering = np.argsort(np.asarray(values, dtype=np.float64), kind="stable")
    n = ordering.size
    if n < 2 or draws < 1:
        raise ValueError("a pairing null needs at least two records and one draw")
    if block < 2:
        raise ValueError(
            "a rank-matched block of one admits only the identity pairing, which is "
            "the truth and not a null"
        )
    boundaries = list(range(0, n, block))
    # A trailing block of one cannot be permuted, so it joins the block before it
    # rather than silently pinning one record to its true partner in every draw.
    if len(boundaries) > 1 and n - boundaries[-1] < 2:
        boundaries.pop()
    generator = np.random.default_rng(seed)
    out = np.empty((draws, n), dtype=np.int64)
    for draw in range(draws):
        permuted = ordering.copy()
        for start, stop in zip(boundaries, boundaries[1:] + [n]):
            permuted[start:stop] = _deranged(ordering[start:stop], generator)
        assignment = np.empty(n, dtype=np.int64)
        assignment[ordering] = permuted
        out[draw] = assignment
    return out


def _deranged(block: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    """A permutation of ``block`` with no element left where it started.

    Rejection sampling: about ``1/e`` of draws are derangements, so this costs
    under three draws per block on average and is exact rather than approximate.
    """

    if block.size < 2:
        raise ValueError("a block of one admits no derangement")
    while True:
        candidate = generator.permutation(block)
        if not np.any(candidate == block):
            return candidate


def pairing_match_quality(
    values: Sequence[float], pairings: np.ndarray
) -> dict[str, Any]:
    """How well a null's pairings matched the nuisance the truth pairs on.

    Reported for the same reason ``concept_lens.partition_null_quality`` is: a
    matched null that did not match is a free null under a stricter name.
    """

    ranks = np.empty_like(np.asarray(values, dtype=np.float64))
    ranks[np.argsort(np.asarray(values, dtype=np.float64), kind="stable")] = np.arange(
        len(values), dtype=np.float64
    )
    identity = np.arange(len(values))
    drawn = np.atleast_2d(pairings)
    gaps = [float(np.abs(ranks - ranks[pairing]).mean()) for pairing in drawn]
    fixed = [float(np.mean(pairing == identity)) for pairing in drawn]
    return {
        "mean_absolute_rank_gap": float(np.mean(gaps)),
        "max_absolute_rank_gap": float(np.max(gaps)),
        "mean_fixed_point_fraction": float(np.mean(fixed)),
        "max_fixed_point_fraction": float(np.max(fixed)),
        "fixed_point_note": (
            "the fraction of records a draw left paired with their own true "
            "partner. A fixed point is the truth and not a null draw, so a matched "
            "null is deranged inside its blocks and this reads zero for it; a free "
            "permutation leaves 1/n on average, which is the chance model"
        ),
        "n_records": int(len(values)),
        "free_permutation_expected_gap": float(len(values) / 3.0),
        "note": (
            "mean |rank(i) - rank(pi(i))| over the nuisance variable the true pairs "
            "share. A free permutation of n records averages about n/3; a matched "
            "null must sit far below it or it is not matched"
        ),
    }


def null_distribution(values: Sequence[float]) -> dict[str, Any]:
    """A null reported as a distribution, never as a point."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("a null distribution needs at least two draws")
    if not np.isfinite(array).all():
        raise ValueError("the null carries non-finite draws")
    interval = mean_interval(array)
    return {
        "n_draws": int(array.size),
        "mean": float(array.mean()),
        "mean_ci95": [float(interval["interval"][0]), float(interval["interval"][1])],
        "sd": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
        "q95": float(np.percentile(array, 95.0)),
        "q975": float(np.percentile(array, 97.5)),
        "decision_level": float(np.percentile(array, 97.5)),
        "decision_level_note": (
            "the null's 97.5th percentile, which is what an observed value is "
            "required to clear. The mean of a null is not a bar"
        ),
    }


# ------------------------------------------------------------------ metrics


def mean_metric(truth: np.ndarray, prediction: np.ndarray) -> float:
    """The mean of a per-record statistic -- top-1 accuracy's own metric.

    Handed to :func:`src.transfer.statistics.paired_group_bootstrap` as its
    metric, with the per-record top-1 indicators as the prediction vector, so the
    resampling unit is the group and the statistic is the one the criteria name.
    """

    return float(np.mean(prediction))


def auc_metric(truth: np.ndarray, prediction: np.ndarray) -> float:
    """AUC, returning ``nan`` on a single-class resample rather than raising.

    ``paired_group_bootstrap`` discards a non-finite draw and refuses the whole
    interval if too many of them are, which is the behaviour a single-class
    resample should get: it is a fact about the stratum, and the unit floor
    decides whether what survives is publishable.
    """

    labels = np.asarray(truth).astype(bool)
    if labels.all() or not labels.any():
        return float("nan")
    return _rank_auc(labels, np.asarray(prediction, dtype=np.float64))


# ------------------------------------------------------------------- verdict


def protein_mode_behavioural_status(checkpoint: Path | str) -> dict[str, Any]:
    """What may be claimed about this checkpoint's protein mode.

    Keyed by the checkpoint directory's name, because a checkpoint is reached by
    path in this programme and never by panel-arm name. A checkpoint nobody has
    qualified is ``undeclared``, which is not the same as measurable.
    """

    name = Path(checkpoint).resolve().name
    declared = PROTEIN_MODE_BEHAVIOURAL_STATUS.get(name)
    if declared is None:
        return {
            "checkpoint_name": name,
            "measurable": None,
            "reason": (
                "this checkpoint is not declared in "
                "src.transfer.concept_alignment.PROTEIN_MODE_BEHAVIOURAL_STATUS. "
                "21_joint_mode_qualification.py is what admits a checkpoint to a "
                "mode, and an undeclared checkpoint's protein mode is unqualified "
                "rather than measurable"
            ),
        }
    return {"checkpoint_name": name, **dict(declared)}


def assert_behavioural_read_permitted(checkpoint: Path | str, mode: str) -> None:
    """Refuse to hand a causal stage a target that cannot be read back.

    Called before this stage emits the concept-vector hand-off the causal stage
    consumes. The hand-off is the one thing here that is not representational:
    it names a direction a later stage will steer along and read a behavioural
    response from, and on a checkpoint whose protein mode is behaviourally
    unmeasurable there is no response to read.
    """

    status = protein_mode_behavioural_status(checkpoint)
    if mode != "protein":
        return
    if status.get("measurable") is True:
        return
    raise ValueError(
        f"{status['checkpoint_name']}: no behavioural read of this checkpoint's "
        "protein mode is permitted. "
        + str(
            status.get(
                "reason",
                "its protein mode is not declared measurable by "
                "21_joint_mode_qualification.py",
            )
        )
    )


def assert_ladder_reported(reported: Sequence[str], primary: str) -> None:
    """Refuse a rung read before the rungs below it have been fitted and reported.

    EXP-R2-213's A35-3. The ladder is an *order* and not a menu: the whole point
    of running the mean shift and the orthogonal map before the ridge is that a
    ridge result is only interpretable once the two cheaper explanations have
    been priced, and a stage that could emit the top rung alone would defeat the
    pre-registration it is gated by.
    """

    if primary not in ALIGNMENT_METHODS:
        raise ValueError(
            f"unknown primary rung {primary!r}; the ladder is {list(ALIGNMENT_METHODS)}"
        )
    have = set(reported)
    required = ALIGNMENT_METHODS[: ALIGNMENT_METHODS.index(primary) + 1]
    missing = [rung for rung in required if rung not in have]
    if missing:
        raise ValueError(
            f"the ladder is read at {primary!r} and {missing} were not reported. "
            f"The order is {list(ALIGNMENT_METHODS)} and every rung at or below the "
            "one a verdict is read on must be fitted and reported beside it"
        )


def assert_per_layer_only(payload: Mapping[str, Any], layers: Sequence[int]) -> None:
    """Refuse an artefact that reports a decisive quantity as a cross-layer mean.

    L32's second half: a criterion stated per unit that is evaluated as a mean
    over units is not the same rule at a different resolution -- a dictionary
    averaging 7,608 live latents per layer kept 90 at one of them, so the mean and
    the criterion disagreed in verdict. This walks the payload and refuses any key
    that names an aggregate over layers, and checks that every declared layer
    carries its own cell.
    """

    forbidden = ("across_layers", "cross_layer", "layer_mean", "mean_over_layers")

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                lowered = str(key).lower()
                for marker in forbidden:
                    if marker in lowered:
                        raise ValueError(
                            f"{path}.{key} names an aggregate over layers; every "
                            "quantity this stage decides on is per layer (L32)"
                        )
                walk(value, f"{path}.{key}")
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(payload, "payload")
    cells = payload.get("cells")
    if not isinstance(cells, Mapping):
        raise ValueError("the payload carries no per-layer cells to check")
    for layer in layers:
        if not any(key.startswith(f"layer{int(layer)}__") for key in cells):
            raise ValueError(
                f"layer {layer} was declared and carries no cell of its own"
            )


def baseline_row(
    name: str,
    axis: str,
    observed: float,
    baseline: float | None,
    *,
    chance: float | None = None,
    excess_ratio: float | None = None,
    decisive: bool = True,
    applicable: bool = True,
    inapplicable_reason: str | None = None,
    interval: Mapping[str, Any] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """One baseline under EXP-R2-213's two conditions, both reported separately.

    Condition (i) is the paired group-bootstrap 95% interval of the difference
    excluding zero. Condition (ii) is the effect-size bar
    ``(observed - chance) >= excess_ratio * (baseline - chance)``. They are
    deliberately separate and both required: significance alone is a detection
    criterion and does not license a comparative claim, a distinction that has
    already forced a retraction in this programme's induction-head counts.

    ``interval`` has three states and they are not two. "No interval" and "an
    interval the unit floor refused" read identically in a null field and mean
    opposite things: a null baseline's evidence IS its own draw distribution and
    it was never going to carry a paired bootstrap, while a refused one is a
    stratum too small to bound.
    """

    if applicable and baseline is None:
        raise ValueError(f"baseline {name!r} is applicable but carries no value")
    # An inapplicable baseline MAY still carry its measured value: a surrogate that
    # is not a surrogate for this cell's source is still worth reporting, and
    # dropping the number would make "inapplicable" indistinguishable from "not
    # run". What it may not do is decide anything, which is what `applicable` says.
    if not applicable and inapplicable_reason is None:
        raise ValueError(
            f"baseline {name!r} is inapplicable and carries no reason; an omitted "
            "baseline that does not say why is indistinguishable from one that was "
            "not run"
        )
    margin = None if baseline is None else float(observed) - float(baseline)
    observed_excess = None if chance is None else float(observed) - float(chance)
    baseline_excess = (
        None if (chance is None or baseline is None) else float(baseline) - float(chance)
    )
    meets_ratio: bool | None = None
    if observed_excess is not None and baseline_excess is not None and excess_ratio is not None:
        # A baseline at or below chance carries no excess to multiply, so the bar
        # reduces to the method showing an excess at all. Stated rather than left
        # to the arithmetic, because ``x >= 2 * negative`` is satisfied by a
        # negative x and would read as a pass for a method below chance.
        required = float(excess_ratio) * max(baseline_excess, 0.0)
        meets_ratio = bool(observed_excess > 0.0 and observed_excess >= required)
    excludes_zero: bool | None = None
    if interval is None:
        interval_status = "not_paired"
    elif interval.get("publishable"):
        low, high = interval["bootstrap"]["difference_ci95"]
        excludes_zero = bool(low > 0.0 or high < 0.0)
        interval_status = "published"
    else:
        interval_status = "below_unit_floor"
    passes: bool | None = None
    if applicable:
        conditions = [excludes_zero, meets_ratio]
        passes = None if any(value is None for value in conditions) else all(conditions)
    return {
        "baseline": name,
        "axis": axis,
        "decisive": bool(decisive),
        "applicable": bool(applicable),
        "inapplicable_reason": inapplicable_reason,
        "observed": float(observed),
        "baseline_value": None if baseline is None else float(baseline),
        "chance": None if chance is None else float(chance),
        "observed_excess": observed_excess,
        "baseline_excess": baseline_excess,
        "margin": margin,
        "excess_ratio_required": None if excess_ratio is None else float(excess_ratio),
        "meets_excess_ratio": meets_ratio,
        "interval": None if interval is None else dict(interval),
        "interval_status": interval_status,
        "difference_interval_excludes_zero": excludes_zero,
        "passes_both_conditions": passes,
        "note": note,
    }


def attainability_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A35-0's gate: one baseline, both of A35-1's clauses, and it says which one.

    EXP-R2-213 states A35-0 in the singular -- "if the identical ladder on raw
    descriptions does not clear A35-1's margin over ``shuffled_pair``, the ladder
    is void as a specification defect and the masked arm is not read at all" --
    and its branch table gives the two failures opposite subjects: a raw arm
    failing A35-0 is about *the instrument, not the modality*, while a
    composition or 3-mer surrogate breaking A35-1(ii) is the surface-statistics
    branch and is about *the method*. Reading the gate off the whole decisive set
    merges them, and a measured negative on the second branch then reaches the
    record as a void.

    The gate is deliberately the row's own two clauses and nothing else: the
    detection floor is a run-level flag rather than a frozen criterion, and at the
    pre-registered ``--decision-threshold 0.0`` clause (ii) already subsumes it,
    because :func:`baseline_row` requires a strictly positive excess before it can
    be met. The floor is still read on the raw arm's full verdict, which is
    reported beside this gate.
    """

    matches = [
        row
        for row in rows
        if row["baseline"] == A35_0_GATE_BASELINE and row["axis"] == "primary_top1"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"A35-0's gate is taken on {A35_0_GATE_BASELINE!r} on the primary "
            f"statistic and {len(matches)} such rows were reported. The gate cannot "
            "fall back to another baseline: which one it was taken on is the whole "
            "distinction between a void instrument and a measured negative"
        )
    row = matches[0]
    if not row["applicable"]:
        raise ValueError(
            f"A35-0's gate baseline {A35_0_GATE_BASELINE!r} is reported as "
            f"inapplicable ({row['inapplicable_reason']!r}). It permutes the query "
            "side and leaves the gallery untouched, so it is applicable in every "
            "cell this stage runs; an inapplicable one means the row is not the "
            "row this gate names"
        )
    passes = row["passes_both_conditions"]
    if passes is True:
        status = "cleared"
    elif passes is False:
        status = "not_cleared"
    else:
        status = "not_evaluable"
    return {
        "gate_baseline": A35_0_GATE_BASELINE,
        "attainable": passes is True,
        "status": status,
        "difference_interval_excludes_zero": row["difference_interval_excludes_zero"],
        "interval_status": row["interval_status"],
        "meets_excess_ratio": row["meets_excess_ratio"],
        "excess_ratio_required": row["excess_ratio_required"],
        "observed_excess": row["observed_excess"],
        "baseline_excess": row["baseline_excess"],
        "row": dict(row),
        "note": A35_0_GATE_NOTE,
    }


def admission_verdict(
    rows: Sequence[Mapping[str, Any]],
    *,
    excess_ratio: float,
    detection_floor: float,
    observed_excess: float,
    behavioural_status: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    """EXP-R2-213's decision, and the reason for it.

    Only the rows the pre-registration names decide anything: the frozen decisive
    set is :data:`A35_1_BASELINES`, every one of which must be present. A baseline
    missing from the table is a refusal rather than a shorter table, and a
    baseline outside it -- the ambient nearest-neighbour read, the analytic chance
    level, the whole concept axis -- is reported and non-decisive, because
    widening a frozen criterion is as much a change to it as softening one.

    :data:`A35_1B_BASELINE` is the one row whose decisiveness is *measured* rather
    than frozen (amendment 1). It reaches this function already carrying the
    caller's A35-1b decision in its own ``decisive`` flag, which is the flag
    :func:`baseline_row` exists to carry, so there is still exactly one place a
    verdict is computed from a table of rows.

    ``PASS`` requires the observed excess over chance to clear ``detection_floor``
    and every decisive, applicable baseline to satisfy **both** conditions.
    ``UNDERPOWERED`` is returned rather than ``PASS`` when a condition holds but
    the resampling unit count is below the eight-unit floor: a nominal 95%
    interval over fewer units realises well under 95% coverage and can come out
    narrower than one over hundreds.

    ``REFERENCE_ONLY`` is returned, whatever the numbers say, for a checkpoint
    whose protein mode is behaviourally unmeasurable. Its rows are still reported
    in full: the pre-adaptation reference is a representational comparison and
    that is the whole of what it can be.

    **``REFERENCE_ONLY`` is not a failure and is not evaluated as one.** It says
    "this checkpoint is not an arm"; ``FAIL`` says "this arm lost". So the
    criteria are read on the reference's own numbers first and reported as
    ``criteria_verdict``, and the reference label is an overlay on top of them
    rather than a branch that pre-empts them. A reference arm exists precisely to
    be compared against on the deciding variant, so a caller that needs to know
    whether the reference's own ladder holds up -- A35-0's attainability, or
    whether its masked arm is worth computing -- reads ``criteria_verdict`` and
    never has to infer it from a label that could only ever be one value.
    ``verdict`` stays ``REFERENCE_ONLY`` and authorises nothing.
    """

    decisive = [row for row in rows if row["decisive"]]
    present = {row["baseline"] for row in decisive}
    missing = [name for name in A35_1_BASELINES if name not in present]
    if missing:
        raise ValueError(
            f"the pre-registered baselines {missing} carry no row. A35-1 names "
            f"{list(A35_1_BASELINES)} and every one of them must be reported, "
            "applicable or refused with a reason"
        )
    applicable = [row for row in decisive if row["applicable"]]
    if not applicable:
        raise ValueError(
            "no applicable decisive baseline was reported, so there is nothing for "
            "the alignment to have beaten"
        )
    failed = sorted(
        {row["baseline"] for row in applicable if row["passes_both_conditions"] is False}
    )
    unpowered = sorted(
        {
            row["baseline"]
            for row in applicable
            if row["passes_both_conditions"] is None
            or row["interval_status"] == "below_unit_floor"
        }
    )
    below_floor = float(observed_excess) < float(detection_floor)
    if below_floor:
        criteria_verdict = "FAIL"
        criteria_reason = (
            f"the primary statistic's excess over chance is {observed_excess:.4f}, "
            f"below the pre-registered detection floor of {detection_floor}"
        )
    elif failed:
        criteria_verdict = "FAIL"
        criteria_reason = (
            "did not satisfy both pre-registered conditions against "
            + ", ".join(failed)
            + f" -- a paired 95% interval excluding zero AND an excess over chance at "
            f"least {excess_ratio}x the baseline's"
        )
    elif unpowered:
        criteria_verdict = "UNDERPOWERED"
        criteria_reason = (
            "every decisive baseline was beaten on the conditions that could be "
            "evaluated, and "
            + ", ".join(unpowered)
            + f" carries no usable interval: fewer than {MINIMUM_BOOTSTRAP_UNITS} "
            "resampling units, which is below the floor a percentile interval may be "
            "published at"
        )
    else:
        criteria_verdict = "PASS"
        criteria_reason = (
            f"cleared the detection floor at an excess of {observed_excess:.4f} and "
            f"satisfied both conditions against every decisive applicable baseline: "
            f"{sorted(row['baseline'] for row in applicable)}"
        )
    if mode == "protein" and behavioural_status.get("measurable") is not True:
        verdict = "REFERENCE_ONLY"
        reason = (
            f"{behavioural_status.get('checkpoint_name')}: "
            + str(
                behavioural_status.get(
                    "reason",
                    "this checkpoint's protein mode is not declared behaviourally "
                    "measurable",
                )
            )
            + ". The rows below are a representational comparison and no admission "
            "decision is recorded for them. Its own numbers read "
            + f"{criteria_verdict}, which is reported as criteria_verdict and "
            "authorises nothing: REFERENCE_ONLY says this checkpoint is not an arm, "
            "not that this arm lost"
        )
    else:
        verdict, reason = criteria_verdict, criteria_reason
    return {
        "verdict": verdict,
        "reason": reason,
        "criteria_verdict": criteria_verdict,
        "criteria_reason": criteria_reason,
        "primary_statistic": PRIMARY_STATISTIC,
        "primary_statistic_note": PRIMARY_STATISTIC_NOTE,
        "observed_excess": float(observed_excess),
        "detection_floor": float(detection_floor),
        "excess_ratio": float(excess_ratio),
        "pre_registration": PRE_REGISTRATION,
        "amendments_implemented": list(PRE_REGISTRATION_AMENDMENTS),
        "decisive_baselines": list(A35_1_BASELINES),
        "decisive_applicable_baselines": sorted(row["baseline"] for row in applicable),
        "n_decisive_applicable": len(applicable),
        "baselines_failing_a_condition": failed,
        "baselines_without_a_usable_interval": unpowered,
        "note": (
            "this stage's verdict is about the LINEAR ladder only. The gate also "
            "requires a graded protein-model intervention in the predicted "
            "direction and preservation of unrelated concepts under it, which are "
            "causal and belong to the intervention stage. " + NONLINEAR_ADAPTER_NOTE
        ),
    }
