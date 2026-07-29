"""Decodability versus reliance: probes and concept erasure on decoder residuals.

A probe answers whether a concept is *present* in an activation. It cannot
answer whether the model's next-token computation *consults* it, and in protein
decoders that gap is load-bearing: a generator can carry fold state or family
identity in its residual stream while routing none of it into the next-residue
prediction. An interpretability method that reports such a concept as a
"feature" is then describing a passenger, not a cause. Every probe here is
therefore paired with a concept-erasure intervention, and every erasure with a
dimension-matched random control, because deleting k directions from a residual
stream always costs some loss and only the excess over that matched cost is
evidence of reliance.

Four design rules the numbers depend on.

*Grouped splits.* EC class, Pfam family and DMS fitness are protein- or
family-level labels. A record-level split puts homologues of a test protein in
the training set, so the probe reports family identity rather than the concept.
Every split is disjoint over an explicitly named grouping variable, recorded in
the output; a concept whose grouping variable cannot be built raises rather than
degrading to a record-level split. The one concept where family-disjointness is
degenerate by construction is ``pfam_family`` itself - the label *is* the group -
and its grouping is therefore within-family sequence redundancy, declared as
such with ``family_disjoint`` set to false rather than quietly relabelled.

*Matched capacity.* Arms differ in width (1280 vs 1536) and depth (36 vs 27), so
every probe reads a fold-fitted PCA of one fixed dimension, taken at a
relative-depth grid, and both a linear probe and a small MLP are reported:
"present but not linearly readable" and "absent" are different findings.

*Linear guarantees only.* The erasure is LEACE (Belrose et al., "LEACE: Perfect
linear concept erasure in closed form", NeurIPS 2023, arXiv:2306.03819), an
affine oblique projection that provably zeroes the cross-covariance between the
activation and the concept, hence the skill of *every* linear probe. It makes no
claim about nonlinear readers, so the verification gate is applied to the linear
probe and the post-erasure MLP is reported as a diagnostic, not as a failure.

*Analogy, not identity.* The text arm is scored on next-token class and next-token
rarity. Those are matched-difficulty per-position categorical tasks; they are not
the same tasks as secondary structure or burial. A difference in probe accuracy
between the text and protein arms is therefore not a measurement of how much
structure either modality's decoder encodes.

Structural labels come from AlphaFold CA traces. The three-state secondary
structure is the coordinate-only P-SEA distance criterion already used elsewhere
in this package, not DSSP, and burial is a CA contact-number band, not a
Shrake-Rupley relative solvent accessibility. Both approximations are uniform
across arms, which is what the comparison needs, and neither may be quoted as
its reference-standard namesake.

Two arm-specific refusals are findings rather than gaps.

ProtGPT2 tokenises several residues per token, so a residue-level label cannot
be addressed on its activations at all: secondary structure, burial and the
variant-level fitness read at a mutated site are refused for it. Sequence-level
labels are constant along the sequence and therefore need only the token span
that covers the sequence body, which is recoverable and is verified by decoding
that span back to the exact sequence, so EC class and Pfam family are measured
on ProtGPT2. Because one probe sample is one token on every arm, matching the
number of sampled positions matches the probe training set on tokens rather than
on sequences, which is the comparison ProtGPT2's ~3-5x coarser tokenisation
requires.

ZymCTRL carries an EC tag in its own prompt, and on an enzyme cohort that tag
very nearly determines both the EC class and the Pfam family. Under its native
conditioning, a sequence-level function or family probe on ZymCTRL therefore
measures how well the model copies its own prompt, and those concepts are
refused rather than reported with a caveat. Two conditioning alternatives are
implemented and selectable, both recorded in the output: ``fixed`` gives every
prompt one constant EC tag, which keeps the arm in its native input format while
reducing the tag's mutual information with any label to zero, and
``unconditioned`` drops the tag entirely, which removes the leak at the cost of
taking the arm off the distribution it was trained on. Residue-level structure
probes are *not* affected by this leak in the same way, and are measured under
native conditioning: an EC tag constrains a protein's overall chemistry but does
not determine which residue at which position is helical or buried, and the
measured ZymCTRL structure skills sit alongside the unconditioned arms' rather
than above them.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.hooks import RemovableHandle

from .arms import (
    AA20,
    REPO,
    SWISSPROT_FASTA,
    ZYMCTRL_FASTA,
    Arm,
    Cohort,
    conditioning_boundary_ids,
    env_path,
    iter_fasta,
    require_input_path,
    text_cohort,
    tokenize_batch,
)
from .channels import (
    ALPHAFOLD_ROOT,
    PFAM_RESIDUE_TSV,
    SECONDARY_STRUCTURE_CAVEAT,
    Structure,
    alphafold_models,
    ca_secondary_structure,
    contact_number_bins,
    load_pfam_spans,
    read_alphafold_model,
)
from .relational import (
    homology_clusters,
    require_residue_token_map,
    residue_token_offset,
)
from .scoring import (
    aggregate_variant,
    analysis_layer,
    analysis_layers,
    per_sequence_scores,
    sequence_target_mask,
    target_rule,
)
from .statistics import make_group_splits, paired_group_bootstrap

SCHEMA_VERSION = "r2_transfer_probe_erasure_v1"

PROTEINGYM_ROOT = env_path(
    "TRANSFER_PROTEINGYM_DIR", REPO / "data/proteingym/DMS_ProteinGym_substitutions"
)

#: Relative depths, so a 36-layer and a 27-layer arm are read at comparable
#: places in their own computation rather than at the same absolute index.
LAYER_FRACTIONS: tuple[float, ...] = (0.25, 0.5, 0.75)

#: Burial bands on the CA contact number within 10 angstrom. Prespecified and
#: fixed rather than fitted per cohort, so the label does not move when the
#: cohort does; the realised class balance is reported instead.
CONTACT_RADIUS_ANGSTROM = 10.0
BURIAL_BANDS: tuple[int, int] = (12, 18)
BURIAL_NAMES: tuple[str, str, str] = ("exposed", "intermediate", "buried")
SECONDARY_STRUCTURE_NAMES: tuple[str, str, str] = ("helix", "strand", "coil")
BURIAL_CAVEAT = (
    "ca_contact_number_band_within_10A; not a Shrake-Rupley relative solvent "
    "accessibility and not a DSSP-derived burial state"
)

TEXT_CLASS_NAMES: tuple[str, ...] = (
    "word_start_lower",
    "word_start_capitalised",
    "subword_continuation",
    "punctuation_or_space",
    "numeral",
)
RARITY_NAMES: tuple[str, str, str] = ("common", "mid", "rare")

CHANCE_MODEL = "prior_matched_random_draw_refit_in_every_training_fold"

#: The prespecified control the reliance claim is charged against, and why each
#: control means what it means. Deleting k directions costs cross-entropy
#: whether or not the model reads them, so the headline number is an excess over
#: a control, and which control is used changes what the excess licenses.
PRIMARY_CONTROL = "random_raw_orthonormal"
CONTROL_INTERPRETATION = {
    "random_raw_orthonormal": (
        "matched rank, orthogonal projection in the raw activation basis: the "
        "literal 'k random directions' control, and the prespecified primary "
        "one. An orthogonal projection can only shorten an activation, so it "
        "cannot manufacture an out-of-distribution state, which is what makes "
        "its cost a usable floor on every arm."
    ),
    "random_whitened_orthonormal": (
        "matched rank and matched construction: the LEACE map itself with a "
        "random direction in the whitened space LEACE picks its own direction "
        "from. It is the closest structural analogue of the erasure. Whitening "
        "makes it oblique and scale-blind, so on an arm whose residual "
        "covariance is ill-conditioned it can land on a direction the model is "
        "extremely sensitive to and cost as much as deleting the layer outright "
        "while displacing activations *less* than the erasure does. Where its "
        "cost approaches the mean-ablation reference it is not a usable floor "
        "and the excess against it means nothing."
    ),
    "variance_matched_random": (
        "matched rank and, by bisection on a rotation towards the principal "
        "axes, exactly matched removed variance. It is the strictest control "
        "here, but the variance it deletes is concentrated in whichever "
        "directions carry it, so on an arm whose residual stream is dominated by "
        "a few directions it deletes something the model cannot do without. Its "
        "excess is a conservative bound, not the headline."
    ),
}

#: Above this ratio of activation displacement to the erasure's own, a control
#: is doing something categorically larger than the erasure and its cost is not
#: a matched cost. Recorded per control rather than used to drop it.
CONTROL_DISPLACEMENT_TOLERANCE = 3.0

#: A control that costs this share of the mean-ablation reference is deleting
#: the layer, not matching the erasure, and cannot serve as a floor.
CONTROL_MEAN_ABLATION_SHARE = 0.5

#: How an EC-conditioned arm is prompted. ``native`` is the arm's own format and
#: leaks the label into sequence-level probes; ``fixed`` keeps the format but
#: gives every prompt the same tag, so the tag carries no information about any
#: label; ``unconditioned`` drops the tag and takes the arm off its training
#: distribution. The choice is recorded in every artifact it affects.
EC_CONDITIONING_MODES = ("native", "fixed", "unconditioned")
FIXED_EC_TAG = "1.1.1.1"

ERASURE_METHOD = "leace_affine_oblique_projection"
ERASURE_CITATION = (
    "Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff, Biderman (2023), "
    "'LEACE: Perfect linear concept erasure in closed form', NeurIPS 2023, "
    "arXiv:2306.03819"
)


# --------------------------------------------------------------- concept table


@dataclass(frozen=True)
class ConceptSpec:
    """One probing target, with the split policy it is only valid under."""

    name: str
    modality: str
    level: str
    task_type: str
    metric: str
    grouping: str
    family_disjoint: bool
    label_source: str
    rationale: str


CONCEPTS: dict[str, ConceptSpec] = {
    "ss3": ConceptSpec(
        name="ss3",
        modality="protein",
        level="residue",
        task_type="classification",
        metric="macro_f1",
        grouping="homology_cluster_pfam_architecture_or_kmer",
        family_disjoint=True,
        label_source="alphafold_ca_trace_psea_distance_criterion",
        rationale=(
            "Local fold state is the structural concept a protein decoder is most "
            "often claimed to represent; residue-level labels make it the highest "
            "powered structural probe available without DSSP."
        ),
    ),
    "burial": ConceptSpec(
        name="burial",
        modality="protein",
        level="residue",
        task_type="classification",
        metric="macro_f1",
        grouping="homology_cluster_pfam_architecture_or_kmer",
        family_disjoint=True,
        label_source="alphafold_ca_contact_number_band",
        rationale=(
            "Burial is the graded per-residue structural property that constrains "
            "which residues are substitutable, and is the closest structural "
            "analogue of a continuous per-position text attribute."
        ),
    ),
    "ec_class": ConceptSpec(
        name="ec_class",
        modality="protein",
        level="sequence",
        task_type="classification",
        metric="macro_f1",
        grouping="dominant_pfam_family",
        family_disjoint=True,
        label_source="ec_labelled_swissprot_top_level_ec_digit",
        rationale=(
            "Top-level EC class is a function label that is not a family label: "
            "many families map to one class, so a family-disjoint split still "
            "leaves the concept learnable, which is exactly the property that "
            "makes the grouped result interpretable."
        ),
    ),
    "pfam_family": ConceptSpec(
        name="pfam_family",
        modality="protein",
        level="sequence",
        task_type="classification",
        metric="macro_f1",
        grouping="within_family_kmer_redundancy_cluster",
        family_disjoint=False,
        label_source="interpro_pfam_residue_spans_single_family_proteins",
        rationale=(
            "The label is the family, so a family-disjoint split makes the target "
            "unlearnable by construction. The leak this concept can actually "
            "suffer is near-duplicate homologues of a test protein sitting in the "
            "training set, so the grouping unit is a within-family sequence "
            "redundancy cluster. This is declared, not substituted silently, and "
            "the concept must not be read as evidence about family-disjoint "
            "generalisation."
        ),
    ),
    "fitness": ConceptSpec(
        name="fitness",
        modality="protein",
        level="variant",
        task_type="regression",
        metric="spearman_rho",
        grouping="proteingym_assay_one_assay_per_target_protein",
        family_disjoint=True,
        label_source="proteingym_dms_substitutions_single_mutants",
        rationale=(
            "Fitness is the concept with the clearest downstream use and the "
            "weakest claim to being computed by a generative decoder; an "
            "assay-disjoint split is the only split under which a positive result "
            "is not assay memorisation. ProteinGym ships several assays per "
            "protein -- BLAT_ECOLX alone has four, all of one 286-residue "
            "sequence -- so assay-disjoint is not by itself protein-disjoint, and "
            "family_disjoint would have been an assertion rather than a fact. "
            "fitness_units admits at most one assay per target protein, which is "
            "what makes it true."
        ),
    ),
    "next_token_class": ConceptSpec(
        name="next_token_class",
        modality="text",
        level="token",
        task_type="classification",
        metric="macro_f1",
        grouping="document",
        family_disjoint=False,
        label_source="deterministic_surface_class_of_the_next_token",
        rationale=(
            "A matched-difficulty per-position categorical task for the text arm, "
            "analogous in granularity and class count to three-state secondary "
            "structure. It is an analogue, not the same concept."
        ),
    ),
    "next_token_rarity": ConceptSpec(
        name="next_token_rarity",
        modality="text",
        level="token",
        task_type="classification",
        metric="macro_f1",
        grouping="document",
        family_disjoint=False,
        label_source="cohort_empirical_next_token_frequency_terciles",
        rationale=(
            "A graded per-position text attribute matched to burial in the same "
            "way next-token class is matched to secondary structure."
        ),
    ),
}


def concepts_for_modality(modality: str) -> tuple[str, ...]:
    return tuple(
        name for name, spec in CONCEPTS.items() if spec.modality == modality
    )


def check_ec_conditioning(ec_conditioning: str) -> str:
    if ec_conditioning not in EC_CONDITIONING_MODES:
        raise ValueError(
            f"unknown EC conditioning {ec_conditioning!r}; "
            f"modes are {list(EC_CONDITIONING_MODES)}"
        )
    return ec_conditioning


def refusal_reason(
    concept: str, arm: Arm, *, ec_conditioning: str = "native"
) -> str | None:
    """Why this arm cannot be measured on this concept, or ``None``.

    Refusals are part of the finding. A multi-residue BPE arm has no
    residue-to-token map, so a per-residue label cannot be addressed on it,
    though a sequence-level label - which is constant along the sequence - still
    can. An EC-conditioned arm under its native prompt is handed a functional tag
    that nearly determines any sequence-level function or family label, so such a
    probe would measure prompt copying; the two alternative conditioning modes
    remove that leak and are offered instead of a caveated number.
    """

    if concept not in CONCEPTS:
        raise KeyError(f"unknown concept {concept!r}; known concepts are {sorted(CONCEPTS)}")
    check_ec_conditioning(ec_conditioning)
    spec = CONCEPTS[concept]
    if not arm.supports("pathway"):
        # This module reads the residual stream at a block's output and writes an
        # erasing projection back into it. That is a sublayer-decomposition
        # assumption, and the panel withholds ``pathway`` from exactly the arms
        # for which it does not hold -- ByGPT5's T5 decoder, whose residual stream
        # is not the quantity the rest of the panel measures. Nothing else in this
        # module gated on it, because ``Arm.blocks()`` carries no gate the way
        # ``Arm.mlp()`` and ``Arm.attention()`` do, so such an arm would have run
        # all the way to a reliance figure. The refusal is returned rather than
        # raised so that it lands in the artefact beside the other refusals.
        return (
            f"{arm.name} ({arm.spec.architecture}) does not carry the 'pathway' "
            f"capability, so its residual stream is not the quantity the rest of the "
            f"panel decodes from and erases in; declared capabilities are "
            f"{sorted(arm.spec.capabilities)}"
        )
    if spec.modality != arm.modality:
        return (
            f"{arm.name} is a {arm.modality} arm and {concept} is a "
            f"{spec.modality} concept"
        )
    if arm.modality == "protein" and arm.spec.tokenisation != "residue":
        if spec.level in ("residue", "variant"):
            return (
                f"{arm.name} tokenises as {arm.spec.tokenisation!r}: a "
                "multi-residue BPE vocabulary has no residue-to-token map, so a "
                f"{spec.level}-level label such as {concept} cannot be aligned to "
                "its activations and must not be approximated by a "
                "token-to-residue heuristic. Sequence-level concepts are "
                "measured on this arm; residue-level ones are not."
            )
    if arm.spec.input_format != "ec_conditioned":
        return None
    if ec_conditioning == "native" and spec.level == "sequence":
        return (
            f"{arm.name} receives an EC tag in its own prompt, and on an enzyme "
            "cohort every sequence-level functional or family label is close to a "
            f"deterministic function of that tag. A probe for {concept} under "
            "native conditioning therefore measures how well the model copies its "
            "own prompt, not what it encodes from the sequence. Re-run this arm "
            "with --ec-conditioning fixed, which keeps the native input format but "
            "gives every prompt the same tag, or --ec-conditioning unconditioned, "
            "which drops the tag and takes the arm off its training distribution. "
            "Residue-level structure concepts are not affected by this leak: the "
            "tag constrains a protein's chemistry but does not say which residue "
            "at which position is helical or buried."
        )
    if ec_conditioning == "native" and concept == "fitness":
        return (
            f"{arm.name} requires an EC conditioning tag and ProteinGym assay "
            "targets carry none. Re-run with --ec-conditioning fixed or "
            "unconditioned, both of which define a prompt for an unlabelled "
            "target and record the resulting distribution shift."
        )
    return None


# ------------------------------------------------------------------ containers


@dataclass(frozen=True)
class Unit:
    """One forward pass and the positions read out of it.

    The token ids are carried explicitly rather than re-derived from the input
    string at read time: a decode/encode round trip is not guaranteed to be the
    identity for byte-level BPE, and a one-token drift would silently move every
    label off its activation.
    """

    unit_id: str
    input_string: str
    token_ids: tuple[int, ...]
    group: str
    positions: np.ndarray
    labels: Mapping[str, np.ndarray]
    pool_span: tuple[int, int]
    #: The arm-independent content this unit was built from -- the residue string
    #: for a protein unit, the document text for a text one. ``input_string`` is
    #: that content *rendered* for one arm, so hashing it makes the cohort digest
    #: arm-specific and the cross-arm identity check it exists for cannot fire.
    content: str = ""

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError(f"{self.unit_id}: a unit must declare its raw content")
        if self.positions.ndim != 1 or self.positions.size < 1:
            raise ValueError(f"{self.unit_id}: positions must be a non-empty vector")
        if not self.labels:
            raise ValueError(f"{self.unit_id}: at least one concept label is required")
        for concept, values in self.labels.items():
            if np.asarray(values).shape != self.positions.shape:
                raise ValueError(
                    f"{self.unit_id}: {concept} labels do not align with positions"
                )
        start, end = self.pool_span
        if not 0 <= start < end <= len(self.token_ids):
            raise ValueError(f"{self.unit_id}: invalid pooling span {self.pool_span}")
        if int(self.positions.min()) < 0 or int(self.positions.max()) >= len(self.token_ids):
            raise ValueError(f"{self.unit_id}: positions fall outside the token window")


@dataclass(frozen=True)
class SampleSet:
    """Aligned activations, labels and grouping for one concept."""

    concept: str
    task_type: str
    y: np.ndarray
    groups: np.ndarray
    unit_index: np.ndarray
    states: dict[int, np.ndarray]
    pooled: dict[int, np.ndarray]
    units: list[Unit]
    label_values: list[str] | None
    construction: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.y.shape[0]
        if self.groups.shape != (n,) or self.unit_index.shape != (n,):
            raise ValueError(f"{self.concept}: labels, groups and units must align")
        if set(self.states) != set(self.pooled):
            raise ValueError(f"{self.concept}: state and pooled layers disagree")
        for layer, matrix in self.states.items():
            if matrix.shape[0] != n or self.pooled[layer].shape != matrix.shape:
                raise ValueError(f"{self.concept}: layer {layer} matrix does not align")

    @property
    def n_samples(self) -> int:
        return int(self.y.shape[0])

    @property
    def layers(self) -> list[int]:
        return sorted(self.states)

    def cohort(self) -> Cohort:
        """The cohort's arm-independent content, for the identity digest.

        Hashing the *rendered* input strings made the digest a function of the
        arm: ProtGPT2's FASTA-wrapped rendering and ZymCTRL's EC-conditioned one
        of the same proteins hash differently, so two arms could never be shown
        to have seen the same cohort -- which is the only thing the digest is
        for. The raw content is hashed instead, together with the realised length
        band, and the resulting object is a well-formed ``Cohort`` that
        ``input_strings`` would render rather than double-render.
        """

        kind = "text" if CONCEPTS[self.concept].modality == "text" else "protein"
        records = [unit.content for unit in self.units]
        lengths = [len(record) for record in records]
        return Cohort(
            self.concept,
            kind,
            records,
            min(lengths) if lengths else 0,
            max(lengths) if lengths else 0,
            {},
        )

    def summary(self) -> dict[str, Any]:
        counts = Counter(str(value) for value in self.y)
        block: dict[str, Any] = {
            "n_units": len(self.units),
            "n_samples": self.n_samples,
            "n_groups": int(np.unique(self.groups).size),
            "grouping_variable": CONCEPTS[self.concept].grouping,
            "family_disjoint": CONCEPTS[self.concept].family_disjoint,
            "cohort_digest": self.cohort().digest,
            "construction": dict(self.construction),
        }
        if self.task_type == "classification":
            block["label_counts"] = {key: int(value) for key, value in counts.items()}
            block["n_labels"] = len(self.label_values or [])
        else:
            block["target"] = {
                "mean": float(np.mean(self.y)),
                "std": float(np.std(self.y)),
                "min": float(np.min(self.y)),
                "max": float(np.max(self.y)),
            }
        return block


# --------------------------------------------------------------- activations


@torch.no_grad()
def unit_states(
    arm: Arm, unit: Unit, *, layers: Sequence[int], max_tokens: int
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Per-position states and the unit's pooled state, for one forward pass.

    The last element of ``hidden_states`` is the state *after* the final
    normalisation, not the last block's output. The erasure hook in
    :func:`_install_hook` writes into the block output, so reading the top layer
    here would fit the eraser -- its mean, its whitening, its projection -- on the
    normalised cloud and then apply it to a stream whose norm is roughly twice as
    large. The result is not a projection of anything; it is an arbitrary affine
    displacement, and the reliance figure computed from it is a number with no
    referent. Refused rather than silently read: the same index also mixes
    representation types inside one table, since every other layer is pre-norm.
    """

    if any(not 0 <= layer < arm.n_layer for layer in layers):
        raise ValueError(f"{arm.name}: analysis layer outside [0, {arm.n_layer})")
    if arm.n_layer - 1 in layers:
        raise ValueError(
            f"{arm.name}: layer {arm.n_layer - 1} is the last block, whose raw output "
            "transformers does not expose in hidden_states -- the last entry is the "
            "post-final-normalisation state. Reading it here would fit an eraser on a "
            "normalised cloud and apply it to an un-normalised one. Use a layer "
            "fraction below 1.0"
        )
    ids = list(unit.token_ids)
    if len(ids) > max_tokens:
        raise ValueError(
            f"{unit.unit_id}: {len(ids)} tokens exceed the max_tokens={max_tokens} "
            "window; truncating here would move the read positions"
        )
    tensor = torch.tensor([ids], dtype=torch.long, device=arm.device)
    output = arm.model(input_ids=tensor, output_hidden_states=True)
    if len(output.hidden_states) != arm.n_layer + 1:
        raise RuntimeError(f"{arm.name}: expected {arm.n_layer + 1} hidden-state tensors")
    index = torch.tensor(unit.positions, dtype=torch.long, device=arm.device)
    start, end = unit.pool_span
    states: dict[int, np.ndarray] = {}
    pooled: dict[int, np.ndarray] = {}
    for layer in layers:
        hidden = output.hidden_states[layer + 1][0]
        states[layer] = hidden[index].float().cpu().numpy().astype(np.float32)
        pooled[layer] = (
            hidden[start:end].float().mean(dim=0).cpu().numpy().astype(np.float32)
        )
    return states, pooled


def collect_samples(
    arm: Arm,
    units: Sequence[Unit],
    *,
    layers: Sequence[int],
    max_tokens: int,
    construction: Mapping[str, Any],
) -> dict[str, SampleSet]:
    """Run one forward pass per unit and assemble one SampleSet per concept."""

    if not units:
        raise ValueError("at least one unit is required")
    concepts = list(units[0].labels)
    if any(list(unit.labels) != concepts for unit in units):
        raise ValueError("every unit must carry the same concept labels")
    per_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    per_layer_pooled: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    unit_index: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    labels: dict[str, list[np.ndarray]] = {concept: [] for concept in concepts}
    for index, unit in enumerate(units):
        states, pooled = unit_states(arm, unit, layers=layers, max_tokens=max_tokens)
        count = unit.positions.size
        for layer in layers:
            per_layer[layer].append(states[layer])
            per_layer_pooled[layer].append(np.repeat(pooled[layer][None, :], count, axis=0))
        unit_index.append(np.full(count, index, dtype=np.int64))
        groups.append(np.full(count, unit.group, dtype=object))
        for concept in concepts:
            labels[concept].append(np.asarray(unit.labels[concept]))
    stacked = {layer: np.concatenate(per_layer[layer]) for layer in layers}
    stacked_pooled = {
        layer: np.concatenate(per_layer_pooled[layer]) for layer in layers
    }
    group_vector = np.concatenate(groups).astype(str)
    unit_vector = np.concatenate(unit_index)
    result: dict[str, SampleSet] = {}
    for concept in concepts:
        y = np.concatenate(labels[concept])
        task_type = CONCEPTS[concept].task_type
        if task_type == "classification":
            y = y.astype(str)
            label_values = sorted(set(y.tolist()))
        else:
            y = y.astype(np.float64)
            label_values = None
        result[concept] = SampleSet(
            concept=concept,
            task_type=task_type,
            y=y,
            groups=group_vector,
            unit_index=unit_vector,
            states={layer: stacked[layer] for layer in layers},
            pooled={layer: stacked_pooled[layer] for layer in layers},
            units=list(units),
            label_values=label_values,
            construction=dict(construction),
        )
    return result


def restrict_labels(samples: SampleSet, *, min_groups_per_label: int) -> SampleSet:
    """Drop classes that cannot support a grouped split, and record the drop.

    A class present in only one or two groups cannot appear on both sides of a
    group-disjoint fold, so keeping it would silently convert the split into a
    record-level one for that class. Dropping is explicit and reported.
    """

    if samples.task_type != "classification":
        return samples
    if min_groups_per_label < 2:
        raise ValueError("min_groups_per_label must be at least two")
    keep_labels = sorted(
        {
            label
            for label in set(samples.y.tolist())
            if np.unique(samples.groups[samples.y == label]).size >= min_groups_per_label
        }
    )
    dropped = sorted(set(samples.y.tolist()) - set(keep_labels))
    if len(keep_labels) < 2:
        raise RuntimeError(
            f"{samples.concept}: only {len(keep_labels)} label(s) survive the "
            f"{min_groups_per_label}-group minimum; the cohort cannot support a "
            "grouped split for this concept"
        )
    if not dropped:
        return samples
    mask = np.isin(samples.y, keep_labels)
    construction = dict(samples.construction)
    construction["dropped_labels_below_group_minimum"] = dropped
    construction["min_groups_per_label"] = int(min_groups_per_label)
    return SampleSet(
        concept=samples.concept,
        task_type=samples.task_type,
        y=samples.y[mask],
        groups=samples.groups[mask],
        unit_index=samples.unit_index[mask],
        states={layer: matrix[mask] for layer, matrix in samples.states.items()},
        pooled={layer: matrix[mask] for layer, matrix in samples.pooled.items()},
        units=samples.units,
        label_values=keep_labels,
        construction=construction,
    )


# -------------------------------------------------------------- label sources


def ec_labels_by_accession() -> dict[str, str]:
    """EC conditioning tags keyed by UniProt accession."""

    labels: dict[str, str] = {}
    for header, _ in iter_fasta(ZYMCTRL_FASTA):
        if "|" not in header:
            raise ValueError(f"{ZYMCTRL_FASTA}: header {header!r} has no accession|EC form")
        accession, ec = header.split("|", 1)
        labels[accession] = ec
    if not labels:
        raise RuntimeError(f"{ZYMCTRL_FASTA}: no EC-labelled records")
    return labels


def swissprot_accession(header: str) -> str:
    fields = header.split("|")
    if len(fields) < 3 or fields[0] != "sp":
        raise ValueError(f"unexpected Swiss-Prot header {header!r}")
    return fields[1]


def single_pfam_family(spans: Sequence[tuple[int, int, str]]) -> str | None:
    """The one Pfam family of a protein, or ``None`` if it has several.

    Multi-family proteins make both the family label and the family grouping
    ambiguous, so they are excluded rather than assigned a dominant family by a
    tie-breaking rule that the reader cannot see.
    """

    families = {family for _, _, family in spans}
    return next(iter(families)) if len(families) == 1 else None


def render_input(
    arm: Arm,
    sequence: str,
    ec_label: str | None,
    *,
    ec_conditioning: str = "native",
) -> str:
    """Render one sequence in the arm's input format under a conditioning mode.

    ``native`` and ``fixed`` both go through ``Cohort.input_strings``, so the
    arm's own rendering rules stay the single source of truth; ``fixed`` only
    substitutes one constant tag for the protein's own. ``unconditioned`` is the
    one deliberate departure from the declared format and is only reachable for
    an EC-conditioned arm, where dropping the tag is the point.
    """

    check_ec_conditioning(ec_conditioning)
    if arm.spec.input_format == "ec_conditioned":
        if ec_conditioning == "unconditioned":
            return sequence
        ec_label = FIXED_EC_TAG if ec_conditioning == "fixed" else ec_label
        if ec_label is None:
            raise ValueError(f"{arm.name}: native conditioning needs an EC label")
    metadata = {"ec_labels": [ec_label]} if ec_label is not None else {}
    cohort = Cohort("render", "protein", [sequence], 0, 0, metadata)
    return cohort.input_strings(arm)[0]


def residue_window(
    arm: Arm, input_string: str, sequence: str, *, ec_conditioning: str = "native"
) -> tuple[tuple[int, ...], int, int]:
    """Token ids, the token index of residue 1, and the exclusive sequence end."""

    check_ec_conditioning(ec_conditioning)
    if arm.spec.input_format == "ec_conditioned" and ec_conditioning == "unconditioned":
        # The declared format is not what was rendered, so the offset cannot be
        # taken from the format table; it is verified token by token instead.
        require_residue_token_map(arm)
        ids = arm.tokenizer(input_string, return_tensors=None)["input_ids"]
        tokens = arm.tokenizer.convert_ids_to_tokens(ids)
        if len(tokens) < len(sequence) or any(
            a != b for a, b in zip(tokens[: len(sequence)], sequence)
        ):
            raise ValueError(
                f"{arm.name}: unconditioned rendering does not start with the "
                "residue sequence"
            )
        return tuple(ids), 0, len(sequence)
    ids, offset = residue_token_offset(arm, input_string, sequence)
    return tuple(ids), offset, offset + len(sequence)


def sequence_body_window(
    arm: Arm, input_string: str, sequence: str, *, ec_conditioning: str = "native"
) -> tuple[tuple[int, ...], int, int]:
    """Token ids and the token span covering the sequence body.

    A sequence-level label is constant along the sequence, so it needs only the
    span of tokens that spell the sequence out, not a residue-to-token map. For a
    multi-residue BPE arm that span is recovered from the rendered input and then
    *verified* by decoding it back to the exact sequence, so a rendering change
    upstream breaks the run instead of silently shifting every read position.
    """

    if arm.spec.tokenisation == "residue":
        return residue_window(
            arm, input_string, sequence, ec_conditioning=ec_conditioning
        )
    ids = arm.tokenizer(input_string, return_tensors=None)["input_ids"]
    end = len(ids)
    offset = 0
    while offset < end and not arm.tokenizer.decode([ids[offset]]).strip():
        offset += 1
    eos = arm.tokenizer.eos_token_id
    if eos is not None and ids[0] == eos:
        offset = max(offset, 1)
        while offset < end and not arm.tokenizer.decode([ids[offset]]).strip():
            offset += 1
    if offset >= end:
        raise ValueError(f"{arm.name}: rendered input carries no sequence tokens")
    decoded = "".join(arm.tokenizer.decode(ids[offset:end]).split())
    if decoded != sequence:
        raise ValueError(
            f"{arm.name}: token span [{offset}, {end}) decodes to "
            f"{len(decoded)} residues, not the {len(sequence)} of the sequence"
        )
    return tuple(ids), offset, end


# ------------------------------------------------------------ protein builders


def structure_units(
    arm: Arm,
    *,
    n_proteins: int,
    scan_models: int,
    min_len: int,
    max_len: int,
    min_plddt: float,
    residues_per_protein: int,
    seed: int,
    kmer: int,
    kmer_jaccard: float,
    pfam_jaccard: float,
    ec_conditioning: str = "native",
) -> tuple[list[Unit], dict[str, Any]]:
    """AlphaFold-derived per-residue secondary structure and burial units."""

    require_residue_token_map(arm)
    check_ec_conditioning(ec_conditioning)
    conditioned = (
        arm.spec.input_format == "ec_conditioned" and ec_conditioning == "native"
    )
    ec_labels = ec_labels_by_accession() if conditioned else {}
    catalogue = alphafold_models(ALPHAFOLD_ROOT)
    order = np.random.default_rng(seed).permutation(len(catalogue))[:scan_models]
    generator = np.random.default_rng(seed + 1)
    excluded = {
        "non_canonical_residues": 0,
        "length_out_of_range": 0,
        "no_ec_label": 0,
        "too_few_confident_residues": 0,
    }
    selected: list[tuple[Structure, np.ndarray, np.ndarray, np.ndarray]] = []
    examined = 0
    for index in order:
        structure = read_alphafold_model(catalogue[int(index)])
        examined += 1
        if structure.n_non_canonical_residues > 0:
            excluded["non_canonical_residues"] += 1
            continue
        if not min_len <= len(structure) <= max_len:
            excluded["length_out_of_range"] += 1
            continue
        if conditioned and structure.accession not in ec_labels:
            excluded["no_ec_label"] += 1
            continue
        confident = np.flatnonzero(structure.plddt >= min_plddt)
        if confident.size < residues_per_protein:
            excluded["too_few_confident_residues"] += 1
            continue
        chosen = np.sort(
            generator.choice(confident, size=residues_per_protein, replace=False)
        )
        secondary = ca_secondary_structure(structure.ca)[chosen]
        contacts = contact_number_bins(
            structure.ca, radius=CONTACT_RADIUS_ANGSTROM, bin_width=1, n_bins=127
        )[chosen]
        selected.append((structure, chosen, secondary, contacts))
        if len(selected) >= n_proteins:
            break
    if len(selected) < n_proteins:
        raise RuntimeError(
            f"{arm.name}: only {len(selected)}/{n_proteins} AlphaFold models usable "
            f"out of {examined} examined; exclusions {excluded}"
        )

    accessions = [structure.accession for structure, _, _, _ in selected]
    sequences = [structure.sequence for structure, _, _, _ in selected]
    spans = load_pfam_spans(PFAM_RESIDUE_TSV, accessions=set(accessions))
    pfam_by_accession = {
        accession: {family for _, _, family in entries}
        for accession, entries in spans.items()
    }
    clusters, homology = homology_clusters(
        accessions,
        sequences,
        pfam_by_accession=pfam_by_accession,
        kmer=kmer,
        kmer_jaccard_threshold=kmer_jaccard,
        pfam_jaccard_threshold=pfam_jaccard,
    )

    units: list[Unit] = []
    for cluster, (structure, chosen, secondary, contacts) in zip(clusters, selected):
        ec = ec_labels[structure.accession] if conditioned else None
        input_string = render_input(
            arm, structure.sequence, ec, ec_conditioning=ec_conditioning
        )
        ids, offset, end = residue_window(
            arm, input_string, structure.sequence, ec_conditioning=ec_conditioning
        )
        burial = np.digitize(contacts, BURIAL_BANDS)
        units.append(
            Unit(
                unit_id=structure.accession,
                input_string=input_string,
                content=structure.sequence,
                token_ids=ids,
                group=f"homology_cluster_{int(cluster)}",
                positions=chosen.astype(np.int64) + offset,
                labels={
                    "ss3": np.asarray(
                        [SECONDARY_STRUCTURE_NAMES[int(value)] for value in secondary]
                    ),
                    "burial": np.asarray([BURIAL_NAMES[int(value)] for value in burial]),
                },
                pool_span=(offset, end),
            )
        )
    construction = {
        "source": "alphafold_ca_traces",
        "n_models_examined": examined,
        "model_scan_budget": int(scan_models),
        "model_selection": "seeded_permutation_of_the_full_alphafold_model_set",
        "min_len": int(min_len),
        "max_len": int(max_len),
        "min_plddt": float(min_plddt),
        "residues_per_protein": int(residues_per_protein),
        "excluded": excluded,
        "homology": homology,
        "ec_conditioning": ec_conditioning,
        "secondary_structure_caveat": SECONDARY_STRUCTURE_CAVEAT,
        "burial_caveat": BURIAL_CAVEAT,
        "burial_contact_bands": list(BURIAL_BANDS),
        "burial_contact_radius_angstrom": CONTACT_RADIUS_ANGSTROM,
    }
    return units, construction


def sample_body_positions(
    arm: Arm,
    offset: int,
    end: int,
    *,
    count: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Distinct token positions inside the sequence body.

    One probe sample is one token on every arm, so drawing the same ``count`` on
    a residue-level arm and on a multi-residue BPE arm matches the probe
    training sets on tokens rather than on sequences. A sequence whose body is
    shorter than ``count`` is a design error, not something to sample with
    replacement.
    """

    width = end - offset
    if count < 1 or width < count:
        raise ValueError(
            f"{arm.name}: {width} body tokens cannot supply {count} distinct read "
            "positions; lower the positions per sequence or raise the length band"
        )
    return np.sort(generator.choice(width, size=count, replace=False)).astype(
        np.int64
    ) + offset


#: How a unit builder chooses which eligible records enter the cohort.
RECORD_SELECTION_MODES = ("seeded_permutation", "file_order")


def record_order(total: int, *, seed: int, mode: str = "seeded_permutation") -> list[int]:
    """The order in which eligible records are considered, by declared mode.

    Swiss-Prot is sorted by entry name, so consecutive records are orthologues;
    the EC-labelled corpus is blocked by EC number; ProteinGym is one file per
    assay and each file is ordered by mutation position. In every case a
    file-order prefix is a maximally redundant draw of exactly the thing the
    grouping variable exists to defend against, and the loss lands on the number
    of independent groups, which is what the splits and the group bootstrap are
    clustered on.

    ``file_order`` is retained so a frozen artefact can be reproduced, and the
    mode is written into every ``construction`` record, so the two can never be
    confused for one another after the fact.
    """

    if mode not in RECORD_SELECTION_MODES:
        raise ValueError(
            f"unknown record selection {mode!r}; known {RECORD_SELECTION_MODES}"
        )
    if total < 0:
        raise ValueError("record count must be non-negative")
    if mode == "file_order":
        return list(range(total))
    return [int(index) for index in np.random.default_rng(seed).permutation(total)]


def ec_units(
    arm: Arm,
    *,
    n_proteins: int,
    min_len: int,
    max_len: int,
    positions_per_protein: int,
    max_per_family: int,
    seed: int,
    ec_conditioning: str = "native",
    record_selection: str = "seeded_permutation",
) -> tuple[list[Unit], dict[str, Any]]:
    """EC-labelled Swiss-Prot proteins grouped by their single Pfam family.

    ``record_selection`` decides which eligible proteins enter, and defaults to a
    seeded permutation; see :func:`record_order`.
    """

    refused = refusal_reason("ec_class", arm, ec_conditioning=ec_conditioning)
    if refused is not None:
        raise ValueError(f"{arm.name}: {refused}")
    spans = load_pfam_spans(PFAM_RESIDUE_TSV)
    allowed = set(AA20)
    generator = np.random.default_rng(seed)
    per_family: Counter[str] = Counter()
    units: list[Unit] = []
    excluded = {
        "length_out_of_range": 0,
        "non_canonical_residues": 0,
        "no_single_pfam_family": 0,
        "family_quota_reached": 0,
    }
    examined = 0
    # Two passes. The EC-labelled FASTA is blocked by EC number -- its first eight
    # records share one -- so filling the per-family quota in file order takes a
    # taxonomically and functionally clustered prefix. Measured on the full
    # corpus at a quota of four, file order yields 135 distinct grouping units
    # against 231 under a seeded permutation of the same 400 proteins, with the
    # EC-1 share more than doubled. Both the group count and the class prior are
    # what the splits and the group bootstrap are clustered on, so this is a 40%
    # loss of effective sample size and a different label distribution, neither
    # of which appeared anywhere in the artefact.
    eligible: list[tuple[str, str, str, str]] = []
    for header, body in iter_fasta(ZYMCTRL_FASTA):
        if "<start>" not in body or "<end>" not in body:
            continue
        examined += 1
        sequence = body.split("<start>")[1].split("<end>")[0]
        if not min_len <= len(sequence) <= max_len:
            excluded["length_out_of_range"] += 1
            continue
        if not set(sequence) <= allowed:
            excluded["non_canonical_residues"] += 1
            continue
        accession, ec = header.split("|", 1)
        family = single_pfam_family(spans.get(accession, []))
        if family is None:
            excluded["no_single_pfam_family"] += 1
            continue
        eligible.append((accession, ec, sequence, family))
    for position in record_order(len(eligible), seed=seed, mode=record_selection):
        accession, ec, sequence, family = eligible[position]
        if per_family[family] >= max_per_family:
            excluded["family_quota_reached"] += 1
            continue
        per_family[family] += 1
        input_string = render_input(
            arm, sequence, ec, ec_conditioning=ec_conditioning
        )
        ids, offset, end = sequence_body_window(
            arm, input_string, sequence, ec_conditioning=ec_conditioning
        )
        chosen = sample_body_positions(
            arm, offset, end, count=positions_per_protein, generator=generator
        )
        label = f"EC{ec.split('.')[0]}"
        units.append(
            Unit(
                unit_id=accession,
                input_string=input_string,
                content=sequence,
                token_ids=ids,
                group=family,
                positions=chosen,
                labels={"ec_class": np.asarray([label] * positions_per_protein)},
                pool_span=(offset, end),
            )
        )
        if len(units) >= n_proteins:
            break
    if len(units) < n_proteins:
        raise RuntimeError(
            f"{arm.name}: only {len(units)}/{n_proteins} EC-labelled proteins with a "
            f"single Pfam family out of {examined} examined; exclusions {excluded}"
        )
    construction = {
        "source": str(ZYMCTRL_FASTA),
        "n_records_examined": examined,
        "min_len": int(min_len),
        "max_len": int(max_len),
        "positions_per_protein": int(positions_per_protein),
        "max_proteins_per_family": int(max_per_family),
        "excluded": excluded,
        "label_definition": "top_level_ec_digit",
        "ec_conditioning": ec_conditioning,
        "read_positions": "uniform_sample_of_the_sequence_body_tokens",
        "record_selection": record_selection,
        "n_eligible_records": len(eligible),
    }
    return units, construction


def pfam_units(
    arm: Arm,
    *,
    n_families: int,
    proteins_per_family: int,
    min_len: int,
    max_len: int,
    positions_per_protein: int,
    redundancy_kmer: int,
    redundancy_jaccard: float,
    seed: int,
    ec_conditioning: str = "native",
    record_selection: str = "seeded_permutation",
) -> tuple[list[Unit], dict[str, Any]]:
    """Single-family Swiss-Prot proteins grouped by within-family redundancy.

    ``record_selection`` decides which members of each ranked family enter, and
    defaults to a seeded permutation; see :func:`record_order`. Which *families*
    are ranked is deliberately still deterministic -- the largest families, ties
    broken by name -- because the ranking is over the whole corpus rather than
    over a prefix of it.
    """

    refused = refusal_reason("pfam_family", arm, ec_conditioning=ec_conditioning)
    if refused is not None:
        raise ValueError(f"{arm.name}: {refused}")
    spans = load_pfam_spans(PFAM_RESIDUE_TSV)
    allowed = set(AA20)
    single: dict[str, str] = {}
    for accession, entries in spans.items():
        family = single_pfam_family(entries)
        if family is not None:
            single[accession] = family

    candidates: dict[str, list[tuple[str, str]]] = {}
    examined = 0
    excluded = {
        "length_out_of_range": 0,
        "non_canonical_residues": 0,
        "no_single_pfam_family": 0,
    }
    for header, sequence in iter_fasta(SWISSPROT_FASTA):
        examined += 1
        if not min_len <= len(sequence) <= max_len:
            excluded["length_out_of_range"] += 1
            continue
        if not set(sequence) <= allowed:
            excluded["non_canonical_residues"] += 1
            continue
        accession = swissprot_accession(header)
        family = single.get(accession)
        if family is None:
            excluded["no_single_pfam_family"] += 1
            continue
        candidates.setdefault(family, []).append((accession, sequence))
    ranked = sorted(
        (family for family, rows in candidates.items() if len(rows) >= proteins_per_family),
        key=lambda family: (-len(candidates[family]), family),
    )[:n_families]
    if len(ranked) < n_families:
        raise RuntimeError(
            f"{arm.name}: only {len(ranked)}/{n_families} Pfam families have "
            f"{proteins_per_family} eligible single-family proteins; exclusions {excluded}"
        )

    generator = np.random.default_rng(seed)
    units: list[Unit] = []
    cluster_counts: dict[str, int] = {}
    for family in ranked:
        # Swiss-Prot is ordered by entry name, so the first N members of a family
        # are orthologues of one another. Taking that prefix collapses the
        # within-family redundancy clusters this cohort is grouped on, which is
        # the one defence the pfam_family concept has against family-level
        # memorisation.
        order = record_order(len(candidates[family]), seed=seed, mode=record_selection)
        rows = [candidates[family][index] for index in order[:proteins_per_family]]
        accessions = [accession for accession, _ in rows]
        sequences = [sequence for _, sequence in rows]
        clusters, _ = homology_clusters(
            accessions,
            sequences,
            pfam_by_accession={accession: set() for accession in accessions},
            kmer=redundancy_kmer,
            kmer_jaccard_threshold=redundancy_jaccard,
            pfam_jaccard_threshold=1.0,
        )
        cluster_counts[family] = int(np.unique(clusters).size)
        for cluster, (accession, sequence) in zip(clusters, rows):
            input_string = render_input(
                arm, sequence, None, ec_conditioning=ec_conditioning
            )
            ids, offset, end = sequence_body_window(
                arm, input_string, sequence, ec_conditioning=ec_conditioning
            )
            chosen = sample_body_positions(
                arm, offset, end, count=positions_per_protein, generator=generator
            )
            units.append(
                Unit(
                    unit_id=accession,
                    input_string=input_string,
                    content=sequence,
                    token_ids=ids,
                    group=f"{family}_redundancy_{int(cluster)}",
                    positions=chosen,
                    labels={
                        "pfam_family": np.asarray([family] * positions_per_protein)
                    },
                    pool_span=(offset, end),
                )
            )
    construction = {
        "source": str(SWISSPROT_FASTA),
        "n_records_examined": examined,
        "n_families": int(n_families),
        "proteins_per_family": int(proteins_per_family),
        "positions_per_protein": int(positions_per_protein),
        "min_len": int(min_len),
        "max_len": int(max_len),
        "excluded": excluded,
        "families": list(ranked),
        "redundancy_clusters_per_family": cluster_counts,
        "redundancy_kmer": int(redundancy_kmer),
        "redundancy_kmer_jaccard": float(redundancy_jaccard),
        "ec_conditioning": ec_conditioning,
        "read_positions": "uniform_sample_of_the_sequence_body_tokens",
        "record_selection": record_selection,
        "grouping_caveat": CONCEPTS["pfam_family"].rationale,
    }
    return units, construction


def fitness_units(
    arm: Arm,
    *,
    n_assays: int,
    variants_per_assay: int,
    min_len: int,
    max_len: int,
    seed: int,
    ec_conditioning: str = "native",
    record_selection: str = "seeded_permutation",
) -> tuple[list[Unit], dict[str, Any]]:
    """ProteinGym single substitutions read at the mutated residue.

    ``record_selection`` decides which assay files are considered, and defaults
    to a seeded permutation; see :func:`record_order`. Alphabetical assay order
    groups assays on the same protein together -- ``BLAT_ECOLX`` alone has four
    files, all of the same 286-residue protein -- so a prefix of it can put four
    "groups" that are one protein into the design, and an assay-disjoint fold
    then trains and tests on variants of the same wild type.
    """

    require_residue_token_map(arm)
    refused = refusal_reason("fitness", arm, ec_conditioning=ec_conditioning)
    if refused is not None:
        raise ValueError(f"{arm.name}: {refused}")
    require_input_path(PROTEINGYM_ROOT, "TRANSFER_PROTEINGYM_DIR")
    catalogue = sorted(PROTEINGYM_ROOT.glob("*.csv"))
    if not catalogue:
        raise RuntimeError(f"no ProteinGym assay files under {PROTEINGYM_ROOT}")
    paths = [
        catalogue[index]
        for index in record_order(len(catalogue), seed=seed, mode=record_selection)
    ]
    allowed = set(AA20)
    generator = np.random.default_rng(seed)
    units: list[Unit] = []
    assays: list[dict[str, Any]] = []
    # ProteinGym ships several assays per protein. ``group`` is the assay, so an
    # assay-disjoint split is not a protein-disjoint one; admitting two assays on
    # one wild type would let a fold train and test on sequences differing at a
    # single residue. Refusing the second is what makes CONCEPTS["fitness"].
    # family_disjoint true rather than merely asserted.
    proteins_seen: set[str] = set()
    excluded = {
        "length_out_of_range": 0,
        "too_few_single_mutants": 0,
        "protein_already_represented": 0,
    }
    for path in paths:
        if len(assays) >= n_assays:
            break
        # ProteinGym stems are ``<UniProt-ID>_<first-author>_<year>``; the target
        # protein is the part before the second underscore.
        target_protein = "_".join(path.stem.split("_")[:2])
        if target_protein in proteins_seen:
            excluded["protein_already_represented"] += 1
            continue
        # The length filter is applied from the first record so that a
        # multi-thousand-residue assay is skipped without being parsed.
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            first = next(reader, None)
            if first is None:
                raise ValueError(f"{path}: empty assay file")
            if not min_len <= len(first["mutated_sequence"]) <= max_len:
                excluded["length_out_of_range"] += 1
                continue
            rows = [first, *reader]
        eligible = [
            row
            for row in rows
            if ":" not in row["mutant"]
            and set(row["mutated_sequence"]) <= allowed
            and len(row["mutated_sequence"]) == len(rows[0]["mutated_sequence"])
        ]
        if len(eligible) < variants_per_assay:
            excluded["too_few_single_mutants"] += 1
            continue
        picked = [
            eligible[int(index)]
            for index in np.sort(
                generator.choice(len(eligible), size=variants_per_assay, replace=False)
            )
        ]
        scores = np.asarray([float(row["DMS_score"]) for row in picked])
        if np.ptp(scores) == 0.0:
            raise ValueError(f"{path}: sampled DMS scores do not vary")
        # Ranks inside the assay remove between-assay scale, which is not a
        # property of the protein and would otherwise dominate a pooled
        # correlation across assay-disjoint folds.
        ranks = stats.rankdata(scores) / (scores.size + 1.0)
        assay = path.stem
        proteins_seen.add(target_protein)
        for row, rank in zip(picked, ranks):
            sequence = row["mutated_sequence"]
            position = int(row["mutant"][1:-1])
            if not 1 <= position <= len(sequence):
                raise ValueError(f"{path}: mutant {row['mutant']!r} is outside the sequence")
            # The read position is derived from the mutant string; if the
            # numbering is against a different sequence than the one shipped --
            # a full-length UniProt entry against a domain-restricted
            # ``mutated_sequence``, say -- every variant would be read at the
            # wrong residue with nothing to show for it.
            if sequence[position - 1] != row["mutant"][-1]:
                raise ValueError(
                    f"{path}: mutant {row['mutant']!r} does not match "
                    f"{sequence[position - 1]!r} at position {position} of its own "
                    "mutated_sequence; the assay numbering is not against this sequence"
                )
            input_string = render_input(
                arm, sequence, None, ec_conditioning=ec_conditioning
            )
            ids, offset, end = residue_window(
                arm, input_string, sequence, ec_conditioning=ec_conditioning
            )
            units.append(
                Unit(
                    unit_id=f"{assay}:{row['mutant']}",
                    input_string=input_string,
                    content=sequence,
                    token_ids=ids,
                    group=assay,
                    positions=np.asarray([offset + position - 1], dtype=np.int64),
                    labels={"fitness": np.asarray([float(rank)])},
                    pool_span=(offset, end),
                )
            )
        assays.append(
            {
                "assay": assay,
                "sequence_length": len(rows[0]["mutated_sequence"]),
                "n_eligible_single_mutants": len(eligible),
            }
        )
    if len(assays) < n_assays:
        raise RuntimeError(
            f"{arm.name}: only {len(assays)}/{n_assays} ProteinGym assays satisfy "
            f"the length band [{min_len}, {max_len}] and the "
            f"{variants_per_assay}-variant minimum; exclusions {excluded}"
        )
    construction = {
        "source": str(PROTEINGYM_ROOT),
        "n_assays": len(assays),
        "variants_per_assay": int(variants_per_assay),
        "min_len": int(min_len),
        "max_len": int(max_len),
        "excluded": excluded,
        "assays": assays,
        "target_transform": "within_assay_rank_normalised_dms_score",
        "read_position": "mutated_residue",
        "ec_conditioning": ec_conditioning,
        "record_selection": record_selection,
        "n_assay_files_available": len(catalogue),
        "target_proteins": sorted(proteins_seen),
        "one_assay_per_target_protein": True,
    }
    return units, construction


# --------------------------------------------------------------- text builders


def surface_class(token_text: str) -> str:
    """Deterministic surface class of one decoded token."""

    core = token_text.lstrip(" ")
    if not core.strip():
        return TEXT_CLASS_NAMES[3]
    first = core[0]
    if first.isdigit():
        return TEXT_CLASS_NAMES[4]
    if not first.isalpha():
        return TEXT_CLASS_NAMES[3]
    if not token_text.startswith(" "):
        return TEXT_CLASS_NAMES[2]
    return TEXT_CLASS_NAMES[1] if first.isupper() else TEXT_CLASS_NAMES[0]


def text_units(
    arm: Arm,
    *,
    n_documents: int,
    positions_per_document: int,
    max_tokens: int,
    min_chars: int,
    seed: int,
) -> tuple[list[Unit], dict[str, Any]]:
    """Next-token class and next-token rarity units from OpenWebText documents."""

    if arm.modality != "text":
        raise ValueError(f"{arm.name} is a {arm.modality} arm; this cohort is text")
    cohort = text_cohort(n_documents, min_chars=min_chars, name="openwebtext_probe")
    generator = np.random.default_rng(seed)
    tokenised: list[list[int]] = []
    for document in cohort.records:
        ids = arm.tokenizer(document, return_tensors=None)["input_ids"][:max_tokens]
        if len(ids) < positions_per_document + 2:
            raise RuntimeError(
                f"{arm.name}: a cohort document yields only {len(ids)} tokens, below "
                f"the {positions_per_document + 2} needed at max_tokens={max_tokens}"
            )
        tokenised.append(ids)
    if len(tokenised) != len(cohort.records):
        raise RuntimeError("tokenised documents do not align with the cohort")
    counts = Counter(token for ids in tokenised for token in ids)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    total = sum(counts.values())
    band_by_token: dict[int, str] = {}
    cumulative = 0
    for token, count in ordered:
        band = RARITY_NAMES[min(2, int(3 * cumulative / total))]
        band_by_token[token] = band
        cumulative += count

    decoded = {
        token: arm.tokenizer.decode([token]) for token in counts
    }
    units: list[Unit] = []
    for index, (ids, document) in enumerate(zip(tokenised, cohort.records)):
        chosen = np.sort(
            generator.choice(len(ids) - 1, size=positions_per_document, replace=False)
        )
        following = [ids[int(position) + 1] for position in chosen]
        units.append(
            Unit(
                unit_id=f"document_{index}",
                input_string=arm.tokenizer.decode(ids),
                content=document,
                token_ids=tuple(ids),
                group=f"document_{index}",
                positions=chosen.astype(np.int64),
                labels={
                    "next_token_class": np.asarray(
                        [surface_class(decoded[token]) for token in following]
                    ),
                    "next_token_rarity": np.asarray(
                        [band_by_token[token] for token in following]
                    ),
                },
                pool_span=(0, len(ids)),
            )
        )
    construction = {
        "source": "openwebtext_screen_plain_text",
        "n_documents": int(n_documents),
        "positions_per_document": int(positions_per_document),
        "max_tokens": int(max_tokens),
        "min_document_chars": int(min_chars),
        "cohort_digest": cohort.digest,
        "rarity_definition": "cohort_empirical_next_token_frequency_terciles_by_mass",
        "class_definition": "surface_class_of_the_decoded_next_token",
        "analogy_caveat": (
            "next-token class and rarity are matched-difficulty analogues of "
            "secondary structure and burial, not the same concepts; a text/protein "
            "difference in probe skill is not a difference in encoded structure"
        ),
    }
    return units, construction


# ------------------------------------------------------------------- erasure


@dataclass(frozen=True)
class Eraser:
    """An affine map that deletes a subspace from an activation."""

    method: str
    mean: np.ndarray
    projection: np.ndarray
    rank: int
    covariance_rank: int
    removed_variance_fraction: float
    mean_relative_displacement: float

    def apply(self, x: np.ndarray) -> np.ndarray:
        matrix = np.asarray(x, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.mean.size:
            raise ValueError("eraser applied to a matrix of the wrong width")
        return (matrix - self.mean) @ self.projection.T + self.mean

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "erased_rank": int(self.rank),
            "covariance_rank": int(self.covariance_rank),
            "removed_variance_fraction": float(self.removed_variance_fraction),
            "mean_relative_displacement": float(self.mean_relative_displacement),
            "d_model": int(self.mean.size),
        }


def _whitening(
    covariance: np.ndarray, *, tolerance: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Symmetric inverse-square-root and square-root of a covariance matrix."""

    values, vectors = np.linalg.eigh(covariance)
    if values.max() <= 0.0:
        raise ValueError("covariance matrix has no positive eigenvalue")
    keep = values > tolerance * values.max()
    retained = vectors[:, keep]
    roots = np.sqrt(values[keep])
    inverse_sqrt = (retained / roots) @ retained.T
    sqrt = (retained * roots) @ retained.T
    return inverse_sqrt, sqrt, int(keep.sum())


def _relative_displacement(
    projection: np.ndarray, centred: np.ndarray
) -> float:
    """Mean ``||r(x) - x||`` in units of the mean ``||x - mean||``.

    An oblique projection can move an activation much further than it shortens
    it, and an intervention that lands far outside the data cloud is not a
    matched control however well its rank or removed variance matches. This is
    the number that makes that visible.
    """

    scale = float(np.linalg.norm(centred, axis=1).mean())
    if scale <= 0.0:
        raise ValueError("activations carry no spread around their mean")
    moved = centred @ projection.T - centred
    return float(np.linalg.norm(moved, axis=1).mean()) / scale


def _removed_variance(projection: np.ndarray, covariance: np.ndarray) -> float:
    total = float(np.trace(covariance))
    if total <= 0.0:
        raise ValueError("covariance matrix carries no variance")
    kept = float(np.trace(projection @ covariance @ projection.T))
    return 1.0 - kept / total


def fit_leace(x: np.ndarray, z: np.ndarray, *, tolerance: float = 1e-8) -> Eraser:
    """Fit the LEACE eraser for concept ``z`` on activations ``x``.

    Closed form of Belrose et al. (arXiv:2306.03819): with ``W`` the whitening
    transform of ``Cov(x)``, the eraser is ``I - W^+ P W`` where ``P`` projects
    onto the column space of ``W Cov(x, z)``. The result zeroes ``Cov(r(x), z)``
    exactly on the fitting sample, which is what makes the post-erasure linear
    probe a verification rather than a hope, while changing ``x`` as little as
    possible in the least-squares sense.
    """

    matrix = np.asarray(x, dtype=np.float64)
    concept = np.asarray(z, dtype=np.float64)
    if matrix.ndim != 2 or concept.ndim != 2 or matrix.shape[0] != concept.shape[0]:
        raise ValueError("x and z must be aligned two-dimensional arrays")
    if matrix.shape[0] < matrix.shape[1] // 4:
        raise ValueError(
            f"fitting LEACE on {matrix.shape[0]} samples in {matrix.shape[1]} "
            "dimensions is too under-determined for the erasure to hold out of "
            "sample; enlarge the fitting split"
        )
    if not np.isfinite(matrix).all() or not np.isfinite(concept).all():
        raise ValueError("x and z must be finite")
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    concept_centred = concept - concept.mean(axis=0)
    denominator = matrix.shape[0] - 1
    covariance = centred.T @ centred / denominator
    cross = centred.T @ concept_centred / denominator
    inverse_sqrt, sqrt, covariance_rank = _whitening(covariance, tolerance=tolerance)
    whitened_cross = inverse_sqrt @ cross
    left, singular, _ = np.linalg.svd(whitened_cross, full_matrices=False)
    if singular.size == 0 or singular.max() <= 0.0:
        raise ValueError("the concept has no covariance with the activations")
    rank = int((singular > tolerance * singular.max()).sum())
    basis = left[:, :rank]
    projection = np.eye(matrix.shape[1]) - sqrt @ basis @ basis.T @ inverse_sqrt
    return Eraser(
        method=ERASURE_METHOD,
        mean=mean,
        projection=projection,
        rank=rank,
        covariance_rank=covariance_rank,
        removed_variance_fraction=_removed_variance(projection, covariance),
        mean_relative_displacement=_relative_displacement(projection, centred),
    )


def fit_random_eraser(
    x: np.ndarray, *, rank: int, seed: int, geometry: str, tolerance: float = 1e-8
) -> Eraser:
    """Fit a concept-free eraser of the same rank as a LEACE eraser.

    Deleting ``k`` directions from a residual stream costs cross-entropy whether
    or not the model uses them, so a reliance claim is only the excess over this
    control. Two geometries are offered because the control has to be matched on
    something: ``raw_orthonormal`` deletes a uniformly random orthonormal
    ``k``-subspace of the activation space, and ``whitened_orthonormal`` deletes a
    random subspace of the *whitened* space, which is the space LEACE itself
    picks its subspace from and therefore the tighter control.
    """

    matrix = np.asarray(x, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("x must be a two-dimensional array")
    if rank < 1 or rank >= matrix.shape[1]:
        raise ValueError("rank must lie strictly between zero and the model width")
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    covariance = centred.T @ centred / (matrix.shape[0] - 1)
    generator = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(generator.normal(size=(matrix.shape[1], rank)))
    inverse_sqrt, sqrt, covariance_rank = _whitening(covariance, tolerance=tolerance)
    if geometry == "raw_orthonormal":
        projection = np.eye(matrix.shape[1]) - basis @ basis.T
    elif geometry == "whitened_orthonormal":
        projection = np.eye(matrix.shape[1]) - sqrt @ basis @ basis.T @ inverse_sqrt
    else:
        raise ValueError(f"unknown random-control geometry {geometry!r}")
    return Eraser(
        method=f"random_{geometry}",
        mean=mean,
        projection=projection,
        rank=int(rank),
        covariance_rank=covariance_rank,
        removed_variance_fraction=_removed_variance(projection, covariance),
        mean_relative_displacement=_relative_displacement(projection, centred),
    )


def fit_variance_matched_eraser(
    x: np.ndarray,
    *,
    rank: int,
    target_removed_variance: float,
    seed: int,
    tolerance: float = 1e-8,
    max_relative_error: float = 0.01,
    bisection_steps: int = 80,
) -> Eraser:
    """A concept-free subspace matched on rank *and* on removed variance.

    A rank-matched random subspace of a residual stream removes roughly ``k/d``
    of the variance, while a LEACE subspace lands on concept-carrying - and
    therefore typically high-variance - directions and removes an order of
    magnitude more. Charging the erasure against a control that deletes far less
    of the representation would flatter it.

    The removed variance of a ``k``-dimensional subspace varies continuously as
    the subspace rotates, from the ``k`` smallest eigenvalues to the ``k``
    largest. A random subspace is therefore rotated towards the top or bottom
    eigen-subspace and bisected until it removes exactly as much variance as the
    LEACE subspace does. The result is concept-agnostic, exactly rank-matched
    and exactly variance-matched, and is not aligned to the principal axes.
    """

    matrix = np.asarray(x, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("x must be a two-dimensional array")
    if rank < 1 or rank >= matrix.shape[1]:
        raise ValueError("rank must lie strictly between zero and the model width")
    if not 0.0 < target_removed_variance < 1.0:
        raise ValueError("target removed variance must lie strictly in (0, 1)")
    if bisection_steps < 8:
        raise ValueError("bisection needs at least eight steps")
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    covariance = centred.T @ centred / (matrix.shape[0] - 1)
    values, vectors = np.linalg.eigh(covariance)
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("covariance matrix carries no variance")
    attainable = (
        float(values[:rank].sum()) / total,
        float(values[-rank:].sum()) / total,
    )
    if not attainable[0] <= target_removed_variance <= attainable[1]:
        raise ValueError(
            f"no rank-{rank} subspace can remove {target_removed_variance:.6f} of "
            f"the variance; attainable range is {attainable}"
        )

    def removed(basis: np.ndarray) -> float:
        return float(np.trace(basis.T @ covariance @ basis)) / total

    generator = np.random.default_rng(seed)
    start, _ = np.linalg.qr(generator.normal(size=(matrix.shape[1], rank)))
    anchor = (
        vectors[:, -rank:]
        if removed(start) < target_removed_variance
        else vectors[:, :rank]
    )

    def blend(step: float) -> np.ndarray:
        mixture = (1.0 - step) * start + step * anchor
        if float(np.linalg.svd(mixture, compute_uv=False).min()) < 1e-9:
            raise RuntimeError(
                "the interpolated control subspace lost rank; refit with another seed"
            )
        basis, _ = np.linalg.qr(mixture)
        return basis

    low, high = 0.0, 1.0
    low_sign = removed(start) - target_removed_variance
    basis = anchor
    for _ in range(bisection_steps):
        middle = 0.5 * (low + high)
        basis = blend(middle)
        if (removed(basis) - target_removed_variance) * low_sign > 0.0:
            low = middle
        else:
            high = middle
    projection = np.eye(matrix.shape[1]) - basis @ basis.T
    achieved = _removed_variance(projection, covariance)
    relative_error = abs(achieved - target_removed_variance) / target_removed_variance
    if relative_error > max_relative_error:
        raise RuntimeError(
            f"variance-matched control removed {achieved:.6f} against a target of "
            f"{target_removed_variance:.6f}; the control is not matched and must "
            "not be reported as one"
        )
    _, _, covariance_rank = _whitening(covariance, tolerance=tolerance)
    return Eraser(
        method="variance_matched_random",
        mean=mean,
        projection=projection,
        rank=int(rank),
        covariance_rank=covariance_rank,
        removed_variance_fraction=achieved,
        mean_relative_displacement=_relative_displacement(projection, centred),
    )


def mean_ablation_eraser(x: np.ndarray) -> Eraser:
    """The everything-removed reference: replace the activation by its mean."""

    matrix = np.asarray(x, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("x must be a two-dimensional array")
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    covariance = centred.T @ centred / (matrix.shape[0] - 1)
    _, _, covariance_rank = _whitening(covariance, tolerance=1e-8)
    zero = np.zeros((matrix.shape[1], matrix.shape[1]))
    return Eraser(
        method="mean_ablation",
        mean=mean,
        projection=zero,
        rank=int(matrix.shape[1]),
        covariance_rank=covariance_rank,
        removed_variance_fraction=1.0,
        mean_relative_displacement=_relative_displacement(zero, centred),
    )


def concept_matrix(
    y: np.ndarray, *, task_type: str, label_values: Sequence[str] | None
) -> np.ndarray:
    """The concept indicator matrix LEACE erases."""

    if task_type == "classification":
        if label_values is None:
            raise ValueError("classification requires a frozen label list")
        return np.stack([(y == label).astype(np.float64) for label in label_values], axis=1)
    return np.asarray(y, dtype=np.float64)[:, None]


# ------------------------------------------------------------------- probing


def encode_labels(y: np.ndarray, label_values: Sequence[str]) -> np.ndarray:
    """Integer codes for a frozen, sorted label list.

    The estimators are fitted on codes rather than on the readable label
    strings: sklearn's early-stopping path scores its validation split with a
    numeric predicate and cannot take string predictions. The readable labels
    survive in the report, and the codes are a bijection onto them.
    """

    values = np.asarray(label_values)
    if values.size < 2 or np.any(values[:-1] >= values[1:]):
        raise ValueError("label_values must be a sorted list of at least two labels")
    codes = np.searchsorted(values, y)
    if np.any(values[codes] != y):
        raise ValueError("y contains a label outside the frozen label list")
    return codes.astype(np.int64)


def frozen_metric(
    task_type: str, label_values: Sequence[int] | None
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Macro-F1 over a frozen label set, or Spearman rho."""

    if task_type == "classification":
        labels = list(label_values or [])
        if len(labels) < 2:
            raise ValueError("classification needs at least two frozen labels")

        def macro_f1(y_true: np.ndarray, prediction: np.ndarray) -> float:
            return float(
                f1_score(
                    y_true, prediction, labels=labels, average="macro", zero_division=0
                )
            )

        return macro_f1

    def spearman(y_true: np.ndarray, prediction: np.ndarray) -> float:
        if np.ptp(y_true) == 0.0 or np.ptp(prediction) == 0.0:
            return float("nan")
        return float(stats.spearmanr(y_true, prediction).statistic)

    return spearman


def fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    estimator: str,
    task_type: str,
    seed: int,
    probe_dim: int,
    mlp_hidden: tuple[int, ...],
    mlp_max_iter: int,
) -> np.ndarray:
    """Fit one probe inside a training fold and predict the held-out fold.

    The pipeline is: standardise, project onto a fold-fitted PCA of one common dimension, then
    fit. The PCA is what makes a 1280-wide and a 1536-wide arm comparable at
    equal probe capacity, and it is fitted inside the fold so the held-out side
    stays unseen.
    """

    maximum = min(x_train.shape[0] - 1, x_train.shape[1])
    if probe_dim < 2 or probe_dim > maximum:
        raise ValueError(
            f"probe dimension {probe_dim} is infeasible for a training matrix of "
            f"shape {x_train.shape}; the maximum centred rank is {maximum}"
        )
    if estimator == "chance":
        if task_type == "classification":
            model = DummyClassifier(strategy="stratified", random_state=seed)
            model.fit(np.zeros((x_train.shape[0], 1)), y_train)
            return np.asarray(model.predict(np.zeros((x_test.shape[0], 1))))
        return np.asarray(
            np.random.default_rng(seed).choice(y_train, size=x_test.shape[0], replace=True)
        )
    if estimator == "linear":
        head = (
            LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=3000, random_state=seed
            )
            if task_type == "classification"
            else Ridge(alpha=1.0)
        )
    elif estimator == "mlp":
        head = (
            MLPClassifier(
                hidden_layer_sizes=mlp_hidden,
                alpha=1e-3,
                max_iter=mlp_max_iter,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=seed,
            )
            if task_type == "classification"
            else MLPRegressor(
                hidden_layer_sizes=mlp_hidden,
                alpha=1e-3,
                max_iter=mlp_max_iter,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=seed,
            )
        )
    else:
        raise ValueError(f"unknown estimator {estimator!r}")
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=probe_dim, svd_solver="randomized", random_state=seed),
        head,
    )
    pipeline.fit(x_train, y_train)
    return np.asarray(pipeline.predict(x_test))


def _derived_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def skill_block(
    y: np.ndarray,
    probe_predictions: np.ndarray,
    chance_predictions: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    """Chance-corrected skill with a grouped paired bootstrap interval."""

    chance_score = float(metric(y, chance_predictions))
    if 1.0 - chance_score <= 1e-6:
        raise RuntimeError(
            "the chance model already scores one; the skill normalisation is undefined"
        )

    def chance_corrected_skill(probe_score: float, baseline_score: float) -> float:
        headroom = 1.0 - baseline_score
        if headroom <= 1e-6:
            return float("nan")
        return (probe_score - baseline_score) / headroom

    paired = paired_group_bootstrap(
        y,
        probe_predictions,
        chance_predictions,
        groups,
        metric,
        seed=seed,
        n_bootstrap=n_bootstrap,
        derived_statistic=chance_corrected_skill,
    )
    return {
        "score": paired["left_score"],
        "chance_score": paired["right_score"],
        "skill": paired["derived_score"],
        "skill_ci95": paired["derived_ci95"],
        "raw_difference": paired["difference"],
        "raw_difference_ci95": paired["difference_ci95"],
        "n_groups": paired["n_groups"],
        "n_bootstrap": paired["n_bootstrap"],
    }


def evaluate_probes(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    task_type: str,
    label_values: Sequence[str] | None,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    seed: int,
    probe_dim: int,
    n_bootstrap: int,
    mlp_hidden: tuple[int, ...],
    mlp_max_iter: int,
    eraser_factory: Callable[[np.ndarray], Eraser] | None = None,
) -> dict[str, Any]:
    """Out-of-fold linear, MLP and chance predictions on one representation.

    ``eraser_factory`` is called with the training indices of each fold; the
    eraser it returns is fitted on training rows only and applied to both sides,
    which is what makes the post-erasure number a held-out verification rather
    than a restatement of the closed form.
    """

    if task_type == "classification":
        if label_values is None:
            raise ValueError("classification requires a frozen label list")
        target = encode_labels(y, label_values)
        metric = frozen_metric(task_type, list(range(len(label_values))))
    else:
        target = np.asarray(y, dtype=np.float64)
        metric = frozen_metric(task_type, None)
    names = ("linear", "mlp", "chance")
    predictions = {
        name: np.empty(target.shape[0], dtype=target.dtype) for name in names
    }
    erasers: list[Eraser] = []
    for fold, (train, test) in enumerate(splits):
        if eraser_factory is None:
            x_train, x_test = x[train], x[test]
        else:
            eraser = eraser_factory(train)
            erasers.append(eraser)
            x_train = eraser.apply(x[train])
            x_test = eraser.apply(x[test])
        for name in names:
            predictions[name][test] = fit_probe(
                x_train,
                target[train],
                x_test,
                estimator=name,
                task_type=task_type,
                seed=_derived_seed(seed, fold, name),
                probe_dim=probe_dim,
                mlp_hidden=mlp_hidden,
                mlp_max_iter=mlp_max_iter,
            )
    block: dict[str, Any] = {
        probe: skill_block(
            target,
            predictions[probe],
            predictions["chance"],
            groups,
            metric,
            seed=_derived_seed(seed, "bootstrap", probe),
            n_bootstrap=n_bootstrap,
        )
        for probe in ("linear", "mlp")
    }
    block["chance_model"] = CHANCE_MODEL
    block["n_folds"] = len(splits)
    if erasers:
        block["erasers"] = [eraser.summary() for eraser in erasers]
    return block


# ---------------------------------------------------------- forward intervention


def torch_transform(
    eraser: Eraser, *, device: str
) -> Callable[[torch.Tensor], torch.Tensor]:
    """The eraser as a residual-stream edit, applied at every token position."""

    projection = torch.tensor(eraser.projection, device=device, dtype=torch.float32)
    mean = torch.tensor(eraser.mean, device=device, dtype=torch.float32)

    def transform(hidden: torch.Tensor) -> torch.Tensor:
        original = hidden.dtype
        centred = hidden.to(torch.float32) - mean
        return (centred @ projection.T + mean).to(original)

    return transform


def _install_hook(
    arm: Arm, layer: int, transform: Callable[[torch.Tensor], torch.Tensor]
) -> RemovableHandle:
    """Edit the residual stream leaving block ``layer`` for one forward pass."""

    block = arm.blocks()[layer]

    def hook(module: torch.nn.Module, inputs: tuple, output: object) -> object:
        if isinstance(output, tuple):
            return (transform(output[0]),) + tuple(output[1:])
        if isinstance(output, torch.Tensor):
            return transform(output)
        raise TypeError(
            f"{arm.name}: block {layer} returned {type(output)!r}, which this "
            "intervention cannot edit"
        )

    return block.register_forward_hook(hook)


@torch.no_grad()
def intervention_rows(
    arm: Arm,
    inputs: Sequence[str],
    *,
    layer: int,
    transforms: Mapping[str, Callable[[torch.Tensor], torch.Tensor]],
    max_tokens: int,
    batch_size: int,
    ec_conditioning: str = "native",
) -> dict[str, list[dict[str, float | int]]]:
    """Per-sequence next-token cross-entropy and KL for every intervention mode."""

    if not inputs:
        raise ValueError("at least one input sequence is required")
    if not 0 <= layer < arm.n_layer:
        raise ValueError(f"{arm.name}: intervention layer outside [0, {arm.n_layer})")
    check_ec_conditioning(ec_conditioning)
    # The rule and the boundary ids are derived from the same two inputs, so an
    # unconditioned prompt -- which carries no <start>/<end> -- gets the plain
    # rule and no ids, and a conditioned one gets the boundary rule and both.
    rule = target_rule(arm.spec.input_format, ec_conditioning=ec_conditioning)
    start_id, end_id = conditioning_boundary_ids(arm, ec_conditioning=ec_conditioning)
    rows: dict[str, list[dict[str, float | int]]] = {name: [] for name in transforms}
    for begin in range(0, len(inputs), batch_size):
        batch = list(inputs[begin : begin + batch_size])
        ids, mask = tokenize_batch(arm, batch, max_tokens)
        ids = ids.to(arm.device)
        mask = mask.to(arm.device)
        target_mask = sequence_target_mask(
            ids,
            mask,
            rule=rule,
            start_token_id=start_id,
            end_token_id=end_id,
        )
        clean = arm.model(input_ids=ids, attention_mask=mask).logits
        for name, transform in transforms.items():
            handle = _install_hook(arm, layer, transform)
            try:
                variant = arm.model(input_ids=ids, attention_mask=mask).logits
            finally:
                # Without this, a failed forward leaves the residual-stream edit
                # installed on the block for the rest of the process, and every
                # later measurement in the same run is silently intervened on.
                handle.remove()
            rows[name].extend(per_sequence_scores(clean, variant, ids, target_mask))
    return rows


def stratified_unit_draw(
    candidates: Sequence[int], *, groups: Sequence[Any], limit: int, seed: int
) -> list[int]:
    """Up to ``limit`` unit indices, spread over the groups they belong to.

    Units are built group block by group block, so a prefix of any unit list is a
    prefix of one group. Drawing round-robin over the groups in a seeded order,
    with each group's own members in a seeded order, means a cap smaller than the
    candidate set still spans the design rather than collapsing onto its first
    cell. Returned sorted, so the cohort's order is a property of the sample and
    not of the draw.
    """

    if limit < 1:
        raise ValueError("the unit cap must be positive")
    if len(candidates) != len(groups):
        raise ValueError("candidate units and their groups must align")
    if not candidates:
        return []
    generator = np.random.default_rng(seed)
    by_group: dict[Any, list[int]] = {}
    for unit, group in zip(candidates, groups):
        by_group.setdefault(group, []).append(int(unit))
    order = list(by_group)
    for members in by_group.values():
        generator.shuffle(members)
    generator.shuffle(order)
    drawn: list[int] = []
    depth = 0
    while len(drawn) < limit:
        added = False
        for group in order:
            members = by_group[group]
            if depth < len(members):
                drawn.append(members[depth])
                added = True
                if len(drawn) >= limit:
                    break
        if not added:
            break
        depth += 1
    return sorted(drawn)


def sequence_bootstrap(
    values: Sequence[float],
    weights: Sequence[int],
    *,
    seed: int,
    n_bootstrap: int,
    groups: Sequence[Any] | None = None,
) -> list[float]:
    """Token-weighted mean interval, resampling whole clusters.

    Without ``groups`` the cluster is the scored sequence, which is right for a
    cohort of distinct proteins. It is *not* right for the fitness cohort: its
    units are single substitutions of one wild type, differing at one residue
    each, so forty of them are one sequence measured forty times and an interval
    over rows would be narrow for a reason that has nothing to do with sampling.
    Passing ``groups`` resamples the group -- the protein, or the assay -- and
    keeps every one of its rows, which is the design that was actually drawn.
    """

    value_array = np.asarray(values, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if value_array.shape != weight_array.shape or value_array.size < 2:
        raise ValueError("values and weights must align and cover at least two sequences")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if groups is None:
        members = [np.asarray([index]) for index in range(value_array.size)]
    else:
        if len(groups) != value_array.size:
            raise ValueError("groups must align with the scored values")
        labels = list(dict.fromkeys(groups))
        members = [
            np.asarray([i for i, group in enumerate(groups) if group == label])
            for label in labels
        ]
        if len(members) < 2:
            raise ValueError(
                "a cluster bootstrap needs at least two groups; with one group the "
                "interval would describe a single protein measured many times"
            )
    generator = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_bootstrap):
        picked = generator.integers(0, len(members), size=len(members))
        index = np.concatenate([members[int(choice)] for choice in picked])
        draws.append(
            float(
                (value_array[index] * weight_array[index]).sum()
                / weight_array[index].sum()
            )
        )
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def behaviour_block(
    rows_by_mode: Mapping[str, Sequence[Mapping[str, float | int]]],
    *,
    target_mode: str,
    reference_mode: str,
    control_modes: Sequence[str],
    primary_control: str,
    minimum_ce_denominator: float,
    seed: int,
    n_bootstrap: int,
    scored_groups: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Absolute nats per token, a mean-ablation reference and the control excess.

    ``scored_groups`` is one group identifier per scored sequence and is the
    cluster the excess interval is resampled on. Supply it whenever the scored
    sequences are not independent -- for the fitness cohort they are single
    substitutions of a shared wild type, and a row-level interval there is narrow
    for the wrong reason.

    A ratio alone is uninterpretable: it hides both the size of the effect and
    the size of the denominator. Every mode is therefore reported in absolute
    nats per token first, the mean-ablation reference sits next to it as the
    everything-removed scale, and the reliance claim is the excess of the
    concept erasure over a matched control - reported against every control, so
    that a reader applying the rank-only definition and a reader applying the
    rank-and-variance definition both get their number.
    """

    if target_mode not in rows_by_mode or reference_mode not in rows_by_mode:
        raise ValueError("target and reference modes must both be measured")
    if primary_control not in control_modes or not set(control_modes) <= set(rows_by_mode):
        raise ValueError("every control mode must be measured, including the primary")
    aggregates = {name: aggregate_variant(rows) for name, rows in rows_by_mode.items()}
    clean = {value["clean_ce_nats"] for value in aggregates.values()}
    if max(clean) - min(clean) > 1e-9:
        raise RuntimeError("the clean reference changed between intervention modes")
    clean_ce = float(next(iter(clean)))
    modes = {
        name: {
            "ce_nats": value["variant_ce_nats"],
            "ce_delta_nats": value["variant_ce_nats"] - clean_ce,
            "kl_nats": value["clean_to_variant_kl_nats"],
            "argmax_agreement": value["argmax_agreement"],
        }
        for name, value in aggregates.items()
    }
    denominator = modes[reference_mode]["ce_delta_nats"]
    denominator_valid = denominator >= minimum_ce_denominator
    target_rows = rows_by_mode[target_mode]
    excess: dict[str, Any] = {}
    for control in control_modes:
        control_rows = rows_by_mode[control]
        if len(target_rows) != len(control_rows):
            raise RuntimeError("intervention modes scored different sequence sets")
        weights = [int(row["token_count"]) for row in target_rows]
        per_sequence = [
            (float(a["variant_nll_sum"]) - float(b["variant_nll_sum"]))
            / int(a["token_count"])
            for a, b in zip(target_rows, control_rows)
        ]
        excess[control] = {
            "ce_excess_nats": modes[target_mode]["ce_delta_nats"]
            - modes[control]["ce_delta_nats"],
            "ce_excess_ci95": sequence_bootstrap(
                per_sequence,
                weights,
                seed=_derived_seed(seed, "excess", control),
                n_bootstrap=n_bootstrap,
                groups=scored_groups,
            ),
            "kl_excess_nats": modes[target_mode]["kl_nats"] - modes[control]["kl_nats"],
            "is_primary_control": control == primary_control,
        }
    return {
        "clean_ce_nats": clean_ce,
        "scored_tokens": int(aggregates[reference_mode]["scored_tokens"]),
        "scored_sequences": int(aggregates[reference_mode]["sequences"]),
        "modes": modes,
        "target_mode": target_mode,
        "reference_mode": reference_mode,
        "control_modes": list(control_modes),
        "primary_control": primary_control,
        "control_interpretation": CONTROL_INTERPRETATION,
        "excess_interval_cluster_unit": (
            "scored_sequence" if scored_groups is None else "unit_group"
        ),
        "excess_interval_n_clusters": (
            len(target_rows)
            if scored_groups is None
            else len(dict.fromkeys(scored_groups))
        ),
        "mean_ablation_ce_delta_nats": denominator,
        "mean_ablation_kl_nats": modes[reference_mode]["kl_nats"],
        "minimum_ce_denominator": float(minimum_ce_denominator),
        "denominator_valid": bool(denominator_valid),
        "fraction_of_mean_ablation_ce": (
            modes[target_mode]["ce_delta_nats"] / denominator
            if denominator_valid
            else None
        ),
        "excess_over_control": excess,
        "primary_excess_ce_nats": excess[primary_control]["ce_excess_nats"],
        "primary_excess_ce_ci95": excess[primary_control]["ce_excess_ci95"],
    }


# ------------------------------------------------------------------- assembly


def grouped_splits(
    samples: SampleSet, *, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group-disjoint folds, refusing to continue if the grouping is unusable."""

    n_groups = int(np.unique(samples.groups).size)
    if n_groups < n_splits:
        raise RuntimeError(
            f"{samples.concept}: {n_groups} groups of "
            f"{CONCEPTS[samples.concept].grouping} cannot support {n_splits} "
            "group-disjoint folds; enlarge the cohort rather than splitting by record"
        )
    return make_group_splits(
        samples.y,
        samples.groups,
        n_splits=n_splits,
        seed=seed,
        task_type=samples.task_type,
    )


def erasure_report(
    arm: Arm,
    samples: SampleSet,
    *,
    layer: int,
    layer_fraction: float,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    clean_probe: Mapping[str, Any],
    seed: int,
    probe_dim: int,
    n_bootstrap: int,
    mlp_hidden: tuple[int, ...],
    mlp_max_iter: int,
    max_post_erasure_skill: float,
    min_clean_skill: float,
    minimum_ce_denominator: float,
    max_tokens: int,
    batch_size: int,
    max_scored_units: int,
    ec_conditioning: str,
) -> dict[str, Any]:
    """Verify the erasure, then measure what removing the concept costs the model."""

    x = samples.states[layer].astype(np.float64)
    z = concept_matrix(
        samples.y, task_type=samples.task_type, label_values=samples.label_values
    )
    common = {
        "task_type": samples.task_type,
        "label_values": samples.label_values,
        "splits": splits,
        "seed": seed,
        "probe_dim": probe_dim,
        "n_bootstrap": n_bootstrap,
        "mlp_hidden": mlp_hidden,
        "mlp_max_iter": mlp_max_iter,
    }
    leace_probe = evaluate_probes(
        x,
        samples.y,
        samples.groups,
        **common,
        eraser_factory=lambda train: fit_leace(x[train], z[train]),
    )

    def matched_control(train: np.ndarray) -> Eraser:
        """The fold's variance-matched, concept-agnostic deletion.

        Fitting LEACE again inside the factory is what keeps the control matched
        fold by fold: the rank and the removed-variance target both come from
        the eraser this fold's verification is being compared against.
        """

        reference = fit_leace(x[train], z[train])
        return fit_variance_matched_eraser(
            x[train],
            rank=reference.rank,
            target_removed_variance=reference.removed_variance_fraction,
            seed=_derived_seed(seed, "control", layer),
        )

    control_probe = evaluate_probes(
        x, samples.y, samples.groups, **common, eraser_factory=matched_control
    )

    # The gate is one-sided. Erasure has to remove skill, and a probe that ends
    # up below the prior-matched chance model - which happens when it collapses
    # onto one class and macro-F1 punishes it more than random guessing - has
    # certainly not retained the concept.
    observed = leace_probe["linear"]["skill"]
    passed = observed <= max_post_erasure_skill
    verification = {
        "gate": {
            "max_post_erasure_linear_skill": float(max_post_erasure_skill),
            "observed_post_erasure_linear_skill": leace_probe["linear"]["skill"],
            "observed_post_erasure_linear_skill_ci95": leace_probe["linear"]["skill_ci95"],
            "one_sided": True,
            "passed": bool(passed),
            "min_clean_linear_skill_for_an_informative_gate": float(min_clean_skill),
            "clean_linear_skill": clean_probe["linear"]["skill"],
            "informative": bool(clean_probe["linear"]["skill"] >= min_clean_skill),
            "scope": (
                "LEACE guarantees linear guardedness only; the post-erasure MLP is "
                "reported as a diagnostic and is not gated"
            ),
        },
        "clean": dict(clean_probe),
        "leace": leace_probe,
        "variance_matched_control": control_probe,
    }
    if not passed:
        raise RuntimeError(
            f"{arm.name}/{samples.concept}: post-erasure linear skill "
            f"{leace_probe['linear']['skill']:.4f} exceeds the "
            f"{max_post_erasure_skill} verification gate at layer {layer}; the "
            "erasure is not working and must be fixed rather than reported"
        )

    train, test = splits[0]
    fit_eraser = fit_leace(x[train], z[train])
    control_eraser = fit_random_eraser(
        x[train],
        rank=fit_eraser.rank,
        seed=_derived_seed(seed, "behaviour-control", layer),
        geometry="whitened_orthonormal",
    )
    raw_control = fit_random_eraser(
        x[train],
        rank=fit_eraser.rank,
        seed=_derived_seed(seed, "behaviour-raw-control", layer),
        geometry="raw_orthonormal",
    )
    variance_control = fit_variance_matched_eraser(
        x[train],
        rank=fit_eraser.rank,
        target_removed_variance=fit_eraser.removed_variance_fraction,
        seed=_derived_seed(seed, "behaviour-variance-control", layer),
    )
    reference = mean_ablation_eraser(x[train])
    # The behaviour cohort used to be the *first* ``max_scored_units`` held-out
    # unit indices. Units are appended group block by group block -- assay by
    # assay for fitness, family by family for pfam -- so a prefix of a test fold
    # is one group, and for fitness that is forty single mutants of one protein
    # at its N-terminal-most sampled positions. Every ce_delta and the headline
    # excess were then measured on that. The draw is now spread over the fold's
    # groups round-robin under the run's own seed, so a cap of forty takes forty
    # units from as many groups as the fold has.
    candidates = sorted({int(index) for index in samples.unit_index[test]})
    held_out_units = stratified_unit_draw(
        candidates,
        groups=[samples.units[index].group for index in candidates],
        limit=max_scored_units,
        seed=_derived_seed(seed, "behaviour-cohort", layer),
    )
    if len(held_out_units) < 2:
        raise RuntimeError(
            f"{samples.concept}: fewer than two held-out units to score the "
            "intervention on"
        )
    inputs = [samples.units[index].input_string for index in held_out_units]
    # The resampling unit for the interval below. Two mutants of one protein are
    # not two sequences, and for fitness every scored unit can come from one wild
    # type, so resampling rows would give a nominal-95% interval with no coverage.
    held_out_groups = [samples.units[index].group for index in held_out_units]
    erasers = {
        "leace": fit_eraser,
        "variance_matched_random": variance_control,
        "random_whitened_orthonormal": control_eraser,
        "random_raw_orthonormal": raw_control,
        "mean_ablation": reference,
    }
    rows = intervention_rows(
        arm,
        inputs,
        layer=layer,
        transforms={
            name: torch_transform(eraser, device=arm.device)
            for name, eraser in erasers.items()
        },
        max_tokens=max_tokens,
        batch_size=batch_size,
        ec_conditioning=ec_conditioning,
    )
    behaviour = behaviour_block(
        rows,
        target_mode="leace",
        reference_mode="mean_ablation",
        control_modes=(
            "variance_matched_random",
            "random_whitened_orthonormal",
            "random_raw_orthonormal",
        ),
        primary_control=PRIMARY_CONTROL,
        minimum_ce_denominator=minimum_ce_denominator,
        seed=_derived_seed(seed, "behaviour", layer),
        n_bootstrap=n_bootstrap,
        scored_groups=held_out_groups,
    )
    return {
        "layer": int(layer),
        "layer_fraction": float(layer_fraction),
        "method": ERASURE_METHOD,
        "citation": ERASURE_CITATION,
        "fit_split": "grouped_fold_0_training_side",
        "ec_conditioning": ec_conditioning,
        "fit_representation": "per_position_residual_stream_at_the_read_positions",
        "applied_representation": "every_token_position_in_the_forward_pass",
        "erasers": {name: eraser.summary() for name, eraser in erasers.items()},
        "control_matching": {
            name: {
                "removed_variance_ratio": (
                    erasers[name].removed_variance_fraction
                    / fit_eraser.removed_variance_fraction
                ),
                "displacement_ratio": (
                    erasers[name].mean_relative_displacement
                    / fit_eraser.mean_relative_displacement
                ),
                "displacement_tolerance": float(CONTROL_DISPLACEMENT_TOLERANCE),
                # ``behaviour_block`` already decided whether the mean-ablation
                # reference is a usable denominator, and then this dict divided by
                # it regardless. Zero raises; a small positive value gives a
                # meaningless share; a *negative* one -- which ZymCTRL produces
                # unconditioned, per the audit's own record -- flips the ``<``
                # below, so a control costing more than deleting the whole layer
                # would be stamped as a matched cost. That boolean is the flag a
                # reader uses to decide whether the excess licenses anything.
                "mean_ablation_denominator_valid": bool(
                    behaviour["denominator_valid"]
                ),
                "mean_ablation_share_of_cost": (
                    behaviour["modes"][name]["ce_delta_nats"]
                    / behaviour["mean_ablation_ce_delta_nats"]
                    if behaviour["denominator_valid"]
                    else None
                ),
                "cost_is_a_matched_cost": (
                    bool(
                        erasers[name].mean_relative_displacement
                        <= CONTROL_DISPLACEMENT_TOLERANCE
                        * fit_eraser.mean_relative_displacement
                        and behaviour["modes"][name]["ce_delta_nats"]
                        < CONTROL_MEAN_ABLATION_SHARE
                        * behaviour["mean_ablation_ce_delta_nats"]
                    )
                    if behaviour["denominator_valid"]
                    else None
                ),
                "matched_cost_criteria": (
                    "displaces activations no more than "
                    f"{CONTROL_DISPLACEMENT_TOLERANCE}x the erasure does, and "
                    f"costs less than {CONTROL_MEAN_ABLATION_SHARE} of the "
                    "mean-ablation reference. A control that costs as much as "
                    "deleting the layer is not a floor for anything."
                ),
                "interpretation": CONTROL_INTERPRETATION[name],
            }
            for name in CONTROL_INTERPRETATION
        },
        "verification": verification,
        "behaviour": behaviour,
        "scored_units": len(held_out_units),
        "max_scored_units": int(max_scored_units),
    }


def concept_report(
    arm: Arm,
    samples: SampleSet,
    *,
    layers: Sequence[int],
    layer_fractions: Sequence[float],
    erasure_layer: int,
    erasure_fraction: float,
    n_splits: int,
    seed: int,
    probe_dim: int,
    n_bootstrap: int,
    mlp_hidden: tuple[int, ...],
    mlp_max_iter: int,
    max_post_erasure_skill: float,
    min_clean_skill: float,
    minimum_ce_denominator: float,
    max_tokens: int,
    batch_size: int,
    max_scored_units: int,
    ec_conditioning: str,
) -> dict[str, Any]:
    """Decodability at every layer of the grid, then erasure at one layer."""

    splits = grouped_splits(samples, n_splits=n_splits, seed=seed)
    common = {
        "task_type": samples.task_type,
        "label_values": samples.label_values,
        "splits": splits,
        "seed": seed,
        "probe_dim": probe_dim,
        "n_bootstrap": n_bootstrap,
        "mlp_hidden": mlp_hidden,
        "mlp_max_iter": mlp_max_iter,
    }
    decodability = {
        "per_position": {
            str(layer): evaluate_probes(
                samples.states[layer].astype(np.float64),
                samples.y,
                samples.groups,
                **common,
            )
            for layer in layers
        },
        "mean_pooled_sequence": {
            str(layer): evaluate_probes(
                samples.pooled[layer].astype(np.float64),
                samples.y,
                samples.groups,
                **common,
            )
            for layer in layers
        },
    }
    spec = CONCEPTS[samples.concept]
    return {
        "spec": {
            "name": spec.name,
            "modality": spec.modality,
            "level": spec.level,
            "task_type": spec.task_type,
            "metric": spec.metric,
            "grouping_variable": spec.grouping,
            "family_disjoint": spec.family_disjoint,
            "label_source": spec.label_source,
            "rationale": spec.rationale,
        },
        "cohort": samples.summary(),
        "layer_grid": {
            "layers": [int(layer) for layer in layers],
            "fractions": [float(value) for value in layer_fractions],
        },
        "decodability": decodability,
        "erasure": erasure_report(
            arm,
            samples,
            layer=erasure_layer,
            layer_fraction=erasure_fraction,
            splits=splits,
            clean_probe=decodability["per_position"][str(erasure_layer)],
            seed=seed,
            probe_dim=probe_dim,
            n_bootstrap=n_bootstrap,
            mlp_hidden=mlp_hidden,
            mlp_max_iter=mlp_max_iter,
            max_post_erasure_skill=max_post_erasure_skill,
            min_clean_skill=min_clean_skill,
            minimum_ce_denominator=minimum_ce_denominator,
            max_tokens=max_tokens,
            batch_size=batch_size,
            max_scored_units=max_scored_units,
            ec_conditioning=ec_conditioning,
        ),
    }


def erasure_layer_for(arm: Arm, fraction: float) -> int:
    """The single absolute layer that one relative depth names on this arm.

    Delegates to :func:`src.transfer.scoring.analysis_layer`, which is the
    panel's one depth convention. This function carried a second one --
    ``int(round(fraction * (n_layer - 1)))``, round-half-to-even -- until
    EXP-R2-067. The two disagree wherever ``fraction * (n_layer - 1)`` is an
    exact half: on ``progen2-base`` and ``progen2-medium`` at depth 0.25
    (``0.25 * 26 = 6.5``) round-half-to-even gives layer 6 and the panel
    convention gives layer 7, and on ``bygpt5-base-en`` at depth 0.5.

    Stage 09 reached both conventions in one invocation -- the probe grid
    through :func:`analysis_layer_grid` and the erasure layer through here --
    so on the two ProGen2 arms, which are the protein-side corpus contrast,
    the erasure depth and the probe depth of the same name resolved to
    different layers. At the stage's default ``--erasure-fraction 0.5`` the two
    agree on every campaign arm, so no recorded number moved; away from that
    default the mismatch surfaced as the probe-grid membership check refusing a
    valid depth and blaming the operator.
    """

    return analysis_layer(arm.n_layer, fraction)


def combined_digest(sample_sets: Mapping[str, SampleSet]) -> str:
    """One digest over every concept cohort this arm was measured on."""

    if not sample_sets:
        raise ValueError("at least one sample set is required")
    payload = "|".join(
        f"{concept}:{sample_sets[concept].cohort().digest}"
        for concept in sorted(sample_sets)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def token_budget(sample_sets: Mapping[str, SampleSet]) -> dict[str, int]:
    """Forward-pass token count per concept, so budgets can be compared."""

    return {
        concept: int(sum(len(unit.token_ids) for unit in samples.units))
        for concept, samples in sample_sets.items()
    }


def analysis_layer_grid(arm: Arm, fractions: Sequence[float]) -> list[int]:
    """The relative-depth grid, resolved to this arm's absolute layers.

    Adds one check to :func:`src.transfer.scoring.analysis_layers`: this stage
    reports a *per-depth* row, so two depths landing on the same layer would
    produce two rows of the same measurement under different labels.
    """

    layers = analysis_layers(arm.n_layer, fractions)
    if len(layers) != len(set(fractions)):
        raise ValueError(
            f"{arm.name}: relative depths {list(fractions)} collapse onto "
            f"{layers} on a {arm.n_layer}-layer model; choose distinct depths"
        )
    return layers


def format_skill(block: Mapping[str, Any]) -> str:
    return (
        f"{block['score']:.4f} (chance {block['chance_score']:.4f}, skill "
        f"{block['skill']:+.4f} [{block['skill_ci95'][0]:+.4f}, "
        f"{block['skill_ci95'][1]:+.4f}])"
    )
