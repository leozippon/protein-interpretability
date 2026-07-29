#!/usr/bin/env python3
"""One declaration of which arms each campaign stage may run, and why not.

Before this file existed the answer was spread over five places that could
disagree with each other and with :data:`src.transfer.arms.PANEL`:

* ``run_transfer_h200.sh`` held a hand-written ``KNOWN_ARMS`` string;
* ``h200_worker.sh`` held a *second* copy of it, plus a hand-written modality
  enumeration, a hand-written lens-arm exclusion and a hand-written
  relational-arm inclusion;
* ``08_lens_family.py`` derived its own capability-filtered default;
* ``02_pathway_budget.py`` and ``03_estimand_power.py`` derived no filter at all
  and defaulted to ``sorted(PANEL)``, which includes three arms with no
  ``pathway`` capability;
* ``11_induction_path_patching.py`` checked the ``circuits`` capability but not
  the module layout ``src.transfer.path_patching`` actually requires.

Every one of those is a place where a stage's *panel* -- the set of arms a
number is computed over -- could change without anything downstream looking
wrong. That is the failure mode L18 records: an environment default narrowed a
nine-stage campaign's text side to one model while every downstream number
remained well-formed.

**The predicate.** :func:`arm_can_run` answers, for one (stage, arm) pair,
whether the stage's entry point can produce a commensurate number for that arm,
and when it cannot, *which* declaration refuses it. It composes three sources,
none of which is restated here:

1. ``ArmSpec.capabilities`` -- what the panel intends the arm for.
2. The measuring module's own architecture declaration --
   ``scaling.LENS_ARCHITECTURES``, ``circuits._CIRCUIT_ARCHITECTURES``,
   ``path_patching.SUPPORTED_ARCHITECTURES``. A capability is an *intent* and a
   module declaration is what is *deliverable*; the two are allowed to disagree
   (arms.py says so for ``lens`` on the rotary arms) and a scheduler must obey
   the second, not the first.
3. ``ArmSpec.modality`` and ``ArmSpec.tokenisation``, where the stage's design
   needs them -- ``05_relational_channel.py`` needs a residue-to-token map.
**Why a generated shell file.** The controller and the worker are bash and the
declaration is Python that imports torch. Rather than let bash carry a third
copy, ``--emit`` renders the resolved contract into ``panel_contract.sh``, which
both shell scripts source, and ``--verify`` re-derives it and refuses if the
rendered file disagrees with the live panel. The worker runs ``--verify`` in its
preflight, before any GPU is scheduled, and ``tests/test_transfer_stage_contract.py``
runs it too, so a stale rendering cannot reach a measurement.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.arms import MODEL_ROOT, PANEL, TEXT_MODEL_BASE, TEXT_MODEL_ROOT  # noqa: E402
from src.transfer.circuits import _CIRCUIT_ARCHITECTURES  # noqa: E402
from src.transfer.path_patching import SUPPORTED_ARCHITECTURES  # noqa: E402
from src.transfer.probes import concepts_for_modality  # noqa: E402
from src.transfer.scaling import LENS_ARCHITECTURES  # noqa: E402

#: Where ``--emit`` writes and ``--verify`` reads. Beside this file so that the
#: controller's code freeze picks both up from ``scripts/transfer`` without a
#: second path to keep in step.
SHELL_CONTRACT = Path(__file__).resolve().parent / "panel_contract.sh"

#: v2 adds the per-arm checkpoint path relative to the variable that relocates it
#: (:func:`model_relative_path`), which is what lets the worker preflight a
#: checkpoint rather than a models root.
SCHEMA_VERSION = "r2_transfer_panel_contract_v2"


# --------------------------------------------------------------- campaign panel

#: The arms a campaign may schedule: every :data:`~src.transfer.arms.PANEL`
#: member whose checkpoint is staged on GPFS and byte-verified (EXP-R2-058).
#:
#: This is a *staging* fact and therefore cannot be derived from ``PANEL``, so it
#: is declared -- but :func:`_check_campaign_panel` requires every excluded panel
#: member to carry a reason below, which makes adding an arm to ``PANEL`` without
#: deciding its campaign status an import-time failure rather than a silent
#: omission from every campaign.
CAMPAIGN_PANEL: tuple[str, ...] = (
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "dialogpt-small",
    "qwen2.5-0.5b",
    "llama-3.2-3b",
    "protgpt2",
    "zymctrl",
    "progen2-base",
    "progen2-medium",
)

#: Panel members deliberately outside :data:`CAMPAIGN_PANEL`, with the reason.
PANEL_MEMBERS_NOT_STAGED: dict[str, str] = {
    "bygpt5-small-en": (
        "carries budget and lens capability only and its t5_decoder architecture "
        "is in no measuring module's architecture declaration, so it can enter "
        "exactly one campaign stage (cohort_power) and none of the stages the "
        "campaign exists to run"
    ),
    "bygpt5-base-en": "see bygpt5-small-en",
    "bygpt5-medium-en": "see bygpt5-small-en",
}


def _check_campaign_panel() -> None:
    unknown = [name for name in CAMPAIGN_PANEL if name not in PANEL]
    if unknown:
        raise AssertionError(
            f"CAMPAIGN_PANEL names arms that are not in src.transfer.arms.PANEL: {unknown}"
        )
    if len(set(CAMPAIGN_PANEL)) != len(CAMPAIGN_PANEL):
        raise AssertionError("CAMPAIGN_PANEL repeats an arm")
    undecided = sorted(
        name
        for name in PANEL
        if name not in CAMPAIGN_PANEL and name not in PANEL_MEMBERS_NOT_STAGED
    )
    if undecided:
        raise AssertionError(
            f"panel members {undecided} are neither in CAMPAIGN_PANEL nor given a "
            "reason in PANEL_MEMBERS_NOT_STAGED; a new arm must be admitted or "
            "excluded explicitly, never by omission"
        )


_check_campaign_panel()


# ------------------------------------------------------------- stage contracts


@dataclass(frozen=True)
class StageContract:
    """What one campaign stage requires of an arm, and how it is dispatched.

    ``scope`` is the stage's *panel contract*, made explicit because it was
    previously only inferable from how the worker happened to call the script,
    and getting it wrong cost a run of EXP-R2-060:

    ``per_arm``
        one process per arm; the arm list is a set of independent measurements.
    ``panel_wide``
        one process for the whole arm list, which writes a combined artefact.
        Splitting it produces incomplete panels that overwrite each other.
    ``control_anchored``
        one process whose arm list must contain exactly one text arm, because
        the verdict it produces is anchored on that control (evidence discipline
        rule 1). ``03_estimand_power.py recommend`` enforces this literally.
    ``armless``
        the stage takes no arm argument at all.
    """

    name: str
    entry_point: str
    scope: str
    capabilities: frozenset[str] = frozenset()
    architectures: frozenset[str] | None = None
    architecture_source: str = ""
    modalities: frozenset[str] | None = None
    tokenisations: frozenset[str] | None = None
    tokenisation_reason: str = ""
    declared_arms: tuple[str, ...] | None = None
    declared_arms_source: str = ""
    #: Protein residue band this stage's cohort is drawn on, as the stage's own
    #: argparse defaults set it. See :data:`QUALIFYING_PROTEIN_BAND`.
    protein_band: tuple[int, int] | None = None
    protein_band_reason: str = ""
    notes: str = ""


SCOPES = ("per_arm", "panel_wide", "control_anchored", "armless")

#: The band ``01_cohort_power.py`` qualifies an arm on. Every stage that draws a
#: protein cohort at a *different* band is measuring a different population from
#: the one the qualification verdict covers, so the difference is declared per
#: stage below and written into each artefact rather than left to be discovered
#: by comparing four argparse defaults.
QUALIFYING_PROTEIN_BAND = (64, 246)

STAGE_CONTRACTS: dict[str, StageContract] = {
    "cohort_power": StageContract(
        name="cohort_power",
        entry_point="01_cohort_power.py",
        scope="panel_wide",
        capabilities=frozenset({"budget"}),
        protein_band=(64, 246),
        protein_band_reason="this stage defines the qualifying band",
        notes=(
            "scores every arm passed to one invocation in one process and writes "
            "one combined report, so it cannot be dispatched per arm; the worker "
            "splits it by vocabulary regime instead (see COHORT_POWER_ITEMS)"
        ),
    ),
    "pathway_budget": StageContract(
        name="pathway_budget",
        entry_point="02_pathway_budget.py",
        scope="per_arm",
        capabilities=frozenset({"pathway"}),
        protein_band=(64, 246),
        protein_band_reason="matches the qualifying band",
    ),
    "estimand_power": StageContract(
        name="estimand_power",
        entry_point="03_estimand_power.py",
        scope="per_arm",
        capabilities=frozenset({"pathway"}),
        protein_band=(64, 246),
        protein_band_reason="matches the qualifying band",
        notes=(
            "`measure` is per arm; the `recommend` aggregation that follows it is "
            "control_anchored and takes the text control plus the protein arms only"
        ),
    ),
    "circuit_primitives": StageContract(
        name="circuit_primitives",
        entry_point="04_circuit_primitives.py",
        scope="panel_wide",
        architectures=frozenset(_CIRCUIT_ARCHITECTURES),
        architecture_source="src.transfer.circuits._CIRCUIT_ARCHITECTURES",
        notes=(
            "the `circuits` capability is deliberately NOT required: this stage "
            "carries grant_circuits(), an explicit per-arm override recorded in "
            "its own output, so the architecture declaration is the real gate"
        ),
    ),
    "relational_channel": StageContract(
        name="relational_channel",
        entry_point="05_relational_channel.py",
        scope="per_arm",
        capabilities=frozenset({"relational"}),
        modalities=frozenset({"protein"}),
        tokenisations=frozenset({"residue"}),
        tokenisation_reason=(
            "src.transfer.relational.require_residue_token_map needs one token per "
            "residue; a multi-residue BPE arm has no such map and must not have one "
            "approximated for it"
        ),
    ),
    "explanation_channel": StageContract(
        name="explanation_channel",
        entry_point="06_explanation_channel.py",
        scope="armless",
    ),
    "convergence_control": StageContract(
        name="convergence_control",
        entry_point="07_convergence_control.py",
        scope="armless",
        notes="sweeps src.transfer.scaling's ladder table, not the campaign arm list",
    ),
    "lens_family": StageContract(
        name="lens_family",
        entry_point="08_lens_family.py",
        scope="per_arm",
        capabilities=frozenset({"lens"}),
        architectures=frozenset(LENS_ARCHITECTURES),
        architecture_source="src.transfer.scaling.LENS_ARCHITECTURES",
        protein_band=(64, 120),
        protein_band_reason=(
            "NARROWER THAN THE QUALIFYING BAND. The Jacobian sweep is quadratic in "
            "sequence length and this band was chosen for cost. It means an arm "
            "qualified by cohort_power at 64-246 residues is scored here on a "
            "different protein population, and EXP-R2-060 measured protein "
            "cohort-block sensitivity at 0.16-0.60 nats. Declared so the artefact "
            "records the mismatch; not silently reconciled, because changing it "
            "would move published lens numbers"
        ),
    ),
    "probe_and_erasure": StageContract(
        name="probe_and_erasure",
        entry_point="09_probe_and_erasure.py",
        scope="per_arm",
        notes=(
            "every arm is valid: src.transfer.probes declares concepts for both "
            f"modalities (text {list(concepts_for_modality('text'))}, protein "
            f"{list(concepts_for_modality('protein'))}) and writes per-concept "
            "refusals into the output rather than raising"
        ),
    ),
    "homology_control": StageContract(
        name="homology_control",
        entry_point="10_homology_control.py",
        scope="panel_wide",
        modalities=frozenset({"protein"}),
        declared_arms=("protgpt2", "zymctrl", "progen2-medium"),
        declared_arms_source="10_homology_control.py::PROTEIN_ARMS",
        notes=(
            "the stage declares its own arm set and this mirrors it, checked "
            "against the source by tests/test_transfer_stage_contract.py. Note "
            "that progen2-base is protein and carries every capability this stage "
            "needs but is absent from that declaration; the worker used to pass "
            "its own four-arm protein list, so a campaign run and a direct run "
            "measured different panels"
        ),
    ),
    "induction_path_patching": StageContract(
        name="induction_path_patching",
        entry_point="11_induction_path_patching.py",
        scope="panel_wide",
        capabilities=frozenset({"circuits"}),
        architectures=frozenset(SUPPORTED_ARCHITECTURES),
        architecture_source="src.transfer.path_patching.SUPPORTED_ARCHITECTURES",
    ),
}

#: Campaign stage order. The worker's tier structure depends on it: cohort_power
#: qualifies the cohort every later stage draws on, and estimand_power's
#: recommendation reads pathway_budget's regime.
STAGE_ORDER: tuple[str, ...] = tuple(STAGE_CONTRACTS)


def _check_stage_contracts() -> None:
    for stage, contract in STAGE_CONTRACTS.items():
        if contract.name != stage:
            raise AssertionError(f"stage {stage!r} declares name {contract.name!r}")
        if contract.scope not in SCOPES:
            raise AssertionError(f"stage {stage!r} declares unknown scope {contract.scope!r}")
        if contract.architectures is not None and not contract.architecture_source:
            raise AssertionError(
                f"stage {stage!r} restricts architectures without naming the module "
                "declaration it mirrors"
            )
        if contract.declared_arms is not None:
            unknown = [a for a in contract.declared_arms if a not in PANEL]
            if unknown:
                raise AssertionError(f"stage {stage!r} declares unknown arms {unknown}")
_check_stage_contracts()


# ------------------------------------------------------------------ predicate


@dataclass(frozen=True)
class Eligibility:
    """Whether one arm may enter one stage, and the declaration that decides it."""

    stage: str
    arm: str
    can_run: bool
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.can_run


def require_known(stage: str, arm: str) -> StageContract:
    if stage not in STAGE_CONTRACTS:
        raise KeyError(f"unknown stage {stage!r}; stages are {list(STAGE_CONTRACTS)}")
    if arm not in PANEL:
        raise KeyError(f"unknown arm {arm!r}; panel is {sorted(PANEL)}")
    return STAGE_CONTRACTS[stage]


def arm_can_run(stage: str, arm: str) -> Eligibility:
    """Can ``stage``'s entry point produce a commensurate number for ``arm``?

    A ``False`` is always accompanied by the declaration that refuses, so an
    operator reading a skip line can tell a capability decision (the panel's) from
    a module limitation (the measuring code's) from a staging fact.

    Unknown stage or unknown arm raises rather than returning ``False``: "this
    arm cannot run" and "nobody has heard of this arm" are different facts and
    collapsing them is how a typo becomes a silently narrower panel.
    """

    contract = require_known(stage, arm)
    spec = PANEL[arm]

    if contract.scope == "armless":
        return Eligibility(
            stage,
            arm,
            False,
            f"{contract.entry_point} takes no arm argument; it is not dispatched per arm",
        )

    if contract.declared_arms is not None and arm not in contract.declared_arms:
        return Eligibility(
            stage,
            arm,
            False,
            f"not in {contract.declared_arms_source}, which is this stage's own "
            f"declaration of the arms it measures ({list(contract.declared_arms)})",
        )

    missing = sorted(contract.capabilities - spec.capabilities)
    if missing:
        return Eligibility(
            stage,
            arm,
            False,
            f"ArmSpec.capabilities does not grant {missing}; declared capabilities "
            f"are {sorted(spec.capabilities)}",
        )

    if contract.architectures is not None and spec.architecture not in contract.architectures:
        return Eligibility(
            stage,
            arm,
            False,
            f"architecture {spec.architecture!r} is not in "
            f"{contract.architecture_source} = {sorted(contract.architectures)}, so the "
            "measuring module has no code path for this arm's module layout",
        )

    if contract.modalities is not None and spec.modality not in contract.modalities:
        return Eligibility(
            stage,
            arm,
            False,
            f"modality {spec.modality!r}; this stage measures "
            f"{sorted(contract.modalities)} arms only",
        )

    if contract.tokenisations is not None and spec.tokenisation not in contract.tokenisations:
        return Eligibility(
            stage,
            arm,
            False,
            f"tokenisation {spec.tokenisation!r} is not in "
            f"{sorted(contract.tokenisations)}: {contract.tokenisation_reason}",
        )

    return Eligibility(stage, arm, True)


def stage_arms(
    stage: str, requested: list[str] | tuple[str, ...] | None = None
) -> tuple[list[str], list[Eligibility]]:
    """``(eligible, refused)`` for one stage over ``requested`` (default: the campaign panel).

    Order follows ``requested``, so a caller that cares about invocation order
    keeps it. ``refused`` carries one :class:`Eligibility` per skipped arm, each
    with its reason, for the caller to log -- a skip that is not logged is
    indistinguishable from an arm that was never asked for.
    """

    names = list(CAMPAIGN_PANEL if requested is None else requested)
    eligible: list[str] = []
    refused: list[Eligibility] = []
    for arm in names:
        verdict = arm_can_run(stage, arm)
        if verdict.can_run:
            eligible.append(arm)
        else:
            refused.append(verdict)
    return eligible, refused


def stage_contract_record(stage: str, arms: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """The block a stage writes into its own artefact to declare its panel and band.

    Two facts that were previously only recoverable by comparing argparse
    defaults across four files:

    ``arm_selection``
        which arms ran, which panel members did not, and why. A stage that
        measures a subset of the panel and does not say so is the L18 shape.
    ``cohort_band``
        the protein residue band this stage draws on, beside the band
        ``01_cohort_power.py`` *qualified* the arms on, and a flag for whether
        they agree. They do not agree for ``lens_family``, and EXP-R2-060
        measured protein cohort-block sensitivity at 0.16-0.60 nats, so the
        difference is worth carrying with the number.
    """

    contract = STAGE_CONTRACTS[stage]
    eligible, refused = stage_arms(stage)
    measured = list(arms)
    # Two different reasons an arm is absent, kept apart because they mean
    # different things to a reader. "Refused" is a property of the arm and the
    # module -- it could not have been measured. "Eligible but not asked for" is a
    # property of *this invocation* -- the operator, a default or a worker scoped
    # the run, and that is the narrowing L18 records.
    return {
        "stage": stage,
        "scope": contract.scope,
        "arm_selection": {
            "measured": measured,
            "campaign_panel": list(CAMPAIGN_PANEL),
            "eligible_for_this_stage": eligible,
            "eligible_but_not_measured": [
                name for name in eligible if name not in measured
            ],
            "not_measured": {
                **{
                    name: "eligible for this stage but not in this invocation's arm list"
                    for name in eligible
                    if name not in measured
                },
                **{v.arm: v.reason for v in refused if v.arm not in measured},
            },
        },
        "cohort_band": {
            "protein_residues": (
                None if contract.protein_band is None else list(contract.protein_band)
            ),
            "qualifying_stage_protein_residues": list(QUALIFYING_PROTEIN_BAND),
            "matches_qualifying_stage": (
                None
                if contract.protein_band is None
                else contract.protein_band == QUALIFYING_PROTEIN_BAND
            ),
            "reason": contract.protein_band_reason or None,
        },
    }


# --------------------------------------------------- cohort_power item dispatch


@dataclass(frozen=True)
class CohortPowerItem:
    """One ``01_cohort_power.py`` invocation, and why it is separate from the rest.

    ``01`` writes its combined report only after its whole per-arm loop finishes,
    so one arm raising loses every arm already computed in that invocation. The
    split is therefore by the properties that decide whether an arm *can* be in
    the same process, and each reason is measured rather than assumed.
    """

    item: str
    arms: tuple[str, ...]
    extra_args: tuple[str, ...]
    cohort_name: str | None
    reason: str


#: Arms whose vocabulary exceeds 1024 pieces cannot compute the truncation curve
#: on a transformers build without ``logits_to_keep`` (the pod ships 4.52.4), and
#: ``budget.truncation_curve`` raises rather than trimming, because trimming is
#: numerically non-inert (up to 0.25 in a logit, 0.12 nats in one token's NLL).
_VOCAB_TRUNCATION_LIMIT = 1024

COHORT_POWER_ITEM_RULES: tuple[tuple[str, str], ...] = (
    (
        "text",
        "text arms share one OpenWebText cohort; vocabulary > 1024 so --skip-truncation",
    ),
    (
        "protein_large_vocab",
        "protein arms with vocabulary > 1024: the truncation curve is not computable "
        "on this transformers build, so --skip-truncation",
    ),
    (
        "protein_small_vocab",
        "EC-conditioned residue-level arms: --with-ec, and the truncation curve is "
        "computable so it must NOT be skipped",
    ),
    (
        "protein_progen2_base",
        "ProGen2-base uses the script's declared default dtype; no precision override "
        "is inferred from a different checkpoint",
    ),
    (
        "protein_progen2_medium",
        "ProGen2-medium is isolated at --dtype float32: its "
        "nll_reduction_shortest_to_longest_nats moved 0.6266 -> 0.7293 (+16%) under "
        "bfloat16 in the L20-vs-H200 cross-check and collapsed to 2.6e-7 in float32; "
        "--dtype governs model loading so it cannot be set per arm within one process",
    ),
)


def cohort_power_items(requested: list[str] | tuple[str, ...] | None = None) -> list[CohortPowerItem]:
    """The ``01_cohort_power.py`` invocations covering ``requested``.

    Derived from ``PANEL`` -- modality, tokenisation and the declared EC input
    format -- rather than from arm names, which is what the worker used to do in
    bash. Every protein item gets a distinct ``--cohort-name`` because two
    protein items can otherwise produce byte-identical cohorts under the shared
    default name and collide on the same output filename.
    """

    eligible, _ = stage_arms("cohort_power", requested)
    buckets: dict[str, list[str]] = {name: [] for name, _ in COHORT_POWER_ITEM_RULES}
    for arm in eligible:
        spec = PANEL[arm]
        if spec.modality == "text":
            buckets["text"].append(arm)
        elif spec.input_format == "ec_conditioned":
            buckets["protein_small_vocab"].append(arm)
        elif _vocab_regime(arm) == "large":
            buckets["protein_large_vocab"].append(arm)
        elif arm == "progen2-medium":
            buckets["protein_progen2_medium"].append(arm)
        else:
            buckets["protein_progen2_base"].append(arm)

    extra = {
        "text": ("--skip-truncation",),
        "protein_large_vocab": ("--skip-truncation",),
        "protein_small_vocab": ("--with-ec",),
        "protein_progen2_base": (),
        "protein_progen2_medium": ("--dtype", "float32"),
    }
    cohort_names = {
        "text": None,
        "protein_large_vocab": "swissprot_large_vocab",
        "protein_small_vocab": "swissprot_small_vocab",
        "protein_progen2_base": "swissprot_progen2_base",
        "protein_progen2_medium": "swissprot_progen2_medium_f32",
    }
    items: list[CohortPowerItem] = []
    for item, reason in COHORT_POWER_ITEM_RULES:
        if not buckets[item]:
            continue
        items.append(
            CohortPowerItem(
                item=item,
                arms=tuple(buckets[item]),
                extra_args=extra[item],
                cohort_name=cohort_names[item],
                reason=reason,
            )
        )
    return items


#: Vocabulary regime per arm, keyed off the declared tokenisation family rather
#: than read from a config file, because the campaign panel's regimes are a
#: declared design property (the 1600-fold aperture spread of L8) and a scheduler
#: must not need a checkpoint on disk to plan a run.
_LARGE_VOCAB_TOKENISATIONS = frozenset({"bpe", "multi_residue_bpe"})


def _vocab_regime(arm: str) -> str:
    return "large" if PANEL[arm].tokenisation in _LARGE_VOCAB_TOKENISATIONS else "small"


# ------------------------------------------------------- per-arm data locations

#: Environment variable that relocates each arm's checkpoint, resolved from how
#: ``arms.PANEL`` *builds* the path rather than from the arm's name.
#:
#: The worker used to answer this with ``case "$1" in gpt2-large) ...`` and a
#: modality fallback, and got it wrong for six of seven text arms until
#: 2026-07-29: an arm addressed beneath ``TRANSFER_TEXT_MODEL_BASE_DIR`` had its
#: preflight check ``TRANSFER_MODEL_BASE_DIR`` instead, so a genuinely missing
#: checkpoint reached ``load_arm`` rather than being reported as a skip.
#:
#: The comparison below is invariant to which host it runs on, because every
#: ``ArmSpec.path`` is *constructed* from one of these three constants: gpt2-large
#: is declared as ``TEXT_MODEL_ROOT`` itself, the protein arms as
#: ``MODEL_ROOT / name`` and the remaining text arms as ``TEXT_MODEL_BASE / name``.
#: Re-pointing any of the three environment variables moves the constant and the
#: arm's path together, so the mapping this produces on the controller host is the
#: mapping the pod re-derives under ``--verify``.
MODEL_PATH_VARIABLES = frozenset(
    {"TRANSFER_TEXT_MODEL_DIR", "TRANSFER_MODEL_BASE_DIR", "TRANSFER_TEXT_MODEL_BASE_DIR"}
)


def model_variable(arm: str) -> str:
    """The declared variable, read from the panel rather than inferred from paths.

    This used to compare the resolved ``ArmSpec.path`` against the three
    constants, and claimed to be host-invariant on the grounds that re-pointing a
    variable moves the constant and the arm's path together. That reasoning holds
    only while the three constants resolve to *distinct* directories. They do not
    on the H200 pod: every checkpoint lives in one GPFS directory, so
    ``h200_env.sh`` sets ``TRANSFER_TEXT_MODEL_BASE_DIR="${TRANSFER_MODEL_BASE_DIR}"``,
    the ``path.parent == MODEL_ROOT`` branch matched first, and six text arms
    classified as protein-root arms. The rendered contract therefore disagreed
    with the live panel *inside the pod and nowhere else*, which is why the
    worker's own re-derivation refused the campaign before any GPU was scheduled.

    The variable is now declared beside the path it builds, so an alias cannot
    change the answer.
    """

    variable = PANEL[arm].path_variable
    if variable not in MODEL_PATH_VARIABLES:
        raise AssertionError(
            f"{arm}: declares path_variable {variable!r}, which is not one of "
            f"{sorted(MODEL_PATH_VARIABLES)}, so no environment variable relocates "
            "it and the worker cannot preflight it"
        )
    return variable


def _check_declared_paths_match_their_variables() -> None:
    """The declaration and the construction must agree where they can be compared.

    A declared variable that no longer matches how the path is built would be a
    silent lie, so it is checked -- but only when the constants are distinct,
    because when two of them alias there is nothing to check and the declaration
    is the only thing that carries the answer. That is precisely the situation
    this field exists for.
    """

    roots = {
        "TRANSFER_TEXT_MODEL_DIR": TEXT_MODEL_ROOT,
        "TRANSFER_MODEL_BASE_DIR": MODEL_ROOT,
        "TRANSFER_TEXT_MODEL_BASE_DIR": TEXT_MODEL_BASE,
    }
    if len({str(value) for value in roots.values()}) != len(roots):
        return
    for arm, spec in PANEL.items():
        variable = spec.path_variable
        expected = roots[variable]
        built = spec.path if variable == "TRANSFER_TEXT_MODEL_DIR" else spec.path.parent
        if built != expected:
            raise AssertionError(
                f"{arm}: declares path_variable {variable!r} but its path {spec.path} "
                f"is not built from {expected}"
            )


_check_declared_paths_match_their_variables()


#: The arm's checkpoint path *relative to* :func:`model_variable`'s answer: ``"."``
#: when the arm is declared as that variable itself, and the checkpoint directory's
#: own name otherwise.
#:
#: :func:`model_variable` alone is the wrong granularity for a preflight. Six of the
#: seven text arms resolve ``TRANSFER_TEXT_MODEL_BASE_DIR``, which is the models
#: *root*: it exists as soon as any text checkpoint is staged, so an arm whose own
#: checkpoint was absent passed the worker's data check and raised inside
#: ``load_arm`` instead -- and ``cohort_power`` scores all seven text arms in one
#: process, so that lost the six arms that were fine along with the one that was
#: not. Checking ``${!variable}/<relative>`` turns it back into a logged skip.
#:
#: Derived from ``ArmSpec.path``, never a restated leaf name, and classified through
#: :func:`model_variable` so there is exactly one place that decides which constant
#: an arm's path is built from. Re-pointing any of the three environment variables
#: moves the constant and the arm's path together, so this is as host-independent as
#: the variable mapping it accompanies.
def model_relative_path(arm: str) -> str:
    if model_variable(arm) == "TRANSFER_TEXT_MODEL_DIR":
        return "."
    return PANEL[arm].path.name


#: Corpus variables a cohort covering this arm needs, from the arm's declared
#: evaluation cohort rather than from its name. ``zymctrl_ec`` needs the
#: EC-labelled FASTA *and* nothing else: its records carry the conditioning tag
#: that ``Cohort.input_strings`` rebuilds the native prompt from.
_CORPUS_VARIABLES: dict[str, tuple[str, ...]] = {
    "openwebtext": ("TRANSFER_OPENWEBTEXT_DIR",),
    "swissprot": ("TRANSFER_SWISSPROT_FASTA",),
    "zymctrl_ec": ("TRANSFER_ZYMCTRL_FASTA",),
}


def corpus_variables(arm: str) -> tuple[str, ...]:
    source = PANEL[arm].evaluation_cohort_source
    if source not in _CORPUS_VARIABLES:
        raise AssertionError(
            f"{arm}: evaluation cohort source {source!r} has no declared corpus "
            f"variable; known sources are {sorted(_CORPUS_VARIABLES)}"
        )
    return _CORPUS_VARIABLES[source]


def _check_data_locations() -> None:
    for arm in CAMPAIGN_PANEL:
        model_variable(arm)
        model_relative_path(arm)
        corpus_variables(arm)


_check_data_locations()


# -------------------------------------------------------------- serialisation


def contract_payload() -> dict[str, Any]:
    """The whole resolved contract, as the record a run manifest can embed."""

    stages: dict[str, Any] = {}
    for stage, contract in STAGE_CONTRACTS.items():
        eligible, refused = stage_arms(stage)
        stages[stage] = {
            "entry_point": contract.entry_point,
            "scope": contract.scope,
            "required_capabilities": sorted(contract.capabilities),
            "required_architectures": (
                None if contract.architectures is None else sorted(contract.architectures)
            ),
            "architecture_source": contract.architecture_source or None,
            "required_modalities": (
                None if contract.modalities is None else sorted(contract.modalities)
            ),
            "required_tokenisations": (
                None if contract.tokenisations is None else sorted(contract.tokenisations)
            ),
            "declared_arms": (
                None if contract.declared_arms is None else list(contract.declared_arms)
            ),
            "protein_residue_band": (
                None if contract.protein_band is None else list(contract.protein_band)
            ),
            "protein_band_matches_qualifying_stage": (
                None
                if contract.protein_band is None
                else contract.protein_band == QUALIFYING_PROTEIN_BAND
            ),
            "protein_band_reason": contract.protein_band_reason or None,
            "eligible_arms": eligible,
            "refused_arms": {v.arm: v.reason for v in refused},
            "notes": contract.notes or None,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_panel": list(CAMPAIGN_PANEL),
        "arms": {
            arm: {
                "modality": PANEL[arm].modality,
                "model_variable": model_variable(arm),
                "model_relative_path": model_relative_path(arm),
                "corpus_variables": list(corpus_variables(arm)),
            }
            for arm in CAMPAIGN_PANEL
        },
        "panel_members_not_staged": dict(PANEL_MEMBERS_NOT_STAGED),
        "qualifying_protein_residue_band": list(QUALIFYING_PROTEIN_BAND),
        "stage_order": list(STAGE_ORDER),
        "stages": stages,
        "cohort_power_items": [
            {
                "item": item.item,
                "arms": list(item.arms),
                "extra_args": list(item.extra_args),
                "cohort_name": item.cohort_name,
                "reason": item.reason,
            }
            for item in cohort_power_items()
        ],
    }


def _quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def render_shell() -> str:
    """The bash fragment the controller and the worker source.

    Associative arrays rather than one variable per (stage, arm): arm names carry
    ``.`` and ``-`` (``qwen2.5-0.5b``), which cannot appear in a bash variable
    name, and encoding around that is exactly the kind of second representation
    this file exists to remove.
    """

    payload = contract_payload()
    lines = [
        "# GENERATED by scripts/transfer/panel_contract.py --emit. Do not edit.",
        "# Sourced by run_transfer_h200.sh (controller) and h200_worker.sh (worker),",
        "# which is why the campaign panel and every stage's arm list exist in exactly",
        "# one place. h200_worker.sh re-derives this file from src/transfer/arms.py in",
        "# its preflight (panel_contract.py --verify) and refuses to schedule a GPU if",
        "# the two disagree, so a stale copy cannot reach a measurement.",
        f"TRANSFER_CONTRACT_SCHEMA={_quote(SCHEMA_VERSION)}",
        f"TRANSFER_CAMPAIGN_PANEL={_quote(' '.join(payload['campaign_panel']))}",
        f"TRANSFER_STAGE_ORDER={_quote(' '.join(payload['stage_order']))}",
        "declare -A TRANSFER_STAGE_SCOPE=()",
        "declare -A TRANSFER_STAGE_ENTRY=()",
        "declare -A TRANSFER_STAGE_ARMS=()",
        "declare -A TRANSFER_STAGE_REFUSAL=()",
        "declare -A TRANSFER_ARM_MODALITY=()",
        "declare -A TRANSFER_ARM_MODEL_VAR=()",
        "declare -A TRANSFER_ARM_MODEL_REL=()",
        "declare -A TRANSFER_ARM_CORPUS_VARS=()",
        "declare -A TRANSFER_COHORT_ITEM_ARMS=()",
        "declare -A TRANSFER_COHORT_ITEM_ARGS=()",
        "declare -A TRANSFER_COHORT_ITEM_COHORT_NAME=()",
    ]
    for arm in payload["campaign_panel"]:
        record = payload["arms"][arm]
        key = _quote(arm)
        lines.append(f"TRANSFER_ARM_MODALITY[{key}]={_quote(record['modality'])}")
        lines.append(f"TRANSFER_ARM_MODEL_VAR[{key}]={_quote(record['model_variable'])}")
        lines.append(f"TRANSFER_ARM_MODEL_REL[{key}]={_quote(record['model_relative_path'])}")
        lines.append(
            f"TRANSFER_ARM_CORPUS_VARS[{key}]={_quote(' '.join(record['corpus_variables']))}"
        )
    for stage in payload["stage_order"]:
        record = payload["stages"][stage]
        lines.append(f"TRANSFER_STAGE_SCOPE[{_quote(stage)}]={_quote(record['scope'])}")
        lines.append(f"TRANSFER_STAGE_ENTRY[{_quote(stage)}]={_quote(record['entry_point'])}")
        lines.append(
            f"TRANSFER_STAGE_ARMS[{_quote(stage)}]={_quote(' '.join(record['eligible_arms']))}"
        )
        for arm, reason in sorted(record["refused_arms"].items()):
            key = f"{stage}/{arm}"
            lines.append(f"TRANSFER_STAGE_REFUSAL[{_quote(key)}]={_quote(reason)}")
    cohort_items = [item["item"] for item in payload["cohort_power_items"]]
    lines.append(f"TRANSFER_COHORT_ITEMS={_quote(' '.join(cohort_items))}")
    for item in payload["cohort_power_items"]:
        key = _quote(item["item"])
        lines.append(f"TRANSFER_COHORT_ITEM_ARMS[{key}]={_quote(' '.join(item['arms']))}")
        lines.append(f"TRANSFER_COHORT_ITEM_ARGS[{key}]={_quote(' '.join(item['extra_args']))}")
        lines.append(
            f"TRANSFER_COHORT_ITEM_COHORT_NAME[{key}]="
            f"{_quote(item['cohort_name'] or '')}"
        )
    return "\n".join(lines) + "\n"


def emit(path: Path = SHELL_CONTRACT) -> Path:
    path.write_text(render_shell(), encoding="utf-8")
    return path


def verify(path: Path = SHELL_CONTRACT) -> list[str]:
    """Differences between the rendered file and the live panel; empty means clean."""

    if not path.exists():
        return [f"{path} does not exist; run panel_contract.py --emit"]
    on_disk = path.read_text(encoding="utf-8")
    expected = render_shell()
    if on_disk == expected:
        return []
    disk_lines = on_disk.splitlines()
    want_lines = expected.splitlines()
    problems = [f"{path} disagrees with src/transfer/arms.py"]
    for index in range(max(len(disk_lines), len(want_lines))):
        got = disk_lines[index] if index < len(disk_lines) else "<missing>"
        want = want_lines[index] if index < len(want_lines) else "<missing>"
        if got != want:
            problems.append(f"  line {index + 1}: on disk [{got}] expected [{want}]")
    return problems


def declared_arms_in_source(entry_point: str, symbol: str) -> tuple[str, ...]:
    """A module-level tuple literal read out of a sibling entry point without importing it.

    The numbered entry points cannot be imported by name, and importing one by
    path executes its module body. This reads the declaration statically, which
    is what lets :data:`STAGE_CONTRACTS` mirror ``10_homology_control.py``'s own
    arm list under test instead of restating it and hoping.
    """

    path = Path(__file__).resolve().parent / entry_point
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == symbol:
                    return tuple(ast.literal_eval(node.value))
    raise LookupError(f"{entry_point} has no module-level assignment to {symbol}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", action="store_true", help="write panel_contract.sh")
    parser.add_argument(
        "--verify", action="store_true", help="fail if panel_contract.sh is stale"
    )
    parser.add_argument("--json", action="store_true", help="print the resolved contract")
    parser.add_argument("--path", type=Path, default=SHELL_CONTRACT)
    args = parser.parse_args()

    if args.json:
        print(json.dumps(contract_payload(), indent=2, sort_keys=True))
    if args.emit:
        print(f"wrote {emit(args.path)}")
    if args.verify:
        problems = verify(args.path)
        if problems:
            for line in problems:
                print(line, file=sys.stderr)
            raise SystemExit(2)
        print(f"{args.path} matches src/transfer/arms.py")
    if not (args.emit or args.verify or args.json):
        parser.error("nothing to do: pass --emit, --verify or --json")


if __name__ == "__main__":
    main()
