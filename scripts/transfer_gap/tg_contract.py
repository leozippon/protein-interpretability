#!/usr/bin/env python3
"""One declaration of what population each TG stage measures, and on what seed.

``scripts/transfer/panel_contract.py`` does this for the campaign stages. The TG
series had no equivalent, and it is the part of the repository that has produced
the most retractions.

**The defect class this closes.** Appendix B rule 13 -- *declare a stage's cohort
band against the band the arms were qualified on* -- was earned by
``scripts/transfer/`` and fixed there by EXP-R2-063. The identical shape was
still live here. The eleven TG stages draw protein cohorts on **three different
residue bands**:

* 400-1000 -- TG-01, TG-02, TG-06
* 120-1000 -- TG-03, TG-07, TG-08, TG-09
* 64-246   -- TG-10

Those are legitimate compute choices, individually. Together, undeclared, they
let TG-01's information budget and TG-10's causal headroom be read as two
measurements of one cohort when they share no protein at all below 400 residues,
and EXP-R2-060 prices protein cohort-block sensitivity at 0.16-0.60 nats. Every
band is now stated here with its reason, and a stage whose argparse default
disagrees with this table fails :func:`verify`.

**What the first version of this file still missed, and what closed it.** The
band check looked up the literal argparse keys ``res_min``/``res_max``, so it saw
nothing in the two stages that spell a residue bound differently: ``tg00`` draws
``--render-min/--render-max`` at 600-2000 *and* ``--cohort-min/--cohort-max`` at
200-800, and ``tg05`` drew ``--min-len/--max-len`` at 110-320 while the table
below asserted, in prose, that the stage "has no residue band". Three undeclared
bands inside the mechanism built to stop undeclared bands. The check now matches
on the *shape* of an argument pair and refuses a band in either direction -- one
the code carries and the table omits, or one the table declares and the code has
dropped -- and TG-05's pair is renamed to the panel's spelling.

``stage_contract_record`` had the second half of the same shape: it was written,
tested, and called by nothing, so no artefact in the corrected tree carried a
``cohort_band`` key at all. Every measuring stage writes it now.

**Arm eligibility was declared in two places and enforced in the wrong one.**
``tg05`` and ``tg06`` declared ``arms=None`` here, meaning "the whole TG panel",
and then hard-refused arms in their own bodies -- ``tg05`` on the literal string
``"protgpt2"``, two lines above a correctly written ``input_format`` check.
``tg05`` can only ever produce one of four arms and ``tg06`` three, so ``tg99``'s
strict mode was unsatisfiable by a *fully executed* campaign, which made
``--allow-partial`` mandatory and the strict default decorative. Eligibility is
now one predicate over :class:`ArmSpec` fields per stage, ``arms`` is its
projection onto :data:`TG_PANEL`, and the stages branch on it before loading a
model.

**What verification found when it was first run.** Appendix B rule 12 -- a single
declaration, imported, never reimplemented -- was violated on the cohort seed.
``tg_common.DEFAULT_COHORT_SEED`` exists precisely so that ``skip`` partitions
one permutation across stages, and its docstring says so; nine stages restated
the integer by hand anyway, and one of them restated it *wrong*, as the
pre-correction 20260724. That stage, ``tg01_information_budget.py``, also carried
two ``--seed`` arguments on one parser, so it raised ``argparse.ArgumentError``
before doing anything at all. Nothing in the repository executed it or read its
arguments, which is why a completely dead entry point survived. This file is that
missing reader.

**Static, not imported.** The entry points cannot be imported by name (they begin
with digits) and importing one by path executes its module body, which loads
torch. :func:`argparse_defaults` reads each stage's ``add_argument`` calls out of
the source with ``ast``, the same technique
``panel_contract.declared_arms_in_source`` uses, so the check runs on the
controller host and in the test suite with no GPU and no model.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_STAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _STAGE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(_STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_STAGE_DIR))

from src.transfer.arms import PANEL, ArmSpec  # noqa: E402
from tg_common import DEFAULT_COHORT_SEED, TG_PANEL  # noqa: E402

SCHEMA_VERSION = "r2_transfer_gap_contract_v2"

#: The band on which the TG comparison table in the audit is read. Chosen as
#: 120-1000 because it is the band of TG-03, TG-07, TG-08 and TG-09 -- the
#: dictionary and variance stages the table's load-bearing rows come from -- not
#: because it is better. A stage on a different band is not wrong; it is
#: measuring a different population, and that is what has to be visible.
REFERENCE_PROTEIN_BAND = (120, 1000)


@dataclass(frozen=True)
class ProteinBand:
    """One residue band a stage draws a cohort on, and the argument that sets it.

    ``argument_prefix`` names the argparse option pair the stage carries, so
    :func:`verify` can read the band back out of the source instead of trusting
    this table. It is a *pair* prefix -- ``"res"`` means ``--res-min/--res-max``
    -- and it exists because the previous check looked up the literal keys
    ``res_min``/``res_max`` and therefore saw nothing at all in the two stages
    that spell their band differently. ``tg00`` draws ``--render-min/--render-max``
    at 600-2000 *and* ``--cohort-min/--cohort-max`` at 200-800, and ``tg05`` drew
    ``--min-len/--max-len`` at 110-320 while this table asserted it had no residue
    band. Three live, undeclared bands inside the mechanism built to stop
    undeclared bands.
    """

    argument_prefix: str
    residues: tuple[int, int]
    reason: str = ""

    @property
    def matches_reference(self) -> bool:
        return self.residues == REFERENCE_PROTEIN_BAND


@dataclass(frozen=True)
class TgStage:
    """What one TG stage measures, on which arms, over which residues.

    ``scope``
        ``per_arm`` -- takes ``--arm`` and writes one artefact per arm.
        ``multi_arm`` -- takes ``--arms`` and writes one artefact per arm in one
        process.
        ``armless`` -- takes no arm argument; it measures the corpus, not a model.
        ``summary`` -- reads other stages' artefacts and writes no measurement.

    ``protein_bands``
        every residue band this stage's argparse defaults set, as
        :class:`ProteinBand` records. Empty for a stage that draws no protein
        cohort. Checked against the source by :func:`verify` in both directions,
        so neither an undeclared band in the code nor a declared band that no
        longer exists can survive.

    ``arm_predicate``
        the stage's real eligibility rule, written against :class:`ArmSpec`
        fields. ``arms`` is that predicate's projection onto :data:`TG_PANEL`
        and :func:`_check_stages` asserts the two agree, so the arm set a reader
        (and ``tg99``) sees cannot drift from the rule the stage enforces.

        It exists because two stages hard-refused arms *in their own bodies*
        while declaring ``arms=None`` here, and by arm name rather than by
        property: ``tg05`` raised ``SystemExit`` on the literal ``"protgpt2"``
        two lines above a correctly written ``input_format == "ec_conditioned"``
        check, and ``gpt2-large`` died inside ``encode`` after a model load. The
        consequence was structural: ``tg99``'s strict mode expected four arms from
        a stage that can only ever produce one, so a fully executed campaign could
        never satisfy it and ``--allow-partial`` became mandatory. A default mode
        that cannot be reached is not a default.

    ``arms_reason``
        why the set is narrowed. Required whenever it is.
    """

    name: str
    entry_point: str
    scope: str
    protein_bands: tuple[ProteinBand, ...] = ()
    arm_predicate: Callable[[ArmSpec], bool] | None = field(default=None, compare=False)
    arms: tuple[str, ...] | None = None
    arms_reason: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        """Project ``arm_predicate`` onto the TG panel, once.

        The predicate is the declaration and ``arms`` is derived from it, rather
        than both being written out and hoped to agree -- Appendix B rule 12
        applied to eligibility. A stage that supplies both is checked, so a
        hand-edited tuple cannot quietly diverge from the rule the stage runs.
        """

        if self.arm_predicate is None:
            return
        derived = eligible_arms(self.arm_predicate)
        if self.arms is not None and tuple(self.arms) != derived:
            raise AssertionError(
                f"stage {self.name!r} declares arms {tuple(self.arms)} but its "
                f"eligibility predicate admits {derived}"
            )
        object.__setattr__(self, "arms", derived)

    @property
    def protein_band(self) -> tuple[int, int] | None:
        """The one band this stage draws on, or ``None`` if it draws none or several.

        A stage with two bands has no single population, which is precisely what
        a reader combining two stages' numbers has to know; see ``tg99``'s
        commensurability check.
        """

        if len(self.protein_bands) == 1:
            return self.protein_bands[0].residues
        return None

    def eligible(self, arm: str) -> bool:
        """Whether this stage can measure ``arm`` at all.

        Answered from :class:`ArmSpec` where a predicate exists, so an arm
        outside :data:`TG_PANEL` that has the required properties is still
        admitted, and a declared-arms stage falls back to its literal set.
        """

        if arm not in PANEL:
            raise KeyError(f"unknown arm {arm!r}; panel is {sorted(PANEL)}")
        if self.arm_predicate is not None:
            return self.arm_predicate(PANEL[arm])
        return self.arms is None or arm in self.arms


SCOPES = ("per_arm", "multi_arm", "armless", "summary")


# ------------------------------------------------- eligibility, from ArmSpec


def eligible_arms(predicate: Callable[[ArmSpec], bool]) -> tuple[str, ...]:
    """The TG panel members a stage's eligibility rule admits, in panel order."""

    return tuple(name for name in TG_PANEL if predicate(PANEL[name]))


def residue_addressable(spec: ArmSpec) -> bool:
    """One token per residue, so a residue index maps to a token index.

    ProtGPT2 fails this on ``tokenisation == "multi_residue_bpe"``: its merges
    span several residues, so there is no residue-to-token map to build a
    per-residue readout on. That is a property of the checkpoint, not a name on a
    blocklist, and the distinction matters because ``progen2-base`` -- not in
    ``TG_PANEL`` but loadable -- passes for exactly the same reason ProGen2-medium
    does.
    """

    return spec.modality == "protein" and spec.tokenisation == "residue"


def _tg05_eligible(spec: ArmSpec) -> bool:
    """Residue-addressable, and not conditioned on a label the stage cannot supply."""

    return residue_addressable(spec) and spec.input_format != "ec_conditioned"


def _gpt2_eager_attention(spec: ArmSpec) -> bool:
    """Attention routes through ``transformers.models.gpt2.eager_attention_forward``.

    ``tg06`` monkeypatches that function to capture and inject patterns. ProGen2
    ships its own modelling code with the checkpoint, so the patch never reaches
    it -- which the stage only discovered *after* loading the model.
    """

    return spec.architecture == "gpt2"

TG_STAGES: dict[str, TgStage] = {
    "tg00": TgStage(
        name="tg00",
        entry_point="tg00_input_contract.py",
        scope="multi_arm",
        protein_bands=(
            ProteinBand(
                "render",
                (600, 2000),
                "the rendering control prices the FASTA wrap, which only exists "
                "above 60 residues per line and is priced against EXP-R2-028's "
                "600-2000 cohort so the two numbers are comparable",
            ),
            ProteinBand(
                "cohort",
                (200, 800),
                "the cohort control prices file order, and the head of Swiss-Prot "
                "is only unrepresentative on a band wide enough to contain a "
                "family block; drawn narrower than the rendering band so the two "
                "controls are not measured on one cohort and read as one number",
            ),
        ),
        arms=("protgpt2", "progen2-medium"),
        arms_reason=(
            "the two arms whose rendering the input contract certifies: ProtGPT2 "
            "carries the FASTA wrap worth 1.42-1.78 nats/token (L11) and "
            "ProGen2-medium carries the N-to-C control token, whose omission, "
            "reversal and misplacement the rendering control now prices. ZymCTRL's "
            "rendering is certified by its EC-tag leak measurement (EXP-R2-034) "
            "instead"
        ),
        notes=(
            "the positive-control stage the audit's plan item B1 names; it must be "
            "run before any TG number is quoted"
        ),
    ),
    "tg01": TgStage(
        name="tg01",
        entry_point="tg01_information_budget.py",
        scope="per_arm",
        protein_bands=(
            ProteinBand(
                "res",
                (400, 1000),
                "the truncation curve reaches 128 tokens of visible context, so a "
                "cohort must be long enough that the longest context is still "
                "interior to the sequence on a residue-level arm",
            ),
        ),
    ),
    "tg02": TgStage(
        name="tg02",
        entry_point="tg02_order_vs_composition.py",
        scope="per_arm",
        protein_bands=(
            ProteinBand(
                "res", (400, 1000), "shares TG-01's cohort so the two are read together"
            ),
        ),
    ),
    "tg03": TgStage(
        name="tg03",
        entry_point="tg03_matched_sae.py",
        scope="per_arm",
        protein_bands=(ProteinBand("res", REFERENCE_PROTEIN_BAND),),
    ),
    "tg04": TgStage(
        name="tg04",
        entry_point="tg04_explanation_channel.py",
        scope="armless",
        notes=(
            "measures the annotation channel's bits per symbol, which is a property "
            "of Pfam and the structural oracle and not of any model. This is the L9 "
            "stage. Its text row is not measured here at all: it is read from the "
            "TG-01 artefact, which is why that stage must have run first"
        ),
    ),
    "tg05": TgStage(
        name="tg05",
        entry_point="tg05_relational_channel.py",
        scope="per_arm",
        protein_bands=(
            ProteinBand(
                "res",
                (110, 320),
                "a structure cohort, not a sequence cohort: the band is set by what "
                "an all-pairs contact map and an all-layer attention stack cost at "
                "L^2, not by what the series reads its cross-entropies on. It is a "
                "band nonetheless, and this table used to assert the stage had none",
            ),
        ),
        arm_predicate=_tg05_eligible,
        arms_reason=(
            "an anchored residue-pair readout needs a residue-to-token map, which "
            "only a residue-level tokenisation provides, and it has no per-structure "
            "EC label, so an ec_conditioned arm could only be scored off its own "
            "training distribution. Both facts are ArmSpec fields. Declared here "
            "rather than raised mid-run: the stage used to refuse protgpt2 by name "
            "and kill gpt2-large inside its encoder after paying for a model load, "
            "while telling tg99 it would produce four arms"
        ),
    ),
    "tg06": TgStage(
        name="tg06",
        entry_point="tg06_frozen_attention.py",
        scope="per_arm",
        protein_bands=(
            ProteinBand(
                "res",
                (400, 1000),
                "freezing an attention pattern is only informative where there is a "
                "pattern to freeze; short sequences have too few positions",
            ),
        ),
        arm_predicate=_gpt2_eager_attention,
        arms_reason=(
            "pattern capture and injection monkeypatch "
            "transformers.models.gpt2.eager_attention_forward, which reaches every "
            "gpt2-architecture arm and nothing else. ProGen2 ships its own modelling "
            "code, so the patch never fires; the stage detects that, but only after "
            "loading the model, and tg99 was told to expect an artefact that could "
            "not exist"
        ),
    ),
    "tg07": TgStage(
        name="tg07",
        entry_point="tg07_variance_behaviour.py",
        scope="per_arm",
        protein_bands=(ProteinBand("res", REFERENCE_PROTEIN_BAND),),
    ),
    "tg08": TgStage(
        name="tg08",
        entry_point="tg08_budget_sweep.py",
        scope="per_arm",
        protein_bands=(ProteinBand("res", REFERENCE_PROTEIN_BAND),),
    ),
    "tg09": TgStage(
        name="tg09",
        entry_point="tg09_depth_profile.py",
        scope="per_arm",
        protein_bands=(ProteinBand("res", REFERENCE_PROTEIN_BAND),),
    ),
    "tg10": TgStage(
        name="tg10",
        entry_point="tg10_causal_headroom.py",
        scope="per_arm",
        protein_bands=(
            ProteinBand(
                "res",
                (64, 246),
                "NARROWER THAN THE REFERENCE BAND, and deliberately: this stage's "
                "estimand is the P0-2b one, whose production qualification ran on "
                "64-246, and the whole point of the stage is to price that estimand "
                "on the cohort it was actually gated on. It means TG-10's headroom "
                "is not commensurate with TG-03/07/08/09's loss-recovered numbers",
            ),
        ),
    ),
    "tg99": TgStage(
        name="tg99",
        entry_point="tg99_summarize.py",
        scope="summary",
        notes="reads the other stages' artefacts; measures nothing itself",
    ),
}

STAGE_ORDER: tuple[str, ...] = tuple(TG_STAGES)


def _check_stages() -> None:
    for key, stage in TG_STAGES.items():
        if stage.name != key:
            raise AssertionError(f"stage {key!r} declares name {stage.name!r}")
        if stage.scope not in SCOPES:
            raise AssertionError(f"stage {key!r} declares unknown scope {stage.scope!r}")
        if not (_STAGE_DIR / stage.entry_point).is_file():
            raise AssertionError(f"stage {key!r} names a missing entry point")
        prefixes = [band.argument_prefix for band in stage.protein_bands]
        if len(set(prefixes)) != len(prefixes):
            raise AssertionError(
                f"stage {key!r} declares two bands on one argument pair {prefixes}"
            )
        for band in stage.protein_bands:
            low, high = band.residues
            if low < 1 or high <= low:
                raise AssertionError(f"stage {key!r} declares an empty residue band")
            if not band.matches_reference and not band.reason:
                raise AssertionError(
                    f"stage {key!r} draws on {band.residues}, not the reference "
                    f"band {REFERENCE_PROTEIN_BAND}, without saying why. An undeclared "
                    "band lets a verdict be read as covering a population it was never "
                    "measured on (Appendix B rule 13)"
                )
        if stage.arms is not None:
            unknown = [a for a in stage.arms if a not in PANEL]
            if unknown:
                raise AssertionError(f"stage {key!r} declares unknown arms {unknown}")
            if not stage.arms_reason:
                raise AssertionError(
                    f"stage {key!r} narrows the arm set without saying why"
                )
            if not stage.arms:
                raise AssertionError(
                    f"stage {key!r} admits no arm at all, so no campaign can ever "
                    "satisfy it"
                )


def _check_tg_panel() -> None:
    unknown = [name for name in TG_PANEL if name not in PANEL]
    if unknown:
        raise AssertionError(f"TG_PANEL names arms outside src.transfer.arms.PANEL: {unknown}")


_check_tg_panel()
_check_stages()


def refuse_unless_eligible(stage: str, arm: str) -> None:
    """Stop before a model load if this stage cannot measure this arm.

    Called as the first statement of the stages that have an eligibility rule, so
    a refusal costs a dictionary lookup rather than a checkpoint load and a
    partial forward pass. ``tg05`` used to spend both and then raise, and ``tg06``
    raised only after capturing attention patterns that were never going to
    exist.
    """

    contract = TG_STAGES[stage]
    if contract.eligible(arm):
        return
    raise SystemExit(
        f"{stage}: {arm} is not eligible for this stage. {contract.arms_reason}. "
        f"On the TG panel this stage measures {list(contract.arms or ())}."
    )


# ------------------------------------------------------ reading the entry points


def argparse_defaults(entry_point: str) -> dict[str, Any]:
    """Every ``ap.add_argument("--x", ..., default=D)`` in a stage, as ``{"x": D}``.

    Read statically. A repeated option string is reported rather than silently
    overwritten, because that is what ``tg01_information_budget.py`` carried and
    it made the stage unrunnable: ``argparse`` raises on a duplicate, so the
    script died on parser construction, before its first line of measurement.
    """

    path = _STAGE_DIR / entry_point
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
        ):
            continue
        flags = [
            a.value
            for a in node.args
            if isinstance(a, ast.Constant) and isinstance(a.value, str)
        ]
        long = next((f for f in flags if f.startswith("--")), None)
        if long is None:
            continue
        key = long[2:].replace("-", "_")
        if key in found:
            raise ValueError(
                f"{entry_point} registers --{long[2:]} twice on one parser; "
                "argparse raises on a duplicate option string, so this entry point "
                "cannot run at all"
            )
        default = next(
            (kw.value for kw in node.keywords if kw.arg == "default"), None
        )
        if default is None:
            found[key] = None
        elif isinstance(default, ast.Constant):
            found[key] = default.value
        elif isinstance(default, ast.Name):
            found[key] = f"<name:{default.id}>"
        else:
            found[key] = "<expression>"
    return found


def residue_bound_prefixes(defaults: dict[str, Any]) -> set[str]:
    """Every ``--x-min/--x-max`` (or ``--min-x/--max-x``) pair a stage declares.

    Matched on *shape*, not on the literal names ``res_min``/``res_max``. The
    name-based lookup this replaces saw no band at all in ``tg00``
    (``--render-min/--render-max`` and ``--cohort-min/--cohort-max``) or in
    ``tg05`` (``--min-len/--max-len``), so three live residue bands sat undeclared
    inside the mechanism whose entire purpose is that no band sits undeclared.

    Nothing is excluded from the match. ``--max-len`` is a token truncation in
    eight stages, but none of them carries a ``--min-len`` to pair it with now
    that TG-05 spells its band like everything else -- and an exclusion list keyed
    on ``len`` would have hidden exactly the stage this check was blind to.
    """

    found = set()
    for key in defaults:
        if key.endswith("_min") and f"{key[:-4]}_max" in defaults:
            found.add(key[:-4])
        elif key.startswith("min_") and f"max_{key[4:]}" in defaults:
            found.add(key[4:])
    return found


def _observed_band(defaults: dict[str, Any], prefix: str) -> tuple[Any, Any] | None:
    for low, high in ((f"{prefix}_min", f"{prefix}_max"), (f"min_{prefix}", f"max_{prefix}")):
        if low in defaults and high in defaults:
            return defaults[low], defaults[high]
    return None


def _band_options(defaults: dict[str, Any], prefix: str) -> str:
    """How a stage spells one band pair, so a failure message is copy-pasteable."""

    if f"min_{prefix}" in defaults:
        return f"--min-{prefix}/--max-{prefix}"
    return f"--{prefix}-min/--{prefix}-max"


def verify() -> list[str]:
    """Disagreements between this table and the entry points; empty means clean."""

    problems: list[str] = []
    for name, stage in TG_STAGES.items():
        try:
            defaults = argparse_defaults(stage.entry_point)
        except ValueError as error:
            problems.append(f"{name}: {error}")
            continue

        if stage.scope == "summary":
            continue

        seed = defaults.get("seed")
        if seed is None:
            problems.append(f"{name}: {stage.entry_point} declares no --seed default")
        elif seed != "<name:DEFAULT_COHORT_SEED>":
            problems.append(
                f"{name}: --seed defaults to {seed!r} rather than "
                "tg_common.DEFAULT_COHORT_SEED. The constant exists so that `skip` "
                "partitions one permutation across stages; a stage on its own seed "
                "draws a different ordering and its skip-disjointness is a fiction"
            )

        declared = {band.argument_prefix: band.residues for band in stage.protein_bands}
        observed_prefixes = residue_bound_prefixes(defaults)
        for prefix in sorted(observed_prefixes - set(declared)):
            problems.append(
                f"{name}: {stage.entry_point} draws a residue band "
                f"{_band_options(defaults, prefix)} = {_observed_band(defaults, prefix)} "
                "that TG_STAGES does not declare (Appendix B rule 13)"
            )
        for prefix in sorted(set(declared) - observed_prefixes):
            problems.append(
                f"{name}: TG_STAGES declares a band on --{prefix}-min/--{prefix}-max "
                f"that {stage.entry_point} does not carry"
            )
        for prefix in sorted(observed_prefixes & set(declared)):
            observed = _observed_band(defaults, prefix)
            if observed != declared[prefix]:
                problems.append(
                    f"{name}: {stage.entry_point} draws "
                    f"{_band_options(defaults, prefix)} = {observed} but TG_STAGES "
                    f"declares {declared[prefix]}"
                )

        if stage.scope == "per_arm" and "arm" not in defaults:
            problems.append(f"{name}: declared per_arm but has no --arm argument")
        if stage.scope == "multi_arm" and "arms" not in defaults:
            problems.append(f"{name}: declared multi_arm but has no --arms argument")
        if stage.scope == "armless" and ("arm" in defaults or "arms" in defaults):
            problems.append(f"{name}: declared armless but takes an arm argument")
    return problems


def _band_records(contract: TgStage) -> list[dict[str, Any]]:
    return [
        {
            "argument_prefix": band.argument_prefix,
            "protein_residues": list(band.residues),
            "matches_reference": band.matches_reference,
            "reason": band.reason or None,
        }
        for band in contract.protein_bands
    ]


def stage_contract_record(stage: str, arms: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """The block a TG stage writes into its artefact to declare what it measured.

    The counterpart of ``panel_contract.stage_contract_record``. Two facts that
    were previously only recoverable by comparing argparse defaults across eleven
    files: the residue band this stage drew on beside the band the series is read
    on, and whether the arms measured are the ones the recorded table covers.

    **It is now actually called.** This function existed, was tested, and no stage
    invoked it, so not one artefact in the corrected tree carries a ``cohort_band``
    key -- the declaration lived only in this file, which is the same shape as the
    band being undeclared. Every measuring stage writes it now.
    """

    contract = TG_STAGES[stage]
    measured = list(arms)
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "scope": contract.scope,
        "cohort_seed": DEFAULT_COHORT_SEED,
        "arm_selection": {
            "measured": measured,
            "declared": None if contract.arms is None else list(contract.arms),
            "tg_panel": list(TG_PANEL),
            "outside_tg_panel": [a for a in measured if a not in TG_PANEL],
            "ineligible": [a for a in measured if a in PANEL and not contract.eligible(a)],
            "caveat": (
                "an arm outside TG_PANEL is loadable and will produce a number, but "
                "that number is not comparable to the recorded TG table"
            ),
        },
        "cohort_band": {
            "protein_residue_bands": _band_records(contract),
            "reference_protein_residues": list(REFERENCE_PROTEIN_BAND),
        },
    }


def contract_payload() -> dict[str, Any]:
    """The whole resolved contract, as a record a run manifest can embed."""

    return {
        "schema_version": SCHEMA_VERSION,
        "tg_panel": list(TG_PANEL),
        "cohort_seed": DEFAULT_COHORT_SEED,
        "reference_protein_residue_band": list(REFERENCE_PROTEIN_BAND),
        "stage_order": list(STAGE_ORDER),
        "stages": {
            name: {
                "entry_point": stage.entry_point,
                "scope": stage.scope,
                "protein_residue_bands": _band_records(stage),
                "declared_arms": None if stage.arms is None else list(stage.arms),
                "declared_arms_reason": stage.arms_reason or None,
                "arms_derived_from_arm_spec": stage.arm_predicate is not None,
                "notes": stage.notes or None,
            }
            for name, stage in TG_STAGES.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify", action="store_true", help="fail if a stage disagrees with the table"
    )
    parser.add_argument("--json", action="store_true", help="print the resolved contract")
    args = parser.parse_args()

    if args.json:
        print(json.dumps(contract_payload(), indent=2, sort_keys=True))
    if args.verify:
        problems = verify()
        if problems:
            for line in problems:
                print(line, file=sys.stderr)
            raise SystemExit(2)
        print(f"{len(TG_STAGES)} TG stages agree with tg_contract.py")
    if not (args.verify or args.json):
        parser.error("nothing to do: pass --verify or --json")


if __name__ == "__main__":
    main()
