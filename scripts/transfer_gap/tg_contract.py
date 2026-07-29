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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_STAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _STAGE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(_STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_STAGE_DIR))

from src.transfer.arms import PANEL  # noqa: E402
from tg_common import DEFAULT_COHORT_SEED, TG_PANEL  # noqa: E402

SCHEMA_VERSION = "r2_transfer_gap_contract_v1"

#: The band on which the TG comparison table in the audit is read. Chosen as
#: 120-1000 because it is the band of TG-03, TG-07, TG-08 and TG-09 -- the
#: dictionary and variance stages the table's load-bearing rows come from -- not
#: because it is better. A stage on a different band is not wrong; it is
#: measuring a different population, and that is what has to be visible.
REFERENCE_PROTEIN_BAND = (120, 1000)


@dataclass(frozen=True)
class TgStage:
    """What one TG stage measures, on which arms, over which residues.

    ``scope``
        ``per_arm`` -- takes ``--arm`` and writes one artefact per arm.
        ``multi_arm`` -- takes ``--arms`` and writes one artefact per arm in one
        process.
        ``armless`` -- takes no arm argument; it measures the corpus, not a model.
        ``summary`` -- reads other stages' artefacts and writes no measurement.

    ``protein_band``
        the ``(--res-min, --res-max)`` this stage's argparse defaults set, or
        ``None`` for a stage that draws no protein cohort. Checked against the
        source by :func:`verify`, so this table cannot drift away from the code
        it describes.

    ``band_reason``
        why it is not :data:`REFERENCE_PROTEIN_BAND`. Required whenever it
        differs, and a stage that differs without one is a contract failure --
        an undeclared band is exactly what rule 13 forbids.
    """

    name: str
    entry_point: str
    scope: str
    protein_band: tuple[int, int] | None = None
    band_reason: str = ""
    arms: tuple[str, ...] | None = None
    arms_reason: str = ""
    notes: str = ""


SCOPES = ("per_arm", "multi_arm", "armless", "summary")

TG_STAGES: dict[str, TgStage] = {
    "tg00": TgStage(
        name="tg00",
        entry_point="tg00_input_contract.py",
        scope="multi_arm",
        arms=("protgpt2", "progen2-medium"),
        arms_reason=(
            "the two arms whose rendering the input contract certifies: ProtGPT2 "
            "carries the FASTA wrap worth 1.42-1.78 nats/token (L11) and "
            "ProGen2-medium carries the N-to-C control token. ZymCTRL's rendering "
            "is certified by its EC-tag leak measurement (EXP-R2-034) instead"
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
        protein_band=(400, 1000),
        band_reason=(
            "the truncation curve reaches 128 tokens of visible context, so a "
            "cohort must be long enough that the longest context is still interior "
            "to the sequence on a residue-level arm"
        ),
    ),
    "tg02": TgStage(
        name="tg02",
        entry_point="tg02_order_vs_composition.py",
        scope="per_arm",
        protein_band=(400, 1000),
        band_reason="shares TG-01's cohort so the two are read together",
    ),
    "tg03": TgStage(
        name="tg03",
        entry_point="tg03_matched_sae.py",
        scope="per_arm",
        protein_band=REFERENCE_PROTEIN_BAND,
    ),
    "tg04": TgStage(
        name="tg04",
        entry_point="tg04_explanation_channel.py",
        scope="armless",
        notes=(
            "measures the annotation channel's bits per symbol, which is a property "
            "of Pfam and the structural oracle and not of any model. This is the L9 "
            "stage; its Pfam and text channels are still corpus-prefix draws and say "
            "so per channel in the artefact"
        ),
    ),
    "tg05": TgStage(
        name="tg05",
        entry_point="tg05_relational_channel.py",
        scope="per_arm",
        notes=(
            "draws from AlphaFold structures rather than a residue band, so it has "
            "no --res-min/--res-max; its cohort bound is a structure count"
        ),
    ),
    "tg06": TgStage(
        name="tg06",
        entry_point="tg06_frozen_attention.py",
        scope="per_arm",
        protein_band=(400, 1000),
        band_reason=(
            "freezing an attention pattern is only informative where there is a "
            "pattern to freeze; short sequences have too few positions"
        ),
    ),
    "tg07": TgStage(
        name="tg07",
        entry_point="tg07_variance_behaviour.py",
        scope="per_arm",
        protein_band=REFERENCE_PROTEIN_BAND,
    ),
    "tg08": TgStage(
        name="tg08",
        entry_point="tg08_budget_sweep.py",
        scope="per_arm",
        protein_band=REFERENCE_PROTEIN_BAND,
    ),
    "tg09": TgStage(
        name="tg09",
        entry_point="tg09_depth_profile.py",
        scope="per_arm",
        protein_band=REFERENCE_PROTEIN_BAND,
    ),
    "tg10": TgStage(
        name="tg10",
        entry_point="tg10_causal_headroom.py",
        scope="per_arm",
        protein_band=(64, 246),
        band_reason=(
            "NARROWER THAN THE REFERENCE BAND, and deliberately: this stage's "
            "estimand is the P0-2b one, whose production qualification ran on "
            "64-246, and the whole point of the stage is to price that estimand on "
            "the cohort it was actually gated on. It means TG-10's headroom is not "
            "commensurate with TG-03/07/08/09's loss-recovered numbers"
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
        if stage.protein_band is not None:
            low, high = stage.protein_band
            if low < 1 or high <= low:
                raise AssertionError(f"stage {key!r} declares an empty residue band")
            if stage.protein_band != REFERENCE_PROTEIN_BAND and not stage.band_reason:
                raise AssertionError(
                    f"stage {key!r} draws on {stage.protein_band}, not the reference "
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


def _check_tg_panel() -> None:
    unknown = [name for name in TG_PANEL if name not in PANEL]
    if unknown:
        raise AssertionError(f"TG_PANEL names arms outside src.transfer.arms.PANEL: {unknown}")


_check_tg_panel()
_check_stages()


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

        declared = stage.protein_band
        observed = (
            (defaults["res_min"], defaults["res_max"])
            if "res_min" in defaults and "res_max" in defaults
            else None
        )
        if observed != declared:
            problems.append(
                f"{name}: {stage.entry_point} draws protein residues {observed} but "
                f"TG_STAGES declares {declared}"
            )

        if stage.scope == "per_arm" and "arm" not in defaults:
            problems.append(f"{name}: declared per_arm but has no --arm argument")
        if stage.scope == "multi_arm" and "arms" not in defaults:
            problems.append(f"{name}: declared multi_arm but has no --arms argument")
        if stage.scope == "armless" and ("arm" in defaults or "arms" in defaults):
            problems.append(f"{name}: declared armless but takes an arm argument")
    return problems


def stage_contract_record(stage: str, arms: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """The block a TG stage writes into its artefact to declare what it measured.

    The counterpart of ``panel_contract.stage_contract_record``. Two facts that
    were previously only recoverable by comparing argparse defaults across eleven
    files: the residue band this stage drew on beside the band the series is read
    on, and whether the arms measured are the ones the recorded table covers.
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
            "tg_panel": list(TG_PANEL),
            "outside_tg_panel": [a for a in measured if a not in TG_PANEL],
            "caveat": (
                "an arm outside TG_PANEL is loadable and will produce a number, but "
                "that number is not comparable to the recorded TG table"
            ),
        },
        "cohort_band": {
            "protein_residues": (
                None if contract.protein_band is None else list(contract.protein_band)
            ),
            "reference_protein_residues": list(REFERENCE_PROTEIN_BAND),
            "matches_reference": (
                None
                if contract.protein_band is None
                else contract.protein_band == REFERENCE_PROTEIN_BAND
            ),
            "reason": contract.band_reason or None,
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
                "protein_residue_band": (
                    None if stage.protein_band is None else list(stage.protein_band)
                ),
                "protein_band_matches_reference": (
                    None
                    if stage.protein_band is None
                    else stage.protein_band == REFERENCE_PROTEIN_BAND
                ),
                "protein_band_reason": stage.band_reason or None,
                "declared_arms": None if stage.arms is None else list(stage.arms),
                "declared_arms_reason": stage.arms_reason or None,
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
