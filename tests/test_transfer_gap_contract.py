"""The TG series' stage contract, and the defects it was built to catch.

``scripts/transfer_gap/tg_contract.py`` is to the TG stages what
``scripts/transfer/panel_contract.py`` is to the campaign stages: one declaration
of what population each stage measures, checked against the stages themselves
rather than restated and hoped for.

These tests assert the *properties* the contract restores, and each negative path
here corresponds to a defect that was live in this directory before EXP-R2-066:

* ``tg01_information_budget.py`` registered ``--seed`` twice on one parser, so it
  raised ``argparse.ArgumentError`` on construction and could not run at all;
* nine stages restated ``DEFAULT_COHORT_SEED`` by hand and one restated it wrong,
  as the pre-correction ``20260724``, which silently makes ``skip`` a
  non-partition across stages;
* three different protein residue bands were in use with none of them declared,
  which is the shape Appendix B rule 13 forbids.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = REPO_ROOT / "scripts" / "transfer_gap"
for _path in (str(REPO_ROOT), str(STAGE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import tg_contract  # noqa: E402
from src.transfer.arms import PANEL  # noqa: E402
from tg_common import DEFAULT_COHORT_SEED, TG_PANEL  # noqa: E402


def test_every_stage_agrees_with_the_contract():
    """The whole point: the table cannot drift away from the code it describes."""

    assert tg_contract.verify() == []


def test_every_declared_entry_point_exists():
    for stage in tg_contract.TG_STAGES.values():
        assert (STAGE_DIR / stage.entry_point).is_file(), stage.entry_point


def test_the_contract_covers_every_stage_in_the_directory():
    """A stage nobody declared is a stage whose band and seed nobody checked."""

    on_disk = {
        path.name
        for path in STAGE_DIR.glob("tg*.py")
        if path.name not in ("tg_common.py", "tg_contract.py")
    }
    declared = {stage.entry_point for stage in tg_contract.TG_STAGES.values()}
    assert on_disk == declared


# ------------------------------------------------------------------ the seed


def test_no_stage_restates_the_cohort_seed_as_a_literal():
    """One stage restated it as 20260724 -- the pre-correction permutation.

    ``DEFAULT_COHORT_SEED`` exists so that ``skip`` partitions one ordering
    across stages. A stage on its own seed draws a different ordering, and its
    skip-disjointness against another stage is then a fiction rather than a
    property.
    """

    for name, stage in tg_contract.TG_STAGES.items():
        if stage.scope == "summary":
            continue
        defaults = tg_contract.argparse_defaults(stage.entry_point)
        assert defaults.get("seed") == "<name:DEFAULT_COHORT_SEED>", name


def test_tg01_can_construct_its_parser():
    """It could not. Two ``--seed`` arguments; argparse raises on the duplicate."""

    defaults = tg_contract.argparse_defaults("tg01_information_budget.py")
    assert defaults["seed"] == "<name:DEFAULT_COHORT_SEED>"


def test_a_duplicated_option_string_is_reported_not_overwritten(tmp_path):
    """A dict-building reader would silently keep the last one and see nothing."""

    stage = STAGE_DIR / "tg_duplicate_probe.py"
    stage.write_text(
        "import argparse\n"
        "def main():\n"
        "    ap = argparse.ArgumentParser()\n"
        '    ap.add_argument("--seed", type=int, default=1)\n'
        '    ap.add_argument("--seed", type=int, default=2)\n',
        encoding="utf-8",
    )
    try:
        with pytest.raises(ValueError, match="cannot run at all"):
            tg_contract.argparse_defaults(stage.name)
    finally:
        stage.unlink()


# ------------------------------------------------------------- the cohort band


def test_a_stage_off_the_reference_band_must_say_why():
    """Appendix B rule 13. An undeclared band lets a verdict be over-read."""

    for name, stage in tg_contract.TG_STAGES.items():
        if stage.protein_band is None:
            continue
        if stage.protein_band != tg_contract.REFERENCE_PROTEIN_BAND:
            assert stage.band_reason, name


def test_the_three_bands_actually_in_use_are_all_declared():
    """They differ by more than a rounding: 64-246 and 400-1000 share no protein."""

    bands = {
        stage.protein_band
        for stage in tg_contract.TG_STAGES.values()
        if stage.protein_band is not None
    }
    assert bands == {(64, 246), (120, 1000), (400, 1000)}


def test_an_undeclared_off_reference_band_is_refused():
    off_band = replace(
        tg_contract.TG_STAGES["tg03"], protein_band=(64, 246), band_reason=""
    )
    original = dict(tg_contract.TG_STAGES)
    tg_contract.TG_STAGES["tg03"] = off_band
    try:
        with pytest.raises(AssertionError, match="without saying why"):
            tg_contract._check_stages()
    finally:
        tg_contract.TG_STAGES.clear()
        tg_contract.TG_STAGES.update(original)


def test_an_empty_band_is_refused():
    bad = replace(tg_contract.TG_STAGES["tg03"], protein_band=(1000, 120))
    original = dict(tg_contract.TG_STAGES)
    tg_contract.TG_STAGES["tg03"] = bad
    try:
        with pytest.raises(AssertionError, match="empty residue band"):
            tg_contract._check_stages()
    finally:
        tg_contract.TG_STAGES.clear()
        tg_contract.TG_STAGES.update(original)


# ------------------------------------------------------------------- the arms


def test_tg_panel_is_a_subset_of_the_model_panel():
    assert set(TG_PANEL) <= set(PANEL)


def test_a_stage_narrowing_the_arm_set_must_say_why():
    for name, stage in tg_contract.TG_STAGES.items():
        if stage.arms is not None:
            assert stage.arms_reason, name


def test_the_artefact_record_names_an_arm_outside_the_tg_panel():
    """``load_arm`` accepts any panel member; the recorded table covers four."""

    record = tg_contract.stage_contract_record("tg03", ["protgpt2", "progen2-base"])
    assert record["arm_selection"]["outside_tg_panel"] == ["progen2-base"]
    assert record["cohort_seed"] == DEFAULT_COHORT_SEED

    clean = tg_contract.stage_contract_record("tg03", list(TG_PANEL))
    assert clean["arm_selection"]["outside_tg_panel"] == []


def test_the_artefact_record_carries_the_band_beside_the_reference():
    record = tg_contract.stage_contract_record("tg10", ["protgpt2"])
    assert record["cohort_band"]["protein_residues"] == [64, 246]
    assert record["cohort_band"]["matches_reference"] is False
    assert "P0-2b" in record["cohort_band"]["reason"]

    matched = tg_contract.stage_contract_record("tg03", ["protgpt2"])
    assert matched["cohort_band"]["matches_reference"] is True


def test_an_unknown_arm_in_a_declared_set_is_refused():
    bad = replace(
        tg_contract.TG_STAGES["tg00"],
        arms=("protgpt2", "not-a-model"),
        arms_reason="probe",
    )
    original = dict(tg_contract.TG_STAGES)
    tg_contract.TG_STAGES["tg00"] = bad
    try:
        with pytest.raises(AssertionError, match="unknown arms"):
            tg_contract._check_stages()
    finally:
        tg_contract.TG_STAGES.clear()
        tg_contract.TG_STAGES.update(original)


def test_the_payload_serialises_every_stage():
    payload = tg_contract.contract_payload()
    assert payload["schema_version"] == tg_contract.SCHEMA_VERSION
    assert set(payload["stages"]) == set(tg_contract.TG_STAGES)
    assert payload["cohort_seed"] == DEFAULT_COHORT_SEED
