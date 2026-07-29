"""The TG series' stage contract and cohort machinery, and the defects they close.

``scripts/transfer_gap/tg_contract.py`` is to the TG stages what
``scripts/transfer/panel_contract.py`` is to the campaign stages: one declaration
of what population each stage measures, checked against the stages themselves
rather than restated and hoped for.

These tests assert the *properties* the contract restores, and each negative path
here corresponds to a defect that was live in this directory:

* ``tg01_information_budget.py`` registered ``--seed`` twice on one parser, so it
  raised ``argparse.ArgumentError`` on construction and could not run at all;
* nine stages restated ``DEFAULT_COHORT_SEED`` by hand and one restated it wrong,
  as the pre-correction ``20260724``, which silently makes ``skip`` a
  non-partition across stages;
* six protein residue bands were in use and three of them were undeclared,
  invisible to a band check that matched on the literal names
  ``res_min``/``res_max``; that is the shape Appendix B rule 13 forbids;
* ``stage_contract_record`` was written, tested, and called by nothing;
* two stages declared no arm restriction and then hard-refused arms by name in
  their own bodies, which made strict TG-99 unsatisfiable;
* ``cohort_provenance`` asserted ``selection: "file_order"`` where the truth was
  "unrecorded", and cohorts were drawn under a seed but consumed in corpus order;
* TG-07 and TG-09 published the retracted all-position residual spectrum under
  the names a reader picks up by default.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = REPO_ROOT / "scripts" / "transfer_gap"
for _path in (str(REPO_ROOT), str(STAGE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import tg07_variance_behaviour as tg07  # noqa: E402
import tg_common  # noqa: E402
import tg_contract  # noqa: E402
from src.transfer.arms import PANEL, Cohort, sampling_record  # noqa: E402
from tg_common import DEFAULT_COHORT_SEED, TG_PANEL  # noqa: E402


@dataclass
class _StubTokenizer:
    """Decodes a token id to a one-character piece; ids >= 100 are separators."""

    def decode(self, ids: list[int]) -> str:
        return "\n" if ids[0] >= 100 else "ACDEFGHIKLMNPQRSTVWY"[ids[0] % 20]


@dataclass
class _StubArm:
    """Enough of an ``Arm`` for the position-mask and provenance helpers."""

    name: str = "protgpt2"
    modality: str = "protein"
    tokenizer: _StubTokenizer = None

    def __post_init__(self) -> None:
        self.tokenizer = self.tokenizer or _StubTokenizer()


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
        for band in stage.protein_bands:
            if not band.matches_reference:
                assert band.reason, f"{name}:{band.argument_prefix}"


def test_every_band_actually_in_use_is_declared():
    """They differ by more than a rounding: 64-246 and 400-1000 share no protein.

    Updated from the three-band version of this test, which was true of the
    ``--res-min/--res-max`` stages only. TG-00 draws two more and TG-05 a sixth,
    and the contract's own check could not see any of the three.
    """

    bands = {
        band.residues
        for stage in tg_contract.TG_STAGES.values()
        for band in stage.protein_bands
    }
    assert bands == {
        (64, 246),
        (110, 320),
        (120, 1000),
        (200, 800),
        (400, 1000),
        (600, 2000),
    }


def test_the_band_check_finds_a_pair_however_the_stage_spells_it():
    """The name-based lookup this replaces was blind to two live stages."""

    assert tg_contract.residue_bound_prefixes(
        tg_contract.argparse_defaults("tg00_input_contract.py")
    ) == {"render", "cohort"}
    assert tg_contract.residue_bound_prefixes(
        tg_contract.argparse_defaults("tg05_relational_channel.py")
    ) == {"res"}
    assert tg_contract.residue_bound_prefixes({"min_len": 1, "max_len": 2}) == {"len"}
    # --max-len alone is a token truncation and must not read as a band.
    assert tg_contract.residue_bound_prefixes({"max_len": 256}) == set()


def test_a_band_in_the_code_that_the_table_omits_is_refused():
    """TG-00's 600-2000 and 200-800 were both live and both undeclared."""

    stripped = replace(tg_contract.TG_STAGES["tg00"], protein_bands=())
    original = dict(tg_contract.TG_STAGES)
    tg_contract.TG_STAGES["tg00"] = stripped
    try:
        problems = tg_contract.verify()
    finally:
        tg_contract.TG_STAGES.clear()
        tg_contract.TG_STAGES.update(original)
    assert any("does not declare" in problem for problem in problems), problems


def test_a_band_in_the_table_that_the_code_dropped_is_refused():
    extra = replace(
        tg_contract.TG_STAGES["tg03"],
        protein_bands=tg_contract.TG_STAGES["tg03"].protein_bands
        + (tg_contract.ProteinBand("ghost", (10, 20), "probe"),),
    )
    original = dict(tg_contract.TG_STAGES)
    tg_contract.TG_STAGES["tg03"] = extra
    try:
        problems = tg_contract.verify()
    finally:
        tg_contract.TG_STAGES.clear()
        tg_contract.TG_STAGES.update(original)
    assert any("does not carry" in problem for problem in problems), problems


def test_an_undeclared_off_reference_band_is_refused():
    off_band = replace(
        tg_contract.TG_STAGES["tg03"],
        protein_bands=(tg_contract.ProteinBand("res", (64, 246)),),
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
    bad = replace(
        tg_contract.TG_STAGES["tg03"],
        protein_bands=(tg_contract.ProteinBand("res", (1000, 120), "probe"),),
    )
    original = dict(tg_contract.TG_STAGES)
    tg_contract.TG_STAGES["tg03"] = bad
    try:
        with pytest.raises(AssertionError, match="empty residue band"):
            tg_contract._check_stages()
    finally:
        tg_contract.TG_STAGES.clear()
        tg_contract.TG_STAGES.update(original)


def test_every_measuring_stage_writes_its_contract_record():
    """``stage_contract_record`` existed, was tested, and nothing called it.

    Not one artefact in the corrected tree carries a ``cohort_band`` key, which
    is the same defect as the band being undeclared: the declaration existed only
    in ``tg_contract.py``.
    """

    for name, stage in tg_contract.TG_STAGES.items():
        if stage.scope == "summary":
            continue
        source = (STAGE_DIR / stage.entry_point).read_text(encoding="utf-8")
        assert f'stage_contract_record("{name}"' in source, name


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
    (band,) = tg_contract.stage_contract_record("tg10", ["protgpt2"])["cohort_band"][
        "protein_residue_bands"
    ]
    assert band["protein_residues"] == [64, 246]
    assert band["matches_reference"] is False
    assert "P0-2b" in band["reason"]

    (matched,) = tg_contract.stage_contract_record("tg03", ["protgpt2"])["cohort_band"][
        "protein_residue_bands"
    ]
    assert matched["matches_reference"] is True


def test_the_artefact_record_carries_every_band_of_a_two_band_stage():
    """TG-00's rendering and cohort controls are two populations, not one."""

    bands = tg_contract.stage_contract_record("tg00", ["protgpt2"])["cohort_band"][
        "protein_residue_bands"
    ]
    assert [band["protein_residues"] for band in bands] == [[600, 2000], [200, 800]]
    assert tg_contract.TG_STAGES["tg00"].protein_band is None


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


# --------------------------------------------- eligibility, declared not hidden


def test_the_two_stages_that_refuse_arms_declare_which_ones():
    """They declared ``arms=None`` and then raised inside their own bodies.

    ``tg05`` can produce one artefact of four and ``tg06`` three, so strict TG-99
    could not be satisfied by a *fully executed* campaign -- which made
    ``--allow-partial`` mandatory and the strict default decorative.
    """

    assert tg_contract.TG_STAGES["tg05"].arms == ("progen2-medium",)
    assert tg_contract.TG_STAGES["tg06"].arms == ("gpt2-large", "protgpt2", "zymctrl")


def test_eligibility_is_a_property_of_the_arm_not_its_name():
    """The refusals were literals: ``if arm.name == "protgpt2"``.

    ``progen2-base`` is outside the TG panel and is admitted by TG-05 for exactly
    the reason ``progen2-medium`` is -- residue-level tokenisation and an input
    format that carries no conditioning label.
    """

    tg05, tg06 = tg_contract.TG_STAGES["tg05"], tg_contract.TG_STAGES["tg06"]
    assert tg05.eligible("progen2-base") is True
    assert tg05.eligible("protgpt2") is False       # multi-residue BPE
    assert tg05.eligible("zymctrl") is False        # ec_conditioned
    assert tg05.eligible("gpt2-large") is False     # not a protein arm
    assert tg06.eligible("dialogpt-small") is True  # gpt2 architecture
    assert tg06.eligible("progen2-medium") is False # ships its own attention

    for name in ("protgpt2", "gpt2-large", "zymctrl", "progen2-base"):
        assert PANEL[name].name == name


def test_a_declared_arm_set_that_contradicts_its_predicate_is_refused():
    from tg_contract import TgStage

    with pytest.raises(AssertionError, match="eligibility predicate admits"):
        TgStage(
            name="probe",
            entry_point="tg05_relational_channel.py",
            scope="per_arm",
            arm_predicate=lambda spec: spec.modality == "text",
            arms=("progen2-medium",),
            arms_reason="probe",
        )


def test_refusing_an_ineligible_arm_costs_nothing_and_says_why():
    with pytest.raises(SystemExit) as refusal:
        tg_contract.refuse_unless_eligible("tg05", "protgpt2")
    assert "residue-to-token map" in str(refusal.value)
    tg_contract.refuse_unless_eligible("tg05", "progen2-medium")
    tg_contract.refuse_unless_eligible("tg06", "gpt2-large")


def test_no_stage_narrows_its_arm_set_to_nothing():
    """A stage that admits no arm makes strict mode permanently unsatisfiable."""

    for name, stage in tg_contract.TG_STAGES.items():
        if stage.arms is not None:
            assert stage.arms, name


def test_the_stages_do_not_refuse_arms_by_name():
    """The refusal that mattered was ``arm.name == "protgpt2"`` in a stage body.

    A literal arm name inside a ``raise``/``SystemExit`` is eligibility declared
    where no contract can read it, which is how ``tg99`` came to expect artefacts
    that could not exist.
    """

    for stage in tg_contract.TG_STAGES.values():
        source = (STAGE_DIR / stage.entry_point).read_text(encoding="utf-8")
        for line in source.splitlines():
            code = line.split("#", 1)[0]
            if "arm.name ==" not in code:
                continue
            assert not any(f'"{arm}"' in code for arm in TG_PANEL), (
                f"{stage.entry_point}: {line.strip()}"
            )


# ------------------------------------------ the spectrum a stage is allowed to name


def _sink_dominated_activations(
    n_special: int = 400, n_interior: int = 3600, d: int = 8, seed: int = 0
):
    """Activations whose all-position spectrum is one direction, and whose
    interior alphabet-bearing subset is isotropic.

    A caricature of the measured panel, and in the same direction: at relative
    depth 0.5 GPT-2-large reads PC1 0.809 over all positions against 0.034 on
    interior alphabet-bearing positions, and ProtGPT2 0.971 against 0.439.
    """

    rng = np.random.default_rng(seed)
    total = n_special + n_interior
    acts = torch.tensor(rng.normal(size=(total, d)), dtype=torch.float32)
    acts[:n_special, 0] += 500.0  # the sink / separator direction
    half = n_special // 2
    positions = torch.cat(
        [
            torch.zeros(half, dtype=torch.long),
            torch.arange(1, total - half + 1, dtype=torch.long),
        ]
    )
    # ids >= 100 decode to a newline; the second half of the special block is a
    # separator sitting at an interior position, which is ProtGPT2's FASTA wrap.
    tokens = torch.cat(
        [
            torch.zeros(half, dtype=torch.long),
            torch.full((half,), 100, dtype=torch.long),
            torch.arange(n_interior, dtype=torch.long) % 20,
        ]
    )
    return acts, positions, tokens


def test_the_named_spectrum_is_the_interior_alphabet_bearing_one():
    """Appendix B rule 11, applied to the field names and not only to the prose.

    The all-position spectrum was ``participation_ratio``, ``variance_top1`` and
    ``rank_for_90pct_variance`` in every TG-07 artefact, and the audit retracted
    the comparison built on it; the interior values existed only nested inside
    ``spectrum_by_position_subset``, where nothing downstream read them.
    """

    arm = _StubArm()
    acts, positions, tokens = _sink_dominated_activations()
    subsets = tg07.position_subsets(arm, positions, tokens)
    spectra = tg07.subset_spectra(arm, acts, subsets)

    assert spectra["all_positions"]["variance_top1"] > 0.9
    assert spectra[tg07.PRIMARY_SUBSET]["variance_top1"] < 0.4
    # One effective dimension over all positions; the full eight on the interior
    # alphabet-bearing subset. The panel's real numbers span 1.06 -> 4.86
    # (ProtGPT2) and 1.53 -> 253.4 (GPT-2-large).
    assert spectra["all_positions"]["participation_ratio"] < 1.1
    assert spectra[tg07.PRIMARY_SUBSET]["participation_ratio"] > 0.9 * acts.shape[1]

    evals, _ = tg07.spectrum_of(acts)
    share_all = (evals / evals.sum()).numpy()
    fields = tg07.spectrum_fields(spectra, share_all, evals)

    primary = spectra[tg07.PRIMARY_SUBSET]
    assert tg07.PRIMARY_SUBSET == "interior_symbol_positions"
    assert fields["spectrum_positions"] == tg07.PRIMARY_SUBSET
    for key in ("participation_ratio", "variance_top1", "variance_top10", "variance_top64"):
        assert fields[key] == primary[key]
        assert fields[f"{key}_all_positions"] != primary[key]
    assert fields["variance_top1_all_positions"] == pytest.approx(
        spectra["all_positions"]["variance_top1"], rel=1e-9
    )
    assert "attention sink" in fields["all_position_spectrum_hazard"]
    assert fields["spectrum_by_position_subset"] == spectra


def test_a_subset_too_small_to_carry_a_spectrum_is_refused():
    arm = _StubArm()
    acts, positions, tokens = _sink_dominated_activations(n_special=20, n_interior=80)
    subsets = tg07.position_subsets(arm, positions, tokens)
    with pytest.raises(RuntimeError, match="activations"):
        tg07.subset_spectra(arm, acts, subsets)


def test_variance_along_an_external_basis_is_the_subset_variance():
    """The sweep splices the all-position basis, so the variance it reports has
    to be the variance *of the representation* that basis keeps. Those are two
    objects, and the sweep used to report the wrong one's own spectrum."""

    acts, _, _ = _sink_dominated_activations()
    cov = tg07.covariance_of(acts)
    identity = torch.eye(acts.shape[1])
    assert torch.allclose(
        tg07.variance_along(identity, cov), torch.diagonal(cov).double(), atol=1e-3
    )


def test_both_variance_stages_publish_through_one_naming_declaration():
    """TG-09 had no subset diagnostic at all: it called ``collect`` as
    ``acts, _, _``. A naming rule restated in two files diverges in one."""

    for entry_point in ("tg07_variance_behaviour.py", "tg09_depth_profile.py"):
        source = (STAGE_DIR / entry_point).read_text(encoding="utf-8")
        assert "spectrum_fields(" in source, entry_point
        assert "acts, _, _ = collect" not in source, entry_point


# ------------------------------------------------- what a cohort record admits


def test_an_unrecorded_draw_is_reported_as_unrecorded_not_as_file_order():
    """``sampling.get("mode", "file_order")`` substituted the most hazardous
    specific claim in this programme for an absence, and emitted it beside
    ``sampling: null``. ``Cohort.sampling`` already declared the right answer
    one import away."""

    hand_built = Cohort(
        name="single_record",
        kind="protein",
        records=["ACDEF"],
        min_symbols=5,
        max_symbols=5,
    )
    record = tg_common.cohort_provenance(hand_built, _StubArm())
    assert record["selection"] == "unrecorded"
    assert record["seed"] is None
    assert "not knowable from the artefact" in record["sampling"]["hazard"]


def test_a_file_order_draw_is_still_reported_as_file_order():
    """The value is only wrong when it is invented; a real file-order cohort
    must keep saying so, with its hazard."""

    cohort = Cohort(
        name="head",
        kind="protein",
        records=["ACDEF"],
        min_symbols=5,
        max_symbols=5,
        metadata={
            "sampling": sampling_record(
                seed=None, skip=0, requested=1, eligible=None, corpus="plain_swissprot"
            )
        },
    )
    record = tg_common.cohort_provenance(cohort, _StubArm())
    assert record["selection"] == "file_order"
    assert "near-clonal homologues" in record["sampling"]["hazard"]


# ------------------------------------- a seeded set is not a seeded prefix


def _corpus_ordered_cohort(n: int = 40, with_labels: bool = False) -> Cohort:
    metadata: dict = {
        "sampling": sampling_record(
            seed=DEFAULT_COHORT_SEED,
            skip=0,
            requested=n,
            eligible=1000,
            corpus="plain_swissprot",
        )
    }
    if with_labels:
        metadata["ec_labels"] = [f"EC{i}" for i in range(n)]
    return Cohort(
        name="drawn",
        kind="protein",
        records=[f"SEQ{i}" for i in range(n)],
        min_symbols=1,
        max_symbols=99,
        metadata=metadata,
    )


def test_a_seeded_draw_is_returned_in_seeded_order_not_corpus_order():
    """``arms.selected_positions`` sorts its draw back into ascending corpus
    order, so six stages that slice or short-circuit over a cohort were reading
    the corpus-earliest part of a seeded set -- rule 1 wearing a seed."""

    drawn = _corpus_ordered_cohort()
    ordered = tg_common.in_seeded_record_order(drawn, DEFAULT_COHORT_SEED)

    assert sorted(ordered.records) == sorted(drawn.records)
    assert ordered.records != drawn.records
    assert ordered.metadata["record_order"]["mode"] == (
        "seeded_permutation_of_the_drawn_set"
    )
    assert ordered.metadata["record_order"]["seed"] == DEFAULT_COHORT_SEED
    # Reproducible, so a digest still identifies a cohort.
    assert (
        tg_common.in_seeded_record_order(drawn, DEFAULT_COHORT_SEED).records
        == ordered.records
    )
    assert (
        tg_common.in_seeded_record_order(drawn, DEFAULT_COHORT_SEED + 1).records
        != ordered.records
    )


def test_the_first_slice_of_a_reordered_cohort_is_not_the_corpus_head():
    drawn = _corpus_ordered_cohort(n=200)
    ordered = tg_common.in_seeded_record_order(drawn, DEFAULT_COHORT_SEED)
    head = set(ordered.records[:20])
    assert len(head & set(drawn.records[:20])) < 10


def test_ec_labels_are_reordered_with_their_sequences():
    """A label permuted independently of its record feeds ZymCTRL another
    protein's conditioning tag, and that tag is a 1.73-nat prompt (EXP-R2-034)."""

    drawn = _corpus_ordered_cohort(with_labels=True)
    pairs = dict(zip(drawn.records, drawn.metadata["ec_labels"]))
    ordered = tg_common.in_seeded_record_order(drawn, DEFAULT_COHORT_SEED)
    assert dict(zip(ordered.records, ordered.metadata["ec_labels"])) == pairs


def test_a_cohort_whose_labels_do_not_pair_with_its_records_is_refused():
    drawn = _corpus_ordered_cohort(with_labels=True)
    drawn.metadata["ec_labels"] = drawn.metadata["ec_labels"][:-1]
    with pytest.raises(ValueError, match="cannot be reordered together"):
        tg_common.in_seeded_record_order(drawn, DEFAULT_COHORT_SEED)


def test_cohort_for_applies_the_record_order_and_records_it(monkeypatch):
    """The fix has to live at construction; a stage that has to remember is a
    stage that will forget, and six of them did."""

    drawn = _corpus_ordered_cohort(n=60)
    monkeypatch.setattr(tg_common, "text_cohort", lambda *a, **k: drawn)
    built = tg_common.cohort_for(
        _StubArm(name="gpt2-large", modality="text"),
        60,
        120,
        1000,
        seed=DEFAULT_COHORT_SEED,
    )
    assert built.records != drawn.records
    assert sorted(built.records) == sorted(drawn.records)

    provenance = tg_common.cohort_provenance(built, _StubArm(name="gpt2-large"))
    assert provenance["record_order"]["seed"] == DEFAULT_COHORT_SEED
    assert provenance["selection"] == "seeded_permutation"
