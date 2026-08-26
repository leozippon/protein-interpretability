"""EXP-R2-225's staged second-stage checkpoints: declaration, doors, shapes.

What these pin is not that the staged checkpoints work -- half of them cannot yet
be measured at all -- but that what the registry *says* about each one is true,
and that saying it widened nothing else. Four invariants carry that:

* the first round's door is byte-identical in meaning, so the three stages that
  key on :data:`~src.transfer.arms.STAGED_SCALE_ARMS` refuse exactly what they
  refused before this campaign existed;
* no arm declares a capability its architecture cannot honour, checked against
  the tables that implement each family rather than against a list written here;
* every declared depth, width and scoring alphabet is the one in the
  checkpoint's own ``config.json``, checked against the file;
* the campaign's joint wave is declared as an arm nowhere, which is the rule
  ``21_joint_mode_qualification.py`` states and the one this staging could most
  easily have crossed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts/transfer") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts/transfer"))

from src.transfer import arms as A  # noqa: E402
from src.transfer import scaling  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    PANEL,
    PROTEIN_SCALE_LADDER,
    STAGED_ARMS,
    STAGED_SCALE_ARMS,
    STAGED_SECOND_STAGE_ARMS,
    Arm,
    arm_spec,
    config_shape,
    load_arm,
)
from src.transfer.circuits import _CIRCUIT_ARCHITECTURES  # noqa: E402

#: The architectures this campaign brings in that no measurement family
#: implements. Named here so the test below can assert their absence from every
#: table rather than assert a capability set by hand.
#:
#: ``opt`` was in this set while EXP-R2-225's joint wave was declared in
#: :data:`~src.transfer.arms.STAGED_ARMS`. It is in neither now: the two
#: Galactica rungs are reached by path rather than declared as arms
#: (``21_joint_mode_qualification.py``'s rule, and the comment in
#: ``src.transfer.arms`` where their declaration would have sat), and the
#: architecture itself is served by the lens family --
#: ``tests/test_opt_architecture.py`` is where what it may and may not enter is
#: pinned.
UNIMPLEMENTED_ARCHITECTURES = frozenset({"rita", "proteinglm"})


def _load_stage(filename: str):
    path = REPO_ROOT / "scripts/transfer" / filename
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ the door


def test_the_second_stage_checkpoints_are_staged_and_are_not_panel_members():
    assert STAGED_SECOND_STAGE_ARMS == (
        "qwen2.5-7b",
        "qwen2.5-32b",
        "proteinglm-7b-clm",
        "rita-xl",
    )
    for name in STAGED_SECOND_STAGE_ARMS:
        assert name in STAGED_ARMS, name
        assert name not in PANEL, name
        assert arm_spec(name) is STAGED_ARMS[name], name
        # The panel door stays panel-only: a staged checkpoint is loadable only
        # by a caller that resolved its declaration on purpose.
        with pytest.raises(KeyError):
            load_arm(name, device="cpu")


def test_the_first_round_door_is_unchanged_and_the_two_doors_are_disjoint():
    """EXP-R2-224's frozen set must not be widened by EXP-R2-225's arrival."""

    assert STAGED_SCALE_ARMS == ("progen2-large", "progen2-xlarge")
    assert PROTEIN_SCALE_LADDER == (
        "progen2-small",
        "progen2-medium",
        "progen2-large",
        "progen2-xlarge",
    )
    assert set(STAGED_SCALE_ARMS).isdisjoint(STAGED_SECOND_STAGE_ARMS)
    # And no staged checkpoint is behind neither door, which is what the
    # import-time check refuses.
    assert set(STAGED_ARMS) == set(STAGED_SCALE_ARMS) | set(STAGED_SECOND_STAGE_ARMS)


def test_an_overlapping_or_unknown_second_stage_door_is_refused_at_import():
    original = A.STAGED_SECOND_STAGE_ARMS
    try:
        A.STAGED_SECOND_STAGE_ARMS = original + ("progen2-large",)
        with pytest.raises(AssertionError, match="both staged doors"):
            A._check_second_stage_arms()
        A.STAGED_SECOND_STAGE_ARMS = original + ("not-a-checkpoint",)
        with pytest.raises(AssertionError, match="not declared in"):
            A._check_second_stage_arms()
        A.STAGED_SECOND_STAGE_ARMS = original[:-1]
        with pytest.raises(AssertionError, match="neither opt-in door"):
            A._check_second_stage_arms()
    finally:
        A.STAGED_SECOND_STAGE_ARMS = original
    A._check_second_stage_arms()


# ----------------------------------------------------------- honest capabilities


def test_the_joint_wave_is_not_declared_as_an_arm():
    """``21_joint_mode_qualification.py``'s rule, asserted where it can be broken.

    A joint checkpoint that has not passed that stage "must not be in ``arms.py``
    at all", and nothing about a staging campaign changes that. It costs nothing
    to keep: every Direction-2 stage that reads a joint checkpoint takes it as a
    ``--checkpoint`` with a ``--rendering``, the panel stages refuse a staged name
    outright, and stage 21 itself computes stage 01's estimand on one by path. A
    row here would additionally have blocked the ladder route, because
    ``scaling.register_arm_spec`` refuses by name every checkpoint STAGED_ARMS
    declares.
    """

    for name in ("galactica-125m", "galactica-1.3b", "galactica-6.7b", "galactica-30b"):
        assert name not in STAGED_ARMS, name
        assert name not in PANEL, name
        assert name not in STAGED_SECOND_STAGE_ARMS, name
    assert not any(spec.architecture == "opt" for spec in STAGED_ARMS.values())
    assert not any(spec.architecture == "opt" for spec in PANEL.values())


def test_the_new_architectures_are_in_none_of_the_tables_that_implement_a_family():
    """The premise the empty capability sets rest on, asserted not assumed."""

    for architecture in UNIMPLEMENTED_ARCHITECTURES:
        assert architecture not in A._ATTENTION_PATH, architecture
        assert architecture not in A._DECOMPOSABLE, architecture
        assert architecture not in _CIRCUIT_ARCHITECTURES, architecture
        assert architecture not in scaling.LENS_ARCHITECTURES, architecture


def test_no_second_stage_arm_declares_a_capability_it_cannot_honour():
    for name in STAGED_SECOND_STAGE_ARMS:
        spec = STAGED_ARMS[name]
        if spec.architecture in UNIMPLEMENTED_ARCHITECTURES:
            # Nothing implements these, so the honest set is empty and the
            # refusal carries the reason recorded beside the declaration.
            assert spec.capabilities == frozenset(), name
            continue
        assert spec.architecture == "qwen2", name
        # The rotary grant is the panel's own, not a wider one invented here.
        assert spec.capabilities == PANEL["qwen2.5-0.5b"].capabilities, name
        if "circuits" in spec.capabilities:
            assert spec.architecture in _CIRCUIT_ARCHITECTURES, name
        if "pathway" in spec.capabilities:
            assert spec.architecture in A._DECOMPOSABLE, name


def test_an_empty_capability_set_refuses_every_family_by_name():
    """An empty set is a refusal with a reason, not an unfilled field."""

    arm = Arm(
        spec=STAGED_ARMS["rita-xl"],
        model=SimpleNamespace(config=SimpleNamespace(vocab_size=26)),
        tokenizer=object(),
        device="cpu",
        dtype="float32",
    )
    for capability in sorted(A.CAPABILITIES):
        assert not arm.supports(capability), capability
        with pytest.raises(ValueError, match="does not support"):
            arm.require(capability)
    with pytest.raises(TypeError, match="unsupported architecture"):
        arm.blocks()


def test_an_undeclared_rendering_cannot_be_rendered():
    """This module serves ProteinGLM no rendering, so no cohort renders for it.

    Its native convention has since been evidenced -- a ``<gmask><sop><eos>``
    prefix, identified against a shuffled and an unprefixed control -- and the
    field is still the sentinel, because no branch of ``Cohort.input_strings``
    emits that prefix. The sentinel is what keeps "this repository renders
    nothing here" from being mistaken for a supported format name.
    """

    spec = STAGED_ARMS["proteinglm-7b-clm"]
    assert spec.input_format == A.INPUT_FORMAT_UNDECLARED
    arm = Arm(
        spec=spec,
        model=SimpleNamespace(config=SimpleNamespace(vocab_size=128)),
        tokenizer=object(),
        device="cpu",
        dtype="float16",
    )
    cohort = A.Cohort(
        name="stub", kind="protein", records=["MKT"], min_symbols=0, max_symbols=8
    )
    with pytest.raises(ValueError, match="unsupported input format"):
        cohort.input_strings(arm)


# ------------------------------------------------------ declared against on disk


@pytest.mark.parametrize("name", list(STAGED_SECOND_STAGE_ARMS))
def test_the_declaration_matches_the_checkpoints_own_config(name):
    """Depth, width and scoring alphabet, read off the file rather than trusted.

    The config is read as plain JSON and handed to :func:`config_shape` as an
    attribute bag, so this exercises the loader's real fallback order against the
    real keys without importing a checkpoint's remote modeling code.
    """

    spec = STAGED_ARMS[name]
    config_path = spec.path / "config.json"
    if not config_path.is_file():
        pytest.skip(f"{name} is not staged on this host")
    declared = json.loads(config_path.read_text(encoding="utf-8"))
    n_layer, d_model = config_shape(SimpleNamespace(**declared))
    assert (n_layer, d_model) == (spec.n_layer, spec.d_model)
    # Every second-stage arm's scoring alphabet IS its config vocabulary; the
    # explicit declaration exists because a staged arm must declare one, not
    # because the two differ as they do on progen2-large.
    assert spec.scoring_target_alphabet_size == int(declared["vocab_size"])


@pytest.mark.parametrize("name", ["progen2-large", "progen2-xlarge"])
def test_the_first_round_rungs_still_resolve_their_own_config_spellings(name):
    """The widened fallback order must not have moved the existing rungs."""

    spec = STAGED_ARMS[name]
    config_path = spec.path / "config.json"
    if not config_path.is_file():
        pytest.skip(f"{name} is not staged on this host")
    declared = json.loads(config_path.read_text(encoding="utf-8"))
    assert config_shape(SimpleNamespace(**declared)) == (spec.n_layer, spec.d_model)


# --------------------------------------------------------------- shape fallback


def test_config_shape_reads_every_declared_spelling():
    assert config_shape(SimpleNamespace(n_layer=27, n_embd=1536)) == (27, 1536)
    assert config_shape(SimpleNamespace(num_hidden_layers=28, hidden_size=3584)) == (
        28,
        3584,
    )
    assert config_shape(SimpleNamespace(num_layers=24, d_model=2048)) == (24, 2048)
    assert config_shape(SimpleNamespace(num_layers=36, hidden_size=4096)) == (36, 4096)
    assert config_shape(SimpleNamespace(n_layer=32, embed_dim=4096)) == (32, 4096)


def test_config_shape_prefers_the_earlier_spelling_when_two_are_present():
    """The order is a declaration, so a config carrying both resolves the same way."""

    both = SimpleNamespace(n_layer=27, num_layers=99, n_embd=1536, d_model=99)
    assert config_shape(both) == (27, 1536)


def test_config_shape_refuses_a_config_that_declares_neither():
    with pytest.raises(AttributeError, match="declares no depth"):
        config_shape(SimpleNamespace(depth=12, hidden_size=768))
    with pytest.raises(AttributeError, match="declares no width"):
        config_shape(SimpleNamespace(n_layer=12, width=768))
    # A None is not a declaration either: it is what an unset key reads as.
    with pytest.raises(AttributeError, match="declares no depth"):
        config_shape(SimpleNamespace(n_layer=None, hidden_size=768))


def test_config_shape_does_not_consult_the_output_vocabulary():
    """vocab_size is not a shape, and two ProGen2 rungs spell it differently."""

    with pytest.raises(AttributeError, match="declares no depth"):
        config_shape(SimpleNamespace(vocab_size=51200, hidden_size=2560))


# ------------------------------------------------------------------- stage doors


def test_cohort_power_refuses_second_stage_arms_without_the_opt_in():
    stage = _load_stage("01_cohort_power.py")
    closed = argparse.Namespace(
        kind="text",
        with_ec=False,
        allow_staged_scale_arms=False,
        allow_second_stage_arms=False,
    )
    with pytest.raises(ValueError, match="unknown arms"):
        stage.validate_arms(["qwen2.5-7b"], closed)
    # And the first round's flag is not a door onto this campaign's arms.
    scale_only = argparse.Namespace(
        kind="text",
        with_ec=False,
        allow_staged_scale_arms=True,
        allow_second_stage_arms=False,
    )
    with pytest.raises(ValueError, match="unknown arms"):
        stage.validate_arms(["qwen2.5-7b"], scale_only)


def test_cohort_power_second_stage_opt_in_is_not_a_door_onto_the_first_round():
    stage = _load_stage("01_cohort_power.py")
    args = argparse.Namespace(
        kind="protein",
        with_ec=False,
        allow_staged_scale_arms=False,
        allow_second_stage_arms=True,
    )
    with pytest.raises(ValueError, match="unknown arms"):
        stage.validate_arms(["progen2-large"], args)


def test_cohort_power_admits_a_second_stage_arm_that_can_honour_the_estimand():
    stage = _load_stage("01_cohort_power.py")
    args = argparse.Namespace(
        kind="text",
        with_ec=False,
        allow_staged_scale_arms=False,
        allow_second_stage_arms=True,
    )
    stage.validate_arms(["qwen2.5-7b", "qwen2.5-32b"], args)
    record = stage._second_stage_record(["qwen2.5-7b"], True)
    assert record["not_panel_admission"] is True
    assert record["measured_second_stage_arms"] == ["qwen2.5-7b"]
    assert record["scoring_target_alphabet"]["qwen2.5-7b"]["size"] == 152064
    contract = stage._cohort_power_stage_contract(["qwen2.5-7b"])
    assert contract["not_panel_admission"] is True
    assert contract["measured"] == []


def test_cohort_power_refuses_a_second_stage_arm_that_declares_no_budget():
    """An arm that cannot produce the estimand is refused before it is loaded."""

    stage = _load_stage("01_cohort_power.py")
    args = argparse.Namespace(
        kind="protein",
        with_ec=False,
        allow_staged_scale_arms=False,
        allow_second_stage_arms=True,
    )
    with pytest.raises(ValueError, match="no 'budget' capability"):
        stage.validate_arms(["rita-xl"], args)
    with pytest.raises(ValueError, match="no 'budget' capability"):
        stage.validate_arms(["proteinglm-7b-clm"], args)


def test_the_fitness_stages_still_refuse_every_second_stage_arm():
    """Stages 20 and 29 are frozen on the first round and gained no door."""

    retrieval = _load_stage("20_retrieval_bound.py")
    assert set(retrieval.SCOREABLE_ARMS).isdisjoint(STAGED_SECOND_STAGE_ARMS)
    for name in STAGED_SECOND_STAGE_ARMS:
        with pytest.raises(KeyError):
            retrieval.corpus_record(name)
    designed = _load_stage("29_designed_referent.py")
    assert set(designed.DEFAULT_ARMS).isdisjoint(STAGED_SECOND_STAGE_ARMS)
    assert set(designed.STAGED_SCALE_ARMS) == set(STAGED_SCALE_ARMS)
