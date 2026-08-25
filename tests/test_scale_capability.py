"""Descriptive ProGen2 scale expansion: alphabet, opt-in, alignment, gates."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts/transfer") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts/transfer"))

from src.transfer import arms as A  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    PANEL,
    PROTEIN_SCALE_LADDER,
    STAGED_ARMS,
    STAGED_SCALE_ARMS,
    Arm,
    require_scoring_target_ids,
    scoring_target_alphabet,
)
from src.transfer.budget import record_statistics  # noqa: E402
from src.transfer.budget import ScoredTokens  # noqa: E402


def _load_stage(filename: str):
    path = REPO_ROOT / "scripts/transfer" / filename
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Config:
    def __init__(self, vocab_size=None, *, missing=False):
        self._missing = missing
        if not missing:
            self.vocab_size = vocab_size

    def __getattribute__(self, name):
        if name == "vocab_size" and object.__getattribute__(self, "_missing"):
            raise AttributeError(
                "config.vocab_size was read on the ProGen2 path: progen2-xlarge "
                "declares no such key"
            )
        return object.__getattribute__(self, name)


class _Tokenizer:
    def decode(self, ids):
        return "A" * len(list(ids))


def _arm(spec, *, vocab_size=32, missing=False):
    return Arm(
        spec=spec,
        model=SimpleNamespace(config=_Config(vocab_size, missing=missing)),
        tokenizer=_Tokenizer(),
        device="cpu",
        dtype="float32",
    )


# ------------------------------------------------------------- panel invariant


def test_default_panel_does_not_admit_the_staged_scale_rungs():
    assert "progen2-large" not in PANEL
    assert "progen2-xlarge" not in PANEL
    assert STAGED_SCALE_ARMS == ("progen2-large", "progen2-xlarge")
    assert PROTEIN_SCALE_LADDER[:2] == ("progen2-small", "progen2-medium")
    for name in STAGED_SCALE_ARMS:
        assert name in STAGED_ARMS
        assert name not in PANEL


def test_panel_default_reads_config_vocab_size():
    spec = replace(PANEL["progen2-medium"], scoring_target_alphabet_size=None)
    record = scoring_target_alphabet(spec, _Config(32))
    assert record["size"] == 32
    assert record["source"] == A.SCORING_TARGET_ALPHABET_CONFIG


def test_staged_rungs_declare_thirty_two_and_ignore_a_51200_config():
    spec = STAGED_ARMS["progen2-large"]
    record = scoring_target_alphabet(spec, _Config(51200))
    assert record["size"] == 32
    assert record["source"] == A.SCORING_TARGET_ALPHABET_DECLARED
    xlarge = scoring_target_alphabet(STAGED_ARMS["progen2-xlarge"])
    assert xlarge["size"] == 32


def test_a_missing_declaration_does_not_guess_tokenizer_length():
    spec = replace(PANEL["progen2-medium"], scoring_target_alphabet_size=None)
    with pytest.raises(ValueError, match="tokenizer length is not a substitute"):
        scoring_target_alphabet(spec, _Config(missing=True))
    with pytest.raises(ValueError, match="tokenizer length is not a substitute"):
        scoring_target_alphabet(spec, None)


def test_a_non_positive_declared_alphabet_is_refused():
    with pytest.raises(ValueError, match="positive"):
        replace(STAGED_ARMS["progen2-large"], scoring_target_alphabet_size=0)


def test_target_ids_outside_the_declared_alphabet_are_refused():
    alphabet = scoring_target_alphabet(STAGED_ARMS["progen2-large"])
    require_scoring_target_ids(np.array([0, 31], dtype=np.int64), alphabet, arm="progen2-large")
    with pytest.raises(ValueError, match="outside the scoring-target alphabet"):
        require_scoring_target_ids(
            np.array([0, 32], dtype=np.int64), alphabet, arm="progen2-large"
        )


def test_record_statistics_uses_the_declared_alphabet_not_config_width():
    spec = STAGED_ARMS["progen2-large"]
    arm = _arm(spec, vocab_size=51200)
    scored = ScoredTokens(
        target_ids=np.array([1, 2, 3], dtype=np.int64),
        nll_nats=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        sequence_index=np.array([0, 0, 0], dtype=np.int64),
    )
    records = record_statistics(arm, scored)
    assert records.vocab_size == 32


def test_record_statistics_refuses_an_out_of_range_target():
    arm = _arm(STAGED_ARMS["progen2-xlarge"], missing=True)
    scored = ScoredTokens(
        target_ids=np.array([31, 32], dtype=np.int64),
        nll_nats=np.array([0.1, 0.2], dtype=np.float64),
        sequence_index=np.array([0, 0], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="outside the scoring-target alphabet"):
        record_statistics(arm, scored)


# ------------------------------------------------------------- stage 01 opt-in


def test_cohort_power_refuses_staged_rungs_without_the_opt_in():
    stage = _load_stage("01_cohort_power.py")
    args = argparse.Namespace(kind="protein", with_ec=False, allow_staged_scale_arms=False)
    with pytest.raises(ValueError, match="unknown arms"):
        stage.validate_arms(["progen2-large"], args)
    with pytest.raises(ValueError, match="unknown arms"):
        stage.validate_arms(["progen2-xlarge"], args)


def test_cohort_power_opt_in_accepts_only_ladder_staged_rungs():
    stage = _load_stage("01_cohort_power.py")
    args = argparse.Namespace(kind="protein", with_ec=False, allow_staged_scale_arms=True)
    stage.validate_arms(["progen2-large", "progen2-xlarge"], args)
    stage.validate_arms(["progen2-medium", "progen2-large"], args)
    with pytest.raises(ValueError, match="unknown arms"):
        stage.validate_arms(["not-a-model"], args)


def test_cohort_power_default_arms_stay_inside_the_panel():
    stage = _load_stage("01_cohort_power.py")
    names = stage.default_arms("protein", with_ec=False)
    assert "progen2-large" not in names
    assert "progen2-xlarge" not in names
    assert all(name in PANEL for name in names)


def test_cohort_power_stage_contract_does_not_claim_staged_admission():
    stage = _load_stage("01_cohort_power.py")
    record = stage._staged_scale_record(["progen2-large"], True)
    assert record["not_panel_admission"] is True
    assert record["measured_staged_arms"] == ["progen2-large"]
    assert record["scoring_target_alphabet"]["progen2-large"]["size"] == 32
    spec_record = stage._arm_spec_record("progen2-medium")
    assert "not_panel_admission" not in spec_record
    staged_spec = stage._arm_spec_record("progen2-large")
    assert staged_spec["not_panel_admission"] is True
    assert staged_spec["scoring_target_alphabet_size"] == 32


# ------------------------------------------------------------- stages 20 / 29


def test_retrieval_bound_default_arms_are_unchanged():
    stage = _load_stage("20_retrieval_bound.py")
    assert list(stage.ARM_CORPUS) == ["protgpt2", "progen2-medium", "progen3-112m"] or set(
        stage.ARM_CORPUS
    ) == {"protgpt2", "progen2-medium", "progen3-112m"}
    assert "progen2-large" not in stage.ARM_CORPUS
    corpus = stage.corpus_record("progen2-large")
    assert corpus["declared"] == A.arm_spec("progen2-large").pretraining_corpus
    assert "lower bound" in corpus["identification"]
    assert "progen3-112m" in stage.ARM_CORPUS


def test_designed_referent_default_arms_and_progen3_exclusion_are_unchanged():
    stage = _load_stage("29_designed_referent.py")
    assert stage.DEFAULT_ARMS == (
        "protgpt2",
        "progen2-small",
        "progen2-base",
        "progen2-medium",
    )
    assert "progen3-112m" in stage.D.EXCLUDED_ARMS
    assert "bidirectional" in stage.D.EXCLUDED_ARMS["progen3-112m"]
    assert "progen2-large" in stage.D.ARM_IDENTIFICATION
    assert "progen2-xlarge" in stage.D.ARM_IDENTIFICATION
    assert (
        stage.D.ARM_IDENTIFICATION["progen2-large"]["identification"]
        == "unbounded_in_the_model_favouring_direction"
    )


# ------------------------------------------------------------- stage 42


def _dms_model(arm: str, rhos: dict[str, float], digest: str = "d0") -> dict:
    return {
        "arm": arm,
        "assays": [
            {
                "assay": name,
                "wildtype_id": f"q{index:02d}",
                "mutant_digest": digest,
                "spearman": value,
            }
            for index, (name, value) in enumerate(rhos.items())
        ],
        "skipped": [],
    }


def _lookup(rhos: dict[str, float], *, digest: str = "d0") -> dict:
    return {
        "assays": [
            {
                "assay": name,
                "cluster": f"c{index % 8}",
                "mutant_digest": digest,
                "spearman": {"lookup": 0.10, "blosum62": 0.05},
            }
            for index, name in enumerate(rhos)
        ]
    }


def _mega_model(arm: str, values: dict[str, float], digest: str, *, kind: str) -> dict:
    return {
        "arm": arm,
        "cohort_sha256": digest,
        "wildtypes": {
            name: {
                "kind": kind,
                "unit": f"u{index % 8}",
                "n_variants": 40,
                "spearman": value,
            }
            for index, (name, value) in enumerate(values.items())
        },
    }


def _baselines(names: list[str], digest: str) -> dict:
    return {
        "cohort_sha256": digest,
        "wildtypes": {
            name: {"spearman": {"blosum62": 0.05, "hydropathy_change": 0.02}}
            for name in names
        },
    }


def test_scale_capability_refuses_a_mutant_digest_mismatch():
    stage = _load_stage("42_scale_capability.py")
    assays = {f"a{i}": 0.2 for i in range(8)}
    models = {
        "progen2-medium": _dms_model("progen2-medium", assays, "aaa"),
        "progen2-large": _dms_model("progen2-large", assays, "bbb"),
        "progen2-xlarge": _dms_model("progen2-xlarge", assays, "aaa"),
    }
    with pytest.raises(ValueError, match="mutant_digest"):
        stage.align_dms(models, _lookup(assays, digest="aaa"))


def test_scale_capability_refuses_a_cohort_digest_mismatch():
    stage = _load_stage("42_scale_capability.py")
    names = [f"w{i}" for i in range(8)]
    values = {name: 0.2 for name in names}
    models = {
        "progen2-medium": _mega_model("progen2-medium", values, "one", kind="design"),
        "progen2-large": _mega_model("progen2-large", values, "two", kind="design"),
        "progen2-xlarge": _mega_model("progen2-xlarge", values, "one", kind="design"),
    }
    with pytest.raises(ValueError, match="cohort_sha256"):
        stage.align_megascale(
            models, _baselines(names, "one"), side="design", baseline_name="blosum62"
        )


def test_scale_capability_refuses_the_wrong_rung_set():
    stage = _load_stage("42_scale_capability.py")
    with pytest.raises(ValueError, match="fixed as"):
        stage.require_rung_order(["progen2-small", "progen2-medium", "progen2-large"])


def test_descriptive_gate_transition_needs_both_conditions():
    stage = _load_stage("42_scale_capability.py")
    beats = {"interval": [0.01, 0.10], "degenerate": False}
    delta = {"interval": [0.02, 0.08], "degenerate": False}
    assert stage.descriptive_gate_transition(beats, delta) is True
    assert (
        stage.descriptive_gate_transition({"interval": [-0.01, 0.10], "degenerate": False}, delta)
        is False
    )
    assert (
        stage.descriptive_gate_transition(beats, {"interval": [-0.02, 0.08], "degenerate": False})
        is False
    )
    assert (
        stage.descriptive_gate_transition({"interval": None, "degenerate": True}, delta)
        == "unresolved"
    )


def test_compare_scale_writes_the_required_claims_and_no_total(tmp_path):
    stage = _load_stage("42_scale_capability.py")
    assays = {f"a{i}": 0.20 + 0.01 * (i % 3) for i in range(16)}
    designs = {f"d{i}": 0.15 + 0.01 * (i % 3) for i in range(16)}
    naturals = {f"n{i}": 0.12 + 0.01 * (i % 3) for i in range(16)}
    digest = "abc123"
    dms_models = {
        name: _dms_model(
            name,
            {key: value + offset for key, value in assays.items()},
        )
        for name, offset in (
            ("progen2-medium", 0.00),
            ("progen2-large", 0.05),
            ("progen2-xlarge", 0.10),
        )
    }
    mega_models = {}
    for name, offset in (
        ("progen2-medium", 0.00),
        ("progen2-large", 0.04),
        ("progen2-xlarge", 0.08),
    ):
        design_block = _mega_model(
            name, {key: value + offset for key, value in designs.items()}, digest, kind="design"
        )
        natural_block = _mega_model(
            name, {key: value + offset for key, value in naturals.items()}, digest, kind="natural"
        )
        design_block["wildtypes"].update(natural_block["wildtypes"])
        mega_models[name] = design_block
    baselines = _baselines(list(designs) + list(naturals), digest)
    payload = stage.compare_scale(
        dms_models=dms_models,
        lookup=_lookup(assays),
        megascale_models=mega_models,
        baselines=baselines,
        fragment_order=None,
        qualification=None,
        resamples=200,
        seed=1,
    )
    assert payload["descriptive_not_causal"] is True
    assert payload["no_biological_knowledge_claim"] is True
    assert payload["not_panel_admission"] is True
    assert payload["no_cross_task_total"] is True
    assert "total_score" not in payload
    assert "composite" not in payload
    assert "UniRef90" in payload["corpus_identification_bound"]
    dms_lookup = payload["dms"]["model_minus_lookup"]
    assert set(dms_lookup["per_rung"]) == set(stage.SCALE_RUNGS)
    assert "progen2-medium__progen2-large" in dms_lookup["adjacent_delta_rho"]
    gate = dms_lookup["adjacent_delta_rho"]["progen2-medium__progen2-large"][
        "descriptive_gate_transition"
    ]
    assert gate in (True, False, "unresolved")
    assert payload["megascale"]["designs"]["model_minus_hydropathy"]
    assert payload["megascale"]["control"]["model_minus_blosum62"]
    assert payload["fragment_order"] is None


def test_stage41_summary_must_name_every_rung():
    stage = _load_stage("42_scale_capability.py")
    with pytest.raises(ValueError, match="stage-41"):
        stage._qualification({"arms": [{"arm": "progen2-medium", "status": "PASS"}]})
    record = stage._qualification(
        {
            "arms": [
                {"arm": name, "status": "PASS", "present": True}
                for name in stage.SCALE_RUNGS
            ]
        }
    )
    assert record["rungs"]["progen2-large"]["status"] == "PASS"
