from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from src.revision.io import sha256_file, write_json
from src.revision.prospective_positive_control import (
    CLAIM_SCOPE,
    TOKENS,
    ProspectiveLongRangeLM,
    _all_seed_detection_gates,
    _detection_metrics,
    _fit_and_select,
    _match_controls,
    encode_prospective_cohort,
    execute_frozen_prospective_benchmark,
    freeze_prospective_benchmark,
    generate_prospective_cohort,
    validate_prospective_spec,
)


PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "scripts/69_run_prospective_positive_control.py"
SPEC_PATH = PROJECT / "configs/p0_7_prospective_positive_control_spec.json"


def _spec() -> dict:
    return json.loads(SPEC_PATH.read_text())


def test_unexposed_long_range_grammar_is_balanced_and_future_targeted() -> None:
    spec = validate_prospective_spec(_spec())
    records = generate_prospective_cohort(spec)
    assert len(records) == 3 * spec["cohort"]["per_split"]
    assert TOKENS[:3] == ("<pad>", "<bos>", "<eos>")
    assert not any(
        marker in token
        for token in TOKENS
        for marker in ("lr0", "lr1", "query", "endpoint", "family")
    )
    for split in ("train", "discovery", "assessment"):
        rows = [row for row in records if row.split == split]
        assert sum(row.long_range_equal for row in rows) == len(rows) // 2
        for anchor in ("left_anchor_residue", "right_anchor_residue"):
            assert sum(getattr(row, anchor) == "A" for row in rows) == len(rows) // 2
        assert all(
            row.long_range_equal
            == (row.left_anchor_residue == row.right_anchor_residue)
            for row in rows
        )
        assert all(row.sequence[-1] == ("Y" if row.long_range_equal else "W") for row in rows)
    cohort = encode_prospective_cohort(records[: spec["cohort"]["per_split"]])
    for row, record in enumerate(cohort.records):
        predictor = int(cohort.predictor_positions[row])
        assert cohort.input_ids[row, predictor + 1].item() != cohort.input_ids[
            row, predictor
        ].item()
        assert predictor == int(cohort.attention_mask[row].sum()) - 3


def test_relation_cannot_change_prefix_before_second_anchor() -> None:
    spec = _spec()
    records = generate_prospective_cohort(spec)
    first = next(row for row in records if row.split == "train")
    partner = next(
        row
        for row in records
        if row.split == "train"
        and row.length == first.length
        and row.left_anchor_residue == first.left_anchor_residue
        and row.long_range_equal != first.long_range_equal
    )
    left = encode_prospective_cohort([first])
    right = encode_prospective_cohort([partner])
    # Equalize every visible token before the right anchor; only the distant
    # anchor and later suffix may differ.
    boundary = spec["cohort"]["right_anchor_position"] + 1
    right.input_ids[:, :boundary] = left.input_ids[:, :boundary]
    model = ProspectiveLongRangeLM(spec["model"]["seeds"][0], spec).eval()
    with torch.no_grad():
        left_logits, _, _, _ = model(left.input_ids, left.attention_mask)
        right_logits, _, _, _ = model(right.input_ids, right.attention_mask)
    torch.testing.assert_close(left_logits[:, :boundary], right_logits[:, :boundary])


def test_selection_api_cannot_receive_assessment_rows() -> None:
    parameters = inspect.signature(_fit_and_select).parameters
    assert "train" in parameters and "discovery" in parameters
    assert "assessment" not in parameters and "evaluation" not in parameters


def test_every_seed_must_pass_even_when_the_mean_would_pass() -> None:
    gates = _spec()["gates"]
    rows = [
        {"sensitivity": 1.0, "specificity": 1.0, "false_discovery_rate": 0.0},
        {"sensitivity": 1.0, "specificity": 1.0, "false_discovery_rate": 0.0},
        {"sensitivity": 0.7, "specificity": 1.0, "false_discovery_rate": 0.0},
    ]
    assert sum(row["sensitivity"] for row in rows) / 3 >= gates["sensitivity_min"]
    assert not _all_seed_detection_gates(rows, gates)
    rows[-1]["sensitivity"] = gates["sensitivity_min"]
    assert _all_seed_detection_gates(rows, gates)


def test_detection_truth_excludes_dead_decoder_rows() -> None:
    table = [
        {"feature": 0, "active_ground_truth_positive": True},
        {"feature": 1, "active_ground_truth_positive": False},
        {"feature": 2, "active_ground_truth_positive": False},
    ]
    truth, sensitivity, specificity, fdr = _detection_metrics(table, [0])
    assert truth == {0}
    assert (sensitivity, specificity, fdr) == (1.0, 1.0, 0.0)
    _, sensitivity, specificity, fdr = _detection_metrics(table, [0, 2])
    assert (sensitivity, specificity, fdr) == (1.0, 0.5, 0.5)
    table[1]["active_ground_truth_positive"] = True
    truth, sensitivity, specificity, fdr = _detection_metrics(table, [0])
    assert truth == {0, 1}
    assert (sensitivity, specificity, fdr) == (1.0, 1.0, 0.0)


def test_known_nuisance_controls_are_active_orthogonal_and_inside_calipers() -> None:
    spec = _spec()
    model = ProspectiveLongRangeLM(spec["model"]["seeds"][0], spec)
    table = [
        {
            "feature": 7,
            "firing_frequency": 0.5,
            "mean_activation": 0.25,
            "decoder_norm": 1.2,
            "direct_logit_effect_norm": 2.0,
            "received_attention_mass": 0.1,
            "reconstruction_contribution": 0.3,
        }
    ]
    controls = _match_controls(table, 7, spec, model)
    assert len(controls) == spec["matching"]["control_count"]
    assert all(abs(row["ground_truth_direction_cosine"]) < 1e-6 for row in controls)
    assert all(row["profile"]["firing_frequency"] == 0.5 for row in controls)
    for row in controls:
        diagnostics = row["caliper_diagnostics"]
        assert diagnostics["standardized_distance"] <= spec["matching"][
            "hard_calipers"
        ]["standardized_distance_max"]


def test_freeze_is_hash_bound_and_execution_is_once_only(tmp_path, monkeypatch) -> None:
    spec = _spec()
    spec["cohort"]["split_seeds"] = {
        "train": 7101,
        "discovery": 7201,
        "assessment": 7301,
    }
    path = tmp_path / "spec.json"
    write_json(path, spec)
    frozen = tmp_path / "freeze"
    manifest_path = freeze_prospective_benchmark(
        path,
        sha256_file(path),
        frozen,
        runner_path=RUNNER,
        command=["fixture-freeze"],
    )
    with pytest.raises(FileExistsError, match="overwrite benchmark freeze"):
        freeze_prospective_benchmark(
            path,
            sha256_file(path),
            frozen,
            runner_path=RUNNER,
            command=["fixture-freeze"],
        )

    def fixture_result(_records, _specification, *, checkpoint_dir):
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        return {
            "schema_version": "fixture",
            "status": "fixture_only",
            "claim_scope": CLAIM_SCOPE,
            "models": [{"checkpoint": None}],
            "feature_discovery_rows": [],
            "intervention_rows": [],
        }

    monkeypatch.setattr(
        "src.revision.prospective_positive_control.run_prospective_positive_control",
        fixture_result,
    )
    result = tmp_path / "result"
    execute_frozen_prospective_benchmark(
        frozen,
        sha256_file(manifest_path),
        result,
        runner_path=RUNNER,
        command=["fixture-execute"],
    )
    assert (frozen / "execution_claim.json").is_file()
    with pytest.raises(FileExistsError):
        execute_frozen_prospective_benchmark(
            frozen,
            sha256_file(manifest_path),
            tmp_path / "second-result",
            runner_path=RUNNER,
            command=["fixture-execute-again"],
        )


def test_freeze_tamper_is_rejected_before_claim(tmp_path) -> None:
    spec = deepcopy(_spec())
    spec["cohort"]["split_seeds"] = {
        "train": 8101,
        "discovery": 8201,
        "assessment": 8301,
    }
    path = tmp_path / "spec.json"
    write_json(path, spec)
    frozen = tmp_path / "freeze"
    manifest = freeze_prospective_benchmark(
        path,
        sha256_file(path),
        frozen,
        runner_path=RUNNER,
        command=["fixture-freeze"],
    )
    with (frozen / "cohort.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="frozen artifact changed"):
        execute_frozen_prospective_benchmark(
            frozen,
            sha256_file(manifest),
            tmp_path / "result",
            runner_path=RUNNER,
            command=["fixture-execute"],
        )
    assert not (frozen / "execution_claim.json").exists()
