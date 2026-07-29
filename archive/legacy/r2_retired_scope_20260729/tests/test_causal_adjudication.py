from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.revision.causal_adjudication as causal_adjudication
from src.revision.causal_adjudication import (
    MEASURES,
    adjudicate_pretrained_causal_interventions,
    analyze_paired_cells,
    load_adjudication_spec,
    load_bound_inputs,
    validate_factorial_inventory,
)
from src.revision.io import sha256_file, write_json, write_jsonl
from src.revision.statistics import benjamini_hochberg


SCOPE = "synthetic_pipeline_sensitivity_only_no_pretrained_causal_inference"
SEEDS = (17, 29, 43)
SITES = ("a_off_path", "z_on_path")
MARGIN = 0.2


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _p0_2_receipt(
    *,
    complete: bool,
    run_hashes: dict[str, str] | None = None,
    checkpoint_hashes: dict[str, str] | None = None,
) -> tuple[dict, dict[str, str], dict[str, str]]:
    run_hashes = run_hashes or {str(seed): digest(f"run-{seed}") for seed in SEEDS}
    checkpoint_hashes = checkpoint_hashes or {
        str(seed): digest(f"checkpoint-{seed}") for seed in SEEDS
    }
    source_hashes = {split: digest(split) for split in ("train", "validation", "test")}
    receipt = {
        "schema_version": "r2_p0_2_eligibility_receipt_v1",
        "status": "complete" if complete else "incomplete",
        "artifact_completeness": complete,
        "required_models": ["protgpt2", "zymctrl", "progen2-medium"],
        "required_methods": [
            "topk_clt",
            "relu_l1_sae",
            "gated_sae",
            "dense_low_rank",
        ],
        "profile_sha256": digest("profile"),
        "protocol_sha256": digest("protocol"),
        "mask_validation_receipt_sha256": digest("mask"),
        "frozen_gate": {"required_seeds": list(SEEDS)},
        "model_method_adjudications": [
            {
                "model_name": "tiny",
                "method": "topk_clt",
                "status": "atlas_eligible",
                "atlas_eligible": True,
                "failure_reasons": [],
                "required_seeds": list(SEEDS),
                "source_manifest_sha256_by_split": source_hashes,
                "cache_manifest_sha256": digest("cache-manifest"),
                "cache_content_sha256": digest("cache-content"),
                "selected_layers": [1],
                "eligible_downstream_layers": [1],
                "geometry": {
                    "n_layers": 2,
                    "d_model": 4,
                    "d_clt": 4,
                    "k": 2,
                    "window": 1,
                },
                "runs": [
                    {
                        "run_seed": seed,
                        "run_manifest_sha256": run_hashes[str(seed)],
                        "result_sha256": digest(f"result-{seed}"),
                        "checkpoint_sha256": checkpoint_hashes[str(seed)],
                        "quality_gate_pass": True,
                        "failure_reasons": [],
                    }
                    for seed in SEEDS
                ],
            }
        ],
    }
    return receipt, run_hashes, checkpoint_hashes


def build_fixture(
    root: Path,
    *,
    positive_pass: bool = True,
    p0_2_complete: bool = True,
    overlap: str | None = None,
    omit_binding_kind: str | None = None,
    omit_one_p0_2_binding: bool = False,
) -> SimpleNamespace:
    root.mkdir(parents=True)
    positive_dir = root / "positive_control"
    positive_dir.mkdir()
    positive_summary = positive_dir / "summary.json"
    write_json(
        positive_summary,
        {
            "schema_version": "r2_p0_7_prospective_positive_control_result_v2",
            "status": (
                "prospective_synthetic_gate_passed"
                if positive_pass
                else "prospective_synthetic_gate_failed"
            ),
            "claim_scope": SCOPE,
            "pretrained_model_causal_inference": False,
            "legacy_controls_upgraded": False,
            "aggregate": {
                "prospective_synthetic_gate_passed": positive_pass,
                "all_paths_localized": positive_pass,
                "all_negative_metrics_equivalent": positive_pass,
            },
        },
    )
    positive_manifest = positive_dir / "run_manifest.json"
    write_json(
        positive_manifest,
        {
            "schema_version": "r2_p0_7_prospective_positive_control_manifest_v1",
            "status": "complete",
            "claim_scope": SCOPE,
            "artifact_hashes": {"summary.json": sha256_file(positive_summary)},
            "source_hashes": {
                "prospective_positive_control.py": digest("positive-code")
            },
        },
    )

    dictionary_bindings = []
    run_hashes, checkpoint_hashes = {}, {}
    for seed in SEEDS:
        for artifact, destination in (
            ("run_manifest", run_hashes),
            ("checkpoint", checkpoint_hashes),
        ):
            path = root / f"binding_dictionary_{seed}_{artifact}.bin"
            path.write_bytes(f"immutable-{seed}-{artifact}".encode("utf-8"))
            destination[str(seed)] = sha256_file(path)
            dictionary_bindings.append(
                {
                    "kind": "dictionary",
                    "name": f"tiny:seed_{seed}:{artifact}",
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            )
    p0_2_payload, run_hashes, checkpoint_hashes = _p0_2_receipt(
        complete=p0_2_complete,
        run_hashes=run_hashes,
        checkpoint_hashes=checkpoint_hashes,
    )
    p0_2_path = root / "p0_2_receipt.json"
    write_json(p0_2_path, p0_2_payload)
    p0_5_dir = root / "p0_5"
    p0_5_dir.mkdir()
    p0_5_data = p0_5_dir / "measurements.jsonl"
    write_jsonl(p0_5_data, [{"fixture": "hash-bound P0-5 data"}])
    p0_5_path = p0_5_dir / "run_manifest.json"
    write_json(
        p0_5_path,
        {
            "schema_version": "r2-p05-pretrained-extraction-manifest-v4",
            "status": "verified_production_complete",
            "execution_mode": "production",
            "artifact_hashes": {p0_5_data.name: sha256_file(p0_5_data)},
        },
    )
    p0_6_dir = root / "p0_6"
    p0_6_dir.mkdir()
    p0_6_data = p0_6_dir / "raw_generations.jsonl"
    write_jsonl(p0_6_data, [{"fixture": "hash-bound P0-6 data"}])
    p0_6_path = p0_6_dir / "execution_receipt.json"
    write_json(
        p0_6_path,
        {
            "schema_version": "r2-corrected-steering-execution-receipt-v3",
            "status": "verified_complete",
            "artifacts": {p0_6_data.name: sha256_file(p0_6_data)},
        },
    )

    evaluation_sequences = ["M" + residue * 11 for residue in "ACDEFGHIKNPQ"]
    evaluation_rows = [
        {"protein_id": f"eval-{index:02d}", "sequence": sequence}
        for index, sequence in enumerate(evaluation_sequences)
    ]
    discovery_row = {"protein_id": "discovery-00", "sequence": "RSTVWYRSTVWY"}
    if overlap == "id":
        discovery_row["protein_id"] = evaluation_rows[0]["protein_id"]
    elif overlap == "sequence":
        discovery_row["sequence"] = evaluation_rows[0]["sequence"]
    discovery_path = root / "discovery.jsonl"
    evaluation_path = root / "evaluation.jsonl"
    write_jsonl(discovery_path, [discovery_row])
    write_jsonl(evaluation_path, evaluation_rows)

    profile = {
        "firing_frequency": 0.1,
        "mean_activation": 0.5,
        "decoder_norm": 1.0,
        "direct_logit_effect_norm": 0.8,
        "received_attention_mass": 0.2,
        "reconstruction_contribution": 0.3,
    }
    identity_path = root / "identities.jsonl"
    write_jsonl(
        identity_path,
        [
            {
                "identity_set_id": "identity-1",
                "model": "tiny",
                "dictionary_seed": 17,
                "layer": 1,
                "target_feature": 7,
                "control_feature": 9,
                "path_site": SITES[1],
                "target_profile": profile,
                "control_profile": profile,
                "matching_calipers_passed": True,
                "matching_distance": 0.0,
            }
        ],
    )

    interventions, scores = [], []
    for index, cohort_row in enumerate(evaluation_rows):
        evaluation_id = cohort_row["protein_id"]
        sequence_sha256 = hashlib.sha256(
            cohort_row["sequence"].encode("utf-8")
        ).hexdigest()
        jitter = (index - (len(evaluation_rows) - 1) / 2.0) * 0.001
        for site in SITES:
            for role, feature in (("target", 7), ("control", 9)):
                row_id = f"{evaluation_id}:{site}:{role}"
                difference = 0.0
                if role == "target":
                    difference = 0.75 + jitter if site == SITES[1] else jitter
                signed_lower = -difference
                interventions.append(
                    {
                        "row_id": row_id,
                        "evaluation_id": evaluation_id,
                        "sequence_sha256": sequence_sha256,
                        "identity_set_id": "identity-1",
                        "model": "tiny",
                        "dictionary_seed": 17,
                        "layer": 1,
                        "feature_role": role,
                        "feature": feature,
                        "site": site,
                        "strength": 1.0,
                        "intended_feature_change": difference,
                        "off_target_sparse_code_displacement": signed_lower,
                        "reconstruction_displacement": signed_lower,
                        "logit_displacement": difference,
                    }
                )
                scores.append(
                    {
                        "row_id": row_id,
                        "behavior_endpoint": difference,
                        "path_endpoint": difference,
                    }
                )
    intervention_path = root / "interventions.jsonl"
    score_path = root / "external_scores.jsonl"
    write_jsonl(intervention_path, interventions)
    write_jsonl(score_path, scores)

    bindings = []
    for kind in (
        "model",
        "behavior_scorer",
        "behavior_calibration",
        "path_scorer",
        "path_calibration",
        "code",
    ):
        path = root / f"binding_{kind}.bin"
        path.write_bytes(f"immutable-{kind}".encode("utf-8"))
        if kind != omit_binding_kind:
            bindings.append(
                {
                    "kind": kind,
                    "name": "tiny:model_artifact" if kind == "model" else kind,
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            )
    if omit_binding_kind != "dictionary":
        bindings.extend(dictionary_bindings)
        if omit_one_p0_2_binding:
            bindings.pop()

    factorial = {
        "identity_set_ids": ["identity-1"],
        "sites": list(SITES),
        "strengths": [1.0],
    }
    analysis = {
        "alpha": 0.05,
        "multiplicity": "benjamini_hochberg_all_paired_cells_positive_and_tost",
        "path_localization": "on_path_positive_off_path_equivalent",
        "measures": {
            measure: {
                "direction": (
                    "lower"
                    if measure
                    in {
                        "off_target_sparse_code_displacement",
                        "reconstruction_displacement",
                    }
                    else "higher"
                ),
                "equivalence_margin": MARGIN,
            }
            for measure in MEASURES
        },
    }
    identity_receipt_path = root / "identity_freeze_receipt.json"
    write_json(
        identity_receipt_path,
        {
            "schema_version": "r2_p0_7_pretrained_identity_freeze_receipt_v1",
            "status": "frozen_before_evaluation",
            "created_at_utc": "2026-07-17T00:00:00+00:00",
            "p0_2_eligibility_receipt_sha256": sha256_file(p0_2_path),
            "p0_5_extraction_receipt_sha256": sha256_file(p0_5_path),
            "discovery_cohort_sha256": sha256_file(discovery_path),
            "evaluation_cohort_sha256": sha256_file(evaluation_path),
            "frozen_feature_identities_sha256": sha256_file(identity_path),
            "analysis_contract_sha256": canonical_sha256(
                {"factorial": factorial, "analysis": analysis}
            ),
        },
    )
    evaluation_receipt_path = root / "evaluation_receipt.json"
    write_json(
        evaluation_receipt_path,
        {
            "schema_version": "r2_p0_7_pretrained_intervention_evaluation_receipt_v1",
            "status": "verified_complete",
            "completed_at_utc": "2026-07-17T01:00:00+00:00",
            "identity_freeze_receipt_sha256": sha256_file(identity_receipt_path),
            "p0_2_eligibility_receipt_sha256": sha256_file(p0_2_path),
            "p0_5_extraction_receipt_sha256": sha256_file(p0_5_path),
            "p0_6_execution_receipt_sha256": sha256_file(p0_6_path),
            "discovery_cohort_sha256": sha256_file(discovery_path),
            "evaluation_cohort_sha256": sha256_file(evaluation_path),
            "frozen_feature_identities_sha256": sha256_file(identity_path),
            "intervention_rows_sha256": sha256_file(intervention_path),
            "external_scores_sha256": sha256_file(score_path),
            "artifact_bindings": bindings,
            "scorers": {
                "behavior_endpoint": {
                    "validated": True,
                    "name": "fixture-behavior",
                    "version": "1",
                    "method": "validated_behavior_endpoint",
                    "scorer_sha256": next(
                        row["sha256"]
                        for row in bindings
                        if row["kind"] == "behavior_scorer"
                    ),
                    "calibration_cohort_sha256": next(
                        row["sha256"]
                        for row in bindings
                        if row["kind"] == "behavior_calibration"
                    ),
                },
                "path_endpoint": {
                    "validated": True,
                    "name": "fixture-path",
                    "version": "1",
                    "method": "path_patching",
                    "scorer_sha256": next(
                        row["sha256"]
                        for row in bindings
                        if row["kind"] == "path_scorer"
                    ),
                    "calibration_cohort_sha256": next(
                        row["sha256"]
                        for row in bindings
                        if row["kind"] == "path_calibration"
                    ),
                },
            },
        },
    )
    source_hashes = {split: digest(split) for split in ("train", "validation", "test")}
    spec_path = root / "spec.json"
    write_json(
        spec_path,
        {
            "schema_version": "r2_p0_7_pretrained_causal_adjudication_spec_v1",
            "mode": "production",
            "positive_control": {
                "run_manifest": descriptor(positive_manifest),
                "summary": descriptor(positive_summary),
            },
            "p0_2": {
                "eligibility_receipt": descriptor(p0_2_path),
                "models": [
                    {
                        "name": "tiny",
                        "method": "topk_clt",
                        "run_manifest_sha256_by_seed": run_hashes,
                        "checkpoint_sha256_by_seed": checkpoint_hashes,
                        "source_manifest_sha256_by_split": source_hashes,
                        "requested_layers": [1],
                    }
                ],
            },
            "upstream_receipts": {
                "p0_5_extraction": descriptor(p0_5_path),
                "p0_6_execution": descriptor(p0_6_path),
            },
            "artifacts": {
                "identity_freeze_receipt": descriptor(identity_receipt_path),
                "evaluation_receipt": descriptor(evaluation_receipt_path),
                "discovery_cohort": descriptor(discovery_path),
                "evaluation_cohort": descriptor(evaluation_path),
                "frozen_feature_identities": descriptor(identity_path),
                "intervention_rows": descriptor(intervention_path),
                "external_scores": descriptor(score_path),
            },
            "factorial": factorial,
            "analysis": analysis,
        },
    )
    return SimpleNamespace(
        root=root,
        spec_path=spec_path,
        spec_sha256=sha256_file(spec_path),
        positive_summary=positive_summary,
        p0_2_path=p0_2_path,
        evaluation_receipt_path=evaluation_receipt_path,
        code_binding=root / "binding_code.bin",
        p0_5_data=p0_5_data,
        p0_6_data=p0_6_data,
    )


def load_complete_fixture(fixture: SimpleNamespace) -> tuple[dict, dict, list[dict]]:
    spec = load_adjudication_spec(fixture.spec_path, fixture.spec_sha256)
    inputs = load_bound_inputs(spec)
    rows = validate_factorial_inventory(spec, inputs)
    return spec, inputs, rows


def test_complete_adjudication_is_raw_paired_global_bh_and_path_localized(
    tmp_path: Path,
) -> None:
    fixture = build_fixture(tmp_path / "fixture")
    spec, inputs, rows = load_complete_fixture(fixture)
    assert inputs["positive_control"]["status"] == "prospective_synthetic_gate_passed"
    assert inputs["disjointness"]["selection_evaluation_disjoint"] is True
    assert len(rows) == 12 * 2 * 2
    assert {row["path_site"] for row in rows} == {SITES[1]}

    analysis = analyze_paired_cells(spec, rows)
    assert len(analysis["paired_rows"]) == 12 * 2
    assert all(len(cell["evaluation_ids"]) == 12 for cell in analysis["cells"])
    assert len(analysis["path_results"]) == 1
    path_result = analysis["path_results"][0]
    assert path_result["path_site"] == SITES[1]
    assert path_result["off_path_sites"] == [SITES[0]]
    assert path_result["path_localized"] is True
    assert path_result["intervention_fidelity_passed"] is True
    assert path_result["off_path_behavior_and_logit_equivalent"] is True
    assert path_result["status"] == "localized_positive"
    cells = analysis["cells"]
    expected_q = benjamini_hochberg(
        [cell["positive_p"] for cell in cells]
        + [cell["equivalence"]["p_lower"] for cell in cells]
        + [cell["equivalence"]["p_upper"] for cell in cells]
    )
    n_cells = len(cells)
    assert analysis["multiplicity_family_size"] == 3 * n_cells
    np.testing.assert_allclose(
        [cell["positive_q_bh"] for cell in cells], expected_q[:n_cells]
    )
    np.testing.assert_allclose(
        [cell["equivalence"]["p_lower_q_bh"] for cell in cells],
        expected_q[n_cells : 2 * n_cells],
    )
    np.testing.assert_allclose(
        [cell["equivalence"]["p_upper_q_bh"] for cell in cells],
        expected_q[2 * n_cells :],
    )
    assert all(
        cell["equivalence"]["equivalence_band"] == [-MARGIN, MARGIN] for cell in cells
    )

    output = tmp_path / "adjudication"
    receipt_path = adjudicate_pretrained_causal_interventions(
        fixture.spec_path,
        fixture.spec_sha256,
        output,
        command=["python", "scripts/71_adjudicate_pretrained_causal_interventions.py"],
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "verified_complete"
    assert receipt["scientific_status"] == "pretrained_target_gate_resolved"
    assert receipt["raw_rows_retained_before_cell_inference"] is True
    assert receipt["positive_and_tost_multiplicity_corrected"] is True
    assert summary["n_raw_intervention_rows"] == 48
    assert summary["n_raw_paired_rows"] == 24
    assert summary["multiplicity_family_size"] == 3 * summary["n_cells"]
    assert all(
        sha256_file(output / name) == value
        for name, value in receipt["artifact_hashes"].items()
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        adjudicate_pretrained_causal_interventions(
            fixture.spec_path, fixture.spec_sha256, output
        )


@pytest.mark.parametrize(
    ("fixture_kwargs", "match"),
    [
        ({"positive_pass": False}, "positive-control result"),
        ({"p0_2_complete": False}, "eligibility receipt"),
        ({"overlap": "id"}, "overlap detected"),
        ({"overlap": "sequence"}, "overlap detected"),
        ({"omit_binding_kind": "code"}, "lacks .* bindings"),
        ({"omit_one_p0_2_binding": True}, "omit a P0-2"),
    ],
)
def test_bound_inputs_reject_failed_or_incomplete_prerequisites(
    tmp_path: Path, fixture_kwargs: dict, match: str
) -> None:
    fixture = build_fixture(tmp_path / "fixture", **fixture_kwargs)
    spec = load_adjudication_spec(fixture.spec_path, fixture.spec_sha256)
    with pytest.raises((ValueError, RuntimeError), match=match):
        load_bound_inputs(spec)


@pytest.mark.parametrize(
    "artifact_name",
    [
        "positive_summary",
        "evaluation_receipt_path",
        "code_binding",
        "p0_5_data",
        "p0_6_data",
    ],
)
def test_bound_inputs_reject_stale_or_tampered_receipts_and_code(
    tmp_path: Path, artifact_name: str
) -> None:
    fixture = build_fixture(tmp_path / "fixture")
    spec = load_adjudication_spec(fixture.spec_path, fixture.spec_sha256)
    path = Path(getattr(fixture, artifact_name))
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="changed"):
        load_bound_inputs(spec)


@pytest.mark.parametrize(
    "case",
    [
        "missing_cell",
        "duplicate_cell",
        "early_average",
        "target_feature",
        "control_feature",
        "model",
        "dictionary_seed",
        "layer",
        "site",
        "strength",
        "sequence_sha256",
        "missing_score",
    ],
)
def test_factorial_inventory_rejects_incomplete_or_substituted_raw_cells(
    tmp_path: Path, case: str
) -> None:
    fixture = build_fixture(tmp_path / "fixture")
    spec = load_adjudication_spec(fixture.spec_path, fixture.spec_sha256)
    inputs = load_bound_inputs(spec)
    altered = copy.deepcopy(inputs)
    if case == "missing_cell":
        altered["intervention_rows"].pop()
    elif case == "duplicate_cell":
        duplicate = dict(altered["intervention_rows"][0])
        duplicate["row_id"] = "duplicate-row-id"
        altered["intervention_rows"].append(duplicate)
    elif case == "early_average":
        altered["intervention_rows"] = [{"model": "tiny", "mean": 0.1}]
    elif case == "missing_score":
        altered["external_scores"].pop()
    elif case in {"target_feature", "control_feature"}:
        role = case.removesuffix("_feature")
        row = next(
            row for row in altered["intervention_rows"] if row["feature_role"] == role
        )
        row["feature"] += 100
    else:
        replacement = {
            "model": "other-model",
            "dictionary_seed": 29,
            "layer": 0,
            "site": "unfrozen-site",
            "strength": 3.0,
            "sequence_sha256": digest("substituted-sequence"),
        }[case]
        altered["intervention_rows"][0][case] = replacement
    with pytest.raises(ValueError):
        validate_factorial_inventory(spec, altered)


@pytest.mark.parametrize("measure", MEASURES)
def test_factorial_inventory_rejects_every_nonfinite_endpoint(
    tmp_path: Path, measure: str
) -> None:
    fixture = build_fixture(tmp_path / "fixture")
    spec = load_adjudication_spec(fixture.spec_path, fixture.spec_sha256)
    inputs = load_bound_inputs(spec)
    if measure in MEASURES[:4]:
        inputs["intervention_rows"][0][measure] = float("nan")
    else:
        inputs["external_scores"][0][measure] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        validate_factorial_inventory(spec, inputs)


def test_inconclusive_intervention_fidelity_blocks_gate_resolution(
    tmp_path: Path,
) -> None:
    fixture = build_fixture(tmp_path / "fixture")
    spec, _, rows = load_complete_fixture(fixture)
    for row in rows:
        if row["feature_role"] == "target" and row["site"] == SITES[1]:
            row["intended_feature_change"] = 0.0
    analysis = analyze_paired_cells(spec, rows)
    path_result = analysis["path_results"][0]
    assert path_result["path_localized"] is True
    assert path_result["intervention_fidelity_passed"] is False
    assert path_result["status"] == "inconclusive"
    assert analysis["all_required_contrasts_resolved"] is False


def test_atomic_failure_leaves_no_output_or_completion_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_fixture(tmp_path / "fixture")
    output = tmp_path / "failed_adjudication"
    real_write_json = causal_adjudication.write_json

    def fail_on_summary(path: Path, payload: object) -> None:
        if Path(path).name == "summary.json":
            raise RuntimeError("planted publication failure")
        real_write_json(path, payload)

    monkeypatch.setattr(causal_adjudication, "write_json", fail_on_summary)
    with pytest.raises(RuntimeError, match="planted publication failure"):
        adjudicate_pretrained_causal_interventions(
            fixture.spec_path, fixture.spec_sha256, output
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".failed_adjudication.staging-*"))


def test_midrun_bound_artifact_mutation_fails_final_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_fixture(tmp_path / "fixture")
    output = tmp_path / "mutated_adjudication"
    real_analyze = causal_adjudication.analyze_paired_cells

    def mutate_after_analysis(spec: dict, rows: list[dict]) -> dict:
        result = real_analyze(spec, rows)
        fixture.code_binding.write_bytes(fixture.code_binding.read_bytes() + b"changed")
        return result

    monkeypatch.setattr(
        causal_adjudication, "analyze_paired_cells", mutate_after_analysis
    )
    with pytest.raises(RuntimeError, match="inputs or source changed"):
        adjudicate_pretrained_causal_interventions(
            fixture.spec_path, fixture.spec_sha256, output
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".mutated_adjudication.staging-*"))
