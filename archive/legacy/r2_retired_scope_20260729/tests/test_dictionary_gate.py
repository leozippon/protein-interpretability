from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

import pytest
import torch


R2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R2))

from src.revision.dictionary_controls import WindowedTranscoder  # noqa: E402
from src.revision.dictionary_gate import (  # noqa: E402
    GateError,
    _compatible_clt_from_topk_checkpoint,
    adjudicate_dictionary_panel,
    require_eligible_model_method,
    sha256_file,
    write_eligibility_receipt,
)


MODELS = ("protgpt2", "zymctrl", "progen2-medium")
METHODS = ("topk_clt", "relu_l1_sae", "gated_sae", "dense_low_rank")
SCREENED = ("relu_l1_sae", "gated_sae")
SEEDS = (17, 29, 43)
QUANTILES = {
    "q00": 0.1,
    "q25": 0.1,
    "q50": 0.1,
    "q75": 0.1,
    "q90": 0.1,
    "q95": 0.1,
    "q99": 0.1,
    "q100": 0.1,
}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def candidate_summary(stage: str, model_seed: int, stream_seed: int) -> dict:
    return {
        "stage": stage,
        "model_seed": model_seed,
        "batch_stream_seed": stream_seed,
        "training": {"test_split_accesses_during_training": 0},
    }


def run_manifest(
    *,
    status: str,
    stage: str,
    model: str,
    method: str,
    seed: int,
    result_sha: str,
    checkpoint_sha: str,
    cache: dict,
    module_sha: str,
    profile_sha: str,
    model_seed: int,
    stream_seed: int,
) -> dict:
    return {
        "schema_version": "r2_dictionary_control_run_manifest_v2",
        "status": status,
        "stage": stage,
        "model_name": model,
        "method": method,
        "run_seed": seed,
        "command": ["python", "scripts/58_run_dictionary_controls.py"],
        "gpu_model": "NVIDIA H200",
        "executed_profile_sha256": profile_sha,
        "result_sha256": result_sha,
        "checkpoint_sha256": checkpoint_sha,
        "module_sha256": module_sha,
        "cache_manifest_sha256": cache["manifest_sha256"],
        "cache_content_sha256": cache["content_sha256"],
        "source_manifest_sha256_by_split": cache["source_hashes"],
        "selected_layers": cache["layers"],
        "valid_token_budget_by_split": {
            "train": 1_000_000,
            "validation": 100_000,
            "test": 100_000,
        },
        "cache_storage_dtype": "float16",
        "cache_layout": "preallocated_single_file_per_layer_split",
        "decoder_window": 8,
        "feature_input_hook": "hook_resid_mid",
        "feature_output_hook": "hook_mlp_out",
        "model_seed": model_seed,
        "batch_stream_seed": stream_seed,
    }


def build_cache(
    tmp_path: Path, model: str, n_layers: int, profile_sha: str
) -> tuple[dict, dict]:
    root = tmp_path / "caches" / model
    sources: dict[str, dict] = {}
    selections: dict[str, dict] = {}
    source_hashes: dict[str, str] = {}
    budgets = {"train": 1_000_000, "validation": 100_000, "test": 100_000}
    for split in ("train", "validation", "test"):
        source = root / "source_manifests" / f"{split}.jsonl"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f'{{"model":"{model}","split":"{split}"}}\n')
        source_sha = sha256_file(source)
        source_hashes[split] = source_sha
        sources[split] = {
            "manifest_path": str(source.relative_to(root)),
            "manifest_sha256": source_sha,
        }
        selection = root / "selections" / f"{split}.jsonl"
        selection.parent.mkdir(parents=True, exist_ok=True)
        selection.write_text(f'{{"model":"{model}","split":"{split}"}}\n')
        selection_sha = sha256_file(selection)
        selections[split] = {
            "budget": budgets[split],
            "eligible_valid_token_rows": budgets[split] + 1,
            "selection_path": str(selection.relative_to(root)),
            "selection_sha256": selection_sha,
            "row_order_path": str(selection.relative_to(root)),
            "row_order_sha256": selection_sha,
        }
    layers = list(range(n_layers))
    fingerprints = {split: digest(f"fingerprint-{model}-{split}") for split in budgets}
    payload = {
        "schema_version": "r2_dictionary_activation_cache_v1",
        "objective": "transcode",
        "selected_layers": layers,
        "storage_dtype": "float16",
        "layout": "preallocated_single_file_per_layer_split",
        "mask_contract": {
            "attention_mask_required": True,
            "invalid_rows": "excluded_before_serialization",
            "all_valid_rows": "serialized_without_special_case",
        },
        "content_sha256": digest(f"cache-content-{model}"),
        "source_splits": sources,
        "token_selection": {
            "status": "complete_budgeted_selection",
            "method": "lowest_sha256_record_digest_colon_unpadded_token_position",
            "position_definition": "ordinal_in_unpadded_model_token_stream",
            "serialization_order": "selection_priority_order",
            "by_split": selections,
        },
        "split_summaries": {
            split: {
                "selected_valid_token_rows": budget,
                "valid_token_rows": budget,
                "rows_per_layer": {str(layer): budget for layer in layers},
            }
            for split, budget in budgets.items()
        },
        "activation_provenance": {
            "status": "complete",
            "model_name": model,
            "executed_profile_sha256": profile_sha,
            "valid_token_budget_by_split": budgets,
            "selected_layers": layers,
            "feature_input_hook": "hook_resid_mid",
            "feature_output_hook": "hook_mlp_out",
            "decoder_window": 8,
            "runtime_tokenizer": {
                "padding_side": "right",
                "truncation_side": "right",
            },
            "first_pass_tokenization_sha256_by_split": fingerprints,
            "second_pass_tokenization_sha256_by_split": fingerprints,
        },
        "shards": [
            {
                "split": split,
                "layer": layer,
                "rows": budgets[split],
                "dtype": "float16",
                "input_sha256": digest(f"input-{model}-{split}-{layer}"),
                "target_sha256": digest(f"target-{model}-{split}-{layer}"),
            }
            for split in budgets
            for layer in layers
        ],
    }
    manifest = root / "manifest.json"
    write_json(manifest, payload)
    metadata = {
        "manifest_sha256": sha256_file(manifest),
        "content_sha256": payload["content_sha256"],
        "source_hashes": source_hashes,
        "layers": layers,
    }
    return {"model_name": model, "manifest": descriptor(manifest)}, metadata


def quality_result(layers: list[int], activation_threshold: float) -> dict:
    fvu = [0.1 for _ in layers]
    dead = [0.1 for _ in layers]
    l0 = [128.0 for _ in layers]
    return {
        "split": "test",
        "objective": "windowed_multi_layer_clt_input_to_mlp_output_transcoding",
        "selected_layers": layers,
        "decoder_window": 8,
        "n_valid_tokens": 100_000,
        "fvu_mean": statistics.fmean(fvu),
        "fvu_pooled_layer_centered": 0.1,
        "mse_mean": 0.1,
        "l0_mean": statistics.fmean(l0),
        "activation_threshold": activation_threshold,
        "dead_frequency_threshold": 0.001,
        "dead_fraction_median": statistics.median(dead),
        "target_layers": [
            {"layer": layer, "mse": 0.1, "fvu": fvu[index]}
            for index, layer in enumerate(layers)
        ],
        "source_layers": [
            {
                "layer": layer,
                "l0_mean": l0[index],
                "dead_fraction": dead[index],
                "firing_frequency_quantiles": QUANTILES,
                "decoder_norm_quantiles": QUANTILES,
            }
            for index, layer in enumerate(layers)
        ],
        "reconstruction_error_quantiles": QUANTILES,
    }


def add_artifact(
    tmp_path: Path,
    *,
    stage: str,
    model: str,
    method: str,
    seed: int,
    profile: dict,
    profile_sha: str,
    cache: dict,
    module_sha: str,
    screening: dict | None,
) -> tuple[dict, dict]:
    directory = tmp_path / stage / model / method / f"seed_{seed}"
    directory.mkdir(parents=True)
    checkpoint = directory / "selected_best.pt"
    checkpoint.write_bytes(f"checkpoint:{stage}:{model}:{method}:{seed}".encode())
    checkpoint_sha = sha256_file(checkpoint)
    model_seed = seed * 100 + METHODS.index(method)
    stream_seed = seed * 1000 + MODELS.index(model)
    summary = candidate_summary(stage, model_seed, stream_seed)
    if stage == "screening":
        candidate_count = profile["compute_schedule"]["screening"][
            f"{method.removesuffix('_sae')}_candidates_per_model"
        ]
        summaries = [dict(summary) for _ in range(candidate_count)]
        rows = [
            {
                "candidate_index": index,
                "l1_coefficient": 0.00001 * (index + 1),
                "auxiliary_coefficient": 0.1 if method == "gated_sae" else 0.0,
                "activation_threshold": 0.0,
                "validation_fvu_mean": 0.1 + index * 0.01,
                "validation_l0_mean": 128.0,
            }
            for index in range(candidate_count)
        ]
        selected = rows[0]
        status = "completed_validation_screening"
        result = {
            "schema_version": "r2_dictionary_control_results_v1",
            "status": status,
            "p0_2_eligible": False,
            "stage": stage,
            "method": method,
            "model_name": model,
            "run_seed": seed,
            "cache_content_sha256": cache["content_sha256"],
            "profile_sha256": profile_sha,
            "training_cache_priority_prefix_rows": 100_000,
            "candidate_validation": summaries,
            "selection_rows": rows,
            "selected_validation_configuration": selected,
            "test_evaluation_count": 0,
            "selected_checkpoint": {"sha256": checkpoint_sha},
        }
    else:
        status = "completed_confirmatory_control"
        if screening is None:
            frozen = {
                "candidate_index": 0,
                "l1_coefficient": 0.0,
                "auxiliary_coefficient": 0.0,
                "activation_threshold": 0.0,
            }
            screening_sha = None
        else:
            frozen = screening["selected_configuration"]
            screening_sha = screening["result_sha256"]
        selected = {
            **{
                key: frozen[key]
                for key in (
                    "candidate_index",
                    "l1_coefficient",
                    "auxiliary_coefficient",
                    "activation_threshold",
                )
            },
            "validation_fvu_mean": 0.1,
            "validation_l0_mean": 128.0,
        }
        result = {
            "schema_version": "r2_dictionary_control_results_v1",
            "status": status,
            "p0_2_eligible": None,
            "quality_gate_requires_all_seed_aggregation": True,
            "stage": stage,
            "method": method,
            "model_name": model,
            "run_seed": seed,
            "cache_content_sha256": cache["content_sha256"],
            "profile_sha256": profile_sha,
            "candidate_validation": [summary],
            "selection_rows": [selected],
            "selected_validation_configuration": selected,
            "frozen_screening_configuration": frozen,
            "screening_result_sha256": screening_sha,
            "heldout_test": quality_result(
                cache["layers"], frozen["activation_threshold"]
            ),
            "test_evaluation_count": 1,
            "resources": {
                "raw_trainable_parameter_count": profile["checkpoint_storage_planning"][
                    "raw_trainable_parameters_by_model_method"
                ][model][method],
                "training_flop_proxy_total": 1.0,
                "inference_flop_proxy_per_token": 1.0,
                "wall_time_seconds": 1.0,
                "peak_accelerator_memory_allocated_bytes": 1,
                "peak_accelerator_memory_reserved_bytes": 1,
            },
            "selected_checkpoint": {"sha256": checkpoint_sha},
        }
    result_path = directory / "results.json"
    write_json(result_path, result)
    result_sha = sha256_file(result_path)
    manifest = run_manifest(
        status=status,
        stage=stage,
        model=model,
        method=method,
        seed=seed,
        result_sha=result_sha,
        checkpoint_sha=checkpoint_sha,
        cache=cache,
        module_sha=module_sha,
        profile_sha=profile_sha,
        model_seed=model_seed,
        stream_seed=stream_seed,
    )
    manifest_path = directory / "run_manifest.json"
    write_json(manifest_path, manifest)
    entry = {
        "model_name": model,
        "method": method,
        "run_seed": seed,
        "run_manifest": descriptor(manifest_path),
        "result": descriptor(result_path),
        "checkpoint": descriptor(checkpoint),
    }
    return entry, {
        "selected_configuration": selected,
        "result_sha256": result_sha,
    }


def build_panel(tmp_path: Path) -> tuple[Path, dict]:
    profile_path = R2 / "configs/p0_2_dictionary_controls_production_profile.json"
    protocol_path = R2 / "docs/P0_2_DICTIONARY_PROTOCOL_20260717.md"
    profile = json.loads(profile_path.read_text())
    profile_sha = sha256_file(profile_path)
    module = tmp_path / "code" / "dictionary_controls.py"
    tests = tmp_path / "code" / "test_dictionary_controls.py"
    module.parent.mkdir(parents=True)
    module.write_text("# tested module\n")
    tests.write_text("# mask tests\n")
    mask_receipt = tmp_path / "mask_validation_receipt.json"
    write_json(
        mask_receipt,
        {
            "schema_version": "r2_p0_2_mask_validation_receipt_v1",
            "status": "complete",
            "confirmatory": True,
            "pytest_exit_code": 0,
            "production_scientific_eligibility": True,
            "module": descriptor(module),
            "test_file": descriptor(tests),
            "tests": {
                "test_valid_token_cache_is_padding_invariant_and_all_valid_equivalent": True,
                "test_windowed_metrics_are_padding_invariant_end_to_end": True,
            },
        },
    )
    cache_specs: list[dict] = []
    caches: dict[str, dict] = {}
    for model in MODELS:
        n_layers = profile["cache_extraction"]["model_cache_geometry"][model][
            "n_layers"
        ]
        cache_spec, cache = build_cache(tmp_path, model, n_layers, profile_sha)
        cache_specs.append(cache_spec)
        caches[model] = cache
    screening_entries: list[dict] = []
    screening: dict[tuple[str, str], dict] = {}
    for model in MODELS:
        for method in SCREENED:
            entry, metadata = add_artifact(
                tmp_path,
                stage="screening",
                model=model,
                method=method,
                seed=20260717,
                profile=profile,
                profile_sha=profile_sha,
                cache=caches[model],
                module_sha=sha256_file(module),
                screening=None,
            )
            screening_entries.append(entry)
            screening[(model, method)] = metadata
    full_entries: list[dict] = []
    for model in MODELS:
        for method in METHODS:
            for seed in SEEDS:
                entry, _ = add_artifact(
                    tmp_path,
                    stage="full",
                    model=model,
                    method=method,
                    seed=seed,
                    profile=profile,
                    profile_sha=profile_sha,
                    cache=caches[model],
                    module_sha=sha256_file(module),
                    screening=screening.get((model, method)),
                )
                full_entries.append(entry)
    spec = {
        "schema_version": "r2_p0_2_dictionary_gate_spec_v1",
        "confirmatory": True,
        "profile": descriptor(profile_path),
        "protocol": descriptor(protocol_path),
        "mask_validation_receipt": descriptor(mask_receipt),
        "caches": cache_specs,
        "screening_runs": screening_entries,
        "full_runs": full_entries,
    }
    spec_path = tmp_path / "gate_spec.json"
    write_json(spec_path, spec)
    return spec_path, spec


def update_full_result(spec_path: Path, identity: tuple[str, str, int], mutate) -> None:
    spec = json.loads(spec_path.read_text())
    entry = next(
        row
        for row in spec["full_runs"]
        if (row["model_name"], row["method"], row["run_seed"]) == identity
    )
    result_path = Path(entry["result"]["path"])
    result = json.loads(result_path.read_text())
    mutate(result)
    write_json(result_path, result)
    entry["result"] = descriptor(result_path)
    manifest_path = Path(entry["run_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    manifest["result_sha256"] = entry["result"]["sha256"]
    write_json(manifest_path, manifest)
    entry["run_manifest"] = descriptor(manifest_path)
    write_json(spec_path, spec)


def mark_screening_sparsity_match_failure(
    spec_path: Path, identity: tuple[str, str, int], *, remove_full_runs: bool
) -> None:
    spec = json.loads(spec_path.read_text())
    entry = next(
        row
        for row in spec["screening_runs"]
        if (row["model_name"], row["method"], row["run_seed"]) == identity
    )
    result_path = Path(entry["result"]["path"])
    result = json.loads(result_path.read_text())
    result["status"] = "sparsity_match_failure"
    result["reason"] = "no validation candidate achieved L0 within [115.2, 140.8]"
    result.pop("selected_checkpoint")
    result.pop("selected_validation_configuration")
    result.pop("training_cache_priority_prefix_rows")
    for row in result["selection_rows"]:
        row["validation_l0_mean"] = 200.0
    write_json(result_path, result)
    entry["result"] = descriptor(result_path)
    entry["checkpoint"] = None

    manifest_path = Path(entry["run_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "sparsity_match_failure"
    manifest["result_sha256"] = entry["result"]["sha256"]
    manifest["checkpoint_sha256"] = None
    write_json(manifest_path, manifest)
    entry["run_manifest"] = descriptor(manifest_path)

    if remove_full_runs:
        model, method, _ = identity
        spec["full_runs"] = [
            row
            for row in spec["full_runs"]
            if (row["model_name"], row["method"]) != (model, method)
        ]
    write_json(spec_path, spec)


def test_complete_panel_emits_hash_bound_pass_receipt(tmp_path: Path) -> None:
    spec_path, _ = build_panel(tmp_path)
    receipt, manifest = adjudicate_dictionary_panel(spec_path)
    assert receipt["status"] == "complete"
    assert receipt["p0_2_panel_eligible"] is True
    assert len(receipt["model_method_adjudications"]) == 12
    assert len(manifest["full_artifact_hashes"]) == 36

    output = tmp_path / "receipt"
    receipt_path, _ = write_eligibility_receipt(spec_path, output)
    topk = next(
        row
        for row in receipt["model_method_adjudications"]
        if row["model_name"] == "protgpt2" and row["method"] == "topk_clt"
    )
    selected = require_eligible_model_method(
        receipt_path,
        sha256_file(receipt_path),
        model_name="protgpt2",
        method="topk_clt",
        expected_run_manifest_sha256_by_seed={
            row["run_seed"]: row["run_manifest_sha256"] for row in topk["runs"]
        },
        expected_checkpoint_sha256_by_seed={
            row["run_seed"]: row["checkpoint_sha256"] for row in topk["runs"]
        },
        expected_source_manifest_sha256_by_split=topk[
            "source_manifest_sha256_by_split"
        ],
        requested_layers=[0],
    )
    assert selected["atlas_eligible"] is True
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_eligibility_receipt(spec_path, output)


def test_missing_seed_is_rejected_without_receipt(tmp_path: Path) -> None:
    spec_path, spec = build_panel(tmp_path)
    spec["full_runs"] = spec["full_runs"][:-1]
    write_json(spec_path, spec)
    with pytest.raises(GateError, match="missing=.*43"):
        write_eligibility_receipt(spec_path, tmp_path / "receipt")
    assert not (tmp_path / "receipt").exists()


def test_failed_quality_yields_complete_negative_adjudication(tmp_path: Path) -> None:
    spec_path, _ = build_panel(tmp_path)

    def fail_fvu(result: dict) -> None:
        heldout = result["heldout_test"]
        heldout["fvu_mean"] = 0.6
        for row in heldout["target_layers"]:
            row["fvu"] = 0.6

    update_full_result(spec_path, ("protgpt2", "topk_clt", 17), fail_fvu)
    receipt, _ = adjudicate_dictionary_panel(spec_path)
    assert receipt["status"] == "complete"
    assert receipt["p0_2_panel_eligible"] is False
    topk = next(
        row
        for row in receipt["model_method_adjudications"]
        if row["model_name"] == "protgpt2" and row["method"] == "topk_clt"
    )
    assert topk["atlas_eligible"] is False
    assert topk["runs"][0]["quality_gate_pass"] is False


def test_sparsity_match_failure_is_terminal_negative_without_full_runs(
    tmp_path: Path,
) -> None:
    spec_path, _ = build_panel(tmp_path)
    mark_screening_sparsity_match_failure(
        spec_path,
        ("protgpt2", "gated_sae", 20260717),
        remove_full_runs=True,
    )

    receipt, manifest = adjudicate_dictionary_panel(spec_path)

    assert receipt["status"] == "complete"
    assert receipt["artifact_completeness"] is True
    assert receipt["p0_2_panel_eligible"] is False
    assert receipt["required_full_run_count"] == 33
    gated = next(
        row
        for row in receipt["model_method_adjudications"]
        if row["model_name"] == "protgpt2" and row["method"] == "gated_sae"
    )
    assert gated["status"] == "sparsity_match_failure"
    assert gated["atlas_eligible"] is False
    assert gated["failure_reasons"] == ["validation_sparsity_match_failure"]
    assert gated["eligible_downstream_layers"] == []
    assert gated["runs"] == []
    assert not any(
        row["model_name"] == "protgpt2" and row["method"] == "gated_sae"
        for row in manifest["full_artifact_hashes"]
    )
    screening = next(
        row
        for row in manifest["screening_artifact_hashes"]
        if row["model_name"] == "protgpt2" and row["method"] == "gated_sae"
    )
    assert screening["checkpoint_sha256"] is None

    receipt_path, _ = write_eligibility_receipt(spec_path, tmp_path / "receipt")
    with pytest.raises(GateError, match="did not pass the all-seed P0-2 gate"):
        require_eligible_model_method(
            receipt_path,
            sha256_file(receipt_path),
            model_name="protgpt2",
            method="gated_sae",
        )


def test_sparsity_match_failure_rejects_forbidden_full_runs(tmp_path: Path) -> None:
    spec_path, _ = build_panel(tmp_path)
    mark_screening_sparsity_match_failure(
        spec_path,
        ("protgpt2", "gated_sae", 20260717),
        remove_full_runs=False,
    )

    with pytest.raises(GateError, match="full_runs panel mismatch.*extra"):
        adjudicate_dictionary_panel(spec_path)


def test_padding_contamination_is_rejected(tmp_path: Path) -> None:
    spec_path, _ = build_panel(tmp_path)
    update_full_result(
        spec_path,
        ("protgpt2", "topk_clt", 17),
        lambda result: result["heldout_test"].update(n_valid_tokens=99_999),
    )
    with pytest.raises(GateError, match="padding-clean test split"):
        adjudicate_dictionary_panel(spec_path)


def test_test_leakage_is_rejected(tmp_path: Path) -> None:
    spec_path, _ = build_panel(tmp_path)
    update_full_result(
        spec_path,
        ("protgpt2", "topk_clt", 17),
        lambda result: result["candidate_validation"][0]["training"].update(
            test_split_accesses_during_training=1
        ),
    )
    with pytest.raises(GateError, match="training accessed test data"):
        adjudicate_dictionary_panel(spec_path)


def test_result_hash_tamper_is_rejected(tmp_path: Path) -> None:
    spec_path, spec = build_panel(tmp_path)
    path = Path(spec["full_runs"][0]["result"]["path"])
    path.write_text(path.read_text() + " ")
    with pytest.raises(GateError, match="result SHA-256 mismatch"):
        adjudicate_dictionary_panel(spec_path)


def test_preflight_artifact_is_rejected(tmp_path: Path) -> None:
    spec_path, _ = build_panel(tmp_path)

    def mark_preflight(result: dict) -> None:
        result["status"] = "completed_nonconfirmatory_preflight"

    update_full_result(spec_path, ("protgpt2", "topk_clt", 17), mark_preflight)
    with pytest.raises(GateError, match="result identity changed"):
        adjudicate_dictionary_panel(spec_path)


def test_topk_checkpoint_maps_exactly_to_clt_geometry() -> None:
    torch.manual_seed(5)
    windowed = WindowedTranscoder(
        method="topk_clt",
        n_layers=3,
        input_dim=4,
        target_dim=4,
        width=7,
        window=2,
        topk_k=3,
    )
    candidate_id = "fixture-topk"
    checkpoint = {
        "schema_version": "r2_dictionary_control_best_v1",
        "candidate_id": candidate_id,
        "step": 5_000,
        "validation_fvu_mean": 0.2,
        "model_state_dict": {
            name: tensor.detach().clone()
            for name, tensor in windowed.state_dict().items()
        },
    }
    clt = _compatible_clt_from_topk_checkpoint(
        checkpoint,
        expected_candidate_id=candidate_id,
        geometry={"n_layers": 3, "d_model": 4, "d_clt": 7, "k": 3, "window": 2},
    )
    inputs = [torch.randn(2, 5, 4) for _ in range(3)]
    expected_codes = windowed.encode([value.reshape(-1, 4) for value in inputs])
    observed_codes = clt.encode(inputs)
    for expected, observed in zip(expected_codes, observed_codes):
        torch.testing.assert_close(expected.reshape(2, 5, 7), observed)
    expected_outputs = windowed.decode(expected_codes)
    observed_outputs = clt.decode(observed_codes)
    for expected, observed in zip(expected_outputs, observed_outputs):
        torch.testing.assert_close(expected.reshape(2, 5, 4), observed)
    assert not any(parameter.requires_grad for parameter in clt.parameters())


def test_topk_checkpoint_adapter_rejects_identity_and_shape_tampering() -> None:
    windowed = WindowedTranscoder(
        method="topk_clt",
        n_layers=2,
        input_dim=3,
        target_dim=3,
        width=5,
        window=2,
        topk_k=2,
    )
    checkpoint = {
        "schema_version": "r2_dictionary_control_best_v1",
        "candidate_id": "expected",
        "step": 5_000,
        "validation_fvu_mean": 0.2,
        "model_state_dict": {
            name: tensor.detach().clone()
            for name, tensor in windowed.state_dict().items()
        },
    }
    geometry = {"n_layers": 2, "d_model": 3, "d_clt": 5, "k": 2, "window": 2}
    with pytest.raises(GateError, match="identity or training step"):
        _compatible_clt_from_topk_checkpoint(
            checkpoint, expected_candidate_id="wrong", geometry=geometry
        )
    checkpoint["model_state_dict"]["encoder_weight"] = torch.zeros(2, 5, 4)
    with pytest.raises(GateError, match="tensor is invalid: encoder_weight"):
        _compatible_clt_from_topk_checkpoint(
            checkpoint, expected_candidate_id="expected", geometry=geometry
        )
