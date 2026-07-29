"""Fail-closed adjudication of the frozen P0-2 exact-cache dictionary panel."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SPEC_SCHEMA = "r2_p0_2_dictionary_gate_spec_v1"
RECEIPT_SCHEMA = "r2_p0_2_eligibility_receipt_v1"
RECEIPT_MANIFEST_SCHEMA = "r2_p0_2_eligibility_receipt_manifest_v1"
RUN_MANIFEST_SCHEMA = "r2_dictionary_control_run_manifest_v2"
RESULT_SCHEMA = "r2_dictionary_control_results_v1"
CACHE_SCHEMA = "r2_dictionary_activation_cache_v1"
MASK_RECEIPT_SCHEMA = "r2_p0_2_mask_validation_receipt_v1"
FULL_STATUS = "completed_confirmatory_control"
SCREENING_STATUS = "completed_validation_screening"
SPARSITY_MATCH_FAILURE = "sparsity_match_failure"
SPLITS = ("train", "validation", "test")
MODELS = ("protgpt2", "zymctrl", "progen2-medium")
METHODS = ("topk_clt", "relu_l1_sae", "gated_sae", "dense_low_rank")
SCREENED_METHODS = ("relu_l1_sae", "gated_sae")
SEEDS = (17, 29, 43)
SCREENING_SEED = 20260717
QUANTILE_KEYS = ("q00", "q25", "q50", "q75", "q90", "q95", "q99", "q100")
MASK_TESTS = {
    "test_valid_token_cache_is_padding_invariant_and_all_valid_equivalent",
    "test_windowed_metrics_are_padding_invariant_end_to_end",
}

# These are the verbatim numerical gates frozen in the dated protocol. The
# protocol text is checked before these values are used; they are not tunable
# command-line parameters.
FVU_LIMIT = 0.50
FVU_LAYER_NUMERATOR = 3
FVU_LAYER_DENOMINATOR = 4
MEDIAN_DEAD_LIMIT = 0.50
DOWNSTREAM_DEAD_LIMIT = 0.70


class GateError(ValueError):
    """An input cannot support a complete, auditable P0-2 adjudication."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_strict_json(path: Path) -> dict[str, Any]:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    GateError(f"non-finite JSON constant: {token}")
                ),
            )
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"cannot read strict JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise GateError(f"JSON root must be an object: {path}")
    return value


def _exact_keys(value: object, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise GateError(f"{label} fields changed: {observed}")
    return value


def _resolve(base: Path, raw: object, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise GateError(f"{label} path must be a non-empty string")
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _verify_descriptor(
    descriptor: object,
    *,
    base: Path,
    label: str,
    json_file: bool,
) -> tuple[Path, str, dict[str, Any] | None]:
    record = _exact_keys(descriptor, {"path", "sha256"}, label)
    expected = record["sha256"]
    if not _is_digest(expected):
        raise GateError(f"{label} SHA-256 is invalid")
    path = _resolve(base, record["path"], label)
    if not path.is_file():
        raise GateError(f"{label} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise GateError(f"{label} SHA-256 mismatch")
    return path, observed, load_strict_json(path) if json_file else None


def _finite(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise GateError(f"{label} is invalid")
    return result


def _validate_profile(profile: Mapping[str, Any]) -> None:
    if (
        profile.get("schema_version")
        != "r2_p0_2_dictionary_controls_production_profile_v2"
        or profile.get("confirmatory") is not True
        or profile.get("status") != "amended_before_restart"
        or tuple(profile.get("models", ())) != MODELS
        or tuple(profile.get("seeds", ())) != SEEDS
        or tuple(profile.get("panel", {}).keys()) != METHODS
    ):
        raise GateError("profile is not the frozen confirmatory P0-2 panel")
    schedule = profile.get("compute_schedule", {})
    if (
        schedule.get("screening", {}).get("seed") != SCREENING_SEED
        or tuple(schedule.get("full", {}).get("seeds", ())) != SEEDS
        or schedule.get("screening", {}).get("test_access_count") != 0
        or schedule.get("full", {}).get("test_access_count_after_training") != 1
    ):
        raise GateError("profile seed or test-access schedule changed")
    cache = profile.get("cache_extraction", {})
    geometry = cache.get("model_cache_geometry", {})
    if (
        cache.get("valid_token_budget_by_split")
        != {"train": 1_000_000, "validation": 100_000, "test": 100_000}
        or {model: geometry.get(model, {}).get("n_layers") for model in MODELS}
        != {"protgpt2": 36, "zymctrl": 36, "progen2-medium": 27}
        or cache.get("storage_dtype") != "float16"
        or cache.get("model_inference_dtype") != "bfloat16"
        or cache.get("model_inference_dtype_verification")
        != "all_floating_model_parameters_exactly_declared_before_first_activation"
        or cache.get("activation_finiteness_check")
        != "all_clt_input_and_mlp_output_tensors_before_storage_conversion_and_write"
        or cache.get("layout") != "preallocated_single_file_per_layer_split"
    ):
        raise GateError("profile cache geometry or exact budgets changed")
    panel = profile["panel"]
    if (
        panel["topk_clt"].get("width") != 8192
        or panel["topk_clt"].get("k") != 128
        or panel["relu_l1_sae"].get("width") != 8192
        or panel["gated_sae"].get("width") != 8192
        or panel["dense_low_rank"].get("rank") != 128
        or profile.get("validation_only_selection", {}).get("eligible_l0_interval")
        != [115.2, 140.8]
        or profile.get("validation_only_selection", {}).get(
            "no_eligible_candidate_action"
        )
        != "report_sparsity_match_failure"
    ):
        raise GateError("profile panel width, rank, TopK or L0 contract changed")


def _validate_protocol(text: str) -> None:
    required = (
        "all padding-invariance and all-valid-equivalence tests pass",
        "mean held-out test FVU is below 0.50 in every seed",
        "at least 75% of layers have held-out FVU below 0.50 in every seed",
        "median dead-feature fraction is below 0.50 in every seed",
        "dead-feature fraction below 0.70 in every\n   seed",
        "Run independent\nseeds `17`, `29` and `43`",
        "Test data are evaluated once after validation-only selection",
    )
    missing = [phrase for phrase in required if phrase not in text]
    if missing:
        raise GateError(f"frozen protocol gate text changed: {missing}")


def _validate_mask_receipt(
    payload: Mapping[str, Any], *, receipt_path: Path
) -> tuple[bool, str, dict[str, str]]:
    fields = {
        "schema_version",
        "status",
        "confirmatory",
        "pytest_exit_code",
        "production_scientific_eligibility",
        "module",
        "test_file",
        "tests",
    }
    _exact_keys(payload, fields, "mask-validation receipt")
    if (
        payload["schema_version"] != MASK_RECEIPT_SCHEMA
        or payload["status"] != "complete"
        or payload["confirmatory"] is not True
    ):
        raise GateError("mask-validation receipt is incomplete or nonconfirmatory")
    tests = _exact_keys(payload["tests"], MASK_TESTS, "mask-validation tests")
    if any(type(value) is not bool for value in tests.values()):
        raise GateError("mask-validation test outcomes must be booleans")
    passed = all(tests.values())
    exit_code = payload["pytest_exit_code"]
    if (
        payload["production_scientific_eligibility"] is not passed
        or type(exit_code) is not int
        or (passed and exit_code != 0)
        or (not passed and exit_code == 0)
    ):
        raise GateError("mask receipt outcome fields are internally inconsistent")
    module_path, module_sha, _ = _verify_descriptor(
        payload["module"],
        base=receipt_path.parent,
        label="mask-tested dictionary module",
        json_file=False,
    )
    test_path, test_sha, _ = _verify_descriptor(
        payload["test_file"],
        base=receipt_path.parent,
        label="mask test file",
        json_file=False,
    )
    return (
        passed,
        module_sha,
        {
            "module_sha256": module_sha,
            "test_file_sha256": test_sha,
            "module_name": module_path.name,
            "test_file_name": test_path.name,
        },
    )


def _cache_member(root: Path, raw: object, label: str) -> Path:
    path = _resolve(root, raw, label)
    if not path.is_relative_to(root.resolve()):
        raise GateError(f"{label} escapes the cache root")
    return path


def _validate_cache(
    *,
    model: str,
    descriptor: object,
    spec_base: Path,
    profile: Mapping[str, Any],
    profile_sha256: str,
) -> dict[str, Any]:
    path, manifest_sha, payload_raw = _verify_descriptor(
        descriptor,
        base=spec_base,
        label=f"{model} cache manifest",
        json_file=True,
    )
    assert payload_raw is not None
    payload = payload_raw
    geometry = profile["cache_extraction"]["model_cache_geometry"][model]
    layers = list(range(geometry["n_layers"]))
    cache_profile = profile["cache_extraction"]
    if (
        payload.get("schema_version") != CACHE_SCHEMA
        or payload.get("objective") != "transcode"
        or payload.get("selected_layers") != layers
        or payload.get("storage_dtype") != cache_profile["storage_dtype"]
        or payload.get("layout") != cache_profile["layout"]
        or payload.get("mask_contract")
        != {
            "attention_mask_required": True,
            "invalid_rows": "excluded_before_serialization",
            "all_valid_rows": "serialized_without_special_case",
        }
        or not _is_digest(payload.get("content_sha256"))
    ):
        raise GateError(f"{model} cache is smoke, incomplete or mask-contaminated")
    provenance = payload.get("activation_provenance", {})
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("status") != "complete"
        or provenance.get("model_name") != model
        or provenance.get("executed_profile_sha256") != profile_sha256
        or provenance.get("valid_token_budget_by_split")
        != cache_profile["valid_token_budget_by_split"]
        or provenance.get("selected_layers") != layers
        or provenance.get("feature_input_hook")
        != profile["estimand"]["feature_input_hook"]
        or provenance.get("feature_output_hook")
        != profile["estimand"]["feature_output_hook"]
        or provenance.get("decoder_window") != profile["estimand"]["decoder_window"]
        or "execution_mode" in provenance
        or "production_scientific_eligibility" in provenance
        or "production_cache_reuse_forbidden" in provenance
    ):
        raise GateError(
            f"{model} cache provenance is preflight, synthetic or incomplete"
        )
    tokenizer = provenance.get("runtime_tokenizer", {})
    if (
        not isinstance(tokenizer, Mapping)
        or tokenizer.get("padding_side") != "right"
        or tokenizer.get("truncation_side") != "right"
        or provenance.get("first_pass_tokenization_sha256_by_split")
        != provenance.get("second_pass_tokenization_sha256_by_split")
    ):
        raise GateError(f"{model} cache tokenization/padding fingerprints disagree")
    selection = payload.get("token_selection", {})
    if (
        not isinstance(selection, Mapping)
        or selection.get("status") != "complete_budgeted_selection"
        or selection.get("method") != cache_profile["selection_method"]
        or selection.get("position_definition")
        != cache_profile["token_position_definition"]
        or selection.get("serialization_order") != "selection_priority_order"
        or set(selection.get("by_split", {})) != set(SPLITS)
    ):
        raise GateError(f"{model} cache token selection is not production-complete")
    source_hashes: dict[str, str] = {}
    for split, budget in cache_profile["valid_token_budget_by_split"].items():
        selected = selection["by_split"][split]
        summary = payload.get("split_summaries", {}).get(split, {})
        if (
            selected.get("budget") != budget
            or not _is_digest(selected.get("selection_sha256"))
            or selected.get("selection_sha256") != selected.get("row_order_sha256")
            or summary.get("selected_valid_token_rows") != budget
            or summary.get("valid_token_rows") != budget
            or any(
                summary.get("rows_per_layer", {}).get(str(layer)) != budget
                for layer in layers
            )
        ):
            raise GateError(
                f"{model} {split} cache rows violate padding/budget contract"
            )
        selection_path = _cache_member(
            path.parent, selected.get("selection_path"), f"{model} {split} selection"
        )
        if (
            not selection_path.is_file()
            or sha256_file(selection_path) != selected["selection_sha256"]
        ):
            raise GateError(f"{model} {split} selection hash mismatch")
        source = payload.get("source_splits", {}).get(split, {})
        if not isinstance(source, Mapping) or not _is_digest(
            source.get("manifest_sha256")
        ):
            raise GateError(f"{model} {split} source manifest metadata is invalid")
        source_path = _cache_member(
            path.parent, source.get("manifest_path"), f"{model} {split} source"
        )
        if (
            not source_path.is_file()
            or sha256_file(source_path) != source["manifest_sha256"]
        ):
            raise GateError(f"{model} {split} cohort hash mismatch")
        source_hashes[split] = source["manifest_sha256"]
    if len(set(source_hashes.values())) != len(SPLITS):
        raise GateError(f"{model} train/validation/test cohort hashes are not distinct")
    shards = payload.get("shards")
    if not isinstance(shards, Sequence) or isinstance(shards, (str, bytes)):
        raise GateError(f"{model} cache shards are missing")
    shard_keys: set[tuple[str, int]] = set()
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise GateError(f"{model} cache shard is invalid")
        key = (shard.get("split"), shard.get("layer"))
        if (
            key[0] not in SPLITS
            or key[1] not in layers
            or key in shard_keys
            or shard.get("rows") != cache_profile["valid_token_budget_by_split"][key[0]]
            or shard.get("dtype") != cache_profile["storage_dtype"]
            or not _is_digest(shard.get("input_sha256"))
            or not _is_digest(shard.get("target_sha256"))
        ):
            raise GateError(
                f"{model} cache shard metadata violates the frozen contract"
            )
        shard_keys.add(key)
    if shard_keys != {(split, layer) for split in SPLITS for layer in layers}:
        raise GateError(
            f"{model} cache does not have exactly one shard per split/layer"
        )
    return {
        "model_name": model,
        "manifest_path": path,
        "manifest_sha256": manifest_sha,
        "content_sha256": payload["content_sha256"],
        "source_manifest_sha256_by_split": source_hashes,
        "selected_layers": layers,
    }


def _validate_entry_identities(
    entries: object,
    *,
    expected: set[tuple[str, str, int]],
    label: str,
) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    if not isinstance(entries, list):
        raise GateError(f"{label} must be a list")
    indexed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    fields = {
        "model_name",
        "method",
        "run_seed",
        "run_manifest",
        "result",
        "checkpoint",
    }
    for index, item in enumerate(entries):
        record = _exact_keys(item, fields, f"{label}[{index}]")
        key = (record["model_name"], record["method"], record["run_seed"])
        if key in indexed:
            raise GateError(f"duplicate {label} identity: {key}")
        indexed[key] = record
    missing = expected - set(indexed)
    extra = set(indexed) - expected
    if missing or extra:
        raise GateError(
            f"{label} panel mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return indexed


def _validate_common_run(
    entry: Mapping[str, Any],
    *,
    stage: str,
    status: str | tuple[str, ...],
    cache: Mapping[str, Any],
    profile: Mapping[str, Any],
    profile_sha256: str,
    spec_base: Path,
    tested_module_sha256: str,
) -> dict[str, Any]:
    model, method, seed = entry["model_name"], entry["method"], entry["run_seed"]
    manifest_path, manifest_sha, manifest_raw = _verify_descriptor(
        entry["run_manifest"],
        base=spec_base,
        label=f"{stage} {model}/{method}/{seed} run manifest",
        json_file=True,
    )
    result_path, result_sha, result_raw = _verify_descriptor(
        entry["result"],
        base=spec_base,
        label=f"{stage} {model}/{method}/{seed} result",
        json_file=True,
    )
    assert manifest_raw is not None and result_raw is not None
    manifest, result = manifest_raw, result_raw
    accepted_statuses = (status,) if isinstance(status, str) else status
    observed_status = manifest.get("status")
    if observed_status == SPARSITY_MATCH_FAILURE:
        if (
            stage != "screening"
            or entry["checkpoint"] is not None
            or manifest.get("checkpoint_sha256") is not None
            or "selected_checkpoint" in result
        ):
            raise GateError("sparsity-match failure cannot carry a selected checkpoint")
        checkpoint_sha = None
    else:
        _, checkpoint_sha, _ = _verify_descriptor(
            entry["checkpoint"],
            base=spec_base,
            label=f"{stage} {model}/{method}/{seed} checkpoint",
            json_file=False,
        )
    command = manifest.get("command")
    if (
        manifest.get("schema_version") != RUN_MANIFEST_SCHEMA
        or observed_status not in accepted_statuses
        or manifest.get("stage") != stage
        or manifest.get("model_name") != model
        or manifest.get("method") != method
        or manifest.get("run_seed") != seed
        or manifest.get("executed_profile_sha256") != profile_sha256
        or manifest.get("result_sha256") != result_sha
        or manifest.get("checkpoint_sha256") != checkpoint_sha
        or manifest.get("module_sha256") != tested_module_sha256
        or manifest.get("cache_manifest_sha256") != cache["manifest_sha256"]
        or manifest.get("cache_content_sha256") != cache["content_sha256"]
        or manifest.get("source_manifest_sha256_by_split")
        != cache["source_manifest_sha256_by_split"]
        or manifest.get("selected_layers") != cache["selected_layers"]
        or manifest.get("valid_token_budget_by_split")
        != profile["cache_extraction"]["valid_token_budget_by_split"]
        or manifest.get("cache_storage_dtype")
        != profile["cache_extraction"]["storage_dtype"]
        or manifest.get("cache_layout") != profile["cache_extraction"]["layout"]
        or manifest.get("decoder_window") != profile["estimand"]["decoder_window"]
        or manifest.get("feature_input_hook")
        != profile["estimand"]["feature_input_hook"]
        or manifest.get("feature_output_hook")
        != profile["estimand"]["feature_output_hook"]
        or not isinstance(command, list)
        or "--preflight-only" in command
        or str(manifest.get("gpu_model", "")).lower().startswith("cpu")
    ):
        raise GateError(f"{stage} {model}/{method}/{seed} manifest is incompatible")
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("status") != observed_status
        or result.get("stage") != stage
        or result.get("model_name") != model
        or result.get("method") != method
        or result.get("run_seed") != seed
        or result.get("profile_sha256") != profile_sha256
        or result.get("cache_content_sha256") != cache["content_sha256"]
        or (
            observed_status != SPARSITY_MATCH_FAILURE
            and result.get("selected_checkpoint", {}).get("sha256") != checkpoint_sha
        )
    ):
        raise GateError(f"{stage} {model}/{method}/{seed} result identity changed")
    summaries = result.get("candidate_validation")
    if not isinstance(summaries, list) or not summaries:
        raise GateError(f"{stage} {model}/{method}/{seed} lacks training summaries")
    for summary in summaries:
        training = summary.get("training", {}) if isinstance(summary, Mapping) else {}
        if (
            not isinstance(summary, Mapping)
            or summary.get("stage") != stage
            or summary.get("model_seed") != manifest.get("model_seed")
            or summary.get("batch_stream_seed") != manifest.get("batch_stream_seed")
            or training.get("test_split_accesses_during_training") != 0
        ):
            raise GateError(
                f"{stage} {model}/{method}/{seed} training accessed test data"
            )
    return {
        "manifest_path": manifest_path,
        "result_path": result_path,
        "manifest": manifest,
        "result": result,
        "run_manifest_sha256": manifest_sha,
        "result_sha256": result_sha,
        "checkpoint_sha256": checkpoint_sha,
        "terminal_status": observed_status,
    }


def _selected_screening_row(
    result: Mapping[str, Any], profile: Mapping[str, Any]
) -> dict:
    eligible = _sparsity_matched_screening_rows(result, profile)
    if not eligible:
        raise GateError("completed screening result has no sparsity-matched candidate")
    selected = min(
        eligible,
        key=lambda row: (
            row["validation_fvu_mean"],
            abs(row["validation_l0_mean"] - 128.0),
            row["l1_coefficient"],
            row["activation_threshold"],
            row["auxiliary_coefficient"],
        ),
    )
    if result.get("selected_validation_configuration") != selected:
        raise GateError("screening selection does not implement the frozen L0/FVU rule")
    return selected


def _sparsity_matched_screening_rows(
    result: Mapping[str, Any], profile: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = result.get("selection_rows")
    if not isinstance(rows, list) or not rows:
        raise GateError("screening result has no validation selection rows")
    low, high = profile["validation_only_selection"]["eligible_l0_interval"]
    eligible: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise GateError("screening selection row is invalid")
        row = dict(raw)
        l0 = _finite(
            row.get("validation_l0_mean"), "screening validation L0", minimum=0
        )
        _finite(row.get("validation_fvu_mean"), "screening validation FVU", minimum=0)
        if low <= l0 <= high:
            eligible.append(row)
    return eligible


def _validate_screening(
    entry: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    verified = _validate_common_run(
        entry,
        stage="screening",
        status=(SCREENING_STATUS, SPARSITY_MATCH_FAILURE),
        **kwargs,
    )
    result = verified["result"]
    profile = kwargs["profile"]
    if (
        result.get("p0_2_eligible") is not False
        or result.get("test_evaluation_count") != 0
    ):
        raise GateError("screening result is test-leaky or mislabeled eligible")
    expected_candidates = profile["compute_schedule"]["screening"][
        f"{entry['method'].removesuffix('_sae')}_candidates_per_model"
    ]
    if len(result["candidate_validation"]) != expected_candidates:
        raise GateError("screening candidate panel is incomplete")
    if verified["terminal_status"] == SPARSITY_MATCH_FAILURE:
        forbidden = {
            "frozen_screening_configuration",
            "heldout_test",
            "quality_gate_requires_all_seed_aggregation",
            "screening_result_sha256",
            "selected_checkpoint",
            "selected_validation_configuration",
            "training_cache_priority_prefix_rows",
        }
        if (
            _sparsity_matched_screening_rows(result, profile)
            or forbidden.intersection(result)
            or result.get("reason")
            != "no validation candidate achieved L0 within [115.2, 140.8]"
        ):
            raise GateError(
                "sparsity-match failure is not a terminal calibrated negative"
            )
        return {
            **verified,
            "selected_configuration": None,
            "sparsity_match_failed": True,
        }
    if (
        result.get("training_cache_priority_prefix_rows")
        != profile["compute_schedule"]["screening"]["train_cache_priority_prefix_rows"]
    ):
        raise GateError("screening training-cache prefix changed")
    selected = _selected_screening_row(result, profile)
    return {
        **verified,
        "selected_configuration": selected,
        "sparsity_match_failed": False,
    }


def _quantiles(value: object, label: str, *, upper: float | None = None) -> None:
    record = _exact_keys(value, set(QUANTILE_KEYS), label)
    previous = -math.inf
    for key in QUANTILE_KEYS:
        number = _finite(record[key], f"{label}.{key}", minimum=0)
        if number < previous or (upper is not None and number > upper):
            raise GateError(f"{label} is non-monotone or out of range")
        previous = number


def _quality_gate(
    heldout: Mapping[str, Any],
    *,
    layers: list[int],
    test_budget: int,
    activation_threshold: float,
) -> dict[str, Any]:
    if (
        heldout.get("split") != "test"
        or heldout.get("selected_layers") != layers
        or heldout.get("n_valid_tokens") != test_budget
        or heldout.get("objective")
        != "windowed_multi_layer_clt_input_to_mlp_output_transcoding"
        or heldout.get("activation_threshold") != activation_threshold
    ):
        raise GateError(
            "held-out evaluation is not the frozen padding-clean test split"
        )
    targets = heldout.get("target_layers")
    sources = heldout.get("source_layers")
    if not isinstance(targets, list) or not isinstance(sources, list):
        raise GateError("held-out layer reports are missing")
    target_by_layer = {
        row.get("layer"): row for row in targets if isinstance(row, Mapping)
    }
    source_by_layer = {
        row.get("layer"): row for row in sources if isinstance(row, Mapping)
    }
    if (
        len(targets) != len(layers)
        or len(sources) != len(layers)
        or set(target_by_layer) != set(layers)
        or set(source_by_layer) != set(layers)
    ):
        raise GateError("held-out report does not contain each layer exactly once")
    fvu_values: list[float] = []
    dead_values: list[float] = []
    l0_values: list[float] = []
    downstream_layers: list[int] = []
    for layer in layers:
        target = target_by_layer[layer]
        source = source_by_layer[layer]
        fvu_values.append(_finite(target.get("fvu"), f"layer {layer} FVU", minimum=0))
        _finite(target.get("mse"), f"layer {layer} MSE", minimum=0)
        dead = _finite(
            source.get("dead_fraction"), f"layer {layer} dead fraction", minimum=0
        )
        if dead > 1:
            raise GateError(f"layer {layer} dead fraction exceeds one")
        dead_values.append(dead)
        l0_values.append(_finite(source.get("l0_mean"), f"layer {layer} L0", minimum=0))
        _quantiles(
            source.get("firing_frequency_quantiles"),
            f"layer {layer} firing-frequency quantiles",
            upper=1,
        )
        _quantiles(source.get("decoder_norm_quantiles"), f"layer {layer} decoder norms")
        if dead < DOWNSTREAM_DEAD_LIMIT:
            downstream_layers.append(layer)
    reported_mean = _finite(heldout.get("fvu_mean"), "held-out mean FVU", minimum=0)
    reported_dead = _finite(
        heldout.get("dead_fraction_median"), "held-out median dead fraction", minimum=0
    )
    reported_l0 = _finite(heldout.get("l0_mean"), "held-out mean L0", minimum=0)
    if (
        not math.isclose(
            reported_mean, statistics.fmean(fvu_values), rel_tol=1e-9, abs_tol=1e-12
        )
        or not math.isclose(
            reported_dead, statistics.median(dead_values), rel_tol=1e-9, abs_tol=1e-12
        )
        or not math.isclose(
            reported_l0, statistics.fmean(l0_values), rel_tol=1e-9, abs_tol=1e-12
        )
    ):
        raise GateError("held-out aggregate metrics disagree with layer reports")
    _quantiles(heldout.get("reconstruction_error_quantiles"), "reconstruction errors")
    passing_layers = sum(value < FVU_LIMIT for value in fvu_values)
    mean_pass = reported_mean < FVU_LIMIT
    layer_pass = passing_layers * FVU_LAYER_DENOMINATOR >= (
        FVU_LAYER_NUMERATOR * len(layers)
    )
    median_dead_pass = reported_dead < MEDIAN_DEAD_LIMIT
    reasons: list[str] = []
    if not mean_pass:
        reasons.append("mean_test_fvu_not_below_0.50")
    if not layer_pass:
        reasons.append("fewer_than_75_percent_layers_below_0.50_fvu")
    if not median_dead_pass:
        reasons.append("median_dead_fraction_not_below_0.50")
    return {
        "quality_gate_pass": not reasons,
        "failure_reasons": reasons,
        "mean_test_fvu": reported_mean,
        "layers_below_fvu_0_50": passing_layers,
        "required_layer_count": math.ceil(0.75 * len(layers)),
        "layer_fraction_gate_pass": layer_pass,
        "median_dead_fraction": reported_dead,
        "mean_l0": reported_l0,
        "layers_below_dead_0_70": downstream_layers,
    }


def _validate_full(
    entry: Mapping[str, Any],
    *,
    screening: Mapping[tuple[str, str], Mapping[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    verified = _validate_common_run(
        entry,
        stage="full",
        status=FULL_STATUS,
        **kwargs,
    )
    result = verified["result"]
    profile = kwargs["profile"]
    model, method = entry["model_name"], entry["method"]
    if (
        result.get("p0_2_eligible") is not None
        or result.get("quality_gate_requires_all_seed_aggregation") is not True
        or result.get("test_evaluation_count") != 1
        or len(result["candidate_validation"]) != 1
    ):
        raise GateError(
            f"full {model}/{method}/{entry['run_seed']} is incomplete or preadjudicated"
        )
    if method in SCREENED_METHODS:
        screened = screening[(model, method)]
        if (
            result.get("screening_result_sha256") != screened["result_sha256"]
            or result.get("frozen_screening_configuration")
            != screened["selected_configuration"]
        ):
            raise GateError("full run is not bound to the frozen screening selection")
        frozen = screened["selected_configuration"]
    else:
        frozen = {
            "candidate_index": 0,
            "l1_coefficient": 0.0,
            "auxiliary_coefficient": 0.0,
            "activation_threshold": 0.0,
        }
        if (
            result.get("screening_result_sha256") is not None
            or result.get("frozen_screening_configuration") != frozen
        ):
            raise GateError("unscreened TopK/dense run consumed a screening artifact")
    selected = result.get("selected_validation_configuration", {})
    if any(
        selected.get(field) != frozen[field]
        for field in (
            "candidate_index",
            "l1_coefficient",
            "auxiliary_coefficient",
            "activation_threshold",
        )
    ):
        raise GateError(
            "full validation configuration differs from its frozen selection"
        )
    resources = result.get("resources", {})
    expected_parameters = profile["checkpoint_storage_planning"][
        "raw_trainable_parameters_by_model_method"
    ][model][method]
    for field in (
        "training_flop_proxy_total",
        "inference_flop_proxy_per_token",
        "wall_time_seconds",
        "peak_accelerator_memory_allocated_bytes",
        "peak_accelerator_memory_reserved_bytes",
    ):
        _finite(resources.get(field), f"{model}/{method} resource {field}", minimum=0)
    if resources.get("raw_trainable_parameter_count") != expected_parameters:
        raise GateError("observed parameter count differs from frozen profile")
    quality = _quality_gate(
        result.get("heldout_test", {}),
        layers=kwargs["cache"]["selected_layers"],
        test_budget=profile["cache_extraction"]["valid_token_budget_by_split"]["test"],
        activation_threshold=frozen["activation_threshold"],
    )
    return {**verified, **quality}


def adjudicate_dictionary_panel(
    spec_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify a complete P0-2 panel and return deterministic receipt payloads.

    Missing or invalid artifacts raise :class:`GateError`; scientific quality
    failures produce a complete negative receipt.
    """

    spec_path = Path(spec_path).resolve()
    spec = load_strict_json(spec_path)
    fields = {
        "schema_version",
        "confirmatory",
        "profile",
        "protocol",
        "mask_validation_receipt",
        "caches",
        "screening_runs",
        "full_runs",
    }
    _exact_keys(spec, fields, "dictionary-gate spec")
    if spec["schema_version"] != SPEC_SCHEMA or spec["confirmatory"] is not True:
        raise GateError("dictionary-gate spec is nonconfirmatory or unsupported")
    base = spec_path.parent
    _, profile_sha, profile_raw = _verify_descriptor(
        spec["profile"], base=base, label="frozen production profile", json_file=True
    )
    protocol_path, protocol_sha, _ = _verify_descriptor(
        spec["protocol"], base=base, label="frozen P0-2 protocol", json_file=False
    )
    assert profile_raw is not None
    _validate_profile(profile_raw)
    _validate_protocol(protocol_path.read_text(encoding="utf-8"))
    mask_path, mask_sha, mask_raw = _verify_descriptor(
        spec["mask_validation_receipt"],
        base=base,
        label="mask-validation receipt",
        json_file=True,
    )
    assert mask_raw is not None
    mask_passed, tested_module_sha, mask_evidence = _validate_mask_receipt(
        mask_raw, receipt_path=mask_path
    )

    cache_records = spec["caches"]
    if not isinstance(cache_records, list):
        raise GateError("caches must be a list")
    caches: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(cache_records):
        record = _exact_keys(raw, {"model_name", "manifest"}, f"caches[{index}]")
        model = record["model_name"]
        if model in caches:
            raise GateError(f"duplicate cache model: {model}")
        if model not in MODELS:
            raise GateError(f"cache model is outside frozen panel: {model}")
        caches[model] = _validate_cache(
            model=model,
            descriptor=record["manifest"],
            spec_base=base,
            profile=profile_raw,
            profile_sha256=profile_sha,
        )
    if set(caches) != set(MODELS):
        raise GateError("one production cache per frozen model is required")

    screening_expected = {
        (model, method, SCREENING_SEED)
        for model in MODELS
        for method in SCREENED_METHODS
    }
    screening_entries = _validate_entry_identities(
        spec["screening_runs"], expected=screening_expected, label="screening_runs"
    )
    screening: dict[tuple[str, str], dict[str, Any]] = {}
    for identity in sorted(screening_entries):
        model, method, _ = identity
        screening[(model, method)] = _validate_screening(
            screening_entries[identity],
            cache=caches[model],
            profile=profile_raw,
            profile_sha256=profile_sha,
            spec_base=base,
            tested_module_sha256=tested_module_sha,
        )

    failed_screenings = {
        identity
        for identity, result in screening.items()
        if result["sparsity_match_failed"]
    }
    full_expected = {
        (model, method, seed)
        for model in MODELS
        for method in METHODS
        for seed in SEEDS
        if (model, method) not in failed_screenings
    }
    full_entries = _validate_entry_identities(
        spec["full_runs"], expected=full_expected, label="full_runs"
    )
    runs: dict[tuple[str, str, int], dict[str, Any]] = {}
    for identity in sorted(full_entries):
        model, _, _ = identity
        runs[identity] = _validate_full(
            full_entries[identity],
            screening=screening,
            cache=caches[model],
            profile=profile_raw,
            profile_sha256=profile_sha,
            spec_base=base,
            tested_module_sha256=tested_module_sha,
        )

    model_methods: list[dict[str, Any]] = []
    for model in MODELS:
        cache = caches[model]
        for method in METHODS:
            common = {
                "model_name": model,
                "method": method,
                "required_seeds": list(SEEDS),
                "cache_manifest_sha256": cache["manifest_sha256"],
                "cache_content_sha256": cache["content_sha256"],
                "source_manifest_sha256_by_split": cache[
                    "source_manifest_sha256_by_split"
                ],
                "selected_layers": cache["selected_layers"],
                "geometry": {
                    "n_layers": len(cache["selected_layers"]),
                    "d_model": profile_raw["cache_extraction"]["model_cache_geometry"][
                        model
                    ]["input_dim"],
                    "d_clt": (
                        profile_raw["panel"]["dense_low_rank"]["rank"]
                        if method == "dense_low_rank"
                        else profile_raw["panel"][method]["width"]
                    ),
                    "k": (
                        profile_raw["panel"]["topk_clt"]["k"]
                        if method == "topk_clt"
                        else None
                    ),
                    "window": profile_raw["estimand"]["decoder_window"],
                },
                "downstream_layer_policy": (
                    "use_only_receipt_allowlisted_layers; any further subset must be "
                    "frozen before test-set biological outcomes"
                ),
            }
            if (model, method) in failed_screenings:
                model_methods.append(
                    {
                        **common,
                        "status": SPARSITY_MATCH_FAILURE,
                        "atlas_eligible": False,
                        "failure_reasons": ["validation_sparsity_match_failure"],
                        "eligible_downstream_layers": [],
                        "runs": [],
                    }
                )
                continue
            selected = [runs[(model, method, seed)] for seed in SEEDS]
            eligible_layers = sorted(
                set.intersection(
                    *(set(run["layers_below_dead_0_70"]) for run in selected)
                )
            )
            all_run_quality = all(run["quality_gate_pass"] for run in selected)
            atlas_eligible = mask_passed and all_run_quality and bool(eligible_layers)
            failure_reasons: list[str] = []
            if not mask_passed:
                failure_reasons.append("padding_or_all_valid_test_failure")
            if not all_run_quality:
                failure_reasons.append("one_or_more_seed_quality_gates_failed")
            if not eligible_layers:
                failure_reasons.append(
                    "no_layer_below_0.70_dead_fraction_in_every_seed"
                )
            model_methods.append(
                {
                    **common,
                    "status": "atlas_eligible"
                    if atlas_eligible
                    else "quality_gate_failed",
                    "atlas_eligible": atlas_eligible,
                    "failure_reasons": failure_reasons,
                    "eligible_downstream_layers": eligible_layers,
                    "runs": [
                        {
                            "run_seed": seed,
                            "run_manifest_sha256": run["run_manifest_sha256"],
                            "result_sha256": run["result_sha256"],
                            "checkpoint_sha256": run["checkpoint_sha256"],
                            "quality_gate_pass": run["quality_gate_pass"],
                            "failure_reasons": run["failure_reasons"],
                            "mean_test_fvu": run["mean_test_fvu"],
                            "layers_below_fvu_0_50": run["layers_below_fvu_0_50"],
                            "required_layer_count": run["required_layer_count"],
                            "median_dead_fraction": run["median_dead_fraction"],
                            "mean_l0": run["mean_l0"],
                        }
                        for seed, run in zip(SEEDS, selected)
                    ],
                }
            )
    model_adjudications = [
        {
            "model_name": model,
            "status": (
                "all_methods_atlas_eligible"
                if all(
                    row["atlas_eligible"]
                    for row in model_methods
                    if row["model_name"] == model
                )
                else "one_or_more_methods_quality_gate_failed"
            ),
            "atlas_eligible_methods": [
                row["method"]
                for row in model_methods
                if row["model_name"] == model and row["atlas_eligible"]
            ],
        }
        for model in MODELS
    ]
    panel_eligible = all(row["atlas_eligible"] for row in model_methods)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "complete",
        "artifact_completeness": True,
        "p0_2_panel_eligible": panel_eligible,
        "panel_status": (
            "all_model_method_quality_gates_passed"
            if panel_eligible
            else "one_or_more_model_method_quality_gates_failed"
        ),
        "claim_boundary": (
            "Eligibility authorizes only the named model/method and allowlisted layers; "
            "it does not establish biological conservation or causality."
        ),
        "spec_sha256": sha256_file(spec_path),
        "profile_sha256": profile_sha,
        "protocol_sha256": protocol_sha,
        "mask_validation_receipt_sha256": mask_sha,
        "mask_gate_pass": mask_passed,
        "mask_evidence": mask_evidence,
        "frozen_gate": {
            "mean_test_fvu_strictly_below": FVU_LIMIT,
            "minimum_fraction_layers_fvu_strictly_below_0_50": 0.75,
            "median_dead_fraction_strictly_below": MEDIAN_DEAD_LIMIT,
            "downstream_layer_dead_fraction_strictly_below_every_seed": (
                DOWNSTREAM_DEAD_LIMIT
            ),
            "required_seeds": list(SEEDS),
            "test_access_count_after_validation_selection": 1,
            "relu_and_gated_validation_l0_interval_inclusive": [115.2, 140.8],
        },
        "required_models": list(MODELS),
        "required_methods": list(METHODS),
        "required_full_run_count": len(full_expected),
        "required_screening_run_count": len(screening_expected),
        "model_method_adjudications": model_methods,
        "model_adjudications": model_adjudications,
    }
    manifest = {
        "schema_version": RECEIPT_MANIFEST_SCHEMA,
        "status": "complete",
        "spec_sha256": receipt["spec_sha256"],
        "profile_sha256": profile_sha,
        "protocol_sha256": protocol_sha,
        "mask_validation_receipt_sha256": mask_sha,
        "cache_manifest_sha256_by_model": {
            model: caches[model]["manifest_sha256"] for model in MODELS
        },
        "screening_artifact_hashes": [
            {
                "model_name": model,
                "method": method,
                "run_manifest_sha256": screening[(model, method)][
                    "run_manifest_sha256"
                ],
                "result_sha256": screening[(model, method)]["result_sha256"],
                "checkpoint_sha256": screening[(model, method)]["checkpoint_sha256"],
            }
            for model in MODELS
            for method in SCREENED_METHODS
        ],
        "full_artifact_hashes": [
            {
                "model_name": model,
                "method": method,
                "run_seed": seed,
                "run_manifest_sha256": runs[(model, method, seed)][
                    "run_manifest_sha256"
                ],
                "result_sha256": runs[(model, method, seed)]["result_sha256"],
                "checkpoint_sha256": runs[(model, method, seed)]["checkpoint_sha256"],
            }
            for model in MODELS
            for method in METHODS
            for seed in SEEDS
            if (model, method, seed) in full_expected
        ],
    }
    return receipt, manifest


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def require_eligible_model_method(
    receipt_path: Path,
    expected_sha256: str,
    *,
    model_name: str,
    method: str,
    expected_run_manifest_sha256_by_seed: Mapping[int, str] | None = None,
    expected_checkpoint_sha256_by_seed: Mapping[int, str] | None = None,
    expected_source_manifest_sha256_by_split: Mapping[str, str] | None = None,
    requested_layers: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Hash-verify a receipt and return one downstream-authorized panel entry.

    The checkpoint digest in this contract is the exact selected production
    ``best.pt`` file SHA-256 recorded by ``scripts/58_run_dictionary_controls.py``;
    it is not an online-CLT ``clt.pt`` digest or a manifest digest.
    """

    path = Path(receipt_path).resolve()
    if not _is_digest(expected_sha256):
        raise GateError("eligibility receipt SHA-256 is invalid")
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise GateError("eligibility receipt is missing or its SHA-256 changed")
    receipt = load_strict_json(path)
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("artifact_completeness") is not True
        or tuple(receipt.get("required_models", ())) != MODELS
        or tuple(receipt.get("required_methods", ())) != METHODS
        or receipt.get("frozen_gate", {}).get("required_seeds") != list(SEEDS)
    ):
        raise GateError(
            "eligibility receipt is incomplete or has an unsupported schema"
        )
    matches = [
        row
        for row in receipt.get("model_method_adjudications", ())
        if isinstance(row, Mapping)
        and row.get("model_name") == model_name
        and row.get("method") == method
    ]
    if len(matches) != 1:
        raise GateError("eligibility receipt lacks one exact model/method adjudication")
    selected = dict(matches[0])
    if (
        selected.get("status") != "atlas_eligible"
        or selected.get("atlas_eligible") is not True
        or selected.get("failure_reasons") != []
        or selected.get("required_seeds") != list(SEEDS)
        or set(selected.get("source_manifest_sha256_by_split", {})) != set(SPLITS)
        or any(
            not _is_digest(value)
            for value in selected["source_manifest_sha256_by_split"].values()
        )
        or not _is_digest(selected.get("cache_manifest_sha256"))
        or not _is_digest(selected.get("cache_content_sha256"))
    ):
        raise GateError("requested model/method did not pass the all-seed P0-2 gate")
    run_rows = selected.get("runs")
    if not isinstance(run_rows, list) or len(run_rows) != len(SEEDS):
        raise GateError("eligible adjudication has an incomplete seed panel")
    run_by_seed: dict[int, Mapping[str, Any]] = {}
    for row in run_rows:
        if not isinstance(row, Mapping) or row.get("run_seed") in run_by_seed:
            raise GateError("eligible adjudication contains invalid seed rows")
        seed = row.get("run_seed")
        if (
            seed not in SEEDS
            or row.get("quality_gate_pass") is not True
            or row.get("failure_reasons") != []
            or any(
                not _is_digest(row.get(field))
                for field in (
                    "run_manifest_sha256",
                    "result_sha256",
                    "checkpoint_sha256",
                )
            )
        ):
            raise GateError("eligible adjudication contains a failed or malformed seed")
        run_by_seed[seed] = row
    if set(run_by_seed) != set(SEEDS):
        raise GateError("eligible adjudication does not contain seeds 17, 29 and 43")

    def _normalize_seed_hashes(
        values: Mapping[int, str] | None, label: str
    ) -> dict[int, str] | None:
        if values is None:
            return None
        normalized = {int(seed): digest for seed, digest in values.items()}
        if set(normalized) != set(SEEDS) or any(
            not _is_digest(digest) for digest in normalized.values()
        ):
            raise GateError(f"{label} must bind exactly seeds 17, 29 and 43")
        return normalized

    expected_manifests = _normalize_seed_hashes(
        expected_run_manifest_sha256_by_seed, "expected run-manifest hashes"
    )
    expected_checkpoints = _normalize_seed_hashes(
        expected_checkpoint_sha256_by_seed, "expected checkpoint hashes"
    )
    if expected_manifests is not None and any(
        run_by_seed[seed]["run_manifest_sha256"] != expected_manifests[seed]
        for seed in SEEDS
    ):
        raise GateError(
            "consumer run-manifest hashes differ from the eligibility receipt"
        )
    if expected_checkpoints is not None and any(
        run_by_seed[seed]["checkpoint_sha256"] != expected_checkpoints[seed]
        for seed in SEEDS
    ):
        raise GateError(
            "consumer checkpoint hashes differ from the eligibility receipt"
        )
    if (
        expected_source_manifest_sha256_by_split is not None
        and dict(expected_source_manifest_sha256_by_split)
        != selected["source_manifest_sha256_by_split"]
    ):
        raise GateError("consumer cohort hashes differ from the eligibility receipt")
    allowed_layers = selected.get("eligible_downstream_layers")
    if (
        not isinstance(allowed_layers, list)
        or not allowed_layers
        or len(set(allowed_layers)) != len(allowed_layers)
        or any(type(layer) is not int for layer in allowed_layers)
    ):
        raise GateError("eligible adjudication has no valid downstream layer allowlist")
    if requested_layers is not None:
        requested = list(requested_layers)
        if (
            not requested
            or len(set(requested)) != len(requested)
            or any(type(layer) is not int for layer in requested)
            or not set(requested) <= set(allowed_layers)
        ):
            raise GateError(
                "requested downstream layers are outside the receipt allowlist"
            )
    selected["eligibility_receipt_path"] = str(path)
    selected["eligibility_receipt_sha256"] = expected_sha256
    selected["profile_sha256"] = receipt["profile_sha256"]
    selected["protocol_sha256"] = receipt["protocol_sha256"]
    selected["mask_validation_receipt_sha256"] = receipt[
        "mask_validation_receipt_sha256"
    ]
    return selected


def _compatible_clt_from_topk_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_candidate_id: str,
    geometry: Mapping[str, Any],
) -> Any:
    """Map exact-cache TopK tensors into a no-copy CLTForTraining instance."""

    import torch
    import torch.nn as nn

    from src.training.clt_trainer import CLTForTraining

    n_layers = geometry.get("n_layers")
    d_model = geometry.get("d_model")
    d_clt = geometry.get("d_clt")
    k = geometry.get("k")
    window = geometry.get("window")
    if (
        any(
            type(value) is not int or value < 1
            for value in (n_layers, d_model, d_clt, k, window)
        )
        or k > d_clt
    ):
        raise GateError("eligible TopK geometry is invalid")
    if (
        checkpoint.get("schema_version") != "r2_dictionary_control_best_v1"
        or checkpoint.get("candidate_id") != expected_candidate_id
        or type(checkpoint.get("step")) is not int
        or checkpoint["step"] < 1
        or checkpoint["step"] > 200_000
        or checkpoint["step"] % 5_000 != 0
    ):
        raise GateError("selected best.pt identity or training step is invalid")
    _finite(
        checkpoint.get("validation_fvu_mean"),
        "selected checkpoint validation FVU",
        minimum=0,
    )
    state = checkpoint.get("model_state_dict")
    expected_keys = {
        "encoder_weight",
        "encoder_bias",
        "decoder_bias",
        *(f"decoder_weight.{layer}" for layer in range(n_layers)),
    }
    if not isinstance(state, Mapping) or set(state) != expected_keys:
        raise GateError("selected best.pt is not a complete TopK WindowedTranscoder")
    shapes = {
        "encoder_weight": (n_layers, d_clt, d_model),
        "encoder_bias": (n_layers, d_clt),
        "decoder_bias": (n_layers, d_model),
        **{
            f"decoder_weight.{layer}": (
                d_clt,
                min(window, n_layers - layer),
                d_model,
            )
            for layer in range(n_layers)
        },
    }
    for name, shape in shapes.items():
        tensor = state[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != shape
            or not tensor.is_floating_point()
            or not torch.isfinite(tensor).all()
        ):
            raise GateError(f"selected best.pt tensor is invalid: {name}")
    dtypes = {tensor.dtype for tensor in state.values()}
    devices = {tensor.device for tensor in state.values()}
    if len(dtypes) != 1 or len(devices) != 1:
        raise GateError("selected best.pt tensors do not share one dtype/device")

    # Avoid initializing and then copying a multi-billion-parameter model. The
    # object uses CLTForTraining's tested encode/decode methods, while its
    # parameters directly own the already hash-verified checkpoint tensors.
    clt = CLTForTraining.__new__(CLTForTraining)
    nn.Module.__init__(clt)
    clt.n_layers = n_layers
    clt.d_model = d_model
    clt.d_clt = d_clt
    clt.k = k
    clt.window = min(window, n_layers)
    clt.W_enc = nn.Parameter(state["encoder_weight"], requires_grad=False)
    clt.b_enc = nn.Parameter(state["encoder_bias"], requires_grad=False)
    clt.W_dec = nn.ParameterList(
        [
            nn.Parameter(state[f"decoder_weight.{layer}"], requires_grad=False)
            for layer in range(n_layers)
        ]
    )
    clt.b_dec = nn.Parameter(state["decoder_bias"], requires_grad=False)
    clt.register_buffer(
        "feature_last_fired",
        torch.zeros(
            n_layers, d_clt, dtype=torch.long, device=state["encoder_weight"].device
        ),
    )
    clt.register_buffer(
        "global_step",
        torch.tensor(
            checkpoint["step"],
            dtype=torch.long,
            device=state["encoder_weight"].device,
        ),
    )
    clt.dead_feature_threshold = 5_000
    clt.eval()
    return clt


def load_eligible_topk_clt(
    receipt_path: Path,
    expected_receipt_sha256: str,
    *,
    model_name: str,
    run_seed: int,
    checkpoint_path: Path,
    expected_run_manifest_sha256_by_seed: Mapping[int, str],
    expected_checkpoint_sha256_by_seed: Mapping[int, str],
    expected_source_manifest_sha256_by_split: Mapping[str, str],
    requested_layers: Sequence[int] | None = None,
    map_location: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """Load one all-seed-authorized exact-cache TopK as CLTForTraining.

    No tensor transformation is performed: field names are mapped directly
    because the two TopK architectures use identical encoder, windowed decoder,
    bias and ReLU-then-TopK equations.
    """

    if run_seed not in SEEDS:
        raise GateError("TopK run_seed must be one of 17, 29 or 43")
    eligible = require_eligible_model_method(
        receipt_path,
        expected_receipt_sha256,
        model_name=model_name,
        method="topk_clt",
        expected_run_manifest_sha256_by_seed=(expected_run_manifest_sha256_by_seed),
        expected_checkpoint_sha256_by_seed=expected_checkpoint_sha256_by_seed,
        expected_source_manifest_sha256_by_split=(
            expected_source_manifest_sha256_by_split
        ),
        requested_layers=requested_layers,
    )
    checkpoint = Path(checkpoint_path).resolve()
    expected_checkpoint_sha = {
        int(seed): digest for seed, digest in expected_checkpoint_sha256_by_seed.items()
    }[run_seed]
    if not checkpoint.is_file() or sha256_file(checkpoint) != expected_checkpoint_sha:
        raise GateError("selected TopK best.pt is missing or its SHA-256 changed")
    geometry = eligible.get("geometry", {})
    if (
        geometry.get("n_layers") != len(eligible["selected_layers"])
        or geometry.get("d_clt") != 8192
        or geometry.get("k") != 128
        or geometry.get("window") != 8
        or geometry.get("d_model")
        != {"protgpt2": 1280, "zymctrl": 1280, "progen2-medium": 1536}[model_name]
    ):
        raise GateError("receipt TopK geometry differs from the frozen profile")
    import torch

    try:
        payload = torch.load(checkpoint, map_location=map_location, weights_only=True)
    except Exception as error:
        raise GateError(f"cannot load selected TopK best.pt: {error}") from error
    expected_candidate_id = (
        f"full_topk_clt_seed_{run_seed}_candidate_000_"
        f"profile_{eligible['profile_sha256'][:12]}_"
        f"cache_{eligible['cache_content_sha256'][:12]}"
    )
    clt = _compatible_clt_from_topk_checkpoint(
        payload,
        expected_candidate_id=expected_candidate_id,
        geometry=geometry,
    )
    run = next(row for row in eligible["runs"] if row["run_seed"] == run_seed)
    provenance = {
        "schema_version": "r2_p0_2_eligible_topk_load_v1",
        "model_name": model_name,
        "method": "topk_clt",
        "run_seed": run_seed,
        "eligibility_receipt_sha256": expected_receipt_sha256,
        "profile_sha256": eligible["profile_sha256"],
        "protocol_sha256": eligible["protocol_sha256"],
        "run_manifest_sha256": run["run_manifest_sha256"],
        "checkpoint_sha256": run["checkpoint_sha256"],
        "candidate_id": expected_candidate_id,
        "checkpoint_step": payload["step"],
        "geometry": dict(geometry),
        "eligible_downstream_layers": eligible["eligible_downstream_layers"],
        "source_manifest_sha256_by_split": eligible["source_manifest_sha256_by_split"],
    }
    clt.p0_2_eligibility = provenance
    return clt, provenance


def write_eligibility_receipt(spec_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Adjudicate first, then atomically publish a write-once receipt directory."""

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite eligibility receipt: {output}")
    receipt, manifest = adjudicate_dictionary_panel(spec_path)
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{output.name}.tmp-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"stale receipt staging directory: {staging}")
    staging.mkdir()
    try:
        receipt_path = staging / "p0_2_eligibility_receipt.json"
        receipt_path.write_bytes(_json_bytes(receipt))
        manifest["eligibility_receipt_sha256"] = sha256_file(receipt_path)
        manifest_path = staging / "p0_2_eligibility_receipt.manifest.json"
        manifest_path.write_bytes(_json_bytes(manifest))
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return (
        output / "p0_2_eligibility_receipt.json",
        output / "p0_2_eligibility_receipt.manifest.json",
    )
