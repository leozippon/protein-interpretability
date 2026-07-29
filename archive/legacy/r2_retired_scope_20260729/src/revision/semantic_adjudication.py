"""Fail-closed joint adjudication for conditional-semantics runs.

Script 53 adjusts only the hypotheses present in one invocation.  This module
verifies a frozen, exact collection of those invocations and recomputes one BH
family across every model, layer, representation, feature, label and blocking.
It deliberately treats the per-run q-values as untrusted input metadata.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from .io import sha256_file, write_json
from .statistics import benjamini_hochberg


SHA256 = set("0123456789abcdef")
BLOCKINGS = ("protein", "family")
RUN_FILES = {
    "conditional_effects.tsv",
    "fold_assignments.tsv",
    "label_prevalence.tsv",
    "summary.json",
    "run_manifest.json",
}
HYPOTHESIS_FIELDS = ("representation", "feature", "label", "blocking")
TARGET_FIELDS = ("model", "layer", *HYPOTHESIS_FIELDS)
EFFECT_FIELDS = (
    "representation",
    "feature",
    "label",
    "blocking",
    "n_observations",
    "n_proteins",
    "n_blocks",
    "baseline_mse",
    "full_mse",
    "delta_mse",
    "delta_r2",
    "permutation_pvalue",
    "qvalue",
    "bootstrap_delta_mse_ci95",
    "bootstrap_delta_r2_ci95",
    "bootstrap_standard_error_delta_mse",
    "retrospective_bootstrap_detectable_delta_mse",
    "prospective_minimum_detectable_delta_mse",
    "permutable_row_fraction",
    "permutation_degenerate",
    "fold_hash",
    "representation_role",
    "label_role",
    "label_family",
)
FINITE_EFFECT_FIELDS = (
    "baseline_mse",
    "full_mse",
    "delta_mse",
    "delta_r2",
    "bootstrap_standard_error_delta_mse",
    "retrospective_bootstrap_detectable_delta_mse",
    "permutable_row_fraction",
)


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - SHA256:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _resolve(value: Any, base: Path, label: str) -> Path:
    path = Path(_nonempty(value, label)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if str(result) != str(value).strip() or result < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return result


def _bool_text(value: str, label: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"{label} must be exactly 'True' or 'False'")
    return value == "True"


def _ci(value: str, label: str) -> tuple[float, float]:
    try:
        parsed = json.loads(value, parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite constant {token}")
        ))
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} must be a strict JSON two-vector") from error
    if not isinstance(parsed, list) or len(parsed) != 2:
        raise ValueError(f"{label} must be a strict JSON two-vector")
    lower, upper = (_finite(item, label) for item in parsed)
    if lower > upper:
        raise ValueError(f"{label} lower endpoint exceeds upper endpoint")
    return lower, upper


def _key(record: dict[str, Any], fields: tuple[str, ...], label: str) -> tuple[Any, ...]:
    result: list[Any] = []
    for field in fields:
        value = record.get(field)
        if field == "layer":
            value = _integer(value, f"{label}.layer")
        else:
            value = _nonempty(value, f"{label}.{field}")
        result.append(value)
    if "blocking" in fields and result[fields.index("blocking")] not in BLOCKINGS:
        raise ValueError(f"{label}.blocking must be one of {BLOCKINGS}")
    return tuple(result)


def _expected_hypotheses(run_spec: dict[str, Any]) -> tuple[set[tuple[str, ...]], dict, dict]:
    representations = run_spec.get("representations")
    labels = run_spec.get("labels")
    if not isinstance(representations, list) or not representations:
        raise ValueError("script-53 spec requires non-empty representations")
    if not isinstance(labels, list) or not labels:
        raise ValueError("script-53 spec requires non-empty labels")
    representation_metadata: dict[str, dict] = {}
    for index, record in enumerate(representations):
        if not isinstance(record, dict):
            raise ValueError(f"representation {index} must be an object")
        name = _nonempty(record.get("name"), f"representation {index}.name")
        features = record.get("feature_names")
        if name in representation_metadata or not isinstance(features, list) or not features:
            raise ValueError("representation names and feature lists must be non-empty and unique")
        normalized = [_nonempty(item, f"{name}.feature") for item in features]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"representation {name!r} has duplicate features")
        representation_metadata[name] = {
            "role": _nonempty(record.get("role"), f"{name}.role"),
            "features": normalized,
        }
    label_metadata: dict[str, dict] = {}
    for index, record in enumerate(labels):
        if not isinstance(record, dict):
            raise ValueError(f"label {index} must be an object")
        name = _nonempty(record.get("name"), f"label {index}.name")
        if name in label_metadata:
            raise ValueError(f"duplicate label {name!r}")
        label_metadata[name] = {
            "role": _nonempty(record.get("role"), f"{name}.role"),
            "family": _nonempty(record.get("family"), f"{name}.family"),
            "array": _nonempty(record.get("array"), f"{name}.array"),
        }
    hypotheses = {
        (representation, feature, label, blocking)
        for representation, metadata in representation_metadata.items()
        for feature in metadata["features"]
        for label in label_metadata
        for blocking in BLOCKINGS
    }
    return hypotheses, representation_metadata, label_metadata


def _manifest_outputs(run_dir: Path, manifest: dict[str, Any]) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("run manifest outputs must be a list")
    expected_names = RUN_FILES - {"run_manifest.json"}
    observed: set[str] = set()
    for index, descriptor in enumerate(outputs):
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256", "bytes"}:
            raise ValueError(f"run manifest output {index} has an invalid schema")
        path = Path(_nonempty(descriptor["path"], f"output {index}.path")).expanduser().resolve()
        if path.parent != run_dir or path.name in observed:
            raise ValueError("run manifest outputs must be unique direct children of the run directory")
        observed.add(path.name)
        if not path.is_file():
            raise FileNotFoundError(f"missing manifested output: {path}")
        if sha256_file(path) != _digest(descriptor["sha256"], f"output {path.name}.sha256"):
            raise ValueError(f"manifested output hash mismatch: {path.name}")
        if path.stat().st_size != _integer(descriptor["bytes"], f"output {path.name}.bytes"):
            raise ValueError(f"manifested output size mismatch: {path.name}")
    if observed != expected_names:
        raise ValueError(
            f"run manifest output inventory mismatch; missing={sorted(expected_names - observed)}, "
            f"extra={sorted(observed - expected_names)}"
        )
    children = list(run_dir.iterdir())
    if any(not path.is_file() for path in children):
        raise ValueError("run directory contains a non-file entry outside the frozen inventory")
    actual = {path.name for path in children}
    if actual != RUN_FILES:
        raise ValueError(
            f"run directory file inventory mismatch; missing={sorted(RUN_FILES - actual)}, "
            f"extra={sorted(actual - RUN_FILES)}"
        )


def _load_effects(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("conditional-effects header is missing or duplicated")
        if tuple(reader.fieldnames) != EFFECT_FIELDS:
            raise ValueError("conditional-effects schema does not match script-53 schema")
        rows = list(reader)
    if not rows:
        raise ValueError("conditional-effects table is empty")
    if any(set(row) != set(EFFECT_FIELDS) or any(value is None for value in row.values()) for row in rows):
        raise ValueError("conditional-effects row does not match the exact table schema")
    return rows


def _permutable_fractions(
    data_path: Path,
    run_spec: dict[str, Any],
    labels: dict[str, dict],
) -> tuple[int, int, dict[str, float]]:
    arrays = run_spec.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError("script-53 spec arrays must be an object")
    protein_array = _nonempty(arrays.get("protein_id"), "arrays.protein_id")
    with np.load(data_path, allow_pickle=False) as bundle:
        if protein_array not in bundle.files:
            raise ValueError(f"data bundle lacks protein array {protein_array!r}")
        proteins = np.asarray(bundle[protein_array])
        if proteins.ndim != 1 or proteins.size < 2:
            raise ValueError("protein array must be a vector with at least two rows")
        protein_strings = proteins.astype(str)
        unique_proteins = np.unique(protein_strings)
        fractions: dict[str, float] = {}
        for name, metadata in labels.items():
            array_name = metadata["array"]
            if array_name not in bundle.files:
                raise ValueError(f"data bundle lacks label array {array_name!r}")
            values = np.asarray(bundle[array_name])
            if values.ndim != 1 or values.size != proteins.size:
                raise ValueError(f"label array {array_name!r} has an invalid row dimension")
            strings = values.astype(str)
            if np.unique(strings).size != 2:
                raise ValueError(f"confirmatory label {name!r} must be exactly binary")
            permutable = np.zeros(proteins.size, dtype=bool)
            for protein in unique_proteins:
                rows = protein_strings == protein
                if np.unique(strings[rows]).size > 1:
                    permutable[rows] = True
            fractions[name] = float(np.mean(permutable))
    return int(proteins.size), int(unique_proteins.size), fractions


def _power_plan(
    run_spec: dict[str, Any],
    spec_path: Path,
    expected: set[tuple[str, ...]],
) -> tuple[str | None, dict[tuple[str, ...], float]]:
    descriptor = run_spec.get("prospective_power_plan")
    if descriptor is None:
        return None, {}
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
        raise ValueError("prospective_power_plan must contain exactly path and sha256")
    expected_hash = _digest(descriptor["sha256"], "prospective_power_plan.sha256")
    path = _resolve(descriptor["path"], spec_path.parent, "prospective_power_plan.path")
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise ValueError("prospective power-plan file is missing or has a SHA-256 mismatch")
    plan = _strict_json(path)
    if plan.get("schema_version") != 1:
        raise ValueError("prospective power plan requires schema_version 1")
    source = plan.get("independent_source")
    if not isinstance(source, dict) or source.get("independent_of_confirmatory_data") is not True:
        raise ValueError("prospective power plan must be independent of confirmatory data")
    _nonempty(source.get("description"), "independent_source.description")
    _nonempty(source.get("standard_error_method"), "independent_source.standard_error_method")
    _digest(source.get("run_manifest_sha256"), "independent_source.run_manifest_sha256")
    _digest(source.get("cohort_sha256"), "independent_source.cohort_sha256")
    rows = plan.get("standard_errors_delta_mse")
    if not isinstance(rows, list) or not rows:
        raise ValueError("prospective power plan has no standard errors")
    standard_errors: dict[tuple[str, ...], float] = {}
    required = {*HYPOTHESIS_FIELDS, "standard_error_delta_mse"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError(f"prospective power-plan row {index} has an invalid schema")
        key = _key(row, HYPOTHESIS_FIELDS, f"prospective power-plan row {index}")
        if key in standard_errors:
            raise ValueError(f"duplicate prospective power-plan hypothesis {key!r}")
        value = _finite(row["standard_error_delta_mse"], f"power-plan SE {key!r}")
        if value <= 0:
            raise ValueError(f"power-plan SE {key!r} must be positive")
        standard_errors[key] = value
    if set(standard_errors) != expected:
        raise ValueError("prospective power plan must exactly cover the run hypothesis inventory")
    return expected_hash, standard_errors


def _validate_effect_row(
    row: dict[str, str],
    representations: dict[str, dict],
    labels: dict[str, dict],
    n_observations: int,
    n_proteins: int,
    permutable: dict[str, float],
) -> dict[str, Any]:
    key = _key(row, HYPOTHESIS_FIELDS, "conditional effect")
    representation, _, label, _ = key
    if representation not in representations or label not in labels:
        raise ValueError(f"unknown hypothesis identity {key!r}")
    if row["representation_role"] != representations[representation]["role"]:
        raise ValueError(f"representation role mismatch for {key!r}")
    if row["label_role"] != labels[label]["role"] or row["label_family"] != labels[label]["family"]:
        raise ValueError(f"label metadata mismatch for {key!r}")
    if _integer(row["n_observations"], "n_observations", minimum=1) != n_observations:
        raise ValueError(f"observation count mismatch for {key!r}")
    if _integer(row["n_proteins"], "n_proteins", minimum=1) != n_proteins:
        raise ValueError(f"protein count mismatch for {key!r}")
    _integer(row["n_blocks"], "n_blocks", minimum=2)
    numeric = {field: _finite(row[field], f"{field} for {key!r}") for field in FINITE_EFFECT_FIELDS}
    if numeric["baseline_mse"] < 0 or numeric["full_mse"] < 0:
        raise ValueError(f"MSE must be nonnegative for {key!r}")
    if numeric["bootstrap_standard_error_delta_mse"] < 0:
        raise ValueError(f"bootstrap standard error must be nonnegative for {key!r}")
    if numeric["retrospective_bootstrap_detectable_delta_mse"] < 0:
        raise ValueError(f"retrospective detectable delta must be nonnegative for {key!r}")
    pvalue = _finite(row["permutation_pvalue"], f"permutation_pvalue for {key!r}")
    reported_qvalue = _finite(row["qvalue"], f"qvalue for {key!r}")
    if not 0 < pvalue <= 1 or not 0 <= reported_qvalue <= 1:
        raise ValueError(f"p/q value outside [0, 1] for {key!r}")
    if not 0 <= numeric["permutable_row_fraction"] <= 1:
        raise ValueError(f"permutable-row fraction outside [0, 1] for {key!r}")
    if not math.isclose(numeric["permutable_row_fraction"], permutable[label], abs_tol=1e-12):
        raise ValueError(f"permutable-row fraction does not match the source data for {key!r}")
    degenerate = _bool_text(row["permutation_degenerate"], "permutation_degenerate")
    if degenerate != (permutable[label] == 0.0):
        raise ValueError(f"permutation-degeneracy flag mismatch for {key!r}")
    if labels[label]["role"] == "biological" and degenerate:
        raise ValueError(f"degenerate confirmatory biological-label test {key!r}")
    mse_ci = _ci(row["bootstrap_delta_mse_ci95"], "bootstrap_delta_mse_ci95")
    _ci(row["bootstrap_delta_r2_ci95"], "bootstrap_delta_r2_ci95")
    _digest(row["fold_hash"], f"fold_hash for {key!r}")
    prospective = None
    if row["prospective_minimum_detectable_delta_mse"] != "":
        prospective = _finite(
            row["prospective_minimum_detectable_delta_mse"],
            f"prospective_minimum_detectable_delta_mse for {key!r}",
        )
        if prospective <= 0:
            raise ValueError(f"prospective MDE must be positive for {key!r}")
    return {
        "key": key,
        "pvalue": pvalue,
        "reported_qvalue": reported_qvalue,
        "delta_mse": numeric["delta_mse"],
        "mse_ci": mse_ci,
        "prospective_mde": prospective,
        "raw": row,
    }


def _load_run(
    descriptor: dict[str, Any],
    collector_base: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required = {
        "model",
        "layer",
        "path",
        "run_manifest_sha256",
        "run_spec_sha256",
        "data_sha256",
        "prospective_power_plan_sha256",
        "hypotheses",
    }
    if set(descriptor) != required:
        raise ValueError(f"run descriptor must contain exactly {sorted(required)}")
    model = _nonempty(descriptor["model"], "run.model")
    layer = _integer(descriptor["layer"], "run.layer")
    run_dir = _resolve(descriptor["path"], collector_base, "run.path")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"missing frozen run directory: {run_dir}")
    manifest_path = run_dir / "run_manifest.json"
    expected_manifest_hash = _digest(descriptor["run_manifest_sha256"], "run_manifest_sha256")
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_manifest_hash:
        raise ValueError(f"run manifest is missing or does not match for {model}/layer-{layer}")
    manifest = _strict_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("script-53 run manifest requires schema_version 1")
    _manifest_outputs(run_dir, manifest)

    spec_path = Path(_nonempty(manifest.get("run_spec"), "run_manifest.run_spec")).expanduser().resolve()
    expected_spec_hash = _digest(descriptor["run_spec_sha256"], "run_spec_sha256")
    if manifest.get("run_spec_sha256") != expected_spec_hash:
        raise ValueError("run-manifest/spec descriptor hash disagreement")
    if not spec_path.is_file() or sha256_file(spec_path) != expected_spec_hash:
        raise ValueError("script-53 run spec is missing or has a SHA-256 mismatch")
    run_spec = _strict_json(spec_path)
    if run_spec.get("confirmatory") is not True:
        raise ValueError("joint adjudication rejects non-confirmatory script-53 inputs")

    data_manifest = manifest.get("data")
    if not isinstance(data_manifest, dict):
        raise ValueError("run manifest data descriptor is missing")
    data_path = Path(_nonempty(data_manifest.get("path"), "run_manifest.data.path")).expanduser().resolve()
    expected_data_hash = _digest(descriptor["data_sha256"], "data_sha256")
    data_spec = run_spec.get("data")
    if not isinstance(data_spec, dict) or data_spec.get("sha256") != expected_data_hash:
        raise ValueError("run-spec/data descriptor hash disagreement")
    if data_manifest.get("sha256") != expected_data_hash:
        raise ValueError("run-manifest/data descriptor hash disagreement")
    if not data_path.is_file() or sha256_file(data_path) != expected_data_hash:
        raise ValueError("script-53 data bundle is missing or has a SHA-256 mismatch")

    expected, representations, labels = _expected_hypotheses(run_spec)
    frozen_rows = descriptor["hypotheses"]
    if not isinstance(frozen_rows, list) or not frozen_rows:
        raise ValueError("run descriptor requires a non-empty exact hypothesis list")
    frozen: list[tuple[str, ...]] = []
    for index, row in enumerate(frozen_rows):
        if not isinstance(row, dict) or set(row) != set(HYPOTHESIS_FIELDS):
            raise ValueError(f"frozen hypothesis {index} has an invalid schema")
        frozen.append(_key(row, HYPOTHESIS_FIELDS, f"frozen hypothesis {index}"))
    if len(frozen) != len(set(frozen)):
        raise ValueError("frozen run descriptor contains duplicate hypotheses")
    if set(frozen) != expected:
        raise ValueError("frozen hypothesis inventory does not exactly match the script-53 spec")

    summary = _strict_json(run_dir / "summary.json")
    if summary.get("schema_version") != 1 or summary.get("confirmatory") is not True:
        raise ValueError("joint adjudication requires a confirmatory schema-1 summary")
    if summary.get("n_hypotheses") != len(expected):
        raise ValueError("summary hypothesis count does not match the frozen inventory")
    n_observations, n_proteins, permutable = _permutable_fractions(
        data_path, run_spec, labels
    )
    if any(
        metadata["role"] == "biological" and permutable[name] == 0.0
        for name, metadata in labels.items()
    ):
        raise ValueError("source data contain a degenerate confirmatory biological label")

    power_hash, standard_errors = _power_plan(run_spec, spec_path, expected)
    declared_power_hash = descriptor["prospective_power_plan_sha256"]
    if declared_power_hash is not None:
        declared_power_hash = _digest(declared_power_hash, "prospective_power_plan_sha256")
    if declared_power_hash != power_hash:
        raise ValueError("frozen and observed prospective power-plan hashes disagree")
    for container_name, metadata in (
        ("summary", summary.get("prospective_power_plan")),
        ("run manifest", manifest.get("prospective_power_plan")),
    ):
        observed = None if metadata is None else metadata.get("sha256") if isinstance(metadata, dict) else False
        if observed != power_hash:
            raise ValueError(f"{container_name} prospective power-plan provenance mismatch")

    rows = _load_effects(run_dir / "conditional_effects.tsv")
    parsed = [
        _validate_effect_row(
            row, representations, labels, n_observations, n_proteins, permutable
        )
        for row in rows
    ]
    observed = [record["key"] for record in parsed]
    if len(observed) != len(set(observed)):
        raise ValueError("conditional-effects table contains duplicate hypotheses")
    if set(observed) != expected:
        raise ValueError("conditional-effects table has missing or extra hypotheses")
    for blocking in BLOCKINGS:
        fold_hashes = {record["raw"]["fold_hash"] for record in parsed if record["key"][-1] == blocking}
        if len(fold_hashes) != 1:
            raise ValueError(f"inconsistent {blocking} fold hashes within one script-53 run")

    test = run_spec.get("test")
    if not isinstance(test, dict):
        raise ValueError("script-53 spec test settings are missing")
    local_alpha = _finite(test.get("fdr_alpha"), "test.fdr_alpha")
    power = _finite(test.get("power"), "test.power")
    if not 0 < local_alpha < 1 or not 0 < power < 1:
        raise ValueError("script-53 alpha and power must lie in (0, 1)")
    local_multiplier = NormalDist().inv_cdf(1 - local_alpha / len(expected)) + NormalDist().inv_cdf(power)
    for record in parsed:
        key = record["key"]
        standard_error = standard_errors.get(key)
        expected_mde = None if standard_error is None else standard_error * local_multiplier
        if (record["prospective_mde"] is None) != (expected_mde is None):
            raise ValueError(f"prospective MDE/power-plan mismatch for {key!r}")
        if expected_mde is not None and not math.isclose(
            record["prospective_mde"], expected_mde, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(f"prospective MDE does not reproduce from the power plan for {key!r}")
        record.update(
            {
                "model": model,
                "layer": layer,
                "global_key": (model, layer, *key),
                "prospective_standard_error": standard_error,
            }
        )
    provenance = {
        "model": model,
        "layer": layer,
        "path": str(run_dir),
        "run_manifest_sha256": expected_manifest_hash,
        "run_spec_sha256": expected_spec_hash,
        "data_sha256": expected_data_hash,
        "prospective_power_plan_sha256": power_hash,
        "n_hypotheses": len(parsed),
        "fdr_alpha": local_alpha,
        "power": power,
    }
    return parsed, provenance


def _decision(
    adjudication: dict[str, Any],
    records: dict[tuple[Any, ...], dict[str, Any]],
    alpha: float,
) -> dict[str, Any]:
    mode = adjudication.get("mode")
    if mode == "multiplicity_only":
        if set(adjudication) != {"mode"}:
            raise ValueError("multiplicity_only adjudication accepts no target or threshold fields")
        return {
            "mode": mode,
            "status": "not_scientifically_adjudicated",
            "reason": "joint multiplicity was computed without a frozen scientific decision rule",
        }
    if mode not in {"association", "powered_bound"}:
        raise ValueError("adjudication.mode must be multiplicity_only, association, or powered_bound")
    threshold_name = (
        "minimum_delta_mse" if mode == "association" else "maximum_residual_delta_mse"
    )
    if set(adjudication) != {"mode", "targets", threshold_name}:
        raise ValueError(f"{mode} adjudication has an invalid schema")
    threshold = _finite(adjudication[threshold_name], threshold_name)
    if threshold < 0:
        raise ValueError(f"{threshold_name} must be nonnegative")
    target_rows = adjudication["targets"]
    if not isinstance(target_rows, list) or not target_rows:
        raise ValueError(f"{mode} adjudication requires explicit targets")
    targets: list[tuple[Any, ...]] = []
    for index, row in enumerate(target_rows):
        if not isinstance(row, dict) or set(row) != set(TARGET_FIELDS):
            raise ValueError(f"adjudication target {index} has an invalid schema")
        targets.append(_key(row, TARGET_FIELDS, f"adjudication target {index}"))
    if len(targets) != len(set(targets)):
        raise ValueError("adjudication targets contain duplicates")
    if any(key not in records for key in targets):
        raise ValueError("adjudication target is absent from the frozen global family")
    if len({key[:2] for key in targets}) < 2:
        raise ValueError("scientific adjudication requires at least two distinct model/layer runs")
    grouped: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for key in targets:
        record = records[key]
        if record["raw"]["label_role"] != "biological":
            raise ValueError("scientific adjudication targets must be biological-label hypotheses")
        grouped[key[:-1]].add(str(key[-1]))
    if any(blockings != set(BLOCKINGS) for blockings in grouped.values()):
        raise ValueError("each scientific target must include protein and family blocking")

    outcomes = []
    for key in targets:
        record = records[key]
        if mode == "association":
            passed = (
                record["joint_qvalue"] <= alpha
                and record["delta_mse"] >= threshold
                and record["mse_ci"][0] >= threshold
            )
            evidence = {
                "joint_qvalue": record["joint_qvalue"],
                "delta_mse": record["delta_mse"],
                "bootstrap_delta_mse_ci95_lower": record["mse_ci"][0],
            }
        else:
            if record["joint_prospective_mde"] is None:
                raise ValueError("powered-bound adjudication requires complete independent power plans")
            passed = (
                record["joint_prospective_mde"] <= threshold
                and record["mse_ci"][1] <= threshold
            )
            evidence = {
                "joint_prospective_minimum_detectable_delta_mse": record[
                    "joint_prospective_mde"
                ],
                "bootstrap_delta_mse_ci95_upper": record["mse_ci"][1],
            }
        outcomes.append({"hypothesis": dict(zip(TARGET_FIELDS, key)), "passed": passed, **evidence})
    return {
        "mode": mode,
        "status": "passed" if all(item["passed"] for item in outcomes) else "failed",
        threshold_name: threshold,
        "targets": outcomes,
        "claim_boundary": (
            "This decision applies only to the frozen targets and estimand. It does not establish "
            "a biological primitive, causal mechanism, or dictionary-wide semantic validity."
        ),
    }


def _write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def adjudicate(
    spec_path: Path,
    out_dir: Path,
    *,
    expected_spec_sha256: str,
    collector_script_path: Path | None = None,
) -> Path:
    """Validate the frozen family and atomically publish its joint receipt."""

    spec_path = Path(spec_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    frozen_hash = _digest(expected_spec_sha256, "expected_spec_sha256")
    if not spec_path.is_file() or sha256_file(spec_path) != frozen_hash:
        raise ValueError("collector spec is missing or does not match expected_spec_sha256")
    spec = _strict_json(spec_path)
    if spec.get("schema_version") != 1 or spec.get("confirmatory") is not True:
        raise ValueError("joint collector requires a confirmatory schema-1 spec")
    if set(spec) != {
        "schema_version",
        "confirmatory",
        "fdr_alpha",
        "power",
        "adjudication",
        "runs",
    }:
        raise ValueError("joint collector spec has an invalid top-level schema")
    alpha = _finite(spec["fdr_alpha"], "fdr_alpha")
    power = _finite(spec["power"], "power")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("global alpha and power must lie in (0, 1)")
    runs = spec["runs"]
    if not isinstance(runs, list) or not runs:
        raise ValueError("joint collector requires a non-empty frozen run list")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite output path: {out_dir}")
    staging = out_dir.with_name(f".{out_dir.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"stale adjudication staging path exists: {staging}")

    all_records: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    run_ids: set[tuple[str, int]] = set()
    try:
        for descriptor in runs:
            if not isinstance(descriptor, dict):
                raise ValueError("every run descriptor must be an object")
            records, run_provenance = _load_run(descriptor, spec_path.parent)
            run_id = (run_provenance["model"], run_provenance["layer"])
            if run_id in run_ids:
                raise ValueError(f"duplicate model/layer run {run_id!r}")
            if not math.isclose(run_provenance["power"], power, abs_tol=1e-12):
                raise ValueError(f"run {run_id!r} power does not match the frozen global power")
            if not math.isclose(run_provenance["fdr_alpha"], alpha, abs_tol=1e-12):
                raise ValueError(f"run {run_id!r} alpha does not match the frozen global alpha")
            run_ids.add(run_id)
            all_records.extend(records)
            provenance.append(run_provenance)
        global_keys = [record["global_key"] for record in all_records]
        if len(global_keys) != len(set(global_keys)):
            raise ValueError("duplicate hypothesis in the frozen global family")

        qvalues = benjamini_hochberg([record["pvalue"] for record in all_records])
        joint_multiplier = NormalDist().inv_cdf(1 - alpha / len(all_records)) + NormalDist().inv_cdf(power)
        by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
        for record, qvalue in zip(all_records, qvalues):
            record["joint_qvalue"] = float(qvalue)
            standard_error = record["prospective_standard_error"]
            record["joint_prospective_mde"] = (
                None if standard_error is None else float(standard_error * joint_multiplier)
            )
            by_key[record["global_key"]] = record

        adjudication = spec["adjudication"]
        if not isinstance(adjudication, dict):
            raise ValueError("adjudication must be an object")
        if adjudication.get("mode") == "powered_bound" and any(
            record["prospective_standard_error"] is None for record in all_records
        ):
            raise ValueError(
                "powered-bound adjudication requires complete independent power plans "
                "over the global family"
            )
        decision = _decision(adjudication, by_key, alpha)

        ordered = sorted(all_records, key=lambda record: record["global_key"])
        output_rows: list[dict[str, Any]] = []
        for record in ordered:
            row = {
                "model": record["model"],
                "layer": record["layer"],
                **{field: record["raw"][field] for field in EFFECT_FIELDS if field != "qvalue"},
                "reported_per_run_qvalue_ignored": record["raw"]["qvalue"],
                "joint_bh_qvalue": f"{record['joint_qvalue']:.17g}",
                "prospective_standard_error_delta_mse": (
                    ""
                    if record["prospective_standard_error"] is None
                    else f"{record['prospective_standard_error']:.17g}"
                ),
                "joint_prospective_minimum_detectable_delta_mse": (
                    ""
                    if record["joint_prospective_mde"] is None
                    else f"{record['joint_prospective_mde']:.17g}"
                ),
            }
            output_rows.append(row)
        output_fields = list(output_rows[0])

        staging.mkdir(parents=True)
        table_path = staging / "joint_conditional_effects.tsv"
        _write_tsv(table_path, output_rows, output_fields)
        summary = {
            "schema_version": 2,
            "artifact_validation_status": "verified_complete",
            "scientific_gate_status": decision["status"],
            "confirmatory": True,
            "global_multiplicity_family": (
                "model x layer x representation x feature x label x blocking"
            ),
            "fdr_method": "one global Benjamini-Hochberg adjustment from raw permutation p-values",
            "per_run_qvalues_used": False,
            "fdr_alpha": alpha,
            "power": power,
            "prospective_planning_correction": "Bonferroni over the complete global family",
            "n_runs": len(provenance),
            "n_hypotheses": len(ordered),
            "n_joint_qvalue_le_alpha": sum(record["joint_qvalue"] <= alpha for record in ordered),
            "runs": provenance,
            "decision": decision,
            "p0_4_status": (
                "This receipt is necessary but not sufficient for P0-4. P0-4 remains open unless "
                "these are real frozen runs and all protocol-level scientific criteria pass."
            ),
            "mde_labels": {
                "retrospective_bootstrap_detectable_delta_mse": (
                    "estimated from the analyzed confirmatory observations; retrospective only"
                ),
                "prospective_minimum_detectable_delta_mse": (
                    "script-53 per-run value from a hash-bound independent power plan"
                ),
                "joint_prospective_minimum_detectable_delta_mse": (
                    "recomputed from that independent plan over the complete global family"
                ),
            },
        }
        summary_path = staging / "summary.json"
        write_json(summary_path, summary)
        module_path = Path(__file__).resolve()
        source_hashes = {"module_sha256": sha256_file(module_path)}
        if collector_script_path is not None:
            script_path = Path(collector_script_path).resolve()
            source_hashes["script_sha256"] = sha256_file(script_path)
        receipt = {
            "schema_version": 2,
            "artifact_validation_status": "verified_complete",
            "scientific_gate_status": decision["status"],
            "scientific_decision_mode": decision["mode"],
            "collector_spec": {"path": str(spec_path), "sha256": frozen_hash},
            "sources": source_hashes,
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "input_run_manifest_sha256": [
                record["run_manifest_sha256"] for record in provenance
            ],
            "artifacts": {
                "joint_conditional_effects.tsv": sha256_file(table_path),
                "summary.json": sha256_file(summary_path),
            },
            "n_runs": len(provenance),
            "n_hypotheses": len(ordered),
        }
        receipt_path = staging / "completion_receipt.json"
        write_json(receipt_path, receipt)
        staging.replace(out_dir)
        return out_dir / "completion_receipt.json"
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
