#!/usr/bin/env python3
"""Run the independent-cohort atlas grid from cached activation matrices.

The JSON run spec names disjoint discovery/test cohort manifests, layer maps,
seed metadata, and a prespecified grid.  No model forward pass is performed.
Each cohort manifest has this compact form::

  {
    "cohort_id": "discovery-a",
    "sequence_hashes": ["..."],
    "matrices": [
      {"model": "protgpt2", "layer": "5", "path": "a.npy",
       "sha256": "...", "n_rows": 200, "n_features": 4096}
    ]
  }

Production specs set ``confirmatory=true`` (the default), which enforces at
least 1,000 coherent permutations, both correlation modes, all four requested
matchers, and multi-point pool/threshold/layer sweeps.  ``null.selection`` may
restrict expensive permutations to a prespecified primary matcher/mode and
canonical layer/pool/threshold setting. All matchers and modes remain in the
independent-cohort stability grid.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict
from itertools import combinations, product
from pathlib import Path

import numpy as np
import scipy


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.revision.atlas import (  # noqa: E402
    coherent_permutation_test,
    discover_atlas,
    identity_overlap,
    score_atlas,
)
from src.revision.input_builder import (  # noqa: E402
    ACTIVATION_FINITE_CHECK,
    MODEL_INFERENCE_DTYPE_VERIFICATION,
)


REQUIRED_MATCHERS = {"greedy", "hungarian", "optimal_transport", "joint_triangle"}
REQUIRED_MODES = {"positive", "absolute"}
REQUIRED_DICTIONARY_SEEDS = {17, 29, 43}
HEX = frozenset("0123456789abcdef")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path: Path) -> dict:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def canonical_id(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= HEX:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def require_inference_provenance(
    provenance: dict, *, confirmatory: bool, label: str
) -> None:
    declared = provenance.get("model_inference_dtype")
    if (
        declared not in {"float16", "bfloat16", "float32"}
        or confirmatory
        and declared != "bfloat16"
        or provenance.get("observed_model_parameter_dtypes") != [declared]
        or provenance.get("model_inference_dtype_verification")
        != MODEL_INFERENCE_DTYPE_VERIFICATION
        or provenance.get("model_inference_dtype_verified") is not True
        or provenance.get("activation_finiteness_check") != ACTIVATION_FINITE_CHECK
        or provenance.get("activation_finiteness_verified") is not True
    ):
        raise ValueError(f"{label}: model inference provenance is ineligible")


def load_cohort(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    dictionary_seed: int,
    confirmatory: bool,
) -> tuple[dict, dict, dict]:
    expected_manifest_sha256 = require_sha256(
        expected_manifest_sha256, f"{manifest_path} expected manifest SHA-256"
    )
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(f"{manifest_path}: builder manifest SHA-256 mismatch")
    manifest = strict_json(manifest_path)
    expected_status = (
        "confirmatory_inputs_built_p0_2_eligible_not_adjudicated"
        if confirmatory
        else "nonconfirmatory_fixture_inputs_only"
    )
    if (
        manifest.get("schema_version") != "r2_p0_3_atlas_input_v3"
        or type(manifest.get("confirmatory")) is not bool
        or manifest["confirmatory"] is not confirmatory
        or manifest.get("status") != expected_status
    ):
        raise ValueError(f"{manifest_path}: builder manifest scope/status is ineligible")
    model_seed = manifest.get("model_seed")
    if type(model_seed) is not int or model_seed < 0:
        raise ValueError(f"{manifest_path}: builder model_seed must be a non-negative integer")
    cohort_id = manifest.get("cohort_id")
    sequence_hashes = manifest.get("sequence_hashes")
    records = manifest.get("matrices")
    if not isinstance(cohort_id, str) or not cohort_id:
        raise ValueError(f"{manifest_path}: cohort_id must be a non-empty string")
    if not isinstance(sequence_hashes, list) or len(sequence_hashes) < 2:
        raise ValueError(f"{manifest_path}: sequence_hashes must contain at least two rows")
    if len(set(sequence_hashes)) != len(sequence_hashes) or not all(
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
        for value in sequence_hashes
    ):
        raise ValueError(f"{manifest_path}: sequence_hashes must be unique lowercase SHA-256 values")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{manifest_path}: matrices must be a non-empty list")

    matrices: dict[str, dict[str, np.ndarray]] = {}
    files = []
    seen = set()
    selected_provenance: dict[str, dict] = {}
    receipt = manifest.get("p0_2_eligibility_receipt")
    if confirmatory:
        if not isinstance(receipt, dict):
            raise ValueError(f"{manifest_path}: confirmatory inputs lack a P0-2 receipt")
        receipt_sha256 = require_sha256(
            receipt.get("sha256"), f"{manifest_path} P0-2 receipt SHA-256"
        )
    else:
        if receipt is not None:
            raise ValueError(f"{manifest_path}: nonconfirmatory inputs declare P0-2 eligibility")
        receipt_sha256 = None
    model_provenance = manifest.get("models")
    if not isinstance(model_provenance, dict):
        raise ValueError(f"{manifest_path}: models provenance must be an object")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{manifest_path}: each matrix record must be an object")
        model = record.get("model")
        seed = record.get("dictionary_seed")
        layer = str(record.get("layer"))
        if (
            not isinstance(model, str)
            or not model
            or type(seed) is not int
            or record.get("layer") is None
        ):
            raise ValueError(
                f"{manifest_path}: matrix records require model, dictionary_seed and layer"
            )
        identity = (model, seed, layer)
        if identity in seen:
            raise ValueError(f"{manifest_path}: duplicate matrix {identity}")
        seen.add(identity)
        if seed != dictionary_seed:
            continue
        by_seed = model_provenance.get(model)
        provenance = record.get("dictionary_provenance")
        if (
            not isinstance(by_seed, dict)
            or not isinstance(provenance, dict)
            or by_seed.get(str(seed)) != provenance
        ):
            raise ValueError(f"{manifest_path}: matrix dictionary provenance mismatch")
        if (
            provenance.get("confirmatory") is not confirmatory
            or provenance.get("run_seed") != seed
            or provenance.get("eligibility_receipt_sha256") != receipt_sha256
        ):
            raise ValueError(f"{manifest_path}: matrix seed/eligibility provenance mismatch")
        require_inference_provenance(
            provenance,
            confirmatory=confirmatory,
            label=f"{manifest_path} {model} seed {seed}",
        )
        for field in ("checkpoint_sha256", "run_manifest_sha256"):
            require_sha256(provenance.get(field), f"{manifest_path} {model} seed {seed} {field}")
        model_artifacts = provenance.get("model_artifacts")
        if not isinstance(model_artifacts, dict) or set(model_artifacts) != {
            "model_config_sha256",
            "model_weights_sha256",
            "tokenizer_sha256",
        }:
            raise ValueError(f"{manifest_path}: incomplete {model} model-artifact provenance")
        for field, digest in model_artifacts.items():
            require_sha256(digest, f"{manifest_path} {model} {field}")
        if confirmatory:
            source_hashes = provenance.get("source_manifest_sha256_by_split")
            if not isinstance(source_hashes, dict) or set(source_hashes) != {
                "train",
                "validation",
                "test",
            }:
                raise ValueError(f"{manifest_path}: incomplete P0-2 source-manifest provenance")
            for split, digest in source_hashes.items():
                require_sha256(digest, f"{manifest_path} {model} {split} source hash")
            if layer not in {
                str(value) for value in provenance.get("eligible_downstream_layers", [])
            }:
                raise ValueError(f"{manifest_path}: matrix layer is outside the P0-2 allowlist")
        path = resolve_path(str(record.get("path")), manifest_path.parent)
        if not path.is_file() or path.suffix != ".npy":
            raise FileNotFoundError(f"matrix must be an existing .npy file: {path}")
        actual_hash = sha256_file(path)
        expected_hash = record.get("sha256")
        if not isinstance(expected_hash, str) or expected_hash != actual_hash:
            raise ValueError(f"SHA-256 mismatch or missing expected hash for {path}")
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        if matrix.ndim != 2 or matrix.shape[0] != len(sequence_hashes):
            raise ValueError(f"{path}: expected [{len(sequence_hashes)}, feature] matrix")
        if record.get("n_rows") != matrix.shape[0] or record.get("n_features") != matrix.shape[1]:
            raise ValueError(f"{path}: declared dimensions do not match the array")
        if not np.issubdtype(matrix.dtype, np.number):
            raise ValueError(f"{path}: activation matrix must be numeric")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{path}: activation matrix contains non-finite values")
        matrices.setdefault(model, {})[layer] = matrix
        selected_provenance[model] = provenance
        files.append(
            {
                "model": model,
                "layer": layer,
                "path": str(path),
                "sha256": actual_hash,
                "shape": list(matrix.shape),
                "dtype": str(matrix.dtype),
            }
        )
    if not matrices:
        raise ValueError(
            f"{manifest_path}: no matrices exist for dictionary seed {dictionary_seed}"
        )
    if confirmatory:
        from src.revision.dictionary_gate import require_eligible_model_method

        receipt_path = resolve_path(str(receipt.get("path")), manifest_path.parent)
        for model, provenance in selected_provenance.items():
            by_seed = model_provenance.get(model)
            if not isinstance(by_seed, dict) or set(by_seed) != {
                str(seed) for seed in REQUIRED_DICTIONARY_SEEDS
            }:
                raise ValueError(f"{manifest_path}: {model} lacks exact all-seed provenance")
            run_hashes = {}
            checkpoint_hashes = {}
            for seed in REQUIRED_DICTIONARY_SEEDS:
                row = by_seed[str(seed)]
                if not isinstance(row, dict) or row.get("run_seed") != seed:
                    raise ValueError(f"{manifest_path}: malformed {model} seed provenance")
                run_hashes[seed] = require_sha256(
                    row.get("run_manifest_sha256"),
                    f"{manifest_path} {model} seed {seed} run-manifest hash",
                )
                checkpoint_hashes[seed] = require_sha256(
                    row.get("checkpoint_sha256"),
                    f"{manifest_path} {model} seed {seed} checkpoint hash",
                )
            if len(set(checkpoint_hashes.values())) != len(REQUIRED_DICTIONARY_SEEDS):
                raise ValueError(f"{manifest_path}: {model} seed checkpoints are not distinct")
            require_eligible_model_method(
                receipt_path,
                receipt_sha256,
                model_name=model,
                method="topk_clt",
                expected_run_manifest_sha256_by_seed=run_hashes,
                expected_checkpoint_sha256_by_seed=checkpoint_hashes,
                expected_source_manifest_sha256_by_split=provenance[
                    "source_manifest_sha256_by_split"
                ],
                requested_layers=[int(layer) for layer in matrices[model]],
            )
            checkpoint = Path(str(provenance.get("checkpoint"))).resolve()
            if (
                not checkpoint.is_file()
                or sha256_file(checkpoint) != provenance["checkpoint_sha256"]
            ):
                raise ValueError(
                    f"{manifest_path}: selected {model} seed {dictionary_seed} best.pt changed"
                )
    provenance = {
        "manifest": str(manifest_path),
        "manifest_sha256": actual_manifest_sha256,
        "cohort_id": cohort_id,
        "n_sequences": len(sequence_hashes),
        "ordered_sequence_hash": hashlib.sha256(
            ("\n".join(sequence_hashes) + "\n").encode("utf-8")
        ).hexdigest(),
        "files": files,
        "confirmatory": confirmatory,
        "model_seed": model_seed,
        "dictionary_seed": dictionary_seed,
        "eligibility_receipt_sha256": receipt_sha256,
        "dictionary_artifacts": {
            model: {
                "run_seed": provenance["run_seed"],
                "run_manifest_sha256": provenance["run_manifest_sha256"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "source_manifest_sha256_by_split": provenance[
                    "source_manifest_sha256_by_split"
                ],
                "model_artifacts": provenance["model_artifacts"],
            }
            for model, provenance in sorted(selected_provenance.items())
        },
    }
    return matrices, manifest, provenance


def validate_grid(spec: dict) -> dict:
    grid = spec.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("run spec requires a grid object")
    required = ("feature_pool_sizes", "matchers", "correlation_modes", "thresholds")
    if any(not isinstance(grid.get(key), list) or not grid[key] for key in required):
        raise ValueError(f"grid requires non-empty lists: {', '.join(required)}")
    pools = [int(value) for value in grid["feature_pool_sizes"]]
    thresholds = [float(value) for value in grid["thresholds"]]
    matchers = [str(value) for value in grid["matchers"]]
    modes = [str(value) for value in grid["correlation_modes"]]
    if len(set(pools)) != len(pools) or any(value < 1 for value in pools):
        raise ValueError("feature_pool_sizes must be unique positive integers")
    if len(set(thresholds)) != len(thresholds) or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("thresholds must be unique and within [0, 1]")
    if len(set(matchers)) != len(matchers) or len(set(modes)) != len(modes):
        raise ValueError("matcher and correlation-mode grids must not contain duplicates")
    if not set(matchers) <= REQUIRED_MATCHERS or not set(modes) <= REQUIRED_MODES:
        raise ValueError("unknown matcher or correlation mode")
    result = {
        "feature_pool_sizes": pools,
        "thresholds": thresholds,
        "matchers": matchers,
        "correlation_modes": modes,
        "max_matches": grid.get("max_matches"),
        "ambiguity_tolerance": float(grid.get("ambiguity_tolerance", 0.02)),
        "ot_regularization": float(grid.get("ot_regularization", 0.05)),
        "joint_candidate_width": int(grid.get("joint_candidate_width", 8)),
    }
    limits = grid.get("max_feature_pool_by_matcher", {})
    if not isinstance(limits, dict) or not set(limits) <= set(matchers):
        raise ValueError("max_feature_pool_by_matcher must only name configured matchers")
    result["max_feature_pool_by_matcher"] = {
        matcher: int(limits.get(matcher, max(pools))) for matcher in matchers
    }
    if any(value < min(pools) for value in result["max_feature_pool_by_matcher"].values()):
        raise ValueError("every matcher must permit at least the smallest feature pool")
    if result["max_matches"] is not None:
        result["max_matches"] = int(result["max_matches"])
        if result["max_matches"] < 1:
            raise ValueError("max_matches must be positive or null")
    return result


def validate_confirmatory(spec: dict, grid: dict, analyses: list[dict]) -> None:
    if not spec.get("confirmatory", True):
        return
    if len(analyses) != len(REQUIRED_DICTIONARY_SEEDS):
        raise ValueError("confirmatory atlas requires exactly one analysis per dictionary seed")
    n_permutations = int(spec.get("null", {}).get("n_permutations", 1000))
    if n_permutations < 1_000:
        raise ValueError("confirmatory atlas runs require at least 1,000 null replicates")
    if set(grid["matchers"]) != REQUIRED_MATCHERS:
        raise ValueError("confirmatory grid must include all four requested matchers")
    if set(grid["correlation_modes"]) != REQUIRED_MODES:
        raise ValueError("confirmatory grid must include positive and absolute modes")
    if len(grid["feature_pool_sizes"]) < 2 or len(grid["thresholds"]) < 2:
        raise ValueError("confirmatory grid requires multi-point pool and threshold sweeps")
    if any(len(analysis.get("layer_maps", {})) < 2 for analysis in analyses):
        raise ValueError("confirmatory analyses require at least two prespecified layer maps")
    for analysis in analyses:
        if type(analysis.get("model_seed")) is not int or analysis["model_seed"] < 0:
            raise ValueError("confirmatory analyses require non-negative integer model_seed")
        if type(analysis.get("dictionary_seed")) is not int:
            raise ValueError("confirmatory analyses require integer dictionary_seed")
        for field in (
            "discovery_manifest_sha256",
            "heldout_manifest_sha256",
            "eligibility_receipt_sha256",
        ):
            require_sha256(analysis.get(field), f"confirmatory analysis {field}")
    observed_seeds = {analysis["dictionary_seed"] for analysis in analyses}
    if observed_seeds != REQUIRED_DICTIONARY_SEEDS:
        raise ValueError(
            "confirmatory atlas requires exactly dictionary seeds 17, 29 and 43; "
            f"observed {sorted(observed_seeds)}"
        )


def validate_distinct_eligible_seed_artifacts(seed_artifacts: dict[int, dict]) -> None:
    """Require each confirmatory seed label to resolve to distinct eligible artifacts."""

    if set(seed_artifacts) != REQUIRED_DICTIONARY_SEEDS:
        raise ValueError("confirmatory atlas lacks the exact eligible seed artifact panel")
    model_sets = {tuple(sorted(artifacts)) for artifacts in seed_artifacts.values()}
    if len(model_sets) != 1 or not next(iter(model_sets)):
        raise ValueError("confirmatory seed artifacts do not share one non-empty model panel")
    for seed, artifacts in seed_artifacts.items():
        for model, row in artifacts.items():
            if not isinstance(row, dict) or row.get("run_seed") != seed:
                raise ValueError(
                    f"confirmatory {model} artifact does not bind dictionary seed {seed}"
                )
            for field in ("checkpoint_sha256", "run_manifest_sha256"):
                require_sha256(row.get(field), f"confirmatory {model} seed {seed} {field}")
            model_artifacts = row.get("model_artifacts")
            if not isinstance(model_artifacts, dict) or set(model_artifacts) != {
                "model_config_sha256",
                "model_weights_sha256",
                "tokenizer_sha256",
            }:
                raise ValueError(
                    f"confirmatory {model} seed {seed} lacks model checkpoint provenance"
                )
            for field, digest in model_artifacts.items():
                require_sha256(digest, f"confirmatory {model} seed {seed} {field}")
    for model in next(iter(model_sets)):
        checkpoints = {
            seed_artifacts[seed][model]["checkpoint_sha256"]
            for seed in REQUIRED_DICTIONARY_SEEDS
        }
        runs = {
            seed_artifacts[seed][model]["run_manifest_sha256"]
            for seed in REQUIRED_DICTIONARY_SEEDS
        }
        model_artifact_sets = {
            json.dumps(
                seed_artifacts[seed][model].get("model_artifacts"),
                sort_keys=True,
                separators=(",", ":"),
            )
            for seed in REQUIRED_DICTIONARY_SEEDS
        }
        if len(checkpoints) != len(REQUIRED_DICTIONARY_SEEDS) or len(runs) != len(
            REQUIRED_DICTIONARY_SEEDS
        ):
            raise ValueError(
                f"confirmatory analyses do not point to distinct eligible {model} seed artifacts"
            )
        if len(model_artifact_sets) != 1:
            raise ValueError(
                f"confirmatory {model} seed analyses do not share one model checkpoint provenance"
            )


def normalize_null_selection(null_spec: dict, grid: dict, analyses: list[dict]) -> dict:
    selection = null_spec.get("selection")
    available_layer_maps = {
        str(name) for analysis in analyses for name in analysis.get("layer_maps", {})
    }
    if selection is None:
        return {
            "layer_maps": available_layer_maps,
            "feature_pool_sizes": set(grid["feature_pool_sizes"]),
            "matchers": set(grid["matchers"]),
            "correlation_modes": set(grid["correlation_modes"]),
            "thresholds": set(grid["thresholds"]),
        }
    if not isinstance(selection, dict):
        raise ValueError("null.selection must be an object")
    required = (
        "layer_maps",
        "feature_pool_sizes",
        "matchers",
        "correlation_modes",
        "thresholds",
    )
    if any(not isinstance(selection.get(key), list) or not selection[key] for key in required):
        raise ValueError("null.selection requires non-empty layer_maps, feature_pool_sizes, thresholds")
    result = {
        "layer_maps": {str(value) for value in selection["layer_maps"]},
        "feature_pool_sizes": {int(value) for value in selection["feature_pool_sizes"]},
        "matchers": {str(value) for value in selection["matchers"]},
        "correlation_modes": {str(value) for value in selection["correlation_modes"]},
        "thresholds": {float(value) for value in selection["thresholds"]},
    }
    if not result["layer_maps"] <= available_layer_maps:
        raise ValueError("null.selection contains an unknown layer map")
    if not result["feature_pool_sizes"] <= set(grid["feature_pool_sizes"]):
        raise ValueError("null.selection contains a pool size outside the grid")
    if not result["matchers"] <= set(grid["matchers"]):
        raise ValueError("null.selection contains a matcher outside the grid")
    if not result["correlation_modes"] <= set(grid["correlation_modes"]):
        raise ValueError("null.selection contains a correlation mode outside the grid")
    if not result["thresholds"] <= set(grid["thresholds"]):
        raise ValueError("null.selection contains a threshold outside the grid")
    return result


def selected_for_null(parameters: dict, selection: dict) -> bool:
    return (
        parameters["layer_map"] in selection["layer_maps"]
        and parameters["feature_pool_size"] in selection["feature_pool_sizes"]
        and parameters["matcher"] in selection["matchers"]
        and parameters["correlation_mode"] in selection["correlation_modes"]
        and parameters["threshold"] in selection["thresholds"]
    )


def normalize_layer_maps(value: dict, models: list[str]) -> dict:
    if not isinstance(value, dict) or not value:
        raise ValueError("each analysis requires a non-empty layer_maps object")
    result = {}
    for map_name, groups in value.items():
        if not isinstance(groups, dict) or not groups:
            raise ValueError(f"layer map {map_name!r} must contain layer groups")
        result[str(map_name)] = {}
        for group_name, mapping in groups.items():
            if not isinstance(mapping, dict) or set(mapping) != set(models):
                raise ValueError(f"layer group {group_name!r} must specify every model")
            result[str(map_name)][str(group_name)] = {
                model: str(mapping[model]) for model in models
            }
    return result


def match_rows(config_id: str, atlas, evaluation) -> list[dict]:
    heldout = {match.identity: match for match in evaluation.matches}
    rows = []
    for index, match in enumerate(atlas.matches):
        scored = heldout[match.identity]
        rows.append(
            {
                "config_id": config_id,
                "match_index": index,
                "group": str(match.group),
                "identity_json": json.dumps(match.identity, separators=(",", ":")),
                "discovery_signed_correlations_json": json.dumps(match.discovery_signed_correlations),
                "discovery_matching_scores_json": json.dumps(match.discovery_matching_scores),
                "discovery_ambiguity": match.ambiguity,
                "discovery_confidence": match.confidence,
                "heldout_signed_correlations_json": json.dumps(scored.signed_correlations),
                "heldout_matching_scores_json": json.dumps(scored.matching_scores),
                "heldout_ambiguity": scored.ambiguity,
                "heldout_confidence": scored.confidence,
                "heldout_passes_threshold": int(scored.passes_threshold),
            }
        )
    return rows


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def factor_design(rows: list[dict], factors: list[str], exclude: str | None = None) -> np.ndarray:
    columns = [np.ones(len(rows))]
    for factor in factors:
        if factor == exclude:
            continue
        values = [str(row[factor]) for row in rows]
        levels = sorted(set(values))
        columns.extend(np.asarray(values) == level for level in levels[1:])
    return np.column_stack(columns).astype(np.float64)


def variance_attribution(rows: list[dict], metrics: list[str]) -> list[dict]:
    """Drop-one fixed-effect partial R2 for prespecified stability factors."""

    factors = [
        "discovery_cohort",
        "model_seed",
        "dictionary_seed",
        "matcher",
        "correlation_mode",
        "layer_map",
        "feature_pool_size",
        "threshold",
    ]
    full = factor_design(rows, factors)
    full_rank = int(np.linalg.matrix_rank(full))
    output = []
    for metric in metrics:
        target = np.asarray([row[metric] for row in rows], dtype=np.float64)
        full_residual = target - full @ np.linalg.lstsq(full, target, rcond=None)[0]
        full_sse = float(full_residual @ full_residual)
        for factor in factors:
            levels = len({str(row[factor]) for row in rows})
            reduced = factor_design(rows, factors, exclude=factor)
            estimable = levels > 1 and np.linalg.matrix_rank(reduced) < full_rank
            if estimable:
                residual = target - reduced @ np.linalg.lstsq(reduced, target, rcond=None)[0]
                reduced_sse = float(residual @ residual)
                partial = max(0.0, (reduced_sse - full_sse) / reduced_sse) if reduced_sse > 0 else 0.0
            else:
                partial = None
            output.append(
                {
                    "metric": metric,
                    "factor": factor,
                    "n_levels": levels,
                    "estimable": estimable,
                    "partial_r2": partial,
                    "method": "drop-one fixed-effect partial R2",
                }
            )
    return output


def run(spec_path: Path, out_dir: Path) -> None:
    spec = strict_json(spec_path)
    if type(spec.get("confirmatory", True)) is not bool:
        raise ValueError("confirmatory must be a JSON boolean")
    confirmatory = bool(spec.get("confirmatory", True))
    analyses = spec.get("analyses")
    if not isinstance(analyses, list) or not analyses:
        raise ValueError("run spec requires a non-empty analyses list")
    grid = validate_grid(spec)
    validate_confirmatory(spec, grid, analyses)
    null_spec = spec.get("null", {})
    if not isinstance(null_spec, dict):
        raise ValueError("null must be an object")
    n_permutations = int(null_spec.get("n_permutations", 1_000))
    base_seed = int(null_spec.get("seed", 0))
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    null_selection = normalize_null_selection(null_spec, grid, analyses)

    summary_rows = []
    all_match_rows = []
    null_rows = []
    atlas_by_config = {}
    cohort_provenance = []
    seed_artifacts: dict[int, dict] = {}
    for analysis in analyses:
        if not isinstance(analysis, dict) or not isinstance(analysis.get("name"), str):
            raise ValueError("each analysis requires a string name")
        models = analysis.get("models")
        if not isinstance(models, list) or len(models) != 3 or len(set(models)) != 3:
            raise ValueError("each analysis requires exactly three distinct models")
        layer_maps = normalize_layer_maps(analysis.get("layer_maps"), models)
        dictionary_seed = analysis.get("dictionary_seed")
        if type(dictionary_seed) is not int:
            raise ValueError("each atlas analysis requires an integer dictionary_seed")
        discovery_path = resolve_path(str(analysis.get("discovery_manifest")), spec_path.parent)
        heldout_path = resolve_path(str(analysis.get("heldout_manifest")), spec_path.parent)
        discovery, discovery_manifest, discovery_provenance = load_cohort(
            discovery_path,
            expected_manifest_sha256=analysis.get("discovery_manifest_sha256"),
            dictionary_seed=dictionary_seed,
            confirmatory=confirmatory,
        )
        heldout, heldout_manifest, heldout_provenance = load_cohort(
            heldout_path,
            expected_manifest_sha256=analysis.get("heldout_manifest_sha256"),
            dictionary_seed=dictionary_seed,
            confirmatory=confirmatory,
        )
        if discovery_manifest["cohort_id"] == heldout_manifest["cohort_id"]:
            raise ValueError("discovery and held-out cohort_id values must differ")
        overlap = set(discovery_manifest["sequence_hashes"]) & set(heldout_manifest["sequence_hashes"])
        if overlap:
            raise ValueError(
                f"analysis {analysis['name']!r} discovery/test cohorts overlap by {len(overlap)} sequences"
            )
        if set(discovery) != set(models) or set(heldout) != set(models):
            raise ValueError("cohort model identities must exactly match analysis.models")
        if (
            discovery_provenance["dictionary_artifacts"]
            != heldout_provenance["dictionary_artifacts"]
            or discovery_provenance["eligibility_receipt_sha256"]
            != heldout_provenance["eligibility_receipt_sha256"]
            or discovery_provenance["model_seed"] != heldout_provenance["model_seed"]
        ):
            raise ValueError("discovery/heldout dictionary provenance differs")
        if discovery_provenance["model_seed"] != analysis.get("model_seed"):
            raise ValueError("analysis model_seed differs from the builder manifests")
        if confirmatory and (
            discovery_provenance["eligibility_receipt_sha256"]
            != analysis.get("eligibility_receipt_sha256")
        ):
            raise ValueError("analysis eligibility-receipt hash differs from builder manifests")
        if dictionary_seed in seed_artifacts:
            raise ValueError("atlas analyses duplicate a dictionary seed artifact")
        current_artifacts = discovery_provenance["dictionary_artifacts"]
        if confirmatory:
            for prior_seed, prior_artifacts in seed_artifacts.items():
                for model in current_artifacts:
                    if (
                        current_artifacts[model]["checkpoint_sha256"]
                        == prior_artifacts[model]["checkpoint_sha256"]
                        or current_artifacts[model]["run_manifest_sha256"]
                        == prior_artifacts[model]["run_manifest_sha256"]
                    ):
                        raise ValueError(
                            f"dictionary seeds {prior_seed} and {dictionary_seed} reuse "
                            f"the same eligible {model} artifact"
                        )
        seed_artifacts[dictionary_seed] = current_artifacts
        cohort_provenance.extend((discovery_provenance, heldout_provenance))
        metadata = {
            "analysis": analysis["name"],
            "discovery_cohort": discovery_manifest["cohort_id"],
            "heldout_cohort": heldout_manifest["cohort_id"],
            "model_seed": str(analysis.get("model_seed", "unspecified")),
            "dictionary_seed": str(dictionary_seed),
        }
        for layer_map, pool_size, matcher, mode, threshold in product(
            layer_maps,
            grid["feature_pool_sizes"],
            grid["matchers"],
            grid["correlation_modes"],
            grid["thresholds"],
        ):
            if pool_size > grid["max_feature_pool_by_matcher"][matcher]:
                continue
            parameters = {
                **metadata,
                "layer_map": layer_map,
                "feature_pool_size": pool_size,
                "matcher": matcher,
                "correlation_mode": mode,
                "threshold": threshold,
            }
            config_id = canonical_id(parameters)
            atlas = discover_atlas(
                discovery,
                layer_groups=layer_maps[layer_map],
                models=models,
                feature_pool_size=pool_size,
                matcher=matcher,
                correlation_mode=mode,
                threshold=threshold,
                max_matches=grid["max_matches"],
                ambiguity_tolerance=grid["ambiguity_tolerance"],
                ot_regularization=grid["ot_regularization"],
                joint_candidate_width=grid["joint_candidate_width"],
            )
            evaluation = score_atlas(atlas, heldout)
            run_null = selected_for_null(parameters, null_selection)
            null_seed = base_seed + int(config_id[:12], 16) if run_null else None
            null = (
                coherent_permutation_test(
                    atlas,
                    discovery,
                    n_permutations=n_permutations,
                    seed=null_seed,
                )
                if run_null
                else None
            )
            heldout_passing = [match for match in evaluation.matches if match.passes_threshold]
            heldout_signed_edges = [
                value for match in heldout_passing for value in match.signed_correlations
            ]
            row = {
                "config_id": config_id,
                **parameters,
                "n_discovery_matches": len(atlas.matches),
                "n_heldout_passing": evaluation.n_passing,
                "heldout_retained_identity_jaccard": evaluation.retained_identity_jaccard,
                "discovery_mean_min_score": mean_or_zero(
                    [match.minimum_matching_score for match in atlas.matches]
                ),
                "heldout_mean_min_score": mean_or_zero(
                    [match.minimum_matching_score for match in evaluation.matches]
                ),
                "heldout_passing_mean_signed_correlation": mean_or_zero(
                    [match.mean_signed_correlation for match in heldout_passing]
                ),
                "heldout_passing_negative_edge_fraction": (
                    float(np.mean(np.asarray(heldout_signed_edges) < 0))
                    if heldout_signed_edges
                    else 0.0
                ),
                "discovery_mean_confidence": mean_or_zero(
                    [match.confidence for match in atlas.matches]
                ),
                "discovery_mean_ambiguity": mean_or_zero(
                    [match.ambiguity for match in atlas.matches]
                ),
                "null_seed": null_seed,
                "null_replicates": n_permutations if run_null else 0,
                "null_count_pvalue": null.count_pvalue if null is not None else None,
                "null_mean_score_pvalue": null.mean_score_pvalue if null is not None else None,
            }
            summary_rows.append(row)
            all_match_rows.extend(match_rows(config_id, atlas, evaluation))
            if null is not None:
                null_rows.extend(
                    {
                        "config_id": config_id,
                        "replicate": index,
                        "match_count": count,
                        "mean_min_score": score,
                    }
                    for index, (count, score) in enumerate(
                        zip(null.null_counts, null.null_mean_scores)
                    )
            )
            atlas_by_config[config_id] = atlas

    if confirmatory:
        validate_distinct_eligible_seed_artifacts(seed_artifacts)

    if not null_rows:
        raise ValueError("null.selection did not select any executable grid configuration")

    jaccard_rows = []
    for left, right in combinations(summary_rows, 2):
        overlap = identity_overlap(atlas_by_config[left["config_id"]], atlas_by_config[right["config_id"]])
        jaccard_rows.append(
            {
                "left_config_id": left["config_id"],
                "right_config_id": right["config_id"],
                **asdict(overlap),
            }
        )
    variance_rows = variance_attribution(
        summary_rows,
        ["n_heldout_passing", "heldout_mean_min_score"],
    )

    summary_fields = list(summary_rows[0])
    match_fields = list(all_match_rows[0]) if all_match_rows else [
        "config_id",
        "match_index",
        "group",
        "identity_json",
        "discovery_signed_correlations_json",
        "discovery_matching_scores_json",
        "discovery_ambiguity",
        "discovery_confidence",
        "heldout_signed_correlations_json",
        "heldout_matching_scores_json",
        "heldout_ambiguity",
        "heldout_confidence",
        "heldout_passes_threshold",
    ]
    write_tsv(out_dir / "atlas_grid.tsv", summary_rows, summary_fields)
    write_tsv(out_dir / "atlas_matches.tsv", all_match_rows, match_fields)
    write_tsv(
        out_dir / "atlas_null.tsv",
        null_rows,
        ["config_id", "replicate", "match_count", "mean_min_score"],
    )
    write_tsv(
        out_dir / "atlas_identity_jaccard.tsv",
        jaccard_rows,
        ["left_config_id", "right_config_id", "n_left", "n_right", "n_intersection", "n_union", "jaccard"],
    )
    write_tsv(
        out_dir / "atlas_variance_attribution.tsv",
        variance_rows,
        ["metric", "factor", "n_levels", "estimable", "partial_r2", "method"],
    )
    summary = {
        "schema_version": 2,
        "confirmatory": confirmatory,
        "input_status": (
            "verified_distinct_p0_2_eligible_seed_artifacts"
            if confirmatory
            else "nonconfirmatory_fixture_inputs_only"
        ),
        "gate_status": "not_adjudicated; prespecified scientific acceptance thresholds required",
        "n_configurations": len(summary_rows),
        "n_null_configurations": sum(row["null_replicates"] > 0 for row in summary_rows),
        "n_null_replicates_per_configuration": n_permutations,
        "null_selection": {key: sorted(value) for key, value in null_selection.items()},
        "cohorts": cohort_provenance,
        "dictionary_seed_artifacts": {
            str(seed): artifacts for seed, artifacts in sorted(seed_artifacts.items())
        },
        "grid": grid,
        "variance_method": (
            "Descriptive drop-one fixed-effect partial R2. A factor is marked non-estimable "
            "when only one level is present or the design is rank-confounded."
        ),
        "claim_boundary": (
            "Outputs quantify procedure and cohort stability. They do not establish semantic "
            "identity, causal mechanism, or architecture independence."
        ),
    }
    write_json(out_dir / "summary.json", summary)

    output_paths = sorted(path for path in out_dir.iterdir() if path.is_file())
    run_manifest = {
        "schema_version": 2,
        "confirmatory": confirmatory,
        "input_status": summary["input_status"],
        "command": [sys.executable, *sys.argv],
        "run_spec": str(spec_path),
        "run_spec_sha256": sha256_file(spec_path),
        "script_sha256": sha256_file(Path(__file__)),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "dictionary_seed_artifacts": summary["dictionary_seed_artifacts"],
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in output_paths
        ],
    }
    write_json(out_dir / "run_manifest.json", run_manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run(args.spec.resolve(), args.out_dir.resolve())


if __name__ == "__main__":
    main()
