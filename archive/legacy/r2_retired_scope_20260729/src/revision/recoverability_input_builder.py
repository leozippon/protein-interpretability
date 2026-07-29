"""Build immutable, provenance-bound inputs for P0-8 recoverability.

The production path extracts every representation from one verified local
model and all quality-gated CLT checkpoints.  Targets and identity groups are
kept outside the extractor call, preventing task labels from influencing
representation construction.  No precomputed representation array is an
accepted input.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .input_builder import (
    ACTIVATION_FINITE_CHECK,
    MODEL_INFERENCE_DTYPE_VERIFICATION,
    _require_keys,
    _require_sha256,
    _resolve,
    format_model_input,
    load_json,
    load_jsonl,
    residue_token_indices,
    verify_model_artifacts,
)
from .io import sha256_file, write_json
from .nested_recoverability import PRODUCTION_MINIMUM_SAMPLES


COHORT_FIELDS = {"id", "source", "sequence", "split", "family", "sha256"}
ANNOTATION_FIELDS = {
    "row_id",
    "sequence_sha256",
    "identity_group",
    "target",
    "residue_positions",
}
IDENTITY_FIELDS = {"sequence_sha256", "identity_group", "cohort_role"}
PRIOR_ROLES = ("p0_2_train", "p0_2_validation", "p0_2_test")
MODEL_DIGEST_FIELDS = {
    "model_config_sha256",
    "model_weights_sha256",
    "tokenizer_sha256",
}
def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_descriptor(descriptor: Any, base: Path, label: str) -> tuple[Path, str]:
    descriptor = _require_keys(descriptor, {"path", "sha256"}, set(), label)
    if not isinstance(descriptor["path"], str):
        raise ValueError(f"{label}.path must be a string")
    path = _resolve(descriptor["path"], base)
    digest = _require_sha256(descriptor["sha256"], f"{label}.sha256")
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"{label} path or SHA-256 mismatch")
    return path, digest


def _load_cohort(descriptor: Any, base: Path, expected_role: str) -> dict[str, Any]:
    descriptor = _require_keys(
        descriptor,
        {"role", "cohort_id", "path", "sha256", "split"},
        set(),
        f"{expected_role} cohort",
    )
    if descriptor["role"] != expected_role:
        raise ValueError(f"expected cohort role {expected_role!r}")
    if not isinstance(descriptor["cohort_id"], str) or not descriptor["cohort_id"]:
        raise ValueError("cohort_id must be a non-empty string")
    if not isinstance(descriptor["split"], str) or not descriptor["split"]:
        raise ValueError("cohort split must be a non-empty string")
    path, digest = _load_descriptor(
        {"path": descriptor["path"], "sha256": descriptor["sha256"]},
        base,
        f"{expected_role} cohort",
    )
    rows = load_jsonl(path)
    seen_ids: set[str] = set()
    seen_sequences: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != COHORT_FIELDS:
            raise ValueError(f"invalid cohort fields at {path}:{index + 1}")
        if any(not isinstance(row[field], str) or not row[field] for field in COHORT_FIELDS):
            raise ValueError(f"empty cohort field at {path}:{index + 1}")
        if row["split"] != descriptor["split"]:
            raise ValueError(f"cohort split mismatch at {path}:{index + 1}")
        observed = hashlib.sha256(row["sequence"].encode("utf-8")).hexdigest()
        if observed != row["sha256"]:
            raise ValueError(f"sequence SHA-256 mismatch at {path}:{index + 1}")
        if row["id"] in seen_ids or observed in seen_sequences:
            raise ValueError(f"duplicate cohort id or sequence at {path}:{index + 1}")
        if not row["sequence"].isalpha() or not row["sequence"].isupper():
            raise ValueError(f"sequence must use uppercase residues at {path}:{index + 1}")
        seen_ids.add(row["id"])
        seen_sequences.add(observed)
    return {
        **descriptor,
        "path": path,
        "sha256": digest,
        "rows": rows,
        "sequence_hashes": seen_sequences,
    }


def _load_task_annotations(
    descriptor: Any,
    base: Path,
    cohort: Mapping[str, Any],
    *,
    task_type: str,
    pooling: str,
    minimum_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path, digest = _load_descriptor(descriptor, base, "task_annotations")
    cohort_by_hash = {row["sha256"]: row for row in cohort["rows"]}
    rows = load_jsonl(path)
    seen_ids: set[str] = set()
    covered_sequences: set[str] = set()
    targets: list[Any] = []
    position_count: int | None = None
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != ANNOTATION_FIELDS:
            raise ValueError(f"invalid task-annotation fields at {path}:{index + 1}")
        if not isinstance(row["row_id"], str) or not row["row_id"]:
            raise ValueError("task row_id must be a non-empty string")
        if row["row_id"] in seen_ids:
            raise ValueError(f"duplicate task row_id: {row['row_id']}")
        sequence_hash = _require_sha256(
            row["sequence_sha256"], f"task row {row['row_id']} sequence_sha256"
        )
        if sequence_hash not in cohort_by_hash:
            raise ValueError(f"task row references sequence outside evaluation cohort: {row['row_id']}")
        if not isinstance(row["identity_group"], str) or not row["identity_group"]:
            raise ValueError("identity_group must be a non-empty string")
        positions = row["residue_positions"]
        sequence_length = len(cohort_by_hash[sequence_hash]["sequence"])
        if positions is None:
            positions = list(range(sequence_length))
        if (
            not isinstance(positions, list)
            or not positions
            or any(type(value) is not int or not 0 <= value < sequence_length for value in positions)
            or len(set(positions)) != len(positions)
        ):
            raise ValueError(f"invalid residue_positions for task row {row['row_id']}")
        if pooling == "ordered_concatenate":
            if position_count is None:
                position_count = len(positions)
            elif len(positions) != position_count:
                raise ValueError("ordered_concatenate requires the same position count in every row")
        target = row["target"]
        if task_type == "classification":
            if isinstance(target, bool) or not isinstance(target, (int, str)):
                raise ValueError("classification targets must be integer or string scalars")
        elif task_type == "regression":
            if isinstance(target, bool) or not isinstance(target, (int, float)):
                raise ValueError("regression targets must be numeric scalars")
            target = float(target)
            if not np.isfinite(target):
                raise ValueError("regression targets must be finite")
        else:
            raise ValueError("task_type must be classification or regression")
        rows[index] = {
            **row,
            "sequence_sha256": sequence_hash,
            "residue_positions": positions,
            "target": target,
        }
        seen_ids.add(row["row_id"])
        covered_sequences.add(sequence_hash)
        targets.append(target)
    if len(rows) < minimum_samples:
        raise ValueError(
            f"P0-8 task requires at least {minimum_samples} annotated rows"
        )
    if covered_sequences != cohort["sequence_hashes"]:
        raise ValueError("task annotations must cover every evaluation-cohort sequence")
    if len(set(targets)) < 2:
        raise ValueError("task targets must vary")
    row_order = [
        {
            "row_id": row["row_id"],
            "sequence_sha256": row["sequence_sha256"],
            "identity_group": row["identity_group"],
            "residue_positions": row["residue_positions"],
        }
        for row in rows
    ]
    return rows, {
        "path": path,
        "sha256": digest,
        "n_rows": len(rows),
        "row_order_sha256": _canonical_digest(row_order),
    }


def _validate_identity_leakage(
    descriptor: Any,
    base: Path,
    evaluation: Mapping[str, Any],
    prior: Mapping[str, Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    descriptor = _require_keys(
        descriptor,
        {"path", "sha256", "clustering_receipt"},
        set(),
        "identity_assignments",
    )
    path, digest = _load_descriptor(
        {"path": descriptor["path"], "sha256": descriptor["sha256"]},
        base,
        "identity_assignments",
    )
    expected_role_by_hash: dict[str, str] = {}
    for role, cohort in {**prior, "p0_8_evaluation": evaluation}.items():
        for sequence_hash in cohort["sequence_hashes"]:
            if sequence_hash in expected_role_by_hash:
                raise ValueError("exact sequence overlap between P0-2 and P0-8 cohorts")
            expected_role_by_hash[sequence_hash] = role
    assignments: dict[str, tuple[str, str]] = {}
    for row in load_jsonl(path):
        if not isinstance(row, dict) or set(row) != IDENTITY_FIELDS:
            raise ValueError("identity assignment rows require exact schema")
        sequence_hash = _require_sha256(row["sequence_sha256"], "identity sequence hash")
        if sequence_hash in assignments:
            raise ValueError("duplicate identity assignment")
        if not isinstance(row["identity_group"], str) or not row["identity_group"]:
            raise ValueError("identity assignment group must be non-empty")
        if row["cohort_role"] not in {*PRIOR_ROLES, "p0_8_evaluation"}:
            raise ValueError("unknown identity-assignment cohort role")
        assignments[sequence_hash] = (row["identity_group"], row["cohort_role"])
    if set(assignments) != set(expected_role_by_hash):
        raise ValueError("identity assignments must exactly cover all bound cohorts")
    for sequence_hash, role in expected_role_by_hash.items():
        if assignments[sequence_hash][1] != role:
            raise ValueError("identity assignment cohort-role mismatch")
    annotation_groups: dict[str, str] = {}
    for row in annotations:
        existing = annotation_groups.setdefault(row["sequence_sha256"], row["identity_group"])
        if existing != row["identity_group"]:
            raise ValueError("one sequence cannot have multiple identity groups")
    for sequence_hash, group in annotation_groups.items():
        if assignments[sequence_hash][0] != group:
            raise ValueError("annotation/identity-assignment group mismatch")
    prior_groups = {
        group
        for sequence_hash, (group, role) in assignments.items()
        if role in PRIOR_ROLES and sequence_hash in expected_role_by_hash
    }
    evaluation_groups = {
        group for group, role in assignments.values() if role == "p0_8_evaluation"
    }
    overlap = prior_groups & evaluation_groups
    if overlap:
        raise ValueError(f"identity-group leakage between P0-2 and P0-8 cohorts: {len(overlap)} groups")

    clustering_path, clustering_sha = _load_descriptor(
        descriptor["clustering_receipt"], base, "identity clustering receipt"
    )
    clustering = load_json(clustering_path)
    clustering_fields = {
        "schema_version",
        "status",
        "assignment_sha256",
        "input_cohort_sha256_by_role",
        "algorithm",
    }
    if not isinstance(clustering, dict) or set(clustering) != clustering_fields:
        raise ValueError("identity clustering receipt fields differ")
    expected_inputs = {
        **{role: cohort["sha256"] for role, cohort in prior.items()},
        "p0_8_evaluation": evaluation["sha256"],
    }
    if (
        clustering["schema_version"] != "r2_p0_8_identity_clustering_receipt_v1"
        or clustering["status"] != "verified_complete"
        or clustering["assignment_sha256"] != digest
        or clustering["input_cohort_sha256_by_role"] != expected_inputs
    ):
        raise ValueError("identity clustering receipt is not bound to the supplied cohorts")
    algorithm = _require_keys(
        clustering["algorithm"],
        {
            "name",
            "version",
            "sequence_identity_threshold",
            "coverage_threshold",
            "command",
            "executable",
        },
        set(),
        "identity clustering algorithm",
    )
    for field in ("name", "version"):
        if not isinstance(algorithm[field], str) or not algorithm[field].strip():
            raise ValueError(f"identity clustering algorithm {field} must be non-empty")
    for field in ("sequence_identity_threshold", "coverage_threshold"):
        value = algorithm[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < float(value) <= 1:
            raise ValueError(f"identity clustering {field} must lie in (0, 1]")
    command = algorithm["command"]
    if not isinstance(command, list) or not command or any(
        not isinstance(part, str) or not part for part in command
    ):
        raise ValueError("identity clustering command must be a non-empty string list")
    executable_path, executable_sha = _load_descriptor(
        algorithm["executable"], clustering_path.parent, "identity clustering executable"
    )
    return {
        "path": path,
        "sha256": digest,
        "n_assignments": len(assignments),
        "n_evaluation_groups": len(evaluation_groups),
        "p0_2_p0_8_sequence_overlap": 0,
        "p0_2_p0_8_identity_group_overlap": 0,
        "clustering_receipt": {
            "path": str(clustering_path),
            "sha256": clustering_sha,
            "algorithm": {
                **algorithm,
                "executable": {
                    "path": str(executable_path),
                    "sha256": executable_sha,
                },
            },
        },
    }


def _checkpoint_pairs(dictionaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "seed": item["seed"],
            "checkpoint_sha256": item["checkpoint_sha256"],
            "run_manifest_sha256": item["run_manifest_sha256"],
        }
        for item in sorted(dictionaries, key=lambda item: item["seed"])
    ]


def _load_p0_2_receipt(
    descriptor: Any,
    base: Path,
    *,
    model_name: str,
    prior: Mapping[str, Mapping[str, Any]],
    dictionaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    path, digest = _load_descriptor(descriptor, base, "p0_2_gate_receipt")
    from .dictionary_gate import require_eligible_model_method

    expected_cohorts = {
        role.removeprefix("p0_2_"): cohort["sha256"]
        for role, cohort in prior.items()
    }
    selected = require_eligible_model_method(
        path,
        digest,
        model_name=model_name,
        method="topk_clt",
        expected_run_manifest_sha256_by_seed={
            item["seed"]: item["run_manifest_sha256"] for item in dictionaries
        },
        expected_checkpoint_sha256_by_seed={
            item["seed"]: item["checkpoint_sha256"] for item in dictionaries
        },
        expected_source_manifest_sha256_by_split=expected_cohorts,
        requested_layers=list(dictionaries[0]["requested_layers"]),
    )
    return {"path": path, "sha256": digest, "selected": selected}


def _validate_model(spec: Any, base: Path, *, production: bool) -> dict[str, Any]:
    spec = _require_keys(
        spec,
        {
            "name",
            "model_root",
            "model_artifacts",
            "layers",
            "input_format",
            "model_inference_dtype",
            "model_inference_dtype_verification",
            "activation_finiteness_check",
        },
        set(),
        "model",
    )
    if not isinstance(spec["name"], str) or not spec["name"]:
        raise ValueError("model.name must be non-empty")
    model_root = _resolve(spec["model_root"], base)
    if not isinstance(spec["model_artifacts"], dict) or set(spec["model_artifacts"]) != MODEL_DIGEST_FIELDS:
        raise ValueError("model_artifacts must contain exact config/weights/tokenizer digests")
    for field, value in spec["model_artifacts"].items():
        _require_sha256(value, field)
    layers = spec["layers"]
    if (
        not isinstance(layers, list)
        or not layers
        or any(type(layer) is not int or layer < 0 for layer in layers)
        or layers != sorted(set(layers))
    ):
        raise ValueError("model.layers must be sorted unique non-negative integers")
    if spec["input_format"] not in {"sequence", "zymctrl_ec"}:
        raise ValueError("unsupported model input_format")
    if spec["model_inference_dtype"] not in {"float16", "bfloat16", "float32"}:
        raise ValueError("unsupported model dtype")
    if (
        spec["model_inference_dtype_verification"]
        != MODEL_INFERENCE_DTYPE_VERIFICATION
        or spec["activation_finiteness_check"] != ACTIVATION_FINITE_CHECK
    ):
        raise ValueError("frozen inference verification methods differ")
    if production and spec["model_inference_dtype"] != "bfloat16":
        raise ValueError("production model inference must use bfloat16")
    observed = verify_model_artifacts(model_root, spec["model_artifacts"])
    return {**spec, "model_root": model_root, "model_artifacts": observed}


def _validate_dictionary_descriptors(
    records: Any,
    base: Path,
    model: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(records, list) or len(records) < 3:
        raise ValueError("at least three independently seeded dictionaries are required")
    output: list[dict[str, Any]] = []
    seeds: set[int] = set()
    checkpoints: set[str] = set()
    for index, record in enumerate(records):
        record = _require_keys(
            record,
            {"seed", "checkpoint", "checkpoint_sha256", "run_manifest_sha256"},
            set(),
            f"dictionary {index}",
        )
        seed = record["seed"]
        if type(seed) is not int or seed < 0 or seed in seeds:
            raise ValueError("dictionary seeds must be unique non-negative integers")
        checkpoint = _resolve(record["checkpoint"], base)
        checkpoint_hash = _require_sha256(
            record["checkpoint_sha256"], "checkpoint_sha256"
        )
        run_manifest_hash = _require_sha256(
            record["run_manifest_sha256"], "run_manifest_sha256"
        )
        if checkpoint_hash in checkpoints:
            raise ValueError("dictionary checkpoints must be distinct")
        if not checkpoint.is_file() or sha256_file(checkpoint) != checkpoint_hash:
            raise ValueError(f"exact-cache best.pt SHA-256 mismatch for seed {seed}")
        output.append(
            {
                "seed": seed,
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_hash,
                "run_manifest_sha256": run_manifest_hash,
                "requested_layers": list(model["layers"]),
            }
        )
        seeds.add(seed)
        checkpoints.add(checkpoint_hash)
    return sorted(output, key=lambda item: item["seed"])


def _load_dictionaries(
    records: Sequence[Mapping[str, Any]],
    *,
    device: str,
    receipt_path: Path,
    receipt_sha256: str,
    model_name: str,
    source_manifest_sha256_by_split: Mapping[str, str],
    requested_layers: Sequence[int],
    dictionary_loader: Callable[..., tuple[torch.nn.Module, dict[str, Any]]] | None,
) -> tuple[list[dict[str, Any]], list[torch.nn.Module], dict[str, int]]:
    """Load only exact-cache checkpoints already authorized by the P0-2 gate."""

    if dictionary_loader is None:
        from .dictionary_gate import load_eligible_topk_clt

        dictionary_loader = load_eligible_topk_clt

    output: list[dict[str, Any]] = []
    modules: list[torch.nn.Module] = []
    geometry: dict[str, int] | None = None
    run_hashes = {item["seed"]: item["run_manifest_sha256"] for item in records}
    checkpoint_hashes = {item["seed"]: item["checkpoint_sha256"] for item in records}
    for record in records:
        seed = record["seed"]
        checkpoint = record["checkpoint"]
        clt, provenance = dictionary_loader(
            receipt_path,
            receipt_sha256,
            model_name=model_name,
            run_seed=seed,
            checkpoint_path=checkpoint,
            expected_run_manifest_sha256_by_seed=run_hashes,
            expected_checkpoint_sha256_by_seed=checkpoint_hashes,
            expected_source_manifest_sha256_by_split=(
                source_manifest_sha256_by_split
            ),
            requested_layers=requested_layers,
            map_location=device,
        )
        expected_provenance = {
            "model_name": model_name,
            "method": "topk_clt",
            "run_seed": seed,
            "eligibility_receipt_sha256": receipt_sha256,
            "run_manifest_sha256": record["run_manifest_sha256"],
            "checkpoint_sha256": record["checkpoint_sha256"],
        }
        if any(provenance.get(key) != value for key, value in expected_provenance.items()):
            raise ValueError("eligible TopK loader returned mismatched provenance")
        current = dict(provenance.get("geometry", {}))
        if set(current) != {"n_layers", "d_model", "d_clt", "k", "window"}:
            raise ValueError("eligible TopK loader returned invalid geometry")
        if geometry is None:
            geometry = current
        elif current != geometry:
            raise ValueError("all P0-8 dictionaries must have identical geometry")
        if any(layer >= clt.n_layers for layer in requested_layers):
            raise ValueError("requested representation layer exceeds dictionary depth")
        output.append(
            {
                "seed": seed,
                "checkpoint": checkpoint,
                "checkpoint_sha256": record["checkpoint_sha256"],
                "run_manifest_sha256": record["run_manifest_sha256"],
                "checkpoint_step": provenance["checkpoint_step"],
                "candidate_id": provenance["candidate_id"],
                "requested_layers": list(record["requested_layers"]),
            }
        )
        modules.append(clt)
    assert geometry is not None
    return output, modules, geometry


def _default_model_loader(model: Mapping[str, Any], device: str):
    from src.models.model_loader import load_model

    return load_model(
        str(model["model_root"]),
        device=device,
        dtype=getattr(torch, model["model_inference_dtype"]),
    )


def _pool(values: torch.Tensor, positions: Sequence[int], method: str) -> np.ndarray:
    selected = values.index_select(
        0, torch.as_tensor(positions, dtype=torch.long, device=values.device)
    )
    if method == "mean_selected_residues":
        pooled = selected.mean(dim=0)
    elif method == "ordered_concatenate":
        pooled = selected.reshape(-1)
    else:
        raise ValueError(f"unsupported representation pooling: {method}")
    return pooled.detach().float().cpu().numpy().astype(np.float32)


@torch.no_grad()
def _extract_representations(
    model_spec: Mapping[str, Any],
    protein_model: Any,
    dictionaries: Sequence[Mapping[str, Any]],
    clts: Sequence[torch.nn.Module],
    cohort: Mapping[str, Any],
    extraction_rows: Sequence[Mapping[str, Any]],
    *,
    device: str,
    max_model_tokens: int,
    pooling: str,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, dict[str, Any]]:
    """Extract label-blind rows; task targets/groups are intentionally absent."""

    geometry = (clts[0].n_layers, clts[0].d_model)
    if (protein_model.n_layers, protein_model.d_model) != geometry:
        raise ValueError("local model/dictionary geometry mismatch")
    rows_by_sequence: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for index, row in enumerate(extraction_rows):
        if set(row) != {"row_id", "sequence_sha256", "residue_positions"}:
            raise ValueError("extractor rows must exclude labels and groups")
        rows_by_sequence.setdefault(row["sequence_sha256"], []).append((index, row))
    n_rows = len(extraction_rows)
    layers = [str(layer) for layer in model_spec["layers"]]
    names = ["clt_input", "mlp_output"]
    for dictionary in dictionaries:
        names.extend(
            [
                f"code_seed_{dictionary['seed']}",
                f"reconstruction_seed_{dictionary['seed']}",
            ]
        )
    buffers: dict[str, dict[str, list[np.ndarray | None]]] = {
        name: {layer: [None] * n_rows for layer in layers} for name in names
    }
    error_components = np.full(
        (n_rows, len(dictionaries), len(model_spec["layers"])), np.nan, dtype=np.float64
    )
    from src.models.model_loader import assert_finite_captured_activations

    activation_capture_count = 0
    for record in cohort["rows"]:
        indexed_rows = rows_by_sequence.get(record["sha256"], [])
        if not indexed_rows:
            raise ValueError("every evaluation sequence must have extraction rows")
        text = format_model_input(record, model_spec["input_format"])
        encoded = protein_model.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        if input_ids.shape != attention_mask.shape or input_ids.shape[0] != 1:
            raise ValueError("single-sequence tokenization returned invalid shapes")
        if input_ids.shape[1] > max_model_tokens:
            raise ValueError(f"model token limit exceeded by {record['id']}")
        token_map = residue_token_indices(
            protein_model.tokenizer, input_ids, record["sequence"]
        )
        token_indices = torch.as_tensor(token_map, dtype=torch.long, device=device)
        cache = protein_model.get_activations(input_ids, attention_mask)
        assert_finite_captured_activations(cache)
        activation_capture_count += 1
        raw_layers = [value[0].index_select(0, token_indices).float() for value in cache.clt_input]
        target_layers = [value[0].index_select(0, token_indices).float() for value in cache.mlp_out]
        seed_features: list[list[torch.Tensor]] = []
        seed_reconstructions: list[list[torch.Tensor]] = []
        for clt in clts:
            full_inputs = [
                cache.clt_input[layer].float() for layer in range(clt.n_layers)
            ]
            full_features = clt.encode(full_inputs)
            full_reconstructions = clt.decode(full_features)
            seed_features.append(
                [value[0].index_select(0, token_indices) for value in full_features]
            )
            seed_reconstructions.append(
                [
                    value[0].index_select(0, token_indices).float()
                    for value in full_reconstructions
                ]
            )
        for row_index, row in indexed_rows:
            positions = row["residue_positions"]
            for layer_offset, layer in enumerate(model_spec["layers"]):
                layer_name = str(layer)
                buffers["clt_input"][layer_name][row_index] = _pool(raw_layers[layer], positions, pooling)
                buffers["mlp_output"][layer_name][row_index] = _pool(target_layers[layer], positions, pooling)
                for dictionary_index, dictionary in enumerate(dictionaries):
                    code_name = f"code_seed_{dictionary['seed']}"
                    reconstruction_name = f"reconstruction_seed_{dictionary['seed']}"
                    buffers[code_name][layer_name][row_index] = _pool(
                        seed_features[dictionary_index][layer], positions, pooling
                    )
                    predicted = seed_reconstructions[dictionary_index][layer]
                    buffers[reconstruction_name][layer_name][row_index] = _pool(
                        predicted, positions, pooling
                    )
                    selected_residual = (predicted - target_layers[layer]).index_select(
                        0,
                        torch.as_tensor(
                            positions, dtype=torch.long, device=predicted.device
                        ),
                    )
                    error_components[row_index, dictionary_index, layer_offset] = float(
                        torch.linalg.vector_norm(selected_residual, dim=1).mean().item()
                    )
    representations: dict[str, dict[str, np.ndarray]] = {}
    for name, layer_buffers in buffers.items():
        representations[name] = {}
        for layer, values in layer_buffers.items():
            if any(value is None for value in values):
                raise RuntimeError(f"incomplete extracted representation {name}:{layer}")
            representations[name][layer] = np.stack(values).astype(np.float32)
    if activation_capture_count == 0:
        raise RuntimeError("no frozen-model activations were verified")
    return representations, error_components, {
        "activation_finiteness_check": ACTIVATION_FINITE_CHECK,
        "activation_finiteness_verified": True,
    }


def _load_intervention_evidence(
    descriptor: Any,
    base: Path,
    *,
    task_id: str,
    cohort_sha256: str,
    annotation_sha256: str,
    row_order_sha256: str,
    model_artifacts: Mapping[str, str],
    dictionaries: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    receipt_path, receipt_sha = _load_descriptor(
        descriptor, base, "intervention_evidence"
    )
    receipt = load_json(receipt_path)
    required = {
        "schema_version",
        "status",
        "task_id",
        "cohort_sha256",
        "annotation_sha256",
        "row_order_sha256",
        "model_artifacts",
        "dictionary_checkpoints",
        "quality_inventory",
        "effect_definition",
        "test_evaluation_count",
        "freeze_manifest",
        "artifact",
    }
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise ValueError("intervention-evidence receipt fields differ")
    if (
        receipt["schema_version"] != "r2_p0_8_intervention_evidence_receipt_v3"
        or receipt["status"] != "verified_complete"
        or receipt["task_id"] != task_id
        or receipt["cohort_sha256"] != cohort_sha256
        or receipt["annotation_sha256"] != annotation_sha256
        or receipt["row_order_sha256"] != row_order_sha256
        or receipt["model_artifacts"] != dict(model_artifacts)
        or receipt["dictionary_checkpoints"] != _checkpoint_pairs(dictionaries)
        or receipt["test_evaluation_count"] != 1
    ):
        raise ValueError("intervention evidence is not bound to this frozen input")
    expected_inventory = {
        "dictionary_seeds": [item["seed"] for item in dictionaries],
        "layers": list(dictionaries[0]["requested_layers"]),
    }
    if receipt["quality_inventory"] != expected_inventory:
        raise ValueError("intervention quality seed/layer inventory differs")
    if not isinstance(receipt["effect_definition"], str) or not receipt["effect_definition"].strip():
        raise ValueError("intervention evidence requires an effect definition")
    freeze_path, freeze_sha = _load_descriptor(
        receipt["freeze_manifest"], receipt_path.parent, "intervention freeze manifest"
    )
    freeze = load_json(freeze_path)
    freeze_fields = {
        "schema_version",
        "status",
        "task_id",
        "cohort_sha256",
        "annotation_sha256",
        "row_order_sha256",
        "model_artifacts",
        "dictionary_checkpoints",
        "quality_inventory",
        "effect_definition",
        "producer",
    }
    if not isinstance(freeze, dict) or set(freeze) != freeze_fields:
        raise ValueError("intervention freeze-manifest fields differ")
    if (
        freeze["schema_version"] != "r2_p0_8_intervention_freeze_manifest_v2"
        or freeze["status"] != "frozen_before_test_evaluation"
        or freeze["task_id"] != task_id
        or freeze["cohort_sha256"] != cohort_sha256
        or freeze["annotation_sha256"] != annotation_sha256
        or freeze["row_order_sha256"] != row_order_sha256
        or freeze["model_artifacts"] != dict(model_artifacts)
        or freeze["dictionary_checkpoints"] != _checkpoint_pairs(dictionaries)
        or freeze["quality_inventory"] != expected_inventory
        or freeze["effect_definition"] != receipt["effect_definition"]
    ):
        raise ValueError("intervention freeze manifest is not bound to this frozen input")
    producer = _require_keys(
        freeze["producer"], {"source", "command", "environment"}, set(), "intervention producer"
    )
    if not isinstance(producer["command"], list) or not producer["command"] or any(
        not isinstance(part, str) or not part for part in producer["command"]
    ):
        raise ValueError("intervention producer command must be a non-empty string list")
    if not isinstance(producer["environment"], dict) or not producer["environment"]:
        raise ValueError("intervention producer environment must be a non-empty object")
    producer_path, producer_sha = _load_descriptor(
        producer["source"], freeze_path.parent, "intervention producer source"
    )
    artifact = _require_keys(receipt["artifact"], {"path", "sha256"}, set(), "intervention artifact")
    artifact_path = _resolve(artifact["path"], receipt_path.parent)
    artifact_sha = _require_sha256(artifact["sha256"], "intervention artifact sha256")
    if not artifact_path.is_file() or sha256_file(artifact_path) != artifact_sha:
        raise ValueError("intervention artifact path or SHA-256 mismatch")
    rows = load_jsonl(artifact_path)
    if len(rows) != len(annotations):
        raise ValueError("intervention evidence row count mismatch")
    values = np.full(
        (
            len(annotations),
            len(expected_inventory["dictionary_seeds"]),
            len(expected_inventory["layers"]),
        ),
        np.nan,
        dtype=np.float64,
    )
    for row_index, (expected, observed) in enumerate(
        zip(annotations, rows, strict=True)
    ):
        if not isinstance(observed, dict) or set(observed) != {
            "row_id",
            "sequence_sha256",
            "intervention_effect_by_seed_layer",
        }:
            raise ValueError("intervention artifact row fields differ")
        effects = observed["intervention_effect_by_seed_layer"]
        if (
            observed["row_id"] != expected["row_id"]
            or observed["sequence_sha256"] != expected["sequence_sha256"]
            or not isinstance(effects, dict)
            or set(effects)
            != {str(seed) for seed in expected_inventory["dictionary_seeds"]}
        ):
            raise ValueError("intervention evidence row-order/value mismatch")
        for seed_index, seed in enumerate(expected_inventory["dictionary_seeds"]):
            layer_values = effects[str(seed)]
            if (
                not isinstance(layer_values, dict)
                or set(layer_values)
                != {str(layer) for layer in expected_inventory["layers"]}
            ):
                raise ValueError("intervention quality layer inventory differs")
            for layer_index, layer in enumerate(expected_inventory["layers"]):
                value = layer_values[str(layer)]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value)
                ):
                    raise ValueError("intervention evidence contains a non-finite value")
                values[row_index, seed_index, layer_index] = float(value)
    if not np.isfinite(values).all():
        raise RuntimeError("intervention evidence matrix was not fully populated")
    return values, {
        "receipt_path": receipt_path,
        "receipt_sha256": receipt_sha,
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_sha,
        "effect_definition": receipt["effect_definition"],
        "quality_inventory": expected_inventory,
        "freeze_manifest_path": str(freeze_path),
        "freeze_manifest_sha256": freeze_sha,
        "producer_source_path": str(producer_path),
        "producer_source_sha256": producer_sha,
        "producer_command": producer["command"],
        "producer_environment": producer["environment"],
    }


def _save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_analysis(value: Any, n_groups: int) -> dict[str, Any]:
    required = {
        "analysis_seeds",
        "outer_splits",
        "inner_splits",
        "n_bootstrap",
        "comparison_dimension",
        "active_width_dimension",
        "controls",
    }
    value = _require_keys(value, required, set(), "analysis")
    seeds = value["analysis_seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("analysis_seeds must be unique integers")
    for field in ("outer_splits", "inner_splits", "n_bootstrap", "comparison_dimension"):
        if type(value[field]) is not int or value[field] < (1 if field == "n_bootstrap" else 2):
            raise ValueError(f"invalid analysis parameter: {field}")
    if n_groups < value["outer_splits"]:
        raise ValueError("outer_splits exceeds the number of identity groups")
    active = value["active_width_dimension"]
    if active is not None and (
        type(active) is not int or active < 2 or active >= value["comparison_dimension"]
    ):
        raise ValueError("active_width_dimension must be null or smaller than comparison_dimension")
    allowed = {"pca", "random_projection", "nmf", "ica", "random_dictionary"}
    controls = value["controls"]
    if not isinstance(controls, list) or len(controls) != len(set(controls)) or not set(controls) <= allowed:
        raise ValueError("controls must be unique supported names")
    return dict(value)


def build_recoverability_inputs(
    spec_path: Path,
    output_dir: Path,
    *,
    model_loader: Callable[[Mapping[str, Any], str], Any] | None = None,
    dictionary_loader: Callable[..., tuple[torch.nn.Module, dict[str, Any]]]
    | None = None,
    extractor: Callable[
        ...,
        tuple[dict[str, dict[str, np.ndarray]], np.ndarray, dict[str, Any]],
    ]
    | None = None,
) -> Path:
    """Build one atomic P0-8 input directory and return its receipt path."""

    spec_path = Path(spec_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    spec = load_json(spec_path)
    spec = _require_keys(
        spec,
        {
            "schema_version",
            "mode",
            "device",
            "max_model_tokens",
            "task",
            "prior_p0_2_cohorts",
            "identity_assignments",
            "model",
            "dictionaries",
            "p0_2_gate_receipt",
            "intervention_evidence",
            "analysis",
        },
        set(),
        "P0-8 builder spec",
    )
    if spec["schema_version"] != 3 or spec["mode"] not in {"production", "test_fixture"}:
        raise ValueError("builder requires schema_version 3 and a supported mode")
    if (
        any(value is not None for value in (model_loader, dictionary_loader, extractor))
        and spec["mode"] != "test_fixture"
    ):
        raise ValueError("test loader/extractor overrides are forbidden in production mode")
    device = spec["device"]
    max_model_tokens = spec["max_model_tokens"]
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string")
    if type(max_model_tokens) is not int or max_model_tokens < 2:
        raise ValueError("max_model_tokens must be an integer >=2")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    base = spec_path.parent
    task = _require_keys(
        spec["task"],
        {
            "task_id",
            "task_type",
            "pooling",
            "minimum_samples",
            "cohort",
            "annotations",
        },
        set(),
        "task",
    )
    if not isinstance(task["task_id"], str) or not task["task_id"]:
        raise ValueError("task_id must be non-empty")
    if task["task_type"] not in {"classification", "regression"}:
        raise ValueError("unsupported task_type")
    if task["pooling"] not in {"mean_selected_residues", "ordered_concatenate"}:
        raise ValueError("unsupported pooling")
    minimum_samples = task["minimum_samples"]
    if type(minimum_samples) is not int or minimum_samples < 8:
        raise ValueError("task.minimum_samples must be an integer of at least eight")
    if spec["mode"] == "production" and minimum_samples < PRODUCTION_MINIMUM_SAMPLES:
        raise ValueError(
            "production P0-8 requires the frozen enlarged-cohort minimum of "
            f"{PRODUCTION_MINIMUM_SAMPLES} annotated rows"
        )
    evaluation = _load_cohort(task["cohort"], base, "p0_8_evaluation")
    annotations, annotation_provenance = _load_task_annotations(
        task["annotations"],
        base,
        evaluation,
        task_type=task["task_type"],
        pooling=task["pooling"],
        minimum_samples=minimum_samples,
    )
    prior_records = spec["prior_p0_2_cohorts"]
    if not isinstance(prior_records, list) or len(prior_records) != len(PRIOR_ROLES):
        raise ValueError("prior_p0_2_cohorts must contain train, validation, and test")
    prior: dict[str, dict[str, Any]] = {}
    for record in prior_records:
        if not isinstance(record, dict) or record.get("role") not in PRIOR_ROLES:
            raise ValueError("unknown prior P0-2 cohort role")
        role = record["role"]
        if role in prior:
            raise ValueError("duplicate prior P0-2 cohort role")
        prior[role] = _load_cohort(record, base, role)
    if set(prior) != set(PRIOR_ROLES):
        raise ValueError("all prior P0-2 cohort roles are required")
    identity_provenance = _validate_identity_leakage(
        spec["identity_assignments"], base, evaluation, prior, annotations
    )
    model = _validate_model(
        spec["model"], base, production=spec["mode"] == "production"
    )
    dictionary_descriptors = _validate_dictionary_descriptors(
        spec["dictionaries"], base, model
    )
    p0_2 = _load_p0_2_receipt(
        spec["p0_2_gate_receipt"],
        base,
        model_name=model["name"],
        prior=prior,
        dictionaries=dictionary_descriptors,
    )
    source_hashes = {
        role.removeprefix("p0_2_"): cohort["sha256"]
        for role, cohort in prior.items()
    }
    dictionaries, clts, geometry = _load_dictionaries(
        dictionary_descriptors,
        device=device,
        receipt_path=p0_2["path"],
        receipt_sha256=p0_2["sha256"],
        model_name=model["name"],
        source_manifest_sha256_by_split=source_hashes,
        requested_layers=model["layers"],
        dictionary_loader=dictionary_loader,
    )
    groups = np.asarray([row["identity_group"] for row in annotations])
    analysis = _validate_analysis(spec["analysis"], len(np.unique(groups)))
    extraction_rows = [
        {
            "row_id": row["row_id"],
            "sequence_sha256": row["sequence_sha256"],
            "residue_positions": row["residue_positions"],
        }
        for row in annotations
    ]
    loader = model_loader or _default_model_loader
    protein_model = loader(model, device)
    from src.models.model_loader import verify_frozen_model_inference_dtype

    inference_provenance = verify_frozen_model_inference_dtype(
        protein_model, model["model_inference_dtype"]
    )
    extraction_function = extractor or _extract_representations
    representations, error_components, finiteness_provenance = extraction_function(
        model,
        protein_model,
        dictionaries,
        clts,
        evaluation,
        extraction_rows,
        device=device,
        max_model_tokens=max_model_tokens,
        pooling=task["pooling"],
    )
    expected_finiteness = {
        "activation_finiteness_check": ACTIVATION_FINITE_CHECK,
        "activation_finiteness_verified": True,
    }
    if finiteness_provenance != expected_finiteness:
        raise ValueError("extractor did not verify captured activation finiteness")
    expected_names = {"clt_input", "mlp_output"} | {
        name
        for dictionary in dictionaries
        for name in (
            f"code_seed_{dictionary['seed']}",
            f"reconstruction_seed_{dictionary['seed']}",
        )
    }
    expected_layers = {str(layer) for layer in model["layers"]}
    if set(representations) != expected_names:
        raise ValueError("extractor representation inventory mismatch")
    maximum_dimension = max(
        analysis["comparison_dimension"], analysis["active_width_dimension"] or 0
    )
    for name, layers in representations.items():
        if set(layers) != expected_layers:
            raise ValueError(f"extractor layer inventory mismatch for {name}")
        for layer, matrix in layers.items():
            matrix = np.asarray(matrix)
            if (
                matrix.ndim != 2
                or matrix.shape[0] != len(annotations)
                or matrix.shape[1] < maximum_dimension
                or not np.issubdtype(matrix.dtype, np.floating)
                or not np.isfinite(matrix).all()
            ):
                raise ValueError(f"invalid or dimension-unmatched representation {name}:{layer}")
            layers[layer] = matrix.astype(np.float32, copy=False)
    error_components = np.asarray(error_components, dtype=np.float64)
    expected_error_shape = (
        len(annotations),
        len(dictionaries),
        len(model["layers"]),
    )
    if error_components.shape != expected_error_shape or not np.isfinite(error_components).all() or np.any(error_components < 0):
        raise ValueError("reconstruction-error components are missing or invalid")
    intervention_effect, intervention_provenance = _load_intervention_evidence(
        spec["intervention_evidence"],
        base,
        task_id=task["task_id"],
        cohort_sha256=evaluation["sha256"],
        annotation_sha256=annotation_provenance["sha256"],
        row_order_sha256=annotation_provenance["row_order_sha256"],
        model_artifacts=model["model_artifacts"],
        dictionaries=dictionaries,
        annotations=annotations,
    )
    spec_hash = sha256_file(spec_path)
    staging = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"stale builder staging directory: {staging}")
    staging.mkdir(parents=True)
    try:
        arrays: dict[str, np.ndarray] = {
            "y": np.asarray([row["target"] for row in annotations]),
            "groups": groups,
            "row_id": np.asarray([row["row_id"] for row in annotations]),
            "sequence_sha256": np.asarray([row["sequence_sha256"] for row in annotations]),
        }
        for dictionary_index, dictionary in enumerate(dictionaries):
            floor_name = f"code_seed_{dictionary['seed']}"
            for layer_index, layer in enumerate(model["layers"]):
                arrays[
                    f"quality__reconstruction_error__{floor_name}__{layer}"
                ] = error_components[:, dictionary_index, layer_index]
                arrays[
                    f"quality__intervention_effect__{floor_name}__{layer}"
                ] = intervention_effect[:, dictionary_index, layer_index]
        for name, layers in representations.items():
            for layer, matrix in layers.items():
                arrays[f"rep__{name}__{layer}"] = matrix
        npz_path = staging / "nested_recoverability_input.npz"
        _save_npz(npz_path, arrays)
        floor_names = [f"code_seed_{item['seed']}" for item in dictionaries]
        runner_spec = {
            "schema_version": "r2_p0_8_runner_spec_v2",
            "status": "verified_inputs_not_scientifically_adjudicated",
            "script": "scripts/55_run_nested_recoverability.py",
            "input": {"path": npz_path.name, "sha256": sha256_file(npz_path)},
            "arguments": {
                "task_type": task["task_type"],
                "ceiling": "clt_input",
                "floors": floor_names,
                **analysis,
            },
            "row_order_sha256": annotation_provenance["row_order_sha256"],
            "confirmatory_real": spec["mode"] == "production",
        }
        runner_spec_path = staging / "nested_recoverability_runner_spec.json"
        write_json(runner_spec_path, runner_spec)
        receipt = {
            "schema_version": "r2_p0_8_input_receipt_v3",
            "status": (
                "verified_production_inputs_not_scientifically_adjudicated"
                if spec["mode"] == "production"
                else "verified_test_fixture_inputs_only"
            ),
            "builder_spec": {"path": str(spec_path), "sha256": spec_hash},
            "task": {
                "task_id": task["task_id"],
                "task_type": task["task_type"],
                "pooling": task["pooling"],
                "minimum_samples": minimum_samples,
                "cohort": {
                    "cohort_id": evaluation["cohort_id"],
                    "path": str(evaluation["path"]),
                    "sha256": evaluation["sha256"],
                    "n_sequences": len(evaluation["rows"]),
                },
                "annotations": {
                    **annotation_provenance,
                    "path": str(annotation_provenance["path"]),
                },
                "identity_leakage_guard": {
                    **identity_provenance,
                    "path": str(identity_provenance["path"]),
                },
            },
            "prior_p0_2_cohorts": {
                role: {
                    "cohort_id": cohort["cohort_id"],
                    "path": str(cohort["path"]),
                    "sha256": cohort["sha256"],
                }
                for role, cohort in prior.items()
            },
            "p0_2_gate_receipt": {
                "path": str(p0_2["path"]),
                "sha256": p0_2["sha256"],
                "status": p0_2["selected"]["status"],
                "method": p0_2["selected"]["method"],
                "profile_sha256": p0_2["selected"]["profile_sha256"],
                "cache_manifest_sha256": p0_2["selected"][
                    "cache_manifest_sha256"
                ],
                "cache_content_sha256": p0_2["selected"][
                    "cache_content_sha256"
                ],
                "eligible_downstream_layers": p0_2["selected"][
                    "eligible_downstream_layers"
                ],
            },
            "model": {
                "name": model["name"],
                "root": str(model["model_root"]),
                "artifacts": model["model_artifacts"],
                "layers": model["layers"],
                **inference_provenance,
                **finiteness_provenance,
            },
            "dictionaries": [
                {**item, "checkpoint": str(item["checkpoint"])} for item in dictionaries
            ],
            "dictionary_geometry": geometry,
            "representation_extraction": {
                "source_path": str(Path(__file__).resolve()),
                "source_sha256": sha256_file(Path(__file__)),
                "mode": "built_in_label_blind" if extractor is None else "test_fixture_override",
                "labels_or_groups_passed_to_extractor": False,
                "representation_names": sorted(representations),
                "common_dimension": analysis["comparison_dimension"],
                "active_width_rank_sensitivity_dimension": analysis["active_width_dimension"],
            },
            "quality_evidence": {
                "reconstruction_error": {
                    "definition": "mean per-residue Euclidean MLP-output reconstruction residual within each declared dictionary seed and layer",
                    "component_shape": list(error_components.shape),
                    "derived_inside_builder": True,
                    "seed_layer_aggregation_before_analysis": False,
                },
                "intervention_effect": {
                    **intervention_provenance,
                    "receipt_path": str(intervention_provenance["receipt_path"]),
                    "artifact_path": str(intervention_provenance["artifact_path"]),
                    "seed_layer_aggregation_before_analysis": False,
                },
            },
            "outputs": {
                npz_path.name: sha256_file(npz_path),
                runner_spec_path.name: sha256_file(runner_spec_path),
            },
            "claim_boundary": (
                "This receipt establishes immutable input provenance only. It does not "
                "pass P0-8 or establish biological, causal, or conserved representations."
            ),
        }
        receipt_path = staging / "input_receipt.json"
        write_json(receipt_path, receipt)
        if sha256_file(spec_path) != spec_hash:
            raise RuntimeError("builder spec changed during extraction")
        for cohort in [evaluation, *prior.values()]:
            if sha256_file(cohort["path"]) != cohort["sha256"]:
                raise RuntimeError("cohort changed during extraction")
        if sha256_file(annotation_provenance["path"]) != annotation_provenance["sha256"]:
            raise RuntimeError("task annotations changed during extraction")
        if sha256_file(identity_provenance["path"]) != identity_provenance["sha256"]:
            raise RuntimeError("identity assignments changed during extraction")
        clustering_receipt = identity_provenance["clustering_receipt"]
        if sha256_file(clustering_receipt["path"]) != clustering_receipt["sha256"]:
            raise RuntimeError("identity clustering receipt changed during extraction")
        clustering_executable = clustering_receipt["algorithm"]["executable"]
        if (
            sha256_file(clustering_executable["path"])
            != clustering_executable["sha256"]
        ):
            raise RuntimeError("identity clustering executable changed during extraction")
        if sha256_file(p0_2["path"]) != p0_2["sha256"]:
            raise RuntimeError("P0-2 gate receipt changed during extraction")
        if sha256_file(intervention_provenance["receipt_path"]) != spec["intervention_evidence"]["sha256"]:
            raise RuntimeError("intervention receipt changed during extraction")
        if (
            sha256_file(intervention_provenance["artifact_path"])
            != intervention_provenance["artifact_sha256"]
        ):
            raise RuntimeError("intervention artifact changed during extraction")
        if (
            sha256_file(intervention_provenance["freeze_manifest_path"])
            != intervention_provenance["freeze_manifest_sha256"]
        ):
            raise RuntimeError("intervention freeze manifest changed during extraction")
        if (
            sha256_file(intervention_provenance["producer_source_path"])
            != intervention_provenance["producer_source_sha256"]
        ):
            raise RuntimeError("intervention producer source changed during extraction")
        if verify_model_artifacts(model["model_root"], model["model_artifacts"]) != model["model_artifacts"]:
            raise RuntimeError("model artifacts changed during extraction")
        for dictionary in dictionaries:
            if sha256_file(dictionary["checkpoint"]) != dictionary["checkpoint_sha256"]:
                raise RuntimeError("exact-cache dictionary checkpoint changed during extraction")
        staging.rename(output_dir)
        return output_dir / "input_receipt.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
