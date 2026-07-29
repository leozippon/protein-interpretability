"""Mask-clean activation caches and alternative dictionary quality controls.

The module deliberately separates activation extraction from dictionary
training. Frozen train/validation/test activations are written once after
invalid token rows are removed; every method then consumes the same immutable
rows and deterministic minibatch order.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import random
import resource
import shutil
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .io import sha256_file, write_json


CACHE_SCHEMA = "r2_dictionary_activation_cache_v1"
RESULT_SCHEMA = "r2_dictionary_control_results_v1"
SPLITS = ("train", "validation", "test")
METHODS = ("topk_clt", "relu_l1_sae", "gated_sae", "dense_low_rank")
QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
VOLATILE_CACHE_PROVENANCE_FIELDS = {
    "command",
    "pod_name",
    "node_name",
    "gpu_index",
    "gpu_model",
    "git_commit",
    "git_dirty",
    "python_version",
    "torch_version",
    "cuda_runtime",
    "started_at_utc",
    "finished_at_utc",
    "wall_time_seconds",
    "peak_accelerator_memory_allocated_bytes",
    "peak_accelerator_memory_reserved_bytes",
    "panel_capacity_check",
}


def format_model_input(record: Mapping[str, Any], model_input_format: str) -> str:
    """Reconstruct the exact model prompt from one frozen cohort row."""

    required = {"id", "source", "sequence", "split", "family", "sha256"}
    if set(record) != required:
        raise ValueError("cohort record fields do not match the frozen schema")
    sequence = record["sequence"]
    if hashlib.sha256(sequence.encode("utf-8")).hexdigest() != record["sha256"]:
        raise ValueError("cohort sequence SHA-256 mismatch")
    if model_input_format == "sequence":
        return sequence
    if model_input_format == "zymctrl_ec":
        return f"{record['family']}<sep><start>{sequence}<end>"
    raise ValueError(f"unknown model_input_format: {model_input_format}")


def token_priority(record_sha256: str, token_position: int) -> str:
    """Path-independent priority for one valid token row."""

    if len(record_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in record_sha256
    ):
        raise ValueError("record_sha256 must be a lowercase digest")
    if type(token_position) is not int or token_position < 0:
        raise ValueError("token_position must be a non-negative integer")
    return hashlib.sha256(
        f"{record_sha256}:{token_position}".encode("ascii")
    ).hexdigest()


def select_hash_priority_tokens(
    records: Iterable[tuple[str, int]],
    *,
    budget: int,
) -> list[dict[str, Any]]:
    """Select exactly the lowest-hash valid token keys with bounded memory.

    ``records`` supplies ``(sequence_sha256, valid_token_count)`` pairs after
    exact prompt reconstruction and tokenization. Input order and file paths do
    not affect the selected identities.
    """

    if budget < 1:
        raise ValueError("valid-token budget must be positive")
    heap: list[tuple[int, str, int, str]] = []
    seen_records: set[str] = set()
    eligible = 0
    for record_sha256, valid_token_count in records:
        if record_sha256 in seen_records:
            raise ValueError(
                f"duplicate record SHA-256 in token selector: {record_sha256}"
            )
        seen_records.add(record_sha256)
        if type(valid_token_count) is not int or valid_token_count < 1:
            raise ValueError("every record must have at least one valid model token")
        eligible += valid_token_count
        for position in range(valid_token_count):
            priority = token_priority(record_sha256, position)
            priority_int = int(priority, 16)
            entry = (-priority_int, record_sha256, position, priority)
            if len(heap) < budget:
                heapq.heappush(heap, entry)
            elif priority_int < -heap[0][0]:
                heapq.heapreplace(heap, entry)
    if eligible < budget:
        raise ValueError(
            f"valid-token budget {budget} cannot be met from {eligible} eligible rows"
        )
    selected = [
        {
            "record_sha256": record_sha256,
            "token_position": position,
            "priority_sha256": priority,
        }
        for _, record_sha256, position, priority in heap
    ]
    selected.sort(
        key=lambda row: (
            row["priority_sha256"],
            row["record_sha256"],
            row["token_position"],
        )
    )
    if len(selected) != budget:
        raise AssertionError("hash-priority selector returned the wrong budget")
    return selected


def estimate_activation_cache_bytes(
    *,
    valid_token_rows: int,
    n_layers: int,
    input_dim: int,
    target_dim: int,
    storage_dtype: str,
) -> int:
    if min(valid_token_rows, n_layers, input_dim, target_dim) < 1:
        raise ValueError("cache-size dimensions must be positive")
    if storage_dtype not in {"float16", "float32"}:
        raise ValueError("unsupported cache storage dtype")
    itemsize = np.dtype(storage_dtype).itemsize
    return valid_token_rows * n_layers * (input_dim + target_dim) * itemsize


def require_cache_free_space(
    destination: Path,
    *,
    estimated_bytes: int,
    safety_factor: float,
) -> dict[str, int | float]:
    if estimated_bytes < 1 or safety_factor < 1.0:
        raise ValueError("invalid cache free-space requirement")
    parent = Path(destination).resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(parent).free
    required_bytes = math.ceil(estimated_bytes * safety_factor)
    if free_bytes < required_bytes:
        raise OSError(
            f"insufficient cache space: require {required_bytes} bytes, have {free_bytes}"
        )
    return {
        "estimated_bytes": estimated_bytes,
        "safety_factor": safety_factor,
        "required_free_bytes": required_bytes,
        "observed_free_bytes": free_bytes,
    }


def _canonical_token_selection(
    selected_tokens_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    required_splits: Sequence[str] = SPLITS,
) -> dict[str, list[dict[str, Any]]]:
    split_names = tuple(required_splits)
    if not split_names or set(selected_tokens_by_split) != set(split_names):
        raise ValueError(f"selected_tokens_by_split must contain exactly {split_names}")
    canonical: dict[str, list[dict[str, Any]]] = {}
    record_split: dict[str, str] = {}
    for split in split_names:
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for row in selected_tokens_by_split[split]:
            if set(row) != {
                "record_sha256",
                "token_position",
                "priority_sha256",
            }:
                raise ValueError(f"invalid selected-token fields in {split}")
            record_sha256 = row["record_sha256"]
            owner = record_split.setdefault(record_sha256, split)
            if owner != split:
                raise ValueError(
                    f"selected record appears in both {owner} and {split}: "
                    f"{record_sha256}"
                )
            token_position = row["token_position"]
            priority = token_priority(record_sha256, token_position)
            if row["priority_sha256"] != priority:
                raise ValueError(f"selected-token priority mismatch in {split}")
            key = (record_sha256, token_position)
            if key in seen:
                raise ValueError(f"duplicate selected token in {split}: {key}")
            seen.add(key)
            rows.append(
                {
                    "record_sha256": record_sha256,
                    "token_position": token_position,
                    "priority_sha256": priority,
                }
            )
        if not rows:
            raise ValueError(f"selected-token budget is empty in {split}")
        rows.sort(
            key=lambda row: (
                row["priority_sha256"],
                row["record_sha256"],
                row["token_position"],
            )
        )
        canonical[split] = rows
    return canonical


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_strict_json(path: Path) -> Any:
    """Load JSON while rejecting duplicate keys and non-finite constants."""

    with Path(path).open(encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )


def load_strict_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {path}:{line_number}")
            try:
                rows.append(
                    json.loads(
                        line,
                        parse_constant=_reject_constant,
                        object_pairs_hook=_unique_object,
                    )
                )
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid JSONL row at {path}:{line_number}"
                ) from error
    if not rows:
        raise ValueError(f"empty JSONL file: {path}")
    return rows


def _derived_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def configure_determinism(seed: int) -> None:
    """Configure deterministic CPU/CUDA execution for one independent run."""

    seed = int(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def valid_token_rows(
    tensor: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return only valid token rows from a ``(batch, sequence, width)`` tensor."""

    if tensor.ndim != 3:
        raise ValueError("activation tensor must have shape (batch, sequence, width)")
    if attention_mask.shape != tensor.shape[:2]:
        raise ValueError(
            "attention_mask must match activation batch/sequence dimensions"
        )
    mask = attention_mask.to(device=tensor.device, dtype=torch.bool)
    if not mask.any():
        raise ValueError("attention_mask contains no valid tokens")
    rows = tensor.reshape(-1, tensor.shape[-1])[mask.reshape(-1)]
    if not torch.isfinite(rows).all():
        raise ValueError("valid activation rows contain non-finite values")
    return rows


def _validate_source_splits(
    source_splits: Mapping[str, Mapping[str, Any]],
    *,
    root: Path | None = None,
    verify_splits: Sequence[str] = SPLITS,
) -> dict:
    if set(source_splits) != set(SPLITS):
        raise ValueError(f"source_splits must contain exactly {SPLITS}")
    checked: dict[str, dict[str, str]] = {}
    verified = set(verify_splits)
    if not verified or not verified <= set(SPLITS):
        raise ValueError("verify_splits must be a non-empty subset of cache splits")
    for split in SPLITS:
        record = source_splits[split]
        if set(record) != {"manifest_path", "manifest_sha256"}:
            raise ValueError(
                f"source metadata for {split} must contain manifest_path and manifest_sha256"
            )
        path = str(record["manifest_path"])
        digest = str(record["manifest_sha256"])
        if (
            not path
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError(f"invalid source manifest metadata for {split}")
        local_path = Path(path)
        if root is not None and not local_path.is_absolute():
            local_path = root / local_path
        if split in verified:
            if not local_path.is_file():
                raise FileNotFoundError(
                    f"missing source manifest for {split}: {local_path}"
                )
            if sha256_file(local_path) != digest:
                raise ValueError(f"source manifest SHA-256 mismatch for {split}")
        checked[split] = {
            "manifest_path": str(local_path.resolve()),
            "manifest_sha256": digest,
        }
    return checked


def _cache_identity(
    *,
    objective: str,
    layers: Sequence[int],
    storage_dtype: str,
    sources: Mapping[str, Mapping[str, str]],
    dimensions: Mapping[int, tuple[int, int]],
    shards: Sequence[Mapping[str, Any]],
    activation_provenance: Mapping[str, Any],
    token_selection: Mapping[str, Any],
    layout: str,
) -> dict[str, Any]:
    """Return path- and timestamp-independent activation-cache identity fields."""

    return {
        "objective": objective,
        "selected_layers": list(layers),
        "storage_dtype": storage_dtype,
        "layout": layout,
        "source_manifest_sha256": {
            split: sources[split]["manifest_sha256"] for split in SPLITS
        },
        "dimensions": {
            str(layer): {"input": dimensions[layer][0], "target": dimensions[layer][1]}
            for layer in layers
        },
        "activation_provenance": {
            key: value
            for key, value in activation_provenance.items()
            if key not in VOLATILE_CACHE_PROVENANCE_FIELDS
        },
        "token_selection": token_selection,
        "shards": [
            {
                key: shard[key]
                for key in (
                    "split",
                    "layer",
                    "shard_index",
                    "rows",
                    "input_dim",
                    "target_dim",
                    "dtype",
                    "input_sha256",
                    "target_sha256",
                )
            }
            for shard in shards
        ],
    }


def _identity_sha256(identity: Mapping[str, Any]) -> str:
    payload = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_activation_cache(
    output_dir: Path,
    batches_by_split: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    selected_layers: Sequence[int],
    source_splits: Mapping[str, Mapping[str, Any]],
    activation_provenance: Mapping[str, Any] | None = None,
    objective: str = "autoencode",
    storage_dtype: str = "float32",
    selected_tokens_by_split: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    estimated_cache_bytes: int | None = None,
    free_space_safety_factor: float = 1.2,
    expected_dimensions: Mapping[int, Sequence[int]] | None = None,
    expected_eligible_valid_token_rows_by_split: Mapping[str, int] | None = None,
) -> Path:
    """Write an immutable mask-clean activation cache.

    Each batch has ``inputs``, ``attention_mask`` and, for a ``transcode``
    objective, ``targets``. Budgeted production batches additionally carry one
    sequence digest per row in ``record_sha256``. Selected token positions are
    ordinals in the unpadded model-token stream, so left/right batch padding
    cannot change token identity. ``inputs`` and ``targets`` map integer layer
    IDs to tensors shaped ``(batch, sequence, width)``.
    """

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite activation cache: {destination}")
    if set(batches_by_split) != set(SPLITS):
        raise ValueError(f"batches_by_split must contain exactly {SPLITS}")
    layers = tuple(int(layer) for layer in selected_layers)
    if (
        not layers
        or len(set(layers)) != len(layers)
        or any(layer < 0 for layer in layers)
    ):
        raise ValueError("selected_layers must be unique non-negative integers")
    if objective not in {"autoencode", "transcode"}:
        raise ValueError("objective must be autoencode or transcode")
    if storage_dtype not in {"float16", "float32"}:
        raise ValueError("storage_dtype must be float16 or float32")
    selected_tokens = (
        None
        if selected_tokens_by_split is None
        else _canonical_token_selection(selected_tokens_by_split)
    )
    if expected_dimensions is None:
        frozen_dimensions = None
    else:
        if set(expected_dimensions) != set(layers):
            raise ValueError("expected_dimensions must cover every selected layer")
        frozen_dimensions = {
            layer: tuple(int(value) for value in expected_dimensions[layer])
            for layer in layers
        }
        if any(
            len(value) != 2 or min(value) < 1 for value in frozen_dimensions.values()
        ):
            raise ValueError("invalid expected activation dimensions")
    if expected_eligible_valid_token_rows_by_split is not None:
        if set(expected_eligible_valid_token_rows_by_split) != set(SPLITS) or any(
            type(value) is not int or value < 1
            for value in expected_eligible_valid_token_rows_by_split.values()
        ):
            raise ValueError("invalid expected eligible-token counts")
    if estimated_cache_bytes is None:
        capacity_check = None
    else:
        capacity_check = require_cache_free_space(
            destination,
            estimated_bytes=estimated_cache_bytes,
            safety_factor=free_space_safety_factor,
        )
    original_sources = _validate_source_splits(source_splits)
    if activation_provenance is None:
        activation_provenance = {
            "schema_version": "r2_dictionary_activation_provenance_smoke_v1",
            "status": "incomplete_smoke_only",
        }
    if not isinstance(activation_provenance, Mapping) or not activation_provenance:
        raise ValueError("activation_provenance must be a non-empty object")
    numpy_dtype = np.dtype(storage_dtype)

    staging = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    sources: dict[str, dict[str, str]] = {}
    shards: list[dict[str, Any]] = []
    split_summaries: dict[str, dict[str, Any]] = {}
    dimensions: dict[int, tuple[int, int]] = {}
    selection_files: dict[str, dict[str, Any]] = {}
    layout = (
        "in_memory_single_file_smoke_only"
        if selected_tokens is None
        else "preallocated_single_file_per_layer_split"
    )

    try:
        for split in SPLITS:
            relative = Path("source_manifests") / f"{split}.jsonl"
            source_path = Path(original_sources[split]["manifest_path"])
            (staging / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, staging / relative)
            sources[split] = {
                "manifest_path": relative.as_posix(),
                "manifest_sha256": original_sources[split]["manifest_sha256"],
            }
            if selected_tokens is not None:
                selection_relative = Path("row_order") / f"{split}.jsonl"
                _write_jsonl(staging / selection_relative, selected_tokens[split])
                selection_digest = sha256_file(staging / selection_relative)
                selection_files[split] = {
                    "budget": len(selected_tokens[split]),
                    "selection_path": selection_relative.as_posix(),
                    "selection_sha256": selection_digest,
                    "row_order_path": selection_relative.as_posix(),
                    "row_order_sha256": selection_digest,
                }
        for split in SPLITS:
            input_buffers: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
            target_buffers: dict[int, list[np.ndarray]] = {
                layer: [] for layer in layers
            }
            input_memmaps: dict[int, np.memmap] = {}
            target_memmaps: dict[int, np.memmap] = {}
            input_paths: dict[int, Path] = {}
            target_paths: dict[int, Path] = {}
            total_rows = eligible_rows = selected_rows = excluded_rows = batch_count = 0
            selected_keys = (
                None
                if selected_tokens is None
                else {
                    (row["record_sha256"], row["token_position"])
                    for row in selected_tokens[split]
                }
            )
            destination_by_key = (
                None
                if selected_tokens is None
                else {
                    (row["record_sha256"], row["token_position"]): index
                    for index, row in enumerate(selected_tokens[split])
                }
            )
            observed_selected_keys: set[tuple[str, int]] = set()
            observed_record_shas: set[str] = set()

            expected_fields = (
                {"inputs", "attention_mask"}
                if objective == "autoencode"
                else {"inputs", "targets", "attention_mask"}
            )
            if selected_keys is not None:
                expected_fields = expected_fields | {"record_sha256"}
            for batch in batches_by_split[split]:
                if set(batch) != expected_fields:
                    raise ValueError(
                        f"unexpected {split} batch fields: {sorted(batch)}"
                    )
                input_map = batch["inputs"]
                target_map = (
                    input_map if objective == "autoencode" else batch["targets"]
                )
                if set(input_map) != set(layers) or set(target_map) != set(layers):
                    raise ValueError(
                        f"batch layers must equal selected_layers in {split}"
                    )
                attention_mask = batch["attention_mask"]
                if (
                    not isinstance(attention_mask, torch.Tensor)
                    or attention_mask.ndim != 2
                ):
                    raise ValueError("attention_mask must be a rank-two tensor")
                mask = attention_mask.to(dtype=torch.bool)
                batch_total = int(mask.numel())
                batch_eligible = int(mask.sum().item())
                if batch_eligible == 0:
                    raise ValueError(f"empty valid-token batch in {split}")
                if selected_keys is None:
                    serialization_mask = mask
                    destination_indices = None
                else:
                    record_shas = batch["record_sha256"]
                    if (
                        isinstance(record_shas, (str, bytes))
                        or not isinstance(record_shas, Sequence)
                        or len(record_shas) != mask.shape[0]
                    ):
                        raise ValueError(
                            "record_sha256 must provide one digest per batch row"
                        )
                    serialization_mask = torch.zeros_like(mask)
                    destination_indices = []
                    for row_index, record_sha256 in enumerate(record_shas):
                        token_priority(record_sha256, 0)
                        if record_sha256 in observed_record_shas:
                            raise ValueError(
                                f"duplicate activation record in {split}: {record_sha256}"
                            )
                        observed_record_shas.add(record_sha256)
                        physical_positions = torch.nonzero(
                            mask[row_index], as_tuple=False
                        ).flatten()
                        for token_position, physical_position in enumerate(
                            physical_positions.tolist()
                        ):
                            key = (record_sha256, token_position)
                            if key in selected_keys:
                                if key in observed_selected_keys:
                                    raise ValueError(
                                        f"selected token observed twice in {split}: {key}"
                                    )
                                observed_selected_keys.add(key)
                                serialization_mask[row_index, physical_position] = True
                                destination_indices.append(destination_by_key[key])
                batch_selected = int(serialization_mask.sum().item())
                total_rows += batch_total
                eligible_rows += batch_eligible
                selected_rows += batch_selected
                excluded_rows += batch_total - batch_eligible
                batch_count += 1

                if batch_selected == 0:
                    continue

                for layer in layers:
                    inputs = valid_token_rows(input_map[layer], serialization_mask)
                    targets = valid_token_rows(target_map[layer], serialization_mask)
                    if inputs.shape[0] != targets.shape[0]:
                        raise ValueError("input and target valid-token counts differ")
                    current_dims = (int(inputs.shape[1]), int(targets.shape[1]))
                    if layer in dimensions and dimensions[layer] != current_dims:
                        raise ValueError(
                            f"activation dimensions changed at layer {layer}"
                        )
                    dimensions[layer] = current_dims
                    if (
                        frozen_dimensions is not None
                        and current_dims != frozen_dimensions[layer]
                    ):
                        raise ValueError(
                            f"activation dimensions disagree with frozen geometry at layer {layer}"
                        )
                    # NumPy has no native bfloat16 dtype. Preserve the checked
                    # values through float32 on CPU before the declared cache
                    # storage conversion (and validate that conversion below).
                    input_array = (
                        inputs.detach()
                        .to(device="cpu", dtype=torch.float32)
                        .numpy()
                        .astype(numpy_dtype, copy=False)
                    )
                    target_array = (
                        targets.detach()
                        .to(device="cpu", dtype=torch.float32)
                        .numpy()
                        .astype(numpy_dtype, copy=False)
                    )
                    if (
                        not np.isfinite(input_array).all()
                        or not np.isfinite(target_array).all()
                    ):
                        raise ValueError(
                            "storage dtype conversion produced non-finite values"
                        )
                    if selected_keys is None:
                        input_buffers[layer].append(np.ascontiguousarray(input_array))
                        if objective == "transcode":
                            target_buffers[layer].append(
                                np.ascontiguousarray(target_array)
                            )
                    else:
                        if layer not in input_memmaps:
                            budget = len(selected_keys)
                            relative_dir = Path(split) / f"layer_{layer:03d}"
                            input_paths[layer] = relative_dir / "input.npy"
                            target_paths[layer] = (
                                input_paths[layer]
                                if objective == "autoencode"
                                else relative_dir / "target.npy"
                            )
                            (staging / relative_dir).mkdir(parents=True, exist_ok=True)
                            input_memmaps[layer] = np.lib.format.open_memmap(
                                staging / input_paths[layer],
                                mode="w+",
                                dtype=numpy_dtype,
                                shape=(budget, current_dims[0]),
                            )
                            if objective == "transcode":
                                target_memmaps[layer] = np.lib.format.open_memmap(
                                    staging / target_paths[layer],
                                    mode="w+",
                                    dtype=numpy_dtype,
                                    shape=(budget, current_dims[1]),
                                )
                        input_memmaps[layer][destination_indices] = input_array
                        if objective == "transcode":
                            target_memmaps[layer][destination_indices] = target_array

            if batch_count == 0:
                raise ValueError(f"no activation batches supplied for {split}")
            if selected_keys is not None and observed_selected_keys != selected_keys:
                missing = len(selected_keys - observed_selected_keys)
                raise ValueError(
                    f"exact valid-token budget was not met in {split}: "
                    f"missing {missing} selected rows"
                )
            if (
                expected_eligible_valid_token_rows_by_split is not None
                and eligible_rows != expected_eligible_valid_token_rows_by_split[split]
            ):
                raise ValueError(
                    f"first/second-pass eligible-token count mismatch in {split}"
                )
            for layer in layers:
                if selected_keys is None:
                    input_array = np.concatenate(input_buffers[layer], axis=0)
                    target_array = (
                        input_array
                        if objective == "autoencode"
                        else np.concatenate(target_buffers[layer], axis=0)
                    )
                    relative_dir = Path(split) / f"layer_{layer:03d}"
                    input_paths[layer] = relative_dir / "input.npy"
                    target_paths[layer] = (
                        input_paths[layer]
                        if objective == "autoencode"
                        else relative_dir / "target.npy"
                    )
                    _write_npy(staging / input_paths[layer], input_array)
                    if objective == "transcode":
                        _write_npy(staging / target_paths[layer], target_array)
                else:
                    input_memmaps[layer].flush()
                    if objective == "transcode":
                        target_memmaps[layer].flush()
            input_memmaps.clear()
            target_memmaps.clear()
            for layer in layers:
                input_dim, target_dim = dimensions[layer]
                shards.append(
                    {
                        "split": split,
                        "layer": layer,
                        "shard_index": 0,
                        "rows": selected_rows,
                        "input_dim": input_dim,
                        "target_dim": target_dim,
                        "dtype": storage_dtype,
                        "input_path": input_paths[layer].as_posix(),
                        "input_sha256": sha256_file(staging / input_paths[layer]),
                        "target_path": target_paths[layer].as_posix(),
                        "target_sha256": sha256_file(staging / target_paths[layer]),
                    }
                )
            split_summaries[split] = {
                "batches": batch_count,
                "total_token_rows": total_rows,
                "eligible_valid_token_rows": eligible_rows,
                "selected_valid_token_rows": selected_rows,
                "valid_token_rows": selected_rows,
                "invalid_token_rows_excluded": excluded_rows,
                "unselected_valid_token_rows": eligible_rows - selected_rows,
                "valid_fraction": eligible_rows / total_rows,
                "rows_per_layer": {str(layer): selected_rows for layer in layers},
            }

        if selected_tokens is None:
            token_selection_identity: dict[str, Any] = {
                "status": "all_valid_smoke_only",
                "method": "all_attention_mask_valid_rows",
            }
            token_selection_manifest = dict(token_selection_identity)
        else:
            selection_by_split = {
                split: {
                    "budget": selection_files[split]["budget"],
                    "eligible_valid_token_rows": split_summaries[split][
                        "eligible_valid_token_rows"
                    ],
                    "selection_sha256": selection_files[split]["selection_sha256"],
                    "row_order_sha256": selection_files[split]["row_order_sha256"],
                }
                for split in SPLITS
            }
            token_selection_identity = {
                "status": "complete_budgeted_selection",
                "method": "lowest_sha256_record_digest_colon_unpadded_token_position",
                "position_definition": "ordinal_in_unpadded_model_token_stream",
                "serialization_order": "selection_priority_order",
                "by_split": selection_by_split,
            }
            token_selection_manifest = {
                **token_selection_identity,
                "by_split": {
                    split: {**selection_by_split[split], **selection_files[split]}
                    for split in SPLITS
                },
            }

        # Extraction iterators may add second-pass fingerprints only after their
        # final batch. Freeze the completed provenance before hashing the cache.
        activation_provenance = json.loads(
            json.dumps(activation_provenance, sort_keys=True, allow_nan=False)
        )
        identity = _cache_identity(
            objective=objective,
            layers=layers,
            storage_dtype=storage_dtype,
            sources=sources,
            dimensions=dimensions,
            shards=shards,
            activation_provenance=activation_provenance,
            token_selection=token_selection_identity,
            layout=layout,
        )
        manifest = {
            "schema_version": CACHE_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "objective": objective,
            "selected_layers": list(layers),
            "storage_dtype": storage_dtype,
            "layout": layout,
            "mask_contract": {
                "attention_mask_required": True,
                "invalid_rows": "excluded_before_serialization",
                "all_valid_rows": "serialized_without_special_case",
            },
            "source_splits": sources,
            "source_original_paths": {
                split: original_sources[split]["manifest_path"] for split in SPLITS
            },
            "activation_provenance": activation_provenance,
            "token_selection": token_selection_manifest,
            "capacity_check": capacity_check,
            "dimensions": {
                str(layer): {
                    "input": dimensions[layer][0],
                    "target": dimensions[layer][1],
                }
                for layer in layers
            },
            "split_summaries": split_summaries,
            "shards": shards,
            "content_identity": identity,
            "content_sha256": _identity_sha256(identity),
            "builder_module_sha256": sha256_file(Path(__file__)),
        }
        manifest_path = staging / "manifest.json"
        write_json(manifest_path, manifest)
        os.replace(staging, destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


@dataclass(frozen=True)
class ActivationCache:
    manifest_path: Path
    manifest_sha256: str
    content_sha256: str
    objective: str
    selected_layers: tuple[int, ...]
    dimensions: dict[int, tuple[int, int]]
    shards: tuple[dict[str, Any], ...]
    payload: dict[str, Any]
    verified_splits: tuple[str, ...]


def _resolve_member(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"cache member escapes cache root: {relative}")
    return candidate


def load_activation_cache(
    manifest_path: Path,
    *,
    verify_hashes: bool = True,
    access_splits: Sequence[str] = SPLITS,
) -> ActivationCache:
    """Validate all metadata while opening only explicitly authorized splits."""

    path = Path(manifest_path)
    authorized_splits = tuple(access_splits)
    if (
        not authorized_splits
        or len(set(authorized_splits)) != len(authorized_splits)
        or not set(authorized_splits) <= set(SPLITS)
    ):
        raise ValueError("access_splits must be a unique non-empty split subset")
    payload = load_strict_json(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != CACHE_SCHEMA:
        raise ValueError("unsupported activation-cache schema")
    if payload.get("objective") not in {"autoencode", "transcode"}:
        raise ValueError("invalid activation-cache objective")
    if payload.get("layout") not in {
        "in_memory_single_file_smoke_only",
        "preallocated_single_file_per_layer_split",
    }:
        raise ValueError("invalid activation-cache layout")
    layers = tuple(payload.get("selected_layers", ()))
    if (
        not layers
        or len(set(layers)) != len(layers)
        or any(type(value) is not int or value < 0 for value in layers)
    ):
        raise ValueError("invalid selected_layers in cache manifest")
    sources = _validate_source_splits(
        payload.get("source_splits", {}),
        root=path.parent,
        verify_splits=authorized_splits,
    )
    selection = payload.get("token_selection")
    if selection == {
        "status": "all_valid_smoke_only",
        "method": "all_attention_mask_valid_rows",
    }:
        token_selection_identity = dict(selection)
    else:
        required_selection = {
            "status",
            "method",
            "position_definition",
            "serialization_order",
            "by_split",
        }
        if not isinstance(selection, dict) or set(selection) != required_selection:
            raise ValueError("invalid cache token-selection contract")
        if (
            selection["status"] != "complete_budgeted_selection"
            or selection["method"]
            != "lowest_sha256_record_digest_colon_unpadded_token_position"
            or selection["position_definition"]
            != "ordinal_in_unpadded_model_token_stream"
            or selection["serialization_order"] != "selection_priority_order"
            or set(selection["by_split"]) != set(SPLITS)
        ):
            raise ValueError("unsupported cache token-selection contract")
        identity_by_split: dict[str, dict[str, Any]] = {}
        selection_rows_by_split: dict[str, list[dict[str, Any]]] = {}
        for split in SPLITS:
            record = selection["by_split"][split]
            required_record = {
                "budget",
                "eligible_valid_token_rows",
                "selection_path",
                "selection_sha256",
                "row_order_path",
                "row_order_sha256",
            }
            if not isinstance(record, dict) or set(record) != required_record:
                raise ValueError(f"invalid token-selection metadata for {split}")
            if (
                type(record["budget"]) is not int
                or record["budget"] < 1
                or type(record["eligible_valid_token_rows"]) is not int
                or record["eligible_valid_token_rows"] < record["budget"]
            ):
                raise ValueError(f"invalid token-selection counts for {split}")
            selection_path = _resolve_member(path.parent, record["selection_path"])
            row_order_path = _resolve_member(path.parent, record["row_order_path"])
            if (
                row_order_path != selection_path
                or record["row_order_sha256"] != record["selection_sha256"]
            ):
                raise ValueError(f"selection/row-order identity mismatch for {split}")
            if split in authorized_splits:
                if not selection_path.is_file():
                    raise FileNotFoundError(
                        f"missing token-selection file for {split}: {selection_path}"
                    )
                if sha256_file(selection_path) != record["selection_sha256"]:
                    raise ValueError(f"token-selection SHA-256 mismatch for {split}")
                selection_rows_by_split[split] = load_strict_jsonl(selection_path)
        canonical_by_split = _canonical_token_selection(
            selection_rows_by_split,
            required_splits=authorized_splits,
        )
        for split in SPLITS:
            record = selection["by_split"][split]
            if split in authorized_splits:
                canonical = canonical_by_split[split]
                selection_rows = selection_rows_by_split[split]
                if selection_rows != canonical or len(canonical) != record["budget"]:
                    raise ValueError(f"noncanonical token-selection rows for {split}")
            identity_by_split[split] = {
                "budget": record["budget"],
                "eligible_valid_token_rows": record["eligible_valid_token_rows"],
                "selection_sha256": record["selection_sha256"],
                "row_order_sha256": record["row_order_sha256"],
            }
        token_selection_identity = {
            "status": selection["status"],
            "method": selection["method"],
            "position_definition": selection["position_definition"],
            "serialization_order": selection["serialization_order"],
            "by_split": identity_by_split,
        }
    dimensions_payload = payload.get("dimensions", {})
    dimensions: dict[int, tuple[int, int]] = {}
    for layer in layers:
        record = dimensions_payload.get(str(layer))
        if not isinstance(record, dict) or set(record) != {"input", "target"}:
            raise ValueError(f"missing dimensions for layer {layer}")
        dims = (record["input"], record["target"])
        if any(type(value) is not int or value < 1 for value in dims):
            raise ValueError(f"invalid dimensions for layer {layer}")
        dimensions[layer] = dims

    root = path.parent
    seen: set[tuple[str, int, int]] = set()
    counts = {(split, layer): 0 for split in SPLITS for layer in layers}
    checked_shards: list[dict[str, Any]] = []
    for shard in payload.get("shards", ()):
        required = {
            "split",
            "layer",
            "shard_index",
            "rows",
            "input_dim",
            "target_dim",
            "dtype",
            "input_path",
            "input_sha256",
            "target_path",
            "target_sha256",
        }
        if not isinstance(shard, dict) or set(shard) != required:
            raise ValueError("invalid cache-shard fields")
        key = (shard["split"], shard["layer"], shard["shard_index"])
        if key in seen or key[0] not in SPLITS or key[1] not in layers:
            raise ValueError(f"invalid or duplicate cache shard: {key}")
        if key[2] != counts[(key[0], key[1])]:
            raise ValueError(f"non-contiguous shard indices for {key[:2]}")
        seen.add(key)
        counts[(key[0], key[1])] += 1
        if type(shard["rows"]) is not int or shard["rows"] < 1:
            raise ValueError("cache shard must contain at least one row")
        if (shard["input_dim"], shard["target_dim"]) != dimensions[key[1]]:
            raise ValueError(f"dimension mismatch in cache shard {key}")
        if shard["dtype"] != payload.get("storage_dtype"):
            raise ValueError(f"dtype mismatch in cache shard {key}")
        input_path = _resolve_member(root, shard["input_path"])
        target_path = _resolve_member(root, shard["target_path"])
        if key[0] in authorized_splits:
            if not input_path.is_file() or not target_path.is_file():
                raise FileNotFoundError(f"missing cache shard file for {key}")
            if verify_hashes:
                if sha256_file(input_path) != shard["input_sha256"]:
                    raise ValueError(f"input shard SHA-256 mismatch for {key}")
                if sha256_file(target_path) != shard["target_sha256"]:
                    raise ValueError(f"target shard SHA-256 mismatch for {key}")
            input_array = np.load(input_path, mmap_mode="r", allow_pickle=False)
            target_array = np.load(target_path, mmap_mode="r", allow_pickle=False)
            if input_array.shape != (shard["rows"], shard["input_dim"]):
                raise ValueError(f"input shard shape mismatch for {key}")
            if target_array.shape != (shard["rows"], shard["target_dim"]):
                raise ValueError(f"target shard shape mismatch for {key}")
            expected_dtype = np.dtype(shard["dtype"])
            if (
                input_array.dtype != expected_dtype
                or target_array.dtype != expected_dtype
            ):
                raise ValueError(f"array dtype mismatch for cache shard {key}")
            if (
                not np.isfinite(input_array).all()
                or not np.isfinite(target_array).all()
            ):
                raise ValueError(f"non-finite values in cache shard {key}")
        checked_shards.append(dict(shard))
    if any(count == 0 for count in counts.values()):
        raise ValueError("every selected layer must have shards in every split")

    summaries = payload.get("split_summaries", {})
    if set(summaries) != set(SPLITS):
        raise ValueError("cache manifest lacks required split summaries")
    for split in SPLITS:
        for layer in layers:
            observed = sum(
                shard["rows"]
                for shard in checked_shards
                if shard["split"] == split and shard["layer"] == layer
            )
            expected = summaries[split]["rows_per_layer"].get(str(layer))
            if observed != expected:
                raise ValueError(f"row-count mismatch for {split} layer {layer}")
        summary = summaries[split]
        if selection["status"] == "complete_budgeted_selection":
            selection_record = selection["by_split"][split]
            if (
                summary.get("selected_valid_token_rows") != selection_record["budget"]
                or summary.get("valid_token_rows") != selection_record["budget"]
                or summary.get("eligible_valid_token_rows")
                != selection_record["eligible_valid_token_rows"]
            ):
                raise ValueError(f"token-selection summary mismatch for {split}")
        elif summary.get("selected_valid_token_rows") != summary.get(
            "eligible_valid_token_rows"
        ):
            raise ValueError(f"all-valid cache omitted valid rows in {split}")

    identity = _cache_identity(
        objective=payload["objective"],
        layers=layers,
        storage_dtype=payload["storage_dtype"],
        sources=sources,
        dimensions=dimensions,
        shards=checked_shards,
        activation_provenance=payload.get("activation_provenance", {}),
        token_selection=token_selection_identity,
        layout=payload["layout"],
    )
    if payload.get("content_identity") != identity:
        raise ValueError("activation-cache content identity fields disagree")
    content_sha256 = _identity_sha256(identity)
    if payload.get("content_sha256") != content_sha256:
        raise ValueError("activation-cache content SHA-256 mismatch")

    return ActivationCache(
        manifest_path=path.resolve(),
        manifest_sha256=sha256_file(path),
        content_sha256=content_sha256,
        objective=payload["objective"],
        selected_layers=layers,
        dimensions=dimensions,
        shards=tuple(checked_shards),
        payload=payload,
        verified_splits=authorized_splits,
    )


class CachedLayerRows:
    """Memory-mapped random access to one immutable split/layer pair."""

    def __init__(self, cache: ActivationCache, split: str, layer: int):
        if split not in cache.verified_splits or layer not in cache.selected_layers:
            raise ValueError("unknown cache split or layer")
        records = sorted(
            (
                row
                for row in cache.shards
                if row["split"] == split and row["layer"] == layer
            ),
            key=lambda row: row["shard_index"],
        )
        root = cache.manifest_path.parent
        self.inputs = [
            np.load(
                _resolve_member(root, row["input_path"]),
                mmap_mode="r",
                allow_pickle=False,
            )
            for row in records
        ]
        self.targets = [
            np.load(
                _resolve_member(root, row["target_path"]),
                mmap_mode="r",
                allow_pickle=False,
            )
            for row in records
        ]
        self.ends = np.cumsum([row["rows"] for row in records], dtype=np.int64)
        self.n_rows = int(self.ends[-1])
        self.input_dim, self.target_dim = cache.dimensions[layer]
        self.split = split
        self.layer = layer

    def take(
        self,
        indices: Sequence[int] | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        requested = np.asarray(indices, dtype=np.int64)
        if requested.ndim != 1 or requested.size == 0:
            raise ValueError("indices must be a non-empty vector")
        if requested.min() < 0 or requested.max() >= self.n_rows:
            raise IndexError("cached activation index out of range")
        output_x = np.empty((requested.size, self.input_dim), dtype=np.float32)
        output_y = np.empty((requested.size, self.target_dim), dtype=np.float32)
        shard_ids = np.searchsorted(self.ends, requested, side="right")
        starts = np.concatenate(([0], self.ends[:-1]))
        for shard_id in np.unique(shard_ids):
            positions = np.flatnonzero(shard_ids == shard_id)
            local = requested[positions] - starts[shard_id]
            output_x[positions] = self.inputs[shard_id][local]
            output_y[positions] = self.targets[shard_id][local]
        return torch.from_numpy(output_x), torch.from_numpy(output_y)

    def batches(self, batch_size: int) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for start in range(0, self.n_rows, batch_size):
            yield self.take(np.arange(start, min(start + batch_size, self.n_rows)))


class CachedMultiLayerRows:
    """Aligned cached CLT inputs and MLP targets for all selected layers."""

    def __init__(
        self,
        cache: ActivationCache,
        split: str,
        *,
        prefix_rows: int | None = None,
    ):
        if cache.objective != "transcode":
            raise ValueError(
                "multi-layer dictionary controls require a transcode cache"
            )
        self.layers = cache.selected_layers
        if tuple(sorted(self.layers)) != self.layers:
            raise ValueError("selected cache layers must be ordered")
        self.rows = {
            layer: CachedLayerRows(cache, split, layer) for layer in self.layers
        }
        counts = {rows.n_rows for rows in self.rows.values()}
        if len(counts) != 1:
            raise ValueError("cached layers do not have aligned token-row counts")
        input_dims = {rows.input_dim for rows in self.rows.values()}
        target_dims = {rows.target_dim for rows in self.rows.values()}
        if len(input_dims) != 1 or len(target_dims) != 1:
            raise ValueError(
                "windowed controls require common dimensions across layers"
            )
        available_rows = counts.pop()
        if prefix_rows is not None and not 1 <= prefix_rows <= available_rows:
            raise ValueError("cache prefix_rows is outside the available split")
        self.n_rows = available_rows if prefix_rows is None else prefix_rows
        self.input_dim = input_dims.pop()
        self.target_dim = target_dims.pop()
        self.split = split

    def take(
        self, indices: Sequence[int] | np.ndarray
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        requested = np.asarray(indices, dtype=np.int64)
        if (
            requested.ndim != 1
            or requested.size == 0
            or requested.min() < 0
            or requested.max() >= self.n_rows
        ):
            raise IndexError("cached multi-layer index is outside the selected prefix")
        pairs = [self.rows[layer].take(requested) for layer in self.layers]
        return [pair[0] for pair in pairs], [pair[1] for pair in pairs]

    def batches(
        self, batch_size: int
    ) -> Iterable[tuple[list[torch.Tensor], list[torch.Tensor]]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for start in range(0, self.n_rows, batch_size):
            indices = np.arange(start, min(start + batch_size, self.n_rows))
            yield self.take(indices)


class DeterministicBatchStream:
    """Infinite seeded permutations with an exact, method-independent cursor."""

    def __init__(self, n_rows: int, batch_size: int, seed: int):
        if n_rows < 1 or batch_size < 1:
            raise ValueError("n_rows and batch_size must be positive")
        self.n_rows = int(n_rows)
        self.batch_size = int(batch_size)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.order = torch.randperm(self.n_rows, generator=self.generator).numpy()
        self.cursor = 0

    def next(self) -> np.ndarray:
        pieces: list[np.ndarray] = []
        remaining = self.batch_size
        while remaining:
            available = self.n_rows - self.cursor
            take = min(remaining, available)
            pieces.append(self.order[self.cursor : self.cursor + take])
            self.cursor += take
            remaining -= take
            if self.cursor == self.n_rows:
                self.order = torch.randperm(
                    self.n_rows,
                    generator=self.generator,
                ).numpy()
                self.cursor = 0
        return np.concatenate(pieces)

    def state_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "batch_size": self.batch_size,
            "generator_state": self.generator.get_state(),
            "order": self.order.copy(),
            "cursor": self.cursor,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("n_rows") != self.n_rows
            or state.get("batch_size") != self.batch_size
        ):
            raise ValueError("batch-stream dimensions changed across resume")
        order = np.asarray(state.get("order"), dtype=np.int64)
        cursor = state.get("cursor")
        if (
            order.shape != (self.n_rows,)
            or order.min(initial=0) != 0
            or order.max(initial=-1) != self.n_rows - 1
        ):
            raise ValueError("invalid resumed batch permutation")
        if type(cursor) is not int or not 0 <= cursor < self.n_rows:
            raise ValueError("invalid resumed batch cursor")
        self.generator.set_state(state["generator_state"])
        self.order = order.copy()
        self.cursor = cursor


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def windowed_transcoder_parameter_count(
    *,
    method: str,
    n_layers: int,
    input_dim: int,
    target_dim: int,
    sparse_width: int,
    dense_rank: int,
    window: int,
) -> int:
    """Return the exact trainable count without allocating a production model."""

    if method not in METHODS:
        raise ValueError(f"unknown dictionary method: {method}")
    if min(n_layers, input_dim, target_dim, sparse_width, dense_rank, window) < 1:
        raise ValueError("windowed transcoder dimensions must be positive")
    width = dense_rank if method == "dense_low_rank" else sparse_width
    effective_window = min(window, n_layers)
    encoder = n_layers * width * input_dim + n_layers * width
    decoder = sum(
        width * min(effective_window, n_layers - layer) * target_dim
        for layer in range(n_layers)
    )
    biases = n_layers * target_dim
    gated = 2 * n_layers * width if method == "gated_sae" else 0
    return encoder + decoder + biases + gated


class WindowedTranscoder(nn.Module):
    """Windowed CLT-input-to-MLP-output control with a selectable sparsifier."""

    def __init__(
        self,
        *,
        method: str,
        n_layers: int,
        input_dim: int,
        target_dim: int,
        width: int,
        window: int,
        topk_k: int = 128,
        l1_coefficient: float = 0.0,
        gated_auxiliary_coefficient: float = 1.0,
    ) -> None:
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"unknown dictionary method: {method}")
        if min(n_layers, input_dim, target_dim, width, window) < 1:
            raise ValueError("windowed transcoder dimensions must be positive")
        if method == "dense_low_rank" and width >= min(input_dim, target_dim):
            raise ValueError(
                "dense bottleneck rank must be below both activation widths"
            )
        if l1_coefficient < 0 or gated_auxiliary_coefficient < 0:
            raise ValueError("loss coefficients must be non-negative")
        if method == "topk_clt" and not 1 <= topk_k <= width:
            raise ValueError("TopK k must lie between one and dictionary width")
        self.method = method
        self.n_layers = int(n_layers)
        self.input_dim = int(input_dim)
        self.target_dim = int(target_dim)
        self.width = int(width)
        self.window = min(int(window), self.n_layers)
        self.topk_k = int(topk_k)
        self.l1_coefficient = float(l1_coefficient)
        self.gated_auxiliary_coefficient = float(gated_auxiliary_coefficient)

        self.encoder_weight = nn.Parameter(
            torch.empty(self.n_layers, self.width, self.input_dim)
        )
        self.encoder_bias = nn.Parameter(torch.zeros(self.n_layers, self.width))
        if self.method == "gated_sae":
            self.log_magnitude_scale = nn.Parameter(
                torch.zeros(self.n_layers, self.width)
            )
            self.magnitude_bias = nn.Parameter(torch.zeros(self.n_layers, self.width))
        else:
            self.register_parameter("log_magnitude_scale", None)
            self.register_parameter("magnitude_bias", None)
        self.decoder_weight = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.width,
                        min(self.window, self.n_layers - layer),
                        self.target_dim,
                    )
                )
                for layer in range(self.n_layers)
            ]
        )
        self.decoder_bias = nn.Parameter(torch.zeros(self.n_layers, self.target_dim))
        self.reset_parameters()

    @property
    def sparse(self) -> bool:
        return self.method != "dense_low_rank"

    def reset_parameters(self) -> None:
        for layer in range(self.n_layers):
            nn.init.kaiming_uniform_(self.encoder_weight[layer], a=math.sqrt(5))
            nn.init.normal_(
                self.decoder_weight[layer],
                mean=0.0,
                std=self.target_dim**-0.5,
            )
        self.normalize_decoder()

    def _encode_details(
        self, inputs: Sequence[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        if len(inputs) != self.n_layers:
            raise ValueError("input layer count does not match transcoder")
        codes: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        for layer, values in enumerate(inputs):
            if values.ndim != 2 or values.shape[1] != self.input_dim:
                raise ValueError(f"invalid input shape at layer {layer}")
            direction = F.linear(values, self.encoder_weight[layer])
            preactivation = direction + self.encoder_bias[layer]
            if self.method == "topk_clt":
                positive = F.relu(preactivation)
                top_values, top_indices = torch.topk(
                    positive,
                    self.topk_k,
                    dim=1,
                    sorted=False,
                )
                codes.append(
                    torch.zeros_like(positive).scatter(1, top_indices, top_values)
                )
            elif self.method == "relu_l1_sae":
                codes.append(F.relu(preactivation))
            elif self.method == "gated_sae":
                magnitude = (
                    direction * self.log_magnitude_scale[layer].exp()
                    + self.magnitude_bias[layer]
                )
                codes.append(
                    (preactivation > 0).to(magnitude.dtype) * F.relu(magnitude)
                )
                gates.append(F.relu(preactivation))
            else:
                codes.append(preactivation)
        return codes, gates if self.method == "gated_sae" else None

    def encode(self, inputs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return self._encode_details(inputs)[0]

    def _threshold_codes(
        self,
        codes: Sequence[torch.Tensor],
        activation_threshold: float,
    ) -> list[torch.Tensor]:
        if activation_threshold < 0:
            raise ValueError("activation_threshold must be non-negative")
        if not self.sparse or activation_threshold == 0:
            return list(codes)
        return [
            torch.where(code > activation_threshold, code, torch.zeros_like(code))
            for code in codes
        ]

    def decode(
        self,
        codes: Sequence[torch.Tensor],
        *,
        detach_parameters: bool = False,
    ) -> list[torch.Tensor]:
        if len(codes) != self.n_layers:
            raise ValueError("code layer count does not match transcoder")
        batch = codes[0].shape[0]
        biases = self.decoder_bias.detach() if detach_parameters else self.decoder_bias
        outputs = [
            biases[layer].expand(batch, -1).clone() for layer in range(self.n_layers)
        ]
        for source, values in enumerate(codes):
            weights = (
                self.decoder_weight[source].detach()
                if detach_parameters
                else self.decoder_weight[source]
            )
            contribution = torch.einsum("bf,ftd->btd", values, weights)
            for offset in range(contribution.shape[1]):
                outputs[source + offset] = (
                    outputs[source + offset] + contribution[:, offset]
                )
        return outputs

    def reconstruct(
        self,
        inputs: Sequence[torch.Tensor],
        *,
        activation_threshold: float = 0.0,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        codes = self.encode(inputs)
        thresholded = self._threshold_codes(codes, activation_threshold)
        return self.decode(thresholded), thresholded

    def objective(
        self,
        inputs: Sequence[torch.Tensor],
        targets: Sequence[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if len(targets) != self.n_layers:
            raise ValueError("target layer count does not match transcoder")
        raw_codes, gates = self._encode_details(inputs)
        reconstructions = self.decode(raw_codes)
        mse = torch.stack(
            [
                F.mse_loss(reconstruction, target)
                for reconstruction, target in zip(reconstructions, targets)
            ]
        ).mean()
        if self.method in {"topk_clt", "dense_low_rank"}:
            return {"loss": mse, "reconstruction_mse": mse}
        penalty_values = gates if gates is not None else raw_codes
        l1 = torch.stack(
            [values.abs().sum(dim=1).mean() for values in penalty_values]
        ).mean()
        loss = mse + self.l1_coefficient * l1
        result = {"loss": loss, "reconstruction_mse": mse, "l1": l1}
        if gates is not None:
            auxiliary = self.decode(gates, detach_parameters=True)
            auxiliary_mse = torch.stack(
                [
                    F.mse_loss(reconstruction, target)
                    for reconstruction, target in zip(auxiliary, targets)
                ]
            ).mean()
            result["gate_auxiliary_mse"] = auxiliary_mse
            result["loss"] = (
                result["loss"] + self.gated_auxiliary_coefficient * auxiliary_mse
            )
        return result

    def activity_mask(
        self,
        codes: Sequence[torch.Tensor],
        threshold: float,
    ) -> list[torch.Tensor]:
        if self.sparse:
            return [code > threshold for code in codes]
        return [code.abs() > threshold for code in codes]

    def decoder_norms(self) -> torch.Tensor:
        return torch.stack(
            [weights.square().sum(dim=(1, 2)).sqrt() for weights in self.decoder_weight]
        )

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        if not self.sparse:
            return
        for weights in self.decoder_weight:
            norms = (
                weights.square().sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-12)
            )
            weights.div_(norms)

    def forward_matmul_macs_per_token(self) -> int:
        encoder = self.n_layers * self.input_dim * self.width
        return encoder + self.decoder_matmul_macs_per_token()

    def decoder_matmul_macs_per_token(self) -> int:
        return sum(
            self.width * min(self.window, self.n_layers - layer) * self.target_dim
            for layer in range(self.n_layers)
        )

    def training_flop_proxy_per_token(self) -> int:
        """Prespecified matmul proxy including the gated auxiliary decode."""

        proxy = 6 * self.forward_matmul_macs_per_token()
        if self.method == "gated_sae":
            # Auxiliary decoder: forward multiply/add plus input-gradient multiply/add.
            proxy += 4 * self.decoder_matmul_macs_per_token()
        return proxy


def build_windowed_transcoder(
    *,
    method: str,
    n_layers: int,
    input_dim: int,
    target_dim: int,
    sparse_width: int,
    dense_rank: int,
    window: int,
    l1_coefficient: float,
    gated_auxiliary_coefficient: float,
    topk_k: int = 128,
) -> WindowedTranscoder:
    width = dense_rank if method == "dense_low_rank" else sparse_width
    return WindowedTranscoder(
        method=method,
        n_layers=n_layers,
        input_dim=input_dim,
        target_dim=target_dim,
        width=width,
        window=window,
        topk_k=topk_k,
        l1_coefficient=l1_coefficient,
        gated_auxiliary_coefficient=gated_auxiliary_coefficient,
    )


def _quantile_dict(values: torch.Tensor) -> dict[str, float]:
    if values.ndim != 1 or values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("quantile values must be a non-empty finite vector")
    probabilities = torch.tensor(QUANTILES, dtype=torch.float64)
    results = torch.quantile(values.to(torch.float64), probabilities)
    return {
        f"q{int(round(q * 100)):02d}": float(value)
        for q, value in zip(QUANTILES, results)
    }


@torch.inference_mode()
def evaluate_windowed_transcoder(
    model: WindowedTranscoder,
    rows: CachedMultiLayerRows,
    *,
    device: torch.device,
    batch_size: int,
    activation_threshold: float,
    dead_frequency_threshold: float,
    detailed: bool,
) -> dict[str, Any]:
    """Evaluate the common multi-layer transcode estimand on one frozen split."""

    if model.n_layers != len(rows.layers):
        raise ValueError("cache/model layer counts differ")
    if batch_size < 1 or activation_threshold < 0:
        raise ValueError("invalid evaluation batch size or activation threshold")
    if not 0 <= dead_frequency_threshold <= 1:
        raise ValueError("dead_frequency_threshold must lie in [0, 1]")
    model.eval()
    model_dtype = model.encoder_weight.dtype
    squared_error = np.zeros(model.n_layers, dtype=np.float64)
    target_sum = np.zeros(model.n_layers, dtype=np.float64)
    target_square_sum = np.zeros(model.n_layers, dtype=np.float64)
    target_elements = np.zeros(model.n_layers, dtype=np.int64)
    l0_sum = np.zeros(model.n_layers, dtype=np.float64)
    firing_counts = [torch.zeros(model.width, dtype=torch.int64) for _ in rows.layers]
    token_errors: list[torch.Tensor] = []

    for inputs, targets in rows.batches(batch_size):
        inputs = [values.to(device=device, dtype=model_dtype) for values in inputs]
        targets = [values.to(device=device, dtype=model_dtype) for values in targets]
        reconstructions, codes = model.reconstruct(
            inputs,
            activation_threshold=activation_threshold,
        )
        active = model.activity_mask(codes, activation_threshold)
        for index, (reconstruction, target, mask) in enumerate(
            zip(reconstructions, targets, active)
        ):
            difference = (reconstruction - target).to(torch.float64)
            target64 = target.to(torch.float64)
            squared_error[index] += float(difference.square().sum().item())
            target_sum[index] += float(target64.sum().item())
            target_square_sum[index] += float(target64.square().sum().item())
            target_elements[index] += target64.numel()
            firing_counts[index] += mask.sum(dim=0).to(
                device="cpu",
                dtype=torch.int64,
            )
            l0_sum[index] += float(mask.sum().item())
            if detailed:
                token_errors.append(difference.square().mean(dim=1).detach().cpu())

    centered = target_square_sum - np.square(target_sum) / target_elements
    if np.any(centered <= 0):
        raise ValueError("target variance is zero in at least one layer")
    fvu = squared_error / centered
    mse = squared_error / target_elements
    decoder_norms = model.decoder_norms().detach().cpu().to(torch.float64)
    source_rows: list[dict[str, Any]] = []
    for index, layer in enumerate(rows.layers):
        frequencies = firing_counts[index].to(torch.float64) / rows.n_rows
        norms = decoder_norms[index]
        source = {
            "layer": layer,
            "l0_mean": float(l0_sum[index] / rows.n_rows),
            "dead_fraction": float(
                (frequencies < dead_frequency_threshold).to(torch.float64).mean()
            ),
            "firing_frequency_quantiles": _quantile_dict(frequencies),
            "decoder_norm_quantiles": _quantile_dict(norms),
        }
        if detailed:
            source["firing_frequency_per_feature"] = frequencies.tolist()
            source["decoder_norm_per_feature"] = norms.tolist()
        source_rows.append(source)
    target_rows = [
        {
            "layer": layer,
            "mse": float(mse[index]),
            "fvu": float(fvu[index]),
        }
        for index, layer in enumerate(rows.layers)
    ]
    result: dict[str, Any] = {
        "split": rows.split,
        "objective": "windowed_multi_layer_clt_input_to_mlp_output_transcoding",
        "selected_layers": list(rows.layers),
        "decoder_window": model.window,
        "n_valid_tokens": rows.n_rows,
        "fvu_mean": float(fvu.mean()),
        "fvu_pooled_layer_centered": float(squared_error.sum() / centered.sum()),
        "mse_mean": float(mse.mean()),
        "l0_mean": float(l0_sum.sum() / (rows.n_rows * model.n_layers)),
        "activation_threshold": activation_threshold,
        "dead_frequency_threshold": dead_frequency_threshold,
        "dead_fraction_median": float(
            np.median([row["dead_fraction"] for row in source_rows])
        ),
        "target_layers": target_rows,
        "source_layers": source_rows,
    }
    if detailed:
        result["reconstruction_error_quantiles"] = _quantile_dict(
            torch.cat(token_errors)
        )
    return result


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    steps: int
    batch_size: int
    evaluation_batch_size: int
    learning_rate: float
    validation_every: int
    gradient_clip_norm: float
    warmup_steps: int = 0
    checkpoint_every: int = 0
    activation_threshold: float = 1e-8
    dead_frequency_threshold: float = 0.001

    def validate(self) -> None:
        if self.steps < 1 or self.batch_size < 1 or self.evaluation_batch_size < 1:
            raise ValueError("steps and batch sizes must be positive")
        if self.learning_rate <= 0 or self.validation_every < 1:
            raise ValueError("learning rate and validation interval must be positive")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.warmup_steps < 0 or self.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must be non-negative and below total steps")
        if self.checkpoint_every < 0:
            raise ValueError("checkpoint_every must be non-negative")
        if self.activation_threshold < 0 or not 0 <= self.dead_frequency_threshold <= 1:
            raise ValueError("invalid activity/dead thresholds")


def _max_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * 1024)  # Linux reports KiB.


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def train_windowed_transcoder(
    model: WindowedTranscoder,
    train_rows: CachedMultiLayerRows,
    validation_rows: CachedMultiLayerRows,
    *,
    device: torch.device,
    config: TrainingConfig,
    stream_seed: int,
    candidate_id: str,
    progress_path: Path,
    best_path: Path,
    resume: bool,
) -> dict[str, Any]:
    """Train one candidate with resumable optimizer/scheduler/RNG/data state."""

    config.validate()
    if not candidate_id:
        raise ValueError("candidate_id must be non-empty")
    model.to(device)
    model_dtype = model.encoder_weight.dtype
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def learning_rate_factor(step_index: int) -> float:
        if config.warmup_steps == 0:
            return 1.0
        return min((step_index + 1) / config.warmup_steps, 1.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=learning_rate_factor,
    )
    stream = DeterministicBatchStream(
        train_rows.n_rows,
        config.batch_size,
        stream_seed,
    )
    start_step = 0
    best_fvu = math.inf
    best_step = -1
    validation_history: list[dict[str, Any]] = []
    resumed = False
    elapsed_before = 0.0
    peak_allocated_before = 0
    peak_reserved_before = 0
    last_components: dict[str, float] = {}
    checkpoint_io_seconds = 0.0
    timing_path = progress_path.with_name(f"{progress_path.name}.timing.json")

    if progress_path.exists():
        if not resume:
            raise FileExistsError(
                f"progress checkpoint exists; pass --resume: {progress_path}"
            )
        state = torch.load(progress_path, map_location="cpu", weights_only=False)
        if state.get("schema_version") != "r2_dictionary_control_progress_v2":
            raise ValueError("unsupported progress-checkpoint schema")
        if state.get("candidate_id") != candidate_id:
            raise ValueError("candidate identity changed across resume")
        if state.get("training_config") != training_config_dict(config):
            raise ValueError("training configuration changed across resume")
        if state.get("stream_seed") != stream_seed:
            raise ValueError("batch-stream seed changed across resume")
        model.load_state_dict(state["model_state_dict"])
        model.to(device)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if isinstance(value, torch.Tensor):
                    optimizer_state[key] = value.to(device)
        scheduler.load_state_dict(state["scheduler_state_dict"])
        stream.load_state_dict(state["batch_stream_state"])
        _restore_rng_state(state["rng_state"])
        start_step = int(state["step"])
        best_fvu = float(state["best_validation_fvu"])
        best_step = int(state["best_validation_step"])
        validation_history = list(state["validation_history"])
        timing = load_strict_json(timing_path)
        if (
            set(timing)
            != {
                "schema_version",
                "candidate_id",
                "step",
                "progress_file_size_bytes",
                "checkpoint_io_seconds",
                "wall_time_seconds",
            }
            or timing["schema_version"] != "r2_dictionary_control_timing_v1"
            or timing["candidate_id"] != candidate_id
            or timing["step"] != start_step
            or timing["progress_file_size_bytes"] != progress_path.stat().st_size
            or timing["wall_time_seconds"] < state["wall_time_seconds"]
        ):
            raise ValueError("progress timing sidecar does not match checkpoint")
        elapsed_before = float(timing["wall_time_seconds"])
        checkpoint_io_seconds = float(timing["checkpoint_io_seconds"])
        peak_allocated_before = int(
            state["peak_accelerator_memory_allocated_bytes"] or 0
        )
        peak_reserved_before = int(state["peak_accelerator_memory_reserved_bytes"] or 0)
        last_components = dict(state["last_training_components"])
        resumed = True
    elif resume:
        raise FileNotFoundError(f"no progress checkpoint to resume: {progress_path}")
    elif timing_path.exists():
        raise FileExistsError(f"orphan progress timing sidecar exists: {timing_path}")

    if start_step > config.steps:
        raise ValueError("progress checkpoint is beyond total steps")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    for step in range(start_step + 1, config.steps + 1):
        model.train()
        inputs, targets = train_rows.take(stream.next())
        inputs = [values.to(device=device, dtype=model_dtype) for values in inputs]
        targets = [values.to(device=device, dtype=model_dtype) for values in targets]
        optimizer.zero_grad(set_to_none=True)
        components = model.objective(inputs, targets)
        loss = components["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        model.normalize_decoder()
        scheduler.step()
        last_components = {
            name: float(value.detach().item()) for name, value in components.items()
        }

        validate = step % config.validation_every == 0 or step == config.steps
        if validate:
            validation = evaluate_windowed_transcoder(
                model,
                validation_rows,
                device=device,
                batch_size=config.evaluation_batch_size,
                activation_threshold=config.activation_threshold,
                dead_frequency_threshold=config.dead_frequency_threshold,
                detailed=False,
            )
            validation_history.append(
                {
                    "step": step,
                    "fvu_mean": validation["fvu_mean"],
                    "l0_mean_at_zero_threshold": validation["l0_mean"],
                }
            )
            if validation["fvu_mean"] < best_fvu:
                best_fvu = validation["fvu_mean"]
                best_step = step
                atomic_torch_save(
                    best_path,
                    {
                        "schema_version": "r2_dictionary_control_best_v1",
                        "candidate_id": candidate_id,
                        "step": step,
                        "validation_fvu_mean": best_fvu,
                        "model_state_dict": {
                            name: value.detach().cpu()
                            for name, value in model.state_dict().items()
                        },
                    },
                )

        checkpoint = config.checkpoint_every > 0 and (
            step % config.checkpoint_every == 0 or step == config.steps
        )
        if checkpoint:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_allocated_now: int | None = max(
                    peak_allocated_before,
                    int(torch.cuda.max_memory_allocated(device)),
                )
                peak_reserved_now: int | None = max(
                    peak_reserved_before,
                    int(torch.cuda.max_memory_reserved(device)),
                )
            else:
                peak_allocated_now = peak_reserved_now = None
            checkpoint_started = time.perf_counter()
            atomic_torch_save(
                progress_path,
                {
                    "schema_version": "r2_dictionary_control_progress_v2",
                    "candidate_id": candidate_id,
                    "training_config": training_config_dict(config),
                    "stream_seed": stream_seed,
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "batch_stream_state": stream.state_dict(),
                    "rng_state": _capture_rng_state(),
                    "best_validation_fvu": best_fvu,
                    "best_validation_step": best_step,
                    "validation_history": validation_history,
                    "last_training_components": last_components,
                    "wall_time_seconds": elapsed_before + time.perf_counter() - started,
                    "peak_accelerator_memory_allocated_bytes": peak_allocated_now,
                    "peak_accelerator_memory_reserved_bytes": peak_reserved_now,
                },
            )
            checkpoint_io_seconds += time.perf_counter() - checkpoint_started
            write_json(
                timing_path,
                {
                    "schema_version": "r2_dictionary_control_timing_v1",
                    "candidate_id": candidate_id,
                    "step": step,
                    "progress_file_size_bytes": progress_path.stat().st_size,
                    "checkpoint_io_seconds": checkpoint_io_seconds,
                    "wall_time_seconds": elapsed_before + time.perf_counter() - started,
                },
            )

    if not best_path.is_file():
        raise RuntimeError("validation selection produced no best checkpoint")

    # A completed candidate no longer needs gradients or optimizer state. Release
    # them explicitly so CUDA tensors cannot survive through scheduler/optimizer
    # reference cycles into the next candidate in the same process.
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    optimizer.state.clear()
    del scheduler, optimizer

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if best.get("candidate_id") != candidate_id or best.get("step") != best_step:
        raise ValueError("best-checkpoint identity does not match training state")
    model.load_state_dict(best["model_state_dict"])
    model.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated: int | None = max(
            peak_allocated_before,
            int(torch.cuda.max_memory_allocated(device)),
        )
        peak_reserved: int | None = max(
            peak_reserved_before,
            int(torch.cuda.max_memory_reserved(device)),
        )
    else:
        peak_allocated = peak_reserved = None
    return {
        "resumed": resumed,
        "resume_start_step": start_step,
        "best_validation_step": best_step,
        "best_validation_fvu_mean_at_checkpoint_threshold": best_fvu,
        "checkpoint_activation_threshold": config.activation_threshold,
        "validation_history": validation_history,
        "last_training_components": last_components,
        "wall_time_seconds": elapsed_before + time.perf_counter() - started,
        "checkpoint_io_seconds": checkpoint_io_seconds,
        "wall_time_definition": (
            "cumulative_compute_and_progress_checkpoint_io_through_commit; "
            "excludes_atomic_timing_sidecar_bookkeeping"
        ),
        "peak_accelerator_memory_allocated_bytes": peak_allocated,
        "peak_accelerator_memory_reserved_bytes": peak_reserved,
        "process_max_rss_bytes": _max_rss_bytes(),
        "raw_trainable_parameter_count": trainable_parameter_count(model),
        "forward_matmul_macs_per_token": model.forward_matmul_macs_per_token(),
        "inference_flop_proxy_per_token": 2 * model.forward_matmul_macs_per_token(),
        "training_flop_proxy_per_token": model.training_flop_proxy_per_token(),
        "topk_selection_candidates_per_token": (
            model.n_layers * model.width if model.method == "topk_clt" else 0
        ),
        "deterministic_stream_seed": int(stream_seed),
        "progress_checkpoint": str(progress_path),
        "best_checkpoint": str(best_path),
        "test_split_accesses_during_training": 0,
    }


def atomic_torch_save(path: Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def model_seed(run_seed: int, layer: int, method: str) -> int:
    if method not in METHODS:
        raise ValueError(f"unknown dictionary method: {method}")
    return _derived_seed("dictionary-model", run_seed, layer, method)


def batch_stream_seed(run_seed: int, layer: int, cache_sha256: str) -> int:
    return _derived_seed("dictionary-batches", run_seed, layer, cache_sha256)


def training_config_dict(config: TrainingConfig) -> dict[str, Any]:
    config.validate()
    return asdict(config)


def load_production_profile(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load and validate the frozen P0-2 control profile fail-closed."""

    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("profile SHA-256 must be a lowercase digest")
    if sha256_file(path) != expected_sha256:
        raise ValueError("production-profile SHA-256 mismatch")
    profile = load_strict_json(path)
    if (
        profile.get("schema_version")
        != "r2_p0_2_dictionary_controls_production_profile_v2"
    ):
        raise ValueError("unsupported production-profile schema")
    if (
        profile.get("confirmatory") is not True
        or profile.get("status") != "amended_before_restart"
        or profile.get("amendment_reason")
        != "reproduced_nonfinite_fp16_frozen_model_activations_with_finite_bfloat16_replay"
    ):
        raise ValueError("production profile is not frozen and confirmatory")
    estimand = profile.get("estimand", {})
    if (
        estimand.get("objective")
        != "windowed_multi_layer_clt_input_to_mlp_output_transcoding"
        or estimand.get("source_tensor") != "architecture_specific_clt_input"
        or estimand.get("target_tensor") != "mlp_output"
        or estimand.get("decoder_window") != 8
    ):
        raise ValueError(
            "production profile does not encode the common window-8 estimand"
        )
    if profile.get("seeds") != [17, 29, 43]:
        raise ValueError("production profile seed panel changed")
    if profile.get("models") != ["protgpt2", "zymctrl", "progen2-medium"]:
        raise ValueError("production profile model panel changed")
    cache_profile = profile.get("cache_extraction", {})
    expected_budgets = {
        "train": 1_000_000,
        "validation": 100_000,
        "test": 100_000,
    }
    expected_formats = {
        "protgpt2": "sequence",
        "zymctrl": "zymctrl_ec",
        "progen2-medium": "sequence",
    }
    expected_geometry = {
        "protgpt2": {"n_layers": 36, "input_dim": 1280, "target_dim": 1280},
        "zymctrl": {"n_layers": 36, "input_dim": 1280, "target_dim": 1280},
        "progen2-medium": {
            "n_layers": 27,
            "input_dim": 1536,
            "target_dim": 1536,
        },
    }
    expected_tokenization = {
        "add_special_tokens": True,
        "padding": "longest",
        "padding_side": "right",
        "truncation_side": "right",
        "truncation": True,
        "max_model_tokens": 256,
    }
    if (
        cache_profile.get("selection_schema") != "r2_hash_priority_valid_tokens_v1"
        or cache_profile.get("selection_method")
        != "lowest_sha256_record_digest_colon_unpadded_token_position"
        or cache_profile.get("token_position_definition")
        != "ordinal_in_unpadded_model_token_stream"
        or cache_profile.get("valid_token_budget_by_split") != expected_budgets
        or cache_profile.get("model_input_format_by_model") != expected_formats
        or cache_profile.get("tokenization") != expected_tokenization
        or cache_profile.get("first_pass_tokenizer_batch_size") != 256
        or cache_profile.get("model_forward_sequence_batch_size") != 2
        or cache_profile.get("model_inference_dtype") != "bfloat16"
        or cache_profile.get("model_inference_dtype_verification")
        != "all_floating_model_parameters_exactly_declared_before_first_activation"
        or cache_profile.get("activation_finiteness_check")
        != "all_clt_input_and_mlp_output_tensors_before_storage_conversion_and_write"
        or cache_profile.get("storage_dtype") != "float16"
        or cache_profile.get("layout") != "preallocated_single_file_per_layer_split"
        or cache_profile.get("files_per_layer_split") != 2
        or cache_profile.get("free_space_safety_factor") != 1.2
        or cache_profile.get("model_cache_geometry") != expected_geometry
        or cache_profile.get("split_sequence_disjointness_required") is not True
        or cache_profile.get("exact_budget_failure_action") != "abort_without_cache"
    ):
        raise ValueError("production activation-cache contract changed")
    total_rows = sum(expected_budgets.values())
    expected_bytes = {
        model: estimate_activation_cache_bytes(
            valid_token_rows=total_rows,
            n_layers=geometry["n_layers"],
            input_dim=geometry["input_dim"],
            target_dim=geometry["target_dim"],
            storage_dtype="float16",
        )
        for model, geometry in expected_geometry.items()
    }
    expected_total = sum(expected_bytes.values())
    if (
        cache_profile.get("estimated_activation_payload_bytes_by_model")
        != expected_bytes
        or cache_profile.get("estimated_activation_payload_bytes_total")
        != expected_total
        or cache_profile.get("required_free_bytes_total_at_safety_factor")
        != math.ceil(expected_total * 1.2)
        or cache_profile.get("panel_free_space_gate_required_before_each_model")
        is not True
        or cache_profile.get("panel_completion_receipt_schema")
        != "r2_dictionary_cache_completion_receipt_v3"
    ):
        raise ValueError("production cache-size estimates changed or are incorrect")
    schedule = profile.get("compute_schedule", {})
    screening = schedule.get("screening", {})
    full = schedule.get("full", {})
    planning = schedule.get("planning_estimate", {})
    if (
        screening.get("seed") != 20260717
        or screening.get("train_cache_priority_prefix_rows") != 100_000
        or screening.get("steps_per_candidate") != 10_000
        or screening.get("warmup_steps") != 1_000
        or screening.get("validation_every_steps") != 1_000
        or screening.get("checkpoint_every_steps") != 1_000
        or screening.get("validation_rows") != 100_000
        or screening.get("relu_l1_candidates_per_model") != 5
        or screening.get("gated_candidates_per_model") != 10
        or screening.get("topk_candidates_per_model") != 0
        or screening.get("dense_candidates_per_model") != 0
        or screening.get("freeze_selected_coefficient_and_threshold_before_full_runs")
        is not True
        or screening.get("test_access_count") != 0
        or full.get("seeds") != [17, 29, 43]
        or full.get("steps_per_run") != 200_000
        or full.get(
            "screened_sparse_runs_use_frozen_screening_coefficient_and_threshold"
        )
        is not True
        or full.get("topk_runs_skip_screening") is not True
        or full.get("dense_runs_skip_screening") is not True
        or full.get("test_access_count_after_training") != 1
    ):
        raise ValueError("production two-stage compute schedule changed")
    required_planning = {
        "assumed_optimizer_steps_per_second",
        "screening_gpu_hours_per_candidate",
        "screening_candidates_per_model",
        "screening_gpu_hours_per_model",
        "screening_gpu_hours_all_models",
        "full_runs_per_model",
        "full_gpu_hours_per_run",
        "full_gpu_hours_per_model",
        "full_gpu_hours_all_models",
        "aggregate_gpu_hours_all_models",
        "screening_validation_passes_per_candidate",
        "full_validation_passes_per_run",
        "full_heldout_test_passes_per_run",
        "cache_extraction_and_evaluation_are_measured_not_in_step_rate_proxy",
        "basis",
    }
    if set(planning) != required_planning:
        raise ValueError("production profile lacks complete GPU-hour estimates")
    rate = planning["assumed_optimizer_steps_per_second"]
    screening_per_candidate = screening["steps_per_candidate"] / rate / 3600
    full_per_run = full["steps_per_run"] / rate / 3600
    if (
        rate != 1.0
        or not math.isclose(
            planning["screening_gpu_hours_per_candidate"], screening_per_candidate
        )
        or planning["screening_candidates_per_model"] != 15
        or not math.isclose(
            planning["screening_gpu_hours_per_model"],
            15 * screening_per_candidate,
        )
        or not math.isclose(
            planning["screening_gpu_hours_all_models"],
            45 * screening_per_candidate,
        )
        or planning["full_runs_per_model"] != 12
        or not math.isclose(planning["full_gpu_hours_per_run"], full_per_run)
        or not math.isclose(planning["full_gpu_hours_per_model"], 12 * full_per_run)
        or not math.isclose(planning["full_gpu_hours_all_models"], 36 * full_per_run)
        or not math.isclose(
            planning["aggregate_gpu_hours_all_models"],
            45 * screening_per_candidate + 36 * full_per_run,
        )
        or planning["screening_validation_passes_per_candidate"] != 10
        or planning["full_validation_passes_per_run"] != 40
        or planning["full_heldout_test_passes_per_run"] != 1
        or planning[
            "cache_extraction_and_evaluation_are_measured_not_in_step_rate_proxy"
        ]
        is not True
    ):
        raise ValueError("production GPU-hour estimates are internally inconsistent")
    preflight = profile.get("preflight", {})
    if preflight != {
        "mode": "bounded_nonconfirmatory_h200_preflight",
        "records_per_split": 2,
        "valid_token_rows_per_split": 2,
        "dictionary_optimizer_steps": 1,
        "dictionary_batch_valid_token_rows": 2,
        "dictionary_seed": 20260717,
        "production_scientific_eligibility": False,
        "production_cache_reuse_forbidden": True,
    }:
        raise ValueError("production profile lacks the bounded preflight contract")
    panel = profile.get("panel", {})
    if (
        panel.get("topk_clt", {}).get("width") != 8192
        or panel.get("topk_clt", {}).get("k") != 128
        or panel.get("relu_l1_sae", {}).get("width") != 8192
        or panel.get("gated_sae", {}).get("width") != 8192
        or panel.get("dense_low_rank", {}).get("rank") != 128
        or panel.get("dense_low_rank", {}).get("matched_quantity")
        != "active_bottleneck_width"
        or panel.get("dense_low_rank", {}).get("raw_parameter_matched") is not False
    ):
        raise ValueError("production dictionary panel changed")
    storage = profile.get("checkpoint_storage_planning", {})
    expected_storage_fields = {
        "parameter_state_definition",
        "retained_bytes_per_parameter_per_candidate",
        "atomic_progress_rewrite_temporary_bytes_per_parameter",
        "free_space_gate_includes_atomic_rewrite_temporary",
        "free_space_safety_factor_per_invocation",
        "raw_trainable_parameters_by_model_method",
        "retained_screening_checkpoint_bytes_by_model",
        "retained_screening_checkpoint_bytes_total",
        "retained_full_checkpoint_bytes_by_model",
        "retained_full_checkpoint_bytes_total",
        "retained_checkpoint_bytes_total",
        "worst_case_checkpoint_bytes_written_screening",
        "worst_case_checkpoint_bytes_written_full",
        "worst_case_checkpoint_bytes_written_total",
        "worst_case_definition",
    }
    if set(storage) != expected_storage_fields:
        raise ValueError(
            "production profile lacks complete checkpoint storage planning"
        )
    expected_parameters = {
        model: {
            method: windowed_transcoder_parameter_count(
                method=method,
                n_layers=expected_geometry[model]["n_layers"],
                input_dim=expected_geometry[model]["input_dim"],
                target_dim=expected_geometry[model]["target_dim"],
                sparse_width=8192,
                dense_rank=128,
                window=8,
            )
            for method in METHODS
        }
        for model in profile["models"]
    }
    retained_per_parameter = 16
    screening_bytes = {
        model: retained_per_parameter
        * (5 * counts["relu_l1_sae"] + 10 * counts["gated_sae"])
        for model, counts in expected_parameters.items()
    }
    full_bytes = {
        model: retained_per_parameter * 3 * sum(counts.values())
        for model, counts in expected_parameters.items()
    }
    screening_total = sum(screening_bytes.values())
    full_total = sum(full_bytes.values())
    if (
        storage["parameter_state_definition"]
        != "best_fp32_model_4_bytes_plus_progress_fp32_model_and_adam_states_12_bytes"
        or storage["retained_bytes_per_parameter_per_candidate"]
        != retained_per_parameter
        or storage["atomic_progress_rewrite_temporary_bytes_per_parameter"] != 12
        or storage["free_space_gate_includes_atomic_rewrite_temporary"] is not True
        or storage["free_space_safety_factor_per_invocation"] != 1.1
        or storage["raw_trainable_parameters_by_model_method"] != expected_parameters
        or storage["retained_screening_checkpoint_bytes_by_model"] != screening_bytes
        or storage["retained_screening_checkpoint_bytes_total"] != screening_total
        or storage["retained_full_checkpoint_bytes_by_model"] != full_bytes
        or storage["retained_full_checkpoint_bytes_total"] != full_total
        or storage["retained_checkpoint_bytes_total"] != screening_total + full_total
        or storage["worst_case_checkpoint_bytes_written_screening"]
        != screening_total * 10
        or storage["worst_case_checkpoint_bytes_written_full"] != full_total * 40
        or storage["worst_case_checkpoint_bytes_written_total"]
        != screening_total * 10 + full_total * 40
        or storage["worst_case_definition"]
        != "every scheduled validation improves best and every progress checkpoint is written"
    ):
        raise ValueError("production checkpoint storage planning is inconsistent")
    selection = profile.get("validation_only_selection", {})
    if (
        selection.get("eligible_l0_interval") != [115.2, 140.8]
        or selection.get("primary_selection_metric") != "validation_fvu_min"
        or selection.get("test_set_selection_forbidden") is not True
    ):
        raise ValueError("validation-only sparsity selection rule changed")
    training = profile.get("training", {})
    for key in (
        "total_steps",
        "train_batch_valid_token_rows",
        "evaluation_batch_valid_token_rows",
        "learning_rate",
        "warmup_steps",
        "validation_every_steps",
        "gradient_clip_norm",
        "checkpoint_every_steps",
    ):
        if key not in training:
            raise ValueError(f"production training profile lacks {key}")
    if (
        training.get("resume_requires_optimizer_scheduler_rng_and_data_cursor")
        is not True
    ):
        raise ValueError("production profile must require complete resumability")
    return profile


def validate_production_cache(
    cache: ActivationCache,
    profile: Mapping[str, Any],
    *,
    model_name: str,
    access_splits: Sequence[str] = SPLITS,
) -> dict[str, Any]:
    """Bind a transcode cache to the frozen model/hook/split provenance."""

    authorized_splits = tuple(access_splits)
    if (
        not authorized_splits
        or len(set(authorized_splits)) != len(authorized_splits)
        or not set(authorized_splits) <= set(cache.verified_splits)
    ):
        raise ValueError("production validation requested an unverified split")

    if model_name not in profile["models"]:
        raise ValueError(f"model is outside the frozen production panel: {model_name}")
    if cache.objective != "transcode":
        raise ValueError("production controls require a transcode activation cache")
    if cache.selected_layers != tuple(range(len(cache.selected_layers))):
        raise ValueError("production cache must contain all ordered model layers")
    cache_profile = profile["cache_extraction"]
    geometry = cache_profile["model_cache_geometry"][model_name]
    expected_dimensions = (geometry["input_dim"], geometry["target_dim"])
    if len(cache.selected_layers) != geometry["n_layers"] or any(
        cache.dimensions[layer] != expected_dimensions
        for layer in cache.selected_layers
    ):
        raise ValueError("production cache geometry does not match the frozen model")
    if (
        cache.payload.get("storage_dtype") != cache_profile["storage_dtype"]
        or cache.payload.get("layout") != cache_profile["layout"]
        or any(
            sum(
                shard["split"] == split and shard["layer"] == layer
                for shard in cache.shards
            )
            != 1
            for split in SPLITS
            for layer in cache.selected_layers
        )
    ):
        raise ValueError("production cache storage configuration changed")
    selection = cache.payload.get("token_selection", {})
    if (
        selection.get("status") != "complete_budgeted_selection"
        or selection.get("method") != cache_profile["selection_method"]
        or selection.get("position_definition")
        != cache_profile["token_position_definition"]
        or selection.get("serialization_order") != "selection_priority_order"
    ):
        raise ValueError("production cache lacks the frozen token selection")
    for split, budget in cache_profile["valid_token_budget_by_split"].items():
        if (
            selection.get("by_split", {}).get(split, {}).get("budget") != budget
            or cache.payload["split_summaries"][split]["selected_valid_token_rows"]
            != budget
        ):
            raise ValueError(f"production cache budget mismatch for {split}")
    expected_bytes = cache_profile["estimated_activation_payload_bytes_by_model"][
        model_name
    ]
    capacity = cache.payload.get("capacity_check")
    if (
        not isinstance(capacity, dict)
        or set(capacity)
        != {
            "estimated_bytes",
            "safety_factor",
            "required_free_bytes",
            "observed_free_bytes",
        }
        or capacity["estimated_bytes"] != expected_bytes
        or capacity["safety_factor"] != cache_profile["free_space_safety_factor"]
        or capacity["required_free_bytes"] != math.ceil(expected_bytes * 1.2)
        or capacity["observed_free_bytes"] < capacity["required_free_bytes"]
    ):
        raise ValueError("production cache lacks a valid frozen free-space gate")
    provenance = cache.payload.get("activation_provenance", {})
    required = {
        "schema_version",
        "status",
        "model_name",
        "model_revision",
        "model_config_sha256",
        "model_weights_sha256",
        "tokenizer_sha256",
        "model_artifact_hash_definition",
        "command",
        "pod_name",
        "node_name",
        "gpu_index",
        "gpu_model",
        "git_commit",
        "git_dirty",
        "python_version",
        "torch_version",
        "cuda_runtime",
        "started_at_utc",
        "finished_at_utc",
        "wall_time_seconds",
        "peak_accelerator_memory_allocated_bytes",
        "peak_accelerator_memory_reserved_bytes",
        "panel_capacity_check",
        "code_archive_sha256",
        "code_content_manifest_sha256",
        "code_content_inventory_verified",
        "model_loader_sha256",
        "extraction_script_sha256",
        "model_inference_dtype",
        "observed_model_parameter_dtypes",
        "model_inference_dtype_verification",
        "model_inference_dtype_verified",
        "activation_finiteness_check",
        "tokenization_config_sha256",
        "tokenization_config",
        "runtime_tokenizer",
        "model_input_format",
        "token_selection_method",
        "valid_token_budget_by_split",
        "eligible_valid_token_rows_by_split",
        "cache_storage_dtype",
        "feature_input_hook",
        "feature_output_hook",
        "concrete_hook_contract",
        "decoder_window",
        "selected_layers",
        "observed_n_layers",
        "observed_input_dim",
        "observed_target_dim",
        "source_manifest_sha256_by_split",
        "first_pass_tokenization_sha256_by_split",
        "second_pass_tokenization_sha256_by_split",
        "executed_profile_sha256",
    }
    if not isinstance(provenance, dict) or set(provenance) != required:
        raise ValueError("production activation provenance fields are incomplete")
    if provenance["schema_version"] != "r2_dictionary_activation_provenance_v3":
        raise ValueError("unsupported activation-provenance schema")
    if provenance["status"] != "complete" or provenance["model_name"] != model_name:
        raise ValueError("activation provenance status/model mismatch")
    if provenance["code_content_inventory_verified"] is not True:
        raise ValueError("activation provenance lacks verified code inventory")
    panel_capacity = provenance["panel_capacity_check"]
    if (
        not isinstance(panel_capacity, dict)
        or panel_capacity.get("scope") != "remaining_incomplete_panel_payload"
        or panel_capacity.get("safety_factor")
        != cache_profile["free_space_safety_factor"]
        or panel_capacity.get("observed_free_bytes", 0)
        < panel_capacity.get("required_free_bytes", 1)
    ):
        raise ValueError("activation provenance lacks a valid panel capacity gate")
    if provenance["model_artifact_hash_definition"] != (
        "config_json_file_sha256; canonical_path_size_sha256_tree_for_weights; "
        "canonical_path_size_sha256_tree_for_remaining_tokenizer_support_files"
    ):
        raise ValueError("unsupported model-artifact hash definition")
    estimand = profile["estimand"]
    if (
        provenance["feature_input_hook"] != estimand["feature_input_hook"]
        or provenance["feature_output_hook"] != estimand["feature_output_hook"]
        or provenance["decoder_window"] != estimand["decoder_window"]
        or provenance["selected_layers"] != list(cache.selected_layers)
    ):
        raise ValueError("activation provenance does not match the frozen estimand")
    if (
        provenance["concrete_hook_contract"]
        != "transformer_block.mlp_input_and_mlp_output_indexed_by_declared_layer"
        or provenance["observed_n_layers"] != geometry["n_layers"]
        or provenance["observed_input_dim"] != geometry["input_dim"]
        or provenance["observed_target_dim"] != geometry["target_dim"]
        or provenance["model_loader_sha256"]
        != sha256_file(Path(__file__).parents[1] / "models/model_loader.py")
    ):
        raise ValueError("activation hook/model-loader provenance mismatch")
    if (
        provenance["model_inference_dtype"] != cache_profile["model_inference_dtype"]
        or provenance["observed_model_parameter_dtypes"] != ["bfloat16"]
        or provenance["model_inference_dtype_verification"]
        != cache_profile["model_inference_dtype_verification"]
        or provenance["model_inference_dtype_verified"] is not True
        or provenance["activation_finiteness_check"]
        != cache_profile["activation_finiteness_check"]
        or provenance["model_input_format"]
        != cache_profile["model_input_format_by_model"][model_name]
        or provenance["tokenization_config"] != cache_profile["tokenization"]
        or provenance["token_selection_method"] != cache_profile["selection_method"]
        or provenance["valid_token_budget_by_split"]
        != cache_profile["valid_token_budget_by_split"]
        or provenance["cache_storage_dtype"] != cache_profile["storage_dtype"]
        or provenance["tokenization_config_sha256"]
        != _identity_sha256(cache_profile["tokenization"])
    ):
        raise ValueError(
            "activation provenance does not match cache extraction profile"
        )
    runtime_tokenizer = provenance["runtime_tokenizer"]
    if (
        not isinstance(runtime_tokenizer, dict)
        or runtime_tokenizer.get("padding_side") != "right"
        or runtime_tokenizer.get("truncation_side") != "right"
        or type(runtime_tokenizer.get("pad_token_id")) is not int
        or not isinstance(runtime_tokenizer.get("special_tokens_map"), dict)
    ):
        raise ValueError("runtime tokenizer provenance violates the frozen contract")
    if (
        provenance["first_pass_tokenization_sha256_by_split"]
        != provenance["second_pass_tokenization_sha256_by_split"]
        or set(provenance["first_pass_tokenization_sha256_by_split"]) != set(SPLITS)
        or provenance["eligible_valid_token_rows_by_split"]
        != {
            split: cache.payload["split_summaries"][split]["eligible_valid_token_rows"]
            for split in SPLITS
        }
    ):
        raise ValueError("first/second-pass tokenization provenance mismatch")
    source_hashes = {
        split: cache.payload["source_splits"][split]["manifest_sha256"]
        for split in SPLITS
    }
    if provenance["source_manifest_sha256_by_split"] != source_hashes:
        raise ValueError("activation provenance source hashes disagree with cache")
    source_sequence_hashes: set[str] = set()
    for split in authorized_splits:
        source_path = _resolve_member(
            cache.manifest_path.parent,
            cache.payload["source_splits"][split]["manifest_path"],
        )
        split_hashes: set[str] = set()
        for row in load_strict_jsonl(source_path):
            if not isinstance(row, dict) or row.get("split") != split:
                raise ValueError(f"invalid frozen source record in {split}")
            format_model_input(row, provenance["model_input_format"])
            digest = row["sha256"]
            if digest in split_hashes or digest in source_sequence_hashes:
                raise ValueError("source manifests are not sequence-disjoint")
            split_hashes.add(digest)
            source_sequence_hashes.add(digest)
        selection_path = _resolve_member(
            cache.manifest_path.parent,
            selection["by_split"][split]["selection_path"],
        )
        for selected in load_strict_jsonl(selection_path):
            if selected["record_sha256"] not in split_hashes:
                raise ValueError(
                    f"selected token is outside the frozen {split} source manifest"
                )
    for field in (
        "model_config_sha256",
        "model_weights_sha256",
        "tokenizer_sha256",
        "model_loader_sha256",
        "extraction_script_sha256",
        "tokenization_config_sha256",
        "executed_profile_sha256",
        "code_archive_sha256",
        "code_content_manifest_sha256",
    ):
        value = provenance[field]
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"invalid activation-provenance digest: {field}")
    return provenance
