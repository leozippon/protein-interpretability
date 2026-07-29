#!/usr/bin/env python3
"""Build one frozen, budgeted P0-2 activation cache from cohort manifests."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import torch

R2_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
SPLITS = ("train", "validation", "test")
_LOCAL_CODE_LOADED = False
assert_finite_captured_activations: Any = None
inference_dtype: Any = None
load_model: Any = None
verify_frozen_model_inference_dtype: Any = None
format_model_input: Any = None
load_activation_cache: Any = None
load_production_profile: Any = None
load_strict_json: Any = None
load_strict_jsonl: Any = None
require_cache_free_space: Any = None
select_hash_priority_tokens: Any = None
validate_production_cache: Any = None
write_activation_cache: Any = None
write_json: Any = None
write_jsonl: Any = None


def _load_verified_local_code() -> None:
    """Import project modules only after the deployed code tree is verified."""

    global _LOCAL_CODE_LOADED
    if _LOCAL_CODE_LOADED:
        return
    if str(R2_ROOT) not in sys.path:
        sys.path.insert(0, str(R2_ROOT))
    from src.models.model_loader import (  # noqa: PLC0415
        assert_finite_captured_activations,
        inference_dtype,
        load_model,
        verify_frozen_model_inference_dtype,
    )
    from src.revision.dictionary_controls import (  # noqa: PLC0415
        SPLITS as imported_splits,
        format_model_input,
        load_activation_cache,
        load_production_profile,
        load_strict_json,
        load_strict_jsonl,
        require_cache_free_space,
        select_hash_priority_tokens,
        validate_production_cache,
        write_activation_cache,
    )
    from src.revision.io import write_json, write_jsonl  # noqa: PLC0415

    if tuple(imported_splits) != SPLITS:
        raise ValueError("local split contract changed")
    globals().update(
        {
            "assert_finite_captured_activations": assert_finite_captured_activations,
            "inference_dtype": inference_dtype,
            "load_model": load_model,
            "verify_frozen_model_inference_dtype": verify_frozen_model_inference_dtype,
            "format_model_input": format_model_input,
            "load_activation_cache": load_activation_cache,
            "load_production_profile": load_production_profile,
            "load_strict_json": load_strict_json,
            "load_strict_jsonl": load_strict_jsonl,
            "require_cache_free_space": require_cache_free_space,
            "select_hash_priority_tokens": select_hash_priority_tokens,
            "validate_production_cache": validate_production_cache,
            "write_activation_cache": write_activation_cache,
            "write_json": write_json,
            "write_jsonl": write_jsonl,
        }
    )
    _LOCAL_CODE_LOADED = True


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _chunks(rows: list[Any], size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _parse_code_manifest(payload: bytes, manifest_name: str) -> dict[str, str]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("code content manifest is not UTF-8") from error
    expected: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if len(line) < 68 or line[64:66] != "  ":
            raise ValueError(f"invalid code manifest line {line_number}")
        digest, declared = line[:64], line[66:]
        _require_sha256(digest, f"code manifest line {line_number} digest")
        if not declared.startswith("./") or "\\" in declared:
            raise ValueError(f"unsafe code manifest path on line {line_number}")
        relative_text = declared[2:]
        relative = PurePosixPath(relative_text)
        if (
            not relative_text
            or relative.is_absolute()
            or relative.as_posix() != relative_text
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative_text == manifest_name
            or relative_text in expected
        ):
            raise ValueError(f"unsafe code manifest path on line {line_number}")
        expected[relative_text] = digest
    if not expected:
        raise ValueError("code content manifest is empty")
    return expected


def _verify_archive_tree(
    archive: Path,
    *,
    manifest_bytes: bytes,
    manifest_name: str,
    expected: dict[str, str],
) -> None:
    archive_root = "r2_interpretability_transfer"
    archived_files: dict[str, bytes] = {}
    seen: set[str] = set()
    try:
        with tarfile.open(archive, mode="r:*") as handle:
            for member in handle.getmembers():
                name = (
                    member.name[:-1]
                    if member.isdir() and member.name.endswith("/")
                    else member.name
                )
                path = PurePosixPath(name)
                if (
                    not name
                    or path.is_absolute()
                    or path.as_posix() != name
                    or any(part in {"", ".", ".."} for part in path.parts)
                    or path.parts[0] != archive_root
                    or name in seen
                ):
                    raise ValueError("unsafe or duplicate code archive member")
                seen.add(name)
                if len(path.parts) == 1:
                    if not member.isdir():
                        raise ValueError("code archive root is not a directory")
                    continue
                if member.issym() or member.islnk():
                    raise ValueError("code archive contains a link")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError("code archive contains a non-regular member")
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise ValueError("code archive member cannot be read")
                archived_files[PurePosixPath(*path.parts[1:]).as_posix()] = (
                    extracted.read()
                )
    except tarfile.TarError as error:
        raise ValueError("code archive is not a valid tar archive") from error
    if set(archived_files) != {*expected, manifest_name}:
        raise ValueError("code archive inventory differs from manifest")
    if archived_files[manifest_name] != manifest_bytes:
        raise ValueError("code archive manifest differs from deployed manifest")
    for relative_text, digest in expected.items():
        if hashlib.sha256(archived_files[relative_text]).hexdigest() != digest:
            raise ValueError(f"code archive content mismatch: ./{relative_text}")


def verify_code_binding(
    *,
    archive_path: Path,
    archive_sha256: str,
    content_manifest_path: Path,
    content_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify the frozen archive and the exact deployed code-tree inventory."""

    _require_sha256(archive_sha256, "code archive SHA-256")
    _require_sha256(content_manifest_sha256, "code content manifest SHA-256")
    archive = Path(archive_path)
    manifest = Path(content_manifest_path)
    expected_manifest = (R2_ROOT / "CODE_CONTENT_SHA256SUMS").resolve()
    if manifest.resolve() != expected_manifest:
        raise ValueError("code content manifest is not the runner project manifest")
    if archive.is_symlink() or not archive.is_file():
        raise FileNotFoundError(f"code archive is not a regular file: {archive}")
    if _sha256_file(archive) != archive_sha256:
        raise ValueError("code archive SHA-256 mismatch")
    if manifest.is_symlink() or not manifest.is_file():
        raise FileNotFoundError(
            f"code content manifest is not a regular file: {manifest}"
        )
    if _sha256_file(manifest) != content_manifest_sha256:
        raise ValueError("code content manifest SHA-256 mismatch")

    root = manifest.parent.resolve()
    manifest_bytes = manifest.read_bytes()
    expected = _parse_code_manifest(manifest_bytes, manifest.name)
    _verify_archive_tree(
        archive,
        manifest_bytes=manifest_bytes,
        manifest_name=manifest.name,
        expected=expected,
    )

    actual: set[str] = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            if (directory_path / name).is_symlink():
                raise ValueError("deployed code tree contains a symbolic link")
        for name in file_names:
            path = directory_path / name
            if path.is_symlink():
                raise ValueError("deployed code tree contains a symbolic link")
            if not path.is_file():
                raise ValueError("deployed code tree contains a non-regular file")
            if path == manifest.resolve():
                continue
            actual.add(path.relative_to(root).as_posix())
    if actual != set(expected):
        raise ValueError("deployed code-tree inventory differs from manifest")
    for relative_text, digest in expected.items():
        path = root.joinpath(*PurePosixPath(relative_text).parts)
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError("code manifest path escapes deployed root") from error
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != digest:
            raise ValueError(f"deployed code content mismatch: ./{relative_text}")
    return {
        "code_archive_sha256": archive_sha256,
        "code_content_manifest_sha256": content_manifest_sha256,
        "code_content_inventory_verified": True,
    }


def _tree_digest(root: Path, paths: list[Path]) -> str:
    if not paths:
        raise ValueError(f"empty artifact file class under {root}")
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(paths)
    ]
    return _digest(entries)


def verify_model_artifacts(
    model_root: Path,
    expected: dict[str, str],
) -> dict[str, str]:
    """Bind declared config/weight/tokenizer digests to deployed local files."""

    root = Path(model_root).resolve()
    config = root / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"missing model config: {config}")
    files = [path for path in root.rglob("*") if path.is_file()]
    weights = [
        path
        for path in files
        if path.suffix == ".safetensors"
        or path.name.startswith("pytorch_model")
        and path.suffix in {".bin", ".json"}
        or path.name.startswith("model.safetensors")
    ]
    support = [path for path in files if path != config and path not in weights]
    observed = {
        "model_config_sha256": _sha256_file(config),
        "model_weights_sha256": _tree_digest(root, weights),
        "tokenizer_sha256": _tree_digest(root, support),
    }
    for field, digest in observed.items():
        _require_sha256(expected[field], field)
        if digest != expected[field]:
            raise ValueError(f"deployed artifact digest mismatch: {field}")
    return {
        **observed,
        "model_artifact_hash_definition": (
            "config_json_file_sha256; canonical_path_size_sha256_tree_for_weights; "
            "canonical_path_size_sha256_tree_for_remaining_tokenizer_support_files"
        ),
    }


def enforce_panel_cache_capacity(
    *,
    panel_cache_root: Path,
    output_dir: Path,
    profile: dict[str, Any],
    profile_sha256: str,
    model_name: str,
    code_binding: dict[str, Any],
) -> dict[str, Any]:
    """Gate remaining panel payload, crediting only verified completion receipts."""

    root = Path(panel_cache_root).resolve()
    expected_output = root / model_name
    if Path(output_dir).resolve() != expected_output:
        raise ValueError("production output must be <panel-cache-root>/<model-name>")
    cache_profile = profile["cache_extraction"]
    receipt_schema = cache_profile["panel_completion_receipt_schema"]
    if set(code_binding) != {
        "code_archive_sha256",
        "code_content_manifest_sha256",
        "code_content_inventory_verified",
    } or code_binding["code_content_inventory_verified"] is not True:
        raise ValueError("current code binding is incomplete")
    for field in ("code_archive_sha256", "code_content_manifest_sha256"):
        _require_sha256(code_binding[field], field)
    completed: list[str] = []
    remaining_bytes = 0
    for panel_model in profile["models"]:
        payload_bytes = cache_profile["estimated_activation_payload_bytes_by_model"][
            panel_model
        ]
        receipt_path = root / panel_model / "completion_receipt.json"
        if not receipt_path.is_file():
            remaining_bytes += payload_bytes
            continue
        receipt = load_strict_json(receipt_path)
        required = {
            "schema_version",
            "status",
            "model_name",
            "profile_sha256",
            "cache_manifest_path",
            "cache_manifest_sha256",
            "cache_content_sha256",
            "activation_payload_bytes",
            "model_inference_dtype",
            "observed_model_parameter_dtypes",
            "model_inference_dtype_verification",
            "model_inference_dtype_verified",
            "activation_finiteness_check",
            "cache_storage_dtype",
            "code_archive_sha256",
            "code_content_manifest_sha256",
            "code_content_inventory_verified",
            "execution_report_path",
            "execution_report_sha256",
            "completed_at_utc",
        }
        manifest_path = root / panel_model / "manifest.json"
        execution_report_path = root / panel_model / "cache_execution_report.json"
        if (
            not isinstance(receipt, dict)
            or set(receipt) != required
            or receipt["schema_version"] != receipt_schema
            or receipt["status"] != "verified_complete"
            or receipt["model_name"] != panel_model
            or receipt["profile_sha256"] != profile_sha256
            or Path(receipt["cache_manifest_path"]).resolve() != manifest_path
            or receipt["activation_payload_bytes"] != payload_bytes
            or receipt["model_inference_dtype"]
            != cache_profile["model_inference_dtype"]
            or receipt["observed_model_parameter_dtypes"] != ["bfloat16"]
            or receipt["model_inference_dtype_verification"]
            != cache_profile["model_inference_dtype_verification"]
            or receipt["model_inference_dtype_verified"] is not True
            or receipt["activation_finiteness_check"]
            != cache_profile["activation_finiteness_check"]
            or receipt["cache_storage_dtype"] != cache_profile["storage_dtype"]
            or receipt["code_archive_sha256"]
            != code_binding["code_archive_sha256"]
            or receipt["code_content_manifest_sha256"]
            != code_binding["code_content_manifest_sha256"]
            or receipt["code_content_inventory_verified"] is not True
            or not manifest_path.is_file()
            or _sha256_file(manifest_path) != receipt["cache_manifest_sha256"]
            or Path(receipt["execution_report_path"]).resolve()
            != execution_report_path
            or not execution_report_path.is_file()
            or _sha256_file(execution_report_path)
            != receipt["execution_report_sha256"]
        ):
            raise ValueError(f"invalid panel completion receipt for {panel_model}")
        manifest_payload = load_strict_json(manifest_path)
        if manifest_payload.get("content_sha256") != receipt["cache_content_sha256"]:
            raise ValueError(f"completion receipt content mismatch for {panel_model}")
        provenance = manifest_payload.get("activation_provenance", {})
        execution_report = load_strict_json(execution_report_path)
        if (
            not isinstance(provenance, dict)
            or provenance.get("schema_version")
            != "r2_dictionary_activation_provenance_v3"
            or not isinstance(execution_report, dict)
            or execution_report.get("schema_version")
            != "r2_dictionary_cache_execution_report_v3"
            or execution_report.get("status") != "verified_complete"
            or execution_report.get("model_name") != panel_model
            or execution_report.get("profile_sha256") != profile_sha256
            or execution_report.get("cache_manifest_sha256")
            != receipt["cache_manifest_sha256"]
            or execution_report.get("cache_content_sha256")
            != receipt["cache_content_sha256"]
            or any(
                provenance.get(field) != code_binding[field]
                or execution_report.get(field) != code_binding[field]
                for field in code_binding
            )
        ):
            raise ValueError(f"completion receipt code binding mismatch for {panel_model}")
        completed.append(panel_model)
    if model_name in completed:
        raise FileExistsError(f"verified cache already exists for {model_name}")
    capacity = require_cache_free_space(
        root / ".pending",
        estimated_bytes=remaining_bytes,
        safety_factor=cache_profile["free_space_safety_factor"],
    )
    return {
        **capacity,
        "scope": "remaining_incomplete_panel_payload",
        "completed_models": completed,
        "remaining_models": [
            model for model in profile["models"] if model not in completed
        ],
    }


def write_completion_receipt(
    *,
    cache,
    profile: dict[str, Any],
    profile_sha256: str,
    model_name: str,
    execution_report_path: Path,
) -> Path:
    provenance = cache.payload["activation_provenance"]
    execution_report = load_strict_json(execution_report_path)
    code_binding = {
        field: provenance[field]
        for field in (
            "code_archive_sha256",
            "code_content_manifest_sha256",
            "code_content_inventory_verified",
        )
    }
    if (
        provenance.get("schema_version") != "r2_dictionary_activation_provenance_v3"
        or execution_report.get("schema_version")
        != "r2_dictionary_cache_execution_report_v3"
        or any(execution_report.get(field) != value for field, value in code_binding.items())
        or code_binding["code_content_inventory_verified"] is not True
    ):
        raise ValueError("cannot issue receipt for an invalid code binding")
    receipt_path = cache.manifest_path.parent / "completion_receipt.json"
    write_json(
        receipt_path,
        {
            "schema_version": profile["cache_extraction"][
                "panel_completion_receipt_schema"
            ],
            "status": "verified_complete",
            "model_name": model_name,
            "profile_sha256": profile_sha256,
            "cache_manifest_path": str(cache.manifest_path),
            "cache_manifest_sha256": cache.manifest_sha256,
            "cache_content_sha256": cache.content_sha256,
            "activation_payload_bytes": profile["cache_extraction"][
                "estimated_activation_payload_bytes_by_model"
            ][model_name],
            "model_inference_dtype": cache.payload["activation_provenance"][
                "model_inference_dtype"
            ],
            "observed_model_parameter_dtypes": cache.payload[
                "activation_provenance"
            ]["observed_model_parameter_dtypes"],
            "model_inference_dtype_verification": cache.payload[
                "activation_provenance"
            ]["model_inference_dtype_verification"],
            "model_inference_dtype_verified": cache.payload[
                "activation_provenance"
            ]["model_inference_dtype_verified"],
            "activation_finiteness_check": cache.payload["activation_provenance"][
                "activation_finiteness_check"
            ],
            "cache_storage_dtype": cache.payload["storage_dtype"],
            **code_binding,
            "execution_report_path": str(Path(execution_report_path).resolve()),
            "execution_report_sha256": _sha256_file(execution_report_path),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return receipt_path


def build_preflight_profile(
    profile: dict[str, Any],
    *,
    output_dir: Path,
) -> tuple[dict[str, Any], str]:
    """Archive a tiny derived profile that production validation will reject."""

    derived = copy.deepcopy(profile)
    rows = profile["preflight"]["valid_token_rows_per_split"]
    derived["confirmatory"] = False
    derived["status"] = "bounded_nonconfirmatory_h200_preflight"
    derived["cache_extraction"]["valid_token_budget_by_split"] = {
        split: rows for split in SPLITS
    }
    derived["cache_extraction"]["free_space_safety_factor"] = 1.0
    for model, geometry in derived["cache_extraction"][
        "model_cache_geometry"
    ].items():
        derived["cache_extraction"]["estimated_activation_payload_bytes_by_model"][
            model
        ] = (
            rows
            * len(SPLITS)
            * geometry["n_layers"]
            * (geometry["input_dim"] + geometry["target_dim"])
            * 2
        )
    estimates = derived["cache_extraction"][
        "estimated_activation_payload_bytes_by_model"
    ]
    derived["cache_extraction"]["estimated_activation_payload_bytes_total"] = sum(
        estimates.values()
    )
    derived["cache_extraction"][
        "required_free_bytes_total_at_safety_factor"
    ] = sum(estimates.values())
    path = Path(output_dir) / "executed_preflight_profile.json"
    write_json(path, derived)
    return derived, _sha256_file(path)


def prepare_preflight_sources(
    source_splits: dict[str, dict[str, str]],
    *,
    output_dir: Path,
    records_per_split: int,
) -> dict[str, dict[str, str]]:
    destination = Path(output_dir) / "source_manifests"
    result: dict[str, dict[str, str]] = {}
    for split in SPLITS:
        source = Path(source_splits[split]["manifest_path"])
        if _sha256_file(source) != source_splits[split]["manifest_sha256"]:
            raise ValueError(f"source manifest SHA-256 mismatch for {split}")
        rows = load_strict_jsonl(source)
        if len(rows) < records_per_split:
            raise ValueError(f"preflight source lacks {records_per_split} {split} rows")
        path = destination / f"{split}.jsonl"
        write_jsonl(path, rows[:records_per_split])
        result[split] = {
            "manifest_path": str(path),
            "manifest_sha256": _sha256_file(path),
        }
    return result


def build_cache_from_model(
    *,
    protein_model,
    model_name: str,
    profile: dict[str, Any],
    profile_sha256: str,
    source_splits: dict[str, dict[str, str]],
    output_dir: Path,
    model_provenance: dict[str, str],
    execution_provenance: dict[str, Any],
) -> Path:
    """Run strict manifest/tokenizer pass, then aligned activation extraction."""

    extraction_started = time.perf_counter()
    cache_profile = profile["cache_extraction"]
    if model_name not in profile["models"]:
        raise ValueError("model is outside the frozen panel")
    if set(source_splits) != set(SPLITS):
        raise ValueError(f"source_splits must contain exactly {SPLITS}")
    for field in (
        "model_revision",
        "model_config_sha256",
        "model_weights_sha256",
        "tokenizer_sha256",
    ):
        if field not in model_provenance:
            raise ValueError(f"model provenance lacks {field}")
        if field != "model_revision":
            _require_sha256(model_provenance[field], field)
    for field in ("code_archive_sha256", "code_content_manifest_sha256"):
        if field not in execution_provenance:
            raise ValueError(f"execution provenance lacks {field}")
        _require_sha256(execution_provenance[field], field)
    if execution_provenance.get("code_content_inventory_verified") is not True:
        raise ValueError("execution provenance lacks verified code inventory")

    geometry = cache_profile["model_cache_geometry"][model_name]
    if (
        protein_model.n_layers != geometry["n_layers"]
        or protein_model.d_model != geometry["input_dim"]
        or geometry["input_dim"] != geometry["target_dim"]
    ):
        raise ValueError("loaded model geometry disagrees with the frozen profile")
    dtype_receipt = verify_frozen_model_inference_dtype(
        protein_model,
        cache_profile["model_inference_dtype"],
    )
    tokenizer = protein_model.tokenizer
    tokenization = cache_profile["tokenization"]
    tokenizer.padding_side = tokenization["padding_side"]
    tokenizer.truncation_side = tokenization["truncation_side"]
    if (
        tokenizer.padding_side != "right"
        or tokenizer.truncation_side != "right"
        or tokenizer.pad_token_id is None
    ):
        raise ValueError("tokenizer cannot satisfy the frozen right-padding contract")

    input_format = cache_profile["model_input_format_by_model"][model_name]
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    seen_ids: set[str] = set()
    seen_sequences: set[str] = set()
    for split in SPLITS:
        path = Path(source_splits[split]["manifest_path"])
        expected_hash = source_splits[split]["manifest_sha256"]
        _require_sha256(expected_hash, f"{split} manifest hash")
        if _sha256_file(path) != expected_hash:
            raise ValueError(f"source manifest SHA-256 mismatch for {split}")
        rows = load_strict_jsonl(path)
        for row in rows:
            if not isinstance(row, dict) or row.get("split") != split:
                raise ValueError(f"invalid source row in {split}")
            format_model_input(row, input_format)
            if row["id"] in seen_ids or row["sha256"] in seen_sequences:
                raise ValueError("source manifests are not ID/sequence-disjoint")
            seen_ids.add(row["id"])
            seen_sequences.add(row["sha256"])
        records_by_split[split] = rows

    first_fingerprints: dict[str, str] = {}
    first_record_fingerprints: dict[str, dict[str, str]] = {}
    eligible_counts: dict[str, int] = {}
    selected_tokens: dict[str, list[dict[str, Any]]] = {}
    tokenizer_batch = cache_profile["first_pass_tokenizer_batch_size"]
    for split in SPLITS:
        aggregate = hashlib.sha256()
        per_record: dict[str, str] = {}
        counts: list[tuple[str, int]] = []
        for batch in _chunks(records_by_split[split], tokenizer_batch):
            texts = [format_model_input(row, input_format) for row in batch]
            encoded = tokenizer(
                texts,
                add_special_tokens=tokenization["add_special_tokens"],
                padding=False,
                truncation=tokenization["truncation"],
                max_length=tokenization["max_model_tokens"],
                return_attention_mask=True,
            )
            for row, input_ids, attention_mask in zip(
                batch, encoded["input_ids"], encoded["attention_mask"]
            ):
                input_ids = [int(value) for value in input_ids]
                attention_mask = [int(value) for value in attention_mask]
                if (
                    not input_ids
                    or len(input_ids) != len(attention_mask)
                    or any(value != 1 for value in attention_mask)
                ):
                    raise ValueError("unpadded tokenizer pass returned an invalid mask")
                fingerprint = _digest(
                    {"record_sha256": row["sha256"], "input_ids": input_ids}
                )
                per_record[row["sha256"]] = fingerprint
                aggregate.update(f"{row['sha256']}:{fingerprint}\n".encode("ascii"))
                counts.append((row["sha256"], len(input_ids)))
        eligible_counts[split] = sum(count for _, count in counts)
        first_fingerprints[split] = aggregate.hexdigest()
        first_record_fingerprints[split] = per_record
        selected_tokens[split] = select_hash_priority_tokens(
            counts,
            budget=cache_profile["valid_token_budget_by_split"][split],
        )

    runtime_tokenizer = {
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "special_tokens_map": tokenizer.special_tokens_map,
    }
    provenance: dict[str, Any] = {
        "schema_version": "r2_dictionary_activation_provenance_v3",
        "status": "complete",
        "model_name": model_name,
        **model_provenance,
        **execution_provenance,
        "model_loader_sha256": _sha256_file(R2_ROOT / "src/models/model_loader.py"),
        "extraction_script_sha256": _sha256_file(Path(__file__)),
        **dtype_receipt,
        "activation_finiteness_check": cache_profile[
            "activation_finiteness_check"
        ],
        "tokenization_config_sha256": _digest(tokenization),
        "tokenization_config": tokenization,
        "runtime_tokenizer": runtime_tokenizer,
        "model_input_format": input_format,
        "token_selection_method": cache_profile["selection_method"],
        "valid_token_budget_by_split": cache_profile["valid_token_budget_by_split"],
        "eligible_valid_token_rows_by_split": eligible_counts,
        "cache_storage_dtype": cache_profile["storage_dtype"],
        "feature_input_hook": profile["estimand"]["feature_input_hook"],
        "feature_output_hook": profile["estimand"]["feature_output_hook"],
        "concrete_hook_contract": "transformer_block.mlp_input_and_mlp_output_indexed_by_declared_layer",
        "decoder_window": profile["estimand"]["decoder_window"],
        "selected_layers": list(range(protein_model.n_layers)),
        "observed_n_layers": protein_model.n_layers,
        "observed_input_dim": protein_model.d_model,
        "observed_target_dim": protein_model.d_model,
        "source_manifest_sha256_by_split": {
            split: source_splits[split]["manifest_sha256"] for split in SPLITS
        },
        "first_pass_tokenization_sha256_by_split": first_fingerprints,
        "second_pass_tokenization_sha256_by_split": {},
        "executed_profile_sha256": profile_sha256,
    }

    def activation_batches(split: str):
        aggregate = hashlib.sha256()
        for batch in _chunks(
            records_by_split[split],
            cache_profile["model_forward_sequence_batch_size"],
        ):
            texts = [format_model_input(row, input_format) for row in batch]
            encoded = tokenizer(
                texts,
                add_special_tokens=tokenization["add_special_tokens"],
                padding=tokenization["padding"],
                truncation=tokenization["truncation"],
                max_length=tokenization["max_model_tokens"],
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            for index, row in enumerate(batch):
                valid_ids = input_ids[index][attention_mask[index].bool()].tolist()
                fingerprint = _digest(
                    {"record_sha256": row["sha256"], "input_ids": valid_ids}
                )
                if fingerprint != first_record_fingerprints[split][row["sha256"]]:
                    raise ValueError(
                        f"tokenization changed between passes for {row['id']}"
                    )
                aggregate.update(f"{row['sha256']}:{fingerprint}\n".encode("ascii"))
            activations = protein_model.get_activations(input_ids, attention_mask)
            assert_finite_captured_activations(activations)
            yield {
                "inputs": {
                    layer: activations.clt_input[layer]
                    for layer in range(protein_model.n_layers)
                },
                "targets": {
                    layer: activations.mlp_out[layer]
                    for layer in range(protein_model.n_layers)
                },
                "attention_mask": attention_mask,
                "record_sha256": [row["sha256"] for row in batch],
            }
        digest = aggregate.hexdigest()
        if digest != first_fingerprints[split]:
            raise ValueError(f"aggregate tokenization fingerprint changed in {split}")
        provenance["second_pass_tokenization_sha256_by_split"][split] = digest
        if split == "test":
            provenance["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            provenance["wall_time_seconds"] = time.perf_counter() - extraction_started
            device = torch.device(protein_model.device)
            if device.type == "cuda":
                provenance["peak_accelerator_memory_allocated_bytes"] = (
                    torch.cuda.max_memory_allocated(device)
                )
                provenance["peak_accelerator_memory_reserved_bytes"] = (
                    torch.cuda.max_memory_reserved(device)
                )

    dimensions = {
        layer: (geometry["input_dim"], geometry["target_dim"])
        for layer in range(geometry["n_layers"])
    }
    manifest = write_activation_cache(
        output_dir,
        {split: activation_batches(split) for split in SPLITS},
        selected_layers=list(range(geometry["n_layers"])),
        source_splits=source_splits,
        activation_provenance=provenance,
        objective="transcode",
        storage_dtype=cache_profile["storage_dtype"],
        selected_tokens_by_split=selected_tokens,
        estimated_cache_bytes=cache_profile[
            "estimated_activation_payload_bytes_by_model"
        ][model_name],
        free_space_safety_factor=cache_profile["free_space_safety_factor"],
        expected_dimensions=dimensions,
        expected_eligible_valid_token_rows_by_split=eligible_counts,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--code-archive", type=Path, required=True)
    parser.add_argument("--code-archive-sha256", required=True)
    parser.add_argument("--code-content-manifest", type=Path, required=True)
    parser.add_argument("--code-content-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-cache-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    for split in SPLITS:
        parser.add_argument(f"--{split}-manifest", type=Path, required=True)
        parser.add_argument(f"--{split}-manifest-sha256", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-config-sha256", required=True)
    parser.add_argument("--model-weights-sha256", required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--pod-name", required=True)
    parser.add_argument("--node-name", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--git-dirty", choices=("true", "false"), required=True)
    return parser.parse_args()


def main() -> None:
    execution_started = time.perf_counter()
    args = parse_args()
    code_binding = verify_code_binding(
        archive_path=args.code_archive,
        archive_sha256=args.code_archive_sha256,
        content_manifest_path=args.code_content_manifest,
        content_manifest_sha256=args.code_content_manifest_sha256,
    )
    _load_verified_local_code()
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("production cache extraction requires explicit CUDA")
    device = torch.device(args.device)
    if device.index != args.gpu_index:
        raise ValueError("--device and --gpu-index disagree")
    torch.cuda.set_device(device)
    profile = load_production_profile(args.profile, args.profile_sha256)
    panel_capacity_check = None
    if not args.preflight_only:
        panel_capacity_check = enforce_panel_cache_capacity(
            panel_cache_root=args.panel_cache_root,
            output_dir=args.output_dir,
            profile=profile,
            profile_sha256=args.profile_sha256,
            model_name=args.model_name,
            code_binding=code_binding,
        )
    torch.cuda.reset_peak_memory_stats(device)
    declared_artifacts = {
        "model_config_sha256": args.model_config_sha256,
        "model_weights_sha256": args.model_weights_sha256,
        "tokenizer_sha256": args.tokenizer_sha256,
    }
    verified_artifacts = verify_model_artifacts(args.model_root, declared_artifacts)
    protein_model = load_model(
        str(args.model_root),
        device=args.device,
        dtype=inference_dtype(profile["cache_extraction"]["model_inference_dtype"]),
    )
    source_splits = {
        split: {
            "manifest_path": str(getattr(args, f"{split}_manifest")),
            "manifest_sha256": getattr(args, f"{split}_manifest_sha256"),
        }
        for split in SPLITS
    }
    model_provenance = {
        "model_revision": args.model_revision,
        **verified_artifacts,
    }
    execution_provenance = {
        "command": sys.argv,
        "pod_name": args.pod_name,
        "node_name": args.node_name,
        "gpu_index": args.gpu_index,
        "gpu_model": torch.cuda.get_device_name(device),
        "git_commit": args.git_commit,
        "git_dirty": args.git_dirty == "true",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel_capacity_check": panel_capacity_check,
        **code_binding,
    }
    executed_profile = profile
    executed_profile_sha256 = args.profile_sha256
    cache_output = args.output_dir
    if args.preflight_only:
        if args.output_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite preflight directory: {args.output_dir}"
            )
        args.output_dir.mkdir(parents=True)
        executed_profile, executed_profile_sha256 = build_preflight_profile(
            profile,
            output_dir=args.output_dir,
        )
        source_splits = prepare_preflight_sources(
            source_splits,
            output_dir=args.output_dir,
            records_per_split=profile["preflight"]["records_per_split"],
        )
        cache_output = args.output_dir / "cache"
        execution_provenance.update(
            {
                "panel_capacity_check": {"scope": "bounded_preflight_only"},
                "execution_mode": profile["preflight"]["mode"],
                "production_scientific_eligibility": False,
                "production_cache_reuse_forbidden": True,
                "production_profile_sha256": args.profile_sha256,
            }
        )
    cache_build_started = time.perf_counter()
    manifest = build_cache_from_model(
        protein_model=protein_model,
        model_name=args.model_name,
        profile=executed_profile,
        profile_sha256=executed_profile_sha256,
        source_splits=source_splits,
        output_dir=cache_output,
        model_provenance=model_provenance,
        execution_provenance=execution_provenance,
    )
    cache_build_wall_time = time.perf_counter() - cache_build_started
    validation_started = time.perf_counter()
    cache = load_activation_cache(manifest, verify_hashes=True)
    cache_validation_wall_time = time.perf_counter() - validation_started
    if args.preflight_only:
        provenance = cache.payload["activation_provenance"]
        write_json(
            args.output_dir / "preflight_report.json",
            {
                "schema_version": "r2_dictionary_h200_preflight_report_v3",
                "status": "completed_nonconfirmatory_preflight",
                "p0_2_eligible": False,
                "production_cache_reuse_forbidden": True,
                "model_name": args.model_name,
                "production_profile_sha256": args.profile_sha256,
                "executed_preflight_profile_sha256": executed_profile_sha256,
                "cache_manifest_path": str(cache.manifest_path),
                "cache_manifest_sha256": cache.manifest_sha256,
                "cache_content_sha256": cache.content_sha256,
                "model_inference_dtype": provenance["model_inference_dtype"],
                "model_inference_dtype_verified": provenance[
                    "model_inference_dtype_verified"
                ],
                "activation_finiteness_check": provenance[
                    "activation_finiteness_check"
                ],
                "cache_storage_dtype": cache.payload["storage_dtype"],
                "code_archive_sha256": provenance["code_archive_sha256"],
                "code_content_manifest_sha256": provenance[
                    "code_content_manifest_sha256"
                ],
                "code_content_inventory_verified": provenance[
                    "code_content_inventory_verified"
                ],
                "valid_token_rows_by_split": {
                    split: cache.payload["split_summaries"][split][
                        "selected_valid_token_rows"
                    ]
                    for split in SPLITS
                },
                "cache_build_wall_time_seconds": cache_build_wall_time,
                "cache_validation_wall_time_seconds": cache_validation_wall_time,
                "wall_time_seconds": time.perf_counter() - execution_started,
                "peak_accelerator_memory_allocated_bytes": provenance[
                    "peak_accelerator_memory_allocated_bytes"
                ],
                "peak_accelerator_memory_reserved_bytes": provenance[
                    "peak_accelerator_memory_reserved_bytes"
                ],
            },
        )
        print(f"wrote bounded nonconfirmatory H200 preflight: {manifest}")
        return
    validation_started = time.perf_counter()
    validate_production_cache(cache, profile, model_name=args.model_name)
    production_validation_wall_time = time.perf_counter() - validation_started
    execution_report = cache.manifest_path.parent / "cache_execution_report.json"
    write_json(
        execution_report,
        {
            "schema_version": "r2_dictionary_cache_execution_report_v3",
            "status": "verified_complete",
            "model_name": args.model_name,
            "profile_sha256": args.profile_sha256,
            "cache_manifest_path": str(cache.manifest_path),
            "cache_manifest_sha256": cache.manifest_sha256,
            "cache_content_sha256": cache.content_sha256,
            "model_inference_dtype": cache.payload["activation_provenance"][
                "model_inference_dtype"
            ],
            "model_inference_dtype_verified": cache.payload[
                "activation_provenance"
            ]["model_inference_dtype_verified"],
            "activation_finiteness_check": cache.payload["activation_provenance"][
                "activation_finiteness_check"
            ],
            "cache_storage_dtype": cache.payload["storage_dtype"],
            "code_archive_sha256": cache.payload["activation_provenance"][
                "code_archive_sha256"
            ],
            "code_content_manifest_sha256": cache.payload[
                "activation_provenance"
            ]["code_content_manifest_sha256"],
            "code_content_inventory_verified": cache.payload[
                "activation_provenance"
            ]["code_content_inventory_verified"],
            "cache_build_wall_time_seconds": cache_build_wall_time,
            "cache_load_hash_scan_wall_time_seconds": cache_validation_wall_time,
            "production_contract_validation_wall_time_seconds": (
                production_validation_wall_time
            ),
            "wall_time_seconds": time.perf_counter() - execution_started,
            "peak_accelerator_memory_allocated_bytes": torch.cuda.max_memory_allocated(
                device
            ),
            "peak_accelerator_memory_reserved_bytes": torch.cuda.max_memory_reserved(
                device
            ),
            "panel_capacity_check": panel_capacity_check,
        },
    )
    receipt = write_completion_receipt(
        cache=cache,
        profile=profile,
        profile_sha256=args.profile_sha256,
        model_name=args.model_name,
        execution_report_path=execution_report,
    )
    print(f"wrote verified production activation cache: {manifest}; receipt: {receipt}")


if __name__ == "__main__":
    main()
