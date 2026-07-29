#!/usr/bin/env python3
"""Build deterministic P0-1 cohort and local model-file manifests.

The retained component records comprise 100 real lysozymes followed by 100
random UniRef50 controls.  Without the exact historical cohort, this script
reconstructs the documented alternating order and labels it unverified.  When
an independently recovered historical file and its SHA-256 are supplied, the
same path validates the exact archived row order instead of inferring it from
atlas outputs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import shlex
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
R2 = REPO / "r2_interpretability_transfer"
DEFAULT_SOURCE = (
    R2
    / "results/ec_metrics/calibration_lysozyme_20260507/calibration_sequences.json"
)
DEFAULT_OUTPUT = R2 / "results/npj_revision_20260716/manifests"

REAL_SOURCE = "real_lysozyme"
RANDOM_SOURCE = "random_uniref50"
EXPECTED_PER_SOURCE = 100
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
ADDED_RECORD_KEYS = frozenset({"cohort_index", "split", "family", "sha256"})
MODEL_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
CONFIG_NAMES = frozenset(
    {"config.json", "generation_config.json", "tokenizer_config.json"}
)
IDENTIFIER_KEYS = (
    "_name_or_path",
    "name_or_path",
    "_commit_hash",
    "revision",
    "model_type",
    "architectures",
    "tokenizer_class",
    "auto_map",
    "transformers_version",
    "vocab_size",
    "hidden_size",
    "n_embd",
    "num_hidden_layers",
    "n_layer",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "unk_token_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--historical-cohort",
        type=Path,
        help="Independently recovered historical balanced-200 JSON file.",
    )
    parser.add_argument(
        "--historical-cohort-sha256",
        help="Externally recorded lowercase SHA-256 of --historical-cohort.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model-root",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Recursively hash one deployed model/tokenizer tree; repeat for "
            "multiple models. Historical run revisions remain explicitly unretained."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate and hash without writing.")
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="Rebuild in memory and verify the existing output files byte-for-byte.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace existing build outputs; disabled by default.",
    )
    args = parser.parse_args()
    if args.overwrite and (args.dry_run or args.verify_only):
        parser.error("--overwrite applies only to a build")
    if (args.historical_cohort is None) != (args.historical_cohort_sha256 is None):
        parser.error(
            "--historical-cohort and --historical-cohort-sha256 must be supplied together"
        )
    if args.historical_cohort_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", args.historical_cohort_sha256
    ):
        parser.error("--historical-cohort-sha256 must be a lowercase SHA-256 digest")
    return args


def reject_constant(value: str) -> None:
    raise ValueError(f"non-RFC8259 numeric constant: {value}")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def strict_json_load_bytes(data: bytes, source: Path) -> Any:
    try:
        text = data.decode("utf-8")
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid strict JSON in {source}: {exc}") from exc


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def output_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_file_bytes(path: Path) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(path)
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    signature_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    signature_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if signature_before != signature_after or len(data) != after.st_size:
        raise RuntimeError(f"file changed while being read: {path}")
    return data


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    after = path.stat()
    signature_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    signature_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if signature_before != signature_after:
        raise RuntimeError(f"file changed while being hashed: {path}")
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def file_descriptor(path: Path) -> dict[str, Any]:
    return {
        "path": display_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_json_atomic(path: Path, value: Any, overwrite: bool) -> None:
    data = output_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_source_record(record: Any, index: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TypeError(f"source record {index} is not an object")
    missing = {"id", "source", "sequence"} - record.keys()
    if missing:
        raise ValueError(f"source record {index} lacks keys: {sorted(missing)}")
    collision = ADDED_RECORD_KEYS & record.keys()
    if collision:
        raise ValueError(f"source record {index} already has reserved keys: {sorted(collision)}")
    if not isinstance(record["id"], str) or not record["id"]:
        raise ValueError(f"source record {index} has an invalid id")
    if record["source"] not in {REAL_SOURCE, RANDOM_SOURCE}:
        raise ValueError(f"source record {index} has unexpected source {record['source']!r}")
    sequence = record["sequence"]
    if not isinstance(sequence, str) or not sequence:
        raise ValueError(f"source record {index} has an invalid sequence")
    invalid = sorted(set(sequence) - AA)
    if invalid:
        raise ValueError(f"source record {index} contains noncanonical residues: {invalid}")
    return record


def build_cohort(
    source_path: Path,
    source_bytes: bytes,
    *,
    historical_path: Path | None = None,
    historical_bytes: bytes | None = None,
    historical_sha256: str | None = None,
) -> dict[str, Any]:
    payload = strict_json_load_bytes(source_bytes, source_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise TypeError("cohort source must be an object containing a records array")
    records = [validate_source_record(record, i) for i, record in enumerate(payload["records"])]
    expected_total = 2 * EXPECTED_PER_SOURCE
    if payload.get("n_records") != expected_total or len(records) != expected_total:
        raise ValueError(
            f"expected source n_records and array length {expected_total}; got "
            f"{payload.get('n_records')!r} and {len(records)}"
        )
    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("cohort source contains duplicate record ids")

    labels = [record["source"] for record in records]
    expected_labels = [REAL_SOURCE] * EXPECTED_PER_SOURCE + [RANDOM_SOURCE] * EXPECTED_PER_SOURCE
    if labels != expected_labels:
        raise ValueError("retained source is not the expected real-then-random component layout")
    real = records[:EXPECTED_PER_SOURCE]
    random_records = records[EXPECTED_PER_SOURCE:]

    ordered_source_records: list[dict[str, Any]] = []
    ordered: list[dict[str, Any]] = []
    for pair_index, (real_record, random_record) in enumerate(zip(real, random_records, strict=True)):
        for record in (real_record, random_record):
            ordered_source_records.append(copy.deepcopy(record))
            entry = copy.deepcopy(record)
            entry["cohort_index"] = len(ordered)
            entry["split"] = "historical_atlas_shared_discovery_evaluation"
            entry["family"] = None
            entry["sha256"] = sha256_bytes(entry["sequence"].encode("utf-8"))
            ordered.append(entry)
        if ordered[-2]["cohort_index"] != 2 * pair_index:
            raise RuntimeError("internal interleaving error")

    if [record["source"] for record in ordered[0::2]] != [REAL_SOURCE] * EXPECTED_PER_SOURCE:
        raise RuntimeError("even cohort rows are not all real lysozymes")
    if [record["source"] for record in ordered[1::2]] != [RANDOM_SOURCE] * EXPECTED_PER_SOURCE:
        raise RuntimeError("odd cohort rows are not all random UniRef50 controls")

    historical_descriptor = None
    historical_status = "reconstructed_unverified"
    historical_note = (
        "The retained records and documented alternation determine this reconstruction, "
        "but common row permutation leaves atlas correlations unchanged. Historical order "
        "is not proven until independent provenance is resolved and the canonical atlas is rerun."
    )
    if any(
        value is not None
        for value in (historical_path, historical_bytes, historical_sha256)
    ):
        if historical_path is None or historical_bytes is None or historical_sha256 is None:
            raise ValueError("historical cohort path, bytes and SHA-256 are jointly required")
        if not re.fullmatch(r"[0-9a-f]{64}", historical_sha256):
            raise ValueError("historical cohort SHA-256 must be a lowercase digest")
        if sha256_bytes(historical_bytes) != historical_sha256:
            raise ValueError("historical cohort SHA-256 mismatch")
        historical = strict_json_load_bytes(historical_bytes, historical_path)
        if not isinstance(historical, dict) or set(historical) != {
            "records",
            "source",
            "construction",
        }:
            raise ValueError("historical cohort must have exact records/source/construction fields")
        historical_records = [
            validate_source_record(record, index)
            for index, record in enumerate(historical["records"])
        ]
        if historical["source"] != (
            "Research2/results/ec_metrics/calibration_lysozyme_20260507/"
            "calibration_sequences.json"
        ):
            raise ValueError("historical cohort names an unexpected retained source")
        if historical["construction"] != (
            "interleaved real_lysozyme and random_uniref50 controls"
        ):
            raise ValueError("historical cohort names an unexpected construction")
        if historical_records != ordered_source_records:
            raise ValueError(
                "historical cohort records differ from the retained records or alternating order"
            )
        historical_descriptor = {
            "path": display_path(historical_path),
            "size_bytes": len(historical_bytes),
            "sha256": historical_sha256,
            "archived_source": historical["source"],
            "archived_construction": historical["construction"],
            "record_equality": "exact ordered JSON object equality before provenance fields",
        }
        historical_status = "historical_exact_file_verified"
        historical_note = (
            "An independently recovered historical sequence-bearing file named by the "
            "May atlas provenance has the externally pinned SHA-256 and exactly matches "
            "all 200 retained records in their archived alternating order."
        )

    ordered_hash = sha256_bytes(canonical_json_bytes(ordered))
    return {
        "schema_version": "1.0",
        "artifact": "canonical_balanced_200_cohort_validation",
        "historical_ordering_status": historical_status,
        "historical_ordering_note": historical_note,
        "historical_artifact": historical_descriptor,
        "source": {
            "path": display_path(source_path),
            "size_bytes": len(source_bytes),
            "sha256": sha256_bytes(source_bytes),
            "retained_layout": "100 real_lysozyme records, then 100 random_uniref50 records",
        },
        "construction": {
            "method": "stable zip interleave of the retained source groups",
            "index_rule": "real_i at 2*i; random_i at 2*i+1",
            "within_source_order": "preserved exactly from the retained source records",
            "sequence_transformation": "none; full retained sequences and metadata preserved",
            "family_field_status": "unavailable in the retained source; stored as null",
        },
        "counts": {
            "total": len(ordered),
            REAL_SOURCE: EXPECTED_PER_SOURCE,
            RANDOM_SOURCE: EXPECTED_PER_SOURCE,
        },
        "ordered_cohort_sha256": ordered_hash,
        "ordered_cohort_hash_definition": (
            "SHA-256 of UTF-8 JSON for the records array with lexicographically sorted object "
            "keys and no insignificant whitespace"
        ),
        "records": ordered,
    }


def parse_model_roots(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    labels: set[str] = set()
    roots: set[Path] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"--model-root must be NAME=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        if not MODEL_LABEL.fullmatch(label):
            raise ValueError(f"invalid model-root name: {label!r}")
        if label in labels:
            raise ValueError(f"duplicate model-root name: {label}")
        path = Path(raw_path).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise NotADirectoryError(path)
        if path in roots:
            raise ValueError(f"duplicate resolved model root: {path}")
        labels.add(label)
        roots.add(path)
        result.append((label, path))
    return sorted(result, key=lambda item: item[0])


def model_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for current, directories, filenames in os.walk(root, topdown=True, followlinks=False):
        directories.sort()
        filenames.sort()
        current_path = Path(current)
        for directory in directories:
            candidate = current_path / directory
            if candidate.is_symlink():
                raise ValueError(f"directory symlinks are unsupported; pass the resolved tree: {candidate}")
        for filename in filenames:
            candidate = current_path / filename
            if candidate.is_symlink():
                target = candidate.resolve(strict=True)
                if not target.is_file():
                    raise ValueError(f"model-tree symlink does not resolve to a file: {candidate}")
            elif not candidate.is_file():
                raise ValueError(f"non-regular entry in model tree: {candidate}")
            files.append(candidate)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def config_identifiers(path: Path, root: Path) -> dict[str, Any]:
    payload = strict_json_load_bytes(stable_file_bytes(path), path)
    if not isinstance(payload, dict):
        raise TypeError(f"model config is not a JSON object: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "identifiers": {key: payload[key] for key in IDENTIFIER_KEYS if key in payload},
    }


def build_model_root(label: str, root: Path) -> dict[str, Any]:
    paths = model_files(root)
    if not paths:
        raise ValueError(f"model root contains no files: {root}")
    entries: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    for path in paths:
        stat = path.stat()
        entry: dict[str, Any] = {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": stat.st_size,
            "sha256": sha256_file(path),
            "storage": "symlink" if path.is_symlink() else "regular_file",
        }
        if path.is_symlink():
            entry["symlink_target"] = os.readlink(path)
        entries.append(entry)
        if path.name in CONFIG_NAMES:
            configs.append(config_identifiers(path, root))
    return {
        "name": label,
        "root": str(root),
        "historical_upstream_revision": {
            "status": "not_retained",
            "value": None,
            "note": (
                "Identifiers detected in the current local configs do not establish the "
                "revision used by the historical atlas run."
            ),
        },
        "file_count": len(entries),
        "total_size_bytes": sum(entry["size_bytes"] for entry in entries),
        "tree_sha256": sha256_bytes(canonical_json_bytes(entries)),
        "tree_hash_definition": "SHA-256 of canonical JSON for the ordered files array",
        "config_identifiers": configs,
        "files": entries,
    }


def build_model_manifest(specifications: list[tuple[str, Path]]) -> dict[str, Any]:
    roots = [build_model_root(label, path) for label, path in specifications]
    return {
        "schema_version": "1.0",
        "artifact": "recursive_local_model_tokenizer_file_manifest",
        "status": "complete" if roots else "not_requested",
        "scope_note": (
            "Every regular file under each explicitly supplied root is sized and hashed. "
            "No model roots were supplied for this run."
            if not roots
            else "Every regular file under each explicitly supplied root is sized and hashed."
        ),
        "roots": roots,
    }


def environment_manifest() -> dict[str, Any]:
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "working_directory": str(Path.cwd().resolve()),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git": {
            "status": "not_available_in_execution_mirror",
            "commit": None,
        },
    }


def build_run_manifest(
    args: argparse.Namespace,
    source_path: Path,
    cohort_path: Path,
    model_path: Path,
    cohort: dict[str, Any],
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    argv = [sys.executable, *sys.argv]
    ineligibility_reasons = [
        "historical upstream model/tokenizer revisions are not retained",
        *(
            []
            if args.model_root
            else ["deployed model/tokenizer roots were not supplied to this run"]
        ),
    ]
    if cohort["historical_ordering_status"] != "historical_exact_file_verified":
        ineligibility_reasons.insert(
            0, "historical cohort ordering is reconstructed_unverified"
        )
    inputs = {"cohort_source": file_descriptor(source_path)}
    if args.historical_cohort is not None:
        inputs["historical_cohort"] = file_descriptor(
            args.historical_cohort.expanduser().resolve(strict=True)
        )
    return {
        "schema_version": "1.0",
        "artifact": "npj_revision_p0_1_manifest_build_run",
        "status": "partial_p0_1_ineligible",
        "p0_1_eligible": False,
        "ineligibility_reasons": ineligibility_reasons,
        "historical_ordering_status": cohort["historical_ordering_status"],
        "command": {
            "argv": argv,
            "shell_escaped": shlex.join(argv),
        },
        "resolved_parameters": {
            "source": display_path(source_path),
            "historical_cohort": (
                None
                if args.historical_cohort is None
                else display_path(args.historical_cohort)
            ),
            "historical_cohort_sha256": args.historical_cohort_sha256,
            "out_dir": display_path(args.out_dir),
            "model_roots": list(args.model_root),
            "overwrite": args.overwrite,
        },
        "script": file_descriptor(script_path),
        "environment": environment_manifest(),
        "inputs": inputs,
        "outputs": {
            "cohort": file_descriptor(cohort_path),
            "model_tokenizer_files": file_descriptor(model_path),
        },
        "self_hash_status": (
            "not embedded because a file cannot contain its own cryptographic hash"
        ),
    }


def summary_payload(cohort: dict[str, Any], model_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "validated_in_memory",
        "historical_ordering_status": cohort["historical_ordering_status"],
        "counts": cohort["counts"],
        "ordered_cohort_sha256": cohort["ordered_cohort_sha256"],
        "cohort_output_sha256": sha256_bytes(output_json_bytes(cohort)),
        "model_root_count": len(model_manifest["roots"]),
        "model_manifest_output_sha256": sha256_bytes(output_json_bytes(model_manifest)),
    }


def verify_outputs(
    args: argparse.Namespace,
    source_path: Path,
    cohort: dict[str, Any],
    model_manifest: dict[str, Any],
) -> dict[str, Any]:
    cohort_path = args.out_dir / "canonical_cohort_reconstructed.json"
    model_path = args.out_dir / "model_tokenizer_file_manifest.json"
    run_path = args.out_dir / "run_manifest.json"
    saved_cohort = strict_json_load_bytes(stable_file_bytes(cohort_path), cohort_path)
    saved_models = strict_json_load_bytes(stable_file_bytes(model_path), model_path)
    saved_run = strict_json_load_bytes(stable_file_bytes(run_path), run_path)
    if saved_cohort != cohort:
        raise ValueError(f"saved cohort does not match deterministic reconstruction: {cohort_path}")
    if saved_models != model_manifest:
        raise ValueError(f"saved model manifest does not match current roots: {model_path}")
    if not isinstance(saved_run, dict):
        raise TypeError(f"run manifest is not an object: {run_path}")
    expected_outputs = {
        "cohort": file_descriptor(cohort_path),
        "model_tokenizer_files": file_descriptor(model_path),
    }
    if saved_run.get("outputs") != expected_outputs:
        raise ValueError("run-manifest output hashes do not match saved outputs")
    if saved_run.get("inputs", {}).get("cohort_source") != file_descriptor(source_path):
        raise ValueError("run-manifest cohort-source hash does not match the current source")
    expected_historical = (
        None
        if args.historical_cohort is None
        else file_descriptor(args.historical_cohort.expanduser().resolve(strict=True))
    )
    if saved_run.get("inputs", {}).get("historical_cohort") != expected_historical:
        raise ValueError("run-manifest historical-cohort hash does not match the current source")
    if saved_run.get("script") != file_descriptor(Path(__file__).resolve()):
        raise ValueError("run-manifest script hash does not match the current script")
    result = summary_payload(cohort, model_manifest)
    result["status"] = "verified"
    result["run_manifest_sha256"] = sha256_file(run_path)
    return result


def main() -> None:
    args = parse_args()
    source_path = args.source.expanduser().resolve(strict=True)
    historical_path = (
        None
        if args.historical_cohort is None
        else args.historical_cohort.expanduser().resolve(strict=True)
    )
    out_dir = args.out_dir.expanduser().resolve()
    args.out_dir = out_dir
    specifications = parse_model_roots(args.model_root)
    if any(out_dir == root or out_dir.is_relative_to(root) for _, root in specifications):
        raise ValueError("output directory must not be inside a hashed model root")

    source_bytes = stable_file_bytes(source_path)
    cohort = build_cohort(
        source_path,
        source_bytes,
        historical_path=historical_path,
        historical_bytes=(
            None if historical_path is None else stable_file_bytes(historical_path)
        ),
        historical_sha256=args.historical_cohort_sha256,
    )
    model_manifest = build_model_manifest(specifications)

    if args.dry_run:
        print(output_json_bytes(summary_payload(cohort, model_manifest)).decode("utf-8"), end="")
        return
    if args.verify_only:
        print(
            output_json_bytes(verify_outputs(args, source_path, cohort, model_manifest)).decode("utf-8"),
            end="",
        )
        return

    cohort_path = out_dir / "canonical_cohort_reconstructed.json"
    model_path = out_dir / "model_tokenizer_file_manifest.json"
    run_path = out_dir / "run_manifest.json"
    if not args.overwrite:
        existing = [path for path in (cohort_path, model_path, run_path) if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")
    write_json_atomic(cohort_path, cohort, args.overwrite)
    write_json_atomic(model_path, model_manifest, args.overwrite)
    run_manifest = build_run_manifest(args, source_path, cohort_path, model_path, cohort)
    write_json_atomic(run_path, run_manifest, args.overwrite)
    result = summary_payload(cohort, model_manifest)
    result["status"] = "written"
    result["run_manifest_sha256"] = sha256_file(run_path)
    print(output_json_bytes(result).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
