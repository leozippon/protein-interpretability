"""Strict, atomic serialization helpers for confirmatory revision artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


SHA256_LENGTH = 64
PROVENANCE_FIELDS = (
    "model_revision",
    "tokenizer_revision",
    "clt_checkpoint_sha256",
    "selection_cohort_sha256",
    "evaluation_cohort_sha256",
    "code_revision",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_provenance(provenance: Mapping[str, object]) -> dict[str, str]:
    """Require immutable model, dictionary, cohort and code provenance."""

    if not isinstance(provenance, Mapping):
        raise ValueError("provenance must be an object")
    missing = set(PROVENANCE_FIELDS) - set(provenance)
    if missing:
        raise ValueError(f"provenance is missing fields: {sorted(missing)}")
    normalized = {field: str(provenance[field]).strip() for field in PROVENANCE_FIELDS}
    if any(not value for value in normalized.values()):
        raise ValueError("provenance values must be non-empty strings")
    for field in (
        "clt_checkpoint_sha256",
        "selection_cohort_sha256",
        "evaluation_cohort_sha256",
    ):
        value = normalized[field].lower()
        if len(value) != SHA256_LENGTH or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"{field} must be a lowercase SHA-256 digest")
        normalized[field] = value
    if normalized["selection_cohort_sha256"] == normalized["evaluation_cohort_sha256"]:
        raise ValueError("selection and evaluation cohort hashes must be distinct")
    return normalized


def _json_bytes(value: Any, *, indent: int | None) -> bytes:
    text = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
        separators=None if indent is not None else (",", ":"),
    )
    return (text + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: Any, *, indent: int = 2) -> None:
    """Write RFC-compliant JSON atomically, rejecting NaN and infinity."""

    _atomic_write(Path(path), _json_bytes(value, indent=indent))


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    """Write strict JSON Lines atomically."""

    payload = b"".join(_json_bytes(row, indent=None) for row in rows)
    _atomic_write(Path(path), payload)
