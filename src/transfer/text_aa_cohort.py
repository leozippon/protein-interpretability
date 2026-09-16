"""Tokenizer-only ProteinGym census for the 13 text amino-acid string controls.

This module accounts request encodings, application windows, and support sets.
It does not load weights, score likelihoods, compute Spearman, or admit an
experiment. Paths come from ``arm_spec``; the boundary table does not restate
checkpoint directories.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .arms import arm_spec, require_input_path
from .io import sha256_file
from .precision_policy import TEXT_AA_FP32_V1
from .text_aa_fitness import (
    TextAAEncodingError,
    encode_text_aa,
    load_text_aa_tokenizer,
    resolve_text_aa_boundary,
)

BOUNDARIES_PATH = Path(__file__).with_name("text_aa_boundaries.json")
ENCODE_FAIL = "encode_fail"
EXCEEDS_HARD_CONTEXT = "exceeds_hard_context"
RESEARCH_ROLE = "text-aa-string-control"
BOOTSTRAP_SEED_BASE = 20260913
TEXT_AA_SEED_OFFSET = 2000
BOOTSTRAP_RESAMPLES = 2000
UNIMPLEMENTED_PHASES = ("probe", "score", "analyse")
_PATH_KEYS = frozenset({"local_dir", "path", "checkpoint", "checkpoint_dir"})
_SMALL_TOKENIZER_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "tokenization_bygpt5.py",
    "configuration_bygpt5.py",
)
_SOURCE_RELATIVE = (
    "src/transfer/text_aa_fitness.py",
    "src/transfer/text_aa_cohort.py",
    "src/transfer/text_aa_boundaries.json",
    "scripts/transfer/text_aa_dms.py",
    "src/transfer/amino_acids.py",
)

__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED_BASE",
    "BOUNDARIES_PATH",
    "ENCODE_FAIL",
    "EXCEEDS_HARD_CONTEXT",
    "RESEARCH_ROLE",
    "TEXT_AA_SEED_OFFSET",
    "UNIMPLEMENTED_PHASES",
    "TextAAEncodingError",
    "census_models",
    "census_one_model",
    "load_text_aa_boundary_table",
    "resolve_text_aa_checkpoint",
    "source_fingerprints",
    "support_sets",
    "text_aa_model_names",
]


def _as_mapping(value: Any, *, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{what} must be a mapping")
    return value


def load_text_aa_boundary_table(path: Path | None = None) -> dict[str, Any]:
    """Load the tracked 13-model boundary table and refuse a second path source."""

    table_path = BOUNDARIES_PATH if path is None else Path(path)
    payload = json.loads(table_path.read_text(encoding="utf-8"))
    table = _as_mapping(payload, what="text-AA boundary table")
    order = table.get("order")
    checkpoints = table.get("text_aa_checkpoints")
    families = table.get("families")
    if not isinstance(order, list) or not order:
        raise ValueError("boundary table is missing order")
    if not isinstance(checkpoints, Mapping):
        raise ValueError("boundary table is missing text_aa_checkpoints")
    if not isinstance(families, Mapping):
        raise ValueError("boundary table is missing families")
    if list(order) != list(dict.fromkeys(order)):
        raise ValueError("boundary table order has duplicate names")
    if set(order) != set(checkpoints):
        raise ValueError("boundary table order and text_aa_checkpoints disagree")
    for name, spec in checkpoints.items():
        entry = _as_mapping(spec, what=f"{name} boundary spec")
        extra = _PATH_KEYS.intersection(entry)
        if extra:
            raise ValueError(
                f"{name}: checkpoint directories are resolved from arm_spec, "
                f"not from {sorted(extra)}"
            )
    if "qwen3-8b-base" not in checkpoints:
        raise ValueError("Qwen3 must be declared as qwen3-8b-base")
    qwen3 = resolve_text_aa_boundary("qwen3-8b-base", table)
    if qwen3.model_type != "qwen3":
        raise ValueError(
            f"qwen3-8b-base must resolve to model_type 'qwen3', got {qwen3.model_type!r}"
        )
    if qwen3.name != "qwen3-8b-base":
        raise ValueError("Qwen3 text-AA control is the Base checkpoint, not Instruct")
    expected_families = {
        "gpt2": ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"),
        "qwen2.5": ("qwen2.5-0.5b", "qwen2.5-7b", "qwen2.5-32b"),
        "bygpt5": ("bygpt5-small-en", "bygpt5-base-en", "bygpt5-medium-en"),
    }
    for family, members in expected_families.items():
        got = tuple(families.get(family) or ())
        if got != members:
            raise ValueError(f"{family} family members must be {members}, got {got}")
        if family == "gpt2" and "dialogpt-small" in got:
            raise ValueError("DialoGPT-small is not a GPT-2 scale rung")
        if family == "qwen2.5" and "qwen3-8b-base" in got:
            raise ValueError("Qwen3-8B-Base is not a Qwen2.5 scale rung")
    return dict(table)


def text_aa_model_names(table: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    payload = load_text_aa_boundary_table() if table is None else table
    return tuple(str(name) for name in payload["order"])


def resolve_text_aa_checkpoint(name: str) -> Path:
    """Resolve one text-AA checkpoint from the existing ArmSpec, not a second table."""

    spec = arm_spec(name)
    return require_input_path(Path(spec.path).resolve(), spec.path_variable)


def source_fingerprints(*, repo_root: Path | None = None) -> dict[str, str]:
    root = Path(__file__).resolve().parents[2] if repo_root is None else Path(repo_root)
    fingerprints: dict[str, str] = {}
    for relative in _SOURCE_RELATIVE:
        path = root / relative
        fingerprints[relative] = sha256_file(path)
    fingerprints["boundaries"] = fingerprints["src/transfer/text_aa_boundaries.json"]
    return fingerprints


def tokenizer_small_file_sha256(checkpoint: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in _SMALL_TOKENIZER_FILES:
        path = Path(checkpoint) / name
        if path.is_file():
            hashes[name] = sha256_file(path)
    return hashes


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _ids_sha256(ids: Sequence[int]) -> str:
    payload = b"".join(int(token).to_bytes(4, "little", signed=False) for token in ids)
    return hashlib.sha256(payload).hexdigest()


def _update_stream(digest: Any, *parts: bytes) -> None:
    for part in parts:
        digest.update(len(part).to_bytes(4, "little", signed=False))
        digest.update(part)


def _require_sequence_str(value: Any, *, what: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a str, got {type(value).__name__}")
    return value


def _request_items(assay: Mapping[str, Any], mutants: Sequence[str]) -> list[tuple[str, int | None, str]]:
    items: list[tuple[str, int | None, str]] = [
        (
            "wildtype",
            None,
            _require_sequence_str(assay["wildtype_sequence"], what="wildtype_sequence"),
        )
    ]
    for index, sequence in enumerate(mutants):
        items.append(
            (
                "mutant",
                int(index),
                _require_sequence_str(sequence, what=f"mutant {index}"),
            )
        )
    return items


def census_one_model(
    name: str,
    *,
    tokenizer: Any,
    boundary: Any,
    hard_context: int | None,
    request: Mapping[str, Any],
    documented_context: int | None,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Encode every frozen WT and selected mutant. Do not resample or drop variants."""

    if hard_context != documented_context:
        raise ValueError(
            f"{name}: documented context {documented_context} disagrees with "
            f"config hard_context {hard_context}"
        )
    assays = list(request["assays"])
    sequences = _as_mapping(request["sequences"], what="request sequences")
    assay_rows: list[dict[str, Any]] = []
    native_fixed: list[str] = []
    legal_candidates: list[dict[str, Any]] = []
    n_sequences = 0
    n_encode_ok = 0
    n_encode_fail = 0
    n_over = 0
    for assay in assays:
        name_assay = str(assay["assay"])
        mutants = list(sequences[name_assay])
        if len(mutants) != int(assay["n_variants"]):
            raise ValueError(
                f"{name_assay}: request n_variants {assay['n_variants']} disagrees "
                f"with sequence list {len(mutants)}"
            )
        items = _request_items(assay, mutants)
        failures: list[dict[str, Any]] = []
        n_input_ok: list[int] = []
        stream = hashlib.sha256()
        encode_fail = False
        over_window = False
        n_row_encode_fail = 0
        n_row_over = 0
        wt_n_input: int | None = None
        for role, mutant_index, sequence in items:
            n_sequences += 1
            _update_stream(
                stream,
                role.encode("utf-8"),
                b"" if mutant_index is None else str(mutant_index).encode("ascii"),
                _sequence_sha256(sequence).encode("ascii"),
            )
            try:
                ids = encode_text_aa(tokenizer, sequence, boundary)
            except TextAAEncodingError as exc:
                n_encode_fail += 1
                n_row_encode_fail += 1
                encode_fail = True
                reason = ENCODE_FAIL
                _update_stream(stream, reason.encode("ascii"), type(exc).__name__.encode("ascii"))
                failures.append(
                    {
                        "role": role,
                        "mutant_index": mutant_index,
                        "reason": reason,
                        "error_class": type(exc).__name__,
                        "sequence_sha256": _sequence_sha256(sequence),
                        "n_residues": len(sequence),
                    }
                )
                continue
            n_encode_ok += 1
            width = len(ids)
            ids_sha = _ids_sha256(ids)
            _update_stream(stream, ids_sha.encode("ascii"))
            if role == "wildtype":
                wt_n_input = width
            if hard_context is not None and width > hard_context:
                n_over += 1
                n_row_over += 1
                over_window = True
                failures.append(
                    {
                        "role": role,
                        "mutant_index": mutant_index,
                        "reason": EXCEEDS_HARD_CONTEXT,
                        "n_input_tokens": width,
                        "sequence_sha256": _sequence_sha256(sequence),
                        "encoded_ids_sha256": ids_sha,
                        "n_residues": len(sequence),
                    }
                )
                continue
            n_input_ok.append(width)
            legal_candidates.append(
                {
                    "assay": name_assay,
                    "index": int(assay["index"]),
                    "role": role,
                    "mutant_index": mutant_index,
                    "n_input_tokens": width,
                    "n_residues": len(sequence),
                    "sequence_sha256": _sequence_sha256(sequence),
                    "encoded_ids_sha256": ids_sha,
                }
            )
        if encode_fail:
            exclude_reason = ENCODE_FAIL
            admitted = False
        elif over_window:
            exclude_reason = EXCEEDS_HARD_CONTEXT
            admitted = False
        else:
            exclude_reason = None
            admitted = True
            native_fixed.append(name_assay)
        assay_rows.append(
            {
                "assay": name_assay,
                "index": int(assay["index"]),
                "cluster": assay["cluster"],
                "n_variants": int(assay["n_variants"]),
                "mutant_digest": assay["mutant_digest"],
                "csv_sha256": assay["csv_sha256"],
                "seed": int(assay["seed"]),
                "wildtype_id": assay.get("wildtype_id"),
                "n_residues_wt": len(assay["wildtype_sequence"]),
                "n_input_tokens_wt": wt_n_input,
                "n_input_tokens_max_legal": (max(n_input_ok) if n_input_ok else None),
                "n_encode_ok": len(items) - n_row_encode_fail,
                "n_encode_fail": n_row_encode_fail,
                "n_exceeds_hard_context": n_row_over,
                "n_eligible_sequences": len(n_input_ok),
                "n_failed": len(failures),
                "admitted": admitted,
                "exclude_reason": exclude_reason,
                "failed_sequences": failures,
                "encoded_ids_digest": stream.hexdigest(),
            }
        )
    if not legal_candidates:
        raise ValueError(
            f"{name}: no strictly encodable sequence fits the scoring window; "
            "a window will not be invented"
        )
    if hard_context is not None:
        application_window = int(hard_context)
    else:
        application_window = max(int(item["n_input_tokens"]) for item in legal_candidates)
    probe = None
    for item in legal_candidates:
        if int(item["n_input_tokens"]) > application_window:
            continue
        if probe is None or int(item["n_input_tokens"]) > int(probe["n_input_tokens"]):
            probe = {
                "assay": item["assay"],
                "index": item["index"],
                "role": item["role"],
                "mutant_index": item["mutant_index"],
                "n_input_tokens": item["n_input_tokens"],
                "n_residues": item["n_residues"],
                "sequence_sha256": item["sequence_sha256"],
                "encoded_ids_sha256": item["encoded_ids_sha256"],
            }
    if probe is None:
        raise ValueError(f"{name}: no legal sequence fits application_window {application_window}")
    if n_encode_ok + n_encode_fail != n_sequences:
        raise RuntimeError(
            f"{name}: n_encode_ok {n_encode_ok} + n_encode_fail {n_encode_fail} "
            f"!= n_sequences {n_sequences}"
        )
    if n_encode_ok != sum(int(row["n_encode_ok"]) for row in assay_rows):
        raise RuntimeError(f"{name}: model n_encode_ok disagrees with assay rows")
    if n_encode_fail != sum(int(row["n_encode_fail"]) for row in assay_rows):
        raise RuntimeError(f"{name}: model n_encode_fail disagrees with assay rows")
    if n_over != sum(int(row["n_exceeds_hard_context"]) for row in assay_rows):
        raise RuntimeError(f"{name}: model n_exceeds_hard_context disagrees with assay rows")
    payload = {
        "name": name,
        "protocol_id": TEXT_AA_FP32_V1,
        "research_role": RESEARCH_ROLE,
        "hard_context": hard_context,
        "documented_context": documented_context,
        "application_window_tokens": application_window,
        "window_rule": (
            "config_hard_context" if hard_context is not None else "max_legal_request_input"
        ),
        "native_fixed": native_fixed,
        "n_assays": len(assays),
        "n_admitted": len(native_fixed),
        "n_sequences": n_sequences,
        "n_encode_ok": n_encode_ok,
        "n_encode_fail": n_encode_fail,
        "n_exceeds_hard_context": n_over,
        "assays": assay_rows,
        "longest_probe_identity": probe,
        "experiment_admitted": False,
    }
    if checkpoint is not None:
        payload["checkpoint"] = str(Path(checkpoint))
        payload["tokenizer_small_files_sha256"] = tokenizer_small_file_sha256(checkpoint)
    return payload


def support_sets(
    model_payloads: Mapping[str, Mapping[str, Any]],
    *,
    table: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Intersect census-admitted assays. Incomplete model lists are not all-13."""

    payload = load_text_aa_boundary_table() if table is None else table
    order = tuple(str(name) for name in payload["order"])
    families = {
        str(family): tuple(str(member) for member in members)
        for family, members in payload["families"].items()
    }
    assay_order: list[str] = []
    for name in order:
        if name in model_payloads:
            assay_order = [str(row["assay"]) for row in model_payloads[name]["assays"]]
            break

    def _intersection(members: Sequence[str]) -> dict[str, Any]:
        missing = [name for name in members if name not in model_payloads]
        if missing:
            return {
                "complete": False,
                "missing_models": missing,
                "assays": None,
                "n_assays": None,
            }
        admitted = [
            set(model_payloads[name]["native_fixed"]) for name in members
        ]
        shared = set.intersection(*admitted) if admitted else set()
        assays = [name for name in assay_order if name in shared]
        return {
            "complete": True,
            "missing_models": [],
            "assays": assays,
            "n_assays": len(assays),
        }

    result = {
        "native_fixed": {
            name: list(model_payloads[name]["native_fixed"])
            for name in order
            if name in model_payloads
        },
        "common_all13": _intersection(order),
        "families": {
            family: _intersection(members) for family, members in families.items()
        },
    }
    return result


def census_models(
    names: Sequence[str],
    *,
    request: Mapping[str, Any],
    table: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run tokenizer-only census for the named models against one frozen request."""

    payload = load_text_aa_boundary_table() if table is None else table
    allowed = text_aa_model_names(payload)
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise KeyError(f"unknown text-AA models {unknown}; declared {list(allowed)}")
    if list(names) != list(dict.fromkeys(names)):
        raise ValueError("census model list has duplicates")
    models: dict[str, Any] = {}
    for name in names:
        checkpoint = resolve_text_aa_checkpoint(name)
        boundary = resolve_text_aa_boundary(name, payload)
        bundle = load_text_aa_tokenizer(checkpoint, boundary=boundary, name=name)
        models[name] = census_one_model(
            name,
            tokenizer=bundle.tokenizer,
            boundary=bundle.boundary,
            hard_context=bundle.hard_context,
            request=request,
            documented_context=bundle.boundary.documented_context,
            checkpoint=checkpoint,
        )
        models[name]["path_variable"] = arm_spec(name).path_variable
        models[name]["model_type"] = bundle.boundary.model_type
    return {
        "schema": "text-aa-census/v1",
        "protocol_id": TEXT_AA_FP32_V1,
        "research_role": RESEARCH_ROLE,
        "phase": "census",
        "implemented_phases": ["census"],
        "unimplemented_phases": list(UNIMPLEMENTED_PHASES),
        "models_requested": list(names),
        "models": models,
        "support_sets": support_sets(models, table=payload),
        "source_sha256": source_fingerprints(repo_root=repo_root),
        "input_counts": {
            "n_assays": int(request["declared_assays"]),
            "n_models": len(names),
        },
        "experiment_admitted": False,
        "details_approved": False,
    }
