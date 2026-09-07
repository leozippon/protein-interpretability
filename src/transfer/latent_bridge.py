"""M0/M1 capability pilot: a frozen pLM-to-LLM latent bridge for 7-way EC class JSON.

This module is a **new Direction-1 capability experiment**. It does not reopen
D3.g, does not consult stage 35/36 gates, and does not claim mechanistic
explanation or verified biological function. Success here would mean a frozen
text model used protein-side information that arrived through a trained bridge
on a closed 7-class label task.

Training targets are exactly the seven canonical JSON objects
``{"class":1,"name":"oxidoreductase"}`` … ``{"class":7,"name":"translocase"}``.
Supervision is the record's ``ec`` field, never ``description_masked``. The
protein donor sees only the native ``1`` + sequence rendering.

Do not depend on nnsight or peft. Do not use the panel tokeniser helper that silently truncates.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD

from src.transfer.io import sha256_file, write_json
from src.transfer.kmer_background import ALPHABET as KMER_ALPHABET
from src.transfer.kmer_background import kmer_index
from src.transfer.sequence_description import (
    EC_CLASS_NAMES,
    FULL_EC_PATTERN,
    SPLIT_NAMES,
    STANDARD_RESIDUES,
    SequenceDescriptionRecord,
    read_records,
)
from src.transfer.statistics import MINIMUM_BOOTSTRAP_UNITS, bootstrap_unit_floor

PILOT_KIND = "m0_m1_ec_topclass_latent_bridge"
PILOT_NOTE = (
    "Direction-1 capability pilot: frozen generative pLM prefill, trained "
    "pooling-MLP bridge, frozen text model emitting restricted EC-class JSON. "
    "Not free-form function explanation, not a mechanistic claim, not a stage "
    "35/36 continuation. The stage-34 queue is historical reuse, not new "
    "independent biological evidence."
)
CACHE_SCHEMA = "latent_bridge_cache_v2"
PREPARED_SCHEMA = "latent_bridge_prepared_v1"
REPORT_SCHEMA = "latent_bridge_report_v2"
BRIDGE_SCHEMA = "latent_bridge_checkpoint_v2"
LATENT_CONDITIONS = {
    "pretrained_donor": "pretrained", "random_donor": "random_init", "kmer": "kmer"
}
LOSS_UNIT = "mean negative log likelihood per supervised answer token (including EOS)"
DECODE_NOTE = "Greedy decoding recomputes the full context each step; no KV-cache speedup."
MACRO_F1_POLICY = (
    "macro_f1 averages classes with gold support>0; zero-support classes are listed "
    "and excluded. macro_f1_all_seven includes all seven, with absent classes F1=0."
)
K_SOFT_DEFAULT = 8
ALLOWED_DONORS = ("progen2-small", "progen2-medium", "progen2-base")
SOURCE_TO_ROLE = {"fit": "train", "eval": "val", "family_holdout": "test"}
ROLE_TO_SOURCE = {role: source for source, role in SOURCE_TO_ROLE.items()}
DIRECTION_MARKER = "1"
DONOR_RENDERING = "n_to_c_control"
POOLING = "mean_content"
IGNORE_INDEX = -100
ALLOWED_PROMPT_MODES = ("plain", "chat")
ALLOWED_NN_METRICS = ("cosine", "euclidean")
ALLOWED_DTYPES = ("float32", "bfloat16")
SUBSTAGES = ("prepare", "interface", "cache", "train", "eval", "baselines")
BLOCK_INDEX_SEMANTICS = (
    "0-based index into Arm.blocks(): 0 is the first transformer block, "
    "n_layer-1 is the last block. The hooked tensor is that block's forward "
    "output (output[0] if the block returns a tuple). It is not "
    "hidden_states[-1] (often already past final LayerNorm) and not "
    "hidden_states[layer] (often the next block's input). Only this one "
    "block is retained; other layers are not cached. The hook is removed "
    "in a finally block."
)
KNOWN_STAGE34_QUEUE = {
    "compute_dir": "results/transfer/sequence_description_cohort_non_iea",
    "h200_dir": (
        "/gpfs/jiaotongdamoxing/zhk_zip/InterpretabilityTransfer/results/"
        "transfer/sequence_description_cohort_non_iea"
    ),
    "records_sha256": (
        "58871ab58bd92bd292810801792decd1635f3d55c47cab9f5193ea69ac46377c"
    ),
    "cohort_json_sha256": (
        "de91db01dfe9aed025a6e75f0d5fcb8d3edad67fe554c78b555b616f82656e15"
    ),
    "length_band": (64, 512),
    "admitted_concepts_note": (
        "the historical stage-34 admission listed EC classes 1-6; class 7 "
        "(translocase) is still a legal target JSON but may have support 0. "
        "prepare reports per-class counts honestly and does not promise "
        "seven-way support."
    ),
}
TOKENIZER_IDENTITY_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "added_tokens.json",
    "chat_template.jinja",
)

CLASS_IDS: tuple[int, ...] = tuple(range(1, 8))
CLASS_NAME_BY_ID: dict[int, str] = {int(k): v for k, v in EC_CLASS_NAMES.items()}
CANONICAL_JSON: dict[int, str] = {
    class_id: json.dumps(
        {"class": class_id, "name": CLASS_NAME_BY_ID[class_id]},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    for class_id in CLASS_IDS
}


def target_json(class_id: int) -> str:
    """The unique supervised string for one EC top-level class."""

    if class_id not in CANONICAL_JSON:
        raise ValueError(f"EC top-level class must be 1-7, got {class_id!r}")
    return CANONICAL_JSON[class_id]


def parse_generated_json(text: str) -> dict[str, Any]:
    """Strict schema parse; canonical name agreement is an auxiliary metric.

    A digit recovered by regex from surrounding prose is a format failure.
    """

    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as error:
        return {
            "valid": False,
            "class_id": None,
            "name": None,
            "name_consistent": False,
            "error": f"not json: {error}",
        }
    if not isinstance(payload, dict):
        return {"valid": False, "class_id": None, "name": None,
                "name_consistent": False, "error": "not an object"}
    if set(payload) != {"class", "name"}:
        return {
            "valid": False,
            "class_id": None,
            "name": None,
            "name_consistent": False,
            "error": f"keys {sorted(payload)} are not exactly ['class', 'name']",
        }
    class_value = payload["class"]
    name_value = payload["name"]
    if type(class_value) is not int or class_value not in CLASS_NAME_BY_ID:
        return {
            "valid": False,
            "class_id": None,
            "name": None,
            "name_consistent": False,
            "error": f"class {class_value!r} is not an integer 1-7",
        }
    if not isinstance(name_value, str):
        return {
            "valid": False, "class_id": None, "name": None,
            "name_consistent": False, "error": "name must be a string",
        }
    return {
        "valid": True, "class_id": class_value, "name": name_value,
        "name_consistent": name_value == CLASS_NAME_BY_ID[class_value], "error": None,
    }


def top_level_ec_class(ec: Sequence[str]) -> int | None:
    """Single top-level class, or None if empty or mixed.

    Multiple four-field numbers in the **same** top-level class are kept.
    """

    if not ec:
        return None
    classes: set[int] = set()
    for number in ec:
        if not FULL_EC_PATTERN.match(number):
            raise ValueError(f"EC {number!r} is not a four-field number")
        classes.add(int(number.split(".", 1)[0]))
    if len(classes) != 1:
        return None
    class_id = next(iter(classes))
    if class_id not in CLASS_NAME_BY_ID:
        return None
    return class_id


@dataclass(frozen=True)
class PreparedRecord:
    accession: str
    sequence: str
    length: int
    ec: tuple[str, ...]
    ec_class: int
    class_name: str
    target_json: str
    dup_group: int
    family_group: str
    role: str
    source_split: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "accession": self.accession,
            "sequence": self.sequence,
            "length": int(self.length),
            "ec": list(self.ec),
            "ec_class": int(self.ec_class),
            "class_name": self.class_name,
            "target_json": self.target_json,
            "dup_group": int(self.dup_group),
            "family_group": self.family_group,
            "role": self.role,
            "source_split": self.source_split,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PreparedRecord:
        return cls(
            accession=str(payload["accession"]),
            sequence=str(payload["sequence"]),
            length=int(payload["length"]),
            ec=tuple(payload["ec"]),
            ec_class=int(payload["ec_class"]),
            class_name=str(payload["class_name"]),
            target_json=str(payload["target_json"]),
            dup_group=int(payload["dup_group"]),
            family_group=str(payload["family_group"]),
            role=str(payload["role"]),
            source_split=str(payload["source_split"]),
        )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(payload)


def cap_key(accession: str, seed: int) -> str:
    """Deterministic cap order: hash(seed, accession), never a label or score."""

    return _sha256_text(f"{seed}:{accession}")


def software_record(*, torch_imported: bool = True) -> dict[str, Any]:
    """Interpreter and library versions. H200 may differ from Compute; record both at run time."""

    record: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_implementation": sys.implementation.name,
        "executable": sys.executable,
        "note": (
            "H200's interpreter is provisioned independently of Compute; this "
            "record is the process that actually ran, not a claimed match"
        ),
    }
    if torch_imported:
        record["torch"] = torch.__version__
        record["torch_cuda"] = bool(torch.cuda.is_available())
        record["gpu_count"] = torch.cuda.device_count()
    try:
        import transformers

        record["transformers"] = transformers.__version__
    except ImportError:
        record["transformers"] = None
    return record


def class_distribution(records: Sequence[PreparedRecord]) -> dict[str, int]:
    counts = {str(class_id): 0 for class_id in CLASS_IDS}
    for record in records:
        counts[str(record.ec_class)] += 1
    return counts


def family_count(records: Sequence[PreparedRecord]) -> int:
    return len({record.family_group for record in records})


def verify_split_invariants(records: Sequence[SequenceDescriptionRecord]) -> None:
    """Accession uniqueness, dup_group disjoint splits, holdout family disjointness."""

    accessions = [record.accession for record in records]
    if len(accessions) != len(set(accessions)):
        raise ValueError("accessions are not unique in the cohort")
    dup_owners: dict[int, set[str]] = {}
    family_owners: dict[str, set[str]] = {}
    unknown = [record.split for record in records if record.split not in SPLIT_NAMES]
    if unknown:
        raise ValueError(f"unknown splits {sorted(set(unknown))}; declared {SPLIT_NAMES}")
    for record in records:
        dup_owners.setdefault(record.dup_group, set()).add(record.split)
        family_owners.setdefault(record.family_group, set()).add(record.split)
    straddling_dup = sorted(
        str(group) for group, splits in dup_owners.items() if len(splits) > 1
    )
    if straddling_dup:
        raise ValueError(
            f"{len(straddling_dup)} dup_group values straddle splits "
            f"(first {straddling_dup[:3]})"
        )
    leaked = sorted(
        family
        for family, splits in family_owners.items()
        if "family_holdout" in splits and splits != {"family_holdout"}
    )
    if leaked:
        raise ValueError(
            f"{len(leaked)} family_group values appear in family_holdout and "
            f"fit/eval (first {leaked[:3]})"
        )


def filter_records(
    records: Sequence[SequenceDescriptionRecord],
    *,
    max_residues: int | None = None,
) -> tuple[list[PreparedRecord], dict[str, int]]:
    """Keep single-class, standard-alphabet records; count every exclusion."""

    exclusions = {
        "empty_ec": 0,
        "multiple_top_level_classes": 0,
        "illegal_aa": 0,
        "over_max_residues": 0,
        "kept": 0,
    }
    kept: list[PreparedRecord] = []
    for record in records:
        if max_residues is not None and record.length > max_residues:
            exclusions["over_max_residues"] += 1
            continue
        if not set(record.sequence) <= STANDARD_RESIDUES:
            exclusions["illegal_aa"] += 1
            continue
        class_id = top_level_ec_class(record.ec)
        if class_id is None:
            if not record.ec:
                exclusions["empty_ec"] += 1
            else:
                exclusions["multiple_top_level_classes"] += 1
            continue
        kept.append(
            PreparedRecord(
                accession=record.accession,
                sequence=record.sequence,
                length=record.length,
                ec=record.ec,
                ec_class=class_id,
                class_name=CLASS_NAME_BY_ID[class_id],
                target_json=target_json(class_id),
                dup_group=record.dup_group,
                family_group=record.family_group,
                role=SOURCE_TO_ROLE[record.split],
                source_split=record.split,
            )
        )
        exclusions["kept"] += 1
    return kept, exclusions


def cap_split(
    records: Sequence[PreparedRecord], n: int | None, *, seed: int
) -> list[PreparedRecord]:
    ordered = sorted(records, key=lambda record: (cap_key(record.accession, seed), record.accession))
    if n is None or n >= len(ordered):
        return list(ordered)
    if n < 1:
        raise ValueError("a split cap must be positive when set")
    return ordered[:n]


def load_stage34_provenance(
    records_path: Path,
    cohort_json: Path,
    *,
    expect_records_sha256: str | None = None,
    expect_cohort_sha256: str | None = None,
) -> dict[str, Any]:
    if not records_path.exists():
        raise FileNotFoundError(f"{records_path} does not exist")
    if not cohort_json.exists():
        raise FileNotFoundError(
            f"{cohort_json} does not exist; prepare requires the original "
            "cohort.json provenance beside records.jsonl"
        )
    manifest = json.loads(cohort_json.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{cohort_json} is not a JSON object")
    records_sha = sha256_file(records_path)
    cohort_sha = sha256_file(cohort_json)
    if expect_records_sha256 and records_sha != expect_records_sha256:
        raise ValueError(
            f"records sha256 {records_sha} != expected {expect_records_sha256}"
        )
    if expect_cohort_sha256 and cohort_sha != expect_cohort_sha256:
        raise ValueError(
            f"cohort.json sha256 {cohort_sha} != expected {expect_cohort_sha256}"
        )
    return {
        "records_path": str(records_path),
        "cohort_json_path": str(cohort_json),
        "records_sha256": records_sha,
        "cohort_json_sha256": cohort_sha,
        "cohort_manifest_keys": sorted(manifest),
        "cohort_status": "historical_stage34_queue_reuse_not_new_biological_evidence",
        "known_queue": KNOWN_STAGE34_QUEUE,
        "original_length_band": list(KNOWN_STAGE34_QUEUE["length_band"]),
        "admitted_concepts_note": KNOWN_STAGE34_QUEUE["admitted_concepts_note"],
    }


def prepare_cohort(
    records_path: Path,
    cohort_json: Path,
    *,
    seed: int,
    max_residues: int | None = None,
    max_train: int | None = None,
    max_val: int | None = None,
    max_test: int | None = None,
    expect_records_sha256: str | None = None,
    expect_cohort_sha256: str | None = None,
) -> dict[str, Any]:
    provenance = load_stage34_provenance(
        records_path,
        cohort_json,
        expect_records_sha256=expect_records_sha256,
        expect_cohort_sha256=expect_cohort_sha256,
    )
    raw = read_records(records_path)
    verify_split_invariants(raw)
    filtered, exclusions = filter_records(raw, max_residues=max_residues)
    by_role: dict[str, list[PreparedRecord]] = {"train": [], "val": [], "test": []}
    for record in filtered:
        by_role[record.role].append(record)
    before_cap = {
        role: {
            "n": len(items),
            "class_counts": class_distribution(items),
            "n_family_groups": family_count(items),
        }
        for role, items in by_role.items()
    }
    caps = {"train": max_train, "val": max_val, "test": max_test}
    capped = {
        role: cap_split(items, caps[role], seed=seed) for role, items in by_role.items()
    }
    after_cap = {
        role: {
            "n": len(items),
            "class_counts": class_distribution(items),
            "n_family_groups": family_count(items),
        }
        for role, items in capped.items()
    }
    payload = {
        "schema_version": PREPARED_SCHEMA,
        "pilot": PILOT_KIND,
        "pilot_note": PILOT_NOTE,
        "seed": int(seed),
        "max_residues": max_residues,
        "caps": {role: caps[role] for role in caps},
        "provenance": provenance,
        "exclusions": exclusions,
        "n_raw": len(raw),
        "before_cap": before_cap,
        "after_cap": after_cap,
        "classes_with_support_after_cap": {
            role: [cid for cid, n in summary["class_counts"].items() if n > 0]
            for role, summary in after_cap.items()
        },
        "zero_support_classes_after_cap": {
            role: [cid for cid, n in summary["class_counts"].items() if n == 0]
            for role, summary in after_cap.items()
        },
        "ec7_note": KNOWN_STAGE34_QUEUE["admitted_concepts_note"],
        "records": {
            role: [record.to_dict() for record in items] for role, items in capped.items()
        },
    }
    payload["prepared_sha256"] = _sha256_json(
        {key: value for key, value in payload.items() if key != "prepared_sha256"}
    )
    return payload


def load_prepared_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != PREPARED_SCHEMA:
        raise ValueError(
            f"prepared schema {payload.get('schema_version')!r} is not {PREPARED_SCHEMA}"
        )
    expected = _sha256_json({k: v for k, v in payload.items() if k != "prepared_sha256"})
    if payload.get("prepared_sha256") != expected:
        raise ValueError("prepared.json content does not match prepared_sha256")
    return payload


def prepared_records(payload: Mapping[str, Any], role: str) -> list[PreparedRecord]:
    if payload.get("schema_version") != PREPARED_SCHEMA:
        raise ValueError(
            f"prepared schema {payload.get('schema_version')!r} is not {PREPARED_SCHEMA}"
        )
    return [PreparedRecord.from_dict(row) for row in payload["records"][role]]


def all_prepared_records(payload: Mapping[str, Any]) -> list[PreparedRecord]:
    rows: list[PreparedRecord] = []
    for role in ("train", "val", "test"):
        rows.extend(prepared_records(payload, role))
    return rows


def accession_order_hash(records: Sequence[PreparedRecord]) -> str:
    return _sha256_json([record.accession for record in records])


def kmer_features(sequences: Sequence[str], k: int = 3) -> np.ndarray:
    """Row-normalised k-mer frequencies, 20**k columns, no learned projection."""

    if k < 1:
        raise ValueError("a k-mer has at least one symbol")
    if KMER_ALPHABET != "".join(sorted(STANDARD_RESIDUES, key=KMER_ALPHABET.find)):
        raise RuntimeError("k-mer alphabet drifted from STANDARD_RESIDUES")
    width = len(KMER_ALPHABET) ** k
    out = np.zeros((len(sequences), width), dtype=np.float64)
    for row, sequence in enumerate(sequences):
        if len(sequence) < k:
            raise ValueError(f"a {len(sequence)}-residue sequence carries no {k}-mer")
        for start in range(len(sequence) - k + 1):
            out[row, kmer_index(sequence[start : start + k])] += 1.0
        out[row] /= len(sequence) - k + 1
        if not np.isfinite(out[row]).all():
            raise ValueError("k-mer features produced a non-finite row")
    return out


def composition_features(sequences: Sequence[str]) -> np.ndarray:
    out = np.zeros((len(sequences), len(KMER_ALPHABET)), dtype=np.float64)
    index = {symbol: position for position, symbol in enumerate(KMER_ALPHABET)}
    for row, sequence in enumerate(sequences):
        if not sequence:
            raise ValueError("an empty sequence has no composition")
        for symbol in sequence:
            position = index.get(symbol)
            if position is None:
                raise ValueError(f"{symbol!r} is outside {KMER_ALPHABET}")
            out[row, position] += 1.0
        out[row] /= len(sequence)
    return out


def nearest_neighbor_predict(
    query: np.ndarray,
    gallery: np.ndarray,
    gallery_labels: Sequence[int],
    gallery_accessions: Sequence[str],
    *,
    metric: str = "cosine",
) -> list[int]:
    """Fit-only 1-NN. Ties break by smaller accession, then smaller column index."""

    if metric not in ALLOWED_NN_METRICS:
        raise ValueError(f"unknown nn metric {metric!r}")
    if query.ndim != 2 or gallery.ndim != 2 or query.shape[1] != gallery.shape[1]:
        raise ValueError("query and gallery must be (n, d) with a shared d")
    if not (len(gallery) == len(gallery_labels) == len(gallery_accessions)):
        raise ValueError("gallery rows, labels and accessions are misaligned")
    if not gallery_accessions:
        raise ValueError("an empty gallery is not a nearest-neighbour classifier")
    if metric == "cosine":
        query_n = query / np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-12)
        gallery_n = gallery / np.linalg.norm(gallery, axis=1, keepdims=True).clip(min=1e-12)
        scores = query_n @ gallery_n.T
        better = np.greater
    else:
        scores = -np.linalg.norm(query[:, None, :] - gallery[None, :, :], axis=2)
        better = np.greater
    accession_rank = np.argsort(np.argsort(gallery_accessions))
    predicted: list[int] = []
    for row in scores:
        best = 0
        for j, score in enumerate(row):
            if better(score, row[best]) or (
                score == row[best] and accession_rank[j] < accession_rank[best]
            ):
                best = j
        predicted.append(int(gallery_labels[best]))
    return predicted


def tokenize_full(tokenizer: Any, text: str, *, max_tokens: int) -> list[int]:
    """Tokenise one string and refuse rather than truncate."""

    ids = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"{text[:40]!r} tokenised to no ids")
    if len(ids) > max_tokens:
        raise ValueError(
            f"tokenised length {len(ids)} exceeds declared max_tokens={max_tokens}; "
            "refusing silent truncation"
        )
    return [int(i) for i in ids]


def render_donor_sequence(sequence: str) -> str:
    return DIRECTION_MARKER + sequence


def content_mask_from_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    marker_id: int,
    pad_id: int,
) -> torch.Tensor:
    """True on residue tokens; false on the direction marker and pad."""

    mask = attention_mask.bool() & (input_ids != pad_id)
    first = attention_mask.bool().int().argmax(dim=1)
    rows = torch.arange(input_ids.shape[0], device=input_ids.device)
    is_marker = input_ids[rows, first] == marker_id
    drop = torch.zeros_like(mask)
    drop[rows[is_marker], first[is_marker]] = True
    content = mask & ~drop
    if bool((content.sum(dim=1) < 1).any()):
        raise ValueError("a record has no content positions after dropping the direction marker")
    return content


def mean_content(hidden: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
    weights = content.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * weights).sum(dim=1)
    denom = content.sum(dim=1).clamp(min=1).unsqueeze(-1).to(hidden.dtype)
    pooled = summed / denom
    if not torch.isfinite(pooled).all():
        raise ValueError("mean-content pooling produced a non-finite vector")
    return pooled


def hidden_width_for_budget(
    *, d_in: int, d_recv: int, k: int, budget: int
) -> tuple[int, int]:
    """Largest hidden width whose two Linear layers stay within ``budget`` parameters."""

    if min(d_in, d_recv, k, budget) < 1:
        raise ValueError("d_in, d_recv, k and budget must be positive")
    denom = d_in + 1 + k * d_recv
    numer = budget - k * d_recv
    if numer < denom:
        raise ValueError(
            f"trainable budget {budget} is too small for d_in={d_in}, k={k}, d_recv={d_recv}"
        )
    hidden = int(numer // denom)
    actual = d_in * hidden + hidden + hidden * k * d_recv + k * d_recv
    return hidden, actual


class PoolingMLPBridge(nn.Module):
    """Frozen LayerNorm, two-layer MLP, K soft tokens scaled to embedding std."""

    output_scale: torch.Tensor

    def __init__(
        self,
        *,
        d_in: int,
        d_recv: int,
        k: int,
        hidden: int,
        output_scale: float,
    ) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.d_recv = int(d_recv)
        self.k = int(k)
        self.hidden = int(hidden)
        self.input_norm = nn.LayerNorm(d_in)
        for parameter in self.input_norm.parameters():
            parameter.requires_grad_(False)
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, k * d_recv)
        nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        self.register_buffer("output_scale", torch.tensor(float(output_scale), dtype=torch.float32))

    def train(self, mode: bool = True) -> PoolingMLPBridge:
        super().train(mode)
        self.input_norm.eval()
        return self

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.ndim != 2 or pooled.shape[-1] != self.d_in:
            raise ValueError(f"expected (batch, {self.d_in}), got {tuple(pooled.shape)}")
        hidden = F.gelu(self.fc1(self.input_norm(pooled.float())))
        tokens = self.fc2(hidden).view(pooled.shape[0], self.k, self.d_recv)
        return tokens * self.output_scale.to(dtype=tokens.dtype, device=tokens.device)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def config(self) -> dict[str, Any]:
        return {
            "schema_version": BRIDGE_SCHEMA,
            "d_in": self.d_in,
            "d_recv": self.d_recv,
            "k": self.k,
            "hidden": self.hidden,
            "output_scale": float(self.output_scale.item()),
            "actual_parameters": int(self.trainable_parameter_count()),
        }


def inject_soft_prefix(
    *,
    soft: torch.Tensor,
    prompt_ids: torch.Tensor,
    answer_ids: torch.Tensor | None,
    embed: nn.Module,
    pad_id: int,
    prompt_mask: torch.Tensor | None = None,
    answer_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Concatenate K soft tokens, then prompt, then optional answer.

    Labels supervise only answer token ids (including a declared EOS if it is
    inside ``answer_ids``). Prefix, prompt and pad are ``IGNORE_INDEX``.
    """

    if soft.ndim != 3:
        raise ValueError("soft prefix must be (batch, k, d)")
    batch, k, width = soft.shape
    if prompt_ids.shape[0] != batch:
        raise ValueError("prompt batch does not match the prefix")
    prompt_emb = embed(prompt_ids)
    if prompt_emb.shape[-1] != width:
        raise ValueError(
            f"receiver embedding width {prompt_emb.shape[-1]} != prefix width {width}"
        )
    if prompt_mask is None:
        prompt_mask = (prompt_ids != pad_id).long()
    pieces = [soft.to(dtype=prompt_emb.dtype), prompt_emb]
    masks = [torch.ones(batch, k, dtype=torch.long, device=soft.device), prompt_mask.to(soft.device)]
    label_parts = [
        torch.full((batch, k), IGNORE_INDEX, dtype=torch.long, device=soft.device),
        torch.full(prompt_ids.shape, IGNORE_INDEX, dtype=torch.long, device=soft.device),
    ]
    if answer_ids is not None:
        if answer_mask is None:
            answer_mask = (answer_ids != pad_id).long()
        pieces.append(embed(answer_ids))
        masks.append(answer_mask.to(soft.device))
        answer_labels = answer_ids.clone()
        answer_labels[answer_mask.to(dtype=torch.bool) == False] = IGNORE_INDEX  # noqa: E712
        label_parts.append(answer_labels.to(soft.device))
    inputs_embeds = torch.cat(pieces, dim=1)
    attention_mask = torch.cat(masks, dim=1)
    labels = torch.cat(label_parts, dim=1)
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "labels": labels,
    }


def shifted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must share (batch, time)")
    return F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=IGNORE_INDEX,
    )


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def parameter_fingerprint(module: nn.Module) -> str:
    """Hash every parameter byte, in bounded CPU chunks; use only at boundaries."""

    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters(), key=lambda item: item[0]):
        digest.update(f"{name}:{tuple(parameter.shape)}:{parameter.dtype}".encode())
        flat = parameter.detach().reshape(-1)
        for chunk in flat.split(1_048_576):
            digest.update(chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def require_bridge_gradients(bridge: nn.Module) -> None:
    nonzero = False
    for name, parameter in bridge.named_parameters():
        if not parameter.requires_grad:
            continue
        grad = parameter.grad
        if grad is None or not bool(torch.isfinite(grad).all()):
            raise RuntimeError(f"bridge gradient missing or non-finite: {name}")
        nonzero |= bool((grad != 0).any())
    if not nonzero:
        raise RuntimeError("bridge received no nonzero gradient")


@contextmanager
def seeded_initialization(seed: int):
    """Seed CPU construction only, restoring the caller's torch/numpy/random RNGs."""

    numpy_state, python_state = np.random.get_state(), random.getstate()
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(int(seed)).get_state())
        np.random.seed(int(seed))
        random.seed(int(seed))
        try:
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_feature_cache(
    path: Path,
    features: np.ndarray,
    records: Sequence[PreparedRecord],
    manifest: Mapping[str, Any],
) -> None:
    if features.shape[0] != len(records):
        raise ValueError("feature rows and records are misaligned")
    if not np.isfinite(features).all():
        raise ValueError("refusing to write a cache that contains NaN or inf")
    payload = dict(manifest)
    payload["schema_version"] = CACHE_SCHEMA
    payload["n"] = len(records)
    payload["d"] = int(features.shape[1])
    payload["accessions"] = [record.accession for record in records]
    payload["accession_order_hash"] = accession_order_hash(records)
    payload["feature_sha256"] = hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()
    npz_path = Path(path)
    json_path = npz_path.with_suffix(".json")
    _atomic_npz(npz_path, {"features": np.asarray(features)})
    write_json(json_path, payload)


def load_feature_cache(path: Path, records: Sequence[PreparedRecord]) -> tuple[np.ndarray, dict[str, Any]]:
    npz_path = Path(path)
    json_path = npz_path.with_suffix(".json")
    manifest = json.loads(json_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CACHE_SCHEMA:
        raise ValueError(
            f"cache schema {manifest.get('schema_version')!r} is not {CACHE_SCHEMA}"
        )
    with np.load(npz_path) as loaded:
        features = np.asarray(loaded["features"])
    if features.ndim != 2 or features.shape != (len(records), manifest.get("d")):
        raise ValueError("cache feature shape does not match records/manifest")
    if manifest.get("n") != len(records):
        raise ValueError("cache manifest n does not match records")
    expected_order = [record.accession for record in records]
    if manifest.get("accessions") != expected_order:
        raise ValueError("cache accession order does not match the requested records")
    if manifest.get("accession_order_hash") != accession_order_hash(records):
        raise ValueError("cache accession-order hash mismatch")
    digest = hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()
    if digest != manifest.get("feature_sha256"):
        raise ValueError("cache feature bytes do not match the manifest digest")
    if not np.isfinite(features).all():
        raise ValueError("loaded cache contains NaN or inf")
    return features, manifest


def plain_instruction() -> str:
    lines = [
        "Answer with a single JSON object and nothing else.",
        'The object has exactly two keys: "class" (integer 1-7) and "name".',
        "The seven enzyme classes are:",
    ]
    for class_id in CLASS_IDS:
        lines.append(f"{class_id} {CLASS_NAME_BY_ID[class_id]}")
    lines.append("Return one of these objects, with no extra keys or text:")
    for class_id in CLASS_IDS:
        lines.append(target_json(class_id))
    return "\n".join(lines)


def user_prompt(*, mode: str, sequence: str | None, oracle_class: int | None) -> str:
    if mode not in ("latent", "raw_sequence", "oracle"):
        raise ValueError(f"unknown user-prompt mode {mode!r}")
    parts = [plain_instruction()]
    if mode == "raw_sequence":
        if not sequence:
            raise ValueError("raw_sequence prompts require the full residue string")
        parts.append("Protein sequence:")
        parts.append(sequence)
    if mode == "oracle":
        if oracle_class not in CLASS_NAME_BY_ID:
            raise ValueError("oracle prompts require a true class 1-7")
        parts.append(
            "The correct class is "
            f"{oracle_class} {CLASS_NAME_BY_ID[oracle_class]}. "
            "Emit the matching JSON object."
        )
    return "\n".join(parts)


def encode_prompt(
    tokenizer: Any,
    text: str,
    *,
    prompt_mode: str,
    max_tokens: int,
) -> list[int]:
    if prompt_mode not in ALLOWED_PROMPT_MODES:
        raise ValueError(f"prompt-mode must be plain or chat, got {prompt_mode!r}")
    if prompt_mode == "plain":
        return tokenize_full(tokenizer, text, max_tokens=max_tokens)
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        raise ValueError(
            "prompt-mode=chat but the tokenizer has no chat_template; "
            "refusing to fall back to plain"
        )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenize_full(tokenizer, rendered, max_tokens=max_tokens)


def pad_rows(rows: Sequence[Sequence[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(row) for row in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, : len(row)] = torch.tensor(list(row), dtype=torch.long)
        mask[i, : len(row)] = 1
    return ids, mask


def embedding_std(embed: nn.Embedding) -> float:
    weight = embed.weight.detach().float()
    value = float(weight.std().cpu())
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"receiver embedding std {value} is not a usable output scale")
    return value


def greedy_continue(
    *,
    model: nn.Module,
    embed: nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    max_new_tokens: int,
    eos_id: int | None,
) -> torch.Tensor:
    """Greedy decode with full-context recomputation (no KV cache).

    Right-padding is not a continuation site. Each row resumes from its last
    attended token and advances that token's logical position, not the pad tail.
    """

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    produced: list[torch.Tensor] = []
    embeds = inputs_embeds
    mask = attention_mask
    positions = position_ids
    finished = torch.zeros(embeds.shape[0], dtype=torch.bool, device=embeds.device)
    for _ in range(max_new_tokens):
        logits = model(
            inputs_embeds=embeds,
            attention_mask=mask,
            position_ids=positions,
        ).logits
        indices = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        last = indices.masked_fill(~mask.bool(), -1).max(dim=1).values
        if bool((last < 0).any()):
            raise ValueError("cannot continue an empty context")
        batch_indices = torch.arange(mask.shape[0], device=mask.device)
        next_id = logits[batch_indices, last].argmax(dim=-1)
        next_position = positions[batch_indices, last].unsqueeze(1) + 1
        if eos_id is not None:
            next_id = torch.where(finished, torch.full_like(next_id, eos_id), next_id)
            finished = finished | (next_id == eos_id)
        produced.append(next_id)
        next_emb = embed(next_id).unsqueeze(1)
        embeds = torch.cat([embeds, next_emb], dim=1)
        mask = torch.cat(
            [mask, torch.ones(mask.shape[0], 1, dtype=mask.dtype, device=mask.device)],
            dim=1,
        )
        positions = torch.cat([positions, next_position], dim=1)
        if bool(finished.all()):
            break
    return torch.stack(produced, dim=1)


def decode_ids(tokenizer: Any, token_ids: Sequence[int], eos_id: int | None) -> str:
    trimmed = list(token_ids)
    if eos_id is not None and eos_id in trimmed:
        trimmed = trimmed[: trimmed.index(eos_id)]
    return tokenizer.decode(trimmed, skip_special_tokens=True)


def prediction_row(record: PreparedRecord, text: str) -> dict[str, Any]:
    parsed = parse_generated_json(text)
    correct = bool(parsed["valid"] and parsed["class_id"] == record.ec_class)
    return {
        "accession": record.accession, "family_group": record.family_group,
        "dup_group": record.dup_group, "role": record.role,
        "true_class": record.ec_class, "true_name": record.class_name,
        "predicted_class": parsed["class_id"], "predicted_name": parsed["name"],
        "valid_json": parsed["valid"], "name_consistent": parsed["name_consistent"],
        "name_correct": bool(parsed["valid"] and parsed["name"] == record.class_name),
        "correct": correct, "joint_correct": correct and parsed["name_consistent"],
        "text": text, "parse_error": parsed["error"],
    }


def prediction_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return confusion_and_scores(
        [row["true_class"] for row in rows],
        [row["predicted_class"] if row["valid_json"] else None for row in rows],
        names=[row["predicted_name"] for row in rows],
    )


def confusion_and_scores(
    true_ids: Sequence[int], predicted: Sequence[int | None],
    *, names: Sequence[str | None] | None = None,
) -> dict[str, Any]:
    if len(true_ids) != len(predicted):
        raise ValueError("true and predicted lists are misaligned")
    n = len(true_ids)
    n_valid = sum(1 for value in predicted if value is not None)
    n_class_exact = sum(
        1 for truth, pred in zip(true_ids, predicted) if pred is not None and pred == truth
    )
    if names is not None and len(names) != n:
        raise ValueError("predicted names are misaligned")
    n_name_correct = None if names is None else sum(
        pred is not None and name == CLASS_NAME_BY_ID[truth]
        for truth, pred, name in zip(true_ids, predicted, names)
    )
    n_consistent = None if names is None else sum(
        pred is not None and name == CLASS_NAME_BY_ID[pred]
        for pred, name in zip(predicted, names)
    )
    n_joint = None if names is None else sum(
        pred == truth and name == CLASS_NAME_BY_ID[truth]
        for truth, pred, name in zip(true_ids, predicted, names)
    )
    confusion = {str(c): {str(k): 0 for k in list(CLASS_IDS) + ["invalid"]} for c in CLASS_IDS}
    support = {str(c): 0 for c in CLASS_IDS}
    f1_parts: list[float] = []
    for truth, pred in zip(true_ids, predicted):
        support[str(truth)] += 1
        column = str(pred) if pred in CLASS_NAME_BY_ID else "invalid"
        confusion[str(truth)][column] += 1
    f1_by_class: dict[str, float] = {}
    for class_id in CLASS_IDS:
        tp = sum(
            1
            for truth, pred in zip(true_ids, predicted)
            if truth == class_id and pred == class_id
        )
        fp = sum(
            1
            for truth, pred in zip(true_ids, predicted)
            if truth != class_id and pred == class_id
        )
        fn = sum(
            1
            for truth, pred in zip(true_ids, predicted)
            if truth == class_id and pred != class_id
        )
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_by_class[str(class_id)] = f1
        f1_parts.append(f1)
    zero_support = [str(cid) for cid in CLASS_IDS if support[str(cid)] == 0]
    supported = [str(cid) for cid in CLASS_IDS if support[str(cid)] > 0]
    macro_supported = (
        float(np.mean([f1_by_class[cid] for cid in supported])) if supported else None
    )
    return {
        "n_attempts": n,
        "n_valid_json": n_valid,
        "n_class_exact": n_class_exact,
        "n_name_correct": n_name_correct,
        "valid_json_rate": n_valid / n if n else 0.0,
        "class_exact_accuracy": n_class_exact / n if n else 0.0,
        "name_exact_rate": n_name_correct / n if n and n_name_correct is not None else None,
        "name_consistency_rate": n_consistent / n if n and n_consistent is not None else None,
        "name_consistency_among_valid": n_consistent / n_valid if n_valid and n_consistent is not None else None,
        "joint_accuracy": n_joint / n if n and n_joint is not None else None,
        "macro_f1": macro_supported,
        "macro_f1_over_supported_classes": macro_supported,
        "macro_f1_all_seven": float(sum(f1_parts) / len(f1_parts)),
        "f1_by_class": f1_by_class,
        "support": support,
        "zero_support_classes": zero_support,
        "supported_classes": supported,
        "macro_f1_policy": MACRO_F1_POLICY,
        "confusion": confusion,
        "denominator": "all_attempts",
    }


def paired_family_bootstrap(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    minimum_units: int = MINIMUM_BOOTSTRAP_UNITS,
) -> dict[str, Any]:
    """Paired family-cluster bootstrap of accuracy difference. No fake CI."""

    left_map = {row["accession"]: row for row in left}
    right_map = {row["accession"]: row for row in right}
    if set(left_map) != set(right_map):
        raise ValueError("paired bootstrap requires identical accession sets")
    families: dict[str, list[str]] = {}
    for accession, row in left_map.items():
        if row["family_group"] != right_map[accession]["family_group"]:
            raise ValueError(f"{accession} has mismatched family_group across arms")
        families.setdefault(row["family_group"], []).append(accession)
    floor = bootstrap_unit_floor(len(families), minimum_units=minimum_units)
    if floor["degenerate"]:
        return {
            "available": False,
            "reason": floor["degenerate_reason"],
            "n_families": int(len(families)),
            "delta": None,
            "interval": None,
        }

    def family_delta(names: Sequence[str]) -> float:
        left_hits = []
        right_hits = []
        for family in names:
            accessions = families[family]
            left_hits.append(
                np.mean([float(left_map[acc]["correct"]) for acc in accessions])
            )
            right_hits.append(
                np.mean([float(right_map[acc]["correct"]) for acc in accessions])
            )
        return float(np.mean(left_hits) - np.mean(right_hits))

    names = tuple(sorted(families))
    point = family_delta(names)
    rng = np.random.default_rng(seed)
    name_array = np.array(names, dtype=object)
    samples = [
        family_delta(tuple(str(item) for item in rng.choice(name_array, size=len(names), replace=True)))
        for _ in range(n_resamples)
    ]
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {
        "available": True,
        "reason": None,
        "n_families": int(len(families)),
        "delta": point,
        "interval": [float(lo), float(hi)],
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def sync_if_cuda(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_memory_bytes(device: str) -> int | None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return None


def timed_cuda(device: str, fn):  # type: ignore[no-untyped-def]
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    sync_if_cuda(device)
    start = time.perf_counter()
    result = fn()
    sync_if_cuda(device)
    elapsed = time.perf_counter() - start
    return result, elapsed, peak_memory_bytes(device)


def progen_blocks(model: nn.Module) -> nn.ModuleList:
    transformer = getattr(model, "transformer", None)
    blocks = getattr(transformer, "h", None) if transformer is not None else None
    if blocks is None:
        raise TypeError("donor is not a ProGen-style model with transformer.h")
    return blocks


class DonorHandle:
    """Minimal Arm.blocks() surface for a random-init donor."""

    def __init__(
        self,
        *,
        name: str,
        model: nn.Module,
        tokenizer: Any,
        n_layer: int,
        d_model: int,
        kind: str,
        checkpoint_digest: Mapping[str, Any],
        input_format: str = DONOR_RENDERING,
    ) -> None:
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.n_layer = int(n_layer)
        self.d_model = int(d_model)
        self.kind = kind
        self.checkpoint_digest = dict(checkpoint_digest)
        self.input_format = input_format

    def blocks(self) -> nn.ModuleList:
        return progen_blocks(self.model)


def checkpoint_content_digest(checkpoint: Path) -> dict[str, Any]:
    """Stream config, tokenizer files, and weight shards. Path is not identity."""

    root = Path(checkpoint)
    config = root / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"{root} has no config.json")
    weight_files: list[Path] = []
    for pattern in ("*.safetensors", "pytorch_model*.bin"):
        found = sorted(p for p in root.glob(pattern) if p.is_file())
        if found:
            weight_files = found
            break
    if not weight_files:
        raise FileNotFoundError(
            f"{root} carries no *.safetensors and no pytorch_model*.bin; "
            "refusing a path-only checkpoint identity"
        )
    tokenizer_files = [root / name for name in TOKENIZER_IDENTITY_FILES if (root / name).is_file()]
    if not tokenizer_files:
        raise FileNotFoundError(f"{root} has no tokenizer identity files")
    # Remote model code, shard maps and generation defaults can affect the interface.
    auxiliary = sorted({*root.glob("*.py"), *root.glob("*.index.json")})
    generation_config = root / "generation_config.json"
    if generation_config.is_file():
        auxiliary.append(generation_config)
    identity_files = [config, *weight_files, *tokenizer_files, *auxiliary]
    per_file = {path.name: sha256_file(path) for path in identity_files}
    combined = hashlib.sha256()
    for name in sorted(per_file):
        combined.update(f"{name} {per_file[name]}\n".encode())
    return {
        "checkpoint": str(root),
        "files": per_file,
        "content_digest": combined.hexdigest(),
        "weight_files": [path.name for path in weight_files],
        "tokenizer_files": [path.name for path in tokenizer_files],
        "note": (
            "identity is the streamed SHA-256 of config.json, tokenizer files, "
            "and weight shards. A path or config-only key is not sufficient."
        ),
    }


def capture_block_output(
    donor: Any,
    *,
    layer: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Read-only hook on ``donor.blocks()[layer]`` (0-based). See BLOCK_INDEX_SEMANTICS."""

    if not hasattr(donor, "blocks"):
        raise TypeError("donor must expose Arm.blocks(); do not index hidden_states")
    blocks = donor.blocks()
    if not 0 <= layer < len(blocks):
        raise ValueError(f"layer {layer} is outside 0..{len(blocks) - 1}")
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden

    handle = blocks[layer].register_forward_hook(hook)
    try:
        model = donor.model
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError(f"block {layer} produced no hooked output")
    if handle.id in blocks[layer]._forward_hooks:
        raise RuntimeError("block hook was not removed")
    return captured["hidden"]


def assert_allowed_donor(arm: Any) -> None:
    if arm.name not in ALLOWED_DONORS:
        raise ValueError(f"donor {arm.name!r} is not one of {ALLOWED_DONORS}")
    if arm.spec.input_format != DONOR_RENDERING:
        raise ValueError(
            f"{arm.name} input_format {arm.spec.input_format!r} is not {DONOR_RENDERING}; "
            "refusing an EC-conditioned rendering"
        )
    from src.transfer.arms import N_TO_C_MARKER

    if N_TO_C_MARKER != DIRECTION_MARKER:
        raise RuntimeError("DIRECTION_MARKER drifted from arms.N_TO_C_MARKER")


def load_donor_arm(name: str, *, device: str, dtype: str) -> DonorHandle:
    from src.transfer.arms import load_arm

    if name not in ALLOWED_DONORS:
        raise ValueError(f"donor {name!r} is not load_arm-able for this pilot")
    arm = load_arm(name, device=device, dtype=dtype)
    freeze_module(arm.model)
    assert_allowed_donor(arm)
    digest = checkpoint_content_digest(arm.spec.path)
    digest["kind"] = "pretrained"
    digest["parameter_fingerprint"] = parameter_fingerprint(arm.model)
    return DonorHandle(
        name=arm.name,
        model=arm.model,
        tokenizer=arm.tokenizer,
        n_layer=arm.n_layer,
        d_model=arm.d_model,
        kind="pretrained",
        checkpoint_digest=digest,
        input_format=arm.spec.input_format,
    )


def build_random_donor(
    checkpoint_path: Path,
    tokenizer: Any,
    *,
    name: str,
    seed: int,
    device: str,
    dtype: str,
) -> DonorHandle:
    """Same config, freshly initialised weights. Not dropout and not is_initialized."""

    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        str(checkpoint_path), local_files_only=True, trust_remote_code=True
    )
    with seeded_initialization(seed):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    model.to(device=device, dtype=torch_dtype)
    freeze_module(model)
    source = checkpoint_content_digest(checkpoint_path)
    digest = {
        "kind": "random_init",
        "random_seed": int(seed),
        "source_checkpoint": source,
        "parameter_fingerprint": parameter_fingerprint(model),
        "note": (
            "weights are a seeded from_config init, not the files in "
            "source_checkpoint; that digest identifies architecture/tokenizer only"
        ),
    }
    from src.transfer.arms import config_shape

    n_layer, d_model = config_shape(config)
    return DonorHandle(
        name=f"{name}-random{seed}",
        model=model,
        tokenizer=tokenizer,
        n_layer=n_layer,
        d_model=d_model,
        kind="random_init",
        checkpoint_digest=digest,
    )


def load_receiver(path: Path, *, device: str, dtype: str, prompt_mode: str) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if prompt_mode not in ALLOWED_PROMPT_MODES:
        raise ValueError(f"prompt-mode must be plain or chat, got {prompt_mode!r}")
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError(f"{path} tokenizer has neither pad nor eos")
        tokenizer.pad_token = tokenizer.eos_token
    if prompt_mode == "chat" and not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            f"{path} has no chat_template; prompt-mode=chat refuses a silent plain fallback"
        )
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map={"": device},
    )
    freeze_module(model)
    return model, tokenizer


def donor_layer_index(n_layer: int, requested: int | None) -> int:
    layer = n_layer - 1 if requested is None else int(requested)
    if not 0 <= layer < n_layer:
        raise ValueError(f"donor layer {layer} is outside 0..{n_layer - 1}")
    return layer


def cache_donor_features(
    *,
    donor: Any,
    records: Sequence[PreparedRecord],
    device: str,
    layer: int,
    max_tokens: int,
    batch_size: int,
    max_residues: int | None = None,
    audit: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    if not records or batch_size < 1:
        raise ValueError("donor extraction requires records and a positive batch size")
    tokenizer = donor.tokenizer
    marker_id = tokenizer.convert_tokens_to_ids(DIRECTION_MARKER)
    pad_id = tokenizer.pad_token_id
    if marker_id is None or marker_id == tokenizer.unk_token_id:
        raise ValueError("donor tokenizer has no direction-marker id")
    if pad_id is None:
        raise ValueError("donor tokenizer has no pad token")
    rows: list[np.ndarray] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        token_rows = [
            donor_token_ids(tokenizer, record, max_tokens=max_tokens, max_residues=max_residues)
            for record in chunk
        ]
        ids, mask = pad_rows(token_rows, int(pad_id))
        ids = ids.to(device)
        mask = mask.to(device)
        hidden = capture_block_output(
            donor, layer=layer, input_ids=ids, attention_mask=mask
        )
        content = content_mask_from_ids(
            ids, mask, marker_id=int(marker_id), pad_id=int(pad_id)
        )
        counts = content.sum(dim=1).tolist()
        if counts != [len(record.sequence) for record in chunk]:
            raise ValueError("content mask does not retain exactly one token per full residue")
        if audit is not None:
            audit.extend(
                {"accession": record.accession, "residues": len(record.sequence),
                 "token_count": len(tokens), "content_tokens": count, "marker_excluded": True}
                for record, tokens, count in zip(chunk, token_rows, counts)
            )
        pooled = mean_content(hidden.float(), content)
        rows.append(pooled.cpu().numpy().astype(np.float32))
    features = np.concatenate(rows, axis=0)
    if not np.isfinite(features).all():
        raise ValueError("donor features contain NaN or inf")
    return features


def donor_token_ids(
    tokenizer: Any, record: PreparedRecord, *, max_tokens: int,
    max_residues: int | None,
) -> list[int]:
    sequence = record.sequence
    if not sequence or record.length != len(sequence) or not set(sequence) <= STANDARD_RESIDUES:
        raise ValueError("donor requires a complete standard-AA sequence with matching length")
    if max_residues is not None and len(sequence) > max_residues:
        raise ValueError("sequence exceeds max_residues; refusing truncation")
    ids = tokenize_full(tokenizer, render_donor_sequence(sequence), max_tokens=max_tokens)
    # ProGen's native alphabet is one token per residue. Compare exact IDs, not only length.
    expected = [tokenizer.convert_tokens_to_ids(symbol) for symbol in DIRECTION_MARKER + sequence]
    if any(token is None or token == tokenizer.unk_token_id for token in expected) or ids != expected:
        raise ValueError("native donor rendering must be marker + every residue token, without truncation")
    return ids


def build_bridge_for_features(
    features: np.ndarray,
    *,
    d_recv: int,
    k: int,
    budget: int,
    output_scale: float,
    seed: int,
) -> PoolingMLPBridge:
    hidden, _actual = hidden_width_for_budget(
        d_in=features.shape[1], d_recv=d_recv, k=k, budget=budget
    )
    with seeded_initialization(seed):
        return PoolingMLPBridge(
            d_in=features.shape[1], d_recv=d_recv, k=k,
            hidden=hidden, output_scale=output_scale,
        )


def _answer_ids(tokenizer: Any, text: str, eos_id: int | None) -> list[int]:
    ids = tokenize_full(tokenizer, text, max_tokens=64)
    if eos_id is not None and (not ids or ids[-1] != eos_id):
        ids = ids + [int(eos_id)]
    return ids


def train_bridge(
    *,
    bridge: PoolingMLPBridge,
    receiver: Any,
    tokenizer: Any,
    features: np.ndarray,
    records: Sequence[PreparedRecord],
    val_features: np.ndarray,
    val_records: Sequence[PreparedRecord],
    prompt_mode: str,
    user_mode: str,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    grad_clip: float,
    seed: int,
    max_prompt_tokens: int,
) -> dict[str, Any]:
    if user_mode not in ("latent",):
        raise ValueError("train_bridge trains the latent prefix; raw/oracle are not this path")
    freeze_module(receiver)
    embed = receiver.get_input_embeddings()
    pad_id = int(tokenizer.pad_token_id)
    eos_id = tokenizer.eos_token_id
    instruction = user_prompt(mode="latent", sequence=None, oracle_class=None)
    prompt_row = encode_prompt(
        tokenizer, instruction, prompt_mode=prompt_mode, max_tokens=max_prompt_tokens
    )
    if not records or not val_records or min(batch_size, epochs) < 1:
        raise ValueError("training requires nonempty train/val and positive batch size/epochs")
    optimizer = AdamW(
        [parameter for parameter in bridge.parameters() if parameter.requires_grad],
        lr=lr,
    )
    bridge.to(device)
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_val = float("inf")
    saw_nonzero_bridge_grad = False

    def epoch_loss(
        feat: np.ndarray, rows: Sequence[PreparedRecord], train: bool
    ) -> float:
        nonlocal saw_nonzero_bridge_grad
        if train:
            bridge.train()
        else:
            bridge.eval()
        order = np.arange(len(rows))
        if train:
            rng = np.random.default_rng(seed + 17)
            rng.shuffle(order)
        loss_sum, token_count = 0.0, 0
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            batch_records = [rows[int(i)] for i in index]
            pooled = torch.tensor(feat[index], dtype=torch.float32, device=device)
            if train:
                optimizer.zero_grad(set_to_none=True)
                soft = bridge(pooled)
            else:
                with torch.no_grad():
                    soft = bridge(pooled)
            prompt_ids = torch.tensor([prompt_row] * len(index), dtype=torch.long, device=device)
            answers = [
                _answer_ids(tokenizer, record.target_json, eos_id) for record in batch_records
            ]
            answer_ids, answer_mask = pad_rows(answers, pad_id)
            packed = inject_soft_prefix(
                soft=soft,
                prompt_ids=prompt_ids,
                answer_ids=answer_ids.to(device),
                embed=embed,
                pad_id=pad_id,
                answer_mask=answer_mask.to(device),
            )
            outputs = receiver(
                inputs_embeds=packed["inputs_embeds"],
                attention_mask=packed["attention_mask"],
                position_ids=packed["position_ids"],
            )
            loss = shifted_cross_entropy(outputs.logits, packed["labels"])
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            if train:
                loss.backward()
                for parameter in receiver.parameters():
                    if parameter.grad is not None:
                        raise RuntimeError("frozen receiver received a gradient")
                require_bridge_gradients(bridge)
                saw_nonzero_bridge_grad = True
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        [p for p in bridge.parameters() if p.requires_grad], grad_clip,
                        error_if_nonfinite=True,
                    )
                optimizer.step()
            count = int((packed["labels"][:, 1:] != IGNORE_INDEX).sum().item())
            loss_sum += float(loss.detach().cpu()) * count
            token_count += count
        if token_count == 0:
            raise ValueError("no supervised answer tokens")
        return loss_sum / token_count

    for epoch in range(1, epochs + 1):
        train_loss = epoch_loss(features, records, True)
        with torch.no_grad():
            val_loss = epoch_loss(val_features, val_records, False)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in bridge.state_dict().items()}
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    if not saw_nonzero_bridge_grad:
        raise RuntimeError("no non-zero bridge gradient was observed")
    bridge.load_state_dict(best_state)
    return {
        "history": history,
        "best_val_loss": best_val,
        "loss_unit": LOSS_UNIT,
        "bridge_config": bridge.config(),
        "n_train": len(records),
        "n_val": len(val_records),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "grad_clip": grad_clip,
        "seed": seed,
        "prompt_mode": prompt_mode,
        "saw_nonzero_bridge_grad": True,
        "test_metrics_computed": False,
    }


def generate_predictions(
    *,
    bridge: PoolingMLPBridge | None,
    receiver: Any,
    tokenizer: Any,
    features: np.ndarray | None,
    records: Sequence[PreparedRecord],
    prompt_mode: str,
    user_mode: str,
    device: str,
    max_new_tokens: int,
    max_prompt_tokens: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    freeze_module(receiver)
    if bridge is not None:
        bridge.eval()
        bridge.to(device)
    embed = receiver.get_input_embeddings()
    pad_id = int(tokenizer.pad_token_id)
    eos_id = tokenizer.eos_token_id
    rows: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        texts = [
            user_prompt(
                mode=user_mode,
                sequence=record.sequence if user_mode == "raw_sequence" else None,
                oracle_class=record.ec_class if user_mode == "oracle" else None,
            )
            for record in chunk
        ]
        prompt_rows = [
            encode_prompt(tokenizer, text, prompt_mode=prompt_mode, max_tokens=max_prompt_tokens)
            for text in texts
        ]
        prompt_ids, prompt_mask = pad_rows(prompt_rows, pad_id)
        prompt_ids = prompt_ids.to(device)
        prompt_mask = prompt_mask.to(device)
        if user_mode == "latent":
            if bridge is None or features is None:
                raise ValueError("latent generation needs a bridge and features")
            pooled = torch.tensor(features[start : start + len(chunk)], dtype=torch.float32, device=device)
            with torch.no_grad():
                soft = bridge(pooled)
            packed = inject_soft_prefix(
                soft=soft,
                prompt_ids=prompt_ids,
                answer_ids=None,
                embed=embed,
                pad_id=pad_id,
                prompt_mask=prompt_mask,
            )
        else:
            prompt_emb = embed(prompt_ids)
            packed = {
                "inputs_embeds": prompt_emb,
                "attention_mask": prompt_mask,
                "position_ids": (prompt_mask.cumsum(dim=1) - 1).clamp(min=0),
            }
        with torch.no_grad():
            token_matrix = greedy_continue(
                model=receiver,
                embed=embed,
                inputs_embeds=packed["inputs_embeds"],
                attention_mask=packed["attention_mask"],
                position_ids=packed["position_ids"],
                max_new_tokens=max_new_tokens,
                eos_id=None if eos_id is None else int(eos_id),
            )
        for i, record in enumerate(chunk):
            text = decode_ids(
                tokenizer,
                token_matrix[i].tolist(),
                None if eos_id is None else int(eos_id),
            )
            rows.append(prediction_row(record, text))
    return rows


def evaluation_mode(condition: str) -> str:
    """The CLI's single final-evaluation dispatch; oracle is M0-only, NN is separate."""

    if condition in LATENT_CONDITIONS:
        return "latent"
    if condition == "raw_sequence":
        return "raw_sequence"
    raise ValueError(f"condition {condition!r} is not a final eval arm; oracle is M0-only, nn uses baselines")


def portable_identity(value: Any) -> Any:
    """Remove only checkpoint location strings; preserve all scientific manifest fields."""

    if isinstance(value, Mapping):
        return {k: portable_identity(v) for k, v in value.items()
                if not (k == "checkpoint" and isinstance(v, str))}
    if isinstance(value, list):
        return [portable_identity(v) for v in value]
    return value


def validate_cache_binding(condition: str, prepared_hash: str, manifest: Mapping[str, Any]) -> None:
    if condition not in LATENT_CONDITIONS:
        raise ValueError(f"condition {condition!r} is not a latent training arm")
    required = {
        "schema_version", "data_hash", "donor_kind", "checkpoint_digest", "layer",
        "pooling", "rendering", "n", "d", "accessions", "accession_order_hash", "feature_sha256",
    }
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"incompatible cache: missing {sorted(missing)}")
    if manifest["schema_version"] != CACHE_SCHEMA or manifest["data_hash"] != prepared_hash:
        raise ValueError("cache schema/data hash mismatch")
    if manifest["donor_kind"] != LATENT_CONDITIONS[condition]:
        raise ValueError("condition and cache donor_kind mismatch")
    digest = manifest["checkpoint_digest"]
    if condition == "random_donor":
        if not {"random_seed", "parameter_fingerprint", "source_checkpoint"} <= digest.keys():
            raise ValueError("random cache lacks seed/parameter/source identity")
        digest = digest["source_checkpoint"]
    if not digest.get("content_digest"):
        raise ValueError("cache lacks checkpoint content digest")
    if condition != "kmer":
        required_donor = {"donor_name", "n_layer", "max_tokens", "max_residues", "dtype", "direction_marker"}
        if not required_donor <= manifest.keys():
            raise ValueError("cache lacks native donor interface settings")
        if (manifest["pooling"], manifest["rendering"], manifest["direction_marker"]) != (
            POOLING, DONOR_RENDERING, DIRECTION_MARKER
        ):
            raise ValueError("cache is not native mean-content donor rendering")
        if type(manifest["layer"]) is not int or not 0 <= manifest["layer"] < manifest["n_layer"]:
            raise ValueError("cache donor layer is out of range")


def bridge_identity(
    *, condition: str, prepared_hash: str, cache_manifest: Mapping[str, Any],
    receiver_digest: Mapping[str, Any], prompt_mode: str,
    bridge_config: Mapping[str, Any], seed: int,
) -> dict[str, Any]:
    validate_cache_binding(condition, prepared_hash, cache_manifest)
    if not {"files", "content_digest", "weight_files", "tokenizer_files"} <= receiver_digest.keys():
        raise ValueError("receiver lacks weight/tokenizer content identity")
    if not receiver_digest["tokenizer_files"] or not receiver_digest["weight_files"]:
        raise ValueError("receiver identity must include tokenizer and weights")
    if prompt_mode not in ALLOWED_PROMPT_MODES or bridge_config.get("schema_version") != BRIDGE_SCHEMA:
        raise ValueError("incompatible prompt mode or bridge schema")
    if bridge_config["d_in"] != cache_manifest["d"]:
        raise ValueError("bridge input width and cache feature width mismatch")
    return {
        "schema_version": BRIDGE_SCHEMA, "condition": condition,
        "prepared_sha256": prepared_hash,
        "cache_manifest_sha256": _sha256_json(portable_identity(cache_manifest)),
        "receiver": portable_identity(receiver_digest), "prompt_mode": prompt_mode,
        "bridge_config": dict(bridge_config), "seed": int(seed),
    }


def validate_bridge_identity(payload: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    actual = payload.get("extra", {}).get("identity")
    if not isinstance(actual, dict) or _sha256_json(actual) != _sha256_json(expected):
        raise ValueError("bridge artifact identity mismatch or missing identity; retrain with the current format")
    if payload.get("config") != expected["bridge_config"]:
        raise ValueError("bridge configuration does not match its bound identity")


def interface_optimizer_step(
    *, bridge: PoolingMLPBridge, donor: nn.Module, receiver: Any,
    tokenizer: Any, features: np.ndarray, record: PreparedRecord,
    prompt_mode: str, max_prompt_tokens: int, device: str,
) -> dict[str, Any]:
    """One SGD correctness step, not M1 training. Only M0 hashes frozen weights."""

    for model in (donor, receiver):
        if any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError("M0 endpoints must already be frozen and gradient-free")
    before = {"donor": parameter_fingerprint(donor), "receiver": parameter_fingerprint(receiver)}
    bridge.to(device)
    bridge_before = parameter_fingerprint(bridge)
    prompt = encode_prompt(tokenizer, user_prompt(mode="latent", sequence=None, oracle_class=None),
                           prompt_mode=prompt_mode, max_tokens=max_prompt_tokens)
    answer = _answer_ids(tokenizer, record.target_json, tokenizer.eos_token_id)
    bridge.zero_grad(set_to_none=True)
    packed = inject_soft_prefix(
        soft=bridge(torch.tensor(features[:1], dtype=torch.float32, device=device)),
        prompt_ids=torch.tensor([prompt], device=device),
        answer_ids=torch.tensor([answer], device=device),
        answer_mask=torch.ones(1, len(answer), dtype=torch.long, device=device),
        embed=receiver.get_input_embeddings(), pad_id=int(tokenizer.pad_token_id),
    )
    logits = receiver(inputs_embeds=packed["inputs_embeds"],
                      attention_mask=packed["attention_mask"], position_ids=packed["position_ids"]).logits
    loss = shifted_cross_entropy(logits, packed["labels"])
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("M0 answer CE is non-finite")
    loss.backward()
    require_bridge_gradients(bridge)
    SGD([p for p in bridge.parameters() if p.requires_grad], lr=1e-3).step()
    if any(not bool(torch.isfinite(p).all()) for p in bridge.parameters()):
        raise RuntimeError("M0 optimizer produced non-finite parameters")
    changed = parameter_fingerprint(bridge) != bridge_before
    after = {"donor": parameter_fingerprint(donor), "receiver": parameter_fingerprint(receiver)}
    no_grad = all(p.grad is None for model in (donor, receiver) for p in model.parameters())
    if not changed or before != after or not no_grad:
        raise RuntimeError("M0 optimizer/frozen endpoint invariant failed")
    return {
        "bridge_gradients_finite_nonzero": True, "bridge_changed": changed,
        "frozen_parameters_unchanged": before == after, "frozen_parameters_no_grad": no_grad,
        "fingerprints_before": before, "fingerprints_after": after,
        "answer_ce": float(loss.detach().cpu()), "loss_unit": LOSS_UNIT,
        "optimizer": {"name": "SGD", "lr": 1e-3, "steps": 1, "records": 1},
    }


def save_bridge(path: Path, bridge: PoolingMLPBridge, extra: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": bridge.config(),
        "extra": dict(extra),
        "state_dict": {key: value.detach().cpu() for key, value in bridge.state_dict().items()},
    }
    torch.save(payload, destination)


def load_bridge(path: Path) -> tuple[PoolingMLPBridge, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu")
    config = payload["config"]
    if config.get("schema_version") != BRIDGE_SCHEMA:
        raise ValueError("incompatible bridge checkpoint schema")
    with seeded_initialization(0):
        bridge = PoolingMLPBridge(
            d_in=config["d_in"], d_recv=config["d_recv"], k=config["k"],
            hidden=config["hidden"], output_scale=config["output_scale"],
        )
    bridge.load_state_dict(payload["state_dict"])
    return bridge, payload


def command_record(args: Any) -> dict[str, Any]:
    payload = vars(args).copy()
    payload["macro_f1_policy"] = MACRO_F1_POLICY
    payload["decode_note"] = DECODE_NOTE
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload
