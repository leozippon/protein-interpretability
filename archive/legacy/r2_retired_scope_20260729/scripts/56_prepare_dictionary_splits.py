#!/usr/bin/env python3
"""Freeze disjoint, hash-verified FASTA splits for confirmatory dictionary runs.

The selector intentionally uses a documented prefix of eligible records.  The
selected records are shuffled once with a fixed seed, exact duplicate sequences
are removed before splitting, and every released row carries the common R2
cohort fields required by the major-revision protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Iterator


REQUIRED_FIELDS = ("id", "source", "sequence", "split", "family", "sha256")
SPLITS = ("train", "validation", "test")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode()


def iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    header: str | None = None
    sequence: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(sequence)
                header = line[1:].strip()
                sequence = []
            else:
                if header is None:
                    raise ValueError(f"sequence encountered before FASTA header in {path}")
                sequence.append("".join(line.split()))
    if header is not None:
        yield header, "".join(sequence)


def record_id(header: str) -> str:
    value = header.split(maxsplit=1)[0]
    if not value:
        raise ValueError("empty FASTA record identifier")
    return value


def family_value(header: str, identifier: str, mode: str) -> str:
    if mode == "record_id":
        return identifier
    if mode == "pipe_second":
        fields = identifier.split("|")
        if len(fields) < 2 or not fields[1]:
            raise ValueError(f"cannot parse second pipe field from {identifier!r}")
        return fields[1]
    raise ValueError(f"unknown family mode: {mode}")


def normalize_sequence(raw_sequence: str, family: str, mode: str) -> str:
    if mode == "plain":
        sequence = raw_sequence.upper()
    elif mode == "zymctrl":
        match = re.fullmatch(r"([^<]+)<sep><start>([A-Za-z*.-]+)<end>", raw_sequence)
        if match is None:
            raise ValueError("invalid ZymCTRL EC<sep><start>SEQUENCE<end> record")
        if match.group(1) != family:
            raise ValueError(
                f"ZymCTRL prompt/header EC mismatch: {match.group(1)!r} != {family!r}"
            )
        sequence = match.group(2).upper()
    else:
        raise ValueError(f"unknown sequence format: {mode}")
    if not re.fullmatch(r"[A-Z*.-]+", sequence):
        raise ValueError("non-protein characters in normalized sequence")
    return sequence


def select_records(args: argparse.Namespace) -> tuple[list[dict], dict]:
    requested = args.train_count + args.validation_count + args.test_count
    selected: list[dict] = []
    seen_ids: set[str] = set()
    seen_sequences: set[str] = set()
    scanned = 0
    eligible = 0
    duplicate_sequences = 0

    for header, raw_sequence in iter_fasta(args.fasta):
        scanned += 1
        identifier = record_id(header)
        family = family_value(header, identifier, args.family_mode)
        try:
            sequence = normalize_sequence(raw_sequence, family, args.sequence_format)
        except ValueError as error:
            raise ValueError(f"{error} for FASTA record {header!r}") from error
        if not args.min_seq_len <= len(sequence) <= args.max_seq_len:
            continue
        eligible += 1
        if identifier in seen_ids:
            raise ValueError(f"duplicate FASTA identifier before selection completed: {identifier}")
        seen_ids.add(identifier)
        sequence_hash = sha256_bytes(sequence.encode())
        if sequence_hash in seen_sequences:
            duplicate_sequences += 1
            continue
        seen_sequences.add(sequence_hash)
        selected.append(
            {
                "id": identifier,
                "source": args.source,
                "sequence": sequence,
                "family": family,
                "sha256": sequence_hash,
            }
        )
        if len(selected) == requested:
            break

    if len(selected) != requested:
        raise ValueError(
            f"requested {requested} unique eligible sequences but found {len(selected)} "
            f"after scanning {scanned} records"
        )

    random.Random(args.seed).shuffle(selected)
    boundaries = (args.train_count, args.train_count + args.validation_count)
    for index, record in enumerate(selected):
        if index < boundaries[0]:
            record["split"] = "train"
        elif index < boundaries[1]:
            record["split"] = "validation"
        else:
            record["split"] = "test"

    selection = {
        "eligible_records_seen": eligible,
        "records_scanned": scanned,
        "exact_duplicate_sequences_skipped": duplicate_sequences,
        "selection_method": "first_n_unique_eligible_then_seeded_shuffle",
    }
    return selected, selection


def validate_records(records: list[dict], counts: dict[str, int]) -> None:
    ids: set[str] = set()
    sequence_hashes: set[str] = set()
    observed = {split: 0 for split in SPLITS}
    for record in records:
        if tuple(sorted(record)) != tuple(sorted(REQUIRED_FIELDS)):
            raise ValueError(f"unexpected record fields: {sorted(record)}")
        if record["split"] not in observed:
            raise ValueError(f"invalid split: {record['split']}")
        if sha256_bytes(record["sequence"].encode()) != record["sha256"]:
            raise ValueError(f"sequence hash mismatch: {record['id']}")
        if record["id"] in ids:
            raise ValueError(f"duplicate selected identifier: {record['id']}")
        if record["sha256"] in sequence_hashes:
            raise ValueError(f"exact sequence leakage: {record['id']}")
        ids.add(record["id"])
        sequence_hashes.add(record["sha256"])
        observed[record["split"]] += 1
    if observed != counts:
        raise ValueError(f"split-count mismatch: expected {counts}, observed {observed}")


def write_atomic(path: Path, payload: bytes, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--family-mode", choices=("record_id", "pipe_second"), default="record_id")
    parser.add_argument("--sequence-format", choices=("plain", "zymctrl"), default="plain")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--validation-count", type=int, required=True)
    parser.add_argument("--test-count", type=int, required=True)
    parser.add_argument("--min-seq-len", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--source-sha256", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.fasta.is_file():
        raise FileNotFoundError(args.fasta)
    if min(args.train_count, args.validation_count, args.test_count) <= 0:
        raise ValueError("all split counts must be positive")
    if args.source_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", args.source_sha256):
        raise ValueError("--source-sha256 must be a lowercase SHA-256 digest")

    records, selection = select_records(args)
    counts = {
        "train": args.train_count,
        "validation": args.validation_count,
        "test": args.test_count,
    }
    validate_records(records, counts)

    outputs: dict[str, dict] = {}
    for split in SPLITS:
        path = args.out_dir / f"{split}.jsonl"
        payload = b"".join(
            strict_json_bytes(record) for record in records if record["split"] == split
        )
        write_atomic(path, payload, args.overwrite)
        outputs[split] = {
            "path": path.name,
            "rows": counts[split],
            "sha256": sha256_bytes(payload),
            "ordered_sequence_sha256": sha256_bytes(
                "\n".join(
                    record["sha256"] for record in records if record["split"] == split
                ).encode()
            ),
        }

    script_path = Path(__file__).resolve()
    summary = {
        "schema_version": "r2_dictionary_split_v1",
        "created_utc_date": "2026-07-17",
        "fasta": str(args.fasta.resolve()),
        "source": args.source,
        "source_bytes": args.fasta.stat().st_size,
        "source_sha256": args.source_sha256,
        "source_hash_status": "verified_external" if args.source_sha256 else "not_computed",
        "seed": args.seed,
        "min_seq_len": args.min_seq_len,
        "max_seq_len": args.max_seq_len,
        "family_mode": args.family_mode,
        "sequence_format": args.sequence_format,
        "counts": counts,
        "selection": selection,
        "outputs": outputs,
        "script_sha256": sha256_file(script_path),
    }
    write_atomic(args.out_dir / "manifest.json", strict_json_bytes(summary), args.overwrite)
    print(strict_json_bytes(summary).decode(), end="")


if __name__ == "__main__":
    main()
