#!/usr/bin/env python3
"""Regenerate the leakage measurement that decided how the diffing pool is split.

**Why this exists.** EXP-R2-175 changed `25_model_diffing_baselines.py` from a
record-level split to a near-duplicate-group split on the strength of a DIAMOND
all-against-all of the Swiss-Prot pool: **847 of 2,048 held-out records (41.4%)
kept a relative in the training side at 95% identity or above, while only 17.4%
were exact.** That number is quoted in `summary.md`, in `draw_splits`'s own
docstring and in the audit's R2.4 row, and it existed nowhere as an artefact --
no FASTA, no DIAMOND output, no JSON. A figure that decides a design and reaches
the user-facing summary has to be re-derivable from the repository, and this
script is that derivation (EXP-R2-203).

It runs on the workstation and not in a pod, because DIAMOND is the offline
standard the shingle relation was calibrated against and no aligner is staged in
a pod. The binary is not a runtime dependency of anything else: pass `--diamond`
the path to an extracted `external_resources/tools/diamond-linux64-*.tar.gz`.

**Both splits are measured, because the comparison is the point.** The pool is
the one `draw_splits` draws. It is then split two ways and the same relation is
asked of each:

*Record level*, every record its own group -- the procedure EXP-R2-175 replaced,
and the one whose 41.4% is the quoted figure.

*Near-duplicate group level*, the committed procedure -- whose claim is that the
same measurement goes to zero at every identity boundary from 90% up.

The alignment is run once over the whole pool and the two splits read the same
hits, so the difference between the two rows is the split and nothing else.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts/transfer"))

from src.transfer.arms import corpus_location, iter_corpus_records  # noqa: E402
from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.near_duplicates import (  # noqa: E402
    NEAR_DUPLICATE_CONTAINMENT,
    group_disjoint_split,
    near_duplicate_groups,
)
from src.transfer.relational import homology_disjoint_split  # noqa: E402

SCHEMA_VERSION = "r2_transfer_pool_homology_leakage_v1"

#: The identity boundaries the verdict is read at. 95% is the one EXP-R2-175
#: quoted and 90% is the one the group split's claim is stated at; the rest are
#: reported so the reading is a curve rather than a point.
BOUNDARIES = (100.0, 95.0, 90.0, 70.0, 50.0)


def stream_pool(source: str, *, band: tuple[int, int], seed: int, skip: int, size: int):
    """The pool `25_model_diffing_baselines.py` draws, through its own stream."""

    import importlib.util

    path = REPO_ROOT / "scripts/transfer/17_train_transcoder.py"
    spec = importlib.util.spec_from_file_location("_stage17_for_leakage", path)
    stage17 = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = stage17
    spec.loader.exec_module(stage17)

    def records() -> Iterator[tuple[str, str | None]]:
        return iter_corpus_records(source, min_symbols=band[0], max_symbols=band[1])

    pool = list(stage17.stream_records(records, seed=seed, skip=skip, limit=size))
    if len(pool) < size:
        raise RuntimeError(
            f"the corpus ran out: {len(pool)} of {size} eligible records past a skip "
            f"of {skip}; the pool this measurement is about cannot be rebuilt"
        )
    return [record for record, _ in pool]


def write_fasta(path: Path, sequences: dict[int, str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, sequence in sequences.items():
            handle.write(f">r{index}\n{sequence}\n")


def align(
    diamond: Path, query: Path, subject: Path, work: Path, *, threads: int
) -> list[tuple[int, int, int, int, int]]:
    """DIAMOND query-against-subject, at EXP-R2-175's declared settings.

    Returns ``(query, subject, n_identical, query_length, subject_length)``, which
    is what "identity over the shorter sequence" needs and what ``pident`` --
    identity over the *alignment* -- does not give.
    """

    database = work / "subject.dmnd"
    hits = work / "hits.tsv"
    subprocess.run(
        [str(diamond), "makedb", "--in", str(subject), "--db", str(database), "--quiet"],
        check=True,
    )
    subprocess.run(
        [
            str(diamond), "blastp",
            "--query", str(query),
            "--db", str(database),
            "--out", str(hits),
            "--very-sensitive",
            "--masking", "0",
            "--evalue", "1e-3",
            "--threads", str(threads),
            "--max-target-seqs", "0",
            "--outfmt", "6", "qseqid", "sseqid", "nident", "qlen", "slen",
            "--quiet",
        ],
        check=True,
    )
    rows = []
    with hits.open(encoding="utf-8") as handle:
        for line in handle:
            q, s, nident, qlen, slen = line.split()
            rows.append((int(q[1:]), int(s[1:]), int(nident), int(qlen), int(slen)))
    return rows


def read_split(
    sequences: list[str], hits, train_mask: np.ndarray, *, label: str
) -> dict[str, Any]:
    """The leakage this split leaves, at every declared identity boundary.

    ``hits`` is the whole-pool alignment, so both splits read the same alignment
    and the only thing that differs between two of these records is which records
    are on which side.
    """

    held_out = np.flatnonzero(~train_mask)
    best = {int(index): 0.0 for index in held_out}
    on_train = train_mask
    for query, subject, nident, qlen, slen in hits:
        if query == subject or query not in best or not on_train[subject]:
            continue
        identity = 100.0 * nident / min(qlen, slen)
        if identity > best[query]:
            best[query] = identity
    values = np.array([best[int(index)] for index in held_out], dtype=np.float64)
    return {
        "split": label,
        "n_train": int(train_mask.sum()),
        "n_held_out": int(values.size),
        "at_or_above": {
            f"{boundary:g}": {
                "n": int((values >= boundary).sum()),
                "fraction": float((values >= boundary).mean()),
            }
            for boundary in BOUNDARIES
        },
        "median_max_identity": float(np.median(values)),
        "n_with_no_detectable_homologue": int((values == 0.0).sum()),
        "fraction_with_no_detectable_homologue": float((values == 0.0).mean()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="swissprot", choices=("swissprot", "uniref50"))
    parser.add_argument("--band", type=int, nargs=2, default=(32, 507))
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--train-records", type=int, default=8192)
    parser.add_argument("--eval-records", type=int, default=2048)
    parser.add_argument("--diamond", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True,
                        help="scratch directory for FASTA, database and hit table; "
                        "kept outside the repository")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    total = args.train_records + args.eval_records

    corpus = corpus_location(args.source)
    print(f"[paths] corpus {corpus}")
    sequences = stream_pool(
        args.source, band=tuple(args.band), seed=args.seed, skip=args.skip, size=total
    )
    print(f"[pool] {len(sequences)} records, {len(set(sequences))} distinct")

    groups, grouping = near_duplicate_groups(sequences, unit="residues")
    group_mask, group_split = group_disjoint_split(
        groups, n_train=args.train_records, seed=args.seed + 1
    )
    # The pre-repair procedure, restated exactly: every record is its own group,
    # which is the singleton case of the same mask. Written this way rather than
    # as a fresh permutation so the two rows differ in the GROUPING and in
    # nothing else -- not in the seed, not in the fraction, not in the code path.
    record_mask = homology_disjoint_split(
        np.arange(len(sequences)),
        train_fraction=args.train_records / total,
        seed=args.seed + 1,
        min_side=1,
    )

    fasta = args.work / "pool.fasta"
    write_fasta(fasta, dict(enumerate(sequences)))
    print(f"[diamond] all-against-all over {len(sequences)} records")
    hits = align(args.diamond, fasta, fasta, args.work, threads=args.threads)
    print(f"  {len(hits)} hits")

    # The raw output beside the summary, because the absence of exactly this is
    # what made the original measurement unauditable. Compressed, digested, and
    # keyed by the record index the FASTA carries, so a reader can recompute any
    # row of the table below without re-running the aligner.
    raw = {}
    for name, source_path in (("pool.fasta", fasta), ("hits.tsv", args.work / "hits.tsv")):
        destination = args.out / f"{args.source}_{name}.gz"
        with source_path.open("rb") as reader, gzip.open(destination, "wb") as writer:
            shutil.copyfileobj(reader, writer)
        raw[name] = {"path": destination.name, "sha256": sha256_file(destination)}

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "corpus": {
            "source": args.source,
            "path": str(corpus),
            "sha256": sha256_file(Path(corpus)) if Path(corpus).is_file() else None,
        },
        "pool": {
            "n_records": len(sequences),
            "n_distinct": len(set(sequences)),
            "band": list(args.band),
            "grouping": grouping,
            "group_split": group_split,
        },
        "aligner": {
            "tool": "DIAMOND",
            "version": subprocess.run(
                [str(args.diamond), "--version"], capture_output=True, text=True, check=True
            ).stdout.strip(),
            "settings": "blastp --very-sensitive --masking 0 --evalue 1e-3, "
            "all-against-all over the pool; identity is n_identical over the "
            "shorter of the two sequences, not over the alignment",
            "n_hits": len(hits),
        },
        "raw_output": {
            **raw,
            "note": "the alignment itself, gzipped, keyed by the record index in "
            "pool.fasta. Present because the measurement this regenerates had no "
            "raw output persisted anywhere and so could not be re-read at all",
        },
        "readings": [
            read_split(sequences, hits, record_mask, label="record_level"),
            read_split(sequences, hits, group_mask, label="near_duplicate_group"),
        ],
        "near_duplicate_relation": {
            "containment_threshold": NEAR_DUPLICATE_CONTAINMENT,
            "unit": "residues",
        },
        "note": (
            "the record-level row is the procedure EXP-R2-175 replaced and is the "
            "one its 41.4% figure was measured under; the group row is the "
            "committed procedure. Both read one alignment of one pool, so the only "
            "thing that differs between them is which records are held out"
        ),
    }
    write_json(args.out / f"pool_homology_leakage_{args.source}.json", payload)
    for reading in payload["readings"]:
        at95 = reading["at_or_above"]["95"]
        print(
            f"[{reading['split']}] {at95['n']}/{reading['n_held_out']} "
            f"({at95['fraction']:.1%}) at >=95%, median max identity "
            f"{reading['median_max_identity']:.1f}%, "
            f"{reading['n_with_no_detectable_homologue']} with no detectable homologue"
        )
    print(f"[done] wrote {args.out / f'pool_homology_leakage_{args.source}.json'}")


if __name__ == "__main__":
    main()
