#!/usr/bin/env python3
"""Recover R227 reference coverage or annotate the new native ProGen3 batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.transfer import generation_annotations as ga  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    recover = commands.add_parser("recover-r227")
    recover.add_argument("--attempts", type=Path, default=Path("results/transfer/generation_evidence/attempts.jsonl"))
    recover.add_argument("--table", type=Path, default=Path("/Data/lzp/work/r227/identity/generations.tsv"))
    recover.add_argument("--query-fasta", type=Path, default=Path("/Data/lzp/work/r227/identity/generations.fasta"))
    recover.add_argument("--report", type=Path, default=Path("results/transfer/conditioned_generation/conditioned_generation.json"))
    recover.add_argument("--out", type=Path, default=Path("results/transfer/generation_evidence/reference_annotations"))
    native = commands.add_parser("native-progen3")
    for name in ("attempts", "out", "hmmscan", "pfam-hmm", "diamond", "diamond-db", "reference-metadata"):
        native.add_argument(f"--{name}", type=Path, required=True)
    native.add_argument("--threads", type=int, default=8)
    native.add_argument("--shards", type=int, default=4)
    args = vars(parser.parse_args())
    command = args.pop("command")
    args["output"] = args.pop("out")
    result = ga.recover_r227(**args) if command == "recover-r227" else ga.annotate_native(**args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
