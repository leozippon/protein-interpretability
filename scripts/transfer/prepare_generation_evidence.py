#!/usr/bin/env python3
"""Prepare the frozen EXP-R2-232 full-output ledger and structural subsets on CPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.transfer.generation_evidence import prepare  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, default=REPO / "results/transfer/conditioned_generation")
    parser.add_argument("--source-work", type=Path, default=Path("/Data/lzp/work/r227"))
    parser.add_argument("--queue", type=Path, default=REPO / "evidence/conditioned_generation_20260826/class_queue.json")
    parser.add_argument("--out", type=Path, default=REPO / "results/transfer/generation_evidence")
    parser.add_argument("--evidence", type=Path, default=REPO / "evidence/generation_evidence_20260905")
    args = parser.parse_args()
    manifest = prepare(args.source_results, args.source_work, args.queue, args.out, args.evidence)
    print(json.dumps({"campaign": manifest["campaign"], "census": manifest["census"], "outputs": manifest["outputs"]}, sort_keys=True))


if __name__ == "__main__":
    main()
