#!/usr/bin/env python3
"""Extract hash-bound pretrained-model measurements for the P0-5 factorial."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.revision.n_terminal_extractor import run_extractor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument(
        "--spec-sha256",
        required=True,
        help="Exact lowercase SHA-256 of the frozen extraction specification",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = run_extractor(
        args.spec,
        args.spec_sha256,
        args.out_dir,
        mode="production",
        command=[sys.executable, *sys.argv],
    )
    print(f"wrote verified P0-5 measurement extraction: {manifest}")


if __name__ == "__main__":
    main()
