#!/usr/bin/env python3
"""Verify and globally adjudicate a frozen collection of script-53 runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.revision.semantic_adjudication import adjudicate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = adjudicate(
        args.spec,
        args.out_dir,
        expected_spec_sha256=args.spec_sha256,
        collector_script_path=Path(__file__),
    )
    print(receipt)


if __name__ == "__main__":
    main()

