#!/usr/bin/env python3
"""Build one immutable, receipt-backed P0-8 input bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.revision.recoverability_input_builder import (  # noqa: E402
    build_recoverability_inputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_recoverability_inputs(args.spec, args.out_dir)
    print(f"wrote verified P0-8 input receipt: {receipt}")


if __name__ == "__main__":
    main()

