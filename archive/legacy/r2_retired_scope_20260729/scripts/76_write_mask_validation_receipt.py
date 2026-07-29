#!/usr/bin/env python3
"""Run the frozen P0-2 mask tests and publish their hash-bound receipt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


R2_ROOT = Path(__file__).resolve().parents[1]
if str(R2_ROOT) not in sys.path:
    sys.path.insert(0, str(R2_ROOT))

from src.revision.io import sha256_file  # noqa: E402
from src.revision.mask_validation_receipt import (  # noqa: E402
    produce_mask_validation_receipt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-module-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path, payload = produce_mask_validation_receipt(
        r2_root=R2_ROOT,
        output_path=args.output,
        expected_module_sha256=args.expected_module_sha256,
    )
    print(f"receipt={path}")
    print(f"receipt_sha256={sha256_file(path)}")
    print(
        f"production_scientific_eligibility={payload['production_scientific_eligibility']}"
    )
    raise SystemExit(payload["pytest_exit_code"])


if __name__ == "__main__":
    main()
