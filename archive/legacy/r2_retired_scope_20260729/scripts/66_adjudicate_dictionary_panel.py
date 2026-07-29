#!/usr/bin/env python3
"""Publish the immutable eligibility receipt for a complete P0-2 panel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


R2_ROOT = Path(__file__).resolve().parents[1]
if str(R2_ROOT) not in sys.path:
    sys.path.insert(0, str(R2_ROOT))

from src.revision.dictionary_gate import (  # noqa: E402
    load_strict_json,
    sha256_file,
    write_eligibility_receipt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt_path, manifest_path = write_eligibility_receipt(
        args.spec, args.output_dir
    )
    receipt = load_strict_json(receipt_path)
    print(f"receipt={receipt_path}")
    print(f"receipt_sha256={sha256_file(receipt_path)}")
    print(f"manifest={manifest_path}")
    print(f"manifest_sha256={sha256_file(manifest_path)}")
    print(f"panel_status={receipt['panel_status']}")


if __name__ == "__main__":
    main()

