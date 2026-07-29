#!/usr/bin/env python3
"""Build P0-2-eligible confirmatory or explicit fixture P0-3/P0-4 inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.revision.input_builder import build_inputs, load_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_inputs(args.spec, args.out_dir)
    payload = load_json(manifest)
    print(
        f"wrote P0-3/P0-4 input build: {manifest} "
        f"confirmatory={payload['confirmatory']} status={payload['status']}"
    )


if __name__ == "__main__":
    main()
