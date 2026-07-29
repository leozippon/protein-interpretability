#!/usr/bin/env python3
"""Execute every row of an immutable corrected-steering generation freeze."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.revision.steering_execution import run_execution  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--freeze-manifest-sha256", required=True)
    parser.add_argument("--execution-spec", type=Path, required=True)
    parser.add_argument("--execution-spec-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_execution(
        frozen_dir=args.frozen_dir,
        freeze_manifest_sha256=args.freeze_manifest_sha256,
        spec_path=args.execution_spec,
        spec_sha256=args.execution_spec_sha256,
        output_dir=args.out_dir,
        command=[sys.executable, *sys.argv],
    )
    print(f"wrote complete corrected-steering generations: {receipt}")


if __name__ == "__main__":
    main()
