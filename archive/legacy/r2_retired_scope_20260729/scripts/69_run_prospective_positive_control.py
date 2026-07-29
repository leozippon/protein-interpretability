#!/usr/bin/env python3
"""Freeze or once-only execute the prospective P0-7 synthetic benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.revision.prospective_positive_control import (  # noqa: E402
    execute_frozen_prospective_benchmark,
    freeze_prospective_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--spec", type=Path, required=True)
    freeze.add_argument("--spec-sha256", required=True)
    freeze.add_argument("--out-dir", type=Path, required=True)

    execute = subparsers.add_parser("execute")
    execute.add_argument("--frozen-dir", type=Path, required=True)
    execute.add_argument("--freeze-manifest-sha256", required=True)
    execute.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [sys.executable, *sys.argv]
    if args.stage == "freeze":
        manifest = freeze_prospective_benchmark(
            args.spec,
            args.spec_sha256,
            args.out_dir,
            runner_path=Path(__file__),
            command=command,
        )
        print(f"wrote unexecuted prospective P0-7 freeze: {manifest}")
        return
    manifest = execute_frozen_prospective_benchmark(
        args.frozen_dir,
        args.freeze_manifest_sha256,
        args.out_dir,
        runner_path=Path(__file__),
        command=command,
    )
    print(f"wrote once-only prospective P0-7 result: {manifest}")


if __name__ == "__main__":
    main()
