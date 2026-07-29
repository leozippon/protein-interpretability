#!/usr/bin/env python3
"""Adjudicate immutable held-out pretrained-model P0-7 evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.revision.causal_adjudication import (  # noqa: E402
    adjudicate_pretrained_causal_interventions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = adjudicate_pretrained_causal_interventions(
        args.spec,
        args.spec_sha256,
        args.out_dir,
        cli_path=Path(__file__),
        command=[sys.executable, *sys.argv],
    )
    print(f"wrote complete pretrained P0-7 adjudication: {receipt}")


if __name__ == "__main__":
    main()
