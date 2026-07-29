#!/usr/bin/env python3
"""Run the trained autoregressive planted-mechanism control for P0-7."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.revision.causal_positive_control import generate_grammar_cohort  # noqa: E402
from src.revision.io import sha256_file, write_json, write_jsonl  # noqa: E402
from src.revision.trained_positive_control import (  # noqa: E402
    ControlConfig,
    run_trained_positive_control,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "results/npj_revision_20260716/trained_planted_control",
    )
    parser.add_argument("--cohort-seed", type=int, default=20260717)
    parser.add_argument("--per-split", type=int, default=48)
    parser.add_argument("--min-length", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--lm-steps", type=int, default=20)
    parser.add_argument("--clt-steps", type=int, default=150)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    records = generate_grammar_cohort(
        per_split=args.per_split,
        seed=args.cohort_seed,
        splits=("train", "discovery", "evaluation"),
        min_length=args.min_length,
        max_length=args.max_length,
    )
    config = ControlConfig(lm_steps=args.lm_steps, clt_steps=args.clt_steps)
    result = run_trained_positive_control(
        records,
        model_seeds=args.model_seeds,
        config=config,
        device=args.device,
        checkpoint_dir=args.out_dir / "checkpoints",
    )
    feature_rows = result.pop("feature_discovery_rows")
    intervention_rows = result.pop("intervention_rows")
    cohort_path = args.out_dir / "cohort.jsonl"
    feature_path = args.out_dir / "feature_discovery.jsonl"
    intervention_path = args.out_dir / "interventions.jsonl"
    summary_path = args.out_dir / "summary.json"
    write_jsonl(cohort_path, (record.to_dict() for record in records))
    write_jsonl(feature_path, feature_rows)
    write_jsonl(intervention_path, intervention_rows)
    result["artifacts"] = {
        "cohort": cohort_path.name,
        "feature_discovery": feature_path.name,
        "interventions": intervention_path.name,
        "checkpoints": [row["checkpoint"] for row in result["models"]],
    }
    write_json(summary_path, result)
    artifacts = (
        cohort_path,
        feature_path,
        intervention_path,
        summary_path,
        *sorted((args.out_dir / "checkpoints").glob("*.pt")),
    )
    write_json(
        args.out_dir / "run_manifest.json",
        {
            "schema_version": "r2-trained-planted-control-manifest-v1",
            "command": [sys.executable, *sys.argv],
            "parameters": {
                "cohort_seed": args.cohort_seed,
                "per_split": args.per_split,
                "min_length": args.min_length,
                "max_length": args.max_length,
                "model_seeds": args.model_seeds,
                "device": args.device,
                "config": result["config"],
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "torch": torch.__version__,
            },
            "source_hashes": {
                Path(__file__).name: sha256_file(Path(__file__)),
                "trained_positive_control.py": sha256_file(
                    PROJECT / "src/revision/trained_positive_control.py"
                ),
                "causal_positive_control.py": sha256_file(
                    PROJECT / "src/revision/causal_positive_control.py"
                ),
                "clt_trainer.py": sha256_file(PROJECT / "src/training/clt_trainer.py"),
                "statistics.py": sha256_file(PROJECT / "src/revision/statistics.py"),
            },
            "artifact_hashes": {
                str(path.relative_to(args.out_dir)): sha256_file(path) for path in artifacts
            },
        },
    )
    print(
        f"posthoc_development_smoke_passed={result['aggregate']['posthoc_development_smoke_passed']} "
        f"models={len(args.model_seeds)} device={args.device} out={args.out_dir}"
    )


if __name__ == "__main__":
    main()
