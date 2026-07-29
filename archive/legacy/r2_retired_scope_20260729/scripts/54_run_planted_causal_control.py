#!/usr/bin/env python3
"""Run the CPU planted-grammar sensitivity control for P0-7."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import scipy

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.revision.causal_positive_control import (  # noqa: E402
    generate_grammar_cohort,
    run_planted_positive_control,
)
from src.revision.io import sha256_file, write_json, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "results/npj_revision_20260716/planted_causal_control",
    )
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--per-split", type=int, default=256)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[11, 29, 47, 71, 89])
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--matched-controls", type=int, default=3)
    parser.add_argument("--selection-alpha", type=float, default=0.05)
    parser.add_argument("--equivalence-margin", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = generate_grammar_cohort(per_split=args.per_split, seed=args.seed)
    result = run_planted_positive_control(
        records,
        model_seeds=args.model_seeds,
        strength=args.strength,
        matched_control_count=args.matched_controls,
        selection_alpha=args.selection_alpha,
        equivalence_margin=args.equivalence_margin,
    )
    feature_rows = result.pop("feature_discovery_rows")
    intervention_rows = result.pop("intervention_rows")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = args.out_dir / "cohort.jsonl"
    features_path = args.out_dir / "feature_discovery.jsonl"
    interventions_path = args.out_dir / "interventions.jsonl"
    summary_path = args.out_dir / "summary.json"
    write_jsonl(cohort_path, (record.to_dict() for record in records))
    write_jsonl(features_path, feature_rows)
    write_jsonl(interventions_path, intervention_rows)
    result["artifacts"] = {
        "cohort": cohort_path.name,
        "feature_discovery": features_path.name,
        "interventions": interventions_path.name,
    }
    write_json(summary_path, result)
    artifacts = [cohort_path, features_path, interventions_path, summary_path]
    write_json(
        args.out_dir / "run_manifest.json",
        {
            "schema_version": "r2-planted-causal-control-manifest-v1",
            "command": [sys.executable, *sys.argv],
            "parameters": {
                "seed": args.seed,
                "per_split": args.per_split,
                "model_seeds": args.model_seeds,
                "strength": args.strength,
                "matched_controls": args.matched_controls,
                "selection_alpha": args.selection_alpha,
                "equivalence_margin": args.equivalence_margin,
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "source_hashes": {
                Path(__file__).name: sha256_file(Path(__file__)),
                "causal_positive_control.py": sha256_file(
                    PROJECT / "src/revision/causal_positive_control.py"
                ),
                "statistics.py": sha256_file(PROJECT / "src/revision/statistics.py"),
                "io.py": sha256_file(PROJECT / "src/revision/io.py"),
            },
            "artifact_hashes": {path.name: sha256_file(path) for path in artifacts},
        },
    )
    print(
        f"analytic_synthetic_checks_passed={result['aggregate']['analytic_synthetic_checks_passed']} "
        f"models={len(args.model_seeds)} out={args.out_dir}"
    )


if __name__ == "__main__":
    main()
