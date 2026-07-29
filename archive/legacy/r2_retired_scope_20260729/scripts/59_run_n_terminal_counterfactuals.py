#!/usr/bin/env python3
"""Prepare or analyze the P0-5 N-terminal counterfactual factorial."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.models.model_loader import INFERENCE_DTYPE_VERIFICATION  # noqa: E402
from src.revision.io import sha256_file, write_json, write_jsonl  # noqa: E402
from src.revision.n_terminal_counterfactuals import (  # noqa: E402
    analyze_n_terminal_counterfactuals,
    build_counterfactual_variants,
    synthetic_n_terminal_fixture,
    validate_disjoint_cohorts,
    validate_equivalence_spec,
)
from src.revision.n_terminal_extractor import ACTIVATION_FINITE_CHECK  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--sequences", type=Path, help="JSONL with protein_id and natural MXX sequence")
    parser.add_argument("--measurements", type=Path, help="Complete condition x BOS measurement JSONL")
    parser.add_argument("--sequences-sha256")
    parser.add_argument("--measurements-sha256")
    parser.add_argument("--extractor-receipt", type=Path)
    parser.add_argument("--extractor-receipt-sha256")
    parser.add_argument(
        "--discovery-cohort",
        type=Path,
        help="JSONL discovery proteins; required for real held-out analysis",
    )
    parser.add_argument(
        "--equivalence-spec",
        type=Path,
        help="Frozen JSON alpha/margins; required for real held-out analysis",
    )
    parser.add_argument("--discovery-cohort-sha256")
    parser.add_argument("--equivalence-spec-sha256")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "results/npj_revision_20260716/n_terminal_counterfactuals",
    )
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument(
        "--control-count",
        type=int,
        help="Synthetic override only; real analysis uses the extractor-frozen count",
    )
    parser.add_argument("--synthetic-proteins", type=int, default=48)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            rows.append(
                json.loads(
                    line,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"{path}:{line_number}: non-finite constant {value}")
                    ),
                )
            )
    if not rows:
        raise ValueError(f"{path} contains no records")
    return rows


def read_json(path: Path) -> dict:
    value = json.loads(
        path.read_text(),
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"{path}: non-finite constant {constant}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require_sha256(value: object, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _verify_external_hash(path: Path, expected: object, label: str) -> str:
    digest = _require_sha256(expected, f"{label} SHA-256")
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"{label} path or externally supplied SHA-256 mismatch")
    return digest


def verify_extractor_inputs(args: argparse.Namespace) -> dict:
    """Bind real analysis to one externally pinned production extraction."""

    required = (
        "measurements",
        "measurements_sha256",
        "extractor_receipt",
        "extractor_receipt_sha256",
        "discovery_cohort",
        "discovery_cohort_sha256",
        "equivalence_spec",
        "equivalence_spec_sha256",
    )
    missing = [name for name in required if getattr(args, name, None) is None]
    if missing:
        raise ValueError(
            "real measurement analysis requires externally pinned inputs: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    sequences = args.sequences.resolve()
    measurements = args.measurements.resolve()
    discovery = args.discovery_cohort.resolve()
    equivalence = args.equivalence_spec.resolve()
    receipt_path = args.extractor_receipt.resolve()
    sequence_sha = _verify_external_hash(
        sequences, args.sequences_sha256, "natural cohort"
    )
    measurement_sha = _verify_external_hash(
        measurements, args.measurements_sha256, "extractor measurements"
    )
    discovery_sha = _verify_external_hash(
        discovery, args.discovery_cohort_sha256, "discovery cohort"
    )
    equivalence_sha = _verify_external_hash(
        equivalence, args.equivalence_spec_sha256, "equivalence specification"
    )
    receipt_sha = _verify_external_hash(
        receipt_path, args.extractor_receipt_sha256, "extractor receipt"
    )
    receipt = read_json(receipt_path)
    if (
        receipt.get("schema_version") != "r2-p05-pretrained-extraction-manifest-v4"
        or receipt.get("status") != "verified_production_complete"
        or receipt.get("execution_mode") != "production"
    ):
        raise ValueError("extractor receipt is not verified production evidence")
    model = receipt.get("model")
    if (
        not isinstance(model, dict)
        or model.get("model_inference_dtype") != "bfloat16"
        or model.get("observed_model_parameter_dtypes") != ["bfloat16"]
        or model.get("model_inference_dtype_verification")
        != INFERENCE_DTYPE_VERIFICATION
        or model.get("model_inference_dtype_verified") is not True
        or model.get("activation_finiteness_check") != ACTIVATION_FINITE_CHECK
        or model.get("activation_finiteness_verified") is not True
    ):
        raise ValueError("extractor receipt lacks verified bfloat16 numerical integrity")

    artifact_hashes = receipt.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise ValueError("extractor receipt has no artifact inventory")
    for name, expected in artifact_hashes.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
        ):
            raise ValueError("extractor artifact names must be plain file names")
        _verify_external_hash(
            receipt_path.parent / name,
            expected,
            f"extractor artifact {name}",
        )
    expected_sequences = (receipt_path.parent / "natural_cohort.jsonl").resolve()
    expected_measurements = (receipt_path.parent / "measurements.jsonl").resolve()
    if sequences != expected_sequences or measurements != expected_measurements:
        raise ValueError(
            "sequences and measurements must be the artifacts colocated with the "
            "extractor receipt"
        )
    if (
        artifact_hashes.get(expected_sequences.name) != sequence_sha
        or artifact_hashes.get(expected_measurements.name) != measurement_sha
    ):
        raise ValueError("external input hashes differ from the extractor receipt")
    feature_match_path = receipt_path.parent / "feature_matches.json"
    if feature_match_path.name not in artifact_hashes:
        raise ValueError("extractor receipt does not bind feature_matches.json")
    feature_match_rows = json.loads(
        feature_match_path.read_text(),
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(
                f"{feature_match_path}: non-finite constant {constant}"
            )
        ),
    )

    inputs = receipt.get("inputs")
    discovery_descriptor = (
        inputs.get("discovery_cohort") if isinstance(inputs, dict) else None
    )
    if (
        not isinstance(discovery_descriptor, dict)
        or set(discovery_descriptor) != {"path", "sha256", "n_records"}
        or Path(discovery_descriptor["path"]).resolve() != discovery
        or discovery_descriptor["sha256"] != discovery_sha
    ):
        raise ValueError("discovery cohort differs from the extractor-bound input")

    matching = receipt.get("matching")
    feature_matching = matching.get("feature") if isinstance(matching, dict) else None
    match_ids = matching.get("feature_match_ids") if isinstance(matching, dict) else None
    if not isinstance(feature_matching, dict):
        raise ValueError("extractor receipt has no frozen feature-matching contract")
    control_count = feature_matching.get("control_count")
    if type(control_count) is not int or control_count < 1:
        raise ValueError("extractor receipt has an invalid frozen control count")
    if (
        not isinstance(match_ids, list)
        or not match_ids
        or len(set(match_ids)) != len(match_ids)
        or any(_require_sha256(value, "feature_match_id") != value for value in match_ids)
    ):
        raise ValueError("extractor receipt has invalid frozen feature match IDs")
    frozen_matches = []
    required_match_fields = {
        "feature_match_id",
        "model",
        "layer",
        "target_feature",
        "control_features",
        "max_abs_log10_firing_ratio",
        "max_abs_log_input_norm_ratio",
        "target_profile",
        "controls",
        "method",
    }
    if not isinstance(feature_match_rows, list) or not feature_match_rows:
        raise ValueError("feature_matches.json must contain a non-empty array")
    for row in feature_match_rows:
        if not isinstance(row, dict) or set(row) != required_match_fields:
            raise ValueError("feature_matches.json rows differ from the frozen schema")
        controls = row["control_features"]
        if (
            row["feature_match_id"] not in match_ids
            or not isinstance(row["model"], str)
            or not row["model"]
            or type(row["layer"]) is not int
            or type(row["target_feature"]) is not int
            or not isinstance(controls, list)
            or len(controls) != control_count
            or len(set(controls)) != len(controls)
            or any(type(feature) is not int for feature in controls)
        ):
            raise ValueError("feature_matches.json contains an invalid frozen pair")
        frozen_matches.append(
            {
                "feature_match_id": row["feature_match_id"],
                "model": row["model"],
                "layer": row["layer"],
                "target_feature": row["target_feature"],
                "control_features": sorted(controls),
            }
        )
    if sorted(row["feature_match_id"] for row in frozen_matches) != sorted(match_ids):
        raise ValueError("feature_matches.json inventory differs from the receipt")
    if args.control_count is not None and args.control_count != control_count:
        raise ValueError("--control-count cannot change the extractor-frozen match sets")
    return {
        "receipt": {"path": str(receipt_path), "sha256": receipt_sha},
        "natural_cohort": {"path": str(sequences), "sha256": sequence_sha},
        "measurements": {"path": str(measurements), "sha256": measurement_sha},
        "discovery_cohort": {"path": str(discovery), "sha256": discovery_sha},
        "equivalence_spec": {"path": str(equivalence), "sha256": equivalence_sha},
        "control_count": control_count,
        "feature_match_ids": match_ids,
        "feature_matches": sorted(
            frozen_matches, key=lambda row: row["feature_match_id"]
        ),
    }


def _run(args: argparse.Namespace, out_dir: Path) -> dict:
    if args.synthetic:
        records, variants, measurements = synthetic_n_terminal_fixture(
            seed=args.seed, n_proteins=args.synthetic_proteins
        )
        equivalence_spec = validate_equivalence_spec(
            {
                "alpha": 0.05,
                "multiplicity": "holm_all_feature_control_and_protein_pair_did_cells",
                "margins": {
                    "feature_activation_pre": 0.10,
                    "normalized_received_attention": 0.02,
                    "suffix_nll_increase_key_masked": 0.05,
                    "suffix_observed_token_logit_change_key_masked": 0.05,
                },
            }
        )
        cohort_validation = {
            "status": "synthetic_disjointness_fixture_only",
            "discovery_evaluation_disjoint": True,
            "n_evaluation_proteins": len(records),
        }
        input_hashes = {}
        extractor_binding = None
        control_count = 2 if args.control_count is None else args.control_count
    else:
        if args.sequences_sha256 is None:
            raise ValueError("real runs require --sequences-sha256")
        _verify_external_hash(
            args.sequences.resolve(), args.sequences_sha256, "natural cohort"
        )
        records = read_jsonl(args.sequences)
        variants = build_counterfactual_variants(records)
        measurements = read_jsonl(args.measurements) if args.measurements else None
        input_hashes = {
            "natural_cohort": {
                "path": str(args.sequences.resolve()),
                "sha256": args.sequences_sha256,
            }
        }
        extractor_binding = None
        control_count = args.control_count
        if args.measurements:
            extractor_binding = verify_extractor_inputs(args)
            control_count = extractor_binding["control_count"]
            discovery_records = read_jsonl(args.discovery_cohort)
            equivalence_spec = validate_equivalence_spec(read_json(args.equivalence_spec))
            cohort_validation = validate_disjoint_cohorts(records, discovery_records)
            input_hashes = {
                key: value
                for key, value in extractor_binding.items()
                if isinstance(value, dict)
            }
        else:
            equivalence_spec = None
            cohort_validation = None

    cohort_path = out_dir / "natural_cohort.jsonl"
    variants_path = out_dir / "counterfactual_variants.jsonl"
    write_jsonl(cohort_path, records)
    write_jsonl(variants_path, (variant.to_dict() for variant in variants))
    artifacts = [cohort_path, variants_path]
    if measurements is None:
        summary = {
            "schema_version": "r2-n-terminal-counterfactual-preparation-v1",
            "status": "variants_prepared_measurements_pending",
            "scope": "No pretrained-model inference or scientific gate was run.",
            "n_proteins": len(records),
            "n_variants": len(variants),
        }
    else:
        summary = analyze_n_terminal_counterfactuals(
            measurements,
            variants=variants,
            equivalence_spec=equivalence_spec,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            control_count=control_count,
        )
        if extractor_binding is not None:
            observed_matches = [
                {
                    "feature_match_id": row["feature_match_id"],
                    "model": row["target"]["model"],
                    "layer": row["target"]["layer"],
                    "target_feature": row["target"]["feature"],
                    "control_features": [
                        control["profile"]["feature"]
                        for control in row["controls"]
                    ],
                }
                for row in summary["feature_matches"]
            ]
            if observed_matches != extractor_binding["feature_matches"]:
                raise ValueError(
                    "measurement feature pairs differ from feature_matches.json"
                )
        normalized_rows = summary.pop("normalized_rows")
        measurements_path = out_dir / "normalized_measurements.jsonl"
        write_jsonl(measurements_path, normalized_rows)
        summary["status"] = (
            "synthetic_pipeline_validation_only"
            if args.synthetic
            else "extractor_bound_heldout_measurements_analyzed"
        )
        summary["artifacts"] = {"normalized_measurements": measurements_path.name}
        summary["cohort_validation"] = cohort_validation
        summary["extractor_binding"] = extractor_binding
        artifacts.append(measurements_path)
    summary_path = out_dir / "summary.json"
    write_json(summary_path, summary)
    artifacts.append(summary_path)
    write_json(
        out_dir / "run_manifest.json",
        {
            "schema_version": "r2-n-terminal-counterfactual-manifest-v2",
            "command": [sys.executable, *sys.argv],
            "parameters": {
                "seed": args.seed,
                "n_bootstrap": args.n_bootstrap,
                "control_count": control_count,
                "synthetic": args.synthetic,
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "input_hashes": input_hashes,
            "source_hashes": {
                Path(__file__).name: sha256_file(Path(__file__)),
                "n_terminal_counterfactuals.py": sha256_file(
                    PROJECT / "src/revision/n_terminal_counterfactuals.py"
                ),
                "io.py": sha256_file(PROJECT / "src/revision/io.py"),
            },
            "artifact_hashes": {path.name: sha256_file(path) for path in artifacts},
        },
    )
    if extractor_binding is not None:
        repeated = verify_extractor_inputs(args)
        if repeated != extractor_binding:
            raise RuntimeError("extractor-bound inputs changed during analysis")
    return summary


def run(args: argparse.Namespace) -> dict:
    """Publish a complete new result directory or leave no partial output."""

    destination = args.out_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"stale counterfactual staging directory: {staging}")
    staging.mkdir()
    try:
        summary = _run(args, staging)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(f"status={summary['status']} out={args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
