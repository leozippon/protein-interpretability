#!/usr/bin/env python3
"""Run P0-8 nested repeated recoverability on an NPZ cache or CPU fixture."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
import sklearn

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.revision.io import sha256_file, write_json, write_jsonl  # noqa: E402
from src.revision.input_builder import (  # noqa: E402
    ACTIVATION_FINITE_CHECK,
    MODEL_INFERENCE_DTYPE_VERIFICATION,
    load_json,
)
from src.revision.nested_recoverability import (  # noqa: E402
    run_nested_recoverability,
    synthetic_recoverability_fixture,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--input-sha256")
    parser.add_argument("--input-receipt", type=Path)
    parser.add_argument("--input-receipt-sha256")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "results/npj_revision_20260716/nested_recoverability",
    )
    parser.add_argument(
        "--task-type",
        choices=("classification", "regression"),
        default="classification",
    )
    parser.add_argument("--ceiling", default="ceiling")
    parser.add_argument("--floors", nargs="+", default=["code_seed_0", "code_seed_1"])
    parser.add_argument(
        "--analysis-seeds", type=int, nargs="+", default=[101, 211, 307, 401, 503]
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument(
        "--comparison-dimension",
        type=int,
        help="exact fold-fitted dimension for every primary representation arm",
    )
    parser.add_argument(
        "--active-width-dimension",
        type=int,
        help="optional smaller, separately labelled rank-sensitivity dimension",
    )
    parser.add_argument(
        "--controls",
        nargs="*",
        default=["pca", "random_projection", "nmf", "ica", "random_dictionary"],
    )
    parser.add_argument("--synthetic-seed", type=int, default=20260717)
    parser.add_argument("--synthetic-groups", type=int, default=48)
    parser.add_argument("--synthetic-samples-per-group", type=int, default=4)
    return parser.parse_args()


def verify_production_input_receipt(
    args: argparse.Namespace, input_path: Path
) -> tuple[Path, str]:
    """Require the immutable builder receipt before enabling real-mode claims."""

    if args.input_receipt is None or args.input_receipt_sha256 is None:
        raise ValueError(
            "real runs require --input-receipt and --input-receipt-sha256"
        )
    receipt_path = args.input_receipt.resolve()
    if sha256_file(receipt_path) != args.input_receipt_sha256:
        raise ValueError("P0-8 input receipt SHA-256 mismatch")
    receipt = load_json(receipt_path)
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != "r2_p0_8_input_receipt_v3"
        or receipt.get("status")
        != "verified_production_inputs_not_scientifically_adjudicated"
    ):
        raise ValueError("P0-8 input receipt is not production-eligible")
    model = receipt.get("model")
    if (
        not isinstance(model, dict)
        or model.get("model_inference_dtype") != "bfloat16"
        or model.get("observed_model_parameter_dtypes") != ["bfloat16"]
        or model.get("model_inference_dtype_verification")
        != MODEL_INFERENCE_DTYPE_VERIFICATION
        or model.get("model_inference_dtype_verified") is not True
        or model.get("activation_finiteness_check") != ACTIVATION_FINITE_CHECK
        or model.get("activation_finiteness_verified") is not True
    ):
        raise ValueError("P0-8 input receipt lacks verified bfloat16 finite inference")
    expected_input = receipt_path.parent / "nested_recoverability_input.npz"
    expected_runner = receipt_path.parent / "nested_recoverability_runner_spec.json"
    if input_path != expected_input.resolve():
        raise ValueError("input NPZ is not the one colocated with the receipt")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {
        expected_input.name,
        expected_runner.name,
    }:
        raise ValueError("P0-8 receipt output inventory differs")
    if (
        outputs[expected_input.name] != args.input_sha256
        or sha256_file(expected_input) != outputs[expected_input.name]
        or sha256_file(expected_runner) != outputs[expected_runner.name]
    ):
        raise ValueError("P0-8 receipt output hashes differ")
    runner = load_json(expected_runner)
    expected_arguments = {
        "task_type": args.task_type,
        "ceiling": args.ceiling,
        "floors": args.floors,
        "analysis_seeds": args.analysis_seeds,
        "outer_splits": args.outer_splits,
        "inner_splits": args.inner_splits,
        "n_bootstrap": args.n_bootstrap,
        "comparison_dimension": args.comparison_dimension,
        "active_width_dimension": args.active_width_dimension,
        "controls": args.controls,
    }
    if (
        not isinstance(runner, dict)
        or set(runner)
        != {
            "schema_version",
            "status",
            "script",
            "input",
            "arguments",
            "row_order_sha256",
            "confirmatory_real",
        }
        or runner["schema_version"] != "r2_p0_8_runner_spec_v2"
        or runner["status"] != "verified_inputs_not_scientifically_adjudicated"
        or runner["script"] != "scripts/55_run_nested_recoverability.py"
        or runner["input"]
        != {"path": expected_input.name, "sha256": args.input_sha256}
        or runner["arguments"] != expected_arguments
        or runner["confirmatory_real"] is not True
    ):
        raise ValueError("script-55 arguments differ from the frozen runner spec")
    return receipt_path, args.input_receipt_sha256


def load_npz(
    path: Path,
) -> tuple[dict, np.ndarray, np.ndarray, dict | None, dict | None]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"y", "groups"}
        if not required <= set(payload.files):
            raise ValueError("input NPZ must contain y and groups")
        representations: dict[str, dict[str, np.ndarray]] = {}
        for key in payload.files:
            if not key.startswith("rep__"):
                continue
            parts = key.split("__", 2)
            if len(parts) != 3 or not parts[1] or not parts[2]:
                raise ValueError(f"invalid representation key: {key}")
            representations.setdefault(parts[1], {})[parts[2]] = payload[key]
        if not representations:
            raise ValueError("input NPZ has no rep__<name>__<layer> arrays")
        quality: dict[str, dict[str, dict[str, np.ndarray]]] = {
            "reconstruction_error": {},
            "intervention_effect": {},
        }
        for key in payload.files:
            if not key.startswith("quality__"):
                continue
            parts = key.split("__")
            if (
                len(parts) != 4
                or parts[1] not in quality
                or not parts[2]
                or not parts[3]
            ):
                raise ValueError(f"invalid quality key: {key}")
            quality[parts[1]].setdefault(parts[2], {})[parts[3]] = payload[key]
        reconstruction = quality["reconstruction_error"] or None
        intervention = quality["intervention_effect"] or None
        return (
            representations,
            payload["y"],
            payload["groups"],
            reconstruction,
            intervention,
        )


def save_fixture(
    path: Path,
    representations: dict,
    y: np.ndarray,
    groups: np.ndarray,
    reconstruction: dict,
    intervention: dict,
) -> None:
    arrays = {
        "y": y,
        "groups": groups,
    }
    for name, layers in representations.items():
        for layer, value in layers.items():
            arrays[f"rep__{name}__{layer}"] = value
    for quality_name, values in (
        ("reconstruction_error", reconstruction),
        ("intervention_effect", intervention),
    ):
        for floor_name, layers in values.items():
            for layer, value in layers.items():
                arrays[f"quality__{quality_name}__{floor_name}__{layer}"] = value
    np.savez_compressed(path, **arrays)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {args.out_dir}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_receipt: tuple[Path, str] | None = None
    if args.synthetic:
        if any(
            value is not None
            for value in (
                args.input_sha256,
                args.input_receipt,
                args.input_receipt_sha256,
            )
        ):
            raise ValueError("input hashes and receipts are only valid with --input")
        dictionary_seeds = [
            int(name.removeprefix("code_seed_")) for name in args.floors
        ]
        representations, y, groups, reconstruction, intervention = (
            synthetic_recoverability_fixture(
                seed=args.synthetic_seed,
                n_groups=args.synthetic_groups,
                samples_per_group=args.synthetic_samples_per_group,
                dictionary_seeds=dictionary_seeds,
            )
        )
        input_path = args.out_dir / "synthetic_input.npz"
        save_fixture(
            input_path, representations, y, groups, reconstruction, intervention
        )
        comparison_dimension = (
            5 if args.comparison_dimension is None else args.comparison_dimension
        )
    else:
        input_path = args.input.resolve()
        if args.input_sha256 is None:
            raise ValueError("real runs require --input-sha256")
        if args.comparison_dimension is None:
            raise ValueError("real runs require --comparison-dimension")
        if sha256_file(input_path) != args.input_sha256:
            raise ValueError("input NPZ SHA-256 mismatch")
        input_receipt = verify_production_input_receipt(args, input_path)
        representations, y, groups, reconstruction, intervention = load_npz(input_path)
        if reconstruction is None or intervention is None:
            raise ValueError(
                "confirmatory real input must contain seed/layer-specific "
                "reconstruction error and intervention effect"
            )
        comparison_dimension = args.comparison_dimension
    input_sha256 = sha256_file(input_path)
    result = run_nested_recoverability(
        representations,
        y,
        groups,
        ceiling_name=args.ceiling,
        floor_names=args.floors,
        task_type=args.task_type,
        analysis_seeds=args.analysis_seeds,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        control_methods=args.controls,
        comparison_dimension=comparison_dimension,
        active_width_dimension=args.active_width_dimension,
        n_bootstrap=args.n_bootstrap,
        reconstruction_error_by_floor_layer=reconstruction,
        intervention_effect_by_floor_layer=intervention,
        confirmatory_real=not args.synthetic,
    )
    fold_manifest = result.pop("fold_manifest")
    fold_results = result.pop("fold_results")
    predictions = result.pop("outer_predictions")
    fold_path = args.out_dir / "fold_manifest.json"
    rows_path = args.out_dir / "fold_results.jsonl"
    predictions_path = args.out_dir / "outer_predictions.json"
    summary_path = args.out_dir / "summary.json"
    write_json(fold_path, fold_manifest)
    write_jsonl(rows_path, fold_results)
    write_json(predictions_path, predictions)
    result["artifacts"] = {
        "input": str(input_path),
        "fold_manifest": fold_path.name,
        "fold_results": rows_path.name,
        "outer_predictions": predictions_path.name,
    }
    write_json(summary_path, result)
    artifacts = [fold_path, rows_path, predictions_path, summary_path]
    if args.synthetic:
        artifacts.append(input_path)
    write_json(
        args.out_dir / "run_manifest.json",
        {
            "schema_version": "r2-nested-recoverability-manifest-v3",
            "command": [sys.executable, *sys.argv],
            "parameters": {
                "task_type": args.task_type,
                "ceiling": args.ceiling,
                "floors": args.floors,
                "analysis_seeds": args.analysis_seeds,
                "outer_splits": args.outer_splits,
                "inner_splits": args.inner_splits,
                "n_bootstrap": args.n_bootstrap,
                "comparison_dimension": comparison_dimension,
                "active_width_dimension": args.active_width_dimension,
                "controls": args.controls,
                "confirmatory_real": not args.synthetic,
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "input": {"path": str(input_path), "sha256": input_sha256},
            "input_receipt": (
                None
                if input_receipt is None
                else {
                    "path": str(input_receipt[0]),
                    "sha256": input_receipt[1],
                }
            ),
            "source_hashes": {
                Path(__file__).name: sha256_file(Path(__file__)),
                "nested_recoverability.py": sha256_file(
                    PROJECT / "src/revision/nested_recoverability.py"
                ),
                "statistics.py": sha256_file(PROJECT / "src/revision/statistics.py"),
                "io.py": sha256_file(PROJECT / "src/revision/io.py"),
            },
            "artifact_hashes": {path.name: sha256_file(path) for path in artifacts},
        },
    )
    print(
        f"nested_recoverability samples={result['n_samples']} "
        f"groups={result['n_identity_groups']} out={args.out_dir}"
    )


if __name__ == "__main__":
    main()
