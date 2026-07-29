#!/usr/bin/env python3
"""Run continuous conditional-semantic tests from an immutable NPZ bundle.

The script evaluates prespecified activation columns after position, hashed
k-mer, input-norm, protein-length, and sequence-source covariates.  It uses the
same protein/family folds for sparse, dense, and randomized representations,
within-protein label randomization, protein bootstrap intervals, and one global
BH correction across representations, features, layers, labels, and folds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.revision.semantics import (  # noqa: E402
    blocked_fold_ids,
    fold_assignment_hash,
    low_level_design,
    run_conditional_semantics,
)
from src.revision.input_builder import (  # noqa: E402
    ACTIVATION_FINITE_CHECK,
    MODEL_INFERENCE_DTYPE_VERIFICATION,
)


P0_4_RUNNER_SCHEMA = "r2_p0_4_conditional_semantics_spec_v3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path: Path) -> dict:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256_value(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def validate_input_provenance(spec: dict) -> None:
    """Require v3 finite-inference provenance for builder-backed inputs."""

    schema = spec.get("schema_version")
    if schema is None and spec["confirmatory"] is False:
        return
    provenance = spec.get("input_provenance")
    if schema != P0_4_RUNNER_SCHEMA or not isinstance(provenance, dict):
        raise ValueError("conditional-semantics input provenance is not v3 eligible")
    declared = provenance.get("model_inference_dtype")
    if (
        declared not in {"float16", "bfloat16", "float32"}
        or spec["confirmatory"]
        and declared != "bfloat16"
        or provenance.get("observed_model_parameter_dtypes") != [declared]
        or provenance.get("model_inference_dtype_verification")
        != MODEL_INFERENCE_DTYPE_VERIFICATION
        or provenance.get("model_inference_dtype_verified") is not True
        or provenance.get("activation_finiteness_check") != ACTIVATION_FINITE_CHECK
        or provenance.get("activation_finiteness_verified") is not True
    ):
        raise ValueError("conditional-semantics finite-inference provenance is ineligible")


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def array(bundle: np.lib.npyio.NpzFile, key: str, name: str, n: int | None = None) -> np.ndarray:
    if key not in bundle.files:
        raise KeyError(f"missing NPZ array {key!r} for {name}")
    value = bundle[key]
    if value.ndim == 0 or (n is not None and value.shape[0] != n):
        raise ValueError(f"array {key!r} for {name} has an invalid row dimension")
    return value


def category_counts(values: np.ndarray) -> list[dict]:
    categories, counts = np.unique(values.astype(str), return_counts=True)
    return [
        {"category": str(category), "count": int(count), "fraction": float(count / values.size)}
        for category, count in zip(categories, counts)
    ]


def load_prospective_power_plan(
    spec: dict,
    spec_path: Path,
    feature_names: dict[str, list[str]],
    label_names: set[str],
) -> tuple[dict[tuple[str, str, str, str], float] | None, dict | None]:
    """Load independently estimated standard errors for prospective MDEs."""

    confirmatory = spec["confirmatory"]
    reference = spec.get("prospective_power_plan")
    if reference is None:
        if confirmatory:
            raise ValueError(
                "confirmatory semantic runs require a SHA-bound prospective_power_plan"
            )
        return None, None
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        raise ValueError("prospective_power_plan requires path and sha256 strings")
    expected_hash = sha256_value(reference.get("sha256"), "prospective_power_plan.sha256")
    path = resolve_path(reference["path"], spec_path.parent)
    if not path.is_file() or path.suffix != ".json":
        raise FileNotFoundError(f"prospective power plan must be an existing JSON file: {path}")
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise ValueError("prospective power-plan SHA-256 mismatch")
    plan = strict_json(path)
    if plan.get("schema_version") != 1:
        raise ValueError("prospective power plan requires schema_version 1")
    source = plan.get("independent_source")
    if not isinstance(source, dict):
        raise ValueError("prospective power plan requires independent_source metadata")
    for field in ("description", "standard_error_method"):
        if not isinstance(source.get(field), str) or not source[field].strip():
            raise ValueError(f"independent_source.{field} must be a non-empty string")
    for field in ("run_manifest_sha256", "cohort_sha256"):
        sha256_value(source.get(field), f"independent_source.{field}")
    if source.get("independent_of_confirmatory_data") is not True:
        raise ValueError(
            "independent_source.independent_of_confirmatory_data must be true"
        )

    rows = plan.get("standard_errors_delta_mse")
    if not isinstance(rows, list) or not rows:
        raise ValueError("prospective power plan requires non-empty standard_errors_delta_mse")
    expected_keys = {
        (representation, feature, label, blocking)
        for representation, features in feature_names.items()
        for feature in features
        for label in label_names
        for blocking in ("protein", "family")
    }
    standard_errors: dict[tuple[str, str, str, str], float] = {}
    required = {"representation", "feature", "label", "blocking", "standard_error_delta_mse"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError(
                f"power-plan row {index} must contain exactly {sorted(required)}"
            )
        key = tuple(str(row[field]) for field in ("representation", "feature", "label", "blocking"))
        if key in standard_errors:
            raise ValueError(f"duplicate prospective power-plan hypothesis {key!r}")
        value = float(row["standard_error_delta_mse"])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"power-plan standard error for {key!r} must be finite and positive")
        standard_errors[key] = value
    if set(standard_errors) != expected_keys:
        missing = sorted(expected_keys - set(standard_errors))
        extra = sorted(set(standard_errors) - expected_keys)
        raise ValueError(
            "prospective power plan must exactly cover all hypotheses; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    metadata = {
        "path": str(path),
        "sha256": observed_hash,
        "independent_source": source,
        "n_standard_errors": len(standard_errors),
    }
    return standard_errors, metadata


def validate_controls(spec: dict, representations: list[dict], labels: list[dict], bundle) -> None:
    if not spec.get("confirmatory", True):
        return
    test = spec.get("test", {})
    if int(test.get("n_permutations", 1_000)) < 1_000:
        raise ValueError("confirmatory semantic runs require at least 1,000 permutations")
    if int(test.get("n_bootstrap", 1_000)) < 1_000:
        raise ValueError("confirmatory semantic runs require at least 1,000 protein bootstraps")
    representation_roles = {record.get("role") for record in representations}
    if not {"sparse", "dense", "randomized"} <= representation_roles:
        raise ValueError("confirmatory representations require sparse, dense, and randomized roles")
    if any(not isinstance(record.get("construction"), str) or not record["construction"] for record in representations):
        raise ValueError("confirmatory representations require frozen construction metadata")
    for record in representations:
        if record["role"] == "randomized" and type(record.get("seed")) is not int:
            raise ValueError("randomized representations require an integer construction seed")
    biological = [record for record in labels if record.get("role") == "biological"]
    negative = [record for record in labels if record.get("role") == "negative"]
    if not biological or not negative:
        raise ValueError("confirmatory labels require biological and matched negative controls")
    if any(not isinstance(record.get("construction"), str) or not record["construction"] for record in labels):
        raise ValueError("confirmatory labels require frozen construction metadata")
    by_name = {record["name"]: record for record in labels}
    biological_names = {record["name"] for record in biological}
    for record in labels:
        if len(np.unique(bundle[record["array"]].astype(str))) != 2:
            raise ValueError(
                f"confirmatory label {record['name']!r} must be one prespecified binary hypothesis"
            )
    for control in negative:
        if type(control.get("seed")) is not int:
            raise ValueError(f"negative control {control['name']!r} requires an integer seed")
        if control.get("matched_to") not in biological_names:
            raise ValueError(
                f"negative control {control['name']!r} must name a biological matched_to label"
            )
    for target in biological:
        controls = [record for record in negative if record.get("matched_to") == target["name"]]
        if not controls:
            raise ValueError(f"biological label {target['name']!r} lacks a matched negative control")
        target_counts = sorted(
            np.unique(bundle[target["array"]].astype(str), return_counts=True)[1].tolist()
        )
        for control in controls:
            if control.get("matched_to") not in by_name:
                raise ValueError(f"negative control {control['name']!r} has unknown matched_to")
            control_counts = sorted(
                np.unique(bundle[control["array"]].astype(str), return_counts=True)[1].tolist()
            )
            if control_counts != target_counts:
                raise ValueError(
                    f"negative control {control['name']!r} does not match prevalence of {target['name']!r}"
                )


def run(spec_path: Path, out_dir: Path) -> None:
    spec = strict_json(spec_path)
    if type(spec.get("confirmatory", True)) is not bool:
        raise ValueError("confirmatory must be a boolean")
    spec["confirmatory"] = bool(spec.get("confirmatory", True))
    validate_input_provenance(spec)
    data = spec.get("data")
    arrays = spec.get("arrays")
    representations_spec = spec.get("representations")
    labels_spec = spec.get("labels")
    if not isinstance(data, dict) or not isinstance(arrays, dict):
        raise ValueError("run spec requires data and arrays objects")
    if not isinstance(representations_spec, list) or not representations_spec:
        raise ValueError("run spec requires non-empty representations")
    if not isinstance(labels_spec, list) or not labels_spec:
        raise ValueError("run spec requires non-empty labels")
    bundle_path = resolve_path(str(data.get("path")), spec_path.parent)
    if not bundle_path.is_file() or bundle_path.suffix != ".npz":
        raise FileNotFoundError(f"data.path must name an existing NPZ bundle: {bundle_path}")
    bundle_hash = sha256_file(bundle_path)
    if data.get("sha256") != bundle_hash:
        raise ValueError("NPZ SHA-256 mismatch or missing expected hash")

    required_arrays = (
        "protein_id",
        "family_id",
        "position",
        "kmer",
        "input_norm",
        "protein_length",
        "sequence_source",
    )
    if any(not isinstance(arrays.get(name), str) for name in required_arrays):
        raise ValueError(f"arrays must map every required field: {', '.join(required_arrays)}")
    with np.load(bundle_path, allow_pickle=False) as bundle:
        proteins = array(bundle, arrays["protein_id"], "protein_id")
        if proteins.ndim != 1 or proteins.size < 2:
            raise ValueError("protein_id must be a vector with at least two observations")
        n = proteins.size
        families = array(bundle, arrays["family_id"], "family_id", n)
        position = array(bundle, arrays["position"], "position", n)
        kmer = array(bundle, arrays["kmer"], "kmer", n)
        input_norm = array(bundle, arrays["input_norm"], "input_norm", n)
        protein_length = array(bundle, arrays["protein_length"], "protein_length", n)
        sequence_source = array(bundle, arrays["sequence_source"], "sequence_source", n)
        if any(value.ndim != 1 for value in (families, position, kmer, input_norm, protein_length, sequence_source)):
            raise ValueError("metadata and covariate arrays must be one-dimensional")

        names = set()
        representations = {}
        feature_names = {}
        representation_roles = {}
        representation_metadata = {}
        for record in representations_spec:
            if not isinstance(record, dict) or not all(
                isinstance(record.get(key), str) for key in ("name", "role", "array")
            ):
                raise ValueError("representation records require name, role, and array strings")
            name = record["name"]
            if name in names:
                raise ValueError(f"duplicate representation name {name!r}")
            names.add(name)
            values = array(bundle, record["array"], name, n)
            if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
                raise ValueError(f"representation {name!r} must be a numeric matrix")
            supplied_names = record.get("feature_names")
            if (
                not isinstance(supplied_names, list)
                or len(supplied_names) != values.shape[1]
                or not all(isinstance(value, str) and value for value in supplied_names)
            ):
                raise ValueError(f"representation {name!r} requires one feature name per column")
            if len(set(supplied_names)) != len(supplied_names):
                raise ValueError(f"representation {name!r} feature names must be unique")
            representations[name] = values
            feature_names[name] = supplied_names
            representation_roles[name] = record["role"]
            representation_metadata[name] = {
                "role": record["role"],
                "construction": record.get("construction"),
                "seed": record.get("seed"),
            }

        label_names = set()
        labels = {}
        label_metadata = {}
        for record in labels_spec:
            if not isinstance(record, dict) or not all(
                isinstance(record.get(key), str) for key in ("name", "role", "array", "family")
            ):
                raise ValueError("label records require name, role, array, and family strings")
            name = record["name"]
            if name in label_names:
                raise ValueError(f"duplicate label name {name!r}")
            label_names.add(name)
            values = array(bundle, record["array"], name, n)
            if values.ndim != 1 or len(np.unique(values.astype(str))) < 2:
                raise ValueError(f"label {name!r} must be a nonconstant vector")
            labels[name] = values
            label_metadata[name] = {
                "role": record["role"],
                "family": record["family"],
                "matched_to": record.get("matched_to"),
                "construction": record.get("construction"),
                "seed": record.get("seed"),
                "prevalence": category_counts(values),
            }

        validate_controls(spec, representations_spec, labels_spec, bundle)
        prospective_errors, power_plan_metadata = load_prospective_power_plan(
            spec,
            spec_path,
            feature_names,
            label_names,
        )
        covariate_spec = spec.get("covariates", {})
        if not isinstance(covariate_spec, dict):
            raise ValueError("covariates must be an object")
        covariates, covariate_names = low_level_design(
            position=position,
            kmer=kmer,
            input_norm=input_norm,
            protein_length=protein_length,
            sequence_source=sequence_source,
            position_degree=int(covariate_spec.get("position_degree", 3)),
            kmer_hash_buckets=int(covariate_spec.get("kmer_hash_buckets", 64)),
        )
        test = spec.get("test", {})
        if not isinstance(test, dict):
            raise ValueError("test must be an object")
        parameters = {
            "n_folds": int(test.get("n_folds", 5)),
            "n_permutations": int(test.get("n_permutations", 1_000)),
            "n_bootstrap": int(test.get("n_bootstrap", 1_000)),
            "ridge_alpha": float(test.get("ridge_alpha", 1.0)),
            "seed": int(test.get("seed", 0)),
            "fdr_alpha": float(test.get("fdr_alpha", 0.05)),
            "power": float(test.get("power", 0.8)),
            "require_equal_dimensions": True,
        }
        results = run_conditional_semantics(
            representations,
            labels,
            covariates,
            proteins,
            families,
            feature_names=feature_names,
            prospective_standard_errors_delta_mse=prospective_errors,
            **parameters,
        )

        effect_rows = []
        for result in results:
            row = asdict(result)
            row["representation_role"] = representation_roles[result.representation]
            row["label_role"] = label_metadata[result.label]["role"]
            row["label_family"] = label_metadata[result.label]["family"]
            row["bootstrap_delta_mse_ci95"] = json.dumps(row["bootstrap_delta_mse_ci95"])
            row["bootstrap_delta_r2_ci95"] = json.dumps(row["bootstrap_delta_r2_ci95"])
            effect_rows.append(row)
        effect_fields = list(effect_rows[0])
        write_tsv(out_dir / "conditional_effects.tsv", effect_rows, effect_fields)

        protein_folds = blocked_fold_ids(proteins, n_folds=parameters["n_folds"], seed=parameters["seed"])
        family_folds = blocked_fold_ids(
            families, n_folds=parameters["n_folds"], seed=parameters["seed"] + 1
        )
        fold_rows = [
            {
                "row_index": index,
                "protein_id": str(proteins[index]),
                "family_id": str(families[index]),
                "protein_fold": int(protein_folds[index]),
                "family_fold": int(family_folds[index]),
            }
            for index in range(n)
        ]
        write_tsv(
            out_dir / "fold_assignments.tsv",
            fold_rows,
            ["row_index", "protein_id", "family_id", "protein_fold", "family_fold"],
        )
        prevalence_rows = []
        for name, metadata in label_metadata.items():
            for value in metadata["prevalence"]:
                prevalence_rows.append(
                    {
                        "label": name,
                        "role": metadata["role"],
                        "family": metadata["family"],
                        "matched_to": metadata["matched_to"],
                        **value,
                    }
                )
        write_tsv(
            out_dir / "label_prevalence.tsv",
            prevalence_rows,
            ["label", "role", "family", "matched_to", "category", "count", "fraction"],
        )

    significant = [result for result in results if result.qvalue <= parameters["fdr_alpha"]]
    summary = {
        "schema_version": 1,
        "confirmatory": bool(spec.get("confirmatory", True)),
        "gate_status": "not_adjudicated; prespecified biological effect/bound criterion required",
        "n_observations": n,
        "n_proteins": len(set(proteins.astype(str))),
        "n_families": len(set(families.astype(str))),
        "representations": representation_metadata,
        "labels": label_metadata,
        "covariate_names": covariate_names,
        "test": parameters,
        "prospective_power_plan": power_plan_metadata,
        "n_hypotheses": len(results),
        "n_qvalue_le_alpha": len(significant),
        "fold_hashes": {
            "protein": fold_assignment_hash(proteins, protein_folds),
            "family": fold_assignment_hash(families, family_folds),
        },
        "claim_boundary": (
            "A small q-value indicates held-out incremental prediction under this conditional "
            "model; it does not by itself establish a biological primitive or causal mechanism. "
            "Prospective MDEs require a hash-bound independent pilot plan; bootstrap-based "
            "detectability values estimated on these observations are retrospective. Degenerate "
            "within-protein randomizations are explicitly marked in the result table."
        ),
    }
    write_json(out_dir / "summary.json", summary)
    output_paths = sorted(path for path in out_dir.iterdir() if path.is_file())
    run_manifest = {
        "schema_version": 1,
        "command": [sys.executable, *sys.argv],
        "run_spec": str(spec_path),
        "run_spec_sha256": sha256_file(spec_path),
        "script_sha256": sha256_file(Path(__file__)),
        "data": {"path": str(bundle_path), "sha256": bundle_hash, "bytes": bundle_path.stat().st_size},
        "prospective_power_plan": power_plan_metadata,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in output_paths
        ],
    }
    write_json(out_dir / "run_manifest.json", run_manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run(args.spec.resolve(), args.out_dir.resolve())


if __name__ == "__main__":
    main()
