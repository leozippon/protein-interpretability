from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

from src.revision.io import sha256_file
from src.revision.semantic_adjudication import EFFECT_FIELDS, adjudicate


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _hypotheses() -> list[dict[str, str]]:
    return [
        {
            "representation": "sparse",
            "feature": "f0",
            "label": label,
            "blocking": blocking,
        }
        for label in ("biological", "negative")
        for blocking in ("protein", "family")
    ]


def _rehash_manifest(run_dir: Path) -> str:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"] = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "run_manifest.json"
    ]
    _write_json(manifest_path, manifest)
    return sha256_file(manifest_path)


def _make_run(
    root: Path,
    model: str,
    layer: int,
    *,
    confirmatory: bool = True,
    degenerate: bool = False,
    with_power: bool = True,
) -> tuple[dict, dict[str, Path]]:
    run_dir = root / f"run_{model}_{layer}"
    run_dir.mkdir()
    proteins = np.repeat(np.array(["p0", "p1", "p2", "p3"]), 2)
    if degenerate:
        biological = np.repeat(np.array([0, 1, 0, 1]), 2)
        permutable_fraction = 0.0
    else:
        biological = np.tile(np.array([0, 1]), 4)
        permutable_fraction = 1.0
    negative = np.tile(np.array([1, 0]), 4)
    data_path = root / f"data_{model}_{layer}.npz"
    np.savez(
        data_path,
        protein_id=proteins,
        biological_label=biological,
        negative_label=negative,
    )
    data_hash = sha256_file(data_path)

    power_path = root / f"power_{model}_{layer}.json"
    power_hash = None
    standard_error = 0.1
    if with_power:
        power_plan = {
            "schema_version": 1,
            "independent_source": {
                "description": "independent pilot cohort",
                "standard_error_method": "protein-cluster bootstrap",
                "run_manifest_sha256": "1" * 64,
                "cohort_sha256": "2" * 64,
                "independent_of_confirmatory_data": True,
            },
            "standard_errors_delta_mse": [
                {**hypothesis, "standard_error_delta_mse": standard_error}
                for hypothesis in _hypotheses()
            ],
        }
        _write_json(power_path, power_plan)
        power_hash = sha256_file(power_path)

    run_spec = {
        "confirmatory": confirmatory,
        "data": {"path": str(data_path.resolve()), "sha256": data_hash},
        "arrays": {"protein_id": "protein_id"},
        "representations": [
            {
                "name": "sparse",
                "role": "sparse",
                "array": "unused",
                "feature_names": ["f0"],
            }
        ],
        "labels": [
            {
                "name": "biological",
                "role": "biological",
                "family": "fixture",
                "array": "biological_label",
            },
            {
                "name": "negative",
                "role": "negative",
                "family": "fixture",
                "array": "negative_label",
            },
        ],
        "test": {"fdr_alpha": 0.05, "power": 0.8},
    }
    if with_power:
        run_spec["prospective_power_plan"] = {
            "path": str(power_path.resolve()),
            "sha256": power_hash,
        }
    run_spec_path = root / f"spec_{model}_{layer}.json"
    _write_json(run_spec_path, run_spec)
    run_spec_hash = sha256_file(run_spec_path)

    local_multiplier = NormalDist().inv_cdf(1 - 0.05 / 4) + NormalDist().inv_cdf(0.8)
    effect_rows = []
    for index, hypothesis in enumerate(_hypotheses()):
        is_biological = hypothesis["label"] == "biological"
        fraction = permutable_fraction if is_biological else 1.0
        pvalue = 0.001 * (index + 1) if is_biological else 0.8 + 0.02 * index
        row = {
            **hypothesis,
            "n_observations": 8,
            "n_proteins": 4,
            "n_blocks": 4 if hypothesis["blocking"] == "protein" else 2,
            "baseline_mse": 1.0,
            "full_mse": 0.8,
            "delta_mse": 0.2,
            "delta_r2": 0.2,
            "permutation_pvalue": pvalue,
            "qvalue": 0.0,
            "bootstrap_delta_mse_ci95": json.dumps([0.1, 0.3]),
            "bootstrap_delta_r2_ci95": json.dumps([0.1, 0.3]),
            "bootstrap_standard_error_delta_mse": 0.05,
            "retrospective_bootstrap_detectable_delta_mse": 0.2,
            "prospective_minimum_detectable_delta_mse": (
                standard_error * local_multiplier if with_power else None
            ),
            "permutable_row_fraction": fraction,
            "permutation_degenerate": is_biological and degenerate,
            "fold_hash": "a" * 64 if hypothesis["blocking"] == "protein" else "b" * 64,
            "representation_role": "sparse",
            "label_role": "biological" if is_biological else "negative",
            "label_family": "fixture",
        }
        effect_rows.append(row)
    effects_path = run_dir / "conditional_effects.tsv"
    with effects_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EFFECT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(effect_rows)
    (run_dir / "fold_assignments.tsv").write_text("row_index\n0\n", encoding="utf-8")
    (run_dir / "label_prevalence.tsv").write_text("label\nbiological\n", encoding="utf-8")
    power_metadata = (
        None
        if not with_power
        else {
            "path": str(power_path.resolve()),
            "sha256": power_hash,
            "independent_source": json.loads(power_path.read_text(encoding="utf-8"))[
                "independent_source"
            ],
            "n_standard_errors": 4,
        }
    )
    summary = {
        "schema_version": 1,
        "confirmatory": confirmatory,
        "n_hypotheses": 4,
        "prospective_power_plan": power_metadata,
    }
    _write_json(run_dir / "summary.json", summary)
    manifest = {
        "schema_version": 1,
        "run_spec": str(run_spec_path.resolve()),
        "run_spec_sha256": run_spec_hash,
        "data": {"path": str(data_path.resolve()), "sha256": data_hash, "bytes": data_path.stat().st_size},
        "prospective_power_plan": power_metadata,
        "outputs": [],
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    manifest_hash = _rehash_manifest(run_dir)
    descriptor = {
        "model": model,
        "layer": layer,
        "path": str(run_dir.resolve()),
        "run_manifest_sha256": manifest_hash,
        "run_spec_sha256": run_spec_hash,
        "data_sha256": data_hash,
        "prospective_power_plan_sha256": power_hash,
        "hypotheses": _hypotheses(),
    }
    return descriptor, {
        "run": run_dir,
        "data": data_path,
        "spec": run_spec_path,
        "effects": effects_path,
    }


def _collector_spec(root: Path, runs: list[dict], adjudication: dict | None = None) -> Path:
    path = root / f"collector_{len(list(root.glob('collector_*.json')))}.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "confirmatory": True,
            "fdr_alpha": 0.05,
            "power": 0.8,
            "adjudication": adjudication or {"mode": "multiplicity_only"},
            "runs": runs,
        },
    )
    return path


def _run(spec: Path, output: Path) -> Path:
    return adjudicate(spec, output, expected_spec_sha256=sha256_file(spec))


def test_joint_collector_recomputes_global_bh_and_emits_hashed_receipt(tmp_path: Path):
    first, _ = _make_run(tmp_path, "model_a", 1)
    second, _ = _make_run(tmp_path, "model_b", 2)
    spec = _collector_spec(tmp_path, [first, second])
    receipt_path = _run(spec, tmp_path / "joint")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "joint" / "summary.json").read_text(encoding="utf-8"))
    with (tmp_path / "joint" / "joint_conditional_effects.tsv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert receipt["artifact_validation_status"] == "verified_complete"
    assert receipt["scientific_gate_status"] == "not_scientifically_adjudicated"
    assert summary["artifact_validation_status"] == "verified_complete"
    assert summary["scientific_gate_status"] == "not_scientifically_adjudicated"
    assert summary["n_runs"] == 2
    assert summary["n_hypotheses"] == 8
    assert summary["per_run_qvalues_used"] is False
    assert {row["reported_per_run_qvalue_ignored"] for row in rows} == {"0.0"}
    assert all(float(row["joint_bh_qvalue"]) > 0 for row in rows)
    assert all(row["joint_prospective_minimum_detectable_delta_mse"] for row in rows)
    for name, digest in receipt["artifacts"].items():
        assert sha256_file(tmp_path / "joint" / name) == digest


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_rejects_incomplete_or_duplicate_frozen_hypotheses(tmp_path: Path, mutation: str):
    descriptor, _ = _make_run(tmp_path, "model_a", 1)
    if mutation == "missing":
        descriptor["hypotheses"].pop()
    else:
        descriptor["hypotheses"].append(dict(descriptor["hypotheses"][0]))
    spec = _collector_spec(tmp_path, [descriptor])
    with pytest.raises(ValueError, match="hypotheses|hypothesis inventory"):
        _run(spec, tmp_path / "joint")


def test_rejects_extra_effect_hypothesis_even_when_manifest_is_rehashed(tmp_path: Path):
    descriptor, paths = _make_run(tmp_path, "model_a", 1)
    with paths["effects"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    extra = dict(rows[0])
    extra["feature"] = "undeclared_feature"
    rows.append(extra)
    with paths["effects"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EFFECT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    descriptor["run_manifest_sha256"] = _rehash_manifest(paths["run"])
    spec = _collector_spec(tmp_path, [descriptor])
    with pytest.raises(ValueError, match="unknown hypothesis|missing or extra"):
        _run(spec, tmp_path / "joint")


def test_rejects_output_tampering(tmp_path: Path):
    descriptor, paths = _make_run(tmp_path, "model_a", 1)
    paths["effects"].write_text(
        paths["effects"].read_text(encoding="utf-8") + "tamper\n", encoding="utf-8"
    )
    spec = _collector_spec(tmp_path, [descriptor])
    with pytest.raises(ValueError, match="hash mismatch"):
        _run(spec, tmp_path / "joint")


def test_rejects_nonconfirmatory_input(tmp_path: Path):
    descriptor, _ = _make_run(tmp_path, "model_a", 1, confirmatory=False)
    spec = _collector_spec(tmp_path, [descriptor])
    with pytest.raises(ValueError, match="non-confirmatory"):
        _run(spec, tmp_path / "joint")


def test_rejects_degenerate_biological_label_from_source_data(tmp_path: Path):
    descriptor, _ = _make_run(tmp_path, "model_a", 1, degenerate=True)
    spec = _collector_spec(tmp_path, [descriptor])
    with pytest.raises(ValueError, match="degenerate confirmatory biological"):
        _run(spec, tmp_path / "joint")


def test_powered_bound_requires_complete_independent_power_plans(tmp_path: Path):
    first, _ = _make_run(tmp_path, "model_a", 1)
    second, _ = _make_run(tmp_path, "model_b", 2, with_power=False)
    targets = [
        {
            "model": descriptor["model"],
            "layer": descriptor["layer"],
            "representation": "sparse",
            "feature": "f0",
            "label": "biological",
            "blocking": blocking,
        }
        for descriptor in (first, second)
        for blocking in ("protein", "family")
    ]
    spec = _collector_spec(
        tmp_path,
        [first, second],
        {
            "mode": "powered_bound",
            "targets": targets,
            "maximum_residual_delta_mse": 1.0,
        },
    )
    with pytest.raises(ValueError, match="complete independent power plans"):
        _run(spec, tmp_path / "joint")


@pytest.mark.parametrize("mode", ["association", "powered_bound"])
def test_replicated_scientific_rules_use_both_global_blockings(tmp_path: Path, mode: str):
    first, _ = _make_run(tmp_path, "model_a", 1)
    second, _ = _make_run(tmp_path, "model_b", 2)
    targets = [
        {
            "model": descriptor["model"],
            "layer": descriptor["layer"],
            "representation": "sparse",
            "feature": "f0",
            "label": "biological",
            "blocking": blocking,
        }
        for descriptor in (first, second)
        for blocking in ("protein", "family")
    ]
    threshold = (
        {"minimum_delta_mse": 0.05}
        if mode == "association"
        else {"maximum_residual_delta_mse": 1.0}
    )
    spec = _collector_spec(
        tmp_path,
        [first, second],
        {"mode": mode, "targets": targets, **threshold},
    )
    _run(spec, tmp_path / "joint")
    summary = json.loads((tmp_path / "joint" / "summary.json").read_text(encoding="utf-8"))
    assert summary["decision"]["mode"] == mode
    assert summary["decision"]["status"] == "passed"
    assert all(target["passed"] for target in summary["decision"]["targets"])


def test_failed_scientific_gate_is_not_reported_as_generic_completion(tmp_path: Path):
    first, _ = _make_run(tmp_path, "model_a", 1)
    second, _ = _make_run(tmp_path, "model_b", 2)
    targets = [
        {
            "model": descriptor["model"],
            "layer": descriptor["layer"],
            "representation": "sparse",
            "feature": "f0",
            "label": "biological",
            "blocking": blocking,
        }
        for descriptor in (first, second)
        for blocking in ("protein", "family")
    ]
    spec = _collector_spec(
        tmp_path,
        [first, second],
        {"mode": "association", "targets": targets, "minimum_delta_mse": 10.0},
    )
    receipt_path = _run(spec, tmp_path / "joint")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "joint" / "summary.json").read_text(encoding="utf-8"))
    assert receipt["artifact_validation_status"] == "verified_complete"
    assert receipt["scientific_gate_status"] == "failed"
    assert summary["scientific_gate_status"] == "failed"


def test_rejects_duplicate_model_layer_run(tmp_path: Path):
    descriptor, _ = _make_run(tmp_path, "model_a", 1)
    duplicate = dict(descriptor)
    duplicate["hypotheses"] = [dict(item) for item in descriptor["hypotheses"]]
    spec = _collector_spec(tmp_path, [descriptor, duplicate])
    with pytest.raises(ValueError, match="duplicate model/layer"):
        _run(spec, tmp_path / "joint")


def test_collector_spec_requires_external_hash_binding(tmp_path: Path):
    descriptor, _ = _make_run(tmp_path, "model_a", 1)
    spec = _collector_spec(tmp_path, [descriptor])
    with pytest.raises(ValueError, match="expected_spec_sha256"):
        adjudicate(spec, tmp_path / "joint", expected_spec_sha256="f" * 64)
