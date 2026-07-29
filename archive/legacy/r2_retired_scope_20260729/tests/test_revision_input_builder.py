from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from src.revision.input_builder import (
    ACTIVATION_FINITE_CHECK,
    _eligible_checkpoint,
    _validate_models,
    build_inputs,
    fit_dense_directions,
    residue_token_indices,
    verify_model_artifacts,
)
from src.revision.io import sha256_file
from src.training.clt_trainer import CLTForTraining


R2 = Path(__file__).resolve().parents[1]


class TinyTokenizer:
    def __call__(
        self,
        text,
        *,
        return_tensors,
        add_special_tokens,
        truncation,
        return_attention_mask,
    ):
        assert return_tensors == "pt"
        assert not truncation
        ids = [1000 + ord(character) for character in text]
        if add_special_tokens:
            ids = [1, *ids, 2]
        values = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": values, "attention_mask": torch.ones_like(values)}

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens
        token = ids[0]
        return "" if token in {1, 2} else chr(token - 1000)


class TinyProteinModel:
    n_layers = 2
    d_model = 3

    def __init__(self, *, dtype: torch.dtype = torch.float32, nonfinite: bool = False):
        self.tokenizer = TinyTokenizer()
        self.model = torch.nn.Linear(1, 1, bias=False, dtype=dtype)
        self.nonfinite = nonfinite

    def get_activations(self, input_ids, attention_mask):
        assert input_ids.shape == attention_mask.shape
        values = input_ids.float() - 1000
        base = torch.stack(
            (values / 100.0, values.square() / 10_000.0, torch.sin(values)), dim=-1
        )
        if self.nonfinite:
            base[0, 0, 0] = torch.nan
        return SimpleNamespace(
            clt_input=[base, base + 0.1],
            mlp_out=[base * 0.2, base * 0.3],
        )


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _model_root(path: Path) -> dict[str, str]:
    path.mkdir()
    (path / "config.json").write_text('{"model_type":"fixture"}\n', encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"fixture weights")
    (path / "tokenizer.json").write_text('{"fixture":true}\n', encoding="utf-8")
    config_hash = sha256_file(path / "config.json")
    expected = {
        "model_config_sha256": config_hash,
        "model_weights_sha256": "0" * 64,
        "tokenizer_sha256": "0" * 64,
    }
    observed = verify_model_artifacts(
        path,
        {
            "model_config_sha256": config_hash,
            "model_weights_sha256": hashlib.sha256(
                json.dumps(
                    [
                        {
                            "path": "model.safetensors",
                            "sha256": sha256_file(path / "model.safetensors"),
                            "size_bytes": (path / "model.safetensors").stat().st_size,
                        }
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "tokenizer_sha256": hashlib.sha256(
                json.dumps(
                    [
                        {
                            "path": "tokenizer.json",
                            "sha256": sha256_file(path / "tokenizer.json"),
                            "size_bytes": (path / "tokenizer.json").stat().st_size,
                        }
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        },
    )
    assert observed != expected
    return observed


def _checkpoint(path: Path, model_name: str, seed: int) -> str:
    path.mkdir(parents=True)
    torch.manual_seed(seed)
    clt = CLTForTraining(n_layers=2, d_model=3, d_clt=4, k=2, window=2)
    clt.global_step.fill_(10)
    torch.save(clt.state_dict(), path / "clt.pt")
    config = {
        "model": {"name": model_name},
        "clt": {"d_clt": 4, "k": 2, "window": 2},
    }
    (path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    files = {
        name: {
            "bytes": (path / name).stat().st_size,
            "sha256": sha256_file(path / name),
        }
        for name in ("clt.pt", "config.yaml")
    }
    _write_json(
        path / "checkpoint_manifest.json",
        {
            "schema_version": 2,
            "complete": True,
            "step": 10,
            "kind": "analysis_model_only",
            "world_size": 1,
            "trainer_source_sha256": "a" * 64,
            "files": files,
        },
    )
    return sha256_file(path / "checkpoint_manifest.json")


def _cohort(path: Path, split: str, prefix: str, sequences: list[str]) -> list[dict]:
    rows = []
    for index, sequence in enumerate(sequences):
        rows.append(
            {
                "id": f"{prefix}{index}",
                "source": f"source-{index % 2}",
                "sequence": sequence,
                "split": split,
                "family": f"family-{index}",
                "sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            }
        )
    _write_jsonl(path, rows)
    return rows


def _fixture_spec(tmp_path: Path) -> Path:
    discovery_path = tmp_path / "discovery.jsonl"
    heldout_path = tmp_path / "heldout.jsonl"
    _cohort(discovery_path, "validation", "d", ["MKWV", "ACDF", "GHIK", "LNPQ"])
    heldout = _cohort(heldout_path, "test", "h", ["RSTV", "WYAC", "DEFG", "HIKL"])
    annotation_path = tmp_path / "annotations.jsonl"
    annotation_rows = []
    for record in heldout:
        for position in range(len(record["sequence"])):
            annotation_rows.append(
                {
                    "sequence_sha256": record["sha256"],
                    "position": position,
                    "labels": {"domain": position % 2},
                }
            )
    _write_jsonl(annotation_path, annotation_rows)

    models = []
    for seed, name in enumerate(("alpha", "beta", "gamma"), 1):
        model_root = tmp_path / f"model-{name}"
        artifacts = _model_root(model_root)
        checkpoint = tmp_path / f"checkpoint-{name}" / "step_10"
        checkpoint_hash = _checkpoint(checkpoint, name, seed)
        models.append(
            {
                "name": name,
                "model_root": str(model_root),
                "model_artifacts": artifacts,
                "nonconfirmatory_fixture_checkpoint": {
                    "directory": str(checkpoint),
                    "manifest_sha256": checkpoint_hash,
                    "run_seed": seed,
                },
                "layers": [0, 1],
                "input_format": "sequence",
                "model_inference_dtype": "float32",
                "model_inference_dtype_verification": (
                    "all_floating_model_parameters_exactly_declared_before_first_activation"
                ),
                "activation_finiteness_check": ACTIVATION_FINITE_CHECK,
            }
        )
    spec = {
        "schema_version": 3,
        "confirmatory": False,
        "model_seed": 0,
        "device": "cpu",
        "max_model_tokens": 32,
        "cohorts": [
            {
                "role": "discovery",
                "cohort_id": "fixture-discovery",
                "path": str(discovery_path),
                "sha256": sha256_file(discovery_path),
                "split": "validation",
            },
            {
                "role": "heldout",
                "cohort_id": "fixture-heldout",
                "path": str(heldout_path),
                "sha256": sha256_file(heldout_path),
                "split": "test",
            },
        ],
        "models": models,
        "semantics": [
            {
                "name": "alpha-layer0",
                "model": "alpha",
                "run_seed": 1,
                "layer": 0,
                "features": [0, 1],
                "fit_role": "discovery",
                "evaluation_role": "heldout",
                "dense_fit_residue_budget": 12,
                "dense_seed": 17,
                "dense_oversample": 1,
                "random_seed": 29,
                "annotation": {
                    "path": str(annotation_path),
                    "sha256": sha256_file(annotation_path),
                },
                "labels": [
                    {
                        "name": "domain",
                        "family": "domain",
                        "annotation_key": "domain",
                        "construction": "fixture binary residue label",
                        "negative_name": "domain-negative",
                        "negative_seed": 43,
                    }
                ],
                "confirmatory": False,
                "covariates": {"position_degree": 2, "kmer_hash_buckets": 8},
                "test": {
                    "n_folds": 2,
                    "n_permutations": 3,
                    "n_bootstrap": 5,
                    "ridge_alpha": 1.0,
                    "seed": 5,
                    "fdr_alpha": 0.05,
                    "power": 0.8,
                },
            }
        ],
    }
    spec_path = tmp_path / "builder_spec.json"
    _write_json(spec_path, spec)
    return spec_path


def _attach_power_plan(spec_path: Path) -> None:
    spec = json.loads(spec_path.read_text())
    analysis = spec["semantics"][0]
    feature_names = {
        "sparse_aligned": ["alpha:S1:L0:F0", "alpha:S1:L0:F1"],
        "dense_matched": ["alpha:S1:L0:PC0", "alpha:S1:L0:PC1"],
        "randomized_dictionary": ["alpha:S1:L0:R0", "alpha:S1:L0:R1"],
    }
    rows = [
        {
            "representation": representation,
            "feature": feature,
            "label": label,
            "blocking": blocking,
            "standard_error_delta_mse": 0.1,
        }
        for representation, names in feature_names.items()
        for feature in names
        for label in ("domain", "domain-negative")
        for blocking in ("protein", "family")
    ]
    plan_path = spec_path.parent / "power_plan.json"
    _write_json(
        plan_path,
        {
            "schema_version": 1,
            "independent_source": {
                "description": "independent fixture pilot",
                "standard_error_method": "fixture cluster bootstrap",
                "run_manifest_sha256": "a" * 64,
                "cohort_sha256": "b" * 64,
                "independent_of_confirmatory_data": True,
            },
            "standard_errors_delta_mse": rows,
        },
    )
    analysis["confirmatory"] = True
    analysis["test"]["n_permutations"] = 1000
    analysis["test"]["n_bootstrap"] = 1000
    analysis["prospective_power_plan"] = {
        "path": str(plan_path),
        "sha256": sha256_file(plan_path),
    }
    _write_json(spec_path, spec)


def test_strict_residue_mapping_and_randomized_pca_are_deterministic():
    tokenizer = TinyTokenizer()
    encoded = tokenizer(
        "XXMKWVYY",
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=True,
    )
    np.testing.assert_array_equal(
        residue_token_indices(tokenizer, encoded["input_ids"], "MKWV"),
        np.array([3, 4, 5, 6]),
    )
    with pytest.raises(ValueError, match="exactly once"):
        residue_token_indices(tokenizer, encoded["input_ids"], "X")

    sample = np.arange(60, dtype=float).reshape(20, 3)
    sample[:, 2] += np.sin(np.arange(20))
    first = fit_dense_directions(sample, 2, seed=7, oversample=1)
    second = fit_dense_directions(sample, 2, seed=7, oversample=1)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_end_to_end_builder_outputs_feed_existing_semantics_runner(tmp_path):
    spec_path = _fixture_spec(tmp_path)
    output = tmp_path / "built"
    manifest = build_inputs(
        spec_path,
        output,
        model_loader=lambda model, device: TinyProteinModel(),
    )
    assert manifest == output / "run_manifest.json"
    run = json.loads(manifest.read_text())
    assert run["status"] == "nonconfirmatory_fixture_inputs_only"
    assert run["confirmatory"] is False
    assert run["model_seed"] == 0
    assert run["schema_version"] == "r2_p0_3_p0_4_input_build_v3"

    for role in ("discovery", "heldout"):
        atlas_manifest = json.loads((output / "atlas" / f"{role}_manifest.json").read_text())
        assert len(atlas_manifest["matrices"]) == 6
        assert atlas_manifest["status"] == "nonconfirmatory_fixture_inputs_only"
        assert atlas_manifest["model_seed"] == 0
        for record in atlas_manifest["matrices"]:
            assert type(record["dictionary_seed"]) is int
            assert record["dictionary_provenance"]["confirmatory"] is False
            assert record["dictionary_provenance"]["model_inference_dtype"] == "float32"
            assert record["dictionary_provenance"]["observed_model_parameter_dtypes"] == [
                "float32"
            ]
            assert record["dictionary_provenance"]["model_inference_dtype_verified"] is True
            assert (
                record["dictionary_provenance"]["activation_finiteness_check"]
                == ACTIVATION_FINITE_CHECK
            )
            assert record["dictionary_provenance"]["activation_finiteness_verified"] is True
            matrix_path = output / "atlas" / record["path"]
            assert sha256_file(matrix_path) == record["sha256"]
            matrix = np.load(matrix_path, allow_pickle=False)
            assert matrix.shape == (4, 4)
            assert np.isfinite(matrix).all()

    semantic_dir = output / "semantics" / "alpha-layer0"
    input_manifest = json.loads((semantic_dir / "input_manifest.json").read_text())
    with np.load(semantic_dir / "continuous_activations.npz", allow_pickle=False) as bundle:
        assert bundle["sparse_activations"].shape == (16, 2)
        assert bundle["dense_matched_directions"].shape == (16, 2)
        assert bundle["randomized_dictionary_activations"].shape == (16, 2)
        for protein in np.unique(bundle["protein_id"]):
            mask = bundle["protein_id"] == protein
            assert bundle["biological_label_0"][mask].sum() == bundle["negative_label_0"][mask].sum()
    assert input_manifest["n_residues"] == 16
    assert input_manifest["run_seed"] == 1
    assert input_manifest["model_checkpoint"]["confirmatory"] is False
    assert input_manifest["schema_version"] == "r2_p0_4_continuous_input_v3"
    conditional_spec = json.loads(
        (semantic_dir / "conditional_semantics_spec.json").read_text(encoding="utf-8")
    )
    assert conditional_spec["schema_version"] == "r2_p0_4_conditional_semantics_spec_v3"
    assert conditional_spec["input_provenance"]["activation_finiteness_verified"] is True

    script = R2 / "scripts/53_run_conditional_semantics.py"
    module_spec = importlib.util.spec_from_file_location("conditional_runner_fixture", script)
    runner = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(runner)
    runner.run(
        semantic_dir / "conditional_semantics_spec.json",
        tmp_path / "semantic-results",
    )
    summary = json.loads((tmp_path / "semantic-results" / "summary.json").read_text())
    assert summary["n_observations"] == 16
    assert summary["n_hypotheses"] == 24


def test_builder_fails_closed_on_cohort_or_model_tampering(tmp_path):
    spec_path = _fixture_spec(tmp_path)
    spec = json.loads(spec_path.read_text())
    checkpoint = (
        Path(spec["models"][0]["nonconfirmatory_fixture_checkpoint"]["directory"])
        / "clt.pt"
    )
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint.write_bytes(checkpoint_bytes + b"tampered")
    with pytest.raises(ValueError, match="checkpoint size mismatch"):
        build_inputs(
            spec_path,
            tmp_path / "bad-checkpoint",
            model_loader=lambda model, device: TinyProteinModel(),
        )
    checkpoint.write_bytes(checkpoint_bytes)

    heldout = Path(spec["cohorts"][1]["path"])
    heldout.write_text(heldout.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cohort SHA-256 mismatch"):
        build_inputs(
            spec_path,
            tmp_path / "bad-cohort",
            model_loader=lambda model, device: TinyProteinModel(),
        )

    model = Path(spec["models"][0]["model_root"])
    (model / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="model_weights_sha256"):
        verify_model_artifacts(model, spec["models"][0]["model_artifacts"])


def test_builder_rejects_dtype_mismatch_and_nonfinite_capture(tmp_path):
    spec_path = _fixture_spec(tmp_path)
    with pytest.raises(ValueError, match="parameter dtype disagrees"):
        build_inputs(
            spec_path,
            tmp_path / "wrong-dtype",
            model_loader=lambda model, device: TinyProteinModel(dtype=torch.bfloat16),
        )
    with pytest.raises(FloatingPointError, match="non-finite frozen-model"):
        build_inputs(
            spec_path,
            tmp_path / "nonfinite",
            model_loader=lambda model, device: TinyProteinModel(nonfinite=True),
        )


def test_confirmatory_spec_freezes_bfloat16_inference(tmp_path):
    spec_path = _fixture_spec(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    for model in spec["models"]:
        model.pop("nonconfirmatory_fixture_checkpoint")
        model["dictionaries"] = []
        model["source_manifest_sha256_by_split"] = {
            "train": "a" * 64,
            "validation": "b" * 64,
            "test": "c" * 64,
        }
    with pytest.raises(ValueError, match="confirmatory model inference must use bfloat16"):
        _validate_models(
            spec,
            spec_path.parent,
            confirmatory=True,
            eligibility_receipt={"path": tmp_path / "receipt", "sha256": "d" * 64},
        )

    example = json.loads(
        (R2 / "configs/npj_revision_input_builder_spec.example.json").read_text(
            encoding="utf-8"
        )
    )
    assert example["schema_version"] == 3
    assert {model["model_inference_dtype"] for model in example["models"]} == {
        "bfloat16"
    }
    assert {
        model["model_inference_dtype_verification"] for model in example["models"]
    } == {"all_floating_model_parameters_exactly_declared_before_first_activation"}
    assert {model["activation_finiteness_check"] for model in example["models"]} == {
        ACTIVATION_FINITE_CHECK
    }


def test_nonconfirmatory_fixture_cannot_emit_confirmatory_semantics(tmp_path):
    spec_path = _fixture_spec(tmp_path)
    spec = json.loads(spec_path.read_text())
    spec["semantics"][0]["confirmatory"] = True
    spec["semantics"][0]["test"]["n_permutations"] = 1000
    spec["semantics"][0]["test"]["n_bootstrap"] = 1000
    _write_json(spec_path, spec)
    with pytest.raises(ValueError, match="confirmation status must match"):
        build_inputs(
            spec_path,
            tmp_path / "missing-plan",
            model_loader=lambda model, device: TinyProteinModel(),
        )

    _attach_power_plan(spec_path)
    with pytest.raises(ValueError, match="confirmation status must match"):
        build_inputs(
            spec_path,
            tmp_path / "with-plan",
            model_loader=lambda model, device: TinyProteinModel(),
        )


def test_confirmatory_builder_rejects_fixture_checkpoint_schema_and_loader_bypass(tmp_path):
    spec_path = _fixture_spec(tmp_path)
    spec = json.loads(spec_path.read_text())
    spec["confirmatory"] = True
    spec["p0_2_eligibility_receipt"] = {
        "path": str(tmp_path / "receipt.json"),
        "sha256": "a" * 64,
    }
    _write_json(tmp_path / "receipt.json", {"fixture": True})
    spec["p0_2_eligibility_receipt"]["sha256"] = sha256_file(tmp_path / "receipt.json")
    for analysis in spec["semantics"]:
        analysis["confirmatory"] = True
    _write_json(spec_path, spec)
    with pytest.raises(ValueError, match="custom model loaders are forbidden"):
        build_inputs(
            spec_path,
            tmp_path / "loader-bypass",
            model_loader=lambda model, device: TinyProteinModel(),
        )
    with pytest.raises(ValueError, match="model 0 fields differ"):
        build_inputs(spec_path, tmp_path / "legacy-schema")


def test_eligible_loader_receives_exact_all_seed_contract(monkeypatch, tmp_path):
    from src.revision import dictionary_gate

    dictionaries = [
        {
            "confirmatory": True,
            "run_seed": seed,
            "checkpoint": tmp_path / f"seed-{seed}-best.pt",
            "checkpoint_sha256": f"{seed:064x}",
            "run_manifest_sha256": f"{seed + 100:064x}",
        }
        for seed in (17, 29, 43)
    ]
    source_hashes = {
        "train": "a" * 64,
        "validation": "b" * 64,
        "test": "c" * 64,
    }
    model = {
        "confirmatory": True,
        "name": "protgpt2",
        "layers": [0, 1],
        "dictionaries": dictionaries,
        "eligibility_receipt_path": tmp_path / "receipt.json",
        "eligibility_receipt_sha256": "d" * 64,
        "source_manifest_sha256_by_split": source_hashes,
    }
    captured = {}

    def fake_loader(receipt_path, receipt_sha256, **kwargs):
        captured.update(
            receipt_path=receipt_path,
            receipt_sha256=receipt_sha256,
            **kwargs,
        )
        clt = CLTForTraining(n_layers=2, d_model=3, d_clt=4, k=2, window=2)
        seed = kwargs["run_seed"]
        return clt, {
            "schema_version": "r2_p0_2_eligible_topk_load_v1",
            "model_name": "protgpt2",
            "method": "topk_clt",
            "run_seed": seed,
            "eligibility_receipt_sha256": "d" * 64,
            "run_manifest_sha256": f"{seed + 100:064x}",
            "checkpoint_sha256": f"{seed:064x}",
            "source_manifest_sha256_by_split": source_hashes,
            "eligible_downstream_layers": [0, 1],
            "checkpoint_step": 200_000,
            "candidate_id": f"seed-{seed}",
            "geometry": {
                "n_layers": 2,
                "d_model": 3,
                "d_clt": 4,
                "k": 2,
                "window": 2,
            },
        }

    monkeypatch.setattr(dictionary_gate, "load_eligible_topk_clt", fake_loader)
    _, provenance = _eligible_checkpoint(model, dictionaries[0], "cpu")
    assert provenance["confirmatory"] is True
    assert captured["requested_layers"] == [0, 1]
    assert captured["expected_run_manifest_sha256_by_seed"] == {
        seed: f"{seed + 100:064x}" for seed in (17, 29, 43)
    }
    assert captured["expected_checkpoint_sha256_by_seed"] == {
        seed: f"{seed:064x}" for seed in (17, 29, 43)
    }
    assert captured["expected_source_manifest_sha256_by_split"] == source_hashes

    def mismatched_loader(*args, **kwargs):
        clt, provenance = fake_loader(*args, **kwargs)
        provenance["run_seed"] = 29
        return clt, provenance

    monkeypatch.setattr(dictionary_gate, "load_eligible_topk_clt", mismatched_loader)
    with pytest.raises(ValueError, match="mismatched P0-2 provenance"):
        _eligible_checkpoint(model, dictionaries[0], "cpu")
