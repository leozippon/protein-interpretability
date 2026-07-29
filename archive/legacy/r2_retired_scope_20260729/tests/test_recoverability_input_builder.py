from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.revision.input_builder import verify_model_artifacts
from src.revision.dictionary_controls import WindowedTranscoder
from src.revision.io import sha256_file
from src.revision.nested_recoverability import run_nested_recoverability
from src.revision.recoverability_input_builder import (
    _canonical_digest,
    _validate_model,
    build_recoverability_inputs,
)
from src.revision.input_builder import ACTIVATION_FINITE_CHECK
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
        assert add_special_tokens and not truncation and return_attention_mask
        tokens = [1, *(1000 + ord(character) for character in text), 2]
        values = torch.tensor([tokens], dtype=torch.long)
        return {"input_ids": values, "attention_mask": torch.ones_like(values)}

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens
        return "" if ids[0] in {1, 2} else chr(ids[0] - 1000)


class TinyProteinModel:
    n_layers = 2
    d_model = 4

    def __init__(self, *, dtype: torch.dtype = torch.float32, nonfinite: bool = False):
        self.tokenizer = TinyTokenizer()
        self.model = torch.nn.Linear(1, 1, bias=False, dtype=dtype)
        self.nonfinite = nonfinite

    def get_activations(self, input_ids, attention_mask):
        assert input_ids.shape == attention_mask.shape
        values = input_ids.float() - 1000
        base = torch.stack(
            (
                values / 100.0,
                values.square() / 10_000.0,
                torch.sin(values),
                torch.cos(values),
            ),
            dim=-1,
        )
        clt_input = [base, base + 0.1]
        mlp_out = [base * 0.2, base * 0.3]
        if self.nonfinite:
            mlp_out[0][0, 0, 0] = torch.inf
        return SimpleNamespace(clt_input=clt_input, mlp_out=mlp_out)


def _fake_dictionary_loader(
    receipt_path,
    expected_receipt_sha256,
    *,
    model_name,
    run_seed,
    checkpoint_path,
    expected_run_manifest_sha256_by_seed,
    expected_checkpoint_sha256_by_seed,
    expected_source_manifest_sha256_by_split,
    requested_layers,
    map_location,
):
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    state = payload["model_state_dict"]
    clt = CLTForTraining(
        n_layers=2, d_model=4, d_clt=128, k=128, window=2
    ).to(map_location)
    with torch.no_grad():
        clt.W_enc.copy_(state["encoder_weight"])
        clt.b_enc.copy_(state["encoder_bias"])
        clt.b_dec.copy_(state["decoder_bias"])
        for layer in range(2):
            clt.W_dec[layer].copy_(state[f"decoder_weight.{layer}"])
        clt.global_step.fill_(payload["step"])
    return clt.eval(), {
        "model_name": model_name,
        "method": "topk_clt",
        "run_seed": run_seed,
        "eligibility_receipt_sha256": expected_receipt_sha256,
        "run_manifest_sha256": expected_run_manifest_sha256_by_seed[run_seed],
        "checkpoint_sha256": expected_checkpoint_sha256_by_seed[run_seed],
        "candidate_id": payload["candidate_id"],
        "checkpoint_step": payload["step"],
        "geometry": {
            "n_layers": 2,
            "d_model": 4,
            "d_clt": 128,
            "k": 128,
            "window": 2,
        },
        "eligible_downstream_layers": list(requested_layers),
        "source_manifest_sha256_by_split": dict(
            expected_source_manifest_sha256_by_split
        ),
    }


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _cohort(
    path: Path, role: str, split: str, sequences: list[str]
) -> tuple[dict, list[dict]]:
    rows = [
        {
            "id": f"{role}-{index}",
            "source": "cpu-fixture",
            "sequence": sequence,
            "split": split,
            "family": f"family-{role}-{index}",
            "sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        }
        for index, sequence in enumerate(sequences)
    ]
    _write_jsonl(path, rows)
    return (
        {
            "role": role,
            "cohort_id": f"fixture-{role}",
            "path": str(path),
            "sha256": sha256_file(path),
            "split": split,
        },
        rows,
    )


def _model_root(path: Path) -> dict[str, str]:
    path.mkdir()
    (path / "config.json").write_text('{"model_type":"fixture"}\n', encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"fixture weights")
    (path / "tokenizer.json").write_text('{"fixture":true}\n', encoding="utf-8")
    config_sha = sha256_file(path / "config.json")
    temporary = {
        "model_config_sha256": config_sha,
        "model_weights_sha256": "0" * 64,
        "tokenizer_sha256": "0" * 64,
    }
    weights = [
        {
            "path": "model.safetensors",
            "sha256": sha256_file(path / "model.safetensors"),
            "size_bytes": (path / "model.safetensors").stat().st_size,
        }
    ]
    support = [
        {
            "path": "tokenizer.json",
            "sha256": sha256_file(path / "tokenizer.json"),
            "size_bytes": (path / "tokenizer.json").stat().st_size,
        }
    ]
    temporary["model_weights_sha256"] = hashlib.sha256(
        json.dumps(weights, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    temporary["tokenizer_sha256"] = hashlib.sha256(
        json.dumps(support, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return verify_model_artifacts(path, temporary)


def _checkpoint(path: Path, seed: int) -> dict:
    path.parent.mkdir(parents=True)
    torch.manual_seed(seed)
    model = WindowedTranscoder(
        method="topk_clt",
        n_layers=2,
        input_dim=4,
        target_dim=4,
        width=128,
        window=2,
        topk_k=128,
    )
    torch.save(
        {
            "schema_version": "r2_dictionary_control_best_v1",
            "candidate_id": f"full_topk_clt_seed_{seed}_candidate_000_fixture",
            "step": 10,
            "validation_fvu_mean": 0.1,
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return {
        "seed": seed,
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "run_manifest_sha256": hashlib.sha256(f"run-{seed}".encode()).hexdigest(),
    }


def _fixture(tmp_path: Path) -> tuple[Path, dict]:
    prior_specs = []
    all_identity_rows = []
    prior_sequences = {
        "p0_2_train": ["AAAA", "AAAC"],
        "p0_2_validation": ["AAAD", "AAAE"],
        "p0_2_test": ["AAAF", "AAAG"],
    }
    split_by_role = {
        "p0_2_train": "train",
        "p0_2_validation": "validation",
        "p0_2_test": "test",
    }
    for role, sequences in prior_sequences.items():
        descriptor, rows = _cohort(
            tmp_path / f"{role}.jsonl", role, split_by_role[role], sequences
        )
        prior_specs.append(descriptor)
        all_identity_rows.extend(
            {
                "sequence_sha256": row["sha256"],
                "identity_group": f"prior-{role}-{index}",
                "cohort_role": role,
            }
            for index, row in enumerate(rows)
        )
    evaluation_spec, evaluation_rows = _cohort(
        tmp_path / "evaluation.jsonl",
        "p0_8_evaluation",
        "test",
        [
            "ACDE",
            "CDEF",
            "DEFG",
            "EFGH",
            "FGHI",
            "GHIK",
            "HIKL",
            "IKLM",
            "KLMN",
            "LMNP",
            "MNPQ",
            "NPQR",
            "PQRS",
            "QRST",
            "RSTV",
            "STVW",
        ],
    )
    annotation_rows = []
    for index, row in enumerate(evaluation_rows):
        group = f"evaluation-{index}"
        annotation_rows.append(
            {
                "row_id": f"row-{index}",
                "sequence_sha256": row["sha256"],
                "identity_group": group,
                "target": index % 2,
                "residue_positions": [0],
            }
        )
        all_identity_rows.append(
            {
                "sequence_sha256": row["sha256"],
                "identity_group": group,
                "cohort_role": "p0_8_evaluation",
            }
        )
    annotation_path = tmp_path / "annotations.jsonl"
    identity_path = tmp_path / "identity.jsonl"
    _write_jsonl(annotation_path, annotation_rows)
    _write_jsonl(identity_path, all_identity_rows)
    clusterer_path = tmp_path / "clusterer.bin"
    clusterer_path.write_bytes(b"fixture identity clusterer")
    clustering_receipt_path = tmp_path / "identity_clustering_receipt.json"
    _write_json(
        clustering_receipt_path,
        {
            "schema_version": "r2_p0_8_identity_clustering_receipt_v1",
            "status": "verified_complete",
            "assignment_sha256": sha256_file(identity_path),
            "input_cohort_sha256_by_role": {
                **{item["role"]: item["sha256"] for item in prior_specs},
                "p0_8_evaluation": evaluation_spec["sha256"],
            },
            "algorithm": {
                "name": "fixture_identity_clusterer",
                "version": "1.0",
                "sequence_identity_threshold": 0.3,
                "coverage_threshold": 0.8,
                "command": [str(clusterer_path), "--fixture"],
                "executable": {
                    "path": str(clusterer_path),
                    "sha256": sha256_file(clusterer_path),
                },
            },
        },
    )
    model_root = tmp_path / "model"
    model_artifacts = _model_root(model_root)
    dictionaries = [
        _checkpoint(tmp_path / f"checkpoint-{seed}" / "best.pt", seed)
        for seed in (17, 29, 43)
    ]
    source_hashes = {
        descriptor["role"].removeprefix("p0_2_"): descriptor["sha256"]
        for descriptor in prior_specs
    }
    runs = [
        {
            "run_seed": item["seed"],
            "run_manifest_sha256": item["run_manifest_sha256"],
            "result_sha256": hashlib.sha256(
                f"result-{item['seed']}".encode()
            ).hexdigest(),
            "checkpoint_sha256": item["checkpoint_sha256"],
            "quality_gate_pass": True,
            "failure_reasons": [],
        }
        for item in dictionaries
    ]
    p0_2_receipt = {
        "schema_version": "r2_p0_2_eligibility_receipt_v1",
        "status": "complete",
        "artifact_completeness": True,
        "required_models": ["protgpt2", "zymctrl", "progen2-medium"],
        "required_methods": [
            "topk_clt",
            "relu_l1_sae",
            "gated_sae",
            "dense_low_rank",
        ],
        "profile_sha256": "b" * 64,
        "protocol_sha256": "c" * 64,
        "mask_validation_receipt_sha256": "d" * 64,
        "frozen_gate": {"required_seeds": [17, 29, 43]},
        "model_method_adjudications": [
            {
                "model_name": "fixture-model",
                "method": "topk_clt",
                "status": "atlas_eligible",
                "atlas_eligible": True,
                "failure_reasons": [],
                "required_seeds": [17, 29, 43],
                "source_manifest_sha256_by_split": source_hashes,
                "cache_manifest_sha256": "e" * 64,
                "cache_content_sha256": "f" * 64,
                "eligible_downstream_layers": [0, 1],
                "runs": runs,
            }
        ],
    }
    p0_2_path = tmp_path / "p0_2_receipt.json"
    _write_json(p0_2_path, p0_2_receipt)
    row_order_sha = _canonical_digest(
        [
            {
                "row_id": row["row_id"],
                "sequence_sha256": row["sequence_sha256"],
                "identity_group": row["identity_group"],
                "residue_positions": row["residue_positions"],
            }
            for row in annotation_rows
        ]
    )
    intervention_rows = [
        {
            "row_id": row["row_id"],
            "sequence_sha256": row["sequence_sha256"],
            "intervention_effect_by_seed_layer": {
                str(seed): {
                    str(layer): 0.1
                    + index * 0.01
                    + seed * 0.0001
                    + layer * 0.001
                    for layer in (0, 1)
                }
                for seed in (17, 29, 43)
            },
        }
        for index, row in enumerate(annotation_rows)
    ]
    intervention_path = tmp_path / "intervention.jsonl"
    _write_jsonl(intervention_path, intervention_rows)
    effect_definition = "paired endpoint displacement in CPU fixture units"
    producer_path = tmp_path / "intervention_producer.py"
    producer_path.write_text("# fixture producer\n", encoding="utf-8")
    freeze_manifest_path = tmp_path / "intervention_freeze_manifest.json"
    _write_json(
        freeze_manifest_path,
        {
            "schema_version": "r2_p0_8_intervention_freeze_manifest_v2",
            "status": "frozen_before_test_evaluation",
            "task_id": "fixture-task",
            "cohort_sha256": evaluation_spec["sha256"],
            "annotation_sha256": sha256_file(annotation_path),
            "row_order_sha256": row_order_sha,
            "model_artifacts": model_artifacts,
            "dictionary_checkpoints": [
                {
                    "seed": item["seed"],
                    "checkpoint_sha256": item["checkpoint_sha256"],
                    "run_manifest_sha256": item["run_manifest_sha256"],
                }
                for item in dictionaries
            ],
            "quality_inventory": {
                "dictionary_seeds": [17, 29, 43],
                "layers": [0, 1],
            },
            "effect_definition": effect_definition,
            "producer": {
                "source": {
                    "path": str(producer_path),
                    "sha256": sha256_file(producer_path),
                },
                "command": ["python", str(producer_path), "--frozen-fixture"],
                "environment": {"python": "fixture"},
            },
        },
    )
    intervention_receipt = {
        "schema_version": "r2_p0_8_intervention_evidence_receipt_v3",
        "status": "verified_complete",
        "task_id": "fixture-task",
        "cohort_sha256": evaluation_spec["sha256"],
        "annotation_sha256": sha256_file(annotation_path),
        "row_order_sha256": row_order_sha,
        "model_artifacts": model_artifacts,
        "dictionary_checkpoints": [
            {
                "seed": item["seed"],
                "checkpoint_sha256": item["checkpoint_sha256"],
                "run_manifest_sha256": item["run_manifest_sha256"],
            }
            for item in dictionaries
        ],
        "quality_inventory": {
            "dictionary_seeds": [17, 29, 43],
            "layers": [0, 1],
        },
        "effect_definition": effect_definition,
        "test_evaluation_count": 1,
        "freeze_manifest": {
            "path": str(freeze_manifest_path),
            "sha256": sha256_file(freeze_manifest_path),
        },
        "artifact": {
            "path": str(intervention_path),
            "sha256": sha256_file(intervention_path),
        },
    }
    intervention_receipt_path = tmp_path / "intervention_receipt.json"
    _write_json(intervention_receipt_path, intervention_receipt)
    spec = {
        "schema_version": 3,
        "mode": "test_fixture",
        "device": "cpu",
        "max_model_tokens": 16,
        "task": {
            "task_id": "fixture-task",
            "task_type": "classification",
            "pooling": "mean_selected_residues",
            "minimum_samples": 8,
            "cohort": evaluation_spec,
            "annotations": {
                "path": str(annotation_path),
                "sha256": sha256_file(annotation_path),
            },
        },
        "prior_p0_2_cohorts": prior_specs,
        "identity_assignments": {
            "path": str(identity_path),
            "sha256": sha256_file(identity_path),
            "clustering_receipt": {
                "path": str(clustering_receipt_path),
                "sha256": sha256_file(clustering_receipt_path),
            },
        },
        "model": {
            "name": "fixture-model",
            "model_root": str(model_root),
            "model_artifacts": model_artifacts,
            "layers": [0, 1],
            "input_format": "sequence",
            "model_inference_dtype": "float32",
            "model_inference_dtype_verification": (
                "all_floating_model_parameters_exactly_declared_before_first_activation"
            ),
            "activation_finiteness_check": ACTIVATION_FINITE_CHECK,
        },
        "dictionaries": dictionaries,
        "p0_2_gate_receipt": {
            "path": str(p0_2_path),
            "sha256": sha256_file(p0_2_path),
        },
        "intervention_evidence": {
            "path": str(intervention_receipt_path),
            "sha256": sha256_file(intervention_receipt_path),
        },
        "analysis": {
            "analysis_seeds": [5],
            "outer_splits": 2,
            "inner_splits": 2,
            "n_bootstrap": 3,
            "comparison_dimension": 2,
            "active_width_dimension": None,
            "controls": [],
        },
    }
    spec_path = tmp_path / "spec.json"
    _write_json(spec_path, spec)
    return spec_path, spec


def _load_script55():
    path = R2 / "scripts/55_run_nested_recoverability.py"
    module_spec = importlib.util.spec_from_file_location("script55", path)
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)
    return module


def test_cpu_fake_model_build_is_directly_consumable_by_script55(tmp_path):
    spec_path, _ = _fixture(tmp_path)
    output = tmp_path / "output"
    receipt_path = build_recoverability_inputs(
        spec_path,
        output,
        model_loader=lambda model, device: TinyProteinModel(),
        dictionary_loader=_fake_dictionary_loader,
    )
    assert {path.name for path in output.iterdir()} == {
        "nested_recoverability_input.npz",
        "nested_recoverability_runner_spec.json",
        "input_receipt.json",
    }
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "verified_test_fixture_inputs_only"
    assert receipt["schema_version"] == "r2_p0_8_input_receipt_v3"
    assert receipt["model"]["model_inference_dtype"] == "float32"
    assert receipt["model"]["observed_model_parameter_dtypes"] == ["float32"]
    assert receipt["model"]["model_inference_dtype_verified"] is True
    assert receipt["model"]["activation_finiteness_check"] == ACTIVATION_FINITE_CHECK
    assert receipt["model"]["activation_finiteness_verified"] is True
    assert receipt["representation_extraction"]["labels_or_groups_passed_to_extractor"] is False
    runner = json.loads(
        (output / "nested_recoverability_runner_spec.json").read_text(encoding="utf-8")
    )
    assert runner["input"]["sha256"] == sha256_file(
        output / "nested_recoverability_input.npz"
    )
    representations, y, groups, reconstruction, intervention = _load_script55().load_npz(
        output / "nested_recoverability_input.npz"
    )
    assert set(representations) == {
        "clt_input",
        "mlp_output",
        "code_seed_17",
        "code_seed_29",
        "code_seed_43",
        "reconstruction_seed_17",
        "reconstruction_seed_29",
        "reconstruction_seed_43",
    }
    result = run_nested_recoverability(
        representations,
        y,
        groups,
        ceiling_name="clt_input",
        floor_names=["code_seed_17", "code_seed_29", "code_seed_43"],
        task_type="classification",
        analysis_seeds=[5],
        outer_splits=2,
        inner_splits=2,
        control_methods=[],
        comparison_dimension=2,
        n_bootstrap=3,
        reconstruction_error_by_floor_layer=reconstruction,
        intervention_effect_by_floor_layer=intervention,
        confirmatory_real=False,
    )
    assert result["confirmatory_real"] is False
    assert result["n_samples"] == 16
    assert set(reconstruction) == {"code_seed_17", "code_seed_29", "code_seed_43"}
    assert all(set(layers) == {"0", "1"} for layers in reconstruction.values())
    assert all(set(layers) == {"0", "1"} for layers in intervention.values())


def test_script55_real_mode_requires_matching_production_receipt(tmp_path):
    spec_path, _ = _fixture(tmp_path)
    output = tmp_path / "output"
    receipt_path = build_recoverability_inputs(
        spec_path,
        output,
        model_loader=lambda model, device: TinyProteinModel(),
        dictionary_loader=_fake_dictionary_loader,
    )
    runner_path = output / "nested_recoverability_runner_spec.json"
    runner = json.loads(runner_path.read_text(encoding="utf-8"))
    runner["confirmatory_real"] = True
    _write_json(runner_path, runner)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "verified_production_inputs_not_scientifically_adjudicated"
    receipt["model"]["model_inference_dtype"] = "bfloat16"
    receipt["model"]["observed_model_parameter_dtypes"] = ["bfloat16"]
    receipt["outputs"][runner_path.name] = sha256_file(runner_path)
    _write_json(receipt_path, receipt)
    arguments = runner["arguments"]
    args = SimpleNamespace(
        input_sha256=runner["input"]["sha256"],
        input_receipt=receipt_path,
        input_receipt_sha256=sha256_file(receipt_path),
        **arguments,
    )
    observed = _load_script55().verify_production_input_receipt(
        args, output / "nested_recoverability_input.npz"
    )
    assert observed == (receipt_path.resolve(), sha256_file(receipt_path))
    args.outer_splits += 1
    with pytest.raises(ValueError, match="arguments differ"):
        _load_script55().verify_production_input_receipt(
            args, output / "nested_recoverability_input.npz"
        )
    args.outer_splits -= 1
    receipt["model"]["activation_finiteness_verified"] = False
    _write_json(receipt_path, receipt)
    args.input_receipt_sha256 = sha256_file(receipt_path)
    with pytest.raises(ValueError, match="lacks verified bfloat16 finite inference"):
        _load_script55().verify_production_input_receipt(
            args, output / "nested_recoverability_input.npz"
        )
    receipt["model"]["activation_finiteness_verified"] = True
    receipt["schema_version"] = "r2_p0_8_input_receipt_v2"
    _write_json(receipt_path, receipt)
    args.input_receipt_sha256 = sha256_file(receipt_path)
    with pytest.raises(ValueError, match="not production-eligible"):
        _load_script55().verify_production_input_receipt(
            args, output / "nested_recoverability_input.npz"
        )


def test_script55_rejects_fixture_receipt_for_real_mode(tmp_path):
    spec_path, _ = _fixture(tmp_path)
    output = tmp_path / "output"
    receipt_path = build_recoverability_inputs(
        spec_path,
        output,
        model_loader=lambda model, device: TinyProteinModel(),
        dictionary_loader=_fake_dictionary_loader,
    )
    runner = json.loads(
        (output / "nested_recoverability_runner_spec.json").read_text(encoding="utf-8")
    )
    args = SimpleNamespace(
        input_sha256=runner["input"]["sha256"],
        input_receipt=receipt_path,
        input_receipt_sha256=sha256_file(receipt_path),
        **runner["arguments"],
    )
    with pytest.raises(ValueError, match="not production-eligible"):
        _load_script55().verify_production_input_receipt(
            args, output / "nested_recoverability_input.npz"
        )


def test_rejects_arbitrary_representation_arrays_in_spec(tmp_path):
    spec_path, spec = _fixture(tmp_path)
    spec["representations"] = {"path": "arbitrary.npz"}
    _write_json(spec_path, spec)
    with pytest.raises(ValueError, match="fields differ"):
        build_recoverability_inputs(
            spec_path,
            tmp_path / "output",
            model_loader=lambda model, device: TinyProteinModel(),
            dictionary_loader=_fake_dictionary_loader,
        )


def test_rejects_nonpassing_p0_2_receipt(tmp_path):
    spec_path, spec = _fixture(tmp_path)
    receipt_path = Path(spec["p0_2_gate_receipt"]["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["model_method_adjudications"][0]["status"] = "gate_failed"
    receipt["model_method_adjudications"][0]["atlas_eligible"] = False
    receipt["model_method_adjudications"][0]["failure_reasons"] = ["fixture failure"]
    _write_json(receipt_path, receipt)
    spec["p0_2_gate_receipt"]["sha256"] = sha256_file(receipt_path)
    _write_json(spec_path, spec)
    with pytest.raises(ValueError, match="did not pass"):
        build_recoverability_inputs(
            spec_path,
            tmp_path / "output",
            model_loader=lambda model, device: TinyProteinModel(),
            dictionary_loader=_fake_dictionary_loader,
        )


def test_rejects_checkpoint_hash_or_seed_tamper(tmp_path):
    spec_path, spec = _fixture(tmp_path)
    spec["dictionaries"][0]["checkpoint_sha256"] = "0" * 64
    _write_json(spec_path, spec)
    with pytest.raises(ValueError, match="best.pt SHA-256 mismatch"):
        build_recoverability_inputs(
            spec_path,
            tmp_path / "output",
            model_loader=lambda model, device: TinyProteinModel(),
            dictionary_loader=_fake_dictionary_loader,
        )


def test_rejects_identity_assignments_not_bound_by_clustering_receipt(tmp_path):
    spec_path, spec = _fixture(tmp_path)
    identity_path = Path(spec["identity_assignments"]["path"])
    rows = [json.loads(line) for line in identity_path.read_text().splitlines()]
    rows[0]["identity_group"] = "self-asserted-new-group"
    _write_jsonl(identity_path, rows)
    spec["identity_assignments"]["sha256"] = sha256_file(identity_path)
    _write_json(spec_path, spec)
    with pytest.raises(ValueError, match="clustering receipt is not bound"):
        build_recoverability_inputs(
            spec_path,
            tmp_path / "output",
            model_loader=lambda model, device: TinyProteinModel(),
            dictionary_loader=_fake_dictionary_loader,
        )


def test_rejects_identity_group_leakage(tmp_path):
    spec_path, spec = _fixture(tmp_path)
    identity_path = Path(spec["identity_assignments"]["path"])
    rows = [json.loads(line) for line in identity_path.read_text().splitlines()]
    prior_group = rows[0]["identity_group"]
    next(row for row in rows if row["cohort_role"] == "p0_8_evaluation")[
        "identity_group"
    ] = prior_group
    _write_jsonl(identity_path, rows)
    annotation_path = Path(spec["task"]["annotations"]["path"])
    annotations = [json.loads(line) for line in annotation_path.read_text().splitlines()]
    annotations[0]["identity_group"] = prior_group
    _write_jsonl(annotation_path, annotations)
    spec["identity_assignments"]["sha256"] = sha256_file(identity_path)
    clustering_path = Path(
        spec["identity_assignments"]["clustering_receipt"]["path"]
    )
    clustering = json.loads(clustering_path.read_text(encoding="utf-8"))
    clustering["assignment_sha256"] = sha256_file(identity_path)
    _write_json(clustering_path, clustering)
    spec["identity_assignments"]["clustering_receipt"]["sha256"] = sha256_file(
        clustering_path
    )
    spec["task"]["annotations"]["sha256"] = sha256_file(annotation_path)
    _write_json(spec_path, spec)
    with pytest.raises(ValueError, match="identity-group leakage"):
        build_recoverability_inputs(
            spec_path,
            tmp_path / "output",
            model_loader=lambda model, device: TinyProteinModel(),
            dictionary_loader=_fake_dictionary_loader,
        )


def test_rejects_ungrounded_intervention_freeze_manifest_digest(tmp_path):
    spec_path, spec = _fixture(tmp_path)
    receipt_path = Path(spec["intervention_evidence"]["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["freeze_manifest"]["sha256"] = "0" * 64
    _write_json(receipt_path, receipt)
    spec["intervention_evidence"]["sha256"] = sha256_file(receipt_path)
    _write_json(spec_path, spec)
    with pytest.raises(ValueError, match="freeze manifest path or SHA-256 mismatch"):
        build_recoverability_inputs(
            spec_path,
            tmp_path / "output",
            model_loader=lambda model, device: TinyProteinModel(),
            dictionary_loader=_fake_dictionary_loader,
        )


def test_rejects_missing_or_reordered_intervention_quality_rows(tmp_path):
    spec_path, spec = _fixture(tmp_path)
    receipt_path = Path(spec["intervention_evidence"]["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    artifact_path = Path(receipt["artifact"]["path"])
    rows = [json.loads(line) for line in artifact_path.read_text().splitlines()]
    _write_jsonl(artifact_path, rows[:-1])
    receipt["artifact"]["sha256"] = sha256_file(artifact_path)
    _write_json(receipt_path, receipt)
    spec["intervention_evidence"]["sha256"] = sha256_file(receipt_path)
    _write_json(spec_path, spec)
    with pytest.raises(ValueError, match="row count mismatch"):
        build_recoverability_inputs(
            spec_path,
            tmp_path / "output",
            model_loader=lambda model, device: TinyProteinModel(),
            dictionary_loader=_fake_dictionary_loader,
        )


def test_rejects_incomplete_intervention_seed_layer_inventory(tmp_path):
    spec_path, spec = _fixture(tmp_path)
    receipt_path = Path(spec["intervention_evidence"]["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    artifact_path = Path(receipt["artifact"]["path"])
    rows = [json.loads(line) for line in artifact_path.read_text().splitlines()]
    rows[0]["intervention_effect_by_seed_layer"]["17"].pop("1")
    _write_jsonl(artifact_path, rows)
    receipt["artifact"]["sha256"] = sha256_file(artifact_path)
    _write_json(receipt_path, receipt)
    spec["intervention_evidence"]["sha256"] = sha256_file(receipt_path)
    _write_json(spec_path, spec)
    with pytest.raises(ValueError, match="layer inventory differs"):
        build_recoverability_inputs(
            spec_path,
            tmp_path / "output",
            model_loader=lambda model, device: TinyProteinModel(),
            dictionary_loader=_fake_dictionary_loader,
        )


def test_production_rejects_pre_enlargement_sample_minimum(tmp_path):
    spec_path, spec = _fixture(tmp_path)
    spec["mode"] = "production"
    _write_json(spec_path, spec)
    with pytest.raises(ValueError, match="enlarged-cohort minimum of 480"):
        build_recoverability_inputs(spec_path, tmp_path / "output")


def test_production_freezes_bfloat16_and_example_schema(tmp_path):
    spec_path, spec = _fixture(tmp_path)
    with pytest.raises(ValueError, match="production model inference must use bfloat16"):
        _validate_model(spec["model"], spec_path.parent, production=True)
    example = json.loads(
        (
            R2
            / "configs/p0_8_recoverability_input_builder_spec.example.json"
        ).read_text(encoding="utf-8")
    )
    assert example["schema_version"] == 3
    assert example["model"]["model_inference_dtype"] == "bfloat16"
    assert (
        example["model"]["model_inference_dtype_verification"]
        == "all_floating_model_parameters_exactly_declared_before_first_activation"
    )
    assert example["model"]["activation_finiteness_check"] == ACTIVATION_FINITE_CHECK


def test_rejects_dtype_mismatch_and_nonfinite_capture(tmp_path):
    spec_path, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="parameter dtype disagrees"):
        build_recoverability_inputs(
            spec_path,
            tmp_path / "wrong-dtype",
            model_loader=lambda model, device: TinyProteinModel(dtype=torch.bfloat16),
            dictionary_loader=_fake_dictionary_loader,
        )
    with pytest.raises(FloatingPointError, match="non-finite frozen-model"):
        build_recoverability_inputs(
            spec_path,
            tmp_path / "nonfinite",
            model_loader=lambda model, device: TinyProteinModel(nonfinite=True),
            dictionary_loader=_fake_dictionary_loader,
        )


def test_rejects_custom_extractor_in_production_mode(tmp_path):
    spec_path, spec = _fixture(tmp_path)
    spec["mode"] = "production"
    _write_json(spec_path, spec)
    with pytest.raises(ValueError, match="forbidden in production"):
        build_recoverability_inputs(
            spec_path,
            tmp_path / "output",
            model_loader=lambda model, device: TinyProteinModel(),
            dictionary_loader=_fake_dictionary_loader,
            extractor=lambda *args, **kwargs: ({}, np.empty(0)),
        )
