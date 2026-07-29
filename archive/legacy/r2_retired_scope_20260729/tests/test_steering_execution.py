from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.revision.input_builder import _tree_digest
from src.revision.io import sha256_file, write_json
from src.revision.steering_execution import (
    ACTIVATION_FINITE_CHECK,
    execute_plan_rows,
    load_execution_spec,
    load_verified_freeze,
    run_execution,
    verify_execution_receipt,
)
from src.revision.steering_protocol import (
    build_steering_plan,
    select_positive_features,
    validate_completed_generations,
)
from src.training.clt_trainer import CLTForTraining


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/60_prepare_corrected_steering.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("steering_freeze_60", SCRIPT)
FREEZER = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC.loader is not None
SCRIPT_SPEC.loader.exec_module(FREEZER)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class TinyTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    alphabet = "ACDEFGHIKLMNPQRSTVWY"

    def __call__(self, _text, *, return_tensors):
        assert return_tensors == "pt"
        input_ids = torch.tensor([[30, 31]], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens
        return "".join(self.alphabet[token_id - 2] for token_id in token_ids if token_id)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Identity()


class TinyGenerator(nn.Module):
    def __init__(
        self,
        *,
        fail: bool = False,
        parameter_dtype=torch.bfloat16,
        nonfinite_activation=False,
        nonfinite_logits=False,
    ):
        super().__init__()
        self.transformer = SimpleNamespace(h=nn.ModuleList([TinyBlock()]))
        self.anchor = nn.Parameter(torch.zeros((), dtype=parameter_dtype))
        self.fail = fail
        self.nonfinite_activation = nonfinite_activation
        self.nonfinite_logits = nonfinite_logits
        self.forward_calls = 0

    def forward(self, *, input_ids, attention_mask, use_cache):
        assert attention_mask.shape == input_ids.shape and use_cache
        self.forward_calls += 1
        hidden = torch.zeros((1, 1, 4), dtype=torch.float32)
        if self.nonfinite_activation:
            hidden[0, 0, 0] = torch.inf
        hidden = self.transformer.h[0].mlp(hidden)
        logits = torch.zeros((1, 1, 32), dtype=torch.float32)
        if bool(torch.isfinite(hidden).all().item()):
            logits[0, 0, 0] = hidden.abs().sum()
        if self.nonfinite_logits:
            logits[0, 0, 0] = torch.nan
        return SimpleNamespace(logits=logits)

    def generate(
        self,
        *,
        input_ids,
        attention_mask,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        do_sample,
        use_cache,
        pad_token_id,
        eos_token_id,
    ):
        if self.fail:
            raise RuntimeError("planted generation failure")
        assert attention_mask.shape == input_ids.shape
        assert temperature == 0.8 and top_p == 0.95
        assert top_k == 0 and do_sample and use_cache
        assert pad_token_id == 0 and eos_token_id == [0]
        output = input_ids.clone()
        for _ in range(max_new_tokens):
            forward = self(
                input_ids=output,
                attention_mask=torch.ones_like(output),
                use_cache=use_cache,
            )
            random_index = int(torch.randint(0, 20, (1,)).item())
            hook_offset = int(round(float(forward.logits[0, 0, 0]) * 3.0)) % 20
            token = torch.tensor([[2 + (random_index + hook_offset) % 20]])
            output = torch.cat((output, token), dim=1)
        return output


class TinyProteinModel:
    n_layers = 1
    d_model = 4
    device = "cpu"

    def __init__(self, **model_kwargs):
        self.tokenizer = TinyTokenizer()
        self.model = TinyGenerator(**model_kwargs).eval()

    def _get_block(self, layer):
        return self.model.transformer.h[layer]

    @staticmethod
    def _get_mlp(block):
        return block.mlp


def make_clt() -> CLTForTraining:
    clt = CLTForTraining(n_layers=1, d_model=4, d_clt=4, k=2, window=1)
    with torch.no_grad():
        clt.W_dec[0].zero_()
        for feature in range(4):
            clt.W_dec[0][feature, 0, feature] = 1.0
    return clt.eval()


def plan_inputs(sites=("clt_input", "mlp_output")):
    attributions = [
        {
            "ec_class": "lysozyme",
            "layer": 0,
            "site": site,
            "feature": 0,
            "direct_effect": 1.0,
            "decoder_norm": 1.0,
            "split_id": "selection-v1",
        }
        for site in sites
    ]
    pool = [
        {
            "layer": 0,
            "site": site,
            "feature": feature,
            "decoder_norm": 1.0,
        }
        for site in sites
        for feature in (1, 2, 3)
    ]
    return attributions, pool


def make_plan(*, sites=("clt_input", "mlp_output"), semantics="additive_decoder_direction_v1"):
    attributions, pool = plan_inputs(sites)
    selection = select_positive_features(
        attributions,
        selection_split_id="selection-v1",
        evaluation_split_id="evaluation-v1",
        classes=["lysozyme"],
        layers=[0],
        sites=list(sites),
        features_per_cell=1,
    )
    return build_steering_plan(
        selection,
        pool,
        classes=["lysozyme"],
        layers=[0],
        sites=list(sites),
        doses=[0.5, 1.0],
        n_per_arm=2,
        generation_set_size=1,
        seed_base=101,
        sampler={"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 4},
        norm_log_caliper=0.1,
        content_binding_sha256=digest("binding"),
        generator_revision="generator-v1",
        model_revision="tiny-model-v1",
        tokenizer_revision="tiny-tokenizer-v1",
        clt_checkpoint_sha256=digest("temporary-clt"),
        multiplier_semantics=semantics,
    )


def make_checkpoint(tmp_path: Path, clt: CLTForTraining) -> Path:
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "schema_version": "r2_dictionary_control_best_v1",
            "candidate_id": "full_topk_clt_seed_17_candidate_000_fixture",
            "step": 5_000,
            "validation_fvu_mean": 0.1,
            "model_state_dict": clt.state_dict(),
        },
        checkpoint,
    )
    return checkpoint


def make_p0_2_receipt(tmp_path: Path, checkpoint: Path) -> tuple[Path, dict, dict]:
    run_hashes = {str(seed): digest(f"run-{seed}") for seed in (17, 29, 43)}
    checkpoint_hashes = {
        "17": sha256_file(checkpoint),
        "29": digest("checkpoint-29"),
        "43": digest("checkpoint-43"),
    }
    source_hashes = {
        "train": digest("train"),
        "validation": digest("validation"),
        "test": digest("test"),
    }
    receipt = {
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
        "profile_sha256": digest("profile"),
        "protocol_sha256": digest("protocol"),
        "mask_validation_receipt_sha256": digest("mask"),
        "frozen_gate": {"required_seeds": [17, 29, 43]},
        "model_method_adjudications": [
            {
                "model_name": "tiny",
                "method": "topk_clt",
                "status": "atlas_eligible",
                "atlas_eligible": True,
                "failure_reasons": [],
                "required_seeds": [17, 29, 43],
                "source_manifest_sha256_by_split": source_hashes,
                "cache_manifest_sha256": digest("cache-manifest"),
                "cache_content_sha256": digest("cache-content"),
                "selected_layers": [0],
                "eligible_downstream_layers": [0],
                "geometry": {
                    "n_layers": 1,
                    "d_model": 4,
                    "d_clt": 4,
                    "k": 2,
                    "window": 1,
                },
                "runs": [
                    {
                        "run_seed": seed,
                        "run_manifest_sha256": run_hashes[str(seed)],
                        "result_sha256": digest(f"result-{seed}"),
                        "checkpoint_sha256": checkpoint_hashes[str(seed)],
                        "quality_gate_pass": True,
                        "failure_reasons": [],
                    }
                    for seed in (17, 29, 43)
                ],
            }
        ],
    }
    path = tmp_path / "p0_2_eligibility_receipt.json"
    write_json(path, receipt)
    return path, run_hashes, checkpoint_hashes


def fake_dictionary_loader(
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
    assert sha256_file(receipt_path) == expected_receipt_sha256
    assert sha256_file(checkpoint_path) == expected_checkpoint_sha256_by_seed[run_seed]
    clt = make_clt().to(map_location)
    return clt, {
        "model_name": model_name,
        "method": "topk_clt",
        "run_seed": run_seed,
        "eligibility_receipt_sha256": expected_receipt_sha256,
        "run_manifest_sha256": expected_run_manifest_sha256_by_seed[run_seed],
        "checkpoint_sha256": expected_checkpoint_sha256_by_seed[run_seed],
        "candidate_id": "fixture",
        "checkpoint_step": 5_000,
        "geometry": {
            "n_layers": 1,
            "d_model": 4,
            "d_clt": 4,
            "k": 2,
            "window": 1,
        },
        "eligible_downstream_layers": list(requested_layers),
        "source_manifest_sha256_by_split": dict(
            expected_source_manifest_sha256_by_split
        ),
    }


def make_model_tree(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"tiny"}\n', encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"tiny-weights")
    (root / "tokenizer.json").write_text('{"tiny":true}\n', encoding="utf-8")
    return root, {
        "model_config_sha256": sha256_file(root / "config.json"),
        "model_weights_sha256": _tree_digest(root, [root / "model.safetensors"]),
        "tokenizer_sha256": _tree_digest(root, [root / "tokenizer.json"]),
    }


def make_freeze(tmp_path: Path, checkpoint: Path) -> Path:
    attributions, pool = plan_inputs()
    args = SimpleNamespace(
        selection_split_id="selection-v1",
        evaluation_split_id="evaluation-v1",
        classes=["lysozyme"],
        layers=[0],
        sites=["clt_input", "mlp_output"],
        doses=[0.5, 1.0],
        features_per_cell=1,
        n_per_arm=2,
        generation_set_size=1,
        norm_log_caliper=0.1,
        seed_base=101,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=4,
        multiplier_semantics="additive_decoder_direction_v1",
    )
    clt_hash = sha256_file(checkpoint)
    provenance = {
        "model_revision": "tiny-model-v1",
        "tokenizer_revision": "tiny-tokenizer-v1",
        "clt_checkpoint_sha256": clt_hash,
        "selection_cohort_sha256": digest("selection-cohort"),
        "evaluation_cohort_sha256": digest("evaluation-cohort"),
        "code_revision": "generator-v1",
    }
    scorer_path = tmp_path / "fixture_scorer.bin"
    calibration_path = tmp_path / "fixture_calibration.jsonl"
    scorer_path.write_bytes(b"fixture-scorer")
    calibration_path.write_bytes(b'{"fixture":"calibration"}\n')
    endpoints = {
        "validated": {
            "direction": "higher",
            "experimental_unit": "generation",
            "equivalence_margin": 0.05,
            "validated": True,
            "primary": True,
            "scorer_name": "fixture-scorer",
            "scorer_version": "v1",
            "scorer_path": str(scorer_path),
            "scorer_sha256": sha256_file(scorer_path),
            "calibration_cohort_path": str(calibration_path),
            "calibration_cohort_sha256": sha256_file(calibration_path),
        }
    }
    analysis = {
        "alpha": 0.05,
        "n_resamples": 100,
        "random_seed": 101,
        "multiplicity": "holm_all_arm_and_specificity_cells",
        "decision_rule": "target_vs_prompt_and_both_controls",
    }
    frozen_dir = tmp_path / "freeze"
    FREEZER.freeze_protocol(
        out_dir=frozen_dir,
        attribution_rows=attributions,
        feature_pool=pool,
        endpoint_specs=endpoints,
        provenance=provenance,
        analysis_spec=analysis,
        args=args,
        input_hashes={},
        cohort_validation={"selection_evaluation_disjoint": True},
        synthetic=False,
        endpoint_artifact_base=tmp_path,
    )
    return frozen_dir


def make_spec(
    tmp_path: Path, model_root: Path, model_artifacts: dict, checkpoint: Path
) -> Path:
    receipt_path, run_hashes, checkpoint_hashes = make_p0_2_receipt(
        tmp_path, checkpoint
    )
    spec_path = tmp_path / "execution_spec.json"
    write_json(
        spec_path,
        {
            "schema_version": 3,
            "device": "cpu",
            "model": {
                "name": "tiny",
                "model_root": str(model_root),
                "model_revision": "tiny-model-v1",
                "tokenizer_revision": "tiny-tokenizer-v1",
                "model_artifacts": model_artifacts,
                "model_inference_dtype": "bfloat16",
                "model_inference_dtype_verification": (
                    "all_floating_model_parameters_exactly_declared_before_first_activation"
                ),
                "activation_finiteness_check": ACTIVATION_FINITE_CHECK,
            },
            "p0_2": {
                "eligibility_receipt": {
                    "path": str(receipt_path),
                    "sha256": sha256_file(receipt_path),
                },
                "method": "topk_clt",
                "run_seed": 17,
                "checkpoint_path": str(checkpoint),
                "run_manifest_sha256_by_seed": run_hashes,
                "checkpoint_sha256_by_seed": checkpoint_hashes,
                "source_manifest_sha256_by_split": {
                    "train": digest("train"),
                    "validation": digest("validation"),
                    "test": digest("test"),
                },
                "requested_layers": [0],
            },
        },
    )
    return spec_path


def test_prompt_target_random_norm_sites_and_deterministic_rng() -> None:
    clt = make_clt()
    rows = make_plan()["rows"]
    clt_hash = rows[0]["clt_checkpoint_sha256"]
    assert clt_hash == digest("temporary-clt")
    model = TinyProteinModel()
    first = execute_plan_rows(model, clt, rows, device="cpu")
    second = execute_plan_rows(model, clt, rows, device="cpu")
    assert len(first) == 26
    assert {row["arm"] for row in rows} == {
        "prompt_only",
        "target",
        "random_feature",
        "norm_matched_feature",
    }
    assert {row["site"] for row in rows} == {"none", "clt_input", "mlp_output"}
    assert [row["token_ids"] for row in first] == [row["token_ids"] for row in second]
    assert [row["sequence"] for row in first] == [row["sequence"] for row in second]
    completed = validate_completed_generations(rows, first)
    assert len(completed) == len(rows)
    assert all(row["runtime"]["hook_effect_verified"] for row in first)
    assert all(row["runtime"]["activation_finiteness_verified"] for row in first)
    assert all(row["runtime"]["generation_logit_tensors_checked"] == 4 for row in first)
    assert all(
        row["runtime"]["hook_receipts"]
        for row, plan in zip(first, rows)
        if plan["arm"] != "prompt_only"
    )
    block = model.model.transformer.h[0]
    assert not block.mlp._forward_hooks and not block.mlp._forward_pre_hooks


def test_unsupported_site_and_multiplier_fail_before_generation() -> None:
    clt = make_clt()
    with pytest.raises(ValueError, match="unsupported multiplier semantics"):
        execute_plan_rows(
            TinyProteinModel(),
            clt,
            make_plan(semantics="multiply_active_feature_v0")["rows"],
            device="cpu",
        )
    with pytest.raises(ValueError, match="unsupported hook site"):
        execute_plan_rows(
            TinyProteinModel(),
            clt,
            make_plan(sites=("resid_post",))["rows"],
            device="cpu",
        )


@pytest.mark.parametrize(
    ("model_kwargs", "message"),
    [
        ({"nonfinite_activation": True}, "hook tensor geometry/value failure"),
        ({"nonfinite_logits": True}, "non-finite generation logits"),
    ],
)
def test_generation_rejects_nonfinite_activation_or_logits_before_use(
    model_kwargs, message
) -> None:
    with pytest.raises((ValueError, FloatingPointError), match=message):
        execute_plan_rows(
            TinyProteinModel(**model_kwargs),
            make_clt(),
            make_plan()["rows"],
            device="cpu",
        )


def test_execution_spec_rejects_online_clt_checkpoint(tmp_path: Path) -> None:
    clt = make_clt()
    checkpoint = make_checkpoint(tmp_path, clt)
    model_root, artifacts = make_model_tree(tmp_path)
    spec_path = make_spec(tmp_path, model_root, artifacts, checkpoint)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["p0_2"]["checkpoint_path"] = str(tmp_path / "preliminary" / "clt.pt")
    write_json(spec_path, spec)
    with pytest.raises(ValueError, match="rejects online clt.pt"):
        load_execution_spec(spec_path, sha256_file(spec_path))


def test_freeze_tamper_detection_and_atomic_full_execution(tmp_path: Path) -> None:
    clt = make_clt()
    checkpoint = make_checkpoint(tmp_path, clt)
    model_root, artifacts = make_model_tree(tmp_path)
    frozen_dir = make_freeze(tmp_path, checkpoint)
    spec_path = make_spec(tmp_path, model_root, artifacts, checkpoint)
    freeze_hash = sha256_file(frozen_dir / "run_manifest.json")
    verified = load_verified_freeze(frozen_dir, freeze_hash)
    assert len(verified["rows"]) == 26

    output = tmp_path / "execution"
    receipt_path = run_execution(
        frozen_dir=frozen_dir,
        freeze_manifest_sha256=freeze_hash,
        spec_path=spec_path,
        spec_sha256=sha256_file(spec_path),
        output_dir=output,
        model_loader=lambda *_args, **_kwargs: TinyProteinModel(),
        dictionary_loader=fake_dictionary_loader,
        command=["cpu-contract-test"],
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    outputs = [
        json.loads(line)
        for line in (output / "generation_outputs.jsonl").read_text().splitlines()
    ]
    assert receipt["status"] == "verified_test_fixture_complete"
    assert receipt["schema_version"] == "r2-corrected-steering-execution-receipt-v3"
    assert receipt["model"]["model_inference_dtype"] == "bfloat16"
    assert receipt["model"]["observed_model_parameter_dtypes"] == ["bfloat16"]
    assert receipt["model"]["model_inference_dtype_verified"] is True
    assert receipt["model"]["activation_finiteness_check"] == ACTIVATION_FINITE_CHECK
    assert receipt["model"]["activation_finiteness_verified"] is True
    assert receipt["model"]["p0_2_eligibility"]["method"] == "topk_clt"
    assert receipt["model"]["p0_2_eligibility"]["checkpoint_sha256"] == sha256_file(
        checkpoint
    )
    assert receipt["freeze"]["manifest_sha256"] == freeze_hash
    assert receipt["artifacts"]["generation_outputs.jsonl"] == sha256_file(
        output / "generation_outputs.jsonl"
    )
    assert len(outputs) == 26
    validate_completed_generations(verified["rows"], outputs)
    verified_receipt = verify_execution_receipt(
        receipt_path,
        sha256_file(receipt_path),
        generation_outputs_path=output / "generation_outputs.jsonl",
        freeze_id=verified["summary"]["freeze_id"],
        freeze_manifest_sha256=freeze_hash,
        allow_test_fixture=True,
    )
    assert verified_receipt["model"]["p0_2_eligibility"]["run_seed"] == 17
    with pytest.raises(ValueError, match="complete v3 production receipt"):
        verify_execution_receipt(
            receipt_path,
            sha256_file(receipt_path),
            generation_outputs_path=output / "generation_outputs.jsonl",
            freeze_id=verified["summary"]["freeze_id"],
            freeze_manifest_sha256=freeze_hash,
        )
    with pytest.raises(FileExistsError, match="overwrite"):
        run_execution(
            frozen_dir=frozen_dir,
            freeze_manifest_sha256=freeze_hash,
            spec_path=spec_path,
            spec_sha256=sha256_file(spec_path),
            output_dir=output,
            model_loader=lambda *_args, **_kwargs: TinyProteinModel(),
            dictionary_loader=fake_dictionary_loader,
        )

    failed_output = tmp_path / "failed_execution"
    with pytest.raises(RuntimeError, match="planted generation failure"):
        run_execution(
            frozen_dir=frozen_dir,
            freeze_manifest_sha256=freeze_hash,
            spec_path=spec_path,
            spec_sha256=sha256_file(spec_path),
            output_dir=failed_output,
            model_loader=lambda *_args, **_kwargs: TinyProteinModel(fail=True),
            dictionary_loader=fake_dictionary_loader,
        )
    assert not failed_output.exists()
    assert not list(tmp_path.glob(".failed_execution.tmp-*"))

    wrong_dtype_output = tmp_path / "wrong_dtype_execution"
    wrong_dtype_model = TinyProteinModel(parameter_dtype=torch.float32)
    with pytest.raises(ValueError, match="parameter dtype disagrees"):
        run_execution(
            frozen_dir=frozen_dir,
            freeze_manifest_sha256=freeze_hash,
            spec_path=spec_path,
            spec_sha256=sha256_file(spec_path),
            output_dir=wrong_dtype_output,
            model_loader=lambda *_args, **_kwargs: wrong_dtype_model,
            dictionary_loader=fake_dictionary_loader,
        )
    assert wrong_dtype_model.model.forward_calls == 0
    assert not wrong_dtype_output.exists()

    receipt_bytes = receipt_path.read_bytes()
    tampered_receipt = json.loads(receipt_bytes)
    tampered_receipt["model"]["activation_finiteness_verified"] = False
    write_json(receipt_path, tampered_receipt)
    with pytest.raises(ValueError, match="bfloat16 numerical integrity"):
        verify_execution_receipt(
            receipt_path,
            sha256_file(receipt_path),
            generation_outputs_path=output / "generation_outputs.jsonl",
            freeze_id=verified["summary"]["freeze_id"],
            freeze_manifest_sha256=freeze_hash,
            allow_test_fixture=True,
        )
    receipt_path.write_bytes(receipt_bytes)

    legacy_receipt = json.loads(receipt_bytes)
    legacy_receipt["schema_version"] = "r2-corrected-steering-execution-receipt-v2"
    write_json(receipt_path, legacy_receipt)
    with pytest.raises(ValueError, match="complete v3 production receipt"):
        verify_execution_receipt(
            receipt_path,
            sha256_file(receipt_path),
            generation_outputs_path=output / "generation_outputs.jsonl",
            freeze_id=verified["summary"]["freeze_id"],
            freeze_manifest_sha256=freeze_hash,
            allow_test_fixture=True,
        )
    receipt_path.write_bytes(receipt_bytes)

    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint.write_bytes(checkpoint_bytes + b"tamper")
    with pytest.raises(ValueError, match="hash drifted"):
        verify_execution_receipt(
            receipt_path,
            sha256_file(receipt_path),
            generation_outputs_path=output / "generation_outputs.jsonl",
            freeze_id=verified["summary"]["freeze_id"],
            freeze_manifest_sha256=freeze_hash,
            allow_test_fixture=True,
        )
    checkpoint.write_bytes(checkpoint_bytes)

    with (frozen_dir / "generation_plan.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="frozen artifact SHA-256 mismatch"):
        load_verified_freeze(frozen_dir, freeze_hash)
