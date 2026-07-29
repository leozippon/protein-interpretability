from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.revision.input_builder import _tree_digest
from src.revision.io import sha256_file, write_json, write_jsonl
from src.revision.n_terminal_counterfactuals import build_counterfactual_variants
from src.revision.n_terminal_extractor import (
    ACTIVATION_FINITE_CHECK,
    _load_eligible_dictionary,
    capture_clt_inputs_and_attentions,
    extract_measurement_rows,
    load_and_match_features,
    match_protein_controls,
    run_extractor,
    tokenize_bos_factorial,
    validate_focal_position_contract,
)
from src.training.clt_trainer import CLTForTraining


class TinyTokenizer:
    bos_token_id = 1
    _alphabet = {
        residue: index + 2 for index, residue in enumerate("ACDEFGHIKLMNPQRSTVWY")
    }
    _inverse = {value: key for key, value in _alphabet.items()}

    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        return_tensors,
        truncation,
        return_attention_mask,
    ):
        assert return_tensors == "pt" and truncation is False and return_attention_mask
        ids = [self._alphabet[residue] for residue in text]
        if add_special_tokens:
            ids = [self.bos_token_id, *ids]
        tensor = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}

    def decode(self, token_ids, *, skip_special_tokens):
        pieces = []
        for token_id in token_ids:
            if token_id == self.bos_token_id and skip_special_tokens:
                continue
            pieces.append(self._inverse.get(token_id, ""))
        return "".join(pieces)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Identity()


class TinyCausalModel(nn.Module):
    def __init__(
        self,
        *,
        parameter_dtype=torch.bfloat16,
        nonfinite_activation=False,
        nonfinite_logits=False,
    ):
        super().__init__()
        self.transformer = SimpleNamespace(h=nn.ModuleList([TinyBlock()]))
        self.anchor = nn.Parameter(torch.zeros((), dtype=parameter_dtype))
        self.nonfinite_activation = nonfinite_activation
        self.nonfinite_logits = nonfinite_logits
        self.forward_calls = 0

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        output_attentions,
        use_cache,
    ):
        self.forward_calls += 1
        assert output_attentions is True and use_cache is False
        hidden = torch.stack(
            (input_ids.float(), torch.ones_like(input_ids).float()), dim=-1
        )
        if self.nonfinite_activation:
            hidden[0, 0, 0] = torch.inf
        hidden = self.transformer.h[0].mlp(hidden)
        length = input_ids.shape[1]
        attention = torch.zeros((1, 1, length, length), dtype=torch.float32)
        for query in range(length):
            valid_keys = [
                key
                for key in range(query + 1)
                if attention_mask[0, key].item() == 1
            ]
            if valid_keys:
                attention[0, 0, query, valid_keys] = 1.0 / len(valid_keys)
        context = attention[0, 0] @ hidden[0]
        logits = torch.zeros((1, length, 32), dtype=torch.float32)
        logits[0, :, :2] = context
        if self.nonfinite_logits:
            logits[0, 0, 0] = torch.nan
        return SimpleNamespace(attentions=(attention,), logits=logits)


class TinyProteinModel:
    n_layers = 1
    d_model = 2
    device = "cpu"

    def __init__(self, **model_kwargs):
        self.tokenizer = TinyTokenizer()
        self.model = TinyCausalModel(**model_kwargs).eval()

    def _get_block(self, layer):
        return self.model.transformer.h[layer]


def _record(protein_id: str, residue: str, length: int, focal: int) -> dict:
    sequence = "M" + residue * (length - 1)
    return {
        "protein_id": protein_id,
        "sequence": sequence,
        "sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "focal_position": focal,
    }


def _internal_focal(length: int) -> int:
    return min(max(int(round((length - 3) * 0.55)), 4), length - 4)


def test_protein_controls_use_complete_global_caliper_matching() -> None:
    targets = [_record("t1", "A", 20, 0), _record("t2", "C", 30, 9)]
    controls = [_record("c1", "D", 29, 9), _record("c2", "E", 21, 0)]
    pairs = match_protein_controls(
        targets,
        controls,
        max_length_difference=3,
        max_normalized_position_difference=0.04,
    )
    assert {(row["target_protein_id"], row["control_protein_id"]) for row in pairs} == {
        ("t1", "c2"),
        ("t2", "c1"),
    }
    assert all(len(row["protein_pair_id"]) == 64 for row in pairs)
    with pytest.raises(ValueError, match="too small|no protein control|no complete"):
        match_protein_controls(
            targets,
            controls[:1],
            max_length_difference=1,
            max_normalized_position_difference=0.001,
        )


def test_feature_controls_are_same_layer_and_fail_outside_calipers() -> None:
    profiles = [
        {
            "model": "tiny",
            "layer": 0,
            "feature": 0,
            "feature_role": "target",
            "firing_frequency": 0.10,
            "input_norm": 2.0,
        },
        {
            "model": "tiny",
            "layer": 0,
            "feature": 1,
            "feature_role": "candidate_control",
            "firing_frequency": 0.11,
            "input_norm": 2.1,
        },
        {
            "model": "tiny",
            "layer": 0,
            "feature": 2,
            "feature_role": "candidate_control",
            "firing_frequency": 0.09,
            "input_norm": 1.9,
        },
    ]
    selected, matches = load_and_match_features(
        profiles,
        model_name="tiny",
        control_count=2,
        max_abs_log10_firing_ratio=0.2,
        max_abs_log_input_norm_ratio=0.2,
    )
    assert [row["feature"] for row in selected if row["feature_role"] == "control"] == [
        1,
        2,
    ]
    assert matches[0]["control_features"] == [1, 2]
    with pytest.raises(ValueError, match="only 0 feature controls"):
        load_and_match_features(
            profiles,
            model_name="tiny",
            control_count=1,
            max_abs_log10_firing_ratio=0.001,
            max_abs_log_input_norm_ratio=0.001,
        )


def test_exact_bos_factorial_and_same_pass_measurement_schema() -> None:
    protein_model = TinyProteinModel()
    clt = CLTForTraining(n_layers=1, d_model=2, d_clt=2, k=2, window=1)
    with torch.no_grad():
        clt.W_enc.zero_()
        clt.b_enc.zero_()
        clt.W_enc[0, 0] = torch.tensor([1.0, 0.0])
        clt.W_enc[0, 1] = torch.tensor([0.0, 1.0])
    natural = [
        _record("target", "A", 20, _internal_focal(20)),
        _record("control", "C", 20, _internal_focal(20)),
    ]
    variants = build_counterfactual_variants(natural)
    selected_features = [
        {
            "model": "tiny",
            "layer": 0,
            "feature": 0,
            "feature_role": "target",
            "firing_frequency": 0.1,
            "input_norm": 2.0,
            "feature_match_id": "f" * 64,
        },
        {
            "model": "tiny",
            "layer": 0,
            "feature": 1,
            "feature_role": "control",
            "firing_frequency": 0.11,
            "input_norm": 2.1,
            "feature_match_id": "f" * 64,
            "matched_target_feature": 0,
        },
    ]
    metadata = {
        "target": {
            "protein_pair_id": "a" * 64,
            "protein_match_role": "target",
            "matched_protein_id": "control",
            "protein_match_focal_position": _internal_focal(20),
            "protein_match_normalized_position": _internal_focal(20) / 19,
        },
        "control": {
            "protein_pair_id": "a" * 64,
            "protein_match_role": "control",
            "matched_protein_id": "target",
            "protein_match_focal_position": _internal_focal(20),
            "protein_match_normalized_position": _internal_focal(20) / 19,
        },
    }
    rows = extract_measurement_rows(
        protein_model=protein_model,
        clt=clt,
        model_name="tiny",
        model_revision="tiny-model-v1",
        tokenizer_revision="tiny-tokenizer-v1",
        input_format="sequence",
        bos_construction="explicit_prepend_tokenizer_bos",
        max_model_tokens=64,
        variants=variants,
        source_records={row["protein_id"]: row for row in natural},
        selected_features=selected_features,
        protein_metadata=metadata,
    )
    assert len(rows) == 2 * 4 * 2 * 2
    natural_target = next(
        row
        for row in rows
        if row["protein_id"] == "target"
        and row["condition"] == "natural_mxx"
        and row["bos_policy"] == "native"
        and row["feature"] == 0
    )
    assert natural_target["focal_token_index"] == 1
    assert natural_target["eligible_query_count"] == 20
    expected_raw = sum(1.0 / (query + 1) for query in range(1, 21))
    assert natural_target["received_attention_raw"] == pytest.approx(expected_raw)
    assert natural_target["normalized_received_attention"] == pytest.approx(
        expected_raw / 20
    )
    assert (
        natural_target["feature_measurement_timing"] == "same_unintervened_forward_pass"
    )
    assert natural_target["attention_key_mask_max_abs_strict_suffix"] == 0.0
    assert "not formal feature mediation" in natural_target[
        "attention_path_interpretation"
    ]
    assert natural_target["suffix_nll_increase_key_masked"] == pytest.approx(
        natural_target["key_masked_suffix_nll"]
        - natural_target["baseline_suffix_nll"]
    )
    assert natural_target["token_ids_sha256"]
    assert not protein_model.model.transformer.h[0].mlp._forward_hooks


@pytest.mark.parametrize(
    ("model_kwargs", "message"),
    [
        ({"nonfinite_activation": True}, "non-finite CLT-input capture"),
        ({"nonfinite_logits": True}, "finite causal-LM logits"),
    ],
)
def test_capture_rejects_nonfinite_activations_and_logits_before_use(
    model_kwargs, message
) -> None:
    protein_model = TinyProteinModel(**model_kwargs)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with pytest.raises((FloatingPointError, ValueError), match=message):
        capture_clt_inputs_and_attentions(
            protein_model,
            input_ids,
            torch.ones_like(input_ids),
            [0],
        )


def test_focal_position_contract_is_the_frozen_internal_site() -> None:
    length = 20
    valid = [_record("valid", "A", length, _internal_focal(length))]
    validate_focal_position_contract(valid, internal_fraction=0.55)
    with pytest.raises(ValueError, match="frozen internal insertion site"):
        validate_focal_position_contract(
            [_record("invalid", "C", length, 0)], internal_fraction=0.55
        )


def test_dictionary_loader_rejects_online_checkpoint_and_failed_receipt(
    tmp_path,
) -> None:
    failed_receipt = tmp_path / "failed_receipt.json"
    write_json(failed_receipt, {"status": "failed"})
    best_checkpoint = tmp_path / "best.pt"
    best_checkpoint.write_bytes(b"not-reached")
    hashes = {
        "run_manifest_sha256_by_seed": {
            "17": "1" * 64,
            "29": "2" * 64,
            "43": "3" * 64,
        },
        "checkpoint_sha256_by_seed": {
            "17": sha256_file(best_checkpoint),
            "29": "4" * 64,
            "43": "5" * 64,
        },
        "source_manifest_sha256_by_split": {
            "train": "6" * 64,
            "validation": "7" * 64,
            "test": "8" * 64,
        },
    }
    dictionary = {
        "run_seed": 17,
        "checkpoint_path": str(best_checkpoint),
        **hashes,
    }
    with pytest.raises(ValueError, match="eligibility receipt is incomplete"):
        _load_eligible_dictionary(
            {"path": str(failed_receipt), "sha256": sha256_file(failed_receipt)},
            dictionary,
            base=tmp_path,
            model_name="protgpt2",
            device="cpu",
            requested_layers=[0],
            dictionary_loader=None,
        )

    online_checkpoint = tmp_path / "clt.pt"
    online_checkpoint.write_bytes(b"online-checkpoint")
    dictionary["checkpoint_path"] = str(online_checkpoint)
    dictionary["checkpoint_sha256_by_seed"]["17"] = sha256_file(
        online_checkpoint
    )
    with pytest.raises(ValueError, match="exact-cache best.pt"):
        _load_eligible_dictionary(
            {"path": str(failed_receipt), "sha256": sha256_file(failed_receipt)},
            dictionary,
            base=tmp_path,
            model_name="protgpt2",
            device="cpu",
            requested_layers=[0],
            dictionary_loader=lambda *_args, **_kwargs: None,
        )


def test_tokenizer_native_mode_refuses_a_missing_leading_bos() -> None:
    tokenizer = TinyTokenizer()
    native = tokenize_bos_factorial(
        tokenizer,
        "MAAAAAAAAAAAAAAAAAA",
        construction="tokenizer_native_leading_bos",
        max_model_tokens=32,
    )
    assert native["native"].shape[1] == native["removed"].shape[1] + 1

    tokenizer.bos_token_id = None
    with pytest.raises(ValueError, match="required leading BOS"):
        tokenize_bos_factorial(
            tokenizer,
            "MAAAAAAAAAAAAAAAAAA",
            construction="tokenizer_native_leading_bos",
            max_model_tokens=32,
        )


def test_hash_bound_cpu_extractor_writes_analyzer_ready_artifacts(tmp_path) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text('{"model_type":"tiny"}\n')
    (model_root / "model.safetensors").write_bytes(b"tiny-weights")
    (model_root / "tokenizer.json").write_text('{"tiny":true}\n')
    model_artifacts = {
        "model_config_sha256": sha256_file(model_root / "config.json"),
        "model_weights_sha256": _tree_digest(
            model_root, [model_root / "model.safetensors"]
        ),
        "tokenizer_sha256": _tree_digest(model_root, [model_root / "tokenizer.json"]),
    }

    clt = CLTForTraining(n_layers=1, d_model=2, d_clt=2, k=2, window=1)
    checkpoint_dir = tmp_path / "seed17"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "best.pt"
    checkpoint.write_bytes(b"exact-cache-topk-fixture")
    receipt = tmp_path / "p0_2_eligibility_receipt.json"
    write_json(receipt, {"fixture": "loader_injection_still_hash_bound"})
    run_manifest_hashes = {17: "1" * 64, 29: "2" * 64, 43: "3" * 64}
    checkpoint_hashes = {
        17: sha256_file(checkpoint),
        29: "4" * 64,
        43: "5" * 64,
    }
    source_hashes = {
        "train": "6" * 64,
        "validation": "7" * 64,
        "test": "8" * 64,
    }

    target = _record("target", "A", 20, _internal_focal(20))
    control = _record("control", "C", 20, _internal_focal(20))
    discovery = _record("discovery", "D", 22, 0)
    paths = {}
    for name, rows in {
        "target": [target],
        "control": [control],
        "discovery": [discovery],
    }.items():
        paths[name] = tmp_path / f"{name}.jsonl"
        write_jsonl(paths[name], rows)
    profiles = [
        {
            "model": "tiny",
            "layer": 0,
            "feature": 0,
            "feature_role": "target",
            "firing_frequency": 0.1,
            "input_norm": 2.0,
        },
        {
            "model": "tiny",
            "layer": 0,
            "feature": 1,
            "feature_role": "candidate_control",
            "firing_frequency": 0.11,
            "input_norm": 2.1,
        },
    ]
    profile_path = tmp_path / "profiles.jsonl"
    write_jsonl(profile_path, profiles)
    spec = {
        "schema_version": 3,
        "code_revision": "cpu-contract-fixture",
        "device": "cpu",
        "max_model_tokens": 64,
        "model": {
            "name": "tiny",
            "model_root": str(model_root),
            "model_revision": "tiny-v1",
            "tokenizer_revision": "tiny-tokenizer-v1",
            "model_artifacts": model_artifacts,
            "input_format": "sequence",
            "model_inference_dtype": "bfloat16",
            "model_inference_dtype_verification": (
                "all_floating_model_parameters_exactly_declared_before_first_activation"
            ),
            "activation_finiteness_check": ACTIVATION_FINITE_CHECK,
            "bos_construction": "explicit_prepend_tokenizer_bos",
        },
        "p0_2_gate_receipt": {
            "path": str(receipt),
            "sha256": sha256_file(receipt),
        },
        "dictionary": {
            "run_seed": 17,
            "checkpoint_path": str(checkpoint),
            "run_manifest_sha256_by_seed": {
                str(seed): digest for seed, digest in run_manifest_hashes.items()
            },
            "checkpoint_sha256_by_seed": {
                str(seed): digest for seed, digest in checkpoint_hashes.items()
            },
            "source_manifest_sha256_by_split": source_hashes,
        },
        "cohorts": {
            "target": {
                "cohort_id": "target-v1",
                "path": str(paths["target"]),
                "sha256": sha256_file(paths["target"]),
            },
            "protein_control_pool": {
                "cohort_id": "controls-v1",
                "path": str(paths["control"]),
                "sha256": sha256_file(paths["control"]),
            },
            "discovery": {
                "cohort_id": "discovery-v1",
                "path": str(paths["discovery"]),
                "sha256": sha256_file(paths["discovery"]),
            },
        },
        "feature_profiles": {
            "path": str(profile_path),
            "sha256": sha256_file(profile_path),
            "discovery_cohort_sha256": sha256_file(paths["discovery"]),
        },
        "protein_matching": {
            "max_length_difference": 2,
            "max_normalized_position_difference": 0.01,
        },
        "feature_matching": {
            "control_count": 1,
            "max_abs_log10_firing_ratio": 0.2,
            "max_abs_log_input_norm_ratio": 0.2,
        },
        "counterfactual": {"internal_fraction": 0.55},
    }
    spec_path = tmp_path / "spec.json"
    write_json(spec_path, spec)
    output = tmp_path / "output"

    def fixture_dictionary_loader(
        receipt_path,
        receipt_sha,
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
        assert receipt_path == receipt.resolve()
        assert checkpoint_path == checkpoint.resolve()
        assert map_location == "cpu"
        assert requested_layers == [0]
        return clt, {
            "schema_version": "r2_p0_2_eligible_topk_load_v1",
            "model_name": model_name,
            "method": "topk_clt",
            "run_seed": run_seed,
            "eligibility_receipt_sha256": receipt_sha,
            "profile_sha256": "9" * 64,
            "protocol_sha256": "a" * 64,
            "run_manifest_sha256": expected_run_manifest_sha256_by_seed[run_seed],
            "checkpoint_sha256": expected_checkpoint_sha256_by_seed[run_seed],
            "candidate_id": "fixture-selected-topk",
            "checkpoint_step": 5_000,
            "geometry": {
                "n_layers": 1,
                "d_model": 2,
                "d_clt": 2,
                "k": 2,
                "window": 1,
            },
            "eligible_downstream_layers": [0],
            "source_manifest_sha256_by_split": (
                expected_source_manifest_sha256_by_split
            ),
        }

    with pytest.raises(ValueError, match="forbidden in production"):
        run_extractor(
            spec_path,
            sha256_file(spec_path),
            output,
            mode="production",
            model_loader=lambda *_args, **_kwargs: TinyProteinModel(),
            dictionary_loader=fixture_dictionary_loader,
        )
    assert not output.exists()

    failed_output = tmp_path / "failed_output"

    def failing_model_loader(*_args, **_kwargs):
        raise RuntimeError("deliberate fixture loader failure")

    with pytest.raises(RuntimeError, match="deliberate fixture"):
        run_extractor(
            spec_path,
            sha256_file(spec_path),
            failed_output,
            mode="test_fixture",
            model_loader=failing_model_loader,
            dictionary_loader=fixture_dictionary_loader,
        )
    assert not failed_output.exists()
    assert not list(tmp_path.glob(".failed_output.tmp-*"))

    wrong_dtype_output = tmp_path / "wrong_dtype_output"
    wrong_dtype_model = TinyProteinModel(parameter_dtype=torch.float32)
    with pytest.raises(ValueError, match="parameter dtype disagrees"):
        run_extractor(
            spec_path,
            sha256_file(spec_path),
            wrong_dtype_output,
            mode="test_fixture",
            model_loader=lambda *_args, **_kwargs: wrong_dtype_model,
            dictionary_loader=fixture_dictionary_loader,
        )
    assert wrong_dtype_model.model.forward_calls == 0
    assert not wrong_dtype_output.exists()

    manifest_path = run_extractor(
        spec_path,
        sha256_file(spec_path),
        output,
        mode="test_fixture",
        model_loader=lambda *_args, **_kwargs: TinyProteinModel(),
        dictionary_loader=fixture_dictionary_loader,
        command=["cpu-contract-test"],
    )
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads((output / "extraction_summary.json").read_text())
    measurements = [
        json.loads(line)
        for line in (output / "measurements.jsonl").read_text().splitlines()
    ]
    assert manifest["status"] == "verified_test_fixture_complete"
    assert manifest["execution_mode"] == "test_fixture"
    assert summary["execution_mode"] == "test_fixture"
    for payload in (manifest["model"], summary):
        assert payload["model_inference_dtype"] == "bfloat16"
        assert payload["observed_model_parameter_dtypes"] == ["bfloat16"]
        assert payload["model_inference_dtype_verified"] is True
        assert payload["activation_finiteness_check"] == ACTIVATION_FINITE_CHECK
        assert payload["activation_finiteness_verified"] is True
    assert manifest["inputs"]["feature_profiles"][
        "discovery_cohort_sha256"
    ] == sha256_file(paths["discovery"])
    assert summary["n_measurement_rows"] == 32
    assert len(measurements) == 32
    assert all(row["protein_pair_id"] for row in measurements)
    assert all(
        row["attention_key_mask_max_abs_strict_suffix"] == 0.0
        for row in measurements
    )
    assert manifest["eligible_dictionary"]["eligibility_receipt_sha256"] == (
        sha256_file(receipt)
    )
    assert manifest["eligible_dictionary"]["checkpoint_sha256"] == sha256_file(
        checkpoint
    )
    assert manifest["artifact_hashes"]["measurements.jsonl"] == sha256_file(
        output / "measurements.jsonl"
    )
