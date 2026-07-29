from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

R2 = Path(__file__).resolve().parents[1]
SCRIPT = R2 / "scripts/61_build_dictionary_activation_cache.py"
SPEC = importlib.util.spec_from_file_location("dictionary_cache_builder", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
LOCAL_CODE_IMPORTS_DEFERRED = MODULE._LOCAL_CODE_LOADED is False
MODULE._load_verified_local_code()

from src.revision.dictionary_controls import (  # noqa: E402
    CachedMultiLayerRows,
    load_activation_cache,
)
from src.models.model_loader import (  # noqa: E402
    assert_finite_captured_activations,
    verify_frozen_model_inference_dtype,
)
from src.revision.io import sha256_file, write_json  # noqa: E402


def test_runner_defers_local_imports_until_after_code_verification(
    tmp_path, monkeypatch
):
    assert LOCAL_CODE_IMPORTS_DEFERRED
    spec = importlib.util.spec_from_file_location("unverified_cache_builder", SCRIPT)
    unverified = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(unverified)
    arguments, _, manifest = code_binding_fixture(tmp_path)
    monkeypatch.setattr(unverified, "R2_ROOT", manifest.parent)
    arguments["archive_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="archive SHA-256 mismatch"):
        unverified.verify_code_binding(**arguments)
    assert unverified._LOCAL_CODE_LOADED is False


class TinyTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    padding_side = "right"
    truncation_side = "right"
    special_tokens_map = {"pad_token": "<pad>", "bos_token": "<bos>"}

    def __call__(
        self,
        texts,
        *,
        add_special_tokens,
        padding,
        truncation,
        max_length,
        return_attention_mask,
        return_tensors=None,
    ):
        rows = []
        for text in texts:
            values = [4 + ord(character) % 13 for character in text]
            if add_special_tokens:
                values = [self.bos_token_id, *values, self.eos_token_id]
            rows.append(values[:max_length] if truncation else values)
        width = max(map(len, rows)) if padding else None
        ids = (
            [row + [self.pad_token_id] * (width - len(row)) for row in rows]
            if padding
            else rows
        )
        masks = [[int(value != self.pad_token_id) for value in row] for row in ids]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": masks}


class TinyProteinModel:
    n_layers = 2
    d_model = 2
    device = "cpu"

    def __init__(self):
        self.tokenizer = TinyTokenizer()
        self.model = torch.nn.Linear(1, 1, bias=False).to(torch.bfloat16)

    def get_activations(self, input_ids, attention_mask):
        base = torch.stack((input_ids.float(), input_ids.float().square()), dim=-1)
        return SimpleNamespace(
            clt_input=[base + layer for layer in range(self.n_layers)],
            mlp_out=[0.5 * base + layer for layer in range(self.n_layers)],
        )


def code_binding_fixture(root: Path):
    archive = root / "code.tar"
    archive.parent.mkdir(parents=True, exist_ok=True)
    deployed = root / "deployed"
    (deployed / "scripts").mkdir(parents=True)
    runner = deployed / "scripts/runner.py"
    runner.write_text("print('frozen')\n")
    manifest = deployed / "CODE_CONTENT_SHA256SUMS"
    manifest.write_text(f"{sha256_file(runner)}  ./scripts/runner.py\n")
    with tarfile.open(archive, mode="w") as handle:
        handle.add(deployed, arcname="r2_interpretability_transfer")
    arguments = {
        "archive_path": archive,
        "archive_sha256": sha256_file(archive),
        "content_manifest_path": manifest,
        "content_manifest_sha256": sha256_file(manifest),
    }
    return arguments, runner, manifest


def rewrite_code_archive(archive: Path, files: dict[str, bytes], *, link=None) -> None:
    root = "r2_interpretability_transfer"
    with tarfile.open(archive, mode="w") as handle:
        directory = tarfile.TarInfo(root)
        directory.type = tarfile.DIRTYPE
        handle.addfile(directory)
        for name, payload in files.items():
            member = tarfile.TarInfo(f"{root}/{name}")
            member.size = len(payload)
            handle.addfile(member, io.BytesIO(payload))
        if link is not None:
            member = tarfile.TarInfo(f"{root}/{link}")
            member.type = tarfile.SYMTYPE
            member.linkname = "scripts/runner.py"
            handle.addfile(member)


def test_exact_code_binding_verifies_archive_manifest_content_and_inventory(
    tmp_path, monkeypatch
):
    arguments, _, _ = code_binding_fixture(tmp_path)
    monkeypatch.setattr(MODULE, "R2_ROOT", arguments["content_manifest_path"].parent)
    assert MODULE.verify_code_binding(**arguments) == {
        "code_archive_sha256": arguments["archive_sha256"],
        "code_content_manifest_sha256": arguments[
            "content_manifest_sha256"
        ],
        "code_content_inventory_verified": True,
    }


def test_exact_code_binding_rejects_archive_and_manifest_tamper(
    tmp_path, monkeypatch
):
    arguments, _, manifest = code_binding_fixture(tmp_path / "archive")
    monkeypatch.setattr(MODULE, "R2_ROOT", manifest.parent)
    arguments["archive_path"].write_bytes(b"tampered archive")
    with pytest.raises(ValueError, match="archive SHA-256 mismatch"):
        MODULE.verify_code_binding(**arguments)

    arguments, _, manifest = code_binding_fixture(tmp_path / "manifest")
    monkeypatch.setattr(MODULE, "R2_ROOT", manifest.parent)
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        MODULE.verify_code_binding(**arguments)


def test_exact_code_binding_rejects_content_inventory_symlink_and_escape(
    tmp_path, monkeypatch
):
    arguments, runner, _ = code_binding_fixture(tmp_path / "content")
    monkeypatch.setattr(MODULE, "R2_ROOT", runner.parents[1])
    runner.write_text("print('tampered')\n")
    with pytest.raises(ValueError, match="code content mismatch"):
        MODULE.verify_code_binding(**arguments)

    arguments, _, manifest = code_binding_fixture(tmp_path / "inventory")
    monkeypatch.setattr(MODULE, "R2_ROOT", manifest.parent)
    (manifest.parent / "unlisted.py").write_text("unlisted\n")
    with pytest.raises(ValueError, match="inventory differs"):
        MODULE.verify_code_binding(**arguments)

    arguments, _, manifest = code_binding_fixture(tmp_path / "symlink")
    monkeypatch.setattr(MODULE, "R2_ROOT", manifest.parent)
    (manifest.parent / "alias.py").symlink_to("scripts/runner.py")
    with pytest.raises(ValueError, match="symbolic link"):
        MODULE.verify_code_binding(**arguments)

    arguments, _, manifest = code_binding_fixture(tmp_path / "escape")
    monkeypatch.setattr(MODULE, "R2_ROOT", manifest.parent)
    manifest.write_text(f"{'0' * 64}  ./../outside.py\n")
    arguments["content_manifest_sha256"] = sha256_file(manifest)
    with pytest.raises(ValueError, match="unsafe code manifest path"):
        MODULE.verify_code_binding(**arguments)


def test_exact_code_binding_rejects_unrelated_manifest_and_archive_tree_tamper(
    tmp_path, monkeypatch
):
    arguments, runner, manifest = code_binding_fixture(tmp_path / "unrelated")
    monkeypatch.setattr(MODULE, "R2_ROOT", manifest.parent)
    other = tmp_path / "other" / "CODE_CONTENT_SHA256SUMS"
    other.parent.mkdir()
    other.write_bytes(manifest.read_bytes())
    with pytest.raises(ValueError, match="runner project manifest"):
        MODULE.verify_code_binding(
            **{
                **arguments,
                "content_manifest_path": other,
                "content_manifest_sha256": sha256_file(other),
            }
        )

    files = {
        "scripts/runner.py": runner.read_bytes(),
        manifest.name: b"different embedded manifest\n",
    }
    rewrite_code_archive(arguments["archive_path"], files)
    arguments["archive_sha256"] = sha256_file(arguments["archive_path"])
    with pytest.raises(ValueError, match="archive manifest differs"):
        MODULE.verify_code_binding(**arguments)

    files[manifest.name] = manifest.read_bytes()
    files["scripts/runner.py"] = b"tampered archived runner\n"
    rewrite_code_archive(arguments["archive_path"], files)
    arguments["archive_sha256"] = sha256_file(arguments["archive_path"])
    with pytest.raises(ValueError, match="archive content mismatch"):
        MODULE.verify_code_binding(**arguments)

    files["scripts/runner.py"] = runner.read_bytes()
    files["unlisted.py"] = b"unlisted\n"
    rewrite_code_archive(arguments["archive_path"], files)
    arguments["archive_sha256"] = sha256_file(arguments["archive_path"])
    with pytest.raises(ValueError, match="archive inventory differs"):
        MODULE.verify_code_binding(**arguments)

    del files["unlisted.py"]
    rewrite_code_archive(arguments["archive_path"], files, link="alias.py")
    arguments["archive_sha256"] = sha256_file(arguments["archive_path"])
    with pytest.raises(ValueError, match="archive contains a link"):
        MODULE.verify_code_binding(**arguments)

    files["../escape.py"] = b"escape\n"
    rewrite_code_archive(arguments["archive_path"], files)
    arguments["archive_sha256"] = sha256_file(arguments["archive_path"])
    with pytest.raises(ValueError, match="unsafe or duplicate"):
        MODULE.verify_code_binding(**arguments)


def test_deployed_model_artifact_hashes_are_verified(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"fixture"}\n')
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "tokenizer.json").write_text('{"fixture":true}\n')
    expected = {
        "model_config_sha256": sha256_file(root / "config.json"),
        "model_weights_sha256": MODULE._tree_digest(
            root, [root / "model.safetensors"]
        ),
        "tokenizer_sha256": MODULE._tree_digest(root, [root / "tokenizer.json"]),
    }
    verified = MODULE.verify_model_artifacts(root, expected)
    assert verified.items() >= expected.items()
    with pytest.raises(ValueError, match="model_weights_sha256"):
        MODULE.verify_model_artifacts(
            root,
            {**expected, "model_weights_sha256": "0" * 64},
        )


def test_panel_capacity_gate_counts_all_incomplete_caches(tmp_path, monkeypatch):
    profile_path = R2 / "configs/p0_2_dictionary_controls_production_profile.json"
    profile_sha256 = sha256_file(profile_path)
    profile = MODULE.load_production_profile(profile_path, profile_sha256)
    calls = []

    def fake_gate(destination, *, estimated_bytes, safety_factor):
        calls.append((destination, estimated_bytes, safety_factor))
        return {
            "estimated_bytes": estimated_bytes,
            "safety_factor": safety_factor,
            "required_free_bytes": estimated_bytes,
            "observed_free_bytes": estimated_bytes,
        }

    monkeypatch.setattr(MODULE, "require_cache_free_space", fake_gate)
    root = tmp_path / "panel"
    code_binding = {
        "code_archive_sha256": "1" * 64,
        "code_content_manifest_sha256": "2" * 64,
        "code_content_inventory_verified": True,
    }
    first = MODULE.enforce_panel_cache_capacity(
        panel_cache_root=root,
        output_dir=root / "protgpt2",
        profile=profile,
        profile_sha256=profile_sha256,
        model_name="protgpt2",
        code_binding=code_binding,
    )
    assert first["estimated_bytes"] == 641_433_600_000

    completed_dir = root / "protgpt2"
    completed_dir.mkdir(parents=True)
    manifest = completed_dir / "manifest.json"
    write_json(
        manifest,
        {
            "content_sha256": "a" * 64,
            "activation_provenance": {
                "schema_version": "r2_dictionary_activation_provenance_v3",
                **code_binding,
            },
        },
    )
    execution_report = completed_dir / "cache_execution_report.json"
    write_json(
        execution_report,
        {
            "schema_version": "r2_dictionary_cache_execution_report_v3",
            "status": "verified_complete",
            "model_name": "protgpt2",
            "profile_sha256": profile_sha256,
            "cache_manifest_sha256": sha256_file(manifest),
            "cache_content_sha256": "a" * 64,
            **code_binding,
        },
    )
    receipt = {
        "schema_version": "r2_dictionary_cache_completion_receipt_v3",
        "status": "verified_complete",
        "model_name": "protgpt2",
        "profile_sha256": profile_sha256,
        "cache_manifest_path": str(manifest.resolve()),
        "cache_manifest_sha256": sha256_file(manifest),
        "cache_content_sha256": "a" * 64,
        "activation_payload_bytes": 221_184_000_000,
        "model_inference_dtype": "bfloat16",
        "observed_model_parameter_dtypes": ["bfloat16"],
        "model_inference_dtype_verification": (
            "all_floating_model_parameters_exactly_declared_before_first_activation"
        ),
        "model_inference_dtype_verified": True,
        "activation_finiteness_check": (
            "all_clt_input_and_mlp_output_tensors_before_storage_conversion_and_write"
        ),
        "cache_storage_dtype": "float16",
        **code_binding,
        "execution_report_path": str(execution_report.resolve()),
        "execution_report_sha256": sha256_file(execution_report),
        "completed_at_utc": "2026-07-17T00:00:00+00:00",
    }
    write_json(
        completed_dir / "completion_receipt.json",
        receipt,
    )
    second = MODULE.enforce_panel_cache_capacity(
        panel_cache_root=root,
        output_dir=root / "zymctrl",
        profile=profile,
        profile_sha256=profile_sha256,
        model_name="zymctrl",
        code_binding=code_binding,
    )
    assert second["completed_models"] == ["protgpt2"]
    assert second["estimated_bytes"] == 420_249_600_000
    assert len(calls) == 2

    for field, replacement in (
        ("code_archive_sha256", "3" * 64),
        ("code_content_manifest_sha256", "4" * 64),
        ("code_content_inventory_verified", False),
    ):
        write_json(
            completed_dir / "completion_receipt.json",
            {**receipt, field: replacement},
        )
        with pytest.raises(ValueError, match="invalid panel completion receipt"):
            MODULE.enforce_panel_cache_capacity(
                panel_cache_root=root,
                output_dir=root / "zymctrl",
                profile=profile,
                profile_sha256=profile_sha256,
                model_name="zymctrl",
                code_binding=code_binding,
            )


def test_completion_receipt_records_verified_code_binding(tmp_path):
    output = tmp_path / "cache"
    output.mkdir()
    manifest = output / "manifest.json"
    write_json(manifest, {"content_sha256": "a" * 64})
    code_binding = {
        "code_archive_sha256": "1" * 64,
        "code_content_manifest_sha256": "2" * 64,
        "code_content_inventory_verified": True,
    }
    report = output / "cache_execution_report.json"
    write_json(
        report,
        {
            "schema_version": "r2_dictionary_cache_execution_report_v3",
            **code_binding,
        },
    )
    cache = SimpleNamespace(
        manifest_path=manifest,
        manifest_sha256=sha256_file(manifest),
        content_sha256="a" * 64,
        payload={
            "storage_dtype": "float16",
            "activation_provenance": {
                "schema_version": "r2_dictionary_activation_provenance_v3",
                "model_inference_dtype": "bfloat16",
                "observed_model_parameter_dtypes": ["bfloat16"],
                "model_inference_dtype_verification": "verified",
                "model_inference_dtype_verified": True,
                "activation_finiteness_check": "finite",
                **code_binding,
            },
        },
    )
    receipt_path = MODULE.write_completion_receipt(
        cache=cache,
        profile={
            "cache_extraction": {
                "panel_completion_receipt_schema": (
                    "r2_dictionary_cache_completion_receipt_v3"
                ),
                "estimated_activation_payload_bytes_by_model": {"tiny": 1},
            }
        },
        profile_sha256="3" * 64,
        model_name="tiny",
        execution_report_path=report,
    )
    receipt = MODULE.load_strict_json(receipt_path)
    assert all(receipt[field] == value for field, value in code_binding.items())
    assert receipt["schema_version"] == "r2_dictionary_cache_completion_receipt_v3"

    report_payload = MODULE.load_strict_json(report)
    del report_payload["code_content_manifest_sha256"]
    write_json(report, report_payload)
    with pytest.raises(ValueError, match="invalid code binding"):
        MODULE.write_completion_receipt(
            cache=cache,
            profile={
                "cache_extraction": {
                    "panel_completion_receipt_schema": (
                        "r2_dictionary_cache_completion_receipt_v3"
                    ),
                    "estimated_activation_payload_bytes_by_model": {"tiny": 1},
                }
            },
            profile_sha256="3" * 64,
            model_name="tiny",
            execution_report_path=report,
        )


def test_preflight_profile_is_bounded_and_nonconfirmatory(tmp_path):
    profile_path = R2 / "configs/p0_2_dictionary_controls_production_profile.json"
    profile = MODULE.load_production_profile(profile_path, sha256_file(profile_path))
    output = tmp_path / "preflight"
    output.mkdir()
    derived, digest = MODULE.build_preflight_profile(profile, output_dir=output)
    assert derived["confirmatory"] is False
    assert derived["cache_extraction"]["valid_token_budget_by_split"] == {
        "train": 2,
        "validation": 2,
        "test": 2,
    }
    assert derived["cache_extraction"][
        "estimated_activation_payload_bytes_by_model"
    ]["progen2-medium"] == 995_328
    assert sha256_file(output / "executed_preflight_profile.json") == digest


def test_two_pass_budgeted_extraction_smoke(tmp_path, monkeypatch):
    source_splits = {}
    sequences = {
        "train": ("MA", "TR"),
        "validation": ("VV", "AA"),
        "test": ("GG", "PP"),
    }
    for split, values in sequences.items():
        path = tmp_path / f"{split}.jsonl"
        records = []
        for index, sequence in enumerate(values):
            records.append(
                {
                    "id": f"{split}-{index}",
                    "source": "fixture",
                    "sequence": sequence,
                    "split": split,
                    "family": f"f{index}",
                    "sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                }
            )
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        source_splits[split] = {
            "manifest_path": str(path),
            "manifest_sha256": sha256_file(path),
        }

    budgets = {"train": 3, "validation": 2, "test": 2}
    profile = {
        "models": ["tiny"],
        "estimand": {
            "feature_input_hook": "hook_resid_mid",
            "feature_output_hook": "hook_mlp_out",
            "decoder_window": 2,
        },
        "cache_extraction": {
            "model_cache_geometry": {
                "tiny": {"n_layers": 2, "input_dim": 2, "target_dim": 2}
            },
            "tokenization": {
                "add_special_tokens": True,
                "padding": "longest",
                "padding_side": "right",
                "truncation_side": "right",
                "truncation": True,
                "max_model_tokens": 8,
            },
            "model_input_format_by_model": {"tiny": "sequence"},
            "first_pass_tokenizer_batch_size": 2,
            "model_forward_sequence_batch_size": 2,
            "model_inference_dtype": "bfloat16",
            "model_inference_dtype_verification": (
                "all_floating_model_parameters_exactly_declared_before_first_activation"
            ),
            "activation_finiteness_check": (
                "all_clt_input_and_mlp_output_tensors_before_storage_conversion_and_write"
            ),
            "valid_token_budget_by_split": budgets,
            "selection_method": (
                "lowest_sha256_record_digest_colon_unpadded_token_position"
            ),
            "storage_dtype": "float16",
            "estimated_activation_payload_bytes_by_model": {"tiny": 112},
            "free_space_safety_factor": 1.0,
        },
    }
    checks = []
    original_check = MODULE.assert_finite_captured_activations

    def counted_check(cache):
        checks.append(True)
        original_check(cache)

    monkeypatch.setattr(MODULE, "assert_finite_captured_activations", counted_check)
    manifest = MODULE.build_cache_from_model(
        protein_model=TinyProteinModel(),
        model_name="tiny",
        profile=profile,
        profile_sha256="a" * 64,
        source_splits=source_splits,
        output_dir=tmp_path / "cache",
        model_provenance={
            "model_revision": "fixture",
            "model_config_sha256": "b" * 64,
            "model_weights_sha256": "c" * 64,
            "tokenizer_sha256": "d" * 64,
        },
        execution_provenance={
            "scope": "cpu_test",
            "code_archive_sha256": "e" * 64,
            "code_content_manifest_sha256": "f" * 64,
            "code_content_inventory_verified": True,
        },
    )
    cache = load_activation_cache(manifest)
    assert cache.payload["layout"] == "preallocated_single_file_per_layer_split"
    assert len(cache.shards) == 6
    provenance = cache.payload["activation_provenance"]
    assert provenance["schema_version"] == "r2_dictionary_activation_provenance_v3"
    assert provenance["code_archive_sha256"] == "e" * 64
    assert provenance["code_content_manifest_sha256"] == "f" * 64
    assert provenance["code_content_inventory_verified"] is True
    assert (
        provenance["first_pass_tokenization_sha256_by_split"]
        == provenance["second_pass_tokenization_sha256_by_split"]
    )
    assert CachedMultiLayerRows(cache, "train").n_rows == 3
    assert len(checks) == 3


def test_bfloat16_inference_dtype_is_verified_and_nonfinite_activation_fails():
    model = TinyProteinModel()
    assert verify_frozen_model_inference_dtype(model, "bfloat16")[
        "model_inference_dtype_verified"
    ] is True
    model.model = torch.nn.Linear(1, 1, bias=False).float()
    with pytest.raises(ValueError, match="parameter dtype disagrees"):
        verify_frozen_model_inference_dtype(model, "bfloat16")

    cache = SimpleNamespace(
        clt_input=[torch.tensor([[[1.0], [float("nan")]]])],
        mlp_out=[torch.ones(1, 2, 1)],
    )
    with pytest.raises(FloatingPointError, match="CLT-input.*layer 0"):
        assert_finite_captured_activations(cache)


def test_production_profile_rejects_non_bfloat16_inference(tmp_path):
    source = R2 / "configs/p0_2_dictionary_controls_production_profile.json"
    payload = json.loads(source.read_text())
    payload["cache_extraction"]["model_inference_dtype"] = "float16"
    amended = tmp_path / "bad-profile.json"
    amended.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="activation-cache contract changed"):
        MODULE.load_production_profile(amended, sha256_file(amended))
