"""Execute a hash-bound corrected-steering generation plan without discovery.

This module is deliberately limited to generation.  It verifies the immutable
freeze produced by ``scripts/60_prepare_corrected_steering.py``, verifies the
deployed model and CLT artifacts, executes every plan row, and publishes a
complete output directory atomically.  Endpoint scoring and scientific gate
adjudication remain separate operations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import shutil
import socket
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from ..models.model_loader import (
    INFERENCE_DTYPE_VERIFICATION,
    inference_dtype,
    verify_frozen_model_inference_dtype,
)
from .input_builder import load_json, load_jsonl, verify_model_artifacts
from .io import sha256_file, write_json, write_jsonl
from .steering_protocol import (
    validate_completed_generations,
    validate_plan_rows,
    validate_provenance,
)


PROJECT = Path(__file__).resolve().parents[2]
FREEZE_FILES = {
    "generation_plan.jsonl",
    "feature_decisions.jsonl",
    "control_decisions.json",
    "frozen_endpoint_specs.json",
    "frozen_provenance.json",
    "frozen_analysis_spec.json",
    "summary.json",
    "run_manifest.json",
}
FROZEN_ARTIFACTS = FREEZE_FILES - {"summary.json", "run_manifest.json"}
SUPPORTED_SITES = {"clt_input", "mlp_output"}
SUPPORTED_MULTIPLIER_SEMANTICS = "additive_decoder_direction_v1"
VALID_ARMS = {
    "prompt_only",
    "target",
    "random_feature",
    "norm_matched_feature",
}
HEX = frozenset("0123456789abcdef")
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
ACTIVATION_FINITE_CHECK = (
    "all_required_layer_captured_activation_and_logit_tensors_before_"
    "downstream_conversion_or_use"
)


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= HEX:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_exact_fields(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        observed = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{label} fields differ: missing={sorted(fields - observed)}, "
            f"extra={sorted(observed - fields)}"
        )
    return dict(value)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _protocol_source_hashes() -> dict[str, str]:
    return {
        "60_prepare_corrected_steering.py": sha256_file(
            PROJECT / "scripts/60_prepare_corrected_steering.py"
        ),
        "steering_protocol.py": sha256_file(
            PROJECT / "src/revision/steering_protocol.py"
        ),
        "statistics.py": sha256_file(PROJECT / "src/revision/statistics.py"),
        "io.py": sha256_file(PROJECT / "src/revision/io.py"),
    }


def _execution_source_hashes() -> dict[str, str]:
    return {
        "steering_execution.py": sha256_file(Path(__file__)),
        "63_execute_corrected_steering.py": sha256_file(
            PROJECT / "scripts/63_execute_corrected_steering.py"
        ),
        **_protocol_source_hashes(),
        "model_loader.py": sha256_file(PROJECT / "src/models/model_loader.py"),
        "clt_trainer.py": sha256_file(PROJECT / "src/training/clt_trainer.py"),
        "dictionary_gate.py": sha256_file(
            PROJECT / "src/revision/dictionary_gate.py"
        ),
    }


def load_verified_freeze(
    frozen_dir: Path, expected_manifest_sha256: str
) -> dict[str, Any]:
    """Load a real script-60 freeze after verifying every content binding."""

    frozen_dir = Path(frozen_dir).resolve()
    expected_manifest = _require_sha256(
        expected_manifest_sha256, "freeze manifest SHA-256"
    )
    if not frozen_dir.is_dir():
        raise FileNotFoundError(f"missing frozen protocol directory: {frozen_dir}")
    entries = {path.name for path in frozen_dir.iterdir()}
    if entries != FREEZE_FILES:
        raise ValueError(
            "frozen protocol file inventory differs: "
            f"missing={sorted(FREEZE_FILES - entries)}, "
            f"extra={sorted(entries - FREEZE_FILES)}"
        )
    manifest_path = frozen_dir / "run_manifest.json"
    if sha256_file(manifest_path) != expected_manifest:
        raise ValueError("freeze run-manifest SHA-256 mismatch")

    manifest = load_json(manifest_path)
    summary = load_json(frozen_dir / "summary.json")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != "r2-corrected-steering-freeze-manifest-v3"
        or manifest.get("stage") != "freeze"
    ):
        raise ValueError("unsupported or malformed corrected-steering freeze manifest")
    if summary.get("status") != "generation_plan_frozen":
        raise ValueError("executor refuses synthetic or non-generation-ready freezes")
    if manifest.get("source_hashes") != _protocol_source_hashes():
        raise ValueError("current protocol source hashes differ from the immutable freeze")

    artifact_hashes = summary.get("frozen_artifact_hashes")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != FROZEN_ARTIFACTS:
        raise ValueError("frozen summary has an invalid artifact inventory")
    for name, expected in artifact_hashes.items():
        expected = _require_sha256(expected, f"frozen artifact {name} SHA-256")
        if sha256_file(frozen_dir / name) != expected:
            raise ValueError(f"frozen artifact SHA-256 mismatch: {name}")

    expected_freeze_id = _canonical_sha256(
        {
            "content_binding_sha256": summary.get("content_binding_sha256"),
            "artifact_hashes": artifact_hashes,
        }
    )
    freeze_id = _require_sha256(summary.get("freeze_id"), "freeze_id")
    if freeze_id != expected_freeze_id or manifest.get("freeze_id") != freeze_id:
        raise ValueError("freeze_id does not bind the immutable artifact set")
    expected_manifest_artifacts = {
        **artifact_hashes,
        "summary.json": sha256_file(frozen_dir / "summary.json"),
    }
    if manifest.get("artifact_hashes") != expected_manifest_artifacts:
        raise ValueError("freeze manifest artifact hashes disagree with the frozen summary")

    rows = validate_plan_rows(load_jsonl(frozen_dir / "generation_plan.jsonl"))
    binding = _require_sha256(
        summary.get("content_binding_sha256"), "content_binding_sha256"
    )
    if {row.get("content_binding_sha256") for row in rows} != {binding}:
        raise ValueError("plan rows do not bind the frozen protocol context")
    provenance = validate_provenance(load_json(frozen_dir / "frozen_provenance.json"))
    identities = {
        "generator_revision": {str(row["generator_revision"]) for row in rows},
        "model_revision": {str(row["model_revision"]) for row in rows},
        "tokenizer_revision": {str(row["tokenizer_revision"]) for row in rows},
        "clt_checkpoint_sha256": {
            str(row["clt_checkpoint_sha256"]) for row in rows
        },
    }
    if any(len(values) != 1 for values in identities.values()):
        raise ValueError("frozen plan mixes generator/model/tokenizer/CLT revisions")
    expected_identities = {
        "generator_revision": provenance["code_revision"],
        "model_revision": provenance["model_revision"],
        "tokenizer_revision": provenance["tokenizer_revision"],
        "clt_checkpoint_sha256": provenance["clt_checkpoint_sha256"],
    }
    if any(next(iter(identities[field])) != expected for field, expected in expected_identities.items()):
        raise ValueError("frozen plan identities disagree with frozen provenance")
    return {
        "directory": frozen_dir,
        "manifest": manifest,
        "manifest_sha256": expected_manifest,
        "summary": summary,
        "rows": rows,
        "provenance": provenance,
        "artifact_hashes": artifact_hashes,
    }


def load_execution_spec(spec_path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load a deployment declaration bound to one eligible exact-cache TopK."""

    spec_path = Path(spec_path).resolve()
    expected = _require_sha256(expected_sha256, "execution spec SHA-256")
    if sha256_file(spec_path) != expected:
        raise ValueError("execution spec SHA-256 mismatch")
    spec = _require_exact_fields(
        load_json(spec_path),
        {"schema_version", "device", "model", "p0_2"},
        "execution spec",
    )
    if spec["schema_version"] != 3:
        raise ValueError("unsupported steering execution spec schema_version")
    if not isinstance(spec["device"], str) or not spec["device"].strip():
        raise ValueError("execution device must be a non-empty string")
    model = _require_exact_fields(
        spec["model"],
        {
            "name",
            "model_root",
            "model_revision",
            "tokenizer_revision",
            "model_artifacts",
            "model_inference_dtype",
            "model_inference_dtype_verification",
            "activation_finiteness_check",
        },
        "execution model",
    )
    for field in ("name", "model_root", "model_revision", "tokenizer_revision"):
        if not isinstance(model[field], str) or not model[field].strip():
            raise ValueError(f"execution model {field} must be non-empty")
    if (
        model["model_inference_dtype"] != "bfloat16"
        or model["model_inference_dtype_verification"]
        != INFERENCE_DTYPE_VERIFICATION
        or model["activation_finiteness_check"] != ACTIVATION_FINITE_CHECK
    ):
        raise ValueError(
            "production steering requires the frozen bfloat16 inference and "
            "activation-finiteness contract"
        )
    _require_exact_fields(
        model["model_artifacts"],
        {"model_config_sha256", "model_weights_sha256", "tokenizer_sha256"},
        "model_artifacts",
    )
    p0_2 = _require_exact_fields(
        spec["p0_2"],
        {
            "eligibility_receipt",
            "method",
            "run_seed",
            "checkpoint_path",
            "run_manifest_sha256_by_seed",
            "checkpoint_sha256_by_seed",
            "source_manifest_sha256_by_split",
            "requested_layers",
        },
        "P0-2 execution binding",
    )
    receipt = _require_exact_fields(
        p0_2["eligibility_receipt"], {"path", "sha256"}, "P0-2 eligibility receipt"
    )
    if not isinstance(receipt["path"], str) or not receipt["path"].strip():
        raise ValueError("P0-2 eligibility-receipt path must be non-empty")
    _require_sha256(receipt["sha256"], "P0-2 eligibility-receipt SHA-256")
    if p0_2["method"] != "topk_clt":
        raise ValueError("steering execution requires the eligible TopK CLT method")
    if p0_2["run_seed"] not in {17, 29, 43}:
        raise ValueError("P0-2 run_seed must be one of 17, 29 or 43")
    if not isinstance(p0_2["checkpoint_path"], str) or not p0_2[
        "checkpoint_path"
    ].strip():
        raise ValueError("exact-cache best.pt path must be non-empty")
    if Path(p0_2["checkpoint_path"]).name != "best.pt":
        raise ValueError("steering execution rejects online clt.pt; exact-cache best.pt is required")

    def seed_hashes(value: Any, label: str) -> dict[str, str]:
        result = _require_exact_fields(value, {"17", "29", "43"}, label)
        for seed, digest in result.items():
            _require_sha256(digest, f"{label} seed {seed}")
        return result

    p0_2["run_manifest_sha256_by_seed"] = seed_hashes(
        p0_2["run_manifest_sha256_by_seed"], "P0-2 run-manifest hashes"
    )
    p0_2["checkpoint_sha256_by_seed"] = seed_hashes(
        p0_2["checkpoint_sha256_by_seed"], "P0-2 checkpoint hashes"
    )
    source_hashes = _require_exact_fields(
        p0_2["source_manifest_sha256_by_split"],
        {"train", "validation", "test"},
        "P0-2 source-manifest hashes",
    )
    for split, digest in source_hashes.items():
        _require_sha256(digest, f"P0-2 {split} source-manifest SHA-256")
    layers = p0_2["requested_layers"]
    if (
        not isinstance(layers, list)
        or not layers
        or any(type(layer) is not int or layer < 0 for layer in layers)
        or layers != sorted(set(layers))
    ):
        raise ValueError("P0-2 requested_layers must be sorted unique indices")
    p0_2["eligibility_receipt"] = receipt
    spec["model"] = model
    spec["p0_2"] = p0_2
    spec["path"] = spec_path
    spec["sha256"] = expected
    return spec


def _resolve(path: str, base: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def _require_numerical_integrity(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = {
        field: value.get(field)
        for field in (
            "model_inference_dtype",
            "observed_model_parameter_dtypes",
            "model_inference_dtype_verification",
            "model_inference_dtype_verified",
            "activation_finiteness_check",
            "activation_finiteness_verified",
        )
    }
    if receipt != {
        "model_inference_dtype": "bfloat16",
        "observed_model_parameter_dtypes": ["bfloat16"],
        "model_inference_dtype_verification": INFERENCE_DTYPE_VERIFICATION,
        "model_inference_dtype_verified": True,
        "activation_finiteness_check": ACTIVATION_FINITE_CHECK,
        "activation_finiteness_verified": True,
    }:
        raise ValueError("receipt lacks verified bfloat16 numerical integrity")
    return receipt


def _verify_execution_identity(freeze: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    provenance = freeze["provenance"]
    model = spec["model"]
    p0_2 = spec["p0_2"]
    if model["model_revision"] != provenance["model_revision"]:
        raise ValueError("execution model revision differs from the frozen plan")
    if model["tokenizer_revision"] != provenance["tokenizer_revision"]:
        raise ValueError("execution tokenizer revision differs from the frozen plan")
    selected_checkpoint = p0_2["checkpoint_sha256_by_seed"][
        str(p0_2["run_seed"])
    ]
    if selected_checkpoint != provenance["clt_checkpoint_sha256"]:
        raise ValueError("execution CLT SHA-256 differs from the frozen plan")
    used_layers = sorted(
        {
            int(intervention["layer"])
            for row in freeze["rows"]
            for intervention in row["interventions"]
        }
    )
    if used_layers != p0_2["requested_layers"]:
        raise ValueError(
            "P0-2 requested layers differ from the frozen intervention layers"
        )


def load_execution_resources(
    freeze: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    model_loader: Callable | None = None,
    dictionary_loader: Callable | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Verify and load the model and P0-2-authorized exact-cache TopK CLT."""

    from src.revision.dictionary_gate import (
        load_eligible_topk_clt,
        require_eligible_model_method,
    )

    _verify_execution_identity(freeze, spec)
    base = Path(spec["path"]).parent
    device = torch.device(spec["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA execution was declared but CUDA is unavailable")
        torch.cuda.set_device(device)

    model_spec = spec["model"]
    model_root = _resolve(model_spec["model_root"], base)
    model_artifacts = verify_model_artifacts(
        model_root, model_spec["model_artifacts"]
    )

    p0_2 = spec["p0_2"]
    receipt_path = _resolve(p0_2["eligibility_receipt"]["path"], base)
    checkpoint_path = _resolve(p0_2["checkpoint_path"], base)
    run_hashes = {
        int(seed): digest
        for seed, digest in p0_2["run_manifest_sha256_by_seed"].items()
    }
    checkpoint_hashes = {
        int(seed): digest
        for seed, digest in p0_2["checkpoint_sha256_by_seed"].items()
    }
    selected = require_eligible_model_method(
        receipt_path,
        p0_2["eligibility_receipt"]["sha256"],
        model_name=model_spec["name"],
        method="topk_clt",
        expected_run_manifest_sha256_by_seed=run_hashes,
        expected_checkpoint_sha256_by_seed=checkpoint_hashes,
        expected_source_manifest_sha256_by_split=p0_2[
            "source_manifest_sha256_by_split"
        ],
        requested_layers=p0_2["requested_layers"],
    )
    if dictionary_loader is None:
        dictionary_loader = load_eligible_topk_clt
    clt, clt_provenance = dictionary_loader(
        receipt_path,
        p0_2["eligibility_receipt"]["sha256"],
        model_name=model_spec["name"],
        run_seed=p0_2["run_seed"],
        checkpoint_path=checkpoint_path,
        expected_run_manifest_sha256_by_seed=run_hashes,
        expected_checkpoint_sha256_by_seed=checkpoint_hashes,
        expected_source_manifest_sha256_by_split=p0_2[
            "source_manifest_sha256_by_split"
        ],
        requested_layers=p0_2["requested_layers"],
        map_location=str(device),
    )
    expected_load = {
        "model_name": model_spec["name"],
        "method": "topk_clt",
        "run_seed": p0_2["run_seed"],
        "eligibility_receipt_sha256": p0_2["eligibility_receipt"]["sha256"],
        "run_manifest_sha256": run_hashes[p0_2["run_seed"]],
        "checkpoint_sha256": checkpoint_hashes[p0_2["run_seed"]],
    }
    if any(clt_provenance.get(key) != value for key, value in expected_load.items()):
        raise ValueError("eligible TopK loader provenance differs from the execution spec")

    if model_loader is None:
        from src.models.model_loader import load_model

        model_loader = load_model
    protein_model = model_loader(
        str(model_root),
        device=str(device),
        dtype=inference_dtype(model_spec["model_inference_dtype"]),
    )
    protein_model.model.eval()
    dtype_receipt = verify_frozen_model_inference_dtype(
        protein_model, model_spec["model_inference_dtype"]
    )
    if protein_model.model.training or clt.training:
        raise RuntimeError("model and CLT must be in evaluation mode")
    if protein_model.n_layers != clt.n_layers or protein_model.d_model != clt.d_model:
        raise ValueError("pretrained model and CLT geometry disagree")
    return protein_model, clt, {
        "model_root": str(model_root),
        "model_artifacts": model_artifacts,
        **dtype_receipt,
        "p0_2_eligibility": {
            "eligibility_receipt_path": str(receipt_path),
            "eligibility_receipt_sha256": p0_2["eligibility_receipt"]["sha256"],
            "model_name": model_spec["name"],
            "method": "topk_clt",
            "run_seed": p0_2["run_seed"],
            "run_manifest_sha256": run_hashes[p0_2["run_seed"]],
            "run_manifest_sha256_by_seed": p0_2[
                "run_manifest_sha256_by_seed"
            ],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hashes[p0_2["run_seed"]],
            "checkpoint_sha256_by_seed": p0_2["checkpoint_sha256_by_seed"],
            "source_manifest_sha256_by_split": p0_2[
                "source_manifest_sha256_by_split"
            ],
            "requested_layers": p0_2["requested_layers"],
            "eligible_downstream_layers": selected["eligible_downstream_layers"],
            "profile_sha256": selected["profile_sha256"],
            "protocol_sha256": selected["protocol_sha256"],
            "cache_manifest_sha256": selected["cache_manifest_sha256"],
            "cache_content_sha256": selected["cache_content_sha256"],
        },
        "geometry": {
            "n_layers": int(clt.n_layers),
            "d_model": int(clt.d_model),
            "d_clt": int(clt.d_clt),
            "k": int(clt.k),
            "window": int(clt.window),
        },
    }


def _validate_sampler(value: Any) -> dict[str, Any]:
    sampler = _require_exact_fields(
        value, {"temperature", "top_p", "max_new_tokens"}, "plan sampler"
    )
    temperature = float(sampler["temperature"])
    top_p = float(sampler["top_p"])
    max_new_tokens = sampler["max_new_tokens"]
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("sampler temperature must be finite and positive")
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("sampler top_p must lie in (0, 1]")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("sampler max_new_tokens must be a positive integer")
    return {
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }


def validate_execution_rows(rows: Sequence[Mapping[str, Any]], clt: Any) -> list[dict]:
    """Validate supported arms/sites and bind all interventions to the CLT."""

    normalized = validate_plan_rows(rows)
    paired: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for source in normalized:
        row = dict(source)
        if row.get("arm") not in VALID_ARMS:
            raise ValueError(f"unsupported steering arm: {row.get('arm')}")
        sampler = _validate_sampler(row.get("sampler"))
        if type(row.get("seed")) is not int or row["seed"] < 0:
            raise ValueError("plan seed must be a non-negative integer")
        if row.get("multiplier_semantics") != SUPPORTED_MULTIPLIER_SEMANTICS:
            raise ValueError(
                f"unsupported multiplier semantics: {row.get('multiplier_semantics')}"
            )
        interventions = row.get("interventions")
        if not isinstance(interventions, list):
            raise ValueError("plan interventions must be a list")
        if row["arm"] == "prompt_only":
            if row.get("site") != "none" or float(row.get("dose", -1)) != 0.0 or interventions:
                raise ValueError("prompt-only rows require site=none, dose=0 and no interventions")
        else:
            if row.get("site") not in SUPPORTED_SITES:
                raise ValueError(f"unsupported hook site: {row.get('site')}")
            if not interventions:
                raise ValueError("intervention arms require at least one feature")
            dose = float(row.get("dose", 0.0))
            if not math.isfinite(dose) or dose <= 0.0:
                raise ValueError("intervention dose must be finite and positive")
            identities: set[tuple[int, int]] = set()
            for intervention in interventions:
                fields = {
                    "layer",
                    "feature",
                    "site",
                    "multiplier",
                    "decoder_norm",
                }
                item = _require_exact_fields(intervention, fields, "plan intervention")
                layer, feature = item["layer"], item["feature"]
                if (
                    type(layer) is not int
                    or not 0 <= layer < clt.n_layers
                    or type(feature) is not int
                    or not 0 <= feature < clt.d_clt
                ):
                    raise ValueError("intervention layer/feature exceeds CLT geometry")
                if item["site"] != row["site"]:
                    raise ValueError("intervention site differs from its plan row")
                multiplier = float(item["multiplier"])
                if not math.isfinite(multiplier) or multiplier != dose:
                    raise ValueError("intervention multiplier differs from the frozen dose")
                identity = (layer, feature)
                if identity in identities:
                    raise ValueError("duplicate layer/feature intervention in one plan row")
                identities.add(identity)
                actual_norm = float(
                    clt.W_dec[layer][feature, 0].detach().float().norm()
                )
                declared_norm = float(item["decoder_norm"])
                if (
                    not math.isfinite(declared_norm)
                    or declared_norm <= 0.0
                    or not math.isclose(
                        actual_norm, declared_norm, rel_tol=1e-5, abs_tol=1e-7
                    )
                ):
                    raise ValueError("frozen decoder norm differs from the verified CLT")
        row["sampler"] = sampler
        paired[(str(row["ec_class"]), int(row["seed"]))].append(row)

    for key, members in paired.items():
        prompt_rows = [row for row in members if row["arm"] == "prompt_only"]
        if len(prompt_rows) != 1:
            raise ValueError(f"paired stream {key} requires exactly one prompt-only row")
        for field in ("prompt", "rng_stream_id", "sampler"):
            values = {_canonical_sha256(row[field]) for row in members}
            if len(values) != 1:
                raise ValueError(f"paired stream {key} mixes {field}")
    return normalized


def _tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().float().cpu().contiguous().numpy().astype("<f4").tobytes()
    return hashlib.sha256(raw).hexdigest()


class _GenerationFinitenessTracker:
    """Reject each causal-LM logit tensor before generation can consume it."""

    def __init__(self):
        self.logit_tensors_checked = 0
        self.logit_elements_checked = 0

    def capture(self, _module, _inputs, output) -> None:
        logits = getattr(output, "logits", None)
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 3
            or not logits.is_floating_point()
        ):
            raise TypeError("generation forward pass did not expose rank-three logits")
        if not bool(torch.isfinite(logits).all().item()):
            raise FloatingPointError("non-finite generation logits before sampling")
        self.logit_tensors_checked += 1
        self.logit_elements_checked += logits.numel()

    def receipt(self) -> dict[str, Any]:
        if self.logit_tensors_checked < 1 or self.logit_elements_checked < 1:
            raise RuntimeError("generation produced no verified causal-LM logits")
        return {
            "activation_finiteness_check": ACTIVATION_FINITE_CHECK,
            "activation_finiteness_verified": True,
            "generation_logit_tensors_checked": self.logit_tensors_checked,
            "generation_logit_elements_checked": self.logit_elements_checked,
        }


class _HookTracker:
    """Apply one layer's additive decoder vector and verify the realized shift."""

    def __init__(self, layer: int, vector: torch.Tensor):
        self.layer = layer
        vector = vector.detach()
        if (
            vector.ndim != 1
            or not bool(torch.isfinite(vector).all().item())
        ):
            raise ValueError("intervention vector must be a finite rank-one vector")
        self.vector = vector.float()
        if not bool(torch.any(self.vector != 0)):
            raise ValueError("intervention vector cancels to exactly zero")
        self.invocations = 0
        self.tokens_modified = 0
        self.max_abs_shift = 0.0
        self.max_abs_error = 0.0
        self.max_allowed_error = 0.0

    def apply(self, value: Any) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.ndim < 2:
            raise TypeError(f"unsupported hook tensor at layer {self.layer}")
        if value.shape[-1] != self.vector.numel() or not bool(
            torch.isfinite(value).all().item()
        ):
            raise ValueError(f"hook tensor geometry/value failure at layer {self.layer}")
        vector = self.vector.to(device=value.device, dtype=value.dtype)
        changed = value + vector
        if not bool(torch.isfinite(changed).all().item()):
            raise ValueError(f"non-finite steered tensor at layer {self.layer}")
        realized = changed.float() - value.float()
        intended = vector.float().expand_as(realized)
        error = float((realized - intended).abs().max())
        scale = float(value.detach().float().abs().max()) + float(intended.abs().max()) + 1.0
        epsilon = torch.finfo(value.dtype).eps if value.is_floating_point() else 0.0
        tolerance = max(1e-7, 8.0 * epsilon * scale)
        if error > tolerance:
            raise RuntimeError(
                f"hook displacement verification failed at layer {self.layer}: "
                f"error={error} tolerance={tolerance}"
            )
        self.invocations += 1
        self.tokens_modified += int(value.numel() // value.shape[-1])
        self.max_abs_shift = max(self.max_abs_shift, float(realized.abs().max()))
        self.max_abs_error = max(self.max_abs_error, error)
        self.max_allowed_error = max(self.max_allowed_error, tolerance)
        return changed

    def receipt(self) -> dict[str, Any]:
        if self.invocations < 1 or self.tokens_modified < 1 or self.max_abs_shift <= 0.0:
            raise RuntimeError(f"intended hook had no realized effect at layer {self.layer}")
        return {
            "layer": self.layer,
            "vector_sha256_float32_le": _tensor_sha256(self.vector),
            "vector_l2_norm": float(self.vector.norm()),
            "invocations": self.invocations,
            "tokens_modified": self.tokens_modified,
            "max_abs_realized_shift": self.max_abs_shift,
            "max_abs_displacement_error": self.max_abs_error,
            "max_allowed_displacement_error": self.max_allowed_error,
            "effect_verified": True,
        }


def _register_hooks(protein_model: Any, clt: Any, row: Mapping[str, Any]):
    by_layer: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for intervention in row["interventions"]:
        by_layer[int(intervention["layer"])].append(intervention)
    trackers: list[_HookTracker] = []
    handles = []
    for layer, interventions in sorted(by_layer.items()):
        vector = sum(
            (
                clt.W_dec[layer][int(item["feature"]), 0].detach().float()
                * float(item["multiplier"])
                for item in interventions
            ),
            torch.zeros(
                clt.d_model, device=clt.W_dec[0].device, dtype=torch.float32
            ),
        )
        tracker = _HookTracker(layer, vector)
        block = protein_model._get_block(layer)
        if row["site"] == "clt_input":
            if hasattr(block, "mlp"):
                module = block.mlp
            elif hasattr(block, "final_layer_norm"):
                module = block.final_layer_norm
            else:
                raise ValueError(f"unsupported CLT-input module at layer {layer}")

            def pre_hook(_module, inputs, *, tracker=tracker):
                if not inputs:
                    raise TypeError("CLT-input hook received no positional tensor")
                return (tracker.apply(inputs[0]), *inputs[1:])

            handles.append(module.register_forward_pre_hook(pre_hook))
        elif row["site"] == "mlp_output":
            module = protein_model._get_mlp(block)

            def output_hook(_module, _inputs, output, *, tracker=tracker):
                return tracker.apply(output)

            handles.append(module.register_forward_hook(output_hook))
        else:
            raise ValueError(f"unsupported hook site: {row['site']}")
        trackers.append(tracker)
    return handles, trackers


def _tokenize_prompt(tokenizer: Any, prompt: str, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt")
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise TypeError("tokenizer must return an input_ids mapping")
    tensors = {
        key: value.to(device)
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }
    input_ids = tensors["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError("prompt tokenization must produce one non-empty row")
    if "attention_mask" not in tensors:
        tensors["attention_mask"] = torch.ones_like(input_ids)
    if tensors["attention_mask"].shape != input_ids.shape:
        raise ValueError("prompt attention_mask shape mismatch")
    return tensors


def _eos_ids(tokenizer: Any) -> set[int]:
    value = getattr(tokenizer, "eos_token_id", None)
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, int) for item in value):
        return set(value)
    raise ValueError("tokenizer must declare an EOS token ID")


def _fork_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]


def _generate_one(
    protein_model: Any,
    clt: Any,
    row: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    encoded = _tokenize_prompt(protein_model.tokenizer, str(row["prompt"]), device)
    prompt_ids = encoded["input_ids"][0].detach().cpu().tolist()
    eos_ids = _eos_ids(protein_model.tokenizer)
    pad_id = getattr(protein_model.tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = min(eos_ids)
    handles, trackers = ([], [])
    if row["arm"] != "prompt_only":
        handles, trackers = _register_hooks(protein_model, clt, row)
    finiteness_tracker = _GenerationFinitenessTracker()
    handles.append(protein_model.model.register_forward_hook(finiteness_tracker.capture))
    started_at = _utc_now()
    start = time.perf_counter()
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    try:
        random.seed(int(row["seed"]))
        np.random.seed(int(row["seed"]) % (2**32))
        with torch.random.fork_rng(devices=_fork_devices(device)):
            torch.manual_seed(int(row["seed"]))
            if device.type == "cuda":
                torch.cuda.manual_seed(int(row["seed"]))
            with torch.inference_mode():
                output = protein_model.model.generate(
                    **encoded,
                    max_new_tokens=row["sampler"]["max_new_tokens"],
                    temperature=row["sampler"]["temperature"],
                    top_p=row["sampler"]["top_p"],
                    top_k=0,
                    do_sample=True,
                    use_cache=True,
                    pad_token_id=int(pad_id),
                    eos_token_id=sorted(eos_ids),
                )
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        for handle in handles:
            handle.remove()
    elapsed = time.perf_counter() - start
    if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != 1:
        raise TypeError("model.generate must return one rank-two token tensor")
    prompt_length = len(prompt_ids)
    if output.shape[1] <= prompt_length or not torch.equal(
        output[0, :prompt_length].detach().cpu(), encoded["input_ids"][0].detach().cpu()
    ):
        raise ValueError("generated token tensor does not preserve the exact prompt prefix")
    generated_ids = [int(value) for value in output[0, prompt_length:].detach().cpu().tolist()]
    decoded = protein_model.tokenizer.decode(generated_ids, skip_special_tokens=True)
    sequence = "".join(str(decoded).split())
    if not sequence or sequence != sequence.upper() or any(residue not in AA for residue in sequence):
        raise ValueError("generated continuation is not a full canonical amino-acid sequence")
    hook_receipts = [tracker.receipt() for tracker in trackers]
    finiteness_receipt = finiteness_tracker.receipt()
    stop_reason = "eos_token" if generated_ids[-1] in eos_ids else "max_new_tokens"
    return {
        "plan_id": row["plan_id"],
        "sequence": sequence,
        "sequence_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
        "token_ids": generated_ids,
        "token_ids_sha256": _canonical_sha256(generated_ids),
        "stop_reason": stop_reason,
        "runtime": {
            "generator_revision": row["generator_revision"],
            "model_revision": row["model_revision"],
            "tokenizer_revision": row["tokenizer_revision"],
            "clt_checkpoint_sha256": row["clt_checkpoint_sha256"],
            "hostname": socket.gethostname(),
            "device": str(device),
            "started_at_utc": started_at,
            "elapsed_seconds": elapsed,
            "evaluation_mode": "eval",
            "hook_site": row["site"],
            "multiplier_semantics": row["multiplier_semantics"],
            "rng_stream_id": row["rng_stream_id"],
            "rng_seed": row["seed"],
            "rng_reset_per_plan_row": True,
            "rng_implementation": (
                "python_and_numpy_state_restore_plus_torch_global_rng_inside_fork_rng"
            ),
            "sampler": row["sampler"],
            "fixed_sampler_settings": {
                "do_sample": True,
                "top_k": 0,
                "use_cache": True,
                "pad_token_id": int(pad_id),
                "eos_token_ids": sorted(eos_ids),
            },
            "prompt_token_ids": prompt_ids,
            "prompt_token_ids_sha256": _canonical_sha256(prompt_ids),
            "token_ids_scope": "generated_continuation_including_stop_token",
            "hook_effect_verified": bool(hook_receipts) or row["arm"] == "prompt_only",
            "hook_effect_kind": (
                "none_prompt_only_control"
                if row["arm"] == "prompt_only"
                else SUPPORTED_MULTIPLIER_SEMANTICS
            ),
            "hook_receipts": hook_receipts,
            **finiteness_receipt,
        },
    }


def execute_plan_rows(
    protein_model: Any,
    clt: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    device: str,
) -> list[dict[str, Any]]:
    """Execute every validated row with paired RNG reset before each arm."""

    torch_device = torch.device(device)
    if protein_model.model.training or clt.training:
        raise RuntimeError("model and CLT must remain in evaluation mode")
    normalized = validate_execution_rows(rows, clt)
    outputs = [
        _generate_one(protein_model, clt, row, device=torch_device)
        for row in normalized
    ]
    if not all(
        row["runtime"].get("activation_finiteness_check")
        == ACTIVATION_FINITE_CHECK
        and row["runtime"].get("activation_finiteness_verified") is True
        for row in outputs
    ):
        raise RuntimeError("generation finiteness verification did not complete")
    validate_completed_generations(normalized, outputs)
    return outputs


def _resource_snapshot(device: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": device,
    }
    hardware = torch.device(device)
    if hardware.type == "cuda":
        free, total = torch.cuda.mem_get_info(hardware)
        result.update(
            accelerator=torch.cuda.get_device_name(hardware),
            accelerator_free_bytes=int(free),
            accelerator_total_bytes=int(total),
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(hardware)),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(hardware)),
        )
    return result


def _prepare_staging(output_dir: Path) -> Path:
    output_dir = Path(output_dir).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite execution output: {output_dir}")
    stale = sorted(output_dir.parent.glob(f".{output_dir.name}.tmp-*"))
    if stale:
        raise FileExistsError(f"stale partial execution staging directory exists: {stale[0]}")
    staging = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    staging.mkdir()
    return staging


def _verify_p0_2_execution_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Revalidate every external artifact behind one execution receipt."""

    from src.revision.dictionary_gate import require_eligible_model_method

    fields = {
        "eligibility_receipt_path",
        "eligibility_receipt_sha256",
        "model_name",
        "method",
        "run_seed",
        "run_manifest_sha256",
        "run_manifest_sha256_by_seed",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_sha256_by_seed",
        "source_manifest_sha256_by_split",
        "requested_layers",
        "eligible_downstream_layers",
        "profile_sha256",
        "protocol_sha256",
        "cache_manifest_sha256",
        "cache_content_sha256",
    }
    value = _require_exact_fields(binding, fields, "execution P0-2 binding")
    receipt_path = Path(value["eligibility_receipt_path"]).resolve()
    checkpoint_path = Path(value["checkpoint_path"]).resolve()
    receipt_sha256 = _require_sha256(
        value["eligibility_receipt_sha256"], "P0-2 eligibility receipt SHA-256"
    )
    checkpoint_sha256 = _require_sha256(
        value["checkpoint_sha256"], "P0-2 selected checkpoint SHA-256"
    )
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != checkpoint_sha256:
        raise ValueError("selected exact-cache best.pt is missing or its hash drifted")
    run_hashes = {
        int(seed): _require_sha256(digest, f"P0-2 run manifest seed {seed}")
        for seed, digest in _require_exact_fields(
            value["run_manifest_sha256_by_seed"],
            {"17", "29", "43"},
            "execution P0-2 run-manifest hashes",
        ).items()
    }
    checkpoint_hashes = {
        int(seed): _require_sha256(digest, f"P0-2 checkpoint seed {seed}")
        for seed, digest in _require_exact_fields(
            value["checkpoint_sha256_by_seed"],
            {"17", "29", "43"},
            "execution P0-2 checkpoint hashes",
        ).items()
    }
    run_seed = value["run_seed"]
    if (
        run_seed not in {17, 29, 43}
        or run_hashes[run_seed] != value["run_manifest_sha256"]
        or checkpoint_hashes[run_seed] != checkpoint_sha256
    ):
        raise ValueError("selected P0-2 seed artifacts differ from the all-seed binding")
    selected = require_eligible_model_method(
        receipt_path,
        receipt_sha256,
        model_name=value["model_name"],
        method=value["method"],
        expected_run_manifest_sha256_by_seed=run_hashes,
        expected_checkpoint_sha256_by_seed=checkpoint_hashes,
        expected_source_manifest_sha256_by_split=value[
            "source_manifest_sha256_by_split"
        ],
        requested_layers=value["requested_layers"],
    )
    for field in (
        "eligible_downstream_layers",
        "profile_sha256",
        "protocol_sha256",
        "cache_manifest_sha256",
        "cache_content_sha256",
    ):
        if value[field] != selected[field]:
            raise ValueError(f"execution P0-2 {field} differs from the eligibility receipt")
    return selected


def verify_execution_receipt(
    receipt_path: Path,
    expected_sha256: str,
    *,
    generation_outputs_path: Path,
    freeze_id: str,
    freeze_manifest_sha256: str,
    allow_test_fixture: bool = False,
) -> dict[str, Any]:
    """Verify a generation receipt for downstream scoring or analysis."""

    receipt_path = Path(receipt_path).resolve()
    expected_sha256 = _require_sha256(expected_sha256, "execution receipt SHA-256")
    if not receipt_path.is_file() or sha256_file(receipt_path) != expected_sha256:
        raise ValueError("execution receipt is missing or its SHA-256 changed")
    receipt = _require_exact_fields(
        load_json(receipt_path),
        {
            "schema_version",
            "status",
            "claim_boundary",
            "command",
            "started_at_utc",
            "completed_at_utc",
            "freeze",
            "execution_spec",
            "model",
            "environment",
            "source_hashes",
            "artifacts",
        },
        "execution receipt",
    )
    expected_status = (
        "verified_test_fixture_complete" if allow_test_fixture else "verified_complete"
    )
    if (
        receipt["schema_version"]
        != "r2-corrected-steering-execution-receipt-v3"
        or receipt["status"] != expected_status
    ):
        raise ValueError("execution receipt is not a complete v3 production receipt")
    freeze = receipt["freeze"]
    if (
        not isinstance(freeze, Mapping)
        or freeze.get("freeze_id") != freeze_id
        or freeze.get("manifest_sha256") != freeze_manifest_sha256
    ):
        raise ValueError("execution receipt freeze binding differs")
    verified_freeze = load_verified_freeze(
        Path(freeze["directory"]), freeze_manifest_sha256
    )
    outputs_path = Path(generation_outputs_path).resolve()
    if outputs_path != receipt_path.parent / "generation_outputs.jsonl":
        raise ValueError("generation outputs are not colocated with their execution receipt")
    artifacts = receipt["artifacts"]
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts) != {"generation_outputs.jsonl", "execution_summary.json"}
        or sha256_file(outputs_path) != artifacts["generation_outputs.jsonl"]
        or sha256_file(receipt_path.parent / "execution_summary.json")
        != artifacts["execution_summary.json"]
    ):
        raise ValueError("execution receipt artifact hashes differ")
    summary = load_json(receipt_path.parent / "execution_summary.json")
    if (
        not isinstance(summary, Mapping)
        or summary.get("schema_version")
        != "r2-corrected-steering-execution-summary-v2"
    ):
        raise ValueError("execution summary is not the amended v2 contract")
    _require_numerical_integrity(summary)
    if receipt["source_hashes"] != _execution_source_hashes():
        raise ValueError("execution source hashes drifted after generation")
    spec = receipt["execution_spec"]
    if (
        not isinstance(spec, Mapping)
        or set(spec) != {"path", "sha256"}
        or sha256_file(Path(spec["path"])) != spec["sha256"]
    ):
        raise ValueError("execution specification hash drifted")
    verified_spec = load_execution_spec(Path(spec["path"]), spec["sha256"])
    model = receipt["model"]
    if not isinstance(model, Mapping) or "p0_2_eligibility" not in model:
        raise ValueError("execution receipt lacks P0-2 eligibility provenance")
    _require_numerical_integrity(model)
    if (
        model.get("name") != verified_spec["model"]["name"]
        or model.get("model_revision")
        != verified_spec["model"]["model_revision"]
        or model.get("tokenizer_revision")
        != verified_spec["model"]["tokenizer_revision"]
    ):
        raise ValueError("execution receipt model identity differs from its specification")
    _verify_p0_2_execution_binding(model["p0_2_eligibility"])
    if (
        model["p0_2_eligibility"]["checkpoint_sha256"]
        != verified_freeze["provenance"]["clt_checkpoint_sha256"]
    ):
        raise ValueError("execution P0-2 checkpoint differs from the frozen plan")
    return receipt


def run_execution(
    *,
    frozen_dir: Path,
    freeze_manifest_sha256: str,
    spec_path: Path,
    spec_sha256: str,
    output_dir: Path,
    model_loader: Callable | None = None,
    dictionary_loader: Callable | None = None,
    command: Sequence[str] | None = None,
) -> Path:
    """Run the complete generation plan and atomically publish its receipt."""

    freeze = load_verified_freeze(frozen_dir, freeze_manifest_sha256)
    spec = load_execution_spec(spec_path, spec_sha256)
    output_dir = Path(output_dir).resolve()
    staging = _prepare_staging(output_dir)
    started = _utc_now()
    test_overrides = model_loader is not None or dictionary_loader is not None
    try:
        source_hashes = _execution_source_hashes()
        if spec["device"].startswith("cuda"):
            cuda_device = torch.device(spec["device"])
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA execution was declared but CUDA is unavailable")
            torch.cuda.set_device(cuda_device)
            torch.cuda.reset_peak_memory_stats(cuda_device)
        protein_model, clt, resource_provenance = load_execution_resources(
            freeze,
            spec,
            model_loader=model_loader,
            dictionary_loader=dictionary_loader,
        )
        outputs = execute_plan_rows(
            protein_model, clt, freeze["rows"], device=spec["device"]
        )
        numerical_integrity = _require_numerical_integrity(
            {
                **resource_provenance,
                "activation_finiteness_check": ACTIVATION_FINITE_CHECK,
                "activation_finiteness_verified": True,
            }
        )
        if _execution_source_hashes() != source_hashes:
            raise RuntimeError("execution source files changed during generation")
        generations_path = staging / "generation_outputs.jsonl"
        write_jsonl(generations_path, outputs)
        arm_counts = Counter(row["arm"] for row in freeze["rows"])
        site_counts = Counter(row["site"] for row in freeze["rows"])
        summary_path = staging / "execution_summary.json"
        write_json(
            summary_path,
            {
                "schema_version": "r2-corrected-steering-execution-summary-v2",
                "status": "verified_complete_generation_only_analysis_not_run",
                "claim_boundary": (
                    "Complete generation infrastructure is not P0-6 scientific evidence; "
                    "validated endpoint scoring and frozen analysis remain unexecuted."
                ),
                "freeze_id": freeze["summary"]["freeze_id"],
                "n_planned_rows": len(freeze["rows"]),
                "n_completed_rows": len(outputs),
                "arm_counts": dict(sorted(arm_counts.items())),
                "site_counts": dict(sorted(site_counts.items())),
                "all_hook_effects_verified": all(
                    row["runtime"]["hook_effect_verified"] for row in outputs
                ),
                "paired_rng_reset_per_row": True,
                "multiplier_semantics": SUPPORTED_MULTIPLIER_SEMANTICS,
                **numerical_integrity,
            },
        )
        artifact_hashes = {
            path.name: sha256_file(path) for path in (generations_path, summary_path)
        }
        receipt_path = staging / "execution_receipt.json"
        write_json(
            receipt_path,
            {
                "schema_version": "r2-corrected-steering-execution-receipt-v3",
                "status": (
                    "verified_test_fixture_complete"
                    if test_overrides
                    else "verified_complete"
                ),
                "claim_boundary": "Generation completion alone does not pass P0-6.",
                "command": list(command or [sys.executable]),
                "started_at_utc": started,
                "completed_at_utc": _utc_now(),
                "freeze": {
                    "directory": str(freeze["directory"]),
                    "freeze_id": freeze["summary"]["freeze_id"],
                    "manifest_sha256": freeze["manifest_sha256"],
                    "generation_plan_sha256": freeze["artifact_hashes"][
                        "generation_plan.jsonl"
                    ],
                    "artifact_hashes": freeze["artifact_hashes"],
                },
                "execution_spec": {
                    "path": str(spec["path"]),
                    "sha256": spec["sha256"],
                },
                "model": {
                    "name": spec["model"]["name"],
                    "model_revision": spec["model"]["model_revision"],
                    "tokenizer_revision": spec["model"]["tokenizer_revision"],
                    **resource_provenance,
                    **numerical_integrity,
                },
                "environment": _resource_snapshot(spec["device"]),
                "source_hashes": source_hashes,
                "artifacts": artifact_hashes,
            },
        )
        verify_execution_receipt(
            receipt_path,
            sha256_file(receipt_path),
            generation_outputs_path=generations_path,
            freeze_id=freeze["summary"]["freeze_id"],
            freeze_manifest_sha256=freeze["manifest_sha256"],
            allow_test_fixture=test_overrides,
        )
        if output_dir.exists():
            raise FileExistsError(f"execution output appeared during run: {output_dir}")
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output_dir / "execution_receipt.json"
