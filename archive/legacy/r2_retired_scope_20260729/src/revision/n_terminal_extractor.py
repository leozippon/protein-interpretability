"""Hash-bound pretrained-model measurements for the P0-5 factorial."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ..models.model_loader import (
    INFERENCE_DTYPE_VERIFICATION,
    inference_dtype,
    verify_frozen_model_inference_dtype,
)
from .input_builder import (
    format_model_input,
    load_json,
    load_jsonl,
    residue_token_indices,
    verify_model_artifacts,
)
from .io import sha256_file, write_json, write_jsonl
from .n_terminal_counterfactuals import (
    BOS_POLICIES,
    CounterfactualVariant,
    build_counterfactual_variants,
    normalize_measurement_rows,
    received_attention_by_key,
    token_ids_sha256,
)


AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
HEX = frozenset("0123456789abcdef")
ACTIVATION_FINITE_CHECK = (
    "all_required_layer_captured_activation_and_logit_tensors_before_"
    "downstream_conversion_or_use"
)
FEATURE_FIELDS = {
    "model",
    "layer",
    "feature",
    "feature_role",
    "firing_frequency",
    "input_norm",
}
SPEC_FIELDS = {
    "schema_version",
    "code_revision",
    "device",
    "max_model_tokens",
    "model",
    "p0_2_gate_receipt",
    "dictionary",
    "cohorts",
    "feature_profiles",
    "protein_matching",
    "feature_matching",
    "counterfactual",
}


def validate_focal_position_contract(
    rows: Sequence[Mapping[str, Any]], *, internal_fraction: float
) -> None:
    """Bind matching focal positions to the exact counterfactual insertion site."""

    for row in rows:
        length = len(row["sequence"])
        expected = int(round((length - 3) * internal_fraction))
        expected = min(max(expected, 4), length - 4)
        if row["focal_position"] != expected:
            raise ValueError(
                f"protein {row['protein_id']} focal_position={row['focal_position']} "
                f"does not equal frozen internal insertion site {expected}"
            )


def _require_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{label} fields differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= HEX:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve(path: str, base: Path) -> Path:
    candidate = Path(path).expanduser()
    return (
        candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    )


def _descriptor(value: Any, base: Path, label: str) -> tuple[dict[str, Any], Path]:
    descriptor = _require_keys(
        value, {"cohort_id", "path", "sha256"}, f"{label} descriptor"
    )
    if not isinstance(descriptor["cohort_id"], str) or not descriptor["cohort_id"]:
        raise ValueError(f"{label} cohort_id must be non-empty")
    path = _resolve(descriptor["path"], base)
    expected = _require_sha256(descriptor["sha256"], f"{label} SHA-256")
    if sha256_file(path) != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    return descriptor, path


def _clean_sequence(value: Any, label: str) -> str:
    sequence = str(value)
    if (
        len(sequence) < 18
        or sequence != sequence.upper()
        or any(residue not in AA for residue in sequence)
    ):
        raise ValueError(
            f"{label} must contain at least 18 uppercase canonical residues"
        )
    return sequence


def load_protein_manifest(
    descriptor: Mapping[str, Any],
    *,
    base: Path,
    label: str,
    require_focal_position: bool,
    require_mxx_start: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load one immutable protein manifest without silently repairing records."""

    normalized_descriptor, path = _descriptor(dict(descriptor), base, label)
    rows = load_jsonl(path)
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_sequences: set[str] = set()
    for index, source in enumerate(rows, 1):
        if not isinstance(source, dict):
            raise ValueError(f"{label}:{index} must be an object")
        protein_id = str(source.get("protein_id") or source.get("id") or "").strip()
        if not protein_id or protein_id in seen_ids:
            raise ValueError(f"{label} protein IDs must be non-empty and unique")
        sequence = _clean_sequence(
            source.get("sequence", ""), f"{label}:{index} sequence"
        )
        digest = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        if "sha256" in source and source["sha256"] != digest:
            raise ValueError(f"{label}:{index} sequence SHA-256 mismatch")
        if digest in seen_sequences:
            raise ValueError(f"{label} contains duplicate exact sequences")
        if require_mxx_start and sequence[0] != "M":
            raise ValueError(f"{label} factorial proteins must have natural MXX starts")
        record = dict(source)
        record["protein_id"] = protein_id
        record["sequence"] = sequence
        record["sha256"] = digest
        if require_focal_position:
            position = source.get("focal_position")
            if type(position) is not int or not 0 <= position < len(sequence):
                raise ValueError(
                    f"{label}:{index} requires an in-range integer focal_position"
                )
            record["focal_position"] = position
        seen_ids.add(protein_id)
        seen_sequences.add(digest)
        normalized.append(record)
    return normalized, {
        "cohort_id": normalized_descriptor["cohort_id"],
        "path": str(path),
        "sha256": normalized_descriptor["sha256"],
        "n_records": len(normalized),
    }


def validate_three_way_disjoint(
    targets: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    discovery: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require target, full control pool, and discovery to be ID/sequence disjoint."""

    groups = {
        "target": targets,
        "protein_control_pool": controls,
        "discovery": discovery,
    }
    identities = {
        name: ({row["protein_id"] for row in rows}, {row["sha256"] for row in rows})
        for name, rows in groups.items()
    }
    names = list(groups)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap_ids = identities[left][0] & identities[right][0]
            overlap_sequences = identities[left][1] & identities[right][1]
            if overlap_ids or overlap_sequences:
                raise ValueError(
                    f"{left}/{right} overlap: ids={sorted(overlap_ids)}, "
                    f"sequence_sha256={sorted(overlap_sequences)}"
                )
    return {
        "all_input_cohorts_disjoint": True,
        "comparison": "protein_id_and_exact_sequence_sha256",
        "counts": {name: len(rows) for name, rows in groups.items()},
    }


def match_protein_controls(
    targets: Sequence[Mapping[str, Any]],
    control_pool: Sequence[Mapping[str, Any]],
    *,
    max_length_difference: int,
    max_normalized_position_difference: float,
) -> list[dict[str, Any]]:
    """Globally match one distinct held-out control protein to every target."""

    if type(max_length_difference) is not int or max_length_difference < 1:
        raise ValueError("max_length_difference must be a positive integer")
    if not np.isfinite(max_normalized_position_difference) or not (
        0.0 < max_normalized_position_difference <= 1.0
    ):
        raise ValueError("max_normalized_position_difference must lie in (0, 1]")
    ordered_targets = sorted(targets, key=lambda row: row["protein_id"])
    ordered_controls = sorted(control_pool, key=lambda row: row["protein_id"])
    if not ordered_targets or len(ordered_controls) < len(ordered_targets):
        raise ValueError("protein control pool is too small for one-to-one matching")

    costs = np.full((len(ordered_targets), len(ordered_controls)), np.inf)
    diagnostics: dict[tuple[int, int], tuple[int, float]] = {}
    for target_index, target in enumerate(ordered_targets):
        target_position = target["focal_position"] / max(len(target["sequence"]) - 1, 1)
        for control_index, control in enumerate(ordered_controls):
            control_position = control["focal_position"] / max(
                len(control["sequence"]) - 1, 1
            )
            length_difference = abs(len(target["sequence"]) - len(control["sequence"]))
            position_difference = abs(target_position - control_position)
            diagnostics[(target_index, control_index)] = (
                length_difference,
                position_difference,
            )
            if (
                length_difference <= max_length_difference
                and position_difference <= max_normalized_position_difference
            ):
                costs[target_index, control_index] = (
                    length_difference / max_length_difference
                    + position_difference / max_normalized_position_difference
                )
    if np.any(~np.isfinite(costs).any(axis=1)):
        raise ValueError(
            "at least one target has no protein control inside both calipers"
        )
    try:
        rows, columns = linear_sum_assignment(costs)
    except ValueError as error:
        raise ValueError(
            "no complete one-to-one protein matching exists inside both calipers"
        ) from error
    if len(rows) != len(ordered_targets) or not np.isfinite(costs[rows, columns]).all():
        raise ValueError(
            "no complete one-to-one protein matching exists inside both calipers"
        )

    pairs = []
    for target_index, control_index in sorted(zip(rows.tolist(), columns.tolist())):
        target = ordered_targets[target_index]
        control = ordered_controls[control_index]
        length_difference, position_difference = diagnostics[
            (target_index, control_index)
        ]
        identity = {
            "target_protein_id": target["protein_id"],
            "target_sequence_sha256": target["sha256"],
            "control_protein_id": control["protein_id"],
            "control_sequence_sha256": control["sha256"],
            "max_length_difference": max_length_difference,
            "max_normalized_position_difference": max_normalized_position_difference,
        }
        pairs.append(
            {
                "protein_pair_id": _canonical_sha256(identity),
                **identity,
                "target_length": len(target["sequence"]),
                "control_length": len(control["sequence"]),
                "target_focal_position": target["focal_position"],
                "control_focal_position": control["focal_position"],
                "absolute_length_difference": length_difference,
                "target_normalized_position": target["focal_position"]
                / max(len(target["sequence"]) - 1, 1),
                "control_normalized_position": control["focal_position"]
                / max(len(control["sequence"]) - 1, 1),
                "absolute_normalized_position_difference": position_difference,
                "cost": float(costs[target_index, control_index]),
                "method": "global_minimum_cost_one_to_one_scipy_linear_sum_assignment",
            }
        )
    return pairs


def load_and_match_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_name: str,
    control_count: int,
    max_abs_log10_firing_ratio: float,
    max_abs_log_input_norm_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Freeze same-layer feature controls using discovery-only profiles."""

    if type(control_count) is not int or control_count < 1:
        raise ValueError("feature control_count must be a positive integer")
    calipers = (max_abs_log10_firing_ratio, max_abs_log_input_norm_ratio)
    if any(not np.isfinite(value) or value <= 0.0 for value in calipers):
        raise ValueError("feature log-ratio calipers must be finite and positive")
    profiles: list[dict[str, Any]] = []
    identities: set[tuple[int, int]] = set()
    for index, source in enumerate(rows, 1):
        if not isinstance(source, Mapping) or set(source) != FEATURE_FIELDS:
            raise ValueError(
                f"feature profile {index} fields differ from the frozen schema"
            )
        if source["model"] != model_name:
            raise ValueError("feature profile model identity mismatch")
        layer, feature = source["layer"], source["feature"]
        if (
            type(layer) is not int
            or layer < 0
            or type(feature) is not int
            or feature < 0
        ):
            raise ValueError("feature layer/index must be non-negative integers")
        if (layer, feature) in identities:
            raise ValueError("duplicate feature profile identity")
        role = source["feature_role"]
        if role not in {"target", "candidate_control"}:
            raise ValueError("feature_role must be target or candidate_control")
        firing = float(source["firing_frequency"])
        input_norm = float(source["input_norm"])
        if not (np.isfinite(firing) and 0.0 < firing <= 1.0):
            raise ValueError("feature firing_frequency must lie in (0, 1]")
        if not (np.isfinite(input_norm) and input_norm > 0.0):
            raise ValueError("feature input_norm must be finite and positive")
        identities.add((layer, feature))
        profiles.append(
            {
                "model": model_name,
                "layer": layer,
                "feature": feature,
                "feature_role": role,
                "firing_frequency": firing,
                "input_norm": input_norm,
            }
        )

    targets = [profile for profile in profiles if profile["feature_role"] == "target"]
    if not targets:
        raise ValueError("at least one target feature is required")
    layers = [profile["layer"] for profile in targets]
    if len(layers) != len(set(layers)):
        raise ValueError(
            "the stable analyzer contract permits at most one target feature per model/layer"
        )

    selected: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    for target in sorted(targets, key=lambda row: (row["layer"], row["feature"])):
        candidates = [
            profile
            for profile in profiles
            if profile["feature_role"] == "candidate_control"
            and profile["layer"] == target["layer"]
        ]
        eligible = []
        for candidate in candidates:
            firing_difference = abs(
                np.log10(candidate["firing_frequency"] / target["firing_frequency"])
            )
            norm_difference = abs(
                np.log(candidate["input_norm"] / target["input_norm"])
            )
            if (
                firing_difference <= max_abs_log10_firing_ratio
                and norm_difference <= max_abs_log_input_norm_ratio
            ):
                distance = float(
                    np.hypot(
                        firing_difference / max_abs_log10_firing_ratio,
                        norm_difference / max_abs_log_input_norm_ratio,
                    )
                )
                eligible.append(
                    (
                        distance,
                        candidate["feature"],
                        candidate,
                        firing_difference,
                        norm_difference,
                    )
                )
        eligible.sort(key=lambda item: (item[0], item[1]))
        if len(eligible) < control_count:
            raise ValueError(
                f"target layer={target['layer']} feature={target['feature']} has only "
                f"{len(eligible)} feature controls inside both calipers"
            )
        chosen = eligible[:control_count]
        match_identity = {
            "model": model_name,
            "layer": target["layer"],
            "target_feature": target["feature"],
            "control_features": [item[2]["feature"] for item in chosen],
            "max_abs_log10_firing_ratio": max_abs_log10_firing_ratio,
            "max_abs_log_input_norm_ratio": max_abs_log_input_norm_ratio,
        }
        match_id = _canonical_sha256(match_identity)
        target_output = {
            **target,
            "feature_role": "target",
            "feature_match_id": match_id,
        }
        selected.append(target_output)
        control_rows = []
        for distance, _, candidate, firing_difference, norm_difference in chosen:
            output = {
                **candidate,
                "feature_role": "control",
                "feature_match_id": match_id,
                "matched_target_feature": target["feature"],
            }
            selected.append(output)
            control_rows.append(
                {
                    "feature": candidate["feature"],
                    "firing_frequency": candidate["firing_frequency"],
                    "input_norm": candidate["input_norm"],
                    "absolute_log10_firing_ratio": float(firing_difference),
                    "absolute_log_input_norm_ratio": float(norm_difference),
                    "caliper_scaled_distance": distance,
                }
            )
        matches.append(
            {
                "feature_match_id": match_id,
                **match_identity,
                "target_profile": {
                    "firing_frequency": target["firing_frequency"],
                    "input_norm": target["input_norm"],
                },
                "controls": control_rows,
                "method": "same_model_layer_log_ratio_calipers_then_scaled_euclidean",
            }
        )
    return selected, matches


def tokenize_bos_factorial(
    tokenizer,
    text: str,
    *,
    construction: str,
    max_model_tokens: int,
) -> dict[str, torch.Tensor]:
    """Create native/removed BOS inputs under one explicit frozen construction."""

    kwargs = {
        "return_tensors": "pt",
        "truncation": False,
        "return_attention_mask": True,
    }
    if construction == "tokenizer_native_leading_bos":
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            raise ValueError(
                "tokenizer native encoding does not contain the required leading BOS"
            )
        encoded = tokenizer(text, add_special_tokens=True, **kwargs)
        native = encoded["input_ids"]
        if native.ndim != 2 or native.shape[0] != 1 or native[0, 0].item() != bos_id:
            raise ValueError(
                "tokenizer native encoding does not contain the required leading BOS"
            )
        removed = native[:, 1:]
    elif construction == "explicit_prepend_tokenizer_bos":
        encoded = tokenizer(text, add_special_tokens=False, **kwargs)
        removed = encoded["input_ids"]
        bos_id = tokenizer.bos_token_id
        if bos_id is None or removed.ndim != 2 or removed.shape[0] != 1:
            raise ValueError(
                "tokenizer lacks the BOS ID required by the frozen construction"
            )
        if removed.shape[1] and removed[0, 0].item() == bos_id:
            raise ValueError(
                "no-special-token encoding unexpectedly already begins with BOS"
            )
        native = torch.cat(
            (torch.tensor([[bos_id]], dtype=removed.dtype), removed), dim=1
        )
    else:
        raise ValueError("unknown BOS construction")
    if removed.shape[1] < 1:
        raise ValueError("BOS removal produced an empty model input")
    if native.shape[1] > max_model_tokens or removed.shape[1] > max_model_tokens:
        raise ValueError("counterfactual exceeds the frozen model-token limit")
    return {"native": native, "removed": removed}


def capture_clt_inputs_and_attentions(
    protein_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layers: Sequence[int],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], torch.Tensor]:
    """Capture the model-loader CLT hook sites and exact same-pass attentions."""

    captured: dict[int, torch.Tensor] = {}
    handles = []

    def hook(layer: int):
        def capture(_module, inputs, _output):
            if layer in captured:
                raise RuntimeError(f"duplicate CLT-input capture at layer {layer}")
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError(f"invalid CLT-input hook value at layer {layer}")
            captured[layer] = inputs[0].detach()

        return capture

    for layer in layers:
        block = protein_model._get_block(layer)
        if hasattr(block, "mlp"):
            site = block.mlp
        elif hasattr(block, "final_layer_norm"):
            site = block.final_layer_norm
        else:
            raise ValueError(f"unsupported CLT-input hook site at layer {layer}")
        handles.append(site.register_forward_hook(hook(layer)))
    try:
        with torch.inference_mode():
            output = protein_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()
    attentions = getattr(output, "attentions", None)
    logits = getattr(output, "logits", None)
    if set(captured) != set(layers) or attentions is None:
        raise RuntimeError(
            "model did not return every requested CLT input and attention tensor"
        )
    selected_attentions: dict[int, torch.Tensor] = {}
    for layer in layers:
        values = captured[layer]
        attention = attentions[layer] if layer < len(attentions) else None
        if (
            values.ndim != 3
            or values.shape[:2] != input_ids.shape
            or values.shape[-1] != protein_model.d_model
        ):
            raise ValueError(f"CLT-input shape mismatch at layer {layer}")
        if not bool(torch.isfinite(values).all().item()):
            raise FloatingPointError(f"non-finite CLT-input capture at layer {layer}")
        if (
            not isinstance(attention, torch.Tensor)
            or attention.ndim != 4
            or attention.shape[0] != 1
            or attention.shape[-2:] != input_ids.shape[-1:] * 2
            or not torch.isfinite(attention).all()
        ):
            raise ValueError(f"attention shape/value mismatch at layer {layer}")
        if torch.any(attention < -1e-7):
            raise ValueError(f"attention contains negative values at layer {layer}")
        if torch.any(torch.triu(attention, diagonal=1).abs() > 1e-6):
            raise ValueError(f"attention violates causal masking at layer {layer}")
        selected_attentions[layer] = attention.detach()
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 3
        or logits.shape[:2] != input_ids.shape
        or logits.shape[-1] <= int(input_ids.max().item())
        or not torch.isfinite(logits).all()
    ):
        raise ValueError("model did not return finite causal-LM logits")
    return captured, selected_attentions, logits.detach()


def measure_focal_key_intervention(
    *,
    protein_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    focal_token_index: int,
    baseline_logits: torch.Tensor,
) -> dict[str, Any]:
    """Mask the focal key and measure strict-suffix causal-LM displacement.

    Predictions made at the focal query are excluded. This avoids treating an
    empty first-token attention row as a valid intervention when BOS is removed.
    """

    length = input_ids.shape[1]
    if not 0 <= focal_token_index < length - 2:
        raise ValueError("focal-key intervention requires at least two suffix tokens")
    intervened_mask = attention_mask.clone()
    if intervened_mask[0, focal_token_index].item() != 1:
        raise ValueError("baseline focal token must be a valid attention key")
    intervened_mask[0, focal_token_index] = 0
    with torch.inference_mode():
        output = protein_model.model(
            input_ids=input_ids,
            attention_mask=intervened_mask,
            output_attentions=True,
            use_cache=False,
        )
    logits = getattr(output, "logits", None)
    attentions = getattr(output, "attentions", None)
    if (
        not isinstance(logits, torch.Tensor)
        or logits.shape != baseline_logits.shape
        or not torch.isfinite(logits).all()
        or not isinstance(attentions, (tuple, list))
        or len(attentions) != protein_model.n_layers
    ):
        raise ValueError("model wrapper cannot safely execute the focal-key intervention")
    max_abs = 0.0
    for layer, attention in enumerate(attentions):
        if (
            not isinstance(attention, torch.Tensor)
            or attention.ndim != 4
            or attention.shape[-2:] != (length, length)
            or not torch.isfinite(attention).all()
        ):
            raise ValueError(f"invalid intervened attention tensor at layer {layer}")
        strict_suffix = attention[..., focal_token_index + 1 :, focal_token_index]
        max_abs = max(max_abs, float(strict_suffix.abs().max().item()))
    if max_abs > 1e-6:
        raise ValueError(
            "model attention mask did not remove the focal key from every strict-suffix query"
        )

    # Query positions q>focal predict observed tokens q+1. The immediate token
    # after the focal position is deliberately excluded because its predictor
    # is the focal query itself.
    prediction_logits = baseline_logits[0, focal_token_index + 1 : -1].float()
    intervened_logits = logits[0, focal_token_index + 1 : -1].float()
    observed = input_ids[0, focal_token_index + 2 :]
    if prediction_logits.shape[0] != observed.shape[0] or observed.numel() < 1:
        raise RuntimeError("strict-suffix logit/target alignment failed")
    baseline_nll = F.cross_entropy(prediction_logits, observed, reduction="mean")
    intervened_nll = F.cross_entropy(intervened_logits, observed, reduction="mean")
    baseline_observed_logits = prediction_logits.gather(1, observed[:, None]).mean()
    intervened_observed_logits = intervened_logits.gather(1, observed[:, None]).mean()
    return {
        "attention_key_mask_scope": (
            "focal_key_masked_all_layers_for_strict_suffix_queries_q_gt_focal"
        ),
        "attention_key_mask_max_abs_strict_suffix": max_abs,
        "baseline_suffix_nll": float(baseline_nll.item()),
        "key_masked_suffix_nll": float(intervened_nll.item()),
        "suffix_nll_increase_key_masked": float(
            (intervened_nll - baseline_nll).item()
        ),
        "baseline_suffix_observed_token_logit_mean": float(
            baseline_observed_logits.item()
        ),
        "key_masked_suffix_observed_token_logit_mean": float(
            intervened_observed_logits.item()
        ),
        "suffix_observed_token_logit_change_key_masked": float(
            (intervened_observed_logits - baseline_observed_logits).item()
        ),
        "attention_path_interpretation": (
            "focal-key path perturbation conditional on position/length; "
            "not formal feature mediation"
        ),
    }


def _encode_selected_layer(clt, layer: int, values: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        preactivation = torch.relu(
            torch.einsum("bsd,fd->bsf", values, clt.W_enc[layer]) + clt.b_enc[layer]
        )
        top_values, top_indices = preactivation.topk(clt.k, dim=-1)
        return torch.zeros_like(preactivation).scatter_(-1, top_indices, top_values)


def extract_measurement_rows(
    *,
    protein_model,
    clt,
    model_name: str,
    model_revision: str,
    tokenizer_revision: str,
    input_format: str,
    bos_construction: str,
    max_model_tokens: int,
    variants: Sequence[CounterfactualVariant],
    source_records: Mapping[str, Mapping[str, Any]],
    selected_features: Sequence[Mapping[str, Any]],
    protein_metadata: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Measure the full protein x condition x BOS x feature table."""

    layers = sorted({int(profile["layer"]) for profile in selected_features})
    if not layers or min(layers) < 0 or max(layers) >= clt.n_layers:
        raise ValueError("selected feature layer exceeds CLT depth")
    if protein_model.n_layers != clt.n_layers or protein_model.d_model != clt.d_model:
        raise ValueError("pretrained model and CLT geometry disagree")
    if any(int(profile["feature"]) >= clt.d_clt for profile in selected_features):
        raise ValueError("selected feature exceeds CLT width")
    profiles_by_layer = {
        layer: [profile for profile in selected_features if profile["layer"] == layer]
        for layer in layers
    }
    rows: list[dict[str, Any]] = []
    device = torch.device(protein_model.device)
    for variant in variants:
        source = source_records[variant.protein_id]
        formatted = format_model_input(
            {**source, "sequence": variant.sequence}, input_format
        )
        tokenizations = tokenize_bos_factorial(
            protein_model.tokenizer,
            formatted,
            construction=bos_construction,
            max_model_tokens=max_model_tokens,
        )
        for bos_policy in BOS_POLICIES:
            input_ids = tokenizations[bos_policy].to(device)
            attention_mask = torch.ones_like(input_ids)
            residue_tokens = residue_token_indices(
                protein_model.tokenizer, input_ids, variant.sequence
            )
            focal_token_index = int(residue_tokens[variant.focal_start])
            clt_inputs, attentions, baseline_logits = capture_clt_inputs_and_attentions(
                protein_model, input_ids, attention_mask, layers
            )
            intervention = measure_focal_key_intervention(
                protein_model=protein_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                focal_token_index=focal_token_index,
                baseline_logits=baseline_logits,
            )
            token_ids = input_ids[0].detach().cpu().tolist()
            token_digest = token_ids_sha256(token_ids)
            for layer in layers:
                raw = clt_inputs[layer].float()
                sparse = _encode_selected_layer(clt, layer, raw)
                head_mean = attentions[layer][0].float().mean(dim=0).cpu().numpy()
                received = received_attention_by_key(
                    head_mean, np.ones(len(token_ids), dtype=bool)
                )
                eligible = int(received["eligible_query_count"][focal_token_index])
                if eligible != len(token_ids) - focal_token_index:
                    raise RuntimeError(
                        "eligible causal-query count violated its exact contract"
                    )
                for profile in profiles_by_layer[layer]:
                    metadata = protein_metadata[variant.protein_id]
                    rows.append(
                        {
                            "protein_id": variant.protein_id,
                            "condition": variant.condition,
                            "bos_policy": bos_policy,
                            "model": model_name,
                            "layer": layer,
                            "feature": int(profile["feature"]),
                            "feature_role": profile["feature_role"],
                            "feature_activation_pre": float(
                                sparse[0, focal_token_index, profile["feature"]].item()
                            ),
                            "received_attention_raw": float(
                                received["raw_received_attention"][focal_token_index]
                            ),
                            "eligible_query_count": eligible,
                            "normalized_focal_position": variant.normalized_focal_position,
                            "sequence_length": variant.sequence_length,
                            "firing_frequency": float(profile["firing_frequency"]),
                            "input_norm": float(profile["input_norm"]),
                            "variant_sha256": variant.sha256,
                            "tokenizer_revision": tokenizer_revision,
                            "token_ids": token_ids,
                            "token_ids_sha256": token_digest,
                            "focal_token_index": focal_token_index,
                            "model_revision": model_revision,
                            "formatted_model_input_sha256": hashlib.sha256(
                                formatted.encode("utf-8")
                            ).hexdigest(),
                            "input_format": input_format,
                            "bos_construction": bos_construction,
                            "attention_layer": layer,
                            "attention_head_count": int(attentions[layer].shape[1]),
                            "attention_head_aggregation": "arithmetic_mean_before_query_sum",
                            "feature_measurement_timing": "same_unintervened_forward_pass",
                            **intervention,
                            "focal_clt_input_norm": float(
                                raw[0, focal_token_index].norm().item()
                            ),
                            "feature_match_id": profile["feature_match_id"],
                            **metadata,
                        }
                    )
    return normalize_measurement_rows(rows, variants)


def _hash_map(value: Any, *, keys: set[int], label: str) -> dict[int, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    try:
        normalized_keys = {int(seed): digest for seed, digest in value.items()}
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} seed keys must be integers") from error
    if len(normalized_keys) != len(value) or set(normalized_keys) != keys:
        raise ValueError(f"{label} must contain exactly seeds {sorted(keys)}")
    return {
        seed: _require_sha256(digest, f"{label} seed {seed}")
        for seed, digest in normalized_keys.items()
    }


def _load_eligible_dictionary(
    receipt_spec: Mapping[str, Any],
    dictionary_spec: Mapping[str, Any],
    *,
    base: Path,
    model_name: str,
    device: str,
    requested_layers: Sequence[int],
    dictionary_loader: Callable | None,
):
    receipt = _require_keys(
        dict(receipt_spec), {"path", "sha256"}, "P0-2 gate receipt"
    )
    receipt_path = _resolve(receipt["path"], base)
    receipt_sha = _require_sha256(receipt["sha256"], "P0-2 gate receipt SHA-256")
    if not receipt_path.is_file() or sha256_file(receipt_path) != receipt_sha:
        raise ValueError("P0-2 gate receipt SHA-256 mismatch")
    dictionary = _require_keys(
        dict(dictionary_spec),
        {
            "run_seed",
            "checkpoint_path",
            "run_manifest_sha256_by_seed",
            "checkpoint_sha256_by_seed",
            "source_manifest_sha256_by_split",
        },
        "eligible TopK dictionary",
    )
    run_seed = dictionary["run_seed"]
    if type(run_seed) is not int or run_seed not in {17, 29, 43}:
        raise ValueError("eligible TopK run_seed must be 17, 29 or 43")
    run_manifests = _hash_map(
        dictionary["run_manifest_sha256_by_seed"],
        keys={17, 29, 43},
        label="run manifest SHA-256 map",
    )
    checkpoints = _hash_map(
        dictionary["checkpoint_sha256_by_seed"],
        keys={17, 29, 43},
        label="checkpoint SHA-256 map",
    )
    source_hashes = dictionary["source_manifest_sha256_by_split"]
    if (
        not isinstance(source_hashes, Mapping)
        or set(source_hashes) != {"train", "validation", "test"}
    ):
        raise ValueError("source manifest SHA-256 map requires train/validation/test")
    source_hashes = {
        split: _require_sha256(value, f"{split} source manifest SHA-256")
        for split, value in source_hashes.items()
    }
    checkpoint_path = _resolve(dictionary["checkpoint_path"], base)
    if checkpoint_path.name != "best.pt":
        raise ValueError(
            "eligible TopK checkpoint_path must name the exact-cache best.pt, "
            "not an online clt.pt"
        )
    if dictionary_loader is None:
        from src.revision.dictionary_gate import load_eligible_topk_clt

        dictionary_loader = load_eligible_topk_clt
    clt, provenance = dictionary_loader(
        receipt_path,
        receipt_sha,
        model_name=model_name,
        run_seed=run_seed,
        checkpoint_path=checkpoint_path,
        expected_run_manifest_sha256_by_seed=run_manifests,
        expected_checkpoint_sha256_by_seed=checkpoints,
        expected_source_manifest_sha256_by_split=source_hashes,
        requested_layers=list(requested_layers),
        map_location=device,
    )
    required_provenance = {
        "schema_version",
        "model_name",
        "method",
        "run_seed",
        "eligibility_receipt_sha256",
        "profile_sha256",
        "protocol_sha256",
        "run_manifest_sha256",
        "checkpoint_sha256",
        "candidate_id",
        "checkpoint_step",
        "geometry",
        "eligible_downstream_layers",
        "source_manifest_sha256_by_split",
    }
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != required_provenance
        or provenance["schema_version"] != "r2_p0_2_eligible_topk_load_v1"
        or provenance["model_name"] != model_name
        or provenance["method"] != "topk_clt"
        or provenance["run_seed"] != run_seed
        or provenance["eligibility_receipt_sha256"] != receipt_sha
        or provenance["run_manifest_sha256"] != run_manifests[run_seed]
        or provenance["checkpoint_sha256"] != checkpoints[run_seed]
        or provenance["source_manifest_sha256_by_split"] != source_hashes
        or not set(requested_layers) <= set(provenance["eligible_downstream_layers"])
    ):
        raise ValueError("loaded dictionary provenance does not match the P0-2 receipt")
    return clt, dict(provenance)


def _resource_snapshot(device: str) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "device": device,
    }
    if device.startswith("cuda"):
        hardware = torch.device(device)
        free, total = torch.cuda.mem_get_info(hardware)
        snapshot.update(
            {
                "accelerator": torch.cuda.get_device_name(hardware),
                "accelerator_free_bytes": int(free),
                "accelerator_total_bytes": int(total),
            }
        )
    return snapshot


def _run_extractor_into(
    spec_path: Path,
    spec_sha256: str,
    output_dir: Path,
    *,
    mode: str,
    model_loader: Callable | None = None,
    dictionary_loader: Callable | None = None,
    command: Sequence[str] | None = None,
) -> Path:
    """Execute one already-scoped extraction inside a private staging directory."""

    spec_path = Path(spec_path).resolve()
    expected_spec_hash = _require_sha256(spec_sha256, "spec SHA-256")
    if sha256_file(spec_path) != expected_spec_hash:
        raise ValueError("spec SHA-256 mismatch")
    spec = _require_keys(load_json(spec_path), SPEC_FIELDS, "extractor spec")
    if spec["schema_version"] != 3:
        raise ValueError("unsupported extractor spec schema_version")
    if not isinstance(spec["code_revision"], str) or not spec["code_revision"].strip():
        raise ValueError("code_revision must be non-empty")
    if type(spec["max_model_tokens"]) is not int or spec["max_model_tokens"] < 8:
        raise ValueError("max_model_tokens must be an integer of at least eight")
    if not isinstance(spec["device"], str) or not spec["device"]:
        raise ValueError("device must be non-empty")
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_dir() or any(output_dir.iterdir()):
        raise RuntimeError("private extractor staging directory must exist and be empty")
    started = datetime.now(timezone.utc).isoformat()
    base = spec_path.parent

    cohorts = _require_keys(
        spec["cohorts"], {"target", "protein_control_pool", "discovery"}, "cohorts"
    )
    targets, target_provenance = load_protein_manifest(
        cohorts["target"],
        base=base,
        label="target cohort",
        require_focal_position=True,
        require_mxx_start=True,
    )
    controls, control_provenance = load_protein_manifest(
        cohorts["protein_control_pool"],
        base=base,
        label="protein control pool",
        require_focal_position=True,
        require_mxx_start=True,
    )
    discovery, discovery_provenance = load_protein_manifest(
        cohorts["discovery"],
        base=base,
        label="discovery cohort",
        require_focal_position=False,
        require_mxx_start=False,
    )
    disjointness = validate_three_way_disjoint(targets, controls, discovery)

    counterfactual = _require_keys(
        spec["counterfactual"], {"internal_fraction"}, "counterfactual"
    )
    if float(counterfactual["internal_fraction"]) != 0.55:
        raise ValueError(
            "internal_fraction must equal the analyzer's frozen value 0.55"
        )
    internal_fraction = float(counterfactual["internal_fraction"])
    validate_focal_position_contract(targets, internal_fraction=internal_fraction)
    validate_focal_position_contract(controls, internal_fraction=internal_fraction)

    protein_matching = _require_keys(
        spec["protein_matching"],
        {"max_length_difference", "max_normalized_position_difference"},
        "protein_matching",
    )
    protein_pairs = match_protein_controls(targets, controls, **protein_matching)
    control_lookup = {row["protein_id"]: row for row in controls}
    selected_controls = [
        control_lookup[pair["control_protein_id"]] for pair in protein_pairs
    ]
    natural_records = [*targets, *selected_controls]
    source_records = {row["protein_id"]: row for row in natural_records}
    protein_metadata: dict[str, dict[str, Any]] = {}
    for pair in protein_pairs:
        common = {
            "protein_pair_id": pair["protein_pair_id"],
            "protein_match_absolute_length_difference": pair[
                "absolute_length_difference"
            ],
            "protein_match_absolute_normalized_position_difference": pair[
                "absolute_normalized_position_difference"
            ],
            "protein_match_max_length_difference": pair["max_length_difference"],
            "protein_match_max_normalized_position_difference": pair[
                "max_normalized_position_difference"
            ],
        }
        protein_metadata[pair["target_protein_id"]] = {
            **common,
            "protein_match_role": "target",
            "matched_protein_id": pair["control_protein_id"],
            "protein_match_focal_position": pair["target_focal_position"],
            "protein_match_normalized_position": pair["target_normalized_position"],
        }
        protein_metadata[pair["control_protein_id"]] = {
            **common,
            "protein_match_role": "control",
            "matched_protein_id": pair["target_protein_id"],
            "protein_match_focal_position": pair["control_focal_position"],
            "protein_match_normalized_position": pair["control_normalized_position"],
        }

    model_spec = _require_keys(
        spec["model"],
        {
            "name",
            "model_root",
            "model_revision",
            "tokenizer_revision",
            "model_artifacts",
            "input_format",
            "model_inference_dtype",
            "model_inference_dtype_verification",
            "activation_finiteness_check",
            "bos_construction",
        },
        "model",
    )
    for field in ("name", "model_revision", "tokenizer_revision"):
        if not isinstance(model_spec[field], str) or not model_spec[field].strip():
            raise ValueError(f"model {field} must be non-empty")
    if model_spec["input_format"] not in {"sequence", "zymctrl_ec"}:
        raise ValueError("input_format must be sequence or zymctrl_ec")
    if (
        model_spec["model_inference_dtype"] != "bfloat16"
        or model_spec["model_inference_dtype_verification"]
        != INFERENCE_DTYPE_VERIFICATION
        or model_spec["activation_finiteness_check"] != ACTIVATION_FINITE_CHECK
    ):
        raise ValueError(
            "production extraction requires the frozen bfloat16 inference and "
            "activation-finiteness contract"
        )
    if model_spec["bos_construction"] not in {
        "tokenizer_native_leading_bos",
        "explicit_prepend_tokenizer_bos",
    }:
        raise ValueError("unknown frozen BOS construction")
    if model_spec["input_format"] == "zymctrl_ec" and any(
        not isinstance(row.get("family"), str) or not row["family"]
        for row in natural_records
    ):
        raise ValueError(
            "zymctrl_ec inputs require a non-empty family for every protein"
        )
    model_root = _resolve(model_spec["model_root"], base)
    verified_model_artifacts = verify_model_artifacts(
        model_root, model_spec["model_artifacts"]
    )

    profile_descriptor = _require_keys(
        spec["feature_profiles"],
        {"path", "sha256", "discovery_cohort_sha256"},
        "feature_profiles",
    )
    profile_path = _resolve(profile_descriptor["path"], base)
    profile_hash = _require_sha256(
        profile_descriptor["sha256"], "feature profile SHA-256"
    )
    if sha256_file(profile_path) != profile_hash:
        raise ValueError("feature profile SHA-256 mismatch")
    if (
        _require_sha256(
            profile_descriptor["discovery_cohort_sha256"],
            "feature profile discovery cohort SHA-256",
        )
        != discovery_provenance["sha256"]
    ):
        raise ValueError(
            "feature profiles are not bound to the supplied discovery cohort"
        )
    feature_matching = _require_keys(
        spec["feature_matching"],
        {
            "control_count",
            "max_abs_log10_firing_ratio",
            "max_abs_log_input_norm_ratio",
        },
        "feature_matching",
    )
    selected_features, feature_matches = load_and_match_features(
        load_jsonl(profile_path), model_name=model_spec["name"], **feature_matching
    )

    variants = build_counterfactual_variants(
        natural_records, internal_fraction=internal_fraction
    )
    selected_layers = sorted({int(row["layer"]) for row in selected_features})
    clt, checkpoint_provenance = _load_eligible_dictionary(
        spec["p0_2_gate_receipt"],
        spec["dictionary"],
        base=base,
        model_name=model_spec["name"],
        device=spec["device"],
        requested_layers=selected_layers,
        dictionary_loader=dictionary_loader,
    )
    if model_loader is None:
        from src.models.model_loader import load_model

        model_loader = load_model
    protein_model = model_loader(
        str(model_root),
        device=spec["device"],
        dtype=inference_dtype(model_spec["model_inference_dtype"]),
    )
    dtype_receipt = verify_frozen_model_inference_dtype(
        protein_model, model_spec["model_inference_dtype"]
    )
    if protein_model.model.training:
        raise RuntimeError("pretrained model must be in evaluation mode")
    rows = extract_measurement_rows(
        protein_model=protein_model,
        clt=clt,
        model_name=model_spec["name"],
        model_revision=model_spec["model_revision"],
        tokenizer_revision=model_spec["tokenizer_revision"],
        input_format=model_spec["input_format"],
        bos_construction=model_spec["bos_construction"],
        max_model_tokens=spec["max_model_tokens"],
        variants=variants,
        source_records=source_records,
        selected_features=selected_features,
        protein_metadata=protein_metadata,
    )
    numerical_integrity = {
        **dtype_receipt,
        "activation_finiteness_check": ACTIVATION_FINITE_CHECK,
        "activation_finiteness_verified": True,
    }

    cohort_path = output_dir / "natural_cohort.jsonl"
    variants_path = output_dir / "counterfactual_variants.jsonl"
    protein_matches_path = output_dir / "protein_matches.jsonl"
    feature_matches_path = output_dir / "feature_matches.json"
    measurements_path = output_dir / "measurements.jsonl"
    write_jsonl(
        cohort_path,
        (
            {
                "protein_id": row["protein_id"],
                "sequence": row["sequence"],
                "sequence_sha256": row["sha256"],
                "focal_position": row["focal_position"],
                **protein_metadata[row["protein_id"]],
            }
            for row in natural_records
        ),
    )
    write_jsonl(variants_path, (variant.to_dict() for variant in variants))
    write_jsonl(protein_matches_path, protein_pairs)
    write_json(feature_matches_path, feature_matches)
    write_jsonl(measurements_path, rows)
    artifacts = [
        cohort_path,
        variants_path,
        protein_matches_path,
        feature_matches_path,
        measurements_path,
    ]
    summary_path = output_dir / "extraction_summary.json"
    write_json(
        summary_path,
        {
            "schema_version": "r2-p05-pretrained-extraction-summary-v4",
            "status": (
                "pretrained_measurements_extracted_analysis_not_run"
                if mode == "production"
                else "test_fixture_measurements_extracted_analysis_not_run"
            ),
            "execution_mode": mode,
            "claim_boundary": (
                "Measurement extraction alone does not establish an attention-sink "
                "mechanism or pass the P0-5 scientific gate."
            ),
            "model": model_spec["name"],
            "n_target_proteins": len(targets),
            "n_matched_protein_controls": len(selected_controls),
            "n_variants": len(variants),
            "n_features": len(selected_features),
            "n_measurement_rows": len(rows),
            "factorial": "protein x four_conditions x native_or_removed_BOS x feature",
            "attention_estimand": (
                "head_mean_received_mass_at_feature_layer / eligible_valid_causal_queries"
            ),
            "feature_measurement_timing": "same_unintervened_forward_pass",
            "attention_path_intervention": (
                "focal key masked across all layers for strict-suffix queries; "
                "suffix NLL/logit displacement; not formal feature mediation"
            ),
            "p0_2_dictionary_gate": {
                "receipt_sha256": checkpoint_provenance[
                    "eligibility_receipt_sha256"
                ],
                "run_seed": checkpoint_provenance["run_seed"],
                "checkpoint_sha256": checkpoint_provenance["checkpoint_sha256"],
                "eligible_layers": checkpoint_provenance[
                    "eligible_downstream_layers"
                ],
            },
            "protein_matching": protein_matching,
            "feature_matching": feature_matching,
            "disjointness": disjointness,
            **numerical_integrity,
        },
    )
    artifacts.append(summary_path)

    source_root = Path(__file__).resolve().parents[2]
    manifest_path = output_dir / "run_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": "r2-p05-pretrained-extraction-manifest-v4",
            "status": (
                "verified_production_complete"
                if mode == "production"
                else "verified_test_fixture_complete"
            ),
            "execution_mode": mode,
            "command": list(command or [sys.executable]),
            "started_at_utc": started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "spec": {"path": str(spec_path), "sha256": expected_spec_hash},
            "code_revision": spec["code_revision"],
            "inputs": {
                "target_cohort": target_provenance,
                "protein_control_pool": control_provenance,
                "discovery_cohort": discovery_provenance,
                "feature_profiles": {
                    "path": str(profile_path),
                    "sha256": profile_hash,
                    "discovery_cohort_sha256": profile_descriptor[
                        "discovery_cohort_sha256"
                    ],
                },
            },
            "model": {
                "name": model_spec["name"],
                "root": str(model_root),
                "revision": model_spec["model_revision"],
                "tokenizer_revision": model_spec["tokenizer_revision"],
                "artifacts": verified_model_artifacts,
                "geometry": {
                    "n_layers": protein_model.n_layers,
                    "d_model": protein_model.d_model,
                },
                "input_format": model_spec["input_format"],
                "bos_construction": model_spec["bos_construction"],
                **numerical_integrity,
            },
            "eligible_dictionary": checkpoint_provenance,
            "matching": {
                "protein": protein_matching,
                "feature": feature_matching,
                "protein_pair_ids": [pair["protein_pair_id"] for pair in protein_pairs],
                "feature_match_ids": [
                    row["feature_match_id"] for row in feature_matches
                ],
            },
            "environment": _resource_snapshot(spec["device"]),
            "source_hashes": {
                "n_terminal_extractor.py": sha256_file(Path(__file__)),
                "n_terminal_counterfactuals.py": sha256_file(
                    source_root / "src/revision/n_terminal_counterfactuals.py"
                ),
                "model_loader.py": sha256_file(
                    source_root / "src/models/model_loader.py"
                ),
                "clt_trainer.py": sha256_file(
                    source_root / "src/training/clt_trainer.py"
                ),
                "dictionary_gate.py": sha256_file(
                    source_root / "src/revision/dictionary_gate.py"
                ),
                "65_extract_n_terminal_measurements.py": sha256_file(
                    source_root / "scripts/65_extract_n_terminal_measurements.py"
                ),
            },
            "artifact_hashes": {path.name: sha256_file(path) for path in artifacts},
        },
    )
    return manifest_path


def run_extractor(
    spec_path: Path,
    spec_sha256: str,
    output_dir: Path,
    *,
    mode: str,
    model_loader: Callable | None = None,
    dictionary_loader: Callable | None = None,
    command: Sequence[str] | None = None,
) -> Path:
    """Run one scoped P0-5 extraction and atomically publish its directory.

    Production runs must use the built-in pretrained-model and eligible-TopK
    loaders.  Loader injection is reserved for an explicitly labelled test
    fixture, so fixture artifacts cannot masquerade as production evidence.
    """

    if mode not in {"production", "test_fixture"}:
        raise ValueError("mode must be production or test_fixture")
    overrides = (model_loader, dictionary_loader)
    if mode == "production" and any(loader is not None for loader in overrides):
        raise ValueError("loader overrides are forbidden in production mode")
    if mode == "test_fixture" and any(loader is None for loader in overrides):
        raise ValueError("test_fixture mode requires both explicit loader overrides")

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite extractor output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"stale extractor staging directory: {staging}")
    staging.mkdir()
    try:
        manifest = _run_extractor_into(
            spec_path,
            spec_sha256,
            staging,
            mode=mode,
            model_loader=model_loader,
            dictionary_loader=dictionary_loader,
            command=command,
        )
        os.replace(staging, destination)
        return destination / manifest.name
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
