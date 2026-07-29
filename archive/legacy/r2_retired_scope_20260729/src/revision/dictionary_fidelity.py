"""Behavioral fidelity for completed windowed dictionary checkpoints.

This module evaluates one prespecified MLP target layer. Earlier source-layer
MLP inputs are captured from the live forward pass, encoded with the completed
windowed transcoder, and decoded only onto the target MLP output. Downstream
model computation is then allowed to recompute normally.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .dictionary_controls import WindowedTranscoder, load_strict_json
from .io import sha256_file


MODES = ("clean", "reinject", "mean_ablate", "dictionary")


def analysis_layer(n_layers: int, fraction: float) -> int:
    if n_layers < 1 or not 0 <= fraction <= 1:
        raise ValueError("invalid layer count or analysis-layer fraction")
    return int(math.floor(fraction * (n_layers - 1) + 0.5))


def source_layers_for_target(
    target_layer: int, *, n_layers: int, window: int
) -> tuple[int, ...]:
    if not 0 <= target_layer < n_layers or window < 1:
        raise ValueError("invalid target layer or decoder window")
    return tuple(range(max(0, target_layer - window + 1), target_layer + 1))


def encode_source(
    dictionary: WindowedTranscoder,
    source_layer: int,
    values: torch.Tensor,
    *,
    activation_threshold: float,
) -> torch.Tensor:
    """Encode one absolute source layer using the trained method."""

    if (
        not 0 <= source_layer < dictionary.n_layers
        or values.ndim != 2
        or values.shape[1] != dictionary.input_dim
        or activation_threshold < 0
    ):
        raise ValueError("invalid source-layer encoding request")
    direction = F.linear(values, dictionary.encoder_weight[source_layer])
    preactivation = direction + dictionary.encoder_bias[source_layer]
    if dictionary.method == "topk_clt":
        positive = F.relu(preactivation)
        top_values, top_indices = torch.topk(
            positive, dictionary.topk_k, dim=1, sorted=False
        )
        code = torch.zeros_like(positive).scatter(1, top_indices, top_values)
    elif dictionary.method == "relu_l1_sae":
        code = F.relu(preactivation)
    elif dictionary.method == "gated_sae":
        magnitude = (
            direction * dictionary.log_magnitude_scale[source_layer].exp()
            + dictionary.magnitude_bias[source_layer]
        )
        code = (preactivation > 0).to(magnitude.dtype) * F.relu(magnitude)
    elif dictionary.method == "dense_low_rank":
        code = preactivation
    else:  # pragma: no cover - constructor rejects unknown methods
        raise AssertionError("unknown dictionary method")
    if dictionary.sparse and activation_threshold:
        code = torch.where(code > activation_threshold, code, torch.zeros_like(code))
    return code


def reconstruct_target(
    dictionary: WindowedTranscoder,
    captured_inputs: Mapping[int, torch.Tensor],
    *,
    target_layer: int,
    activation_threshold: float,
) -> torch.Tensor:
    """Decode live source-layer inputs onto one target MLP output."""

    required = source_layers_for_target(
        target_layer, n_layers=dictionary.n_layers, window=dictionary.window
    )
    if set(captured_inputs) != set(required):
        raise ValueError("captured input layers do not match the target window")
    shapes = {tuple(captured_inputs[layer].shape) for layer in required}
    if len(shapes) != 1:
        raise ValueError("captured source inputs have inconsistent shapes")
    shape = shapes.pop()
    if len(shape) != 3 or shape[-1] != dictionary.input_dim:
        raise ValueError("captured source inputs must have shape [batch, token, d]")
    batch, tokens, _ = shape
    dtype = dictionary.encoder_weight.dtype
    device = dictionary.encoder_weight.device
    output = dictionary.decoder_bias[target_layer].expand(batch * tokens, -1).clone()
    for source in required:
        values = (
            captured_inputs[source]
            .reshape(batch * tokens, -1)
            .to(device=device, dtype=dtype)
        )
        code = encode_source(
            dictionary,
            source,
            values,
            activation_threshold=activation_threshold,
        )
        offset = target_layer - source
        output.add_(code @ dictionary.decoder_weight[source][:, offset, :])
    return output.reshape(batch, tokens, dictionary.target_dim)


def sequence_target_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    model_name: str,
    start_token_id: int | None = None,
    end_token_id: int | None = None,
) -> torch.Tensor:
    """Return a mask over next-token targets belonging to the protein sequence."""

    if (
        input_ids.ndim != 2
        or attention_mask.shape != input_ids.shape
        or input_ids.shape[1] < 2
    ):
        raise ValueError("invalid token or attention-mask shape")
    valid = attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
    if model_name != "zymctrl":
        return valid
    if start_token_id is None or end_token_id is None:
        raise ValueError("ZymCTRL scoring requires start and end token IDs")
    result = torch.zeros_like(valid)
    for row in range(input_ids.shape[0]):
        ids = input_ids[row]
        starts = torch.nonzero(ids == start_token_id, as_tuple=False).flatten()
        ends = torch.nonzero(ids == end_token_id, as_tuple=False).flatten()
        if starts.numel() != 1 or ends.numel() != 1 or ends[0] <= starts[0] + 1:
            raise ValueError("ZymCTRL prompt lacks one valid start/end boundary")
        # Target column q predicts input token q+1. Keep amino-acid targets
        # strictly after <start> and strictly before <end>.
        result[row, int(starts[0]) : int(ends[0]) - 1] = True
    return result & valid


def per_sequence_scores(
    clean_logits: torch.Tensor,
    variant_logits: torch.Tensor,
    input_ids: torch.Tensor,
    target_mask: torch.Tensor,
) -> list[dict[str, float | int]]:
    if (
        clean_logits.shape != variant_logits.shape
        or clean_logits.ndim != 3
        or input_ids.shape != clean_logits.shape[:2]
        or target_mask.shape != (input_ids.shape[0], input_ids.shape[1] - 1)
    ):
        raise ValueError("logit, token and target-mask shapes disagree")
    clean_logp = F.log_softmax(clean_logits[:, :-1].float(), dim=-1)
    variant_logp = F.log_softmax(variant_logits[:, :-1].float(), dim=-1)
    clean_p = clean_logp.exp()
    targets = input_ids[:, 1:]
    clean_nll = -clean_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    variant_nll = -variant_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    kl = (clean_p * (clean_logp - variant_logp)).sum(-1)
    agreement = clean_logp.argmax(-1) == variant_logp.argmax(-1)
    rows: list[dict[str, float | int]] = []
    for index in range(input_ids.shape[0]):
        mask = target_mask[index]
        count = int(mask.sum())
        if count < 1:
            raise ValueError("sequence has no scored next-token targets")
        rows.append(
            {
                "token_count": count,
                "clean_nll_sum": float(clean_nll[index][mask].sum()),
                "variant_nll_sum": float(variant_nll[index][mask].sum()),
                "kl_sum": float(kl[index][mask].sum()),
                "argmax_agreement_count": int(agreement[index][mask].sum()),
            }
        )
    return rows


def aggregate_variant(rows: Sequence[Mapping[str, float | int]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty sequence set")
    tokens = sum(int(row["token_count"]) for row in rows)
    if tokens < 1:
        raise ValueError("aggregate contains no scored targets")
    return {
        "clean_ce_nats": sum(float(row["clean_nll_sum"]) for row in rows) / tokens,
        "variant_ce_nats": sum(float(row["variant_nll_sum"]) for row in rows) / tokens,
        "clean_to_variant_kl_nats": sum(float(row["kl_sum"]) for row in rows) / tokens,
        "argmax_agreement": sum(int(row["argmax_agreement_count"]) for row in rows)
        / tokens,
        "scored_tokens": tokens,
        "sequences": len(rows),
    }


def fidelity_metrics(
    dictionary_rows: Sequence[Mapping[str, float | int]],
    mean_rows: Sequence[Mapping[str, float | int]],
    *,
    minimum_ce_denominator: float,
    minimum_kl_denominator: float,
) -> dict[str, float | bool]:
    if len(dictionary_rows) != len(mean_rows):
        raise ValueError("dictionary and mean-ablation sequence sets differ")
    dictionary = aggregate_variant(dictionary_rows)
    mean = aggregate_variant(mean_rows)
    if not math.isclose(
        dictionary["clean_ce_nats"], mean["clean_ce_nats"], abs_tol=1e-9
    ):
        raise ValueError("clean reference changed between intervention modes")
    ce_denominator = mean["variant_ce_nats"] - mean["clean_ce_nats"]
    kl_denominator = mean["clean_to_variant_kl_nats"]
    denominators_valid = (
        ce_denominator >= minimum_ce_denominator
        and kl_denominator >= minimum_kl_denominator
    )
    loss_recovered = (
        (mean["variant_ce_nats"] - dictionary["variant_ce_nats"]) / ce_denominator
        if denominators_valid
        else None
    )
    kl_recovered = (
        1.0 - dictionary["clean_to_variant_kl_nats"] / kl_denominator
        if denominators_valid
        else None
    )
    return {
        "denominators_valid": denominators_valid,
        "mean_ablation_ce_delta_nats": ce_denominator,
        "mean_ablation_kl_nats": kl_denominator,
        "loss_recovered": loss_recovered,
        "kl_recovered": kl_recovered,
    }


def cluster_bootstrap(
    dictionary_rows: Sequence[Mapping[str, float | int]],
    mean_rows: Sequence[Mapping[str, float | int]],
    *,
    samples: int,
    seed: int,
    minimum_ce_denominator: float,
    minimum_kl_denominator: float,
) -> dict[str, Any]:
    if len(dictionary_rows) != len(mean_rows) or not dictionary_rows:
        raise ValueError("bootstrap rows must be non-empty and paired")
    if samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    generator = np.random.default_rng(seed)
    loss: list[float] = []
    kl: list[float] = []
    invalid = 0
    n = len(dictionary_rows)
    for _ in range(samples):
        indices = generator.integers(0, n, size=n)
        metrics = fidelity_metrics(
            [dictionary_rows[int(index)] for index in indices],
            [mean_rows[int(index)] for index in indices],
            minimum_ce_denominator=minimum_ce_denominator,
            minimum_kl_denominator=minimum_kl_denominator,
        )
        if not metrics["denominators_valid"]:
            invalid += 1
            continue
        loss.append(float(metrics["loss_recovered"]))
        kl.append(float(metrics["kl_recovered"]))

    def interval(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        return {
            "q025": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "q975": float(np.quantile(values, 0.975)),
        }

    return {
        "schema_version": "r2_dictionary_fidelity_cluster_bootstrap_v1",
        "cluster_unit": "sequence",
        "samples": samples,
        "seed": seed,
        "invalid_denominator_samples": invalid,
        "loss_recovered": interval(loss),
        "kl_recovered": interval(kl),
    }


def checkpoint_state(
    checkpoint_path: Path,
    *,
    expected_sha256: str,
    expected_candidate_id: str,
) -> dict[str, Any]:
    if sha256_file(checkpoint_path) != expected_sha256:
        raise ValueError("dictionary checkpoint SHA-256 mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != "r2_dictionary_control_best_v1"
        or payload.get("candidate_id") != expected_candidate_id
        or not isinstance(payload.get("model_state_dict"), Mapping)
    ):
        raise ValueError("dictionary checkpoint identity changed")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at line {line_number}: {path}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row at line {line_number}: {path}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty JSONL file: {path}")
    return rows


def verify_artifact(descriptor: Mapping[str, Any], label: str) -> Path:
    if set(descriptor) != {"path", "sha256"}:
        raise ValueError(f"{label} descriptor fields changed")
    path = Path(descriptor["path"]).resolve()
    digest = descriptor["sha256"]
    if (
        not path.is_file()
        or not isinstance(digest, str)
        or len(digest) != 64
        or sha256_file(path) != digest
    ):
        raise ValueError(f"{label} is missing or changed")
    return path


def hash_payload(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_model_artifacts(root: Path, expected: Mapping[str, str]) -> dict[str, str]:
    """Verify the deployed config, weight tree, and tokenizer/support tree."""

    root = Path(root).resolve()
    config = root / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"missing model config: {config}")
    files = [path for path in root.rglob("*") if path.is_file()]
    weights = [
        path
        for path in files
        if path.suffix == ".safetensors"
        or path.name.startswith("pytorch_model")
        and path.suffix in {".bin", ".json"}
        or path.name.startswith("model.safetensors")
    ]
    support = [path for path in files if path != config and path not in weights]

    def tree_digest(paths: Sequence[Path]) -> str:
        if not paths:
            raise ValueError(f"empty model-artifact class under {root}")
        return hash_payload(
            [
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(paths)
            ]
        )

    observed = {
        "model_config_sha256": sha256_file(config),
        "model_weights_sha256": tree_digest(weights),
        "tokenizer_sha256": tree_digest(support),
    }
    if dict(expected) != observed:
        raise ValueError("deployed base-model artifacts changed")
    return observed


def load_fidelity_spec(path: Path) -> dict[str, Any]:
    payload = load_strict_json(path)
    required = {
        "schema_version",
        "status",
        "confirmatory_scope",
        "p0_2_gate_spec",
        "p0_2_eligibility_receipt",
        "profile",
        "protocol",
        "implementation",
        "evaluation_cohort",
        "unavailable_model_methods",
        "target_means",
        "model_artifacts_by_model",
        "model_input_format_by_model",
        "analysis_layer_fraction",
        "tokenization",
        "scoring",
        "bootstrap",
        "fidelity_gate",
        "runs",
    }
    if set(payload) != required:
        raise ValueError("dictionary-fidelity spec fields changed")
    if (
        payload["schema_version"] != "r2_dictionary_fidelity_spec_v1"
        or payload["status"] != "frozen_before_evaluation"
        or payload["confirmatory_scope"]
        != "prospective_new_cohort_instrument_qualification_not_p0_2_regating"
    ):
        raise ValueError("dictionary-fidelity spec is not the frozen amendment")
    if set(payload["implementation"]) != {
        "runner",
        "fidelity_module",
        "dictionary_module",
        "model_loader",
    }:
        raise ValueError("dictionary-fidelity implementation binding changed")
    return payload
