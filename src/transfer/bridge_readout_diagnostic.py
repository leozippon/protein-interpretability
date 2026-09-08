"""Read-only val readout diagnostic for frozen latent bridges and classifier heads.

Direction-1 follow-up: teacher-forced answer-token NLL split, free-greedy JSON
accuracy, and direct classifier scores on the already-used 512-row val split.
No training, no test forward or scoring, no seed picking, and not M2.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transfer.classifier_handoff import (
    ALLOWED_HEADS,
    bound_pretrained_cache,
    cell_relative_dir,
    classifier_direct_rows,
    predict_classes,
    split_cache_features,
)
from src.transfer.io import sha256_file, write_json
from src.transfer.latent_bridge import (
    BRIDGE_SCHEMA,
    CLASS_IDS,
    IGNORE_INDEX,
    PreparedRecord,
    _answer_ids,
    bridge_identity,
    encode_prompt,
    freeze_module,
    generate_predictions,
    inject_soft_prefix,
    load_bridge,
    pad_rows,
    parameter_fingerprint,
    prediction_metrics,
    shifted_cross_entropy,
    target_json,
    user_prompt,
    validate_bridge_identity,
)

DIAGNOSTIC_SCHEMA = "bridge_readout_diagnostic_v1"
REPORT_SCHEMA = "bridge_readout_report_v1"
PILOT_KIND = "val_readout_diagnostic"
PILOT_NOTE = (
    "Read-only validation diagnostic on frozen Phase-A checkpoints. "
    "Train/val families overlap, so val scores are not new generalization "
    "evidence. No test forward, scoring, or seed selection. Not M2."
)
EVALUATION_CLAIM = (
    "diagnostic reuse of the already-used prepared_v1 val split; not an "
    "independent confirmatory test"
)
SUBSTAGES = ("interface", "run")
CONDITION = "pretrained_donor"
DEFAULT_SEEDS = (20260908, 20260909, 20260910)
DEFAULT_HEADS = ("linear", "mlp")
DEFAULT_BRIDGE_LABELS = (
    "train_ext_pt_s20260908",
    "train_ext_pt_s20260909",
    "train_ext_pt_s20260910",
)
DEFAULT_BUDGET = 3_000_000
DEFAULT_PROMPT_MODE = "chat"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_PROMPT_TOKENS = 1024
DEFAULT_MAX_NEW_TOKENS = 64
MAX_INTERFACE_RECORDS = 32
INTERFACE_MECHANICAL_RECORDS = 2
CAUSAL_ATOL = 1e-5
CAUSAL_RTOL = 1e-5
RECONSTRUCT_ATOL = 1e-5
RECONSTRUCT_RTOL = 1e-5
PRODUCTION_BRIDGE_PIN = "acb254abc74c4bb9f2fc5ac09ee963ea2a58192f"
PRODUCTION_CLASSIFIER_PIN = "ee3d8e199d56f4f1c1a6531e17064aefdf8d6f6a"
PRODUCTION_BRIDGE_RUN = "20260907112059_d18a3093e3d3"
PRODUCTION_CLASSIFIER_RUN = "20260908004829_4fd47698fd36"
PRODUCTION_CLASSIFIER_STAGE = "49_classifier_handoff.py"
PRODUCTION_PREPARED_SHA256 = "8335e4cee35991c4993bda86eb1ad2052c8f3d462a40a7ba3c5d869f623d06c0"
PRODUCTION_RECEIVER_CONTENT_DIGEST = (
    "f8331098b7d2bb09151a2ab1dfaf7403d6f4e429a285ada3aad757cfd18d6188"
)
LOSS_PATH_NOTE = (
    "legacy_native calls shifted_cross_entropy on the receiver's native logits "
    "dtype and token-weights like train_bridge. fp32_unreduced upcasts logits "
    "and splits digit vs remainder/EOS, summing FP32 token CE in float64/Python. "
    "The two paths are not required to be "
    "bit-equal. Full-vocabulary digit CE is not a 7-way classifier CE."
)


def write_report_json(out: Path, payload: Mapping[str, Any]) -> str:
    """Write report.json through write_json and emit a sha256sum sidecar."""

    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "report.json"
    write_json(path, dict(payload))
    digest = sha256_file(path)
    (destination / "report.json.sha256").write_text(
        f"{digest}  report.json\n", encoding="utf-8"
    )
    return digest


def refuse_existing_output(out: Path) -> None:
    destination = Path(out)
    if destination.is_file():
        raise FileExistsError(f"refusing to overwrite existing file {destination}")
    if destination.is_dir() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing output {destination}")


def load_acceptance(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} is not a 64-hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} is not a 64-hex digest") from error
    return value


def validate_bridge_acceptance(
    payload: Mapping[str, Any],
    *,
    expected_pin: str,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    labels: Sequence[str] = DEFAULT_BRIDGE_LABELS,
) -> list[dict[str, Any]]:
    if payload.get("execution_pin") != expected_pin:
        raise ValueError("bridge acceptance execution_pin mismatch")
    if payload.get("run_id") != PRODUCTION_BRIDGE_RUN:
        raise ValueError("bridge acceptance run_id mismatch")
    replicas = payload.get("replicas")
    if not isinstance(replicas, list) or len(replicas) != len(seeds):
        raise ValueError("bridge acceptance replica matrix mismatch")
    expected = list(zip(seeds, labels, strict=True))
    actual: list[tuple[int, str]] = []
    cleaned: list[dict[str, Any]] = []
    for replica in replicas:
        if not isinstance(replica, dict):
            raise ValueError("bridge acceptance replica is not an object")
        seed = replica.get("seed")
        label = replica.get("label")
        if type(seed) is not int or not isinstance(label, str):
            raise ValueError("bridge acceptance seed/label missing")
        actual.append((seed, label))
        _require_sha256(replica.get("checkpoint_sha256"), "bridge checkpoint_sha256")
        loss = replica.get("best_val_loss")
        if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss):
            raise ValueError("bridge acceptance requires finite non-bool best_val_loss")
        cleaned.append(dict(replica))
    if len(set(actual)) != len(actual):
        raise ValueError("bridge acceptance has a duplicate seed")
    if actual != expected:
        raise ValueError("bridge acceptance seed/label matrix mismatch")
    return cleaned


def validate_classifier_acceptance(
    payload: Mapping[str, Any],
    *,
    expected_pin: str,
    heads: Sequence[str] = DEFAULT_HEADS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> list[dict[str, Any]]:
    if payload.get("execution_pin") != expected_pin:
        raise ValueError("classifier acceptance execution_pin mismatch")
    if payload.get("run_id") != PRODUCTION_CLASSIFIER_RUN:
        raise ValueError("classifier acceptance run_id mismatch")
    if payload.get("stage") != PRODUCTION_CLASSIFIER_STAGE:
        raise ValueError("classifier acceptance stage mismatch")
    if payload.get("prepared_sha256") != PRODUCTION_PREPARED_SHA256:
        raise ValueError("classifier acceptance prepared_sha256 mismatch")
    cells = payload.get("cells")
    expected = [(kind, int(seed)) for kind in heads for seed in seeds]
    if not isinstance(cells, list) or len(cells) != len(expected):
        raise ValueError("classifier acceptance cell matrix mismatch")
    actual: list[tuple[str, int]] = []
    cleaned: list[dict[str, Any]] = []
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("classifier acceptance cell is not an object")
        kind, seed = cell.get("kind"), cell.get("seed")
        if kind not in ALLOWED_HEADS or type(seed) is not int:
            raise ValueError("classifier acceptance has an invalid head or seed")
        if cell.get("dir") != cell_relative_dir(kind, seed):
            raise ValueError("classifier acceptance directory is not canonical")
        _require_sha256(cell.get("checkpoint_sha256"), "classifier checkpoint_sha256")
        actual.append((kind, seed))
        cleaned.append(dict(cell))
    if len(set(actual)) != len(actual):
        raise ValueError("classifier acceptance has a duplicate cell")
    if actual != expected:
        raise ValueError("classifier acceptance head/seed matrix mismatch")
    return cleaned


def require_checkpoint_sha256(path: Path, expected: str) -> str:
    digest = sha256_file(Path(path))
    if digest != expected:
        raise ValueError(f"checkpoint sha256 mismatch for {Path(path).name}")
    return digest


def require_finite_module(module: nn.Module, name: str) -> None:
    for parameter_name, tensor in module.state_dict().items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"non-finite {name} state: {parameter_name}")


def require_finite_features(features: np.ndarray, name: str = "features") -> None:
    if features.ndim != 2 or features.shape[0] < 1:
        raise ValueError(f"{name} must be a nonempty 2-D array")
    if not np.isfinite(features).all():
        raise ValueError(f"non-finite {name}")


def require_roles(records: Sequence[PreparedRecord], role: str, name: str) -> None:
    if not records:
        raise ValueError(f"{name} is empty")
    if any(record.role != role for record in records):
        raise ValueError(f"{name} contains a non-{role} record")


def require_positive_limits(**limits: int) -> None:
    for name, value in limits.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def validate_forward_inputs(
    features: np.ndarray, records: Sequence[PreparedRecord], batch_size: int
) -> None:
    """All public forwards are bounded train mechanics or homogeneous val only."""
    require_positive_limits(batch_size=batch_size)
    if not records or records[0].role not in {"train", "val"}:
        raise ValueError("diagnostic forwards require train/val records, never test")
    require_roles(records, records[0].role, "diagnostic records")
    if records[0].role == "train" and len(records) > MAX_INTERFACE_RECORDS:
        raise ValueError("diagnostic train forwards are limited to 32 rows")
    require_finite_features(features)
    if features.shape[0] != len(records):
        raise ValueError("diagnostic features and records are misaligned")


def bridge_prefix_width(bridge: nn.Module) -> int:
    k = getattr(bridge, "k", None)
    if type(k) is not int or k < 1:
        raise ValueError("bridge lacks a positive integer prefix width k")
    return k


def class_logit_index(label_position: int) -> int:
    """Causal index whose logits predict the class-digit label at ``t``."""

    if type(label_position) is not int or label_position < 1:
        raise ValueError("class-digit label position t must be an integer >= 1")
    return label_position - 1


def toy_off_by_one_proof() -> dict[str, Any]:
    """Asymmetric logits: reading t instead of t-1 yields the wrong digit."""

    label_position = 3
    logit_index = class_logit_index(label_position)
    logits = torch.zeros(1, 5, 8)
    logits[0, logit_index, 5] = 4.0
    logits[0, label_position, 7] = 9.0
    correct = int(logits[0, logit_index].argmax().item())
    off_by_one = int(logits[0, label_position].argmax().item())
    if correct == off_by_one:
        raise RuntimeError("toy logits are not an off-by-one detector")
    return {
        "label_position": label_position,
        "logit_index": logit_index,
        "correct_token": correct,
        "off_by_one_token": off_by_one,
        "detected": True,
    }


def class_digit_layout(tokenizer: Any, eos_id: int | None) -> dict[str, Any]:
    """Shared prefix plus one first-divergent token that decodes to 1-7."""

    answers = {
        class_id: _answer_ids(tokenizer, target_json(class_id), eos_id)
        for class_id in CLASS_IDS
    }
    sequences = [answers[class_id] for class_id in CLASS_IDS]
    prefix_len = 0
    while True:
        if any(prefix_len >= len(sequence) for sequence in sequences):
            break
        token = sequences[0][prefix_len]
        if any(sequence[prefix_len] != token for sequence in sequences[1:]):
            break
        prefix_len += 1
    if any(prefix_len >= len(sequence) for sequence in sequences):
        raise ValueError("canonical answers have no class-digit token after the shared prefix")
    digit_token_by_class: dict[int, int] = {}
    for class_id, sequence in zip(CLASS_IDS, sequences, strict=True):
        token = int(sequence[prefix_len])
        decoded = tokenizer.decode([token])
        if decoded != str(class_id):
            raise ValueError(
                f"first divergent token for class {class_id} decoded to {decoded!r}; "
                "refusing guessed digit ids or merged tokens"
            )
        digit_token_by_class[class_id] = token
    if len(set(digit_token_by_class.values())) != len(CLASS_IDS):
        raise ValueError("class-digit tokens are not distinct")
    return {
        "prefix_ids": list(sequences[0][:prefix_len]),
        "prefix_len": prefix_len,
        "digit_token_by_class": digit_token_by_class,
        "class_by_digit_token": {
            token: class_id for class_id, token in digit_token_by_class.items()
        },
        "answer_ids_by_class": answers,
    }


def shared_latent_prompt_ids(
    tokenizer: Any, *, prompt_mode: str, max_prompt_tokens: int
) -> list[int]:
    text = user_prompt(mode="latent", sequence=None, oracle_class=None)
    return encode_prompt(
        tokenizer, text, prompt_mode=prompt_mode, max_tokens=max_prompt_tokens
    )


def freeze_read_only(module: nn.Module) -> str:
    fingerprint = parameter_fingerprint(module)
    freeze_module(module)
    if parameter_fingerprint(module) != fingerprint:
        raise RuntimeError("freeze changed parameter bytes")
    if any(parameter.grad is not None for parameter in module.parameters()):
        raise RuntimeError("read-only freeze left a gradient")
    return fingerprint


def bound_splits(prepared: Mapping[str, Any], cache_path: Path) -> dict[str, Any]:
    features, manifest, train_rows, val_rows, test_rows = bound_pretrained_cache(
        cache_path, prepared
    )
    train_feat, val_feat, test_feat = split_cache_features(
        features, train_rows, val_rows, test_rows
    )
    del test_feat
    return {
        "manifest": manifest,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_test": len(test_rows),
        "train_features": train_feat,
        "val_features": val_feat,
        "test_records_loaded_for_binding": True,
        "test_features_forwarded": False,
    }


def select_interface_records(
    train_rows: Sequence[PreparedRecord],
    *,
    max_interface_records: int,
    mechanical_records: int = INTERFACE_MECHANICAL_RECORDS,
) -> dict[str, Any]:
    require_roles(train_rows, "train", "interface records")
    if max_interface_records < 1 or max_interface_records > MAX_INTERFACE_RECORDS:
        raise ValueError(
            f"interface may use 1..{MAX_INTERFACE_RECORDS} train records, "
            f"got {max_interface_records}"
        )
    selected = list(train_rows[:max_interface_records])
    n_mechanical = min(mechanical_records, len(selected))
    if n_mechanical < 1:
        raise ValueError("interface needs at least one train record")
    mechanical = selected[:n_mechanical]
    return {
        "selection": "first_train_records",
        "max_interface_records": int(max_interface_records),
        "n_selected": len(selected),
        "n_mechanical": n_mechanical,
        "selected_accessions": [row.accession for row in selected],
        "mechanical_accessions": [row.accession for row in mechanical],
        "selected": selected,
        "mechanical": mechanical,
    }


def qualify_bridges(
    *,
    bridge_root: Path,
    acceptance: Mapping[str, Any],
    expected_pin: str,
    prepared_hash: str,
    cache_manifest: Mapping[str, Any],
    receiver_digest: Mapping[str, Any],
    prompt_mode: str,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    labels: Sequence[str] = DEFAULT_BRIDGE_LABELS,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    replicas = validate_bridge_acceptance(
        acceptance, expected_pin=expected_pin, seeds=seeds, labels=labels
    )
    paths = [Path(bridge_root) / replica["label"] / "bridge.pt" for replica in replicas]
    for path, replica in zip(paths, replicas, strict=True):
        require_checkpoint_sha256(path, replica["checkpoint_sha256"])
    qualified: list[dict[str, Any]] = []
    for path, replica in zip(paths, replicas, strict=True):
        bridge, payload = load_bridge(path)
        require_finite_module(bridge, "bridge")
        config = dict(bridge.config())
        if config.get("schema_version") != BRIDGE_SCHEMA:
            raise ValueError("loaded bridge schema is not the frozen v2 format")
        expected = bridge_identity(
            condition=CONDITION,
            prepared_hash=prepared_hash,
            cache_manifest=cache_manifest,
            receiver_digest=receiver_digest,
            prompt_mode=prompt_mode,
            bridge_config=config,
            seed=int(replica["seed"]),
        )
        validate_bridge_identity(payload, expected)
        fingerprint = freeze_read_only(bridge.to(device))
        qualified.append(
            {
                "seed": int(replica["seed"]),
                "label": replica["label"],
                "checkpoint_sha256": replica["checkpoint_sha256"],
                "best_val_loss": float(replica["best_val_loss"]),
                "bridge": bridge,
                "config": config,
                "actual_parameters": int(config["actual_parameters"]),
                "fingerprint": fingerprint,
                "identity": expected,
            }
        )
    return qualified


def _class_from_digit_token(layout: Mapping[str, Any], token: int) -> int | None:
    mapped = layout["class_by_digit_token"].get(int(token))
    return int(mapped) if mapped is not None else None


def teacher_forced_decomposition(
    *,
    bridge: nn.Module,
    receiver: Any,
    tokenizer: Any,
    features: np.ndarray,
    records: Sequence[PreparedRecord],
    layout: Mapping[str, Any],
    prompt_mode: str,
    device: str,
    batch_size: int,
    max_prompt_tokens: int,
) -> dict[str, Any]:
    validate_forward_inputs(features, records, batch_size)
    require_positive_limits(max_prompt_tokens=max_prompt_tokens)
    freeze_module(receiver)
    freeze_module(bridge.to(device))
    embed = receiver.get_input_embeddings()
    pad_id = int(tokenizer.pad_token_id)
    eos_id = tokenizer.eos_token_id
    prompt_row = shared_latent_prompt_ids(
        tokenizer, prompt_mode=prompt_mode, max_prompt_tokens=max_prompt_tokens
    )
    k = bridge_prefix_width(bridge)
    answer_start = k + len(prompt_row)
    digit_t = answer_start + int(layout["prefix_len"])
    logit_index = class_logit_index(digit_t)
    per_record: list[dict[str, Any]] = []
    legacy_sum = 0.0
    legacy_count = 0
    fp32_digit_sum = 0.0
    fp32_digit_count = 0
    fp32_rest_sum = 0.0
    fp32_rest_count = 0
    fp32_total_sum = 0.0
    fp32_total_count = 0
    logits_dtype: str | None = None
    for start in range(0, len(records), batch_size):
        chunk = list(records[start : start + batch_size])
        pooled = torch.tensor(
            features[start : start + len(chunk)], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            soft = bridge(pooled)
        if not bool(torch.isfinite(soft).all()):
            raise ValueError("non-finite soft prefix")
        answers = [
            _answer_ids(tokenizer, record.target_json, eos_id) for record in chunk
        ]
        for answer, record in zip(answers, chunk, strict=True):
            expected = layout["answer_ids_by_class"][record.ec_class]
            if answer != expected:
                raise ValueError("answer tokenisation does not match the frozen digit layout")
            if len(answer) <= int(layout["prefix_len"]):
                raise ValueError("answer is shorter than the class-digit prefix")
        answer_ids, answer_mask = pad_rows(answers, pad_id)
        prompt_ids = torch.tensor([prompt_row] * len(chunk), dtype=torch.long, device=device)
        packed = inject_soft_prefix(
            soft=soft,
            prompt_ids=prompt_ids,
            answer_ids=answer_ids.to(device),
            embed=embed,
            pad_id=pad_id,
            answer_mask=answer_mask.to(device),
        )
        with torch.no_grad():
            logits = receiver(
                inputs_embeds=packed["inputs_embeds"],
                attention_mask=packed["attention_mask"],
                position_ids=packed["position_ids"],
            ).logits
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("non-finite teacher-forced logits")
        logits_dtype = str(logits.dtype)
        native_loss = shifted_cross_entropy(logits, packed["labels"])
        if not bool(torch.isfinite(native_loss)):
            raise ValueError("non-finite legacy token NLL")
        batch_count = int((packed["labels"][:, 1:] != IGNORE_INDEX).sum().item())
        legacy_sum += float(native_loss.detach().cpu()) * batch_count
        legacy_count += batch_count
        shift_logits = logits.float()[:, :-1].contiguous()
        shift_labels = packed["labels"][:, 1:].contiguous()
        token_ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view(shift_labels.shape)
        if not bool(torch.isfinite(token_ce).all()):
            raise ValueError("non-finite FP32 token CE")
        supervised = shift_labels != IGNORE_INDEX
        for i, record in enumerate(chunk):
            if int(packed["labels"][i, digit_t].item()) == IGNORE_INDEX:
                raise ValueError("class-digit label is not supervised")
            row_ce = token_ce[i]
            row_sup = supervised[i]
            digit_ce = float(row_ce[logit_index].item())
            rest_mask = row_sup.clone()
            rest_mask[logit_index] = False
            rest_sum = float(row_ce[rest_mask].double().sum().item())
            rest_count = int(rest_mask.sum().item())
            total_sum = float(row_ce[row_sup].double().sum().item())
            total_count = int(row_sup.sum().item())
            predicted_token = int(shift_logits[i, logit_index].argmax().item())
            target_token = int(layout["digit_token_by_class"][record.ec_class])
            predicted_class = _class_from_digit_token(layout, predicted_token)
            digit_correct = predicted_token == target_token
            fp32_digit_sum += digit_ce
            fp32_digit_count += 1
            fp32_rest_sum += rest_sum
            fp32_rest_count += rest_count
            fp32_total_sum += total_sum
            fp32_total_count += total_count
            per_record.append(
                {
                    "accession": record.accession,
                    "family_group": record.family_group,
                    "role": record.role,
                    "true_class": record.ec_class,
                    "digit_ce": digit_ce,
                    "digit_target_token": target_token,
                    "digit_predicted_token": predicted_token,
                    "digit_predicted_class": predicted_class,
                    "digit_correct": digit_correct,
                    "total_ce_sum": total_sum,
                    "total_token_count": total_count,
                    "remainder_ce_sum": rest_sum,
                    "remainder_token_count": rest_count,
                }
            )
    if legacy_count < 1 or fp32_total_count < 1 or fp32_digit_count != len(records):
        raise ValueError("teacher forcing produced no supervised answer tokens")
    if not all(math.isfinite(value) for value in (
        legacy_sum, fp32_digit_sum, fp32_rest_sum, fp32_total_sum
    )):
        raise ValueError("non-finite accumulated token CE")
    reconstructed = fp32_digit_sum + fp32_rest_sum
    residual = abs(reconstructed - fp32_total_sum)
    scale = max(abs(fp32_total_sum), 1.0)
    if residual > RECONSTRUCT_ATOL and residual / scale > RECONSTRUCT_RTOL:
        raise ValueError("digit+rest do not reconstruct the FP32 answer-token CE sum")
    return {
        "n": len(records),
        "role": records[0].role,
        "logits_dtype": logits_dtype,
        "loss_path_note": LOSS_PATH_NOTE,
        "digit_label_position": digit_t,
        "digit_logit_index": logit_index,
        "legacy_native": {
            "token_weighted_nll": legacy_sum / legacy_count,
            "ce_sum": legacy_sum,
            "token_count": legacy_count,
        },
        "fp32_unreduced": {
            "token_ce_dtype": "torch.float32",
            "sum_dtype": "float64/Python float",
            "digit_mean_ce": fp32_digit_sum / fp32_digit_count,
            "digit_accuracy": sum(row["digit_correct"] for row in per_record) / len(per_record),
            "digit_ce_sum": fp32_digit_sum,
            "digit_count": fp32_digit_count,
            "remainder_token_weighted_nll": (
                fp32_rest_sum / fp32_rest_count if fp32_rest_count else None
            ),
            "remainder_ce_sum": fp32_rest_sum,
            "remainder_token_count": fp32_rest_count,
            "total_token_weighted_nll": fp32_total_sum / fp32_total_count,
            "total_ce_sum": fp32_total_sum,
            "total_token_count": fp32_total_count,
            "digit_plus_rest_sum": reconstructed,
            "reconstruction_residual": residual,
        },
        "records": per_record,
    }


def free_greedy_predictions(
    *,
    bridge: nn.Module,
    receiver: Any,
    tokenizer: Any,
    features: np.ndarray,
    records: Sequence[PreparedRecord],
    prompt_mode: str,
    device: str,
    batch_size: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    validate_forward_inputs(features, records, batch_size)
    require_positive_limits(
        max_prompt_tokens=max_prompt_tokens, max_new_tokens=max_new_tokens
    )
    freeze_module(bridge.to(device))

    def finite_logits(_module: nn.Module, _args: Any, output: Any) -> None:
        if not bool(torch.isfinite(output.logits).all()):
            raise ValueError("non-finite greedy logits")

    hook = receiver.register_forward_hook(finite_logits)
    try:
        rows = generate_predictions(
            bridge=bridge,  # type: ignore[arg-type]
            receiver=receiver,
            tokenizer=tokenizer,
            features=features,
            records=records,
            prompt_mode=prompt_mode,
            user_mode="latent",
            device=device,
            max_new_tokens=max_new_tokens,
            max_prompt_tokens=max_prompt_tokens,
            batch_size=batch_size,
        )
    finally:
        hook.remove()
    return {
        "n": len(rows),
        "role": records[0].role if records else None,
        "metrics": prediction_metrics(rows),
        "rows": rows,
        "answer_ids_fed": False,
    }


def suffix_invariance_check(
    *,
    bridge: nn.Module,
    receiver: Any,
    tokenizer: Any,
    feature_row: np.ndarray,
    record: PreparedRecord,
    layout: Mapping[str, Any],
    prompt_mode: str,
    device: str,
    max_prompt_tokens: int,
    atol: float = CAUSAL_ATOL,
    rtol: float = CAUSAL_RTOL,
) -> dict[str, Any]:
    validate_forward_inputs(feature_row.reshape(1, -1), [record], 1)
    require_positive_limits(max_prompt_tokens=max_prompt_tokens)
    freeze_module(receiver)
    freeze_module(bridge.to(device))
    embed = receiver.get_input_embeddings()
    pad_id = int(tokenizer.pad_token_id)
    eos_id = tokenizer.eos_token_id
    prompt_row = shared_latent_prompt_ids(
        tokenizer, prompt_mode=prompt_mode, max_prompt_tokens=max_prompt_tokens
    )
    answer = list(layout["answer_ids_by_class"][record.ec_class])
    prefix_len = int(layout["prefix_len"])
    suffix_at = prefix_len + 1
    if suffix_at >= len(answer):
        raise ValueError("canonical answer has no same-length suffix token to edit")
    original = int(answer[suffix_at])
    vocab = int(embed.weight.shape[0])
    replacement = None
    for candidate in range(vocab):
        if candidate not in {original, pad_id} and (
            eos_id is None or candidate != int(eos_id)
        ):
            replacement = candidate
            break
    if replacement is None:
        raise ValueError("no legal same-length suffix replacement token")
    edited = list(answer)
    edited[suffix_at] = replacement
    if len(edited) != len(answer):
        raise RuntimeError("suffix edit changed answer length")
    pooled = torch.tensor(feature_row.reshape(1, -1), dtype=torch.float32, device=device)
    with torch.no_grad():
        soft = bridge(pooled)
    if not bool(torch.isfinite(soft).all()):
        raise ValueError("non-finite suffix-check soft prefix")
    prompt_ids = torch.tensor([prompt_row], dtype=torch.long, device=device)
    logits = []
    for tokens in (answer, edited):
        answer_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        packed = inject_soft_prefix(
            soft=soft,
            prompt_ids=prompt_ids,
            answer_ids=answer_ids,
            embed=embed,
            pad_id=pad_id,
            answer_mask=torch.ones(1, len(tokens), dtype=torch.long, device=device),
        )
        with torch.no_grad():
            logits.append(
                receiver(
                    inputs_embeds=packed["inputs_embeds"],
                    attention_mask=packed["attention_mask"],
                    position_ids=packed["position_ids"],
                ).logits
            )
    if not all(bool(torch.isfinite(value).all()) for value in logits):
        raise ValueError("non-finite suffix-check logits")
    digit_t = bridge_prefix_width(bridge) + len(prompt_row) + prefix_len
    index = class_logit_index(digit_t)
    before = logits[0][:, index]
    after = logits[1][:, index]
    if not torch.allclose(before, after, atol=atol, rtol=rtol):
        raise ValueError("pre-digit causal logits changed after a same-length suffix edit")
    return {
        "digit_label_position": digit_t,
        "digit_logit_index": index,
        "suffix_index_in_answer": suffix_at,
        "original_suffix_token": original,
        "replacement_suffix_token": replacement,
        "pre_digit_logits_close": True,
        "atol": atol,
        "rtol": rtol,
    }


def direct_classifier_metrics(
    *,
    head: nn.Module,
    features: np.ndarray,
    records: Sequence[PreparedRecord],
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    validate_forward_inputs(features, records, batch_size)
    freeze_module(head.to(device))
    predicted = predict_classes(
        head=head,  # type: ignore[arg-type]
        features=features,
        device=device,
        batch_size=batch_size,
    )
    rows = classifier_direct_rows(records, predicted)
    return {
        "n": len(rows),
        "role": records[0].role if records else None,
        "metrics": prediction_metrics(rows),
        "rows": rows,
        "source": "classifier_direct_template",
        "receiver_handoff_ran": False,
    }


def assert_parameters_unchanged(
    modules: Sequence[tuple[str, nn.Module, str]],
) -> dict[str, str]:
    fingerprints = {}
    for name, module, expected in modules:
        actual = parameter_fingerprint(module)
        if actual != expected:
            raise RuntimeError(f"{name} parameter fingerprint changed")
        if any(parameter.grad is not None for parameter in module.parameters()):
            raise RuntimeError(f"{name} acquired a gradient")
        fingerprints[name] = actual
    return fingerprints
