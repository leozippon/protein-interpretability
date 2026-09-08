"""Explicit classifier → JSON-label → frozen text-model handoff.

Direction-1 comparison against the frozen latent-prefix bridge: the same
pretrained donor cache feeds a small classification head, the predicted EC
class is written as canonical JSON in the prompt, and the same frozen
receiver must copy that prediction. This is not a mechanistic claim, not a
reopening of D3.g, and evaluation on the already-viewed prepared_v1 holdout
is exploratory.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD

from src.transfer.io import write_json
from src.transfer.latent_bridge import (
    BRIDGE_SCHEMA,
    CLASS_IDS,
    PreparedRecord,
    all_prepared_records,
    decode_ids,
    encode_prompt,
    freeze_module,
    greedy_continue,
    load_feature_cache,
    pad_rows,
    parse_generated_json,
    plain_instruction,
    portable_identity,
    prediction_metrics,
    prediction_row,
    prepared_records,
    seeded_initialization,
    target_json,
    timed_cuda,
    validate_cache_binding,
)
from src.transfer.latent_bridge import _sha256_json

PILOT_KIND = "m0_m1_ec_topclass_classifier_handoff"
PILOT_NOTE = (
    "Direction-1 capability comparison: frozen pretrained-donor features, a "
    "supervised classification head, explicit predicted-label text, and a "
    "frozen text model that must emit restricted EC-class JSON. Not free-form "
    "function explanation, not a latent-prefix substitute claim by itself, "
    "and not a stage 35/36 continuation."
)
CLASSIFIER_SCHEMA = "classifier_handoff_checkpoint_v1"
REPORT_SCHEMA = "classifier_handoff_report_v1"
CONDITION = "pretrained_donor"
ALLOWED_HEADS = ("linear", "mlp")
SUBSTAGES = ("interface", "train", "eval")
N_CLASSES = 7
DEFAULT_HEADS = ("linear", "mlp")
DEFAULT_SEEDS = (20260908, 20260909, 20260910)
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_BETAS = (0.9, 0.999)
DEFAULT_EPS = 1e-8
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_BUDGET = 3_000_000
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_MAX_PROMPT_TOKENS = 1024
DEFAULT_PROMPT_MODE = "chat"
DEFAULT_DTYPE = "bfloat16"
LOSS_UNIT = "mean cross entropy per record"
EVALUATION_CLAIM = (
    "exploratory reuse of the already-viewed prepared_v1 family-holdout split; "
    "not an independent confirmatory test"
)
HANDOFF_FOLLOWUP = (
    "The classifier predicted {prediction}. Return that prediction as the specified JSON."
)


def mlp_hidden_width(*, d_in: int, n_classes: int, budget: int) -> tuple[int, int]:
    """Largest MLP hidden width whose two Linear layers stay within ``budget``.

    LayerNorm is frozen and is not counted. ``h = floor((budget-n_classes)/(d_in+n_classes+1))``.
    """

    if min(d_in, n_classes, budget) < 1:
        raise ValueError("d_in, n_classes and budget must be positive")
    denom = d_in + n_classes + 1
    numerator = budget - n_classes
    if numerator < denom:
        raise ValueError(
            f"trainable budget {budget} is too small for mlp d_in={d_in}, n_classes={n_classes}"
        )
    hidden = int(numerator // denom)
    actual = d_in * hidden + hidden + hidden * n_classes + n_classes
    if actual > budget:
        raise RuntimeError("mlp width exceeded the trainable budget")
    return hidden, actual


def linear_parameter_count(*, d_in: int, n_classes: int) -> int:
    if min(d_in, n_classes) < 1:
        raise ValueError("d_in and n_classes must be positive")
    return d_in * n_classes + n_classes


def cpu_state_clone(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def train_config_record(
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    betas: Sequence[float],
    eps: float,
    grad_clip: float,
    seed: int,
) -> dict[str, Any]:
    if len(betas) != 2:
        raise ValueError("AdamW betas must be a pair")
    return {
        "optimizer": "AdamW",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "betas": [float(betas[0]), float(betas[1])],
        "eps": float(eps),
        "grad_clip": float(grad_clip),
        "seed": int(seed),
        "loss": LOSS_UNIT,
        "checkpoint_rule": "min_val_ce_strict_lt_keeps_first_tie",
        "head_dtype": "float32",
        "shuffle": "numpy_default_rng(seed + epoch)",
    }


class ClassifierHead(nn.Module):
    """Frozen input LayerNorm plus a trainable classification stack."""

    kind: str
    d_in: int
    hidden: int | None
    n_classes: int

    def __init__(self, *, kind: str, d_in: int, n_classes: int, hidden: int | None) -> None:
        super().__init__()
        if n_classes != len(CLASS_IDS):
            raise ValueError(f"classifier heads are 7-way, got n_classes={n_classes}")
        self.kind = kind
        self.d_in = int(d_in)
        self.hidden = None if hidden is None else int(hidden)
        self.n_classes = int(n_classes)
        self.input_norm = nn.LayerNorm(d_in)
        for parameter in self.input_norm.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> ClassifierHead:
        super().train(mode)
        self.input_norm.eval()
        return self

    def _check_input(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.ndim != 2 or pooled.shape[-1] != self.d_in:
            raise ValueError(f"expected (batch, {self.d_in}), got {tuple(pooled.shape)}")
        return pooled.float()

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def config(self) -> dict[str, Any]:
        return {
            "schema_version": CLASSIFIER_SCHEMA,
            "kind": self.kind,
            "d_in": self.d_in,
            "hidden": self.hidden,
            "n_classes": self.n_classes,
            "actual_parameters": int(self.trainable_parameter_count()),
            "input_norm": "LayerNorm_frozen",
            "last_layer_init": {"weight": "normal_std_0.02", "bias": "zeros"},
        }


class LinearClassifierHead(ClassifierHead):
    def __init__(self, *, d_in: int, n_classes: int = N_CLASSES) -> None:
        super().__init__(kind="linear", d_in=d_in, n_classes=n_classes, hidden=None)
        self.fc = nn.Linear(d_in, n_classes)
        nn.init.zeros_(self.fc.bias)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.02)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.fc(self.input_norm(self._check_input(pooled)))


class MLPClassifierHead(ClassifierHead):
    def __init__(self, *, d_in: int, hidden: int, n_classes: int = N_CLASSES) -> None:
        if hidden < 1:
            raise ValueError("mlp hidden width must be positive")
        super().__init__(kind="mlp", d_in=d_in, n_classes=n_classes, hidden=int(hidden))
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)
        nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(self.input_norm(self._check_input(pooled))))
        return self.fc2(hidden)


def build_classifier_head(
    *,
    kind: str,
    d_in: int,
    budget: int,
    seed: int,
    n_classes: int = N_CLASSES,
) -> ClassifierHead:
    if kind not in ALLOWED_HEADS:
        raise ValueError(f"head kind must be linear or mlp, got {kind!r}")
    if min(d_in, budget, n_classes) < 1:
        raise ValueError("d_in, budget and n_classes must be positive")
    with seeded_initialization(seed):
        if kind == "linear":
            actual = linear_parameter_count(d_in=d_in, n_classes=n_classes)
            if actual > budget:
                raise ValueError(f"linear head needs {actual} parameters > budget {budget}")
            head: ClassifierHead = LinearClassifierHead(d_in=d_in, n_classes=n_classes)
        else:
            hidden, actual = mlp_hidden_width(d_in=d_in, n_classes=n_classes, budget=budget)
            head = MLPClassifierHead(d_in=d_in, hidden=hidden, n_classes=n_classes)
    if head.trainable_parameter_count() != actual:
        raise RuntimeError(
            f"{kind} trainable count {head.trainable_parameter_count()} != expected {actual}"
        )
    return head


def require_class_id(value: Any) -> int:
    if type(value) is not int:
        raise ValueError(
            f"predicted class must be a true int 1-7, got {value!r} of type {type(value).__name__}"
        )
    if value not in CLASS_IDS:
        raise ValueError(f"predicted class must be a true int 1-7, got {value!r}")
    return value


def require_class_id_sequence(values: Any) -> list[int]:
    if isinstance(values, (str, bytes)):
        raise ValueError("predicted_classes must be a sequence of ints, not a string")
    try:
        items = list(values)
    except TypeError as error:
        raise ValueError("predicted_classes must be a nonempty sequence of ints") from error
    if not items:
        raise ValueError("predicted_classes must be nonempty")
    return [require_class_id(value) for value in items]


def handoff_prompt(predicted_class: int) -> str:
    predicted_class = require_class_id(predicted_class)
    followup = HANDOFF_FOLLOWUP.format(prediction=target_json(predicted_class))
    return "\n".join([plain_instruction(), followup])


def require_head_gradients(head: nn.Module) -> None:
    nonzero = False
    for name, parameter in head.named_parameters():
        if not parameter.requires_grad:
            continue
        grad = parameter.grad
        if grad is None or not bool(torch.isfinite(grad).all()):
            raise RuntimeError(f"classifier gradient missing or non-finite: {name}")
        nonzero |= bool((grad != 0).any())
    if not nonzero:
        raise RuntimeError("classifier head received no nonzero gradient")


def bound_pretrained_cache(
    cache_path: Path,
    prepared: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any], list[PreparedRecord], list[PreparedRecord], list[PreparedRecord]]:
    rows = all_prepared_records(prepared)
    features, manifest = load_feature_cache(cache_path, rows)
    validate_cache_binding(CONDITION, prepared["prepared_sha256"], manifest)
    if int(manifest["d"]) != int(features.shape[1]):
        raise ValueError("cache manifest d does not match feature width")
    train_rows = prepared_records(prepared, "train")
    val_rows = prepared_records(prepared, "val")
    test_rows = prepared_records(prepared, "test")
    expected_n = len(train_rows) + len(val_rows) + len(test_rows)
    if features.shape[0] != expected_n:
        raise ValueError("cache rows do not match train+val+test")
    return features, manifest, train_rows, val_rows, test_rows


def split_cache_features(
    features: np.ndarray,
    train_rows: Sequence[PreparedRecord],
    val_rows: Sequence[PreparedRecord],
    test_rows: Sequence[PreparedRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train, n_val, n_test = len(train_rows), len(val_rows), len(test_rows)
    if features.shape[0] != n_train + n_val + n_test:
        raise ValueError("feature rows do not match split lengths")
    train_feat = features[:n_train]
    val_feat = features[n_train : n_train + n_val]
    test_feat = features[n_train + n_val :]
    return train_feat, val_feat, test_feat


def _require_roles(records: Sequence[PreparedRecord], role: str, name: str) -> None:
    if not records:
        raise ValueError(f"{name} is empty")
    if any(record.role != role for record in records):
        raise ValueError(f"{name} contains a non-{role} record")


def _ce_targets(records: Sequence[PreparedRecord], device: str) -> torch.Tensor:
    labels = [record.ec_class - 1 for record in records]
    if any(label < 0 or label >= N_CLASSES for label in labels):
        raise ValueError("EC class is outside 1-7")
    return torch.tensor(labels, dtype=torch.long, device=device)


def train_classifier_head(
    *,
    head: ClassifierHead,
    train_features: np.ndarray,
    train_records: Sequence[PreparedRecord],
    val_features: np.ndarray,
    val_records: Sequence[PreparedRecord],
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    betas: Sequence[float],
    eps: float,
    grad_clip: float,
    seed: int,
) -> dict[str, Any]:
    """Fit one head on train/val only. There is no test argument."""

    _require_roles(train_records, "train", "train_records")
    _require_roles(val_records, "val", "val_records")
    if train_features.shape != (len(train_records), head.d_in):
        raise ValueError("train features do not match records/head width")
    if val_features.shape != (len(val_records), head.d_in):
        raise ValueError("val features do not match records/head width")
    if min(epochs, batch_size) < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not np.isfinite(train_features).all() or not np.isfinite(val_features).all():
        raise ValueError("features contain NaN or inf")
    config = train_config_record(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        grad_clip=grad_clip,
        seed=seed,
    )
    head.to(device)
    optimizer = AdamW(
        [parameter for parameter in head.parameters() if parameter.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
        betas=(float(betas[0]), float(betas[1])),
        eps=eps,
    )
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = 0
    saw_nonzero_head_grad = False

    def epoch_ce(feat: np.ndarray, rows: Sequence[PreparedRecord], *, train: bool, epoch: int) -> float:
        nonlocal saw_nonzero_head_grad
        if train:
            head.train()
            order = np.random.default_rng(int(seed) + epoch).permutation(len(rows))
        else:
            head.eval()
            order = np.arange(len(rows))
        loss_sum = 0.0
        n_seen = 0
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            batch_rows = [rows[int(i)] for i in index]
            pooled = torch.tensor(feat[index], dtype=torch.float32, device=device)
            targets = _ce_targets(batch_rows, device)
            if train:
                optimizer.zero_grad(set_to_none=True)
                logits = head(pooled)
            else:
                with torch.no_grad():
                    logits = head(pooled)
            loss = F.cross_entropy(logits, targets, reduction="mean")
            if not bool(torch.isfinite(loss)):
                raise ValueError("non-finite classifier CE")
            if train:
                loss.backward()
                require_head_gradients(head)
                saw_nonzero_head_grad = True
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        [parameter for parameter in head.parameters() if parameter.requires_grad],
                        grad_clip,
                        error_if_nonfinite=True,
                    )
                optimizer.step()
            count = len(batch_rows)
            loss_sum += float(loss.detach().cpu()) * count
            n_seen += count
        if n_seen != len(rows):
            raise RuntimeError("epoch did not consume every record")
        return loss_sum / n_seen

    for epoch in range(1, epochs + 1):
        train_ce = epoch_ce(train_features, train_records, train=True, epoch=epoch)
        with torch.no_grad():
            val_ce = epoch_ce(val_features, val_records, train=False, epoch=epoch)
        history.append({"epoch": epoch, "train_ce": train_ce, "val_ce": val_ce})
        if val_ce < best_val:
            best_val = val_ce
            best_epoch = epoch
            best_state = cpu_state_clone(head)
    if best_state is None or best_epoch < 1:
        raise RuntimeError("training produced no checkpoint")
    if not saw_nonzero_head_grad:
        raise RuntimeError("no non-zero classifier gradient was observed")
    head.load_state_dict(best_state)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_ce": best_val,
        "loss_unit": LOSS_UNIT,
        "actual_parameters": head.trainable_parameter_count(),
        "n_train": len(train_records),
        "n_val": len(val_records),
        "saw_nonzero_head_grad": True,
        "test_metrics_computed": False,
        "train_config": config,
        "head_config": head.config(),
        "seed": int(seed),
    }


def classifier_identity(
    *,
    head: ClassifierHead,
    budget: int,
    seed: int,
    prepared_hash: str,
    cache_manifest: Mapping[str, Any],
    train_config: Mapping[str, Any],
) -> dict[str, Any]:
    validate_cache_binding(CONDITION, prepared_hash, cache_manifest)
    if int(cache_manifest["d"]) != head.d_in:
        raise ValueError("head input width and cache feature width mismatch")
    required = {
        "optimizer",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "betas",
        "eps",
        "grad_clip",
        "seed",
        "loss",
        "checkpoint_rule",
        "head_dtype",
    }
    missing = required - set(train_config)
    if missing:
        raise ValueError(f"train_config missing {sorted(missing)}")
    if int(train_config["seed"]) != int(seed):
        raise ValueError("train_config seed does not match identity seed")
    canonical_config = json.loads(json.dumps(dict(train_config)))
    return {
        "schema_version": CLASSIFIER_SCHEMA,
        "condition": CONDITION,
        "prepared_sha256": prepared_hash,
        "cache_manifest_sha256": _sha256_json(portable_identity(cache_manifest)),
        "head_config": dict(head.config()),
        "trainable_budget": int(budget),
        "seed": int(seed),
        "train_config": canonical_config,
    }


def validate_classifier_identity(payload: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    actual = payload.get("extra", {}).get("identity")
    if not isinstance(actual, dict) or _sha256_json(actual) != _sha256_json(expected):
        raise ValueError(
            "classifier artifact identity mismatch or missing identity; "
            "retrain with the current format"
        )
    if payload.get("config") != expected["head_config"]:
        raise ValueError("classifier configuration does not match its bound identity")


def save_classifier(path: Path, head: ClassifierHead, extra: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CLASSIFIER_SCHEMA,
        "config": head.config(),
        "extra": dict(extra),
        "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
    }
    torch.save(payload, destination)


def load_classifier(path: Path) -> tuple[ClassifierHead, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("classifier checkpoint is not a mapping")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("classifier checkpoint lacks config")
    schema = payload.get("schema_version", config.get("schema_version"))
    if schema == BRIDGE_SCHEMA or {"k", "d_recv", "output_scale"} <= set(config):
        raise ValueError(
            f"refusing latent-bridge checkpoint; classifier_handoff requires {CLASSIFIER_SCHEMA}"
        )
    if schema != CLASSIFIER_SCHEMA:
        raise ValueError(f"incompatible classifier schema {schema!r}; expected {CLASSIFIER_SCHEMA}")
    kind = config.get("kind")
    with seeded_initialization(0):
        if kind == "linear":
            head: ClassifierHead = LinearClassifierHead(
                d_in=config["d_in"], n_classes=config["n_classes"]
            )
        elif kind == "mlp":
            head = MLPClassifierHead(
                d_in=config["d_in"], hidden=config["hidden"], n_classes=config["n_classes"]
            )
        else:
            raise ValueError(f"unknown head kind {kind!r}")
    head.load_state_dict(payload["state_dict"])
    for name, value in head.state_dict().items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"non-finite classifier state: {name}")
    if head.config() != config:
        raise ValueError("loaded classifier config does not match checkpoint config")
    return head, payload


def cell_relative_dir(kind: str, seed: int) -> str:
    if kind not in ALLOWED_HEADS:
        raise ValueError(f"head kind must be linear or mlp, got {kind!r}")
    return f"{kind}/{int(seed)}"


def interface_head_step(
    *,
    head: ClassifierHead,
    features: np.ndarray,
    record: PreparedRecord,
    device: str,
) -> dict[str, Any]:
    if features.ndim != 2 or features.shape[0] < 1:
        raise ValueError("interface step needs at least one feature row")
    if record.role != "train":
        raise ValueError("interface step is train-only")
    head.to(device)
    ln_before = cpu_state_clone(head.input_norm)
    trainable_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in head.named_parameters()
        if parameter.requires_grad
    }
    pooled = torch.tensor(features[:1], dtype=torch.float32, device=device)
    targets = _ce_targets([record], device)
    head.zero_grad(set_to_none=True)
    loss = F.cross_entropy(head(pooled), targets)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("interface CE is non-finite")
    loss.backward()
    require_head_gradients(head)
    SGD([parameter for parameter in head.parameters() if parameter.requires_grad], lr=1e-3).step()
    for key, tensor in ln_before.items():
        current = head.input_norm.state_dict()[key].detach().cpu()
        if not torch.equal(tensor, current):
            raise RuntimeError("frozen LayerNorm changed during the interface step")
    changed = False
    for name, parameter in head.named_parameters():
        if parameter.requires_grad and not torch.equal(parameter.detach().cpu(), trainable_before[name]):
            changed = True
    if not changed:
        raise RuntimeError("interface step did not change trainable head parameters")
    return {
        "head_gradients_finite_nonzero": True,
        "head_changed": True,
        "layer_norm_unchanged": True,
        "ce": float(loss.detach().cpu()),
        "kind": head.kind,
        "actual_parameters": head.trainable_parameter_count(),
        "optimizer": {"name": "SGD", "lr": 1e-3, "steps": 1, "records": 1},
    }


def predict_classes(
    *,
    head: ClassifierHead,
    features: np.ndarray,
    device: str,
    batch_size: int,
) -> list[int]:
    if batch_size < 1 or features.ndim != 2 or features.shape[0] < 1:
        raise ValueError("predict_classes needs features and a positive batch size")
    if features.shape[1] != head.d_in:
        raise ValueError("feature width does not match the classifier")
    head.eval()
    head.to(device)
    predicted: list[int] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            pooled = torch.tensor(features[start : start + batch_size], dtype=torch.float32, device=device)
            logits = head(pooled)
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("non-finite classifier logits; refusing to render a class")
            labels = logits.argmax(dim=-1).cpu().tolist()
            predicted.extend(require_class_id(int(label) + 1) for label in labels)
    if len(predicted) != features.shape[0]:
        raise RuntimeError("class prediction count does not match features")
    return predicted


def classifier_direct_rows(
    records: Sequence[PreparedRecord],
    predicted_classes: Sequence[int],
) -> list[dict[str, Any]]:
    classes = require_class_id_sequence(predicted_classes)
    if len(records) != len(classes):
        raise ValueError("classifier predictions and records have different lengths")
    rows: list[dict[str, Any]] = []
    for record, sent in zip(records, classes):
        row = prediction_row(record, target_json(sent))
        row["sent_class"] = sent
        row["source"] = "classifier_direct_template"
        rows.append(row)
    return rows


def generate_handoff(
    *,
    predicted_classes: Sequence[int],
    receiver: Any,
    tokenizer: Any,
    prompt_mode: str,
    device: str,
    max_new_tokens: int,
    max_prompt_tokens: int,
    batch_size: int,
) -> list[str]:
    """Decode from predicted classes only. No records, gold labels, or sequences."""

    classes = require_class_id_sequence(predicted_classes)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    freeze_module(receiver)
    embed = receiver.get_input_embeddings()
    pad_id = int(tokenizer.pad_token_id)
    eos_id = tokenizer.eos_token_id
    texts: list[str] = []
    for start in range(0, len(classes), batch_size):
        chunk = classes[start : start + batch_size]
        prompt_rows = [
            encode_prompt(
                tokenizer,
                handoff_prompt(class_id),
                prompt_mode=prompt_mode,
                max_tokens=max_prompt_tokens,
            )
            for class_id in chunk
        ]
        prompt_ids, prompt_mask = pad_rows(prompt_rows, pad_id)
        prompt_ids = prompt_ids.to(device)
        prompt_mask = prompt_mask.to(device)
        with torch.no_grad():
            token_matrix = greedy_continue(
                model=receiver,
                embed=embed,
                inputs_embeds=embed(prompt_ids),
                attention_mask=prompt_mask,
                position_ids=(prompt_mask.cumsum(dim=1) - 1).clamp(min=0),
                max_new_tokens=max_new_tokens,
                eos_id=None if eos_id is None else int(eos_id),
            )
        for row in token_matrix:
            texts.append(
                decode_ids(tokenizer, row.tolist(), None if eos_id is None else int(eos_id))
            )
    return texts


def receiver_handoff_rows(
    records: Sequence[PreparedRecord],
    texts: Sequence[str],
    sent_classes: Sequence[int],
) -> list[dict[str, Any]]:
    classes = require_class_id_sequence(sent_classes)
    if len(records) != len(texts) or len(texts) != len(classes):
        raise ValueError("handoff texts, sent classes, and records have different lengths")
    rows: list[dict[str, Any]] = []
    for record, text, sent in zip(records, texts, classes):
        row = prediction_row(record, text)
        row["sent_class"] = sent
        row["source"] = "receiver_handoff"
        rows.append(row)
    return rows


def handoff_error_decomposition(
    classifier_rows: Sequence[Mapping[str, Any]],
    receiver_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(classifier_rows) != len(receiver_rows):
        raise ValueError("classifier and receiver rows are misaligned")
    match = 0
    introduced = 0
    corrected = 0
    counts = {
        "classifier_correct_handoff_preserved": 0,
        "classifier_correct_handoff_broken": 0,
        "classifier_incorrect_handoff_preserved": 0,
        "classifier_incorrect_handoff_broken": 0,
    }
    for classifier_row, receiver_row in zip(classifier_rows, receiver_rows):
        if classifier_row["accession"] != receiver_row["accession"]:
            raise ValueError("classifier and receiver rows are not accession-aligned")
        sent = require_class_id(receiver_row.get("sent_class", classifier_row.get("sent_class")))
        preserved = bool(receiver_row["valid_json"] and receiver_row["predicted_class"] == sent)
        classifier_ok = bool(classifier_row["correct"])
        receiver_ok = bool(receiver_row["correct"])
        if preserved:
            match += 1
        if classifier_ok and not receiver_ok:
            introduced += 1
        if (not classifier_ok) and receiver_ok:
            corrected += 1
        key = (
            "classifier_correct_handoff_preserved"
            if classifier_ok and preserved
            else "classifier_correct_handoff_broken"
            if classifier_ok
            else "classifier_incorrect_handoff_preserved"
            if preserved
            else "classifier_incorrect_handoff_broken"
        )
        counts[key] += 1
    n = len(classifier_rows)
    return {
        "n_attempts": n,
        "receiver_matches_sent_class": match / n if n else 0.0,
        "n_receiver_matches_sent_class": match,
        "invalid_counts_as_mismatch": True,
        "introduced_errors": introduced,
        "corrected_errors": corrected,
        "counts_2x2": counts,
        "denominator": "all_attempts",
        "note": "transmission mismatch is counted even when the classifier was already wrong",
    }


def evaluate_classifier_cell(
    *,
    head: ClassifierHead,
    features: np.ndarray,
    records: Sequence[PreparedRecord],
    receiver: Any,
    tokenizer: Any,
    prompt_mode: str,
    device: str,
    max_new_tokens: int,
    max_prompt_tokens: int,
    batch_size: int,
) -> dict[str, Any]:
    if len(records) != features.shape[0]:
        raise ValueError("eval features and records are misaligned")

    def _infer() -> list[int]:
        return predict_classes(head=head, features=features, device=device, batch_size=batch_size)

    predicted, infer_s, infer_peak = timed_cuda(device, _infer)
    direct_rows = classifier_direct_rows(records, predicted)

    def _decode() -> list[str]:
        return generate_handoff(
            predicted_classes=predicted,
            receiver=receiver,
            tokenizer=tokenizer,
            prompt_mode=prompt_mode,
            device=device,
            max_new_tokens=max_new_tokens,
            max_prompt_tokens=max_prompt_tokens,
            batch_size=batch_size,
        )

    texts, decode_s, decode_peak = timed_cuda(device, _decode)
    handoff_rows = receiver_handoff_rows(records, texts, predicted)
    return {
        "classifier_direct_metrics": prediction_metrics(direct_rows),
        "receiver_handoff_metrics": prediction_metrics(handoff_rows),
        "error_decomposition": handoff_error_decomposition(direct_rows, handoff_rows),
        "n": len(records),
        "classifier_inference_walltime_s": infer_s,
        "classifier_inference_peak_memory_bytes": infer_peak,
        "receiver_decode_walltime_s": decode_s,
        "receiver_decode_peak_memory_bytes": decode_peak,
        "direct_rows": direct_rows,
        "handoff_rows": handoff_rows,
        "evaluation_status": "exploratory",
        "evaluation_claim": EVALUATION_CLAIM,
    }


def train_all_cells(
    *,
    train_features: np.ndarray,
    train_records: Sequence[PreparedRecord],
    val_features: np.ndarray,
    val_records: Sequence[PreparedRecord],
    d_in: int,
    budget: int,
    heads: Sequence[str],
    seeds: Sequence[int],
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    betas: Sequence[float],
    eps: float,
    grad_clip: float,
    prepared_hash: str,
    cache_manifest: Mapping[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    if not heads or not seeds:
        raise ValueError("train_all_cells needs at least one head and one seed")
    if len(set(heads)) != len(heads):
        raise ValueError("duplicate heads")
    if len(set(seeds)) != len(list(seeds)):
        raise ValueError("duplicate seeds")
    if any(kind not in ALLOWED_HEADS for kind in heads):
        raise ValueError(f"head kind must be linear or mlp, got {list(heads)!r}")
    cells: list[dict[str, Any]] = []
    for kind in heads:
        for seed in seeds:
            relative = cell_relative_dir(kind, seed)
            cell_dir = Path(out_dir) / relative
            cell_dir.mkdir(parents=True, exist_ok=True)
            head = build_classifier_head(kind=kind, d_in=d_in, budget=budget, seed=seed)

            def _fit(
                current_head: ClassifierHead = head,
                current_seed: int = int(seed),
            ) -> dict[str, Any]:
                return train_classifier_head(
                    head=current_head,
                    train_features=train_features,
                    train_records=train_records,
                    val_features=val_features,
                    val_records=val_records,
                    device=device,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=betas,
                    eps=eps,
                    grad_clip=grad_clip,
                    seed=current_seed,
                )

            result, elapsed, peak = timed_cuda(device, _fit)
            identity = classifier_identity(
                head=head,
                budget=budget,
                seed=seed,
                prepared_hash=prepared_hash,
                cache_manifest=cache_manifest,
                train_config=result["train_config"],
            )
            save_classifier(cell_dir / "classifier.pt", head, extra={"identity": identity})
            cell_report = {**result, "identity": identity, "test_metrics_computed": False}
            write_json(cell_dir / "train_report.json", cell_report)
            cells.append(
                {
                    "kind": kind,
                    "seed": int(seed),
                    "dir": relative,
                    "best_epoch": result["best_epoch"],
                    "best_val_ce": result["best_val_ce"],
                    "actual_parameters": result["actual_parameters"],
                    "saw_nonzero_head_grad": True,
                    "train_walltime_s": elapsed,
                    "train_peak_memory_bytes": peak,
                    "identity": identity,
                }
            )
    return {
        "substage": "train",
        "cells": cells,
        "n_cells": len(cells),
        "d_in": int(d_in),
        "trainable_budget": int(budget),
        "test_touched": False,
        "test_metrics_computed": False,
        "loss_unit": LOSS_UNIT,
    }


def sent_class_preservation(texts: Sequence[str], sent_classes: Sequence[int]) -> list[dict[str, Any]]:
    classes = require_class_id_sequence(sent_classes)
    if len(texts) != len(classes):
        raise ValueError("sent-class texts and labels have different lengths")
    rows: list[dict[str, Any]] = []
    for text, sent in zip(texts, classes):
        parsed = parse_generated_json(text)
        rows.append(
            {
                "sent_class": sent,
                "text": text,
                "valid_json": parsed["valid"],
                "predicted_class": parsed["class_id"],
                "preserved": bool(parsed["valid"] and parsed["class_id"] == sent),
                "parse_error": parsed["error"],
            }
        )
    return rows


def command_record(args: Any) -> dict[str, Any]:
    payload = vars(args).copy()
    payload["evaluation_claim"] = EVALUATION_CLAIM
    payload["loss_unit"] = LOSS_UNIT
    payload["pilot_note"] = PILOT_NOTE
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload
