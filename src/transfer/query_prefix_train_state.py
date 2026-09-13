"""Bounded QueryPrefix adapter update engine with sealed boundary checkpoints.

One small state object owns the adapter reference, its AdamW, committed
progress, a private sampler, and boundary/failed status. It is not a
Trainer, DataLoader, collator, scheduler, AMP/scaler, DDP wrapper, or
command-line runner. Callbacks are trusted caller-owned inputs; this
module does not parse QA, load data, or authenticate frozen endpoints.
Host MemAvailable/disk gates and source authentication stay with the
operator. Resume is same-device-scope only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast
import hashlib
import io
import json
import math
import os
import random
import stat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_ as _clip_grad_norm_
from torch.optim import AdamW

from src.transfer.latent_bridge import IGNORE_INDEX, LOSS_UNIT

SCHEMA = "query_prefix_train_state_v1"
SEAL_SCHEMA = "query_prefix_train_state_seal_v1"
PAYLOAD_FILENAME = "payload.pt"
SEAL_FILENAME = "seal.json"
MAX_PAYLOAD_BYTES = 128 * 1024 * 1024
MAX_SEAL_BYTES = 64 * 1024
TOKEN_NORMALIZATION = "predeclared_total_supervised_tokens"
HEX64 = 64
U32_MAX = 0xFFFFFFFF
MT_WORDS = 624
PYTHON_MT_LEN = 625
ADAMW_GROUP_KEYS = frozenset(
    {
        "amsgrad",
        "betas",
        "capturable",
        "decoupled_weight_decay",
        "differentiable",
        "eps",
        "foreach",
        "fused",
        "lr",
        "maximize",
        "params",
        "weight_decay",
    }
)
ADAMW_STATE_KEYS = frozenset({"step", "exp_avg", "exp_avg_sq"})
ADAMW_FIXED_FLAGS: dict[str, bool] = {
    "amsgrad": False,
    "capturable": False,
    "differentiable": False,
    "foreach": False,
    "fused": False,
    "maximize": False,
}


class QueryPrefixTrainStateError(ValueError):
    """Fail-closed training-state error with a stable code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise QueryPrefixTrainStateError(code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        _fail(code)


def _positive_int(value: object, code: str) -> int:
    _require(type(value) is int, code)
    number = cast(int, value)
    _require(number >= 1, code)
    return number


def _nonnegative_int(value: object, code: str) -> int:
    _require(type(value) is int, code)
    number = cast(int, value)
    _require(number >= 0, code)
    return number


def _finite_float(value: object, code: str) -> float:
    _require(type(value) is float or type(value) is int, code)
    _require(type(value) is not bool, code)
    number = float(cast(int | float, value))
    _require(math.isfinite(number), code)
    return number


def _hex64(value: object, code: str) -> str:
    _require(type(value) is str, code)
    text = cast(str, value)
    _require(len(text) == HEX64, code)
    _require(all(ch in "0123456789abcdef" for ch in text), code)
    return text


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise QueryPrefixTrainStateError("identity_not_json") from exc


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _json_object(value: object) -> dict[str, Any]:
    parsed = json.loads(_canonical_bytes(value).decode("ascii"))
    _require(type(parsed) is dict, "identity_not_object")
    return cast(dict[str, Any], parsed)


def _cpu_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu").contiguous().clone()


def _cpu_int64(values: Sequence[int]) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.int64, device="cpu")


def _cpu_randperm(size: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randperm(size, generator=generator, device="cpu")


def _is_dense_fp32(parameter: torch.Tensor) -> bool:
    return (
        parameter.dtype == torch.float32
        and parameter.layout == torch.strided
        and not parameter.is_sparse
    )


def _layout(adapter: nn.Module) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, parameter in adapter.named_parameters():
        rows.append(
            {
                "kind": "parameter",
                "name": name,
                "shape": [int(dim) for dim in parameter.shape],
                "dtype": str(parameter.dtype),
                "requires_grad": bool(parameter.requires_grad),
            }
        )
    for name, buffer in adapter.named_buffers():
        rows.append(
            {
                "kind": "buffer",
                "name": name,
                "shape": [int(dim) for dim in buffer.shape],
                "dtype": str(buffer.dtype),
                "requires_grad": False,
            }
        )
    return rows


def _device_scope(adapter: nn.Module) -> dict[str, object]:
    seen: set[tuple[str, int | None]] = set()
    tensors = list(adapter.parameters()) + list(adapter.buffers())
    _require(len(tensors) > 0, "adapter_empty")
    for tensor in tensors:
        device = tensor.device
        seen.add((device.type, device.index))
    _require(len(seen) == 1, "adapter_device")
    kind, index = next(iter(seen))
    if kind == "cpu":
        return {"type": "cpu", "index": None}
    if kind == "cuda":
        _require(type(index) is int, "cuda_device_unindexed")
        return {"type": "cuda", "index": int(cast(int, index))}
    _fail("unsupported_device")
    return {"type": "cpu", "index": None}


def _trainable(adapter: nn.Module) -> list[torch.Tensor]:
    parameters: list[torch.Tensor] = []
    for parameter in adapter.parameters():
        if not parameter.requires_grad:
            continue
        _require(_is_dense_fp32(parameter), "trainable_not_dense_fp32")
        parameters.append(parameter)
    _require(len(parameters) > 0, "no_trainable_parameters")
    return parameters


def _grads_are_none(parameters: Sequence[torch.Tensor]) -> bool:
    return all(parameter.grad is None for parameter in parameters)


@dataclass(frozen=True, eq=False)
class PreparedMicrobatch:
    labels: torch.Tensor
    forward: Callable[[], torch.Tensor]

    def __post_init__(self) -> None:
        if type(self.labels) is not torch.Tensor:
            _fail("prepared_labels")
        _require(callable(self.forward), "prepared_forward")


class _BoundedBuffer(io.BytesIO):
    def __init__(self, cap: int) -> None:
        super().__init__()
        self._cap = cap

    def write(self, b: Any) -> int:
        data = bytes(b)
        if self.tell() + len(data) > self._cap:
            _fail("payload_exceeds_cap")
        return super().write(data)


def _reject_traversal(path: os.PathLike[str] | str) -> None:
    parts = PathLike(path).parts
    _require(".." not in parts, "path_traversal")
    _require(PathLike(path).name not in ("", ".", "..", "bridge.pt"), "invalid_checkpoint_name")


class PathLike:
    """Lexical path view without resolving or following links."""

    def __init__(self, path: os.PathLike[str] | str) -> None:
        self.raw = os.fspath(path)
        self.parts = tuple(part for part in self.raw.replace("\\", "/").split("/") if part not in ("", "."))
        self.name = os.path.basename(self.raw.rstrip("/")) or os.path.basename(self.raw)

    def __str__(self) -> str:
        return self.raw


def _abspath_lexical(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.getcwd(), path)


def _ancestor_dirs(path: str) -> list[str]:
    current = os.path.dirname(_abspath_lexical(path.rstrip("/")))
    ancestors: list[str] = []
    while True:
        ancestors.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return ancestors


def _require_real_dir(path: str, *, code: str) -> None:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise QueryPrefixTrainStateError(code) from exc
    _require(not stat.S_ISLNK(info.st_mode), "symlink_ancestor")
    _require(stat.S_ISDIR(info.st_mode), "ancestor_not_directory")


def _validate_parent_tree(target: str) -> None:
    _reject_traversal(target)
    parent = os.path.dirname(_abspath_lexical(target.rstrip("/")))
    _require(parent != _abspath_lexical(target.rstrip("/")), "invalid_checkpoint_name")
    for ancestor in _ancestor_dirs(target):
        _require_real_dir(ancestor, code="missing_parent")


def _exclusive_mkdir(target: str) -> None:
    _validate_parent_tree(target)
    try:
        os.lstat(target)
    except FileNotFoundError:
        pass
    else:
        _fail("occupied_checkpoint")
    try:
        os.mkdir(target, 0o755)
    except FileExistsError as exc:
        raise QueryPrefixTrainStateError("occupied_checkpoint") from exc


def _exclusive_write(path: str, data: bytes, *, cap: int) -> None:
    _require(len(data) <= cap, "oversize_write")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    fd = os.open(path, flags, 0o644)
    try:
        view = memoryview(data)
        written = 0
        while written < len(data):
            n = os.write(fd, view[written:])
            _require(n > 0, "short_write")
            written += n
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_regular_nofollow(path: str, *, cap: int) -> bytes:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise QueryPrefixTrainStateError("missing_checkpoint_file") from exc
    _require(not stat.S_ISLNK(before.st_mode), "symlink_file")
    _require(stat.S_ISREG(before.st_mode), "nonregular_file")
    _require(before.st_size <= cap, "oversize_file")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            _require(total <= cap, "oversize_file")
            chunks.append(chunk)
        data = b"".join(chunks)
    finally:
        os.close(fd)
    after = os.lstat(path)
    _require(
        (before.st_ino, before.st_dev, before.st_size, before.st_mtime_ns)
        == (after.st_ino, after.st_dev, after.st_size, after.st_mtime_ns),
        "stat_changed",
    )
    _require(len(data) == before.st_size, "size_mismatch")
    return data


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _as_int(value: object, code: str) -> int:
    if type(value) is bool:
        _fail(code)
    if type(value) is int:
        return value
    if isinstance(value, np.integer):
        return int(value)  # pyright: ignore[reportUnknownArgumentType]
    _fail(code)
    return 0


def _require_cpu_tensor(
    value: object,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...] | None,
    code: str,
) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), code)
    tensor = cast(torch.Tensor, value)
    _require(tensor.device.type == "cpu", code)
    _require(tensor.dtype == dtype, code)
    if shape is not None:
        _require(tuple(tensor.shape) == shape, code)
    return tensor


def _tensor_ints(tensor: torch.Tensor) -> list[int]:
    cpu = tensor.detach().to(device="cpu").reshape(-1)
    values: list[int] = []
    for index in range(int(cpu.numel())):
        values.append(int(cpu[index].item()))
    return values


def _u32_range(value: int, code: str) -> int:
    _require(0 <= value <= U32_MAX, code)
    return value


def _mt_position(value: int, code: str) -> int:
    _require(0 <= value <= MT_WORDS, code)
    return value


def _encode_python_rng(state: object) -> dict[str, object]:
    _require(type(state) is tuple and len(cast(tuple[object, ...], state)) == 3, "python_rng")
    version, key, gauss = cast(tuple[object, object, object], state)
    _require(type(version) is int, "python_rng")
    _require(version == 3, "python_rng")
    _require(type(key) is tuple and len(cast(tuple[object, ...], key)) == PYTHON_MT_LEN, "python_rng")
    ints: list[int] = []
    for index, item in enumerate(cast(tuple[object, ...], key)):
        _require(type(item) is int, "python_rng")
        number = int(cast(int, item))
        if index < MT_WORDS:
            _u32_range(number, "python_rng")
        else:
            _mt_position(number, "python_rng")
        ints.append(number)
    cached: float | None
    if gauss is None:
        cached = None
    else:
        cached = _finite_float(gauss, "python_rng")
    return {
        "version": 3,
        "mt": _cpu_int64(ints),
        "cached_gaussian": cached,
    }


def _decode_python_rng(payload: object) -> tuple[int, tuple[int, ...], float | None]:
    _require(type(payload) is dict, "python_rng")
    body = cast(dict[str, object], payload)
    version = body.get("version")
    mt = body.get("mt")
    _require(type(version) is int, "python_rng")
    _require(version == 3, "python_rng")
    tensor = _require_cpu_tensor(mt, dtype=torch.int64, shape=(PYTHON_MT_LEN,), code="python_rng")
    ints = _tensor_ints(tensor)
    _require(len(ints) == PYTHON_MT_LEN, "python_rng")
    for number in ints[:MT_WORDS]:
        _u32_range(number, "python_rng")
    _mt_position(ints[MT_WORDS], "python_rng")
    gauss = body.get("cached_gaussian")
    cached: float | None
    if gauss is None:
        cached = None
    else:
        cached = _finite_float(gauss, "python_rng")
    decoded = (3, tuple(ints), cached)
    probe = random.Random()
    try:
        probe.setstate(decoded)
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise QueryPrefixTrainStateError("python_rng") from exc
    return decoded


def _encode_numpy_rng(state: object) -> dict[str, object]:
    _require(type(state) is tuple and len(cast(tuple[object, ...], state)) == 5, "numpy_rng")
    algo, keys, pos, has_gauss, cached = cast(tuple[object, object, object, object, object], state)
    _require(algo == "MT19937", "numpy_rng")
    array = np.asarray(keys)
    _require(array.dtype == np.uint32 and array.shape == (MT_WORDS,), "numpy_rng")
    words = [int(value) for value in array.tolist()]
    for word in words:
        _u32_range(word, "numpy_rng")
    position = _mt_position(_as_int(pos, "numpy_rng"), "numpy_rng")
    flag = _as_int(has_gauss, "numpy_rng")
    _require(flag in (0, 1), "numpy_rng")
    cached_float = _finite_float(cached, "numpy_rng")
    return {
        "algorithm": "MT19937",
        "keys": _cpu_int64(words),
        "pos": position,
        "has_gauss": flag,
        "cached_gaussian": cached_float,
    }


def _decode_numpy_rng(payload: object) -> tuple[str, Any, int, int, float]:
    _require(type(payload) is dict, "numpy_rng")
    body = cast(dict[str, object], payload)
    _require(body.get("algorithm") == "MT19937", "numpy_rng")
    tensor = _require_cpu_tensor(body.get("keys"), dtype=torch.int64, shape=(MT_WORDS,), code="numpy_rng")
    words = _tensor_ints(tensor)
    for word in words:
        _u32_range(word, "numpy_rng")
    pos = _mt_position(_as_int(body.get("pos"), "numpy_rng"), "numpy_rng")
    has_gauss = _as_int(body.get("has_gauss"), "numpy_rng")
    _require(has_gauss in (0, 1), "numpy_rng")
    cached_float = _finite_float(body.get("cached_gaussian"), "numpy_rng")
    array = np.array(words, dtype=np.uint32)
    decoded = ("MT19937", array, pos, has_gauss, cached_float)
    probe = np.random.RandomState()
    try:
        probe.set_state(decoded)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise QueryPrefixTrainStateError("numpy_rng") from exc
    return decoded


def _validate_torch_byte_state(value: object, *, code: str) -> torch.Tensor:
    tensor = _require_cpu_tensor(value, dtype=torch.uint8, shape=None, code=code)
    _require(tensor.ndim == 1, code)
    cloned = _cpu_clone(tensor)
    probe = torch.Generator(device="cpu")
    try:
        probe.set_state(cloned)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise QueryPrefixTrainStateError(code) from exc
    return cloned


def _capture_rng(*, cuda_index: int | None) -> dict[str, object]:
    payload: dict[str, object] = {
        "python": _encode_python_rng(random.getstate()),
        "numpy": _encode_numpy_rng(np.random.get_state()),
        "torch_cpu": _cpu_clone(torch.get_rng_state()),
    }
    if cuda_index is not None:
        device = torch.device("cuda", cuda_index)
        payload["torch_cuda"] = _cpu_clone(torch.cuda.get_rng_state(device))
        payload["cuda_index"] = cuda_index
    return payload


def _validated_rng(
    payload: object, *, cuda_index: int | None
) -> tuple[tuple[int, tuple[int, ...], float | None], tuple[str, Any, int, int, float], torch.Tensor, torch.Tensor | None]:
    _require(type(payload) is dict, "rng_payload")
    body = cast(dict[str, object], payload)
    python_state = _decode_python_rng(body.get("python"))
    numpy_state = _decode_numpy_rng(body.get("numpy"))
    cpu_tensor = _validate_torch_byte_state(body.get("torch_cpu"), code="torch_cpu_rng")
    if cuda_index is None:
        _require("torch_cuda" not in body, "unexpected_cuda_rng")
        return python_state, numpy_state, cpu_tensor, None
    recorded = body.get("cuda_index")
    _require(recorded == cuda_index, "cuda_rng_scope")
    cuda_tensor = _require_cpu_tensor(body.get("torch_cuda"), dtype=torch.uint8, shape=None, code="torch_cuda_rng")
    _require(cuda_tensor.ndim == 1 and int(cuda_tensor.numel()) >= 1, "torch_cuda_rng")
    return python_state, numpy_state, cpu_tensor, _cpu_clone(cuda_tensor)


def _restore_rng(payload: object, *, cuda_index: int | None) -> None:
    python_state, numpy_state, cpu_tensor, cuda_tensor = _validated_rng(payload, cuda_index=cuda_index)
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(cpu_tensor)
    if cuda_index is None:
        return
    _require(cuda_tensor is not None, "torch_cuda_rng")
    torch.cuda.set_rng_state(cast(torch.Tensor, cuda_tensor), torch.device("cuda", cuda_index))


def _recipe_dict(
    *,
    dataset_size: int,
    microbatch_size: int,
    accumulation_steps: int,
    sampler_seed: int,
    total_update_budget: int,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    grad_clip: float,
) -> dict[str, object]:
    return {
        "dataset_size": dataset_size,
        "microbatch_size": microbatch_size,
        "accumulation_steps": accumulation_steps,
        "sampler_seed": sampler_seed,
        "total_update_budget": total_update_budget,
        "lr": lr,
        "betas": [betas[0], betas[1]],
        "eps": eps,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "ignore_index": int(IGNORE_INDEX),
        "loss_unit": LOSS_UNIT,
        "normalization": TOKEN_NORMALIZATION,
        "optimizer": "adamw",
        "amsgrad": False,
        "foreach": False,
        "fused": False,
    }


def _clone_state_tree(value: object) -> object:
    if isinstance(value, torch.Tensor):
        cloned = _cpu_clone(value)
        if cloned.is_floating_point():
            _require(bool(torch.isfinite(cloned).all()), "nonfinite_state_tensor")
        return cloned
    if isinstance(value, dict):
        return {key: _clone_state_tree(item) for key, item in cast(dict[object, object], value).items()}
    if type(value) is list:
        return [_clone_state_tree(item) for item in cast(list[object], value)]
    if type(value) is tuple:
        return tuple(_clone_state_tree(item) for item in cast(tuple[object, ...], value))
    if type(value) in (int, float, bool, str) or value is None:
        if type(value) is float:
            _require(math.isfinite(value), "nonfinite_state_number")
        return value
    _fail("unserializable_state")
    return None


def _move_state_tree(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device).contiguous().clone()
    if isinstance(value, dict):
        return {key: _move_state_tree(item, device) for key, item in cast(dict[object, object], value).items()}
    if type(value) is list:
        return [_move_state_tree(item, device) for item in cast(list[object], value)]
    if type(value) is tuple:
        return tuple(_move_state_tree(item, device) for item in cast(tuple[object, ...], value))
    return value


def _shifted_nll_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    vocab = int(logits.shape[-1])
    return F.cross_entropy(
        logits[:, :-1].contiguous().reshape(-1, vocab),
        labels[:, 1:].contiguous().reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )


def _integral_step(value: object, expected: int) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value
        _require(tensor.device.type == "cpu", "optimizer_state")
        _require(int(tensor.ndim) == 0 and int(tensor.numel()) == 1, "optimizer_state")
        _require(tensor.dtype != torch.bool, "optimizer_state")
        _require(bool(tensor.dtype.is_floating_point) or tensor.dtype == torch.int64, "optimizer_state")
        number = float(tensor.detach().item())
        _require(math.isfinite(number), "optimizer_state")
        _require(number == float(expected) and number == float(int(number)), "optimizer_state")
        return
    _require(type(value) is int, "optimizer_state")
    _require(int(cast(int, value)) == expected, "optimizer_state")


class QueryPrefixTrainState:
    """Owns one adapter, one AdamW, sampler cursor, and sealed-boundary status."""

    def __init__(
        self,
        *,
        adapter: nn.Module,
        identity: Mapping[str, object],
        dataset_size: int,
        microbatch_size: int,
        accumulation_steps: int,
        sampler_seed: int,
        total_update_budget: int,
        lr: float,
        betas: Sequence[float],
        eps: float,
        weight_decay: float,
        grad_clip: float,
    ) -> None:
        if not isinstance(adapter, nn.Module):  # pyright: ignore[reportUnnecessaryIsInstance]
            _fail("adapter_type")
        self._adapter = adapter
        self._trainable = _trainable(adapter)
        self._device = self._trainable[0].device
        for parameter in self._trainable:
            _require(parameter.device == self._device, "adapter_device")
        for tensor in list(adapter.parameters()) + list(adapter.buffers()):
            _require(tensor.device == self._device, "adapter_device")
        _require(_grads_are_none(self._trainable), "initial_grads_not_none")
        self._identity = _json_object(dict(identity))
        self._dataset_size = _positive_int(dataset_size, "dataset_size")
        self._microbatch_size = _positive_int(microbatch_size, "microbatch_size")
        self._accumulation_steps = _positive_int(accumulation_steps, "accumulation_steps")
        self._sampler_seed = _nonnegative_int(sampler_seed, "sampler_seed")
        self._budget = _positive_int(total_update_budget, "total_update_budget")
        self._lr = _finite_float(lr, "lr")
        _require(self._lr > 0.0, "lr")
        _require(type(betas) is tuple or type(betas) is list, "betas")
        _require(len(betas) == 2, "betas")
        beta1 = _finite_float(betas[0], "betas")
        beta2 = _finite_float(betas[1], "betas")
        _require(0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0, "betas")
        self._betas = (beta1, beta2)
        self._eps = _finite_float(eps, "eps")
        _require(self._eps > 0.0, "eps")
        self._weight_decay = _finite_float(weight_decay, "weight_decay")
        _require(self._weight_decay >= 0.0, "weight_decay")
        self._grad_clip = _finite_float(grad_clip, "grad_clip")
        _require(self._grad_clip > 0.0, "grad_clip")
        self._recipe = _recipe_dict(
            dataset_size=self._dataset_size,
            microbatch_size=self._microbatch_size,
            accumulation_steps=self._accumulation_steps,
            sampler_seed=self._sampler_seed,
            total_update_budget=self._budget,
            lr=self._lr,
            betas=self._betas,
            eps=self._eps,
            weight_decay=self._weight_decay,
            grad_clip=self._grad_clip,
        )
        self._layout = _layout(adapter)
        self._device_scope = _device_scope(adapter)
        self._bound = {
            "schema": SCHEMA,
            "identity": self._identity,
            "recipe": self._recipe,
            "layout": self._layout,
            "device_scope": self._device_scope,
        }
        self._bound_sha256 = _canonical_sha256(self._bound)
        self._identity_sha256 = _canonical_sha256(self._identity)
        self._recipe_sha256 = _canonical_sha256(self._recipe)
        self._optimizer = AdamW(
            self._trainable,
            lr=self._lr,
            betas=self._betas,
            eps=self._eps,
            weight_decay=self._weight_decay,
            amsgrad=False,
            foreach=False,
            fused=False,
        )
        _require(len(self._optimizer.param_groups) == 1, "optimizer_state")
        constructed = self._optimizer.param_groups[0]
        _require(frozenset(constructed.keys()) == ADAMW_GROUP_KEYS, "optimizer_state")
        for flag, expected in ADAMW_FIXED_FLAGS.items():
            _require(constructed[flag] is expected, "optimizer_state")
        _require(type(constructed["decoupled_weight_decay"]) is bool, "optimizer_state")
        _require(float(constructed["lr"]) == self._lr, "optimizer_state")
        _require(float(constructed["eps"]) == self._eps, "optimizer_state")
        _require(float(constructed["weight_decay"]) == self._weight_decay, "optimizer_state")
        constructed_betas = constructed["betas"]
        _require(type(constructed_betas) is tuple and len(constructed_betas) == 2, "optimizer_state")
        beta_pair = cast(tuple[object, object], constructed_betas)
        _require(
            _finite_float(beta_pair[0], "optimizer_state") == self._betas[0]
            and _finite_float(beta_pair[1], "optimizer_state") == self._betas[1],
            "optimizer_state",
        )
        self._canonical_group: dict[str, object] = {
            "lr": self._lr,
            "betas": [self._betas[0], self._betas[1]],
            "eps": self._eps,
            "weight_decay": self._weight_decay,
            "amsgrad": False,
            "foreach": False,
            "fused": False,
            "capturable": False,
            "differentiable": False,
            "maximize": False,
            "decoupled_weight_decay": bool(constructed["decoupled_weight_decay"]),
        }
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self._sampler_seed)
        self._permutation = _cpu_randperm(self._dataset_size, self._generator)
        self._cursor = 0
        self._epoch = 0
        self._consumed_rows = 0
        self._updates = 0
        self._tokens = 0
        self._examples = 0
        self._at_boundary = True
        self._failed = False
        self._window_rows = self._microbatch_size * self._accumulation_steps

    @property
    def bound_sha256(self) -> str:
        return self._bound_sha256

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def at_boundary(self) -> bool:
        return self._at_boundary and not self._failed

    def _invalidate(self) -> None:
        self._failed = True
        self._at_boundary = False

    def _cuda_index(self) -> int | None:
        if self._device_scope["type"] != "cuda":
            return None
        index = self._device_scope["index"]
        _require(type(index) is int, "cuda_device_unindexed")
        return int(cast(int, index))

    def _cancelled(self, cancelled: Callable[[], bool]) -> None:
        if cancelled():
            _fail("cancelled")

    def _draw_indices(self) -> tuple[int, ...]:
        chosen: list[int] = []
        remaining = self._window_rows
        while remaining > 0:
            available = self._dataset_size - self._cursor
            take = min(available, remaining)
            sl = self._permutation[self._cursor : self._cursor + take]
            chosen.extend(_tensor_ints(sl))
            self._cursor += take
            remaining -= take
            if self._cursor == self._dataset_size:
                self._epoch += 1
                self._permutation = _cpu_randperm(self._dataset_size, self._generator)
                self._cursor = 0
        self._consumed_rows += self._window_rows
        _require(len(chosen) == self._window_rows, "sampler_draw")
        return tuple(chosen)

    def _check_labels(self, labels: torch.Tensor) -> int:
        _require(labels.dtype == torch.long, "bad_labels")
        _require(labels.ndim == 2, "bad_labels")
        _require(int(labels.shape[0]) == self._microbatch_size, "bad_labels")
        _require(int(labels.shape[1]) >= 2, "bad_labels")
        _require(not bool(labels.requires_grad), "bad_labels")
        _require(labels.device == self._device, "bad_labels")
        support = labels[:, 1:] != IGNORE_INDEX
        count = int(support.sum().item())
        _require(count > 0, "zero_tokens")
        return count

    def _check_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        _require(logits.ndim == 3, "bad_logits")
        _require(logits.device == self._device, "bad_logits")
        _require(logits.is_floating_point(), "bad_logits")
        _require(tuple(logits.shape[:2]) == tuple(labels.shape), "bad_logits")
        vocab = int(logits.shape[-1])
        _require(vocab >= 1, "bad_logits")
        supervised = labels[:, 1:]
        valid = (supervised == IGNORE_INDEX) | ((supervised >= 0) & (supervised < vocab))
        _require(bool(valid.all()), "label_range")

    def _verify_grads(self) -> None:
        nonzero = False
        for parameter in self._trainable:
            grad = parameter.grad
            _require(grad is not None, "missing_grad")
            tensor = cast(torch.Tensor, grad)
            _require(bool(torch.isfinite(tensor).all()), "nonfinite_grad")
            nonzero = nonzero or bool((tensor != 0).any())
        _require(nonzero, "zero_grad")

    def update(
        self,
        prepare: Callable[[tuple[int, ...]], Sequence[PreparedMicrobatch]],
        *,
        cancelled: Callable[[], bool],
    ) -> dict[str, object]:
        _require(not self._failed, "failed_state")
        _require(self._at_boundary, "not_at_boundary")
        _require(self._updates < self._budget, "budget_consumed")
        _require(_grads_are_none(self._trainable), "boundary_grads")
        self._at_boundary = False
        try:
            self._cancelled(cancelled)
            indices = self._draw_indices()
            self._cancelled(cancelled)
            prepared = prepare(indices)
            _require(type(prepared) is list or type(prepared) is tuple, "prepare_type")
            batches: list[PreparedMicrobatch] = []
            for item in prepared:
                if not isinstance(item, PreparedMicrobatch):
                    _fail("prepared_type")
                batches.append(cast(PreparedMicrobatch, item))
            _require(len(batches) == self._accumulation_steps, "prepare_count")
            counts: list[int] = []
            for batch in batches:
                counts.append(self._check_labels(batch.labels))
            window_tokens = int(sum(counts))
            _require(window_tokens > 0, "zero_window_tokens")
            denominator = float(window_tokens)
            self._cancelled(cancelled)
            nll_total = torch.zeros((), dtype=torch.float32, device=self._device)
            for batch, declared in zip(batches, counts, strict=True):
                self._cancelled(cancelled)
                logits = batch.forward()
                self._check_logits(logits, batch.labels)
                recount = int((batch.labels[:, 1:] != IGNORE_INDEX).sum().item())
                _require(recount == declared, "token_count_changed")
                logits_fp32 = logits.float()
                nll = _shifted_nll_sum(logits_fp32, batch.labels)
                _require(nll.dtype == torch.float32, "nll_dtype")
                _require(bool(torch.isfinite(nll)), "nonfinite_loss")
                (nll / denominator).backward()  # pyright: ignore[reportUnknownMemberType]
                nll_total = nll_total + nll.detach()
            self._cancelled(cancelled)
            self._verify_grads()
            _clip_grad_norm_(
                self._trainable,
                max_norm=self._grad_clip,
                error_if_nonfinite=True,
            )
            self._optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            self._optimizer.zero_grad(set_to_none=True)
            self._updates += 1
            self._tokens += window_tokens
            self._examples += self._window_rows
            mean_nll = float((nll_total / denominator).item())
            _require(math.isfinite(mean_nll), "nonfinite_metrics")
            self._at_boundary = True
            return {
                "updates": self._updates,
                "window_tokens": window_tokens,
                "mean_answer_nll": mean_nll,
                "loss_unit": LOSS_UNIT,
                "window_examples": self._window_rows,
                "completed_tokens": self._tokens,
                "completed_examples": self._examples,
                "epoch": self._epoch,
                "cursor": self._cursor,
                "consumed_rows": self._consumed_rows,
            }
        except BaseException:
            self._invalidate()
            raise

    def _current_binding(self) -> None:
        _require(_layout(self._adapter) == self._layout, "layout_mismatch")
        _require(_device_scope(self._adapter) == self._device_scope, "device_scope_mismatch")
        live = _trainable(self._adapter)
        _require(len(live) == len(self._trainable), "layout_mismatch")
        for left, right in zip(live, self._trainable, strict=True):
            _require(left is right, "layout_mismatch")
        groups = self._optimizer.param_groups
        _require(len(groups) == 1, "optimizer_state")
        ordered = groups[0].get("params")
        _require(type(ordered) is list, "optimizer_state")
        _require(len(cast(list[object], ordered)) == len(self._trainable), "optimizer_state")
        for left, right in zip(cast(list[object], ordered), self._trainable, strict=True):
            _require(left is right, "optimizer_state")

    def _validate_adapter_snapshot(self, snapshot: object) -> dict[str, torch.Tensor]:
        self._current_binding()
        _require(isinstance(snapshot, dict), "adapter_state")
        saved = cast(dict[str, object], snapshot)
        live = self._adapter.state_dict()
        _require(set(saved.keys()) == set(live.keys()), "layout_mismatch")
        out: dict[str, torch.Tensor] = {}
        for name, live_tensor in live.items():
            source_obj = saved[name]
            _require(isinstance(source_obj, torch.Tensor), "adapter_state")
            source = cast(torch.Tensor, source_obj)
            _require(source.device.type == "cpu", "adapter_state")
            _require(tuple(source.shape) == tuple(live_tensor.shape), "layout_mismatch")
            _require(source.dtype == live_tensor.dtype, "layout_mismatch")
            if source.is_floating_point():
                _require(bool(torch.isfinite(source).all()), "nonfinite_state_tensor")
            out[name] = source
        return out

    def _validate_progress(self, progress: object) -> tuple[int, int, int]:
        _require(isinstance(progress, dict), "progress")
        body = cast(dict[str, object], progress)
        updates = _nonnegative_int(body.get("updates"), "progress")
        tokens = _nonnegative_int(body.get("tokens"), "progress")
        examples = _nonnegative_int(body.get("examples"), "progress")
        _require(updates <= self._budget, "progress")
        _require(examples == updates * self._window_rows, "progress")
        if updates == 0:
            _require(tokens == 0, "progress")
        else:
            _require(tokens >= updates * self._accumulation_steps, "progress")
        return updates, tokens, examples

    def _validate_group_settings(self, group: Mapping[str, object]) -> None:
        _require(frozenset(group.keys()) == ADAMW_GROUP_KEYS, "optimizer_state")
        _require(type(group["lr"]) is float and float(group["lr"]) == self._lr, "optimizer_state")
        betas = group["betas"]
        _require(type(betas) is tuple or type(betas) is list, "optimizer_state")
        pair = cast(Sequence[object], betas)
        _require(len(pair) == 2, "optimizer_state")
        _require(
            _finite_float(pair[0], "optimizer_state") == self._betas[0]
            and _finite_float(pair[1], "optimizer_state") == self._betas[1],
            "optimizer_state",
        )
        _require(type(group["eps"]) is float and float(group["eps"]) == self._eps, "optimizer_state")
        _require(
            type(group["weight_decay"]) is float and float(group["weight_decay"]) == self._weight_decay,
            "optimizer_state",
        )
        for flag, expected in ADAMW_FIXED_FLAGS.items():
            _require(group[flag] is expected, "optimizer_state")
        _require(
            type(group["decoupled_weight_decay"]) is bool
            and bool(group["decoupled_weight_decay"]) is bool(self._canonical_group["decoupled_weight_decay"]),
            "optimizer_state",
        )

    def _validate_optimizer_tree(self, optimizer: object, *, updates: int) -> dict[str, object]:
        _require(isinstance(optimizer, dict), "optimizer_state")
        body = cast(dict[str, object], optimizer)
        groups = body.get("param_groups")
        state_map = body.get("state")
        _require(type(groups) is list, "optimizer_state")
        group_list = cast(list[object], groups)
        _require(len(group_list) == 1, "optimizer_state")
        _require(isinstance(group_list[0], dict), "optimizer_state")
        group = cast(dict[str, object], group_list[0])
        self._validate_group_settings(group)
        params = group.get("params")
        expected_ids = list(range(len(self._trainable)))
        _require(type(params) is list and cast(list[object], params) == expected_ids, "optimizer_state")
        _require(isinstance(state_map, dict), "optimizer_state")
        state_body = cast(dict[object, object], state_map)
        if updates == 0:
            _require(len(state_body) == 0, "optimizer_state")
            return body
        keys: list[int] = []
        for key in state_body.keys():
            _require(type(key) is int, "optimizer_state")
            keys.append(int(cast(int, key)))
        _require(sorted(keys) == expected_ids, "optimizer_state")
        for index, parameter in enumerate(self._trainable):
            entry_obj = state_body[index]
            _require(isinstance(entry_obj, dict), "optimizer_state")
            entry = cast(dict[str, object], entry_obj)
            _require(frozenset(entry.keys()) == ADAMW_STATE_KEYS, "optimizer_state")
            _integral_step(entry.get("step"), updates)
            exp_avg = entry.get("exp_avg")
            exp_avg_sq = entry.get("exp_avg_sq")
            _require(isinstance(exp_avg, torch.Tensor), "optimizer_state")
            _require(isinstance(exp_avg_sq, torch.Tensor), "optimizer_state")
            first = cast(torch.Tensor, exp_avg)
            second = cast(torch.Tensor, exp_avg_sq)
            _require(first.device.type == "cpu" and second.device.type == "cpu", "optimizer_state")
            _require(_is_dense_fp32(first) and _is_dense_fp32(second), "optimizer_state")
            expected_shape = tuple(parameter.shape)
            _require(tuple(first.shape) == expected_shape and tuple(second.shape) == expected_shape, "optimizer_state")
            _require(bool(torch.isfinite(first).all()) and bool(torch.isfinite(second).all()), "optimizer_state")
            _require(bool((second >= 0).all()), "optimizer_state")
        return body

    def _validate_sampler_payload(self, sampler: object, *, examples: int) -> dict[str, object]:
        _require(isinstance(sampler, dict), "bad_sampler")
        body = cast(dict[str, object], sampler)
        epoch = _nonnegative_int(body.get("epoch"), "bad_sampler")
        cursor = _nonnegative_int(body.get("cursor"), "bad_sampler")
        consumed = _nonnegative_int(body.get("consumed_rows"), "bad_sampler")
        permutation = body.get("permutation")
        generator_state = body.get("generator_state")
        perm = _require_cpu_tensor(
            permutation,
            dtype=torch.int64,
            shape=(self._dataset_size,),
            code="bad_sampler",
        )
        _require(
            tuple(_tensor_ints(torch.sort(perm).values)) == tuple(range(self._dataset_size)),
            "bad_sampler",
        )
        _require(cursor < self._dataset_size, "bad_sampler")
        _require(consumed == epoch * self._dataset_size + cursor, "bad_sampler")
        _require(consumed == examples, "bad_sampler")
        gen = _validate_torch_byte_state(generator_state, code="bad_sampler")
        return {
            "epoch": epoch,
            "cursor": cursor,
            "consumed_rows": consumed,
            "permutation": _cpu_clone(perm),
            "generator_state": gen,
        }

    def _validate_boundary_payload(self, payload: Mapping[str, object]) -> dict[str, object]:
        _require(payload.get("schema") == SCHEMA, "schema_mismatch")
        _require(payload.get("identity_sha256") == self._identity_sha256, "identity_mismatch")
        _require(payload.get("recipe_sha256") == self._recipe_sha256, "config_mismatch")
        _require(payload.get("identity") == self._identity, "identity_mismatch")
        _require(payload.get("recipe") == self._recipe, "config_mismatch")
        _require(payload.get("layout") == self._layout, "layout_mismatch")
        _require(payload.get("device_scope") == self._device_scope, "device_scope_mismatch")
        _require(payload.get("bound_sha256") == self._bound_sha256, "identity_mismatch")
        adapter = self._validate_adapter_snapshot(payload.get("adapter"))
        updates, tokens, examples = self._validate_progress(payload.get("progress"))
        optimizer = self._validate_optimizer_tree(payload.get("optimizer"), updates=updates)
        sampler = self._validate_sampler_payload(payload.get("sampler"), examples=examples)
        _validated_rng(payload.get("rng"), cuda_index=self._cuda_index())
        return {
            "adapter": adapter,
            "optimizer": optimizer,
            "updates": updates,
            "tokens": tokens,
            "examples": examples,
            "sampler": sampler,
        }

    def _snapshot_adapter(self) -> dict[str, torch.Tensor]:
        payload: dict[str, torch.Tensor] = {}
        for name, tensor in self._adapter.state_dict().items():
            payload[name] = _cpu_clone(tensor)
        return payload

    def _sampler_payload(self) -> dict[str, object]:
        _require(self._permutation.device.type == "cpu", "bad_sampler")
        permutation = _cpu_clone(self._permutation)
        _require(permutation.dtype == torch.int64, "bad_sampler")
        _require(
            tuple(_tensor_ints(torch.sort(permutation).values)) == tuple(range(self._dataset_size)),
            "bad_sampler",
        )
        _require(0 <= self._cursor < self._dataset_size, "bad_sampler")
        _require(self._consumed_rows == self._epoch * self._dataset_size + self._cursor, "bad_sampler")
        return {
            "epoch": self._epoch,
            "cursor": self._cursor,
            "consumed_rows": self._consumed_rows,
            "permutation": permutation,
            "generator_state": _cpu_clone(self._generator.get_state()),
        }

    def _payload(self) -> dict[str, object]:
        self._current_binding()
        payload: dict[str, object] = {
            "schema": SCHEMA,
            "identity": self._identity,
            "identity_sha256": self._identity_sha256,
            "recipe": self._recipe,
            "recipe_sha256": self._recipe_sha256,
            "layout": self._layout,
            "device_scope": self._device_scope,
            "bound_sha256": self._bound_sha256,
            "adapter": self._snapshot_adapter(),
            "optimizer": _clone_state_tree(self._optimizer.state_dict()),
            "progress": {
                "updates": self._updates,
                "tokens": self._tokens,
                "examples": self._examples,
            },
            "sampler": self._sampler_payload(),
            "rng": _capture_rng(cuda_index=self._cuda_index()),
        }
        self._validate_boundary_payload(payload)
        return payload

    def publish(
        self,
        directory: os.PathLike[str] | str,
        *,
        max_payload_bytes: int = MAX_PAYLOAD_BYTES,
        max_seal_bytes: int = MAX_SEAL_BYTES,
    ) -> str:
        _require(not self._failed, "failed_state")
        _require(self._at_boundary, "not_at_boundary")
        _require(_grads_are_none(self._trainable), "boundary_grads")
        payload_cap = _positive_int(max_payload_bytes, "payload_cap")
        seal_cap = _positive_int(max_seal_bytes, "seal_cap")
        _require(payload_cap <= MAX_PAYLOAD_BYTES, "payload_cap")
        _require(seal_cap <= MAX_SEAL_BYTES, "seal_cap")
        buffer = _BoundedBuffer(payload_cap)
        torch.save(self._payload(), buffer)
        payload_bytes = buffer.getvalue()
        _require(len(payload_bytes) <= payload_cap, "payload_exceeds_cap")
        payload_sha256 = _sha256_bytes(payload_bytes)
        seal = {
            "schema": SEAL_SCHEMA,
            "payload_filename": PAYLOAD_FILENAME,
            "payload_sha256": payload_sha256,
            "payload_bytes": len(payload_bytes),
            "identity_sha256": self._identity_sha256,
            "recipe_sha256": self._recipe_sha256,
            "bound_sha256": self._bound_sha256,
        }
        seal_bytes = _canonical_bytes(seal) + b"\n"
        _require(len(seal_bytes) <= seal_cap, "seal_exceeds_cap")
        target = os.fspath(directory)
        _exclusive_mkdir(target)
        _exclusive_write(
            os.path.join(target, PAYLOAD_FILENAME),
            payload_bytes,
            cap=payload_cap,
        )
        observed = _read_regular_nofollow(
            os.path.join(target, PAYLOAD_FILENAME),
            cap=payload_cap,
        )
        _require(observed == payload_bytes, "payload_rewrite")
        _require(_sha256_bytes(observed) == payload_sha256, "payload_hash_mismatch")
        _exclusive_write(os.path.join(target, SEAL_FILENAME), seal_bytes, cap=seal_cap)
        closed_seal = _read_regular_nofollow(os.path.join(target, SEAL_FILENAME), cap=seal_cap)
        _require(closed_seal == seal_bytes, "seal_rewrite")
        _fsync_dir(target)
        return _sha256_bytes(seal_bytes)

    def _install(self, payload: Mapping[str, object]) -> None:
        validated = self._validate_boundary_payload(payload)
        adapter_state = cast(dict[str, torch.Tensor], validated["adapter"])
        current = self._adapter.state_dict()
        with torch.no_grad():
            for name, tensor in current.items():
                tensor.copy_(adapter_state[name].to(device=tensor.device))
        moved = _move_state_tree(validated["optimizer"], self._device)
        _require(isinstance(moved, dict), "optimizer_state")
        self._optimizer.load_state_dict(cast(dict[str, Any], moved))
        sampler = cast(dict[str, object], validated["sampler"])
        self._permutation = cast(torch.Tensor, sampler["permutation"])
        self._cursor = int(cast(int, sampler["cursor"]))
        self._epoch = int(cast(int, sampler["epoch"]))
        self._consumed_rows = int(cast(int, sampler["consumed_rows"]))
        self._generator.set_state(cast(torch.Tensor, sampler["generator_state"]))
        self._updates = int(cast(int, validated["updates"]))
        self._tokens = int(cast(int, validated["tokens"]))
        self._examples = int(cast(int, validated["examples"]))
        self._optimizer.zero_grad(set_to_none=True)
        _restore_rng(payload.get("rng"), cuda_index=self._cuda_index())
        self._at_boundary = True
        self._failed = False

    @classmethod
    def restore(
        cls,
        *,
        adapter: nn.Module,
        identity: Mapping[str, object],
        dataset_size: int,
        microbatch_size: int,
        accumulation_steps: int,
        sampler_seed: int,
        total_update_budget: int,
        lr: float,
        betas: Sequence[float],
        eps: float,
        weight_decay: float,
        grad_clip: float,
        checkpoint_dir: os.PathLike[str] | str,
        expected_seal_sha256: str,
        max_payload_bytes: int = MAX_PAYLOAD_BYTES,
        max_seal_bytes: int = MAX_SEAL_BYTES,
    ) -> QueryPrefixTrainState:
        payload_cap = _positive_int(max_payload_bytes, "payload_cap")
        seal_cap = _positive_int(max_seal_bytes, "seal_cap")
        _require(payload_cap <= MAX_PAYLOAD_BYTES, "payload_cap")
        _require(seal_cap <= MAX_SEAL_BYTES, "seal_cap")
        expected = _hex64(expected_seal_sha256, "seal_hash")
        target = os.fspath(checkpoint_dir)
        _validate_parent_tree(target)
        try:
            info = os.lstat(target)
        except OSError as exc:
            raise QueryPrefixTrainStateError("missing_checkpoint") from exc
        _require(not stat.S_ISLNK(info.st_mode), "symlink_checkpoint")
        _require(stat.S_ISDIR(info.st_mode), "checkpoint_not_directory")
        seal_bytes = _read_regular_nofollow(os.path.join(target, SEAL_FILENAME), cap=seal_cap)
        _require(_sha256_bytes(seal_bytes) == expected, "seal_hash_mismatch")
        try:
            seal_obj = json.loads(seal_bytes.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QueryPrefixTrainStateError("malformed_seal") from exc
        _require(type(seal_obj) is dict, "malformed_seal")
        seal = cast(dict[str, object], seal_obj)
        _require(seal.get("schema") == SEAL_SCHEMA, "seal_schema")
        _require(seal.get("payload_filename") == PAYLOAD_FILENAME, "seal_payload_name")
        payload_bytes_decl = seal.get("payload_bytes")
        payload_hash = seal.get("payload_sha256")
        _require(type(payload_bytes_decl) is int, "seal_payload_size")
        declared = int(cast(int, payload_bytes_decl))
        _require(1 <= declared <= payload_cap, "oversize_file")
        _hex64(payload_hash, "seal_payload_hash")
        state: QueryPrefixTrainState | None = None
        try:
            state = cls(
                adapter=adapter,
                identity=identity,
                dataset_size=dataset_size,
                microbatch_size=microbatch_size,
                accumulation_steps=accumulation_steps,
                sampler_seed=sampler_seed,
                total_update_budget=total_update_budget,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                grad_clip=grad_clip,
            )
            _require(seal.get("identity_sha256") == state._identity_sha256, "identity_mismatch")
            _require(seal.get("recipe_sha256") == state._recipe_sha256, "config_mismatch")
            _require(seal.get("bound_sha256") == state._bound_sha256, "layout_mismatch")
            payload_bytes = _read_regular_nofollow(
                os.path.join(target, PAYLOAD_FILENAME),
                cap=payload_cap,
            )
            _require(len(payload_bytes) == declared, "payload_size_mismatch")
            _require(_sha256_bytes(payload_bytes) == payload_hash, "payload_hash_mismatch")
            loaded = torch.load(io.BytesIO(payload_bytes), map_location="cpu", weights_only=True)
            _require(type(loaded) is dict, "payload_type")
            state._install(cast(dict[str, object], loaded))
            return state
        except BaseException:
            if state is not None:
                state._invalidate()
            raise
