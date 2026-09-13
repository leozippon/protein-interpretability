"""Synthetic CPU checks for QueryPrefix update/state and sealed resume."""

# pyright: reportUnknownArgumentType=false

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, cast
from unittest import mock
import hashlib
import io
import json
import os
import random
import resource
import stat
import subprocess
import sys
import tempfile
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.transfer.latent_bridge import IGNORE_INDEX, LOSS_UNIT
from src.transfer.query_prefix_train_state import (
    MAX_PAYLOAD_BYTES,
    MAX_SEAL_BYTES,
    PAYLOAD_FILENAME,
    SEAL_FILENAME,
    PreparedMicrobatch,
    QueryPrefixTrainState,
    QueryPrefixTrainStateError,
)

INTERRUPT_CODE = 17
IDENTITY: dict[str, object] = {
    "data_sha256": "ab" * 32,
    "feature_sha256": "cd" * 32,
    "provenance_sha256": "ef" * 32,
    "code_commit": "c" * 40,
    "architecture": "tiny_linear",
}


def _limit_as() -> None:
    cap = 16 * 1024 ** 3
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_hard = cap if hard == resource.RLIM_INFINITY else min(cap, hard)
    new_soft = cap if soft == resource.RLIM_INFINITY else min(cap, soft)
    resource.setrlimit(resource.RLIMIT_AS, (min(new_soft, new_hard), new_hard))


def _alive(_: object = None) -> bool:
    return False


def _recipe(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "dataset_size": 5,
        "microbatch_size": 2,
        "accumulation_steps": 2,
        "sampler_seed": 11,
        "total_update_budget": 8,
        "lr": 0.05,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
    }
    body.update(overrides)
    return body


class TinyAdapter(nn.Module):
    def __init__(self, width: int = 3, vocab: int = 8) -> None:
        nn.Module.__init__(self)  # pyright: ignore[reportUnknownMemberType]
        self.proj = nn.Linear(width, vocab)
        self.unused = nn.Linear(width, vocab)
        for parameter in self.unused.parameters():
            parameter.requires_grad_(False)


def _clone_module(module: TinyAdapter) -> TinyAdapter:
    cloned = TinyAdapter(width=int(module.proj.in_features), vocab=int(module.proj.out_features))
    cloned.load_state_dict({key: value.detach().clone() for key, value in module.state_dict().items()})
    return cloned


def _state(adapter: nn.Module, **overrides: Any) -> QueryPrefixTrainState:
    r = _recipe(**overrides)
    return QueryPrefixTrainState(
        adapter=adapter,
        identity=IDENTITY,
        dataset_size=cast(int, r["dataset_size"]),
        microbatch_size=cast(int, r["microbatch_size"]),
        accumulation_steps=cast(int, r["accumulation_steps"]),
        sampler_seed=cast(int, r["sampler_seed"]),
        total_update_budget=cast(int, r["total_update_budget"]),
        lr=cast(float, r["lr"]),
        betas=cast(tuple[float, float], r["betas"]),
        eps=cast(float, r["eps"]),
        weight_decay=cast(float, r["weight_decay"]),
        grad_clip=cast(float, r["grad_clip"]),
    )


def _restore(
    adapter: nn.Module,
    checkpoint_dir: Path,
    seal: str,
    *,
    identity: dict[str, object] | None = None,
    max_payload_bytes: int = MAX_PAYLOAD_BYTES,
    max_seal_bytes: int = MAX_SEAL_BYTES,
    **overrides: Any,
) -> QueryPrefixTrainState:
    r = _recipe(**overrides)
    return QueryPrefixTrainState.restore(
        adapter=adapter,
        identity=IDENTITY if identity is None else identity,
        dataset_size=cast(int, r["dataset_size"]),
        microbatch_size=cast(int, r["microbatch_size"]),
        accumulation_steps=cast(int, r["accumulation_steps"]),
        sampler_seed=cast(int, r["sampler_seed"]),
        total_update_budget=cast(int, r["total_update_budget"]),
        lr=cast(float, r["lr"]),
        betas=cast(tuple[float, float], r["betas"]),
        eps=cast(float, r["eps"]),
        weight_decay=cast(float, r["weight_decay"]),
        grad_clip=cast(float, r["grad_clip"]),
        checkpoint_dir=checkpoint_dir,
        expected_seal_sha256=seal,
        max_payload_bytes=max_payload_bytes,
        max_seal_bytes=max_seal_bytes,
    )


def _ce_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    vocab = int(logits.shape[-1])
    return F.cross_entropy(
        logits[:, :-1].contiguous().reshape(-1, vocab),
        labels[:, 1:].contiguous().reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )


def _tables() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    features = torch.randn(5, 4, 3, generator=generator)
    labels = torch.full((5, 4), IGNORE_INDEX, dtype=torch.long)
    labels[:, 1] = torch.tensor([0, 1, 2, 3, 4])
    labels[:, 2] = torch.tensor([1, 2, 3, 4, 5])
    labels[3, 3] = 6
    labels[4, 3] = 7
    return features, labels


def _cpu_seed(seed: int) -> None:
    cast(Any, torch.random).default_generator.manual_seed(seed)


def _prepare(
    adapter: TinyAdapter,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    exit_on: tuple[int, int] | None = None,
    consume_rng: bool = True,
    mutate: Callable[[torch.Tensor], None] | None = None,
    logits_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Callable[[tuple[int, ...]], list[PreparedMicrobatch]]:
    calls = {"n": 0}

    def prepare(indices: tuple[int, ...]) -> list[PreparedMicrobatch]:
        calls["n"] += 1
        batches: list[PreparedMicrobatch] = []
        width = 2 if len(indices) % 2 == 0 else len(indices)
        groups = len(indices) // width
        for index in range(groups):
            sl = indices[index * width : (index + 1) * width]
            rows = [int(v) for v in sl]
            x = features[rows]
            y = labels[rows].clone()
            update_i = calls["n"]
            micro_i = index

            def forward(
                x: torch.Tensor = x,
                y: torch.Tensor = y,
                update_i: int = update_i,
                micro_i: int = micro_i,
            ) -> torch.Tensor:
                if exit_on == (update_i, micro_i):
                    os._exit(INTERRUPT_CODE)
                if mutate is not None:
                    mutate(y)
                hidden = adapter.proj(x)
                if logits_fn is not None:
                    return logits_fn(hidden)
                if consume_rng:
                    py = random.random()
                    npv = float(np.random.random())
                    tv = float(torch.rand((), dtype=torch.float32, device="cpu").item())
                    noise = 0.05 * (py + npv + tv)
                    scale = torch.linspace(
                        -1.0,
                        1.0,
                        hidden.shape[-1],
                        dtype=hidden.dtype,
                        device=hidden.device,
                    )
                    return hidden + noise * scale
                return hidden

            batches.append(PreparedMicrobatch(labels=y, forward=forward))
        return batches

    return prepare


def _cpu_tree(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in cast(dict[object, object], value).items()}
    if type(value) is list:
        return [_cpu_tree(item) for item in cast(list[object], value)]
    if type(value) is tuple:
        return tuple(_cpu_tree(item) for item in cast(tuple[object, ...], value))
    return value


def _snapshot(state: QueryPrefixTrainState, path: Path) -> None:
    adapter = cast(nn.Module, getattr(state, "_adapter"))
    optimizer = getattr(state, "_optimizer")
    permutation = cast(torch.Tensor, getattr(state, "_permutation"))
    generator = getattr(state, "_generator")
    payload = {
        "adapter": _cpu_tree(adapter.state_dict()),
        "opt": _cpu_tree(optimizer.state_dict()),
        "updates": getattr(state, "_updates"),
        "tokens": getattr(state, "_tokens"),
        "examples": getattr(state, "_examples"),
        "epoch": getattr(state, "_epoch"),
        "cursor": getattr(state, "_cursor"),
        "consumed": getattr(state, "_consumed_rows"),
        "perm": permutation.detach().cpu().clone(),
        "gen": generator.get_state().detach().cpu().clone(),
        "python_gauss_cache": random.getstate()[2],
        "next_python": random.random(),
        "next_numpy": float(np.random.random()),
        "next_torch": float(torch.rand((), dtype=torch.float32, device="cpu").item()),
        "next_gauss": random.gauss(0.0, 1.0),
    }
    torch.save(payload, path)


def _equal_tree(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(left.dtype == right.dtype and torch.equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        left_d = cast(dict[object, object], left)
        right_d = cast(dict[object, object], right)
        if left_d.keys() != right_d.keys():
            return False
        for key in left_d:
            if not _equal_tree(left_d[key], right_d[key]):
                return False
        return True
    if type(left) is list and type(right) is list:
        left_l = cast(list[object], left)
        right_l = cast(list[object], right)
        if len(left_l) != len(right_l):
            return False
        for index in range(len(left_l)):
            if not _equal_tree(left_l[index], right_l[index]):  # pyright: ignore[reportUnknownArgumentType]
                return False
        return True
    if type(left) is tuple and type(right) is tuple:
        left_t = cast(tuple[object, ...], left)
        right_t = cast(tuple[object, ...], right)
        if len(left_t) != len(right_t):
            return False
        for index in range(len(left_t)):
            if not _equal_tree(left_t[index], right_t[index]):  # pyright: ignore[reportUnknownArgumentType]
                return False
        return True
    return bool(left == right)


def _reseal(
    root: Path,
    source: Path,
    name: str,
    mutate: Callable[[dict[str, Any]], None],
) -> tuple[Path, str]:
    payload = torch.load(
        io.BytesIO((source / PAYLOAD_FILENAME).read_bytes()),
        map_location="cpu",
        weights_only=True,
    )
    mutate(payload)
    target = root / name
    target.mkdir()
    buf = io.BytesIO()
    torch.save(payload, buf)
    data = buf.getvalue()
    (target / PAYLOAD_FILENAME).write_bytes(data)
    source_seal = json.loads((source / SEAL_FILENAME).read_text(encoding="ascii"))
    seal_obj = {
        "schema": source_seal["schema"],
        "payload_filename": PAYLOAD_FILENAME,
        "payload_sha256": hashlib.sha256(data).hexdigest(),
        "payload_bytes": len(data),
        "identity_sha256": source_seal["identity_sha256"],
        "recipe_sha256": source_seal["recipe_sha256"],
        "bound_sha256": source_seal["bound_sha256"],
    }
    seal_bytes = json.dumps(
        seal_obj,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii") + b"\n"
    (target / SEAL_FILENAME).write_bytes(seal_bytes)
    return target, hashlib.sha256(seal_bytes).hexdigest()


def _spawn(mode: str, ckpt: Path, snap: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONOPTIMIZE"] = "0"
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "1"
    return subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", mode, str(ckpt), str(snap)],
        cwd=str(REPO),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _worker(mode: str, ckpt_s: str, snap_s: str) -> int:
    _limit_as()
    torch.set_num_threads(2)
    ckpt = Path(ckpt_s)
    snap = Path(snap_s)
    features, labels = _tables()
    if mode == "uninterrupted":
        _cpu_seed(0)
        np.random.seed(0)
        random.seed(0)
        adapter = TinyAdapter()
        random.seed(123)
        np.random.seed(123)
        _cpu_seed(123)
        state = _state(adapter)
        random.gauss(0.0, 1.0)
        if type(random.getstate()[2]) is not float:
            raise RuntimeError("expected_nonempty_python_gaussian_cache")
        prepare = _prepare(adapter, features, labels)
        state.update(prepare, cancelled=_alive)
        if type(random.getstate()[2]) is not float:
            raise RuntimeError("gaussian_cache_lost_before_publish")
        seal = state.publish(ckpt)
        ckpt.with_name(ckpt.name + ".sealhash").write_text(seal + "\n", encoding="ascii")
        state.update(prepare, cancelled=_alive)
        state.update(prepare, cancelled=_alive)
        _snapshot(state, snap)
        return 0
    adapter = TinyAdapter()
    seal = ckpt.with_name(ckpt.name + ".sealhash").read_text(encoding="ascii").strip()
    state = _restore(adapter, ckpt, seal)
    if mode == "interrupt":
        prepare = _prepare(adapter, features, labels, exit_on=(1, 1))
        state.update(prepare, cancelled=_alive)
        return 1
    if mode == "resume":
        prepare = _prepare(adapter, features, labels)
        state.update(prepare, cancelled=_alive)
        state.update(prepare, cancelled=_alive)
        _snapshot(state, snap)
        return 0
    return 2


class QueryPrefixTrainStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)

    def test_window_token_normalization_and_single_step(self) -> None:
        _cpu_seed(3)
        adapter = TinyAdapter(width=3, vocab=6)
        endpoint = nn.Linear(6, 6)
        for parameter in endpoint.parameters():
            parameter.requires_grad_(False)
        unused_before = {key: value.detach().clone() for key, value in adapter.unused.state_dict().items()}
        endpoint_before = {key: value.detach().clone() for key, value in endpoint.state_dict().items()}
        correct = _clone_module(adapter)
        wrong = _clone_module(adapter)
        x0 = torch.randn(1, 6, 3)
        x1 = torch.randn(1, 6, 3)
        lab0 = torch.full((1, 6), IGNORE_INDEX, dtype=torch.long)
        lab0[0, 1] = 1
        lab0[0, 2] = 2
        lab1 = torch.full((1, 6), IGNORE_INDEX, dtype=torch.long)
        lab1[0, 1:6] = torch.arange(5)
        features = torch.cat([x0, x1], dim=0)
        labels = torch.cat([lab0, lab1], dim=0)

        def nll(module: TinyAdapter, x: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
            return _ce_sum(endpoint(module.proj(x)).float(), lab)

        s0 = nll(correct, x0, lab0)
        s1 = nll(correct, x1, lab1)
        ((s0 + s1) / 7.0).backward()  # pyright: ignore[reportUnknownMemberType]
        clip_grad_norm_(list(correct.proj.parameters()), max_norm=0.25, error_if_nonfinite=True)
        opt_c = AdamW(
            list(correct.proj.parameters()),
            lr=0.02,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
            foreach=False,
            fused=False,
        )
        opt_c.step()  # pyright: ignore[reportUnknownMemberType]
        w0 = nll(wrong, x0, lab0)
        w1 = nll(wrong, x1, lab1)
        (w0 / 2.0).backward()  # pyright: ignore[reportUnknownMemberType]
        (w1 / 5.0).backward()  # pyright: ignore[reportUnknownMemberType]
        clip_grad_norm_(list(wrong.proj.parameters()), max_norm=0.25, error_if_nonfinite=True)
        opt_w = AdamW(
            list(wrong.proj.parameters()),
            lr=0.02,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
            foreach=False,
            fused=False,
        )
        opt_w.step()  # pyright: ignore[reportUnknownMemberType]

        state = QueryPrefixTrainState(
            adapter=adapter,
            identity=IDENTITY,
            dataset_size=2,
            microbatch_size=1,
            accumulation_steps=2,
            sampler_seed=0,
            total_update_budget=3,
            lr=0.02,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            grad_clip=0.25,
        )
        steps = {"n": 0}
        optimizer = getattr(state, "_optimizer")
        original_step = optimizer.step

        def counted_step(*args: Any, **kwargs: Any) -> Any:
            steps["n"] += 1
            return original_step(*args, **kwargs)

        optimizer.step = counted_step
        clips = {"n": 0}
        import src.transfer.query_prefix_train_state as qpts

        real_clip = qpts._clip_grad_norm_  # pyright: ignore[reportPrivateUsage]

        def counted_clip(
            parameters: Any,
            max_norm: float,
            norm_type: float = 2.0,
            error_if_nonfinite: bool = False,
            foreach: bool | None = None,
        ) -> torch.Tensor:
            clips["n"] += 1
            return real_clip(
                parameters,
                max_norm=max_norm,
                norm_type=norm_type,
                error_if_nonfinite=error_if_nonfinite,
                foreach=foreach,
            )

        def prepare(indices: tuple[int, ...]) -> list[PreparedMicrobatch]:
            batches: list[PreparedMicrobatch] = []
            for idx in indices:
                x = features[int(idx) : int(idx) + 1]
                y = labels[int(idx) : int(idx) + 1].clone()

                def forward(x: torch.Tensor = x, y: torch.Tensor = y) -> torch.Tensor:
                    return endpoint(adapter.proj(x))

                batches.append(PreparedMicrobatch(labels=y, forward=forward))
            return batches

        with mock.patch.object(qpts, "_clip_grad_norm_", counted_clip):
            metrics = state.update(prepare, cancelled=_alive)
        self.assertEqual(clips["n"], 1)
        self.assertEqual(steps["n"], 1)
        self.assertEqual(metrics["window_tokens"], 7)
        self.assertEqual(metrics["updates"], 1)
        self.assertEqual(metrics["completed_tokens"], 7)
        self.assertEqual(metrics["completed_examples"], 2)
        self.assertEqual(metrics["loss_unit"], LOSS_UNIT)
        self.assertTrue(state.at_boundary)
        for key, value in adapter.state_dict().items():
            if not key.startswith("proj."):
                continue
            self.assertTrue(torch.equal(value.detach().cpu(), correct.state_dict()[key].detach().cpu()))
            self.assertFalse(torch.equal(value.detach().cpu(), wrong.state_dict()[key].detach().cpu()))
        for key, value in adapter.unused.state_dict().items():
            self.assertTrue(torch.equal(value.detach().cpu(), unused_before[key]))
        for parameter in adapter.unused.parameters():
            self.assertIsNone(parameter.grad)
        for parameter in endpoint.parameters():
            self.assertIsNone(parameter.grad)
        for key, value in endpoint.state_dict().items():
            self.assertTrue(torch.equal(value.detach().cpu(), endpoint_before[key].cpu()))
        self.assertEqual(len(getattr(state, "_optimizer").param_groups[0]["params"]), 2)

    def test_fresh_process_resume_matches_uninterrupted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ckpt = root / "boundary1"
            unint = root / "unint.pt"
            resume = root / "resume.pt"
            first = _spawn("uninterrupted", ckpt, unint)
            self.assertEqual(first.returncode, 0, msg=first.stderr)
            second = _spawn("resume", ckpt, resume)
            self.assertEqual(second.returncode, 0, msg=second.stderr)
            left = torch.load(unint, map_location="cpu", weights_only=True)
            right = torch.load(resume, map_location="cpu", weights_only=True)
            self.assertTrue(_equal_tree(left["adapter"], right["adapter"]))
            self.assertTrue(_equal_tree(left["opt"], right["opt"]))
            self.assertEqual(left["updates"], right["updates"])
            self.assertEqual(left["tokens"], right["tokens"])
            self.assertEqual(left["examples"], right["examples"])
            self.assertEqual(left["epoch"], right["epoch"])
            self.assertEqual(left["cursor"], right["cursor"])
            self.assertTrue(torch.equal(left["perm"], right["perm"]))
            self.assertTrue(torch.equal(left["gen"], right["gen"]))
            self.assertEqual(left["next_python"], right["next_python"])
            self.assertEqual(left["next_numpy"], right["next_numpy"])
            self.assertEqual(left["next_torch"], right["next_torch"])
            self.assertIsInstance(left["python_gauss_cache"], float)
            self.assertEqual(left["python_gauss_cache"], right["python_gauss_cache"])
            self.assertEqual(left["next_gauss"], right["next_gauss"])
            self.assertGreaterEqual(int(left["epoch"]), 1)
            self.assertNotEqual(int(left["cursor"]), 0)
            self.assertEqual(int(left["updates"]), 3)

    def test_interrupted_window_replays_from_sealed_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ckpt = root / "boundary1"
            unint = root / "unint.pt"
            replay = root / "replay.pt"
            dummy = root / "dummy.pt"
            first = _spawn("uninterrupted", ckpt, unint)
            self.assertEqual(first.returncode, 0, msg=first.stderr)
            interrupted = _spawn("interrupt", ckpt, dummy)
            self.assertEqual(interrupted.returncode, INTERRUPT_CODE, msg=interrupted.stderr)
            self.assertFalse((root / "dummy.pt").exists())
            resumed = _spawn("resume", ckpt, replay)
            self.assertEqual(resumed.returncode, 0, msg=resumed.stderr)
            left = torch.load(unint, map_location="cpu", weights_only=True)
            right = torch.load(replay, map_location="cpu", weights_only=True)
            self.assertTrue(_equal_tree(left, right))

    def test_refusals_grouped(self) -> None:
        _cpu_seed(4)
        features, labels = _tables()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)

            with self.subTest("constructor_types"):
                adapter = TinyAdapter()
                with self.assertRaises(QueryPrefixTrainStateError):
                    _state(adapter, dataset_size=True)
                with self.assertRaises(QueryPrefixTrainStateError):
                    _state(adapter, lr=float("nan"))

            adapter = TinyAdapter()
            state = _state(adapter, total_update_budget=1)
            prepare = _prepare(adapter, features, labels, consume_rng=False)
            state.update(prepare, cancelled=_alive)
            with self.subTest("exhausted_budget"):
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "budget_consumed"):
                    state.update(prepare, cancelled=_alive)
                self.assertTrue(state.at_boundary)
                seal = state.publish(root / "ok")
                self.assertEqual(len(seal), 64)

            with self.subTest("occupied_and_traversal_and_symlink"):
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "occupied_checkpoint"):
                    state.publish(root / "ok")
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "path_traversal"):
                    state.publish(root / ".." / "escaped")
                real_parent = root / "real"
                real_parent.mkdir()
                linked = root / "linked"
                os.symlink(real_parent, linked)
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "symlink_ancestor"):
                    state.publish(linked / "ckpt")

            with self.subTest("identity_config_layout"):
                other = TinyAdapter()
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "identity_mismatch"):
                    _restore(
                        other,
                        root / "ok",
                        seal,
                        identity={**IDENTITY, "architecture": "other"},
                        total_update_budget=1,
                    )
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "config_mismatch"):
                    _restore(TinyAdapter(), root / "ok", seal, total_update_budget=1, lr=0.01)
                wide = TinyAdapter(width=4, vocab=8)
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "layout_mismatch"):
                    _restore(wide, root / "ok", seal, total_update_budget=1)

            with self.subTest("missing_partial_corrupt_overcap"):
                missing = root / "empty"
                missing.mkdir()
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "missing_checkpoint_file"):
                    _restore(TinyAdapter(), missing, seal, total_update_budget=1)
                partial = root / "partial"
                os.mkdir(partial)
                Path(partial, SEAL_FILENAME).write_bytes(b"{")
                with self.assertRaises(QueryPrefixTrainStateError):
                    _restore(TinyAdapter(), partial, "a" * 64, total_update_budget=1)
                payload_path = root / "ok" / PAYLOAD_FILENAME
                original = payload_path.read_bytes()
                mutated = bytearray(original)
                mutated[-1] = mutated[-1] ^ 0x01
                os.chmod(payload_path, stat.S_IRUSR | stat.S_IWUSR)
                payload_path.write_bytes(bytes(mutated))
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "payload_hash_mismatch"):
                    _restore(TinyAdapter(), root / "ok", seal, total_update_budget=1)
                payload_path.write_bytes(original)
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "oversize_file"):
                    _restore(
                        TinyAdapter(),
                        root / "ok",
                        seal,
                        max_payload_bytes=1,
                        total_update_budget=1,
                    )

            with self.subTest("failed_update_then_refuse"):
                live = TinyAdapter()
                failing = _state(live)

                def boom(_indices: tuple[int, ...]) -> list[PreparedMicrobatch]:
                    raise RuntimeError("prepare_failed")

                with self.assertRaises(RuntimeError):
                    failing.update(boom, cancelled=_alive)
                self.assertTrue(failing.failed)
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "failed_state"):
                    failing.update(prepare, cancelled=_alive)
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "failed_state"):
                    failing.publish(root / "failed_pub")

            with self.subTest("zero_changed_nonfinite_cancel_sampler"):
                zero_adapter = TinyAdapter()
                zero_state = _state(zero_adapter)
                zero_labels = torch.full_like(labels, IGNORE_INDEX)

                def zero_prepare(indices: tuple[int, ...]) -> list[PreparedMicrobatch]:
                    batches: list[PreparedMicrobatch] = []
                    for offset in range(0, 4, 2):
                        y = zero_labels[list(indices[offset : offset + 2])].clone()
                        x = features[list(indices[offset : offset + 2])]
                        batches.append(
                            PreparedMicrobatch(
                                labels=y,
                                forward=lambda x=x: zero_adapter.proj(x),
                            )
                        )
                    return batches

                with self.assertRaisesRegex(QueryPrefixTrainStateError, "zero_tokens"):
                    zero_state.update(zero_prepare, cancelled=_alive)

                changed_adapter = TinyAdapter()
                changed_state = _state(changed_adapter)

                def changer(y: torch.Tensor) -> None:
                    y[:, 3] = 1

                with self.assertRaisesRegex(QueryPrefixTrainStateError, "token_count_changed"):
                    changed_state.update(
                        _prepare(changed_adapter, features, labels, consume_rng=False, mutate=changer),
                        cancelled=_alive,
                    )

                nan_adapter = TinyAdapter()
                nan_state = _state(nan_adapter)
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "nonfinite_loss"):
                    nan_state.update(
                        _prepare(
                            nan_adapter,
                            features,
                            labels,
                            consume_rng=False,
                            logits_fn=lambda hidden: hidden * float("nan"),
                        ),
                        cancelled=_alive,
                    )

                grad_adapter = TinyAdapter()
                grad_state = _state(grad_adapter)
                def _inf_hook(grad: torch.Tensor) -> torch.Tensor:
                    return torch.full_like(grad, float("inf"))

                grad_adapter.proj.weight.register_hook(_inf_hook)  # pyright: ignore[reportUnknownMemberType]
                with self.assertRaises(Exception):
                    grad_state.update(
                        _prepare(grad_adapter, features, labels, consume_rng=False),
                        cancelled=_alive,
                    )
                self.assertTrue(grad_state.failed)

                cancel_adapter = TinyAdapter()
                cancel_state = _state(cancel_adapter)
                hits = {"n": 0}

                def cancel() -> bool:
                    hits["n"] += 1
                    return hits["n"] > 1

                with self.assertRaisesRegex(QueryPrefixTrainStateError, "cancelled"):
                    cancel_state.update(
                        _prepare(cancel_adapter, features, labels, consume_rng=False),
                        cancelled=cancel,
                    )
                self.assertTrue(cancel_state.failed)

                good = TinyAdapter()
                publisher = _state(good)
                publisher.update(_prepare(good, features, labels, consume_rng=False), cancelled=_alive)
                src = root / "sampler_src"
                publisher.publish(src)
                payload = torch.load(
                    io.BytesIO((src / PAYLOAD_FILENAME).read_bytes()),
                    map_location="cpu",
                    weights_only=True,
                )
                perm = payload["sampler"]["permutation"].clone()
                perm[0] = perm[1]
                payload["sampler"]["permutation"] = perm
                bad_dir = root / "bad_sampler"
                os.mkdir(bad_dir)
                buf = io.BytesIO()
                torch.save(payload, buf)
                data = buf.getvalue()
                (bad_dir / PAYLOAD_FILENAME).write_bytes(data)
                digest = hashlib.sha256(data).hexdigest()
                seal_obj = {
                    "schema": "query_prefix_train_state_seal_v1",
                    "payload_filename": PAYLOAD_FILENAME,
                    "payload_sha256": digest,
                    "payload_bytes": len(data),
                    "identity_sha256": getattr(publisher, "_identity_sha256"),
                    "recipe_sha256": getattr(publisher, "_recipe_sha256"),
                    "bound_sha256": publisher.bound_sha256,
                }
                seal_bytes = json.dumps(
                    seal_obj,
                    allow_nan=False,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii") + b"\n"
                (bad_dir / SEAL_FILENAME).write_bytes(seal_bytes)
                bad_hash = hashlib.sha256(seal_bytes).hexdigest()
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "bad_sampler"):
                    _restore(TinyAdapter(), bad_dir, bad_hash)

    def test_boundary_optimizer_and_seal_refusals(self) -> None:
        _cpu_seed(5)
        features, labels = _tables()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            adapter = TinyAdapter()
            state = _state(adapter)
            state.update(_prepare(adapter, features, labels, consume_rng=False), cancelled=_alive)
            src = root / "good"
            seal = state.publish(src)

            with self.subTest("wrong_lr"):
                path, digest = _reseal(
                    root, src, "wrong_lr", lambda body: body["optimizer"]["param_groups"][0].__setitem__("lr", 999.0)
                )
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "optimizer_state"):
                    _restore(TinyAdapter(), path, digest)

            with self.subTest("missing_exp_avg"):
                def missing(body: dict[str, Any]) -> None:
                    del next(iter(body["optimizer"]["state"].values()))["exp_avg"]

                path, digest = _reseal(root, src, "missing_moment", missing)
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "optimizer_state"):
                    _restore(TinyAdapter(), path, digest)

            with self.subTest("misassociated_extra_state"):
                def extra(body: dict[str, Any]) -> None:
                    first = next(iter(body["optimizer"]["state"].values()))
                    body["optimizer"]["state"][99] = {
                        key: (value.clone() if isinstance(value, torch.Tensor) else value)
                        for key, value in first.items()
                    }

                path, digest = _reseal(root, src, "extra_state", extra)
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "optimizer_state"):
                    _restore(TinyAdapter(), path, digest)

            with self.subTest("wrong_shaped_moment"):
                def wrong_shape(body: dict[str, Any]) -> None:
                    first = next(iter(body["optimizer"]["state"].values()))
                    first["exp_avg"] = torch.zeros(1, dtype=torch.float32, device="cpu")

                path, digest = _reseal(root, src, "wrong_shape", wrong_shape)
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "optimizer_state"):
                    _restore(TinyAdapter(), path, digest)

            with self.subTest("nonfinite_moment"):
                def nonfinite(body: dict[str, Any]) -> None:
                    first = next(iter(body["optimizer"]["state"].values()))
                    first["exp_avg"] = torch.full_like(first["exp_avg"], float("nan"))

                path, digest = _reseal(root, src, "nan_moment", nonfinite)
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "optimizer_state"):
                    _restore(TinyAdapter(), path, digest)

            with self.subTest("fractional_step"):
                def fractional(body: dict[str, Any]) -> None:
                    next(iter(body["optimizer"]["state"].values()))["step"] = torch.tensor(
                        1.5, dtype=torch.float32, device="cpu"
                    )

                path, digest = _reseal(root, src, "fractional_step", fractional)
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "optimizer_state"):
                    _restore(TinyAdapter(), path, digest)

            with self.subTest("wrong_identity_config_layout_before_load"):
                import src.transfer.query_prefix_train_state as qpts

                calls = {"n": 0}
                real_load = qpts.torch.load

                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    calls["n"] += 1
                    return real_load(*args, **kwargs)

                with mock.patch.object(qpts.torch, "load", wrapped):
                    with self.assertRaisesRegex(QueryPrefixTrainStateError, "identity_mismatch"):
                        _restore(
                            TinyAdapter(),
                            src,
                            seal,
                            identity={**IDENTITY, "architecture": "other"},
                        )
                    with self.assertRaisesRegex(QueryPrefixTrainStateError, "config_mismatch"):
                        _restore(TinyAdapter(), src, seal, lr=0.01)
                    with self.assertRaisesRegex(QueryPrefixTrainStateError, "layout_mismatch"):
                        _restore(TinyAdapter(width=4, vocab=8), src, seal)
                self.assertEqual(calls["n"], 0)

            with self.subTest("reversed_live_optimizer_params"):
                reversed_adapter = TinyAdapter()
                reversed_state = _state(reversed_adapter)
                reversed_state.update(_prepare(reversed_adapter, features, labels, consume_rng=False), cancelled=_alive)
                group = reversed_state._optimizer.param_groups[0]  # pyright: ignore[reportPrivateUsage]
                group["params"] = list(reversed(group["params"]))
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "optimizer_state"):
                    reversed_state.publish(root / "reversed_params")
                self.assertFalse((root / "reversed_params" / SEAL_FILENAME).exists())

            with self.subTest("boolean_step_tensor"):
                def boolean_step(body: dict[str, Any]) -> None:
                    next(iter(body["optimizer"]["state"].values()))["step"] = torch.tensor(
                        True, device="cpu"
                    )

                path, digest = _reseal(root, src, "boolean_step", boolean_step)
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "optimizer_state"):
                    _restore(TinyAdapter(), path, digest)

            with self.subTest("empty_cuda_rng_before_install"):
                import src.transfer.query_prefix_train_state as qpts

                captured = qpts._capture_rng(cuda_index=None)  # pyright: ignore[reportPrivateUsage]
                captured["cuda_index"] = 0
                captured["torch_cuda"] = torch.tensor([], dtype=torch.uint8, device="cpu")
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "torch_cuda_rng"):
                    qpts._validated_rng(captured, cuda_index=0)  # pyright: ignore[reportPrivateUsage]

            with self.subTest("refuse_nonfinite_and_changed_layout_publish"):
                nan_adapter = TinyAdapter()
                nan_state = _state(nan_adapter)
                with torch.no_grad():
                    nan_adapter.proj.weight.fill_(float("nan"))
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "nonfinite_state_tensor"):
                    nan_state.publish(root / "nan_pub")
                self.assertFalse((root / "nan_pub" / SEAL_FILENAME).exists())
                layout_adapter = TinyAdapter()
                layout_state = _state(layout_adapter)
                for parameter in layout_adapter.unused.parameters():
                    parameter.requires_grad_(True)
                with self.assertRaisesRegex(QueryPrefixTrainStateError, "layout_mismatch"):
                    layout_state.publish(root / "layout_pub")
                self.assertFalse((root / "layout_pub" / SEAL_FILENAME).exists())

    def test_ambient_meta_keeps_cpu_control_tensors(self) -> None:
        _cpu_seed(6)
        adapter = TinyAdapter()
        with torch.device("meta"):
            state = _state(adapter)
            permutation = cast(torch.Tensor, getattr(state, "_permutation"))
            self.assertEqual(str(permutation.device), "cpu")
            import src.transfer.query_prefix_train_state as qpts

            rng = qpts._capture_rng(cuda_index=None)  # pyright: ignore[reportPrivateUsage]
            python_mt = cast(torch.Tensor, cast(dict[str, object], rng["python"])["mt"])
            numpy_keys = cast(torch.Tensor, cast(dict[str, object], rng["numpy"])["keys"])
            self.assertEqual(str(python_mt.device), "cpu")
            self.assertEqual(str(numpy_keys.device), "cpu")
            sampler = state._sampler_payload()  # pyright: ignore[reportPrivateUsage]
            self.assertEqual(str(cast(torch.Tensor, sampler["permutation"]).device), "cpu")
            self.assertEqual(str(cast(torch.Tensor, sampler["generator_state"]).device), "cpu")
        self.assertFalse(state.failed)
        self.assertEqual(str(cast(torch.Tensor, getattr(state, "_permutation")).device), "cpu")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        raise SystemExit(_worker(sys.argv[2], sys.argv[3], sys.argv[4]))
    unittest.main()
