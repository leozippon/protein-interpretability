#!/usr/bin/env python3
"""Native query-prefix recovery stage: CPU controller plus three sequential workers.

Queue-injected CLI is only ``--device`` and ``--out``. The controller never
initializes CUDA, never creates a new session, and observes actual exits 0/75/0
inside one outer process group. Dummy workers exist for CPU tests and do not
load pretrained weights. Native workers are the dispatch path and are not
exercised by the CPU controller tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, cast
import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE_PATH = Path(__file__).resolve()
FIXTURE_PATH = REPO_ROOT / "scripts/transfer/native_query_prefix_recovery.fixture.json"
DONOR_ENV = "NATIVE_QUERY_RECOVERY_DONOR"
RECEIVER_ENV = "NATIVE_QUERY_RECOVERY_RECEIVER"
FIXTURE_SHA256 = "9faa01d99d78d72761e802c28bbc7374e8b9b3cd8240586c487cb33cfdd0eeab"
EXPECTED_FINGERPRINT = "784b2e8a704bc0edf91a31fc45c4fd897e4b62e0737cde509aee7f98743924b9"
CLAIM_DIRNAME = "native_query_recovery_cell"
REPORT_NAME = "report.json"
ROLES = ("continuous", "interrupted", "resume")
EXPECTED_EXITS = (0, 75, 0)
FAULT_EXIT = 75
SEED = 20260912
DATASET_SIZE = 4
MICROBATCH_SIZE = 1
ACCUMULATION_STEPS = 3
TOTAL_UPDATE_BUDGET = 4
LR = 0.001
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
PREFIX_QUERIES = 8
INPUT_WIDTH = 1536
INTERNAL_WIDTH = 256
OUTPUT_WIDTH = 3584
N_HEADS = 8
MAX_TRAINABLE = 1712384
BLOCK_INDEX = 26
MAX_RESIDUES = 512
MAX_TOKENS = 513
MAX_PROMPT_TOKENS = 128
MAX_ANSWER_TOKENS = 16
MAX_RECEIVER_CONTEXT = 1024
MAX_TENSOR_BYTES = 4 * 1024 * 1024
EXAMPLE_ORDER = ("A/short", "A/long", "B/short", "B/long")
ANSWER_IDS = (
    [7141, 151645],
    [7141, 2038, 151645],
    [19127, 151645],
    [19127, 2038, 151645],
)
SHIFTED_SUPPORT = (2, 3, 2, 3)
PROMPT_LENGTHS = (34, 42, 34, 42)
DONOR_TOKEN_COUNTS = {"engineering-A": 289, "engineering-B": 318}
RESIDUE_COUNTS = {"engineering-A": 288, "engineering-B": 317}
SEQUENCE_SHA256 = {
    "engineering-A": "49f369397e01dce56b3e066db6986f2be52b3c62a8196adb2bafe405e5242dcd",
    "engineering-B": "ada13df3337a474feabe684060ae1cc1b32c4ead9ffcd955df5dee5787b1f847",
}
WINDOW_ROWS = ([2, 0, 3], [1, 1, 2], [3, 0, 3], [2, 0, 1])
UPDATE2 = {"epoch": 1, "cursor": 2, "consumed": 6}
RECEIVER_EOS_ID = 151645
DONOR_PAD_ID = 30
RECEIVER_PAD_ID = 151643
MODEL_CALL_BUDGET = {
    "donor_forwards": 4,
    "receiver_forwards": 29,
    "adapter_backwards": 27,
}
ROLE_BUDGET = {
    "continuous": {
        "donor_forwards": 2,
        "receiver_forwards": 16,
        "adapter_backwards": 14,
        "completed_updates": 4,
    },
    "interrupted": {
        "donor_forwards": 1,
        "receiver_forwards": 7,
        "adapter_backwards": 7,
        "completed_updates": 2,
    },
    "resume": {
        "donor_forwards": 1,
        "receiver_forwards": 6,
        "adapter_backwards": 6,
        "completed_updates": 2,
    },
}
ROLE_ENV = "NATIVE_QUERY_RECOVERY_ROLE"
IMPL_ENV = "NATIVE_QUERY_RECOVERY_IMPL"
CLAIM_ENV = "NATIVE_QUERY_RECOVERY_CLAIM"
FIXTURE_ENV = "NATIVE_QUERY_RECOVERY_FIXTURE"
ENV_IMPL_DUMMY = "dummy"
ENV_IMPL_NATIVE = "native"
HEX64 = 64

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class RecoveryStageError(RuntimeError):
    """Fail-closed stage error with a stable code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> NoReturn:
    raise RecoveryStageError(code)


def _require(condition: bool, code: str) -> None:
    if not condition:
        _fail(code)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def write_bytes_exclusive(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    fd = os.open(str(path), flags, 0o644)
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


def write_text_exclusive(path: Path, text: str) -> None:
    write_bytes_exclusive(path, text.encode("ascii"))


def write_json_exclusive(path: Path, value: object) -> None:
    payload = json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    write_bytes_exclusive(path, payload.encode("ascii"))


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _require_real_dir(path: Path, *, code: str) -> None:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise RecoveryStageError(code) from exc
    _require(not stat.S_ISLNK(info.st_mode), "symlink_path")
    _require(stat.S_ISDIR(info.st_mode), code)


def _require_real_file(path: Path, *, code: str) -> None:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise RecoveryStageError(code) from exc
    _require(not stat.S_ISLNK(info.st_mode), "symlink_path")
    _require(stat.S_ISREG(info.st_mode), code)


def _int_list(value: object, *, code: str) -> list[int]:
    _require(type(value) is list, code)
    rows = cast(list[object], value)
    out: list[int] = []
    for item in rows:
        _require(type(item) is int and type(item) is not bool, code)
        out.append(int(cast(int, item)))
    return out


def _text(value: object, *, code: str) -> str:
    _require(type(value) is str, code)
    return cast(str, value)


def _int(value: object, *, code: str) -> int:
    _require(type(value) is int and type(value) is not bool, code)
    return int(cast(int, value))


def fixture_path() -> Path:
    override = os.environ.get(FIXTURE_ENV)
    if override:
        return Path(override)
    return FIXTURE_PATH


def load_fixture() -> dict[str, Any]:
    path = fixture_path()
    _require_real_file(path, code="missing_fixture")
    data = path.read_bytes()
    digest = _sha256_bytes(data)
    _require(digest == FIXTURE_SHA256, "fixture_sha256_mismatch")
    parsed = json.loads(data.decode("utf-8"))
    _require(type(parsed) is dict, "fixture_not_object")
    return cast(dict[str, Any], parsed)


def bind_fixture(payload: Mapping[str, Any]) -> dict[str, Any]:
    order = payload.get("example_order")
    _require(type(order) is list, "example_order")
    _require(tuple(cast(list[object], order)) == EXAMPLE_ORDER, "example_order")
    examples_obj = payload.get("examples")
    _require(type(examples_obj) is list and len(cast(list[object], examples_obj)) == 4, "examples")
    examples = cast(list[object], examples_obj)
    bound_examples: list[dict[str, Any]] = []
    for index, raw in enumerate(examples):
        _require(type(raw) is dict, "example_row")
        row = cast(dict[str, Any], raw)
        _require(_text(row.get("example_id"), code="example_id") == EXAMPLE_ORDER[index], "example_id")
        accession = _text(row.get("accession"), code="accession")
        _require(accession in DONOR_TOKEN_COUNTS, "accession")
        _require(
            _int(row.get("native_donor_token_count"), code="native_donor_token_count")
            == DONOR_TOKEN_COUNTS[accession],
            "native_donor_token_count",
        )
        _require(
            _text(row.get("sequence_sha256"), code="sequence_sha256") == SEQUENCE_SHA256[accession],
            "sequence_sha256",
        )
        prompt_ids = _int_list(row.get("prompt_ids"), code="prompt_ids")
        answer_ids = _int_list(row.get("answer_ids"), code="answer_ids")
        _require(len(prompt_ids) == PROMPT_LENGTHS[index], "prompt_length")
        _require(len(prompt_ids) <= MAX_PROMPT_TOKENS, "prompt_overflow")
        _require(answer_ids == ANSWER_IDS[index], "answer_ids")
        _require(len(answer_ids) <= MAX_ANSWER_TOKENS, "answer_overflow")
        _require(
            _int(row.get("shifted_support_count"), code="shifted_support") == SHIFTED_SUPPORT[index],
            "shifted_support",
        )
        _require(_int(row.get("answer_eos_id"), code="answer_eos") == RECEIVER_EOS_ID, "answer_eos")
        context = PREFIX_QUERIES + len(prompt_ids) + len(answer_ids)
        _require(context <= MAX_RECEIVER_CONTEXT, "receiver_context_overflow")
        _require(_int(row.get("prefix_queries"), code="prefix_queries") == PREFIX_QUERIES, "prefix_queries")
        bound_examples.append(row)
    sequences_obj = payload.get("sequences")
    _require(type(sequences_obj) is dict, "sequences")
    sequences = cast(dict[str, Any], sequences_obj)
    for accession, residues in RESIDUE_COUNTS.items():
        raw_seq = sequences.get(accession)
        _require(type(raw_seq) is dict, "sequence_row")
        row = cast(dict[str, Any], raw_seq)
        _require(_int(row.get("residue_count"), code="residue_count") == residues, "residue_count")
        _require(
            _int(row.get("native_donor_token_count"), code="native_donor_token_count")
            == DONOR_TOKEN_COUNTS[accession],
            "native_donor_token_count",
        )
        _require(
            _text(row.get("sequence_sha256"), code="sequence_sha256") == SEQUENCE_SHA256[accession],
            "sequence_sha256",
        )
        token_ids = _int_list(row.get("native_donor_token_ids"), code="native_donor_token_ids")
        _require(len(token_ids) == DONOR_TOKEN_COUNTS[accession], "native_donor_token_ids")
        _require(len(token_ids) == residues + 1, "native_geometry")
    query_obj = payload.get("query_prefix")
    _require(type(query_obj) is dict, "query_prefix")
    query = cast(dict[str, Any], query_obj)
    fingerprint = _text(query.get("parameter_fingerprint"), code="parameter_fingerprint")
    _require(len(fingerprint) == HEX64, "parameter_fingerprint")
    _require(fingerprint == EXPECTED_FINGERPRINT, "parameter_fingerprint")
    _require(_int(query.get("actual_parameters"), code="actual_parameters") == MAX_TRAINABLE, "actual_parameters")
    _require(_int(query.get("input_width"), code="input_width") == INPUT_WIDTH, "input_width")
    _require(_int(query.get("internal_width"), code="internal_width") == INTERNAL_WIDTH, "internal_width")
    _require(_int(query.get("output_width"), code="output_width") == OUTPUT_WIDTH, "output_width")
    _require(_int(query.get("n_queries"), code="n_queries") == PREFIX_QUERIES, "n_queries")
    _require(_int(query.get("n_heads"), code="n_heads") == N_HEADS, "n_heads")
    sampler_obj = payload.get("sampler_expectation")
    _require(type(sampler_obj) is dict, "sampler_expectation")
    sampler = cast(dict[str, Any], sampler_obj)
    _require(_int(sampler.get("seed"), code="sampler_seed") == SEED, "sampler_seed")
    _require(_int(sampler.get("dataset_size"), code="dataset_size") == DATASET_SIZE, "dataset_size")
    _require(_int(sampler.get("accumulation"), code="accumulation") == ACCUMULATION_STEPS, "accumulation")
    _require(_int(sampler.get("n_updates"), code="n_updates") == TOTAL_UPDATE_BUDGET, "n_updates")
    windows_obj = sampler.get("windows")
    _require(type(windows_obj) is list and len(cast(list[object], windows_obj)) == 4, "windows")
    windows = cast(list[object], windows_obj)
    for index, raw_window in enumerate(windows):
        _require(type(raw_window) is dict, "window")
        window = cast(dict[str, Any], raw_window)
        _require(_int_list(window.get("rows"), code="window_rows") == WINDOW_ROWS[index], "window_rows")
    update2_obj = sampler.get("update2")
    _require(type(update2_obj) is dict, "update2")
    update2 = cast(dict[str, Any], update2_obj)
    _require(_int(update2.get("epoch"), code="update2_epoch") == UPDATE2["epoch"], "update2_epoch")
    _require(_int(update2.get("cursor"), code="update2_cursor") == UPDATE2["cursor"], "update2_cursor")
    _require(_int(update2.get("consumed"), code="update2_consumed") == UPDATE2["consumed"], "update2_consumed")
    _require(_int(payload.get("seed"), code="seed") == SEED, "seed")
    _require(_int(payload.get("receiver_eos_id"), code="receiver_eos_id") == RECEIVER_EOS_ID, "receiver_eos_id")
    donor_padding_obj = payload.get("donor_padding")
    _require(type(donor_padding_obj) is dict, "donor_padding")
    donor_padding = cast(dict[str, Any], donor_padding_obj)
    _require(_int(donor_padding.get("pad_token_id"), code="donor_pad") == DONOR_PAD_ID, "donor_pad")
    receiver_padding_obj = payload.get("receiver_padding")
    _require(type(receiver_padding_obj) is dict, "receiver_padding")
    receiver_padding = cast(dict[str, Any], receiver_padding_obj)
    _require(_int(receiver_padding.get("pad_token_id"), code="receiver_pad") == RECEIVER_PAD_ID, "receiver_pad")
    _require(_int(receiver_padding.get("eos_token_id"), code="receiver_eos") == RECEIVER_EOS_ID, "receiver_eos")
    return {
        "examples": bound_examples,
        "sequences": sequences,
        "fingerprint": fingerprint,
        "query_prefix": query,
        "assets": payload.get("assets"),
    }


def process_identity() -> dict[str, int]:
    pid = os.getpid()
    return {"pid": pid, "pgid": os.getpgid(0), "sid": os.getsid(0)}


def impl_name() -> str:
    value = os.environ.get(IMPL_ENV, ENV_IMPL_NATIVE)
    _require(value in (ENV_IMPL_DUMMY, ENV_IMPL_NATIVE), "unknown_impl")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def train_identity(*, fingerprint: str) -> dict[str, object]:
    return {
        "block_index": BLOCK_INDEX,
        "capture_dtype": "bfloat16",
        "example_order": ",".join(EXAMPLE_ORDER),
        "fixture_sha256": FIXTURE_SHA256,
        "query_prefix_fingerprint": fingerprint,
    }


def recipe_kwargs() -> dict[str, object]:
    return {
        "dataset_size": DATASET_SIZE,
        "microbatch_size": MICROBATCH_SIZE,
        "accumulation_steps": ACCUMULATION_STEPS,
        "sampler_seed": SEED,
        "total_update_budget": TOTAL_UPDATE_BUDGET,
        "lr": LR,
        "betas": BETAS,
        "eps": EPS,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
    }


def run_controller(device: str, out: Path) -> int:
    _require_real_dir(out, code="missing_out")
    report_path = out / REPORT_NAME
    if report_path.exists() or report_path.is_symlink():
        _fail("occupied_report")
    claim = out / CLAIM_DIRNAME
    try:
        os.lstat(claim)
    except FileNotFoundError:
        pass
    else:
        _fail("occupied_claim")
    try:
        os.mkdir(claim, 0o755)
    except FileExistsError as exc:
        raise RecoveryStageError("occupied_claim") from exc
    payload = load_fixture()
    bound = bind_fixture(payload)
    impl = impl_name()
    controller = {
        "claim": str(claim),
        "device": device,
        "fixture_sha256": FIXTURE_SHA256,
        "impl": impl,
        "model_call_budget": MODEL_CALL_BUDGET,
        "process": process_identity(),
        "roles": list(ROLES),
        "start_new_session": False,
    }
    write_json_exclusive(claim / "controller.json", controller)
    _fsync_dir(claim)
    exits: list[int] = []
    python = sys.executable
    for role, expected in zip(ROLES, EXPECTED_EXITS, strict=True):
        child_env = os.environ.copy()
        child_env[ROLE_ENV] = role
        child_env[IMPL_ENV] = impl
        child_env[CLAIM_ENV] = str(claim)
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"
        child_env["PYTHONOPTIMIZE"] = "0"
        stdout_path = claim / f"{role}.stdout"
        stderr_path = claim / f"{role}.stderr"
        argv = [python, "-B", str(STAGE_PATH), "--device", device, "--out", str(out)]
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            proc = subprocess.Popen(
                argv,
                cwd=str(REPO_ROOT),
                env=child_env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
            code = int(proc.wait())
        exits.append(code)
        if code != expected:
            write_json_exclusive(
                claim / "controller_failure.json",
                {
                    "code": "unexpected_worker_exit",
                    "exits": exits,
                    "expected": expected,
                    "role": role,
                    "start_new_session": False,
                },
            )
            return 1
    report = {
        "claim": str(claim),
        "claim_name": CLAIM_DIRNAME,
        "device": device,
        "exits": exits,
        "expected_exits": list(EXPECTED_EXITS),
        "fault_exit_recorded": FAULT_EXIT,
        "fixture_sha256": FIXTURE_SHA256,
        "impl": impl,
        "model_call_budget": MODEL_CALL_BUDGET,
        "parameter_fingerprint": bound["fingerprint"],
        "pretrained_models_loaded": 0 if impl == ENV_IMPL_DUMMY else None,
        "process": controller["process"],
        "roles": list(ROLES),
        "schema": "native_query_prefix_recovery_controller_v1",
        "start_new_session": False,
        "status": "controller_observed_exits_0_75_0",
        "limitations": [
            "CPU controller tests are not native execution, freeze, or parent acceptance.",
            "Dummy workers do not load pretrained models or exercise the native call budget.",
            "Native probes, BF16 capture, and replay comparisons remain unrun here.",
        ],
    }
    if impl == ENV_IMPL_NATIVE:
        totals = {"donor_forwards": 0, "receiver_forwards": 0, "adapter_backwards": 0}
        for role in ROLES:
            name = "fault_ready.json" if role == "interrupted" else f"worker_{role}.json"
            receipt = json.loads((claim / name).read_text(encoding="ascii"))
            _require(type(receipt) is dict, "worker_receipt")
            body = cast(dict[str, Any], receipt)
            for key in totals:
                totals[key] += _int(body.get(key), code=key)
        _require(totals == MODEL_CALL_BUDGET, "model_call_budget")
        report["model_call_totals"] = totals
        report["pretrained_models_loaded"] = 3
    write_json_exclusive(report_path, report)
    _fsync_dir(out)
    return 0


def _seed_supported(*, device: str, torch: Any, np: Any, random: Any) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(SEED)
    torch.set_rng_state(cpu_generator.get_state())
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
        torch.cuda.manual_seed(SEED)


def _draw_sentinels(*, device: str, torch: Any, np: Any, random: Any) -> dict[str, float | None]:
    python_draw = float(random.random())
    numpy_draw = float(np.random.random())
    cpu_draw = float(torch.rand((), dtype=torch.float32, device="cpu").item())
    cuda_draw: float | None
    if device.startswith("cuda"):
        cuda_draw = float(torch.rand((), dtype=torch.float32, device=device).item())
    else:
        cuda_draw = None
    return {
        "numpy": numpy_draw,
        "python": python_draw,
        "torch_cpu": cpu_draw,
        "torch_cuda": cuda_draw,
    }


def _reconstruct_query_prefix_fingerprint(torch: Any) -> str:
    from src.transfer.latent_bridge import parameter_fingerprint
    from src.transfer.residue_prefix import QueryPrefix, query_prefix_parameter_count

    predicted = query_prefix_parameter_count(
        input_width=INPUT_WIDTH,
        internal_width=INTERNAL_WIDTH,
        output_width=OUTPUT_WIDTH,
        n_queries=PREFIX_QUERIES,
    )
    _require(predicted == MAX_TRAINABLE, "query_prefix_formula")
    if torch.get_default_dtype() != torch.float32:
        torch.set_default_dtype(torch.float32)
    with torch.device("cpu"):
        adapter = QueryPrefix(
            input_width=INPUT_WIDTH,
            internal_width=INTERNAL_WIDTH,
            output_width=OUTPUT_WIDTH,
            n_queries=PREFIX_QUERIES,
            n_heads=N_HEADS,
            max_trainable_parameters=MAX_TRAINABLE,
        )
    actual = int(adapter.trainable_parameter_count())
    _require(actual == MAX_TRAINABLE, "query_prefix_count")
    fingerprint = parameter_fingerprint(adapter)
    _require(fingerprint == EXPECTED_FINGERPRINT, "parameter_fingerprint")
    del adapter
    return fingerprint


def _verify_in_flight_gradient(adapter: Any, torch: Any) -> float:
    total = 0.0
    nonzero = False
    for parameter in adapter.parameters():
        if not bool(parameter.requires_grad):
            continue
        grad = parameter.grad
        if grad is None:
            _fail("missing_in_flight_grad")
        if not bool(torch.isfinite(grad).all()):
            _fail("nonfinite_in_flight_grad")
        abs_sum = float(grad.detach().abs().sum().item())
        total += abs_sum
        nonzero = nonzero or abs_sum > 0.0
    _require(nonzero, "zero_in_flight_grad")
    return total


def run_dummy_worker(*, role: str, device: str, claim: Path) -> int:
    import random

    import numpy as np
    import torch
    from torch import nn

    from src.transfer.latent_bridge import IGNORE_INDEX
    from src.transfer.query_prefix_train_state import PreparedMicrobatch, QueryPrefixTrainState

    _require(role in ROLES, "unknown_role")
    _require(device == "cpu" or device.startswith("cpu"), "dummy_requires_cpu_device")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "dummy_cuda_not_hidden")
    payload = load_fixture()
    bound = bind_fixture(payload)
    _seed_supported(device="cpu", torch=torch, np=np, random=random)
    fingerprint = _reconstruct_query_prefix_fingerprint(torch)
    _require(fingerprint == bound["fingerprint"], "parameter_fingerprint")
    _seed_supported(device="cpu", torch=torch, np=np, random=random)

    class TinyAdapter(nn.Module):
        def __init__(self) -> None:
            nn.Module.__init__(self)  # pyright: ignore[reportUnknownMemberType]
            self.proj = nn.Linear(4, 8)

    adapter = TinyAdapter()
    _seed_supported(device="cpu", torch=torch, np=np, random=random)
    features = torch.zeros((DATASET_SIZE, 2, 4), dtype=torch.float32)
    for row in range(DATASET_SIZE):
        features[row, 0, 0] = float(row + 1)
        features[row, 1, 1] = float(row + 2)
    labels = torch.full((DATASET_SIZE, 2), IGNORE_INDEX, dtype=torch.long)
    labels[:, 1] = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    identity = train_identity(fingerprint=fingerprint)
    recipe = recipe_kwargs()
    windows: list[list[int]] = []
    sentinels: list[dict[str, float | None]] = []

    def cancelled() -> bool:
        return False

    def make_state() -> QueryPrefixTrainState:
        return QueryPrefixTrainState(
            adapter=adapter,
            identity=identity,
            dataset_size=cast(int, recipe["dataset_size"]),
            microbatch_size=cast(int, recipe["microbatch_size"]),
            accumulation_steps=cast(int, recipe["accumulation_steps"]),
            sampler_seed=cast(int, recipe["sampler_seed"]),
            total_update_budget=cast(int, recipe["total_update_budget"]),
            lr=cast(float, recipe["lr"]),
            betas=cast(tuple[float, float], recipe["betas"]),
            eps=cast(float, recipe["eps"]),
            weight_decay=cast(float, recipe["weight_decay"]),
            grad_clip=cast(float, recipe["grad_clip"]),
        )

    def prepare(indices: tuple[int, ...]) -> list[PreparedMicrobatch]:
        sentinels.append(_draw_sentinels(device="cpu", torch=torch, np=np, random=random))
        rows = [int(index) for index in indices]
        windows.append(rows)
        update_index = len(windows)
        batches: list[PreparedMicrobatch] = []
        for micro_i, example_index in enumerate(rows):
            x = features[example_index : example_index + 1]
            y = labels[example_index : example_index + 1].clone()

            def forward(
                x: Any = x,
                micro_i: int = micro_i,
                update_index: int = update_index,
            ) -> Any:
                if role == "interrupted" and update_index == 3 and micro_i == 1:
                    grad_abs_sum = _verify_in_flight_gradient(adapter, torch)
                    marker = {
                        "adapter_backwards": 7,
                        "donor_forwards": 0,
                        "grad_abs_sum": grad_abs_sum,
                        "impl": ENV_IMPL_DUMMY,
                        "in_flight_grad_nonzero_finite": True,
                        "pretrained_models_loaded": 0,
                        "process": process_identity(),
                        "receiver_forwards": 0,
                        "role": role,
                        "schema": "native_query_recovery_fault_ready_v1",
                        "sentinels": sentinels,
                        "windows": windows,
                    }
                    write_json_exclusive(claim / "fault_ready.json", marker)
                    write_json_exclusive(claim / "worker_interrupted.json", marker)
                    _fsync_dir(claim)
                    os._exit(FAULT_EXIT)
                return adapter.proj(x)

            batches.append(PreparedMicrobatch(labels=y, forward=forward))
        return batches

    if role == "resume":
        seal_path = claim / "update2.sealhash"
        _require_real_file(seal_path, code="missing_sealhash")
        seal = seal_path.read_text(encoding="ascii").strip()
        state = QueryPrefixTrainState.restore(
            adapter=adapter,
            identity=identity,
            dataset_size=cast(int, recipe["dataset_size"]),
            microbatch_size=cast(int, recipe["microbatch_size"]),
            accumulation_steps=cast(int, recipe["accumulation_steps"]),
            sampler_seed=cast(int, recipe["sampler_seed"]),
            total_update_budget=cast(int, recipe["total_update_budget"]),
            lr=cast(float, recipe["lr"]),
            betas=cast(tuple[float, float], recipe["betas"]),
            eps=cast(float, recipe["eps"]),
            weight_decay=cast(float, recipe["weight_decay"]),
            grad_clip=cast(float, recipe["grad_clip"]),
            checkpoint_dir=claim / "update2",
            expected_seal_sha256=seal,
        )
        state.update(prepare, cancelled=cancelled)
        state.update(prepare, cancelled=cancelled)
        _require(windows == [list(WINDOW_ROWS[2]), list(WINDOW_ROWS[3])], "resume_windows")
        receipt = {
            "adapter_backwards": 6,
            "donor_forwards": 0,
            "impl": ENV_IMPL_DUMMY,
            "pretrained_models_loaded": 0,
            "process": process_identity(),
            "receiver_forwards": 0,
            "role": role,
            "sentinels": sentinels,
            "windows": windows,
        }
        write_json_exclusive(claim / "worker_resume.json", receipt)
        _fsync_dir(claim)
        return 0

    state = make_state()
    completed = 4 if role == "continuous" else 2
    for _ in range(completed):
        state.update(prepare, cancelled=cancelled)
    if role == "continuous":
        _require(windows == [list(row) for row in WINDOW_ROWS], "continuous_windows")
        receipt = {
            "adapter_backwards": 12,
            "donor_forwards": 0,
            "impl": ENV_IMPL_DUMMY,
            "pretrained_models_loaded": 0,
            "process": process_identity(),
            "receiver_forwards": 0,
            "role": role,
            "sentinels": sentinels,
            "windows": windows,
        }
        write_json_exclusive(claim / "worker_continuous.json", receipt)
        _fsync_dir(claim)
        return 0

    _require(windows == [list(WINDOW_ROWS[0]), list(WINDOW_ROWS[1])], "interrupted_windows")
    _require(int(getattr(state, "_epoch")) == UPDATE2["epoch"], "update2_epoch")
    _require(int(getattr(state, "_cursor")) == UPDATE2["cursor"], "update2_cursor")
    _require(int(getattr(state, "_consumed_rows")) == UPDATE2["consumed"], "update2_consumed")
    seal = state.publish(claim / "update2")
    write_text_exclusive(claim / "update2.sealhash", seal + "\n")
    _fsync_dir(claim)
    state.update(prepare, cancelled=cancelled)
    _fail("interrupted_did_not_exit")


def _allclose(left: Any, right: Any, *, torch: Any) -> dict[str, object]:
    delta = (left.detach().float() - right.detach().float()).abs()
    max_abs = float(delta.max().item()) if int(delta.numel()) > 0 else 0.0
    equal = bool(torch.equal(left, right))
    close = bool(torch.allclose(left, right, atol=1e-5, rtol=1e-5))
    _require(close, "native_allclose")
    return {"close": close, "exact": equal, "max_abs": max_abs}


def _select_row(
    batch: Any,
    index: int,
    records: Sequence[Any],
    provenance: Any,
    residue_batch_cls: Any,
) -> Any:
    return residue_batch_cls(
        hidden=batch.hidden[index : index + 1],
        input_ids=batch.input_ids[index : index + 1],
        attention_mask=batch.attention_mask[index : index + 1],
        content_mask=batch.content_mask[index : index + 1],
        records=(records[index],),
        provenance=provenance,
    )


def run_native_worker(*, role: str, device: str, claim: Path) -> int:
    _require(role in ROLES, "unknown_role")
    _require(device.startswith("cuda"), "native_requires_cuda")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") != "", "native_hidden_cuda")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import random

    import numpy as np
    import torch

    from src.transfer.latent_bridge import (
        BLOCK_INDEX_SEMANTICS,
        DonorHandle,
        IGNORE_INDEX,
        freeze_module,
        inject_soft_prefix,
        parameter_fingerprint,
    )
    from src.transfer.query_prefix_train_state import PreparedMicrobatch, QueryPrefixTrainState
    from src.transfer.residue_extraction import ResidueSequence, extract_residue_batch
    from src.transfer.residue_prefix import (
        IN_MEMORY_SCHEMA,
        QueryPrefix,
        ResidueBatch,
        ResidueProvenance,
        ResidueRecord,
        mean_from_residues,
        query_prefix_parameter_count,
    )

    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True)
    except (RuntimeError, ValueError) as exc:
        raise RecoveryStageError("determinism_unsupported") from exc

    payload = load_fixture()
    bound = bind_fixture(payload)
    assets_obj = bound["assets"]
    _require(type(assets_obj) is dict, "assets")
    assets = cast(dict[str, Any], assets_obj)
    donor_override = os.environ.get(DONOR_ENV)
    receiver_override = os.environ.get(RECEIVER_ENV)
    donor_path = Path(donor_override) if donor_override else Path(_text(assets.get("donor_path"), code="donor_path"))
    receiver_path = Path(receiver_override) if receiver_override else Path(_text(assets.get("receiver_path"), code="receiver_path"))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.transfer.arms import config_shape

    torch_dtype = torch.bfloat16
    tokenizer_loader: Any = AutoTokenizer
    model_loader: Any = AutoModelForCausalLM
    donor_tokenizer: Any = tokenizer_loader.from_pretrained(
        str(donor_path), local_files_only=True, trust_remote_code=True
    )
    if donor_tokenizer.pad_token is None:
        donor_tokenizer.pad_token = donor_tokenizer.eos_token
    _require(int(donor_tokenizer.pad_token_id) == DONOR_PAD_ID, "donor_pad")
    donor_model: Any = model_loader.from_pretrained(
        str(donor_path),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map={"": device},
    )
    freeze_module(donor_model)
    n_layer, d_model = config_shape(donor_model.config)
    _require(int(n_layer) == 27, "donor_n_layer")
    _require(int(d_model) == INPUT_WIDTH, "donor_width")
    donor = DonorHandle(
        name="progen2-medium",
        model=donor_model,
        tokenizer=donor_tokenizer,
        n_layer=int(n_layer),
        d_model=int(d_model),
        kind="pretrained",
        checkpoint_digest={"path": str(donor_path)},
    )
    receiver_tokenizer: Any = tokenizer_loader.from_pretrained(
        str(receiver_path), local_files_only=True, trust_remote_code=True
    )
    if receiver_tokenizer.pad_token is None:
        receiver_tokenizer.pad_token = receiver_tokenizer.eos_token
    _require(int(receiver_tokenizer.pad_token_id) == RECEIVER_PAD_ID, "receiver_pad")
    _require(int(receiver_tokenizer.eos_token_id) == RECEIVER_EOS_ID, "receiver_eos")
    receiver_model: Any = model_loader.from_pretrained(
        str(receiver_path),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map={"": device},
    )
    freeze_module(receiver_model)
    embed: Any = receiver_model.get_input_embeddings()
    _seed_supported(device=device, torch=torch, np=np, random=random)
    predicted = query_prefix_parameter_count(
        input_width=INPUT_WIDTH,
        internal_width=INTERNAL_WIDTH,
        output_width=OUTPUT_WIDTH,
        n_queries=PREFIX_QUERIES,
    )
    _require(predicted == MAX_TRAINABLE, "query_prefix_formula")
    if torch.get_default_dtype() != torch.float32:
        torch.set_default_dtype(torch.float32)
    with torch.device("cpu"):
        adapter = QueryPrefix(
            input_width=INPUT_WIDTH,
            internal_width=INTERNAL_WIDTH,
            output_width=OUTPUT_WIDTH,
            n_queries=PREFIX_QUERIES,
            n_heads=N_HEADS,
            max_trainable_parameters=MAX_TRAINABLE,
        )
    fingerprint = parameter_fingerprint(adapter)
    _require(fingerprint == EXPECTED_FINGERPRINT, "parameter_fingerprint")
    adapter.to(device=torch.device(device))
    for parameter in adapter.parameters():
        _require(parameter.dtype == torch.float32, "adapter_dtype")
        _require(str(parameter.device) == str(torch.device(device)), "adapter_device")

    sequences_map = cast(dict[str, Any], bound["sequences"])
    residue_sequences: list[ResidueSequence] = []
    records: list[ResidueRecord] = []
    for accession in ("engineering-A", "engineering-B"):
        row = cast(dict[str, Any], sequences_map[accession])
        sequence = _text(row.get("sequence"), code="sequence")
        token_ids = tuple(_int_list(row.get("native_donor_token_ids"), code="native_donor_token_ids"))
        residue_sequences.append(ResidueSequence(accession=accession, sequence=sequence))
        records.append(
            ResidueRecord(
                accession=accession,
                sequence_sha256=_text(row.get("sequence_sha256"), code="sequence_sha256"),
                residue_count=_int(row.get("residue_count"), code="residue_count"),
                native_token_ids=token_ids,
            )
        )
    donor_used = cast(dict[str, Any], assets["donor_used_nonweight_after"])
    inventory_map = {
        name: _text(cast(dict[str, Any], body).get("sha256"), code="donor_file_sha")
        for name, body in donor_used.items()
    }
    inventory = _sha256_bytes(_canonical_bytes(inventory_map))
    code_commit = _sha256_bytes(f"{FIXTURE_SHA256}:native_query_prefix_recovery".encode("ascii"))[:40]
    marker_id = int(records[0].native_token_ids[0])
    provenance = ResidueProvenance(
        schema=IN_MEMORY_SCHEMA,
        donor_inventory_sha256=inventory,
        code_commit=code_commit,
        block_index=BLOCK_INDEX,
        block_index_semantics=BLOCK_INDEX_SEMANTICS,
        capture_dtype="bfloat16",
        hidden_width=INPUT_WIDTH,
        marker_id=marker_id,
        pad_id=DONOR_PAD_ID,
    )
    limits = ROLE_BUDGET[role]
    counts = {"donor_forwards": 0, "receiver_forwards": 0, "adapter_backwards": 0}

    def check() -> None:
        return None

    def capture() -> ResidueBatch:
        batch = extract_residue_batch(
            donor,
            residue_sequences,
            expected_records=records,
            expected_provenance=provenance,
            device=device,
            max_residues=MAX_RESIDUES,
            max_tokens=MAX_TOKENS,
            max_tensor_bytes=MAX_TENSOR_BYTES,
            check=check,
        )
        counts["donor_forwards"] += 1
        _require(counts["donor_forwards"] <= int(limits["donor_forwards"]), "donor_budget")
        return batch

    first = capture()
    if role == "continuous":
        repeat = capture()
        mean_a = mean_from_residues(
            first,
            expected_records=records,
            expected_provenance=provenance,
            max_tensor_bytes=MAX_TENSOR_BYTES,
        )
        mean_b = mean_from_residues(
            repeat,
            expected_records=records,
            expected_provenance=provenance,
            max_tensor_bytes=MAX_TENSOR_BYTES,
        )
        query_a = adapter(
            first,
            expected_records=records,
            expected_provenance=provenance,
            max_tensor_bytes=MAX_TENSOR_BYTES,
        )
        query_b = adapter(
            repeat,
            expected_records=records,
            expected_provenance=provenance,
            max_tensor_bytes=MAX_TENSOR_BYTES,
        )
        _allclose(mean_a, mean_b, torch=torch)
        _allclose(query_a, query_b, torch=torch)
        example = cast(dict[str, Any], bound["examples"][0])
        prompt_ids = torch.tensor(
            [_int_list(example.get("prompt_ids"), code="prompt_ids")],
            dtype=torch.long,
            device=device,
        )
        answer_ids = torch.tensor(
            [_int_list(example.get("answer_ids"), code="answer_ids")],
            dtype=torch.long,
            device=device,
        )
        ids = torch.cat([prompt_ids, answer_ids], dim=1)
        attention = torch.ones_like(ids)
        position_ids = (attention.cumsum(dim=1) - 1).clamp(min=0)
        with torch.no_grad():
            logits_ids = receiver_model(
                input_ids=ids,
                attention_mask=attention,
                position_ids=position_ids,
            ).logits
            counts["receiver_forwards"] += 1
            logits_emb = receiver_model(
                inputs_embeds=embed(ids),
                attention_mask=attention,
                position_ids=position_ids,
            ).logits
            counts["receiver_forwards"] += 1
        _allclose(logits_ids, logits_emb, torch=torch)

        def handoff_backward(batch: ResidueBatch) -> dict[str, Any]:
            adapter.zero_grad(set_to_none=True)
            row = _select_row(batch, 0, records, provenance, ResidueBatch)
            soft = adapter(
                row,
                expected_records=(records[0],),
                expected_provenance=provenance,
                max_tensor_bytes=MAX_TENSOR_BYTES,
            )
            packed = inject_soft_prefix(
                soft=soft,
                prompt_ids=prompt_ids,
                answer_ids=answer_ids,
                embed=embed,
                pad_id=RECEIVER_PAD_ID,
            )
            logits = receiver_model(
                inputs_embeds=packed["inputs_embeds"],
                attention_mask=packed["attention_mask"],
                position_ids=packed["position_ids"],
            ).logits
            counts["receiver_forwards"] += 1
            support = packed["labels"][:, 1:] != IGNORE_INDEX
            nll = torch.nn.functional.cross_entropy(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                logits[:, :-1].float().contiguous().reshape(-1, int(logits.shape[-1])),
                packed["labels"][:, 1:].contiguous().reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            (nll / float(int(support.sum().item()))).backward()  # pyright: ignore[reportUnknownMemberType]
            counts["adapter_backwards"] += 1
            snapshot: dict[str, Any] = {}
            for name, parameter in adapter.named_parameters():
                grad = parameter.grad
                if grad is None:
                    _fail("probe_grad")
                snapshot[name] = grad.detach().float().cpu().clone()
            adapter.zero_grad(set_to_none=True)
            return snapshot

        grads_a = handoff_backward(first)
        grads_b = handoff_backward(repeat)
        for name in grads_a:
            _allclose(grads_a[name], grads_b[name], torch=torch)
        adapter.zero_grad(set_to_none=True)

    _seed_supported(device=device, torch=torch, np=np, random=random)
    captured = first
    examples = cast(list[dict[str, Any]], bound["examples"])
    identity = train_identity(fingerprint=fingerprint)
    recipe = recipe_kwargs()
    windows: list[list[int]] = []
    sentinels: list[dict[str, float | None]] = []
    accession_row = {"engineering-A": 0, "engineering-B": 1}

    def cancelled() -> bool:
        return False

    def query_receiver(example_index: int) -> tuple[Any, Any]:
        example = examples[example_index]
        accession = _text(example.get("accession"), code="accession")
        row_index = accession_row[accession]
        row = _select_row(captured, row_index, records, provenance, ResidueBatch)
        soft = adapter(
            row,
            expected_records=(records[row_index],),
            expected_provenance=provenance,
            max_tensor_bytes=MAX_TENSOR_BYTES,
        )
        prompt_ids = torch.tensor(
            [_int_list(example.get("prompt_ids"), code="prompt_ids")],
            dtype=torch.long,
            device=device,
        )
        answer_ids = torch.tensor(
            [_int_list(example.get("answer_ids"), code="answer_ids")],
            dtype=torch.long,
            device=device,
        )
        packed = inject_soft_prefix(
            soft=soft,
            prompt_ids=prompt_ids,
            answer_ids=answer_ids,
            embed=embed,
            pad_id=RECEIVER_PAD_ID,
        )
        logits = receiver_model(
            inputs_embeds=packed["inputs_embeds"],
            attention_mask=packed["attention_mask"],
            position_ids=packed["position_ids"],
        ).logits
        counts["receiver_forwards"] += 1
        _require(counts["receiver_forwards"] <= int(limits["receiver_forwards"]), "receiver_budget")
        return logits, packed["labels"]

    def labels_for(example_index: int) -> Any:
        example = examples[example_index]
        prompt = _int_list(example.get("prompt_ids"), code="prompt_ids")
        answer = _int_list(example.get("answer_ids"), code="answer_ids")
        values = [IGNORE_INDEX] * (PREFIX_QUERIES + len(prompt)) + answer
        return torch.tensor([values], dtype=torch.long, device=device)

    def prepare(indices: tuple[int, ...]) -> list[PreparedMicrobatch]:
        sentinels.append(_draw_sentinels(device=device, torch=torch, np=np, random=random))
        rows = [int(index) for index in indices]
        windows.append(rows)
        update_index = len(windows)
        batches: list[PreparedMicrobatch] = []
        for micro_i, example_index in enumerate(rows):
            packed_labels = labels_for(example_index)

            def forward(
                example_index: int = example_index,
                micro_i: int = micro_i,
                update_index: int = update_index,
            ) -> Any:
                if role == "interrupted" and update_index == 3 and micro_i == 1:
                    grad_abs_sum = _verify_in_flight_gradient(adapter, torch)
                    counts["adapter_backwards"] += 1
                    _require(counts == {key: int(limits[key]) for key in counts}, "interrupted_budget")
                    marker = {
                        "adapter_backwards": counts["adapter_backwards"],
                        "donor_forwards": counts["donor_forwards"],
                        "grad_abs_sum": grad_abs_sum,
                        "impl": ENV_IMPL_NATIVE,
                        "in_flight_grad_nonzero_finite": True,
                        "pretrained_models_loaded": 2,
                        "process": process_identity(),
                        "receiver_forwards": counts["receiver_forwards"],
                        "role": role,
                        "schema": "native_query_recovery_fault_ready_v1",
                        "sentinels": sentinels,
                        "windows": windows,
                    }
                    write_json_exclusive(claim / "fault_ready.json", marker)
                    write_json_exclusive(claim / "worker_interrupted.json", marker)
                    _fsync_dir(claim)
                    os._exit(FAULT_EXIT)
                logits, packed = query_receiver(example_index)
                _require(packed.dtype == torch.long, "packed_labels")
                return logits

            batches.append(PreparedMicrobatch(labels=packed_labels, forward=forward))
        return batches

    def finish_receipt() -> dict[str, object]:
        _require(counts["donor_forwards"] == int(limits["donor_forwards"]), "donor_budget")
        _require(counts["receiver_forwards"] == int(limits["receiver_forwards"]), "receiver_budget")
        _require(counts["adapter_backwards"] == int(limits["adapter_backwards"]), "backward_budget")
        return {
            "adapter_backwards": counts["adapter_backwards"],
            "donor_forwards": counts["donor_forwards"],
            "impl": ENV_IMPL_NATIVE,
            "pretrained_models_loaded": 2,
            "process": process_identity(),
            "receiver_forwards": counts["receiver_forwards"],
            "role": role,
            "sentinels": sentinels,
            "windows": windows,
        }

    if role == "resume":
        seal = (claim / "update2.sealhash").read_text(encoding="ascii").strip()
        state = QueryPrefixTrainState.restore(
            adapter=adapter,
            identity=identity,
            dataset_size=cast(int, recipe["dataset_size"]),
            microbatch_size=cast(int, recipe["microbatch_size"]),
            accumulation_steps=cast(int, recipe["accumulation_steps"]),
            sampler_seed=cast(int, recipe["sampler_seed"]),
            total_update_budget=cast(int, recipe["total_update_budget"]),
            lr=cast(float, recipe["lr"]),
            betas=cast(tuple[float, float], recipe["betas"]),
            eps=cast(float, recipe["eps"]),
            weight_decay=cast(float, recipe["weight_decay"]),
            grad_clip=cast(float, recipe["grad_clip"]),
            checkpoint_dir=claim / "update2",
            expected_seal_sha256=seal,
        )
        metrics = state.update(prepare, cancelled=cancelled)
        counts["adapter_backwards"] += ACCUMULATION_STEPS
        metrics = state.update(prepare, cancelled=cancelled)
        counts["adapter_backwards"] += ACCUMULATION_STEPS
        del metrics
        write_json_exclusive(claim / "worker_resume.json", finish_receipt())
        _fsync_dir(claim)
        return 0

    state = QueryPrefixTrainState(
        adapter=adapter,
        identity=identity,
        dataset_size=cast(int, recipe["dataset_size"]),
        microbatch_size=cast(int, recipe["microbatch_size"]),
        accumulation_steps=cast(int, recipe["accumulation_steps"]),
        sampler_seed=cast(int, recipe["sampler_seed"]),
        total_update_budget=cast(int, recipe["total_update_budget"]),
        lr=cast(float, recipe["lr"]),
        betas=cast(tuple[float, float], recipe["betas"]),
        eps=cast(float, recipe["eps"]),
        weight_decay=cast(float, recipe["weight_decay"]),
        grad_clip=cast(float, recipe["grad_clip"]),
    )
    completed = 4 if role == "continuous" else 2
    for _ in range(completed):
        state.update(prepare, cancelled=cancelled)
        counts["adapter_backwards"] += ACCUMULATION_STEPS
    if role == "continuous":
        write_json_exclusive(claim / "worker_continuous.json", finish_receipt())
        _fsync_dir(claim)
        return 0
    seal = state.publish(claim / "update2")
    write_text_exclusive(claim / "update2.sealhash", seal + "\n")
    _fsync_dir(claim)
    state.update(prepare, cancelled=cancelled)
    _fail("interrupted_did_not_exit")


def run_worker(role: str, device: str, out: Path) -> int:
    claim_value = os.environ.get(CLAIM_ENV)
    _require(bool(claim_value), "missing_claim_env")
    claim = Path(cast(str, claim_value))
    _require_real_dir(claim, code="missing_claim")
    _require_real_dir(out, code="missing_out")
    impl = impl_name()
    if impl == ENV_IMPL_DUMMY:
        return run_dummy_worker(role=role, device=device, claim=claim)
    return run_native_worker(role=role, device=device, claim=claim)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    role = os.environ.get(ROLE_ENV)
    try:
        if role:
            return run_worker(role, str(args.device), Path(args.out))
        return run_controller(str(args.device), Path(args.out))
    except RecoveryStageError as error:
        sys.stderr.write(error.code + "\n")
        if error.code in {"occupied_claim", "occupied_report"}:
            return 2
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
