"""CPU controller tests for native query-prefix recovery. No pretrained loads."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping
import json
import os
import resource
import subprocess
import sys
import tempfile
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("PYTHONOPTIMIZE", "0")

REPO = Path(__file__).resolve().parents[1]
STAGE = REPO / "scripts/transfer/native_query_prefix_recovery.py"
CLAIM_DIRNAME = "native_query_recovery_cell"
FAULT_EXIT = 75
FIXTURE = (
    REPO
    / "logs/bridge_complementarity_20260912/native_query_fixture_parent_qualification_01"
    / "fixture.json"
)
FIXTURE_SHA256 = "9faa01d99d78d72761e802c28bbc7374e8b9b3cd8240586c487cb33cfdd0eeab"


def _limit_as() -> None:
    cap = 16 * 1024 ** 3
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_hard = cap if hard == resource.RLIM_INFINITY else min(cap, hard)
    new_soft = cap if soft == resource.RLIM_INFINITY else min(cap, soft)
    resource.setrlimit(resource.RLIMIT_AS, (min(new_soft, new_hard), new_hard))


def _env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["NATIVE_QUERY_RECOVERY_IMPL"] = "dummy"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONOPTIMIZE"] = "0"
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "1"
    if extra:
        env.update(extra)
    return env


def _run(
    argv: list[str],
    *,
    extra: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=str(REPO),
        env=_env(extra),
        text=True,
        capture_output=True,
        check=False,
    )


def _controller(out: Path) -> subprocess.CompletedProcess[str]:
    return _run(
        [sys.executable, "-B", str(STAGE), "--device", "cpu", "--out", str(out)],
    )


def _worker(role: str, out: Path, claim: Path) -> subprocess.CompletedProcess[str]:
    return _run(
        [sys.executable, "-B", str(STAGE), "--device", "cpu", "--out", str(out)],
        extra={
            "NATIVE_QUERY_RECOVERY_ROLE": role,
            "NATIVE_QUERY_RECOVERY_CLAIM": str(claim),
        },
    )


class NativeQueryRecoveryControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _limit_as()
        digest = __import__("hashlib").sha256(FIXTURE.read_bytes()).hexdigest()
        if digest != FIXTURE_SHA256:
            raise RuntimeError("fixture_sha256_mismatch")

    def test_controller_observes_sequential_exits_0_75_0(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            completed = _controller(out)
            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            report = json.loads((out / "report.json").read_text(encoding="ascii"))
            self.assertEqual(report["exits"], [0, FAULT_EXIT, 0])
            self.assertEqual(report["fault_exit_recorded"], FAULT_EXIT)
            self.assertEqual(report["impl"], "dummy")
            self.assertFalse(report["start_new_session"])
            self.assertEqual(report["pretrained_models_loaded"], 0)
            claim = out / CLAIM_DIRNAME
            controller = json.loads((claim / "controller.json").read_text(encoding="ascii"))
            continuous = json.loads((claim / "worker_continuous.json").read_text(encoding="ascii"))
            interrupted = json.loads((claim / "fault_ready.json").read_text(encoding="ascii"))
            resumed = json.loads((claim / "worker_resume.json").read_text(encoding="ascii"))
            controller_proc = controller["process"]
            for receipt in (continuous, interrupted, resumed):
                self.assertEqual(receipt["process"]["pgid"], controller_proc["pgid"])
                self.assertEqual(receipt["process"]["sid"], controller_proc["sid"])
                self.assertNotEqual(receipt["process"]["pid"], receipt["process"]["sid"])
                self.assertEqual(receipt["pretrained_models_loaded"], 0)
            self.assertTrue(interrupted["in_flight_grad_nonzero_finite"])
            self.assertGreater(float(interrupted["grad_abs_sum"]), 0.0)
            self.assertTrue((claim / "update2" / "seal.json").is_file())
            self.assertEqual(continuous["windows"], [[2, 0, 3], [1, 1, 2], [3, 0, 3], [2, 0, 1]])
            self.assertEqual(resumed["windows"], [[3, 0, 3], [2, 0, 1]])

    def test_occupied_claim_directory_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            (out / CLAIM_DIRNAME).mkdir()
            completed = _controller(out)
            self.assertEqual(completed.returncode, 2, msg=completed.stderr)
            self.assertIn("occupied_claim", completed.stderr)
            self.assertFalse((out / "report.json").exists())

    def test_dummy_worker_exit_75_after_in_flight_gradient(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            claim = out / CLAIM_DIRNAME
            claim.mkdir()
            completed = _worker("interrupted", out, claim)
            self.assertEqual(completed.returncode, FAULT_EXIT, msg=completed.stderr)
            marker = json.loads((claim / "fault_ready.json").read_text(encoding="ascii"))
            self.assertEqual(marker["schema"], "native_query_recovery_fault_ready_v1")
            self.assertTrue(marker["in_flight_grad_nonzero_finite"])
            self.assertGreater(float(marker["grad_abs_sum"]), 0.0)
            self.assertEqual(marker["pretrained_models_loaded"], 0)
            self.assertEqual(marker["impl"], "dummy")
            self.assertTrue((claim / "update2" / "payload.pt").is_file())
            self.assertTrue((claim / "update2" / "seal.json").is_file())
            self.assertNotEqual(marker["process"]["pid"], marker["process"]["sid"])


if __name__ == "__main__":
    unittest.main()
