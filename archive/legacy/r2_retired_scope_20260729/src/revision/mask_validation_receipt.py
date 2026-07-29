"""Produce the hash-bound P0-2 mask-validation receipt."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .io import sha256_file, write_json


SCHEMA_VERSION = "r2_p0_2_mask_validation_receipt_v1"
MASK_TESTS = (
    "test_valid_token_cache_is_padding_invariant_and_all_valid_equivalent",
    "test_windowed_metrics_are_padding_invariant_end_to_end",
)


class MaskReceiptError(ValueError):
    """The exact mask tests cannot support a complete receipt."""


def _require_digest(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MaskReceiptError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_sources(r2_root: Path) -> tuple[Path, Path]:
    root = Path(r2_root).resolve()
    module = root / "src/revision/dictionary_controls.py"
    test_file = root / "tests/test_dictionary_controls.py"
    for path, label in ((module, "dictionary module"), (test_file, "mask test file")):
        if not path.is_file():
            raise MaskReceiptError(f"canonical {label} is missing: {path}")
    return module, test_file


def _relative_to_receipt(path: Path, receipt: Path) -> str:
    return Path(os.path.relpath(path, start=receipt.parent)).as_posix()


def _parse_junit(path: Path) -> dict[str, bool]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise MaskReceiptError(f"cannot read pytest JUnit report: {error}") from error

    observed: dict[str, bool] = {}
    for case in root.iter("testcase"):
        name = case.get("name")
        if name not in MASK_TESTS or name in observed:
            raise MaskReceiptError(f"unexpected pytest test case: {name!r}")
        if case.find("error") is not None or case.find("skipped") is not None:
            raise MaskReceiptError(f"mask test did not complete normally: {name}")
        observed[name] = case.find("failure") is None
    if set(observed) != set(MASK_TESTS):
        raise MaskReceiptError(
            f"pytest did not report the exact mask tests: {sorted(observed)}"
        )
    return observed


def _run_exact_tests(
    *, r2_root: Path, test_file: Path, python_executable: str
) -> tuple[int, dict[str, bool]]:
    with tempfile.TemporaryDirectory(prefix="r2-mask-validation-") as temporary:
        temporary_path = Path(temporary)
        junit = temporary_path / "pytest.xml"
        command = [
            python_executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--basetemp={temporary_path / 'pytest'}",
            f"--junitxml={junit}",
            *(f"{test_file}::{name}" for name in MASK_TESTS),
        ]
        completed = subprocess.run(
            command,
            cwd=r2_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode not in (0, 1):
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            raise MaskReceiptError(
                f"pytest infrastructure failed with exit code {completed.returncode}: "
                f"{detail}"
            )
        outcomes = _parse_junit(junit)
        expected_exit = 0 if all(outcomes.values()) else 1
        if completed.returncode != expected_exit:
            raise MaskReceiptError(
                "pytest exit code disagrees with exact test outcomes"
            )
        return completed.returncode, outcomes


def produce_mask_validation_receipt(
    *,
    r2_root: Path,
    output_path: Path,
    expected_module_sha256: str,
    python_executable: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run the two frozen tests and publish a write-once receipt.

    A normal assertion failure produces a complete negative scientific receipt.
    Collection, execution, provenance, or serialization failures produce no receipt.
    """

    expected_module_sha256 = _require_digest(
        expected_module_sha256, "expected module SHA-256"
    )
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite mask receipt: {output}")
    module, test_file = _canonical_sources(r2_root)
    before = {
        "module": sha256_file(module),
        "test_file": sha256_file(test_file),
    }
    if before["module"] != expected_module_sha256:
        raise MaskReceiptError(
            "dictionary module SHA-256 differs from the expected run SHA"
        )

    exit_code, outcomes = _run_exact_tests(
        r2_root=Path(r2_root).resolve(),
        test_file=test_file,
        python_executable=python_executable or sys.executable,
    )
    after = {
        "module": sha256_file(module),
        "test_file": sha256_file(test_file),
    }
    if after != before:
        raise MaskReceiptError("tested module or test file changed during pytest")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite mask receipt: {output}")

    passed = all(outcomes.values())
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "confirmatory": True,
        "pytest_exit_code": exit_code,
        "production_scientific_eligibility": passed,
        "module": {
            "path": _relative_to_receipt(module, output),
            "sha256": before["module"],
        },
        "test_file": {
            "path": _relative_to_receipt(test_file, output),
            "sha256": before["test_file"],
        },
        "tests": outcomes,
    }
    write_json(output, payload)
    return output, payload
