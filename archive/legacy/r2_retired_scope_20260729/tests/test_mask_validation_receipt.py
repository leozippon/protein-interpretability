from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


R2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R2))

from src.revision.dictionary_gate import _validate_mask_receipt  # noqa: E402
from src.revision.io import sha256_file  # noqa: E402
from src.revision.mask_validation_receipt import (  # noqa: E402
    MASK_TESTS,
    MaskReceiptError,
    produce_mask_validation_receipt,
)


def source_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "r2"
    module = root / "src/revision/dictionary_controls.py"
    tests = root / "tests/test_dictionary_controls.py"
    module.parent.mkdir(parents=True)
    tests.parent.mkdir(parents=True)
    module.write_text("# exact training module\n", encoding="utf-8")
    tests.write_text("# exact mask tests\n", encoding="utf-8")
    return root, module, tests


def fake_pytest(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: dict[str, bool],
    *,
    returncode: int,
    error_test: str | None = None,
    mutate: Path | None = None,
) -> list[tuple[list[str], dict]]:
    calls: list[tuple[list[str], dict]] = []

    def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        junit = Path(
            next(
                part.split("=", 1)[1]
                for part in command
                if part.startswith("--junitxml=")
            )
        )
        suite = ET.Element("testsuite")
        for name, passed in outcomes.items():
            case = ET.SubElement(suite, "testcase", name=name)
            if name == error_test:
                ET.SubElement(case, "error")
            elif not passed:
                ET.SubElement(case, "failure")
        ET.ElementTree(suite).write(junit, encoding="utf-8", xml_declaration=True)
        if mutate is not None:
            mutate.write_text("# changed during pytest\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, returncode, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_pass_receipt_is_exact_and_accepted_by_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, module, test_file = source_tree(tmp_path)
    calls = fake_pytest(
        monkeypatch,
        {name: True for name in MASK_TESTS},
        returncode=0,
    )
    output = tmp_path / "mask_validation_receipt.json"
    path, payload = produce_mask_validation_receipt(
        r2_root=root,
        output_path=output,
        expected_module_sha256=sha256_file(module),
        python_executable="/exact/conda/python",
    )

    assert path == output.resolve()
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert set(payload) == {
        "schema_version",
        "status",
        "confirmatory",
        "pytest_exit_code",
        "production_scientific_eligibility",
        "module",
        "test_file",
        "tests",
    }
    assert payload["module"] == {
        "path": "r2/src/revision/dictionary_controls.py",
        "sha256": sha256_file(module),
    }
    assert payload["test_file"] == {
        "path": "r2/tests/test_dictionary_controls.py",
        "sha256": sha256_file(test_file),
    }
    passed, module_sha, evidence = _validate_mask_receipt(payload, receipt_path=path)
    assert passed is True
    assert module_sha == sha256_file(module)
    assert evidence["test_file_sha256"] == sha256_file(test_file)

    command, kwargs = calls[0]
    assert command[:3] == ["/exact/conda/python", "-m", "pytest"]
    assert command[-2:] == [f"{test_file}::{name}" for name in MASK_TESTS]
    assert kwargs["cwd"] == root.resolve()
    assert kwargs["check"] is False


def test_assertion_failure_emits_complete_negative_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, module, _ = source_tree(tmp_path)
    outcomes = {name: index == 0 for index, name in enumerate(MASK_TESTS)}
    fake_pytest(monkeypatch, outcomes, returncode=1)
    output = tmp_path / "negative.json"

    _, payload = produce_mask_validation_receipt(
        r2_root=root,
        output_path=output,
        expected_module_sha256=sha256_file(module),
    )

    assert payload["pytest_exit_code"] == 1
    assert payload["production_scientific_eligibility"] is False
    assert payload["tests"] == outcomes


@pytest.mark.parametrize("failure_mode", ["test_error", "source_changed"])
def test_incomplete_or_unstable_run_publishes_no_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    root, module, test_file = source_tree(tmp_path)
    kwargs = (
        {"error_test": MASK_TESTS[0]}
        if failure_mode == "test_error"
        else {"mutate": test_file}
    )
    fake_pytest(
        monkeypatch,
        {name: True for name in MASK_TESTS},
        returncode=1 if failure_mode == "test_error" else 0,
        **kwargs,
    )
    output = tmp_path / "invalid.json"

    with pytest.raises(MaskReceiptError):
        produce_mask_validation_receipt(
            r2_root=root,
            output_path=output,
            expected_module_sha256=sha256_file(module),
        )
    assert not output.exists()


def test_expected_module_hash_and_write_once_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, _ = source_tree(tmp_path)

    def unexpected(*args, **kwargs):
        raise AssertionError("pytest must not run")

    monkeypatch.setattr(subprocess, "run", unexpected)
    output = tmp_path / "receipt.json"
    with pytest.raises(MaskReceiptError, match="expected run SHA"):
        produce_mask_validation_receipt(
            r2_root=root,
            output_path=output,
            expected_module_sha256="0" * 64,
        )
    output.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        produce_mask_validation_receipt(
            r2_root=root,
            output_path=output,
            expected_module_sha256="0" * 64,
        )
