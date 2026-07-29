from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/51_prepare_revision_manifests.py"
SPEC = importlib.util.spec_from_file_location("revision_manifests", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _records() -> list[dict]:
    real = [
        {
            "id": f"real-{index:03d}",
            "source": "real_lysozyme",
            "sequence": "M" + "A" * 20 + "C" * (index % 3),
            "meta": {"index": index},
        }
        for index in range(100)
    ]
    random = [
        {
            "id": f"random-{index:03d}",
            "source": "random_uniref50",
            "sequence": "M" + "G" * 20 + "T" * (index % 3),
            "meta": {"index": index},
        }
        for index in range(100)
    ]
    return [*real, *random]


def _bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode()


def test_exact_historical_file_upgrades_only_the_ordering_status(tmp_path: Path) -> None:
    source_path = tmp_path / "retained.json"
    source_bytes = _bytes({"n_records": 200, "records": _records()})
    source_path.write_bytes(source_bytes)

    reconstructed = MODULE.build_cohort(source_path, source_bytes)
    assert reconstructed["historical_ordering_status"] == "reconstructed_unverified"
    assert reconstructed["historical_artifact"] is None

    rows = _records()
    interleaved = [item for pair in zip(rows[:100], rows[100:], strict=True) for item in pair]
    historical_path = tmp_path / "calibration_lysozyme_balanced200_20260511.json"
    historical_bytes = _bytes(
        {
            "records": interleaved,
            "source": (
                "Research2/results/ec_metrics/calibration_lysozyme_20260507/"
                "calibration_sequences.json"
            ),
            "construction": "interleaved real_lysozyme and random_uniref50 controls",
        }
    )
    historical_path.write_bytes(historical_bytes)
    digest = MODULE.sha256_bytes(historical_bytes)
    verified = MODULE.build_cohort(
        source_path,
        source_bytes,
        historical_path=historical_path,
        historical_bytes=historical_bytes,
        historical_sha256=digest,
    )
    assert verified["historical_ordering_status"] == "historical_exact_file_verified"
    assert verified["ordered_cohort_sha256"] == reconstructed["ordered_cohort_sha256"]
    assert verified["historical_artifact"]["sha256"] == digest


def test_historical_hash_or_record_order_mismatch_fails(tmp_path: Path) -> None:
    source_path = tmp_path / "retained.json"
    source_bytes = _bytes({"n_records": 200, "records": _records()})
    source_path.write_bytes(source_bytes)
    rows = _records()
    interleaved = [item for pair in zip(rows[:100], rows[100:], strict=True) for item in pair]
    historical_path = tmp_path / "historical.json"
    payload = {
        "records": interleaved,
        "source": (
            "Research2/results/ec_metrics/calibration_lysozyme_20260507/"
            "calibration_sequences.json"
        ),
        "construction": "interleaved real_lysozyme and random_uniref50 controls",
    }
    historical_bytes = _bytes(payload)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        MODULE.build_cohort(
            source_path,
            source_bytes,
            historical_path=historical_path,
            historical_bytes=historical_bytes,
            historical_sha256="0" * 64,
        )

    payload["records"] = [interleaved[1], interleaved[0], *interleaved[2:]]
    reordered = _bytes(payload)
    with pytest.raises(ValueError, match="records differ"):
        MODULE.build_cohort(
            source_path,
            source_bytes,
            historical_path=historical_path,
            historical_bytes=reordered,
            historical_sha256=MODULE.sha256_bytes(reordered),
        )
