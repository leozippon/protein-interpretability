from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest


R2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R2))

from src.training.clt_trainer import _load_sequence_manifest


def load_split_module():
    path = R2 / "scripts/56_prepare_dictionary_splits.py"
    spec = importlib.util.spec_from_file_location("prepare_dictionary_splits", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SPLITS = load_split_module()


def args_for(fasta: Path) -> argparse.Namespace:
    return argparse.Namespace(
        fasta=fasta,
        source="fixture",
        family_mode="record_id",
        sequence_format="plain",
        seed=17,
        train_count=4,
        validation_count=2,
        test_count=2,
        min_seq_len=1,
        max_seq_len=32,
    )


def test_split_selection_is_deterministic_disjoint_and_hash_verified(tmp_path):
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(
        "".join(f">id{i} family_{i % 2}\nM{'A' * i}G\n" for i in range(1, 10))
    )
    first, metadata = SPLITS.select_records(args_for(fasta))
    second, _ = SPLITS.select_records(args_for(fasta))

    assert first == second
    assert metadata["selection_method"] == "first_n_unique_eligible_then_seeded_shuffle"
    counts = {"train": 4, "validation": 2, "test": 2}
    SPLITS.validate_records(first, counts)
    assert len({row["id"] for row in first}) == 8
    assert len({row["sha256"] for row in first}) == 8


def test_manifest_loader_rejects_wrong_hash_and_wrong_split(tmp_path):
    sequence = "MPEPTIDE"
    good = {
        "id": "p1",
        "source": "fixture",
        "sequence": sequence,
        "split": "train",
        "family": "f1",
        "sha256": SPLITS.sha256_bytes(sequence.encode()),
    }
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(good, allow_nan=False) + "\n")
    assert _load_sequence_manifest(path, "train") == [sequence]
    assert _load_sequence_manifest(path, "train", model_input_format="zymctrl_ec") == [
        "f1<sep><start>MPEPTIDE<end>"
    ]

    bad = dict(good, sha256="0" * 64)
    path.write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _load_sequence_manifest(path, "train")

    path.write_text(json.dumps(good) + "\n")
    with pytest.raises(ValueError, match="expected split"):
        _load_sequence_manifest(path, "test")


def test_zymctrl_normalization_separates_prompt_from_protein_sequence():
    assert SPLITS.normalize_sequence(
        "1.2.3.4<sep><start>MPEPTIDE<end>", "1.2.3.4", "zymctrl"
    ) == "MPEPTIDE"
    with pytest.raises(ValueError, match="prompt/header EC mismatch"):
        SPLITS.normalize_sequence(
            "9.9.9.9<sep><start>MPEPTIDE<end>", "1.2.3.4", "zymctrl"
        )
