from __future__ import annotations

import json
import hashlib
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

R2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R2))

from src.revision.dictionary_controls import (  # noqa: E402
    CachedLayerRows,
    CachedMultiLayerRows,
    DeterministicBatchStream,
    TrainingConfig,
    batch_stream_seed,
    build_windowed_transcoder,
    evaluate_windowed_transcoder,
    estimate_activation_cache_bytes,
    format_model_input,
    load_activation_cache,
    load_production_profile,
    require_cache_free_space,
    select_hash_priority_tokens,
    train_windowed_transcoder,
    trainable_parameter_count,
    valid_token_rows,
    windowed_transcoder_parameter_count,
    write_activation_cache,
)
from src.revision.io import sha256_file  # noqa: E402


def source_splits(tmp_path: Path) -> dict:
    result = {}
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(json.dumps({"split": split}) + "\n")
        result[split] = {
            "manifest_path": str(path),
            "manifest_sha256": sha256_file(path),
        }
    return result


def batches(padded_value: float, *, width: int = 4) -> dict:
    result = {}
    for split_index, split in enumerate(("train", "validation", "test")):
        base = torch.arange(
            2 * 3 * width,
            dtype=torch.float32,
        ).reshape(2, 3, width)
        base = base / 20 + split_index * 0.1
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        base[~mask.bool()] = padded_value
        result[split] = [{"inputs": {2: base}, "attention_mask": mask}]
    return result


def build_cache(
    tmp_path: Path,
    name: str,
    padded_value: float = 0.0,
    *,
    width: int = 4,
) -> Path:
    return write_activation_cache(
        tmp_path / name,
        batches(padded_value, width=width),
        selected_layers=[2],
        source_splits=source_splits(tmp_path),
    )


def transcode_batches(padded_value: float) -> dict:
    result = {}
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    for split_index, split in enumerate(("train", "validation", "test")):
        inputs = {}
        targets = {}
        for layer in (0, 1):
            values = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
            values = values / 10 + split_index * 0.2 + layer * 0.1
            target = 0.5 * values + layer
            values[~mask.bool()] = padded_value
            target[~mask.bool()] = -padded_value
            inputs[layer] = values
            targets[layer] = target
        result[split] = [
            {
                "inputs": inputs,
                "targets": targets,
                "attention_mask": mask,
            }
        ]
    return result


def build_transcode_cache(
    tmp_path: Path,
    name: str,
    padded_value: float = 0.0,
    *,
    storage_dtype: str = "float32",
) -> Path:
    return write_activation_cache(
        tmp_path / name,
        transcode_batches(padded_value),
        selected_layers=[0, 1],
        source_splits=source_splits(tmp_path),
        objective="transcode",
        storage_dtype=storage_dtype,
    )


def test_valid_token_cache_is_padding_invariant_and_all_valid_equivalent(tmp_path):
    first = load_activation_cache(build_cache(tmp_path, "cache-a", -999.0))
    second = load_activation_cache(build_cache(tmp_path, "cache-b", 12345.0))
    first_hashes = [(row["input_sha256"], row["target_sha256"]) for row in first.shards]
    second_hashes = [
        (row["input_sha256"], row["target_sha256"]) for row in second.shards
    ]
    assert first_hashes == second_hashes
    for split in ("train", "validation", "test"):
        summary = first.payload["split_summaries"][split]
        assert summary["total_token_rows"] == 6
        assert summary["valid_token_rows"] == 3
        assert summary["invalid_token_rows_excluded"] == 3

    tensor = torch.randn(2, 3, 4)
    mask = torch.ones(2, 3, dtype=torch.long)
    torch.testing.assert_close(
        valid_token_rows(tensor, mask),
        tensor.reshape(-1, 4),
    )


def test_prompt_reconstruction_and_hash_priority_selection_are_frozen():
    sequence = "MPEPTIDE"
    record = {
        "id": "x",
        "source": "fixture",
        "sequence": sequence,
        "split": "train",
        "family": "1.2.3.4",
        "sha256": hashlib.sha256(sequence.encode()).hexdigest(),
    }
    assert format_model_input(record, "sequence") == sequence
    assert (
        format_model_input(record, "zymctrl_ec") == "1.2.3.4<sep><start>MPEPTIDE<end>"
    )
    bad = dict(record, sha256="0" * 64)
    with pytest.raises(ValueError, match="sequence SHA-256 mismatch"):
        format_model_input(bad, "sequence")

    rows = [
        (hashlib.sha256(b"a").hexdigest(), 3),
        (hashlib.sha256(b"b").hexdigest(), 2),
    ]
    first = select_hash_priority_tokens(rows, budget=4)
    second = select_hash_priority_tokens(reversed(rows), budget=4)
    assert first == second
    assert len(first) == 4
    with pytest.raises(ValueError, match="cannot be met"):
        select_hash_priority_tokens(rows, budget=6)


def test_budgeted_cache_is_exact_priority_ordered_and_padding_clean(tmp_path):
    sources = {}
    batches_by_split = {}
    selected = {}
    for split_index, split in enumerate(("train", "validation", "test")):
        sequences = [f"M{split_index}A", f"M{split_index}B"]
        digests = [hashlib.sha256(value.encode()).hexdigest() for value in sequences]
        path = tmp_path / f"budget-{split}.jsonl"
        records = [
            {
                "id": f"{split}-{index}",
                "source": "fixture",
                "sequence": sequence,
                "split": split,
                "family": f"family-{index}",
                "sha256": digest,
            }
            for index, (sequence, digest) in enumerate(zip(sequences, digests))
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        sources[split] = {
            "manifest_path": str(path),
            "manifest_sha256": sha256_file(path),
        }
        selected[split] = select_hash_priority_tokens(
            [(digests[0], 2), (digests[1], 1)], budget=2
        )
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        values = (
            torch.tensor(
                [
                    [[10.0], [11.0], [-999.0]],
                    [[20.0], [-999.0], [-999.0]],
                ]
            )
            + split_index
        )
        batches_by_split[split] = [
            {
                "inputs": {0: values},
                "targets": {0: values + 0.5},
                "attention_mask": mask,
                "record_sha256": digests,
            }
        ]
    manifest = write_activation_cache(
        tmp_path / "budgeted-cache",
        batches_by_split,
        selected_layers=[0],
        source_splits=sources,
        objective="transcode",
        selected_tokens_by_split=selected,
        expected_dimensions={0: (1, 1)},
        expected_eligible_valid_token_rows_by_split={
            "train": 3,
            "validation": 3,
            "test": 3,
        },
    )
    cache = load_activation_cache(manifest)
    assert cache.payload["layout"] == "preallocated_single_file_per_layer_split"
    assert len(cache.shards) == 3
    for split in ("train", "validation", "test"):
        summary = cache.payload["split_summaries"][split]
        assert summary["selected_valid_token_rows"] == 2
        assert summary["eligible_valid_token_rows"] == 3
        assert summary["invalid_token_rows_excluded"] == 3
        rows = CachedLayerRows(cache, split, 0)
        assert rows.n_rows == 2
        assert np.all(np.concatenate(rows.inputs) > -100)

    impossible = {
        split: [
            {
                "record_sha256": rows[0]["record_sha256"],
                "token_position": 99,
                "priority_sha256": hashlib.sha256(
                    f"{rows[0]['record_sha256']}:99".encode("ascii")
                ).hexdigest(),
            }
        ]
        for split, rows in selected.items()
    }
    with pytest.raises(ValueError, match="exact valid-token budget was not met"):
        write_activation_cache(
            tmp_path / "impossible-cache",
            batches_by_split,
            selected_layers=[0],
            source_splits=sources,
            objective="transcode",
            selected_tokens_by_split=impossible,
        )

    test_source = (
        manifest.parent / cache.payload["source_splits"]["test"]["manifest_path"]
    )
    test_selection = (
        manifest.parent
        / cache.payload["token_selection"]["by_split"]["test"]["selection_path"]
    )
    test_source.unlink()
    test_selection.unlink()
    for shard in cache.shards:
        if shard["split"] == "test":
            (manifest.parent / shard["input_path"]).unlink()
            (manifest.parent / shard["target_path"]).unlink()
    scoped = load_activation_cache(
        manifest,
        access_splits=("train", "validation"),
    )
    assert scoped.verified_splits == ("train", "validation")
    with pytest.raises(ValueError, match="unknown cache split"):
        CachedMultiLayerRows(scoped, "test")
    with pytest.raises(FileNotFoundError, match="test"):
        load_activation_cache(manifest)


def test_cache_size_estimate_and_free_space_gate(tmp_path, monkeypatch):
    assert (
        estimate_activation_cache_bytes(
            valid_token_rows=1_200_000,
            n_layers=36,
            input_dim=1280,
            target_dim=1280,
            storage_dtype="float16",
        )
        == 221_184_000_000
    )
    usage = type("Usage", (), {"free": 100})()
    monkeypatch.setattr(
        "src.revision.dictionary_controls.shutil.disk_usage", lambda _: usage
    )
    with pytest.raises(OSError, match="insufficient cache space"):
        require_cache_free_space(
            tmp_path / "cache", estimated_bytes=90, safety_factor=1.2
        )


def test_cache_hash_tampering_fails_fast(tmp_path):
    manifest = build_cache(tmp_path, "cache")
    cache = load_activation_cache(manifest)
    shard = manifest.parent / cache.shards[0]["input_path"]
    with shard.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_activation_cache(manifest)


def test_cache_content_identity_excludes_runtime_provenance(tmp_path):
    stable = {"model_revision": "fixture", "executed_profile_sha256": "a" * 64}
    first = write_activation_cache(
        tmp_path / "runtime-a",
        transcode_batches(0.0),
        selected_layers=[0, 1],
        source_splits=source_splits(tmp_path),
        activation_provenance={
            **stable,
            "command": ["first"],
            "pod_name": "pod-a",
            "started_at_utc": "2026-01-01T00:00:00+00:00",
            "wall_time_seconds": 1.0,
        },
        objective="transcode",
    )
    second = write_activation_cache(
        tmp_path / "runtime-b",
        transcode_batches(0.0),
        selected_layers=[0, 1],
        source_splits=source_splits(tmp_path),
        activation_provenance={
            **stable,
            "command": ["second"],
            "pod_name": "pod-b",
            "started_at_utc": "2026-01-02T00:00:00+00:00",
            "wall_time_seconds": 9.0,
        },
        objective="transcode",
    )
    left = load_activation_cache(first)
    right = load_activation_cache(second)
    assert left.content_sha256 == right.content_sha256
    assert batch_stream_seed(17, -1, left.content_sha256) == batch_stream_seed(
        17, -1, right.content_sha256
    )


def test_dense_control_is_active_width_matched_not_raw_parameter_matched():
    dense = build_windowed_transcoder(
        method="dense_low_rank",
        n_layers=2,
        input_dim=8,
        target_dim=8,
        sparse_width=5,
        dense_rank=3,
        window=2,
        l1_coefficient=0.0,
        gated_auxiliary_coefficient=0.0,
    )
    sparse = build_windowed_transcoder(
        method="relu_l1_sae",
        n_layers=2,
        input_dim=8,
        target_dim=8,
        sparse_width=5,
        dense_rank=3,
        window=2,
        l1_coefficient=1e-3,
        gated_auxiliary_coefficient=0.0,
    )
    assert dense.width == 3
    assert trainable_parameter_count(dense) != trainable_parameter_count(sparse)
    with pytest.raises(ValueError, match="below both activation widths"):
        build_windowed_transcoder(
            method="dense_low_rank",
            n_layers=2,
            input_dim=8,
            target_dim=8,
            sparse_width=5,
            dense_rank=8,
            window=2,
            l1_coefficient=0.0,
            gated_auxiliary_coefficient=0.0,
        )


def test_cached_topk_has_exact_bottleneck_and_parameter_plan():
    model = build_windowed_transcoder(
        method="topk_clt",
        n_layers=2,
        input_dim=4,
        target_dim=4,
        sparse_width=5,
        dense_rank=2,
        window=2,
        topk_k=2,
        l1_coefficient=0.0,
        gated_auxiliary_coefficient=0.0,
    )
    codes = model.encode([torch.randn(7, 4), torch.randn(7, 4)])
    assert all((code > 0).sum(dim=1).max().item() <= 2 for code in codes)
    assert trainable_parameter_count(model) == windowed_transcoder_parameter_count(
        method="topk_clt",
        n_layers=2,
        input_dim=4,
        target_dim=4,
        sparse_width=5,
        dense_rank=2,
        window=2,
    )


def test_windowed_controls_share_transcode_target_and_cross_layer_edges(tmp_path):
    cache = load_activation_cache(build_transcode_cache(tmp_path, "cache"))
    rows = CachedMultiLayerRows(cache, "validation")
    prefix = CachedMultiLayerRows(cache, "validation", prefix_rows=2)
    assert prefix.n_rows == 2
    with pytest.raises(IndexError, match="selected prefix"):
        prefix.take([2])
    model = build_windowed_transcoder(
        method="dense_low_rank",
        n_layers=2,
        input_dim=2,
        target_dim=2,
        sparse_width=3,
        dense_rank=1,
        window=2,
        l1_coefficient=0.0,
        gated_auxiliary_coefficient=0.0,
    )
    with torch.no_grad():
        model.encoder_weight.zero_()
        model.encoder_bias.zero_()
        model.decoder_weight[0].zero_()
        model.decoder_weight[1].zero_()
        model.decoder_bias.zero_()
        model.encoder_weight[0, 0, 0] = 1.0
        model.decoder_weight[0][0, 1, 1] = 2.0
    inputs, _ = rows.take([0])
    reconstruction, _ = model.reconstruct(inputs)
    assert reconstruction[0].equal(torch.zeros_like(reconstruction[0]))
    assert reconstruction[1][0, 1] == 2 * inputs[0][0, 0]
    original_reconstruct = model.reconstruct

    def assert_inference_mode(*args, **kwargs):
        assert not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled()
        return original_reconstruct(*args, **kwargs)

    model.reconstruct = assert_inference_mode
    metrics = evaluate_windowed_transcoder(
        model,
        rows,
        device=torch.device("cpu"),
        batch_size=2,
        activation_threshold=0.0,
        dead_frequency_threshold=0.001,
        detailed=True,
    )
    assert metrics["objective"] == (
        "windowed_multi_layer_clt_input_to_mlp_output_transcoding"
    )
    assert metrics["decoder_window"] == 2
    assert len(metrics["target_layers"]) == 2
    assert len(metrics["source_layers"]) == 2
    assert set(metrics["reconstruction_error_quantiles"]) == {
        "q00",
        "q25",
        "q50",
        "q75",
        "q90",
        "q95",
        "q99",
        "q100",
    }
    assert all(
        len(row["firing_frequency_per_feature"]) == model.width
        for row in metrics["source_layers"]
    )


def test_windowed_metrics_are_padding_invariant_end_to_end(tmp_path):
    first = load_activation_cache(build_transcode_cache(tmp_path, "first", -1000.0))
    second = load_activation_cache(build_transcode_cache(tmp_path, "second", 1000.0))
    assert first.content_sha256 == second.content_sha256
    torch.manual_seed(17)
    model = build_windowed_transcoder(
        method="relu_l1_sae",
        n_layers=2,
        input_dim=2,
        target_dim=2,
        sparse_width=3,
        dense_rank=1,
        window=2,
        l1_coefficient=1e-3,
        gated_auxiliary_coefficient=0.0,
    )
    kwargs = {
        "device": torch.device("cpu"),
        "batch_size": 2,
        "activation_threshold": 0.0,
        "dead_frequency_threshold": 0.001,
        "detailed": True,
    }
    left = evaluate_windowed_transcoder(
        model,
        CachedMultiLayerRows(first, "test"),
        **kwargs,
    )
    right = evaluate_windowed_transcoder(
        model,
        CachedMultiLayerRows(second, "test"),
        **kwargs,
    )
    assert left == right


def test_float16_production_cache_is_cast_to_model_dtype(tmp_path):
    cache = load_activation_cache(
        build_transcode_cache(tmp_path, "float16-cache", storage_dtype="float16")
    )
    rows = CachedMultiLayerRows(cache, "validation")
    model = build_windowed_transcoder(
        method="dense_low_rank",
        n_layers=2,
        input_dim=2,
        target_dim=2,
        sparse_width=3,
        dense_rank=1,
        window=2,
        l1_coefficient=0.0,
        gated_auxiliary_coefficient=0.0,
    )
    assert model.encoder_weight.dtype == torch.float32
    metrics = evaluate_windowed_transcoder(
        model,
        rows,
        device=torch.device("cpu"),
        batch_size=2,
        activation_threshold=0.0,
        dead_frequency_threshold=0.001,
        detailed=False,
    )
    assert np.isfinite(metrics["fvu_mean"])


def test_bfloat16_activations_round_trip_through_float16_cache(tmp_path):
    activation_batches = transcode_batches(0.0)
    for split_batches in activation_batches.values():
        for batch in split_batches:
            batch["inputs"] = {
                layer: tensor.to(torch.bfloat16)
                for layer, tensor in batch["inputs"].items()
            }
            batch["targets"] = {
                layer: tensor.to(torch.bfloat16)
                for layer, tensor in batch["targets"].items()
            }

    manifest = write_activation_cache(
        tmp_path / "bfloat16-cache",
        activation_batches,
        selected_layers=[0, 1],
        source_splits=source_splits(tmp_path),
        objective="transcode",
        storage_dtype="float16",
    )
    cache = load_activation_cache(manifest)
    shard = next(
        row for row in cache.shards if row["split"] == "train" and row["layer"] == 0
    )
    stored = np.load(manifest.parent / shard["input_path"], allow_pickle=False)
    assert stored.dtype == np.float16
    assert np.isfinite(stored).all()


def test_minibatch_stream_is_deterministic_and_resumable():
    seed = batch_stream_seed(17, 2, "a" * 64)
    first = DeterministicBatchStream(5, 2, seed)
    second = DeterministicBatchStream(5, 2, seed)
    for _ in range(5):
        np.testing.assert_array_equal(first.next(), second.next())
    state = first.state_dict()
    restored = DeterministicBatchStream(5, 2, seed)
    restored.load_state_dict(state)
    np.testing.assert_array_equal(first.next(), restored.next())


def test_windowed_training_resumes_optimizer_rng_and_stream(tmp_path):
    cache = load_activation_cache(build_transcode_cache(tmp_path, "cache"))
    train_rows = CachedMultiLayerRows(cache, "train")
    validation_rows = CachedMultiLayerRows(cache, "validation")
    config = TrainingConfig(
        seed=17,
        steps=4,
        batch_size=2,
        evaluation_batch_size=3,
        learning_rate=1e-3,
        validation_every=2,
        gradient_clip_norm=1.0,
        warmup_steps=1,
        checkpoint_every=2,
    )
    kwargs = {
        "method": "dense_low_rank",
        "n_layers": 2,
        "input_dim": 2,
        "target_dim": 2,
        "sparse_width": 3,
        "dense_rank": 1,
        "window": 2,
        "l1_coefficient": 0.0,
        "gated_auxiliary_coefficient": 0.0,
    }
    torch.manual_seed(41)
    interrupted = build_windowed_transcoder(**kwargs)
    original_objective = interrupted.objective
    calls = 0

    def interrupt_after_checkpoint(inputs, targets):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("synthetic interruption")
        return original_objective(inputs, targets)

    interrupted.objective = interrupt_after_checkpoint
    progress = tmp_path / "progress.pt"
    best = tmp_path / "best.pt"
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        train_windowed_transcoder(
            interrupted,
            train_rows,
            validation_rows,
            device=torch.device("cpu"),
            config=config,
            stream_seed=91,
            candidate_id="fixture",
            progress_path=progress,
            best_path=best,
            resume=False,
        )
    assert progress.is_file()

    torch.manual_seed(999)
    resumed_model = build_windowed_transcoder(**kwargs)
    resumed = train_windowed_transcoder(
        resumed_model,
        train_rows,
        validation_rows,
        device=torch.device("cpu"),
        config=config,
        stream_seed=91,
        candidate_id="fixture",
        progress_path=progress,
        best_path=best,
        resume=True,
    )
    assert resumed["resumed"] is True
    assert resumed["resume_start_step"] == 2
    assert resumed["best_validation_step"] in {2, 4}


def test_windowed_training_resumes_from_completed_progress(tmp_path):
    cache = load_activation_cache(build_transcode_cache(tmp_path, "cache"))
    train_rows = CachedMultiLayerRows(cache, "train")
    validation_rows = CachedMultiLayerRows(cache, "validation")
    config = TrainingConfig(
        seed=17,
        steps=2,
        batch_size=2,
        evaluation_batch_size=3,
        learning_rate=1e-3,
        validation_every=1,
        gradient_clip_norm=1.0,
        checkpoint_every=1,
    )
    kwargs = {
        "method": "dense_low_rank",
        "n_layers": 2,
        "input_dim": 2,
        "target_dim": 2,
        "sparse_width": 3,
        "dense_rank": 1,
        "window": 2,
        "l1_coefficient": 0.0,
        "gated_auxiliary_coefficient": 0.0,
    }
    progress = tmp_path / "final-progress.pt"
    best = tmp_path / "final-best.pt"
    first = train_windowed_transcoder(
        build_windowed_transcoder(**kwargs),
        train_rows,
        validation_rows,
        device=torch.device("cpu"),
        config=config,
        stream_seed=92,
        candidate_id="completed-fixture",
        progress_path=progress,
        best_path=best,
        resume=False,
    )
    checkpoint_wall_time = torch.load(progress, map_location="cpu", weights_only=False)[
        "wall_time_seconds"
    ]
    resumed = train_windowed_transcoder(
        build_windowed_transcoder(**kwargs),
        train_rows,
        validation_rows,
        device=torch.device("cpu"),
        config=config,
        stream_seed=92,
        candidate_id="completed-fixture",
        progress_path=progress,
        best_path=best,
        resume=True,
    )
    assert resumed["resume_start_step"] == config.steps
    assert resumed["validation_history"] == first["validation_history"]
    assert resumed["wall_time_seconds"] >= checkpoint_wall_time
    assert first["wall_time_seconds"] >= checkpoint_wall_time
    assert resumed["checkpoint_io_seconds"] > 0
    timing_path = progress.with_name(f"{progress.name}.timing.json")
    timing = json.loads(timing_path.read_text())
    timing["step"] = 1
    timing_path.write_text(json.dumps(timing))
    with pytest.raises(ValueError, match="timing sidecar"):
        train_windowed_transcoder(
            build_windowed_transcoder(**kwargs),
            train_rows,
            validation_rows,
            device=torch.device("cpu"),
            config=config,
            stream_seed=92,
            candidate_id="completed-fixture",
            progress_path=progress,
            best_path=best,
            resume=True,
        )


def test_completed_training_releases_optimizer_state(monkeypatch, tmp_path):
    cache = load_activation_cache(build_transcode_cache(tmp_path, "cache"))
    train_rows = CachedMultiLayerRows(cache, "train")
    validation_rows = CachedMultiLayerRows(cache, "validation")
    optimizers = []
    adam = torch.optim.Adam

    class TrackingAdam(adam):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            optimizers.append(self)

    monkeypatch.setattr(torch.optim, "Adam", TrackingAdam)
    model = build_windowed_transcoder(
        method="dense_low_rank",
        n_layers=2,
        input_dim=2,
        target_dim=2,
        sparse_width=3,
        dense_rank=1,
        window=2,
        l1_coefficient=0.0,
        gated_auxiliary_coefficient=0.0,
    )
    train_windowed_transcoder(
        model,
        train_rows,
        validation_rows,
        device=torch.device("cpu"),
        config=TrainingConfig(
            seed=17,
            steps=2,
            batch_size=2,
            evaluation_batch_size=3,
            learning_rate=1e-3,
            validation_every=1,
            gradient_clip_norm=1.0,
            checkpoint_every=1,
        ),
        stream_seed=93,
        candidate_id="release-fixture",
        progress_path=tmp_path / "release-progress.pt",
        best_path=tmp_path / "release-best.pt",
        resume=False,
    )

    assert len(optimizers) == 1
    assert not optimizers[0].state
    assert all(parameter.grad is None for parameter in model.parameters())


def test_runner_validation_reloads_best_checkpoint_before_metrics(tmp_path):
    runner = runpy.run_path(str(R2 / "scripts/58_run_dictionary_controls.py"))
    reload_best = runner["_load_best_checkpoint_into_model"]
    cache = load_activation_cache(build_transcode_cache(tmp_path, "cache"))
    validation_rows = CachedMultiLayerRows(cache, "validation")
    kwargs = {
        "method": "dense_low_rank",
        "n_layers": 2,
        "input_dim": 2,
        "target_dim": 2,
        "sparse_width": 3,
        "dense_rank": 1,
        "window": 2,
        "l1_coefficient": 0.0,
        "gated_auxiliary_coefficient": 0.0,
    }
    torch.manual_seed(7)
    model = build_windowed_transcoder(**kwargs)
    expected = evaluate_windowed_transcoder(
        model,
        validation_rows,
        device=torch.device("cpu"),
        batch_size=3,
        activation_threshold=0.0,
        dead_frequency_threshold=0.001,
        detailed=False,
    )
    best_path = tmp_path / "best.pt"
    torch.save(
        {
            "schema_version": "r2_dictionary_control_best_v1",
            "candidate_id": "fixture",
            "step": 2,
            "validation_fvu_mean": expected["fvu_mean"],
            "model_state_dict": {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            },
        },
        best_path,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(4.0)
    final_weights = evaluate_windowed_transcoder(
        model,
        validation_rows,
        device=torch.device("cpu"),
        batch_size=3,
        activation_threshold=0.0,
        dead_frequency_threshold=0.001,
        detailed=False,
    )
    assert final_weights["fvu_mean"] != pytest.approx(expected["fvu_mean"])

    reload_best(
        model,
        best_path,
        candidate_id="fixture",
        device=torch.device("cpu"),
    )
    reloaded = evaluate_windowed_transcoder(
        model,
        validation_rows,
        device=torch.device("cpu"),
        batch_size=3,
        activation_threshold=0.0,
        dead_frequency_threshold=0.001,
        detailed=False,
    )
    assert reloaded["fvu_mean"] == pytest.approx(expected["fvu_mean"], abs=0.0)


def test_runner_preflight_cache_contract_is_nonconfirmatory():
    runner = runpy.run_path(str(R2 / "scripts/58_run_dictionary_controls.py"))
    validate = runner["_validate_h200_preflight_cache"]
    profile = {
        "preflight": {
            "mode": "bounded_nonconfirmatory_h200_preflight",
            "valid_token_rows_per_split": 2,
        },
        "cache_extraction": {
            "model_cache_geometry": {
                "tiny": {"n_layers": 2, "input_dim": 3, "target_dim": 3}
            }
        },
    }
    payload = {
        "activation_provenance": {
            "execution_mode": "bounded_nonconfirmatory_h200_preflight",
            "production_scientific_eligibility": False,
            "production_cache_reuse_forbidden": True,
            "production_profile_sha256": "a" * 64,
        },
        "split_summaries": {
            split: {"selected_valid_token_rows": 2}
            for split in ("train", "validation", "test")
        },
    }
    cache = SimpleNamespace(
        objective="transcode",
        selected_layers=(0, 1),
        dimensions={0: (3, 3), 1: (3, 3)},
        payload=payload,
    )
    validate(cache, profile, model_name="tiny", profile_sha256="a" * 64)
    payload["activation_provenance"]["production_scientific_eligibility"] = True
    with pytest.raises(ValueError, match="nonconfirmatory preflight cache"):
        validate(cache, profile, model_name="tiny", profile_sha256="a" * 64)


def test_frozen_production_profile_is_strict_and_common_objective():
    path = R2 / "configs/p0_2_dictionary_controls_production_profile.json"
    digest = sha256_file(path)
    profile = load_production_profile(path, digest)
    assert profile["cache_extraction"]["panel_completion_receipt_schema"] == (
        "r2_dictionary_cache_completion_receipt_v3"
    )
    assert profile["estimand"]["decoder_window"] == 8
    assert profile["panel"]["relu_l1_sae"]["width"] == 8192
    assert profile["compute_schedule"]["planning_estimate"]["full_runs_per_model"] == 12
    assert (
        profile["checkpoint_storage_planning"]["retained_checkpoint_bytes_total"]
        == 3_429_943_879_680
    )
    assert profile["panel"]["dense_low_rank"]["rank"] == 128
    assert profile["panel"]["dense_low_rank"]["raw_parameter_matched"] is False
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_production_profile(path, "0" * 64)
