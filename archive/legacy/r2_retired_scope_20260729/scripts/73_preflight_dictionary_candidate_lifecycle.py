#!/usr/bin/env python3
"""Repeat full-width one-step candidates to verify bounded CUDA cleanup."""

from __future__ import annotations

import argparse
import gc
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

R2_ROOT = Path(__file__).resolve().parents[1]
if str(R2_ROOT) not in sys.path:
    sys.path.insert(0, str(R2_ROOT))

from src.revision.dictionary_controls import (  # noqa: E402
    CachedMultiLayerRows,
    TrainingConfig,
    batch_stream_seed,
    build_windowed_transcoder,
    configure_determinism,
    load_activation_cache,
    load_production_profile,
    model_seed,
    train_windowed_transcoder,
)
from src.revision.io import sha256_file, write_json  # noqa: E402

MAX_POST_CANDIDATE_CUDA_BYTES = 128 * 1024**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--method", choices=("relu_l1_sae", "gated_sae"), required=True)
    parser.add_argument("--repetitions", type=int, default=4)
    return parser.parse_args()


def validate_preflight_cache(cache, profile: dict, args: argparse.Namespace) -> None:
    preflight = profile["preflight"]
    geometry = profile["cache_extraction"]["model_cache_geometry"][args.model_name]
    provenance = cache.payload.get("activation_provenance", {})
    if (
        cache.manifest_sha256 != args.cache_sha256
        or cache.objective != "transcode"
        or cache.selected_layers != tuple(range(geometry["n_layers"]))
        or any(
            cache.dimensions[layer] != (geometry["input_dim"], geometry["target_dim"])
            for layer in cache.selected_layers
        )
        or provenance.get("execution_mode") != preflight["mode"]
        or provenance.get("production_scientific_eligibility") is not False
        or provenance.get("production_cache_reuse_forbidden") is not True
        or provenance.get("production_profile_sha256") != args.profile_sha256
        or any(
            cache.payload["split_summaries"][split]["selected_valid_token_rows"]
            != preflight["valid_token_rows_per_split"]
            for split in ("train", "validation", "test")
        )
    ):
        raise ValueError("cache is not the pinned nonconfirmatory preflight cache")


def build_model(profile: dict, cache, method: str):
    return build_windowed_transcoder(
        method=method,
        n_layers=len(cache.selected_layers),
        input_dim=cache.dimensions[cache.selected_layers[0]][0],
        target_dim=cache.dimensions[cache.selected_layers[0]][1],
        sparse_width=profile["panel"]["relu_l1_sae"]["width"],
        dense_rank=profile["panel"]["dense_low_rank"]["rank"],
        window=profile["estimand"]["decoder_window"],
        l1_coefficient=profile["loss_grids"][method]["l1_coefficient"][0],
        gated_auxiliary_coefficient=(
            profile["loss_grids"][method]["auxiliary_coefficient"][-1]
            if method == "gated_sae"
            else 0.0
        ),
        topk_k=profile["panel"]["topk_clt"]["k"],
    )


def main() -> None:
    args = parse_args()
    if not 2 <= args.repetitions <= 4:
        raise ValueError("repetitions must be in 2..4")
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("candidate-lifecycle preflight requires CUDA")
    device = torch.device(args.device)
    if device.index != args.gpu_index:
        raise ValueError("--device and --gpu-index disagree")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite preflight: {args.output_dir}")

    torch.cuda.set_device(device)
    profile = load_production_profile(args.profile, args.profile_sha256)
    cache = load_activation_cache(
        args.cache_manifest,
        verify_hashes=True,
        access_splits=("train", "validation"),
    )
    validate_preflight_cache(cache, profile, args)
    args.output_dir.mkdir(parents=True)
    train_rows = CachedMultiLayerRows(cache, "train")
    validation_rows = CachedMultiLayerRows(cache, "validation")
    seed = profile["preflight"]["dictionary_seed"]
    config = TrainingConfig(
        seed=seed,
        steps=1,
        batch_size=profile["preflight"]["dictionary_batch_valid_token_rows"],
        evaluation_batch_size=profile["preflight"]["dictionary_batch_valid_token_rows"],
        learning_rate=profile["training"]["learning_rate"],
        validation_every=1,
        gradient_clip_norm=profile["training"]["gradient_clip_norm"],
    )
    repetitions = []
    cleanup_baseline: int | None = None
    for index in range(args.repetitions):
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        before = int(torch.cuda.memory_allocated(device))
        expected_before = 0 if cleanup_baseline is None else cleanup_baseline
        if before != expected_before:
            raise RuntimeError(
                "preflight repetition starts outside its established baseline: "
                f"{expected_before} -> {before} bytes"
            )
        torch.cuda.reset_peak_memory_stats(device)
        configure_determinism(model_seed(seed, -1, args.method))
        model = build_model(profile, cache, args.method)
        best_path = args.output_dir / f"candidate_{index:03d}.best.pt"
        training = train_windowed_transcoder(
            model,
            train_rows,
            validation_rows,
            device=device,
            config=config,
            stream_seed=batch_stream_seed(seed, -1, cache.content_sha256),
            candidate_id=f"candidate_lifecycle_preflight_{index:03d}",
            progress_path=args.output_dir / f"candidate_{index:03d}.progress.pt",
            best_path=best_path,
            resume=False,
        )
        checkpoint = {
            "bytes": best_path.stat().st_size,
            "sha256": sha256_file(best_path),
        }
        best_path.unlink()
        del model
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        after = int(torch.cuda.memory_allocated(device))
        reserved = int(torch.cuda.memory_reserved(device))
        if after > MAX_POST_CANDIDATE_CUDA_BYTES:
            raise RuntimeError(f"preflight repetition left {after} live bytes")
        if cleanup_baseline is None:
            cleanup_baseline = after
        elif after != cleanup_baseline:
            raise RuntimeError(
                "preflight cleanup baseline changed: "
                f"{cleanup_baseline} -> {after} bytes"
            )
        repetitions.append(
            {
                "repetition": index,
                "allocated_bytes_before": before,
                "peak_allocated_bytes": training[
                    "peak_accelerator_memory_allocated_bytes"
                ],
                "peak_reserved_bytes": training[
                    "peak_accelerator_memory_reserved_bytes"
                ],
                "allocated_bytes_after": after,
                "allocated_bytes_limit": MAX_POST_CANDIDATE_CUDA_BYTES,
                "reserved_bytes_after_empty_cache": reserved,
                "training_loss": training["last_training_components"]["loss"],
                "validation_fvu_mean": training["validation_history"][-1]["fvu_mean"],
                "checkpoint": checkpoint,
            }
        )

    report = {
        "schema_version": "r2_dictionary_candidate_lifecycle_preflight_v1",
        "status": "completed_nonconfirmatory_candidate_lifecycle_preflight",
        "p0_2_eligible": False,
        "production_result_reuse_forbidden": True,
        "model_name": args.model_name,
        "method": args.method,
        "repetitions": repetitions,
        "test_evaluation_count": 0,
        "cache_manifest_sha256": cache.manifest_sha256,
        "cache_content_sha256": cache.content_sha256,
        "profile_sha256": args.profile_sha256,
        "runner_sha256": sha256_file(Path(__file__)),
        "training_module_sha256": sha256_file(
            R2_ROOT / "src/revision/dictionary_controls.py"
        ),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(args.output_dir / "preflight_report.json", report)
    print("completed repeated candidate-lifecycle preflight")


if __name__ == "__main__":
    main()
