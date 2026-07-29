#!/usr/bin/env python3
"""Train the frozen P0-2 alternative dictionary panel from one activation cache."""

from __future__ import annotations

import argparse
import gc
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

R2_ROOT = Path(__file__).resolve().parents[1]
if str(R2_ROOT) not in sys.path:
    sys.path.insert(0, str(R2_ROOT))

from src.revision.dictionary_controls import (  # noqa: E402
    METHODS,
    RESULT_SCHEMA,
    CachedMultiLayerRows,
    TrainingConfig,
    batch_stream_seed,
    build_windowed_transcoder,
    configure_determinism,
    evaluate_windowed_transcoder,
    load_activation_cache,
    load_production_profile,
    load_strict_json,
    model_seed,
    require_cache_free_space,
    train_windowed_transcoder,
    validate_production_cache,
    windowed_transcoder_parameter_count,
)
from src.revision.io import sha256_file, write_json  # noqa: E402


FROZEN_SEEDS = (17, 29, 43)
MAX_POST_CANDIDATE_CUDA_BYTES = 128 * 1024**2


def _load_best_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_path: Path,
    *,
    candidate_id: str,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "r2_dictionary_control_best_v1":
        raise ValueError("unsupported best-checkpoint schema")
    if checkpoint.get("candidate_id") != candidate_id:
        raise ValueError("best-checkpoint candidate identity mismatch")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--stage", choices=("screening", "full"))
    parser.add_argument("--screening-result", type=Path)
    parser.add_argument("--screening-result-sha256")
    parser.add_argument("--model-name")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--profile-sha256")
    parser.add_argument("--pod-name")
    parser.add_argument("--node-name")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--git-commit")
    parser.add_argument("--git-dirty", choices=("true", "false"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _require_production_args(args: argparse.Namespace) -> None:
    required = {
        "stage": args.stage,
        "method": args.method,
        "model_name": args.model_name,
        "profile": args.profile,
        "profile_sha256": args.profile_sha256,
        "pod_name": args.pod_name,
        "node_name": args.node_name,
        "gpu_index": args.gpu_index,
        "git_commit": args.git_commit,
        "git_dirty": args.git_dirty,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"production mode lacks required arguments: {missing}")


def _candidate_grid(profile: dict, method: str) -> list[dict]:
    if method in {"topk_clt", "dense_low_rank"}:
        return [
            {
                "candidate_index": 0,
                "l1_coefficient": 0.0,
                "auxiliary_coefficient": 0.0,
            }
        ]
    loss = profile["loss_grids"][method]
    if method == "relu_l1_sae":
        return [
            {
                "candidate_index": index,
                "l1_coefficient": value,
                "auxiliary_coefficient": 0.0,
            }
            for index, value in enumerate(loss["l1_coefficient"])
        ]
    if method == "gated_sae":
        return [
            {
                "candidate_index": index,
                "l1_coefficient": l1,
                "auxiliary_coefficient": auxiliary,
            }
            for index, (l1, auxiliary) in enumerate(
                (l1, auxiliary)
                for l1 in loss["l1_coefficient"]
                for auxiliary in loss["auxiliary_coefficient"]
            )
        ]
    raise AssertionError("unreachable dictionary method")


def _build_production_model(
    profile: dict,
    cache,
    method: str,
    candidate: dict,
):
    return build_windowed_transcoder(
        method=method,
        n_layers=len(cache.selected_layers),
        input_dim=cache.dimensions[cache.selected_layers[0]][0],
        target_dim=cache.dimensions[cache.selected_layers[0]][1],
        sparse_width=profile["panel"]["relu_l1_sae"]["width"],
        dense_rank=profile["panel"]["dense_low_rank"]["rank"],
        window=profile["estimand"]["decoder_window"],
        l1_coefficient=candidate["l1_coefficient"],
        gated_auxiliary_coefficient=candidate["auxiliary_coefficient"],
        topk_k=profile["panel"]["topk_clt"]["k"],
    )


def _checkpoint_storage_requirement(
    profile: dict,
    *,
    model_name: str,
    method: str,
    candidate_count: int,
) -> tuple[int, int, int]:
    geometry = profile["cache_extraction"]["model_cache_geometry"][model_name]
    parameters = windowed_transcoder_parameter_count(
        method=method,
        n_layers=geometry["n_layers"],
        input_dim=geometry["input_dim"],
        target_dim=geometry["target_dim"],
        sparse_width=profile["panel"]["relu_l1_sae"]["width"],
        dense_rank=profile["panel"]["dense_low_rank"]["rank"],
        window=profile["estimand"]["decoder_window"],
    )
    retained_bytes = (
        parameters
        * candidate_count
        * profile["checkpoint_storage_planning"][
            "retained_bytes_per_parameter_per_candidate"
        ]
    )
    peak_live_bytes = (
        retained_bytes
        + parameters
        * profile["checkpoint_storage_planning"][
            "atomic_progress_rewrite_temporary_bytes_per_parameter"
        ]
    )
    return parameters, retained_bytes, peak_live_bytes


def _validate_h200_preflight_cache(
    cache,
    profile: dict,
    *,
    model_name: str,
    profile_sha256: str,
) -> None:
    preflight = profile["preflight"]
    geometry = profile["cache_extraction"]["model_cache_geometry"][model_name]
    provenance = cache.payload.get("activation_provenance", {})
    rows_per_split = preflight["valid_token_rows_per_split"]
    if (
        cache.objective != "transcode"
        or cache.selected_layers != tuple(range(geometry["n_layers"]))
        or any(
            cache.dimensions[layer] != (geometry["input_dim"], geometry["target_dim"])
            for layer in cache.selected_layers
        )
        or provenance.get("execution_mode") != preflight["mode"]
        or provenance.get("production_scientific_eligibility") is not False
        or provenance.get("production_cache_reuse_forbidden") is not True
        or provenance.get("production_profile_sha256") != profile_sha256
        or any(
            cache.payload["split_summaries"][split]["selected_valid_token_rows"]
            != rows_per_split
            for split in ("train", "validation", "test")
        )
    ):
        raise ValueError("cache is not a valid bounded nonconfirmatory preflight cache")


def run_h200_preflight(args: argparse.Namespace) -> None:
    """Execute one full-size optimizer step on a bounded nonconfirmatory cache."""

    required = {
        "method": args.method,
        "model_name": args.model_name,
        "profile": args.profile,
        "profile_sha256": args.profile_sha256,
        "pod_name": args.pod_name,
        "node_name": args.node_name,
        "gpu_index": args.gpu_index,
        "git_commit": args.git_commit,
        "git_dirty": args.git_dirty,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"preflight lacks required arguments: {missing}")
    if (
        args.stage is not None
        or args.resume
        or args.screening_result is not None
        or args.screening_result_sha256 is not None
    ):
        raise ValueError("preflight forbids stage, resume and screening-result inputs")
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("dictionary preflight requires one explicit CUDA device")
    device = torch.device(args.device)
    if device.index != args.gpu_index:
        raise ValueError("--device and --gpu-index disagree")
    torch.cuda.set_device(device)
    profile = load_production_profile(args.profile, args.profile_sha256)
    preflight = profile["preflight"]
    if args.seed != preflight["dictionary_seed"]:
        raise ValueError("preflight seed differs from the frozen profile")
    cache = load_activation_cache(
        args.cache_manifest,
        verify_hashes=True,
        access_splits=("train", "validation"),
    )
    if cache.manifest_sha256 != args.cache_sha256:
        raise ValueError("preflight cache manifest SHA-256 mismatch")
    _validate_h200_preflight_cache(
        cache,
        profile,
        model_name=args.model_name,
        profile_sha256=args.profile_sha256,
    )
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite preflight: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    configure_determinism(model_seed(args.seed, -1, args.method))
    candidate = {
        "candidate_index": 0,
        "l1_coefficient": 0.0,
        "auxiliary_coefficient": 1.0 if args.method == "gated_sae" else 0.0,
    }
    model = _build_production_model(profile, cache, args.method, candidate).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=profile["training"]["learning_rate"]
    )
    batch_size = preflight["dictionary_batch_valid_token_rows"]
    train_rows = CachedMultiLayerRows(cache, "train")
    validation_rows = CachedMultiLayerRows(cache, "validation")
    train_inputs, train_targets = train_rows.take(range(batch_size))
    model_dtype = model.encoder_weight.dtype
    train_inputs = [
        values.to(device=device, dtype=model_dtype) for values in train_inputs
    ]
    train_targets = [
        values.to(device=device, dtype=model_dtype) for values in train_targets
    ]
    torch.cuda.synchronize(device)
    training_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    components = model.objective(train_inputs, train_targets)
    if not torch.isfinite(components["loss"]):
        raise FloatingPointError("preflight produced a non-finite training loss")
    components["loss"].backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        profile["training"]["gradient_clip_norm"],
        error_if_nonfinite=True,
    )
    optimizer.step()
    model.normalize_decoder()
    torch.cuda.synchronize(device)
    training_wall_time = time.perf_counter() - training_started
    validation_inputs, validation_targets = validation_rows.take(range(batch_size))
    validation_inputs = [
        values.to(device=device, dtype=model_dtype) for values in validation_inputs
    ]
    validation_targets = [
        values.to(device=device, dtype=model_dtype) for values in validation_targets
    ]
    validation_started = time.perf_counter()
    with torch.inference_mode():
        validation = model.objective(validation_inputs, validation_targets)
    if not torch.isfinite(validation["loss"]):
        raise FloatingPointError("preflight produced a non-finite validation loss")
    torch.cuda.synchronize(device)
    validation_wall_time = time.perf_counter() - validation_started
    report = {
        "schema_version": "r2_dictionary_h200_optimizer_preflight_v1",
        "status": "completed_nonconfirmatory_preflight",
        "p0_2_eligible": False,
        "production_result_reuse_forbidden": True,
        "command": sys.argv,
        "pod_name": args.pod_name,
        "node_name": args.node_name,
        "gpu_index": args.gpu_index,
        "gpu_model": torch.cuda.get_device_name(device),
        "model_name": args.model_name,
        "method": args.method,
        "seed": args.seed,
        "profile_sha256": args.profile_sha256,
        "cache_manifest_sha256": cache.manifest_sha256,
        "cache_content_sha256": cache.content_sha256,
        "optimizer_steps": preflight["dictionary_optimizer_steps"],
        "batch_valid_token_rows": batch_size,
        "training_loss": float(components["loss"].detach().item()),
        "validation_loss": float(validation["loss"].item()),
        "optimizer_step_wall_time_seconds": training_wall_time,
        "optimizer_steps_per_second": 1.0 / training_wall_time,
        "validation_wall_time_seconds": validation_wall_time,
        "raw_trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "training_flop_proxy_per_token": model.training_flop_proxy_per_token(),
        "peak_accelerator_memory_allocated_bytes": torch.cuda.max_memory_allocated(
            device
        ),
        "peak_accelerator_memory_reserved_bytes": torch.cuda.max_memory_reserved(
            device
        ),
        "test_evaluation_count": 0,
        "git_commit": args.git_commit,
        "git_dirty": args.git_dirty == "true",
        "runner_sha256": sha256_file(Path(__file__)),
        "module_sha256": sha256_file(R2_ROOT / "src/revision/dictionary_controls.py"),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(args.output_dir / "preflight_report.json", report)
    print("wrote bounded nonconfirmatory dictionary-optimizer H200 preflight")


def _write_final_production_artifacts(
    *,
    args: argparse.Namespace,
    profile: dict,
    profile_sha256: str,
    cache,
    provenance: dict,
    result: dict,
    started_at: str,
    selected_checkpoint: Path | None,
    model_seed_value: int,
    stream_seed: int,
) -> None:
    result_path = args.output_dir / "results.json"
    write_json(result_path, result)
    result_sha256 = sha256_file(result_path)
    checkpoint_sha256 = (
        sha256_file(selected_checkpoint) if selected_checkpoint is not None else None
    )
    device = torch.device(args.device)
    gpu_model = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu_smoke_forbidden"
    )
    manifest = {
        "schema_version": "r2_dictionary_control_run_manifest_v2",
        "status": result["status"],
        "command": sys.argv,
        "pod_name": args.pod_name,
        "node_name": args.node_name,
        "gpu_index": args.gpu_index,
        "gpu_model": gpu_model,
        "model_name": args.model_name,
        "model_revision": provenance["model_revision"],
        "model_config_sha256": provenance["model_config_sha256"],
        "model_weights_sha256": provenance["model_weights_sha256"],
        "tokenizer_sha256": provenance["tokenizer_sha256"],
        "model_loader_sha256": provenance["model_loader_sha256"],
        "model_inference_dtype": provenance["model_inference_dtype"],
        "observed_model_parameter_dtypes": provenance[
            "observed_model_parameter_dtypes"
        ],
        "model_inference_dtype_verification": provenance[
            "model_inference_dtype_verification"
        ],
        "model_inference_dtype_verified": provenance["model_inference_dtype_verified"],
        "activation_finiteness_check": provenance["activation_finiteness_check"],
        "model_input_format": provenance["model_input_format"],
        "tokenization_config_sha256": provenance["tokenization_config_sha256"],
        "source_manifest_sha256_by_split": provenance[
            "source_manifest_sha256_by_split"
        ],
        "cache_manifest_path": str(cache.manifest_path),
        "cache_manifest_sha256": cache.manifest_sha256,
        "cache_content_sha256": cache.content_sha256,
        "cache_layout": cache.payload["layout"],
        "cache_storage_dtype": cache.payload["storage_dtype"],
        "token_selection_method": cache.payload["token_selection"]["method"],
        "valid_token_budget_by_split": provenance["valid_token_budget_by_split"],
        "cache_row_order_sha256_by_split": {
            split: cache.payload["token_selection"]["by_split"][split][
                "row_order_sha256"
            ]
            for split in ("train", "validation", "test")
        },
        "cache_shard_sha256": [
            {
                "split": shard["split"],
                "layer": shard["layer"],
                "shard_index": shard["shard_index"],
                "input_sha256": shard["input_sha256"],
                "target_sha256": shard["target_sha256"],
            }
            for shard in cache.shards
        ],
        "selected_layers": list(cache.selected_layers),
        "feature_input_hook": provenance["feature_input_hook"],
        "feature_output_hook": provenance["feature_output_hook"],
        "decoder_window": profile["estimand"]["decoder_window"],
        "stage": args.stage,
        "method": args.method,
        "run_seed": args.seed,
        "model_seed": model_seed_value,
        "batch_stream_seed": stream_seed,
        "executed_profile_path": str(args.profile.resolve()),
        "executed_profile_sha256": profile_sha256,
        "script_sha256": sha256_file(Path(__file__)),
        "module_sha256": sha256_file(R2_ROOT / "src/revision/dictionary_controls.py"),
        "git_commit": args.git_commit,
        "git_dirty": args.git_dirty == "true",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": checkpoint_sha256,
        "result_sha256": result_sha256,
    }
    missing = set(profile["required_provenance_fields"]) - set(manifest)
    if missing:
        raise ValueError(
            f"run manifest lacks frozen provenance fields: {sorted(missing)}"
        )
    write_json(args.output_dir / "run_manifest.json", manifest)
    write_json(
        args.output_dir / "run_state.json",
        {
            "schema_version": "r2_dictionary_control_run_state_v1",
            "status": result["status"],
            "stage": args.stage,
            "method": args.method,
            "model_name": args.model_name,
            "run_seed": args.seed,
            "cache_content_sha256": cache.content_sha256,
            "profile_sha256": profile_sha256,
        },
    )


def run_production(args: argparse.Namespace) -> None:
    _require_production_args(args)
    if not args.device.startswith("cuda:"):
        raise ValueError(
            "production dictionary controls require one explicit CUDA device"
        )
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.index != args.gpu_index:
        raise ValueError("--device and --gpu-index disagree")
    torch.cuda.set_device(device)
    profile = load_production_profile(args.profile, args.profile_sha256)
    schedule = profile["compute_schedule"]
    screening_stage = args.stage == "screening"
    if screening_stage:
        if args.seed != schedule["screening"]["seed"]:
            raise ValueError("screening must use the frozen screening seed")
        if args.method in {"topk_clt", "dense_low_rank"}:
            raise ValueError("TopK and dense controls skip coefficient screening")
        if (
            args.screening_result is not None
            or args.screening_result_sha256 is not None
        ):
            raise ValueError("screening cannot consume a prior screening result")
    elif args.seed not in schedule["full"]["seeds"]:
        raise ValueError("full-run seed is outside the frozen profile")
    cache_access_splits = (
        ("train", "validation") if screening_stage else ("train", "validation", "test")
    )
    cache = load_activation_cache(
        args.cache_manifest,
        verify_hashes=True,
        access_splits=cache_access_splits,
    )
    if cache.manifest_sha256 != args.cache_sha256:
        raise ValueError("cache manifest SHA-256 mismatch")
    provenance = validate_production_cache(
        cache,
        profile,
        model_name=args.model_name,
        access_splits=cache_access_splits,
    )
    train_rows = CachedMultiLayerRows(
        cache,
        "train",
        prefix_rows=(
            schedule["screening"]["train_cache_priority_prefix_rows"]
            if screening_stage
            else None
        ),
    )
    validation_rows = CachedMultiLayerRows(cache, "validation")
    training_profile = profile["training"]
    frozen_selection = None
    screening_result_sha256 = None
    screened_methods = {"relu_l1_sae", "gated_sae"}
    if not screening_stage and args.method in screened_methods:
        if args.screening_result is None or args.screening_result_sha256 is None:
            raise ValueError(
                "full sparse runs require a hash-verified screening result"
            )
        if sha256_file(args.screening_result) != args.screening_result_sha256:
            raise ValueError("screening-result SHA-256 mismatch")
        screening_result = load_strict_json(args.screening_result)
        if (
            screening_result.get("status") != "completed_validation_screening"
            or screening_result.get("method") != args.method
            or screening_result.get("model_name") != args.model_name
            or screening_result.get("run_seed") != schedule["screening"]["seed"]
            or screening_result.get("cache_content_sha256") != cache.content_sha256
            or screening_result.get("profile_sha256") != args.profile_sha256
            or screening_result.get("test_evaluation_count") != 0
        ):
            raise ValueError("screening result does not match the requested full run")
        frozen_selection = screening_result["selected_validation_configuration"]
        matching_candidates = [
            candidate
            for candidate in _candidate_grid(profile, args.method)
            if candidate["candidate_index"] == frozen_selection.get("candidate_index")
            and candidate["l1_coefficient"] == frozen_selection.get("l1_coefficient")
            and candidate["auxiliary_coefficient"]
            == frozen_selection.get("auxiliary_coefficient")
        ]
        if (
            len(matching_candidates) != 1
            or frozen_selection.get("activation_threshold")
            not in profile["validation_only_selection"]["activation_threshold_grid"]
            or not profile["validation_only_selection"]["eligible_l0_interval"][0]
            <= frozen_selection.get("validation_l0_mean", float("inf"))
            <= profile["validation_only_selection"]["eligible_l0_interval"][1]
        ):
            raise ValueError("screening selection is outside the frozen grid/L0 gate")
        screening_result_sha256 = args.screening_result_sha256
    elif not screening_stage:
        if (
            args.screening_result is not None
            or args.screening_result_sha256 is not None
        ):
            raise ValueError("unscreened full runs do not consume a screening result")
        frozen_selection = {
            "candidate_index": 0,
            "l1_coefficient": 0.0,
            "auxiliary_coefficient": 0.0,
            "activation_threshold": 0.0,
        }

    stage_profile = schedule["screening"] if screening_stage else schedule["full"]
    checkpoint_threshold = (
        0.0 if screening_stage else frozen_selection["activation_threshold"]
    )
    config = TrainingConfig(
        seed=args.seed,
        steps=(
            stage_profile["steps_per_candidate"]
            if screening_stage
            else stage_profile["steps_per_run"]
        ),
        batch_size=training_profile["train_batch_valid_token_rows"],
        evaluation_batch_size=training_profile["evaluation_batch_valid_token_rows"],
        learning_rate=training_profile["learning_rate"],
        validation_every=(
            stage_profile["validation_every_steps"]
            if screening_stage
            else training_profile["validation_every_steps"]
        ),
        gradient_clip_norm=training_profile["gradient_clip_norm"],
        warmup_steps=(
            stage_profile["warmup_steps"]
            if screening_stage
            else training_profile["warmup_steps"]
        ),
        checkpoint_every=(
            stage_profile["checkpoint_every_steps"]
            if screening_stage
            else training_profile["checkpoint_every_steps"]
        ),
        activation_threshold=checkpoint_threshold,
        dead_frequency_threshold=0.001,
    )
    config.validate()
    model_seed_value = model_seed(args.seed, -1, args.method)
    stream_seed = batch_stream_seed(args.seed, -1, cache.content_sha256)
    started_at = datetime.now(timezone.utc).isoformat()

    candidate_count = (
        len(_candidate_grid(profile, args.method)) if screening_stage else 1
    )
    (
        planned_parameters,
        retained_checkpoint_bytes,
        peak_live_checkpoint_bytes,
    ) = _checkpoint_storage_requirement(
        profile,
        model_name=args.model_name,
        method=args.method,
        candidate_count=candidate_count,
    )
    checkpoint_storage_check = require_cache_free_space(
        args.output_dir,
        estimated_bytes=peak_live_checkpoint_bytes,
        safety_factor=profile["checkpoint_storage_planning"][
            "free_space_safety_factor_per_invocation"
        ],
    )
    checkpoint_storage_check = {
        **checkpoint_storage_check,
        "raw_trainable_parameter_count_per_candidate": planned_parameters,
        "candidate_count_in_invocation": candidate_count,
        "retained_checkpoint_bytes": retained_checkpoint_bytes,
        "atomic_rewrite_temporary_bytes": (
            peak_live_checkpoint_bytes - retained_checkpoint_bytes
        ),
        "retained_bytes_per_parameter_per_candidate": profile[
            "checkpoint_storage_planning"
        ]["retained_bytes_per_parameter_per_candidate"],
    }

    if args.output_dir.exists():
        if not args.resume:
            raise FileExistsError(
                f"run directory exists; pass --resume: {args.output_dir}"
            )
        state = load_strict_json(args.output_dir / "run_state.json")
        expected_state = {
            "schema_version": "r2_dictionary_control_run_state_v1",
            "status": "in_progress",
            "stage": args.stage,
            "method": args.method,
            "model_name": args.model_name,
            "run_seed": args.seed,
            "cache_content_sha256": cache.content_sha256,
            "profile_sha256": args.profile_sha256,
        }
        if state != expected_state:
            raise ValueError(
                "existing run-state identity does not match resume request"
            )
    else:
        if args.resume:
            raise FileNotFoundError("--resume requested but run directory is absent")
        args.output_dir.mkdir(parents=True)
        write_json(
            args.output_dir / "run_state.json",
            {
                "schema_version": "r2_dictionary_control_run_state_v1",
                "status": "in_progress",
                "stage": args.stage,
                "method": args.method,
                "model_name": args.model_name,
                "run_seed": args.seed,
                "cache_content_sha256": cache.content_sha256,
                "profile_sha256": args.profile_sha256,
            },
        )

    candidate_summaries: list[dict] = []
    candidates = (
        _candidate_grid(profile, args.method)
        if screening_stage
        else [
            {
                "candidate_index": frozen_selection["candidate_index"],
                "l1_coefficient": frozen_selection["l1_coefficient"],
                "auxiliary_coefficient": frozen_selection["auxiliary_coefficient"],
            }
        ]
    )
    candidate_cleanup_baseline: int | None = None
    for candidate in candidates:
        index = candidate["candidate_index"]
        candidate_id = (
            f"{args.stage}_{args.method}_seed_{args.seed}_candidate_{index:03d}_"
            f"profile_{args.profile_sha256[:12]}_cache_{cache.content_sha256[:12]}"
        )
        directory = args.output_dir / "candidates" / f"candidate_{index:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        summary_path = directory / "validation_result.json"
        best_path = directory / "best.pt"
        progress_path = directory / "progress.pt"
        if summary_path.is_file():
            summary = load_strict_json(summary_path)
            if summary.get("candidate_id") != candidate_id:
                raise ValueError("completed candidate identity changed")
            candidate_summaries.append(summary)
            continue

        configure_determinism(model_seed_value)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = _build_production_model(profile, cache, args.method, candidate)
        training = train_windowed_transcoder(
            model,
            train_rows,
            validation_rows,
            device=device,
            config=config,
            stream_seed=stream_seed,
            candidate_id=candidate_id,
            progress_path=progress_path,
            best_path=best_path,
            resume=progress_path.exists(),
        )
        posttraining_validation_started = time.perf_counter()
        best_checkpoint = _load_best_checkpoint_into_model(
            model,
            best_path,
            candidate_id=candidate_id,
            device=device,
        )
        if best_checkpoint.get("step") != training["best_validation_step"]:
            raise ValueError("best-checkpoint step does not match training summary")
        thresholds = (
            profile["validation_only_selection"]["activation_threshold_grid"]
            if screening_stage
            else [frozen_selection["activation_threshold"]]
        )
        validation_grid = [
            evaluate_windowed_transcoder(
                model,
                validation_rows,
                device=device,
                batch_size=config.evaluation_batch_size,
                activation_threshold=threshold,
                dead_frequency_threshold=config.dead_frequency_threshold,
                detailed=False,
            )
            for threshold in thresholds
        ]
        training["posttraining_validation_wall_time_seconds"] = (
            time.perf_counter() - posttraining_validation_started
        )
        training["wall_time_seconds"] += training[
            "posttraining_validation_wall_time_seconds"
        ]
        training["peak_accelerator_memory_allocated_bytes"] = max(
            training["peak_accelerator_memory_allocated_bytes"] or 0,
            int(torch.cuda.max_memory_allocated(device)),
        )
        training["peak_accelerator_memory_reserved_bytes"] = max(
            training["peak_accelerator_memory_reserved_bytes"] or 0,
            int(torch.cuda.max_memory_reserved(device)),
        )
        summary = {
            "schema_version": "r2_dictionary_control_candidate_validation_v1",
            "candidate_id": candidate_id,
            "stage": args.stage,
            **candidate,
            "model_seed": model_seed_value,
            "batch_stream_seed": stream_seed,
            "training": training,
            "validation_grid": validation_grid,
            "best_checkpoint": {
                "path": str(best_path),
                "sha256": sha256_file(best_path),
            },
        }
        del best_checkpoint
        del model
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        post_candidate_memory = int(torch.cuda.memory_allocated(device))
        post_candidate_reserved = int(torch.cuda.memory_reserved(device))
        if post_candidate_memory > MAX_POST_CANDIDATE_CUDA_BYTES:
            raise RuntimeError(
                "completed candidate left CUDA tensors allocated: "
                f"{post_candidate_memory} bytes"
            )
        if candidate_cleanup_baseline is None:
            candidate_cleanup_baseline = post_candidate_memory
        elif post_candidate_memory != candidate_cleanup_baseline:
            raise RuntimeError(
                "post-candidate CUDA allocation changed from the established "
                f"baseline: {candidate_cleanup_baseline} -> "
                f"{post_candidate_memory} bytes"
            )
        training["post_candidate_accelerator_memory_allocated_bytes"] = (
            post_candidate_memory
        )
        training["post_candidate_accelerator_memory_reserved_bytes"] = (
            post_candidate_reserved
        )
        training["post_candidate_accelerator_memory_limit_bytes"] = (
            MAX_POST_CANDIDATE_CUDA_BYTES
        )
        write_json(summary_path, summary)
        candidate_summaries.append(summary)

    selection_rows: list[dict] = []
    for summary in candidate_summaries:
        for validation in summary["validation_grid"]:
            selection_rows.append(
                {
                    "candidate_index": summary["candidate_index"],
                    "l1_coefficient": summary["l1_coefficient"],
                    "auxiliary_coefficient": summary["auxiliary_coefficient"],
                    "activation_threshold": validation["activation_threshold"],
                    "validation_fvu_mean": validation["fvu_mean"],
                    "validation_l0_mean": validation["l0_mean"],
                }
            )
    if screening_stage:
        low, high = profile["validation_only_selection"]["eligible_l0_interval"]
        eligible = [
            row for row in selection_rows if low <= row["validation_l0_mean"] <= high
        ]
    else:
        eligible = selection_rows
    if screening_stage and not eligible:
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "sparsity_match_failure",
            "p0_2_eligible": False,
            "stage": args.stage,
            "method": args.method,
            "model_name": args.model_name,
            "run_seed": args.seed,
            "cache_content_sha256": cache.content_sha256,
            "profile_sha256": args.profile_sha256,
            "candidate_validation": candidate_summaries,
            "selection_rows": selection_rows,
            "test_evaluation_count": 0,
            "reason": "no validation candidate achieved L0 within [115.2, 140.8]",
            "checkpoint_storage_check": checkpoint_storage_check,
        }
        _write_final_production_artifacts(
            args=args,
            profile=profile,
            profile_sha256=args.profile_sha256,
            cache=cache,
            provenance=provenance,
            result=result,
            started_at=started_at,
            selected_checkpoint=None,
            model_seed_value=model_seed_value,
            stream_seed=stream_seed,
        )
        print("screening sparsity-match failure; test split was not accessed")
        return

    selected = min(
        eligible,
        key=lambda row: (
            row["validation_fvu_mean"],
            abs(row["validation_l0_mean"] - 128.0),
            row["l1_coefficient"],
            row["activation_threshold"],
            row["auxiliary_coefficient"],
        ),
    )
    selected_summary = next(
        summary
        for summary in candidate_summaries
        if summary["candidate_index"] == selected["candidate_index"]
    )
    selected_checkpoint = Path(selected_summary["best_checkpoint"]["path"])
    if (
        sha256_file(selected_checkpoint)
        != selected_summary["best_checkpoint"]["sha256"]
    ):
        raise ValueError("selected checkpoint SHA-256 mismatch")
    if screening_stage:
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "completed_validation_screening",
            "p0_2_eligible": False,
            "stage": args.stage,
            "method": args.method,
            "model_name": args.model_name,
            "run_seed": args.seed,
            "cache_content_sha256": cache.content_sha256,
            "profile_sha256": args.profile_sha256,
            "training_cache_priority_prefix_rows": train_rows.n_rows,
            "candidate_validation": candidate_summaries,
            "selection_rows": selection_rows,
            "selected_validation_configuration": selected,
            "test_evaluation_count": 0,
            "selected_checkpoint": {
                "path": str(selected_checkpoint),
                "sha256": sha256_file(selected_checkpoint),
            },
            "planning_gpu_hours_per_candidate": schedule["planning_estimate"][
                "screening_gpu_hours_per_candidate"
            ],
            "checkpoint_storage_check": checkpoint_storage_check,
        }
        _write_final_production_artifacts(
            args=args,
            profile=profile,
            profile_sha256=args.profile_sha256,
            cache=cache,
            provenance=provenance,
            result=result,
            started_at=started_at,
            selected_checkpoint=selected_checkpoint,
            model_seed_value=model_seed_value,
            stream_seed=stream_seed,
        )
        print("wrote validation-only screening result; test split was not accessed")
        return

    selected_candidate = candidates[0]
    heldout_test_started = time.perf_counter()
    configure_determinism(model_seed_value)
    model = _build_production_model(
        profile,
        cache,
        args.method,
        selected_candidate,
    )
    _load_best_checkpoint_into_model(
        model,
        selected_checkpoint,
        candidate_id=selected_summary["candidate_id"],
        device=device,
    )
    test_rows = CachedMultiLayerRows(cache, "test")
    heldout_test = evaluate_windowed_transcoder(
        model,
        test_rows,
        device=device,
        batch_size=config.evaluation_batch_size,
        activation_threshold=selected["activation_threshold"],
        dead_frequency_threshold=config.dead_frequency_threshold,
        detailed=True,
    )
    resources = dict(selected_summary["training"])
    resources["heldout_test_wall_time_seconds"] = (
        time.perf_counter() - heldout_test_started
    )
    resources["wall_time_seconds"] += resources["heldout_test_wall_time_seconds"]
    resources["peak_accelerator_memory_allocated_bytes"] = max(
        resources["peak_accelerator_memory_allocated_bytes"] or 0,
        int(torch.cuda.max_memory_allocated(device)),
    )
    resources["peak_accelerator_memory_reserved_bytes"] = max(
        resources["peak_accelerator_memory_reserved_bytes"] or 0,
        int(torch.cuda.max_memory_reserved(device)),
    )
    if resources["raw_trainable_parameter_count"] != planned_parameters:
        raise ValueError("observed parameter count disagrees with frozen storage plan")
    resources["training_flop_proxy_total"] = (
        resources["training_flop_proxy_per_token"] * config.batch_size * config.steps
    )
    resources["checkpoint_storage_check"] = checkpoint_storage_check
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "completed_confirmatory_control",
        "p0_2_eligible": None,
        "quality_gate_requires_all_seed_aggregation": True,
        "stage": args.stage,
        "method": args.method,
        "model_name": args.model_name,
        "run_seed": args.seed,
        "objective": profile["estimand"]["objective"],
        "decoder_window": profile["estimand"]["decoder_window"],
        "cache_content_sha256": cache.content_sha256,
        "profile_sha256": args.profile_sha256,
        "dense_match_statement": (
            "rank_128_active_bottleneck_width_matched_not_raw_parameter_matched"
            if args.method == "dense_low_rank"
            else None
        ),
        "cached_topk_exact_comparison": args.method == "topk_clt",
        "candidate_validation": candidate_summaries,
        "selection_rows": selection_rows,
        "selected_validation_configuration": selected,
        "frozen_screening_configuration": frozen_selection,
        "screening_result_sha256": screening_result_sha256,
        "heldout_test": heldout_test,
        "test_evaluation_count": 1,
        "resources": resources,
        "planning_gpu_hours_per_run": schedule["planning_estimate"][
            "full_gpu_hours_per_run"
        ],
        "selected_checkpoint": {
            "path": str(selected_checkpoint),
            "sha256": sha256_file(selected_checkpoint),
        },
    }
    _write_final_production_artifacts(
        args=args,
        profile=profile,
        profile_sha256=args.profile_sha256,
        cache=cache,
        provenance=provenance,
        result=result,
        started_at=started_at,
        selected_checkpoint=selected_checkpoint,
        model_seed_value=model_seed_value,
        stream_seed=stream_seed,
    )
    print(f"wrote confirmatory control result: {args.output_dir / 'results.json'}")


def main() -> None:
    args = parse_args()
    if args.preflight_only:
        run_h200_preflight(args)
    else:
        run_production(args)


if __name__ == "__main__":
    main()
