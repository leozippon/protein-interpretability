#!/usr/bin/env python3
"""Replay a resumable CLT checkpoint and diagnose its first non-finite loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml


PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def finite_number(value: Any) -> float | int | None:
    number = float(value)
    return number if math.isfinite(number) else None


def main() -> None:
    args = parse_args()
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic: {args.report}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    import torch

    from src.training.clt_trainer import (
        CLTTrainer,
        _sha256_file,
        _valid_token_rows,
    )

    def write_report(payload: dict[str, Any]) -> None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_name(f".{args.report.name}.tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.replace(temporary, args.report)

    config = yaml.safe_load(args.config.read_text())
    trainer = CLTTrainer(config)
    observed: dict[str, Any] = {}
    original_make_batch = trainer._make_batch
    original_forward = trainer.clt_ddp.forward

    def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
        finite = torch.isfinite(tensor)
        finite_count = int(finite.sum().item())
        values = tensor[finite]
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "finite_count": finite_count,
            "nonfinite_count": tensor.numel() - finite_count,
            "finite_abs_max": (
                float(values.abs().max().item()) if finite_count else None
            ),
        }

    def make_batch(sequences, batch_idx, batch_size):
        input_ids, attention_mask = original_make_batch(
            sequences, batch_idx, batch_size
        )
        observed.update(
            {
                "requested_step": int(batch_idx),
                "token_batch_sha256": hashlib.sha256(
                    input_ids.detach().cpu().numpy().tobytes()
                ).hexdigest(),
                "input_ids_shape": list(input_ids.shape),
                "valid_tokens": int(attention_mask.sum().item()),
                "sequence_cursor_after_batch": int(trainer._seq_cursor),
                "epoch": int(trainer._epoch),
            }
        )
        return input_ids, attention_mask

    def checked_forward(resid_pre, mlp_out, attention_mask):
        result = original_forward(resid_pre, mlp_out, attention_mask)
        if torch.isfinite(result["loss"]):
            return result

        parameter_summaries = {
            name: summary
            for name, parameter in trainer.clt_module.named_parameters()
            if (summary := tensor_summary(parameter))["nonfinite_count"]
        }
        optimizer_summaries: dict[str, dict[str, Any]] = {}
        for name, parameter in trainer.clt_module.named_parameters():
            for state_name, value in trainer.optimizer.state.get(parameter, {}).items():
                if torch.is_tensor(value):
                    summary = tensor_summary(value)
                    if summary["nonfinite_count"]:
                        optimizer_summaries[f"{name}:{state_name}"] = summary

        with torch.no_grad():
            features = trainer.clt_module.encode(resid_pre)
            reconstructions = trainer.clt_module.decode(features)
            layer_reconstruction = []
            for layer, (target, reconstruction) in enumerate(
                zip(mlp_out, reconstructions, strict=True)
            ):
                difference = _valid_token_rows(
                    reconstruction - target, attention_mask
                )
                mse = difference.square().mean()
                layer_reconstruction.append(
                    {
                        "layer": layer,
                        "mse": finite_number(mse.item()),
                        "target": tensor_summary(
                            _valid_token_rows(target, attention_mask)
                        ),
                        "difference": tensor_summary(difference),
                    }
                )

        write_report(
            {
                "schema_version": "r2_clt_nonfinite_replay_diagnostic_v1",
                "status": "reproduced_nonfinite_loss",
                "config_path": str(args.config.resolve()),
                "config_sha256": _sha256_file(args.config),
                "resume_path": str(args.resume.resolve()),
                "checkpoint_manifest_sha256": _sha256_file(
                    args.resume / "checkpoint_manifest.json"
                ),
                "physical_gpu_index": args.gpu,
                "observed_batch": observed,
                "reported_loss": finite_number(result["loss"].item()),
                "fvu_per_layer": [
                    finite_number(value) for value in result["fvu_per_layer"]
                ],
                "input_layers": [tensor_summary(value) for value in resid_pre],
                "target_layers": [tensor_summary(value) for value in mlp_out],
                "nonfinite_parameters": parameter_summaries,
                "nonfinite_optimizer_state": optimizer_summaries,
                "layer_reconstruction": layer_reconstruction,
            },
        )
        return result

    trainer._make_batch = make_batch
    trainer.clt_ddp.forward = checked_forward

    def refuse_checkpoint(step=None):
        raise RuntimeError(
            f"diagnostic replay reached checkpoint step {step}; refusing any write"
        )

    trainer.save_checkpoint = refuse_checkpoint
    try:
        trainer.fit(resume_from=str(args.resume))
    except Exception as error:
        if not args.report.exists():
            write_report(
                {
                    "schema_version": "r2_clt_nonfinite_replay_diagnostic_v1",
                    "status": "replay_stopped_without_nonfinite_diagnostic",
                    "config_path": str(args.config.resolve()),
                    "config_sha256": _sha256_file(args.config),
                    "resume_path": str(args.resume.resolve()),
                    "checkpoint_manifest_sha256": _sha256_file(
                        args.resume / "checkpoint_manifest.json"
                    ),
                    "physical_gpu_index": args.gpu,
                    "observed_batch": observed,
                    "exception_type": type(error).__name__,
                    "exception": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise


if __name__ == "__main__":
    main()
