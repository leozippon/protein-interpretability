#!/usr/bin/env python
"""Train Cross-Layer Transcoders on protein generation models.

Usage:
  # Single GPU:
  python scripts/01_train_clt.py --config configs/clt_training.yaml --gpus 0

  # Multi-GPU DDP (2 GPUs):
  python scripts/01_train_clt.py --config configs/clt_training.yaml --gpus 0,1

  # Override hyperparams:
  python scripts/01_train_clt.py --config configs/clt_training.yaml --gpus 0,1 \
      --override clt.d_clt=16384 training.total_steps=50000

  # Resume from checkpoint:
  python scripts/01_train_clt.py --config configs/clt_training.yaml --gpus 0 \
      --resume results/clt_weights/protgpt2/step_50000
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def load_config(config_path: str, overrides: list | None = None) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if overrides:
        for override in overrides:
            key, value = override.split("=", 1)
            parts = key.split(".")
            d = config
            for p in parts[:-1]:
                d = d[p]
            # Auto-parse types
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.lower() in ("null", "none"):
                value = None
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            d[parts[-1]] = value
    return config


def _run_trainer(config: dict, resume_from: str | None = None):
    """Run the trainer in-process (single GPU or one torchrun worker)."""
    from src.training.clt_trainer import CLTTrainer

    trainer = CLTTrainer(config)
    trainer.fit(resume_from=resume_from)


def main():
    parser = argparse.ArgumentParser(description="Train CLT on protein generation models")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs (e.g., '0,1' for 2-GPU DDP)")
    # Keep --gpu for backward compatibility (single GPU)
    parser.add_argument("--gpu", type=int, default=None,
                        help="(deprecated) Single GPU index. Use --gpus instead.")
    parser.add_argument("--override", nargs="*", help="Config overrides: key=value")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--verify-checkpoint", type=Path, default=None,
                        help="Verify a completed checkpoint and exit")
    parser.add_argument("--expected-step", type=int, default=None,
                        help="Required step for --verify-checkpoint")
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    if args.verify_checkpoint is not None:
        from src.training.clt_trainer import (
            _sha256_file,
            verify_checkpoint_directory,
        )

        trainer_source = Path(__file__).resolve().parents[1] / "src/training/clt_trainer.py"
        manifest = verify_checkpoint_directory(
            args.verify_checkpoint,
            expected_step=args.expected_step,
            expected_config=config,
            expected_trainer_sha256=_sha256_file(trainer_source),
            require_resumable=True,
        )
        print(
            f"Verified resumable checkpoint step={manifest['step']}: "
            f"{args.verify_checkpoint}"
        )
        return

    # If we're inside torchrun (LOCAL_RANK set), run trainer directly
    if "LOCAL_RANK" in os.environ and "RANK" in os.environ:
        _run_trainer(config, resume_from=args.resume)
        return

    # Determine GPUs
    if args.gpu is not None:
        gpu_ids = [args.gpu]
    else:
        gpu_ids = [int(x) for x in args.gpus.split(",")]

    if len(gpu_ids) == 1:
        # Single GPU — run in-process
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        _run_trainer(config, resume_from=args.resume)
    else:
        # Multi-GPU — launch via torchrun
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        # Write resolved config to temp file for torchrun workers
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f, default_flow_style=False)
            tmp_config = f.name

        cmd = [
            sys.executable,
            "-m", "torch.distributed.run",
            f"--nproc_per_node={len(gpu_ids)}",
            __file__,
            "--config", tmp_config,
            # After CUDA_VISIBLE_DEVICES remapping, GPUs are 0..N-1
            "--gpus", ",".join(map(str, range(len(gpu_ids)))),
        ]
        if args.resume:
            cmd.extend(["--resume", args.resume])

        print(f"Launching DDP training on {len(gpu_ids)} GPUs: {gpu_ids}")
        try:
            subprocess.run(cmd, check=True)
        finally:
            os.unlink(tmp_config)


if __name__ == "__main__":
    main()
