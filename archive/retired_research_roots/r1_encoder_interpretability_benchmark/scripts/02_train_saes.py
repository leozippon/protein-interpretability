#!/usr/bin/env python
"""Online SAE training on ESM-2-3B.

Usage:
  # Train single layer (auto-detect idle GPUs):
  python scripts/02_train_saes.py --config configs/sae_training.yaml --layer 23

  # Train all 9 layers sequentially:
  python scripts/02_train_saes.py --config configs/sae_training.yaml --all-layers

  # Specify GPUs:
  python scripts/02_train_saes.py --config configs/sae_training.yaml --layer 23 --gpus 0,1,2,3

  # Override hyperparams:
  python scripts/02_train_saes.py --config configs/sae_training.yaml --layer 23 \\
      --override sae.d_sae=81920 training.total_steps=200000

  # Resume from checkpoint:
  python scripts/02_train_saes.py --config configs/sae_training.yaml --layer 23 \\
      --override checkpoint.resume_from=results/sae_weights/layer_23/step_10000
"""

import argparse
import os
import subprocess
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in __import__("sys").path:
    __import__("sys").path.insert(0, _PROJECT_ROOT)
import sys

import yaml


def get_idle_gpus(memory_threshold_mb: int = 1000) -> list[int]:
    """Return GPU indices with less than threshold MB memory used.

    Uses nvidia-smi to avoid initializing CUDA context on every GPU
    (which would leave ~284MB residual memory per card).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    idle = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        idx_str, used_str = line.split(",")
        idx = int(idx_str.strip())
        used_mb = int(used_str.strip())
        if used_mb < memory_threshold_mb:
            idle.append(idx)
    return idle


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
            elif value.lower() == "null" or value.lower() == "none":
                value = None
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # keep as string
            d[parts[-1]] = value
    return config


def launch_training(config: dict, gpu_ids: list[int]):
    """Launch training: direct for 1 GPU, torchrun for multiple."""
    if len(gpu_ids) == 1:
        # Only set CUDA_VISIBLE_DEVICES if not already set externally
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        _run_trainer(config)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        # Write resolved config to temp file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f, default_flow_style=False)
            tmp_config = f.name

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={len(gpu_ids)}",
            __file__,
            "--config",
            tmp_config,
            "--layer",
            str(config["training"]["target_layer"]),
            # After CUDA_VISIBLE_DEVICES remapping, GPUs are 0..N-1
            "--gpus",
            ",".join(map(str, range(len(gpu_ids)))),
        ]

        try:
            subprocess.run(cmd, check=True)
        finally:
            os.unlink(tmp_config)


def _run_trainer(config: dict):
    """Run the trainer in-process."""
    # Import here to avoid loading torch.distributed when not needed
    from src.training.trainer import OnlineSAETrainer

    trainer = OnlineSAETrainer(config)
    try:
        trainer.fit()
    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        trainer.save_checkpoint()


def main():
    parser = argparse.ArgumentParser(description="Online SAE training on ESM-2-3B")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--layer", type=int, default=None, help="Layer to train (0-indexed)")
    parser.add_argument("--all-layers", action="store_true", help="Train all 9 layers sequentially")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs (default: auto-detect)")
    parser.add_argument("--override", nargs="*", help="Config overrides: key=value")
    args = parser.parse_args()

    config = load_config(args.config, args.override)

    # If we're inside torchrun (LOCAL_RANK set), run trainer directly
    if "LOCAL_RANK" in os.environ and "RANK" in os.environ:
        if args.layer is not None:
            config["training"]["target_layer"] = args.layer
        _run_trainer(config)
        return

    # Determine GPUs
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(",")]
    else:
        gpu_ids = get_idle_gpus()
        if not gpu_ids:
            raise RuntimeError("No idle GPUs found! Use --gpus to specify manually.")
        print(f"Auto-detected {len(gpu_ids)} idle GPUs: {gpu_ids}")

    # Determine layers
    if args.layer is not None:
        layers = [args.layer]
    elif args.all_layers:
        layers = config["training"]["layers_to_train"]
    else:
        layers = [config["training"]["target_layer"]]

    print(f"Will train {len(layers)} layer(s): {layers}")
    print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")

    for i, layer in enumerate(layers):
        print(f"\n{'=' * 60}")
        print(f"  Layer {layer} ({i + 1}/{len(layers)})")
        print(f"{'=' * 60}\n")

        config["training"]["target_layer"] = layer
        # Reset resume for each new layer unless explicitly set
        if args.layer is None and i > 0:
            config["checkpoint"]["resume_from"] = None

        launch_training(config, gpu_ids)


if __name__ == "__main__":
    main()
