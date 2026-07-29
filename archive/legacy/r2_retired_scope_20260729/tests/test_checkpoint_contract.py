"""Checkpoint atomicity, integrity and retention contract."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from src.training.clt_trainer import CLTForTraining, CLTTrainer


def _trainer(root: Path) -> CLTTrainer:
    clt = CLTForTraining(n_layers=1, d_model=2, d_clt=4, k=1, window=1)
    trainer = CLTTrainer.__new__(CLTTrainer)
    trainer.clt_module = clt
    trainer.optimizer = torch.optim.Adam(clt.parameters(), lr=1e-3)
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=1)
    trainer.config = {
        "checkpoint": {"save_dir": str(root)},
        "training": {"seed": 17, "total_steps": 10},
    }
    trainer.ckpt_cfg = trainer.config["checkpoint"]
    trainer.save_dir = root
    trainer.keep_last_checkpoints = 2
    trainer.analysis_every_steps = 4
    trainer.require_checkpoint_manifest = True
    trainer.is_main = True
    trainer.world_size = 1
    trainer.seed = 17
    trainer.total_steps = 10
    trainer.device = torch.device("cpu")
    trainer.data_cfg = {"manifest_path": "immutable.jsonl"}
    trainer.model_inference_dtype_receipt = {
        "model_inference_dtype": "bfloat16",
        "observed_model_parameter_dtypes": ["bfloat16"],
        "model_inference_dtype_verification": (
            "all_floating_model_parameters_exactly_declared_before_first_activation"
        ),
        "model_inference_dtype_verified": True,
    }
    trainer._epoch = 0
    trainer._seq_order = [0, 1]
    trainer._seq_cursor = 1
    root.mkdir(exist_ok=True)
    return trainer


def test_checkpoint_hashes_and_retention(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path / "checkpoints")
    for step in (2, 4, 6, 8):
        trainer.clt_module.global_step.fill_(step)
        trainer.scheduler.last_epoch = step
        trainer.save_checkpoint(step)

    assert not (trainer.save_dir / "step_2").exists()
    assert (trainer.save_dir / "step_4" / "clt.pt").is_file()
    assert not (trainer.save_dir / "step_4" / "optimizer.pt").exists()
    for step in (6, 8):
        checkpoint = trainer.save_dir / f"step_{step}"
        assert (checkpoint / "optimizer.pt").is_file()
        manifest = json.loads(
            (checkpoint / "checkpoint_manifest.json").read_text()
        )
        assert manifest["kind"] == "resumable"
        trainer._verify_checkpoint(checkpoint)
    assert not list(trainer.save_dir.glob(".step_*.tmp-*"))


def test_checkpoint_corruption_fails_before_resume(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path / "checkpoints")
    trainer.clt_module.global_step.fill_(2)
    trainer.scheduler.last_epoch = 2
    trainer.save_checkpoint(2)
    checkpoint = trainer.save_dir / "step_2"
    with (checkpoint / "clt.pt").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="size mismatch"):
        trainer.load_checkpoint(str(checkpoint))


def test_resume_binds_config_step_and_sequence_order(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    trainer = _trainer(root)
    trainer.clt_module.global_step.fill_(2)
    trainer.scheduler.last_epoch = 2
    trainer.save_checkpoint(2)

    resumed = _trainer(root)
    assert resumed.load_checkpoint(
        str(root / "step_2"),
        expected_sequence_count=2,
    ) == 2
    assert resumed.scheduler.last_epoch == 2

    mismatched = _trainer(root)
    mismatched.config["training"]["seed"] = 29
    with pytest.raises(ValueError, match="configuration mismatch"):
        mismatched.load_checkpoint(str(root / "step_2"))

    wrong_dtype = _trainer(root)
    wrong_dtype.model_inference_dtype_receipt["model_inference_dtype"] = "float16"
    with pytest.raises(ValueError, match="invalid checkpoint trainer state"):
        wrong_dtype.load_checkpoint(str(root / "step_2"))

    manifest_path = root / "step_2" / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["step"] = 3
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="path/manifest step mismatch"):
        resumed.load_checkpoint(str(root / "step_2"))


def test_corruption_cannot_be_recertified_during_retention(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path / "checkpoints")
    for step in (2, 4, 6):
        trainer.clt_module.global_step.fill_(step)
        trainer.scheduler.last_epoch = step
        trainer.save_checkpoint(step)
    corrupted = trainer.save_dir / "step_4" / "clt.pt"
    with corrupted.open("ab") as handle:
        handle.write(b"corrupt")
    trainer.clt_module.global_step.fill_(8)
    trainer.scheduler.last_epoch = 8
    with pytest.raises(ValueError, match="size mismatch"):
        trainer.save_checkpoint(8)
    assert (trainer.save_dir / "step_4" / "optimizer.pt").is_file()


def test_checkpoint_rejects_collision_staging_and_nonfinite_state(
    tmp_path: Path,
) -> None:
    trainer = _trainer(tmp_path / "checkpoints")
    trainer.clt_module.global_step.fill_(2)
    trainer.scheduler.last_epoch = 2
    trainer.save_checkpoint(2)
    with pytest.raises(FileExistsError, match="collision"):
        trainer.save_checkpoint(2)

    trainer.clt_module.global_step.fill_(4)
    trainer.scheduler.last_epoch = 4
    (trainer.save_dir / ".step_4.tmp-999").mkdir()
    with pytest.raises(FileExistsError, match="staging"):
        trainer.save_checkpoint(4)
    (trainer.save_dir / ".step_4.tmp-999").rmdir()

    with torch.no_grad():
        trainer.clt_module.W_enc[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite tensor"):
        trainer.save_checkpoint(4)


def test_keyboard_interrupt_propagates_without_checkpoint(monkeypatch) -> None:
    import src.training.clt_trainer as trainer_module

    class InterruptedTrainer:
        def __init__(self, config):
            self.config = config

        def fit(self, resume_from=None):
            raise KeyboardInterrupt

    monkeypatch.setattr(trainer_module, "CLTTrainer", InterruptedTrainer)
    path = Path(__file__).resolve().parents[1] / "scripts/01_train_clt.py"
    spec = importlib.util.spec_from_file_location("train_clt_entrypoint", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    with pytest.raises(KeyboardInterrupt):
        module._run_trainer({}, resume_from=None)
