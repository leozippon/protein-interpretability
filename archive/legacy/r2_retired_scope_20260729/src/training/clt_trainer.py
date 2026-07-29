"""Cross-Layer Transcoder (CLT) training for protein generation models.

Trains a CLT to decompose MLP computations across all layers of a decoder-only
protein generator (ProtGPT2, ZymCTRL, etc.) into interpretable sparse features.

Training objective:
  For each layer l, reconstruct mlp_out[l] from sparse features encoded at
  architecture-specific normalized CLT inputs at preceding layers:
    features[l] = ReLU(W_enc[l] @ resid_pre[l] + b_enc[l])
    mlp_hat[l]  = sum over i<=l of (W_dec[i→l] @ features[i]) + b_dec[l]
    loss = sum_l MSE(mlp_hat[l], mlp_out[l]) + aux_loss

Reference: Lindsey et al. 2025 "On the Biology of a Large Language Model"
"""

import hashlib
import json
import os
import random
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.model_loader import (  # noqa: E402
    INFERENCE_DTYPE_VERIFICATION,
    assert_finite_captured_activations,
    inference_dtype,
    load_model,
    verify_frozen_model_inference_dtype,
)


_COHORT_FIELDS = {"id", "source", "sequence", "split", "family", "sha256"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


_CHECKPOINT_FILES = {
    "resumable": {
        "clt.pt",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.pt",
        "config.yaml",
    },
    "analysis_model_only": {"clt.pt", "config.yaml"},
}


def _resolve_model_inference_contract(
    model_config: dict,
    *,
    confirmatory: bool,
) -> tuple[str, torch.dtype]:
    """Resolve the model dtype and enforce the confirmatory BF16 contract."""

    declared = model_config.get("inference_dtype", model_config.get("dtype", "float16"))
    legacy = model_config.get("dtype")
    if legacy is not None and model_config.get("inference_dtype") not in (None, legacy):
        raise ValueError("model.dtype and model.inference_dtype disagree")
    if confirmatory and (
        declared != "bfloat16"
        or model_config.get("inference_dtype_verification")
        != INFERENCE_DTYPE_VERIFICATION
    ):
        raise ValueError(
            "confirmatory training requires declared bfloat16 model inference and "
            "exact parameter-dtype verification"
        )
    return declared, inference_dtype(declared)


def verify_checkpoint_directory(
    ckpt_dir: Path,
    *,
    expected_step: int | None = None,
    expected_config: dict | None = None,
    expected_trainer_sha256: str | None = None,
    require_resumable: bool = False,
) -> dict:
    """Verify a complete checkpoint without loading tensors onto an accelerator."""
    ckpt_dir = Path(ckpt_dir)
    manifest_path = ckpt_dir / "checkpoint_manifest.json"
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )
    required_fields = {
        "schema_version",
        "complete",
        "step",
        "kind",
        "world_size",
        "trainer_source_sha256",
        "files",
    }
    if not isinstance(manifest, dict) or set(manifest) != required_fields:
        raise ValueError(f"invalid checkpoint manifest fields: {manifest_path}")
    kind = manifest["kind"]
    if (
        manifest["schema_version"] != 2
        or manifest["complete"] is not True
        or kind not in _CHECKPOINT_FILES
        or type(manifest["step"]) is not int
        or manifest["step"] < 0
        or type(manifest["world_size"]) is not int
        or manifest["world_size"] < 1
        or not isinstance(manifest["trainer_source_sha256"], str)
        or len(manifest["trainer_source_sha256"]) != 64
        or not isinstance(manifest["files"], dict)
    ):
        raise ValueError(f"invalid checkpoint manifest: {manifest_path}")
    if require_resumable and kind != "resumable":
        raise ValueError(f"checkpoint is not resumable: {ckpt_dir}")
    try:
        directory_step = int(ckpt_dir.name.removeprefix("step_"))
    except ValueError as error:
        raise ValueError(f"invalid checkpoint directory name: {ckpt_dir}") from error
    if ckpt_dir.name != f"step_{directory_step}" or manifest["step"] != directory_step:
        raise ValueError(f"checkpoint path/manifest step mismatch: {ckpt_dir}")
    if expected_step is not None and manifest["step"] != expected_step:
        raise ValueError(
            f"checkpoint step mismatch: expected {expected_step}, got {manifest['step']}"
        )
    if (
        expected_trainer_sha256 is not None
        and manifest["trainer_source_sha256"] != expected_trainer_sha256
    ):
        raise ValueError("checkpoint trainer source SHA-256 mismatch")

    expected_files = _CHECKPOINT_FILES[kind]
    if set(manifest["files"]) != expected_files:
        raise ValueError(f"checkpoint file inventory mismatch: {ckpt_dir}")
    actual_entries = {path.name for path in ckpt_dir.iterdir()}
    if actual_entries != expected_files | {"checkpoint_manifest.json"}:
        raise ValueError(f"unexpected checkpoint directory entries: {ckpt_dir}")
    for filename, expected in manifest["files"].items():
        if (
            not isinstance(expected, dict)
            or set(expected) != {"bytes", "sha256"}
            or type(expected["bytes"]) is not int
            or expected["bytes"] < 0
            or not isinstance(expected["sha256"], str)
            or len(expected["sha256"]) != 64
        ):
            raise ValueError(f"invalid checkpoint file record: {filename}")
        path = ckpt_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing checkpoint file: {path}")
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"checkpoint size mismatch: {path}")
        if _sha256_file(path) != expected["sha256"]:
            raise ValueError(f"checkpoint SHA-256 mismatch: {path}")
    if expected_config is not None:
        saved_config = yaml.safe_load(
            (ckpt_dir / "config.yaml").read_text(encoding="utf-8")
        )
        if saved_config != expected_config:
            raise ValueError("checkpoint configuration mismatch")
    return manifest


def _load_sequence_manifest(
    path: Path,
    expected_split: str,
    max_sequences: int | None = None,
    model_input_format: str = "sequence",
) -> list[str]:
    """Load and validate one immutable cohort JSONL file."""
    sequences: list[str] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank line in {path}:{line_number}")
            record = json.loads(line, parse_constant=_reject_json_constant)
            if not isinstance(record, dict) or set(record) != _COHORT_FIELDS:
                raise ValueError(f"invalid cohort fields in {path}:{line_number}")
            if record["split"] != expected_split:
                raise ValueError(
                    f"expected split {expected_split!r}, got {record['split']!r} "
                    f"in {path}:{line_number}"
                )
            if not all(isinstance(record[key], str) and record[key] for key in _COHORT_FIELDS):
                raise ValueError(f"empty or non-string cohort field in {path}:{line_number}")
            sequence_hash = hashlib.sha256(record["sequence"].encode()).hexdigest()
            if sequence_hash != record["sha256"]:
                raise ValueError(f"sequence SHA-256 mismatch in {path}:{line_number}")
            if record["id"] in seen_ids:
                raise ValueError(f"duplicate cohort id in {path}:{line_number}")
            if sequence_hash in seen_hashes:
                raise ValueError(f"duplicate cohort sequence in {path}:{line_number}")
            seen_ids.add(record["id"])
            seen_hashes.add(sequence_hash)
            if model_input_format == "sequence":
                model_input = record["sequence"]
            elif model_input_format == "zymctrl_ec":
                model_input = (
                    f"{record['family']}<sep><start>{record['sequence']}<end>"
                )
            else:
                raise ValueError(f"unknown model-input format: {model_input_format}")
            sequences.append(model_input)
            if max_sequences is not None and len(sequences) == max_sequences:
                break
    if not sequences:
        raise ValueError(f"no sequences loaded from {path}")
    if max_sequences is not None and len(sequences) != max_sequences:
        raise ValueError(
            f"requested {max_sequences} sequences but {path} contains only {len(sequences)}"
        )
    return sequences


def _validate_attention_mask(
    attention_mask: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    """Return a boolean token mask, using ``None`` for the all-valid case."""
    if attention_mask is None:
        return None
    if attention_mask.shape != reference.shape[:2]:
        raise ValueError(
            "attention_mask must match the activation batch/sequence shape; "
            f"got {attention_mask.shape} and {reference.shape[:2]}"
        )
    mask = attention_mask.to(device=reference.device, dtype=torch.bool)
    if not mask.any():
        raise ValueError("attention_mask must contain at least one valid token")
    return None if mask.all() else mask


def _valid_token_rows(
    tensor: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Select valid token rows while preserving the all-valid code path."""
    return tensor if attention_mask is None else tensor[attention_mask]


class CLTForTraining(nn.Module):
    """Cross-Layer Transcoder optimized for training.

    Uses a windowed decoder to keep memory tractable: each feature at layer l
    can write to layers l through l+window-1 (not all subsequent layers).
    This captures most cross-layer information flow while reducing decoder
    parameters from O(n_layers^2) to O(n_layers * window).

    Architecture per layer l:
      encode:  features[l] = TopK(ReLU(W_enc[l] @ resid_pre[l] + b_enc[l]))
      decode:  for each target layer t in [l, l+window):
                 mlp_hat[t] += W_dec[l,t-l] @ features[l]
               mlp_hat[t] += b_dec[t]
    """

    def __init__(self, n_layers: int, d_model: int, d_clt: int, k: int = 64,
                 window: int = 8):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_clt = d_clt
        self.k = k
        self.window = min(window, n_layers)  # cap at n_layers

        # Encoder: one per layer; resid_pre is the legacy CLT-input field name.
        self.W_enc = nn.Parameter(torch.empty(n_layers, d_clt, d_model))
        self.b_enc = nn.Parameter(torch.zeros(n_layers, d_clt))

        # Windowed decoder: W_dec[l] has shape (d_clt, min(window, n_layers-l), d_model)
        self.W_dec = nn.ParameterList([
            nn.Parameter(torch.empty(d_clt, min(self.window, n_layers - l), d_model))
            for l in range(n_layers)
        ])

        # Decoder bias (per target layer)
        self.b_dec = nn.Parameter(torch.zeros(n_layers, d_model))

        # Track feature firing for dead feature detection
        self.register_buffer(
            "feature_last_fired",
            torch.zeros(n_layers, d_clt, dtype=torch.long),
        )
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))
        self.dead_feature_threshold = 10000

        self._init_weights()

    def _init_weights(self):
        """Initialize with small random weights."""
        for l in range(self.n_layers):
            nn.init.kaiming_uniform_(self.W_enc[l])
            self.W_enc.data[l] *= 0.1
            n_targets = min(self.window, self.n_layers - l)
            for t in range(n_targets):
                nn.init.kaiming_uniform_(self.W_dec[l][:, t, :])
                self.W_dec[l].data[:, t, :] *= 0.1

    def encode(self, resid_pre: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode architecture-specific CLT-input activations to sparse features.

        Args:
            resid_pre: List of (batch, seq, d_model) tensors, one per layer

        Returns:
            List of (batch, seq, d_clt) sparse feature tensors, one per layer
        """
        features = []
        for l in range(self.n_layers):
            # (batch, seq, d_model) @ (d_model, d_clt) -> (batch, seq, d_clt)
            pre_act = torch.einsum("bsd,fd->bsf", resid_pre[l], self.W_enc[l]) + self.b_enc[l]
            pre_act = F.relu(pre_act)

            # TopK sparsity per token
            topk_vals, topk_idx = pre_act.topk(self.k, dim=-1)
            sparse = torch.zeros_like(pre_act)
            sparse.scatter_(-1, topk_idx, topk_vals)
            features.append(sparse)

        return features

    def decode(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decode sparse features to MLP output reconstructions.

        For each target layer t, sum contributions from source layers within
        the decode window: layers max(0, t-window+1) through t.

        Args:
            features: List of (batch, seq, d_clt) tensors

        Returns:
            List of (batch, seq, d_model) reconstructed MLP outputs
        """
        batch, seq = features[0].shape[:2]
        mlp_hat = [self.b_dec[t].expand(batch, seq, -1).clone()
                   for t in range(self.n_layers)]

        for l in range(self.n_layers):
            n_targets = min(self.window, self.n_layers - l)
            # features[l]: (batch, seq, d_clt)
            # W_dec[l]: (d_clt, n_targets, d_model)
            contrib = torch.einsum("bsf,ftd->bstd", features[l], self.W_dec[l])
            for t_offset in range(n_targets):
                t = l + t_offset
                mlp_hat[t] = mlp_hat[t] + contrib[:, :, t_offset, :]

        return mlp_hat

    def forward(
        self,
        resid_pre: list[torch.Tensor],
        mlp_out: list[torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> dict:
        """Full forward pass with loss computation.

        Args:
            resid_pre: Per-layer MLP inputs (batch, seq, d_model)
            mlp_out: Per-layer MLP outputs (batch, seq, d_model)
            attention_mask: Valid-token mask with shape (batch, seq). Padding
                rows are excluded from loss and all reported/tracked metrics.

        Returns:
            Dict with loss, metrics, and reconstructions
        """
        attention_mask = _validate_attention_mask(attention_mask, resid_pre[0])
        features = self.encode(resid_pre)
        mlp_hat = self.decode(features)

        # Reconstruction loss: MSE per layer, averaged
        recon_losses = []
        fvu_per_layer = []
        for t in range(self.n_layers):
            diff = _valid_token_rows(mlp_hat[t] - mlp_out[t], attention_mask)
            target = _valid_token_rows(mlp_out[t], attention_mask)
            mse = (diff ** 2).mean()
            recon_losses.append(mse)

            # FVU = Var(residual) / Var(target)
            var_target = target.var()
            fvu = mse / (var_target + 1e-8)
            fvu_per_layer.append(fvu.item())

        total_loss = sum(recon_losses) / self.n_layers

        # Sparsity stats
        l0_per_layer = []
        dead_per_layer = []
        for l in range(self.n_layers):
            active = (features[l] > 0).float()
            active = _valid_token_rows(active, attention_mask)
            l0_per_layer.append(active.sum(-1).mean().item())

            # Track feature firing
            fired = active.bool().reshape(-1, self.d_clt).any(dim=0)
            self.feature_last_fired[l][fired.bool()] = self.global_step
            dead_frac = (
                self.global_step - self.feature_last_fired[l] > self.dead_feature_threshold
            ).float().mean()
            dead_per_layer.append(dead_frac.item())

        self.global_step += 1

        return {
            "loss": total_loss,
            "fvu_mean": sum(fvu_per_layer) / len(fvu_per_layer),
            "fvu_per_layer": fvu_per_layer,
            "l0_mean": sum(l0_per_layer) / len(l0_per_layer),
            "dead_mean": sum(dead_per_layer) / len(dead_per_layer),
        }


class CLTTrainer:
    """Online CLT trainer with DDP support.

    Each GPU holds:
      - Frozen protein model in the explicitly declared inference dtype
      - CLTForTraining in fp32 (DDP-wrapped for gradient sync)
      - Optimizer states

    DDP synchronizes CLT gradients across GPUs. Each rank processes
    different data batches; dead feature tracking is synced via all_reduce.
    """

    def __init__(self, config: dict):
        self.config = config
        self.rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_main = self.rank == 0
        self.device = torch.device(f"cuda:{self.rank}")
        self.seed = int(config["training"].get("seed", 0))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        if self.world_size > 1 and not dist.is_initialized():
            torch.cuda.set_device(self.rank)
            dist.init_process_group(backend="nccl", timeout=timedelta(hours=12))

        # Load and verify the frozen protein model (each rank holds a copy).
        model_cfg = config["model"]
        confirmatory_value = os.environ.get("CONFIRMATORY", "0")
        if confirmatory_value not in {"0", "1"}:
            raise ValueError("CONFIRMATORY must be 0 or 1")
        confirmatory = confirmatory_value == "1"
        declared_dtype, torch_dtype = _resolve_model_inference_contract(
            model_cfg,
            confirmatory=confirmatory,
        )
        self.protein_model = load_model(
            model_cfg["name"],
            device=self.device,
            dtype=torch_dtype,
        )
        self.protein_model.model.eval()
        for p in self.protein_model.model.parameters():
            p.requires_grad_(False)
        self.model_inference_dtype_receipt = verify_frozen_model_inference_dtype(
            self.protein_model,
            declared_dtype,
        )

        # Build CLT
        clt_cfg = config["clt"]
        self.clt = CLTForTraining(
            n_layers=self.protein_model.n_layers,
            d_model=self.protein_model.d_model,
            d_clt=clt_cfg["d_clt"],
            k=clt_cfg["k"],
            window=clt_cfg.get("window", 8),
        ).to(self.device).float()

        # DDP wrapper for gradient sync
        if self.world_size > 1:
            self.clt_ddp = DDP(self.clt, device_ids=[self.rank])
        else:
            self.clt_ddp = self.clt
        # Direct access to unwrapped module for attributes and resampling
        self.clt_module = self.clt_ddp.module if isinstance(self.clt_ddp, DDP) else self.clt_ddp

        # Resampling config
        self.resample_every = clt_cfg.get("resample_every", 5000)
        self.dead_threshold = clt_cfg.get("dead_feature_threshold", 10000)
        self.max_resample_fraction = float(
            clt_cfg.get("max_resample_fraction", 1.0)
        )
        if not 0.0 < self.max_resample_fraction <= 1.0:
            raise ValueError("clt.max_resample_fraction must lie in (0, 1]")
        self.clt_module.dead_feature_threshold = self.dead_threshold

        # Optimizer (operates on unwrapped parameters)
        train_cfg = config["training"]
        self.optimizer = torch.optim.Adam(
            self.clt_module.parameters(),
            lr=train_cfg["lr"],
            betas=(0.9, 0.999),
            foreach=False,
        )
        self.total_steps = train_cfg["total_steps"]
        if self.total_steps < 4:
            raise ValueError("training.total_steps must be at least 4 for OneCycleLR")
        self.grad_clip = train_cfg.get("grad_clip_norm", 1.0)

        # Warmup + cosine schedule
        warmup = train_cfg.get("lr_warmup_steps", 1000)
        # OneCycleLR requires pct_start in (0, 1]. Cap warmup for short smoke tests.
        pct_start = warmup / self.total_steps
        min_pct = 1 / self.total_steps
        max_pct = 1.0 - min_pct if self.total_steps > 1 else 0.5
        pct_start = min(max(pct_start, min_pct), max_pct)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=train_cfg["lr"],
            total_steps=self.total_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=100.0,
        )

        # Data
        self.data_cfg = config["data"]
        self.tokenizer = self.protein_model.tokenizer

        # Checkpoint
        self.ckpt_cfg = config["checkpoint"]
        self.save_dir = Path(self.ckpt_cfg["save_dir"])
        self.keep_last_checkpoints = int(self.ckpt_cfg.get("keep_last", 2))
        self.analysis_every_steps = int(
            self.ckpt_cfg.get("analysis_every_steps", 0)
        )
        self.require_checkpoint_manifest = bool(
            self.ckpt_cfg.get("require_sha256_manifest", False)
        )
        required_prefix = self.ckpt_cfg.get("required_path_prefix")
        if self.keep_last_checkpoints < 1:
            raise ValueError("checkpoint.keep_last must be at least 1")
        if self.analysis_every_steps < 0:
            raise ValueError("checkpoint.analysis_every_steps must be non-negative")
        if required_prefix and not str(self.save_dir.resolve()).startswith(
            str(Path(required_prefix).resolve()) + os.sep
        ):
            raise ValueError(
                f"checkpoint.save_dir must be under {required_prefix}"
            )
        if self.require_checkpoint_manifest and self.world_size != 1:
            raise ValueError("confirmatory checkpointing requires one GPU per seed")
        if self.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            stale = sorted(self.save_dir.glob(".step_*.tmp-*"))
            if stale:
                raise FileExistsError(f"stale checkpoint staging directories: {stale}")

        # Logging
        self.log_cfg = config.get("logging", {})
        self.log_every = self.log_cfg.get("log_every_steps", 50)

        # WandB (rank 0 only)
        self.wandb_run = None
        if self.is_main and self.log_cfg.get("wandb_project"):
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.log_cfg["wandb_project"],
                    entity=self.log_cfg.get("wandb_entity"),
                    config=config,
                    name=f"clt_{config['model']['name']}",
                )
            except Exception as e:
                print(f"WandB init failed: {e}")

    def _load_sequences(self) -> list[str]:
        """Load the frozen training manifest, or a legacy FASTA when configured."""
        manifest_path = self.data_cfg.get("manifest_path")
        if manifest_path:
            path = Path(manifest_path)
            split = self.data_cfg.get("split")
            if not split:
                raise ValueError("data.split is required with data.manifest_path")
            digest = _sha256_file(path)
            expected_digest = self.data_cfg.get("manifest_sha256")
            if expected_digest is not None and digest != expected_digest:
                raise ValueError(
                    f"manifest SHA-256 mismatch: expected {expected_digest}, got {digest}"
                )
            self.data_cfg["manifest_sha256"] = digest
            num_sequences = self.data_cfg.get("num_sequences")
            if self.is_main:
                print(f"Loading immutable {split} cohort from {path} ({digest})...")
            sequences = _load_sequence_manifest(
                path,
                split,
                num_sequences,
                self.data_cfg.get("model_input_format", "sequence"),
            )
            if self.is_main:
                print(f"  Loaded {len(sequences)} hash-verified sequences")
            return sequences

        fasta_path = self.data_cfg["fasta_path"]
        max_seq_len = self.data_cfg.get("max_seq_len", 512)
        num_sequences = self.data_cfg.get("num_sequences", 100000)

        if self.is_main:
            print(f"Loading sequences from {fasta_path} (max_len={max_seq_len})...")
        sequences = []
        current_seq = []

        with open(fasta_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_seq:
                        seq = "".join(current_seq)
                        if len(seq) <= max_seq_len:
                            sequences.append(seq)
                            if len(sequences) >= num_sequences:
                                break
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_seq and len(sequences) < num_sequences:
                seq = "".join(current_seq)
                if len(seq) <= max_seq_len:
                    sequences.append(seq)

        if self.is_main:
            print(f"  Loaded {len(sequences)} sequences")
        if not sequences:
            raise ValueError(f"no eligible sequences loaded from {fasta_path}")
        return sequences

    def _make_batch(
        self,
        sequences: list[str],
        batch_idx: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch of sequences with rank-aware shuffled ordering.

        Each rank uses a different random seed so GPUs see different data,
        effectively multiplying the batch size by world_size.
        """
        if not hasattr(self, "_seq_order") or self._seq_order is None:
            self._epoch = 0
            self._reshuffle(len(sequences))

        # Collect batch_size indices, reshuffling at epoch boundary
        indices = []
        while len(indices) < batch_size:
            remaining = batch_size - len(indices)
            if self._seq_cursor + remaining <= len(self._seq_order):
                indices.extend(self._seq_order[self._seq_cursor:self._seq_cursor + remaining])
                self._seq_cursor += remaining
            else:
                indices.extend(self._seq_order[self._seq_cursor:])
                self._epoch += 1
                self._reshuffle(len(sequences))

        batch_seqs = [sequences[i] for i in indices]
        tokens = self.tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.data_cfg.get("max_seq_len", 512),
        )
        return (
            tokens["input_ids"].to(self.device),
            tokens["attention_mask"].to(self.device),
        )

    def _reshuffle(self, n: int):
        """Make a deterministic global permutation, then give each rank a shard."""
        seed = getattr(self, "seed", 0) + self._epoch
        g = torch.Generator().manual_seed(seed)
        global_order = torch.randperm(n, generator=g).tolist()
        self._seq_order = global_order[self.rank::self.world_size]
        if not self._seq_order:
            raise ValueError(
                f"rank {self.rank} received no sequences from a cohort of size {n}"
            )
        self._seq_cursor = 0

    def resample_dead_features(
        self,
        resid_pre: list[torch.Tensor],
        mlp_out: list[torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> int:
        """Resample dead features per layer from high-loss directions.

        For each layer, identifies features that haven't fired in `dead_threshold`
        steps and reinitializes them from the reconstruction residual of that layer.

        In DDP mode, rank 0 computes resampling directions and broadcasts them
        to all ranks so weights stay synchronized.

        Returns:
            Total number of features resampled across all layers.
        """
        clt = self.clt_module
        attention_mask = _validate_attention_mask(attention_mask, resid_pre[0])
        total_resampled = 0
        step = clt.global_step.item()

        with torch.no_grad():
            features = clt.encode(resid_pre)
            mlp_hat = clt.decode(features)

        for l in range(clt.n_layers):
            # dead_mask is identical across ranks (feature_last_fired is synced)
            dead_mask = (step - clt.feature_last_fired[l]) > self.dead_threshold
            n_dead = int(dead_mask.sum())
            if n_dead == 0:
                continue

            dead_idx = dead_mask.nonzero(as_tuple=True)[0]
            maximum = max(1, int(clt.d_clt * self.max_resample_fraction))
            if n_dead > maximum:
                last_fired = clt.feature_last_fired[l, dead_idx]
                order = torch.argsort(last_fired, stable=True)
                dead_idx = dead_idx[order[:maximum]]
                n_dead = maximum

            # Rank 0 computes resampling directions; others receive via broadcast
            directions = torch.zeros(n_dead, clt.d_model, device=self.device)
            avg_norm = torch.tensor(1.0, device=self.device)

            if self.is_main:
                residual = mlp_out[l] - mlp_hat[l]
                valid_residual = _valid_token_rows(residual, attention_mask).reshape(
                    -1, clt.d_model
                )
                flat_loss = valid_residual.pow(2).sum(dim=-1)
                if not torch.isfinite(flat_loss).all():
                    raise FloatingPointError(
                        f"non-finite resampling loss at layer {l}, step {step}"
                    )

                if flat_loss.sum() >= 1e-8:
                    probs = flat_loss / flat_loss.sum()
                    n_samples = min(n_dead, len(probs))
                    sample_idx = torch.multinomial(probs, n_samples,
                                                   replacement=(n_dead > len(probs)))

                    sampled_dirs = valid_residual[sample_idx]
                    norms = sampled_dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    sampled_dirs = sampled_dirs / norms

                    if n_dead > n_samples:
                        repeats = (n_dead + n_samples - 1) // n_samples
                        sampled_dirs = sampled_dirs.repeat(repeats, 1)[:n_dead]
                    directions.copy_(sampled_dirs)

                alive_mask = ~dead_mask
                if alive_mask.any():
                    avg_norm.fill_(clt.W_enc[l][alive_mask].norm(dim=1).mean().item())

            # Broadcast to all ranks
            if self.world_size > 1:
                dist.broadcast(directions, src=0)
                dist.broadcast(avg_norm, src=0)

            # Apply resampled weights on all ranks
            with torch.no_grad():
                clt.W_enc.data[l, dead_idx] = directions * avg_norm * 0.2
                clt.b_enc.data[l, dead_idx] = 0.0
                clt.W_dec[l].data[dead_idx, 0, :] = directions
                n_targets = clt.W_dec[l].shape[1]
                if n_targets > 1:
                    clt.W_dec[l].data[dead_idx, 1:, :] = 0.0

            self._reset_optimizer_state(l, dead_idx)
            clt.feature_last_fired[l, dead_idx] = step
            total_resampled += n_dead

        return total_resampled

    def _reset_optimizer_state(self, layer: int, feature_indices: torch.Tensor):
        """Reset Adam momentum/variance for resampled features at a specific layer."""
        clt = self.clt_module
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p not in self.optimizer.state:
                    continue
                state = self.optimizer.state[p]
                for key in ("exp_avg", "exp_avg_sq"):
                    if key not in state:
                        continue
                    buf = state[key]
                    # W_enc: shape (n_layers, d_clt, d_model)
                    if buf.shape == clt.W_enc.shape:
                        buf[layer, feature_indices] = 0
                    # b_enc: shape (n_layers, d_clt)
                    elif buf.shape == clt.b_enc.shape:
                        buf[layer, feature_indices] = 0
                    # W_dec[l]: shape (d_clt, n_targets, d_model) — ParameterList
                    # These are separate parameters, matched by data_ptr
                    elif p.data_ptr() == clt.W_dec[layer].data_ptr():
                        buf[feature_indices] = 0

    def load_checkpoint(
        self,
        path: str,
        *,
        expected_sequence_count: int | None = None,
    ):
        """Resume training from a checkpoint (all ranks load the same state)."""
        ckpt_dir = Path(path)
        manifest_path = ckpt_dir / "checkpoint_manifest.json"
        manifest = None
        if manifest_path.exists():
            manifest = verify_checkpoint_directory(
                ckpt_dir,
                expected_config=self.config,
                expected_trainer_sha256=_sha256_file(Path(__file__)),
                require_resumable=True,
            )
            if manifest["world_size"] != self.world_size:
                raise ValueError("checkpoint world-size mismatch")
        elif self.require_checkpoint_manifest:
            raise FileNotFoundError(
                f"checkpoint_manifest.json is required in {ckpt_dir}"
            )
        state = torch.load(ckpt_dir / "clt.pt", map_location=self.device)
        self.clt_module.load_state_dict(state)
        step = int(self.clt_module.global_step.item())
        if manifest is not None and step != manifest["step"]:
            raise ValueError("checkpoint model/manifest step mismatch")
        self.optimizer.load_state_dict(
            torch.load(ckpt_dir / "optimizer.pt", map_location=self.device)
        )
        scheduler_state = torch.load(
            ckpt_dir / "scheduler.pt",
            map_location=self.device,
            weights_only=False,
        )
        if manifest is not None and scheduler_state.get("last_epoch") != step:
            raise ValueError("checkpoint scheduler/model step mismatch")
        self.scheduler.load_state_dict(scheduler_state)
        trainer_state_path = ckpt_dir / "trainer_state.pt"
        if trainer_state_path.exists():
            trainer_state = torch.load(
                trainer_state_path,
                map_location="cpu",
                weights_only=False,
            )
            if manifest is not None:
                expected_fields = {
                    "schema_version",
                    "step",
                    "world_size",
                    "seed",
                    "python_rng_state",
                    "numpy_rng_state",
                    "torch_rng_state",
                    "cuda_rng_state",
                    "epoch",
                    "sequence_order",
                    "sequence_cursor",
                    "model_inference_dtype_receipt",
                }
                if set(trainer_state) != expected_fields:
                    raise ValueError("checkpoint trainer-state fields mismatch")
                sequence_order = trainer_state["sequence_order"]
                sequence_cursor = trainer_state["sequence_cursor"]
                if (
                    trainer_state["schema_version"] != 2
                    or trainer_state["step"] != step
                    or trainer_state["world_size"] != self.world_size
                    or trainer_state["seed"] != self.seed
                    or type(trainer_state["epoch"]) is not int
                    or trainer_state["epoch"] < 0
                    or not isinstance(sequence_order, list)
                    or type(sequence_cursor) is not int
                    or not 0 <= sequence_cursor <= len(sequence_order)
                    or trainer_state["model_inference_dtype_receipt"]
                    != self.model_inference_dtype_receipt
                ):
                    raise ValueError("invalid checkpoint trainer state")
                if expected_sequence_count is not None and (
                    len(sequence_order) != expected_sequence_count
                    or sorted(sequence_order) != list(range(expected_sequence_count))
                ):
                    raise ValueError("checkpoint sequence order/cohort mismatch")
            random.setstate(trainer_state["python_rng_state"])
            np.random.set_state(trainer_state["numpy_rng_state"])
            torch.set_rng_state(trainer_state["torch_rng_state"])
            if torch.cuda.is_available() and trainer_state["cuda_rng_state"] is not None:
                torch.cuda.set_rng_state_all(trainer_state["cuda_rng_state"])
            self._epoch = trainer_state["epoch"]
            self._seq_order = trainer_state["sequence_order"]
            self._seq_cursor = trainer_state["sequence_cursor"]
        elif self.data_cfg.get("manifest_path"):
            raise FileNotFoundError(
                f"confirmatory resume requires trainer_state.pt in {ckpt_dir}"
            )
        if step > self.total_steps:
            raise ValueError("checkpoint step exceeds configured total steps")
        if self.is_main:
            print(f"Resumed from {path} at step {step}")
        return step

    def _sync_feature_tracking(self):
        """Sync dead feature tracking across ranks.

        After each forward pass, each rank has updated feature_last_fired
        for features that fired on its local data. all_reduce(MAX) ensures
        a feature is marked as alive if it fired on ANY rank.
        """
        if self.world_size > 1:
            dist.all_reduce(self.clt_module.feature_last_fired, op=dist.ReduceOp.MAX)

    def fit(self, resume_from: str | None = None):
        """Main training loop with DDP support.

        Args:
            resume_from: Path to checkpoint directory to resume from.
        """
        clt = self.clt_module
        sequences = self._load_sequences()
        batch_size = self.config["training"]["batch_size"]

        start_step = 0
        if resume_from:
            start_step = self.load_checkpoint(
                resume_from,
                expected_sequence_count=len(sequences),
            )

        if self.is_main:
            print("\nStarting CLT training:")
            print(f"  Model: {self.config['model']['name']}")
            print(
                "  Frozen-model inference dtype: "
                f"{self.model_inference_dtype_receipt['model_inference_dtype']} "
                "(verified)"
            )
            print(f"  d_model={clt.d_model}, d_clt={clt.d_clt}, k={clt.k}")
            print(f"  n_layers={clt.n_layers}")
            print(f"  Steps: {start_step}/{self.total_steps}")
            print(f"  Batch size: {batch_size} (x{self.world_size} GPUs = {batch_size * self.world_size} effective)")
            print(f"  Resample every: {self.resample_every} steps")
            print(f"  Dead threshold: {self.dead_threshold} steps")
            print(f"  Maximum resampled fraction/event: {self.max_resample_fraction:.3f}")
            print(f"  GPUs: {self.world_size}")
            print()

        step = start_step
        start_time = time.time()

        while step < self.total_steps:
            input_ids, attention_mask = self._make_batch(sequences, step, batch_size)

            # Extract activations from frozen protein model
            with torch.no_grad():
                cache = self.protein_model.get_activations(input_ids, attention_mask)
            assert_finite_captured_activations(cache)

            # CLT forward via DDP wrapper (syncs gradients in backward)
            resid_pre = [x.float() for x in cache.resid_pre]
            mlp_out = [x.float() for x in cache.mlp_out]

            result = self.clt_ddp(resid_pre, mlp_out, attention_mask)
            if not torch.isfinite(result["loss"]):
                token_digest = hashlib.sha256(
                    input_ids.detach().cpu().numpy().tobytes()
                ).hexdigest()
                raise FloatingPointError(
                    f"non-finite loss at step {step} for token batch {token_digest}"
                )

            # Backward (DDP averages gradients across ranks)
            self.optimizer.zero_grad()
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                clt.parameters(),
                self.grad_clip,
                error_if_nonfinite=True,
            )
            self.optimizer.step()
            self.scheduler.step()

            # Sync dead feature tracking across ranks
            self._sync_feature_tracking()

            step += 1

            # Dead feature resampling (all ranks participate)
            if (self.resample_every > 0 and step > 0
                    and step % self.resample_every == 0):
                n_resampled = self.resample_dead_features(
                    resid_pre,
                    mlp_out,
                    attention_mask,
                )
                if self.is_main and n_resampled > 0:
                    pct = n_resampled / (clt.n_layers * clt.d_clt) * 100
                    print(f"  [Step {step}] Resampled {n_resampled} dead features ({pct:.1f}%)")

            # Logging (rank 0 only)
            if self.is_main and step % self.log_every == 0:
                elapsed = time.time() - start_time
                steps_done = step - start_step
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"Step {step:>7d}/{self.total_steps} | "
                    f"loss={result['loss'].item():.4f} | "
                    f"FVU={result['fvu_mean']:.4f} | "
                    f"L0={result['l0_mean']:.1f} | "
                    f"dead={result['dead_mean']:.3f} | "
                    f"lr={lr:.2e} | "
                    f"{elapsed/max(steps_done,1):.2f}s/step"
                )

                if self.wandb_run:
                    import wandb
                    log = {
                        "loss": result["loss"].item(),
                        "fvu_mean": result["fvu_mean"],
                        "l0_mean": result["l0_mean"],
                        "dead_mean": result["dead_mean"],
                        "lr": lr,
                    }
                    for l, fvu in enumerate(result["fvu_per_layer"]):
                        log[f"fvu/layer_{l}"] = fvu
                    wandb.log(log, step=step)

            # Checkpoint (rank 0 saves, others wait)
            if step % self.ckpt_cfg.get("save_every_steps", 10000) == 0:
                self.save_checkpoint(step)
                if self.world_size > 1:
                    dist.barrier(device_ids=[self.rank])

        if step % self.ckpt_cfg.get("save_every_steps", 10000) != 0:
            self.save_checkpoint(step)

        if self.is_main:
            print(f"\nTraining complete! Final step: {step}")
            if self.wandb_run:
                self.wandb_run.finish()

        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

    def _checkpoint_manifest(self, ckpt_dir: Path, step: int, kind: str) -> dict:
        filenames = sorted(_CHECKPOINT_FILES[kind])
        files = {}
        for filename in filenames:
            path = ckpt_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"missing checkpoint file: {path}")
            files[filename] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        return {
            "schema_version": 2,
            "complete": True,
            "step": int(step),
            "kind": kind,
            "world_size": self.world_size,
            "trainer_source_sha256": _sha256_file(Path(__file__)),
            "files": files,
        }

    @staticmethod
    def _write_checkpoint_manifest(ckpt_dir: Path, manifest: dict) -> None:
        path = ckpt_dir / "checkpoint_manifest.json"
        temporary = ckpt_dir / ".checkpoint_manifest.json.tmp"
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _verify_checkpoint(self, ckpt_dir: Path) -> dict:
        return verify_checkpoint_directory(
            ckpt_dir,
            expected_config=self.config,
            expected_trainer_sha256=_sha256_file(Path(__file__)),
        )

    def _prune_checkpoints(self) -> None:
        checkpoints = []
        for path in self.save_dir.glob("step_*"):
            try:
                step = int(path.name.removeprefix("step_"))
            except ValueError:
                continue
            if path.is_dir():
                checkpoints.append((step, path))
        checkpoints.sort()
        resumable_steps = {
            step for step, _ in checkpoints[-self.keep_last_checkpoints :]
        }
        for step, path in checkpoints:
            if step in resumable_steps:
                continue
            existing_manifest = json.loads(
                (path / "checkpoint_manifest.json").read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
            )
            if existing_manifest.get("kind") == "analysis_model_only":
                continue
            manifest = self._verify_checkpoint(path)
            keep_for_analysis = (
                self.analysis_every_steps > 0
                and step % self.analysis_every_steps == 0
            )
            if not keep_for_analysis:
                shutil.rmtree(path)
                continue
            for filename in ("optimizer.pt", "scheduler.pt", "trainer_state.pt"):
                state_path = path / filename
                if state_path.exists():
                    state_path.unlink()
            manifest = {
                **manifest,
                "kind": "analysis_model_only",
                "files": {
                    filename: manifest["files"][filename]
                    for filename in sorted(_CHECKPOINT_FILES["analysis_model_only"])
                },
            }
            self._write_checkpoint_manifest(path, manifest)

    @staticmethod
    def _assert_finite_tensors(label: str, value) -> None:
        if torch.is_tensor(value):
            if (value.is_floating_point() or value.is_complex()) and not bool(
                torch.isfinite(value).all().item()
            ):
                raise FloatingPointError(f"non-finite tensor before checkpoint: {label}")
            return
        if isinstance(value, dict):
            for key, nested in value.items():
                CLTTrainer._assert_finite_tensors(f"{label}.{key}", nested)
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                CLTTrainer._assert_finite_tensors(f"{label}[{index}]", nested)

    def save_checkpoint(self, step: int | None = None):
        """Stage, hash and publish a checkpoint, then apply retention policy."""
        if not self.is_main:
            return
        if step is None:
            step = self.clt_module.global_step.item()
        step = int(step)
        if int(self.clt_module.global_step.item()) != step:
            raise ValueError("checkpoint step does not equal model global step")
        if self.scheduler.state_dict().get("last_epoch") != step:
            raise ValueError("checkpoint step does not equal scheduler step")
        self._assert_finite_tensors("model", self.clt_module.state_dict())
        self._assert_finite_tensors("optimizer", self.optimizer.state_dict())
        ckpt_dir = self.save_dir / f"step_{step}"
        if ckpt_dir.exists():
            raise FileExistsError(f"refusing checkpoint collision: {ckpt_dir}")
        stale = sorted(self.save_dir.glob(".step_*.tmp-*"))
        if stale:
            raise FileExistsError(f"stale checkpoint staging directories: {stale}")
        temporary = self.save_dir / f".step_{step}.tmp-{os.getpid()}"
        temporary.mkdir(parents=True)

        torch.save(self.clt_module.state_dict(), temporary / "clt.pt")
        torch.save(self.optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), temporary / "scheduler.pt")
        torch.save(
            {
                "schema_version": 2,
                "step": step,
                "world_size": self.world_size,
                "seed": self.seed,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "epoch": getattr(self, "_epoch", 0),
                "sequence_order": getattr(self, "_seq_order", None),
                "sequence_cursor": getattr(self, "_seq_cursor", 0),
                "model_inference_dtype_receipt": self.model_inference_dtype_receipt,
            },
            temporary / "trainer_state.pt",
        )

        with (temporary / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.config, handle, default_flow_style=False)
        manifest = self._checkpoint_manifest(temporary, int(step), kind="resumable")
        self._write_checkpoint_manifest(temporary, manifest)
        temporary.rename(ckpt_dir)
        self._prune_checkpoints()

        print(f"  Checkpoint saved: {ckpt_dir}")

    def export_for_circuit_tracer(self, step: int | None = None):
        """Export trained CLT in circuit-tracer's safetensors format (rank 0 only)."""
        if not self.is_main:
            return None
        from safetensors.torch import save_file

        clt = self.clt_module
        if step is None:
            step = clt.global_step.item()
        export_dir = self.save_dir / f"step_{step}" / "circuit_tracer_format"
        export_dir.mkdir(parents=True, exist_ok=True)

        for l in range(clt.n_layers):
            enc_tensors = {
                f"W_enc_{l}": clt.W_enc[l].data.to(torch.bfloat16),
                f"b_enc_{l}": clt.b_enc[l].data.to(torch.bfloat16),
            }
            save_file(enc_tensors, export_dir / f"W_enc_{l}.safetensors")

            dec_tensors = {
                f"W_dec_{l}": clt.W_dec[l].data.to(torch.bfloat16),
            }
            save_file(dec_tensors, export_dir / f"W_dec_{l}.safetensors")

        save_file(
            {"b_dec": clt.b_dec.data.to(torch.bfloat16)},
            export_dir / "b_dec.safetensors",
        )

        ct_config = {
            "model_kind": "cross_layer_transcoder",
            "n_layers": clt.n_layers,
            "d_model": clt.d_model,
            "d_transcoder": clt.d_clt,
            "activation_function": "relu",
            "feature_input_hook": "hook_resid_mid",
            "feature_output_hook": "hook_mlp_out",
        }
        with open(export_dir / "config.yaml", "w") as f:
            yaml.dump(ct_config, f)

        print(f"  Exported for circuit-tracer: {export_dir}")
        return export_dir
