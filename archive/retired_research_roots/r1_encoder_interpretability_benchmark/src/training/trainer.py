"""Online SAE trainer for ESM-2-3B.

Each GPU holds:
  - ESM-2-3B frozen in fp16 (~12GB VRAM)
  - BatchTopKSAE in fp32 (~0.7GB)
  - Optimizer states (~1.4GB)

DDP synchronizes SAE gradients across GPUs. Activations are computed
on-the-fly via ESM-2 forward pass with output_hidden_states=True.
"""

import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, EsmModel

from src.data.protein_dataset import ProteinTokenBatchDataset
from src.training.optim import ConstrainedAdam, get_cosine_schedule_with_warmup
from src.training.sae import BatchTopKSAE


class OnlineSAETrainer:
    """Online SAE trainer with DDP support."""

    def __init__(self, config: dict):
        self.config = config
        self.rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_main = self.rank == 0
        self.device = torch.device(f"cuda:{self.rank}")

        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.rank)

        self._load_esm2()
        self._build_sae()
        self._build_optimizer()
        self._build_dataloader()
        self._init_wandb()

        self.global_step = 0
        self.tokens_seen = 0
        self.num_tokens_since_fired = torch.zeros(
            self.config["sae"]["d_sae"], device=self.device, dtype=torch.long
        )
        self._b_dec_initialized = False

    def _load_esm2(self):
        model_path = self.config["model"]["name_or_path"]
        if self.is_main:
            print(f"Loading ESM-2 from {model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.esm_model = EsmModel.from_pretrained(
            model_path,
            add_pooling_layer=False,
            torch_dtype=torch.float16,
        ).to(self.device)

        self.esm_model.eval()
        self.esm_model.requires_grad_(False)

        if self.is_main:
            mem_gb = torch.cuda.memory_allocated(self.device) / 1e9
            print(f"ESM-2 loaded on GPU {self.rank} ({mem_gb:.1f} GB)")

    def _build_sae(self):
        cfg = self.config["sae"]
        self.sae = BatchTopKSAE(
            d_in=cfg["d_in"],
            d_sae=cfg["d_sae"],
            k=cfg["k"],
            topk_mode=cfg.get("topk_mode", "example"),
            normalize_activations=cfg.get("normalize_activations", False),
        ).to(self.device)

        if self.world_size > 1:
            self.sae_ddp = DDP(self.sae, device_ids=[self.rank])
        else:
            self.sae_ddp = self.sae

        self.sae_module = self.sae_ddp.module if isinstance(self.sae_ddp, DDP) else self.sae_ddp

        if self.is_main:
            n_params = sum(p.numel() for p in self.sae.parameters())
            print(f"SAE: d_in={cfg['d_in']}, d_sae={cfg['d_sae']}, k={cfg['k']}, params={n_params:,}")

    def _build_optimizer(self):
        tcfg = self.config["training"]
        self.optimizer = ConstrainedAdam(
            self.sae_module.parameters(),
            constrained_params=[self.sae_module.W_dec],
            lr=tcfg["lr"],
            betas=(0.9, 0.999),
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=tcfg["lr_warmup_steps"],
            total_steps=tcfg["total_steps"],
            decay_start=tcfg["lr_decay_start"],
        )

    def _build_dataloader(self):
        dcfg = self.config["data"]
        tcfg = self.config["training"]
        self.dataset = ProteinTokenBatchDataset(
            fasta_path=dcfg["fasta_path"],
            tokenizer=self.tokenizer,
            batch_size_tokens=tcfg["batch_size_tokens"],
            max_seq_len=dcfg["max_seq_len"],
            num_sequences=dcfg["num_sequences"],
            seed=dcfg["seed"],
            rank=self.rank,
            world_size=self.world_size,
        )
        # Dataset yields pre-batched dicts, so batch_size=None
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=None,
            num_workers=dcfg.get("num_workers", 4),
            pin_memory=True,
        )

    def _init_wandb(self):
        self.wandb_run = None
        lcfg = self.config.get("logging", {})
        if self.is_main and lcfg.get("wandb_project"):
            try:
                import wandb

                layer = self.config["training"]["target_layer"]
                scfg = self.config["sae"]
                self.wandb_run = wandb.init(
                    project=lcfg["wandb_project"],
                    entity=lcfg.get("wandb_entity"),
                    name=f"esm2-3B_layer{layer}_d{scfg['d_sae']}_k{scfg['k']}",
                    config=self.config,
                    resume="allow",
                )
            except ImportError:
                print("wandb not installed, logging to stdout only")

    def _extract_activations(self, batch: dict, target_layer: int) -> torch.Tensor:
        """Run ESM-2 and extract residual stream activations from target layer.

        Args:
            batch: dict with input_ids, attention_mask, token_mask
            target_layer: 0-indexed layer number

        Returns:
            (total_valid_tokens, d_in) float32 tensor
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        token_mask = batch["token_mask"].to(self.device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = self.esm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # hidden_states[0] = embeddings, hidden_states[i+1] = layer i output
        layer_acts = outputs.hidden_states[target_layer + 1]  # (B, L, d_in)

        # Extract valid AA tokens only
        valid_acts = layer_acts[token_mask]  # (N_tokens, d_in)
        return valid_acts.float()

    def train_step(self, batch: dict) -> dict:
        """One training step. Returns metrics dict."""
        tcfg = self.config["training"]
        scfg = self.config["sae"]
        target_layer = tcfg["target_layer"]

        # 1. Extract activations
        activations = self._extract_activations(batch, target_layer)
        num_tokens = activations.size(0)

        if num_tokens == 0:
            return {"skipped": True}

        # 2. Initialize b_dec to mean of activations on first step
        #    When normalize_activations=True, b_dec lives in normalized space,
        #    so we must normalize before computing the mean.
        if not self._b_dec_initialized:
            with torch.no_grad():
                if self.sae_module.normalize_activations:
                    init_acts, _ = self.sae_module._normalize(activations)
                else:
                    init_acts = activations
                mean_act = init_acts.mean(dim=0)
                if self.world_size > 1:
                    dist.all_reduce(mean_act, op=dist.ReduceOp.SUM)
                    mean_act /= self.world_size
                self.sae_module.b_dec.data.copy_(mean_act)
            self._b_dec_initialized = True

        # 3. Dead feature mask
        dead_threshold = scfg.get("dead_feature_threshold", 10_000_000)
        dead_mask = self.num_tokens_since_fired > dead_threshold

        # 4. SAE forward
        output = self.sae_ddp(
            activations,
            dead_mask=dead_mask,
            auxk_alpha=scfg.get("auxk_alpha", 1 / 32),
        )

        # 5. Backward
        output.loss.backward()

        # 6. Gradient clipping
        clip_norm = tcfg.get("grad_clip_norm")
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(self.sae_module.parameters(), clip_norm)

        # 7. Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        # 8. Update threshold EMA
        if self.global_step > tcfg.get("threshold_start_step", 1000):
            self._update_threshold(output.f)

        # 9. Update dead feature tracking
        if self.world_size > 1:
            total_tokens_t = torch.tensor(num_tokens, device=self.device)
            dist.all_reduce(total_tokens_t)
            num_tokens_global = total_tokens_t.item()
            # Sync active mask
            active_long = output.active_mask.long()
            dist.all_reduce(active_long, op=dist.ReduceOp.MAX)
            global_active = active_long.bool()
        else:
            num_tokens_global = num_tokens
            global_active = output.active_mask

        self.num_tokens_since_fired += num_tokens_global
        self.num_tokens_since_fired[global_active] = 0
        self.tokens_seen += num_tokens_global

        # 10. Metrics
        metrics = dict(output.metrics)
        metrics["features/dead_count"] = int(dead_mask.sum())
        metrics["features/dead_pct"] = dead_mask.float().mean().item()
        metrics["training/lr"] = self.scheduler.get_last_lr()[0]
        metrics["training/threshold"] = self.sae_module.threshold.item()
        metrics["training/tokens_per_step"] = num_tokens_global
        metrics["training/tokens_seen"] = self.tokens_seen

        return metrics

    def resample_dead_features(self, batch: dict) -> int:
        """Resample dead features by reinitializing from high-loss examples.

        Standard technique from Anthropic/OpenAI SAE papers. Solves the
        gradient dead zone problem: auxk alone cannot update W_enc for
        features stuck in ReLU's zero region.

        Steps:
          1. Forward pass to find high-reconstruction-error tokens
          2. Sample residual directions weighted by per-token loss
          3. Set W_dec[dead] = normalized residual direction
          4. Set W_enc[dead] = same direction * small scale
          5. Reset b_enc[dead] = 0 and optimizer state
        """
        scfg = self.config["sae"]
        dead_threshold = scfg.get("dead_feature_threshold", 1_000_000)
        dead_mask = self.num_tokens_since_fired > dead_threshold
        n_dead = int(dead_mask.sum())

        if n_dead == 0:
            return 0

        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        target_layer = self.config["training"]["target_layer"]

        # Compute resampling directions on rank 0
        directions = torch.zeros(n_dead, self.sae_module.d_in, device=self.device)
        avg_norm = torch.tensor(1.0, device=self.device)

        if self.is_main:
            activations = self._extract_activations(batch, target_layer)

            if activations.size(0) > 0:
                with torch.no_grad():
                    output = self.sae_module(activations)
                    per_token_loss = (activations - output.x_hat).pow(2).sum(dim=-1)

                # Sample high-loss tokens (probability weighted by loss)
                probs = per_token_loss / per_token_loss.sum().clamp(min=1e-8)
                n_samples = min(n_dead, len(probs))
                sample_idx = torch.multinomial(
                    probs, num_samples=n_samples,
                    replacement=(n_dead > len(probs)),
                )

                # Residual directions in the space where W_dec lives
                # When normalize_activations=True, W_dec operates in normalized space
                if self.sae_module.normalize_activations:
                    acts_norm, _ = self.sae_module._normalize(activations)
                    f, _, _ = self.sae_module.encode(acts_norm)
                    x_hat_norm = self.sae_module.decode(f)
                    residuals = acts_norm[sample_idx] - x_hat_norm[sample_idx]
                else:
                    residuals = activations[sample_idx] - output.x_hat[sample_idx]
                norms = residuals.norm(dim=1, keepdim=True).clamp(min=1e-8)
                sampled_dirs = residuals / norms

                # Tile if more dead features than samples
                if n_dead > n_samples:
                    repeats = (n_dead + n_samples - 1) // n_samples
                    sampled_dirs = sampled_dirs.repeat(repeats, 1)[:n_dead]
                directions.copy_(sampled_dirs)

                # Average encoder norm of alive features (for scaling)
                alive_mask = ~dead_mask
                if alive_mask.any():
                    avg_norm.fill_(
                        self.sae_module.W_enc[alive_mask].norm(dim=1).mean().item()
                    )

        # Broadcast to all ranks
        if self.world_size > 1:
            dist.broadcast(directions, src=0)
            dist.broadcast(avg_norm, src=0)

        # Apply resampled weights on all ranks
        with torch.no_grad():
            self.sae_module.W_dec.data[dead_indices] = directions
            self.sae_module.W_enc.data[dead_indices] = directions * avg_norm * 0.2
            self.sae_module.b_enc.data[dead_indices] = 0.0

        # Reset optimizer state for resampled features
        self._reset_optimizer_state_for_features(dead_indices)

        # Reset dead feature counters
        self.num_tokens_since_fired[dead_indices] = 0

        return n_dead

    def _reset_optimizer_state_for_features(self, feature_indices: torch.Tensor):
        """Reset Adam momentum/variance for resampled feature parameters."""
        d_sae = self.sae_module.d_sae
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p not in self.optimizer.state:
                    continue
                state = self.optimizer.state[p]
                for key in ("exp_avg", "exp_avg_sq"):
                    if key not in state:
                        continue
                    buf = state[key]
                    # W_enc or W_dec: shape (d_sae, d_in), reset rows
                    if buf.dim() == 2 and buf.shape[0] == d_sae:
                        buf[feature_indices] = 0
                    # b_enc: shape (d_sae,), reset entries
                    elif buf.dim() == 1 and buf.shape[0] == d_sae:
                        buf[feature_indices] = 0

    @torch.no_grad()
    def _update_threshold(self, f: torch.Tensor):
        beta = self.config["training"].get("threshold_beta", 0.999)
        active = f[f > 0]
        if active.numel() == 0:
            return
        min_val = active.min().float()
        if self.sae_module.threshold < 0:
            self.sae_module.threshold.fill_(min_val)
        else:
            self.sae_module.threshold.mul_(beta).add_((1 - beta) * min_val)

    def save_checkpoint(self, path: str | None = None):
        if not self.is_main:
            return
        if path is None:
            layer = self.config["training"]["target_layer"]
            path = os.path.join(
                self.config["checkpoint"]["save_dir"],
                f"layer_{layer}",
                f"step_{self.global_step}",
            )
        os.makedirs(path, exist_ok=True)
        torch.save(self.sae_module.state_dict(), os.path.join(path, "sae.pt"))
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "tokens_seen": self.tokens_seen,
                "num_tokens_since_fired": self.num_tokens_since_fired.cpu(),
                "b_dec_initialized": self._b_dec_initialized,
            },
            os.path.join(path, "trainer_state.pt"),
        )
        # Save config alongside checkpoint
        with open(os.path.join(path, "config.yaml"), "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
        print(f"[Step {self.global_step}] Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        sae_state = torch.load(os.path.join(path, "sae.pt"), map_location=self.device, weights_only=True)
        self.sae_module.load_state_dict(sae_state)

        state = torch.load(os.path.join(path, "trainer_state.pt"), map_location=self.device, weights_only=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["global_step"]
        self.tokens_seen = state.get("tokens_seen", 0)
        self.num_tokens_since_fired = state["num_tokens_since_fired"].to(self.device)
        self._b_dec_initialized = state.get("b_dec_initialized", True)

        if self.is_main:
            print(f"Resumed from step {self.global_step} at {path}")

    def fit(self):
        """Main training loop."""
        tcfg = self.config["training"]
        total_steps = tcfg["total_steps"]
        save_every = self.config["checkpoint"]["save_every_steps"]
        log_every = self.config["logging"]["log_every_steps"]
        layer = tcfg["target_layer"]

        # Resume
        resume_path = self.config["checkpoint"].get("resume_from")
        if resume_path:
            self.load_checkpoint(resume_path)

        if self.is_main:
            print(f"\n{'=' * 60}")
            print(f"  Training SAE for ESM-2-3B layer {layer}")
            print(f"  Steps: {self.global_step}/{total_steps}, GPUs: {self.world_size}")
            print(f"  SAE: d_sae={self.config['sae']['d_sae']}, k={self.config['sae']['k']}")
            print(f"{'=' * 60}\n")

        torch.set_float32_matmul_precision("high")
        step_times = []

        for batch in self.dataloader:
            if self.global_step >= total_steps:
                break

            t0 = time.time()
            metrics = self.train_step(batch)
            step_time = time.time() - t0

            if "skipped" in metrics:
                continue

            step_times.append(step_time)
            self.global_step += 1

            # Log
            if self.is_main and self.global_step % log_every == 0:
                avg_time = sum(step_times[-100:]) / len(step_times[-100:])
                tps = metrics["training/tokens_per_step"] / avg_time
                metrics["training/step_time_s"] = avg_time
                metrics["training/tokens_per_sec"] = tps

                if self.wandb_run:
                    self.wandb_run.log(metrics, step=self.global_step)

                print(
                    f"Step {self.global_step:>7d}/{total_steps} | "
                    f"loss={metrics['loss/total']:.4f} | "
                    f"FVU={metrics['performance/fvu']:.4f} | "
                    f"L0={metrics['performance/l0']:.1f} | "
                    f"dead={metrics['features/dead_pct']:.3f} | "
                    f"lr={metrics['training/lr']:.2e} | "
                    f"{tps:.0f} tok/s | "
                    f"{avg_time:.2f}s/step"
                )

            # Feature resampling
            resample_every = self.config["sae"].get("resample_every", 0)
            if resample_every and self.global_step > 0 and self.global_step % resample_every == 0:
                n_resampled = self.resample_dead_features(batch)
                if self.is_main and n_resampled > 0:
                    pct = n_resampled / self.config["sae"]["d_sae"] * 100
                    print(
                        f"[Step {self.global_step}] Resampled {n_resampled} "
                        f"dead features ({pct:.1f}%)"
                    )

            # Checkpoint
            if self.global_step % save_every == 0:
                self.save_checkpoint()
                if self.world_size > 1:
                    dist.barrier()

        # Final checkpoint
        self.save_checkpoint()

        if self.is_main:
            print(f"\nTraining complete. Layer {layer}, {self.global_step} steps, {self.tokens_seen:,} tokens.")
            if self.wandb_run:
                self.wandb_run.finish()

        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
