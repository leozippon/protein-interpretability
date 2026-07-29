"""TopK Sparse Autoencoder for protein language model interpretability.

Architecture (with sqrt(d) normalization):
  normalize: x_norm, original_norms = sqrt_d_normalize(x)
  encode: f = TopK(ReLU(W_enc @ (x_norm - b_dec) + b_enc))
  decode: x_hat_norm = W_dec @ f + b_dec
  unnormalize: x_hat = x_hat_norm * original_norms / sqrt(d)

Supports two TopK modes:
  - "example": per-example TopK (InterProt style) — each token independently selects top-k
  - "batch": BatchTopK — selects top k*batch_size across entire batch (high dead risk)

W_dec rows are constrained to unit norm.
Encoder is initialized as transpose of decoder.

Reference: Gao et al. 2024 "Scaling and evaluating sparse autoencoders"
"""

import torch
import torch.nn as nn
from typing import NamedTuple


class SAEOutput(NamedTuple):
    x_hat: torch.Tensor       # reconstructed activations (batch, d_in)
    f: torch.Tensor            # sparse feature activations (batch, d_sae)
    active_mask: torch.Tensor  # which features fired in this batch (d_sae,)
    loss: torch.Tensor         # total loss scalar
    metrics: dict              # logging metrics


class BatchTopKSAE(nn.Module):
    """TopK Sparse Autoencoder with configurable topk mode.

    Args:
        d_in: input dimension (ESM-2 hidden size)
        d_sae: dictionary size (number of SAE features)
        k: number of top activations to keep
        topk_mode: "example" (per-token, recommended) or "batch" (global)
        normalize_activations: if True, apply Anthropic's sqrt(d) normalization
            before encoding and undo after decoding. Normalizes each token to
            unit L2 norm then scales by sqrt(d_in), preserving direction while
            equalizing magnitude across layers. Critical for deeper ESM-2 layers.
    """

    def __init__(self, d_in: int, d_sae: int, k: int, topk_mode: str = "example",
                 normalize_activations: bool = False):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self.topk_mode = topk_mode
        self.normalize_activations = normalize_activations

        # Decoder: (d_sae -> d_in), rows are feature directions
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        nn.init.kaiming_uniform_(self.W_dec)
        # Normalize rows to unit norm
        with torch.no_grad():
            self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=1, keepdim=True)

        # Encoder: (d_in -> d_sae), initialized as decoder transpose
        self.W_enc = nn.Parameter(self.W_dec.data.clone())
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Decoder bias (mean-centered, initialized at step 0 from data)
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Threshold for inference (updated via EMA during training)
        self.register_buffer("threshold", torch.tensor(-1.0))

    def _normalize(self, x: torch.Tensor):
        """Anthropic's sqrt(d) normalization: unit-norm vectors scaled by sqrt(d).

        Preserves activation direction while equalizing magnitude across layers.
        Unlike LayerNorm, does not destroy information about relative activation
        patterns — only scales the vector norm. Used by Anthropic for Claude SAEs
        and by InterPLM for ESM-2 SAEs.

        Returns:
            x_norm: normalized activations (batch, d_in)
            original_norms: per-token L2 norms (batch, 1) for un-normalization
        """
        original_norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sqrt_d = (x.shape[-1] ** 0.5)
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1) * sqrt_d
        return x_norm, original_norms

    def _unnormalize(self, x_hat_norm: torch.Tensor, original_norms: torch.Tensor):
        """Undo sqrt(d) normalization."""
        sqrt_d = (x_hat_norm.shape[-1] ** 0.5)
        return x_hat_norm * original_norms / sqrt_d

    @torch.no_grad()
    def normalize_decoder(self):
        """Re-normalize decoder weight rows to unit norm."""
        norms = self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data /= norms

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """Project out gradient component parallel to decoder directions."""
        if self.W_dec.grad is None:
            return
        W = self.W_dec.data  # (d_sae, d_in)
        G = self.W_dec.grad  # (d_sae, d_in)
        W_normed = W / W.norm(dim=1, keepdim=True).clamp(min=1e-8)
        parallel = (G * W_normed).sum(dim=1, keepdim=True)  # (d_sae, 1)
        G -= parallel * W_normed

    def _topk_activation(self, pre_acts: torch.Tensor) -> torch.Tensor:
        """Apply TopK activation based on configured mode.

        pre_acts are RAW (no pre-ReLU). TopK selects highest values,
        then ReLU ensures non-negative output activations.

        Args:
            pre_acts: (batch, d_sae) raw pre-activations (can be negative)

        Returns:
            f: (batch, d_sae) sparse non-negative activations with only top-k kept
        """
        if self.topk_mode == "example":
            # Per-example TopK: each token independently selects its top-k
            topk = torch.topk(pre_acts, k=min(self.k, pre_acts.size(-1)), dim=-1, sorted=False)
            values = torch.relu(topk.values)
            f = torch.zeros_like(pre_acts)
            f.scatter_(-1, topk.indices, values)
        else:
            # BatchTopK: select top k*batch_size across entire batch
            batch_size = pre_acts.size(0)
            total_k = self.k * batch_size
            flat = pre_acts.reshape(-1)
            topk_vals, topk_indices = flat.topk(min(total_k, flat.numel()), sorted=False)
            f_flat = torch.zeros_like(flat)
            f_flat.scatter_(0, topk_indices, torch.relu(topk_vals))
            f = f_flat.reshape(pre_acts.shape)
        return f

    def encode(self, x: torch.Tensor, use_threshold: bool = False):
        """Encode input to sparse features.

        Args:
            x: (batch, d_in) — already normalized if normalize_activations is True
            use_threshold: if True, use learned threshold (for inference)

        Returns:
            f: (batch, d_sae) sparse feature activations
            active_mask: (d_sae,) which features fired
            pre_acts: (batch, d_sae) raw pre-activations (NO ReLU, for auxk gradient flow)
        """
        # Raw pre-activations — NO ReLU here!
        # ReLU is applied inside _topk_activation() on selected values only.
        # This preserves non-zero pre_acts for dead features, enabling auxk
        # gradient to flow back through W_enc (critical for feature revival).
        pre_acts = torch.nn.functional.linear(x - self.b_dec, self.W_enc, self.b_enc)

        if use_threshold and self.threshold >= 0:
            f = torch.relu(pre_acts) * (pre_acts > self.threshold)
        else:
            f = self._topk_activation(pre_acts)

        active_mask = (f.sum(dim=0) > 0)
        return f, active_mask, pre_acts

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to activation space."""
        return torch.nn.functional.linear(f, self.W_dec.T) + self.b_dec

    def forward(
        self,
        x: torch.Tensor,
        dead_mask: torch.Tensor | None = None,
        auxk_alpha: float = 1 / 32,
    ) -> SAEOutput:
        """Full forward pass with loss computation.

        Args:
            x: (batch, d_in) input activations (raw, before normalization)
            dead_mask: (d_sae,) boolean mask of dead features
            auxk_alpha: weight for auxiliary dead-feature loss
        """
        # Optional sqrt(d) normalization (critical for deeper ESM-2 layers)
        if self.normalize_activations:
            x_norm, original_norms = self._normalize(x)
        else:
            x_norm = x

        f, active_mask, pre_acts = self.encode(x_norm)
        x_hat_norm = self.decode(f)

        # Compute residual and loss in normalized space
        residual = x_norm - x_hat_norm
        recon_loss = residual.pow(2).sum(dim=-1).mean()

        # Auxiliary loss for dead features (in normalized space)
        auxk_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if dead_mask is not None and dead_mask.any():
            auxk_loss = self._auxk_loss(residual.detach(), pre_acts, dead_mask)

        total_loss = recon_loss + auxk_alpha * auxk_loss

        # Unnormalize reconstruction for output
        if self.normalize_activations:
            x_hat = self._unnormalize(x_hat_norm, original_norms)
        else:
            x_hat = x_hat_norm

        with torch.no_grad():
            l0 = (f > 0).float().sum(dim=-1).mean()
            x_norm_centered = x_norm - x_norm.mean(dim=0)
            total_var = x_norm_centered.pow(2).sum()
            fvu = residual.pow(2).sum() / total_var.clamp(min=1e-8)

        metrics = {
            "loss/reconstruction": recon_loss.item(),
            "loss/auxiliary": auxk_loss.item(),
            "loss/total": total_loss.item(),
            "performance/l0": l0.item(),
            "performance/fvu": fvu.item(),
        }

        return SAEOutput(x_hat, f, active_mask, total_loss, metrics)

    def _auxk_loss(
        self,
        residual: torch.Tensor,
        pre_acts: torch.Tensor,
        dead_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Auxiliary loss: dead features reconstruct residual error.

        Follows Gao et al. 2024 (OpenAI TopK SAE paper) + InterProt:
        1. Select top-k activations from dead features only
        2. Apply ReLU to selected values (match main path behavior)
        3. Decode them to predict the residual
        4. MSE normalized by residual variance
        """
        num_dead = int(dead_mask.sum())
        k_aux = min(self.d_in // 2, num_dead)
        if k_aux == 0:
            return torch.tensor(0.0, device=residual.device, dtype=residual.dtype)

        # Mask living features to -inf
        dead_pre_acts = torch.where(dead_mask.unsqueeze(0), pre_acts, torch.tensor(-torch.inf, device=pre_acts.device, dtype=pre_acts.dtype))
        topk_vals, topk_indices = dead_pre_acts.topk(k_aux, dim=-1, sorted=False)

        # Apply ReLU to selected values (matching InterProt and _topk_activation)
        # This ensures only non-negative activations are decoded, consistent with
        # how the decoder is trained in the main loss path.
        topk_vals = torch.relu(topk_vals)

        # Scatter into full-size buffer and decode
        auxk_f = torch.zeros_like(pre_acts)
        auxk_f.scatter_(1, topk_indices, topk_vals)
        e_hat = torch.nn.functional.linear(auxk_f, self.W_dec.T)  # no b_dec for aux

        auxk_mse = (residual.float() - e_hat.float()).pow(2).sum(dim=-1).mean()
        residual_var = (residual.float() - residual.float().mean(dim=0)).pow(2).sum(dim=-1).mean()

        return (auxk_mse / residual_var.clamp(min=1e-8)).nan_to_num(0.0)
