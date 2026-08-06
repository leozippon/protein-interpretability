"""Cross-layer and per-layer transcoders, in plain torch.

**Why this exists.** ProGenMech's central claim is that a cross-layer transcoder
(CLT) beats a per-layer transcoder (PLT) at replacing ProGen3-112M's MoE blocks.
Their CLT weights are unobtainable -- the mirror serves the PLT directory and
returns HTTP 403 for the CLT one on the same token -- and their trainer cannot
run in an offline pod, which needs ``pytorch_lightning``, ``polars`` and
``wandb``. So the claim could be gated on their baseline and not on their
headline (EXP-R2-130).

This module removes that bound the only way left: **train both, ourselves, under
identical conditions.** That is a stronger comparison than their released weights
could have supported in any case. Their PLT was trained on their data with their
schedule; ours are trained on the same data, for the same steps, from the same
seed, differing in exactly the thing under test -- whether a latent may write to
layers downstream of the one it was read from.

**What is reproduced, and what is not.** The architecture and objective follow
their ``training/clt_model.py`` and ``training_transcoder/plt_model.py`` as read
from the released code: a hand-rolled layer norm whose statistics are kept for
de-normalisation, a bias subtracted *after* that norm and added back before it,
TopK on the pre-activations followed by ReLU on the selected values, a per-layer
NMSE objective with no L1 term, and an auxiliary loss that revives dead latents
through the self-layer decoder only. Three deviations are deliberate and are
recorded rather than smoothed:

1. **Decoder initialisation.** Their CLT re-runs ``kaiming_uniform_`` on the
   *encoder* weight inside the decoder-init loop, so each of the 55 decoders is
   a different random draw and the surviving encoder weight is only the last of
   them. That is a defect, not a design; we initialise each decoder directly and
   leave the encoders alone. Their PLT's init loop does not have it.
2. **Decoder normalisation.** Their ``norm_weights``/``norm_grad`` are defined
   and never called anywhere in their repository, so decoders are unconstrained
   during training. We match that -- unconstrained -- rather than adding a
   constraint they did not train under.
3. **Scale.** This is a bounded reproduction: a declared step budget, not their
   full 5M-sequence epoch. The comparison it supports is CLT against PLT at
   equal budget, not an absolute loss against their published number.

The estimand a downstream stage measures is the same either way: the transcoder
reads a MoE block's **input** (the output of ``post_attention_layernorm``) and
predicts that block's **output** before the residual add.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TranscoderConfig:
    """Everything that defines a transcoder, recorded in every checkpoint."""

    num_layers: int = 10
    d_model: int = 384
    d_hidden: int = 4608
    k: int = 64
    auxk: int = 128
    #: Steps a latent may stay silent before the auxiliary loss revives it.
    #: Their ``dead_steps_threshold`` is in sequences and is divided by the batch
    #: size to reach steps; the quotient is what matters, so it is declared here
    #: in the unit it is used in.
    dead_steps: int = 625
    aux_weight: float = 1.0 / 32.0
    cross_layer: bool = True

    def record(self) -> dict[str, Any]:
        return {
            "architecture": "CLT" if self.cross_layer else "PLT",
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "d_hidden": self.d_hidden,
            "k": self.k,
            "auxk": self.auxk,
            "dead_steps": self.dead_steps,
            "aux_weight": self.aux_weight,
        }


def normalise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Layer norm that hands back its statistics, so the output can be undone.

    ``std`` is torch's default (unbiased, n-1) and the epsilon sits *outside* the
    root and *inside* the divisor, matching the released implementation. The
    de-normalisation divides by ``std`` alone rather than ``std + eps``, which is
    an asymmetry in their code and is reproduced because the trained weights
    depend on it.
    """

    mean = x.mean(dim=-1, keepdim=True)
    centred = x - mean
    std = centred.std(dim=-1, keepdim=True)
    return centred / (std + 1e-5), mean, std


def topk_relu(pre: torch.Tensor, k: int) -> torch.Tensor:
    """TopK on the pre-activations, then ReLU on the values that survived.

    In that order, which is theirs and is not the same as ReLU-then-TopK: a
    selected pre-activation that is negative becomes exactly zero, so a latent
    can be in the top-k and still contribute nothing.
    """

    values, indices = torch.topk(pre, k=k, dim=-1)
    out = torch.zeros_like(pre)
    return out.scatter_(-1, indices, F.relu(values))


class Transcoder(nn.Module):
    """A CLT or a PLT; the only difference is which decoders exist.

    One class rather than two because the objective, the normalisation, the
    activation, the dead-latent bookkeeping and the auxiliary loss are identical
    -- and because the comparison this module exists for is only meaningful if
    everything except the decoder connectivity is literally the same code.
    """

    def __init__(self, config: TranscoderConfig):
        super().__init__()
        self.config = config
        L, d, h = config.num_layers, config.d_model, config.d_hidden

        self.encoders = nn.ModuleList([nn.Linear(d, h) for _ in range(L)])
        self.b_pre = nn.Parameter(torch.zeros(L, d))
        pairs = (
            [(s, t) for s in range(L) for t in range(s, L)]
            if config.cross_layer
            else [(layer, layer) for layer in range(L)]
        )
        self.pairs = pairs
        self.decoders = nn.ParameterDict(
            {
                f"{s}_{t}": nn.Parameter(torch.empty(h, d))
                for s, t in pairs
            }
        )
        for parameter in self.decoders.values():
            nn.init.kaiming_uniform_(parameter, a=5 ** 0.5)
        # Steps since each latent last fired, for the auxiliary revival loss.
        self.register_buffer("silent_steps", torch.zeros(L, h, dtype=torch.long))

    @property
    def writes_to(self) -> dict[int, list[int]]:
        """Target layers each source layer may write to, for the record."""

        out: dict[int, list[int]] = {}
        for source, target in self.pairs:
            out.setdefault(source, []).append(target)
        return out

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``inputs`` is ``(L, tokens, d_model)``; returns latents and statistics."""

        normed, means, stds = [], [], []
        pre_activations = []
        for layer in range(self.config.num_layers):
            hat, mean, std = normalise(inputs[layer])
            pre_activations.append(self.encoders[layer](hat - self.b_pre[layer]))
            normed.append(hat)
            means.append(mean)
            stds.append(std)
        pre = torch.stack(pre_activations)
        latents = topk_relu(pre, self.config.k)
        return latents, pre, torch.stack(means), torch.stack(stds)

    def decode(
        self, latents: torch.Tensor, means: torch.Tensor, stds: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct every layer's MoE output from the latents that may reach it."""

        outputs = []
        for target in range(self.config.num_layers):
            total = None
            for source, other in self.pairs:
                if other != target:
                    continue
                contribution = latents[source] @ self.decoders[f"{source}_{target}"]
                total = contribution if total is None else total + contribution
            assert total is not None
            total = total + self.b_pre[target]
            outputs.append(total * stds[target] + means[target])
        return torch.stack(outputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latents, _, means, stds = self.encode(inputs)
        return self.decode(latents, means, stds)

    # ------------------------------------------------------------------ losses

    def objective(
        self, inputs: torch.Tensor, targets: torch.Tensor, *, training: bool
    ) -> dict[str, Any]:
        """Per-layer NMSE, plus the dead-latent auxiliary term.

        There is no sparsity penalty: sparsity is TopK's, entirely. A reader
        expecting an L1 term should not go looking for one.
        """

        latents, pre, means, stds = self.encode(inputs)
        reconstruction = self.decode(latents, means, stds)

        variance = targets.var(dim=(1, 2), unbiased=False)
        per_layer = ((reconstruction - targets) ** 2).mean(dim=(1, 2)) / (variance + 1e-8)
        nmse = per_layer.sum()

        aux = torch.zeros((), device=inputs.device, dtype=nmse.dtype)
        n_dead = 0
        if training:
            fired = (latents > 0).any(dim=1)
            self.silent_steps = torch.where(
                fired, torch.zeros_like(self.silent_steps), self.silent_steps + 1
            )
            dead = self.silent_steps > self.config.dead_steps
            n_dead = int(dead.sum())
            if n_dead:
                residual = (targets - reconstruction).detach()
                for layer in range(self.config.num_layers):
                    layer_dead = dead[layer]
                    if not bool(layer_dead.any()):
                        continue
                    k_aux = int(min(self.config.auxk, int(layer_dead.sum())))
                    masked = pre[layer].masked_fill(~layer_dead, float("-inf"))
                    revived = topk_relu(masked, k_aux)
                    # Self-layer decoder only: a dead latent is revived where it
                    # was read, not by borrowing another layer's output.
                    predicted = revived @ self.decoders[f"{layer}_{layer}"]
                    aux = aux + F.mse_loss(predicted, residual[layer])

        return {
            "loss": nmse + self.config.aux_weight * aux,
            "nmse_sum": nmse,
            "nmse_per_layer": per_layer.detach(),
            "aux": aux.detach(),
            "n_dead": n_dead,
            "active_fraction": float((latents > 0).float().mean()),
        }


@dataclass
class TrainingRecord:
    """What a run reports about itself, beyond its weights."""

    steps: int = 0
    tokens: int = 0
    sequences: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "tokens": self.tokens,
            "sequences": self.sequences,
            "history": self.history,
        }
