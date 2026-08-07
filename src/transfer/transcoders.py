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

**The released PLT lives here too**, in :class:`PerLayerTranscoder`. It was
written inside ``15_replacement_faithfulness.py`` and could not leave it: a
module whose name begins with a digit cannot be imported, so the second stage
that needed to load the same checkpoint would have had to carry its own copy of
the forward pass -- which is the shape of the rendering defect Appendix B rule 12
exists to stop. One declaration of what a transcoder is, trained or released.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .arms import REPO

#: The released ProGenMech per-layer transcoder. Under ``external_resources``,
#: which is git-ignored and CC BY-NC-ND: this repository reads its tensors and
#: imports none of its code.
#: Overridable, because the staged copy does not sit where the B-local one does:
#: the pod's tree carries no ``baselines/`` segment, so a hard-coded default
#: resolves to a path that does not exist there and a run dies after the process
#: is already up. ``h200_env.sh`` exports the pod-side location.
DEFAULT_REPLACEMENT = Path(
    os.environ.get(
        "TRANSFER_PROGENMECH_PLT",
        REPO
        / "external_resources/baselines/ProGenMechModels/ProGen3_PLT_L10_D4608"
        / "checkpoints/last.ckpt",
    )
)

#: Their dead-latent threshold, in the unit their config states it in. The model
#: divides by the batch size to reach a step count, so a trainer that hard-codes
#: the step count makes ``--batch-size`` silently change how long a latent may
#: stay silent. Declared here once and divided where the batch is known.
DEAD_STEPS_SEQUENCES = 10_000


@dataclass(frozen=True)
class TranscoderConfig:
    """Everything that defines a transcoder, recorded in every checkpoint."""

    num_layers: int = 10
    d_model: int = 384
    d_hidden: int = 4608
    k: int = 64
    #: Dead latents revived per layer per step. Their *scripts* declare 128 and
    #: their *models* never read it: ``clt_model.py`` and ``plt_model.py`` both
    #: use ``min(d_model // 2, n_dead)``, which is 192 here. The effective value
    #: is the one reproduced, because the declared one is dead code in their
    #: repository and reproducing it would be reproducing a file rather than a
    #: computation.
    auxk: int = 192
    #: Steps a latent may stay silent before the auxiliary loss revives it.
    #: Their ``dead_steps_threshold`` is in *sequences*; the quotient with the
    #: batch size is what the model uses, so the trainer derives this from the
    #: batch it actually runs (see :data:`DEAD_STEPS_SEQUENCES`) rather than
    #: letting a fixed step count silently mean different things at different
    #: batch sizes.
    dead_steps: int = 625
    aux_weight: float = 1.0 / 32.0
    cross_layer: bool = True

    def n_parameters(self) -> int:
        """Trainable parameters, in closed form, so an arm can be sized before it runs.

        Encoders ``L*(d*h + h)``, decoders ``pairs*h*d``, ``b_pre`` ``L*d``. The
        CLT's ``pairs`` is ``L(L+1)/2`` against the PLT's ``L`` -- a 3.25x
        parameter advantage at ``L=10``, which is why a parameter-matched PLT is
        a control this comparison needs rather than a refinement it can skip.
        """

        pairs = (
            self.num_layers * (self.num_layers + 1) // 2 if self.cross_layer else self.num_layers
        )
        encoders = self.num_layers * (self.d_model * self.d_hidden + self.d_hidden)
        decoders = pairs * self.d_hidden * self.d_model
        return encoders + decoders + self.num_layers * self.d_model

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
            "n_parameters": self.n_parameters(),
            "active_latents_per_token": self.k,
            "active_fraction_of_dictionary": self.k / self.d_hidden,
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

    def encode_layer(
        self, layer: int, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One layer's pre-activations and normalisation statistics.

        The single place a layer is encoded. :meth:`encode` calls it for the
        batched training path and :class:`TranscoderReplacement` calls it one
        block at a time inside a live forward pass, so a replacement cannot come
        to disagree with the objective it was trained against.
        """

        hat, mean, std = normalise(inputs)
        return self.encoders[layer](hat - self.b_pre[layer]), mean, std

    def decode_target(
        self,
        target: int,
        latents: dict[int, torch.Tensor],
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """One target layer's reconstruction, from whichever sources have fired.

        ``latents`` is keyed by source layer. Every pair reaching ``target`` must
        be present: a CLT source that has not been encoded yet would silently
        contribute nothing and the reconstruction would be a different, better-
        looking object than the one that was trained. Since every pair runs
        ``source <= target`` and MoE blocks execute in layer order, a live
        forward pass always has them -- so a missing one is a defect, and it
        raises.
        """

        total = None
        for source, other in self.pairs:
            if other != target:
                continue
            if source not in latents:
                raise KeyError(
                    f"decoding layer {target} needs source layer {source}, which has "
                    "not been encoded; a cross-layer reconstruction missing a source "
                    "is not the model that was trained"
                )
            contribution = latents[source] @ self.decoders[f"{source}_{target}"]
            total = contribution if total is None else total + contribution
        assert total is not None
        return (total + self.b_pre[target]) * std + mean

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``inputs`` is ``(L, tokens, d_model)``; returns latents and statistics."""

        means, stds = [], []
        pre_activations = []
        for layer in range(self.config.num_layers):
            pre_layer, mean, std = self.encode_layer(layer, inputs[layer])
            pre_activations.append(pre_layer)
            means.append(mean)
            stds.append(std)
        pre = torch.stack(pre_activations)
        latents = topk_relu(pre, self.config.k)
        return latents, pre, torch.stack(means), torch.stack(stds)

    def decode(
        self, latents: torch.Tensor, means: torch.Tensor, stds: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct every layer's MoE output from the latents that may reach it."""

        by_source = {layer: latents[layer] for layer in range(self.config.num_layers)}
        return torch.stack(
            [
                self.decode_target(target, by_source, means[target], stds[target])
                for target in range(self.config.num_layers)
            ]
        )

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
                    #
                    # The bias and the de-normalisation are not decoration. The
                    # target is a residual in raw activation space; a decoder
                    # output is in the normalised space the encoder read from, and
                    # the two differ by this layer's own scale. Comparing them
                    # directly -- which this did until the defect was measured --
                    # makes the auxiliary gradient 1.8x to 19.3x too large and
                    # points it in the wrong direction (cosine 0.70-0.94 against
                    # the correct gradient in the first six layers), worst where
                    # the activation scale is smallest. It is also asymmetric
                    # across the comparison this module exists for: this decoder
                    # is the PLT's only contributor to its layer and one of up to
                    # ten in the CLT. Their ``clt_model.py`` and ``plt_model.py``
                    # both de-normalise here, and so does this now.
                    predicted = (
                        revived @ self.decoders[f"{layer}_{layer}"] + self.b_pre[layer]
                    ) * stds[layer] + means[layer]
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


# ------------------------------------------------------------- the free baseline


class LinearReplacement:
    """The replacement that needs no dictionary: one affine map per block.

    **Why this exists.** Standing rule 28 says a method is scored against the
    trivial baseline available from its own coordinates, and it has never been
    applied to the replacement stage -- which scored exactly three conditions,
    the original, the transcoder, and the mean-ablated floor that is the
    *denominator*. Nothing stood between the floor and the method. Twice before,
    applying this rule changed what a result meant: a selector that knows only a
    head's layer index beat a census (EXP-R2-131), and a BLOSUM62 lookup was
    not separable from a model's zero-shot fitness on the panel its recovery
    ratios were quoted against (EXP-R2-133/134).

    A per-layer least-squares map ``y ~= (x - mu_x) W + mu_y`` carries
    ``d_model^2 + d_model`` parameters -- **147,840** against a cross-layer
    transcoder's 115,065,600, a factor of 778 -- has no sparsity, no latents, no
    training loop and no hyper-parameters beyond a ridge term. It is what a MoE
    block would be if it were a linear function of its input.

    If this recovers what the transcoders recover, then nothing measured on this
    estimand is evidence that a sparse dictionary captured anything: the
    quantity would be a property of how linear the block is, not of the method.
    That is the outcome this class exists to be able to report.
    """

    def __init__(self, weight: torch.Tensor, x_mean: torch.Tensor, y_mean: torch.Tensor) -> None:
        self.weight = weight  # (layers, d_model, d_model)
        self.x_mean = x_mean  # (layers, d_model)
        self.y_mean = y_mean
        self.num_layers = int(weight.shape[0])

    @property
    def n_parameters(self) -> int:
        layers, d_model, _ = self.weight.shape
        return layers * (d_model * d_model + d_model)

    def to(self, device: torch.device | str) -> LinearReplacement:
        self.weight = self.weight.to(device=device, dtype=torch.float32)
        self.x_mean = self.x_mean.to(device=device, dtype=torch.float32)
        self.y_mean = self.y_mean.to(device=device, dtype=torch.float32)
        return self

    def reset(self) -> None:
        """No cross-token or cross-layer state. Present so callers need not care."""

    @torch.no_grad()
    def __call__(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).float()
        out = (flat - self.x_mean[layer]) @ self.weight[layer] + self.y_mean[layer]
        return out.reshape(shape).to(x.dtype)

    def record(self) -> dict[str, Any]:
        return {
            "architecture": "LINEAR",
            "num_layers": self.num_layers,
            "d_model": int(self.weight.shape[1]),
            "n_parameters": self.n_parameters,
            "note": "per-layer affine least-squares map from block input to block "
            "output; the free baseline standing rule 28 requires, carrying no "
            "dictionary, no sparsity and no trained latents",
        }


class LinearReplacementFitter:
    """Accumulates the Gram matrices one affine map per layer is solved from.

    Streaming rather than storing activations: the normal equations need only
    ``X^T X`` and ``X^T Y`` per layer, which are ``d_model`` square regardless of
    how many tokens are seen. Accumulated in float64 because ``X^T X`` on
    activations whose scale spans two orders of magnitude across layers loses
    conditioning in float32 long before it loses it in float64.
    """

    def __init__(self, num_layers: int, d_model: int) -> None:
        self.num_layers, self.d_model = num_layers, d_model
        self.xtx = torch.zeros(num_layers, d_model, d_model, dtype=torch.float64)
        self.xty = torch.zeros(num_layers, d_model, d_model, dtype=torch.float64)
        self.x_sum = torch.zeros(num_layers, d_model, dtype=torch.float64)
        self.y_sum = torch.zeros(num_layers, d_model, dtype=torch.float64)
        self.count = torch.zeros(num_layers, dtype=torch.float64)

    def update(self, layer: int, x: torch.Tensor, y: torch.Tensor) -> None:
        """``x`` and ``y`` are ``(tokens, d_model)``, already masked to scored positions."""

        xd, yd = x.double(), y.double()
        self.xtx[layer] += (xd.T @ xd).cpu()
        self.xty[layer] += (xd.T @ yd).cpu()
        self.x_sum[layer] += xd.sum(0).cpu()
        self.y_sum[layer] += yd.sum(0).cpu()
        self.count[layer] += float(x.shape[0])

    def solve(self, *, ridge: float = 1e-6) -> LinearReplacement:
        """Centred normal equations, one layer at a time.

        The ridge is relative to each layer's own scale (a fraction of
        ``trace(X^T X)/d``), not absolute, because block-input scale varies by a
        factor of seven across ProGen3's depth and one absolute constant would be
        a different regulariser at every layer.
        """

        if float(self.count.min()) <= self.d_model:
            raise ValueError(
                f"a layer saw {float(self.count.min()):.0f} tokens for {self.d_model} "
                "features; the normal equations are underdetermined and the fit "
                "would be memorising the cohort rather than estimating a map"
            )
        weights, x_means, y_means = [], [], []
        for layer in range(self.num_layers):
            n = self.count[layer]
            mu_x = self.x_sum[layer] / n
            mu_y = self.y_sum[layer] / n
            centred_xtx = self.xtx[layer] - n * torch.outer(mu_x, mu_x)
            centred_xty = self.xty[layer] - n * torch.outer(mu_x, mu_y)
            scale = float(torch.diagonal(centred_xtx).mean())
            regularised = centred_xtx + ridge * scale * torch.eye(self.d_model, dtype=torch.float64)
            weights.append(torch.linalg.solve(regularised, centred_xty))
            x_means.append(mu_x)
            y_means.append(mu_y)
        return LinearReplacement(
            torch.stack(weights).float(),
            torch.stack(x_means).float(),
            torch.stack(y_means).float(),
        )


# -------------------------------------------------- a trained transcoder, spliced


class TranscoderReplacement:
    """A locally trained CLT or PLT, callable one MoE block at a time.

    The faithfulness stage substitutes a block's output through
    ``moe_intercept``, which sees one layer at a time and in layer order. A PLT
    fits that shape directly. A **CLT does not**, because layer ``t``'s
    reconstruction needs the latents of every source ``s <= t`` -- which is why
    the trained checkpoints had no path through that stage at all, and why the
    only thing the first four runs could deliver was reconstruction NMSE, the one
    quantity EXP-R2-132 established is not sufficient.

    The resolution is that ``s <= t`` for every pair by construction, so a
    forward pass has already encoded every source it needs by the time a target
    fires. This accumulates them and :meth:`decode_target` refuses if one is
    missing rather than quietly reconstructing from a subset.

    **The accumulator clears itself when layer 0 fires**, because that is what
    beginning a forward pass *is* on this model: MoE blocks execute in layer
    order and every pass starts at layer 0. That is the same fact the
    accumulation already depends on for correctness, so using it here adds no
    new assumption -- whereas requiring every caller to remember an explicit
    reset would add a way to be silently wrong. Stale latents from a previous
    batch usually differ in token count and would raise on the matmul, but two
    batches of equal length would not, and the reconstruction would quietly mix
    them. :meth:`reset` remains available for a caller that wants to be explicit.
    """

    def __init__(self, model: Transcoder) -> None:
        self.model = model.eval()
        self.config = model.config
        self._latents: dict[int, torch.Tensor] = {}

    @property
    def num_layers(self) -> int:
        return self.config.num_layers

    def to(self, device: torch.device | str) -> TranscoderReplacement:
        self.model = self.model.to(device=device, dtype=torch.float32)
        return self

    def reset(self) -> None:
        self._latents.clear()

    @torch.no_grad()
    def __call__(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        if layer == 0:
            self._latents.clear()
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).float()
        pre, mean, std = self.model.encode_layer(layer, flat)
        self._latents[layer] = topk_relu(pre, self.config.k)
        reconstruction = self.model.decode_target(layer, self._latents, mean, std)
        return reconstruction.reshape(shape).to(x.dtype)


def load_trained_transcoder(path: Path) -> tuple[TranscoderReplacement, dict[str, Any]]:
    """Read a checkpoint written by ``17_train_transcoder.py``.

    Separate from :func:`load_replacement` because they are different objects:
    that one reads a third-party Lightning checkpoint carrying an embedded
    backbone, this one reads ours, which carries none. Conflating them behind one
    function would mean guessing which is on disk, and a wrong guess produces a
    shape error a long way from its cause.
    """

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    missing = sorted({"config", "state_dict"} - set(checkpoint))
    if missing:
        raise RuntimeError(
            f"{path} is not a checkpoint from 17_train_transcoder.py (missing "
            f"{missing}). A released ProGenMech checkpoint is read by "
            "load_replacement instead."
        )
    recorded = checkpoint["config"]
    config = TranscoderConfig(
        num_layers=int(recorded["num_layers"]),
        d_model=int(recorded["d_model"]),
        d_hidden=int(recorded["d_hidden"]),
        k=int(recorded["k"]),
        auxk=int(recorded["auxk"]),
        dead_steps=int(recorded["dead_steps"]),
        aux_weight=float(recorded["aux_weight"]),
        cross_layer=recorded["architecture"] == "CLT",
    )
    model = Transcoder(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return TranscoderReplacement(model), recorded


# ------------------------------------------------------- the released transcoder


class PerLayerTranscoder:
    """ProGenMech's released per-layer transcoder, re-implemented from its weights.

    Their class lives in a CC BY-NC-ND tree and imports ``pytorch_lightning``,
    which is absent here and not installable in the offline pods. The weights
    themselves need neither: ``torch.load(weights_only=True)`` reads them once
    ``argparse.Namespace`` is allowlisted. So the forward pass is restated here,
    line for line against ``training_transcoder/plt_model.py``:

        centre and scale over the feature axis, subtract ``b_pre``, encode,
        keep the top ``k`` pre-activations under ReLU, decode, add ``b_pre``,
        undo the scaling.

    The scale is ``Tensor.std``'s **unbiased** estimate and the epsilon is added
    to it rather than inside the square root, because that is what their code
    does and this is a re-implementation, not an improvement. Each layer is
    independent, so a per-layer hook reproduces it exactly.
    """

    def __init__(self, state: dict[str, torch.Tensor], hyperparameters: Any) -> None:
        self.num_layers = int(hyperparameters.num_layers)
        self.d_model = int(hyperparameters.d_model)
        self.d_hidden = int(hyperparameters.d_hidden)
        self.k = int(hyperparameters.k)
        expected = {"stats_last_nonzero"}
        for layer in range(self.num_layers):
            expected |= {
                f"encoders.{layer}.weight",
                f"encoders.{layer}.bias",
                f"decoders.{layer}",
                f"b_pre.{layer}",
            }
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        if missing or unexpected:
            raise RuntimeError(
                "released transcoder state dict does not match the declared "
                f"architecture (L={self.num_layers}, d_model={self.d_model}, "
                f"d_hidden={self.d_hidden}, k={self.k}).\n"
                f"  missing ({len(missing)}): {missing}\n"
                f"  unexpected ({len(unexpected)}): {unexpected}"
            )
        self.encoder_weight = [state[f"encoders.{i}.weight"] for i in range(self.num_layers)]
        self.encoder_bias = [state[f"encoders.{i}.bias"] for i in range(self.num_layers)]
        self.decoder = [state[f"decoders.{i}"] for i in range(self.num_layers)]
        self.b_pre = [state[f"b_pre.{i}"] for i in range(self.num_layers)]
        for name, tensors, shape in (
            ("encoders", self.encoder_weight, (self.d_hidden, self.d_model)),
            ("decoders", self.decoder, (self.d_hidden, self.d_model)),
            ("b_pre", self.b_pre, (self.d_model,)),
        ):
            wrong = [i for i, t in enumerate(tensors) if tuple(t.shape) != shape]
            if wrong:
                raise RuntimeError(
                    f"released transcoder {name} at layers {wrong} do not have "
                    f"shape {shape}; the hyper-parameters and the weights disagree"
                )

    def to(self, device: torch.device) -> PerLayerTranscoder:
        for group in (self.encoder_weight, self.encoder_bias, self.decoder, self.b_pre):
            group[:] = [tensor.to(device=device, dtype=torch.float32) for tensor in group]
        return self

    def __call__(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        source = x.float()
        mu = source.mean(dim=-1, keepdim=True)
        centred = source - mu
        std = centred.std(dim=-1, keepdim=True)
        normalised = centred / (std + 1e-5) - self.b_pre[layer]
        pre = F.linear(normalised, self.encoder_weight[layer], self.encoder_bias[layer])
        top = torch.topk(pre, k=self.k, dim=-1, sorted=False)
        latents = torch.zeros_like(pre).scatter_(-1, top.indices, F.relu(top.values))
        recon = latents @ self.decoder[layer] + self.b_pre[layer]
        return (recon * std + mu).to(x.dtype)


def load_replacement(path: Path) -> tuple[PerLayerTranscoder, dict[str, torch.Tensor], Any]:
    """The released checkpoint's transcoder weights and its embedded backbone."""

    torch.serialization.add_safe_globals([argparse.Namespace])
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    hyperparameters = checkpoint["hyper_parameters"]["args"]
    transcoder = PerLayerTranscoder(
        {key[len("plt.") :]: value for key, value in state.items() if key.startswith("plt.")},
        hyperparameters,
    )
    backbone = {
        key[len("progen3_model.") :]: value
        for key, value in state.items()
        if key.startswith("progen3_model.")
    }
    return transcoder, backbone, hyperparameters
