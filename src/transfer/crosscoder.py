"""A two-model Crosscoder, and the readout that separates shared from model-specific latents.

**What this is for.** R2.4's same-input Model Diffing unit compares
``Llama-2-7b-hf`` (pre-adaptation) with ``ProLLaMA_Stage_1`` (protein-adapted) on
*identical* rendered inputs. Stage 25 has already answered the question a
Crosscoder is not needed for -- whether an affine or orthogonal map removes the
difference -- and the answer is that it does not, in either mode. What no stage
can yet do is say *which* directions the difference lives in and whether they are
shared between the two checkpoints or specific to one. That is what a Crosscoder
is: one dictionary trained jointly over both checkpoints' activations at the same
position of the same record, with a shared latent space and per-model decoders.

**The published formulation this follows.** Lindsey et al., *Sparse Crosscoders
for Cross-Layer Features and Model Diffing* (Transformer Circuits, Oct 2024), §2
and §4; and Mishra-Sharma et al., *Insights on Crosscoder Model Diffing*
(Transformer Circuits, 2025). Latents are computed from a sum over the models'
encoders, each model gets its own decoder, and the sparsity penalty is the
**L1 of per-model decoder norms** -- ``sum_i f_i(x) sum_m ||W^m_{dec,i}||`` --
which is the term that produces model-exclusive latents at all. Both notes are
explicit that summing the two models' decoder norms *before* multiplying by the
activation is the design choice that creates exclusivity, and that taking one
norm over the concatenation does not. The readout is the relative decoder norm
``||W^B_{dec,i}|| / (||W^A_{dec,i}|| + ||W^B_{dec,i}||)``, whose distribution is
trimodal on a base/finetune pair: a peak near 0 (base-specific), one near 1
(adapted-specific) and one near 0.5 (shared).

**Four deviations, each because this programme's pair is not theirs.**

1. **TopK activation, and the decoder-norm L1 kept beside it.** Their exposition
   uses ReLU with an L1 activation penalty; the 2025 note says TopK may be
   substituted. It is substituted here, because every dictionary R2.3 fitted on
   this lineage is a TopK dictionary at ``k`` 32 and a Crosscoder whose L0 was
   set by a penalty could not be read against them. But TopK alone carries **no**
   exclusivity pressure -- with no penalty the decoder norms are free and the
   relative-norm histogram has no reason to be trimodal -- so the published L1 of
   per-model decoder norms is retained as a separate term at a declared
   coefficient. ``--decoder-norm-penalty 0`` recovers the pure-TopK crosscoder and
   is a control on how much of the trimodal structure the penalty itself creates.
   The term is **gauge-invariant** under the reparametrisation that TopK admits
   (scale a latent's decoder by ``c`` and its encoder row by ``1/c``: the
   reconstruction, the TopK selection and ``f_i ||W_dec,i||`` are all unchanged),
   and so is the relative decoder norm, since both models' norms scale by the
   same ``c``. The readout therefore does not depend on a gauge nobody fixed.

2. **Per-site latent banks rather than one dictionary at one layer.** Their model
   diffing experiment trains at one middle layer. R2.4's operative admission rule
   is *per layer* -- a diff may be reported at layer ``l`` only where both cells'
   dictionaries clear that layer's own effective dimension -- so a single-layer
   object cannot answer it. This trains a bank of independent two-model
   crosscoders, one per declared site, in one loop and under one objective.
   Latent ``i`` at site ``l`` and latent ``i`` at site ``l'`` are different
   features; nothing is shared across sites. That is ``L`` crosscoders sharing a
   training loop, not a cross-layer crosscoder, and it is deliberately *not* the
   acausal cross-layer variant of their §2: a cross-layer object would fold the
   per-layer admissibility question away.

3. **Per-(site, model) normalisation scalars, frozen before training.** They
   "separately normalize the activations of each layer prior to training ... so
   that each layer contributes comparably to the loss". Here the same is required
   *across models* and for a stronger reason: the relative decoder norm is a ratio
   of two decoder norms, so if one checkpoint's activations are twice the other's
   at a site, every latent at that site reads as specific to the larger one and
   the readout measures scale. One scalar per (site, model), estimated on a
   declared warm-up prefix of the training stream as ``E||x||_2 / sqrt(d)``, then
   frozen, recorded in the checkpoint and in the artefact. The two checkpoints'
   scale ratio is therefore **removed from the readout and reported beside it**,
   rather than silently dominating it.

4. **NMSE rather than MSE.** The repository's transcoders report per-layer NMSE
   (Appendix B rule 21: a scale-free statistic, because this backbone's residual
   scale grows by orders of magnitude with depth). The same is used here, per
   (site, model), so a Crosscoder's reconstruction number at a site is directly
   comparable with the per-layer transcoder number R2.3 already published for the
   same checkpoint at the same site.

**What is not implemented, and is not a deviation.** The 2025 note's mitigation
for polysemantic exclusive latents -- a designated subset of weight-shared latents
at a reduced sparsity penalty -- addresses the *qualitative interpretability* of
exclusive features. This unit counts and categorises latents and reads none of
them, so the mitigation buys nothing it could measure. It is recorded here as the
next refinement rather than built speculatively.

**The identical-input guarantee is not this module's.** It belongs to
``25_model_diffing_baselines.py``, whose ``assert_identical_tokenizers``,
``assert_comparable_shape``, ``assert_identical_batches`` and ``paired_capture``
are imported by the stage unchanged. This module never touches a checkpoint; it
consumes pairs of ``(n_sites, tokens, d_model)`` tensors and does not know or ask
where they came from. That is what lets the whole instrument be certified on
synthetic data with known ground truth before it is pointed at 6.74B parameters
of anything.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .transcoders import (
    FiringCensus,
    MatchedTraining,
    TrainingRecord,
    compare_matched_training,
    live_latents_per_layer,
    topk_relu,
)

#: The two pairings every quantity in this unit may be computed under. ``true``
#: is the measurement; ``shuffled`` is the null and is never reported instead of
#: it.
PAIRINGS: tuple[str, str] = ("true", "shuffled")

#: The two roles, in the order every stacked tensor and every readout indexes
#: them. ``base`` is index 0 and ``adapted`` is index 1, so the relative decoder
#: norm below runs 0 (base-specific) to 1 (adapted-specific) and a reader does
#: not have to look up which way round it is.
ROLES: tuple[str, str] = ("base", "adapted")

#: The four buckets the relative decoder norm is read into. ``intermediate`` is
#: not decoration: with two cuts there is a band between "exclusive" and "shared"
#: that belongs to neither, and folding it into either would make the counts sum
#: to the dictionary while meaning something else.
CATEGORIES: tuple[str, ...] = (
    "base_specific",
    "shared",
    "adapted_specific",
    "intermediate",
)

#: Bin edges of the relative-decoder-norm histogram carried per site, so a reader
#: can re-cut the categories from the artefact instead of trusting the two cuts
#: this run happened to declare.
HISTOGRAM_BINS = 20

#: A key whose name ends in this suffix **must** be one value per fitted site.
#: Enforced recursively over a whole artefact by :func:`assert_per_layer_fields`,
#: and enforced by convention rather than by a list so that a field added later
#: cannot escape the check by not being registered. The defect this exists to
#: stop is on the record: ``Transcoder.objective`` collapsed a
#: ``(num_layers, d_hidden)`` dead mask into one cross-layer scalar before
#: anything downstream saw it, so R2.4's basis criterion -- stated per layer --
#: had never been evaluated at the resolution it was specified at, in either
#: direction (EXP-R2-203).
PER_SITE_SUFFIX = "_per_site"

#: Fields the readout must carry per site. Presence is checked as well as shape:
#: a per-layer quantity that is simply absent is the same failure as one that was
#: averaged, and it is easier to miss.
REQUIRED_PER_SITE_FIELDS: tuple[str, ...] = (
    "live_latents_per_site",
    "nmse_per_site",
    "relative_norm_histogram_per_site",
    "category_counts_per_site",
)

_TINY = 1e-12


# --------------------------------------------------------------------- config


@dataclass(frozen=True)
class CrosscoderConfig:
    """Everything that defines a Crosscoder, recorded in every checkpoint."""

    #: Backbone layer indices this Crosscoder carries parameters for, ascending
    #: and unique. **Not** necessarily every layer of the model: R2.4's admission
    #: rule restricts where a protein diff may be reported to six layers, and
    #: fitting only those cuts the parameter count by the ratio of the two sets.
    sites: tuple[int, ...]
    d_model: int
    d_hidden: int
    k: int
    auxk: int
    dead_steps: int
    aux_weight: float = 1.0 / 32.0
    #: Coefficient of the published L1-of-per-model-decoder-norms term. Zero is a
    #: legitimate setting and is the pure-TopK control, not a disabled feature.
    decoder_norm_penalty: float = 0.0
    #: ``true`` or ``shuffled``. Part of the config and not only of the run,
    #: because a shuffled-pairing Crosscoder is a different fitted object and a
    #: checkpoint that did not say so could be read as the measurement.
    pairing: str = "true"
    #: What the two roles are, by name, so an artefact says which checkpoint is
    #: index 0 without a reader consulting the command line.
    role_names: tuple[str, str] = ROLES

    def __post_init__(self) -> None:
        if len(self.sites) == 0:
            raise ValueError("a Crosscoder with no sites has nothing to fit")
        if sorted(set(self.sites)) != list(self.sites):
            raise ValueError(
                f"sites must be ascending and unique; got {list(self.sites)}"
            )
        if min(self.sites) < 0:
            raise ValueError("a site is a backbone layer index and cannot be negative")
        for name, value in (
            ("d_model", self.d_model),
            ("d_hidden", self.d_hidden),
            ("k", self.k),
            ("auxk", self.auxk),
            ("dead_steps", self.dead_steps),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive; got {value}")
        if self.k > self.d_hidden:
            raise ValueError(
                f"k={self.k} exceeds d_hidden={self.d_hidden}: TopK cannot select "
                "more latents than the dictionary has"
            )
        if self.decoder_norm_penalty < 0.0:
            raise ValueError("the decoder-norm penalty coefficient cannot be negative")
        if self.pairing not in PAIRINGS:
            raise ValueError(f"pairing must be one of {PAIRINGS}; got {self.pairing!r}")

    @property
    def n_sites(self) -> int:
        return len(self.sites)

    def n_parameters(self) -> int:
        """Trainable parameters, in closed form, so a campaign can be sized before it runs.

        Two encoders and two decoders per site, against a per-layer transcoder's
        one of each: a Crosscoder is **twice** the dictionary at the same
        ``(d_model, d_hidden)`` and per site, which is the arithmetic that decides
        whether a width fits in one card's memory.
        """

        per_site = (
            2 * self.d_model * self.d_hidden  # encoders
            + self.d_hidden  # encoder bias
            + 2 * self.d_hidden * self.d_model  # decoders
            + 2 * self.d_model  # decoder biases
        )
        return self.n_sites * per_site

    def record(self) -> dict[str, Any]:
        return {
            "architecture": "CROSSCODER",
            "roles": list(self.role_names),
            "sites": list(self.sites),
            "n_sites": self.n_sites,
            "d_model": self.d_model,
            "d_hidden": self.d_hidden,
            "k": self.k,
            "auxk": self.auxk,
            "dead_steps": self.dead_steps,
            "aux_weight": self.aux_weight,
            "decoder_norm_penalty": self.decoder_norm_penalty,
            "pairing": self.pairing,
            "n_parameters": self.n_parameters(),
            "active_latents_per_token": self.k,
            "active_fraction_of_dictionary": self.k / self.d_hidden,
        }


#: Why a site's fitted dictionary does not depend on which other sites were in
#: the run. This decides the shape of a campaign -- whether a Crosscoder licensed
#: to report a diff at one or two layers must nonetheless be trained across the
#: whole stack -- so it is stated once here and copied into every artefact rather
#: than left as an inference from the architecture.
SITE_INDEPENDENCE_NOTE = (
    "the sites of this Crosscoder are parameter-disjoint: site l has its own "
    "encoders, decoders, biases and normalisation constants, and shares nothing "
    "with site l'. The objective is a plain sum of per-site terms, so the "
    "gradient of the loss with respect to site l's parameters depends only on "
    "site l. Two mechanisms could nonetheless have coupled the sites and both are "
    "removed rather than argued away: the initialisation is drawn PER SITE from a "
    "generator keyed to that site's backbone layer index rather than from one "
    "draw over the stacked tensor, and the gradient clip is applied PER SITE "
    "rather than over one global norm. AdamW's step and its decoupled weight "
    "decay are already per parameter. The consequence a campaign needs: training "
    "the full stack and reporting only at admissible layers yields THE SAME "
    "fitted dictionary at those layers as training the admissible layers alone, "
    "so the choice between them is economics and not science, and the narrow run "
    "is preferred because it is cheaper. Checked rather than asserted -- "
    "tests/test_crosscoder.py fits one site alone and inside a wider run and "
    "compares the decoder norms. What is NOT independent across sites is the "
    "cohort: every site sees the same records, positions and pairing "
    "permutation, which is what makes a within-run cross-site comparison "
    "meaningful and is deliberate"
)

#: When a per-site reconstruction number may be read against another one, carried
#: at every site beside the number it qualifies rather than in a caveat elsewhere.
#:
#: This unit's recurring failure has been readings whose limits were written down
#: somewhere other than where the number appears, so this string is emitted into
#: each per-site record and not only into the artefact's limitations block.
#:
#: **The rule is stated positively and narrowly**, because the axis list is not
#: closed by argument: the achievable NMSE depends on the effective dimension of
#: the cloud being reconstructed, and that dimension varies across modes, across
#: the two checkpoints, and across layers. Enumerating prohibitions invites the
#: next axis to be missed; naming the one valid comparison does not.
NMSE_COMPARABILITY_NOTE = (
    "VALID: comparison between dictionaries fitted to THE SAME activation cloud "
    "-- same mode, same role, same site -- which for this object means against "
    "R2.3's per-layer transcoder figure at the same site and the same width. That "
    "is the comparison this number was built for and the effective dimension does "
    "not confound it. FORBIDDEN: comparison across modes, across ROLES, or across "
    "SITES, including between the two halves of the [base, adapted] pair beside "
    "this note and between the entries of any per-site NMSE vector. The "
    "achievable NMSE depends on how many directions the cloud occupies, so a "
    "difference read across any of those axes is a difference in what the data "
    "does and not in how well the dictionary describes it. Measured on this "
    "lineage's four L8 baseline cells: ACROSS MODES, protein reads 0.0670 and "
    "0.0585 at layers 27-28 against text's 0.5739 and 0.5369, roughly ninefold "
    "better, on clouds of markedly lower dimension. ACROSS ROLES within text, "
    "median r99 3,670 against 2,954 with held-out NMSE sums 16.08 against 5.71 -- "
    "the larger cloud reconstructing worse -- and the protein pair moves the same "
    "way, 2,588 against 2,709 with sums 3.72 against 5.46. ACROSS SITES within a "
    "cell, the Spearman rank correlation between r99 and NMSE over the interior "
    "layers is +0.98, +0.82 and +0.73 in three of the four cells. The "
    "relationship is DIRECTIONAL AND NOT PROPORTIONAL, and base/text is an "
    "exception at -0.04 in its interior, which is why this is a prohibition and "
    "not a correction: there is no factor to divide out. r99_effective_dimension "
    "is carried per role beside this number and index-aligned with it, so the "
    "dependence is visible here rather than inferable"
)

#: What a permutation within one batch is and is not. Stage 25 declares the same
#: bound for the same reason and this is deliberately the same sentence: a global
#: permutation would need every position of the split resident at once.
SHUFFLE_NOTE = (
    "the null pairs the base checkpoint's position i with the adapted "
    "checkpoint's position pi(i) for a permutation drawn per batch from the "
    "declared seed, so token-level correspondence is destroyed while both "
    "marginal distributions are exactly preserved. The permutation is WITHIN a "
    "batch and not across the whole stream, because a global permutation would "
    "need every position resident at once. A batch holds unrelated records drawn "
    "from a seeded shuffled stream, so what the null retains is between-batch "
    "covariance and nothing finer -- which makes it CONSERVATIVE: it can only "
    "make the null look more like the measurement than a global permutation "
    "would, and therefore only understate the gap between the two"
)


# ---------------------------------------------------------------- the network


class Crosscoder(nn.Module):
    """One dictionary over two checkpoints' activations at the same position.

    Shapes throughout: activations arrive as ``(n_sites, tokens, d_model)`` per
    role, latents are ``(n_sites, tokens, d_hidden)``, and every stacked
    two-role tensor puts the role axis **first** so that ``[0]`` is the base and
    ``[1]`` is the adapted checkpoint.
    """

    def __init__(self, config: CrosscoderConfig, *, init_seed: int = 0):
        super().__init__()
        self.config = config
        sites, d, h = config.n_sites, config.d_model, config.d_hidden

        # U(-1/sqrt(fan_in), 1/sqrt(fan_in)) on each side, which is exactly what
        # `nn.init.kaiming_uniform_(a=sqrt(5))` reduces to and is what
        # `Transcoder` initialises its decoders with -- written out because the
        # encoder's fan-in here is `d_model` while the tensor's second axis is
        # `d_hidden`, and the library helper would read the wrong one.
        bound_encoder = 1.0 / math.sqrt(d)
        bound_decoder = 1.0 / math.sqrt(h)
        self.W_enc = nn.Parameter(torch.empty(2, sites, d, h))
        self.b_enc = nn.Parameter(torch.zeros(sites, h))
        self.W_dec = nn.Parameter(torch.empty(2, sites, h, d))
        self.b_dec = nn.Parameter(torch.zeros(2, sites, d))
        # Drawn **per site, from a generator keyed to that site's backbone layer
        # index** rather than from one draw over the stacked tensor. That is what
        # makes site independence exact rather than approximate: a run over
        # layers (27, 28) and a run over all 32 layers initialise layer 27
        # identically, so the two produce the same fitted dictionary there. See
        # :func:`clip_per_site_grad_norm_` for the other half of the property and
        # :meth:`site_independence_note` for why it decides a campaign's shape.
        with torch.no_grad():
            for index, layer in enumerate(config.sites):
                generator = torch.Generator().manual_seed(
                    (int(init_seed) * 1_000_003 + int(layer)) % (2**63 - 1)
                )
                for role in range(2):
                    self.W_enc[role, index].uniform_(
                        -bound_encoder, bound_encoder, generator=generator
                    )
                    self.W_dec[role, index].uniform_(
                        -bound_decoder, bound_decoder, generator=generator
                    )

        # Frozen per-(role, site) normalisation scalars. A separate flag rather
        # than a sentinel value: `scale` of all ones is a legitimate setting on
        # data that is already normalised, and "not yet estimated" has to be
        # distinguishable from it or a run that forgot the warm-up would train
        # silently on unequalised scales and produce a readout of them.
        self.register_buffer("scale", torch.ones(2, sites))
        self.register_buffer("scale_is_set", torch.zeros((), dtype=torch.bool))
        # Steps since each latent last fired, per site. Same buffer, same
        # semantics and same name as `Transcoder`, so
        # `transcoders.live_latents_per_layer` and `transcoders.FiringCensus`
        # read a Crosscoder checkpoint without a second definition of "live".
        self.register_buffer("silent_steps", torch.zeros(sites, h, dtype=torch.long))

    # ------------------------------------------------------------- the scales

    def set_scales(self, scales: torch.Tensor) -> None:
        """Freeze the per-(role, site) normalisation constants.

        Called once, before the first optimiser step, from
        :func:`estimate_scales`. Refuses a second call: the scales are part of
        what the fitted decoders mean, and changing them mid-run would silently
        rescale every decoder norm the readout is computed from.
        """

        if bool(self.scale_is_set):
            raise RuntimeError(
                "this Crosscoder's normalisation scales are already frozen; "
                "re-setting them mid-run would rescale every decoder norm the "
                "specificity readout is computed from"
            )
        expected = (2, self.config.n_sites)
        if tuple(scales.shape) != expected:
            raise ValueError(
                f"expected {expected} normalisation scales (role, site), got "
                f"{tuple(scales.shape)}"
            )
        values = scales.to(device=self.scale.device, dtype=self.scale.dtype)
        if not bool(torch.isfinite(values).all()) or bool((values <= 0).any()):
            raise ValueError(
                "every normalisation scale must be finite and positive; a "
                "non-positive one means a site whose activations were all zero, "
                "which is a capture failure and not a scale"
            )
        self.scale.copy_(values)
        self.scale_is_set.fill_(True)

    def _scaled(
        self, base: torch.Tensor, adapted: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(self.scale_is_set):
            raise RuntimeError(
                "this Crosscoder's normalisation scales have not been estimated. "
                "Without them the relative decoder norm reads the ratio of the two "
                "checkpoints' activation scales rather than their feature "
                "specificity; call estimate_scales() before training"
            )
        self._check_shape(base, "base")
        self._check_shape(adapted, "adapted")
        if base.shape != adapted.shape:
            raise ValueError(
                f"the two roles' activations must have identical shapes; got "
                f"{tuple(base.shape)} and {tuple(adapted.shape)}. A Crosscoder is "
                "defined only on the same positions of the same records"
            )
        return (
            base / self.scale[0].view(-1, 1, 1),
            adapted / self.scale[1].view(-1, 1, 1),
        )

    def _check_shape(self, value: torch.Tensor, role: str) -> None:
        if value.ndim != 3:
            raise ValueError(
                f"expected (n_sites, tokens, d_model) {role} activations, got shape "
                f"{tuple(value.shape)}"
            )
        if value.shape[0] != self.config.n_sites or value.shape[2] != self.config.d_model:
            raise ValueError(
                f"this Crosscoder was built for {self.config.n_sites} sites of "
                f"{self.config.d_model} dimensions and was handed {role} activations "
                f"of {value.shape[0]} x {value.shape[2]}"
            )

    # -------------------------------------------------------- encode / decode

    def encode(
        self, base: torch.Tensor, adapted: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Latents, pre-activations and the scaled targets they were read from.

        ``f = TopK-ReLU(W_enc^A a^A + W_enc^B a^B + b_enc)``, per site. The sum
        over roles is the crosscoder's defining property: one latent activation
        explains a position in both checkpoints, so "the same feature" means "it
        fires on the same datapoint", not "it points the same way".
        """

        scaled_base, scaled_adapted = self._scaled(base, adapted)
        pre = (
            torch.bmm(scaled_base, self.W_enc[0])
            + torch.bmm(scaled_adapted, self.W_enc[1])
            + self.b_enc[:, None, :]
        )
        latents = topk_relu(pre, self.config.k)
        return latents, pre, torch.stack([scaled_base, scaled_adapted])

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Both roles' reconstructions in the scaled space, ``(2, n_sites, tokens, d)``."""

        return torch.stack(
            [
                torch.bmm(latents, self.W_dec[role]) + self.b_dec[role][:, None, :]
                for role in range(2)
            ]
        )

    def reconstruct(
        self, base: torch.Tensor, adapted: torch.Tensor
    ) -> torch.Tensor:
        """Both roles' reconstructions in the **raw** activation space.

        The objective is computed in the scaled space, where the two roles are
        comparable; this is the only place the scaling is undone, and it exists so
        that a caller who wants a reconstruction in the units the backbone
        produced does not have to re-derive the constants.
        """

        latents, _, _ = self.encode(base, adapted)
        return self.decode(latents) * self.scale[:, :, None, None]

    def forward(self, base: torch.Tensor, adapted: torch.Tensor) -> torch.Tensor:
        return self.reconstruct(base, adapted)

    # ----------------------------------------------------------------- losses

    def objective(
        self, base: torch.Tensor, adapted: torch.Tensor, *, training: bool
    ) -> dict[str, Any]:
        """Per-(site, role) NMSE, the decoder-norm penalty and the revival term.

        Every per-site quantity is returned **as a vector over sites** and is
        never reduced here. The scalars beside them are sums and are labelled as
        such; a caller that wants a per-layer answer must not be able to get one
        by dividing.
        """

        latents, pre, targets = self.encode(base, adapted)
        reconstruction = self.decode(latents)

        variance = targets.var(dim=(2, 3), unbiased=False)
        nmse = ((reconstruction - targets) ** 2).mean(dim=(2, 3)) / (variance + 1e-8)

        # The published sparsity term: per-latent decoder norms summed over ROLES
        # first, then weighted by the activation. Summing over roles before
        # multiplying is what makes a latent that writes to one model cheaper than
        # one that writes to both, and is therefore the entire mechanism behind
        # the exclusive/shared separation the readout reports.
        # Through the squared sum with a floor rather than `.norm()`: the L2 norm's
        # derivative is undefined at zero, and a latent whose decoder the penalty
        # has driven to exactly zero would put a NaN into the gradient of a run
        # that is otherwise healthy. The floor is far below any norm that survives
        # training and does not change the term's value anywhere it is defined.
        decoder_norm = (self.W_dec**2).sum(dim=3).clamp_min(_TINY).sqrt()
        per_latent_cost = decoder_norm.sum(dim=0)
        penalty_per_site = (latents * per_latent_cost[:, None, :]).sum(dim=2).mean(dim=1)

        fired_counts = (latents > 0).sum(dim=1)

        aux = torch.zeros((), device=latents.device, dtype=nmse.dtype)
        dead_per_site: list[int] | None = None
        if training:
            fired = fired_counts > 0
            self.silent_steps = torch.where(
                fired, torch.zeros_like(self.silent_steps), self.silent_steps + 1
            )
            dead = self.silent_steps > self.config.dead_steps
            per_site_dead = dead.sum(dim=1)
            dead_per_site = [int(value) for value in per_site_dead]
            if int(per_site_dead.sum()):
                residual = (targets - reconstruction).detach()
                for site in range(self.config.n_sites):
                    site_dead = dead[site]
                    if not bool(site_dead.any()):
                        continue
                    k_aux = int(min(self.config.auxk, int(site_dead.sum())))
                    masked = pre[site].masked_fill(~site_dead, float("-inf"))
                    revived = topk_relu(masked, k_aux)
                    # Through BOTH roles' decoders, because a dead latent in a
                    # Crosscoder is dead for both models and reviving it against
                    # one alone would bias every revived latent toward that role
                    # -- straight into the readout this object exists for. No
                    # de-normalisation is needed here, unlike `Transcoder`: the
                    # residual and the decoder output are both in the scaled
                    # space, which is the asymmetry that module had to repair.
                    for role in range(2):
                        predicted = (
                            revived @ self.W_dec[role, site] + self.b_dec[role, site]
                        )
                        aux = aux + F.mse_loss(predicted, residual[role, site])

        loss = (
            nmse.sum()
            + self.config.decoder_norm_penalty * penalty_per_site.sum()
            + self.config.aux_weight * aux
        )
        return {
            "loss": loss,
            "nmse_sum": nmse.sum().detach(),
            # (role, site). Kept two-dimensional deliberately: the two roles'
            # reconstruction quality can diverge at a site and a sum over roles
            # would hide exactly the asymmetry a diff is about.
            "nmse_role_by_site": nmse.detach(),
            "nmse_per_site": nmse.sum(dim=0).detach(),
            "decoder_penalty_per_site": penalty_per_site.detach(),
            "aux": aux.detach(),
            "n_dead_per_site": dead_per_site,
            "fired_per_latent": fired_counts.detach(),
            "active_fraction": float((latents > 0).float().mean()),
        }

    # ------------------------------------------------------------- the basis

    def live_latents_per_site(self, dead_steps: int | None = None) -> list[int]:
        """Live latents at each site under this checkpoint's own dead-latent counter."""

        return live_latents_per_layer(
            self.silent_steps,
            self.config.dead_steps if dead_steps is None else dead_steps,
        )



# ----------------------------------------------------- normalisation estimate


def estimate_scales(
    batches: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]],
    *,
    n_sites: int,
    d_model: int,
    n_batches: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """One scalar per (role, site): ``E||x||_2 / sqrt(d_model)``, on a warm-up prefix.

    So that a scaled activation has mean L2 norm ``sqrt(d_model)``, which is the
    scale an isotropic standard normal would have and therefore a unit that means
    the same thing at every site of both checkpoints.

    ``batches`` is a **factory**, called here to open its own iterator: the
    warm-up must not consume the stream the training loop will then read, or the
    two would see different data and the scales would be estimated on positions
    the fit never sees.
    """

    if n_batches < 1:
        raise ValueError("estimating a scale needs at least one batch")
    totals = torch.zeros(2, n_sites, dtype=torch.float64)
    counts = 0
    seen = 0
    for pair in batches():
        for role, activations in enumerate(pair):
            if activations.ndim != 3 or activations.shape[0] != n_sites:
                raise ValueError(
                    f"expected ({n_sites}, tokens, {d_model}) activations for role "
                    f"{ROLES[role]}, got {tuple(activations.shape)}"
                )
            if activations.shape[2] != d_model:
                raise ValueError(
                    f"role {ROLES[role]} carries width {activations.shape[2]}, not "
                    f"{d_model}"
                )
            totals[role] += (
                activations.detach().to(torch.float64).norm(dim=2).sum(dim=1).cpu()
            )
        counts += int(pair[0].shape[1])
        seen += 1
        if seen >= n_batches:
            break
    if counts == 0:
        raise RuntimeError(
            "the warm-up pass produced no token positions, so no normalisation "
            "scale exists; refusing to train on unequalised activation scales"
        )
    scales = (totals / counts) / math.sqrt(d_model)
    if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
        raise RuntimeError(
            "a warm-up pass returned a non-positive or non-finite activation norm "
            f"at some site: {scales.tolist()}"
        )
    ratio = (scales[1] / scales[0]).tolist()
    return scales.to(torch.float32), {
        "definition": "E||x||_2 / sqrt(d_model), per (role, site), frozen before "
        "the first optimiser step",
        "warm_up_batches": seen,
        "warm_up_positions": counts,
        # Site-major, one ``[base, adapted]`` pair per site. Role-major would be a
        # length-2 list under a name ending in ``_per_site``, which is exactly the
        # shape :func:`assert_per_layer_fields` exists to catch.
        "scale_per_site": [[float(scales[0, site]), float(scales[1, site])]
                           for site in range(n_sites)],
        # The quantity the normalisation removes from the readout, reported so it
        # is visible rather than erased. A site where the adapted checkpoint's
        # activations are twice the base's is a real difference between the two
        # models; it is simply not a difference the relative decoder norm is
        # capable of expressing, and a reader who is not shown it would take the
        # readout for more than it claims.
        "adapted_over_base_scale_ratio_per_site": ratio,
        "note": (
            "the relative decoder norm is a ratio of two decoder norms, so an "
            "unequalised activation scale would make every latent read as specific "
            "to the larger checkpoint. These constants remove that and are "
            "reported so the scale difference itself is not lost"
        ),
    }


# ------------------------------------------------------------------ the null


def apply_pairing(
    adapted: torch.Tensor, *, pairing: str, generator: torch.Generator
) -> torch.Tensor:
    """The adapted role's positions, paired truly or permuted within the batch.

    The single place the null is applied. A source always emits genuinely paired
    activations -- that is what ``25_model_diffing_baselines.paired_capture``
    guarantees and what this unit's premise rests on -- and the null is a declared
    transformation of them, so there is exactly one definition of what "shuffled"
    means and it cannot drift between the trainer and the evaluation.
    """

    if pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {PAIRINGS}; got {pairing!r}")
    if pairing == "true":
        return adapted
    permutation = torch.randperm(adapted.shape[1], generator=generator)
    return adapted[:, permutation.to(adapted.device), :]


# --------------------------------------------------------------- the readout


def decoder_norms(model: Crosscoder) -> torch.Tensor:
    """``(2, n_sites, d_hidden)`` L2 norms of each latent's decoder in each role.

    Returned on the **host**, as is every readout quantity below. They are
    ``2 x n_sites x d_hidden`` at most -- half a megabyte at the campaign's shape
    -- and the mask they are read against comes from a
    :class:`~src.transfer.transcoders.FiringCensus`, which accumulates on the host
    because it is an int64 count and not a training tensor. One side has to move
    and it is cheaper, and less error-prone, for it to always be this one.
    """

    return model.W_dec.detach().norm(dim=3).cpu()


def relative_decoder_norm(norms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``||W^adapted|| / (||W^base|| + ||W^adapted||)`` per site and latent.

    The construction from Lindsey et al. §4.3 and Mishra-Sharma et al.: 0 is
    base-specific, 1 is adapted-specific, 0.5 is shared. Returns the ratio and a
    mask of where it is **defined** -- a latent whose decoder norms are both zero
    in both roles has no ratio, and returning 0.5 for it would put it in the
    shared peak, which is the one answer it must never give.
    """

    if norms.ndim != 3 or norms.shape[0] != 2:
        raise ValueError(
            f"expected (2, n_sites, d_hidden) decoder norms, got {tuple(norms.shape)}"
        )
    total = norms.sum(dim=0)
    defined = total > _TINY
    ratio = torch.where(
        defined, norms[1] / total.clamp_min(_TINY), torch.zeros_like(total)
    )
    return ratio, defined


def decoder_cosine(model: Crosscoder) -> torch.Tensor:
    """Cosine between the two roles' decoder directions, per site and latent.

    The second half of the published readout. A shared latent whose two decoder
    directions are aligned represents the same concept used the same way; the 2024
    note found a few thousand shared features with low or negative cosine and read
    them as a concept the adapted model uses *differently*, which is a distinct
    finding from a model-specific feature and must not be merged with it.
    """

    return F.cosine_similarity(
        model.W_dec[0].detach(), model.W_dec[1].detach(), dim=2, eps=_TINY
    ).cpu()


def categorise(
    ratio: torch.Tensor, *, exclusive_cut: float, shared_halfwidth: float
) -> torch.Tensor:
    """Category index per site and latent, in the order of :data:`CATEGORIES`.

    ``exclusive_cut`` is the 2025 note's 0.95: a latent is adapted-specific above
    it and base-specific below its reflection. ``shared_halfwidth`` is the band
    around 0.5 the shared peak is read over. The two cuts are declared by the
    caller and travel into the artefact with the counts they produced, and the
    full histogram travels beside them so a reader can re-cut.
    """

    if not 0.5 < exclusive_cut < 1.0:
        raise ValueError(
            f"exclusive_cut must lie in (0.5, 1); got {exclusive_cut}. At or below "
            "0.5 the exclusive and shared bands would overlap"
        )
    if not 0.0 < shared_halfwidth < (exclusive_cut - 0.5):
        raise ValueError(
            f"shared_halfwidth must lie in (0, {exclusive_cut - 0.5}); got "
            f"{shared_halfwidth}. A wider band would overlap the exclusive cuts"
        )
    codes = torch.full(ratio.shape, CATEGORIES.index("intermediate"), dtype=torch.long)
    codes[ratio <= 1.0 - exclusive_cut] = CATEGORIES.index("base_specific")
    codes[ratio >= exclusive_cut] = CATEGORIES.index("adapted_specific")
    codes[(ratio - 0.5).abs() <= shared_halfwidth] = CATEGORIES.index("shared")
    return codes


def site_readout(
    *,
    ratio: torch.Tensor,
    cosine: torch.Tensor,
    defined: torch.Tensor,
    live: torch.Tensor,
    exclusive_cut: float,
    shared_halfwidth: float,
) -> dict[str, Any]:
    """One site's specificity readout, over its live latents only.

    **Live latents only, and that is not a detail.** A latent that never fires has
    whatever decoder norms the initialisation and the penalty left it with, and on
    a dictionary where most latents are dead the histogram would be a picture of
    the initialiser. The live mask is per site and comes from the held-out firing
    census, which is the definition a diff is actually read under.
    """

    for name, value in (("ratio", ratio), ("cosine", cosine), ("live", live)):
        if value.ndim != 1:
            raise ValueError(f"{name} must be one value per latent at one site")
    usable = live & defined
    n_usable = int(usable.sum())
    if n_usable == 0:
        raise RuntimeError(
            "this site has no live latent with a defined relative decoder norm, so "
            "there is nothing to categorise. A readout over zero latents is not a "
            "smaller version of this statistic"
        )
    selected = ratio[usable]
    codes = categorise(
        selected, exclusive_cut=exclusive_cut, shared_halfwidth=shared_halfwidth
    )
    counts = {
        name: int((codes == index).sum()) for index, name in enumerate(CATEGORIES)
    }
    shared_mask = codes == CATEGORIES.index("shared")
    shared_cosine = cosine[usable][shared_mask]
    histogram = torch.histc(selected.float(), bins=HISTOGRAM_BINS, min=0.0, max=1.0)
    return {
        "n_live": int(live.sum()),
        "n_live_with_defined_ratio": n_usable,
        "n_live_without_defined_ratio": int(live.sum()) - n_usable,
        "counts": counts,
        "fractions": {name: value / n_usable for name, value in counts.items()},
        "histogram": [int(value) for value in histogram],
        "median_relative_norm": float(selected.median()),
        "shared_decoder_cosine": {
            "n": int(shared_mask.sum()),
            "mean": float(shared_cosine.mean()) if int(shared_mask.sum()) else None,
            "median": float(shared_cosine.median()) if int(shared_mask.sum()) else None,
            "fraction_below_zero": (
                float((shared_cosine < 0).float().mean())
                if int(shared_mask.sum())
                else None
            ),
        },
    }


def specificity_readout(
    model: Crosscoder,
    *,
    live: torch.Tensor,
    admissible: Sequence[int],
    exclusive_cut: float,
    shared_halfwidth: float,
) -> dict[str, Any]:
    """The per-site diff readout, refused at every inadmissible site.

    ``admissible`` names backbone layers, not positions in ``config.sites``, and
    must be a subset of them. A site that is fitted but not admissible carries an
    explicit refusal in place of its numbers: R2.4's operative rule is that a diff
    may be reported at layer ``l`` only where **both** cells' dictionaries carry
    at least that layer's own effective dimension ``r99`` in live latents, and a
    stage that reported a category count outside that set would be reporting the
    thing the rule forbids while looking exactly like the thing it permits.

    Reconstruction quality is *not* refused anywhere -- it is a statement about
    the dictionary and not a diff -- and is reported by the caller for every
    fitted site.
    """

    admitted = assert_admissible_subset(admissible, model.config.sites)
    norms = decoder_norms(model)
    ratio, defined = relative_decoder_norm(norms)
    cosine = decoder_cosine(model)
    live = live.cpu()
    if live.shape != ratio.shape:
        raise ValueError(
            f"the live mask is {tuple(live.shape)} and the dictionary is "
            f"{tuple(ratio.shape)}; a per-site readout cannot mix the two"
        )

    per_site: list[dict[str, Any]] = []
    for index, layer in enumerate(model.config.sites):
        entry: dict[str, Any] = {"layer": int(layer), "admissible": layer in admitted}
        if layer not in admitted:
            entry["verdict"] = "ADMISSIBILITY_REFUSED"
            entry["reason"] = (
                "layer is outside the declared admissible set, where at least one "
                "of the two cells' dictionaries does not carry that layer's own "
                "r99 in live latents. No diff is reported here"
            )
        else:
            entry.update(
                site_readout(
                    ratio=ratio[index],
                    cosine=cosine[index],
                    defined=defined[index],
                    live=live[index],
                    exclusive_cut=exclusive_cut,
                    shared_halfwidth=shared_halfwidth,
                )
            )
        per_site.append(entry)

    reported = [entry for entry in per_site if entry["admissible"]]
    return {
        "cuts": {
            "exclusive": exclusive_cut,
            "shared_halfwidth": shared_halfwidth,
            "note": (
                "a latent is adapted-specific at relative decoder norm >= "
                f"{exclusive_cut}, base-specific at <= {1.0 - exclusive_cut}, and "
                f"shared within {shared_halfwidth} of 0.5. The band between the two "
                "is reported as 'intermediate' rather than folded into either. The "
                "full histogram travels per site so the cuts can be re-read"
            ),
        },
        "admissible_layers": list(admitted),
        "fitted_layers": list(model.config.sites),
        "inadmissible_layers": [
            int(layer) for layer in model.config.sites if layer not in admitted
        ],
        "site_per_site": per_site,
        "relative_norm_histogram_per_site": [
            entry.get("histogram") for entry in per_site
        ],
        "category_counts_per_site": [entry.get("counts") for entry in per_site],
        "n_admissible_sites_reported": len(reported),
        "histogram_bins": HISTOGRAM_BINS,
    }


# --------------------------------------------------------------- refusals


def assert_admissible_subset(
    admissible: Sequence[int], sites: Sequence[int]
) -> tuple[int, ...]:
    """The admissible layer set, checked against the fitted sites.

    Raises rather than intersecting silently. An admissible layer the run did not
    fit is a configuration error -- the diff it asks for does not exist -- and
    quietly dropping it would leave an artefact that names a smaller admissible
    set than the campaign declared, which is the shape of a pre-registered
    criterion being narrowed after the fact.
    """

    fitted = tuple(int(value) for value in sites)
    admitted = tuple(int(value) for value in admissible)
    if len(admitted) == 0:
        raise ValueError(
            "the admissible layer set is empty, so no diff could be reported "
            "anywhere. R2.4's admission rule is a positive statement about where a "
            "diff is defined and an empty set is a refusal to run, not a run"
        )
    if sorted(set(admitted)) != sorted(admitted):
        raise ValueError(f"the admissible layer set repeats a layer: {list(admitted)}")
    outside = sorted(set(admitted) - set(fitted))
    if outside:
        raise ValueError(
            f"layers {outside} are declared admissible but were not fitted "
            f"(fitted: {list(fitted)}). A diff cannot be reported at a layer this "
            "Crosscoder carries no parameters for"
        )
    return tuple(sorted(admitted))


def assert_effective_dimension(
    values: Sequence[int], sites: Sequence[int], *, d_model: int
) -> tuple[int, ...]:
    """One measured effective dimension per fitted site, checked before it is used.

    ``r99`` is the smallest number of principal directions carrying 99% of the
    activation cloud's variance at that site, in that mode
    (:mod:`src.transfer.spectrum`, measured by
    ``scripts/transfer/30_activation_spectrum.py``). It is an **input** to this
    stage and is never inferred here, for the same reason the admissible layer set
    is: it depends on a measurement another stage owns, on a population and a mode
    this run does not re-derive.

    It is required rather than optional on a real checkpoint pair, because the
    number it qualifies -- the per-site reconstruction NMSE -- is not readable
    without it. The achievable NMSE depends on this quantity, and it varies across
    modes, across the two checkpoints and across layers, so an NMSE difference on
    any of those axes is confounded with it (see
    :data:`NMSE_COMPARABILITY_NOTE`). An artefact reporting the NMSE and not the
    dimension would be publishing half of a comparison that invites the wrong
    conclusion on three axes at once.

    ``r99`` cannot exceed ``ceil(0.99 * d_model)`` even on a flat spectrum with
    infinite samples, which is the attainability defect EXP-R2-202 recorded
    against its own threshold; the bound is checked here so a value above it is
    caught as a transcription error rather than reasoned about.
    """

    supplied = tuple(int(value) for value in values)
    if len(supplied) != len(sites):
        raise ValueError(
            f"{len(supplied)} effective dimensions were given for "
            f"{len(sites)} fitted sites {list(sites)}; there must be exactly one "
            "per site, in the same order, because each one qualifies that site's "
            "own reconstruction number"
        )
    ceiling = -(-99 * int(d_model) // 100)
    bad = {
        int(layer): value
        for layer, value in zip(sites, supplied)
        if not 1 <= value <= ceiling
    }
    if bad:
        raise ValueError(
            f"effective dimensions {bad} are outside 1..{ceiling}. r99 cannot "
            f"exceed ceil(0.99 * d_model) = {ceiling} even on a flat spectrum with "
            "infinite samples, so a value above it is a transcription error"
        )
    return supplied


def reconstruction_per_site(
    *,
    sites: Sequence[int],
    nmse_by_role: Sequence[Sequence[float]],
    nmse_total: Sequence[float],
    live: Sequence[int],
    effective_dimension: Sequence[int] | None,
) -> list[dict[str, Any]]:
    """Each site's reconstruction number **with the quantity that qualifies it**.

    One record per site carrying the NMSE, the effective dimension of the cloud
    that NMSE was measured against, this Crosscoder's own live basis, and their
    ratio -- so a reader who takes the number takes its limit with it.

    ``effective_dimension`` is **one ``[base, adapted]`` pair per site**, not one
    value. The two checkpoints do not share an effective dimension at the same
    layer and mode -- at layer 28 in protein mode the measured ``r99`` is 2,232 on
    the pre-adaptation checkpoint against 1,563 on the adapted one, and at layer
    31 it is 471 against 87 -- so a single value would qualify one role's
    reconstruction with the other role's geometry.

    ``live_over_r99`` is the Crosscoder's own live basis against the layer's
    effective dimension. It is **not** R2.4's admission rule and must not be read
    as it: that rule is stated over the two per-layer transcoders' live bases, one
    per checkpoint, and this is one dictionary over both. It is the analogous
    quantity for this object, reported because a reader comparing the two would
    otherwise construct it themselves and might construct it differently.
    """

    if effective_dimension is not None and len(effective_dimension) != len(sites):
        raise ValueError("one effective dimension pair per site, or none at all")
    records: list[dict[str, Any]] = []
    for index, layer in enumerate(sites):
        pair = None if effective_dimension is None else [
            int(value) for value in effective_dimension[index]
        ]
        if pair is not None and len(pair) != 2:
            raise ValueError(
                "each site's effective dimension is a [base, adapted] pair, one "
                "per role, because each role's NMSE is qualified by its own "
                f"checkpoint's cloud; got {pair} at layer {layer}"
            )
        binding = None if pair is None else min(pair)
        entry: dict[str, Any] = {
            "layer": int(layer),
            "held_out_nmse": float(nmse_total[index]),
            # Index-aligned with r99_effective_dimension below: position 0 is the
            # base checkpoint in both, position 1 the adapted. A role's NMSE and
            # the dimension of the cloud that role's NMSE was measured against sit
            # at the same index, so the pairing is positional rather than asserted.
            "held_out_nmse_by_role": [float(value) for value in nmse_by_role[index]],
            "r99_effective_dimension": pair,
            "live_latents": int(live[index]),
            "live_over_r99": (
                None if not binding else float(live[index]) / float(binding)
            ),
            "live_over_r99_uses": (
                "the smaller of the two roles' r99, which is the binding "
                "constraint on what a shared latent space can resolve"
            ),
            "nmse_comparability": NMSE_COMPARABILITY_NOTE,
        }
        if pair is None:
            entry["r99_note"] = (
                "not supplied. This run did not declare the effective dimension at "
                "this site, so its NMSE cannot be qualified and must not be read "
                "against another mode's at all"
            )
        records.append(entry)
    return records


def assert_per_layer_fields(payload: Any, *, n_sites: int, path: str = "") -> None:
    """Refuse any per-site field that was reduced to a scalar or lost its length.

    Walks a whole artefact and checks every key ending in ``_per_site``. The
    defect this exists to stop already happened in this exact pipeline: a
    ``(num_layers, d_hidden)`` dead mask was summed to one cross-layer scalar
    before it reached any artefact, so R2.4's basis criterion -- pre-registered as
    a per-layer condition -- could only ever be read as a mean, and the gate that
    was blocking the unit had never been evaluated at the resolution it was
    specified at (EXP-R2-203). A mean over layers is not a statement about any
    layer, and once it is written it cannot be un-collapsed.
    """

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            here = f"{path}.{key}" if path else str(key)
            if isinstance(key, str) and key.endswith(PER_SITE_SUFFIX):
                if isinstance(value, (bool, int, float)) or value is None:
                    raise ValueError(
                        f"{here} is a per-site field and was written as "
                        f"{type(value).__name__}. A per-layer quantity reduced to a "
                        "scalar cannot be un-collapsed and voids any criterion "
                        "stated per layer"
                    )
                if not isinstance(value, (list, tuple)):
                    raise ValueError(
                        f"{here} is a per-site field and must be a sequence, not "
                        f"{type(value).__name__}"
                    )
                if len(value) != n_sites:
                    raise ValueError(
                        f"{here} carries {len(value)} values for {n_sites} fitted "
                        "sites"
                    )
            assert_per_layer_fields(value, n_sites=n_sites, path=here)
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            assert_per_layer_fields(item, n_sites=n_sites, path=f"{path}[{index}]")


def assert_required_per_site_fields(payload: Any) -> None:
    """Every field of :data:`REQUIRED_PER_SITE_FIELDS` appears somewhere in the artefact.

    Absence and collapse are the same failure from a reader's point of view, and
    absence is the easier one to miss.
    """

    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if isinstance(key, str) and key in REQUIRED_PER_SITE_FIELDS:
                    found.add(key)
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(payload)
    missing = sorted(set(REQUIRED_PER_SITE_FIELDS) - found)
    if missing:
        raise ValueError(
            f"this artefact carries no {missing}; a per-layer quantity that is "
            "absent is as unreadable as one that was averaged"
        )


# ------------------------------------------------------------ the certificate


#: What two Crosscoder runs must agree on **beyond** what
#: :data:`src.transfer.transcoders.MATCHED_TRAINING_FIELDS` already covers.
#:
#: The shared list is imported and not extended: it is frozen, every existing
#: dictionary's digest depends on it, and none of these four fields exists on a
#: transcoder at all. ``sites`` is here rather than folded into ``architecture``
#: because the fitted site set is a set and deserves to be compared as one;
#: ``decoder_norm_penalty`` because it is the term that creates the exclusivity
#: this unit reports, so two runs at different values are not comparable readouts;
#: ``pairing`` because a shuffled-pairing run is the null and must never certify
#: as the measurement; and ``backbone_pair_sha256`` because a Crosscoder has two
#: backbones and the single-backbone field cannot name a pair.
CROSSCODER_MATCHED_FIELDS: tuple[str, ...] = (
    "sites",
    "decoder_norm_penalty",
    "pairing",
    "backbone_pair_sha256",
)


def pair_backbone_digest(base_sha256: str, adapted_sha256: str) -> str:
    """One digest naming the *pair* of checkpoints a Crosscoder was fitted to.

    ``MatchedTraining.backbone_sha256`` names one checkpoint, which is the right
    field for a transcoder and cannot express a Crosscoder's target. Two runs of
    this stage -- the text mode and the protein mode of one pair -- must certify
    as the same target, and a run against a different pair must not; the digest
    over both weight digests in role order is what says so.
    """

    for name, value in (("base", base_sha256), ("adapted", adapted_sha256)):
        if not value:
            raise ValueError(f"the {name} checkpoint has no weight digest")
    if base_sha256 == adapted_sha256:
        raise ValueError(
            "the two roles carry the same weight digest, so this is one checkpoint "
            "against itself. That is a legitimate plumbing check and is not a "
            "model diff; run it with --allow-self-pair, which records the fact"
        )
    return hashlib.sha256(
        json.dumps([base_sha256, adapted_sha256], separators=(",", ":")).encode()
    ).hexdigest()


def crosscoder_certificate(
    left: MatchedTraining,
    right: MatchedTraining,
    *,
    left_extra: Mapping[str, Any],
    right_extra: Mapping[str, Any],
) -> dict[str, Any]:
    """Whether two Crosscoder runs may be read against each other.

    Composes :func:`src.transfer.transcoders.compare_matched_training` over the
    fields both object families share, then compares the four a Crosscoder adds.
    A disagreement on either half is a ``MISMATCH``: the shared verdict cannot be
    allowed to read ``MATCHED`` on a pair that differs in its sparsity penalty or
    in its pairing, which are the two settings that decide what the readout says.
    """

    base = compare_matched_training(left, right)
    missing = sorted(
        name
        for name in CROSSCODER_MATCHED_FIELDS
        if name not in left_extra or name not in right_extra
    )
    if missing:
        raise KeyError(
            f"a Crosscoder certificate needs {missing} from both runs; without them "
            "the fields that decide the readout are not compared at all"
        )
    fields = {
        name: {
            "values": [left_extra[name], right_extra[name]],
            "agree": left_extra[name] == right_extra[name],
        }
        for name in CROSSCODER_MATCHED_FIELDS
    }
    extra_disagreements = sorted(
        name for name, entry in fields.items() if not entry["agree"]
    )
    verdict = base["verdict"]
    if extra_disagreements:
        verdict = "MISMATCH"
    return {
        **base,
        "crosscoder_fields": fields,
        "crosscoder_matched_fields": list(CROSSCODER_MATCHED_FIELDS),
        "crosscoder_disagreements": extra_disagreements,
        "verdict": verdict,
        "note": (
            "the shared verdict comes from transcoders.compare_matched_training "
            "over the fields every dictionary in this repository is refused on; "
            "the four Crosscoder-specific fields are compared here because none of "
            "them exists on a transcoder and that list is frozen. Any disagreement "
            "on either half is MISMATCH"
        ),
    }


# --------------------------------------------------------------- the trainer


PairedBatches = Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]]


def clip_per_site_grad_norm_(model: Crosscoder, max_norm: float) -> list[float]:
    """Clip each site's gradients to ``max_norm`` **separately**, and report the norms.

    One global clip would couple the sites: the global norm grows with how many
    sites are in the run, so the same layer would receive a differently scaled
    gradient in a two-layer run and in a thirty-two-layer one, and the two would
    not produce the same dictionary at that layer. That coupling is the only one
    this architecture has left, and it is removed here rather than disclosed --
    see :data:`SITE_INDEPENDENCE_NOTE`, whose whole claim rests on this function.

    Returns the pre-clip norm per site, which is a per-layer diagnostic worth
    keeping: a site whose gradient norm is orders of magnitude from its
    neighbours' is a site whose activations are not on the scale the rest are.
    """

    if max_norm <= 0.0:
        raise ValueError("the gradient clip must be positive")
    norms: list[float] = []
    for index in range(model.config.n_sites):
        slices = [
            model.W_enc.grad[:, index] if model.W_enc.grad is not None else None,
            model.b_enc.grad[index] if model.b_enc.grad is not None else None,
            model.W_dec.grad[:, index] if model.W_dec.grad is not None else None,
            model.b_dec.grad[:, index] if model.b_dec.grad is not None else None,
        ]
        present = [value for value in slices if value is not None]
        if not present:
            norms.append(0.0)
            continue
        total = torch.sqrt(
            sum((value.double() ** 2).sum() for value in present)
        )
        norms.append(float(total))
        if float(total) > max_norm:
            factor = max_norm / (float(total) + 1e-12)
            for value in present:
                value.mul_(factor)
    return norms


def evaluate_crosscoder(
    model: Crosscoder,
    batches: PairedBatches,
    *,
    pairing: str,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Held-out per-(site, role) NMSE and a per-site firing census.

    The firing census is the live-basis definition a diff is read under -- a
    latent is live on the cohort when it fires on it -- and it is what
    :func:`specificity_readout` masks with. The checkpoint's own ``silent_steps``
    counter is the other definition and is recorded beside it, never blended:
    they answer different questions.
    """

    model.eval()
    sites = model.config.n_sites
    totals = torch.zeros(2, sites, dtype=torch.float64)
    census = FiringCensus(sites, model.config.d_hidden)
    batch_count = 0
    positions = 0
    with torch.no_grad():
        for base, adapted in batches():
            adapted = apply_pairing(adapted, pairing=pairing, generator=generator)
            if base.shape[1] == 0:
                continue
            report = model.objective(base, adapted, training=False)
            totals += report["nmse_role_by_site"].double().cpu()
            census.update(report["fired_per_latent"].cpu(), int(base.shape[1]))
            batch_count += 1
            positions += int(base.shape[1])
    model.train()
    if batch_count == 0:
        raise RuntimeError(
            "the held-out pass produced no token positions, so this Crosscoder has "
            "no score; refusing to report a readout with no evaluation behind it"
        )
    mean = totals / batch_count
    return {
        "n_batches": batch_count,
        "n_positions": positions,
        # Site-major, one ``[base, adapted]`` pair per site: the two roles'
        # reconstruction quality can diverge at a site and a sum over roles would
        # hide exactly the asymmetry a diff is about.
        "nmse_by_role_per_site": [
            [float(mean[0, site]), float(mean[1, site])] for site in range(sites)
        ],
        "nmse_per_site": mean.sum(dim=0).tolist(),
        "nmse_sum": float(mean.sum()),
        "firing_census": census.record(),
        "_counts": census.counts,
    }


def live_mask(counts: torch.Tensor, *, minimum: int) -> torch.Tensor:
    """Which latents fired at least ``minimum`` times on the held-out cohort, per site."""

    if minimum < 1:
        raise ValueError("a latent that fires fewer than once has not fired")
    return counts >= int(minimum)


def train_crosscoders(
    configs: Sequence[CrosscoderConfig],
    training: PairedBatches,
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    seed: int,
    device: str | torch.device = "cpu",
    warm_up_batches: int = 8,
    token_budget: int | None = None,
    held_out: PairedBatches | None = None,
    eval_every: int = 0,
    log: Callable[[str], None] | None = None,
) -> tuple[list[Crosscoder], list[TrainingRecord], dict[str, Any]]:
    """Fit several Crosscoders **on one pass over the activations**.

    The reason this is plural: the measurement and its shuffled-pairing null are
    two fitted objects, and running them as two invocations would let the null be
    dispatched separately, dispatched later, or not dispatched at all -- and would
    put the two fits on two draws of the corpus. Here they see literally the same
    captured activations, in the same order, under the same frozen normalisation
    constants, and differ in exactly the declared pairing. The backbone forward
    pass, which dominates the cost on a narrow site set, is paid once.

    Every config must agree on ``sites``, ``d_model`` and ``d_hidden`` -- they are
    consuming one stream -- and must differ somewhere, or the run is fitting the
    same object twice.

    ``training`` and ``held_out`` are **factories**: the scale warm-up, each
    evaluation and the training loop each open their own iterator, so a run that
    evaluates does not consume the stream it trains on.

    ``token_budget`` stops at the first step to reach it, exactly as
    ``17_train_transcoder.py`` does and for the same reason: a text record and a
    protein record carry different numbers of scored positions, so two modes run
    for equal *steps* are a matched schedule over unequal data. ``steps`` then
    bounds the run and sets the held-out offset. Without it two mode cells can
    only ever certify ``UNMATCHED_BUDGET``.
    """

    if steps < 1:
        raise ValueError("a training run needs at least one step")
    if token_budget is not None and token_budget < 1:
        raise ValueError("a token budget must be positive, or absent")
    if not configs:
        raise ValueError("no Crosscoder was requested")
    shapes = {(entry.sites, entry.d_model, entry.d_hidden) for entry in configs}
    if len(shapes) != 1:
        raise ValueError(
            "every Crosscoder in one run must share sites, d_model and d_hidden; "
            f"got {sorted(shapes)}. They consume one stream of activations"
        )
    if len({(entry.pairing, entry.decoder_norm_penalty, entry.k) for entry in configs}) != len(
        configs
    ):
        raise ValueError(
            "two Crosscoders in this run are identical in pairing, sparsity "
            "penalty and k, so the run would fit the same object twice"
        )

    torch.manual_seed(seed)
    reference = configs[0]
    models = [
        Crosscoder(entry, init_seed=seed).to(device).float() for entry in configs
    ]
    # One estimate, shared. The normalisation constants are a property of the
    # activations and not of the pairing, and a null normalised differently from
    # the measurement would differ in something nobody declared.
    scales, scale_record = estimate_scales(
        training,
        n_sites=reference.n_sites,
        d_model=reference.d_model,
        n_batches=warm_up_batches,
    )
    for model in models:
        model.set_scales(scales.to(device))

    optimisers = [
        torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for model in models
    ]
    generators = [
        torch.Generator().manual_seed(seed + 1 + index) for index in range(len(configs))
    ]
    evaluation_seed = seed + 101

    records = [TrainingRecord() for _ in configs]
    finals: list[dict[str, Any] | None] = [None for _ in configs]
    # The largest batch the run actually drew, in scored positions. Recorded
    # because the memory a run needs scales with it and the *cap* is not it: on
    # this programme's protein cohort the realised mean is 919 tokens against a
    # 4,096-token cap, and the largest four-record batch over a whole run is about
    # 2,230. A campaign sized against the cap is sized against a batch the corpus
    # never produces.
    widest_batch = 0
    stream = training()
    for step in range(1, steps + 1):
        try:
            base, adapted = next(stream)
        except StopIteration:
            raise RuntimeError(
                f"the paired activation stream ran out at step {step} of {steps}; "
                "a Crosscoder trained on fewer positions than declared cannot be "
                "matched against the other mode's"
            ) from None
        if base.shape[1] == 0:
            continue
        widest_batch = max(widest_batch, int(base.shape[1]))
        # The budget is reached at the FIRST step to cross it, so the realised
        # total overshoots by at most one batch -- which is why the matched
        # declaration compares the budget and records the realised count beside
        # it rather than the other way round.
        reached = token_budget is not None and (
            records[0].tokens + int(base.shape[1]) >= token_budget
        )
        for index, (model, optimiser) in enumerate(zip(models, optimisers)):
            paired = apply_pairing(
                adapted, pairing=configs[index].pairing, generator=generators[index]
            )
            report = model.objective(base, paired, training=True)
            optimiser.zero_grad(set_to_none=True)
            report["loss"].backward()
            grad_norms = clip_per_site_grad_norm_(model, grad_clip)
            optimiser.step()

            records[index].steps = step
            records[index].tokens += int(base.shape[1])
            due = reached or (eval_every and (step % eval_every == 0 or step == steps))
            if due and held_out is not None:
                final = evaluate_crosscoder(
                    model,
                    held_out,
                    pairing=configs[index].pairing,
                    generator=torch.Generator().manual_seed(evaluation_seed),
                )
                finals[index] = final
                entry = {
                    "step": step,
                    "train_nmse_sum": float(report["nmse_sum"]),
                    "held_out_nmse_sum": final["nmse_sum"],
                    "held_out_nmse_per_site": final["nmse_per_site"],
                    "n_dead_per_site": report["n_dead_per_site"],
                    "decoder_penalty_per_site": [
                        float(value) for value in report["decoder_penalty_per_site"]
                    ],
                    "grad_norm_per_site": grad_norms,
                    "active_fraction": report["active_fraction"],
                    "tokens": records[index].tokens,
                }
                records[index].history.append(entry)
                if log is not None:
                    log(
                        f"  [{configs[index].pairing:8s}] step {step:6d}  train "
                        f"{entry['train_nmse_sum']:8.4f}  held-out "
                        f"{entry['held_out_nmse_sum']:8.4f}  dead "
                        f"{sum(entry['n_dead_per_site'] or [0]):6d}"
                    )
        if reached:
            break

    if token_budget is not None and records[0].tokens < token_budget:
        raise RuntimeError(
            f"steps {steps} ran out after {records[0].tokens} of {token_budget} "
            "scored tokens, so this Crosscoder saw less data than the budget it "
            "declares and could not be matched against the other mode's. Raise "
            "--steps: it bounds the run and sets the held-out offset, and the "
            "token budget is what stops it"
        )
    if held_out is not None:
        for index, model in enumerate(models):
            if finals[index] is None:
                finals[index] = evaluate_crosscoder(
                    model,
                    held_out,
                    pairing=configs[index].pairing,
                    generator=torch.Generator().manual_seed(evaluation_seed),
                )
    return (
        models,
        records,
        {
            "scales": scale_record,
            "held_out": finals,
            "widest_batch_positions": widest_batch,
            "mean_batch_positions": (
                records[0].tokens / records[0].steps if records[0].steps else 0.0
            ),
        },
    )


def train_crosscoder(
    config: CrosscoderConfig, training: PairedBatches, **options: Any
) -> tuple[Crosscoder, TrainingRecord, dict[str, Any]]:
    """One Crosscoder, through :func:`train_crosscoders`. No second training loop."""

    models, records, extra = train_crosscoders([config], training, **options)
    return (
        models[0],
        records[0],
        {**extra, "held_out": extra["held_out"][0]},
    )


# ------------------------------------------- the synthetic ground-truth check


@dataclass(frozen=True)
class SyntheticGroundTruth:
    """Paired activations with a known number of shared and role-specific features.

    **Why this is in ``src`` and not only in a test.** A Crosscoder's readout is
    an unsupervised claim about which latents belong to which model, and nothing
    in a real run can check it. The only place the claim is checkable is on data
    whose answer is known in advance, so the construction that makes it checkable
    is part of the instrument, exactly as
    :func:`src.transfer.spectrum.isotropic_control_spectrum` is part of the
    spectrum estimator. The stage can run it and write its own certificate.

    The construction follows the 2025 note's toy model. Each true feature ``j``
    carries a non-negative coefficient ``c_j(t)`` at token ``t``; a **shared**
    feature has the same coefficient in both roles and a direction in each,
    optionally rotated between them; a **role-specific** feature has a direction
    in one role only. That is what "the same feature in two models" means for a
    crosscoder -- it fires on the same datapoint -- and it is deliberately not
    "the same direction", which the 2024 note distinguishes from it explicitly.

    ``rank`` confines every direction to a random subspace of that dimension. The
    activations this programme actually fits dictionaries to are low-rank -- the
    measured effective dimension at the dictionary site is 2,588 to 3,670 against
    ``d_model`` 4,096 (EXP-R2-202) -- so a rank-deficient draw is a realistic
    operating condition and not a corner case.
    """

    d_model: int
    n_sites: int
    n_shared: int
    n_base_specific: int
    n_adapted_specific: int
    active_per_token: int
    seed: int
    #: Dimension of the subspace every direction lives in. ``None`` is full rank.
    rank: int | None = None
    #: Cosine-scale rotation applied to a shared feature's direction between the
    #: two roles. 0 means identical directions in both.
    shared_rotation: float = 0.0
    #: Isotropic noise added to both roles, as a fraction of the mean coefficient.
    noise: float = 0.0

    def __post_init__(self) -> None:
        if self.n_features == 0:
            raise ValueError("a ground truth with no features has nothing to recover")
        if self.active_per_token < 1 or self.active_per_token > self.n_features:
            raise ValueError(
                f"active_per_token must lie in [1, {self.n_features}]; got "
                f"{self.active_per_token}"
            )
        if self.rank is not None and not 1 <= self.rank <= self.d_model:
            raise ValueError(f"rank must lie in [1, {self.d_model}]; got {self.rank}")

    @property
    def n_features(self) -> int:
        return self.n_shared + self.n_base_specific + self.n_adapted_specific

    @property
    def categories(self) -> list[str]:
        """Each true feature's category, in the order the coefficient matrix uses."""

        return (
            ["shared"] * self.n_shared
            + ["base_specific"] * self.n_base_specific
            + ["adapted_specific"] * self.n_adapted_specific
        )

    def directions(self) -> torch.Tensor:
        """``(2, n_sites, n_features, d_model)``; zero where a feature is absent."""

        rng = np.random.default_rng(self.seed)
        basis = None
        if self.rank is not None:
            raw = rng.standard_normal((self.d_model, self.rank))
            basis, _ = np.linalg.qr(raw)

        def draw(count: int) -> np.ndarray:
            if basis is None:
                vectors = rng.standard_normal((count, self.d_model))
            else:
                vectors = rng.standard_normal((count, self.rank)) @ basis.T
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors / np.maximum(norms, 1e-12)

        out = np.zeros((2, self.n_sites, self.n_features, self.d_model))
        shared = slice(0, self.n_shared)
        base_only = slice(self.n_shared, self.n_shared + self.n_base_specific)
        adapted_only = slice(self.n_shared + self.n_base_specific, self.n_features)
        for site in range(self.n_sites):
            common = draw(self.n_shared)
            out[0, site, shared] = common
            if self.shared_rotation <= 0.0:
                out[1, site, shared] = common
            else:
                perturbation = draw(self.n_shared) * self.shared_rotation
                rotated = common + perturbation
                rotated /= np.maximum(
                    np.linalg.norm(rotated, axis=1, keepdims=True), 1e-12
                )
                out[1, site, shared] = rotated
            out[0, site, base_only] = draw(self.n_base_specific)
            out[1, site, adapted_only] = draw(self.n_adapted_specific)
        return torch.from_numpy(out).float()

    def draw(
        self, tokens: int, *, rng: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(base, adapted, coefficients)`` for one batch of ``tokens`` positions."""

        coefficients = np.zeros((tokens, self.n_features))
        for token in range(tokens):
            active = rng.choice(
                self.n_features, size=self.active_per_token, replace=False
            )
            coefficients[token, active] = rng.uniform(0.5, 1.5, self.active_per_token)
        weights = torch.from_numpy(coefficients).float()
        directions = self._cached_directions
        activations = torch.einsum("tf,rsfd->rstd", weights, directions)
        if self.noise > 0.0:
            noise = torch.from_numpy(
                rng.standard_normal((2, self.n_sites, tokens, self.d_model))
            ).float()
            activations = activations + self.noise * noise
        return activations[0], activations[1], weights

    @property
    def _cached_directions(self) -> torch.Tensor:
        cached = getattr(self, "_directions_cache", None)
        if cached is None:
            cached = self.directions()
            object.__setattr__(self, "_directions_cache", cached)
        return cached

    def batches(
        self, *, tokens_per_batch: int, n_batches: int, seed: int
    ) -> PairedBatches:
        """A factory of paired activation batches, seeded so every pass matches."""

        def factory() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            rng = np.random.default_rng(seed)
            for _ in range(n_batches):
                base, adapted, _ = self.draw(tokens_per_batch, rng=rng)
                yield base, adapted

        return factory

    def record(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "n_sites": self.n_sites,
            "n_features": self.n_features,
            "injected_per_category": {
                "shared": self.n_shared,
                "base_specific": self.n_base_specific,
                "adapted_specific": self.n_adapted_specific,
            },
            "active_per_token": self.active_per_token,
            "rank": self.rank if self.rank is not None else self.d_model,
            "rank_is_deficient": self.rank is not None and self.rank < self.d_model,
            "shared_rotation": self.shared_rotation,
            "noise": self.noise,
            "seed": self.seed,
        }


def match_features_to_latents(
    coefficients: torch.Tensor, latents: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each true feature, the best-correlated latent and that correlation.

    Correlation over token positions rather than direction cosine, because a
    Crosscoder's claim about a feature is that it fires on the same datapoints --
    which is the definition the 2024 note draws the distinction on -- and because
    a shared feature is allowed to point in different directions in the two
    models, so a direction match would count a correct recovery as a failure.
    """

    if coefficients.ndim != 2 or latents.ndim != 2:
        raise ValueError("expected (tokens, features) and (tokens, latents)")
    if coefficients.shape[0] != latents.shape[0]:
        raise ValueError(
            f"{coefficients.shape[0]} coefficient rows against "
            f"{latents.shape[0]} latent rows; they must be the same positions"
        )

    def standardise(value: torch.Tensor) -> torch.Tensor:
        centred = value.double() - value.double().mean(dim=0, keepdim=True)
        return centred / centred.norm(dim=0, keepdim=True).clamp_min(_TINY)

    correlation = standardise(coefficients).T @ standardise(latents)
    best = correlation.max(dim=1)
    return best.indices, best.values


def recovery_report(
    truth: SyntheticGroundTruth,
    model: Crosscoder,
    *,
    site: int,
    base: torch.Tensor,
    adapted: torch.Tensor,
    coefficients: torch.Tensor,
    live: torch.Tensor,
    exclusive_cut: float,
    shared_halfwidth: float,
    correlation_floor: float = 0.5,
) -> dict[str, Any]:
    """Injected against recovered counts per category, at one site.

    A true feature counts as **recovered** when some latent's activation
    correlates with its coefficient above ``correlation_floor`` over the held-out
    positions, and as **correctly categorised** when that latent's relative
    decoder norm puts it in the injected category. The two are reported
    separately: a Crosscoder that finds every feature and mislabels half of them
    is a different failure from one that finds none, and a single "accuracy" would
    not tell them apart.
    """

    device = model.W_dec.device
    with torch.no_grad():
        latents, _, _ = model.encode(base.to(device), adapted.to(device))
    norms = decoder_norms(model)
    ratio, defined = relative_decoder_norm(norms)
    codes = categorise(
        ratio[site], exclusive_cut=exclusive_cut, shared_halfwidth=shared_halfwidth
    )
    live = live.cpu()
    usable = live[site] & defined[site]

    indices, correlations = match_features_to_latents(
        coefficients.cpu(), latents[site].cpu()
    )
    per_category: dict[str, dict[str, Any]] = {}
    for name in ("shared", "base_specific", "adapted_specific"):
        selected = [
            index for index, label in enumerate(truth.categories) if label == name
        ]
        if not selected:
            per_category[name] = {"injected": 0, "recovered": 0, "categorised": 0}
            continue
        recovered = 0
        categorised = 0
        for feature in selected:
            latent = int(indices[feature])
            if float(correlations[feature]) < correlation_floor:
                continue
            recovered += 1
            if bool(usable[latent]) and CATEGORIES[int(codes[latent])] == name:
                categorised += 1
        per_category[name] = {
            "injected": len(selected),
            "recovered": recovered,
            "categorised": categorised,
            "median_correlation": float(
                correlations[torch.tensor(selected)].median()
            ),
        }
    return {
        "site": int(site),
        "correlation_floor": correlation_floor,
        "per_category": per_category,
        "n_live": int(live[site].sum()),
        "live_category_counts": {
            name: int(((codes == index) & usable).sum())
            for index, name in enumerate(CATEGORIES)
        },
    }
