"""What each layer is trying to say: the logit, tuned and Jacobian lenses.

The programme hypothesis is that protein decoders have a *limited output
semantic interface*. A text unembedding maps onto ~50k word-like tokens which
are themselves the language explanations are written in; a protein unembedding
maps onto twenty amino acids carrying at most log2(20) = 4.32 bits and admitting
no semantic decomposition. Lenses test this directly, because a lens is nothing
but the unembedding applied early: whatever a lens can say about a layer is
bounded by what the output interface can express.

Three methods are implemented behind one code path so that a 36-layer text
decoder and a 27-layer protein decoder are measured identically on a
relative-depth grid.

*Logit lens.* Project the residual stream at layer ``l`` through the model's own
final layer norm and unembedding and read the induced next-token distribution.

*Tuned lens.* The logit lens is biased: intermediate states are not in the final
basis, so part of any measured deficit is a basis error rather than missing
prediction. A per-layer affine translator is fitted to remove that bias, trained
on a disjoint split of the same cohort.

*Jacobian lens (J-lens).* The two lenses above report what a layer *emits*
through the output interface. The J-lens asks what the interface is *sensitive
to*: the singular structure of the Jacobian of the final logits with respect to
the layer-l residual stream, compared against the subspace the layer's
activations actually occupy.

Layer-norm folding, and why the logit lens is only approximate
--------------------------------------------------------------
``LensHead`` applies the model's trained final ``LayerNorm`` (gain, bias and
epsilon exactly as trained) to an intermediate residual state and then the
trained unembedding. This is the standard logit lens and it is an approximation
in three specific ways, none of which this module can remove:

1. *Statistics mismatch.* The normaliser subtracts the mean and divides by the
   standard deviation of the state it is given. Residual norm grows steeply with
   depth, so at layer ``l`` the final LayerNorm is computing normalisation
   statistics on a distribution it never saw in training. The re-scaling that
   results is a rescaling of the *early* state, not the one the trained gain was
   calibrated for.
2. *Basis mismatch.* Even after normalisation, an intermediate state is not
   expressed in the basis the unembedding reads. This is exactly the bias the
   tuned lens is fitted to remove, and the untuned-versus-tuned gap is the only
   honest measurement of its size.
3. *Mean-direction blindness.* LayerNorm centres its input, so the logit lens is
   invariant to the all-ones direction of the residual stream. Any information a
   layer carries purely in that direction is invisible to both the logit and the
   tuned lens.

No attempt is made to "fold" the LayerNorm into the unembedding as a single
linear map. That folding is exact only if the normalisation scale is treated as
a constant, which it is not; pretending otherwise would silently change the
measured trajectory. The normalisation is therefore applied as the nonlinear
function it is, and the approximation is declared rather than hidden.

Comparability across arms
-------------------------
Per-token quantities are tokenizer-dependent. ProtGPT2 uses multi-residue BPE
while ZymCTRL and ProGen2-medium are residue-level, so a per-token cross-entropy
of one is not comparable with a per-token cross-entropy of the other.
``per_symbol_view`` re-expresses every rate metric per symbol using the measured
tokenizer expansion. Top-1 agreement with the final prediction is a per-token
event count with no per-symbol reading and is reported as such.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .arms import (
    AA20,
    Arm,
    Cohort,
    conditioning_boundary_ids,
    tokenize_batch,
)
from .scoring import analysis_layer, sequence_target_mask, target_rule

SCHEMA_VERSION_LOGIT_LENS = "r2_transfer_logit_lens_v1"
SCHEMA_VERSION_TUNED_LENS = "r2_transfer_tuned_lens_v1"
SCHEMA_VERSION_JACOBIAN_LENS = "r2_transfer_jacobian_lens_v1"
SCHEMA_VERSION_BOOTSTRAP = "r2_transfer_lens_cluster_bootstrap_v1"
SCHEMA_VERSION_RESIDUE_CLASS = "r2_transfer_lens_residue_class_v1"

LN2 = math.log(2.0)

#: Relative-depth grid. Fractions rather than layer indices, so that a 36-layer
#: and a 27-layer decoder are read at the same relative depth and their
#: trajectories can be placed on one axis.
DEFAULT_DEPTH_FRACTIONS: tuple[float, ...] = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
)

#: A four-way chemical partition of the canonical alphabet. Every residue
#: belongs to exactly one class, which is what makes the class distribution a
#: coarsening of the residue distribution and therefore lets residue entropy
#: decompose exactly into class entropy plus within-class entropy.
#:
#: ``special`` collects cysteine (disulfide bonding), glycine (backbone
#: flexibility) and proline (backbone rigidity): three residues whose behaviour
#: is dominated by backbone or covalent chemistry rather than by side-chain
#: polarity, so grouping them with either hydrophobic or polar residues would
#: make the partition chemically incoherent.
AA_CLASSES: dict[str, str] = {
    "charged": "DEKRH",
    "hydrophobic": "AVLIMFW",
    "polar": "STNQY",
    "special": "CGP",
}

CLASS_NAMES: tuple[str, ...] = tuple(sorted(AA_CLASSES))

CLASS_OF_RESIDUE: dict[str, str] = {
    residue: name for name, residues in AA_CLASSES.items() for residue in residues
}

if sorted(CLASS_OF_RESIDUE) != sorted(AA20) or len(CLASS_OF_RESIDUE) != len(AA20):
    raise AssertionError("AA_CLASSES is not a partition of the canonical alphabet")


# ---------------------------------------------------------------- layer grid


@dataclass(frozen=True)
class LayerPoint:
    """One point of the relative-depth grid.

    ``layer`` indexes transformer blocks, and the residual stream it names is the
    *output* of that block. ``relative_depth`` is ``(layer + 1) / n_layer`` so
    that the deepest point is 1.0 on every arm regardless of depth.
    """

    layer: int
    relative_depth: float
    depth_fractions: tuple[float, ...]


def layer_grid(n_layer: int, fractions: Sequence[float]) -> tuple[LayerPoint, ...]:
    """Resolve depth fractions onto block indices, de-duplicating collisions.

    Two fractions can round to the same block in a shallow model. Measuring that
    block twice would double its weight in every summary without adding
    evidence, so points are keyed by block and carry every fraction that
    produced them.
    """

    if n_layer < 1:
        raise ValueError("n_layer must be positive")
    if not fractions:
        raise ValueError("at least one depth fraction is required")
    if any(not 0.0 <= float(fraction) <= 1.0 for fraction in fractions):
        raise ValueError("depth fractions must lie in [0, 1]")
    collected: dict[int, list[float]] = {}
    for fraction in fractions:
        layer = analysis_layer(n_layer, float(fraction))
        collected.setdefault(layer, []).append(float(fraction))
    return tuple(
        LayerPoint(
            layer=layer,
            relative_depth=(layer + 1) / n_layer,
            depth_fractions=tuple(values),
        )
        for layer, values in sorted(collected.items())
    )


# ----------------------------------------------------------------- lens head


@dataclass(frozen=True)
class LensHead:
    """The model's own final normalisation and unembedding, held in float32.

    The head is materialised in float32 rather than the inference dtype because
    every lens quantity is a log-probability difference between two
    distributions that are nearly identical at the deepest layers; bfloat16
    rounding on the logits is comparable to the effect being measured.
    ``verify_lens_head`` checks that this float32 head reproduces the model's own
    final distribution, so the substitution is measured rather than assumed.
    """

    weight: torch.Tensor
    bias: torch.Tensor | None
    norm_weight: torch.Tensor
    norm_bias: torch.Tensor
    norm_eps: float
    d_model: int
    vocab_size: int

    def normalise(self, hidden: torch.Tensor) -> torch.Tensor:
        """The trained final LayerNorm, applied to whatever state it is given."""

        if hidden.shape[-1] != self.d_model:
            raise ValueError(
                f"lens head expects width {self.d_model}, got {hidden.shape[-1]}"
            )
        return F.layer_norm(
            hidden.float(),
            (self.d_model,),
            weight=self.norm_weight,
            bias=self.norm_bias,
            eps=self.norm_eps,
        )

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(self.normalise(hidden), self.weight, self.bias)

    def log_probs(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.logits(hidden), dim=-1)

    def centred_weight(self) -> torch.Tensor:
        """Unembedding with the constant logit direction removed.

        Adding a constant to every logit leaves the predictive distribution
        unchanged, so the all-ones direction of logit space carries no
        information. Removing it before any rank statement is taken makes the
        algebraic bound on the Jacobian rank exactly ``min(d_model, V - 1)``
        instead of a vacuous ``min(d_model, V)``.
        """

        return self.weight - self.weight.mean(dim=0, keepdim=True)


def lens_head(arm: Arm) -> LensHead:
    """Extract the final norm and unembedding, failing on any other topology."""

    transformer = getattr(arm.model, "transformer", None)
    if transformer is None or not hasattr(transformer, "ln_f"):
        raise TypeError(f"{arm.name}: no transformer.ln_f final normalisation")
    head = getattr(arm.model, "lm_head", None)
    if not isinstance(head, nn.Linear):
        raise TypeError(f"{arm.name}: lm_head is not a linear unembedding")
    norm = transformer.ln_f
    if not isinstance(norm, nn.LayerNorm):
        raise TypeError(f"{arm.name}: transformer.ln_f is not a LayerNorm")
    if norm.normalized_shape != (arm.d_model,):
        raise ValueError(
            f"{arm.name}: ln_f normalises {norm.normalized_shape}, expected {(arm.d_model,)}"
        )
    if norm.weight is None or norm.bias is None:
        raise ValueError(f"{arm.name}: ln_f has no learned gain or bias")
    if head.in_features != arm.d_model:
        raise ValueError(
            f"{arm.name}: lm_head reads width {head.in_features}, expected {arm.d_model}"
        )
    vocab = int(arm.model.config.vocab_size)
    if head.out_features != vocab:
        raise ValueError(
            f"{arm.name}: lm_head emits {head.out_features} logits, config declares {vocab}"
        )
    return LensHead(
        weight=head.weight.detach().float(),
        bias=None if head.bias is None else head.bias.detach().float(),
        norm_weight=norm.weight.detach().float(),
        norm_bias=norm.bias.detach().float(),
        norm_eps=float(norm.eps),
        d_model=arm.d_model,
        vocab_size=vocab,
    )


# -------------------------------------------------------- cohorts and windows


def split_cohort(
    cohort: Cohort, train_fraction: float, seed: int
) -> tuple[Cohort, Cohort]:
    """Split a cohort into disjoint training and evaluation sub-cohorts.

    The tuned lens is a fitted object, so any number it produces on the
    sequences it was fitted on is a training fit. The split unit is the sequence
    rather than the token because tokens inside one sequence are not
    independent: a token-level split would leak the sequence's own statistics
    across the boundary.
    """

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one")
    total = len(cohort)
    n_train = int(round(train_fraction * total))
    if n_train < 1 or total - n_train < 1:
        raise ValueError(
            f"cohort of {total} sequences cannot be split at {train_fraction}"
        )
    generator = np.random.default_rng(seed)
    order = generator.permutation(total)
    parts: list[Cohort] = []
    for label, indices in (
        ("train", sorted(int(i) for i in order[:n_train])),
        ("eval", sorted(int(i) for i in order[n_train:])),
    ):
        metadata: dict[str, Any] = {
            "sampling": {
                "mode": "split",
                "seed": int(seed),
                "requested": len(indices),
                "role": label,
                "train_fraction": float(train_fraction),
                "indices": indices,
                "parent_name": cohort.name,
                "parent_digest": cohort.digest,
                "parent_provenance_digest": cohort.provenance_digest,
                "parent_sampling": cohort.sampling,
            }
        }
        labels = cohort.metadata.get("ec_labels")
        if labels is not None:
            if len(labels) != total:
                raise ValueError(f"cohort {cohort.name!r}: EC labels do not align")
            metadata["ec_labels"] = [labels[i] for i in indices]
        parts.append(
            Cohort(
                name=f"{cohort.name}_{label}{len(indices)}_seed{seed}",
                kind=cohort.kind,
                records=[cohort.records[i] for i in indices],
                min_symbols=cohort.min_symbols,
                max_symbols=cohort.max_symbols,
                metadata=metadata,
            )
        )
    return parts[0], parts[1]


@dataclass(frozen=True)
class ScoredWindow:
    """One tokenised batch and the next-token targets belonging to the cohort."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_mask: torch.Tensor
    sequence_indices: tuple[int, ...]


def prepare_windows(
    arm: Arm, cohort: Cohort, *, max_len: int, batch_size: int
) -> list[ScoredWindow]:
    """Tokenise a cohort once, in the arm's native input format."""

    if max_len < 2 or batch_size < 1:
        raise ValueError("max_len must admit a target and batch_size must be positive")
    texts = cohort.input_strings(arm)
    if not texts:
        raise ValueError(f"{arm.name}: cohort {cohort.name!r} is empty")
    start_id, end_id = conditioning_boundary_ids(arm)
    windows: list[ScoredWindow] = []
    for offset in range(0, len(texts), batch_size):
        chunk = texts[offset : offset + batch_size]
        ids, mask = tokenize_batch(arm, chunk, max_len)
        if ids.shape[1] < 2:
            raise ValueError(
                f"{arm.name}: batch at sequence {offset} tokenises to fewer than two tokens"
            )
        ids = ids.to(arm.device)
        mask = mask.to(arm.device)
        if start_id is not None and end_id is not None:
            complete = (ids == end_id).sum(dim=1).eq(1) & (ids == start_id).sum(dim=1).eq(1)
            if not bool(complete.all()):
                raise ValueError(
                    f"{arm.name}: max_len={max_len} truncates the EC-conditioned prompt "
                    "before its <end> boundary; the scored window would be undefined"
                )
        target_mask = sequence_target_mask(
            ids,
            mask,
            rule=target_rule(arm.spec.input_format),
            start_token_id=start_id,
            end_token_id=end_id,
        )
        empty = torch.nonzero(target_mask.sum(dim=1) < 1, as_tuple=False).flatten()
        if empty.numel() > 0:
            raise ValueError(
                f"{arm.name}: sequences {[offset + int(i) for i in empty]} have no scored targets"
            )
        windows.append(
            ScoredWindow(
                input_ids=ids,
                attention_mask=mask,
                target_mask=target_mask,
                sequence_indices=tuple(range(offset, offset + len(chunk))),
            )
        )
    return windows


def scored_position_count(windows: Sequence[ScoredWindow]) -> int:
    return sum(int(window.target_mask.sum()) for window in windows)


# ------------------------------------------------------------ residual cache


def block_output_tensor(output: Any, label: str, width: int) -> torch.Tensor:
    """The residual stream carried by a block output, tensor or tuple alike.

    GPT-2 blocks and ProGen2 blocks both return tuples whose first element is
    the residual stream; anything else is a panel change and must fail rather
    than be guessed at.
    """

    tensor = output[0] if isinstance(output, tuple) else output
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{label}: block output is not a tensor or tuple of tensors")
    if tensor.ndim != 3 or tensor.shape[-1] != width:
        raise ValueError(
            f"{label}: expected a [batch, token, {width}] residual, got {tuple(tensor.shape)}"
        )
    return tensor


class _BlockCapture:
    """Capture the residual stream at the output of named blocks."""

    def __init__(self, arm: Arm, layers: Sequence[int]) -> None:
        self.arm = arm
        self.layers = tuple(dict.fromkeys(int(layer) for layer in layers))
        if not self.layers:
            raise ValueError("at least one block must be captured")
        if any(not 0 <= layer < arm.n_layer for layer in self.layers):
            raise ValueError(f"{arm.name}: capture layer outside 0..{arm.n_layer - 1}")
        self.captured: dict[int, torch.Tensor] = {}
        self.fired: dict[int, int] = {layer: 0 for layer in self.layers}
        self._handles: list[Any] = []

    def __enter__(self) -> _BlockCapture:
        blocks = self.arm.blocks()
        for layer in self.layers:
            self._handles.append(blocks[layer].register_forward_hook(self._hook(layer)))
        return self

    def __exit__(self, *_exception: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _hook(self, layer: int):
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            self.captured[layer] = block_output_tensor(
                output, f"{self.arm.name} block {layer}", self.arm.d_model
            )
            self.fired[layer] += 1

        return hook


@dataclass
class ResidualCache:
    """Scored-position residual streams, flattened across the cohort.

    Positions are stored on the host because the number of grid layers times the
    number of scored positions is large enough to crowd the accelerator, and the
    lens metrics read them in chunks anyway.
    """

    layers: tuple[int, ...]
    final_layer: int
    residual: dict[int, torch.Tensor]
    final_residual: torch.Tensor
    target_ids: torch.Tensor
    sequence_index: torch.Tensor

    def __post_init__(self) -> None:
        n = int(self.target_ids.shape[0])
        if n < 1:
            raise ValueError("residual cache is empty")
        if self.sequence_index.shape != self.target_ids.shape:
            raise ValueError("cache sequence index does not align with targets")
        if set(self.residual) != set(self.layers):
            raise ValueError("cache layers do not match the stored residuals")
        for layer, values in self.residual.items():
            if values.shape[0] != n:
                raise ValueError(f"cache layer {layer} holds {values.shape[0]} of {n} positions")
        if self.final_residual.shape[0] != n:
            raise ValueError("cache final residual does not align with targets")

    def __len__(self) -> int:
        return int(self.target_ids.shape[0])

    @property
    def n_sequences(self) -> int:
        return int(torch.unique(self.sequence_index).numel())


@torch.inference_mode()
def cache_residuals(
    arm: Arm,
    windows: Sequence[ScoredWindow],
    layers: Sequence[int],
    *,
    max_bytes: int,
) -> ResidualCache:
    """Residual streams at the grid layers, restricted to scored positions.

    The deepest block is always captured whether or not it is on the grid,
    because every lens metric is read against the model's own final
    distribution and that distribution has to come from the same forward pass as
    the intermediate states it is compared with.
    """

    if not windows:
        raise ValueError(f"{arm.name}: no scored windows")
    grid = tuple(dict.fromkeys(int(layer) for layer in layers))
    if not grid:
        raise ValueError("at least one grid layer is required")
    final_layer = arm.n_layer - 1
    wanted = tuple(dict.fromkeys((*grid, final_layer)))
    positions = scored_position_count(windows)
    estimate = len(wanted) * positions * arm.d_model * 4
    if estimate > max_bytes:
        raise RuntimeError(
            f"{arm.name}: residual cache would need {estimate / 2**30:.2f} GiB for "
            f"{positions} positions across {len(wanted)} layers, above the "
            f"{max_bytes / 2**30:.2f} GiB budget; reduce --n-seq, --max-len or --depths"
        )

    per_layer: dict[int, list[torch.Tensor]] = {layer: [] for layer in wanted}
    targets: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    for window in windows:
        with _BlockCapture(arm, wanted) as capture:
            arm.model(
                input_ids=window.input_ids,
                attention_mask=window.attention_mask,
                use_cache=False,
            )
        unexpected = {layer: count for layer, count in capture.fired.items() if count != 1}
        if unexpected:
            raise RuntimeError(f"{arm.name}: capture hooks fired {unexpected} times")
        mask = window.target_mask
        for layer in wanted:
            state = capture.captured[layer][:, :-1, :][mask]
            if not bool(torch.isfinite(state).all()):
                raise FloatingPointError(f"{arm.name}: non-finite residual at layer {layer}")
            per_layer[layer].append(state.float().cpu())
        targets.append(window.input_ids[:, 1:][mask].cpu())
        rows = torch.tensor(window.sequence_indices, device=mask.device, dtype=torch.long)
        indices.append(rows.unsqueeze(1).expand_as(mask)[mask].cpu())

    stacked = {layer: torch.cat(values, dim=0) for layer, values in per_layer.items()}
    return ResidualCache(
        layers=grid,
        final_layer=final_layer,
        residual={layer: stacked[layer] for layer in grid},
        final_residual=stacked[final_layer],
        target_ids=torch.cat(targets, dim=0),
        sequence_index=torch.cat(indices, dim=0),
    )


@torch.inference_mode()
def verify_lens_head(
    arm: Arm,
    head: LensHead,
    window: ScoredWindow,
    *,
    tolerance_nats: float,
) -> dict[str, Any]:
    """Check the float32 lens head against the model's own final distribution.

    Every lens number is a comparison against ``p_final``. If the head does not
    reproduce ``p_final`` then nothing downstream means what it claims to, so
    this is checked on real cohort positions and fails loudly rather than being
    assumed. Running the arm in bfloat16 will exceed a tight tolerance; that is
    the intended signal, not a reason to loosen it.
    """

    if tolerance_nats <= 0.0:
        raise ValueError("lens-head tolerance must be positive")
    with _BlockCapture(arm, (arm.n_layer - 1,)) as capture:
        model_logits = arm.model(
            input_ids=window.input_ids,
            attention_mask=window.attention_mask,
            use_cache=False,
        ).logits
    final_state = capture.captured[arm.n_layer - 1][:, :-1, :][window.target_mask]
    model_log_probs = F.log_softmax(model_logits[:, :-1].float(), dim=-1)[window.target_mask]
    head_log_probs = head.log_probs(final_state)
    kl = (model_log_probs.exp() * (model_log_probs - head_log_probs)).sum(dim=-1)
    max_kl = float(kl.max())
    max_logit_gap = float((model_log_probs - head_log_probs).abs().max())
    if not math.isfinite(max_kl) or max_kl > tolerance_nats:
        raise FloatingPointError(
            f"{arm.name}: float32 lens head disagrees with the model's own final "
            f"distribution by {max_kl:.3e} nats, above the {tolerance_nats:.3e} tolerance; "
            f"inference dtype is {arm.dtype}"
        )
    return {
        "positions": int(kl.numel()),
        "max_kl_nats": max_kl,
        "mean_kl_nats": float(kl.mean()),
        "max_abs_log_prob_difference": max_logit_gap,
        "tolerance_nats": float(tolerance_nats),
    }


# ------------------------------------------------------------ lens trajectory


class AffineTranslator(nn.Module):
    """``h -> h + A h + b``: the tuned-lens translator, identity-initialised.

    Parameterising the translator as a residual correction with ``A`` and ``b``
    zero-initialised means the fit *starts* exactly at the untuned logit lens.
    The reported improvement over untuned therefore starts at zero and can only
    be produced by optimisation, and any layer where the tuned lens is worse than
    untuned on held-out data is unambiguously overfitting rather than a bad
    initialisation.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        self.correction = nn.Linear(d_model, d_model, bias=True)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.correction(hidden)


def _sequence_slots(sequence_index: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Map global sequence indices onto a contiguous range for accumulation."""

    unique, inverse = torch.unique(sequence_index, sorted=True, return_inverse=True)
    return inverse, int(unique.numel())


@torch.no_grad()
def lens_trajectory(
    head: LensHead,
    cache: ResidualCache,
    *,
    device: str,
    chunk: int,
    translators: Mapping[int, AffineTranslator] | None = None,
) -> dict[int, list[dict[str, float | int]]]:
    """Per-sequence lens sums at every grid layer, against the final distribution.

    Sums rather than means, because the cluster bootstrap resamples sequences and
    a mean of per-sequence means would silently weight short sequences like long
    ones.

    Per-sequence accumulation happens on the accelerator and is brought back to
    the host once, at the end. Scattering each chunk's results into host-side
    accumulators would put a synchronising device-to-host copy and a
    thread-pool-bound scatter inside the inner loop.
    """

    if chunk < 1:
        raise ValueError("chunk must be positive")
    slots, n_slots = _sequence_slots(cache.sequence_index)
    slots = slots.to(device)
    fields = ("token_count", "ce_sum", "kl_sum", "agreement_count", "entropy_sum")
    accumulators = {
        layer: {
            name: torch.zeros(n_slots, dtype=torch.float64, device=device) for name in fields
        }
        for layer in cache.layers
    }
    if translators is not None and set(translators) != set(cache.layers):
        raise ValueError("translators do not cover exactly the cached grid layers")

    total = len(cache)
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        slot = slots[start:stop]
        targets = cache.target_ids[start:stop].to(device)
        final_log_probs = head.log_probs(cache.final_residual[start:stop].to(device))
        final_probs = final_log_probs.exp()
        final_top1 = final_log_probs.argmax(dim=-1)
        ones = torch.ones(stop - start, dtype=torch.float64, device=device)
        for layer in cache.layers:
            state = cache.residual[layer][start:stop].to(device)
            if translators is not None:
                state = translators[layer](state)
            log_probs = head.log_probs(state)
            ce = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            kl = (final_probs * (final_log_probs - log_probs)).sum(dim=-1)
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            agreement = (log_probs.argmax(dim=-1) == final_top1).double()
            if not bool(torch.isfinite(ce).all() and torch.isfinite(kl).all()):
                raise FloatingPointError(f"non-finite lens metrics at layer {layer}")
            bucket = accumulators[layer]
            bucket["token_count"].index_add_(0, slot, ones)
            bucket["ce_sum"].index_add_(0, slot, ce.double())
            bucket["kl_sum"].index_add_(0, slot, kl.double())
            bucket["entropy_sum"].index_add_(0, slot, entropy.double())
            bucket["agreement_count"].index_add_(0, slot, agreement)

    rows_by_layer: dict[int, list[dict[str, float | int]]] = {}
    for layer, device_bucket in accumulators.items():
        bucket = {name: values.cpu() for name, values in device_bucket.items()}
        rows: list[dict[str, float | int]] = []
        for slot_index in range(n_slots):
            count = int(bucket["token_count"][slot_index])
            if count < 1:
                raise ValueError(f"layer {layer}: a sequence contributed no scored tokens")
            rows.append(
                {
                    "token_count": count,
                    "ce_sum": float(bucket["ce_sum"][slot_index]),
                    "kl_sum": float(bucket["kl_sum"][slot_index]),
                    "agreement_count": float(bucket["agreement_count"][slot_index]),
                    "entropy_sum": float(bucket["entropy_sum"][slot_index]),
                }
            )
        rows_by_layer[layer] = rows
    return rows_by_layer


def lens_metrics(rows: Sequence[Mapping[str, float | int]]) -> dict[str, Any]:
    """Token-weighted lens metrics for one layer."""

    if not rows:
        raise ValueError("cannot aggregate an empty sequence set")
    tokens = sum(int(row["token_count"]) for row in rows)
    if tokens < 1:
        raise ValueError("aggregate contains no scored targets")
    return {
        "schema_version": SCHEMA_VERSION_LOGIT_LENS,
        "ce_nats": sum(float(row["ce_sum"]) for row in rows) / tokens,
        "kl_to_final_nats": sum(float(row["kl_sum"]) for row in rows) / tokens,
        "top1_agreement_with_final": sum(float(row["agreement_count"]) for row in rows) / tokens,
        "entropy_nats": sum(float(row["entropy_sum"]) for row in rows) / tokens,
        "scored_tokens": tokens,
        "sequences": len(rows),
    }


_LENS_RATE_KEYS = ("ce_nats", "kl_to_final_nats", "entropy_nats")


def lens_cluster_bootstrap(
    rows: Sequence[Mapping[str, float | int]], *, samples: int, seed: int
) -> dict[str, Any]:
    """Sequence-cluster bootstrap of one layer's lens metrics.

    Tokens within a sequence share context and are not independent, so the
    resampling unit is the sequence. Point estimates are never reported without
    this interval.
    """

    if not rows:
        raise ValueError("bootstrap needs at least one sequence row")
    if samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    generator = np.random.default_rng(seed)
    n = len(rows)
    draws: dict[str, list[float]] = {
        key: [] for key in (*_LENS_RATE_KEYS, "top1_agreement_with_final")
    }
    for _ in range(samples):
        indices = generator.integers(0, n, size=n)
        metrics = lens_metrics([rows[int(index)] for index in indices])
        for key in draws:
            draws[key].append(float(metrics[key]))
    return {
        "schema_version": SCHEMA_VERSION_BOOTSTRAP,
        "cluster_unit": "sequence",
        "samples": int(samples),
        "seed": int(seed),
        "clusters": n,
        **{key: quantile_interval(values) for key, values in draws.items()},
    }


def quantile_interval(values: Sequence[float]) -> dict[str, float]:
    """Percentile summary of a bootstrap draw set."""

    array = np.asarray(values, dtype=np.float64)
    if array.size < 1:
        raise ValueError("interval needs at least one draw")
    if not np.isfinite(array).all():
        raise ValueError("interval draws contain non-finite values")
    return {
        "q025": float(np.quantile(array, 0.025)),
        "median": float(np.quantile(array, 0.5)),
        "q975": float(np.quantile(array, 0.975)),
    }


def per_symbol_view(metrics: Mapping[str, Any], symbols_per_token: float) -> dict[str, Any]:
    """Re-express token-rate metrics per symbol.

    ProtGPT2 emits several residues per token while ZymCTRL and ProGen2-medium
    emit one, so a per-token cross-entropy is not comparable across the protein
    arms. Dividing by the measured expansion converts a nats-per-token rate into
    a nats-per-symbol rate, which is comparable. Top-1 agreement is a per-token
    event and has no per-symbol reading; it is named here so that its absence is
    explicit rather than an omission.
    """

    if not symbols_per_token > 0.0:
        raise ValueError("symbols_per_token must be positive")
    view: dict[str, Any] = {
        "symbols_per_token": float(symbols_per_token),
        "not_convertible_to_per_symbol": ["top1_agreement_with_final"],
    }
    for key in _LENS_RATE_KEYS:
        value = float(metrics[key])
        view[f"{key}_per_symbol"] = value / symbols_per_token
        view[f"{key}_bits_per_symbol"] = value / LN2 / symbols_per_token
    return view


def trajectory_summary(
    grid: Sequence[LayerPoint], metrics_by_layer: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    """Shape of a lens trajectory across depth.

    Whether cross-entropy falls and agreement rises monotonically with depth is
    the primary sanity anchor for the layer-norm handling: a text decoder whose
    logit-lens trajectory does not improve with depth indicates a broken final
    normalisation rather than an interesting result.
    """

    if not grid:
        raise ValueError("trajectory summary needs at least one grid point")
    layers = [point.layer for point in grid]
    missing = [layer for layer in layers if layer not in metrics_by_layer]
    if missing:
        raise ValueError(f"trajectory summary is missing layers {missing}")
    ce = [float(metrics_by_layer[layer]["ce_nats"]) for layer in layers]
    kl = [float(metrics_by_layer[layer]["kl_to_final_nats"]) for layer in layers]
    agreement = [
        float(metrics_by_layer[layer]["top1_agreement_with_final"]) for layer in layers
    ]
    entropy = [float(metrics_by_layer[layer]["entropy_nats"]) for layer in layers]

    def non_increasing(values: list[float]) -> bool:
        return all(b <= a + 1e-12 for a, b in zip(values, values[1:]))

    def non_decreasing(values: list[float]) -> bool:
        return all(b >= a - 1e-12 for a, b in zip(values, values[1:]))

    return {
        "layers": layers,
        "relative_depth": [point.relative_depth for point in grid],
        "ce_nats": ce,
        "kl_to_final_nats": kl,
        "top1_agreement_with_final": agreement,
        "entropy_nats": entropy,
        "ce_shallowest_minus_deepest_nats": ce[0] - ce[-1],
        "kl_shallowest_minus_deepest_nats": kl[0] - kl[-1],
        "ce_monotone_non_increasing_with_depth": non_increasing(ce),
        "kl_monotone_non_increasing_with_depth": non_increasing(kl),
        "agreement_monotone_non_decreasing_with_depth": non_decreasing(agreement),
    }


# -------------------------------------------------- residue-class trajectory


@dataclass(frozen=True)
class ResidueVocabulary:
    """Mapping from output tokens onto the residue each token starts with.

    For residue-level arms this is an identity between the twenty single-residue
    tokens and the twenty residues. For a multi-residue BPE arm a token emits
    several residues at once, and the only residue-level question its next-token
    distribution answers is *which residue comes next*; the mapping therefore
    keys on the first residue of the token. Tokens whose decoded form is not
    made entirely of canonical residues are excluded, which drops control tokens
    and, for the GPT-2 vocabulary ProtGPT2 inherits, ordinary word pieces.
    """

    token_ids: tuple[int, ...]
    leading_residue: tuple[str, ...]
    residues_per_token: tuple[int, ...]
    group_index: tuple[torch.Tensor, ...]
    vocab_size: int

    @property
    def n_mapped_tokens(self) -> int:
        return len(self.token_ids)


def residue_vocabulary(arm: Arm, *, device: str) -> ResidueVocabulary:
    """Build the token-to-residue mapping for a protein arm."""

    if arm.modality != "protein":
        raise ValueError(f"{arm.name}: residue vocabulary is defined for protein arms only")
    tokenizer = arm.tokenizer
    vocab_size = int(arm.model.config.vocab_size)
    limit = min(vocab_size, len(tokenizer))
    allowed = set(AA20)
    token_ids: list[int] = []
    leading: list[str] = []
    lengths: list[int] = []
    for token_id in range(limit):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        if piece is None:
            continue
        text = tokenizer.convert_tokens_to_string([piece]).strip()
        if not text or any(character not in allowed for character in text):
            continue
        token_ids.append(token_id)
        leading.append(text[0])
        lengths.append(len(text))
    if not token_ids:
        raise ValueError(f"{arm.name}: no output token decodes to canonical residues")
    covered = {residue for residue in leading}
    if covered != allowed:
        raise ValueError(
            f"{arm.name}: residue mapping covers {len(covered)} of 20 residues; "
            f"missing {sorted(allowed - covered)}"
        )
    groups = tuple(
        torch.tensor(
            [token for token, residue in zip(token_ids, leading) if residue == target],
            dtype=torch.long,
            device=device,
        )
        for target in AA20
    )
    return ResidueVocabulary(
        token_ids=tuple(token_ids),
        leading_residue=tuple(leading),
        residues_per_token=tuple(lengths),
        group_index=groups,
        vocab_size=vocab_size,
    )


_CLASS_MEMBERSHIP = torch.tensor(
    [[1.0 if CLASS_OF_RESIDUE[residue] == name else 0.0 for residue in AA20] for name in CLASS_NAMES],
    dtype=torch.float32,
)


def _residue_log_mass(log_probs: torch.Tensor, vocabulary: ResidueVocabulary) -> torch.Tensor:
    """Log probability mass on each of the twenty residues, per position."""

    columns = [
        torch.logsumexp(log_probs.index_select(-1, group), dim=-1)
        for group in vocabulary.group_index
    ]
    return torch.stack(columns, dim=-1)


def _class_log_mass(residue_log_mass: torch.Tensor) -> torch.Tensor:
    membership = _CLASS_MEMBERSHIP.to(residue_log_mass.device)
    masked = residue_log_mass.unsqueeze(-2) + torch.log(membership).unsqueeze(0)
    return torch.logsumexp(masked, dim=-1)


@torch.no_grad()
def residue_class_trajectory(
    head: LensHead,
    cache: ResidualCache,
    vocabulary: ResidueVocabulary,
    *,
    device: str,
    chunk: int,
    translators: Mapping[int, AffineTranslator] | None = None,
) -> tuple[dict[int, list[dict[str, float | int]]], dict[int, dict[str, Any]]]:
    """Does depth buy coarse-to-fine residue structure?

    At every grid layer the lens distribution is restricted to residue-bearing
    tokens and renormalised, giving a distribution over the twenty residues.
    Because ``AA_CLASSES`` is a partition, that distribution factorises exactly
    into a distribution over four chemical classes and a within-class
    distribution, and residue cross-entropy decomposes into class cross-entropy
    plus a within-class term. Coarse-to-fine structure means the class term
    resolves at shallower depth than the within-class term.

    Returns per-sequence rows for the bootstrap and, separately, the mean
    residue and class marginals at each layer, which are what "the top predicted
    amino acids at this layer" is read from.
    """

    if chunk < 1:
        raise ValueError("chunk must be positive")
    if translators is not None and set(translators) != set(cache.layers):
        raise ValueError("translators do not cover exactly the cached grid layers")
    residue_of_token = torch.full((vocabulary.vocab_size,), -1, dtype=torch.long)
    for token_id, residue in zip(vocabulary.token_ids, vocabulary.leading_residue):
        residue_of_token[token_id] = AA20.index(residue)
    residue_of_token = residue_of_token.to(device)
    class_of_residue = torch.tensor(
        [CLASS_NAMES.index(CLASS_OF_RESIDUE[residue]) for residue in AA20],
        dtype=torch.long,
        device=device,
    )

    slots, n_slots = _sequence_slots(cache.sequence_index)
    slots = slots.to(device)
    fields = (
        "token_count",
        "residue_mass_sum",
        "class_ce_sum",
        "residue_ce_sum",
        "class_correct",
        "residue_correct",
        "class_entropy_sum",
        "within_class_entropy_sum",
    )
    accumulators = {
        layer: {
            name: torch.zeros(n_slots, dtype=torch.float64, device=device) for name in fields
        }
        for layer in cache.layers
    }
    marginals = {
        layer: {
            "residue": torch.zeros(len(AA20), dtype=torch.float64, device=device),
            "class": torch.zeros(len(CLASS_NAMES), dtype=torch.float64, device=device),
            "positions": 0,
        }
        for layer in cache.layers
    }

    total = len(cache)
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        targets = cache.target_ids[start:stop].to(device)
        true_residue = residue_of_token[targets]
        keep = true_residue >= 0
        n_kept = int(keep.sum())
        if n_kept < 1:
            continue
        slot = slots[start:stop][keep]
        true_residue = true_residue[keep]
        true_class = class_of_residue[true_residue]
        ones = torch.ones(n_kept, dtype=torch.float64, device=device)
        for layer in cache.layers:
            state = cache.residual[layer][start:stop].to(device)
            if translators is not None:
                state = translators[layer](state)
            log_probs = head.log_probs(state)[keep]
            residue_log_mass = _residue_log_mass(log_probs, vocabulary)
            total_log_mass = torch.logsumexp(residue_log_mass, dim=-1)
            residue_log_p = residue_log_mass - total_log_mass.unsqueeze(-1)
            class_log_p = _class_log_mass(residue_log_p)
            residue_p = residue_log_p.exp()
            class_p = class_log_p.exp()
            residue_entropy = -(residue_p * residue_log_p).sum(dim=-1)
            class_entropy = -(class_p * class_log_p).sum(dim=-1)
            bucket = accumulators[layer]
            bucket["token_count"].index_add_(0, slot, ones)
            bucket["residue_mass_sum"].index_add_(0, slot, total_log_mass.exp().double())
            bucket["class_ce_sum"].index_add_(
                0,
                slot,
                (-class_log_p.gather(-1, true_class.unsqueeze(-1)).squeeze(-1)).double(),
            )
            bucket["residue_ce_sum"].index_add_(
                0,
                slot,
                (-residue_log_p.gather(-1, true_residue.unsqueeze(-1)).squeeze(-1)).double(),
            )
            bucket["class_correct"].index_add_(
                0, slot, (class_log_p.argmax(dim=-1) == true_class).double()
            )
            bucket["residue_correct"].index_add_(
                0, slot, (residue_log_p.argmax(dim=-1) == true_residue).double()
            )
            bucket["class_entropy_sum"].index_add_(0, slot, class_entropy.double())
            bucket["within_class_entropy_sum"].index_add_(
                0, slot, (residue_entropy - class_entropy).double()
            )
            marginals[layer]["residue"] += residue_p.double().sum(dim=0)
            marginals[layer]["class"] += class_p.double().sum(dim=0)
            marginals[layer]["positions"] += n_kept

    rows_by_layer: dict[int, list[dict[str, float | int]]] = {}
    for layer, device_bucket in accumulators.items():
        bucket = {name: values.cpu() for name, values in device_bucket.items()}
        rows: list[dict[str, float | int]] = []
        for slot_index in range(n_slots):
            count = int(bucket["token_count"][slot_index])
            if count < 1:
                continue
            rows.append(
                {name: float(bucket[name][slot_index]) for name in fields if name != "token_count"}
                | {"token_count": count}
            )
        if not rows:
            raise ValueError(f"layer {layer}: no residue-bearing targets in the cohort")
        rows_by_layer[layer] = rows

    marginal_report: dict[int, dict[str, Any]] = {}
    for layer, values in marginals.items():
        positions = int(values["positions"])
        if positions < 1:
            raise ValueError(f"layer {layer}: no residue-bearing positions")
        residue_mean = (values["residue"] / positions).cpu().numpy()
        class_mean = (values["class"] / positions).cpu().numpy()
        order = np.argsort(-residue_mean)
        marginal_report[layer] = {
            "positions": positions,
            "mean_residue_probability": {
                AA20[i]: float(residue_mean[i]) for i in range(len(AA20))
            },
            "mean_class_probability": {
                CLASS_NAMES[i]: float(class_mean[i]) for i in range(len(CLASS_NAMES))
            },
            "top_residues": [
                {
                    "residue": AA20[int(i)],
                    "chemical_class": CLASS_OF_RESIDUE[AA20[int(i)]],
                    "mean_probability": float(residue_mean[int(i)]),
                }
                for i in order[:5]
            ],
        }
    return rows_by_layer, marginal_report


def residue_class_metrics(rows: Sequence[Mapping[str, float | int]]) -> dict[str, Any]:
    """Token-weighted residue and class metrics for one layer."""

    if not rows:
        raise ValueError("cannot aggregate an empty sequence set")
    tokens = sum(int(row["token_count"]) for row in rows)
    if tokens < 1:
        raise ValueError("aggregate contains no residue-bearing targets")

    def rate(name: str) -> float:
        return sum(float(row[name]) for row in rows) / tokens

    class_ce = rate("class_ce_sum")
    residue_ce = rate("residue_ce_sum")
    return {
        "schema_version": SCHEMA_VERSION_RESIDUE_CLASS,
        "residue_mass_fraction": rate("residue_mass_sum"),
        "class_ce_nats": class_ce,
        "residue_ce_nats": residue_ce,
        "within_class_ce_nats": residue_ce - class_ce,
        "class_top1_accuracy": rate("class_correct"),
        "residue_top1_accuracy": rate("residue_correct"),
        "class_entropy_nats": rate("class_entropy_sum"),
        "within_class_entropy_nats": rate("within_class_entropy_sum"),
        "scored_residue_targets": tokens,
        "sequences": len(rows),
    }


_RESIDUE_CLASS_KEYS = (
    "residue_mass_fraction",
    "class_ce_nats",
    "residue_ce_nats",
    "within_class_ce_nats",
    "class_top1_accuracy",
    "residue_top1_accuracy",
    "class_entropy_nats",
    "within_class_entropy_nats",
)


def residue_class_cluster_bootstrap(
    rows: Sequence[Mapping[str, float | int]], *, samples: int, seed: int
) -> dict[str, Any]:
    """Sequence-cluster bootstrap of one layer's residue-class metrics."""

    if not rows:
        raise ValueError("bootstrap needs at least one sequence row")
    if samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    generator = np.random.default_rng(seed)
    n = len(rows)
    draws: dict[str, list[float]] = {key: [] for key in _RESIDUE_CLASS_KEYS}
    for _ in range(samples):
        indices = generator.integers(0, n, size=n)
        metrics = residue_class_metrics([rows[int(index)] for index in indices])
        for key in draws:
            draws[key].append(float(metrics[key]))
    return {
        "schema_version": SCHEMA_VERSION_BOOTSTRAP,
        "cluster_unit": "sequence",
        "samples": int(samples),
        "seed": int(seed),
        "clusters": n,
        **{key: quantile_interval(values) for key, values in draws.items()},
    }


def resolution_depth(
    depths: Sequence[float], values: Sequence[float], tau: float
) -> float | None:
    """Relative depth at which ``tau`` of a quantity's total reduction is reached.

    Returns ``None`` when the quantity does not fall across the grid, because the
    crossing point is undefined in that case and reporting one would invent a
    resolution depth for a trajectory that never resolved.

    The fraction is a parameter because a depth ordering read at one fraction is
    a threshold result: Appendix B rule 17 requires it to survive a sweep, which
    :data:`src.transfer.concept_lens.RESOLUTION_TAUS` supplies.
    """

    if not 0.0 < tau < 1.0:
        raise ValueError("tau must lie strictly between zero and one")
    if len(depths) != len(values) or len(depths) < 2:
        raise ValueError("depths and values must be aligned vectors of length at least two")
    start, end = float(values[0]), float(values[-1])
    span = start - end
    if not math.isfinite(span) or span <= 0.0:
        return None
    threshold = start - tau * span
    for index in range(1, len(values)):
        previous, current = float(values[index - 1]), float(values[index])
        if current <= threshold <= previous:
            if math.isclose(previous, current):
                return float(depths[index])
            weight = (previous - threshold) / (previous - current)
            return float(depths[index - 1]) + weight * (
                float(depths[index]) - float(depths[index - 1])
            )
    return float(depths[-1])


def half_resolution_depth(depths: Sequence[float], values: Sequence[float]) -> float | None:
    """Relative depth at which half of a quantity's total reduction is reached.

    The name the published lens fields are keyed on, and one call of
    :func:`resolution_depth`. The two used to be separate implementations in two
    modules, agreeing at ``tau = 0.5`` because a test held them to it; they now
    agree because there is one of them.
    """

    return resolution_depth(depths, values, 0.5)


def coarse_to_fine_gap(
    depths: Sequence[float], class_values: Sequence[float], within_values: Sequence[float]
) -> dict[str, Any]:
    """Depth ordering of coarse (class) versus fine (within-class) resolution.

    A positive gap means the class term reaches half of its total reduction at
    shallower relative depth than the within-class term: the trajectory passes
    through chemically coherent groups before committing to specific residues.
    """

    class_depth = half_resolution_depth(depths, class_values)
    within_depth = half_resolution_depth(depths, within_values)
    gap = (
        None
        if class_depth is None or within_depth is None
        else float(within_depth - class_depth)
    )
    return {
        "class_half_resolution_depth": class_depth,
        "within_class_half_resolution_depth": within_depth,
        "coarse_to_fine_depth_gap": gap,
        "coarse_to_fine": None if gap is None else bool(gap > 0.0),
        "undefined_reason": (
            None
            if gap is not None
            else "one of the two terms does not decrease across the grid"
        ),
    }


# ------------------------------------------------------------- tuned lens fit


def train_tuned_lens(
    head: LensHead,
    cache: ResidualCache,
    *,
    device: str,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    log_every: int,
    max_bytes: int,
    progress: bool,
) -> tuple[dict[int, AffineTranslator], dict[str, Any]]:
    """Fit one affine translator per grid layer against the final distribution.

    The objective is ``KL(p_final || p_lens)`` averaged over positions, following
    the tuned-lens formulation: the model's own final distribution is the
    reference and the translator is fitted to reproduce it from an intermediate
    state. Forward KL is used rather than reverse KL because the reference is the
    thing being explained; a reverse-KL fit would be free to drop modes the model
    actually predicts.

    All fitted layers are trained jointly in one loop. The translators do not
    interact, but the reference distribution is shared, so training them
    together computes it once per minibatch instead of once per layer.

    The deepest grid layer is held at the exact identity and excluded from the
    optimisation. Its residual stream *is* the final state, so its lens
    distribution already equals the reference and its KL is exactly zero, which
    is the global minimum of a non-negative objective; the analytic solution is
    therefore known and the sub-problem is degenerate. Handing it to Adam would
    not leave it alone: the true gradient vanishes, Adam divides by the running
    root-mean-square of the gradient, and the surviving floating-point round-off
    is rescaled into full-size steps. The translator would drift away from the
    optimum and the drift would be reported as a lens deficit that does not
    exist.

    The training split is moved onto the accelerator for the duration of the
    fit. Random minibatch gathers are the inner loop, and a host-side gather is
    dominated by the PyTorch CPU thread pool: on a 96-core host, selecting 256
    rows from a 17808 x 1280 host tensor measures 105 ms against 0.07 ms
    single-threaded. Gathering on the device removes that pathology and the
    per-step host-to-device copy at once, at the cost of a declared and
    guarded amount of accelerator memory.
    """

    if steps < 1 or batch_size < 1 or log_every < 1:
        raise ValueError("steps, batch_size and log_every must be positive")
    if not learning_rate > 0.0 or weight_decay < 0.0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    total = len(cache)
    if total < batch_size:
        raise ValueError(
            f"tuned lens: {total} training positions cannot fill a batch of {batch_size}"
        )
    fitted = tuple(layer for layer in cache.layers if layer != cache.final_layer)
    identity = tuple(layer for layer in cache.layers if layer == cache.final_layer)
    if not fitted:
        raise ValueError("the layer grid contains no fittable layer")
    resident_bytes = (len(fitted) + 1) * total * head.d_model * 4
    if resident_bytes > max_bytes:
        raise RuntimeError(
            f"tuned lens: making the training split device-resident needs "
            f"{resident_bytes / 2**30:.2f} GiB, above the {max_bytes / 2**30:.2f} GiB "
            "budget; reduce --n-seq, --max-len or --depths"
        )
    resident = {layer: cache.residual[layer].to(device) for layer in fitted}
    resident_final = cache.final_residual.to(device)
    torch.manual_seed(seed)
    translators = {
        layer: AffineTranslator(head.d_model).to(device=device, dtype=torch.float32)
        for layer in cache.layers
    }
    parameters = [
        parameter
        for layer in fitted
        for parameter in translators[layer].parameters()
    ]
    optimiser = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps)
    generator = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []

    for step in range(1, steps + 1):
        indices = torch.from_numpy(
            generator.integers(0, total, size=batch_size).astype(np.int64)
        ).to(device)
        with torch.no_grad():
            final_log_probs = head.log_probs(resident_final.index_select(0, indices))
            final_probs = final_log_probs.exp()
        optimiser.zero_grad(set_to_none=True)
        losses: dict[int, float] = {}
        objective = torch.zeros((), device=device, dtype=torch.float32)
        for layer in fitted:
            state = resident[layer].index_select(0, indices)
            log_probs = head.log_probs(translators[layer](state))
            kl = (final_probs * (final_log_probs - log_probs)).sum(dim=-1).mean()
            objective = objective + kl
            losses[layer] = float(kl.detach())
        if not torch.isfinite(objective):
            raise FloatingPointError(f"tuned lens: non-finite objective at step {step}")
        objective.backward()
        optimiser.step()
        schedule.step()
        if step % log_every == 0 or step == steps:
            entry = {
                "step": step,
                "learning_rate": float(schedule.get_last_lr()[0]),
                "mean_train_kl_nats": sum(losses.values()) / len(losses),
                "train_kl_nats_by_layer": {str(layer): losses[layer] for layer in fitted},
            }
            history.append(entry)
            if progress:
                print(
                    f"    tuned lens step {step}/{steps} "
                    f"mean train KL {entry['mean_train_kl_nats']:.4f} nats",
                    flush=True,
                )

    del resident, resident_final
    for translator in translators.values():
        translator.eval()
        for parameter in translator.parameters():
            parameter.requires_grad_(False)
    return translators, {
        "schema_version": SCHEMA_VERSION_TUNED_LENS,
        "objective": "forward_kl_final_to_lens_minibatch_mean",
        "parameterisation": "h -> h + A h + b, A and b zero-initialised",
        "steps": int(steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "lr_schedule": "cosine_annealing_to_zero",
        "optimiser": "adamw",
        "seed": int(seed),
        "train_positions": total,
        "train_sequences": cache.n_sequences,
        "positions_drawn": int(steps) * int(batch_size),
        "parameters_per_layer": head.d_model * head.d_model + head.d_model,
        "device_resident_cache_bytes": int(resident_bytes),
        "fitted_layers": [int(layer) for layer in fitted],
        "identity_layers": [int(layer) for layer in identity],
        "identity_layer_rationale": (
            "the deepest residual stream is the final state, so the identity translator "
            "attains the objective's global minimum of zero exactly; it is held there "
            "rather than optimised"
        ),
        "history": history,
    }


def tuned_versus_untuned(
    grid: Sequence[LayerPoint],
    untuned: Mapping[int, Mapping[str, Any]],
    tuned: Mapping[int, Mapping[str, Any]],
    *,
    identity_layer: int,
    tolerance_nats: float,
) -> dict[str, Any]:
    """Layer-by-layer improvement of the tuned lens over the untuned lens.

    The deepest grid layer is the identity case: its residual stream already is
    the final state, so both lenses sit at zero KL and no improvement is
    possible. It is reported but excluded from the "improves at every layer"
    verdict, which it could otherwise only fail by floating-point noise.
    """

    if tolerance_nats < 0.0:
        raise ValueError("tolerance must be non-negative")
    rows: list[dict[str, Any]] = []
    for point in grid:
        before = untuned[point.layer]
        after = tuned[point.layer]
        rows.append(
            {
                "layer": point.layer,
                "relative_depth": point.relative_depth,
                "is_identity_layer": point.layer == identity_layer,
                "kl_to_final_nats_untuned": float(before["kl_to_final_nats"]),
                "kl_to_final_nats_tuned": float(after["kl_to_final_nats"]),
                "kl_reduction_nats": float(before["kl_to_final_nats"])
                - float(after["kl_to_final_nats"]),
                "ce_nats_untuned": float(before["ce_nats"]),
                "ce_nats_tuned": float(after["ce_nats"]),
                "ce_reduction_nats": float(before["ce_nats"]) - float(after["ce_nats"]),
                "agreement_untuned": float(before["top1_agreement_with_final"]),
                "agreement_tuned": float(after["top1_agreement_with_final"]),
            }
        )
    non_identity = [row for row in rows if not row["is_identity_layer"]]
    if not non_identity:
        raise ValueError("the layer grid contains only the identity layer")
    return {
        "per_layer": rows,
        "tolerance_nats": float(tolerance_nats),
        "kl_improves_at_every_non_identity_layer": all(
            row["kl_reduction_nats"] >= -tolerance_nats for row in non_identity
        ),
        "kl_strictly_improves_at_every_non_identity_layer": all(
            row["kl_reduction_nats"] > tolerance_nats for row in non_identity
        ),
        "ce_improves_at_every_non_identity_layer": all(
            row["ce_reduction_nats"] >= -tolerance_nats for row in non_identity
        ),
        "mean_kl_reduction_nats": sum(row["kl_reduction_nats"] for row in non_identity)
        / len(non_identity),
        "worst_layer": min(non_identity, key=lambda row: row["kl_reduction_nats"])["layer"],
    }


# ------------------------------------------------------------ Jacobian lens


class _JacobianTap:
    """Inject a differentiable zero at every grid layer and capture the last block.

    Adding a zero-valued leaf tensor to a block output makes that residual
    stream a graph input without detaching anything above it, so a single
    backward pass yields the Jacobian with respect to *every* grid layer at once.
    That is what makes the exact ``d_model``-column Jacobian affordable: the cost
    is one forward plus ``d_model / chunk`` backward passes per probe position,
    shared across the whole layer grid.
    """

    def __init__(self, arm: Arm, layers: Sequence[int]) -> None:
        self.arm = arm
        self.layers = tuple(dict.fromkeys(int(layer) for layer in layers))
        if not self.layers:
            raise ValueError("at least one grid layer is required")
        if any(not 0 <= layer < arm.n_layer for layer in self.layers):
            raise ValueError(f"{arm.name}: grid layer outside 0..{arm.n_layer - 1}")
        self.epsilons: dict[int, torch.Tensor] = {}
        self.final_state: torch.Tensor | None = None
        self.fired: dict[int, int] = {layer: 0 for layer in self.layers}
        self._handles: list[Any] = []

    def __enter__(self) -> _JacobianTap:
        blocks = self.arm.blocks()
        for layer in self.layers:
            self._handles.append(blocks[layer].register_forward_hook(self._perturb(layer)))
        # Registered last so that it observes the perturbed output when the
        # deepest block is itself a grid layer.
        self._handles.append(
            blocks[self.arm.n_layer - 1].register_forward_hook(self._capture_final)
        )
        return self

    def __exit__(self, *_exception: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _perturb(self, layer: int):
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> Any:
            tensor = block_output_tensor(
                output, f"{self.arm.name} block {layer}", self.arm.d_model
            )
            epsilon = torch.zeros_like(tensor, requires_grad=True)
            self.epsilons[layer] = epsilon
            self.fired[layer] += 1
            perturbed = tensor + epsilon
            if isinstance(output, tuple):
                return (perturbed, *output[1:])
            return perturbed

        return hook

    def _capture_final(self, _module: nn.Module, _inputs: Any, output: Any) -> None:
        self.final_state = block_output_tensor(
            output, f"{self.arm.name} final block", self.arm.d_model
        )


def freeze_parameters(arm: Arm) -> None:
    """Stop the Jacobian backward from allocating parameter gradients.

    The J-lens differentiates with respect to activations only. Leaving the
    parameters differentiable would build and retain a full parameter-gradient
    graph for every one of the ``d_model / chunk`` backward passes.
    """

    for parameter in arm.model.parameters():
        parameter.requires_grad_(False)


@dataclass(frozen=True)
class JacobianProbe:
    """One (sequence, position) pair at which the Jacobian is evaluated."""

    sequence_index: int
    position: int
    context_tokens: int
    input_ids: torch.Tensor


def sample_jacobian_probes(
    windows: Sequence[ScoredWindow],
    *,
    count: int,
    relative_position: float,
    seed: int,
) -> list[JacobianProbe]:
    """One probe per sampled sequence, at a fixed relative position.

    Probes are drawn from distinct sequences so that the cluster bootstrap over
    probes remains a sequence-cluster bootstrap. The query position is the
    scored position at a fixed quantile of each sequence's scored range rather
    than a random one, so that the compute per probe and the amount of visible
    context are matched across arms instead of varying with the draw.
    """

    if count < 1:
        raise ValueError("probe count must be positive")
    if not 0.0 < relative_position <= 1.0:
        raise ValueError("relative_position must lie in (0, 1]")
    candidates: list[tuple[int, torch.Tensor, int]] = []
    for window in windows:
        for row, sequence_index in enumerate(window.sequence_indices):
            scored = torch.nonzero(window.target_mask[row], as_tuple=False).flatten()
            if scored.numel() < 1:
                raise ValueError(f"sequence {sequence_index} has no scored positions")
            rank = min(int(math.ceil(relative_position * scored.numel())) - 1, scored.numel() - 1)
            position = int(scored[max(rank, 0)])
            candidates.append((sequence_index, window.input_ids[row].detach().clone(), position))
    if len(candidates) < count:
        raise ValueError(
            f"only {len(candidates)} sequences available for {count} Jacobian probes"
        )
    generator = np.random.default_rng(seed)
    chosen = sorted(int(i) for i in generator.choice(len(candidates), size=count, replace=False))
    probes: list[JacobianProbe] = []
    for index in chosen:
        sequence_index, ids, position = candidates[index]
        probes.append(
            JacobianProbe(
                sequence_index=sequence_index,
                position=position,
                context_tokens=position + 1,
                input_ids=ids[: position + 1],
            )
        )
    return probes


def jacobian_gram(head: LensHead) -> torch.Tensor:
    """``W~^T W~`` for the centred unembedding, in float64.

    The full Jacobian ``J = W~ M`` is ``vocab x d_model`` and never has to be
    materialised: every quantity reported here depends on ``J`` only through
    ``J^T J = M^T (W~^T W~) M``, which is ``d_model x d_model``. Forming the Gram
    matrix once per arm turns an otherwise infeasible 50257-row SVD into a
    1280-dimensional symmetric eigenproblem, exactly.
    """

    centred = head.centred_weight().double()
    return centred.transpose(0, 1) @ centred


def jacobian_matrices(
    arm: Arm,
    head: LensHead,
    probe: JacobianProbe,
    layers: Sequence[int],
    *,
    chunk: int,
) -> dict[int, torch.Tensor]:
    """Exact ``d_model x d_model`` Jacobians of the pre-unembedding state.

    Returns ``M[l][i, j] = d n_i / d h_{l,j}`` where ``n`` is the final
    normalised state at the probe position and ``h_l`` is the layer-``l``
    residual stream *at that same position*. The final logits are
    ``W_U n + b_U``, so ``J = W_U M`` is exact and no logit-space approximation
    is involved.

    Two things are approximations and are named as such.

    *Same-position restriction.* Only the diagonal block is taken. The layer-``l``
    state at earlier positions also influences the logits at the probe position
    through attention, and that path is excluded. The question the J-lens asks
    is what *this* position's state could express about its own next symbol, so
    the restriction is deliberate, but it means the reported Jacobian is a
    sub-block of the full input-output Jacobian and its rank is a lower bound on
    the rank of that larger object.

    *Local linearisation.* A Jacobian describes the model's response to
    infinitesimal perturbations at one point on one sequence. Nothing here
    licenses a claim about what the model would emit under a finite intervention.

    The sequence is truncated at the probe position, which is exact for a causal
    decoder: no later token can influence the state at or before the cut.
    """

    if chunk < 1:
        raise ValueError("chunk must be positive")
    grid = tuple(dict.fromkeys(int(layer) for layer in layers))
    ids = probe.input_ids
    if ids.ndim != 1 or ids.numel() != probe.context_tokens:
        raise ValueError("probe token ids do not match the declared context length")
    width = arm.d_model
    batch = ids.unsqueeze(0).expand(chunk, -1).contiguous().to(arm.device)
    mask = torch.ones_like(batch)
    matrices = {layer: torch.zeros(width, width, dtype=torch.float64) for layer in grid}

    with torch.enable_grad():
        with _JacobianTap(arm, grid) as tap:
            arm.model(input_ids=batch, attention_mask=mask, use_cache=False)
        unexpected = {layer: fired for layer, fired in tap.fired.items() if fired != 1}
        if unexpected or tap.final_state is None:
            raise RuntimeError(f"{arm.name}: Jacobian hooks fired {unexpected} times")
        normalised = head.normalise(tap.final_state[:, -1, :])
        epsilons = [tap.epsilons[layer] for layer in grid]
        for start in range(0, width, chunk):
            size = min(chunk, width - start)
            cotangent = torch.zeros(chunk, width, device=arm.device, dtype=normalised.dtype)
            rows = torch.arange(size, device=arm.device)
            cotangent[rows, rows + start] = 1.0
            grads = torch.autograd.grad(
                outputs=normalised,
                inputs=epsilons,
                grad_outputs=cotangent,
                retain_graph=True,
            )
            for layer, grad in zip(grid, grads):
                block = grad[:size, -1, :].double().cpu()
                if not bool(torch.isfinite(block).all()):
                    raise FloatingPointError(
                        f"{arm.name}: non-finite Jacobian rows at layer {layer}"
                    )
                matrices[layer][start : start + size] = block
    return matrices


@torch.no_grad()
def jacobian_finite_difference_check(
    arm: Arm,
    head: LensHead,
    probe: JacobianProbe,
    matrix: torch.Tensor,
    layer: int,
    *,
    epsilon: float,
    seed: int,
) -> dict[str, Any]:
    """Validate one Jacobian against a central finite difference.

    This checks the hook placement, the row/column orientation of ``M`` and the
    numerical precision of the backward pass in one measurement, on the real
    model rather than on a surrogate. A large relative error means the reported
    spectra describe something other than the model's sensitivity.
    """

    if not epsilon > 0.0:
        raise ValueError("finite-difference epsilon must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    direction = torch.randn(arm.d_model, generator=generator, dtype=torch.float64)
    direction = direction / direction.norm()
    predicted = matrix @ direction

    ids = probe.input_ids.unsqueeze(0).to(arm.device)
    mask = torch.ones_like(ids)
    outputs: list[torch.Tensor] = []
    for sign in (1.0, -1.0):
        shift = (sign * epsilon * direction).to(device=arm.device)
        capture: dict[str, torch.Tensor] = {}

        def perturb(_module: nn.Module, _inputs: Any, output: Any) -> Any:
            tensor = block_output_tensor(output, f"{arm.name} block {layer}", arm.d_model)
            perturbed = tensor.clone()
            perturbed[:, -1, :] = perturbed[:, -1, :] + shift.to(tensor.dtype)
            if isinstance(output, tuple):
                return (perturbed, *output[1:])
            return perturbed

        def capture_final(_module: nn.Module, _inputs: Any, output: Any) -> None:
            capture["state"] = block_output_tensor(
                output, f"{arm.name} final block", arm.d_model
            )

        blocks = arm.blocks()
        handles = [
            blocks[layer].register_forward_hook(perturb),
            blocks[arm.n_layer - 1].register_forward_hook(capture_final),
        ]
        try:
            arm.model(input_ids=ids, attention_mask=mask, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        outputs.append(head.normalise(capture["state"][:, -1, :]).double().squeeze(0).cpu())

    observed = (outputs[0] - outputs[1]) / (2.0 * epsilon)
    error = float((predicted - observed).norm())
    scale = float(observed.norm())
    return {
        "layer": int(layer),
        "epsilon": float(epsilon),
        "seed": int(seed),
        "predicted_norm": float(predicted.norm()),
        "observed_norm": scale,
        "absolute_error": error,
        "relative_error": error / scale if scale > 0.0 else None,
    }


def spectrum_summary(singular_values: np.ndarray, *, floor_relative: float) -> dict[str, Any]:
    """Effective-rank statistics of a singular-value spectrum.

    Three measures are reported because they disagree by design.
    ``energy_rank_90`` and ``energy_rank_99`` are decided by the head of the
    spectrum and are numerically robust. ``entropy_effective_rank`` weights the
    whole spectrum and is the most informative, but it is sensitive to the tail
    and therefore to floating-point noise, which is why values below
    ``floor_relative`` times the largest singular value are discarded first and
    the floor is recorded alongside the number. ``stable_rank`` needs no
    threshold at all and is included as the threshold-free anchor.
    """

    if not 0.0 < floor_relative < 1.0:
        raise ValueError("floor_relative must lie strictly between zero and one")
    values = np.asarray(singular_values, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("singular values must be a non-empty vector")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("singular values must be finite and non-negative")
    values = np.sort(values)[::-1]
    largest = float(values[0])
    if largest <= 0.0:
        raise ValueError("the Jacobian spectrum is identically zero")
    kept = values[values > floor_relative * largest]
    energy = np.cumsum(kept**2) / float((kept**2).sum())
    weights = kept / kept.sum()
    return {
        "dimension": int(values.size),
        "numerical_rank": int(kept.size),
        "floor_relative": float(floor_relative),
        "largest_singular_value": largest,
        "stable_rank": float((values**2).sum() / largest**2),
        "entropy_effective_rank": float(np.exp(-(weights * np.log(weights)).sum())),
        "energy_rank_90": int(np.searchsorted(energy, 0.90) + 1),
        "energy_rank_99": int(np.searchsorted(energy, 0.99) + 1),
        "top_singular_values": [float(value) for value in values[:16]],
    }


@dataclass(frozen=True)
class ActivationSubspace:
    """Centred second-order structure of one layer's cohort activations."""

    covariance: torch.Tensor
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    mean: torch.Tensor
    n_positions: int


def activation_subspace(states: torch.Tensor, *, device: str) -> ActivationSubspace:
    """Eigenstructure of the layer's activation covariance over the cohort.

    Centred, because "the subspace the layer's activations occupy" is a statement
    about variation across positions; the mean direction is enormous in a
    transformer residual stream and would otherwise dominate every eigenvector.
    The mean is kept separately so that its own alignment can be reported rather
    than discarded.
    """

    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError("activation states must be a [positions, d_model] matrix")
    values = states.to(device=device, dtype=torch.float32)
    mean = values.mean(dim=0)
    centred = values - mean
    covariance = (centred.transpose(0, 1) @ centred).double() / (values.shape[0] - 1)
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    return ActivationSubspace(
        covariance=covariance,
        eigenvalues=eigenvalues[order].clamp_min(0.0),
        eigenvectors=eigenvectors[:, order],
        mean=mean.double(),
        n_positions=int(values.shape[0]),
    )


def activation_subspace_summary(subspace: ActivationSubspace) -> dict[str, Any]:
    """Effective dimensionality of the subspace a layer's activations occupy.

    Reported on the same footing as the Jacobian spectrum so that "how many
    directions the layer varies in" and "how many directions the output map can
    see" are read off comparable statistics. The eigenvalue spectrum of a
    covariance estimated from ``n`` positions has rank at most ``n - 1``, so the
    number of scored positions bounds these ranks and is reported with them.
    """

    eigenvalues = subspace.eigenvalues.cpu().numpy().astype(np.float64)
    total = float(eigenvalues.sum())
    if total <= 0.0:
        raise ValueError("activation covariance has no variance to allocate")
    weights = eigenvalues / total
    positive = weights[weights > 0.0]
    energy = np.cumsum(eigenvalues) / total
    return {
        "positions": subspace.n_positions,
        "rank_bound_from_positions": min(subspace.n_positions - 1, eigenvalues.size),
        "entropy_effective_rank": float(np.exp(-(positive * np.log(positive)).sum())),
        "energy_rank_90": int(np.searchsorted(energy, 0.90) + 1),
        "energy_rank_99": int(np.searchsorted(energy, 0.99) + 1),
        "total_variance": total,
        "mean_norm": float(subspace.mean.norm()),
    }


def jacobian_alignment(
    matrix: torch.Tensor,
    gram: torch.Tensor,
    subspace: ActivationSubspace,
    *,
    rank: int,
    floor_relative: float,
) -> dict[str, Any]:
    """Spectrum of ``J`` at one probe and its overlap with the activation subspace.

    ``J^T J = M^T (W~^T W~) M`` is formed exactly and eigendecomposed; its
    eigenvectors are the right singular directions of ``J``, i.e. the directions
    of the layer-``l`` residual stream that the linearised output map is most
    sensitive to.

    What the reported alignment does and does not show. ``expressed_energy_
    fraction`` is the share of the layer's activation *variance* that lies inside
    the top-``rank`` sensitive subspace, and ``blind_variance_fraction`` is the
    share that lies in the numerical null space of ``J`` -- directions the
    linearised output map cannot see at all. Both are properties of a local
    linearisation on sampled positions. A large blind fraction is consistent with
    an output-interface bottleneck, but for an arm whose vocabulary is smaller
    than its width most of that fraction is forced algebraically by
    ``rank(J) <= V - 1`` and is not an empirical discovery; the empirical content
    is *which* directions survive and how the layer's variance distributes over
    them, which is what ``gain_alignment_ratio`` reports.
    """

    if rank < 1:
        raise ValueError("alignment rank must be positive")
    width = matrix.shape[0]
    if matrix.shape != (width, width) or gram.shape != (width, width):
        raise ValueError("Jacobian and Gram matrices must be square and conformable")
    if rank > width:
        raise ValueError("alignment rank exceeds the model width")
    device = subspace.covariance.device
    m = matrix.to(device=device, dtype=torch.float64)
    inner = m.transpose(0, 1) @ gram.to(device) @ m
    inner = 0.5 * (inner + inner.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(inner)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    eigenvectors = eigenvectors[:, order]
    singular = torch.sqrt(eigenvalues)
    spectrum = spectrum_summary(singular.cpu().numpy(), floor_relative=floor_relative)

    top = eigenvectors[:, :rank]
    covariance = subspace.covariance
    total_variance = float(torch.diagonal(covariance).sum())
    if total_variance <= 0.0:
        raise ValueError("activation covariance has no variance to allocate")
    expressed = float(torch.einsum("di,de,ei->", top, covariance, top))
    sensitivity_trace = float(torch.diagonal(inner).sum())
    weighted = float((covariance * inner).sum())
    supported = eigenvectors[:, : spectrum["numerical_rank"]]
    supported_variance = float(torch.einsum("di,de,ei->", supported, covariance, supported))
    activation_top = subspace.eigenvectors[:, :rank]
    overlap = float((activation_top.transpose(0, 1) @ top).pow(2).sum()) / rank
    mean_norm = float(subspace.mean.norm())
    mean_expressed = (
        float((top.transpose(0, 1) @ subspace.mean).pow(2).sum()) / mean_norm**2
        if mean_norm > 0.0
        else None
    )
    return {
        "spectrum": spectrum,
        "alignment_rank": int(rank),
        "chance_expressed_fraction": rank / width,
        "expressed_energy_fraction": expressed / total_variance,
        "blind_variance_fraction": 1.0 - supported_variance / total_variance,
        "mean_squared_principal_cosine": overlap,
        "mean_direction_expressed_fraction": mean_expressed,
        "gain_alignment_ratio": (
            (weighted / total_variance) / (sensitivity_trace / width)
            if sensitivity_trace > 0.0
            else None
        ),
    }


_JACOBIAN_PROBE_KEYS = (
    "entropy_effective_rank",
    "stable_rank",
    "energy_rank_90",
    "energy_rank_99",
    "numerical_rank",
    "expressed_energy_fraction",
    "blind_variance_fraction",
    "mean_squared_principal_cosine",
    "mean_direction_expressed_fraction",
    "gain_alignment_ratio",
)


def jacobian_probe_row(alignment: Mapping[str, Any]) -> dict[str, float]:
    """Flatten one probe's alignment record into the bootstrapped scalars."""

    spectrum = alignment["spectrum"]
    row: dict[str, float] = {}
    for key in _JACOBIAN_PROBE_KEYS:
        value = spectrum[key] if key in spectrum else alignment[key]
        if value is None:
            raise ValueError(f"Jacobian probe produced no value for {key!r}")
        row[key] = float(value)
    return row


def jacobian_cluster_bootstrap(
    rows: Sequence[Mapping[str, float]], *, samples: int, seed: int
) -> dict[str, Any]:
    """Sequence-cluster bootstrap over Jacobian probes.

    Each probe comes from a distinct sequence, so resampling probes with
    replacement is a sequence-cluster bootstrap. With a probe count in the tens
    the interval is wide; that is the honest state of the measurement at
    validation scale, not a defect to be hidden by resampling positions instead.
    """

    if not rows:
        raise ValueError("bootstrap needs at least one probe row")
    if samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    generator = np.random.default_rng(seed)
    n = len(rows)
    draws: dict[str, list[float]] = {key: [] for key in _JACOBIAN_PROBE_KEYS}
    for _ in range(samples):
        indices = generator.integers(0, n, size=n)
        for key in draws:
            draws[key].append(float(np.mean([rows[int(i)][key] for i in indices])))
    return {
        "schema_version": SCHEMA_VERSION_BOOTSTRAP,
        "cluster_unit": "sequence",
        "samples": int(samples),
        "seed": int(seed),
        "clusters": n,
        **{key: quantile_interval(values) for key, values in draws.items()},
    }


def jacobian_formulation(head: LensHead, *, layers: Sequence[int]) -> dict[str, Any]:
    """The exact J-lens formulation, recorded with every result.

    This method is far less standardised than the logit and tuned lenses, so the
    formulation is written into the output rather than left to a paper section.
    """

    return {
        "schema_version": SCHEMA_VERSION_JACOBIAN_LENS,
        "estimand": (
            "J = d logits(position q) / d h_l(position q), the same-position block of the "
            "final-logit Jacobian with respect to the layer-l residual stream"
        ),
        "factorisation": (
            "logits = W_U n + b_U with n = ln_f(h_final), so J = W_U M with "
            "M = d n / d h_l computed exactly by d_model reverse-mode passes"
        ),
        "logit_space": (
            "W_U is centred over the vocabulary before any rank statement, because adding "
            "a constant to all logits leaves the predictive distribution unchanged"
        ),
        "spectral_route": "J^T J = M^T (W_U~^T W_U~) M, a d_model x d_model symmetric eigenproblem",
        "algebraic_rank_bound": min(head.d_model, head.vocab_size - 1),
        "d_model": head.d_model,
        "vocab_size": head.vocab_size,
        "approximations": [
            "same-position block only: paths from earlier positions' layer-l states through "
            "attention are excluded, so the reported rank lower-bounds the full Jacobian rank",
            "local linearisation at sampled positions on sampled sequences; it describes "
            "infinitesimal sensitivity, not the model's behaviour under finite intervention",
            "the activation subspace is the centred covariance over the evaluation split's "
            "scored positions, so the mean residual direction is reported separately",
            "spectra are computed in float64 from a float32 backward pass; singular values "
            "below the recorded relative floor are treated as numerical zero",
        ],
        "claims_not_supported": [
            "that a direction inside the top-J subspace but outside the activation subspace "
            "encodes any particular content",
            "that a large blind-variance fraction is empirical rather than, for arms with "
            "vocab_size - 1 < d_model, largely forced by the algebraic rank bound",
        ],
        "grid_layers": [int(layer) for layer in layers],
    }
