#!/usr/bin/env python3
"""Is a size-k circuit faithful in the model's OWN neuron basis, on protein?

**What this discriminates.** F11 records that under sequential per-layer
transcoder replacement two text decoders recover 0.9084-0.9322 of their
clean-to-mean-ablated cross-entropy gap while three protein decoders recover
0.0916-0.1641, with no overlap, and that the controls built so far exclude method
invalidity, depth, training budget, sparse-MoE routing, tokenisation, cohort band
and dictionary saturation without identifying the proximal cause. Two hypotheses
remain live and this stage separates them in one training-free measurement:

* **the learned dictionary is at fault** -- then a circuit built on the model's
  raw MLP neurons, which no one trained, is about as faithful on protein as on
  text, and the deficit is a property of the METHOD;
* **protein MLP computation is genuinely dense** -- then neurons fail on protein
  as badly as dictionaries do, no sparse basis will recover it, and the deficit
  is a property of the MODEL and a first-principles difference between the
  families.

**The estimand is stage 15's, deliberately.** The fraction of the
clean-to-mean-ablated cross-entropy gap recovered when only the selected units
are kept and the complement is replaced by its per-unit mean over the cohort's
content positions. Only the *basis* changes: stage 15 keeps a whole block's
transcoder reconstruction, this keeps k raw neurons per block. That the two share
a denominator is not asserted, it is measured -- the down-projection is affine,
so mean-ablating every neuron and mean-ablating the block output are the same
intervention, and the ``endpoints`` gate reports both floors and their
difference. Without that the two stages' ratios could not be placed side by side.

**The tensor is the MLP's hidden layer and nothing else.** A neuron is a
coordinate of the ``d_mlp``-wide post-nonlinearity tensor the down-projection
consumes, not of the ``d_model``-wide output it produces. The output is a dense
mixture of every neuron; a circuit measured there would understate what a sparse
basis can do on any arm and would manufacture a "proteins need dictionaries"
conclusion out of our own transfer failure. The tensor is resolved from
:data:`src.transfer.arms._MLP_NEURON_TENSOR`, an explicit per-architecture
declaration that refuses an undeclared architecture rather than duck-typing it,
and three facts about the tensor actually hooked are checked against the loaded
config before anything is measured: its width is the declared ``d_mlp``, that
width differs from ``d_model``, and its minimum respects the declared
nonlinearity's lower bound -- which a pre-activation tensor, unbounded below,
cannot.

**Arms.** The architecture-matched non-gated triple ``gpt2-large``, ``protgpt2``
and ``zymctrl``: all GPT-2, all 36 x 1280, all non-gated GELU, so the gating
confound is held fixed across the modality contrast and cannot be part of the
answer. ``progen3`` is reachable through ``--arm`` because the eligible set is
composed rather than written down, and is refused with its reason: a sparse-MoE
block has no single pre-down-projection tensor for this declaration to name.

**Selection, and why the result is a curve.** Neurons are ranked by gradient x
activation on the arm's own scored-token cross-entropy, which is standard and
costs one backward pass. The ranking is a *heuristic*: it is a first-order
approximation to zero-ablating a neuron, while the intervention actually applied
is mean ablation, so ``--score mean_ablation_attribution`` offers the
approximation matched to the estimand and the artefact records which was used.
What makes any ranking readable is the size-matched random control beside it,
which the transcoder literature this stage answers does not have anywhere.

The reported result is a **faithfulness curve over circuit size**, not a point.
A single-point ratio is the defect this repository's own limitation catalogue
names, and the 2026-08-10 literature review put "a faithfulness curve
marginalised over circuit size" on the short list of things worth adopting.
``k`` is a **per-layer** budget: the score's scale varies with depth, so a global
top-k would let whichever layers carry large gradients buy the whole budget and
would report that allocation as a property of the arm. A per-layer budget also
makes the two endpoints exact -- ``k = 0`` is the mean-ablated floor and
``k = d_mlp`` is the clean model -- so the curve is anchored at both ends by
construction and both are measured rather than assumed.

**Everything the reader needs to divide by is in the artefact.** Standing rule
27: a recovery ratio whose denominator is not published is not a measurement.
Both endpoints are reported in absolute nats per scored token, every curve point
carries its damage in nats beside its ratio, and the arm's measured symbols per
token is recorded so a per-symbol figure -- the scale on which EXP-R2-156 found
protein absolute damage to be 8-17x the text arms' -- is derivable.
"""

from __future__ import annotations

import argparse
import gc
import sys
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# The stage directory itself, so `panel_contract` imports under every invocation
# rather than only when the caller happens to run from scripts/transfer.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from panel_contract import CAMPAIGN_PANEL  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    DEFAULT_CORPUS_DRAW_SEED,
    PANEL,
    REPO,
    Arm,
    Cohort,
    mlp_neuron_declaration,
    protein_cohort,
    symbols_per_token,
    text_cohort,
)
from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.progen3 import token_nll  # noqa: E402
from src.transfer.replaceable import (  # noqa: E402
    PROGEN3_ARM,
    DenseReplaceable,
    ReplaceableModel,
    arm_evaluation_cohort_source,
    eligible_arms,
    load_replaceable,
)
from src.transfer.statistics import bootstrap_unit_floor, mean_interval  # noqa: E402

SCHEMA_VERSION = "r2_transfer_neuron_basis_circuit_v1"
DEFAULT_OUT = REPO / "results/transfer/neuron_basis_circuit"

#: Modules whose content decides these numbers, hashed into the artefact. The
#: arms module is first because it carries the neuron-tensor declaration, which
#: is the one decision that can silently change what is being measured.
PROVENANCE_MODULES = (
    "src/transfer/arms.py",
    "src/transfer/replaceable.py",
    "src/transfer/scoring.py",
    "src/transfer/statistics.py",
)

#: Circuit sizes swept by default, as neurons kept **per layer**. Doubling from
#: 32 to 2048 covers 0.6% to 40% of the 5120 neurons every arm of the matched
#: triple has, which is the range over which a sparse basis either does or does
#: not exist; the floor (0) and the whole layer (d_mlp) are appended by the stage
#: because they anchor the curve rather than sample it.
DEFAULT_CIRCUIT_SIZES = (32, 64, 128, 256, 512, 1024, 2048)

#: How far below the declared nonlinearity's lower bound a measured minimum may
#: fall before the hooked tensor is refused. It exists only to absorb bfloat16
#: rounding of a bound stated to two decimals; a pre-activation tensor sits many
#: units below it and a wrongly hooked one is caught by the width check first.
NEURON_MINIMUM_SLACK = 0.01

#: How far the k = d_mlp point may sit from the clean cross-entropy, in nats per
#: token. Keeping every neuron substitutes each element by itself, so the
#: intervention is an identity and the difference should be zero; anything
#: measurable means the hook path is not a no-op and the whole curve is shifted.
IDENTITY_TOLERANCE_NATS = 1e-6

#: How far the neuron-basis floor may sit from the block-output floor stage 15
#: divides by, **as a fraction of the denominator the two of them define**. The
#: two are the same intervention because the down-projection is affine, so the
#: difference is accumulated float error rather than a modelling choice: the two
#: means are rounded to the model's dtype on opposite sides of the projection --
#: a d_mlp-wide vector here, a d_model-wide one there -- and the difference is
#: carried through every remaining layer.
#:
#: Relative rather than absolute, and sized from a measurement. On ``gpt2`` at
#: bfloat16 the gap is 0.0251 nats against a 3.4941-nat denominator, which is
#: 0.72%; an absolute tolerance at that scale would be tighter on a text arm than
#: on a protein one purely because the protein denominators are larger, which is
#: the normalisation error EXP-R2-156 was about. 2% leaves nearly three times the
#: observed rounding, and the failure it has to catch is nothing like that size:
#: a mean taken over the wrong positions, or an intervention applied at the wrong
#: ones, moves a floor by an appreciable fraction of the denominator itself.
COMMENSURABILITY_TOLERANCE_FRACTION = 0.02


# --------------------------------------------------------------- the neuron tap


def declared_architecture(arm: str) -> str:
    """The architecture whose neuron declaration this arm resolves to.

    Answered from the panel alone so that an arm this stage cannot measure is
    refused in a second, before a checkpoint is read. ``progen3`` is named
    explicitly because it is not a panel member and its refusal is a statement
    about its block rather than about its absence from a table.
    """

    if arm == PROGEN3_ARM:
        raise ValueError(
            "progen3's block is a sparse mixture of experts: there is no single "
            "pre-down-projection activation for a neuron basis to name, and the "
            "expert-wise ones are not the same object across tokens. This stage "
            "measures the architecture-matched non-gated triple, where the "
            "gating and routing confounds are held fixed"
        )
    if arm not in PANEL:
        raise ValueError(f"{arm!r} is not a panel arm; panel is {sorted(PANEL)}")
    return PANEL[arm].architecture


def dense_arm(model: ReplaceableModel) -> Arm:
    """The loaded :class:`~src.transfer.arms.Arm` behind a dense replaceable model.

    An explicit type check rather than an attribute probe: the neuron tap needs
    the panel's module declarations, and a model that does not carry an ``Arm``
    has no declaration to be resolved against.
    """

    if not isinstance(model, DenseReplaceable):
        raise TypeError(
            f"{type(model).__name__} carries no panel Arm, so its MLP hidden "
            "activation cannot be resolved from a declaration"
        )
    return model.arm


@contextmanager
def neuron_intercept(
    model: ReplaceableModel,
    fn: Callable[[int, torch.Tensor], torch.Tensor | None],
) -> Iterator[None]:
    """Read or replace every layer's MLP hidden activation while the model runs.

    ``fn(layer, activation)`` returns a tensor to substitute for it, or ``None``
    to leave it alone -- the one primitive the mean, the attribution and the
    circuit interventions all need, and the neuron-basis counterpart of
    :meth:`src.transfer.replaceable.DenseReplaceable.block_intercept`.

    Registered as a **pre-hook on the down-projection**, so the tensor handed to
    ``fn`` is by construction the one the projection consumes. Nothing here can
    drift to the MLP's output without the declaration changing.
    """

    arm = dense_arm(model)
    handles = []
    for layer in range(model.n_layers):

        def hook(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            layer: int = layer,
        ) -> tuple[Any, ...] | None:
            if not inputs:
                raise TypeError(
                    f"{arm.name}: layer {layer}'s down-projection was called with no "
                    "positional input, so the hidden activation cannot be read"
                )
            replaced = fn(layer, inputs[0])
            if replaced is None:
                return None
            return (replaced,) + tuple(inputs[1:])

        handles.append(arm.mlp_down_projection(layer).register_forward_pre_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


# --------------------------------------------------------------- measurement


def _masked_sequence_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sequence mean over the scored positions, refusing an empty row.

    A row with no scored target contributes ``nan`` to every mean built on it and
    to nothing else, so it would travel to the artefact as a non-finite value and
    be rejected only at serialisation, after the run.
    """

    counts = mask.sum(1)
    if bool((counts == 0).any()):
        raise RuntimeError(
            "a cohort sequence has no scored target; its cross-entropy is undefined "
            "and would enter the curve as a non-finite value"
        )
    return ((values * mask).sum(1) / counts).double().cpu()


@torch.no_grad()
def neuron_reference(
    model: ReplaceableModel,
    inputs: list[str],
    *,
    batch_size: int,
    declared_width: int,
) -> dict[str, Any]:
    """One clean sweep: the per-neuron mean, the block-output mean, and the checks.

    Both means are taken over the cohort's **content** positions -- padding,
    delimiters and any conditioning prompt excluded -- which is the convention
    stage 15's fully-ablated endpoint is defined under, so the two floors are
    endpoints of the same estimand. The substitution that uses them is applied at
    every position, again as stage 15 does.

    The block-output mean is collected in the same pass purely so the two floors
    can be compared later: the down-projection is affine, so mean-ablating every
    neuron and mean-ablating the block output are one intervention, and a
    measured difference is what licenses placing this stage's ratios beside
    stage 15's.
    """

    n_layers, width = model.n_layers, model.width
    neuron_total = torch.zeros(n_layers, declared_width, dtype=torch.float64)
    block_total = torch.zeros(n_layers, width, dtype=torch.float64)
    counted = torch.zeros(n_layers, dtype=torch.float64)
    observed_width: set[int] = set()
    minimum = float("inf")
    scored: dict[str, torch.Tensor] = {}

    def read_neurons(layer: int, activation: torch.Tensor) -> None:
        keep = scored["mask"]
        observed_width.add(int(activation.shape[-1]))
        flat = activation.reshape(-1, activation.shape[-1]).float()[keep]
        neuron_total[layer] += flat.sum(0).double().cpu()
        counted[layer] += float(keep.sum())
        nonlocal minimum
        minimum = min(minimum, float(activation.min()))
        return None

    def read_block(layer: int, x: torch.Tensor, y: torch.Tensor) -> None:
        keep = scored["mask"]
        block_total[layer] += y.reshape(-1, y.shape[-1]).float()[keep].sum(0).double().cpu()
        return None

    for start in range(0, len(inputs), batch_size):
        batch = model.batch(inputs[start : start + batch_size])
        scored["mask"] = model.content_mask(batch).reshape(-1)
        with neuron_intercept(model, read_neurons), model.block_intercept(read_block):
            model.run(batch)

    if not counted.gt(0).all():
        raise RuntimeError("the cohort supplied no content positions to average over")
    if observed_width != {declared_width}:
        raise RuntimeError(
            f"the hooked tensor is {sorted(observed_width)} wide and the declaration "
            f"says {declared_width}. The MLP's output is {width} wide: a tensor of "
            "that width means the hook moved past the down-projection, and a "
            "neuron-basis result read there would understate every arm's sparsity"
        )
    return {
        "neuron_mean": (neuron_total / counted[:, None]).float(),
        "block_output_mean": (block_total / counted[:, None]).float(),
        "n_content_positions_per_layer": counted.tolist(),
        "measured_width": int(next(iter(observed_width))),
        "measured_minimum": minimum,
    }


def verify_neuron_tensor(facts: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Refuse a tensor that is not the post-nonlinearity hidden activation.

    Two independent properties pin the tensor, and they are checked where each
    can first be known. The **width** separates the hidden layer from the MLP
    output -- the regression that would flatter every arm's sparsity -- and is
    checked in :func:`neuron_reference`, which cannot accumulate into an array of
    the declared width without it. The **minimum** separates it from the
    pre-activation the first projection produces, which is unbounded below where
    a GELU output is not, and is checked here because it is only knowable once
    the sweep is finished. Neither alone would be enough.
    """

    bound = float(facts["activation_lower_bound"])
    minimum = float(reference["measured_minimum"])
    if minimum < bound - NEURON_MINIMUM_SLACK:
        raise RuntimeError(
            f"the hooked tensor's minimum is {minimum:.4f}, below the "
            f"{facts['activation']} lower bound {bound} less {NEURON_MINIMUM_SLACK}. "
            "A post-nonlinearity activation cannot go there; a pre-activation one "
            "can, so the hook is reading the wrong side of the nonlinearity"
        )
    record = {
        **facts,
        "measured_width": int(reference["measured_width"]),
        "measured_minimum": minimum,
        "minimum_slack": NEURON_MINIMUM_SLACK,
        "width_differs_from_d_model": bool(facts["declared_width"] != facts["d_model"]),
        "verdict": "PASS",
    }
    if not record["width_differs_from_d_model"]:
        # Not a failure: an architecture could in principle set d_mlp = d_model.
        # It is recorded because on this panel they differ by 4x, and a run where
        # they coincide is one where the width check has stopped separating the
        # hidden tensor from the output.
        record["note"] = (
            "d_mlp equals d_model on this checkpoint, so the width check cannot "
            "separate the hidden tensor from the MLP output; only the minimum does"
        )
    return record


@torch.no_grad()
def scored_cross_entropy(
    model: ReplaceableModel,
    inputs: list[str],
    *,
    batch_size: int,
    factory: Callable[[], Any] | None = None,
) -> np.ndarray:
    """Per-sequence cross-entropy in nats per scored token, under one condition."""

    rows: list[torch.Tensor] = []
    for start in range(0, len(inputs), batch_size):
        batch = model.batch(inputs[start : start + batch_size])
        with (factory() if factory is not None else nullcontext()):
            logits, targets, mask = model.scored_logits(batch)
        rows.append(_masked_sequence_mean(token_nll(logits, targets), mask))
    return torch.cat(rows).numpy()


def attribution_scores(
    model: ReplaceableModel,
    inputs: list[str],
    *,
    neuron_mean: torch.Tensor,
    batch_size: int,
    score: str,
) -> np.ndarray:
    """Rank neurons by attribution on the arm's own scored-token cross-entropy.

    The objective is the **total** negative log-likelihood over the cohort's
    scored targets, not a per-batch mean, so the accumulated score does not
    depend on how the cohort was cut into batches.

    Gradients are taken with :func:`torch.autograd.grad` against the captured
    activations, which computes only the edges that reach them: the parameter
    gradients -- three gigabytes on a 774M arm at float32 -- are never
    materialised, and no ``.grad`` on the backbone is touched. The activations
    are captured as they are and **not** detached: re-entering each layer's
    activation as a leaf would cut the path from every earlier layer to the loss
    and score all but the last layer at zero.

    ``gradient_x_activation`` is ``|sum_t a . dL/da|``, the standard cheap
    ranking and the one this stage declares. It approximates *zeroing* a neuron,
    while the intervention the estimand applies is *mean* ablation, so a neuron
    that is large and nearly constant scores highly here and has no causal effect
    there. ``mean_ablation_attribution`` is the same first-order expansion taken
    about the mean, ``|sum_t (a - a_mean) . dL/da|``, which is the ranking matched
    to the intervention. Both are computed from the same two accumulators; the
    choice is recorded in the artefact, and neither is trusted -- the
    size-matched random control is what says whether the ranking bought anything.
    """

    arm = dense_arm(model)
    n_layers, width = neuron_mean.shape
    total_ag = torch.zeros(n_layers, width, dtype=torch.float64)
    total_g = torch.zeros(n_layers, width, dtype=torch.float64)
    for start in range(0, len(inputs), batch_size):
        batch = model.batch(inputs[start : start + batch_size])
        # Valid rather than content positions: the substitution this ranking
        # serves is applied at every non-padding position, including a
        # conditioning prompt, whose neurons reach the scored residues through
        # attention. Padding contributes zero gradient under right padding and a
        # causal mask, and is excluded anyway.
        valid = batch["attention_mask"].bool().unsqueeze(-1)
        captured: dict[int, torch.Tensor] = {}

        def keep(layer: int, activation: torch.Tensor) -> None:
            captured[layer] = activation
            return None

        with torch.enable_grad(), neuron_intercept(model, keep):
            logits, targets, mask = model.scored_logits(batch)
            loss = (token_nll(logits, targets) * mask).sum()
        if len(captured) != n_layers:
            raise RuntimeError(
                f"{arm.name}: {len(captured)} of {n_layers} layers were intercepted; "
                "an unhooked layer would be scored as if it had no neurons"
            )
        ordered = [captured[layer] for layer in range(n_layers)]
        if not all(activation.requires_grad for activation in ordered):
            raise RuntimeError(
                f"{arm.name}: the hidden activations carry no gradient. The backbone's "
                "parameters must require grad for this pass, or every neuron would "
                "score zero and the ranking would be an arbitrary order"
            )
        grads = torch.autograd.grad(loss, ordered)
        for layer, (activation, grad) in enumerate(zip(ordered, grads)):
            masked = grad.float() * valid
            total_ag[layer] += (activation.detach().float() * masked).sum((0, 1)).double().cpu()
            total_g[layer] += masked.sum((0, 1)).double().cpu()
        del captured, ordered, grads, loss, logits
    gc.collect()

    if score == "gradient_x_activation":
        return total_ag.abs().numpy()
    if score == "mean_ablation_attribution":
        return (total_ag - neuron_mean.double() * total_g).abs().numpy()
    raise ValueError(f"unknown attribution score {score!r}")


# ------------------------------------------------------------------- circuits


def _sized(keep: torch.Tensor, k: int, what: str) -> torch.Tensor:
    """Refuse a circuit that is not the size it claims.

    The random control is only a control if it is size-matched to the circuit it
    controls, and a mask built by scattering or permuting is exactly the kind of
    thing that is off by one without saying so.
    """

    counts = keep.sum(1).tolist()
    if any(int(count) != int(k) for count in counts):
        raise RuntimeError(
            f"{what} keeps {sorted(set(int(c) for c in counts))} neurons per layer "
            f"where {k} was requested; a control that is not size-matched is not a "
            "control"
        )
    return keep


def top_k_mask(scores: np.ndarray, k: int) -> torch.Tensor:
    """The k highest-scoring neurons of every layer."""

    values = torch.from_numpy(np.ascontiguousarray(scores))
    keep = torch.zeros(values.shape, dtype=torch.bool)
    if k > 0:
        keep.scatter_(1, values.topk(int(k), dim=1).indices, True)
    return _sized(keep, k, "the selected circuit")


def random_mask(shape: tuple[int, int], k: int, *, seed: int) -> torch.Tensor:
    """A uniform circuit of the same size, drawn independently in every layer."""

    generator = torch.Generator().manual_seed(int(seed))
    keep = torch.zeros(shape, dtype=torch.bool)
    for layer in range(shape[0]):
        keep[layer, torch.randperm(shape[1], generator=generator)[: int(k)]] = True
    return _sized(keep, k, f"the random control at seed {seed}")


def circuit_context(
    model: ReplaceableModel, keep: torch.Tensor, neuron_mean: torch.Tensor
) -> Callable[[], Any]:
    """Keep the selected neurons; replace every other one by its cohort mean."""

    resident_keep = keep.to(model.device)
    resident_mean = neuron_mean.to(model.device)

    def factory() -> Any:
        return neuron_intercept(
            model,
            lambda layer, activation: torch.where(
                resident_keep[layer], activation, resident_mean[layer].to(activation.dtype)
            ),
        )

    return factory


def block_mean_context(
    model: ReplaceableModel, block_mean: torch.Tensor
) -> Callable[[], Any]:
    """Stage 15's fully-ablated endpoint, for the commensurability check."""

    resident = block_mean.to(model.device)

    def factory() -> Any:
        return model.block_intercept(
            lambda layer, x, y: resident[layer].to(y.dtype).expand_as(y)
        )

    return factory


# ----------------------------------------------------------------- statistics


def bootstrap_indices(n: int, *, replicates: int, seed: int) -> np.ndarray:
    """One set of resampled sequence indices per replicate, shared by every point.

    Drawn once and reused across the whole curve and both controls, so that two
    points of one curve differ by their circuit rather than by their resample.
    """

    return np.random.default_rng(seed).integers(0, n, size=(replicates, n))


def recovery_record(
    clean: np.ndarray,
    circuit: np.ndarray,
    ablated: np.ndarray,
    *,
    picks: np.ndarray,
) -> dict[str, Any]:
    """One point of the curve: the ratio, the nats, and an interval on both.

    Resampling is paired -- one index set per replicate scores all three
    conditions -- because they are measured on the same cohort sequences and an
    unpaired interval on a ratio of means would be wider than the estimand
    supports. This is the computation ``15_replacement_faithfulness.py``'s
    ``_paired_recovery`` performs, with the index sets lifted out so that every
    circuit size sees the same ones.
    """

    denominator = float(ablated.mean() - clean.mean())
    damage = float(circuit.mean() - clean.mean())
    ratios = np.full(len(picks), np.nan, dtype=np.float64)
    for index, pick in enumerate(picks):
        c, r, a = clean[pick].mean(), circuit[pick].mean(), ablated[pick].mean()
        if a - c > 0:
            ratios[index] = (a - r) / (a - c)
    finite = ratios[np.isfinite(ratios)]
    return {
        "cross_entropy_nats_per_token": float(circuit.mean()),
        "damage_nats_per_token": damage,
        "damage_interval": mean_interval((circuit - clean).tolist()),
        "denominator_nats_per_token": denominator,
        "recovery": (denominator - damage) / denominator if denominator > 0 else None,
        "recovery_interval": (
            [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]
            if finite.size >= 0.95 * len(picks)
            else None
        ),
        "recovery_replicates_used": int(finite.size),
    }


def endpoints_record(
    clean: np.ndarray,
    ablated: np.ndarray,
    full_circuit: np.ndarray,
    block_floor: np.ndarray,
) -> dict[str, Any]:
    """Both ends of the ratio in absolute nats, and the two checks on them.

    Standing rule 27: a recovery ratio whose denominator is not published is not
    a measurement, because the same 0.9 can come from a 0.02-nat gap or a 2-nat
    one. Both endpoints are therefore reported as nats per scored token with
    their intervals, and the ratio is derived from them rather than the other way
    round.

    Two properties are checked here rather than assumed. The **identity**: at
    ``k = d_mlp`` every neuron is substituted by itself, so that point must
    reproduce the clean cross-entropy exactly; anything measurable means the
    intervention path is not a no-op and the whole curve is shifted. The
    **commensurability**: mean-ablating every neuron and mean-ablating the block
    output are one intervention because the down-projection is affine, so the two
    floors must agree, and if they do not this stage's denominator is not stage
    15's and the two sets of ratios cannot be placed side by side.
    """

    denominator = float(ablated.mean() - clean.mean())
    identity_gap = float(abs(full_circuit.mean() - clean.mean()))
    commensurability_gap = float(abs(ablated.mean() - block_floor.mean()))
    relative_gap = commensurability_gap / denominator if denominator > 0 else None
    return {
        "clean_nats_per_token": float(clean.mean()),
        "clean_interval": mean_interval(clean.tolist()),
        "mean_ablated_nats_per_token": float(ablated.mean()),
        "mean_ablated_interval": mean_interval(ablated.tolist()),
        "denominator_nats_per_token": denominator,
        "denominator_definition": "mean_ablated - clean, over the same cohort sequences",
        "full_circuit_minus_clean_nats": identity_gap,
        "identity_tolerance_nats": IDENTITY_TOLERANCE_NATS,
        "block_output_floor_nats_per_token": float(block_floor.mean()),
        "neuron_floor_minus_block_floor_nats": commensurability_gap,
        "neuron_floor_minus_block_floor_fraction_of_denominator": relative_gap,
        "commensurability_tolerance_fraction": COMMENSURABILITY_TOLERANCE_FRACTION,
        "commensurability_note": (
            "the down-projection is affine, so mean-ablating every neuron and "
            "mean-ablating the block output are the same intervention, and the "
            "residual difference is the two means being rounded to the model's "
            "dtype on opposite sides of that projection. A gap beyond the "
            "tolerance means this stage's denominator is not stage 15's and the "
            "two sets of ratios must not be compared"
        ),
        "verdict": (
            "PASS"
            if denominator > 0
            and identity_gap <= IDENTITY_TOLERANCE_NATS
            and relative_gap is not None
            and relative_gap <= COMMENSURABILITY_TOLERANCE_FRACTION
            else "FAIL"
        ),
    }


# --------------------------------------------------------------------- driver


def circuit_sizes(requested: list[int], *, d_mlp: int) -> list[int]:
    """The swept grid, anchored at the floor and at the whole layer.

    ``0`` is the mean-ablated floor whose recovery is zero by construction and
    ``d_mlp`` is the clean model whose recovery is one by construction. Both are
    measured rather than assumed: the first is the denominator every ratio is
    divided by, and the second is the only check that the intervention path is a
    no-op when it should be.
    """

    for value in requested:
        if value < 0:
            raise ValueError("a circuit size is a count of neurons and cannot be negative")
        if value > d_mlp:
            raise ValueError(
                f"a circuit of {value} neurons per layer exceeds the {d_mlp} this arm has"
            )
    return sorted({0, *(int(value) for value in requested), int(d_mlp)})


def build_cohort(args: argparse.Namespace) -> Cohort:
    """The cohort this arm is scored on, drawn from the corpus the panel declares.

    One dispatch on the arm's declared evaluation corpus, so that the population
    is the one ``15_replacement_faithfulness.py`` scores the same arm on and the
    two stages' ratios describe the same sequences.
    """

    source = arm_evaluation_cohort_source(args.arm)
    label = f"{args.arm}_neuron_basis"
    if source == "openwebtext":
        return text_cohort(
            args.sequences,
            args.text_min_chars,
            skip=args.cohort_skip,
            name=label,
            seed=args.cohort_draw_seed or None,
        )
    if source in ("swissprot", "zymctrl_ec"):
        return protein_cohort(
            args.sequences,
            args.protein_min_len,
            args.protein_max_len,
            skip=args.cohort_skip,
            name=label,
            with_ec=source == "zymctrl_ec",
            seed=args.cohort_draw_seed or None,
        )
    raise ValueError(
        f"{args.arm} draws its cohort from {source!r}, which this stage cannot build"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        default="gpt2-large",
        choices=eligible_arms(CAMPAIGN_PANEL),
        help="which decoder to measure. The eligible set is composed by "
        "src.transfer.replaceable.eligible_arms and is shared with stages 15 and "
        "17 so the three describe the same population of arms; the ones this "
        "stage cannot resolve a neuron basis for are refused with their reason",
    )
    parser.add_argument(
        "--score",
        default="gradient_x_activation",
        choices=("gradient_x_activation", "mean_ablation_attribution"),
        help="the ranking neurons are selected by. The default is the standard "
        "one and approximates zero-ablation; the alternative is the first-order "
        "approximation of the mean ablation this estimand actually applies",
    )
    parser.add_argument(
        "--circuit-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_CIRCUIT_SIZES),
        help="neurons kept PER LAYER. The floor (0) and the whole layer are "
        "always added, because they anchor the curve",
    )
    parser.add_argument(
        "--control-seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="seeds for the size-matched random circuits, which are what says "
        "whether the ranking bought anything",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--sequences", type=int, default=128)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="token cap the arm's inputs are truncated to",
    )
    parser.add_argument(
        "--text-min-chars",
        type=int,
        default=800,
        help="floor of the text cohort a text arm is scored on, in characters; "
        "src.transfer.arms.text_cohort's own default",
    )
    parser.add_argument(
        "--protein-min-len",
        type=int,
        default=64,
        help="lower edge of the residue band. The default is the shared 64-246 "
        "band the replacement comparison is anchored on (F11)",
    )
    parser.add_argument("--protein-max-len", type=int, default=246)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--attribution-batch-size",
        type=int,
        default=2,
        help="batch size of the one pass that keeps a graph. Smaller than "
        "--batch-size because every layer's hidden activation is retained",
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--cohort-draw-seed",
        type=int,
        default=DEFAULT_CORPUS_DRAW_SEED,
        help="seed for the permutation this stage's cohort is drawn under; "
        "0 selects the historical file-order prefix, which is a declared choice "
        "and not a default (transfer audit, Appendix B rule 1)",
    )
    parser.add_argument("--cohort-skip", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Before anything is loaded or drawn: an arm whose block this declaration
    # does not name cannot be measured here, and saying so costs a lookup.
    architecture = declared_architecture(args.arm)
    mlp_neuron_declaration(architecture)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "provenance": {
            "runner": {
                "path": "scripts/transfer/22_neuron_basis_circuit.py",
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "modules": {
                name: sha256_file(REPO_ROOT / name) for name in PROVENANCE_MODULES
            },
        },
        "estimand": (
            "fraction of the clean-to-mean-ablated cross-entropy gap recovered by a "
            "size-k circuit in the MLP hidden (pre-down-projection) basis: the k "
            "highest-attribution neurons of every layer are kept and every other "
            "neuron is replaced by its mean over the cohort's content positions. "
            "The same estimand 15_replacement_faithfulness.py measures, in a "
            "different basis, so the two are directly comparable"
        ),
    }

    print("[cohort] drawing")
    cohort = build_cohort(args)

    print(f"[loader] loading {args.arm} and running its self-check")
    model = load_replaceable(
        args.arm,
        campaign_panel=CAMPAIGN_PANEL,
        device=args.device,
        dtype=args.dtype,
        max_tokens=args.max_tokens,
    )
    loader_gate = model.self_check()
    print(f"  self-check NLL {loader_gate['nll']:.4f} in {loader_gate['band']}")
    arm = dense_arm(model)
    scored_inputs = model.render(
        cohort.records, ec_labels=cohort.metadata.get("ec_labels")
    )
    facts = arm.mlp_neuron_facts()
    d_mlp = int(facts["declared_width"])
    print(f"  neuron tensor: {facts['tensor']}, width {d_mlp} ({facts['width_source']})")

    print("[reference] per-neuron mean, block-output mean, and the tensor checks")
    reference = neuron_reference(
        model, scored_inputs, batch_size=args.batch_size, declared_width=d_mlp
    )
    neuron_gate = verify_neuron_tensor(facts, reference)
    print(
        f"  measured width {neuron_gate['measured_width']} (d_model "
        f"{neuron_gate['d_model']}), minimum {neuron_gate['measured_minimum']:.4f} "
        f"against the {neuron_gate['activation']} bound "
        f"{neuron_gate['activation_lower_bound']}  {neuron_gate['verdict']}"
    )

    print("[clean] scoring the unmodified model")
    clean = scored_cross_entropy(model, scored_inputs, batch_size=args.batch_size)

    print(f"[attribution] ranking neurons by {args.score}")
    scores = attribution_scores(
        model,
        scored_inputs,
        neuron_mean=reference["neuron_mean"],
        batch_size=args.attribution_batch_size,
        score=args.score,
    )

    sizes = circuit_sizes(list(args.circuit_sizes), d_mlp=d_mlp)
    picks = bootstrap_indices(len(clean), replicates=args.bootstrap, seed=args.seed)

    print(f"[sweep] {len(sizes)} circuit sizes, {sizes}")
    measured: dict[int, np.ndarray] = {}
    for k in sizes:
        measured[k] = scored_cross_entropy(
            model,
            scored_inputs,
            batch_size=args.batch_size,
            factory=circuit_context(
                model, top_k_mask(scores, k), reference["neuron_mean"]
            ),
        )
        print(f"  k={k:5d}  CE {measured[k].mean():.4f}")
    ablated = measured[0]
    curve = [
        {
            "k_per_layer": int(k),
            "fraction_of_d_mlp": float(k / d_mlp),
            "n_neurons_kept": int(k) * model.n_layers,
            **recovery_record(clean, measured[k], ablated, picks=picks),
        }
        for k in sizes
    ]
    selected_recovery = {point["k_per_layer"]: point["recovery"] for point in curve}

    print("[control] size-matched random circuits")
    control: list[dict[str, Any]] = []
    for k in sizes:
        if k in (0, d_mlp):
            # At both anchors every circuit of that size is the same circuit, so
            # a "random" one would be the selected one under another name.
            continue
        seeds = []
        for seed in args.control_seeds:
            drawn = scored_cross_entropy(
                model,
                scored_inputs,
                batch_size=args.batch_size,
                factory=circuit_context(
                    model,
                    random_mask((model.n_layers, d_mlp), k, seed=seed),
                    reference["neuron_mean"],
                ),
            )
            seeds.append(
                {"seed": int(seed), **recovery_record(clean, drawn, ablated, picks=picks)}
            )
        recoveries = [record["recovery"] for record in seeds if record["recovery"] is not None]
        selected = selected_recovery[k]
        control.append(
            {
                "k_per_layer": int(k),
                "seeds": seeds,
                "recovery_mean": float(np.mean(recoveries)) if recoveries else None,
                "recovery_range": (
                    [float(min(recoveries)), float(max(recoveries))] if recoveries else None
                ),
                "damage_nats_per_token_mean": float(
                    np.mean([record["damage_nats_per_token"] for record in seeds])
                ),
                "selected_minus_random_recovery": (
                    float(selected - float(np.mean(recoveries)))
                    if recoveries and selected is not None
                    else None
                ),
            }
        )
        print(
            f"  k={k:5d}  random recovery "
            f"{control[-1]['recovery_mean'] if control[-1]['recovery_mean'] is None else round(control[-1]['recovery_mean'], 4)}"
            f"  selected {selected if selected is None else round(selected, 4)}"
        )

    print("[endpoints] the block-output floor stage 15 divides by")
    block_floor = scored_cross_entropy(
        model,
        scored_inputs,
        batch_size=args.batch_size,
        factory=block_mean_context(model, reference["block_output_mean"]),
    )
    endpoints = endpoints_record(clean, ablated, measured[d_mlp], block_floor)
    print(
        f"  clean {endpoints['clean_nats_per_token']:.4f} -> mean-ablated "
        f"{endpoints['mean_ablated_nats_per_token']:.4f} (denominator "
        f"{endpoints['denominator_nats_per_token']:.4f}); identity gap "
        f"{endpoints['full_circuit_minus_clean_nats']:.3e}, block-floor gap "
        f"{endpoints['neuron_floor_minus_block_floor_nats']:.3e}  {endpoints['verdict']}"
    )

    expansion = symbols_per_token(arm, scored_inputs, args.max_tokens)

    payload.update(
        {
            "cohort": {
                "name": cohort.name,
                "kind": cohort.kind,
                "digest": cohort.digest,
                "provenance_digest": cohort.provenance_digest,
                "sampling": cohort.sampling,
                "n_sequences": len(cohort),
                "residue_band": [args.protein_min_len, args.protein_max_len],
                "text_min_chars": args.text_min_chars,
                "band_note": (
                    "the residue band applies to a protein cohort and the character "
                    "floor to a text one; which is in force is decided by kind above"
                ),
            },
            "model": {
                "arm": args.arm,
                "checkpoint": str(model.checkpoint),
                "weights_sha256": model.weights_digest(),
                "n_layers": model.n_layers,
                "n_heads": model.n_heads,
                "d_model": model.width,
                "d_mlp": d_mlp,
                "n_neurons": d_mlp * model.n_layers,
                "dtype": args.dtype,
                "device": str(model.device),
                "loading_note": model.loading_note,
                "scoring_note": model.scoring_note,
            },
            "symbols_per_token": {
                "value": float(expansion),
                "basis": (
                    "src.transfer.arms.symbols_per_token over the rendered inputs "
                    "truncated to --max-tokens: residues for a protein arm, "
                    "characters for a text one"
                ),
                "note": (
                    "divide any nats-per-token figure above by this to read it per "
                    "content symbol; the two units are not interchangeable across "
                    "arms (Appendix B rule 26, L23)"
                ),
            },
            "selection": {
                "score": args.score,
                "granularity": "k neurons per layer, ranked within the layer",
                "granularity_reason": (
                    "attribution scale varies with depth, so a global top-k would "
                    "report an allocation across layers as a property of the arm"
                ),
                "attribution_positions": "every non-padding position",
                "mean_positions": "content positions only, as stage 15's endpoint is",
                "limitation": (
                    "gradient x activation is a first-order approximation to zeroing "
                    "a neuron while the applied intervention is mean ablation; the "
                    "size-matched random control below is what bounds what the "
                    "ranking is worth"
                ),
            },
            "gates": {
                "loader": loader_gate,
                "neuron_tensor": neuron_gate,
                "endpoints": endpoints,
            },
            "resampling": {
                "unit": "cohort sequence, one index set shared by every circuit size",
                "bootstrap_replicates": int(args.bootstrap),
                **bootstrap_unit_floor(len(clean)),
            },
            "curve": curve,
            "random_control": control,
            "verdict": (
                "PASS"
                if all(
                    record["verdict"] == "PASS"
                    for record in (loader_gate, neuron_gate, endpoints)
                )
                else "FAIL"
            ),
        }
    )
    destination = args.out / "neuron_basis_circuit.json"
    write_json(destination, payload)
    print()
    for point in curve:
        recovery = point["recovery"]
        print(
            f"  k={point['k_per_layer']:5d} ({point['fraction_of_d_mlp']:6.2%})  "
            f"damage {point['damage_nats_per_token']:+.4f} nats  recovery "
            f"{recovery if recovery is None else round(recovery, 4)}"
        )
    print(f"wrote {destination}  verdict {payload['verdict']}")


if __name__ == "__main__":
    main()
