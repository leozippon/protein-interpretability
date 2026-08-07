"""Distributed alignment search: is there a subspace that carries the antecedent?

**The question.** On a repeat probe, a decoder predicting the token after a
repeated motif has to carry one thing forward: *what followed the antecedent*.
DAS asks whether that variable lives in a low-dimensional linear subspace of the
residual stream, by learning an orthonormal basis ``B`` and swapping only the
projection onto it between a base run and a counterfactual source run. If the
model's prediction switches to the source's continuation, the subspace mediates
the variable. The statistic is interchange-intervention accuracy (IIA).

**Why it is worth porting.** The literature gate found DAS and causal abstraction
applied to text decoders (Geiger et al., arXiv:2303.02536; Boundless DAS, Wu et
al., arXiv:2305.08809; the survey at arXiv:2410.20161) and **no application to
any protein or biological-sequence model**. The one "DAS" that appears in a
biological context is an unrelated distributed *alignment system* for sequence
alignment. So this is a transfer test of a named text method, which is direction
two of the objective, and it is not a replication.

**What this module does not do.** It does not build its own counterfactual pairs.
``path_patching.PathCase`` already is one -- base and source differing at exactly
one position, verified in ``__post_init__`` -- and ``build_path_cases`` already
samples them under a seeded permutation with a per-record cap. A second case type
would be a second opinion about what a counterfactual is (Appendix B rule 12).

**The intervention, and why only one vector is captured.** With ``b`` and ``s``
the residual streams entering block ``L`` at the read-out position under base and
source, the patched stream is

    b' = b - B Bᵀ b + B Bᵀ s = b + B Bᵀ (s - b)

so only ``Δ = s - b`` is needed; the base run supplies ``b`` in flight. That
collapse is what makes the dimension sweep cheap, and it means the captured
tensor is one ``d_model`` vector per case rather than two.

**Precision.** bfloat16 is refused. Appendix B rule 15b was earned on a metric of
order 0.05 logits against a bfloat16 quantisation step of 0.0625, and a
*gradient* taken through the same arithmetic is more exposed than the metric was.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .arms import Arm
from .path_patching import PathCase, require_supported_layout

SCHEMA_VERSION = "r2_transfer_das_v1"

#: The swept dimension ladder. Declared once, because a single dimension is a
#: threshold and Appendix B rule 17 asks for a sweep wherever one is unavoidable.
#: ``0`` is the null patch and ``-1`` stands for the full vector, the arithmetic
#: ceiling any subspace is bounded by.
SUBSPACE_DIMENSIONS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)

#: Refused outright rather than warned about: see the module docstring.
DAS_REFUSED_DTYPES = ("torch.bfloat16", "torch.float16")


@dataclass(frozen=True)
class DasConfig:
    """One cell of the sweep."""

    layer: int
    dimension: int
    steps: int = 400
    learning_rate: float = 1e-2
    batch_size: int = 16
    seed: int = 0

    def record(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "dimension": self.dimension,
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }


class Subspace(nn.Module):
    """A learned orthonormal basis, parameterised so it cannot drift off-manifold.

    DAS nominally learns a full ``d_model x d_model`` rotation and uses its first
    ``d`` rows. Only those rows ever enter the computation -- the remaining
    columns are unobservable in the objective -- so they are learned directly and
    orthonormalised by a thin QR at every use. That is the smallest complete
    parameterisation, and it makes the dimension sweep cheap rather than
    quadratic in ``d_model``.
    """

    def __init__(self, d_model: int, dimension: int, *, seed: int):
        super().__init__()
        if not 0 < dimension <= d_model:
            raise ValueError(f"dimension {dimension} outside (0, {d_model}]")
        generator = torch.Generator().manual_seed(seed)
        weight = torch.randn(d_model, dimension, generator=generator) / d_model**0.5
        self.weight = nn.Parameter(weight)

    def basis(self) -> torch.Tensor:
        basis, _ = torch.linalg.qr(self.weight, mode="reduced")
        return basis

    def project(self, delta: torch.Tensor) -> torch.Tensor:
        """``B Bᵀ Δ``, never forming the ``d_model x d_model`` projector."""

        basis = self.basis()
        return (delta @ basis) @ basis.T


# ------------------------------------------------------------------- baselines


def random_basis(d_model: int, dimension: int, *, seed: int) -> torch.Tensor:
    """A uniformly drawn orthonormal basis: does *any* subspace of this size do it?"""

    generator = torch.Generator().manual_seed(seed)
    basis, _ = torch.linalg.qr(
        torch.randn(d_model, dimension, generator=generator), mode="reduced"
    )
    return basis


def unembedding_difference_basis(
    unembedding: torch.Tensor, cases: Sequence[PathCase], dimension: int
) -> torch.Tensor:
    """The top directions of ``u[clean] - u[corrupt]`` over the case set.

    The sharpest form of the free baseline standing rule 28 asks for: computable
    from the model's weights and the case labels alone, with no activations and
    no training. A learned subspace that does not beat it has rediscovered the
    output aperture (limitation L8) rather than found a mediating variable.
    """

    rows = torch.stack(
        [
            unembedding[case.token_clean] - unembedding[case.token_corrupt]
            for case in cases
        ]
    ).float()
    _, _, right = torch.linalg.svd(rows - rows.mean(0, keepdim=True), full_matrices=False)
    return right[:dimension].T.contiguous()


def variance_matched_dimension(
    deltas: torch.Tensor, basis: torch.Tensor
) -> float:
    """Fraction of the swap's own variance this basis carries.

    The cross-arm abscissa, and it is not optional. Appendix B rule 11 measured
    interior residual participation ratio at 253.4 on gpt2-large against 4.86 on
    ProtGPT2, so a rank-8 subspace is a few percent of one arm's variation and
    most of the other's. Comparing two arms at equal ``d`` compares two different
    fractions of the thing being swapped; this is what makes them comparable.
    """

    centred = (deltas - deltas.mean(0, keepdim=True)).float()
    total = float((centred**2).sum())
    if total <= 0.0:
        return float("nan")
    carried = float(((centred @ basis) ** 2).sum())
    return carried / total


# --------------------------------------------------------------- interventions


def _residual_writer(position: int, delta: torch.Tensor):
    """A pre-hook that adds ``delta`` to the residual stream at one position.

    Grad-carrying on purpose: the whole point is that ``delta`` depends on the
    learned basis, so this cannot use the detached read that
    ``prediction_addressed._block_input`` uses. The site is the same -- the block
    pre-hook's first positional argument -- and that is not re-decided here.
    """

    def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
        hidden = args[0]
        patched = hidden.clone()
        patched[:, position] = patched[:, position] + delta
        return (patched,) + tuple(args[1:]), kwargs

    return hook


def capture_residual(
    arm: Arm, ids: torch.Tensor, *, layer: int, position: int
) -> torch.Tensor:
    """The residual stream entering ``layer`` at ``position``, detached."""

    captured: dict[str, torch.Tensor] = {}

    def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
        captured["value"] = args[0][:, position].detach().float()
        return None

    handle = arm.blocks()[layer].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            arm.model(input_ids=ids)
    finally:
        handle.remove()
    if "value" not in captured:
        raise RuntimeError(
            f"block {layer} was never entered, so no residual stream was captured; "
            "the layer index or the block accessor is wrong and every downstream "
            "number would be taken on an unpatched model"
        )
    return captured["value"]


def patched_logits(
    arm: Arm,
    ids: torch.Tensor,
    *,
    layer: int,
    position: int,
    delta: torch.Tensor | None,
) -> torch.Tensor:
    """Logits at ``position`` with ``delta`` added to the residual entering ``layer``.

    ``delta=None`` is the clean run. Only the read-out row is returned, so the
    ``[batch, tokens, vocab]` tensor is never materialised for the caller.
    """

    if delta is None:
        output = arm.model(input_ids=ids)
        return output.logits[:, position].float()
    handle = arm.blocks()[layer].register_forward_pre_hook(
        _residual_writer(position, delta), with_kwargs=True
    )
    try:
        output = arm.model(input_ids=ids)
    finally:
        handle.remove()
    return output.logits[:, position].float()


def metric(logits: torch.Tensor, clean: torch.Tensor, corrupt: torch.Tensor) -> torch.Tensor:
    """``logit[clean] - logit[corrupt]`` at the read-out, the path-patching metric.

    The same definition ``path_patching.PathPatcher._metric`` uses, so an effect
    measured here is on the scale that document's figures are on.
    """

    return logits.gather(1, clean[:, None])[:, 0] - logits.gather(1, corrupt[:, None])[:, 0]


# ---------------------------------------------------------------- evaluation


def evaluate_basis(
    arm: Arm,
    batches: Sequence[dict[str, torch.Tensor]],
    basis: torch.Tensor | None,
    *,
    layer: int,
    full_vector: bool = False,
) -> dict[str, Any]:
    """IIA and the recovery fraction for one basis, over prepared batches.

    One evaluator serves the learned subspace, every free baseline and the
    ceiling. That is what makes the comparison a comparison rather than four code
    paths that could each be wrong differently.

    ``basis=None`` with ``full_vector=True`` is the full-vector patch: the
    arithmetic maximum any subspace can reach at this site, and the attainability
    check standing rule 2 requires before any threshold is read.
    """

    hits: list[float] = []
    recovery: list[float] = []
    groups: list[int] = []
    with torch.no_grad():
        for batch in batches:
            delta = batch["delta"]
            if full_vector:
                applied = delta
            elif basis is None:
                applied = torch.zeros_like(delta)
            else:
                applied = (delta @ basis) @ basis.T
            # Each batch carries its own read-out row: cases are grouped by
            # length, and a case is truncated so that its read-out is its own
            # last token. One position for all batches would read the wrong row.
            position = int(batch["position"])
            logits = patched_logits(
                arm, batch["ids"], layer=layer, position=position, delta=applied
            )
            value = metric(logits, batch["clean"], batch["corrupt"])
            hits.extend(
                (logits.argmax(dim=-1) == batch["corrupt"]).float().cpu().tolist()
            )
            denominator = batch["metric_base"] - batch["metric_source"]
            recovery.extend(
                ((batch["metric_base"] - value) / denominator).cpu().tolist()
            )
            groups.extend(batch["group"].cpu().tolist())
    return {
        "iia": float(np.mean(hits)) if hits else float("nan"),
        "recovery_mean": float(np.mean(recovery)) if recovery else float("nan"),
        "recovery_quantiles": (
            {
                str(q): float(np.quantile(recovery, q))
                for q in (0.05, 0.25, 0.5, 0.75, 0.95)
            }
            if recovery
            else {}
        ),
        "n_cases": len(hits),
        "n_groups": len(set(groups)),
        "per_case_hit": hits,
        "per_case_group": groups,
    }


def invariants(
    arm: Arm,
    batch: dict[str, torch.Tensor],
    *,
    layer: int,
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Refuse to measure unless the intervention is doing what it claims.

    Three checks, and the third is the one that matters. A hook that fails to
    bind passes every null invariant while silently measuring an unpatched model
    -- ``path_patching`` records that exact failure -- so a null test alone is not
    evidence that the write happened. The positive control writes a large
    perturbation and requires the metric to MOVE.

    **The perturbation is a random direction, not a constant, and that is not a
    detail.** Every block here begins with a layer norm, which subtracts the
    feature mean, so a constant vector added to the residual stream is very
    nearly annihilated: measured on gpt2-large at layer 9, a uniform +10 across
    all 1280 dimensions moves the logits by **1.9e-5**, while a random direction
    of the same norm moves them by **0.62**. A constant-vector positive control
    therefore reports a correctly bound hook as unbound -- which is what it did
    here before this was understood, and is why the direction is now seeded and
    random rather than convenient.
    """

    position = int(batch["position"])
    generator = torch.Generator(device="cpu").manual_seed(20260806)
    direction = torch.randn(
        batch["delta"].shape, generator=generator, dtype=torch.float32
    ).to(batch["delta"].device)
    scale = float(batch["delta"].norm(dim=-1).mean().clamp(min=1.0))
    direction = direction / direction.norm(dim=-1, keepdim=True) * scale

    with torch.no_grad():
        clean = patched_logits(arm, batch["ids"], layer=layer, position=position, delta=None)
        zero = patched_logits(
            arm,
            batch["ids"],
            layer=layer,
            position=position,
            delta=torch.zeros_like(batch["delta"]),
        )
        shifted = patched_logits(
            arm, batch["ids"], layer=layer, position=position, delta=direction
        )
    null_gap = float((clean - zero).abs().max())
    moved = float((clean - shifted).abs().max())
    record = {
        "null_patch_max_logit_gap": null_gap,
        "perturbed_patch_max_logit_gap": moved,
        "perturbation": "seeded random direction scaled to the mean delta norm; "
        "a constant vector is annihilated by the block's layer norm and would "
        "report a bound hook as unbound",
        "perturbation_norm": scale,
        "tolerance": tolerance,
    }
    if null_gap > tolerance:
        raise RuntimeError(
            f"writing a zero delta at layer {layer} moved the logits by {null_gap:.3g}, "
            "so the intervention is not the identity when it should be and every "
            "measured effect is partly the hook itself"
        )
    if moved <= tolerance:
        raise RuntimeError(
            f"writing a large delta at layer {layer} moved the logits by {moved:.3g}, "
            "which is within the null tolerance: the hook is not bound to the "
            "residual stream. A null-only check cannot see this, which is why "
            "this positive control exists"
        )
    return record


# ------------------------------------------------------------------ training


def train_subspace(
    arm: Arm,
    batches: Sequence[dict[str, torch.Tensor]],
    config: DasConfig,
) -> tuple[Subspace, list[dict[str, Any]]]:
    """Learn the basis by matching the source run's distribution at the read-out.

    The objective is the KL from the source run's next-token distribution to the
    patched run's, **not** the IIA the result is reported on. Training on IIA's
    own argmax threshold would fit the threshold; training on a two-way logit
    difference would ask the subspace to win one comparison rather than to carry
    the whole variable. The two functions are deliberately different and the
    alternative is named here rather than tried and dropped.

    Every model parameter is frozen, and that is asserted rather than assumed: an
    optimiser handed the wrong parameter list would fine-tune a panel checkpoint
    and report the result as an interpretability finding.
    """

    arm.model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in arm.model.parameters()):
        raise RuntimeError("the model is still trainable; refusing to optimise into it")

    d_model = int(batches[0]["delta"].shape[-1])
    subspace = Subspace(d_model, config.dimension, seed=config.seed).to(
        batches[0]["delta"].device
    )
    optimiser = torch.optim.AdamW([subspace.weight], lr=config.learning_rate, weight_decay=0.0)
    generator = np.random.default_rng(config.seed)
    history: list[dict[str, Any]] = []

    for step in range(1, config.steps + 1):
        batch = batches[int(generator.integers(0, len(batches)))]
        applied = subspace.project(batch["delta"])
        logits = patched_logits(
            arm,
            batch["ids"],
            layer=config.layer,
            position=int(batch["position"]),
            delta=applied,
        )
        loss = torch.nn.functional.kl_div(
            torch.log_softmax(logits, dim=-1),
            batch["source_logprobs"],
            log_target=True,
            reduction="batchmean",
        )
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            basis = subspace.basis()
            drift = float((basis.T @ basis - torch.eye(config.dimension, device=basis.device)).abs().max())
        if drift > 1e-3:
            raise RuntimeError(
                f"the basis lost orthonormality (max |BᵀB - I| = {drift:.3g}); a "
                "rank-deficient parameter is not a subspace and the projection "
                "would silently be onto fewer dimensions than reported"
            )
        if step % max(1, config.steps // 8) == 0 or step == config.steps:
            history.append({"step": step, "kl": float(loss.detach())})
    return subspace, history


def principal_angles(left: torch.Tensor, right: torch.Tensor) -> list[float]:
    """Principal angles in degrees between two subspaces.

    Two seeds reaching the same IIA through unrelated subspaces is a different
    object from two seeds converging on one subspace, and only the second
    supports a claim that *there is* a subspace.
    """

    singular = torch.linalg.svdvals(left.T.float() @ right.float()).clamp(-1.0, 1.0)
    return [float(torch.rad2deg(torch.arccos(value))) for value in singular]


def require_das_dtype(arm: Arm) -> None:
    """Refuse half precision, for the reason Appendix B rule 15b records."""

    dtype = str(next(arm.model.parameters()).dtype)
    if dtype in DAS_REFUSED_DTYPES:
        raise RuntimeError(
            f"{arm.name} is loaded in {dtype}. The metric here is a logit "
            "difference of order 0.05 against a bfloat16 quantisation step of "
            "0.0625, and a gradient through that arithmetic is more exposed than "
            "the metric was. Load in float32."
        )
    require_supported_layout(arm)
