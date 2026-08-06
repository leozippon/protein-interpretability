#!/usr/bin/env python3
"""Is a released replacement model faithful enough to carry a causal claim?

A replacement model swaps a network's MLP-like blocks for a sparse dictionary
and is then used to make statements about the *original* network. ProGenMech
releases one for ProGen3-112M (a per-layer transcoder, ``ProGen3_PLT_L10_D4608``)
and reports its reconstruction loss. Reconstruction is not the property the
downstream claims need, and neither is behavioural agreement. This stage
measures all three separately and refuses to collapse them.

Gates, in the order they can kill the claim:

``loader``         the backbone is really loaded. ``from_pretrained`` returns a
                   ProGen3 whose every expert is random *without raising*, so
                   this is a gate and not a formality (see
                   ``src.transfer.progen3``).
``backbone``       the replacement was trained against the backbone we loaded,
                   checked tensor by tensor against the weights embedded in the
                   released checkpoint. A replacement fitted to a different
                   backbone measures that other backbone.
``behavioural``    NLL and KL of the replacement against the original, with the
                   clean and fully-ablated endpoints emitted beside every
                   ratio. Standing rule 27: a "recovery ratio" whose denominator
                   is not published is not a measurement, because the same 0.9
                   can come from a 0.02-nat gap or a 2-nat one.
``attainability``  whether the causal estimand has enough footprint to be
                   measured at all, BEFORE any threshold is applied (rule 2, the
                   L1 shape). The cross-model rank correlation is bounded above
                   by each model's own split-half reliability; if that bound
                   sits under the gate, a failing cross-model number says
                   nothing about the replacement.
``causal``         the question the other gates do not answer: ablate the same
                   components in the original and in the replacement and ask
                   whether the effects RANK the same, and whether the
                   replacement recovers the original's causal top-k above a
                   sparsity-matched random control.

The component set is every attention head and every MoE block -- the units that
exist unchanged in both models. Experts are excluded on purpose: a transcoder
replacement has none, so an expert ablation has no counterpart.

Not in this pass, deliberately: the DMS/fitness arm and the family-disjoint
split. They extend a behavioural-plus-causal core that has to be right first.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.arms import (  # noqa: E402
    DEFAULT_CORPUS_DRAW_SEED,
    REPO,
    Cohort,
    protein_cohort,
)
from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.progen3 import (  # noqa: E402
    DROPPED_KEYS,
    content_mask,
    Component,
    ablated,
    components,
    load_progen3,
    moe_intercept,
    released_state_dict,
    scored_logits,
    self_check,
    token_nll,
)
from src.transfer.statistics import bootstrap_unit_floor, mean_interval  # noqa: E402

SCHEMA_VERSION = "r2_transfer_replacement_faithfulness_v1"
DEFAULT_OUT = REPO / "results/transfer/replacement_faithfulness"

#: The released ProGenMech per-layer transcoder. Under ``external_resources``,
#: which is git-ignored and CC BY-NC-ND: this stage reads its tensors and imports
#: none of its code.
#: Overridable, because the staged copy does not sit where the B-local one does:
#: the pod's tree carries no ``baselines/`` segment, so a hard-coded default
#: resolves to a path that does not exist there and the run dies after the
#: process is already up. ``h200_env.sh`` exports the pod-side location.
DEFAULT_REPLACEMENT = Path(
    os.environ.get(
        "TRANSFER_PROGENMECH_PLT",
        REPO
        / "external_resources/baselines/ProGenMechModels/ProGen3_PLT_L10_D4608"
        / "checkpoints/last.ckpt",
    )
)

#: Component families, each scored on its own. Pooling them would let the ten
#: MoE blocks -- whose ablation effects are an order of magnitude larger than a
#: single head's -- carry the rank correlation on their own and report it as a
#: property of the seventy.
FAMILIES = ("attention_head", "moe_block")


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


def backbone_identity(embedded: dict[str, torch.Tensor], released: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Whether the replacement was fitted to the backbone this stage loaded."""

    shared = sorted(set(embedded) & set(released))
    identical = sum(int(torch.equal(embedded[key], released[key])) for key in shared)
    only_embedded = sorted(set(embedded) - set(released))
    only_released = sorted(set(released) - set(embedded))
    unexplained = sorted(set(only_released) - set(DROPPED_KEYS))
    return {
        "n_embedded": len(embedded),
        "n_released": len(released),
        "n_shared": len(shared),
        "n_bit_identical": int(identical),
        "keys_only_in_replacement_checkpoint": only_embedded,
        "keys_only_in_released_checkpoint": unexplained,
        "released_keys_dropped_by_design": sorted(set(only_released) & set(DROPPED_KEYS)),
        "verdict": (
            "PASS"
            if identical == len(shared)
            and not only_embedded
            and not unexplained
            and len(shared) > 0
            else "FAIL"
        ),
        "note": (
            "coverage is part of the gate, not only agreement: a replacement "
            "embedding a strict subset of the backbone would otherwise PASS "
            "while the weights it was actually fitted to went uncompared. The "
            "only released tensors allowed to be absent are the ones "
            "src.transfer.progen3 drops by design"
        ),
    }


# --------------------------------------------------------------- measurement


def _masked_sequence_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((values * mask).sum(1) / mask.sum(1)).double().cpu()


@torch.no_grad()
def clean_pass(
    pg: Any, transcoder: PerLayerTranscoder, sequences: list[str], *, batch_size: int
) -> dict[str, Any]:
    """One clean sweep: the per-layer mean MoE output, and the transcoder's NMSE.

    The mean is the fully-ablated endpoint every recovery ratio is divided by,
    so it is measured on this cohort rather than assumed to be zero. The NMSE is
    the number the release itself reports, recomputed here so that the artefact
    can put reconstruction, behaviour and causality side by side and show that
    they are three different things.
    """

    n_layers = pg.n_layers
    total = torch.zeros(n_layers, pg.config.hidden_size, dtype=torch.float64)
    count = torch.zeros(n_layers, dtype=torch.float64)
    nmse = torch.zeros(n_layers, dtype=torch.float64)
    batches = 0
    scored: dict[str, torch.Tensor] = {}

    def tap(layer: int, x: torch.Tensor, y: torch.Tensor) -> None:
        keep = scored["mask"]
        flat = y.reshape(-1, y.shape[-1])[keep].float()
        total[layer] += flat.sum(0).double().cpu()
        count[layer] += float(keep.sum())
        recon = transcoder(layer, x).reshape(-1, y.shape[-1])[keep].float()
        nmse[layer] += float(F.mse_loss(recon, flat) / (flat.var() + 1e-8))
        return None

    for start in range(0, len(sequences), batch_size):
        batch = pg.batch(sequences[start : start + batch_size])
        scored["mask"] = content_mask(pg, batch["input_ids"]).reshape(-1)
        with moe_intercept(pg, tap):
            pg.model(
                input_ids=batch["input_ids"],
                position_ids=batch["position_ids"],
                sequence_ids=batch["sequence_ids"],
                use_cache=False,
                return_dict=True,
            )
        batches += 1
    return {
        "moe_output_mean": (total / count[:, None]).float(),
        "reconstruction_nmse_per_layer": (nmse / batches).tolist(),
        "n_scored_tokens": count.tolist(),
    }


#: Tokens the reconstruction statistics exclude: padding, the sequence
#: delimiters and the N->C / C->N direction markers ``1`` and ``2``. They are
#: named rather than filtered by presence, because a name this tokenizer does not
#: carry means the vocabulary moved, and silently scoring the padding would move
#: both the fully-ablated endpoint and the NMSE without moving anything visible.
def replacement_context(pg: Any, transcoder: PerLayerTranscoder) -> Callable[[], Any]:
    def factory() -> Any:
        return moe_intercept(pg, lambda layer, x, y: transcoder(layer, x))

    return factory


def mean_ablation_context(pg: Any, means: torch.Tensor) -> Callable[[], Any]:
    resident = means.to(pg.device)

    def factory() -> Any:
        return moe_intercept(
            pg,
            lambda layer, x, y: resident[layer].to(y.dtype).expand_as(y),
        )

    return factory


@torch.no_grad()
def behavioural_scores(
    pg: Any,
    sequences: list[str],
    conditions: dict[str, Callable[[], Any] | None],
    *,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-sequence NLL for every condition, and KL against the original.

    ``conditions`` must begin with ``original`` mapped to ``None``: the KL is
    ``KL(original || condition)`` over the full vocabulary at every scored
    target, so the original's distribution has to exist before the others run.
    """

    if next(iter(conditions)) != "original":
        raise ValueError("the first condition must be 'original'; the KL is taken against it")
    collected: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {"nll": [], "kl": []} for name in conditions
    }
    for start in range(0, len(sequences), batch_size):
        batch = pg.batch(sequences[start : start + batch_size])
        reference: torch.Tensor | None = None
        for name, factory in conditions.items():
            with (factory() if factory is not None else nullcontext()):
                logits, targets, mask = scored_logits(pg, batch)
            collected[name]["nll"].append(
                _masked_sequence_mean(token_nll(logits, targets), mask)
            )
            log_probabilities = torch.log_softmax(logits, dim=-1)
            if reference is None:
                reference = log_probabilities
                divergence = torch.zeros_like(targets, dtype=torch.float32)
            else:
                divergence = (
                    reference.exp() * (reference - log_probabilities)
                ).sum(-1)
            collected[name]["kl"].append(_masked_sequence_mean(divergence, mask))
    return {
        name: {key: torch.cat(values).numpy() for key, values in payload.items()}
        for name, payload in collected.items()
    }


@torch.no_grad()
def component_effects(
    pg: Any,
    sequences: list[str],
    grid: list[Component],
    *,
    batch_size: int,
    wrapper: Callable[[], Any] | None = None,
) -> np.ndarray:
    """Per-sequence NLL increase under each component's ablation.

    ``(n_components, n_sequences)``, so the resampling unit downstream is the
    cohort sequence rather than the component -- the population that was drawn.
    """

    def sweep(component: Component | None) -> np.ndarray:
        rows: list[torch.Tensor] = []
        for start in range(0, len(sequences), batch_size):
            batch = pg.batch(sequences[start : start + batch_size])
            outer = wrapper() if wrapper is not None else nullcontext()
            with outer:
                inner = ablated(pg, component) if component is not None else nullcontext()
                with inner:
                    logits, targets, mask = scored_logits(pg, batch)
            rows.append(_masked_sequence_mean(token_nll(logits, targets), mask))
        return torch.cat(rows).numpy()

    baseline = sweep(None)
    return np.stack([sweep(component) - baseline for component in grid])


# ----------------------------------------------------------------- statistics


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 3:
        return None
    value = stats.spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else None


def _spearman_brown(value: float | None) -> float | None:
    """Split-half reliability corrected to the full cohort's length."""

    if value is None:
        return None
    return float(2.0 * value / (1.0 + value)) if value > -1.0 else None


def attainability(
    original: np.ndarray, replacement: np.ndarray, *, seed: int, gate_rho: float
) -> dict[str, Any]:
    """Can this estimand be measured at all, before any threshold is applied?

    Two independent obstacles, both reported as numbers rather than folded into
    a verdict. **Reliability**: the cross-model rank correlation is attenuated by
    the noise in each model's own effect vector, and the split-half reliability
    of that vector bounds it -- so a ceiling below the gate means a failing
    cross-model rho is a fact about the cohort, not about the replacement.
    **Resolution**: how many components have an effect this cohort can separate
    from zero at all; a grid of indistinguishable components has a rank order
    that is noise whatever its reliability says.
    """

    n_sequences = original.shape[1]
    order = np.random.default_rng(seed).permutation(n_sequences)
    left, right = order[: n_sequences // 2], order[n_sequences // 2 :]
    halves = {}
    for name, matrix in (("original", original), ("replacement", replacement)):
        raw = _spearman(matrix[:, left].mean(axis=1), matrix[:, right].mean(axis=1))
        halves[name] = {"split_half": raw, "spearman_brown": _spearman_brown(raw)}
    corrected = [
        halves[name]["spearman_brown"] for name in ("original", "replacement")
    ]
    ceiling = (
        float(np.sqrt(max(corrected[0], 0.0) * max(corrected[1], 0.0)))
        if all(value is not None for value in corrected)
        else None
    )
    resolved = {}
    for name, matrix in (("original", original), ("replacement", replacement)):
        intervals = [mean_interval(row.tolist()) for row in matrix]
        resolved[name] = int(
            sum(1 for record in intervals if record["interval"][0] > 0.0)
        )
    return {
        "resampling_unit": "cohort sequence",
        **bootstrap_unit_floor(n_sequences),
        "split_half_reliability": halves,
        "cross_model_rho_ceiling": ceiling,
        "gate_rho": float(gate_rho),
        "n_components": int(original.shape[0]),
        "n_components_resolved_above_zero": resolved,
        "verdict": (
            "ATTAINABLE"
            if ceiling is not None and ceiling >= gate_rho and resolved["original"] > 0
            else "UNATTAINABLE"
        ),
        "note": (
            "a cross-model rank correlation cannot exceed the geometric mean of "
            "the two models' own split-half reliabilities; below the gate, a "
            "failing cross-model value is uninformative about the replacement"
        ),
    }


def causal_agreement(
    original: np.ndarray,
    replacement: np.ndarray,
    *,
    seed: int,
    replicates: int,
    top_k: int,
    alpha: float,
) -> dict[str, Any]:
    """Do the two models rank the same components, and recover the same top-k?

    The top-k control is read from the **exact** null rather than from a
    resampled quantile. Overlap between a fixed top-k set and a uniform draw of
    k from n is hypergeometric, and the empirical q95 of a discrete distribution
    is a cliff: on the retained draws an overlap of 4 (p = 0.052) and one of 5
    (p = 0.0078) sit either side of a q95 of exactly 4.0, so the flag moved
    between draws for a reason that is arithmetic rather than evidential
    (Appendix B rule 17). The cliff also ran at two undeclared levels, because
    ``k`` is clamped to ``n // 3`` and the 10-component MoE family therefore
    tested k=3 of 10, where only a perfect 3/3 could ever clear q95 -- the same
    named gate at p=0.0083 for one family and p=0.05 for the other.
    """

    n_components, n_sequences = original.shape
    rho = _spearman(original.mean(axis=1), replacement.mean(axis=1))
    generator = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        picked = generator.integers(0, n_sequences, size=n_sequences)
        value = _spearman(
            original[:, picked].mean(axis=1), replacement[:, picked].mean(axis=1)
        )
        if value is not None:
            draws.append(value)
    interval = (
        [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
        if len(draws) >= 0.95 * replicates
        else None
    )

    k = int(min(top_k, max(1, n_components // 3)))
    top_original = set(np.argsort(-original.mean(axis=1))[:k].tolist())
    top_replacement = set(np.argsort(-replacement.mean(axis=1))[:k].tolist())
    overlap = len(top_original & top_replacement)
    control = np.array(
        [
            len(top_original & set(generator.choice(n_components, size=k, replace=False).tolist()))
            for _ in range(replicates)
        ],
        dtype=np.float64,
    )
    null = stats.hypergeom(n_components, k, k)
    p_value = float(null.sf(overlap - 1))
    attainable = [j for j in range(k + 1) if null.sf(j - 1) <= alpha]
    smallest_significant = attainable[0] if attainable else k + 1
    return {
        "spearman": rho,
        "spearman_interval": interval,
        "bootstrap_replicates": int(replicates),
        "bootstrap_draws_used": len(draws),
        "top_k": k,
        "top_k_requested": int(top_k),
        "top_k_clamped": bool(k != top_k),
        "top_k_overlap": int(overlap),
        "top_k_overlap_fraction": float(overlap / k),
        "sparsity_matched_random_control": {
            "description": (
                f"{replicates} uniform draws of {k} of the {n_components} "
                "components, matched to the observed top-k size; reported as "
                "description, while the verdict reads the exact null below"
            ),
            "mean_overlap": float(control.mean()),
            "q95_overlap": float(np.quantile(control, 0.95)),
            "exact_expected_overlap": float(k * k / n_components),
        },
        "exact_null": {
            "distribution": (
                f"hypergeometric: overlap of a fixed top-{k} set with a uniform "
                f"draw of {k} from {n_components}"
            ),
            "p_value_one_sided": p_value,
            "alpha": float(alpha),
            "smallest_significant_overlap": int(smallest_significant),
            "note": (
                "with a discrete null the attainable significance is coarse; "
                "smallest_significant_overlap is the least overlap this family "
                "could ever record as exceeding the control, and it differs "
                "between families of different size"
            ),
        },
        "exceeds_random_control": bool(p_value <= alpha),
    }


# --------------------------------------------------------------------- driver


def _paired_recovery(
    clean: np.ndarray,
    replacement: np.ndarray,
    ablated: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """The recovery ratio with an interval, resampling sequences jointly.

    The three conditions are measured on the *same* cohort sequences, so the
    marginal Student-t intervals beside them support only the weaker unpaired
    reading. One index set is drawn for all three per replicate, which is what
    makes this an interval on the ratio rather than on three means that happen
    to be reported together.
    """

    rng = np.random.default_rng(seed)
    n = clean.size
    gaps = np.empty(replicates, dtype=np.float64)
    ratios = np.full(replicates, np.nan, dtype=np.float64)
    for index in range(replicates):
        pick = rng.integers(0, n, size=n)
        c, r, a = clean[pick].mean(), replacement[pick].mean(), ablated[pick].mean()
        gaps[index] = r - c
        if a - c > 0:
            ratios[index] = (a - r) / (a - c)
    finite = ratios[np.isfinite(ratios)]
    return {
        "replacement_minus_clean": mean_interval((replacement - clean).tolist()),
        "recovery_interval": (
            [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]
            if finite.size >= 0.95 * replicates
            else None
        ),
        "recovery_replicates_used": int(finite.size),
        "bootstrap_replicates": int(replicates),
        "resampling_unit": "cohort sequence, one index set shared by all three conditions",
    }


def build_cohort(args: argparse.Namespace) -> Cohort:
    return protein_cohort(
        args.sequences,
        args.protein_min_len,
        args.protein_max_len,
        skip=args.cohort_skip,
        name="progen3_replacement",
        seed=args.cohort_draw_seed or None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--replacement", type=Path, default=DEFAULT_REPLACEMENT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--sequences", type=int, default=128)
    parser.add_argument("--protein-min-len", type=int, default=64)
    parser.add_argument("--protein-max-len", type=int, default=246)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--control-alpha",
        type=float,
        default=0.05,
        help="significance level the top-k overlap is read against, under the "
        "exact hypergeometric null. Declared here rather than implied by an "
        "empirical q95, whose attainable level differs between families of "
        "different size",
    )
    parser.add_argument(
        "--gate-rho",
        type=float,
        default=0.5,
        help="rank-correlation a faithful replacement must reach, and the level "
        "the attainability ceiling is compared against. A declared convention, "
        "recorded in the artefact; the correlation and its interval are the "
        "result, the verdict is a reading of them",
    )
    parser.add_argument(
        "--gate-recovery",
        type=float,
        default=0.8,
        help="fraction of the clean-to-fully-ablated NLL gap the replacement must "
        "close. Declared convention, same standing as --gate-rho",
    )
    parser.add_argument(
        "--cohort-draw-seed",
        type=int,
        default=DEFAULT_CORPUS_DRAW_SEED,
        help="seed for the permutation this stage's cohort is drawn under; "
        "0 selects the historical file-order prefix, which is a declared choice "
        "and not a default (transfer audit, Appendix B rule 1)",
    )
    parser.add_argument("--cohort-skip", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
    }

    print("[cohort] drawing")
    cohort = build_cohort(args)
    sequences = list(cohort.records)

    print("[loader] loading ProGen3-112M and self-checking the conversion")
    load_kwargs: dict[str, Any] = {
        "device": args.device,
        "dtype": getattr(torch, args.dtype),
    }
    if args.checkpoint is not None:
        load_kwargs["checkpoint"] = args.checkpoint
    pg = load_progen3(**load_kwargs)
    loader_gate = self_check(pg)
    print(f"  self-check NLL {loader_gate['nll']:.4f} in {loader_gate['band']}")

    print("[backbone] reading the released replacement and its embedded backbone")
    transcoder, embedded, hyperparameters = load_replacement(args.replacement)
    transcoder.to(pg.device)
    released = released_state_dict(pg.checkpoint)
    # The replacement is spliced in by positional layer index. A checkpoint
    # covering a different depth or width -- or a layer subset -- would be
    # applied to the wrong blocks without raising, and the run would emit a
    # complete artefact for a misaligned replacement. This is the same failure
    # class src.transfer.progen3 exists to make impossible, and the consumer has
    # to carry its own half of it.
    if int(hyperparameters.num_layers) != pg.n_layers:
        raise RuntimeError(
            f"the replacement covers {hyperparameters.num_layers} layers and this "
            f"ProGen3 has {pg.n_layers}; splicing by positional index would "
            "measure the wrong blocks"
        )
    if int(hyperparameters.d_model) != int(pg.config.hidden_size):
        raise RuntimeError(
            f"the replacement was fitted at d_model {hyperparameters.d_model} and "
            f"this ProGen3 is {pg.config.hidden_size} wide"
        )
    backbone_gate = backbone_identity(embedded, released)
    del embedded, released
    gc.collect()
    print(
        f"  {backbone_gate['n_bit_identical']}/{backbone_gate['n_shared']} tensors "
        f"bit-identical  {backbone_gate['verdict']}"
    )

    condition = {
        "replacement": "ProGenMech ProGen3_PLT_L10_D4608 (per-layer transcoder)",
        "replacement_sha256": sha256_file(args.replacement),
        "replacement_hyperparameters": {
            "num_layers": int(hyperparameters.num_layers),
            "d_model": int(hyperparameters.d_model),
            "d_hidden": int(hyperparameters.d_hidden),
            "k": int(hyperparameters.k),
        },
        "backbone_sha256": sha256_file(pg.checkpoint / "model.safetensors"),
        "backbone_loading": "eager MoE, converted from the released megablocks "
        "packing by src.transfer.progen3; from_pretrained's own eager path "
        "returns random experts without raising",
        "layers_replaced": "all MoE blocks",
        "scoring_direction": "N->C only (the reverse direction doubles every "
        "sweep and this stage runs one per component per model)",
        "reconstruction_measured_under": "clean inputs, teacher-forced -- the "
        "convention the release's own val/loss was measured under, and NOT the "
        "sequential replacement that behaviour and causality are measured under. "
        "A reconstruction figure from this artefact is comparable to theirs; a "
        "behavioural one is not",
        "fully_ablated_endpoint": "every MoE block output replaced by its "
        "per-layer mean over this cohort's residue positions",
        "component_families": list(FAMILIES),
        "ablation": "zero the component's contribution to the residual stream",
        "resampling_unit": "cohort sequence",
    }

    print("[behavioural] clean sweep, reconstruction NMSE and the ablation endpoint")
    reference = clean_pass(pg, transcoder, sequences, batch_size=args.batch_size)
    conditions: dict[str, Callable[[], Any] | None] = {
        "original": None,
        "replacement": replacement_context(pg, transcoder),
        "mean_ablated": mean_ablation_context(pg, reference["moe_output_mean"]),
    }
    scores = behavioural_scores(pg, sequences, conditions, batch_size=args.batch_size)
    clean_nll = float(scores["original"]["nll"].mean())
    replacement_nll = float(scores["replacement"]["nll"].mean())
    ablated_nll = float(scores["mean_ablated"]["nll"].mean())
    replacement_kl = float(scores["replacement"]["kl"].mean())
    ablated_kl = float(scores["mean_ablated"]["kl"].mean())
    nll_denominator = ablated_nll - clean_nll
    behavioural_gate = {
        "nll_nats_per_token": {
            "clean": clean_nll,
            "replacement": replacement_nll,
            "fully_ablated": ablated_nll,
            "replacement_minus_clean": replacement_nll - clean_nll,
            "denominator": nll_denominator,
            "denominator_definition": "fully_ablated - clean",
            "recovery": (
                (ablated_nll - replacement_nll) / nll_denominator
                if nll_denominator > 0
                else None
            ),
        },
        "kl_nats_per_token": {
            "replacement": replacement_kl,
            "fully_ablated": ablated_kl,
            "denominator": ablated_kl,
            "denominator_definition": "KL(original || fully_ablated)",
            "recovery": (1.0 - replacement_kl / ablated_kl) if ablated_kl > 0 else None,
        },
        "per_sequence_nll_interval": {
            name: mean_interval(scores[name]["nll"].tolist()) for name in scores
        },
        "paired_per_sequence": _paired_recovery(
            scores["original"]["nll"],
            scores["replacement"]["nll"],
            scores["mean_ablated"]["nll"],
            replicates=args.bootstrap,
            seed=args.seed,
        ),
        "reconstruction_nmse_per_layer": reference["reconstruction_nmse_per_layer"],
        "reconstruction_nmse_sum": float(
            sum(reference["reconstruction_nmse_per_layer"])
        ),
        "reconstruction_scored_tokens_per_layer": reference["n_scored_tokens"],
        "gate_recovery": float(args.gate_recovery),
    }
    recovery = behavioural_gate["nll_nats_per_token"]["recovery"]
    behavioural_gate["verdict"] = (
        "PASS" if recovery is not None and recovery >= args.gate_recovery else "FAIL"
    )
    print(
        f"  NLL clean {clean_nll:.4f} -> replacement {replacement_nll:.4f} -> "
        f"fully ablated {ablated_nll:.4f}  (denominator {nll_denominator:.4f}, "
        f"recovery {recovery if recovery is None else round(recovery, 4)})"
    )
    print(f"  KL(original||replacement) {replacement_kl:.4f} vs ablated {ablated_kl:.4f}")

    print("[causal] ablating every component in both models")
    grid = components(pg)
    effects = {
        "original": component_effects(
            pg, sequences, grid, batch_size=args.batch_size
        ),
        "replacement": component_effects(
            pg,
            sequences,
            grid,
            batch_size=args.batch_size,
            wrapper=replacement_context(pg, transcoder),
        ),
    }
    labels = [component.label for component in grid]
    np.savez_compressed(
        args.out / "component_effects.npz",
        labels=np.asarray(labels),
        kinds=np.asarray([component.kind for component in grid]),
        original=effects["original"],
        replacement=effects["replacement"],
    )

    attainability_gate: dict[str, Any] = {}
    causal_gate: dict[str, Any] = {}
    for family in FAMILIES:
        rows = [index for index, component in enumerate(grid) if component.kind == family]
        left, right = effects["original"][rows], effects["replacement"][rows]
        attainability_gate[family] = attainability(
            left, right, seed=args.seed + 1, gate_rho=args.gate_rho
        )
        agreement = causal_agreement(
            left,
            right,
            seed=args.seed + 2,
            replicates=args.bootstrap,
            top_k=args.top_k,
            alpha=args.control_alpha,
        )
        if attainability_gate[family]["verdict"] != "ATTAINABLE":
            agreement["verdict"] = "WITHHELD_UNATTAINABLE"
            agreement["withheld_reason"] = (
                "the attainability ceiling for this family is below --gate-rho, so "
                "the cross-model correlation cannot distinguish an unfaithful "
                "replacement from a cohort that cannot resolve the ranking"
            )
        else:
            interval = agreement["spearman_interval"]
            agreement["verdict"] = (
                "PASS"
                if interval is not None
                and interval[0] >= args.gate_rho
                and agreement["exceeds_random_control"]
                else "FAIL"
            )
        causal_gate[family] = agreement
        print(
            f"  {family:15s} attainability {attainability_gate[family]['verdict']:13s} "
            f"ceiling {attainability_gate[family]['cross_model_rho_ceiling']}  "
            f"rho {agreement['spearman']}  top-{agreement['top_k']} overlap "
            f"{agreement['top_k_overlap']} exact p "
            f"{agreement['exact_null']['p_value_one_sided']:.4g}  "
            f"{agreement['verdict']}"
        )

    gates = {
        "loader": loader_gate,
        "backbone": backbone_gate,
        "behavioural": behavioural_gate,
        "attainability": attainability_gate,
        "causal": causal_gate,
    }
    # The loader is a precondition, not a result: `check_nll` raises rather than
    # returning FAIL, so including it here made its PASS unconditional, made
    # `any(...)` unconditionally true, and made the FAIL branch below
    # unreachable -- a run in which every substantive gate failed still reported
    # PARTIAL. The roll-up therefore reads the gates that can actually fail.
    verdicts = [
        gates["backbone"]["verdict"],
        gates["behavioural"]["verdict"],
        *[record["verdict"] for record in attainability_gate.values()],
        *[record["verdict"] for record in causal_gate.values()],
    ]
    payload.update(
        {
            "condition": condition,
            "cohort": {
                "name": cohort.name,
                "digest": cohort.digest,
                "provenance_digest": cohort.provenance_digest,
                "sampling": cohort.sampling,
                "n_sequences": len(cohort),
                "residue_band": [args.protein_min_len, args.protein_max_len],
            },
            "model": {
                "checkpoint": str(pg.checkpoint),
                "n_layers": pg.n_layers,
                "n_heads": pg.n_heads,
                "n_components": len(grid),
                "dtype": args.dtype,
            },
            "gates": gates,
            "verdict": (
                "PASS"
                if all(value in ("PASS", "ATTAINABLE") for value in verdicts)
                else "PARTIAL"
                if any(value in ("PASS", "ATTAINABLE") for value in verdicts)
                else "FAIL"
            ),
        }
    )
    write_json(args.out / "replacement_faithfulness.json", payload)
    print(f"wrote {args.out / 'replacement_faithfulness.json'}  verdict {payload['verdict']}")


if __name__ == "__main__":
    main()
