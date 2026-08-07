"""Where a sparse-MoE replacement's error lives, read against the router.

**The question, and why it is the right one.** A transcoder replaces a MoE block
with a continuous function of that block's input. The block itself is not
continuous: which two of eight experts run is a ``topk`` of a linear map, so the
true function is piecewise and its pieces meet at routing boundaries. ProGenMech
say plainly that their cross-layer transcoder abstracts the router away. This
module measures the consequence they did not: **is the replacement's residual
structured by the routing decision, or is it diffuse?**

**The statistic is a correction, not a correlation.** Correlating a per-token
error against a router statistic answers a weaker question and inherits every
confound that a rank correlation against a depth-varying readout has already
cost this programme (EXP-R2-120, F4). Instead, tokens are grouped into cells and
a per-cell mean residual is fitted on one set of sequences and *subtracted on
another*. How much held-out error a grouping removes is how much that grouping
knew. The comparison is then between groupings, at matched cardinality, which is
a statement about information rather than about scale.

**Three groupings, each with a declared role.**

``routing``      the token's selected expert set. The hypothesis.
``random``       a seeded assignment to the *same number* of cells. The free
                 baseline of standing rule 28 -- it costs the same degrees of
                 freedom and knows nothing, so anything ``routing`` earns above
                 it is information rather than capacity.
``residue``      the token's amino acid. Not a cardinality control but the
                 scientific confound: if this model's routing is largely a
                 lookup on the current residue, then a routing cell and a
                 residue cell are the same cell wearing two names, and a routing
                 result would say nothing about routing. Measured directly, as
                 the accuracy of predicting the expert set from the residue
                 alone, so the reader is not left to infer it.

**Attainability comes first (standing rule 2).** A grouping can only remove error
if it has cells to remove it into. If one expert set holds nearly every token at
a layer, that layer's routing carries no usable variation and it is reported
*unmeasurable* rather than as a failure of the hypothesis. The in-sample
reduction is reported beside every held-out one for the same reason: it is the
arithmetic ceiling that grouping could reach at that cardinality, and a held-out
number is only interpretable against it.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

#: A layer whose most common expert set holds at least this share of tokens has
#: no routing variation to spend, and is reported unmeasurable. Declared here
#: because the gate reads it and the record prints it.
DEGENERATE_CELL_SHARE = 0.95

#: Cells must hold at least this many training tokens for their mean to be worth
#: subtracting. Smaller cells are not dropped -- dropping them would change the
#: token population between groupings and break the pairing -- they are given a
#: zero correction, which is what "this cell taught us nothing" means.
MINIMUM_CELL_TOKENS = 32


def expert_sets(n_experts: int, top_k: int) -> list[tuple[int, ...]]:
    """Every selectable expert set, in a fixed order, as the cell vocabulary.

    Unordered, because ``topk`` returns the two experts by gate weight and a
    token routed to experts ``{3, 5}`` computes the same mixture whichever came
    first up to the weights -- which the correction does not see anyway.
    """

    if not 1 <= top_k <= n_experts:
        raise ValueError(f"top_k {top_k} is not in 1..{n_experts}")
    return list(combinations(range(n_experts), top_k))


def routing_cells(router_probs: np.ndarray, *, top_k: int) -> np.ndarray:
    """Each token's expert-set cell index, from the post-softmax router output.

    ``router_probs`` is ``(tokens, n_experts)`` and must be the **float32**
    distribution, which is what the block selects on --
    :func:`src.transfer.progen3.router_probabilities` produces it. The block also
    *returns* a distribution, and that one is in the hidden states' dtype: EXP-R2-130
    measured a float32-against-bfloat16 router softmax flipping the selected pair
    on 3.3% of tokens, so reading the returned tensor would mislabel that many
    cells. The selection is reproduced here because the block exposes the
    distribution and not the choice.
    """

    if router_probs.ndim != 2:
        raise ValueError(f"router_probs must be (tokens, experts), got {router_probs.shape}")
    n_experts = router_probs.shape[1]
    vocabulary = {combo: index for index, combo in enumerate(expert_sets(n_experts, top_k))}
    chosen = np.argpartition(-router_probs, kth=top_k - 1, axis=1)[:, :top_k]
    chosen.sort(axis=1)
    return np.array([vocabulary[tuple(row)] for row in chosen], dtype=np.int64)


def router_dispersion(router_probs: np.ndarray, *, top_k: int) -> dict[str, Any]:
    """How much of a decision the router is making, before anything is asked of it.

    ``margin`` is the gap between the largest gate weight and the first one that
    was *not* selected -- the distance to the nearest boundary in gate space,
    which is the quantity a piecewise-function account is about. With
    ``top_k = 2`` of eight that is ``p[2] - p[3]`` in sorted order, not
    ``p[1] - p[2]``, and the difference matters: the second is the gap *inside*
    the selected pair and crossing it changes no expert at all.
    """

    ordered = -np.sort(-router_probs, axis=1)
    margin = ordered[:, top_k - 1] - ordered[:, top_k]
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(router_probs > 0, np.log(np.clip(router_probs, 1e-12, None)), 0.0)
    entropy = -(router_probs * logs).sum(axis=1)
    return {
        "margin_mean": float(margin.mean()),
        "margin_quantiles": [float(q) for q in np.quantile(margin, [0.05, 0.25, 0.5, 0.75, 0.95])],
        "entropy_mean_nats": float(entropy.mean()),
        "entropy_max_nats": float(np.log(router_probs.shape[1])),
        "n_tokens": int(router_probs.shape[0]),
    }


def cell_occupancy(cells: np.ndarray, n_cells: int) -> dict[str, Any]:
    """How the tokens are spread over the cells, and whether that is usable."""

    counts = np.bincount(cells, minlength=n_cells)
    share = counts / max(int(counts.sum()), 1)
    occupied = int((counts > 0).sum())
    largest = float(share.max()) if counts.sum() else 1.0
    return {
        "n_cells": int(n_cells),
        "n_occupied": occupied,
        "largest_cell_share": largest,
        "cells_above_minimum": int((counts >= MINIMUM_CELL_TOKENS).sum()),
        "degenerate": bool(largest >= DEGENERATE_CELL_SHARE),
        "degenerate_reason": (
            f"one cell holds {largest:.3f} of tokens, at or above the "
            f"{DEGENERATE_CELL_SHARE} share above which a per-cell mean has no "
            "variation to fit"
            if largest >= DEGENERATE_CELL_SHARE
            else None
        ),
    }


@dataclass(frozen=True)
class Correction:
    """A per-cell mean residual, fitted on training tokens."""

    means: np.ndarray  # (n_cells, d_model)
    fitted_cells: int

    def apply(self, residual: np.ndarray, cells: np.ndarray) -> np.ndarray:
        return residual - self.means[cells]


def fit_correction(residual: np.ndarray, cells: np.ndarray, n_cells: int) -> Correction:
    """The mean residual inside each cell, zero where the cell is too small.

    A cell below :data:`MINIMUM_CELL_TOKENS` gets a zero correction rather than
    its own noisy mean. Zero is the honest estimate for a cell that has not been
    measured, and it keeps every grouping scored on the identical token set --
    without which the paired comparison between groupings is not paired.
    """

    if residual.shape[0] != cells.shape[0]:
        raise ValueError("residual and cells must agree on the token axis")
    means = np.zeros((n_cells, residual.shape[1]), dtype=np.float64)
    fitted = 0
    for cell in range(n_cells):
        rows = cells == cell
        n = int(rows.sum())
        if n < MINIMUM_CELL_TOKENS:
            continue
        means[cell] = residual[rows].mean(axis=0)
        fitted += 1
    return Correction(means=means, fitted_cells=fitted)


def normalised_error(residual: np.ndarray, target_variance: float) -> float:
    """Mean squared residual over the target's own variance -- the NMSE this
    programme's replacement stage already reports, restricted to a token set."""

    if target_variance <= 0.0:
        raise ValueError("target variance must be positive")
    return float((residual**2).mean() / target_variance)


def grouping_reduction(
    residual: np.ndarray,
    cells: np.ndarray,
    n_cells: int,
    *,
    train: np.ndarray,
    test: np.ndarray,
    target_variance: float,
) -> dict[str, Any]:
    """Held-out NMSE reduction from grouping tokens this way, with its ceiling.

    ``in_sample`` fits and scores on the test tokens themselves. It is not a
    result; it is the arithmetic most a mean-per-cell correction at this
    cardinality could remove, and the held-out number is reported against it
    because a small held-out reduction under a small ceiling means something
    different from a small one under a large ceiling.
    """

    before = normalised_error(residual[test], target_variance)
    fitted = fit_correction(residual[train], cells[train], n_cells)
    after = normalised_error(fitted.apply(residual[test], cells[test]), target_variance)
    oracle = fit_correction(residual[test], cells[test], n_cells)
    ceiling = normalised_error(oracle.apply(residual[test], cells[test]), target_variance)
    return {
        "nmse_before": before,
        "nmse_after": after,
        "reduction": before - after,
        "reduction_fraction": (before - after) / before if before > 0 else 0.0,
        "in_sample_ceiling_reduction": before - ceiling,
        "cells_fitted_on_train": fitted.fitted_cells,
        "n_train_tokens": int(train.size),
        "n_test_tokens": int(test.size),
    }


def expert_set_from_residue(
    cells: np.ndarray, residues: np.ndarray, *, train: np.ndarray, test: np.ndarray
) -> dict[str, Any]:
    """Can the expert set be predicted from the amino acid alone?

    The majority expert set per residue, fitted on train and scored on test,
    against the majority-cell rate that ignores the residue too. If this reads
    near one, ``routing`` and ``residue`` are not two groupings and no result
    that contrasts them is about routing.
    """

    table: dict[int, int] = {}
    for residue in np.unique(residues[train]):
        rows = train[residues[train] == residue]
        table[int(residue)] = int(np.bincount(cells[rows]).argmax())
    fallback = int(np.bincount(cells[train]).argmax())
    predicted = np.array([table.get(int(r), fallback) for r in residues[test]], dtype=np.int64)
    accuracy = float((predicted == cells[test]).mean())
    majority = float((cells[test] == fallback).mean())
    return {
        "accuracy_from_residue": accuracy,
        "majority_cell_rate": majority,
        "skill_over_majority": (accuracy - majority) / (1.0 - majority)
        if majority < 1.0
        else 0.0,
        "n_residue_types_seen": len(table),
    }
