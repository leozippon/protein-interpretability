#!/usr/bin/env python3
"""Does a MoE replacement's error live where the router decides?

ProGenMech's cross-layer transcoder abstracts ProGen3's router away by
construction, and their paper says so. This stage measures the consequence.
A transcoder is a continuous function of a MoE block's input; the block is not,
because which two of eight experts run is a ``topk`` and the true function is
piecewise. **If the replacement's residual is structured by the routing decision,
the failure has a named mechanism and a matched control can say whether that
mechanism is a property of sparse MoE models or of protein ones. If it is not,
"the router was abstracted away" is not the operative defect and the failure is
diffuse.** Both answers are worth the compute; only one of them leads anywhere.

This is D2.g item 4 of the audit document's plan, which states it as a
within-model question needing no second MoE to *pose* -- and it is deliberately
posed before a text MoE arm is admitted, so that arm is only ever bought against
a positive result.

**Literature gate, run before this stage was written.** Queries: "mixture of
experts interpretability", "router analysis expert specialisation", "expert
ablation interpretability", "path patching mixture of experts", "transcoder MoE
replacement", each crossed with protein language model and with sparse MoE. The
gate's findings are recorded in the EXP entry for this track; what it changed
here is that the statistic is a *correction* rather than a correlation, because
a rank correlation between a per-token error and a router statistic is the shape
this programme has already had to retract twice (F4, EXP-R2-120).

Gates, in the order they can kill the claim:

``loader``            the backbone is really loaded, through the self-check that
                      separates a converted ProGen3 from one whose experts came
                      back silently random (L24).
``router_addressing`` the router being read is the router the block selects on.
                      The block *returns* a distribution in the hidden states'
                      dtype while selecting on a float32 one, and EXP-R2-130
                      measured those two disagreeing on 3.3% of tokens, so the
                      distribution is recomputed and then checked against the
                      returned one rather than trusted.
``attainability``     a grouping can only remove error if it has cells to remove
                      it into. A layer whose routing is concentrated in one
                      expert set is reported **unmeasurable**, not failing
                      (standing rule 2), and every held-out reduction is
                      reported against the in-sample ceiling at the same
                      cardinality.
``routing_locality``  the hypothesis, against two baselines with declared roles:
                      a random grouping at **matched cardinality** (standing rule
                      28 -- it costs the same degrees of freedom and knows
                      nothing), and the residue identity, which is the confound
                      rather than the control. If routing in this model is
                      largely a lookup on the current amino acid, then a routing
                      cell and a residue cell are one cell under two names; the
                      incremental test asks what routing adds *beyond* the
                      residue, and the descriptive section measures the lookup
                      directly instead of leaving it to be inferred.

Outside the panel contract by design, like the two stages before it: it measures
ProGen3-112M, which is not a panel arm.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.arms import (  # noqa: E402
    DEFAULT_CORPUS_DRAW_SEED,
    REPO,
    protein_cohort,
)
from src.transfer.io import write_json  # noqa: E402
from src.transfer.progen3 import (  # noqa: E402
    ProGen3,
    content_mask,
    load_progen3,
    moe_intercept,
    router_probabilities,
    router_probabilities_agree,
    self_check,
)
from src.transfer.routing import (  # noqa: E402
    cell_occupancy,
    expert_set_from_residue,
    expert_sets,
    fit_correction,
    normalised_error,
    router_dispersion,
    routing_cells,
)
from src.transfer.statistics import make_group_splits, paired_group_bootstrap  # noqa: E402
from src.transfer.transcoders import DEFAULT_REPLACEMENT, load_replacement  # noqa: E402

SCHEMA_VERSION = "r2_transfer_routing_locality_v1"
DEFAULT_OUT = REPO / "results/transfer/routing_locality"

#: Groupings scored at every layer, with the role each one plays. Declared as
#: data so the artefact carries the roles and a reader never has to reconstruct
#: which of these is the hypothesis and which is the free baseline.
GROUPING_ROLES = {
    "routing": "hypothesis: the token's selected expert set",
    "random": "free baseline at matched cardinality; knows nothing, costs the "
    "same degrees of freedom (standing rule 28)",
    "residue": "confound, not control: if routing is a lookup on the amino acid "
    "then routing and residue are one grouping under two names",
}


@torch.no_grad()
def collect(
    pg: ProGen3,
    transcoder: Any,
    sequences: list[str],
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Per-token residuals, routing cells and residue labels, for every layer.

    One forward pass per batch. The transcoder is applied inside the tap and its
    output is *not* substituted -- this stage measures where the replacement's
    error is, which is a property of the original run, not of a replaced one.
    """

    n_layers = pg.n_layers
    top_k = int(pg.config.num_experts_per_tok)
    residuals: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    cells: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    dispersions: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    target_squares = np.zeros(n_layers, dtype=np.float64)
    target_sums = np.zeros(n_layers, dtype=np.float64)
    residues: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    scored: dict[str, Any] = {}

    def tap(layer: int, x: torch.Tensor, y: torch.Tensor) -> None:
        keep = scored["mask"]
        width = y.shape[-1]
        target = y.reshape(-1, width)[keep].float()
        recon = transcoder(layer, x).reshape(-1, width)[keep].float()
        residuals[layer].append((target - recon).cpu().numpy().astype(np.float32))
        probabilities = (
            router_probabilities(pg, layer, x).reshape(-1, pg.config.num_experts)[keep].cpu().numpy()
        )
        cells[layer].append(routing_cells(probabilities.astype(np.float64), top_k=top_k))
        dispersions[layer].append(probabilities.astype(np.float64))
        target_squares[layer] += float((target.double() ** 2).sum())
        target_sums[layer] += float(target.double().sum())
        return None

    n_tokens = 0
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        batch = pg.batch(chunk)
        mask = content_mask(pg, batch["input_ids"])
        scored["mask"] = mask.reshape(-1)
        with moe_intercept(pg, tap):
            pg.model(
                input_ids=batch["input_ids"],
                position_ids=batch["position_ids"],
                sequence_ids=batch["sequence_ids"],
                use_cache=False,
                return_dict=True,
            )
        flat = mask.reshape(-1)
        residues.append(batch["input_ids"].reshape(-1)[flat].cpu().numpy())
        index = torch.arange(len(chunk), device=mask.device).unsqueeze(1).expand_as(mask)
        groups.append((index.reshape(-1)[flat].cpu().numpy() + start).astype(np.int64))
        n_tokens += int(flat.sum())

    stacked_residual = [np.concatenate(layer_parts) for layer_parts in residuals]
    counts = float(n_tokens)
    width = stacked_residual[0].shape[1]
    # Variance of the target over tokens and features, matching the per-layer
    # NMSE the replacement stage reports, accumulated so the tensors need not be
    # kept: E[y^2] - E[y]^2 over the same scored population.
    variance = target_squares / (counts * width) - (target_sums / (counts * width)) ** 2
    return {
        "residual": stacked_residual,
        "cells": [np.concatenate(part) for part in cells],
        "router_probabilities": [np.concatenate(part) for part in dispersions],
        "residues": np.concatenate(residues),
        "groups": np.concatenate(groups),
        "target_variance": variance,
        "n_tokens": n_tokens,
    }


def cross_fitted_error(
    residual: np.ndarray,
    cells: np.ndarray,
    n_cells: int,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Per-token squared residual after a correction fitted on other sequences.

    Every token is corrected by a mean fitted on folds it is not in, and
    ``make_group_splits`` has already checked that the folds are disjoint in
    sequence and that each token is tested exactly once. So the returned vector
    is entirely out-of-sample and can be resampled as one population.
    """

    out = np.zeros(residual.shape[0], dtype=np.float64)
    for train, test in splits:
        correction = fit_correction(residual[train], cells[train], n_cells)
        out[test] = (correction.apply(residual[test], cells[test]) ** 2).sum(axis=1)
    return out


def removed_fraction(uncorrected: np.ndarray, corrected: np.ndarray) -> float:
    """Share of squared error a correction removed. Scale-free within a layer."""

    total = float(uncorrected.mean())
    if total <= 0.0:
        raise ValueError("the uncorrected error is not positive; nothing can be removed")
    return 1.0 - float(corrected.mean()) / total


def compare(
    uncorrected: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    """Paired sequence-level bootstrap of one grouping's advantage over another."""

    record = paired_group_bootstrap(
        uncorrected,
        left,
        right,
        groups,
        lambda truth, prediction: removed_fraction(truth, prediction),
        seed=seed,
        n_bootstrap=replicates,
    )
    record["resampling_unit"] = "cohort sequence"
    record["excludes_zero"] = bool(
        record["difference_ci95"][0] > 0.0 or record["difference_ci95"][1] < 0.0
    )
    return record


def layer_record(
    data: dict[str, Any],
    layer: int,
    *,
    n_expert_cells: int,
    top_k: int,
    seed: int,
    folds: int,
    replicates: int,
) -> dict[str, Any]:
    """Everything measured at one layer, with its own attainability verdict."""

    residual = data["residual"][layer].astype(np.float64)
    cells = data["cells"][layer]
    residues = data["residues"]
    groups = data["groups"]
    rng = np.random.default_rng(seed + layer)

    occupancy = cell_occupancy(cells, n_expert_cells)
    record: dict[str, Any] = {
        "layer": layer,
        "target_variance": float(data["target_variance"][layer]),
        "nmse": normalised_error(residual, float(data["target_variance"][layer])),
        "router": router_dispersion(data["router_probabilities"][layer], top_k=top_k),
        "occupancy": occupancy,
    }
    if occupancy["degenerate"]:
        record["verdict"] = "UNMEASURABLE"
        record["note"] = occupancy["degenerate_reason"]
        return record

    splits = make_group_splits(
        np.zeros(cells.size), groups, n_splits=folds, seed=seed + layer, task_type="regression"
    )
    record["expert_set_from_residue"] = expert_set_from_residue(
        cells, residues, train=splits[0][0], test=splits[0][1]
    )

    assignments = {
        "routing": (cells, n_expert_cells),
        "random": (rng.integers(0, n_expert_cells, size=cells.size), n_expert_cells),
        "residue": (
            np.unique(residues, return_inverse=True)[1],
            int(np.unique(residues).size),
        ),
    }
    uncorrected = (residual**2).sum(axis=1)
    corrected = {
        name: cross_fitted_error(residual, assignment, n_cells, splits)
        for name, (assignment, n_cells) in assignments.items()
    }
    record["removed_fraction"] = {
        name: removed_fraction(uncorrected, values) for name, values in corrected.items()
    }
    record["in_sample_ceiling"] = {
        name: removed_fraction(
            uncorrected,
            (
                fit_correction(residual, assignment, n_cells).apply(residual, assignment) ** 2
            ).sum(axis=1),
        )
        for name, (assignment, n_cells) in assignments.items()
    }
    record["routing_over_random"] = compare(
        uncorrected, corrected["routing"], corrected["random"], groups,
        seed=seed + 100 * layer, replicates=replicates,
    )

    # What routing adds beyond the residue: correct for the residue first, then
    # ask the same question of the remainder, against the same free baseline at
    # matched cardinality. Without this an affirmative result would be
    # indistinguishable from "experts are chosen by amino acid".
    residue_assignment, residue_cells = assignments["residue"]
    remainder = np.zeros_like(residual)
    for train, test in splits:
        correction = fit_correction(residual[train], residue_assignment[train], residue_cells)
        remainder[test] = correction.apply(residual[test], residue_assignment[test])
    remainder_uncorrected = (remainder**2).sum(axis=1)
    incremental = {
        name: cross_fitted_error(remainder, assignments[name][0], assignments[name][1], splits)
        for name in ("routing", "random")
    }
    record["incremental_over_residue"] = {
        "removed_fraction": {
            name: removed_fraction(remainder_uncorrected, values)
            for name, values in incremental.items()
        },
        "routing_over_random": compare(
            remainder_uncorrected, incremental["routing"], incremental["random"], groups,
            seed=seed + 100 * layer + 7, replicates=replicates,
        ),
    }
    difference = record["routing_over_random"]
    increment = record["incremental_over_residue"]["routing_over_random"]
    positive = difference["difference"] > 0 and difference["difference_ci95"][0] > 0.0
    increments = increment["difference"] > 0 and increment["difference_ci95"][0] > 0.0
    record["verdict"] = "PASS" if positive and increments else "FAIL"
    return record


def panel_verdict(layers: list[dict[str, Any]]) -> dict[str, Any]:
    """The reading over layers, stated without a majority threshold."""

    measured = [row for row in layers if row["verdict"] != "UNMEASURABLE"]
    passing = [row["layer"] for row in measured if row["verdict"] == "PASS"]
    unmeasurable = [row["layer"] for row in layers if row["verdict"] == "UNMEASURABLE"]
    if not measured:
        verdict = "UNMEASURABLE"
    elif len(passing) == len(measured):
        verdict = "PASS"
    elif passing:
        verdict = "MIXED"
    else:
        verdict = "FAIL"
    return {
        "verdict": verdict,
        "layers_measured": [row["layer"] for row in measured],
        "layers_passing": passing,
        "layers_unmeasurable": unmeasurable,
        "note": (
            "PASS on a layer means the routing grouping removed more held-out "
            "residual than a random grouping of the same cardinality, and still "
            "did so after the residue identity had been corrected for first. "
            "FAIL means the replacement's error at that layer is not localised "
            "by the routing decision, which makes 'the router was abstracted "
            "away' the wrong account of why it fails there"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--replacement", type=Path, default=DEFAULT_REPLACEMENT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--sequences", type=int, default=256)
    parser.add_argument("--protein-min-len", type=int, default=64)
    parser.add_argument("--protein-max-len", type=int, default=246)
    parser.add_argument("--cohort-draw-seed", type=int, default=DEFAULT_CORPUS_DRAW_SEED)
    parser.add_argument("--cohort-skip", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "grouping_roles": GROUPING_ROLES,
        "cohort_band_residues": [args.protein_min_len, args.protein_max_len],
    }

    print("[cohort] drawing")
    cohort = protein_cohort(
        args.sequences,
        args.protein_min_len,
        args.protein_max_len,
        skip=args.cohort_skip,
        seed=args.cohort_draw_seed or None,
    )
    sequences = list(cohort.records)
    payload["cohort"] = {
        "name": cohort.name,
        "digest": cohort.digest,
        "provenance_digest": cohort.provenance_digest,
        "sampling": cohort.sampling,
        "n_sequences": len(cohort),
    }
    print(f"  {len(sequences)} sequences")

    print("[loader] loading ProGen3-112M and self-checking the conversion")
    load_kwargs: dict[str, Any] = {"device": args.device, "dtype": getattr(torch, args.dtype)}
    if args.checkpoint is not None:
        load_kwargs["checkpoint"] = args.checkpoint
    pg = load_progen3(**load_kwargs)
    payload["gates"] = {"loader": self_check(pg)}
    print(f"  self-check NLL {payload['gates']['loader']['nll']:.4f} PASS")

    transcoder, _, hyperparameters = load_replacement(args.replacement)
    transcoder.to(pg.device)
    payload["replacement"] = {
        "path": str(args.replacement),
        "num_layers": int(hyperparameters.num_layers),
        "d_hidden": int(hyperparameters.d_hidden),
        "k": int(hyperparameters.k),
    }

    print("[gate] checking the router being read is the router the block selects on")
    probe = pg.batch(sequences[: args.batch_size])
    seen: dict[int, dict[str, Any]] = {}
    handles = []
    for layer, block in enumerate(pg.moe_blocks):

        def hook(module: Any, inputs: Any, output: Any, layer: int = layer) -> None:
            seen[layer] = router_probabilities_agree(pg, layer, inputs[0], output[1])
            return None

        handles.append(block.register_forward_hook(hook))
    try:
        with torch.no_grad():
            pg.model(
                input_ids=probe["input_ids"],
                position_ids=probe["position_ids"],
                sequence_ids=probe["sequence_ids"],
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    worst = max(row["max_absolute_difference"] for row in seen.values())
    disagreement = max(row["selected_set_disagreement"] for row in seen.values())
    payload["gates"]["router_addressing"] = {
        "per_layer": seen,
        "max_absolute_difference": worst,
        "max_selected_set_disagreement": disagreement,
        "verdict": "PASS" if worst < 1e-2 else "FAIL",
        "note": (
            "the recomputed float32 router and the distribution the block returns "
            "in the hidden states' dtype must agree to a dtype tolerance; the "
            "selected-set disagreement is reported because it is the quantity "
            "EXP-R2-130 measured at 3.3% and is the reason this stage recomputes"
        ),
    }
    if payload["gates"]["router_addressing"]["verdict"] != "PASS":
        write_json(args.out / "routing_locality.json", payload)
        raise RuntimeError(
            f"router addressing gate failed: recomputed and returned router "
            f"distributions differ by {worst:.4g}, which is not a dtype difference. "
            "Every routing cell would be a label for the wrong module."
        )
    print(f"  max |difference| {worst:.3e}, selected-set disagreement {disagreement:.4f} PASS")

    print("[collect] one forward pass per batch, residuals and routing per token")
    data = collect(pg, transcoder, sequences, batch_size=args.batch_size)
    print(f"  {data['n_tokens']} scored residue positions")

    n_expert_cells = len(expert_sets(int(pg.config.num_experts), int(pg.config.num_experts_per_tok)))
    payload["condition"] = {
        "n_experts": int(pg.config.num_experts),
        "experts_per_token": int(pg.config.num_experts_per_tok),
        "n_expert_set_cells": n_expert_cells,
        "n_tokens": data["n_tokens"],
        "estimand": "per-token squared residual of the released replacement at "
        "each MoE block's output, corrected by a per-cell mean fitted on "
        "sequence-disjoint folds",
    }

    print("[measure] per layer")
    layers = [
        layer_record(
            data,
            layer,
            n_expert_cells=n_expert_cells,
            top_k=int(pg.config.num_experts_per_tok),
            seed=args.seed,
            folds=args.folds,
            replicates=args.bootstrap,
        )
        for layer in range(pg.n_layers)
    ]
    for row in layers:
        if row["verdict"] == "UNMEASURABLE":
            print(f"  L{row['layer']:<2d} UNMEASURABLE  {row['note']}")
            continue
        removed = row["removed_fraction"]
        difference = row["routing_over_random"]
        increment = row["incremental_over_residue"]["routing_over_random"]
        print(
            f"  L{row['layer']:<2d} nmse {row['nmse']:.4f}  removed routing "
            f"{removed['routing']:+.4f} random {removed['random']:+.4f} residue "
            f"{removed['residue']:+.4f}  routing-random {difference['difference']:+.4f} "
            f"{difference['difference_ci95']}  incremental {increment['difference']:+.4f} "
            f"{increment['difference_ci95']}  {row['verdict']}"
        )

    payload["layers"] = layers
    payload["gates"]["attainability"] = {
        "unmeasurable_layers": [r["layer"] for r in layers if r["verdict"] == "UNMEASURABLE"],
        "note": "a layer whose routing is concentrated in one expert set has no "
        "variation for a per-cell mean to fit, and is reported unmeasurable "
        "rather than as a failure of the hypothesis (standing rule 2)",
    }
    payload["gates"]["routing_locality"] = panel_verdict(layers)
    payload["verdict"] = payload["gates"]["routing_locality"]["verdict"]

    write_json(args.out / "routing_locality.json", payload)
    print()
    print(f"[verdict] {payload['verdict']}  "
          f"passing {payload['gates']['routing_locality']['layers_passing']}  "
          f"unmeasurable {payload['gates']['routing_locality']['layers_unmeasurable']}")


if __name__ == "__main__":
    main()
