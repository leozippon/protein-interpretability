#!/usr/bin/env python3
"""Does a low-dimensional subspace carry the antecedent, on protein as on text?

Distributed alignment search, ported to protein decoders. On a repeat probe the
decoder must carry one variable to the read-out: *what followed the antecedent*.
DAS learns an orthonormal basis and swaps only the projection onto it between a
base run and a counterfactual source run; if the prediction switches to the
source's continuation, that subspace mediates the variable. The statistic is
interchange-intervention accuracy.

**Literature gate, run before this stage was written.** Queries: "distributed
alignment search" / "interchange intervention" / "causal abstraction" /
"interchange intervention accuracy", each crossed with protein language model,
biological sequence model and genomic language model. Found: Geiger et al.
(arXiv:2303.02536) which introduces DAS; Wu et al. (arXiv:2305.08809) Boundless
DAS; the causal-abstraction foundation (arXiv:2301.04709) and survey
(arXiv:2410.20161); and 2026 work applying subspace interventions to in-context
learning (arXiv:2605.18830). **No application of DAS or of any interchange
intervention to a protein or biological-sequence model was found** -- the only
"DAS" in a biological context is an unrelated distributed *alignment system* for
sequence alignment, which is worth recording because it makes the search term
look productive when it is not. So this is a transfer test of a named text
method rather than a replication.

Gates, in the order they can kill the claim:

``layout``          the arm's architecture is one the intervention understands,
                    and it is not loaded in half precision (rule 15b).
``invariants``      writing a zero delta is the identity, and writing a large one
                    MOVES the metric. The second is the load-bearing one: a hook
                    that never bound passes every null test while measuring an
                    unpatched model.
``attainability``   the full-vector patch at this site, which is the arithmetic
                    ceiling any subspace is bounded by. **Checked on the text
                    control first** (rule 2). A subspace threshold applied where
                    the ceiling is low is a specification defect, not a result.
``free baselines``  a random subspace, a variance-matched random subspace, and
                    the unembedding-difference subspace -- the last computable
                    from the model's weights and the case labels with no
                    activations and no training (rule 28). A learned subspace
                    that does not beat it has rediscovered the output aperture.

Every reported number is held out: the split is group-disjoint at the probe
record, because ``build_path_cases`` draws several nested prefixes of one
sequence and a case-level split would leak the whole context.

Dense arms only, and that is a property of the panel rather than a filter here:
every member of ``arms.PANEL`` is dense, and the only MoE model in this
repository is ProGen3-112M, which is deliberately outside the panel.
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

from src.transfer import das  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    DEFAULT_CORPUS_DRAW_SEED,
    REPO,
    load_arm,
    symbols_per_token,
)
from src.transfer.circuits import (  # noqa: E402
    PROTEIN_EXACT_CRITERION,
    TEXT_EXACT_CRITERION,
    fit_unigram,
    natural_repeat_probes,
    protein_repeat_cohort,
    text_repeat_cohort,
)
from src.transfer.io import write_json  # noqa: E402
from src.transfer.path_patching import build_path_cases  # noqa: E402
from src.transfer.scoring import analysis_layer  # noqa: E402
from src.transfer.statistics import bootstrap_unit_floor  # noqa: E402

SCHEMA_VERSION = das.SCHEMA_VERSION
DEFAULT_OUT = REPO / "results/transfer/das_subspace"

#: Relative depths, resolved per arm. A layer index is not comparable across a
#: 36-layer and a 27-layer arm; a fraction is (Appendix B rule 26).
LAYER_FRACTIONS = (0.25, 0.5, 0.75)


def prepare_batches(
    arm: Any, cases: list[Any], *, layer: int, batch_size: int, device: str
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    """Everything a cell needs, captured once: deltas, metrics and eligibility.

    The source run's log-probabilities are captured here too, because the
    training objective matches them and re-running the source every step would
    triple the cost for a tensor that never changes.
    """

    batches: list[dict[str, torch.Tensor]] = []
    source_hits: list[float] = []
    denominators: list[float] = []
    # Every case is truncated so that its read-out is its own last token, so
    # cases of different lengths have different read-out rows. Batching them
    # together would need padding, and padding moves the read-out; grouping by
    # length keeps one read-out row per batch and leaves the geometry alone.
    by_length: dict[int, list[Any]] = {}
    for case in cases:
        by_length.setdefault(len(case.input_ids), []).append(case)
    chunks = [
        group[start : start + batch_size]
        for group in by_length.values()
        for start in range(0, len(group), batch_size)
    ]
    for chunk in chunks:
        if not chunk:
            continue
        position = chunk[0].position_q
        base = torch.tensor([c.input_ids for c in chunk], device=device)
        source = torch.tensor([c.corrupt_ids for c in chunk], device=device)
        clean = torch.tensor([c.token_clean for c in chunk], device=device)
        corrupt = torch.tensor([c.token_corrupt for c in chunk], device=device)

        base_residual = das.capture_residual(arm, base, layer=layer, position=position)
        source_residual = das.capture_residual(arm, source, layer=layer, position=position)
        with torch.no_grad():
            base_logits = das.patched_logits(
                arm, base, layer=layer, position=position, delta=None
            )
            source_logits = das.patched_logits(
                arm, source, layer=layer, position=position, delta=None
            )
        metric_base = das.metric(base_logits, clean, corrupt)
        metric_source = das.metric(source_logits, clean, corrupt)
        source_hits.extend(
            (source_logits.argmax(dim=-1) == corrupt).float().cpu().tolist()
        )
        denominators.extend((metric_base - metric_source).cpu().tolist())
        batches.append(
            {
                "ids": base,
                "delta": source_residual - base_residual,
                "clean": clean,
                "corrupt": corrupt,
                "metric_base": metric_base,
                "metric_source": metric_source,
                "source_logprobs": torch.log_softmax(source_logits, dim=-1),
                "group": torch.tensor([c.probe_index for c in chunk], device=device),
                "position": position,
            }
        )
    return batches, {
        "source_predicts_counterfactual": (
            float(np.mean(source_hits)) if source_hits else float("nan")
        ),
        "mean_denominator_logits": (
            float(np.mean(denominators)) if denominators else float("nan")
        ),
        "n_batches": len(batches),
    }


def split_batches(
    batches: list[dict[str, torch.Tensor]], *, seed: int, holdout: float
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    """Group-disjoint split at the probe record, never at the case."""

    groups = sorted({int(g) for batch in batches for g in batch["group"].cpu().tolist()})
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(groups)
    n_test = max(1, int(round(len(groups) * holdout)))
    test = set(int(g) for g in shuffled[:n_test])
    train_batches, test_batches = [], []
    for batch in batches:
        member = [int(g) in test for g in batch["group"].cpu().tolist()]
        if all(member):
            test_batches.append(batch)
        elif not any(member):
            train_batches.append(batch)
        # A batch straddling the split is dropped rather than assigned: keeping
        # it would leak train records into a held-out number.
    return train_batches, test_batches


def run_arm(arm_name: str, args: argparse.Namespace) -> dict[str, Any]:
    arm = load_arm(arm_name, device=args.device, dtype="float32")
    das.require_das_dtype(arm)

    if arm.modality == "protein":
        cohort = protein_repeat_cohort(
            args.records,
            min_len=args.protein_min_len,
            max_len=args.protein_max_len,
            criterion=PROTEIN_EXACT_CRITERION,
            seed=args.cohort_draw_seed,
            name="das_protein",
        )
    else:
        cohort = text_repeat_cohort(
            args.records,
            criterion=TEXT_EXACT_CRITERION,
            seed=args.cohort_draw_seed,
            name="das_text",
        )

    probes = natural_repeat_probes(arm, cohort, max_tokens=args.max_tokens)
    unigram = fit_unigram(arm, cohort.input_strings(arm), max_tokens=args.max_tokens)
    cases, provenance = build_path_cases(
        arm,
        probes,
        unigram,
        n_cases=args.cases,
        cases_per_probe=args.cases_per_probe,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    if not cases:
        raise RuntimeError(f"{arm_name}: no eligible cases; nothing can be measured")

    unembedding = arm.model.get_output_embeddings().weight.detach().float()
    record: dict[str, Any] = {
        "arm": arm_name,
        "modality": arm.modality,
        "n_layers": arm.n_layer,
        "d_model": arm.d_model,
        # Reported because a token distance is not a matched geometry across
        # arms differing 4.4x in symbols per token (Appendix B rule 26).
        "symbols_per_token": symbols_per_token(
            arm, cohort.input_strings(arm)[: min(16, len(cohort.records))], args.max_tokens
        ),
        "cohort": {
            "name": cohort.name,
            "digest": cohort.digest,
            "n_records": len(cohort.records),
            "band": [args.protein_min_len, args.protein_max_len],
        },
        "case_provenance": provenance,
        "cells": [],
    }

    for fraction in LAYER_FRACTIONS:
        layer = analysis_layer(arm.n_layer, fraction)
        batches, diagnostics = prepare_batches(
            arm, cases, layer=layer, batch_size=args.batch_size, device=args.device
        )
        if not batches:
            continue
        das.invariants(arm, batches[0], layer=layer)
        train, test = split_batches(batches, seed=args.seed, holdout=args.holdout)
        if not train or not test:
            continue

        ceiling = das.evaluate_basis(
            arm, test, None, layer=layer, full_vector=True
        )
        null = das.evaluate_basis(arm, test, None, layer=layer)
        cell: dict[str, Any] = {
            "layer_fraction": fraction,
            "layer": layer,
            "diagnostics": diagnostics,
            "n_train_batches": len(train),
            "n_test_batches": len(test),
            "unit_floor": bootstrap_unit_floor(ceiling["n_groups"]),
            "attainability": {
                "full_vector_iia": ceiling["iia"],
                "full_vector_recovery": ceiling["recovery_mean"],
                "null_patch_iia": null["iia"],
                "note": "the full-vector patch is the arithmetic ceiling any "
                "subspace is bounded by at this site; a subspace threshold "
                "applied where this is low is a specification defect",
            },
            "dimensions": [],
        }

        for dimension in args.dimensions:
            if dimension > arm.d_model:
                continue
            learned: list[dict[str, Any]] = []
            bases: list[torch.Tensor] = []
            for seed_index in range(args.rotation_seeds):
                config = das.DasConfig(
                    layer=layer,
                    dimension=dimension,
                    steps=args.steps,
                    learning_rate=args.learning_rate,
                    seed=args.seed + 7919 * seed_index,
                )
                subspace, history = das.train_subspace(arm, train, config)
                basis = subspace.basis().detach()
                bases.append(basis)
                scored = das.evaluate_basis(arm, test, basis, layer=layer)
                in_sample = das.evaluate_basis(arm, train, basis, layer=layer)
                learned.append(
                    {
                        "seed": config.seed,
                        "held_out_iia": scored["iia"],
                        "in_sample_iia": in_sample["iia"],
                        "held_out_recovery": scored["recovery_mean"],
                        "recovery_quantiles": scored["recovery_quantiles"],
                        "swapped_variance_fraction": das.variance_matched_dimension(
                            torch.cat([b["delta"] for b in test]), basis
                        ),
                        "final_kl": history[-1]["kl"] if history else None,
                    }
                )
            random = das.evaluate_basis(
                arm,
                test,
                das.random_basis(arm.d_model, dimension, seed=args.seed).to(args.device),
                layer=layer,
            )
            aperture = das.evaluate_basis(
                arm,
                test,
                das.unembedding_difference_basis(unembedding, cases, dimension).to(
                    args.device
                ),
                layer=layer,
            )
            angles = (
                das.principal_angles(bases[0], bases[1]) if len(bases) > 1 else []
            )
            cell["dimensions"].append(
                {
                    "dimension": dimension,
                    "dimension_fraction": dimension / arm.d_model,
                    "learned": learned,
                    "learned_iia_median": float(
                        np.median([entry["held_out_iia"] for entry in learned])
                    ),
                    "learned_iia_spread": [
                        float(min(entry["held_out_iia"] for entry in learned)),
                        float(max(entry["held_out_iia"] for entry in learned)),
                    ],
                    "random_subspace_iia": random["iia"],
                    "unembedding_difference_iia": aperture["iia"],
                    "cross_seed_principal_angles_deg": angles[:8],
                }
            )
        record["cells"].append(cell)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["gpt2-large", "protgpt2"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--records", type=int, default=96)
    parser.add_argument("--cases", type=int, default=192)
    parser.add_argument("--cases-per-probe", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--protein-min-len", type=int, default=200)
    parser.add_argument("--protein-max-len", type=int, default=800)
    parser.add_argument("--dimensions", type=int, nargs="+", default=list(das.SUBSPACE_DIMENSIONS))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--rotation-seeds", type=int, default=3)
    parser.add_argument("--holdout", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--cohort-draw-seed", type=int, default=DEFAULT_CORPUS_DRAW_SEED)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "condition": {
            "estimand": "interchange-intervention accuracy: swap the projection "
            "onto a learned subspace of the residual stream entering the analysis "
            "layer at the read-out position, between a base run and a source run "
            "differing at exactly one antecedent-successor token",
            "layer_declaration": "relative depth, resolved per arm by "
            "scoring.analysis_layer; a layer index is not comparable across arms "
            "of different depth",
            "training_objective": "KL from the source run's next-token "
            "distribution; deliberately not the reported IIA, which is an argmax "
            "threshold that training on would fit",
            "split": "group-disjoint at the probe record; batches straddling the "
            "split are dropped rather than assigned",
            "free_baselines": "random subspace of the same dimension, and the "
            "unembedding-difference subspace computable from weights and labels "
            "alone (standing rule 28)",
            "dense_only": "every arm in the panel is dense; the only MoE model "
            "in this repository is outside the panel by design",
        },
        "arms": [],
    }
    for arm_name in args.arms:
        print(f"[arm] {arm_name}")
        payload["arms"].append(run_arm(arm_name, args))
        write_json(args.out / "das_subspace.json", payload)
    print(f"wrote {args.out / 'das_subspace.json'}")


if __name__ == "__main__":
    main()
