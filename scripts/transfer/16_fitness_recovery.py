#!/usr/bin/env python3
"""Is ProGenMech's fitness base above what is free, before any recovery ratio is read?

ProGenMech (arXiv:2606.16044) reports that a sparse circuit over a cross-layer
transcoder recovers "~95%" and "~80%" of ProGen3-112M's zero-shot fitness
performance. Read from their released artefacts, both are **ratios of Spearman
correlations**: 0.28/0.29 and 0.23/0.28, over eight ProteinGym assays, quoted as
mean +/- SD across assays with no interval on the ratio and no floor under it.

Standing rule 28: a selector is scored against the trivial baseline available
from its own coordinates. The analogue for a fitness predictor is the score
computable from the mutation string before any model exists. Standing rule 2:
a gate is checked on the control before it is applied. Both say the same thing
here -- **the base has to be shown to be above free before a ratio against it
means anything** -- and that is the only question this stage answers.

Gates, in the order they can kill the claim:

``loader``          the backbone is really loaded, via the self-check that
                    separates a correctly converted ProGen3 from one whose
                    experts came back silently random (see ``src.transfer.progen3``).
``cohort``          every assay is a genuine single-wildtype substitution set and
                    was drawn under a seeded permutation rather than as a prefix.
``free_baseline``   whether the model's own zero-shot Spearman exceeds BLOSUM62's
                    on the same variants, paired across assays. **If it does not,
                    no recovery ratio computed against that base is interpretable**,
                    and the limitation belongs to the evaluation interface rather
                    than to any dictionary built on top of it.

What is deliberately not here. The replacement arm -- ProGen3 with the released
per-layer transcoder substituted -- is the *next* pass. Rule 2 orders them: the
attainability of the base is a precondition for reading a recovery ratio, so it
is measured first and on its own. The fully-ablated endpoint belongs with the
replacement, because a floor is a property of a ratio and this pass computes none.

One convention, recorded because it removes a candidate explanation rather than
because it is a choice. Within one assay the wild type is fixed and the sequences
are equal length, so the mutant-minus-wildtype log-likelihood, the raw mutant
log-likelihood, and their per-residue means are monotone transforms of each other
and give identical Spearman -- verified bit-identical on 400 single mutants of
GRB2_HUMAN_Faure_2021. A disagreement between two reported fitness numbers on one
assay is therefore never explained by which of those three was used.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.arms import REPO  # noqa: E402
from src.transfer.fitness import (  # noqa: E402
    PROGENMECH_ASSAYS,
    PROGENMECH_TEST_SEQUENCES,
    PROGENMECH_TRAIN_SEQUENCES,
    Assay,
    load_assay,
)
from src.transfer.io import write_json  # noqa: E402
from src.transfer.progen3 import (  # noqa: E402
    ProGen3,
    load_progen3,
    scored_logits,
    self_check,
    token_nll,
)
from src.transfer.statistics import bootstrap_unit_floor, mean_interval  # noqa: E402

SCHEMA_VERSION = "r2_transfer_fitness_recovery_v1"
DEFAULT_OUT = REPO / "results/transfer/fitness_recovery"

#: What ProGenMech reports, so the artefact carries the numbers it is read
#: against instead of leaving them to a reader's memory. Quoted from
#: arXiv:2606.16044, not measured here.
PROGENMECH_REPORTED = {
    "progen3_zero_shot_spearman": 0.29,
    "full_clt_spearman": [0.28, 0.12],
    "circuit_spearman": [0.23, 0.13],
    "performance_recovery_full": "0.28 / 0.29",
    "performance_recovery_circuit": "0.23 / 0.28 -- denominator is the CLT, not the model",
    "n_test_sequences_per_assay": PROGENMECH_TEST_SEQUENCES,
}


@torch.no_grad()
def sequence_logprob(
    pg: ProGen3,
    sequences: list[str],
    *,
    batch_size: int,
    directions: tuple[bool, ...],
) -> np.ndarray:
    """Summed log-likelihood per sequence, averaged over the scored directions.

    ProGen3 is trained N->C and C->N and its published perplexities are the mean
    of the two, so a single-direction score is a different estimand and has to be
    asked for explicitly.
    """

    totals = np.zeros((len(sequences), len(directions)), dtype=np.float64)
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        for index, reverse in enumerate(directions):
            batch = pg.batch(chunk, reverse=reverse)
            logits, targets, mask = scored_logits(pg, batch)
            nll = token_nll(logits, targets)
            totals[start : start + len(chunk), index] = (
                -(nll * mask).sum(1).double().cpu().numpy()
            )
    return totals.mean(axis=1)


def spearman_interval(
    prediction: np.ndarray, measured: np.ndarray, *, replicates: int, seed: int
) -> dict[str, Any]:
    """Spearman with a percentile interval resampled over variants."""

    point = float(stats.spearmanr(prediction, measured).statistic)
    rng = np.random.default_rng(seed)
    n = len(measured)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        pick = rng.integers(0, n, size=n)
        draws[index] = stats.spearmanr(prediction[pick], measured[pick]).statistic
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return {
        "spearman": point,
        "interval": [float(low), float(high)],
        "confidence": 0.95,
        "replicates": int(replicates),
        "resampling_unit": "assay variant",
    }


def score_assay(
    pg: ProGen3,
    assay: Assay,
    *,
    batch_size: int,
    directions: tuple[bool, ...],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Every predictor's correlation with measured fitness on one assay."""

    model = sequence_logprob(pg, assay.sequences, batch_size=batch_size, directions=directions)
    blosum = assay.blosum
    shuffled = np.random.default_rng(seed + 1).permutation(assay.scores)

    record = assay.record()
    record["model"] = spearman_interval(model, assay.scores, replicates=replicates, seed=seed)
    record["blosum62"] = spearman_interval(
        blosum, assay.scores, replicates=replicates, seed=seed
    )
    # The negative control shares the model's own predictions and destroys only
    # the pairing, so a non-zero reading here is a defect in the statistic rather
    # than a property of any predictor.
    record["shuffled_label_control"] = {
        "spearman": float(stats.spearmanr(model, shuffled).statistic)
    }
    record["model_minus_blosum62"] = (
        record["model"]["spearman"] - record["blosum62"]["spearman"]
    )
    return record


def free_baseline_gate(assays: list[dict[str, Any]], *, gate: float) -> dict[str, Any]:
    """Does the model's own zero-shot fitness exceed what is free?

    Paired across assays, because the assays differ enormously in difficulty --
    the same predictor reads 0.05 on one and 0.40 on another -- and an unpaired
    comparison of two means over eight such units measures assay selection more
    than it measures the predictors.
    """

    differences = [a["model_minus_blosum62"] for a in assays]
    interval = mean_interval(differences)
    floor = bootstrap_unit_floor(len(differences))
    wins = sum(d > 0.0 for d in differences)
    # Exact two-sided sign test: with eight assays a t-interval leans on a
    # normality assumption the sample cannot support, so the distribution-free
    # statement is reported beside it rather than instead of it.
    sign_p = float(stats.binomtest(wins, len(differences), 0.5).pvalue)
    passes = bool(interval["interval"][0] > 0.0)
    return {
        "gate_margin": float(gate),
        "paired_difference": interval,
        "unit_floor": floor,
        "assays_where_model_wins": int(wins),
        "assays_total": len(differences),
        "sign_test_two_sided_p": sign_p,
        "model_mean_spearman": float(np.mean([a["model"]["spearman"] for a in assays])),
        "model_sd_spearman": float(np.std([a["model"]["spearman"] for a in assays], ddof=1)),
        "blosum62_mean_spearman": float(np.mean([a["blosum62"]["spearman"] for a in assays])),
        "blosum62_sd_spearman": float(
            np.std([a["blosum62"]["spearman"] for a in assays], ddof=1)
        ),
        "verdict": "PASS" if passes else "FAIL",
        "note": (
            "PASS means the model's own zero-shot fitness is above the free "
            "baseline on these assays, so a recovery ratio quoted against it has "
            "a base worth recovering. FAIL means it is not, and no ratio against "
            "that base -- theirs or ours -- is interpretable."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument(
        "--assays",
        nargs="+",
        default=list(PROGENMECH_ASSAYS),
        help="ProteinGym substitution assay names; the default is the eight read "
        "from ProGenMech's own released circuit-discovery outputs",
    )
    parser.add_argument("--variants", type=int, default=PROGENMECH_TEST_SEQUENCES)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--single-mutants-only",
        action="store_true",
        help="drop multi-mutant variants. Off by default because their own "
        "sampling does not filter them, and because dropping them changes the "
        "assay rather than cleaning it",
    )
    parser.add_argument(
        "--direction",
        default="both",
        choices=("both", "n_to_c", "c_to_n"),
        help="which training direction(s) to score; ProGen3's published "
        "perplexities are the mean of both",
    )
    parser.add_argument(
        "--sampling",
        default="uniform",
        choices=("uniform", "progenmech_stratified"),
        help="how the variants are drawn. 'uniform' is a seeded permutation of "
        "the eligible rows. 'progenmech_stratified' reproduces their design -- "
        "equal counts per DMS_score_bin with a 256-variant train split removed "
        "first -- which is what their 0.29 base was measured on, where "
        "ProteinGym's own benchmark records 0.497 for the same checkpoint",
    )
    parser.add_argument(
        "--gate-margin",
        type=float,
        default=0.0,
        help="how far above the free baseline the model must sit for the gate to "
        "pass. A declared convention recorded in the artefact",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    directions = {"both": (False, True), "n_to_c": (False,), "c_to_n": (True,)}[args.direction]

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "progenmech_reported": PROGENMECH_REPORTED,
    }

    print("[cohort] drawing assays")
    assays: list[Assay] = []
    for index, name in enumerate(args.assays):
        stratified = args.sampling == "progenmech_stratified"
        assay = load_assay(
            name,
            n=args.variants,
            seed=args.seed + index,
            include_multi=not args.single_mutants_only,
            stratify_by_score_bin=stratified,
            train_holdout=PROGENMECH_TRAIN_SEQUENCES if stratified else 0,
        )
        print(
            f"  {name:36s} n={len(assay.sequences):5d} of {assay.n_eligible:7d} "
            f"eligible  wt={len(assay.wildtype)} aa"
        )
        assays.append(assay)

    print("[loader] loading ProGen3-112M and self-checking the conversion")
    load_kwargs: dict[str, Any] = {"device": args.device, "dtype": getattr(torch, args.dtype)}
    if args.checkpoint is not None:
        load_kwargs["checkpoint"] = args.checkpoint
    pg = load_progen3(**load_kwargs)
    payload["gates"] = {"loader": self_check(pg)}
    print(f"  self-check NLL {payload['gates']['loader']['nll']:.4f} PASS")

    payload["condition"] = {
        "predictors": {
            "model": "summed log-likelihood of the variant sequence under "
            "ProGen3-112M, averaged over the scored directions",
            "blosum62": "summed BLOSUM62 score of the variant's substitutions; "
            "no model, computable from the mutation string alone",
            "shuffled_label_control": "the model's own predictions against "
            "permuted fitness labels",
        },
        "directions": args.direction,
        "variant_draw": "seeded permutation of the eligible rows, never a prefix "
        "(Appendix B rule 1: a ProteinGym CSV is ordered by position)",
        "assay_provenance": "the eight assay names inside ProGenMech's released "
        "ProGenMechData/functions.tar.gz circuit-discovery outputs",
    }

    print("[score] scoring assays")
    scored: list[dict[str, Any]] = []
    for index, assay in enumerate(assays):
        record = score_assay(
            pg,
            assay,
            batch_size=args.batch_size,
            directions=directions,
            replicates=args.bootstrap,
            seed=args.seed + 1000 * (index + 1),
        )
        print(
            f"  {assay.name:36s} model {record['model']['spearman']:+.4f}  "
            f"blosum62 {record['blosum62']['spearman']:+.4f}  "
            f"diff {record['model_minus_blosum62']:+.4f}"
        )
        scored.append(record)

    payload["assays"] = scored
    payload["gates"]["free_baseline"] = free_baseline_gate(scored, gate=args.gate_margin)
    payload["verdict"] = payload["gates"]["free_baseline"]["verdict"]

    write_json(args.out / "fitness_recovery.json", payload)
    gate = payload["gates"]["free_baseline"]
    print()
    print(f"[gate] model  {gate['model_mean_spearman']:+.4f} +/- {gate['model_sd_spearman']:.4f}")
    print(
        f"[gate] blosum {gate['blosum62_mean_spearman']:+.4f} "
        f"+/- {gate['blosum62_sd_spearman']:.4f}"
    )
    print(
        f"[gate] paired difference {gate['paired_difference']['mean']:+.4f} "
        f"CI {gate['paired_difference']['interval']}  "
        f"wins {gate['assays_where_model_wins']}/{gate['assays_total']}  "
        f"sign p={gate['sign_test_two_sided_p']:.4f}"
    )
    print(f"[verdict] {payload['verdict']}")


if __name__ == "__main__":
    main()
