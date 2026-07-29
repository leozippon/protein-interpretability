#!/usr/bin/env python3
"""Headline table: how much next-token computation each sublayer pathway carries.

The transfer hypothesis is that transcoder and cross-layer-transcoder circuit
tracing decomposes the MLP sublayer, and that in protein decoders the MLP
pathway carries a much smaller share of next-token computation than it does in
text decoders. That is a statement about the models, not about any dictionary,
so it is measured by ablating the pathway and reading what the model loses.

For each arm this script sweeps depth for the anchored scopes, reports every
scope's cross-entropy cost, KL from the clean predictive distribution and share
of the cohort's context information, and attaches sequence-cluster bootstrap
intervals. A seed is a cohort subsample, which is the only randomness in the
measurement and is also what the ``cohort_mean`` baseline is estimated on.

All protein arms share one corpus, defaulting to the EC-labelled Swiss-Prot set
the production P0-2b qualification evaluated on. Scoring some protein arms on
plain Swiss-Prot and others on the EC-labelled set moves clean cross-entropy by
more than a nat per token and makes the arms incomparable, so the corpus is a
declared parameter and is recorded in every output.

The context-free baseline that normalises every share is estimated on a held-out
corpus by default. The in-cohort plug-in estimator is biased low, by roughly a
nat per token on the 50k-vocabulary arms, which inflates the share it normalises;
it remains available as an explicit opt-in and never as a fallback.

One JSON per arm is written under ``results/transfer/pathway_budget/``.
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# See 08_lens_family.py for why the stage directory is added explicitly.
_STAGE_DIR = str(Path(__file__).resolve().parent)
if _STAGE_DIR not in sys.path:
    sys.path.insert(0, _STAGE_DIR)

from panel_contract import arm_can_run, stage_arms, stage_contract_record  # noqa: E402
from src.transfer.scoring import analysis_layer  # noqa: E402
from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    PANEL,
    Arm,
    Cohort,
    load_arm,
    protein_cohort,
    symbols_per_token,
    text_cohort,
)
from src.transfer.pathways import (  # noqa: E402
    BASELINE_KINDS,
    P0_2B_MINIMUM_CE_DELTA_NATS,
    P0_2B_MINIMUM_KL_NATS,
    PROTEIN_COHORT_SOURCES,
    TEXT_COHORT_SOURCE,
    UNIGRAM_ESTIMATORS,
    AblationScope,
    Target,
    assert_disjoint,
    attn_all,
    attn_single,
    attn_window,
    build_baseline,
    cohort_composition,
    cohort_target_token_counts,
    held_out_cohort,
    measure_pathways,
    mlp_all,
    mlp_and_attn_all,
    mlp_single,
    mlp_window,
    pathway_cluster_bootstrap,
    pathway_metrics,
    prepare_batches,
    resid_block,
    scope_record,
    subsample_cohort,
    unigram_baseline,
)

SCHEMA_VERSION = "r2_transfer_pathway_budget_v1"
DEFAULT_DEPTHS = (0.15, 0.33, 0.50, 0.67, 0.85)
DEFAULT_SEEDS = (20260728, 20260729, 20260730)


def build_scopes(
    n_layer: int, depths: tuple[float, ...], window: int
) -> tuple[list[AblationScope], dict[str, list[float]]]:
    """Depth-swept anchored scopes plus the whole-pathway scopes.

    Two depth fractions can round to the same layer in a shallow model, so
    scopes are de-duplicated by name and every depth fraction that produced a
    scope is recorded against it. Measuring the same layer twice would inflate
    the seed count without adding evidence.
    """

    if window < 1:
        raise ValueError("window width must be positive")
    if not depths:
        raise ValueError("at least one depth fraction is required")
    scopes: dict[str, AblationScope] = {}
    depths_by_name: dict[str, list[float]] = {}

    def register(scope: AblationScope, depth: float | None) -> None:
        if scope.name not in scopes:
            scopes[scope.name] = scope
            depths_by_name[scope.name] = []
        if depth is not None and depth not in depths_by_name[scope.name]:
            depths_by_name[scope.name].append(depth)

    for depth in depths:
        layer = analysis_layer(n_layer, depth)
        register(mlp_single(layer), depth)
        register(attn_single(layer), depth)
        register(mlp_window(layer, window), depth)
        register(attn_window(layer, window), depth)
        register(resid_block(layer), depth)
    for scope in (mlp_all(), attn_all(), mlp_and_attn_all()):
        register(scope, None)
    return list(scopes.values()), depths_by_name


def parse_entropy_overrides(values: list[str]) -> dict[str, float]:
    """Parse ``arm=nats`` overrides for the context-free baseline."""

    overrides: dict[str, float] = {}
    for value in values:
        if value.count("=") != 1:
            raise ValueError(f"expected arm=nats, got {value!r}")
        name, raw = value.split("=")
        if name not in PANEL:
            raise ValueError(f"unknown arm {name!r} in unigram-entropy override")
        if name in overrides:
            raise ValueError(f"duplicate unigram-entropy override for {name!r}")
        entropy = float(raw)
        if not entropy > 0:
            raise ValueError(f"unigram entropy for {name!r} must be positive")
        overrides[name] = entropy
    return overrides


def default_pathway_arms() -> list[str]:
    return stage_arms("pathway_budget")[0]


def cohort_pool(arm_names: list[str], args: argparse.Namespace) -> dict[str, Cohort]:
    """One pool per modality, so arms of the same modality share their cohort.

    Every protein arm is drawn from the same corpus, including the unconditional
    ones. ``Cohort.input_strings`` then renders that corpus in each arm's native
    format, so ZymCTRL gets its EC conditioning tag while ProtGPT2 and ProGen2
    get the bare sequence from the same records. Giving different protein arms
    different corpora would make their cross-entropies incomparable.
    """

    if args.protein_source not in PROTEIN_COHORT_SOURCES:
        raise ValueError(f"unknown protein cohort source {args.protein_source!r}")
    modalities = {PANEL[name].modality for name in arm_names}
    pools: dict[str, Cohort] = {}
    if "text" in modalities:
        pools["text"] = text_cohort(
            args.pool_size, min_chars=args.text_min_chars, name=TEXT_COHORT_SOURCE
        )
    if "protein" in modalities:
        pools["protein"] = protein_cohort(
            args.pool_size,
            args.res_min,
            args.res_max,
            name=args.protein_source,
            with_ec=args.protein_source == "ec_labelled_swissprot",
        )
    return pools


def cohort_source(arm: Arm, args: argparse.Namespace) -> str:
    return TEXT_COHORT_SOURCE if arm.modality == "text" else args.protein_source


def reference_pool(
    pools: dict[str, Cohort], args: argparse.Namespace
) -> dict[str, tuple[Cohort, dict[str, int]]]:
    """Held-out corpora for the context-free baseline, one per modality.

    Drawn by skipping exactly the measurement pool, so the reference is the next
    block of eligible records. That is not sufficient on its own -- the corpora
    repeat sequences under different accessions -- so records shared with the
    measurement pool are removed by content and the removal is counted, then
    disjointness is asserted. Construction-by-argument is exactly what the
    plug-in estimator's bias was hiding behind.
    """

    if args.unigram_estimator != "disjoint":
        return {}
    if args.unigram_reference_size < 1:
        raise ValueError("--unigram-reference-size must be positive")
    references: dict[str, Cohort] = {}
    if "text" in pools:
        references["text"] = text_cohort(
            args.unigram_reference_size,
            min_chars=args.text_min_chars,
            skip=args.pool_size,
            name=f"{TEXT_COHORT_SOURCE}_reference",
        )
    if "protein" in pools:
        references["protein"] = protein_cohort(
            args.unigram_reference_size,
            args.res_min,
            args.res_max,
            skip=args.pool_size,
            name=f"{args.protein_source}_reference",
            with_ec=args.protein_source == "ec_labelled_swissprot",
        )
    deduplicated: dict[str, tuple[Cohort, dict[str, int]]] = {}
    for modality, reference in references.items():
        cohort, counts = held_out_cohort(reference, pools[modality])
        assert_disjoint(pools[modality], cohort)
        deduplicated[modality] = (cohort, counts)
    return deduplicated


def seed_requirement(
    scope_rows: list[dict[str, Any]], *, seeds: int, guard_nats: float, fraction: float
) -> dict[str, Any]:
    """How many cohort seeds the production campaign needs, from observed spread.

    The single-layer scopes are the binding case: their footprints sit within a
    factor of a few of the guard, so a seed-mean whose standard error is a
    sizeable fraction of the guard can flip an attainability verdict. The
    requirement is the seed count that pulls that standard error below
    ``fraction`` of the guard for the worst single-layer scope observed here.

    With few seeds the between-seed standard deviation is itself barely
    estimated, so the number is a floor for planning, not a precise power
    calculation, and the sample size it came from is reported with it.
    """

    if seeds < 2:
        raise ValueError("a seed requirement needs at least two seeds of spread")
    if not 0.0 < fraction < 1.0 or guard_nats <= 0.0:
        raise ValueError("fraction must lie in (0, 1) and the guard must be positive")
    target_error = fraction * guard_nats
    worst: dict[str, Any] | None = None
    for row in scope_rows:
        if row["family"] != "single" or row["submodules"] == ["block"]:
            continue
        values = [entry["metrics"]["ce_delta_nats"] for entry in row["seeds"]]
        deviation = float(np.std(values, ddof=1))
        required = max(2, math.ceil((deviation / target_error) ** 2))
        if worst is None or required > worst["required_seeds"]:
            worst = {
                "scope": row["name"],
                "between_seed_sd_nats": deviation,
                "observed_range_nats": max(values) - min(values),
                "required_seeds": required,
            }
    if worst is None:
        raise ValueError("no single-layer scope was measured; cannot size the campaign")
    return {
        "rule": (
            "seeds such that the standard error of the seed-mean CE delta falls below "
            f"{fraction} of the {guard_nats} nats/token guard, for the worst single-layer "
            "sublayer scope"
        ),
        "target_standard_error_nats": target_error,
        "estimated_from_seeds": seeds,
        "binding_scope": worst["scope"],
        "binding_between_seed_sd_nats": worst["between_seed_sd_nats"],
        "binding_observed_range_nats": worst["observed_range_nats"],
        "recommended_minimum_seeds": worst["required_seeds"],
        "caveat": (
            f"the between-seed standard deviation is estimated from {seeds} seeds and is "
            "itself uncertain; treat this as a planning floor, not a power calculation"
        ),
    }


def measure_arm(
    arm: Arm,
    pool: Cohort,
    reference: tuple[Cohort, dict[str, int]] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Every scope on every seed for one arm."""

    scopes, depths_by_name = build_scopes(arm.n_layer, tuple(args.depths), args.window)
    entropy_override = parse_entropy_overrides(args.unigram_entropy).get(arm.name)
    device = torch.device(arm.device)
    torch.cuda.reset_peak_memory_stats(device)

    # The held-out corpus is fixed across seeds and needs tokenisation only, so
    # its counts are computed once per arm rather than once per seed.
    reference_counts = None
    reference_record = None
    if reference is not None:
        reference_cohort, reference_dedup = reference
        reference_counts = cohort_target_token_counts(
            arm, reference_cohort, max_len=args.max_len
        )
        reference_record = {
            **cohort_composition(reference_cohort, source=cohort_source(arm, args)),
            **reference_dedup,
        }

    seed_records: list[dict[str, Any]] = []
    per_scope_seeds: dict[str, list[dict[str, Any]]] = {scope.name: [] for scope in scopes}
    targets_by_scope: dict[str, tuple[Target, ...]] = {}
    measured_symbols_per_token: float | None = None

    for seed in args.seeds:
        cohort = subsample_cohort(pool, args.n_seq, seed)
        batches = prepare_batches(
            arm, cohort, max_len=args.max_len, batch_size=args.batch_size
        )
        if measured_symbols_per_token is None:
            measured_symbols_per_token = symbols_per_token(
                arm, cohort.input_strings(arm), args.max_len
            )
        all_targets = sorted(
            {target for scope in scopes for target in scope.resolve(arm.n_layer)}
        )
        bank = build_baseline(
            arm, batches, all_targets, kind=args.baseline, cohort_digest=cohort.digest
        )
        run = measure_pathways(arm, batches, scopes, bank)
        targets_by_scope = run.targets_by_scope
        baseline = unigram_baseline(
            arm,
            estimator=args.unigram_estimator,
            target_counts=run.target_token_counts,
            reference_counts=reference_counts,
            reference=reference_record,
            override_nats=entropy_override,
        )
        unigram = baseline["nats"]

        for scope in scopes:
            rows = run.rows_by_scope[scope.name]
            metrics = pathway_metrics(
                rows,
                unigram_entropy_nats=unigram,
                minimum_ce_delta_nats=args.minimum_ce_delta_nats,
                minimum_kl_nats=args.minimum_kl_nats,
            )
            bootstrap = pathway_cluster_bootstrap(
                rows,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed + seed,
                unigram_entropy_nats=unigram,
                minimum_ce_delta_nats=args.minimum_ce_delta_nats,
                minimum_kl_nats=args.minimum_kl_nats,
            )
            per_scope_seeds[scope.name].append(
                {"seed": seed, "metrics": metrics, "cluster_bootstrap": bootstrap}
            )
            share = metrics["share_of_context_information"]
            if metrics["measurable"]:
                verdict = "measurable"
            elif not metrics["context_information_valid"]:
                verdict = "OFF-DISTRIBUTION"
            else:
                verdict = "GUARD-FAIL"
            print(
                f"  [{arm.name} seed {seed}] {scope.name:26s} "
                f"dCE={metrics['ce_delta_nats']:+.4f} "
                f"KL={metrics['kl_clean_to_ablated_nats']:.4f} "
                f"share={'      none' if share is None else f'{share:+.4f}'} "
                f"{verdict}",
                flush=True,
            )

        seed_records.append(
            {
                "seed": seed,
                "cohort": cohort_composition(cohort, source=cohort_source(arm, args)),
                "scored_tokens": run.scored_tokens,
                "scored_sequences": run.scored_sequences,
                "unigram_entropy_nats": unigram,
                "unigram_entropy_source": baseline["source"],
                "unigram_baseline": baseline,
                "baseline_provenance": bank.provenance,
            }
        )
        del batches, bank, run
        gc.collect()
        torch.cuda.empty_cache()

    scope_rows: list[dict[str, Any]] = []
    for scope in scopes:
        seeds = per_scope_seeds[scope.name]
        deltas = [entry["metrics"]["ce_delta_nats"] for entry in seeds]
        kls = [entry["metrics"]["kl_clean_to_ablated_nats"] for entry in seeds]
        # A share exists only for seeds on which the arm is on-distribution, so
        # the summary must not average across a mixture of the two.
        shares = [
            entry["metrics"]["share_of_context_information"]
            for entry in seeds
            if entry["metrics"]["context_information_valid"]
        ]
        scope_rows.append(
            {
                **scope_record(scope, targets_by_scope[scope.name]),
                "depth_fractions": depths_by_name[scope.name],
                "seeds": seeds,
                "across_seeds": {
                    "ce_delta_nats_mean": sum(deltas) / len(deltas),
                    "ce_delta_nats_min": min(deltas),
                    "ce_delta_nats_max": max(deltas),
                    "kl_clean_to_ablated_nats_mean": sum(kls) / len(kls),
                    "share_seeds": len(shares),
                    "share_of_context_information_mean": (
                        sum(shares) / len(shares) if shares else None
                    ),
                    "share_of_context_information_min": min(shares) if shares else None,
                    "share_of_context_information_max": max(shares) if shares else None,
                    "context_information_valid_every_seed": all(
                        entry["metrics"]["context_information_valid"] for entry in seeds
                    ),
                    "measurable_every_seed": all(
                        entry["metrics"]["measurable"] for entry in seeds
                    ),
                    "measurable_any_seed": any(
                        entry["metrics"]["measurable"] for entry in seeds
                    ),
                },
            }
        )

    if measured_symbols_per_token is None:
        raise RuntimeError(f"{arm.name}: no seed produced a cohort")
    return {
        "arm": {
            "name": arm.name,
            "modality": arm.modality,
            "path": str(arm.spec.path),
            "n_layer": arm.n_layer,
            "d_model": arm.d_model,
            "tokenisation": arm.spec.tokenisation,
            "input_format": arm.spec.input_format,
            "source": arm.spec.source,
            "dtype": arm.dtype,
            "vocab_size": int(arm.model.config.vocab_size),
            "symbols_per_token": measured_symbols_per_token,
        },
        "cohort_pool": cohort_composition(pool, source=cohort_source(arm, args)),
        "unigram_reference_cohort": reference_record,
        "seeds": seed_records,
        "scopes": scope_rows,
        "seed_requirement": seed_requirement(
            scope_rows,
            seeds=len(args.seeds),
            guard_nats=args.minimum_ce_delta_nats,
            fraction=args.seed_target_error_fraction,
        ),
        "resources": {
            "peak_accelerator_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "peak_accelerator_memory_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Default: the campaign arms this stage can actually measure, not sorted(PANEL).
    # sorted(PANEL) admits the three ByGPT5 rungs, which carry no `pathway`
    # capability, so a bare invocation scheduled three arms that could only fail
    # -- and it failed inside measure_pathways, after the checkpoint was loaded.
    parser.add_argument(
        "--arms", nargs="+", default=default_pathway_arms(), choices=sorted(PANEL)
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--n-seq", type=int, default=200)
    # The protein pool is taken in file order from an EC-grouped source, so it
    # must be far larger than the per-seed draw or every seed sees the same few
    # families; see pathways.cohort_composition.
    parser.add_argument("--pool-size", type=int, default=4000)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--depths", nargs="+", type=float, default=list(DEFAULT_DEPTHS))
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--baseline", default="cohort_mean", choices=list(BASELINE_KINDS))
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    parser.add_argument(
        "--minimum-ce-delta-nats", type=float, default=P0_2B_MINIMUM_CE_DELTA_NATS
    )
    parser.add_argument("--minimum-kl-nats", type=float, default=P0_2B_MINIMUM_KL_NATS)
    parser.add_argument("--res-min", type=int, default=64)
    parser.add_argument("--res-max", type=int, default=246)
    parser.add_argument("--text-min-chars", type=int, default=800)
    # The production P0-2b qualification evaluated on the EC-labelled corpus, so
    # that is the default: a plain-Swiss-Prot run cannot be compared with it.
    parser.add_argument(
        "--protein-source",
        default="ec_labelled_swissprot",
        choices=list(PROTEIN_COHORT_SOURCES),
    )
    parser.add_argument(
        "--unigram-estimator", default="disjoint", choices=list(UNIGRAM_ESTIMATORS)
    )
    parser.add_argument("--unigram-reference-size", type=int, default=4000)
    parser.add_argument(
        "--unigram-entropy",
        action="append",
        default=[],
        metavar="ARM=NATS",
        help="context-free baseline supplied externally, overriding the estimator",
    )
    parser.add_argument("--seed-target-error-fraction", type=float, default=0.2)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "results" / "transfer" / "pathway_budget",
    )
    args = parser.parse_args()

    if len(set(args.arms)) != len(args.arms):
        raise ValueError("--arms repeats an arm")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds repeats a seed")
    if args.n_seq > args.pool_size:
        raise ValueError("--n-seq cannot exceed --pool-size")
    if any(not 0.0 <= depth <= 1.0 for depth in args.depths):
        raise ValueError("--depths must lie in [0, 1]")
    parse_entropy_overrides(args.unigram_entropy)

    if len(args.seeds) < 2:
        raise ValueError("--seeds must name at least two cohort seeds to size the campaign")
    # seed_requirement's own copies of these two checks run at the very end of
    # measure_arm, after the checkpoint is loaded and every seed x every scope has
    # been ablated. Both read only the command line.
    if not 0.0 < args.seed_target_error_fraction < 1.0:
        raise ValueError("--seed-target-error-fraction must lie strictly in (0, 1)")
    if args.minimum_ce_delta_nats <= 0.0:
        raise ValueError("--minimum-ce-delta-nats must be positive; it is a guard, not a floor")
    if args.minimum_kl_nats <= 0.0:
        raise ValueError("--minimum-kl-nats must be positive")
    # build_scopes checks this inside measure_arm, i.e. after load_arm.
    if args.window < 1:
        raise ValueError("--window must be positive")
    if not args.depths:
        raise ValueError("--depths must name at least one fraction")
    refused = {
        arm: verdict.reason
        for arm in args.arms
        if not (verdict := arm_can_run("pathway_budget", arm)).can_run
    }
    if refused:
        # --arms defaults to the pathway-capable campaign panel, but an explicit
        # list bypasses that. Without this, an arm with no `pathway` capability
        # reaches Arm.mlp()/Arm.attention() deep inside measure_pathways -- after
        # the checkpoint is on the GPU and the baseline bank has been built.
        raise ValueError(
            f"02_pathway_budget.py cannot measure {sorted(refused)}: {refused}"
        )

    pools = cohort_pool(list(args.arms), args)
    references = reference_pool(pools, args)
    output_root = args.output_root.resolve()
    started = datetime.now(timezone.utc).isoformat()

    for name in args.arms:
        arm = load_arm(name, device=args.device, dtype=args.dtype)
        payload = measure_arm(
            arm, pools[arm.modality], references.get(arm.modality), args
        )
        write_json(
            output_root / f"{name}.json",
            {
                "schema_version": SCHEMA_VERSION,
                "started_at_utc": started,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "runner_sha256": sha256_file(Path(__file__)),
                "pathways_module_sha256": sha256_file(
                    REPO_ROOT / "src" / "transfer" / "pathways.py"
                ),
                "arms_module_sha256": sha256_file(REPO_ROOT / "src" / "transfer" / "arms.py"),
                "configuration": {
                    "device": args.device,
                    "dtype": args.dtype,
                    "sequences_per_seed": args.n_seq,
                    "pool_size": args.pool_size,
                    "max_len": args.max_len,
                    "batch_size": args.batch_size,
                    "seeds": list(args.seeds),
                    "depth_fractions": list(args.depths),
                    "window": args.window,
                    "ablation_baseline": args.baseline,
                    "protein_cohort_source": args.protein_source,
                    "cohort_source": cohort_source(arm, args),
                    "unigram_estimator": args.unigram_estimator,
                    "unigram_reference_size": args.unigram_reference_size,
                    "residue_length_range": [args.res_min, args.res_max],
                    "text_min_chars": args.text_min_chars,
                    "scored_positions": (
                        "attention_mask_valid_and_next_token_valid_no_modality_filter"
                    ),
                },
                "thresholds": {
                    "minimum_ce_delta_nats": args.minimum_ce_delta_nats,
                    "minimum_kl_nats": args.minimum_kl_nats,
                    "provenance": "production_P0_2b_denominator_guards",
                },
                "bootstrap": {
                    "cluster_unit": "sequence",
                    "samples": args.bootstrap_samples,
                    "base_seed": args.bootstrap_seed,
                },
                "stage_contract": stage_contract_record("pathway_budget", list(args.arms)),
                **payload,
            },
        )
        print(f"wrote {output_root / f'{name}.json'}", flush=True)
        del arm
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
