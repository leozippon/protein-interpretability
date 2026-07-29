#!/usr/bin/env python3
"""Which causal estimands are large enough to be measured at all.

The production P0-2b qualification put an 80%-recovery gate on the ablation of a
single MLP output at relative depth 0.5. That estimand's causal footprint on
GPT-2-large is about 0.02 nats/token, below the 0.05 guard the gate itself
required, and smaller than on some of the protein models the gate was scoring.
The gate was therefore mis-specified *relative to its own control*: it failed on
a text decoder that is in-distribution and richly contextual, so its failure on
the protein arms could not be read as evidence about the dictionaries. That is a
sharper and more useful claim than "unattainable for any model", which is not
what the measurement shows -- a protein arm can clear an estimand the text
control cannot -- and the output states which of the two situations obtains.

This script makes that check routine. It sweeps pathway x depth x ablation
baseline, reports which estimands clear both P0-2b guards on each arm, and
derives a recommended *powered* estimand: one whose 95% sequence-cluster
interval sits entirely above both guards on the text arm, and then on the
protein arms. Estimand identifiers are depth-relative rather than layer-
absolute, so a 36-layer and a 27-layer model can be compared on the same row.

Every protein arm is scored on one shared corpus, defaulting to the EC-labelled
Swiss-Prot set the production qualification used, because a cross-arm verdict
drawn from per-arm corpora is not a verdict about the estimand. An arm that is
off-distribution on that corpus has no context information for an intervention
to remove; it is named and excluded from the panel verdict rather than scored.

``measure`` writes one JSON per arm; ``recommend`` turns those into the panel
attainability verdict. Both live under ``results/transfer/estimand_power/``.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# See 08_lens_family.py for why the stage directory is added explicitly.
_STAGE_DIR = str(Path(__file__).resolve().parent)
if _STAGE_DIR not in sys.path:
    sys.path.insert(0, _STAGE_DIR)

from src.transfer.scoring import analysis_layer  # noqa: E402
from src.transfer.io import sha256_file, write_json  # noqa: E402
from panel_contract import arm_can_run, stage_arms, stage_contract_record  # noqa: E402
from src.transfer.arms import PANEL, Arm, Cohort, load_arm, protein_cohort, text_cohort  # noqa: E402
from src.transfer.pathways import (  # noqa: E402
    BASELINE_KINDS,
    P0_2B_MINIMUM_CE_DELTA_NATS,
    P0_2B_MINIMUM_KL_NATS,
    PROTEIN_COHORT_SOURCES,
    TEXT_COHORT_SOURCE,
    UNIGRAM_ESTIMATORS,
    AblationScope,
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

ARM_SCHEMA_VERSION = "r2_transfer_estimand_power_v1"
PANEL_SCHEMA_VERSION = "r2_transfer_estimand_power_recommendation_v1"
DEFAULT_DEPTHS = (0.15, 0.33, 0.50, 0.67, 0.85)
DEFAULT_WIDTHS = (4, 8)

#: The estimand the production qualification actually gated on.
P0_2B_ESTIMAND_ID = "mlp_single@d0.50@cohort_mean"

PATHWAYS = ("attn", "block", "mlp", "mlp_and_attn")


@dataclass(frozen=True)
class EstimandSpec:
    """A depth-relative, layer-count-independent identity for one estimand."""

    pathway: str
    family: str
    depth_fraction: float | None
    width: int | None
    baseline: str

    def __post_init__(self) -> None:
        if self.pathway not in PATHWAYS:
            raise ValueError(f"unknown pathway {self.pathway!r}")
        if self.baseline not in BASELINE_KINDS:
            raise ValueError(f"unknown ablation baseline {self.baseline!r}")
        if self.family == "all":
            if self.depth_fraction is not None or self.width is not None:
                raise ValueError("a whole-pathway estimand has neither depth nor width")
        elif self.family == "single":
            if self.depth_fraction is None or self.width is not None:
                raise ValueError("a single-layer estimand needs a depth and no width")
        elif self.family == "window":
            if self.depth_fraction is None or self.width is None:
                raise ValueError("a window estimand needs a depth and a width")
        else:
            raise ValueError(f"unknown estimand family {self.family!r}")
        if self.pathway == "block" and self.family != "single":
            raise ValueError("the residual-block estimand is defined at a single depth")
        if self.pathway == "mlp_and_attn" and self.family != "all":
            raise ValueError("the joint-pathway estimand is defined over all layers")

    @property
    def estimand_id(self) -> str:
        family = self.family if self.width is None else f"{self.family}{self.width}"
        depth = "global" if self.depth_fraction is None else f"d{self.depth_fraction:.2f}"
        return f"{self.pathway}_{family}@{depth}@{self.baseline}"

    def scope(self, n_layer: int) -> AblationScope:
        if self.family == "all":
            if self.pathway == "mlp":
                return mlp_all()
            if self.pathway == "attn":
                return attn_all()
            return mlp_and_attn_all()
        layer = analysis_layer(n_layer, self.depth_fraction)
        if self.family == "single":
            if self.pathway == "mlp":
                return mlp_single(layer)
            if self.pathway == "attn":
                return attn_single(layer)
            return resid_block(layer)
        if self.pathway == "mlp":
            return mlp_window(layer, self.width)
        return attn_window(layer, self.width)


def build_estimands(
    depths: tuple[float, ...], widths: tuple[int, ...], baselines: tuple[str, ...]
) -> list[EstimandSpec]:
    """The full sweep, ordered so that the cheapest estimands come first."""

    if not depths or not widths or not baselines:
        raise ValueError("the sweep needs at least one depth, width and baseline")
    specs: list[EstimandSpec] = []
    for baseline in baselines:
        for depth in depths:
            for pathway in ("mlp", "attn", "block"):
                specs.append(EstimandSpec(pathway, "single", depth, None, baseline))
            for width in widths:
                for pathway in ("mlp", "attn"):
                    specs.append(EstimandSpec(pathway, "window", depth, width, baseline))
        for pathway in ("mlp", "attn", "mlp_and_attn"):
            specs.append(EstimandSpec(pathway, "all", None, None, baseline))
    identifiers = [spec.estimand_id for spec in specs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("the estimand sweep contains duplicate identifiers")
    return specs


def attainability_driver(
    text_powered: bool, protein_powered: dict[str, bool]
) -> tuple[str | None, str]:
    """Why an estimand is not attainable panel-wide, and a reader-facing sentence.

    This distinction is the whole diagnostic. An estimand that fails on the text
    positive control is mis-specified: its footprint is too small to gate on in a
    model that is known to be in-distribution and richly contextual, so nothing
    about the protein arms or their dictionaries can be inferred from the
    failure. An estimand that clears the control and fails only on protein arms
    is a genuine statement about those models.
    """

    failing = sorted(name for name, value in protein_powered.items() if not value)
    if text_powered and not failing:
        return None, "attainable on the text positive control and on every scored protein arm"
    if not text_powered and failing:
        return (
            "text_positive_control_and_protein_arms",
            "unattainable on the text positive control, so the estimand is mis-specified "
            f"and the failures on {', '.join(failing)} are not interpretable",
        )
    if not text_powered:
        return (
            "text_positive_control",
            "unattainable on the text positive control even though every scored protein "
            "arm clears it; the estimand is mis-specified relative to its own control",
        )
    return (
        "protein_arms",
        "attainable on the text positive control but not on "
        f"{', '.join(failing)}; this is a statement about those models",
    )


def powered(metrics: dict[str, Any], bootstrap: dict[str, Any]) -> bool:
    """Both guards cleared with the whole 95% cluster interval above threshold.

    Clearing a guard on the point estimate alone is not enough to hang a
    recovery-fraction gate on: an estimand whose interval straddles the guard
    will fail the guard on some cohorts and pass on others.
    """

    return bool(
        metrics["measurable"]
        and bootstrap["ce_delta_nats"]["q025"] >= metrics["minimum_ce_delta_nats"]
        and bootstrap["kl_clean_to_ablated_nats"]["q025"] >= metrics["minimum_kl_nats"]
    )


#: The text arm the panel verdict is anchored on. ``recommend`` requires exactly
#: one, by evidence discipline rule 1: a gate is applied to a protein arm only
#: after it has been shown attainable on *the* text positive control.
TEXT_POSITIVE_CONTROL = "gpt2-large"


def default_estimand_arms() -> list[str]:
    return stage_arms("estimand_power")[0]


def default_recommend_arms() -> list[str]:
    """The control-anchored subset: the text positive control plus every protein arm.

    Not every measured arm. ``measure`` runs for the whole panel and each arm
    keeps its own per-estimand ``powered`` flag on disk; what this scopes is which
    single arm the *panel verdict* is anchored on.
    """

    eligible = stage_arms("estimand_power")[0]
    if TEXT_POSITIVE_CONTROL not in eligible:
        raise AssertionError(
            f"the text positive control {TEXT_POSITIVE_CONTROL!r} is not eligible for "
            "estimand_power, so no panel verdict can be anchored"
        )
    return [TEXT_POSITIVE_CONTROL] + [
        name for name in eligible if PANEL[name].modality == "protein"
    ]


def cohort_pool(arm_names: list[str], args: argparse.Namespace) -> dict[str, Cohort]:
    """One pool per modality, so arms of the same modality share their cohort.

    Every protein arm is drawn from the same corpus and rendered by
    ``Cohort.input_strings`` in its own native format, so an attainability
    verdict compares arms on identical content rather than on whichever corpus
    each arm happened to be given.
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

    Records shared with the measurement pool are removed by content before use:
    the corpora repeat sequences under different accessions, so a later block of
    file-ordered records is not automatically held out.
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


def measure_arm(
    arm: Arm,
    pool: Cohort,
    reference: tuple[Cohort, dict[str, int]] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Every estimand in the sweep for one arm, grouped by ablation baseline."""

    specs = build_estimands(tuple(args.depths), tuple(args.widths), tuple(args.baselines))
    cohort = subsample_cohort(pool, args.n_seq, args.seed)
    batches = prepare_batches(arm, cohort, max_len=args.max_len, batch_size=args.batch_size)
    device = torch.device(arm.device)
    torch.cuda.reset_peak_memory_stats(device)

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

    records: list[dict[str, Any]] = []
    baseline_record: dict[str, Any] | None = None
    unigram: float | None = None
    scored_tokens = 0
    baseline_provenance: dict[str, Any] = {}

    for baseline in args.baselines:
        group = [spec for spec in specs if spec.baseline == baseline]
        scopes_by_id = {spec.estimand_id: spec.scope(arm.n_layer) for spec in group}
        # Two estimands can resolve to the same scope in a shallow model, and
        # measure_pathways rejects duplicate scope names, so the measurement is
        # keyed by scope and the results are fanned back out to the estimands.
        unique_scopes = list({scope.name: scope for scope in scopes_by_id.values()}.values())
        all_targets = sorted(
            {target for scope in unique_scopes for target in scope.resolve(arm.n_layer)}
        )
        bank = build_baseline(
            arm, batches, all_targets, kind=baseline, cohort_digest=cohort.digest
        )
        baseline_provenance[baseline] = bank.provenance
        run = measure_pathways(arm, batches, unique_scopes, bank)
        if unigram is None:
            baseline_record = unigram_baseline(
                arm,
                estimator=args.unigram_estimator,
                target_counts=run.target_token_counts,
                reference_counts=reference_counts,
                reference=reference_record,
                override_nats=args.unigram_entropy_nats,
            )
            unigram = baseline_record["nats"]
            scored_tokens = run.scored_tokens
        elif run.scored_tokens != scored_tokens:
            raise RuntimeError(f"{arm.name}: scored-token count changed between baselines")

        for spec in group:
            scope = scopes_by_id[spec.estimand_id]
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
                seed=args.bootstrap_seed,
                unigram_entropy_nats=unigram,
                minimum_ce_delta_nats=args.minimum_ce_delta_nats,
                minimum_kl_nats=args.minimum_kl_nats,
            )
            records.append(
                {
                    "estimand_id": spec.estimand_id,
                    "pathway": spec.pathway,
                    "family": spec.family,
                    "depth_fraction": spec.depth_fraction,
                    "width": spec.width,
                    "ablation_baseline": spec.baseline,
                    "scope": scope_record(scope, run.targets_by_scope[scope.name]),
                    "metrics": metrics,
                    "cluster_bootstrap": bootstrap,
                    "passes_ce_guard": metrics["passes_ce_guard"],
                    "passes_kl_guard": metrics["passes_kl_guard"],
                    "context_information_valid": metrics["context_information_valid"],
                    "clears_both_guards": metrics["measurable"],
                    "powered": powered(metrics, bootstrap),
                }
            )
            share = metrics["share_of_context_information"]
            if records[-1]["powered"]:
                verdict = "powered"
            elif not metrics["context_information_valid"]:
                verdict = "OFF-DISTRIBUTION"
            else:
                verdict = "UNDERPOWERED"
            print(
                f"  [{arm.name}] {spec.estimand_id:34s} "
                f"dCE={metrics['ce_delta_nats']:+.4f} "
                f"KL={metrics['kl_clean_to_ablated_nats']:.4f} "
                f"share={'      none' if share is None else f'{share:+.4f}'} "
                f"{verdict}",
                flush=True,
            )
        del bank, run
        gc.collect()
        torch.cuda.empty_cache()

    if unigram is None:
        raise RuntimeError(f"{arm.name}: the estimand sweep produced no measurement")
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
        },
        "cohort_pool": cohort_composition(pool, source=cohort_source(arm, args)),
        "cohort": {
            **cohort_composition(cohort, source=cohort_source(arm, args)),
            "scored_tokens": scored_tokens,
        },
        "unigram_reference_cohort": reference_record,
        "unigram_entropy_nats": unigram,
        "unigram_entropy_source": baseline_record["source"],
        "unigram_baseline": baseline_record,
        "baseline_provenance": baseline_provenance,
        "estimands": records,
        "resources": {
            "peak_accelerator_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "peak_accelerator_memory_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)
            ),
        },
    }


def measure(args: argparse.Namespace) -> None:
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("--arms repeats an arm")
    if len(set(args.baselines)) != len(args.baselines):
        raise ValueError("--baselines repeats a baseline")
    if args.n_seq > args.pool_size:
        raise ValueError("--n-seq cannot exceed --pool-size")
    if any(not 0.0 <= depth <= 1.0 for depth in args.depths):
        raise ValueError("--depths must lie in [0, 1]")
    # An estimand id is rendered as d{depth:.2f}, so two depths that round to the
    # same two decimals -- or a literal repeat -- collide. build_estimands catches
    # it, but build_estimands runs inside measure_arm, after load_arm. 08 checks
    # the same thing up front; this is the matching check.
    if len({f"{depth:.2f}" for depth in args.depths}) != len(args.depths):
        raise ValueError(
            "--depths contains entries that are equal to two decimal places; "
            "estimand ids are rendered as d{depth:.2f} and would collide"
        )
    if len(set(args.widths)) != len(args.widths):
        raise ValueError("--widths repeats a width")
    if args.unigram_entropy_nats is not None and not args.unigram_entropy_nats > 0:
        raise ValueError("--unigram-entropy-nats must be positive when supplied")
    refused = {
        arm: verdict.reason
        for arm in args.arms
        if not (verdict := arm_can_run("estimand_power", arm)).can_run
    }
    if refused:
        raise ValueError(
            f"03_estimand_power.py measure cannot serve {sorted(refused)}: {refused}"
        )
    pools = cohort_pool(list(args.arms), args)
    references = reference_pool(pools, args)
    output_root = args.output_root.resolve()
    started = datetime.now(timezone.utc).isoformat()
    for name in args.arms:
        arm = load_arm(name, device=args.device, dtype=args.dtype)
        payload = measure_arm(arm, pools[arm.modality], references.get(arm.modality), args)
        write_json(
            output_root / f"{name}.json",
            {
                "schema_version": ARM_SCHEMA_VERSION,
                "started_at_utc": started,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "runner_sha256": sha256_file(Path(__file__)),
                "pathways_module_sha256": sha256_file(
                    REPO_ROOT / "src" / "transfer" / "pathways.py"
                ),
                "arms_module_sha256": sha256_file(REPO_ROOT / "src" / "transfer" / "arms.py"),
                "stage_contract": stage_contract_record("estimand_power", list(args.arms)),
                "configuration": {
                    "device": args.device,
                    "dtype": args.dtype,
                    "sequences": args.n_seq,
                    "pool_size": args.pool_size,
                    "max_len": args.max_len,
                    "batch_size": args.batch_size,
                    "seed": args.seed,
                    "depth_fractions": list(args.depths),
                    "window_widths": list(args.widths),
                    "ablation_baselines": list(args.baselines),
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
                    "powered_rule": (
                        "point_estimate_clears_both_guards_and_bootstrap_q025_"
                        "of_ce_delta_and_kl_also_clear_them"
                    ),
                },
                "bootstrap": {
                    "cluster_unit": "sequence",
                    "samples": args.bootstrap_samples,
                    "seed": args.bootstrap_seed,
                },
                **payload,
            },
        )
        print(f"wrote {output_root / f'{name}.json'}", flush=True)
        del arm
        gc.collect()
        torch.cuda.empty_cache()


def load_arm_results(results_root: Path, arms: list[str]) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for name in arms:
        path = (results_root / f"{name}.json").resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing estimand-power result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != ARM_SCHEMA_VERSION:
            raise ValueError(f"unexpected schema in {path}")
        if payload["arm"]["name"] != name:
            raise ValueError(f"result {path} does not describe arm {name!r}")
        loaded[name] = payload
    if not loaded:
        raise ValueError("no arm results were requested")
    return loaded


def recommend(args: argparse.Namespace) -> None:
    """Turn per-arm attainability into one panel verdict and one recommendation."""

    results = load_arm_results(args.results_root.resolve(), list(args.arms))
    text_arms = [name for name, payload in results.items() if payload["arm"]["modality"] == "text"]
    if len(text_arms) != 1:
        raise ValueError(
            f"the attainability check needs exactly one text positive control, found {text_arms}"
        )
    text_arm = text_arms[0]
    protein_arms = sorted(set(results) - {text_arm})
    if not protein_arms:
        raise ValueError("the attainability check needs at least one protein arm")

    per_arm = {
        name: {row["estimand_id"]: row for row in payload["estimands"]}
        for name, payload in results.items()
    }
    shared = set.intersection(*(set(rows) for rows in per_arm.values()))
    if not shared:
        raise ValueError("the arms share no estimand identifiers")
    thresholds = {
        name: (
            payload["thresholds"]["minimum_ce_delta_nats"],
            payload["thresholds"]["minimum_kl_nats"],
        )
        for name, payload in results.items()
    }
    if len(set(thresholds.values())) != 1:
        raise ValueError(f"arms were scored against different guards: {thresholds}")
    protein_sources = {results[name]["cohort"]["source"] for name in protein_arms}
    if len(protein_sources) != 1:
        raise ValueError(f"protein arms were scored on different corpora: {protein_sources}")
    protein_digests = {results[name]["cohort"]["digest"] for name in protein_arms}
    if len(protein_digests) != 1:
        raise ValueError("protein arms were scored on different cohorts of the same corpus")

    # An arm whose clean cross-entropy is not below its own context-free baseline
    # is off-distribution on this corpus. It has no context information for any
    # intervention to remove, so requiring an estimand to be powered on it would
    # veto every estimand for a reason that has nothing to do with the estimand.
    # Such arms are named and excluded from the panel verdict, never scored.
    off_distribution: list[str] = []
    for name, indexed in per_arm.items():
        validity = {row["context_information_valid"] for row in indexed.values()}
        if len(validity) != 1:
            raise ValueError(f"{name}: context-information validity varies between estimands")
        if not validity.pop():
            off_distribution.append(name)
    if text_arm in off_distribution:
        raise ValueError(
            f"the text positive control {text_arm!r} is off-distribution on its own cohort"
        )
    scored_protein_arms = [name for name in protein_arms if name not in off_distribution]
    if not scored_protein_arms:
        raise ValueError("every protein arm is off-distribution on this corpus")

    rows: list[dict[str, Any]] = []
    for estimand_id in sorted(shared):
        text_row = per_arm[text_arm][estimand_id]
        arm_rows = {name: per_arm[name][estimand_id] for name in results}
        attainable_by_arm = {name: bool(row["powered"]) for name, row in arm_rows.items()}
        protein_attainable = {name: attainable_by_arm[name] for name in scored_protein_arms}
        driver, statement = attainability_driver(text_row["powered"], protein_attainable)
        rows.append(
            {
                "estimand_id": estimand_id,
                "pathway": text_row["pathway"],
                "family": text_row["family"],
                "depth_fraction": text_row["depth_fraction"],
                "width": text_row["width"],
                "ablation_baseline": text_row["ablation_baseline"],
                "text_control_targets": text_row["scope"]["n_targets"],
                "by_arm": {
                    name: {
                        "ce_clean_nats": row["metrics"]["ce_clean_nats"],
                        "ce_delta_nats": row["metrics"]["ce_delta_nats"],
                        "ce_delta_nats_q025": row["cluster_bootstrap"]["ce_delta_nats"]["q025"],
                        "kl_clean_to_ablated_nats": row["metrics"]["kl_clean_to_ablated_nats"],
                        "context_information_nats": row["metrics"]["context_information_nats"],
                        "context_information_valid": row["context_information_valid"],
                        "share_of_context_information": row["metrics"][
                            "share_of_context_information"
                        ],
                        "clears_both_guards": row["clears_both_guards"],
                        "powered": row["powered"],
                    }
                    for name, row in arm_rows.items()
                },
                "attainable_by_arm": attainable_by_arm,
                "attainable_on_scored_protein_arms": protein_attainable,
                "powered_on_text_control": text_row["powered"],
                "powered_on_every_scored_protein_arm": all(protein_attainable.values()),
                "powered_panel_wide": text_row["powered"] and all(protein_attainable.values()),
                "panel_wide_failure_driver": driver,
                "attainability_statement": statement,
            }
        )

    # The recommended estimand is the least one can ablate and still measure
    # everywhere. Ranking by text-arm footprint rather than by target count is
    # deliberate: a whole residual block is one target but the coarsest possible
    # claim, whereas a narrow MLP window is several targets and a far tighter one.
    candidates = [row for row in rows if row["powered_panel_wide"]]
    candidates.sort(
        key=lambda row: (
            row["by_arm"][text_arm]["ce_delta_nats"],
            row["text_control_targets"],
            row["estimand_id"],
        )
    )
    recommended = candidates[0]["estimand_id"] if candidates else None

    p0_2b = next((row for row in rows if row["estimand_id"] == P0_2B_ESTIMAND_ID), None)
    output = args.output.resolve()
    write_json(
        output,
        {
            "schema_version": PANEL_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "runner_sha256": sha256_file(Path(__file__)),
            "text_positive_control": text_arm,
            "protein_arms": protein_arms,
            "scored_protein_arms": scored_protein_arms,
            "off_distribution_arms": sorted(off_distribution),
            "protein_cohort_source": protein_sources.pop(),
            "arm_results": {
                name: {
                    "path": str((args.results_root / f"{name}.json").resolve()),
                    "sha256": sha256_file(args.results_root / f"{name}.json"),
                    "cohort_source": payload["cohort"]["source"],
                    "cohort_digest": payload["cohort"]["digest"],
                    "unigram_entropy_nats": payload["unigram_entropy_nats"],
                    "ce_clean_nats": payload["estimands"][0]["metrics"]["ce_clean_nats"],
                    "context_information_nats": payload["estimands"][0]["metrics"][
                        "context_information_nats"
                    ],
                    "context_information_valid": name not in off_distribution,
                }
                for name, payload in results.items()
            },
            "thresholds": {
                "minimum_ce_delta_nats": next(iter(thresholds.values()))[0],
                "minimum_kl_nats": next(iter(thresholds.values()))[1],
                "provenance": "production_P0_2b_denominator_guards",
            },
            "selection_rule": (
                "powered_on_the_text_positive_control_first_then_on_every_scored_protein_"
                "arm_then_smallest_text_control_ce_delta_then_fewest_ablated_submodules"
            ),
            "recommended_powered_estimand": recommended,
            "powered_panel_wide_estimands": [row["estimand_id"] for row in candidates],
            "powered_on_text_control_only": sorted(
                row["estimand_id"]
                for row in rows
                if row["powered_on_text_control"] and not row["powered_panel_wide"]
            ),
            "powered_on_no_arm": sorted(
                row["estimand_id"]
                for row in rows
                if not any(entry["powered"] for entry in row["by_arm"].values())
            ),
            "p0_2b_estimand": {
                "estimand_id": P0_2B_ESTIMAND_ID,
                "present_in_sweep": p0_2b is not None,
                "attainable_on_text_control": (
                    None if p0_2b is None else p0_2b["powered_on_text_control"]
                ),
                "attainable_panel_wide": None if p0_2b is None else p0_2b["powered_panel_wide"],
                "attainable_by_arm": None if p0_2b is None else p0_2b["attainable_by_arm"],
                "attainable_on_scored_protein_arms": (
                    None if p0_2b is None else p0_2b["attainable_on_scored_protein_arms"]
                ),
                "panel_wide_failure_driver": (
                    None if p0_2b is None else p0_2b["panel_wide_failure_driver"]
                ),
                "attainability_statement": (
                    None if p0_2b is None else p0_2b["attainability_statement"]
                ),
                "off_distribution_arms": sorted(off_distribution),
                "by_arm": None if p0_2b is None else p0_2b["by_arm"],
            },
            "mis_specified_estimands": sorted(
                row["estimand_id"]
                for row in rows
                if row["panel_wide_failure_driver"] == "text_positive_control"
            ),
            "protein_limited_estimands": sorted(
                row["estimand_id"]
                for row in rows
                if row["panel_wide_failure_driver"] == "protein_arms"
            ),
            "estimands": rows,
        },
    )
    print(f"wrote {output}", flush=True)
    print(f"recommended_powered_estimand={recommended}", flush=True)
    if p0_2b is not None:
        print(f"p0_2b_estimand_attainable_by_arm={p0_2b['attainable_by_arm']}", flush=True)
        print(f"p0_2b_failure_driver={p0_2b['panel_wide_failure_driver']}", flush=True)
        print(f"p0_2b_statement={p0_2b['attainability_statement']}", flush=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    default_root = REPO_ROOT / "results" / "transfer" / "estimand_power"
    subparsers = root.add_subparsers(dest="command", required=True)

    sweep = subparsers.add_parser("measure")
    # Default: the campaign arms this stage can measure. sorted(PANEL) admitted
    # three ByGPT5 rungs with no `pathway` capability, which could only fail, and
    # failed only after the checkpoint was loaded.
    sweep.add_argument(
        "--arms", nargs="+", default=default_estimand_arms(), choices=sorted(PANEL)
    )
    sweep.add_argument("--device", default="cuda:0")
    sweep.add_argument("--dtype", default="bfloat16")
    sweep.add_argument("--n-seq", type=int, default=200)
    # The protein pool is taken in file order from an EC-grouped source, so it
    # must be far larger than the per-seed draw or every seed sees the same few
    # families; see pathways.cohort_composition.
    sweep.add_argument("--pool-size", type=int, default=4000)
    sweep.add_argument("--max-len", type=int, default=256)
    sweep.add_argument("--batch-size", type=int, default=4)
    sweep.add_argument("--seed", type=int, default=20260728)
    sweep.add_argument("--depths", nargs="+", type=float, default=list(DEFAULT_DEPTHS))
    sweep.add_argument("--widths", nargs="+", type=int, default=list(DEFAULT_WIDTHS))
    sweep.add_argument(
        "--baselines", nargs="+", default=list(BASELINE_KINDS), choices=list(BASELINE_KINDS)
    )
    sweep.add_argument("--bootstrap-samples", type=int, default=1000)
    sweep.add_argument("--bootstrap-seed", type=int, default=20260728)
    sweep.add_argument(
        "--minimum-ce-delta-nats", type=float, default=P0_2B_MINIMUM_CE_DELTA_NATS
    )
    sweep.add_argument("--minimum-kl-nats", type=float, default=P0_2B_MINIMUM_KL_NATS)
    sweep.add_argument("--res-min", type=int, default=64)
    sweep.add_argument("--res-max", type=int, default=246)
    sweep.add_argument("--text-min-chars", type=int, default=800)
    # The production P0-2b qualification evaluated on the EC-labelled corpus, so
    # that is the default: a plain-Swiss-Prot run cannot be compared with it.
    sweep.add_argument(
        "--protein-source",
        default="ec_labelled_swissprot",
        choices=list(PROTEIN_COHORT_SOURCES),
    )
    sweep.add_argument(
        "--unigram-estimator", default="disjoint", choices=list(UNIGRAM_ESTIMATORS)
    )
    sweep.add_argument("--unigram-reference-size", type=int, default=4000)
    sweep.add_argument("--unigram-entropy-nats", type=float, default=None)
    sweep.add_argument("--output-root", type=Path, default=default_root)
    sweep.set_defaults(handler=measure)

    verdict = subparsers.add_parser("recommend")
    # `recommend` is control-anchored, not a survey: it raises unless exactly one
    # arm in its list is text. sorted(PANEL) is therefore not merely a wide
    # default, it is one that can never work -- and it lost a scheduled run of
    # EXP-R2-060 when the worker passed the whole panel. The default is now the
    # contract's own control-anchored set: the text control plus the protein arms.
    verdict.add_argument(
        "--arms", nargs="+", default=default_recommend_arms(), choices=sorted(PANEL)
    )
    verdict.add_argument("--results-root", type=Path, default=default_root)
    verdict.add_argument("--output", type=Path, default=default_root / "recommendation.json")
    verdict.set_defaults(handler=recommend)
    return root


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
