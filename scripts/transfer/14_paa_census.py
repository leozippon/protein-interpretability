#!/usr/bin/env python3
"""Go/no-go gate for a prediction-addressed-attention (PAA) census.

Gates, in the order they can kill the design:

``gate0``   cohort power on every arm's native cohort, held-out unigram
            baseline.  A protein arm below 0.5 nats/token of context-derived
            information cannot host a matched instance pool at all.
``census``  A1 candidate pool, A5 dissociation from induction, and the
            decoy-correction check on the *existing* induction census.
``causal``  A3 class non-emptiness and A4 causal magnitude on the text control.
``match``   A2 matching feasibility against a protein arm.
``query``   A6 query-source intervention.

The decisive uncertainty is A3/A4: copy suppression is documented for
GPT-2-small's L10H7, and its prevalence in gpt2-large is not established.  If no
sparse, causally confirmable suppressive head population exists there, the
design should be abandoned before any protein work is scheduled, and this script
is written to report that outcome as cleanly as the positive one.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.io import write_json  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    DEFAULT_CORPUS_DRAW_SEED,
    PANEL,
    REPO,
    Cohort,
    load_arm,
    protein_cohort,
    text_cohort,
)
from src.transfer.circuits import (  # noqa: E402
    RepeatProbe,
    fit_unigram,
    n_head,
    summarise_head_matrix,
    synthetic_repeat_probes,
)
from src.transfer.pathways import held_out_cohort  # noqa: E402
from src.transfer.prediction_addressed import (  # noqa: E402
    MINIMUM_DRAWS_IN_TAIL,
    InstancePool,
    build_instance_pool,
    cluster_bootstrap,
    coarsened_cells,
    cohort_power_held_out,
    corruption_effects,
    decoy_corrected_prefix_matching,
    flat_head_index,
    knockout_effects,
    paa_attention_scores,
    partial_spearman,
    query_source_intervention,
    scored_target_counts,
    tokenised_rows,
    top_set_jaccard,
)

SCHEMA_VERSION = "r2_transfer_paa_gate_v1"
DEFAULT_OUT = REPO / "results/transfer/paa_gate"

#: Gate 0 floor, in nats/token of context-derived information.
GATE0_MIN_NATS = 0.5

STAGES = ("gate0", "census", "causal", "match", "query")


# ---------------------------------------------------------------- gate 0


def gate0(args: argparse.Namespace) -> dict[str, Any]:
    """Context-derived information on each arm's native cohort."""

    rows: list[dict[str, Any]] = []
    draw_seed = args.cohort_draw_seed or None
    for name in args.gate0_arms:
        spec = PANEL[name]
        if spec.modality == "text":
            cohort = text_cohort(
                args.gate0_sequences,
                min_chars=args.text_min_chars,
                skip=args.cohort_skip,
                seed=draw_seed,
            )
            reference = text_cohort(
                args.reference_sequences,
                min_chars=args.text_min_chars,
                skip=args.cohort_skip + args.gate0_sequences,
                name="openwebtext_reference",
                seed=draw_seed,
            )
            max_len = args.text_max_len
        else:
            # ZymCTRL must be drawn from the EC-labelled corpus because its
            # native rendering needs the tag; for the unconditional arms the
            # corpus is a free choice and has previously moved ProGen2-medium's
            # clean cross-entropy by more than a nat, so it is a declared axis
            # rather than a default.
            with_ec = spec.input_format == "ec_conditioned" or args.protein_source == "ec"
            cohort = protein_cohort(
                args.gate0_sequences,
                args.protein_min_len,
                args.protein_max_len,
                skip=args.cohort_skip,
                with_ec=with_ec,
                seed=draw_seed,
            )
            reference = protein_cohort(
                args.reference_sequences,
                args.protein_min_len,
                args.protein_max_len,
                skip=args.cohort_skip + args.gate0_sequences,
                name="swissprot_reference",
                with_ec=with_ec,
                seed=draw_seed,
            )
            max_len = args.protein_max_len + 32
        reference, overlap = held_out_cohort(reference, cohort)
        arm = load_arm(name, device=args.device, dtype=args.dtype)
        # Every panel member has a 1024-position context. A band wider than that
        # is a legitimate request -- the point of sweeping the band is to find
        # where an arm has context to work with -- but the window must be
        # clamped rather than allowed to index past the position table.
        max_len = min(max_len, int(arm.model.config.n_positions))
        report = cohort_power_held_out(
            arm,
            cohort,
            reference,
            max_len=max_len,
            batch_size=args.gate0_batch,
            threshold_nats=args.gate0_threshold,
        )
        report["reference_overlap"] = overlap
        report["cohort_band"] = (
            [args.protein_min_len, args.protein_max_len]
            if spec.modality == "protein"
            else [args.text_min_chars, None]
        )
        report["protein_source"] = args.protein_source
        report["cohort_skip"] = int(args.cohort_skip)
        rows.append(report)
        print(
            f"  {name:16s} unigram(held-out) {report['unigram_held_out_nats']:7.4f}  "
            f"clean CE {report['clean_ce_nats']:7.4f}  "
            f"context {report['context_information_nats']:+7.4f}  "
            f"{report['verdict']}"
        )
        del arm
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "threshold_nats": float(args.gate0_threshold),
        "sequences": int(args.gate0_sequences),
        "arms": rows,
        "verdict": "PASS"
        if all(row["verdict"] == "PASS" for row in rows)
        else "PARTIAL"
        if any(row["verdict"] == "PASS" for row in rows)
        else "FAIL",
    }


# ------------------------------------------------------------ pool helpers


def build_cohorts(args: argparse.Namespace, name: str) -> tuple[Cohort, Cohort]:
    spec = PANEL[name]
    draw_seed = args.cohort_draw_seed or None
    if spec.modality == "text":
        cohort = text_cohort(
            args.census_sequences * 3,
            min_chars=args.census_text_min_chars,
            name="paa_text",
            seed=draw_seed,
        )
        reference = text_cohort(
            args.reference_sequences,
            min_chars=args.census_text_min_chars,
            skip=args.census_sequences * 3,
            name="paa_text_reference",
            seed=draw_seed,
        )
    else:
        with_ec = spec.input_format == "ec_conditioned"
        cohort = protein_cohort(
            args.census_sequences * 2,
            args.census_protein_min_len,
            args.protein_max_len,
            name="paa_protein",
            with_ec=with_ec,
            seed=draw_seed,
        )
        reference = protein_cohort(
            args.reference_sequences,
            args.census_protein_min_len,
            args.protein_max_len,
            skip=args.census_sequences * 2,
            name="paa_protein_reference",
            with_ec=with_ec,
            seed=draw_seed,
        )
    reference, _ = held_out_cohort(reference, cohort)
    return cohort, reference


def make_pool(
    arm,
    args: argparse.Namespace,
    cohort: Cohort,
    reference: Cohort,
    *,
    ban_depth: int | None = None,
):
    rows, low = tokenised_rows(arm, cohort.input_strings(arm), width=args.width)
    rows = rows[: args.census_sequences]
    if len(rows) < args.min_sequences:
        raise RuntimeError(
            f"{arm.name}: only {len(rows)} cohort records reached {args.width} tokens"
        )
    unigram_counts = scored_target_counts(
        arm, reference.input_strings(arm), max_len=args.protein_max_len + 32
    )
    pool = build_instance_pool(
        arm,
        rows,
        unigram_counts=unigram_counts,
        query_min=max(low + 1, args.query_min),
        top_k=args.top_k,
        candidate_depth=args.candidate_depth,
        min_confidence=args.min_confidence,
        n_decoys=args.n_decoys,
        seed=args.seed,
        batch_size=args.pool_batch,
        ban_depth=ban_depth,
    )
    return rows, pool, unigram_counts


def save_pool(path: Path, rows: list[list[int]], pool: InstancePool) -> None:
    """Persist **every** field of the pool, checked against the dataclass.

    ``content_low`` was absent here and from :func:`load_pool` and
    :func:`_subset`, so a reloaded pool silently took the dataclass default of 0
    and ``antecedent_sets`` searched from position 1 instead of from the first
    content token. On a protein arm that adds ProtGPT2's newline wrapping or
    ZymCTRL's ``<sep>`` to the key set the causal knockout removes -- exactly the
    failure ``antecedent_sets`` documents itself as preventing -- and attributes
    the resulting effect to the antecedent. It was latent only because the one
    pool ever reloaded is the text control's, whose ``content_low`` is 0 anyway.

    The field list is derived from the dataclass rather than written out, so a
    field added later cannot be dropped by this round trip without failing here.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    stored = {
        "rows": np.asarray(rows, dtype=np.int64),
        "arm": pool.arm,
        "cascade": json.dumps(pool.cascade),
        "content_low": np.asarray(pool.content_low, dtype=np.int64),
    }
    for field in fields(InstancePool):
        if field.name in stored:
            continue
        stored[field.name] = getattr(pool, field.name)
    missing = {field.name for field in fields(InstancePool)} - set(stored)
    if missing:
        raise RuntimeError(f"pool fields would not survive the round trip: {sorted(missing)}")
    np.savez_compressed(path, **stored)


def load_pool(path: Path) -> tuple[list[list[int]], InstancePool]:
    payload = np.load(path, allow_pickle=True)
    rows = [[int(value) for value in row] for row in payload["rows"]]
    absent = [
        field.name for field in fields(InstancePool) if field.name not in payload.files
    ]
    if absent:
        # A pool written before a field existed cannot be silently completed from
        # a default: the default reads as a measured value downstream.
        raise RuntimeError(
            f"{path}: pool is missing {sorted(absent)}; it predates the current "
            "InstancePool and must be rebuilt rather than defaulted"
        )
    pool = InstancePool(
        arm=str(payload["arm"]),
        sequence=payload["sequence"],
        query=payload["query"],
        antecedent=payload["antecedent"],
        predicted_token=payload["predicted_token"],
        confidence=payload["confidence"],
        distance=payload["distance"],
        unigram_percentile=payload["unigram_percentile"],
        decoys=payload["decoys"],
        clean_logit_target=payload["clean_logit_target"],
        clean_logit_runner_up=payload["clean_logit_runner_up"],
        cascade=json.loads(str(payload["cascade"])),
        content_low=int(payload["content_low"]),
    )
    return rows, pool


def stratified_sample(
    pool: InstancePool, *, total: int, per_sequence: int, seed: int
) -> np.ndarray:
    """Instances spread over as many sequences as possible, for cluster bootstrap."""

    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for sequence in np.unique(pool.sequence):
        members = np.flatnonzero(pool.sequence == sequence)
        take = min(per_sequence, members.size)
        chosen.extend(int(value) for value in rng.choice(members, size=take, replace=False))
    chosen_array = np.asarray(sorted(chosen), dtype=np.int64)
    if chosen_array.size > total:
        chosen_array = np.sort(rng.choice(chosen_array, size=total, replace=False))
    return chosen_array


def per_cluster_matrix(
    values: np.ndarray, sequence: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean of ``values`` (statistic, instance) within each contributing sequence."""

    clusters = np.unique(sequence)
    matrix = np.zeros((clusters.size, values.shape[0]), dtype=np.float64)
    weights = np.zeros(clusters.size, dtype=np.float64)
    for index, cluster in enumerate(clusters):
        members = sequence == cluster
        matrix[index] = values[:, members].mean(axis=1)
        weights[index] = float(members.sum())
    return matrix, weights, clusters


# ---------------------------------------------------------------- census


def census(
    args: argparse.Namespace,
    out: Path,
    arm_name: str,
    *,
    ban_depth: int | None = None,
) -> dict[str, Any]:
    """Census one arm. ``arm_name`` is the arm being censused, not necessarily the
    text control: D2.c needs this stage on a protein arm, and every helper below it
    (:func:`build_cohorts`, :func:`make_pool`) was already arm-generic. Arms are run
    in separate ``--out`` directories, so the per-arm artefacts do not collide and
    need no renaming.

    ``ban_depth`` is threaded because it is not a free parameter across modalities:
    the decoy ban is over the model's top-``ban_depth`` predictions, and EXP-R2-088
    measured it emptying the decoy pool for 28589 of 31115 eligible positions on
    ProGen2-medium (vocabulary 31) against 5 of 10042 on ProtGPT2 (vocabulary
    50257). Per-residue tokenisers need the relaxed depth; the matched pair does
    not."""
    arm = load_arm(
        arm_name,
        device=args.device,
        dtype=args.census_dtype,
        attn_implementation="eager",
    )
    cohort, reference = build_cohorts(args, arm_name)
    rows, pool, unigram_counts = make_pool(
        arm, args, cohort, reference, ban_depth=ban_depth
    )
    save_pool(out / f"pool_{arm_name}.npz", rows, pool)
    print(f"  A1 cascade: {json.dumps(pool.cascade)}")

    scores = paa_attention_scores(arm, rows, pool, batch_size=args.attention_batch)
    layers, heads = arm.n_layer, n_head(arm)
    active = scores["active_sequences"]
    weights = scores["instances_per_sequence"].astype(np.float64)
    paa_flat = scores["paa_specific"].reshape(len(rows), layers * heads)
    booted = cluster_bootstrap(
        paa_flat, weights, replicates=args.bootstrap, seed=args.seed + 1
    )
    paa_mean = booted["mean"].reshape(layers, heads)
    non_sink = (
        scores["non_sink_mass"][active] * weights[active, None, None]
    ).sum(axis=0) / weights[active].sum()

    unigram = fit_unigram(
        arm, cohort.input_strings(arm), max_tokens=args.unigram_max_tokens
    )
    probes: list[RepeatProbe] = synthetic_repeat_probes(
        arm,
        unigram,
        n_probes=args.synthetic_probes,
        copy_len=args.synthetic_copy_len,
        seed=args.seed,
    )
    induction = decoy_corrected_prefix_matching(
        arm,
        probes,
        batch_size=args.probe_batch,
        n_decoys=args.n_decoys,
        seed=args.seed + 2,
    )
    prefix_raw = induction["per_probe_prefix_matching"].mean(axis=0)
    prefix_corrected = induction["per_probe_prefix_matching_decoy_corrected"].mean(axis=0)

    dissociation = partial_spearman(
        prefix_raw.reshape(-1), paa_mean.reshape(-1), non_sink.reshape(-1)
    )
    jaccard = top_set_jaccard(prefix_raw, paa_mean, count=args.top_set)
    stop = dissociation["partial_spearman"] > 0.5 or jaccard["jaccard"] > 0.3

    ranking = np.argsort(paa_mean, axis=None)[::-1]
    top_paa = [flat_head_index(paa_mean, int(flat)) for flat in ranking[: args.causal_heads]]
    rng = np.random.default_rng(args.seed + 3)
    control_pool = ranking[args.control_offset :]
    controls = [
        flat_head_index(paa_mean, int(flat))
        for flat in rng.choice(control_pool, size=args.control_heads, replace=False)
    ]
    top_induction = [
        flat_head_index(prefix_raw, int(flat))
        for flat in np.argsort(prefix_raw, axis=None)[::-1][: args.query_heads]
    ]

    np.savez_compressed(
        out / "census_matrices.npz",
        paa_specific_per_sequence=scores["paa_specific"],
        antecedent_attention_per_sequence=scores["antecedent_attention"],
        decoy_attention_per_sequence=scores["decoy_attention"],
        non_sink_mass_per_sequence=scores["non_sink_mass"],
        instances_per_sequence=scores["instances_per_sequence"],
        # The knockout-matched score and the key count that makes it differ from
        # the one above. Persisted rather than recomputed because the attention
        # pattern is not retained and re-deriving it costs the whole pass.
        paa_specific_matched_per_sequence=scores["paa_specific_matched"],
        antecedent_set_attention_per_sequence=scores["antecedent_set_attention"],
        decoy_attention_size_matched_per_sequence=scores["decoy_attention_size_matched"],
        keys_per_instance=scores["keys_per_instance"],
        paa_specific_mean=paa_mean,
        paa_specific_q_low=booted["q_low"].reshape(layers, heads),
        paa_specific_q_high=booted["q_high"].reshape(layers, heads),
        prefix_matching_per_probe=induction["per_probe_prefix_matching"],
        prefix_matching_decoy_corrected_per_probe=induction[
            "per_probe_prefix_matching_decoy_corrected"
        ],
        non_sink_mass=non_sink,
    )
    np.save(out / f"unigram_counts_{arm_name}.npy", unigram_counts)

    report: dict[str, Any] = {
        "arm": arm_name,
        "cohort_digest": cohort.digest,
        "n_sequences": len(rows),
        "width": int(args.width),
        "a1_candidate_pool": {
            **pool.cascade,
            "gate_minimum": int(args.a1_minimum),
            "verdict": "PASS" if len(pool) >= args.a1_minimum else "FAIL",
            # Two denominators, both named. The first is the historical one and
            # its denominator is NOT "candidates": it is the induction-blocked
            # positions plus the ones that yielded an instance, which omits the
            # distance-blocked positions and the ones where no candidate reached
            # the antecedent test at all -- 16.2% of gpt2-large's scored
            # positions under the accounting that lost them. It is kept because
            # EXP-R2-059 quotes 10.6%; the second is the rate a reader means.
            "induction_target_discard_rate": (
                pool.cascade["candidates_discarded_by_induction_target"]
                / max(
                    pool.cascade["candidates_discarded_by_induction_target"]
                    + pool.cascade["positions_with_eligible_candidate"],
                    1,
                )
            ),
            "induction_target_discard_rate_denominator": (
                "candidates_discarded_by_induction_target + "
                "positions_with_eligible_candidate"
            ),
            "induction_target_discard_rate_of_scored_positions": (
                (
                    pool.cascade["candidates_discarded_by_induction_target"]
                    + pool.cascade["candidates_discarded_by_induction_and_distance"]
                )
                / max(pool.cascade["positions_scored"], 1)
            ),
            "instances": len(pool),
            "median_distance": float(np.median(pool.distance)),
            "median_confidence": float(np.median(pool.confidence)),
        },
        "paa_head_distribution": summarise_head_matrix(paa_mean, "paa_specific"),
        # How far the selector's key set diverges from the one the causal
        # knockout removes, on this arm. The rank correlation is the quantity
        # that decides whether a census-to-causal comparison may use the cheaper
        # score: the two definitions coincide only where the predicted token
        # occurs once before the query, and how often that happens is a property
        # of the alphabet, not of the model.
        "knockout_matched_score": {
            "mean_keys_per_instance": float(
                np.nanmean(scores["keys_per_instance"][active])
            ),
            "distribution": summarise_head_matrix(
                np.nanmean(scores["paa_specific_matched"][active], axis=0),
                "paa_specific_matched",
            ),
            "spearman_against_paa_specific": partial_spearman(
                paa_mean.reshape(-1),
                np.nanmean(scores["paa_specific_matched"][active], axis=0).reshape(-1),
                non_sink.reshape(-1),
            ),
            "note": (
                "paa_specific scores the nearest earlier occurrence of the "
                "predicted token; knockout_effects removes every earlier "
                "occurrence. paa_specific_matched is the score on that key set, "
                "with a decoy baseline scaled to the same number of keys. Use the "
                "matched score for any comparison against a causal effect"
            ),
        },
        "top_paa_heads": [
            {
                "layer": layer,
                "head": head,
                "paa_specific": float(paa_mean[layer, head]),
                "q_low": float(booted["q_low"].reshape(layers, heads)[layer, head]),
                "q_high": float(booted["q_high"].reshape(layers, heads)[layer, head]),
                "antecedent_attention": float(
                    (
                        scores["antecedent_attention"][active, layer, head]
                        * weights[active]
                    ).sum()
                    / weights[active].sum()
                ),
                "prefix_matching": float(prefix_raw[layer, head]),
                "depth_fraction": layer / max(layers - 1, 1),
            }
            for layer, head in top_paa
        ],
        "a5_dissociation": {
            **dissociation,
            "top_set_jaccard": jaccard,
            "stop_rule_triggered": bool(stop),
            "verdict": "FAIL_NOT_INDEPENDENT" if stop else "PASS",
        },
        "induction_decoy_correction": {
            "n_probes": len(probes),
            "copy_len_tokens": int(args.synthetic_copy_len),
            "uncorrected": summarise_head_matrix(prefix_raw, "prefix_matching"),
            "decoy_corrected": summarise_head_matrix(
                prefix_corrected, "prefix_matching_decoy_corrected"
            ),
            "n_above_0.10_uncorrected": int((prefix_raw >= 0.10).sum()),
            "n_above_0.10_corrected": int((prefix_corrected >= 0.10).sum()),
            "fraction_above_0.10_uncorrected": float((prefix_raw >= 0.10).mean()),
            "fraction_above_0.10_corrected": float((prefix_corrected >= 0.10).mean()),
            "top20_jaccard": top_set_jaccard(prefix_raw, prefix_corrected, count=20),
            "spearman_uncorrected_vs_corrected": partial_spearman(
                prefix_raw.reshape(-1),
                prefix_corrected.reshape(-1),
                non_sink.reshape(-1),
            ),
            "mean_decoy_attention_at_top_head": float(
                prefix_raw.max() - prefix_corrected.reshape(-1)[int(np.argmax(prefix_raw))]
            ),
        },
        "selected_heads": {
            "causal_top_paa": [[layer, head] for layer, head in top_paa],
            "causal_controls": [[layer, head] for layer, head in controls],
            "query_source_paa": [[layer, head] for layer, head in top_paa[: args.query_heads]],
            "query_source_induction": [[layer, head] for layer, head in top_induction],
        },
    }
    del arm
    gc.collect()
    torch.cuda.empty_cache()
    return report


# ---------------------------------------------------------------- causal


def causal(
    args: argparse.Namespace, out: Path, selection: dict[str, Any], arm_name: str
) -> dict[str, Any]:
    rows, pool = load_pool(out / f"pool_{arm_name}.npz")
    arm = load_arm(
        arm_name,
        device=args.device,
        dtype=args.census_dtype,
        attn_implementation="eager",
    )
    heads = [tuple(item) for item in selection["causal_top_paa"]] + [
        tuple(item) for item in selection["causal_controls"]
    ]
    labels = [f"L{layer}H{head}" for layer, head in heads]
    is_control = [False] * len(selection["causal_top_paa"]) + [True] * len(
        selection["causal_controls"]
    )
    selected = stratified_sample(
        pool,
        total=args.causal_instances,
        per_sequence=args.causal_per_sequence,
        seed=args.seed + 4,
    )
    effects = knockout_effects(
        arm, rows, pool, selected, heads, batch_size=args.causal_batch
    )
    sequence = pool.sequence[selected]
    gap_matrix, weights, clusters = per_cluster_matrix(effects["delta_m_gap"], sequence)
    probability_matrix, _, _ = per_cluster_matrix(effects["delta_probability"], sequence)
    gap = cluster_bootstrap(
        gap_matrix, weights, replicates=args.bootstrap, seed=args.seed + 5
    )
    # The Bonferroni column asks for a percentile at 0.05/(2*n_heads). Whether
    # `--bootstrap` replicates can resolve that tail is a property of the head
    # count, and the head count is exactly what an exhaustive census multiplies by
    # thirty. At 24 heads and 1000 replicates the requested percentile sits at
    # sorted index 1.04, so the "bound" is the second-smallest draw and moves by
    # 0.0028 logits between seeds -- the size of the smallest effect in the table
    # beside it. `cluster_bootstrap` now refuses that rather than returning it, so
    # the choice is made here and recorded: buy the replicates, or publish no
    # Bonferroni bound and say why.
    strict_alpha = 0.05 / len(heads)
    strict_replicates = int(math.ceil(MINIMUM_DRAWS_IN_TAIL * 2.0 / strict_alpha))
    gap_strict = None
    strict_note = (
        f"a percentile at alpha/2 = {strict_alpha / 2:.3e} needs "
        f"{strict_replicates} replicates to put {MINIMUM_DRAWS_IN_TAIL:.0f} draws "
        f"below it; the cap is --max-bonferroni-bootstrap={args.max_bonferroni_bootstrap}"
    )
    if strict_replicates <= args.max_bonferroni_bootstrap:
        gap_strict = cluster_bootstrap(
            gap_matrix,
            weights,
            replicates=max(strict_replicates, args.bootstrap),
            seed=args.seed + 5,
            alpha=strict_alpha,
        )
        strict_note = (
            f"percentile interval at alpha={strict_alpha:.3e} over "
            f"{max(strict_replicates, args.bootstrap)} replicates"
        )
    probability = cluster_bootstrap(
        probability_matrix, weights, replicates=args.bootstrap, seed=args.seed + 6
    )

    mass = effects["antecedent_attention_mass"]
    np.savez_compressed(
        out / "causal_matrices.npz",
        delta_m_gap=effects["delta_m_gap"],
        delta_probability=effects["delta_probability"],
        antecedent_attention_mass=mass,
        per_cluster_delta_m_gap=gap_matrix,
        per_cluster_delta_probability=probability_matrix,
        cluster_weights=weights,
        clusters=clusters,
        heads=np.asarray(heads, dtype=np.int64),
        instance_index=selected,
    )

    # A pooled mean over every instance cannot separate "no suppressive head"
    # from "a head that suppresses hard wherever it is actually addressed".
    # The decile is the instances where this head reads the antecedent most.
    decile = max(int(round(0.1 * selected.size)), 8)
    table = [
        {
            "head": labels[index],
            "layer": int(heads[index][0]),
            "head_index": int(heads[index][1]),
            "is_control": bool(is_control[index]),
            "delta_m_gap": float(gap["mean"][index]),
            "delta_m_gap_q025": float(gap["q_low"][index]),
            "delta_m_gap_q975": float(gap["q_high"][index]),
            "delta_m_gap_q_bonferroni_low": (
                float(gap_strict["q_low"][index]) if gap_strict is not None else None
            ),
            "delta_m_gap_q_bonferroni_basis": strict_note,
            "delta_probability": float(probability["mean"][index]),
            "delta_probability_q025": float(probability["q_low"][index]),
            "delta_probability_q975": float(probability["q_high"][index]),
            "bootstrap_fraction_positive": float(gap["fraction_positive"][index]),
            "antecedent_attention_mass_mean": float(mass[index].mean()),
            "antecedent_attention_mass_p90": float(np.quantile(mass[index], 0.9)),
            "delta_m_gap_top_decile_by_mass": float(
                effects["delta_m_gap"][index][np.argsort(mass[index])[::-1][:decile]].mean()
            ),
            "delta_probability_top_decile_by_mass": float(
                effects["delta_probability"][index][
                    np.argsort(mass[index])[::-1][:decile]
                ].mean()
            ),
        }
        for index in range(len(heads))
    ]
    table.sort(key=lambda row: row["delta_m_gap"], reverse=True)
    suppressive = [
        row for row in table if row["delta_m_gap_q025"] > 0 and not row["is_control"]
    ]
    control_positive = [
        row for row in table if row["delta_m_gap_q025"] > 0 and row["is_control"]
    ]
    del arm
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "n_instances": int(selected.size),
        "n_clusters": int(clusters.size),
        "bootstrap_replicates": int(args.bootstrap),
        "zero_mask_max_logit_difference": float(effects["zero_mask_max_logit_difference"]),
        "heads": table,
        "a3_class_non_emptiness": {
            "n_heads_tested": len(heads),
            "n_suppressive_lower_bound_positive": len(suppressive),
            "n_control_heads_lower_bound_positive": len(control_positive),
            "verdict": "PASS" if suppressive else "FAIL",
        },
        "a4_causal_magnitude": {
            "top_head": table[0]["head"],
            "delta_m_gap": table[0]["delta_m_gap"],
            "delta_m_gap_q025": table[0]["delta_m_gap_q025"],
            "delta_probability": table[0]["delta_probability"],
            "delta_m_gap_top_decile_by_mass": table[0]["delta_m_gap_top_decile_by_mass"],
            "delta_probability_top_decile_by_mass": table[0][
                "delta_probability_top_decile_by_mass"
            ],
            "clean_m_gap_mean": float(pool.m_gap[selected].mean()),
            "delta_m_gap_relative_to_clean": float(
                table[0]["delta_m_gap"] / max(abs(pool.m_gap[selected].mean()), 1e-9)
            ),
            "best_control_delta_m_gap": max(
                row["delta_m_gap"] for row in table if row["is_control"]
            ),
        },
    }


# ---------------------------------------------------------------- matching


def matching(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    text_rows, text_pool = load_pool(out / f"pool_{args.text_arm}.npz")
    text_unigram = np.load(out / f"unigram_counts_{args.text_arm}.npy")
    text_arm = load_arm(args.text_arm, device=args.device, dtype=args.census_dtype)
    text_selected = stratified_sample(
        text_pool,
        total=args.match_instances,
        per_sequence=args.match_per_sequence,
        seed=args.seed + 7,
    )
    text_corruption = corruption_effects(
        text_arm,
        text_rows,
        text_pool,
        text_selected,
        unigram_counts=text_unigram,
        batch_size=args.corruption_batch,
        seed=args.seed + 8,
    )
    del text_arm
    gc.collect()
    torch.cuda.empty_cache()

    # The decoy rule bans a decoy token from being one of the model's top
    # candidates at q.  Over a twenty-symbol alphabet the top-20 ban is the
    # whole alphabet, so the design as specified cannot construct a protein
    # instance at all.  Both depths are run: the specified one, and a relaxed
    # one that says how much of any failure is the ban rather than the
    # covariates.
    protein: dict[str, Any] = {}
    protein_cells: dict[tuple[str, int], dict[str, set[str]]] = {}
    for name in args.protein_arms:
        arm = load_arm(name, device=args.device, dtype=args.census_dtype)
        cohort, reference = build_cohorts(args, name)
        for depth in args.protein_ban_depths:
            rows, pool, unigram_counts = make_pool(
                arm, args, cohort, reference, ban_depth=depth
            )
            save_pool(out / f"pool_{name}_ban{depth}.npz", rows, pool)
            selected = stratified_sample(
                pool,
                total=args.match_instances,
                per_sequence=args.match_per_sequence,
                seed=args.seed + 7,
            )
            effects = corruption_effects(
                arm,
                rows,
                pool,
                selected,
                unigram_counts=unigram_counts,
                batch_size=args.corruption_batch,
                seed=args.seed + 8,
            )
            subpool = _subset(pool, selected)
            protein[f"{name}@ban{depth}"] = {
                "cascade": pool.cascade,
                "instances": len(pool),
                "positions_scored": pool.cascade["positions_scored"],
                "instance_yield": len(pool) / max(pool.cascade["positions_scored"], 1),
                "matched_subset": int(selected.size),
                "median_distance": float(np.median(subpool.distance)),
                "median_confidence": float(np.median(subpool.confidence)),
                "median_unigram_percentile": float(np.median(subpool.unigram_percentile)),
                "median_corruption_effect": float(np.median(effects)),
            }
            protein_cells[(name, depth)] = {}
            for gates in _gate_prefixes():
                cells = coarsened_cells(subpool, effects, gates=gates)
                protein_cells[(name, depth)]["|".join(gates)] = set(cells.tolist())
        del arm
        gc.collect()
        torch.cuda.empty_cache()

    text_subpool = _subset(text_pool, text_selected)
    cascades: dict[str, list[dict[str, Any]]] = {}
    verdicts: dict[str, Any] = {}
    for depth in args.protein_ban_depths:
        cascade: list[dict[str, Any]] = []
        for gates in _gate_prefixes():
            key = "|".join(gates)
            cells = coarsened_cells(text_subpool, text_corruption, gates=gates)
            available: set[str] = set()
            for name in args.protein_arms:
                available |= protein_cells[(name, depth)][key]
            retained = int(sum(1 for cell in cells if cell in available))
            cascade.append(
                {
                    "gates": list(gates),
                    "text_instances": int(cells.size),
                    "retained": retained,
                    "retention": retained / max(cells.size, 1),
                    "scaled_to_full_pool": int(
                        round(retained / max(cells.size, 1) * len(text_pool))
                    ),
                }
            )
        cascades[f"ban{depth}"] = cascade
        full = cascade[-1]
        verdicts[f"ban{depth}"] = {
            "retention_all_gates": full["retention"],
            "surviving_in_measured_subset": full["retained"],
            "projected_surviving_full_pool": full["scaled_to_full_pool"],
            "verdict": "PASS" if full["scaled_to_full_pool"] >= args.a2_minimum else "FAIL",
        }
    specified = f"ban{args.protein_ban_depths[0]}"
    return {
        "text_arm": args.text_arm,
        "protein_arms": list(args.protein_arms),
        "text_matched_subset": int(text_selected.size),
        "text_pool_size": len(text_pool),
        "text_median_confidence": float(np.median(text_subpool.confidence)),
        "text_median_distance": float(np.median(text_subpool.distance)),
        "text_median_unigram_percentile": float(np.median(text_subpool.unigram_percentile)),
        "text_median_corruption_effect": float(np.median(text_corruption)),
        "protein": protein,
        "retention_cascades": cascades,
        "a2_matching_feasibility": {
            "gate_minimum": int(args.a2_minimum),
            "as_specified_ban_depth": int(args.protein_ban_depths[0]),
            **verdicts[specified],
            "by_ban_depth": verdicts,
        },
    }


def _gate_prefixes() -> list[tuple[str, ...]]:
    order = ("distance", "unigram_percentile", "confidence", "corruption")
    return [tuple(order[: index + 1]) for index in range(len(order))]


def _subset(pool: InstancePool, selected: np.ndarray) -> InstancePool:
    """A pool restricted to ``selected``. ``content_low`` is a property of the
    rendering, not of the instances, so it is carried unchanged; dropping it here
    reset it to 0 for the same reason :func:`save_pool` describes."""

    return InstancePool(
        arm=pool.arm,
        sequence=pool.sequence[selected],
        query=pool.query[selected],
        antecedent=pool.antecedent[selected],
        predicted_token=pool.predicted_token[selected],
        confidence=pool.confidence[selected],
        distance=pool.distance[selected],
        unigram_percentile=pool.unigram_percentile[selected],
        decoys=pool.decoys[selected],
        clean_logit_target=pool.clean_logit_target[selected],
        clean_logit_runner_up=pool.clean_logit_runner_up[selected],
        cascade=pool.cascade,
        content_low=pool.content_low,
    )


# ------------------------------------------------------------ query source


def query_source(
    args: argparse.Namespace, out: Path, selection: dict[str, Any]
) -> dict[str, Any]:
    rows, pool = load_pool(out / f"pool_{args.text_arm}.npz")
    unigram_counts = np.load(out / f"unigram_counts_{args.text_arm}.npy")
    arm = load_arm(
        args.text_arm,
        device=args.device,
        dtype=args.census_dtype,
        attn_implementation="eager",
    )
    paa_heads = [tuple(item) for item in selection["query_source_paa"]]
    induction_heads = [tuple(item) for item in selection["query_source_induction"]]
    heads = sorted(set(paa_heads) | set(induction_heads))
    selected = stratified_sample(
        pool, total=args.query_instances, per_sequence=1, seed=args.seed + 9
    )
    substitutes = _absent_frequency_matched(rows, pool, selected, unigram_counts)
    result = query_source_intervention(
        arm,
        rows,
        pool,
        selected,
        heads,
        substitutes,
        alphas=args.alphas,
        batch_size=args.query_batch,
        seed=args.seed + 10,
    )
    summary: dict[str, Any] = {"paa": [], "induction": []}
    for label, members in (("paa", paa_heads), ("induction", induction_heads)):
        for layer, head in members:
            record = result["heads"][f"L{layer}H{head}"]
            mass = record["antecedent_mass_by_alpha"]
            control = record["antecedent_mass_by_alpha_random_control"]
            summary[label].append(
                {
                    "head": f"L{layer}H{head}",
                    "antecedent_mass_by_alpha": mass,
                    "antecedent_mass_by_alpha_random_control": control,
                    "relative_change": (mass[-1] - mass[0]) / max(mass[0], 1e-9),
                    "relative_change_random_control": (control[-1] - control[0])
                    / max(control[0], 1e-9),
                    "excess_over_random_control": (mass[-1] - control[-1])
                    / max(mass[0], 1e-9),
                    "monotone_decreasing": all(
                        mass[index + 1] <= mass[index] + 1e-9 for index in range(len(mass) - 1)
                    ),
                }
            )
    del arm
    gc.collect()
    torch.cuda.empty_cache()
    for label in ("paa", "induction"):
        summary[label].sort(key=lambda row: row["relative_change"])
    return {
        "n_instances": int(selected.size),
        "alphas": result["alphas"],
        "manipulation_check": result["manipulation_check"],
        "by_class": summary,
        "a6_verdict": {
            "paa_monotone_fraction": float(
                np.mean([row["monotone_decreasing"] for row in summary["paa"]])
            ),
            "induction_monotone_fraction": float(
                np.mean([row["monotone_decreasing"] for row in summary["induction"]])
            ),
            "paa_median_relative_change": float(
                np.median([row["relative_change"] for row in summary["paa"]])
            ),
            "induction_median_relative_change": float(
                np.median([row["relative_change"] for row in summary["induction"]])
            ),
            "paa_median_relative_change_random_control": float(
                np.median([row["relative_change_random_control"] for row in summary["paa"]])
            ),
            "induction_median_relative_change_random_control": float(
                np.median(
                    [row["relative_change_random_control"] for row in summary["induction"]]
                )
            ),
            "paa_median_excess_over_random_control": float(
                np.median([row["excess_over_random_control"] for row in summary["paa"]])
            ),
            "induction_median_excess_over_random_control": float(
                np.median(
                    [row["excess_over_random_control"] for row in summary["induction"]]
                )
            ),
        },
    }


def _absent_frequency_matched(
    rows: list[list[int]],
    pool: InstancePool,
    selected: np.ndarray,
    unigram_counts: np.ndarray,
) -> np.ndarray:
    """A token absent from the context whose unigram count is closest to X's.

    Absent from the context so that the steered prediction has no antecedent of
    its own, frequency-matched so the nudge is not confounded with steering
    towards a systematically commoner token.
    """

    counts = unigram_counts.astype(np.float64)
    candidates = np.argsort(counts)[::-1][:5000]
    candidate_counts = counts[candidates]
    substitutes = np.zeros(selected.size, dtype=np.int64)
    for position, index in enumerate(selected):
        present = set(int(value) for value in rows[int(pool.sequence[index])])
        target = int(pool.predicted_token[index])
        wanted = counts[target]
        order = np.argsort(np.abs(candidate_counts - wanted))
        for offset in order:
            candidate = int(candidates[offset])
            if candidate != target and candidate not in present:
                substitutes[position] = candidate
                break
        else:
            raise RuntimeError("no absent frequency-matched substitute token available")
    return substitutes


# ------------------------------------------------------------------ driver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    parser.add_argument("--text-arm", default="gpt2-large")
    parser.add_argument("--gate0-arms", nargs="+", default=None)
    parser.add_argument("--protein-arms", nargs="+", default=["progen2-medium"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--census-dtype",
        default="float32",
        help=(
            "precision for every stage that reads attention weights or logit "
            "differences; bfloat16 quantises the M-gap change to multiples of "
            "1/512, which is the size of the effect being measured"
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260728)

    parser.add_argument("--gate0-sequences", type=int, default=200)
    parser.add_argument("--gate0-batch", type=int, default=4)
    parser.add_argument("--gate0-threshold", type=float, default=GATE0_MIN_NATS)
    parser.add_argument("--reference-sequences", type=int, default=4000)
    parser.add_argument("--protein-min-len", type=int, default=200)
    parser.add_argument("--protein-max-len", type=int, default=800)
    parser.add_argument("--text-min-chars", type=int, default=800)
    parser.add_argument("--text-max-len", type=int, default=512)
    parser.add_argument("--protein-source", choices=("plain", "ec"), default="plain")
    parser.add_argument("--gate0-label", default="gate0")
    parser.add_argument(
        "--cohort-draw-seed",
        type=int,
        default=DEFAULT_CORPUS_DRAW_SEED,
        help="seed for the permutation every cohort in this gate is drawn under; "
        "0 selects the historical file-order prefix, which is a declared choice "
        "and not a default (transfer audit, Appendix B rule 1). Under a seed "
        "--cohort-skip indexes a disjoint window of the same permutation rather "
        "than a later prefix of the same file",
    )
    parser.add_argument(
        "--cohort-skip",
        type=int,
        default=0,
        help=(
            "records to skip before the cohort starts; a second, disjoint block "
            "is how a file-order selection effect is detected rather than assumed absent"
        ),
    )

    parser.add_argument("--census-sequences", type=int, default=200)
    parser.add_argument("--min-sequences", type=int, default=64)
    parser.add_argument("--census-text-min-chars", type=int, default=4000)
    parser.add_argument("--census-protein-min-len", type=int, default=520)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--query-min", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-depth", type=int, default=5)
    parser.add_argument("--min-confidence", type=float, default=0.01)
    parser.add_argument("--n-decoys", type=int, default=4)
    parser.add_argument("--pool-batch", type=int, default=4)
    parser.add_argument("--attention-batch", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--max-bonferroni-bootstrap",
        type=int,
        default=40000,
        help=(
            "replicate ceiling for the head-count-corrected interval. The "
            "requested percentile is 0.05/(2*n_heads), so the replicates needed "
            "to resolve it grow linearly in the head count: about 9,600 at the 24 "
            "heads this stage screens and about 288,000 at the 720 an exhaustive "
            "census patches. Above the ceiling the column is withheld with its "
            "reason rather than published as an order statistic of the extreme draws"
        ),
    )
    parser.add_argument("--a1-minimum", type=int, default=20000)
    parser.add_argument("--a2-minimum", type=int, default=2000)
    parser.add_argument("--top-set", type=int, default=20)

    parser.add_argument("--unigram-max-tokens", type=int, default=512)
    parser.add_argument("--synthetic-probes", type=int, default=16)
    parser.add_argument("--synthetic-copy-len", type=int, default=64)
    parser.add_argument("--probe-batch", type=int, default=4)

    parser.add_argument("--causal-heads", type=int, default=16)
    parser.add_argument("--control-heads", type=int, default=8)
    parser.add_argument("--control-offset", type=int, default=120)
    parser.add_argument("--causal-instances", type=int, default=800)
    parser.add_argument("--causal-per-sequence", type=int, default=4)
    parser.add_argument("--causal-batch", type=int, default=16)

    parser.add_argument("--protein-ban-depths", type=int, nargs="+", default=[20, 3])
    parser.add_argument("--match-instances", type=int, default=3000)
    parser.add_argument("--match-per-sequence", type=int, default=16)
    parser.add_argument("--corruption-batch", type=int, default=16)

    parser.add_argument("--query-heads", type=int, default=12)
    parser.add_argument("--query-instances", type=int, default=128)
    parser.add_argument("--query-batch", type=int, default=8)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    # The arm the `census` and `causal` stages run on. Defaults to the text control,
    # which is every use before D2.c. `match` and `query` are unaffected: they
    # compare the protein arms AGAINST the text control and keep naming --text-arm.
    parser.add_argument("--census-arm", default=None)
    # Depth of the decoy ban, passed to the census pool. None keeps
    # `build_instance_pool`'s own default. This is not a free parameter across
    # modalities -- see census().
    parser.add_argument("--census-ban-depth", type=int, default=None)
    args = parser.parse_args()

    if args.gate0_arms is None:
        args.gate0_arms = [args.text_arm, "protgpt2", "zymctrl", "progen2-medium"]
    census_arm = args.census_arm or args.text_arm
    unknown = [
        name
        for name in args.gate0_arms + args.protein_arms + [args.text_arm, census_arm]
        if name not in PANEL
    ]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; panel is {sorted(PANEL)}")
    # `match` and `query` read the text control's pool and unigram counts, which only
    # the `census` stage writes and only for the arm it censused. Running them in the
    # same invocation as a census of a DIFFERENT arm would look like it worked and
    # would silently score against whatever pool happened to be in the directory, so
    # it is refused rather than ordered around.
    if census_arm != args.text_arm and {"match", "query"} & set(args.stages):
        raise ValueError(
            f"--census-arm {census_arm!r} differs from --text-arm {args.text_arm!r}, "
            f"but stages {sorted({'match', 'query'} & set(args.stages))} consume the "
            f"text control's pool and unigram counts. Run the text census in its own "
            f"--out directory first, then those stages against it."
        )
    args.out.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stages_requested": list(args.stages),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
    }
    selection_path = args.out / "selected_heads.json"

    if "gate0" in args.stages:
        print("[gate0] cohort power, held-out unigram baseline")
        payload["gate0"] = gate0(args)
        write_json(args.out / f"{args.gate0_label}.json", payload["gate0"])
    if "census" in args.stages:
        print("[census] A1 pool, A5 dissociation, induction decoy correction")
        payload["census"] = census(
            args, args.out, census_arm, ban_depth=args.census_ban_depth
        )
        write_json(args.out / "census.json", payload["census"])
        write_json(selection_path, payload["census"]["selected_heads"])
    if "causal" in args.stages:
        print("[causal] A3 class non-emptiness, A4 causal magnitude")
        selection = json.loads(selection_path.read_text())
        payload["causal"] = causal(args, args.out, selection, census_arm)
        write_json(args.out / "causal.json", payload["causal"])
    if "match" in args.stages:
        print("[match] A2 matching feasibility against the protein arms")
        payload["match"] = matching(args, args.out)
        write_json(args.out / "match.json", payload["match"])
    if "query" in args.stages:
        print("[query] A6 query-source intervention")
        selection = json.loads(selection_path.read_text())
        payload["query_source"] = query_source(args, args.out, selection)
        write_json(args.out / "query_source.json", payload["query_source"])

    write_json(args.out / "paa_gate_report.json", payload)
    print(f"wrote {args.out / 'paa_gate_report.json'}")


if __name__ == "__main__":
    main()
