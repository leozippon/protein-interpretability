#!/usr/bin/env python3
"""The lens family: what each layer of each panel arm is trying to say.

A text unembedding maps onto ~50k word-like tokens which are themselves the
language explanations are written in. A protein unembedding maps onto twenty
amino acids carrying at most 4.32 bits and admitting no semantic decomposition.
If that difference is why mechanistic-interpretability methods transfer poorly
from text decoders to protein decoders, it should be visible in the simplest
possible probe: read the residual stream through the model's own output
interface at every depth and see what comes out.

Three methods run under one code path, on one relative-depth grid, with matched
cohort size and matched protocol, so that a 36-layer text decoder and a
27-layer protein decoder produce comparable rows.

*Logit lens* -- the residual stream projected through the trained final layer
norm and unembedding. Reported per layer: cross-entropy against the true next
token, KL to the model's own final distribution, top-1 agreement with the final
prediction, and the entropy of the lens distribution. For protein arms the same
pass also reports the top predicted amino acids per layer and whether the
trajectory resolves chemical class before it resolves specific residues.

*Tuned lens* -- a per-layer affine translator fitted to remove the basis error
the logit lens suffers from, trained on a disjoint split of the same cohort and
evaluated on the split the logit lens is scored on.

*J-lens* -- the exact Jacobian of the final logits with respect to the layer-l
residual stream, summarised by its singular structure and compared against the
subspace the layer's activations actually occupy. Its formulation is written
into every output because it is far less standardised than the other two.

Measurability
-------------
``src.transfer.budget`` measures whether an arm extracts any information from
context on a cohort at all. An arm whose clean cross-entropy exceeds its own
unigram entropy on the cohort is off-distribution: its next-token distribution
is worse than the cohort's context-free marginal, so every lens quantity would
be describing a trajectory towards a prediction the arm should not be making.
Such an arm is flagged and *not* scored. ProtGPT2 is known to be off-distribution
on short EC-labelled cohorts and is expected to trigger this.

The weaker guard in ``budget.power_status`` -- context information below
``MIN_CONTEXT_INFORMATION_NATS`` -- is recorded but does not stop scoring. That
threshold exists to protect *normalised recovery ratios*, whose denominator is
the context information; no quantity here is such a ratio, so a small
denominator widens no interval and invalidates nothing. Arms in that band are
reported with their power verdict attached.

One JSON per arm is written under ``results/transfer/lens_family/``.
"""

from __future__ import annotations

import argparse
import gc
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# The stage directory itself, so `panel_contract` imports under every invocation
# style this file sees: `python scripts/transfer/08_lens_family.py` (which puts
# the directory on sys.path anyway) and the worker's import preflight, which
# loads this file through importlib.util.spec_from_file_location and does not.
_STAGE_DIR = str(Path(__file__).resolve().parent)
if _STAGE_DIR not in sys.path:
    sys.path.insert(0, _STAGE_DIR)

from panel_contract import CAMPAIGN_PANEL, arm_can_run, stage_contract_record  # noqa: E402
from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    DEFAULT_CORPUS_DRAW_SEED,
    PANEL,
    Arm,
    Cohort,
    load_arm,
    protein_cohort,
    text_cohort,
)
from src.transfer.budget import (  # noqa: E402
    MIN_CONTEXT_INFORMATION_NATS,
    arm_power,
)
from src.transfer.lenses import (  # noqa: E402
    AA_CLASSES,
    DEFAULT_DEPTH_FRACTIONS,
    ActivationSubspace,
    LayerPoint,
    activation_subspace,
    activation_subspace_summary,
    cache_residuals,
    coarse_to_fine_gap,
    freeze_parameters,
    jacobian_alignment,
    jacobian_cluster_bootstrap,
    jacobian_finite_difference_check,
    jacobian_formulation,
    jacobian_gram,
    jacobian_matrices,
    jacobian_probe_row,
    layer_grid,
    lens_cluster_bootstrap,
    lens_head,
    lens_metrics,
    lens_trajectory,
    per_symbol_view,
    prepare_windows,
    residue_class_cluster_bootstrap,
    residue_class_metrics,
    residue_class_trajectory,
    residue_vocabulary,
    sample_jacobian_probes,
    scored_position_count,
    split_cohort,
    train_tuned_lens,
    trajectory_summary,
    tuned_versus_untuned,
    verify_lens_head,
)
from src.transfer.pathways import (  # noqa: E402
    PROTEIN_COHORT_SOURCES,
    TEXT_COHORT_SOURCE,
    cohort_composition,
    subsample_cohort,
)

SCHEMA_VERSION = "r2_transfer_lens_family_v1"

#: Tolerance on "the tuned lens beats the untuned lens", in nats. Held-out KL is
#: an average over tens of thousands of positions, so anything inside this band
#: is a tie rather than a regression.
TUNED_IMPROVEMENT_TOLERANCE_NATS = 1e-4


def cohort_pool(arm_names: list[str], args: argparse.Namespace) -> dict[str, Cohort]:
    """One pool per modality, so arms of the same modality share their cohort.

    Every protein arm draws from the same corpus, including the unconditional
    ones; ``Cohort.input_strings`` then renders it in each arm's native format.
    Giving different protein arms different corpora would make their
    cross-entropies, and therefore their lens trajectories, incomparable.
    """

    if args.protein_source not in PROTEIN_COHORT_SOURCES:
        raise ValueError(f"unknown protein cohort source {args.protein_source!r}")
    modalities = {PANEL[name].modality for name in arm_names}
    pools: dict[str, Cohort] = {}
    seed = args.cohort_draw_seed or None
    if "text" in modalities:
        pools["text"] = text_cohort(
            args.pool_size,
            min_chars=args.text_min_chars,
            name=TEXT_COHORT_SOURCE,
            seed=seed,
        )
    if "protein" in modalities:
        pools["protein"] = protein_cohort(
            args.pool_size,
            args.res_min,
            args.res_max,
            name=args.protein_source,
            with_ec=args.protein_source == "ec_labelled_swissprot",
            seed=seed,
        )
    return pools


def cohort_source(arm: Arm, args: argparse.Namespace) -> str:
    return TEXT_COHORT_SOURCE if arm.modality == "text" else args.protein_source


def arm_record(arm: Arm) -> dict[str, Any]:
    return {
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
    }


def grid_record(grid: tuple[LayerPoint, ...], n_layer: int) -> list[dict[str, Any]]:
    return [
        {
            "layer": point.layer,
            "relative_depth": point.relative_depth,
            "depth_fractions": list(point.depth_fractions),
            "n_layer": n_layer,
        }
        for point in grid
    ]


def measure_logit_and_tuned(
    arm: Arm,
    head: Any,
    grid: tuple[LayerPoint, ...],
    eval_cache: Any,
    train_cache: Any,
    args: argparse.Namespace,
    symbols_per_token: float,
    max_bytes: int,
) -> dict[str, Any]:
    """Both distributional lenses, scored on the same held-out positions."""

    layers = [point.layer for point in grid]
    untuned_rows = lens_trajectory(
        head, eval_cache, device=arm.device, chunk=args.metric_chunk
    )
    untuned = {layer: lens_metrics(untuned_rows[layer]) for layer in layers}

    translators, training = train_tuned_lens(
        head,
        train_cache,
        device=arm.device,
        steps=args.tuned_steps,
        batch_size=args.tuned_batch_size,
        learning_rate=args.tuned_lr,
        weight_decay=args.tuned_weight_decay,
        seed=args.tuned_seed,
        log_every=args.tuned_log_every,
        max_bytes=max_bytes,
        progress=True,
    )
    tuned_rows = lens_trajectory(
        head,
        eval_cache,
        device=arm.device,
        chunk=args.metric_chunk,
        translators=translators,
    )
    tuned = {layer: lens_metrics(tuned_rows[layer]) for layer in layers}

    def layer_rows(
        metrics: dict[int, dict[str, Any]], rows: dict[int, list[dict[str, float | int]]]
    ) -> list[dict[str, Any]]:
        return [
            {
                "layer": point.layer,
                "relative_depth": point.relative_depth,
                "depth_fractions": list(point.depth_fractions),
                "per_token": metrics[point.layer],
                "per_symbol": per_symbol_view(metrics[point.layer], symbols_per_token),
                "cluster_bootstrap": lens_cluster_bootstrap(
                    rows[point.layer],
                    samples=args.bootstrap_samples,
                    seed=args.bootstrap_seed + point.layer,
                ),
            }
            for point in grid
        ]

    return {
        "translators": translators,
        "logit_lens": {
            "layers": layer_rows(untuned, untuned_rows),
            "trajectory": trajectory_summary(grid, untuned),
        },
        "tuned_lens": {
            "training": training,
            "layers": layer_rows(tuned, tuned_rows),
            "trajectory": trajectory_summary(grid, tuned),
            "improvement_over_untuned": tuned_versus_untuned(
                grid,
                untuned,
                tuned,
                identity_layer=eval_cache.final_layer,
                tolerance_nats=TUNED_IMPROVEMENT_TOLERANCE_NATS,
            ),
        },
    }


def measure_residue_classes(
    arm: Arm,
    head: Any,
    grid: tuple[LayerPoint, ...],
    eval_cache: Any,
    translators: dict[int, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Coarse-to-fine residue structure, for the untuned and the tuned lens.

    Both lenses are reported because a coarse-to-fine trajectory seen only
    through the untuned lens could be an artefact of the basis error the tuned
    lens removes; if it survives tuning it is a property of the model.
    """

    vocabulary = residue_vocabulary(arm, device=arm.device)
    depths = [point.relative_depth for point in grid]
    layers = [point.layer for point in grid]
    report: dict[str, Any] = {
        "chemical_classes": {name: list(residues) for name, residues in AA_CLASSES.items()},
        "token_to_residue_mapping": {
            "rule": (
                "a token maps to a residue when its decoded form consists only of canonical "
                "residues; the mapped residue is the first one the token emits"
            ),
            "mapped_tokens": vocabulary.n_mapped_tokens,
            "vocab_size": vocabulary.vocab_size,
            "max_residues_per_token": max(vocabulary.residues_per_token),
            "exact_for_residue_level_arms": max(vocabulary.residues_per_token) == 1,
        },
    }
    for label, active in (("logit_lens", None), ("tuned_lens", translators)):
        rows, marginals = residue_class_trajectory(
            head,
            eval_cache,
            vocabulary,
            device=arm.device,
            chunk=args.metric_chunk,
            translators=active,
        )
        metrics = {layer: residue_class_metrics(rows[layer]) for layer in layers}
        report[label] = {
            "layers": [
                {
                    "layer": point.layer,
                    "relative_depth": point.relative_depth,
                    "metrics": metrics[point.layer],
                    "marginals": marginals[point.layer],
                    "cluster_bootstrap": residue_class_cluster_bootstrap(
                        rows[point.layer],
                        samples=args.bootstrap_samples,
                        seed=args.bootstrap_seed + 1000 + point.layer,
                    ),
                }
                for point in grid
            ],
            "coarse_to_fine": coarse_to_fine_gap(
                depths,
                [metrics[layer]["class_ce_nats"] for layer in layers],
                [metrics[layer]["within_class_ce_nats"] for layer in layers],
            ),
            "class_ce_nats": [metrics[layer]["class_ce_nats"] for layer in layers],
            "within_class_ce_nats": [metrics[layer]["within_class_ce_nats"] for layer in layers],
            "residue_ce_nats": [metrics[layer]["residue_ce_nats"] for layer in layers],
        }
    return report


def measure_jacobian(
    arm: Arm,
    head: Any,
    grid: tuple[LayerPoint, ...],
    eval_cache: Any,
    eval_windows: list[Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """The J-lens on a matched number of probe positions per arm."""

    layers = [point.layer for point in grid]
    freeze_parameters(arm)
    gram = jacobian_gram(head)
    subspaces: dict[int, ActivationSubspace] = {
        layer: activation_subspace(eval_cache.residual[layer], device=arm.device)
        for layer in layers
    }
    probes = sample_jacobian_probes(
        eval_windows,
        count=args.jacobian_positions,
        relative_position=args.jacobian_relative_position,
        seed=args.jacobian_seed,
    )
    check_layer = min(layers, key=lambda layer: abs((layer + 1) / arm.n_layer - 0.5))
    rows_by_layer: dict[int, list[dict[str, float]]] = {layer: [] for layer in layers}
    alignments_by_layer: dict[int, list[dict[str, Any]]] = {layer: [] for layer in layers}
    finite_difference: dict[str, Any] | None = None

    for index, probe in enumerate(probes):
        matrices = jacobian_matrices(
            arm, head, probe, layers, chunk=args.jacobian_chunk
        )
        if index == 0:
            finite_difference = jacobian_finite_difference_check(
                arm,
                head,
                probe,
                matrices[check_layer],
                check_layer,
                epsilon=args.jacobian_finite_difference_epsilon,
                seed=args.jacobian_seed,
            )
            relative = finite_difference["relative_error"]
            if relative is None or relative > args.jacobian_finite_difference_tolerance:
                raise FloatingPointError(
                    f"{arm.name}: Jacobian disagrees with a central finite difference by "
                    f"{relative} at layer {check_layer}"
                )
        for layer in layers:
            alignment = jacobian_alignment(
                matrices[layer],
                gram,
                subspaces[layer],
                rank=args.alignment_rank,
                floor_relative=args.spectrum_floor,
            )
            alignments_by_layer[layer].append(alignment)
            rows_by_layer[layer].append(jacobian_probe_row(alignment))
        del matrices
        print(
            f"  [{arm.name}] Jacobian probe {index + 1}/{len(probes)} "
            f"sequence {probe.sequence_index} position {probe.position}",
            flush=True,
        )

    layer_records: list[dict[str, Any]] = []
    for point in grid:
        rows = rows_by_layer[point.layer]
        alignments = alignments_by_layer[point.layer]
        spectra = [alignment["spectrum"]["top_singular_values"] for alignment in alignments]
        layer_records.append(
            {
                "layer": point.layer,
                "relative_depth": point.relative_depth,
                "probes": len(rows),
                # An isotropic subspace of the same size would capture this share
                # of the activation variance; expressed_energy_fraction is only
                # interpretable against it.
                "chance_expressed_fraction": args.alignment_rank / arm.d_model,
                "algebraic_rank_bound": min(arm.d_model, int(head.vocab_size) - 1),
                "probe_mean": {
                    key: sum(row[key] for row in rows) / len(rows) for key in rows[0]
                },
                "cluster_bootstrap": jacobian_cluster_bootstrap(
                    rows,
                    samples=args.bootstrap_samples,
                    seed=args.bootstrap_seed + 2000 + point.layer,
                ),
                "mean_top_singular_values": [
                    sum(spectrum[index] for spectrum in spectra) / len(spectra)
                    for index in range(len(spectra[0]))
                ],
                "activation_subspace": activation_subspace_summary(subspaces[point.layer]),
                "per_probe": [
                    {"sequence_index": probe.sequence_index, "position": probe.position, **row}
                    for probe, row in zip(probes, rows)
                ],
            }
        )

    return {
        "formulation": jacobian_formulation(head, layers=layers),
        "probes": [
            {
                "sequence_index": probe.sequence_index,
                "position": probe.position,
                "context_tokens": probe.context_tokens,
            }
            for probe in probes
        ],
        "probe_selection": {
            "count": args.jacobian_positions,
            "relative_position": args.jacobian_relative_position,
            "seed": args.jacobian_seed,
            "one_probe_per_sequence": True,
        },
        "reverse_mode_columns_per_probe": arm.d_model,
        "chunk": args.jacobian_chunk,
        "finite_difference_check": finite_difference,
        "layers": layer_records,
    }


def measure_arm(arm: Arm, pool: Cohort, args: argparse.Namespace) -> dict[str, Any]:
    """Every lens on one arm, or a flagged record if the arm is off-distribution."""

    device = torch.device(arm.device)
    torch.cuda.reset_peak_memory_stats(device)
    source = cohort_source(arm, args)
    cohort = subsample_cohort(pool, args.n_seq, args.cohort_seed)
    train_cohort, eval_cohort = split_cohort(cohort, args.train_fraction, args.split_seed)

    power = arm_power(
        arm,
        eval_cohort,
        max_len=args.max_len,
        batch_size=args.batch_size,
        minimum_context_information_nats=args.minimum_context_information_nats,
    )
    off_distribution = power["context_information_nats"] <= 0.0
    measurability = {
        "evaluated_on": "evaluation_split",
        "off_distribution": bool(off_distribution),
        "off_distribution_rule": "clean cross-entropy exceeds the cohort unigram entropy",
        "power": power,
    }
    cohort_block = {
        "source": source,
        "pool": cohort_composition(pool, source=source),
        "subsample": cohort_composition(cohort, source=source),
        "train_split": cohort_composition(train_cohort, source=source),
        "evaluation_split": cohort_composition(eval_cohort, source=source),
        "digest": eval_cohort.digest,
        "splits_disjoint": True,
        "split_unit": "sequence",
    }
    header = {
        "arm": arm_record(arm),
        "cohort_source": source,
        "cohort": cohort_block,
        "measurability": measurability,
    }
    if off_distribution:
        print(
            f"  [{arm.name}] OFF-DISTRIBUTION on {source}: clean CE "
            f"{power['clean_ce_nats']:.4f} nats exceeds unigram entropy "
            f"{power['unigram_entropy_on_cohort_nats']:.4f} nats; flagged, not scored",
            flush=True,
        )
        return {
            **header,
            "scored": False,
            "skipped_reason": "off_distribution_on_this_cohort",
            "layer_grid": grid_record(layer_grid(arm.n_layer, args.depths), arm.n_layer),
            "resources": {
                "peak_accelerator_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "peak_accelerator_memory_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
            },
        }

    symbols_per_token = float(power["symbols_per_token"])
    grid = layer_grid(arm.n_layer, args.depths)
    layers = [point.layer for point in grid]
    head = lens_head(arm)
    max_bytes = int(args.max_cache_gib * 2**30)

    eval_windows = prepare_windows(
        arm, eval_cohort, max_len=args.max_len, batch_size=args.batch_size
    )
    verification = verify_lens_head(
        arm, head, eval_windows[0], tolerance_nats=args.lens_head_tolerance_nats
    )
    eval_cache = cache_residuals(arm, eval_windows, layers, max_bytes=max_bytes)
    print(
        f"  [{arm.name}] evaluation split: {eval_cache.n_sequences} sequences, "
        f"{len(eval_cache)} scored positions, {len(layers)} grid layers",
        flush=True,
    )

    train_windows = prepare_windows(
        arm, train_cohort, max_len=args.max_len, batch_size=args.batch_size
    )
    train_cache = cache_residuals(arm, train_windows, layers, max_bytes=max_bytes)
    print(
        f"  [{arm.name}] training split: {train_cache.n_sequences} sequences, "
        f"{len(train_cache)} positions",
        flush=True,
    )

    distributional = measure_logit_and_tuned(
        arm, head, grid, eval_cache, train_cache, args, symbols_per_token, max_bytes
    )
    translators = distributional.pop("translators")
    del train_cache, train_windows
    gc.collect()
    torch.cuda.empty_cache()

    for label in ("logit_lens", "tuned_lens"):
        trajectory = distributional[label]["trajectory"]
        print(
            f"  [{arm.name}] {label:11s} CE {trajectory['ce_nats'][0]:.4f} -> "
            f"{trajectory['ce_nats'][-1]:.4f} nats/token, KL "
            f"{trajectory['kl_to_final_nats'][0]:.4f} -> "
            f"{trajectory['kl_to_final_nats'][-1]:.4f}, agreement "
            f"{trajectory['top1_agreement_with_final'][0]:.3f} -> "
            f"{trajectory['top1_agreement_with_final'][-1]:.3f}",
            flush=True,
        )
    improvement = distributional["tuned_lens"]["improvement_over_untuned"]
    print(
        f"  [{arm.name}] tuned lens beats untuned at every non-identity layer: "
        f"{improvement['kl_improves_at_every_non_identity_layer']} "
        f"(mean KL reduction {improvement['mean_kl_reduction_nats']:.4f} nats)",
        flush=True,
    )

    residue = (
        measure_residue_classes(arm, head, grid, eval_cache, translators, args)
        if arm.modality == "protein"
        else None
    )
    jacobian = measure_jacobian(arm, head, grid, eval_cache, eval_windows, args)

    return {
        **header,
        "scored": True,
        "skipped_reason": None,
        "symbols_per_token": symbols_per_token,
        "layer_grid": grid_record(grid, arm.n_layer),
        "scored_positions": {
            "rule": (
                "attention-mask-valid next-token targets belonging to the cohort's own "
                "content; for EC-conditioned ZymCTRL the conditioning prefix and the "
                "terminator are excluded, following src.transfer.budget"
            ),
            "evaluation_positions": len(eval_cache),
            "evaluation_sequences": eval_cache.n_sequences,
            "evaluation_window_positions": scored_position_count(eval_windows),
        },
        "lens_head_verification": verification,
        "logit_lens": distributional["logit_lens"],
        "tuned_lens": distributional["tuned_lens"],
        "residue_class_trajectory": residue,
        "jacobian_lens": jacobian,
        "resources": {
            "peak_accelerator_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "peak_accelerator_memory_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)
            ),
        },
    }


def default_lens_arms() -> list[str]:
    """The campaign arms this script can actually serve, from the one declaration.

    Not the arms whose ``ArmSpec`` declares the ``lens`` capability:
    ``lens_head`` resolves the final normalisation as ``transformer.ln_f`` and
    requires an ``nn.LayerNorm`` with a learned gain and bias, so an RMSNorm
    rotary decoder raises partway through a scheduled run. The
    intended-versus-deliverable split is deliberate and recorded on ``ArmSpec``
    and in ``src.transfer.scaling.lens_supported``; a default that schedules an
    arm the module cannot serve is a defect regardless of intent.

    This file carried its own ``LENS_ARCHITECTURES = ("gpt2", "progen")`` until
    EXP-R2-067 -- a second copy of ``scaling.LENS_ARCHITECTURES``, which
    ``panel_contract.STAGE_CONTRACTS["lens_family"]`` imports as the
    authoritative one. ``panel_contract``'s own docstring lists "08_lens_family
    derived its own capability-filtered default" as one of the five duplications
    it exists to remove, and a test pins scaling's copy while nothing pinned
    this one. Filtering on ``CAMPAIGN_PANEL`` rather than all of ``PANEL`` also
    stops this default from proposing an arm no campaign stages.
    """

    return sorted(
        name
        for name in CAMPAIGN_PANEL
        if arm_can_run("lens_family", name).can_run
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms", nargs="+", default=default_lens_arms(), choices=sorted(PANEL)
    )
    parser.add_argument("--device", default="cuda:0")
    # float32 by default: every lens quantity is a difference between two nearly
    # identical distributions, and the Jacobian spectrum's tail is what the
    # effective-rank statistics read. bfloat16 rounding is comparable to both.
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--n-seq", type=int, default=128)
    parser.add_argument("--pool-size", type=int, default=4000)
    parser.add_argument(
        "--cohort-draw-seed",
        type=int,
        default=DEFAULT_CORPUS_DRAW_SEED,
        help="seed for the permutation the corpus pool is drawn under, distinct "
        "from --cohort-seed, which subsamples that pool; 0 selects the historical "
        "file-order prefix (transfer audit, Appendix B rule 1)",
    )
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cohort-seed", type=int, default=20260728)
    parser.add_argument("--split-seed", type=int, default=20260729)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument(
        "--depths", nargs="+", type=float, default=list(DEFAULT_DEPTH_FRACTIONS)
    )
    parser.add_argument("--metric-chunk", type=int, default=512)
    parser.add_argument("--max-cache-gib", type=float, default=8.0)
    parser.add_argument("--lens-head-tolerance-nats", type=float, default=1e-3)

    parser.add_argument("--tuned-steps", type=int, default=2000)
    parser.add_argument("--tuned-batch-size", type=int, default=256)
    parser.add_argument("--tuned-lr", type=float, default=1e-3)
    parser.add_argument("--tuned-weight-decay", type=float, default=0.0)
    parser.add_argument("--tuned-seed", type=int, default=20260730)
    parser.add_argument("--tuned-log-every", type=int, default=100)

    parser.add_argument("--jacobian-positions", type=int, default=16)
    parser.add_argument("--jacobian-chunk", type=int, default=16)
    parser.add_argument("--jacobian-relative-position", type=float, default=0.6)
    parser.add_argument("--jacobian-seed", type=int, default=20260731)
    # The check is limited by float32 cancellation in the forward pass, not by
    # the truncation error of the central difference: measured relative error on
    # gpt2-large falls monotonically from 0.69 at 1e-4 to 5e-4 at 3e-1. A step of
    # 1e-1 is small against residual-stream norms of order 10 to 300 and large
    # enough to be resolved.
    parser.add_argument("--jacobian-finite-difference-epsilon", type=float, default=1e-1)
    parser.add_argument("--jacobian-finite-difference-tolerance", type=float, default=2e-2)
    parser.add_argument("--alignment-rank", type=int, default=16)
    parser.add_argument("--spectrum-floor", type=float, default=1e-6)

    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)

    parser.add_argument("--res-min", type=int, default=64)
    parser.add_argument("--res-max", type=int, default=120)
    parser.add_argument("--text-min-chars", type=int, default=800)
    parser.add_argument(
        "--protein-source",
        default="ec_labelled_swissprot",
        choices=list(PROTEIN_COHORT_SOURCES),
    )
    parser.add_argument(
        "--minimum-context-information-nats",
        type=float,
        default=MIN_CONTEXT_INFORMATION_NATS,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "results" / "transfer" / "lens_family",
    )
    return parser


def evaluation_split_size(n_seq: int, train_fraction: float) -> int:
    """Sequences the evaluation split will hold, by ``split_cohort``'s arithmetic.

    Duplicated deliberately, and only here: it is the quantity two argument
    combinations have to be checked against before a run starts, and
    :func:`src.transfer.lenses.split_cohort` cannot be asked without a cohort.
    :func:`test_evaluation_split_size_matches_split_cohort` holds the two
    together.
    """

    return n_seq - int(round(train_fraction * n_seq))


def validate(args: argparse.Namespace) -> None:
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("--arms repeats an arm")
    refused = {
        arm: verdict.reason
        for arm in args.arms
        if not (verdict := arm_can_run("lens_family", arm)).can_run
    }
    if refused:
        # The capability-filtered default (default_lens_arms) only protects a run
        # that passes no --arms at all, and the worker always passes one. An arm
        # this module cannot serve otherwise raises inside lens_head, after the
        # checkpoint is loaded and a full arm_power pass has run.
        raise ValueError(
            f"08_lens_family.py cannot serve {sorted(refused)}: {refused}. "
            "The lens capability on an ArmSpec is an intent; "
            "src.transfer.scaling.LENS_ARCHITECTURES is what the module delivers."
        )
    if args.n_seq > args.pool_size:
        raise ValueError("--n-seq cannot exceed --pool-size")
    if any(not 0.0 <= depth <= 1.0 for depth in args.depths):
        raise ValueError("--depths must lie in [0, 1]")
    if len(set(args.depths)) != len(args.depths):
        raise ValueError("--depths repeats a fraction")
    if args.jacobian_positions < 2:
        raise ValueError("--jacobian-positions must be at least two for a cluster bootstrap")
    if args.alignment_rank < 1:
        raise ValueError("--alignment-rank must be positive")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must lie strictly between zero and one")
    # --jacobian-positions is silently coupled to --n-seq and --train-fraction:
    # sample_jacobian_probes draws one probe per *evaluation* sequence, so the
    # ceiling is n_seq x (1 - train_fraction). Unchecked, the mismatch surfaced as
    # "only 10 sequences available for 16 Jacobian probes" from inside
    # measure_jacobian -- after the checkpoint was loaded, both splits were
    # tokenised and cached, and the tuned lens had been trained for --tuned-steps
    # (default 2000) steps. All three inputs are known at parse time.
    evaluation_sequences = evaluation_split_size(args.n_seq, args.train_fraction)
    if evaluation_sequences < 1:
        raise ValueError(
            f"--n-seq {args.n_seq} at --train-fraction {args.train_fraction} leaves "
            "no evaluation sequences"
        )
    if args.jacobian_positions > evaluation_sequences:
        raise ValueError(
            f"--jacobian-positions {args.jacobian_positions} exceeds the "
            f"{evaluation_sequences} evaluation sequences that --n-seq {args.n_seq} "
            f"at --train-fraction {args.train_fraction} leaves; one probe is drawn "
            "per evaluation sequence so that the bootstrap over probes stays a "
            "sequence-cluster bootstrap"
        )


def main() -> None:
    args = build_parser().parse_args()
    validate(args)

    pools = cohort_pool(list(args.arms), args)
    output_root = args.output_root.resolve()
    started = datetime.now(timezone.utc).isoformat()

    for name in args.arms:
        print(f"[{name}] loading", flush=True)
        arm = load_arm(name, device=args.device, dtype=args.dtype)
        payload = measure_arm(arm, pools[arm.modality], args)
        write_json(
            output_root / f"{name}.json",
            {
                "schema_version": SCHEMA_VERSION,
                "started_at_utc": started,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "runner_sha256": sha256_file(Path(__file__)),
                "lenses_module_sha256": sha256_file(
                    REPO_ROOT / "src" / "transfer" / "lenses.py"
                ),
                "arms_module_sha256": sha256_file(REPO_ROOT / "src" / "transfer" / "arms.py"),
                "budget_module_sha256": sha256_file(
                    REPO_ROOT / "src" / "transfer" / "budget.py"
                ),
                "configuration": {
                    "device": args.device,
                    "dtype": args.dtype,
                    "sequences": args.n_seq,
                    "pool_size": args.pool_size,
                    "max_len": args.max_len,
                    "batch_size": args.batch_size,
                    "train_fraction": args.train_fraction,
                    "depth_fractions": list(args.depths),
                    "metric_chunk": args.metric_chunk,
                    "protein_cohort_source": args.protein_source,
                    "cohort_source": cohort_source(arm, args),
                    "cohort_draw_seed": int(args.cohort_draw_seed),
                    "residue_length_range": [args.res_min, args.res_max],
                    "text_min_chars": args.text_min_chars,
                },
                # Which arms this stage measured and which it did not, and the
                # protein band it drew on beside the band cohort_power qualified
                # the arms on. This stage's band is 64-120 against the qualifying
                # 64-246, so an arm's PASS verdict does not cover the population
                # scored here; the artefact now says so instead of leaving it to
                # be found by comparing two argparse defaults.
                "stage_contract": stage_contract_record("lens_family", list(args.arms)),
                "matched_compute": {
                    "statement": (
                        "cohort size, relative-depth grid size, tuned-lens optimisation "
                        "steps and batch size, Jacobian probe count and the number of "
                        "reverse-mode columns per probe (exactly d_model) are identical "
                        "across arms; wall-clock and FLOP cost differ because vocabulary "
                        "size and width differ, and that difference is the object of study"
                    ),
                    "sequences": args.n_seq,
                    "max_len": args.max_len,
                    "depth_fractions": list(args.depths),
                    "tuned_steps": args.tuned_steps,
                    "tuned_batch_size": args.tuned_batch_size,
                    "jacobian_positions": args.jacobian_positions,
                    "jacobian_columns_per_probe": "d_model",
                    "bootstrap_samples": args.bootstrap_samples,
                },
                "seeds": {
                    "cohort_subsample": args.cohort_seed,
                    "train_eval_split": args.split_seed,
                    "tuned_lens": args.tuned_seed,
                    "jacobian_probes": args.jacobian_seed,
                    "bootstrap_base": args.bootstrap_seed,
                },
                "thresholds": {
                    "minimum_context_information_nats": args.minimum_context_information_nats,
                    "minimum_context_information_provenance": (
                        "src.transfer.budget.MIN_CONTEXT_INFORMATION_NATS; recorded but not "
                        "used to skip scoring, only the off-distribution rule skips an arm"
                    ),
                    "lens_head_tolerance_nats": args.lens_head_tolerance_nats,
                    "tuned_improvement_tolerance_nats": TUNED_IMPROVEMENT_TOLERANCE_NATS,
                    "jacobian_spectrum_floor_relative": args.spectrum_floor,
                    "jacobian_alignment_rank": args.alignment_rank,
                    "jacobian_finite_difference_tolerance": (
                        args.jacobian_finite_difference_tolerance
                    ),
                },
                "bootstrap": {
                    "cluster_unit": "sequence",
                    "samples": args.bootstrap_samples,
                    "base_seed": args.bootstrap_seed,
                },
                **payload,
            },
        )
        print(f"wrote {output_root / f'{name}.json'}", flush=True)
        del arm
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
