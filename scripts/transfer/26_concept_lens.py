#!/usr/bin/env python3
"""Phase A of the concept-aligned lens (D3.b / R3.1): decoding, nulls, aperture.

**What this measures.** At every point of the relative-depth grid, the lens
distribution is read through a pre-declared token-to-property table -- three
biochemical properties on a protein arm, three surface properties on a text arm
-- and the readout is scored against the property the model's own next token
actually carries. Three statistics, all declared before the first run: a
within-protein-centred Spearman of the readout against the realised value; the
coarsened cross-entropy of the true property class, swept over 2 to 5 classes;
and the aperture gain, the depth at which the property resolves minus the depth
at which symbol identity does, swept over three fractions.

**What decides anything is the null, not the value.** Every coarsening of a
distribution looks good: collapsing twenty residues onto three classes takes the
64-246 band marginal from 4.1566 bits to about 1.58, so the lens has under 40%
as much to be right about. Each statistic is therefore reported against a
shuffled-property null and a rank-matched partition null that holds the coarsened
entropy fixed, and the excess over the null is the result. A raw value from this
stage is not quotable on its own.

**The positive control is a precondition, not a reading.** At the final layer
the lens *is* the model, so an arm whose final-layer statistic does not clear its
null's 99.9th percentile is reported ``unmeasurable_on_this_cohort`` and its
intermediate layers are not read at all. This is Appendix B rule 2 in the form
that binds here: the control is internal and guaranteed by construction, so a
failure is an instrument failure rather than a null result.

**The text arm is here to close the method reading.** The same procedure with a
surface-property table. If the aperture gain is as large on gpt2-large as on the
protein arms, the effect is a property of coarsening and therefore of the method
(§5's organising rule). It cannot open a modality reading in the other direction:
a text arm needs no renormalisation while a protein arm discards non-residue
mass, so the two are not the same estimand and no cross-modality coefficient is
computed. D3.b admits cross-modality claims only on modality-neutral
abstractions, and a biochemical property is not one.

**Phase B is gated on this stage and is not run from it.** No intervention, no
steering, no ablation. If the rank-matched null is not cleared, or the text
control shows the same or a larger aperture gain, Phase B is not authorised and
the result is reported as a method property.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.arms import (  # noqa: E402
    AA20,
    DEFAULT_CORPUS_DRAW_SEED,
    _eligible_protein_records,
    PANEL,
    SWISSPROT_FASTA,
    ZYMCTRL_FASTA,
    Cohort,
    iter_fasta,
    load_arm,
    text_cohort,
)
from src.transfer.concept_lens import (  # noqa: E402
    CLASS_COUNT_SWEEP,
    DECISION_QUANTILE,
    NULL_DRAWS,
    POSITIVE_CONTROL_QUANTILE,
    PROPERTY_BASIS,
    TEXT_PROPERTY_NAMES,
    aperture_gain,
    basis_correlations,
    equal_mass_classes,
    null_excess,
    partition_null_quality,
    rank_matched_partitions,
    residue_axis,
    shuffled_property_null,
    target_symbols,
    text_property_table,
    within_unit_centred_spearman,
)
from src.transfer.families import (  # noqa: E402
    boundary_leakage,
    family_assignment,
    family_disjoint_split,
    load_cath_superfamilies,
    load_pfam_families,
)
from src.transfer.io import write_json  # noqa: E402
from src.transfer.lenses import (  # noqa: E402
    DEFAULT_DEPTH_FRACTIONS,
    cache_residuals,
    layer_grid,
    lens_head,
    prepare_windows,
    residue_vocabulary,
    split_cohort,
    verify_lens_head,
)
from src.transfer.probes import FIXED_EC_TAG  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "results" / "transfer" / "concept_lens"

#: EC-conditioning modes this stage can measure, and the one it cannot.
#:
#: ``native`` is the arm's own prompt. ``fixed`` gives every protein one constant
#: tag: the rendering stays the declared one, the arm stays on the distribution
#: it was trained on, and the tag's mutual information with the protein drops to
#: zero -- which is the decisive test of whether a conditioning tag, rather than
#: the sequence, is carrying a result.
#:
#: ``unconditioned`` is **structurally unavailable to this estimand** and is not
#: offered. The scored span of an EC-conditioned arm is delimited by its own
#: start and end tokens -- that is what ``target_rule("ec_conditioned")`` reads --
#: so dropping the tag drops the boundaries and leaves no definition of which
#: positions are scored. A number produced by scoring some other span would not
#: be comparable with the native and fixed cells it exists to be compared with.
#: It is also the weaker of the two tests, because it takes the arm off its
#: training distribution and confounds tag removal with distribution shift.
EC_CONDITIONING_MEASURABLE: tuple[str, ...] = ("native", "fixed")

#: The six arms Phase A is declared over: the matched pair that is the only
#: controlled modality comparison, the ProGen2 matched-corpus contrast, the
#: residue-level EC-conditioned arm, and gpt2 as the depth control §5.05(c)
#: requires, since estimand attainability in this programme falls with depth.
PHASE_A_ARMS: tuple[str, ...] = (
    "gpt2-large",
    "gpt2",
    "protgpt2",
    "progen2-base",
    "progen2-medium",
    "zymctrl",
)


# ------------------------------------------------------------- cohort


def accession_cohort(
    *,
    n: int,
    min_len: int,
    max_len: int,
    seed: int,
    skip: int,
    with_ec: bool,
) -> Cohort:
    """A Swiss-Prot cohort that keeps the accession of every record.

    ``arms.protein_cohort`` discards accessions, and a family-disjoint split
    needs them, so the draw is rebuilt here with the same discipline rather than
    the corpus being read in file order: the whole eligible set is enumerated,
    permuted under a declared seed, and a window taken (Appendix B rule 1). The
    rendering is untouched -- the returned object is an ordinary ``Cohort`` and
    ``Cohort.input_strings`` remains the one renderer (rule 12).
    """

    if seed is None:
        raise ValueError("a seeded draw is required; file order has manufactured three effects")
    source = ZYMCTRL_FASTA if with_ec else SWISSPROT_FASTA
    allowed = set(AA20)
    records: list[tuple[str, str, str | None]] = []
    for header, body in iter_fasta(source):
        if with_ec:
            # The EC corpus stores the *rendered* prompt, not the bare sequence:
            # "3.5.99.7<sep><start>MNLQ...<end>". Reading it as a sequence makes
            # every record fail the alphabet filter and the cohort come back
            # empty, which is how this was found.
            if "<start>" not in body or "<end>" not in body:
                continue
            sequence = body.split("<start>")[1].split("<end>")[0]
            label: str | None = body.split("<sep>")[0]
            accession = header.partition("|")[0]
        else:
            sequence, label = body, None
            fields = header.split("|")
            if len(fields) < 3 or fields[0] != "sp":
                raise ValueError(f"unexpected Swiss-Prot header {header!r}")
            accession = fields[1]
        if not min_len <= len(sequence) <= max_len or not set(sequence) <= allowed:
            continue
        records.append((accession, sequence, label))
    if not records:
        raise RuntimeError(f"{source}: no eligible record in {min_len}-{max_len}")

    # Which records exist is a declaration that lives in ``arms``, and this
    # function needs accessions that the declaration does not yield. Rather than
    # trusting two readers of one corpus to agree, the sequence stream is checked
    # against the declaration's own, so a divergence raises here instead of
    # silently drawing a different population (Appendix B rule 12). The check is
    # what caught the EC rendering above.
    declared = [
        sequence for sequence, _ in _eligible_protein_records(min_len, max_len, with_ec=with_ec)
    ]
    mine = [sequence for _, sequence, _ in records]
    if mine != declared:
        raise RuntimeError(
            f"{source}: this stage's eligible set ({len(mine)} records) disagrees with "
            f"arms._eligible_protein_records ({len(declared)}); the two readers of one "
            "corpus have diverged and the draw would not be the panel's population"
        )
    order = np.random.default_rng(seed).permutation(len(records))
    chosen = order[skip : skip + n]
    if chosen.size < n:
        raise RuntimeError(
            f"{source}: {len(records)} eligible records cannot supply {n} past skip {skip}"
        )
    drawn = [records[int(index)] for index in chosen]
    metadata: dict[str, Any] = {
        "sampling": {
            "mode": "seeded_permutation",
            "seed": int(seed),
            "skip": int(skip),
            "requested": int(n),
            "eligible": len(records),
            "corpus": str(source),
        },
        "accessions": [accession for accession, _, _ in drawn],
    }
    if with_ec:
        labels = [label for _, _, label in drawn]
        if any(label is None for label in labels):
            raise RuntimeError("an EC-labelled draw is missing a label")
        metadata["ec_labels"] = labels
    return Cohort(
        "swissprot_accession" + ("_ec" if with_ec else ""),
        "protein",
        [sequence for _, sequence, _ in drawn],
        min_len,
        max_len,
        metadata,
    )


def family_sides(cohort: Cohort, args: argparse.Namespace) -> dict[str, Any]:
    """Split the protein cohort by curated family, or say why it could not be."""

    accessions = list(cohort.metadata["accessions"])
    loader = (
        load_pfam_families if args.family_source == "pfam" else load_cath_superfamilies
    )
    families = loader(set(accessions))
    assignment = family_assignment(
        accessions, families, source=args.family_source, unlabelled="drop"
    )
    split = family_disjoint_split(
        assignment, seed=args.family_seed, train_fraction=args.train_fraction
    )
    position = {accession: index for index, accession in enumerate(accessions)}
    seen = [position[unit] for unit in split.unit_ids("train")]
    unseen = [position[unit] for unit in split.unit_ids("test")]
    sequences = {
        accession: cohort.records[position[accession]] for accession in assignment.unit_ids
    }
    return {
        "seen_positions": seen,
        "unseen_positions": unseen,
        "summary": split.summary,
        "leakage": boundary_leakage(split, sequences, seed=args.family_seed),
    }


# -------------------------------------------------------- one arm, phase A


def select_positions(
    realised: np.ndarray, unit: np.ndarray, *, per_sequence: int, seed: int
) -> np.ndarray:
    """A seeded, equal-count draw of scored positions from each sequence.

    Appendix B rule 21: give every arm the same number of sampling units before
    comparing them. Sequences differ in length and tokenisers differ in symbols
    per token, so taking every scored position would hand a residue-level arm
    three to five times the positions of a multi-residue BPE arm and make the
    comparison partly a reading of sample size. It also bounds memory, which is
    what makes the text arm's full-vocabulary readout affordable at all.
    """

    if per_sequence < 1:
        raise ValueError("per_sequence must be positive")
    generator = np.random.default_rng(seed)
    keep: list[int] = []
    for sequence in np.unique(unit):
        candidates = np.flatnonzero((unit == sequence) & (realised >= 0))
        if candidates.size == 0:
            continue
        take = min(per_sequence, candidates.size)
        keep.extend(int(index) for index in generator.choice(candidates, take, replace=False))
    if not keep:
        raise RuntimeError("no scored position survived selection")
    return np.sort(np.asarray(keep, dtype=np.int64))


def held_out_symbol_marginal(arm, axis, args) -> tuple[np.ndarray, dict[str, Any]]:
    """The symbol marginal, counted on a draw disjoint from the scored cohort.

    The draw is taken at ``cohort_skip + sequences`` under the same seeded
    permutation, so it is a genuinely disjoint window of the same corpus rather
    than a reseed. The targets are obtained through ``cache_residuals`` on one
    layer rather than by re-deriving the scored-target rule here: which position
    is a target is a declaration that lives in ``lenses`` and ``scoring``, and
    Appendix B rule 12 is that such a decision is imported and never restated.
    """

    skip = args.cohort_skip + args.sequences
    if arm.modality == "text":
        cohort = text_cohort(args.marginal_sequences, seed=args.cohort_draw_seed, skip=skip)
    else:
        cohort = accession_cohort(
            n=args.marginal_sequences,
            min_len=args.protein_min_len,
            max_len=args.protein_max_len,
            seed=args.cohort_draw_seed,
            skip=skip,
            with_ec=arm.spec.input_format == "ec_conditioned",
        )
    windows = prepare_windows(
        arm, cohort, max_len=args.max_len, batch_size=args.batch_size
    )
    cache = cache_residuals(arm, windows, [0], max_bytes=args.max_cache_bytes)
    symbols = target_symbols(cache, axis, vocab_size=int(arm.model.config.vocab_size))
    counts = np.bincount(symbols[symbols >= 0], minlength=axis.n_symbols).astype(np.float64)
    provenance = {
        "source": "held_out_draw",
        "cohort": cohort.name,
        "digest": cohort.digest,
        "skip": int(skip),
        "sequences": int(args.marginal_sequences),
        "counted_targets": int((symbols >= 0).sum()),
        "unmapped_targets": int((symbols < 0).sum()),
    }
    return counts, provenance


def measure_arm(
    arm_name: str, cohort: Cohort, side_of_record: np.ndarray | None, args
) -> dict[str, Any]:
    """Phase A on one arm: trajectories, nulls, aperture, and the gate."""

    from src.transfer.concept_lens import layer_concept_statistics, token_axis

    started = time.time()
    arm = load_arm(arm_name, device=args.device, dtype=args.dtype)
    head = lens_head(arm)
    grid = layer_grid(arm.n_layer, tuple(args.depth_fractions))
    windows = prepare_windows(arm, cohort, max_len=args.max_len, batch_size=args.batch_size)
    verification = verify_lens_head(
        arm, head, windows[0], tolerance_nats=args.lens_tolerance_nats
    )
    cache = cache_residuals(
        arm, windows, [point.layer for point in grid], max_bytes=args.max_cache_bytes
    )

    if arm.modality == "protein":
        axis = residue_axis(residue_vocabulary(arm, device=args.device), device=args.device)
        table = {
            name: np.asarray([values[residue] for residue in AA20], dtype=np.float64)
            for name, values in PROPERTY_BASIS.items()
        }
        partition_draws = args.partition_null_draws
    else:
        axis = token_axis(arm)
        table = {
            name: values
            for name, values in text_property_table(arm).items()
            if name in TEXT_PROPERTY_NAMES
        }
        partition_draws = args.text_partition_null_draws

    vocab_size = int(arm.model.config.vocab_size)
    realised_all = target_symbols(cache, axis, vocab_size=vocab_size)
    unit_all = cache.sequence_index.cpu().numpy().astype(np.int64)
    keep = select_positions(
        realised_all,
        unit_all,
        per_sequence=args.positions_per_sequence,
        seed=args.position_seed,
    )
    realised = realised_all[keep]
    unit = unit_all[keep]

    if side_of_record is None:
        train, _ = split_cohort(cohort, args.train_fraction, args.family_seed)
        seen_records = set(train.metadata["sampling"]["indices"])
        side_of_record = np.asarray(
            [0 if index in seen_records else 1 for index in range(len(cohort.records))],
            dtype=np.int64,
        )
    side = side_of_record[unit]

    # The symbol marginal decides the equal-mass partition and therefore both
    # nulls, so it is estimated on a HELD-OUT draw rather than on the scored
    # cohort (Appendix B rule 3). On a twenty-residue arm the two barely differ;
    # on a 50,257-token text arm the scored cohort leaves the overwhelming
    # majority of the alphabet at zero count, and a plug-in marginal there turns
    # an equal-*mass* partition into an equal-*count* one over tokens that never
    # occur -- L12's vocabulary-dependent plug-in bias arriving in a partition
    # instead of in an entropy.
    marginal, marginal_provenance = held_out_symbol_marginal(arm, axis, args)
    absent = int((marginal == 0.0).sum())
    if axis.token_groups is not None and absent:
        missing = [axis.symbols[i] for i in np.flatnonzero(marginal == 0.0)][:8]
        raise RuntimeError(
            f"{arm_name}: residues {missing} never occur as a target on the held-out "
            "draw; an equal-mass partition is not defined against a zero marginal"
        )
    smoothed = marginal + (1.0 if absent else 0.0)
    smoothed = smoothed / smoothed.sum()

    nulls = {
        name: {
            "property": shuffled_property_null(
                values, draws=args.null_draws, seed=args.null_seed
            ),
            "partitions": {
                k: rank_matched_partitions(
                    equal_mass_classes(values, smoothed, k),
                    smoothed,
                    draws=partition_draws,
                    seed=args.null_seed + k,
                )
                for k in CLASS_COUNT_SWEEP
            },
        }
        for name, values in table.items()
    }

    depths = [point.relative_depth for point in grid]
    trajectories: dict[str, dict[str, list[float]]] = {
        side_name: {"symbol_ce": []} for side_name in ("seen", "unseen")
    }
    for side_name in trajectories:
        for name in table:
            for k in CLASS_COUNT_SWEEP:
                trajectories[side_name][f"{name}_k{k}"] = []

    layers: dict[str, Any] = {}
    final_readout: dict[str, np.ndarray] = {}
    for point in grid:
        residual = cache.residual[point.layer][keep]
        entry: dict[str, Any] = {
            "layer": int(point.layer),
            "relative_depth": float(point.relative_depth),
            "sides": {},
        }
        for side_name, side_value in (("seen", 0), ("unseen", 1)):
            mask = side == side_value
            if int(mask.sum()) < 2:
                raise RuntimeError(f"{arm_name}: side {side_name} carries under two positions")
            statistics = layer_concept_statistics(
                head,
                residual[torch.as_tensor(np.flatnonzero(mask))],
                axis,
                targets=realised[mask],
                properties={name: block["property"] for name, block in nulls.items()},
                partitions={name: block["partitions"] for name, block in nulls.items()},
                device=args.device,
                chunk=args.chunk,
            )
            trajectories[side_name]["symbol_ce"].append(statistics["symbol_cross_entropy_nats"])
            block: dict[str, Any] = {
                "n_positions": int(mask.sum()),
                "n_sequences": int(np.unique(unit[mask]).size),
                "symbol_cross_entropy_nats": statistics["symbol_cross_entropy_nats"],
                "abstain_mass_mean": statistics["abstain_mass_mean"],
                "abstain_mass_max": statistics["abstain_mass_max"],
                "concepts": {},
            }
            for name, values in table.items():
                readout = statistics["readout"][name]
                centred = within_unit_centred_spearman(readout, values[realised[mask]], unit[mask])
                concept: dict[str, Any] = {
                    "spearman_within_unit": null_excess(
                        centred[0], centred[1:], quantile=args.decision_quantile
                    ),
                    "class_cross_entropy": {},
                }
                for k in CLASS_COUNT_SWEEP:
                    ce = statistics["class_cross_entropy_nats"][(name, k)]
                    trajectories[side_name][f"{name}_k{k}"].append(float(ce[0]))
                    # Lower cross-entropy is better, so both the observation and
                    # its null enter with the sign flipped and the excess reads
                    # as an advantage in the same direction as every other
                    # statistic here.
                    concept["class_cross_entropy"][str(k)] = null_excess(
                        -float(ce[0]), -ce[1:], quantile=args.decision_quantile
                    )
                block["concepts"][name] = concept
                if point.layer == grid[-1].layer and side_name == "unseen":
                    final_readout[name] = centred
            entry["sides"][side_name] = block
        layers[str(point.layer)] = entry

    positive_control = {
        name: {
            **null_excess(values[0], values[1:], quantile=POSITIVE_CONTROL_QUANTILE),
            "measurable": bool(
                null_excess(values[0], values[1:], quantile=POSITIVE_CONTROL_QUANTILE)["clears_null"]
            ),
            "final_layer": int(grid[-1].layer),
        }
        for name, values in final_readout.items()
    }
    measurable = {name: block["measurable"] for name, block in positive_control.items()}

    report: dict[str, Any] = {
        "arm": arm_name,
        "modality": arm.modality,
        "n_layer": int(arm.n_layer),
        "verdict": "measurable" if any(measurable.values()) else "unmeasurable_on_this_cohort",
        "measurable_concepts": measurable,
        "positive_control": positive_control,
        "lens_head_verification": verification,
        "symbol_axis": {
            "name": axis.name,
            "n_symbols": axis.n_symbols,
            "renormalised": axis.renormalised,
            "n_symbols_absent_from_targets": absent,
        },
        "positions": {
            "scored_available": int((realised_all >= 0).sum()),
            "unscored_targets": int((realised_all < 0).sum()),
            "selected": int(keep.size),
            "per_sequence_cap": int(args.positions_per_sequence),
            "seed": int(args.position_seed),
        },
        "target_marginal_entropy_bits": float(-(smoothed * np.log2(smoothed)).sum()),
        "symbol_marginal": marginal_provenance,
        "null_draws": {"property": int(args.null_draws), "partition": int(partition_draws)},
        "null_quality": {
            name: {
                str(k): partition_null_quality(block["partitions"][k], smoothed)
                for k in CLASS_COUNT_SWEEP
            }
            for name, block in nulls.items()
        },
        "seconds": round(time.time() - started, 1),
    }
    if arm.modality == "protein":
        report["basis_correlations"] = basis_correlations(smoothed)

    # The intermediate layers are only read on an arm whose final-layer control
    # passed. Reporting a shallow-layer null on an arm where the lens does not
    # reproduce the model is reporting the instrument, not the model.
    if report["verdict"] == "measurable":
        report["layers"] = layers
        report["aperture"] = {
            side_name: {
                name: {
                    str(k): aperture_gain(
                        depths,
                        trajectories[side_name]["symbol_ce"],
                        trajectories[side_name][f"{name}_k{k}"],
                    )
                    for k in CLASS_COUNT_SWEEP
                }
                for name in table
            }
            for side_name in ("seen", "unseen")
        }
    else:
        report["layers_withheld_reason"] = (
            "the final-layer positive control did not clear its null on any declared "
            "concept, so the lens does not reproduce this arm's own distribution on "
            "this cohort and no intermediate depth is interpretable"
        )
    return report


# --------------------------------------------------------------- driver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", nargs="+", default=list(PHASE_A_ARMS))
    parser.add_argument("--sequences", type=int, default=400)
    parser.add_argument("--protein-min-len", type=int, default=64)
    parser.add_argument("--protein-max-len", type=int, default=246)
    parser.add_argument(
        "--cohort-draw-seed", type=int, default=DEFAULT_CORPUS_DRAW_SEED
    )
    parser.add_argument("--cohort-skip", type=int, default=0)
    parser.add_argument("--family-source", choices=("pfam", "cath_superfamily"), default="pfam")
    parser.add_argument("--family-seed", type=int, default=1)
    parser.add_argument(
        "--ec-conditioning",
        nargs="+",
        choices=EC_CONDITIONING_MEASURABLE,
        default=list(EC_CONDITIONING_MEASURABLE),
    )
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--positions-per-sequence", type=int, default=64)
    parser.add_argument("--position-seed", type=int, default=11)
    parser.add_argument("--marginal-sequences", type=int, default=256)
    parser.add_argument("--null-draws", type=int, default=NULL_DRAWS)
    parser.add_argument("--partition-null-draws", type=int, default=NULL_DRAWS)
    # A text arm's partition null is a scatter over 50,257 symbols per draw,
    # which is three orders of magnitude more work per draw than a residue arm's.
    # The count is declared separately and recorded rather than being silently
    # the same number meaning a different cost.
    parser.add_argument("--text-partition-null-draws", type=int, default=100)
    parser.add_argument("--null-seed", type=int, default=7)
    parser.add_argument("--decision-quantile", type=float, default=DECISION_QUANTILE)
    parser.add_argument(
        "--depth-fractions", type=float, nargs="+", default=list(DEFAULT_DEPTH_FRACTIONS)
    )
    parser.add_argument("--max-len", type=int, default=288)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--lens-tolerance-nats", type=float, default=2e-2)
    parser.add_argument("--max-cache-bytes", type=int, default=24_000_000_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tag", default="phase_a")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Appendix B rule 7: print what the environment resolved before measuring
    # anything. One environment variable once narrowed a nine-stage campaign's
    # text side to a single model while every downstream number stayed
    # well-formed.
    print("[paths] resolved before any measurement", flush=True)
    print(f"  swissprot {SWISSPROT_FASTA}")
    print(f"  zymctrl   {ZYMCTRL_FASTA}")
    print(f"  out       {args.out}")
    print(f"  panel     {len(PANEL)} arms; requested {args.arms}", flush=True)
    unknown = [name for name in args.arms if name not in PANEL]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; panel is {sorted(PANEL)}")

    cohorts: dict[bool, Cohort] = {}
    sides: dict[bool, dict[str, Any]] = {}

    def protein_side(with_ec: bool) -> tuple[Cohort, dict[str, Any]]:
        """Build a protein cohort and its family split on first use.

        Lazily and inside the per-arm guard: an unbuildable cohort is a refusal
        for the arms that need it, not for the campaign. Building both up front
        cost a six-arm cell that had already measured nothing when the EC corpus
        came back empty.
        """

        if with_ec not in cohorts:
            cohort = accession_cohort(
                n=args.sequences,
                min_len=args.protein_min_len,
                max_len=args.protein_max_len,
                seed=args.cohort_draw_seed,
                skip=args.cohort_skip,
                with_ec=with_ec,
            )
            cohorts[with_ec] = cohort
            sides[with_ec] = family_sides(cohort, args)
            print(
                f"[cohort] with_ec={with_ec}: {len(cohort.records)} proteins, "
                f"{len(sides[with_ec]['seen_positions'])} seen / "
                f"{len(sides[with_ec]['unseen_positions'])} unseen families, "
                f"cross-boundary max k-mer Jaccard "
                f"{sides[with_ec]['leakage']['cross_max']:.4f}",
                flush=True,
            )
        return cohorts[with_ec], sides[with_ec]

    reports: dict[str, Any] = {}
    for arm_name in args.arms:
        spec = PANEL[arm_name]
        conditioned = spec.input_format == "ec_conditioned"
        modes = args.ec_conditioning if conditioned else ("native",)
        for mode in modes:
            key = f"{arm_name}@{mode}" if conditioned else arm_name
            print(f"[{key}] starting ({spec.modality})", flush=True)
            try:
                if spec.modality == "text":
                    cohort = text_cohort(args.sequences, seed=args.cohort_draw_seed)
                    reports[key] = measure_arm(arm_name, cohort, None, args)
                    reports[key]["split"] = {
                        "unit": "sequence",
                        "reason": "a text cohort has no curated family; the split is by document",
                    }
                else:
                    cohort, block = protein_side(conditioned)
                    if mode == "fixed":
                        # One constant tag in the arm's own format. The rendering
                        # stays Cohort.input_strings' (rule 12); only the tag's
                        # mutual information with the protein changes, and it
                        # changes to zero. The cohort, the positions, the family
                        # split and the nulls are bit-identical to the native
                        # cell, so the contrast is the tag and nothing else.
                        cohort = Cohort(
                            f"{cohort.name}_fixed_ec",
                            cohort.kind,
                            list(cohort.records),
                            cohort.min_symbols,
                            cohort.max_symbols,
                            dict(cohort.metadata)
                            | {"ec_labels": [FIXED_EC_TAG] * len(cohort.records)},
                        )
                    side_of_record = np.full(len(cohort.records), -1, dtype=np.int64)
                    side_of_record[block["seen_positions"]] = 0
                    side_of_record[block["unseen_positions"]] = 1
                    reports[key] = measure_arm(arm_name, cohort, side_of_record, args)
                    reports[key]["split"] = block["summary"]
                    reports[key]["family_leakage"] = block["leakage"]
                reports[key]["ec_conditioning"] = mode if conditioned else None
                print(f"[{key}] {reports[key]['verdict']}", flush=True)
            except Exception as error:  # noqa: BLE001
                # A refusal is a result and is written rather than swallowed, and
                # the campaign continues so one unmeasurable cell does not cost
                # the rest.
                reports[key] = {
                    "arm": arm_name,
                    "ec_conditioning": mode if conditioned else None,
                    "verdict": "refused",
                    "error": f"{type(error).__name__}: {error}",
                }
                print(f"[{key}] REFUSED {type(error).__name__}: {error}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    payload = {
        "stage": "concept_lens_phase_a",
        "schema_version": 1,
        "arguments": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "cohorts": {
            str(with_ec): {
                "name": cohort.name,
                "digest": cohort.digest,
                "provenance_digest": cohort.provenance_digest,
                "sampling": cohort.metadata["sampling"],
                "family_split": sides[with_ec]["summary"],
                "family_leakage": sides[with_ec]["leakage"],
            }
            for with_ec, cohort in cohorts.items()
        },
        "declared_basis": {name: dict(values) for name, values in PROPERTY_BASIS.items()},
        "declared_text_properties": list(TEXT_PROPERTY_NAMES),
        "arms": reports,
    }
    destination = Path(args.out) / f"{args.tag}.json"
    write_json(destination, payload)
    print(f"wrote {destination}")
    print(json.dumps({name: block["verdict"] for name, block in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
