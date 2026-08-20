#!/usr/bin/env python3
"""D3.k: does a protein decoder read a kinase's catalytic machinery, or its family?

**What this stage is.** EXP-R2-214's D3.k, as amendment 1 of 2026-08-19 rewrote it.
Pseudokinases carry the kinase fold and score on the Pfam kinase models while being
experimentally catalytically dead. The amendment measured the premise that the HMM
therefore fails by construction and found it false: the bit score reaches AUROC 0.770
unmatched and 0.762 [0.695, 0.879] against each pseudokinase's nearest active relative,
and only 0.511 [0.444, 0.578] under the 20-bit caliper over 15 matched pairs. The
caliper-matched contrast is therefore the admitted cohort, and the track is a **three-way
read on one contrast**: the recombination ceiling near 0.51, the biology reference -- a
motif-aware catalytic reading, 0.922 [0.824, 0.989] -- near 0.92, and the model between.

**The readout, and why it is not either obvious one.** Scoring the domain's mean
likelihood asks whether a pseudokinase reads as less kinase-like than a bit-score-matched
kinase, which is the agreement-set trap that closed three prior tracks. Reading the
residue at the three catalytic columns is the *biology reference*, not a model readout:
knowing which columns are catalytic is imported knowledge. This stage instead forces the
three catalytic anchors into a live state (K/D/D) and into the classical experimental
kinase-dead state (K->R, D->N, D->N) and measures the difference the forcing makes to the
likelihood of **the rest of the domain**, downstream of the first forced anchor and
outside a declared radius of every one. Because both conditions overwrite the same
positions with the same residues, a record's own catalytic state cancels out of the
statistic entirely, which is what stops it being the motif reader run through a model.

A profile HMM emits columns independently, so its rho is exactly zero; a k-order
fragment conditional's is local and exactly zero beyond radius k-1; and the 20-bit
caliper matches the local context the fragment channel reads. Statistics therefore
predicts chance. A model holding the architecture rather than the family predicts that an
active kinase's domain responds more than a pseudokinase's, because the pseudokinase's
scaffold has lost the coupling. Opposite predictions on one contrast is audit §7.0 clause
4's requirement.

**The gates, in the order they bind.**

1. *Admissibility.* A text arm and an annotation-conditioned arm are refused before
   anything is loaded, and a joint language-protein rendering is refused by name: nine of
   the eighteen dead records state their own inactivity in the text a joint checkpoint
   reads, and an EC tag names kinase activity for 15 of 15 matched actives against 5 of
   18 dead. The refusal reaches the exception text, not a comment.
2. *Arm admission by measurement.* The single-residue-token coverage of the arm's own
   scored windows is measured and an arm below the declared bar is written out as
   ``NOT_MEASURABLE`` with its coverage as the reported result. Nothing is excluded by
   name.
3. *The no-fit refusal.* ``--split`` admits only ``whole_cohort``. The 15 pairs clear the
   shared unit floor only when the whole cohort decides; a 50/50 group-disjoint split
   gives 8 fit and 7 eval.
4. *The write invariant.* Forcing a downstream anchor must move an upstream likelihood by
   exactly zero. It cannot for a leftward-reading scorer, so anything else voids the run.
5. *The site-specificity control.* The identical forcing is applied at a rigid shift of
   the record's own anchor triple, over at least eight displacements, and the catalytic
   effect must exceed their 95th percentile.

**What must be cleared.** Not a shuffled null. The recombination ceiling is the UniRef50
fragment conditional under the identical forcing on the identical windows, read as a
**curve over order** with the verdict taken at the binding rung, plus the statistics
family's own natural readouts on the same labels -- Pfam bit score, nearest-active
fragment retrieval, and the active-pool composition centroid. ``k = 1`` is exactly zero by
construction and is the curve's reachability anchor.

**The counter-stratum is read separately.** ``active_despite_degradation`` (n = 8) is
experimentally active with degraded catalytic machinery, so a motif reader is wrong on
seven of the eight and the Pfam score on the eighth. A model that matches the biology
reference on the primary contrast and fails this stratum is reading motifs rather than
structure, and that is its own verdict.

``--synthetic`` runs the known-answer self-test: four planted decoders -- catalysis,
motif, corpus statistics and null -- through the identical analysis, each of which must
return a different, correct verdict.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.transfer import alphabet_chemistry as ac  # noqa: E402
from src.transfer import arms, joint_modes  # noqa: E402
from src.transfer import catalytic_contradiction as cx  # noqa: E402
from src.transfer.arms import PANEL, REPO  # noqa: E402
from src.transfer.io import sha256_file, write_json  # noqa: E402

SCHEMA_VERSION = cx.SCHEMA_VERSION
DEFAULT_OUT = REPO / "results/transfer/catalytic_contradiction"

PROVENANCE_MODULES = (
    "src/transfer/catalytic_contradiction.py",
    "src/transfer/alphabet_chemistry.py",
    "src/transfer/arms.py",
    "src/transfer/scoring.py",
    "src/transfer/statistics.py",
    "src/transfer/io.py",
)

#: Flags that name a real campaign. ``--synthetic`` requires every one to be absent and
#: they are omitted from the synthetic artefact's ``settings`` rather than echoed as
#: null, which ``32_crosscoder.py`` records the cost of.
CAMPAIGN_ONLY_FLAGS = (
    "arm",
    "cohort",
    "cohort_sha256",
    "kmer_background",
    "high_order_background",
    "ceiling_orders",
    "max_residues",
)

#: Decisions this stage refuses to default, in either mode. Each is a threshold or a
#: count that moves the answer, and a stage that supplied one silently would be reporting
#: a choice as a measurement.
PRE_REGISTERED_DECISIONS = (
    "exclusion_radius", "shift_draws", "ceiling_factor", "split", "seed",
)


def parse_orders(argument: str) -> tuple[int, ...]:
    orders = tuple(sorted({int(piece) for piece in argument.replace(" ", "").split(",") if piece}))
    if not orders:
        raise ValueError("--ceiling-orders names no order")
    outside = [order for order in orders if order not in ac.FRAGMENT_ORDERS]
    if outside:
        raise ValueError(
            f"--ceiling-orders names {outside}, outside the staged background's "
            f"{list(ac.FRAGMENT_ORDERS)}"
        )
    return orders


# ----------------------------------------------------------------- arguments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arm", default=None, choices=sorted(PANEL),
        help="the panel arm to measure. Admission is decided by the arm's declared "
        "modality and rendering and by its MEASURED single-residue-token coverage, never "
        "by this name",
    )
    parser.add_argument(
        "--joint-rendering", default=None, choices=list(joint_modes.RENDERING_NAMES),
        help="present so that the joint refusal is structural rather than promised. "
        "EXP-R2-214 amendment 1 item 2 makes D3.k inadmissible on a joint "
        "language-protein checkpoint, whose annotation channel carries the answer, and "
        "naming one here raises with that reason",
    )
    parser.add_argument(
        "--cohort", type=Path, default=None,
        help="the frozen pseudokinase contradiction set. REQUIRED on a campaign run",
    )
    parser.add_argument(
        "--cohort-sha256", default=None,
        help="SHA-256 the cohort must hash to. REQUIRED and never defaulted: a cohort "
        "rebuilt under the same filename is a different measurement, and the pin is what "
        "makes an artefact identify the records it was read on",
    )
    parser.add_argument(
        "--kmer-background", type=Path, default=None,
        help="directory of the pinned UniRef50 k-mer background. REQUIRED on a campaign "
        "run: the higher-order directory is admissible as a ceiling only because its "
        "shared orders are byte-identical to this one's, which is checked by digest",
    )
    parser.add_argument(
        "--high-order-background", type=Path, default=None,
        help="directory of the k = 1..7 UniRef50 background. REQUIRED on a campaign run",
    )
    parser.add_argument(
        "--ceiling-orders", default=None,
        help="orders of the recombination-ceiling curve, comma separated. REQUIRED and "
        "never defaulted. k = 1 must be present -- it reads no context, so its rho is "
        "exactly zero and it is the curve's own reachability anchor -- and k = 3 must be "
        "present because it is the rung EXP-R2-214 froze",
    )
    parser.add_argument(
        "--max-residues", type=int, default=None,
        help="residues of the scored window, ending at the kinase domain's last residue. "
        "REQUIRED: it sets how much upstream context the domain is read in",
    )
    parser.add_argument(
        "--exclusion-radius", type=int, default=None,
        help="residues excluded either side of every forced anchor, at which the headline "
        f"verdict is read. REQUIRED. The sweep over {list(cx.EXCLUSION_RADII)} is measured "
        "and reported whatever this is (Appendix B rule 17): a k-order fragment "
        "conditional's rho is exactly zero beyond radius k-1, so this axis is what "
        "separates a local corpus effect from propagation",
    )
    parser.add_argument(
        "--shift-draws", type=int, default=None,
        help=f"displacements of the record's own anchor triple for the site-specificity "
        f"control, at least {cx.MINIMUM_RANDOM_ANCHOR_DRAWS}. REQUIRED",
    )
    parser.add_argument(
        "--ceiling-factor", type=float, default=None,
        help="the standing §7.0 margin's multiple of the recombination ceiling's excess "
        "over chance. REQUIRED. 2.0 is the value in force for D3.g",
    )
    parser.add_argument(
        "--split", default=None, choices=list(cx.SPLIT_SIDES),
        help="the deciding side. REQUIRED, and only 'whole_cohort' is admitted: the 15 "
        "matched pairs clear the shared unit floor only when the whole cohort decides, "
        "and a 50/50 group-disjoint split gives 8 fit and 7 eval (EXP-R2-214 amendment 1, "
        "item 5). Naming a side raises rather than producing an underpowered number",
    )
    parser.add_argument("--seed", type=int, default=None, help="analysis seed. REQUIRED")
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--synthetic", action="store_true",
        help="run the known-answer self-test instead of a campaign: four planted decoders "
        "-- catalysis, motif, corpus statistics and null -- through the identical "
        "analysis, each of which must return a different, correct verdict",
    )
    parser.add_argument("--synthetic-seed", type=int, default=20260819)
    parser.add_argument(
        "--synthetic-coupling", type=float, default=0.5,
        help="nats per residue of planted catalytic coupling in the self-test",
    )
    return parser


def resolve(args: argparse.Namespace) -> None:
    """Refuse an incoherent or inadmissible request before a cohort or a model is opened."""

    missing = [flag for flag in PRE_REGISTERED_DECISIONS if getattr(args, flag) is None]
    if missing:
        raise ValueError(
            "this stage never defaults "
            + ", ".join(f"--{flag.replace('_', '-')}" for flag in missing)
            + ". Each is a pre-registered decision of EXP-R2-214 D3.k: the exclusion "
            "radius the verdict is read at, the number of shifted-site draws the "
            "site-specificity control buys, the standing margin's multiple of the "
            "recombination ceiling, the deciding side, and the analysis seed"
        )
    if args.joint_rendering is not None:
        cx.refuse_joint_annotation_channel(args.joint_rendering)
    if args.split != cx.WHOLE_COHORT:
        cx.refuse_fitted_probe(args.split, fit_units=8, eval_units=7)
    if args.shift_draws < cx.MINIMUM_RANDOM_ANCHOR_DRAWS:
        raise ValueError(
            f"--shift-draws {args.shift_draws} is below the declared "
            f"{cx.MINIMUM_RANDOM_ANCHOR_DRAWS}; a 95th percentile over fewer draws is one "
            "of a handful of order statistics"
        )
    if args.ceiling_factor < 1.0:
        raise ValueError(
            "a ceiling factor below one lets a model inside the recombination ceiling be "
            "recorded as clearing it"
        )
    if args.exclusion_radius < 0:
        raise ValueError("the exclusion radius is a number of residues and is non-negative")
    if args.exclusion_radius not in cx.EXCLUSION_RADII:
        raise ValueError(
            f"--exclusion-radius {args.exclusion_radius} is outside the declared sweep "
            f"{list(cx.EXCLUSION_RADII)}; the headline rung has to be one of the rungs the "
            "sweep reports or the sweep is not a sweep"
        )
    if args.bootstrap_draws < 1 or args.batch_size < 1:
        raise ValueError("--bootstrap-draws and --batch-size must be positive")

    if args.synthetic:
        present = [flag for flag in CAMPAIGN_ONLY_FLAGS if getattr(args, flag) is not None]
        if present:
            raise ValueError(
                ", ".join(f"--{flag.replace('_', '-')}" for flag in present)
                + " name a real campaign and are meaningless beside --synthetic, which "
                "runs the identical analysis on planted worlds whose answer is known"
            )
        if args.synthetic_coupling <= 0.0:
            raise ValueError("the planted coupling must be positive to be recoverable")
        return

    absent = [flag for flag in CAMPAIGN_ONLY_FLAGS if getattr(args, flag) is None]
    if absent:
        raise ValueError(
            "a campaign run needs "
            + ", ".join(f"--{flag.replace('_', '-')}" for flag in absent)
            + ". The cohort and its digest pin the records the artefact quotes, the two "
            "background directories carry the recombination ceiling, the ceiling orders "
            "are the curve the verdict is read at its binding rung of, and the window "
            "cap sets how much context each domain is read in"
        )
    cx.assert_sequence_only(PANEL[args.arm])
    args.ceiling_orders = parse_orders(args.ceiling_orders)
    for required, why in ((1, "the curve's zero-by-construction reachability anchor"),
                          (ac.PRE_REGISTERED_FRAGMENT_ORDER, "the rung EXP-R2-214 froze")):
        if required not in args.ceiling_orders:
            raise ValueError(
                f"--ceiling-orders must include k = {required}, {why}. The curve is an "
                "amendment that adds orders beside it and never one that substitutes for it"
            )
    if args.max_residues < 2 * cx.MINIMUM_SCORED_POSITIONS:
        raise ValueError(
            f"--max-residues {args.max_residues} cannot carry the declared "
            f"{cx.MINIMUM_SCORED_POSITIONS}-position floor with any context in front of it"
        )


# --------------------------------------------------------------- assembling


def aligned(
    rho: Mapping[str, float | None],
    positives: Sequence[cx.Record],
    negatives: Sequence[cx.Record],
) -> list[tuple[cx.Record, int]]:
    """The records one contrast is read on: those the model could score, in one order."""

    rows: list[tuple[cx.Record, int]] = []
    for records, label in ((positives, 1), (negatives, 0)):
        for record in records:
            if rho.get(record.accession) is None:
                continue
            rows.append((record, label))
    return rows


def vectors(
    rows: Sequence[tuple[cx.Record, int]], score_of: Callable[[cx.Record], float | None]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    labels: list[int] = []
    scores: list[float] = []
    groups: list[int] = []
    dropped: list[str] = []
    for record, label in rows:
        value = score_of(record)
        if value is None:
            dropped.append(record.label)
            continue
        labels.append(label)
        scores.append(float(value))
        groups.append(record.split_unit)
    return (
        np.asarray(labels),
        np.asarray(scores, dtype=np.float64),
        np.asarray(groups),
        dropped,
    )


def rho_score(rho: Mapping[str, float | None]) -> Callable[[cx.Record], float | None]:
    """The readout's orientation, in one place: a SMALLER rho predicts dead."""

    def score(record: cx.Record) -> float | None:
        value = rho.get(record.accession)
        return None if value is None else -float(value)

    return score


def measurement(
    rows: Sequence[tuple[cx.Record, int]],
    score_of: Callable[[cx.Record], float | None],
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    labels, scores, groups, dropped = vectors(rows, score_of)
    if labels.size == 0 or np.unique(labels).size < 2:
        return {
            "auroc": None,
            "interval": None,
            "n_positive": int(labels.sum()) if labels.size else 0,
            "n_negative": int(labels.size - labels.sum()) if labels.size else 0,
            "dropped": dropped,
            "withheld_reason": "the contrast does not carry both classes",
        }
    block = cx.auroc_interval(labels, scores, groups, seed=seed, draws=draws)
    block.update(
        {
            "n_positive": int(labels.sum()),
            "n_negative": int(labels.size - labels.sum()),
            "dropped": dropped,
            "degenerate": cx.is_degenerate(scores),
        }
    )
    return block


def ceiling_block(
    rows: Sequence[tuple[cx.Record, int]],
    model_score: Callable[[cx.Record], float | None],
    sources: Sequence[tuple[str, Callable[[cx.Record], float | None], dict[str, Any]]],
    *,
    factor: float,
    seed: int,
    draws: int,
    sign: int,
) -> list[dict[str, Any]]:
    """Every ceiling estimator against the model, on exactly the records both can score."""

    built: list[dict[str, Any]] = []
    for name, score_of, extra in sources:
        keep = [
            (record, label)
            for record, label in rows
            if model_score(record) is not None and score_of(record) is not None
        ]
        labels, ceiling, groups, _ = vectors(keep, score_of)
        _, model, _, _ = vectors(keep, model_score)
        if labels.size == 0 or np.unique(labels).size < 2:
            built.append({"name": name, "withheld_reason": "no two-class overlap", **extra})
            continue
        row = cx.ceiling_row(
            name,
            labels,
            sign * model,
            sign * ceiling,
            groups,
            factor=factor,
            seed=seed,
            draws=draws,
            extra={
                **extra,
                "n_records": int(labels.size),
                "adequacy": cx.ceiling_adequacy(model.tolist(), ceiling.tolist()),
            },
        )
        built.append(row)
    return built


def fragment_sources(
    ceiling_rho: Mapping[int, Mapping[str, float | None]],
    backgrounds: Mapping[int, ac.OrderedFragmentCounts],
) -> list[tuple[str, Callable[[cx.Record], float | None], dict[str, Any]]]:
    sources = []
    for order in sorted(ceiling_rho):
        sources.append(
            (
                f"uniref50_fragment_k{order}",
                rho_score(ceiling_rho[order]),
                {
                    "order": int(order),
                    "family": "same_readout",
                    "background": backgrounds[order].record(),
                    "rho_is_zero_by_construction": bool(order == 1),
                    "pre_registered_rung": bool(order == ac.PRE_REGISTERED_FRAGMENT_ORDER),
                },
            )
        )
    return sources


def natural_sources(
    retrieval: Mapping[str, Mapping[str, float]]
) -> list[tuple[str, Callable[[cx.Record], float | None], dict[str, Any]]]:
    """The statistics family's own readouts on the same labels, at the same unit.

    Audit §7.0 clause 2 asks for the best the clause-1 family achieves on this cohort
    under this readout. These rows are the strictly more demanding reading: what the
    family achieves **by any readout of its own**, which is how the 0.511 the cohort
    build measured for the Pfam score enters this stage rather than being quoted.
    """

    return [
        (
            "pfam_kinase_bit_score",
            lambda record: None if record.domain_bits is None else -float(record.domain_bits),
            {
                "family": "any_readout",
                "definition": (
                    "the Pfam kinase-domain bit score, oriented so that a lower score "
                    "predicts dead. The cohort build measures 0.511 [0.444, 0.578] for it "
                    "on the caliper-matched contrast and 0.770 unmatched"
                ),
            },
        ),
        (
            "nearest_active_fragment_retrieval",
            lambda record: -float(retrieval["nearest_active"][record.accession]),
            {
                "family": "any_readout",
                "definition": (
                    "maximum 3-mer cosine similarity of the record's kinase domain to any "
                    "active-pool domain, oriented so that a lower similarity predicts dead"
                ),
            },
        ),
        (
            "active_pool_composition_centroid",
            lambda record: -float(retrieval["active_centroid"][record.accession]),
            {
                "family": "any_readout",
                "definition": (
                    "3-mer cosine similarity of the record's kinase domain to the "
                    "active-pool centroid, oriented so that a lower similarity predicts dead"
                ),
            },
        ),
        (
            "aligned_anchor_count",
            lambda record: -float(record.n_anchors),
            {
                "family": "nuisance",
                "definition": (
                    "how many of the three catalytic columns the Pfam alignment placed at "
                    "all. Not a corpus model: it is reported because it is the one "
                    "covariate the intervention's own size depends on, and a contrast it "
                    "explains is a contrast about alignment coverage"
                ),
            },
        ),
    ]


def curve_and_rows(
    rows: Sequence[tuple[cx.Record, int]],
    rho: Mapping[str, float | None],
    ceiling_rho: Mapping[int, Mapping[str, float | None]],
    backgrounds: Mapping[int, ac.OrderedFragmentCounts],
    natural: Sequence[tuple[str, Callable[[cx.Record], float | None], dict[str, Any]]],
    *,
    factor: float,
    seed: int,
    draws: int,
    sign: int,
) -> dict[str, Any]:
    """One orientation's whole ceiling: the fragment curve and the natural readouts."""

    orders = sorted(ceiling_rho)
    common = [
        (record, label)
        for record, label in rows
        if all(ceiling_rho[order].get(record.accession) is not None for order in orders)
    ]
    labels, model_vector, groups, _ = vectors(common, rho_score(rho))
    if labels.size == 0 or np.unique(labels).size < 2:
        raise RuntimeError(
            "no record is scorable by the model and by every fragment order at once, so "
            "the ceiling curve has no common footing with the model it bounds"
        )
    by_order = {
        order: sign * vectors(common, rho_score(ceiling_rho[order]))[1] for order in orders
    }
    curve = cx.fragment_ceiling_curve(
        orders,
        backgrounds,
        by_order,
        labels,
        sign * model_vector,
        groups,
        factor=factor,
        seed=seed,
        draws=draws,
    )
    curve["n_records_common_to_every_order"] = int(labels.size)
    curve["n_records_model_scored"] = len(rows)
    any_readout = ceiling_block(
        rows, rho_score(rho), natural, factor=factor, seed=seed, draws=draws, sign=sign
    )
    return {"same_readout": curve, "any_readout": any_readout}


def all_ceiling_rows(block: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = list(block["same_readout"]["rows"])
    rows += [row for row in block["any_readout"] if row.get("family") != "nuisance" and "clears" in row]
    return rows


def analyse(
    *,
    pairs: Sequence[cx.MatchedPair],
    high_pairs: Sequence[cx.MatchedPair],
    counter: Sequence[cx.Record],
    backgrounds: Mapping[int, ac.OrderedFragmentCounts],
    model_by_radius: Mapping[int, Mapping[str, Any]],
    ceiling_by_radius: Mapping[int, Mapping[int, Mapping[str, Any]]],
    natural: Sequence[tuple[str, Callable[[cx.Record], float | None], dict[str, Any]]],
    declared_radius: int,
    factor: float,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    """The whole read, from rho tables to one combined verdict."""

    declared = model_by_radius[declared_radius]
    rho = declared["catalytic"]
    ceiling_rho = {
        order: block["catalytic"] for order, block in ceiling_by_radius[declared_radius].items()
    }
    dead = [pair.dead for pair in pairs]
    active = [pair.active for pair in pairs]
    rows = aligned(rho, dead, active)
    model = measurement(rows, rho_score(rho), seed=seed, draws=draws)

    forward = curve_and_rows(
        rows, rho, ceiling_rho, backgrounds, natural,
        factor=factor, seed=seed, draws=draws, sign=1,
    )
    reversed_block = curve_and_rows(
        rows, rho, ceiling_rho, backgrounds, natural,
        factor=factor, seed=seed, draws=draws, sign=-1,
    )
    # What share of the model's own scored positions the corpus can score at this order.
    # At k = 7 the staged background leaves 14% of its k-mers unobserved and this design
    # does not smooth, so the share is the number that says whether the ceiling is being
    # read on the same positions as the model or on an easier subset of them.
    for block in (forward, reversed_block):
        for row in block["same_readout"]["rows"]:
            fractions = [
                entry["scored_fraction"]
                for entry in ceiling_by_radius[declared_radius][row["order"]]["rows"]
            ]
            row["scored_fraction"] = {
                "mean": float(np.mean(fractions)),
                "minimum": float(min(fractions)),
                "definition": (
                    "share of the positions the model scored that this order could score "
                    "in BOTH forced conditions; the corpus is silent where it is below one"
                ),
            }

    # The site-specificity control: the same AUROC under a rigid shift of each record's
    # own anchor triple, so the number of anchors, their spacing and the scored-set shape
    # are held fixed and only the site moves.
    shifted_aurocs: list[float] = []
    for draw in declared.get("shifted", []):
        shifted_rows = aligned(draw, dead, active)
        block = measurement(shifted_rows, rho_score(draw), seed=seed, draws=draws)
        if block["auroc"] is not None:
            shifted_aurocs.append(float(block["auroc"]))
    specificity = cx.site_specificity(model["auroc"] or 0.5, shifted_aurocs)

    # Rule 40, attached to the number rather than replacing it: does this arm's rho
    # separate a real kinase domain from its own composition-preserving shuffle?
    shuffled = declared.get("shuffled", {})
    real_records = [record for record, _ in rows]
    surrogates = [
        cx.Record(**{**record.__dict__, "accession": record.accession + "_shuffled"})
        for record in real_records
        if shuffled.get(record.accession + "_shuffled") is not None
    ]
    combined_rho = {**rho, **shuffled}
    # The positive class is the SHUFFLE, so that this contrast keeps the one orientation
    # convention the whole stage runs on: a smaller rho ranks high. A permuted domain has
    # no architecture for a forced catalytic state to be coherent with, so it is the side
    # a working readout puts at the low-rho end.
    response = cx.architecture_response(
        measurement(
            aligned(combined_rho, surrogates, real_records),
            rho_score(combined_rho),
            seed=seed,
            draws=draws,
        )
    )

    counter_rows = aligned(rho, dead, list(counter))
    counter_measurement = measurement(counter_rows, rho_score(rho), seed=seed, draws=draws)

    high_rows = aligned(rho, [pair.dead for pair in high_pairs], [pair.active for pair in high_pairs])
    sensitivity = {
        "n_pairs": len(high_pairs),
        "dropped_genes": sorted(
            set(cx.MODERATE_CONFIDENCE_GENES)
            & {pair.dead.label for pair in pairs}
        ),
        "floor": cx.bootstrap_unit_floor(len(high_pairs)),
        "measurement": measurement(high_rows, rho_score(rho), seed=seed, draws=draws),
        "reading": (
            "the primary contrast with the eight moderate-confidence dead records dropped "
            "by gene symbol. A filter over the frozen cohort and never a rebuild, and it "
            "is always computed rather than selected by a flag"
        ),
    }

    biology = {
        "statistic": (
            "the number of intact catalytic columns, oriented so that fewer predicts dead"
        ),
        "side": (
            "BIOLOGY, not a ceiling row. Knowing WHICH three columns are catalytic is "
            "knowledge imported from biochemistry rather than anything buildable from "
            "co-occurrence, so audit §7.0 clause 1's family does not contain it. It is the "
            "value a reader with catalytic knowledge attains and is never a bar the model "
            "must clear (EXP-R2-214 amendment 1, item 3)"
        ),
        "defeated_by": (
            "the counter-stratum: seven of its eight members are experimentally active "
            "with degraded catalytic machinery, and PKDCC -- the eighth -- reads all three "
            "columns intact and fails on bit score instead"
        ),
        "primary": measurement(rows, lambda record: -float(record.n_intact), seed=seed, draws=draws),
        "counter_stratum": measurement(
            counter_rows, lambda record: -float(record.n_intact), seed=seed, draws=draws
        ),
    }

    sweep: dict[str, Any] = {}
    for radius in sorted(model_by_radius):
        rho_r = model_by_radius[radius]["catalytic"]
        rows_r = aligned(rho_r, dead, active)
        model_r = measurement(rows_r, rho_score(rho_r), seed=seed, draws=draws)
        ceiling_r = []
        for order, block in sorted(ceiling_by_radius[radius].items()):
            ceiling_rho_r = block["catalytic"]
            keep = [
                (record, label)
                for record, label in rows_r
                if ceiling_rho_r.get(record.accession) is not None
            ]
            measured = measurement(keep, rho_score(ceiling_rho_r), seed=seed, draws=draws)
            ceiling_r.append(
                {
                    "order": int(order),
                    "auroc": measured["auroc"],
                    "adequacy": cx.ceiling_adequacy(
                        [value for value in rho_r.values() if value is not None],
                        [value for value in ceiling_rho_r.values() if value is not None],
                    ),
                }
            )
        sweep[str(radius)] = {
            "model_auroc": model_r["auroc"],
            "model_interval": model_r["interval"],
            "ceiling_by_order": ceiling_r,
            "n_records": len(rows_r),
        }

    primary = cx.primary_verdict(
        degenerate=bool(model.get("degenerate")),
        invariant=declared["invariant"],
        measurement=model,
        ceiling_rows=all_ceiling_rows(forward),
        reversed_rows=all_ceiling_rows(reversed_block),
        specificity=specificity,
        response=response,
    )
    counter_verdict = cx.counter_stratum_verdict(
        degenerate=bool(counter_measurement.get("degenerate")), measurement=counter_measurement
    )
    return {
        "primary_contrast": {
            "declared_radius": int(declared_radius),
            "n_pairs": len(pairs),
            "measurement": model,
            "ceiling": {"toward_experiment": forward, "reversed": reversed_block},
            "site_specificity": specificity,
            "architecture_response": response,
            "rho_rows": declared["rows"],
            "dropped": declared["dropped"],
            "verdict": primary,
        },
        "sensitivity_high_confidence_only": sensitivity,
        "counter_stratum": {
            "n_records": len(counter),
            "measurement": counter_measurement,
            "verdict": counter_verdict,
            "reading": (
                "a second, orthogonal contradiction set with its own verdict rather than a "
                "robustness check on the first (EXP-R2-214 amendment 1, item 4)"
            ),
        },
        "biology_reference": biology,
        "exclusion_radius_sweep": sweep,
        "invariants": declared["invariant"],
        "verdict": cx.combined_verdict(primary, counter_verdict),
    }


# ---------------------------------------------------------------- the artefact


def artefact_name(kind: str, arm: str, cohort_digest: str, seed: int) -> str:
    """Basename from arm, cohort identity, intervention and seed -- never a fixed string.

    ``21_joint_mode_qualification.py`` writes every run to one fixed name, so a second
    checkpoint in one output directory overwrites the first without a word. All four of
    this stage's campaign axes are in the name, and the cohort digest is one of them
    because a rebuilt cohort under the same filename is a different measurement.
    """

    def safe(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", value)

    return (
        f"catalytic_contradiction__{safe(kind)}__{safe(arm)}__{safe(cohort_digest)[:12]}"
        f"__{safe(cx.INTERVENTION)}__seed{int(seed)}.json"
    )


def provenance() -> dict[str, Any]:
    return {
        "runner": {
            "path": "scripts/transfer/40_catalytic_contradiction.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "modules": {name: sha256_file(REPO_ROOT / name) for name in PROVENANCE_MODULES},
    }


def pre_registration_block() -> dict[str, Any]:
    return {
        "record": cx.PRE_REGISTRATION,
        "track": cx.PRE_REGISTRATION_TRACK,
        "amendments_implemented": list(cx.PRE_REGISTRATION_AMENDMENTS),
        "scope": cx.PRE_REGISTRATION_SCOPE,
        "governing_rule": "audit §7.0, the recombination ceiling as a standing admission rule",
        "required_flags": list(PRE_REGISTERED_DECISIONS),
        "estimand": (
            "AUROC of -rho separating experimentally dead kinases from their 20-bit "
            "caliper-matched active controls, where rho(x) = [NLL of the domain "
            "downstream of the first forced anchor and outside the exclusion radius, with "
            "the three catalytic anchors forced to the experimental kinase-dead state] "
            "minus [the same with them forced to the live catalytic state], per residue. "
            "Both conditions overwrite the same positions with the same residues, so the "
            "record's own catalytic state does not enter the statistic"
        ),
        "live_state": dict(cx.LIVE_STATE),
        "dead_state": dict(cx.DEAD_STATE),
        "state_source": cx.STATE_SOURCE,
        "what_each_account_predicts": {
            "evolutionary_statistics": (
                "chance. A profile HMM emits columns independently so its rho is exactly "
                "zero; a k-order fragment conditional's rho is local and exactly zero "
                "beyond radius k-1; and the 20-bit caliper matches the local context the "
                "fragment channel reads between a pseudokinase and its control"
            ),
            "catalytic_knowledge": (
                "an active kinase's domain responds more to the forced state than a "
                "pseudokinase's, because a pseudokinase's scaffold has lost the coupling "
                "that a working catalytic loop, activation segment and alphaC helix "
                "maintain around a functioning K/D/D triad"
            ),
            "why_they_differ": (
                "the statistics family factorises, so a forced residue can only reach "
                "positions inside its own window; the catalytic account is a claim about "
                "long-range coupling, which is exactly what the exclusion-radius axis "
                "separates"
            ),
        },
        "biology_reference": (
            "the motif-aware catalytic reading, AUROC 0.922 [0.824, 0.989] on this cohort. "
            "Reported beside the result as the value a reader with catalytic knowledge "
            "attains and never as a bar the model must clear"
        ),
        "resolvable_auroc": cx.RESOLVABLE_AUROC,
        "minimum_bootstrap_units": int(cx.MINIMUM_BOOTSTRAP_UNITS),
        "minimum_random_anchor_draws": int(cx.MINIMUM_RANDOM_ANCHOR_DRAWS),
        "exclusion_radii": list(cx.EXCLUSION_RADII),
        "dtype": cx.DTYPE,
        "deciding_side": cx.WHOLE_COHORT,
        "leakage": cx.LEAKAGE_CLAUSE,
        "per_layer_quantities": (
            "none. The intervention is on the input sequence and the readout is a "
            "likelihood, so this stage has no layer axis to report per layer or to average "
            "over"
        ),
    }


LIMITATIONS: dict[str, Any] = {
    "the_experimental_label_is_not_externally_certified": (
        "EXP-R2-214 required the catalytically-dead label to come from a named published "
        "experimental compilation pinned by digest under external_resources/. That is not "
        "what exists: the label is curated from the primary literature by the agent that "
        "wrote the cohort build, on a host with no route to a citation database (manifest "
        "limitation L-PK-1). Every result here is conditional on that curation, the "
        "per-record label_provenance, label_evidence, label_citation and label_confidence "
        "fields are the audit trail, and an independent check of the labels is an open "
        "requirement rather than a discharged one. It is surfaced here rather than "
        "inherited quietly"
    ),
    "the_unit_count_does_not_improve_with_effort": (
        "15 matched pairs, bounded by the number of human genes with published catalysis "
        "experiments, which is of order twenty. Orthologues add records while joining the "
        "same near-duplicate group (manifest limitation L-PK-3). The realised intervals "
        f"are about +/-0.07, so only separations of roughly {cx.RESOLVABLE_AUROC} are "
        "resolvable; that is adequate here only because the two accounts predict 0.51 and "
        "0.92, and it would not be adequate for a subtler contrast"
    ),
    "a_potts_or_msa_coupling_ceiling_is_named_and_not_run": (
        "the readout measures long-range coupling between the catalytic anchors and the "
        "rest of the domain, and the fragment family staged here reads at most seven "
        "residues of context, so it CANNOT express that coupling. A Potts or MSA coupling "
        "model could, and audit §7.0 clause 1 puts it squarely on the ceiling side. It is "
        "not run: no such model is staged, and building one needs a kinase-family "
        "alignment this stage does not construct. This is the one ceiling row that is "
        "named and open rather than measured, and a CLEARS verdict here is bounded by it "
        "-- it means the model exceeds what FRAGMENT and RETRIEVAL statistics achieve, not "
        "what every member of the clause-1 family achieves"
    ),
    "the_window_is_truncated_at_the_n_terminus": (
        "a record longer than the declared window is cut at its N terminus, which is "
        "off-distribution for a model trained on whole sequences. It is cut identically in "
        "both forced conditions and rho is a within-record difference, so the truncation "
        "cancels to first order; what it does not cancel is any interaction between "
        "truncation and the coupling being measured, and that is not bounded here"
    ),
    "the_intervention_size_varies_with_alignment_coverage": (
        "a record whose Pfam alignment leaves a catalytic column unplaced carries two "
        "forced anchors rather than three, and rho is a per-residue mean of a difference "
        "driven by however many there are. The count is carried per record and its own "
        "AUROC is reported as a nuisance row, because a contrast that row explains is a "
        "contrast about alignment coverage rather than about catalysis"
    ),
    "cross_arm_magnitudes_are_not_comparable": (
        "rho is in nats per residue on a residue-tokenised arm and this stage admits no "
        "other kind, so magnitudes are comparable within that class and no further (L23). "
        "The verdict is an AUROC, which is a rank statistic and is not affected"
    ),
    "no_split_is_drawn": (
        "nothing is fitted, so there is no train/test split to make group-disjoint. The "
        "cohort's near-duplicate structure enters through the resampling unit -- the "
        "matched pair, which the cohort build merged into one split unit so a pair is "
        "never divided -- and not through a split of this stage's own"
    ),
    "pre_registration_is_not_admission": cx.PRE_REGISTRATION_SCOPE,
    "leakage_clause": cx.LEAKAGE_CLAUSE,
}

SYNTHETIC_LIMITATIONS: dict[str, Any] = {
    "the_world_is_a_trigram_corpus_with_a_planted_gate": (
        "every planted decoder shares one base -- the exact fragment conditional of the "
        "synthetic world's own corpus -- and the catalysis and motif plantings add a gate "
        "that fires where the forced anchors are live and the planted feature is set. That "
        "exercises the whole analysis end to end: the forcing, the scored-position rule, "
        "the write invariant, the ceiling curve, the margin, both contrasts and all four "
        "verdicts. It certifies the analysis and the write, not that a thirty-six-block "
        "decoder behaves like the gate"
    ),
    "the_ceiling_here_is_the_generating_distribution": (
        "the synthetic sequences are sampled from the declared trigram table, so the "
        "fragment conditional built on that table is the EXACT corpus-statistics predictor "
        "of this world rather than an estimate of it. That is why the statistics planting "
        "lands inside its own ceiling by construction rather than by tuning -- and it is "
        "also why the coverage and observations-per-k-mer figures in this artefact "
        "describe a synthetic corpus and say nothing about UniRef50"
    ),
    "no_checkpoint_is_loaded": (
        "no arm, no tokeniser and no residue-to-token map is exercised here. The map is "
        "verified against the rendered window on every real cell instead, and it raises "
        "rather than returning a plausible number at the wrong positions"
    ),
}


# ---------------------------------------------------------------------- cells


def base_payload(args: argparse.Namespace, *, kind: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "pre_registration": pre_registration_block(),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
            if not (args.synthetic and key in CAMPAIGN_ONLY_FLAGS)
        },
        "provenance": provenance(),
    }


def run_synthetic(args: argparse.Namespace) -> dict[str, Any]:
    """Four planted decoders through the identical analysis, each with a known verdict."""

    started = time.time()
    backgrounds = cx.synthetic_background(args.synthetic_seed)
    pairs, counter = cx.synthetic_cohort(backgrounds, seed=args.synthetic_seed)
    worlds: dict[str, Any] = {}
    for planting in cx.PLANTINGS:
        likelihood = cx.PlantedLikelihood(planting, backgrounds, coupling=args.synthetic_coupling)
        model_by_radius: dict[int, Any] = {}
        ceiling_by_radius: dict[int, dict[int, Any]] = {}
        for radius in cx.EXCLUSION_RADII:
            design = cx.Design(
                pairs=pairs,
                counter=counter,
                radius=radius,
                max_residues=cx.SYNTHETIC_WINDOW,
                seed=args.seed,
                shift_draws=args.shift_draws,
            )
            declared = radius == args.exclusion_radius
            model_by_radius[radius] = cx.measure(
                likelihood, design, with_shifts=declared, with_shuffle=declared
            )
            ceiling_by_radius[radius] = {
                order: cx.measure(
                    cx.FragmentLikelihood(backgrounds[order]),
                    design,
                    with_shifts=False,
                    with_shuffle=False,
                )
                for order in cx.SYNTHETIC_ORDERS
            }
        worlds[planting] = analyse(
            pairs=pairs,
            high_pairs=pairs,
            counter=counter,
            backgrounds=backgrounds,
            model_by_radius=model_by_radius,
            ceiling_by_radius=ceiling_by_radius,
            natural=[],
            declared_radius=args.exclusion_radius,
            factor=args.ceiling_factor,
            seed=args.seed,
            draws=args.bootstrap_draws,
        )
    recovered = {
        planting: worlds[planting]["verdict"]["verdict"] for planting in cx.PLANTINGS
    }
    expected = dict(cx.EXPECTED_SYNTHETIC_VERDICT)
    passed = recovered == expected
    return {
        "worlds": worlds,
        "certificate": {
            "expected": expected,
            "recovered": recovered,
            "recovered_every_verdict": bool(passed),
            "distinct_verdicts": len(set(recovered.values())),
            "reason": (
                "the four plantings are the four ways this readout can be wrong and each "
                "has a different remedy: catalytic knowledge, a motif reading that survives "
                "the primary contrast and dies on the counter-stratum, a corpus-statistical "
                "reading that lands inside its own ceiling, and an intervention that moved "
                "nothing. A self-test that only checked thresholds would pass on a pipeline "
                "that cannot reach three of them"
            ),
        },
        "limitations": SYNTHETIC_LIMITATIONS,
        "cost": {"wall_seconds": round(time.time() - started, 1)},
    }


def rendered_windows(arm: Any, records: Sequence[cx.Record], *, max_residues: int) -> list[str]:
    """The arm's own rendering of exactly the windows rho will be read on."""

    windows: list[str] = []
    for record in records:
        start, end = cx.window_bounds(record, max_residues=max_residues)
        windows.append(record.sequence[start - 1 : end])
    cohort = arms.Cohort(
        name="catalytic_contradiction_coverage",
        kind="protein",
        records=windows,
        min_symbols=min(len(window) for window in windows),
        max_symbols=max(len(window) for window in windows),
    )
    return cohort.input_strings(arm)


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    spec = PANEL[args.arm]
    admissibility = cx.assert_sequence_only(spec)
    cohort = cx.load_cohort(args.cohort, sha256=args.cohort_sha256)
    feasibility = cohort.manifest["feasibility"]
    cx.refuse_fitted_probe(
        args.split,
        fit_units=int(feasibility["matched_pairs_per_split_side"]["fit"]),
        eval_units=int(feasibility["matched_pairs_per_split_side"]["eval"]),
    )
    pairs = cx.matched_pairs(cohort)
    high_pairs = cx.matched_pairs(cohort, high_confidence_only=True)
    counter = cohort.by_stratum(cx.COUNTER_STRATUM)
    pool = cohort.by_stratum("active_pool")
    records = tuple(record for pair in pairs for record in (pair.dead, pair.active)) + counter

    body: dict[str, Any] = {
        "arm": {
            "name": spec.name,
            "modality": spec.modality,
            "architecture": spec.architecture,
            "tokenisation": spec.tokenisation,
            "input_format": spec.input_format,
            "pretraining_corpus": spec.pretraining_corpus,
        },
        "admissibility": admissibility,
        "cohort": {
            "path": str(args.cohort),
            "sha256": cohort.sha256,
            "n_records": len(cohort.records),
            "n_matched_pairs": len(pairs),
            "n_high_confidence_pairs": len(high_pairs),
            "n_counter_stratum": len(counter),
            "n_active_pool": len(pool),
            "contested_held_out": sorted(cx.CONTESTED_GENES),
            "moderate_confidence": sorted(cx.MODERATE_CONFIDENCE_GENES),
            "feasibility": feasibility,
            "hmm_baseline_from_the_build": cohort.manifest["contradiction"]["hmm_baseline"],
            "motif_baseline_from_the_build": cohort.manifest["contradiction"]["motif_baseline"],
        },
        "limitations": LIMITATIONS,
    }

    arm = arms.load_arm(args.arm, device=args.device, dtype=cx.DTYPE)
    alphabet = ac.protein_alphabet(arm)
    windows = rendered_windows(arm, records, max_residues=args.max_residues)
    coverage = ac.symbol_token_coverage(
        arm, windows, alphabet=alphabet, max_len=args.max_residues + 64
    )
    admission = ac.admit_arm(coverage, spec.name, minimum=ac.MINIMUM_SYMBOL_TOKEN_COVERAGE)
    body["admission"] = {**admission, "coverage": coverage}
    if not admission["admitted"]:
        body["verdict"] = {
            "verdict": "NOT_MEASURABLE",
            "reason": admission["reason"],
        }
        body["cost"] = {"wall_seconds": round(time.time() - started, 1)}
        return body

    backgrounds = ac.load_ordered_counts(
        args.high_order_background, args.ceiling_orders, pinned=args.kmer_background
    )
    likelihood = cx.ArmLikelihood(arm, batch_size=args.batch_size)
    model_by_radius: dict[int, Any] = {}
    ceiling_by_radius: dict[int, dict[int, Any]] = {}
    for radius in cx.EXCLUSION_RADII:
        design = cx.Design(
            pairs=pairs,
            counter=counter,
            radius=radius,
            max_residues=args.max_residues,
            seed=args.seed,
            shift_draws=args.shift_draws,
        )
        declared = radius == args.exclusion_radius
        model_by_radius[radius] = cx.measure(
            likelihood, design, with_shifts=declared, with_shuffle=declared
        )
        ceiling_by_radius[radius] = {
            order: cx.measure(
                cx.FragmentLikelihood(backgrounds[order]),
                design,
                with_shifts=False,
                with_shuffle=False,
            )
            for order in args.ceiling_orders
        }
    retrieval = cx.retrieval_scores(records, pool)
    body.update(
        analyse(
            pairs=pairs,
            high_pairs=high_pairs,
            counter=counter,
            backgrounds=backgrounds,
            model_by_radius=model_by_radius,
            ceiling_by_radius=ceiling_by_radius,
            natural=natural_sources(retrieval),
            declared_radius=args.exclusion_radius,
            factor=args.ceiling_factor,
            seed=args.seed,
            draws=args.bootstrap_draws,
        )
    )
    body["cost"] = {**likelihood.cost(), "wall_seconds": round(time.time() - started, 1)}
    return body


def main() -> None:
    args = build_parser().parse_args()
    resolve(args)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        payload = {**base_payload(args, kind="synthetic_known_answer"), **run_synthetic(args)}
        destination = args.out / artefact_name(
            "synthetic", "planted", str(args.synthetic_seed), args.seed
        )
        write_json(destination, payload)
        certificate = payload["certificate"]
        for planting in cx.PLANTINGS:
            print(f"[{planting}] {certificate['recovered'][planting]}")
        print(f"recovered every verdict: {certificate['recovered_every_verdict']}")
        print(f"wrote {destination}")
        return

    body = run_campaign(args)
    payload = {**base_payload(args, kind="campaign_cell"), **body}
    destination = args.out / artefact_name(
        "cell", args.arm, payload["cohort"]["sha256"], args.seed
    )
    write_json(destination, payload)
    verdict = payload["verdict"]
    print(f"[{args.arm}] {verdict['verdict']}")
    if "primary_contrast" in payload:
        primary = payload["primary_contrast"]
        print(f"  primary AUROC: {primary['measurement']['auroc']} {primary['measurement']['interval']}")
        print(f"  primary verdict: {primary['verdict']['verdict']}")
        print(f"  counter-stratum: {payload['counter_stratum']['verdict']['verdict']}")
        print(f"  biology reference: {payload['biology_reference']['primary']['auroc']}")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
