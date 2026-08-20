#!/usr/bin/env python3
"""D3.l: does the continuation follow the sequence neighbour or the fold neighbour?

**What this stage is.** EXP-R2-214's third track, read on the composition-matched
/ fold-discordant triple set amendment 2 (D3.l) admitted. Every admitted record
names a **different** partner under evolutionary statistics and under structure,
which is §7.0 clause 4's requirement and the thing F10, F12 and D3.g's stage 35
all lacked. The design and its three confounds are set out in
``src/transfer/fold_discordance.py``; this file is the run.

**The order the gates bind, and it is structural rather than promised.**

1. *The cohort is the pinned one.* ``load_cohort`` refuses any file that does not
   hash to the digest amendment 2 froze, and re-derives the prefix rule against
   every stored ``anchor_prefix`` before applying it to the partners.
2. *Arm admission, by measurement.* The estimand differences the same residue
   position under two contexts, so it is defined only where a continuation
   tokenises to one token per residue **and keeps that tokenisation when it is
   spliced onto the prefix**. The measured alignment is the reported result for an
   arm below the floor, and nothing is computed behind that gate. An arm whose
   rendering needs a conditioning label the cohort does not define is refused on
   the same footing, with the coverage figure rather than a name.
3. *The ceiling curve's own reachability anchor.* k = 1 reads no context, so its
   prefix advantage is exactly zero at every position and its contrast is exactly
   zero. A curve whose first point is not exactly zero is an indexing defect and
   the run stops there.
4. *The nulls.* A contrast that survives a random relabelling of the two
   candidates, or one that survives a composition-preserving permutation of the
   anchor's prefix, is voided rather than reported.

**What must be cleared.** Not a shuffled null -- under §7.0 clearing one admits
nothing. The recombination ceiling is the UniRef50 fragment conditional at every
staged order plus the prefix-adapted composition channel, and the verdict is read
at the **binding** member of that family, never the friendliest. The adequacy
ratio of each member is reported beside the verdict, because a ceiling doing 2%
of the arm's own work turns "at least twice the ceiling" into "greater than zero"
and that is what happened to D3.j's frozen k = 3 rung.

Every pre-registered decision is a required flag with no default, and
``--synthetic`` runs the known-answer self-test: a decoder that follows the
prefix's composition, one that follows the prefix's fold arrangement, and one
that reads no context at all.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import alphabet_chemistry as ac  # noqa: E402
from src.transfer import arms, fold_discordance as fdisc  # noqa: E402
from src.transfer.arms import PANEL, REPO  # noqa: E402
from src.transfer.io import sha256_file, write_json  # noqa: E402

SCHEMA_VERSION = "r2_transfer_fold_discordance_v1"
DEFAULT_OUT = REPO / "results/transfer/fold_discordance"
DEFAULT_COHORT = REPO / "results/transfer/composition_matched_fold_set/composition_matched_fold_set.jsonl"

PROVENANCE_MODULES = (
    "src/transfer/fold_discordance.py",
    "src/transfer/alphabet_chemistry.py",
    "src/transfer/arms.py",
    "src/transfer/kmer_background.py",
    "src/transfer/statistics.py",
    "src/transfer/io.py",
)

#: Decisions this stage refuses to default, in either mode. Each moves the
#: answer, and a stage that supplied one silently would be reporting a choice as
#: a measurement.
PRE_REGISTERED_DECISIONS = (
    "window",
    "junction_offset",
    "min_window",
    "resampling_unit",
    "ceiling_orders",
    "ceiling_factor",
    "seed",
)

#: Flags that name a real campaign. ``--synthetic`` requires every one to be
#: absent and they are omitted from the synthetic artefact's ``settings`` rather
#: than echoed as null.
CAMPAIGN_ONLY_FLAGS = (
    "arm",
    "cohort",
    "kmer_background",
    "high_order_background",
    "hmmer_bin",
    "pfam_hmm",
    "corpus_fasta",
    "profile_workdir",
)


def parse_orders(argument: str, *, name: str) -> tuple[int, ...]:
    orders = tuple(sorted({int(piece) for piece in argument.replace(" ", "").split(",") if piece}))
    if not orders:
        raise ValueError(f"{name} names no order")
    outside = [order for order in orders if order not in fdisc.FRAGMENT_ORDERS]
    if outside:
        raise ValueError(
            f"{name} names {outside}, outside the staged background's {list(fdisc.FRAGMENT_ORDERS)}"
        )
    return orders


# ----------------------------------------------------------------- arguments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", default=None, choices=sorted(PANEL),
        help="the panel arm to measure. Admission is decided by the measured "
        "token alignment of the splice and by whether the cohort defines the "
        "conditioning label the rendering needs, never by this name",
    )
    parser.add_argument(
        "--cohort", type=Path, default=None,
        help=f"the staged triple set. REQUIRED on a campaign run; it must hash to "
        f"{fdisc.COHORT_DIGEST}, the digest EXP-R2-214 amendment 2 froze",
    )
    parser.add_argument(
        "--kmer-background", type=Path, default=None,
        help="directory of the pinned UniRef50 k-mer background. REQUIRED on a "
        "campaign run: the high-order vectors are admissible as a ceiling only "
        "because their shared orders are byte-identical to this one's",
    )
    parser.add_argument(
        "--high-order-background", type=Path, default=None,
        help="directory of the k = 1..7 UniRef50 background. REQUIRED on a campaign "
        "run. The retired uniref50_line_local_superseded_20260812 directory is "
        "refused by name inside the loader",
    )
    parser.add_argument(
        "--hmmer-bin", type=Path, default=None,
        help="bin directory of a built HMMER 3.4. REQUIRED on a campaign run: §7.0 "
        "clause 1's profile-HMM member is what separates structural knowledge from "
        "remote homology on this cohort, and the structure partner shares the "
        "anchor's CATH superfamily by construction. Build it from the staged "
        "external_resources/tools/hmmer-3.4.tar.gz; every binary is hashed into the "
        "artefact",
    )
    parser.add_argument(
        "--pfam-hmm", type=Path, default=None,
        help="the pressed Pfam-A.hmm. REQUIRED on a campaign run. Pfam profiles are "
        "curated over UniProt and are not built from this cohort, so this member "
        "carries no circularity at all",
    )
    parser.add_argument(
        "--corpus-fasta", type=Path, default=None,
        help="the protein corpus the second profile member searches from each "
        "anchor's prefix. REQUIRED on a campaign run. Covers the prefixes Pfam does "
        "not annotate, and its recruitment counts are the most direct measurement of "
        "the remote-homology account this stage makes",
    )
    parser.add_argument(
        "--profile-workdir", type=Path, default=None,
        help="working directory for the profile members' HMMER intermediates. "
        "REQUIRED on a campaign run and must lie outside the repository: a jackhmmer "
        "alignment per triple is not version-controlled evidence",
    )
    parser.add_argument("--hmmer-cpu", type=int, default=8)
    parser.add_argument("--profile-parallel", type=int, default=12)
    parser.add_argument(
        "--window", default=None, choices=sorted(fdisc.WINDOWS),
        help="the length reading the headline verdict is taken at. 'raw' gives each "
        "candidate its own positions; 'length_controlled' truncates both to the "
        "shorter candidate, so the two means cover the same count of positions and "
        "the same indices. Both are always measured and reported",
    )
    parser.add_argument(
        "--junction-offset", type=int, default=None,
        help="residues after the splice at which the scored window begins. Never "
        "defaulted: splicing a foreign continuation onto a prefix creates a "
        "boundary that exists in neither protein, and a Markov ceiling of order k "
        "carries information across exactly k - 1 positions, so this flag decides "
        "whether the fragment ceiling reaches the window at all",
    )
    parser.add_argument(
        "--min-window", type=int, default=None,
        help="scored residues a triple must retain after the offset and the length "
        "control, below which it is dropped with its count reported",
    )
    parser.add_argument(
        "--resampling-unit", default=None, choices=sorted(fdisc.RESAMPLING_UNITS),
        help="the unit the interval is taken over. Both are always computed and "
        "reported; this names the one the verdict is read at",
    )
    parser.add_argument(
        "--ceiling-orders", default=None,
        help="orders of the recombination-ceiling curve, comma separated, e.g. "
        "1,2,3,4,5,6,7. Never defaulted. k = 1 reads no context and its contrast is "
        "exactly zero, which is the curve's own reachability anchor; k = 3 is "
        "EXP-R2-214's frozen rung and is kept in the table whatever else is asked for",
    )
    parser.add_argument(
        "--ceiling-factor", type=float, default=None,
        help="the standing §7.0 margin's multiple of the recombination ceiling. "
        "2.0 is the value in force for D3.g",
    )
    parser.add_argument("--seed", type=int, default=None, help="analysis seed")
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--synthetic", action="store_true",
        help="run the known-answer self-test instead of a campaign: a decoder that "
        "follows the prefix's composition, one that follows the prefix's fold "
        "arrangement, and one that reads no context",
    )
    parser.add_argument("--synthetic-seed", type=int, default=20260819)
    return parser


def resolve(args: argparse.Namespace) -> None:
    """Refuse an incoherent request before a cohort is opened or a model is loaded."""

    missing = [flag for flag in PRE_REGISTERED_DECISIONS if getattr(args, flag) is None]
    if missing:
        raise ValueError(
            "this stage never defaults "
            + ", ".join(f"--{flag.replace('_', '-')}" for flag in missing)
            + ". Each is a pre-registered decision of EXP-R2-214 D3.l: the length "
            "reading the verdict is taken at, where the scored window begins after "
            "the splice, the smallest window a triple may keep, the resampling unit, "
            "the orders of the ceiling curve, the standing margin's factor and the "
            "analysis seed"
        )
    if args.synthetic:
        present = [flag for flag in CAMPAIGN_ONLY_FLAGS if getattr(args, flag) is not None]
        if present:
            raise ValueError(
                ", ".join(f"--{flag.replace('_', '-')}" for flag in present)
                + " name a real campaign and are meaningless beside --synthetic, which "
                "runs the same analysis on worlds whose answer is known"
            )
    else:
        absent = [flag for flag in CAMPAIGN_ONLY_FLAGS if getattr(args, flag) is None]
        if absent:
            raise ValueError(
                "a campaign run needs "
                + ", ".join(f"--{flag.replace('_', '-')}" for flag in absent)
                + ". The cohort is the pinned contradiction set, and the two "
                "backgrounds carry the recombination ceiling"
            )
        if args.profile_parallel < 1 or args.hmmer_cpu < 1:
            raise ValueError("--profile-parallel and --hmmer-cpu must be positive")
        if PANEL[args.arm].modality != "protein":
            raise ValueError(
                f"{args.arm} is a {PANEL[args.arm].modality} arm. This estimand is a "
                "contrast between two protein continuations under a protein prefix; it "
                "has no text instantiation, so a text arm is not a control here and is "
                "refused rather than run and reported"
            )
    args.ceiling_orders = parse_orders(args.ceiling_orders, name="--ceiling-orders")
    if fdisc.REACHABILITY_ORDER not in args.ceiling_orders:
        raise ValueError(
            f"--ceiling-orders must include k = {fdisc.REACHABILITY_ORDER}. It reads no "
            "context, so its contrast is exactly zero by construction and it is the "
            "curve's only check that the whole curve is indexed correctly"
        )
    if fdisc.PRE_REGISTERED_FRAGMENT_ORDER not in args.ceiling_orders:
        raise ValueError(
            f"--ceiling-orders must include k = {fdisc.PRE_REGISTERED_FRAGMENT_ORDER}, the "
            "rung EXP-R2-214 froze. The curve is an amendment that adds orders beside it "
            "and never one that substitutes for it"
        )
    if args.ceiling_factor < 1.0:
        raise ValueError(
            "a ceiling factor below one lets an arm inside the recombination ceiling be "
            "recorded as clearing it"
        )
    if args.junction_offset < 0:
        raise ValueError("--junction-offset counts residues after the splice and cannot be negative")
    if args.min_window < 1:
        raise ValueError("--min-window must be positive")
    if args.batch_size < 1 or args.bootstrap_draws < 1:
        raise ValueError("--batch-size and --bootstrap-draws must be positive")


# ------------------------------------------------------------------ measuring


def shuffled_prefix(prefix: str, *, seed: int, index: int) -> str:
    """A composition-preserving permutation of the anchor's prefix.

    The null this feeds is confound 3's answer on the model side: the shuffle keeps
    every residue the prefix contains and destroys every arrangement it has, so a
    contrast that survives it is a contrast about composition and not about fold.
    """

    rng = np.random.default_rng([seed, index])
    residues = np.asarray(list(prefix))
    rng.shuffle(residues)
    return "".join(residues.tolist())


def build_legs(
    triples: Sequence[fdisc.Triple], *, seed: int
) -> tuple[list[fdisc.ScoringRequest], dict[str, list[int]], list[str]]:
    """Every leg the arm has to score, and where each triple's legs live."""

    legs: list[fdisc.ScoringRequest] = []
    index: dict[str, list[int]] = {"conditioned": [], "free": [], "shuffled": []}
    shuffles: list[str] = []
    for position, triple in enumerate(triples):
        permuted = shuffled_prefix(triple.prefix, seed=seed, index=position)
        shuffles.append(permuted)
        for label in fdisc.CANDIDATES:
            continuation = triple.continuations[label]
            index["conditioned"].append(len(legs))
            legs.append(fdisc.ScoringRequest(triple.prefix, continuation))
        for label in fdisc.CANDIDATES:
            continuation = triple.continuations[label]
            index["free"].append(len(legs))
            legs.append(fdisc.ScoringRequest("", continuation))
        for label in fdisc.CANDIDATES:
            continuation = triple.continuations[label]
            index["shuffled"].append(len(legs))
            legs.append(fdisc.ScoringRequest(permuted, continuation))
    return legs, index, shuffles


def per_triple_advantages(
    triples: Sequence[fdisc.Triple],
    scores: Sequence[np.ndarray],
    index: Mapping[str, Sequence[int]],
) -> dict[str, list[dict[str, np.ndarray]]]:
    """Prefix advantage per residue, per candidate, under the real and shuffled prefix."""

    advantage: dict[str, list[dict[str, np.ndarray]]] = {"real": [], "shuffled": [], "conditional": []}
    for position, triple in enumerate(triples):
        real: dict[str, np.ndarray] = {}
        shuffled: dict[str, np.ndarray] = {}
        conditional: dict[str, np.ndarray] = {}
        for offset, label in enumerate(fdisc.CANDIDATES):
            slot = 2 * position + offset
            conditioned = scores[index["conditioned"][slot]]
            free = scores[index["free"][slot]]
            permuted = scores[index["shuffled"][slot]]
            expected = len(triple.continuations[label])
            for name, vector in (("conditioned", conditioned), ("free", free), ("shuffled", permuted)):
                if vector.shape != (expected,):
                    raise RuntimeError(
                        f"{triple.anchor} {label}: the {name} pass scored "
                        f"{vector.shape} positions and the continuation has {expected}"
                    )
            real[label] = conditioned - free
            shuffled[label] = permuted - free
            conditional[label] = conditioned
        advantage["real"].append(real)
        advantage["shuffled"].append(shuffled)
        advantage["conditional"].append(conditional)
    return advantage


def windowed_contrast(
    values: Mapping[str, np.ndarray], window: Mapping[str, np.ndarray]
) -> float:
    """``mean over the structure partner's window - mean over the sequence partner's``."""

    means = {}
    for label in fdisc.CANDIDATES:
        selected = np.asarray(values[label])[np.asarray(window[label], dtype=np.int64)]
        if selected.size == 0:
            raise ValueError(f"the {label} window is empty; a triple with no window is dropped upstream")
        means[label] = float(selected.mean())
    return means["structure_partner"] - means["sequence_partner"]


def ceiling_members(
    ordered: Mapping[int, ac.OrderedFragmentCounts],
    orders: Sequence[int],
    profiles: Sequence[fdisc.ProfileHomologyMember] = (),
) -> dict[str, Any]:
    """The clause-1 family this cohort admits, keyed by the name it is reported under.

    Three kinds, and the third is the one that decides the verdict. The fragment
    conditional reads ``k - 1`` residues of context; the prefix-adapted composition
    channel reads the prefix's residue counts; the profile members read the
    prefix's **family**, which is the only one of the three that can detect the
    remote homology this cohort's construction guarantees between an anchor and its
    structure partner. §7.0 clause 2's ceiling is the best any member achieves, so
    all three sit in one family and the verdict is read at the binding one.
    """

    members: dict[str, Any] = {
        str(order): fdisc.FragmentPrefixConditional(ordered, order=order) for order in orders
    }
    members[fdisc.PREFIX_COMPOSITION_MEMBER] = fdisc.PrefixAdaptedComposition(ordered[1])
    for member in profiles:
        members[member.name] = member
    return members


def measure(
    *,
    triples: Sequence[fdisc.Triple],
    arm: arms.Arm,
    ordered: Mapping[int, ac.OrderedFragmentCounts],
    args: argparse.Namespace,
    profiles: Sequence[fdisc.ProfileHomologyMember] = (),
) -> dict[str, Any]:
    """The whole read, identical for a campaign arm and for a planted world."""

    started = time.time()
    body: dict[str, Any] = {
        "arm": {
            "name": arm.spec.name,
            "modality": arm.spec.modality,
            "architecture": arm.spec.architecture,
            "tokenisation": arm.spec.tokenisation,
            "input_format": arm.spec.input_format,
            "pretraining_corpus": arm.spec.pretraining_corpus,
        },
        "conditioning_label": fdisc.conditioning_label_coverage(arm, len(triples)),
    }
    if body["conditioning_label"]["coverage"] < 1.0:
        body["verdict"] = {
            "verdict": "NOT_MEASURABLE",
            "reason": (
                f"this arm's rendering requires a {body['conditioning_label']['field']} "
                f"conditioning label and the cohort defines it on "
                f"{body['conditioning_label']['n_triples_with_label']} of "
                f"{len(triples)} triples. Scoring without the tag is 1.73 nats off the "
                "arm's training distribution (EXP-R2-034) and scoring with a fabricated "
                "one is worse, so nothing is computed behind this gate"
            ),
        }
        return body

    legs, index, shuffles = build_legs(triples, seed=args.seed)
    census = fdisc.alignment_census(arm, legs, sample=len(legs))
    admission = fdisc.admit_arm(census, arm.spec.name, minimum=fdisc.MINIMUM_TOKEN_ALIGNMENT)
    body["admission"] = {"census": census, "verdict": admission}
    if not admission["admitted"]:
        body["verdict"] = {"verdict": "NOT_MEASURABLE", "reason": admission["reason"]}
        return body

    scorer = fdisc.ResidueSequenceScorer(arm, batch_size=args.batch_size)
    scores = scorer.logprobs(legs)
    advantage = per_triple_advantages(triples, scores, index)

    members = ceiling_members(ordered, args.ceiling_orders, profiles)
    groups_all = {
        unit: fdisc.triple_groups(triples, unit=unit) for unit in fdisc.RESAMPLING_UNITS
    }
    body["cohort_units"] = fdisc.unit_census(triples)

    readings: dict[str, Any] = {}
    for mode in fdisc.WINDOWS:
        readings[mode] = window_reading(
            triples=triples,
            advantage=advantage,
            members=members,
            groups_all=groups_all,
            mode=mode,
            args=args,
        )
    body["readings"] = readings
    body["declared"] = {
        "window": args.window,
        "junction_offset": int(args.junction_offset),
        "min_window": int(args.min_window),
        "resampling_unit": args.resampling_unit,
        "ceiling_factor": float(args.ceiling_factor),
        "ceiling_orders": list(args.ceiling_orders),
    }
    headline = readings[args.window]
    body["verdict"] = {
        **headline["verdict"],
        "window": args.window,
        "resampling_unit": args.resampling_unit,
        "verdict_by_window": {mode: readings[mode]["verdict"]["verdict"] for mode in fdisc.WINDOWS},
        "same_verdict_under_both_length_readings": len(
            {readings[mode]["verdict"]["verdict"] for mode in fdisc.WINDOWS}
        ) == 1,
        "verdict_by_resampling_unit": headline["verdict_by_resampling_unit"],
        "same_verdict_under_both_resampling_units": len(
            set(headline["verdict_by_resampling_unit"].values())
        ) == 1,
    }
    body["cost"] = {**scorer.cost(), "wall_seconds": round(time.time() - started, 1)}
    body["shuffled_prefix_example"] = {
        "anchor": triples[0].anchor,
        "prefix_residues": len(triples[0].prefix),
        "permutation_preserves_composition": sorted(triples[0].prefix) == sorted(shuffles[0]),
    }
    return body


def window_reading(
    *,
    triples: Sequence[fdisc.Triple],
    advantage: Mapping[str, Sequence[Mapping[str, np.ndarray]]],
    members: Mapping[str, Any],
    groups_all: Mapping[str, np.ndarray],
    mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """One length reading: the arm, the whole ceiling family, the nulls, the verdict."""

    kept: list[int] = []
    windows: list[dict[str, np.ndarray]] = []
    for position, triple in enumerate(triples):
        lengths = {label: len(triple.continuations[label]) for label in fdisc.CANDIDATES}
        window = fdisc.window_indices(
            lengths, mode=mode, offset=args.junction_offset, minimum=args.min_window
        )
        if window is None:
            continue
        kept.append(position)
        windows.append(window)
    if len(kept) < fdisc.MINIMUM_TRIPLE_GROUPS:
        raise RuntimeError(
            f"{len(kept)} triples keep a window of {args.min_window} residues at offset "
            f"{args.junction_offset} under the {mode} reading, below the unit floor. "
            "That is a statement about the cohort's continuation lengths and the "
            "declared offset, and it is reported rather than worked around"
        )

    arm_values = np.asarray(
        [windowed_contrast(advantage["real"][position], window) for position, window in zip(kept, windows)]
    )
    shuffled_values = np.asarray(
        [windowed_contrast(advantage["shuffled"][position], window) for position, window in zip(kept, windows)]
    )
    conditional_values = np.asarray(
        [windowed_contrast(advantage["conditional"][position], window) for position, window in zip(kept, windows)]
    )
    groups = {unit: np.asarray(groups_all[unit])[np.asarray(kept)] for unit in fdisc.RESAMPLING_UNITS}

    own = {
        unit: fdisc.contrast_interval(
            arm_values, groups[unit], seed=args.seed, draws=args.bootstrap_draws
        )
        for unit in fdisc.RESAMPLING_UNITS
    }
    nulls = {
        "sign_permutation": fdisc.sign_permutation_contrast(
            arm_values, seed=args.seed + 1, draws=args.bootstrap_draws
        ),
        "shuffled_prefix": shuffled_prefix_null(
            shuffled_values, groups[args.resampling_unit], seed=args.seed + 2, draws=args.bootstrap_draws
        ),
    }

    curve: dict[str, Any] = {}
    for name, member in members.items():
        curve[name] = ceiling_block(
            name=name,
            member=member,
            triples=triples,
            advantage=advantage,
            kept=kept,
            windows=windows,
            groups=groups,
            args=args,
        )
    if curve["1"]["ceiling_contrast_is_exactly_zero"] is not True:
        raise RuntimeError(
            "the k = 1 rung of the ceiling curve is not exactly zero. An order-1 "
            "conditional reads no context, so its prefix advantage is zero at every "
            "position by construction, and every higher order shares the same "
            "indexing: a non-zero first point is an indexing defect and no verdict "
            "may be read from the curve above it"
        )

    binding = max(curve, key=lambda name: curve[name]["against"][args.resampling_unit]["reference_contrast"])
    verdict_by_unit = {
        unit: fdisc.fold_verdict(
            margin=curve[binding]["margin"][unit], arm_block=own[unit], nulls=nulls
        )["verdict"]
        for unit in fdisc.RESAMPLING_UNITS
    }
    verdict = fdisc.fold_verdict(
        margin=curve[binding]["margin"][args.resampling_unit],
        arm_block=own[args.resampling_unit],
        nulls=nulls,
    )
    verdict = {
        **verdict,
        "read_against_ceiling_member": binding,
        "verdict_by_ceiling_order": {
            name: fdisc.fold_verdict(
                margin=curve[name]["margin"][args.resampling_unit],
                arm_block=own[args.resampling_unit],
                nulls=nulls,
            )["verdict"]
            for name in curve
        },
    }
    verdict["survives_every_ceiling_order"] = len(set(verdict["verdict_by_ceiling_order"].values())) == 1
    agreeing = sorted(name for name in curve if curve[name]["ceiling_contrast"] > 0.0)
    verdict["binding_member_ceiling_contrast"] = curve[binding]["ceiling_contrast"]
    verdict["binding_member_adequacy_ratio"] = curve[binding]["adequacy"]["ratio"]
    verdict["ceiling_members_favouring_the_structure_partner"] = agreeing
    verdict["ceiling_family_direction"] = (
        "every member of the clause-1 family points at the composition-matched "
        "sequence partner, so the corpus account and the structural account make "
        "opposite-signed predictions on this cohort and the estimand can land on "
        "either side of them"
        if not agreeing
        else f"{agreeing} point at the structure partner too, so the contradiction set is "
        "not two-sided against those members and a positive contrast there is not "
        "evidence against the corpus account"
    )
    verdict["ceiling_adequacy_by_order"] = {
        name: curve[name]["adequacy"]["ratio"] for name in curve
    }
    verdict["first_binding_ceiling_member"] = fdisc.first_binding_order(curve)

    return {
        "window": mode,
        "n_triples_scored": len(kept),
        "n_triples_dropped_for_window": len(triples) - len(kept),
        "scored_positions": {
            "min": int(min(len(window[label]) for window in windows for label in fdisc.CANDIDATES)),
            "median": float(np.median([len(window[label]) for window in windows for label in fdisc.CANDIDATES])),
            "max": int(max(len(window[label]) for window in windows for label in fdisc.CANDIDATES)),
            "equal_across_candidates": bool(
                all(len(window["sequence_partner"]) == len(window["structure_partner"]) for window in windows)
            ),
        },
        "contrast": own,
        "conditional_contrast": {
            "mean": float(conditional_values.mean()),
            "n_triples_positive": int((conditional_values > 0).sum()),
            "status": fdisc.ESTIMAND_DEVIATION,
        },
        "per_triple": [
            {
                "anchor": triples[position].anchor,
                "sequence_partner": triples[position].partner_ids["sequence_partner"],
                "structure_partner": triples[position].partner_ids["structure_partner"],
                "contrast": float(contrast),
                "shuffled_prefix_contrast": float(shuffled),
                "conditional_contrast": float(conditional),
                # Carried so the retrieval-aware clause §8 still owes can be read off
                # this artefact alone: the structure partner shares the anchor's fold,
                # and remote homology below DIAMOND's detection limit predicts exactly
                # the sign this estimand reports.
                "tm_verified_structure_partner": triples[position].tm_structure_partner_verified,
                "tm_ordering_holds": triples[position].tm_ordering_holds,
                "scored_positions": {
                    label: int(window[label].size) for label in fdisc.CANDIDATES
                },
            }
            for position, window, contrast, shuffled, conditional in zip(
                kept, windows, arm_values, shuffled_values, conditional_values
            )
        ],
        "nulls": nulls,
        "ceiling": {
            "family": (
                "the UniRef50 fragment conditional at every declared order, plus the "
                "prefix-adapted composition channel. §7.0 clause 2's ceiling is the best "
                "any member of the clause-1 family achieves, so the verdict is read at "
                "the binding member and never at the friendliest"
            ),
            "binding_member": binding,
            "curve": curve,
        },
        "verdict": verdict,
        "verdict_by_resampling_unit": verdict_by_unit,
    }


def shuffled_prefix_null(
    values: Sequence[float], groups: Sequence[int], *, seed: int, draws: int
) -> dict[str, Any]:
    """Confound 3 on the model side: does the contrast survive a composition-preserving shuffle?"""

    block = fdisc.contrast_interval(values, groups, seed=seed, draws=draws)
    interval = block.get("difference_ci95")
    return {
        "shuffled_contrast": block["contrast"],
        "difference_ci95": interval,
        "fires": bool(interval is not None and interval[0] > 0.0),
        "criterion": (
            "the null fires when the anchor's prefix, with its residues permuted and "
            "its composition therefore untouched, still reads as the structure partner. "
            "The cohort measured the prefix as compositionally closer to the sequence "
            "partner on 94.0% of triples, so a structural reading that survives the "
            "shuffle is a reading about something other than the prefix's arrangement"
        ),
    }


def ceiling_block(
    *,
    name: str,
    member: Any,
    triples: Sequence[fdisc.Triple],
    advantage: Mapping[str, Sequence[Mapping[str, np.ndarray]]],
    kept: Sequence[int],
    windows: Sequence[Mapping[str, np.ndarray]],
    groups: Mapping[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """One ceiling member, scored on exactly the positions it can score.

    D3.l-A0: a ceiling that cannot be evaluated on a triple is not a ceiling for
    that triple, so the triple is dropped with its count reported rather than
    scored against a missing baseline. Where the member can score only some of the
    window, the **arm is restricted to the same positions**, because a difference
    between two quantities measured on different positions is not a difference.
    """

    ceiling_values: list[float] = []
    arm_values: list[float] = []
    survivors: list[int] = []
    covered: list[float] = []
    junction_ceiling: list[float] = []
    junction_arm: list[float] = []
    for slot, (position, window) in enumerate(zip(kept, windows)):
        triple = triples[position]
        restricted: dict[str, np.ndarray] = {}
        ceiling_curve: dict[str, np.ndarray] = {}
        usable_fraction: list[float] = []
        for label in fdisc.CANDIDATES:
            values, usable = member.advantage(triple.prefix, triple.continuations[label])
            indices = np.asarray(window[label], dtype=np.int64)
            keep = indices[usable[indices]]
            usable_fraction.append(keep.size / max(indices.size, 1))
            restricted[label] = keep
            ceiling_curve[label] = values
        if any(restricted[label].size == 0 for label in fdisc.CANDIDATES):
            continue
        survivors.append(position)
        covered.append(float(np.mean(usable_fraction)))
        ceiling_values.append(windowed_contrast(ceiling_curve, restricted))
        arm_values.append(windowed_contrast(advantage["real"][position], restricted))
        junction = {
            label: restricted[label][restricted[label] < _reach(member)]
            for label in fdisc.CANDIDATES
        }
        if all(junction[label].size > 0 for label in fdisc.CANDIDATES):
            junction_ceiling.append(windowed_contrast(ceiling_curve, junction))
            junction_arm.append(windowed_contrast(advantage["real"][position], junction))

    if len(survivors) < fdisc.MINIMUM_TRIPLE_GROUPS:
        raise RuntimeError(
            f"the {name} ceiling member is computable on {len(survivors)} triples, below "
            f"the {fdisc.MINIMUM_TRIPLE_GROUPS}-unit floor. A ceiling that cannot be "
            "evaluated is not a ceiling, and this is a statement about the corpus's "
            "coverage at this order rather than about any model"
        )
    keep_mask = np.isin(np.asarray(kept), np.asarray(survivors))
    member_groups = {
        unit: np.asarray(groups[unit])[keep_mask] for unit in fdisc.RESAMPLING_UNITS
    }
    arm_array = np.asarray(arm_values)
    ceiling_array = np.asarray(ceiling_values)
    against = {
        unit: fdisc.contrast_interval(
            arm_array,
            member_groups[unit],
            seed=args.seed,
            draws=args.bootstrap_draws,
            reference=ceiling_array,
            reference_name=f"uniref50_{name}" if name.isdigit() else name,
        )
        for unit in fdisc.RESAMPLING_UNITS
    }
    own_on_member = {
        unit: fdisc.contrast_interval(
            arm_array, member_groups[unit], seed=args.seed, draws=args.bootstrap_draws
        )
        for unit in fdisc.RESAMPLING_UNITS
    }
    margin = {
        unit: fdisc.ceiling_margin(
            arm_block=own_on_member[unit], against_block=against[unit], factor=args.ceiling_factor
        )
        for unit in fdisc.RESAMPLING_UNITS
    }
    return {
        "member": name,
        "background": member.record(),
        "reaches_window": bool(member.reaches(args.junction_offset)),
        "n_triples_scored": len(survivors),
        "n_triples_dropped_for_coverage": len(kept) - len(survivors),
        "mean_covered_fraction": float(np.mean(covered)),
        "ceiling_contrast": float(ceiling_array.mean()),
        "ceiling_contrast_is_exactly_zero": bool(np.all(ceiling_array == 0.0)),
        "adequacy": adequacy_block(arm_array, ceiling_array),
        "junction_adequacy": junction_adequacy(
            member, np.asarray(junction_arm), np.asarray(junction_ceiling)
        ),
        "against": against,
        "margin": margin,
        "pre_registered_rung": name == str(fdisc.PRE_REGISTERED_FRAGMENT_ORDER),
        "reachability_anchor": name == str(fdisc.REACHABILITY_ORDER),
    }


def junction_adequacy(member: Any, arm: np.ndarray, ceiling: np.ndarray) -> dict[str, Any]:
    """The same ratio over the junction window alone, where a Markov member can act.

    A fragment conditional of order k is non-zero only over the first ``k - 1``
    positions of the continuation, so its ratio over the whole window is diluted by
    the window's length and understates what it does where it can do anything. This
    is the undiluted reading, and it is a diagnostic rather than the quantity the
    margin is taken on.
    """

    if _reach(member) > 1 << 20:
        return {
            "ratio": None,
            "adequate": None,
            "reading": (
                "this member reads the whole prefix at every position, so it has no "
                "junction window distinct from the scored window and its adequacy ratio "
                "above is already undiluted"
            ),
        }
    if ceiling.size == 0:
        return {
            "ratio": None,
            "adequate": False,
            "reading": "no scored position of any triple falls inside this member's reach",
        }
    return adequacy_block(arm, ceiling)


def _reach(member: Any) -> int:
    """How many continuation positions this member's conditioning can still reach."""

    order = getattr(member, "order", None)
    return int(order) - 1 if order is not None else 1 << 30


def adequacy_block(arm: np.ndarray, ceiling: np.ndarray) -> dict[str, Any]:
    """Is this ceiling member doing anything on this estimand?

    ``ceiling_adequacy`` refuses an arm whose contrast is exactly zero on every
    triple, which is the right refusal in a campaign and is a *reachable state* in
    the self-test's third world, whose decoder reads no context at all. The
    condition is therefore checked here rather than caught, because catching it
    would also swallow a campaign arm that produced nothing.
    """

    if arm.size == 0 or float(np.abs(arm).mean()) == 0.0:
        return {
            "arm_mean_absolute_contrast": float(np.abs(arm).mean()) if arm.size else None,
            "ceiling_mean_absolute_contrast": float(np.abs(ceiling).mean()) if ceiling.size else None,
            "ratio": None,
            "adequate": False,
            "reading": (
                "the arm's contrast is exactly zero on every triple, so there is no "
                "quantity for the ceiling to be a fraction of. A decoder that reads no "
                "context produces exactly this"
            ),
        }
    return ac.ceiling_adequacy(arm, ceiling, floor=fdisc.CEILING_ADEQUACY_FLOOR)


# ------------------------------------------------------- known-answer self-test


def run_synthetic_check(args: argparse.Namespace) -> dict[str, Any]:
    """Three worlds whose answer is known, through the identical analysis path."""

    worlds: dict[str, Any] = {}
    for planted in fdisc.PLANTINGS:
        world = fdisc.synthetic_world(
            planted=planted,
            seed=args.synthetic_seed,
            device=args.device,
            ceiling_orders=[order for order in args.ceiling_orders if order <= 3],
        )
        body = measure(triples=world.triples, arm=world.arm, ordered=world.ceiling, args=args)
        worlds[planted] = {
            "settings": world.settings,
            "expected": world.settings["expected"],
            "recovered": body["verdict"]["verdict"],
            "recovered_matches_planted": bool(body["verdict"]["verdict"] == world.settings["expected"]),
            "contrast": body["readings"][args.window]["contrast"][args.resampling_unit],
            "nulls": body["readings"][args.window]["nulls"],
            "ceiling_contrast": {
                name: block["ceiling_contrast"]
                for name, block in body["readings"][args.window]["ceiling"]["curve"].items()
            },
            "binding_member": body["readings"][args.window]["ceiling"]["binding_member"],
            "verdict": body["verdict"],
        }
    refused = fdisc.synthetic_world(
        planted="structure", seed=args.synthetic_seed, device=args.device, paired_tokenisation=True,
        ceiling_orders=[order for order in args.ceiling_orders if order <= 3],
    )
    refusal = measure(triples=refused.triples, arm=refused.arm, ordered=refused.ceiling, args=args)
    behind_the_gate = sorted(
        key for key in refusal if key in {"readings", "cost", "cohort_units", "shuffled_prefix_example"}
    )

    failures = [name for name, block in worlds.items() if not block["recovered_matches_planted"]]
    fired = {
        name: sorted(key for key, null in block["nulls"].items() if null["fires"])
        for name, block in worlds.items()
    }
    if failures:
        raise RuntimeError(
            f"the known-answer check did not recover {failures}: "
            + "; ".join(
                f"{name} planted, {worlds[name]['recovered']} recovered "
                f"(contrast {worlds[name]['contrast']['contrast']:.6g}, interval "
                f"{worlds[name]['contrast']['difference_ci95']})"
                for name in failures
            )
            + ". No campaign number from this instrument may be trusted"
        )
    if any(fired.values()):
        raise RuntimeError(f"a null fired in a planted world: {fired}. The instrument is not clean")
    if refusal["verdict"]["verdict"] != "NOT_MEASURABLE" or behind_the_gate:
        raise RuntimeError(
            f"the paired-tokenisation world returned {refusal['verdict']['verdict']} and "
            f"computed {behind_the_gate} behind its gate; a refused arm must compute nothing"
        )
    return {
        "limitations": limitations_block(kind="synthetic"),
        "verdict": {
            "verdict": "KNOWN_ANSWER_RECOVERED",
            "reason": (
                "the composition-following world is recovered as recombination, the "
                "fold-following world as a structural candidate, and the context-free "
                "world as undecided; neither null fires in any of the three, and a "
                "multi-residue tokenisation is refused with nothing computed behind the "
                "gate. No campaign number from this instrument may be trusted without "
                "this certificate"
            ),
        },
        "worlds": worlds,
        "refusal": {
            "arm": refusal["arm"],
            "admission": refusal["admission"],
            "verdict": refusal["verdict"],
            "keys_behind_the_gate": behind_the_gate,
        },
        "certificate": {
            "plantings": list(fdisc.PLANTINGS),
            "recovered": {name: block["recovered"] for name, block in worlds.items()},
            "expected": {name: block["expected"] for name, block in worlds.items()},
            "nulls_fired": fired,
            "context_free_world_contrast_is_exactly_zero": bool(
                worlds["neither"]["contrast"]["contrast"] == 0.0
            ),
            "opposite_verdicts": bool(
                worlds["sequence_statistics"]["recovered"] != worlds["structure"]["recovered"]
            ),
            "read_on": (
                "the prefix advantage A(c) = log p(c | x) - log p(c), per scored residue. "
                "The composition world's decoder continues the prefix's residue "
                "distribution and the fold world's reads the phase of a marker residue "
                "whose density is identical in both folds, so the two plants are "
                "separated by arrangement and never by composition"
            ),
        },
    }


# ------------------------------------------------------------------------ main


def limitations_block(*, kind: str) -> dict[str, Any]:
    common = {
        "the_ceiling_is_markov_and_the_estimand_is_not": (
            "a fragment conditional of order k carries information across exactly k - 1 "
            "positions, so its prefix advantage is exactly zero beyond position k - 2 of "
            "the continuation and identically zero over any window starting at offset "
            "k - 1 or later. On a continuation of about a hundred residues the fragment "
            "family can therefore only ever contest the first few positions, and its "
            "contrast is diluted by the window length. This is reported per order as "
            "reaches_window and as the adequacy ratio, and it is why the prefix-adapted "
            "composition member -- which reads the whole prefix at every position -- is "
            "carried in the same family"
        ),
        "clearing_the_ceiling_is_not_knowledge": (
            "§7.0 clause 5: a result outside the ceiling is a candidate and nothing more. "
            "§8's causal, retrieval-aware and independent-biological clauses are separate "
            "and none of them is touched here"
        ),
        "no_split_is_drawn": (
            "nothing is fitted, so there is no train/test split to make group-disjoint. "
            "The cohort's own group-disjoint split exists for a stage that fits; this one "
            "reads every admitted triple as one set"
        ),
        "the_two_resampling_units_are_not_equivalent": (
            "a triple's anchor group and the connected component of triples sharing any "
            "near-duplicate group are both implementations of L30's rule and they "
            "disagree on this cohort. Both are computed, the declared one carries the "
            "verdict, and same_verdict_under_both_resampling_units says whether the "
            "choice mattered"
        ),
    }
    if kind == "synthetic":
        return {
            "the_world_is_a_two_rule_decoder": (
                "next-residue logits are a closed-form function of the context: an "
                "adapted unigram in one world, a marker-phase detector in another, a "
                "constant in the third. The rendering, the tokenisation, the alignment "
                "census, the batching, the log-softmax, the gather, the windows, the "
                "ceiling family, both nulls and the verdict are all exercised end to "
                "end; what is not exercised is that the same analysis behaves the same "
                "way through thirty-six transformer blocks"
            ),
            "the_staged_ceiling_is_not_exercised_here": (
                "the fragment members are built from the synthetic universe's own k-mer "
                "counts rather than from UniRef50, because a synthetic corpus of this "
                "size leaves most real trigrams unobserved. The staged background is "
                "exercised in tests/ and in the campaign path"
            ),
            "the_profile_members_are_campaign_only": (
                "a Pfam profile and a jackhmmer search have no meaning on synthetic "
                "residues drawn from a Dirichlet, so the two profile members are not "
                "built here. Their per-residue arithmetic -- profile selection by the "
                "prefix, the bits-to-nats conversion, the span-uniform distribution and "
                "the maximum over covering domains -- is exercised on planted HMMER "
                "tables in tests/, and the members themselves run in the campaign path"
            ),
            "the_fold_signature_is_a_phase_and_not_a_fold": (
                "the structure world's signature is the phase of a periodic marker "
                "residue. It is a property of arrangement that no composition model can "
                "read, which is what the certificate needs; it is not a claim that "
                "protein fold is periodic"
            ),
        }
    common["the_structures_are_predictions"] = (
        "the cohort's fold labels come from CATH assignment and from a coordinate "
        "descriptor over AlphaFold models, with TM-align agreeing on 192 of 199 triples. "
        "90 of 199 structure partners fall below TM 0.5, which is a consequence of "
        "selecting that partner away from sequence space and of excluding every "
        "DIAMOND-detectable homologue; the TM-verified subset is carried per record"
    )
    common["the_background_is_not_any_arm_pretraining_corpus"] = (
        "the ceiling reads the staged UniRef50 background. That is ProtGPT2's declared "
        "corpus and close to ProGen2's UniRef90+BFD30 mixture, and it is neither "
        "ZymCTRL's EC-annotated UniProt nor progen2-base's own mixture. The ceiling is a "
        "corpus-statistics model of protein sequence in general and not a reconstruction "
        "of any one arm's training distribution"
    )
    common["the_junction_exists_in_neither_protein"] = (
        "every scored leg splices a continuation from one protein onto a prefix from "
        "another, so the boundary is off-distribution for both. --junction-offset "
        "declares how far after the splice the window begins and the choice reaches the "
        "artefact; it bounds the estimand and it does not remove the fact that the "
        "conditioned pass reads a chimera"
    )
    common["the_clause_one_family_and_what_of_it_is_built"] = (
        "§7.0 clause 1 names fragment and k-mer statistics, profile-HMM scores and "
        "Potts/MSA couplings, and clause 2 defines the ceiling as the best any member "
        "achieves. Built here: the fragment conditional at every staged order, the "
        "prefix-adapted composition channel, and TWO profile members -- Pfam-A at its "
        "own gathering threshold, and a jackhmmer profile built from each anchor's "
        "prefix against the staged corpus with both candidate accessions removed from "
        "the recruited alignment. The profile members are the ones that matter, because "
        "the structure partner shares the anchor's CATH superfamily by construction and "
        "a remote-homology detector is a corpus-statistics object under clause 1. Not "
        "built: " + fdisc.POTTS_MEMBER_ABSENT
    )
    common["both_profile_members_are_lower_bounds"] = (
        "Pfam annotates only part of any proteome, so a prefix outside Pfam contributes "
        "exactly zero to that member; and the corpus member searches the staged "
        "Swiss-Prot at three jackhmmer iterations rather than the arms' own "
        "UniRef50/UniRef90 mixtures. Both bounds run in the same direction -- they "
        "understate what the profile family achieves and therefore flatter the arm -- "
        "and each member's context coverage and recruitment counts are reported so the "
        "size of the understatement is visible rather than argued"
    )
    common["the_cohort_is_one_proteome"] = (
        "the triple set is drawn from the human AlphaFold subset on disk, so any "
        "biological reading inherits a single-proteome scope"
    )
    return common


def artefact_name(arm: str, digest: str, seed: int) -> str:
    """Basename from arm, cohort digest and seed -- never a fixed string."""

    def safe(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", value)

    return f"fold_discordance__{safe(arm)}__{safe(digest)[:12]}__seed{int(seed)}.json"


def provenance() -> dict[str, Any]:
    return {
        "runner": {
            "path": "scripts/transfer/39_fold_discordance.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "modules": {name: sha256_file(REPO_ROOT / name) for name in PROVENANCE_MODULES},
    }


def pre_registration_block() -> dict[str, Any]:
    return {
        "record": fdisc.PRE_REGISTRATION,
        "track": fdisc.PRE_REGISTRATION_TRACK,
        "amendments_implemented": list(fdisc.PRE_REGISTRATION_AMENDMENTS),
        "governing_rule": "audit §7.0, the recombination ceiling as a standing admission rule",
        "cohort_digest": fdisc.COHORT_DIGEST,
        "required_flags": list(PRE_REGISTERED_DECISIONS),
        "estimand": (
            "P = A(structure partner) - A(sequence partner), where "
            "A(c) = mean over the scored window of [log p(c_i | x, c_<i) - log p(c_i | c_<i)] "
            "and x is the anchor's prefix. P > 0 follows the partner carrying the "
            "anchor's fold; P < 0 follows the composition-matched sequence partner"
        ),
        "estimand_deviation_from_the_frozen_text": fdisc.ESTIMAND_DEVIATION,
        "margin_placement": (
            "EXP-R2-214 amendment 2 froze that §7.0's 2x margin belongs to the "
            "model-side effect and never to the cohort's neighbour separation: a 2x "
            "ordering-margin requirement would have cut the cohort from 199 triples to "
            "4. Nothing here gates on a cohort ordering margin"
        ),
        "ceiling": (
            "one clause-1 family read as a curve with the verdict at the binding member: "
            "the UniRef50 fragment conditional at every declared order, the "
            "prefix-adapted composition channel that captures the trivial 'continue the "
            "prefix's composition' account the cohort's own 94.0% prefix figure warns "
            "about, and two profile-HMM members -- Pfam-A at its gathering threshold and "
            "a jackhmmer profile from each prefix against the staged corpus with both "
            "candidates removed from the recruited alignment. The profile members are "
            "what separates structural knowledge from remote homology, since the "
            "structure partner shares the anchor's CATH superfamily by construction"
        ),
        "potts_member": fdisc.POTTS_MEMBER_ABSENT,
        "margin_on_a_signed_ceiling": (
            "§7.0's multiplicative clause is written for an excess over chance and is "
            "inert on a signed contrast whose ceiling is negative, which is the case by "
            "construction here: 'at least twice the ceiling's positive part' reduces to "
            "'greater than zero'. The deciding statistic is the sign of the contrast and "
            "the paired group-bootstrap interval of the arm-minus-ceiling difference at "
            "the binding member. multiplicative_clause_binds is reported at every member "
            "so the inertness stays visible rather than being quietly dropped"
        ),
        "confound_resolutions": {
            "length": (
                "per-scored-position means throughout, and a length_controlled reading "
                "that truncates both candidates to the shorter one so the two means "
                "cover the same count of positions and the same indices. Both readings "
                "are reported"
            ),
            "composition": (
                "each candidate's own no-prefix score is subtracted before differencing, "
                "so the candidate's composition, length and fragment typicality cancel "
                "and only the prefix's contribution remains"
            ),
            "prefix_favours_the_sequence_partner": (
                "kept in the estimand and answered twice: by the prefix-adapted "
                "composition ceiling member on the statistics side, and by the "
                "composition-preserving prefix-shuffle null on the model side"
            ),
            "junction": (
                "--junction-offset is a required decision naming where the window "
                "begins after the splice, and reaches_window reports per ceiling order "
                "whether the fragment family can contest that window at all"
            ),
        },
        "minimum_units": int(fdisc.MINIMUM_TRIPLE_GROUPS),
        "minimum_token_alignment": float(fdisc.MINIMUM_TOKEN_ALIGNMENT),
        "dtype": fdisc.DTYPE,
        "per_layer_quantities": (
            "none. The readout is a sequence likelihood under two contexts, so this "
            "stage has no layer axis to report per layer or to average over"
        ),
    }


def base_payload(args: argparse.Namespace, *, kind: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "pre_registration": pre_registration_block(),
        "settings": {
            key: (str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value)
            for key, value in vars(args).items()
            if not (args.synthetic and key in CAMPAIGN_ONLY_FLAGS)
        },
        "provenance": provenance(),
    }


def main() -> None:
    args = build_parser().parse_args()
    resolve(args)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        payload = {**base_payload(args, kind="synthetic_known_answer"), **run_synthetic_check(args)}
        destination = args.out / artefact_name("synthetic", "synthetic", args.seed)
        write_json(destination, payload)
        print(f"[synthetic] recovered {payload['certificate']['recovered']}")
        print(f"wrote {destination}")
        return

    triples, cohort_record = fdisc.load_cohort(args.cohort, expected_digest=fdisc.COHORT_DIGEST)
    staging = time.time()
    ordered = fdisc.load_ceiling(
        args.high_order_background, args.ceiling_orders, pinned=args.kmer_background
    )
    ceiling_load_seconds = time.time() - staging
    tool = fdisc.prepare_hmmer(args.hmmer_bin)
    profile_started = time.time()
    profiles = [
        fdisc.build_pfam_profile_member(
            tool=tool, pfam_hmm=args.pfam_hmm, triples=triples,
            workdir=args.profile_workdir / "pfam", cpu=args.hmmer_cpu,
        ),
        fdisc.build_corpus_profile_member(
            tool=tool, corpus_fasta=args.corpus_fasta, triples=triples,
            workdir=args.profile_workdir / "corpus", cpu=args.hmmer_cpu,
            parallel=args.profile_parallel,
        ),
    ]
    profile_seconds = time.time() - profile_started
    arm = arms.load_arm(args.arm, device=args.device, dtype=fdisc.DTYPE)
    body = measure(triples=triples, arm=arm, ordered=ordered, args=args, profiles=profiles)
    if "cost" in body:
        body["cost"]["ceiling_stage_and_digest_seconds"] = round(ceiling_load_seconds, 1)
        body["cost"]["profile_member_seconds"] = round(profile_seconds, 1)
    payload = {
        **base_payload(args, kind="protein_cell"),
        "cohort": cohort_record,
        "limitations": limitations_block(kind="protein"),
        **body,
    }
    destination = args.out / artefact_name(args.arm, cohort_record["sha256"], args.seed)
    write_json(destination, payload)
    verdict = payload["verdict"]
    print(f"[{args.arm}] {verdict['verdict']}")
    for key in ("contrast", "read_against_ceiling_member", "window"):
        if key in verdict:
            print(f"  {key}: {verdict[key]}")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
