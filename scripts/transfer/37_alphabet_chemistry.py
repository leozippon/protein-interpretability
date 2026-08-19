#!/usr/bin/env python3
"""D3.j variant (a): is an amino acid a chemical entity to the model, or a mark?

**What this stage is.** The whole of EXP-R2-214's D3.j variant (a), the cheapest
of the three tracks admitted under audit §7.0. For an ordered residue pair
``r -> s`` it replaces the input embedding row of ``r`` with that of ``s`` -- so
the model reads every ``r`` as an ``s`` while still being required to predict
``r`` -- and measures the damage to held-out likelihood. The design's whole
content is *which pairs are read*: only those on which the declared
physicochemical similarity and the corpus-distributional similarity **disagree**,
because on the agreement set both accounts predict the same damage ordering and
no result there is informative (rule 41).

**The order the gates bind, and it is structural rather than promised.**

1. *D3.j-A0, arm admission.* The single-symbol-token coverage of the arm's own
   scored window is measured, and an arm below the declared bar is written out as
   ``NOT_MEASURABLE`` with its coverage as the reported result. Nothing is
   excluded by name.
2. *D3.j-A1, the byte-level text control.* A protein cell must be handed the text
   control's own artefact and refuses on anything but its ``PASS``. Running the
   control is a separate invocation of this same stage with ``--arm
   bygpt5-medium-en``; there is no flag that lets a protein cell skip it.
3. *The write invariants.* An identity substitution must move the likelihood by
   exactly zero and a seeded random direction of the same norm must move it.
4. *D3.j-A3, the contradiction set's attainability*, on CPU and before the
   checkpoint is touched for measurement: at least eight unordered pairs in each
   quadrant at the declared cut, with the sweep at terciles, quartiles and
   quintiles reported whatever the declared cut is.
5. *Rule 40, reachability*, on the agreement set: the most dissimilar agreement
   pairs must damage the model more than the most similar ones. A failure here
   voids the contradiction read as an instrument failure and the contradiction
   set is **not measured**, so a null cannot be produced by a broken instrument
   and then read as a result.

**What must be cleared.** Not a shuffled null -- under §7.0 clearing one admits
nothing. The recombination ceiling is the UniRef50 3-mer conditional under the
identical substitution on the identical held-out sequences, and the standing
margin is the paired group-bootstrap interval of the arm-minus-ceiling difference
excluding zero over at least eight groups, the arm's own effect at least the
declared factor times the ceiling's positive part, and the effect above the 95th
percentile of at least eight norm-matched random substitute rows.

**Cross-arm magnitudes are not compared.** Damage is in nats per scored token and
a scored token is a residue on a protein arm and a character on the byte-level
control, so no number here is commensurable across the two (L23). The control
closes one half of the method reading and opens no modality reading.

Every pre-registered decision is a required flag with no default, and
``--synthetic`` runs the known-answer self-test: three worlds whose decoder reads
the chemical half, the distributional half, or a third block that neither axis
measures, with the last being the null that must not fire.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.transfer import alphabet_chemistry as ac  # noqa: E402
from src.transfer import arms, kmer_background, scoring  # noqa: E402
from src.transfer.arms import AA20, DEFAULT_CORPUS_DRAW_SEED, PANEL, REPO  # noqa: E402
from src.transfer.io import sha256_file, write_json  # noqa: E402

from panel_contract import QUALIFYING_PROTEIN_BAND  # noqa: E402

SCHEMA_VERSION = "r2_transfer_alphabet_chemistry_v1"
DEFAULT_OUT = REPO / "results/transfer/alphabet_chemistry"

PROVENANCE_MODULES = (
    "src/transfer/alphabet_chemistry.py",
    "src/transfer/arms.py",
    "src/transfer/concept_lens.py",
    "src/transfer/kmer_background.py",
    "src/transfer/scoring.py",
    "src/transfer/statistics.py",
    "src/transfer/io.py",
)

#: Flags that name a real campaign. ``--synthetic`` requires every one to be
#: absent and they are omitted from the synthetic artefact's ``settings`` rather
#: than echoed as null; ``32_crosscoder.py`` records what an echoed null cost it.
CAMPAIGN_ONLY_FLAGS = (
    "arm",
    "kmer_background",
    "high_order_background",
    "text_control",
    "records",
    "max_tokens",
    "min_symbol_occurrences",
    "background_records",
)

#: Decisions this stage refuses to default, in either mode. Each is a threshold
#: or a count that moves the answer, and a stage that supplied one silently would
#: be reporting a choice as a measurement (``32_crosscoder.py:784`` is the same
#: mechanism).
PRE_REGISTERED_DECISIONS = ("cut", "max_pairs", "null_draws", "seed")

#: Decisions the contradiction-set read needs and the byte-level control does
#: not: the control has one axis, so it has no agreement set to check
#: reachability on and no quadrant composition for a random-direction control to
#: price. They are required on a protein cell and on the self-test, and refused
#: on a text arm rather than accepted and echoed into its artefact as settings
#: that decided nothing.
CONTRADICTION_DECISIONS = ("reachability_pairs", "reachability_margin", "random_directions")

#: The ceiling comparison exists only where there is a corpus fragment model to
#: compare against, which is the protein cell.
CEILING_DECISIONS = ("ceiling_factor",)


def parse_orders(argument: str | None, *, name: str) -> tuple[int, ...]:
    """``"1,2,3"`` into ``(1, 2, 3)``, refusing an order the background lacks."""

    if argument is None:
        return ()
    orders = tuple(sorted({int(piece) for piece in argument.replace(" ", "").split(",") if piece}))
    if not orders:
        raise ValueError(f"{name} names no order")
    outside = [order for order in orders if order not in ac.FRAGMENT_ORDERS]
    if outside:
        raise ValueError(
            f"{name} names {outside}, outside the staged background's "
            f"{list(ac.FRAGMENT_ORDERS)}"
        )
    return orders

#: The text control's minimum document length, in characters. Declared here
#: rather than exposed, because it selects nothing the verdict reads: the control
#: needs documents long enough to carry letters, and any length does.
TEXT_MIN_CHARACTERS = 800

#: The order the sweep is measured and reported in, loosest first. Damage is
#: measured once on the loosest readable rung and every stricter rung is a subset
#: of it, so the verdict is read at every rung at no extra cost (rule 17).
SWEEP_ORDER = ("tercile", "quartile", "quintile")


# ----------------------------------------------------------------- arguments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", default=None, choices=sorted(PANEL),
        help="the panel arm to measure. A protein arm runs the contradiction-set "
        "cell; a text arm runs D3.j-A1's byte-level control. Admission is decided "
        "by the measured single-symbol-token coverage and never by this name",
    )
    parser.add_argument(
        "--kmer-background", type=Path, default=None,
        help="directory of the pinned UniRef50 k-mer background. REQUIRED on a "
        "protein arm and never defaulted: it supplies both the distributional axis "
        "and the recombination ceiling, and its digests are checked at load",
    )
    parser.add_argument(
        "--high-order-background", type=Path, default=None,
        help="directory of the k = 1..7 UniRef50 background. REQUIRED on a protein "
        "arm. Its k = 3 and k = 4 vectors must be byte-identical to "
        "--kmer-background's, which is checked by digest before anything is read: "
        "the curve has to extend the pinned background rather than be a second "
        "opinion about the corpus",
    )
    parser.add_argument(
        "--ceiling-orders", default=None,
        help="orders of the recombination-ceiling curve, comma separated, e.g. "
        "1,2,3,4,5,6,7. REQUIRED on a protein arm and never defaulted. k = 1 reads "
        "no context and its Delta is exactly zero by construction, which is the "
        "curve's own reachability anchor; k = 3 is EXP-R2-214's frozen rung and is "
        "kept in the table whatever else is asked for",
    )
    parser.add_argument(
        "--axis-correlation-orders", default=None,
        help="odd orders at which the corpus context axis is recomputed and "
        "correlated against BLOSUM62 and against the chemical axis, comma "
        "separated. Reported only; the admission axis stays the pre-registered "
        "k = 3 one. Omit to report none",
    )
    parser.add_argument(
        "--text-control", type=Path, default=None,
        help="artefact of this stage's byte-level text-control run. REQUIRED on a "
        "protein arm. D3.j-A1 says that if the control fails, no protein arm is "
        "read; this flag is how that ordering is enforced rather than promised",
    )
    parser.add_argument(
        "--records", type=int, default=None,
        help="held-out cohort size. Never defaulted: it sets every damage "
        "estimate's precision",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="scored window per record, in tokens",
    )
    parser.add_argument(
        "--background-records", type=int, default=None,
        help="documents drawn for the text control's distributional background, "
        "disjoint from the scored cohort by the same seed at a skip of --records. "
        "REQUIRED on a text arm and meaningless on a protein arm, whose background "
        "is the pinned k-mer corpus",
    )
    parser.add_argument(
        "--min-symbol-occurrences", type=int, default=None,
        help="reads a symbol needs in the scored cohort to be measurable. On a "
        "protein arm the alphabet is the twenty residues and a shortfall is a "
        "refusal; on a text arm the letter is dropped and recorded",
    )
    parser.add_argument(
        "--cut", default=None, choices=sorted(ac.CUTS),
        help="the rung of D3.j-A2's sweep the headline verdict is read at. The "
        "other rungs are measured and reported from the same damages",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=None,
        help="budget on measured ordered pairs. The draw is round-robin over "
        "(quadrant, substituted symbol) and keeps the strictest sweep rung first",
    )
    parser.add_argument(
        "--reachability-pairs", type=int, default=None,
        help="agreement pairs per end for the rule-40 reachability check",
    )
    parser.add_argument(
        "--reachability-margin", type=float, default=None,
        help="nats/token by which the dissimilar end of the agreement set must "
        "exceed the similar end before the contradiction set is measured at all",
    )
    parser.add_argument(
        "--random-directions", type=int, default=None,
        help=f"norm-matched random substitute rows for D3.j-A5's control, at least "
        f"{ac.MINIMUM_RANDOM_DIRECTIONS}. Rule 39: directions, not positions",
    )
    parser.add_argument(
        "--ceiling-factor", type=float, default=None,
        help="the standing §7.0 margin's multiple of the recombination ceiling. "
        "2.0 is the value in force for D3.g",
    )
    parser.add_argument(
        "--null-draws", type=int, default=None,
        help="draws of the within-symbol shuffled null. It is a detection "
        "criterion and admits nothing on its own",
    )
    parser.add_argument("--seed", type=int, default=None, help="analysis seed")
    parser.add_argument(
        "--cohort-draw-seed", type=int, default=DEFAULT_CORPUS_DRAW_SEED,
        help="seed of the cohort draw, defaulting to the panel's own declared "
        "constant rather than to a literal (Appendix B rule 12). An unseeded "
        "protein draw is a block of near-clonal homologues and has moved a "
        "headline by 1.01 nats, which is why the file-order variant is not "
        "reachable from this stage at all",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--synthetic", action="store_true",
        help="run the known-answer self-test instead of a campaign: three worlds "
        "whose decoder reads the chemical half, the distributional half, or a "
        "nuisance block neither axis measures",
    )
    parser.add_argument("--synthetic-seed", type=int, default=20260819)
    return parser


def resolve(args: argparse.Namespace) -> None:
    """Refuse an incoherent request before a corpus is opened or a model is loaded."""

    required = list(PRE_REGISTERED_DECISIONS)
    forbidden_decisions: list[str] = []
    if args.synthetic:
        required += list(CONTRADICTION_DECISIONS)
        forbidden_decisions += list(CEILING_DECISIONS)
    elif args.arm is not None and PANEL[args.arm].modality == "protein":
        required += list(CONTRADICTION_DECISIONS) + list(CEILING_DECISIONS)
    elif args.arm is not None:
        forbidden_decisions += list(CONTRADICTION_DECISIONS) + list(CEILING_DECISIONS)
    missing = [flag for flag in required if getattr(args, flag) is None]
    if missing:
        raise ValueError(
            "this stage never defaults "
            + ", ".join(f"--{flag.replace('_', '-')}" for flag in missing)
            + ". Each is a pre-registered decision of EXP-R2-214 D3.j: the rung of "
            "the admission sweep the verdict is read at, the pair budget, the "
            "reachability check's size and margin, the number of random substitute "
            "directions, the ceiling margin's factor, the null's draw count and the "
            "analysis seed"
        )
    unused = [flag for flag in forbidden_decisions if getattr(args, flag) is not None]
    if unused:
        raise ValueError(
            ", ".join(f"--{flag.replace('_', '-')}" for flag in unused)
            + " decide nothing on this cell -- a one-axis control has no agreement "
            "set and no corpus fragment ceiling -- and would enter its artefact as "
            "settings that look as though they did"
        )
    if args.synthetic:
        present = [flag for flag in CAMPAIGN_ONLY_FLAGS if getattr(args, flag) is not None]
        if present:
            raise ValueError(
                ", ".join(f"--{flag.replace('_', '-')}" for flag in present)
                + " name a real campaign and are meaningless beside --synthetic, "
                "which runs the same analysis on worlds whose answer is known"
            )
    else:
        if args.arm is None:
            raise ValueError("a campaign run needs --arm")
        modality = PANEL[args.arm].modality
        required = ["records", "max_tokens", "min_symbol_occurrences"]
        if modality == "protein":
            required += ["kmer_background", "high_order_background", "text_control", "ceiling_orders"]
        else:
            required += ["background_records"]
        absent = [flag for flag in required if getattr(args, flag) is None]
        if absent:
            raise ValueError(
                f"a {modality} arm needs "
                + ", ".join(f"--{flag.replace('_', '-')}" for flag in absent)
                + ". The k-mer background carries both the distributional axis and "
                "the recombination ceiling, and the text-control artefact is what "
                "makes D3.j-A1's ordering structural"
            )
        forbidden = [
            flag
            for flag in (
                ["background_records"]
                if modality == "protein"
                else ["kmer_background", "high_order_background", "text_control", "ceiling_orders"]
            )
            if getattr(args, flag) is not None
        ]
        if forbidden:
            raise ValueError(
                ", ".join(f"--{flag.replace('_', '-')}" for flag in forbidden)
                + f" mean nothing on a {modality} arm and would be echoed into the "
                "artefact as a setting that decided something"
            )
    args.ceiling_orders = parse_orders(args.ceiling_orders, name="--ceiling-orders")
    args.axis_correlation_orders = parse_orders(
        args.axis_correlation_orders, name="--axis-correlation-orders"
    )
    even = [order for order in args.axis_correlation_orders if order % 2 == 0]
    if even:
        raise ValueError(
            f"--axis-correlation-orders {even} are even; a symmetric (left, right) "
            "context split is defined only at odd orders and this stage will not "
            "invent an asymmetric one"
        )
    if args.ceiling_orders and ac.PRE_REGISTERED_FRAGMENT_ORDER not in args.ceiling_orders:
        raise ValueError(
            f"--ceiling-orders must include k = {ac.PRE_REGISTERED_FRAGMENT_ORDER}, "
            "the rung EXP-R2-214 froze. The curve is an amendment that adds orders "
            "beside it and never one that substitutes for it"
        )
    if args.random_directions is not None and args.random_directions < ac.MINIMUM_RANDOM_DIRECTIONS:
        raise ValueError(
            f"--random-directions {args.random_directions} is below the declared "
            f"{ac.MINIMUM_RANDOM_DIRECTIONS}; a 95th percentile over fewer draws is "
            "one of a handful of order statistics (rule 39 asks for directions)"
        )
    if args.ceiling_factor is not None and args.ceiling_factor < 1.0:
        raise ValueError(
            "a ceiling factor below one lets an arm inside the recombination "
            "ceiling be recorded as clearing it"
        )
    if args.reachability_margin is not None and args.reachability_margin < 0.0:
        raise ValueError("a negative reachability margin passes an instrument that ranks backwards")
    if (args.reachability_pairs is not None and args.reachability_pairs < 1) or args.max_pairs < 1:
        raise ValueError("--reachability-pairs and --max-pairs must be positive")
    if args.null_draws < 20:
        raise ValueError(
            "a null reported as a distribution needs draws to be a distribution"
        )
    if args.batch_size < 1 or args.bootstrap_draws < 1:
        raise ValueError("--batch-size and --bootstrap-draws must be positive")


# -------------------------------------------------------------------- inputs


def build_cohort(args: argparse.Namespace, spec: arms.ArmSpec) -> arms.Cohort:
    """The held-out cohort, drawn under a seed and in the panel's qualifying band."""

    if spec.modality == "protein":
        low, high = QUALIFYING_PROTEIN_BAND
        return arms.protein_cohort(
            args.records,
            low,
            high,
            name=spec.evaluation_cohort_source,
            with_ec=spec.evaluation_cohort_source == "zymctrl_ec",
            seed=args.cohort_draw_seed,
        )
    return arms.text_cohort(
        args.records, min_chars=TEXT_MIN_CHARACTERS, name=spec.evaluation_cohort_source,
        seed=args.cohort_draw_seed,
    )


def scoring_cohort(arm: arms.Arm, texts: Sequence[str], *, max_tokens: int) -> tuple[
    ac.ScoringCohort, dict[str, Any]
]:
    """Tokenise, and mask off everything that is not scored cohort content."""

    ids, mask = arms.tokenize_batch(arm, list(texts), max_tokens)
    rule = scoring.target_rule(arm.spec.input_format)
    start, end = arms.conditioning_boundary_ids(arm)
    target = scoring.sequence_target_mask(
        ids, mask, rule=rule, start_token_id=start, end_token_id=end
    )
    device = arm.device
    cohort = ac.ScoringCohort(
        input_ids=ids.to(device), attention_mask=mask.to(device), target_mask=target
    )
    return cohort, {
        "target_rule": rule,
        "input_format": arm.spec.input_format,
        "n_records": int(ids.shape[0]),
        "padded_width": int(ids.shape[1]),
        "n_scored_targets": int(target.sum()),
        "conditioning_boundaries": [start, end],
    }


def admit_symbols(
    symbols: Sequence[ac.Symbol], counts: Mapping[int, int], *, minimum: int, fixed: bool
) -> tuple[tuple[ac.Symbol, ...], dict[str, Any]]:
    """Drop, or refuse over, symbols the cohort does not read often enough.

    On a protein arm the alphabet **is** the twenty residues: dropping one would
    change both similarity matrices and the quadrants computed from them, so a
    shortfall is a refusal and the fix is a larger cohort. On the text control the
    alphabet is a convenience and a rare letter is dropped with its count.
    """

    short = {s.label: int(counts.get(s.token_id, 0)) for s in symbols if counts.get(s.token_id, 0) < minimum}
    if fixed and short:
        raise ValueError(
            f"{sorted(short)} are read fewer than {minimum} times in this cohort "
            f"({short}). The alphabet of a protein cell is the twenty residues, so "
            "dropping one would change both similarity matrices; draw more records"
        )
    admitted = tuple(s for s in symbols if counts.get(s.token_id, 0) >= minimum)
    if len(admitted) < 2 * ac.MINIMUM_QUADRANT_PAIRS:
        raise ValueError(
            f"only {len(admitted)} symbols reach {minimum} reads; the quadrants "
            "cannot carry the unit floor"
        )
    return admitted, {
        "minimum_occurrences": int(minimum),
        "n_candidates": len(symbols),
        "n_admitted": len(admitted),
        "occurrences": {s.label: int(counts.get(s.token_id, 0)) for s in symbols},
        "dropped": sorted(short) if not fixed else [],
    }


def read_text_control(path: Path) -> dict[str, Any]:
    """D3.j-A1's gate, read from the control's own artefact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} carries schema {payload.get('schema_version')!r}, not this "
            f"stage's {SCHEMA_VERSION}"
        )
    if payload.get("kind") != "text_control":
        raise ValueError(f"{path} is a {payload.get('kind')!r} artefact, not a text control")
    verdict = payload.get("verdict", {}).get("verdict")
    record = {
        "path": str(path),
        "sha256": sha256_file(Path(path)),
        "arm": payload.get("arm", {}).get("name"),
        "verdict": verdict,
        "delta": payload.get("verdict", {}).get("delta"),
        "difference_ci95": payload.get("verdict", {}).get("difference_ci95"),
    }
    if verdict != "PASS":
        raise RuntimeError(
            f"the byte-level text control at {path} returned {verdict!r}. D3.j-A1: "
            "if the control cannot detect substitute similarity at all, the readout "
            "is VOID as a specification defect and NO protein arm is read"
        )
    return record


# ------------------------------------------------------------------ measuring


def measure_pairs(
    scorer: ac.DamageScorer, pairs: ac.PairSet, symbols: Sequence[ac.Symbol]
) -> tuple[list[float], ac.PairSet, dict[str, Any]]:
    """Damage for every pair, with the unmeasurable ones dropped and named."""

    damages: list[float] = []
    kept_pairs: list[tuple[int, int]] = []
    kept_classes: list[str] = []
    dropped: list[dict[str, Any]] = []
    scored: list[int] = []
    excluded: list[int] = []
    for (x, y), klass in zip(pairs.pairs, pairs.classes):
        record = scorer.damage(symbols[x].token_id, symbols[y].token_id)
        if not record["measurable"]:
            dropped.append(
                {
                    "substituted": symbols[x].label,
                    "substitute": symbols[y].label,
                    "reason": record["unmeasurable_reason"],
                }
            )
            continue
        damages.append(float(record["nats_per_scored_token"]))
        kept_pairs.append((x, y))
        kept_classes.append(klass)
        scored.append(int(record["n_scored_tokens"]))
        excluded.append(int(record["excluded_by_pair_identity"]))
    if not kept_pairs:
        raise RuntimeError("no pair of this set is measurable on this cohort")
    return damages, ac.PairSet(tuple(kept_pairs), tuple(kept_classes)), {
        "n_requested": len(pairs),
        "n_measured": len(kept_pairs),
        "n_dropped": len(dropped),
        "dropped": dropped,
        "scored_tokens_per_pair": {
            "min": int(min(scored)), "median": int(np.median(scored)), "max": int(max(scored))
        },
        "excluded_by_pair_identity_total": int(sum(excluded)),
        "exclusion_rule": (
            "targets whose own identity is the substituted or the substitute symbol "
            "are excluded, so no position is scored on which the substitution's own "
            "label decides the answer"
        ),
    }


def measure_random_directions(
    scorer: ac.DamageScorer,
    pairs: ac.PairSet,
    symbols: Sequence[ac.Symbol],
    rows: Sequence[torch.Tensor],
) -> list[list[float]]:
    """D3.j-A5's control, one damage vector per random substitute row.

    A random row does not depend on the pair's substitute, so damage is computed
    once per (substituted symbol, direction) and reused across that symbol's
    pairs; the quadrant labels are untouched, which is what makes the resulting
    ``Delta`` the contrast the two quadrants' *composition* produces on its own.
    """

    per_direction: list[list[float]] = []
    for row in rows:
        cache: dict[int, float] = {}
        values: list[float] = []
        for x, _ in pairs.pairs:
            token = symbols[x].token_id
            if token not in cache:
                record = scorer.damage(token, token, values=row)
                if not record["measurable"]:
                    raise RuntimeError(
                        f"a random substitute row leaves {symbols[x].label} with no "
                        "scored position, which cannot happen once the symbol is admitted"
                    )
                cache[token] = float(record["nats_per_scored_token"])
            values.append(cache[token])
        per_direction.append(values)
    return per_direction


def strictness_ranks(
    pairs: ac.PairSet, per_cut: Mapping[str, Mapping[str, Any]]
) -> list[int]:
    """0 for a pair the strictest rung admits, rising to the loosest."""

    membership = {
        cut: {tuple(sorted(pair)) for name in record["members"] for pair in record["members"][name]}
        for cut, record in per_cut.items()
    }
    order = list(reversed(SWEEP_ORDER))
    ranks = []
    for x, y in pairs.pairs:
        key = tuple(sorted((x, y)))
        ranks.append(next((index for index, cut in enumerate(order) if key in membership[cut]), len(order)))
    return ranks


def cut_block(
    *,
    cut: str,
    members: Mapping[str, Sequence[tuple[int, int]]],
    class_order: Sequence[str],
    pairs: ac.PairSet,
    damages: Sequence[float],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """The arm's own contrast at one rung of the sweep, from measured damages."""

    keys = {name: {tuple(sorted(pair)) for pair in members[name]} for name in class_order}
    selected = [
        index
        for index, pair in enumerate(pairs.pairs)
        if tuple(sorted(pair)) in keys[pairs.classes[index]]
    ]
    if not selected:
        return None
    subset = ac.PairSet(
        tuple(pairs.pairs[index] for index in selected),
        tuple(pairs.classes[index] for index in selected),
    )
    counts = {name: sum(1 for k in subset.classes if k == name) for name in class_order}
    if min(counts.values()) == 0:
        return None
    values = [damages[index] for index in selected]
    return {
        "cut": cut,
        "measured_ordered_counts": counts,
        "own": ac.delta_contrast(
            codes=subset.codes(class_order), damage=values, groups=subset.groups,
            seed=args.seed, n_bootstrap=args.bootstrap_draws,
        ),
        "_indices": selected,
        "_codes": subset.codes(class_order),
        "_groups": subset.groups,
    }


def spearman_or_none(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Spearman, or ``None`` where one side is constant and it is undefined.

    The k = 1 rung of the ceiling curve is exactly zero for every pair by
    construction, so a correlation against it has no value rather than a value of
    zero. Emitting NaN would be refused by ``write_json`` three steps later, where
    the cause is no longer visible.
    """

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.std() == 0.0 or b.std() == 0.0:
        return None
    value = float(stats.spearmanr(a, b).statistic)
    return value if np.isfinite(value) else None


def axis_order_correlations(
    args: argparse.Namespace,
    *,
    chemical: np.ndarray,
    blosum: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    """The corpus context axis recomputed at higher orders, reported only.

    The admission axis stays the pre-registered k = 3 one; this answers a separate
    question the ceiling curve raises. BLOSUM62 sits on the ceiling side because it
    is estimated from aligned families, and at k = 3 the two statistics estimators
    do not agree at all. If a higher-order corpus axis converged on BLOSUM that
    would say the substitution matrix is a long-context corpus statistic; if it
    moves away, it is measuring something the corpus stream does not carry.
    """

    if not args.axis_correlation_orders:
        return {"orders": [], "note": "not requested"}
    rows, columns = np.triu_indices(chemical.shape[0], 1)
    loaded = ac.load_ordered_counts(
        args.high_order_background, args.axis_correlation_orders, pinned=args.kmer_background
    )
    out: dict[str, Any] = {}
    for order in args.axis_correlation_orders:
        profiles = ac.residue_context_profiles_at_order(loaded[order])
        distance = ac.cosine_distance(profiles)
        out[str(order)] = {
            "context": f"{(order - 1) // 2} left and {(order - 1) // 2} right residues",
            "cells_per_residue": int(profiles.shape[1]),
            "spearman_against_blosum62": float(
                stats.spearmanr(blosum[rows, columns], distance[rows, columns]).statistic
            ),
            "spearman_against_blosum62_p": float(
                stats.spearmanr(blosum[rows, columns], distance[rows, columns]).pvalue
            ),
            "spearman_against_chemical_axis": float(
                stats.spearmanr(chemical[rows, columns], distance[rows, columns]).statistic
            ),
            "spearman_against_admission_axis": float(
                stats.spearmanr(reference[rows, columns], distance[rows, columns]).statistic
            ),
        }
    return {
        "orders": list(args.axis_correlation_orders),
        "status": "reported; the admission axis is the pre-registered k = 3 one",
        "per_order": out,
    }


def matrix_record(matrix: np.ndarray | None) -> list[list[float]] | None:
    if matrix is None:
        return None
    return [[float(value) for value in row] for row in matrix]


def artefact_name(kind: str, arm: str, seed: int) -> str:
    """Basename from arm, intervention and seed -- never a fixed string.

    ``21_joint_mode_qualification.py`` writes every run to one fixed name, so a
    second checkpoint in one output directory overwrites the first without a
    word. All three of this stage's campaign axes are in the name.
    """

    def safe(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", value)

    return f"alphabet_chemistry__{safe(arm)}__{safe(ac.INTERVENTION)}__seed{int(seed)}.json"


def provenance() -> dict[str, Any]:
    return {
        "runner": {
            "path": "scripts/transfer/37_alphabet_chemistry.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "modules": {name: sha256_file(REPO_ROOT / name) for name in PROVENANCE_MODULES},
    }


def pre_registration_block() -> dict[str, Any]:
    return {
        "record": ac.PRE_REGISTRATION,
        "track": ac.PRE_REGISTRATION_TRACK,
        "amendments_implemented": list(ac.PRE_REGISTRATION_AMENDMENTS),
        "governing_rule": "audit §7.0, the recombination ceiling as a standing admission rule",
        "required_flags": {
            "always": list(PRE_REGISTERED_DECISIONS),
            "contradiction_set_read": list(CONTRADICTION_DECISIONS),
            "ceiling_comparison": list(CEILING_DECISIONS),
        },
        "estimand": (
            "Delta_chem = mean held-out NLL damage over chemically-dissimilar / "
            "distributionally-similar ordered pairs minus mean damage over "
            "chemically-similar / distributionally-dissimilar ordered pairs, under "
            "an input-embedding row substitution with the output head untouched"
        ),
        "chemical_axis": ac.CHEMICAL_AXIS_SOURCE,
        "polarity_source": ac.GRANTHAM_SOURCE,
        "distributional_axis": (
            "each symbol's normalised distribution over (left, right) neighbour "
            "pairs, compared by cosine, with symmetric KL as the declared "
            "alternative and both reported"
        ),
        "blosum62": {"source": ac.BLOSUM62_SOURCE, "side": ac.BLOSUM62_SIDE_NOTE},
        "cut_sweep": list(SWEEP_ORDER),
        "minimum_quadrant_pairs": int(ac.MINIMUM_QUADRANT_PAIRS),
        "minimum_symbol_token_coverage": float(ac.MINIMUM_SYMBOL_TOKEN_COVERAGE),
        "minimum_random_directions": int(ac.MINIMUM_RANDOM_DIRECTIONS),
        "dtype": ac.DTYPE,
        "per_layer_quantities": (
            "none. The intervention is on the input embedding and the readout is a "
            "sequence likelihood, so this stage has no layer axis to report per "
            "layer or to average over"
        ),
    }


def limitations_block(*, modality: str) -> dict[str, Any]:
    common = {
        "local_scoring_window": (
            "damage is read only at the target immediately following each read of "
            "the substituted symbol. A swap corrupts the residual stream at every "
            "later position too, and that longer-range damage is not measured. The "
            "narrowing is deliberate -- it removes the dilution by the symbol's own "
            "corpus frequency, which is the statistic this design holds apart from "
            "chemistry -- and it bounds the estimand rather than the model"
        ),
        "ordered_pairs_share_an_axis_value": (
            "both similarity axes are symmetric, so (r, s) and (s, r) carry identical "
            "axis values and different damage. They are separate observations in "
            "separate resampling groups; they add damage replication and no axis "
            "information"
        ),
        "no_cross_arm_magnitude": (
            "damage is in nats per scored token, and a scored token is a residue on a "
            "protein arm and a character on the byte-level control. No magnitude here "
            "is commensurable across the two (L23)"
        ),
        "random_direction_percentile_is_coarse": (
            "the D3.j-A5 clause reads the 95th percentile of the random substitute "
            "directions, and at the pre-registered minimum of eight draws that "
            "percentile is the maximum of eight numbers. A campaign that can afford "
            "more directions should run more; the count is recorded beside the bar"
        ),
    }
    if modality == "synthetic":
        return {
            "the_world_is_a_bigram_decoder": (
                "next-token logits are a linear readout of the current symbol's "
                "embedding row, so the pipeline is exercised end to end -- the swap, "
                "the mask, the cross-entropy, the axes, the quadrants, the interval "
                "and both nulls -- on a decoder with no depth and no normalisation. "
                "It certifies the analysis and the write, not that the analysis "
                "behaves the same way through thirty-six blocks"
            ),
            "no_normalisation_so_no_constant_annihilation": (
                "the synthetic decoder has no layer norm, so the constant-offset "
                "diagnostic moves it as much as a random offset. On a real arm it "
                "does not, and that difference is exactly why the constant is a "
                "diagnostic and the random replacement is the control"
            ),
            "the_ceiling_is_not_exercised_here": (
                "a synthetic corpus of this size leaves most trigrams unobserved and "
                "FragmentConditional refuses an incomplete background rather than "
                "smoothing it, so the recombination ceiling is exercised against the "
                "staged background in tests/ and not in this artefact"
            ),
            "the_random_direction_percentile_is_coarse": (
                "at the pre-registered minimum of eight directions the 95th "
                "percentile is the maximum of eight numbers, and in the null world it "
                "can be exceeded by chance. The null world is caught by the "
                "reachability gate rather than by that clause, which is what the "
                "certificate records"
            ),
        }
    if modality != "protein":
        common["text_closes_only_the_method_reading"] = (
            "text has no chemistry, so this cell establishes only that the readout "
            "detects substitute similarity at all. It opens no modality comparison "
            "and licenses nothing about protein knowledge"
        )
        return common
    common["background_is_not_every_arm_pretraining_corpus"] = (
        "the distributional axis and the ceiling are both the staged UniRef50 k-mer "
        "background. That is ProtGPT2's declared corpus and close to ProGen2's "
        "UniRef90+BFD30 mixture; it is neither ZymCTRL's EC-annotated UniProt nor "
        "progen2-base's own mixture. The ceiling is therefore a corpus-statistics "
        "model of protein sequence in general and not a reconstruction of any one "
        "arm's training distribution"
    )
    common["the_corpus_side_is_instantiated_at_k_equals_three"] = (
        "both the distributional axis and the ceiling read the pinned 3-mer "
        "background, which EXP-R2-214 fixes. A decoder whose corpus dependence is "
        "longer-ranged than a trigram is therefore compared against a corpus model "
        "weaker than itself, and two things follow that the artefact reports rather "
        "than assumes: 'distributionally similar' means similar in immediate "
        "context, and the ceiling's own damage may be a small fraction of the arm's "
        "(see ceiling.adequacy). data/kmer_background/uniref50_high_order carries "
        "k = 1 to 7 over the same corpus and byte-identical k = 3 and k = 4 vectors, "
        "so a higher-order axis and ceiling are a strengthening this stage does not "
        "take on its own authority"
    )
    common["ceiling_factor_clamped_at_zero"] = (
        "the standing margin's 'twice the ceiling' is stated against the ceiling's "
        "positive part. The rule is written for an excess over chance, which is "
        "non-negative; Delta_ceiling is not, and is predicted negative here, so "
        "doubling it unclamped would weaken the clause exactly as the corpus account "
        "held more strongly"
    )
    common["no_split_is_drawn"] = (
        "nothing is fitted, so there is no train/test split to make group-disjoint. "
        "The cohort's near-duplicate structure does not enter the interval because "
        "sequences are not the resampling unit; the substituted symbol is"
    )
    return common


def scored_symbol_records(
    cohort: ac.ScoringCohort, alphabet: Sequence[ac.Symbol]
) -> list[str]:
    """The symbol strings the model actually saw, for the ceiling to be scored on.

    D3.j-A5 requires the ceiling to run on the *same* held-out sequences, and the
    model sees a truncated, rendered window rather than the raw record: a
    conditioning tag, a direction marker and a token cap all move where the
    sequence starts and stops. Reconstructing the string from the tokenised batch
    rather than from the cohort's records is what makes "the same sequences" true
    instead of approximately true. Contiguous runs of alphabet tokens are kept and
    a non-alphabet token ends a run, so no fragment spans a marker.
    """

    label_of = {symbol.token_id: symbol.label for symbol in alphabet}
    records: list[str] = []
    ids = cohort.input_ids.cpu()
    mask = cohort.attention_mask.cpu()
    for row in range(ids.shape[0]):
        current: list[str] = []
        for position in range(ids.shape[1]):
            if not bool(mask[row, position]):
                break
            label = label_of.get(int(ids[row, position]))
            if label is None:
                if len(current) >= 3:
                    records.append("".join(current))
                current = []
                continue
            current.append(label)
        if len(current) >= 3:
            records.append("".join(current))
    if not records:
        raise ValueError("the tokenised cohort yields no symbol run long enough to score")
    return records


# ------------------------------------------------------- known-answer self-test


def run_synthetic_check(args: argparse.Namespace) -> dict[str, Any]:
    """Three worlds whose answer is known, through the identical analysis path.

    ``chemistry`` and ``distribution`` are the two accounts the design exists to
    separate. ``neither`` is the null that must not fire: its decoder reads a
    third block of the embedding row that neither axis measures, so the
    substitutions do real damage that is unrelated to both axes, and an
    instrument returning a signed verdict there would be reading the quadrants'
    composition rather than the substitute.
    """

    worlds: dict[str, Any] = {}
    for planted in ac.PLANTINGS:
        world = ac.synthetic_world(planted=planted, seed=args.synthetic_seed)
        labels = [symbol.label for symbol in world.symbols]
        chemical = ac.property_distance(
            world.property_table, source="synthetic property block"
        )
        profiles = ac.context_profiles(world.context_counts())
        distributional = ac.cosine_distance(profiles)
        embedding = ac.embedding_distance(
            world.model.weight, [symbol.token_id for symbol in world.symbols]
        )
        sweep = ac.cut_sweep(chemical.distance, distributional)
        quadrants = ac.quadrants_at_cut(chemical.distance, distributional, cut=args.cut)
        scorer = ac.DamageScorer(world.model, world.cohort(), batch_size=max(args.batch_size, 16))
        invariants = ac.intervention_invariants(
            scorer,
            symbol_token=world.symbols[0].token_id,
            alphabet_tokens=[symbol.token_id for symbol in world.symbols],
            seed=args.seed,
            tolerance=1e-6,
        )
        agreement, agreement_record = ac.agreement_extremes(
            chemical.distance, distributional, cut=args.cut, count=args.reachability_pairs
        )
        agreement_damage, agreement_kept, _ = measure_pairs(scorer, agreement, world.symbols)
        reachability = ac.reachability_verdict(
            agreement_damage, agreement_kept.classes, margin=args.reachability_margin
        )
        if not reachability["reachable"]:
            if planted != "neither":
                raise RuntimeError(
                    f"the {planted} world fails its own reachability check "
                    f"({reachability['dissimilar_mean_nats']:.4g} against "
                    f"{reachability['similar_mean_nats']:.4g} nats/token). The gate "
                    "is reachable by construction in a world whose decoder reads an "
                    "axis, so this is the instrument and not the world"
                )
            # The null world's decoder reads a block neither axis measures, so
            # rule 40's gate is *supposed* to be able to stop here: the instrument
            # refuses to read an arm whose alphabet structure it cannot detect,
            # which is the correct outcome and not a failure of the check.
            worlds[planted] = {
                "settings": world.settings,
                "labels": labels,
                "axis_sweep": sweep,
                "invariants": invariants,
                "reachability": {**reachability, "selection": agreement_record},
                "verdict": "VOID_INSTRUMENT_UNATTAINABLE",
                "recovered": "neither",
                "recovered_matches_planted": True,
                "note": (
                    "the contradiction set was not measured, because the reachability "
                    "gate stopped the read. That is the designed behaviour on a "
                    "decoder whose alphabet structure neither axis describes"
                ),
            }
            continue
        pairs = ac.ordered_pair_set(quadrants)
        pairs, cap = ac.cap_pairs(
            pairs,
            strictness=strictness_ranks(
                pairs, {cut: ac.quadrants_at_cut(chemical.distance, distributional, cut=cut) for cut in ac.CUTS}
            ),
            maximum=args.max_pairs,
            seed=args.seed,
        )
        damages, kept, measurement = measure_pairs(scorer, pairs, world.symbols)
        codes = kept.codes(ac.QUADRANTS)
        delta = ac.delta_contrast(
            codes=codes, damage=damages, groups=kept.groups,
            seed=args.seed, n_bootstrap=args.bootstrap_draws,
        )
        rows = ac.norm_matched_random_rows(
            world.model.weight, [symbol.token_id for symbol in world.symbols],
            count=args.random_directions, seed=args.seed + 1,
        )
        random_null = ac.random_direction_delta_null(
            codes=codes,
            per_direction=measure_random_directions(scorer, kept, world.symbols, rows),
            observed=delta["delta"],
        )
        shuffled = ac.shuffled_difference_null(
            damage=damages, codes=codes, groups=kept.groups,
            seed=args.seed + 2, draws=args.null_draws,
        )
        secondary = ac.association(
            damage=damages,
            chemical=[chemical.distance[x, y] for x, y in kept.pairs],
            distributional=[distributional[x, y] for x, y in kept.pairs],
            embedding=[embedding[x, y] for x, y in kept.pairs],
            groups=kept.groups, seed=args.seed + 3, n_bootstrap=args.bootstrap_draws,
        )
        controlled = secondary["embedding_distance_controlled"]
        interval = controlled["difference_ci95"]
        recovered = (
            "chemistry" if interval is not None and interval[0] > 0.0
            else "distribution" if interval is not None and interval[1] < 0.0
            else "neither"
        )
        worlds[planted] = {
            "settings": world.settings,
            "labels": labels,
            "axis_sweep": sweep,
            "invariants": invariants,
            "reachability": {**reachability, "selection": agreement_record},
            "pair_budget": cap,
            "measurement": measurement,
            "delta": delta,
            "random_direction_null": random_null,
            "shuffled_null": shuffled,
            "secondary_association": secondary,
            "recovered": recovered,
            "recovered_matches_planted": bool(recovered == planted),
        }
    failures = [name for name, block in worlds.items() if not block["recovered_matches_planted"]]
    if failures:
        raise RuntimeError(
            f"the known-answer check did not recover {failures}: "
            + "; ".join(
                f"{name} planted, {worlds[name]['recovered']} recovered "
                f"(controlled difference {worlds[name]['secondary_association']['embedding_distance_controlled']['difference']:.4f} "
                f"ci {worlds[name]['secondary_association']['embedding_distance_controlled']['difference_ci95']})"
                for name in failures
            )
            + ". No campaign number from this instrument may be trusted"
        )
    return {
        "limitations": limitations_block(modality="synthetic"),
        "verdict": {
            "verdict": "KNOWN_ANSWER_RECOVERED",
            "reason": (
                "each planted world is recovered as itself and the null world is "
                "stopped by the reachability gate; no campaign number from this "
                "instrument may be trusted without this certificate"
            ),
        },
        "worlds": worlds,
        "certificate": {
            "plantings": list(ac.PLANTINGS),
            "recovered": {name: block["recovered"] for name, block in worlds.items()},
            "identity_damage_exactly_zero": {
                name: block["invariants"]["identity_damage_nats"] == 0.0
                for name, block in worlds.items()
            },
            "null_world_fires": worlds["neither"]["recovered"] != "neither",
            "null_world_outcome": worlds["neither"].get("verdict", "read_and_undecided"),
            "read_on": (
                "the embedding-distance-controlled, within-symbol-centred association "
                "difference, because Delta itself retains the quadrants' composition "
                "and is not centred at zero under a within-symbol shuffle. That is a "
                "property of the estimand and the reason D3.j-A5's norm-matched "
                "random-direction control is a required clause rather than a courtesy"
            ),
            "constant_vector_note": (
                "the synthetic decoder has no layer norm, so the constant-vector "
                "diagnostic moves it as much as a random direction does. On a real "
                "arm it does not, which is exactly why the constant is a diagnostic "
                "and the random direction is the control"
            ),
        },
    }


# ------------------------------------------------------------------------ main


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


def run_protein_cell(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    spec = PANEL[args.arm]
    control = read_text_control(args.text_control)

    # The axes and the sweep are a function of the declared descriptor table and
    # the pinned corpus alone, so D3.j-A3 runs here -- before the checkpoint is
    # loaded -- and a cut with no contradiction set costs no GPU.
    background = kmer_background.load(args.kmer_background)
    ordered = ac.load_ordered_counts(
        args.high_order_background, args.ceiling_orders, pinned=args.kmer_background
    )
    labels = list(AA20)
    chemical = ac.property_distance(
        ac.chemical_property_table(labels), source=ac.CHEMICAL_AXIS_SOURCE
    )
    profiles = ac.context_profiles(ac.residue_context_counts(background, labels))
    distributional = ac.cosine_distance(profiles)
    alternative = ac.symmetric_kl_distance(profiles)
    blosum = ac.blosum62_distance(labels)
    rows, columns = np.triu_indices(len(labels), 1)
    axes = {
        "labels": labels,
        "chemical": chemical.record(),
        "distributional": {
            "metric": ac.DISTRIBUTIONAL_METRICS[0],
            "alternative_metric": ac.DISTRIBUTIONAL_METRICS[1],
            "source": str(args.kmer_background),
            "background_records": int(background.records),
            "background_residues": int(background.residues),
            "definition": (
                "each residue's normalised distribution over (left, right) neighbour "
                "pairs in the pinned 3-mer windows"
            ),
            "alternative_agreement_spearman": (
                float(stats.spearmanr(distributional[rows, columns], alternative[rows, columns]).statistic)
                if alternative is not None
                else None
            ),
        },
        "ceiling_side_second_estimator": {
            "name": "blosum62",
            "source": ac.BLOSUM62_SOURCE,
            "side_note": ac.BLOSUM62_SIDE_NOTE,
            "spearman_against_kmer_axis": float(
                stats.spearmanr(blosum[rows, columns], distributional[rows, columns]).statistic
            ),
            "spearman_against_chemical_axis": float(
                stats.spearmanr(blosum[rows, columns], chemical.distance[rows, columns]).statistic
            ),
        },
        "order_correlations": axis_order_correlations(
            args, chemical=chemical.distance, blosum=blosum, reference=distributional
        ),
        "matrices": {
            "chemical": matrix_record(chemical.distance),
            "distributional_cosine": matrix_record(distributional),
            "distributional_symmetric_kl": matrix_record(alternative),
            "blosum62": matrix_record(blosum),
        },
    }
    sweep = ac.cut_sweep(chemical.distance, distributional)
    per_cut = {cut: ac.quadrants_at_cut(chemical.distance, distributional, cut=cut) for cut in ac.CUTS}
    body: dict[str, Any] = {
        "arm": {
            "name": spec.name, "modality": spec.modality, "architecture": spec.architecture,
            "tokenisation": spec.tokenisation, "input_format": spec.input_format,
            "pretraining_corpus": spec.pretraining_corpus,
        },
        "text_control": control,
        "axes": axes,
        "contradiction_set": {
            "sweep": sweep,
            "declared_cut": args.cut,
            "unordered_members": {
                cut: {
                    name: [f"{labels[x]}{labels[y]}" for x, y in record["members"][name]]
                    for name in ac.QUADRANTS
                }
                for cut, record in per_cut.items()
            },
        },
        "limitations": limitations_block(modality="protein"),
    }
    if not per_cut[args.cut]["readable"]:
        body["verdict"] = {
            "verdict": "NO_CONTRADICTION_SET_AT_DECLARED_CUT",
            "reason": (
                f"the {args.cut} cut admits "
                f"{per_cut[args.cut]['unordered_counts']} unordered pairs against a "
                f"floor of {ac.MINIMUM_QUADRANT_PAIRS} per quadrant (D3.j-A3). This "
                "is a statement about the alphabet and the corpus, not about any "
                "model, and no checkpoint was loaded"
            ),
        }
        return body

    cohort = build_cohort(args, spec)
    arm = arms.load_arm(args.arm, device=args.device, dtype=ac.DTYPE)
    texts = cohort.input_strings(arm)
    try:
        alphabet = ac.protein_alphabet(arm)
    except ValueError as error:
        body["verdict"] = {
            "verdict": "NOT_MEASURABLE",
            "reason": f"the alphabet is not addressable on this arm: {error}",
        }
        return body
    coverage = ac.symbol_token_coverage(
        arm, texts, alphabet=alphabet, max_len=args.max_tokens
    )
    admission = ac.admit_arm(
        coverage, arm.name, minimum=ac.MINIMUM_SYMBOL_TOKEN_COVERAGE
    )
    body["admission"] = {"coverage": coverage, "verdict": admission}
    if not admission["admitted"]:
        body["verdict"] = {"verdict": "NOT_MEASURABLE", "reason": admission["reason"]}
        return body

    model = ac.ArmAlphabetModel(arm)
    body["intervention"] = model.record()
    scoring_batch, cohort_record = scoring_cohort(arm, texts, max_tokens=args.max_tokens)
    body["cohort"] = {
        **cohort_record,
        "name": cohort.name,
        "digest": cohort.digest,
        "provenance_digest": cohort.provenance_digest,
        "sampling": cohort.sampling,
        "band_residues": list(QUALIFYING_PROTEIN_BAND),
    }
    occurrences = ac.context_counts(scoring_batch, [s.token_id for s in alphabet])
    alphabet, occupancy = admit_symbols(
        alphabet, occurrences, minimum=args.min_symbol_occurrences, fixed=True
    )
    body["alphabet"] = occupancy

    scorer = ac.DamageScorer(model, scoring_batch, batch_size=args.batch_size)
    probe = max(alphabet, key=lambda symbol: occurrences[symbol.token_id])
    body["invariants"] = ac.intervention_invariants(
        scorer,
        symbol_token=probe.token_id,
        alphabet_tokens=[symbol.token_id for symbol in alphabet],
        seed=args.seed,
        tolerance=1e-6,
    )

    agreement, agreement_record = ac.agreement_extremes(
        chemical.distance, distributional, cut=args.cut, count=args.reachability_pairs
    )
    agreement_damage, agreement_kept, agreement_measurement = measure_pairs(
        scorer, agreement, alphabet
    )
    reachability = ac.reachability_verdict(
        agreement_damage, agreement_kept.classes, margin=args.reachability_margin
    )
    body["reachability"] = {
        **reachability, "selection": agreement_record, "measurement": agreement_measurement,
        "pairs": agreement_kept.labelled(alphabet),
    }
    if not reachability["reachable"]:
        body["verdict"] = {
            "verdict": "VOID_INSTRUMENT_UNATTAINABLE",
            "reason": reachability["consequence_if_failed"],
        }
        return body

    measured_cut = SWEEP_ORDER[0]
    pairs = ac.ordered_pair_set(per_cut[measured_cut])
    pairs, budget = ac.cap_pairs(
        pairs, strictness=strictness_ranks(pairs, per_cut),
        maximum=args.max_pairs, seed=args.seed,
    )
    damages, kept, measurement = measure_pairs(scorer, pairs, alphabet)
    body["measurement"] = {
        "measured_at_cut": measured_cut,
        "budget": budget, **measurement,
        "pairs": kept.labelled(alphabet),
        "damage_nats_per_scored_token": [float(value) for value in damages],
    }

    ceiling_records = scored_symbol_records(scoring_batch, alphabet)
    ceiling_by_order: dict[int, list[float]] = {}
    ceiling_curve: dict[str, Any] = {}
    for order in args.ceiling_orders:
        model_k = ac.FragmentConditional(ordered[order])
        values: list[float] = []
        fractions: list[float] = []
        for x, y in kept.pairs:
            record = model_k.damage(ceiling_records, alphabet[x].label, alphabet[y].label)
            if not record["measurable"]:
                raise RuntimeError(
                    f"the k = {order} ceiling has no scored k-gram for "
                    f"{alphabet[x].label} -> {alphabet[y].label}, so the arm and the "
                    "ceiling would be compared on different pairs"
                )
            values.append(float(record["nats_per_scored_token"]))
            fractions.append(float(record["scored_fraction"]))
        ceiling_by_order[order] = values
        ceiling_curve[str(order)] = {
            **ordered[order].record(),
            "damage_nats_per_scored_token": values,
            "mean_scored_fraction": float(np.mean(fractions)),
            "adequacy": ac.ceiling_adequacy(damages, values, floor=ac.CEILING_ADEQUACY_FLOOR),
            "pre_registered_rung": order == ac.PRE_REGISTERED_FRAGMENT_ORDER,
            "spearman_against_arm_damage": spearman_or_none(damages, values),
            "spearman_against_distributional_axis": spearman_or_none(
                [distributional[x, y] for x, y in kept.pairs], values
            ),
            "spearman_against_chemical_axis": spearman_or_none(
                [chemical.distance[x, y] for x, y in kept.pairs], values
            ),
            "undefined_correlation_reason": (
                "this order reads no context, so its damage is exactly zero for every "
                "pair and a correlation against it has no value"
                if order == 1
                else None
            ),
        }
    body["ceiling"] = {
        "model": "uniref50 fragment conditional, per order",
        "source": str(args.high_order_background),
        "orders": list(args.ceiling_orders),
        "pre_registered_order": ac.PRE_REGISTERED_FRAGMENT_ORDER,
        "n_records": len(ceiling_records),
        "definition": (
            "P(residue | the previous order-1 residues) as a plug-in estimate with no "
            "smoothing, under the identical substitution -- every r in the context "
            "read as s, the target untouched -- on the residue runs the model itself "
            "scored. A position enters only where both the clean and the substituted "
            "k-gram were observed; the covered fraction is reported per order"
        ),
        "amendment": (
            "EXP-R2-214 froze k = 3. At that order the ceiling's damage is a few "
            "percent of a decoder's on this estimand, so 'at least twice the ceiling' "
            "degenerates into 'greater than zero' and audit §7.0's null stops binding. "
            "The curve is the pre-data amendment of 2026-08-19; k = 3 is retained in "
            "the table so the amendment's effect is visible"
        ),
        "curve": ceiling_curve,
    }

    directions = ac.norm_matched_random_rows(
        model.weight, [symbol.token_id for symbol in alphabet],
        count=args.random_directions, seed=args.seed + 1,
    )
    per_direction = measure_random_directions(scorer, kept, alphabet, directions)

    blocks: dict[str, Any] = {}
    for cut in SWEEP_ORDER:
        block = cut_block(
            cut=cut, members=per_cut[cut]["members"], class_order=ac.QUADRANTS,
            pairs=kept, damages=damages, args=args,
        )
        if block is None:
            blocks[cut] = {"cut": cut, "readable": False,
                           "reason": "no measured pair of one quadrant survives at this rung"}
            continue
        indices = block.pop("_indices")
        codes = block.pop("_codes")
        groups = block.pop("_groups")
        random_null = ac.random_direction_delta_null(
            codes=codes,
            per_direction=[[values[i] for i in indices] for values in per_direction],
            observed=block["own"]["delta"],
        )
        block["random_direction_null"] = random_null
        by_order: dict[str, Any] = {}
        for order in args.ceiling_orders:
            against = ac.delta_contrast(
                codes=codes, damage=[damages[i] for i in indices], groups=groups,
                seed=args.seed, n_bootstrap=args.bootstrap_draws,
                reference=[ceiling_by_order[order][i] for i in indices],
                reference_name=f"uniref50_{order}mer_conditional",
            )
            margin = ac.ceiling_margin(
                delta_block=block["own"], ceiling_block=against,
                random_null=random_null, factor=args.ceiling_factor,
            )
            by_order[str(order)] = {
                "against_ceiling": against,
                "ceiling_margin": margin,
                "verdict": ac.protein_verdict(margin=margin, delta_block=block["own"]),
                "ceiling_adequacy_ratio": ceiling_curve[str(order)]["adequacy"]["ratio"],
                "ceiling_adequate": ceiling_curve[str(order)]["adequacy"]["adequate"],
            }
        # The binding order is the most demanding ceiling this curve reaches, and
        # the verdict is read there rather than at the friendliest rung. A ceiling
        # that binds harder is the whole point of the amendment; protecting the
        # k = 3 reading against it would be the failure it exists to prevent.
        binding = max(args.ceiling_orders, key=lambda k: by_order[str(k)]["against_ceiling"]["reference_delta"])
        block["by_ceiling_order"] = by_order
        block["binding_order"] = int(binding)
        block["verdict"] = {
            **by_order[str(binding)]["verdict"],
            "read_against_ceiling_order": int(binding),
            "verdict_by_ceiling_order": {
                order: by_order[order]["verdict"]["verdict"] for order in by_order
            },
            "survives_every_ceiling_order": len(
                {by_order[order]["verdict"]["verdict"] for order in by_order}
            ) == 1,
        }
        block["readable"] = True
        blocks[cut] = block

    headline = blocks[args.cut]
    if "verdict" not in headline:
        body["verdict"] = {
            "verdict": "NO_CONTRADICTION_SET_AT_DECLARED_CUT",
            "reason": (
                f"after the pair budget and the cohort's own unmeasurable pairs, the "
                f"{args.cut} rung retains no measured pair in one quadrant: "
                f"{headline.get('reason')}"
            ),
        }
        body["sweep"] = {"per_cut": blocks, "declared_cut": args.cut}
        return body
    verdicts = {cut: block.get("verdict", {}).get("verdict") for cut, block in blocks.items()}
    stable = len({value for value in verdicts.values() if value is not None}) == 1
    body["sweep"] = {
        "per_cut": blocks,
        "declared_cut": args.cut,
        "verdicts": verdicts,
        "ordering_invariant_across_sweep": stable,
        "rule": (
            "rule 17: the cut survives into the verdict, so it is swept, and so does "
            "the ceiling's order. Damage is measured once on the loosest rung and "
            "every stricter rung is a subset of it, so both sweeps cost no further "
            "forward pass"
        ),
    }
    binding_curve = headline["binding_order"]
    body["verdict"] = {
        **headline["verdict"],
        "cut": args.cut,
        "ordering_invariant_across_sweep": stable,
        "ceiling_adequacy_ratio": ceiling_curve[str(binding_curve)]["adequacy"]["ratio"],
        "ceiling_adequate": ceiling_curve[str(binding_curve)]["adequacy"]["adequate"],
        "ceiling_adequacy_reading": ceiling_curve[str(binding_curve)]["adequacy"]["reading"],
        "ceiling_adequacy_by_order": {
            order: ceiling_curve[order]["adequacy"]["ratio"] for order in ceiling_curve
        },
    }

    body["cost"] = {**scorer.cost(), "wall_seconds": round(time.time() - started, 1)}
    codes = kept.codes(ac.QUADRANTS)
    embedding = ac.embedding_distance(
        model.weight, [symbol.token_id for symbol in alphabet]
    )
    body["secondary"] = {
        "shuffled_null": ac.shuffled_difference_null(
            damage=damages, codes=codes, groups=kept.groups,
            seed=args.seed + 2, draws=args.null_draws,
        ),
        "association": ac.association(
            damage=damages,
            chemical=[chemical.distance[x, y] for x, y in kept.pairs],
            distributional=[distributional[x, y] for x, y in kept.pairs],
            embedding=[embedding[x, y] for x, y in kept.pairs],
            groups=kept.groups, seed=args.seed + 3, n_bootstrap=args.bootstrap_draws,
        ),
    }
    return body


def run_text_control(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    spec = PANEL[args.arm]
    cohort = build_cohort(args, spec)
    arm = arms.load_arm(args.arm, device=args.device, dtype=ac.DTYPE)
    texts = cohort.input_strings(arm)
    body: dict[str, Any] = {
        "arm": {
            "name": spec.name, "modality": spec.modality, "architecture": spec.architecture,
            "tokenisation": spec.tokenisation, "input_format": spec.input_format,
            "pretraining_corpus": spec.pretraining_corpus,
        },
        "limitations": limitations_block(modality="text"),
    }
    try:
        alphabet = ac.text_alphabet(arm)
    except ValueError as error:
        body["verdict"] = {
            "verdict": "NOT_MEASURABLE",
            "reason": f"the alphabet is not addressable on this arm: {error}",
        }
        return body
    coverage = ac.symbol_token_coverage(arm, texts, alphabet=alphabet, max_len=args.max_tokens)
    admission = ac.admit_arm(coverage, arm.name, minimum=ac.MINIMUM_SYMBOL_TOKEN_COVERAGE)
    body["admission"] = {"coverage": coverage, "verdict": admission}
    if not admission["admitted"]:
        body["verdict"] = {"verdict": "NOT_MEASURABLE", "reason": admission["reason"]}
        return body

    model = ac.ArmAlphabetModel(arm)
    body["intervention"] = model.record()
    scoring_batch, cohort_record = scoring_cohort(arm, texts, max_tokens=args.max_tokens)
    body["cohort"] = {
        **cohort_record, "name": cohort.name, "digest": cohort.digest,
        "provenance_digest": cohort.provenance_digest, "sampling": cohort.sampling,
        "min_characters": TEXT_MIN_CHARACTERS,
    }
    occurrences = ac.context_counts(scoring_batch, [s.token_id for s in alphabet])
    alphabet, occupancy = admit_symbols(
        alphabet, occurrences, minimum=args.min_symbol_occurrences, fixed=False
    )
    body["alphabet"] = occupancy

    background = arms.text_cohort(
        args.background_records, min_chars=TEXT_MIN_CHARACTERS,
        skip=args.records, name=spec.evaluation_cohort_source, seed=args.cohort_draw_seed,
    )
    sequences = [
        arm.tokenizer(text, return_tensors=None)["input_ids"][: args.max_tokens]
        for text in background.records
    ]
    buckets, n_buckets, coarsening = ac.lowercase_letter_buckets(arm)
    profiles = ac.context_profiles(
        ac.token_context_counts(
            sequences, alphabet, bucket_of_token=buckets, n_buckets=n_buckets
        )
    )
    distributional = ac.cosine_distance(profiles)
    alternative = ac.symmetric_kl_distance(profiles)
    labels = [symbol.label for symbol in alphabet]
    body["axes"] = {
        "labels": labels,
        "chemical": None,
        "chemical_absent_reason": (
            "text has no chemistry. D3.j-A1 reads one axis, which is the only half "
            "of the design a text arm can validate"
        ),
        "distributional": {
            "metric": ac.DISTRIBUTIONAL_METRICS[0],
            "coarsening": coarsening,
            "background_records": len(background.records),
            "background_digest": background.digest,
            "background_disjoint": (
                "drawn under the same seed at a skip of --records, so the background "
                "documents and the scored documents are disjoint by construction"
            ),
            "symmetric_kl_defined": alternative is not None,
        },
        "matrices": {
            "distributional_cosine": matrix_record(distributional),
            "distributional_symmetric_kl": matrix_record(alternative),
        },
    }

    pairs, band = ac.text_control_pair_set(distributional, cut=args.cut)
    body["pair_set"] = band
    if not band["readable"]:
        body["verdict"] = {
            "verdict": "VOID",
            "reason": (
                f"the {args.cut} cut admits {band['unordered_counts']} unordered "
                f"pairs against a floor of {ac.MINIMUM_QUADRANT_PAIRS} per band"
            ),
        }
        return body

    scorer = ac.DamageScorer(model, scoring_batch, batch_size=args.batch_size)
    probe = max(alphabet, key=lambda symbol: occurrences[symbol.token_id])
    body["invariants"] = ac.intervention_invariants(
        scorer, symbol_token=probe.token_id,
        alphabet_tokens=[symbol.token_id for symbol in alphabet],
        seed=args.seed, tolerance=1e-6,
    )
    pairs, budget = ac.cap_pairs(
        pairs, strictness=[0] * len(pairs), maximum=args.max_pairs, seed=args.seed
    )
    damages, kept, measurement = measure_pairs(scorer, pairs, alphabet)
    codes = kept.codes(ac.TEXT_BANDS)
    delta = ac.delta_contrast(
        codes=codes, damage=damages, groups=kept.groups,
        seed=args.seed, n_bootstrap=args.bootstrap_draws,
    )
    body["measurement"] = {
        "budget": budget, **measurement, "pairs": kept.labelled(alphabet),
        "damage_nats_per_scored_token": [float(value) for value in damages],
    }
    body["delta"] = delta
    body["secondary"] = {
        "shuffled_null": ac.shuffled_difference_null(
            damage=damages, codes=codes, groups=kept.groups,
            seed=args.seed + 2, draws=args.null_draws,
        )
    }
    body["cost"] = {**scorer.cost(), "wall_seconds": round(time.time() - started, 1)}
    body["verdict"] = ac.text_control_verdict(delta)
    return body


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

    modality = PANEL[args.arm].modality
    kind = "protein_cell" if modality == "protein" else "text_control"
    body = run_protein_cell(args) if modality == "protein" else run_text_control(args)
    payload = {**base_payload(args, kind=kind), **body}
    destination = args.out / artefact_name(kind, args.arm, args.seed)
    write_json(destination, payload)
    verdict = payload["verdict"]
    print(f"[{args.arm}] {verdict['verdict']}")
    for key in ("delta", "difference_ci95", "cut"):
        if key in verdict:
            print(f"  {key}: {verdict[key]}")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
