"""Invariants of the substitution-tolerant natural-repeat probe.

The approximate probe exists to test whether the induction census's head-count
deficit is a property of protein decoders or of a probe that demanded literal
identity.  That question is only answerable if the approximate criterion really
is the exact criterion with one clause relaxed, so the properties asserted here
are the ones a reader has to be able to assume when comparing the two columns:
the criteria are nested, the substitution cap and the ungapped, two-occurrence
scope of arXiv:2602.23179 v5 are honoured, the BLOSUM62 rule actually rejects
adverse substitutions, and the text control is strictly more permissive than the
protein criterion rather than differently strict.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.arms import AA20  # noqa: E402
from src.transfer.circuits import (  # noqa: E402
    BLOSUM62,
    BLOSUM62_ORDER,
    PROTEIN_APPROXIMATE_CRITERION,
    PROTEIN_EXACT_CRITERION,
    TEXT_APPROXIMATE_CRITERION,
    TEXT_EXACT_CRITERION,
    RepeatCriterion,
    RepeatHit,
    find_approximate_internal_repeat,
    find_internal_repeat,
    find_repeat,
    induction_headline,
    scan_for_repeats,
)

#: An approximate protein criterion with the BLOSUM62 rule removed, used to
#: separate "the substitution cap rejected this" from "BLOSUM62 rejected this".
PROTEIN_APPROXIMATE_NO_SIMILARITY = replace(
    PROTEIN_APPROXIMATE_CRITERION, similarity="identity"
)


def conservative_partner(residue: str) -> str:
    """The residue BLOSUM62 scores highest against ``residue``, excluding itself."""

    row = BLOSUM62[AA20.index(residue)]
    order = sorted(
        (other for other in AA20 if other != residue),
        key=lambda other: (-int(row[AA20.index(other)]), other),
    )
    return order[0]


def adverse_partner(residue: str) -> str:
    """The residue BLOSUM62 penalises most against ``residue``."""

    row = BLOSUM62[AA20.index(residue)]
    order = sorted(
        (other for other in AA20 if other != residue),
        key=lambda other: (int(row[AA20.index(other)]), other),
    )
    return order[0]


def random_residues(rng: np.random.Generator, count: int) -> str:
    return "".join(rng.choice(list(AA20), size=count))


def tandem(rng: np.random.Generator, first: str, second: str, flank: int = 40) -> str:
    return random_residues(rng, flank) + first + second + random_residues(rng, flank)


# ------------------------------------------------------------------- BLOSUM62


def test_blosum62_is_symmetric_and_matches_publication() -> None:
    assert BLOSUM62.shape == (20, 20)
    assert (BLOSUM62 == BLOSUM62.T).all()
    assert sorted(BLOSUM62_ORDER) == sorted(AA20)
    for left, right, expected in (
        ("A", "A", 4),
        ("W", "W", 11),
        ("C", "C", 9),
        ("I", "V", 3),
        ("F", "Y", 3),
        ("K", "R", 2),
        ("P", "W", -4),
    ):
        assert int(BLOSUM62[AA20.index(left), AA20.index(right)]) == expected


def test_blosum62_diagonal_dominates_its_row() -> None:
    """A residue must be its own best match, or "conservative" means nothing."""

    for index in range(20):
        assert BLOSUM62[index, index] == BLOSUM62[index].max()


# ----------------------------------------------------------------- criteria


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "sloppy"},
        {"similarity": "cosine"},
        {"max_substitution_rate": 1.0},
        {"max_substitution_rate": -0.1},
        {"min_unit": 1},
        {"max_gap_ratio": 0.5},
        {"min_distinct": 0},
        {"max_substitution_rate": 1.0 / 3.0},
    ],
)
def test_criterion_rejects_incoherent_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        replace(PROTEIN_APPROXIMATE_CRITERION, **kwargs)


def test_exact_criterion_may_not_tolerate_substitution() -> None:
    with pytest.raises(ValueError):
        replace(PROTEIN_EXACT_CRITERION, max_substitution_rate=0.5)
    with pytest.raises(ValueError):
        replace(PROTEIN_EXACT_CRITERION, similarity="blosum62_nonadverse")


def test_approximate_criterion_must_tolerate_substitution() -> None:
    with pytest.raises(ValueError):
        replace(PROTEIN_APPROXIMATE_CRITERION, max_substitution_rate=0.0)


def test_declared_criteria_follow_the_prior_work_scope() -> None:
    """Two occurrences, no indels, at most 50 per cent substitution."""

    for criterion in (PROTEIN_APPROXIMATE_CRITERION, TEXT_APPROXIMATE_CRITERION):
        record = criterion.as_dict()
        assert record["occurrences"] == 2
        assert record["indels"] is False
        assert record["max_substitution_rate"] == 0.5
    # Geometry is held at the exact probe's values so exactly one clause moves.
    for approximate, exact in (
        (PROTEIN_APPROXIMATE_CRITERION, PROTEIN_EXACT_CRITERION),
        (TEXT_APPROXIMATE_CRITERION, TEXT_EXACT_CRITERION),
    ):
        assert approximate.min_unit == exact.min_unit
        assert approximate.max_gap_ratio == exact.max_gap_ratio
        assert approximate.min_distinct == exact.min_distinct


def test_approximate_search_refuses_an_exact_criterion() -> None:
    with pytest.raises(ValueError):
        find_approximate_internal_repeat("A" * 200, PROTEIN_EXACT_CRITERION)


def test_amino_acid_criterion_rejects_a_non_residue_symbol() -> None:
    with pytest.raises(ValueError):
        find_approximate_internal_repeat("X" * 200, PROTEIN_APPROXIMATE_CRITERION)


# ------------------------------------------------------------------- geometry


def check_hit(symbols: str, hit: RepeatHit, criterion: RepeatCriterion) -> None:
    """Every clause of the criterion, re-derived from the returned coordinates."""

    first = symbols[hit.first_start : hit.first_start + hit.length]
    second = symbols[hit.second_start : hit.second_start + hit.length]
    assert len(first) == len(second) == hit.length, "an indel would break equal lengths"
    gap = hit.second_start - hit.first_start
    assert hit.length <= gap <= criterion.max_gap_ratio * hit.length
    assert hit.length >= criterion.min_unit
    assert len(set(first)) >= criterion.min_distinct
    substituted = [i for i in range(hit.length) if first[i] != second[i]]
    assert len(substituted) == hit.substituted
    assert hit.substituted <= criterion.max_substitution_rate * hit.length
    assert hit.identity_fraction >= 1.0 - criterion.max_substitution_rate
    if criterion.similarity == "blosum62_nonadverse" and substituted:
        scores = [BLOSUM62[AA20.index(first[i]), AA20.index(second[i])] for i in substituted]
        assert float(np.mean(scores)) >= 0.0
        assert hit.mean_blosum62_substituted == pytest.approx(float(np.mean(scores)))


def test_exact_tandem_repeat_is_found_by_both_criteria() -> None:
    rng = np.random.default_rng(11)
    unit = random_residues(rng, 24)
    symbols = tandem(rng, unit, unit)
    exact = find_repeat(symbols, PROTEIN_EXACT_CRITERION)
    approximate = find_repeat(symbols, PROTEIN_APPROXIMATE_CRITERION)
    assert exact is not None and approximate is not None
    assert approximate.length >= exact.length
    assert approximate.substituted == 0
    assert approximate.mean_blosum62_substituted is None
    check_hit(symbols, approximate, PROTEIN_APPROXIMATE_CRITERION)


def test_approximate_subsumes_exact_on_generated_records() -> None:
    """The nesting the whole comparison rests on, checked over many records.

    This is the property the first implementation got wrong: it selected the
    longest window at each period and then applied the gates, so a chance
    alignment elsewhere in the record could shadow the real repeat.
    """

    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(60):
        unit = random_residues(rng, int(rng.integers(16, 40)))
        symbols = tandem(rng, unit, unit, flank=int(rng.integers(20, 120)))
        exact = find_repeat(symbols, PROTEIN_EXACT_CRITERION)
        if exact is None:
            continue
        checked += 1
        approximate = find_repeat(symbols, PROTEIN_APPROXIMATE_CRITERION)
        assert approximate is not None, "an exact repeat must satisfy the approximate criterion"
        assert approximate.length >= exact.length
    assert checked >= 40, "the generator stopped producing exact repeats"


def test_conservative_substitutions_are_accepted() -> None:
    rng = np.random.default_rng(3)
    unit = random_residues(rng, 32)
    diverged = list(unit)
    for index in rng.choice(32, size=16, replace=False):
        diverged[index] = conservative_partner(unit[index])
    symbols = tandem(rng, unit, "".join(diverged))
    hit = find_repeat(symbols, PROTEIN_APPROXIMATE_CRITERION)
    assert hit is not None
    assert hit.substituted > 0
    check_hit(symbols, hit, PROTEIN_APPROXIMATE_CRITERION)
    assert find_repeat(symbols, PROTEIN_EXACT_CRITERION) is None


def test_adverse_substitutions_are_rejected_by_the_blosum_rule_only() -> None:
    """The similarity rule, not the substitution cap, is what rejects these.

    The same record is accepted once the rule is dropped, which is exactly the
    difference between the protein criterion and its text analogue.
    """

    rng = np.random.default_rng(7)
    unit = random_residues(rng, 32)
    diverged = list(unit)
    for index in rng.choice(32, size=16, replace=False):
        diverged[index] = adverse_partner(unit[index])
    symbols = tandem(rng, unit, "".join(diverged))
    assert find_repeat(symbols, PROTEIN_APPROXIMATE_CRITERION) is None
    permissive = find_repeat(symbols, PROTEIN_APPROXIMATE_NO_SIMILARITY)
    assert permissive is not None
    check_hit(symbols, permissive, PROTEIN_APPROXIMATE_NO_SIMILARITY)


def test_text_control_is_strictly_more_permissive_than_the_protein_criterion() -> None:
    """No BLOSUM62 analogue exists for text, so the text rule is 'no rule'."""

    assert TEXT_APPROXIMATE_CRITERION.similarity == "identity"
    assert PROTEIN_APPROXIMATE_CRITERION.similarity == "blosum62_nonadverse"
    assert (
        TEXT_APPROXIMATE_CRITERION.max_substitution_rate
        == PROTEIN_APPROXIMATE_CRITERION.max_substitution_rate
    )


def test_substitution_cap_is_binding() -> None:
    """More than half substituted must be rejected however conservative it is."""

    rng = np.random.default_rng(13)
    unit = random_residues(rng, 20)
    diverged = [conservative_partner(residue) for residue in unit]
    symbols = tandem(rng, unit, "".join(diverged))
    hit = find_repeat(symbols, PROTEIN_APPROXIMATE_CRITERION)
    if hit is not None:
        # Any accepted window is a sub-window that still respects the cap.
        check_hit(symbols, hit, PROTEIN_APPROXIMATE_CRITERION)
        assert hit.substituted <= 0.5 * hit.length


def test_low_complexity_runs_are_rejected() -> None:
    rng = np.random.default_rng(17)
    for unit in ("A" * 40, "AG" * 20, "QQQQQE" * 8):
        symbols = tandem(rng, unit, unit)
        for criterion in (PROTEIN_EXACT_CRITERION, PROTEIN_APPROXIMATE_CRITERION):
            hit = find_repeat(symbols, criterion)
            if hit is not None:
                # Whatever was returned must still clear the complexity floor.
                window = symbols[hit.first_start : hit.first_start + hit.length]
                assert len(set(window)) >= criterion.min_distinct


def test_non_tandem_pair_is_rejected() -> None:
    """A repeat whose copies are far apart is not a tandem repeat."""

    rng = np.random.default_rng(23)
    unit = random_residues(rng, 20)
    symbols = random_residues(rng, 30) + unit + random_residues(rng, 400) + unit
    for criterion in (PROTEIN_EXACT_CRITERION, PROTEIN_APPROXIMATE_CRITERION):
        hit = find_repeat(symbols, criterion)
        if hit is not None:
            gap = hit.second_start - hit.first_start
            assert gap <= criterion.max_gap_ratio * hit.length


def test_text_criterion_finds_a_diverged_span() -> None:
    rng = np.random.default_rng(29)
    alphabet = list("abcdefghijklmnopqrstuvwxyz ")
    unit = "".join(rng.choice(alphabet, size=60))
    diverged = list(unit)
    for index in rng.choice(60, size=24, replace=False):
        diverged[index] = str(rng.choice(alphabet))
    filler = "".join(rng.choice(alphabet, size=200))
    symbols = filler + unit + "".join(diverged) + filler
    hit = find_repeat(symbols, TEXT_APPROXIMATE_CRITERION)
    assert hit is not None
    check_hit(symbols, hit, TEXT_APPROXIMATE_CRITERION)


# ----------------------------------------------------------------------- scan


def test_scan_is_invariant_to_worker_count() -> None:
    rng = np.random.default_rng(31)
    records = []
    for _ in range(24):
        unit = random_residues(rng, 24)
        records.append(tandem(rng, unit, unit))
        records.append(random_residues(rng, 300))
    single = scan_for_repeats(records, PROTEIN_APPROXIMATE_CRITERION, workers=1)
    parallel = scan_for_repeats(records, PROTEIN_APPROXIMATE_CRITERION, workers=4)
    assert single == parallel
    assert any(hit is not None for hit in single)


def test_scan_rejects_an_empty_corpus_and_zero_workers() -> None:
    with pytest.raises(ValueError):
        scan_for_repeats([], PROTEIN_APPROXIMATE_CRITERION, workers=1)
    with pytest.raises(ValueError):
        scan_for_repeats(["A" * 100], PROTEIN_APPROXIMATE_CRITERION, workers=0)


def test_exact_search_is_untouched_by_the_new_code_path() -> None:
    """``find_repeat`` under an exact criterion must be the original search.

    The exact column is the baseline the approximate column is read against, so
    it has to be the same number the census reported before this probe existed.
    """

    rng = np.random.default_rng(37)
    for _ in range(30):
        unit = random_residues(rng, int(rng.integers(16, 40)))
        symbols = tandem(rng, unit, unit, flank=int(rng.integers(0, 100)))
        legacy = find_internal_repeat(symbols, min_unit=16, max_gap_ratio=2.0, min_distinct=8)
        current = find_repeat(symbols, PROTEIN_EXACT_CRITERION)
        if legacy is None:
            assert current is None
        else:
            assert current is not None and current.coordinates == legacy


# ------------------------------------------------------------------- headline


def test_induction_headline_normalises_by_baseline_and_head_count() -> None:
    alignment = {
        "n_probes": 32,
        "scored_query_positions": 900,
        "mean_coverage": 0.9,
        "uniform_baseline": 0.005,
        "kind": "natural_repeat_approximate",
    }
    census = {
        "distribution": {"max": 0.5, "n_heads": 432},
        "count_above_threshold": {"0.05": 9, "0.10": 6, "0.20": 5, "0.30": 4},
        "count_above_data_driven": 5,
    }
    headline = induction_headline(alignment, census, threshold=0.10)
    assert headline["peak_over_uniform"] == pytest.approx(100.0)
    assert headline["fraction_above_threshold"] == pytest.approx(6 / 432)
    with pytest.raises(KeyError):
        induction_headline(alignment, census, threshold=0.15)


def test_induction_headline_rejects_a_degenerate_baseline() -> None:
    alignment = {
        "n_probes": 1,
        "scored_query_positions": 1,
        "mean_coverage": 1.0,
        "uniform_baseline": 0.0,
    }
    census = {
        "distribution": {"max": 0.5, "n_heads": 8},
        "count_above_threshold": {"0.10": 1},
        "count_above_data_driven": 1,
    }
    with pytest.raises(ValueError):
        induction_headline(alignment, census, threshold=0.10)
