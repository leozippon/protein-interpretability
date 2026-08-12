"""Tests for the routing-locality statistics.

The load-bearing ones are the negative paths. This module's whole purpose is to
say whether a grouping of tokens knows something about a replacement's residual,
and the way that goes wrong is a grouping with enough cells reading as
informative because it memorised the tokens it was fitted on. So the tests that
matter are: a grouping that knows nothing must read as knowing nothing on held-out
tokens *while its in-sample ceiling is large*, and the margin must be the distance
to the routing boundary rather than the gap inside the selected pair.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.transfer.routing import (
    DEGENERATE_CELL_SHARE,
    MINIMUM_CELL_TOKENS,
    boundary_cells,
    cell_occupancy,
    expert_set_from_residue,
    expert_sets,
    fit_correction,
    grouping_reduction,
    normalised_error,
    router_dispersion,
    routing_cells,
)


def test_expert_sets_enumerates_every_unordered_selection() -> None:
    sets = expert_sets(8, 2)
    assert len(sets) == 28
    assert sets[0] == (0, 1)
    assert len(set(sets)) == len(sets)
    assert all(tuple(sorted(s)) == s for s in sets)


def test_expert_sets_refuses_an_impossible_top_k() -> None:
    with pytest.raises(ValueError):
        expert_sets(8, 0)
    with pytest.raises(ValueError):
        expert_sets(4, 5)


def test_routing_cells_labels_the_selected_pair() -> None:
    probs = np.array(
        [
            [0.5, 0.3, 0.1, 0.1],  # experts {0, 1}
            [0.1, 0.1, 0.4, 0.4],  # experts {2, 3}
            [0.35, 0.05, 0.55, 0.05],  # experts {0, 2}
        ]
    )
    vocabulary = expert_sets(4, 2)
    cells = routing_cells(probs, top_k=2)
    assert [vocabulary[c] for c in cells] == [(0, 1), (2, 3), (0, 2)]


def test_routing_cells_refuses_a_wrong_shape() -> None:
    with pytest.raises(ValueError):
        routing_cells(np.zeros(8), top_k=2)


def test_margin_is_the_distance_to_the_boundary_not_the_gap_inside_the_pair() -> None:
    """The two are different numbers and only one is about routing.

    This token's selected pair is lopsided -- 0.60 against 0.20 -- while the
    second-choice expert sits 0.001 below the pair's weaker member. Crossing the
    inside gap changes nothing; crossing the boundary swaps an expert. A margin
    that reported 0.40 here would call this token confidently routed when it is
    one thousandth of a probability away from routing elsewhere.
    """

    probs = np.array([[0.60, 0.20, 0.199, 0.001]])
    dispersion = router_dispersion(probs, top_k=2)
    assert dispersion["margin_mean"] == pytest.approx(0.001, abs=1e-9)
    assert dispersion["entropy_max_nats"] == pytest.approx(np.log(4))


def test_cell_occupancy_flags_a_router_with_no_variation_to_spend() -> None:
    cells = np.zeros(1000, dtype=np.int64)
    cells[:10] = 1
    record = cell_occupancy(cells, 28)
    assert record["largest_cell_share"] >= DEGENERATE_CELL_SHARE
    assert record["degenerate"] is True
    assert record["degenerate_reason"]
    assert record["n_occupied"] == 2

    spread = np.arange(1000) % 28
    healthy = cell_occupancy(spread, 28)
    assert healthy["degenerate"] is False
    assert healthy["degenerate_reason"] is None
    assert healthy["cells_above_minimum"] == 28


def _tied_margin_router(n_tied: int, n_free: int) -> np.ndarray:
    """A router that leaves ``n_tied`` tokens at an identical margin.

    Each row is a proper distribution over four experts. The tied rows put the
    second and third weights on the same value, so with ``top_k = 2`` their
    distance to the routing boundary is exactly zero; the free rows carry
    distinct positive margins.
    """

    tied = np.tile([0.40, 0.30, 0.30, 0.00], (n_tied, 1))
    margins = np.linspace(0.01, 0.19, n_free)
    free = np.stack(
        [
            np.full(n_free, 0.40),
            0.30 + margins / 2.0,
            0.30 - margins / 2.0,
            np.zeros(n_free),
        ],
        axis=1,
    )
    return np.concatenate([tied, free])


def test_boundary_cells_attain_far_fewer_cells_than_they_request() -> None:
    """Quantile bins collapse on tied margins, and ``degenerate`` does not see it.

    ``n_cells`` is the cardinality asked for, not the one attained: a router that
    leaves 90% of its tokens at an identical margin has one quantile edge repeated
    twenty-five times, so twenty-eight requested cells become three occupied ones.
    A ``boundary`` grouping in that state can remove nothing on held-out tokens
    however much proximity to a routing boundary carries, so its null is
    uninterpretable -- which is the whole reason attainability is measured before
    a result is read (standing rule 2).

    The share it concentrates at, 0.929, sits *below* the concentration threshold,
    so ``degenerate`` reads False here and ``n_occupied`` is the field that carries
    the failure. Both are asserted, because a future change that made the flag
    catch this case must not do so silently.
    """

    cells = boundary_cells(_tied_margin_router(900, 100), top_k=2, n_cells=28)
    record = cell_occupancy(cells, 28)

    assert record["n_cells"] == 28
    assert record["n_occupied"] == 3
    assert record["largest_cell_share"] == pytest.approx(0.929, abs=5e-3)
    assert record["largest_cell_share"] < DEGENERATE_CELL_SHARE
    assert record["degenerate"] is False


def test_boundary_cells_use_the_full_grid_when_the_margins_are_spread() -> None:
    """The positive path: with no ties every requested cell is occupied."""

    cells = boundary_cells(_tied_margin_router(0, 1000), top_k=2, n_cells=28)
    record = cell_occupancy(cells, 28)

    assert record["n_occupied"] == 28
    assert record["largest_cell_share"] < 0.1
    assert record["degenerate"] is False


def test_fit_correction_leaves_undersized_cells_at_zero() -> None:
    residual = np.ones((MINIMUM_CELL_TOKENS + 4, 3))
    cells = np.zeros(residual.shape[0], dtype=np.int64)
    cells[MINIMUM_CELL_TOKENS:] = 1  # four tokens, below the floor
    correction = fit_correction(residual, cells, 2)
    assert correction.fitted_cells == 1
    assert np.allclose(correction.means[0], 1.0)
    assert np.allclose(correction.means[1], 0.0)


def test_fit_correction_refuses_misaligned_inputs() -> None:
    with pytest.raises(ValueError):
        fit_correction(np.zeros((10, 3)), np.zeros(9, dtype=np.int64), 2)


def test_normalised_error_refuses_a_non_positive_denominator() -> None:
    with pytest.raises(ValueError):
        normalised_error(np.zeros((4, 2)), 0.0)


def _split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return order[: n // 2], order[n // 2 :]


def test_an_uninformative_grouping_reads_as_uninformative_out_of_sample() -> None:
    """The negative path, and the reason the held-out split exists.

    The residual is pure noise and the cells are assigned at random, so the
    grouping knows nothing. Held out it must not *gain*; in sample it gains
    anyway, because twenty-eight means fitted to the tokens they are scored on
    always remove something. The two are near mirror images at ``n_cells /
    n_train`` of the variance, which is the whole argument for reporting the
    in-sample number as a ceiling and never as a result.
    """

    rng = np.random.default_rng(11)
    residual = rng.normal(size=(4000, 16))
    cells = rng.integers(0, 28, size=4000)
    train, test = _split(4000, seed=3)
    record = grouping_reduction(
        residual, cells, 28, train=train, test=test, target_variance=1.0
    )
    expected = 28 / train.size
    assert record["reduction"] <= 0.0
    assert record["reduction"] == pytest.approx(-expected, abs=0.5 * expected)
    assert record["in_sample_ceiling_reduction"] == pytest.approx(expected, abs=0.5 * expected)


def test_a_grouping_that_carries_the_residual_removes_it_out_of_sample() -> None:
    rng = np.random.default_rng(12)
    cells = rng.integers(0, 28, size=4000)
    offsets = rng.normal(size=(28, 16)) * 3.0
    residual = offsets[cells] + rng.normal(size=(4000, 16)) * 0.5
    train, test = _split(4000, seed=4)
    variance = float(residual.var())
    record = grouping_reduction(
        residual, cells, 28, train=train, test=test, target_variance=variance
    )
    assert record["reduction_fraction"] > 0.9
    assert record["cells_fitted_on_train"] == 28


def test_expert_set_from_residue_separates_a_lookup_from_a_context_decision() -> None:
    rng = np.random.default_rng(13)
    residues = rng.integers(0, 20, size=3000)
    train, test = _split(3000, seed=5)

    lookup = residues % 28  # routing is a pure function of the residue
    deterministic = expert_set_from_residue(lookup, residues, train=train, test=test)
    assert deterministic["accuracy_from_residue"] == pytest.approx(1.0)
    assert deterministic["skill_over_majority"] == pytest.approx(1.0)

    independent = rng.integers(0, 28, size=3000)
    contextual = expert_set_from_residue(independent, residues, train=train, test=test)
    assert contextual["skill_over_majority"] < 0.15
