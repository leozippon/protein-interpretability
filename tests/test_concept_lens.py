"""Ways the concept lens could report a coarsening as a concept.

Each test corresponds to a failure mode the design names rather than to a line of
implementation:

* a statistic that beats its null because the null is easier -- the rank-matched
  partition has to hold the coarsened entropy, or every "concept" result is the
  entropy drop;
* a readout that is really the alphabet's composition, which within-unit
  centring has to remove *by construction* rather than by control;
* a fast streaming path that quietly disagrees with the direct arithmetic it
  replaces, on an arm nobody will recompute by hand;
* a resolution-depth statistic that drifts from the one ``lenses`` already
  publishes, so two numbers in one repository mean different things under one
  name;
* an aperture gain that exists at one threshold only, which Appendix B rule 17
  exists to catch;
* an empirical p-value of zero, which claims evidence a thousand draws cannot
  carry;
* a target token that emits no residue being scored as if it did.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.arms import AA20  # noqa: E402
from src.transfer.concept_lens import (  # noqa: E402
    CLASS_COUNT_SWEEP,
    PROPERTY_BASIS,
    SymbolAxis,
    aperture_gain,
    basis_correlations,
    class_mass_profile,
    coarsened_cross_entropy,
    equal_mass_classes,
    layer_concept_statistics,
    null_excess,
    partition_null_quality,
    rank_matched_partitions,
    resolution_depth,
    shuffled_property_null,
    within_unit_centred_spearman,
)
from src.transfer.lenses import LensHead, half_resolution_depth  # noqa: E402


def _marginal(seed: int = 0) -> np.ndarray:
    weights = np.random.default_rng(seed).uniform(0.5, 3.0, size=len(AA20))
    return weights / weights.sum()


# ----------------------------------------------------- the declared basis


def test_the_declared_basis_covers_the_alphabet_and_is_not_degenerate() -> None:
    for name, values in PROPERTY_BASIS.items():
        assert set(values) == set(AA20), name
        assert len(set(values.values())) > 1, name


def test_basis_correlations_name_the_risk_before_any_run() -> None:
    """The pre-declaration has to state volume's confound with frequency."""

    report = basis_correlations(_marginal())
    assert report["unweighted"]["charge__volume"]["pearson"] > 0.2
    assert abs(report["unweighted"]["hydropathy__charge"]["pearson"]) < 0.2
    assert "volume__log_frequency" in report["unweighted"]
    assert report["marginal_entropy_bits"] > 0.0

    with pytest.raises(ValueError, match="distribution summing to one"):
        basis_correlations(np.ones(len(AA20)))


# ------------------------------------------------------ nulls that are nulls


@pytest.mark.parametrize("k", CLASS_COUNT_SWEEP)
def test_equal_mass_classes_balance_probability_not_count(k: int) -> None:
    weights = _marginal(3)
    values = np.asarray([PROPERTY_BASIS["hydropathy"][residue] for residue in AA20])
    classes = equal_mass_classes(values, weights, k)
    assert classes.min() == 0 and classes.max() == k - 1
    profile = class_mass_profile(classes, weights)
    # Mass, not count: no class may be more than double the balanced share.
    assert profile.max() < 2.0 / k
    # And the partition must respect the property's order -- a class is an
    # interval of the property, not an arbitrary set.
    order = np.argsort(values)
    assert np.all(np.diff(classes[order]) >= 0)


def test_the_rank_matched_null_holds_the_coarsened_entropy() -> None:
    """The null has to be as easy as the concept, or it measures the coarsening."""

    weights = _marginal(5)
    values = np.asarray([PROPERTY_BASIS["volume"][residue] for residue in AA20])
    declared = equal_mass_classes(values, weights, 3)
    partitions = rank_matched_partitions(declared, weights, draws=64, seed=1)
    quality = partition_null_quality(partitions, weights)
    assert quality["n_classes_declared"] == 3
    assert quality["n_null_partitions_with_fewer_classes"] == 0
    assert quality["max_absolute_entropy_mismatch_bits"] < 0.25
    # Matched in difficulty, different in membership: a null that reproduced the
    # declared partition would be no null at all.
    assert not np.array_equal(partitions[:, 0], partitions[:, 1])


def test_the_shuffled_null_preserves_the_multiset_and_moves_the_assignment() -> None:
    values = np.asarray([PROPERTY_BASIS["charge"][residue] for residue in AA20])
    columns = shuffled_property_null(values, draws=32, seed=2)
    assert np.array_equal(columns[:, 0], values)
    for draw in range(1, columns.shape[1]):
        assert np.array_equal(np.sort(columns[:, draw]), np.sort(values))
    assert not np.array_equal(columns[:, 0], columns[:, 1])


def test_an_empirical_p_value_is_bounded_by_the_draws_taken() -> None:
    null = np.linspace(-1.0, 1.0, 200)
    beats_everything = null_excess(5.0, null, quantile=0.99)
    assert beats_everything["clears_null"] is True
    assert beats_everything["empirical_p"] == pytest.approx(1 / 201)
    assert beats_everything["empirical_p"] > 0.0

    inside = null_excess(0.0, null, quantile=0.99)
    assert inside["clears_null"] is False

    with pytest.raises(ValueError, match="at least two draws"):
        null_excess(1.0, np.asarray([0.5]), quantile=0.99)


# ------------------------------------------- centring removes priors exactly


def test_within_unit_centring_zeroes_a_per_protein_constant_predictor() -> None:
    """The unigram and composition priors are constant along a protein.

    Centring within the protein therefore has to send them to exactly zero. If
    it did not, a lens that had learned nothing but amino-acid composition could
    clear the null, and the design's claim that the statistic is prior-free by
    construction would be false.
    """

    rng = np.random.default_rng(7)
    unit = np.repeat(np.arange(30), 20)
    realised = rng.normal(size=unit.size)
    per_protein_constant = np.repeat(rng.normal(size=30), 20)
    readout = np.stack([per_protein_constant, realised], axis=1)
    scores = within_unit_centred_spearman(readout, realised, unit)
    assert scores[0] == pytest.approx(0.0, abs=1e-12)
    assert scores[1] > 0.99

    with pytest.raises(ValueError, match="aligned"):
        within_unit_centred_spearman(readout, realised[:-1], unit)


def test_a_singleton_protein_cannot_contribute_to_a_centred_statistic() -> None:
    unit = np.asarray([0, 0, 0, 1])
    realised = np.asarray([1.0, 2.0, 3.0, 9.0])
    readout = np.asarray([[1.0], [2.0], [3.0], [-5.0]])
    # The lone position of protein 1 centres to exactly zero on both axes and
    # would otherwise be a free, uninformative agreement.
    assert within_unit_centred_spearman(readout, realised, unit)[0] > 0.99


# ---------------------------------------- the fast path equals the slow one


class _StubHead:
    """A lens head standing in for the trained one, with the same interface."""

    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight

    def log_probs(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(hidden @ self.weight.T, dim=-1)


def _direct_reference(head, residual, axis, targets, properties, partitions):
    log_probs = head.log_probs(residual.float())
    if axis.token_groups is None:
        log_posterior = log_probs
    else:
        stacked = torch.stack(
            [torch.logsumexp(log_probs.index_select(-1, g), dim=-1) for g in axis.token_groups],
            dim=-1,
        )
        log_posterior = stacked - torch.logsumexp(stacked, dim=-1, keepdim=True)
    numpy_posterior = log_posterior.numpy().astype(np.float64)
    readout = {
        name: np.exp(numpy_posterior) @ columns for name, columns in properties.items()
    }
    class_ce = {
        (name, k): coarsened_cross_entropy(numpy_posterior, targets, columns)
        for name, by_k in partitions.items()
        for k, columns in by_k.items()
    }
    symbol_ce = float(-numpy_posterior[np.arange(targets.size), targets].mean())
    return readout, class_ce, symbol_ce


@pytest.mark.parametrize("grouped", [True, False])
def test_the_streaming_path_reproduces_the_direct_arithmetic(grouped: bool) -> None:
    """The optimisation is only admissible if it changes nothing.

    ``layer_concept_statistics`` flattens a thousand null partitions into one
    matrix multiply because looping them turns a seven-minute layer into an hour
    on a 50,257-symbol arm. Nothing about that reindexing is obvious, and no
    reader of an artefact would recompute it, so it is checked here against the
    unoptimised implementation it replaces.
    """

    torch.manual_seed(0)
    n_symbols = len(AA20) if grouped else 37
    width, n = 16, 53
    weight = torch.randn(n_symbols if not grouped else 29, width)
    head = _StubHead(weight)
    residual = torch.randn(n, width)
    if grouped:
        groups = [torch.tensor([i, i + 20]) if i < 9 else torch.tensor([i]) for i in range(20)]
        axis = SymbolAxis("residue", tuple(AA20), tuple(groups), True)
    else:
        axis = SymbolAxis("token", tuple(str(i) for i in range(n_symbols)), None, False)

    weights = np.full(axis.n_symbols, 1.0 / axis.n_symbols)
    values = np.random.default_rng(1).normal(size=axis.n_symbols)
    properties = {"p": shuffled_property_null(values, draws=6, seed=3)}
    partitions = {
        "p": {k: rank_matched_partitions(equal_mass_classes(values, weights, k), weights, draws=4, seed=k)
              for k in (2, 3)}
    }
    targets = np.random.default_rng(2).integers(0, axis.n_symbols, size=n)

    fast = layer_concept_statistics(
        head, residual, axis, targets=targets, properties=properties,
        partitions=partitions, device="cpu", chunk=7,
    )
    slow_readout, slow_ce, slow_symbol = _direct_reference(
        head, residual, axis, targets, properties, partitions
    )
    assert fast["symbol_cross_entropy_nats"] == pytest.approx(slow_symbol, abs=1e-5)
    np.testing.assert_allclose(fast["readout"]["p"], slow_readout["p"], atol=1e-5)
    for key, expected in slow_ce.items():
        np.testing.assert_allclose(fast["class_cross_entropy_nats"][key], expected, atol=1e-5)


def test_the_streaming_path_reports_discarded_mass_rather_than_dividing_it_away() -> None:
    """On ProtGPT2 the abstained mass is the FASTA newline, and §0.1 is what
    happens when a format token disappears into a denominator."""

    torch.manual_seed(1)
    width = 8
    head = _StubHead(torch.randn(25, width))
    groups = tuple(torch.tensor([i]) for i in range(20))
    axis = SymbolAxis("residue", tuple(AA20), groups, True)
    residual = torch.randn(11, width)
    values = np.arange(20, dtype=np.float64)
    statistics = layer_concept_statistics(
        head, residual, axis,
        targets=np.zeros(11, dtype=np.int64),
        properties={"p": shuffled_property_null(values, draws=2, seed=0)},
        partitions={},
        device="cpu", chunk=5,
    )
    # Five of the twenty-five output tokens are outside the alphabet, so real
    # mass is discarded and has to be visible.
    assert statistics["abstain_mass_mean"] > 0.0
    assert statistics["abstain_mass_max"] >= statistics["abstain_mass_mean"]


# ------------------------------------------------- depth statistics agree


@pytest.mark.parametrize("seed", range(8))
def test_resolution_depth_agrees_with_the_published_half_depth(seed: int) -> None:
    """Two numbers in one repository must not mean different things.

    The agreement is now structural -- ``half_resolution_depth`` is one call of
    ``resolution_depth`` -- so what this pins is the fraction that "half" names,
    which no type or signature holds.
    """

    rng = np.random.default_rng(seed)
    depths = np.linspace(0.0, 1.0, 11)
    values = np.sort(rng.uniform(0.0, 5.0, size=11))[::-1]
    assert resolution_depth(depths, values, 0.5) == pytest.approx(
        half_resolution_depth(depths, values)
    )


def test_resolution_depth_refuses_a_trajectory_that_never_resolves() -> None:
    depths = np.linspace(0.0, 1.0, 5)
    assert resolution_depth(depths, [1.0, 1.0, 1.0, 1.0, 1.0], 0.5) is None
    assert resolution_depth(depths, [1.0, 2.0, 3.0, 4.0, 5.0], 0.5) is None
    with pytest.raises(ValueError, match="tau must lie"):
        resolution_depth(depths, [5.0, 4.0, 3.0, 2.0, 1.0], 1.0)


def test_an_aperture_gain_at_one_threshold_only_does_not_survive_the_sweep() -> None:
    depths = [0.0, 0.25, 0.5, 0.75, 1.0]
    symbol = [4.0, 3.9, 3.0, 1.5, 1.0]
    early = [4.0, 2.0, 1.5, 1.2, 1.0]
    gain = aperture_gain(depths, symbol, early)
    assert gain["sign_invariant_across_sweep"] is True
    assert all(
        block["aperture_gain"] > 0 for block in gain["per_tau"].values()
    )

    crossing = [4.0, 3.95, 3.9, 1.2, 1.0]
    mixed = aperture_gain(depths, symbol, crossing)
    signs = {np.sign(block["aperture_gain"]) for block in mixed["per_tau"].values()}
    assert (len(signs) > 1) == (not mixed["sign_invariant_across_sweep"])


def test_an_undefined_trajectory_yields_no_gain_rather_than_a_number() -> None:
    depths = [0.0, 0.5, 1.0]
    gain = aperture_gain(depths, [3.0, 2.0, 1.0], [1.0, 1.0, 1.0])
    assert gain["n_tau_defined"] == 0
    assert gain["sign_invariant_across_sweep"] is False
    assert all(block["aperture_gain"] is None for block in gain["per_tau"].values())
