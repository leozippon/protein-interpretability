"""What the spectrum estimator must get right before a rank claim rests on it.

The claim this instrument is built to support is a refusal: that basis-adequacy
gate B2 asks a dictionary for more live latents than the activations have
directions to put them in. That claim is only as good as the estimator, and
every way it can be wrong produces a plausible-looking number rather than an
error.

**A full-rank cloud must read as full rank.** An estimator that deflates ``r99``
on isotropic data would report a low effective dimension everywhere, and the
conclusion would be about the instrument. This is the isotropic control the stage
refuses its own verdict without, tested here at the same ``N/d`` ratio the
campaign runs at.

**An injected rank must come back exactly.** Not the variance cut -- ``r99`` and
even ``r999`` sit *below* an injected rank whenever the cloud's own eigenvalues
are spread, which is a property of a variance cut and not a defect -- but the
algebraic rank, which must be the number of directions that were put in and must
be separated from the null space by a gap no rounding could close.

**The variance cuts must be arithmetic, not approximation.** Checked against a
spectrum whose ranks can be computed by hand.

**A budget below the ambient dimension must be visible.** A covariance from
``N < d`` samples is rank-limited by its own sampling, and a spectrum that did
not carry that fact would be indistinguishable from a genuinely low-dimensional
one.

**The position draw must spend its budget on records and not on the longest
records.** An unequal cap would weight long proteins and long documents, and
token positions inside one record are strongly correlated, so the covariance
would be an estimate of a few records' geometry reported as a corpus's.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.transfer.spectrum import (
    CovarianceAccumulator,
    coordinate_independent_spectrum,
    isotropic_control_spectrum,
    sample_positions,
    spectrum_statistics,
)


def _low_rank_cloud(
    *, d_model: int, rank: int, n_samples: int, offset_scale: float, seed: int
) -> CovarianceAccumulator:
    """A cloud that lives in exactly ``rank`` directions, far from the origin.

    The offset is not decoration. A feed-forward output carries coordinates whose
    mean dwarfs their spread, which is where an uncentred second-moment
    accumulator loses its significant digits; putting one here is what makes the
    accumulator's fixed shift a tested property rather than a comment.
    """

    generator = torch.Generator().manual_seed(seed)
    basis = torch.randn((rank, d_model), generator=generator, dtype=torch.float64)
    offset = offset_scale * torch.randn(
        (d_model,), generator=generator, dtype=torch.float64
    )
    accumulator = CovarianceAccumulator(n_layers=1, d_model=d_model, device="cpu")
    drawn = 0
    while drawn < n_samples:
        rows = min(1024, n_samples - drawn)
        coefficients = torch.randn(
            (rows, rank), generator=generator, dtype=torch.float64
        )
        accumulator.update((coefficients @ basis + offset)[None])
        drawn += rows
    return accumulator


def test_isotropic_control_recovers_nearly_full_rank() -> None:
    generator = torch.Generator().manual_seed(20260812)
    d_model = 256
    summary = isotropic_control_spectrum(
        total_variance=3.0 * d_model,
        d_model=d_model,
        n_samples=16 * d_model,
        chunk=1024,
        device="cpu",
        generator=generator,
    )
    # The stage's own refusal threshold. Finite sampling spreads the spectrum of
    # exactly isotropic data, so this sits a little below d_model and must not
    # sit far below it.
    assert summary["r99"] >= 0.95 * d_model
    assert summary["r99"] <= d_model
    assert summary["sample_rank_ceiling"] == d_model
    assert summary["total_variance"] == pytest.approx(3.0 * d_model, rel=0.05)


def test_injected_algebraic_rank_is_recovered_exactly() -> None:
    d_model, rank = 192, 31
    accumulator = _low_rank_cloud(
        d_model=d_model, rank=rank, n_samples=8192, offset_scale=25.0, seed=11
    )
    eigenvalues = torch.linalg.eigvalsh(accumulator.covariance_at(0)).numpy()
    ordered = np.sort(eigenvalues)[::-1]
    assert int((ordered > ordered[0] * 1e-10).sum()) == rank
    # A gap of many orders of magnitude, not a threshold that happened to work.
    assert ordered[rank - 1] / max(abs(ordered[rank]), 1e-300) > 1e8
    summary = spectrum_statistics(eigenvalues, n_samples=8192, d_model=d_model)
    assert summary["cumulative_mass"]["32"] == pytest.approx(1.0, abs=1e-9)
    # The variance cuts sit at or below the algebraic rank and never above it.
    assert summary["r95"] <= summary["r99"] <= summary["r999"] <= rank
    assert summary["negative_eigenvalue_mass"] < 1e-9 * summary["total_variance"]


def test_the_fixed_shift_survives_a_large_offset() -> None:
    """The same cloud at two offsets must give the same spectrum.

    A covariance does not depend on where the cloud sits, so an estimator whose
    answer moves when the cloud is translated is losing digits rather than
    measuring geometry.
    """

    near = _low_rank_cloud(
        d_model=128, rank=9, n_samples=4096, offset_scale=0.0, seed=5
    )
    far = _low_rank_cloud(
        d_model=128, rank=9, n_samples=4096, offset_scale=1.0e5, seed=5
    )
    near_eigenvalues = torch.linalg.eigvalsh(near.covariance_at(0))
    far_eigenvalues = torch.linalg.eigvalsh(far.covariance_at(0))
    assert torch.allclose(near_eigenvalues, far_eigenvalues, rtol=1e-8, atol=1e-8)
    assert far.mean_shift_residual() < 1.0e-3 * 1.0e5


def test_variance_cuts_are_arithmetic() -> None:
    d_model = 64
    # Four eigenvalues of 10 and sixty of 1: total 100. 95% needs 40 + 55 ones,
    # so r95 = 59; 99% needs four more, so r99 = 63; 99.9% needs the last, 64.
    eigenvalues = np.array([10.0] * 4 + [1.0] * 60)
    summary = spectrum_statistics(eigenvalues, n_samples=100_000, d_model=d_model)
    assert (summary["r95"], summary["r99"], summary["r999"]) == (59, 63, 64)
    assert summary["participation_ratio"] == pytest.approx(100.0**2 / (4 * 100 + 60))
    assert summary["sample_rank_ceiling"] == d_model
    assert summary["cumulative_mass"]["1"] == pytest.approx(0.1)


def test_sample_rank_ceiling_reports_a_budget_below_the_ambient_dimension() -> None:
    summary = spectrum_statistics(
        np.ones(64), n_samples=10, d_model=64
    )
    assert summary["sample_rank_ceiling"] == 10
    assert summary["n_samples"] == 10


def test_coordinate_independent_null_is_the_diagonal_and_sees_no_correlation() -> None:
    """The control must be blind to exactly what the measurement is sensitive to."""

    accumulator = _low_rank_cloud(
        d_model=128, rank=8, n_samples=8192, offset_scale=0.0, seed=17
    )
    covariance = accumulator.covariance_at(0)
    observed = spectrum_statistics(
        torch.linalg.eigvalsh(covariance), n_samples=8192, d_model=128
    )
    null = coordinate_independent_spectrum(covariance, n_samples=8192, d_model=128)
    assert observed["r99"] <= 8
    # Destroying the correlations restores nearly every direction: the cloud is
    # low-dimensional because its coordinates covary, not because they have
    # unequal variances.
    assert null["r99"] > 100
    assert null["total_variance"] == pytest.approx(observed["total_variance"])


def test_spectrum_statistics_refuses_a_truncated_spectrum() -> None:
    with pytest.raises(ValueError, match="expected 64 eigenvalues"):
        spectrum_statistics(np.ones(32), n_samples=1024, d_model=64)


def test_accumulator_refuses_a_mismatched_block() -> None:
    accumulator = CovarianceAccumulator(n_layers=3, d_model=16, device="cpu")
    with pytest.raises(ValueError, match="was handed"):
        accumulator.update(torch.zeros((2, 5, 16), dtype=torch.float64))
    with pytest.raises(RuntimeError, match="at least two samples"):
        accumulator.covariance_at(0)


def test_position_draw_caps_every_record_equally_and_refuses_short_ones() -> None:
    mask = torch.zeros((5, 20), dtype=torch.bool)
    mask[0, :10] = True          # exactly the cap
    mask[1, 3:19] = True         # more than the cap, so a choice is made
    mask[2, :2] = True           # below the cap
    mask[3, ::2] = True          # the cap, scattered rather than contiguous
    mask[4, :12] = True
    indices, used, short = sample_positions(
        mask, per_record=10, generator=torch.Generator().manual_seed(3)
    )
    assert used == [0, 1, 3, 4]
    assert short == [2]
    assert indices.numel() == 40
    assert len(set(indices.tolist())) == 40

    # Every index must land inside the flattened run of its own record, which is
    # what makes the cap a per-record cap rather than a global one.
    counts = mask.sum(dim=1)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]])
    for row, block in zip(used, indices.split(10)):
        assert (block >= offsets[row]).all()
        assert (block < offsets[row] + counts[row]).all()


def test_position_draw_is_seeded_and_not_a_prefix() -> None:
    mask = torch.ones((2, 64), dtype=torch.bool)
    first, _, _ = sample_positions(
        mask, per_record=8, generator=torch.Generator().manual_seed(1)
    )
    second, _, _ = sample_positions(
        mask, per_record=8, generator=torch.Generator().manual_seed(2)
    )
    repeat, _, _ = sample_positions(
        mask, per_record=8, generator=torch.Generator().manual_seed(1)
    )
    assert torch.equal(first, repeat)
    assert not torch.equal(first, second)
    # Taking the first positions of each record would sample the start of every
    # protein and every document, which is a region and not a sample.
    assert not torch.equal(first[:8], torch.arange(8))
