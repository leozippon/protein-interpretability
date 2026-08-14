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

import importlib.util
import sys
from pathlib import Path

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


def _load_stage(filename: str):
    """Import a stage whose module name starts with a digit, as the stages do."""

    path = Path(__file__).resolve().parents[1] / "scripts" / "transfer" / filename
    spec = importlib.util.spec_from_file_location(f"_spectrum_stage_{filename[:2]}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _spectra(r99_values: list[int], *, d_model: int) -> dict[str, list[dict]]:
    """A spectrum artefact shaped like the stage's, carrying chosen ``r99`` values."""

    return {
        site: [
            {"layer": i, "observed": {"r95": v, "r99": v, "r999": v}}
            for i, v in enumerate(r99_values)
        ]
        for site in ("block_input", "block_output")
    }


def _controls(*, passes: bool) -> dict[str, dict]:
    return {
        site: {"isotropic_passes": passes} for site in ("block_input", "block_output")
    }


def test_the_verdict_is_not_inverted() -> None:
    """A verdict that reads the wrong way round is invisible in every other number.

    The whole point of this stage is a comparison against a threshold, so the
    direction of that comparison is the one thing no artefact reveals by
    inspection: an inverted rule produces a complete, well-formed, plausible
    record. It is pinned here in both directions and at the boundary.
    """

    stage = _load_stage("30_activation_spectrum.py")
    d_model = 4096

    below = stage.verdict_record(
        _spectra([1000] * 32, d_model=d_model), _controls(passes=True), d_model=d_model
    )
    assert below["reading"] == "R99_MEDIAN_BELOW_D_MODEL"
    assert below["r99_median"] == 1000

    above = stage.verdict_record(
        _spectra([4096] * 32, d_model=d_model), _controls(passes=True), d_model=d_model
    )
    assert above["reading"] == "R99_MEDIAN_AT_OR_ABOVE_D_MODEL", (
        "the threshold is stated as >= d_model, so exactly d_model must read as "
        "at-or-above"
    )

    # One layer either side of the threshold must not decide a median verdict.
    mixed = stage.verdict_record(
        _spectra([4095] * 17 + [4096] * 15, d_model=d_model),
        _controls(passes=True),
        d_model=d_model,
    )
    assert mixed["reading"] == "R99_MEDIAN_BELOW_D_MODEL"
    assert mixed["r99_median"] == 4095

    # And a failed instrument control overrides the comparison in both directions,
    # rather than being reported beside a verdict it does not support.
    for values in ([1000] * 32, [4096] * 32):
        broken = stage.verdict_record(
            _spectra(values, d_model=d_model), _controls(passes=False), d_model=d_model
        )
        assert broken["reading"] == "INSTRUMENT_CONTROL_FAILED"
        assert broken["instrument_controls_pass"] is False


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


def test_leading_positions_can_be_excluded_from_the_draw() -> None:
    """The interior-position rule, as a testable exclusion rather than an argument.

    A rank claim that rests on an early layer has to be separable from a massive
    activation sitting on that layer's first content positions, so the exclusion
    must actually shift the pool and must charge for itself in record depth.
    """

    mask = torch.zeros((3, 40), dtype=torch.bool)
    mask[0, :20] = True
    mask[1, 5:35] = True     # 30 scored
    mask[2, :12] = True      # 12 scored: enough for a cap of 10 alone, not with a drop
    counts = mask.sum(dim=1)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]])

    kept, used, short = sample_positions(
        mask, per_record=10, generator=torch.Generator().manual_seed(3), drop_leading=4
    )
    assert used == [0, 1]
    assert short == [2], "a record that no longer has room for the cap must be refused"
    for row, block in zip(used, kept.split(10)):
        assert (block >= offsets[row] + 4).all(), "an excluded leading position was drawn"
        assert (block < offsets[row] + counts[row]).all()

    # And with no exclusion the same record is usable, so the refusal above is
    # the exclusion's doing and not the mask's.
    _, used_none, short_none = sample_positions(
        mask, per_record=10, generator=torch.Generator().manual_seed(3), drop_leading=0
    )
    assert used_none == [0, 1, 2] and short_none == []

    with pytest.raises(ValueError, match="drop_leading cannot be negative"):
        sample_positions(
            mask, per_record=2, generator=torch.Generator().manual_seed(0), drop_leading=-1
        )


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
