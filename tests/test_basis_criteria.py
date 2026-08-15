"""Tests for D3.h's adequacy criteria and for the per-layer basis they read.

**The defect these exist against.** R2.4's basis-adequacy gate was pre-registered
as a per-layer condition -- Criterion B must hold "at the layers a difference is
reported on" -- and read as a cross-layer mean, because the trainer summed a
``(num_layers, d_hidden)`` dead mask into one scalar before recording it and the
published "live latents per layer" figures were ``d_hidden - n_dead/num_layers``.
A mean over 32 layers is not a statement about any layer, and the first test
below fails if the statistic is ever collapsed that way again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.transfer.basis_criteria import (
    INTERIOR_MARGIN,
    alignment_residuals,
    basis_gate,
    criterion_a,
    criterion_b1_descriptive,
    criterion_b2,
    interior_layers,
)
from src.transfer.transcoders import (
    FiringCensus,
    Transcoder,
    TranscoderConfig,
    live_latents_per_layer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DIFFING = REPO_ROOT / "results/transfer/external_baseline"
TEXT_ALIGNMENT = (
    DIFFING
    / "20260811023351_a322a416b8e9/s25_base_to_stage1_text"
    / "model_diffing__Llama-2-7b-hf__to__ProLLaMA_Stage_1__text__block_output.json"
)
PROTEIN_ALIGNMENT = (
    DIFFING
    / "20260812130041_d2fe9a1dca1d/s25_base_to_stage1_protein"
    / "model_diffing__Llama-2-7b-hf__to__ProLLaMA_Stage_1__protein__block_output.json"
)


# ------------------------------------------- the statistic must stay per layer


def test_a_dictionary_whose_mean_clears_the_cut_can_still_fail_it_at_a_layer() -> None:
    """The test that fails if B2 is ever collapsed back to a cross-layer mean.

    This is not a hypothetical shape. EXP-R2-191's `base/text` cell published
    7,608 live latents per layer against `d_model` 4,096 -- a comfortable PASS --
    while its own layer 1 kept 90. The mean reading and the per-layer reading give
    opposite answers on the same dictionary, so a gate read through the mean was
    never the gate that was pre-registered.
    """

    counts = [8192] * 32
    counts[1] = 90
    reading = criterion_b2(counts, d_model=4096)

    assert reading["mean_live_per_layer"] > 4096
    assert reading["mean_reading"] == "PASS"
    assert reading["verdict"] == "FAIL"
    assert reading["mean_reading_agrees"] is False
    assert reading["failing_layers"] == [1]
    assert reading["min_live_per_layer"] == 90
    assert reading["argmin_layer"] == 1


def test_the_mean_reading_is_reported_and_is_not_the_verdict() -> None:
    """Continuity with the published figures, without letting them decide."""

    reading = criterion_b2([4000] * 32, d_model=4096)
    assert reading["mean_reading"] == "FAIL"
    assert reading["verdict"] == "FAIL"
    assert reading["mean_reading_agrees"] is True
    assert reading["mean_live_per_layer"] == pytest.approx(4000.0)


def test_both_layer_windows_are_reported_and_the_verdict_is_the_wider_one() -> None:
    """A failure outside Criterion A's window still fails B2, and is located.

    A's interior window exists because the outermost layers carry degenerate
    adjacent-layer denominators. B2 has no denominator, so it is not narrowed by
    that window -- but a reader who restricts a diff to layers 2-29 has the
    interior reading beside the verdict rather than having to recompute it.
    """

    counts = [8192] * 32
    counts[0] = 10
    reading = criterion_b2(counts, d_model=4096)

    assert reading["per_layer_reading_all_layers"] == "FAIL"
    assert reading["per_layer_reading_interior"] == "PASS"
    assert reading["verdict"] == "FAIL"
    assert reading["failing_layers"] == [0]
    assert reading["interior_layers"] == [INTERIOR_MARGIN, 31 - INTERIOR_MARGIN]


@pytest.mark.parametrize(
    "counts, d_model, message",
    [
        ([], 4096, "one non-negative integer per layer"),
        ([100, -1, 100, 100, 100, 100], 4096, "cannot be negative"),
        ([100] * 6, 0, "d_model must be positive"),
    ],
)
def test_a_basis_reading_refuses_an_input_it_cannot_mean(counts, d_model, message) -> None:
    with pytest.raises(ValueError, match=message):
        criterion_b2(counts, d_model=d_model)


def test_a_model_too_shallow_for_an_interior_window_is_refused_not_narrowed() -> None:
    """A median over no layers is not a smaller version of this statistic."""

    with pytest.raises(ValueError, match="no interior window"):
        interior_layers(2 * INTERIOR_MARGIN)
    assert interior_layers(32) == list(range(2, 30))


def test_the_gate_fails_when_any_one_cell_fails() -> None:
    """'In all four cells' is what Criterion B was declared as."""

    passing = criterion_b2([8192] * 32, d_model=4096)
    failing = criterion_b2([1634] * 32, d_model=4096)
    gate = basis_gate(
        [
            {"cell": "base/text", "b2": passing},
            {"cell": "stage1/text", "b2": passing},
            {"cell": "base/protein", "b2": failing},
            {"cell": "stage1/protein", "b2": failing},
        ]
    )
    assert gate["verdict"] == "FAIL"
    assert gate["failing_cells"] == ["base/protein", "stage1/protein"]

    with pytest.raises(ValueError, match="no cells"):
        basis_gate([])


# ---------------------------------------------------- the live basis it reads


def test_the_dead_mask_reaches_the_record_per_layer_and_not_only_summed() -> None:
    """The trainer's objective must report the vector, not only its sum.

    Constructed so the layers genuinely differ: layer 0 is fed the same inputs
    the model has already seen fire, layer 1 is fed zeros so a different set of
    latents survives TopK. The assertion that matters is the last one -- the
    vector is not recoverable from the scalar, so recording only the scalar
    destroys the statistic the gate is read on.
    """

    config = TranscoderConfig(
        num_layers=3, d_model=8, d_hidden=16, k=2, auxk=2, dead_steps=0, cross_layer=False
    )
    # The GLOBAL generator, seeded, and not only the local one below. The data
    # was already seeded; the *model* was not, and `Transcoder.__init__` draws its
    # encoders and decoders from the global RNG. That left the fixture at the
    # mercy of whatever consumed global randomness earlier in the session, and the
    # final assertion -- that the three layers report different dead counts -- is
    # sensitive to it: replayed over 400 global states this test failed in 9.2% of
    # them, so it was a one-in-eleven false failure whose trigger was the order and
    # contents of the rest of the suite. What is asserted is unchanged.
    torch.manual_seed(20260814)
    model = Transcoder(config)
    generator = torch.Generator().manual_seed(11)
    inputs = torch.randn(3, 32, 8, generator=generator)
    targets = torch.randn(3, 32, 8, generator=generator)

    first = model.objective(inputs, targets, training=True)
    assert first["n_dead_per_layer"] is not None
    assert len(first["n_dead_per_layer"]) == 3
    assert sum(first["n_dead_per_layer"]) == first["n_dead"]

    second = model.objective(inputs, targets, training=True)
    per_layer = second["n_dead_per_layer"]
    assert sum(per_layer) == second["n_dead"]
    # dead_steps=0 means every latent that missed this batch is already dead, so
    # each layer reports d_hidden minus the latents that fired there.
    fired = (second["fired_per_latent"] > 0).sum(dim=1)
    assert per_layer == [config.d_hidden - int(value) for value in fired]
    assert len(set(per_layer)) > 1, "the fixture must exercise layers that differ"


def test_a_held_out_pass_reports_no_dead_count_rather_than_a_zero_one() -> None:
    """The counter is not advanced on an evaluation pass, so no count exists."""

    config = TranscoderConfig(
        num_layers=2, d_model=4, d_hidden=8, k=2, auxk=2, dead_steps=0, cross_layer=False
    )
    model = Transcoder(config)
    inputs = torch.randn(2, 16, 4)
    report = model.objective(inputs, torch.randn(2, 16, 4), training=False)

    assert report["n_dead_per_layer"] is None
    assert report["n_dead"] == 0
    assert int(model.silent_steps.sum()) == 0


def test_live_latents_are_counted_at_the_layer_they_belong_to() -> None:
    silent = torch.tensor([[0, 5, 100], [100, 100, 100], [0, 0, 0]], dtype=torch.long)
    assert live_latents_per_layer(silent, dead_steps=10) == [2, 0, 3]
    assert live_latents_per_layer(silent, dead_steps=1000) == [3, 3, 3]


@pytest.mark.parametrize(
    "buffer, dead_steps, message",
    [
        (torch.zeros(8, dtype=torch.long), 10, "must be \\(num_layers, d_hidden\\)"),
        (torch.zeros(2, 8, dtype=torch.long), -1, "cannot be negative"),
    ],
)
def test_a_live_basis_refuses_a_buffer_it_cannot_read(buffer, dead_steps, message) -> None:
    with pytest.raises(ValueError, match=message):
        live_latents_per_layer(buffer, dead_steps=dead_steps)


def test_the_firing_census_counts_rather_than_flags_and_is_monotone_in_its_cut() -> None:
    census = FiringCensus(2, 4)
    census.update(torch.tensor([[3, 0, 1, 0], [0, 0, 0, 0]]), n_tokens=10)
    census.update(torch.tensor([[4, 0, 0, 2], [1, 0, 0, 0]]), n_tokens=10)

    assert census.tokens == 20
    # counts are [7, 0, 1, 2] at layer 0 and [1, 0, 0, 0] at layer 1
    assert census.live_per_layer(1) == [3, 1]
    assert census.live_per_layer(2) == [2, 0]
    assert census.live_per_layer(3) == [1, 0]
    assert census.live_per_layer(8) == [0, 0]
    record = census.record((1, 10))
    assert record["live_per_layer"]["1"] == [3, 1]
    assert record["n_tokens"] == 20

    with pytest.raises(ValueError, match="fewer than once"):
        census.live_per_layer(0)
    with pytest.raises(ValueError, match="cannot take a batch"):
        census.update(torch.zeros(3, 4, dtype=torch.long), n_tokens=1)


def test_a_dictionary_can_be_live_by_its_counter_and_dead_on_a_cohort() -> None:
    """The two definitions answer different questions and must not be blended.

    A latent that last fired 100 training steps ago is live by the checkpoint's
    own counter at dead_steps 2500, and invisible to a held-out cohort it never
    fires on. Nothing may report one as an estimate of the other.
    """

    silent = torch.tensor([[0, 100, 3000, 3000]], dtype=torch.long)
    assert live_latents_per_layer(silent, dead_steps=2500) == [2]

    census = FiringCensus(1, 4)
    census.update(torch.tensor([[7, 0, 0, 0]]), n_tokens=500)
    assert census.live_per_layer(1) == [1]


# ------------------------------------------------- criterion A, on real artefacts


def test_criterion_a_refuses_a_denominator_that_is_not_a_cost() -> None:
    cross = [1.0] * 32
    adjacent = [1.0] * 32
    adjacent[10] = 0.0
    with pytest.raises(ValueError, match="not\\s+positive"):
        criterion_a(cross, adjacent)

    adjacent[10] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        criterion_a(cross, adjacent)


def test_criterion_a_separates_its_three_verdicts() -> None:
    """The UNRESOLVED band is a verdict withheld, not a pass."""

    assert criterion_a([3.0] * 32, [1.0] * 32)["verdict"] == "INSUFFICIENT"
    assert criterion_a([0.2] * 32, [1.0] * 32)["verdict"] == "SUFFICIENT"
    straddling = [0.9 if layer % 2 else 1.1 for layer in range(32)]
    reading = criterion_a(straddling, [1.0] * 32)
    assert reading["verdict"] == "UNRESOLVED"
    assert reading["iqr"][0] <= 1.0 <= reading["iqr"][1]


@pytest.mark.skipif(
    not PROTEIN_ALIGNMENT.is_file() or not TEXT_ALIGNMENT.is_file(),
    reason="the EXP-R2-175 diffing artefacts are host-local and are not in the repository",
)
def test_criterion_a_reproduces_the_published_reading_from_the_artefacts() -> None:
    """EXP-R2-191 published these four numbers per mode. They come from here now.

    Pinned because they were computed by hand: the audit's INSUFFICIENT verdict on
    the protein mode and its withheld verdict on the text mode now have one
    implementation, and this fails if the two part company.
    """

    protein = criterion_a(
        *alignment_residuals(json.loads(PROTEIN_ALIGNMENT.read_text(encoding="utf-8")))
    )
    assert protein["median"] == pytest.approx(3.116, abs=5e-4)
    assert protein["iqr"][0] == pytest.approx(1.999, abs=5e-4)
    assert protein["iqr"][1] == pytest.approx(4.264, abs=5e-4)
    assert protein["fraction_above_one"] == pytest.approx(0.821, abs=5e-4)
    assert protein["verdict"] == "INSUFFICIENT"

    text = criterion_a(
        *alignment_residuals(json.loads(TEXT_ALIGNMENT.read_text(encoding="utf-8")))
    )
    assert text["median"] == pytest.approx(0.956, abs=5e-4)
    assert text["iqr"][0] == pytest.approx(0.904, abs=5e-4)
    assert text["iqr"][1] == pytest.approx(1.026, abs=5e-4)
    assert text["fraction_above_one"] == pytest.approx(0.321, abs=5e-4)
    assert text["verdict"] == "UNRESOLVED"


# ---------------------------------------------------------- criterion B1 is void


def test_criterion_b1_never_returns_a_verdict_however_good_its_numbers_are() -> None:
    """A criterion its own control cannot pass is a specification defect."""

    for nmse in ([0.01] * 32, [10.0] * 32):
        reading = criterion_b1_descriptive(nmse, [1.0] * 32)
        assert reading["verdict"] == "VOID"
        assert "PASS" not in reading["verdict"] and "FAIL" not in reading["verdict"]
        assert "median" in reading and "max" in reading
