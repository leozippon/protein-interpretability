"""Tests for the within-unit label-entropy measurement of the explanation channel.

The load-bearing property is that the window's *position* is measured rather than
assumed. Every headline this measurement produces is a ratio between a text
channel and a protein channel read off one window, and at offset zero that window
is the least representative position either modality has: a protein's N-terminus
carries its signal or transit peptide, its disordered tail and -- by construction
-- no Pfam annotation, while a document's carries its title and lede. So the tests
here are about the offset: that it moves the window, that the sensitivity is
paired on units long enough for both windows, and that a unit set which cannot
supply the second window says so instead of leaving stationarity assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.transfer.channels import UNANNOTATED, within_unit_label_entropy  # noqa: E402

WINDOW = 50
MIN_UNITS = 5


def _n_terminal_unit(*, tail_labels: int) -> list[str]:
    """A protein-shaped unit: an unannotated first window, domains in the second."""

    head = [UNANNOTATED] * WINDOW
    tail = [f"PF{position % tail_labels:05d}" for position in range(WINDOW)]
    return head + tail


def _short_unit() -> list[str]:
    """Long enough for the first window, too short for the second."""

    return [f"PF{position % 5:05d}" for position in range(WINDOW)]


def test_the_window_moves_with_the_offset_and_the_sensitivity_is_paired() -> None:
    """The N-terminal case, which is the one the protein channels are read on.

    Every long unit is constant over its first window and carries several domains
    over its second. At offset zero those units read zero bits and the matched
    permutation null is degenerate on each of them -- a test built on it has no
    power at all -- while one window further along the same proteins the same
    channel carries real entropy. Reporting only the first would attribute the
    whole difference to the modality.

    The three short units are the reason the sensitivity is paired: they reach the
    first window and not the second, so counting them on one side and not the other
    would confound the offset with unit length.
    """

    units = [_n_terminal_unit(tail_labels=2 + index % 4) for index in range(12)]
    units += [_short_unit() for _ in range(3)]

    record = within_unit_label_entropy(units, window=WINDOW, min_units=MIN_UNITS)

    assert record["window_symbols"] == WINDOW
    assert record["window_offset"] == 0
    assert record["n_units"] == 15
    assert record["n_units_considered"] == 15
    assert record["permutation_null_degenerate_fraction"] == pytest.approx(12 / 15)

    sensitivity = record["window_offset_sensitivity"]
    assert sensitivity["measured"] is True
    assert sensitivity["unmeasured_reason"] is None
    assert sensitivity["offsets"] == [0, WINDOW]
    assert sensitivity["n_units_paired"] == 12
    assert sensitivity["at_offset"]["entropy_bits"]["mean"] == pytest.approx(0.0)
    assert sensitivity["at_offset"]["permutation_null_degenerate_fraction"] == 1.0
    assert sensitivity["at_next_offset"]["entropy_bits"]["mean"] > 1.0
    assert sensitivity["at_next_offset"]["permutation_null_degenerate_fraction"] == 0.0
    assert sensitivity["entropy_bits_difference"]["mean"] == pytest.approx(
        sensitivity["at_next_offset"]["entropy_bits"]["mean"]
        - sensitivity["at_offset"]["entropy_bits"]["mean"]
    )
    assert sensitivity["entropy_bits_difference"]["interval"][0] > 0.0
    # The headline the artefact leads with is the one measured at offset zero, and
    # on these units it understates the channel.
    assert (
        record["entropy_bits"]["mean"]
        < sensitivity["at_next_offset"]["entropy_bits"]["mean"]
    )

    # Measuring the second window directly must give the paired block back, or the
    # sensitivity would be reporting something other than what `offset=` measures.
    moved = within_unit_label_entropy(
        units, window=WINDOW, offset=WINDOW, min_units=MIN_UNITS
    )
    assert moved["window_offset"] == WINDOW
    assert moved["n_units"] == 12
    assert moved["entropy_bits"]["mean"] == pytest.approx(
        sensitivity["at_next_offset"]["entropy_bits"]["mean"]
    )


def test_the_sensitivity_says_so_when_no_unit_supplies_a_second_window() -> None:
    """Unmeasured is recorded as unmeasured, not as agreement between offsets."""

    record = within_unit_label_entropy(
        [_short_unit() for _ in range(9)], window=WINDOW, min_units=MIN_UNITS
    )
    sensitivity = record["window_offset_sensitivity"]

    assert sensitivity["measured"] is False
    assert sensitivity["n_units_paired"] == 0
    assert "unmeasured here, not absent" in sensitivity["unmeasured_reason"]
    assert sensitivity["at_offset"] is None
    assert sensitivity["at_next_offset"] is None
    assert sensitivity["entropy_bits_difference"] is None


def test_an_offset_the_units_cannot_reach_refuses_and_names_it() -> None:
    units = [_short_unit() for _ in range(9)]

    with pytest.raises(RuntimeError, match=f"offset {WINDOW}"):
        within_unit_label_entropy(
            units, window=WINDOW, offset=WINDOW, min_units=MIN_UNITS
        )
    with pytest.raises(ValueError, match="offset"):
        within_unit_label_entropy(units, window=WINDOW, offset=-1, min_units=MIN_UNITS)
