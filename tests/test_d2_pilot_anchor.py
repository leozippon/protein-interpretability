"""Conditions the anchor-restricted pilot analysis must always hold.

The stage's strongest validation is not here: it reproduces two admitted
common-support figures, the Llama-2-7B parent's raw likelihood endpoint at
-0.0096077001 and ProLLaMA Stage 1's at +0.1433810357, to 1.7e-18 and 2.8e-17
from the restricted records. What is tested here is the logic that validation
cannot exercise -- what the three bootstrap seeds are allowed to move, what they
are not, and the shape of a recovered fraction whose scale is the ceiling.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    'analyse_d2_pilot_anchor', ROOT / 'scripts/transfer/analyse_d2_pilot_anchor.py')
pilot = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pilot)


def rows(values, clusters=None):
    clusters = clusters or [f'fam{index // 2}' for index in range(len(values))]
    return [{'assay': f'A{index}', 'cluster': clusters[index], 'spearman': value}
            for index, value in enumerate(values)]


def test_the_bootstrap_seed_moves_the_interval_and_never_the_point():
    """A resampling seed is not a fitting seed: it cannot move the estimate.

    This is the property that makes the three seeds a stability check rather than
    three different measurements, and it is what the record's seed claim rests on.
    """
    record = pilot.endpoint_at_seeds(rows([0.1, 0.2, 0.15, 0.25, 0.3, 0.05,
                                           0.22, 0.18, 0.27, 0.12, 0.09, 0.31,
                                           0.14, 0.26, 0.19, 0.23]))
    assert record['point_identical_across_seeds'] is True
    assert record['point'] is not None
    points = {entry['point'] for entry in record['per_bootstrap_seed'].values()}
    assert len(points) == 1
    assert set(record['per_bootstrap_seed']) == {str(s) for s in pilot.BOOTSTRAP_SEEDS}
    # Three seeds, three intervals, and the spread is reported rather than hidden.
    lows = [entry['interval'][0] for entry in record['per_bootstrap_seed'].values()]
    assert record['interval_low_range'] == [min(lows), max(lows)]
    assert record['resolves_above_zero_at_every_seed'] is True


def test_resolution_requires_every_seed_not_the_best_one():
    """A quantity resolved at one seed and not another is not resolved."""
    straddling = pilot.endpoint_at_seeds(rows([0.02, -0.03, 0.05, -0.04, 0.01, -0.02,
                                               0.03, -0.05, 0.04, -0.01, 0.02, -0.03,
                                               0.01, -0.04, 0.05, -0.02]))
    assert straddling['resolves_above_zero_at_every_seed'] is False
    assert straddling['resolves_below_zero_at_every_seed'] is False
    low, high = straddling['interval_low_range'], straddling['interval_high_range']
    assert low[0] <= low[1] and high[0] <= high[1]


def test_the_ceiling_recovers_its_own_gap_exactly_and_the_floor_recovers_none():
    """The scale a recovered fraction is read against, asserted at both ends."""
    floor = rows([0.0] * 16)
    ceiling = rows([0.2] * 16)
    at_ceiling = pilot.contrast_at_seeds(ceiling, floor, ceiling)
    primary = at_ceiling['per_bootstrap_seed'][str(pilot.BOOTSTRAP_SEEDS[0])]
    assert primary['recovered_fraction']['share'] == pytest.approx(1.0)
    assert primary['minus_floor']['point'] == pytest.approx(0.2)
    at_floor = pilot.contrast_at_seeds(floor, floor, ceiling)
    assert at_floor['per_bootstrap_seed'][str(pilot.BOOTSTRAP_SEEDS[0])][
        'recovered_fraction']['share'] == pytest.approx(0.0)
    assert at_floor['minus_floor_resolves_above_zero_at_every_seed'] is False
    assert at_ceiling['families'] == 8 and at_ceiling['assays'] == 16


def test_a_partial_cell_recovers_a_fraction_between_the_ends():
    floor = rows([0.0] * 16)
    ceiling = rows([0.2] * 16)
    half = rows([0.1] * 16)
    record = pilot.contrast_at_seeds(half, floor, ceiling)
    share = record['per_bootstrap_seed'][str(pilot.BOOTSTRAP_SEEDS[0])][
        'recovered_fraction']['share']
    assert 0.0 < share < 1.0 and share == pytest.approx(0.5)


def test_the_contrast_uses_only_assays_every_cell_scored():
    floor = rows([0.0] * 16)
    ceiling = rows([0.2] * 16)
    short = rows([0.1] * 16)[:12]
    record = pilot.contrast_at_seeds(short, floor, ceiling)
    assert record['assays'] == 12
    # Six families is below the shared eight-unit floor, so the interval is
    # withheld and the contrast resolves nothing rather than crashing.
    assert record['families'] == 6
    assert record['degenerate_at_some_seed'] is True
    assert record['minus_floor_resolves_above_zero_at_every_seed'] is False
    assert record['minus_floor_interval_low_range'] is None
    assert 'below the 8-' in record['degenerate_reason']


def test_the_control_vocabulary_names_the_null_exemption():
    """The byte-identity refusal is not weakened; the null cell is declared a control."""
    assert set(pilot.CONTROL_CELLS) == {'floor', 'ceiling', 'null', 'scrambled'}
    assert 'algebraic no-op' in pilot.CONTROL_CELLS['null']
    assert 'bit-identically' in pilot.CONTROL_CELLS['null']
    assert 'measured magnitude' in pilot.CONTROL_CELLS['scrambled']
    assert pilot.BOOTSTRAP_SEEDS[0] == 20260923 and len(pilot.BOOTSTRAP_SEEDS) == 3
    assert pilot.BOOTSTRAP_DRAWS == 2000
