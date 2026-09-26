"""The three integration conditions, and the order they are read in.

Tested on constructed analysis records rather than on the pilot's own, because the
property that matters is what the stage does when a condition fails: it must stop
there and not report the later ones as though they had been assessed.
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
    'assess_d2_pilot_conditions', ROOT / 'scripts/transfer/assess_d2_pilot_conditions.py')
assess = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(assess)

SEEDS = [20260923, 20260924, 20260925]


def cell(kind, *, point=0.05, share=0.3, resolved=True, seed_stable=True, lows=None,
         control='none'):
    lows = lows if lows is not None else ([0.01, 0.012, 0.011] if resolved
                                          else [-0.01, 0.002, -0.003])
    return {
        'kind': kind,
        'control': control,
        'endpoint': {'point_identical_across_seeds': seed_stable,
                     'point': point if seed_stable else None},
        'contrast': {
            'minus_floor_resolves_above_zero_at_every_seed': resolved,
            'minus_floor_interval_low_range': [min(lows), max(lows)],
            'minus_floor_interval_high_range': [0.09, 0.1],
            'per_bootstrap_seed': {str(seed): {
                'minus_floor': {'point': point, 'interval': [min(lows), 0.1]},
                'recovered_fraction': {'share': share}} for seed in SEEDS},
        },
    }


def analysis(cells):
    return {'arm': 'full_anchor', 'bootstrap': {'seeds': SEEDS},
            'support': {'panel': 'anchor201'},
            'split_seed_stability': 'undefined for this endpoint',
            'gate': {'ends_reproduce_their_admitted_endpoints': True,
                     'null_is_bit_identical_to_floor': True},
            'cells': cells}


def test_seed_stability_is_met_when_every_cell_is_stable_at_every_seed():
    record = assess.condition_one(analysis({
        'attention_q1': cell('treatment'), 'mlp_q1': cell('treatment')}))
    assert record['met'] is True
    assert record['unstable_cells'] == []
    assert 'undefined' in record['axis']


def test_a_point_that_moves_with_the_resampling_seed_fails_condition_one():
    record = assess.condition_one(analysis({
        'attention_q1': cell('treatment', seed_stable=False)}))
    assert record['met'] is False
    assert record['unstable_cells'] == ['attention_q1']


def test_a_resolution_that_flips_between_seeds_fails_condition_one():
    straddling = cell('treatment', resolved=False, lows=[-0.004, 0.003, 0.001])
    record = assess.condition_one(analysis({'attention_q1': straddling}))
    assert record['met'] is False


def test_condition_two_needs_the_null_because_that_is_the_uncertainty_term():
    record = assess.condition_two(analysis({'attention_q1': cell('treatment')}))
    assert record['met'] is None
    assert 'replay uncertainty has no measured value' in record['not_assessable']


def test_replay_uncertainty_is_the_null_and_the_scramble_is_not_a_bar():
    """A treatment smaller than the scramble's excursion must still pass."""
    cells = {'attention_q1': cell('treatment', point=0.05),
             'null_attention_q1': cell('control', point=0.0, share=0.0, resolved=False,
                                       control='null'),
             'scrambled_attention_q1': cell('control', point=-0.30, share=-2.0,
                                            resolved=False, control='scrambled')}
    record = assess.condition_two(analysis(cells))
    # 0.05 is far smaller in absolute value than the scramble's 0.30, and the
    # condition is met anyway: the scramble is not the term being exceeded.
    assert record['met'] is True
    assert record['replay_uncertainty']['is_exactly_zero'] is True
    assert 'by construction rather than by comparison' in record['replay_uncertainty']['reading']
    assert record['specificity_control']['is_not_an_uncertainty_bar'] is True
    assert 'not disqualified' in record['specificity_control']['reading']
    assert 'not on any comparison against the scramble' in record['met_turns_on']
    row = record['per_cell']['attention_q1']
    assert row['above_replay_uncertainty'] is True
    assert row['direction_differs_from_the_scramble'] is True
    assert row['magnitude_relative_to_the_scramble'] == pytest.approx(0.05 / 0.30)


def test_the_scrambles_asymmetry_is_reported_as_the_specificity_finding():
    cells = {'attention_q1': cell('treatment', point=0.05),
             'mlp_q1': cell('treatment', point=0.02),
             'null_attention_q1': cell('control', point=0.0, resolved=False, control='null'),
             'scrambled_attention_q1': cell('control', point=-0.30, resolved=False,
                                            control='scrambled')}
    record = assess.condition_two(analysis(cells))
    asymmetry = record['specificity_control']['asymmetry_is_the_finding']
    assert asymmetry['scramble_direction'] == 'down'
    assert asymmetry['treatments_opposing_it'] == ['attention_q1', 'mlp_q1']


def test_a_null_control_that_is_not_zero_fails_condition_two():
    cells = {'attention_q1': cell('treatment'),
             'null_attention_q1': cell('control', point=0.004, resolved=False,
                                       control='null'),
             'scrambled_attention_q1': cell('control', point=-0.08, resolved=False,
                                            control='scrambled')}
    assert assess.condition_two(analysis(cells))['met'] is False


def test_condition_three_reports_the_profile_and_declines_the_verdict():
    cells = {'attention_q1': cell('treatment', share=0.62),
             'attention_q2': cell('treatment', share=0.08),
             'mlp_q1': cell('treatment', share=0.05),
             'embedding': cell('treatment', share=0.02)}
    record = assess.condition_three(analysis(cells))
    assert record['met'] is None
    assert 'judgement' in record['verdict_is_not_mechanical']
    assert record['by_component']['attention']['cells'] == 2
    assert record['by_component']['attention']['max_share'] == pytest.approx(0.62)
    assert record['largest_against_runner_up']['largest']['cell'] == 'attention_q1'
    assert record['largest_against_runner_up']['ratio'] == pytest.approx(0.62 / 0.08)
    assert set(record['by_quarter']) == {'attention_q1', 'attention_q2', 'mlp_q1'}


def test_controls_are_not_counted_as_treatments_anywhere():
    cells = {'attention_q1': cell('treatment'), 'pfloor': cell('control'),
             'pceiling': cell('control')}
    assert set(assess.treatments(analysis(cells))) == {'attention_q1'}
    assert set(assess.condition_three(analysis(cells))['per_cell']) == {'attention_q1'}
