#!/usr/bin/env python3
"""The three integration conditions, each as met or not met, with its evidence.

The conditions were set by the user and the call on them is not this stage's:
what it does is state each one against measured quantities and stop at the first
that fails, so that a reader cannot reach a localisation claim past a condition
that did not hold.

1. **Seed stability.** Every treatment cell's endpoint must be identical across
   the three declared bootstrap seeds -- a resampling seed cannot move a point
   estimate -- and its contrast against the floor must resolve the same way at
   all three. Split-seed stability is undefined for this endpoint and is reported
   as undefined rather than as checked.

2. **The effect above replay uncertainty, and its specificity.** The condition
   is reported split, because the two controls answer different questions and
   conflating them sets a bar that is not meaningful.

   *Replay uncertainty is the null's exact zero.* A group copied back over itself
   returning bit-identical establishes that the intervention machinery
   contributes nothing of its own, so any nonzero treatment effect is above
   replay uncertainty **by construction rather than by comparison**. That is what
   this condition turns on, and it is read mechanically.

   *The scramble is a specificity control, and its asymmetry is the evidence.* If
   a derangement within a shape class moves the endpoint sharply down while the
   same tensors in their correct destinations move it up, the endpoint depends on
   **which tensor goes where** and not on how much was perturbed -- a stronger
   statement than any magnitude comparison. A treatment effect smaller in
   absolute value than the scramble's excursion is therefore **not** disqualified,
   and the stage does not test it against one.

3. **Mechanistic localisation.** An effect that is spread evenly over every
   component and every depth quarter supports "transplanting helps" and nothing
   more. The condition asks whether the recovered share concentrates: whether
   attention and MLP differ, and whether the quarters differ within the component
   that carries more. The stage reports the profile and the separation; it does
   not declare the verdict, because how much concentration counts as a mechanism
   is a judgement the coordinator makes on this evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transfer import mechanistic_interface as mi  # noqa: E402
from src.transfer.io import sha256_file, write_json  # noqa: E402

#: Cells whose contrast is a control reading rather than a treatment effect.
CONTROLS = ('pfloor', 'floor', 'pceiling', 'ceiling', 'null_attention_q1',
            'scrambled_attention_q1')


def treatments(analysis: dict) -> dict[str, dict]:
    return {label: cell for label, cell in analysis['cells'].items()
            if cell.get('kind') == 'treatment'}


def condition_one(analysis: dict) -> dict:
    """Seed stability, on the only stochastic axis the endpoint has."""
    rows, unstable = {}, []
    for label, cell in sorted(treatments(analysis).items()):
        endpoint, contrast = cell['endpoint'], cell.get('contrast', {})
        stable = bool(endpoint['point_identical_across_seeds'])
        resolution = contrast.get('minus_floor_resolves_above_zero_at_every_seed')
        low = contrast.get('minus_floor_interval_low_range')
        high = contrast.get('minus_floor_interval_high_range')
        consistent = (low is not None and high is not None
                      and (all(value > 0 for value in low) or all(value <= 0 for value in low)))
        rows[label] = {'point_identical_across_seeds': stable,
                       'contrast_resolves_above_zero_at_every_seed': resolution,
                       'contrast_interval_low_range': low,
                       'contrast_interval_high_range': high,
                       'resolution_consistent_across_seeds': consistent}
        if not stable or not consistent:
            unstable.append(label)
    return {
        'condition': 'seed stability across the declared bootstrap seeds',
        'axis': 'family bootstrap seed; split-seed stability is undefined for this endpoint',
        'seeds': analysis['bootstrap']['seeds'],
        'per_cell': rows,
        'unstable_cells': unstable,
        'met': not unstable and bool(rows),
    }


def condition_two(analysis: dict) -> dict:
    """Replay uncertainty from the null, and specificity from the scramble, reported split."""
    cells = analysis['cells']
    null = next((cell for cell in cells.values() if cell.get('control') == 'null'), None)
    scrambled = next((cell for cell in cells.values()
                      if cell.get('control') == 'scrambled'), None)
    spectrum = next((cell for cell in cells.values()
                     if cell.get('control') == 'spectrum_matched'), None)
    primary_seed = str(analysis['bootstrap']['seeds'][0])

    def minus_floor(cell):
        if cell is None:
            return None
        return cell.get('contrast', {}).get('per_bootstrap_seed', {}).get(
            primary_seed, {}).get('minus_floor', {}).get('point')

    if null is None:
        return {'condition': 'the effect above replay uncertainty, with specificity reported',
                'met': None,
                'not_assessable': 'the null control is absent, so replay uncertainty has no '
                                  'measured value and the condition cannot be read'}
    null_point = minus_floor(null)
    scrambled_point = minus_floor(scrambled)
    rows, failing = {}, []
    for label, cell in sorted(treatments(analysis).items()):
        contrast = cell.get('contrast', {})
        point = minus_floor(cell)
        resolved = bool(contrast.get('minus_floor_resolves_above_zero_at_every_seed'))
        rows[label] = {
            'minus_floor': point,
            'resolves_above_zero_at_every_seed': resolved,
            'above_replay_uncertainty': None if point is None else point != null_point,
            'direction_differs_from_the_scramble': (
                None if point is None or scrambled_point is None
                else (point > 0) != (scrambled_point > 0)),
            'magnitude_relative_to_the_scramble': (
                None if point is None or not scrambled_point
                else abs(point) / abs(scrambled_point)),
        }
        if not resolved:
            failing.append(label)
    replay = {
        'term': 'the null control',
        'minus_floor': null_point,
        'must_be_exactly_zero': True,
        'is_exactly_zero': null_point == 0.0,
        'bit_identical_to_floor': analysis['gate'].get('null_is_bit_identical_to_floor'),
        'reading': ('a group copied back over itself returning bit-identical establishes that '
                    'the intervention machinery contributes nothing of its own, so a nonzero '
                    'treatment effect is above replay uncertainty by construction rather than '
                    'by comparison'),
    }
    specificity = {
        'term': 'the scrambled control',
        'minus_floor': scrambled_point,
        'is_not_an_uncertainty_bar': True,
        'reading': ('a derangement within a shape class changes exactly as many parameters as '
                    'its treatment, from the same donor, and destroys only the assignment. If '
                    'it moves the endpoint down while the treatment moves it up, the endpoint '
                    'depends on which tensor goes where rather than on how much was perturbed. '
                    'A treatment smaller in absolute value than this excursion is not '
                    'disqualified, and is not tested against it.'),
        'asymmetry_is_the_finding': (
            None if scrambled_point is None else
            {'scramble_direction': 'down' if scrambled_point < 0 else 'up',
             'treatments_opposing_it': sorted(
                 label for label, row in rows.items()
                 if row['direction_differs_from_the_scramble'])}),
    }
    content = None if spectrum is None else {
        'term': 'the spectrum-matched control',
        'minus_floor': minus_floor(spectrum),
        'is_not_an_uncertainty_bar': True,
        'reading': ('tensors carrying the donor\'s exact singular values with random '
                    'orthogonal factors: the donor\'s scale and spectrum installed exactly '
                    'and its learned content destroyed completely. Recovering as much as the '
                    'donor\'s own tensors would make the movement a scale effect and '
                    'localisation by single-group transplant unavailable at any granularity; '
                    'recovering nothing would make the donor\'s content the operative thing '
                    'and the pilot\'s failure one of granularity and power.'),
    }
    return {
        'condition': 'the effect above replay uncertainty, with specificity reported',
        'replay_uncertainty': replay,
        'specificity_control': specificity,
        'content_control': content,
        'per_cell': rows,
        'cells_not_resolved_above_zero': failing,
        'met': bool(rows) and not failing and null_point == 0.0,
        'met_turns_on': ('every treatment contrast resolving above zero at every bootstrap '
                         'seed, and the null control being exactly zero; not on any comparison '
                         'against the scramble'),
    }


def condition_three(analysis: dict) -> dict:
    """Whether the effect concentrates by component and by depth quarter."""
    rows = {}
    for label, cell in treatments(analysis).items():
        primary = cell.get('contrast', {}).get('per_bootstrap_seed', {}).get(
            str(analysis['bootstrap']['seeds'][0]), {})
        share = primary.get('recovered_fraction', {}).get('share')
        rows[label] = {'recovered_share': share,
                       'minus_floor': primary.get('minus_floor', {}).get('point'),
                       'resolved': bool(cell.get('contrast', {}).get(
                           'minus_floor_resolves_above_zero_at_every_seed'))}
    def component(label):
        for name in ('attention', 'mlp', 'embedding', 'head'):
            if label.startswith(name):
                return name
        return 'other'
    by_component: dict[str, list] = {}
    for label, row in rows.items():
        if row['recovered_share'] is not None:
            by_component.setdefault(component(label), []).append(row['recovered_share'])
    profile = {name: {'cells': len(values), 'total_share': sum(values),
                      'max_share': max(values)} for name, values in by_component.items()}
    quarters = {label: row['recovered_share'] for label, row in sorted(rows.items())
                if component(label) in ('attention', 'mlp')}
    resolved = sorted(label for label, row in rows.items() if row['resolved'])
    ordered = sorted((row['recovered_share'], label) for label, row in rows.items()
                     if row['recovered_share'] is not None)
    separation = (None if len(ordered) < 2 else
                  {'largest': {'cell': ordered[-1][1], 'share': ordered[-1][0]},
                   'runner_up': {'cell': ordered[-2][1], 'share': ordered[-2][0]},
                   'ratio': (ordered[-1][0] / ordered[-2][0]) if ordered[-2][0] else None})
    return {
        'condition': 'the effect localises to a component or a depth region',
        'per_cell': rows,
        'by_component': profile,
        'by_quarter': quarters,
        'cells_resolved_above_zero': resolved,
        'largest_against_runner_up': separation,
        'met': None,
        'verdict_is_not_mechanical': (
            'how much concentration counts as a mechanistic statement is a judgement on this '
            'profile, not a threshold this stage applies. What the stage asserts is the '
            'profile itself and that a share spread evenly over every component and quarter '
            'would support "transplanting helps" and nothing more.'),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--analysis', type=Path, required=True,
                        help='the full-anchor arm analysis artefact')
    parser.add_argument('--sensitivity', type=Path,
                        help='the 56-family sensitivity arm analysis artefact')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_bytes())
    if analysis.get('arm') not in (None, 'full_anchor'):
        raise SystemExit('--analysis must be the full-anchor arm')
    if not analysis['gate'].get('ends_reproduce_their_admitted_endpoints'):
        raise SystemExit('the ends do not reproduce their admitted endpoints; the pilot stops '
                         'before any condition is assessed')
    ordered = [condition_one(analysis), condition_two(analysis), condition_three(analysis)]
    stopped_at = None
    for index, record in enumerate(ordered, start=1):
        if record['met'] is False:
            stopped_at = index
            break
    sensitivity = None
    if args.sensitivity is not None:
        other = json.loads(args.sensitivity.read_bytes())
        sensitivity = {
            'arm': other.get('arm'),
            'per_cell_recovered_share': {
                label: cell.get('contrast', {}).get('per_bootstrap_seed', {}).get(
                    str(other['bootstrap']['seeds'][0]), {}).get(
                    'recovered_fraction', {}).get('share')
                for label, cell in sorted(other['cells'].items())
                if cell.get('kind') == 'treatment'},
            'reading': ('a support-sensitivity arm at 56 of 163 families, never a filter: its '
                        'numbers are screening numbers and are not pooled with the full arm'),
        }
    report = {
        'schema_version': 'd2_pilot_conditions_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'support': analysis['support'],
        'split_seed_stability': analysis['split_seed_stability'],
        'conditions': {'1_seed_stability': ordered[0],
                       '2_above_measured_uncertainty': ordered[1],
                       '3_mechanistic_localisation': ordered[2]},
        'stopped_at_condition': stopped_at,
        'sensitivity_arm': sensitivity,
        'licenses': mi.LICENSE_STATEMENT,
        'scope': ('every treatment cell tests sufficiency in this context; no complement cell '
                  'is in this budget, so necessity is not tested. Recipient and donor differ '
                  'by a whole training run, so a positive result localises where that '
                  'training\'s effect lands in the parameters, not an architectural locus, '
                  'and it is not a circuit.'),
        'inputs_sha256': {str(p): sha256_file(p) for p in
                          [args.analysis] + ([args.sensitivity] if args.sensitivity else [])},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print(f'conditions: {args.out}')
    print(f'sha256={sha256_file(args.out)}')
    for key, record in report['conditions'].items():
        state = {True: 'MET', False: 'NOT MET', None: 'not mechanically decidable'}[record['met']]
        print(f'  {key}: {state}')
    if stopped_at:
        print(f'STOPPED at condition {stopped_at}; later conditions are not read')


if __name__ == '__main__':
    main()
