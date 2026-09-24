#!/usr/bin/env python3
"""Collect completed depth-sweep cell reports into one panel summary.

It copies numbers out of the cell reports and counts interval directions over
them. It fits nothing, resamples nothing and loads no model: a count of resolved
intervals here is a count, not a corrected significance statement and not a
prevalence in any model population.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.transfer import readout_depth as rd

AXES = ('depth', 'pos', 'wide', 'full')


def digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def direction(record) -> str:
    interval = record.get('interval')
    if interval is None:
        return 'degenerate'
    low, high = interval
    if low > 0:
        return 'resolved_positive'
    if high < 0:
        return 'resolved_negative'
    return 'unresolved'


def axis_profile(summaries: dict, axis: str, depth: int) -> list[dict]:
    """One axis's increment over the matched baseline, depth by depth."""
    profile = []
    for index in range(depth):
        key = f'{axis}{index:03d}_delta_spearman'
        if key not in summaries:
            continue
        record = summaries[key]
        representation = summaries.get(f'{axis}{index:03d}/R_minus_raw_M_spearman', {})
        profile.append(dict(
            depth=index, relative_depth=round(index / max(depth - 1, 1), 4),
            delta_spearman=record['point'], interval=record.get('interval'),
            direction=direction(record),
            delta_rank_mse=summaries[f'{axis}{index:03d}_delta_rank_mse']['point'],
            delta_rank_mse_interval=summaries[f'{axis}{index:03d}_delta_rank_mse'].get('interval'),
            r_spearman=summaries[f'{axis}{index:03d}/R_spearman']['point'],
            b_r_spearman=summaries[f'{axis}{index:03d}/B_R_spearman']['point'],
            r_minus_raw_m=representation.get('point'),
            r_minus_raw_m_interval=representation.get('interval')))
    return profile


def collect(paths: list[Path]) -> dict:
    cells = []
    for path in sorted(paths):
        report = json.loads(path.read_text())
        if report.get('schema_version') != rd.ANALYSIS_SCHEMA or report['status'] != 'complete':
            raise ValueError(f'{path} is not a complete depth-sweep cell report')
        depth = report['extraction']['block_count']
        middle, final = rd.admitted_block_indices(depth)
        cell = dict(path=str(path), sha256=digest(path), arm=report['arm'],
                    panel=report['panel'], stratum=report['stratum'], depth=depth,
                    hidden_width=report['extraction']['hidden_width'],
                    position_resolved=report['extraction']['position_resolved'],
                    admitted_block_indices=[middle, final],
                    support=report['support'],
                    extraction_agreement=report['extraction_agreement'],
                    prefix_control=report['prefix_control'], seeds={})
        for seed, payload in report['seeds'].items():
            summaries = payload['summaries']
            entry = dict(
                reproduced_admitted_delta_spearman=summaries['admitted_delta_spearman']['point'],
                reproduced_admitted_interval=summaries['admitted_delta_spearman'].get('interval'),
                reproduced_admitted_direction=direction(summaries['admitted_delta_spearman']),
                raw_M_spearman=summaries['raw_M_spearman']['point'],
                raw_P_spearman=summaries['raw_P_spearman']['point'],
                baseline_spearman=summaries['baseline/B_spearman']['point'],
                baseline_agreement=payload['baseline_agreement'],
                axes={})
            for axis in AXES:
                profile = axis_profile(summaries, axis, depth)
                if not profile:
                    continue
                resolved = [row for row in profile if row['direction'] == 'resolved_positive']
                best = max(profile, key=lambda row: (row['delta_spearman'] is not None,
                                                     row['delta_spearman']
                                                     if row['delta_spearman'] is not None
                                                     else float('-inf')))
                entry['axes'][axis] = dict(
                    profile=profile, depths=len(profile),
                    resolved_positive=len(resolved),
                    resolved_negative=sum(1 for row in profile
                                          if row['direction'] == 'resolved_negative'),
                    unresolved=sum(1 for row in profile if row['direction'] == 'unresolved'),
                    resolved_positive_depths=[row['depth'] for row in resolved],
                    max_delta_spearman=best['delta_spearman'],
                    max_delta_spearman_depth=best['depth'],
                    max_delta_spearman_interval=best['interval'],
                    max_delta_spearman_direction=best['direction'])
            for extra in ('union',):
                key = f'{extra}_delta_spearman'
                if key in summaries:
                    entry[extra] = dict(delta_spearman=summaries[key]['point'],
                                        interval=summaries[key].get('interval'),
                                        direction=direction(summaries[key]),
                                        r_spearman=summaries[f'{extra}/R_spearman']['point'],
                                        b_r_spearman=summaries[f'{extra}/B_R_spearman']['point'])
            for label in ('permuted/B_spearman', 'permuted/admitted_B_R_spearman'):
                if label in summaries:
                    entry.setdefault('label_shuffle', {})[label] = dict(
                        point=summaries[label]['point'], interval=summaries[label].get('interval'),
                        direction=direction(summaries[label]))
            cell['seeds'][seed] = entry
        cells.append(cell)
    return cells


def panel_counts(cells: list[dict]) -> dict:
    """Interval-direction counts per stratum and per axis, never pooling strata."""
    counts: dict[str, dict] = {}
    for cell in cells:
        for seed, entry in cell['seeds'].items():
            for axis, payload in entry['axes'].items():
                key = f"{cell['panel']}/{cell['stratum']}/{axis}"
                bucket = counts.setdefault(key, dict(
                    panel=cell['panel'], stratum=cell['stratum'], axis=axis, arms=set(),
                    arm_seed_cells=0, depth_cells=0, resolved_positive=0, resolved_negative=0,
                    unresolved=0, arms_with_any_resolved_positive=set(),
                    max_delta_spearman=None, max_delta_spearman_arm=None,
                    max_delta_spearman_depth=None))
                bucket['arms'].add(cell['arm'])
                bucket['arm_seed_cells'] += 1
                bucket['depth_cells'] += payload['depths']
                for name in ('resolved_positive', 'resolved_negative', 'unresolved'):
                    bucket[name] += payload[name]
                if payload['resolved_positive']:
                    bucket['arms_with_any_resolved_positive'].add(cell['arm'])
                point = payload['max_delta_spearman']
                if point is not None and (bucket['max_delta_spearman'] is None
                                          or point > bucket['max_delta_spearman']):
                    bucket.update(max_delta_spearman=point, max_delta_spearman_arm=cell['arm'],
                                  max_delta_spearman_depth=payload['max_delta_spearman_depth'],
                                  max_delta_spearman_seed=seed)
    for bucket in counts.values():
        bucket['arms'] = sorted(bucket['arms'])
        bucket['arms_with_any_resolved_positive'] = sorted(bucket['arms_with_any_resolved_positive'])
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', action='append', type=Path, required=True,
                        help='One completed depth-sweep cell report; repeat as needed')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--device', default='cpu', choices=['cpu'])
    args = parser.parse_args()
    cells = collect(args.report)
    reproduction = [dict(arm=cell['arm'], panel=cell['panel'], seed=seed,
                         extraction_max_relative_l2=cell['extraction_agreement']['max_relative_l2'],
                         extraction_exactly_equal=cell['extraction_agreement']['exactly_equal'],
                         likelihood_difference_nats=cell['extraction_agreement'][
                             'max_absolute_likelihood_difference_nats'],
                         worst_admitted_prediction_deviation=max(
                             value for key, value in
                             entry['baseline_agreement']['max_absolute_deviation'].items()
                             if key.endswith('_prediction')),
                         pipeline_identity=entry['baseline_agreement']
                         .get('pipeline_identity', {}).get('worst_prediction_deviation'))
                    for cell in cells for seed, entry in cell['seeds'].items()]
    payload = dict(schema_version='d1_readout_depth_panel_v1', status='complete',
                   created_utc=datetime.now(timezone.utc).isoformat(),
                   n_cells=len(cells), arms=sorted({cell['arm'] for cell in cells}),
                   panels=sorted({cell['panel'] for cell in cells}),
                   strata=sorted({cell['stratum'] for cell in cells}),
                   reproduction=reproduction, counts=panel_counts(cells), cells=cells,
                   summariser_sha256=digest(__file__))
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / 'readout_depth_panel.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + '\n')
    temporary.replace(destination)
    print(json.dumps(dict(cells=len(cells), arms=len(payload['arms']),
                          counts={key: {name: bucket[name] for name in
                                        ('arm_seed_cells', 'depth_cells', 'resolved_positive',
                                         'resolved_negative', 'unresolved', 'max_delta_spearman',
                                         'max_delta_spearman_arm', 'max_delta_spearman_depth')}
                                  for key, bucket in payload['counts'].items()}), indent=1))


if __name__ == '__main__':
    main()
