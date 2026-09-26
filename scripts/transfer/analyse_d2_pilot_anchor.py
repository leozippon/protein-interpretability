#!/usr/bin/env python3
"""Anchor-restricted endpoints for the ProLLaMA transplant pilot, at three bootstrap seeds.

Why this stage exists beside the admitted transplant analysis. The admitted
analysis reports the endpoint on the lineage's own 211-assay native support,
which does not compose with a Direction-1 capability result read on the 201-assay
anchor. Containment has been established exactly -- every anchor assay is in the
native support, and restricting a cell's per-variant records reproduces the
anchor panel's 201 assays, 163 wild-type families and 25,728 variants with every
mutation digest and per-assay count matching -- so this stage restricts rather
than re-extracts, and no forward pass is needed to move a cell onto the anchor.

**Split-seed stability is undefined for this endpoint and is not checked here.**
The native likelihood enters as one scalar column: there is no fitted readout, so
no folds, no penalty grid, no projection and no split seed exist. Asserting three
split seeds would be asserting something the endpoint has no axis for. The two
axes it does have are reported instead, and both are declared: the **family
bootstrap seed**, resampled at three declared values, and the **support**, read
at the full anchor and at the declared label-blind 56-family screen panel as a
sensitivity arm rather than as a filter.

Every refusal of :mod:`src.transfer.mechanistic_interface` stays in the path. A
cell that cannot pass one is reported as refused; none is weakened to make a cell
run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transfer import mechanistic_interface as mi  # noqa: E402
from src.transfer.capability_transplant import resolve_groups, support_digest  # noqa: E402
from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.profile_increment import (  # noqa: E402
    correlation, standardized_rank, summarize)
from src.transfer.profiles import share_bootstrap  # noqa: E402

#: The three bootstrap seeds. The first is the admitted one, so a cell's primary
#: reading is comparable with every other quantity in the programme; the other
#: two exist to show that the reading is a property of the family sample rather
#: than of one resampling draw.
BOOTSTRAP_SEEDS: tuple[int, ...] = (20260923, 20260924, 20260925)
BOOTSTRAP_DRAWS = 2000

#: Cell labels that are controls rather than treatments, with what each shows.
#: A null cell is deliberately an algebraic no-op, which is the one case where
#: the byte-identity refusal must not fire: the refusal exists to stop a no-op
#: being *reported as a localisation*, and a null control is a no-op reported as
#: a no-op. Treatment cells keep the refusal.
CONTROL_CELLS: dict[str, str] = {
    'floor': 'the recipient untouched; must reproduce its admitted anchor endpoint',
    'ceiling': 'the donor reconstructed; must reproduce its admitted anchor endpoint',
    'null': ('a declared group copied from the recipient into itself: a deliberate '
             'algebraic no-op, and the one cell exempt from the byte-identity refusal, '
             'which exists to stop a no-op being reported as a localisation rather than '
             'to stop a no-op being reported as a no-op. It must return the floor '
             'bit-identically, which is what shows the intervention machinery '
             'contributes nothing of its own'),
    'scrambled': ('a declared group installed at permuted destinations; gives replay '
                  'uncertainty a measured magnitude instead of an assumed one'),
}


def read_cell(paths: list[Path], anchor: set[str], cohort: dict) -> dict:
    """One cell's manifest, verified records, and its anchor-restricted rows."""
    manifests = [json.loads(path.read_bytes()) for path in paths]
    if any(m.get('status') != 'complete' for m in manifests):
        raise SystemExit(f'{paths}: incomplete cell manifest')
    identity = manifests[0]['identity']
    if any(m['identity'] != identity for m in manifests[1:]):
        raise SystemExit(f'{paths}: shard identities differ')
    rows, seen = [], set()
    for path, manifest in zip(paths, manifests):
        records = path.parent / manifest['records_file']
        raw = records.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest['records_sha256']:
            raise SystemExit(f'{records}: record digest mismatch')
        for line in raw.decode().splitlines():
            record = json.loads(line)
            if record['assay'] in seen:
                raise SystemExit(f"{record['assay']}: duplicated record")
            seen.add(record['assay'])
            if record['assay'] not in anchor:
                continue
            source = cohort[record['assay']]
            if (record['cluster'] != source['cluster']
                    or record['mutant_digest'] != source['mutant_digest']
                    or len(record['scores']) != len(source['mutants'])):
                raise SystemExit(f"{record['assay']}: record does not match the bound cohort")
            values = np.asarray(record['scores'], dtype=np.float64)
            if not np.isfinite(values).all():
                raise SystemExit(f"{record['assay']}: nonfinite mutation score")
            measured = np.asarray(source['measured'], dtype=np.float64)
            rows.append({'assay': record['assay'], 'cluster': record['cluster'],
                         'spearman': correlation(rankdata(values), standardized_rank(measured))})
    return {'identity': identity, 'transplant': manifests[0]['transplant'],
            'assays_in_record': len(seen), 'rows': sorted(rows, key=lambda r: r['assay'])}


def endpoint_at_seeds(rows: list[dict]) -> dict:
    """The endpoint at each declared bootstrap seed, and the spread across them."""
    per_seed = {str(seed): summarize(rows, 'spearman', bootstrap=BOOTSTRAP_DRAWS, seed=seed)
                for seed in BOOTSTRAP_SEEDS}
    points = {record['point'] for record in per_seed.values()}
    lows = [record['interval'][0] for record in per_seed.values()]
    highs = [record['interval'][1] for record in per_seed.values()]
    return {
        'per_bootstrap_seed': per_seed,
        'point_identical_across_seeds': len(points) == 1,
        'point': next(iter(points)) if len(points) == 1 else None,
        'interval_low_range': [min(lows), max(lows)],
        'interval_high_range': [min(highs), max(highs)],
        'resolves_above_zero_at_every_seed': all(low > 0 for low in lows),
        'resolves_below_zero_at_every_seed': all(high < 0 for high in highs),
    }


def contrast_at_seeds(cell_rows: list[dict], floor_rows: list[dict],
                      ceiling_rows: list[dict]) -> dict:
    """Cell minus floor, and the share of the ceiling-minus-floor gap it recovers."""
    by_assay = {label: {row['assay']: row['spearman'] for row in rows}
                for label, rows in (('cell', cell_rows), ('floor', floor_rows),
                                    ('ceiling', ceiling_rows))}
    shared = sorted(set(by_assay['cell']) & set(by_assay['floor']) & set(by_assay['ceiling']))
    clusters = {row['assay']: row['cluster'] for row in floor_rows}
    families = [clusters[assay] for assay in shared]
    numerator = [by_assay['cell'][assay] - by_assay['floor'][assay] for assay in shared]
    denominator = [by_assay['ceiling'][assay] - by_assay['floor'][assay] for assay in shared]
    out = {'assays': len(shared), 'families': len(set(families)), 'per_bootstrap_seed': {}}
    for seed in BOOTSTRAP_SEEDS:
        out['per_bootstrap_seed'][str(seed)] = {
            'minus_floor': summarize(
                [{'cluster': f, 'value': v} for f, v in zip(families, numerator)], 'value',
                bootstrap=BOOTSTRAP_DRAWS, seed=seed),
            'recovered_fraction': share_bootstrap(
                numerator, denominator, families, resamples=BOOTSTRAP_DRAWS, seed=seed),
        }
    # Below the shared unit floor ``summarize`` withholds the interval, and a
    # contrast with no interval resolves nothing. Reported as degenerate rather
    # than crashed on or silently read as unresolved-with-an-interval: the anchor
    # carries 163 families and the screen panel 56, so this fires only if a cell
    # lost assays, which is itself the finding.
    intervals = [r['minus_floor']['interval'] for r in out['per_bootstrap_seed'].values()]
    out['degenerate_at_some_seed'] = any(interval is None for interval in intervals)
    if out['degenerate_at_some_seed']:
        out['minus_floor_resolves_above_zero_at_every_seed'] = False
        out['minus_floor_interval_low_range'] = None
        out['minus_floor_interval_high_range'] = None
        out['degenerate_reason'] = next(
            r['minus_floor'].get('degenerate_reason')
            for r in out['per_bootstrap_seed'].values() if r['minus_floor']['interval'] is None)
        return out
    lows = [interval[0] for interval in intervals]
    highs = [interval[1] for interval in intervals]
    out['minus_floor_resolves_above_zero_at_every_seed'] = all(low > 0 for low in lows)
    out['minus_floor_interval_low_range'] = [min(lows), max(lows)]
    out['minus_floor_interval_high_range'] = [min(highs), max(highs)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--roster', type=Path, required=True,
                        help='admitted expansion roster; supplies the anchor assay ids')
    parser.add_argument('--census', type=Path, required=True)
    parser.add_argument('--cell', action='append', required=True, metavar='LABEL=PATH[,PATH...]')
    parser.add_argument('--admitted-floor', type=Path,
                        help='admitted common-support readout report for the recipient arm')
    parser.add_argument('--admitted-ceiling', type=Path)
    parser.add_argument('--sensitivity-arm', action='store_true',
                        help='the 56-family arm: skip the admitted-endpoint gate, because the '
                             'admitted figure is a 201-assay quantity and this arm scores 56, '
                             'so containment in it would be a comparison across supports')
    parser.add_argument('--support', default='anchor201')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    roster = json.loads(args.roster.read_bytes())
    anchor_ids = sorted(roster['panels'][args.support]['assay_ids'])
    anchor_digest = support_digest(anchor_ids)
    if anchor_digest != mi.ANCHOR['assay_id_sha256']:
        raise SystemExit(f'roster anchor digest {anchor_digest} is not the declared '
                         f"{mi.ANCHOR['assay_id_sha256']}")
    cohort = {a['assay']: a for a in json.loads(args.cohort.read_bytes())['assays']}
    census = json.loads(args.census.read_bytes())
    census['checkpoint_order_arms'] = ['recipient', 'donor']

    cells = {}
    for item in args.cell:
        label, _, paths = item.partition('=')
        if not label or not paths or label in cells:
            raise SystemExit(f'invalid or repeated cell specification: {item}')
        cells[label] = read_cell([Path(p) for p in paths.split(',')], set(anchor_ids), cohort)
    for required in ('floor', 'ceiling'):
        if required not in cells:
            raise SystemExit(f'the pilot requires its {required} cell')

    admitted = {}
    if not args.sensitivity_arm:
        if args.admitted_floor is None or args.admitted_ceiling is None:
            raise SystemExit('the full arm requires --admitted-floor and --admitted-ceiling')
        admitted = {'floor': json.loads(args.admitted_floor.read_bytes()),
                    'ceiling': json.loads(args.admitted_ceiling.read_bytes())}
        for end, report in admitted.items():
            if report['support']['definition'] != 'common' or report['n_assays'] != len(anchor_ids):
                raise SystemExit(f'{end}: admitted report is not the {len(anchor_ids)}-assay '
                                 'common-support fit')

    report: dict = {
        'schema_version': 'd2_pilot_anchor_analysis_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'support': {'panel': args.support, 'assays': len(anchor_ids),
                    'assay_id_sha256': anchor_digest,
                    'restricted_from': 'the lineage 211-assay native support, containment '
                                       'established exactly at assay and variant level'},
        'endpoint_contract': mi.endpoint_contract(),
        'bootstrap': {'seeds': list(BOOTSTRAP_SEEDS), 'draws': BOOTSTRAP_DRAWS,
                      'unit': mi.ANCHOR['resampling_unit']},
        'split_seed_stability': ('undefined for this endpoint: the native likelihood enters as '
                                 'one scalar column, so there is no fitted readout and no '
                                 'split seed. Not checked, and not to be assumed checked.'),
        'control_cells': CONTROL_CELLS,
        'cells': {},
        'gate': {},
        'licenses': mi.LICENSE_STATEMENT,
        'arm': 'sensitivity_56_families' if args.sensitivity_arm else 'full_anchor',
        'inputs_sha256': {str(p): sha256_file(p) for p in
                          [args.cohort, args.roster, args.census]
                          + [x for x in (args.admitted_floor, args.admitted_ceiling)
                             if x is not None]},
    }

    for label, cell in sorted(cells.items()):
        groups = resolve_groups(cell['identity']['groups'],
                               allowed=tuple(census['group_order']))
        # A cell is a control because its own receipt says so, not because of the
        # label a caller chose: classifying by label silently reclassified
        # `null_attention_q1` and `scrambled_attention_q1` as treatments, which
        # would have put two controls into the treatment set the conditions are
        # read over.
        declared_control = cell['identity'].get('control', 'none')
        entry: dict = {
            'groups_specification': cell['identity']['groups'],
            'selected_groups': list(groups),
            'control': declared_control,
            'kind': ('control' if label in CONTROL_CELLS or declared_control != 'none'
                     else 'treatment'),
            'tests': ('necessity in this context' if cell['identity']['groups'].startswith(
                'complement:') else 'sufficiency in this context'),
            'assays_in_record': cell['assays_in_record'],
            'restricted': {'assays': len(cell['rows']),
                           'families': len({row['cluster'] for row in cell['rows']})},
            'code_sha256': cell['identity'].get('code_sha256', {}),
            'endpoint': endpoint_at_seeds(cell['rows']),
        }
        if label in ('floor', 'ceiling'):
            entry['reconstruction'] = mi.check_reconstruction(
                cell['transplant']['verified_tensor_fp32_sha256'], census, end=label)
            if args.sensitivity_arm:
                report['gate'][label] = {
                    'skipped': 'sensitivity arm; the admitted figure is a 201-assay quantity '
                               'and this arm scores 56, so a containment check against it '
                               'would compare across supports',
                    'within_admitted_interval': True}
            else:
                arm = 'recipient' if label == 'floor' else 'donor'
                declared = admitted[label]['summaries']['raw_M_spearman']
                observed = entry['endpoint']['per_bootstrap_seed'][str(BOOTSTRAP_SEEDS[0])]
                entry['admitted_comparison'] = mi.check_admitted_endpoint(
                    {'point': observed['point']},
                    {'point': declared['point'], 'interval': declared['interval']})
                entry['admitted_comparison']['arm'] = arm
                report['gate'][label] = entry['admitted_comparison']
        elif entry['kind'] == 'treatment':
            entry['provenance'] = mi.group_provenance(census, groups)
        report['cells'][label] = entry

    for label, cell in sorted(cells.items()):
        if label == 'floor':
            continue
        report['cells'][label]['contrast'] = contrast_at_seeds(
            cell['rows'], cells['floor']['rows'], cells['ceiling']['rows'])
    nulls = [label for label, cell in cells.items()
             if cell['identity'].get('control') == 'null']
    for label in nulls:
        floor_points = [row['spearman'] for row in cells['floor']['rows']]
        null_points = [row['spearman'] for row in cells[label]['rows']]
        report['gate']['null_is_bit_identical_to_floor'] = floor_points == null_points
        report['gate']['null_cell'] = label

    failed = [end for end in ('floor', 'ceiling')
              if not report['gate'][end]['within_admitted_interval']]
    report['gate']['ends_reproduce_their_admitted_endpoints'] = not failed
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print(f'analysis: {args.out}')
    print(f'sha256={sha256_file(args.out)}')
    for end in ('floor', 'ceiling'):
        g = report['gate'][end]
        if 'skipped' in g:
            print(f"  {end}: admitted gate skipped ({report['arm']})")
        else:
            print(f"  {end}: observed {g['observed']:.10f} admitted {g['admitted']:.10f} "
                  f"|diff| {g['absolute_difference']:.3e} inside={g['within_admitted_interval']}")
    if failed:
        raise SystemExit(f'the pilot stops: {failed} do not reproduce their admitted endpoints, '
                         'so no localisation number is computed against that scale')


if __name__ == '__main__':
    main()
