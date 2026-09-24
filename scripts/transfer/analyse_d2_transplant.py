#!/usr/bin/env python3
"""Endpoints and recovered fractions for the transplanted cells of one wave."""
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
sys.path.insert(0, str(ROOT))
from src.transfer import capability_transplant as ct
from src.transfer.io import write_json
from src.transfer.profile_increment import correlation, standardized_rank, summarize
from src.transfer.profiles import share_bootstrap

FLOOR, CEILING = 'floor', 'ceiling'
#: Identity fields every cell of one wave must agree on, because a contrast
#: across cells that differ on any of them is not a contrast of components.
SHARED = ('schema_version', 'arm', 'dtype', 'batch_size', 'budget', 'destination_checkpoint',
          'source_checkpoint', 'cohort_sha256', 'census_sha256', 'support_manifest_sha256',
          'support', 'screen_panel', 'code_sha256', 'scoring_stratum', 'scoring')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, hashes):
    raw = Path(path).read_bytes()
    hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def load_cell(paths, hashes, cohort_rows):
    """One cell, reassembled from its shards and checked against the cohort."""
    manifests = [read(path, hashes) for path in paths]
    if any(m.get('status') != 'complete' for m in manifests):
        raise ValueError('incomplete cell manifest')
    identity = manifests[0]['identity']
    if any(m['identity'] != identity for m in manifests[1:]):
        raise ValueError(f"{identity['cell']}: shard identities differ")
    partitions = [m['execution_partition'] for m in manifests]
    if len(manifests) > 1:
        counts = {p and p['count'] for p in partitions}
        indices = sorted(p['index'] for p in partitions if p)
        if counts != {len(manifests)} or indices != list(range(len(manifests))):
            raise ValueError(f"{identity['cell']}: shards do not cover the partition")
    elif partitions[0] is not None:
        raise ValueError(f"{identity['cell']}: a single manifest carries a shard partition")
    rows = {}
    for path, manifest in zip(paths, manifests):
        records_path = Path(path).parent/manifest['records_file']
        if sha(records_path) != manifest['records_sha256']:
            raise ValueError(f'{records_path}: record digest mismatch')
        hashes[str(records_path)] = manifest['records_sha256']
        declared = {row['assay'] for row in manifest['assays']}
        for line in records_path.read_text().splitlines():
            record = json.loads(line)
            if record['identity'] != identity:
                raise ValueError('record identity differs from its manifest')
            if record['assay'] not in declared or record['assay'] in rows:
                raise ValueError(f"{record['assay']}: undeclared or duplicated record")
            rows[record['assay']] = record
        if declared - set(rows):
            raise ValueError(f"{identity['cell']}: declared assays without records")
    receipt = dict(
        selected_groups=manifests[0]['transplant']['selected_groups'],
        n_transplanted_tensors=manifests[0]['transplant']['n_transplanted_tensors'],
        n_transplanted_elements=manifests[0]['transplant']['n_transplanted_elements'],
        n_transplanted_differing_elements=manifests[0]['transplant']['n_transplanted_differing_elements'],
        absent_destinations=manifests[0]['transplant']['absent_destinations'])
    for manifest in manifests[1:]:
        if {k: v for k, v in manifest['transplant'].items() if k != 'verified_tensor_fp32_sha256'} != \
           {k: v for k, v in manifests[0]['transplant'].items() if k != 'verified_tensor_fp32_sha256'}:
            raise ValueError(f"{identity['cell']}: shard transplant receipts differ")
        if manifest['transplant']['verified_tensor_fp32_sha256'] != manifests[0]['transplant']['verified_tensor_fp32_sha256']:
            raise ValueError(f"{identity['cell']}: shards verified different model states")
    for assay, record in rows.items():
        source = cohort_rows[assay]
        if (record['cluster'] != source['cluster'] or record['mutant_digest'] != source['mutant_digest']
                or record['n_variants'] != len(source['mutants'])
                or len(record['scores']) != len(source['mutants'])):
            raise ValueError(f'{assay}: cell record does not match the bound cohort')
    return identity, rows, receipt


def assay_spearman(scores, measured):
    """The readout study's own within-assay rank correlation, spelled its way."""
    values = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite mutation score')
    return correlation(rankdata(values), standardized_rank(np.asarray(measured, dtype=np.float64)))


def admitted_raw_m(report, arm, support):
    if report['arm'] != arm or report['support']['definition'] != 'native':
        raise ValueError(f'{arm}: admitted report is not the native-support fit')
    if sorted(report['support']['assay_ids']) != sorted(support['assay_ids']):
        raise ValueError(f'{arm}: admitted support differs from the scored support')
    record = report['summaries']['raw_M_spearman']
    if record.get('excluded_assays'):
        raise ValueError(f'{arm}: admitted raw likelihood summary excludes assays')
    return record


def compare(observed, admitted):
    """Exact reproduction is the expectation; interval containment is the gate."""
    low, high = admitted['interval']
    return dict(
        observed=observed['point'], admitted=admitted['point'], admitted_interval=admitted['interval'],
        exact=observed['point'] == admitted['point'] and observed['interval'] == admitted['interval'],
        absolute_difference=abs(observed['point']-admitted['point']),
        within_admitted_interval=bool(low <= observed['point'] <= high))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--cell', action='append', required=True, metavar='LABEL=PATH[,PATH...]')
    p.add_argument('--admitted-floor', type=Path, help='Admitted native-support parent readout report')
    p.add_argument('--admitted-ceiling', type=Path, help='Admitted native-support Stage 1 readout report')
    p.add_argument('--wave', required=True, help='Declared wave label, screen or confirm')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=20260923)
    args = p.parse_args()
    if (args.wave == 'confirm') != (args.admitted_floor is not None and args.admitted_ceiling is not None):
        raise ValueError('the confirmation wave requires both admitted reports and the screen takes none')
    hashes = {}
    cohort = read(args.cohort, hashes)
    cohort_rows = {row['assay']: row for row in cohort['assays']}
    if len(cohort_rows) != len(cohort['assays']):
        raise ValueError('duplicate cohort assay')
    cells, identities, receipts = {}, {}, {}
    for item in args.cell:
        label, _, paths = item.partition('=')
        if not label or not paths or label in cells:
            raise ValueError(f'invalid or repeated cell specification: {item}')
        identity, rows, receipt = load_cell([Path(x) for x in paths.split(',')], hashes, cohort_rows)
        if identity['cell'] != label:
            raise ValueError(f"cell label {label} does not match the manifest's {identity['cell']}")
        cells[label], identities[label], receipts[label] = rows, identity, receipt
    if set(cells) < {FLOOR, CEILING}:
        raise ValueError('every wave requires its floor and ceiling cells')
    reference = identities[FLOOR]
    for label, identity in identities.items():
        if any(identity[key] != reference[key] for key in SHARED):
            raise ValueError(f'{label}: identity differs from the floor on a shared field')
    if identities[FLOOR]['selected_groups'] or sorted(identities[CEILING]['selected_groups']) != sorted(ct.group_names()):
        raise ValueError('the floor must transplant nothing and the ceiling every group')
    support = reference['support']
    scored = sorted(set.intersection(*(set(rows) for rows in cells.values())))
    if any(sorted(rows) != scored for rows in cells.values()):
        raise ValueError('cells do not share one scored support')
    panel = reference['screen_panel']
    expected = sorted(panel['selected_assay_ids']) if panel else sorted(support['assay_ids'])
    if scored != expected:
        raise ValueError('scored support is neither the admitted support nor the declared panel')
    per_cell = {}
    for label, rows in cells.items():
        per_cell[label] = [dict(assay=assay, cluster=rows[assay]['cluster'],
                                spearman=assay_spearman(rows[assay]['scores'], cohort_rows[assay]['measured']))
                           for assay in scored]
    undefined = sorted({row['assay'] for rows in per_cell.values() for row in rows if row['spearman'] is None})
    common = [assay for assay in scored if assay not in set(undefined)]
    if len(common) < 8:
        raise ValueError('too few assays with a defined correlation in every cell')
    endpoints = {label: summarize(rows, 'spearman', bootstrap=args.bootstrap, seed=args.seed)
                 for label, rows in per_cell.items()}
    by_assay = {label: {row['assay']: row['spearman'] for row in rows} for label, rows in per_cell.items()}
    families = [cohort_rows[assay]['cluster'] for assay in common]
    denominator = [by_assay[CEILING][assay]-by_assay[FLOOR][assay] for assay in common]
    contrasts = {}
    for label in sorted(cells):
        numerator = [by_assay[label][assay]-by_assay[FLOOR][assay] for assay in common]
        above_ceiling = [by_assay[label][assay]-by_assay[CEILING][assay] for assay in common]
        contrasts[label] = dict(
            minus_floor=summarize([dict(cluster=f, value=v) for f, v in zip(families, numerator)],
                                  'value', bootstrap=args.bootstrap, seed=args.seed),
            minus_ceiling=summarize([dict(cluster=f, value=v) for f, v in zip(families, above_ceiling)],
                                    'value', bootstrap=args.bootstrap, seed=args.seed),
            recovered_fraction=share_bootstrap(numerator, denominator, families,
                                               resamples=args.bootstrap, seed=args.seed),
            transplanted=receipts[label])
    gate = None
    if args.wave == 'confirm':
        floor_report = read(args.admitted_floor, hashes)
        ceiling_report = read(args.admitted_ceiling, hashes)
        gate = dict(
            floor=compare(endpoints[FLOOR], admitted_raw_m(floor_report, 'llama-2-7b', support)),
            ceiling=compare(endpoints[CEILING], admitted_raw_m(ceiling_report, 'prollama-stage-1', support)))
        failed = [name for name, record in gate.items() if not record['within_admitted_interval']]
        if failed:
            raise ValueError(
                'the transplant path does not reproduce the admitted endpoint for '
                f'{failed}; no localisation number is computed against a scale that '
                'does not reconstruct the admitted models')
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out/f'd2_transplant_{args.wave}.json', dict(
        schema_version='d2_transplant_analysis_v1', status='complete', wave=args.wave,
        created_utc=datetime.now(timezone.utc).isoformat(), input_sha256=hashes,
        analysis_code_sha256={str(path.relative_to(ROOT)): sha(path) for path in
                              [Path(__file__), ROOT/'src/transfer/capability_transplant.py',
                               ROOT/'src/transfer/profile_increment.py', ROOT/'src/transfer/profiles.py']},
        support=support, screen_panel=panel, n_scored_assays=len(scored),
        n_contrast_assays=len(common), n_contrast_families=len(set(families)),
        undefined_correlation_exclusions=undefined,
        cells={label: dict(groups=identities[label]['selected_groups'],
                           group_specification=identities[label]['groups'],
                           assays=per_cell[label]) for label in sorted(cells)},
        endpoints=endpoints, contrasts=contrasts, admitted_comparison=gate,
        bootstrap=dict(resamples=args.bootstrap, seed=args.seed,
                       unit='wild-type family at 50% identity',
                       pairing='every contrast and share resamples the same families for every cell'),
        estimand=('within-assay Spearman of the native mutant-minus-wild-type summed log '
                  'likelihood against the measured effect, averaged within family and then '
                  'equally over families; the recovered fraction is that quantity minus the '
                  'floor over the ceiling minus the floor, taken inside each resample'),
        claim=('A cell that recovers the gain shows its parameter values are sufficient in '
               'this context to carry it against this parent, on this support, under this '
               'scoring. It does not show they are necessary, does not identify a circuit, '
               'and does not say what the computation is. A complement cell measures '
               'necessity and is reported separately.'),
        confound=('The destination and source checkpoints differ by continued pretraining on '
                  'a protein corpus, which fixed a corpus, a schedule and a data order at '
                  'once. Any localisation here is of where that training step\'s effect '
                  'lands in the parameters, not of an architectural locus.'),
        limitation=('Intervals are unadjusted pointwise percentile intervals conditional on '
                    'these checkpoints and this cohort; they omit training-seed and '
                    'cohort-selection variation and are not corrected for the number of '
                    'cells. Scoring ran at batch size one, where a batch-composition repeat '
                    'check is the same single-row computation as the production forward, so '
                    'no arm here is numerically validated on that basis.')))


if __name__ == '__main__':
    main()
