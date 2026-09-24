#!/usr/bin/env python
"""Refit published representation cells at a selected readout class and depth.

Three subcommands, in the order they are used.

``plan`` joins a cell roster with the readout reassessment's per-arm selection and
reports which cells can run and which cannot, before any array is read. A cell is
refused when its gate fitted no representation design, when the gate's retained
artefacts cannot serve the selection kind, when no adapter loads that gate's
cells, or when its published report is not where the roster says. Refusing at
plan time is the point: which gates need a fresh extraction is a property of the
retention declaration and should be learned from it rather than from a
half-finished campaign.

``replay`` is the pipeline-identity control and is the only subcommand that runs
before the reassessment lands. It refits one cell at the **admitted** selection
from the retained states and requires the published held-out predictions back
inside the tolerance the admitted admission receipt declares, on identical
support, folds, seeds, weighting and label budget. A cell that fails it raises.

``run`` is the recomputation. It refuses without the reassessment's own selection
declaration, because a recomputation against a default selection would report an
increment that no reassessment chose.

    python scripts/transfer/recompute_representation_cells.py plan \
      --selection <reassessment selection>.json --cells <roster>.json \
      --out logs/d1_recomputation_20260924/plan.json

    python scripts/transfer/recompute_representation_cells.py replay \
      --cohort logs/d1_readout_20260923/cohort.json \
      --manifest <extraction>/manifest_<arm>.json --arm <arm> --panel anchor201 \
      --split-seed 20260923 --published <admitted analysis>.json \
      --out logs/d1_recomputation_20260924/replay_<arm>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.io import write_json  # noqa: E402
from src.transfer.recomputation import (  # noqa: E402
    CellRequest, IMPLEMENTED_ADAPTERS, ReadoutSelection, RecomputationRefused, load_selection,
    plan, readout_cell, readout_rows)

#: Roster fields a cell record must carry. The support is not among them: it is
#: read from the published report, so a cell is fitted on the support its own
#: published increment was fitted on rather than on a separately declared one.
CELL_FIELDS = ('gate', 'arm', 'panel', 'split_seed', 'published', 'cohort', 'manifest')


def _roster(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f'no cell roster at {path}')
    record = json.loads(path.read_text())
    cells = record.get('cells')
    if not isinstance(cells, list) or not cells:
        raise SystemExit(f'{path} lists no cells')
    for index, cell in enumerate(cells):
        missing = [field for field in CELL_FIELDS if field not in cell]
        if missing:
            raise SystemExit(f'{path} cell {index} is missing {missing}')
    return cells


def _requests(cells: list[dict], selection: dict[str, ReadoutSelection]) -> list[CellRequest]:
    requests = []
    for cell in cells:
        arm = cell['arm']
        if arm not in selection:
            raise SystemExit(f'the selection declares no class and depth for arm {arm!r}; a cell '
                             'is not recomputed at a selection the reassessment did not make')
        requests.append(CellRequest(gate=cell['gate'], arm=arm, panel=cell['panel'],
                                    split_seed=int(cell['split_seed']),
                                    selection=selection[arm],
                                    published=Path(cell['published'])))
    return requests


def _published(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f'no published cell report at {path}')
    return json.loads(path.read_text())


def _load(cohort, manifest, arm: str, support):
    """Retained states for one arm, with an absent archive named as a refusal.

    An archive the manifest lists and the filesystem does not hold is a retention
    finding -- the arrays were not staged where this host can read them -- rather
    than a fault in the loader, so it is reported as a refusal naming the file
    instead of as a traceback.
    """
    try:
        return readout_rows(cohort, manifest, arm, support)
    except FileNotFoundError as missing:
        raise SystemExit(
            f'refused: {arm} cannot be recomputed from this host because a retained archive the '
            f'manifest lists is not present: {missing.filename}. The admitted extraction archives '
            'live on the allocation\'s shared filesystem; the replay runs where they are.'
        ) from missing


def _replay(args) -> int:
    published = _published(args.published)
    support = published.get('support', {}).get('assay_ids')
    if support is None:
        raise SystemExit(f'{args.published} declares no support.assay_ids, so the replay has no '
                         'published support to be fitted on')
    request = CellRequest(gate=args.gate, arm=args.arm, panel=args.panel,
                          split_seed=args.split_seed,
                          selection=ReadoutSelection(arm=args.arm), published=args.published)
    rows, blocks, counts, provenance = _load(args.cohort, args.manifest, args.arm, support)
    record = readout_cell(request, rows, published, admitted_blocks=blocks, variant_counts=counts,
                          receipt=args.receipt,
                          progress=(lambda name: print(f'  {name}', flush=True))
                          if args.progress else None)
    record['extraction_provenance'] = dict(
        arm=provenance['arm'], hidden_width=provenance['hidden_width'],
        n_assays=provenance['n_assays'], n_clusters=provenance['n_clusters'],
        n_variants=provenance['n_variants'],
        projection_blas_threads=provenance['projection_blas_threads'],
        extraction_block_indices=provenance['extraction_block_indices'],
        source_sha256=provenance['source_sha256'])
    write_json(args.out, record)
    print(json.dumps(dict(out=str(args.out), gate=args.gate, arm=args.arm, panel=args.panel,
                          split_seed=args.split_seed,
                          identity=record['identity'],
                          prediction_agreement=record['prediction_agreement'],
                          product_blocking=record['product_blocking'],
                          increments=[dict(metric=row['metric'],
                                           published=row.get('published'),
                                           recomputed=row.get('recomputed'),
                                           point_change=row.get('point_change'))
                                      for row in record['increments']]), indent=1))
    return 0


def _key(cell) -> tuple:
    if isinstance(cell, dict):
        return cell['gate'], cell['arm'], cell['panel'], int(cell['split_seed'])
    return cell.gate, cell.arm, cell.panel, cell.split_seed


def _run(args) -> int:
    selection = load_selection(args.selection)
    cells = _roster(args.cells)
    requests = _requests(cells, selection)
    schedule = plan(requests)
    if not schedule['runnable']:
        raise SystemExit('the plan admits no runnable cell:\n  '
                         + '\n  '.join(f"{cell['gate']}/{cell['arm']}/{cell['panel']}: "
                                       f"{cell['refusal']}" for cell in schedule['refused']))
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'plan.json', schedule)
    by_key = {_key(cell): cell for cell in cells}
    by_request = {_key(request): request for request in requests}
    written = []
    for entry in schedule['runnable']:
        request = by_request[_key(entry)]
        cell = by_key[_key(entry)]
        published = _published(request.published)
        support = published.get('support', {}).get('assay_ids')
        if support is None:
            raise SystemExit(f'{request.published} declares no support.assay_ids')
        rows, blocks, counts, _ = _load(cell['cohort'], cell['manifest'], request.arm, support)
        record = readout_cell(request, rows, published, admitted_blocks=blocks,
                              variant_counts=counts, receipt=args.receipt)
        path = args.out / f'{request.gate}_{request.arm}_{request.panel}_{request.split_seed}.json'
        write_json(path, record)
        written.append(str(path))
    print(json.dumps(dict(out=str(args.out), written=written,
                          refused=[dict(gate=c['gate'], arm=c['arm'], panel=c['panel'],
                                        refusal=c['refusal']) for c in schedule['refused']]),
                     indent=1))
    return 0


def _plan(args) -> int:
    selection = load_selection(args.selection)
    schedule = plan(_requests(_roster(args.cells), selection))
    write_json(args.out, schedule)
    print(json.dumps(dict(out=str(args.out), cells=schedule['cells'],
                          runnable=len(schedule['runnable']), refused=len(schedule['refused']),
                          adapters=sorted(IMPLEMENTED_ADAPTERS),
                          refusals=[dict(gate=c['gate'], arm=c['arm'], panel=c['panel'],
                                         refusal=c['refusal']) for c in schedule['refused']]),
                     indent=1))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    for name, handler in (('plan', _plan), ('run', _run)):
        stage = sub.add_parser(name, help=handler.__doc__)
        stage.add_argument('--selection', type=Path, required=True,
                           help="the readout reassessment's own per-arm class and depth")
        stage.add_argument('--cells', type=Path, required=True, help='the cell roster')
        stage.add_argument('--out', type=Path, required=True,
                           help='report path for plan; output directory for run')
        stage.add_argument('--receipt', type=Path,
                           default=REPO_ROOT / 'results/transfer/readout_20260923/'
                                               'final_admission.json',
                           help='the admitted admission receipt the tolerance is read from')
        stage.set_defaults(handler=handler)

    replay = sub.add_parser('replay', help='the pipeline-identity control at the admitted selection')
    replay.add_argument('--gate', default='readout_panel')
    replay.add_argument('--arm', required=True)
    replay.add_argument('--panel', required=True)
    replay.add_argument('--split-seed', type=int, required=True)
    replay.add_argument('--cohort', type=Path, required=True)
    replay.add_argument('--manifest', type=Path, required=True)
    replay.add_argument('--published', type=Path, required=True)
    replay.add_argument('--out', type=Path, required=True)
    replay.add_argument('--progress', action='store_true')
    replay.add_argument('--receipt', type=Path,
                        default=REPO_ROOT / 'results/transfer/readout_20260923/'
                                            'final_admission.json')
    replay.set_defaults(handler=_replay)

    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except RecomputationRefused as refusal:
        raise SystemExit(f'refused: {refusal}') from refusal


if __name__ == '__main__':
    raise SystemExit(main())
