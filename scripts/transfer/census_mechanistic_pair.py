#!/usr/bin/env python3
"""Census over candidate mechanism pairs. CPU-only, no cohort, no forward pass.

Two stages, and neither reads a model quantity against a label.

``pair`` is the compatibility census the handoff says is needed. A transplant
needs corresponding tensor names, corresponding shapes and a partition that
covers the checkpoint; only two of the handoff's six pairs could satisfy that,
and neither satisfies it as a matter of record. For the recommended non-Llama
pair the head counts are undeclared, the embedding and head widths stand at
50,000 against 50,304 rows, and whether the two checkpoints share an
initialisation is declared nowhere. This stage decides those questions by
measurement, and records a refusal rather than raising on it, because the
refusal is the finding.

``composition`` applies the depth composition rule to every outcome-differing
pair, reading the handoff's own separation report and property table rather than
transcribing them, so this stage cannot disagree with the document that owns
them.

    python scripts/transfer/census_mechanistic_pair.py pair \
      --destination <recipient checkpoint dir> --source <donor checkpoint dir> \
      --arms galactica-1.3b,instructprotein \
      --out logs/d2_mechanistic_interface_20260924/pair_census.json
    python scripts/transfer/census_mechanistic_pair.py composition \
      --separation logs/matched_pair_handoff/separation.json \
      --properties logs/matched_pair_handoff/arm_properties.json \
      --out logs/d2_mechanistic_interface_20260924/composition_rule.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transfer import mechanistic_interface as mi  # noqa: E402
from src.transfer.io import sha256_file, write_json  # noqa: E402

#: Torch thread count the census runs under. A float64 reduction's order depends
#: on it, so it is pinned rather than left to the host, at the count the admitted
#: projection ran at. It is also what keeps a census polite on a pod whose CPU is
#: shared between four allocations on one node.
CENSUS_THREADS = 4


#: The six pairs the handoff names, recipient first where a direction is
#: meaningful. Named here so the composition stage reports them in the handoff's
#: own ranking order; every fact about them is read from its artefacts.
NAMED_PAIRS: tuple[tuple[str, str], ...] = (
    ('llama-2-7b', 'prollama-stage-1'),
    ('gpt2-large', 'protgpt2'),
    ('progen3-112m', 'protgpt3-1.3b'),
    ('progen3-3b', 'rita-xl'),
    ('galactica-1.3b', 'instructprotein'),
    ('galactica-6.7b', 'progen2-xlarge'),
)


def stage_composition(args: argparse.Namespace) -> dict[str, object]:
    """The depth composition rule over every outcome-differing pair."""
    separation = json.loads(args.separation.read_bytes())
    properties = json.loads(args.properties.read_bytes())
    depths = {name: int(entry['depth_blocks']) for name, entry in properties['arms'].items()
              if entry.get('depth_blocks') is not None}
    index = {tuple(sorted(pair['arms'])): pair for pair in separation['pairs']}
    named, absent = [], []
    for arms in NAMED_PAIRS:
        pair = index.get(tuple(sorted(arms)))
        if pair is None:
            absent.append(list(arms))
            continue
        named.append(mi.composition_rule(pair, depths=depths))
    if absent:
        raise SystemExit(f'pairs absent from the separation report: {absent}')
    every = [mi.composition_rule(pair, depths=depths) for pair in separation['pairs']
             if pair['likelihood_outcome_differs'] or pair['representation_outcome_differs']]
    eligible = [record for record in every
                if record['conditions']['both_position_resolved']
                and record['conditions']['equal_depth']]
    return {
        'schema_version': 'd2_composition_rule_v1',
        'rule': mi.composition_rule.__doc__.strip().splitlines()[0],
        'inputs_sha256': {str(path): sha256_file(path)
                          for path in (args.separation, args.properties)},
        'property_table_sha256': separation['property_table_sha256'],
        'named_pairs': named,
        'outcome_differing_pairs': len(every),
        'pairs_satisfying_the_first_two_conditions': [record['arms'] for record in eligible],
        'pending_depth_sweep': (
            'The third condition -- per-block increment profiles differing at an '
            'identifiable block or contiguous band -- is left None until the depth sweep '
            'reports, so no pair is recorded as satisfying the rule on two conditions out '
            'of three.'),
        'licenses': mi.LICENSE_STATEMENT,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=('pair', 'composition', 'audit'))
    parser.add_argument('--destination', type=Path,
                        help='the recipient checkpoint: the one a transplant copies into')
    parser.add_argument('--source', type=Path,
                        help='the donor checkpoint: the one a transplant copies from')
    parser.add_argument('--arms', help='the two arm names, recipient first, comma separated')
    parser.add_argument('--separation', type=Path, help="the handoff's separation report")
    parser.add_argument('--checkpoint', action='append', default=[], metavar='NAME=PATH',
                        help='audit stage: a checkpoint directory to read config.json from')
    parser.add_argument('--properties', type=Path, help="the handoff's property table")
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=CENSUS_THREADS)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    if args.stage == 'audit':
        if not args.checkpoint:
            raise SystemExit('audit requires at least one --checkpoint NAME=PATH')
        roster = {}
        for item in args.checkpoint:
            name, _, path = item.partition('=')
            if not name or not path or name in roster:
                raise SystemExit(f'invalid or repeated checkpoint specification: {item}')
            roster[name] = Path(path)
        record = mi.config_key_audit(roster)
        record['created_utc'] = datetime.now(timezone.utc).isoformat()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, record)
        print(f'audit: {args.out}')
        print(f'sha256={sha256_file(args.out)}')
        print('blind fields: ' + ', '.join(
            record['fields_a_single_key_comparison_would_miss']) or 'none')
        return
    if args.stage == 'composition':
        if args.separation is None or args.properties is None:
            raise SystemExit('composition requires --separation and --properties')
        record = stage_composition(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, record)
        print(f'composition: {args.out}')
        print(f'sha256={sha256_file(args.out)}')
        return
    if args.destination is None or args.source is None or args.arms is None:
        raise SystemExit('pair requires --destination, --source and --arms')
    arms = [piece.strip() for piece in args.arms.split(',') if piece.strip()]
    if len(arms) != 2:
        raise SystemExit('--arms takes exactly two comma-separated arm names')

    import torch

    torch.set_num_threads(int(args.threads))

    def progress(row):
        if not args.quiet:
            print(f"  {row['name']:<52} {row['shapes'][0]} vs {row['shapes'][1]} "
                  f"compatible={row['shape_compatible']} r={row['pearson_r']}",
                  file=sys.stderr, flush=True)

    census = mi.pair_census(args.destination, args.source, arms=arms, progress=progress)
    initialisation = mi.shared_initialisation_verdict(census)
    try:
        admitted = mi.assert_pair_transplantable(census)
        refusal = None
    except ValueError as error:
        admitted, refusal = None, str(error)
    usage = shutil.disk_usage(args.out.parent if args.out.parent.exists() else ROOT)
    record = {
        **census,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'shared_initialisation': initialisation,
        'transplant_defined': refusal is None,
        'transplant_refusal': refusal,
        'transplant_admission': admitted,
        'licenses': mi.LICENSE_STATEMENT,
        'execution': {
            'python': sys.version.split()[0],
            'torch': torch.__version__,
            'torch_threads': int(args.threads),
            'cpu_count': os.cpu_count(),
            'disk_free_gib': round(usage.free / 2 ** 30, 1),
            'reads': ('two checkpoints\' config.json and stored tensors; no cohort, no '
                      'measured effect, no forward pass and no GPU'),
        },
        'code_sha256': {
            str(path.relative_to(ROOT)): sha256_file(path) for path in
            [Path(__file__), ROOT / 'src/transfer/mechanistic_interface.py',
             ROOT / 'src/transfer/capability_transplant.py']},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, record)
    print(f'census: {args.out}')
    print(f'sha256={sha256_file(args.out)}')
    print(f'transplant_defined={record["transplant_defined"]}')
    if refusal:
        print(f'refusal={refusal}')


if __name__ == '__main__':
    main()
