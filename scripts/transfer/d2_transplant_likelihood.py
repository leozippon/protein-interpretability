#!/usr/bin/env python3
"""Native likelihood of one transplanted cell on the admitted readout support."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.transfer import capability_transplant as ct
from src.transfer import context_homologue as ch
from src.transfer.io import write_json
from src.transfer.readout_extraction import (FEATURE_NAMES, extract_batch, load_readout_arm,
                                             representation_blocks)

ARM = 'llama-2-7b'
BUDGET = 1024


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), ROOT/'scripts/transfer'/name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eligible_support(arm, cohort, extraction):
    """Replay the extraction's own eligibility, then bind it to the admitted support."""
    selected, skipped = [], []
    for row in cohort['assays']:
        maximum = extraction.eligible_max_tokens(arm, [row['wildtype'], *row['sequences']], BUDGET)
        if maximum is None:
            skipped.append(dict(assay=row['assay'], reason='full target exceeds 1024 positions'))
        else:
            selected.append(dict(row, max_packed_tokens=maximum))
    return selected, skipped


def bind_admitted_support(selected, skipped, manifest):
    """Refuse any support that is not the admitted arm's, assay for assay."""
    if manifest['status'] != 'complete' or manifest['identity']['arm'] != ARM:
        raise ValueError('support manifest is not the admitted parent extraction')
    if manifest['identity']['budget'] != BUDGET or manifest.get('execution_partition') is not None:
        raise ValueError('support manifest is sharded or uses another position budget')
    admitted = {row['assay']: row for row in manifest['assays']}
    if len(admitted) != len(manifest['assays']):
        raise ValueError('duplicate admitted assay')
    if sorted(row['assay'] for row in selected) != sorted(admitted):
        raise ValueError('eligible support differs from the admitted readout support')
    if sorted(row['assay'] for row in skipped) != sorted(row['assay'] for row in manifest['skipped']):
        raise ValueError('excluded support differs from the admitted readout support')
    for row in selected:
        reference = admitted[row['assay']]
        if (reference['cluster'] != row['cluster'] or reference['wildtype_id'] != row['wildtype_id']
                or reference['mutant_digest'] != row['mutant_digest']
                or reference['variants'] != len(row['mutants'])
                or reference['max_packed_tokens'] != row['max_packed_tokens']):
            raise ValueError(f"{row['assay']}: admitted support identity mismatch")
    return dict(
        n_assays=len(selected), n_families=len({row['cluster'] for row in selected}),
        n_variants=sum(len(row['mutants']) for row in selected),
        assay_ids=sorted(row['assay'] for row in selected),
        digest=ct.support_digest(row['assay'] for row in selected))


def score_assay(arm, stage46, row, path, metadata):
    """One packed row at a time, through the admitted extraction's own forward.

    The likelihood is the endpoint; the four block summaries are retained
    full-width beside it, at their native hidden size and without the readout
    study's fixed Gaussian projection, so that a later readout of these same
    transplanted states needs no further model inference.
    """
    features, values = [], []
    for sequence in [row['wildtype'], *row['sequences']]:
        block, likelihood = extract_batch(arm, [sequence], stage46)
        features.append(block)
        values.append(likelihood)
    block = np.concatenate(features)
    values = np.concatenate(values).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{row['assay']}: nonfinite likelihood")
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, features=(block[1:]-block[0]).astype(np.float32),
                            wt_features=block[0], likelihood=values[1:]-values[0],
                            wt_likelihood=values[0], mutants=np.asarray(row['mutants']),
                            metadata=json.dumps(metadata))
    temporary.replace(path)
    return (values[1:]-values[0]).tolist(), float(values[0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--support-manifest', type=Path, required=True,
                   help='Admitted parent extraction manifest that declares the readout support')
    p.add_argument('--census', type=Path, required=True)
    p.add_argument('--source-checkpoint', type=Path, required=True)
    p.add_argument('--cell', required=True, help='Declared cell label')
    p.add_argument('--groups', required=True, help="'none', 'all', a group list, or 'complement:<list>'")
    p.add_argument('--control', choices=('none', 'null', 'scrambled', 'spectrum_matched'),
                   default='none',
                   help="'null' copies the recipient's own tensors for the selected groups back "
                        "over themselves and must leave the model bit-identical to the floor; "
                        "'scrambled' installs the donor's tensors at a declared derangement of "
                        "same-shape destinations; 'spectrum_matched' installs tensors "
                        "carrying the donor's exact singular values with random orthogonal "
                        "factors. All three are controls, not treatments.")
    p.add_argument('--screen-families', type=int, default=0,
                   help='Label-blind family subsample size; 0 scores the full admitted support')
    p.add_argument('--screen-seed', type=int, default=20260923)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--shard-count', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    args = p.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError('Invalid shard index/count')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    stage46 = load_script('46_context_homologue.py')
    extraction = load_script('extract_frozen_readout.py')
    cohort = json.loads(args.cohort.read_text())
    census = json.loads(args.census.read_text())
    if census.get('schema_version') != 'd2_transplant_census_v1' or census.get('status') != 'complete':
        raise ValueError('incompatible tensor census')
    selected_groups = ct.resolve_groups(args.groups)
    tokenizer_arm = load_readout_arm(ARM, stage46, dtype='float32')
    selected, skipped = eligible_support(tokenizer_arm, cohort, extraction)
    support = bind_admitted_support(selected, skipped, json.loads(args.support_manifest.read_text()))
    panel = None
    if args.screen_families:
        panel = ct.select_screen_panel(selected, families=args.screen_families, seed=args.screen_seed)
        chosen = set(panel['selected_assay_ids'])
        selected = [row for row in selected if row['assay'] in chosen]
    partition = None
    if args.shard_count > 1:
        universe = [dict(assay=row['assay'], max_packed_tokens=row['max_packed_tokens'],
                         variants=len(row['mutants'])) for row in selected]
        assigned = extraction.partition_assays(universe, args.shard_count)[args.shard_index]
        partition = dict(method='greedy_packed_token_work_v1', count=args.shard_count,
                         index=args.shard_index, universe=universe, assigned_assays=assigned)
        selected = [row for row in selected if row['assay'] in set(assigned)]
    if not selected:
        raise ValueError('empty scoring support')
    code_paths = [Path(__file__), ROOT/'src/transfer/capability_transplant.py',
                  ROOT/'src/transfer/readout_extraction.py', ROOT/'src/transfer/context_homologue.py',
                  ROOT/'scripts/transfer/46_context_homologue.py',
                  ROOT/'scripts/transfer/extract_frozen_readout.py', ROOT/'src/transfer/arms.py',
                  ROOT/'src/transfer/joint_lineage.py']
    destination = Path(census['sources'][0]['checkpoint'])
    identity = dict(
        schema_version='d2_transplant_likelihood_v1', cell=args.cell, groups=args.groups,
        control=args.control, selected_groups=list(selected_groups), arm=ARM, dtype='float32',
        batch_size=1, budget=BUDGET,
        destination_checkpoint=str(destination.resolve()),
        source_checkpoint=str(args.source_checkpoint.resolve()),
        cohort_sha256=sha(args.cohort), census_sha256=sha(args.census),
        support_manifest_sha256=sha(args.support_manifest), support=support,
        screen_panel=panel and dict(panel, digest=ct.panel_digest(panel)),
        code_sha256={str(x.relative_to(ROOT)): sha(x) for x in code_paths},
        scoring_stratum='target_only_native_packed_residue_span',
        scoring='native summed target log likelihood, mutant minus wild type',
        feature_names=list(FEATURE_NAMES),
        block_semantics='zero-based transformer block output, before final output normalization',
        representation_semantics=('mutant-minus-WT; residue-bearing token mean and last token; '
                                  'no EOS or padding; retained at native hidden width with no '
                                  'projection; BPE token means are not per-residue means'),
        retention=('per-assay NPZ beside each record, carrying the full-width block summaries, '
                   'the wild-type states and the per-variant likelihood delta; no measured '
                   'effect or profile score is written by this stage'),
        precision_gate=('none; at batch size one the repeat check and the production '
                        'forward are the same single-row computation, so a batch-composition '
                        'gate here would be vacuous and no numerical validation is claimed'))
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.cell if args.shard_count == 1 else f'{args.cell}_shard{args.shard_index}'
    manifest_path = args.out/f'd2_transplant_{stem}.json'
    records_path = args.out/f'd2_transplant_{stem}.jsonl'
    done = {}
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            record = json.loads(line)
            if record['identity'] != identity:
                raise ValueError('Resume identity differs; use a fresh output directory')
            if sha(args.out/record['features_file']) != record['features_sha256']:
                raise ValueError(f"Resume feature archive digest mismatch: {record['features_file']}")
            done[record['assay']] = record
    if args.source_checkpoint.resolve() != Path(census['sources'][1]['checkpoint']).resolve():
        raise ValueError('source checkpoint is not the census source')
    for index, source in enumerate(census['sources']):
        root = Path(source['checkpoint'])
        for name, expected in source['files'].items():
            if sha(root/name) != expected:
                raise ValueError(f'census checkpoint digest mismatch: {root.name}/{name}')
    arm = load_readout_arm(ARM, stage46, device=args.device, dtype='float32')
    arm.model.eval().requires_grad_(False)
    ch.require_position_budget(arm.model.config, arm=ARM)
    if args.control == 'null':
        receipt = ct.transplant_null(arm.model, census, selected_groups)
    elif args.control == 'scrambled':
        receipt = ct.transplant_scrambled(arm.model, args.source_checkpoint, census,
                                          selected_groups)
    elif args.control == 'spectrum_matched':
        receipt = ct.transplant_spectrum_matched(arm.model, args.source_checkpoint, census,
                                                 selected_groups)
    else:
        receipt = ct.transplant(arm.model, args.source_checkpoint, census, selected_groups)
    print(json.dumps(dict(cell=args.cell, transplanted_tensors=receipt['n_transplanted_tensors'],
                          transplanted_elements=receipt['n_transplanted_elements'])), flush=True)
    blocks = len(representation_blocks(arm))
    began = time.monotonic()
    features = args.out/f'features_{stem}'
    features.mkdir(parents=True, exist_ok=True)
    with records_path.open('a') as stream:
        for row in selected:
            if row['assay'] in done:
                continue
            name = f"{args.cell}_{hashlib.sha256(row['assay'].encode()).hexdigest()[:20]}.npz"
            metadata = dict(identity=identity, assay=row['assay'], mutant_digest=row['mutant_digest'])
            scores, wildtype = score_assay(arm, stage46, row, features/name, metadata)
            record = dict(identity=identity, assay=row['assay'], cluster=row['cluster'],
                          wildtype_id=row['wildtype_id'], mutant_digest=row['mutant_digest'],
                          n_variants=len(row['mutants']), scores=scores,
                          wildtype_log_likelihood=wildtype,
                          features_file=f'{features.name}/{name}',
                          features_sha256=sha(features/name))
            stream.write(json.dumps(record, allow_nan=False)+'\n')
            stream.flush()
            done[row['assay']] = record
            print(f"{args.cell}: {len(done)}/{len(selected)} {row['assay']}", flush=True)
    write_json(manifest_path, dict(
        status='complete', identity=identity, execution_partition=partition,
        transplant=receipt, records_file=records_path.name, records_sha256=sha(records_path),
        block_indices=[(blocks-1)//2, blocks-1], features_directory=features.name,
        assays=[dict(assay=row['assay'], cluster=row['cluster'], wildtype_id=row['wildtype_id'],
                     mutant_digest=row['mutant_digest'], n_variants=len(row['mutants']),
                     max_packed_tokens=row['max_packed_tokens'],
                     features_sha256=done[row['assay']]['features_sha256']) for row in selected],
        skipped=skipped, updated_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-began, gpu=torch.cuda.get_device_name(arm.device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(arm.device),
        torch_version=torch.__version__))


if __name__ == '__main__':
    main()
