#!/usr/bin/env python3
"""Analyze the frozen attention/MLP factorial on paired families and seed batches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.stats import spearmanr

from src.transfer.io import sha256_file, write_json
from src.transfer.profiles import cluster_bootstrap
from src.transfer.component_statistics import paired_generation_contrast
from src.transfer.generation_failure import decompose
from scripts.transfer.analyse_prollama_components import representatives
from scripts.transfer.prollama_attention_mlp import select_panel, require_refinement_inventory


KEYS = ('00', '01', '10', '11')  # attention, MLP; 0=Stage 1, 1=Stage 2
POLICY = dict(attempts=64, residue_budget=400, token_safety_cap=1024,
              seed=20260923, batch_size=8, temperature=.85, top_p=.95,
              top_k=50, repetition_penalty=1.)
CONTRASTS = {
    'native_stage2_minus_stage1': {'00': -1., '11': 1.},
    'attention_marginal': {'00': -.5, '01': -.5, '10': .5, '11': .5},
    'mlp_marginal': {'00': -.5, '10': -.5, '01': .5, '11': .5},
    'attention_at_stage1_mlp': {'00': -1., '10': 1.},
    'attention_at_stage2_mlp': {'01': -1., '11': 1.},
    'mlp_at_stage1_attention': {'00': -1., '01': 1.},
    'mlp_at_stage2_attention': {'10': -1., '11': 1.},
    'attention_by_mlp_interaction': {'00': 1., '01': -1., '10': -1., '11': 1.},
}


def read_json(path, hashes):
    hashes[str(path)] = sha256_file(path)
    return json.loads(path.read_text())


def unique_rows(rows):
    result = {r['assay']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('duplicate assay')
    return result


def check_precision(manifest):
    precision = manifest['mutation_precision']
    if any(precision.get(k) != v for k, v in dict(dtype='float32', tf32=False, batch_size=1).items()):
        raise ValueError('mutation precision mismatch')
    compatibility = manifest['compatibility']
    if (compatibility.get('architecture_equal') is not True
            or compatibility.get('untied_embeddings') is not True
            or compatibility.get('native_reconstruction_max_absolute_logit_error') != [0., 0.]):
        raise ValueError('native architecture/reconstruction check failed')
    facts = compatibility['endpoint_loading_facts']
    if len(facts) != 2 or any(f.get('dtype_requested') != 'float32'
                            or f.get('dtype_observed') != ['float32'] for f in facts):
        raise ValueError('observed endpoint dtype is not FP32')
    return compatibility


def validate_panel(panel, cohort, native, *, cohort_hash, native_hash, inventory_hash):
    if panel.get('status') != 'complete' or panel.get('schema_version') != 'attention_mlp_panel_v1':
        raise ValueError('incomplete or incompatible panel')
    if (panel.get('source_cohort_sha256') != cohort_hash
            or native.get('cohort_sha256') != cohort_hash
            or panel.get('native_manifest_sha256') != native_hash
            or panel.get('inventory_sha256') != inventory_hash):
        raise ValueError('panel source digest mismatch')
    if panel.get('seed') != 20260923:
        raise ValueError('panel selection seed mismatch')
    replay = select_panel(cohort, native, seed=20260923)
    if any(panel.get(k) != value for k, value in replay.items()):
        raise ValueError('panel does not replay from source cohort and label-blind selection')
    source = unique_rows(cohort['assays'])
    rows = unique_rows(panel['assays'])
    eligible = native['common_assays']
    if len(set(eligible)) != len(eligible) or not set(eligible) <= source.keys():
        raise ValueError('invalid native eligible support')
    if sorted(panel['eligible_assay_ids']) != sorted(eligible):
        raise ValueError('eligible support mismatch')
    if (panel.get('n_families') != 32 or len(rows) != 32
            or sorted(rows) != sorted(panel['selected_assay_ids'])
            or not rows.keys() <= set(eligible)):
        raise ValueError('frozen 32-family panel support mismatch')
    family_ids = [row['cluster'] for row in rows.values()]
    if (len(set(family_ids)) != 32
            or sorted(family_ids, key=str) != sorted(panel['selected_family_ids'], key=str)):
        raise ValueError('one assay per selected family required')
    expected_families = {source[name]['cluster'] for name in eligible}
    if sorted(panel['eligible_family_ids'], key=str) != sorted(expected_families, key=str):
        raise ValueError('eligible family support mismatch')
    for name, row in rows.items():
        if row != source[name]:
            raise ValueError('panel assay differs from bound source cohort')
        ids = row['mutants']
        digest = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
        if (not 2 <= len(ids) <= 128 or len(set(ids)) != len(ids)
                or row['mutant_digest'] != digest
                or len(row['sequences']) != len(ids) or len(row['measured']) != len(ids)):
            raise ValueError('mutation IDs/order/full draw mismatch')
        if not np.isfinite(np.asarray(row['measured'], dtype=float)).all():
            raise ValueError('nonfinite experimental effects')
    return rows


def validate_mutation_rows(payload, expected_key, panel_rows, expected_support):
    if payload.get('combination') != expected_key:
        raise ValueError('mutation cell label mismatch')
    rows = unique_rows(payload['assays'])
    if sorted(rows) != sorted(expected_support):
        raise ValueError('mutation assay support mismatch')
    retained = {}
    for name, baseline in panel_rows.items():
        row = rows[name]
        if any(row.get(field) != baseline[field] for field in ('cluster', 'mutant_digest', 'measured')):
            raise ValueError('mutation IDs/family/effects mismatch')
        if 'mutants' in row and row['mutants'] != baseline['mutants']:
            raise ValueError('explicit mutation ID mismatch')
        scores = np.asarray(row['scores'], dtype=float)
        if row.get('n_variants') != len(baseline['mutants']) or scores.shape != (len(baseline['mutants']),) or not np.isfinite(scores).all():
            raise ValueError('invalid mutation score vector')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rho = float(spearmanr(scores, baseline['measured']).statistic)
        expected = rho if np.isfinite(rho) else None
        stored = row.get('spearman')
        if (expected is None) != (stored is None) or (expected is not None and
                (not np.isfinite(stored) or abs(stored - expected) > 1e-12)):
            raise ValueError('stored Spearman does not replay from vectors')
        retained[name] = {**row, 'spearman': expected}
    return retained


def validate_generation(rows, expected_key):
    if len(rows) != 64 or sorted(row['attempt_index'] for row in rows) != list(range(64)):
        raise ValueError('generation requires all 64 indexed attempts')
    for row in rows:
        if row.get('combination') != expected_key:
            raise ValueError('generation cell label mismatch')
        if 'batch_index' in row and row['batch_index'] != row['attempt_index'] // 8:
            raise ValueError('generation batch index mismatch')
        if type(row.get('any_profile_hit')) is not bool or row['any_profile_hit'] != bool(row['pfam_families']):
            raise ValueError('Pfam flags unavailable or inconsistent')
        if (row.get('residue_budget') != 400 or not row.get('generated_token_ids')
                or row['generated_tokens'] != len(row['generated_token_ids'])
                or row['generated_tokens'] > 1024):
            raise ValueError('generation budget/token trace mismatch')
        replay = decompose(row['raw_continuation'], 'prollama', residue_budget=400,
                           stop_reason=row['stop_reason'])
        if any(row.get(field) != value for field, value in replay.items()):
            raise ValueError('generation outcome does not replay')
    return rows


def analyze(cells, panel_rows):
    support = sorted(panel_rows)
    valid = [name for name in support if all(cells[key][0][name]['spearman'] is not None for key in KEYS)]
    if not valid:
        raise ValueError('no common defined mutation correlations')
    families = [panel_rows[name]['cluster'] for name in valid]
    outputs = {}
    for label, weights in CONTRASTS.items():
        differences = [sum(w * cells[key][0][name]['spearman'] for key, w in weights.items()) for name in valid]
        outputs[label] = dict(weights=weights,
            mutation_spearman_difference=cluster_bootstrap(differences, families, resamples=2000, seed=20260923),
            generation=paired_generation_contrast({key: cell[1] for key, cell in cells.items()}, weights))
    endpoints = {key: {
        'mutation_spearman': cluster_bootstrap([cells[key][0][name]['spearman'] for name in valid],
                                             families, resamples=2000, seed=20260923),
        'generation': paired_generation_contrast({key: cells[key][1]}, {key: 1.})}
        for key in KEYS}
    return dict(n_common_assays=len(valid), n_common_families=len(set(families)),
                common_assay_ids=valid, undefined_correlation_exclusions=[
                    dict(assay=name, cells=[key for key in KEYS if cells[key][0][name]['spearman'] is None])
                    for name in support if name not in valid],
                cell_endpoints=endpoints, contrasts=outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('panel', 'cohort', 'inventory', 'native-root', 'hybrid-root', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    hashes = {}
    panel = read_json(args.panel, hashes)
    cohort = read_json(args.cohort, hashes)
    inventory = read_json(args.inventory, hashes)
    require_refinement_inventory(inventory)
    # Prove unchanged embedding/head before native endpoints represent factorial corners.
    mapping = representatives(['000', '111'], inventory)
    if mapping != {k: ('000' if k[1] == '0' else '111') for k in mapping}:
        raise ValueError('inventory does not establish interface identity')
    native_path = args.native_root / 'intervention_manifest.json'
    native = read_json(native_path, hashes)
    compatibility = check_precision(native)
    if native.get('component_order') != ['embedding', 'body_including_final_norm', 'head'] or native.get('stage_index') != {'0': 'stage1', '1': 'stage2'}:
        raise ValueError('native component ordering mismatch')
    panel_rows = validate_panel(panel, cohort, native, cohort_hash=hashes[str(args.cohort)],
                               native_hash=hashes[str(native_path)], inventory_hash=hashes[str(args.inventory)])
    locations = {'00': (args.native_root / '000', '000', native['common_assays']),
                 '11': (args.native_root / '111', '111', native['common_assays'])}
    manifests = list(args.hybrid_root.rglob('attention_mlp_manifest.json'))
    if not manifests:
        raise ValueError('no hybrid manifests')
    for path in manifests:
        manifest = read_json(path, hashes)
        observed = check_precision(manifest)
        expected = dict(cohort_sha256=hashes[str(args.panel)], source_cohort_sha256=hashes[str(args.cohort)],
                        inventory_sha256=hashes[str(args.inventory)], native_manifest_sha256=hashes[str(native_path)],
                        component_order=['attention', 'mlp'], stage_index={'0': 'stage1', '1': 'stage2'},
                        generation_precision={'dtype': 'float32', 'tf32': False}, generation_policy=POLICY)
        if any(manifest.get(k) != v for k, v in expected.items()):
            raise ValueError('hybrid provenance/precision/policy mismatch')
        if sorted(manifest['common_assays']) != sorted(panel_rows):
            raise ValueError('hybrid manifest support mismatch')
        for field in ('tokenizer_sha256', 'endpoint_config_sha256'):
            if observed.get(field) != compatibility.get(field):
                raise ValueError('native/hybrid interface provenance mismatch')
        for key in ('01', '10'):
            cell = path.parent / key
            if (cell / 'mutation_scores.json').exists():
                if key in locations:
                    raise ValueError('duplicate hybrid cell')
                locations[key] = (cell, key, list(panel_rows))
    if set(locations) != set(KEYS):
        raise ValueError('missing factorial cells')
    cells, generation_sources = {}, []
    for key in KEYS:
        directory, original_key, support = locations[key]
        scores = read_json(directory / 'mutation_scores.json', hashes)
        rows = validate_mutation_rows(scores, original_key, panel_rows, support)
        path = directory / 'attempts.jsonl'
        hashes[str(path)] = sha256_file(path)
        attempts = validate_generation([json.loads(line) for line in path.read_text().splitlines()], original_key)
        summary = read_json(directory / 'summary.json', hashes)
        if summary['combination'] != original_key or summary['n_assays'] != len(support):
            raise ValueError('cell summary support mismatch')
        provenance = summary['generation_provenance']
        generation_sources.append({k: provenance[k] for k in ('runner_sha256', 'module_sha256', 'torch_version', 'sampling')})
        cells[key] = (rows, attempts)
    if any(source != generation_sources[0] for source in generation_sources[1:]):
        raise ValueError('native/hybrid generation implementation mismatch')
    result = analyze(cells, panel_rows)
    write_json(args.out / 'attention_mlp_contrasts.json', dict(
        schema_version='attention_mlp_analysis_v1', status='complete', input_sha256=hashes,
        component_order=['attention', 'mlp'], stage_index={'0': 'stage1', '1': 'stage2'},
        generation_policy=POLICY, generation_source=generation_sources[0], **result,
        uncertainty='Exploratory unadjusted percentile 95% intervals: paired family bootstrap for mutation; jointly resampled eight shared seed batches of eight attempts for generation. Only eight generation randomization units. Fixed checkpoint and selected panel; no training-seed or panel-selection uncertainty.',
        limitation='Marginal effects require the conditional effects and interaction for interpretation. Cross-stage incompatibility may damage behavior; a projection-group intervention does not identify a circuit or the training cause. Native endpoint recipe is inherited from the frozen coarse runner; observed FP32 loading, shared generation implementation hashes and complete token traces are checked here.'))


if __name__ == '__main__':
    main()
