"""Synthetic factorial checks: provenance, fixed support and shared randomness."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from scripts.transfer.analyse_prollama_attention_mlp import (
    analyze, check_precision, validate_mutation_rows, validate_panel, POLICY,
)
from src.transfer.component_statistics import paired_generation_contrast
from src.transfer.generation_failure import decompose
from src.transfer.io import sha256_file
from scripts.transfer.prollama_attention_mlp import select_panel


def assay(i, n=128):
    ids = [f'A{j+1}V' for j in range(n)]
    return dict(assay=f'assay{i:02}', cluster=i, mutants=ids,
                mutant_digest=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
                wildtype='A' * n, sequences=['V' + 'A' * (n-1)] * n,
                measured=list(range(n)))


def mutation(row, key):
    n = len(row['mutants'])
    scores = list(range(n)) if key in ('000', '00', '10') else list(range(n-1, -1, -1))
    return dict(assay=row['assay'], cluster=row['cluster'], mutant_digest=row['mutant_digest'],
                measured=row['measured'], n_variants=n, scores=scores,
                spearman=1. if scores[0] == 0 else -1.)


def attempts(key):
    rows = []
    for i in range(64):
        row = decompose('ACD>', 'prollama', residue_budget=400, stop_reason='native_terminal')
        row.update(attempt_index=i, combination=key, generated_token_ids=[1, 2], generated_tokens=2,
                   pfam_families=['PF1'] if (i // 8) % 2 else [], any_profile_hit=bool((i // 8) % 2))
        rows.append(row)
    return rows


class AttentionMLPAnalysisTests(unittest.TestCase):
    def test_shared_batch_bootstrap_preserves_common_random_numbers(self):
        rows = attempts('00')
        result = paired_generation_contrast({'00': rows, '11': copy.deepcopy(rows)}, {'00': -1., '11': 1.})
        self.assertEqual(result['any_profile_hit']['paired_seed_batch_percentile_95'], [0., 0.])
        # Independent arm/attempt resampling would fabricate uncertainty here.
        single = paired_generation_contrast({'00': rows}, {'00': 1.})['any_profile_hit']
        self.assertEqual(single['n_seed_batches'], 8)
        self.assertGreater(single['paired_seed_batch_percentile_95'][1] - single['paired_seed_batch_percentile_95'][0], .5)
        bad = copy.deepcopy(rows); bad[-1]['attempt_index'] = 0
        with self.assertRaisesRegex(ValueError, 'indices'):
            paired_generation_contrast({'00': bad}, {'00': 1.})

    def test_fixed_mutant_support_replays_short_full_draw_and_rejects_drift(self):
        row = assay(0, n=63)
        payload = dict(combination='00', assays=[mutation(row, '00')])
        validate_mutation_rows(payload, '00', {row['assay']: row}, [row['assay']])
        for field, value in [('measured', [0] * 63), ('mutant_digest', 'bad'), ('spearman', .5)]:
            bad = copy.deepcopy(payload); bad['assays'][0][field] = value
            with self.assertRaises(ValueError):
                validate_mutation_rows(bad, '00', {row['assay']: row}, [row['assay']])

    def test_factorial_interaction_and_undefined_support(self):
        rows = {r['assay']: r for r in [assay(i, 16) for i in range(8)]}
        cells = {}
        # Fully synthetic endpoint values make the difference-in-differences exact.
        for key, rho in [('00', 0.), ('01', .1), ('10', .2), ('11', .6)]:
            cells[key] = ({name: dict(spearman=rho) for name in rows}, attempts(key))
        result = analyze(cells, rows)
        interaction = result['contrasts']['attention_by_mlp_interaction']['mutation_spearman_difference']
        self.assertAlmostEqual(interaction['point'], .3)
        self.assertEqual(result['n_common_families'], 8)
        cells['01'][0]['assay00']['spearman'] = None
        result = analyze(cells, rows)
        self.assertEqual(result['undefined_correlation_exclusions'], [dict(assay='assay00', cells=['01'])])
        self.assertIsNone(result['contrasts']['attention_marginal']['mutation_spearman_difference']['interval'])

    def test_cli_provenance_and_complete_small_factorial(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def save(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
                return sha256_file(path)
            rows = [assay(i, 63 if i == 0 else 128) for i in range(32)]
            cohort = root / 'cohort.json'; cohort_hash = save(cohort, dict(assays=rows, variants=128))
            tensor_rows = [dict(name=name, component=component, n_elements=1,
                               fp32_sha256=['a'*64, ('b' if i == 1 else 'a')*64],
                               exact_equal_fp32=i != 1, differing_elements_fp32=int(i == 1))
                           for i, (name, component) in enumerate(zip(
                               ['model.embed_tokens.weight', 'model.layers.0.self_attn.q_proj.weight', 'lm_head.weight'],
                               ['embedding', 'body_including_final_norm', 'head']))]
            inventory = root / 'inventory.json'
            inventory_hash = save(inventory, dict(status='complete', component_order=[r['component'] for r in tensor_rows],
                stage_index={'0': 'stage1', '1': 'stage2'}, tensors=tensor_rows,
                components={r['component']: dict(n_tensors=1, n_elements=1, fp32_bytes_equal=r['exact_equal_fp32']) for r in tensor_rows},
                parameter_equivalent_combinations=[['000', '001', '100', '101'], ['010', '011', '110', '111']]))
            compatibility = dict(architecture_equal=True, untied_embeddings=True,
                                 native_reconstruction_max_absolute_logit_error=[0., 0.],
                                 endpoint_loading_facts=[dict(dtype_requested='float32', dtype_observed=['float32'])]*2,
                                 tokenizer_sha256='c'*64, endpoint_config_sha256=['d'*64, 'e'*64])
            native = dict(cohort_sha256=cohort_hash, common_assays=[r['assay'] for r in rows],
                          mutation_precision=dict(dtype='float32', tf32=False, batch_size=1),
                          compatibility=compatibility, component_order=['embedding', 'body_including_final_norm', 'head'],
                          stage_index={'0': 'stage1', '1': 'stage2'})
            native_root = root / 'native'
            native_hash = save(native_root / 'intervention_manifest.json', native)
            panel_path = root / 'panel.json'
            panel = select_panel(dict(assays=rows, variants=128), native)
            panel.update(source_cohort_sha256=cohort_hash, native_manifest_sha256=native_hash,
                         inventory_sha256=inventory_hash)
            panel_hash = save(panel_path, panel)
            bad_panel = copy.deepcopy(panel); bad_panel['assays'][0]['measured'][0] = -1
            with self.assertRaisesRegex(ValueError, 'source cohort'):
                validate_panel(bad_panel, dict(assays=rows, variants=128), native, cohort_hash=cohort_hash,
                               native_hash=native_hash, inventory_hash=inventory_hash)
            bad_native = copy.deepcopy(native); bad_native['mutation_precision']['dtype'] = 'bfloat16'
            with self.assertRaisesRegex(ValueError, 'precision'):
                check_precision(bad_native)
            hybrid_root = root / 'hybrids'
            for key in ('000', '111', '01', '10'):
                directory = (native_root / key) if len(key) == 3 else (hybrid_root / key / key)
                if len(key) == 2:
                    manifest = {**native, 'cohort_sha256': panel_hash, 'source_cohort_sha256': cohort_hash,
                                'native_manifest_sha256': native_hash, 'inventory_sha256': inventory_hash,
                                'component_order': ['attention', 'mlp'],
                                'generation_precision': dict(dtype='float32', tf32=False), 'generation_policy': POLICY}
                    save(directory.parent / 'attention_mlp_manifest.json', manifest)
                save(directory / 'mutation_scores.json', dict(combination=key, assays=[mutation(row, key) for row in rows]))
                (directory / 'attempts.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in attempts(key)))
                save(directory / 'summary.json', dict(combination=key, n_assays=32,
                     generation_provenance=dict(runner_sha256='f'*64, module_sha256='a'*64, torch_version='test', sampling='fixed')))
            command = [sys.executable, 'scripts/transfer/analyse_prollama_attention_mlp.py',
                       '--panel', str(panel_path), '--cohort', str(cohort), '--inventory', str(inventory),
                       '--native-root', str(native_root), '--hybrid-root', str(hybrid_root), '--out', str(root / 'out')]
            subprocess.run(command, check=True, capture_output=True, text=True)
            result = json.loads((root / 'out' / 'attention_mlp_contrasts.json').read_text())
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['n_common_assays'], 32)
            self.assertEqual(len(result['contrasts']), 8)
            # A same-shaped but wrong-source panel must fail before any result.
            panel['source_cohort_sha256'] = '0'*64; save(panel_path, panel)
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('source digest mismatch', failed.stderr)


if __name__ == '__main__':
    unittest.main()
