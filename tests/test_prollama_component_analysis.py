"""Equivalent factorial labels must not create independent generated samples."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from scripts.transfer.analyse_prollama_components import aggregate_weights, generation_contrast, representatives


def fixture_inventory():
    names=['model.embed_tokens.weight','model.layers.0.weight','lm_head.weight']
    components=['embedding','body_including_final_norm','head']
    rows=[]
    for i,(name,component) in enumerate(zip(names,components)):
        same=i!=1
        rows.append(dict(name=name,component=component,n_elements=1,
                         fp32_sha256=['a'*64,('a' if same else 'b')*64],
                         exact_equal_fp32=same,differing_elements_fp32=int(not same)))
    return dict(status='complete',component_order=components,stage_index={'0':'stage1','1':'stage2'},
                tensors=rows,components={r['component']:dict(n_tensors=1,n_elements=1,
                    fp32_bytes_equal=r['exact_equal_fp32']) for r in rows},
                parameter_equivalent_combinations=[['000','001','100','101'],['010','011','110','111']])


class ComponentAnalysisTests(unittest.TestCase):
    def test_representatives_require_full_exact_evidence(self):
        mapping=representatives(['000','111'],fixture_inventory())
        self.assertEqual(mapping['101'],'000')
        self.assertEqual(mapping['010'],'111')
        with self.assertRaisesRegex(ValueError,'all eight'):
            representatives(['000','111'])
        with self.assertRaisesRegex(ValueError,'missing observed'):
            representatives(['000'],fixture_inventory())
        bad=fixture_inventory();bad['tensors'][0]['differing_elements_fp32']=1
        with self.assertRaisesRegex(ValueError,'contradictory'):
            representatives(['000','111'],bad)
        bad=fixture_inventory();bad['parameter_equivalent_combinations']=[['000','111']]
        with self.assertRaisesRegex(ValueError,'equivalence classes'):
            representatives(['000','111'],bad)

    def test_noop_cancels_before_resampling(self):
        mapping=representatives(['000','111'],fixture_inventory())
        weights=aggregate_weights({'000':-1.,'100':1.},mapping)
        self.assertEqual(weights,{})
        rng=np.random.default_rng(6);state=copy.deepcopy(rng.bit_generator.state)
        result=generation_contrast({},weights,rng)
        self.assertEqual(rng.bit_generator.state,state)
        self.assertEqual(result['any_profile_hit']['paired_seed_batch_percentile_95'],[0.,0.])
        marginal={k:.25 if k[1]=='1' else -.25 for k in mapping}
        self.assertEqual(aggregate_weights(marginal,mapping),{'000':-1.,'111':1.})

    def test_cli_two_endpoints_yield_one_shared_body_contrast(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            inventory=root/'inventory.json';inventory.write_text(json.dumps(fixture_inventory()))
            for key,rho in [('000',.1),('111',.3)]:
                cell=root/key;cell.mkdir()
                rows=[dict(assay=str(i),cluster=i,mutant_digest='same',measured=[1,2,3],spearman=rho) for i in range(8)]
                (cell/'mutation_scores.json').write_text(json.dumps(dict(combination=key,assays=rows)))
                attempts=[dict(attempt_index=i,any_profile_hit=bool(i%2) if key=='000' else True,native_complete=True) for i in range(64)]
                (cell/'attempts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in attempts))
            subprocess.run([sys.executable,'scripts/transfer/analyse_prollama_components.py',
                            '--root',str(root),'--out',str(root/'out'),'--inventory',str(inventory)],
                           check=True,capture_output=True,text=True)
            result=json.loads((root/'out'/'component_contrasts.json').read_text())
            contrasts=result['contrasts']
            native=contrasts['native_stage2_minus_stage1']
            body=contrasts['body_marginal']
            self.assertEqual(body['generation'],native['generation'])
            self.assertEqual(body['mutation_spearman_difference'],native['mutation_spearman_difference'])
            self.assertEqual(contrasts['embedding_marginal']['mutation_spearman_difference']['interval'],[0.,0.])
            self.assertEqual(result['observed_combinations'],['000','111'])
            self.assertEqual(len(result['inventory_sha256']),64)


if __name__=='__main__':unittest.main()
