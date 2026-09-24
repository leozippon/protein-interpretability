"""Small checkpoint fixtures for exact component-intervention identity."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from scripts.transfer.inventory_prollama_components import inventory


class ComponentInventoryTests(unittest.TestCase):
    def checkpoints(self, root):
        state = {'model.embed_tokens.weight': torch.arange(6).reshape(3,2).float(),
                 'model.layers.0.mlp.weight': torch.ones(2,2),
                 'model.norm.weight': torch.ones(2),
                 'lm_head.weight': torch.arange(6).reshape(3,2).float()}
        paths = [root/'stage1', root/'stage2']
        for i,path in enumerate(paths):
            path.mkdir()
            values = {k:v.clone() for k,v in state.items()}
            if i: values['model.layers.0.mlp.weight'][0,0] = 2
            save_file(values, str(path/'model.safetensors'))
            (path/'model.safetensors.index.json').write_text(json.dumps(
                {'weight_map':{k:'model.safetensors' for k in values}}))
        return paths

    def test_identical_embedding_head_reduce_to_two_parameter_models(self):
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            paths=self.checkpoints(Path(directory))
            result=inventory(*paths,chunk_elements=2)
        self.assertEqual(result['parameter_equivalent_combinations'],
                         [['000','001','100','101'],['010','011','110','111']])
        body=result['components']['body_including_final_norm']
        self.assertEqual((body['n_tensors'],body['n_elements'],body['changed_tensors_fp32'],body['differing_elements_fp32']),
                         (2,6,1,1))
        self.assertTrue(result['components']['head']['exact_equal_fp32'])
        self.assertTrue(result['components']['embedding']['native_bytes_equal'])
        self.assertEqual(len(result['sources'][0]['files']),2)
        self.assertEqual(result['components']['head']['component_sha256'][0],
                         result['components']['head']['component_sha256'][1])

    def test_unclassified_tensor_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=self.checkpoints(Path(directory))
            for path in paths:
                save_file({'other.weight':torch.ones(1)},str(path/'other.safetensors'))
                index=path/'model.safetensors.index.json'
                value=json.loads(index.read_text())
                value['weight_map']['other.weight']='other.safetensors'
                index.write_text(json.dumps(value))
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'unclassified'):
                inventory(*paths)


if __name__=='__main__':unittest.main()
