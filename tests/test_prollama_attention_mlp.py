"""Label-blind panel selection and exact native reconstruction for two hybrids."""
import copy
from dataclasses import dataclass
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from scripts.transfer.prollama_attention_mlp import AttentionMLPPair,select_panel,verify_checkpoint_sources
from src.transfer.io import sha256_file


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(tie_word_embeddings=False)
        self.model=nn.Module()
        self.model.embed_tokens=nn.Embedding(8,4)
        self.model.layers=nn.ModuleList([nn.Module() for _ in range(2)])
        for layer in self.model.layers:
            layer.self_attn=nn.Linear(4,4,bias=False)
            layer.mlp=nn.Linear(4,4,bias=False)
        self.lm_head=nn.Linear(4,8,bias=False)
    def forward(self,input_ids,use_cache=False):
        x=self.model.embed_tokens(input_ids)
        for layer in self.model.layers:x=layer.mlp(layer.self_attn(x))
        return SimpleNamespace(logits=self.lm_head(x))


class Tokenizer:
    def get_vocab(self):return {'A':1}
    def __call__(self,*args,**kwargs):return {'input_ids':torch.tensor([[1,2,3]])}


@dataclass
class Loaded:
    model: object
    tokenizer: object
    rung: object
    facts: dict
    device: str='cpu'


class AttentionMLPTests(unittest.TestCase):
    def test_checkpoint_digest_must_match_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'weights'
            path.write_bytes(b'fixed checkpoint')
            inventory={'sources':[{'checkpoint':str(root),'files':{'weights':sha256_file(path)}}]}
            verify_checkpoint_sources([root],inventory)
            path.write_bytes(b'changed checkpoint')
            with self.assertRaisesRegex(ValueError,'digest mismatch'):
                verify_checkpoint_sources([root],inventory)

    def test_panel_is_family_balanced_and_label_blind(self):
        rows=[dict(assay=f'{f:02d}_{a}',cluster=f,mutants=['A1C'],measured=[f+a])
              for f in range(40) for a in range(2)]
        cohort=dict(variants=128,assays=rows)
        native={'common_assays':[r['assay'] for r in rows]}
        panel=select_panel(cohort,native)
        changed=copy.deepcopy(cohort)
        for row in changed['assays']:row['measured']=[-999]
        second=select_panel(changed,native)
        self.assertEqual(panel['selected_assay_ids'],second['selected_assay_ids'])
        self.assertEqual(len(panel['assays']),32)
        self.assertEqual(len({r['cluster'] for r in panel['assays']}),32)
        self.assertTrue(all(r is rows[rows.index(r)] for r in panel['assays']))
        with self.assertRaisesRegex(ValueError,'32 eligible'):
            select_panel(cohort,{'common_assays':[r['assay'] for r in rows[:10]]})

    def test_native_reconstruction_and_switching_preserve_sources(self):
        torch.manual_seed(4)
        first=TinyModel();second=copy.deepcopy(first)
        with torch.no_grad():
            for layer in second.model.layers:
                layer.self_attn.weight.add_(.2);layer.mlp.weight.sub_(.1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'tokenizer.model').write_bytes(b'fixed tokenizer')
            (root/'config.json').write_text('{}')
            loaded=[Loaded(m,Tokenizer(),SimpleNamespace(checkpoint=root),{}) for m in (first,second)]
            original=[[ (layer.self_attn,layer.mlp) for layer in m.model.layers] for m in (first,second)]
            pair=AttentionMLPPair(*loaded)
            self.assertEqual(pair.verify_endpoints()['native_reconstruction_max_absolute_logit_error'],[0.,0.])
            for attention,mlp in [(0,1),(1,0),(0,0),(1,1)]:
                selected=pair.select(attention,mlp)
                for i,layer in enumerate(selected.model.model.layers):
                    self.assertIs(layer.self_attn,original[attention][i][0])
                    self.assertIs(layer.mlp,original[mlp][i][1])
            for i,layer in enumerate(second.model.layers):
                self.assertIs(layer.self_attn,original[1][i][0])
                self.assertIs(layer.mlp,original[1][i][1])


if __name__=='__main__':unittest.main()
