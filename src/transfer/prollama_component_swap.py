"""Factorial component interventions with exact contemporaneous native controls."""
from __future__ import annotations
from dataclasses import replace
from .io import sha256_file


def validate_pair(first, second):
    """Require identical interfaces and independent input/output weight storage."""
    keys=('model_type','hidden_size','intermediate_size','num_hidden_layers',
          'num_attention_heads','num_key_value_heads','vocab_size','rms_norm_eps',
          'rope_theta','rope_scaling','max_position_embeddings','hidden_act','attention_bias','mlp_bias')
    mismatches={k:(getattr(first.model.config,k,None),getattr(second.model.config,k,None))
                for k in keys if getattr(first.model.config,k,None)!=getattr(second.model.config,k,None)}
    if mismatches: raise ValueError(f'architecture mismatch: {mismatches}')
    hashes=[sha256_file(x.rung.checkpoint/'tokenizer.model') for x in (first,second)]
    if hashes[0]!=hashes[1] or first.tokenizer.get_vocab()!=second.tokenizer.get_vocab():
        raise ValueError('tokenizer mismatch')
    for x in (first,second):
        if x.model.config.tie_word_embeddings or x.model.model.embed_tokens.weight.data_ptr()==x.model.lm_head.weight.data_ptr():
            raise ValueError('embedding/head intervention undefined for tied weights')
    return {'architecture_equal':True,'tokenizer_sha256':hashes[0],'untied_embeddings':True,
            'endpoint_loading_facts':[x.facts for x in (first,second)],
            'endpoint_config_sha256':[sha256_file(x.rung.checkpoint/'config.json') for x in (first,second)]}


class ComponentPair:
    """Use one complete model as shell; retained module refs avoid copying weights."""
    def __init__(self, first, second):
        self.first=first; self.second=second
        self.receipt=validate_pair(first,second)
        self.parts=[(x.model.model.embed_tokens,x.model.model.layers,x.model.model.norm,x.model.lm_head)
                    for x in (first,second)]

    def select(self, embedding: int, body: int, head: int):
        if any(i not in (0,1) for i in (embedding,body,head)):raise ValueError('component indices must be 0/1')
        shell=self.first.model
        shell.model.embed_tokens=self.parts[embedding][0]
        shell.model.layers=self.parts[body][1]
        shell.model.norm=self.parts[body][2]
        shell.lm_head=self.parts[head][3]
        return replace(self.first,model=shell)

    def verify_endpoints(self):
        import torch
        ids=self.first.tokenizer('Seq=<ACDEFGHIKLMNPQRSTVWY>',return_tensors='pt')['input_ids'].to(self.first.device)
        with torch.no_grad():
            native=[x.model(input_ids=ids,use_cache=False).logits.float().cpu() for x in (self.first,self.second)]
            errors=[]
            for i in (0,1):
                observed=self.select(i,i,i).model(input_ids=ids,use_cache=False).logits.float().cpu()
                error=float((observed-native[i]).abs().max())
                if error!=0.:raise RuntimeError(f'native reconstruction differs: {error}')
                errors.append(error)
        self.receipt['native_reconstruction_max_absolute_logit_error']=errors
        return self.receipt
