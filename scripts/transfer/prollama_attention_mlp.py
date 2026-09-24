#!/usr/bin/env python3
"""Two predeclared attention/MLP hybrids on a frozen family-balanced panel."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.transfer.io import write_json,sha256_file
from src.transfer.prollama_component_swap import validate_pair

GENERATION_POLICY=dict(attempts=64,residue_budget=400,token_safety_cap=1024,seed=20260923,
                       batch_size=8,temperature=.85,top_p=.95,top_k=50,repetition_penalty=1.)


def select_panel(cohort, native, *, seed=20260923):
    """Use only eligible assay and family identifiers to choose the panel."""
    import numpy as np
    if cohort['variants']!=128:raise ValueError('requires the original 128-variant draw')
    rows={r['assay']:r for r in cohort['assays']}
    if len(rows)!=len(cohort['assays']):raise ValueError('duplicate cohort assays')
    eligible=sorted(native['common_assays'])
    if len(set(eligible))!=len(eligible) or not set(eligible)<=set(rows):
        raise ValueError('invalid native eligible assay support')
    family_to_assays={}
    for name in eligible:
        family_to_assays.setdefault(rows[name]['cluster'],[]).append(name)
    families=sorted(family_to_assays,key=str)
    if len(families)<32:raise ValueError('requires at least 32 eligible families')
    rng=np.random.default_rng(seed)
    selected_families=sorted([families[int(i)] for i in rng.permutation(len(families))[:32]],key=str)
    selected=sorted(family_to_assays[f][int(rng.integers(len(family_to_assays[f])))] for f in selected_families)
    return dict(status='complete',schema_version='attention_mlp_panel_v1',seed=seed,n_families=32,
                selection={'method':'sort families by str; seeded NumPy permutation selects 32; sort selected families by str; seeded integer selects one assay from each sorted assay list; sort selected assay IDs',
                           'uses_outcomes':False,'preserves_original_variant_draw':True},
                eligible_assay_ids=eligible,eligible_family_ids=families,
                selected_assay_ids=selected,selected_family_ids=selected_families,
                assays=[rows[name] for name in selected])


def require_refinement_inventory(inventory):
    if inventory.get('status')!='complete':raise ValueError('incomplete inventory')
    for name in ('embedding','head'):
        if not inventory['components'][name]['fp32_bytes_equal']:
            raise ValueError('refinement requires identical embedding and head')
    for row in inventory['tensors']:
        name=row['name']
        changed_block=name.startswith('model.layers.') and ('.self_attn.' in name or '.mlp.' in name)
        if not changed_block and row['fp32_sha256'][0]!=row['fp32_sha256'][1]:
            raise ValueError(f'changed tensor outside attention/MLP: {name}')


def verify_checkpoint_sources(paths, inventory):
    """Bind the staged checkpoint files actually loaded to the exact inventory."""
    if len(paths)!=len(inventory['sources']):raise ValueError('inventory source count mismatch')
    for path,source in zip(paths,inventory['sources']):
        if path.resolve()!=Path(source['checkpoint']).resolve():
            raise ValueError('inventory/model checkpoint path mismatch')
        for name,expected in source['files'].items():
            if sha256_file(path/name)!=expected:
                raise ValueError(f'inventory/model checkpoint digest mismatch: {name}')


class AttentionMLPPair:
    """Retain original submodule references before mutating the Stage 1 shell."""
    def __init__(self,first,second):
        self.first=first;self.second=second
        self.receipt=validate_pair(first,second)
        self.parts=[[(layer.self_attn,layer.mlp) for layer in x.model.model.layers]
                    for x in (first,second)]

    def select(self,attention,mlp):
        if attention not in (0,1) or mlp not in (0,1):raise ValueError('stage indices must be 0/1')
        for i,layer in enumerate(self.first.model.model.layers):
            layer.self_attn=self.parts[attention][i][0]
            layer.mlp=self.parts[mlp][i][1]
        return replace(self.first,model=self.first.model)

    def verify_endpoints(self):
        import torch
        ids=self.first.tokenizer('Seq=<ACDEFGHIKLMNPQRSTVWY>',return_tensors='pt')['input_ids'].to(self.first.device)
        with torch.no_grad():
            native=[x.model(input_ids=ids,use_cache=False).logits.float().cpu() for x in (self.first,self.second)]
            errors=[]
            for i in (0,1):
                observed=self.select(i,i).model(input_ids=ids,use_cache=False).logits.float().cpu()
                error=float((observed-native[i]).abs().max())
                if error!=0.:raise RuntimeError(f'native reconstruction differs: {error}')
                errors.append(error)
        self.receipt['native_reconstruction_max_absolute_logit_error']=errors
        return self.receipt


def prepare(args):
    cohort=json.loads(args.cohort.read_text());native=json.loads(args.native_manifest.read_text())
    inventory=json.loads(args.inventory.read_text());require_refinement_inventory(inventory)
    source_hash=sha256_file(args.cohort)
    if native['cohort_sha256']!=source_hash:raise ValueError('native/cohort hash mismatch')
    panel=select_panel(cohort,native)
    panel.update(source_cohort_sha256=source_hash,native_manifest_sha256=sha256_file(args.native_manifest),
                 inventory_sha256=sha256_file(args.inventory))
    if args.panel.exists() and json.loads(args.panel.read_text())!=panel:
        raise ValueError('refuse to replace a different frozen panel')
    write_json(args.panel,panel)
    print(json.dumps(dict(panel_sha256=sha256_file(args.panel),n_assays=len(panel['assays']),
                          n_families=panel['n_families'],n_variants=sum(len(r['mutants']) for r in panel['assays']))))


def run(args):
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from src.transfer.joint_lineage import load_rung,rung,BareBlockScorer
    from src.transfer import generation_failure as gf,generation_evidence as ge,conditioned_generation as cg
    from scripts.transfer.generation_failure_confirmation import generate
    panel=json.loads(args.panel.read_text());inventory=json.loads(args.inventory.read_text())
    require_refinement_inventory(inventory)
    if panel['status']!='complete' or panel['inventory_sha256']!=sha256_file(args.inventory):
        raise ValueError('unbound or incomplete panel')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    verify_checkpoint_sources([rung(name).checkpoint for name in ('prollama-stage-1','prollama')],inventory)
    first=load_rung('prollama-stage-1',device=args.device,dtype='float32')
    second=load_rung('prollama',device=args.device,dtype='float32')
    pair=AttentionMLPPair(first,second);check=pair.verify_endpoints()
    scorer=BareBlockScorer(first,batch_size=1)
    for row in panel['assays']:
        if max(scorer.token_lengths(scorer.render([row['wildtype'],*row['sequences']])))>scorer.context:
            raise ValueError('frozen panel exceeds native context; do not redraw')
    manifest=dict(common_assays=panel['selected_assay_ids'],cohort_sha256=sha256_file(args.panel),
                  source_cohort_sha256=panel['source_cohort_sha256'],inventory_sha256=panel['inventory_sha256'],
                  native_manifest_sha256=panel['native_manifest_sha256'],compatibility=check,
                  component_order=['attention','mlp'],stage_index={'0':'stage1','1':'stage2'},
                  mutation_precision={'dtype':'float32','tf32':False,'batch_size':1},
                  generation_precision={'dtype':'float32','tf32':False},generation_policy=GENERATION_POLICY,
                  scoring='bare-block summed scored-token mutant-minus-WT',
                  runner_sha256=sha256_file(Path(__file__)))
    manifest['checkpoint_inventory_digests_verified']=True
    args.out.mkdir(parents=True,exist_ok=True)
    write_json(args.out/'attention_mlp_manifest.json',manifest)
    key=args.combination;out=args.out/key;out.mkdir(parents=True,exist_ok=True)
    loaded=pair.select(*(int(c) for c in key));scorer=BareBlockScorer(loaded,batch_size=1)
    assays=[]
    for row in panel['assays']:
        values=scorer.log_likelihood(scorer.render([row['wildtype'],*row['sequences']]))
        scores=values[1:]-values[0];rho=float(spearmanr(scores,row['measured']).statistic)
        assays.append(dict(assay=row['assay'],cluster=row['cluster'],scores=scores.tolist(),
                           measured=row['measured'],spearman=rho if np.isfinite(rho) else None,
                           n_variants=len(scores),mutant_digest=row['mutant_digest']))
    write_json(out/'mutation_scores.json',{'combination':key,'assays':assays,'accounting':scorer.residue_accounting()})
    params=SimpleNamespace(arm='prollama',device=args.device,out=out,
                           **{k:GENERATION_POLICY[k] for k in ('attempts','residue_budget','token_safety_cap','seed')})
    rows,provenance=generate(params,loaded=(loaded.model,loaded.tokenizer))
    hits,pfam=cg.annotate({str(i):r['sequence'] for i,r in enumerate(rows)},tool=SimpleNamespace(hmmscan=args.hmmscan),
                        database=SimpleNamespace(path=args.pfam_hmm),workspace=out/'pfam',threads=4,shards=2,
                        label='attention_mlp_'+key)
    for i,row in enumerate(rows):
        row['pfam_families']=ge._families(hits,str(i));row['any_profile_hit']=bool(row['pfam_families'])
        row['combination']=key;row['batch_index']=row['attempt_index']//8
    ge.write_immutable(out/'attempts.jsonl',ge.jsonl_bytes(rows))
    summary=dict(combination=key,generation=gf.summarize(rows),generation_provenance=provenance,pfam_receipt=pfam,
                 n_assays=len(assays),mean_assay_spearman=float(np.mean([r['spearman'] for r in assays if r['spearman'] is not None])))
    write_json(out/'summary.json',summary)
    write_json(args.out/'attention_mlp_summary.json',{'manifest':manifest,'combinations':{key:summary}})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=['prepare','run'])
    p.add_argument('--cohort',type=Path);p.add_argument('--native-manifest',type=Path)
    p.add_argument('--panel',required=True,type=Path);p.add_argument('--inventory',required=True,type=Path)
    p.add_argument('--combination',choices=['01','10']);p.add_argument('--out',type=Path)
    p.add_argument('--device',default='cuda:0');p.add_argument('--hmmscan',type=Path);p.add_argument('--pfam-hmm',type=Path)
    args=p.parse_args()
    needed=('cohort','native_manifest') if args.stage=='prepare' else ('combination','out','hmmscan','pfam_hmm')
    if any(getattr(args,k) is None for k in needed):p.error('missing stage-specific inputs: '+', '.join(needed))
    (prepare if args.stage=='prepare' else run)(args)


if __name__=='__main__':main()
