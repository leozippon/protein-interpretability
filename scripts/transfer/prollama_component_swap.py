#!/usr/bin/env python3
"""Stage 1/2 2x2x2 interventions on fixed mutation support and native generation."""
from __future__ import annotations
import argparse
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.transfer.joint_lineage import load_rung,BareBlockScorer
from src.transfer.prollama_component_swap import ComponentPair
from src.transfer.io import write_json,sha256_file
from src.transfer import generation_failure as gf, generation_evidence as ge, conditioned_generation as cg
from scripts.transfer.generation_failure_confirmation import generate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--hmmscan',required=True,type=Path)
    p.add_argument('--pfam-hmm',required=True,type=Path)
    p.add_argument('--combinations',default='000,111,001,010,011,100,101,110')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    from scipy.stats import spearmanr
    import numpy as np
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    first=load_rung('prollama-stage-1',device=a.device,dtype='float32')
    second=load_rung('prollama',device=a.device,dtype='float32')
    pair=ComponentPair(first,second)
    check=pair.verify_endpoints()
    cohort=json.loads(a.cohort.read_text())
    preflight=BareBlockScorer(first,batch_size=1)
    supported=[]; exclusions=[]
    for assay in cohort['assays']:
        rendered=preflight.render([assay['wildtype'],*assay['sequences']])
        maximum=max(preflight.token_lengths(rendered))
        if maximum>preflight.context:
            exclusions.append({'assay':assay['assay'],'max_tokens':maximum,'context':preflight.context,'reason':'target_exceeds_context'})
        else:supported.append(assay)
    if not supported:raise ValueError('no assays inside common context support')
    receipt={'common_assays':[r['assay'] for r in supported],'excluded_assays':exclusions,
             'mutation_precision':{'dtype':'float32','tf32':False,'batch_size':1,'batch_invariance':'single-example reference throughout'},
             'cohort_sha256':sha256_file(a.cohort),'compatibility':check,
             'component_order':['embedding','body_including_final_norm','head'],
             'stage_index':{'0':'stage1','1':'stage2'},'scoring':'bare-block summed scored-token mutant-minus-WT'}
    write_json(a.out/'intervention_manifest.json',receipt)
    summaries={}
    for key in a.combinations.split(','):
        if len(key)!=3 or any(c not in '01' for c in key):raise ValueError('invalid combination')
        out=a.out/key;out.mkdir(parents=True,exist_ok=True)
        loaded=pair.select(*(int(c) for c in key))
        scorer=BareBlockScorer(loaded,batch_size=1)
        assays=[]
        for assay in supported:
            seqs=[assay['wildtype'],*assay['sequences']]
            values=scorer.log_likelihood(scorer.render(seqs));scores=values[1:]-values[0]
            rho=float(spearmanr(scores,assay['measured']).statistic)
            assays.append({'assay':assay['assay'],'cluster':assay['cluster'],'scores':scores.tolist(),
                           'measured':assay['measured'],'spearman':rho if np.isfinite(rho) else None,
                           'n_variants':len(scores),'mutant_digest':assay['mutant_digest']})
        write_json(out/'mutation_scores.json',{'combination':key,'assays':assays,'accounting':scorer.residue_accounting()})
        params=SimpleNamespace(arm='prollama',device=a.device,attempts=64,residue_budget=400,
                               token_safety_cap=1024,seed=20260923,out=out)
        rows,provenance=generate(params,loaded=(loaded.model,loaded.tokenizer))
        hits,pfam=cg.annotate({str(i):r['sequence'] for i,r in enumerate(rows)},tool=SimpleNamespace(hmmscan=a.hmmscan),database=SimpleNamespace(path=a.pfam_hmm),workspace=out/'pfam',threads=4,shards=2,label='prollama_components_'+key)
        for i,r in enumerate(rows):
            r['pfam_families']=ge._families(hits,str(i));r['any_profile_hit']=bool(r['pfam_families']);r['combination']=key
        ge.write_immutable(out/'attempts.jsonl',ge.jsonl_bytes(rows))
        summary={'combination':key,'generation':gf.summarize(rows),'generation_provenance':provenance,'pfam_receipt':pfam,
                 'n_assays':len(assays),'mean_assay_spearman':float(np.mean([x['spearman'] for x in assays if x['spearman'] is not None]))}
        write_json(out/'summary.json',summary);summaries[key]=summary
        print(f'Finished combination {key}',flush=True)
    write_json(a.out/'component_swap_summary.json',{'manifest':receipt,'combinations':summaries})

if __name__=='__main__':main()
