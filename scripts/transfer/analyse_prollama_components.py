#!/usr/bin/env python3
"""Paired family-equal mutation and shared-seed-batch generation contrasts."""
from __future__ import annotations
import argparse
import itertools
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from src.transfer.io import write_json,sha256_file
from src.transfer.profiles import cluster_bootstrap
from src.transfer.component_statistics import paired_generation_contrast


KEYS=[''.join(x) for x in itertools.product('01',repeat=3)]


def representatives(observed, inventory=None):
    """Map factorial labels to observed models, only with exact tensor evidence."""
    observed=set(observed)
    if not observed <= set(KEYS):raise ValueError('invalid observed combination')
    if inventory is None:
        if observed != set(KEYS):raise ValueError('all eight combinations required without inventory')
        return {k:k for k in KEYS}
    order=['embedding','body_including_final_norm','head']
    if inventory.get('status')!='complete' or inventory.get('component_order')!=order:
        raise ValueError('incomplete or incompatible component inventory')
    if inventory.get('stage_index')!={'0':'stage1','1':'stage2'}:
        raise ValueError('inventory stage order mismatch')
    from scripts.transfer.inventory_prollama_components import component
    tensors=inventory['tensors']
    if len({r['name'] for r in tensors})!=len(tensors):raise ValueError('duplicate inventory tensor')
    if any(component(r['name'])!=r['component'] for r in tensors):raise ValueError('inventory component mismatch')
    equal=[]
    for name in order:
        rows=[r for r in tensors if r['component']==name]
        group=inventory['components'][name]
        if not rows or len(rows)!=group['n_tensors'] or sum(r['n_elements'] for r in rows)!=group['n_elements']:
            raise ValueError('inventory tensor counts mismatch')
        for r in rows:
            hashes=r['fp32_sha256']
            if len(hashes)!=2 or any(len(h)!=64 for h in hashes):raise ValueError('missing FP32 tensor digest')
            if hashes[0]==hashes[1] and (not r['exact_equal_fp32'] or r['differing_elements_fp32']!=0):
                raise ValueError('contradictory tensor equality evidence')
        same=all(r['fp32_sha256'][0]==r['fp32_sha256'][1] for r in rows)
        if same!=group['fp32_bytes_equal']:raise ValueError('contradictory component equality evidence')
        equal.append(same)
    classes={}
    for key in KEYS:
        canonical=''.join('0' if same else bit for bit,same in zip(key,equal))
        classes.setdefault(canonical,[]).append(key)
    expected=sorted(sorted(v) for v in classes.values())
    if expected!=sorted(sorted(v) for v in inventory['parameter_equivalent_combinations']):
        raise ValueError('inventory equivalence classes do not follow tensor evidence')
    mapping={}
    for group in classes.values():
        available=observed.intersection(group)
        if not available:raise ValueError(f'missing observed representative for {group}')
        selected=min(available,key=lambda k:(k not in ('000','111'),k))
        mapping.update({k:selected for k in group})
    return mapping


def aggregate_weights(weights, mapping):
    aggregated={}
    for key,weight in weights.items():
        representative=mapping[key]
        aggregated[representative]=aggregated.get(representative,0.)+weight
    return {k:v for k,v in sorted(aggregated.items()) if v!=0.}


def generation_contrast(cells, weights, rng):
    return paired_generation_contrast({key:cell[1] for key,cell in cells.items()}, weights)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--inventory',type=Path,help='completed exact checkpoint tensor inventory permitting equivalent-cell reuse')
    a=p.parse_args()
    cells={}; hashes={}
    for path in sorted(a.root.rglob('mutation_scores.json')):
        value=json.loads(path.read_text());key=value['combination']
        if key in cells:raise ValueError(f'duplicate combination {key}')
        rows={r['assay']:r for r in value['assays']}
        if len(rows)!=len(value['assays']):raise ValueError('duplicate assay')
        attempts=[json.loads(line) for line in path.with_name('attempts.jsonl').read_text().splitlines()]
        if not attempts or any(r.get('any_profile_hit') is None for r in attempts):raise ValueError('missing generation/profile outcomes')
        cells[key]=(rows,attempts);hashes[str(path)]=sha256_file(path)
        hashes[str(path.with_name('attempts.jsonl'))]=sha256_file(path.with_name('attempts.jsonl'))
    keys=KEYS
    inventory=json.loads(a.inventory.read_text()) if a.inventory else None
    mapping=representatives(cells,inventory)
    if a.inventory:hashes[str(a.inventory)]=sha256_file(a.inventory)
    baseline_key=mapping['000']
    support=sorted(cells[baseline_key][0])
    for key,(rows,_) in cells.items():
        if sorted(rows)!=support:raise ValueError('assay support mismatch')
        for name in support:
            baseline=cells[baseline_key][0][name]
            if any(rows[name][field]!=baseline[field] for field in ['mutant_digest','cluster','measured']):raise ValueError('variant/family/target mismatch')
    valid=[name for name in support if all(cells[key][0][name]['spearman'] is not None for key in set(mapping.values()))]
    if not valid:raise ValueError('no common defined assay correlations')
    weights={'native_stage2_minus_stage1':{'000':-1.,'111':1.}}
    for i,component in enumerate(['embedding','body','head']):
        weights[component+'_marginal']={key:(.25 if key[i]=='1' else -.25) for key in keys}
    # Effects conditional on the other components are needed to expose incompatibility.
    for i,component in enumerate(['embedding','body','head']):
        for key in keys:
            if key[i]=='0':
                changed=key[:i]+'1'+key[i+1:]
                weights[f'{component}_{changed}_minus_{key}']={key:-1.,changed:1.}
    rng=np.random.default_rng(20260923);results={}
    families=[cells[baseline_key][0][name]['cluster'] for name in valid]
    generation_cache={}
    for label,w in weights.items():
        aggregated=aggregate_weights(w,mapping)
        values=[sum(weight*cells[key][0][name]['spearman'] for key,weight in aggregated.items()) for name in valid]
        mutation=cluster_bootstrap(values,families,resamples=2000,seed=20260923)
        if not aggregated:
            mutation.update(point=0.,interval=[0.,0.],interval_status='exact_parameter_identity_not_sampling_inference')
        cache_key=tuple(aggregated.items())
        if cache_key not in generation_cache:
            generation_cache[cache_key]=generation_contrast(cells,aggregated,rng)
        results[label]={'weights':w,'observed_representative_weights':aggregated,
                       'interpretation':'exact no-op contrast; no evidence of zero causal sensitivity' if not aggregated else 'contrast between observed representatives',
                       'mutation_spearman_difference':mutation,'generation':generation_cache[cache_key]}
    write_json(a.out/'component_contrasts.json',{'input_sha256':hashes,'n_common_assays':len(valid),
        'inventory_sha256':sha256_file(a.inventory) if a.inventory else None,
        'observed_combinations':sorted(cells),'combination_to_observed_representative':mapping,
        'unused_equivalent_observed_combinations':sorted(set(cells)-set(mapping.values())),
        'undefined_correlation_exclusions':[name for name in support if name not in valid],
        'contrasts':results,'uncertainty':'Exploratory unadjusted 95% intervals; paired family bootstrap for mutation, paired seed-batch bootstrap across observed representatives for generation. The same eight seeds induce common randomness across cells; each batch contains eight attempts. Equivalent labels share one sample: weights are aggregated before resampling. No-op intervals are algebraic identities, not sampling inference. Fixed models; no training-seed uncertainty.','limitation':'Identical-component swaps are no interventions and do not measure causal sensitivity. Hybrid interface damage and component interactions prevent attributing a capability to one isolated component.'})

if __name__=='__main__':main()
