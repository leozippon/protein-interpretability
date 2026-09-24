#!/usr/bin/env python3
"""Exact CPU tensor inventory for the two staged ProLLaMA checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.transfer.io import sha256_file, write_json


def component(name):
    if name.startswith('model.embed_tokens.'):
        return 'embedding'
    if name.startswith(('model.layers.', 'model.norm.')):
        return 'body_including_final_norm'
    if name.startswith('lm_head.'):
        return 'head'
    raise ValueError(f'unclassified checkpoint tensor: {name}')


def inventory(first: Path, second: Path, *, chunk_elements: int = 1_048_576):
    import torch
    from safetensors import safe_open

    if chunk_elements < 1:
        raise ValueError('chunk size must be positive')
    torch.set_num_threads(4)
    roots = [first, second]
    indices = [json.loads((r/'model.safetensors.index.json').read_text())['weight_map'] for r in roots]
    if set(indices[0]) != set(indices[1]):
        raise ValueError('checkpoint tensor keys differ')
    names = sorted(indices[0])
    groups = {k: dict(n_tensors=0, n_elements=0, changed_tensors_fp32=0,
                     differing_elements_fp32=0, exact_equal_fp32=True,
                     native_bytes_equal=True, fp32_bytes_equal=True, dtype_pairs=[], tensor_names=[])
              for k in ('embedding', 'body_including_final_norm', 'head')}
    tensor_rows = []
    for name in names:
        group = groups[component(name)]
        tensors = []
        for root, index in zip(roots, indices):
            with safe_open(root/index[name], framework='pt', device='cpu') as handle:
                tensors.append(handle.get_tensor(name))
        a, b = tensors
        if a.shape != b.shape:
            raise ValueError(f'{name}: shape mismatch')
        if not (a.is_floating_point() and b.is_floating_point()):
            raise ValueError(f'{name}: expected floating-point model tensor')
        native_hashes = [hashlib.sha256(), hashlib.sha256()]
        fp32_hashes = [hashlib.sha256(), hashlib.sha256()]
        different = 0
        flat = [x.reshape(-1) for x in tensors]
        for start in range(0, a.numel(), chunk_elements):
            chunks = [x[start:start+chunk_elements].contiguous() for x in flat]
            fp32 = [x.float() for x in chunks]
            if not all(torch.isfinite(x).all().item() for x in fp32):
                raise ValueError(f'{name}: nonfinite FP32 checkpoint values')
            different += int(torch.count_nonzero(fp32[0] != fp32[1]).item())
            for i in range(2):
                native_hashes[i].update(chunks[i].view(torch.uint8).numpy().tobytes())
                fp32_hashes[i].update(fp32[i].view(torch.uint8).numpy().tobytes())
        native = [h.hexdigest() for h in native_hashes]
        fp32_digest = [h.hexdigest() for h in fp32_hashes]
        dtypes = [str(x.dtype) for x in tensors]
        byte_equal = dtypes[0] == dtypes[1] and native[0] == native[1]
        row = dict(name=name, component=component(name), shape=list(a.shape), n_elements=a.numel(),
                   dtypes=dtypes, native_sha256=native, fp32_sha256=fp32_digest,
                   native_bytes_equal=byte_equal, exact_equal_fp32=different == 0,
                   differing_elements_fp32=different)
        tensor_rows.append(row)
        group['n_tensors'] += 1
        group['n_elements'] += a.numel()
        group['changed_tensors_fp32'] += int(different > 0)
        group['differing_elements_fp32'] += different
        group['exact_equal_fp32'] &= different == 0
        group['native_bytes_equal'] &= byte_equal
        group['fp32_bytes_equal'] &= fp32_digest[0] == fp32_digest[1]
        if dtypes not in group['dtype_pairs']:
            group['dtype_pairs'].append(dtypes)
        group['tensor_names'].append(name)
        print(f'{name}: differing FP32 elements={different}/{a.numel()}', flush=True)
        del tensors, a, b, flat, chunks, fp32
    if any(not g['n_tensors'] for g in groups.values()):
        raise ValueError('missing intervention component')
    for key, group in groups.items():
        rows = [r for r in tensor_rows if r['component'] == key]
        # Aggregate digests bind names, shapes, dtypes and tensor digests.
        group['component_sha256'] = [hashlib.sha256(json.dumps(
            [(r['name'], r['shape'], r['dtypes'][i], r['native_sha256'][i]) for r in rows],
            separators=(',', ':')).encode()).hexdigest() for i in range(2)]
    equivalence = {}
    for combo in itertools.product('01', repeat=3):
        representative = ''.join('0' if g['fp32_bytes_equal'] else bit
                                 for bit, g in zip(combo, groups.values()))
        equivalence.setdefault(representative, []).append(''.join(combo))
    sources = []
    for root, index in zip(roots, indices):
        files = ['model.safetensors.index.json', *sorted(set(index.values()))]
        sources.append(dict(checkpoint=str(root), files={f: sha256_file(root/f) for f in files}))
    return dict(status='complete', component_order=list(groups), stage_index={'0':'stage1','1':'stage2'},
                comparison='exact elementwise equality after FP32 conversion, the intervention dtype; native byte equality reported separately',
                equivalence_rule='collapse only components with identical FP32 bytes, matching the intervention dtype; different storage locations alone are not interventions',
                limitation='An identical component swap is a no-op, not evidence of zero causal sensitivity; differing tensors do not establish a biological mechanism.',
                sources=sources, components=groups, tensors=tensor_rows,
                parameter_equivalent_combinations=list(equivalence.values()),
                script_sha256=sha256_file(Path(__file__)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage1', required=True, type=Path)
    p.add_argument('--stage2', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--device', default='cpu', choices=['cpu'])
    args = p.parse_args()
    result = inventory(args.stage1, args.stage2)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out/'component_tensor_inventory.json', result)


if __name__ == '__main__':
    main()
