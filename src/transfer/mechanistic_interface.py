"""The Direction-1 to Direction-2 interface: one endpoint, and the refusals around it.

Why this exists. Direction 1 closed with two outcomes on one 33-arm panel that
do not line up. The **likelihood** boundary is perfectly separated by protein
pretraining, 14 of 14 inside and 0 of 19 outside. The **representation**
boundary is separated by nothing but architecture family, which is a lineage
label and therefore circular, and the ProGen2 ladder -- five rungs across 12 to
32 blocks, 1,024 to 4,096 width and 151M to 6.44B parameters, all resolved --
refutes any scale, depth, width or precision account of it. No well-matched pair
exists for the representation boundary, and transplant feasibility fails for all
six candidate pairs as a matter of record.

What this module is. Not an experiment: the interface a mechanism cell has to
pass through so that its result composes with the capability result rather than
sitting beside it. One endpoint -- the same within-assay Spearman on the same
201-assay anchor, the same folds, the same seeds, the same
wild-type-family-at-50%-identity resampling unit -- and six conditions enforced
as refusals rather than conventions:

* a **floor** that reconstructs the untouched recipient bit-identically and
  reproduces its admitted endpoint inside its admitted interval;
* a **ceiling** that reconstructs the donor bit-identically and does the same;
* **tensor-level provenance** for every transplanted group, with its parameter
  count, matched against the census;
* **support and fold identity** against the admitted fits;
* **no held-out label reaching any tuning step**;
* and a **compatibility census** that refuses a shape mismatch rather than
  resizing it.

What the design licenses, stated here because it belongs with the machinery. A
transplant that recovers a gain shows those parameter values are **sufficient in
this context** -- against this recipient, on this support, under this scoring. It
does not show they are necessary, and necessity is a separate cell, the
complement. It does not identify a circuit and it does not say what the
computation is. And because a parent and a continued-pretraining stage differ by
a whole training run -- a corpus, a schedule, a data order and a token budget at
once -- any localisation is of **where that training's effect lands in the
parameters**, not of an architectural locus.

Imports are lazy where they would pull the fitting stack in, so the compatibility
census runs in an environment that has torch and safetensors and nothing else.
The partition's depth-quarter convention and the tensor digest are imported from
:mod:`src.transfer.capability_transplant` rather than restated.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from src.transfer.capability_transplant import (
    CHUNK_ELEMENTS, CONFIG_FIELD_ALIASES, DEPTH_BLOCKS, compare_configs,
    depth_block, fp32_digest, resolve_config_field, support_digest)
from src.transfer.io import sha256_file

# --------------------------------------------------------------------------- #
# The endpoint contract.
# --------------------------------------------------------------------------- #

#: The frozen anchor every mechanism cell is measured on, transcribed from
#: docs/D1_MATCHED_PAIR_HANDOFF.md, which owns it. Transcription rather than
#: import because the anchor's counts live in a report rather than in a module
#: constant; the digest is what binds them, and a cell whose support does not
#: hash to it is refused.
ANCHOR: dict[str, Any] = {
    'assays': 201,
    'wildtype_clusters': 163,
    'variants': 25_728,
    'cohort_sha256': '4093ac34368cd9e7be02e5c84a22655ce78cb9bb4e4d58d626f281868dbe7992',
    'resampling_unit': 'wild-type family at 50% identity',
    'grouping': 'single-linkage at 50% identity with 80% coverage',
    'binding_control': 'C+P+wall',
    'binding_control_spearman_by_seed': {'20260923': 0.47955, '20260924': 0.47906,
                                         '20260925': 0.48290},
}

#: Admitted endpoints on the anchor, per arm, as point and 95% interval in
#: within-assay Spearman of the native mutant-minus-wild-type summed target log
#: likelihood. Transcribed from docs/D1_MATCHED_PAIR_HANDOFF.md. These are the
#: scales a floor and a ceiling have to reproduce; a recovered fraction read
#: against a scale that does not reconstruct the admitted models means nothing.
ADMITTED_ANCHOR_LIKELIHOOD: dict[str, dict[str, Any]] = {
    'galactica-1.3b': {'point': 0.04754, 'interval': [0.02567, 0.06915]},
    'instructprotein': {'point': 0.28372, 'interval': [0.25509, 0.31163]},
    'gpt2-large': {'point': -0.01597, 'interval': [-0.03564, 0.00493]},
    'protgpt2': {'point': 0.33865, 'interval': [0.30871, 0.36565]},
    'galactica-6.7b': {'point': 0.08444, 'interval': [0.06207, 0.10641]},
    'progen2-xlarge': {'point': 0.39206, 'interval': [0.36438, 0.41839]},
}

#: The ProLLaMA lineage's admitted endpoints are on its own 211-assay / 169-family
#: native support, not on the anchor. Carried so the difference is a refusal
#: rather than a footnote: a cell scored on 211 assays does not compose with an
#: anchor result, and :func:`check_support_identity` will say so.
ADMITTED_NATIVE_LIKELIHOOD: dict[str, dict[str, Any]] = {
    'llama-2-7b': {'point': -0.00740, 'interval': [-0.02713, 0.01226],
                   'support': 'native', 'assays': 211, 'families': 169},
    'prollama-stage-1': {'point': 0.14604, 'interval': [0.11814, 0.17435],
                         'support': 'native', 'assays': 211, 'families': 169},
    'prollama-stage-2': {'point': 0.15865, 'interval': [0.13170, 0.18655],
                         'support': 'native', 'assays': 211, 'families': 169},
}

#: Tolerance for the pipeline-identity control on a representation cell: the
#: admitted class refitted from the admitted retained states through new code
#: must agree with the admitted report to this absolute bound on the held-out
#: predictions. The admitted final admission receipt used the same figure, and
#: the depth sweep measured a whole-panel float32 product departing from the
#: per-assay one by 3.483e-08, above it -- which is why the reduction order is
#: part of the contract and not the host's business.
PIPELINE_IDENTITY_TOLERANCE = 1e-8


def endpoint_contract() -> dict[str, Any]:
    """The one endpoint every cell reports, assembled from the admitted modules.

    The fold counts, split seeds, ridge grid, projection contract and bootstrap
    are imported from the modules that declare them rather than restated, and
    the two modules that both declare the split seeds are required to agree --
    the transcription check the handoff's property assembler uses, for the same
    reason: a second copy of a constant is a second source of truth until
    something compares them.
    """
    from src.transfer import local_context, pairwise_epistasis as pe
    from src.transfer.readout_class_sweep import (
        PROJECTION_BLAS_THREADS, PROJECTION_DIM, PROJECTION_SEED)

    if tuple(local_context.SPLIT_SEEDS) != tuple(pe.SPLIT_SEEDS):
        raise ValueError(
            f'split seeds disagree between modules: {local_context.SPLIT_SEEDS} against '
            f'{pe.SPLIT_SEEDS}; the endpoint cannot be declared against two of them')
    if (PROJECTION_DIM, PROJECTION_SEED) != (pe.PROJECTION_DIM, pe.PROJECTION_SEED):
        raise ValueError('the projection contract disagrees between modules')
    return {
        'estimand': (
            'within-assay Spearman of the fitted predictor, or of the native '
            'mutant-minus-wild-type summed target log likelihood for a likelihood cell, '
            'against the measured effect; averaged within wild-type family at 50% '
            'identity and then equally over families'),
        'anchor': dict(ANCHOR),
        'split_seeds': list(pe.SPLIT_SEEDS),
        'outer_folds': pe.OUTER_SPLITS,
        'inner_folds': pe.INNER_SPLITS,
        'inner_seed_rule': 'outer split seed + 100 + zero-based outer fold index',
        'ridge_grid': list(pe.ALPHAS),
        'ridge_tie_rule': 'ties broken toward the stronger penalty',
        'projection': {'dim': PROJECTION_DIM, 'seed': PROJECTION_SEED,
                       'seeds_consumed': [PROJECTION_SEED + offset for offset in range(4)],
                       'blas_threads': PROJECTION_BLAS_THREADS},
        'bootstrap': {'draws': pe.BOOTSTRAP_DRAWS, 'seed': pe.BOOTSTRAP_SEED,
                      'unit': ANCHOR['resampling_unit'], 'alpha': 0.05},
        'weights': ('equal per wild-type cluster, per assay within a cluster and per '
                    'variant within an assay'),
        'target': 'within-assay standardized average ranks',
        'scaling': 'training-only weighted feature means and scales; unpenalized intercept',
        'pipeline_identity_tolerance': PIPELINE_IDENTITY_TOLERANCE,
        'row_identity_rendering': (
            'every target rendered through float() before repr(), because '
            'repr(np.float64(0.5)) differs between NumPy 1.26 and NumPy 2 and a '
            'NumPy-scalar rendering would hash identical rows to different digests'),
    }


# --------------------------------------------------------------------------- #
# The refusals.
# --------------------------------------------------------------------------- #

def row_identity_digest(groups: Sequence, keys: Sequence, targets: Sequence) -> str:
    """Digest of the evaluated rows: group, key and target, in order.

    The target is rendered through ``float`` before ``repr`` rather than left as
    a NumPy scalar, for the reason the admitted pairwise and external-confirmation
    spellings give: ``repr(np.float64(0.5))`` is ``0.5`` under NumPy 1.26 and
    ``np.float64(0.5)`` under NumPy 2, so a NumPy-scalar rendering hashes
    identical data to different values across interpreters and makes the check
    unreliable in both directions.
    """
    if not (len(groups) == len(keys) == len(targets)):
        raise ValueError('row identity needs aligned group, key and target sequences')
    return hashlib.sha256('\n'.join(
        f'{group}|{key}|{float(target)!r}'
        for group, key, target in zip(groups, keys, targets)).encode()).hexdigest()


def fold_identity_digest(membership: Sequence[Sequence]) -> str:
    """Digest of a fold map's held-group membership, in fold order.

    One spelling, declared here: each fold's members sorted as strings, the fold
    order preserved, canonical JSON. The admitted fits digest the same content
    under their own group vocabulary -- ``held_groups`` in the pairwise and
    global-context fits, ``held_families`` in the retrieval-strata analysis --
    so a caller hands this function the membership lists and the digest is
    comparable by construction rather than by two spellings agreeing.
    """
    if not membership:
        raise ValueError('a fold identity needs at least one fold')
    return hashlib.sha256(json.dumps(
        [sorted(map(str, fold)) for fold in membership], sort_keys=True).encode()).hexdigest()


def check_support_identity(assay_ids: Iterable[str], *, expected_digest: str,
                           expected_count: int = ANCHOR['assays']) -> dict[str, Any]:
    """Refuse a cell whose scored support is not the anchor's, assay for assay."""
    identifiers = sorted(assay_ids)
    observed = support_digest(identifiers)
    if observed != expected_digest or len(identifiers) != expected_count:
        raise ValueError(
            f'support digest {observed} over {len(identifiers)} assays against the expected '
            f'{expected_digest} over {expected_count}; a cell scored on another support '
            'does not compose with the capability result it would be read against')
    return {'assays': len(identifiers), 'support_sha256': observed}


def check_fold_identity(membership: Sequence[Sequence], *, expected_digest: str,
                        expected_folds: int) -> dict[str, Any]:
    """Refuse a cell whose outer fold map is not the admitted one."""
    if len(membership) != expected_folds:
        raise ValueError(f'{len(membership)} outer folds against the admitted {expected_folds}')
    observed = fold_identity_digest(membership)
    if observed != expected_digest:
        raise ValueError(
            f'fold identity {observed} against the admitted {expected_digest}; an increment '
            'measured on other folds is not comparable with the admitted increment')
    return {'folds': len(membership), 'fold_identity_sha256': observed}


def check_tuning_isolation(folds: Sequence[Mapping[str, Sequence]]) -> dict[str, Any]:
    """Refuse a fit in which a held-out group reached a tuning step.

    The admitted nested fit partitions the *training* groups for penalty
    selection, so the held groups of an outer fold appear in none of its inner
    folds. Checked on membership rather than trusted from the construction,
    because a refit that rebuilt the inner partition over all groups would
    select its penalty against the labels it is about to be scored on and would
    still produce a plausible number.
    """
    if not folds:
        raise ValueError('tuning isolation needs at least one outer fold')
    report = []
    for index, fold in enumerate(folds):
        held = {str(group) for group in fold['held_groups']}
        inner = [{str(group) for group in members} for members in fold['inner_folds']]
        if not inner:
            raise ValueError(f'outer fold {index} declares no inner folds')
        leaked = sorted(held & set().union(*inner))
        if leaked:
            raise ValueError(
                f'outer fold {index}: held-out groups {leaked} appear in the inner '
                'partition, so a held-out label reached penalty selection')
        overlap = sorted(set.intersection(*inner)) if len(inner) > 1 else []
        if overlap:
            raise ValueError(
                f'outer fold {index}: groups {overlap} appear in every inner fold, so the '
                'inner partition is not a partition')
        report.append({'fold': index, 'held_groups': len(held),
                       'inner_folds': [len(members) for members in inner]})
    return {'outer_folds': len(folds), 'per_fold': report, 'held_labels_reach_tuning': False}


def group_provenance(census: Mapping[str, Any], selected: Sequence[str]) -> dict[str, Any]:
    """Tensor-level provenance for the transplanted groups, with parameter counts.

    Every group names its tensors, its parameter count and the digest each of
    its tensors must carry after the copy. A localisation claim is only as good
    as the record of what was moved, so the record is per tensor and the counts
    are sums of the census's own element counts rather than a second derivation.
    """
    chosen = set(selected)
    groups = census['groups']
    undeclared = sorted(chosen - set(groups))
    if undeclared:
        raise ValueError(f'undeclared groups in the selection: {undeclared}')
    refused = sorted(name for name in chosen if not groups[name].get('transplantable', True))
    if refused:
        raise ValueError(
            f'groups {refused} are not shape-compatible across the two checkpoints; a '
            'transplant refuses rather than resizing a tensor')
    identical = sorted(name for name in chosen if groups[name]['fp32_bytes_equal'])
    if identical:
        raise ValueError(
            f'groups {identical} are byte-identical across the two checkpoints: '
            'transplanting them is an algebraic no-op, and a zero contrast from an '
            'identity is not evidence that the group is insensitive')
    by_name = {row['name']: row for row in census['tensors']}
    detail = {}
    for name in sorted(chosen):
        tensors = groups[name]['tensor_names']
        detail[name] = {
            'tensors': list(tensors),
            'parameters': sum(by_name[tensor]['n_elements'] for tensor in tensors),
            'differing_elements_fp32': sum(
                by_name[tensor]['differing_elements_fp32'] for tensor in tensors),
            'source_fp32_sha256': {tensor: by_name[tensor]['fp32_sha256'][1]
                                   for tensor in tensors},
        }
    return {
        'selected_groups': sorted(chosen),
        'parameters_transplanted': sum(entry['parameters'] for entry in detail.values()),
        'tensors_transplanted': sum(len(entry['tensors']) for entry in detail.values()),
        'groups': detail,
    }


def check_reconstruction(observed: Mapping[str, str], census: Mapping[str, Any],
                         *, end: str) -> dict[str, Any]:
    """Refuse a floor or ceiling that does not reconstruct its checkpoint bit-identically.

    The floor is the empty selection and must come back as the recipient,
    tensor for tensor; the ceiling is the full selection and must come back as
    the donor. Exact digest equality rather than a tolerance, because both ends
    are copies of stored values and a forward pass is deterministic at batch
    size one: anything but equality means the partition did not cover the
    checkpoint, and every recovered fraction between the two ends would then be
    read against the wrong scale.
    """
    if end not in ('floor', 'ceiling'):
        raise ValueError(f'an endpoint is floor or ceiling, not {end!r}')
    column = 0 if end == 'floor' else 1
    expected = {row['name']: row['fp32_sha256'][column] for row in census['tensors']}
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    mismatched = sorted(name for name in set(expected) & set(observed)
                        if observed[name] != expected[name])
    if missing or extra or mismatched:
        raise ValueError(
            f'the {end} does not reconstruct its checkpoint bit-identically: '
            f'missing {missing}, unexpected {extra}, mismatched {mismatched}')
    return {'end': end, 'tensors': len(expected), 'bit_identical': True,
            'checkpoint': census['checkpoint_order'][column]}


def check_admitted_endpoint(observed: Mapping[str, Any],
                            admitted: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse an end whose endpoint falls outside its admitted interval.

    Exact reproduction is the expectation rather than the requirement: the
    transplanted model is bit-identical to the checkpoint it reconstructs, so a
    deterministic forward reproduces the admitted number. Interval containment
    is the gate because the admitted interval is what a later reader will
    compare against.
    """
    low, high = admitted['interval']
    inside = low <= observed['point'] <= high
    if not inside:
        raise ValueError(
            f'observed endpoint {observed["point"]} outside the admitted interval '
            f'[{low}, {high}]; no localisation number is computed against a scale that '
            'does not reconstruct the admitted model')
    return {'observed': observed['point'], 'admitted': admitted['point'],
            'admitted_interval': [low, high],
            'exact': observed['point'] == admitted['point'],
            'absolute_difference': abs(observed['point'] - admitted['point']),
            'within_admitted_interval': True}


def assert_cell_admitted(cell: Mapping[str, Any], census: Mapping[str, Any], *,
                         admitted: Mapping[str, Mapping[str, Any]],
                         support_expectation: str,
                         fold_expectation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Every refusal, in one call, before any localisation number is computed.

    A cell is a mapping carrying ``selected_groups`` (already resolved by the
    transplant runner, whose ``resolve_groups`` owns the ``complement:``
    spelling so that a necessity arm and a sufficiency arm are one code path),
    ``assays``, ``verified_tensor_fp32_sha256`` and ``endpoint``; a floor or
    ceiling cell additionally names its ``end``. Composition is the point: a
    caller that ran only some of these checks would publish a recovered
    fraction whose scale, support or fold map was never bound.
    """
    selected = sorted(cell['selected_groups'])
    undeclared = sorted(set(selected) - set(census['group_order']))
    if undeclared:
        raise ValueError(f'undeclared groups in the selection: {undeclared}')
    if len(set(selected)) != len(selected):
        raise ValueError('repeated group in the selection')
    record: dict[str, Any] = {
        'selected_groups': list(selected),
        'support': check_support_identity(cell['assays'], expected_digest=support_expectation),
        'endpoint_contract': endpoint_contract(),
    }
    end = cell.get('end')
    if end is not None:
        if (end == 'floor') != (not selected):
            raise ValueError('a floor selects no group and a ceiling selects every group')
        if end == 'ceiling' and sorted(selected) != sorted(census['group_order']):
            raise ValueError('a ceiling must select every declared group')
        record['reconstruction'] = check_reconstruction(
            cell['verified_tensor_fp32_sha256'], census, end=end)
        arm = census['checkpoint_order_arms'][0 if end == 'floor' else 1]
        record['admitted_comparison'] = check_admitted_endpoint(cell['endpoint'], admitted[arm])
    else:
        record['provenance'] = group_provenance(census, selected)
    if fold_expectation is not None:
        record['folds'] = check_fold_identity(
            fold_expectation['membership'], expected_digest=fold_expectation['digest'],
            expected_folds=fold_expectation['folds'])
        record['tuning_isolation'] = check_tuning_isolation(fold_expectation['per_fold'])
    record['licenses'] = LICENSE_STATEMENT
    return record


#: What a cell that passes every refusal may and may not be read as. Carried in
#: the record rather than in prose alone, so a reader of one artefact cannot
#: reach the recovered fraction without it.
LICENSE_STATEMENT: dict[str, str] = {
    'sufficiency': (
        'A group whose transplant recovers the gain shows those parameter values are '
        'sufficient in this context: against this recipient, on this support, under this '
        'scoring. It does not show they are necessary, which is the complement cell.'),
    'not_a_circuit': (
        'No cell identifies a circuit or says what the computation is. A recovered '
        'fraction is a parameter-space localisation, not a mechanism.'),
    'confound': (
        'Recipient and donor differ by a whole training run -- a corpus, a schedule, a '
        'data order and a token budget at once -- so any localisation is of where that '
        'training\'s effect lands in the parameters, not of an architectural locus.'),
    'batch_one': (
        'At batch size one the repeat check\'s reference forward and the production '
        'forward are the same single-row computation, so a zero drift is true by '
        'construction. A batch-one cell is internally consistent, not numerically '
        'validated.'),
}


# --------------------------------------------------------------------------- #
# The compatibility census, CPU-only.
# --------------------------------------------------------------------------- #

#: Fields whose disagreement makes a transplant undefined rather than merely
#: worth recording. A differing activation installs weights trained under one
#: nonlinearity into a body that applies another; a differing vocabulary makes
#: the embedding and the head shape-incompatible, which is a refusal at the
#: tensor level rather than at the configuration level and so is not blocking
#: here. Named in the shared semantic vocabulary of
#: :data:`capability_transplant.CONFIG_FIELD_ALIASES`, not in one family's key
#: names: comparing ``hidden_act`` alone is how a gelu-against-relu OPT pair read
#: as matching.
BLOCKING_FIELDS: tuple[str, ...] = ('model_type', 'hidden_width', 'mlp_width', 'depth',
                                    'attention_heads', 'activation')

#: Family-specific keys carried beside the semantic comparison because they
#: change what a block computes and no other family declares them, so they have
#: no semantic alias to resolve.
EXTRA_CONFIG_KEYS: tuple[str, ...] = ('word_embed_proj_dim', 'do_layer_norm_before',
                                      '_remove_final_layer_norm',
                                      'layer_norm_elementwise_affine')

#: Median per-tensor correlation above which two checkpoints' shape-compatible
#: body tensors cannot have been initialised independently. Two independent
#: initialisations of n elements correlate at order 1/sqrt(n), which is 2e-5 for
#: a 2048x2048 projection, so 0.1 is many orders of magnitude above the
#: independent case and far below what continued pretraining leaves: the
#: ProLLaMA pair changed 98.905% of its elements and remains a
#: continued-pretraining pair. The measured values are reported beside the
#: verdict, so the threshold is checkable rather than load-bearing on its own.
SHARED_INITIALISATION_CORRELATION = 0.1

_OPT_LAYER = re.compile(r'^(?:model\.)?decoder\.layers\.(\d+)\.(.+)$')
_OPT_ATTENTION = frozenset(
    f'self_attn.{key}_proj.{part}' for key in ('q', 'k', 'v', 'out') for part in ('weight', 'bias'))
_OPT_MLP = frozenset(f'fc{index}.{part}' for index in (1, 2) for part in ('weight', 'bias'))
_OPT_NORM = frozenset(f'{prefix}.{part}' for prefix in ('self_attn_layer_norm', 'final_layer_norm')
                      for part in ('weight', 'bias'))


def opt_group_of(name: str, n_layers: int) -> str:
    """The one declared group an OPT tensor belongs to, or a refusal.

    Refusing an unrecognised name is the load-bearing behaviour, as it is in the
    Llama partition: a tensor this function cannot place would be left out of
    every selection, and the full transplant would stop being the donor.
    """
    tail = name[len('model.'):] if name.startswith('model.') else name
    if tail in ('decoder.embed_tokens.weight',):
        return 'embedding'
    if tail in ('decoder.embed_positions.weight',):
        return 'positions'
    if name in ('lm_head.weight',):
        return 'head'
    if tail in ('decoder.final_layer_norm.weight', 'decoder.final_layer_norm.bias'):
        return 'norm'
    match = _OPT_LAYER.fullmatch(name)
    if match is None:
        raise ValueError(f'unclassified checkpoint tensor: {name}')
    layer, member = int(match.group(1)), match.group(2)
    quarter = depth_block(layer, n_layers) + 1
    if member in _OPT_NORM:
        return 'norm'
    if member in _OPT_ATTENTION:
        return f'attention_q{quarter}'
    if member in _OPT_MLP:
        return f'mlp_q{quarter}'
    raise ValueError(f'unclassified checkpoint tensor: {name}')


def opt_group_names() -> tuple[str, ...]:
    """Every declared OPT group, in a fixed order."""
    depth = tuple(f'{kind}_q{index + 1}' for kind in ('attention', 'mlp')
                  for index in range(DEPTH_BLOCKS))
    return ('embedding', 'positions', 'head', 'norm') + depth


def opt_partition(names: Iterable[str], n_layers: int) -> dict[str, list[str]]:
    """Group every name, and require the declared groups to be exactly covered."""
    result: dict[str, list[str]] = {group: [] for group in opt_group_names()}
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f'duplicate checkpoint tensor: {name}')
        seen.add(name)
        result[opt_group_of(name, n_layers)].append(name)
    empty = [group for group, members in result.items()
             if not members and group not in ('head',)]
    if empty:
        raise ValueError(f'declared groups absent from the checkpoint: {empty}')
    return {group: sorted(members) for group, members in result.items()}


def _load_state(checkpoint: Path):
    """Tensor names and a reader, for a sharded safetensors or a single .bin.

    Both formats are handled because the recommended pair is released in one
    each: the donor as sharded safetensors with an index and the recipient as a
    single pickled state dict. A census that supported only the index would
    report the pair as uncensusable, which is a statement about the loader.
    """
    import torch

    index = checkpoint / 'model.safetensors.index.json'
    if index.is_file():
        from safetensors import safe_open

        mapping = json.loads(index.read_text())['weight_map']

        def read(name: str):
            with safe_open(checkpoint / mapping[name], framework='pt', device='cpu') as handle:
                return handle.get_tensor(name)

        return sorted(mapping), read, sorted({str(value) for value in mapping.values()})
    single = checkpoint / 'model.safetensors'
    if single.is_file():
        from safetensors import safe_open

        with safe_open(single, framework='pt', device='cpu') as handle:
            names = sorted(handle.keys())

        def read(name: str):
            with safe_open(single, framework='pt', device='cpu') as handle:
                return handle.get_tensor(name)

        return names, read, ['model.safetensors']
    binary = checkpoint / 'pytorch_model.bin'
    if not binary.is_file():
        raise ValueError(f'{checkpoint}: no safetensors index, safetensors file or .bin')
    state = torch.load(binary, map_location='cpu', weights_only=True)
    return sorted(state), (lambda name: state[name]), ['pytorch_model.bin']


def config_comparison(destination: Path, source: Path, *,
                      blocking: Sequence[str] = BLOCKING_FIELDS,
                      extra_keys: Sequence[str] = EXTRA_CONFIG_KEYS) -> dict[str, Any]:
    """Field-by-field configuration comparison, recorded rather than raised on.

    The census's job is to report *why* a transplant is or is not defined, so a
    mismatch is a finding here and :func:`assert_pair_transplantable` is where it
    becomes a refusal. The comparison is the shared semantic one, so the
    activation in force is compared whichever key declares it, and a field
    neither config declares is reported in ``absent_on_both`` rather than
    silently reading as a match. ``tied_embeddings`` resolves to the transformers
    family default when absent, which makes an embedding and a head transplant
    one intervention rather than two.
    """
    configs = [json.loads((root / 'config.json').read_text()) for root in (destination, source)]
    comparison = compare_configs(configs)
    extra = {key: [config.get(key, 'not_declared') for config in configs] for key in extra_keys}
    extra_mismatched = sorted(key for key, pair in extra.items() if pair[0] != pair[1])
    return {
        **comparison,
        'mismatched': sorted(comparison['mismatched'] + extra_mismatched),
        'family_specific_fields': extra,
        'blocking_mismatches': sorted(field for field in comparison['mismatched']
                                      if field in set(blocking)),
        'tied_embeddings': comparison['fields']['tied_embeddings']['values'],
        'tied_embeddings_source': comparison['fields']['tied_embeddings']['sources'],
        'config_sha256': [hashlib.sha256((root / 'config.json').read_bytes()).hexdigest()
                          for root in (destination, source)],
    }


def config_key_audit(checkpoints: Mapping[str, Path]) -> dict[str, Any]:
    """Which key each architecture declares each semantic field under, measured.

    The answer to "does any other field the census compares have the same
    problem as the activation". It is read off the panel's own configs rather
    than recalled: for every semantic field, the audit records which key supplied
    it per checkpoint, and then names the fields on which a **single-key**
    comparison would be blind — a field two checkpoints declare under different
    keys, which a raw-key comparison reads as absent on both and therefore as
    matching whatever the values are.
    """
    resolved: dict[str, dict[str, Any]] = {}
    families: dict[str, str] = {}
    for name, root in sorted(checkpoints.items()):
        config = json.loads((Path(root) / 'config.json').read_text())
        families[name] = str(config.get('model_type', 'not_declared'))
        resolved[name] = {field: resolve_config_field(config, field)
                          for field in CONFIG_FIELD_ALIASES}
    fields: dict[str, Any] = {}
    for field in CONFIG_FIELD_ALIASES:
        keys: dict[str, list[str]] = {}
        for name in resolved:
            entry = resolved[name][field]
            label = entry['key'] or f"<{entry['source']}>"
            keys.setdefault(label, []).append(name)
        declared = sorted(key for key in keys if not key.startswith('<'))
        fields[field] = {
            'aliases': list(CONFIG_FIELD_ALIASES[field]),
            'keys_in_use': {key: sorted(names) for key, names in sorted(keys.items())},
            'distinct_declaring_keys': len(declared),
            'single_key_comparison_is_blind': len(declared) > 1,
        }
    blind = sorted(field for field, entry in fields.items()
                   if entry['single_key_comparison_is_blind'])
    return {
        'schema_version': 'd2_config_key_audit_v1',
        'checkpoints': sorted(checkpoints),
        'model_types': families,
        'fields': fields,
        'fields_a_single_key_comparison_would_miss': blind,
        'per_checkpoint': {name: {field: {'key': entry['key'], 'value': entry['value'],
                                          'source': entry['source']}
                                  for field, entry in values.items()}
                           for name, values in resolved.items()},
        'statement': (
            'A field declared under different keys by different families is one a '
            'single-key comparison reads as absent on both sides and therefore as '
            'matching, whatever the two values are. Every field listed in '
            'fields_a_single_key_comparison_would_miss carries that hazard across '
            'the checkpoints audited here.'),
    }


def pair_census(destination: Path, source: Path, *, arms: Sequence[str],
                chunk_elements: int = CHUNK_ELEMENTS, progress=None) -> dict[str, Any]:
    """Tensor-name, shape and value census over two checkpoints. CPU-only.

    Refuses to resize: a tensor whose shapes differ is recorded with
    ``shape_compatible`` false and its group is marked untransplantable, so a
    selection naming that group raises in :func:`group_provenance` instead of
    silently moving a truncated or padded tensor.

    Beyond the element-difference count the admitted census reports, each
    shape-compatible tensor carries the Pearson correlation and the relative L2
    distance between the two checkpoints' values. Element counts cannot separate
    a continued-pretraining pair from two independent training runs -- continued
    pretraining changed 98.905% of the ProLLaMA pair's elements -- and a
    correlation can: independent initialisations correlate at order
    ``1 / sqrt(n)``.
    """
    import torch

    n_layers = int(resolve_config_field(
        json.loads((destination / 'config.json').read_text()), 'depth')['value'])
    configuration = config_comparison(destination, source)
    names, files, readers = [], [], []
    for root in (destination, source):
        listed, reader, members = _load_state(root)
        names.append(listed)
        readers.append(reader)
        files.append({'checkpoint': str(root), 'members': members,
                      'member_sha256': {member: sha256_file(root / member)
                                        for member in members}})
    shared = sorted(set(names[0]) & set(names[1]))
    rows: list[dict[str, Any]] = []
    for name in shared:
        tensors = [read(name) for read in readers]
        left, right = tensors
        compatible = tuple(left.shape) == tuple(right.shape)
        row: dict[str, Any] = {
            'name': name,
            'group': opt_group_of(name, n_layers),
            'shapes': [list(left.shape), list(right.shape)],
            'shape_compatible': compatible,
            'n_elements': int(left.numel()),
            'dtypes': [str(left.dtype), str(right.dtype)],
        }
        if compatible:
            # The digest is the shared definition rather than a second one, so a
            # destination tensor digested off a loaded model after a transplant is
            # comparable with its census entry by construction.
            digests = [fp32_digest(tensor, chunk_elements=chunk_elements) for tensor in tensors]
            flat = [tensor.detach().reshape(-1) for tensor in tensors]
            totals = torch.zeros(5, dtype=torch.float64)
            differing = 0
            for start in range(0, left.numel(), chunk_elements):
                chunks = [piece[start:start + chunk_elements].contiguous().to(torch.float64)
                          for piece in flat]
                differing += int(torch.count_nonzero(
                    chunks[0].to(torch.float32) != chunks[1].to(torch.float32)).item())
                a, b = chunks
                totals += torch.stack([a.sum(), b.sum(), (a * a).sum(), (b * b).sum(),
                                       (a * b).sum()])
            count = float(left.numel())
            sum_a, sum_b, sq_a, sq_b, cross = (float(value) for value in totals)
            var_a = max(sq_a / count - (sum_a / count) ** 2, 0.0)
            var_b = max(sq_b / count - (sum_b / count) ** 2, 0.0)
            covariance = cross / count - (sum_a / count) * (sum_b / count)
            row.update(
                fp32_sha256=digests,
                differing_elements_fp32=differing,
                fp32_bytes_equal=digests[0] == digests[1],
                pearson_r=(covariance / (var_a * var_b) ** 0.5
                           if var_a > 0 and var_b > 0 else None),
                relative_l2=(((sq_a - 2 * cross + sq_b) / sq_a) ** 0.5 if sq_a > 0 else None),
            )
        else:
            row.update(fp32_sha256=[None, None], differing_elements_fp32=None,
                       fp32_bytes_equal=False, pearson_r=None, relative_l2=None)
        rows.append(row)
        if progress is not None:
            progress(row)
        del tensors, left, right
    members = opt_partition(shared, n_layers)
    by_name = {row['name']: row for row in rows}
    groups: dict[str, Any] = {}
    for group, group_members in members.items():
        selected = [by_name[name] for name in group_members]
        compatible = all(row['shape_compatible'] for row in selected)
        correlations = [row['pearson_r'] for row in selected if row['pearson_r'] is not None]
        groups[group] = {
            'n_tensors': len(selected),
            'parameters': sum(row['n_elements'] for row in selected),
            'shape_compatible': compatible,
            'transplantable': bool(compatible and selected),
            'differing_elements_fp32': (sum(row['differing_elements_fp32'] for row in selected)
                                        if compatible else None),
            'fp32_bytes_equal': bool(selected) and all(row['fp32_bytes_equal']
                                                       for row in selected),
            'median_pearson_r': _median(correlations),
            'median_relative_l2': _median([row['relative_l2'] for row in selected
                                           if row['relative_l2'] is not None]),
            'tensor_names': group_members,
        }
    return {
        'schema_version': 'd2_mechanistic_pair_census_v1',
        'status': 'complete',
        'arms': list(arms),
        'checkpoint_order': ['destination', 'source'],
        'checkpoint_order_arms': list(arms),
        'num_hidden_layers': n_layers,
        'configuration': configuration,
        'tensor_names': {'destination_only': sorted(set(names[0]) - set(names[1])),
                         'source_only': sorted(set(names[1]) - set(names[0])),
                         'shared': len(shared)},
        'group_order': list(opt_group_names()),
        'groups': groups,
        'tensors': rows,
        'sources': files,
        'comparison': ('exact elementwise equality after FP32 conversion, with the Pearson '
                       'correlation and relative L2 of the two stored tensors beside it'),
        'limitation': ('A census of stored values. Differing tensors do not establish which '
                       'values carry a behaviour, and identical tensors are a no-op '
                       'transplant rather than evidence of insensitivity.'),
    }


def _median(values: Sequence[float]) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    return (ordered[middle] if len(ordered) % 2
            else 0.5 * (ordered[middle - 1] + ordered[middle]))


def shared_initialisation_verdict(census: Mapping[str, Any]) -> dict[str, Any]:
    """Whether the two checkpoints' body tensors evidence a shared initialisation.

    The handoff records that shared initialisation is undeclared for the
    recommended pair and that a census decides it. This is that decision, and it
    is a measurement rather than a reading of model cards: the median per-tensor
    correlation over the shape-compatible body groups against what two
    independent initialisations would give.
    """
    body = [name for name in census['group_order']
            if name.startswith(('attention_', 'mlp_')) and census['groups'][name]['n_tensors']]
    correlations = [census['groups'][name]['median_pearson_r'] for name in body
                    if census['groups'][name]['median_pearson_r'] is not None]
    median = _median(correlations)
    evidenced = median is not None and median > SHARED_INITIALISATION_CORRELATION
    return {
        'body_groups': body,
        'median_body_pearson_r': median,
        'threshold': SHARED_INITIALISATION_CORRELATION,
        'shared_initialisation_evidenced': bool(evidenced),
        'statement': (
            'The body tensors correlate far above what independent initialisations would '
            'give, so the two checkpoints share an ancestor and a partial transplant is '
            'defined on the shape-compatible groups.' if evidenced else
            'The body tensors correlate at the level two independent training runs would '
            'give, so no shared initialisation is evidenced: the pair is an '
            'architecture-matched comparison and no transplant is licensed.'),
    }


def assert_pair_transplantable(census: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse a pair whose census does not define a transplant, and say which part.

    Partial transplantability is a real state and is reported as one: a pair can
    be shape-incompatible at the embedding and the head while every projection
    corresponds, in which case the groups that correspond are the only ones a
    cell may name -- and the floor and ceiling are then no longer the two
    admitted checkpoints, which is the condition that makes the recovered
    fraction unreadable.
    """
    blocking = census['configuration']['blocking_mismatches']
    if blocking:
        raise ValueError(
            f'configuration fields {blocking} differ, so a transplant is undefined: '
            'installing weights trained under one architecture into another is not a '
            'parameter transplant')
    initialisation = shared_initialisation_verdict(census)
    if not initialisation['shared_initialisation_evidenced']:
        raise ValueError(
            'no shared initialisation is evidenced between these checkpoints; '
            + initialisation['statement'])
    untransplantable = sorted(name for name, group in census['groups'].items()
                              if group['n_tensors'] and not group['transplantable'])
    if untransplantable:
        raise ValueError(
            f'groups {untransplantable} are shape-incompatible, so the full selection is '
            'not the donor and the ceiling cannot reconstruct it; a partial partition '
            'has no exact ceiling and no readable recovered fraction')
    return {'transplantable': True, 'initialisation': initialisation}


# --------------------------------------------------------------------------- #
# The composition rule with the depth sweep.
# --------------------------------------------------------------------------- #

def composition_rule(pair: Mapping[str, Any], *, depths: Mapping[str, int],
                     profiles_differ_at_a_block: bool | None = None) -> dict[str, Any]:
    """Whether a pair is a sharp patching target, on the three declared conditions.

    A pair is sharper when all three hold together: both sides are among the
    position-resolved arms, so a mutant-minus-wild-type difference at a residue
    index is defined on both and a residue-indexed intervention site exists;
    their depths are equal, so block *j* on one side corresponds to block *j* on
    the other with no rescaling convention; and their per-block increment
    profiles differ at an identifiable block or contiguous band rather than only
    in the pooled endpoint. Under all three the intervention has a declared site
    and a declared prediction. A pooled endpoint supplies neither, which is why
    a pair whose outcomes differ and whose depth profiles diverge at an
    identifiable block is a far sharper target than a pair differing only at the
    endpoint.

    The pair's declared facts are read from the handoff's own separation report
    rather than transcribed, so this function cannot disagree with it.
    """
    arms = list(pair['arms'])
    equal_depth = depths[arms[0]] == depths[arms[1]]
    conditions = {
        'both_position_resolved': bool(pair['position_resolved_both']),
        'equal_depth': bool(equal_depth),
        'profiles_differ_at_a_block': profiles_differ_at_a_block,
    }
    degradations = []
    if not conditions['both_position_resolved']:
        degradations.append(
            'at least one side is multi-residue, so no position-resolved summary is '
            'defined and the pair supports whole-block or parameter-group interventions '
            'only, with no residue-indexed site however the sweep comes out')
    if not conditions['equal_depth']:
        degradations.append(
            f'depths differ, {depths[arms[0]]} against {depths[arms[1]]}, so any block '
            'correspondence is a convention: the comparison is between normalised depths '
            'j/(L-1) and the site is a matched band rather than a matched index, and the '
            'convention must be recorded with the number')
    satisfied = (conditions['both_position_resolved'] and conditions['equal_depth']
                 and profiles_differ_at_a_block is True)
    return {
        'arms': arms,
        'depths': [depths[arms[0]], depths[arms[1]]],
        'outcome_differs': {'likelihood': bool(pair['likelihood_outcome_differs']),
                            'representation': bool(pair['representation_outcome_differs'])},
        'conditions': conditions,
        'satisfies_rule': bool(satisfied),
        'pending_depth_sweep': profiles_differ_at_a_block is None,
        'degradations': degradations,
    }
