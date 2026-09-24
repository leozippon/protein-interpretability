"""Conditions the Direction-1 to Direction-2 interface must always hold, and its refusals.

The interface exists so a mechanism result composes with the capability result
rather than sitting beside it, and every condition below is one whose silent
failure would produce a plausible recovered fraction read against the wrong
scale, the wrong support or the wrong fold map. The negative paths are the point:
each refusal is exercised by making it fire.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.transfer import mechanistic_interface as mi
from src.transfer import capability_transplant as ct
from src.transfer.capability_transplant import support_digest


# --------------------------------------------------------------------------- #
# Synthetic OPT checkpoints, the cheapest thing that exercises the real census.
# --------------------------------------------------------------------------- #

CONFIG = {
    'model_type': 'opt', 'hidden_size': 8, 'ffn_dim': 16, 'num_hidden_layers': 4,
    'num_attention_heads': 2, 'vocab_size': 10, 'word_embed_proj_dim': 8,
    'max_position_embeddings': 16, 'activation_function': 'relu',
    'do_layer_norm_before': True, '_remove_final_layer_norm': False, 'enable_bias': True,
    'layer_norm_elementwise_affine': True,
}


def write_checkpoint(root: Path, *, vocab=10, activation='relu', seed=0, drift=0.0,
                     base_seed=None):
    """A tiny OPT-shaped checkpoint, optionally a drifted copy of a shared ancestor.

    ``base_seed`` shares the initialisation and ``drift`` perturbs it, which is
    what a continued-pretraining pair looks like; leaving ``base_seed`` unset
    draws independent values, which is what two separate training runs look like.
    """
    import hashlib

    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=True)
    config = dict(CONFIG, vocab_size=vocab, activation_function=activation)
    (root / 'config.json').write_text(json.dumps(config, indent=1))
    ancestor_seed = base_seed if base_seed is not None else seed
    hidden, ffn, layers = config['hidden_size'], config['ffn_dim'], config['num_hidden_layers']

    def tensor(name, *shape):
        # One generator per tensor name, so a checkpoint with a different
        # vocabulary size still draws the same body values from a shared
        # ancestor seed; a single sequential generator would shift every
        # subsequent draw and hide the shared initialisation.
        key = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'big')
        value = torch.randn(*shape, generator=torch.Generator().manual_seed(
            (ancestor_seed * 1_000_003 + key) % (2 ** 31)))
        if drift:
            value = value + drift * torch.randn(*shape, generator=torch.Generator(
            ).manual_seed((seed * 7_919 + key + 9_000) % (2 ** 31)))
        return value.to(torch.float32)

    names = {
        'model.decoder.embed_tokens.weight': (vocab, hidden),
        'model.decoder.embed_positions.weight': (config['max_position_embeddings'], hidden),
        'model.decoder.final_layer_norm.weight': (hidden,),
        'model.decoder.final_layer_norm.bias': (hidden,),
    }
    for layer in range(layers):
        prefix = f'model.decoder.layers.{layer}'
        for key in ('q', 'k', 'v', 'out'):
            names[f'{prefix}.self_attn.{key}_proj.weight'] = (hidden, hidden)
            names[f'{prefix}.self_attn.{key}_proj.bias'] = (hidden,)
        names[f'{prefix}.fc1.weight'] = (ffn, hidden)
        names[f'{prefix}.fc1.bias'] = (ffn,)
        names[f'{prefix}.fc2.weight'] = (hidden, ffn)
        names[f'{prefix}.fc2.bias'] = (hidden,)
        for name in ('self_attn_layer_norm', 'final_layer_norm'):
            names[f'{prefix}.{name}.weight'] = (hidden,)
            names[f'{prefix}.{name}.bias'] = (hidden,)
    save_file({name: tensor(name, *shape) for name, shape in names.items()},
              str(root / 'model.safetensors'))
    return root


# --------------------------------------------------------------------------- #
# The endpoint contract.
# --------------------------------------------------------------------------- #

def test_the_endpoint_contract_is_assembled_from_the_admitted_modules():
    contract = mi.endpoint_contract()
    assert contract['split_seeds'] == [20260923, 20260924, 20260925]
    assert contract['outer_folds'] == 5 and contract['inner_folds'] == 4
    assert contract['inner_seed_rule'].startswith('outer split seed + 100')
    assert contract['anchor']['assays'] == 201
    assert contract['anchor']['wildtype_clusters'] == 163
    assert contract['anchor']['resampling_unit'] == 'wild-type family at 50% identity'
    assert contract['bootstrap']['draws'] == 2000
    assert contract['projection']['blas_threads'] == 4
    assert contract['pipeline_identity_tolerance'] == 1e-8


def test_two_modules_declaring_the_split_seeds_must_agree(monkeypatch):
    """A second copy of a constant is a second source of truth until something compares them."""
    from src.transfer import local_context

    monkeypatch.setattr(local_context, 'SPLIT_SEEDS', (1, 2, 3), raising=True)
    with pytest.raises(ValueError, match='split seeds disagree'):
        mi.endpoint_contract()


# --------------------------------------------------------------------------- #
# Support, folds and the tuning isolation.
# --------------------------------------------------------------------------- #

def test_the_row_identity_digest_does_not_depend_on_the_numpy_version():
    """``repr(np.float64(0.5))`` differs between NumPy 1.26 and NumPy 2."""
    groups, keys = ['a', 'b'], ['x', 'y']
    python = mi.row_identity_digest(groups, keys, [0.5, -1.25])
    numpy_scalars = mi.row_identity_digest(groups, keys, [np.float64(0.5), np.float32(-1.25)])
    assert python == numpy_scalars
    with pytest.raises(ValueError, match='aligned'):
        mi.row_identity_digest(groups, keys, [0.5])


def test_the_fold_identity_digest_is_membership_sorted_and_fold_ordered():
    first = mi.fold_identity_digest([['b', 'a'], ['c']])
    assert first == mi.fold_identity_digest([['a', 'b'], ['c']])
    assert first != mi.fold_identity_digest([['c'], ['a', 'b']])
    assert first == mi.fold_identity_digest([[np.str_('b'), 'a'], ['c']])
    with pytest.raises(ValueError, match='at least one fold'):
        mi.fold_identity_digest([])


def test_a_cell_on_another_support_is_refused():
    assays = [f'A{index}' for index in range(201)]
    digest = support_digest(assays)
    assert mi.check_support_identity(assays, expected_digest=digest)['assays'] == 201
    with pytest.raises(ValueError, match='does not compose'):
        mi.check_support_identity(assays[:200], expected_digest=digest)
    with pytest.raises(ValueError, match='does not compose'):
        mi.check_support_identity(assays + ['A999'], expected_digest=digest)
    # The ProLLaMA lineage's admitted endpoints are on a 211-assay native support,
    # which is exactly the mismatch this refusal exists for.
    assert mi.ADMITTED_NATIVE_LIKELIHOOD['prollama-stage-1']['assays'] == 211
    with pytest.raises(ValueError, match='does not compose'):
        mi.check_support_identity([f'A{index}' for index in range(211)],
                                  expected_digest=digest)


def test_a_cell_on_other_folds_is_refused():
    membership = [['a', 'b'], ['c'], ['d'], ['e'], ['f']]
    digest = mi.fold_identity_digest(membership)
    assert mi.check_fold_identity(membership, expected_digest=digest,
                                  expected_folds=5)['folds'] == 5
    with pytest.raises(ValueError, match='outer folds against'):
        mi.check_fold_identity(membership[:4], expected_digest=digest, expected_folds=5)
    with pytest.raises(ValueError, match='fold identity'):
        mi.check_fold_identity([['a'], ['b'], ['c'], ['d'], ['e', 'f']],
                               expected_digest=digest, expected_folds=5)


def test_the_admitted_nested_construction_keeps_held_labels_out_of_tuning():
    """Realistic end to end: the admitted splitter, the admitted inner seed rule."""
    from src.transfer.pairwise_epistasis import INNER_SPLITS, OUTER_SPLITS, SPLIT_SEEDS
    from src.transfer.readout_analysis import family_folds

    groups = np.array([f'cluster{index % 40}' for index in range(400)])
    seed = SPLIT_SEEDS[0]
    folds = []
    for outer, held in enumerate(family_folds(groups, OUTER_SPLITS, seed)):
        train = groups[~np.isin(groups, held)]
        folds.append({'held_groups': list(held),
                      'inner_folds': family_folds(train, INNER_SPLITS, seed + 100 + outer)})
    report = mi.check_tuning_isolation(folds)
    assert report['held_labels_reach_tuning'] is False
    assert report['outer_folds'] == OUTER_SPLITS
    assert all(len(entry['inner_folds']) == INNER_SPLITS for entry in report['per_fold'])


def test_a_held_out_label_reaching_penalty_selection_is_refused():
    """A refit that partitioned all groups would select against the labels it scores."""
    leaked = [{'held_groups': ['a', 'b'], 'inner_folds': [['a'], ['c'], ['d'], ['e']]}]
    with pytest.raises(ValueError, match='reached penalty selection'):
        mi.check_tuning_isolation(leaked)
    with pytest.raises(ValueError, match='not a partition'):
        mi.check_tuning_isolation([{'held_groups': ['z'],
                                    'inner_folds': [['a', 'b'], ['b', 'c']]}])
    with pytest.raises(ValueError, match='no inner folds'):
        mi.check_tuning_isolation([{'held_groups': ['z'], 'inner_folds': []}])
    with pytest.raises(ValueError, match='at least one outer fold'):
        mi.check_tuning_isolation([])


# --------------------------------------------------------------------------- #
# The floor, the ceiling and the provenance.
# --------------------------------------------------------------------------- #

def census_stub():
    return {
        'checkpoint_order': ['destination', 'source'],
        'checkpoint_order_arms': ['recipient', 'donor'],
        'group_order': ['embedding', 'norm', 'attention_q1', 'mlp_q1'],
        'groups': {
            'embedding': {'n_tensors': 1, 'parameters': 40, 'shape_compatible': False,
                          'transplantable': False, 'fp32_bytes_equal': False,
                          'tensor_names': ['embed']},
            'norm': {'n_tensors': 1, 'parameters': 8, 'shape_compatible': True,
                     'transplantable': True, 'fp32_bytes_equal': True,
                     'tensor_names': ['norm']},
            'attention_q1': {'n_tensors': 1, 'parameters': 64, 'shape_compatible': True,
                             'transplantable': True, 'fp32_bytes_equal': False,
                             'tensor_names': ['attn']},
            'mlp_q1': {'n_tensors': 1, 'parameters': 128, 'shape_compatible': True,
                       'transplantable': True, 'fp32_bytes_equal': False,
                       'tensor_names': ['mlp']},
        },
        'tensors': [
            {'name': 'embed', 'group': 'embedding', 'n_elements': 40,
             'fp32_sha256': [None, None], 'differing_elements_fp32': None},
            {'name': 'norm', 'group': 'norm', 'n_elements': 8,
             'fp32_sha256': ['n0', 'n0'], 'differing_elements_fp32': 0},
            {'name': 'attn', 'group': 'attention_q1', 'n_elements': 64,
             'fp32_sha256': ['a0', 'a1'], 'differing_elements_fp32': 60},
            {'name': 'mlp', 'group': 'mlp_q1', 'n_elements': 128,
             'fp32_sha256': ['m0', 'm1'], 'differing_elements_fp32': 120},
        ],
    }


def test_the_floor_and_the_ceiling_must_reconstruct_their_checkpoints_bit_identically():
    census = census_stub()
    floor = {'embed': None, 'norm': 'n0', 'attn': 'a0', 'mlp': 'm0'}
    ceiling = {'embed': None, 'norm': 'n0', 'attn': 'a1', 'mlp': 'm1'}
    assert mi.check_reconstruction(floor, census, end='floor')['bit_identical'] is True
    assert mi.check_reconstruction(ceiling, census, end='ceiling')['checkpoint'] == 'source'
    with pytest.raises(ValueError, match='bit-identically'):
        mi.check_reconstruction(ceiling, census, end='floor')
    with pytest.raises(ValueError, match='missing'):
        mi.check_reconstruction({'norm': 'n0', 'attn': 'a0', 'mlp': 'm0'}, census, end='floor')
    with pytest.raises(ValueError, match='unexpected'):
        mi.check_reconstruction(dict(floor, spare='x'), census, end='floor')
    with pytest.raises(ValueError, match='floor or ceiling'):
        mi.check_reconstruction(floor, census, end='middle')


def test_an_endpoint_outside_its_admitted_interval_is_refused():
    admitted = mi.ADMITTED_ANCHOR_LIKELIHOOD['galactica-1.3b']
    inside = mi.check_admitted_endpoint({'point': admitted['point']}, admitted)
    assert inside['exact'] is True and inside['within_admitted_interval'] is True
    assert mi.check_admitted_endpoint({'point': 0.026}, admitted)['exact'] is False
    with pytest.raises(ValueError, match='does not reconstruct the admitted model'):
        mi.check_admitted_endpoint({'point': 0.20}, admitted)


def test_group_provenance_reports_parameter_counts_and_refuses_no_op_groups():
    census = census_stub()
    record = mi.group_provenance(census, ['attention_q1', 'mlp_q1'])
    assert record['parameters_transplanted'] == 192
    assert record['tensors_transplanted'] == 2
    assert record['groups']['mlp_q1']['source_fp32_sha256'] == {'mlp': 'm1'}
    assert record['groups']['attention_q1']['differing_elements_fp32'] == 60
    with pytest.raises(ValueError, match='byte-identical'):
        mi.group_provenance(census, ['norm'])
    with pytest.raises(ValueError, match='not shape-compatible'):
        mi.group_provenance(census, ['embedding'])
    with pytest.raises(ValueError, match='undeclared groups'):
        mi.group_provenance(census, ['rope'])


def test_a_cell_passes_only_when_every_refusal_passes():
    census = census_stub()
    assays = [f'A{index}' for index in range(201)]
    digest = support_digest(assays)
    admitted = {'recipient': {'point': 0.0, 'interval': [-0.01, 0.01]},
                'donor': {'point': 0.3, 'interval': [0.28, 0.32]}}
    floor = {'selected_groups': [], 'assays': assays, 'end': 'floor',
             'endpoint': {'point': 0.0},
             'verified_tensor_fp32_sha256': {'embed': None, 'norm': 'n0', 'attn': 'a0',
                                             'mlp': 'm0'}}
    record = mi.assert_cell_admitted(floor, census, admitted=admitted,
                                     support_expectation=digest)
    assert record['reconstruction']['bit_identical'] is True
    assert record['admitted_comparison']['within_admitted_interval'] is True
    assert 'sufficient in this context' in record['licenses']['sufficiency']
    with pytest.raises(ValueError, match='floor selects no group'):
        mi.assert_cell_admitted(dict(floor, selected_groups=['mlp_q1']), census,
                                admitted=admitted, support_expectation=digest)
    with pytest.raises(ValueError, match='ceiling must select every declared group'):
        mi.assert_cell_admitted(dict(floor, end='ceiling',
                                     selected_groups=['attention_q1', 'mlp_q1'],
                                     endpoint={'point': 0.3}),
                                census, admitted=admitted, support_expectation=digest)
    body = {'selected_groups': ['mlp_q1'], 'assays': assays, 'endpoint': {'point': 0.1},
            'verified_tensor_fp32_sha256': {}}
    assert mi.assert_cell_admitted(body, census, admitted=admitted,
                                   support_expectation=digest
                                   )['provenance']['parameters_transplanted'] == 128
    with pytest.raises(ValueError, match='undeclared groups'):
        mi.assert_cell_admitted(dict(body, selected_groups=['nowhere']), census,
                                admitted=admitted, support_expectation=digest)


# --------------------------------------------------------------------------- #
# The OPT partition and the compatibility census.
# --------------------------------------------------------------------------- #

#: The key tuple the census compared before the semantic table replaced it,
#: kept here as the thing the regression tests below compare against rather than
#: as a live code path.
RAW_KEYS = (
    'model_type', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
    'num_attention_heads', 'num_key_value_heads', 'vocab_size', 'rms_norm_eps',
    'rope_theta', 'rope_scaling', 'max_position_embeddings', 'hidden_act',
    'attention_bias', 'mlp_bias', 'tie_word_embeddings',
)

GPT2 = {'model_type': 'gpt2', 'n_embd': 1280, 'n_layer': 36, 'n_head': 20,
        'n_positions': 1024, 'n_inner': None, 'layer_norm_epsilon': 1e-05,
        'activation_function': 'gelu_new', 'vocab_size': 50257}


def raw_key_mismatches(configs):
    """What a single-key comparison would have found: the defect, spelled out."""
    return sorted(key for key in RAW_KEYS
                  if configs[0].get(key) != configs[1].get(key))


def test_a_semantic_field_resolves_under_whichever_key_declares_it():
    llama = {'hidden_act': 'silu', 'tie_word_embeddings': False}
    opt = {'activation_function': 'relu'}
    assert ct.resolve_config_field(llama, 'activation')['value'] == 'silu'
    assert ct.resolve_config_field(llama, 'activation')['key'] == 'hidden_act'
    assert ct.resolve_config_field(opt, 'activation')['value'] == 'relu'
    assert ct.resolve_config_field(opt, 'activation')['key'] == 'activation_function'
    assert ct.resolve_config_field(GPT2, 'depth')['value'] == 36
    assert ct.resolve_config_field(GPT2, 'hidden_width')['value'] == 1280
    # An omitted tie flag means tied, which is the refusing direction.
    assert ct.resolve_config_field(opt, 'tied_embeddings') == {
        'field': 'tied_embeddings', 'key': None, 'value': True, 'source': 'family default'}
    assert ct.resolve_config_field(llama, 'tied_embeddings')['source'] == 'declared'
    assert ct.resolve_config_field({}, 'rope_theta')['source'] == 'absent'
    with pytest.raises(ValueError, match='undeclared configuration field'):
        ct.resolve_config_field(llama, 'nonsense')


def test_the_raw_key_comparison_was_blind_where_the_semantic_one_is_not():
    """The defect the census found, asserted in both directions.

    A GPT-2 pair differing in depth and width is caught by the semantic
    comparison and missed entirely by a single-key one, which reads six
    load-bearing fields as absent on both sides and therefore as matching.
    """
    deeper = dict(GPT2, n_layer=48, n_embd=1600)
    assert raw_key_mismatches([GPT2, deeper]) == []
    semantic = ct.compare_configs([GPT2, deeper])
    assert semantic['mismatched'] == ['depth', 'hidden_width']
    assert semantic['fields']['depth']['values'] == [36, 48]
    assert semantic['fields']['depth']['keys'] == ['n_layer', 'n_layer']

    gelu = {'model_type': 'opt', 'hidden_size': 2048, 'ffn_dim': 8192,
            'num_hidden_layers': 24, 'num_attention_heads': 32, 'vocab_size': 50000,
            'activation_function': 'gelu', 'enable_bias': True}
    relu = dict(gelu, activation_function='relu')
    assert raw_key_mismatches([gelu, relu]) == []
    assert ct.compare_configs([gelu, relu])['mismatched'] == ['activation']


def test_the_admitted_llama_comparison_is_unchanged_by_the_repair():
    """The one compatibility statement the repair has to make.

    The admitted ProLLaMA census's behaviour must not move: its two configs
    declare every field under the keys the old tuple named, so the semantic
    comparison resolves the same values and finds the same nothing.
    """
    llama = {'model_type': 'llama', 'hidden_size': 4096, 'intermediate_size': 11008,
             'num_hidden_layers': 32, 'num_attention_heads': 32, 'num_key_value_heads': 32,
             'vocab_size': 32000, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0,
             'rope_scaling': None, 'max_position_embeddings': 4096, 'hidden_act': 'silu',
             'attention_bias': False, 'mlp_bias': False, 'tie_word_embeddings': False}
    comparison = ct.compare_configs([llama, llama])
    assert comparison['mismatched'] == []
    assert comparison['absent_on_both'] == []
    for field in ('activation', 'depth', 'hidden_width', 'mlp_width', 'attention_heads'):
        assert comparison['fields'][field]['sources'] == ['declared', 'declared']
    assert ct.compare_configs([llama, dict(llama, hidden_act='gelu')])['mismatched'] == [
        'activation']
    with pytest.raises(ValueError, match='exactly two configs'):
        ct.compare_configs([llama])


def test_an_omitted_tie_flag_no_longer_admits_a_tied_pair(tmp_path):
    """The second defect in the same function: a missing flag read as untied."""
    config = {'model_type': 'opt', 'hidden_size': 8, 'ffn_dim': 16, 'num_hidden_layers': 4,
              'num_attention_heads': 2, 'vocab_size': 10, 'activation_function': 'relu',
              'max_position_embeddings': 16, 'enable_bias': True}
    roots = []
    for name in ('tied', 'tied_too'):
        root = tmp_path / name
        root.mkdir()
        (root / 'config.json').write_text(json.dumps(config))
        (root / 'tokenizer.model').write_bytes(b'tokenizer')
        (root / 'model.safetensors.index.json').write_text(
            json.dumps({'weight_map': {'w': 'model.safetensors'}}))
        roots.append(root)
    with pytest.raises(ValueError, match='undefined for tied weights'):
        ct.architecture_record(*roots)
    for root in roots:
        (root / 'config.json').write_text(json.dumps(dict(config, tie_word_embeddings=False)))
    record = ct.architecture_record(*roots)
    assert record['untied_embeddings'] is True
    assert record['num_hidden_layers'] == 4
    assert record['configuration']['mismatched'] == []


def test_the_config_key_audit_names_the_fields_a_single_key_would_miss(tmp_path):
    roster = {}
    for name, config in (('llama', {'model_type': 'llama', 'hidden_act': 'silu',
                                    'num_hidden_layers': 32, 'hidden_size': 4096,
                                    'vocab_size': 32000}),
                         ('opt', {'model_type': 'opt', 'activation_function': 'relu',
                                  'num_hidden_layers': 24, 'hidden_size': 2048,
                                  'vocab_size': 50000}),
                         ('gpt2', dict(GPT2))):
        root = tmp_path / name
        root.mkdir()
        (root / 'config.json').write_text(json.dumps(config))
        roster[name] = root
    audit = mi.config_key_audit(roster)
    blind = audit['fields_a_single_key_comparison_would_miss']
    assert 'activation' in blind and 'depth' in blind and 'hidden_width' in blind
    # Every checkpoint declares the vocabulary under one key, so it is not blind.
    assert 'vocabulary' not in blind
    assert audit['fields']['activation']['keys_in_use']['hidden_act'] == ['llama']
    assert sorted(audit['fields']['activation']['keys_in_use']['activation_function']) == [
        'gpt2', 'opt']
    assert audit['model_types'] == {'llama': 'llama', 'opt': 'opt', 'gpt2': 'gpt2'}
    assert audit['per_checkpoint']['gpt2']['depth']['key'] == 'n_layer'


def test_the_opt_partition_covers_the_checkpoint_or_refuses():
    names = [
        'model.decoder.embed_tokens.weight', 'model.decoder.embed_positions.weight',
        'model.decoder.final_layer_norm.weight', 'model.decoder.final_layer_norm.bias',
    ]
    for layer in range(4):
        prefix = f'model.decoder.layers.{layer}'
        names += [f'{prefix}.self_attn.q_proj.weight', f'{prefix}.fc1.weight',
                  f'{prefix}.self_attn_layer_norm.weight']
    members = mi.opt_partition(names, 4)
    assert set(members) == set(mi.opt_group_names())
    assert members['attention_q1'] == ['model.decoder.layers.0.self_attn.q_proj.weight']
    with pytest.raises(ValueError, match='unclassified'):
        mi.opt_partition(names + ['model.decoder.layers.0.unknown.weight'], 4)
    with pytest.raises(ValueError, match='duplicate'):
        mi.opt_partition(names + names[:1], 4)
    with pytest.raises(ValueError, match='absent from the checkpoint'):
        mi.opt_partition(names[:4], 4)


def test_the_census_refuses_a_shape_mismatch_rather_than_resizing(tmp_path):
    """The recommended pair's embedding and head differ by 304 rows.

    The body here shares an initialisation, so the shape mismatch is the only
    reason the transplant is refused and the refusal names it.
    """
    destination = write_checkpoint(tmp_path / 'left', vocab=10, seed=1, base_seed=11)
    source = write_checkpoint(tmp_path / 'right', vocab=12, seed=2, drift=0.2, base_seed=11)
    census = mi.pair_census(destination, source, arms=['left', 'right'])
    embedding = census['groups']['embedding']
    assert embedding['shape_compatible'] is False
    assert embedding['transplantable'] is False
    assert embedding['differing_elements_fp32'] is None
    row = next(row for row in census['tensors'] if row['group'] == 'embedding')
    assert row['shapes'] == [[10, 8], [12, 8]]
    assert census['configuration']['mismatched'] == ['vocabulary']
    assert census['configuration']['blocking_mismatches'] == []
    with pytest.raises(ValueError, match='shape-incompatible'):
        mi.assert_pair_transplantable(census)


def test_the_census_detects_a_shared_initialisation_and_its_absence(tmp_path):
    """Element counts cannot separate continued pretraining from two training runs.

    Continued pretraining changed 98.905% of the ProLLaMA pair's elements, so a
    difference count says nothing about a shared ancestor. A correlation does:
    independent initialisations correlate at order ``1 / sqrt(n)``.
    """
    ancestor = write_checkpoint(tmp_path / 'parent', vocab=10, seed=3)
    drifted = write_checkpoint(tmp_path / 'child', vocab=10, seed=4, drift=0.3, base_seed=3)
    census = mi.pair_census(ancestor, drifted, arms=['parent', 'child'])
    verdict = mi.shared_initialisation_verdict(census)
    assert verdict['shared_initialisation_evidenced'] is True
    assert verdict['median_body_pearson_r'] > 0.9
    assert census['groups']['mlp_q1']['differing_elements_fp32'] > 0
    assert mi.assert_pair_transplantable(census)['transplantable'] is True

    independent = write_checkpoint(tmp_path / 'other', vocab=10, seed=77)
    census = mi.pair_census(ancestor, independent, arms=['parent', 'other'])
    verdict = mi.shared_initialisation_verdict(census)
    assert verdict['shared_initialisation_evidenced'] is False
    assert abs(verdict['median_body_pearson_r']) < 0.1
    with pytest.raises(ValueError, match='no shared initialisation'):
        mi.assert_pair_transplantable(census)


def test_a_differing_activation_function_blocks_a_transplant(tmp_path):
    """The field an OPT pair differs on, which a Llama-shaped census does not compare."""
    destination = write_checkpoint(tmp_path / 'gelu', activation='gelu', seed=5)
    source = write_checkpoint(tmp_path / 'relu', activation='relu', seed=5)
    census = mi.pair_census(destination, source, arms=['gelu', 'relu'])
    assert 'activation' in census['configuration']['blocking_mismatches']
    assert 'activation' in ct.CONFIG_FIELD_ALIASES
    assert census['configuration']['fields']['activation']['keys'] == [
        'activation_function', 'activation_function']
    with pytest.raises(ValueError, match='activation'):
        mi.assert_pair_transplantable(census)


def test_a_tied_head_is_recorded_as_one_intervention_not_two(tmp_path):
    """OPT ties input and output embeddings unless the config says otherwise."""
    left = write_checkpoint(tmp_path / 'a', seed=6)
    right = write_checkpoint(tmp_path / 'b', seed=7)
    census = mi.pair_census(left, right, arms=['a', 'b'])
    assert census['configuration']['tied_embeddings'] == [True, True]
    assert all('default' in entry
               for entry in census['configuration']['tied_embeddings_source'])
    assert census['groups']['head']['n_tensors'] == 0
    assert census['groups']['head']['transplantable'] is False


# --------------------------------------------------------------------------- #
# The composition rule.
# --------------------------------------------------------------------------- #

def test_the_composition_rule_needs_all_three_conditions():
    pair = {'arms': ['a', 'b'], 'position_resolved_both': True,
            'likelihood_outcome_differs': True, 'representation_outcome_differs': False}
    depths = {'a': 24, 'b': 24}
    pending = mi.composition_rule(pair, depths=depths)
    assert pending['conditions'] == {'both_position_resolved': True, 'equal_depth': True,
                                     'profiles_differ_at_a_block': None}
    assert pending['satisfies_rule'] is False
    assert pending['pending_depth_sweep'] is True
    assert pending['degradations'] == []
    sharp = mi.composition_rule(pair, depths=depths, profiles_differ_at_a_block=True)
    assert sharp['satisfies_rule'] is True and sharp['pending_depth_sweep'] is False
    endpoint_only = mi.composition_rule(pair, depths=depths,
                                        profiles_differ_at_a_block=False)
    assert endpoint_only['satisfies_rule'] is False


def test_a_multi_residue_pair_fails_the_rule_whatever_the_sweep_returns():
    """Both sides of the ProLLaMA lineage and of the GPT-2 pair are multi-residue."""
    pair = {'arms': ['llama-2-7b', 'prollama-stage-1'], 'position_resolved_both': False,
            'likelihood_outcome_differs': True, 'representation_outcome_differs': True}
    record = mi.composition_rule(pair, depths={'llama-2-7b': 32, 'prollama-stage-1': 32},
                                 profiles_differ_at_a_block=True)
    assert record['satisfies_rule'] is False
    assert any('multi-residue' in reason for reason in record['degradations'])


def test_unequal_depths_degrade_to_a_normalised_band_and_say_so():
    pair = {'arms': ['progen3-112m', 'protgpt3-1.3b'], 'position_resolved_both': True,
            'likelihood_outcome_differs': False, 'representation_outcome_differs': True}
    record = mi.composition_rule(pair, depths={'progen3-112m': 10, 'protgpt3-1.3b': 17},
                                 profiles_differ_at_a_block=True)
    assert record['satisfies_rule'] is False
    assert any('normalised depths' in reason for reason in record['degradations'])
    assert record['depths'] == [10, 17]


def test_the_licence_statement_travels_with_every_record():
    for key in ('sufficiency', 'not_a_circuit', 'confound', 'batch_one'):
        assert key in mi.LICENSE_STATEMENT
    assert 'not show they are necessary' in mi.LICENSE_STATEMENT['sufficiency']
    assert 'architectural locus' in mi.LICENSE_STATEMENT['confound']
    assert 'true by' in mi.LICENSE_STATEMENT['batch_one']
