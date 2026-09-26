"""The two declared control transplants, and what each must leave behind.

The null control must leave the recipient bit-identical, which is what shows the
intervention machinery contributes nothing of its own. The scrambled control must
install the donor's tensors at a declared derangement of same-shape destinations,
which is what gives "the effect is above control uncertainty" a magnitude instead
of an assumption. Both are exercised against real safetensors checkpoints and a
real census rather than against stubs of themselves.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transfer import capability_transplant as ct

# Eight layers so each declared depth quarter holds two, which is the smallest
# size at which an MLP quarter's two shape classes both hold enough tensors to
# be deranged -- the real 32-layer checkpoint gives classes of 16 and 8.
LAYERS, HIDDEN, INTERMEDIATE, VOCAB = 8, 8, 16, 10

CONFIG = {
    'model_type': 'llama', 'hidden_size': HIDDEN, 'intermediate_size': INTERMEDIATE,
    'num_hidden_layers': LAYERS, 'num_attention_heads': 4, 'num_key_value_heads': 4,
    'vocab_size': VOCAB, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0, 'rope_scaling': None,
    'max_position_embeddings': 32, 'hidden_act': 'silu', 'attention_bias': False,
    'mlp_bias': False, 'tie_word_embeddings': False,
}


def tensor_shapes() -> dict[str, tuple[int, ...]]:
    shapes = {'model.embed_tokens.weight': (VOCAB, HIDDEN),
              'lm_head.weight': (VOCAB, HIDDEN),
              'model.norm.weight': (HIDDEN,)}
    for layer in range(LAYERS):
        prefix = f'model.layers.{layer}'
        for key in ('q', 'k', 'v', 'o'):
            shapes[f'{prefix}.self_attn.{key}_proj.weight'] = (HIDDEN, HIDDEN)
        shapes[f'{prefix}.mlp.gate_proj.weight'] = (INTERMEDIATE, HIDDEN)
        shapes[f'{prefix}.mlp.up_proj.weight'] = (INTERMEDIATE, HIDDEN)
        shapes[f'{prefix}.mlp.down_proj.weight'] = (HIDDEN, INTERMEDIATE)
        shapes[f'{prefix}.input_layernorm.weight'] = (HIDDEN,)
        shapes[f'{prefix}.post_attention_layernorm.weight'] = (HIDDEN,)
    return shapes


def write_checkpoint(root: Path, *, seed: int, shared_norms: dict | None = None) -> dict:
    """A tiny Llama-shaped safetensors checkpoint with an index."""
    from safetensors.torch import save_file

    root.mkdir(parents=True, exist_ok=True)
    (root / 'config.json').write_text(json.dumps(CONFIG))
    (root / 'tokenizer.model').write_bytes(b'tokenizer')
    generator = torch.Generator().manual_seed(seed)
    state = {}
    for name, shape in tensor_shapes().items():
        if shared_norms is not None and 'norm' in name:
            state[name] = shared_norms[name].clone()
        else:
            state[name] = torch.randn(*shape, generator=generator).to(torch.float32)
    save_file(state, str(root / 'model.safetensors'))
    (root / 'model.safetensors.index.json').write_text(json.dumps(
        {'weight_map': {name: 'model.safetensors' for name in state}}))
    return state


class StubModel:
    """Only what a transplant touches: named parameters that can be copied into."""

    def __init__(self, state: dict[str, torch.Tensor]):
        self._state = {name: value.clone().to(torch.float32) for name, value in state.items()}

    def named_parameters(self):
        return list(self._state.items())

    def named_buffers(self):
        return []


@pytest.fixture(scope='module')
def pair(tmp_path_factory):
    root = tmp_path_factory.mktemp('pair')
    recipient = root / 'recipient'
    donor = root / 'donor'
    state = write_checkpoint(recipient, seed=11)
    norms = {name: value for name, value in state.items() if 'norm' in name}
    # Norms shared so `norm` is byte-identical, as it is in the real lineage.
    write_checkpoint(donor, seed=22, shared_norms=norms)
    census = ct.census(recipient, donor)
    return recipient, donor, census, state


def test_the_census_of_the_synthetic_pair_matches_the_real_shape(pair):
    _, _, census, _ = pair
    assert census['groups']['norm']['fp32_bytes_equal'] is True
    assert census['groups']['attention_q1']['fp32_bytes_equal'] is False
    assert len(census['tensors']) == len(tensor_shapes())


def test_the_permutation_is_a_seeded_derangement_within_shape_classes(pair):
    _, _, census, _ = pair
    mapping = ct.destination_permutation(census, ['attention_q1'])
    assert mapping == ct.destination_permutation(census, ['attention_q1'])
    assert all(destination != source for destination, source in mapping.items())
    shapes = {tuple(row['shape']) for row in census['tensors']}
    assert shapes  # sanity
    rows = {row['name']: row for row in census['tensors']}
    for destination, source in mapping.items():
        assert rows[destination]['shape'] == rows[source]['shape']
    assert set(mapping) == set(mapping.values())
    other = ct.destination_permutation(census, ['attention_q1'], seed=ct.SCRAMBLE_SEED + 1)
    assert other != mapping


def test_a_single_tensor_group_cannot_be_scrambled(pair):
    """The embedding and the head are one tensor each; a derangement does not exist."""
    _, _, census, _ = pair
    for group in ('embedding', 'head'):
        with pytest.raises(ValueError, match='cannot be deranged'):
            ct.destination_permutation(census, [group])


def test_the_mlp_permutation_respects_two_shape_classes(pair):
    _, _, census, _ = pair
    classes = ct.shape_classes(census, census['groups']['mlp_q1']['tensor_names'])
    assert len(classes) == 2
    mapping = ct.destination_permutation(census, ['mlp_q1'])
    rows = {row['name']: row for row in census['tensors']}
    for destination, source in mapping.items():
        assert rows[destination]['shape'] == rows[source]['shape']


def test_the_null_control_leaves_the_recipient_bit_identical(pair):
    recipient, _, census, state = pair
    model = StubModel(state)
    receipt = ct.transplant_null(model, census, ['attention_q1'])
    assert receipt['control'] == 'null'
    assert receipt['n_transplanted_tensors'] == 8
    assert 'reported as a no-op' in receipt['byte_identity_refusal']
    expected = ct.census_expectations(census, ())
    for name, digest in receipt['verified_tensor_fp32_sha256'].items():
        assert digest == expected[name]
    # And the floor's own selection is the same state: a null over any group is
    # indistinguishable from transplanting nothing.
    untouched = ct.transplant(StubModel(state), recipient, census, ())
    assert (receipt['verified_tensor_fp32_sha256']
            == untouched['verified_tensor_fp32_sha256'])


def test_the_null_control_refuses_a_model_that_is_not_the_recipient(pair):
    """If the loaded state was not the recipient, the null must refuse, not pass."""
    _, _, census, state = pair
    perturbed = {name: value.clone() for name, value in state.items()}
    perturbed['model.layers.3.mlp.down_proj.weight'] += 1.0
    with pytest.raises(ValueError, match='did not reproduce the recipient'):
        ct.transplant_null(StubModel(perturbed), census, ['attention_q1'])


def test_the_scrambled_control_installs_the_donor_at_permuted_slots(pair):
    _, donor, census, state = pair
    model = StubModel(state)
    receipt = ct.transplant_scrambled(model, donor, census, ['attention_q1'])
    assert receipt['control'] == 'scrambled'
    assert receipt['permutation_seed'] == ct.SCRAMBLE_SEED
    assert receipt['fixed_points'] == 0
    assert receipt['n_transplanted_tensors'] == 8
    assert 'not a symmetric error bar' in receipt['measures']
    rows = {row['name']: row for row in census['tensors']}
    observed = receipt['verified_tensor_fp32_sha256']
    for destination, source in receipt['permutation'].items():
        # The slot carries the donor's tensor it received, not its own.
        assert observed[destination] == rows[source]['fp32_sha256'][1]
        assert observed[destination] != rows[destination]['fp32_sha256'][0]
    # Every tensor outside the selection is still the recipient's.
    for name, digest in observed.items():
        if name not in receipt['permutation']:
            assert digest == rows[name]['fp32_sha256'][0]


def test_the_scrambled_control_changes_the_same_parameter_count_as_its_treatment(pair):
    """Size-matched by construction, which is what makes it the right control."""
    _, donor, census, state = pair
    scrambled = ct.transplant_scrambled(StubModel(state), donor, census, ['attention_q1'])
    treatment = ct.transplant(StubModel(state), donor, census, ['attention_q1'])
    assert (scrambled['n_transplanted_elements']
            == treatment['n_transplanted_elements']
            == census['groups']['attention_q1']['n_elements'])


def test_an_undeclared_group_is_refused_by_both_controls(pair):
    _, donor, census, state = pair
    with pytest.raises(ValueError, match='undeclared groups'):
        ct.transplant_null(StubModel(state), census, ['nowhere'])
    with pytest.raises(ValueError, match='undeclared groups'):
        ct.transplant_scrambled(StubModel(state), donor, census, ['nowhere'])


def test_the_spectrum_matched_control_keeps_the_spectrum_and_destroys_the_content(pair):
    """The control that separates a scale effect from a content effect."""
    _, donor, census, state = pair
    import torch
    from safetensors import safe_open

    model = StubModel(state)
    receipt = ct.transplant_spectrum_matched(model, donor, census, ['attention_q1'])
    assert receipt['control'] == 'spectrum_matched'
    assert receipt['spectrum_seed'] == ct.SPECTRUM_MATCHED_SEED
    assert receipt['n_transplanted_tensors'] == 8
    assert receipt['max_relative_spectrum_departure'] < ct.SPECTRUM_TOLERANCE
    rows = {row['name']: row for row in census['tensors']}
    names = census['groups']['attention_q1']['tensor_names']
    for name in names:
        live = model._state[name]
        with safe_open(donor / 'model.safetensors', framework='pt', device='cpu') as handle:
            original = handle.get_tensor(name).to(torch.float32)
        # The spectrum is the donor's ...
        assert torch.allclose(torch.linalg.svdvals(live), torch.linalg.svdvals(original),
                              rtol=1e-3, atol=1e-4)
        # ... the Frobenius norm follows from it ...
        assert float(torch.linalg.matrix_norm(live)) == pytest.approx(
            float(torch.linalg.matrix_norm(original)), rel=1e-3)
        # ... and no direction survives: the synthesised tensor is uncorrelated
        # with the donor's and with the recipient's. The bound scales with the
        # element count rather than being a fixed number, because the sampling
        # standard error of a correlation between two uncorrelated vectors of n
        # elements is 1/sqrt(n) -- 0.125 for these 8x8 fixtures and 0.00024 for
        # the 4096-square tensors this control actually runs on.
        for other in (original, torch.as_tensor(state[name])):
            flat_a, flat_b = live.reshape(-1), other.reshape(-1).to(torch.float32)
            correlation = float(torch.corrcoef(torch.stack([flat_a, flat_b]))[0, 1])
            assert abs(correlation) < 4.0 / (flat_a.numel() ** 0.5)
        # And it is neither checkpoint's stored tensor.
        digest = receipt['verified_tensor_fp32_sha256'][name]
        assert digest not in (rows[name]['fp32_sha256'][0], rows[name]['fp32_sha256'][1])
    # Every tensor outside the selection is untouched.
    for name, digest in receipt['verified_tensor_fp32_sha256'].items():
        if name not in names:
            assert digest == rows[name]['fp32_sha256'][0]


def test_the_spectrum_matched_control_is_size_matched_to_its_treatment(pair):
    _, donor, census, state = pair
    receipt = ct.transplant_spectrum_matched(StubModel(state), donor, census, ['attention_q1'])
    treatment = ct.transplant(StubModel(state), donor, census, ['attention_q1'])
    assert (receipt['n_transplanted_elements']
            == treatment['n_transplanted_elements']
            == census['groups']['attention_q1']['n_elements'])


def test_the_spectrum_matched_control_is_seed_deterministic(pair):
    _, donor, census, state = pair
    first = ct.transplant_spectrum_matched(StubModel(state), donor, census, ['attention_q1'])
    again = ct.transplant_spectrum_matched(StubModel(state), donor, census, ['attention_q1'])
    assert (first['verified_tensor_fp32_sha256'] == again['verified_tensor_fp32_sha256'])
    other = ct.transplant_spectrum_matched(StubModel(state), donor, census, ['attention_q1'],
                                           seed=ct.SPECTRUM_MATCHED_SEED + 1)
    assert other['verified_tensor_fp32_sha256'] != first['verified_tensor_fp32_sha256']


def test_the_spectrum_matched_control_refuses_a_non_matrix_group(pair):
    """`norm` holds one-dimensional tensors; a spectrum is undefined for them."""
    _, donor, census, state = pair
    with pytest.raises(ValueError, match='defined for matrices'):
        ct.transplant_spectrum_matched(StubModel(state), donor, census, ['norm'])
