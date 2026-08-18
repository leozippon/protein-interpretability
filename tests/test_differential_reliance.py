"""What EXP-R2-210's causal readout must always do, and what it must always refuse.

The tests are grouped by the property they hold the instrument to rather than by
the function they call: a ground truth it must recover, a null it must read as a
null, an equivalence the packing must preserve **bitwise**, a per-layer resolution
it must never collapse, and the refusals that stop a finite number being produced
about the wrong object.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

from src.transfer import crosscoder as cc
from src.transfer import differential_reliance as dr


def _stage_33():
    """The stage module, imported the way the stages import each other.

    Cached on the function, because loading it pulls in four sibling stages and
    the transformers stack, and the two tests that need it only read one pure
    scheduling function out of it.
    """

    if getattr(_stage_33, "module", None) is None:
        path = Path(__file__).resolve().parents[1] / "scripts" / "transfer" / "33_differential_reliance.py"
        spec = importlib.util.spec_from_file_location("_reliance_stage_33", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _stage_33.module = module
    return _stage_33.module


SITE = 1
FIRE = 8
READOUT = 5
D_MODEL = 32
VOCAB = 64
WIDTH = 24


def build_pair(
    *, seed: int, gain: float, n_latents: int = 16, coefficient: float = 1.0
):
    """A paired backbone with one injected latent, ``n_latents - 1`` equal-reliance ones.

    The injected direction is the channel the adapted checkpoint reads into one
    token's logit; every other direction is drawn orthogonal to it, so ablating it
    moves that channel by exactly zero and the two checkpoints -- which share every
    weight -- rely on it equally by construction.
    """

    generator = torch.Generator().manual_seed(seed)
    injected = torch.randn(D_MODEL, generator=generator)
    injected = injected / injected.norm()
    base = dr.SyntheticPairedBackbone(
        vocab=VOCAB,
        d_model=D_MODEL,
        n_layers=3,
        seed=seed,
        reliance_site=SITE,
        reliance_gain=0.0,
        reliance_direction=injected,
        readout_token=READOUT,
    )
    batch = dr.synthetic_batch(rows=n_latents, width=WIDTH, vocab=VOCAB, seed=seed + 1)
    batch["input_ids"][:, FIRE + 1] = READOUT
    reference = float(base.reliance_channel(batch["input_ids"])[:, FIRE].mean())
    adapted = base.paired_with(reliance_gain=gain, reliance_reference=reference)
    directions = [injected]
    for _ in range(n_latents - 1):
        draw = torch.randn(D_MODEL, generator=generator)
        orthogonal = draw - (draw @ injected) * injected
        directions.append(orthogonal / orthogonal.norm())
    supports = dr.synthetic_supports(
        rows_per_latent=[[index] for index in range(n_latents)],
        positions_per_latent=[[FIRE] for _ in range(n_latents)],
        coefficient=coefficient,
    )
    return base, adapted, batch, directions, supports, reference


def reliance(base, adapted, batch, directions, supports):
    """Differential reliance per latent, through the same calls the stage makes."""

    clean_base = dr.clean_pass(base, batch, sites=[SITE])
    clean_adapted = dr.clean_pass(adapted, batch, sites=[SITE])
    by_index = dict(enumerate(directions))
    effect_base = dr.measure_pack(
        base, batch, clean_base, site=SITE, pack=supports, directions=by_index
    )
    effect_adapted = dr.measure_pack(
        adapted, batch, clean_adapted, site=SITE, pack=supports, directions=by_index
    )
    values = torch.tensor(
        [
            float((effect_adapted[index] - effect_base[index]).mean())
            for index in range(len(supports))
        ]
    )
    return values, clean_adapted


# ------------------------------------------------- the ground truth it recovers


@pytest.mark.parametrize("seed,gain", [(3, 1.0), (3, 2.0), (11, 2.0), (29, 3.0)])
def test_injected_differential_reliance_matches_its_closed_form(seed, gain):
    """The one latent whose reliance is known reads back to a fraction of a nat.

    Ablating the injected direction lowers the adapted checkpoint's readout logit
    by exactly ``gain * coefficient`` and the base checkpoint's not at all, so the
    change in negative log-likelihood at that position is
    ``delta + log(p e^-delta + 1 - p)`` in closed form. This is the test that says
    the statistic measures what it claims rather than something correlated with it.
    """

    base, adapted, batch, directions, supports, _ = build_pair(seed=seed, gain=gain)
    values, clean_adapted = reliance(base, adapted, batch, directions, supports)
    probability = float(torch.exp(-clean_adapted.nll[0, FIRE]))
    delta = gain * 1.0
    predicted = delta + math.log(probability * math.exp(-delta) + 1.0 - probability)
    assert predicted > 0.3, "the operating point saturated; the check would be vacuous"
    assert values[0] > 0.0
    assert abs(float(values[0]) - predicted) < 0.05


def test_differential_reliance_is_proportional_to_the_injected_gain():
    """Twice the injected reliance reads as twice the statistic, at a fixed operating point."""

    measured = []
    for gain in (1.0, 2.0):
        base, adapted, batch, directions, supports, _ = build_pair(seed=3, gain=gain)
        values, clean = reliance(base, adapted, batch, directions, supports)
        measured.append((float(values[0]), float(torch.exp(-clean.nll[0, FIRE]))))
    # Compared against the closed form at each cell's own clean probability rather
    # than as a bare ratio: the softmax is not linear, so the raw doubling only
    # holds where the readout token is far from saturation.
    for value, probability in measured:
        delta = 1.0 if value == measured[0][0] else 2.0
        predicted = delta + math.log(probability * math.exp(-delta) + 1.0 - probability)
        assert abs(value - predicted) < 0.05
    assert measured[1][0] > measured[0][0]


def test_equal_reliance_reads_as_no_difference():
    """Latents orthogonal to the injected channel must not read as a difference."""

    base, adapted, batch, directions, supports, _ = build_pair(seed=3, gain=2.0)
    values, _ = reliance(base, adapted, batch, directions, supports)
    equal = values[1:]
    assert float(equal.abs().max()) < 0.2
    assert float(values[0]) > 5 * float(equal.abs().max())


def test_identical_checkpoints_read_exactly_zero():
    """No injected reliance means no differential reliance, bitwise and not approximately.

    The negative control the whole statistic rests on: if two identical
    checkpoints produced a non-zero spread, every number this stage reports would
    be sitting on top of it.
    """

    base, _, batch, directions, supports, reference = build_pair(seed=3, gain=2.0)
    twin = base.paired_with(reliance_gain=0.0, reliance_reference=reference)
    clean_base = dr.clean_pass(base, batch, sites=[SITE])
    clean_twin = dr.clean_pass(twin, batch, sites=[SITE])
    by_index = dict(enumerate(directions))
    left = dr.measure_pack(
        base, batch, clean_base, site=SITE, pack=supports, directions=by_index
    )
    right = dr.measure_pack(
        twin, batch, clean_twin, site=SITE, pack=supports, directions=by_index
    )
    for index in left:
        assert torch.equal(left[index], right[index])


# ---------------------------------------------------------- the matched control


def test_matched_random_directions_carry_the_decoder_norms_exactly():
    """The control differs from the measurement in direction alone, per role."""

    torch.manual_seed(0)
    decoder = torch.randn(2, 5, D_MODEL) * torch.tensor([1.0, 4.0]).view(2, 1, 1)
    control = dr.matched_random_directions(
        decoder, seed=17, site=SITE, latents=[3, 9, 11, 12, 40]
    )
    assert torch.allclose(control.norm(dim=2), decoder.norm(dim=2), atol=1e-5)
    # One unit vector per latent, shared by the two roles: two independent draws
    # would add a difference between the roles that the measurement does not have.
    unit_base = control[0] / control[0].norm(dim=1, keepdim=True)
    unit_adapted = control[1] / control[1].norm(dim=1, keepdim=True)
    assert torch.allclose(unit_base, unit_adapted, atol=1e-6)


def test_control_direction_is_keyed_to_the_latent_not_to_its_position():
    """A latent's control direction must not depend on what was measured beside it."""

    torch.manual_seed(0)
    decoder = torch.randn(2, 4, D_MODEL)
    full = dr.matched_random_directions(decoder, seed=5, site=SITE, latents=[7, 8, 9, 10])
    subset = dr.matched_random_directions(
        decoder[:, 2:3], seed=5, site=SITE, latents=[9]
    )
    assert torch.allclose(full[:, 2], subset[:, 0], atol=0.0)


def test_matched_control_is_a_scale_reference_and_not_signal():
    """The control must read zero where there is nothing to find, and bracket the nulls.

    Two halves. On identical checkpoints the control's differential effect is
    exactly zero, so it contributes no spurious signal of its own. On the injected
    pair it is of the same order as the equal-reliance latents and far below the
    injected one, which is what makes it a floor rather than a ceiling.
    """

    base, adapted, batch, directions, supports, reference = build_pair(seed=3, gain=2.0)
    stacked = torch.stack([torch.stack(directions), torch.stack(directions)])
    control = dr.matched_random_directions(
        stacked, seed=99, site=SITE, latents=list(range(len(directions)))
    )
    twin = base.paired_with(reliance_gain=0.0, reliance_reference=reference)
    for left, right in ((base, twin),):
        clean_left = dr.clean_pass(left, batch, sites=[SITE])
        clean_right = dr.clean_pass(right, batch, sites=[SITE])
        first = dr.measure_pack(
            left, batch, clean_left, site=SITE, pack=supports,
            directions={index: control[0, index] for index in range(len(directions))},
        )
        second = dr.measure_pack(
            right, batch, clean_right, site=SITE, pack=supports,
            directions={index: control[1, index] for index in range(len(directions))},
        )
        for index in first:
            assert torch.equal(first[index], second[index])

    clean_base = dr.clean_pass(base, batch, sites=[SITE])
    clean_adapted = dr.clean_pass(adapted, batch, sites=[SITE])
    control_base = dr.measure_pack(
        base, batch, clean_base, site=SITE, pack=supports,
        directions={index: control[0, index] for index in range(len(directions))},
    )
    control_adapted = dr.measure_pack(
        adapted, batch, clean_adapted, site=SITE, pack=supports,
        directions={index: control[1, index] for index in range(len(directions))},
    )
    random_arm = torch.tensor(
        [
            float((control_adapted[index] - control_base[index]).mean())
            for index in range(len(directions))
        ]
    )
    values, _ = reliance(base, adapted, batch, directions, supports)
    assert float(values[0]) > 3 * float(random_arm.std())
    assert float(random_arm.abs().max()) < float(values[0])


# ------------------------------------------- the packing equivalence, bitwise


def test_packing_is_bitwise_identical_to_single_latent_passes():
    """The correctness crux: packing is an optimisation only if it changes nothing.

    Row-disjoint latents share one forward pass. Rows of a batch do not interact,
    so each latent's per-position effect must come back **bitwise** equal to the
    value its own pass would have produced.
    """

    base, adapted, batch, directions, supports, _ = build_pair(seed=3, gain=2.0)
    clean = dr.clean_pass(adapted, batch, sites=[SITE])
    by_index = dict(enumerate(directions))
    packs = dr.pack_disjoint_supports(supports)
    assert len(packs) == 1 and len(packs[0]) == len(supports)
    packed = dr.measure_pack(
        adapted, batch, clean, site=SITE, pack=supports, directions=by_index
    )
    for support in supports:
        alone = dr.measure_pack(
            adapted, batch, clean, site=SITE, pack=[support], directions=by_index
        )
        assert torch.equal(packed[support.latent], alone[support.latent])


def test_position_disjoint_latents_in_one_row_are_not_equivalent():
    """Why the conflict rule is at row granularity, demonstrated rather than argued.

    Two latents firing at different positions of the **same** sequence have
    disjoint supports in the position sense EXP-R2-210 wrote down. Packing them
    changes the answer for the later one, because the earlier ablation reaches it
    through the attention of every layer above the site -- and leaves the earlier
    one untouched, because causal attention does not run backwards. That
    asymmetry is the signature of the mechanism and is asserted here so the rule
    cannot be relaxed by someone who reads the position wording alone.
    """

    base, adapted, batch, directions, _, _ = build_pair(seed=3, gain=2.0)
    early, late = 3, 12
    supports = dr.synthetic_supports(
        rows_per_latent=[[0], [0]], positions_per_latent=[[early], [late]]
    )
    clean = dr.clean_pass(adapted, batch, sites=[SITE])
    by_index = {0: directions[0], 1: directions[1]}
    packed = dr.measure_pack(
        adapted, batch, clean, site=SITE, pack=supports, directions=by_index
    )
    alone = {}
    for support in supports:
        alone.update(
            dr.measure_pack(
                adapted, batch, clean, site=SITE, pack=[support], directions=by_index
            )
        )
    assert torch.equal(packed[0], alone[0]), "an ablation must not reach an earlier position"
    assert not torch.equal(packed[1], alone[1]), (
        "an ablation at an earlier position of the same sequence must reach a later "
        "one; if this passes, the packing rule is being tested on a model with no "
        "cross-position mixing and proves nothing"
    )
    assert dr.pack_disjoint_supports(supports) == [(0,), (1,)]


def test_greedy_colouring_is_deterministic_and_row_disjoint():
    supports = dr.synthetic_supports(
        rows_per_latent=[[0, 1], [1, 2], [2, 3], [4], [0]],
        positions_per_latent=[[2, 3], [3, 4], [4, 5], [6], [7]],
    )
    packs = dr.pack_disjoint_supports(supports)
    assert packs == dr.pack_disjoint_supports(supports)
    for pack in packs:
        seen: set[int] = set()
        for member in pack:
            rows = supports[member].row_set
            assert seen.isdisjoint(rows)
            seen |= rows


def test_the_pass_refuses_two_latents_at_one_position():
    """A schedule and an intervention that disagree must stop, not silently accumulate."""

    base, _, batch, directions, _, _ = build_pair(seed=3, gain=2.0)
    rows = torch.tensor([0, 0])
    positions = torch.tensor([FIRE, FIRE])
    deltas = torch.stack([directions[0], directions[1]])
    with pytest.raises(ValueError, match="same \\(row, position\\)"):
        with dr.subtracted_at(
            base, site=SITE, rows=rows, positions=positions, deltas=deltas
        ):
            pass


def test_an_uninterceptable_site_raises_rather_than_reporting_no_effect():
    base, _, batch, directions, supports, _ = build_pair(seed=3, gain=2.0)
    with pytest.raises(RuntimeError, match="never intercepted"):
        with dr.subtracted_at(
            base,
            site=99,
            rows=supports[0].rows,
            positions=supports[0].positions,
            deltas=dr.ablation_deltas(supports[0], directions[0]),
        ):
            base.scored_logits(batch)


def test_determinism_under_a_fixed_seed():
    base, adapted, batch, directions, supports, _ = build_pair(seed=3, gain=2.0)
    first, _ = reliance(base, adapted, batch, directions, supports)
    second, _ = reliance(base, adapted, batch, directions, supports)
    assert torch.equal(first, second)


# ----------------------------------------------- supports, directions, cost


def test_ablation_directions_are_the_scaled_decoder_rows():
    """What is subtracted is what the dictionary writes, including its frozen scale."""

    config = cc.CrosscoderConfig(sites=(4, 7), d_model=6, d_hidden=5, k=2, auxk=2, dead_steps=1)
    model = cc.Crosscoder(config, init_seed=1)
    model.set_scales(torch.tensor([[2.0, 3.0], [5.0, 7.0]]))
    directions = dr.ablation_directions(model, site=7)
    assert directions.shape == (2, 5, 6)
    assert torch.allclose(directions[0], model.W_dec[0, 1] * 3.0)
    assert torch.allclose(directions[1], model.W_dec[1, 1] * 7.0)
    with pytest.raises(ValueError, match="no parameters for layer"):
        dr.ablation_directions(model, site=5)


def test_latent_supports_keeps_only_live_latents_that_actually_fire():
    latents = torch.zeros(4, 6)
    latents[0, 1] = 0.5
    latents[2, 1] = 1.5
    latents[1, 3] = 2.0
    latents[3, 5] = 9.0  # not in `keep`
    rows = torch.tensor([0, 0, 1, 1])
    positions = torch.tensor([2, 5, 3, 4])
    supports = dr.latent_supports(latents, rows=rows, positions=positions, keep=[1, 3, 4])
    assert sorted(supports) == [1, 3]
    assert supports[1].rows.tolist() == [0, 1]
    assert supports[1].positions.tolist() == [2, 3]
    assert supports[1].coefficients.tolist() == [0.5, 1.5]
    assert supports[1].row_set == frozenset({0, 1})
    assert supports[3].size == 1


def test_latent_supports_refuses_a_ragged_batch():
    with pytest.raises(ValueError, match="rows of latent activations"):
        dr.latent_supports(
            torch.zeros(3, 4), rows=torch.tensor([0, 1]), positions=torch.tensor([0, 1]),
            keep=[0],
        )


def test_packed_cost_counts_both_checkpoints_and_both_arms():
    """The two factors the pre-registered estimate dropped are both in the arithmetic."""

    full = dr.packed_cost(
        live_latents=11_691, cohort_rows=128, rows_per_latent=128, batch_rows=4
    )
    assert full["naive_forward_passes"] == full["packed_forward_passes"]
    assert full["naive_forward_passes"] == 11_691 * 128 * 2 * 2 // 4
    sliced = dr.packed_cost(
        live_latents=11_691, cohort_rows=128, rows_per_latent=8, batch_rows=64
    )
    assert sliced["speedup_over_naive"] == 16.0
    # A pass carries at most cohort_rows / rows_per_latent different latents,
    # whatever the batch is: at eight rows each on a 128-row cohort, sixteen
    # latents exhaust the disjoint row budget and a wider batch buys nothing more.
    assert sliced["latents_per_pass_ceiling"] == 16
    dense = dr.packed_cost(
        live_latents=11_691, cohort_rows=128, rows_per_latent=2, batch_rows=64
    )
    assert dense["latents_per_pass_ceiling"] == 64
    assert dense["one_row_per_batch_achievable"] is True
    with pytest.raises(ValueError, match="cannot be measured on"):
        dr.packed_cost(
            live_latents=10, cohort_rows=8, rows_per_latent=16, batch_rows=4
        )


def test_accumulator_never_averages_an_unmeasured_latent_to_zero():
    accumulator = dr.RelianceAccumulator([4, 9])
    accumulator.update({4: torch.tensor([1.0, 3.0])})
    mean = accumulator.mean()
    assert float(mean[0]) == 2.0
    assert math.isnan(float(mean[1]))
    assert math.isnan(float(dr.differential_reliance(mean, mean)[1]))
    assert float(dr.differential_reliance(mean, mean)[0]) == 0.0


# -------------------------------------------------- persistence and refusals


def _fitted(tmp_path: Path, **overrides):
    config = cc.CrosscoderConfig(sites=(27, 28), d_model=8, d_hidden=6, k=2, auxk=2, dead_steps=1)
    model = cc.Crosscoder(config, init_seed=3)
    model.set_scales(torch.tensor([[1.5, 2.5], [3.5, 4.5]]))
    path = tmp_path / "dictionary.pt"
    dr.save_crosscoder(
        path,
        model,
        **{
            "backbone_pair_sha256": "abc123",
            "mode": "text",
            "tensor": "block_output",
            **overrides,
        },
    )
    return model, path


def test_a_dictionary_round_trips_with_its_scales_and_its_pair(tmp_path):
    model, path = _fitted(tmp_path)
    loaded, manifest = dr.load_crosscoder(path)
    assert bool(loaded.scale_is_set)
    assert torch.equal(loaded.scale, model.scale)
    assert torch.equal(loaded.W_dec, model.W_dec)
    assert manifest["backbone_pair_sha256"] == "abc123"
    assert manifest["config"]["sites"] == [27, 28]
    dr.assert_dictionary_matches(
        manifest,
        backbone_pair_sha256="abc123",
        mode="text",
        tensor="block_output",
        d_model=8,
        n_layers=32,
    )


def test_an_unscaled_dictionary_is_refused_before_it_is_written(tmp_path):
    config = cc.CrosscoderConfig(sites=(1,), d_model=4, d_hidden=3, k=1, auxk=1, dead_steps=1)
    with pytest.raises(ValueError, match="normalisation scales were never frozen"):
        dr.save_crosscoder(
            tmp_path / "d.pt",
            cc.Crosscoder(config),
            backbone_pair_sha256="x",
            mode="text",
            tensor="block_output",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("backbone_pair_sha256", "different"),
        ("mode", "protein"),
        ("tensor", "block_input"),
    ],
)
def test_a_dictionary_that_names_other_checkpoints_is_refused(tmp_path, field, value):
    _, path = _fitted(tmp_path)
    _, manifest = dr.load_crosscoder(path)
    kwargs = {
        "backbone_pair_sha256": "abc123",
        "mode": "text",
        "tensor": "block_output",
        "d_model": 8,
        "n_layers": 32,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match="fitted to"):
        dr.assert_dictionary_matches(manifest, **kwargs)


def test_a_dictionary_of_the_wrong_width_or_depth_is_refused(tmp_path):
    _, path = _fitted(tmp_path)
    _, manifest = dr.load_crosscoder(path)
    with pytest.raises(ValueError, match="dimensional"):
        dr.assert_dictionary_matches(
            manifest, backbone_pair_sha256="abc123", mode="text",
            tensor="block_output", d_model=4096, n_layers=32,
        )
    with pytest.raises(ValueError, match="outside the backbone"):
        dr.assert_dictionary_matches(
            manifest, backbone_pair_sha256="abc123", mode="text",
            tensor="block_output", d_model=8, n_layers=12,
        )


def test_a_dictionary_fitted_on_another_cohort_is_refused(tmp_path):
    """The held-out offset is steps x batch size, so those are properties of the fit.

    The check is against whatever the dictionary recorded, and it must not fire on
    fields it did not record -- otherwise it would refuse every dictionary written
    before the trainer started recording them.
    """

    _, path = _fitted(
        tmp_path, extra={"steps": 26_000, "fit_batch_size": 4, "eval_sequences": 256}
    )
    _, manifest = dr.load_crosscoder(path)
    common = {
        "backbone_pair_sha256": "abc123",
        "mode": "text",
        "tensor": "block_output",
        "d_model": 8,
        "n_layers": 32,
    }
    dr.assert_dictionary_matches(
        manifest,
        **common,
        cohort={"steps": 26_000, "fit_batch_size": 4, "eval_sequences": 256},
    )
    # A field the dictionary never recorded cannot make it disagree.
    dr.assert_dictionary_matches(
        manifest, **common, cohort={"steps": 26_000, "corpus_seed": 20260812}
    )
    with pytest.raises(ValueError, match="fitted, requested"):
        dr.assert_dictionary_matches(
            manifest, **common, cohort={"steps": 20_000, "fit_batch_size": 4}
        )
    with pytest.raises(ValueError, match="fitted, requested"):
        dr.assert_dictionary_matches(
            manifest, **common, cohort={"fit_batch_size": 16}
        )


def test_a_file_that_is_not_a_dictionary_is_refused(tmp_path):
    path = tmp_path / "not.pt"
    torch.save({"schema": "something_else"}, path)
    with pytest.raises(ValueError, match="not a r2_transfer_crosscoder_state_v1"):
        dr.load_crosscoder(path)


def test_a_partial_backbone_is_refused():
    class Half:
        def scored_logits(self, batch):  # pragma: no cover - never reached
            raise AssertionError

    with pytest.raises(TypeError, match="block_intercept"):
        dr.assert_backbone(Half(), role="base")


# --------------------------------------------------- the per-layer discipline


def test_a_collapsed_per_site_field_is_refused():
    """The defect that voided a pre-registered criterion, held against this artefact.

    A ``(num_layers, d_hidden)`` mask was once summed to one cross-layer scalar
    before it reached any artefact, so a criterion stated per layer had never been
    evaluated per layer. Every per-site key of this stage is walked by the same
    guard, and this test fails if one is written as a mean.
    """

    good = {
        "site_per_site": [{"layer": 27}, {"layer": 28}],
        "live_latents_per_site": [5558, 6133],
        "measured_latents_per_site": [5558, 6133],
        "passes_per_site": [11, 13],
        "mean_cohort_rows_per_live_latent_per_site": [108.4, 101.2],
    }
    cc.assert_per_layer_fields(good, n_sites=2)
    dr.assert_required_per_site_fields(good)

    collapsed = {**good, "live_latents_per_site": 5845.5}
    with pytest.raises(ValueError, match="reduced to a scalar"):
        cc.assert_per_layer_fields(collapsed, n_sites=2)

    shortened = {**good, "passes_per_site": [11]}
    with pytest.raises(ValueError, match="carries 1 values for 2 fitted sites"):
        cc.assert_per_layer_fields(shortened, n_sites=2)

    nested = {
        **good,
        "packing": {"detail": {"latents_per_site": 42}},
    }
    with pytest.raises(ValueError, match="reduced to a scalar"):
        cc.assert_per_layer_fields(nested, n_sites=2)


def test_a_missing_per_site_field_is_refused():
    with pytest.raises(ValueError, match="carries no"):
        dr.assert_required_per_site_fields(
            {"site_per_site": [{"layer": 27}], "live_latents_per_site": [1]}
        )


def test_the_packing_sizing_input_is_required_not_optional():
    """Cohort-row occupancy decides whether packing packs, so it may not be omitted.

    It is the number that refuted the registered packing saving. An artefact that
    dropped it would let the next campaign re-derive a saving that is not there,
    which is why it sits in the required list beside the counts rather than in a
    diagnostics block a reader may skip.
    """

    assert "mean_cohort_rows_per_live_latent_per_site" in dr.REQUIRED_PER_SITE_FIELDS
    without = {
        "site_per_site": [{"layer": 27}],
        "live_latents_per_site": [1],
        "measured_latents_per_site": [1],
        "passes_per_site": [1],
    }
    with pytest.raises(ValueError, match="mean_cohort_rows_per_live_latent_per_site"):
        dr.assert_required_per_site_fields(without)


def test_row_assignment_takes_the_declared_number_of_distinct_rows():
    """A latent measured on fewer rows than declared is an unannounced budget cut.

    The stride must stay strictly increasing right up to ``rows_per_latent ==
    n``; a rounded linspace collapses to duplicates there and would quietly
    measure the latent on a smaller slice than the artefact says.
    """

    stage = _stage_33()
    hits = torch.zeros(16, 3, dtype=torch.bool)
    hits[:, 0] = True
    hits[[1, 4, 9], 1] = True
    for wanted in range(1, 17):
        assigned = stage.assign_rows(hits, rows_per_latent=wanted)
        assert int(assigned[:, 0].sum()) == wanted
        # A latent that fires in fewer rows than asked for keeps all of them.
        assert int(assigned[:, 1].sum()) == min(wanted, 3)
        assert int(assigned[:, 2].sum()) == 0
    # Zero means the pre-registered behaviour: every row the latent fires in.
    assert torch.equal(stage.assign_rows(hits, rows_per_latent=0), hits)


def test_row_assignment_spreads_across_batches_rather_than_taking_a_prefix():
    """Spread is what makes packing dense; a prefix would crowd one batch."""

    stage = _stage_33()
    hits = torch.ones(32, 1, dtype=torch.bool)
    rows = stage.assign_rows(hits, rows_per_latent=4)[:, 0].nonzero().flatten().tolist()
    assert rows == [0, 8, 16, 24]


def test_admissibility_is_an_input_and_is_never_widened():
    """A layer the dictionary does not carry cannot be reported at, silently or otherwise."""

    assert cc.assert_admissible_subset([28], [27, 28]) == (28,)
    with pytest.raises(ValueError, match="were not fitted"):
        cc.assert_admissible_subset([29], [27, 28])
    with pytest.raises(ValueError, match="refusal to run"):
        cc.assert_admissible_subset([], [27, 28])
