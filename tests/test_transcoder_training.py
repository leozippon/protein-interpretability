"""Tests for the CLT/PLT trainer's numerics.

598 lines of purely numerical, CPU-testable code had no test at all, and the
consequence was a real defect that trained to completion four times and reported
a plausible float: the auxiliary loss compared a prediction in normalised space
against a target in raw activation space. The first test below is the one that
would have caught it, and it is written as a *dimensional* check rather than a
value check, because the defect was a units error and units errors are what
dimensional checks find.
"""

from __future__ import annotations

import json

import pytest
import torch

from src.transfer.transcoders import (
    DEAD_STEPS_SEQUENCES,
    MATCHED_TRAINING_FIELDS,
    MATCHED_TRAINING_KEY,
    MatchedTraining,
    Transcoder,
    TranscoderConfig,
    TranscoderReplacement,
    compare_matched_training,
    matched_training,
    matched_training_from_artefact,
    normalise,
    topk_relu,
)


def _tiny(cross_layer: bool = True, **overrides) -> TranscoderConfig:
    settings = {
        "num_layers": 3,
        "d_model": 8,
        "d_hidden": 16,
        "k": 4,
        "auxk": 4,
        "dead_steps": 0,  # every latent counts as dead after one step
        "cross_layer": cross_layer,
    }
    settings.update(overrides)
    return TranscoderConfig(**settings)


def _batch(config: TranscoderConfig, *, scale: float, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    shape = (config.num_layers, 64, config.d_model)
    inputs = torch.randn(shape, generator=generator) * scale
    targets = torch.randn(shape, generator=generator) * scale
    return inputs, targets


def test_the_auxiliary_prediction_is_de_normalised_before_it_meets_its_target() -> None:
    """It is compared against a residual in raw activation space, so it must be there too.

    Pinned by construction rather than by a scaling argument -- a scaling
    argument does not discriminate here, because the raw-space *target*
    dominates the mean squared error and carries the activation scale on its own
    whichever space the prediction is in.

    With every decoder and every ``b_pre`` zeroed, the reconstruction is exactly
    the input's per-token mean, and so is a correctly de-normalised auxiliary
    prediction. A prediction left in normalised space is exactly **zero**. The
    two therefore differ by the whole location of the activations, which is
    directly assertable and needs no tolerance argument. Giving the inputs a
    large offset makes the gap unmistakable.
    """

    config = _tiny(dead_steps=0)
    torch.manual_seed(0)
    model = Transcoder(config)
    with torch.no_grad():
        for parameter in model.decoders.values():
            parameter.zero_()
        model.b_pre.zero_()

    # With the decoders and b_pre zeroed, the reconstruction is exactly the
    # input's per-token mean, and so is a de-normalised auxiliary prediction.
    # Choosing the target as twice that mean makes the residual equal to the
    # mean as well -- so the correct prediction hits its target exactly and the
    # auxiliary loss is zero. A prediction left in normalised space is zero
    # instead, and misses by the whole mean.
    inputs, _ = _batch(config, scale=1.0)
    inputs = inputs + 10.0
    location = inputs.mean(dim=-1, keepdim=True).expand_as(inputs)
    targets = 2.0 * location

    model.objective(inputs, targets, training=True)  # first pass marks latents silent
    report = model.objective(inputs, targets, training=True)
    assert report["n_dead"] > 0, "the fixture must actually exercise the aux path"

    missed_by_normalised_space = float((location**2).mean())
    assert missed_by_normalised_space > 1.0, "the fixture does not separate the two forms"
    assert float(report["aux"]) < 1e-6 * missed_by_normalised_space, (
        f"aux is {float(report['aux']):.6g} where a de-normalised prediction must "
        f"give 0; a prediction left in normalised space would give about "
        f"{missed_by_normalised_space:.6g} per contributing layer"
    )


def test_the_auxiliary_term_is_zero_when_nothing_is_dead() -> None:
    config = _tiny(dead_steps=10_000)
    torch.manual_seed(0)
    model = Transcoder(config)
    inputs, targets = _batch(config, scale=1.0)
    report = model.objective(inputs, targets, training=True)
    assert report["n_dead"] == 0
    assert float(report["aux"]) == 0.0


def test_evaluation_never_touches_the_dead_latent_bookkeeping() -> None:
    config = _tiny()
    torch.manual_seed(0)
    model = Transcoder(config)
    inputs, targets = _batch(config, scale=1.0)
    before = model.silent_steps.clone()
    report = model.objective(inputs, targets, training=False)
    assert torch.equal(model.silent_steps, before)
    assert float(report["aux"]) == 0.0


def test_the_cross_layer_arm_writes_downstream_and_the_per_layer_arm_does_not() -> None:
    clt = Transcoder(_tiny(cross_layer=True))
    plt = Transcoder(_tiny(cross_layer=False))
    assert clt.writes_to == {0: [0, 1, 2], 1: [1, 2], 2: [2]}
    assert plt.writes_to == {0: [0], 1: [1], 2: [2]}
    # L(L+1)/2 against L: the parameter advantage the comparison has to declare.
    assert len(clt.pairs) == 6
    assert len(plt.pairs) == 3


def test_the_parameter_count_is_the_one_the_model_actually_builds() -> None:
    """The closed form sizes an arm before it runs, so it must match the module.

    A parameter-matched PLT control is chosen from this arithmetic. If the
    closed form and the built module disagreed, the control would be matched to
    a number no run has.
    """

    for cross_layer in (True, False):
        for d_hidden in (16, 64):
            config = _tiny(cross_layer=cross_layer, d_hidden=d_hidden)
            built = sum(p.numel() for p in Transcoder(config).parameters())
            assert config.n_parameters() == built, config.record()

    # And at the real shape, which is where the 3.25x advantage is claimed.
    clt = TranscoderConfig(cross_layer=True)
    plt = TranscoderConfig(cross_layer=False)
    assert clt.n_parameters() == 115_065_600
    assert plt.n_parameters() == 35_439_360


def test_topk_relu_zeroes_a_selected_negative_pre_activation() -> None:
    """Their order, not the other one: a latent can be selected and contribute nothing."""

    pre = torch.tensor([[-3.0, -1.0, -2.0, -4.0]])
    out = topk_relu(pre, k=2)
    assert torch.equal(out, torch.zeros_like(pre))

    mixed = torch.tensor([[5.0, -1.0, 2.0, -4.0]])
    assert torch.equal(topk_relu(mixed, k=2), torch.tensor([[5.0, 0.0, 2.0, 0.0]]))


def test_normalise_is_invertible_by_the_statistics_it_returns() -> None:
    x = torch.randn(4, 8) * 3.0 + 2.0
    hat, mean, std = normalise(x)
    # De-normalisation divides by std alone while normalisation divides by
    # std + eps -- their asymmetry, reproduced, so the recovery is close and not
    # exact. The test pins the asymmetry rather than asserting a clean round trip.
    recovered = hat * std + mean
    assert torch.allclose(recovered, x, atol=1e-3)
    assert not torch.equal(recovered, x)


def test_the_dead_step_threshold_is_stated_in_sequences() -> None:
    """A step count that ignores the batch size means different things per run."""

    assert DEAD_STEPS_SEQUENCES == 10_000
    assert DEAD_STEPS_SEQUENCES // 16 == 625  # the value the four completed runs used
    assert DEAD_STEPS_SEQUENCES // 32 == 312  # and what doubling the batch must give


def test_the_spliced_replacement_reproduces_the_batched_model_exactly() -> None:
    """A CLT evaluated one block at a time must be the CLT that was trained.

    The faithfulness stage substitutes one MoE block at a time, in layer order,
    while the training objective decodes every layer at once. For a per-layer
    transcoder those are trivially the same computation; for a cross-layer one
    they are only the same if the accumulated source latents are complete at
    every target. Equality is asserted exactly rather than to a tolerance --
    both paths run the same matmuls in the same dtype, so anything but equality
    means they are different objects.
    """

    for cross_layer in (True, False):
        config = _tiny(cross_layer=cross_layer, num_layers=4)
        torch.manual_seed(0)
        model = Transcoder(config)
        generator = torch.Generator().manual_seed(1)
        activations = torch.randn(
            config.num_layers, 5, 3, config.d_model, generator=generator
        )

        batched = model(
            activations.reshape(config.num_layers, -1, config.d_model)
        ).reshape(activations.shape)
        replacement = TranscoderReplacement(model)
        spliced = torch.stack(
            [replacement(layer, activations[layer]) for layer in range(config.num_layers)]
        )
        assert torch.equal(batched, spliced), (
            f"{'CLT' if cross_layer else 'PLT'}: spliced and batched disagree by "
            f"{float((batched - spliced).abs().max()):.3e}"
        )


def test_a_cross_layer_target_refuses_to_decode_before_its_sources_have_fired() -> None:
    """Silently reconstructing from a subset would look like a better transcoder."""

    config = _tiny(cross_layer=True)
    replacement = TranscoderReplacement(Transcoder(config))
    with pytest.raises(KeyError, match="source layer 0"):
        replacement(2, torch.randn(4, config.d_model))


def test_beginning_a_new_forward_pass_clears_the_accumulated_latents() -> None:
    """Layer 0 firing IS the start of a pass; stale latents would mix two batches."""

    config = _tiny(cross_layer=True, num_layers=3)
    torch.manual_seed(0)
    replacement = TranscoderReplacement(Transcoder(config))
    generator = torch.Generator().manual_seed(2)
    first = torch.randn(config.num_layers, 6, config.d_model, generator=generator)
    second = torch.randn(config.num_layers, 6, config.d_model, generator=generator)

    run = lambda batch: torch.stack(  # noqa: E731
        [replacement(layer, batch[layer]) for layer in range(config.num_layers)]
    )
    run(first)
    repeated = run(second)
    replacement.reset()
    fresh = run(second)
    assert torch.equal(repeated, fresh), "the second pass depended on the first"


# ------------------------------------------------- matching two training runs


def _matched(**overrides) -> MatchedTraining:
    """The declaration of one arm of a matched pair, at the joint campaign's shape."""

    settings = {
        "target": "prollama:protein",
        "backbone_sha256": "a" * 64,
        "architecture": "PLT",
        "num_layers": 32,
        "d_model": 4096,
        "d_hidden": 16384,
        "k": 64,
        "training_token_budget": 68_000_000,
        "training_tokens": 68_000_412,
        "evaluation_sequences": 256,
    }
    settings.update(overrides)
    return MatchedTraining(**settings)


def test_the_two_modes_of_one_checkpoint_are_a_matched_pair() -> None:
    """The state the whole comparison rests on, and the one it must certify."""

    protein = _matched()
    text = _matched(target="prollama:text", training_tokens=68_003_919)
    record = compare_matched_training(protein, text)
    assert record["verdict"] == "MATCHED"
    assert record["disagreements"] == []
    assert record["distinct_targets"] is True
    # The digest is over the matched fields ONLY, so it must agree between two
    # conditions that differ in their target and in their realised token count --
    # a digest that moved with either would certify nothing.
    assert record["digests_agree"] is True
    assert protein.digest() == text.digest()
    assert record["training_tokens_realised"] == [68_000_412, 68_003_919]
    assert record["training_tokens_relative_difference"] < 1e-4


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_layers", 24),
        ("d_model", 2048),
        ("d_hidden", 49152),
        ("k", 32),
        ("training_token_budget", 34_000_000),
        ("evaluation_sequences", 128),
        ("backbone_sha256", "b" * 64),
        ("architecture", "CLT"),
    ],
)
def test_every_field_a_difference_could_be_attributed_to_is_detected(field, value) -> None:
    """Each of these would read as modality if it moved between the two modes.

    Parametrised over the declared list rather than spot-checked, because the
    failure this guards is a field being *added* to the pair's configuration and
    not to the set the comparison refuses on.
    """

    record = compare_matched_training(
        _matched(), _matched(target="prollama:text", **{field: value})
    )
    assert record["verdict"] == "MISMATCH"
    assert record["disagreements"] == [field]
    assert record["digests_agree"] is False
    assert record["fields"][field]["agree"] is False


def test_the_refused_set_is_the_declared_set_and_carries_the_backbone() -> None:
    assert set(_matched().matched()) == set(MATCHED_TRAINING_FIELDS)
    # 'prollama:protein' names a MODE; Llama-2-7b-hf, ProLLaMA_Stage_1 and
    # ProLLaMA all answer to it, so without the digest a pair drawn from two of
    # them would be a comparison across training stages wearing the label of a
    # comparison within one set of weights.
    assert "backbone_sha256" in MATCHED_TRAINING_FIELDS
    # The realised count is recorded and NOT refused on: the loop stops at the
    # first step to cross the budget, so it overshoots by up to one batch.
    assert "training_tokens" not in MATCHED_TRAINING_FIELDS
    assert "target" not in MATCHED_TRAINING_FIELDS


def test_an_undeclared_token_budget_is_its_own_verdict() -> None:
    """Equal steps are equal schedules over unequal data, and that is not a match.

    A text record and a protein record carry different numbers of scored
    positions, so two runs of the same step count see different amounts of data.
    Reporting that as MATCHED would certify the one thing the joint comparison
    most needs to be true and is least likely to be.
    """

    record = compare_matched_training(
        _matched(training_token_budget=None),
        _matched(target="prollama:text", training_token_budget=None),
    )
    assert record["verdict"] == "UNMATCHED_BUDGET"
    assert record["disagreements"] == []
    assert record["training_token_budget_declared"] is False
    # One-sided too: a declared budget and an undeclared one are not a pair.
    half = compare_matched_training(
        _matched(), _matched(target="prollama:text", training_token_budget=None)
    )
    assert half["verdict"] == "MISMATCH"


def test_one_condition_compared_with_itself_is_reported_as_such() -> None:
    record = compare_matched_training(_matched(), _matched())
    assert record["verdict"] == "MATCHED"
    assert record["distinct_targets"] is False


def test_the_declaration_round_trips_through_its_own_record() -> None:
    original = _matched()
    restored = matched_training(original.record())
    assert restored == original
    assert restored.digest() == original.digest()
    assert original.record()["digest"] == original.digest()


def test_a_record_missing_a_matched_field_is_refused_rather_than_defaulted(tmp_path) -> None:
    incomplete = _matched().record()
    del incomplete["d_hidden"]
    with pytest.raises(KeyError, match="d_hidden"):
        matched_training(incomplete)

    artefact = tmp_path / "run.json"
    artefact.write_text(json.dumps({"schema_version": "x"}), encoding="utf-8")
    with pytest.raises(KeyError, match=MATCHED_TRAINING_KEY):
        matched_training_from_artefact(artefact)

    artefact.write_text(
        json.dumps({MATCHED_TRAINING_KEY: _matched().record()}), encoding="utf-8"
    )
    assert matched_training_from_artefact(artefact) == _matched()


def test_a_checkpoint_round_trips_into_the_class_that_wrote_it() -> None:
    """The trainer's own state dict must load back, for both arms.

    Whether it loads into the *released* transcoder's reader is a different
    question with a different answer -- the shapes differ by design -- and the
    stage's own claim about that is what the faithfulness bridge has to satisfy.
    """

    for cross_layer in (True, False):
        config = _tiny(cross_layer=cross_layer)
        torch.manual_seed(1)
        model = Transcoder(config)
        state = model.state_dict()
        restored = Transcoder(config)
        restored.load_state_dict(state, strict=True)
        inputs, _ = _batch(config, scale=1.0)
        assert torch.allclose(model(inputs), restored(inputs))
