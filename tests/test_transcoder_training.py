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

import torch

from src.transfer.transcoders import (
    DEAD_STEPS_SEQUENCES,
    Transcoder,
    TranscoderConfig,
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
