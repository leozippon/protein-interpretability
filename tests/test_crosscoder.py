"""What the Crosscoder must get right before any diff rests on it.

A Crosscoder's readout is an unsupervised claim -- *this latent belongs to that
model* -- and nothing inside a real run can falsify it. Every way it can be wrong
produces a finite, plausible-looking histogram rather than an error. So the tests
below are not a happy path; each one is a condition that must always hold, stated
against a construction whose answer is known in advance.

**Injected features must come back, in the right category.** Paired activations
carrying a declared number of shared, base-only and adapted-only features are
recovered and categorised, or the readout is measuring the dictionary rather than
the data.

**The shuffled-pairing null must destroy what it should destroy, and only that.**
Mispairing removes token-level correspondence, so a shared feature must survive as
a *recovered* latent and fail to be *categorised* as shared. Those two moving
independently is the test; a single accuracy number would not separate them.

**A rank-deficient cloud must not produce confident specificity.** This is the
operating regime rather than a corner case: the measured effective dimension at
this programme's dictionary site is 2,588-3,670 against ``d_model`` 4,096, and it
collapses much further at the ends of the stack. With every injected feature
shared, the count of exclusive latents the readout reports is its false-positive
rate.

**A per-layer quantity must never reach an artefact as a mean.** This exact
pipeline lost a pre-registered criterion to a per-layer dead mask collapsed to a
cross-layer scalar before anything downstream saw it (EXP-R2-203), so the guard
against it is tested against the shape of that failure and not only against a
correct payload.

**A site's fitted dictionary must not depend on which other sites were in the
run.** That property is what makes it sound to train a narrow site set and report
there, rather than training the whole stack; it is checked by fitting one site
both ways.

**An inadmissible layer must carry a refusal and not a number.**
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.transfer.crosscoder import (
    CATEGORIES,
    PAIRINGS,
    Crosscoder,
    CrosscoderConfig,
    SyntheticGroundTruth,
    assert_admissible_subset,
    assert_per_layer_fields,
    assert_required_per_site_fields,
    categorise,
    clip_per_site_grad_norm_,
    crosscoder_certificate,
    decoder_cosine,
    decoder_norms,
    live_mask,
    pair_backbone_digest,
    recovery_report,
    relative_decoder_norm,
    specificity_readout,
    train_crosscoders,
)
from src.transfer.transcoders import MatchedTraining


def _load_stage(filename: str):
    """Import a stage whose module name starts with a digit, as the stages do."""

    path = Path(__file__).resolve().parents[1] / "scripts" / "transfer" / filename
    spec = importlib.util.spec_from_file_location(f"_crosscoder_stage_{filename[:2]}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Small enough to run in seconds on CPU and large enough that a 2x over-complete
# TopK dictionary can actually resolve the injected features.
SMALL = {"d_model": 32, "n_sites": 1, "active_per_token": 3}
TOKENS = 192
STEPS = 1200


def _fit(
    truth: SyntheticGroundTruth,
    *,
    d_hidden: int = 48,
    seed: int = 5,
    steps: int = STEPS,
):
    configs = [
        CrosscoderConfig(
            sites=(0,),
            d_model=truth.d_model,
            d_hidden=d_hidden,
            k=truth.active_per_token,
            auxk=8,
            dead_steps=100,
            decoder_norm_penalty=3e-3,
            pairing=pairing,
        )
        for pairing in PAIRINGS
    ]
    models, _, extra = train_crosscoders(
        configs,
        truth.batches(tokens_per_batch=TOKENS, n_batches=steps + 32, seed=seed + 100),
        steps=steps,
        learning_rate=3e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        seed=seed,
        warm_up_batches=4,
        held_out=truth.batches(tokens_per_batch=TOKENS, n_batches=6, seed=seed + 200),
        eval_every=steps,
    )
    out = {}
    rng = np.random.default_rng(seed + 300)
    probe = truth.draw(TOKENS * 4, rng=rng)
    for index, model in enumerate(models):
        evaluation = extra["held_out"][index]
        live = live_mask(evaluation.pop("_counts"), minimum=1)
        out[model.config.pairing] = {
            "model": model,
            "live": live,
            "nmse": evaluation["nmse_per_site"][0],
            "readout": specificity_readout(
                model,
                live=live,
                admissible=(0,),
                exclusive_cut=0.95,
                shared_halfwidth=0.10,
            )["site_per_site"][0],
            "recovery": recovery_report(
                truth,
                model,
                site=0,
                base=probe[0],
                adapted=probe[1],
                coefficients=probe[2],
                live=live,
                exclusive_cut=0.95,
                shared_halfwidth=0.10,
                correlation_floor=0.5,
            )["per_category"],
        }
    return out


# ------------------------------------------------ recovery on known ground truth


@pytest.fixture(scope="module")
def mixed_full_rank():
    """12 shared, 6 base-only and 6 adapted-only features, at full rank."""

    truth = SyntheticGroundTruth(
        n_shared=12, n_base_specific=6, n_adapted_specific=6, seed=11, **SMALL
    )
    return truth, _fit(truth)


def test_injected_features_are_recovered_and_correctly_categorised(mixed_full_rank):
    _, fits = mixed_full_rank
    recovery = fits["true"]["recovery"]
    for category in ("shared", "base_specific", "adapted_specific"):
        entry = recovery[category]
        assert entry["recovered"] == entry["injected"], (
            f"{category}: recovered {entry['recovered']} of {entry['injected']} "
            "injected features"
        )
        assert entry["categorised"] == entry["injected"], (
            f"{category}: {entry['categorised']} of {entry['injected']} recovered "
            "features landed in the right category"
        )


def test_recovered_category_counts_match_the_injection(mixed_full_rank):
    """The live dictionary is the injected feature set and nothing else."""

    _, fits = mixed_full_rank
    counts = fits["true"]["readout"]["counts"]
    assert counts["shared"] == 12
    assert counts["base_specific"] == 6
    assert counts["adapted_specific"] == 6
    assert counts["intermediate"] == 0


def test_reconstruction_is_near_exact_when_the_dictionary_can_span_the_data(
    mixed_full_rank,
):
    _, fits = mixed_full_rank
    assert fits["true"]["nmse"] < 0.05


# ------------------------------------------------------ the shuffled-pairing null


def test_shuffled_pairing_destroys_the_shared_category_but_not_recovery(
    mixed_full_rank,
):
    """The null's signature, stated as the two quantities moving in opposite ways.

    A shared feature is present in the base checkpoint's activations whether or
    not the pairing is true, so it stays *recoverable*. What mispairing removes is
    the position it occupied in the other checkpoint, so it can no longer be
    *categorised* as shared. Both halves are asserted: a null that also destroyed
    recovery would be destroying the data rather than the correspondence.
    """

    _, fits = mixed_full_rank
    true_shared = fits["true"]["recovery"]["shared"]
    null_shared = fits["shuffled"]["recovery"]["shared"]

    assert null_shared["recovered"] >= 0.75 * null_shared["injected"], (
        "the null destroyed recovery itself, not only the pairing"
    )
    assert null_shared["categorised"] <= 0.25 * true_shared["categorised"], (
        f"the null still categorised {null_shared['categorised']} shared features "
        f"against the measurement's {true_shared['categorised']}"
    )

    true_fraction = fits["true"]["readout"]["fractions"]["shared"]
    null_fraction = fits["shuffled"]["readout"]["fractions"]["shared"]
    assert true_fraction > null_fraction + 0.20, (
        f"shared fraction {true_fraction:.3f} under true pairing against "
        f"{null_fraction:.3f} under the null: the gap the readout rests on is absent"
    )


def test_shuffled_pairing_cannot_reconstruct(mixed_full_rank):
    """Mispaired activations are not jointly explicable, and the NMSE says so."""

    _, fits = mixed_full_rank
    assert fits["shuffled"]["nmse"] > 10 * fits["true"]["nmse"]


# ------------------------------------------------------------ rank deficiency


def _all_shared(rank: int | None):
    """24 shared features, no model-specific ones, at a declared effective rank.

    ``shared_rotation`` is non-zero, so the two roles carry genuinely different
    directions for every feature while sharing its firing pattern. That is the
    case the 2024 note distinguishes from a model-specific feature and the one a
    naive readout would confuse with it, and it is what makes this a negative
    control rather than a self-pair.
    """

    return SyntheticGroundTruth(
        n_shared=24, n_base_specific=0, n_adapted_specific=0,
        seed=19, rank=rank, shared_rotation=0.5, **SMALL,
    )


@pytest.fixture(scope="module")
def rank_sweep():
    """The same construction at full rank and at one eighth of it."""

    return {rank: _fit(_all_shared(rank), d_hidden=64, steps=900)
            for rank in (None, 4)}


@pytest.mark.parametrize("rank", [None, 4])
def test_no_spurious_exclusive_latents_are_reported_at_any_effective_rank(
    rank_sweep, rank
):
    """The false-positive test, and it is the property the readout stands on.

    Zero model-specific latents are injected, so every exclusive latent the
    readout reports is spurious by construction. Measured over a rank sweep this
    count is **zero everywhere** -- at full rank, at 20/32 and 8/32, and at 4/32,
    which is past the point where the instrument can identify the features at
    all. Rank deficiency does not manufacture specificity; what it costs is
    measured by the test below.
    """

    counts = rank_sweep[rank]["true"]["readout"]["counts"]
    spurious = counts["base_specific"] + counts["adapted_specific"]
    assert spurious == 0, (
        f"{spurious} exclusive latents were reported at rank {rank} on data "
        "carrying none; the readout invents model-specific structure here"
    )


def test_what_rank_deficiency_costs_is_the_null_gap_and_it_must_be_visible(
    rank_sweep,
):
    """Where the instrument actually fails, so a campaign can refuse to read it.

    The quantity that separates a measurement from its null is the shared
    fraction: mispairing destroys token-level correspondence, so a genuinely
    shared latent cannot exist under it. That gap is what collapses when the
    activation cloud runs out of directions -- measured 1.000 against 0.088 at
    full rank and 0.732 against 0.542 at one eighth of it, on the same
    construction and the same seed.

    This is not a defect to be fixed; it is the reason R2.4's admission rule was
    amended to require a **non-degenerate** ``r99`` and not merely dictionaries
    that clear it. A layer whose effective dimension has collapsed is a layer
    where the null stops discriminating, and the artefact carries the gap so a
    reader can see it rather than inferring it from the rank.
    """

    def gap(fits) -> float:
        return (
            fits["true"]["readout"]["fractions"]["shared"]
            - fits["shuffled"]["readout"]["fractions"]["shared"]
        )

    full, severe = gap(rank_sweep[None]), gap(rank_sweep[4])
    assert full > 0.6, f"the null gap is only {full:.3f} on full-rank data"
    assert severe < 0.5 * full, (
        f"the null gap is {severe:.3f} at one eighth of full rank against "
        f"{full:.3f} at full rank: the collapse this regime must produce is "
        "absent, so nothing in the artefact would warn a reader off a layer whose "
        "effective dimension has gone"
    )


# --------------------------------------------------------------- determinism


def test_two_runs_at_one_seed_are_identical_and_a_different_seed_is_not():
    truth = SyntheticGroundTruth(
        n_shared=8, n_base_specific=4, n_adapted_specific=4, seed=23, **SMALL
    )
    config = CrosscoderConfig(
        sites=(0,), d_model=truth.d_model, d_hidden=32, k=3, auxk=4,
        dead_steps=50, decoder_norm_penalty=2e-3, pairing="true",
    )
    batches = truth.batches(tokens_per_batch=96, n_batches=200, seed=7)

    def fit(seed: int) -> torch.Tensor:
        models, _, _ = train_crosscoders(
            [config], batches, steps=60, learning_rate=3e-3, weight_decay=0.0,
            grad_clip=1.0, seed=seed, warm_up_batches=2,
        )
        return decoder_norms(models[0])

    first, again, other = fit(3), fit(3), fit(4)
    assert torch.equal(first, again)
    assert not torch.equal(first, other)


def test_a_site_is_fitted_identically_alone_and_inside_a_wider_run():
    """The property that makes a narrow site set sound rather than merely cheap.

    R2.4's admissible band is one or two layers. If a site's fit depended on which
    other sites shared the run, training the whole stack and reporting at two
    layers would be a different measurement from training those two -- and the
    campaign would have to pay for the stack. It does not: the sites are
    parameter-disjoint, the initialisation is drawn per site from a generator keyed
    to the backbone layer index, and the gradient clip is per site.
    """

    truth = SyntheticGroundTruth(
        d_model=32, n_sites=2, n_shared=8, n_base_specific=4,
        n_adapted_specific=4, active_per_token=3, seed=29,
    )
    both = truth.batches(tokens_per_batch=96, n_batches=200, seed=13)

    def only_second():
        for base, adapted in both():
            yield base[1:2], adapted[1:2]

    shared = {"d_model": 32, "d_hidden": 32, "k": 3, "auxk": 4, "dead_steps": 50,
              "decoder_norm_penalty": 2e-3, "pairing": "true"}
    options = {"steps": 80, "learning_rate": 3e-3, "weight_decay": 0.0,
               "grad_clip": 1.0, "seed": 31, "warm_up_batches": 2}
    wide, _, _ = train_crosscoders(
        [CrosscoderConfig(sites=(0, 1), **shared)], both, **options
    )
    narrow, _, _ = train_crosscoders(
        [CrosscoderConfig(sites=(1,), **shared)], only_second, **options
    )
    assert torch.equal(decoder_norms(wide[0])[:, 1], decoder_norms(narrow[0])[:, 0])


# ------------------------------------------------------- per-layer preservation


def test_a_per_site_field_reduced_to_a_scalar_is_refused():
    """The regression guard for the defect that voided a pre-registered criterion."""

    good = {"live_latents_per_site": [7, 9], "nested": {"nmse_per_site": [0.1, 0.2]}}
    assert_per_layer_fields(good, n_sites=2)

    with pytest.raises(ValueError, match="reduced to a scalar|per-site field"):
        assert_per_layer_fields({"nmse_per_site": 0.15}, n_sites=2)
    with pytest.raises(ValueError, match="per-site field"):
        assert_per_layer_fields({"nmse_per_site": 3}, n_sites=2)
    with pytest.raises(ValueError, match="values for"):
        assert_per_layer_fields({"nmse_per_site": [0.1]}, n_sites=2)
    with pytest.raises(ValueError, match="per-site field"):
        assert_per_layer_fields({"n_dead_per_site": None}, n_sites=2)


def test_a_per_site_field_buried_in_a_list_is_still_checked():
    """The mean this guard exists to catch was several levels down in its payload."""

    payload = {"fitted": {"true": [{"deep": {"nmse_per_site": 0.5}}]}}
    with pytest.raises(ValueError, match="fitted.true\\[0\\].deep.nmse_per_site"):
        assert_per_layer_fields(payload, n_sites=3)


def test_a_missing_per_site_field_is_refused_like_a_collapsed_one():
    with pytest.raises(ValueError, match="carries no"):
        assert_required_per_site_fields({"live_latents_per_site": [1]})


# ----------------------------------------------------------- admissibility


def test_admissible_layers_must_be_a_declared_subset_of_the_fitted_ones():
    assert assert_admissible_subset([28, 27], (0, 27, 28)) == (27, 28)
    with pytest.raises(ValueError, match="empty"):
        assert_admissible_subset([], (0, 1))
    with pytest.raises(ValueError, match="repeats"):
        assert_admissible_subset([1, 1], (0, 1))
    with pytest.raises(ValueError, match="not fitted"):
        assert_admissible_subset([2], (0, 1))


def test_no_diff_is_reported_at_an_inadmissible_layer():
    config = CrosscoderConfig(
        sites=(0, 27, 28), d_model=8, d_hidden=16, k=2, auxk=2, dead_steps=10
    )
    model = Crosscoder(config, init_seed=1)
    model.set_scales(torch.ones(2, 3))
    live = torch.ones(3, 16, dtype=torch.bool)
    readout = specificity_readout(
        model, live=live, admissible=(27, 28), exclusive_cut=0.95, shared_halfwidth=0.10
    )
    refused = readout["site_per_site"][0]
    assert refused["layer"] == 0
    assert refused["admissible"] is False
    assert refused["verdict"] == "ADMISSIBILITY_REFUSED"
    assert "counts" not in refused
    for entry in readout["site_per_site"][1:]:
        assert entry["admissible"] is True
        assert set(entry["counts"]) == set(CATEGORIES)
    assert readout["inadmissible_layers"] == [0]
    assert readout["category_counts_per_site"][0] is None


# ----------------------------------------------- the readout's own properties


def test_the_relative_decoder_norm_is_invariant_to_the_gauge_topk_leaves_free():
    """Scaling a latent's two decoders together must not move its category.

    TopK fixes the selection but not the magnitude, so ``W_dec -> c W_dec`` with
    ``W_enc -> W_enc / c`` leaves the reconstruction, the selection and the
    published sparsity term unchanged. A readout that moved under it would be
    reporting an arbitrary parametrisation.
    """

    config = CrosscoderConfig(sites=(0,), d_model=8, d_hidden=6, k=2, auxk=2, dead_steps=10)
    model = Crosscoder(config, init_seed=2)
    before, _ = relative_decoder_norm(decoder_norms(model))
    with torch.no_grad():
        for latent, factor in enumerate([0.1, 10.0, 1.0, 3.0, 0.5, 7.0]):
            model.W_dec[:, 0, latent] *= factor
    after, _ = relative_decoder_norm(decoder_norms(model))
    assert torch.allclose(before, after, atol=1e-6)


def test_a_latent_with_no_decoder_in_either_role_has_no_ratio():
    """It must not fall into the shared peak, which is the one answer it cannot give."""

    config = CrosscoderConfig(sites=(0,), d_model=8, d_hidden=4, k=2, auxk=2, dead_steps=10)
    model = Crosscoder(config, init_seed=3)
    with torch.no_grad():
        model.W_dec[:, 0, 0] = 0.0
    ratio, defined = relative_decoder_norm(decoder_norms(model))
    assert not bool(defined[0, 0])
    assert bool(defined[0, 1:].all())


def test_a_self_pair_reads_as_entirely_shared():
    """Two identical roles must give ratio 0.5 and cosine 1 at every latent."""

    config = CrosscoderConfig(sites=(0,), d_model=8, d_hidden=5, k=2, auxk=2, dead_steps=10)
    model = Crosscoder(config, init_seed=4)
    with torch.no_grad():
        model.W_dec[1] = model.W_dec[0]
    ratio, defined = relative_decoder_norm(decoder_norms(model))
    assert bool(defined.all())
    assert torch.allclose(ratio, torch.full_like(ratio, 0.5), atol=1e-6)
    assert torch.allclose(decoder_cosine(model), torch.ones_like(ratio), atol=1e-5)


def test_the_category_cuts_refuse_an_incoherent_pair():
    ratio = torch.tensor([0.0, 0.5, 1.0])
    codes = categorise(ratio, exclusive_cut=0.95, shared_halfwidth=0.1)
    assert [CATEGORIES[int(value)] for value in codes] == [
        "base_specific", "shared", "adapted_specific"
    ]
    with pytest.raises(ValueError, match="exclusive_cut"):
        categorise(ratio, exclusive_cut=0.4, shared_halfwidth=0.1)
    with pytest.raises(ValueError, match="shared_halfwidth"):
        categorise(ratio, exclusive_cut=0.6, shared_halfwidth=0.3)


# ------------------------------------------------------------------ refusals


def test_activations_of_different_shapes_are_refused():
    """The tensor-level half of the identical-input guarantee."""

    config = CrosscoderConfig(sites=(0,), d_model=4, d_hidden=8, k=2, auxk=2, dead_steps=10)
    model = Crosscoder(config, init_seed=5)
    model.set_scales(torch.ones(2, 1))
    base = torch.randn(1, 6, 4)
    with pytest.raises(ValueError, match="identical shapes"):
        model.objective(base, torch.randn(1, 5, 4), training=False)
    with pytest.raises(ValueError, match="built for"):
        model.objective(base, torch.randn(2, 6, 4), training=False)


def test_training_without_estimated_scales_is_refused():
    config = CrosscoderConfig(sites=(0,), d_model=4, d_hidden=8, k=2, auxk=2, dead_steps=10)
    model = Crosscoder(config, init_seed=6)
    with pytest.raises(RuntimeError, match="have not been estimated"):
        model.objective(torch.randn(1, 4, 4), torch.randn(1, 4, 4), training=False)
    model.set_scales(torch.ones(2, 1))
    with pytest.raises(RuntimeError, match="already frozen"):
        model.set_scales(torch.ones(2, 1))


def test_a_non_positive_scale_is_refused():
    config = CrosscoderConfig(sites=(0,), d_model=4, d_hidden=8, k=2, auxk=2, dead_steps=10)
    model = Crosscoder(config, init_seed=7)
    with pytest.raises(ValueError, match="finite and positive"):
        model.set_scales(torch.tensor([[1.0], [0.0]]))


def test_the_gradient_clip_is_applied_per_site():
    """One global clip would couple the sites and break the property above."""

    config = CrosscoderConfig(sites=(0, 1), d_model=4, d_hidden=8, k=2, auxk=2, dead_steps=10)
    model = Crosscoder(config, init_seed=8)
    model.set_scales(torch.ones(2, 2))
    model.objective(torch.randn(2, 16, 4), torch.randn(2, 16, 4), training=True)[
        "loss"
    ].backward()
    with torch.no_grad():
        model.W_dec.grad[:, 0] *= 1e4
    norms = clip_per_site_grad_norm_(model, 1.0)
    assert norms[0] > norms[1]
    for index in range(2):
        clipped = torch.sqrt(
            (model.W_enc.grad[:, index].double() ** 2).sum()
            + (model.b_enc.grad[index].double() ** 2).sum()
            + (model.W_dec.grad[:, index].double() ** 2).sum()
            + (model.b_dec.grad[:, index].double() ** 2).sum()
        )
        assert float(clipped) <= 1.0 + 1e-6


def test_an_identical_configuration_twice_in_one_run_is_refused():
    truth = SyntheticGroundTruth(
        n_shared=4, n_base_specific=2, n_adapted_specific=2, seed=37, **SMALL
    )
    config = CrosscoderConfig(sites=(0,), d_model=32, d_hidden=16, k=3, auxk=2, dead_steps=10)
    with pytest.raises(ValueError, match="same object twice"):
        train_crosscoders(
            [config, config],
            truth.batches(tokens_per_batch=32, n_batches=8, seed=1),
            steps=2, learning_rate=1e-3, weight_decay=0.0, grad_clip=1.0,
            seed=1, warm_up_batches=1,
        )


def test_configs_disagreeing_on_shape_cannot_share_one_stream():
    left = CrosscoderConfig(sites=(0,), d_model=32, d_hidden=16, k=3, auxk=2, dead_steps=10)
    right = CrosscoderConfig(sites=(0,), d_model=32, d_hidden=32, k=3, auxk=2, dead_steps=10)
    truth = SyntheticGroundTruth(
        n_shared=4, n_base_specific=2, n_adapted_specific=2, seed=41, **SMALL
    )
    with pytest.raises(ValueError, match="must share sites"):
        train_crosscoders(
            [left, right],
            truth.batches(tokens_per_batch=32, n_batches=8, seed=1),
            steps=2, learning_rate=1e-3, weight_decay=0.0, grad_clip=1.0,
            seed=1, warm_up_batches=1,
        )


def test_an_incoherent_configuration_is_refused_at_construction():
    with pytest.raises(ValueError, match="ascending and unique"):
        CrosscoderConfig(sites=(1, 0), d_model=4, d_hidden=8, k=2, auxk=2, dead_steps=1)
    with pytest.raises(ValueError, match="exceeds d_hidden"):
        CrosscoderConfig(sites=(0,), d_model=4, d_hidden=2, k=8, auxk=2, dead_steps=1)
    with pytest.raises(ValueError, match="pairing"):
        CrosscoderConfig(
            sites=(0,), d_model=4, d_hidden=8, k=2, auxk=2, dead_steps=1, pairing="both"
        )


# ----------------------------------------------------------- the certificate


def _declaration(**overrides):
    fields = {
        "target": "prollama:protein",
        "backbone_sha256": "a" * 64,
        "architecture": "CROSSCODER",
        "num_layers": 2,
        "d_model": 4096,
        "d_hidden": 8192,
        "k": 32,
        "auxk": 192,
        "training_token_budget": 1000,
        "training_tokens": 1000,
        "evaluation_sequences": 256,
        "learning_rate": 2e-4,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "batch_size": 4,
        "seed": 1,
        "corpus_seed": 2,
        "max_tokens": 1024,
    }
    fields.update(overrides)
    return MatchedTraining(**fields)


def _extra(**overrides):
    fields = {
        "sites": [27, 28],
        "decoder_norm_penalty": 3e-3,
        "pairing": ["true", "shuffled"],
        "backbone_pair_sha256": "b" * 64,
    }
    fields.update(overrides)
    return fields


def test_two_modes_of_one_pair_certify_matched():
    verdict = crosscoder_certificate(
        _declaration(target="prollama:text"),
        _declaration(target="prollama:protein"),
        left_extra=_extra(),
        right_extra=_extra(),
    )
    assert verdict["verdict"] == "MATCHED"
    assert verdict["distinct_targets"] is True
    assert verdict["crosscoder_disagreements"] == []


@pytest.mark.parametrize(
    "override", [{"sites": [27]}, {"decoder_norm_penalty": 0.0},
                 {"pairing": ["true"]}, {"backbone_pair_sha256": "c" * 64}]
)
def test_a_disagreement_on_a_crosscoder_specific_field_is_a_mismatch(override):
    verdict = crosscoder_certificate(
        _declaration(target="prollama:text"),
        _declaration(target="prollama:protein"),
        left_extra=_extra(),
        right_extra=_extra(**override),
    )
    assert verdict["verdict"] == "MISMATCH"
    assert verdict["crosscoder_disagreements"] == sorted(override)


def test_a_certificate_missing_a_crosscoder_field_is_refused():
    with pytest.raises(KeyError, match="sites"):
        crosscoder_certificate(
            _declaration(), _declaration(),
            left_extra={"decoder_norm_penalty": 0.0, "pairing": ["true"],
                        "backbone_pair_sha256": "b" * 64},
            right_extra=_extra(),
        )


def test_the_pair_digest_names_the_pair_and_refuses_a_self_pair():
    first = pair_backbone_digest("a" * 64, "b" * 64)
    assert first == pair_backbone_digest("a" * 64, "b" * 64)
    assert first != pair_backbone_digest("b" * 64, "a" * 64)
    with pytest.raises(ValueError, match="same weight digest"):
        pair_backbone_digest("a" * 64, "a" * 64)


# -------------------------------------------------------------- the stage CLI


STAGE = _load_stage("32_crosscoder.py")


def test_a_layer_set_is_parsed_with_inclusive_ranges():
    assert STAGE.parse_layers("0,27-29,31") == (0, 27, 28, 29, 31)
    assert STAGE.parse_layers("27") == (27,)
    with pytest.raises(Exception):
        STAGE.parse_layers("29-27")
    with pytest.raises(Exception):
        STAGE.parse_layers("1,1")


def test_the_admissible_layer_set_is_never_defaulted():
    """It is the pre-registered statement of where a diff may be reported."""

    args = STAGE.build_parser().parse_args(
        ["--base", "x", "--adapted", "y", "--rendering", "prollama",
         "--mode", "protein", "--layers", "27,28"]
    )
    assert args.admissible_layers is None
    with pytest.raises(ValueError, match="admissible-layers"):
        STAGE.resolve(args)


def test_a_synthetic_check_refuses_a_real_campaigns_arguments():
    args = STAGE.build_parser().parse_args(
        ["--synthetic-check", "--base", "x"]
    )
    with pytest.raises(ValueError, match="meaningless beside --synthetic-check"):
        STAGE.resolve(args)


def test_the_memory_arithmetic_scales_with_the_site_set():
    """The finding that decides a campaign's site set, held as arithmetic."""

    wide = CrosscoderConfig(
        sites=tuple(range(32)), d_model=4096, d_hidden=8192, k=32, auxk=192, dead_steps=625
    )
    narrow = CrosscoderConfig(
        sites=(27, 28), d_model=4096, d_hidden=8192, k=32, auxk=192, dead_steps=625
    )
    # Two encoders and two decoders per site: twice a per-layer transcoder's
    # dictionary at the same width, which is the whole memory story.
    assert wide.n_parameters() == pytest.approx(4 * 4096 * 8192 * 32, rel=1e-3)
    assert wide.n_parameters() / narrow.n_parameters() == pytest.approx(16.0, rel=1e-3)

    options = {"n_dictionaries": 2, "tokens_per_batch": 2230, "n_backbones": 2,
               "backbone_parameters": 6.74e9}
    assert (
        STAGE.memory_arithmetic(narrow, **options)["live_total_mib"]
        < STAGE.memory_arithmetic(wide, **options)["live_total_mib"]
    )
    # Both backbones are resident in both, so the floor does not scale away.
    assert STAGE.memory_arithmetic(narrow, **options)["backbones_bf16_mib"] == (
        STAGE.memory_arithmetic(wide, **options)["backbones_bf16_mib"]
    )
