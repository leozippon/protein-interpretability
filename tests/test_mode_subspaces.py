"""What the mode-subspace measurement must get right before any reading rests on it.

Every way this instrument can be wrong produces a finite, plausible number rather
than an error. An unbound hook reports zero damage; a hook that is bound but never
substitutes reports zero damage; an overlap statistic reported without its chance
level reads as evidence at a value two random subspaces reach by construction; and
a damage figure that is entirely a shift in the model's unigram output is a bias
term wearing the clothes of a computation. So each test below is a condition that
must always hold, stated against a construction whose answer exists in advance.

**A null ablation must be an identity, and a large random ablation must not be.**
Both halves are needed and neither alone is sufficient: a hook that failed to bind
passes the first and would pass a null-only suite forever, which is a failure
``src.transfer.path_patching`` has actually recorded.

**An overlap statistic must return its stated chance level on random subspaces.**
The closed form is ``max(r_a, r_b) / d`` and it is checked against a Monte-Carlo
draw at several shapes, because the whole interpretation of a measured overlap is
its distance from that number.

**A behavioural read on an unmeasurable mode must raise.** ``Llama-2-7b-hf``'s
protein mode reads +0.0843 nats/token of context information against a 0.30-nat
floor (EXP-R2-152, re-measured at EXP-R2-174), and the refusal has to be keyed to
that number rather than to the checkpoint's name -- so the test drives it with the
number and also checks that the same code admits the four ProLLaMA cells.

**The unigram decomposition must close.** ``total = unigram + residual`` is an
identity of the logarithm taken per position, so it must hold to float precision
and not approximately; the headline claim is licensed by the residual alone, and a
decomposition that did not close would make that licence meaningless.

**The synthetic path must recover a planted overlap, and the nulls must not
fire.** It is the only place any of this is falsifiable.

**The stage's pre-registered decisions must be refused when absent.** A flag with
a default is a decision taken by whoever last edited the default.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import mode_subspaces as ms  # noqa: E402
from src.transfer.budget import SCREENING_CONTEXT_INFORMATION_NATS  # noqa: E402
from src.transfer.statistics import MINIMUM_BOOTSTRAP_UNITS  # noqa: E402

RULE = ms.decision_rule("residual_licensed_v1")

#: The four ProLLaMA cells and the one refused cell, as EXP-R2-152 measured them
#: (Llama-2's protein figure re-measured at EXP-R2-174). Written once so a test
#: about the refusal and a test about the admission read the same evidence.
MEASURED_CONTEXT_INFORMATION: dict[tuple[str, str], float] = {
    ("ProLLaMA_Stage_1", "text"): 0.8336,
    ("ProLLaMA_Stage_1", "protein"): 0.5505,
    ("ProLLaMA", "text"): 0.7368,
    ("ProLLaMA", "protein"): 0.5215,
    ("Llama-2-7b-hf", "protein"): 0.0843,
}


def _stage():
    """The stage entry point, imported by path the way the worker's preflight does."""

    path = REPO_ROOT / "scripts" / "transfer" / "38_mode_subspaces.py"
    spec = importlib.util.spec_from_file_location("_stage_38_mode_subspaces", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------- the projection itself


def test_a_null_ablation_is_the_identity_bit_for_bit():
    # Not "close to": a rank-0 projection subtracts nothing, and an implementation
    # that routed it through a matrix multiply with an empty basis would return a
    # rounded copy. Every damage figure is a difference between two passes, so a
    # rounding at the identity is a floor under every effect the ladder can report.
    activations = torch.randn(64, 128)
    empty = torch.zeros((128, 0))
    assert torch.equal(ms.project_out(activations, empty), activations)


def test_a_full_rank_ablation_is_the_zero_write_the_handle_already_performs():
    # The ladder's top anchor. `JointReplaceable.ablated` zeroes the block's write;
    # projecting out the whole space must be the same object, or the ladder and its
    # own denominator are two different interventions.
    activations = torch.randn(16, 32)
    whole = torch.eye(32)
    assert torch.allclose(
        ms.project_out(activations, whole), torch.zeros_like(activations), atol=1e-6
    )


def test_a_projection_removes_exactly_the_ablated_component_and_nothing_else():
    generator = torch.Generator().manual_seed(11)
    basis = ms.random_orthonormal_basis(32, 5, generator=generator)
    activations = torch.randn(40, 32)
    removed = ms.project_out(activations, basis)
    # Nothing left along the basis, and the orthogonal complement untouched.
    assert torch.allclose(removed @ basis, torch.zeros(40, 5), atol=1e-5)
    assert torch.allclose(
        removed + (activations @ basis) @ basis.T, activations, atol=1e-5
    )


def test_the_hook_invariants_pass_on_a_bound_interceptor():
    # A stand-in for a block whose output is spliced: the "logits" are a linear
    # readout of the (possibly projected) activation, so a bound interceptor moves
    # them and an unbound one cannot. `None` is the unhooked path.
    torch.manual_seed(3)
    activation = torch.randn(8, 64)
    readout = torch.randn(64, 40)

    def logits_for_basis(basis: torch.Tensor | None) -> torch.Tensor:
        if basis is None:
            return activation @ readout
        return ms.project_out(activation, basis) @ readout

    record = ms.intervention_invariants(
        logits_for_basis,
        d_model=64,
        rank=32,
        layer=7,
        seed=5,
        tolerance=RULE.logit_tolerance,
    )
    assert record["null_projection_max_logit_gap"] == 0.0
    assert record["random_projection_max_logit_gap"] > RULE.logit_tolerance


def test_an_unbound_hook_is_caught_by_the_positive_control_and_not_by_the_null():
    # The failure the positive control exists for: an interceptor that silently
    # ignores the basis. The null test passes -- it always does -- so a suite
    # carrying only the null would certify an unpatched model as patched.
    torch.manual_seed(4)
    activation = torch.randn(8, 64)
    readout = torch.randn(64, 40)

    def unbound(_basis: torch.Tensor | None) -> torch.Tensor:
        return activation @ readout

    # The null half of the same check is satisfied by the broken hook, which is
    # exactly why it cannot be the whole check.
    assert float((unbound(None) - unbound(torch.zeros((64, 0)))).abs().max()) == 0.0
    with pytest.raises(RuntimeError, match="not bound"):
        ms.intervention_invariants(
            unbound,
            d_model=64,
            rank=32,
            layer=7,
            seed=5,
            tolerance=RULE.logit_tolerance,
        )


def test_an_interceptor_that_is_not_the_identity_at_rank_zero_is_refused():
    torch.manual_seed(5)
    readout = torch.randn(16, 8)

    def drifting(basis: torch.Tensor | None) -> torch.Tensor:
        # A hook that perturbs the tensor merely by being bound -- a dtype
        # round-trip, an in-place normalisation -- so the rank-0 interception is
        # not the identity even though it removes no direction.
        offset = 0.0 if basis is None else 1.0
        return (torch.ones(4, 16) + offset) @ readout

    with pytest.raises(RuntimeError, match="not the identity"):
        ms.intervention_invariants(
            drifting, d_model=16, rank=8, layer=0, seed=1, tolerance=1e-3
        )


# -------------------------------------------------------------- overlap


@pytest.mark.parametrize(
    "d_model,rank_left,rank_right", [(256, 8, 8), (256, 32, 32), (512, 16, 64)]
)
def test_the_overlap_statistic_returns_its_stated_chance_level_on_random_subspaces(
    d_model: int, rank_left: int, rank_right: int
):
    # The whole interpretation of a measured overlap is its distance from this
    # number, and the number is not small: two random 512-dimensional subspaces of
    # R^4096 already share a mean squared principal cosine of 0.125.
    generator = torch.Generator().manual_seed(20260819)
    left = ms.random_orthonormal_basis(d_model, rank_left, generator=generator)
    right = ms.random_orthonormal_basis(d_model, rank_right, generator=generator)
    report = ms.subspace_overlap(left, right, seed=7, chance_draws=128)
    closed_form = max(rank_left, rank_right) / d_model
    assert report["chance"][ms.MEAN_SQUARED_COSINE]["closed_form_mean"] == pytest.approx(
        closed_form
    )
    # The Monte-Carlo band must reproduce the closed form, and the two random
    # subspaces this test drew must land inside their own band.
    assert report["chance"][ms.MEAN_SQUARED_COSINE]["mean"] == pytest.approx(
        closed_form, rel=0.15
    )
    measured = report[ms.MEAN_SQUARED_COSINE]
    assert (
        report["chance"][ms.MEAN_SQUARED_COSINE]["p2.5"] * 0.6
        <= measured
        <= report["chance"][ms.MEAN_SQUARED_COSINE]["p97.5"] * 1.4
    )


def test_two_identical_subspaces_overlap_at_one_and_two_orthogonal_ones_at_zero():
    generator = torch.Generator().manual_seed(2)
    basis = ms.random_orthonormal_basis(64, 8, generator=generator)
    same = ms.subspace_overlap(basis, basis, seed=1, chance_draws=16)
    assert same[ms.MEAN_SQUARED_COSINE] == pytest.approx(1.0, abs=1e-6)
    assert same[ms.FIRST_PRINCIPAL_ANGLE_COSINE] == pytest.approx(1.0, abs=1e-6)
    whole = ms.random_orthonormal_basis(64, 16, generator=generator)
    orthogonal = ms.subspace_overlap(whole[:, :8], whole[:, 8:], seed=1, chance_draws=16)
    assert orthogonal[ms.MEAN_SQUARED_COSINE] == pytest.approx(0.0, abs=1e-6)


def test_a_planted_partial_overlap_is_recovered_by_the_statistic():
    generator = torch.Generator().manual_seed(9)
    frame = ms.random_orthonormal_basis(128, 12, generator=generator)
    left = frame[:, :8]
    right = torch.cat([frame[:, :4], frame[:, 8:]], dim=1)
    report = ms.subspace_overlap(left, right, seed=3, chance_draws=32)
    assert report[ms.MEAN_SQUARED_COSINE] == pytest.approx(4 / 8, abs=1e-6)


def test_an_overlap_report_always_carries_a_chance_level_for_both_statistics():
    # An overlap without its chance level is uninterpretable, so the chance level
    # cannot be optional in the return shape.
    generator = torch.Generator().manual_seed(1)
    left = ms.random_orthonormal_basis(64, 4, generator=generator)
    right = ms.random_orthonormal_basis(64, 4, generator=generator)
    report = ms.subspace_overlap(left, right, seed=1, chance_draws=8)
    for statistic in ms.OVERLAP_STATISTICS:
        assert statistic in report
        assert set(report["chance"][statistic]) >= {"mean", "p2.5", "p97.5"}


# ------------------------------------------------ the behavioural refusal


def test_a_behavioural_read_on_llama_2_protein_mode_raises():
    # Keyed to the measured number, not to the name. +0.0843 nats/token against a
    # 0.30-nat floor (EXP-R2-152, re-measured at EXP-R2-174).
    record = ms.mode_measurability(
        "protein", MEASURED_CONTEXT_INFORMATION[("Llama-2-7b-hf", "protein")]
    )
    assert record["measurability"] != "measurable"
    assert record["behavioural_read_admitted"] is False
    with pytest.raises(ValueError) as raised:
        ms.assert_behavioural_read(record)
    message = str(raised.value)
    assert "0.0843" in message
    assert "EXP-R2-152" in message
    # The catalogue ends at L32, so the message cites the evidence and says in as
    # many words that there is no limitation number for it. Inventing an L33 would
    # be worse than citing nothing.
    assert "there is no L33" in message
    assert "NOT a catalogued limitation" in message


@pytest.mark.parametrize(
    "checkpoint,mode",
    [
        ("ProLLaMA_Stage_1", "text"),
        ("ProLLaMA_Stage_1", "protein"),
        ("ProLLaMA", "text"),
        ("ProLLaMA", "protein"),
    ],
)
def test_every_qualified_prollama_cell_is_admitted(checkpoint: str, mode: str):
    # The other half of the refusal: a guard that refused everything would pass the
    # test above and measure nothing.
    record = ms.mode_measurability(
        mode, MEASURED_CONTEXT_INFORMATION[(checkpoint, mode)]
    )
    assert record["behavioural_read_admitted"] is True
    ms.assert_behavioural_read(record)


def test_the_refusal_boundary_is_this_stages_own_declared_floor():
    floor = ms.MODE_BEHAVIOURAL_READ_FLOOR_NATS
    just_below = ms.mode_measurability("protein", floor - 1e-6)
    just_above = ms.mode_measurability("protein", floor)
    assert just_below["behavioural_read_admitted"] is False
    assert just_above["behavioural_read_admitted"] is True


def test_the_mode_floor_is_not_the_panels_screening_floor_and_says_so():
    """EXP-R2-218 split the shared floor; this gate could not follow either half.

    The calibrated identification floor admits ``Llama-2-7b-hf``'s protein mode
    at +0.0843 nats/token, whose reversal cost is -0.0013 nats/residue, so
    adopting it here would turn a published refusal into an admission.  The
    precision-referenced criterion that would decide it properly cannot be
    evaluated, because the declared mode readings carry no standard error.  The
    incumbent magnitude therefore stays, and every record has to carry the fact
    that it is underived rather than let a reader assume otherwise.
    """

    llama = MEASURED_CONTEXT_INFORMATION[("Llama-2-7b-hf", "protein")]
    assert llama > SCREENING_CONTEXT_INFORMATION_NATS
    assert ms.MODE_BEHAVIOURAL_READ_FLOOR_NATS > SCREENING_CONTEXT_INFORMATION_NATS

    record = ms.mode_measurability("protein", llama)
    assert record["behavioural_read_admitted"] is False
    assert record["threshold_status"].startswith("UNDERIVED")
    assert "0.0843" in record["threshold_status"]
    # Nats alone is not a cross-arm threshold, and neither conversion is
    # available from a reading declared on the command line.
    assert record["threshold"]["nats_per_token"] == pytest.approx(
        ms.MODE_BEHAVIOURAL_READ_FLOOR_NATS
    )
    assert record["threshold"]["relative_to_baseline"] is None
    assert record["threshold"]["relative_to_baseline_undefined_because"]
    assert record["threshold"]["bits_per_symbol"] is None


# ---------------------------------------------- the unigram decomposition


def _scored_pass(label: str, seed: int, *, shift: float = 0.0, groups: int = 12):
    """A ScoredPass built from a small explicit predictive distribution."""

    rng = np.random.default_rng(seed)
    vocabulary = 24
    per_group = 8
    n = groups * per_group
    logits = rng.normal(size=(n, vocabulary))
    logits[:, 0] += shift
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    targets = np.asarray(
        [rng.choice(vocabulary, p=probabilities[row]) for row in range(n)], dtype=np.int64
    )
    group_ids = np.repeat(np.arange(groups), per_group)
    half_ids = (group_ids % 2).astype(np.int64)
    marginal = np.stack(
        [probabilities[half_ids == half].sum(axis=0) for half in (0, 1)]
    )
    counts = np.asarray([int((half_ids == half).sum()) for half in (0, 1)])
    return ms.ScoredPass(
        label=label,
        target_ids=targets,
        nll_nats=-np.log(probabilities[np.arange(n), targets]),
        group_ids=group_ids,
        half_ids=half_ids,
        marginal=marginal,
        marginal_counts=counts,
    )


def test_the_unigram_decomposition_sums_to_the_total_damage():
    clean = _scored_pass("clean", 1)
    ablated = dataclasses.replace(_scored_pass("clean", 1), label="ablated")
    # Same positions, different predictive distribution: rebuild the ablated pass
    # on the same targets so the two are paired, then perturb its distribution.
    rng = np.random.default_rng(5)
    perturbed = clean.marginal * rng.uniform(0.5, 1.5, size=clean.marginal.shape)
    ablated = ms.ScoredPass(
        label="ablated",
        target_ids=clean.target_ids,
        nll_nats=clean.nll_nats + rng.uniform(0.0, 0.4, size=clean.nll_nats.shape),
        group_ids=clean.group_ids,
        half_ids=clean.half_ids,
        marginal=perturbed,
        marginal_counts=clean.marginal_counts,
    )
    report = ms.unigram_decomposition(clean, ablated, seed=3, n_bootstrap=200)
    assert report["decomposition_closes_to_nats"] < 1e-12
    assert report["total_damage_nats"] == pytest.approx(
        report["unigram_damage_nats"] + report["residual_damage_nats"], abs=1e-12
    )


def test_the_decomposition_closes_at_every_position_and_not_only_on_average():
    clean = _scored_pass("clean", 7)
    rng = np.random.default_rng(11)
    ablated = ms.ScoredPass(
        label="ablated",
        target_ids=clean.target_ids,
        nll_nats=clean.nll_nats + rng.uniform(0.0, 1.0, size=clean.nll_nats.shape),
        group_ids=clean.group_ids,
        half_ids=clean.half_ids,
        marginal=clean.marginal * 2.0,
        marginal_counts=clean.marginal_counts,
    )
    total = ablated.nll_nats - clean.nll_nats
    unigram = ablated.unigram_nll_nats - clean.unigram_nll_nats
    residual = ablated.conditional_nll_nats - clean.conditional_nll_nats
    assert np.allclose(total, unigram + residual, atol=1e-12)


def test_the_unigram_estimator_is_held_out_across_groups():
    # A position in half A is scored against the marginal accumulated on half B.
    # The plug-in value is computed as well, and the two must differ -- if they did
    # not, the "held-out" estimator would be the plug-in under another name and
    # L12's bias would be inside the headline figure.
    scored = _scored_pass("clean", 13)
    assert not np.allclose(scored.unigram_nll_nats, scored.plug_in_unigram_nll_nats)


def test_a_pass_whose_half_carries_no_positions_is_refused():
    scored = _scored_pass("clean", 17)
    with pytest.raises(ValueError, match="no positions"):
        ms.ScoredPass(
            label="broken",
            target_ids=scored.target_ids,
            nll_nats=scored.nll_nats,
            group_ids=scored.group_ids,
            half_ids=np.zeros_like(scored.half_ids),
            marginal=scored.marginal,
            marginal_counts=np.asarray([int(scored.half_ids.size), 0]),
        )


def test_two_passes_over_different_positions_cannot_be_differenced():
    clean = _scored_pass("clean", 19)
    other = _scored_pass("other", 23)
    with pytest.raises(ValueError, match="different targets"):
        ms.unigram_decomposition(clean, other, seed=1, n_bootstrap=64)


def test_the_damage_interval_honours_the_shared_bootstrap_unit_floor():
    below = _scored_pass("clean", 29, groups=MINIMUM_BOOTSTRAP_UNITS - 1)
    with pytest.raises(ValueError, match="below the"):
        ms.damage_interval(
            below.nll_nats + 0.1,
            below.nll_nats,
            below.target_ids,
            below.group_ids,
            seed=1,
            n_bootstrap=64,
        )


def test_this_module_adds_no_resampler_of_its_own():
    # A repository invariant test enumerates every function in `src.transfer` whose
    # name mentions resampling and requires it to reach the shared unit floor. This
    # module must add none: the group bootstrap has one implementation and this
    # module calls it.
    import inspect

    offenders = [
        name
        for name, member in vars(ms).items()
        if inspect.isfunction(member)
        and member.__module__ == ms.__name__
        and ("bootstrap" in name.lower() or "resampl" in name.lower())
    ]
    assert offenders == []


# ------------------------------------------------------- necessity and verdict


def test_a_layer_whose_full_ablation_does_not_damage_has_no_necessary_subspace():
    # Appendix B rule 2 on the denominator rather than on a threshold: a fraction of
    # a number that is not attainable is not a criterion.
    report = ms.necessary_rank((1, 2, 4), (0.0, 0.0, 0.0), 0.0, RULE)
    assert report["necessary_rank"] is None
    assert report["attainable"] is False
    assert "no attainable denominator" in report["withheld_reason"]


def test_the_necessary_rank_is_the_smallest_rung_that_reaches_the_fraction():
    report = ms.necessary_rank((1, 2, 4, 8), (0.1, 0.3, 0.6, 0.9), 1.0, RULE)
    assert report["necessary_rank"] == 4
    assert report["ladder_exhausted"] is False


def test_a_ladder_that_never_reaches_the_fraction_says_so():
    report = ms.necessary_rank((1, 2), (0.1, 0.2), 1.0, RULE)
    assert report["necessary_rank"] is None
    assert report["attainable"] is True
    assert report["ladder_exhausted"] is True


def _damage(value: float, low: float, high: float) -> dict:
    return {"difference": value, "difference_ci95": [low, high]}


def _own(residual: float, share: float, total: float | None = None) -> dict:
    resolved = residual / share if total is None and share else (total or 1.0)
    return {
        "total": _damage(resolved, resolved - 0.01, resolved + 0.01),
        "residual": _damage(residual, residual - 0.01, residual + 0.01),
        "residual_share": share,
    }


def test_a_large_damage_with_a_small_residual_reads_as_unigram_only():
    # The clause the headline claim rests on. A "mode-specific" direction whose
    # ablation only moves the unigram output is a bias term.
    verdict = ms.layer_verdict(
        layer=11,
        modes=("text", "protein"),
        own={"text": _own(0.4, 0.1), "protein": _own(0.4, 0.1)},
        asymmetry={
            "text": _damage(0.3, 0.2, 0.4),
            "protein": _damage(0.3, 0.2, 0.4),
        },
        overlap=None,
        attainable={"text": True, "protein": True},
        invariants_held=True,
        rule=RULE,
        statistic=ms.MEAN_SQUARED_COSINE,
    )
    assert verdict["verdict"] == "UNIGRAM_ONLY"
    assert "unigram statistics" in verdict["reading"]


def test_a_layer_with_no_established_damage_is_not_a_subspace_claim():
    # There is nothing to decompose before there is damage, and "the ablation did
    # not measurably hurt" is a statement about the site and the cohort rather than
    # a claim that the subspace is unnecessary.
    verdict = ms.layer_verdict(
        layer=5,
        modes=("text", "protein"),
        own={
            "text": _own(0.4, 0.9, total=0.45),
            "protein": {
                "total": _damage(0.02, -0.05, 0.09),
                "residual": _damage(0.01, -0.03, 0.05),
                "residual_share": 0.5,
            },
        },
        asymmetry={
            "text": _damage(0.3, 0.2, 0.4),
            "protein": _damage(0.3, 0.2, 0.4),
        },
        overlap=None,
        attainable={"text": True, "protein": True},
        invariants_held=True,
        rule=RULE,
        statistic=ms.MEAN_SQUARED_COSINE,
    )
    assert verdict["verdict"] == "NO_MEASURED_DAMAGE"


def test_a_layer_whose_instrument_did_not_hold_is_void_and_not_negative():
    verdict = ms.layer_verdict(
        layer=3,
        modes=("text", "protein"),
        own={"text": _own(0.4, 0.9), "protein": _own(0.4, 0.9)},
        asymmetry={
            "text": _damage(0.3, 0.2, 0.4),
            "protein": _damage(0.3, 0.2, 0.4),
        },
        overlap=None,
        attainable={"text": True, "protein": True},
        invariants_held=False,
        rule=RULE,
        statistic=ms.MEAN_SQUARED_COSINE,
    )
    assert verdict["verdict"] == "VOID_INSTRUMENT"


def test_the_verdict_is_stated_per_layer_and_carries_its_own_rule():
    verdict = ms.layer_verdict(
        layer=27,
        modes=("text", "protein"),
        own={"text": _own(0.4, 0.9), "protein": _own(0.4, 0.9)},
        asymmetry={
            "text": _damage(0.3, 0.2, 0.4),
            "protein": _damage(0.3, 0.2, 0.4),
        },
        overlap=None,
        attainable={"text": True, "protein": True},
        invariants_held=True,
        rule=RULE,
        statistic=ms.MEAN_SQUARED_COSINE,
    )
    assert verdict["layer"] == 27
    assert verdict["decision_rule"]["name"] == "residual_licensed_v1"


# ------------------------------------------------------------- the fixture


#: Bootstrap draws the fixture's own intervals are taken over. 200 rather than the
#: 1,000 a campaign uses: every check below reads a point estimate or a sign, the
#: fixture is deterministic, and 1,000 draws over the whole certificate cost this
#: suite four minutes -- paid twice, because one orchestration test re-runs the
#: suite inside a project copy.
FIXTURE_BOOTSTRAP = 200


@pytest.fixture(scope="module")
def certificate():
    design = ms.SyntheticDesign()
    return ms.synthetic_certificate(
        design, ladder=(1, 2, 4, 8, 12), rule=RULE, n_bootstrap=FIXTURE_BOOTSTRAP
    )


def test_the_synthetic_path_recovers_the_planted_overlap(certificate):
    check = certificate["checks"]["overlap_recovers_the_planted_value"]
    assert check["passed"], check
    assert check["recovered"] == pytest.approx(check["planted"], abs=0.05)


def test_the_synthetic_nulls_do_not_fire(certificate):
    checks = certificate["checks"]
    # The overlap null: chance reproduces its closed form and the measured value
    # stands clear of it.
    assert checks["chance_level_reproduces_the_closed_form"]["passed"]
    assert checks["overlap_null_does_not_fire"]["passed"]
    # The intervention nulls: rank 0 is a no-op and a random subspace of the same
    # rank as the mode's own basis damages it far less.
    assert checks["null_ablation_is_a_no_op"]["passed"]
    assert checks["own_basis_beats_the_random_control"]["passed"]


def test_the_synthetic_separates_a_planted_bias_block_from_a_planted_context_block(
    certificate,
):
    checks = certificate["checks"]
    assert checks["bias_block_reads_as_unigram"]["passed"], checks[
        "bias_block_reads_as_unigram"
    ]
    assert checks["context_block_reads_as_residual"]["passed"], checks[
        "context_block_reads_as_residual"
    ]
    for mode in ("text", "protein"):
        assert (
            checks["bias_block_reads_as_unigram"][mode]["residual_share"]
            < checks["context_block_reads_as_residual"][mode]["residual_share"]
        )


def test_the_synthetic_decomposition_closes_everywhere(certificate):
    assert certificate["checks"]["decomposition_closes"]["passed"]


def test_the_synthetic_certificate_passes_as_a_whole(certificate):
    assert certificate["certificate"] == "PASSED"


def test_every_verdict_the_rule_can_return_is_reachable_on_planted_geometry():
    # Appendix B rule 2 applied to a verdict: a rule that cannot reach one of its
    # own outcomes on data built to produce it was decided before the measurement.
    report = ms.synthetic_verdict_attainability(
        ms.SyntheticDesign(), rule=RULE, n_bootstrap=FIXTURE_BOOTSTRAP
    )
    assert report["passed"], {
        name: (cell["expected_verdict"], cell["verdict"])
        for name, cell in report["cells"].items()
    }
    assert {cell["verdict"] for cell in report["cells"].values()} == {
        "DISTINCT_SUBSPACES",
        "MIXED",
        "SHARED_SUBSPACE",
    }


def test_a_ladder_too_coarse_to_leave_saturation_loses_the_shared_corner():
    # Recorded as a property rather than hidden: at a rung that removes everything
    # a mode computes, the paired own-minus-other contrast resolves the difference
    # between two ESTIMATES of one subspace. Two spans planted identical read a
    # mean squared cosine of 0.9987 and the contrast still excludes zero there.
    shared = dataclasses.replace(ms.SyntheticDesign(), shared=12)
    coarse = ms.synthetic_certificate(
        shared, ladder=(1, 12), rule=RULE, n_bootstrap=FIXTURE_BOOTSTRAP
    )
    fine = ms.synthetic_certificate(
        shared, ladder=(1, 2, 4, 8, 12), rule=RULE, n_bootstrap=FIXTURE_BOOTSTRAP
    )
    assert coarse["necessary_rank_used"]["text"] == shared.context_rank
    assert fine["necessary_rank_used"]["text"] < shared.context_rank
    assert coarse["verdict"]["verdict"] == "MIXED"
    assert fine["verdict"]["verdict"] == "SHARED_SUBSPACE"
    assert (
        coarse["overlap_of_necessary_subspaces"][ms.MEAN_SQUARED_COSINE] > 0.99
    ), "the two spans are planted identical, so a saturated rung is the only story"


def test_a_flat_spectrum_makes_a_top_rank_basis_unidentified():
    # The limitation the module records rather than hides: two clouds occupying the
    # SAME span recover different top-r subspaces where the spectrum is flat, and
    # the overlap statistic then reads the chance value for that shared span rather
    # than 1. This is why the eigenvalue gap travels with every basis.
    flat = dataclasses.replace(
        ms.SyntheticDesign(), spectrum_decay=1.0, shared=12, bias_rank=1
    )
    report = ms.synthetic_certificate(flat, ladder=(1, 8, 12), rule=RULE)
    overlap = report["overlap_of_necessary_subspaces"][ms.MEAN_SQUARED_COSINE]
    ranks = report["necessary_rank_used"]
    if ranks["text"] < flat.context_rank:
        # The spans are identical, so a fully identified basis would read 1.0.
        assert overlap < 0.95
    gap = ms.eigen_gap(
        np.asarray([1.0] * 12 + [0.001] * 4), ranks["text"]
    )
    assert "relative_gap" in gap


# ----------------------------------------------------------------- the stage


def test_the_stage_refuses_a_run_that_names_no_pre_registered_decision():
    stage = _stage()
    parser = stage.build_parser()
    args = parser.parse_args([])
    with pytest.raises(ValueError) as raised:
        stage.resolve(args)
    for flag in ("--layers", "--overlap-statistic", "--decision-rule", "--modes"):
        assert flag in str(raised.value)


def test_the_stage_requires_a_context_information_figure_for_every_measured_mode():
    stage = _stage()
    parser = stage.build_parser()
    args = parser.parse_args(
        [
            "--checkpoint", "/nowhere",
            "--rendering", "prollama",
            "--modes", "text", "protein",
            "--layers", "16",
            "--overlap-statistic", "mean_squared_cosine",
            "--decision-rule", "residual_licensed_v1",
            "--context-information", "text=0.8336",
        ]
    )
    with pytest.raises(ValueError, match="protein"):
        stage.resolve(args)


def test_the_stage_refuses_a_campaign_flag_beside_the_synthetic_check():
    stage = _stage()
    parser = stage.build_parser()
    args = parser.parse_args(
        [
            "--synthetic",
            "--overlap-statistic", "mean_squared_cosine",
            "--decision-rule", "residual_licensed_v1",
            "--layers", "16",
        ]
    )
    with pytest.raises(ValueError, match="--layers"):
        stage.resolve(args)


def test_the_synthetic_check_still_requires_the_rule_it_validates():
    stage = _stage()
    parser = stage.build_parser()
    args = parser.parse_args(["--synthetic", "--overlap-statistic", "mean_squared_cosine"])
    with pytest.raises(ValueError, match="--decision-rule"):
        stage.resolve(args)


def test_the_stage_refuses_a_draw_that_would_read_the_format_separator():
    stage = _stage()
    parser = stage.build_parser()
    args = parser.parse_args(
        [
            "--synthetic",
            "--overlap-statistic", "mean_squared_cosine",
            "--decision-rule", "residual_licensed_v1",
            "--drop-leading", "0",
        ]
    )
    with pytest.raises(ValueError, match="drop-leading"):
        stage.resolve(args)


def test_the_stage_accepts_the_uniform_external_baseline_flags():
    # `--device cuda:N` and `--out DIR` are the contract every stage the H200
    # dispatcher launches is invoked under.
    stage = _stage()
    args = stage.build_parser().parse_args(
        [
            "--synthetic",
            "--overlap-statistic", "mean_squared_cosine",
            "--decision-rule", "residual_licensed_v1",
            "--device", "cuda:3",
            "--out", "/tmp/does-not-need-to-exist",
        ]
    )
    assert args.device == "cuda:3"
    assert Path(args.out) == Path("/tmp/does-not-need-to-exist")


def test_the_rank_ladder_refuses_rank_zero_and_a_descending_ladder():
    with pytest.raises(ValueError, match="clean pass"):
        ms.parse_rank_ladder("0,4")
    with pytest.raises(ValueError, match="ascending"):
        ms.parse_rank_ladder("8,4")
    with pytest.raises(ValueError, match="twice"):
        ms.parse_rank_ladder("4,4")
    assert ms.parse_rank_ladder("1, 2,4") == (1, 2, 4)


def test_the_stage_derives_its_output_name_from_the_checkpoint_modes_and_layers():
    # A fixed basename lets a second cell overwrite the first, and this stage runs
    # once per checkpoint per layer set.
    source = (REPO_ROOT / "scripts" / "transfer" / "38_mode_subspaces.py").read_text()
    assert 'args.rendering' in source and 'mode_subspaces__' in source
    stem = source.split('destination = args.out / (\n            "mode_subspaces__"', 1)[1]
    for piece in ("clean_name", "args.modes", "args.layers", "args.tensor"):
        assert piece in stem.split("write_json", 1)[0]


def test_the_stage_names_its_pre_registration_and_still_refuses_to_call_it_admission():
    # The entry that froze the design is named, so a reader can find the criteria
    # this run was decided under. What the entry does NOT do is admit a result, and
    # the artefact has to keep saying so: the constant used to carry a refusal, and
    # replacing a refusal with an identifier alone would discharge more than the
    # pre-registration actually discharges.
    assert ms.PRE_REGISTRATION == "EXP-R2-215"
    assert ms.PRE_REGISTRATION in ms.PRE_REGISTRATION_STATUS
    assert "docs/EXPERIMENT_LOG.md" in ms.PRE_REGISTRATION_SCOPE
    stage = _stage()
    assert (
        stage.LIMITATIONS["pre_registration_is_not_admission"]
        == ms.PRE_REGISTRATION_SCOPE
    )


def test_the_named_pre_registration_exists_in_the_experiment_log():
    # A stage may not name an identifier the log does not carry: that is the one
    # failure mode worse than naming none at all.
    log = (REPO_ROOT / "docs" / "EXPERIMENT_LOG.md").read_text(encoding="utf-8")
    assert f"## 2026-08-19 — {ms.PRE_REGISTRATION} pre-registered" in log


def test_the_decision_rule_thresholds_cannot_be_passed_on_the_command_line():
    # A threshold that can be passed can be passed again, after the numbers exist.
    stage = _stage()
    flags = {
        action.option_strings[0]
        for action in stage.build_parser()._actions
        if action.option_strings
    }
    for forbidden in (
        "--necessity-fraction",
        "--residual-share-floor",
        "--overlap-margin",
        "--logit-tolerance",
    ):
        assert forbidden not in flags


def test_the_written_artefact_carries_the_per_layer_guard_and_the_limitations():
    stage = _stage()
    for key in (
        "objective_scope",
        "pre_registration_is_not_admission",
        "eigen_order_identifiability",
        "unigram_estimator",
        "resampling_unit",
        "no_token_alignment_required",
    ):
        assert key in stage.LIMITATIONS
    assert "7.0" in stage.LIMITATIONS["objective_scope"]


def test_every_per_layer_vector_the_stage_writes_carries_the_guarded_suffix():
    # `assert_per_layer_fields` matches keys ending in `_per_site` and nothing
    # else, so a vector written as `"per_site"` -- eight characters, not nine --
    # walks straight past it. That is the exact shape of the defect L32 records:
    # the guard runs, reports nothing, and the collapse stays invisible. A bare
    # `per_site` key in this stage is therefore a defect and not a style choice.
    source = (REPO_ROOT / "scripts" / "transfer" / "38_mode_subspaces.py").read_text()
    assert '"per_site"' not in source
    for key in (
        "occupancy_per_site",
        "eigen_gap_per_site",
        "necessity_per_site",
        "invariants_per_site",
        "overlap_per_site",
        "verdict_per_site",
    ):
        assert f'"{key}"' in source


def test_the_stage_writes_a_readable_synthetic_artefact_end_to_end(tmp_path):
    # The whole entry point, through the same `write_json` every artefact of this
    # programme is written by -- which rejects NaN and infinity, so a non-finite
    # intermediate anywhere in the pipeline stops here rather than being plotted.
    stage = _stage()
    argv = [
        "38_mode_subspaces.py",
        "--synthetic",
        "--overlap-statistic", "mean_squared_cosine",
        "--decision-rule", "residual_licensed_v1",
        "--rank-ladder", "1,12",
        "--n-bootstrap", str(FIXTURE_BOOTSTRAP),
        "--device", "cpu",
        "--out", str(tmp_path),
    ]
    original = sys.argv
    sys.argv = argv
    try:
        stage.main()
    finally:
        sys.argv = original
    written = list(tmp_path.glob("mode_subspaces__synthetic_check__*.json"))
    assert len(written) == 1
    payload = json.loads(written[0].read_text())
    assert payload["schema_version"] == ms.SCHEMA_VERSION
    assert payload["passed"] is True
    assert payload["certificate"]["certificate"] == "PASSED"
    assert payload["pre_registration"]["status"] == ms.PRE_REGISTRATION_STATUS
    assert payload["pre_registration"]["entry"] == "EXP-R2-215"
    assert "objective_scope" in payload["limitations"]
    assert "pre_registration_is_not_admission" in payload["limitations"]
    assert payload["pre_registration"]["decision_rule"]["name"] == "residual_licensed_v1"
    assert payload["provenance"]["runner"]["sha256"]
    assert set(payload["provenance"]["modules"]) >= {"src/transfer/mode_subspaces.py"}
    assert "objective_scope" in payload["limitations"]
    # The campaign-only flags describe nothing about a synthetic run and must not
    # reach the artefact as nulls beside the settings that do describe it.
    for flag in stage.CAMPAIGN_ONLY_FLAGS:
        assert flag not in payload["settings"]


def test_a_collapsed_per_site_field_is_refused_before_the_artefact_is_written():
    from src.transfer.crosscoder import assert_per_layer_fields

    good = {"verdict_per_site": [{"layer": 0}, {"layer": 1}]}
    assert_per_layer_fields(good, n_sites=2)
    with pytest.raises(ValueError, match="per-site field"):
        assert_per_layer_fields({"verdict_per_site": 3}, n_sites=2)
    with pytest.raises(ValueError, match="values for 2"):
        assert_per_layer_fields({"verdict_per_site": [{"layer": 0}]}, n_sites=2)
