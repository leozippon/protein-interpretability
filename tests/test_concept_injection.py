"""Conditions D3.g's causal stage must always hold, and its negative paths.

Written against the frozen criteria of EXP-R2-213 (A36-0 to A36-6) and against the
properties rather than the implementation, so a re-implementation that keeps them
keeps these tests. Four are the programme's own hard-won lessons rather than
generic hygiene:

* an intervention hook that fails to bind passes every null test while silently
  measuring an unpatched model (``path_patching`` records that happening), so the
  positive control is tested as carefully as the null;
* a behavioural read on ``Llama-2-7b-hf``'s protein mode measures the cohort and
  not the model (EXP-R2-152's -0.0013 nats reversal cost), so it must raise;
* a concept direction refitted to permuted labels must not pass A36-3, because a
  direction that does is carrying something the label assignment does not;
* A36-4 is per concept against its own row's 95th percentile, and a row mean may
  not substitute for it -- that substitution is L32 exactly.

``torch.set_num_threads(1)`` is set for the same measured reason the stage sets it
on its synthetic path: the toy model is 64-dimensional and the host has 96 cores,
where one forward costs 3.2 s in thread launch against 0.004 s single-threaded. It
changes no number.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import concept_alignment as ca  # noqa: E402
from src.transfer import concept_injection as ci  # noqa: E402


def _load_stage():
    path = REPO_ROOT / "scripts" / "transfer" / "36_concept_injection.py"
    spec = importlib.util.spec_from_file_location("_stage_36", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE = _load_stage()
LADDER = ci.FROZEN_ALPHA_LADDER


@pytest.fixture(scope="module")
def planted():
    """A toy checkpoint whose response to the planted directions is known.

    Eight orthonormal concept directions, a present and an absent token block per
    concept per mode, and a head that reads each direction in both modes, so
    writing ``alpha * sigma * u_k`` at the feed-forward's input makes ``Delta``
    negative, graded in ``alpha`` and confined to concept ``k``.

    The width and the concept count are not free: A36-3(b)'s bar is set by
    ISOTROPIC random directions, whose component on a planted direction falls as
    one over the root of the width, while a permuted-label refit lands in the span
    of the planted directions and its component falls as one over the root of the
    concept count. At three concepts and width 32 the worst of eight permuted
    refits carried 0.89 of the planted direction and passed every clause of A36-3
    on this fixture, so the known-answer test had no teeth against A36-5. Eight
    concepts at width 128 is where the two curves separate.
    """

    model = ci.SyntheticConceptModel(d_model=128, n_layers=3, n_concepts=8, seed=7)
    handles = {
        mode: ci.SyntheticJointHandle(model, mode=mode) for mode in ("text", "protein")
    }
    records, names = ci.synthetic_records(model, n_records=192, span=6, seed=7)
    batches = {
        mode: ci.synthetic_batches(handles[mode], records, batch_size=8)
        for mode in ("text", "protein")
    }
    bearing = {name: ci.bearing_flags(records, name) for name in names}
    representations, _ = ci.pooled_representations(handles["text"], batches["text"], layer=1)
    directions = {
        name: ci.ConceptDirection.from_concept_vector(
            ca.concept_vector(representations, bearing[name]),
            concept=name,
            layer=1,
            provenance="test",
        )
        for name in names
    }
    baseline = {
        mode: ci.scored_response(handles[mode], batches[mode], layer=1, delta=None)
        for mode in ("text", "protein")
    }
    return {
        "model": model,
        "handles": handles,
        "records": records,
        "names": names,
        "batches": batches,
        "bearing": bearing,
        "representations": representations,
        "directions": directions,
        "baseline": baseline,
    }


def _response(planted, direction, alpha, *, mode="protein"):
    return ci.scored_response(
        planted["handles"][mode],
        planted["batches"][mode],
        layer=1,
        delta=direction.delta(alpha, device="cpu", dtype=torch.float32),
    )


def _delta(planted, direction, alpha, *, scored, mode="protein"):
    return ci.delta_nll_shift(
        planted["baseline"][mode],
        _response(planted, direction, alpha, mode=mode),
        planted["bearing"][scored],
        seed=11,
        n_bootstrap=200,
    )


# ------------------------------------------------------------ A36-0 invariants


def test_a_null_patch_is_a_no_op_and_a_large_random_patch_moves_the_logits(planted):
    record = ci.invariants(
        planted["handles"]["protein"],
        planted["batches"]["protein"][0],
        layer=1,
        scale=10.0,
        seed=3,
    )
    assert record["criterion"] == "A36-0"
    assert record["null_patch_max_logit_gap"] == pytest.approx(0.0, abs=1e-6)
    assert record["perturbed_patch_max_logit_gap"] > record["tolerance"]
    assert record["site"] == ci.INJECTION_SITE
    assert record["n_written_positions"] > 0


def test_the_positive_control_refuses_a_perturbation_that_does_not_move_the_logits(planted):
    """The check that catches an unbound hook, exercised through its own refusal.

    A perturbation small enough to move the logits by less than the tolerance is
    indistinguishable, to this check, from a hook that never bound -- and that is
    the point: it must raise rather than proceed, because a null-only invariant
    cannot see the difference.
    """

    with pytest.raises(RuntimeError, match="not bound to the site"):
        ci.invariants(
            planted["handles"]["protein"],
            planted["batches"]["protein"][0],
            layer=1,
            scale=1e-9,
            seed=3,
        )


def test_delta_responds_to_the_direction_and_not_to_the_write_norm(planted):
    """The check that would expose a Delta driven by the size of the write.

    If Delta responded to ``|alpha * sigma|`` rather than to where the write
    points, then every norm-matched vector would move it equally: A36-3(b)'s
    random-direction control could not discriminate, A36-5's permuted control could
    not either, and a positive protein result would mean nothing. So three writes
    of IDENTICAL norm are compared -- the planted direction, a direction orthogonal
    to every planted direction, and the isotropic random control -- and the planted
    one has to dominate both.
    """

    name = planted["names"][0]
    true = planted["directions"][name]
    basis = planted["model"].concept_directions.numpy().astype(np.float64)
    span, _ = np.linalg.qr(
        np.concatenate(
            [basis.T, np.random.default_rng(0).standard_normal((true.d_model, 4))], axis=1
        )
    )
    orthogonal = ci.ConceptDirection(
        concept=name,
        layer=1,
        direction=span[:, basis.shape[0]],
        sigma=true.sigma,
        provenance="orthogonal to every planted direction, at the planted sigma",
    )
    assert np.abs(basis @ orthogonal.direction).max() < 1e-6

    def point(direction):
        return ci.delta_point(
            planted["baseline"]["protein"],
            _response(planted, direction, 1.0),
            planted["bearing"][name],
        )

    planted_delta = point(true)
    orthogonal_delta = point(orthogonal)
    control = ci.random_direction_control(
        [point(draw) for draw in ci.norm_matched_random_directions(true, n_draws=8, seed=5)]
    )
    assert planted_delta < 0.0
    assert abs(orthogonal_delta) < 0.25 * abs(planted_delta), (
        planted_delta,
        orthogonal_delta,
    )
    assert abs(planted_delta) > ci.FROZEN_RANDOM_NULL_MARGIN * control["absolute_p95"], (
        planted_delta,
        control,
    )


def test_a_cohort_too_small_for_the_per_side_floor_is_refused(planted):
    with pytest.raises(ValueError, match="cannot put 8 near-duplicate groups"):
        ci.synthetic_records(planted["model"], n_records=8, span=6, seed=1)


def test_a_direction_of_the_wrong_width_cannot_be_written(planted):
    wrong = ci.ConceptDirection(
        concept="wrong-width",
        layer=1,
        direction=np.eye(1, 8)[0],
        sigma=1.0,
        provenance="test",
    )
    with pytest.raises(TypeError, match="estimated at a different width"):
        ci.scored_response(
            planted["handles"]["protein"],
            planted["batches"]["protein"],
            layer=1,
            delta=wrong.delta(1.0, device="cpu", dtype=torch.float32),
        )


def test_a_non_unit_direction_is_refused_because_it_rescales_the_ladder():
    with pytest.raises(ValueError, match="rather than 1"):
        ci.ConceptDirection(
            concept="c", layer=0, direction=np.array([1.0, 1.0]), sigma=1.0, provenance="t"
        )


# -------------------------------------------------------------------- refusals


def test_a_behavioural_read_on_llama2_protein_mode_raises(tmp_path):
    with pytest.raises(ValueError) as error:
        ci.require_behavioural_modes(tmp_path / "Llama-2-7b-hf")
    message = str(error.value)
    assert "protein mode" in message
    assert "-0.0013" in message


def test_an_undeclared_checkpoint_is_refused_rather_than_assumed_measurable(tmp_path):
    with pytest.raises(ValueError):
        ci.require_behavioural_modes(tmp_path / "some-unqualified-checkpoint")


def test_the_qualified_prollama_stages_are_admitted(tmp_path):
    for name in ("ProLLaMA_Stage_1", "ProLLaMA"):
        record = ci.require_behavioural_modes(tmp_path / name)
        assert record["status"]["measurable"] is True


def test_half_precision_is_refused_on_this_estimand():
    """Built fresh rather than casting the shared fixture.

    A round trip through bfloat16 and back is lossy, so casting the module-scoped
    model would leave every later test measuring a differently-quantised model --
    which is a smaller version of the defect this refusal exists to prevent.
    """

    model = ci.SyntheticConceptModel(d_model=16, n_layers=2, n_concepts=2, seed=1)
    handle = ci.SyntheticJointHandle(model.to(torch.bfloat16), mode="protein")
    with pytest.raises(ValueError, match="Appendix B rule 15b"):
        ci.require_full_precision(handle)
    assert ci.require_full_precision(
        ci.SyntheticJointHandle(model.to(torch.float32), mode="protein")
    ) == "float32"


def test_a_description_that_still_names_its_concept_is_refused():
    records = [
        {"accession": "A", "description_masked": "catalyses the hydrolysis of ATP"},
        {"accession": "B", "description_masked": "a Hydrolase acting on ester bonds"},
    ]
    with pytest.raises(ValueError, match="masked description"):
        ci.assert_descriptions_masked(records, "hydrolase")
    assert ci.assert_descriptions_masked(records[:1], "hydrolase")["n_checked"] == 1


def test_a_substring_inside_a_longer_word_is_not_a_leak():
    records = [{"accession": "A", "description_masked": "binds dehydrolaseless motifs"}]
    assert ci.assert_descriptions_masked(records, "hydrolase")["n_checked"] == 1


# ------------------------------------------------- A36-1, A36-3: the ladder


def test_the_frozen_ladder_is_symmetric_contains_zero_and_has_nine_rungs():
    assert len(ci.FROZEN_ALPHA_LADDER) == 9
    assert 0.0 in ci.FROZEN_ALPHA_LADDER
    assert sorted(-value for value in ci.FROZEN_ALPHA_LADDER) == list(
        ci.FROZEN_ALPHA_LADDER
    )
    assert list(ci.FROZEN_ALPHA_LADDER) == sorted(ci.FROZEN_ALPHA_LADDER)


def test_delta_is_zero_at_alpha_zero_and_negative_and_graded_above_it(planted):
    """The sign convention is the criterion, so it is tested as one.

    Delta is the NLL shift of A36-3, so a working concept vector makes it negative
    at positive alpha, zero at zero by construction, positive below zero, and
    monotonically decreasing across the ladder -- Spearman at most -0.8.
    """

    name = planted["names"][0]
    direction = planted["directions"][name]
    values = {
        alpha: _delta(planted, direction, alpha, scored=name)["delta_nats_per_token"]
        for alpha in LADDER
    }
    assert values[0.0] == pytest.approx(0.0, abs=1e-9)
    assert values[1.0] < 0.0 < values[-1.0], values
    assert values[2.0] < values[1.0], values
    graded = ci.graded_record(values)
    assert graded["spearman"] is not None
    assert graded["spearman"] <= ci.FROZEN_SPEARMAN_CEILING, graded
    assert graded["sign_reverses_below_zero"] is True
    assert graded["monotone_step_fraction"] == 1.0


def test_the_coherence_floor_is_read_on_non_bearing_sequences_only(planted):
    name = planted["names"][0]
    direction = planted["directions"][name]
    coherence = {}
    for alpha in LADDER:
        coherence[alpha] = ci.coherence_record(
            planted["baseline"]["protein"],
            _response(planted, direction, alpha),
            planted["bearing"][name],
        )
    assert coherence[0.0]["non_bearing_nll_inflation_nats_per_token"] == pytest.approx(
        0.0, abs=1e-9
    )
    assert coherence[0.0]["n_non_bearing"] == int((~planted["bearing"][name]).sum())
    # The bound is applied to the non-bearing figure, so a concept that helps its
    # bearers cannot buy itself admissibility.
    assert "bearing_nll_inflation_nats_per_token" in coherence[4.0]
    wide = ci.admissible_alphas(coherence, max_nll_inflation=1e6)
    assert wide == tuple(sorted(LADDER))
    assert ci.admissible_alphas(coherence, max_nll_inflation=1e-12) != wide


def test_a_ladder_the_coherence_bound_empties_leaves_no_positive_rung(planted):
    name = planted["names"][0]
    coherence = {
        alpha: ci.coherence_record(
            planted["baseline"]["protein"],
            _response(planted, planted["directions"][name], alpha),
            planted["bearing"][name],
        )
        for alpha in LADDER
    }
    admissible = ci.admissible_alphas(coherence, max_nll_inflation=1e-12)
    assert [alpha for alpha in admissible if alpha > 0.0] == []


def test_the_group_floor_is_per_side_and_not_on_the_total():
    groups = ["g%02d" % index for index in range(20)]
    bearing = [index < 2 for index in range(20)]
    counts = ci.per_side_group_counts(groups, bearing)
    assert counts == {"bearing": 2, "non_bearing": 18}
    with pytest.raises(ValueError, match="per SIDE"):
        ci.require_per_side_group_floor(groups, bearing, label="two bearers")
    balanced = [index < 10 for index in range(20)]
    assert ci.require_per_side_group_floor(groups, balanced, label="balanced") == {
        "bearing": 10,
        "non_bearing": 10,
    }


# ------------------------------------------------------- A36-3(b), A36-5 nulls


def test_the_random_direction_control_refuses_fewer_than_eight_directions():
    with pytest.raises(ValueError, match="direction floor"):
        ci.random_direction_control([0.1] * (ci.MINIMUM_CONTROL_DIRECTIONS - 1))
    control = ci.random_direction_control(list(np.linspace(-0.1, 0.1, 12)))
    assert control["n_directions"] == 12
    assert control["absolute_p95"] > 0.0


def test_both_controls_refuse_fewer_than_eight_distinct_directions(planted):
    name = planted["names"][0]
    with pytest.raises(ValueError, match="direction floor"):
        ci.norm_matched_random_directions(planted["directions"][name], n_draws=4, seed=1)
    with pytest.raises(ValueError, match="direction floor"):
        ci.permuted_label_directions(
            planted["representations"],
            planted["bearing"][name],
            concept=name,
            layer=1,
            n_draws=2,
            seed=1,
        )


def test_random_directions_are_norm_matched_and_distinct(planted):
    name = planted["names"][0]
    draws = ci.norm_matched_random_directions(planted["directions"][name], n_draws=8, seed=5)
    assert len({tuple(np.round(draw.direction, 9)) for draw in draws}) == 8
    for draw in draws:
        assert draw.sigma == planted["directions"][name].sigma
        assert np.linalg.norm(draw.direction) == pytest.approx(1.0)


def test_permuted_concept_vectors_do_not_pass_a36_3_on_the_planted_effect(planted):
    """A36-5, on data where the answer is known, with its margin asserted.

    The planted direction must clear A36-3 and every permuted-label refit must
    fail it, through the same function, against the same random-direction control,
    at the same rungs. The worst refit's margin is asserted rather than left
    implicit, because a known-answer test whose separation is one seed wide is a
    test a wrong answer also passes: at three concepts and width 32 a permuted
    refit cleared every clause here, which is why the fixture's geometry is what
    it is.
    """

    name = planted["names"][0]
    direction = planted["directions"][name]
    controls = {}
    deltas = {}
    for alpha in LADDER:
        deltas[alpha] = _delta(planted, direction, alpha, scored=name)
        controls[alpha] = ci.random_direction_control(
            [
                ci.delta_point(
                    planted["baseline"]["protein"],
                    _response(planted, draw, alpha),
                    planted["bearing"][name],
                )
                for draw in ci.norm_matched_random_directions(direction, n_draws=8, seed=5)
            ]
        )
    coherence = {
        alpha: ci.coherence_record(
            planted["baseline"]["protein"],
            _response(planted, direction, alpha),
            planted["bearing"][name],
        )
        for alpha in LADDER
    }
    admissible = ci.admissible_alphas(coherence, max_nll_inflation=1e6)
    observed = ci.evaluate_a36_3(
        deltas=deltas,
        controls=controls,
        admissible=admissible,
        margin=ci.FROZEN_RANDOM_NULL_MARGIN,
        spearman_ceiling=ci.FROZEN_SPEARMAN_CEILING,
    )
    assert observed["passed"] is True, observed
    assert ci.operating_alpha(observed, rule=ci.OPERATING_ALPHA_RULES[0]) is not None

    worst = 0.0
    for draw in ci.permuted_label_directions(
        planted["representations"],
        planted["bearing"][name],
        concept=name,
        layer=1,
        n_draws=16,
        seed=5,
    ):
        permuted = ci.evaluate_a36_3(
            deltas={
                alpha: _delta(planted, draw, alpha, scored=name) for alpha in LADDER
            },
            controls=controls,
            admissible=admissible,
            margin=ci.FROZEN_RANDOM_NULL_MARGIN,
            spearman_ceiling=ci.FROZEN_SPEARMAN_CEILING,
        )
        assert permuted["passed"] is False, (draw.concept, permuted)
        for cell in permuted["per_alpha"].values():
            worst = max(
                worst,
                abs(cell["delta_nats_per_token"]) / cell["b_control_bar"],
            )
    # The separation, stated as a number: no permuted refit may come within a
    # factor of the control bar it has to clear, and the planted direction must sit
    # above it.
    assert worst < 0.9, worst
    best = max(
        abs(observed["per_alpha"][key]["delta_nats_per_token"])
        / observed["per_alpha"][key]["b_control_bar"]
        for key in observed["per_alpha"]
    )
    assert best > 1.0 and best > worst, (best, worst)


# -------------------------------------------------------------- A36-4 specificity


def test_the_specificity_matrix_is_square_and_its_diagonal_is_the_intended_concept(planted):
    names = planted["names"]
    cells = {}
    for injected in names:
        response = _response(planted, planted["directions"][injected], 1.0)
        for scored in names:
            cells[(injected, scored)] = ci.delta_point(
                planted["baseline"]["protein"], response, planted["bearing"][scored]
            )
    matrix = ci.specificity_matrix(cells, names, rule=ci.SPECIFICITY_RULES[0])
    assert matrix["criterion"] == "A36-4"
    assert matrix["concepts"] == names
    assert len(matrix["delta_matrix"]) == len(names)
    assert all(len(row) == len(names) for row in matrix["delta_matrix"])
    # The effect matrix is -Delta, so the diagonal is the largest entry of its own
    # row when the injected concept is the one that moved.
    for index, name in enumerate(names):
        row = matrix["per_concept"][name]
        assert row["diagonal_effect"] == pytest.approx(-matrix["delta_matrix"][index][index])
        assert len(row["off_diagonal_effect"]) == len(names) - 1
    assert matrix["admitted_concepts"] == names, matrix["per_concept"]
    assert "mean_diagonal_minus_mean_off_diagonal" in matrix["reported_but_not_a_criterion"]


def test_an_incomplete_single_concept_or_unknown_rule_matrix_is_refused():
    with pytest.raises(ValueError, match="at least two distinct concepts"):
        ci.specificity_matrix({("a", "a"): 0.0}, ["a"], rule=ci.SPECIFICITY_RULES[0])
    with pytest.raises(ValueError, match="incomplete"):
        ci.specificity_matrix(
            {("a", "a"): 0.0, ("b", "b"): 0.0}, ["a", "b"], rule=ci.SPECIFICITY_RULES[0]
        )
    with pytest.raises(ValueError, match="unknown specificity rule"):
        ci.specificity_matrix(
            {(x, y): 0.0 for x in "ab" for y in "ab"}, ["a", "b"], rule="row_mean"
        )


def test_a_row_whose_diagonal_does_not_beat_its_own_p95_is_not_admitted():
    names = ["a", "b", "c"]
    cells = {
        ("a", "a"): -1.0,
        ("a", "b"): -0.1,
        ("a", "c"): -0.1,
        ("b", "a"): -0.5,
        ("b", "b"): -0.2,
        ("b", "c"): -0.1,
        ("c", "a"): 0.0,
        ("c", "b"): 0.0,
        ("c", "c"): 0.0,
    }
    matrix = ci.specificity_matrix(cells, names, rule=ci.SPECIFICITY_RULES[0])
    assert matrix["per_concept"]["a"]["admitted"] is True
    assert matrix["per_concept"]["b"]["admitted"] is False
    assert matrix["per_concept"]["c"]["admitted"] is False
    assert matrix["admitted_concepts"] == ["a"]


# ------------------------------------------------------------ STOP-36 branches


def _verdict_inputs(**overrides):
    firing = {
        "condition_a_and_b": True,
        "a_and_b_firing_alphas": [1.0],
        "condition_a": True,
        "condition_b": True,
        "condition_c": True,
        "passed": True,
        "graded": {"spearman": -1.0, "sign_reverses_below_zero": True},
    }
    base = {
        "concept": "c",
        "text_control": dict(firing),
        "protein": dict(firing),
        "permuted_passes": [],
        "specificity_row": {
            "admitted": True,
            "diagonal_effect": 0.4,
            "off_diagonal_p95": 0.05,
        },
        "admissible": (-1.0, 0.0, 1.0),
    }
    base.update(overrides)
    return base


def test_a_failing_text_control_voids_the_run_before_anything_else_is_read():
    failing = {
        "condition_a_and_b": False,
        "a_and_b_firing_alphas": [],
        "condition_a": False,
        "condition_b": False,
        "condition_c": False,
        "passed": False,
        "graded": {"spearman": 0.0, "sign_reverses_below_zero": False},
    }
    outcome = ci.verdict(**_verdict_inputs(text_control=failing))
    assert outcome["outcome"] == "VOID_INSTRUMENT"
    # Even with everything else failing, the instrument branch is the one reported.
    outcome = ci.verdict(
        **_verdict_inputs(
            text_control=failing, protein=failing, permuted_passes=["c::permuted_000"]
        )
    )
    assert outcome["outcome"] == "VOID_INSTRUMENT"


def test_each_stop_36_branch_is_reachable_and_distinct():
    assert ci.verdict(**_verdict_inputs())["outcome"] == "TRANSFERS"
    assert (
        ci.verdict(**_verdict_inputs(permuted_passes=["c::permuted_003"]))["outcome"]
        == "VOID_PERMUTED_CONTROL_PASSES"
    )
    assert (
        ci.verdict(**_verdict_inputs(admissible=(-1.0, 0.0)))["outcome"]
        == "NO_ADMISSIBLE_COEFFICIENT_RANGE"
    )
    failing = dict(_verdict_inputs()["protein"])
    failing["passed"] = False
    failing["condition_c"] = False
    assert ci.verdict(**_verdict_inputs(protein=failing))["outcome"] == "MEASURED_NEGATIVE"
    unspecific = {"admitted": False, "diagonal_effect": 0.01, "off_diagonal_p95": 0.4}
    assert (
        ci.verdict(**_verdict_inputs(specificity_row=unspecific))["outcome"]
        == "NULL_NO_CONCEPT_CLEARS_ITS_ROW"
    )
    assert set(ci.OUTCOMES) >= {
        "TRANSFERS",
        "VOID_INSTRUMENT",
        "VOID_PERMUTED_CONTROL_PASSES",
        "NO_ADMISSIBLE_COEFFICIENT_RANGE",
        "MEASURED_NEGATIVE",
        "NULL_NO_CONCEPT_CLEARS_ITS_ROW",
    }


# ---------------------------------------------------------------- A36-6 readout B


def test_the_pfam_referent_comes_from_the_fit_split_and_can_be_empty():
    records = [
        {"accession": "A", "pfam": ["PF00069.28", "PF07714"]},
        {"accession": "B", "pfam": ["PF00069.28"]},
        {"accession": "C", "pfam": ["PF00001"]},
    ]
    bearing = [True, True, False]
    assert ci.pfam_referent(records, bearing, min_bearing_records=2) == ("PF00069",)
    assert ci.pfam_referent(records, bearing, min_bearing_records=3) == ()
    assert ci.pfam_referent(records, [False, False, True], min_bearing_records=1) == (
        "PF00001",
    )


def test_annotation_rates_report_attainability_before_the_concept_rate():
    hits = {
        "s0": [{"accession_unversioned": "PF00069", "evalue": 1e-30}],
        "s1": [{"accession_unversioned": "PF99999", "evalue": 1e-10}],
    }
    names = ["s0", "s1", "s2"]
    rates = ci.annotation_rates(hits, names, ["PF00069"])
    assert rates["any_family_rate"] == pytest.approx(2 / 3)
    assert rates["concept_family_rate"] == pytest.approx(1 / 3)
    assert rates["per_sequence_concept_hit"] == {"s0": True, "s1": False, "s2": False}


def test_the_annotation_rate_contrast_refuses_fewer_than_eight_sequences_a_side():
    small = {
        "n_sequences": 4,
        "concept_family_rate": 0.5,
        "per_sequence_concept_hit": {f"s{i}": i < 2 for i in range(4)},
    }
    with pytest.raises(ValueError, match="usable sequences"):
        ci.annotation_rate_contrast(small, small, seed=1, n_bootstrap=100)
    injected = {
        "n_sequences": 16,
        "concept_family_rate": 0.75,
        "per_sequence_concept_hit": {f"i{i}": i < 12 for i in range(16)},
    }
    baseline = {
        "n_sequences": 16,
        "concept_family_rate": 0.125,
        "per_sequence_concept_hit": {f"b{i}": i < 2 for i in range(16)},
    }
    contrast = ci.annotation_rate_contrast(injected, baseline, seed=1, n_bootstrap=400)
    assert contrast["rate_difference"] == pytest.approx(0.625)
    assert contrast["excludes_zero"] is True
    assert contrast["criterion"] == "A36-6"


def test_a_generated_continuation_that_wandered_into_prose_contributes_little():
    assert ci.extract_generated_sequence("MKVLA> and then", end_delimiter=">") == "MKVLA"
    assert ci.extract_generated_sequence("MKV lorem ipsum", end_delimiter=">") == "MKV"
    assert ci.extract_generated_sequence("hello", end_delimiter=">") == ""


# ----------------------------------------------------------- the stage contract


def _stage_args(**overrides):
    args = argparse.Namespace(
        checkpoint=Path("/models/ProLLaMA_Stage_1"),
        rendering="prollama",
        cohort=Path("/cohort"),
        concepts=(("go_propagated", "GO:0016787"), ("ec", "3.2.1")),
        pooling="mean_content",
        eval_split="eval",
        generation_readout="refused",
        generation_refusal_reason="declared for the test",
        layer=16,
        alphas=ci.FROZEN_ALPHA_LADDER,
        injection_site=ci.INJECTION_SITE,
        coherence_max_nll_inflation=ci.COHERENCE_PRIMARY_BOUND,
        spearman_ceiling=ci.FROZEN_SPEARMAN_CEILING,
        random_null_margin=ci.FROZEN_RANDOM_NULL_MARGIN,
        random_directions=8,
        permuted_directions=8,
        specificity_rule=ci.SPECIFICITY_RULES[0],
        operating_alpha_rule=ci.OPERATING_ALPHA_RULES[0],
        max_concepts=None,
        synthetic=False,
        synthetic_concepts=3,
        hmmer_root=None,
        pfam_root=None,
        generate_sequences=32,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_resolve_accepts_the_frozen_configuration():
    STAGE.resolve(_stage_args())


def test_resolve_names_every_missing_frozen_criterion():
    args = _stage_args()
    for flag in STAGE.PRE_REGISTERED_DECISIONS:
        setattr(args, flag, None)
    with pytest.raises(ValueError) as error:
        STAGE.resolve(args)
    message = str(error.value)
    for flag in STAGE.PRE_REGISTERED_DECISIONS:
        assert f"--{flag.replace('_', '-')}" in message, flag


def test_resolve_refuses_a_value_the_pre_registration_did_not_freeze():
    with pytest.raises(ValueError, match="freezes the ladder"):
        STAGE.resolve(_stage_args(alphas=(-1.0, 0.0, 1.0)))
    with pytest.raises(ValueError, match="freezes it at -0.8"):
        STAGE.resolve(_stage_args(spearman_ceiling=0.8))
    with pytest.raises(ValueError, match="freezes it at 2.0"):
        STAGE.resolve(_stage_args(random_null_margin=1.0))
    with pytest.raises(ValueError, match="freezes the primary bound"):
        STAGE.resolve(_stage_args(coherence_max_nll_inflation=0.9))
    with pytest.raises(ValueError, match="at least 8 distinct directions"):
        STAGE.resolve(_stage_args(random_directions=4))
    with pytest.raises(ValueError, match="at least 8 distinct directions"):
        STAGE.resolve(_stage_args(permuted_directions=7))


def test_resolve_requires_the_generation_refusal_to_carry_its_reason():
    with pytest.raises(ValueError, match="needs --generation-refusal-reason"):
        STAGE.resolve(_stage_args(generation_refusal_reason=None))
    with pytest.raises(ValueError, match="needs --hmmer-root and --pfam-root"):
        STAGE.resolve(_stage_args(generation_readout="hmmer_pfam"))


def test_resolve_refuses_campaign_flags_beside_the_synthetic_check():
    with pytest.raises(ValueError, match="meaningless beside --synthetic"):
        STAGE.resolve(_stage_args(synthetic=True))
    args = _stage_args(synthetic=True)
    for flag in STAGE.CAMPAIGN_ONLY_FLAGS:
        setattr(args, flag, None)
    STAGE.resolve(args)
    # The toy has no anchor for A36-1's 0.25, so the synthetic path may declare its
    # own coherence bound and records that it did; every dimensionless criterion is
    # still validated.
    STAGE.resolve(_stage_args(**{
        **{flag: None for flag in STAGE.CAMPAIGN_ONLY_FLAGS},
        "synthetic": True,
        "coherence_max_nll_inflation": 1.0,
    }))


def test_the_artefact_basename_is_derived_from_checkpoint_concepts_split_and_layer():
    """The collision bug 21_joint_mode_qualification.py carries must not be copied."""

    variants = [
        _stage_args(),
        _stage_args(layer=20),
        _stage_args(concepts=(("ec", "3.2.1"), ("ec", "2.7.11"))),
        _stage_args(eval_split="family_holdout"),
        _stage_args(checkpoint=Path("/models/ProLLaMA")),
    ]
    names = set()
    for args in variants:
        concepts = [STAGE.concept_name(concept) for concept in args.concepts]
        names.add(
            f"concept_injection__{Path(args.checkpoint).name}__{args.rendering}"
            f"__{args.eval_split}__L{args.layer:02d}"
            f"__{len(concepts)}concepts-{ci.digest_of(concepts)}.json"
        )
    assert len(names) == len(variants), names


def test_the_text_side_is_the_masked_description_and_is_not_configurable():
    assert STAGE.TEXT_FIELD == "description_masked"
    assert not any(
        action.dest == "text_field" for action in STAGE.build_parser()._actions
    )


def test_the_reduction_rule_drops_the_smallest_eval_group_counts_first():
    records = [
        {"accession": f"a{i}", "dup_group": f"g{i}", "ec": ["1.1.1"], "go": []}
        for i in range(20)
    ]
    for record in records[:3]:
        record["go"] = ["GO:rare"]
    for record in records[:12]:
        record["ec"] = ["1.1.1", "2.2.2"]
    args = _stage_args(
        concepts=(("ec", "1.1.1"), ("ec", "2.2.2"), ("go", "GO:rare")), max_concepts=2
    )
    kept, reduction = STAGE.select_concepts(args, records)
    assert reduction["dropped"] == ["go|GO:rare"]
    assert set(kept) == {"ec|1.1.1", "ec|2.2.2"}
    assert reduction["eval_group_counts_per_side"]["go|GO:rare"]["bearing"] == 3
