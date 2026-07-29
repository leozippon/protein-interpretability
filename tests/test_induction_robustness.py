"""Invariants of the threshold-robustness and scale-separation analysis.

This module exists because the induction-head finding is a *thresholded* claim
about a *seven-point* panel, and both of those are ways for an analysis to
produce a confident number from a design that cannot support one.  The
properties asserted here are the ones a reader has to be able to assume:

- the rank statistics really are threshold-free, so applying any strictly
  increasing transform to the head scores must not move them at all;
- a collinear design is refused rather than fitted, since a modality coefficient
  read off a panel where modality and scale are the same vector is the failure
  this analysis was built to avoid;
- the model-level permutation test enumerates every assignment and reports its
  own floor, so a p-value at 1/35 cannot be mistaken for strong evidence;
- the probe bootstrap resamples probes and nothing else, and in particular a
  degenerate interval on an arm whose heads are far from the cut is a correct
  answer rather than a bug;
- the sweep calls a broken separation broken.  A negative result that reports
  itself as a small positive one is worse than no analysis.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.induction_robustness import (  # noqa: E402
    COLLINEARITY_LIMIT,
    ArmCensus,
    auc,
    cluster_bootstrap_fraction,
    collinearity_report,
    model_level_exact_test,
    one_sided_ks,
    pairwise_auc,
    quantile_dominance,
    scale_modality_fit,
    survival_dominance,
    threshold_sweep,
)


def make_arm(
    name: str,
    modality: str,
    scores: np.ndarray,
    *,
    parameters: int = 100_000_000,
    d_model: int = 128,
) -> ArmCensus:
    layers, heads = scores.shape
    return ArmCensus(
        name=name,
        modality=modality,
        architecture="gpt2",
        n_layer=layers,
        n_head_per_layer=heads,
        d_model=d_model,
        parameters=parameters,
        probe="synthetic_repeat",
        scores=scores,
        uniform_baseline=0.01,
        n_probes=16,
        stored_counts={f"{t:.2f}": int((scores >= t).sum()) for t in (0.05, 0.10, 0.20, 0.30)},
        data_driven_threshold=float(scores.mean() + 3.0 * scores.std(ddof=1)),
        stored_data_driven_count=int(
            (scores >= scores.mean() + 3.0 * scores.std(ddof=1)).sum()
        ),
        source=Path("/dev/null"),
    )


def separated_panel(seed: int = 0) -> list[ArmCensus]:
    """Four text and three protein arms with a genuine upper-tail difference."""

    rng = np.random.default_rng(seed)
    arms: list[ArmCensus] = []
    for index, name in enumerate(("t1", "t2", "t3", "t4")):
        scores = rng.uniform(0.0, 0.02, size=(8, 8))
        scores.reshape(-1)[: 6 - index] = 0.5  # a shrinking but non-empty tail
        arms.append(make_arm(name, "text", scores, parameters=10 ** (8 + index * 0.3)))
    for index, name in enumerate(("p1", "p2", "p3")):
        scores = rng.uniform(0.0, 0.02, size=(8, 8))
        scores.reshape(-1)[: 1 - min(index, 1)] = 0.5
        arms.append(make_arm(name, "protein", scores, parameters=10 ** (8 + index * 0.3)))
    return arms


# ----------------------------------------------------------- threshold freedom


def test_auc_is_invariant_under_any_increasing_transform():
    """A rank statistic that moved under a rescaling would not be threshold-free.

    This is the whole justification for preferring the AUC to the fraction: if
    the answer depended on the units the scores are quoted in, it would depend on
    a choice as arbitrary as the cut-off it was meant to replace.
    """

    rng = np.random.default_rng(11)
    higher = rng.gamma(2.0, 1.0, size=200)
    lower = rng.gamma(1.5, 1.0, size=180)
    baseline = auc(higher, lower)
    for transform in (
        lambda x: 7.3 * x,
        lambda x: np.log1p(x),
        lambda x: np.sqrt(x),
        lambda x: x**3,
        lambda x: 1.0 / (1.0 + np.exp(-x)),
    ):
        assert auc(transform(higher), transform(lower)) == pytest.approx(baseline, abs=1e-12)


def test_auc_of_a_distribution_against_itself_is_one_half():
    rng = np.random.default_rng(3)
    sample = rng.normal(size=400)
    assert auc(sample, sample) == pytest.approx(0.5, abs=1e-12)


def test_one_sided_ks_finds_a_tail_difference_the_auc_misses():
    """The instrument has to match the claim: prevalence is a tail statement.

    Two distributions with the same bulk and different tails have an AUC close to
    one half -- the statistic the finding is *not* about -- while the one-sided KS
    statistic reports the tail gap and the cut where it lives.  If this ever
    stopped holding, the analysis would be reporting the bulk and calling it
    prevalence.
    """

    rng = np.random.default_rng(5)
    bulk = rng.uniform(0.0, 0.01, size=500)
    with_tail = bulk.copy()
    with_tail[:50] = 0.9
    without_tail = bulk.copy()
    without_tail[:5] = 0.9

    assert abs(auc(with_tail, without_tail) - 0.5) < 0.06
    result = one_sided_ks(with_tail, without_tail)
    assert result["d_plus"] == pytest.approx(0.09, abs=1e-9)
    assert result["cut_at_d_plus"] == pytest.approx(0.9)


def test_survival_and_quantile_dominance_agree_on_a_clean_separation():
    arms = separated_panel()
    survival = survival_dominance(arms)
    quantiles = quantile_dominance(arms)
    assert survival["separating_intervals"]
    assert survival["fraction_of_informative_grid_where_separation_holds"] > 0.0
    # The panel was built with a tail difference and no bulk difference, so the
    # upper quantiles must separate and the median must not.
    rows = {row["quantile"]: row["separates"] for row in quantiles["rows"]}
    assert rows[0.95] is True
    assert rows[0.5] is False


# ------------------------------------------------------------- sweep honesty


def test_sweep_reports_a_broken_separation_as_broken():
    """A crossing at one threshold must not be smoothed into a small positive."""

    # Text leads at 0.05 (half its heads clear it, against a quarter of protein's)
    # and is overtaken at 0.10, where none of its heads survive and protein's
    # quarter does. That crossing is exactly the pattern the natural-repeat
    # probes turn out to show, so the sweep has to name it.
    text_scores = np.zeros((4, 4))
    text_scores.reshape(-1)[:8] = 0.06
    protein_scores = np.zeros((4, 4))
    protein_scores.reshape(-1)[:4] = 0.25
    text = make_arm("t", "text", text_scores)
    protein = make_arm("p", "protein", protein_scores)
    sweep = threshold_sweep([text, protein])
    by_threshold = {row["threshold"]: row for row in sweep["rows"]}
    assert by_threshold["0.05"]["separation_holds"] is True
    assert by_threshold["0.10"]["separation_holds"] is False
    assert by_threshold["0.10"]["worst_text_over_best_protein"] == pytest.approx(0.0)
    assert sweep["separation_holds_at_every_threshold"] is False
    assert "0.10" in sweep["thresholds_where_separation_breaks"]


def test_sweep_treats_a_zero_protein_arm_as_a_separation_not_a_failure():
    """An undefined ratio and a failed ordering are different states."""

    text = make_arm("t", "text", np.full((4, 4), 0.5))
    protein = make_arm("p", "protein", np.zeros((4, 4)))
    row = next(r for r in threshold_sweep([text, protein])["rows"] if r["threshold"] == "0.10")
    assert row["worst_text_over_best_protein"] is None
    assert row["separation_holds"] is True


# ------------------------------------------------------- collinearity refusal


def test_a_collinear_design_is_refused_rather_than_fitted():
    """The failure this analysis exists to avoid: a clean-looking artefact.

    When every text arm is small and every protein arm is large, the modality
    indicator IS the scale covariate.  The fit must decline to report a modality
    effect instead of emitting the coefficient that arithmetic will happily
    produce.
    """

    rng = np.random.default_rng(7)
    arms = []
    for index, name in enumerate(("t1", "t2", "t3")):
        scores = rng.uniform(0.0, 0.02, size=(6, 6))
        scores.reshape(-1)[:4] = 0.5
        arms.append(make_arm(name, "text", scores, parameters=int(1e8 * (1 + index * 0.01))))
    for index, name in enumerate(("p1", "p2", "p3")):
        scores = rng.uniform(0.0, 0.02, size=(6, 6))
        arms.append(make_arm(name, "protein", scores, parameters=int(1e10 * (1 + index * 0.01))))

    fit = scale_modality_fit(arms, threshold=0.10, covariate="log10_parameters")
    assert fit["collinearity"]["abs_correlation"] > COLLINEARITY_LIMIT
    assert fit["readable"] is False
    assert fit["verdict"] == "cannot_separate"
    assert "cannot attribute" in fit["verdict_reason"]


def test_a_separable_design_is_read_and_declares_itself_non_inferential():
    arms = separated_panel()
    fit = scale_modality_fit(arms, threshold=0.10, covariate="log10_parameters")
    assert fit["readable"] is True
    assert fit["inferential"] is False
    assert fit["verdict"] in {"modality_offset_excludes_zero", "inconclusive"}


def test_collinearity_limit_matches_a_variance_inflation_factor_of_ten():
    report = collinearity_report(
        np.array([0.0, 0.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0, 4.0])
    )
    assert report["vif"] == pytest.approx(1.0 / (1.0 - report["correlation"] ** 2))
    assert COLLINEARITY_LIMIT**2 == pytest.approx(0.9, abs=1e-3)


# ------------------------------------------------------ model-level inference


def test_exact_test_enumerates_every_assignment_and_reports_its_floor():
    arms = separated_panel()
    values = {arm.name: arm.fraction_above(0.10) for arm in arms}
    result = model_level_exact_test(arms, statistic="fraction", values=values)
    assert result["n_assignments"] == 35  # C(7, 3)
    assert result["smallest_attainable_p"] == pytest.approx(1.0 / 35.0)
    assert result["unit"] == "model"
    assert result["p_one_sided"] >= result["smallest_attainable_p"]


def test_complete_separation_lands_exactly_on_the_floor_and_says_so():
    """A p at its own floor must be flagged, not quoted as strong evidence."""

    arms = separated_panel()
    values = {arm.name: 1.0 if arm.modality == "text" else 0.0 for arm in arms}
    result = model_level_exact_test(arms, statistic="synthetic", values=values)
    assert result["complete_separation"] is True
    assert result["at_floor"] is True
    assert result["p_one_sided"] == pytest.approx(1.0 / 35.0)


# ---------------------------------------------------------- probe bootstrap


def test_bootstrap_resamples_probes_and_reproduces_the_point_estimate():
    rng = np.random.default_rng(2)
    per_probe = rng.uniform(0.0, 0.02, size=(16, 6, 6))
    per_probe[:, 0, :3] = 0.5
    result = cluster_bootstrap_fraction(per_probe, threshold=0.10, resamples=500, seed=1)
    assert result["resampling_unit"] == "probe"
    assert result["n_probe_clusters"] == 16
    assert result["point_estimate"] == pytest.approx(3.0 / 36.0)
    assert result["interval"][0] <= result["point_estimate"] <= result["interval"][1]


def test_a_degenerate_interval_is_a_correct_answer_when_heads_are_far_from_the_cut():
    """ProtGPT2's zero-width interval is the design working, not the design failing.

    When every head sits either far above or far below the cut, no reweighting of
    probes moves the count, and the honest interval is a point.  A bootstrap that
    manufactured width here would be inventing uncertainty.
    """

    per_probe = np.full((16, 4, 4), 0.001)
    per_probe[:, 0, 0] = 0.9
    result = cluster_bootstrap_fraction(per_probe, threshold=0.10, resamples=500, seed=1)
    assert result["interval"] == [result["point_estimate"], result["point_estimate"]]
    assert result["bootstrap_sd"] == pytest.approx(0.0)


def test_bootstrap_refuses_a_single_probe():
    with pytest.raises(ValueError, match="at least two probes"):
        cluster_bootstrap_fraction(
            np.zeros((1, 4, 4)), threshold=0.1, resamples=10, seed=0
        )


def test_bootstrap_refuses_a_matrix_without_a_probe_axis():
    with pytest.raises(ValueError, match=r"\(probe, layer, head\)"):
        cluster_bootstrap_fraction(np.zeros((4, 4)), threshold=0.1, resamples=10, seed=0)


# ------------------------------------------------------------- guard rails


def test_pairwise_auc_needs_both_modalities():
    arms = [a for a in separated_panel() if a.modality == "text"]
    with pytest.raises(ValueError, match="at least one arm of each modality"):
        pairwise_auc(arms)


def test_census_refuses_a_declared_shape_that_contradicts_the_matrix():
    scores = np.zeros((4, 4))
    arm = make_arm("x", "text", scores)
    with pytest.raises(ValueError, match="does not match the declared shape"):
        replace(arm, n_layer=5)


def test_census_refuses_a_non_finite_score():
    scores = np.zeros((4, 4))
    scores[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        make_arm("x", "text", scores)
