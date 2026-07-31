"""The path-patching statistics EXP-R2-071 published, and the guards around them.

Every statistic this file exercises was quoted in the audit before any of it
existed in the repository: the Gini of the causal-effect grid, the share of the
grid carrying half that effect, the census-to-causal rank correlation split at a
top-k cut, and the per-head reliability that ruled out measurement noise as the
explanation for a low correlation.  ``grep`` found none of them in ``src``,
``scripts`` or ``tests``.  They were computed once, in a script that is gone, and
one of the four numbers does not reproduce.

The tests are therefore of two kinds.  The first pins the *invariants* the
statistics have to satisfy -- head-count freedom above all, which is why these
two concentration statistics were chosen over a top-k share (Appendix B rule
21).  The second recomputes the published numbers from the shipped artefacts
through the new code path, so that the artefact and the document can no longer
drift apart unnoticed.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.arms import Arm, PANEL  # noqa: E402
from src.transfer.path_patching import (  # noqa: E402
    EXHAUSTIVE_CRITERION,
    PathCase,
    PathPatcher,
    RESAMPLED_POPULATION,
    SenderHead,
    bootstrap_difference,
    effect_concentration,
    gini,
    head_effect_reliability,
    sender_effects,
    sender_set_overlap,
    share_of_grid_carrying_half_effect,
)

D2BPRIME = REPO_ROOT / "results/transfer_20260730/d2bprime"


# ------------------------------------------------- concentration, head-count free


def _row(effect: float) -> dict[str, object]:
    return {"effects": {"total": effect}}


def test_gini_and_share_are_free_of_the_head_count():
    """Rule 21's requirement, and the reason a top-k share was rejected for these.

    Replicating every head leaves a grid with the same shape and twice the heads.
    A count-based statistic moves; these must not, because head counts on this
    panel run from 144 to 720 and the count is the disputed quantity.
    """

    values = [0.5, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005]
    doubled = values * 2
    assert gini(doubled) == pytest.approx(gini(values))
    assert share_of_grid_carrying_half_effect(doubled) == pytest.approx(
        share_of_grid_carrying_half_effect(values)
    )
    # And free of a common positive rescaling, so the two are concentration
    # statistics rather than magnitude statistics in disguise.
    scaled = [value * 37.0 for value in values]
    assert gini(scaled) == pytest.approx(gini(values))
    assert share_of_grid_carrying_half_effect(scaled) == pytest.approx(
        share_of_grid_carrying_half_effect(values)
    )


def test_gini_spans_the_even_grid_and_the_single_head():
    even = [0.3] * 10
    assert gini(even) == pytest.approx(0.0)
    assert share_of_grid_carrying_half_effect(even) == pytest.approx(0.5)

    one_head = [1.0] + [0.0] * 9
    assert gini(one_head) == pytest.approx(0.9)
    assert share_of_grid_carrying_half_effect(one_head) == pytest.approx(0.1)


def test_concentration_reads_the_magnitude_and_counts_the_sign_changes():
    rows = [_row(0.5), _row(-0.5), _row(0.0), _row(0.0)]
    report = effect_concentration(rows)
    assert report["n_heads"] == 4
    assert report["n_heads_negative_effect"] == 1
    assert report["total_absolute_effect"] == pytest.approx(1.0)
    # Two heads of opposite sign concentrate exactly as two of the same sign do.
    assert report["gini"] == pytest.approx(gini([0.5, 0.5, 0.0, 0.0]))


def test_a_grid_with_no_effect_at_all_refuses_rather_than_returning_zero():
    """0.0 is what a perfectly even grid returns; a dead grid must not borrow it."""

    with pytest.raises(ValueError, match="no distribution to concentrate"):
        gini([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="half of nothing"):
        share_of_grid_carrying_half_effect([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="fewer than two"):
        gini([0.4])


# ------------------------------------------------------------------- reliability


#: "Not supplied", which is not the same as a supplied ``None``: a head that
#: contributed no eligible case carries ``mean_probe_clustered`` explicitly absent,
#: and a fixture that cannot express that cannot test the withholding path.
_SAME_AS_EFFECT = object()


def _reliability_row(
    effect: float,
    sem: float | None,
    clustered: float | None,
    *,
    clustered_effect: float | None | object = _SAME_AS_EFFECT,
):
    """One head's record, carrying **both** centres the module publishes.

    The fixture used to omit ``mean_probe_clustered``, which is why nothing caught
    the reliability function reading the case-weighted centre under both units.
    ``clustered_effect`` defaults to ``effect`` so that a test which does not care
    about the distinction reads unchanged, and the tests that do care set it.
    """

    return {
        "recovery": {
            "total": {
                "mean": effect,
                "mean_probe_clustered": (
                    effect if clustered_effect is _SAME_AS_EFFECT else clustered_effect
                ),
                "sem": sem,
                "sem_probe_clustered": clustered,
            }
        }
    }


def test_reliability_is_published_on_both_sampling_units_and_labelled():
    """The unit is the defect, not the formula.

    This module declares the probe record as the sampling unit and publishes both
    standard errors.  EXP-R2-071's range came from the case-level one, which
    treats several nested prefixes of one protein as independent observations.
    Both are returned so the difference is visible in the artefact.
    """

    rows = [
        _reliability_row(0.40, 0.02, 0.10),
        _reliability_row(0.20, 0.02, 0.10),
        _reliability_row(0.10, 0.02, 0.10),
        _reliability_row(0.05, 0.02, 0.10),
    ]
    report = head_effect_reliability(rows)
    assert report["primary_sem_cluster_unit"] == "probe"
    probe = report["by_sem_cluster_unit"]["probe"]
    case = report["by_sem_cluster_unit"]["case"]
    assert probe["sem_key"] == "sem_probe_clustered"
    assert case["sem_key"] == "sem"

    observed = float(np.var([0.40, 0.20, 0.10, 0.05], ddof=1))
    assert probe["reliability_signed_effect"] == pytest.approx(
        (observed - 0.10**2) / observed
    )
    assert case["reliability_signed_effect"] == pytest.approx(
        (observed - 0.02**2) / observed
    )
    # The clustered error is the larger one here, so it must give the lower
    # reliability. A run that quotes the case figure overstates the ranking.
    assert probe["reliability_signed_effect"] < case["reliability_signed_effect"]


def test_reliability_is_withheld_when_a_head_carries_no_standard_error():
    """Not "average over the heads that have one": that is a different population.

    A head whose eligible cases all came from one probe has no probe-clustered
    standard error at all, and an error variance averaged over the rest divided by
    an observed variance taken over every head is not a reliability.
    """

    rows = [
        _reliability_row(0.40, 0.02, 0.10),
        _reliability_row(0.20, 0.02, None),
        _reliability_row(0.10, 0.02, 0.10),
    ]
    report = head_effect_reliability(rows)
    probe = report["by_sem_cluster_unit"]["probe"]
    assert probe["reliability_signed_effect"] is None
    assert probe["n_heads_without_a_standard_error"] == 1
    assert "different head population" in probe["withheld_reason"]
    # The case-level unit is unaffected: its standard errors are all present.
    assert report["by_sem_cluster_unit"]["case"]["reliability_signed_effect"] is not None


def test_each_unit_takes_its_observed_variance_over_its_own_centre():
    """The standard error and the mean it describes travel together, or neither does.

    ``sender_recoveries`` says at the point it builds the record that ``mean`` and
    ``sem_probe_clustered`` are not a point estimate and its standard error:
    ``mean`` weights each probe by its case count, the clustered standard error
    describes ``mean_probe_clustered``, which weights probes equally.  This
    function read the case-weighted centre under both units, so the probe-unit
    reliability divided a probe-unit error variance by a case-unit observed
    variance.  On ZymCTRL's approximate-repeat grid that read 0.008 against a
    paired 0.170 (EXP-R2-078).

    The two centres are given a deliberately different spread here: the clustered
    centres vary four times as widely, so a function that ignored ``mean_key``
    would return the *same* observed variance for both units and fail.
    """

    rows = [
        _reliability_row(0.40, 0.02, 0.10, clustered_effect=1.60),
        _reliability_row(0.20, 0.02, 0.10, clustered_effect=0.80),
        _reliability_row(0.10, 0.02, 0.10, clustered_effect=0.40),
        _reliability_row(0.05, 0.02, 0.10, clustered_effect=0.20),
    ]
    report = head_effect_reliability(rows)
    probe = report["by_sem_cluster_unit"]["probe"]
    case = report["by_sem_cluster_unit"]["case"]

    assert probe["mean_key"] == "mean_probe_clustered"
    assert case["mean_key"] == "mean"

    observed_probe = float(np.var([1.60, 0.80, 0.40, 0.20], ddof=1))
    observed_case = float(np.var([0.40, 0.20, 0.10, 0.05], ddof=1))
    assert probe["observed_variance_signed_effect"] == pytest.approx(observed_probe)
    assert case["observed_variance_signed_effect"] == pytest.approx(observed_case)
    assert probe["reliability_signed_effect"] == pytest.approx(
        (observed_probe - 0.10**2) / observed_probe
    )
    assert case["reliability_signed_effect"] == pytest.approx(
        (observed_case - 0.02**2) / observed_case
    )


def test_a_head_missing_either_half_of_the_pair_withholds_the_unit():
    """A centre is as load-bearing as a standard error, and may be absent alone.

    ``_probe_clustered_sem`` returns ``mean_probe_clustered`` as ``None`` when a
    head contributed no eligible case at all, while the case-level pair may still
    be present.  Withholding on the standard error alone would then take the
    observed variance over every head and the error variance over a subset.
    """

    rows = [
        _reliability_row(0.40, 0.02, 0.10),
        _reliability_row(0.20, 0.02, 0.10, clustered_effect=None),
        _reliability_row(0.10, 0.02, 0.10),
    ]
    report = head_effect_reliability(rows)
    probe = report["by_sem_cluster_unit"]["probe"]
    assert probe["reliability_signed_effect"] is None
    assert probe["n_heads_without_a_standard_error"] == 1
    assert "mean_probe_clustered" in probe["withheld_reason"]
    assert report["by_sem_cluster_unit"]["case"]["reliability_signed_effect"] is not None


def test_reliability_refuses_a_pathway_that_has_no_standard_error():
    with pytest.raises(ValueError, match="unknown pathway"):
        head_effect_reliability([_reliability_row(0.1, 0.01, 0.01)] * 2, pathway="mediated")


# ---------------------------------------------- the two effect scales, per head


def _recoveries(direct: float, via_mlp: float, via_attn: float, total: float, factor: float):
    return {
        "direct": {"mean": direct, "mean_logits": direct * factor},
        "via_mlp": {"mean": via_mlp, "mean_logits": via_mlp * factor},
        "via_attn": {"mean": via_attn, "mean_logits": via_attn * factor},
        "total": {"mean": total, "mean_logits": total * factor},
    }


def test_the_decomposition_holds_on_both_scales_and_the_scale_is_declared():
    recoveries = _recoveries(0.04, 0.09, 0.06, 0.10, factor=20.68)
    recovery_scale = sender_effects(recoveries)
    logit_scale = sender_effects(recoveries, key="mean_logits")

    assert recovery_scale["mediated"] == pytest.approx(0.06)
    assert logit_scale["mediated"] == pytest.approx(0.06 * 20.68)
    for key, value in recovery_scale.items():
        assert logit_scale[key] == pytest.approx(value * 20.68)
    with pytest.raises(ValueError, match="unknown effect scale"):
        sender_effects(recoveries, key="mean_probe_clustered")


def test_a_recovery_magnitude_ordering_can_reverse_on_the_logit_scale():
    """The confound EXP-R2-071 published a sentence on, in one assertion.

    Two arms, two heads. The head with the larger recovery is the head with the
    smaller write to the logits, because the two arms divide by denominators that
    differ by a factor of 27.
    """

    zymctrl = sender_effects(_recoveries(0.10, 0.20, 0.15, 0.3049, factor=0.7626))
    gpt2_large = sender_effects(_recoveries(0.005, 0.01, 0.008, 0.0194, factor=24.096))

    assert zymctrl["total"] > gpt2_large["total"]
    assert (
        sender_effects(_recoveries(0.10, 0.20, 0.15, 0.3049, factor=0.7626), key="mean")
        is not None
    )
    zym_logits = sender_effects(
        _recoveries(0.10, 0.20, 0.15, 0.3049, factor=0.7626), key="mean_logits"
    )
    large_logits = sender_effects(
        _recoveries(0.005, 0.01, 0.008, 0.0194, factor=24.096), key="mean_logits"
    )
    assert zym_logits["total"] < large_logits["total"]


# ------------------------------------------------ the guards around the statistics


def test_a_full_grid_compared_against_a_full_grid_is_not_a_stability_check():
    """All six shipped arms published jaccard 1.0 under "stability is shown"."""

    heads = [
        SenderHead(layer=0, head=0, prefix_matching=0.5, above_threshold=True),
        SenderHead(layer=0, head=1, prefix_matching=0.01, above_threshold=False),
    ]
    trivial = sender_set_overlap(
        heads,
        heads,
        left_criterion=EXHAUSTIVE_CRITERION,
        right_criterion=EXHAUSTIVE_CRITERION,
    )
    assert trivial["jaccard"] == pytest.approx(1.0)
    assert trivial["comparison_is_trivial"] is True
    assert "no value other than 1.0 is reachable" in trivial["stability_verdict"]

    real = sender_set_overlap(
        heads,
        heads[:1],
        left_criterion="prefix_matching_above_threshold",
        right_criterion="prefix_matching_above_threshold",
    )
    assert real["comparison_is_trivial"] is False
    assert "carries information" in real["stability_verdict"]


def test_the_bootstrap_caveat_is_a_function_of_the_criterion_that_selected_the_heads():
    left = [0.4, 0.5, 0.6, 0.55, 0.45, 0.52, 0.58, 0.47]
    right = [0.1, 0.2, 0.15, 0.05, 0.12, 0.18, 0.08, 0.11]
    exhaustive = bootstrap_difference(
        left,
        right,
        resamples=200,
        seed=1,
        left_criterion=EXHAUSTIVE_CRITERION,
        right_criterion=EXHAUSTIVE_CRITERION,
    )
    selective = bootstrap_difference(
        left,
        right,
        resamples=200,
        seed=1,
        left_criterion="prefix_matching_above_threshold",
        right_criterion="prefix_matching_above_threshold",
    )
    assert (
        "entire attention-head grid"
        in exhaustive["resampled_population"]["left"]["what_is_resampled"]
    )
    assert (
        "prefix-matching threshold"
        in selective["resampled_population"]["left"]["what_is_resampled"]
    )
    assert "threshold" not in exhaustive["resampled_population"]["left"][
        "what_is_resampled"
    ].replace("threshold-selected subset", "")

    # The two sides need not agree: an arm with no head above threshold enters on
    # the top-k fallback while its comparator does not.
    mixed = bootstrap_difference(
        left,
        right,
        resamples=200,
        seed=1,
        left_criterion="top_k_no_head_above_threshold",
        right_criterion="prefix_matching_above_threshold",
    )
    assert mixed["resampled_population"]["left"]["sender_criterion"] != (
        mixed["resampled_population"]["right"]["sender_criterion"]
    )
    assert set(RESAMPLED_POPULATION) == {
        "prefix_matching_above_threshold",
        "top_k_no_head_above_threshold",
        EXHAUSTIVE_CRITERION,
    }


def test_the_bootstrap_refuses_a_criterion_whose_population_is_undeclared():
    with pytest.raises(ValueError, match="no declared population"):
        bootstrap_difference(
            [0.1] * 8,
            [0.2] * 8,
            resamples=200,
            seed=1,
            left_criterion="whatever_the_caller_felt_like",
            right_criterion="prefix_matching_above_threshold",
        )


# ------------------------------------------------- a standard error that is absent


def _tiny_gpt2() -> Arm:
    """A real two-layer GPT-2 on the CPU.

    The property under test lives inside ``sender_recoveries``, after a forward
    pass, so a stub that never runs one cannot reach it. Eight dimensions, random
    weights, milliseconds.
    """

    config = GPT2Config(
        n_layer=2, n_head=2, n_embd=8, vocab_size=16, n_positions=32,
        attn_implementation="eager",
    )
    torch.manual_seed(3)

    class _PadTokenizer:
        pad_token_id = 0

    return Arm(
        spec=replace(PANEL["gpt2-large"], name="tiny-gpt2", n_layer=2, d_model=8),
        model=GPT2LMHeadModel(config).eval(),
        tokenizer=_PadTokenizer(),
        device="cpu",
        dtype="float32",
        attn_implementation="eager",
    )


def _case(tokens: list[int], k: int, corrupt_token: int, probe: int) -> PathCase:
    corrupted = list(tokens)
    corrupted[k] = corrupt_token
    return PathCase(
        tuple(tokens), tuple(corrupted), len(tokens) - 1, k, tokens[k], corrupt_token,
        probe, "exact",
    )


def test_one_eligible_case_publishes_no_standard_error_rather_than_zero():
    """A fabricated zero reads as perfect precision -- the sibling's own words.

    ``_probe_clustered_sem`` returns ``None`` when one probe contributed and says
    why in its docstring. Nine lines away the case-level branch returned 0.0 in
    the same situation, and a downstream reliability built on it would read the
    head as measured without error.
    """

    arm = _tiny_gpt2()
    cases = [
        _case([3, 5, 7, 9, 4, 5, 8, 6, 2, 5], 2, 11, 0),
        _case([4, 6, 7, 2, 9, 6, 3, 5, 1, 6], 3, 13, 0),
        _case([2, 8, 5, 7, 3, 6, 9, 4, 1, 8], 4, 15, 1),
        _case([5, 9, 4, 6, 8, 2, 7, 3, 5, 9], 5, 12, 1),
    ]
    survey = PathPatcher(arm, cases, batch_size=4, minimum_effect=1e-4)
    eligible = [
        index
        for index, keep in enumerate(survey.batches[0].eligible.tolist())
        if keep
    ]
    assert eligible, "the tiny arm produced no eligible case to build the single-case run on"

    patcher = PathPatcher(arm, [cases[eligible[0]]], batch_size=1, minimum_effect=1e-4)
    sender = SenderHead(layer=0, head=1, prefix_matching=0.5, above_threshold=True)
    recoveries = patcher.sender_recoveries(sender)
    for pathway, record in recoveries.items():
        assert record["n"] == 1, pathway
        assert record["sem"] is None, pathway
        assert record["sem_probe_clustered"] is None, pathway
        # The point estimate survives on both scales; only the precision claim goes.
        assert isinstance(record["mean"], float)
        assert isinstance(record["mean_logits"], float)


def test_the_two_scales_are_not_a_rescaling_of_each_other_by_one_number():
    """Each case divides by its own denominator, so the map is not a constant.

    If it were, the logit scale would carry no information the recovery scale does
    not and there would be nothing to publish.
    """

    arm = _tiny_gpt2()
    cases = [
        _case([3, 5, 7, 9, 4, 5, 8, 6, 2, 5], 2, 11, 0),
        _case([4, 6, 7, 2, 9, 6, 3, 5, 1, 6], 3, 13, 0),
        _case([2, 8, 5, 7, 3, 6, 9, 4, 1, 8], 4, 15, 1),
        _case([5, 9, 4, 6, 8, 2, 7, 3, 5, 9], 5, 12, 1),
    ]
    patcher = PathPatcher(arm, cases, batch_size=4, minimum_effect=1e-4)
    sender = SenderHead(layer=0, head=1, prefix_matching=0.5, above_threshold=True)
    recoveries = patcher.sender_recoveries(sender)
    ratios = [
        record["mean_logits"] / record["mean"]
        for record in recoveries.values()
        if abs(record["mean"]) > 1e-9
    ]
    assert len(ratios) >= 2
    assert max(ratios) - min(ratios) > 1e-9


# --------------------------------------------- the shipped artefacts, recomputed


@pytest.mark.skipif(
    not (D2BPRIME / "gpt2.json").is_file(),
    reason="the EXP-R2-071 artefacts are host-local and are not in the repository",
)
def test_the_published_concentration_numbers_reproduce_from_the_artefact():
    """EXP-R2-071 quoted Gini 0.829 and 4.9% of the grid for gpt2. Both reproduce.

    They are pinned here because they were computed by a script that no longer
    exists: a number in the audit and a number in an artefact now come from one
    implementation, and this test fails if they part company.
    """

    payload = json.loads((D2BPRIME / "gpt2.json").read_text(encoding="utf-8"))
    rows = payload["conditions"]["senders_exact__cases_exact"]["per_sender_head"]
    report = effect_concentration(rows)

    assert report["n_heads"] == 144
    assert report["gini"] == pytest.approx(0.829, abs=5e-4)
    assert report["share_of_grid_carrying_half_effect"] == pytest.approx(0.049, abs=5e-4)
    assert report["n_heads_carrying_half_effect"] == 7


@pytest.mark.skipif(
    not (D2BPRIME / "zymctrl.json").is_file(),
    reason="the EXP-R2-071 artefacts are host-local and are not in the repository",
)
def test_the_published_reliability_range_is_the_case_level_reading():
    """0.916 to 0.991 is the case-level magnitude reading, and the floor moves.

    ZymCTRL sets the published floor. On the probe-clustered pair this module
    calls correct it falls to 0.82 on the exact-repeat case set and to 0.17 on the
    approximate one. The test pins the gap rather than either number, because the
    gap is what the audit has to record.

    **The 0.008 that reached the audit document is not on this axis at all.** It
    came from dividing a probe-clustered error variance by an observed variance
    taken over the case-weighted centre -- two different estimators (EXP-R2-078).
    Both mismatched figures are pinned below so that a regression to them fails
    loudly rather than quietly restoring a retracted number.
    """

    payload = json.loads((D2BPRIME / "zymctrl.json").read_text(encoding="utf-8"))
    conditions = payload["conditions"]

    exact = head_effect_reliability(
        conditions["senders_exact__cases_exact"]["per_sender_head"]
    )["by_sem_cluster_unit"]
    assert exact["case"]["reliability_magnitude_ranking"] == pytest.approx(0.916, abs=5e-4)
    assert exact["probe"]["reliability_magnitude_ranking"] == pytest.approx(0.818, abs=5e-3)

    approximate_rows = conditions["senders_exact__cases_approximate"]["per_sender_head"]
    approximate = head_effect_reliability(approximate_rows)["by_sem_cluster_unit"]
    assert approximate["case"]["reliability_magnitude_ranking"] == pytest.approx(
        0.657, abs=5e-3
    )
    assert approximate["probe"]["reliability_signed_effect"] == pytest.approx(
        0.443, abs=5e-3
    )
    assert approximate["probe"]["reliability_magnitude_ranking"] == pytest.approx(
        0.170, abs=5e-3
    )
    # Still far below the case-level reading: the correction raises the floor by
    # 20x and does not lift this grid into rankable territory.
    assert (
        approximate["probe"]["reliability_magnitude_ranking"]
        < approximate["case"]["reliability_magnitude_ranking"]
    )

    # The retracted pairing, reconstructed here so its numbers are on record as
    # what the defect produced rather than as what the artefact says.
    mismatched = np.array(
        [float(row["recovery"]["total"]["mean"]) for row in approximate_rows]
    )
    clustered_sem = np.array(
        [float(row["recovery"]["total"]["sem_probe_clustered"]) for row in approximate_rows]
    )
    error_variance = float(np.mean(clustered_sem**2))
    for scale, values in (("signed", mismatched), ("magnitude", np.abs(mismatched))):
        observed = float(np.var(values, ddof=1))
        retracted = (observed - error_variance) / observed
        assert retracted == pytest.approx(0.189 if scale == "signed" else 0.008, abs=5e-3)


# --------------------------------------------------- the stage's contract record


def test_the_panel_scoped_stage_writes_its_contract_record():
    """A one-arm run and a full-panel run were distinguishable only by counting rows.

    ``stage_contract_record`` is the declaration L18 earned; stages 01, 02, 03 and
    08 call it and this one wrote ``panel_summary.json`` without it. The precedent
    for asserting it in source is ``test_transfer_gap_contract.py``.
    """

    source = (
        REPO_ROOT / "scripts/transfer/11_induction_path_patching.py"
    ).read_text(encoding="utf-8")
    assert 'stage_contract_record(\n            "induction_path_patching"' in source
    assert "from panel_contract import stage_contract_record" in source
