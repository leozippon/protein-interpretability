"""The corpus-disjoint referent estimand: its grammar, its baselines, its rule.

Three failures here would produce numbers that look entirely normal, and each has
a test that targets it rather than the happy path.

*A variant that is not a substitution.* ``dataset2`` spells insertions,
deletions, wild-type rows and no-op tokens (``T38T``) in the same column as real
substitutions. Admitting one silently changes what the estimand is measured over,
and a no-op token would enter the free baselines as a substitution of a residue
for itself.

*The superseded k-mer background.* The corrected and the superseded counts are
both on disk. As a distribution they differ by 0.00028 total variation, so a
fragment baseline built on the wrong one reads plausibly -- while one 4-mer's
frequency is off by 42%. The refusal is by total, not by path.

*The pre-registered rule read as a disjunction.* A positive needs the arm to beat
**every** baseline, and a control failure must dominate whatever the design side
did. Both are asserted directly, including the case where the design side passes
and the control does not, which is the one that would otherwise be reported as a
discovery.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from src.transfer import designed_referent as D
from src.transfer.kmer_background import ALPHABET, KmerBackground


# ------------------------------------------------------------ variant grammar


@pytest.mark.parametrize(
    "mut_type",
    ["wt", "ins12A", "del12", "T38T", "E16E:A43A", "D28A:T32T", "", "12A", "A12"],
)
def test_ineligible_variants_are_refused(mut_type):
    assert D.eligible_substitutions(mut_type) is None


def test_eligible_variants_parse_to_the_shared_grammar():
    assert D.eligible_substitutions("A12G") == (("A", 12, "G"),)
    assert D.eligible_substitutions("A12G:C40W") == (("A", 12, "G"), ("C", 40, "W"))


def test_apply_substitutions_reconstructs_and_checks_the_wild_type():
    wild = "ACDEFGHIK"
    assert D.apply_substitutions(wild, (("C", 2, "W"), ("K", 9, "A"))) == "AWDEFGHIA"
    with pytest.raises(ValueError, match="mutation string says"):
        D.apply_substitutions(wild, (("W", 2, "A"),))
    with pytest.raises(ValueError, match="outside a wild type"):
        D.apply_substitutions(wild, (("A", 40, "G"),))


# ---------------------------------------------------------- the background gate


def _background(totals: dict[int, int], *, zero_a_kmer: bool = False) -> KmerBackground:
    counts = {}
    for k, total in totals.items():
        width = len(ALPHABET) ** k
        vector = np.full(width, total // width, dtype=np.int64)
        vector[0] += total - int(vector.sum())
        if zero_a_kmer:
            vector[1] += vector[2]
            vector[2] = 0
        counts[k] = vector
    return KmerBackground(
        counts=counts,
        residues=17_277_105_157,
        records=60_315_044,
        source=Path("uniref50.fasta"),
        source_bytes=1,
        wall_seconds=1.0,
    )


def test_the_corrected_background_is_accepted():
    record = D.require_corrected_background(_background(dict(D.CORRECTED_KMER_TOTALS)))
    assert record["totals"] == dict(D.CORRECTED_KMER_TOTALS)


def test_the_superseded_background_is_refused_by_its_total():
    with pytest.raises(ValueError, match="superseded line-local"):
        D.require_corrected_background(_background(dict(D.SUPERSEDED_KMER_TOTALS)))


def test_a_background_with_an_unobserved_kmer_is_refused():
    with pytest.raises(ValueError, match="observes"):
        D.require_corrected_background(
            _background(dict(D.CORRECTED_KMER_TOTALS), zero_a_kmer=True)
        )


# ------------------------------------------------------- the fragment baseline


def test_the_conditional_normalises_over_every_context():
    table = D.conditional_log_probabilities(
        _background(dict(D.CORRECTED_KMER_TOTALS)), 3
    )
    rows = np.exp(table).reshape(len(ALPHABET) ** 2, len(ALPHABET))
    assert np.allclose(rows.sum(axis=1), 1.0)


def test_a_uniform_background_scores_every_sequence_of_one_length_alike():
    table = D.conditional_log_probabilities(
        _background({3: len(ALPHABET) ** 3 * 1000}), 3
    )
    sequences = ["ACDEFGHIKL", "LKIHGFEDCA", "AAAAAAAAAA"]
    scores = D.fragment_log_likelihood(sequences, table, 3)
    assert np.allclose(scores, 8 * np.log(1.0 / len(ALPHABET)))


def test_the_fragment_score_is_the_sum_of_its_own_conditionals():
    background = _background(dict(D.CORRECTED_KMER_TOTALS))
    # Make one context strongly prefer one residue, so a hand-summed expectation
    # is not the uniform answer by accident.
    counts = background.counts[3]
    width = len(ALPHABET)
    context = ALPHABET.index("A") * width + ALPHABET.index("C")
    counts[context * width + ALPHABET.index("D")] += 10**9
    table = D.conditional_log_probabilities(background, 3)
    sequence = "ACDACD"
    expected = sum(
        table[
            ALPHABET.index(sequence[i]) * width**2
            + ALPHABET.index(sequence[i + 1]) * width
            + ALPHABET.index(sequence[i + 2])
        ]
        for i in range(len(sequence) - 2)
    )
    assert D.fragment_log_likelihood([sequence], table, 3)[0] == pytest.approx(expected)


def test_encoding_refuses_ragged_and_non_canonical_input():
    with pytest.raises(ValueError, match="same length"):
        D.encode_sequences(["ACDE", "ACD"])
    with pytest.raises(ValueError, match="canonical alphabet"):
        D.encode_sequences(["ACDX"])


# ------------------------------------------------------------------ statistics


def test_spearman_returns_none_rather_than_a_nan_on_a_constant_channel():
    assert D.spearman([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]) is None
    assert D.spearman([1.0, 2.0], [1.0, 2.0]) is None
    value = D.spearman([1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0])
    assert value == pytest.approx(
        float(stats.spearmanr([1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0]).statistic)
    )


def test_the_unit_bootstrap_refuses_below_the_package_floor():
    record = D.unit_bootstrap([0.1] * 4, list("abcd"), resamples=100, seed=1)
    assert record["degenerate"] is True
    assert record["interval"] is None
    record = D.unit_bootstrap(
        [0.1] * 16, [f"u{i}" for i in range(16)], resamples=200, seed=1
    )
    assert record["degenerate"] is False
    assert record["interval"] is not None
    assert record["n_units"] == 16


def test_a_contrast_reports_the_wild_types_only_one_channel_could_be_read_on():
    units = {f"w{i}": f"u{i % 8}" for i in range(16)}
    model = {f"w{i}": 0.5 for i in range(16)}
    baseline = {f"w{i}": 0.1 for i in range(15)}
    record = D.channel_comparison(model, baseline, units, resamples=200, seed=3)
    assert record["n_wildtypes"] == 15
    assert record["n_wildtypes_dropped"] == 1
    assert record["beats_baseline"] is True


# ------------------------------------------------------------ the frozen rule


def _gates(**overrides: bool) -> dict[str, bool]:
    gates = {name: True for name in D.BASELINES}
    gates.update(overrides)
    return gates


def test_a_positive_needs_every_baseline_on_both_sides():
    assert D.arm_verdict(_gates(), _gates())["verdict"] == "positive"


def test_one_failed_baseline_on_the_designs_is_a_clean_negative():
    verdict = D.arm_verdict(_gates(fragment_markov_k4=False), _gates())
    assert verdict["verdict"] == "negative"
    assert verdict["design_failures"] == ["fragment_markov_k4"]


def test_a_failing_control_dominates_a_passing_design_side():
    verdict = D.arm_verdict(_gates(), _gates(blosum62=False))
    assert verdict["verdict"] == "uninterpretable_instrument_bound"
    assert verdict["beats_every_baseline_on_designs"] is True
    assert verdict["control_failures"] == ["blosum62"]


def test_the_verdict_refuses_a_partial_baseline_set():
    partial = {name: True for name in D.BASELINES[:-1]}
    with pytest.raises(ValueError, match="needs every baseline"):
        D.arm_verdict(partial, _gates())


# ------------------------------------------------------------------- the cohort


def _wildtype(name: str, kind: str, series: str, cluster: str, **overrides) -> D.WildType:
    fields = {
        "name": name,
        "kind": kind,
        "series": series,
        "cluster": cluster,
        "zero_hit": kind == "design",
        "series_zero_hit": kind == "design",
        "sequence": "ACDEFGHIKLMNPQRSTVWY",
        "mutants": ("A1C", "C2D", "D3E", "E4F"),
        "phenotype": np.array([0.5, -0.5, 1.5, -1.5]),
        "replicates": (1, 1, 2, 1),
    }
    fields.update(overrides)
    return D.WildType(**fields)


def test_a_wild_types_unit_is_its_series_or_its_cluster():
    assert _wildtype("d", "design", "HHH/rd1", "HHH").unit == "design:HHH/rd1"
    assert _wildtype("n", "natural", "-", "15").unit == "natural:15"


def test_the_cohort_round_trips_through_its_artefact(tmp_path):
    referent = D.Referent(
        wildtypes=(
            _wildtype("d0", "design", "HHH/rd1", "HHH"),
            _wildtype("n0", "natural", "-", "15"),
        ),
        provenance={"sampling": {"mode": "census"}},
    )
    path = tmp_path / "cohort.json"
    D.save_referent(referent, path)
    reloaded = D.load_referent(path)
    assert [wt.record() for wt in reloaded.wildtypes] == [
        wt.record() for wt in referent.wildtypes
    ]
    assert reloaded.provenance == referent.provenance


def test_a_cohort_artefact_of_the_wrong_schema_is_refused(tmp_path):
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps({"schema_version": "other", "wildtypes": []}))
    with pytest.raises(ValueError, match="not r2_designed_referent_v1"):
        D.load_referent(path)


def test_the_design_side_excludes_the_designs_that_hit_the_corpus():
    referent = D.Referent(
        wildtypes=(
            _wildtype("clean", "design", "HHH/rd1", "HHH"),
            _wildtype(
                "hit", "design", "EEHEE/rd3", "EEHEE", zero_hit=False, series_zero_hit=False
            ),
            _wildtype("nat", "natural", "-", "15"),
        ),
        provenance={},
    )
    assert [wt.name for wt in referent.side("design")] == ["clean"]
    assert [wt.name for wt in referent.side("design", zero_hit_only=False)] == [
        "clean",
        "hit",
    ]
    assert [wt.name for wt in referent.side("natural")] == ["nat"]


def test_cohort_counts_track_the_variant_floor():
    wildtypes = (
        _wildtype("d0", "design", "HHH/rd1", "HHH"),
        _wildtype("d1", "design", "HHH/rd2", "HHH", mutants=("A1C",), phenotype=np.array([0.1]), replicates=(1,)),
        _wildtype("n0", "natural", "-", "15"),
    )
    assert D.cohort_counts(wildtypes)["design_series_scored"] == 2
    counts = D.cohort_counts(wildtypes, min_variants=4)
    assert counts["design_series_scored"] == 1
    assert counts["designs_zero_hit_scored"] == 1
    assert counts["naturals_scored"] == 1


def test_the_post_hoc_length_bands_are_taken_from_the_design_lengths():
    designs = [
        _wildtype(f"d{index}", "design", "s", "c", sequence="A" * length)
        for index, length in enumerate([40, 42, 44, 50, 60])
    ]
    assert D.design_length_bands(designs) == ((40, 44), (40, 50), (40, 60))
    with pytest.raises(ValueError, match="no designs"):
        D.design_length_bands([])


def test_a_wild_type_whose_arrays_disagree_is_refused():
    with pytest.raises(ValueError, match="disagree in length"):
        _wildtype("d", "design", "s", "c", phenotype=np.array([0.1]))
    with pytest.raises(ValueError, match="non-finite phenotype"):
        _wildtype("d", "design", "s", "c", phenotype=np.array([0.1, np.nan, 0.2, 0.3]))


# ------------------------------------------------------------------ baselines


def test_every_declared_baseline_is_actually_computed():
    background = _background(dict(D.CORRECTED_KMER_TOTALS))
    residues = np.full(20, 1.0 / 20.0)
    scores = D.baseline_scores(
        _wildtype("d", "design", "HHH/rd1", "HHH"),
        residue_background=residues,
        log_conditional={k: D.conditional_log_probabilities(background, k) for k in (3, 4)},
    )
    assert list(scores) == list(D.BASELINES)
    assert all(value.shape == (4,) for value in scores.values())
    # BLOSUM62 of A->C, C->D, D->E, E->F, read off the matrix rather than restated.
    assert scores["blosum62"].tolist() == [0.0, -3.0, 2.0, -3.0]


def test_the_fragment_baselines_see_the_mutation():
    background = _background(dict(D.CORRECTED_KMER_TOTALS))
    counts = background.counts[3]
    width = len(ALPHABET)
    counts[
        (ALPHABET.index("A") * width + ALPHABET.index("C")) * width + ALPHABET.index("D")
    ] += 10**9
    tables = {k: D.conditional_log_probabilities(background, k) for k in (3, 4)}
    scores = D.baseline_scores(
        _wildtype("d", "design", "HHH/rd1", "HHH"),
        residue_background=np.full(20, 1.0 / 20.0),
        log_conditional=tables,
    )
    # Only E4F leaves the enriched ACD window intact; A1C, C2D and D3E all
    # destroy it, so the fragment channel has to separate the fourth variant from
    # the other three rather than scoring on composition alone.
    values = scores["fragment_markov_k3"]
    assert values[3] > values[:3].max()


# ------------------------------------------------- the design/natural interaction
#
# The three ways this statistic goes wrong silently. It can be built from two
# separately resampled means, which gives an interval that is not an interval on
# the difference. It can drift away from the unit definition the unadjusted
# estimator uses once covariates are added, so the two answer different questions
# while looking like a sensitivity. And its four-way reading can be written so
# that underpower reports as refutation, which is the one failure this entry
# exists to avoid.


def test_every_scored_design_series_has_a_declared_family():
    for group, family in D.DESIGN_FAMILY_OF_GROUP.items():
        assert D.design_family(f"{group}/rd1") == family
        assert family in D.DESIGN_FAMILIES
    with pytest.raises(ValueError, match="no declared design family"):
        D.design_family("NOVEL/rd1")


def test_the_interaction_is_the_difference_of_unit_mean_averages():
    record = D.interaction_bootstrap(
        [1.0, 3.0, 10.0], ["u1", "u1", "u2"],
        [0.0, 2.0], ["v1", "v2"],
        resamples=64, seed=1,
    )
    # Unit means are (2.0, 10.0) and (0.0, 2.0): 6.0 - 1.0.
    assert record["point"] == pytest.approx(5.0)
    assert record["design_side"] == pytest.approx(6.0)
    assert record["natural_side"] == pytest.approx(1.0)


def test_the_interaction_is_refused_when_either_side_is_below_the_floor():
    many = ([1.0] * 12, [f"u{index}" for index in range(12)])
    few = ([0.0] * 4, [f"v{index}" for index in range(4)])
    for design, natural in ((many, few), (few, many)):
        record = D.interaction_bootstrap(*design, *natural, resamples=64, seed=1)
        assert record["degenerate"] is True
        assert record["interval"] is None
        assert record["point"] is not None
        assert D.interaction_outcome(record) == "unresolved"


def test_the_interval_is_taken_on_the_difference_and_not_on_two_means():
    # Both sides are the same numbers in the same order, so a joint draw that
    # recomputed the difference inside the replicate straddles zero. The failure
    # this catches is an interval assembled from two separately resampled means,
    # which on identical sides would be centred on zero but far too narrow only
    # if the draws were shared -- and far too wide if they were not. What must
    # hold is that the point is exactly zero and the interval contains it.
    values = list(np.linspace(-1.0, 1.0, 16))
    units = [f"u{index}" for index in range(16)]
    record = D.interaction_bootstrap(values, units, values, units, resamples=512, seed=7)
    assert record["point"] == pytest.approx(0.0)
    assert record["interval"][0] < 0.0 < record["interval"][1]
    assert record["excludes_zero"] is False


@pytest.mark.parametrize(
    "interval,point,expected",
    [
        ([-0.40, -0.20], -0.30, "confirms"),
        ([-0.12, -0.02], -0.07, "attenuated"),
        ([-0.10, 0.05], -0.02, "refutes"),
        ([0.10, 0.30], 0.20, "refutes"),
        ([-0.45, 0.05], -0.20, "unresolved"),
        (None, -0.30, "unresolved"),
    ],
)
def test_the_four_way_reading_separates_underpower_from_refutation(interval, point, expected):
    assert D.interaction_outcome({"interval": interval, "point": point}) == expected


def test_a_wide_interval_around_a_large_point_is_never_a_refutation():
    # The failure mode this programme keeps catching: an underpowered interval
    # read as absence. A point at the declared magnitude with an interval that
    # covers zero must come back unresolved, not refuted.
    assert (
        D.interaction_outcome({"interval": [-1.0, 0.3], "point": -0.5}) == "unresolved"
    )


def test_the_adjusted_interaction_reduces_to_the_unadjusted_one_without_covariates():
    values = [1.0, 3.0, 10.0, 0.0, 2.0]
    units = ["d1", "d1", "d2", "n1", "n2"]
    designed = [True, True, True, False, False]
    plain = D.interaction_bootstrap(
        values[:3], units[:3], values[3:], units[3:], resamples=32, seed=3
    )
    adjusted = D.adjusted_interaction_bootstrap(
        values, units, designed, np.zeros((5, 0)), resamples=32, seed=3
    )
    assert adjusted["point"] == pytest.approx(plain["point"])
    assert adjusted["n_design_units"] == 2
    assert adjusted["n_natural_units"] == 2


def test_the_adjusted_interaction_removes_a_covariate_that_explains_the_gap():
    # Designs and naturals differ only through the covariate: y = 2 * x, with the
    # designs sitting at high x. The unadjusted difference is large and the
    # adjusted coefficient is zero, which is exactly what a length adjustment has
    # to be able to do before it can be trusted to report a residual.
    x = np.array([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                  14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0])
    designed = list(x >= 14.0)
    values = (2.0 * x).tolist()
    units = [f"{'d' if flag else 'n'}{index}" for index, flag in enumerate(designed)]
    adjusted = D.adjusted_interaction_bootstrap(
        values, units, designed, x[:, None], resamples=64, seed=5
    )
    assert adjusted["point"] == pytest.approx(0.0, abs=1e-9)


def test_the_adjusted_interaction_refuses_a_unit_that_mixes_the_two_sides():
    with pytest.raises(ValueError, match="mixes designs and naturals"):
        D.adjusted_interaction_bootstrap(
            [1.0, 2.0], ["u1", "u1"], [True, False], np.zeros((2, 0)),
            resamples=8, seed=1,
        )


def test_balance_is_reported_rather_than_assumed_from_a_restriction():
    left = np.array([[1.0, 10.0], [2.0, 11.0], [3.0, 12.0]])
    close = left + 0.05
    far = left + 10.0
    assert D.standardised_mean_differences(left, close, ["a", "b"])["balanced"] is True
    report = D.standardised_mean_differences(left, far, ["a", "b"])
    assert report["balanced"] is False
    assert report["max_abs_smd"] > 0.25
