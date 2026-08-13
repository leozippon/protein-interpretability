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
