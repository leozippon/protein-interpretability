"""Ways a language-mediated alignment could report a correspondence it has not found.

Every failure mode below produces a finite, plausible artefact rather than an
error, which is why each test is stated against a construction whose answer is
known before the code runs. The criteria are EXP-R2-213's and are not restated
here except where a test pins one.

**The primary statistic needs a common gallery, or it is not one statistic.**
Top-1 accuracy in excess of chance is only a single number if every target ranks
against the same number of admissible distractors: after the near-duplicate
exclusion a record with three copies in the cohort would otherwise rank in a
smaller field than a singleton. The drawn field is checked to be common, to
exclude every near-duplicate, and to be reproducible from its seed -- which is
what makes the paired bootstrap paired.

**A near-duplicate must not be retrievable as its own neighbour.** L30 measured
42.5% of a held-out Swiss-Prot block keeping a >=95%-identity relative.

**The shuffled-pair null must sit on chance, with and without a signal to
destroy**, because the identity that puts it there does not depend on one: the
truth carried by its ``1/n`` fixed points is exactly the deficit the remaining
draws leave. **The rank-matched null must be matched and must contain no truth**:
a free permutation inside a width-``b`` block leaves ``1/b`` of the records paired
with their own partner, an eighth of the answer at the default width.

**A map fitted on the fit split must never be refitted on the evaluation one.**
Checked behaviourally, on a construction where an eval-fitted map would be
near-perfect and the fit-fitted one is at chance, and again on the digest of the
arrays the map was fitted from.

**The ladder is an order.** A rung may not be read before the rungs below it have
been fitted, and the mean rung may not be the deciding one at all, because
A35-1's shuffled-fit baseline is vacuous for a map that never reads the pairing.

**Attainability comes before control, and A35-0's gate is ONE baseline.**
EXP-R2-213 names it in the singular -- A35-1's margin over ``shuffled_pair`` --
and the singular is load-bearing: a raw arm that clears ``shuffled_pair`` and
loses to a 3-mer surrogate is the pre-declared surface-statistics branch, a
measured negative about the method, while a raw arm that cannot clear
``shuffled_pair`` is a void instrument. A gate read off the whole decisive set
files the first under the second, which is what happened on the production
cohort at a ``shuffled_pair`` margin of 183x.

**``REFERENCE_ONLY`` is not ``FAIL``.** One says this checkpoint is not an arm and
the other says this arm lost, so the reference's criteria are read on its own
numbers, its masked arm is computed, and its verdict still authorises nothing.

**A concept's identifier is not its name.** ``masked_terms`` is a vocabulary of
the surface forms the cohort removed from the descriptions, so the bridge test is
keyed on ``sequence_description.concept_surface_forms`` -- the object the cohort
stage built that vocabulary from -- and never on a GO id or an EC number, which
cannot appear in it.

**An amendment recorded in the register and not implemented in the instrument is
a detectable gap.** The artefact's declaration of which amendments the code
implements is checked against the frozen constants each of them implies.

**The pre-adaptation checkpoint's protein mode must refuse a behavioural read.**
It pays -0.0013 nats/residue to have a sequence reversed (EXP-R2-152), so there
is no graded response for a causal stage to read, and the code -- not a docstring
-- has to be what says no.

**Both of A35-1's conditions are required and are separate.** Significance alone
is a detection criterion and does not license a comparative claim; the effect-size
bar is checked independently of the interval, and a baseline missing from the
table is a refusal rather than a shorter table.

**The instrument must recover an alignment that exists and refuse one that does
not.** Both cells of the synthetic check are run end to end, because a linear
alignment's retrieval number is an unsupervised claim that nothing in a real run
can falsify.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import concept_alignment as ca  # noqa: E402
from src.transfer.statistics import MINIMUM_BOOTSTRAP_UNITS  # noqa: E402


def _load_stage(filename: str):
    path = REPO_ROOT / "scripts/transfer" / filename
    spec = importlib.util.spec_from_file_location(f"_alignment_stage_{filename[:2]}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Registered before execution: a frozen dataclass defined in the module
    # resolves its own annotations through ``sys.modules[cls.__module__]``, so a
    # module that is not there yet fails at class-creation time.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE = _load_stage("35_concept_alignment.py")

#: EXP-R2-213's frozen numbers, as a command line. Every one of these is a flag
#: the stage refuses to default.
FROZEN = [
    "--layers", "8",
    "--decision-layer", "8",
    "--decision-threshold", "0.0",
    "--pca-components", "20",
    "--pooling", "mean_content",
    "--alignment-method", "procrustes",
    "--gallery-size", "16",
    "--excess-ratio", "2.0",
]


def _settings(**overrides) -> argparse.Namespace:
    """A resolved argument set for the analysis, with no checkpoint in it."""

    args = STAGE.build_parser().parse_args(
        ["--synthetic-check", *FROZEN, "--null-draws", "40",
         "--bootstrap-draws", "200", "--synthetic-records", "240"]
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    STAGE.resolve(args)
    return args


def _synthetic_status() -> dict:
    return {"checkpoint_name": "synthetic", "measurable": True}


# --------------------------------------------------------------- the gallery


def test_the_common_gallery_is_common_excludes_duplicates_and_is_reproducible() -> None:
    """The three properties the primary statistic's chance level rests on."""

    groups = [f"g{index // 3}" for index in range(30)]
    field = ca.common_gallery(groups, gallery_size=8, seed=11)
    assert field.shape == (30, 8)
    assert np.array_equal(field[:, 0], np.arange(30))
    labels = np.asarray(groups)
    assert not np.any(labels[field[:, 1:]] == labels[:, None])
    assert np.array_equal(field, ca.common_gallery(groups, gallery_size=8, seed=11))
    assert not np.array_equal(field, ca.common_gallery(groups, gallery_size=8, seed=12))


def test_a_gallery_larger_than_the_admissible_field_is_refused() -> None:
    """Shrinking it for one query alone would give that query a different chance level."""

    groups = ["a"] * 6 + ["b"] * 2
    with pytest.raises(ValueError, match="admissible distractors"):
        ca.common_gallery(groups, gallery_size=6, seed=0)


def test_top1_chance_is_exactly_one_over_the_common_gallery() -> None:
    rng = np.random.default_rng(0)
    groups = [f"g{index // 2}" for index in range(24)]
    field = ca.common_gallery(groups, gallery_size=12, seed=1)
    metrics = ca.retrieval_metrics(
        rng.normal(size=(24, 5)), rng.normal(size=(24, 5)), groups, gallery=field
    )
    assert metrics["common_gallery_size"] is True
    assert metrics["gallery_size"] == 12.0
    assert metrics["top1_chance"] == pytest.approx(1.0 / 12.0)
    assert metrics["top1_excess"] == pytest.approx(metrics["top1_accuracy"] - 1.0 / 12.0)


def test_a_gallery_index_carrying_a_near_duplicate_is_refused() -> None:
    """The one field that must never contain a near-copy of the answer."""

    groups = ["g0", "g0", "g1", "g2"]
    bad = np.array([[0, 1], [1, 0], [2, 3], [3, 2]])
    with pytest.raises(ValueError, match="near-duplicate group"):
        ca.ranks_from_scores(np.eye(4), groups, gallery=bad)


def test_a_near_duplicate_cannot_be_retrieved_as_its_own_neighbour() -> None:
    """Planting an exact copy of the target must not change the target's rank."""

    rng = np.random.default_rng(1)
    base_query = rng.normal(size=(8, 6))
    base_gallery = base_query + 0.05 * rng.normal(size=(8, 6))
    alone = ca.retrieval_ranks(base_query, base_gallery, [f"g{i}" for i in range(8)])[0]

    query = base_query.copy()
    gallery = base_gallery.copy()
    gallery[1] = gallery[0]
    query[1] = query[0]
    groups = ["g0", "g0"] + [f"g{i}" for i in range(2, 8)]
    ranks, sizes = ca.retrieval_ranks(query, gallery, groups)
    assert sizes[0] == 7.0 and sizes[2] == 8.0
    assert ranks[0] == pytest.approx(alone[0])


# ------------------------------------------------------------------ the nulls


def _null_draws(query: np.ndarray, gallery: np.ndarray, groups: list[str], *, draws: int):
    return [
        float(np.mean(1.0 / ca.retrieval_ranks(query[order], gallery, groups)[0]))
        for order in ca.shuffled_pairing(len(groups), draws=draws, seed=3)
    ]


def test_the_shuffled_pair_null_is_centred_at_chance_when_there_is_no_signal() -> None:
    rng = np.random.default_rng(2)
    n = 60
    groups = [f"g{index}" for index in range(n)]
    query = rng.normal(size=(n, 12))
    gallery = rng.normal(size=(n, 12))
    truth = ca.retrieval_metrics(query, gallery, groups)
    null = ca.null_distribution(_null_draws(query, gallery, groups, draws=300))
    low, high = null["mean_ci95"]
    assert low <= truth["mrr_chance"] <= high


def test_the_shuffled_pair_null_stays_on_chance_with_a_perfect_signal() -> None:
    """The identity does not depend on the signal, and that is what makes it a null.

    Here the true pairing retrieves almost perfectly, so the ``1/n`` fixed points
    contribute a reciprocal rank of about 1. The null must still sit on chance:
    what the fixed points add is exactly what the remaining draws lose by never
    being able to take the slot the true item occupies.
    """

    rng = np.random.default_rng(2)
    n = 60
    groups = [f"g{index}" for index in range(n)]
    query = rng.normal(size=(n, 12))
    gallery = query + 0.02 * rng.normal(size=(n, 12))
    truth = ca.retrieval_metrics(query, gallery, groups)
    assert truth["mrr"] > 0.95
    null = ca.null_distribution(_null_draws(query, gallery, groups, draws=300))
    low, high = null["mean_ci95"]
    assert low <= truth["mrr_chance"] <= high
    assert truth["mrr"] > null["decision_level"] + 0.5


def test_the_rank_matched_null_holds_the_nuisance_and_contains_no_truth() -> None:
    rng = np.random.default_rng(4)
    lengths = rng.integers(50, 900, size=96).astype(float)
    matched = ca.rank_matched_pairing(lengths, draws=50, seed=5, block=8)
    free = ca.shuffled_pairing(96, draws=50, seed=5)
    matched_quality = ca.pairing_match_quality(lengths, matched)
    free_quality = ca.pairing_match_quality(lengths, free)
    assert matched_quality["max_fixed_point_fraction"] == 0.0
    assert (
        matched_quality["mean_absolute_rank_gap"]
        < free_quality["mean_absolute_rank_gap"] / 3
    )
    assert set(np.unique(matched)) == set(range(96))


def test_a_rank_matched_block_of_one_is_refused() -> None:
    with pytest.raises(ValueError, match="admits only the identity"):
        ca.rank_matched_pairing([1.0, 2.0, 3.0], draws=2, seed=0, block=1)


# ----------------------------------------------------------------- the ladder


def test_procrustes_returns_an_orthogonal_matrix_and_recovers_a_rotation() -> None:
    rng = np.random.default_rng(6)
    rotation = np.linalg.qr(rng.normal(size=(9, 9)))[0]
    source = rng.normal(size=(40, 9))
    target = source @ rotation + 4.0
    fitted = ca.fit_alignment(source, target, "procrustes")
    assert np.allclose(fitted.weight @ fitted.weight.T, np.eye(9), atol=1e-9)
    assert abs(abs(float(np.linalg.det(fitted.weight))) - 1.0) < 1e-9
    assert np.allclose(ca.apply_alignment(fitted, source), target, atol=1e-8)


def test_the_affine_rung_chooses_its_penalty_inside_the_fit_fold() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(48, 6))
    target = source @ rng.normal(size=(6, 6)) + 0.3 * rng.normal(size=(48, 6))
    groups = [f"g{index // 3}" for index in range(48)]
    fitted = ca.fit_alignment(source, target, "affine", groups=groups, seed=8)
    selection = fitted.penalty_selection
    assert selection is not None
    assert selection["selected"] == fitted.penalty
    assert selection["n_folds"] >= 2
    assert len(selection["held_out_residual_per_penalty"]) == len(selection["grid"])
    assert fitted.free_parameters() == 6 * 6 + 6


def test_the_affine_rung_refuses_a_fit_split_it_cannot_fold() -> None:
    rng = np.random.default_rng(9)
    source = rng.normal(size=(10, 4))
    target = rng.normal(size=(10, 4))
    with pytest.raises(ValueError, match="no group-disjoint fold"):
        ca.fit_alignment(source, target, "affine", groups=["one"] * 10, seed=0)
    with pytest.raises(ValueError, match="needs the fit split's grouping"):
        ca.fit_alignment(source, target, "affine")


def test_a_map_fitted_on_the_fit_split_is_never_refitted_on_the_evaluation_one() -> None:
    """The evaluation block is deliberately aligned by a *different* rotation.

    An implementation that refitted on the evaluation split would recover it and
    read a near-perfect retrieval; one that carries the fit split's map across must
    read chance. The digest of the fitted arrays is checked as well, so the
    property is visible in the artefact and not only in this construction.
    """

    rng = np.random.default_rng(10)
    dimension = 8
    fit_rotation = np.linalg.qr(rng.normal(size=(dimension, dimension)))[0]
    eval_rotation = np.linalg.qr(rng.normal(size=(dimension, dimension)))[0]
    fit_source = rng.normal(size=(50, dimension))
    eval_source = rng.normal(size=(50, dimension))
    fitted = ca.fit_alignment(fit_source, fit_source @ fit_rotation, "procrustes")

    groups = [f"g{index}" for index in range(50)]
    field = ca.common_gallery(groups, gallery_size=16, seed=0)
    eval_target = eval_source @ eval_rotation
    carried = ca.retrieval_metrics(
        eval_target, ca.apply_alignment(fitted, eval_source), groups, gallery=field
    )
    refitted = ca.fit_alignment(eval_source, eval_target, "procrustes")
    leaked = ca.retrieval_metrics(
        eval_target, ca.apply_alignment(refitted, eval_source), groups, gallery=field
    )

    assert leaked["top1_accuracy"] == pytest.approx(1.0)
    assert carried["top1_excess"] < 0.05
    assert fitted.fit_digest != refitted.fit_digest
    assert fitted.n_fit == 50


def test_the_ladder_order_is_enforced_in_code() -> None:
    """A35-3: a rung may not be read before the rungs below it were fitted."""

    ca.assert_ladder_reported(["mean", "procrustes"], "procrustes")
    ca.assert_ladder_reported(["mean", "procrustes", "affine"], "affine")
    with pytest.raises(ValueError, match="were not reported"):
        ca.assert_ladder_reported(["affine"], "affine")
    with pytest.raises(ValueError, match="unknown primary rung"):
        ca.assert_ladder_reported(["mean"], "adapter_mlp")


def test_the_mean_rung_cannot_be_the_deciding_one() -> None:
    """Its shuffled-fit null is the truth: a mean shift never reads the pairing."""

    args = STAGE.build_parser().parse_args(
        ["--synthetic-check", "--layers", "8", "--decision-layer", "8",
         "--decision-threshold", "0.0", "--pca-components", "20",
         "--pooling", "mean_content", "--gallery-size", "16",
         "--excess-ratio", "2.0", "--alignment-method", "mean"]
    )
    with pytest.raises(ValueError, match="never reads the pairing"):
        STAGE.resolve(args)


# ------------------------------------------------------------------ concepts


def test_a_concept_vector_is_a_unit_direction_with_the_population_scale() -> None:
    rng = np.random.default_rng(11)
    direction = np.array([0.0, 3.0, 0.0, -4.0])
    labels = np.array([True] * 30 + [False] * 30)
    reps = rng.normal(size=(60, 4)) + np.outer(labels.astype(float), direction)
    vector = ca.concept_vector(reps, labels)
    assert float(np.linalg.norm(vector.direction)) == pytest.approx(1.0)
    assert vector.sigma == pytest.approx(float((reps @ vector.direction).std(ddof=1)))
    assert vector.separation_sigma > 1.0
    assert vector.n_positive == 30 and vector.n_negative == 30
    assert float(np.dot(vector.direction, direction / np.linalg.norm(direction))) > 0.9


def test_a_concept_vector_refuses_one_class_and_an_undeclared_method() -> None:
    reps = np.random.default_rng(12).normal(size=(20, 3))
    with pytest.raises(ValueError, match="two records on each side"):
        ca.concept_vector(reps, np.ones(20, dtype=bool))
    with pytest.raises(ValueError, match="unknown concept-vector method"):
        ca.concept_vector(reps, np.array([True] * 10 + [False] * 10), method="logistic")


def test_the_rank_auc_is_the_reference_implementation_including_ties() -> None:
    """One AUC definition, pinned to sklearn's rather than assumed equal to it."""

    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(15)
    for size, tie_grid in ((60, None), (40, 5), (25, 2)):
        labels = rng.random(size) < 0.4
        if not labels.any() or labels.all():
            continue
        scores = rng.normal(size=size)
        if tie_grid is not None:
            scores = np.round(scores * tie_grid) / tie_grid
        assert ca.concept_auc(scores, labels) == pytest.approx(
            float(roc_auc_score(labels, scores))
        )
    assert np.isnan(ca.auc_metric(np.ones(4, dtype=bool), np.arange(4.0)))


def test_surface_features_are_frequencies_and_refuse_a_non_canonical_symbol() -> None:
    composition = ca.composition_features(["AAAC", "WWWW"])
    assert composition.sum(axis=1) == pytest.approx(np.ones(2))
    assert composition[0, 0] == pytest.approx(0.75)
    kmer = ca.kmer_features(["ACDACD"], k=3)
    assert kmer.sum() == pytest.approx(1.0)
    assert kmer.shape == (1, 20**3)
    with pytest.raises(ValueError, match="outside the canonical alphabet"):
        ca.composition_features(["AAXA"])


# --------------------------------------------------------------------- cohort


def _write_cohort(directory: Path, records: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    (directory / "cohort.json").write_text(
        json.dumps({"manifest": "test"}), encoding="utf-8"
    )
    return directory


def _record(index: int, split: str, dup: str, family: str) -> dict:
    return {
        "accession": f"P{index:05d}",
        "sequence": "ACDEFGHIKLMNPQRSTVWY"[: 10 + index % 5] * 3,
        "length": 30,
        "name": f"Protein {index}",
        "function_text": "does a thing",
        "description_raw": f"Protein {index}, a hydrolase",
        "description_masked": f"Protein {index}, a [MASK]",
        "masked_terms": ["hydrolase"],
        "ec": ["3.1.-.-"],
        "go": [],
        "go_propagated": [],
        "pfam": [],
        "cath": [],
        "dup_group": dup,
        "family_group": family,
        "split": split,
    }


def test_load_cohort_names_the_path_it_could_not_find(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="absent/records.jsonl"):
        ca.load_cohort(tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="elsewhere.jsonl"):
        ca.load_cohort(tmp_path / "elsewhere.jsonl")


def test_load_cohort_refuses_a_near_duplicate_group_that_straddles_two_splits(
    tmp_path: Path,
) -> None:
    """L30's defect, evaluated rather than asserted in the manifest's prose."""

    records = [
        _record(0, "fit", "d0", "f0"),
        _record(1, "fit", "d0", "f0"),
        _record(2, "eval", "d0", "f1"),
        _record(3, "eval", "d2", "f1"),
    ]
    _write_cohort(tmp_path / "straddle", records)
    with pytest.raises(ValueError, match="dup_group values straddle"):
        ca.load_cohort(tmp_path / "straddle")


def test_load_cohort_refuses_a_family_seen_outside_the_family_holdout(
    tmp_path: Path,
) -> None:
    records = [
        _record(0, "fit", "d0", "f0"),
        _record(1, "eval", "d1", "f1"),
        _record(2, "family_holdout", "d2", "f0"),
    ]
    _write_cohort(tmp_path / "family", records)
    with pytest.raises(ValueError, match="family_group values appear both"):
        ca.load_cohort(tmp_path / "family")


def test_load_cohort_refuses_a_record_missing_a_schema_field(tmp_path: Path) -> None:
    record = _record(0, "fit", "d0", "f0")
    record.pop("description_masked")
    _write_cohort(tmp_path / "short", [record])
    with pytest.raises(ValueError, match="description_masked"):
        ca.load_cohort(tmp_path / "short")


# -------------------------------------------------- the pre-adaptation bound


def test_the_pre_adaptation_checkpoint_refuses_a_behavioural_protein_read() -> None:
    """It pays -0.0013 nats/residue to reverse a sequence, so there is nothing to read."""

    with pytest.raises(ValueError, match="no behavioural read"):
        ca.assert_behavioural_read_permitted(Path("/models/Llama-2-7b-hf"), "protein")
    ca.assert_behavioural_read_permitted(Path("/models/Llama-2-7b-hf"), "text")
    ca.assert_behavioural_read_permitted(Path("/models/ProLLaMA_Stage_1"), "protein")
    with pytest.raises(ValueError, match="not declared"):
        ca.assert_behavioural_read_permitted(
            Path("/models/some-unqualified-model"), "protein"
        )


# ----------------------------------------------------- the frozen conditions


def _interval(low: float, high: float) -> dict:
    return {
        "publishable": True,
        "floor": {"degenerate": False},
        "bootstrap": {"difference_ci95": [low, high]},
    }


def _passing_rows(**overrides) -> list[dict]:
    """One row per pre-registered baseline, all passing, unless overridden."""

    return [
        ca.baseline_row(
            name, "primary_top1", 0.8, 0.2, chance=0.1, excess_ratio=2.0,
            interval=overrides.get(f"{name}_interval", _interval(0.2, 0.6)),
        )
        for name in ca.A35_1_BASELINES
    ]


def test_both_conditions_are_required_and_are_evaluated_separately() -> None:
    """Significance without an effect size is a detection criterion, not a claim."""

    ratio_fails = ca.baseline_row(
        "kmer", "primary_top1", 0.5, 0.4, chance=0.1, excess_ratio=2.0,
        interval=_interval(0.05, 0.15),
    )
    assert ratio_fails["difference_interval_excludes_zero"] is True
    assert ratio_fails["meets_excess_ratio"] is False
    assert ratio_fails["passes_both_conditions"] is False

    interval_fails = ca.baseline_row(
        "kmer", "primary_top1", 0.9, 0.2, chance=0.1, excess_ratio=2.0,
        interval=_interval(-0.1, 0.9),
    )
    assert interval_fails["meets_excess_ratio"] is True
    assert interval_fails["difference_interval_excludes_zero"] is False
    assert interval_fails["passes_both_conditions"] is False


def test_a_method_below_chance_never_satisfies_the_effect_size_bar() -> None:
    """``x >= 2 * negative`` is satisfied by a negative x, and must not read as a pass."""

    row = ca.baseline_row(
        "rank_matched", "primary_top1", 0.05, 0.02, chance=0.1, excess_ratio=2.0
    )
    assert row["observed_excess"] < 0.0
    assert row["meets_excess_ratio"] is False


def test_a_missing_pre_registered_baseline_is_a_refusal_not_a_shorter_table() -> None:
    rows = [row for row in _passing_rows() if row["baseline"] != "kmer"]
    with pytest.raises(ValueError, match="kmer"):
        ca.admission_verdict(
            rows, excess_ratio=2.0, detection_floor=0.0, observed_excess=0.7,
            behavioural_status=_synthetic_status(), mode="text",
        )


def test_the_amended_decisive_set_does_not_require_the_bridge_baseline() -> None:
    """Amendment 1: bridge_specific left the decisive set and became A35-1b."""

    verdict = ca.admission_verdict(
        _passing_rows(), excess_ratio=2.0, detection_floor=0.0, observed_excess=0.7,
        behavioural_status=_synthetic_status(), mode="text",
    )
    assert ca.A35_1B_BASELINE not in verdict["decisive_baselines"]
    assert verdict["verdict"] == "PASS"


def test_the_detection_floor_is_read_before_any_comparison() -> None:
    verdict = ca.admission_verdict(
        _passing_rows(), excess_ratio=2.0, detection_floor=0.9, observed_excess=0.7,
        behavioural_status=_synthetic_status(), mode="text",
    )
    assert verdict["verdict"] == "FAIL"
    assert "detection floor" in verdict["reason"]


def test_a_condition_without_a_usable_interval_is_underpowered_not_a_pass() -> None:
    below = STAGE._paired_interval(
        np.array([1.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.asarray(["a", "b", "c", "d"]),
        args=_settings(),
    )
    assert below["publishable"] is False
    assert below["floor"]["minimum_units"] == MINIMUM_BOOTSTRAP_UNITS
    verdict = ca.admission_verdict(
        _passing_rows(kmer_interval=below), excess_ratio=2.0, detection_floor=0.0,
        observed_excess=0.7, behavioural_status=_synthetic_status(), mode="text",
    )
    assert verdict["verdict"] == "UNDERPOWERED"
    assert "kmer" in verdict["baselines_without_a_usable_interval"]


def test_the_pre_adaptation_checkpoint_is_referenced_and_never_admitted() -> None:
    """Every row can pass and the verdict must still not be an admission."""

    verdict = ca.admission_verdict(
        _passing_rows(), excess_ratio=2.0, detection_floor=0.0, observed_excess=0.7,
        behavioural_status=ca.protein_mode_behavioural_status(
            Path("/models/Llama-2-7b-hf")
        ),
        mode="protein",
    )
    assert verdict["verdict"] == "REFERENCE_ONLY"
    assert "behaviourally unmeasurable" in verdict["reason"]


def test_an_inapplicable_baseline_must_carry_its_reason() -> None:
    with pytest.raises(ValueError, match="does not say why"):
        ca.baseline_row(
            "bridge_specific", "primary_top1", 0.5, None, chance=0.1, applicable=False
        )


def test_stop_35_authorises_stage_36_only_on_a_pass() -> None:
    assert STAGE.stop_35_record("PASS")["stage36_authorised"] is True
    for verdict in (
        "FAIL", "UNDERPOWERED", "REFERENCE_ONLY", "VOID_SPECIFICATION_DEFECT",
    ):
        record = STAGE.stop_35_record(verdict)
        assert record["triggered"] is True
        assert record["stage36_authorised"] is False
        assert "UNAUTHORISED" in record["reason"]


# ------------------------------------------ A35-0's gate, A35-1b, the reference

#: The ProLLaMA_Stage_1 protein cell's own raw-arm numbers at the decision layer
#: on the eval split, as the campaign of 2026-08-19 measured them: a common
#: gallery of 1,000, so chance is 0.001. ``shuffled_pair`` is cleared by 183x on
#: the excess-over-chance scale clause (ii) is written on, and ``kmer`` is not
#: cleared at all. The pair is the whole of the defect these tests pin: one
#: outcome is a void instrument and the other is a measured negative about the
#: method, and a gate read off both at once reports the second as the first.
PRIMARY_CELL_CHANCE = 0.001
PRIMARY_CELL_OBSERVED_EXCESS = 0.011447210491220272
PRIMARY_CELL_BASELINE_EXCESS = {
    "shuffled_pair": 6.24583240720158e-05,
    "rank_matched": 0.0012249388753056236,
    "shuffled_fit": -6.723716381418091e-06,
    "composition": 0.005668148477439431,
    "kmer": 0.029451211380306735,
    "description_only": 0.0005559013114025339,
}


def _rows_from_excesses(
    excesses: dict,
    *,
    observed_excess: float = PRIMARY_CELL_OBSERVED_EXCESS,
    chance: float = PRIMARY_CELL_CHANCE,
    bridge_excess: float | None = None,
    bridge_decisive: bool = False,
    bridge_inapplicable_reason: str | None = None,
) -> list[dict]:
    """One primary-statistic row per baseline, built from excesses over chance."""

    accuracy = chance + observed_excess
    rows = [
        ca.baseline_row(
            name, "primary_top1", accuracy, chance + excess, chance=chance,
            excess_ratio=2.0, interval=_interval(0.001, 0.02),
        )
        for name, excess in excesses.items()
    ]
    if bridge_excess is not None:
        rows.append(
            ca.baseline_row(
                ca.A35_1B_BASELINE, "primary_top1", accuracy, chance + bridge_excess,
                chance=chance, excess_ratio=2.0, interval=_interval(0.001, 0.02),
                decisive=bridge_decisive,
                applicable=bridge_inapplicable_reason is None,
                inapplicable_reason=bridge_inapplicable_reason,
            )
        )
    return rows


def _cell(rows_per_split: dict, *, method: str = "procrustes") -> dict:
    """The part of a cell A35-0, A35-1b and the verdict actually read."""

    return {
        "splits": {
            split: {
                "baseline_rows": rows,
                "ladder": {
                    method: {
                        STAGE.PRIMARY_DIRECTION: {
                            "top1_excess": rows[0]["observed_excess"],
                            "top1_accuracy": rows[0]["observed"],
                        }
                    }
                },
            }
            for split, rows in rows_per_split.items()
        }
    }


def _protein_settings(**overrides) -> argparse.Namespace:
    """A resolved analysis argument set read as a protein cell.

    ``--mode`` cannot be parsed beside ``--synthetic-check`` -- the stage refuses a
    campaign flag there and that refusal is itself tested -- so the mode is set
    after resolution, which is the only thing the verdict reads it for.
    """

    args = _settings(**overrides)
    args.mode = "protein"
    return args


def _primary_cell_rows(**overrides) -> list[dict]:
    return _rows_from_excesses(dict(PRIMARY_CELL_BASELINE_EXCESS), **overrides)


def test_a35_0_gates_on_shuffled_pair_alone_and_a_surrogate_failure_is_not_a_void() -> None:
    """The production cell: shuffled_pair cleared 183x, kmer lost, and it is not VOID.

    EXP-R2-213 states A35-0 over ``shuffled_pair`` in the singular, and its branch
    table gives the two outcomes opposite subjects. Reading the gate off the whole
    decisive set makes a surface-statistics negative -- a statement about the
    method, read on the masked arm -- arrive as a void instrument.
    """

    args = _settings()
    rows = _primary_cell_rows()
    cell = _cell({"eval": rows})
    gate_row = next(row for row in rows if row["baseline"] == "shuffled_pair")
    assert gate_row["observed_excess"] / gate_row["baseline_excess"] > 100.0
    assert gate_row["passes_both_conditions"] is True

    a35_0 = STAGE.attainability_verdict(cell, args, status=_synthetic_status())
    assert a35_0["verdict"] == "ATTAINABLE"
    assert a35_0["attainable"] is True
    assert a35_0["gate_baseline"] == ca.A35_0_GATE_BASELINE == "shuffled_pair"
    assert a35_0["gate"]["row"]["baseline"] == "shuffled_pair"
    assert a35_0["gate"]["status"] == "cleared"

    # The rest of the table is reported and disagrees, which is the point: the raw
    # arm's own A35-1 verdict fails on the 3-mer surrogate and gates nothing.
    raw_verdict = a35_0["raw_arm_verdict"]
    assert raw_verdict["verdict"] == "FAIL"
    assert "kmer" in raw_verdict["baselines_failing_a_condition"]

    # And the outcome routes to A35-1's surface-statistics branch, not to a void.
    masked = ca.admission_verdict(
        rows, excess_ratio=2.0, detection_floor=0.0,
        observed_excess=PRIMARY_CELL_OBSERVED_EXCESS,
        behavioural_status=_synthetic_status(), mode="protein",
    )
    branch = STAGE.frozen_branch(masked["verdict"], {"eval": masked})
    assert masked["verdict"] == "FAIL"
    assert branch["branch"] == "surface_statistics"
    assert branch["statement_about"] == "the method"
    assert "kmer" in branch["surrogate_baselines_failing"]
    assert STAGE.stop_35_record(masked["verdict"])["stage36_authorised"] is False


def test_a35_0_is_void_only_when_its_own_gate_baseline_fails() -> None:
    """A raw arm that cannot clear shuffled_pair has aligned nothing at all."""

    args = _settings()
    excesses = dict(PRIMARY_CELL_BASELINE_EXCESS)
    excesses["shuffled_pair"] = PRIMARY_CELL_OBSERVED_EXCESS  # no margin at all
    a35_0 = STAGE.attainability_verdict(
        _cell({"eval": _rows_from_excesses(excesses)}), args, status=_synthetic_status()
    )
    assert a35_0["verdict"] == "VOID_SPECIFICATION_DEFECT"
    assert a35_0["attainable"] is False
    assert a35_0["gate"]["status"] == "not_cleared"
    assert a35_0["gate"]["meets_excess_ratio"] is False
    branch = STAGE.frozen_branch(a35_0["verdict"], {})
    assert branch["branch"] == "a35_0_specification_defect"
    assert branch["statement_about"] == "the instrument, not the modality"


def test_a35_0_refuses_a_table_that_does_not_carry_its_gate_baseline() -> None:
    """The gate may not fall back to another baseline; which one it was is the point."""

    rows = [row for row in _primary_cell_rows() if row["baseline"] != "shuffled_pair"]
    with pytest.raises(ValueError, match="shuffled_pair"):
        ca.attainability_gate(rows)


def test_an_unreadable_gate_is_not_a_cleared_gate() -> None:
    """A margin that could not be bounded has not been shown, so the ladder is void."""

    excesses = dict(PRIMARY_CELL_BASELINE_EXCESS)
    rows = _rows_from_excesses(excesses)
    for row in rows:
        if row["baseline"] == "shuffled_pair":
            row["interval_status"] = "below_unit_floor"
            row["difference_interval_excludes_zero"] = None
            row["passes_both_conditions"] = None
    gate = ca.attainability_gate(rows)
    assert gate["status"] == "not_evaluable"
    assert gate["attainable"] is False


def test_a_reference_only_cell_reads_its_own_numbers_and_authorises_nothing() -> None:
    """REFERENCE_ONLY says this checkpoint is not an arm, not that this arm lost.

    Ordering the reference branch before the criteria made A35-0 unreachable for
    the pre-adaptation checkpoint whatever its numbers were, so its masked arm was
    withheld and the protein-side A35-4 was read on the non-deciding raw variant.
    """

    status = ca.protein_mode_behavioural_status(Path("/models/Llama-2-7b-hf"))
    verdict = ca.admission_verdict(
        _passing_rows(), excess_ratio=2.0, detection_floor=0.0, observed_excess=0.7,
        behavioural_status=status, mode="protein",
    )
    assert verdict["verdict"] == "REFERENCE_ONLY"
    assert verdict["criteria_verdict"] == "PASS"
    assert STAGE.stop_35_record(verdict["verdict"])["stage36_authorised"] is False

    # ... and a reference whose own numbers do not hold reads FAIL on the criteria
    # while the label it carries is unchanged.
    losing = ca.admission_verdict(
        _passing_rows(), excess_ratio=2.0, detection_floor=0.9, observed_excess=0.7,
        behavioural_status=status, mode="protein",
    )
    assert losing["verdict"] == "REFERENCE_ONLY"
    assert losing["criteria_verdict"] == "FAIL"

    # A35-0 is read on the gate baseline and never on the cell verdict, so the
    # reference's masked arm is computed and there is something to compare against
    # on the deciding variant.
    a35_0 = STAGE.attainability_verdict(
        _cell({"eval": _primary_cell_rows()}), _protein_settings(), status=status
    )
    assert a35_0["attainable"] is True
    assert a35_0["raw_arm_verdict"]["verdict"] == "REFERENCE_ONLY"
    assert STAGE.frozen_branch("REFERENCE_ONLY", {})["branch"] == "pre_adaptation_reference"


def test_a35_1b_gates_only_where_the_raw_arm_shows_the_margin_is_reachable() -> None:
    """Amendment 1: the restriction is reported, and decisive only where it can be."""

    args = _settings()
    reachable = _cell(
        {
            split: _primary_cell_rows(bridge_excess=0.001)
            for split in ("eval", "family_holdout")
        }
    )
    block = STAGE.a35_1b_restriction(reachable, None, args)
    assert block["gating"] is True
    assert block["inapplicable_reason"] is None
    assert block["bridge_over_full"]["raw"]["eval"]["bridge_over_full"] < 0.5

    # The amendment's own case: a restriction that loses almost nothing, which the
    # hypothesis predicts rather than forbids.
    unreachable_excess = 0.9 * PRIMARY_CELL_OBSERVED_EXCESS
    unreachable = _cell(
        {
            split: _primary_cell_rows(bridge_excess=unreachable_excess)
            for split in ("eval", "family_holdout")
        }
    )
    block = STAGE.a35_1b_restriction(unreachable, None, args)
    assert block["gating"] is False
    assert block["margin_reachable_on_the_raw_arm"] is False
    assert block["inapplicable_reason"] is not None
    ratios = block["bridge_over_full"]["raw"]
    assert set(ratios) == {"eval", "family_holdout"}
    assert ratios["eval"]["bridge_over_full"] == pytest.approx(0.9)

    # And a non-gating A35-1b cannot change the verdict: the row is reported with
    # its reason and decides nothing.
    rows = _rows_from_excesses(
        {name: 0.0001 for name in ca.A35_1_BASELINES},
        bridge_excess=unreachable_excess,
        bridge_decisive=True,
        bridge_inapplicable_reason=block["inapplicable_reason"],
    )
    verdict = ca.admission_verdict(
        rows, excess_ratio=2.0, detection_floor=0.0,
        observed_excess=PRIMARY_CELL_OBSERVED_EXCESS,
        behavioural_status=_synthetic_status(), mode="text",
    )
    assert verdict["verdict"] == "PASS"
    assert ca.A35_1B_BASELINE not in verdict["decisive_applicable_baselines"]


def test_a35_1b_gating_makes_the_bridge_baseline_decide() -> None:
    """Where the margin IS reachable the restriction is a criterion, not a comment."""

    rows = _rows_from_excesses(
        {name: 0.0001 for name in ca.A35_1_BASELINES},
        bridge_excess=0.9 * PRIMARY_CELL_OBSERVED_EXCESS,
        bridge_decisive=True,
    )
    verdict = ca.admission_verdict(
        rows, excess_ratio=2.0, detection_floor=0.0,
        observed_excess=PRIMARY_CELL_OBSERVED_EXCESS,
        behavioural_status=_synthetic_status(), mode="text",
    )
    assert verdict["verdict"] == "FAIL"
    assert ca.A35_1B_BASELINE in verdict["baselines_failing_a_condition"]


# --------------------------------------------- the bridge concept's two sides


def _bridge_record(masked_terms: list[str]) -> dict:
    return {"masked_terms": masked_terms}


def test_bridge_concepts_are_keyed_on_the_surface_forms_the_cohort_masked() -> None:
    """A GO id is not what a curated description calls the concept.

    The cohort stage builds ``masked_terms`` from
    ``sequence_description.concept_surface_forms``, so the consumer asks the same
    object. Keyed on the identifier instead, the two vocabularies cannot meet: on
    the production cohort none of 3,970 distinct masked terms was GO-id-shaped or
    Pfam-accession-shaped, and the intersection was empty for all 1,174 concepts.
    """

    records = [
        _bridge_record(["membrane", "ATP"]),
        _bridge_record(["DNA repair"]),
    ]
    concepts = [
        ("go_propagated", "GO:0016020"),
        ("go_propagated", "GO:0006281"),
        ("go_propagated", "GO:0005524"),
        ("pfam", "PF00069"),
    ]
    surface_forms = {
        ("go_propagated", "GO:0016020"): ("GO:0016020", "membrane", "whole membrane"),
        ("go_propagated", "GO:0006281"): ("GO:0006281", "DNA repair"),
        ("go_propagated", "GO:0005524"): ("GO:0005524", "ATP binding"),
    }
    bridge = ca.bridge_concepts(records, concepts, surface_forms=surface_forms)
    assert bridge == [("go_propagated", "GO:0016020"), ("go_propagated", "GO:0006281")]

    # The identifier-keyed reading finds nothing, because no identifier is in the
    # vocabulary at all -- which is what made the production run report 0 of 1,174.
    vocabulary = ca.masked_term_vocabulary(records)
    assert vocabulary == {"membrane", "atp", "dna repair"}
    assert not any(term.lower() in vocabulary for _, term in concepts)

    # A concept the cohort declares no surface form for has no declared text side.
    assert ("pfam", "PF00069") not in bridge


def test_a_bridge_concept_is_matched_on_a_whole_form_and_not_a_substring() -> None:
    """``ligase`` inside ``DNA ligase`` is a different concept's name."""

    records = [_bridge_record(["DNA ligase"])]
    forms = {("ec", "6"): ("ligase", "EC 6")}
    assert ca.bridge_concepts(records, [("ec", "6")], surface_forms=forms) == []
    assert ca.bridge_concepts(
        [_bridge_record(["ligase"])], [("ec", "6")], surface_forms=forms
    ) == [("ec", "6")]


def test_the_declared_surface_forms_come_from_the_cohorts_own_ontology(
    tmp_path: Path,
) -> None:
    """End to end over the producer's own function, including a synonym-only match.

    On the production cohort ``go_metal_ion_binding`` is named in the descriptions
    only through the ontology synonym ``metal binding``, never through its primary
    name, so a consumer that used the declared name alone would have missed it.
    """

    obo = tmp_path / "go-basic.obo"
    obo.write_text(
        "format-version: 1.2\n"
        "data-version: releases/2026-01-01\n"
        "\n"
        "[Term]\n"
        "id: GO:0046872\n"
        "name: metal ion binding\n"
        "namespace: molecular_function\n"
        'synonym: "metal binding" EXACT []\n',
        encoding="utf-8",
    )
    forms = STAGE.declared_surface_forms(obo)
    key = ("go_propagated", "GO:0046872")
    assert "metal binding" in forms[key]
    # Reached from either GO column, because a cohort concept key carries the
    # column it was declared from.
    assert forms[("go", "GO:0046872")] == forms[key]

    records = [_bridge_record(["metal binding"])]
    assert ca.bridge_concepts(records, [key], surface_forms=forms) == [key]


# ------------------------------------------------- the amendment declaration


def test_the_declared_amendments_match_the_constants_they_imply() -> None:
    """An amendment in the register that the instrument does not implement is a gap.

    EXP-R2-213 amendment 1 was recorded on 2026-08-19 and stage 35 still carried
    the pre-amendment seven-baseline set; nothing detected it, and the campaign ran
    under a criterion the register said had been superseded. This is that gap made
    detectable: the artefact declares which amendments the code implements, and the
    declaration is checked against the frozen constants each one implies.
    """

    block = STAGE.pre_registration_block(_settings())
    assert block["record"] == ca.PRE_REGISTRATION == "EXP-R2-213"
    assert block["amendments_implemented"] == list(ca.PRE_REGISTRATION_AMENDMENTS)
    assert block["decisive_baselines"] == list(ca.A35_1_BASELINES)
    assert block["a35_0_gate_baseline"] == ca.A35_0_GATE_BASELINE
    assert block["a35_1b_baseline"] == ca.A35_1B_BASELINE

    if "amendment 1" in block["amendments_implemented"]:
        # The decisive set is SIX, named, and bridge_specific is not one of them.
        assert set(ca.A35_1_BASELINES) == {
            "shuffled_pair", "shuffled_fit", "rank_matched",
            "composition", "kmer", "description_only",
        }
        assert ca.A35_1B_BASELINE == "bridge_specific"
        assert ca.A35_1B_BASELINE not in ca.A35_1_BASELINES
        # The primary statistic is description -> sequence top-1 in excess of chance.
        assert ca.PRIMARY_STATISTIC == "top1_excess"
        assert STAGE.PRIMARY_DIRECTION == "description_to_sequence"
        # The mean rung cannot be the deciding one.
        with pytest.raises(ValueError, match="mean rung cannot be the decisive one"):
            _settings(alignment_method="mean")

    unknown = [
        name
        for name in ca.PRE_REGISTRATION_AMENDMENTS
        if not name.startswith("amendment ")
    ]
    assert not unknown, unknown


def test_the_verdict_and_the_artefact_declare_one_amendment_set() -> None:
    """Two declarations of what a run was produced under would eventually disagree."""

    verdict = ca.admission_verdict(
        _passing_rows(), excess_ratio=2.0, detection_floor=0.0, observed_excess=0.7,
        behavioural_status=_synthetic_status(), mode="text",
    )
    block = STAGE.pre_registration_block(_settings())
    assert verdict["amendments_implemented"] == block["amendments_implemented"]
    assert verdict["decisive_baselines"] == block["decisive_baselines"]

# ----------------------------------------------------- the L32 per-layer rule


def test_an_artefact_reporting_a_cross_layer_aggregate_is_refused() -> None:
    """L32: a criterion stated per layer, read as a mean over layers, disagrees in verdict."""

    ca.assert_per_layer_only({"cells": {"layer8__raw": {"top1": 0.4}}}, [8])
    with pytest.raises(ValueError, match="aggregate over layers"):
        ca.assert_per_layer_only(
            {"cells": {"layer8__raw": {"top1_mean_over_layers": 0.4}}}, [8]
        )
    with pytest.raises(ValueError, match="carries no cell of its own"):
        ca.assert_per_layer_only({"cells": {"layer8__raw": {}}}, [8, 16])


# ----------------------------------------------------- representation guard


def test_mode_representations_refuses_anything_but_the_loaded_handle() -> None:
    """The joint-checkpoint loader is stage 21's, and this module carries no second one."""

    with pytest.raises(TypeError, match="loaded joint checkpoint handle"):
        ca.mode_representations(
            Path("/models/ProLLaMA_Stage_1"), "prollama", "protein", ["ACDEF"],
            [8], "mean_content", "cuda:0", 1, "float32",
        )


# -------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def synthetic_check() -> dict:
    """One run of the instrument check, shared by the tests that read it."""

    return STAGE.run_synthetic_check(_settings())


def test_the_instrument_recovers_a_planted_alignment_and_refuses_none(
    synthetic_check: dict,
) -> None:
    certificate = synthetic_check["certificate"]
    assert certificate["planted_alignment"]["observed"] == "PASS"
    assert certificate["no_alignment"]["observed"] == "FAIL"
    assert all(row["agrees"] for row in certificate.values())
    planted = certificate["planted_alignment"]
    assert planted["eval_top1"] > planted["eval_top1_chance"] + 0.3
    assert synthetic_check["cells"]["planted_alignment"]["stop_35"]["stage36_authorised"]
    assert not synthetic_check["cells"]["no_alignment"]["stop_35"]["stage36_authorised"]


def test_every_pre_registered_baseline_is_evaluated_under_both_conditions(
    synthetic_check: dict,
) -> None:
    """A35-1 in the artefact: six decisive rows, each with both conditions read."""

    rows = synthetic_check["cells"]["planted_alignment"]["splits"]["eval"]["baseline_rows"]
    decisive = {row["baseline"]: row for row in rows if row["decisive"]}
    assert set(decisive) == set(ca.A35_1_BASELINES)
    assert len(ca.A35_1_BASELINES) == 6
    # A35-1b is reported beside them and is not one of them until its margin has
    # been shown reachable on the raw arm.
    bridge = [row for row in rows if row["baseline"] == ca.A35_1B_BASELINE]
    assert len(bridge) == 1 and bridge[0]["decisive"] is False
    for name, row in decisive.items():
        assert row["applicable"] is True, name
        assert row["difference_interval_excludes_zero"] is not None, name
        assert row["meets_excess_ratio"] is not None, name
        assert row["chance"] == pytest.approx(1.0 / 16.0)
    assert any(
        row["baseline"] == "nearest_neighbour" and not row["decisive"] for row in rows
    )


def test_the_surface_surrogate_is_a_live_competitor_rather_than_a_formality(
    synthetic_check: dict,
) -> None:
    """One latent factor drives composition, so the surrogate must recover part of it.

    D3.b died to a conditioning leak and F12 to a free hydropathy baseline. A
    composition baseline that reads chance on data whose composition carries the
    concept is not a baseline, it is a decoration.
    """

    # The features themselves carry the concept, by construction and deterministically.
    inputs = STAGE.synthetic_inputs(_settings(), planted=True)
    fit = inputs.index("fit")
    held = inputs.index("eval")
    for features in (
        ca.composition_features(inputs.sequences),
        ca.kmer_features(inputs.sequences, k=3),
    ):
        key = inputs.concepts[0]
        vector = ca.concept_vector(features[fit], inputs.labels[key][fit])
        assert ca.concept_auc(vector.project(features[held]), inputs.labels[key][held]) > 0.7

    # And the surrogate arm carries that signal into the retrieval task itself.
    rows = {
        row["baseline"]: row
        for row in synthetic_check["cells"]["planted_alignment"]["splits"]["eval"][
            "baseline_rows"
        ]
    }
    for name in ("composition", "kmer"):
        assert rows[name]["baseline_excess"] > 0.0, name


def test_the_description_only_arm_is_a_retrieval_arm_on_the_same_gallery(
    synthetic_check: dict,
) -> None:
    """Commensurability: the same statistic, the same field, or A35-1(ii) is ill-posed."""

    block = synthetic_check["cells"]["planted_alignment"]["splits"]["eval"]
    arm = block["arms"]["description_only"]
    assert arm["gallery_size"] == 16.0
    assert arm["top1_chance"] == pytest.approx(1.0 / 16.0)
    assert arm["construction"] in ("gallery_typicality", "length_prior")
    components = block["description_only_components"]
    assert arm["top1_accuracy"] == pytest.approx(
        max(
            components["gallery_typicality"]["top1_accuracy"],
            components["length_prior"]["top1_accuracy"],
        )
    )
    assert block["concept"]["description_only_ceiling"]["is_decisive"] is False


def test_the_attainability_arm_gates_the_control_arm(synthetic_check: dict) -> None:
    """A35-0: where the raw arm cannot clear the bar, the ladder is VOID."""

    args = _settings()
    status = _synthetic_status()
    planted = STAGE.attainability_verdict(
        synthetic_check["cells"]["planted_alignment"], args, status=status
    )
    absent = STAGE.attainability_verdict(
        synthetic_check["cells"]["no_alignment"], args, status=status
    )
    assert planted["verdict"] == "ATTAINABLE" and planted["attainable"] is True
    assert absent["verdict"] == "VOID_SPECIFICATION_DEFECT"
    assert absent["attainable"] is False
    assert "specification defect" in absent["reason"]


# ------------------------------------------------------------------ artefact


def test_the_artefact_basename_carries_checkpoint_mode_and_rendering() -> None:
    """21_joint_mode_qualification.py:919 writes every run to one fixed name."""

    first = STAGE.artefact_name(Path("/models/ProLLaMA_Stage_1"), "protein", "prollama")
    second = STAGE.artefact_name(Path("/models/Llama-2-7b-hf"), "protein", "prollama")
    third = STAGE.artefact_name(Path("/models/ProLLaMA_Stage_1"), "text", "prollama")
    assert len({first, second, third}) == 3
    assert first == "concept_alignment__ProLLaMA_Stage_1__protein__prollama.json"


def test_every_pre_registered_decision_is_named_when_it_is_missing() -> None:
    args = STAGE.build_parser().parse_args(["--synthetic-check"])
    with pytest.raises(ValueError) as raised:
        STAGE.resolve(args)
    message = str(raised.value)
    for flag in STAGE.PRE_REGISTERED_FLAGS:
        assert f"--{flag.replace('_', '-')}" in message


def test_a_real_campaign_flag_beside_the_synthetic_check_is_refused() -> None:
    args = STAGE.build_parser().parse_args(
        ["--synthetic-check", "--mode", "protein", *FROZEN]
    )
    with pytest.raises(ValueError, match="meaningless beside"):
        STAGE.resolve(args)


def test_a_real_run_refuses_to_start_without_its_inputs() -> None:
    args = STAGE.build_parser().parse_args(FROZEN)
    with pytest.raises(ValueError, match="--checkpoint, --rendering, --mode, --cohort"):
        STAGE.resolve(args)


def test_an_excess_ratio_below_one_is_refused() -> None:
    args = STAGE.build_parser().parse_args(["--synthetic-check", *FROZEN[:-1], "0.5"])
    with pytest.raises(ValueError, match="excess ratio below one"):
        STAGE.resolve(args)
