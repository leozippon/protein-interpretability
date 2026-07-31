from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.arms import Arm, ArmSpec, Cohort, PANEL  # noqa: E402
from src.transfer.circuits import content_bounds  # noqa: E402
from src.transfer.induction_robustness import contrast_ratio_bootstrap  # noqa: E402
from src.transfer.lenses import split_cohort  # noqa: E402
from src.transfer.path_patching import (  # noqa: E402
    attention_output_projection,
    causal_census_agreement,
    require_supported_layout,
    select_senders,
)
from src.transfer.prediction_addressed import unigram_percentiles  # noqa: E402
from src.transfer.probes import skill_block  # noqa: E402
from src.transfer.statistics import paired_group_bootstrap  # noqa: E402


class _ProGenAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.out_proj = nn.Linear(width, width, bias=False)


class _ProGenBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.attn = _ProGenAttention(width)
        self.mlp = nn.Linear(width, width)


class _ProGenTransformer(nn.Module):
    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.h = nn.ModuleList([_ProGenBlock(width) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(width)


class _ProGenModel(nn.Module):
    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.transformer = _ProGenTransformer(width, layers)
        self.lm_head = nn.Linear(width, 7, bias=False)


def _progen_arm() -> Arm:
    spec = ArmSpec(
        name="progen-test",
        path=PANEL["progen2-medium"].path,
        modality="protein",
        n_layer=2,
        d_model=4,
        tokenisation="residue",
        input_format="n_to_c_control",
        evaluation_cohort_source="test",
        architecture="progen",
        path_variable=PANEL["progen2-medium"].path_variable,
    )
    return Arm(
        spec=spec,
        model=_ProGenModel(width=4, layers=2),
        tokenizer=object(),
        device="cpu",
        dtype="float32",
    )


def _mean_metric(_truth: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(predictions))


def test_arm_architecture_is_required_and_progen2_is_declared_progen() -> None:
    assert (
        inspect.signature(ArmSpec).parameters["architecture"].default
        is inspect.Parameter.empty
    )
    assert PANEL["progen2-base"].architecture == "progen"
    assert PANEL["progen2-medium"].architecture == "progen"


def test_path_patching_resolves_the_declared_progen_layout() -> None:
    arm = _progen_arm()
    require_supported_layout(arm)
    assert (
        attention_output_projection(arm, 0) is arm.model.transformer.h[0].attn.out_proj
    )


def test_skill_interval_bootstraps_the_chance_corrected_statistic() -> None:
    truth = np.zeros(10)
    probe = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50])
    chance = np.array([0.80, 0.10, 0.70, 0.20, 0.60, 0.30, 0.50, 0.40, 0.20, 0.10])
    groups = np.arange(10)
    seed = 17
    n_bootstrap = 1000

    result = skill_block(
        truth,
        probe,
        chance,
        groups,
        _mean_metric,
        seed=seed,
        n_bootstrap=n_bootstrap,
    )

    generator = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        sampled = generator.choice(groups, size=groups.size, replace=True)
        probe_score = float(probe[sampled].mean())
        chance_score = float(chance[sampled].mean())
        draws.append((probe_score - chance_score) / (1.0 - chance_score))
    expected = np.percentile(draws, [2.5, 97.5])

    assert result["skill_ci95"] == pytest.approx(expected)
    assert result["skill"] == pytest.approx(
        (probe.mean() - chance.mean()) / (1.0 - chance.mean())
    )


def test_ratio_interval_is_absent_below_the_finite_draw_floor() -> None:
    truth = np.zeros(10)
    left = np.ones(10)
    right = np.array([1.0] + [0.0] * 9)
    result = paired_group_bootstrap(
        truth,
        left,
        right,
        np.arange(10),
        _mean_metric,
        seed=0,
        n_bootstrap=1000,
    )

    assert result["ratio"] == pytest.approx(10.0)
    assert result["n_ratio_draws"] < 950
    assert result["ratio_ci95"] is None


def test_induction_ratio_interval_requires_95_percent_defined_draws() -> None:
    result = contrast_ratio_bootstrap(
        np.ones((10, 1, 1)),
        np.array([1.0] * 5 + [0.0] * 5).reshape(10, 1, 1),
        threshold=0.5,
        resamples=1000,
        seed=0,
    )

    assert result["ratio"] == pytest.approx(1.0)
    assert result["finite_resamples"] < result["minimum_finite_resamples"]
    assert result["interval"] is None
    assert result["interval_status"].startswith("undefined_denominator")


def test_unigram_percentiles_assign_one_value_to_each_frequency_tie() -> None:
    assert unigram_percentiles(np.array([1, 1, 2])).tolist() == pytest.approx(
        [0.5, 0.5, 1.0]
    )
    assert unigram_percentiles(np.array([0, 0, 2])).tolist() == pytest.approx(
        [0.0, 0.0, 1.0]
    )


class _FastaTokenizer:
    eos_token_id = 0

    @staticmethod
    def decode(token_ids: list[int]) -> str:
        return {0: "<eot>", 1: "\n", 2: "\r\n", 3: "AC"}[token_ids[0]]


def _fasta_arm() -> SimpleNamespace:
    return SimpleNamespace(
        name="protgpt2",
        spec=SimpleNamespace(input_format="fasta_wrapped"),
        tokenizer=_FastaTokenizer(),
    )


def test_content_bounds_exclude_the_complete_fasta_prefix() -> None:
    assert content_bounds(_fasta_arm(), [0, 1, 2, 3], 4) == (3, 4)


def test_content_bounds_refuse_a_malformed_fasta_prefix() -> None:
    with pytest.raises(ValueError, match="does not start with end-of-text"):
        content_bounds(_fasta_arm(), [3, 1, 3], 3)
    with pytest.raises(ValueError, match="has no line break"):
        content_bounds(_fasta_arm(), [0, 3], 2)


def test_split_cohort_records_parent_provenance_and_split_indices() -> None:
    cohort = Cohort(
        name="parent",
        kind="protein",
        records=["AAAA", "CCCC", "DDDD", "EEEE"],
        min_symbols=1,
        max_symbols=10,
        metadata={
            "sampling": {"mode": "seeded", "seed": 7},
            "source_version": "test-v1",
            "ec_labels": ["1", "2", "3", "4"],
        },
    )
    train, evaluation = split_cohort(cohort, train_fraction=0.5, seed=11)

    all_indices = []
    for role, child in (("train", train), ("eval", evaluation)):
        sampling = child.sampling
        assert sampling["mode"] == "split"
        assert sampling["role"] == role
        assert sampling["seed"] == 11
        assert sampling["parent_digest"] == cohort.digest
        assert sampling["parent_provenance_digest"] == cohort.provenance_digest
        assert sampling["parent_sampling"] == cohort.sampling
        indices = sampling["indices"]
        assert child.records == [cohort.records[index] for index in indices]
        assert child.metadata["ec_labels"] == [
            cohort.metadata["ec_labels"][index] for index in indices
        ]
        all_indices.extend(indices)

    assert sorted(all_indices) == list(range(len(cohort)))
    assert set(train.sampling["indices"]).isdisjoint(evaluation.sampling["indices"])


# --------------------------------------------------------- exhaustive senders
#
# Audit item D2.b asked for a top-20 Jaccard between the causal ranking and the
# prefix-matching census and got 1.0 on every arm, because the sender set was the
# census-selected set: both rankings ranked the same heads, so no head the census
# rejected could ever appear. The failure is invisible in the output -- a real
# agreement would give 1.0 too -- so these tests pin the precondition rather than
# the number.


def _scores() -> np.ndarray:
    return np.array([[0.30, 0.02, 0.11], [0.15, 0.01, 0.09]], dtype=np.float64)


def test_exhaustive_selection_admits_heads_the_census_rejects() -> None:
    selective, selective_provenance = select_senders(
        _scores(), threshold=0.10, fallback_top_k=2
    )
    exhaustive, exhaustive_provenance = select_senders(
        _scores(), threshold=0.10, fallback_top_k=2, exhaustive=True
    )
    assert selective_provenance["criterion"] == "prefix_matching_above_threshold"
    assert exhaustive_provenance["criterion"] == "exhaustive_all_heads"
    assert len(exhaustive) == _scores().size
    # The whole point: the exhaustive set spans both sides of the threshold.
    assert exhaustive_provenance["n_senders_below_threshold"] > 0
    assert selective_provenance["n_senders_below_threshold"] == 0
    assert {s.label for s in selective} < {s.label for s in exhaustive}


def test_above_threshold_is_each_head_s_own_score_not_the_set_s() -> None:
    """The per-head flag must not be the set-level answer broadcast to every head.

    On the two selective criteria the two agree, so this is a no-op there; on the
    exhaustive set only the per-head answer is true, and it is the field that says
    which causally-ranked heads the census would have missed.
    """

    exhaustive, _ = select_senders(
        _scores(), threshold=0.10, fallback_top_k=2, exhaustive=True
    )
    for sender in exhaustive:
        assert sender.above_threshold == (sender.prefix_matching >= 0.10)
    assert len({s.above_threshold for s in exhaustive}) == 2

    # Unchanged on the paths that existed before.
    for threshold, expected in ((0.10, True), (0.99, False)):
        senders, _ = select_senders(_scores(), threshold=threshold, fallback_top_k=2)
        assert {s.above_threshold for s in senders} == {expected}


#: The head grid ``_scores`` describes. The agreement statistic's precondition is
#: that every one of these heads carries a causal effect.
_GRID = 6


def _rows(effects: list[float]) -> list[dict[str, object]]:
    senders, _ = select_senders(_scores(), threshold=0.10, fallback_top_k=2, exhaustive=True)
    return [
        {
            "label": s.label,
            "prefix_matching": s.prefix_matching,
            "effects": {"total": e},
            # A denominator of 10 logits, so the two scales are distinguishable and
            # a test cannot pass by reading the wrong one.
            "effects_logits": {"total": e * 10.0},
        }
        for s, e in zip(senders, effects, strict=True)
    ]


def test_selection_refuses_to_truncate_an_exhaustive_set() -> None:
    """``--exhaustive-senders --max-senders 20`` rebuilt the D2.b artefact exactly.

    Truncation keeps the highest-scoring heads, so the surviving set is
    census-selected while still carrying the exhaustive label: every head is above
    threshold, ``n_senders_below_threshold`` is 0, and the agreement statistic
    computed on it is circular again with nothing in the record saying so.
    """

    with pytest.raises(ValueError, match="mutually exclusive"):
        select_senders(
            _scores(), threshold=0.10, fallback_top_k=2, max_senders=2, exhaustive=True
        )
    # Truncation of a *selective* set is untouched: it never claimed to be a grid.
    _, provenance = select_senders(
        _scores(), threshold=0.10, fallback_top_k=2, max_senders=1
    )
    assert provenance["truncated_to_max_senders"] is True


def test_agreement_refuses_a_sender_set_smaller_than_the_head_grid() -> None:
    """The negative path is the defect: a selective set yields 1.0 and means nothing.

    The precondition is checked against the size of the head grid, which
    ``select_senders`` publishes, rather than against a boolean the caller
    supplies -- a boolean is only as true as the arguments beside it, and
    ``max_senders`` used to falsify it silently.
    """

    rows = _rows([0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    with pytest.raises(ValueError, match="every head in the grid"):
        causal_census_agreement(rows[:4], threshold=0.10, n_heads_in_grid=_GRID)
    _, provenance = select_senders(
        _scores(), threshold=0.10, fallback_top_k=2, exhaustive=True
    )
    assert provenance["n_heads_in_grid"] == _GRID


def test_agreement_refuses_per_head_records_without_the_logit_scale() -> None:
    """A recovery-scale magnitude is not comparable across arms; the logit one is."""

    rows = _rows([0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    for row in rows:
        del row["effects_logits"]
    with pytest.raises(ValueError, match="effects_logits"):
        causal_census_agreement(rows, threshold=0.10, n_heads_in_grid=_GRID)


def test_agreement_surfaces_a_head_the_census_ranks_below_threshold() -> None:
    # Head ranked last by the census carries the largest causal effect.
    rows = _rows([0.01, 0.02, 0.03, 0.04, 0.05, 0.90])
    lowest = rows[-1]["label"]
    report = causal_census_agreement(rows, threshold=0.10, n_heads_in_grid=_GRID)

    assert report["n_below_census_threshold"] == 3
    missed = report["strongest_head_below_census_threshold"]
    assert missed["label"] == lowest
    assert missed["causal_rank"] == 0
    # Both scales, so a cross-arm quote cannot be built on the normalised one
    # without the denominator that produced it.
    assert missed["effect_total"] == pytest.approx(0.90)
    assert missed["effect_total_logits"] == pytest.approx(9.0)
    top_5 = report["top_k"][repr(5)]
    assert lowest in top_5["causal_top_heads"]
    assert lowest in top_5["causal_top_heads_below_census_threshold"]
    # A census miss must cost agreement; the old circular statistic could not.
    assert top_5["jaccard"] < 1.0


def test_agreement_reports_a_swept_k_and_a_threshold_free_primary() -> None:
    rows = _rows([0.60, 0.50, 0.40, 0.30, 0.20, 0.10])  # causal order == census order
    report = causal_census_agreement(rows, threshold=0.10, n_heads_in_grid=_GRID)

    # Standing rule 8: the primary statistic needs no cut.
    assert report["spearman_census_vs_causal_magnitude"]["rho"] == pytest.approx(1.0)
    # 32 is in the ladder because EXP-R2-071 quoted its rank split at that cut.
    assert [c["k"] for c in report["top_k"].values()] == [5, 10, 20, 32, 40]
    for cut in report["top_k"].values():
        assert cut["jaccard"] == pytest.approx(1.0)
        assert cut["k_effective"] <= len(rows)
        assert cut["truncated_to_head_count"] == (cut["k"] > len(rows))


def test_the_rank_split_is_reported_at_every_cut_and_withheld_when_undefined() -> None:
    """The retraction of EXP-R2-071's first reading turns on this split.

    An all-grid rho near zero is compatible with the census ordering the top of
    the grid well and the bulk not at all, which is what the protein arms turned
    out to do -- so the split is a published statistic, not a diagnostic run once
    in a script that no longer exists.
    """

    # Census rank and causal rank agree on the top three and reverse below them.
    rows = _rows([0.60, 0.50, 0.40, 0.01, 0.02, 0.03])
    report = causal_census_agreement(rows, threshold=0.10, n_heads_in_grid=_GRID)

    at_five = report["top_k"][repr(5)]
    assert at_five["spearman_top_k"]["n"] == 5
    assert at_five["n_remaining"] == 1
    # One head left over: no ranking exists, and none is invented.
    assert at_five["spearman_remaining"]["rho"] is None
    assert "fewer than three heads" in at_five["spearman_remaining"]["withheld_reason"]

    # At a cut the grid can support, both halves are real and they disagree.
    at_three = causal_census_agreement(
        rows, threshold=0.10, n_heads_in_grid=_GRID, top_k=(3,)
    )["top_k"][repr(3)]
    assert at_three["spearman_top_k"]["rho"] == pytest.approx(1.0)
    assert at_three["spearman_remaining"]["rho"] == pytest.approx(-1.0)
    assert report["rank_split"]["split_by"] == "census_rank"


def test_agreement_rejects_duplicate_head_labels() -> None:
    rows = _rows([0.1] * 6)
    rows[1]["label"] = rows[0]["label"]
    with pytest.raises(ValueError, match="duplicate head label"):
        causal_census_agreement(rows, threshold=0.10, n_heads_in_grid=_GRID)
