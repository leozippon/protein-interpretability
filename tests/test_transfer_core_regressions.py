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
    require_supported_layout,
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
