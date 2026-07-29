from __future__ import annotations

import unittest

import numpy as np

from src.revision.atlas import (
    apply_model_permutations,
    coherent_permutation_test,
    discover_atlas,
    draw_model_permutations,
    empirical_pvalue,
    greedy_match,
    hungarian_match,
    identity_overlap,
    score_atlas,
)


def planted_cohort(seed: int, n_samples: int) -> dict[str, dict[int, np.ndarray]]:
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=(n_samples, 3))
    layouts = {
        "alpha": (0, 1, 2),
        "beta": (2, 0, 1),
        "gamma": (1, 2, 0),
    }
    result = {}
    for model_index, (model, layout) in enumerate(layouts.items()):
        noise = rng.normal(scale=0.025, size=(n_samples, 3))
        nuisance = rng.normal(scale=0.3, size=(n_samples, 2))
        features = np.column_stack([shared[:, factor] for factor in layout]) + noise
        result[model] = {
            0: np.column_stack((features, nuisance)),
            1: np.column_stack((features + model_index * 0.001, nuisance)),
        }
    return result


class RevisionAtlasTest(unittest.TestCase):
    def test_planted_factors_replicate_on_heldout_cohort_for_every_matcher(self):
        discovery = planted_cohort(11, 240)
        heldout = planted_cohort(29, 180)
        planted = {
            (("alpha", 0, 0), ("beta", 0, 1), ("gamma", 0, 2)),
            (("alpha", 0, 1), ("beta", 0, 2), ("gamma", 0, 0)),
            (("alpha", 0, 2), ("beta", 0, 0), ("gamma", 0, 1)),
        }
        for matcher in ("greedy", "hungarian", "optimal_transport", "joint_triangle"):
            with self.subTest(matcher=matcher):
                atlas = discover_atlas(
                    discovery,
                    layer_groups={"anchor": {model: 0 for model in discovery}},
                    feature_pool_size=5,
                    matcher=matcher,
                    threshold=0.95,
                )
                identities = {match.identity for match in atlas.matches}
                self.assertTrue(planted.issubset(identities))
                evaluation = score_atlas(atlas, heldout)
                planted_scores = [
                    match for match in evaluation.matches if match.identity in planted
                ]
                self.assertEqual(len(planted_scores), 3)
                self.assertTrue(all(match.passes_threshold for match in planted_scores))
                self.assertTrue(
                    all(min(match.signed_correlations) > 0.99 for match in planted_scores)
                )

    def test_positive_and_absolute_modes_do_not_conflate_anticorrelation(self):
        rng = np.random.default_rng(7)
        factor = rng.normal(size=128)
        cohort = {
            "left": {0: factor[:, None]},
            "right": {0: (-factor)[:, None]},
        }
        positive = discover_atlas(cohort, matcher="greedy", threshold=0.9)
        absolute = discover_atlas(
            cohort, matcher="greedy", correlation_mode="absolute", threshold=0.9
        )
        self.assertEqual(len(positive.matches), 0)
        self.assertEqual(len(absolute.matches), 1)
        self.assertAlmostEqual(
            absolute.matches[0].discovery_signed_correlations[0], -1.0, places=12
        )

    def test_heldout_scoring_never_rediscovers_feature_identity(self):
        atlas = discover_atlas(
            planted_cohort(3, 160),
            layer_groups={"anchor": {"alpha": 0, "beta": 0, "gamma": 0}},
            matcher="hungarian",
            threshold=0.95,
        )
        before = tuple(match.identity for match in atlas.matches)
        evaluation = score_atlas(atlas, planted_cohort(5, 140))
        self.assertEqual(before, tuple(match.identity for match in evaluation.matches))
        self.assertEqual(evaluation.n_passing, len(atlas.matches))
        self.assertEqual(evaluation.retained_identity_jaccard, 1.0)
        overlap = identity_overlap(atlas, evaluation)
        self.assertEqual(overlap.jaccard, 1.0)
        self.assertEqual(overlap.n_intersection, len(atlas.matches))

    def test_one_model_permutation_is_reused_at_every_layer(self):
        matrices = planted_cohort(19, 24)
        permutations = draw_model_permutations(
            matrices, np.random.default_rng(1234)
        )
        shuffled = apply_model_permutations(matrices, permutations)
        for model, permutation in permutations.items():
            np.testing.assert_array_equal(
                shuffled[model][0], matrices[model][0][permutation]
            )
            np.testing.assert_array_equal(
                shuffled[model][1], matrices[model][1][permutation]
            )

    def test_coherent_null_and_plus_one_pvalues(self):
        matrices = planted_cohort(23, 100)
        atlas = discover_atlas(
            matrices,
            matcher="joint_triangle",
            threshold=0.95,
            feature_pool_size=5,
        )
        result = coherent_permutation_test(
            atlas,
            matrices,
            n_permutations=19,
            seed=91,
            return_permutations=True,
        )
        self.assertEqual(len(result.null_counts), 19)
        self.assertEqual(len(result.permutations), 19)
        self.assertGreaterEqual(result.count_pvalue, 1 / 20)
        self.assertEqual(
            result.count_pvalue,
            (1 + sum(x >= result.observed_count for x in result.null_counts)) / 20,
        )
        self.assertEqual(empirical_pvalue(2, [0, 1, 2]), 0.5)

    def test_match_diagnostics_expose_ties(self):
        matches = greedy_match(np.array([[1.0, 1.0], [0.0, 0.9]]), max_matches=1)
        self.assertEqual(matches[0].ambiguity, 1)
        self.assertEqual(matches[0].confidence, 0.0)

    def test_hungarian_threshold_uses_explicit_unmatched_endpoints(self):
        scores = np.array([[0.90, 0.80], [0.85, -1.0]])
        matches = hungarian_match(scores, min_score=0.82)
        self.assertEqual(
            [(match.left, match.right, match.score) for match in matches],
            [(0, 0, 0.90)],
        )


if __name__ == "__main__":
    unittest.main()
