from __future__ import annotations

import unittest

import numpy as np

from src.revision.nested_recoverability import (
    _control_transform,
    make_group_splits,
    run_nested_recoverability,
    synthetic_recoverability_fixture,
)


class NestedRecoverabilityTest(unittest.TestCase):
    def test_group_splits_are_disjoint_and_cover_each_sample_once(self):
        groups = np.repeat(np.arange(18), 3)
        y = np.repeat(np.arange(18) % 2, 3)
        splits = make_group_splits(
            y, groups, n_splits=3, seed=19, task_type="classification"
        )
        counts = np.zeros(y.size, dtype=int)
        for train, test in splits:
            self.assertFalse(set(groups[train]) & set(groups[test]))
            counts[test] += 1
        np.testing.assert_array_equal(counts, np.ones_like(counts))

    def test_nested_protocol_reuses_folds_and_selects_inside_outer_train(self):
        representations, y, groups, reconstruction, intervention = (
            synthetic_recoverability_fixture(
                seed=37, n_groups=24, samples_per_group=3, dictionary_seeds=(0, 1)
            )
        )
        result = run_nested_recoverability(
            representations,
            y,
            groups,
            ceiling_name="ceiling",
            floor_names=["code_seed_0", "code_seed_1"],
            task_type="classification",
            analysis_seeds=[5, 17],
            outer_splits=3,
            inner_splits=2,
            control_methods=["pca", "random_projection", "random_dictionary"],
            comparison_dimension=4,
            active_width_dimension=3,
            n_bootstrap=50,
            reconstruction_error_by_floor_layer=reconstruction,
            intervention_effect_by_floor_layer=intervention,
        )
        self.assertEqual(len(result["fold_manifest"]), 6)
        for fold in result["fold_manifest"]:
            self.assertFalse(set(fold["train_groups"]) & set(fold["test_groups"]))
            self.assertEqual(len(fold["inner_folds"]), 2)
            for inner in fold["inner_folds"]:
                self.assertFalse(set(inner["train_groups"]) & set(inner["test_groups"]))
        expected_rows = 6 * (len(representations) + 3) * 2
        self.assertEqual(len(result["fold_results"]), expected_rows)
        for seed_predictions in result["outer_predictions"].values():
            self.assertEqual(
                set(seed_predictions),
                {"primary_common_dimension", "active_width_rank_sensitivity"},
            )
            for track_predictions in seed_predictions.values():
                self.assertTrue(
                    all(len(values) == len(y) for values in track_predictions.values())
                )
        self.assertEqual(len(result["paired_comparisons"]), 2 * (1 + 3) * 2)
        for comparison in result["paired_comparisons"]:
            self.assertEqual(len(comparison["seed_results"]), 2)
            self.assertTrue(
                all(
                    len(seed_result["difference_ci95"]) == 2
                    for seed_result in comparison["seed_results"]
                )
            )
        self.assertEqual(
            {row["reference"] for row in result["paired_comparisons"]},
            {"ceiling", "pca", "random_projection", "random_dictionary"},
        )
        dimensions = {
            row["dimension_track"]: row["probe_input_dimension"]
            for row in result["fold_results"]
        }
        self.assertEqual(
            dimensions,
            {"primary_common_dimension": 4, "active_width_rank_sensitivity": 3},
        )
        self.assertTrue(
            result["dimensional_matching"][
                "all_arm_probe_inputs_exact_declared_dimension"
            ]
        )
        self.assertTrue(
            all(
                row["track_role"] == "confirmatory_primary"
                for row in result["paired_comparisons"]
                if row["dimension_track"] == "primary_common_dimension"
            )
        )
        self.assertTrue(
            all(
                row["track_role"] == "sensitivity_only"
                for row in result["paired_comparisons"]
                if row["dimension_track"] == "active_width_rank_sensitivity"
            )
        )
        self.assertTrue(result["quality_relationships"])
        self.assertEqual(
            {
                (row["representation"], row["quality_layer"])
                for row in result["quality_relationships"]
            },
            {
                (f"code_seed_{seed}", str(layer))
                for seed in (0, 1)
                for layer in (0, 1, 2)
            },
        )
        self.assertEqual(
            {
                row["relationship"] for row in result["quality_relationships"]
            },
            {
                "reconstruction_error_vs_probe_error",
                "intervention_effect_vs_probe_error",
                "reconstruction_error_vs_intervention_effect",
            },
        )

    def test_confirmatory_real_requires_both_quality_relationship_inputs(self):
        representations, y, groups, _, _ = synthetic_recoverability_fixture(
            seed=39, n_groups=12, samples_per_group=2, dictionary_seeds=(0,)
        )
        with self.assertRaisesRegex(
            ValueError, "requires seed/layer-specific reconstruction"
        ):
            run_nested_recoverability(
                representations,
                y,
                groups,
                ceiling_name="ceiling",
                floor_names=["code_seed_0"],
                task_type="classification",
                analysis_seeds=[5],
                outer_splits=2,
                inner_splits=2,
                control_methods=[],
                comparison_dimension=4,
                n_bootstrap=5,
                confirmatory_real=True,
            )

    def test_quality_inventory_cannot_drop_or_average_a_seed_layer(self):
        representations, y, groups, reconstruction, intervention = (
            synthetic_recoverability_fixture(
                seed=41, n_groups=12, samples_per_group=2, dictionary_seeds=(0,)
            )
        )
        reconstruction["code_seed_0"].pop(2)
        with self.assertRaisesRegex(ValueError, "exactly every representation layer"):
            run_nested_recoverability(
                representations,
                y,
                groups,
                ceiling_name="ceiling",
                floor_names=["code_seed_0"],
                task_type="classification",
                analysis_seeds=[5],
                outer_splits=2,
                inner_splits=2,
                control_methods=[],
                comparison_dimension=3,
                n_bootstrap=5,
                reconstruction_error_by_floor_layer=reconstruction,
                intervention_effect_by_floor_layer=intervention,
                confirmatory_real=True,
            )

    def test_confirmatory_real_rejects_the_legacy_small_cohort(self):
        representations, y, groups, reconstruction, intervention = (
            synthetic_recoverability_fixture(
                seed=44, n_groups=12, samples_per_group=2, dictionary_seeds=(0,)
            )
        )
        with self.assertRaisesRegex(ValueError, "enlarged-cohort minimum of 480"):
            run_nested_recoverability(
                representations,
                y,
                groups,
                ceiling_name="ceiling",
                floor_names=["code_seed_0"],
                task_type="classification",
                analysis_seeds=[5],
                outer_splits=2,
                inner_splits=2,
                control_methods=[],
                comparison_dimension=3,
                n_bootstrap=5,
                reconstruction_error_by_floor_layer=reconstruction,
                intervention_effect_by_floor_layer=intervention,
                confirmatory_real=True,
            )

    def test_synthetic_plumbing_does_not_require_quality_relationship_inputs(self):
        representations, y, groups, _, _ = synthetic_recoverability_fixture(
            seed=40, n_groups=12, samples_per_group=2, dictionary_seeds=(0,)
        )
        result = run_nested_recoverability(
            representations,
            y,
            groups,
            ceiling_name="ceiling",
            floor_names=["code_seed_0"],
            task_type="classification",
            analysis_seeds=[5],
            outer_splits=2,
            inner_splits=2,
            control_methods=[],
            comparison_dimension=3,
            n_bootstrap=5,
        )
        self.assertFalse(result["confirmatory_real"])
        self.assertEqual(result["quality_relationships"], [])

    def test_common_dimension_is_fail_closed_for_every_arm(self):
        representations, y, groups, _, _ = synthetic_recoverability_fixture(
            seed=42, n_groups=12, samples_per_group=2, dictionary_seeds=(0,)
        )
        with self.assertRaisesRegex(ValueError, "largest declared common dimension"):
            run_nested_recoverability(
                representations,
                y,
                groups,
                ceiling_name="ceiling",
                floor_names=["code_seed_0"],
                task_type="classification",
                analysis_seeds=[5],
                outer_splits=2,
                inner_splits=2,
                control_methods=[],
                comparison_dimension=6,
                n_bootstrap=5,
            )

    def test_nmf_and_ica_controls_are_train_fit_and_matched_dimension(self):
        rng = np.random.default_rng(41)
        train_sources = rng.laplace(size=(300, 5))
        test_sources = rng.laplace(size=(100, 5))
        mixing = rng.normal(size=(5, 10))
        train = train_sources @ mixing
        test = test_sources @ mixing
        for method in ("nmf", "ica"):
            with self.subTest(method=method):
                metadata = {}
                transformed_train, transformed_test = _control_transform(
                    method, train, test, dimension=5, seed=7, metadata=metadata
                )
                self.assertEqual(transformed_train.shape, (300, 5))
                self.assertEqual(transformed_test.shape, (100, 5))
                self.assertTrue(np.isfinite(transformed_train).all())
                self.assertTrue(np.isfinite(transformed_test).all())
                self.assertEqual(metadata["method"], method)

    def test_control_dimension_is_never_silently_capped(self):
        rng = np.random.default_rng(43)
        train = rng.normal(size=(20, 6))
        test = rng.normal(size=(5, 6))
        with self.assertRaisesRegex(ValueError, "silently unmatched"):
            _control_transform("pca", train, test, dimension=7, seed=11, metadata={})


if __name__ == "__main__":
    unittest.main()
