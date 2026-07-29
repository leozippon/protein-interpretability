from __future__ import annotations

import unittest

from src.revision.causal_positive_control import generate_grammar_cohort
from src.revision.trained_positive_control import (
    ControlConfig,
    TOKEN_TO_ID,
    encode_cohort,
    run_trained_positive_control,
)


class TrainedPositiveControlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = generate_grammar_cohort(
            per_split=24,
            seed=20260717,
            splits=("train", "discovery", "evaluation"),
            min_length=24,
            max_length=28,
        )

    def test_encoding_is_causal_and_split_records_are_immutable(self):
        cohort = encode_cohort([row for row in self.records if row.split == "discovery"])
        rows = range(cohort.input_ids.shape[0])
        self.assertTrue(
            all(
                cohort.input_ids[row, cohort.predictor_positions[row]]
                == TOKEN_TO_ID["<query>"]
                for row in rows
            )
        )
        split_hashes = {
            split: {row.sha256 for row in self.records if row.split == split}
            for split in ("train", "discovery", "evaluation")
        }
        self.assertFalse(split_hashes["train"] & split_hashes["discovery"])
        self.assertFalse(split_hashes["train"] & split_hashes["evaluation"])
        self.assertFalse(split_hashes["discovery"] & split_hashes["evaluation"])

    def test_three_trained_models_recover_the_heldout_planted_path(self):
        result = run_trained_positive_control(
            self.records,
            model_seeds=[11, 29, 47],
            config=ControlConfig(),
            device="cpu",
        )
        self.assertTrue(result["aggregate"]["posthoc_development_smoke_passed"])
        self.assertEqual(result["aggregate"]["sensitivity"], 1.0)
        self.assertEqual(result["aggregate"]["specificity"], 1.0)
        self.assertEqual(result["aggregate"]["false_discovery_rate"], 0.0)
        self.assertEqual(len(set(result["split_sha256"].values())), 3)
        self.assertEqual(len(result["cross_model_alignment"]), 3)
        for alignment in result["cross_model_alignment"]:
            self.assertGreaterEqual(
                alignment["heldout_confound_adjusted_activation_spearman"], 0.8
            )
        for model in result["models"]:
            self.assertTrue(model["lm_training"]["loss_improved"])
            self.assertLess(model["clt_training"]["fvu_mean"], 0.5)
            self.assertLess(model["clt_training"]["fvu_per_layer"][1], 0.5)
            self.assertGreaterEqual(model["lm_training"]["evaluation_endpoint_accuracy"], 0.8)
            self.assertTrue(model["path_localized"])
            self.assertGreaterEqual(model["selected_ground_truth_cosine"], 0.7)
            self.assertGreater(
                model["semantic_specificity"][
                    "incremental_r2_beyond_position_length_kmer"
                ],
                0.0,
            )
            self.assertTrue(
                all(row["equivalence"]["equivalent"] for row in model["matched_controls"])
            )
            self.assertTrue(
                all(0.8 <= row["effect_recovery_ratio"] <= 1.2 for row in model["dose_sweep"])
            )
            self.assertEqual(len(model["model_state_sha256"]), 64)
            self.assertEqual(len(model["clt_state_sha256"]), 64)

    def test_at_least_three_unique_model_seeds_are_required(self):
        with self.assertRaisesRegex(ValueError, "three unique"):
            run_trained_positive_control(self.records, model_seeds=[11, 29])
        with self.assertRaisesRegex(ValueError, "three unique"):
            run_trained_positive_control(self.records, model_seeds=[11, 11, 29])


if __name__ == "__main__":
    unittest.main()
