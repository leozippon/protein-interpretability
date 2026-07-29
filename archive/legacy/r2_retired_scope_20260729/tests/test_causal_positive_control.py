from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.revision.causal_positive_control import (
    generate_grammar_cohort,
    run_planted_positive_control,
)
from src.revision.io import write_json
from src.revision.statistics import tost_paired


class CausalPositiveControlTest(unittest.TestCase):
    def test_grammar_has_disjoint_hashed_splits_and_all_planted_variables(self):
        records = generate_grammar_cohort(per_split=96, seed=17)
        discovery = {row.record_id for row in records if row.split == "discovery"}
        evaluation = {row.record_id for row in records if row.split == "evaluation"}
        self.assertFalse(discovery & evaluation)
        self.assertEqual(len({row.sequence for row in records}), len(records))
        self.assertTrue(all(len(row.sha256) == 64 for row in records))
        self.assertEqual({row.n_terminal for row in records}, {"A", "M"})
        self.assertEqual({row.long_range for row in records}, {False, True})
        self.assertEqual({row.motif_position for row in records}, {"early", "middle", "late"})

    def test_pipeline_recovers_and_controls_rotated_planted_path(self):
        result = run_planted_positive_control(
            generate_grammar_cohort(per_split=128, seed=23),
            model_seeds=[11, 29, 47],
            matched_control_count=3,
            equivalence_margin=0.10,
        )
        self.assertTrue(result["aggregate"]["analytic_synthetic_checks_passed"])
        self.assertEqual(result["aggregate"]["sensitivity"], 1.0)
        self.assertGreaterEqual(result["aggregate"]["specificity"], 0.95)
        for model in result["models"]:
            self.assertIn(model["planted_feature"], model["selected_features"])
            self.assertTrue(model["path_localized"])
            self.assertGreater(model["target_evaluation"]["ci95"][0], 0.0)
            self.assertTrue(
                all(row["equivalence"]["equivalent"] for row in model["control_evaluations"])
            )

    def test_feature_discovery_does_not_depend_on_evaluation_records(self):
        original = generate_grammar_cohort(per_split=96, seed=31)
        replacement = generate_grammar_cohort(per_split=96, seed=53)
        combined = [row for row in original if row.split == "discovery"] + [
            row for row in replacement if row.split == "evaluation"
        ]
        first = run_planted_positive_control(original, model_seeds=[13])
        second = run_planted_positive_control(combined, model_seeds=[13])
        self.assertEqual(
            first["models"][0]["selected_features"],
            second["models"][0]["selected_features"],
        )

    def test_tost_distinguishes_equivalence_from_a_meaningful_effect(self):
        self.assertTrue(tost_paired([0.0] * 20, margin=0.10)["equivalent"])
        self.assertFalse(tost_paired([0.25] * 20, margin=0.10)["equivalent"])

    def test_strict_writer_rejects_nonfinite_values(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "strict.json"
            with self.assertRaises(ValueError):
                write_json(path, {"invalid": float("nan")})
            write_json(path, {"valid": None})
            self.assertEqual(json.loads(path.read_text()), {"valid": None})


if __name__ == "__main__":
    unittest.main()
