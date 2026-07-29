from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "11_steering_benchmark.py"
SPEC = importlib.util.spec_from_file_location("steering_benchmark_11", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SteeringSelectionTest(unittest.TestCase):
    def setUp(self):
        self.direct_effect = {
            "per_ec": {
                "target": {
                    "layers": {
                        "3": [
                            {"feature": 1, "direct_effect": 0.4},
                            {"feature": 2, "direct_effect": -0.3},
                            {"feature": 3, "direct_effect": 0.2},
                        ]
                    }
                }
            }
        }

    def test_positive_only_selection(self):
        selected = MODULE.pick_direct_effect_features(
            self.direct_effect, "target", 3, k=2
        )
        self.assertEqual(selected, [(1, 0.4), (3, 0.2)])

    def test_insufficient_positive_candidates_fail(self):
        with self.assertRaisesRegex(ValueError, "Refusing opposite-sign fallback"):
            MODULE.pick_direct_effect_features(
                self.direct_effect, "target", 3, k=3
            )

    def test_paired_statistics_use_pair_differences(self):
        control = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        target = control + 0.25
        result = MODULE.paired_bootstrap_diff(target, control, n_boot=2000)
        self.assertAlmostEqual(result["obs_diff"], 0.25, places=6)
        self.assertAlmostEqual(result["ci95"][0], 0.25, places=6)
        self.assertAlmostEqual(result["ci95"][1], 0.25, places=6)
        self.assertGreater(result["paired_sign_randomization_p"], 0.05)


if __name__ == "__main__":
    unittest.main()
