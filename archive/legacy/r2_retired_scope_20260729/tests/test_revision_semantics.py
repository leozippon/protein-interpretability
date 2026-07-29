from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.revision.semantics import (
    benjamini_hochberg,
    blocked_fold_ids,
    low_level_design,
    run_conditional_semantics,
    within_protein_permutation,
)


RUNNER_SPEC = importlib.util.spec_from_file_location(
    "conditional_semantics_runner",
    Path(__file__).resolve().parents[1] / "scripts/53_run_conditional_semantics.py",
)
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
RUNNER_SPEC.loader.exec_module(RUNNER)


class RevisionSemanticsTest(unittest.TestCase):
    def test_v3_input_provenance_is_required_for_confirmatory_inputs(self):
        provenance = {
            "model_inference_dtype": "float32",
            "observed_model_parameter_dtypes": ["float32"],
            "model_inference_dtype_verification": (
                "all_floating_model_parameters_exactly_declared_before_first_activation"
            ),
            "model_inference_dtype_verified": True,
            "activation_finiteness_check": (
                "all_required_layer_captured_activation_and_logit_tensors_before_"
                "downstream_conversion_or_use"
            ),
            "activation_finiteness_verified": True,
        }
        RUNNER.validate_input_provenance(
            {
                "schema_version": RUNNER.P0_4_RUNNER_SCHEMA,
                "confirmatory": False,
                "input_provenance": provenance,
            }
        )
        RUNNER.validate_input_provenance({"confirmatory": False})
        with self.assertRaisesRegex(ValueError, "not v3 eligible"):
            RUNNER.validate_input_provenance(
                {
                    "schema_version": "r2_p0_4_conditional_semantics_spec_v2",
                    "confirmatory": False,
                    "input_provenance": provenance,
                }
            )
        with self.assertRaisesRegex(ValueError, "finite-inference provenance"):
            RUNNER.validate_input_provenance(
                {
                    "schema_version": RUNNER.P0_4_RUNNER_SCHEMA,
                    "confirmatory": True,
                    "input_provenance": provenance,
                }
            )
        provenance["model_inference_dtype"] = "bfloat16"
        provenance["observed_model_parameter_dtypes"] = ["bfloat16"]
        provenance["activation_finiteness_verified"] = False
        with self.assertRaisesRegex(ValueError, "finite-inference provenance"):
            RUNNER.validate_input_provenance(
                {
                    "schema_version": RUNNER.P0_4_RUNNER_SCHEMA,
                    "confirmatory": True,
                    "input_provenance": provenance,
                }
            )

    def test_blocked_folds_never_split_a_block(self):
        blocks = np.repeat([f"P{index}" for index in range(15)], np.arange(1, 16))
        folds = blocked_fold_ids(blocks, n_folds=5, seed=17)
        self.assertEqual(set(folds), set(range(5)))
        for block in set(blocks):
            self.assertEqual(len(set(folds[blocks == block])), 1)
        np.testing.assert_array_equal(folds, blocked_fold_ids(blocks, n_folds=5, seed=17))

    def test_low_level_design_is_deterministic_and_complete(self):
        n = 12
        kwargs = {
            "position": np.linspace(0, 1, n),
            "kmer": np.resize(np.array(["AAA", "CGT", "MKW"]), n),
            "input_norm": np.linspace(1, 2, n),
            "protein_length": np.full(n, 120),
            "sequence_source": np.resize(np.array(["real", "random"]), n),
            "position_degree": 2,
            "kmer_hash_buckets": 8,
        }
        first, names = low_level_design(**kwargs)
        second, second_names = low_level_design(**kwargs)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(names, second_names)
        self.assertIn("input_norm", names)
        self.assertIn("log_protein_length", names)
        self.assertEqual(first.shape, (n, 2 + 2 + 8 + 2))

    def test_within_protein_permutation_preserves_each_prevalence(self):
        labels = np.array([0, 1, 1, 0, 0, 1, 1, 1])
        proteins = np.array(["a"] * 4 + ["b"] * 4)
        permuted = within_protein_permutation(labels, proteins, np.random.default_rng(9))
        for protein in ("a", "b"):
            np.testing.assert_array_equal(
                np.sort(permuted[proteins == protein]), np.sort(labels[proteins == protein])
            )

    def test_bh_qvalues_are_monotone_in_pvalue_rank(self):
        pvalues = np.array([0.04, 0.001, 0.03, 0.8])
        qvalues = benjamini_hochberg(pvalues)
        order = np.argsort(pvalues)
        self.assertTrue(np.all(np.diff(qvalues[order]) >= 0))
        np.testing.assert_allclose(qvalues, [0.053333333333, 0.004, 0.053333333333, 0.8])

    def test_planted_label_adds_heldout_information_after_covariates(self):
        rng = np.random.default_rng(123)
        n_proteins, residues_per_protein = 30, 8
        n = n_proteins * residues_per_protein
        proteins = np.repeat([f"P{index:02d}" for index in range(n_proteins)], residues_per_protein)
        families = np.repeat(
            [f"F{index:02d}" for index in range(10)], 3 * residues_per_protein
        )
        position = np.tile(np.linspace(0, 1, residues_per_protein), n_proteins)
        labels = np.empty(n, dtype=int)
        for protein in range(n_proteins):
            current = np.array([0] * 4 + [1] * 4)
            labels[protein * residues_per_protein : (protein + 1) * residues_per_protein] = rng.permutation(current)
        covariates, _ = low_level_design(
            position=position,
            kmer=np.resize(np.array(["AAA", "CGT", "MKW", "DDD"]), n),
            input_norm=rng.normal(5, 0.2, n),
            protein_length=np.full(n, residues_per_protein),
            sequence_source=np.full(n, "synthetic"),
            position_degree=2,
            kmer_hash_buckets=8,
        )
        activation = (3.0 * labels + 0.4 * position + rng.normal(0, 0.15, n))[:, None]
        results = run_conditional_semantics(
            {"sparse": activation},
            {"planted": labels},
            covariates,
            proteins,
            families,
            feature_names={"sparse": ["feature_0"]},
            n_folds=5,
            n_permutations=39,
            n_bootstrap=80,
            seed=41,
        )
        self.assertEqual({result.blocking for result in results}, {"protein", "family"})
        self.assertTrue(all(result.delta_mse > 1.0 for result in results))
        self.assertTrue(all(result.delta_r2 > 0.8 for result in results))
        self.assertTrue(all(result.permutation_pvalue == 1 / 40 for result in results))
        self.assertTrue(all(result.qvalue <= 0.025 for result in results))
        self.assertTrue(all(result.bootstrap_delta_mse_ci95[0] > 1.0 for result in results))
        self.assertTrue(all(result.bootstrap_standard_error_delta_mse > 0 for result in results))
        self.assertTrue(
            all(result.retrospective_bootstrap_detectable_delta_mse > 0 for result in results)
        )
        self.assertTrue(
            all(result.prospective_minimum_detectable_delta_mse is None for result in results)
        )
        self.assertTrue(all(not result.permutation_degenerate for result in results))

    def test_prospective_mde_requires_complete_external_standard_errors(self):
        proteins = np.repeat([f"P{index}" for index in range(10)], 2)
        families = np.repeat([f"F{index}" for index in range(5)], 4)
        labels = np.resize([0, 1], 20)
        kwargs = {
            "representations": {"sparse": labels[:, None].astype(float)},
            "labels": {"label": labels},
            "covariates": np.ones((20, 1)),
            "protein_ids": proteins,
            "family_ids": families,
            "feature_names": {"sparse": ["feature_0"]},
            "n_folds": 5,
            "n_permutations": 1,
            "n_bootstrap": 3,
            "seed": 7,
        }
        with self.assertRaisesRegex(ValueError, "exactly cover all hypotheses"):
            run_conditional_semantics(
                **kwargs,
                prospective_standard_errors_delta_mse={
                    ("sparse", "feature_0", "label", "protein"): 0.25
                },
            )
        results = run_conditional_semantics(
            **kwargs,
            prospective_standard_errors_delta_mse={
                ("sparse", "feature_0", "label", "protein"): 0.25,
                ("sparse", "feature_0", "label", "family"): 0.50,
            },
        )
        mde = {result.blocking: result.prospective_minimum_detectable_delta_mse for result in results}
        self.assertIsNotNone(mde["protein"])
        self.assertIsNotNone(mde["family"])
        self.assertAlmostEqual(mde["family"] / mde["protein"], 2.0)

    def test_confirmatory_runner_requires_hash_bound_power_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec_path = root / "run.json"
            spec_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "require a SHA-bound"):
                RUNNER.load_prospective_power_plan(
                    {"confirmatory": True}, spec_path, {"sparse": ["f0"]}, {"label"}
                )

            plan = {
                "schema_version": 1,
                "independent_source": {
                    "description": "independent development cohort",
                    "standard_error_method": "protein-cluster bootstrap",
                    "run_manifest_sha256": "0" * 64,
                    "cohort_sha256": "1" * 64,
                    "independent_of_confirmatory_data": True,
                },
                "standard_errors_delta_mse": [
                    {
                        "representation": "sparse",
                        "feature": "f0",
                        "label": "label",
                        "blocking": blocking,
                        "standard_error_delta_mse": value,
                    }
                    for blocking, value in (("protein", 0.1), ("family", 0.2))
                ],
            }
            plan_path = root / "power.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
            values, metadata = RUNNER.load_prospective_power_plan(
                {
                    "confirmatory": True,
                    "prospective_power_plan": {"path": "power.json", "sha256": digest},
                },
                spec_path,
                {"sparse": ["f0"]},
                {"label"},
            )
            self.assertEqual(values[("sparse", "f0", "label", "protein")], 0.1)
            self.assertEqual(metadata["sha256"], digest)

    def test_confirmatory_dimensions_must_match(self):
        n = 20
        with self.assertRaisesRegex(ValueError, "matched dimensions"):
            run_conditional_semantics(
                {"sparse": np.ones((n, 2)), "dense": np.ones((n, 3))},
                {"label": np.resize([0, 1], n)},
                np.ones((n, 1)),
                np.repeat(np.arange(10), 2),
                np.repeat(np.arange(5), 4),
                n_folds=5,
                n_permutations=1,
                n_bootstrap=2,
            )

    def test_protein_constant_label_is_marked_as_degenerate(self):
        proteins = np.repeat([f"P{index}" for index in range(10)], 2)
        families = np.repeat([f"F{index}" for index in range(5)], 4)
        labels = np.repeat(np.resize([0, 1], 10), 2)
        activation = labels[:, None].astype(float)
        results = run_conditional_semantics(
            {"sparse": activation},
            {"protein_label": labels},
            np.ones((20, 1)),
            proteins,
            families,
            n_folds=5,
            n_permutations=3,
            n_bootstrap=5,
            seed=5,
        )
        self.assertTrue(all(result.permutation_degenerate for result in results))
        self.assertTrue(all(result.permutable_row_fraction == 0 for result in results))
        self.assertTrue(all(result.permutation_pvalue == 1 for result in results))


if __name__ == "__main__":
    unittest.main()
