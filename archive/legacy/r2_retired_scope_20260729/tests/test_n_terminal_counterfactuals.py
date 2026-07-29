from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.revision.io import sha256_file
from src.revision.n_terminal_counterfactuals import (
    analyze_n_terminal_counterfactuals,
    build_counterfactual_variants,
    received_attention_by_key,
    synthetic_n_terminal_fixture,
    token_ids_sha256,
    validate_disjoint_cohorts,
)


class NTerminalCounterfactualTest(unittest.TestCase):
    equivalence_spec = {
        "alpha": 0.05,
        "multiplicity": "holm_all_feature_control_and_protein_pair_did_cells",
        "margins": {
            "feature_activation_pre": 0.10,
            "normalized_received_attention": 0.02,
            "suffix_nll_increase_key_masked": 0.05,
            "suffix_observed_token_logit_change_key_masked": 0.05,
        },
    }

    def test_received_attention_is_normalized_by_eligible_causal_queries(self):
        attention = np.tril(np.ones((4, 4), dtype=float))
        result = received_attention_by_key(attention, np.array([1, 1, 1, 0], dtype=bool))
        np.testing.assert_array_equal(result["eligible_query_count"], [3, 2, 1, 0])
        np.testing.assert_allclose(result["raw_received_attention"], [3, 2, 1, 0])
        np.testing.assert_allclose(result["normalized_received_attention"], [1, 1, 1, 0])

    def test_sequence_counterfactuals_preserve_the_intended_mxx_copy(self):
        natural = "MKT" + "ACDEFGHIKLMNPQRSTVWY" * 2
        variants = build_counterfactual_variants([{"protein_id": "p1", "sequence": natural}])
        by_condition = {variant.condition: variant for variant in variants}
        self.assertEqual(by_condition["natural_mxx"].sequence[:3], "MKT")
        self.assertEqual(by_condition["m_to_a"].sequence[:3], "AKT")
        internal = by_condition["internal_mxx_insertion"]
        self.assertEqual(internal.sequence[internal.focal_start : internal.focal_start + 3], "MKT")
        self.assertEqual(by_condition["artificial_truncation"].sequence[:3], "MKT")
        self.assertEqual(internal.sequence_length, len(natural) + 3)

    def test_synthetic_factorial_reports_paired_effects_and_feature_matches(self):
        _, variants, rows = synthetic_n_terminal_fixture(seed=13, n_proteins=32)
        result = analyze_n_terminal_counterfactuals(
            rows,
            variants=variants,
            equivalence_spec=self.equivalence_spec,
            n_bootstrap=100,
            seed=19,
            control_count=2,
        )
        self.assertEqual(result["feature_measurement_timing"], "before_any_attention_intervention")
        self.assertEqual(len(result["feature_matches"]), 1)
        self.assertEqual(len(result["feature_matches"][0]["controls"]), 2)
        self.assertEqual(
            result["feature_matches"][0]["matching_source"],
            "extractor_frozen_feature_match_id",
        )
        target = next(feature for feature in result["features"] if feature["feature_role"] == "target")
        self.assertGreater(
            target["contrasts"]["initiator_m_to_a"]["feature_activation_pre"]["mean"],
            -1.0,
        )
        self.assertLess(
            target["contrasts"]["initiator_m_to_a"]["feature_activation_pre"]["mean"],
            0.0,
        )
        self.assertGreater(
            target["contrasts"]["artificial_start_vs_internal"]["feature_activation_pre"]["mean"],
            0.0,
        )
        self.assertIn("normalized_received_attention", target["conditional_model"])
        self.assertEqual(len(result["target_minus_control_inference"]), 8)
        self.assertTrue(result["protein_pair_difference_in_differences"])
        self.assertIn(
            "suffix_nll_increase_key_masked",
            result["conditional_attention_path_effects"],
        )
        self.assertTrue(
            all("p_tost_holm" in row["equivalence"] for row in result["target_minus_control_inference"])
        )

    def test_incomplete_or_nonreciprocal_protein_pair_roles_fail(self):
        _, variants, rows = synthetic_n_terminal_fixture(seed=29, n_proteins=24)
        bad = [dict(row) for row in rows]
        protein = bad[0]["protein_id"]
        for row in bad:
            if row["protein_id"] == protein:
                row["protein_match_role"] = "control"
        with self.assertRaisesRegex(ValueError, "duplicate target/control roles"):
            analyze_n_terminal_counterfactuals(
                bad,
                variants=variants,
                equivalence_spec=self.equivalence_spec,
                n_bootstrap=20,
            )

    def test_incomplete_condition_factorial_fails(self):
        _, variants, rows = synthetic_n_terminal_fixture(seed=17, n_proteins=24)
        with self.assertRaisesRegex(ValueError, "identical frozen protein set|complete condition"):
            analyze_n_terminal_counterfactuals(
                rows[:-1],
                variants=variants,
                equivalence_spec=self.equivalence_spec,
                n_bootstrap=20,
            )

    def test_focal_key_effects_must_match_the_recorded_endpoints(self):
        _, variants, rows = synthetic_n_terminal_fixture(seed=18, n_proteins=24)
        bad = [dict(row) for row in rows]
        bad[0]["suffix_nll_increase_key_masked"] += 0.1
        with self.assertRaisesRegex(ValueError, "key-masked minus baseline"):
            analyze_n_terminal_counterfactuals(
                bad,
                variants=variants,
                equivalence_spec=self.equivalence_spec,
                n_bootstrap=20,
            )

        fractional_focal = [dict(row) for row in rows]
        fractional_focal[0]["protein_match_focal_position"] += 0.5
        with self.assertRaisesRegex(ValueError, "must be integers"):
            analyze_n_terminal_counterfactuals(
                fractional_focal,
                variants=variants,
                equivalence_spec=self.equivalence_spec,
                n_bootstrap=20,
            )

    def test_measurements_must_join_exact_variant_and_tokenization_hashes(self):
        _, variants, rows = synthetic_n_terminal_fixture(seed=21, n_proteins=24)
        bad_variant = [dict(row) for row in rows]
        bad_variant[0]["variant_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "variant hash mismatch"):
            analyze_n_terminal_counterfactuals(
                bad_variant,
                variants=variants,
                equivalence_spec=self.equivalence_spec,
                n_bootstrap=20,
            )
        bad_tokens = [dict(row) for row in rows]
        bad_tokens[0]["token_ids"] = [*bad_tokens[0]["token_ids"], 1]
        self.assertNotEqual(
            token_ids_sha256(bad_tokens[0]["token_ids"]), bad_tokens[0]["token_ids_sha256"]
        )
        with self.assertRaisesRegex(ValueError, "tokenization hash mismatch"):
            analyze_n_terminal_counterfactuals(
                bad_tokens,
                variants=variants,
                equivalence_spec=self.equivalence_spec,
                n_bootstrap=20,
            )

    def test_every_target_and_control_requires_the_identical_protein_set(self):
        _, variants, rows = synthetic_n_terminal_fixture(seed=25, n_proteins=24)
        protein = rows[0]["protein_id"]
        reduced = [
            row for row in rows if not (row["feature"] == 202 and row["protein_id"] == protein)
        ]
        with self.assertRaisesRegex(ValueError, "identical frozen protein set"):
            analyze_n_terminal_counterfactuals(
                reduced,
                variants=variants,
                equivalence_spec=self.equivalence_spec,
                n_bootstrap=20,
            )

    def test_frozen_feature_match_ids_cannot_be_rematched_or_postselected(self):
        _, variants, rows = synthetic_n_terminal_fixture(seed=26, n_proteins=24)
        with self.assertRaisesRegex(ValueError, "exactly one target and exactly 1 controls"):
            analyze_n_terminal_counterfactuals(
                rows,
                variants=variants,
                equivalence_spec=self.equivalence_spec,
                n_bootstrap=20,
                control_count=1,
            )
        bad = [dict(row) for row in rows]
        for row in bad:
            if row["feature"] == 202:
                row["feature_match_id"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "exactly one target"):
            analyze_n_terminal_counterfactuals(
                bad,
                variants=variants,
                equivalence_spec=self.equivalence_spec,
                n_bootstrap=20,
                control_count=2,
            )

    def test_discovery_and_evaluation_reject_id_or_sequence_overlap(self):
        evaluation = [{"protein_id": "eval", "sequence": "M" + "A" * 20}]
        discovery = [{"protein_id": "discovery", "sequence": "M" + "C" * 20}]
        self.assertTrue(
            validate_disjoint_cohorts(evaluation, discovery)["discovery_evaluation_disjoint"]
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_disjoint_cohorts(
                evaluation, [{"protein_id": "other", "sequence": "M" + "A" * 20}]
            )


def _load_script59():
    script = Path(__file__).resolve().parents[1] / "scripts/59_run_n_terminal_counterfactuals.py"
    module_spec = importlib.util.spec_from_file_location("script59", script)
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _script59_receipt_fixture(root: Path) -> SimpleNamespace:
    extraction = root / "extraction"
    extraction.mkdir()
    sequences = extraction / "natural_cohort.jsonl"
    measurements = extraction / "measurements.jsonl"
    feature_matches = extraction / "feature_matches.json"
    discovery = root / "discovery.jsonl"
    equivalence = root / "equivalence.json"
    sequences.write_text('{"protein_id":"p","sequence":"MAAAAAAAAAAAAAAAAA"}\n')
    measurements.write_text('{"fixture":true}\n')
    _write_json(
        feature_matches,
        [
            {
                "feature_match_id": "a" * 64,
                "model": "fixture-model",
                "layer": 2,
                "target_feature": 101,
                "control_features": [201, 202],
                "max_abs_log10_firing_ratio": 0.2,
                "max_abs_log_input_norm_ratio": 0.2,
                "target_profile": {
                    "firing_frequency": 0.3,
                    "input_norm": 1.2,
                },
                "controls": [],
                "method": (
                    "same_model_layer_log_ratio_calipers_then_scaled_euclidean"
                ),
            }
        ],
    )
    discovery.write_text('{"protein_id":"d","sequence":"MCCCCCCCCCCCCCCCCC"}\n')
    _write_json(equivalence, {"fixture": True})
    receipt = extraction / "run_manifest.json"
    _write_json(
        receipt,
        {
            "schema_version": "r2-p05-pretrained-extraction-manifest-v4",
            "status": "verified_production_complete",
            "execution_mode": "production",
            "model": {
                "model_inference_dtype": "bfloat16",
                "observed_model_parameter_dtypes": ["bfloat16"],
                "model_inference_dtype_verification": (
                    "all_floating_model_parameters_exactly_declared_before_first_activation"
                ),
                "model_inference_dtype_verified": True,
                "activation_finiteness_check": (
                    "all_required_layer_captured_activation_and_logit_tensors_"
                    "before_downstream_conversion_or_use"
                ),
                "activation_finiteness_verified": True,
            },
            "inputs": {
                "discovery_cohort": {
                    "path": str(discovery),
                    "sha256": sha256_file(discovery),
                    "n_records": 1,
                }
            },
            "matching": {
                "feature": {"control_count": 2},
                "feature_match_ids": ["a" * 64],
            },
            "artifact_hashes": {
                sequences.name: sha256_file(sequences),
                measurements.name: sha256_file(measurements),
                feature_matches.name: sha256_file(feature_matches),
            },
        },
    )
    return SimpleNamespace(
        synthetic=False,
        sequences=sequences,
        sequences_sha256=sha256_file(sequences),
        measurements=measurements,
        measurements_sha256=sha256_file(measurements),
        extractor_receipt=receipt,
        extractor_receipt_sha256=sha256_file(receipt),
        discovery_cohort=discovery,
        discovery_cohort_sha256=sha256_file(discovery),
        equivalence_spec=equivalence,
        equivalence_spec_sha256=sha256_file(equivalence),
        control_count=None,
    )


def test_script59_requires_external_pins_and_rehashes_extractor_artifacts():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        args = _script59_receipt_fixture(root)
        observed = _load_script59().verify_extractor_inputs(args)
        assert observed["control_count"] == 2
        args.measurements_sha256 = None
        with np.testing.assert_raises_regex(ValueError, "externally pinned inputs"):
            _load_script59().verify_extractor_inputs(args)
        args.measurements_sha256 = sha256_file(args.measurements)
        args.measurements.write_text('{"tampered":true}\n')
        with np.testing.assert_raises_regex(ValueError, "externally supplied SHA-256"):
            _load_script59().verify_extractor_inputs(args)


def test_script59_rejects_unverified_extractor_numerical_integrity():
    with tempfile.TemporaryDirectory() as temporary:
        args = _script59_receipt_fixture(Path(temporary))
        receipt = json.loads(args.extractor_receipt.read_text(encoding="utf-8"))
        receipt["model"]["activation_finiteness_verified"] = False
        _write_json(args.extractor_receipt, receipt)
        args.extractor_receipt_sha256 = sha256_file(args.extractor_receipt)
        with np.testing.assert_raises_regex(ValueError, "bfloat16 numerical integrity"):
            _load_script59().verify_extractor_inputs(args)
        receipt["model"]["activation_finiteness_verified"] = True
        receipt["schema_version"] = "r2-p05-pretrained-extraction-manifest-v3"
        _write_json(args.extractor_receipt, receipt)
        args.extractor_receipt_sha256 = sha256_file(args.extractor_receipt)
        with np.testing.assert_raises_regex(ValueError, "verified production evidence"):
            _load_script59().verify_extractor_inputs(args)


def test_script59_atomic_new_directory_publish_cleans_failed_staging():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        destination = root / "result"
        args = SimpleNamespace(
            synthetic=True,
            out_dir=destination,
            seed=7,
            n_bootstrap=2,
            control_count=1,
            synthetic_proteins=24,
        )
        with np.testing.assert_raises_regex(ValueError, "exactly 1 controls"):
            _load_script59().run(args)
        assert not destination.exists()
        assert not list(root.glob(".result.tmp-*"))

        args.control_count = 2
        summary = _load_script59().run(args)
        assert summary["status"] == "synthetic_pipeline_validation_only"
        assert (destination / "run_manifest.json").is_file()
        with np.testing.assert_raises(FileExistsError):
            _load_script59().run(args)


if __name__ == "__main__":
    unittest.main()
