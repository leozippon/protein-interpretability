from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "revision_atlas_runner",
    Path(__file__).resolve().parents[1] / "scripts/52_run_revision_atlas.py",
)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


class RevisionAtlasRunnerContractTest(unittest.TestCase):
    def analysis(self, seed: int | str) -> dict:
        return {
            "model_seed": 0,
            "dictionary_seed": seed,
            "discovery_manifest_sha256": "a" * 64,
            "heldout_manifest_sha256": "b" * 64,
            "eligibility_receipt_sha256": "c" * 64,
            "layer_maps": {"primary": {}, "sensitivity": {}},
        }

    def test_confirmatory_grid_requires_exact_integer_seed_panel(self):
        grid = {
            "matchers": sorted(RUNNER.REQUIRED_MATCHERS),
            "correlation_modes": sorted(RUNNER.REQUIRED_MODES),
            "feature_pool_sizes": [128, 256],
            "thresholds": [0.9, 0.95],
        }
        spec = {"confirmatory": True, "null": {"n_permutations": 1_000}}
        with self.assertRaisesRegex(ValueError, "integer dictionary_seed"):
            RUNNER.validate_confirmatory(
                spec,
                grid,
                [self.analysis(seed) for seed in ("17", 29, 43)],
            )
        with self.assertRaisesRegex(ValueError, "exactly dictionary seeds"):
            RUNNER.validate_confirmatory(
                spec,
                grid,
                [self.analysis(seed) for seed in (17, 17, 29)],
            )
        RUNNER.validate_confirmatory(
            spec,
            grid,
            [self.analysis(seed) for seed in (17, 29, 43)],
        )

    def test_confirmatory_seed_labels_must_bind_distinct_artifacts(self):
        artifacts = {
            seed: {
                model: {
                    "run_seed": seed,
                    "checkpoint_sha256": f"{seed:064x}",
                    "run_manifest_sha256": f"{seed + 100:064x}",
                    "model_artifacts": {
                        "model_config_sha256": "a" * 64,
                        "model_weights_sha256": "b" * 64,
                        "tokenizer_sha256": "c" * 64,
                    },
                }
                for model in ("protgpt2", "zymctrl", "progen2-medium")
            }
            for seed in (17, 29, 43)
        }
        RUNNER.validate_distinct_eligible_seed_artifacts(artifacts)
        mislabeled = {
            seed: {model: dict(row) for model, row in panel.items()}
            for seed, panel in artifacts.items()
        }
        mislabeled[29]["protgpt2"]["run_seed"] = 17
        with self.assertRaisesRegex(ValueError, "does not bind dictionary seed"):
            RUNNER.validate_distinct_eligible_seed_artifacts(mislabeled)
        duplicated = {
            seed: {model: dict(row) for model, row in panel.items()}
            for seed, panel in artifacts.items()
        }
        duplicated[29]["protgpt2"]["checkpoint_sha256"] = duplicated[17][
            "protgpt2"
        ]["checkpoint_sha256"]
        with self.assertRaisesRegex(ValueError, "distinct eligible protgpt2"):
            RUNNER.validate_distinct_eligible_seed_artifacts(duplicated)

    def test_builder_manifest_is_hash_pinned_and_fixture_scope_cannot_upgrade(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            matrix_path = directory / "matrix.npy"
            np.save(matrix_path, np.arange(8, dtype=np.float32).reshape(2, 4))
            provenance = {
                "confirmatory": False,
                "run_seed": 0,
                "eligibility_receipt_sha256": None,
                "checkpoint_sha256": "a" * 64,
                "run_manifest_sha256": "b" * 64,
                "source_manifest_sha256_by_split": None,
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
                "model_artifacts": {
                    "model_config_sha256": "c" * 64,
                    "model_weights_sha256": "d" * 64,
                    "tokenizer_sha256": "e" * 64,
                },
            }
            manifest = {
                "schema_version": "r2_p0_3_atlas_input_v3",
                "confirmatory": False,
                "status": "nonconfirmatory_fixture_inputs_only",
                "model_seed": 0,
                "cohort_id": "fixture",
                "sequence_hashes": [
                    hashlib.sha256(value.encode()).hexdigest()
                    for value in ("first", "second")
                ],
                "p0_2_eligibility_receipt": None,
                "models": {"alpha": {"0": provenance}},
                "matrices": [
                    {
                        "model": "alpha",
                        "dictionary_seed": 0,
                        "layer": "0",
                        "path": matrix_path.name,
                        "sha256": RUNNER.sha256_file(matrix_path),
                        "n_rows": 2,
                        "n_features": 4,
                        "dictionary_provenance": provenance,
                    }
                ],
            }
            manifest_path = directory / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            digest = RUNNER.sha256_file(manifest_path)
            matrices, _, loaded = RUNNER.load_cohort(
                manifest_path,
                expected_manifest_sha256=digest,
                dictionary_seed=0,
                confirmatory=False,
            )
            self.assertEqual(set(matrices), {"alpha"})
            self.assertEqual(loaded["dictionary_seed"], 0)
            with self.assertRaisesRegex(ValueError, "builder manifest SHA-256 mismatch"):
                RUNNER.load_cohort(
                    manifest_path,
                    expected_manifest_sha256="0" * 64,
                    dictionary_seed=0,
                    confirmatory=False,
                )
            with self.assertRaisesRegex(ValueError, "scope/status is ineligible"):
                RUNNER.load_cohort(
                    manifest_path,
                    expected_manifest_sha256=digest,
                    dictionary_seed=17,
                    confirmatory=True,
                )
            manifest["schema_version"] = "r2_p0_3_atlas_input_v2"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scope/status is ineligible"):
                RUNNER.load_cohort(
                    manifest_path,
                    expected_manifest_sha256=RUNNER.sha256_file(manifest_path),
                    dictionary_seed=0,
                    confirmatory=False,
                )
            manifest["schema_version"] = "r2_p0_3_atlas_input_v3"
            provenance["activation_finiteness_verified"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inference provenance is ineligible"):
                RUNNER.load_cohort(
                    manifest_path,
                    expected_manifest_sha256=RUNNER.sha256_file(manifest_path),
                    dictionary_seed=0,
                    confirmatory=False,
                )


if __name__ == "__main__":
    unittest.main()
