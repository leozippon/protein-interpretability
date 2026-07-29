from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.revision.steering_protocol import (
    EC_PROMPTS,
    analyze_steering_scores,
    build_steering_plan,
    select_positive_features,
    synthetic_steering_fixture,
    validate_completed_generations,
    validate_disjoint_selection_evaluation_cohorts,
    validate_endpoint_specs,
    validate_plan_rows,
    validate_provenance,
    validate_score_receipt,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class SteeringProtocolTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.artifact_dir = Path(temporary.name)
        self.scorer_path = self.artifact_dir / "scorer.bin"
        self.calibration_path = self.artifact_dir / "calibration.jsonl"
        self.scorer_path.write_bytes(b"frozen-scorer")
        self.calibration_path.write_bytes(b'{"cohort":"calibration"}\n')
        self.attributions, self.pool = synthetic_steering_fixture(23)
        self.endpoint_specs = {
            "validated": {
                "direction": "higher",
                "experimental_unit": "generation",
                "equivalence_margin": 0.05,
                "validated": True,
                "primary": True,
                "scorer_name": "calibrated-test-scorer",
                "scorer_version": "v1",
                "scorer_path": str(self.scorer_path),
                "scorer_sha256": hashlib.sha256(self.scorer_path.read_bytes()).hexdigest(),
                "calibration_cohort_path": str(self.calibration_path),
                "calibration_cohort_sha256": hashlib.sha256(
                    self.calibration_path.read_bytes()
                ).hexdigest(),
            }
        }
        self.analysis_spec = {
            "alpha": 0.05,
            "n_resamples": 2000,
            "random_seed": 41,
            "multiplicity": "holm_all_arm_and_specificity_cells",
            "decision_rule": "target_vs_prompt_and_both_controls",
        }

    def selection(self, classes=None, layers=None, sites=None):
        return select_positive_features(
            self.attributions,
            selection_split_id="synthetic_selection",
            evaluation_split_id="synthetic_evaluation",
            classes=classes or list(EC_PROMPTS),
            layers=layers or [3, 12, 30],
            sites=sites or ["clt_input", "mlp_output"],
            features_per_cell=3,
        )

    def plan(
        self,
        *,
        classes=None,
        layers=None,
        sites=None,
        doses=None,
        n_per_arm=8,
        set_size=2,
        caliper=0.20,
    ):
        classes = classes or list(EC_PROMPTS)
        layers = layers or [3, 12, 30]
        sites = sites or ["clt_input"]
        return build_steering_plan(
            self.selection(classes, layers, sites),
            self.pool,
            classes=classes,
            layers=layers,
            sites=sites,
            doses=doses or [0.5, 1.0],
            n_per_arm=n_per_arm,
            generation_set_size=set_size,
            seed_base=100,
            sampler={"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 80},
            norm_log_caliper=caliper,
            content_binding_sha256=digest("frozen-context"),
            generator_revision="generator-v1",
            model_revision="model-v1",
            tokenizer_revision="tokenizer-v1",
            clt_checkpoint_sha256=digest("clt"),
            multiplier_semantics="additive_decoder_direction_v1",
        )

    @staticmethod
    def outputs(rows):
        result = []
        for row in rows:
            token_ids = [1, 2, 3, 4]
            result.append(
                {
                    "plan_id": row["plan_id"],
                    "sequence": "MACD",
                    "token_ids": token_ids,
                    "token_ids_sha256": digest("[1,2,3,4]"),
                    "stop_reason": "eos",
                    "runtime": {
                        "generator_revision": "generator-v1",
                        "model_revision": row["model_revision"],
                        "tokenizer_revision": row["tokenizer_revision"],
                        "clt_checkpoint_sha256": row["clt_checkpoint_sha256"],
                        "hostname": "test-host",
                        "device": "cpu",
                        "started_at_utc": "2026-07-17T00:00:00Z",
                        "elapsed_seconds": 0.1,
                        "evaluation_mode": "eval",
                        "hook_site": row["site"],
                        "multiplier_semantics": "additive_decoder_direction_v1",
                        "rng_stream_id": row["rng_stream_id"],
                    },
                }
            )
        return result

    def test_selection_is_independent_positive_only_and_complete(self):
        selection = self.selection()
        self.assertFalse(selection["opposite_sign_fallback"])
        self.assertEqual(len(selection["selected_by_cell"]), 8 * 3 * 2)
        self.assertTrue(
            all(
                row["direct_effect"] > 0
                for rows in selection["selected_by_cell"].values()
                for row in rows
            )
        )
        with self.assertRaisesRegex(ValueError, "distinct"):
            select_positive_features(
                self.attributions,
                selection_split_id="same",
                evaluation_split_id="same",
                classes=["lysozyme"],
                layers=[3],
                sites=["clt_input"],
                features_per_cell=1,
            )

    def test_insufficient_positive_cell_refuses_fallback(self):
        rows = [
            row
            for row in self.attributions
            if not (
                row["ec_class"] == "lysozyme"
                and row["layer"] == 3
                and row["site"] == "clt_input"
                and row["direct_effect"] > 0
            )
        ]
        with self.assertRaisesRegex(ValueError, "Refusing opposite-sign fallback"):
            select_positive_features(
                rows,
                selection_split_id="synthetic_selection",
                evaluation_split_id="synthetic_evaluation",
                classes=["lysozyme"],
                layers=[3],
                sites=["clt_input"],
                features_per_cell=1,
            )

    def test_plan_ids_bind_content_sets_pair_rng_and_report_norm_balance(self):
        plan = self.plan(classes=["lysozyme"], layers=[3], n_per_arm=4, set_size=2)
        self.assertTrue(plan["norm_match_balance"]["all_within_caliper"])
        self.assertEqual(plan["n_generation_sets"], 2)
        validate_plan_rows(plan["rows"])
        seed_streams = {}
        for row in plan["rows"]:
            seed_streams.setdefault(row["seed"], set()).add(row["rng_stream_id"])
        self.assertTrue(all(len(streams) == 1 for streams in seed_streams.values()))
        tampered = [dict(row) for row in plan["rows"]]
        tampered[0]["sampler"] = {**tampered[0]["sampler"], "temperature": 0.2}
        with self.assertRaisesRegex(ValueError, "plan_id is not bound"):
            validate_plan_rows(tampered)

    def test_norm_match_caliper_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "norm control within log caliper"):
            self.plan(classes=["lysozyme"], layers=[3], caliper=1e-8)

    def test_endpoint_specs_require_unit_and_validated_scorer_provenance(self):
        with self.assertRaisesRegex(ValueError, "experimental_unit"):
            validate_endpoint_specs(
                {
                    "bad": {
                        "direction": "higher",
                        "equivalence_margin": 0.05,
                        "validated": True,
                        "primary": True,
                    }
                }
            )
        missing_scorer = {
            "bad": {
                "direction": "higher",
                "experimental_unit": "generation",
                "equivalence_margin": 0.05,
                "validated": True,
                "primary": True,
            }
        }
        with self.assertRaisesRegex(ValueError, "scorer/calibration provenance"):
            validate_endpoint_specs(missing_scorer)
        ungrounded = copy.deepcopy(self.endpoint_specs)
        ungrounded["validated"].pop("scorer_path")
        ungrounded["validated"].pop("calibration_cohort_path")
        with self.assertRaisesRegex(ValueError, "requires real scorer"):
            validate_endpoint_specs(ungrounded, require_artifacts=True)

        tampered = copy.deepcopy(self.endpoint_specs)
        tampered["validated"]["scorer_sha256"] = digest("self-asserted-wrong-hash")
        with self.assertRaisesRegex(ValueError, "scorer artifact path or SHA-256 mismatch"):
            validate_endpoint_specs(tampered, require_artifacts=True)

    def test_provenance_requires_distinct_hashed_cohorts(self):
        provenance = {
            "model_revision": "model@revision",
            "tokenizer_revision": "tokenizer@revision",
            "clt_checkpoint_sha256": "c" * 64,
            "selection_cohort_sha256": "a" * 64,
            "evaluation_cohort_sha256": "b" * 64,
            "code_revision": "git-revision",
        }
        self.assertEqual(validate_provenance(provenance), provenance)
        provenance["evaluation_cohort_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            validate_provenance(provenance)

    def test_selection_and_evaluation_cohort_membership_must_be_disjoint(self):
        selection = [{"protein_id": "s1", "sequence": "M" + "A" * 20}]
        evaluation = [{"protein_id": "e1", "sequence": "M" + "C" * 20}]
        self.assertTrue(
            validate_disjoint_selection_evaluation_cohorts(selection, evaluation)[
                "selection_evaluation_disjoint"
            ]
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_disjoint_selection_evaluation_cohorts(
                selection, [{"protein_id": "e2", "sequence": "M" + "A" * 20}]
            )

    def test_completed_generation_requires_raw_tokens_runtime_and_exact_rng(self):
        plan = self.plan(classes=["lysozyme"], layers=[3], n_per_arm=4)
        outputs = self.outputs(plan["rows"])
        completed = validate_completed_generations(plan["rows"], outputs)
        self.assertEqual(len(completed), len(plan["rows"]))
        broken = [dict(row) for row in outputs]
        broken[0] = {key: value for key, value in broken[0].items() if key != "token_ids"}
        with self.assertRaisesRegex(ValueError, "token_ids"):
            validate_completed_generations(plan["rows"], broken)
        broken = [dict(row) for row in outputs]
        broken[0] = {**broken[0], "runtime": {**broken[0]["runtime"], "rng_stream_id": "wrong"}}
        with self.assertRaisesRegex(ValueError, "RNG stream"):
            validate_completed_generations(plan["rows"], broken)

    def _generation_scores(self, completed, control_effect=0.0):
        scores = []
        for row in completed:
            base = 0.4 + (row["seed"] % 5) * 0.01
            effect = 0.0
            if row["arm"] == "target" and row["ec_class"] == "lysozyme":
                effect = 0.2
            elif row["arm"] in {"random_feature", "norm_matched_feature"}:
                effect = control_effect
            scores.append({"plan_id": row["plan_id"], "endpoint": "validated", "score": base + effect})
        return scores

    @staticmethod
    def _score_receipt(completed, scores, endpoint_specs):
        plan_ids = sorted(row["plan_id"] for row in completed)
        set_ids = sorted({row["generation_set_id"] for row in completed})
        scored = {}
        for row in scores:
            endpoint = row["endpoint"]
            unit_field = (
                "plan_id"
                if endpoint_specs[endpoint]["experimental_unit"] == "generation"
                else "generation_set_id"
            )
            scored.setdefault(endpoint, []).append(row[unit_field])
        executions = []
        for endpoint, spec in sorted(endpoint_specs.items()):
            expected_ids = (
                plan_ids if spec["experimental_unit"] == "generation" else set_ids
            )
            scored_ids = sorted(scored[endpoint])
            expected_coverage = canonical_digest(
                {
                    "endpoint": endpoint,
                    "experimental_unit": spec["experimental_unit"],
                    "unit_ids": expected_ids,
                }
            )
            scored_coverage = canonical_digest(
                {
                    "endpoint": endpoint,
                    "experimental_unit": spec["experimental_unit"],
                    "unit_ids": scored_ids,
                }
            )
            executions.append(
                {
                    "endpoint": endpoint,
                    "status": "complete",
                    "execution_mode": (
                        "independently_validated"
                        if spec["validated"]
                        else "heuristic_supporting_only"
                    ),
                    "validated": spec["validated"],
                    "primary": spec["primary"],
                    "scorer_name": spec.get("scorer_name"),
                    "scorer_version": spec.get("scorer_version"),
                    "scorer_path": spec.get("scorer_path"),
                    "scorer_sha256": spec.get("scorer_sha256"),
                    "calibration_cohort_path": spec.get("calibration_cohort_path"),
                    "calibration_cohort_sha256": spec.get(
                        "calibration_cohort_sha256"
                    ),
                    "experimental_unit": spec["experimental_unit"],
                    "expected_unit_count": len(expected_ids),
                    "scored_unit_count": len(scored_ids),
                    "expected_coverage_sha256": expected_coverage,
                    "scored_coverage_sha256": scored_coverage,
                }
            )
        return {
            "schema_version": "r2-corrected-steering-score-receipt-v2",
            "status": "verified_complete",
            "synthetic": False,
            "freeze_id": digest("freeze"),
            "generation_outputs_sha256": digest("generations"),
            "generation_execution_receipt_sha256": digest("execution-receipt"),
            "scores_sha256": digest("scores"),
            "frozen_endpoint_specs_sha256": digest("endpoint-specs"),
            "scorer_executions": executions,
        }

    def test_score_receipt_is_exact_complete_and_hash_bound(self):
        plan = self.plan(classes=["lysozyme"], layers=[3], n_per_arm=4)
        completed = validate_completed_generations(plan["rows"], self.outputs(plan["rows"]))
        specs = {
            **self.endpoint_specs,
            "heuristic": {
                "direction": "higher",
                "experimental_unit": "generation_set",
                "equivalence_margin": 0.05,
                "validated": False,
                "primary": False,
            },
        }
        scores = self._generation_scores(completed)
        scores.extend(
            {
                "generation_set_id": set_id,
                "endpoint": "heuristic",
                "score": 0.5,
            }
            for set_id in sorted({row["generation_set_id"] for row in completed})
        )
        receipt = self._score_receipt(completed, scores, specs)
        result = validate_score_receipt(
            receipt,
            completed,
            scores,
            specs,
            freeze_id=digest("freeze"),
            generation_outputs_sha256=digest("generations"),
            generation_execution_receipt_sha256=digest("execution-receipt"),
            scores_sha256=digest("scores"),
            frozen_endpoint_specs_sha256=digest("endpoint-specs"),
            synthetic=False,
        )
        self.assertTrue(result["receipt_validated"])
        self.assertEqual(result["n_scorer_executions"], 2)

        def rejected(mutator, message):
            changed = copy.deepcopy(receipt)
            mutator(changed)
            with self.assertRaisesRegex(ValueError, message):
                validate_score_receipt(
                    changed,
                    completed,
                    scores,
                    specs,
                    freeze_id=digest("freeze"),
                    generation_outputs_sha256=digest("generations"),
                    generation_execution_receipt_sha256=digest("execution-receipt"),
                    scores_sha256=digest("scores"),
                    frozen_endpoint_specs_sha256=digest("endpoint-specs"),
                    synthetic=False,
                )

        rejected(lambda value: value.update(freeze_id=digest("tampered")), "freeze_id")
        rejected(
            lambda value: value.update(generation_outputs_sha256=digest("other")),
            "generation_outputs_sha256",
        )
        rejected(
            lambda value: value.update(
                generation_execution_receipt_sha256=digest("other")
            ),
            "generation_execution_receipt_sha256",
        )
        rejected(lambda value: value.update(scores_sha256=digest("other")), "scores_sha256")
        rejected(
            lambda value: value.update(frozen_endpoint_specs_sha256=digest("other")),
            "frozen_endpoint_specs_sha256",
        )
        rejected(lambda value: value.update(status="running"), "non-complete status")
        rejected(lambda value: value.update(unexpected=True), "exact frozen schema")
        rejected(lambda value: value["scorer_executions"].pop(), "exactly one execution")
        rejected(
            lambda value: value["scorer_executions"].append(
                {**value["scorer_executions"][0], "endpoint": "extra"}
            ),
            "unknown or duplicate endpoint",
        )

        def masquerade(value):
            row = next(
                item for item in value["scorer_executions"] if item["endpoint"] == "heuristic"
            )
            row["validated"] = True
            row["execution_mode"] = "independently_validated"

        rejected(masquerade, "validation class differs")

        def wrong_scorer(value):
            row = next(
                item for item in value["scorer_executions"] if item["endpoint"] == "validated"
            )
            row["scorer_sha256"] = digest("wrong-scorer")

        rejected(wrong_scorer, "scorer_sha256 differs")

        def incomplete(value):
            value["scorer_executions"][0]["status"] = "partial"

        rejected(incomplete, "non-complete")

        def partial_coverage(value):
            value["scorer_executions"][0]["scored_unit_count"] -= 1

        rejected(partial_coverage, "coverage differs")

        def boolean_count(value):
            value["scorer_executions"][0]["scored_unit_count"] = True

        rejected(boolean_count, "counts must be integers")

        partial_scores = scores[:-1]
        with self.assertRaisesRegex(ValueError, "exact expected coverage"):
            validate_score_receipt(
                receipt,
                completed,
                partial_scores,
                specs,
                freeze_id=digest("freeze"),
                generation_outputs_sha256=digest("generations"),
                generation_execution_receipt_sha256=digest("execution-receipt"),
                scores_sha256=digest("scores"),
                frozen_endpoint_specs_sha256=digest("endpoint-specs"),
                synthetic=False,
            )

        self.scorer_path.write_bytes(b"tampered-after-freeze")
        with self.assertRaisesRegex(ValueError, "scorer artifact path or SHA-256 mismatch"):
            validate_score_receipt(
                receipt,
                completed,
                scores,
                specs,
                freeze_id=digest("freeze"),
                generation_outputs_sha256=digest("generations"),
                generation_execution_receipt_sha256=digest("execution-receipt"),
                scores_sha256=digest("scores"),
                frozen_endpoint_specs_sha256=digest("endpoint-specs"),
                synthetic=False,
            )

    def test_positive_requires_target_minus_random_and_norm_specificity(self):
        plan = self.plan(n_per_arm=16, set_size=4)
        completed = validate_completed_generations(plan["rows"], self.outputs(plan["rows"]))
        result = analyze_steering_scores(
            completed,
            self._generation_scores(completed),
            self.endpoint_specs,
            analysis_spec=self.analysis_spec,
        )
        statuses = {row["ec_class"]: row["status"] for row in result["classes"]}
        self.assertEqual(statuses["lysozyme"], "positive_specific")
        self.assertTrue(
            all(statuses[name] == "equivalent_negative" for name in EC_PROMPTS if name != "lysozyme")
        )
        self.assertTrue(result["all_eight_resolved"])
        self.assertTrue(result["specificity_cells"])

        nonspecific = analyze_steering_scores(
            completed,
            self._generation_scores(completed, control_effect=0.2),
            self.endpoint_specs,
            analysis_spec=self.analysis_spec,
        )
        self.assertEqual(
            next(row["status"] for row in nonspecific["classes"] if row["ec_class"] == "lysozyme"),
            "inconclusive",
        )

    def test_generation_set_endpoint_requires_exact_members_and_uses_sets_as_pairs(self):
        plan = self.plan(classes=["lysozyme"], layers=[3], n_per_arm=8, set_size=2)
        completed = validate_completed_generations(plan["rows"], self.outputs(plan["rows"]))
        set_spec = {
            "set_metric": {
                **self.endpoint_specs["validated"],
                "experimental_unit": "generation_set",
            }
        }
        groups = {}
        for row in completed:
            groups.setdefault(row["generation_set_id"], []).append(row)
        scores = []
        for set_id, members in groups.items():
            row = members[0]
            scores.append(
                {
                    "generation_set_id": set_id,
                    "member_plan_ids": sorted(member["plan_id"] for member in members),
                    "endpoint": "set_metric",
                    "score": 0.5 + (0.2 if row["arm"] == "target" else 0.0),
                }
            )
        result = analyze_steering_scores(
            completed, scores, set_spec, analysis_spec=self.analysis_spec
        )
        target = next(
            cell for cell in result["cells"] if cell["arm"] == "target"
        )
        self.assertEqual(target["experimental_unit"], "generation_set")
        self.assertEqual(target["n_pairs"], 4)
        scores[0]["member_plan_ids"] = scores[0]["member_plan_ids"][:-1]
        with self.assertRaisesRegex(ValueError, "member_plan_ids"):
            analyze_steering_scores(
                completed, scores, set_spec, analysis_spec=self.analysis_spec
            )


if __name__ == "__main__":
    unittest.main()
