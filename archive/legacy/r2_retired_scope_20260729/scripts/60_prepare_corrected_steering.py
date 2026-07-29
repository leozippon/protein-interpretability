#!/usr/bin/env python3
"""Freeze or analyze the fail-closed P0-6 corrected steering protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.revision.io import sha256_file, write_json, write_jsonl  # noqa: E402
from src.revision.steering_protocol import (  # noqa: E402
    EC_PROMPTS,
    analyze_steering_scores,
    build_steering_plan,
    select_positive_features,
    synthetic_steering_fixture,
    validate_analysis_spec,
    validate_completed_generations,
    validate_disjoint_selection_evaluation_cohorts,
    validate_endpoint_specs,
    validate_plan_rows,
    validate_provenance,
    validate_score_receipt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "analyze"))
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--attributions", type=Path)
    parser.add_argument("--feature-pool", type=Path)
    parser.add_argument("--endpoint-specs", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--selection-cohort", type=Path)
    parser.add_argument("--evaluation-cohort", type=Path)
    parser.add_argument("--frozen-dir", type=Path)
    parser.add_argument("--generation-outputs", type=Path)
    parser.add_argument("--generation-execution-receipt", type=Path)
    parser.add_argument("--generation-execution-receipt-sha256")
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--score-receipt", type=Path)
    parser.add_argument("--score-receipt-sha256")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "results/npj_revision_20260716/corrected_steering",
    )
    parser.add_argument("--selection-split-id", default="synthetic_selection")
    parser.add_argument("--evaluation-split-id", default="synthetic_evaluation")
    parser.add_argument("--classes", nargs="+", default=list(EC_PROMPTS))
    parser.add_argument("--layers", type=int, nargs="+", default=[3, 12, 30])
    parser.add_argument("--sites", nargs="+", default=["clt_input", "mlp_output"])
    parser.add_argument("--doses", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--features-per-cell", type=int, default=3)
    parser.add_argument("--n-per-arm", type=int, default=24)
    parser.add_argument("--generation-set-size", type=int, default=4)
    parser.add_argument("--norm-log-caliper", type=float, default=0.15)
    parser.add_argument("--seed-base", type=int, default=20260717)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--multiplier-semantics",
        default="additive_decoder_direction_v1",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-resamples", type=int, default=10000)
    return parser.parse_args()


def _strict_json_loads(text: str, label: object) -> object:
    return json.loads(
        text,
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"{label}: non-finite constant {constant}")
        ),
    )


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            value = _strict_json_loads(line, f"{path}:{line_number}")
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def read_json(path: Path) -> dict:
    value = _strict_json_loads(path.read_text(), path)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_hashed_jsonl(path: Path) -> tuple[list[dict], str]:
    payload = path.read_bytes()
    rows = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if line.strip():
            value = _strict_json_loads(line, f"{path}:{line_number}")
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows, hashlib.sha256(payload).hexdigest()


def read_hashed_json(path: Path) -> tuple[dict, str]:
    payload = path.read_bytes()
    value = _strict_json_loads(payload.decode("utf-8"), path)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def jsonl_rows_sha256(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (
                json.dumps(
                    row,
                    allow_nan=False,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def require_sha256(value: object, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def protocol_source_hashes() -> dict[str, str]:
    return {
        Path(__file__).name: sha256_file(Path(__file__)),
        "steering_protocol.py": sha256_file(PROJECT / "src/revision/steering_protocol.py"),
        "statistics.py": sha256_file(PROJECT / "src/revision/statistics.py"),
        "io.py": sha256_file(PROJECT / "src/revision/io.py"),
    }


def refuse_nonempty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def synthetic_endpoint_specs() -> dict:
    provenance = {
        "scorer_name": "synthetic-calibrated-scorer",
        "scorer_version": "v1",
        "scorer_sha256": text_sha256("synthetic-scorer"),
        "calibration_cohort_sha256": text_sha256("synthetic-calibration"),
    }
    return {
        "synthetic_validated_class_score": {
            "direction": "higher",
            "experimental_unit": "generation",
            "equivalence_margin": 0.05,
            "validated": True,
            "primary": True,
            **provenance,
        },
        "synthetic_validated_set_diversity": {
            "direction": "higher",
            "experimental_unit": "generation_set",
            "equivalence_margin": 0.05,
            "validated": True,
            "primary": True,
            **{**provenance, "scorer_name": "synthetic-set-scorer"},
        },
        "synthetic_heuristic_score": {
            "direction": "higher",
            "experimental_unit": "generation",
            "equivalence_margin": 0.05,
            "validated": False,
            "primary": False,
        },
    }


def synthetic_provenance() -> dict:
    return {
        "model_revision": "synthetic_fixture",
        "tokenizer_revision": "synthetic_fixture",
        "clt_checkpoint_sha256": text_sha256("synthetic_clt"),
        "selection_cohort_sha256": text_sha256("synthetic_selection"),
        "evaluation_cohort_sha256": text_sha256("synthetic_evaluation"),
        "code_revision": "synthetic_fixture",
    }


def frozen_analysis_spec(args: argparse.Namespace) -> dict:
    return validate_analysis_spec(
        {
            "alpha": args.alpha,
            "n_resamples": args.n_resamples,
            "random_seed": args.seed_base,
            "multiplicity": "holm_all_arm_and_specificity_cells",
            "decision_rule": "target_vs_prompt_and_both_controls",
        }
    )


def freeze_protocol(
    *,
    out_dir: Path,
    attribution_rows: list[dict],
    feature_pool: list[dict],
    endpoint_specs: dict,
    provenance: dict,
    analysis_spec: dict,
    args: argparse.Namespace,
    input_hashes: dict,
    cohort_validation: dict,
    synthetic: bool,
    endpoint_artifact_base: Path | None = None,
) -> dict:
    refuse_nonempty(out_dir)
    provenance = validate_provenance(provenance)
    endpoint_specs = validate_endpoint_specs(
        endpoint_specs,
        evaluation_cohort_sha256=provenance["evaluation_cohort_sha256"],
        artifact_base=endpoint_artifact_base,
        require_artifacts=not synthetic,
    )
    analysis_spec = validate_analysis_spec(analysis_spec)
    if any(spec["experimental_unit"] == "generation_set" for spec in endpoint_specs.values()):
        if args.generation_set_size < 2:
            raise ValueError("generation-set endpoints require generation_set_size of at least two")
    source_hashes = protocol_source_hashes()
    selection = select_positive_features(
        attribution_rows,
        selection_split_id=args.selection_split_id,
        evaluation_split_id=args.evaluation_split_id,
        classes=args.classes,
        layers=args.layers,
        sites=args.sites,
        features_per_cell=args.features_per_cell,
    )
    freeze_context = {
        "attributions": attribution_rows,
        "feature_pool": feature_pool,
        "endpoint_specs": endpoint_specs,
        "provenance": provenance,
        "analysis_spec": analysis_spec,
        "cohort_validation": cohort_validation,
        "source_hashes": source_hashes,
        "grid": {
            "classes": args.classes,
            "layers": args.layers,
            "sites": args.sites,
            "doses": args.doses,
            "features_per_cell": args.features_per_cell,
            "n_per_arm": args.n_per_arm,
            "generation_set_size": args.generation_set_size,
            "norm_log_caliper": args.norm_log_caliper,
            "seed_base": args.seed_base,
            "sampler": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
            },
            "multiplier_semantics": args.multiplier_semantics,
        },
    }
    content_binding = canonical_sha256(freeze_context)
    plan = build_steering_plan(
        selection,
        feature_pool,
        classes=args.classes,
        layers=args.layers,
        sites=args.sites,
        doses=args.doses,
        n_per_arm=args.n_per_arm,
        generation_set_size=args.generation_set_size,
        seed_base=args.seed_base,
        sampler=freeze_context["grid"]["sampler"],
        norm_log_caliper=args.norm_log_caliper,
        content_binding_sha256=content_binding,
        generator_revision=provenance["code_revision"],
        model_revision=provenance["model_revision"],
        tokenizer_revision=provenance["tokenizer_revision"],
        clt_checkpoint_sha256=provenance["clt_checkpoint_sha256"],
        multiplier_semantics=args.multiplier_semantics,
    )
    plan_rows = plan.pop("rows")
    decisions = plan.pop("selection_decisions")
    controls = plan.pop("control_decisions")
    artifacts = {
        "generation_plan.jsonl": plan_rows,
        "feature_decisions.jsonl": decisions,
        "control_decisions.json": controls,
        "frozen_endpoint_specs.json": endpoint_specs,
        "frozen_provenance.json": provenance,
        "frozen_analysis_spec.json": analysis_spec,
    }
    write_jsonl(out_dir / "generation_plan.jsonl", plan_rows)
    write_jsonl(out_dir / "feature_decisions.jsonl", decisions)
    for name in (
        "control_decisions.json",
        "frozen_endpoint_specs.json",
        "frozen_provenance.json",
        "frozen_analysis_spec.json",
    ):
        write_json(out_dir / name, artifacts[name])
    artifact_hashes = {name: sha256_file(out_dir / name) for name in artifacts}
    freeze_id = canonical_sha256(
        {"content_binding_sha256": content_binding, "artifact_hashes": artifact_hashes}
    )
    summary = {
        **plan,
        "status": "synthetic_freeze_only" if synthetic else "generation_plan_frozen",
        "freeze_id": freeze_id,
        "content_binding_sha256": content_binding,
        "frozen_artifact_hashes": artifact_hashes,
        "cohort_validation": cohort_validation,
    }
    write_json(out_dir / "summary.json", summary)
    write_json(
        out_dir / "run_manifest.json",
        {
            "schema_version": "r2-corrected-steering-freeze-manifest-v3",
            "stage": "freeze",
            "freeze_id": freeze_id,
            "command": [sys.executable, *sys.argv],
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "input_hashes": input_hashes,
            "source_hashes": source_hashes,
            "artifact_hashes": {
                **artifact_hashes,
                "summary.json": sha256_file(out_dir / "summary.json"),
            },
        },
    )
    return {"summary": summary, "plan_rows": plan_rows, "endpoint_specs": endpoint_specs}


def load_frozen_protocol(frozen_dir: Path) -> dict:
    summary = read_json(frozen_dir / "summary.json")
    freeze_manifest = read_json(frozen_dir / "run_manifest.json")
    if (
        freeze_manifest.get("schema_version")
        != "r2-corrected-steering-freeze-manifest-v3"
        or freeze_manifest.get("stage") != "freeze"
    ):
        raise ValueError("unsupported or malformed corrected-steering freeze manifest")
    if freeze_manifest.get("source_hashes") != protocol_source_hashes():
        raise ValueError("current protocol source hashes differ from the immutable freeze")
    artifact_hashes = summary.get("frozen_artifact_hashes")
    if not isinstance(artifact_hashes, dict):
        raise ValueError("frozen summary lacks artifact hashes")
    for name, expected in artifact_hashes.items():
        actual = sha256_file(frozen_dir / name)
        if actual != expected:
            raise ValueError(f"frozen artifact hash mismatch: {name}")
    expected_freeze_id = canonical_sha256(
        {
            "content_binding_sha256": summary["content_binding_sha256"],
            "artifact_hashes": artifact_hashes,
        }
    )
    if summary.get("freeze_id") != expected_freeze_id:
        raise ValueError("freeze_id does not bind the frozen artifact set")
    if freeze_manifest.get("freeze_id") != summary["freeze_id"]:
        raise ValueError("freeze manifest and summary freeze IDs differ")
    expected_manifest_artifacts = {
        **artifact_hashes,
        "summary.json": sha256_file(frozen_dir / "summary.json"),
    }
    if freeze_manifest.get("artifact_hashes") != expected_manifest_artifacts:
        raise ValueError("freeze manifest artifact hashes disagree with the frozen summary")
    plan_rows = validate_plan_rows(read_jsonl(frozen_dir / "generation_plan.jsonl"))
    binding = {row["content_binding_sha256"] for row in plan_rows}
    if binding != {summary["content_binding_sha256"]}:
        raise ValueError("plan rows do not bind the frozen protocol context")
    return {
        "summary": summary,
        "plan_rows": plan_rows,
        "endpoint_specs": read_json(frozen_dir / "frozen_endpoint_specs.json"),
        "analysis_spec": read_json(frozen_dir / "frozen_analysis_spec.json"),
    }


def analyze_protocol(
    *,
    frozen_dir: Path,
    out_dir: Path,
    outputs: list[dict],
    score_rows: list[dict],
    args: argparse.Namespace,
    input_hashes: dict,
    generation_outputs_sha256: str,
    generation_outputs_path: Path | None,
    generation_execution_receipt_path: Path | None,
    generation_execution_receipt_sha256: str | None,
    scores_sha256: str,
    score_receipt: dict,
    score_receipt_sha256: str,
    synthetic: bool,
) -> dict:
    if frozen_dir.resolve() == out_dir.resolve():
        raise ValueError("analysis output must be separate from the immutable frozen directory")
    frozen = load_frozen_protocol(frozen_dir)
    expected_freeze_status = "synthetic_freeze_only" if synthetic else "generation_plan_frozen"
    if frozen["summary"].get("status") != expected_freeze_status:
        raise ValueError("frozen protocol status does not match the requested analysis scope")
    if synthetic:
        if any(
            value is not None
            for value in (
                generation_outputs_path,
                generation_execution_receipt_path,
                generation_execution_receipt_sha256,
            )
        ):
            raise ValueError("synthetic analysis cannot claim a production execution receipt")
        verified_execution_receipt_sha256 = None
    else:
        if (
            generation_outputs_path is None
            or generation_execution_receipt_path is None
            or generation_execution_receipt_sha256 is None
        ):
            raise ValueError("real analysis requires a pinned generation execution receipt")
        from src.revision.steering_execution import verify_execution_receipt

        verified_execution_receipt_sha256 = require_sha256(
            generation_execution_receipt_sha256,
            "generation_execution_receipt_sha256",
        )
        verify_execution_receipt(
            generation_execution_receipt_path,
            verified_execution_receipt_sha256,
            generation_outputs_path=generation_outputs_path,
            freeze_id=frozen["summary"]["freeze_id"],
            freeze_manifest_sha256=sha256_file(frozen_dir / "run_manifest.json"),
        )
    completed = validate_completed_generations(frozen["plan_rows"], outputs)
    receipt_validation = validate_score_receipt(
        score_receipt,
        completed,
        score_rows,
        frozen["endpoint_specs"],
        freeze_id=frozen["summary"]["freeze_id"],
        generation_outputs_sha256=generation_outputs_sha256,
        generation_execution_receipt_sha256=verified_execution_receipt_sha256,
        scores_sha256=scores_sha256,
        frozen_endpoint_specs_sha256=sha256_file(
            frozen_dir / "frozen_endpoint_specs.json"
        ),
        synthetic=synthetic,
    )
    receipt_validation["score_receipt_sha256"] = require_sha256(
        score_receipt_sha256, "score_receipt_sha256"
    )
    refuse_nonempty(out_dir)
    analysis = analyze_steering_scores(
        completed,
        score_rows,
        frozen["endpoint_specs"],
        analysis_spec=frozen["analysis_spec"],
    )
    analysis["freeze_id"] = frozen["summary"]["freeze_id"]
    analysis["frozen_generation_plan_sha256"] = sha256_file(
        frozen_dir / "generation_plan.jsonl"
    )
    analysis["score_receipt_validation"] = receipt_validation
    analysis["status"] = (
        "synthetic_pipeline_validation_only" if synthetic else "completed_confirmatory_analysis"
    )
    write_jsonl(out_dir / "completed_generations.jsonl", completed)
    write_jsonl(out_dir / "endpoint_scores.jsonl", score_rows)
    write_json(out_dir / "validated_score_receipt.json", score_receipt)
    write_json(out_dir / "analysis.json", analysis)
    write_json(
        out_dir / "run_manifest.json",
        {
            "schema_version": "r2-corrected-steering-analysis-manifest-v3",
            "stage": "analyze",
            "freeze_id": frozen["summary"]["freeze_id"],
            "command": [sys.executable, *sys.argv],
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "input_hashes": input_hashes,
            "score_receipt_sha256": receipt_validation["score_receipt_sha256"],
            "frozen_artifact_hashes": frozen["summary"]["frozen_artifact_hashes"],
            "source_hashes": protocol_source_hashes(),
            "artifact_hashes": {
                name: sha256_file(out_dir / name)
                for name in (
                    "completed_generations.jsonl",
                    "endpoint_scores.jsonl",
                    "validated_score_receipt.json",
                    "analysis.json",
                )
            },
        },
    )
    return analysis


def synthetic_score_receipt(
    *,
    freeze_id: str,
    generation_outputs_sha256: str,
    scores_sha256: str,
    frozen_endpoint_specs_sha256: str,
    completed_rows: list[dict],
    score_rows: list[dict],
    endpoint_specs: dict,
) -> dict:
    """Construct an explicitly synthetic scorer receipt for the CPU fixture only."""

    set_ids = sorted({str(row["generation_set_id"]) for row in completed_rows})
    plan_ids = sorted(str(row["plan_id"]) for row in completed_rows)
    scored_by_endpoint: dict[str, list[str]] = defaultdict(list)
    for row in score_rows:
        endpoint = str(row["endpoint"])
        unit_field = (
            "plan_id"
            if endpoint_specs[endpoint]["experimental_unit"] == "generation"
            else "generation_set_id"
        )
        scored_by_endpoint[endpoint].append(str(row[unit_field]))
    executions = []
    for endpoint, spec in sorted(endpoint_specs.items()):
        expected = plan_ids if spec["experimental_unit"] == "generation" else set_ids
        scored = sorted(scored_by_endpoint[endpoint])
        expected_digest = canonical_sha256(
            {
                "endpoint": endpoint,
                "experimental_unit": spec["experimental_unit"],
                "unit_ids": expected,
            }
        )
        scored_digest = canonical_sha256(
            {
                "endpoint": endpoint,
                "experimental_unit": spec["experimental_unit"],
                "unit_ids": scored,
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
                "calibration_cohort_sha256": spec.get("calibration_cohort_sha256"),
                "experimental_unit": spec["experimental_unit"],
                "expected_unit_count": len(expected),
                "scored_unit_count": len(scored),
                "expected_coverage_sha256": expected_digest,
                "scored_coverage_sha256": scored_digest,
            }
        )
    return {
        "schema_version": "r2-corrected-steering-score-receipt-v2",
        "status": "synthetic_fixture_complete",
        "synthetic": True,
        "freeze_id": freeze_id,
        "generation_outputs_sha256": generation_outputs_sha256,
        "generation_execution_receipt_sha256": None,
        "scores_sha256": scores_sha256,
        "frozen_endpoint_specs_sha256": frozen_endpoint_specs_sha256,
        "scorer_executions": executions,
    }


def synthetic_outputs(plan_rows: list[dict], endpoint_specs: dict) -> tuple[list[dict], list[dict]]:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    positive_classes = {"lysozyme", "kinase"}
    outputs = []
    for row in plan_rows:
        sequence = "M" + "".join(
            alphabet[(row["seed"] + index + len(row["arm"])) % len(alphabet)]
            for index in range(79)
        )
        token_ids = [alphabet.index(residue) + 1 for residue in sequence]
        outputs.append(
            {
                "plan_id": row["plan_id"],
                "sequence": sequence,
                "token_ids": token_ids,
                "token_ids_sha256": canonical_sha256(token_ids),
                "stop_reason": "synthetic_length_limit",
                "runtime": {
                    "generator_revision": row["generator_revision"],
                    "model_revision": row["model_revision"],
                    "tokenizer_revision": row["tokenizer_revision"],
                    "clt_checkpoint_sha256": row["clt_checkpoint_sha256"],
                    "hostname": "synthetic-host",
                    "device": "cpu",
                    "started_at_utc": "2026-07-17T00:00:00Z",
                    "elapsed_seconds": 0.001,
                    "evaluation_mode": "eval",
                    "hook_site": row["site"],
                    "multiplier_semantics": row["multiplier_semantics"],
                    "rng_stream_id": row["rng_stream_id"],
                },
            }
        )

    scores = []
    for endpoint, spec in endpoint_specs.items():
        if spec["experimental_unit"] == "generation":
            for row in plan_rows:
                base = 0.45 + (row["seed"] % 7) * 0.005
                effect = (
                    0.18 * row["dose"]
                    if row["arm"] == "target" and row["ec_class"] in positive_classes
                    else 0.0
                )
                scores.append(
                    {"plan_id": row["plan_id"], "endpoint": endpoint, "score": base + effect}
                )
        else:
            groups: dict[str, list[dict]] = defaultdict(list)
            for row in plan_rows:
                groups[row["generation_set_id"]].append(row)
            for set_id, members in groups.items():
                representative = members[0]
                base = 0.55 + representative["generation_set_index"] * 0.002
                effect = (
                    0.18 * representative["dose"]
                    if representative["arm"] == "target"
                    and representative["ec_class"] in positive_classes
                    else 0.0
                )
                scores.append(
                    {
                        "generation_set_id": set_id,
                        "member_plan_ids": sorted(row["plan_id"] for row in members),
                        "endpoint": endpoint,
                        "score": base + effect,
                    }
                )
    return outputs, scores


def main() -> None:
    args = parse_args()
    if args.synthetic:
        if args.stage:
            raise ValueError("--synthetic runs both stages; do not also pass --stage")
        refuse_nonempty(args.out_dir)
        freeze_dir, analysis_dir = args.out_dir / "freeze", args.out_dir / "analysis"
        attributions, pool = synthetic_steering_fixture(args.seed_base)
        provenance = validate_provenance(synthetic_provenance())
        endpoints = validate_endpoint_specs(
            synthetic_endpoint_specs(),
            evaluation_cohort_sha256=provenance["evaluation_cohort_sha256"],
        )
        frozen = freeze_protocol(
            out_dir=freeze_dir,
            attribution_rows=attributions,
            feature_pool=pool,
            endpoint_specs=endpoints,
            provenance=provenance,
            analysis_spec=frozen_analysis_spec(args),
            args=args,
            input_hashes={},
            cohort_validation={
                "status": "synthetic_disjointness_fixture_only",
                "selection_evaluation_disjoint": True,
            },
            synthetic=True,
        )
        outputs, scores = synthetic_outputs(frozen["plan_rows"], endpoints)
        generation_outputs_sha256 = jsonl_rows_sha256(outputs)
        scores_sha256 = jsonl_rows_sha256(scores)
        completed = validate_completed_generations(frozen["plan_rows"], outputs)
        score_receipt = synthetic_score_receipt(
            freeze_id=frozen["summary"]["freeze_id"],
            generation_outputs_sha256=generation_outputs_sha256,
            scores_sha256=scores_sha256,
            frozen_endpoint_specs_sha256=sha256_file(
                freeze_dir / "frozen_endpoint_specs.json"
            ),
            completed_rows=completed,
            score_rows=scores,
            endpoint_specs=endpoints,
        )
        analysis = analyze_protocol(
            frozen_dir=freeze_dir,
            out_dir=analysis_dir,
            outputs=outputs,
            score_rows=scores,
            args=args,
            input_hashes={
                "synthetic_generation_outputs": generation_outputs_sha256,
                "synthetic_scores": scores_sha256,
                "synthetic_score_receipt": canonical_sha256(score_receipt),
            },
            generation_outputs_sha256=generation_outputs_sha256,
            generation_outputs_path=None,
            generation_execution_receipt_path=None,
            generation_execution_receipt_sha256=None,
            scores_sha256=scores_sha256,
            score_receipt=score_receipt,
            score_receipt_sha256=canonical_sha256(score_receipt),
            synthetic=True,
        )
        print(
            f"status={analysis['status']} all_eight_resolved={analysis['all_eight_resolved']} "
            f"freeze_id={analysis['freeze_id']} out={args.out_dir}"
        )
        return

    if args.stage == "freeze":
        required = (
            args.attributions,
            args.feature_pool,
            args.endpoint_specs,
            args.provenance,
            args.selection_cohort,
            args.evaluation_cohort,
        )
        if not all(required):
            raise ValueError(
                "freeze requires --attributions, --feature-pool, --endpoint-specs, --provenance, "
                "--selection-cohort and --evaluation-cohort"
            )
        paths = [path for path in required if path is not None]
        provenance = read_json(args.provenance)
        selection_hash = sha256_file(args.selection_cohort)
        evaluation_hash = sha256_file(args.evaluation_cohort)
        if selection_hash != str(provenance.get("selection_cohort_sha256", "")):
            raise ValueError("selection cohort file hash does not match frozen provenance")
        if evaluation_hash != str(provenance.get("evaluation_cohort_sha256", "")):
            raise ValueError("evaluation cohort file hash does not match frozen provenance")
        cohort_validation = validate_disjoint_selection_evaluation_cohorts(
            read_jsonl(args.selection_cohort), read_jsonl(args.evaluation_cohort)
        )
        frozen = freeze_protocol(
            out_dir=args.out_dir,
            attribution_rows=read_jsonl(args.attributions),
            feature_pool=read_jsonl(args.feature_pool),
            endpoint_specs=read_json(args.endpoint_specs),
            provenance=provenance,
            analysis_spec=frozen_analysis_spec(args),
            args=args,
            input_hashes={str(path): sha256_file(path) for path in paths},
            cohort_validation=cohort_validation,
            synthetic=False,
            endpoint_artifact_base=args.endpoint_specs.parent,
        )
        print(
            f"status=generation_plan_frozen rows={len(frozen['plan_rows'])} "
            f"freeze_id={frozen['summary']['freeze_id']} out={args.out_dir}"
        )
        return

    if args.stage == "analyze":
        if not all(
            (
                args.frozen_dir,
                args.generation_outputs,
                args.generation_execution_receipt,
                args.generation_execution_receipt_sha256,
                args.scores,
                args.score_receipt,
                args.score_receipt_sha256,
            )
        ):
            raise ValueError(
                "analyze requires --frozen-dir, --generation-outputs, --scores, "
                "--generation-execution-receipt, --generation-execution-receipt-sha256, "
                "--score-receipt and --score-receipt-sha256"
            )
        input_paths = (
            args.generation_outputs,
            args.generation_execution_receipt,
            args.scores,
            args.score_receipt,
        )
        if len({path.resolve() for path in input_paths}) != len(input_paths):
            raise ValueError(
                "generation outputs, execution receipt, scores and score receipt "
                "must be separate files"
            )
        pinned_execution_receipt_sha256 = require_sha256(
            args.generation_execution_receipt_sha256,
            "generation_execution_receipt_sha256",
        )
        actual_execution_receipt_sha256 = sha256_file(
            args.generation_execution_receipt
        )
        if actual_execution_receipt_sha256 != pinned_execution_receipt_sha256:
            raise ValueError(
                "generation execution receipt file hash differs from "
                "--generation-execution-receipt-sha256"
            )
        pinned_receipt_sha256 = require_sha256(
            args.score_receipt_sha256, "score_receipt_sha256"
        )
        score_receipt, actual_receipt_sha256 = read_hashed_json(args.score_receipt)
        if actual_receipt_sha256 != pinned_receipt_sha256:
            raise ValueError("score receipt file hash differs from --score-receipt-sha256")
        outputs, generation_outputs_sha256 = read_hashed_jsonl(args.generation_outputs)
        scores, scores_sha256 = read_hashed_jsonl(args.scores)
        analysis = analyze_protocol(
            frozen_dir=args.frozen_dir,
            out_dir=args.out_dir,
            outputs=outputs,
            score_rows=scores,
            args=args,
            input_hashes={
                str(args.generation_outputs): generation_outputs_sha256,
                str(args.generation_execution_receipt): actual_execution_receipt_sha256,
                str(args.scores): scores_sha256,
                str(args.score_receipt): actual_receipt_sha256,
            },
            generation_outputs_sha256=generation_outputs_sha256,
            generation_outputs_path=args.generation_outputs,
            generation_execution_receipt_path=args.generation_execution_receipt,
            generation_execution_receipt_sha256=actual_execution_receipt_sha256,
            scores_sha256=scores_sha256,
            score_receipt=score_receipt,
            score_receipt_sha256=actual_receipt_sha256,
            synthetic=False,
        )
        print(
            f"status={analysis['status']} all_eight_resolved={analysis['all_eight_resolved']} "
            f"freeze_id={analysis['freeze_id']} out={args.out_dir}"
        )
        return
    raise ValueError("choose --stage freeze, --stage analyze, or --synthetic")


if __name__ == "__main__":
    main()
