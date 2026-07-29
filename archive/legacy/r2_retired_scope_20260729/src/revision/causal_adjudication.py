"""Receipt-bound P0-7 adjudication for pretrained-model interventions.

This module performs no model inference and no feature matching.  It verifies
immutable discovery, intervention and external-score artifacts, retains every
evaluation-unit pair, and adjudicates the frozen target/control factorial.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from .dictionary_gate import require_eligible_model_method
from .io import sha256_file, write_json, write_jsonl
from .statistics import benjamini_hochberg, mean_interval, tost_paired
from .steering_protocol import validate_disjoint_selection_evaluation_cohorts


SPEC_SCHEMA = "r2_p0_7_pretrained_causal_adjudication_spec_v1"
IDENTITY_RECEIPT_SCHEMA = "r2_p0_7_pretrained_identity_freeze_receipt_v1"
EVALUATION_RECEIPT_SCHEMA = "r2_p0_7_pretrained_intervention_evaluation_receipt_v1"
SUMMARY_SCHEMA = "r2_p0_7_pretrained_causal_adjudication_summary_v1"
RECEIPT_SCHEMA = "r2_p0_7_pretrained_causal_adjudication_receipt_v1"
POSITIVE_CONTROL_SCOPE = (
    "synthetic_pipeline_sensitivity_only_no_pretrained_causal_inference"
)
MEASURES = (
    "intended_feature_change",
    "off_target_sparse_code_displacement",
    "reconstruction_displacement",
    "logit_displacement",
    "behavior_endpoint",
    "path_endpoint",
)
PROFILE_FIELDS = (
    "firing_frequency",
    "mean_activation",
    "decoder_norm",
    "direct_logit_effect_norm",
    "received_attention_mass",
    "reconstruction_contribution",
)
ROLES = ("target", "control")
REQUIRED_BINDING_KINDS = {
    "model",
    "dictionary",
    "behavior_scorer",
    "behavior_calibration",
    "path_scorer",
    "path_calibration",
    "code",
}
HEX = frozenset("0123456789abcdef")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(
            handle,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _strict_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(
            line,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} must contain at least one row")
    return rows


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        observed = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{label} fields differ: missing={sorted(fields - observed)}, "
            f"extra={sorted(observed - fields)}"
        )
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= HEX:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or positive and number <= 0.0:
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return number


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve(path: Any, base: Path, label: str) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{label} path must be non-empty")
    candidate = Path(path).expanduser()
    return (
        candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    )


def _artifact(descriptor: Any, base: Path, label: str) -> dict[str, Any]:
    record = _exact(descriptor, {"path", "sha256"}, label)
    digest = _digest(record["sha256"], f"{label} SHA-256")
    path = _resolve(record["path"], base, label)
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"{label} is missing or its SHA-256 changed")
    return {"path": path, "sha256": digest}


def _rehash_colocated(receipt_path: Path, hashes: Any, label: str) -> dict[str, str]:
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"{label} artifact inventory must be non-empty")
    root = Path(receipt_path).resolve().parent
    verified = {}
    for name, expected in hashes.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} artifact names must be non-empty strings")
        relative = Path(name)
        path = (root / relative).resolve()
        digest = _digest(expected, f"{label} artifact {name}")
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"{label} artifact path escapes its receipt directory")
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"{label} artifact changed: {name}")
        verified[name] = digest
    return verified


def _validate_positive_control(
    descriptor: Mapping[str, Any], base: Path
) -> dict[str, Any]:
    record = _exact(dict(descriptor), {"run_manifest", "summary"}, "positive_control")
    manifest_file = _artifact(record["run_manifest"], base, "positive-control manifest")
    summary_file = _artifact(record["summary"], base, "positive-control summary")
    manifest = _strict_json(manifest_file["path"])
    summary = _strict_json(summary_file["path"])
    artifacts = manifest.get("artifact_hashes")
    if (
        manifest.get("schema_version")
        != "r2_p0_7_prospective_positive_control_manifest_v1"
        or manifest.get("status") != "complete"
        or manifest.get("claim_scope") != POSITIVE_CONTROL_SCOPE
        or not isinstance(artifacts, dict)
        or artifacts.get("summary.json") != summary_file["sha256"]
    ):
        raise ValueError(
            "positive-control manifest is not the complete production receipt"
        )
    if summary_file["path"] != manifest_file["path"].parent / "summary.json":
        raise ValueError("positive-control summary is not colocated with its manifest")
    _rehash_colocated(manifest_file["path"], artifacts, "positive-control")
    aggregate = summary.get("aggregate")
    if (
        summary.get("schema_version")
        != "r2_p0_7_prospective_positive_control_result_v2"
        or summary.get("status") != "prospective_synthetic_gate_passed"
        or summary.get("claim_scope") != POSITIVE_CONTROL_SCOPE
        or summary.get("pretrained_model_causal_inference") is not False
        or summary.get("legacy_controls_upgraded") is not False
        or not isinstance(aggregate, dict)
        or aggregate.get("prospective_synthetic_gate_passed") is not True
        or aggregate.get("all_paths_localized") is not True
        or aggregate.get("all_negative_metrics_equivalent") is not True
    ):
        raise ValueError("positive-control result did not exactly pass its frozen gate")
    for name, digest in manifest.get("source_hashes", {}).items():
        _digest(digest, f"positive-control source {name}")
    return {
        "run_manifest_path": manifest_file["path"],
        "run_manifest_sha256": manifest_file["sha256"],
        "summary_path": summary_file["path"],
        "summary_sha256": summary_file["sha256"],
        "status": summary["status"],
    }


def _validate_p0_2(descriptor: Any, base: Path) -> dict[str, Any]:
    value = _exact(descriptor, {"eligibility_receipt", "models"}, "p0_2")
    receipt = _artifact(value["eligibility_receipt"], base, "P0-2 eligibility receipt")
    models = value["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("P0-2 models must be a non-empty list")
    verified, names = [], set()
    fields = {
        "name",
        "method",
        "run_manifest_sha256_by_seed",
        "checkpoint_sha256_by_seed",
        "source_manifest_sha256_by_split",
        "requested_layers",
    }
    for index, source in enumerate(models):
        model = _exact(source, fields, f"P0-2 model {index}")
        name = model["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("P0-2 model names must be unique non-empty strings")
        names.add(name)
        if model["method"] != "topk_clt":
            raise ValueError("P0-7 accepts only P0-2-eligible topk_clt dictionaries")
        seed_manifests = {
            int(seed): _digest(digest, f"{name} run-manifest SHA-256")
            for seed, digest in model["run_manifest_sha256_by_seed"].items()
        }
        seed_checkpoints = {
            int(seed): _digest(digest, f"{name} checkpoint SHA-256")
            for seed, digest in model["checkpoint_sha256_by_seed"].items()
        }
        sources = {
            split: _digest(digest, f"{name} {split} source SHA-256")
            for split, digest in model["source_manifest_sha256_by_split"].items()
        }
        layers = model["requested_layers"]
        if (
            not isinstance(layers, list)
            or not layers
            or len(set(layers)) != len(layers)
            or any(type(layer) is not int or layer < 0 for layer in layers)
        ):
            raise ValueError(
                f"{name} requested_layers must be unique non-negative integers"
            )
        selected = require_eligible_model_method(
            receipt["path"],
            receipt["sha256"],
            model_name=name,
            method="topk_clt",
            expected_run_manifest_sha256_by_seed=seed_manifests,
            expected_checkpoint_sha256_by_seed=seed_checkpoints,
            expected_source_manifest_sha256_by_split=sources,
            requested_layers=layers,
        )
        verified.append(
            {
                "name": name,
                "requested_layers": list(layers),
                "run_manifest_sha256_by_seed": {
                    str(seed): digest for seed, digest in sorted(seed_manifests.items())
                },
                "checkpoint_sha256_by_seed": {
                    str(seed): digest
                    for seed, digest in sorted(seed_checkpoints.items())
                },
                "source_manifest_sha256_by_split": sources,
                "eligibility": selected,
            }
        )
    return {
        "receipt_path": receipt["path"],
        "receipt_sha256": receipt["sha256"],
        "models": verified,
    }


def _validate_upstream_receipt(
    descriptor: Any,
    base: Path,
    *,
    label: str,
    schema: str,
    status: str,
) -> dict[str, Any]:
    artifact = _artifact(descriptor, base, label)
    receipt = _strict_json(artifact["path"])
    if receipt.get("schema_version") != schema or receipt.get("status") != status:
        raise ValueError(f"{label} is not an eligible complete production receipt")
    hashes = receipt.get("artifact_hashes", receipt.get("artifacts"))
    _rehash_colocated(artifact["path"], hashes, label)
    return {**artifact, "receipt": receipt}


def _validate_profile(profile: Any, label: str) -> dict[str, float]:
    record = _exact(profile, set(PROFILE_FIELDS), label)
    output = {
        field: _finite(record[field], f"{label} {field}") for field in PROFILE_FIELDS
    }
    if output["firing_frequency"] < 0.0 or output["firing_frequency"] > 1.0:
        raise ValueError(f"{label} firing_frequency must lie in [0, 1]")
    if output["decoder_norm"] <= 0.0:
        raise ValueError(f"{label} decoder_norm must be positive")
    return output


def _cohort_identity(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for row in rows:
        protein_id = str(row.get("protein_id") or row.get("id") or "").strip()
        sequence = "".join(str(row.get("sequence", "")).upper().split())
        if not protein_id or protein_id in output or not sequence:
            raise ValueError(
                f"{label} cohort IDs and sequences must be unique and non-empty"
            )
        output[protein_id] = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
    return output


def _validate_identity_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = {
        "identity_set_id",
        "model",
        "dictionary_seed",
        "layer",
        "target_feature",
        "control_feature",
        "path_site",
        "target_profile",
        "control_profile",
        "matching_calipers_passed",
        "matching_distance",
    }
    output, identities = [], set()
    for index, source in enumerate(rows):
        row = _exact(source, fields, f"frozen identity row {index}")
        identity = row["identity_set_id"]
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError(
                "frozen identity_set_id values must be unique and non-empty"
            )
        identities.add(identity)
        if (
            not isinstance(row["model"], str)
            or not row["model"]
            or type(row["dictionary_seed"]) is not int
            or type(row["layer"]) is not int
            or type(row["target_feature"]) is not int
            or type(row["control_feature"]) is not int
            or min(
                row["dictionary_seed"],
                row["layer"],
                row["target_feature"],
                row["control_feature"],
            )
            < 0
            or row["target_feature"] == row["control_feature"]
            or not isinstance(row["path_site"], str)
            or not row["path_site"]
            or row["matching_calipers_passed"] is not True
        ):
            raise ValueError("frozen target/control identity is malformed or unmatched")
        output.append(
            {
                **row,
                "target_profile": _validate_profile(
                    row["target_profile"], f"identity {identity} target profile"
                ),
                "control_profile": _validate_profile(
                    row["control_profile"], f"identity {identity} control profile"
                ),
                "matching_distance": _finite(
                    row["matching_distance"], f"identity {identity} matching_distance"
                ),
            }
        )
    return output


def _validate_binding_artifacts(bindings: Any, base: Path) -> list[dict[str, str]]:
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("evaluation receipt requires artifact_bindings")
    output, names, kinds = [], set(), set()
    for index, source in enumerate(bindings):
        row = _exact(source, {"kind", "name", "path", "sha256"}, f"binding {index}")
        kind, name = row["kind"], row["name"]
        if (
            not isinstance(kind, str)
            or not kind
            or not isinstance(name, str)
            or not name
            or name in names
        ):
            raise ValueError(
                "artifact binding kinds/names must be non-empty and names unique"
            )
        artifact = _artifact(
            {"path": row["path"], "sha256": row["sha256"]},
            base,
            f"{kind} binding {name}",
        )
        names.add(name)
        kinds.add(kind)
        output.append(
            {
                "kind": kind,
                "name": name,
                "path": str(artifact["path"]),
                "sha256": artifact["sha256"],
            }
        )
    if not REQUIRED_BINDING_KINDS <= kinds:
        raise ValueError(
            "evaluation receipt lacks model/dictionary/scorer/calibration/code bindings"
        )
    return output


def _validate_scorers(
    value: Any, bindings: Sequence[Mapping[str, str]]
) -> dict[str, Any]:
    scorers = _exact(value, {"behavior_endpoint", "path_endpoint"}, "scorers")
    binding_digests = {(row["kind"], row["sha256"]) for row in bindings}
    output = {}
    for endpoint, required_kinds in {
        "behavior_endpoint": ("behavior_scorer", "behavior_calibration"),
        "path_endpoint": ("path_scorer", "path_calibration"),
    }.items():
        fields = {
            "validated",
            "name",
            "version",
            "method",
            "scorer_sha256",
            "calibration_cohort_sha256",
        }
        row = _exact(scorers[endpoint], fields, f"{endpoint} scorer")
        if (
            row["validated"] is not True
            or not isinstance(row["name"], str)
            or not row["name"]
            or not isinstance(row["version"], str)
            or not row["version"]
        ):
            raise ValueError(f"{endpoint} must be independently validated")
        method = row["method"]
        if endpoint == "path_endpoint" and method not in {
            "path_patching",
            "causal_mediation",
        }:
            raise ValueError("path endpoint must use path_patching or causal_mediation")
        if endpoint == "behavior_endpoint" and method != "validated_behavior_endpoint":
            raise ValueError(
                "behavior endpoint method differs from the frozen contract"
            )
        scorer_sha = _digest(row["scorer_sha256"], f"{endpoint} scorer SHA-256")
        calibration_sha = _digest(
            row["calibration_cohort_sha256"],
            f"{endpoint} calibration cohort SHA-256",
        )
        if (required_kinds[0], scorer_sha) not in binding_digests or (
            required_kinds[1],
            calibration_sha,
        ) not in binding_digests:
            raise ValueError(f"{endpoint} scorer/calibration bytes are not hash-bound")
        output[endpoint] = {
            **row,
            "scorer_sha256": scorer_sha,
            "calibration_cohort_sha256": calibration_sha,
        }
    return output


def _validate_p0_2_artifact_bindings(
    p0_2: Mapping[str, Any], bindings: Sequence[Mapping[str, str]]
) -> None:
    dictionary_digests = {
        row["sha256"] for row in bindings if row["kind"] == "dictionary"
    }
    required_dictionary_digests = {
        digest
        for model in p0_2["models"]
        for field in (
            "run_manifest_sha256_by_seed",
            "checkpoint_sha256_by_seed",
        )
        for digest in model[field].values()
    }
    if not required_dictionary_digests <= dictionary_digests:
        raise ValueError("artifact bindings omit a P0-2 run manifest or checkpoint")
    model_binding_names = {row["name"] for row in bindings if row["kind"] == "model"}
    if any(
        not any(
            name == model["name"] or name.startswith(f"{model['name']}:")
            for name in model_binding_names
        )
        for model in p0_2["models"]
    ):
        raise ValueError("artifact bindings omit a requested pretrained model")


def _validate_analysis(value: Any) -> dict[str, Any]:
    spec = _exact(
        value,
        {"alpha", "multiplicity", "path_localization", "measures"},
        "analysis",
    )
    alpha = _finite(spec["alpha"], "analysis alpha", positive=True)
    if alpha >= 0.5:
        raise ValueError("analysis alpha must be below 0.5")
    if spec["multiplicity"] != "benjamini_hochberg_all_paired_cells_positive_and_tost":
        raise ValueError("analysis multiplicity differs from the frozen global family")
    if spec["path_localization"] != "on_path_positive_off_path_equivalent":
        raise ValueError("unsupported path-localization criterion")
    measures = _exact(spec["measures"], set(MEASURES), "analysis measures")
    normalized = {}
    for measure in MEASURES:
        row = _exact(
            measures[measure],
            {"direction", "equivalence_margin"},
            f"analysis measure {measure}",
        )
        if row["direction"] not in {"higher", "lower"}:
            raise ValueError(f"{measure} direction must be higher or lower")
        normalized[measure] = {
            "direction": row["direction"],
            "equivalence_margin": _finite(
                row["equivalence_margin"],
                f"{measure} equivalence margin",
                positive=True,
            ),
        }
    return {
        "alpha": alpha,
        "multiplicity": spec["multiplicity"],
        "path_localization": spec["path_localization"],
        "measures": normalized,
    }


def load_adjudication_spec(spec_path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load and validate the immutable P0-7 adjudication specification."""

    path = Path(spec_path).resolve()
    expected = _digest(expected_sha256, "adjudication spec SHA-256")
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError("adjudication spec is missing or its SHA-256 changed")
    raw = _exact(
        _strict_json(path),
        {
            "schema_version",
            "mode",
            "positive_control",
            "p0_2",
            "upstream_receipts",
            "artifacts",
            "factorial",
            "analysis",
        },
        "adjudication spec",
    )
    if raw["schema_version"] != SPEC_SCHEMA or raw["mode"] != "production":
        raise ValueError("P0-7 adjudication requires the production v1 schema")
    factorial = _exact(
        raw["factorial"], {"identity_set_ids", "sites", "strengths"}, "factorial"
    )
    identity_ids = factorial["identity_set_ids"]
    sites = factorial["sites"]
    strengths = factorial["strengths"]
    if (
        not isinstance(identity_ids, list)
        or not identity_ids
        or len(set(identity_ids)) != len(identity_ids)
        or any(not isinstance(value, str) or not value for value in identity_ids)
        or not isinstance(sites, list)
        or len(sites) < 2
        or len(set(sites)) != len(sites)
        or any(not isinstance(value, str) or not value for value in sites)
        or not isinstance(strengths, list)
        or not strengths
    ):
        raise ValueError(
            "factorial identities/sites/strengths are incomplete or duplicated"
        )
    normalized_strengths = [
        _finite(value, "factorial strength", positive=True) for value in strengths
    ]
    if len(set(normalized_strengths)) != len(normalized_strengths):
        raise ValueError("factorial strengths must be unique")
    return {
        **raw,
        "path": path,
        "sha256": expected,
        "factorial": {
            "identity_set_ids": list(identity_ids),
            "sites": list(sites),
            "strengths": normalized_strengths,
        },
        "analysis": _validate_analysis(raw["analysis"]),
    }


def load_bound_inputs(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Verify all immutable prerequisites and load the raw held-out evidence."""

    base = Path(spec["path"]).parent
    positive = _validate_positive_control(spec["positive_control"], base)
    p0_2 = _validate_p0_2(spec["p0_2"], base)
    upstream = _exact(
        spec["upstream_receipts"],
        {"p0_5_extraction", "p0_6_execution"},
        "upstream_receipts",
    )
    p0_5 = _validate_upstream_receipt(
        upstream["p0_5_extraction"],
        base,
        label="P0-5 extraction receipt",
        schema="r2-p05-pretrained-extraction-manifest-v4",
        status="verified_production_complete",
    )
    if p0_5["receipt"].get("execution_mode") != "production":
        raise ValueError("P0-5 extraction receipt is not production mode")
    p0_6 = _validate_upstream_receipt(
        upstream["p0_6_execution"],
        base,
        label="P0-6 execution receipt",
        schema="r2-corrected-steering-execution-receipt-v3",
        status="verified_complete",
    )
    artifacts = _exact(
        spec["artifacts"],
        {
            "identity_freeze_receipt",
            "evaluation_receipt",
            "discovery_cohort",
            "evaluation_cohort",
            "frozen_feature_identities",
            "intervention_rows",
            "external_scores",
        },
        "artifacts",
    )
    files = {
        name: _artifact(descriptor, base, name)
        for name, descriptor in artifacts.items()
    }
    identity_receipt = _exact(
        _strict_json(files["identity_freeze_receipt"]["path"]),
        {
            "schema_version",
            "status",
            "created_at_utc",
            "p0_2_eligibility_receipt_sha256",
            "p0_5_extraction_receipt_sha256",
            "discovery_cohort_sha256",
            "evaluation_cohort_sha256",
            "frozen_feature_identities_sha256",
            "analysis_contract_sha256",
        },
        "identity-freeze receipt",
    )
    analysis_contract_sha = _canonical_sha256(
        {"factorial": spec["factorial"], "analysis": spec["analysis"]}
    )
    if (
        identity_receipt["schema_version"] != IDENTITY_RECEIPT_SCHEMA
        or identity_receipt["status"] != "frozen_before_evaluation"
        or identity_receipt["p0_2_eligibility_receipt_sha256"] != p0_2["receipt_sha256"]
        or identity_receipt["p0_5_extraction_receipt_sha256"] != p0_5["sha256"]
        or identity_receipt["discovery_cohort_sha256"]
        != files["discovery_cohort"]["sha256"]
        or identity_receipt["evaluation_cohort_sha256"]
        != files["evaluation_cohort"]["sha256"]
        or identity_receipt["frozen_feature_identities_sha256"]
        != files["frozen_feature_identities"]["sha256"]
        or identity_receipt["analysis_contract_sha256"] != analysis_contract_sha
    ):
        raise ValueError(
            "identity-freeze receipt differs from the frozen P0-7 contract"
        )
    evaluation_receipt = _exact(
        _strict_json(files["evaluation_receipt"]["path"]),
        {
            "schema_version",
            "status",
            "completed_at_utc",
            "identity_freeze_receipt_sha256",
            "p0_2_eligibility_receipt_sha256",
            "p0_5_extraction_receipt_sha256",
            "p0_6_execution_receipt_sha256",
            "discovery_cohort_sha256",
            "evaluation_cohort_sha256",
            "frozen_feature_identities_sha256",
            "intervention_rows_sha256",
            "external_scores_sha256",
            "artifact_bindings",
            "scorers",
        },
        "evaluation receipt",
    )
    expected_evaluation = {
        "schema_version": EVALUATION_RECEIPT_SCHEMA,
        "status": "verified_complete",
        "identity_freeze_receipt_sha256": files["identity_freeze_receipt"]["sha256"],
        "p0_2_eligibility_receipt_sha256": p0_2["receipt_sha256"],
        "p0_5_extraction_receipt_sha256": p0_5["sha256"],
        "p0_6_execution_receipt_sha256": p0_6["sha256"],
        "discovery_cohort_sha256": files["discovery_cohort"]["sha256"],
        "evaluation_cohort_sha256": files["evaluation_cohort"]["sha256"],
        "frozen_feature_identities_sha256": files["frozen_feature_identities"][
            "sha256"
        ],
        "intervention_rows_sha256": files["intervention_rows"]["sha256"],
        "external_scores_sha256": files["external_scores"]["sha256"],
    }
    if any(
        evaluation_receipt.get(key) != value
        for key, value in expected_evaluation.items()
    ):
        raise ValueError(
            "evaluation receipt differs from the immutable artifact contract"
        )
    bindings = _validate_binding_artifacts(
        evaluation_receipt["artifact_bindings"], base
    )
    _validate_p0_2_artifact_bindings(p0_2, bindings)
    scorers = _validate_scorers(evaluation_receipt["scorers"], bindings)
    discovery_rows = _strict_jsonl(files["discovery_cohort"]["path"])
    evaluation_rows = _strict_jsonl(files["evaluation_cohort"]["path"])
    disjointness = validate_disjoint_selection_evaluation_cohorts(
        discovery_rows, evaluation_rows
    )
    identities = _validate_identity_rows(
        _strict_jsonl(files["frozen_feature_identities"]["path"])
    )
    return {
        "positive_control": positive,
        "p0_2": p0_2,
        "p0_5": p0_5,
        "p0_6": p0_6,
        "files": files,
        "identity_receipt": identity_receipt,
        "evaluation_receipt": evaluation_receipt,
        "bindings": bindings,
        "scorers": scorers,
        "discovery_rows": discovery_rows,
        "evaluation_rows": evaluation_rows,
        "evaluation_identity": _cohort_identity(evaluation_rows, "evaluation"),
        "disjointness": disjointness,
        "identities": identities,
        "intervention_rows": _strict_jsonl(files["intervention_rows"]["path"]),
        "external_scores": _strict_jsonl(files["external_scores"]["path"]),
    }


def validate_factorial_inventory(
    spec: Mapping[str, Any], inputs: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Require the complete raw evaluation factorial and exact score coverage."""

    identities = {row["identity_set_id"]: row for row in inputs["identities"]}
    if set(identities) != set(spec["factorial"]["identity_set_ids"]):
        raise ValueError("frozen identity inventory differs from the adjudication spec")
    p0_2_models = {row["name"]: row for row in inputs["p0_2"]["models"]}
    p0_2_seeds = {
        name: {int(seed) for seed in row["checkpoint_sha256_by_seed"]}
        for name, row in p0_2_models.items()
    }
    for identity in identities.values():
        model = identity["model"]
        if (
            model not in p0_2_models
            or identity["layer"] not in p0_2_models[model]["requested_layers"]
            or identity["dictionary_seed"] not in p0_2_seeds[model]
            or identity["path_site"] not in spec["factorial"]["sites"]
        ):
            raise ValueError("frozen identity is outside the P0-2/factorial allowlist")
    evaluation = inputs["evaluation_identity"]
    if len(evaluation) < 2:
        raise ValueError(
            "paired P0-7 inference requires at least two evaluation identities"
        )
    row_fields = {
        "row_id",
        "evaluation_id",
        "sequence_sha256",
        "identity_set_id",
        "model",
        "dictionary_seed",
        "layer",
        "feature_role",
        "feature",
        "site",
        "strength",
        "intended_feature_change",
        "off_target_sparse_code_displacement",
        "reconstruction_displacement",
        "logit_displacement",
    }
    observed: dict[tuple[Any, ...], dict[str, Any]] = {}
    row_ids = set()
    for index, source in enumerate(inputs["intervention_rows"]):
        row = _exact(source, row_fields, f"intervention row {index}")
        row_id = row["row_id"]
        evaluation_id = row["evaluation_id"]
        identity_id = row["identity_set_id"]
        role = row["feature_role"]
        if (
            not isinstance(row_id, str)
            or not row_id
            or row_id in row_ids
            or evaluation_id not in evaluation
            or identity_id not in identities
            or role not in ROLES
        ):
            raise ValueError(
                "intervention row has duplicate/unknown identities or role"
            )
        identity = identities[identity_id]
        expected_feature = identity[f"{role}_feature"]
        strength = _finite(row["strength"], "intervention strength", positive=True)
        if (
            row["sequence_sha256"] != evaluation[evaluation_id]
            or row["model"] != identity["model"]
            or row["dictionary_seed"] != identity["dictionary_seed"]
            or row["layer"] != identity["layer"]
            or row["feature"] != expected_feature
            or row["site"] not in spec["factorial"]["sites"]
            or strength not in spec["factorial"]["strengths"]
        ):
            raise ValueError("intervention row substituted a frozen identity/factor")
        normalized = {
            **row,
            "strength": strength,
            "path_site": identity["path_site"],
            **{
                measure: _finite(row[measure], f"row {row_id} {measure}")
                for measure in MEASURES[:4]
            },
        }
        key = (evaluation_id, identity_id, role, row["site"], strength)
        if key in observed:
            raise ValueError("duplicate intervention factorial cell")
        observed[key] = normalized
        row_ids.add(row_id)
    expected = {
        (evaluation_id, identity_id, role, site, strength)
        for evaluation_id in evaluation
        for identity_id in spec["factorial"]["identity_set_ids"]
        for role in ROLES
        for site in spec["factorial"]["sites"]
        for strength in spec["factorial"]["strengths"]
    }
    if set(observed) != expected:
        raise ValueError("intervention rows do not cover the exact frozen factorial")
    score_fields = {"row_id", "behavior_endpoint", "path_endpoint"}
    scores: dict[str, dict[str, float]] = {}
    for index, source in enumerate(inputs["external_scores"]):
        row = _exact(source, score_fields, f"external score row {index}")
        row_id = row["row_id"]
        if not isinstance(row_id, str) or row_id in scores:
            raise ValueError("external score row IDs must be unique strings")
        scores[row_id] = {
            endpoint: _finite(row[endpoint], f"score {row_id} {endpoint}")
            for endpoint in ("behavior_endpoint", "path_endpoint")
        }
    if set(scores) != row_ids:
        raise ValueError(
            "external scores do not cover every intervention row exactly once"
        )
    return [
        {**row, **scores[row["row_id"]]}
        for _, row in sorted(observed.items(), key=lambda item: str(item[0]))
    ]


def _positive_pvalue(values: Sequence[float]) -> float:
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or sample.size < 2 or not np.isfinite(sample).all():
        raise ValueError("paired effects must be finite with at least two units")
    if np.all(sample == sample[0]):
        return 0.0 if sample[0] > 0.0 else 1.0
    return float(stats.ttest_1samp(sample, 0.0, alternative="greater").pvalue)


def analyze_paired_cells(
    spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Pair target/control rows before any averaging and adjudicate every cell."""

    lookup = {
        (
            row["evaluation_id"],
            row["identity_set_id"],
            row["feature_role"],
            row["site"],
            float(row["strength"]),
        ): row
        for row in rows
    }
    paired_rows, cells = [], []
    evaluation_ids = sorted({str(row["evaluation_id"]) for row in rows})
    for identity_id in spec["factorial"]["identity_set_ids"]:
        for site in spec["factorial"]["sites"]:
            for strength in spec["factorial"]["strengths"]:
                pairs = []
                for evaluation_id in evaluation_ids:
                    target = lookup[
                        (evaluation_id, identity_id, "target", site, strength)
                    ]
                    control = lookup[
                        (evaluation_id, identity_id, "control", site, strength)
                    ]
                    effects = {
                        measure: float(target[measure]) - float(control[measure])
                        for measure in MEASURES
                    }
                    pair = {
                        "evaluation_id": evaluation_id,
                        "sequence_sha256": target["sequence_sha256"],
                        "identity_set_id": identity_id,
                        "model": target["model"],
                        "dictionary_seed": target["dictionary_seed"],
                        "layer": target["layer"],
                        "target_feature": target["feature"],
                        "control_feature": control["feature"],
                        "site": site,
                        "strength": strength,
                        "target_row_id": target["row_id"],
                        "control_row_id": control["row_id"],
                        "target_values": {
                            measure: float(target[measure]) for measure in MEASURES
                        },
                        "control_values": {
                            measure: float(control[measure]) for measure in MEASURES
                        },
                        "target_minus_control": effects,
                    }
                    paired_rows.append(pair)
                    pairs.append(pair)
                for measure in MEASURES:
                    endpoint = spec["analysis"]["measures"][measure]
                    sign = 1.0 if endpoint["direction"] == "higher" else -1.0
                    differences = [
                        sign * pair["target_minus_control"][measure] for pair in pairs
                    ]
                    cells.append(
                        {
                            "identity_set_id": identity_id,
                            "model": pairs[0]["model"],
                            "dictionary_seed": pairs[0]["dictionary_seed"],
                            "layer": pairs[0]["layer"],
                            "target_feature": pairs[0]["target_feature"],
                            "control_feature": pairs[0]["control_feature"],
                            "site": site,
                            "strength": strength,
                            "measure": measure,
                            "direction": endpoint["direction"],
                            "evaluation_ids": evaluation_ids,
                            "paired_directional_differences": differences,
                            "effect": mean_interval(differences),
                            "positive_p": _positive_pvalue(differences),
                            "equivalence": tost_paired(
                                differences,
                                endpoint["equivalence_margin"],
                                alpha=spec["analysis"]["alpha"],
                            ),
                        }
                    )
    family_size = len(cells)
    adjusted = benjamini_hochberg(
        [cell["positive_p"] for cell in cells]
        + [cell["equivalence"]["p_lower"] for cell in cells]
        + [cell["equivalence"]["p_upper"] for cell in cells]
    )
    alpha = spec["analysis"]["alpha"]
    for index, cell in enumerate(cells):
        q_positive = float(adjusted[index])
        q_lower = float(adjusted[family_size + index])
        q_upper = float(adjusted[2 * family_size + index])
        cell["positive_q_bh"] = float(q_positive)
        cell["equivalence"]["p_lower_q_bh"] = q_lower
        cell["equivalence"]["p_upper_q_bh"] = q_upper
        cell["equivalence"]["p_tost_q_bh"] = max(q_lower, q_upper)
        cell["equivalence"]["equivalent_bh"] = bool(q_lower < alpha and q_upper < alpha)
        if cell["effect"]["mean"] > 0.0 and q_positive < alpha:
            cell["status"] = "positive"
        elif cell["equivalence"]["equivalent_bh"]:
            cell["status"] = "equivalent"
        else:
            cell["status"] = "inconclusive"
    cell_lookup = {
        (cell["identity_set_id"], cell["site"], cell["strength"], cell["measure"]): cell
        for cell in cells
    }
    path_results = []
    identity_lookup = {row["identity_set_id"]: row for row in rows}
    for identity_id in spec["factorial"]["identity_set_ids"]:
        path_site = identity_lookup[identity_id]["path_site"]
        for strength in spec["factorial"]["strengths"]:
            on_path = cell_lookup[(identity_id, path_site, strength, "path_endpoint")]
            off_path = [
                cell_lookup[(identity_id, site, strength, "path_endpoint")]
                for site in spec["factorial"]["sites"]
                if site != path_site
            ]
            behavior = cell_lookup[
                (identity_id, path_site, strength, "behavior_endpoint")
            ]
            fidelity = cell_lookup[
                (identity_id, path_site, strength, "intended_feature_change")
            ]
            off_target = cell_lookup[
                (
                    identity_id,
                    path_site,
                    strength,
                    "off_target_sparse_code_displacement",
                )
            ]
            reconstruction = cell_lookup[
                (identity_id, path_site, strength, "reconstruction_displacement")
            ]
            logit = cell_lookup[
                (identity_id, path_site, strength, "logit_displacement")
            ]
            off_path_behavior_logit = [
                cell_lookup[(identity_id, site, strength, measure)]
                for site in spec["factorial"]["sites"]
                if site != path_site
                for measure in ("behavior_endpoint", "logit_displacement")
            ]
            localized = bool(
                on_path["status"] == "positive"
                and off_path
                and all(cell["status"] == "equivalent" for cell in off_path)
            )
            fidelity_passed = bool(
                fidelity["status"] == "positive"
                and off_target["status"] in {"positive", "equivalent"}
                and reconstruction["status"] in {"positive", "equivalent"}
            )
            off_path_equivalent = bool(
                off_path_behavior_logit
                and all(
                    cell["status"] == "equivalent" for cell in off_path_behavior_logit
                )
            )
            if (
                localized
                and fidelity_passed
                and off_path_equivalent
                and logit["status"] == "positive"
                and behavior["status"] == "positive"
            ):
                status = "localized_positive"
            elif (
                fidelity_passed
                and off_path_equivalent
                and logit["status"] == "equivalent"
                and behavior["status"] == "equivalent"
                and on_path["status"] == "equivalent"
                and all(cell["status"] == "equivalent" for cell in off_path)
            ):
                status = "equivalence_bounded_negative"
            else:
                status = "inconclusive"
            path_results.append(
                {
                    "identity_set_id": identity_id,
                    "strength": strength,
                    "path_site": path_site,
                    "off_path_sites": [cell["site"] for cell in off_path],
                    "path_localized": localized,
                    "intervention_fidelity_passed": fidelity_passed,
                    "off_path_behavior_and_logit_equivalent": off_path_equivalent,
                    "status": status,
                }
            )
    return {
        "paired_rows": paired_rows,
        "cells": cells,
        "path_results": path_results,
        "multiplicity_family_size": 3 * len(cells),
        "all_required_contrasts_resolved": bool(
            path_results
            and all(row["status"] != "inconclusive" for row in path_results)
        ),
    }


def _input_hashes(inputs: Mapping[str, Any]) -> dict[str, str]:
    hashes = {
        "positive_control_run_manifest": sha256_file(
            inputs["positive_control"]["run_manifest_path"]
        ),
        "positive_control_summary": sha256_file(
            inputs["positive_control"]["summary_path"]
        ),
        "p0_2_eligibility_receipt": sha256_file(inputs["p0_2"]["receipt_path"]),
        "p0_5_extraction_receipt": sha256_file(inputs["p0_5"]["path"]),
        "p0_6_execution_receipt": sha256_file(inputs["p0_6"]["path"]),
        **{
            name: sha256_file(record["path"])
            for name, record in inputs["files"].items()
        },
        **{
            f"bound:{row['name']}": sha256_file(Path(row["path"]))
            for row in inputs["bindings"]
        },
    }
    colocated = (
        (
            "positive_control",
            inputs["positive_control"]["run_manifest_path"],
            "artifact_hashes",
        ),
        ("p0_5", inputs["p0_5"]["path"], "artifact_hashes"),
        ("p0_6", inputs["p0_6"]["path"], "artifacts"),
    )
    for prefix, receipt_path, field in colocated:
        receipt = _strict_json(Path(receipt_path))
        verified = _rehash_colocated(
            Path(receipt_path), receipt.get(field), f"{prefix} rehash"
        )
        hashes.update(
            {f"{prefix}_artifact:{name}": digest for name, digest in verified.items()}
        )
    return hashes


def adjudicate_pretrained_causal_interventions(
    spec_path: Path,
    expected_sha256: str,
    output_dir: Path,
    *,
    cli_path: Path | None = None,
    command: Sequence[str] | None = None,
) -> Path:
    """Verify, analyze and atomically publish one complete P0-7 adjudication."""

    source_paths = [Path(__file__).resolve()]
    if cli_path is not None:
        source_paths.append(Path(cli_path).resolve())
    initial_source_hashes = {path.name: sha256_file(path) for path in source_paths}
    spec = load_adjudication_spec(spec_path, expected_sha256)
    inputs = load_bound_inputs(spec)
    rows = validate_factorial_inventory(spec, inputs)
    initial_hashes = _input_hashes(inputs)
    analysis = analyze_paired_cells(spec, rows)
    output = Path(output_dir).resolve()
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite P0-7 adjudication: {output}")
    if staging.exists():
        raise FileExistsError(f"stale P0-7 staging directory: {staging}")
    staging.mkdir(parents=True)
    try:
        paired_path = staging / "paired_rows.jsonl"
        cells_path = staging / "cell_results.jsonl"
        summary_path = staging / "summary.json"
        write_jsonl(paired_path, analysis["paired_rows"])
        write_jsonl(cells_path, analysis["cells"])
        scientific_status = (
            "pretrained_target_gate_resolved"
            if analysis["all_required_contrasts_resolved"]
            else "pretrained_target_gate_inconclusive"
        )
        write_json(
            summary_path,
            {
                "schema_version": SUMMARY_SCHEMA,
                "status": scientific_status,
                "claim_boundary": (
                    "A complete adjudication does not establish a biological primitive; "
                    "only path-localized positives or prespecified equivalence-bounded "
                    "negatives are resolved cells."
                ),
                "spec_sha256": spec["sha256"],
                "positive_control_status": inputs["positive_control"]["status"],
                "p0_2_models": [row["name"] for row in inputs["p0_2"]["models"]],
                "disjointness": inputs["disjointness"],
                "n_raw_intervention_rows": len(rows),
                "n_raw_paired_rows": len(analysis["paired_rows"]),
                "n_cells": len(analysis["cells"]),
                "multiplicity_family_size": analysis["multiplicity_family_size"],
                "all_required_contrasts_resolved": analysis[
                    "all_required_contrasts_resolved"
                ],
                "path_results": analysis["path_results"],
            },
        )
        artifact_hashes = {
            path.name: sha256_file(path)
            for path in (paired_path, cells_path, summary_path)
        }
        if (
            _input_hashes(inputs) != initial_hashes
            or sha256_file(spec["path"]) != spec["sha256"]
            or {path.name: sha256_file(path) for path in source_paths}
            != initial_source_hashes
        ):
            raise RuntimeError("P0-7 inputs or source changed during adjudication")
        receipt_path = staging / "completion_receipt.json"
        write_json(
            receipt_path,
            {
                "schema_version": RECEIPT_SCHEMA,
                "status": "verified_complete",
                "scientific_status": scientific_status,
                "claim_boundary": "Completion is not itself a positive causal result.",
                "command": list(command or [sys.executable]),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "spec": {"path": str(spec["path"]), "sha256": spec["sha256"]},
                "input_hashes": initial_hashes,
                "source_hashes": initial_source_hashes,
                "artifact_hashes": artifact_hashes,
                "raw_rows_retained_before_cell_inference": True,
                "positive_and_tost_multiplicity_corrected": True,
                "path_localization_required": True,
            },
        )
        if (
            _input_hashes(inputs) != initial_hashes
            or sha256_file(spec["path"]) != spec["sha256"]
            or {path.name: sha256_file(path) for path in source_paths}
            != initial_source_hashes
            or any(
                sha256_file(staging / name) != digest
                for name, digest in artifact_hashes.items()
            )
        ):
            raise RuntimeError("P0-7 final publication rehash failed")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "completion_receipt.json"
