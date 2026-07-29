"""Fail-closed contracts for the prospective corrected P0-6 steering rerun."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .io import sha256_file, validate_provenance as validate_provenance
from .statistics import tost_paired


EC_PROMPTS = {
    "lysozyme": "3.2.1.17",
    "trypsin": "3.4.21.4",
    "ADH": "1.1.1.1",
    "catalase": "1.11.1.6",
    "DNA_polymerase": "2.7.7.7",
    "lipase": "3.1.1.3",
    "kinase": "2.7.11.1",
    "carbonic_anh": "4.2.1.1",
}
AA = set("ACDEFGHIKLMNPQRSTVWY")
PLAN_OUTPUT_FIELDS = {
    "sequence",
    "sequence_sha256",
    "token_ids",
    "token_ids_sha256",
    "stop_reason",
    "runtime",
}
RUNTIME_FIELDS = {
    "generator_revision",
    "model_revision",
    "tokenizer_revision",
    "clt_checkpoint_sha256",
    "hostname",
    "device",
    "started_at_utc",
    "elapsed_seconds",
    "evaluation_mode",
    "hook_site",
    "multiplier_semantics",
    "rng_stream_id",
}
SCORE_RECEIPT_FIELDS = {
    "schema_version",
    "status",
    "synthetic",
    "freeze_id",
    "generation_outputs_sha256",
    "generation_execution_receipt_sha256",
    "scores_sha256",
    "frozen_endpoint_specs_sha256",
    "scorer_executions",
}
SCORER_EXECUTION_FIELDS = {
    "endpoint",
    "status",
    "execution_mode",
    "validated",
    "primary",
    "scorer_name",
    "scorer_version",
    "scorer_path",
    "scorer_sha256",
    "calibration_cohort_path",
    "calibration_cohort_sha256",
    "experimental_unit",
    "expected_unit_count",
    "scored_unit_count",
    "expected_coverage_sha256",
    "scored_coverage_sha256",
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: object, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _derived_seed(*parts: object) -> int:
    return int.from_bytes(
        hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()[:8],
        "little",
    )


def _token_ids_sha256(token_ids: Sequence[int]) -> tuple[list[int], str]:
    normalized = []
    for token_id in token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, (int, np.integer)):
            raise ValueError("raw token_ids must contain integers")
        if int(token_id) < 0:
            raise ValueError("raw token_ids must be non-negative")
        normalized.append(int(token_id))
    if not normalized:
        raise ValueError("raw token_ids must be non-empty")
    return normalized, _canonical_sha256(normalized)


def _plan_content(row: Mapping[str, object]) -> dict:
    return {key: value for key, value in row.items() if key not in PLAN_OUTPUT_FIELDS | {"plan_id", "generation_set_id"}}


def _plan_id(row: Mapping[str, object]) -> str:
    return _canonical_sha256({"schema": "r2-corrected-steering-plan-row-v2", "row": _plan_content(row)})


def validate_endpoint_specs(
    endpoint_specs: Mapping[str, Mapping[str, object]],
    *,
    evaluation_cohort_sha256: str | None = None,
    artifact_base: Path | None = None,
    require_artifacts: bool = False,
) -> dict[str, dict]:
    """Validate frozen endpoints, including scorer/calibration provenance."""

    if not endpoint_specs:
        raise ValueError("at least one frozen endpoint specification is required")
    evaluation_hash = (
        _valid_sha256(evaluation_cohort_sha256, "evaluation_cohort_sha256")
        if evaluation_cohort_sha256 is not None
        else None
    )
    normalized = {}
    for endpoint, source in endpoint_specs.items():
        name = str(endpoint).strip()
        if not name or not isinstance(source, Mapping):
            raise ValueError("endpoint names must be non-empty and specifications must be objects")
        spec = dict(source)
        margin = float(spec.get("equivalence_margin", 0.0))
        if not np.isfinite(margin) or margin <= 0.0:
            raise ValueError(f"endpoint {name} lacks a positive frozen equivalence margin")
        if spec.get("direction") not in {"higher", "lower"}:
            raise ValueError(f"endpoint {name} direction must be higher or lower")
        if spec.get("experimental_unit") not in {"generation", "generation_set"}:
            raise ValueError(
                f"endpoint {name} experimental_unit must be generation or generation_set"
            )
        for flag in ("validated", "primary"):
            if flag in spec and type(spec[flag]) is not bool:
                raise ValueError(f"endpoint {name} {flag} must be a JSON boolean")
        spec["equivalence_margin"] = margin
        spec["validated"] = bool(spec.get("validated", False))
        spec["primary"] = bool(spec.get("primary", False))
        if spec["primary"] and not spec["validated"]:
            raise ValueError(f"primary endpoint {name} must be independently validated")
        scorer_identity = ("scorer_name", "scorer_version", "scorer_sha256")
        present_identity = [field in spec for field in scorer_identity]
        if any(present_identity) and not all(present_identity):
            raise ValueError(
                f"endpoint {name} must freeze all or none of scorer name/version/artifact SHA"
            )
        if all(present_identity):
            if not str(spec["scorer_name"]).strip() or not str(spec["scorer_version"]).strip():
                raise ValueError(f"endpoint {name} scorer identity must be non-empty")
            spec["scorer_name"] = str(spec["scorer_name"]).strip()
            spec["scorer_version"] = str(spec["scorer_version"]).strip()
            spec["scorer_sha256"] = _valid_sha256(
                spec["scorer_sha256"], f"endpoint {name} scorer_sha256"
            )
        if spec["validated"]:
            required = {
                "scorer_name",
                "scorer_version",
                "scorer_sha256",
                "calibration_cohort_sha256",
            }
            missing = required - set(spec)
            if missing:
                raise ValueError(
                    f"validated endpoint {name} lacks scorer/calibration provenance: {sorted(missing)}"
                )
            spec["calibration_cohort_sha256"] = _valid_sha256(
                spec["calibration_cohort_sha256"],
                f"endpoint {name} calibration_cohort_sha256",
            )
            if evaluation_hash and spec["calibration_cohort_sha256"] == evaluation_hash:
                raise ValueError(
                    f"validated endpoint {name} calibration and evaluation cohorts must differ"
                )
            artifact_fields = ("scorer_path", "calibration_cohort_path")
            present_artifacts = [field in spec for field in artifact_fields]
            if require_artifacts and not all(present_artifacts):
                raise ValueError(
                    f"validated endpoint {name} requires real scorer and calibration artifact paths"
                )
            if any(present_artifacts) and not all(present_artifacts):
                raise ValueError(
                    f"endpoint {name} must provide both scorer and calibration artifact paths"
                )
            if all(present_artifacts):
                base = Path(artifact_base or ".").expanduser().resolve()
                bindings = (
                    ("scorer_path", "scorer_sha256", "scorer artifact"),
                    (
                        "calibration_cohort_path",
                        "calibration_cohort_sha256",
                        "calibration cohort",
                    ),
                )
                for path_field, digest_field, label in bindings:
                    raw_path = spec[path_field]
                    if not isinstance(raw_path, str) or not raw_path.strip():
                        raise ValueError(f"endpoint {name} {path_field} must be non-empty")
                    candidate = Path(raw_path).expanduser()
                    path = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
                    if not path.is_file() or sha256_file(path) != spec[digest_field]:
                        raise ValueError(f"endpoint {name} {label} path or SHA-256 mismatch")
                    spec[path_field] = str(path)
        normalized[name] = spec
    if not any(spec["validated"] and spec["primary"] for spec in normalized.values()):
        raise ValueError("at least one independently validated primary endpoint is required")
    return normalized


def validate_analysis_spec(spec: Mapping[str, object]) -> dict:
    """Freeze inference, equivalence and decision rules before generation."""

    alpha = float(spec.get("alpha", 0.0))
    n_resamples = int(spec.get("n_resamples", 0))
    random_seed = int(spec.get("random_seed", -1))
    if not 0.0 < alpha < 0.5 or n_resamples < 100:
        raise ValueError("analysis requires alpha in (0, 0.5) and at least 100 resamples")
    if random_seed < 0:
        raise ValueError("analysis random_seed must be a non-negative integer")
    multiplicity = str(spec.get("multiplicity", ""))
    decision_rule = str(spec.get("decision_rule", ""))
    if multiplicity != "holm_all_arm_and_specificity_cells":
        raise ValueError("multiplicity must be holm_all_arm_and_specificity_cells")
    if decision_rule != "target_vs_prompt_and_both_controls":
        raise ValueError("decision_rule must be target_vs_prompt_and_both_controls")
    return {
        "alpha": alpha,
        "n_resamples": n_resamples,
        "random_seed": random_seed,
        "multiplicity": multiplicity,
        "decision_rule": decision_rule,
    }


def validate_disjoint_selection_evaluation_cohorts(
    selection_records: Sequence[Mapping[str, object]],
    evaluation_records: Sequence[Mapping[str, object]],
) -> dict:
    """Require exact ID- and sequence-disjoint selection/evaluation cohorts."""

    def identities(records: Sequence[Mapping[str, object]], label: str) -> tuple[set[str], set[str]]:
        protein_ids, sequence_hashes = set(), set()
        for source in records:
            protein_id = str(source.get("protein_id") or source.get("id") or "").strip()
            sequence = "".join(str(source.get("sequence", "")).upper().split())
            if not protein_id or protein_id in protein_ids:
                raise ValueError(f"{label} cohort protein IDs must be non-empty and unique")
            if not sequence or any(residue not in AA for residue in sequence):
                raise ValueError(f"{label} cohort sequences must be canonical amino acids")
            sequence_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
            if sequence_hash in sequence_hashes:
                raise ValueError(f"{label} cohort contains duplicate exact sequences")
            protein_ids.add(protein_id)
            sequence_hashes.add(sequence_hash)
        if not protein_ids:
            raise ValueError(f"{label} cohort must be non-empty")
        return protein_ids, sequence_hashes

    selection_ids, selection_sequences = identities(selection_records, "selection")
    evaluation_ids, evaluation_sequences = identities(evaluation_records, "evaluation")
    overlap_ids = selection_ids & evaluation_ids
    overlap_sequences = selection_sequences & evaluation_sequences
    if overlap_ids or overlap_sequences:
        raise ValueError(
            "selection/evaluation cohort overlap detected: "
            f"protein_ids={sorted(overlap_ids)}, sequence_hashes={sorted(overlap_sequences)}"
        )
    return {
        "selection_evaluation_disjoint": True,
        "n_selection_proteins": len(selection_ids),
        "n_evaluation_proteins": len(evaluation_ids),
        "overlap_by_id": 0,
        "overlap_by_exact_sequence": 0,
    }


def select_positive_features(
    attribution_rows: Sequence[Mapping[str, object]],
    *,
    selection_split_id: str,
    evaluation_split_id: str,
    classes: Sequence[str],
    layers: Sequence[int],
    sites: Sequence[str],
    features_per_cell: int,
) -> dict:
    """Freeze strictly positive feature identities on an independent split."""

    if not selection_split_id or selection_split_id == evaluation_split_id:
        raise ValueError("selection and evaluation split identifiers must be non-empty and distinct")
    if features_per_cell < 1:
        raise ValueError("features_per_cell must be positive")
    required = {"ec_class", "layer", "site", "feature", "direct_effect", "decoder_norm", "split_id"}
    cells: dict[tuple, list[dict]] = defaultdict(list)
    identities: set[tuple] = set()
    for source in attribution_rows:
        missing = required - set(source)
        if missing:
            raise ValueError(f"attribution row is missing fields: {sorted(missing)}")
        row = dict(source)
        if str(row["split_id"]) != selection_split_id:
            raise ValueError("every attribution row must come from the frozen selection split")
        identity = (str(row["ec_class"]), int(row["layer"]), str(row["site"]), int(row["feature"]))
        if identity in identities:
            raise ValueError(f"duplicate attribution identity: {identity}")
        identities.add(identity)
        effect, norm = float(row["direct_effect"]), float(row["decoder_norm"])
        if not np.isfinite(effect) or not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("direct effects and positive decoder norms must be finite")
        row.update(
            ec_class=identity[0],
            layer=identity[1],
            site=identity[2],
            feature=identity[3],
            direct_effect=effect,
            decoder_norm=norm,
            split_id=selection_split_id,
        )
        cells[identity[:3]].append(row)

    decisions, selected_by_cell = [], {}
    for ec_class in classes:
        if ec_class not in EC_PROMPTS:
            raise ValueError(f"unknown EC class: {ec_class}")
        for layer in layers:
            for site in sites:
                cell = (ec_class, int(layer), str(site))
                candidates = sorted(cells.get(cell, []), key=lambda row: (-row["direct_effect"], row["feature"]))
                positive = [row for row in candidates if row["direct_effect"] > 0.0]
                if len(positive) < features_per_cell:
                    raise ValueError(
                        f"{ec_class} layer {layer} site {site} has {len(positive)} positive "
                        f"candidates; requires {features_per_cell}. Refusing opposite-sign fallback."
                    )
                selected = positive[:features_per_cell]
                selected_ids = {row["feature"] for row in selected}
                selected_by_cell[cell] = selected
                for row in candidates:
                    decisions.append(
                        {
                            **row,
                            "selected": row["feature"] in selected_ids,
                            "decision": (
                                "selected_positive" if row["feature"] in selected_ids else "not_selected"
                            ),
                        }
                    )
    return {
        "selection_split_id": selection_split_id,
        "evaluation_split_id": evaluation_split_id,
        "features_per_cell": int(features_per_cell),
        "opposite_sign_fallback": False,
        "selected_by_cell": selected_by_cell,
        "decisions": decisions,
    }


def _control_features(
    feature_pool: Sequence[Mapping[str, object]],
    selected: Mapping[tuple, Sequence[dict]],
    *,
    seed: int,
    norm_log_caliper: float,
) -> dict[tuple, dict[str, list[dict]]]:
    if not np.isfinite(norm_log_caliper) or norm_log_caliper <= 0.0:
        raise ValueError("norm_log_caliper must be finite and positive")
    required = {"layer", "site", "feature", "decoder_norm"}
    pool_by_cell: dict[tuple, list[dict]] = defaultdict(list)
    identities = set()
    for source in feature_pool:
        if required - set(source):
            raise ValueError("feature-pool rows require layer, site, feature and decoder_norm")
        row = {
            "layer": int(source["layer"]),
            "site": str(source["site"]),
            "feature": int(source["feature"]),
            "decoder_norm": float(source["decoder_norm"]),
        }
        identity = (row["layer"], row["site"], row["feature"])
        if identity in identities:
            raise ValueError(f"duplicate feature-pool identity: {identity}")
        identities.add(identity)
        if not np.isfinite(row["decoder_norm"]) or row["decoder_norm"] <= 0.0:
            raise ValueError("feature-pool decoder norms must be finite and positive")
        pool_by_cell[(row["layer"], row["site"])].append(row)

    selected_ids_by_cell: dict[tuple[int, str], set[int]] = defaultdict(set)
    for (_, layer, site), targets in selected.items():
        selected_ids_by_cell[(layer, site)].update(int(row["feature"]) for row in targets)

    controls = {}
    for cell, targets in selected.items():
        ec_class, layer, site = cell
        eligible = [
            row
            for row in pool_by_cell.get((layer, site), [])
            if row["feature"] not in selected_ids_by_cell[(layer, site)]
        ]
        feasible = {
            int(target["feature"]): [
                row
                for row in eligible
                if abs(np.log(row["decoder_norm"] / target["decoder_norm"])) <= norm_log_caliper
            ]
            for target in targets
        }
        target_order = sorted(
            targets,
            key=lambda target: (len(feasible[int(target["feature"])]), int(target["feature"])),
        )
        norm_by_target, used = {}, set()
        for target in target_order:
            candidates = [
                row for row in feasible[int(target["feature"])] if row["feature"] not in used
            ]
            if not candidates:
                raise ValueError(
                    f"no unused norm control within log caliper {norm_log_caliper} for {cell} "
                    f"target feature {target['feature']}"
                )
            chosen = min(
                candidates,
                key=lambda row: (
                    abs(np.log(row["decoder_norm"] / target["decoder_norm"])),
                    row["feature"],
                ),
            )
            used.add(chosen["feature"])
            norm_by_target[int(target["feature"])] = {
                **chosen,
                "target_feature": int(target["feature"]),
                "target_decoder_norm": float(target["decoder_norm"]),
                "log_norm_difference": float(
                    abs(np.log(chosen["decoder_norm"] / target["decoder_norm"]))
                ),
                "relative_norm_difference": float(
                    abs(chosen["decoder_norm"] - target["decoder_norm"]) / target["decoder_norm"]
                ),
                "norm_log_caliper": float(norm_log_caliper),
                "within_caliper": True,
            }
        norm_rows = [norm_by_target[int(target["feature"])] for target in targets]
        random_pool = [row for row in eligible if row["feature"] not in used]
        if len(random_pool) < len(targets):
            raise ValueError(f"too few disjoint random controls for {cell}")
        rng = np.random.default_rng(_derived_seed(seed, ec_class, layer, site, "random"))
        random_rows = [
            random_pool[int(index)]
            for index in rng.choice(len(random_pool), size=len(targets), replace=False)
        ]
        controls[cell] = {"random_feature": random_rows, "norm_matched_feature": norm_rows}
    return controls


def _attach_generation_set_ids(rows: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["ec_class"],
                row["arm"],
                row["site"],
                row["dose"],
                row["generation_set_index"],
            )
        ].append(row)
    for key, members in groups.items():
        member_ids = sorted(row["plan_id"] for row in members)
        set_id = _canonical_sha256(
            {"schema": "r2-steering-generation-set-v1", "cell": key, "member_plan_ids": member_ids}
        )
        for row in members:
            row["generation_set_id"] = set_id


def build_steering_plan(
    selection: dict,
    feature_pool: Sequence[Mapping[str, object]],
    *,
    classes: Sequence[str],
    layers: Sequence[int],
    sites: Sequence[str],
    doses: Sequence[float],
    n_per_arm: int,
    generation_set_size: int,
    seed_base: int,
    sampler: Mapping[str, object],
    norm_log_caliper: float,
    content_binding_sha256: str,
    generator_revision: str,
    model_revision: str,
    tokenizer_revision: str,
    clt_checkpoint_sha256: str,
    multiplier_semantics: str,
) -> dict:
    """Create content-bound rows with paired RNG streams and replicated sets."""

    binding = _valid_sha256(content_binding_sha256, "content_binding_sha256")
    generator_revision = str(generator_revision).strip()
    model_revision = str(model_revision).strip()
    tokenizer_revision = str(tokenizer_revision).strip()
    clt_checkpoint_sha256 = _valid_sha256(clt_checkpoint_sha256, "clt_checkpoint_sha256")
    multiplier_semantics = str(multiplier_semantics).strip()
    if not generator_revision or not model_revision or not tokenizer_revision or not multiplier_semantics:
        raise ValueError("generator/model/tokenizer revisions and multiplier semantics must be frozen")
    if n_per_arm < 2 or generation_set_size < 1 or n_per_arm % generation_set_size:
        raise ValueError("n_per_arm must be >=2 and divisible by generation_set_size")
    if n_per_arm // generation_set_size < 2:
        raise ValueError("at least two replicated generation sets are required")
    dose_values = [float(dose) for dose in doses]
    if len(dose_values) < 2 or len(set(dose_values)) != len(dose_values):
        raise ValueError("the confirmatory dose sweep requires at least two unique doses")
    if any(not np.isfinite(dose) or dose <= 0.0 for dose in dose_values):
        raise ValueError("steering doses must be finite and positive")
    controls = _control_features(
        feature_pool,
        selection["selected_by_cell"],
        seed=seed_base,
        norm_log_caliper=norm_log_caliper,
    )
    rows = []

    def add_row(
        ec_class: str,
        arm: str,
        site: str,
        dose: float,
        generation_seed: int,
        seed_index: int,
        interventions: list[dict],
    ) -> None:
        row = {
            "ec_class": ec_class,
            "prompt": f"{EC_PROMPTS[ec_class]}<sep><start>",
            "arm": arm,
            "site": site,
            "dose": float(dose),
            "seed": int(generation_seed),
            "rng_stream_id": _canonical_sha256(
                {"ec_class": ec_class, "seed": int(generation_seed), "seed_base": int(seed_base)}
            ),
            "generation_set_index": int(seed_index // generation_set_size),
            "interventions": interventions,
            "sampler": dict(sampler),
            "content_binding_sha256": binding,
            "generator_revision": generator_revision,
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "clt_checkpoint_sha256": clt_checkpoint_sha256,
            "multiplier_semantics": multiplier_semantics,
        }
        row["plan_id"] = _plan_id(row)
        rows.append(row)

    for class_index, ec_class in enumerate(classes):
        seeds = [seed_base + class_index * n_per_arm + index for index in range(n_per_arm)]
        for seed_index, generation_seed in enumerate(seeds):
            add_row(ec_class, "prompt_only", "none", 0.0, generation_seed, seed_index, [])
        for site in sites:
            interventions_by_arm = {"target": [], "random_feature": [], "norm_matched_feature": []}
            for layer in layers:
                cell = (ec_class, int(layer), str(site))
                for arm, features in (
                    ("target", selection["selected_by_cell"][cell]),
                    ("random_feature", controls[cell]["random_feature"]),
                    ("norm_matched_feature", controls[cell]["norm_matched_feature"]),
                ):
                    interventions_by_arm[arm].extend(
                        {
                            "layer": int(feature["layer"]),
                            "feature": int(feature["feature"]),
                            "site": str(site),
                            "multiplier": None,
                            "decoder_norm": float(feature["decoder_norm"]),
                        }
                        for feature in features
                    )
            for dose in dose_values:
                for arm, features in interventions_by_arm.items():
                    interventions = [{**feature, "multiplier": dose} for feature in features]
                    for seed_index, generation_seed in enumerate(seeds):
                        add_row(
                            ec_class,
                            arm,
                            str(site),
                            dose,
                            generation_seed,
                            seed_index,
                            interventions,
                        )
    if len({row["plan_id"] for row in rows}) != len(rows):
        raise RuntimeError("generation plan identifiers are not unique")
    _attach_generation_set_ids(rows)
    norm_differences = [
        feature["log_norm_difference"]
        for decision in controls.values()
        for feature in decision["norm_matched_feature"]
    ]
    return {
        "schema_version": "r2-corrected-steering-plan-v2",
        "scope": "Prospective generation plan only; no steering outcome is implied.",
        "selection_split_id": selection["selection_split_id"],
        "evaluation_split_id": selection["evaluation_split_id"],
        "paired_random_streams": True,
        "classes": list(classes),
        "layers": [int(layer) for layer in layers],
        "sites": [str(site) for site in sites],
        "doses": dose_values,
        "n_per_arm": int(n_per_arm),
        "generation_set_size": int(generation_set_size),
        "n_generation_sets": int(n_per_arm // generation_set_size),
        "content_binding_sha256": binding,
        "norm_match_balance": {
            "caliper_metric": "absolute_log_decoder_norm_ratio",
            "norm_log_caliper": float(norm_log_caliper),
            "n_matches": len(norm_differences),
            "mean_log_difference": float(np.mean(norm_differences)),
            "max_log_difference": float(np.max(norm_differences)),
            "all_within_caliper": bool(max(norm_differences) <= norm_log_caliper),
        },
        "selection_decisions": selection["decisions"],
        "control_decisions": [
            {"ec_class": cell[0], "layer": cell[1], "site": cell[2], **decision}
            for cell, decision in sorted(controls.items(), key=lambda item: str(item[0]))
        ],
        "rows": rows,
    }


def validate_plan_rows(plan_rows: Sequence[Mapping[str, object]]) -> list[dict]:
    """Recompute every content-bound row and generation-set identifier."""

    rows = [dict(row) for row in plan_rows]
    if not rows or len({str(row.get("plan_id", "")) for row in rows}) != len(rows):
        raise ValueError("plan rows must have unique non-empty identifiers")
    for row in rows:
        if str(row["plan_id"]) != _plan_id(row):
            raise ValueError("plan_id is not bound to the complete frozen row content")
    expected = [{key: value for key, value in row.items() if key != "generation_set_id"} for row in rows]
    _attach_generation_set_ids(expected)
    for row, derived in zip(rows, expected):
        if row.get("generation_set_id") != derived["generation_set_id"]:
            raise ValueError("generation_set_id is not bound to its exact member plan IDs")
    return rows


def validate_completed_generations(
    plan_rows: Sequence[Mapping[str, object]], outputs: Sequence[Mapping[str, object]]
) -> list[dict]:
    """Require complete raw tokens, sequence, stop reason and runtime provenance."""

    plan_rows = validate_plan_rows(plan_rows)
    plan = {str(row["plan_id"]): row for row in plan_rows}
    output_by_id = {}
    for source in outputs:
        output = dict(source)
        plan_id = str(output.get("plan_id", ""))
        if plan_id not in plan or plan_id in output_by_id:
            raise ValueError("generation outputs contain unknown or duplicate plan identifiers")
        sequence = "".join(str(output.get("sequence", "")).upper().split())
        if not sequence or any(residue not in AA for residue in sequence):
            raise ValueError("every output must contain a non-empty canonical amino-acid sequence")
        sequence_digest = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        if output.get("sequence_sha256") not in (None, sequence_digest):
            raise ValueError("output sequence_sha256 does not match the full sequence")
        token_ids, token_digest = _token_ids_sha256(output.get("token_ids", []))
        if str(output.get("token_ids_sha256", "")) != token_digest:
            raise ValueError("output token_ids_sha256 does not match raw token_ids")
        stop_reason = str(output.get("stop_reason", "")).strip()
        if not stop_reason:
            raise ValueError("every output requires a non-empty stop_reason")
        runtime = output.get("runtime")
        if not isinstance(runtime, Mapping) or RUNTIME_FIELDS - set(runtime):
            raise ValueError(f"runtime metadata requires fields: {sorted(RUNTIME_FIELDS)}")
        runtime = dict(runtime)
        for field in RUNTIME_FIELDS - {"elapsed_seconds"}:
            if not str(runtime[field]).strip():
                raise ValueError(f"runtime field {field} must be non-empty")
        elapsed = float(runtime["elapsed_seconds"])
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("runtime elapsed_seconds must be finite and non-negative")
        if runtime["evaluation_mode"] != "eval":
            raise ValueError("generation must run in model evaluation mode")
        if str(runtime["generator_revision"]) != str(plan[plan_id]["generator_revision"]):
            raise ValueError("runtime generator_revision does not match the frozen plan")
        for revision_field in ("model_revision", "tokenizer_revision", "clt_checkpoint_sha256"):
            if str(runtime[revision_field]) != str(plan[plan_id][revision_field]):
                raise ValueError(f"runtime {revision_field} does not match the frozen plan")
        if str(runtime["hook_site"]) != str(plan[plan_id]["site"]):
            raise ValueError("runtime hook_site does not match the frozen plan")
        if str(runtime["multiplier_semantics"]) != str(
            plan[plan_id]["multiplier_semantics"]
        ):
            raise ValueError("runtime multiplier_semantics does not match the frozen plan")
        if str(runtime["rng_stream_id"]) != str(plan[plan_id]["rng_stream_id"]):
            raise ValueError("runtime RNG stream does not match the frozen paired stream")
        runtime["elapsed_seconds"] = elapsed
        output_by_id[plan_id] = {
            "sequence": sequence,
            "sequence_sha256": sequence_digest,
            "token_ids": token_ids,
            "token_ids_sha256": token_digest,
            "stop_reason": stop_reason,
            "runtime": runtime,
        }
    missing = set(plan) - set(output_by_id)
    if missing:
        raise ValueError(f"missing {len(missing)} planned generation outputs")
    return [{**row, **output_by_id[row["plan_id"]]} for row in plan_rows]


def _paired_statistics(values: np.ndarray, *, seed: int, n_resamples: int) -> dict:
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("paired steering differences must be finite with at least two units")
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(0, values.size, size=(n_resamples, values.size))
    boot = values[bootstrap_indices].mean(axis=1)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, values.size))
    null = (values * signs).mean(axis=1)
    observed = float(values.mean())
    return {
        "mean_difference": observed,
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "positive_p": float((1 + np.sum(null >= observed)) / (n_resamples + 1)),
        "n_pairs": int(values.size),
    }


def _holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    adjusted_ranked = np.maximum.accumulate((p.size - np.arange(p.size)) * ranked)
    result = np.empty_like(p)
    result[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return result


def _generation_sets(completed: Sequence[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in completed:
        groups[str(row["generation_set_id"])].append(row)
    result = {}
    for set_id, members in groups.items():
        metadata = {
            (row["ec_class"], row["arm"], row["site"], row["dose"], row["generation_set_index"])
            for row in members
        }
        if len(metadata) != 1:
            raise ValueError("generation set mixes incompatible plan cells")
        ec_class, arm, site, dose, set_index = metadata.pop()
        result[set_id] = {
            "generation_set_id": set_id,
            "ec_class": ec_class,
            "arm": arm,
            "site": site,
            "dose": dose,
            "pair_id": int(set_index),
            "member_plan_ids": sorted(row["plan_id"] for row in members),
        }
    return result


def validate_score_receipt(
    receipt: Mapping[str, object],
    completed_rows: Sequence[Mapping[str, object]],
    score_rows: Sequence[Mapping[str, object]],
    endpoint_specs: Mapping[str, Mapping[str, object]],
    *,
    freeze_id: str,
    generation_outputs_sha256: str,
    generation_execution_receipt_sha256: str | None,
    scores_sha256: str,
    frozen_endpoint_specs_sha256: str,
    synthetic: bool,
) -> dict:
    """Validate one hash-bound, complete scorer execution per frozen endpoint."""

    if not isinstance(receipt, Mapping) or set(receipt) != SCORE_RECEIPT_FIELDS:
        raise ValueError("score receipt fields differ from the exact frozen schema")
    if type(receipt["synthetic"]) is not bool or receipt["synthetic"] is not synthetic:
        raise ValueError("score receipt synthetic scope does not match the analysis mode")
    expected_status = "synthetic_fixture_complete" if synthetic else "verified_complete"
    if (
        receipt["schema_version"] != "r2-corrected-steering-score-receipt-v2"
        or receipt["status"] != expected_status
    ):
        raise ValueError("score receipt has an unsupported schema or non-complete status")

    bindings = {
        "freeze_id": _valid_sha256(freeze_id, "freeze_id"),
        "generation_outputs_sha256": _valid_sha256(
            generation_outputs_sha256, "generation_outputs_sha256"
        ),
        "scores_sha256": _valid_sha256(scores_sha256, "scores_sha256"),
        "frozen_endpoint_specs_sha256": _valid_sha256(
            frozen_endpoint_specs_sha256, "frozen_endpoint_specs_sha256"
        ),
    }
    for field, expected in bindings.items():
        if receipt[field] != expected:
            raise ValueError(f"score receipt {field} does not match the analyzed artifact")
    if synthetic:
        if receipt["generation_execution_receipt_sha256"] is not None:
            raise ValueError("synthetic score receipt cannot claim a production execution receipt")
        execution_receipt_binding = None
    else:
        execution_receipt_binding = _valid_sha256(
            generation_execution_receipt_sha256,
            "generation_execution_receipt_sha256",
        )
        if receipt["generation_execution_receipt_sha256"] != execution_receipt_binding:
            raise ValueError(
                "score receipt generation_execution_receipt_sha256 does not match "
                "the verified generation receipt"
            )

    specs = validate_endpoint_specs(endpoint_specs, require_artifacts=not synthetic)
    completed = [dict(row) for row in completed_rows]
    completed_by_id = {str(row.get("plan_id", "")): row for row in completed}
    if not completed or len(completed_by_id) != len(completed):
        raise ValueError("score receipt requires unique completed generation plan IDs")
    generation_sets = _generation_sets(completed)
    expected_units = {
        endpoint: sorted(
            completed_by_id if spec["experimental_unit"] == "generation" else generation_sets
        )
        for endpoint, spec in specs.items()
    }
    scored_units: dict[str, list[str]] = defaultdict(list)
    for source in score_rows:
        endpoint = str(source.get("endpoint", ""))
        if endpoint not in specs:
            raise ValueError("score receipt input includes an unknown endpoint")
        unit_field = (
            "plan_id" if specs[endpoint]["experimental_unit"] == "generation" else "generation_set_id"
        )
        scored_units[endpoint].append(str(source.get(unit_field, "")))
    for endpoint, unit_ids in expected_units.items():
        actual = scored_units.get(endpoint, [])
        if len(actual) != len(set(actual)) or sorted(actual) != unit_ids:
            raise ValueError(
                f"score receipt input lacks exact expected coverage for endpoint {endpoint}"
            )

    executions = receipt["scorer_executions"]
    if not isinstance(executions, list):
        raise ValueError("score receipt scorer_executions must be a list")
    execution_by_endpoint = {}
    for source in executions:
        if not isinstance(source, Mapping) or set(source) != SCORER_EXECUTION_FIELDS:
            raise ValueError("scorer execution fields differ from the exact receipt schema")
        execution = dict(source)
        endpoint = str(execution["endpoint"])
        if endpoint not in specs or endpoint in execution_by_endpoint:
            raise ValueError("score receipt contains an unknown or duplicate endpoint execution")
        execution_by_endpoint[endpoint] = execution
    if set(execution_by_endpoint) != set(specs):
        raise ValueError("score receipt must contain exactly one execution per frozen endpoint")

    for endpoint, spec in specs.items():
        execution = execution_by_endpoint[endpoint]
        if execution["status"] != "complete":
            raise ValueError(f"scorer execution for endpoint {endpoint} is non-complete")
        if type(execution["validated"]) is not bool or type(execution["primary"]) is not bool:
            raise ValueError("scorer execution validated/primary fields must be JSON booleans")
        if execution["validated"] is not spec["validated"] or execution["primary"] is not spec["primary"]:
            raise ValueError(f"scorer execution validation class differs for endpoint {endpoint}")
        expected_mode = "independently_validated" if spec["validated"] else "heuristic_supporting_only"
        if execution["execution_mode"] != expected_mode:
            raise ValueError(
                f"heuristic/validated scorer execution mode differs for endpoint {endpoint}"
            )
        for field in (
            "scorer_name",
            "scorer_version",
            "scorer_path",
            "scorer_sha256",
            "calibration_cohort_path",
            "calibration_cohort_sha256",
        ):
            if execution[field] != spec.get(field):
                raise ValueError(f"scorer execution {field} differs for endpoint {endpoint}")
        if execution["experimental_unit"] != spec["experimental_unit"]:
            raise ValueError(f"scorer execution experimental_unit differs for endpoint {endpoint}")
        units = expected_units[endpoint]
        coverage_sha256 = _canonical_sha256(
            {
                "endpoint": endpoint,
                "experimental_unit": spec["experimental_unit"],
                "unit_ids": units,
            }
        )
        if (
            type(execution["expected_unit_count"]) is not int
            or type(execution["scored_unit_count"]) is not int
        ):
            raise ValueError(f"scorer execution counts must be integers for endpoint {endpoint}")
        if (
            execution["expected_unit_count"] != len(units)
            or execution["scored_unit_count"] != len(units)
            or execution["expected_coverage_sha256"] != coverage_sha256
            or execution["scored_coverage_sha256"] != coverage_sha256
        ):
            raise ValueError(f"scorer execution coverage differs for endpoint {endpoint}")

    return {
        "schema_version": receipt["schema_version"],
        "status": receipt["status"],
        "synthetic": synthetic,
        "receipt_validated": True,
        **bindings,
        "generation_execution_receipt_sha256": execution_receipt_binding,
        "n_scorer_executions": len(execution_by_endpoint),
        "validated_endpoints": sorted(
            endpoint for endpoint, spec in specs.items() if spec["validated"]
        ),
        "heuristic_endpoints": sorted(
            endpoint for endpoint, spec in specs.items() if not spec["validated"]
        ),
    }


def analyze_steering_scores(
    completed_rows: Sequence[Mapping[str, object]],
    score_rows: Sequence[Mapping[str, object]],
    endpoint_specs: Mapping[str, Mapping[str, object]],
    *,
    analysis_spec: Mapping[str, object],
) -> dict:
    """Analyze generation and replicated-set endpoints with specificity gates."""

    endpoint_specs = validate_endpoint_specs(endpoint_specs)
    frozen_analysis = validate_analysis_spec(analysis_spec)
    completed = [dict(row) for row in completed_rows]
    completed_by_id = {str(row["plan_id"]): row for row in completed}
    if len(completed_by_id) != len(completed):
        raise ValueError("completed generations contain duplicate plan IDs")
    generation_sets = _generation_sets(completed)
    score_lookup: dict[tuple[str, str], float] = {}
    for source in score_rows:
        endpoint = str(source.get("endpoint", ""))
        if endpoint not in endpoint_specs:
            raise ValueError("score row references an unknown endpoint")
        unit = endpoint_specs[endpoint]["experimental_unit"]
        if unit == "generation":
            unit_id = str(source.get("plan_id", ""))
            if unit_id not in completed_by_id:
                raise ValueError("generation score references an unknown plan_id")
        else:
            unit_id = str(source.get("generation_set_id", ""))
            if unit_id not in generation_sets:
                raise ValueError("set score references an unknown generation_set_id")
            members = sorted(str(value) for value in source.get("member_plan_ids", []))
            if members != generation_sets[unit_id]["member_plan_ids"]:
                raise ValueError("set score member_plan_ids do not match the frozen generation set")
        key = (endpoint, unit_id)
        if key in score_lookup:
            raise ValueError("duplicate endpoint/unit score")
        score = float(source["score"])
        if not np.isfinite(score):
            raise ValueError("endpoint scores must be finite")
        score_lookup[key] = score
    expected = {
        (endpoint, unit_id)
        for endpoint, spec in endpoint_specs.items()
        for unit_id in (
            completed_by_id if spec["experimental_unit"] == "generation" else generation_sets
        )
    }
    if set(score_lookup) != expected:
        raise ValueError("scores must cover every frozen endpoint at its exact experimental unit")

    directional_scores: dict[tuple, float] = {}
    for endpoint, spec in endpoint_specs.items():
        direction = 1.0 if spec["direction"] == "higher" else -1.0
        units = (
            [
                {
                    "unit_id": row["plan_id"],
                    "ec_class": row["ec_class"],
                    "arm": row["arm"],
                    "site": row["site"],
                    "dose": row["dose"],
                    "pair_id": row["seed"],
                }
                for row in completed
            ]
            if spec["experimental_unit"] == "generation"
            else [{"unit_id": key, **value} for key, value in generation_sets.items()]
        )
        for unit in units:
            key = (
                endpoint,
                unit["ec_class"],
                unit["arm"],
                unit["site"],
                float(unit["dose"]),
                unit["pair_id"],
            )
            if key in directional_scores:
                raise ValueError("duplicate endpoint/cell/pair unit")
            directional_scores[key] = direction * score_lookup[(endpoint, unit["unit_id"])]

    grouped: dict[tuple, list[float]] = defaultdict(list)
    specificity_grouped: dict[tuple, list[float]] = defaultdict(list)
    for key, value in directional_scores.items():
        endpoint, ec_class, arm, site, dose, pair_id = key
        if arm == "prompt_only":
            continue
        baseline_key = (endpoint, ec_class, "prompt_only", "none", 0.0, pair_id)
        if baseline_key not in directional_scores:
            raise ValueError("an intervention unit lacks its paired prompt-only unit")
        grouped[(ec_class, arm, site, dose, endpoint)].append(
            value - directional_scores[baseline_key]
        )
        if arm == "target":
            for control_arm in ("random_feature", "norm_matched_feature"):
                control_key = (endpoint, ec_class, control_arm, site, dose, pair_id)
                if control_key not in directional_scores:
                    raise ValueError("target unit lacks a paired control unit")
                specificity_grouped[
                    (ec_class, control_arm, site, dose, endpoint)
                ].append(value - directional_scores[control_key])

    def make_cell(key: tuple, values: Sequence[float], comparison: str) -> dict:
        endpoint = key[-1]
        array = np.asarray(values, dtype=np.float64)
        return {
            "ec_class": key[0],
            "arm": key[1],
            "site": key[2],
            "dose": key[3],
            "endpoint": endpoint,
            "experimental_unit": endpoint_specs[endpoint]["experimental_unit"],
            "comparison": comparison,
            "endpoint_validated": bool(endpoint_specs[endpoint]["validated"]),
            "endpoint_primary": bool(endpoint_specs[endpoint]["primary"]),
            **_paired_statistics(
                array,
                seed=_derived_seed(frozen_analysis["random_seed"], comparison, key),
                n_resamples=frozen_analysis["n_resamples"],
            ),
            "equivalence": tost_paired(
                array,
                float(endpoint_specs[endpoint]["equivalence_margin"]),
                alpha=frozen_analysis["alpha"],
            ),
        }

    cells = [
        make_cell(key, differences, "arm_minus_prompt_only")
        for key, differences in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]
    specificity_cells = [
        make_cell(key, differences, "target_minus_control")
        for key, differences in sorted(specificity_grouped.items(), key=lambda item: str(item[0]))
    ]
    family = [*cells, *specificity_cells]
    positive_adjusted = _holm_adjust([cell["positive_p"] for cell in family])
    equivalence_adjusted = _holm_adjust(
        [cell["equivalence"]["p_tost"] for cell in family]
    )
    for cell, positive_p, equivalent_p in zip(family, positive_adjusted, equivalence_adjusted):
        cell["positive_p_holm"] = float(positive_p)
        cell["equivalence"]["p_tost_holm"] = float(equivalent_p)
        cell["equivalence"]["equivalent_holm"] = bool(
            equivalent_p < frozen_analysis["alpha"]
        )
        cell["claim_eligible"] = cell["endpoint_validated"] and cell["endpoint_primary"]
        if cell["claim_eligible"] and cell["mean_difference"] > 0.0 and positive_p < frozen_analysis["alpha"]:
            cell["status"] = "positive"
        elif cell["claim_eligible"] and cell["equivalence"]["equivalent_holm"]:
            cell["status"] = "equivalent"
        else:
            cell["status"] = "inconclusive"

    class_results = []
    planned_classes = sorted({row["ec_class"] for row in completed})
    for ec_class in planned_classes:
        target = [
            cell
            for cell in cells
            if cell["ec_class"] == ec_class and cell["arm"] == "target" and cell["claim_eligible"]
        ]
        positive_cells = []
        for cell in target:
            if cell["status"] != "positive":
                continue
            matched_specificity = [
                specificity
                for specificity in specificity_cells
                if specificity["ec_class"] == ec_class
                and specificity["site"] == cell["site"]
                and specificity["dose"] == cell["dose"]
                and specificity["endpoint"] == cell["endpoint"]
                and specificity["claim_eligible"]
            ]
            if {row["arm"] for row in matched_specificity if row["status"] == "positive"} == {
                "random_feature",
                "norm_matched_feature",
            }:
                positive_cells.append(
                    {key: cell[key] for key in ("site", "dose", "endpoint", "experimental_unit")}
                )
        if positive_cells:
            status = "positive_specific"
        elif target and all(cell["status"] == "equivalent" for cell in target):
            status = "equivalent_negative"
        else:
            status = "inconclusive"
        class_results.append(
            {
                "ec_class": ec_class,
                "status": status,
                "n_primary_target_cells": len(target),
                "positive_specific_cells": positive_cells,
            }
        )
    return {
        "schema_version": "r2-corrected-steering-analysis-v2",
        "scope": "Frozen, paired, specificity-, multiplicity- and equivalence-aware analysis.",
        "frozen_analysis_spec": frozen_analysis,
        "endpoint_specs": endpoint_specs,
        "multiplicity_family_size": len(family),
        "cells": cells,
        "specificity_cells": specificity_cells,
        "classes": class_results,
        "all_eight_resolved": bool(
            set(planned_classes) == set(EC_PROMPTS)
            and all(
                row["status"] in {"positive_specific", "equivalent_negative"}
                for row in class_results
            )
        ),
    }


def synthetic_steering_fixture(seed: int = 20260717) -> tuple[list[dict], list[dict]]:
    """Create complete positive-only attribution and feature-pool fixtures."""

    rng = np.random.default_rng(seed)
    attributions, pool = [], []
    for layer in (3, 12, 30):
        for site_index, site in enumerate(("clt_input", "mlp_output")):
            for feature in range(80):
                base_feature = feature % 40
                pool.append(
                    {
                        "layer": layer,
                        "site": site,
                        "feature": feature,
                        "decoder_norm": (
                            0.8
                            + 0.02 * base_feature
                            + 0.01 * site_index
                            + (0.001 if feature >= 40 else 0.0)
                        ),
                    }
                )
            for class_index, ec_class in enumerate(EC_PROMPTS):
                for rank in range(6):
                    feature = (class_index * 7 + rank) % 40
                    attributions.append(
                        {
                            "ec_class": ec_class,
                            "layer": layer,
                            "site": site,
                            "feature": feature,
                            "direct_effect": 0.30 - 0.04 * rank if rank < 4 else -0.05 * (rank - 3),
                            "decoder_norm": 0.8 + 0.02 * feature + 0.01 * site_index,
                            "split_id": "synthetic_selection",
                            "measurement_noise": float(rng.normal(scale=0.001)),
                        }
                    )
    return attributions, pool
