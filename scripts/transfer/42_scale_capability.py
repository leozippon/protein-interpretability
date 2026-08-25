#!/usr/bin/env python3
"""Descriptive ProGen2 medium→large→xlarge comparison. Not a causal scale claim.

This stage is CPU-only. It consumes existing stage-20 and stage-29 artefacts
and a full stage-41 report. It does not load a model, does not download
weights, and does not synthesise a cross-task total.

The comparison is fixed as progen2-medium → progen2-large → progen2-xlarge.
Checkpoint differences are descriptive of those checkpoints. They are not a
parameter-count causal effect: corpus identification is a model-favouring bound
on UniRef90+BFD30, and nothing here is a claim about biological knowledge.

A run that cannot qualify all three rungs on the same stage-41 blocks does not
emit DMS or MegaScale gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.arms import (  # noqa: E402
    REPO,
    STAGED_SCALE_ARMS,
    UNIREF90_BFD30_INCOMPLETE_SEARCH,
    arm_spec,
)
from src.transfer import designed_referent as D  # noqa: E402
from src.transfer.io import write_json  # noqa: E402

SCHEMA_VERSION = "r2_transfer_scale_capability_v2"
DEFAULT_OUT = REPO / "results/transfer/scale_capability"
SCALE_RUNGS = ("progen2-medium", "progen2-large", "progen2-xlarge")
ADJACENT_PAIRS = (
    ("progen2-medium", "progen2-large"),
    ("progen2-large", "progen2-xlarge"),
)
BOOTSTRAP_RESAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 20260825
HYDROPATHY_BASELINE = "hydropathy_change"
FRAGMENT_MAX_ORDER = 7

#: Fixed EXP-R2-224 census, from the stage-20/29 artefacts this comparison reads.
DMS_FIXED_ASSAYS = 217
DMS_FIXED_CLUSTERS = 174
MEGASCALE_DESIGN_WILDTYPES = 130
MEGASCALE_DESIGN_SERIES = 40
MEGASCALE_NATURAL_WILDTYPES = 266
MEGASCALE_NATURAL_CLUSTERS = 124

DESCRIPTIVE_NOT_CAUSAL = (
    "checkpoint differences on this ladder are descriptive of the named "
    "checkpoints. They are not identified as a causal effect of parameter count"
)
NO_BIOLOGICAL_KNOWLEDGE_CLAIM = (
    "no biological-knowledge claim is licensed. Beating a sequence baseline is "
    "not evidence that a model has learned biology"
)


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    return json.loads(path.read_text(encoding="utf-8"))


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_rung_order(names: list[str]) -> None:
    """Refuse any comparison whose rung list is not medium, large, xlarge."""

    got = list(names)
    if got != list(SCALE_RUNGS):
        raise ValueError(
            f"the descriptive comparison is fixed as {list(SCALE_RUNGS)}; "
            f"got {got}"
        )


def lower_bound_positive(record: dict[str, Any]) -> bool | None:
    """True/False when an interval exists; None when it cannot be read."""

    if record.get("degenerate") or record.get("interval") is None:
        return None
    return bool(record["interval"][0] > 0.0)


def compound_verdict(conditions: dict[str, bool | None]) -> bool | str:
    """True only when every named condition is True; unresolved if any is missing."""

    if any(value is None for value in conditions.values()):
        return "unresolved"
    return all(bool(value) for value in conditions.values())


def _aligned_keys(left: dict[str, Any], right: dict[str, Any], *, label: str) -> list[str]:
    if set(left) != set(right):
        missing = sorted(set(left) ^ set(right))
        raise ValueError(f"{label} keys disagree: {missing}")
    return sorted(left)


def paired_unit_delta(
    smaller: dict[str, float],
    larger: dict[str, float],
    units: dict[str, str],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Paired Δρ over the original unit labels, resampled as those units."""

    names = _aligned_keys(smaller, larger, label="paired Δρ")
    if not names:
        raise ValueError("paired Δρ has no shared units")
    missing = [name for name in names if name not in units]
    if missing:
        raise ValueError(f"paired Δρ is missing unit labels for {missing}")
    extra = sorted(set(units) - set(names))
    if extra:
        raise ValueError(f"paired Δρ has unit labels with no paired values: {extra}")
    unit_names = [units[name] for name in names]
    if any(not label for label in unit_names):
        raise ValueError("paired Δρ has an empty unit label")
    values = [larger[name] - smaller[name] for name in names]
    return D.unit_bootstrap(values, unit_names, resamples=resamples, seed=seed)


def unit_mean_record(
    values: dict[str, float],
    units: dict[str, str],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    names = sorted(values)
    missing = [name for name in names if name not in units]
    if missing:
        raise ValueError(f"unit labels missing for {missing}")
    return D.unit_bootstrap(
        [values[name] for name in names],
        [units[name] for name in names],
        resamples=resamples,
        seed=seed,
    )


def _endpoint(
    per_rung: dict[str, dict[str, float]],
    units: dict[str, str],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    rungs = {
        name: unit_mean_record(per_rung[name], units, resamples=resamples, seed=seed + index)
        for index, name in enumerate(SCALE_RUNGS)
    }
    adjacent = {}
    for offset, (smaller, larger) in enumerate(ADJACENT_PAIRS):
        delta = paired_unit_delta(
            per_rung[smaller],
            per_rung[larger],
            units,
            resamples=resamples,
            seed=seed + 10 + offset,
        )
        adjacent[f"{smaller}__{larger}"] = delta
    return {"per_rung": rungs, "adjacent_delta_rho": adjacent}


def _assay_names(payload: dict[str, Any], *, label: str) -> list[str]:
    names = [row["assay"] for row in payload["assays"]]
    if len(names) != len(set(names)):
        raise ValueError(f"{label} repeats an assay")
    return names


def align_dms(
    models: dict[str, dict[str, Any]],
    lookup: dict[str, Any],
    *,
    require_fixed_census: bool = True,
) -> tuple[
    list[str],
    dict[str, str],
    dict[str, dict[str, float]],
    dict[str, dict[str, dict[str, float]]],
]:
    """Align the fixed DMS cohort: three model assay sets equal the LOOKUP set."""

    require_rung_order(list(models))
    for name, payload in models.items():
        skipped = payload.get("skipped") or []
        if skipped:
            raise ValueError(
                f"{name} skipped {len(skipped)} assays; a jointly skipped assay "
                "must not disappear from the fixed 217-assay cohort"
            )
    lookup_assays = _assay_names(lookup, label="LOOKUP")
    lookup_set = set(lookup_assays)
    for name in SCALE_RUNGS:
        model_set = set(_assay_names(models[name], label=name))
        if model_set != lookup_set:
            raise ValueError(
                f"{name} assay set disagrees with LOOKUP: "
                f"{sorted(model_set ^ lookup_set)}"
            )
    by_assay = {row["assay"]: row for row in lookup["assays"]}
    units: dict[str, str] = {}
    raw: dict[str, dict[str, float]] = {name: {} for name in SCALE_RUNGS}
    contrasts: dict[str, dict[str, dict[str, float]]] = {
        "model_minus_lookup": {name: {} for name in SCALE_RUNGS},
        "model_minus_blosum62": {name: {} for name in SCALE_RUNGS},
    }
    for assay in lookup_assays:
        lookup_row = by_assay[assay]
        expected_digest = lookup_row["mutant_digest"]
        expected_wildtype = lookup_row.get("wildtype_id")
        expected_n = lookup_row.get("n_variants")
        for rung in SCALE_RUNGS:
            rows = {row["assay"]: row for row in models[rung]["assays"]}
            entry = rows[assay]
            if entry["mutant_digest"] != expected_digest:
                raise ValueError(
                    f"{assay}: mutant_digest disagrees between {rung} and LOOKUP"
                )
            if expected_wildtype is not None and entry.get("wildtype_id") not in (
                None,
                expected_wildtype,
            ):
                raise ValueError(
                    f"{assay}: wildtype_id disagrees between {rung} and LOOKUP"
                )
            if expected_n is not None and entry.get("n_variants") not in (
                None,
                expected_n,
            ):
                raise ValueError(
                    f"{assay}: n_variants disagrees between {rung} and LOOKUP"
                )
            rho = float(entry["spearman"])
            raw[rung][assay] = rho
            contrasts["model_minus_lookup"][rung][assay] = (
                rho - float(lookup_row["spearman"]["lookup"])
            )
            contrasts["model_minus_blosum62"][rung][assay] = (
                rho - float(lookup_row["spearman"]["blosum62"])
            )
        units[assay] = str(lookup_row["cluster"])
    if require_fixed_census:
        if len(lookup_assays) != DMS_FIXED_ASSAYS:
            raise ValueError(
                f"DMS cohort is {len(lookup_assays)} assays, not the fixed "
                f"{DMS_FIXED_ASSAYS}-assay ProteinGym substitution census"
            )
        if len(set(units.values())) != DMS_FIXED_CLUSTERS:
            raise ValueError(
                f"LOOKUP has {len(set(units.values()))} clusters, not the fixed "
                f"{DMS_FIXED_CLUSTERS}-family census"
            )
    return lookup_assays, units, raw, contrasts


def _side_keys(
    payload: dict[str, Any],
    *,
    side: str,
    spearman_of,
) -> set[str]:
    keys = set()
    for name, entry in payload["wildtypes"].items():
        if entry.get("kind") != side:
            continue
        if spearman_of(entry) is None:
            continue
        keys.add(name)
    return keys


def align_megascale(
    models: dict[str, dict[str, Any]],
    baselines: dict[str, Any],
    *,
    side: str,
    baseline_name: str,
    require_fixed_census: bool = True,
) -> tuple[dict[str, str], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Align one MegaScale side: non-null keys equal across rungs and the baseline."""

    require_rung_order(list(models))
    digests = {name: payload["cohort_sha256"] for name, payload in models.items()}
    if len(set(digests.values())) != 1:
        raise ValueError(f"MegaScale cohort_sha256 disagrees across rungs: {digests}")
    model_keys = {
        name: _side_keys(
            payload, side=side, spearman_of=lambda entry: entry.get("spearman")
        )
        for name, payload in models.items()
    }
    first_keys = model_keys[SCALE_RUNGS[0]]
    for name, keys in model_keys.items():
        if keys != first_keys:
            raise ValueError(
                f"{name} {side} non-null Spearman keys disagree with "
                f"{SCALE_RUNGS[0]}: {sorted(keys ^ first_keys)}"
            )
    baseline_keys = _side_keys(
        baselines,
        side=side,
        spearman_of=lambda entry: (entry.get("spearman") or {}).get(baseline_name),
    )
    if baseline_keys != first_keys:
        raise ValueError(
            f"baseline {side} keys for {baseline_name} disagree with the models: "
            f"{sorted(baseline_keys ^ first_keys)}"
        )
    expected_n, expected_units, unit_label = (
        (MEGASCALE_DESIGN_WILDTYPES, MEGASCALE_DESIGN_SERIES, "design series")
        if side == "design"
        else (MEGASCALE_NATURAL_WILDTYPES, MEGASCALE_NATURAL_CLUSTERS, "WT_cluster")
    )
    if require_fixed_census and len(first_keys) != expected_n:
        raise ValueError(
            f"MegaScale {side} has {len(first_keys)} wild types, not the fixed "
            f"{expected_n}"
        )
    units: dict[str, str] = {}
    raw: dict[str, dict[str, float]] = {name: {} for name in SCALE_RUNGS}
    contrast: dict[str, dict[str, float]] = {name: {} for name in SCALE_RUNGS}
    for wildtype in sorted(first_keys):
        baseline_entry = baselines["wildtypes"][wildtype]
        expected_kind = side
        expected_unit = str(baseline_entry["unit"])
        if baseline_entry.get("kind") != expected_kind:
            raise ValueError(f"{wildtype}: baseline kind is not {expected_kind}")
        for rung, payload in models.items():
            entry = payload["wildtypes"][wildtype]
            if entry.get("kind") != expected_kind:
                raise ValueError(f"{wildtype}: {rung} kind disagrees with the baseline")
            if str(entry["unit"]) != expected_unit:
                raise ValueError(f"{wildtype}: {rung} unit disagrees with the baseline")
            rho = float(entry["spearman"])
            raw[rung][wildtype] = rho
            contrast[rung][wildtype] = rho - float(baseline_entry["spearman"][baseline_name])
            if entry.get("n_variants") not in (None, baseline_entry.get("n_variants")):
                raise ValueError(
                    f"{wildtype}: {rung} n_variants disagrees with the baseline"
                )
        units[wildtype] = expected_unit
    n_units = len(set(units.values()))
    if require_fixed_census and n_units != expected_units:
        raise ValueError(
            f"MegaScale {side} has {n_units} {unit_label} units, not the fixed "
            f"{expected_units}"
        )
    return units, raw, contrast


def _require_same_cohort(*payloads: dict[str, Any], label: str) -> str:
    digests = [payload["cohort_sha256"] for payload in payloads]
    if len(set(digests)) != 1:
        raise ValueError(f"{label} cohort_sha256 disagree: {digests}")
    return digests[0]


def _fragment_margins(fragment_order: dict[str, Any]) -> dict[str, Any]:
    """Report every supported fragment margin up to k=7, without a composite score."""

    admissibility = fragment_order.get("admissibility") or {}
    highest = admissibility.get("highest_supported_order")
    supported = []
    if highest is not None:
        supported = [order for order in range(3, min(int(highest), FRAGMENT_MAX_ORDER) + 1)]
    settings = fragment_order.get("settings") or {}
    schemes = list(settings.get("schemes") or D.FRAGMENT_SMOOTHING)
    arms = fragment_order.get("arms") or {}
    missing = [name for name in SCALE_RUNGS if name not in arms]
    if missing:
        raise ValueError(f"fragment_order is missing rungs {missing}")
    margins: dict[str, Any] = {"designs": {}, "control": {}}
    for side in ("designs", "control"):
        for order in supported:
            for scheme in schemes:
                key = D.fragment_channel_name(order, scheme)
                per_rung = {}
                for rung in SCALE_RUNGS:
                    block = arms[rung].get(side) or {}
                    if key not in block:
                        raise ValueError(
                            f"fragment_order {rung} {side} lacks {key} though "
                            f"k={order} is supported"
                        )
                    per_rung[rung] = block[key]
                margins[side][key] = {"per_rung": per_rung, "supported_to_k": order}
    return {
        "highest_supported_order": highest,
        "supported_orders": supported,
        "schemes": schemes,
        "margins": margins,
        "cohort_sha256": fragment_order.get("cohort_sha256"),
    }


def qualify_stage41(report: dict[str, Any]) -> dict[str, Any]:
    """Qualify the three rungs from a full stage-41 report's ``arm_results``.

    Summary-only artefacts, missing rungs or blocks, mixed identification
    verdicts, and disagreed cohort digests are refusals. Every retained row
    must be ``PASS``.
    """

    if "arm_results" not in report:
        raise ValueError(
            "stage 42 needs a full stage-41 report with arm_results; "
            "a summary-only artefact is not an identification record"
        )
    rows = [
        row
        for row in report["arm_results"]
        if row.get("arm") in SCALE_RUNGS and not row.get("is_unigram_null_control")
    ]
    by_rung: dict[str, list[dict[str, Any]]] = {name: [] for name in SCALE_RUNGS}
    for row in rows:
        by_rung[row["arm"]].append(row)
    missing = [name for name in SCALE_RUNGS if not by_rung[name]]
    if missing:
        raise ValueError(f"stage-41 arm_results is missing rungs {missing}")
    block_sets = {
        name: frozenset(row["block_id"] for row in by_rung[name]) for name in SCALE_RUNGS
    }
    reference_blocks = block_sets[SCALE_RUNGS[0]]
    if not reference_blocks:
        raise ValueError("stage-41 arm_results carries no blocks for the scale rungs")
    for name, blocks in block_sets.items():
        if blocks != reference_blocks:
            raise ValueError(
                f"{name} covers blocks {sorted(blocks)}, not {sorted(reference_blocks)}"
            )
    record: dict[str, Any] = {
        "source": "stage41_arm_results",
        "passed": True,
        "blocks": sorted(reference_blocks),
        "rungs": {},
    }
    statuses: list[str] = []
    for name in SCALE_RUNGS:
        per_block: dict[str, Any] = {}
        for row in by_rung[name]:
            status = row.get("per_arm_identification_status")
            statuses.append(str(status))
            interval = row.get("displacement_corrected_ci_95")
            if status != "PASS" or interval is None:
                raise ValueError(
                    f"{name} block {row.get('block_id')} identification is "
                    f"{status!r}, not PASS with a displacement-corrected interval"
                )
            per_block[row["block_id"]] = {
                "per_arm_identification_status": status,
                "displacement_corrected_ci_95": list(interval),
                "cohort_digest": row["cohort_digest"],
            }
        record["rungs"][name] = {"blocks": per_block}
    mixed = sorted(set(statuses))
    if mixed != ["PASS"]:
        raise ValueError(
            f"stage-41 identification is not uniformly PASS across rungs: {mixed}"
        )
    for block_id in reference_blocks:
        digests = {
            name: record["rungs"][name]["blocks"][block_id]["cohort_digest"]
            for name in SCALE_RUNGS
        }
        if len(set(digests.values())) != 1:
            raise ValueError(
                f"block {block_id} cohort_digest disagrees across rungs: {digests}"
            )
    return record


def descriptive_gate_transitions(
    dms: dict[str, Any], megascale: dict[str, Any]
) -> dict[str, Any]:
    """The one pre-registered gate family. No per-endpoint copies."""

    dms_gates: dict[str, Any] = {}
    mega_gates: dict[str, Any] = {}
    for smaller, larger in ADJACENT_PAIRS:
        pair = f"{smaller}__{larger}"
        dms_conditions = {
            "larger_model_minus_lookup": lower_bound_positive(
                dms["model_minus_lookup"]["per_rung"][larger]
            ),
            "raw_spearman_delta": lower_bound_positive(
                dms["raw_spearman"]["adjacent_delta_rho"][pair]
            ),
        }
        dms_gates[pair] = {
            "verdict": compound_verdict(dms_conditions),
            "conditions": dms_conditions,
            "blosum62_is_not_a_dms_gate": True,
        }
        mega_conditions = {
            "design_larger_model_minus_hydropathy": lower_bound_positive(
                megascale["designs"]["model_minus_hydropathy"]["per_rung"][larger]
            ),
            "design_larger_model_minus_blosum62": lower_bound_positive(
                megascale["designs"]["model_minus_blosum62"]["per_rung"][larger]
            ),
            "natural_larger_model_minus_hydropathy": lower_bound_positive(
                megascale["control"]["model_minus_hydropathy"]["per_rung"][larger]
            ),
            "natural_larger_model_minus_blosum62": lower_bound_positive(
                megascale["control"]["model_minus_blosum62"]["per_rung"][larger]
            ),
            "design_raw_spearman_delta": lower_bound_positive(
                megascale["designs"]["raw_spearman"]["adjacent_delta_rho"][pair]
            ),
        }
        mega_gates[pair] = {
            "verdict": compound_verdict(mega_conditions),
            "conditions": mega_conditions,
            "reported_not_gated": {
                "natural_raw_spearman_delta": lower_bound_positive(
                    megascale["control"]["raw_spearman"]["adjacent_delta_rho"][pair]
                )
            },
        }
    return {"dms": dms_gates, "megascale": mega_gates}


def compare_scale(
    *,
    dms_models: dict[str, dict[str, Any]],
    lookup: dict[str, Any],
    megascale_models: dict[str, dict[str, Any]],
    baselines: dict[str, Any],
    fragment_order: dict[str, Any] | None,
    qualification_report: dict[str, Any],
    resamples: int,
    seed: int,
    require_fixed_census: bool = True,
) -> dict[str, Any]:
    qualification = qualify_stage41(qualification_report)
    require_rung_order(list(dms_models))
    require_rung_order(list(megascale_models))
    assays, dms_units, dms_raw, dms_contrasts = align_dms(
        dms_models, lookup, require_fixed_census=require_fixed_census
    )
    dms = {
        "n_assays": len(assays),
        "unit": "wild-type family at 50% identity",
        "raw_spearman": _endpoint(dms_raw, dms_units, resamples=resamples, seed=seed),
        "model_minus_lookup": _endpoint(
            dms_contrasts["model_minus_lookup"], dms_units, resamples=resamples, seed=seed + 20
        ),
        "model_minus_blosum62": _endpoint(
            dms_contrasts["model_minus_blosum62"], dms_units, resamples=resamples, seed=seed + 40
        ),
    }

    mega_digest = _require_same_cohort(
        *megascale_models.values(), baselines, label="MegaScale model/baseline"
    )
    if fragment_order is not None:
        if "cohort_sha256" not in fragment_order:
            raise ValueError("fragment_order exists but carries no cohort_sha256")
        _require_same_cohort(
            megascale_models[SCALE_RUNGS[0]],
            fragment_order,
            label="MegaScale model/fragment_order",
        )
    megascale: dict[str, Any] = {
        "cohort_sha256": mega_digest,
        "hydropathy_baseline": HYDROPATHY_BASELINE,
    }
    for side, label, unit_name, offset in (
        ("design", "designs", "design series", 60),
        ("natural", "control", "WT_cluster", 80),
    ):
        units, raw, hydro = align_megascale(
            megascale_models,
            baselines,
            side=side,
            baseline_name=HYDROPATHY_BASELINE,
            require_fixed_census=require_fixed_census,
        )
        _, _, blosum = align_megascale(
            megascale_models,
            baselines,
            side=side,
            baseline_name="blosum62",
            require_fixed_census=require_fixed_census,
        )
        megascale[label] = {
            "unit": unit_name,
            "n_wildtypes": len(units),
            "raw_spearman": _endpoint(raw, units, resamples=resamples, seed=seed + offset),
            "model_minus_hydropathy": _endpoint(
                hydro, units, resamples=resamples, seed=seed + offset + 5
            ),
            "model_minus_blosum62": _endpoint(
                blosum, units, resamples=resamples, seed=seed + offset + 10
            ),
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "scale_capability_report",
        "created_utc": _timestamp(),
        "rungs": list(SCALE_RUNGS),
        "not_panel_admission": True,
        "staged_scale": {
            "scope": "progen2_training_lineage_medium_to_xlarge",
            "measured_staged_arms": list(STAGED_SCALE_ARMS),
            "scoring_target_alphabet": {
                name: {
                    "size": arm_spec(name).scoring_target_alphabet_size,
                    "source": "arm_spec.scoring_target_alphabet_size",
                }
                for name in STAGED_SCALE_ARMS
            },
        },
        "descriptive_not_causal": True,
        "descriptive_not_causal_note": DESCRIPTIVE_NOT_CAUSAL,
        "corpus_identification_bound": UNIREF90_BFD30_INCOMPLETE_SEARCH,
        "no_biological_knowledge_claim": True,
        "no_biological_knowledge_claim_note": NO_BIOLOGICAL_KNOWLEDGE_CLAIM,
        "no_cross_task_total": True,
        "bootstrap": {
            "resamples": int(resamples),
            "seed": int(seed),
            "default_seed": DEFAULT_BOOTSTRAP_SEED,
        },
        "qualification": qualification,
        "dms": dms,
        "megascale": megascale,
        "descriptive_gate_transitions": descriptive_gate_transitions(dms, megascale),
        "fragment_order": None if fragment_order is None else _fragment_margins(fragment_order),
    }
    return payload


def _load_rung_models(directory: Path, prefix: str) -> dict[str, dict[str, Any]]:
    models = {}
    for name in SCALE_RUNGS:
        path = directory / f"{prefix}{name}.json"
        payload = _read(path)
        if payload.get("arm") != name:
            raise ValueError(f"{path} declares arm {payload.get('arm')!r}, expected {name}")
        models[name] = payload
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieval-bound-dir",
        type=Path,
        required=True,
        help="directory holding model_<arm>.json and lookup.json from stage 20",
    )
    parser.add_argument(
        "--designed-referent-dir",
        type=Path,
        required=True,
        help="directory holding model_<arm>.json and baselines.json from stage 29",
    )
    parser.add_argument(
        "--context-information-summary",
        type=Path,
        required=True,
        help="full stage-41 report with arm_results; summary-only JSON is refused",
    )
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    qualification_report = _read(args.context_information_summary)
    qualify_stage41(qualification_report)

    dms_models = _load_rung_models(args.retrieval_bound_dir, "model_")
    lookup = _read(args.retrieval_bound_dir / "lookup.json")
    megascale_models = _load_rung_models(args.designed_referent_dir, "model_")
    baselines = _read(args.designed_referent_dir / "baselines.json")
    fragment_path = args.designed_referent_dir / "fragment_order.json"
    fragment_order = _read(fragment_path) if fragment_path.is_file() else None

    payload = compare_scale(
        dms_models=dms_models,
        lookup=lookup,
        megascale_models=megascale_models,
        baselines=baselines,
        fragment_order=fragment_order,
        qualification_report=qualification_report,
        resamples=args.bootstrap,
        seed=args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "scale_capability.json"
    write_json(destination, payload)
    print(f"wrote {destination}")
    gates = payload["descriptive_gate_transitions"]
    for pair, block in gates["dms"].items():
        print(f"DMS gate {pair}: {block['verdict']}")
    for pair, block in gates["megascale"].items():
        print(f"MegaScale gate {pair}: {block['verdict']}")


if __name__ == "__main__":
    main()
