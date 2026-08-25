#!/usr/bin/env python3
"""Descriptive ProGen2 medium→large→xlarge comparison. Not a causal scale claim.

This stage is CPU-only. It consumes existing stage-20 and stage-29 artefacts
and, optionally, a stage-41 qualification summary. It does not load a model,
does not download weights, and does not synthesise a cross-task total.

The comparison is fixed as progen2-medium → progen2-large → progen2-xlarge.
Checkpoint differences are descriptive of those checkpoints. They are not a
parameter-count causal effect: corpus identification is a model-favouring bound
on UniRef90+BFD30, and nothing here is a claim about biological knowledge.
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

SCHEMA_VERSION = "r2_transfer_scale_capability_v1"
DEFAULT_OUT = REPO / "results/transfer/scale_capability"
SCALE_RUNGS = ("progen2-medium", "progen2-large", "progen2-xlarge")
ADJACENT_PAIRS = (
    ("progen2-medium", "progen2-large"),
    ("progen2-large", "progen2-xlarge"),
)
BOOTSTRAP_RESAMPLES = 2000
HYDROPATHY_BASELINE = "hydropathy_change"
FRAGMENT_MAX_ORDER = 7

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
    """Refuse any comparison that is not exactly medium, large and xlarge."""

    got = list(names)
    if set(got) != set(SCALE_RUNGS):
        raise ValueError(
            f"the descriptive comparison is fixed as {list(SCALE_RUNGS)}; "
            f"got {got}"
        )


def descriptive_gate_transition(
    larger_contrast: dict[str, Any],
    delta: dict[str, Any],
) -> str | bool:
    """True only when the larger rung beats the baseline and Δρ's 95% floor is > 0."""

    larger_interval = larger_contrast.get("interval")
    delta_interval = delta.get("interval")
    if (
        larger_contrast.get("degenerate")
        or delta.get("degenerate")
        or larger_interval is None
        or delta_interval is None
    ):
        return "unresolved"
    larger_beats = bool(larger_interval[0] > 0.0)
    delta_positive = bool(delta_interval[0] > 0.0)
    return bool(larger_beats and delta_positive)


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
    unit_names = [units[name] for name in names]
    if any(units[name] != unit_names[index] for index, name in enumerate(names)):
        raise ValueError("unit labels are not aligned with the paired keys")
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
        adjacent[f"{smaller}__{larger}"] = {
            **delta,
            "descriptive_gate_transition": descriptive_gate_transition(rungs[larger], delta),
        }
    return {"per_rung": rungs, "adjacent_delta_rho": adjacent}


def align_dms(
    models: dict[str, dict[str, Any]],
    lookup: dict[str, Any],
) -> tuple[
    list[str],
    dict[str, str],
    dict[str, dict[str, float]],
    dict[str, dict[str, dict[str, float]]],
]:
    """Align assays across rungs on name and mutant_digest; return units and scores."""

    require_rung_order(list(models))
    by_assay = {row["assay"]: row for row in lookup["assays"]}
    first = models[SCALE_RUNGS[0]]["assays"]
    assays = [row["assay"] for row in first]
    if len(assays) != len(set(assays)):
        raise ValueError("a DMS model file repeats an assay")
    units: dict[str, str] = {}
    raw: dict[str, dict[str, float]] = {name: {} for name in SCALE_RUNGS}
    contrasts: dict[str, dict[str, dict[str, float]]] = {
        "model_minus_lookup": {name: {} for name in SCALE_RUNGS},
        "model_minus_blosum62": {name: {} for name in SCALE_RUNGS},
    }
    for assay in assays:
        lookup_row = by_assay.get(assay)
        if lookup_row is None:
            raise ValueError(f"{assay}: scored by a model rung but absent from LOOKUP")
        expected_digest = None
        expected_cluster = str(lookup_row["cluster"])
        for rung, payload in models.items():
            rows = {row["assay"]: row for row in payload["assays"]}
            if assay not in rows:
                raise ValueError(f"{assay}: missing from {rung}")
            entry = rows[assay]
            if entry["mutant_digest"] != lookup_row["mutant_digest"]:
                raise ValueError(
                    f"{assay}: mutant_digest disagrees between {rung} and LOOKUP"
                )
            if expected_digest is None:
                expected_digest = entry["mutant_digest"]
            elif entry["mutant_digest"] != expected_digest:
                raise ValueError(f"{assay}: mutant_digest disagrees across rungs")
            rho = float(entry["spearman"])
            raw[rung][assay] = rho
            contrasts["model_minus_lookup"][rung][assay] = (
                rho - float(lookup_row["spearman"]["lookup"])
            )
            contrasts["model_minus_blosum62"][rung][assay] = (
                rho - float(lookup_row["spearman"]["blosum62"])
            )
        units[assay] = expected_cluster
    for payload in models.values():
        extra = sorted({row["assay"] for row in payload["assays"]} - set(assays))
        if extra:
            raise ValueError(f"rung {payload['arm']} has extra assays {extra}")
    return assays, units, raw, contrasts


def align_megascale(
    models: dict[str, dict[str, Any]],
    baselines: dict[str, dict[str, Any]],
    *,
    side: str,
    baseline_name: str,
) -> tuple[dict[str, str], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Align wild types on one MegaScale side; check digest, kind and unit."""

    require_rung_order(list(models))
    digests = {name: payload["cohort_sha256"] for name, payload in models.items()}
    if len(set(digests.values())) != 1:
        raise ValueError(f"MegaScale cohort_sha256 disagrees across rungs: {digests}")
    first = models[SCALE_RUNGS[0]]
    names = sorted(
        name
        for name, entry in first["wildtypes"].items()
        if entry.get("kind") == side and entry.get("spearman") is not None
    )
    units: dict[str, str] = {}
    raw: dict[str, dict[str, float]] = {name: {} for name in SCALE_RUNGS}
    contrast: dict[str, dict[str, float]] = {name: {} for name in SCALE_RUNGS}
    for wildtype in names:
        baseline_entry = baselines["wildtypes"].get(wildtype)
        if baseline_entry is None:
            raise ValueError(f"{wildtype}: missing from baselines")
        base = baseline_entry["spearman"].get(baseline_name)
        if base is None:
            raise ValueError(f"{wildtype}: baseline {baseline_name} is missing")
        expected_unit = None
        expected_kind = side
        for rung, payload in models.items():
            if payload["cohort_sha256"] != first["cohort_sha256"]:
                raise ValueError(f"{rung}: cohort_sha256 mismatch")
            entry = payload["wildtypes"].get(wildtype)
            if entry is None or entry.get("spearman") is None:
                raise ValueError(f"{wildtype}: missing from {rung}")
            if entry["kind"] != expected_kind:
                raise ValueError(f"{wildtype}: kind disagrees on {rung}")
            if expected_unit is None:
                expected_unit = str(entry["unit"])
            elif str(entry["unit"]) != expected_unit:
                raise ValueError(f"{wildtype}: unit disagrees across rungs")
            rho = float(entry["spearman"])
            raw[rung][wildtype] = rho
            contrast[rung][wildtype] = rho - float(base)
        units[wildtype] = expected_unit or ""
    return units, raw, contrast


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
    }


def _qualification(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    entries = summary.get("arms") or []
    by_arm = {entry["arm"]: entry for entry in entries if "arm" in entry}
    record = {"source": "stage41", "rungs": {}}
    for rung in SCALE_RUNGS:
        entry = by_arm.get(rung)
        if entry is None:
            raise ValueError(
                f"stage-41 summary does not account for {rung}; pass it in "
                "--expected-arms when re-analysing the staged power records"
            )
        record["rungs"][rung] = {
            "status": entry.get("status"),
            "present": entry.get("present"),
        }
    return record


def compare_scale(
    *,
    dms_models: dict[str, dict[str, Any]],
    lookup: dict[str, Any],
    megascale_models: dict[str, dict[str, Any]],
    baselines: dict[str, Any],
    fragment_order: dict[str, Any] | None,
    qualification: dict[str, Any] | None,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    require_rung_order(list(dms_models))
    require_rung_order(list(megascale_models))
    assays, dms_units, dms_raw, dms_contrasts = align_dms(dms_models, lookup)
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

    megascale: dict[str, Any] = {
        "cohort_sha256": megascale_models[SCALE_RUNGS[0]]["cohort_sha256"],
        "hydropathy_baseline": HYDROPATHY_BASELINE,
    }
    for side, label, unit_name, offset in (
        ("design", "designs", "design series", 60),
        ("natural", "control", "WT_cluster", 80),
    ):
        units, raw, hydro = align_megascale(
            megascale_models, baselines, side=side, baseline_name=HYDROPATHY_BASELINE
        )
        _, _, blosum = align_megascale(
            megascale_models, baselines, side=side, baseline_name="blosum62"
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
        "bootstrap": {"resamples": int(resamples), "seed": int(seed)},
        "qualification": _qualification(qualification),
        "dms": dms,
        "megascale": megascale,
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
        default=None,
        help="optional stage-41 summary; rungs must appear under --expected-arms",
    )
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    dms_models = _load_rung_models(args.retrieval_bound_dir, "model_")
    lookup = _read(args.retrieval_bound_dir / "lookup.json")
    megascale_models = _load_rung_models(args.designed_referent_dir, "model_")
    baselines = _read(args.designed_referent_dir / "baselines.json")
    fragment_path = args.designed_referent_dir / "fragment_order.json"
    fragment_order = _read(fragment_path) if fragment_path.is_file() else None
    qualification = (
        None
        if args.context_information_summary is None
        else _read(args.context_information_summary)
    )
    if baselines.get("cohort_sha256") != megascale_models[SCALE_RUNGS[0]]["cohort_sha256"]:
        raise ValueError("MegaScale baselines and model scores were computed on different cohorts")

    payload = compare_scale(
        dms_models=dms_models,
        lookup=lookup,
        megascale_models=megascale_models,
        baselines=baselines,
        fragment_order=fragment_order,
        qualification=qualification,
        resamples=args.bootstrap,
        seed=args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "scale_capability.json"
    write_json(destination, payload)
    print(f"wrote {destination}")
    for pair, block in payload["dms"]["model_minus_lookup"]["adjacent_delta_rho"].items():
        print(
            f"DMS MODEL-LOOKUP {pair}: gate={block['descriptive_gate_transition']} "
            f"Δρ={block.get('point')}"
        )


if __name__ == "__main__":
    main()
