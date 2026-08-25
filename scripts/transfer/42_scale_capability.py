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

Two cohorts here are narrower than the queues they come from, and both are
recorded rather than implied. ProteinGym declares 217 substitution assays over
174 families; 16 of them render longer than the 1024-position context every
ProGen2 rung shares, so the ladder is read on the 201 assays over 163 families
that every rung can score. MegaScale scores all 146 designs; F12's design census
is the 130 certified zero-hit designs, and that flag is carried by the stage-29
cohort rather than by the per-arm payloads.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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
from src.transfer.io import sha256_file, write_json  # noqa: E402

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
#:
#: The DMS cohort has two counts and they are not the same number. ProteinGym's
#: substitution queue is 217 assays over 174 wild-type families and is what the
#: freeze declares. Every rung of this ladder is a 1024-position ProGen2 with one
#: shared residue tokenizer, so the 16 assays whose rendered variant exceeds that
#: context cannot be scored on any of them; the ladder's analysis set is the 201
#: assays over 163 families that remain. Both are recorded, and the exclusion is
#: a property of the shared interface rather than of a checkpoint or of a score.
DMS_DECLARED_ASSAYS = 217
DMS_DECLARED_CLUSTERS = 174
DMS_ANALYSIS_ASSAYS = 201
DMS_ANALYSIS_CLUSTERS = 163
PROGEN2_CONTEXT = 1024
DMS_CONTEXT_EXCLUSION_REASON = "exceeds this arm's context"
DMS_CONTEXT_EXCLUDED_ASSAYS = (
    "A0A140D2T1_ZIKV_Sourisseau_2019",
    "BRCA1_HUMAN_Findlay_2018",
    "BRCA2_HUMAN_Erwood_2022_HEK293T",
    "CAR11_HUMAN_Meitlis_2020_gof",
    "CAR11_HUMAN_Meitlis_2020_lof",
    "CAS9_STRP1_Spencer_2017_positive",
    "ERBB2_HUMAN_Elazar_2016",
    "KCNH2_HUMAN_Kozek_2020",
    "NPC1_HUMAN_Erwood_2022_HEK293T",
    "NPC1_HUMAN_Erwood_2022_RPE1",
    "POLG_CXB3N_Mattenberger_2021",
    "POLG_HCVJF_Qi_2014",
    "SCN5A_HUMAN_Glazer_2019",
    "SPIKE_SARS2_Starr_2020_binding",
    "SPIKE_SARS2_Starr_2020_expression",
    "UBE4B_MOUSE_Starita_2013",
)
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
DMS_CONTEXT_EXCLUSION_NOTE = (
    "these assays render longer than the 1024-position context every rung of "
    "this ladder shares, so no rung can score them and truncating would score a "
    "sequence that may not contain the mutated position. The exclusion is a "
    "cohort definition fixed by the interface, not a filter on any score; the "
    "ladder's DMS reading covers the 201 assays and 163 families that remain, "
    "not all of ProteinGym's 217 and 174"
)
ZERO_HIT_DESIGN_NOTE = (
    "the design side is F12's census of certified zero-hit designs. "
    "baselines.json and model_<arm>.json score all 146 designs and carry no "
    "zero_hit flag, so the flag is read from the stage-29 cohort itself, whose "
    "SHA-256 is the cohort_sha256 those payloads already declare"
)
FRAGMENT_INCOMPLETE_NOTE = (
    "the stage-29 fragment_order pass carries margins only for the rungs in "
    "rungs_present. 3-7-mer margins are reported and never gated, so a rung "
    "missing from that pass does not stop a gate; it is named here rather than "
    "silently omitted"
)
TRANSITION_MEANS = (
    "the larger rung clears its own baseline contrast and is paired-"
    "significantly better than the smaller; this is not a claim that the "
    "smaller rung failed"
)
PRECISION_NOTE = (
    "one precision per endpoint across all three rungs. A rung scored at a "
    "different precision from the rung it is paired against would make the "
    "paired difference partly a difference of arithmetic"
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


def require_frozen_bootstrap(resamples: int, seed: int) -> None:
    """Refuse CLI changes to the pre-data EXP-R2-224 bootstrap freeze."""

    if resamples != BOOTSTRAP_RESAMPLES or seed != DEFAULT_BOOTSTRAP_SEED:
        raise ValueError(
            "EXP-R2-224 freezes stage 42 at "
            f"resamples={BOOTSTRAP_RESAMPLES}, seed={DEFAULT_BOOTSTRAP_SEED}; "
            f"got resamples={resamples}, seed={seed}"
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


def context_excluded_assays(
    models: dict[str, dict[str, Any]], lookup_set: set[str]
) -> set[str]:
    """The assays no rung can score, refused unless every rung skips the same set.

    A rendered variant longer than the 1024-position context every ProGen2 rung
    shares is unscorable on all three, so skipping it defines the cohort. A skip
    one rung takes and another does not would break the pairing the comparison
    rests on, and a skip taken for any other reason is a scoring failure rather
    than a cohort definition. Both are refusals.
    """

    per_rung: dict[str, set[str]] = {}
    for name in SCALE_RUNGS:
        skipped = models[name].get("skipped")
        if skipped is None:
            raise ValueError(
                f"{name} carries no skipped block; stage 20 writes one on every "
                "scored arm and its absence cannot be read as an empty skip"
            )
        names: set[str] = set()
        for row in skipped:
            assay = str(row["assay"])
            if assay in names:
                raise ValueError(f"{name} repeats skipped assay {assay}")
            reason = str(row.get("reason", ""))
            if DMS_CONTEXT_EXCLUSION_REASON not in reason:
                raise ValueError(
                    f"{name} skipped {assay} for {reason!r}; only a rendered "
                    "variant exceeding this ladder's shared context defines the "
                    "cohort, every other skip is a scoring failure"
                )
            if int(row["context"]) != PROGEN2_CONTEXT:
                raise ValueError(
                    f"{name} skipped {assay} against context {row['context']}, "
                    f"not this ladder's shared {PROGEN2_CONTEXT}"
                )
            if int(row["max_tokens"]) <= PROGEN2_CONTEXT:
                raise ValueError(
                    f"{name} skipped {assay} at {row['max_tokens']} tokens, "
                    f"which fits the {PROGEN2_CONTEXT}-position context"
                )
            names.add(assay)
        per_rung[name] = names
    reference = per_rung[SCALE_RUNGS[0]]
    for name, names in per_rung.items():
        if names != reference:
            raise ValueError(
                f"{name} and {SCALE_RUNGS[0]} disagree on whether "
                f"{sorted(names ^ reference)} can be scored; a paired "
                "comparison needs one common support across the rungs"
            )
    outside = sorted(reference - lookup_set)
    if outside:
        raise ValueError(f"skipped assays {outside} are not in the LOOKUP cohort")
    return reference


def align_dms(
    models: dict[str, dict[str, Any]],
    lookup: dict[str, Any],
    *,
    require_fixed_census: bool = True,
) -> dict[str, Any]:
    """Align the DMS cohort: three model assay sets equal the jointly scored subset.

    LOOKUP declares ProteinGym's 217-assay substitution queue. The rungs score
    the 201 of those assays that fit the 1024-position context they share. The
    returned record carries both, so the artefact can say which cohort was
    declared and which was analysed instead of implying they are the same set.
    """

    require_rung_order(list(models))
    lookup_assays = _assay_names(lookup, label="LOOKUP")
    lookup_set = set(lookup_assays)
    excluded = context_excluded_assays(models, lookup_set)
    analysis_assays = [name for name in lookup_assays if name not in excluded]
    analysis_set = set(analysis_assays)
    for name in SCALE_RUNGS:
        model_set = set(_assay_names(models[name], label=name))
        if model_set != analysis_set:
            raise ValueError(
                f"{name} assay set disagrees with the jointly scored LOOKUP "
                f"subset: {sorted(model_set ^ analysis_set)}"
            )
    by_assay = {row["assay"]: row for row in lookup["assays"]}
    units: dict[str, str] = {}
    raw: dict[str, dict[str, float]] = {name: {} for name in SCALE_RUNGS}
    contrasts: dict[str, dict[str, dict[str, float]]] = {
        "model_minus_lookup": {name: {} for name in SCALE_RUNGS},
        "model_minus_blosum62": {name: {} for name in SCALE_RUNGS},
    }
    for assay in analysis_assays:
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
    declared_clusters = {str(row["cluster"]) for row in lookup["assays"]}
    if require_fixed_census:
        if len(lookup_assays) != DMS_DECLARED_ASSAYS:
            raise ValueError(
                f"DMS cohort is {len(lookup_assays)} assays, not the fixed "
                f"{DMS_DECLARED_ASSAYS}-assay ProteinGym substitution census"
            )
        if len(declared_clusters) != DMS_DECLARED_CLUSTERS:
            raise ValueError(
                f"LOOKUP has {len(declared_clusters)} clusters, not the fixed "
                f"{DMS_DECLARED_CLUSTERS}-family census"
            )
        if sorted(excluded) != list(DMS_CONTEXT_EXCLUDED_ASSAYS):
            raise ValueError(
                "the context-excluded assays are not the frozen EXP-R2-224 set: "
                f"{sorted(set(excluded) ^ set(DMS_CONTEXT_EXCLUDED_ASSAYS))}"
            )
        if len(analysis_assays) != DMS_ANALYSIS_ASSAYS:
            raise ValueError(
                f"the jointly scored DMS set is {len(analysis_assays)} assays, "
                f"not the fixed {DMS_ANALYSIS_ASSAYS}"
            )
        if len(set(units.values())) != DMS_ANALYSIS_CLUSTERS:
            raise ValueError(
                f"the jointly scored DMS set spans {len(set(units.values()))} "
                f"families, not the fixed {DMS_ANALYSIS_CLUSTERS}"
            )
    return {
        "declared_assays": lookup_assays,
        "declared_clusters": len(declared_clusters),
        "context_excluded_assays": sorted(excluded),
        "analysis_assays": analysis_assays,
        "units": units,
        "raw": raw,
        "contrasts": contrasts,
    }


def zero_hit_design_cohort(path: Path) -> dict[str, Any]:
    """F12's design census, read from the stage-29 cohort that carries the flag.

    ``zero_hit`` lives on the cohort's wild types and on neither ``baselines``
    nor ``model_<arm>``, both of which score all 146 designs. Reading it here
    reaches the same 130 certified zero-hit designs through the same
    :meth:`Referent.side` rule the rest of stage 29 uses, and the file's SHA-256
    is the ``cohort_sha256`` those payloads already declare, so the input cannot
    drift without the digest check firing.
    """

    referent = D.load_referent(path)
    names = sorted(wildtype.name for wildtype in referent.side("design"))
    if not names:
        raise ValueError(f"{path} carries no certified zero-hit design")
    return {
        "source": path.name,
        "cohort_sha256": sha256_file(path),
        "zero_hit_designs": names,
        "note": ZERO_HIT_DESIGN_NOTE,
    }


def _side_keys(
    payload: dict[str, Any],
    *,
    side: str,
    spearman_of,
    admissible: frozenset[str] | None = None,
) -> set[str]:
    keys = set()
    for name, entry in payload["wildtypes"].items():
        if entry.get("kind") != side:
            continue
        if admissible is not None and name not in admissible:
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
    admissible: frozenset[str] | None = None,
    require_fixed_census: bool = True,
) -> tuple[dict[str, str], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Align one MegaScale side: non-null keys equal across rungs and the baseline.

    ``admissible`` restricts the side to a declared census. The design side is
    restricted to the certified zero-hit designs; on the natural control, where
    hitting the corpus is the expected state, it has no meaning and is ``None``.
    """

    require_rung_order(list(models))
    digests = {name: payload["cohort_sha256"] for name, payload in models.items()}
    if len(set(digests.values())) != 1:
        raise ValueError(f"MegaScale cohort_sha256 disagrees across rungs: {digests}")
    model_keys = {
        name: _side_keys(
            payload,
            side=side,
            spearman_of=lambda entry: entry.get("spearman"),
            admissible=admissible,
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
    if admissible is not None and first_keys != set(admissible):
        raise ValueError(
            f"the declared {side} census is not scored on every rung: "
            f"{sorted(first_keys ^ set(admissible))}"
        )
    baseline_keys = _side_keys(
        baselines,
        side=side,
        spearman_of=lambda entry: (entry.get("spearman") or {}).get(baseline_name),
        admissible=admissible,
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


def require_uniform_dtype(models: dict[str, dict[str, Any]], *, label: str) -> str:
    """One scoring precision across the three rungs of one endpoint.

    A rung scored at a different precision from the rung it is paired against
    makes the paired difference partly a difference of arithmetic rather than of
    checkpoints. Nothing downstream can detect that from the numbers, so it is
    refused here, where every rung's own record of what it was scored at is in
    hand.
    """

    seen: dict[str, str] = {}
    for name in SCALE_RUNGS:
        settings = models[name].get("settings")
        dtype = settings.get("dtype") if isinstance(settings, Mapping) else None
        if not dtype:
            raise ValueError(f"{label}: {name} records no scoring dtype")
        seen[name] = str(dtype)
    if len(set(seen.values())) != 1:
        raise ValueError(f"{label}: the rungs were scored at mixed precision: {seen}")
    return next(iter(seen.values()))


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
    present = [name for name in SCALE_RUNGS if name in arms]
    missing = [name for name in SCALE_RUNGS if name not in arms]
    margins: dict[str, Any] = {"designs": {}, "control": {}}
    for side in ("designs", "control"):
        for order in supported:
            for scheme in schemes:
                key = D.fragment_channel_name(order, scheme)
                per_rung = {}
                for rung in present:
                    block = arms[rung].get(side) or {}
                    if key not in block:
                        raise ValueError(
                            f"fragment_order {rung} {side} lacks {key} though "
                            f"k={order} is supported"
                        )
                    per_rung[rung] = block[key]
                if per_rung:
                    margins[side][key] = {"per_rung": per_rung, "supported_to_k": order}
    return {
        "highest_supported_order": highest,
        "supported_orders": supported,
        "schemes": schemes,
        "margins": margins,
        "reported_not_gated": True,
        "rungs_present": present,
        "rungs_missing": missing,
        "incomplete_note": FRAGMENT_INCOMPLETE_NOTE if missing else None,
        "cohort_sha256": fragment_order.get("cohort_sha256"),
    }


def _positive_finite_interval(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        interval = [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(endpoint) for endpoint in interval) or interval[0] <= 0.0:
        return None
    return interval


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
    for name in SCALE_RUNGS:
        if len(block_sets[name]) != len(by_rung[name]):
            raise ValueError(f"stage-41 arm_results repeats a block for {name}")
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
            interval = _positive_finite_interval(
                row.get("displacement_corrected_ci_95")
            )
            if status != "PASS" or interval is None:
                raise ValueError(
                    f"{name} block {row.get('block_id')} identification is "
                    f"{status!r}, not PASS with a finite displacement-corrected "
                    "interval strictly above zero"
                )
            per_block[row["block_id"]] = {
                "per_arm_identification_status": status,
                "displacement_corrected_ci_95": list(interval),
                "cohort_digest": row["cohort_digest"],
                "cohort_name": row.get("cohort_name"),
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
    """The one pre-registered gate family. No per-endpoint copies.

    A transition asks the LARGER rung to clear its own baseline contrast and the
    paired delta to exclude zero. It never asks the smaller rung to fail: a rule
    that required the smaller rung's own contrast to be non-significant would
    make the gate depend on accepting a null, so a well-powered pair in which
    both rungs are competent could not register a transition however large the
    improvement. The smaller rung's contrasts are therefore reported beside each
    gate rather than folded into it.
    """

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
            "transition_means": TRANSITION_MEANS,
            "reported_not_gated": {
                "smaller_model_minus_lookup": lower_bound_positive(
                    dms["model_minus_lookup"]["per_rung"][smaller]
                )
            },
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
            "transition_means": TRANSITION_MEANS,
            "reported_not_gated": {
                "natural_raw_spearman_delta": lower_bound_positive(
                    megascale["control"]["raw_spearman"]["adjacent_delta_rho"][pair]
                ),
                "smaller_design_model_minus_hydropathy": lower_bound_positive(
                    megascale["designs"]["model_minus_hydropathy"]["per_rung"][smaller]
                ),
                "smaller_design_model_minus_blosum62": lower_bound_positive(
                    megascale["designs"]["model_minus_blosum62"]["per_rung"][smaller]
                ),
            },
        }
    return {"dms": dms_gates, "megascale": mega_gates}


def compare_scale(
    *,
    dms_models: dict[str, dict[str, Any]],
    lookup: dict[str, Any],
    megascale_models: dict[str, dict[str, Any]],
    baselines: dict[str, Any],
    design_cohort: dict[str, Any],
    fragment_order: dict[str, Any] | None,
    qualification_report: dict[str, Any],
    resamples: int,
    seed: int,
    require_fixed_census: bool = True,
) -> dict[str, Any]:
    qualification = qualify_stage41(qualification_report)
    require_rung_order(list(dms_models))
    require_rung_order(list(megascale_models))
    dms_dtype = require_uniform_dtype(dms_models, label="DMS")
    mega_dtype = require_uniform_dtype(megascale_models, label="MegaScale")
    alignment = align_dms(
        dms_models, lookup, require_fixed_census=require_fixed_census
    )
    dms_units = alignment["units"]
    dms_raw = alignment["raw"]
    dms_contrasts = alignment["contrasts"]
    dms = {
        "declared_cohort": {
            "n_assays": len(alignment["declared_assays"]),
            "n_clusters": alignment["declared_clusters"],
            "source": "ProteinGym substitution assays, F10's units",
        },
        "context_excluded_assays": alignment["context_excluded_assays"],
        "context_exclusion_note": DMS_CONTEXT_EXCLUSION_NOTE,
        "n_assays": len(alignment["analysis_assays"]),
        "n_clusters": len(set(dms_units.values())),
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
    if design_cohort["cohort_sha256"] != mega_digest:
        raise ValueError(
            "the zero-hit design census was read from a different cohort: "
            f"{design_cohort['cohort_sha256']} against {mega_digest}"
        )
    design_names = frozenset(design_cohort["zero_hit_designs"])
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
        "design_census": {
            "source": design_cohort["source"],
            "n_zero_hit_designs": len(design_names),
            "note": design_cohort["note"],
        },
    }
    for side, label, unit_name, offset in (
        ("design", "designs", "design series", 60),
        ("natural", "control", "WT_cluster", 80),
    ):
        admissible = design_names if side == "design" else None
        units, raw, hydro = align_megascale(
            megascale_models,
            baselines,
            side=side,
            baseline_name=HYDROPATHY_BASELINE,
            admissible=admissible,
            require_fixed_census=require_fixed_census,
        )
        _, _, blosum = align_megascale(
            megascale_models,
            baselines,
            side=side,
            baseline_name="blosum62",
            admissible=admissible,
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
            "resamples": resamples,
            "seed": seed,
            "default_seed": DEFAULT_BOOTSTRAP_SEED,
        },
        "precision": {
            "dms": dms_dtype,
            "megascale": mega_dtype,
            "note": PRECISION_NOTE,
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
        help=(
            "directory holding model_<arm>.json, baselines.json and cohort.json "
            "from stage 29"
        ),
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
    require_frozen_bootstrap(args.bootstrap, args.seed)

    qualification_report = _read(args.context_information_summary)
    qualify_stage41(qualification_report)

    dms_models = _load_rung_models(args.retrieval_bound_dir, "model_")
    lookup = _read(args.retrieval_bound_dir / "lookup.json")
    megascale_models = _load_rung_models(args.designed_referent_dir, "model_")
    baselines = _read(args.designed_referent_dir / "baselines.json")
    design_cohort = zero_hit_design_cohort(args.designed_referent_dir / "cohort.json")
    fragment_path = args.designed_referent_dir / "fragment_order.json"
    fragment_order = _read(fragment_path) if fragment_path.is_file() else None

    payload = compare_scale(
        dms_models=dms_models,
        lookup=lookup,
        megascale_models=megascale_models,
        baselines=baselines,
        design_cohort=design_cohort,
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
