#!/usr/bin/env python3
"""The uncertainty audit of context information, run over statistics already on disk.

``docs/CONTEXT_INFORMATION_UNCERTAINTY_PREREGISTRATION.md`` is the frozen analysis
plan and this stage is its executable form. It loads no model, touches no GPU and
reaches no cluster: everything below is arithmetic over the per-record sufficient
statistics ``01_cohort_power.py --record-statistics`` persists beside its report,
which is the whole point of that sidecar. A second sweep of the panel buys nothing
this file cannot compute.

What it produces, in the order the pre-registration declares it.

**A group-level paired interval for I per arm per cohort draw.** The resampling
unit is the near-duplicate group, one cohort draw feeds both terms of
``I = H_baseline - H_model``, the reference is resampled independently and the
smoothed unigram is refitted inside every iteration. All of that lives in
``src.transfer.information_bootstrap``; this stage's job is to assemble the arm
inputs honestly, and the one thing it must not do is invent a grouping. Group
assignments are not persisted on the base cohort path, so they are recomputed here
from the cohort records the companion ``cohort_*.json`` holds. Where those records
are unavailable the fallback to singleton groups is written into every affected
record as a declared limitation rather than applied quietly, because a
record-level interval is narrowest exactly where group dependence is strongest.

**Contrasts that respect where pairing is defined.** Arms sharing one cohort draw
are bootstrapped in one ``bootstrap_arms`` call under common resample indices, so
their contrast is formed inside the iteration. Arms on different cohorts have no
common resampling unit and go through ``unpaired_contrast`` with that fact
attached. The panel does not sit on one cohort -- text arms on OpenWebText,
ProtGPT2 and the residue-level ProGen2 arms on Swiss-Prot, ZymCTRL on the
EC-labelled draw -- so this distinction is the difference between a contrast and a
fiction. The ProGen2-base / ProGen2-medium pair is named explicitly and its sign
is tracked across every draw, because four readings of that gap exist on disk and
one of the four has the opposite sign.

**The three sensitivities the plan asks for.** Between-block spread is reported
beside the within-block interval and never folded into it; below eight blocks it
is reported as a spread with its degrees of freedom and explicitly not as an
interval, which is what the K = 2 pairs on disk have already been misread as. The
alpha sweep is pure CPU here because smoothing is applied at analysis time from
the persisted reference counts, so the vocabulary-dependent smoothing bias whose
published figures were withdrawn as unreproducible is measured rather than quoted,
and carried beside its analytic bound ``log(1 + aV/R)``. The leakage-removed arm
drops every reference record sharing a 30-mer with a cohort record; the sidecar
carries no sequence text, so it runs only when the reference records are supplied
and is emitted as explicitly unavailable otherwise.

Nothing is omitted silently. An arm whose cohort has too few effective groups is
refused with the reason attached, an arm absent from every sidecar is named as
absent rather than left out of the table, and a sensitivity that could not be
computed says why.

The sign criterion is reported and is **not** evidence. It is expected to pass on
every arm, and the pre-registration explains why: I carries a smoothing bias that
grows with vocabulary size and that no bootstrap can touch, so the sign test is
safe exactly where it is redundant and unreliable exactly where it would decide
something. Every record carrying ``sign_status`` carries that beside it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.arms import PANEL  # noqa: E402
from src.transfer.budget import (  # noqa: E402
    MIN_CONTEXT_INFORMATION_NATS,
    POWER_RECORDS_SCHEMA_VERSION,
)
from src.transfer.information_bootstrap import (  # noqa: E402
    DEFAULT_BOOTSTRAP_DRAWS,
    ESTIMAND,
    ArmPanel,
    ArmStatistics,
    CohortStatistics,
    ReferenceStatistics,
    SparseCounts,
    bootstrap_arms,
    unpaired_contrast,
)
from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.near_duplicates import (  # noqa: E402
    NEAR_DUPLICATE_CONTAINMENT,
    near_duplicate_groups,
)
from src.transfer.pathways import SMOOTHING_SWEEP  # noqa: E402
from src.transfer.statistics import MINIMUM_BOOTSTRAP_UNITS  # noqa: E402

SCHEMA_VERSION = "r2_transfer_context_information_bootstrap_v1"

DEFAULT_OUT = REPO_ROOT / "results/transfer/context_information_bootstrap"

#: Modules whose contents decide the numbers in the artefact, hashed into the
#: metadata so that a report can be attributed to the arithmetic that produced it.
HASHED_MODULES: tuple[str, ...] = (
    "information_bootstrap.py",
    "near_duplicates.py",
    "budget.py",
    "pathways.py",
    "statistics.py",
    "arms.py",
)

#: Added to a panel's seed for its leakage-removed re-run. That panel has a
#: different reference group universe, so its draw is a different draw whatever
#: seed it is given; the offset keeps it from colliding with another panel's
#: seed, which is what would make a contrast between the two neither independent
#: nor paired.
LEAKAGE_SEED_OFFSET = 1_000_003

#: Blocks required before a between-block spread is also reported as an interval.
#: E2's target, and the reason it is eight rather than two: with K = 2 the
#: variance component has one degree of freedom, and the two K = 2 pairs on disk
#: disagree with each other about its size by a factor of fifty.
MIN_BLOCKS_FOR_BETWEEN_BLOCK_INTERVAL = 8

#: k of the k-mer the leakage-removed sensitivity screens the reference on.
DEFAULT_LEAKAGE_KMER = 30

SIGN_STATUS_NOTE = (
    "NON-EVIDENTIAL. The sign criterion is reported and not adopted. It is "
    "expected to pass on every arm and must not be read as a gate: I carries a "
    "smoothing bias that grows with vocabulary size and that no bootstrap can "
    "touch, so on an arm whose true context information sits near zero the sign "
    "of the measured I would be decided by the smoothing constant and the "
    "vocabulary rather than by the model. The operative screening gate this round "
    "is the 0.30 nats/token floor reported with an interval; see "
    "docs/CONTEXT_INFORMATION_UNCERTAINTY_PREREGISTRATION.md."
)

SCREENING_NOTE = (
    "Screening, not confirmatory. A criterion here may keep an arm out of a "
    "downstream stage and may say that a denominator is too small to divide by. "
    "It may not be cited as evidence that an arm does or does not extract "
    "information from context, and no number in this artefact may be used to "
    "re-choose the floor it was measured against."
)

COMPARABILITY_NOTE = (
    "Nats per token is not cross-arm comparable and no interval repairs that: a "
    "unigram over merged BPE pieces already encodes the character-level "
    "dependencies inside each piece, so I under two tokenisations is two "
    "estimands. Cross-arm statements are restricted to arms sharing a "
    "tokenisation regime, or are made on relative_information with the caveat "
    "carried."
)

#: Cross-arm contrasts the pre-registration names, with what it says about them.
#: Every other within-cohort pair is still formed -- ``bootstrap_arms`` defaults
#: to all of them -- but these carry their standing note into the record.
DECLARED_CONTRASTS: tuple[dict[str, str], ...] = (
    {
        "left": "progen2-base",
        "right": "progen2-medium",
        "preregistration_note": (
            "The ordering of progen2-base and progen2-medium by context "
            "information is NOT SUPPORTED and must not be reported. Four readings "
            "of this gap exist on disk (+0.047, -0.024, +0.077, +0.095 nats) and "
            "one of the four has the opposite sign; the two campaigns also "
            "disagree about the arms' levels by more than three times the gap "
            "being read, and no reading carries a valid interval."
        ),
    },
)


# --------------------------------------------------------------------------- #
# Reading the sidecar and its companions
# --------------------------------------------------------------------------- #


def _scalar(array: Any) -> Any:
    return np.asarray(array).item()


def select_records(counts: SparseCounts, keep: np.ndarray) -> SparseCounts:
    """``counts`` restricted to the records ``keep`` selects, offsets rebuilt.

    Used twice: to drop reference records carrying no scored target, and to drop
    the ones the leakage screen removes. Both are removals of whole records from
    the resampling population, and both are counted into the artefact.
    """

    keep = np.asarray(keep, dtype=bool)
    if keep.size != counts.n_records:
        raise ValueError("the selection mask does not align with the records")
    starts = counts.record_offsets[:-1][keep]
    stops = counts.record_offsets[1:][keep]
    lengths = (stops - starts).astype(np.int64)
    entries = (
        np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in zip(starts, stops)])
        if lengths.size and int(lengths.sum())
        else np.zeros(0, dtype=np.int64)
    )
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(lengths)])
    return SparseCounts(
        unique_token_ids=counts.unique_token_ids[entries],
        counts=counts.counts[entries],
        record_offsets=offsets,
    )


def cohort_json_for(sidecar: Path) -> Path:
    """The frozen-cohort file ``01_cohort_power.py`` writes beside the sidecar.

    Derived rather than searched for: the producing stage names the two files
    from one cohort name and one digest prefix, so the mapping is a rename, and a
    reader that has to guess is a reader that can pick up another cohort's
    records.
    """

    name = sidecar.name
    if not name.startswith("power_") or not name.endswith(".records.npz"):
        raise ValueError(
            f"{sidecar} is not named power_<cohort>_<digest>.records.npz, so the "
            "companion cohort file cannot be derived; pass --cohort-json"
        )
    stem = name[len("power_") : -len(".records.npz")]
    return sidecar.parent / f"cohort_{stem}.json"


def read_record_strings(path: Path) -> list[str]:
    """A record list from a frozen-cohort file or from a bare JSON list."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("records")
    if not isinstance(records, list) or not all(isinstance(r, str) for r in records):
        raise ValueError(f"{path} carries no list of record strings")
    return records


def grouping_of(
    records: list[str] | None,
    *,
    kind: str,
    containment: float,
    shingle: int | None,
    what: str,
    unavailable_reason: str | None,
    n_records: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Near-duplicate groups over ``records``, or singletons with the reason said.

    The fallback is not a quiet one. A singleton grouping resamples records, and
    a record-level interval is *narrower* than a group-level one exactly where
    the group structure matters, so an artefact that fell back without saying so
    would report its most confident-looking number in the case the group unit
    exists to catch.
    """

    if records is not None:
        unit = "residues" if kind == "protein" else "characters"
        groups, summary = near_duplicate_groups(
            records, unit=unit, containment=containment, shingle=shingle
        )
        return groups, {
            "available": True,
            "what": what,
            "computed_from": (
                "the persisted records, by near_duplicates.near_duplicate_groups"
            ),
            **summary,
        }
    return np.arange(max(int(n_records), 0), dtype=np.int64), {
        "available": False,
        "what": what,
        "fallback": "singleton groups, one record per resampling unit",
        "reason": unavailable_reason,
        "declared_limitation": (
            f"DECLARED LIMITATION: the {what} records were not available to this "
            "re-analysis, so each record was resampled as its own unit. "
            "Near-duplicate structure is therefore ignored, the interval is "
            "narrower than the evidence supports, and it is not an interval at "
            "the unit of dependence the pre-registration freezes. It is reported "
            "so that the fallback is visible, not so that it can be quoted."
        ),
        "n_records": int(n_records),
        "n_groups": int(n_records),
    }


@dataclass(frozen=True)
class ArmInput:
    """One arm of one cohort block, assembled or explicitly refused."""

    name: str
    statistics: ArmStatistics | None
    load_refusal: str | None
    vocab_size: int
    reference_tokens: int
    reference_records_dropped_empty: int
    reference_keep: np.ndarray | None


@dataclass(frozen=True)
class Block:
    """One sidecar: one cohort draw, its arms and the groupings they resample."""

    index: int
    sidecar: Path
    sidecar_sha256: str
    cohort_digest: str
    cohort_name: str
    cohort_kind: str
    reference_digest: str | None
    smoothing: float
    max_len: int
    producer_seeds: dict[str, int]
    arms: list[ArmInput]
    cohort_records: list[str] | None
    reference_records: list[str] | None
    grouping: dict[str, Any]
    reference_grouping: dict[str, Any]

    @property
    def block_id(self) -> str:
        return f"b{self.index}"


def load_block(
    index: int,
    sidecar: Path,
    cohort_json: Path | None,
    reference_json: Path | None,
    *,
    requested_arms: list[str] | None,
    containment: float,
    shingle: int | None,
) -> Block:
    """Assemble one sidecar into arm inputs, refusing rather than repairing."""

    npz = np.load(sidecar)
    version = str(_scalar(npz["schema_version"]))
    if version != POWER_RECORDS_SCHEMA_VERSION:
        raise ValueError(
            f"{sidecar} declares schema {version!r} against this stage's "
            f"{POWER_RECORDS_SCHEMA_VERSION!r}; the field list is the whole of the "
            "sidecar's meaning, so rebuild rather than reinterpret"
        )
    cohort_digest = str(_scalar(npz["cohort_digest"]))
    reference_digest = str(_scalar(npz["reference_digest"])) or None
    smoothing = float(_scalar(npz["smoothing"]))
    max_len = int(_scalar(npz["max_len"]))
    seed_names = [str(name) for name in np.asarray(npz["seed_names"]).tolist()]
    seed_values = [int(value) for value in np.asarray(npz["seed_values"]).tolist()]
    present = [str(name) for name in np.asarray(npz["arms"]).tolist()]
    names = (
        present if requested_arms is None else [n for n in present if n in requested_arms]
    )

    cohort_records: list[str] | None = None
    cohort_unavailable: str | None = None
    cohort_name, cohort_kind = "unknown", "protein"
    path = cohort_json if cohort_json is not None else cohort_json_for(sidecar)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            stored = payload.get("cohort_digest")
            if stored is not None and stored != cohort_digest:
                raise ValueError(
                    f"{path} holds cohort {str(stored)[:12]} and {sidecar} was "
                    f"written against {cohort_digest[:12]}; grouping one cohort's "
                    "records and resampling another's statistics is not a grouping"
                )
            cohort_name = str(payload.get("cohort_name", cohort_name))
            cohort_kind = str(payload.get("cohort_kind", cohort_kind))
        cohort_records = read_record_strings(path)
    else:
        cohort_unavailable = (
            f"the companion frozen-cohort file {path} does not exist, and the "
            "sidecar carries no sequence text of its own"
        )

    reference_records: list[str] | None = None
    reference_unavailable: str | None = (
        "no --reference-json was supplied for this sidecar; the sidecar carries "
        "only order-free token counts, so the reference text has to come from "
        "the reference_<name>_<digest>.json 01_cohort_power.py writes"
    )
    if reference_json is not None:
        if not reference_json.is_file():
            raise ValueError(f"--reference-json names a missing file: {reference_json}")
        reference_records = read_record_strings(reference_json)
        reference_unavailable = None

    n_cohort_records = (
        len(cohort_records)
        if cohort_records is not None
        else int(
            max(
                (int(npz[f"{name}::record_index"].max()) + 1 for name in names),
                default=0,
            )
        )
    )
    cohort_groups, grouping = grouping_of(
        cohort_records,
        kind=cohort_kind,
        containment=containment,
        shingle=shingle,
        what="cohort",
        unavailable_reason=cohort_unavailable,
        n_records=n_cohort_records,
    )

    n_reference_records = 0
    for name in names:
        key = f"{name}::reference_token_count"
        if key in npz.files:
            n_reference_records = max(n_reference_records, int(npz[key].size))
    if reference_records is not None and len(reference_records) != n_reference_records:
        raise ValueError(
            f"{reference_json} holds {len(reference_records)} records against the "
            f"sidecar's {n_reference_records} reference rows; the two do not "
            "describe the same reference set"
        )
    reference_groups, reference_grouping = grouping_of(
        reference_records,
        kind=cohort_kind,
        containment=containment,
        shingle=shingle,
        what="reference",
        unavailable_reason=reference_unavailable,
        n_records=n_reference_records,
    )

    arms = [
        build_arm_input(npz, name, cohort_groups, reference_groups, smoothing)
        for name in names
    ]
    return Block(
        index=index,
        sidecar=sidecar,
        sidecar_sha256=sha256_file(sidecar),
        cohort_digest=cohort_digest,
        cohort_name=cohort_name,
        cohort_kind=cohort_kind,
        reference_digest=reference_digest,
        smoothing=smoothing,
        max_len=max_len,
        producer_seeds=dict(zip(seed_names, seed_values)),
        arms=arms,
        cohort_records=cohort_records,
        reference_records=reference_records,
        grouping=grouping,
        reference_grouping=reference_grouping,
    )


def build_arm_input(
    npz: Any,
    name: str,
    cohort_groups: np.ndarray,
    reference_groups: np.ndarray,
    smoothing: float,
) -> ArmInput:
    """One arm's ``ArmStatistics``, or the reason it cannot have one."""

    prefix = f"{name}::"
    vocab_size = int(_scalar(npz[f"{prefix}vocab_size"]))
    if f"{prefix}reference_counts_offsets" not in npz.files:
        return ArmInput(
            name=name,
            statistics=None,
            load_refusal=(
                "the sidecar carries no reference counts for this arm, so the "
                "held-out unigram this estimand is defined against does not exist "
                "in it. The plug-in baseline is a different estimand and is not "
                "substituted here"
            ),
            vocab_size=vocab_size,
            reference_tokens=0,
            reference_records_dropped_empty=0,
            reference_keep=None,
        )
    if not math.isfinite(smoothing):
        return ArmInput(
            name=name,
            statistics=None,
            load_refusal=(
                "the sidecar records no smoothing constant, which its writer does "
                "only when no reference entered any figure; the baseline cannot be "
                "refitted without it"
            ),
            vocab_size=vocab_size,
            reference_tokens=0,
            reference_records_dropped_empty=0,
            reference_keep=None,
        )

    record_index = np.asarray(npz[f"{prefix}record_index"], dtype=np.int64)
    if record_index.size and int(record_index.max()) >= cohort_groups.size:
        raise ValueError(
            f"{name}: the sidecar scores cohort record {int(record_index.max())} "
            f"but the grouping covers {cohort_groups.size} records"
        )
    cohort = CohortStatistics(
        clean_nll_sum=npz[f"{prefix}clean_nll_sum"],
        token_count=npz[f"{prefix}token_count"],
        n_symbols=npz[f"{prefix}n_symbols"],
        targets=SparseCounts(
            unique_token_ids=npz[f"{prefix}unique_token_ids"],
            counts=npz[f"{prefix}counts"],
            record_offsets=npz[f"{prefix}counts_offsets"],
        ),
        group_id=cohort_groups[record_index],
    )

    reference_targets = SparseCounts(
        unique_token_ids=npz[f"{prefix}reference_unique_token_ids"],
        counts=npz[f"{prefix}reference_counts"],
        record_offsets=npz[f"{prefix}reference_counts_offsets"],
    )
    stored_tokens = np.asarray(npz[f"{prefix}reference_token_count"], dtype=np.int64)
    if not np.array_equal(reference_targets.record_totals, stored_tokens):
        raise ValueError(
            f"{name}: the sidecar's reference token counts disagree with the counts "
            "they are supposed to sum to"
        )
    if stored_tokens.size != reference_groups.size:
        raise ValueError(
            f"{name}: {stored_tokens.size} reference rows against a grouping over "
            f"{reference_groups.size} records"
        )
    keep = stored_tokens > 0
    if not bool(keep.any()):
        return ArmInput(
            name=name,
            statistics=None,
            load_refusal="every reference record carries zero scored targets",
            vocab_size=vocab_size,
            reference_tokens=0,
            reference_records_dropped_empty=int(stored_tokens.size),
            reference_keep=None,
        )
    return ArmInput(
        name=name,
        statistics=ArmStatistics(
            name=name,
            cohort=cohort,
            reference=ReferenceStatistics(
                token_count=stored_tokens[keep],
                targets=select_records(reference_targets, keep),
                group_id=reference_groups[keep],
            ),
            vocab_size=vocab_size,
            smoothing=smoothing,
        ),
        load_refusal=None,
        vocab_size=vocab_size,
        reference_tokens=int(stored_tokens.sum()),
        reference_records_dropped_empty=int((~keep).sum()),
        reference_keep=keep,
    )


# --------------------------------------------------------------------------- #
# Panels: where a common resample index is defined, and where it is not
# --------------------------------------------------------------------------- #


def panel_signature(arm: ArmStatistics) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The group universes a common resample index has to address."""

    return (
        tuple(int(value) for value in np.unique(arm.cohort.group_id)),
        tuple(int(value) for value in np.unique(arm.reference.group_id)),
    )


@dataclass(frozen=True)
class Panel:
    """Arms of one cohort block sharing one resample index, and their result."""

    panel_id: str
    block: Block
    seed: int
    statistics: list[ArmStatistics]
    result: ArmPanel

    @property
    def names(self) -> list[str]:
        return [arm.name for arm in self.statistics]


def partition_block(block: Block) -> list[list[ArmStatistics]]:
    """Arms of one block grouped by the universe a common index would address.

    Normally one partition: every arm of a cohort scores the same records, so the
    group universes coincide. They can fail to when an arm's tokenisation leaves a
    record with no scored target, and then a common index would not address the
    same units. ``bootstrap_arms`` refuses that outright, so the partition is made
    here and the arms that fall out of the main one are bootstrapped separately
    and reported as not contrastable against it -- rather than being handed to
    ``unpaired_contrast``, which is right for independent cohorts and wrong for
    arms that share records.
    """

    partitions: dict[tuple[tuple[int, ...], tuple[int, ...]], list[ArmStatistics]] = {}
    for entry in block.arms:
        if entry.statistics is None:
            continue
        partitions.setdefault(panel_signature(entry.statistics), []).append(entry.statistics)
    return list(partitions.values())


def bootstrap_block(
    block: Block, first_seed: int, *, n_bootstrap: int, confidence: float
) -> list[Panel]:
    """Every panel of one block, each under its own seed."""

    panels: list[Panel] = []
    for ordinal, statistics in enumerate(partition_block(block)):
        seed = first_seed + ordinal
        panels.append(
            Panel(
                panel_id=f"{block.block_id}p{ordinal}",
                block=block,
                seed=seed,
                statistics=statistics,
                result=bootstrap_arms(
                    statistics,
                    seed=seed,
                    n_bootstrap=n_bootstrap,
                    confidence=confidence,
                ),
            )
        )
    return panels


# --------------------------------------------------------------------------- #
# Per-arm rows
# --------------------------------------------------------------------------- #


def load_refusal_record(name: str, reason: str) -> dict[str, Any]:
    """The shape ``arm_row`` reads, for an arm that never reached the bootstrap."""

    return {
        "arm": name,
        "refused": True,
        "refusal_reason": reason,
        "statistics": None,
        "seed": None,
        "n_bootstrap": None,
        "confidence": None,
        "resampling_unit": "group",
        "unit_floor": None,
        "diagnostics": None,
        "cohort_draw_shared_between_terms": True,
        "reference_resampled": True,
    }


def arm_row(
    block: Block,
    panel_id: str | None,
    record: dict[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    """One arm on one cohort draw: the estimate, its interval and how to read it."""

    statistics = record["statistics"]
    information = statistics["information_nats_per_token"] if statistics else None
    relative = statistics["relative_information"] if statistics else None
    bits = statistics["information_bits_per_symbol"] if statistics else None
    diagnostics = record.get("diagnostics") or {}

    point = None if information is None else float(information["point"])
    low, high = (None, None) if information is None else information["interval"]
    if information is None:
        legacy, legacy_interval, sign = "REFUSED", "REFUSED", "REFUSED"
        disagreement = None
    else:
        legacy = "PASS" if point >= threshold else "FAIL"
        legacy_interval = (
            "above"
            if low >= threshold
            else "below"
            if high < threshold
            else "straddles"
        )
        sign = "PASS" if low > 0.0 else "FAIL"
        disagreement = legacy != sign
    return {
        "block_id": block.block_id,
        "panel_id": panel_id,
        "arm": record["arm"],
        "sidecar": str(block.sidecar),
        "cohort_digest": block.cohort_digest,
        "cohort_name": block.cohort_name,
        "cohort_kind": block.cohort_kind,
        "reference_digest": block.reference_digest,
        "smoothing": block.smoothing if math.isfinite(block.smoothing) else None,
        "refused": bool(record["refused"]),
        "refusal_reason": record["refusal_reason"],
        "seed": record["seed"],
        "n_bootstrap": record["n_bootstrap"],
        "confidence": record["confidence"],
        "resampling_unit": record["resampling_unit"],
        "cohort_draw_shared_between_terms": record["cohort_draw_shared_between_terms"],
        "reference_resampled": record["reference_resampled"],
        "context_information_nats": point,
        "bootstrap_ci_95": None if information is None else list(information["interval"]),
        "bootstrap_fraction_nonpositive": (
            None
            if information is None
            else 1.0 - float(information["fraction_of_draws_positive"])
        ),
        "bootstrap_se": None if information is None else information["bootstrap_se"],
        "bootstrap_bias": None if information is None else information["bootstrap_bias"],
        "median_bias_z0": None if information is None else information["median_bias_z0"],
        "interval_mc_se": None if information is None else information["interval_mc_se"],
        "relative_information": None if relative is None else relative["point"],
        "relative_information_ci_95": None if relative is None else list(relative["interval"]),
        "information_bits_per_symbol": None if bits is None else bits["point"],
        "information_bits_per_symbol_ci_95": (
            None if bits is None else list(bits["interval"])
        ),
        "n_scored_records": diagnostics.get("n_records"),
        "n_scored_tokens": diagnostics.get("n_scored_tokens"),
        "n_reference_records": diagnostics.get("n_reference_records"),
        "n_reference_tokens": diagnostics.get("n_reference_tokens"),
        "n_groups": diagnostics.get("n_groups"),
        "n_effective_groups": diagnostics.get("n_effective_groups"),
        "largest_group_token_share": diagnostics.get("largest_group_token_share"),
        "n_singleton_groups": diagnostics.get("n_singleton_groups"),
        "top10_record_token_share": diagnostics.get("top10_record_token_share"),
        "cohort_token_share_unseen_in_reference": diagnostics.get(
            "cohort_token_share_unseen_in_reference"
        ),
        "cohort_token_share_reference_count_at_most_5": diagnostics.get(
            "cohort_token_share_reference_count_at_most_5"
        ),
        "reference_n_effective_groups": diagnostics.get("reference_n_effective_groups"),
        "legacy_threshold_nats": float(threshold),
        "legacy_threshold_status": legacy,
        "legacy_threshold_interval_status": legacy_interval,
        "sign_status": sign,
        "sign_status_is_evidential": False,
        "sign_status_note": SIGN_STATUS_NOTE,
        "status_disagreement": disagreement,
        "unit_floor": record["unit_floor"],
        "diagnostics": record["diagnostics"],
        "cohort_grouping": block.grouping,
        "reference_grouping": block.reference_grouping,
        "statistics": statistics,
        "comparability_note": COMPARABILITY_NOTE,
    }


# --------------------------------------------------------------------------- #
# Contrasts
# --------------------------------------------------------------------------- #


def declared_note(left: str, right: str) -> str | None:
    """The pre-registration's standing note on a named pair, if it has one."""

    for entry in DECLARED_CONTRASTS:
        if {entry["left"], entry["right"]} == {left, right}:
            return entry["preregistration_note"]
    return None


def within_cohort_contrasts(panels: list[Panel]) -> list[dict[str, Any]]:
    """Every pair of arms sharing a cohort draw, differenced inside the iteration."""

    rows: list[dict[str, Any]] = []
    for panel in panels:
        for key, record in panel.result.contrasts.items():
            rows.append(
                {
                    "key": key,
                    "block_id": panel.block.block_id,
                    "panel_id": panel.panel_id,
                    "cohort_digest": panel.block.cohort_digest,
                    "cohort_name": panel.block.cohort_name,
                    "seed": panel.seed,
                    "preregistration_note": declared_note(record["left"], record["right"]),
                    **record,
                }
            )
    return rows


def cross_cohort_contrasts(
    panels: list[Panel], *, confidence: float, requested: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    """Differences between arms on different cohort draws, labelled unpaired.

    Restricted to the pairs that mean something: the same arm on two draws, which
    is the block sensitivity E2 is about; the pairs the pre-registration names;
    and whatever ``--contrast`` asks for. An exhaustive cross-cohort product would
    be thousands of intervals that are each conservative by an unknown amount and
    none of which anybody asked for.
    """

    wanted = {frozenset(pair) for pair in requested}
    rows: list[dict[str, Any]] = []
    for left_panel, right_panel in combinations(panels, 2):
        if left_panel.block.block_id == right_panel.block.block_id:
            continue
        for left in left_panel.names:
            for right in right_panel.names:
                note = declared_note(left, right)
                if not (left == right or note or frozenset((left, right)) in wanted):
                    continue
                record = unpaired_contrast(
                    left_panel.result.arms[left],
                    right_panel.result.arms[right],
                    confidence=confidence,
                )
                rows.append(
                    {
                        "key": f"{left}@{left_panel.block.block_id}"
                        f"_minus_{right}@{right_panel.block.block_id}",
                        "left_block_id": left_panel.block.block_id,
                        "right_block_id": right_panel.block.block_id,
                        "left_cohort_digest": left_panel.block.cohort_digest,
                        "right_cohort_digest": right_panel.block.cohort_digest,
                        "same_arm_across_draws": left == right,
                        "preregistration_note": note,
                        **record,
                    }
                )
    return rows


def cross_panel_refusals(panels: list[Panel]) -> list[dict[str, Any]]:
    """Arms of one cohort whose group universes differ, so no index is common.

    Not handed to ``unpaired_contrast``: that interval is right for independent
    cohorts and too wide for arms that in fact share records, and these arms share
    records. The refusal names the arms and says why, which is the only honest
    thing available without re-deriving a common universe the estimand does not
    have.
    """

    rows: list[dict[str, Any]] = []
    for left_panel, right_panel in combinations(panels, 2):
        if left_panel.block.block_id != right_panel.block.block_id:
            continue
        for left in left_panel.names:
            for right in right_panel.names:
                rows.append(
                    {
                        "key": f"{left}_minus_{right}",
                        "block_id": left_panel.block.block_id,
                        "left": left,
                        "right": right,
                        "refused": True,
                        "refusal_reason": (
                            "these arms share a cohort draw but not a group "
                            "universe, so no common resample index addresses the "
                            "same units; an unpaired interval is not offered "
                            "instead, because it is correct for independent "
                            "cohorts and too wide for arms that share records"
                        ),
                        "statistics": None,
                    }
                )
    return rows


def declared_contrast_sign_tracking(
    within: list[dict[str, Any]], across: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Where a named pair's gap lands on every draw, and whether its sign holds."""

    tracked: list[dict[str, Any]] = []
    for entry in DECLARED_CONTRASTS:
        readings: list[dict[str, Any]] = []
        for record in [*within, *across]:
            if {record.get("left"), record.get("right")} != {entry["left"], entry["right"]}:
                continue
            if record.get("refused") or not record.get("statistics"):
                readings.append(
                    {
                        "key": record["key"],
                        "paired": record.get("paired"),
                        "gap_nats": None,
                        "refused": True,
                    }
                )
                continue
            block = record["statistics"]["information_nats_per_token"]
            oriented = 1.0 if record["left"] == entry["left"] else -1.0
            readings.append(
                {
                    "key": record["key"],
                    "paired": record.get("paired"),
                    "gap_nats": oriented * float(block["point"]),
                    "interval": sorted(oriented * value for value in block["interval"]),
                    "straddles_zero": block["interval"][0] < 0.0 < block["interval"][1],
                    "refused": False,
                }
            )
        signs = {
            (gap > 0.0)
            for gap in (r["gap_nats"] for r in readings)
            if gap is not None and gap != 0.0
        }
        tracked.append(
            {
                "left": entry["left"],
                "right": entry["right"],
                "orientation": f"gap = I({entry['left']}) - I({entry['right']})",
                "preregistration_note": entry["preregistration_note"],
                "readings": readings,
                "n_readings": len(readings),
                "sign_is_stable_across_readings": (None if not signs else len(signs) == 1),
                "ordering_may_be_reported": False,
            }
        )
    return tracked


# --------------------------------------------------------------------------- #
# Between-block variance
# --------------------------------------------------------------------------- #


def block_disjointness(blocks: list[Block]) -> dict[str, Any]:
    """Whether the supplied cohort draws actually hold different records."""

    missing = [b.block_id for b in blocks if b.cohort_records is None]
    if missing:
        return {
            "checked": False,
            "reason": (
                "the frozen-cohort records are unavailable for "
                f"{sorted(missing)}, so block disjointness cannot be verified and "
                "the spread below may mix a between-block component with a "
                "re-reading of overlapping records"
            ),
            "pairs": [],
            "all_disjoint": None,
        }
    pairs = []
    for left, right in combinations(blocks, 2):
        shared = len(set(left.cohort_records) & set(right.cohort_records))
        pairs.append(
            {
                "left": left.block_id,
                "right": right.block_id,
                "shared_records": int(shared),
                "disjoint": shared == 0,
            }
        )
    return {
        "checked": True,
        "reason": None,
        "pairs": pairs,
        "all_disjoint": all(pair["disjoint"] for pair in pairs),
    }


def between_block_variance(
    rows: list[dict[str, Any]], blocks: dict[str, Block]
) -> list[dict[str, Any]]:
    """The block-selection component, reported beside the interval and not inside it.

    The pre-registration is explicit that a variance component estimated from K
    blocks does not belong inside a percentile interval computed from one, so
    nothing here is combined with anything. Below eight blocks the spread is
    reported as a spread with its degrees of freedom and the interval is refused:
    at K = 2 the component has one degree of freedom, and the two K = 2 pairs
    already on disk disagree with each other about its size by a factor of fifty.
    """

    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["context_information_nats"] is None:
            continue
        by_arm.setdefault(row["arm"], []).append(row)

    records: list[dict[str, Any]] = []
    for arm in sorted(by_arm):
        entries = sorted(by_arm[arm], key=lambda row: row["block_id"])
        points = [float(row["context_information_nats"]) for row in entries]
        k = len(points)
        common: dict[str, Any] = {
            "arm": arm,
            "n_blocks": k,
            "blocks": [
                {
                    "block_id": row["block_id"],
                    "cohort_digest": row["cohort_digest"],
                    "cohort_name": row["cohort_name"],
                    "context_information_nats": row["context_information_nats"],
                    "within_block_bootstrap_ci_95": row["bootstrap_ci_95"],
                }
                for row in entries
            ],
            "folded_into_the_bootstrap_interval": False,
            "note": (
                "the bootstrap covers within-cohort and within-reference sampling "
                "error only; this component is block-selection error and is "
                "reported beside it, never inside it"
            ),
        }
        if k < 2:
            records.append(
                {
                    **common,
                    "reported": False,
                    "reason": (
                        "one cohort block was supplied for this arm, so there is no "
                        "between-block component to estimate"
                    ),
                    "between_block_sd_nats": None,
                    "degrees_of_freedom": 0,
                    "between_block_interval": None,
                }
            )
            continue
        spread = float(np.std(points, ddof=1))
        disjoint = block_disjointness([blocks[row["block_id"]] for row in entries])
        interval: list[float] | None = None
        refusal: str | None = (
            f"K = {k} blocks gives {k - 1} degree(s) of freedom, below the "
            f"{MIN_BLOCKS_FOR_BETWEEN_BLOCK_INTERVAL}-block floor this stage "
            "requires before a between-block interval is published. The spread and "
            "its degrees of freedom are reported instead. A K = 2 pair is a "
            "difference, not an interval, and the experiment log forbids "
            "presenting it as one."
        )
        if k >= MIN_BLOCKS_FOR_BETWEEN_BLOCK_INTERVAL:
            from scipy import stats as _stats  # local: only this branch needs it

            half = float(_stats.t.ppf(0.975, k - 1)) * spread / math.sqrt(k)
            mean = float(np.mean(points))
            interval, refusal = [mean - half, mean + half], None
        records.append(
            {
                **common,
                "reported": True,
                "reason": None,
                "mean_nats": float(np.mean(points)),
                "between_block_sd_nats": spread,
                "between_block_range_nats": float(max(points) - min(points)),
                "degrees_of_freedom": k - 1,
                "blocks_disjoint": disjoint,
                "between_block_interval": interval,
                "between_block_interval_refusal": refusal,
            }
        )
    return records


# --------------------------------------------------------------------------- #
# E4: the smoothing sweep
# --------------------------------------------------------------------------- #


def unsmoothed_information(arm: ArmStatistics) -> tuple[float | None, str | None]:
    """``I`` against the unsmoothed held-out unigram, when one exists.

    It exists only when every cohort target token was seen in the reference. One
    unseen target makes the maximum-likelihood unigram assign it zero and the
    cross-entropy infinite, which is the reason the smoothing constant cannot be
    eliminated and can only be swept. Where it exists it is the fixed point the
    sweep is measured against: ``H_baseline(a) - H_baseline(0)`` is exactly
    ``sum_v c(v)/N [log(1 + aV/R) - log(1 + a/r(v))]``, which is at most
    ``log(1 + aV/R)`` because the second term is non-negative. That inequality is
    the analytic bound this stage reports and checks, rather than a claim about
    monotonicity, which does not hold in general: a token whose reference count is
    below ``R/V`` has its surprisal *reduced* by smoothing.
    """

    vocabulary = arm.vocab_size
    cohort = np.bincount(
        arm.cohort.targets.unique_token_ids,
        weights=arm.cohort.targets.counts.astype(np.float64),
        minlength=vocabulary,
    )
    reference = np.bincount(
        arm.reference.targets.unique_token_ids,
        weights=arm.reference.targets.counts.astype(np.float64),
        minlength=vocabulary,
    )
    support = cohort > 0
    if bool(np.any(reference[support] <= 0.0)):
        return None, (
            "at least one cohort target token is unseen in the reference, so the "
            "unsmoothed held-out unigram is infinite and the sweep has no "
            "zero-smoothing anchor on this arm"
        )
    tokens = float(cohort.sum())
    total = float(reference.sum())
    baseline = float(
        -(cohort[support] * (np.log(reference[support]) - math.log(total))).sum() / tokens
    )
    model = float(arm.cohort.clean_nll_sum.sum() / arm.cohort.token_count.sum())
    return baseline - model, None


def alpha_sensitivity(
    panels: list[Panel],
    *,
    sweep: list[float],
    n_bootstrap: int,
    confidence: float,
) -> list[dict[str, Any]]:
    """``I`` re-bootstrapped across the smoothing ladder, per arm and per draw.

    The whole sweep is a CPU job because smoothing is applied at analysis time
    from the persisted reference counts, which is what makes the withdrawn
    published figures replaceable by measurement here. Every alpha runs at the
    panel's own seed, so the draws are identical across the ladder and a
    difference between two alphas is a paired difference rather than two
    independent bootstraps.
    """

    records: list[dict[str, Any]] = []
    for panel in panels:
        by_alpha = {
            float(alpha): bootstrap_arms(
                [replace(arm, smoothing=float(alpha)) for arm in panel.statistics],
                seed=panel.seed,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                contrasts=(),
            )
            for alpha in sweep
        }
        for arm in panel.statistics:
            reference_tokens = float(arm.reference.token_count.sum())
            unsmoothed, unsmoothed_reason = unsmoothed_information(arm)
            entries: list[dict[str, Any]] = []
            for alpha in sweep:
                statistics = by_alpha[float(alpha)].arms[arm.name].record["statistics"]
                bound = math.log1p(float(alpha) * arm.vocab_size / reference_tokens)
                information = (
                    None
                    if statistics is None
                    else float(statistics["information_nats_per_token"]["point"])
                )
                excess = (
                    None
                    if information is None or unsmoothed is None
                    else information - unsmoothed
                )
                entries.append(
                    {
                        "alpha": float(alpha),
                        "information_nats": information,
                        "bootstrap_ci_95": (
                            None
                            if statistics is None
                            else list(statistics["information_nats_per_token"]["interval"])
                        ),
                        "baseline_entropy_nats_per_token": (
                            None
                            if statistics is None
                            else statistics["baseline_entropy_nats_per_token"]["point"]
                        ),
                        "analytic_bound_nats": bound,
                        "excess_over_unsmoothed_nats": excess,
                        "within_analytic_bound": (
                            None if excess is None else bool(excess <= bound + 1e-9)
                        ),
                    }
                )
            measured = [e["information_nats"] for e in entries if e["information_nats"] is not None]
            ordered = sorted(
                (e for e in entries if e["information_nats"] is not None),
                key=lambda e: e["alpha"],
            )
            records.append(
                {
                    "block_id": panel.block.block_id,
                    "panel_id": panel.panel_id,
                    "arm": arm.name,
                    "cohort_digest": panel.block.cohort_digest,
                    "seed": panel.seed,
                    "vocab_size": int(arm.vocab_size),
                    "reference_tokens": int(reference_tokens),
                    "headline_smoothing": float(arm.smoothing),
                    "unsmoothed_information_nats": unsmoothed,
                    "unsmoothed_unavailable_reason": unsmoothed_reason,
                    "by_alpha": entries,
                    "information_range_nats": (
                        None if not measured else float(max(measured) - min(measured))
                    ),
                    "nondecreasing_in_alpha": (
                        None
                        if len(ordered) < 2
                        else all(
                            later["information_nats"] >= earlier["information_nats"] - 1e-12
                            for earlier, later in zip(ordered, ordered[1:])
                        )
                    ),
                    "note": (
                        "the analytic bound log(1 + aV/R) bounds how far smoothing "
                        "can push I above its unsmoothed value; it is not a "
                        "monotonicity claim, because a token whose reference count "
                        "is below R/V has its surprisal reduced by smoothing. No "
                        "bias correction is applied: the constant cannot be "
                        "eliminated, only swept and reported"
                    ),
                }
            )
    return records


# --------------------------------------------------------------------------- #
# E5: the leakage-removed sensitivity
# --------------------------------------------------------------------------- #


def kmer_set(record: str, k: int) -> set[str]:
    return {record[i : i + k] for i in range(len(record) - k + 1)} if len(record) >= k else set()


def leaked_reference_mask(
    cohort_records: list[str], reference_records: list[str], k: int
) -> np.ndarray:
    """Reference records sharing at least one k-mer with any cohort record.

    ``pathways.assert_disjoint`` checks exact content equality, which does not
    reach a reference record that is a near-copy of a scored one. Such a record
    makes the fitted unigram fit the cohort better than a held-out one should,
    which lowers ``H_baseline`` and deflates ``I`` -- unequally across arms, since
    the protein cohorts carry far more near-duplicate structure than the text
    ones.
    """

    pool: set[str] = set()
    for record in cohort_records:
        pool |= kmer_set(record, k)
    return np.asarray(
        [bool(kmer_set(record, k) & pool) for record in reference_records], dtype=bool
    )


def leakage_sensitivity(
    panels: list[Panel],
    *,
    k: int,
    n_bootstrap: int,
    confidence: float,
) -> list[dict[str, Any]]:
    """``I`` recomputed with the leaked part of the reference dropped, or refused."""

    records: list[dict[str, Any]] = []
    for panel in panels:
        block = panel.block
        inputs = {entry.name: entry for entry in block.arms}
        if block.reference_records is None or block.cohort_records is None:
            reason = (
                "the 30-mer overlap between reference and cohort cannot be "
                "determined from what is on disk: the sidecar stores order-free "
                "token counts, not sequence text, and "
                + (
                    "no --reference-json was supplied; 01_cohort_power.py "
                    "persists the held-out reference beside the cohort as "
                    "reference_<name>_<digest>.json, so pass that file"
                    if block.reference_records is None
                    else f"the frozen-cohort file for {block.block_id} is missing"
                )
                + ". The sensitivity is reported as unavailable rather than "
                "approximated from token counts, which cannot see a shared k-mer"
            )
            for arm in panel.statistics:
                records.append(
                    {
                        "block_id": block.block_id,
                        "panel_id": panel.panel_id,
                        "arm": arm.name,
                        "kmer": int(k),
                        "available": False,
                        "reason": reason,
                        "information_nats": None,
                        "bootstrap_ci_95": None,
                        "delta_vs_headline_nats": None,
                    }
                )
            continue

        leaked = leaked_reference_mask(block.cohort_records, block.reference_records, k)
        for arm in panel.statistics:
            entry = inputs[arm.name]
            keep = ~leaked[entry.reference_keep]
            headline = panel.result.arms[arm.name].record["statistics"]
            headline_point = (
                None
                if headline is None
                else float(headline["information_nats_per_token"]["point"])
            )
            common = {
                "block_id": block.block_id,
                "panel_id": panel.panel_id,
                "arm": arm.name,
                "kmer": int(k),
                "available": True,
                "n_reference_records_before": int(keep.size),
                "n_reference_records_dropped": int((~keep).sum()),
                "headline_information_nats": headline_point,
                "note": (
                    "a sensitivity beside the headline, not a replacement for it; "
                    "the headline stays on the declared reference"
                ),
            }
            if not bool(keep.any()):
                records.append(
                    {
                        **common,
                        "reason": (
                            "every reference record shares a "
                            f"{k}-mer with the cohort, so no held-out reference "
                            "survives the screen"
                        ),
                        "information_nats": None,
                        "bootstrap_ci_95": None,
                        "delta_vs_headline_nats": None,
                    }
                )
                continue
            reduced = replace(
                arm,
                reference=ReferenceStatistics(
                    token_count=arm.reference.token_count[keep],
                    targets=select_records(arm.reference.targets, keep),
                    group_id=arm.reference.group_id[keep],
                ),
            )
            result = bootstrap_arms(
                [reduced],
                seed=panel.seed + LEAKAGE_SEED_OFFSET,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                contrasts=(),
            ).arms[arm.name]
            statistics = result.record["statistics"]
            point = (
                None
                if statistics is None
                else float(statistics["information_nats_per_token"]["point"])
            )
            records.append(
                {
                    **common,
                    "seed": panel.seed + LEAKAGE_SEED_OFFSET,
                    "reason": result.record["refusal_reason"],
                    "refused": bool(result.record["refused"]),
                    "n_reference_tokens_after": int(reduced.reference.token_count.sum()),
                    "information_nats": point,
                    "bootstrap_ci_95": (
                        None
                        if statistics is None
                        else list(statistics["information_nats_per_token"]["interval"])
                    ),
                    "delta_vs_headline_nats": (
                        None
                        if point is None or headline_point is None
                        else point - headline_point
                    ),
                    "unit_floor": result.record["unit_floor"],
                }
            )
    return records


# --------------------------------------------------------------------------- #
# The comparison table
# --------------------------------------------------------------------------- #


def summarise(
    rows: list[dict[str, Any]],
    *,
    expected_arms: list[str],
    alpha_records: list[dict[str, Any]],
    leakage_records: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    """One row per arm: the primary result, the sensitivities, and what changes.

    An arm with no record at all is named here as ``ABSENT``. Four of the fifteen
    panel arms have never been through the held-out estimator, and an empty cell
    in a comparison table is read as a pass far more often than it is read as a
    gap; naming it is the only thing that stops that.
    """

    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    alpha_by_arm: dict[str, list[dict[str, Any]]] = {}
    for record in alpha_records:
        alpha_by_arm.setdefault(record["arm"], []).append(record)
    leakage_by_arm: dict[str, list[dict[str, Any]]] = {}
    for record in leakage_records:
        leakage_by_arm.setdefault(record["arm"], []).append(record)

    entries: list[dict[str, Any]] = []
    for arm in sorted(set(expected_arms) | set(by_arm)):
        measured = by_arm.get(arm, [])
        verdicts = [row["legacy_threshold_status"] for row in measured]
        passes = [v for v in verdicts if v == "PASS"]
        fails = [v for v in verdicts if v == "FAIL"]
        if not measured:
            status = "ABSENT"
        elif not passes and not fails:
            status = "REFUSED"
        elif passes and fails:
            status = "INCONSISTENT"
        elif passes:
            status = "PASS"
        else:
            status = "FAIL"
        ranges = [
            record["information_range_nats"]
            for record in alpha_by_arm.get(arm, [])
            if record["information_range_nats"] is not None
        ]
        entries.append(
            {
                "arm": arm,
                "present": bool(measured),
                "status": status,
                "absent_reason": (
                    None
                    if measured
                    else (
                        "no supplied cohort-power sidecar carries this arm, so it "
                        "has not been measured on this estimand here. Absence is "
                        "not a pass"
                    )
                ),
                "blocks": [
                    {
                        "block_id": row["block_id"],
                        "cohort_name": row["cohort_name"],
                        "cohort_digest": row["cohort_digest"],
                        "context_information_nats": row["context_information_nats"],
                        "bootstrap_ci_95": row["bootstrap_ci_95"],
                        "legacy_threshold_status": row["legacy_threshold_status"],
                        "legacy_threshold_interval_status": row[
                            "legacy_threshold_interval_status"
                        ],
                        "sign_status": row["sign_status"],
                        "status_disagreement": row["status_disagreement"],
                        "refused": row["refused"],
                        "refusal_reason": row["refusal_reason"],
                    }
                    for row in sorted(measured, key=lambda row: row["block_id"])
                ],
                "alpha_sensitivity_range_nats": (None if not ranges else float(max(ranges))),
                "leakage_removed": [
                    {
                        "block_id": record["block_id"],
                        "available": record["available"],
                        "information_nats": record["information_nats"],
                        "delta_vs_headline_nats": record["delta_vs_headline_nats"],
                        "reason": record["reason"],
                    }
                    for record in sorted(
                        leakage_by_arm.get(arm, []), key=lambda record: record["block_id"]
                    )
                ],
                "sign_status_is_evidential": False,
            }
        )

    measured_rows = [row for row in rows if row["context_information_nats"] is not None]
    straddling = sorted(
        {
            row["arm"]
            for row in measured_rows
            if row["legacy_threshold_interval_status"] == "straddles"
        }
    )
    disagreeing = sorted({row["arm"] for row in measured_rows if row["status_disagreement"]})
    refused = sorted({row["arm"] for row in rows if row["refused"]})
    absent = sorted(entry["arm"] for entry in entries if not entry["present"])
    return {
        "threshold_nats": float(threshold),
        "operative_gate": (
            "point estimate of I against the 0.30 nats/token floor "
            "(budget.MIN_CONTEXT_INFORMATION_NATS), now reported with a "
            "group-level paired interval"
        ),
        "arms": entries,
        "arms_with_no_record": absent,
        "absence_note": (
            "an arm listed here has no record in any supplied sidecar. It has not "
            "been measured on this estimand and its absence is not a pass"
        ),
        "arms_with_a_refused_interval": refused,
        "arms_whose_interval_straddles_the_threshold": straddling,
        "arms_where_the_sign_rule_disagrees_with_the_floor": disagreeing,
        "conclusion": {
            "interval_alters_the_unmeasurable_set": bool(straddling),
            "interval_statement": (
                "no arm's interval straddles the 0.30 nats/token floor, so "
                "reporting the floor with an interval does not move any arm across "
                "it on these draws"
                if not straddling
                else "the interval crosses the 0.30 nats/token floor for "
                + ", ".join(straddling)
                + ", so for those arms the point comparison and the interval do not "
                "agree about measurability and the point verdict is not supported "
                "by this draw alone"
            ),
            "sign_rule_alters_the_unmeasurable_set": bool(disagreeing),
            "sign_rule_statement": (
                "the sign criterion agrees with the floor on every measured arm "
                "here, which is what the pre-registration expects and is not "
                "evidence for either"
                if not disagreeing
                else "the sign criterion and the floor disagree on "
                + ", ".join(disagreeing)
                + "; the floor remains the operative gate and the sign criterion is "
                "NOT adopted, so the existing conclusion about which arms are "
                "unmeasurable is unchanged"
            ),
            "sign_rule_is_adopted": False,
            "note": SCREENING_NOTE,
        },
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sidecar",
        type=Path,
        nargs="+",
        required=True,
        help="power_<cohort>_<digest>.records.npz files, one per cohort draw",
    )
    parser.add_argument(
        "--cohort-json",
        type=Path,
        nargs="*",
        default=[],
        help="frozen-cohort files, aligned with --sidecar; derived from each "
        "sidecar's name when omitted. They carry the records the near-duplicate "
        "grouping is computed from, and without them the resampling unit falls "
        "back to the record with that limitation declared in every affected row",
    )
    parser.add_argument(
        "--reference-json",
        type=Path,
        nargs="*",
        default=[],
        help="held-out reference records, aligned with --sidecar: the "
        "reference_<name>_<digest>.json 01_cohort_power.py writes beside the "
        "cohort. Both the reference grouping and the leakage-removed "
        "sensitivity are unavailable without it",
    )
    parser.add_argument(
        "--arms",
        nargs="*",
        default=None,
        help="restrict to these arms; every arm in every sidecar by default",
    )
    parser.add_argument(
        "--expected-arms",
        nargs="*",
        default=sorted(PANEL),
        help="arms the summary table must account for, so that an arm with no "
        "record is named as absent rather than left out",
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--threshold-nats",
        type=float,
        default=MIN_CONTEXT_INFORMATION_NATS,
        help="the operative screening floor; reported against, never re-chosen here",
    )
    parser.add_argument(
        "--containment",
        type=float,
        default=NEAR_DUPLICATE_CONTAINMENT,
        help="near-duplicate containment threshold for the grouping",
    )
    parser.add_argument(
        "--shingle",
        type=int,
        default=None,
        help="shingle length for the grouping; the unit's declared default when omitted",
    )
    parser.add_argument(
        "--alpha-sweep",
        type=float,
        nargs="*",
        default=list(SMOOTHING_SWEEP),
        help="smoothing ladder for E4; pathways.SMOOTHING_SWEEP by default",
    )
    parser.add_argument(
        "--leakage-kmer",
        type=int,
        default=DEFAULT_LEAKAGE_KMER,
        help="k of the k-mer the E5 reference screen uses",
    )
    parser.add_argument(
        "--contrast",
        action="append",
        nargs=2,
        metavar=("LEFT", "RIGHT"),
        default=None,
        help="an extra cross-cohort pair to difference; repeatable. Pairs within "
        "one cohort are always formed and need not be named",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--report-name", default="context_information_bootstrap.json"
    )
    args = parser.parse_args(argv)
    for name, values in (("--cohort-json", args.cohort_json), ("--reference-json", args.reference_json)):
        if values and len(values) != len(args.sidecar):
            raise SystemExit(
                f"{name} takes one path per --sidecar ({len(args.sidecar)}); "
                f"got {len(values)}"
            )
    if not args.alpha_sweep or any(value <= 0.0 for value in args.alpha_sweep):
        raise SystemExit("--alpha-sweep must name at least one positive constant")
    if args.leakage_kmer < 1:
        raise SystemExit("--leakage-kmer must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    started = datetime.now(timezone.utc).isoformat()

    blocks: list[Block] = []
    panels: list[Panel] = []
    seed_cursor = int(args.seed)
    for index, sidecar in enumerate(args.sidecar):
        block = load_block(
            index,
            Path(sidecar),
            Path(args.cohort_json[index]) if args.cohort_json else None,
            Path(args.reference_json[index]) if args.reference_json else None,
            requested_arms=list(args.arms) if args.arms else None,
            containment=float(args.containment),
            shingle=args.shingle,
        )
        blocks.append(block)
        block_panels = bootstrap_block(
            block,
            seed_cursor,
            n_bootstrap=int(args.n_bootstrap),
            confidence=float(args.confidence),
        )
        seed_cursor += max(len(block_panels), 1)
        panels.extend(block_panels)

    panel_of: dict[tuple[str, str], Panel] = {
        (panel.block.block_id, name): panel for panel in panels for name in panel.names
    }
    rows: list[dict[str, Any]] = []
    for block in blocks:
        for entry in block.arms:
            if entry.statistics is None:
                rows.append(
                    arm_row(
                        block,
                        None,
                        load_refusal_record(entry.name, entry.load_refusal or "unstated"),
                        threshold=float(args.threshold_nats),
                    )
                )
                continue
            panel = panel_of[(block.block_id, entry.name)]
            rows.append(
                arm_row(
                    block,
                    panel.panel_id,
                    panel.result.arms[entry.name].record,
                    threshold=float(args.threshold_nats),
                )
            )

    requested_pairs = [(left, right) for left, right in (args.contrast or [])]
    within = within_cohort_contrasts(panels)
    across = cross_cohort_contrasts(
        panels, confidence=float(args.confidence), requested=requested_pairs
    )
    alpha_records = alpha_sensitivity(
        panels,
        sweep=[float(value) for value in args.alpha_sweep],
        n_bootstrap=int(args.n_bootstrap),
        confidence=float(args.confidence),
    )
    leakage_records = leakage_sensitivity(
        panels,
        k=int(args.leakage_kmer),
        n_bootstrap=int(args.n_bootstrap),
        confidence=float(args.confidence),
    )
    payload = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "started_at_utc": started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "runner_sha256": sha256_file(Path(__file__)),
            "module_sha256": {
                name: sha256_file(REPO_ROOT / "src" / "transfer" / name)
                for name in HASHED_MODULES
            },
            "preregistration": "docs/CONTEXT_INFORMATION_UNCERTAINTY_PREREGISTRATION.md",
            "estimand": ESTIMAND,
            "screening_note": SCREENING_NOTE,
            "comparability_note": COMPARABILITY_NOTE,
            "sign_status_note": SIGN_STATUS_NOTE,
            "minimum_bootstrap_units": int(MINIMUM_BOOTSTRAP_UNITS),
            "interval_field_note": (
                "bootstrap_ci_95 carries the nominal 95% default this analysis is "
                "specified at; when --confidence is moved the interval is at the "
                "level the confidence field beside it states, and that field is "
                "the authoritative one"
            ),
            "seeds": {
                "base": int(args.seed),
                "panels": {panel.panel_id: panel.seed for panel in panels},
                "leakage_offset": LEAKAGE_SEED_OFFSET,
                "producer_seeds": {
                    block.block_id: block.producer_seeds for block in blocks
                },
            },
            "configuration": {
                key: (
                    [str(item) for item in value]
                    if isinstance(value, list) and value and isinstance(value[0], Path)
                    else str(value)
                    if isinstance(value, Path)
                    else value
                )
                for key, value in sorted(vars(args).items())
            },
        },
        "blocks": [
            {
                "block_id": block.block_id,
                "sidecar": str(block.sidecar),
                "sidecar_sha256": block.sidecar_sha256,
                "cohort_digest": block.cohort_digest,
                "cohort_name": block.cohort_name,
                "cohort_kind": block.cohort_kind,
                "reference_digest": block.reference_digest,
                "smoothing": block.smoothing if math.isfinite(block.smoothing) else None,
                "max_len": block.max_len,
                "producer_seeds": block.producer_seeds,
                "arms_in_sidecar": [entry.name for entry in block.arms],
                "arms_refused_at_load": {
                    entry.name: entry.load_refusal
                    for entry in block.arms
                    if entry.load_refusal is not None
                },
                "reference_records_dropped_empty": {
                    entry.name: entry.reference_records_dropped_empty
                    for entry in block.arms
                },
                "cohort_grouping": block.grouping,
                "reference_grouping": block.reference_grouping,
                "panels": [
                    {
                        "panel_id": panel.panel_id,
                        "arms": panel.names,
                        "seed": panel.seed,
                        "common_resample_indices": True,
                    }
                    for panel in panels
                    if panel.block.block_id == block.block_id
                ],
            }
            for block in blocks
        ],
        "arm_results": rows,
        "contrasts": {
            "within_cohort_paired": within,
            "cross_cohort_unpaired": across,
            "same_cohort_unpairable": cross_panel_refusals(panels),
            "declared": declared_contrast_sign_tracking(within, across),
            "note": (
                "pairing is defined within a cohort and nowhere else; a contrast "
                "that crosses cohorts is reported as unpaired with that fact "
                "attached, and non-overlap of two independently bootstrapped "
                "intervals is not a test of difference"
            ),
        },
        "between_block_variance": between_block_variance(
            rows, {block.block_id: block for block in blocks}
        ),
        "alpha_sensitivity": {
            "sweep": [float(value) for value in args.alpha_sweep],
            "records": alpha_records,
            "note": (
                "E4. Smoothing is applied at analysis time from the persisted "
                "reference counts, so this sweep is CPU arithmetic over the same "
                "statistics the headline is computed from. It replaces the "
                "withdrawn figures in pathways.LAPLACE_SMOOTHING with measurement "
                "on the corpora actually used"
            ),
        },
        "leakage_removed_sensitivity": {
            "kmer": int(args.leakage_kmer),
            "records": leakage_records,
            "note": (
                "E5. A sensitivity beside the headline, not a correction to it; "
                "the headline stays on the declared reference"
            ),
        },
        "summary": summarise(
            rows,
            expected_arms=list(args.expected_arms),
            alpha_records=alpha_records,
            leakage_records=leakage_records,
            threshold=float(args.threshold_nats),
        ),
    }

    destination = Path(args.out) / args.report_name
    write_json(destination, payload)

    print(
        f"{'arm':18s} {'block':6s} {'I nats/token':>13s} "
        f"{'95% interval':>22s}  {'floor':6s} {'interval':10s} sign*"
    )
    for row in rows:
        if row["context_information_nats"] is None:
            print(
                f"{row['arm']:18s} {row['block_id']:6s} "
                f"{'REFUSED':>13s} {'-':>22s}  {'-':6s} {'-':10s} -"
            )
            continue
        low, high = row["bootstrap_ci_95"]
        print(
            f"{row['arm']:18s} {row['block_id']:6s} "
            f"{row['context_information_nats']:+13.4f} "
            f"[{low:+9.4f}, {high:+9.4f}]  "
            f"{row['legacy_threshold_status']:6s} "
            f"{row['legacy_threshold_interval_status']:10s} "
            f"{row['sign_status']}"
        )
    print("* sign_status is NON-EVIDENTIAL and is expected to pass on every arm")
    absent = payload["summary"]["arms_with_no_record"]
    if absent:
        print(f"arms with no record at all (absence is not a pass): {', '.join(absent)}")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
