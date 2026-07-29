#!/usr/bin/env python3
"""Path-patch the induction heads: how much of their logit effect is written directly?

Three independent measurements on this panel point the same way -- that
attention's contribution to next-token prediction is more *indirect* in protein
decoders than in text, mediated through later MLPs rather than written straight
to the unembedding.  Whole-pathway ablation credits attention more than direct
logit attribution does, and the gap is about three times larger in residue-level
protein arms (+0.203 ProtGPT2, +0.233 ProGen2-medium) than in text (+0.079
gpt2-large).  ProtGPT2's prefix-matching heads score at or below chance on the OV
copying statistic while GPT-2's copy directly.  Activation patching recovers
about 1.0 through ``resid_post`` but only 0.1-0.6 through ``attn_out`` or
``mlp_out`` alone.  None of the three separates a direct write from a mediated
one, because none of them holds the rest of the model fixed while it measures.

This script runs the instrument that does: path patching (Wang et al., ICLR
2023; Goldowsky-Dill et al., arXiv:2304.05969) from each induction head, along
the head-to-unembedding path and along the head-to-later-component-to-unembedding
paths, with every component off the tested path pinned to its corrupted-run
value.  The construction, the metric and the freezing regime are stated in
:mod:`src.transfer.path_patching` and were fixed before any number was read.

Three design constraints are load-bearing.

1.  **Per-sender-head normalisation is primary.**  Head counts above threshold
    differ across the panel by an order of magnitude -- about fifty for
    gpt2-large, fourteen for ProtGPT2, six for ProGen2-medium, zero for ZymCTRL
    -- and the head count is itself the disputed quantity.  A recovered logit
    difference summed over senders would confound per-head mediation strength
    with how many heads an arm has, so the per-head mean is the headline and the
    sum is reported beside it under an explicit label.

2.  **ZymCTRL has no head above threshold and cannot enter as an induction arm.**
    It enters on its top-k heads with ``above_threshold`` false on every one and
    ``not_induction_heads_by_panel_criterion`` set on its sender record.  The
    threshold is not lowered for it, and its numbers are not induction-head
    numbers.

3.  **Both sender sets run.**  The census exists under an exact and an
    approximate repeat criterion.  The same heads top both for ProtGPT2, so the
    result should be stable -- but that is demonstrated here, on the full
    sender-set-by-case-set cross, rather than assumed.

The controlled comparison is **gpt2-large versus ProtGPT2**: identical depth,
width, vocabulary and tokenisation.  ProGen2-medium and ZymCTRL differ in
tokenisation, and substitution-tolerant alignment costs the BPE arms most of
their probe coverage (gpt2-large 0.943 -> 0.414, ProtGPT2 0.710 -> 0.310) while
residue-level arms lose almost nothing (0.965 -> 0.956), so a cross-tokenisation
contrast carries a coverage confound the matched pair does not.  Every artefact
labels which comparisons are controlled and which are directional.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.io import write_json  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    DEFAULT_CORPUS_DRAW_SEED,
    MATCHED_PAIR,
    PANEL,
    Arm,
    Cohort,
    load_arm,
    protein_cohort,
    symbols_per_token,
    text_cohort,
)
from src.transfer.circuits import (  # noqa: E402
    INDUCTION_THRESHOLDS,
    PROTEIN_APPROXIMATE_CRITERION,
    PROTEIN_EXACT_CRITERION,
    TEXT_APPROXIMATE_CRITERION,
    TEXT_EXACT_CRITERION,
    RepeatCriterion,
    attention_alignment_scores,
    fit_unigram,
    head_census,
    induction_headline,
    natural_repeat_probes,
    protein_repeat_cohort,
    text_repeat_cohort,
)
from src.transfer.circuits import SCHEMA_VERSION as CENSUS_SCHEMA_VERSION  # noqa: E402
from src.transfer.path_patching import (  # noqa: E402
    DEFAULT_MIN_HEAD_EFFECT,
    DEFAULT_MINIMUM_EFFECT,
    SCHEMA_VERSION,
    PathPatcher,
    bootstrap_difference,
    build_path_cases,
    select_senders,
    sender_effects,
    sender_set_overlap,
    structural_invariants,
    summarise_senders,
)

DEFAULT_OUTPUT = REPO_ROOT / "results/transfer/path_patching"

#: The two repeat criteria, in the order they are reported.  ``exact`` is first
#: because it is the baseline the approximate probe has to be read against.
CRITERIA = ("exact", "approximate")

#: Arms whose tokenisation matches the text reference.  Only these two support a
#: controlled contrast; everything else is directional.
CONTROLLED_PAIR = MATCHED_PAIR


def verify_outputs(directory: Path, names: Sequence[str]) -> None:
    """Fail loudly if an expected artifact is missing or is not this schema."""

    broken: list[str] = []
    for path in [directory / f"{name}.json" for name in names] + [
        directory / "panel_summary.json"
    ]:
        if not path.is_file():
            broken.append(f"{path}: missing")
            continue
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != SCHEMA_VERSION:
            broken.append(f"{path}: unexpected schema_version")
    if broken:
        raise RuntimeError("output verification failed: " + "; ".join(broken))


def cohort_record(cohort: Cohort) -> dict[str, Any]:
    """Cohort provenance, including the repeat criterion that produced it."""

    record: dict[str, Any] = {
        "name": cohort.name,
        "kind": cohort.kind,
        "digest": cohort.digest,
        "n_records": len(cohort),
        "min_symbols": cohort.min_symbols,
        "max_symbols": cohort.max_symbols,
        "source": cohort.metadata.get("source"),
    }
    for key in (
        "criterion",
        "census",
        "cohort_identity_fraction_mean",
        "cohort_repeat_length_mean",
        "cohort_mean_blosum62_substituted",
    ):
        if key in cohort.metadata:
            record[key] = cohort.metadata[key]
    return record


def arm_record(arm: Arm, attn_implementation: str) -> dict[str, Any]:
    spec = arm.spec
    return {
        "name": spec.name,
        "path": str(spec.path),
        "modality": spec.modality,
        "n_layer": spec.n_layer,
        "d_model": spec.d_model,
        "tokenisation": spec.tokenisation,
        "input_format": spec.input_format,
        "source": spec.source,
        "architecture": spec.architecture,
        "dtype": arm.dtype,
        "device": arm.device,
        "attn_implementation": attn_implementation,
        "matched_pair_member": spec.name in CONTROLLED_PAIR,
        "comparison_status": (
            "controlled_matched_pair"
            if spec.name in CONTROLLED_PAIR
            else "directional_cross_tokenisation_only"
        ),
    }


def repeat_criteria(modality: str, args: argparse.Namespace) -> dict[str, RepeatCriterion]:
    """The exact and approximate criteria for one modality, with the unit from the CLI."""

    declared = {
        "text": (TEXT_EXACT_CRITERION, TEXT_APPROXIMATE_CRITERION, args.text_repeat_unit),
        "protein": (
            PROTEIN_EXACT_CRITERION,
            PROTEIN_APPROXIMATE_CRITERION,
            args.protein_repeat_unit,
        ),
    }
    if modality not in declared:
        raise ValueError(f"unsupported modality {modality!r}")
    exact, approximate, min_unit = declared[modality]
    return {
        "exact": replace(exact, min_unit=min_unit),
        "approximate": replace(approximate, min_unit=min_unit),
    }


def build_cohorts(modality: str, args: argparse.Namespace) -> dict[str, Cohort]:
    """One analysis cohort plus one repeat cohort per criterion, shared across arms."""

    criteria = repeat_criteria(modality, args)
    cohorts: dict[str, Cohort] = {}
    draw_seed = args.cohort_draw_seed or None
    if modality == "text":
        cohorts["analysis"] = text_cohort(
            args.cohort_size, min_chars=args.text_min_chars, seed=draw_seed
        )
        for label, criterion in criteria.items():
            cohorts[f"repeat_{label}"] = text_repeat_cohort(
                args.repeat_cohort_size,
                max_chars=args.text_repeat_chars,
                criterion=criterion,
                scan_documents=args.text_repeat_scan,
                workers=args.repeat_scan_workers,
                name=f"openwebtext_repeat_{label}",
                seed=draw_seed,
            )
        return cohorts
    if modality == "protein":
        cohorts["analysis"] = protein_cohort(
            args.cohort_size,
            args.protein_min_len,
            args.protein_max_len,
            with_ec=True,
            name="swissprot_ec_long",
            seed=draw_seed,
        )
        for label, criterion in criteria.items():
            cohorts[f"repeat_{label}"] = protein_repeat_cohort(
                args.repeat_cohort_size,
                min_len=args.repeat_min_len,
                max_len=args.repeat_max_len,
                criterion=criterion,
                workers=args.repeat_scan_workers,
                name=f"swissprot_repeat_{label}",
                seed=draw_seed,
            )
        return cohorts
    raise ValueError(f"unsupported modality {modality!r}")


def run_census(
    arm: Arm, cohort: Cohort, args: argparse.Namespace
) -> tuple[np.ndarray, dict[str, Any], list]:
    """The prefix-matching census this arm's sender set is drawn from."""

    probes = natural_repeat_probes(arm, cohort, max_tokens=args.natural_max_tokens)
    alignment = attention_alignment_scores(arm, probes, batch_size=args.probe_batch_size)
    prefix_matching = alignment["scores"]["prefix_matching"]
    census = head_census(prefix_matching)
    record = {
        "cohort": cohort_record(cohort),
        "probe_kind": alignment["kind"],
        "n_probes": alignment["n_probes"],
        "scored_query_positions": alignment["scored_query_positions"],
        "mean_coverage": alignment["mean_coverage"],
        "uniform_baseline": alignment["uniform_baseline"],
        "census": census,
        "headline": induction_headline(alignment, census, threshold=args.headline_threshold),
        "census_schema_version": CENSUS_SCHEMA_VERSION,
    }
    return prefix_matching, record, probes


def run_condition(
    patcher: PathPatcher,
    senders: Sequence[Any],
    args: argparse.Namespace,
    *,
    resolve: bool,
) -> dict[str, Any]:
    """Every sender's four pathways over one case set."""

    per_head: list[dict[str, Any]] = []
    for index, sender in enumerate(senders):
        recoveries = patcher.sender_recoveries(sender)
        entry: dict[str, Any] = {
            **sender.as_dict(),
            "label": sender.label,
            "recovery": recoveries,
            "effects": sender_effects(recoveries),
        }
        if resolve and index < args.resolve_senders:
            entry["receiver_profile"] = patcher.sender_receiver_profile(sender)
        per_head.append(entry)
    return {
        "per_sender_head": per_head,
        "summary": summarise_senders(per_head, min_head_effect=args.min_head_effect),
    }


def run_arm(
    name: str, args: argparse.Namespace, cohorts: dict[str, dict[str, Cohort]]
) -> dict[str, Any]:
    started = time.time()
    arm = load_arm(
        name, device=args.device, dtype=args.dtype, attn_implementation=args.attn_implementation
    )
    arm.require("circuits")
    torch.cuda.reset_peak_memory_stats(arm.device)
    modality_cohorts = cohorts[arm.modality]
    analysis = modality_cohorts["analysis"]
    strings = analysis.input_strings(arm)
    unigram = fit_unigram(arm, strings, max_tokens=args.unigram_max_tokens)

    censuses: dict[str, dict[str, Any]] = {}
    sender_sets: dict[str, list] = {}
    sender_records: dict[str, dict[str, Any]] = {}
    case_sets: dict[str, list] = {}
    case_records: dict[str, dict[str, Any]] = {}
    for criterion in CRITERIA:
        cohort = modality_cohorts[f"repeat_{criterion}"]
        prefix_matching, record, probes = run_census(arm, cohort, args)
        censuses[criterion] = record
        senders, provenance = select_senders(
            prefix_matching,
            threshold=args.headline_threshold,
            fallback_top_k=args.fallback_top_k,
            max_senders=args.max_senders,
        )
        sender_sets[criterion] = senders
        sender_records[criterion] = {**provenance, "repeat_criterion": criterion}
        cases, case_provenance = build_path_cases(
            arm,
            probes,
            unigram,
            n_cases=args.n_cases,
            cases_per_probe=args.cases_per_probe,
            max_tokens=args.max_case_tokens,
            seed=args.seed + CRITERIA.index(criterion),
        )
        case_sets[criterion] = cases
        case_records[criterion] = {**case_provenance, "repeat_criterion": criterion}
        print(
            f"[{name}/{criterion}] {provenance['n_senders']} senders "
            f"({provenance['criterion']}), {len(cases)} cases",
            flush=True,
        )

    patchers = {
        criterion: PathPatcher(
            arm,
            case_sets[criterion],
            batch_size=args.case_batch_size,
            minimum_effect=args.minimum_effect,
        )
        for criterion in CRITERIA
    }

    invariants = structural_invariants(
        patchers[CRITERIA[0]],
        sender_sets[CRITERIA[0]][0],
        tolerance=args.invariant_tolerance,
        linearity_tolerance=args.linearity_tolerance,
    )
    print(f"[{name}] structural invariants passed", flush=True)

    conditions: dict[str, Any] = {}
    for sender_criterion in CRITERIA:
        for case_criterion in CRITERIA:
            key = f"senders_{sender_criterion}__cases_{case_criterion}"
            conditions[key] = {
                "sender_criterion": sender_criterion,
                "case_criterion": case_criterion,
                "primary": sender_criterion == case_criterion,
                "eligibility": patchers[case_criterion].eligibility(),
                **run_condition(
                    patchers[case_criterion],
                    sender_sets[sender_criterion],
                    args,
                    resolve=args.resolve_receivers
                    and sender_criterion == case_criterion == CRITERIA[0],
                ),
            }
            summary = conditions[key]["summary"]
            print(
                f"[{name}] {key}: direct {summary['per_sender_head_mean']['direct']:+.4f} "
                f"mediated {summary['per_sender_head_mean']['mediated']:+.4f} "
                f"total {summary['per_sender_head_mean']['total']:+.4f}",
                flush=True,
            )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "arm": arm_record(arm, args.attn_implementation),
        "cohorts": {label: cohort_record(cohort) for label, cohort in modality_cohorts.items()},
        "repeat_criteria": {
            label: criterion.as_dict()
            for label, criterion in repeat_criteria(arm.modality, args).items()
        },
        "induction_census": censuses,
        "sender_sets": sender_records,
        "sender_set_stability": sender_set_overlap(
            sender_sets["exact"], sender_sets["approximate"]
        ),
        "case_sets": case_records,
        "structural_invariants": invariants,
        "conditions": conditions,
        "seeds": {"master": args.seed, "bootstrap": args.seed + 100},
        "thresholds": {
            "induction_prefix_matching": list(INDUCTION_THRESHOLDS),
            "sender_threshold": args.headline_threshold,
            "minimum_effect_logits": args.minimum_effect,
            "min_head_effect_recovery": args.min_head_effect,
            "invariant_tolerance": args.invariant_tolerance,
            "linearity_tolerance": args.linearity_tolerance,
            "induction_data_driven_sigma": 3.0,
        },
        "tokenisation": {
            "symbols_per_token": symbols_per_token(
                arm, strings[: args.cohort_size], args.unigram_max_tokens
            ),
            "unigram": unigram.summary(),
        },
        "runtime_seconds": round(time.time() - started, 2),
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(arm.device)),
    }
    del patchers, arm
    torch.cuda.empty_cache()
    return payload


def panel_summary(results: dict[str, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """One row per arm, plus the matched-pair contrast that is the actual question."""

    rows: dict[str, Any] = {}
    for name, payload in results.items():
        row: dict[str, Any] = {
            "modality": payload["arm"]["modality"],
            "tokenisation": payload["arm"]["tokenisation"],
            "comparison_status": payload["arm"]["comparison_status"],
            "sender_set_stability": payload["sender_set_stability"],
            "structural_invariants_passed": payload["structural_invariants"]["passed"],
        }
        for criterion in CRITERIA:
            key = f"senders_{criterion}__cases_{criterion}"
            condition = payload["conditions"][key]
            row[criterion] = {
                "n_senders": condition["summary"]["n_senders"],
                "sender_criterion": payload["sender_sets"][criterion]["criterion"],
                "not_induction_heads_by_panel_criterion": payload["sender_sets"][criterion][
                    "not_induction_heads_by_panel_criterion"
                ],
                "eligible_fraction": condition["eligibility"]["eligible_fraction"],
                "mean_denominator": condition["eligibility"]["mean_denominator"],
                "per_sender_head_mean": condition["summary"]["per_sender_head_mean"],
                "per_sender_head_fraction_mean": condition["summary"][
                    "per_sender_head_fraction_mean"
                ],
                "aggregate_sum_over_senders": condition["summary"]["aggregate_sum_over_senders"],
                "aggregate_effect_weighted_fraction": condition["summary"][
                    "aggregate_effect_weighted_fraction"
                ],
                "fraction_n_heads_used": condition["summary"]["fraction_n_heads_used"],
            }
        rows[name] = row
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "controlled_pair": list(CONTROLLED_PAIR),
        "comparison_policy": (
            "gpt2-large versus protgpt2 is the controlled comparison: identical depth, "
            "width, vocabulary and tokenisation. Contrasts involving progen2-medium or "
            "zymctrl cross a tokenisation boundary and additionally cross a large "
            "probe-coverage gap under the approximate criterion, so they are directional "
            "only and are labelled as such on every arm record."
        ),
        "arms": rows,
        "matched_pair_contrast": matched_pair_contrast(results, args),
        "directional_contrasts": directional_contrasts(results, args),
        "configuration": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in sorted(vars(args).items())
        },
    }


#: Quantities contrasted between arms.  The fraction answers the question the
#: programme asked; the effect is carried alongside because a fraction is a ratio
#: of two small recoveries and can move for reasons the numerator did not.
CONTRAST_QUANTITIES = ("mediated_fraction", "mediated_effect", "direct_effect")


def _head_values(payload: dict[str, Any], criterion: str, quantity: str) -> list[float]:
    """One value per sender head, on the scale ``quantity`` names.

    Fractions are withheld for heads whose total effect does not clear the floor,
    because ``mediated / total`` is not a decomposition of an effect that cannot
    be distinguished from zero.  Effects are not withheld: a head with a small
    total is still a head with a small total, and dropping it from the effect
    scale would be selection on the outcome.
    """

    condition = payload["conditions"][f"senders_{criterion}__cases_{criterion}"]
    floor = float(payload["thresholds"]["min_head_effect_recovery"])
    values: list[float] = []
    for head in condition["per_sender_head"]:
        effects = head["effects"]
        if quantity == "mediated_fraction":
            if abs(float(effects["total"])) < floor:
                continue
            values.append(float(effects["mediated"]) / float(effects["total"]))
        elif quantity == "mediated_effect":
            values.append(float(effects["mediated"]))
        elif quantity == "direct_effect":
            values.append(float(effects["direct"]))
        else:
            raise ValueError(f"unknown contrast quantity {quantity!r}")
    return values


def _contrast(
    results: dict[str, dict[str, Any]], reference: str, other: str, args: argparse.Namespace
) -> dict[str, Any]:
    entry: dict[str, Any] = {"reference_arm": reference, "arm": other}
    for criterion in CRITERIA:
        per_criterion: dict[str, Any] = {}
        for quantity in CONTRAST_QUANTITIES:
            left = _head_values(results[other], criterion, quantity)
            right = _head_values(results[reference], criterion, quantity)
            if len(left) < 2 or len(right) < 2:
                per_criterion[quantity] = {
                    "n_left": len(left),
                    "n_right": len(right),
                    "note": "too few sender heads to bootstrap a difference",
                }
                continue
            per_criterion[quantity] = {
                "arm": float(np.mean(left)),
                "reference": float(np.mean(right)),
                "bootstrap": bootstrap_difference(
                    left, right, resamples=args.bootstrap_resamples, seed=args.seed + 100
                ),
            }
        entry[criterion] = per_criterion
    return entry


def matched_pair_contrast(
    results: dict[str, dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    """ProtGPT2 against gpt2-large: the only contrast on this panel that is controlled."""

    reference, other = CONTROLLED_PAIR
    if reference not in results or other not in results:
        return {
            "available": False,
            "note": f"the controlled pair {CONTROLLED_PAIR} was not both in this run",
        }
    return {
        "available": True,
        "controlled": True,
        "hypothesis": (
            "the mediated fraction of an induction head's total logit effect is larger "
            "in the protein arm than in the matched text control"
        ),
        **_contrast(results, reference, other, args),
    }


def directional_contrasts(
    results: dict[str, dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    """The cross-tokenisation contrasts, which are suggestive and nothing more."""

    reference = CONTROLLED_PAIR[0]
    if reference not in results:
        return {"available": False, "note": f"{reference} was not in this run"}
    return {
        "available": True,
        "controlled": False,
        "note": (
            "cross-tokenisation: these arms differ from the reference in tokenisation and "
            "in probe coverage under the approximate criterion, so a difference here is "
            "not attributable to modality"
        ),
        "arms": {
            name: _contrast(results, reference, name, args)
            for name in results
            if name not in CONTROLLED_PAIR
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(CONTROLLED_PAIR), choices=sorted(PANEL))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260728)

    parser.add_argument("--cohort-size", type=int, default=24)
    parser.add_argument(
        "--cohort-draw-seed",
        type=int,
        default=DEFAULT_CORPUS_DRAW_SEED,
        help="seed for the permutation the analysis cohort is drawn under; 0 "
        "selects the historical file-order prefix, which is a declared choice "
        "and not a default (transfer audit, Appendix B rule 1)",
    )
    parser.add_argument("--protein-min-len", type=int, default=600)
    parser.add_argument("--protein-max-len", type=int, default=1000)
    parser.add_argument("--text-min-chars", type=int, default=3000)
    parser.add_argument("--unigram-max-tokens", type=int, default=256)

    parser.add_argument("--repeat-cohort-size", type=int, default=32)
    parser.add_argument("--repeat-min-len", type=int, default=200)
    parser.add_argument("--repeat-max-len", type=int, default=800)
    parser.add_argument("--protein-repeat-unit", type=int, default=16)
    parser.add_argument("--text-repeat-chars", type=int, default=2000)
    parser.add_argument("--text-repeat-unit", type=int, default=40)
    parser.add_argument("--text-repeat-scan", type=int, default=3000)
    parser.add_argument("--repeat-scan-workers", type=int, default=64)
    parser.add_argument("--natural-max-tokens", type=int, default=840)
    parser.add_argument("--probe-batch-size", type=int, default=2)
    parser.add_argument("--headline-threshold", type=float, default=0.10)

    parser.add_argument(
        "--fallback-top-k",
        type=int,
        default=8,
        help=(
            "heads to admit for an arm with none above threshold; they are flagged "
            "as not induction heads by the panel's own criterion"
        ),
    )
    parser.add_argument("--max-senders", type=int, default=None)
    parser.add_argument("--n-cases", type=int, default=64)
    parser.add_argument("--cases-per-probe", type=int, default=4)
    parser.add_argument("--max-case-tokens", type=int, default=640)
    parser.add_argument("--case-batch-size", type=int, default=8)
    parser.add_argument("--minimum-effect", type=float, default=DEFAULT_MINIMUM_EFFECT)
    parser.add_argument("--min-head-effect", type=float, default=DEFAULT_MIN_HEAD_EFFECT)
    parser.add_argument("--resolve-receivers", action="store_true")
    parser.add_argument("--resolve-senders", type=int, default=4)
    parser.add_argument("--invariant-tolerance", type=float, default=1e-3)
    parser.add_argument("--linearity-tolerance", type=float, default=0.02)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)

    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("duplicate arms requested")
    if f"{args.headline_threshold:.2f}" not in {f"{v:.2f}" for v in INDUCTION_THRESHOLDS}:
        raise ValueError(
            f"sender threshold {args.headline_threshold} is not one of the census "
            f"thresholds {list(INDUCTION_THRESHOLDS)}"
        )
    if args.max_case_tokens > args.natural_max_tokens:
        raise ValueError("the case token cap cannot exceed the probe token cap")
    if args.resolve_senders < 1:
        raise ValueError("resolve_senders must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required")
    torch.manual_seed(args.seed)

    # Cohorts before any checkpoint is loaded: the repeat census forks a process
    # pool, and forking after a CUDA context exists fails as a hang, not an error.
    modalities = {PANEL[name].modality for name in args.arms}
    cohorts = {modality: build_cohorts(modality, args) for modality in sorted(modalities)}
    for modality, built in sorted(cohorts.items()):
        for criterion in CRITERIA:
            census = built[f"repeat_{criterion}"].metadata["census"]
            print(
                f"[cohort] {modality} {criterion}: {census['n_matching']} matching of "
                f"{census['scanned_eligible']} eligible ({census['match_rate']:.5%})",
                flush=True,
            )

    results: dict[str, dict[str, Any]] = {}
    for name in args.arms:
        payload = run_arm(name, args, cohorts)
        write_json(args.output_dir / f"{name}.json", payload)
        results[name] = payload
        print(
            f"[{name}] done in {payload['runtime_seconds']}s, "
            f"peak {payload['peak_gpu_bytes'] / 2**30:.1f} GiB",
            flush=True,
        )

    for name, payload in results.items():
        write_json(args.output_dir / f"{name}.json", payload)
    write_json(args.output_dir / "panel_summary.json", panel_summary(results, args))
    verify_outputs(args.output_dir, sorted(results))
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
