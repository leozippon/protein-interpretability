#!/usr/bin/env python3
"""Measure Anthropic's pre-dictionary circuit toolkit on the matched decoder panel.

Three measurements run per arm, each testing an assumption the toolkit inherits
from text:

1. an induction / copying head census, on repeated-token probes built from the
   arm's own unigram distribution and, separately, on real sequences that contain
   a genuine internal repeat -- once under an exact-repeat criterion and once
   under a substitution-tolerant one;
2. direct logit attribution of the correct-next-token logit onto the embedding
   and every attention and MLP sublayer;
3. an activation-patching map over (component kind, layer, patched position) with
   a sweep of the perturbation-to-read-out distance.

The two natural-repeat probes are both reported and neither is preferred.  The
exact probe is the one the panel's headline was measured on; the approximate
probe exists because Pomerants et al., arXiv:2602.23179 v5, show that on protein
language models approximate-repeat detection subsumes exact-repeat detection, so
an exact probe measures a special case -- and a special case that BPE text
supplies far more readily than protein sequence does.  Whether the head-count
deficit between the two modalities survives the change of probe is the
measurement; the difference between the two columns is the evidence, so a run
that produced only one of them would answer nothing.

The output is one JSON per arm plus a panel summary.  Runs are validation-scale
by default; nothing here is a production sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The stage directory itself, so `panel_contract` imports under every invocation
# regardless of how the entry point was addressed -- the same shape stages 10 and
# 11 use.
STAGE_DIR = Path(__file__).resolve().parent
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

from panel_contract import stage_contract_record  # noqa: E402
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
    tokenize_batch,
)
from src.transfer.circuits import (  # noqa: E402
    DEFAULT_CORRUPTION_SPAN_TOKENS,
    DISTANCE_BANDS,
    DISTANCE_UNITS,
    INDUCTION_THRESHOLDS,
    PATCHING_REFUSED_DTYPES,
    PROTEIN_APPROXIMATE_CRITERION,
    PROTEIN_EXACT_CRITERION,
    SCHEMA_VERSION,
    SYMBOL_DISTANCE_BANDS,
    TEXT_APPROXIMATE_CRITERION,
    TEXT_EXACT_CRITERION,
    RepeatCriterion,
    Unigram,
    activation_patching,
    attention_alignment_scores,
    build_patch_cases,
    conditioned_token_budget,
    content_symbol_name,
    direct_logit_attribution,
    fit_unigram,
    head_census,
    induction_headline,
    matched_copying_scores,
    n_head,
    natural_repeat_probes,
    ov_copying_scores,
    patch_seq_len_refusal,
    probe_record_retention,
    protein_repeat_cohort,
    resolve_distance_bands,
    summarise_head_matrix,
    summarise_patching,
    synthetic_repeat_probes,
    text_repeat_cohort,
    top_heads,
    verify_head_decomposition,
)

DEFAULT_OUTPUT = REPO_ROOT / "results/transfer/circuit_primitives"
SECTIONS = ("induction", "attribution", "patching")

#: The two natural-repeat probes, in the order they are reported.  ``exact`` is
#: first because it is the baseline the approximate probe has to be read against.
REPEAT_PROBES = ("exact", "approximate")


def verify_outputs(directory: Path, names: Sequence[str]) -> None:
    """Fail loudly if an expected artifact is missing or is not this schema.

    The results root is shared with other measurement tracks that recreate it,
    so a run that reports success without its artifacts on disk would be a false
    success.
    """

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
    """Cohort provenance, including the criterion and census when it has one.

    A repeat cohort's criterion travels with every artefact it appears in.  Two
    induction censuses differing only in whether their repeats were exact are not
    comparable without it, and a digest alone does not say which is which.

    ``sampling`` travels for the same reason one step earlier.  The analysis
    cohort's draw moved from a file-order prefix to a seeded permutation of the
    whole corpus (EXP-R2-068), and until then this record said only that a cohort
    existed and what it hashed to -- not which of the two produced it, which is
    the single most expensive thing this programme has got wrong.
    ``Cohort.sampling`` answers ``"unrecorded"`` with its own hazard text rather
    than guessing, so a hand-built cohort is visibly not a declared draw.
    """

    record: dict[str, Any] = {
        "name": cohort.name,
        "kind": cohort.kind,
        "digest": cohort.digest,
        "provenance_digest": cohort.provenance_digest,
        "n_records": len(cohort),
        "min_symbols": cohort.min_symbols,
        "max_symbols": cohort.max_symbols,
        "source": cohort.metadata.get("source"),
        "sampling": cohort.sampling,
    }
    for key in (
        "criterion",
        "census",
        "cohort_identity_fraction_mean",
        "cohort_identity_fraction_min",
        "cohort_repeat_length_mean",
        "cohort_mean_blosum62_substituted",
    ):
        if key in cohort.metadata:
            record[key] = cohort.metadata[key]
    return record


def grant_circuits(arm: Arm) -> bool:
    """Locally declare the ``circuits`` capability for one arm.

    ``arms.py`` withholds ``circuits`` from an arm this module cannot decompose.
    When the module gains that ability the declaration is the panel's to update,
    not this script's, so the override is explicit, opt-in per arm, and recorded
    in the output rather than assumed. It is a statement that the decomposition
    has been verified for this architecture -- which
    :func:`verify_head_decomposition` then checks on every run -- not a way of
    getting past a refusal.
    """

    if arm.supports("circuits"):
        return False
    arm.spec = replace(arm.spec, capabilities=arm.spec.capabilities | {"circuits"})
    return True


def arm_record(arm: Arm, attn_implementation: str, circuits_granted: bool) -> dict[str, Any]:
    spec = arm.spec
    return {
        "circuits_capability_granted_by_runner": circuits_granted,
        "architecture": spec.architecture,
        "name": spec.name,
        "path": str(spec.path),
        "modality": spec.modality,
        "n_layer": spec.n_layer,
        "d_model": spec.d_model,
        "n_head": n_head(arm),
        "tokenisation": spec.tokenisation,
        "input_format": spec.input_format,
        "source": spec.source,
        "dtype": arm.dtype,
        "device": arm.device,
        "attn_implementation": attn_implementation,
        "matched_pair_member": spec.name in MATCHED_PAIR,
    }


def repeat_criteria(modality: str, args: argparse.Namespace) -> dict[str, RepeatCriterion]:
    """The exact and approximate criteria for one modality, with the unit from the CLI.

    Only ``min_unit`` is exposed.  The substitution cap, the two-occurrence,
    no-indel scope and the BLOSUM62 similarity rule are fixed in
    :mod:`src.transfer.circuits` from the prior work's stated scope and are
    deliberately not tunable from a command line: a criterion that can be moved
    per run is a criterion that will be moved until the answer is convenient.
    """

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
    """Analysis cohort plus one repeat cohort per criterion, for one modality.

    Protein cohorts are drawn from the EC-labelled Swiss-Prot source so that a
    single cohort serves ZymCTRL and the unconditional protein arms, keeping the
    digest identical across the protein arms.  Both repeat cohorts are built here
    and shared across every arm of the modality, so the exact and approximate
    columns of the panel differ in the criterion and in nothing else.
    """

    criteria = repeat_criteria(modality, args)
    cohorts: dict[str, Cohort] = {}
    # The repeat cohorts below already draw under a seed (they are censuses over
    # the whole corpus for records meeting a criterion). The analysis cohort did
    # not, and it is the one that fits the unigram every synthetic probe is built
    # from and supplies the patching cases -- so a family-grouped head-of-file
    # block reached the probes themselves.
    draw_seed = args.cohort_draw_seed or None
    # Only the induction section reads the repeat cohorts, and building them is a
    # multi-process scan of the whole corpus for internal repeats. A patching-only
    # run paid for that scan and discarded it, which is most of the wall clock of
    # the run plan item B6 needs.
    want_repeats = "induction" in args.sections
    if modality == "text":
        cohorts["analysis"] = text_cohort(
            args.cohort_size,
            min_chars=args.text_min_chars,
            skip=args.cohort_skip,
            seed=draw_seed,
        )
        for label, criterion in criteria.items() if want_repeats else ():
            cohorts[f"repeat_{label}"] = text_repeat_cohort(
                repeat_cohort_size(label, args),
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
            skip=args.cohort_skip,
            seed=draw_seed,
        )
        for label, criterion in criteria.items() if want_repeats else ():
            cohorts[f"repeat_{label}"] = protein_repeat_cohort(
                repeat_cohort_size(label, args),
                min_len=args.repeat_min_len,
                max_len=args.repeat_max_len,
                criterion=criterion,
                workers=args.repeat_scan_workers,
                name=f"swissprot_repeat_{label}",
                seed=draw_seed,
            )
        return cohorts
    raise ValueError(f"unsupported modality {modality!r}")


def repeat_cohort_size(label: str, args: argparse.Namespace) -> int:
    """Records to draw for one criterion.

    The two criteria may be run at different cohort sizes because their ceilings
    differ by more than an order of magnitude: the exact criterion admits 48
    proteins in the whole EC-labelled corpus, so a run that exercises the
    approximate criterion's achievable cohort cannot also carry an equally large
    exact cohort.  Sizes are therefore separable, and ``probe_comparison``
    records whether a given run was size-matched, because an unmatched pair
    answers "what does the approximate probe measure when it is given the data it
    can have" and a matched pair answers "what does changing only the criterion
    do".  Both are worth having and they are not the same question.
    """

    if label == "exact" or args.approximate_cohort_size is None:
        return args.repeat_cohort_size
    return args.approximate_cohort_size


def run_induction(
    arm: Arm,
    cohorts: dict[str, Cohort],
    unigram: Unigram,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Prefix-matching census plus the OV copying score for every head.

    The natural-repeat census runs once per criterion on the same arm, the same
    heads and the same copying scores, so the exact and approximate columns are a
    paired comparison rather than two runs that happen to share a name.
    """

    analysis = cohorts["analysis"]
    ec_label = None
    if arm.spec.input_format == "ec_conditioned":
        labels = analysis.metadata.get("ec_labels")
        if not labels:
            raise ValueError(f"{arm.name}: analysis cohort carries no EC labels")
        ec_label = labels[0]

    synthetic = synthetic_repeat_probes(
        arm,
        unigram,
        n_probes=args.synthetic_probes,
        copy_len=args.synthetic_copy_len,
        seed=args.seed,
        ec_label=ec_label,
    )
    decomposition_error = verify_head_decomposition(
        arm,
        arm.n_layer // 2,
        torch.tensor([synthetic[0].input_ids], dtype=torch.long),
    )
    synthetic_scores = attention_alignment_scores(
        arm, synthetic, batch_size=args.probe_batch_size
    )

    natural_scores: dict[str, dict[str, Any]] = {}
    natural_retention: dict[str, dict[str, Any]] = {}
    for label in REPEAT_PROBES:
        cohort = cohorts[f"repeat_{label}"]
        probes = natural_repeat_probes(arm, cohort, max_tokens=args.natural_max_tokens)
        # Record-level loss, beside the within-probe loss `mean_coverage` already
        # reports. A record that aligns nowhere contributes no probe, so it is
        # invisible to any mean over probes -- and it is dropped for a reason
        # (identical token boundaries in both copies) that correlates with the
        # criterion under test.
        natural_retention[label] = probe_record_retention(probes, cohort)
        natural_scores[label] = attention_alignment_scores(
            arm, probes, batch_size=args.natural_batch_size
        )

    support = unigram.token_ids[: args.copy_tokens]
    copying = ov_copying_scores(arm, support)
    matched = matched_copying_scores(
        arm,
        unigram.token_ids,
        matched_n=args.copy_matched_n,
        repeats=args.copy_matched_repeats,
        seed=args.seed,
    )

    ranking = head_ranking(
        synthetic_scores["scores"],
        {label: natural_scores[label]["scores"] for label in REPEAT_PROBES},
        copying,
        matched,
    )

    result: dict[str, Any] = {
        "ov_decomposition_relative_error": decomposition_error,
        "synthetic_repeat": {
            **{key: value for key, value in synthetic_scores.items() if key != "scores"},
            "copy_len_tokens": args.synthetic_copy_len,
            "census": head_census(synthetic_scores["scores"]["prefix_matching"]),
            "same_token_distribution": _matrix_summary(
                synthetic_scores["scores"]["same_token"], "same_token"
            ),
            "offset_two_distribution": _matrix_summary(
                synthetic_scores["scores"]["offset_two"], "offset_two"
            ),
        },
        "copying": {
            "support_size": int(support.size),
            "matched_support_size": args.copy_matched_n,
            "matched_repeats": args.copy_matched_repeats,
            "diagonal_fraction": _matrix_summary(
                copying["diagonal_fraction"], "diagonal_fraction"
            ),
            "mean_normalised_rank": _matrix_summary(
                copying["mean_normalised_rank"], "mean_normalised_rank"
            ),
            "matched_diagonal_fraction": _matrix_summary(
                matched["diagonal_fraction"], "matched_diagonal_fraction"
            ),
            "matched_mean_normalised_rank": _matrix_summary(
                matched["mean_normalised_rank"], "matched_mean_normalised_rank"
            ),
        },
        "top_heads_synthetic": top_heads(
            ranking, key="prefix_matching_synthetic", count=args.top_heads
        ),
        "per_head": {name: matrix.tolist() for name, matrix in ranking.items()},
    }

    for label in REPEAT_PROBES:
        scores = natural_scores[label]
        census = head_census(scores["scores"]["prefix_matching"])
        result[f"natural_repeat_{label}"] = {
            **{key: value for key, value in scores.items() if key != "scores"},
            "cohort": cohort_record(cohorts[f"repeat_{label}"]),
            "record_retention": natural_retention[label],
            "census": census,
            "headline": induction_headline(
                scores, census, threshold=args.headline_threshold
            ),
            "same_token_distribution": _matrix_summary(
                scores["scores"]["same_token"], "same_token"
            ),
            "offset_two_distribution": _matrix_summary(
                scores["scores"]["offset_two"], "offset_two"
            ),
        }
        result[f"top_heads_natural_{label}"] = top_heads(
            ranking, key=f"prefix_matching_natural_{label}", count=args.top_heads
        )
    return result


def head_ranking(
    synthetic: Mapping[str, np.ndarray],
    natural: Mapping[str, Mapping[str, np.ndarray]],
    copying: Mapping[str, np.ndarray],
    matched: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Every per-head matrix this stage ranks by and writes into ``per_head``.

    Derived by iterating the statistics the estimator returned, rather than by
    naming three of them.  Naming them is how the ``offset_two`` decoy came to be
    measured on the natural probes and then dropped on the floor: it was computed
    by :func:`attention_alignment_scores` for every probe, and only the synthetic
    probe's copy reached ``per_head``.  A distribution-body battery needs a
    positional decoy to correct with (Appendix B rule 8), and the probes that need
    it most are the natural ones -- for a wrapped arm they are the ones to trust --
    so the correction was missing exactly where it mattered, and recovering it
    would have cost a GPU re-run of the whole panel.

    Key spellings are unchanged; they are read by artefact consumers.
    """

    ranking: dict[str, np.ndarray] = {
        f"{name}_synthetic": matrix for name, matrix in synthetic.items()
    }
    for label, scores in natural.items():
        for name, matrix in scores.items():
            ranking[f"{name}_natural_{label}"] = matrix
    for name, matrix in copying.items():
        ranking[f"copy_{name}"] = matrix
    for name, matrix in matched.items():
        ranking[f"copy_matched_{name}"] = matrix
    return ranking


def _matrix_summary(values: np.ndarray, label: str) -> dict[str, Any]:
    return summarise_head_matrix(values, label)


def run_attribution(arm: Arm, analysis: Cohort, args: argparse.Namespace) -> dict[str, Any]:
    strings = analysis.input_strings(arm)[: args.attribution_sequences]
    ids, mask = tokenize_batch(arm, strings, args.attribution_max_tokens)
    result = direct_logit_attribution(arm, ids, mask)
    result["n_sequences"] = len(strings)
    result["max_tokens"] = args.attribution_max_tokens
    return result


def distance_band_plan(arm: Arm, strings: Sequence[str], args: argparse.Namespace):
    """Resolve this arm's patching geometry, in the unit the run declared.

    Under ``--patch-distance-unit content_symbol`` both legs of the geometry -- the
    perturbation-to-read-out distance and the size of the perturbation -- are
    resolved through this arm's *own* measured symbols per token, so the two arms
    of a comparison are matched in residues (or in characters) rather than in a
    unit whose size differs between them by 4.4x.

    The scale is measured over exactly the truncated rows the cases are cut from,
    at ``--patch-seq-len``, and not reused from the arm-level
    ``tokenisation.symbols_per_token`` in this artefact, which is measured over a
    different window (``--unigram-max-tokens``). Two measurements of one quantity
    are allowed to differ; taking one for the other silently is not.
    """

    unit = args.patch_distance_unit
    measured = (
        symbols_per_token(arm, strings, args.patch_seq_len)
        if unit == "content_symbol"
        else None
    )
    return resolve_distance_bands(
        requested_distance_bands(args),
        unit=unit,
        corruption_span=(
            args.patch_corruption_span
            if args.patch_corruption_span is not None
            else DEFAULT_CORRUPTION_SPAN_TOKENS
        ),
        symbols_per_token=measured,
    )


def run_patching(
    arm: Arm, analysis: Cohort, unigram: Unigram, args: argparse.Namespace
) -> dict[str, Any]:
    strings = analysis.input_strings(arm)
    plan = distance_band_plan(arm, strings, args)
    cases = build_patch_cases(
        arm,
        strings,
        unigram,
        seq_len=args.patch_seq_len,
        plan=plan,
        cases_per_band=args.patch_cases_per_band,
        seed=args.seed + 1,
    )
    result = activation_patching(
        arm,
        cases,
        minimum_effect=args.patch_minimum_effect,
        batch_size=args.patch_batch_size,
    )
    result["distance_geometry"] = {
        **plan.as_dict(),
        "content_symbol": content_symbol_name(arm),
        "symbols_per_token_source": {
            "window_tokens": int(args.patch_seq_len),
            "n_records": len(strings),
            "note": (
                "measured over exactly the rows the cases are cut from; the "
                "arm-level tokenisation.symbols_per_token in this artefact is a "
                "different measurement, over --unigram-max-tokens"
            ),
        },
    }
    result["summary"] = summarise_patching(result, arm=arm)
    return result


def run_arm(
    name: str, args: argparse.Namespace, cohorts: dict[str, dict[str, Cohort]]
) -> dict[str, Any]:
    started = time.time()
    arm = load_arm(
        name,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    circuits_granted = grant_circuits(arm) if name in args.grant_circuits else False
    if arm.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(arm.device)
    modality_cohorts = cohorts[arm.modality]
    analysis = modality_cohorts["analysis"]
    strings = analysis.input_strings(arm)
    unigram_max_tokens = conditioned_token_budget(
        arm, args.unigram_max_tokens, args.protein_max_len
    )
    unigram = fit_unigram(arm, strings, max_tokens=unigram_max_tokens)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "arm": arm_record(arm, args.attn_implementation, circuits_granted),
        "cohorts": {
            label: cohort_record(cohort) for label, cohort in modality_cohorts.items()
        },
        "repeat_criteria": {
            label: criterion.as_dict()
            for label, criterion in repeat_criteria(arm.modality, args).items()
        },
        "seeds": {"master": args.seed, "patching": args.seed + 1},
        "unigram_max_tokens": {
            "requested": int(args.unigram_max_tokens),
            "resolved_for_this_arm": int(unigram_max_tokens),
            "widened": unigram_max_tokens != args.unigram_max_tokens,
            "reason": (
                "an ec_conditioned rendering's <end> marker must survive truncation "
                "or content_bounds refuses the row; see conditioned_token_budget"
            ),
        },
        "thresholds": {
            "induction_prefix_matching": list(INDUCTION_THRESHOLDS),
            "induction_headline_threshold": args.headline_threshold,
            "induction_data_driven_sigma": 3.0,
            "patch_minimum_effect_logits": args.patch_minimum_effect,
            "ov_decomposition_relative_tolerance": 0.05,
            "dla_reconstruction_relative_tolerance": 0.02,
        },
        "tokenisation": {
            "symbols_per_token": symbols_per_token(
                arm, strings[: args.cohort_size], unigram_max_tokens
            ),
            "unigram": unigram.summary(),
        },
        "sections": list(args.sections),
    }

    if "induction" in args.sections:
        payload["induction"] = run_induction(arm, modality_cohorts, unigram, args)
    if "attribution" in args.sections:
        payload["direct_logit_attribution"] = run_attribution(arm, analysis, args)
    if "patching" in args.sections:
        payload["activation_patching"] = run_patching(arm, analysis, unigram, args)

    payload["runtime_seconds"] = round(time.time() - started, 2)
    payload["peak_gpu_bytes"] = (
        int(torch.cuda.max_memory_allocated(arm.device))
        if arm.device.startswith("cuda")
        else 0
    )

    del arm
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return payload


def panel_summary(results: dict[str, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """The cross-arm reading: one row per arm with the headline of each measurement.

    Every arm carries both natural-repeat probes, and ``probe_comparison`` puts
    the two head-count fractions side by side per arm.  That ratio is the whole
    question the approximate probe was built to answer, and leaving a reader to
    divide two numbers out of a nested payload is how a comparison gets quoted
    from the wrong column.
    """

    rows: dict[str, Any] = {}
    for name, payload in results.items():
        row: dict[str, Any] = {"modality": payload["arm"]["modality"]}
        induction = payload.get("induction")
        if induction is not None:
            census = induction["synthetic_repeat"]["census"]
            row["synthetic_repeat_head_mean"] = census["distribution"]["mean"]
            row["synthetic_repeat_head_max"] = census["distribution"]["max"]
            row["synthetic_repeat_n_above_0.10"] = census["count_above_threshold"]["0.10"]
            row["synthetic_repeat_n_above_data_driven"] = census["count_above_data_driven"]
            row["synthetic_repeat_uniform_baseline"] = induction["synthetic_repeat"][
                "uniform_baseline"
            ]
            for label in REPEAT_PROBES:
                probe = induction[f"natural_repeat_{label}"]
                row[f"natural_repeat_{label}"] = {
                    **probe["headline"],
                    "cohort_digest": probe["cohort"]["digest"],
                    "cohort_n_records": probe["cohort"]["n_records"],
                    "cohort_census": probe["cohort"]["census"],
                    "cohort_criterion": probe["cohort"]["criterion"],
                    "head_mean": probe["census"]["distribution"]["mean"],
                    # First-class, not buried: the cohort a census was computed on
                    # is the cohort that reached a probe, and on one arm that is
                    # 27% smaller than the cohort that was drawn.
                    "record_retention": probe["record_retention"]["record_retention"],
                    "n_records_retained": probe["record_retention"]["n_retained"],
                }
            row["copy_matched_mean_rank"] = induction["copying"]["matched_mean_normalised_rank"][
                "max"
            ]
            row["copy_matched_mean_rank_panel_mean"] = induction["copying"][
                "matched_mean_normalised_rank"
            ]["mean"]
        attribution = payload.get("direct_logit_attribution")
        if attribution is not None:
            row["dla_pathway_magnitude_fraction"] = attribution["pathway_magnitude_fraction"]
            row["dla_top1_share"] = attribution["concentration"]["top1_share"]
            row["dla_participation_fraction"] = attribution["concentration"][
                "participation_fraction"
            ]
            row["dla_logit_mean_absolute_error"] = attribution["logit_mean_absolute_error"]
            row["dla_residual_relative_l2_error"] = attribution["residual_relative_l2_error"]
        patching = payload.get("activation_patching")
        if patching is not None:
            row["patch_corruption_effect"] = {
                band: values["mean_absolute_effect"]
                for band, values in patching["corruption_effect"].items()
            }
            row["patch_eligible_fraction"] = {
                band: values["eligible_cases"] / values["n_cases"]
                for band, values in patching["corruption_effect"].items()
            }
            row["patch_best_resid_q"] = {
                band: entry["best_mean"]
                for band, entry in patching["summary"]["resid_post|q"].items()
            }
            # Without these three the band labels above are ambiguous: "33-64" is a
            # token band on one run and a content-symbol band on another, and under
            # the second it resolves to a different token band on every arm.
            geometry = patching["distance_geometry"]
            row["patch_distance_unit"] = geometry["unit"]
            row["patch_content_symbol"] = geometry["content_symbol"]
            row["patch_band_tokens"] = {
                band["label"]: band["tokens"] for band in geometry["bands"]
            }
            row["patch_corruption_span"] = geometry["corruption_span"]
        rows[name] = row
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "matched_pair": list(MATCHED_PAIR),
        "arms": rows,
        # Which eligible arms this invocation left out. Without it a one-arm
        # far-band shard and a full-panel run write panel summaries a reader
        # cannot tell apart -- L18 at the artefact level, on the stage whose
        # declared scope is panel-wide. (EXP-R2-073.)
        "stage_contract": stage_contract_record("circuit_primitives", sorted(rows)),
        "probe_comparison": probe_comparison(rows),
        "configuration": {key: value for key, value in sorted(vars(args).items()) if key != "output_dir"},
    }


def probe_comparison(rows: dict[str, Any]) -> dict[str, Any]:
    """Exact against approximate, per arm, and every arm against the text reference.

    ``deficit_versus_reference`` is the reference arm's fraction of heads above
    threshold divided by this arm's, under each probe: the "five times fewer
    induction heads" reading, recomputed per probe so that a change of probe
    cannot be reported without the number it changed.  It is undefined rather
    than infinite when an arm has no head above threshold, because a zero
    numerator and a zero denominator carry different meanings and dividing them
    would hide both.
    """

    reference = MATCHED_PAIR[0]
    if reference not in rows:
        return {"reference_arm": None, "note": f"{reference} not in this run"}
    sizes = {
        label: {
            row[f"natural_repeat_{label}"]["cohort_n_records"]
            for row in rows.values()
            if f"natural_repeat_{label}" in row
        }
        for label in REPEAT_PROBES
    }
    comparison: dict[str, Any] = {
        "reference_arm": reference,
        "cohort_sizes": {label: sorted(value) for label, value in sizes.items()},
        "size_matched": sizes[REPEAT_PROBES[0]] == sizes[REPEAT_PROBES[1]],
        "arms": {},
    }
    for name, row in rows.items():
        if f"natural_repeat_{REPEAT_PROBES[0]}" not in row:
            continue
        entry: dict[str, Any] = {}
        for label in REPEAT_PROBES:
            probe = row[f"natural_repeat_{label}"]
            reference_fraction = rows[reference][f"natural_repeat_{label}"][
                "fraction_above_threshold"
            ]
            entry[label] = {
                "fraction_above_threshold": probe["fraction_above_threshold"],
                "peak_over_uniform": probe["peak_over_uniform"],
                "cohort_n_matching": probe["cohort_census"]["n_matching"],
                "deficit_versus_reference": (
                    reference_fraction / probe["fraction_above_threshold"]
                    if probe["fraction_above_threshold"] > 0.0
                    else None
                ),
            }
        exact = entry[REPEAT_PROBES[0]]
        approximate = entry[REPEAT_PROBES[1]]
        entry["approximate_over_exact_fraction"] = (
            approximate["fraction_above_threshold"] / exact["fraction_above_threshold"]
            if exact["fraction_above_threshold"] > 0.0
            else None
        )
        entry["approximate_over_exact_peak"] = (
            approximate["peak_over_uniform"] / exact["peak_over_uniform"]
        )
        comparison["arms"][name] = entry
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(MATCHED_PAIR), choices=sorted(PANEL))
    parser.add_argument("--sections", nargs="+", default=list(SECTIONS), choices=SECTIONS)
    parser.add_argument(
        "--grant-circuits",
        nargs="*",
        default=[],
        choices=sorted(PANEL),
        help=(
            "Arms for which this runner declares the 'circuits' capability that "
            "arms.py still withholds. Use only for an architecture whose per-head "
            "decomposition circuits.py implements; verify_head_decomposition checks "
            "it on every run and the grant is recorded in the output. Remove once "
            "arms.py declares the capability itself."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="float32",
        help="inference dtype. float32 rather than bfloat16 because the patching "
        "section measures a difference of two logits of order 0.05 against a "
        "bfloat16 quantisation step of 0.0625 at logit scale, and reading one off "
        "the other inflated an eligible fraction by 60% relative (Appendix B rule "
        "15b). --sections patching is refused outright under a dtype in "
        "circuits.PATCHING_REFUSED_DTYPES; the other sections may still be run "
        "under one, deliberately",
    )
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260728)

    parser.add_argument("--cohort-size", type=int, default=24)
    parser.add_argument(
        "--cohort-skip",
        type=int,
        default=0,
        help="records to pass over before the draw starts. Under a seed this "
        "indexes a DISJOINT window of the same permutation, so two runs at one "
        "seed and different skips are the sampling sensitivity Appendix B rule 1 "
        "asks for rather than two overlapping prefixes",
    )
    parser.add_argument(
        "--cohort-draw-seed",
        type=int,
        default=DEFAULT_CORPUS_DRAW_SEED,
        help="seed for the permutation the analysis cohort is drawn under; 0 "
        "selects the historical file-order prefix, which is a declared choice "
        "and not a default (transfer audit, Appendix B rule 1). At the default "
        "--cohort-size of 24 this matters more than anywhere else in the "
        "package: 24 head-of-file Swiss-Prot records are a single family",
    )
    parser.add_argument("--protein-min-len", type=int, default=600)
    parser.add_argument("--protein-max-len", type=int, default=1000)
    parser.add_argument("--text-min-chars", type=int, default=3000)
    parser.add_argument("--unigram-max-tokens", type=int, default=256)

    parser.add_argument("--repeat-cohort-size", type=int, default=32)
    parser.add_argument(
        "--approximate-cohort-size",
        type=int,
        default=None,
        help=(
            "records in the approximate repeat cohort; defaults to "
            "--repeat-cohort-size, which is the size-matched comparison"
        ),
    )
    parser.add_argument("--repeat-min-len", type=int, default=200)
    parser.add_argument("--repeat-max-len", type=int, default=800)
    parser.add_argument("--protein-repeat-unit", type=int, default=16)
    parser.add_argument("--text-repeat-chars", type=int, default=2000)
    parser.add_argument("--text-repeat-unit", type=int, default=40)
    parser.add_argument("--text-repeat-scan", type=int, default=3000)
    parser.add_argument("--natural-max-tokens", type=int, default=840)
    parser.add_argument("--natural-batch-size", type=int, default=2)
    # The repeat census is a full scan of the eligible corpus, not a scan that
    # stops once the cohort is full, because the number of eligible records is
    # what caps the cohort and it is reported. On the EC-labelled source that is
    # two hundred thousand searches per criterion, so it is worth spreading.
    parser.add_argument("--repeat-scan-workers", type=int, default=64)
    parser.add_argument("--headline-threshold", type=float, default=0.10)

    parser.add_argument("--synthetic-probes", type=int, default=16)
    parser.add_argument("--synthetic-copy-len", type=int, default=64)
    parser.add_argument("--probe-batch-size", type=int, default=4)
    parser.add_argument("--copy-tokens", type=int, default=512)
    parser.add_argument("--copy-matched-n", type=int, default=20)
    parser.add_argument("--copy-matched-repeats", type=int, default=8)
    parser.add_argument("--top-heads", type=int, default=10)

    parser.add_argument("--attribution-sequences", type=int, default=8)
    parser.add_argument("--attribution-max-tokens", type=int, default=256)

    parser.add_argument("--patch-seq-len", type=int, default=128)
    parser.add_argument(
        "--patch-cases-per-band",
        type=int,
        default=32,
        help="cases drawn per distance band. The default is a validation size: "
        "at 32 the 33-64 band yielded 2-16 *eligible* cases across the panel, "
        "which is what plan item B6 raises. Eligibility rates differ by an order "
        "of magnitude across arms (ZymCTRL 0.06, ProGen2-medium 0.50), so a "
        "production run must size this against the weakest arm",
    )
    parser.add_argument(
        "--patch-batch-size",
        type=int,
        default=64,
        help="cases per forward pass. The read-out needs one position but the "
        "model materialises the full [cases, width, vocab] logit tensor, so this "
        "and not the model is what bounds --patch-cases-per-band",
    )
    parser.add_argument("--patch-minimum-effect", type=float, default=0.25)
    parser.add_argument(
        "--patch-distance-unit",
        default="token",
        choices=list(DISTANCE_UNITS),
        help="unit the distance band and the corruption span are declared in. "
        "'token' reproduces every artefact measured before this option existed and "
        "matches the positional distance while leaving the perturbation mismatched "
        "(one token is 1 residue on ProGen2, ~2.8 on ProtGPT2, ~4.4 characters on "
        "GPT-2). 'content_symbol' resolves both legs per arm from its own measured "
        "symbols per token, is refused for a run spanning both modalities because a "
        "residue is not a character, and requires --patch-corruption-span",
    )
    parser.add_argument(
        "--patch-distance-band",
        nargs="+",
        default=None,
        metavar="LOW-HIGH",
        help="distance bands, inclusive, in --patch-distance-unit; defaults to "
        "circuits.DISTANCE_BANDS for tokens and circuits.SYMBOL_DISTANCE_BANDS for "
        "content symbols",
    )
    parser.add_argument(
        "--patch-corruption-span",
        type=int,
        default=None,
        help="size of the perturbation, in --patch-distance-unit. Defaults to one "
        "token, which is what every existing artefact carries. Under a "
        "content-symbol geometry it must be declared and it cannot be finer than "
        "one token of the arm being measured, so the finest matched perturbation a "
        "set of arms can reach is set by its coarsest tokenizer",
    )

    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("duplicate arms requested")
    if "patching" in args.sections:
        validate_patching_arguments(args)
    if args.repeat_scan_workers < 1:
        raise ValueError("repeat scan needs at least one worker")
    if args.approximate_cohort_size is not None and args.approximate_cohort_size < 1:
        raise ValueError("approximate cohort size must be positive")
    if f"{args.headline_threshold:.2f}" not in {f"{v:.2f}" for v in INDUCTION_THRESHOLDS}:
        raise ValueError(
            f"headline threshold {args.headline_threshold} is not one of the census "
            f"thresholds {list(INDUCTION_THRESHOLDS)}"
        )
    return args





def parse_distance_band(text: str) -> tuple[int, int]:
    """``"33-64"`` as an inclusive band -- the spelling the artefact labels use."""

    low, _, high = text.partition("-")
    if not high or not low.strip().isdigit() or not high.strip().isdigit():
        raise ValueError(f"distance band {text!r} must be spelled LOW-HIGH")
    band = (int(low), int(high))
    if band[0] < 1 or band[1] < band[0]:
        raise ValueError(f"invalid distance band {text!r}")
    return band


def requested_distance_bands(args: argparse.Namespace) -> tuple[tuple[int, int], ...]:
    """The ladder this run declared, in ``--patch-distance-unit``."""

    if args.patch_distance_band:
        return tuple(parse_distance_band(item) for item in args.patch_distance_band)
    if args.patch_distance_unit == "content_symbol":
        return SYMBOL_DISTANCE_BANDS
    return DISTANCE_BANDS


def validate_patching_arguments(args: argparse.Namespace) -> None:
    """Every patching precondition that can be decided from the command line.

    These used to be found at run time, after the checkpoint was on the GPU: one
    produced a measurement that had to be withdrawn, one produced a job that could
    not have produced anything at all, and one would have produced a
    cross-modality number in a unit that does not cross modalities.  Reading only
    ``args`` and the panel declaration, they cost nothing and are checked before a
    lane is allocated.

    The per-arm resolution of a content-symbol geometry is *not* checked here: it
    needs the arm's measured symbols per token, so it is enforced in
    :func:`~src.transfer.circuits.resolve_distance_bands` at build time instead.
    """

    bands = requested_distance_bands(args)
    widest = max(high for _, high in bands)
    if args.patch_distance_unit == "token" and args.patch_seq_len <= widest + 2:
        raise ValueError("patch sequence length must exceed the widest distance band")
    if args.patch_corruption_span is not None and args.patch_corruption_span < 1:
        raise ValueError("--patch-corruption-span must be at least one")
    if args.patch_distance_unit == "content_symbol":
        modalities = sorted({PANEL[name].modality for name in args.arms})
        if len(modalities) > 1:
            raise ValueError(
                f"--patch-distance-unit content_symbol is refused for a run spanning "
                f"{modalities}: a content symbol is a residue on a protein arm and a "
                f"character on a text arm, so one band label would name two different "
                f"quantities. The comparisons this geometry identifies are within a "
                f"modality -- run the protein arms and the text arms as separate "
                f"invocations"
            )
        if args.patch_corruption_span is None:
            raise ValueError(
                "--patch-distance-unit content_symbol requires "
                "--patch-corruption-span: matching the distance while leaving the "
                "perturbation at one token per arm swaps one confound for another, "
                "since one token is ~2.8 residues on ProtGPT2 and 1 on ProGen2"
            )
    if args.dtype in PATCHING_REFUSED_DTYPES:
        raise ValueError(
            f"--sections patching is refused at --dtype {args.dtype}: the metric is a "
            f"difference of two logits and --patch-minimum-effect is "
            f"{args.patch_minimum_effect}, which is at or below this dtype's own "
            f"quantisation step at logit scale (Appendix B rule 15b). Run at "
            f"--dtype float32, or drop 'patching' from --sections"
        )
    for name in args.arms:
        refusal = patch_seq_len_refusal(
            PANEL[name],
            args.patch_seq_len,
            min_symbols=args.protein_min_len,
            max_symbols=args.protein_max_len,
        )
        if refusal is not None:
            raise ValueError(refusal)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{args.device} was requested but no CUDA device is present")
    torch.manual_seed(args.seed)

    # Cohorts first, and before any checkpoint is loaded: the repeat census forks
    # a process pool, and forking a process that has already initialised a CUDA
    # context is a class of failure that shows up as a hang rather than an error.
    modalities = {PANEL[name].modality for name in args.arms}
    cohorts = {modality: build_cohorts(modality, args) for modality in sorted(modalities)}
    for modality, built in sorted(cohorts.items()):
        # Only present when the induction section was requested; build_cohorts
        # skips the corpus-wide repeat scan otherwise. Keyed off what was built
        # rather than off the section list a second time, so the two cannot
        # disagree about whether a scan happened.
        for label in [name for name in REPEAT_PROBES if f"repeat_{name}" in built]:
            census = built[f"repeat_{label}"].metadata["census"]
            print(
                f"[cohort] {modality} {label}: {census['n_matching']} matching of "
                f"{census['scanned_eligible']} eligible "
                f"({census['match_rate']:.5%}), cohort of "
                f"{len(built[f'repeat_{label}'])}",
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

    # Re-emit every artifact together at the end: the results root is shared and
    # is recreated by other tracks, so a per-arm write made ten minutes ago may
    # no longer be on disk.
    for name, payload in results.items():
        write_json(args.output_dir / f"{name}.json", payload)
    write_json(args.output_dir / "panel_summary.json", panel_summary(results, args))
    verify_outputs(args.output_dir, sorted(results))
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
