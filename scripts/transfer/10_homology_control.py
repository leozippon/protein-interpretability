#!/usr/bin/env python3
"""Is the protein induction signal a computation, or retrieval of a memorised sequence?

The induction-head finding rests on prefix-matching attention measured over
natural tandem repeats in Swiss-Prot proteins.  Swiss-Prot is largely inside
UniRef50, which is what the protein decoders were pretrained on, so a head that
attends from a repeated span to what followed the earlier copy may be running a
general copying computation or may be retrieving a near-duplicate it has stored.
The two are indistinguishable from the induction score alone and imply opposite
things about the head-count comparison.

This script separates them.  It searches every cohort sequence against the
pretraining corpus with DIAMOND, bins the cohort by maximum sequence identity
using boundaries fixed in :mod:`src.transfer.homology` before any result was
seen, and recomputes the induction quantities within each bin with the same
estimators the headline used.  The synthetic-repeat probe -- constructed in token
space and present in no corpus -- is the negative control that no database can
contaminate.

Two stages, so that the CPU search and the GPU measurement can be run and re-run
independently:

``search``     build the cohort, index the corpus if needed, run DIAMOND, assign
               strata.  CPU only; writes ``homology_assignment.json``.
``induction``  per arm, score every record's probe individually, pool by
               stratum, and report.  Needs one GPU.

The limitation that cannot be engineered away is stated in the module docstring
of ``src.transfer.homology`` and repeated in every artifact this script writes:
GPT-2's training corpus is not public, so there is no symmetric text-arm control
and no matched cross-modal claim is available from this measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
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
    PANEL,
    Arm,
    Cohort,
    env_path,
    load_arm,
)
from src.transfer.circuits import (  # noqa: E402
    INDUCTION_THRESHOLDS,
    PROTEIN_APPROXIMATE_CRITERION,
    PROTEIN_EXACT_CRITERION,
    fit_unigram,
    matched_copying_scores,
    n_head,
    protein_repeat_cohort,
    summarise_head_matrix,
    synthetic_repeat_probes,
)
from src.transfer.homology import (  # noqa: E402
    DIAMOND_FIELDS,
    MINIMUM_BOOTSTRAP_UNITS,
    SCHEMA_VERSION,
    STRATUM_EDGES,
    STRATUM_NAMES,
    HomologyAssignment,
    ProbeScore,
    alignment_block,
    assign_homology,
    bootstrap_stratum,
    build_database,
    covariate_analysis,
    distinct_stratum_counts,
    head_set_overlap,
    parse_hits,
    pool_scores,
    prepare_diamond,
    probe_for_record,
    representative_scores,
    run_diamond_blastp,
    score_probe,
    sequence_groups,
    stratum_counts,
    stratum_integrity,
    stratum_report,
    verify_pooling,
    write_query_fasta,
)

DEFAULT_OUTPUT = REPO_ROOT / "results/transfer/homology_control"
STAGES = ("search", "induction")
ASSIGNMENT_FILE = "homology_assignment.json"

#: The arms this control applies to.  A text arm is deliberately absent: see
#: ``NO_TEXT_ARM`` below, which is written into every artifact.
PROTEIN_ARMS = ("protgpt2", "zymctrl", "progen2-medium")

NO_TEXT_ARM = (
    "GPT-2's training corpus (WebText) was never released. OpenWebText is a "
    "reconstruction of the collection procedure, not the corpus GPT-2 saw, so a "
    "miss against it cannot distinguish 'GPT-2 never saw this document' from "
    "'the reconstruction happens not to contain it'. No text-side homology "
    "stratification is constructed here. This control therefore speaks to "
    "whether the PROTEIN induction signal is general or retrieved; it cannot "
    "support a matched cross-modal memorisation claim."
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist; run this script with --stages search first"
        )
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unexpected schema_version {payload.get('schema_version')!r}")
    return payload


#: The two repeat criteria this control can be run against.  ``exact`` is the one
#: the headline was measured on and is the default, because a control that
#: attacks a finding has to be run on the finding's own cohort.  ``approximate``
#: exists because the exact cohort has a hard ceiling in the low tens, which caps
#: how finely the strata can be cut; the approximate cohort is a superset and buys
#: the stratification real power at the cost of no longer being the headline's
#: exact cohort.  Which one produced an artifact is recorded in it.
CRITERIA = {
    "exact": PROTEIN_EXACT_CRITERION,
    "approximate": PROTEIN_APPROXIMATE_CRITERION,
}


def build_cohort(args: argparse.Namespace) -> Cohort:
    """The natural-repeat cohort exactly as the induction census builds it.

    Same constructor, same criterion, same draw, so under ``--repeat-criterion
    exact`` the cohort stratified here is the cohort the headline was measured on
    rather than a re-derivation of it.  The digest is carried into every artifact
    and checked when the two stages are run separately.

    ``--cohort-draw-seed`` is part of "same draw" and it is the part that moved.
    The census used to take the first ``n`` matching records in corpus file order
    -- which for the approximate criterion is 32 of 817 matching proteins, a four
    per cent head-of-file prefix -- and now draws them under a seeded permutation
    (EXP-R2-068).  This stage follows that default so the two agree; stratifying a
    census stored before the change requires ``--cohort-draw-seed 0``, and a
    mismatch is caught by the digest comparison rather than tolerated.
    """

    return protein_repeat_cohort(
        args.repeat_cohort_size,
        min_len=args.repeat_min_len,
        max_len=args.repeat_max_len,
        criterion=CRITERIA[args.repeat_criterion],
        workers=args.cohort_workers,
        seed=args.cohort_draw_seed or None,
    )


def strata_record() -> dict[str, Any]:
    return {
        "names": list(STRATUM_NAMES),
        "edges_percent_identity": list(STRATUM_EDGES),
        "bands": {
            name: {"low_inclusive": STRATUM_EDGES[index], "high_exclusive": STRATUM_EDGES[index + 1]}
            for index, name in enumerate(STRATUM_NAMES)
        },
        "identity_definition": (
            "100 * nident / qlen for the best DIAMOND HSP, i.e. the percent of the "
            "QUERY that is identically matched. Not pident, which is identity "
            "within the aligned region and would call a corpus fragment a "
            "near-duplicate of a longer query."
        ),
        "fixed_before_results": True,
    }


# ---------------------------------------------------------------- stage: search


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    cohort = build_cohort(args)
    tool = prepare_diamond(args.diamond_tarball, args.diamond_checksum, args.diamond_dir)
    database = build_database(
        tool,
        args.corpus_fasta,
        args.diamond_db,
        threads=args.threads,
        tmpdir=args.diamond_tmpdir,
        rebuild=args.rebuild_db,
    )

    query_fasta = args.output_dir / "cohort_query.faa"
    identifiers = write_query_fasta(cohort, query_fasta)
    hits_tsv = args.output_dir / "diamond_hits.tsv"
    command, log_tail = run_diamond_blastp(
        tool,
        database,
        query_fasta,
        hits_tsv,
        threads=args.threads,
        sensitivity=args.sensitivity,
        evalue=args.evalue,
        max_target_seqs=args.max_target_seqs,
    )
    hits = parse_hits(hits_tsv)
    # ``max_target_seqs`` is threaded through so that ``hit_list_saturated`` is a
    # fact in the artefact rather than a ``None``: a query whose hit list reached
    # the cap has a "closest relative" that is the closest of the ones DIAMOND
    # chose to report, and nothing downstream could otherwise tell.
    assignments = assign_homology(
        cohort, identifiers, hits, max_target_seqs=args.max_target_seqs
    )
    counts = stratum_counts(assignments)
    groups = sequence_groups(cohort)
    duplicates = {}
    for index, group in enumerate(groups):
        duplicates.setdefault(str(group), []).append(index)

    identities = np.asarray(
        [assignment.max_identity_over_query for assignment in assignments], dtype=np.float64
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "search",
        "no_text_arm": NO_TEXT_ARM,
        "cohort": {
            "name": cohort.name,
            "kind": cohort.kind,
            "digest": cohort.digest,
            "n_records": len(cohort),
            "min_symbols": cohort.min_symbols,
            "max_symbols": cohort.max_symbols,
            "requested_size": args.repeat_cohort_size,
            "criterion": cohort.metadata["criterion"],
            "census": cohort.metadata["census"],
            "is_whole_population": len(cohort) == int(cohort.metadata["census"]["n_matching"]),
            "constructor": (
                "src.transfer.circuits.protein_repeat_cohort("
                f"n={args.repeat_cohort_size}, min_len={args.repeat_min_len}, "
                f"max_len={args.repeat_max_len}, "
                f"criterion=PROTEIN_{args.repeat_criterion.upper()}_CRITERION)"
            ),
        },
        "diamond": tool.record(),
        "database": database.record(),
        "search": {
            "command": command,
            "outfmt_fields": list(DIAMOND_FIELDS),
            "sensitivity": args.sensitivity,
            "evalue": args.evalue,
            "max_target_seqs": args.max_target_seqs,
            "threads": args.threads,
            "query_fasta": str(query_fasta),
            "hits_tsv": str(hits_tsv),
            "n_hits": len(hits),
            "queries_with_no_hit": int(sum(1 for a in assignments if a.n_hits == 0)),
            "log_tail": log_tail,
        },
        "strata": strata_record(),
        "stratum_counts": counts,
        # How much of each stratum its own evidence supports: a best hit that does
        # not cover the record's repeat span cannot explain induction on that span,
        # and a saturated hit list means the maximum is over reported hits only.
        "stratum_integrity": stratum_integrity(assignments),
        "stratum_counts_distinct_sequences": distinct_stratum_counts(assignments, groups),
        "sequence_groups": {
            "group_of_record": groups,
            "duplicate_groups": {
                key: value for key, value in duplicates.items() if len(value) > 1
            },
            "n_distinct_sequences": len(set(groups)),
            "note": (
                "the EC-labelled source carries one record per (protein, EC number) "
                "pair, so a protein with several EC numbers appears several times "
                "with a byte-identical sequence. Those records are one observation. "
                "Every interval in this control resamples distinct sequences, and "
                "the per-stratum headline numbers are computed over one probe per "
                "distinct sequence; the duplicate-weighted values the headline "
                "census used are reported alongside as `duplicate_weighted`."
            ),
        },
        "identity_summary": {
            "min": float(identities.min()),
            "median": float(np.median(identities)),
            "mean": float(identities.mean()),
            "max": float(identities.max()),
            "n_at_100": int((identities >= 99.999).sum()),
            "n_best_hit_spans_repeat": int(
                sum(1 for a in assignments if a.best_hit_spans_repeat is True)
            ),
        },
        "assignments": [assignment.record() for assignment in assignments],
        "runtime_seconds": round(time.time() - started, 2),
    }


# ------------------------------------------------------------- stage: induction


def synthetic_control(
    arm: Arm,
    cohort: Cohort,
    unigram: Any,
    copying: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """The probe that cannot have been memorised.

    Synthetic repeats are drawn in token space from the arm's own unigram, so no
    corpus contains them and no amount of training-set overlap can produce
    induction-shaped attention on them by retrieval.  If a stratum contrast is
    ambiguous, this is the arm of the control that still discriminates: retrieval
    scores zero here by construction, a general copying computation does not.
    """

    ec_label = None
    if arm.spec.input_format == "ec_conditioned":
        labels = cohort.metadata.get("ec_labels")
        if not labels:
            raise ValueError(f"{arm.name}: cohort carries no EC labels for conditioning")
        ec_label = labels[0]
    probes = synthetic_repeat_probes(
        arm,
        unigram,
        n_probes=args.synthetic_probes,
        copy_len=args.synthetic_copy_len,
        seed=args.seed,
        ec_label=ec_label,
    )
    # Every synthetic probe is an independent draw from the unigram, so each is
    # its own resampling unit; there is no duplicate structure to collapse.
    scores = [score_probe(arm, probe, index) for index, probe in enumerate(probes)]
    return {
        **alignment_block(
            arm, scores, copying, headline_threshold=args.headline_threshold
        ),
        "measured": True,
        "copy_len_tokens": args.synthetic_copy_len,
        "bootstrap": bootstrap_stratum(
            scores,
            threshold=args.headline_threshold,
            n_heads=n_head(arm) * arm.n_layer,
            resamples=args.bootstrap_resamples,
            seed=args.seed + 7,
        ),
    }


def _is_cuda(device: str) -> bool:
    """Whether a ``--device`` string names a CUDA device.

    The induction stage is arithmetic, not training: it runs the same estimators
    on CPU, more slowly, and the homology re-run after the DIAMOND masking defect
    is CPU-only by instruction. The CUDA bookkeeping calls below are therefore
    made conditional rather than assumed, and the device actually used is written
    into the artefact so that a CPU pass can never be mistaken for the GPU one it
    is compared against.
    """

    return torch.device(device).type == "cuda"


def _intervals_overlap(left: Sequence[float], right: Sequence[float]) -> bool:
    return not (left[1] < right[0] or right[1] < left[0])


def _finite_or_none(value: float) -> float | None:
    result = float(value)
    return result if np.isfinite(result) else None


def adjudicate(
    strata: dict[str, Any], synthetic: dict[str, Any], *, min_probes: int
) -> dict[str, Any]:
    """Apply the three fixed readings, with the decision rule stated in the output.

    The rule is deliberately made of interval-overlap tests only, so that it
    contains no tuned constant that could have been chosen after seeing the
    numbers.  A stratum with fewer than ``min_probes`` probes is excluded from the
    comparison (its bootstrap is degenerate) but is still reported.

    A stratum whose bootstrap came back ``degenerate`` is excluded here too, and
    for a reason independent of ``min_probes``: below
    :data:`~src.transfer.homology.MINIMUM_BOOTSTRAP_UNITS` there is no interval to
    compare, and the previous version would have compared ``None``.  The 2026-07-28
    run decided two ``consistent_with_memorisation`` verdicts by non-overlap of a
    four-unit interval against a four-hundred-unit one, which is what the floor
    exists to prevent; the excluded strata are named in the output as underpowered
    rather than dropped from the report.
    """

    usable: list[str] = []
    underpowered: dict[str, Any] = {}
    for name in STRATUM_NAMES:
        entry = strata[name]
        if not entry["measured"]:
            continue
        units = entry["n_distinct_sequences"]
        degenerate = bool(entry["bootstrap"].get("degenerate"))
        if units >= min_probes and not degenerate:
            usable.append(name)
            continue
        underpowered[name] = {
            "n_distinct_sequences": units,
            "n_records": entry["n_records"],
            "peak_over_uniform": entry["peak_over_uniform"],
            "n_above_threshold": entry["census"]["count_above_threshold"],
            "bootstrap_degenerate": degenerate,
            "reason": entry["bootstrap"].get("degenerate_reason")
            or f"fewer than {min_probes} distinct sequences",
        }
    criteria = {
        "rule": (
            "consistent_with_memorisation if the lowest-homology usable stratum's "
            "peak-over-uniform bootstrap interval lies strictly below the "
            "highest-homology usable stratum's; consistent_with_general_mechanism if "
            "those intervals overlap AND the synthetic-repeat interval overlaps the "
            "lowest-homology stratum's; indeterminate otherwise."
        ),
        "min_distinct_sequences_for_comparison": int(min_probes),
        "bootstrap_unit_floor": MINIMUM_BOOTSTRAP_UNITS,
        "usable_strata": usable,
        "underpowered_strata": underpowered,
    }
    if len(usable) < 2:
        return {
            **criteria,
            "verdict": "indeterminate",
            "reason": (
                f"only {len(usable)} stratum has at least {min_probes} distinct "
                "sequences and a non-degenerate bootstrap, so no "
                "across-stratum comparison is available; the synthetic control is "
                "reported and is the only evidence this arm contributes."
            ),
            "strata_separated": None,
            "synthetic_matches_low_homology": None,
        }

    low, high = usable[0], usable[-1]
    low_ci = strata[low]["bootstrap"]["peak_over_uniform_ci"]
    high_ci = strata[high]["bootstrap"]["peak_over_uniform_ci"]
    synthetic_ci = synthetic["bootstrap"]["peak_over_uniform_ci"]
    separated = low_ci[1] < high_ci[0]
    matches = _intervals_overlap(synthetic_ci, low_ci)
    if separated:
        verdict = "consistent_with_memorisation"
    elif _intervals_overlap(low_ci, high_ci) and matches:
        verdict = "consistent_with_general_mechanism"
    else:
        verdict = "indeterminate"

    # What the test could have seen. A non-separation reached with four distinct
    # sequences in the low bin is mostly a statement about power, and a reader
    # who is handed only the verdict cannot tell that from a genuine null. These
    # are the numbers that make the difference visible; none of them enters the
    # rule above.
    resolution = {
        "note": (
            "the rule separates the strata only when the low bin's interval lies "
            "wholly below the high bin's, so these state how large a memorisation "
            "gradient would have had to be for this cohort to reveal one"
        ),
        "low_stratum_distinct_sequences": strata[low]["n_distinct_sequences"],
        "high_stratum_distinct_sequences": strata[high]["n_distinct_sequences"],
        "separation_margin_peak_over_uniform": _finite_or_none(high_ci[0] - low_ci[1]),
        "high_stratum_peak_would_have_had_to_exceed": low_ci[1],
        "observed_high_over_low_peak_ratio": _finite_or_none(
            strata[high]["peak_over_uniform"] / strata[low]["peak_over_uniform"]
        ),
    }

    # Reported, deliberately NOT part of the verdict rule, which was fixed before
    # any result existed and is not being retuned. The head COUNT is the quantity
    # the headline actually claims ("about five times fewer of them"), so its own
    # interval comparison is worth seeing next to the peak's.
    low_fraction = strata[low]["bootstrap"]["fraction_above_threshold_ci"]
    high_fraction = strata[high]["bootstrap"]["fraction_above_threshold_ci"]
    additional = {
        "note": (
            "reported alongside the verdict, not an input to it; the verdict rule "
            "was fixed on peak-over-uniform before any result was seen"
        ),
        "fraction_above_threshold_ci": {
            low: low_fraction,
            high: high_fraction,
            "synthetic_point": synthetic["fraction_above_threshold"],
        },
        "head_count_strata_separated": bool(low_fraction[1] < high_fraction[0]),
        "head_count_intervals_overlap": bool(_intervals_overlap(low_fraction, high_fraction)),
    }
    return {
        **criteria,
        "compared": {"low_homology": low, "high_homology": high},
        "peak_over_uniform_ci": {low: low_ci, high: high_ci, "synthetic": synthetic_ci},
        "strata_separated": bool(separated),
        "synthetic_matches_low_homology": bool(matches),
        "verdict": verdict,
        "resolution": resolution,
        "additional_tests": additional,
    }


def run_arm(
    name: str,
    cohort: Cohort,
    assignments: Sequence[HomologyAssignment],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    arm = load_arm(
        name,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    on_cuda = _is_cuda(args.device)
    if on_cuda:
        torch.cuda.reset_peak_memory_stats(arm.device)
    strings = cohort.input_strings(arm)
    unigram = fit_unigram(arm, strings, max_tokens=args.unigram_max_tokens)
    copying = matched_copying_scores(
        arm,
        unigram.token_ids,
        matched_n=args.copy_matched_n,
        repeats=args.copy_matched_repeats,
        seed=args.seed,
    )

    scores: list[ProbeScore] = []
    probes = []
    without_probe: list[int] = []
    for index in range(len(cohort.records)):
        probe = probe_for_record(arm, cohort, index, max_tokens=args.natural_max_tokens)
        if probe is None:
            without_probe.append(index)
            continue
        probes.append(probe)
        scores.append(score_probe(arm, probe, index))
    if not scores:
        raise RuntimeError(f"{arm.name}: no cohort record yielded a natural-repeat probe")

    pooling_error = verify_pooling(
        arm,
        probes[: args.pooling_check_probes],
        scores[: args.pooling_check_probes],
        tolerance=args.pooling_tolerance,
    )

    groups = sequence_groups(cohort)
    group_of = {index: group for index, group in enumerate(groups)}
    by_stratum: dict[str, list[ProbeScore]] = {name: [] for name in STRATUM_NAMES}
    for score in scores:
        by_stratum[assignments[score.record_index].stratum].append(score)
    records_per_stratum = stratum_counts(assignments)

    strata = {
        name: stratum_report(
            arm,
            by_stratum[name],
            copying,
            group_of,
            n_records=records_per_stratum[name],
            headline_threshold=args.headline_threshold,
            resamples=args.bootstrap_resamples,
            seed=args.seed + index,
        )
        for index, name in enumerate(STRATUM_NAMES)
    }
    pooled_all = stratum_report(
        arm,
        scores,
        copying,
        group_of,
        n_records=len(cohort.records),
        headline_threshold=args.headline_threshold,
        resamples=args.bootstrap_resamples,
        seed=args.seed + 101,
    )
    synthetic = synthetic_control(arm, cohort, unigram, copying, args)

    # The stratum table bins on identity, but identity is correlated with repeat
    # length across the cohort and repeat length drives prefix matching on its
    # own. Without this, a length gradient dressed as a homology gradient would
    # be indistinguishable from memorisation. The response is measured at the
    # arm's strongest induction head, chosen on the pooled natural probes so that
    # no stratum picks its own head.
    pooled_prefix = pool_scores(
        representative_scores(scores, group_of)
    )["scores"]["prefix_matching"]
    top_layer, top_head = (
        int(value) for value in np.unravel_index(int(pooled_prefix.argmax()), pooled_prefix.shape)
    )
    covariates = covariate_analysis(
        representative_scores(scores, group_of),
        {a.record_index: a.max_identity_over_query for a in assignments},
        layer=top_layer,
        head=top_head,
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "induction",
        "no_text_arm": NO_TEXT_ARM,
        "arm": {
            "name": arm.name,
            "modality": arm.modality,
            "n_layer": arm.n_layer,
            "d_model": arm.d_model,
            "n_head": n_head(arm),
            "tokenisation": arm.spec.tokenisation,
            "input_format": arm.spec.input_format,
            "source": arm.spec.source,
            "dtype": arm.dtype,
            "device": arm.device,
            "attn_implementation": args.attn_implementation,
        },
        "cohort": {
            "name": cohort.name,
            "digest": cohort.digest,
            "n_records": len(cohort),
            "criterion": cohort.metadata["criterion"],
        },
        "strata": strata_record(),
        "stratum_record_counts": records_per_stratum,
        "stratum_distinct_sequence_counts": distinct_stratum_counts(assignments, groups),
        "probe_yield": {
            "n_records": len(cohort.records),
            "n_probes": len(scores),
            "records_without_probe": without_probe,
            "note": (
                "a record without a probe is one whose two repeat copies tokenise "
                "differently on this arm; it is excluded from every stratum and "
                "listed here rather than silently dropped"
            ),
        },
        "verification": {
            "per_probe_pooling_relative_error": pooling_error,
            "pooling_check_probes": min(args.pooling_check_probes, len(probes)),
            "pooling_tolerance": args.pooling_tolerance,
        },
        "copying_matrix": {
            "note": (
                "the OV copying matrices are a function of the weights and the "
                "sampled token support alone and therefore do NOT vary by stratum; "
                "what varies per stratum is which heads that stratum's "
                "prefix-matching selects"
            ),
            "matched_support_size": args.copy_matched_n,
            "matched_repeats": args.copy_matched_repeats,
            "diagonal_fraction": summarise_head_matrix(
                copying["diagonal_fraction"], "diagonal_fraction"
            ),
            "mean_normalised_rank": summarise_head_matrix(
                copying["mean_normalised_rank"], "mean_normalised_rank"
            ),
        },
        "headline_threshold": args.headline_threshold,
        "by_stratum": strata,
        "pooled_all_strata": pooled_all,
        "covariate_analysis": covariates,
        "per_probe": [
            {
                "record_index": score.record_index,
                "sequence_group": int(group_of[score.record_index]),
                "stratum": assignments[score.record_index].stratum,
                "max_identity_over_query": assignments[
                    score.record_index
                ].max_identity_over_query,
                "repeat_symbols": score.repeat_symbols,
                "scored_positions": score.scored_positions,
                "top_head_prefix_matching": float(
                    score.sums["prefix_matching"][top_layer, top_head]
                )
                / score.scored_positions,
            }
            for score in scores
        ],
        "synthetic_repeat_control": synthetic,
        "adjudication": adjudicate(strata, synthetic, min_probes=args.min_stratum_probes),
        "runtime_seconds": round(time.time() - started, 2),
        "peak_gpu_bytes": (
            int(torch.cuda.max_memory_allocated(arm.device)) if on_cuda else None
        ),
    }
    del arm
    if on_cuda:
        torch.cuda.empty_cache()
    return payload


# ------------------------------------------------------------------ summary


def panel_summary(
    search: dict[str, Any], arms: dict[str, dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    """One row per (arm, stratum): the table the report is written from."""

    table: dict[str, Any] = {}
    for name, payload in arms.items():
        rows: dict[str, Any] = {}
        for stratum in STRATUM_NAMES:
            entry = payload["by_stratum"][stratum]
            if not entry["measured"]:
                rows[stratum] = {
                    "measured": False,
                    "n_records": entry["n_records"],
                    "n_distinct_sequences": 0,
                }
                continue
            rows[stratum] = {
                "measured": True,
                "n_records": entry["n_records"],
                "n_probes_all_records": entry["n_probes_all_records"],
                "n_distinct_sequences": entry["n_distinct_sequences"],
                "peak_over_uniform": entry["peak_over_uniform"],
                "peak_over_uniform_ci": entry["bootstrap"]["peak_over_uniform_ci"],
                # No interval is ever published here without the number of
                # resampling units that produced it: a percentile interval over
                # four units is narrower than one over four hundred, and the
                # 2026-07-28 artefact put the two side by side unlabelled.
                "bootstrap_n_units": entry["bootstrap"]["n_units"],
                "bootstrap_degenerate": entry["bootstrap"]["degenerate"],
                "n_above_0.10": entry["census"]["count_above_threshold"]["0.10"],
                "fraction_above_0.10": entry["fraction_above_threshold"]["0.10"],
                "fraction_above_0.10_ci": entry["bootstrap"]["fraction_above_threshold_ci"],
                "ov_mean_normalised_rank_over_induction_heads": entry[
                    "ov_over_induction_heads"
                ]["mean_normalised_rank_mean"],
            }
        synthetic = payload["synthetic_repeat_control"]
        rows["synthetic_repeat_negative_control"] = {
            "measured": True,
            "n_probes": synthetic["n_probes"],
            "peak_over_uniform": synthetic["peak_over_uniform"],
            "peak_over_uniform_ci": synthetic["bootstrap"]["peak_over_uniform_ci"],
            "bootstrap_n_units": synthetic["bootstrap"]["n_units"],
            "bootstrap_degenerate": synthetic["bootstrap"]["degenerate"],
            "n_above_0.10": synthetic["census"]["count_above_threshold"]["0.10"],
            "fraction_above_0.10": synthetic["fraction_above_threshold"]["0.10"],
        }
        synthetic_heads = synthetic["induction_head_set"]
        rows["head_set_overlap_with_synthetic"] = {
            stratum: head_set_overlap(
                payload["by_stratum"][stratum]["induction_head_set"], synthetic_heads
            )
            for stratum in STRATUM_NAMES
            if payload["by_stratum"][stratum]["measured"]
        }
        pooled = payload["pooled_all_strata"]
        rows["pooled_all_strata"] = {
            "measured": True,
            "n_probes_all_records": pooled["n_probes_all_records"],
            "n_distinct_sequences": pooled["n_distinct_sequences"],
            "duplicate_weighted_peak_over_uniform": pooled["duplicate_weighted"][
                "peak_over_uniform"
            ],
            "duplicate_weighted_n_above_0.10": pooled["duplicate_weighted"]["census"][
                "count_above_threshold"
            ]["0.10"],
            "peak_over_uniform": pooled["peak_over_uniform"],
            "n_above_0.10": pooled["census"]["count_above_threshold"]["0.10"],
            "fraction_above_0.10": pooled["fraction_above_threshold"]["0.10"],
        }
        table[name] = {"rows": rows, "adjudication": payload["adjudication"]}

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "summary",
        "no_text_arm": NO_TEXT_ARM,
        "cohort": search["cohort"],
        "diamond": search["diamond"],
        "database": search["database"],
        "search_command": search["search"]["command"],
        "strata": search["strata"],
        "stratum_counts": search["stratum_counts"],
        "stratum_counts_distinct_sequences": search["stratum_counts_distinct_sequences"],
        "sequence_groups": search["sequence_groups"],
        "identity_summary": search["identity_summary"],
        "interpretation_fixed_in_advance": {
            "memorisation": (
                "induction concentrated in high-homology strata and weak at low "
                "homology -> the head-count finding is confounded by training-set "
                "overlap"
            ),
            "general_mechanism": (
                "induction stable across strata and present on synthetic repeats -> "
                "the finding survives"
            ),
            "neither": "intermediate or non-monotone -> reported as such, no verdict forced",
        },
        "arms": table,
        "configuration": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in sorted(vars(args).items())
        },
    }


def verify_outputs(directory: Path, names: Sequence[str]) -> None:
    """Fail loudly if an expected artifact is missing or is not this schema."""

    broken: list[str] = []
    expected = [directory / ASSIGNMENT_FILE, directory / "panel_summary.json"]
    expected += [directory / f"{name}.json" for name in names]
    for path in expected:
        if not path.is_file():
            broken.append(f"{path}: missing")
            continue
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != SCHEMA_VERSION:
            broken.append(f"{path}: unexpected schema_version")
    if broken:
        raise RuntimeError("output verification failed: " + "; ".join(broken))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    parser.add_argument("--arms", nargs="+", default=list(PROTEIN_ARMS), choices=sorted(PANEL))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260728)

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")

    parser.add_argument("--repeat-cohort-size", type=int, default=48)
    parser.add_argument("--repeat-min-len", type=int, default=200)
    parser.add_argument("--repeat-max-len", type=int, default=800)
    parser.add_argument("--repeat-criterion", default="exact", choices=sorted(CRITERIA))
    parser.add_argument(
        "--cohort-draw-seed",
        type=int,
        default=DEFAULT_CORPUS_DRAW_SEED,
        help="must match the draw the census being stratified was built under; "
        "0 selects the file-order prefix every census stored before EXP-R2-068 "
        "used. A mismatch is caught by the cohort digest comparison",
    )
    parser.add_argument("--cohort-workers", type=int, default=8)

    # Host-specific locations. These are environment-backed rather than
    # hardcoded because the same script runs on the local host and inside an
    # H200 pod, which mounts its corpora and scratch on GPFS. Defaults
    # reproduce local behaviour exactly when nothing is exported; see
    # src.transfer.arms.env_path.
    parser.add_argument(
        "--corpus-fasta",
        type=Path,
        default=env_path(
            "TRANSFER_UNIREF50_FASTA",
            REPO_ROOT / "data/uniref50/uniref50.fasta",
        ),
    )
    parser.add_argument(
        "--diamond-tarball",
        type=Path,
        default=env_path(
            "TRANSFER_DIAMOND_TARBALL",
            REPO_ROOT / "external_resources/tools/diamond-linux64-v2.1.24.tar.gz",
        ),
    )
    parser.add_argument(
        "--diamond-checksum",
        type=Path,
        default=env_path(
            "TRANSFER_DIAMOND_CHECKSUM",
            REPO_ROOT / "external_resources/tools/diamond-linux64-v2.1.24.tar.gz.sha256",
        ),
    )
    parser.add_argument(
        "--diamond-dir",
        type=Path,
        default=env_path("TRANSFER_DIAMOND_DIR", Path("/Data/lzp/tools/diamond-2.1.24")),
    )
    parser.add_argument(
        "--diamond-db",
        type=Path,
        default=env_path(
            "TRANSFER_DIAMOND_DB", Path("/Data/lzp/homology_db/uniref50_full.dmnd")
        ),
    )
    parser.add_argument(
        "--diamond-tmpdir",
        type=Path,
        default=env_path("TRANSFER_DIAMOND_TMPDIR", Path("/Data/lzp/homology_db/tmp")),
    )
    parser.add_argument("--rebuild-db", action="store_true")
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--sensitivity", default="very-sensitive")
    parser.add_argument("--evalue", type=float, default=1e-3)
    parser.add_argument("--max-target-seqs", type=int, default=100)

    parser.add_argument("--natural-max-tokens", type=int, default=840)
    parser.add_argument("--unigram-max-tokens", type=int, default=256)
    parser.add_argument("--synthetic-probes", type=int, default=16)
    parser.add_argument("--synthetic-copy-len", type=int, default=64)
    parser.add_argument("--copy-matched-n", type=int, default=20)
    parser.add_argument("--copy-matched-repeats", type=int, default=8)

    parser.add_argument("--headline-threshold", type=float, default=0.10)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--min-stratum-probes", type=int, default=3)
    parser.add_argument("--pooling-check-probes", type=int, default=6)
    parser.add_argument("--pooling-tolerance", type=float, default=1e-6)

    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("duplicate arms requested")
    if any(PANEL[name].modality != "protein" for name in args.arms):
        raise ValueError(
            "this control is protein-only by construction: " + NO_TEXT_ARM
        )
    if args.headline_threshold not in INDUCTION_THRESHOLDS:
        raise ValueError(
            f"--headline-threshold must be one of {list(INDUCTION_THRESHOLDS)} so that the "
            "stratum table and the census agree"
        )
    return args


def recorded_criterion(payload: dict[str, Any]) -> str | None:
    """The repeat criterion an assignment artefact was produced under.

    Read from the constructor string the artefact already records, so this adds
    no field and works on the artefacts EXP-R2-064 produced.
    """

    constructor = str(payload.get("cohort", {}).get("constructor", ""))
    for name in sorted(CRITERIA):
        if f"criterion=PROTEIN_{name.upper()}_CRITERION" in constructor:
            return name
    return None


def refuse_criterion_collision(args: argparse.Namespace) -> None:
    """An output directory holds one criterion's artefacts, or the run stops.

    Every filename this stage writes is fixed -- ``homology_assignment.json``,
    ``diamond_hits.tsv``, ``cohort_query.faa``, ``<arm>.json``,
    ``panel_summary.json`` -- and none carries the criterion. So an
    ``--repeat-criterion approximate`` run into a directory holding an ``exact``
    run silently replaced every one of them, and the replacement verified
    cleanly against its own schema check. The exact and approximate criteria are
    the paired comparison this stage exists to draw, and the pair is what the
    collision destroys.

    Refusing rather than renaming: the artefacts EXP-R2-064 wrote are cited in
    the audit under these names, and moving them would break those citations to
    fix a hazard that a refusal closes. Standing rule 11 -- never delete a result
    artefact -- points the same way.
    """

    existing = args.output_dir / ASSIGNMENT_FILE
    if not existing.is_file():
        return
    try:
        payload = json.loads(existing.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Not this guard's business. read_json refuses an artefact it cannot
        # parse or whose schema it does not know, with a message about that;
        # reporting it here as a criterion collision would name the wrong cause.
        return
    previous = recorded_criterion(payload)
    if previous is None or previous == args.repeat_criterion:
        return
    raise RuntimeError(
        f"{existing} was produced under --repeat-criterion {previous!r} and this "
        f"run requests {args.repeat_criterion!r}. Every filename this stage writes "
        "is fixed, so continuing would overwrite that run's assignment, hits, "
        "per-arm and panel artefacts with a different criterion's, and the result "
        "would pass its own schema check. Pass a separate --output-dir per "
        "criterion; the two are a paired comparison, not two versions of one run"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    refuse_criterion_collision(args)

    # Before the search, not after it: a CUDA request that no device can serve is
    # decidable from the arguments alone, and the search stage that follows is a
    # multi-process scan of the whole corpus. This check used to sit after both
    # that scan and the cohort rebuild.
    if _is_cuda(args.device) and not torch.cuda.is_available():
        raise RuntimeError(f"--device {args.device} requested but no CUDA device is available")

    if "search" in args.stages:
        search = run_search(args)
        write_json(args.output_dir / ASSIGNMENT_FILE, search)
        print(
            f"[search] {search['cohort']['n_records']} records, "
            f"strata {search['stratum_counts']}, "
            f"coverage {search['database']['coverage_fraction']:.6f}",
            flush=True,
        )
    else:
        search = read_json(args.output_dir / ASSIGNMENT_FILE)

    if "induction" not in args.stages:
        print("search stage only; no induction measurement requested", flush=True)
        return

    cohort = build_cohort(args)
    if cohort.digest != search["cohort"]["digest"]:
        raise RuntimeError(
            "cohort digest does not match the recorded search: the stratification on "
            f"disk was computed for {search['cohort']['digest']} but this run built "
            f"{cohort.digest}. Re-run --stages search."
        )
    assignments = [
        HomologyAssignment(
            record_index=entry["record_index"],
            query_id=entry["query_id"],
            query_length=entry["query_length"],
            n_hits=entry["n_hits"],
            max_identity_over_query=entry["max_identity_over_query"],
            max_pident=entry["max_pident"],
            best_subject=entry["best_subject"],
            best_bitscore=entry["best_bitscore"],
            best_qstart=entry["best_qstart"],
            best_qend=entry["best_qend"],
            best_hit_spans_repeat=entry["best_hit_spans_repeat"],
            # ``.get`` rather than ``[]``: a search artefact written before the
            # truncation detector existed carries neither key, and this stage must
            # still be able to read one in order to re-measure the old bins.
            best_hit_looks_truncated=entry.get("best_hit_looks_truncated"),
            hit_list_saturated=entry.get("hit_list_saturated"),
            stratum=entry["stratum"],
        )
        for entry in search["assignments"]
    ]
    if [a.record_index for a in assignments] != list(range(len(cohort.records))):
        raise RuntimeError("recorded assignments are not a complete, ordered cover of the cohort")

    torch.manual_seed(args.seed)

    results: dict[str, dict[str, Any]] = {}
    for name in args.arms:
        payload = run_arm(name, cohort, assignments, args)
        write_json(args.output_dir / f"{name}.json", payload)
        results[name] = payload
        peak = payload["peak_gpu_bytes"]
        print(
            f"[{name}] {payload['probe_yield']['n_probes']} probes, "
            f"verdict {payload['adjudication']['verdict']}, "
            f"{payload['runtime_seconds']}s, "
            + (f"peak {peak / 2**30:.1f} GiB" if peak is not None else f"on {args.device}"),
            flush=True,
        )

    write_json(args.output_dir / "panel_summary.json", panel_summary(search, results, args))
    verify_outputs(args.output_dir, sorted(results))
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
