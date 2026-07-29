#!/usr/bin/env python3
"""Qualify an evaluation cohort before anything scientific is measured on it.

The cohort is built once, frozen by content hash, and then scored arm by arm for
the information the model commits relative to its own context-free baseline. An
arm below the threshold is reported as unmeasurable *on this cohort*: that is a
property of the evaluation set, not evidence about the model or about any
interpretability method, and downstream analyses must exclude the arm rather
than report a negative result from it.
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# See 08_lens_family.py for why the stage directory is added explicitly.
_STAGE_DIR = str(Path(__file__).resolve().parent)
if _STAGE_DIR not in sys.path:
    sys.path.insert(0, _STAGE_DIR)

from panel_contract import CAMPAIGN_PANEL, arm_can_run, stage_contract_record  # noqa: E402
from src.transfer.io import write_json  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    DEFAULT_CORPUS_DRAW_SEED,
    PANEL,
    REPO,
    Cohort,
    load_arm,
    protein_cohort,
    text_cohort,
)
from src.transfer.budget import (  # noqa: E402
    DEFAULT_CONTEXT_LENGTHS,
    MIN_CONTEXT_INFORMATION_NATS,
    arm_power,
    markov_baselines,
    power_status,
    truncation_curve,
)
from src.transfer.pathways import (  # noqa: E402
    UNIGRAM_ESTIMATORS,
    disjoint_unigram_cross_entropy_nats,
    held_out_cohort,
    subsample_cohort,
)
from src.transfer.prediction_addressed import scored_target_counts  # noqa: E402

SCHEMA_VERSION = "r2_transfer_cohort_power_v1"
DEFAULT_OUT = REPO / "results/transfer/cohort_power"


def default_arms(kind: str, with_ec: bool) -> list[str]:
    """Every campaign arm of this modality that this stage can qualify.

    It used to be three hard-coded names -- ``["gpt2-large"]`` for text, and
    ProtGPT2 plus ProGen2-medium for protein -- which meant a bare
    ``01_cohort_power.py --kind text`` qualified a *one-arm* text side of an
    eleven-arm panel and said nothing about the other six. That is L18's shape
    exactly: the run is well formed, every number in it is right, and the panel
    it covers is not the panel a reader assumes. The selection is now derived,
    and whatever it excludes is written into the artefact by
    ``stage_contract_record``.

    ``with_ec`` still gates the EC-conditioned arms: without the EC-labelled
    corpus their conditioning tag does not exist, and ``validate_arms`` refuses
    the pair rather than scoring them unconditioned.
    """

    names = [
        name
        for name in CAMPAIGN_PANEL
        if PANEL[name].modality == kind and arm_can_run("cohort_power", name).can_run
    ]
    if kind != "text" and not with_ec:
        names = [name for name in names if PANEL[name].input_format != "ec_conditioned"]
    if not names:
        raise ValueError(
            f"no campaign arm of modality {kind!r} can enter cohort_power"
            + ("" if with_ec else "; the EC-conditioned arms need --with-ec")
        )
    return names


def draw_records(args: argparse.Namespace, size: int, skip: int, name: str) -> Cohort:
    """``size`` records from the corpus, starting ``skip`` records in.

    ``--cohort-draw-seed`` reaches the corpus constructors, so ``size`` records
    are a window of a seeded permutation of the **whole** corpus rather than a
    prefix of the file. That distinction is the open qualification on EXP-R2-060:
    the campaign's "seeded" draw was seeded *within a head-of-file pool of 4000
    records*, so the 0.16-0.60 nat cohort-block sensitivity it reported bounds
    within-pool selection uncertainty and not corpus-wide selection uncertainty,
    and the audit records that the true figure is plausibly larger. It also makes
    ``--cohort-skip`` mean what it claims: at one seed, two skips index disjoint
    windows of the same permutation, so a skip-offset sensitivity is a
    sensitivity rather than two overlapping prefixes.

    ``--cohort-draw-seed 0`` still reproduces the historical file-order draw, and
    the mode reaches every artefact through ``Cohort.sampling``.
    """

    seed = args.cohort_draw_seed or None
    if args.kind == "text":
        return text_cohort(size, min_chars=args.min_chars, skip=skip, name=name, seed=seed)
    return protein_cohort(
        size,
        args.res_min,
        args.res_max,
        skip=skip,
        name=name,
        with_ec=args.with_ec,
        seed=seed,
    )


def build_cohort(args: argparse.Namespace) -> tuple[Cohort, dict[str, Any]]:
    """The scored cohort and the record of how it was drawn.

    Taking the first ``n_seq`` eligible records in corpus file order has
    manufactured an effect three times in this programme, most recently
    ProGen2-medium's 0.099-nat context information, which moved by +1.01 nats
    simply by reading past the first 48 Swiss-Prot records. File order is not a
    random order: Swiss-Prot's leading block is atypical in length, organism and
    annotation depth. So a seeded permutation is the default here, and the
    file-order draw remains reachable only by explicitly asking for it with
    ``--cohort-draw-seed 0``, which is a declared choice rather than a default
    that nobody notices. ``--cohort-skip`` moves the draw's origin so that the
    same measurement can be repeated on a disjoint block of the corpus, which is
    the sensitivity this hazard is detected by.

    **The seed reaches the corpus, not only the pool.** Until 2026-07-30 the pool
    itself was a file-order prefix of ``--cohort-pool-size`` and only the
    ``n_seq`` draw from it was seeded, so a "seeded" cohort was a seeded sample of
    a head-of-file block. The pool is now a window of a permutation of the whole
    corpus (see :func:`draw_records`), which is what makes the recorded
    cohort-block sensitivity a corpus-wide quantity.
    """

    if args.cohort_draw_seed == 0:
        cohort = draw_records(args, args.n_seq, args.cohort_skip, args.cohort_name)
        return cohort, {
            "sampling": "corpus_file_order",
            "cohort_skip": int(args.cohort_skip),
            "records_consumed": int(args.cohort_skip + args.n_seq),
        }
    if args.cohort_pool_size < args.n_seq:
        raise ValueError(
            f"--cohort-pool-size {args.cohort_pool_size} is smaller than --n-seq {args.n_seq}"
        )
    pool = draw_records(
        args, args.cohort_pool_size, args.cohort_skip, f"{args.cohort_name}_pool"
    )
    drawn = subsample_cohort(pool, args.n_seq, args.cohort_draw_seed)
    # subsample_cohort names the result after the pool and the draw; the output
    # filenames and every downstream reference key off --cohort-name, so the
    # caller's name is restored and the draw is recorded in the provenance block
    # instead.
    cohort = Cohort(
        name=args.cohort_name,
        kind=drawn.kind,
        records=drawn.records,
        min_symbols=drawn.min_symbols,
        max_symbols=drawn.max_symbols,
        metadata=drawn.metadata,
    )
    return cohort, {
        "sampling": "seeded_permutation_over_pool",
        "cohort_skip": int(args.cohort_skip),
        "pool_size": int(args.cohort_pool_size),
        "draw_seed": int(args.cohort_draw_seed),
        "pool_digest": pool.digest,
        "records_consumed": int(args.cohort_skip + args.cohort_pool_size),
    }


def validate_arms(names: list[str], args: argparse.Namespace) -> None:
    """Refuse an arm/cohort combination before the corpus is read.

    Every one of these three checks reads only ``--arms``, ``--kind`` and
    ``--with-ec``, yet all three used to run *after* ``build_cohort`` had scanned
    the corpus for a 4000-record pool, a 4000-record unigram reference and a
    2000-record Markov block. Nothing about the diagnosis needs the cohort: the
    modality comes from ``ArmSpec.modality``, and whether the cohort will carry EC
    labels is decided by ``--with-ec`` alone.

    The EC check is stated against ``ArmSpec.input_format`` rather than the
    literal name ``"zymctrl"``: a second EC-conditioned arm would otherwise be
    scored on an unconditioned prompt, which is separately measured at 1.73 nats
    of conditioning leak (EXP-R2-034).
    """

    unknown = [name for name in names if name not in PANEL]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; panel is {sorted(PANEL)}")
    mismatched = [name for name in names if PANEL[name].modality != args.kind]
    if mismatched:
        raise ValueError(
            f"arms {mismatched} do not match a {args.kind!r} cohort; "
            "a cross-modality cohort is not a measurement"
        )
    refused = {
        name: verdict.reason
        for name in names
        if not (verdict := arm_can_run("cohort_power", name)).can_run
    }
    if refused:
        raise ValueError(f"01_cohort_power.py cannot qualify {sorted(refused)}: {refused}")
    conditioned = [name for name in names if PANEL[name].input_format == "ec_conditioned"]
    if conditioned and not args.with_ec:
        raise ValueError(
            f"arms {conditioned} are EC-conditioned and need an EC-labelled cohort; "
            "rebuild with --with-ec"
        )


def validate_truncation(args: argparse.Namespace) -> None:
    """Refuse a ``--max-len`` / ``--truncation-contexts`` pair before any model load.

    ``budget.truncation_curve`` requires ``max_len`` to exceed the longest
    requested context by at least two, and raises when it does not -- correctly,
    since a window that does not fit carries no truncation information. But it is
    called after ``load_arm`` and after the whole ``arm_power`` scoring pass, so
    a combination that was never going to work costs a checkpoint load and a full
    forward sweep first. Both values are command-line arguments.
    """

    if args.skip_truncation:
        return
    if not args.truncation_contexts:
        raise ValueError("--truncation-contexts is empty; pass --skip-truncation instead")
    longest = max(int(length) for length in args.truncation_contexts)
    if min(int(length) for length in args.truncation_contexts) < 1:
        raise ValueError("--truncation-contexts must all be at least one token")
    if args.max_len <= longest + 1:
        raise ValueError(
            f"--max-len {args.max_len} must exceed the longest --truncation-contexts "
            f"entry ({longest}) by at least two; raise --max-len, drop the long "
            "contexts, or pass --skip-truncation"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("protein", "text"), default="protein")
    parser.add_argument("--cohort-name", default=None)
    parser.add_argument("--n-seq", type=int, default=200)
    parser.add_argument("--res-min", type=int, default=64)
    parser.add_argument("--res-max", type=int, default=246)
    parser.add_argument("--min-chars", type=int, default=800, help="text cohort only")
    parser.add_argument("--with-ec", action="store_true", help="draw EC-labelled records")
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--threshold-nats", type=float, default=MIN_CONTEXT_INFORMATION_NATS
    )
    parser.add_argument("--markov-train", type=int, default=2000)
    parser.add_argument("--skip-markov", action="store_true")
    parser.add_argument("--skip-truncation", action="store_true")
    parser.add_argument(
        "--cohort-skip",
        type=int,
        default=0,
        help="records to pass over before the pool starts; use to repeat a "
        "measurement on a disjoint block and report a skip-offset sensitivity",
    )
    parser.add_argument(
        "--cohort-pool-size",
        type=int,
        default=4000,
        help="pool the seeded draw samples from; ignored with --cohort-draw-seed 0",
    )
    parser.add_argument(
        "--cohort-draw-seed",
        type=int,
        default=DEFAULT_CORPUS_DRAW_SEED,
        help="seed for the permutation that draws --n-seq records from the pool; "
        "0 selects the corpus file order instead, which is a declared choice "
        "and not a default (see build_cohort)",
    )
    parser.add_argument(
        "--unigram-estimator",
        default="disjoint",
        choices=list(UNIGRAM_ESTIMATORS),
        help="context-free baseline: 'disjoint' fits it on a held-out block of "
        "the same corpus, 'plugin' on the scored cohort itself. The plug-in is "
        "biased down by up to 1.29 nats on a 50k-vocabulary arm and 0.01 on a "
        "residue-level one, so it distorts exactly the cross-arm comparison "
        "this stage exists to support",
    )
    parser.add_argument("--unigram-reference-size", type=int, default=4000)
    parser.add_argument(
        "--truncation-contexts",
        type=int,
        nargs="*",
        default=list(DEFAULT_CONTEXT_LENGTHS),
        help=(
            "visible-context ladder in tokens; it is arm-relative because "
            "tokenizer expansion differs, and a multi-residue-BPE arm on a short "
            "cohort needs a shorter ladder than a residue-level arm"
        ),
    )
    parser.add_argument("--truncation-queries", type=int, default=4)
    parser.add_argument("--truncation-batch", type=int, default=64)
    parser.add_argument("--truncation-min-windows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.cohort_name is None:
        args.cohort_name = "swissprot" if args.kind == "protein" else "openwebtext"
    names = args.arms if args.arms else default_arms(args.kind, args.with_ec)

    # Everything that can be refused from the command line alone is refused here,
    # before the corpus is scanned and long before a checkpoint is loaded.
    validate_arms(names, args)
    validate_truncation(args)

    cohort, sampling = build_cohort(args)
    digest = cohort.digest

    # The held-out block starts after everything the scored draw consumed, and
    # is then deduplicated against the cohort by content: Swiss-Prot carries the
    # same sequence under several accessions, so a later file offset is not by
    # itself disjoint. There is no fallback to the plug-in -- asking for the
    # held-out estimator without a reference is a configuration error.
    reference: Cohort | None = None
    reference_overlap: dict[str, int] = {}
    if args.unigram_estimator == "disjoint":
        candidate = draw_records(
            args,
            args.unigram_reference_size,
            int(sampling["records_consumed"]),
            f"{args.cohort_name}_unigram_reference",
        )
        reference, reference_overlap = held_out_cohort(candidate, cohort)
    cohort_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "frozen_cohort",
        "cohort_digest": digest,
        "cohort_name": cohort.name,
        "cohort_kind": cohort.kind,
        "min_symbols": cohort.min_symbols,
        "max_symbols": cohort.max_symbols,
        "n_records": len(cohort),
        "records": cohort.records,
        "metadata": cohort.metadata,
    }
    write_json(args.out / f"cohort_{cohort.name}_{digest[:12]}.json", cohort_payload)

    cohort_level: dict[str, Any] = {}
    if cohort.kind == "protein" and not args.skip_markov:
        # Offset past both the scored draw and the unigram reference block, so
        # that the residue ladder, the context-free baseline and the scored
        # cohort are three disjoint slices of the corpus rather than two.
        markov_skip = int(sampling["records_consumed"]) + (
            args.unigram_reference_size if reference is not None else 0
        )
        held_out = protein_cohort(
            args.markov_train,
            args.res_min,
            args.res_max,
            skip=markov_skip,
            name=f"{cohort.name}_markov_train",
            with_ec=args.with_ec,
            seed=args.cohort_draw_seed or None,
        )
        # A record-count offset is not disjointness. Swiss-Prot carries the same
        # sequence under several accessions and the EC-labelled corpus carries
        # it under several EC tags, so a later block in file order can still
        # repeat the cohort's content; markov_cross_entropy_bits refuses such a
        # pair outright, which is what caught this. Deduplicating by content is
        # the same remedy the unigram reference uses, and the number removed is
        # recorded rather than absorbed.
        held_out, markov_overlap = held_out_cohort(held_out, cohort)
        cohort_level["residue_markov_baselines"] = markov_baselines(
            held_out.records, cohort.records
        )
        cohort_level["residue_markov_train_digest"] = held_out.digest
        cohort_level["residue_markov_train_skip"] = markov_skip
        cohort_level["residue_markov_train_overlap_removed"] = markov_overlap

    arm_reports: dict[str, Any] = {}
    for name in names:
        arm = load_arm(name, device=args.device, dtype=args.dtype)
        report = arm_power(
            arm,
            cohort,
            max_len=args.max_len,
            batch_size=args.batch_size,
            minimum_context_information_nats=args.threshold_nats,
        )
        report["unigram_estimator"] = args.unigram_estimator
        if reference is not None:
            # Both count vectors come from the tokenizer alone, over exactly the
            # multiset arm_power scores, so this costs no second forward pass.
            target_counts = scored_target_counts(
                arm, cohort.input_strings(arm), max_len=args.max_len
            )
            reference_counts = scored_target_counts(
                arm, reference.input_strings(arm), max_len=args.max_len
            )
            held_out_nats = disjoint_unigram_cross_entropy_nats(
                reference_counts, target_counts
            )
            context = held_out_nats - report["clean_ce_nats"]
            verdict, status = power_status(context, args.threshold_nats)
            report["unigram_entropy_held_out_nats"] = held_out_nats
            report["plug_in_bias_nats"] = (
                held_out_nats - report["unigram_entropy_on_cohort_nats"]
            )
            report["context_information_plug_in_nats"] = report["context_information_nats"]
            report["context_information_nats"] = context
            # Derived views of the same quantity, recomputed so that no field in
            # the record is still normalised against the estimator that was
            # replaced.
            report["context_information_bits_per_symbol"] = (
                context / math.log(2.0) / report["symbols_per_token"]
            )
            report["unigram_entropy_bits_per_symbol"] = (
                held_out_nats / math.log(2.0) / report["symbols_per_token"]
            )
            # Every per-sequence context-information value is baseline minus that
            # sequence's cross-entropy, so replacing the baseline shifts the
            # whole distribution by one constant: the mean and both interval
            # endpoints move by the bias, the standard error does not.
            bias = report["plug_in_bias_nats"]
            shifted = dict(report["per_sequence_context_information_interval"])
            shifted["mean"] = shifted["mean"] + bias
            shifted["interval"] = [value + bias for value in shifted["interval"]]
            report["per_sequence_context_information_interval"] = shifted
            report["power_verdict"] = verdict
            report["measurability"] = status
            report["reference_digest"] = reference.digest
            report["reference_sequences"] = len(reference)
        if not args.skip_truncation:
            report["truncation_curve"] = truncation_curve(
                arm,
                cohort.input_strings(arm),
                max_len=args.max_len,
                context_lengths=args.truncation_contexts,
                queries_per_sequence=args.truncation_queries,
                batch_size=args.truncation_batch,
                seed=args.seed,
                min_windows=args.truncation_min_windows,
            )
        arm_reports[name] = report
        baseline_nats = report.get(
            "unigram_entropy_held_out_nats", report["unigram_entropy_on_cohort_nats"]
        )
        print(
            f"{name:16s} unigram({args.unigram_estimator}) {baseline_nats:7.4f}  "
            f"clean_ce {report['clean_ce_nats']:7.4f}  "
            f"context_info {report['context_information_nats']:+8.4f} nats/token  "
            f"{report['power_verdict']} ({report['measurability']})"
        )
        del arm
        gc.collect()
        torch.cuda.empty_cache()

    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "cohort_power_report",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_digest": digest,
        "cohort": {
            "name": cohort.name,
            "kind": cohort.kind,
            "n_records": len(cohort),
            "min_symbols": cohort.min_symbols,
            "max_symbols": cohort.max_symbols,
            "with_ec_labels": bool(cohort.metadata.get("ec_labels")),
        },
        "arm_specs": {
            name: {
                "path": str(PANEL[name].path),
                "modality": PANEL[name].modality,
                "n_layer": PANEL[name].n_layer,
                "d_model": PANEL[name].d_model,
                "tokenisation": PANEL[name].tokenisation,
                "input_format": PANEL[name].input_format,
                "source": PANEL[name].source,
            }
            for name in names
        },
        "seeds": {
            "truncation_curve": int(args.seed),
            "cohort_draw": int(args.cohort_draw_seed),
        },
        "cohort_sampling": sampling,
        "stage_contract": stage_contract_record("cohort_power", names),
        "unigram_baseline": {
            "estimator": args.unigram_estimator,
            "reference_size_requested": int(args.unigram_reference_size),
            "reference_sequences_used": None if reference is None else len(reference),
            "reference_digest": None if reference is None else reference.digest,
            "reference_overlap_removed": reference_overlap,
            # The scored cohort is a seeded draw from a file-order pool; the
            # held-out reference block is the next contiguous block of the corpus
            # in file order, with no permutation of its own. That matters because
            # this block *is* the context-free baseline every context-information
            # figure is measured against, and Appendix B rule 1 applies to it as
            # much as to the scored draw. Carried here rather than in a document:
            # arms.protein_cohort/text_cohort record their own sampling mode and
            # hazard, and this surfaces it beside the number it produced.
            "reference_sampling": (
                None if reference is None else reference.metadata.get("sampling")
            ),
        },
        "truncation_context_lengths": sorted({int(c) for c in args.truncation_contexts}),
        "thresholds": {"minimum_context_information_nats": float(args.threshold_nats)},
        "settings": {
            "device": args.device,
            "dtype": args.dtype,
            "max_len": int(args.max_len),
            "batch_size": int(args.batch_size),
            "markov_train_sequences": int(args.markov_train),
            "truncation_queries_per_sequence": int(args.truncation_queries),
            "truncation_evaluated": not args.skip_truncation,
        },
        "cohort_level": cohort_level,
        "arms": arm_reports,
        "measurable_arms": sorted(
            name for name, report in arm_reports.items() if report["power_verdict"] == "PASS"
        ),
        "unmeasurable_arms": sorted(
            name for name, report in arm_reports.items() if report["power_verdict"] == "FAIL"
        ),
    }
    destination = args.out / f"power_{cohort.name}_{digest[:12]}.json"
    write_json(destination, payload)
    print(f"wrote {destination}")
    if cohort_level.get("residue_markov_baselines"):
        ladder = cohort_level["residue_markov_baselines"]["cross_entropy_bits_per_residue"]
        print("residue Markov (bits/residue): " + "  ".join(
            f"{key} {value:.4f}" for key, value in sorted(ladder.items())
        ))


if __name__ == "__main__":
    main()
