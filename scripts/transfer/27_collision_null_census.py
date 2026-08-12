#!/usr/bin/env python3
"""Score the induction census against each arm's own collision null.

The panel-level induction statement -- worst text arm above best protein arm --
was placed at risk by EXP-R2-155 and may not be quoted as a modality contrast
until every arm is read against a null that holds its vocabulary collision rate
fixed.  This stage is that null.  For each arm it builds the same synthetic
prefix-matching probes the published census uses, pairs every probe with two
seeded permutations of its own content positions, and counts the heads whose
excess over the first null exceeds what the difference between the two nulls
produces by chance on that arm.

Two things separate this from the census it corrects.  The cut is per arm and
derived from that arm's own null rather than fixed at 0.10, which is what makes
the count comparable across alphabets; and the byte-level text arms are admitted,
which is what makes the reading identify anything.  Their collision rate sits
among the residue-tokenised protein arms while their modality is text, so if head
counts track alphabet they pattern with protein, and if they track modality they
pattern with text.  That is the same separation EXP-R2-129 ran for the
prediction-addressed census, on the statistic that carries §4.

Only the synthetic probe is measured.  It is ``induction_robustness.PRIMARY_PROBE``
and the probe the seven-arm comparison was read on, and it is the only probe the
byte-level arms can enter at all: ``natural_repeat_probes`` needs a fast tokenizer
to align a repeat in symbol space and ByGPT5's is not one.  A natural-repeat
collision null is a separate measurement and is not smuggled in here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STAGE_DIR = Path(__file__).resolve().parent
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

from panel_contract import stage_arms, stage_contract_record  # noqa: E402
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
    conditioned_token_budget,
    fit_unigram,
    n_head,
    synthetic_repeat_probes,
)
from src.transfer.collision_null import (  # noqa: E402
    DEFAULT_ALPHAS,
    SCHEMA_VERSION,
    census_row,
    collision_null_census,
)
from src.transfer.io import write_json  # noqa: E402

STAGE = "collision_null_census"
DEFAULT_OUTPUT = REPO_ROOT / "results/transfer/collision_null_census"

#: The level the panel reading is taken at, with the rest of DEFAULT_ALPHAS
#: reported beside it as the invariance check.
HEADLINE_ALPHA = 0.95


def verify_outputs(directory: Path, names: list[str]) -> None:
    """Fail loudly if an expected artefact is missing or is not this schema."""

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
    return {
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


def build_cohort(modality: str, args: argparse.Namespace) -> Cohort:
    """The one cohort this stage needs: the unigram every probe is drawn from.

    Drawn on the same band and under the same seeded permutation as
    ``04_circuit_primitives.py``'s analysis cohort, so the unigram a probe is
    built from is the unigram the published census's probes were built from and
    the two counts are comparable.  Built before any checkpoint is loaded.

    The seed is passed through unchanged rather than through the ``or None``
    idiom the older stages use.  That idiom is their declared escape hatch to the
    historical file-order draw -- ``--cohort-draw-seed 0`` collapses to ``None``,
    which ``protein_cohort`` reads as "take the eligible records in file order" --
    and it is right for a stage with frozen artefacts produced that way.  This
    stage has none, so the hatch would buy nothing and would leave a single
    mistyped zero producing the family-grouped head-of-file block that has
    manufactured an effect three times in this programme.  Zero is a seed here
    like any other and the file-order draw is simply unreachable.
    """

    draw_seed = args.cohort_draw_seed
    if modality == "text":
        return text_cohort(
            args.cohort_size,
            min_chars=args.text_min_chars,
            skip=args.cohort_skip,
            seed=draw_seed,
        )
    if modality == "protein":
        return protein_cohort(
            args.cohort_size,
            args.protein_min_len,
            args.protein_max_len,
            with_ec=True,
            name="swissprot_ec_long",
            skip=args.cohort_skip,
            seed=draw_seed,
        )
    raise ValueError(f"unsupported modality {modality!r}")


def arm_record(arm: Arm, args: argparse.Namespace) -> dict[str, Any]:
    spec = arm.spec
    return {
        "name": spec.name,
        "architecture": spec.architecture,
        "modality": spec.modality,
        "n_layer": spec.n_layer,
        "d_model": spec.d_model,
        "n_head": n_head(arm),
        "tokenisation": spec.tokenisation,
        "input_format": spec.input_format,
        "source": spec.source,
        "path": str(spec.path),
        "dtype": arm.dtype,
        "device": arm.device,
        "attn_implementation": arm.attn_implementation,
        "matched_pair_member": spec.name in MATCHED_PAIR,
        "requested_attn_implementation": args.attn_implementation,
    }


def run_arm(name: str, args: argparse.Namespace, cohorts: dict[str, Cohort]) -> dict[str, Any]:
    started = time.time()
    arm = load_arm(
        name,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    if arm.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(arm.device)
    cohort = cohorts[arm.modality]
    strings = cohort.input_strings(arm)
    unigram_max_tokens = conditioned_token_budget(
        arm, args.unigram_max_tokens, args.protein_max_len
    )
    unigram = fit_unigram(arm, strings, max_tokens=unigram_max_tokens)

    ec_label = None
    if arm.spec.input_format == "ec_conditioned":
        labels = cohort.metadata.get("ec_labels")
        if not labels:
            raise ValueError(f"{arm.name}: cohort carries no EC labels")
        ec_label = labels[0]

    probes = synthetic_repeat_probes(
        arm,
        unigram,
        n_probes=args.probes,
        copy_len=args.copy_len,
        seed=args.seed,
        ec_label=ec_label,
    )
    census = collision_null_census(
        arm,
        probes,
        seed=args.seed,
        batch_size=args.probe_batch_size,
        n_bootstrap=args.bootstrap,
        alphas=tuple(args.alpha),
        thresholds=INDUCTION_THRESHOLDS,
        ec_label=ec_label,
    )

    payload: dict[str, Any] = {
        **census,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "arm": arm_record(arm, args),
        "cohort": cohort_record(cohort),
        "probe": {
            "kind": census["probe_kind"],
            "n_probes": args.probes,
            "copy_len_tokens": args.copy_len,
            "seed": args.seed,
        },
        "unigram_max_tokens": {
            "requested": int(args.unigram_max_tokens),
            "resolved_for_this_arm": int(unigram_max_tokens),
            "widened": unigram_max_tokens != args.unigram_max_tokens,
        },
        "tokenisation": {
            "symbols_per_token": symbols_per_token(arm, strings, unigram_max_tokens),
            "unigram": unigram.summary(),
        },
        "headline": census_row(census, alpha=HEADLINE_ALPHA),
        "runtime_seconds": round(time.time() - started, 2),
        "peak_gpu_bytes": (
            int(torch.cuda.max_memory_allocated(arm.device))
            if arm.device.startswith("cuda")
            else 0
        ),
    }
    del arm
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return payload


def panel_summary(
    results: dict[str, dict[str, Any]],
    refused: list[Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """One row per arm at every level, with the ordering left to the reader.

    No verdict is computed here.  The decision rule for this experiment is
    pre-declared in ``docs/EXPERIMENT_LOG.md`` and keys on the grouping of the
    byte-level text arms against the BPE text arms and the residue protein arms;
    computing it in the code that produced the numbers would let a later edit
    move the rule, which is exactly the failure the pre-declaration exists to
    prevent.
    """

    rows: dict[str, Any] = {}
    for name, payload in results.items():
        arm = payload["arm"]
        rows[name] = {
            "modality": arm["modality"],
            "tokenisation": arm["tokenisation"],
            "architecture": arm["architecture"],
            "n_layer": arm["n_layer"],
            "n_head_per_layer": arm["n_head"],
            "levels": {
                label: census_row(payload, alpha=float(label))
                for label in (f"{float(alpha):.2f}" for alpha in args.alpha)
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "headline_family_wise_level": HEADLINE_ALPHA,
        "levels_swept": [float(alpha) for alpha in args.alpha],
        "matched_pair": list(MATCHED_PAIR),
        "arms": rows,
        "refused_arms": [
            {"arm": item.arm, "reason": item.reason} for item in refused
        ],
        "configuration": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
            if key != "out"
        },
        # The stage names itself as a literal rather than through STAGE, so that
        # a reader (and tests/test_h200_orchestration.py) can see from the source
        # which contract this artefact declares.
        "stage_contract": stage_contract_record("collision_null_census", sorted(rows)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", required=True, choices=sorted(PANEL))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--cohort-size", type=int, default=24)
    parser.add_argument("--cohort-skip", type=int, default=0)
    parser.add_argument("--cohort-draw-seed", type=int, default=DEFAULT_CORPUS_DRAW_SEED)
    parser.add_argument("--protein-min-len", type=int, default=600)
    parser.add_argument("--protein-max-len", type=int, default=1000)
    parser.add_argument("--text-min-chars", type=int, default=3000)
    parser.add_argument("--unigram-max-tokens", type=int, default=256)
    parser.add_argument("--probes", type=int, default=256)
    parser.add_argument("--copy-len", type=int, default=64)
    parser.add_argument("--probe-batch-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--alpha", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        parser.error("--arms contains a duplicate")
    if f"{HEADLINE_ALPHA:.2f}" not in {f"{float(a):.2f}" for a in args.alpha}:
        parser.error(f"--alpha must include the headline level {HEADLINE_ALPHA}")
    return args


def main() -> None:
    args = parse_args()
    eligible, refused = stage_arms(STAGE, args.arms)
    for item in refused:
        print(f"[skip] {item.arm}: {item.reason}", flush=True)
    if not eligible:
        raise SystemExit(f"no requested arm is eligible for {STAGE}")

    print(f"[paths] models resolved from {PANEL[eligible[0]].path.parent}", flush=True)
    modalities = {PANEL[name].modality for name in eligible}
    cohorts = {modality: build_cohort(modality, args) for modality in sorted(modalities)}
    for modality, cohort in cohorts.items():
        print(
            f"[cohort] {modality}: {cohort.name} n={len(cohort)} digest={cohort.digest} "
            f"sampling={cohort.sampling}",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for name in eligible:
        print(f"[arm] {name}", flush=True)
        results[name] = run_arm(name, args, cohorts)
        write_json(args.out / f"{name}.json", results[name])
        head = results[name]["headline"]
        print(
            f"[done] {name} n_above_null={head['n_above_null']} of {head['n_heads']} "
            f"cut={head['null_cut']:.5f} fixed0.10={head['n_above_fixed_0.10']}",
            flush=True,
        )
    # Re-emitted at the end: the results root is shared with other tracks that
    # recreate it, so a per-arm write made earlier in this run may no longer be
    # on disk when the run finishes.
    for name, payload in results.items():
        write_json(args.out / f"{name}.json", payload)
    write_json(args.out / "panel_summary.json", panel_summary(results, refused, args))
    verify_outputs(args.out, sorted(results))


if __name__ == "__main__":
    main()
