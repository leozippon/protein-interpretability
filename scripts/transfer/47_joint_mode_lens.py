#!/usr/bin/env python3
"""Both modes of ONE joint decoder under the logit lens, on one depth grid.

EXP-R2-229. The panel's modality coefficients rest on `protgpt2` alone because no
checkpoint family in it spans both modalities at more than one rung
(`docs/INTERPRETABILITY_TRANSFER_AUDIT.md` §2). Galactica spans both at four, and
`e500d14` taught `Arm.blocks` and `lenses.FINAL_LAYER_NORM_PATH` the `opt`
architecture behind it -- and refused four other architecture tables with
measured structural reasons. This stage is the Direction-2 measurement that
spends exactly what was granted: a logit-lens depth trajectory, and nothing that
needs a feed-forward submodule, an attention pattern or an embedding module.

The design, the estimand, the frozen constants and the binding ceiling live in
:mod:`src.transfer.joint_lens`; this file is the operational sequence. One
invocation loads one checkpoint once and measures four cells over those weights:

``text_declared``             the corpus record itself, over a
                              :data:`~src.transfer.joint_lens.TEXT_WINDOW_TOKENS`
                              window, scored at every non-padding target after
                              the first. This is the **positive control** every
                              gate is checked against before a protein reading is
                              taken (Appendix B rule 2), and it is unusually
                              clean here: it is the same weights.
``protein_declared``          the declared ``[START_AMINO]...[END_AMINO]``
                              rendering with the per-residue escape, scored on
                              exactly the span
                              `21_joint_mode_qualification.py` scores.
``protein_declared_capped``   the same forward pass read under a mask that keeps
                              only scored targets the text window can match on
                              position in context. Reported as a sensitivity;
                              a sign reversal here is a pre-declared branch.
``protein_naive``             the same block with the escape removed, which is
                              what an unaided ``AutoTokenizer`` produces. Prices
                              the rendering (Appendix B rule 4) in the estimand's
                              own units. Reported, never gated.

The checkpoint is named by ``--checkpoint`` and never by an arm name, for
`21_joint_mode_qualification.py`'s reason: a joint checkpoint that has not passed
that stage must not be in ``arms.py`` at all. Nothing here enters ``arms.PANEL``.

**A protein reading is refused unless the rung's protein mode is identified**, and
that verdict is *read* rather than restated: ``--identification`` names the rung's
own `41_context_information_bootstrap.py` report and the sign rule comes from it
-- the displacement-corrected lower bound above zero, EXP-R2-221's criterion. A
rung whose protein mode is not identified still produces its trajectory, because
the primary quantity is read against the model's **own** final prediction and is
therefore target-free and defined on a mode that reads nothing from context; what
it does not produce is a gate verdict.

Two stages, because the verdict is a CPU function of an artefact and a report
while the trajectory needs a card. ``--stage measure`` loads the checkpoint and
writes ``joint_mode_lens.json`` -- every trajectory, every depth statistic, every
contrast and the text control's attainability. ``--stage gate`` loads nothing,
reads that artefact beside the rung's stage-41 report, and writes
``joint_mode_lens_gate.json`` beside it. The split is stage 21's own argument for
its sufficient-statistics sidecar: a verdict that can be re-taken when its input
report is superseded should not need a second GPU sweep, and the accelerator host
need not carry the CPU analysis tree the report lives in.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import joint_lens as jl  # noqa: E402
from src.transfer import joint_modes  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    DEFAULT_CORPUS_DRAW_SEED,
    Cohort,
    load_arm_spec,
    protein_cohort,
    require_input_path,
    text_cohort,
)
from src.transfer.concept_lens import RESOLUTION_TAUS  # noqa: E402
from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.lenses import (  # noqa: E402
    DEFAULT_DEPTH_FRACTIONS,
    layer_grid,
    lens_cluster_bootstrap,
    lens_head,
    prepare_windows,
    verify_lens_head,
)

DEFAULT_OUT = REPO_ROOT / "results" / "transfer" / "joint_mode_lens"


# ------------------------------------------------------------- qualification


def identification_verdict(report: Path, mode_arm: str) -> dict[str, Any]:
    """One arm's identification status, read from its own stage-41 report.

    Read rather than restated. The criterion is EXP-R2-221's sign rule -- the
    displacement-corrected bootstrap lower bound above zero -- and
    `41_context_information_bootstrap.py` is where it is computed; a second
    spelling of it here would be a second declaration of the rule that decides
    whether a protein number may be read at all (Appendix B rule 12).
    """

    payload = json.loads(Path(report).read_text(encoding="utf-8"))
    rows = [row for row in payload["summary"]["arms"] if row["arm"] == mode_arm]
    if len(rows) != 1:
        raise ValueError(
            f"{report}: {len(rows)} rows name the arm {mode_arm!r}; exactly one is "
            f"required. Arms present: {sorted(row['arm'] for row in payload['summary']['arms'])}"
        )
    row = rows[0]
    blocks = row["blocks"]
    if len(blocks) != 1:
        raise ValueError(f"{report}: {mode_arm} carries {len(blocks)} blocks; one is required")
    block = blocks[0]
    # The stage-41 report names its arm ``protein_declared`` and does not name the
    # checkpoint, so which rung it belongs to cannot be read off a field. Its own
    # input paths do name the run that produced it, and they are published here
    # verbatim beside the artefact's checkpoint name so the pairing an operator
    # declared in the manifest is checkable in the record rather than only in the
    # command line that is gone.
    configuration = payload.get("metadata", {}).get("configuration", {})
    return {
        "report": str(report),
        "report_sha256": sha256_file(Path(report)),
        "report_inputs": {
            "sidecar": configuration.get("sidecar"),
            "cohort_json": configuration.get("cohort_json"),
            "reference_json": configuration.get("reference_json"),
            "out": configuration.get("out"),
            "pairing_note": (
                "this report does not name a checkpoint; the rung it belongs to is an "
                "operator declaration and these are the paths it can be checked against"
            ),
        },
        "arm": mode_arm,
        "context_information_nats": block["context_information_nats"],
        "bootstrap_ci_95": block["bootstrap_ci_95"],
        "cohort_digest": block["cohort_digest"],
        "sign_status": block["sign_status"],
        "screening_status": block["screening_status"],
        "identified": block["sign_status"] == "PASS",
        "criterion": (
            "EXP-R2-221's sign rule as 41_context_information_bootstrap.py applies "
            "it: the displacement-corrected near-duplicate-group bootstrap 95% lower "
            "bound on context information is above zero. Read from that report, not "
            "recomputed here"
        ),
    }


# ------------------------------------------------------------------ cohorts


def draw_cohorts(args: argparse.Namespace) -> dict[str, Cohort]:
    """The two frozen draws, and the digest check that they are the qualified ones."""

    protein = protein_cohort(
        args.sequences,
        args.protein_min_len,
        args.protein_max_len,
        name="protein_scored",
        seed=args.cohort_draw_seed,
    )
    text = text_cohort(
        args.sequences,
        args.text_min_chars,
        name="text_scored",
        seed=args.cohort_draw_seed,
    )
    return {"protein": protein, "text": text}


def cohort_record(cohort: Cohort, *, band_unit: str) -> dict[str, Any]:
    return {
        "name": cohort.name,
        "kind": cohort.kind,
        "n_records": len(cohort),
        "digest": cohort.digest,
        "provenance_digest": cohort.provenance_digest,
        "sampling_record": cohort.sampling,
        "band": [cohort.min_symbols, cohort.max_symbols],
        "band_unit": band_unit,
    }


# -------------------------------------------------------------------- cells


def measure_cell(
    arm,
    head,
    grid,
    windows,
    census: dict[str, Any],
    args: argparse.Namespace,
    *,
    label: str,
) -> dict[str, Any]:
    """One mode's trajectory, its per-layer intervals and its resolution depths."""

    layers = [point.layer for point in grid]
    depths = [point.relative_depth for point in grid]
    verification = verify_lens_head(
        arm, head, windows[0], tolerance_nats=args.lens_head_tolerance_nats
    )
    print(
        f"  [{label}] lens head max KL {verification['max_kl_nats']:.3e} nats over "
        f"{verification['positions']} positions",
        flush=True,
    )
    rows = jl.blocked_trajectory(
        arm,
        head,
        windows,
        layers,
        block_windows=args.block_windows,
        metric_chunk=args.metric_chunk,
        max_bytes=int(args.max_cache_gib * 2**30),
    )
    metrics = jl.layer_metrics(rows, layers)
    trajectory = jl.trajectory_record(grid, metrics)
    point = jl.depth_statistics(
        depths, metrics, layers, levels=jl.AGREEMENT_LEVELS, taus=RESOLUTION_TAUS
    )
    bootstrap = jl.depth_bootstrap(
        rows,
        depths,
        layers,
        levels=jl.AGREEMENT_LEVELS,
        taus=RESOLUTION_TAUS,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    print(
        f"  [{label}] agreement {trajectory[jl.AGREEMENT_QUANTITY][0]:.4f} -> "
        f"{trajectory[jl.AGREEMENT_QUANTITY][-1]:.4f} over {len(layers)} depths; "
        f"reaches 0.50 at relative depth {point[jl.agreement_key(0.50)]:.4f}; "
        f"KL {trajectory[jl.KL_QUANTITY][0]:.4f} -> {trajectory[jl.KL_QUANTITY][-1]:.4f}",
        flush=True,
    )
    return {
        "label": label,
        "cohort_census": census,
        "lens_head_verification": verification,
        "layers": [
            {
                "layer": layer,
                "relative_depth": depth,
                "per_token": metrics[layer],
                "cluster_bootstrap": lens_cluster_bootstrap(
                    rows[layer],
                    samples=args.layer_bootstrap_samples,
                    seed=args.bootstrap_seed + layer,
                ),
            }
            for layer, depth in zip(layers, depths)
        ],
        "trajectory": trajectory,
        "depth_statistics": point,
        "depth_bootstrap": {key: value for key, value in bootstrap.items() if key != "draws"},
        "_draws": bootstrap,
        "_point": point,
    }


def take_gate(args: argparse.Namespace) -> dict[str, Any]:
    """The verdict, from a measured artefact and the rung's own identification report.

    Loads no model. Every clause is decided from what
    :func:`src.transfer.joint_lens.mode_gate` is given, and the two conditions
    that can refuse it before that -- an unidentified protein mode and an
    unattainable positive control -- are checked here in that order, because a
    gate applied to a mode that reads nothing from context and a gate its own
    control cannot pass are different defects and must not be reported as one.
    """

    artefact_path = require_input_path(Path(args.artefact).resolve(), "--artefact")
    artefact = json.loads(artefact_path.read_text(encoding="utf-8"))
    if artefact.get("schema_version") != jl.SCHEMA_VERSION:
        raise ValueError(
            f"{artefact_path}: schema {artefact.get('schema_version')!r} is not "
            f"{jl.SCHEMA_VERSION!r}; this verdict is defined on that schema alone"
        )
    identification = identification_verdict(
        require_input_path(Path(args.identification).resolve(), "--identification"),
        jl.MODE_PROTEIN,
    )
    control = artefact["text_control"]
    contrast = artefact["contrasts"][jl.MODE_PROTEIN]
    protein_cohort_digest = artefact["cohorts"]["protein"]["digest"]

    if not identification["identified"]:
        gate = {
            "verdict": "refused",
            "clauses": None,
            "direction": None,
            "reason": (
                "this rung's protein mode is not identified on the qualification "
                f"cohort ({identification['context_information_nats']:.6f} nats, "
                f"interval {identification['bootstrap_ci_95']}, sign rule "
                f"{identification['sign_status']}), so its trajectory is reported and "
                "no capability verdict is taken from it"
            ),
        }
    elif not control["attainable"]:
        gate = {
            "verdict": "refused",
            "clauses": None,
            "direction": None,
            "reason": (
                "the text control's own statistics are not defined across the depth "
                "grid, so the gate is a specification defect rather than a "
                "measurement (Appendix B rule 2)"
            ),
        }
    else:
        gate = jl.mode_gate(contrast)

    return {
        "schema_version": jl.SCHEMA_VERSION,
        "stage": "gate",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": sha256_file(Path(__file__)),
        "joint_lens_module_sha256": sha256_file(REPO_ROOT / "src" / "transfer" / "joint_lens.py"),
        "artefact": {
            "path": str(artefact_path),
            "sha256": sha256_file(artefact_path),
            "checkpoint": artefact["checkpoint"]["name"],
            "n_layer": artefact["checkpoint"]["n_layer"],
            "d_model": artefact["checkpoint"]["d_model"],
            "n_parameters": artefact["checkpoint"]["n_parameters"],
            "dtype_observed": artefact["checkpoint"]["dtype_observed"],
        },
        "identification": identification,
        "cohort_agreement": {
            "artefact_protein_cohort_digest": protein_cohort_digest,
            "identification_cohort_digest": identification["cohort_digest"],
            "same_cohort": protein_cohort_digest == identification["cohort_digest"],
            "note": (
                "the lens and the identification must be read on the same records or "
                "the verdict qualifies a population the trajectory was not taken on "
                "(Appendix B rule 13). A mismatch does not refuse the gate here -- it "
                "is published so a reader cannot miss it -- because the two stages "
                "hash different objects: this stage hashes its own scored cohort and "
                "41_context_information_bootstrap.py records the cohort its sidecar "
                "carried"
            ),
        },
        "text_control": control,
        "contrast": contrast,
        "gate": gate,
        "verdict_vocabulary": list(jl.VERDICTS),
        "ceiling": jl.CEILING,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="measure", choices=("measure", "gate"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--artefact",
        type=Path,
        default=None,
        help="--stage gate only: the joint_mode_lens.json a --stage measure run wrote",
    )
    parser.add_argument("--rendering", default="galactica", choices=list(joint_modes.RENDERING_NAMES))
    parser.add_argument(
        "--identification",
        type=Path,
        default=None,
        help="this rung's own 41_context_information_bootstrap.py protein report. "
        "Without it no gate verdict is produced and the trajectories are reported "
        "as unqualified",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    # bfloat16, and declared rather than inherited. The 30B rung is 57 GB of
    # weights at bfloat16 against 143,771 MiB of card, so float32 is not
    # available on the rung the ladder exists for, and a ladder measured at two
    # precisions is not one ladder. The estimand is a depth normalised by the
    # trajectory's own span, so it is insensitive to a rounding floor several
    # orders below the crossing levels -- unlike the tuned lens and the Jacobian
    # spectrum, neither of which this stage computes. The lens-head tolerance
    # below is what turns that argument into a check.
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"])
    parser.add_argument("--sequences", type=int, default=jl.SEQUENCES)
    parser.add_argument("--protein-min-len", type=int, default=jl.PROTEIN_BAND[0])
    parser.add_argument("--protein-max-len", type=int, default=jl.PROTEIN_BAND[1])
    parser.add_argument("--text-min-chars", type=int, default=jl.TEXT_MIN_CHARS)
    parser.add_argument("--text-window-tokens", type=int, default=jl.TEXT_WINDOW_TOKENS)
    parser.add_argument("--position-cap", type=int, default=jl.POSITION_CAP)
    parser.add_argument("--cohort-draw-seed", type=int, default=DEFAULT_CORPUS_DRAW_SEED)
    parser.add_argument("--protein-context", default=None)
    parser.add_argument("--depths", nargs="+", type=float, default=list(DEFAULT_DEPTH_FRACTIONS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-windows", type=int, default=4)
    parser.add_argument("--metric-chunk", type=int, default=512)
    parser.add_argument("--max-cache-gib", type=float, default=8.0)
    # The float32 lens head cannot reproduce a bfloat16 forward pass to the 1e-3
    # stage 08 uses at float32, and that is the intended signal rather than a
    # reason to loosen a check silently: the residual gap IS the bfloat16
    # forward's own rounding. The tolerance is therefore set against the quantity
    # it guards. The shallowest crossing level of this estimand is
    # (1 - 0.75) x KL(shallowest), about 1.8 nats on every rung measured, so 1e-2
    # nats sits two orders of magnitude below the smallest number the trajectory
    # is read at and an order of magnitude above the measured rounding floor of
    # the positive control (EXP-R2-229's instrument gate: 7.878e-04 nats in text
    # mode and 1.409e-03 in protein mode on galactica-125m). It is a declared
    # ceiling on a known rounding term, not a threshold anything is decided at,
    # and the precision-invariance check the instrument gate records is what
    # actually establishes that bfloat16 does not move the estimand.
    parser.add_argument("--lens-head-tolerance-nats", type=float, default=1e-2)
    parser.add_argument("--bootstrap-resamples", type=int, default=jl.BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=jl.BOOTSTRAP_SEED)
    parser.add_argument("--layer-bootstrap-samples", type=int, default=1000)
    return parser


def validate(args: argparse.Namespace) -> None:
    if args.stage == "gate":
        missing = [
            name
            for name, value in (("--artefact", args.artefact), ("--identification", args.identification))
            if value is None
        ]
        if missing:
            raise ValueError(f"--stage gate needs {missing}")
        if args.checkpoint is not None:
            raise ValueError("--stage gate loads no checkpoint; drop --checkpoint")
        return
    if args.checkpoint is None:
        raise ValueError("--stage measure needs --checkpoint")
    if args.artefact is not None:
        raise ValueError("--artefact belongs to --stage gate")
    if args.sequences < 8:
        raise ValueError(
            "--sequences below the 8-record bootstrap unit floor cannot carry a "
            "percentile interval (src.transfer.statistics.MINIMUM_BOOTSTRAP_UNITS)"
        )
    if any(not 0.0 <= depth <= 1.0 for depth in args.depths):
        raise ValueError("--depths must lie in [0, 1]")
    if len(set(args.depths)) != len(args.depths):
        raise ValueError("--depths repeats a fraction")
    if len(args.depths) < 2:
        raise ValueError("a resolution depth needs at least two grid points")
    if args.text_window_tokens < 2:
        raise ValueError("--text-window-tokens must admit at least one scored target")
    if args.position_cap < 1:
        raise ValueError("--position-cap must be a positive token index")
    if args.block_windows < 1 or args.batch_size < 1:
        raise ValueError("--block-windows and --batch-size must be positive")
    if args.lens_head_tolerance_nats <= 0.0:
        raise ValueError("--lens-head-tolerance-nats must be positive")


def main() -> None:
    args = build_parser().parse_args()
    validate(args)
    out = Path(args.out).resolve()
    if args.stage == "gate":
        report = take_gate(args)
        write_json(out / "joint_mode_lens_gate.json", report)
        print(
            f"wrote {out / 'joint_mode_lens_gate.json'}: gate {report['gate']['verdict']}"
            + (f" ({report['gate']['direction']})" if report["gate"]["direction"] else ""),
            flush=True,
        )
        return
    started = datetime.now(timezone.utc).isoformat()

    checkpoint = require_input_path(args.checkpoint.resolve(), "--checkpoint")
    declaration = joint_modes.rendering(args.rendering)
    # The tokenizer alone first, so a checkpoint/family mismatch is refused
    # before a multi-gigabyte load -- 21_joint_mode_qualification.py's own order.
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    tokenisation = joint_modes.resolve(tokenizer, declaration)
    config = AutoConfig.from_pretrained(str(checkpoint))
    architecture = str(getattr(config, "model_type", "undeclared"))
    name = checkpoint.name

    identification = (
        None
        if args.identification is None
        else identification_verdict(args.identification, "protein_declared")
    )

    cohorts = draw_cohorts(args)
    protein_spec = jl.joint_arm_spec(
        checkpoint, name=name, mode="protein", config=config, architecture=architecture
    )
    text_spec = jl.joint_arm_spec(
        checkpoint, name=name, mode="text", config=config, architecture=architecture
    )

    print(f"[{name}] loading {architecture} at {args.dtype}", flush=True)
    protein_arm = load_arm_spec(
        protein_spec, device=args.device, dtype=args.dtype, strict=True
    )
    text_arm = jl.mode_arm(protein_arm, text_spec)
    torch.cuda.reset_peak_memory_stats(protein_arm.device)
    head = lens_head(protein_arm)
    grid = layer_grid(protein_arm.n_layer, args.depths)
    print(
        f"[{name}] {protein_arm.n_layer}L x {protein_arm.d_model}d, "
        f"{len(grid)} grid points, vocab {head.vocab_size}",
        flush=True,
    )

    text_windows = prepare_windows(
        text_arm,
        cohorts["text"],
        max_len=args.text_window_tokens,
        batch_size=args.batch_size,
    )
    declared = jl.protein_windows(
        protein_arm,
        tokenisation,
        cohorts["protein"],
        protein_context=args.protein_context,
        variant=joint_modes.DECLARED,
        batch_size=args.batch_size,
    )
    capped = jl.protein_windows(
        protein_arm,
        tokenisation,
        cohorts["protein"],
        protein_context=args.protein_context,
        variant=joint_modes.DECLARED,
        batch_size=args.batch_size,
        position_cap=args.position_cap,
    )
    naive = jl.protein_windows(
        protein_arm,
        tokenisation,
        cohorts["protein"],
        protein_context=args.protein_context,
        variant=joint_modes.NAIVE,
        batch_size=args.batch_size,
    )

    plan = (
        (jl.MODE_TEXT, text_arm, text_windows, jl.text_window_census(text_windows)),
        (jl.MODE_PROTEIN, protein_arm, declared.windows, declared.census),
        (jl.MODE_POSITION_CAPPED, protein_arm, capped.windows, capped.census),
        (jl.MODE_PROTEIN_NAIVE, protein_arm, naive.windows, naive.census),
    )
    cells: dict[str, dict[str, Any]] = {}
    for label, arm, windows, census in plan:
        print(f"[{name}] {label}: {census['n_scored_positions']} scored positions", flush=True)
        cells[label] = measure_cell(arm, head, grid, windows, census, args, label=label)
        gc.collect()
        torch.cuda.empty_cache()

    contrasts: dict[str, Any] = {}
    for label in (jl.MODE_PROTEIN, jl.MODE_POSITION_CAPPED, jl.MODE_PROTEIN_NAIVE):
        contrasts[label] = jl.depth_contrast(
            cells[label]["_draws"],
            cells[jl.MODE_TEXT]["_draws"],
            point_protein=cells[label]["_point"],
            point_text=cells[jl.MODE_TEXT]["_point"],
        )

    text_trajectory = cells[jl.MODE_TEXT]["trajectory"]
    text_attainable = {
        "agreement_reaches_every_level": all(
            text_trajectory["agreement_reaches_every_level"].values()
        ),
        "kl_falls_across_the_grid": bool(
            text_trajectory["falls_across_the_grid"][jl.KL_QUANTITY]
        ),
    }
    control = {
        "rule": (
            "Appendix B rule 2: a gate is checked on the positive control before it "
            "is applied to a protein reading. The control here is the SAME weights "
            "in text mode, which is what a joint checkpoint uniquely provides"
        ),
        "clauses": text_attainable,
        "text_depth_statistics_defined": {
            key: cells[jl.MODE_TEXT]["depth_statistics"][key] is not None
            for key in (*jl.PRIMARY_KEYS, jl.SECOND_FUNCTIONAL_KEY)
        },
        "text_agreement_at_shallowest_grid_point": text_trajectory[
            "agreement_at_shallowest_grid_point"
        ],
        "chance_agreement_note": (
            "a lens predicting at random would agree with the final top-1 about "
            "1/20 of the time in protein mode and about 1/50000 in text mode, so the "
            "protein curve carries a floor of at most 0.05. Every level of "
            "AGREEMENT_LEVELS is above it, and the floor makes the protein mode reach "
            "a level EARLIER than it otherwise would -- it biases against a finding "
            "that the protein mode resolves deeper and cannot manufacture one"
        ),
        "text_lens_head_max_kl_nats": cells[jl.MODE_TEXT]["lens_head_verification"][
            "max_kl_nats"
        ],
        "attainable": all(text_attainable.values()),
    }

    if identification is None:
        gate = {
            "verdict": "refused",
            "clauses": None,
            "direction": None,
            "reason": (
                "no --identification report was supplied, so this rung's protein mode "
                "has no identification verdict to read and no protein gate may be "
                "taken here. Run --stage gate on this artefact beside that report"
            ),
        }
    elif not identification["identified"]:
        gate = {
            "verdict": "refused",
            "clauses": None,
            "reason": (
                "this rung's protein mode is not identified on the qualification "
                f"cohort ({identification['context_information_nats']:.6f} nats, "
                f"interval {identification['bootstrap_ci_95']}, sign rule "
                f"{identification['sign_status']}), so its trajectory is reported and "
                "no capability verdict is taken from it"
            ),
        }
    elif not control["attainable"]:
        gate = {
            "verdict": "refused",
            "clauses": None,
            "reason": (
                "the text control's own quantities do not fall across the depth grid, "
                "so the resolution depth is undefined on the positive control and the "
                "gate is a specification defect rather than a measurement (Appendix B "
                "rule 2)"
            ),
        }
    else:
        gate = jl.mode_gate(contrasts[jl.MODE_PROTEIN])

    for cell in cells.values():
        cell.pop("_draws")
        cell.pop("_point")

    write_json(
        out / "joint_mode_lens.json",
        {
            "schema_version": jl.SCHEMA_VERSION,
            "stage": "measure",
            "started_at_utc": started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "runner_sha256": sha256_file(Path(__file__)),
            "joint_lens_module_sha256": sha256_file(REPO_ROOT / "src" / "transfer" / "joint_lens.py"),
            "lenses_module_sha256": sha256_file(REPO_ROOT / "src" / "transfer" / "lenses.py"),
            "joint_modes_module_sha256": sha256_file(
                REPO_ROOT / "src" / "transfer" / "joint_modes.py"
            ),
            "checkpoint": {
                "name": name,
                "resolved_path": str(checkpoint),
                "model_type": architecture,
                "architectures": list(getattr(config, "architectures", []) or []),
                "n_layer": protein_arm.n_layer,
                "d_model": protein_arm.d_model,
                "vocab_size": int(head.vocab_size),
                "n_parameters": int(sum(p.numel() for p in protein_arm.model.parameters())),
                "dtype_requested": args.dtype,
                "dtype_observed": protein_arm.dtype,
                "attn_implementation": protein_arm.attn_implementation,
                "strict_load": protein_arm.strict_load,
                "tokenizer_class": type(tokenizer).__name__,
                "tokenizer_vocab_size": int(len(tokenizer)),
                "shape_check_note": (
                    "the declared shape was read from this checkpoint's own config, so "
                    "load_arm_spec's comparison against that config is a tautology here "
                    "and asserts nothing; for a panel member the declaration is written "
                    "by hand and the comparison is a real check"
                ),
            },
            "declaration": {
                "panel_membership": (
                    "none. This checkpoint is reached by path and is not in "
                    "src.transfer.arms.PANEL or STAGED_ARMS; "
                    "21_joint_mode_qualification.py's rule that an unqualified joint "
                    "checkpoint must not be in arms.py at all is what keeps it out, and "
                    "src.transfer.joint_lens.joint_arm_spec builds the per-run "
                    "declaration load_arm_spec takes"
                ),
                "protein": {
                    "input_format": protein_spec.input_format,
                    "input_format_note": (
                        "the undeclared-rendering sentinel, because no branch of "
                        "arms.Cohort.input_strings emits this format -- not because the "
                        "rendering is unknown. It is declared in src.transfer.joint_modes "
                        "and reached through joint_lens.protein_windows"
                    ),
                    "capabilities": sorted(protein_spec.capabilities),
                },
                "text": {
                    "input_format": text_spec.input_format,
                    "capabilities": sorted(text_spec.capabilities),
                },
                "architecture_support": (
                    "e500d14 declared 'opt' in arms.Arm.blocks and "
                    "lenses.FINAL_LAYER_NORM_PATH only. This stage computes no pathway "
                    "share, no circuit statistic, no attention pattern and no "
                    "prediction-addressed census, because an OPTDecoderLayer has no "
                    "feed-forward submodule and an OPTDecoder builds its initial "
                    "residual inline"
                ),
            },
            "rendering": tokenisation.facts(),
            "cohorts": {
                "protein": cohort_record(cohorts["protein"], band_unit="residues"),
                "text": cohort_record(cohorts["text"], band_unit="characters"),
                "note": (
                    "the draw 21_joint_mode_qualification.py qualified every Galactica "
                    "rung under, at the same seed, count and band, so the lens is read "
                    "on the identical records (Appendix B rule 13). The text SCORED "
                    "WINDOW is 164 tokens rather than that stage's 512, chosen before "
                    "any reading so the two modes match on scored positions per record "
                    "and therefore on position in context"
                ),
            },
            "identification": identification,
            "text_control": control,
            "configuration": {
                "device": args.device,
                "dtype": args.dtype,
                "sequences": args.sequences,
                "protein_band": [args.protein_min_len, args.protein_max_len],
                "text_min_chars": args.text_min_chars,
                "text_window_tokens": args.text_window_tokens,
                "position_cap": args.position_cap,
                "protein_context": args.protein_context,
                "cohort_draw_seed": args.cohort_draw_seed,
                "depth_fractions": list(args.depths),
                "batch_size": args.batch_size,
                "block_windows": args.block_windows,
                "metric_chunk": args.metric_chunk,
            },
            "thresholds": {
                "lens_head_tolerance_nats": args.lens_head_tolerance_nats,
                "agreement_levels": list(jl.AGREEMENT_LEVELS),
                "span_taus": list(RESOLUTION_TAUS),
                "span_tau_provenance": "src.transfer.concept_lens.RESOLUTION_TAUS",
                "primary_statistics": list(jl.PRIMARY_KEYS),
                "second_functional": jl.SECOND_FUNCTIONAL_KEY,
                "max_undefined_draw_fraction": jl.MAX_UNDEFINED_DRAW_FRACTION,
            },
            "bootstrap": {
                "cluster_unit": "record",
                "resamples": args.bootstrap_resamples,
                "seed": args.bootstrap_seed,
                "layer_bootstrap_samples": args.layer_bootstrap_samples,
                "note": (
                    "the resampled record set is drawn once per draw and reused at "
                    "every grid layer, so each draw carries one trajectory; the two "
                    "modes are scored on two corpora and are resampled independently, "
                    "and the contrast is assembled draw by draw"
                ),
            },
            "layer_grid": jl.grid_record(grid, protein_arm.n_layer),
            "cells": cells,
            "contrasts": contrasts,
            "gate": gate,
            "verdict_vocabulary": list(jl.VERDICTS),
            "ceiling": jl.CEILING,
            "resources": {
                "peak_accelerator_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(protein_arm.device)
                ),
                "peak_accelerator_memory_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(protein_arm.device)
                ),
            },
        },
    )
    print(f"wrote {out / 'joint_mode_lens.json'}: gate {gate['verdict']}", flush=True)


if __name__ == "__main__":
    main()
