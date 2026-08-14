#!/usr/bin/env python3
"""How many dimensions does the activation cloud a dictionary is fitted on occupy?

**The question this stage exists to answer, and what it gates.** R2.4's
same-input feature diff is blocked at basis-adequacy gate B2 -- mean live
dictionary latents per layer must exceed ``d_model`` = 4,096 -- which
EXP-R2-191's four dictionaries clear in text mode (7,608 and 4,900) and fail in
protein mode (2,188 and 1,634). Every reading of that failure on disk treats it
as a training-recipe problem: a revival budget, a sparsity, a width or a token
budget that a retraining sweep could move. That reading has never been tested
against its alternative, and the alternative is not exotic. B2's threshold is
``d_model``, which is the dimension of the space the activations are *written
in*; if the activations at that site occupy a subspace of far lower dimension,
then B2 asks a dictionary for more live latents than the data has directions to
put them in, and the gate measures data geometry rather than dictionary
adequacy. The two readings predict opposite things and are separated by one
measurement nothing on disk has: the rank of the cloud itself.

The alternative is not a bare possibility either. In the same four cells a ridge
map aligns the two checkpoints' *protein* activations far better than their text
activations -- cross-checkpoint true-pairing residual 0.400 against 0.684, both
against shuffled-pairing nulls near 1.00 (EXP-R2-175). More linearly alignable
**and** less able to support live latents is exactly the joint signature of a
low-dimensional cloud, and neither observation on its own distinguishes it from a
recipe failure.

**What is measured.** At the site the dictionaries were fitted at -- the
per-layer feed-forward module's output before the residual add, and its input,
every layer -- the centred covariance of the activation vectors over sampled
token positions, accumulated in float64, and the eigenvalue spectrum of that
covariance summarised by :mod:`src.transfer.spectrum`: ``r95``, ``r99``,
``r999``, the participation ratio and the effective rank, each reported beside
the sample-rank ceiling ``min(N, d_model)`` that bounds them.

**Where the positions come from, and why it is not a fresh draw.** The corpus,
the corpus seed, the block-shuffled stream, the held-out offset and the
near-duplicate screen are ``17_train_transcoder.py``'s own, imported rather than
restated, and driven with that run's ``--steps`` and ``--batch-size`` so the
offset is the same one. The candidate list is therefore literally EXP-R2-191's
held-out candidates in the same order, extended past its 1,024 because a
covariance in 4,096 dimensions needs more positions than 256 sequences carry;
its 961-of-1,024 protein screen and its 1,024-of-1,024 text screen reproduce as a
prefix of this one, and the stage reports that prefix as a specification check.
This is a measurement of the population the dictionaries were **scored** on.

**The two things a position budget can quietly break, both refused here.** A
covariance estimated from fewer than ``d_model`` samples is rank-limited by its
own sampling and would report the budget rather than the data, so the stage
refuses a budget below :data:`MIN_POSITION_MULTIPLE` times ``d_model``. And
positions taken from few records measure those records: token positions within
one protein or one document are strongly correlated, so the budget is spent as a
hard equal cap per record over many distinct records, and a record too short to
fill its cap contributes nothing rather than contributing what it has.

**Two instrument controls, and no reading without them.** Isotropic Gaussian
noise matched to each layer's covariance trace, at the same ``N``, through the
same accumulator: an estimator that cannot return ``r99`` near ``d_model`` on
exactly full-rank data is broken, and this stage refuses its own verdict when
that fails. And the coordinate-independent null -- the spectrum of
``diag(C)``, which is what an independent per-coordinate permutation of the
sampled positions induces -- which separates "the coordinates have unequal
variances" from "the coordinates are correlated". :mod:`src.transfer.spectrum`
records why a permutation of whole activation vectors is not a control.

**This stage takes no decision.** One invocation measures one cell, and the
pre-registered rule that reads B2 against ``r99`` is a comparison *between*
matched text and protein cells. The artefact carries this cell's median ``r99``
and says which side of ``d_model`` it falls on; naming a rule is the reader's,
across four artefacts.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from panel_contract import CAMPAIGN_PANEL  # noqa: E402
from src.transfer import joint_modes  # noqa: E402
from src.transfer.arms import REPO, corpus_location, iter_corpus_records  # noqa: E402
from src.transfer.io import write_json  # noqa: E402
from src.transfer.near_duplicates import screen_against_training_stream  # noqa: E402
from src.transfer.replaceable import (  # noqa: E402
    JOINT_MODES,
    PROGEN3_ARM,
    JointReplaceable,
    ReplaceableModel,
    arm_training_corpus,
    eligible_arms,
    joint_mode_corpus,
    joint_tokenisation,
    load_replaceable,
)
from src.transfer.spectrum import (  # noqa: E402
    CovarianceAccumulator,
    coordinate_independent_spectrum,
    isotropic_control_spectrum,
    sample_positions,
    spectrum_statistics,
)


def _load_stage(filename: str) -> Any:
    """Import a stage whose module name starts with a digit.

    The same six lines ``17_train_transcoder.py`` and
    ``23_perturbation_sensitivity.py`` carry, for the same reason: a stage that
    owns a declaration is imported rather than restated (Appendix B rule 12).
    There is no shared importer in ``src/`` to call instead, and adding one would
    mean editing files another agent is auditing.
    """

    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(f"_transfer_stage_{filename[:2]}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


#: The stage that owns the corpus draw, the held-out offset, the near-duplicate
#: screen, the activation capture and the joint-checkpoint band. Everything this
#: stage does to *reach* an activation is that stage's, so the population
#: measured here is the population its dictionaries were scored on. Its own
#: ``STAGE21`` handle is reached through it rather than loaded again, so exactly
#: one copy of the qualification stage is resident.
STAGE17 = _load_stage("17_train_transcoder.py")
STAGE21 = STAGE17.STAGE21

SCHEMA_VERSION = "r2_transfer_activation_spectrum_v1"
DEFAULT_OUT = REPO / "results/transfer/activation_spectrum"

#: The site the pre-registered rule is read at, and the site the transcoders
#: decode into. Its partner ``block_input`` is measured at the same time because
#: the capture returns both and the encoder reads the input, so a reader can see
#: whether the conclusion depends on which end of the module is looked at.
PRIMARY_SITE = "block_output"
SITES: tuple[str, ...] = ("block_input", PRIMARY_SITE)

#: Minimum sampled positions per cell, as a multiple of ``d_model``. Ten is the
#: declared floor -- 40,960 positions on this backbone -- below which the
#: spectrum reports the sampling budget rather than the data. It is a refusal and
#: not a warning: a rank-limited spectrum is indistinguishable from a
#: low-dimensional one in every statistic this stage reports.
MIN_POSITION_MULTIPLE = 10

#: How far below ``d_model`` the isotropic control's ``r99`` may fall before the
#: instrument is called broken. At ``N/d_model`` of order ten the sample spectrum
#: of exactly isotropic data is spread by the Marchenko-Pastur law, and its
#: smallest eigenvalues carry the last percent of the mass, so ``r99`` sits a few
#: percent below ``d_model`` on data that is genuinely full rank. Five percent is
#: wide enough to accommodate that and far tighter than any real reading at
#: stake, which is at factors and not at percents.
ISOTROPIC_CONTROL_FLOOR = 0.95


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--arm",
        default=PROGEN3_ARM,
        choices=eligible_arms(CAMPAIGN_PANEL),
        help="which decoder's blocks to measure, through the same declaration "
        "17_train_transcoder.py trains against",
    )
    target.add_argument(
        "--joint-checkpoint",
        type=Path,
        default=None,
        help="directory of a joint language-protein checkpoint to measure "
        "instead of a panel arm. Requires --rendering and --mode",
    )
    parser.add_argument("--rendering", default=None, choices=joint_modes.RENDERING_NAMES)
    parser.add_argument("--mode", default=None, choices=JOINT_MODES)
    parser.add_argument("--protein-context", default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument(
        "--steps",
        type=int,
        required=True,
        help="the --steps of the dictionary run whose held-out population this "
        "measures. Required and not defaulted: with --batch-size it is what sets "
        "the held-out offset, so a wrong value silently measures a different "
        "population from the one the dictionary was scored on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="that run's --batch-size, which enters the held-out offset, and the "
        "forward batch this stage captures in",
    )
    parser.add_argument(
        "--records",
        type=int,
        default=1024,
        help="how many distinct held-out records the sampled positions are spread "
        "over. With --positions-per-record it fixes the budget exactly, so every "
        "cell of a comparison is estimated at the same N",
    )
    parser.add_argument(
        "--positions-per-record",
        type=int,
        default=64,
        help="scored token positions taken from each record, drawn uniformly "
        "without replacement from within it. A hard equal cap rather than a "
        "ceiling: an unequal draw would weight the longest records",
    )
    parser.add_argument(
        "--isotropic-chunk",
        type=int,
        default=4096,
        help="rows per block of the isotropic control's draw. Bounds its memory "
        "and changes nothing it reports",
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--corpus-seed",
        type=int,
        default=20260812,
        help="seed of the shuffled stream. Pass the dictionary run's value, or "
        "this measures a different draw",
    )
    return parser


def resolve_target(args: argparse.Namespace) -> None:
    """Refuse an incoherent target, through the stage that declares what one is.

    ``17_train_transcoder.resolve_target`` also range-checks ``--train-tokens``,
    which this stage has no analogue of and must not carry into its settings
    block as a dead field. It is therefore asked of a throwaway namespace and the
    one thing it decides -- whether ``--arm`` survives beside a joint checkpoint
    -- is copied back.
    """

    probe = argparse.Namespace(**vars(args), train_tokens=0)
    STAGE17.resolve_target(probe)
    args.arm = probe.arm


def load_target(args: argparse.Namespace) -> tuple[ReplaceableModel, dict[str, Any], int, int | None]:
    """The model, its identity record and the corpus band, as stage 17 builds them."""

    joint = args.joint_checkpoint is not None
    source = joint_mode_corpus(args.mode) if joint else arm_training_corpus(args.arm)
    low, high = STAGE17.CORPUS_BAND[source]

    if joint:
        declaration = joint_modes.rendering(args.rendering)
        print(
            f"[loader] {args.joint_checkpoint} as {declaration.name}:{args.mode} "
            f"on {args.device}"
        )
        resolved, tokenizer = STAGE21.load_tokenizer(args.joint_checkpoint)
        tokenisation = joint_tokenisation(tokenizer, declaration, args.mode)
        if tokenisation is not None:
            low, high = STAGE17.joint_protein_band(
                tokenisation,
                max_tokens=args.max_tokens,
                protein_context=args.protein_context,
            )
        backbone, checkpoint_facts = STAGE21.load_model(
            resolved, tokenizer, device=args.device, dtype="bfloat16"
        )
        checkpoint_facts["requested_path"] = str(args.joint_checkpoint)
        model_handle: ReplaceableModel = JointReplaceable(
            model=backbone,
            tokenizer=tokenizer,
            checkpoint=resolved,
            declaration=declaration,
            mode=args.mode,
            tokenisation=tokenisation,
            max_tokens=args.max_tokens,
            protein_context=args.protein_context,
        )
        target = {
            "kind": "joint_checkpoint",
            "rendering_family": declaration.name,
            "mode": args.mode,
            "checkpoint_facts": checkpoint_facts,
            "rendering": (
                tokenisation.facts()
                if tokenisation is not None
                else {"verdict": "NOT_RESOLVED", "declared_family": declaration.name}
            ),
        }
    else:
        print(f"[loader] loading {args.arm} and running its self-check")
        model_handle = load_replaceable(
            args.arm,
            campaign_panel=CAMPAIGN_PANEL,
            device=args.device,
            dtype="bfloat16",
            max_tokens=args.max_tokens,
            checkpoint=args.checkpoint,
        )
        target = {"kind": "panel_arm", "arm": args.arm}

    target["corpus_source"] = source
    return model_handle, target, low, high


def draw_held_out(
    args: argparse.Namespace, *, source: str, low: int, high: int | None
) -> tuple[list[tuple[str, str | None]], dict[str, Any]]:
    """Stage 17's held-out draw, at this run's offset, extended to this budget.

    Same stream, same seed, same offset arithmetic and same near-duplicate
    screen. The only difference is how many candidates are drawn, and the draw is
    a prefix-stable generator, so the dictionary run's candidates are the first
    of these in the same order and its screen result reproduces as a prefix.
    """

    def records() -> Iterator[tuple[str, str | None]]:
        return iter_corpus_records(
            source, min_symbols=low, max_symbols=high, path=args.corpus
        )

    blocks_touched = -(-(args.steps * args.batch_size) // STAGE17.SHUFFLE_BLOCK)
    held_out_offset = blocks_touched * STAGE17.SHUFFLE_BLOCK
    symbol_unit = "characters" if source == "openwebtext" else "residues"
    wanted = args.records * STAGE17.HELD_OUT_OVERSAMPLE
    candidates = list(
        STAGE17.stream_records(
            records, seed=args.corpus_seed, skip=held_out_offset, limit=wanted
        )
    )
    if len(candidates) < wanted:
        raise RuntimeError(
            f"the corpus ran out at the held-out offset: {len(candidates)} of "
            f"{wanted} candidates past a skip of {held_out_offset}. Lower "
            "--records rather than measuring a population training also reaches"
        )
    print(
        f"[held-out] screening {len(candidates)} candidates against the "
        f"{held_out_offset} training records, on {symbol_unit}"
    )
    started = time.time()
    keep, screen = screen_against_training_stream(
        [entry[0] for entry in candidates],
        (
            entry[0]
            for entry in STAGE17.stream_records(
                records, seed=args.corpus_seed, skip=0, limit=held_out_offset
            )
        ),
        unit=symbol_unit,
    )
    survivors = [entry for entry, kept in zip(candidates, keep) if kept]
    prefix = min(STAGE17.HELD_OUT_OVERSAMPLE * 256, len(candidates))
    screen.update(
        {
            "held_out_offset": held_out_offset,
            "oversample_factor": STAGE17.HELD_OUT_OVERSAMPLE,
            "elapsed_s": round(time.time() - started, 1),
            # The dictionary runs drew 256 sequences at an oversample of 4, so
            # their whole candidate list is this one's first 1,024. Reported so
            # the claim that this is the same draw is checkable rather than
            # argued: EXP-R2-191 and EXP-R2-201 both record 961 kept of 1,024 on
            # Swiss-Prot and 1,024 of 1,024 on OpenWebText.
            "dictionary_run_prefix": {
                "n_candidates": int(prefix),
                "n_kept": int(keep[:prefix].sum()),
            },
        }
    )
    print(
        f"  kept {screen['n_kept']} of {screen['n_candidates']}, "
        f"max containment {screen['max_containment']:.4f}, "
        f"prefix {screen['dictionary_run_prefix']['n_kept']} of {prefix}"
    )
    if len(survivors) < args.records:
        raise RuntimeError(
            f"the near-duplicate screen left {len(survivors)} of {args.records} "
            f"held-out records from {len(candidates)} candidates: this corpus "
            "region is too close to what training reaches"
        )
    return survivors, screen


@torch.no_grad()
def accumulate(
    model_handle: ReplaceableModel,
    survivors: list[tuple[str, str | None]],
    *,
    args: argparse.Namespace,
    n_layers: int,
    d_model: int,
) -> tuple[dict[str, CovarianceAccumulator], dict[str, Any]]:
    """Sampled activations at both sites, as a streaming float64 covariance."""

    accumulators = {
        site: CovarianceAccumulator(
            n_layers=n_layers, d_model=d_model, device=args.device
        )
        for site in SITES
    }
    generator = torch.Generator().manual_seed(args.seed)
    used_records = 0
    short_records = 0
    consumed = 0
    started = time.time()
    while used_records < args.records and consumed < len(survivors):
        chunk = survivors[consumed : consumed + args.batch_size]
        consumed += len(chunk)
        inputs, outputs, mask = STAGE17.capture(model_handle, chunk)
        inputs, outputs = STAGE17.flatten(inputs, outputs, mask)
        indices, used, short = sample_positions(
            mask, per_record=args.positions_per_record, generator=generator
        )
        short_records += len(short)
        if not used:
            continue
        room = args.records - used_records
        if len(used) > room:
            used = used[:room]
            indices = indices[: room * args.positions_per_record]
        selected = indices.to(inputs.device)
        accumulators["block_input"].update(inputs[:, selected])
        accumulators[PRIMARY_SITE].update(outputs[:, selected])
        used_records += len(used)
        if used_records % (args.batch_size * 64) < args.batch_size:
            print(
                f"  {used_records}/{args.records} records  "
                f"{accumulators[PRIMARY_SITE].n_samples} positions  "
                f"{time.time() - started:.0f}s"
            )

    if used_records < args.records:
        raise RuntimeError(
            f"only {used_records} of {args.records} held-out records carry at "
            f"least {args.positions_per_record} scored positions "
            f"({short_records} were shorter, {consumed} consumed). Lower "
            "--positions-per-record or raise --records; do not estimate a "
            "covariance on a budget the draw did not supply"
        )
    n_samples = accumulators[PRIMARY_SITE].n_samples
    expected = args.records * args.positions_per_record
    if n_samples != expected:
        raise RuntimeError(
            f"the position budget did not close: {n_samples} sampled against "
            f"{expected} declared"
        )
    if n_samples < MIN_POSITION_MULTIPLE * d_model:
        raise RuntimeError(
            f"{n_samples} sampled positions is below the declared floor of "
            f"{MIN_POSITION_MULTIPLE} x d_model = {MIN_POSITION_MULTIPLE * d_model}: "
            "a covariance estimated at this budget reports its own sampling and "
            "not the data"
        )
    sampling = {
        "n_positions": int(n_samples),
        "n_records_used": int(used_records),
        "positions_per_record": int(args.positions_per_record),
        "n_records_too_short": int(short_records),
        "n_survivors_consumed": int(consumed),
        "n_survivors_available": int(len(survivors)),
        "position_floor": int(MIN_POSITION_MULTIPLE * d_model),
        "sample_rank_ceiling": int(min(n_samples, d_model)),
        "elapsed_s": round(time.time() - started, 1),
        "draw": (
            "an equal cap of positions per record, drawn uniformly without "
            "replacement within each record from a seeded generator, over "
            "distinct records of the screened held-out draw; records carrying "
            "fewer scored positions than the cap are refused rather than "
            "partially used"
        ),
        "scored_positions": (
            "the content mask the dictionaries were fitted and scored under -- "
            "residue positions alone in protein mode, non-special tokens in text "
            "mode -- which is also what Appendix B rule 11 requires of a spectrum: "
            "an attention sink or a format separator would otherwise dominate it"
        ),
    }
    return accumulators, sampling


def measure(
    accumulators: dict[str, CovarianceAccumulator],
    *,
    args: argparse.Namespace,
    n_layers: int,
    d_model: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Per-layer spectra at every site, with both controls beside each."""

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    spectra: dict[str, list[dict[str, Any]]] = {}
    controls: dict[str, Any] = {}
    for site, accumulator in accumulators.items():
        n_samples = accumulator.n_samples
        layers: list[dict[str, Any]] = []
        started = time.time()
        for layer in range(n_layers):
            covariance = accumulator.covariance_at(layer)
            observed = spectrum_statistics(
                torch.linalg.eigvalsh(covariance),
                n_samples=n_samples,
                d_model=d_model,
            )
            shuffled = coordinate_independent_spectrum(
                covariance, n_samples=n_samples, d_model=d_model
            )
            isotropic = isotropic_control_spectrum(
                total_variance=observed["total_variance"],
                d_model=d_model,
                n_samples=n_samples,
                chunk=args.isotropic_chunk,
                device=args.device,
                generator=generator,
            )
            layers.append(
                {
                    "layer": layer,
                    "observed": observed,
                    "coordinate_independent_control": shuffled,
                    "isotropic_control": isotropic,
                }
            )
            del covariance
        print(
            f"[spectrum] {site}: {n_layers} layers in {time.time() - started:.0f}s  "
            f"median r99 {int(np.median([e['observed']['r99'] for e in layers]))}"
        )
        spectra[site] = layers
        isotropic_r99 = [entry["isotropic_control"]["r99"] for entry in layers]
        controls[site] = {
            "isotropic_r99_min": int(min(isotropic_r99)),
            "isotropic_r99_median": float(np.median(isotropic_r99)),
            "isotropic_floor": int(round(ISOTROPIC_CONTROL_FLOOR * d_model)),
            "isotropic_passes": bool(
                min(isotropic_r99) >= ISOTROPIC_CONTROL_FLOOR * d_model
            ),
            "coordinate_independent_r99_median": float(
                np.median(
                    [entry["coordinate_independent_control"]["r99"] for entry in layers]
                )
            ),
            "mean_shift_residual": accumulators[site].mean_shift_residual(),
            "max_negative_eigenvalue_share": max(
                entry["observed"]["negative_eigenvalue_mass"]
                / entry["observed"]["total_variance"]
                for entry in layers
            ),
        }
    return spectra, controls


def verdict_record(
    spectra: dict[str, list[dict[str, Any]]],
    controls: dict[str, Any],
    *,
    d_model: int,
) -> dict[str, Any]:
    """This cell's median ``r99`` at the primary site, and nothing beyond it."""

    layers = spectra[PRIMARY_SITE]
    r99 = np.asarray([entry["observed"]["r99"] for entry in layers], dtype=np.float64)
    passes = all(controls[site]["isotropic_passes"] for site in spectra)
    if passes:
        reading = (
            "R99_MEDIAN_AT_OR_ABOVE_D_MODEL"
            if float(np.median(r99)) >= d_model
            else "R99_MEDIAN_BELOW_D_MODEL"
        )
    else:
        reading = "INSTRUMENT_CONTROL_FAILED"
    return {
        "primary_site": PRIMARY_SITE,
        "d_model": int(d_model),
        "r99_median": float(np.median(r99)),
        "r99_iqr": [float(np.quantile(r99, 0.25)), float(np.quantile(r99, 0.75))],
        "r99_min": int(r99.min()),
        "r99_max": int(r99.max()),
        "r999_median": float(
            np.median([entry["observed"]["r999"] for entry in layers])
        ),
        "r95_median": float(np.median([entry["observed"]["r95"] for entry in layers])),
        "instrument_controls_pass": bool(passes),
        "reading": reading,
        "scope": (
            "one cell. The pre-registered rule that re-reads basis-adequacy gate "
            "B2 against r99 compares matched text and protein cells of one "
            "checkpoint, so it is not decidable from this artefact and is not "
            "named here"
        ),
        "r99_is_a_variance_based_effective_dimension": (
            "directions beyond r99 carry under 1% of the total variance between "
            "them, which bounds what a fit at this token budget can resolve and "
            "does not assert that they are empty. r999 is reported per layer so "
            "the sensitivity to that cut is readable rather than assumed"
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    resolve_target(args)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    model_handle, target, low, high = load_target(args)
    source = target["corpus_source"]
    corpus = corpus_location(source, path=args.corpus)

    loader_gate = model_handle.self_check()
    band = loader_gate.get("nll")
    print(
        f"  self-check {loader_gate['verdict']}"
        + ("" if band is None else f", NLL {band:.4f}")
    )
    n_layers = model_handle.n_layers
    d_model = model_handle.width
    target.update(
        {
            "name": model_handle.name,
            "checkpoint": str(model_handle.checkpoint),
            "weights_sha256": model_handle.weights_digest(),
            "n_layers": int(n_layers),
            "d_model": int(d_model),
            "loading_note": model_handle.loading_note,
        }
    )
    budget = args.records * args.positions_per_record
    if budget < MIN_POSITION_MULTIPLE * d_model:
        raise ValueError(
            f"--records {args.records} x --positions-per-record "
            f"{args.positions_per_record} = {budget} positions is below the "
            f"declared floor of {MIN_POSITION_MULTIPLE} x d_model = "
            f"{MIN_POSITION_MULTIPLE * d_model}"
        )

    survivors, screen = draw_held_out(args, source=source, low=low, high=high)
    accumulators, sampling = accumulate(
        model_handle,
        survivors,
        args=args,
        n_layers=n_layers,
        d_model=d_model,
    )
    spectra, controls = measure(
        accumulators, args=args, n_layers=n_layers, d_model=d_model
    )
    verdict = verdict_record(spectra, controls, d_model=d_model)

    args.corpus = corpus
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "target": target,
        "loader_gate": loader_gate,
        "held_out": {"near_duplicate_screen": screen},
        "sampling": sampling,
        "spectrum": spectra,
        "controls": controls,
        "verdict": verdict,
        "condition": {
            "site": (
                "the per-layer feed-forward module's output before the residual "
                "add (block_output, the tensor the dictionaries decode into) and "
                "its input (block_input, the tensor the encoder reads); every "
                "layer, one covariance each"
            ),
            "estimator": (
                "centred covariance accumulated in float64 about a fixed shift "
                "taken from the first batch, unbiased by n-1; eigenvalues from a "
                "float64 symmetric eigensolver"
            ),
            "corpus": str(corpus),
            "corpus_source": source,
            "symbol_band": [low, high],
            "symbol_unit": "characters" if source == "openwebtext" else "residues",
            "input_rendering": model_handle.rendering_note,
            "held_out_draw": (
                "17_train_transcoder.py's own stream, seed, offset and "
                "near-duplicate screen, driven with the dictionary run's --steps "
                "and --batch-size, so this is the population that run was scored "
                "on rather than a fresh draw. See "
                "held_out.near_duplicate_screen.dictionary_run_prefix"
            ),
            "controls": (
                "isotropic Gaussian noise at each layer's own covariance trace "
                "and the same N, through the same accumulator -- an instrument "
                "check the verdict is refused without; and the "
                "coordinate-independent null, the exact spectrum of diag(C), "
                "which is what an independent per-coordinate permutation induces. "
                "A permutation of whole activation vectors leaves a covariance "
                "unchanged and is not a control"
            ),
            "what_this_does_not_measure": (
                "an algebraic rank, and the causal relevance of the directions "
                "beyond r99. A variance cut says what a fit at this token budget "
                "can resolve; a direction carrying little variance may still "
                "carry a great deal of computation, and nothing here bounds that"
            ),
        },
    }
    stem = f"{target.get('rendering_family', args.arm)}_{args.mode or 'native'}_spectrum"
    if target["kind"] == "joint_checkpoint":
        stem = f"{args.rendering}_{args.mode}_{Path(str(args.joint_checkpoint)).name}_spectrum"
    write_json(args.out / f"{stem}.json", payload)
    print(
        f"[done] {verdict['reading']}  median r99 {verdict['r99_median']:.0f} of "
        f"d_model {d_model}  (r95 {verdict['r95_median']:.0f}, r999 "
        f"{verdict['r999_median']:.0f})  wrote {args.out / stem}.json"
    )
    if not verdict["instrument_controls_pass"]:
        raise SystemExit(
            "the isotropic instrument control did not reach "
            f"{ISOTROPIC_CONTROL_FLOOR:.0%} of d_model, so this estimator cannot "
            "recover a full-rank spectrum and no reading of the measured spectra "
            "may be taken. The artefact is written so the failure is inspectable"
        )


if __name__ == "__main__":
    main()
