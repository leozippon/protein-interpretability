#!/usr/bin/env python3
"""Under identical weights, do text and protein continuation use distinct subspaces?

**Why this question is askable here and almost nowhere else.** A joint
language-protein decoder holds one set of weights and reads two input formats, so
architecture, scale, tokenizer *and weights* are all held fixed while the mode
moves. Two standalone models control the first three and never the fourth. On this
lineage all four cells are behaviourally measurable and were measured:
``ProLLaMA_Stage_1`` reads +0.8336 nats/token of context information in text and
+0.5505 in protein, ``ProLLaMA`` +0.7368 and +0.5215 (EXP-R2-152). That is what
makes a *behavioural* subspace claim possible rather than only a representational
one.

**This is an Objective 1/2 measurement of what a model does.** Audit section 7.0's
recombination ceiling governs Objective-3 knowledge claims and says in its own
closing paragraph that it does not reach an objective where the measurement of
model behaviour is itself the result. Nothing here is a claim about biological
knowledge, and no number from this stage may be read as one. What section 7.0
*does* govern and this stage inherits is the resampling unit: the near-duplicate
group, never the record (L30).

**No token alignment is required, and that is why this design is available.**
Everything measured here is a property of a *subspace* of R^d_model -- which
directions a mode occupies, which of them it needs -- and a subspace comparison
carries no position index. L31's measurement that a single substitution leaves a
multi-residue-BPE arm token-aligned on 47.0-54.5% of instances, with a survivor
set selected by BPE stability rather than at random, blocks a role swap, an
analogue patch and an intra-fragment intervention on this very lineage (audit
section 7.0). It does not reach this. See
:data:`src.transfer.mode_subspaces.NO_ALIGNMENT_NOTE`.

What the stage does, in order
=============================

1. **Occupancy.** One capture pass per mode over a matched cohort, accumulating
   :class:`src.transfer.spectrum.CovarianceAccumulator` at each declared layer,
   summarised by :func:`src.transfer.spectrum.spectrum_statistics`. That is the
   R1.2 estimator, reused rather than re-derived. Several statistics are reported
   and none is the answer: R1.2's own multipliers on one contrast are 35x on the
   participation ratio, 7.7x on the effective rank and 1.24x on the 99%-variance
   dimension.
2. **Necessity.** Each mode's top-r principal directions at each layer are
   projected out of that layer's feed-forward write and the damage to a mode's
   own next-symbol likelihood is measured, over a declared ladder of r, against
   two anchors that exist independently of the ladder: rank 0, which is the clean
   pass, and the whole block write zeroed through
   :meth:`src.transfer.replaceable.JointReplaceable.ablated`.
3. **Cross-mode driveability.** The same ablation evaluated in the *other* mode,
   giving the full 2x2 with intervals. The decisive contrast is taken **within**
   an evaluated mode -- own basis against the other mode's basis at matched rank,
   on the same positions -- because that is the only form in which it is paired:
   two modes score different symbols and a difference of their two means is not a
   paired quantity.
4. **Overlap** of the two modes' *necessary* subspaces, always beside the chance
   level for two random subspaces of the same dimensions in the same ambient
   dimension. At d_model 4,096 two random 512-dimensional subspaces already share
   a mean squared principal cosine of 0.125.
5. **The unigram control**, which decides whether any of it means anything. Every
   damage figure is split per position, and therefore exactly, into the part
   explained by the shift in the model's own marginal predictive distribution and
   the residual. **The headline claim is licensed only by the residual**; a large
   total damage with a small residual is reported as ``UNIGRAM_ONLY``, which is
   the much weaker finding that the modes differ in their unigram statistics.

Every threshold is a constant of a named rule in
:data:`src.transfer.mode_subspaces.DECISION_RULES`, selected by a required
``--decision-rule`` and never passed as a number, so what counts as a result is
fixed before the result exists.

Where a behavioural read is refused
===================================

``Llama-2-7b-hf`` enters this stage in **text mode only**, as a representational
reference. Its protein mode reads +0.0843 nats/token of context information and a
reversal cost of -0.0013 nats/residue (EXP-R2-152, re-measured at EXP-R2-174),
against a 0.30-nat floor, so no likelihood-based quantity may be read in it. The
refusal is keyed to that measured number -- supplied through
``--context-information`` and decided by
:func:`src.transfer.budget.power_status` -- and never to a checkpoint name,
because every checkpoint here is reached by path and a name-keyed guard would pass
silently on the same weights under a different directory. Occupancy is *not*
refused there: the activations exist and their covariance is a real object, which
is the distinction ``32_crosscoder.py`` already records for the same checkpoint's
protein mode. This is not a catalogued limitation; the catalogue ends at L32.

The matched cohort, and what could not be matched
=================================================

The two modes are compared at **exactly** equal scored-position counts and exactly
equal positions per record: ``--records`` records per mode, ``--positions-per-record``
positions drawn uniformly without replacement inside each, through
:func:`src.transfer.spectrum.sample_positions`. A record that cannot supply the
cap contributes nothing and the run **stops** rather than proceeding with an
unequal draw, so an imbalance is a refusal and never a silent asymmetry. What
could not be matched is declared in the artefact's ``matched_cohort`` block: the
token-length distribution of the two mode's rendered inputs, the symbol unit (a
protein scored token carries about 1.54 residues on this rendering and a text
token about four characters, which is L23's incommensurability), and the corpora
themselves, which are Swiss-Prot and OpenWebText and cannot be made one
population.

Per layer, never a mean
=======================

Every quantity is indexed by layer and every verdict is a per-layer verdict.
``src.transfer.crosscoder.assert_per_layer_fields`` refuses any ``*_per_site``
field collapsed to a scalar before the artefact is written. L32 and Appendix B
rule 33 exist because a criterion stated per unit was instrumented as a
cross-layer mean and returned a verdict its own per-layer vector contradicted.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.transfer import joint_modes  # noqa: E402
from src.transfer import mode_subspaces as ms  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    DEFAULT_CORPUS_DRAW_SEED,
    REPO,
    corpus_location,
    protein_cohort,
    text_cohort,
)
from src.transfer.crosscoder import assert_per_layer_fields  # noqa: E402
from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.near_duplicates import near_duplicate_groups  # noqa: E402
from src.transfer.replaceable import (  # noqa: E402
    JOINT_MODES,
    JointReplaceable,
    joint_mode_corpus,
    joint_tokenisation,
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

    Two of them, and each because a number here is only readable if it came from
    the *same* computation another stage performs: stage 21 owns the checkpoint
    loader and its read-back facts, and stage 25 owns the per-layer tensor's
    declaration. Appendix B rule 12 does not stop applying because the declaration
    lives in a file whose name starts with a digit.
    """

    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(f"_transfer_stage_{filename[:2]}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE21 = _load_stage("21_joint_mode_qualification.py")
STAGE25 = _load_stage("25_model_diffing_baselines.py")

SCHEMA_VERSION = ms.SCHEMA_VERSION
DEFAULT_OUT = REPO / "results/transfer/mode_subspaces"

#: float32, and it is not the usual choice on this programme. Every other joint
#: stage loads at bfloat16 because it compares two checkpoints and a comparison
#: across two precisions is partly a comparison of two quantisations. Here there is
#: **one** checkpoint, so that hazard is absent, and the quantity being read is the
#: difference between two forward passes over the same weights -- which at
#: bfloat16's eight-bit mantissa is comparable to the smallest damages the ladder's
#: low rungs are supposed to resolve. The dictionary of this measurement is the
#: difference itself, so it is taken at the precision the difference needs.
INFERENCE_DTYPE = "float32"

#: Arguments that name a real campaign, refused under ``--synthetic``.
CAMPAIGN_ONLY_FLAGS = (
    "checkpoint",
    "rendering",
    "modes",
    "layers",
    "context_information",
)

PROVENANCE_MODULES = (
    "src/transfer/mode_subspaces.py",
    "src/transfer/spectrum.py",
    "src/transfer/budget.py",
    "src/transfer/statistics.py",
    "src/transfer/joint_modes.py",
    "src/transfer/replaceable.py",
    "src/transfer/near_duplicates.py",
    "src/transfer/crosscoder.py",
    "src/transfer/arms.py",
    "src/transfer/io.py",
    "scripts/transfer/21_joint_mode_qualification.py",
    "scripts/transfer/25_model_diffing_baselines.py",
)


# ------------------------------------------------------------------- helpers


def parse_layers(argument: str) -> tuple[int, ...]:
    """``"0,15-16,31"`` into ``(0, 15, 16, 31)``.

    Ranges are inclusive at both ends, because a layer set is written by a person
    reading a per-layer table and a half-open range would silently drop the layer
    they meant to include. The same rule ``32_crosscoder.py`` states; it is eight
    lines of parsing rather than a declaration, and importing that stage for it
    would pull three model-loading stages into this one's import graph.
    """

    layers: list[int] = []
    for piece in argument.replace(" ", "").split(","):
        if not piece:
            continue
        if "-" in piece.lstrip("-"):
            low, _, high = piece.partition("-")
            start, stop = int(low), int(high)
            if stop < start:
                raise argparse.ArgumentTypeError(
                    f"{piece!r} is a descending range; write it low-high"
                )
            layers.extend(range(start, stop + 1))
        else:
            layers.append(int(piece))
    if not layers:
        raise argparse.ArgumentTypeError("a layer set cannot be empty")
    if len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError(f"{argument!r} names a layer twice")
    return tuple(sorted(layers))


def parse_context_information(values: Sequence[str]) -> dict[str, float]:
    """``["text=0.8336", "protein=0.5505"]`` into a mapping, refusing anything else."""

    declared: dict[str, float] = {}
    for entry in values:
        if entry.count("=") != 1:
            raise argparse.ArgumentTypeError(
                f"{entry!r} is not MODE=NATS; each mode's measured context "
                "information from EXP-R2-152 must be named with its mode"
            )
        mode, _, number = entry.partition("=")
        if mode not in JOINT_MODES:
            raise argparse.ArgumentTypeError(
                f"{mode!r} is not a joint mode; declared: {list(JOINT_MODES)}"
            )
        if mode in declared:
            raise argparse.ArgumentTypeError(f"{mode!r} is declared twice")
        declared[mode] = float(number)
    return declared


def cohort_for(mode: str, args: argparse.Namespace) -> Any:
    """One mode's cohort, drawn through the panel's own seeded constructors.

    Both modes are drawn under the same rule and the same seed, which is the only
    way the text side is a control for the protein side rather than a differently
    sampled population (Appendix B rule 1).
    """

    if mode == "protein":
        return protein_cohort(
            args.records,
            args.protein_min_len,
            args.protein_max_len,
            skip=0,
            name="mode_subspaces_protein",
            seed=args.cohort_draw_seed,
        )
    return text_cohort(
        args.records,
        args.text_min_chars,
        skip=0,
        name="mode_subspaces_text",
        seed=args.cohort_draw_seed,
    )


def build_handle(
    model: Any,
    tokenizer: Any,
    checkpoint: Path,
    declaration: joint_modes.JointRendering,
    mode: str,
    args: argparse.Namespace,
) -> JointReplaceable:
    """One mode's view of the *same loaded weights*.

    Two handles over one model object is the whole design: a mode selects the
    rendering, the corpus and the scored span, and nothing about the parameters
    moves between them.
    """

    return JointReplaceable(
        model=model,
        tokenizer=tokenizer,
        checkpoint=checkpoint,
        declaration=declaration,
        mode=mode,
        tokenisation=joint_tokenisation(tokenizer, declaration, mode),
        max_tokens=args.max_tokens,
        protein_context=args.protein_context,
    )


def prepared_batch(handle: JointReplaceable, records: Sequence[str]) -> dict[str, torch.Tensor]:
    """One batch of a mode's own rendered inputs, with both of its masks.

    ``17_train_transcoder.capture`` is the same two calls followed by a tap; it is
    not reused because it returns neither the batch nor the target mask, and the
    target mask is what this stage measures on. The rendering and both masks still
    come from the handle, so nothing about the input format is decided twice.
    """

    batch = handle.batch(handle.render(list(records)))
    handle.forget_rendered()
    return batch


@torch.no_grad()
def capture_layers(
    handle: JointReplaceable, batch: dict[str, torch.Tensor], layers: Sequence[int]
) -> torch.Tensor:
    """``(n_layers, batch, length, d_model)`` of the declared per-layer tensor."""

    wanted = set(int(layer) for layer in layers)
    captured: dict[int, torch.Tensor] = {}

    def tap(layer: int, block_input: torch.Tensor, block_output: torch.Tensor) -> None:
        if layer in wanted:
            captured[layer] = block_output.detach()
        return None

    with handle.block_intercept(tap):
        handle.run(batch)
    return torch.stack([captured[int(layer)] for layer in layers])


def target_positions(
    batch: dict[str, torch.Tensor], *, per_record: int, drop_leading: int, generator: torch.Generator
) -> tuple[torch.Tensor, list[int], list[int]]:
    """Which of this batch's scored targets are measured, capped equally per record.

    The mask is the **target** mask, not the content mask, and that is what makes
    occupancy and necessity statements about the same positions: the activation at
    flattened target index ``j`` is the hidden state that produces the logits for
    the scored target at ``j``, so "the directions this mode occupies here" and
    "the directions this mode needs here" are indexed identically.

    ``drop_leading`` must be at least one and is Appendix B rule 11's requirement
    made concrete on this rendering: the first scored target of a protein record is
    predicted from the position holding the last delimiter piece of ``Seq=<``, and
    a spectrum that included it would be reading the format separator. It also
    removes the leading content positions where a LLaMA-family massive activation
    lands once the beginning-of-sequence token is gone.
    """

    return sample_positions(
        batch["target_mask"],
        per_record=per_record,
        generator=generator,
        drop_leading=drop_leading,
    )


@torch.no_grad()
def scored_pass(
    handle: JointReplaceable,
    batches: Sequence[dict[str, Any]],
    *,
    layer: int | None,
    basis: torch.Tensor | None,
    zero_block: bool,
    label: str,
    vocabulary: int,
) -> ms.ScoredPass:
    """One configuration's readings over the damage cohort.

    ``basis`` projects that subspace out of ``layer``'s feed-forward output.
    ``zero_block`` instead zeroes the whole write through the handle's own
    :meth:`~src.transfer.replaceable.JointReplaceable.ablated`, which is the
    ladder's top anchor and exists independently of this stage. ``basis=None`` with
    ``zero_block=False`` is the clean pass.
    """

    if basis is not None and zero_block:
        raise ValueError("a projection and a whole-block ablation are two interventions")
    nll: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    halves: list[np.ndarray] = []
    marginal = np.zeros((2, vocabulary), dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)

    for entry in batches:
        batch = entry["batch"]
        selected = entry["positions"].to(batch["input_ids"].device)
        if zero_block:
            component = next(
                item
                for item in handle.components()
                if int(item.layer) == int(layer) and item.kind == handle.block_kind
            )
            context: Any = handle.ablated(component)
        elif basis is not None:

            def substitute(
                at: int, block_input: torch.Tensor, block_output: torch.Tensor
            ) -> torch.Tensor | None:
                return ms.project_out(block_output, basis) if at == int(layer) else None

            context = handle.block_intercept(substitute)
        else:
            context = nullcontext()
        with context:
            logits, ids, mask = handle.scored_logits(batch)
        keep = mask.reshape(-1)
        width = int(logits.shape[-1])
        flat_logits = logits.reshape(-1, width)[keep][selected]
        flat_targets = ids.reshape(-1)[keep][selected]
        log_probabilities = torch.log_softmax(flat_logits.float(), dim=-1)
        nll.append(
            (-log_probabilities.gather(1, flat_targets[:, None])[:, 0]).cpu().numpy()
        )
        targets.append(flat_targets.cpu().numpy().astype(np.int64))
        groups.append(entry["groups"])
        halves.append(entry["halves"])
        probabilities = log_probabilities.exp().double().cpu().numpy()
        for half in (0, 1):
            rows = entry["halves"] == half
            if rows.any():
                marginal[half] += probabilities[rows].sum(axis=0)
                counts[half] += int(rows.sum())
        del logits, ids, flat_logits, log_probabilities, probabilities

    return ms.ScoredPass(
        label=label,
        target_ids=np.concatenate(targets),
        nll_nats=np.concatenate(nll).astype(np.float64),
        group_ids=np.concatenate(groups),
        half_ids=np.concatenate(halves),
        marginal=marginal,
        marginal_counts=counts,
    )


def occupancy_for_mode(
    handle: JointReplaceable,
    cohort: Any,
    group_ids: np.ndarray,
    *,
    args: argparse.Namespace,
    layers: Sequence[int],
    d_model: int,
    rule: ms.DecisionRule,
    log: Any,
) -> dict[str, Any]:
    """One mode's capture pass: the covariances, the batches and the position draw.

    Returns the per-layer covariance, the per-batch record needed by the damage
    passes, and the sampling record. It refuses a draw that did not close: a record
    unable to supply the per-record cap contributes nothing, and the run stops
    rather than comparing two modes at unequal position counts.
    """

    generator = torch.Generator(device="cpu").manual_seed(int(args.position_seed))
    accumulators = {
        int(layer): CovarianceAccumulator(n_layers=1, d_model=d_model, device=handle.device)
        for layer in layers
    }
    prepared: list[dict[str, Any]] = []
    used_records = 0
    short_records = 0
    token_lengths: list[int] = []
    started = time.time()

    for start in range(0, len(cohort.records), args.batch_size):
        chunk = list(cohort.records[start : start + args.batch_size])
        if not chunk:
            continue
        batch = prepared_batch(handle, chunk)
        indices, used, short = target_positions(
            batch,
            per_record=args.positions_per_record,
            drop_leading=args.drop_leading,
            generator=generator,
        )
        short_records += len(short)
        if not used:
            continue
        rows = np.asarray([start + int(row) for row in used], dtype=np.int64)
        per_position_groups = np.repeat(group_ids[rows], args.positions_per_record)
        activations = capture_layers(handle, batch, layers)
        trimmed = activations[:, :, :-1, :]
        n_layers, n_rows, length, width = trimmed.shape
        keep = batch["target_mask"].reshape(-1)
        flat = trimmed.reshape(n_layers, n_rows * length, width)[:, keep]
        selected = indices.to(flat.device)
        for position, layer in enumerate(layers):
            accumulators[int(layer)].update(flat[position : position + 1, selected])
        del activations, trimmed, flat
        token_lengths.extend(
            int(value) for value in batch["attention_mask"].sum(dim=1).cpu().tolist()
        )
        prepared.append(
            {
                "batch": batch,
                "positions": indices,
                "groups": per_position_groups,
                "halves": (per_position_groups % 2).astype(np.int64),
                "n_records": len(used),
            }
        )
        used_records += len(used)
        if used_records % (args.batch_size * 16) < args.batch_size:
            log(
                f"  [{handle.mode}] {used_records}/{args.records} records  "
                f"{used_records * args.positions_per_record} positions  "
                f"{time.time() - started:.0f}s"
            )

    if used_records != args.records:
        raise RuntimeError(
            f"{handle.mode}: {used_records} of {args.records} records carry at least "
            f"{args.positions_per_record + args.drop_leading} scored targets "
            f"({short_records} were shorter). The two modes are compared at exactly "
            "equal position counts, so an incomplete draw is refused rather than "
            "trimmed: lower --positions-per-record, raise --records, or widen the "
            "band"
        )
    n_positions = used_records * args.positions_per_record
    floor = rule.min_positions_per_dimension * d_model
    return {
        "covariances": {
            int(layer): accumulators[int(layer)].covariance_at(0) for layer in layers
        },
        "batches": prepared,
        "sampling": {
            "n_positions": int(n_positions),
            "n_records_used": int(used_records),
            "positions_per_record": int(args.positions_per_record),
            "n_records_too_short": int(short_records),
            "drop_leading": int(args.drop_leading),
            "sample_rank_ceiling": int(min(n_positions, d_model)),
            "position_floor_for_a_rank_statistic": int(floor),
            "occupancy_readable": bool(n_positions >= floor),
            "mean_input_tokens": float(np.mean(token_lengths)),
            "max_input_tokens": int(np.max(token_lengths)),
            "min_input_tokens": int(np.min(token_lengths)),
            "mean_shift_residual": float(
                max(
                    accumulators[int(layer)].mean_shift_residual() for layer in layers
                )
            ),
            "elapsed_s": round(time.time() - started, 1),
            "draw": (
                "an equal hard cap of positions per record, drawn uniformly without "
                "replacement inside each record from a seeded generator, over the "
                "mode's own scored targets with the leading ones excluded. A record "
                "that cannot supply the cap contributes nothing and the run stops"
            ),
        },
    }


def occupancy_record(
    covariance: torch.Tensor,
    eigenvalues: torch.Tensor,
    *,
    n_positions: int,
    d_model: int,
    readable: bool,
    isotropic: bool,
    seed: int,
) -> dict[str, Any]:
    """One layer's spectrum, or an explicit refusal in place of its rank statistics."""

    statistics = spectrum_statistics(eigenvalues, n_samples=n_positions, d_model=d_model)
    record: dict[str, Any] = {
        "reported": {key: statistics[key] for key in ms.OCCUPANCY_STATISTICS}
        if readable
        else ms.OCCUPANCY_UNDERSAMPLED,
        "full_spectrum_summary": statistics,
        "coordinate_independent_control": coordinate_independent_spectrum(
            covariance, n_samples=n_positions, d_model=d_model
        ),
        "note": ms.OCCUPANCY_NOTE,
    }
    if not readable:
        record["withheld_reason"] = (
            f"{n_positions} sampled positions against an ambient dimension of "
            f"{d_model}: a covariance estimated from N samples has rank at most "
            "min(N, d), so every rank statistic here would report the sampling "
            "budget rather than the data. The full summary is retained so a reader "
            "can see what was measured, and it must not be quoted as an occupancy"
        )
    if isotropic:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        record["isotropic_control"] = isotropic_control_spectrum(
            total_variance=float(statistics["total_variance"]),
            d_model=d_model,
            n_samples=n_positions,
            chunk=min(4096, max(2, n_positions)),
            device="cpu",
            generator=generator,
        )
    return record


# ------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="directory of the joint checkpoint, BOTH of whose modes are measured. "
        "A path and not an arm name, for the reason 21_joint_mode_qualification.py "
        "gives: a checkpoint that has not passed that stage must not be in arms.py",
    )
    parser.add_argument(
        "--rendering",
        default=None,
        choices=joint_modes.RENDERING_NAMES,
        help="which declared family's protein format this checkpoint takes",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        choices=list(JOINT_MODES),
        help="which modes to measure. REQUIRED and never defaulted. Naming one mode "
        "produces an occupancy reference and no cross-mode comparison, which is how "
        "a checkpoint with only one measurable mode enters this stage",
    )
    parser.add_argument(
        "--layers",
        type=parse_layers,
        default=None,
        help="backbone layers measured, as '0,15-16,31'. REQUIRED and never "
        "defaulted: every quantity here is per layer and there is no cross-layer "
        "summary, so the set IS the pre-registered site declaration (L32, Appendix "
        "B rule 33)",
    )
    parser.add_argument(
        "--rank-ladder",
        type=ms.parse_rank_ladder,
        default="1,2,4,8,16,32,64,128",
        help="ranks of the ablated subspace, ascending. Rank 0 is not a rung: it IS "
        "the clean pass every rung is differenced against",
    )
    parser.add_argument(
        "--overlap-statistic",
        default=None,
        choices=list(ms.OVERLAP_STATISTICS),
        help="which overlap statistic the decision rule reads. REQUIRED and never "
        "defaulted: both are computed and reported, and a rule that could pick "
        "either after seeing both is not a rule",
    )
    parser.add_argument(
        "--decision-rule",
        default=None,
        choices=sorted(ms.DECISION_RULES),
        help="the frozen threshold bundle this run is decided under. REQUIRED and "
        "never defaulted; the thresholds themselves are constants of the named rule "
        "and cannot be passed on the command line",
    )
    parser.add_argument(
        "--context-information",
        nargs="+",
        default=None,
        help="each measured mode's context information from EXP-R2-152, as "
        "'text=0.8336 protein=0.5505'. REQUIRED and never inferred here: it is what "
        "decides whether a behavioural read may be taken in that mode, through "
        "src.transfer.budget.power_status against a 0.30-nat floor. Llama-2-7b-hf's "
        "protein mode reads 0.0843 and is refused by its own number rather than by "
        "its name",
    )
    parser.add_argument("--tensor", default=ms.TENSOR, choices=[ms.TENSOR],
        help="the per-layer tensor everything is defined on. One choice, because "
        "occupancy and necessity have to be read on one tensor or 'occupies but "
        "does not need' is a sentence about two objects -- and it is the only "
        "per-layer tensor the block interceptor can substitute")
    parser.add_argument("--records", type=int, default=640,
        help="records per mode for the occupancy capture. Both modes take the same "
        "number and the same per-record cap, so the position counts match exactly")
    parser.add_argument("--damage-records", type=int, default=64,
        help="records per mode for the ablation passes, taken as a PREFIX of the "
        "occupancy cohort's own seeded draw. A prefix and not a second draw: it is a "
        "subsample of one population, and the covariance a basis is fitted from and "
        "the positions its damage is read on are then the same cohort")
    parser.add_argument("--positions-per-record", type=int, default=64)
    parser.add_argument("--drop-leading", type=int, default=4,
        help="scored targets excluded from the head of every record. Appendix B "
        "rule 11; must be at least 1, because the first protein target is predicted "
        "from the position holding the last piece of the opening delimiter")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--protein-min-len", type=int, default=128)
    parser.add_argument("--protein-max-len", type=int, default=246)
    parser.add_argument("--text-min-chars", type=int, default=800)
    parser.add_argument("--protein-context", default=None)
    parser.add_argument("--cohort-draw-seed", type=int, default=DEFAULT_CORPUS_DRAW_SEED)
    parser.add_argument("--position-seed", type=int, default=20260819)
    parser.add_argument("--bootstrap-seed", type=int, default=20260819)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--chance-draws", type=int, default=64,
        help="random subspace pairs drawn for every overlap statistic's chance band")
    parser.add_argument("--isotropic-control", action="store_true",
        help="also run the spectrum estimator on exactly full-rank isotropic data "
        "at this N and this d, which prices the finite-sample deflation directly. "
        "Off by default because it re-accumulates an N x d x d covariance and "
        "eigendecomposes it on the CPU once per layer per mode -- minutes per cell "
        "at a campaign's position budget -- and what it certifies is the estimator "
        "rather than the checkpoint, so one run of it covers a campaign")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--synthetic", action="store_true",
        help="run the known-answer self-test on planted geometry and write its "
        "certificate instead of measuring a checkpoint. Required before any real "
        "number from this stage is trusted")
    return parser


def resolve(args: argparse.Namespace) -> None:
    """Refuse an incoherent request before a corpus is opened or a model is loaded."""

    if args.drop_leading < 1:
        raise ValueError(
            "--drop-leading must be at least 1: the first scored protein target is "
            "predicted from the position holding the last piece of the opening "
            "delimiter, and a spectrum that included it would read the format "
            "separator rather than the representation (Appendix B rule 11)"
        )
    if args.synthetic:
        present = [flag for flag in CAMPAIGN_ONLY_FLAGS if getattr(args, flag) is not None]
        if present:
            raise ValueError(
                ", ".join(f"--{flag.replace('_', '-')}" for flag in present)
                + " name a real campaign and are meaningless beside --synthetic, "
                "which runs the same instrument on data whose answer is known"
            )
        for flag in ("overlap_statistic", "decision_rule"):
            if getattr(args, flag) is None:
                raise ValueError(
                    f"--{flag.replace('_', '-')} is required on the synthetic path as "
                    "well: the self-test validates the rule that a campaign would be "
                    "decided under, and a self-test run under a different rule "
                    "validates nothing about the campaign"
                )
        return

    missing = [
        flag
        for flag in (*CAMPAIGN_ONLY_FLAGS, "overlap_statistic", "decision_rule")
        if getattr(args, flag) is None
    ]
    if missing:
        raise ValueError(
            "this stage needs "
            + ", ".join(f"--{flag.replace('_', '-')}" for flag in missing)
            + ". Every one of them is a pre-registered decision and none is "
            "defaulted: the layer set because every verdict is per layer, the "
            "overlap statistic and the decision rule because what counts as a result "
            "must be fixed before the result exists, and --context-information "
            "because it is the measured number that decides whether a behavioural "
            "read may be taken in a mode at all"
        )
    args.modes = tuple(dict.fromkeys(args.modes))
    args.context_information = parse_context_information(args.context_information)
    undeclared = [mode for mode in args.modes if mode not in args.context_information]
    if undeclared:
        raise ValueError(
            f"--context-information declares nothing for {undeclared}; each measured "
            "mode's EXP-R2-152 figure is required, because it is what decides "
            "whether that mode may carry a behavioural read"
        )
    extra = [mode for mode in args.context_information if mode not in args.modes]
    if extra:
        raise ValueError(
            f"--context-information declares {extra}, which --modes does not measure"
        )
    if args.damage_records > args.records:
        raise ValueError(
            "--damage-records is a prefix of the occupancy cohort and cannot exceed "
            "--records"
        )
    if max(args.rank_ladder) > args.records * args.positions_per_record:
        raise ValueError(
            f"the ladder reaches rank {max(args.rank_ladder)} on "
            f"{args.records * args.positions_per_record} sampled positions; a "
            "principal basis at a rank near the sample count is the sampling budget"
        )


# --------------------------------------------------------------------- main


def run_synthetic(args: argparse.Namespace) -> dict[str, Any]:
    rule = ms.decision_rule(args.decision_rule)
    design = ms.SyntheticDesign()
    ladder = tuple(rank for rank in args.rank_ladder if rank <= design.context_rank) or (
        1,
        design.context_rank,
    )
    certificate = ms.synthetic_certificate(
        design, ladder=ladder, rule=rule, n_bootstrap=args.n_bootstrap
    )
    # The attainability corners run at the module's own declared ladder, not at the
    # campaign's: see ms.SYNTHETIC_ATTAINABILITY_LADDER.
    attainability = ms.synthetic_verdict_attainability(
        design, rule=rule, n_bootstrap=args.n_bootstrap
    )
    return {
        "kind": "synthetic_known_answer_check",
        "certificate": certificate,
        "verdict_attainability": attainability,
        "passed": certificate["certificate"] == "PASSED" and attainability["passed"],
    }


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    rule = ms.decision_rule(args.decision_rule)
    declaration = joint_modes.rendering(args.rendering)
    log = lambda line: print(line, flush=True)  # noqa: E731

    checkpoint = Path(args.checkpoint).resolve()
    print(f"[paths] checkpoint {checkpoint}")
    for mode in args.modes:
        print(f"[paths] corpus {mode:8s} {corpus_location(joint_mode_corpus(mode))}")
    print(f"[paths] out        {Path(args.out).resolve()}")

    measurability = {
        mode: ms.mode_measurability(mode, args.context_information[mode])
        for mode in args.modes
    }
    behavioural = tuple(
        mode for mode in args.modes if measurability[mode]["behavioural_read_admitted"]
    )
    for mode in args.modes:
        print(
            f"[mode] {mode:8s} {measurability[mode]['context_information_nats']:+.4f} "
            f"nats/token -> {measurability[mode]['measurability']}"
        )

    resolved, tokenizer = STAGE21.load_tokenizer(checkpoint)
    model, facts = STAGE21.load_model(
        resolved, tokenizer, device=args.device, dtype=INFERENCE_DTYPE
    )
    handles = {
        mode: build_handle(model, tokenizer, resolved, declaration, mode, args)
        for mode in args.modes
    }
    reference = handles[args.modes[0]]
    d_model = int(reference.width)
    n_layers = int(reference.n_layers)
    outside = sorted(layer for layer in args.layers if not 0 <= layer < n_layers)
    if outside:
        raise ValueError(f"--layers names {outside}, outside this backbone's 0..{n_layers - 1}")
    if max(args.rank_ladder) > d_model:
        raise ValueError(
            f"the ladder reaches rank {max(args.rank_ladder)} in {d_model} dimensions"
        )
    print(f"[shape] {n_layers}L x {d_model}d, tensor {args.tensor}, dtype {INFERENCE_DTYPE}")

    cohorts = {mode: cohort_for(mode, args) for mode in args.modes}
    groups = {}
    for mode, cohort in cohorts.items():
        unit = "residues" if mode == "protein" else "characters"
        assignment, grouping = near_duplicate_groups(cohort.records, unit=unit)
        groups[mode] = {"ids": assignment, "record": grouping, "unit": unit}
        print(
            f"[cohort] {mode:8s} {len(cohort.records)} records in "
            f"{int(np.unique(assignment).size)} near-duplicate groups"
        )

    captured = {}
    for mode in args.modes:
        captured[mode] = occupancy_for_mode(
            handles[mode],
            cohorts[mode],
            groups[mode]["ids"],
            args=args,
            layers=args.layers,
            d_model=d_model,
            rule=rule,
            log=log,
        )

    position_counts = {mode: captured[mode]["sampling"]["n_positions"] for mode in args.modes}
    if len(set(position_counts.values())) != 1:
        raise RuntimeError(
            f"the modes were measured at unequal position counts {position_counts}; "
            "the whole comparison rests on them being equal"
        )

    occupancy: dict[str, Any] = {}
    bases: dict[str, dict[int, torch.Tensor]] = {}
    eigen_gaps: dict[str, Any] = {}
    # One eigendecomposition per (mode, layer), reused by the spectrum summary, the
    # basis and every rung's gap. A 4,096 x 4,096 float64 eigensolve costs seconds,
    # and recomputing it per rung would spend minutes per cell on a quantity that
    # does not depend on the rung.
    spectra: dict[str, dict[int, torch.Tensor]] = {
        mode: {
            int(layer): torch.linalg.eigvalsh(captured[mode]["covariances"][int(layer)])
            for layer in args.layers
        }
        for mode in args.modes
    }
    for mode in args.modes:
        readable = bool(captured[mode]["sampling"]["occupancy_readable"])
        occupancy[mode] = {
            "sampling": captured[mode]["sampling"],
            # The suffix is load-bearing: `assert_per_layer_fields` walks the whole
            # payload and checks every key ending in `_per_site`, so a per-layer
            # vector under any other name is unguarded (L32, Appendix B rule 33).
            "occupancy_per_site": [
                occupancy_record(
                    captured[mode]["covariances"][int(layer)],
                    spectra[mode][int(layer)],
                    n_positions=position_counts[mode],
                    d_model=d_model,
                    readable=readable,
                    isotropic=args.isotropic_control,
                    seed=args.position_seed + int(layer),
                )
                for layer in args.layers
            ],
        }
        bases[mode] = {
            int(layer): ms.principal_basis(
                captured[mode]["covariances"][int(layer)], max(args.rank_ladder)
            )
            for layer in args.layers
        }
        eigen_gaps[mode] = {
            "eigen_gap_per_site": [
                [ms.eigen_gap(spectra[mode][int(layer)], rank) for rank in args.rank_ladder]
                for layer in args.layers
            ]
        }

    damage_batches = {}
    for mode in args.modes:
        taken: list[dict[str, Any]] = []
        total = 0
        for entry in captured[mode]["batches"]:
            if total >= args.damage_records:
                break
            taken.append(entry)
            total += int(entry["n_records"])
        damage_batches[mode] = taken
        print(f"[damage] {mode:8s} {total} records over {len(taken)} batches")

    vocabulary = int(model.config.vocab_size)
    invariants: dict[str, Any] = {}
    cells: dict[str, Any] = {}
    # Counted rather than derived, so a successor run can size a campaign off this
    # artefact instead of off an arithmetic guess. One "pass" is one sweep of the
    # damage cohort under one intervention.
    timing: dict[str, Any] = {
        "capture_s_per_mode": {
            mode: captured[mode]["sampling"]["elapsed_s"] for mode in args.modes
        },
        "n_damage_passes": 0,
        "n_damage_batches": 0,
    }
    damage_started = time.time()
    for mode in behavioural:
        ms.assert_behavioural_read(measurability[mode])
        handle = handles[mode]
        batches = damage_batches[mode]
        clean = scored_pass(
            handle, batches, layer=None, basis=None, zero_block=False,
            label=f"{mode}:clean", vocabulary=vocabulary,
        )
        timing["n_damage_passes"] += 1
        timing["n_damage_batches"] += len(batches)
        reference_unigram = ms.cohort_unigram_reference(clean.target_ids, vocabulary)
        per_site = []
        invariant_site = []
        for layer in args.layers:
            def logits_for_basis(
                basis: torch.Tensor | None, layer: int = int(layer)
            ) -> torch.Tensor:
                """One probe batch's scored logits, with or without an interceptor.

                ``None`` binds **no hook at all**, which is what makes the null
                invariant a comparison between two code paths rather than between a
                function and itself.
                """

                probe = batches[0]["batch"]
                if basis is None:
                    context: Any = nullcontext()
                else:

                    def substitute(
                        at: int, block_input: torch.Tensor, block_output: torch.Tensor
                    ) -> torch.Tensor | None:
                        return ms.project_out(block_output, basis) if at == layer else None

                    context = handle.block_intercept(substitute)
                with torch.no_grad(), context:
                    logits, _, mask = handle.scored_logits(probe)
                return logits.reshape(-1, logits.shape[-1])[mask.reshape(-1)].float()

            invariant_site.append(
                ms.intervention_invariants(
                    logits_for_basis,
                    d_model=d_model,
                    rank=max(args.rank_ladder),
                    layer=int(layer),
                    seed=args.position_seed + int(layer),
                    tolerance=rule.logit_tolerance,
                    device=handle.device,
                )
            )
            full = scored_pass(
                handle, batches, layer=int(layer), basis=None, zero_block=True,
                label=f"{mode}:full:L{layer}", vocabulary=vocabulary,
            )
            timing["n_damage_passes"] += 1
            timing["n_damage_batches"] += len(batches)
            full_damage = ms.unigram_decomposition(
                clean, full, seed=args.bootstrap_seed, n_bootstrap=args.n_bootstrap
            )
            control_generator = torch.Generator(device="cpu").manual_seed(
                int(args.position_seed) + 1000 + int(layer)
            )
            rungs = []
            for rank in args.rank_ladder:
                entry: dict[str, Any] = {"rank": int(rank)}
                passes: dict[str, ms.ScoredPass] = {}
                for fit_mode in args.modes:
                    passes[fit_mode] = scored_pass(
                        handle, batches, layer=int(layer),
                        basis=bases[fit_mode][int(layer)][:, :rank], zero_block=False,
                        label=f"{mode}:fit-{fit_mode}:L{layer}:r{rank}",
                        vocabulary=vocabulary,
                    )
                    entry[f"fit_{fit_mode}"] = ms.unigram_decomposition(
                        clean, passes[fit_mode], seed=args.bootstrap_seed,
                        n_bootstrap=args.n_bootstrap,
                    )
                random_basis = ms.random_orthonormal_basis(
                    d_model, int(rank), generator=control_generator, device=handle.device
                )
                random_pass = scored_pass(
                    handle, batches, layer=int(layer), basis=random_basis,
                    zero_block=False, label=f"{mode}:random:L{layer}:r{rank}",
                    vocabulary=vocabulary,
                )
                timing["n_damage_passes"] += len(args.modes) + 1
                timing["n_damage_batches"] += (len(args.modes) + 1) * len(batches)
                entry["random_control"] = ms.unigram_decomposition(
                    clean, random_pass, seed=args.bootstrap_seed,
                    n_bootstrap=args.n_bootstrap,
                )
                other = [name for name in args.modes if name != mode]
                if other:
                    entry["own_minus_other_residual"] = ms.damage_interval(
                        passes[mode].conditional_nll_nats,
                        passes[other[0]].conditional_nll_nats,
                        clean.target_ids,
                        clean.group_ids,
                        seed=args.bootstrap_seed,
                        n_bootstrap=args.n_bootstrap,
                    )
                    entry["other_mode"] = other[0]
                else:
                    entry["own_minus_other_residual"] = ms.SINGLE_MODE_RUN
                entry["own_minus_random_residual"] = ms.damage_interval(
                    passes[mode].conditional_nll_nats,
                    random_pass.conditional_nll_nats,
                    clean.target_ids,
                    clean.group_ids,
                    seed=args.bootstrap_seed,
                    n_bootstrap=args.n_bootstrap,
                )
                rungs.append(entry)
                print(
                    f"[{mode:7s} L{int(layer):2d} r{int(rank):4d}] total "
                    f"{entry[f'fit_{mode}']['total_damage_nats']:+.4f}  residual "
                    f"{entry[f'fit_{mode}']['residual_damage_nats']:+.4f}  random "
                    f"{entry['random_control']['total_damage_nats']:+.4f}",
                    flush=True,
                )
            necessity = ms.necessary_rank(
                [int(rank) for rank in args.rank_ladder],
                [rung[f"fit_{mode}"]["total_damage_nats"] for rung in rungs],
                full_damage["total_damage_nats"],
                rule,
            )
            increments = [
                {
                    "from_rank": int(args.rank_ladder[index - 1]) if index else 0,
                    "to_rank": int(args.rank_ladder[index]),
                    "total_damage_increment_nats": (
                        rungs[index][f"fit_{mode}"]["total_damage_nats"]
                        - (rungs[index - 1][f"fit_{mode}"]["total_damage_nats"] if index else 0.0)
                    ),
                    "residual_damage_increment_nats": (
                        rungs[index][f"fit_{mode}"]["residual_damage_nats"]
                        - (
                            rungs[index - 1][f"fit_{mode}"]["residual_damage_nats"]
                            if index
                            else 0.0
                        )
                    ),
                }
                for index in range(len(rungs))
            ]
            per_site.append(
                {
                    "layer": int(layer),
                    "full_block_ablation": full_damage,
                    "necessary_rank": necessity,
                    "rungs": rungs,
                    "necessity_ranked_increments": increments,
                }
            )
        cells[mode] = {
            "clean_nll_nats": float(np.mean(clean.nll_nats)),
            "clean_context_information_against_model_marginal_nats": float(
                -np.mean(clean.conditional_nll_nats)
            ),
            "clean_context_information_note": ms.MODEL_MARGINAL_CONTEXT_INFORMATION_NOTE,
            "declared_context_information_nats": measurability[mode][
                "context_information_nats"
            ],
            "cohort_unigram_reference": reference_unigram,
            "n_positions": int(clean.target_ids.size),
            "n_groups": int(np.unique(clean.group_ids).size),
            "necessity_per_site": per_site,
        }
        invariants[mode] = {"invariants_per_site": invariant_site}

    timing["damage_s"] = round(time.time() - damage_started, 1)
    verdicts: list[Any] = []
    overlaps: list[Any] = []
    for index, layer in enumerate(args.layers):
        if len(behavioural) < 2:
            overlaps.append(ms.SINGLE_MODE_RUN if len(args.modes) < 2 else ms.BEHAVIOURAL_READ_REFUSED)
            verdicts.append(
                {
                    "layer": int(layer),
                    "verdict": ms.BEHAVIOURAL_READ_REFUSED,
                    "reading": (
                        "fewer than two modes of this checkpoint carry a behavioural "
                        "read on this cohort, so there is no cross-mode comparison to "
                        "decide. " + ms.UNMEASURABLE_MODE_EVIDENCE
                    ),
                }
            )
            continue
        chosen = {
            mode: cells[mode]["necessity_per_site"][index]["necessary_rank"][
                "necessary_rank"
            ]
            for mode in behavioural
        }
        if any(rank is None for rank in chosen.values()):
            overlaps.append(
                {
                    "status": ms.NO_NECESSARY_SUBSPACE,
                    "necessary_ranks": {mode: chosen[mode] for mode in behavioural},
                    "reason": (
                        "at least one mode has no necessary subspace at this layer: "
                        "the ladder did not reach the necessity fraction of its own "
                        "full-block ablation, or that ablation was not attainable. "
                        "There is nothing to compute an overlap between, and this is "
                        "a statement about the site and the ladder rather than about "
                        "either mode's measurability"
                    ),
                }
            )
            verdicts.append(
                ms.layer_verdict(
                    layer=int(layer),
                    modes=behavioural,
                    own={
                        mode: cells[mode]["necessity_per_site"][index]["rungs"][-1][
                            f"fit_{mode}"
                        ]
                        for mode in behavioural
                    },
                    asymmetry={
                        mode: cells[mode]["necessity_per_site"][index]["rungs"][-1][
                            "own_minus_other_residual"
                        ]
                        for mode in behavioural
                    },
                    overlap=None,
                    attainable={
                        mode: cells[mode]["necessity_per_site"][index][
                            "necessary_rank"
                        ]["attainable"]
                        for mode in behavioural
                    },
                    invariants_held=True,
                    rule=rule,
                    statistic=args.overlap_statistic,
                )
            )
            continue
        overlap = ms.subspace_overlap(
            bases[behavioural[0]][int(layer)][:, : chosen[behavioural[0]]],
            bases[behavioural[1]][int(layer)][:, : chosen[behavioural[1]]],
            seed=args.position_seed + 7 + int(layer),
            chance_draws=args.chance_draws,
        )
        overlap["modes"] = list(behavioural)
        overlap["ranks"] = {mode: int(chosen[mode]) for mode in behavioural}
        overlap["eigen_gap"] = {
            mode: ms.eigen_gap(spectra[mode][int(layer)], int(chosen[mode]))
            for mode in behavioural
        }
        overlaps.append(overlap)
        own = {}
        asymmetry = {}
        for mode in behavioural:
            rung = next(
                entry
                for entry in cells[mode]["necessity_per_site"][index]["rungs"]
                if entry["rank"] == chosen[mode]
            )
            own[mode] = rung[f"fit_{mode}"]
            asymmetry[mode] = rung["own_minus_other_residual"]
        verdicts.append(
            ms.layer_verdict(
                layer=int(layer),
                modes=behavioural,
                own=own,
                asymmetry=asymmetry,
                overlap=overlap,
                attainable={
                    mode: cells[mode]["necessity_per_site"][index]["necessary_rank"][
                        "attainable"
                    ]
                    for mode in behavioural
                },
                invariants_held=all(
                    invariants[mode]["invariants_per_site"][index][
                        "random_projection_max_logit_gap"
                    ]
                    > rule.logit_tolerance
                    for mode in behavioural
                ),
                rule=rule,
                statistic=args.overlap_statistic,
            )
        )

    return {
        "kind": "joint_checkpoint_two_modes",
        "checkpoint": STAGE25.checkpoint_record(
            resolved, Path(args.checkpoint), facts, reference, role="joint"
        ),
        # The resolved rendering facts come from whichever handle resolved one --
        # text mode does not resolve, by declaration, so a run naming text first
        # would otherwise record the family's declaration without the ids it was
        # verified against.
        "rendering": next(
            (
                handle.tokenisation.facts()
                for handle in handles.values()
                if handle.tokenisation is not None
            ),
            {
                "name": declaration.name,
                "note": declaration.note,
                "resolved": False,
                "reason": "no protein mode ran, so the rendering was never resolved "
                "against this tokenizer",
            },
        ),
        "tensor": STAGE25.tensor_declaration(reference, args.tensor),
        "modes_measured": list(args.modes),
        "modes_with_a_behavioural_read": list(behavioural),
        "mode_measurability": measurability,
        "matched_cohort": {
            "records_per_mode": int(args.records),
            "damage_records_per_mode": int(args.damage_records),
            "positions_per_record": int(args.positions_per_record),
            "positions_per_mode": position_counts,
            "matched_exactly_on": [
                "records used",
                "positions per record",
                "total scored positions",
            ],
            "not_matched": [
                "the rendered input length: a protein record of this band is shorter "
                "than an OpenWebText document truncated at --max-tokens, so the "
                "context preceding a scored position is not the same length in the "
                "two modes. The realised distributions are reported per mode",
                "the symbol unit: a protein scored token carries about 1.54 residues "
                "on this rendering and a text token about four characters, which is "
                "L23's incommensurability. Every magnitude here is per TOKEN and no "
                "cross-mode magnitude comparison is a per-symbol one",
                "the corpus: Swiss-Prot and OpenWebText are two populations and "
                "nothing makes them one",
            ],
            "per_mode": {
                mode: {
                    "corpus": str(corpus_location(joint_mode_corpus(mode))),
                    "cohort_digest": cohorts[mode].digest,
                    "cohort_provenance_digest": cohorts[mode].provenance_digest,
                    "sampling_record": cohorts[mode].metadata["sampling"],
                    "band": [cohorts[mode].min_symbols, cohorts[mode].max_symbols],
                    "near_duplicate_grouping": groups[mode]["record"],
                    "n_groups": int(np.unique(groups[mode]["ids"]).size),
                    "input_rendering": handles[mode].rendering_note,
                    "scored_positions": handles[mode].scoring_note,
                    **captured[mode]["sampling"],
                }
                for mode in args.modes
            },
        },
        "timing": {
            **timing,
            "seconds_per_damage_pass": (
                timing["damage_s"] / timing["n_damage_passes"]
                if timing["n_damage_passes"]
                else None
            ),
            "seconds_per_damage_batch": (
                timing["damage_s"] / timing["n_damage_batches"]
                if timing["n_damage_batches"]
                else None
            ),
            "note": (
                "measured on this run's own device and cohort. A pass is one sweep "
                "of the damage cohort under one intervention; the bootstrap and the "
                "chance draws are inside damage_s and are CPU work, so a campaign "
                "sized on seconds_per_damage_batch alone under-counts them"
            ),
        },
        "occupancy": occupancy,
        "eigen_gaps": eigen_gaps,
        "necessity": cells,
        "intervention_invariants": invariants,
        "overlap_per_site": overlaps,
        "verdict_per_site": verdicts,
    }


LIMITATIONS: dict[str, Any] = {
    "objective_scope": (
        "an Objective 1/2 measurement of what a model does. Nothing here is an "
        "Objective-3 knowledge claim and audit section 7.0's recombination ceiling "
        "is not this stage's admission rule -- that section says so itself. No "
        "number here says anything about biological knowledge"
    ),
    "one_tensor_one_site_class": (
        "everything is measured on each layer's feed-forward output, which is the "
        "term that block adds to the residual stream. It is NOT the residual stream "
        "itself and it is not the attention contribution, which this layout has "
        "already added by the time the feed-forward runs. A direction a mode needs "
        "in the residual stream but not in any block's write to it is invisible here"
    ),
    "occupancy_statistic_dependence": ms.OCCUPANCY_NOTE,
    "eigen_order_identifiability": ms.EIGEN_ORDER_IDENTIFIABILITY_NOTE,
    "no_token_alignment_required": ms.NO_ALIGNMENT_NOTE,
    "unigram_estimator": ms.UNIGRAM_ESTIMATOR_NOTE,
    "resampling_unit": ms.GROUP_UNIT_NOTE,
    "saturated_rungs": ms.SATURATED_RUNG_NOTE,
    "nested_ablation_only": (
        "the ladder ablates the top-r principal directions, so necessity is read as "
        "a function of OCCUPANCY rank. A direction that a mode needs and barely "
        "varies along sits low in that order and is reached only at a high rung; "
        "this stage does not search for the most necessary rank-r subspace, which "
        "would be a different and much more expensive estimand"
    ),
    "damage_cohort_is_a_prefix": (
        "the ablation passes run over a prefix of the occupancy cohort's own seeded "
        "draw rather than over all of it, because their cost is proportional to "
        "records while the covariance's precision is proportional to positions. It "
        "is a subsample of one population and its size is reported; it is not a "
        "second draw, and the skip-offset sensitivity Appendix B rule 1 asks for is "
        "a second run"
    ),
    "single_draw": (
        "one cohort draw at one seed per mode, one checkpoint per run. Nothing here "
        "is a replicate"
    ),
    "pre_registration_is_not_admission": ms.PRE_REGISTRATION_SCOPE,
}


def main() -> None:
    args = build_parser().parse_args()
    resolve(args)
    args.out.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pre_registration": {
            "entry": ms.PRE_REGISTRATION,
            "status": ms.PRE_REGISTRATION_STATUS,
            "scope": ms.PRE_REGISTRATION_SCOPE,
            "required_never_defaulted_flags": [
                "--layers",
                "--overlap-statistic",
                "--decision-rule",
                "--modes",
                "--checkpoint",
                "--rendering",
                "--context-information",
            ],
            "decision_rule": ms.decision_rule(args.decision_rule).record(),
            "rank_ladder": [int(rank) for rank in args.rank_ladder],
            "overlap_statistic": args.overlap_statistic,
            "overlap_statistic_definitions": dict(ms.OVERLAP_STATISTIC_DEFINITIONS),
            "verdicts": {name: ms.VERDICT_READINGS[name] for name in ms.VERDICTS},
            "rule": (
                "the thresholds are constants of the NAMED rule and cannot be passed "
                "on the command line, so what counts as a result is fixed before the "
                "result exists. The layer set, the modes, the overlap statistic and "
                "the rule are required and never defaulted; resolve() names every "
                "missing one"
            ),
        },
        "settings": {
            key: (
                str(value)
                if isinstance(value, Path)
                else list(value)
                if isinstance(value, tuple)
                else value
            )
            for key, value in vars(args).items()
            if not (args.synthetic and key in CAMPAIGN_ONLY_FLAGS)
        },
        "provenance": {
            "runner": {
                "path": "scripts/transfer/38_mode_subspaces.py",
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "modules": {name: sha256_file(REPO_ROOT / name) for name in PROVENANCE_MODULES},
        },
        "estimand": (
            "per layer and per mode: the spectrum of the centred covariance of that "
            "mode's activations at the layer's feed-forward output (occupancy); the "
            "nats-per-token likelihood damage from projecting that mode's top-r "
            "principal directions out of the same tensor, decomposed per position "
            "into the shift in the model's own held-out marginal and the residual "
            "context information (necessity); the same ablation evaluated in the "
            "other mode (driveability); and the principal-angle overlap of the two "
            "modes' necessary subspaces against the chance level for two random "
            "subspaces of the same dimensions in the same ambient dimension"
        ),
        "limitations": LIMITATIONS,
    }

    if args.synthetic:
        payload.update(run_synthetic(args))
        destination = args.out / (
            f"mode_subspaces__synthetic_check__{args.decision_rule}.json"
        )
    else:
        result = run_campaign(args)
        payload.update(result)
        assert_per_layer_fields(payload, n_sites=len(args.layers))
        clean_name = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(args.checkpoint).resolve().name)
        destination = args.out / (
            "mode_subspaces__"
            + clean_name
            + f"__{args.rendering}"
            + "__"
            + "-".join(args.modes)
            + "__L"
            + "-".join(str(int(layer)) for layer in args.layers)
            + f"__{args.tensor}.json"
        )

    write_json(destination, payload)
    print()
    if args.synthetic:
        print(f"[synthetic] certificate {payload['certificate']['certificate']}")
        for name, cell in payload["verdict_attainability"]["cells"].items():
            print(
                f"[attainable] {name:14s} expected {cell['expected_verdict']:18s} "
                f"got {cell['verdict']}"
            )
    else:
        for entry in payload["verdict_per_site"]:
            print(f"[verdict] L{entry['layer']:2d}  {entry['verdict']}")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
