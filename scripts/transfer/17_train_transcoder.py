#!/usr/bin/env python3
"""Train a CLT and a PLT on one decoder's blocks under identical conditions.

ProGenMech's central claim is that a cross-layer transcoder beats a per-layer one
at replacing ProGen3's MoE blocks. Their CLT weights are unobtainable (HTTP 403
on that directory alone) and their trainer needs packages an offline pod cannot
have, so this repository could gate their baseline and not their headline.

This stage removes that bound by training both ourselves. That is a **stronger**
comparison than their released weights would have permitted: their PLT was
trained on their corpus with their schedule, so a CLT-versus-PLT difference read
across a released PLT and a locally trained CLT would confound architecture with
everything else. Here the two runs differ in exactly one thing -- whether a
latent may write to layers downstream of the one it was read from -- and share
their data order, their seed, their step budget and their optimiser.

**This is a bounded reproduction and says so.** The step budget is declared, not
their full 5M-sequence epoch, and the deliverable is the CLT-minus-PLT
difference at equal budget together with the curve that shows whether the
comparison had converged enough to carry it. An absolute NMSE against their
published 2.46 is not claimed and would not be meaningful at this scale.

Activations are captured at the tensors ProGenMech's own collector uses: the
block's input, and its output before the residual add. Special tokens are
excluded from the objective, as in theirs, because a transcoder scored on padding
is scored on the easiest positions in the batch.

**``--arm`` selects which decoder is read, and that is why this stage is not
ProGen3-only.** Every replacement number this programme owns was measured on one
sparse-MoE protein decoder, so a failure has no attribution: protein, MoE and
transcoder replacement are collinear at n=1. A dense arm supplies the two
missing controls -- ``gpt2-large`` is the text control standing rule 2 requires a
gate be shown attainable on, and ``protgpt2`` is a dense *protein* decoder of
identical architecture, depth, width, vocabulary and parameter count (audit §2's
matched modality pair). :mod:`src.transfer.replaceable` holds the one adapter
that makes those arms present the interface this stage already consumed, and
verifies that a dense block carries the same estimand rather than assuming it.

Each arm streams the corpus its own evaluation cohort is drawn from, rendered in
the format it was trained on, both through the panel declaration in
``src/transfer/arms.py``. ProGen3 keeps UniRef50, which is what its published
runs used.

**``--joint-checkpoint`` plus ``--rendering`` plus ``--mode`` reaches the one
comparison a panel arm cannot make.** Every dictionary comparison this programme
owns is across *different models*, so a difference between two of them is a
difference of architecture, scale, lineage and training data at once -- L25's
shape, where a cross-layer transcoder's win at equal width turned out to be a
3.25x parameter advantage. Two per-layer transcoders trained on **one joint
checkpoint**, one on its text mode and one on its protein mode, share every one
of those: the weights are the same object in both, so a difference cannot be
attributed to any of them. A joint checkpoint is reached by path and not by name
for the reason ``21_joint_mode_qualification.py`` gives -- a checkpoint that has
not passed that stage must not be in the panel -- and this stage loads it, renders
it and locates its scored positions through that stage's own machinery and
:class:`src.transfer.replaceable.JointReplaceable`, so the tokens a mode trains
on are the tokens that mode is qualified and scored on.

**The comparison is only worth making if nothing else moves, so the trainer
declares what must not.** ``--train-tokens`` sets the budget in scored tokens
rather than steps, because a text record and a protein record carry different
numbers of scored positions and equal step counts are therefore equal *schedules*
over unequal data. That budget, the layer count, the dictionary width, ``k``, the
backbone digest and the held-out evaluation budget are written into both the
checkpoint and the JSON record as a
:class:`src.transfer.transcoders.MatchedTraining` declaration with its own
digest, and ``15_replacement_faithfulness.py --matched-against`` refuses a pair
that disagrees on any of them.

Output is a checkpoint plus a JSON record, and the checkpoint is written in the
shape ``15_replacement_faithfulness.py`` can load, so a trained transcoder goes
straight to the faithfulness gate with no conversion step in between.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# The stage directory itself, so `panel_contract` imports under every invocation
# rather than only when the caller happens to run from scripts/transfer.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from panel_contract import CAMPAIGN_PANEL  # noqa: E402
from src.transfer import joint_modes  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    AA20,
    REPO,
    corpus_location,
    iter_corpus_records,
)
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
from src.transfer.transcoders import (  # noqa: E402
    DEAD_STEPS_SEQUENCES,
    MATCHED_TRAINING_KEY,
    FiringCensus,
    MatchedTraining,
    Transcoder,
    TranscoderConfig,
    TrainingRecord,
)


def _load_stage(filename: str) -> Any:
    """Import a stage whose module name starts with a digit.

    ``21_joint_mode_qualification.py`` owns the loading of a joint checkpoint --
    the tokenizer read before the weights so a wrong checkpoint/family pair fails
    in a second, and the shape and dtype read back off the built model rather
    than echoed from the request. Imported rather than restated so that the
    checkpoint this stage trains against is the one that stage qualified
    (Appendix B rule 12), exactly as ``23_perturbation_sensitivity.py`` does.
    """

    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(f"_transfer_stage_{filename[:2]}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE21 = _load_stage("21_joint_mode_qualification.py")

#: Records buffered before each shuffle. The held-out draw must clear a whole
#: number of these, so the block size and the offset that skips past it cannot be
#: allowed to disagree -- it was a literal in two places and they did.
SHUFFLE_BLOCK = 8192

#: How many held-out candidates are drawn per sequence the run will keep.
#:
#: The near-duplicate screen below drops candidates, so the draw has to have
#: something left. Four is what the measured drop rate affords with margin: on
#: the Swiss-Prot pool the same relation removed 41.4% of held-out records at the
#: 95%-identity boundary, so a 4x draw survives a drop rate well past twice that
#: and refuses rather than quietly shrinking the evaluation budget if it does not.
HELD_OUT_OVERSAMPLE = 4

SCHEMA_VERSION = "r2_transfer_transcoder_training_v1"
DEFAULT_OUT = REPO / "results/transfer/transcoder_training"

#: This stage's residue band, and it is **ours rather than theirs**. 1022 is the
#: length their activation-collection script uses; their data module's own length
#: truncation is commented out, and nothing in their repository sets a lower
#: bound. Recorded as a declared choice rather than as inherited provenance,
#: because a band attributed to someone else cannot be revised without appearing
#: to break a reproduction.
MAX_RESIDUES = 1022
MIN_RESIDUES = 32

#: The same band, lowered to what an EC-conditioned arm can actually render.
#:
#: ZymCTRL's context is 1024 positions and its rendering spends ten of them on
#: the prompt and terminator -- seven for the EC number, then ``<sep>``,
#: ``<start>`` and ``<end>`` -- so a 1022-residue record needs 1032 and cannot be
#: fed whole. Truncating it instead is not an option that stays quiet: the
#: scored-content span is delimited by ``<end>``, so a record that loses its
#: terminator has no span, and :func:`src.transfer.scoring.sequence_target_mask`
#: raises. The ceiling is therefore ``1024 - 10``, which is the largest band this
#: arm can carry and is within 0.05% of the residue band the other protein
#: sources use (118 of 242,968 eligible records fall outside it). Declared here,
#: beside the band it modifies, because Appendix B rule 13 asks a stage to say
#: which population it drew rather than to leave it inferable from a token cap.
ZYMCTRL_WRAPPER_TOKENS = 10
ZYMCTRL_CONTEXT = 1024
ZYMCTRL_MAX_RESIDUES = ZYMCTRL_CONTEXT - ZYMCTRL_WRAPPER_TOKENS

#: The floor of this stage's text band, in characters -- the unit an English
#: corpus is made of, as residues are the unit a protein corpus is made of.
#: Deliberately :func:`src.transfer.arms.text_cohort`'s own floor, so that the
#: population a text transcoder is trained on is the population the faithfulness
#: stage later scores it on. There is no ceiling: a long document is truncated at
#: tokenisation, which is what the cohort path does, and discarding it instead
#: would be a different corpus (see :func:`iter_corpus_records`).
MIN_CHARACTERS = 800

#: The eligibility band each corpus is streamed under, in that corpus's own
#: symbol unit. One declaration keyed by the source names
#: :data:`src.transfer.arms.CORPUS_SOURCES` uses, rather than a modality branch
#: at the point of use.
CORPUS_BAND: dict[str, tuple[int, int | None]] = {
    "uniref50": (MIN_RESIDUES, MAX_RESIDUES),
    "swissprot": (MIN_RESIDUES, MAX_RESIDUES),
    "zymctrl_ec": (MIN_RESIDUES, ZYMCTRL_MAX_RESIDUES),
    "openwebtext": (MIN_CHARACTERS, None),
}


def joint_protein_band(
    tokenisation: joint_modes.JointTokenisation,
    *,
    max_tokens: int,
    protein_context: str | None,
) -> tuple[int, int]:
    """The residue band a joint protein mode may be streamed under, measured not guessed.

    ZymCTRL's ceiling above is ``1024 - 10`` because that rendering's wrapper is
    ten tokens on a known tokenizer. A joint family's wrapper is not knowable in
    advance -- ``Seq=<`` is three pieces of the LLaMA-2 vocabulary and a document
    context adds however many its template costs -- so it is **measured** here,
    by rendering a one-residue sequence through the same declaration the training
    stream will use, and the ceiling is the token cap minus it.

    That is an exact upper bound rather than an estimate. Every rendering this
    family accepts spells its sequence on token boundaries -- a residue that
    merged into a delimiter or into the context is refused by
    :func:`src.transfer.joint_modes.scored_target_positions`, not truncated -- so
    a rendered record is exactly the wrapper plus its scored span, and a scored
    span is never longer than the sequence is in residues. A record inside the
    band therefore cannot exceed the cap, which is what turns
    :meth:`src.transfer.replaceable.JointReplaceable.render`'s refusal from a
    hazard that can end a thirty-hour run at hour twenty into one that cannot
    fire at all.
    """

    probe = tokenisation.render(AA20[0], context=protein_context)
    wrapper = len(probe.token_ids) - probe.n_residues
    ceiling = min(MAX_RESIDUES, max_tokens - wrapper)
    if ceiling < MIN_RESIDUES:
        raise ValueError(
            f"--max-tokens {max_tokens} leaves {ceiling} residues once this "
            f"rendering's {wrapper}-token wrapper is paid for, which is below the "
            f"{MIN_RESIDUES}-residue floor: no record of this corpus could be "
            "rendered whole"
        )
    return MIN_RESIDUES, ceiling


def stream_records(
    records: Callable[[], Iterator[Any]], *, seed: int, skip: int, limit: int | None = None
) -> Iterator[Any]:
    """Eligible corpus records in a seeded shuffled order.

    A biological corpus is ordered by cluster and a web corpus by shard, so in
    both cases consecutive records are related and a prefix is a region rather
    than a sample (Appendix B rule 1). A full seeded permutation of 60M records
    is not affordable here, so the draw is a shuffled reservoir: records are read
    in file order into blocks and each block is permuted before it is emitted.
    That breaks the local ordering, which is what the rule is about, and the
    block size is recorded so the residual correlation is a declared property
    rather than an unstated one.

    ``records`` is a factory rather than an iterator because this is called twice
    -- once for the held-out draw and once for training -- and the two must be
    independent passes over the corpus rather than two halves of one.
    """

    rng = np.random.default_rng(seed)
    block: list[Any] = []
    emitted = 0
    seen = 0
    for record in records():
        seen += 1
        if seen <= skip:
            continue
        block.append(record)
        if len(block) < SHUFFLE_BLOCK:
            continue
        for index in rng.permutation(len(block)):
            yield block[index]
            emitted += 1
            if limit is not None and emitted >= limit:
                return
        block = []
    for index in rng.permutation(len(block)):
        yield block[index]
        emitted += 1
        if limit is not None and emitted >= limit:
            return


@torch.no_grad()
def capture(
    model: ReplaceableModel, records: list[tuple[str, str | None]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Block inputs, outputs and a scored-token mask for one batch.

    Returns ``(inputs, outputs, mask)`` with the layer axis first. The mask
    excludes every non-content position: terminus and padding tokens are
    trivially predictable and would flatter any transcoder scored on them, and on
    a conditioned arm the EC prompt is excluded for the stronger reason that it
    is not content at all.

    A record arrives as ``(record, conditioning_label)`` from
    :func:`src.transfer.arms.iter_corpus_records`; the label is ``None`` for every
    unconditioned corpus and the renderer refuses either half of the mismatch.
    """

    batch = model.batch(
        model.render(
            [record for record, _ in records], ec_labels=[label for _, label in records]
        )
    )
    # The batch now carries everything the rendering located, so the per-record
    # state a joint checkpoint keeps between the two calls is dead. Dropping it
    # here rather than never is the difference between a bounded trainer and one
    # whose memory grows with its step count; the two implementations that keep
    # no such state see a no-op.
    model.forget_rendered()
    captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def tap(layer: int, block_input: torch.Tensor, block_output: torch.Tensor) -> None:
        captured[layer] = (block_input.detach(), block_output.detach())
        return None

    with model.block_intercept(tap):
        model.run(batch)

    layers = sorted(captured)
    inputs = torch.stack([captured[layer][0] for layer in layers])
    outputs = torch.stack([captured[layer][1] for layer in layers])
    return inputs, outputs, model.content_mask(batch)


def flatten(
    inputs: torch.Tensor, outputs: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep scored positions only, as ``(layers, tokens, d_model)``."""

    keep = mask.reshape(-1)
    n_layers, batch, length, width = inputs.shape
    x = inputs.reshape(n_layers, batch * length, width)[:, keep]
    y = outputs.reshape(n_layers, batch * length, width)[:, keep]
    return x, y


def open_joint_target(
    checkpoint: Path,
    *,
    rendering: str,
    mode: str,
    device: str,
    max_tokens: int,
    protein_context: str | None,
    band: tuple[int, int | None],
) -> tuple[JointReplaceable, dict[str, Any], tuple[int, int | None]]:
    """Load one mode of a joint checkpoint, and the residue band it may stream.

    Extracted from :func:`main` so that a stage which re-reads a dictionary this
    trainer wrote opens the *same* checkpoint through the *same* qualification
    path -- ``21_joint_mode_qualification.py`` owns the load, this owns the
    rendering, and neither is restated (Appendix B rule 12). A second copy of
    this sequence would be a second declaration of which weights and which
    tokenisation a mode means, and the whole point of the matched pair is that
    there is only one.

    Returns the handle, the facts a record states about it, and the band -- which
    the protein mode narrows from the measured rendering wrapper and the token
    cap, and the text mode leaves alone.
    """

    declaration = joint_modes.rendering(rendering)
    print(f"[loader] {checkpoint} as {declaration.name}:{mode} on {device}")
    # The tokenizer alone first, then the rendering, then the weights: a wrong
    # checkpoint/family/mode triple fails before a multi-gigabyte load.
    resolved, tokenizer = STAGE21.load_tokenizer(checkpoint)
    tokenisation = joint_tokenisation(tokenizer, declaration, mode)
    if tokenisation is not None:
        band = joint_protein_band(
            tokenisation, max_tokens=max_tokens, protein_context=protein_context
        )
    backbone, checkpoint_facts = STAGE21.load_model(
        resolved, tokenizer, device=device, dtype="bfloat16"
    )
    checkpoint_facts["requested_path"] = str(checkpoint)
    handle = JointReplaceable(
        model=backbone,
        tokenizer=tokenizer,
        checkpoint=resolved,
        declaration=declaration,
        mode=mode,
        tokenisation=tokenisation,
        max_tokens=max_tokens,
        protein_context=protein_context,
    )
    target = {
        "kind": "joint_checkpoint",
        "rendering_family": declaration.name,
        "mode": mode,
        "checkpoint_facts": checkpoint_facts,
        "rendering": (
            tokenisation.facts()
            if tokenisation is not None
            else {
                "verdict": "NOT_RESOLVED",
                "declared_family": declaration.name,
                "reason": (
                    "the text mode's scored positions are the tokenizer's own "
                    "next-token targets and do not depend on the protein "
                    "rendering, so the declared family is recorded but not "
                    "resolved against this tokenizer. A protein-mode run "
                    "resolves it and is refused when it does not hold"
                ),
            }
        ),
    }
    return handle, target, band


def held_out_cohort(
    records: Callable[[], Iterator[Any]],
    *,
    corpus_seed: int,
    steps: int,
    batch_size: int,
    eval_sequences: int,
    symbol_unit: str,
) -> tuple[list[Any], dict[str, Any], int]:
    """The evaluation cohort, drawn past the training budget and screened.

    Extracted from :func:`main` for the reason the offset itself exists: which
    records a dictionary is held out on is a property of the run, and a stage
    that re-reads that dictionary must draw the same ones or it is measuring a
    different cohort under the same name. One declaration, two callers.

    Returns the cohort, the near-duplicate screen's own record, and the offset it
    was drawn past -- all three, because the screen and the offset are what make
    the cohort held out and a caller that reports one without the others is
    reporting a draw rather than a held-out set.
    """

    blocks_touched = -(-(steps * batch_size) // SHUFFLE_BLOCK)
    held_out_offset = blocks_touched * SHUFFLE_BLOCK
    candidates = list(
        stream_records(
            records,
            seed=corpus_seed,
            skip=held_out_offset,
            limit=eval_sequences * HELD_OUT_OVERSAMPLE,
        )
    )
    if len(candidates) < eval_sequences:
        raise RuntimeError(
            f"the corpus ran out at the held-out offset: {len(candidates)} of "
            f"{eval_sequences} sequences past a skip of {held_out_offset}. "
            "Lower --steps or --eval-sequences rather than evaluating on a "
            "population the training stream also reaches."
        )
    print(
        f"[held-out] screening {len(candidates)} candidates against the "
        f"{held_out_offset} training records, on {symbol_unit}"
    )
    keep, screen = screen_against_training_stream(
        [entry[0] for entry in candidates],
        (
            entry[0]
            for entry in stream_records(
                records, seed=corpus_seed, skip=0, limit=held_out_offset
            )
        ),
        unit=symbol_unit,
    )
    survivors = [entry for entry, kept in zip(candidates, keep) if kept]
    print(
        f"  kept {screen['n_kept']} of {screen['n_candidates']}, "
        f"max containment {screen['max_containment']:.4f}"
    )
    if len(survivors) < eval_sequences:
        raise RuntimeError(
            f"the near-duplicate screen left {len(survivors)} of "
            f"{eval_sequences} held-out sequences from "
            f"{len(candidates)} candidates: this corpus region is too close to "
            "what training reaches. Draw from further out with a larger --steps "
            "headroom, or raise HELD_OUT_OVERSAMPLE -- do not evaluate on the "
            "unscreened draw, which is the leakage L30 measured"
        )
    held_out = survivors[:eval_sequences]
    screen["n_held_out_taken"] = len(held_out)
    screen["oversample_factor"] = HELD_OUT_OVERSAMPLE
    return held_out, screen, held_out_offset


def evaluate(
    model: Transcoder,
    decoder: ReplaceableModel,
    sequences: list[tuple[str, str | None]],
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Held-out NMSE and live basis, on sequences the run never trains on.

    The basis is counted here as well as scored, because the two definitions of a
    live latent answer different questions and only one of them was ever
    recorded. The checkpoint's own ``silent_steps`` reading says a latent fired
    somewhere in the last ``dead_steps`` *training* steps; the census below says
    it fires on the held-out cohort. A dictionary can look complete by the first
    and be far from it by the second, and the basis-adequacy criterion R2.4 is
    gated on reads the basis a diff would actually be taken over (EXP-R2-203).
    """

    model.eval()
    totals = torch.zeros(model.config.num_layers, dtype=torch.float64)
    census = FiringCensus(model.config.num_layers, model.config.d_hidden)
    batches = 0
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            if not chunk:
                continue
            x, y, mask = capture(decoder, chunk)
            x, y = flatten(x, y, mask)
            if x.shape[1] == 0:
                continue
            report = model.objective(x.float(), y.float(), training=False)
            totals += report["nmse_per_layer"].double().cpu()
            census.update(report["fired_per_latent"].cpu(), int(x.shape[1]))
            batches += 1
    model.train()
    per_layer = (totals / max(batches, 1)).tolist()
    return {
        "nmse_per_layer": per_layer,
        "nmse_sum": float(sum(per_layer)),
        "n_batches": batches,
        "n_sequences": len(sequences),
        "live_basis": census.record(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("clt", "plt"), required=True)
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--arm",
        default=PROGEN3_ARM,
        choices=eligible_arms(CAMPAIGN_PANEL),
        help="which decoder's blocks to train against. The eligible set is "
        "composed by src.transfer.replaceable.eligible_arms from three "
        "declarations -- the campaign panel, the architectures that carry this "
        "estimand, and the arms with a measured loader band -- and is not a list "
        "this stage keeps",
    )
    target.add_argument(
        "--joint-checkpoint",
        type=Path,
        default=None,
        help="directory of a joint language-protein checkpoint to train against "
        "instead of a panel arm. A path and not a name: a checkpoint that has not "
        "passed 21_joint_mode_qualification.py must not be in the panel, so there "
        "is nothing for a default to point at. Requires --rendering and --mode. "
        "Distinct from --checkpoint, which relocates ProGen3's weights and means "
        "something else",
    )
    parser.add_argument(
        "--rendering",
        default=None,
        choices=joint_modes.RENDERING_NAMES,
        help="which declared family's input format --joint-checkpoint takes. "
        "Required with it and refused with --arm, whose rendering is declared by "
        "src.transfer.arms.PANEL. The set is composed by src.transfer.joint_modes, "
        "the single place either mode's format is decided",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=JOINT_MODES,
        help="which mode of --joint-checkpoint to train a dictionary on. One mode "
        "per run and one dictionary per mode: the comparison is between two "
        "dictionaries over one set of weights, so they are two runs and their "
        "matched configuration is what 15_replacement_faithfulness.py refuses on. "
        "Refused with --arm, whose mode is the arm's declared modality",
    )
    parser.add_argument(
        "--protein-context",
        default=None,
        help="optional document context a joint checkpoint's protein block is "
        "embedded in, filled into the family's declared template. Omitted means "
        "the bare block, and whichever was used reaches the artefact and sets the "
        "residue band the corpus is streamed under",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="token cap a dense arm's inputs are truncated to. ProGen3 ignores it: "
        "its batch preparer pads to the longest record and its residue band is "
        "declared in residues. A conditioned arm needs a cap that covers its "
        "declared band plus its rendering wrapper -- zymctrl_ec is 1014 residues "
        "plus 10 tokens, so pass --max-tokens 1024; below that the terminator "
        "that delimits the scored span is truncated away and the run refuses. A "
        "joint protein mode derives its residue ceiling from this cap and its "
        "measured wrapper, so it cannot produce a record that does not fit",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument(
        "--train-tokens",
        type=int,
        default=0,
        help="training budget in SCORED TOKENS. 0 -- the default -- runs --steps "
        "steps and is what every invocation predating this flag computed. A "
        "positive value stops at the first step to reach it and refuses the run if "
        "--steps ran out first, which is what lets two modes of one checkpoint see "
        "the same amount of data: a text record and a protein record carry "
        "different numbers of scored positions, so equal steps are equal schedules "
        "over unequal data. --steps then bounds the run and sets the held-out "
        "offset, which stays past everything training reaches",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-sequences", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-hidden", type=int, default=4608)
    parser.add_argument("--k", type=int, default=64)
    # Defaulted from the config declaration rather than restated. A literal 128
    # stood here and silently shadowed TranscoderConfig's 192 at every
    # invocation, so the campaign that reported moving to their effective value
    # ran at the value it was moving away from -- two authoritative defaults for
    # one fact, and the wrong one won (Appendix B rule 12).
    parser.add_argument("--auxk", type=int, default=TranscoderConfig.auxk)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--corpus-seed",
        type=int,
        default=20260806,
        help="seed for the shuffled stream. **Pass the same value to the CLT and "
        "the PLT run**: the comparison is only controlled if both see the same "
        "sequences in the same order",
    )
    return parser


def resolve_target(args: argparse.Namespace) -> None:
    """Refuse an incoherent target before a corpus is opened or a model is loaded.

    Mutates ``args.arm`` to ``None`` on the joint path, deliberately: ``--arm``
    keeps its ProGen3 default so that every panel invocation means exactly what
    it meant before, and leaving that default standing beside a
    ``--joint-checkpoint`` would put ``"arm": "progen3"`` into the settings block
    of a run that never touched ProGen3.
    """

    if args.train_tokens < 0:
        raise ValueError("--train-tokens cannot be negative")
    if args.joint_checkpoint is None:
        for flag, value in (("--rendering", args.rendering), ("--mode", args.mode)):
            if value is not None:
                raise ValueError(
                    f"{flag} describes a joint checkpoint's input format; a panel "
                    "arm's rendering and modality are declared by "
                    "src.transfer.arms.PANEL and are not chosen here"
                )
        if args.protein_context is not None:
            raise ValueError(
                "--protein-context fills a joint family's declared context "
                "template; a panel arm's rendering carries no such template"
            )
        return
    missing = [
        flag
        for flag, value in (("--rendering", args.rendering), ("--mode", args.mode))
        if value is None
    ]
    if missing:
        raise ValueError(
            f"--joint-checkpoint needs {' and '.join(missing)}: the format a joint "
            "checkpoint was trained on and the mode a dictionary is fitted to are "
            "declarations, and a run that guessed either would train a complete "
            "dictionary against a different object (Appendix B rule 4)"
        )
    if args.checkpoint is not None:
        raise ValueError(
            "--checkpoint relocates ProGen3's weights and is meaningless beside "
            "--joint-checkpoint, which names the weights this run reads"
        )
    args.arm = None


def main() -> None:
    args = build_parser().parse_args()
    resolve_target(args)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    joint = args.joint_checkpoint is not None
    # The corpus is resolved and checked before anything reaches a GPU, so a host
    # that has not staged it fails in a second rather than after a checkpoint is
    # loaded. `--corpus` relocates ProGen3's UniRef50 and is refused for the
    # other sources, which are read through their own declared variables. A joint
    # mode's corpus is the one src.transfer.replaceable declares for it, which is
    # also the one 15_replacement_faithfulness.py will score that mode on.
    source = joint_mode_corpus(args.mode) if joint else arm_training_corpus(args.arm)
    low, high = CORPUS_BAND[source]
    corpus = corpus_location(source, path=args.corpus)

    model_handle: ReplaceableModel
    target: dict[str, Any]
    if joint:
        model_handle, target, (low, high) = open_joint_target(
            args.joint_checkpoint,
            rendering=args.rendering,
            mode=args.mode,
            device=args.device,
            max_tokens=args.max_tokens,
            protein_context=args.protein_context,
            band=(low, high),
        )
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

    def records() -> Iterator[tuple[str, str | None]]:
        return iter_corpus_records(
            source, min_symbols=low, max_symbols=high, path=args.corpus
        )

    loader_gate = model_handle.self_check()
    band = loader_gate.get("nll")
    print(
        f"  self-check {loader_gate['verdict']}"
        + ("" if band is None else f", NLL {band:.4f}")
    )
    # Digested before training rather than after it: a checkpoint whose weight
    # files cannot be identified should stop the run in a second, not after
    # thirty hours, and the digest is the field that says WHICH ProLLaMA a mode's
    # dictionary was fitted to -- 'prollama:protein' names a mode and three
    # checkpoints of one lineage answer to it.
    backbone_digest = model_handle.weights_digest()
    target.update(
        {
            "name": model_handle.name,
            "checkpoint": str(model_handle.checkpoint),
            "weights_sha256": backbone_digest,
            "n_layers": model_handle.n_layers,
            "d_model": model_handle.width,
            "loading_note": model_handle.loading_note,
        }
    )

    config = TranscoderConfig(
        num_layers=model_handle.n_layers,
        d_model=model_handle.width,
        d_hidden=args.d_hidden,
        k=args.k,
        auxk=args.auxk,
        # Their threshold is in sequences; the model uses the quotient with the
        # batch. Derived here so --batch-size cannot silently change how long a
        # latent may stay silent while every other declared setting is unmoved.
        dead_steps=max(1, DEAD_STEPS_SEQUENCES // args.batch_size),
        cross_layer=args.architecture == "clt",
        # The model's own name and not --arm, so that a dictionary fitted to one
        # mode of a joint checkpoint declares 'prollama:protein' rather than a
        # null. For a panel arm the two are the same string, which is why
        # 15_replacement_faithfulness.py can refuse a mismatch with one check for
        # both kinds of target. Without it, a text dictionary spliced into the
        # protein mode of the same weights passes every shape check there is --
        # same depth, same width, same checkpoint -- and produces a complete
        # faithfulness artefact for the wrong measurement.
        arm=model_handle.name,
    )
    model = Transcoder(config).to(args.device).float()
    n_parameters = sum(p.numel() for p in model.parameters())
    print(
        f"[model] {config.record()['architecture']}  {n_parameters/1e6:.1f}M parameters  "
        f"{len(model.pairs)} decoder(s)"
    )

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # The held-out set is drawn from beyond everything training will ever reach,
    # not from the head of the same stream.
    #
    # Taking it from the head was disjoint but not representative, and on this
    # corpus the difference is large: the block shuffle permutes *within* an
    # 8192-record window read in file order, so the first block is a region of a
    # cluster-ordered corpus rather than a sample of it. Measured on UniRef50 at
    # this seed, the first block's records mean 394 residues against 878 for
    # blocks 2-7, while the training stream's realised mean was 932 -- so the
    # model was being evaluated on a population 2.4x shorter than the one it was
    # trained on. EXP-R2-135 priced this model's band sensitivity at 4.1x in NLL
    # recovery, so a 2.4x length gap between train and eval is not a detail.
    # Skipping past the whole training budget costs one extra corpus scan and
    # removes both the overlap and the mismatch (Appendix B rule 1, applied to
    # the evaluation draw and not only to the training order).
    #
    # **What it does not remove, stated rather than implied.** The skip is in
    # file order and the shuffle permutes *within* an 8192-record block, so the
    # two sets are disjoint only when the training budget consumes whole blocks.
    # The skip is counted in records read, and the shuffle permutes *within* a
    # block, so an offset that lands mid-block leaves the two sets overlapping: a
    # partially consumed final block hands training a random subset of it while
    # the held-out pool starts partway into that same block. At the campaign's
    # own setting -- 20000 steps of batch 16, so 320,000 records against a block
    # of 8192 -- training emitted 512 records of the 40th block uniformly at
    # random, each held-out record from that block was available to it with
    # probability 512/8192, and the expectation was 16 of the 256 held-out
    # sequences: 6.25% evaluated on what was trained on.
    #
    # Rounding the skip up to a whole number of blocks removes it exactly,
    # because a block is emitted only once it is complete. **EXP-R2-136 and
    # EXP-R2-138 were run before this repair** and carry the leak; it is
    # symmetric across their arms -- same corpus, seed, stream and offset -- so
    # it does not touch the CLT-minus-PLT comparison they claim, and their
    # independent Swiss-Prot replication does not share the cohort at all. It
    # does bound their absolute held-out NMSE, which those entries already
    # decline to compare against any published figure.
    #
    # **And a disjoint offset is still not a held-out set on a protein corpus**
    # (L30, EXP-R2-175). Everything above makes the two sets disjoint *as
    # records*; on Swiss-Prot that is a small part of the property, because the
    # corpus is non-redundant at the level of the entry and not of the sequence.
    # Measured on the pool `25_model_diffing_baselines.py` draws, 41.4% of
    # held-out records keep a relative in the training side at 95% identity or
    # above while only 17.4% are exact -- so a record-level offset leaves most of
    # the leakage, and the held-out NMSE it reports is optimistic by an amount
    # nothing measures. That stage was refusing to start over this and was right
    # to; this one never checked, so the same defect reached the same lineage
    # through a second door.
    #
    # The remedy is the same relation, asked of a stream: the held-out draw is
    # oversampled and every candidate a training record is a near-duplicate of is
    # dropped. It runs on **both** modes and not only on the protein one --
    # attainability before application (Appendix B rule 2), and on a text corpus
    # the screen is expected to drop nothing, which is the demonstration rather
    # than an assumption.
    symbol_unit = "characters" if source == "openwebtext" else "residues"
    held_out, screen, held_out_offset = held_out_cohort(
        records,
        corpus_seed=args.corpus_seed,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_sequences=args.eval_sequences,
        symbol_unit=symbol_unit,
    )
    training = stream_records(records, seed=args.corpus_seed, skip=0, limit=None)

    record = TrainingRecord()
    final: dict[str, Any] | None = None
    started = time.time()
    budget = int(args.train_tokens)
    print(
        f"[train] {args.steps} steps, batch {args.batch_size}"
        + (f", stopping at {budget} scored tokens" if budget else "")
    )
    for step in range(1, args.steps + 1):
        try:
            chunk = [next(training) for _ in range(args.batch_size)]
        except StopIteration:
            # A bare StopIteration out of a list comprehension surfaces far from
            # its cause. The corpus running dry is a sizing fact and says so.
            raise RuntimeError(
                f"the {source!r} stream ran out at step {step} after "
                f"{record.tokens} of {budget or '(no)'} scored tokens: the band "
                f"{[low, high]} does not hold enough records for this budget. "
                "Lower --train-tokens or widen the band"
            ) from None
        x, y, mask = capture(model_handle, chunk)
        x, y = flatten(x, y, mask)
        if x.shape[1] == 0:
            continue
        report = model.objective(x.float(), y.float(), training=True)
        optimiser.zero_grad(set_to_none=True)
        report["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimiser.step()

        record.steps = step
        record.tokens += int(x.shape[1])
        record.sequences += len(chunk)
        # The budget is reached at the FIRST step to cross it, so the realised
        # total overshoots it by at most one batch. That is why the matched
        # declaration compares the budget and records the realised count beside
        # it rather than the other way round.
        reached = bool(budget) and record.tokens >= budget
        if step % args.eval_every == 0 or step == args.steps or reached:
            held = evaluate(model, model_handle, held_out, batch_size=args.batch_size)
            if step == args.steps or reached:
                final = held
            entry = {
                "step": step,
                "train_nmse_sum": float(report["nmse_sum"].detach()),
                "held_out_nmse_sum": held["nmse_sum"],
                "held_out_nmse_per_layer": held["nmse_per_layer"],
                "n_dead": report["n_dead"],
                # The per-layer vector beside the cross-layer scalar. The scalar
                # cannot be un-collapsed after the fact, and a basis-adequacy
                # reading taken from `d_hidden - n_dead/num_layers` is a mean over
                # layers wearing a per-layer name (EXP-R2-203).
                "n_dead_per_layer": report["n_dead_per_layer"],
                "held_out_live_basis": held["live_basis"],
                "active_fraction": report["active_fraction"],
                "tokens": record.tokens,
                "elapsed_s": round(time.time() - started, 1),
            }
            record.history.append(entry)
            print(
                f"  step {step:6d}  train {entry['train_nmse_sum']:7.4f}  "
                f"held-out {entry['held_out_nmse_sum']:7.4f}  dead {entry['n_dead']:5d}  "
                f"{entry['elapsed_s']:.0f}s"
            )
        if reached:
            break

    if budget and record.tokens < budget:
        raise RuntimeError(
            f"--steps {args.steps} ran out after {record.tokens} of {budget} "
            "scored tokens, so this dictionary saw less data than the budget it "
            "declares and could not be matched against the other mode's. Raise "
            "--steps -- it bounds the run and sets the held-out offset, and the "
            "token budget is what stops it"
        )
    if final is None:
        raise RuntimeError(
            "the training loop never reached its final step, so no held-out "
            "evaluation exists; refusing to write a checkpoint with no score"
        )
    matched = MatchedTraining(
        target=model_handle.name,
        backbone_sha256=backbone_digest,
        architecture=config.record()["architecture"],
        num_layers=config.num_layers,
        d_model=config.d_model,
        d_hidden=config.d_hidden,
        k=config.k,
        auxk=config.auxk,
        training_token_budget=budget or None,
        training_tokens=record.tokens,
        evaluation_sequences=int(args.eval_sequences),
        # Projected from this run's own arguments, which the settings block below
        # writes verbatim. The declaration is what a pair is refused on and the
        # settings block is where the values come from; a second spelling of
        # either would be a second configuration.
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        corpus_seed=int(args.corpus_seed),
        max_tokens=int(args.max_tokens),
    )
    # The last loop evaluation *is* the final one. Recomputing it here ran a
    # second 256-sequence sweep per run and produced, necessarily, the identical
    # number.
    args.corpus = corpus
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "config": config.record(),
        "n_parameters": int(n_parameters),
        "writes_to": {str(k): v for k, v in model.writes_to.items()},
        "target": target,
        MATCHED_TRAINING_KEY: matched.record(),
        "loader_gate": loader_gate,
        "training": record.record(),
        "held_out": {**final, "near_duplicate_screen": screen},
        # The fitted basis, per layer and by both definitions, because R2.4's
        # adequacy gate is read on it and a cross-layer scalar cannot answer a
        # per-layer question. `from_silent_steps` is this checkpoint's own
        # dead-latent counter -- a latent that fired within the last `dead_steps`
        # training steps; `held_out.live_basis` is the census on the evaluation
        # cohort. src.transfer.basis_criteria applies the criterion to either.
        "basis": {
            "live_per_layer_from_silent_steps": model.live_latents_per_layer(),
            "dead_steps": config.dead_steps,
            "d_model": config.d_model,
            "note": (
                "live latents at each layer under the checkpoint's own dead-latent "
                "definition. The held-out census under held_out.live_basis is the "
                "stricter reading and is a different question, not a better "
                "estimate of the same one"
            ),
        },
        "condition": {
            "arm": model_handle.name,
            "corpus": str(corpus),
            "corpus_source": source,
            "symbol_band": [low, high],
            "symbol_unit": symbol_unit,
            "input_rendering": model_handle.rendering_note,
            "training_budget": (
                f"{budget} scored tokens; --steps {args.steps} bounds the run and "
                f"{record.steps} were taken"
                if budget
                else f"{args.steps} steps of {args.batch_size} records, "
                "no token budget declared"
            ),
            "draw": "block-shuffled stream, 8192-record blocks, seeded; a prefix "
            "of a corpus grouped by cluster or shard is a region rather than a "
            "sample",
            "held_out_draw": (
                f"drawn at a skip of {held_out_offset} eligible records, past "
                "everything the training stream reaches at this step budget, so "
                "the evaluation cohort is both disjoint from training and from "
                "the same region of the corpus rather than from its head; then "
                "screened against every training record for near-duplication, "
                "because a record-level offset is not a held-out set on a protein "
                "corpus (L30). See held_out.near_duplicate_screen"
            ),
            "held_out_is_near_duplicate_disjoint_not_homology_disjoint": (
                "the screen removes near-duplicates and deliberately not remote "
                "homologues: a near-duplicate gate is attainable on the text "
                "control under this same procedure and a homology gate has no "
                "text analogue at all, so gating homology would hold the protein "
                "mode to a criterion the text mode is not defined under. What "
                "remains is measured rather than hidden -- the per-candidate "
                "maximum containment is reported threshold-free"
            ),
            "scored_positions": "content tokens only; padding, terminus and "
            "special tokens are excluded from the objective, and on an "
            "ec_conditioned arm so is the whole conditioning prompt -- the "
            "objective is fitted on residue positions alone, which is what the "
            "unconditioned protein arms' content mask keeps",
            "estimand": "reads the replaced block's input, predicts that block's "
            "output before the residual add",
            "bounded_reproduction": "a declared step budget, not ProGenMech's "
            "5M-sequence epoch; the deliverable is CLT minus PLT at equal "
            "budget, not an absolute NMSE against their published value",
            # Every difference from their released trainer, listed. All of them
            # are symmetric across the two arms, so none of them threatens the
            # CLT-minus-PLT difference this stage exists to measure; each of them
            # blocks a comparison against their published absolute number, which
            # the bounded_reproduction note above already disclaims. Recorded in
            # full because a list that named two of six read as complete.
            "deviations_from_released_code": [
                "decoder initialisation: theirs clones the encoder weight and "
                "normalises its columns to unit norm, tying decoder to encoder; "
                "ours draws each decoder independently (kaiming_uniform_, a=sqrt(5), "
                "mean column norm ~2.0 against their 1.0). Their CLT loop also "
                "re-runs kaiming_uniform_ on the *encoder* inside the decoder "
                "loop, which is a defect and is not reproduced",
                "decoder norms unconstrained during training, matching their "
                "trained behaviour (their norm_weights/norm_grad are defined and "
                "never called anywhere in their repository)",
                "optimiser: theirs is AdamW for the CLT and Adam for the PLT; "
                "both arms here use AdamW, because an optimiser difference "
                "between the two arms is the confound this comparison exists to "
                "remove",
                "precision: theirs trains bf16-mixed with high matmul precision; "
                "this trains in strict float32",
                "input distribution: theirs mixes roughly one third GLM/infilling "
                "instances into every batch; this trains on pure causal-LM batches",
                "corpus filter: the residue band here is this stage's own choice "
                "-- 32-1022 for a panel protein arm, and for a joint protein mode "
                "a ceiling derived from --max-tokens and the measured rendering "
                "wrapper. Their data module's length truncation is commented out, "
                "and 1022 appears only in their activation-collection script",
            ],
            "cross_arm_comparability": (
                "an arm's corpus, rendering, symbol band and block estimand are "
                "each declared once and resolved per arm, so two arms differ in "
                "the model and in what its own corpus is -- not in how either was "
                "read. What they do NOT share is a token budget: a residue band "
                "and a character floor are bands in different units, which is why "
                "symbol_unit is recorded beside symbol_band (Appendix B rule 21). "
                "Two MODES of one joint checkpoint do share one: --train-tokens "
                "matches them in scored tokens, which is the unit both bands "
                "resolve to and the only one they have in common"
            ),
        },
    }
    # ProGen3's stem is unchanged, so the campaign already on disk keeps its file
    # names; a panel arm carries its own name because several arms of one
    # comparison are written into one directory and would otherwise overwrite each
    # other under identical hyper-parameters -- which is exactly the configuration
    # the matched pair runs in. A joint checkpoint's two modes are the same case
    # one level down: they share every hyper-parameter by construction, so the
    # mode has to be in the name or the second run would overwrite the first.
    stem = f"{args.architecture}_d{args.d_hidden}_k{args.k}_s{args.corpus_seed}"
    if joint:
        stem = f"{args.rendering}_{args.mode}_{stem}"
    elif args.arm != PROGEN3_ARM:
        stem = f"{args.arm}_{stem}"
    torch.save(
        {
            "config": config.record(),
            "state_dict": model.state_dict(),
            MATCHED_TRAINING_KEY: matched.record(),
            "record": payload,
        },
        args.out / f"{stem}.pt",
    )
    write_json(args.out / f"{stem}.json", payload)
    print(
        f"[done] held-out NMSE sum {final['nmse_sum']:.4f}  "
        f"{record.tokens} scored tokens  matched digest {matched.digest()[:12]}  "
        f"wrote {args.out / stem}.pt"
    )


if __name__ == "__main__":
    main()
