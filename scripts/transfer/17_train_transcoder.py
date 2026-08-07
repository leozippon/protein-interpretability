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

Output is a checkpoint plus a JSON record, and the checkpoint is written in the
shape ``15_replacement_faithfulness.py`` can load, so a trained transcoder goes
straight to the faithfulness gate with no conversion step in between.
"""

from __future__ import annotations

import argparse
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
from src.transfer.arms import REPO, corpus_location, iter_corpus_records  # noqa: E402
from src.transfer.io import write_json  # noqa: E402
from src.transfer.replaceable import (  # noqa: E402
    PROGEN3_ARM,
    ReplaceableModel,
    arm_training_corpus,
    eligible_arms,
    load_replaceable,
)
from src.transfer.transcoders import (  # noqa: E402
    DEAD_STEPS_SEQUENCES,
    Transcoder,
    TranscoderConfig,
    TrainingRecord,
)

#: Records buffered before each shuffle. The held-out draw must clear a whole
#: number of these, so the block size and the offset that skips past it cannot be
#: allowed to disagree -- it was a literal in two places and they did.
SHUFFLE_BLOCK = 8192

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
    "openwebtext": (MIN_CHARACTERS, None),
}


def stream_records(
    records: Callable[[], Iterator[str]], *, seed: int, skip: int, limit: int | None = None
) -> Iterator[str]:
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
    block: list[str] = []
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
    model: ReplaceableModel, records: list[str]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Block inputs, outputs and a scored-token mask for one batch.

    Returns ``(inputs, outputs, mask)`` with the layer axis first. The mask
    excludes every non-content position: terminus and padding tokens are
    trivially predictable and would flatter any transcoder scored on them.
    """

    batch = model.batch(model.render(records))
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


def evaluate(
    model: Transcoder,
    decoder: ReplaceableModel,
    sequences: list[str],
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Held-out NMSE, on sequences the run never trains on."""

    model.eval()
    totals = torch.zeros(model.config.num_layers, dtype=torch.float64)
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
            batches += 1
    model.train()
    per_layer = (totals / max(batches, 1)).tolist()
    return {
        "nmse_per_layer": per_layer,
        "nmse_sum": float(sum(per_layer)),
        "n_batches": batches,
        "n_sequences": len(sequences),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("clt", "plt"), required=True)
    parser.add_argument(
        "--arm",
        default=PROGEN3_ARM,
        choices=eligible_arms(CAMPAIGN_PANEL),
        help="which decoder's blocks to train against. The eligible set is "
        "composed by src.transfer.replaceable.eligible_arms from three "
        "declarations -- the campaign panel, the architectures that carry this "
        "estimand, and the arms with a measured loader band -- and is not a list "
        "this stage keeps",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="token cap a dense arm's inputs are truncated to. ProGen3 ignores it: "
        "its batch preparer pads to the longest record and its residue band is "
        "declared in residues",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=20000)
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
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    # The corpus is resolved and checked before anything reaches a GPU, so a host
    # that has not staged it fails in a second rather than after a checkpoint is
    # loaded. `--corpus` relocates ProGen3's UniRef50 and is refused for the
    # other sources, which are read through their own declared variables.
    source = arm_training_corpus(args.arm)
    low, high = CORPUS_BAND[source]
    corpus = corpus_location(source, path=args.corpus)

    def records() -> Iterator[str]:
        return iter_corpus_records(
            source, min_symbols=low, max_symbols=high, path=args.corpus
        )

    print(f"[loader] loading {args.arm} and running its self-check")
    model_handle: ReplaceableModel = load_replaceable(
        args.arm,
        campaign_panel=CAMPAIGN_PANEL,
        device=args.device,
        dtype="bfloat16",
        max_tokens=args.max_tokens,
        checkpoint=args.checkpoint,
    )
    loader_gate = model_handle.self_check()
    print(f"  self-check NLL {loader_gate['nll']:.4f} PASS")

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
        arm=args.arm,
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
    blocks_touched = -(-(args.steps * args.batch_size) // SHUFFLE_BLOCK)
    held_out_offset = blocks_touched * SHUFFLE_BLOCK
    held_out = list(
        stream_records(
            records, seed=args.corpus_seed, skip=held_out_offset,
            limit=args.eval_sequences,
        )
    )
    if len(held_out) < args.eval_sequences:
        raise RuntimeError(
            f"the corpus ran out at the held-out offset: {len(held_out)} of "
            f"{args.eval_sequences} sequences past a skip of {held_out_offset}. "
            "Lower --steps or --eval-sequences rather than evaluating on a "
            "population the training stream also reaches."
        )
    training = stream_records(records, seed=args.corpus_seed, skip=0, limit=None)

    record = TrainingRecord()
    final: dict[str, Any] | None = None
    started = time.time()
    print(f"[train] {args.steps} steps, batch {args.batch_size}")
    for step in range(1, args.steps + 1):
        chunk = [next(training) for _ in range(args.batch_size)]
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
        if step % args.eval_every == 0 or step == args.steps:
            held = evaluate(model, model_handle, held_out, batch_size=args.batch_size)
            if step == args.steps:
                final = held
            entry = {
                "step": step,
                "train_nmse_sum": float(report["nmse_sum"].detach()),
                "held_out_nmse_sum": held["nmse_sum"],
                "held_out_nmse_per_layer": held["nmse_per_layer"],
                "n_dead": report["n_dead"],
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

    if final is None:
        raise RuntimeError(
            "the training loop never reached its final step, so no held-out "
            "evaluation exists; refusing to write a checkpoint with no score"
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
        "loader_gate": loader_gate,
        "training": record.record(),
        "held_out": final,
        "condition": {
            "arm": args.arm,
            "corpus": str(corpus),
            "corpus_source": source,
            "symbol_band": [low, high],
            "symbol_unit": "characters" if source == "openwebtext" else "residues",
            "input_rendering": (
                "src.transfer.arms.Cohort.input_strings, the panel's one "
                "declaration of what string each arm is fed"
            ),
            "draw": "block-shuffled stream, 8192-record blocks, seeded; a prefix "
            "of a corpus grouped by cluster or shard is a region rather than a "
            "sample",
            "held_out_draw": (
                f"drawn at a skip of {held_out_offset} eligible records, past "
                "everything the training stream reaches at this step budget, so "
                "the evaluation cohort is both disjoint from training and from "
                "the same region of the corpus rather than from its head"
            ),
            "scored_positions": "content tokens only; padding, terminus and "
            "special tokens are excluded from the objective",
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
                "corpus filter: the 32-1022 residue band here is this stage's own "
                "choice. Their data module's length truncation is commented out, "
                "and 1022 appears only in their activation-collection script",
            ],
            "cross_arm_comparability": (
                "an arm's corpus, rendering, symbol band and block estimand are "
                "each declared once and resolved per arm, so two arms differ in "
                "the model and in what its own corpus is -- not in how either was "
                "read. What they do NOT share is a token budget: a residue band "
                "and a character floor are bands in different units, which is why "
                "symbol_unit is recorded beside symbol_band (Appendix B rule 21)"
            ),
        },
    }
    # ProGen3's stem is unchanged, so the campaign already on disk keeps its file
    # names; a panel arm carries its own name because several arms of one
    # comparison are written into one directory and would otherwise overwrite each
    # other under identical hyper-parameters -- which is exactly the configuration
    # the matched pair runs in.
    stem = f"{args.architecture}_d{args.d_hidden}_k{args.k}_s{args.corpus_seed}"
    if args.arm != PROGEN3_ARM:
        stem = f"{args.arm}_{stem}"
    torch.save(
        {"config": config.record(), "state_dict": model.state_dict(), "record": payload},
        args.out / f"{stem}.pt",
    )
    write_json(args.out / f"{stem}.json", payload)
    print(f"[done] held-out NMSE sum {final['nmse_sum']:.4f}  wrote {args.out / stem}.pt")


if __name__ == "__main__":
    main()
