#!/usr/bin/env python3
"""Train a CLT and a PLT on ProGen3-112M under identical conditions.

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

Activations are captured from the eager ProGen3 whose conversion
``src.transfer.progen3`` verifies, at the tensors ProGenMech's own collector
uses: the MoE block's input, and its output before the residual add. Special
tokens are excluded from the objective, as in theirs, because a transcoder
scored on padding is scored on the easiest positions in the batch.

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
from typing import Any, Iterator

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.arms import REPO, UNIREF50_FASTA, iter_fasta  # noqa: E402
from src.transfer.io import write_json  # noqa: E402
from src.transfer.progen3 import (  # noqa: E402
    ProGen3,
    content_mask,
    load_progen3,
    moe_intercept,
    self_check,
)
from src.transfer.transcoders import (  # noqa: E402
    Transcoder,
    TranscoderConfig,
    TrainingRecord,
)

SCHEMA_VERSION = "r2_transfer_transcoder_training_v1"
DEFAULT_OUT = REPO / "results/transfer/transcoder_training"

#: ProGenMech's own filter: UniRef50 sequences of at most 1022 residues.
MAX_RESIDUES = 1022
MIN_RESIDUES = 32


def stream_sequences(
    path: Path, *, seed: int, skip: int, limit: int | None = None
) -> Iterator[str]:
    """UniRef50 records within the declared band, in a seeded shuffled order.

    The corpus is ordered by cluster, so consecutive records are homologues and a
    prefix is a family rather than a sample (Appendix B rule 1). A full seeded
    permutation of 60M records is not affordable here, so the draw is a shuffled
    reservoir: records are read in file order into blocks and each block is
    permuted before it is emitted. That breaks the local homology ordering, which
    is what the rule is about, and the block size is recorded so the residual
    correlation is a declared property rather than an unstated one.
    """

    rng = np.random.default_rng(seed)
    block: list[str] = []
    emitted = 0
    seen = 0
    for _, sequence in iter_fasta(path):
        if not MIN_RESIDUES <= len(sequence) <= MAX_RESIDUES:
            continue
        seen += 1
        if seen <= skip:
            continue
        block.append(sequence)
        if len(block) < 8192:
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
def capture(pg: ProGen3, sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MoE block inputs, outputs and a scored-token mask for one batch.

    Returns ``(inputs, outputs, mask)`` with the layer axis first. The mask
    excludes every non-residue position: ProGen3's terminus markers and its
    special tokens are trivially predictable and would flatter any transcoder
    scored on them.
    """

    batch = pg.batch(sequences, reverse=False)
    captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def tap(layer: int, block_input: torch.Tensor, block_output: torch.Tensor) -> None:
        captured[layer] = (block_input.detach(), block_output.detach())
        return None

    with moe_intercept(pg, tap):
        pg.model(
            input_ids=batch["input_ids"],
            position_ids=batch["position_ids"],
            sequence_ids=batch["sequence_ids"],
            use_cache=False,
            return_dict=True,
        )

    layers = sorted(captured)
    inputs = torch.stack([captured[layer][0] for layer in layers])
    outputs = torch.stack([captured[layer][1] for layer in layers])
    return inputs, outputs, content_mask(pg, batch["input_ids"])


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
    model: Transcoder, pg: ProGen3, sequences: list[str], *, batch_size: int
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
            x, y, mask = capture(pg, chunk)
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
    parser.add_argument("--auxk", type=int, default=128)
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

    corpus = args.corpus if args.corpus is not None else UNIREF50_FASTA
    if not Path(corpus).is_file():
        raise FileNotFoundError(f"no UniRef50 FASTA at {corpus}")

    print("[loader] loading ProGen3-112M and self-checking the conversion")
    load_kwargs: dict[str, Any] = {"device": args.device, "dtype": torch.bfloat16}
    if args.checkpoint is not None:
        load_kwargs["checkpoint"] = args.checkpoint
    pg = load_progen3(**load_kwargs)
    loader_gate = self_check(pg)
    print(f"  self-check NLL {loader_gate['nll']:.4f} PASS")

    config = TranscoderConfig(
        num_layers=pg.n_layers,
        d_model=int(pg.config.hidden_size),
        d_hidden=args.d_hidden,
        k=args.k,
        auxk=args.auxk,
        cross_layer=args.architecture == "clt",
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

    # The held-out set is drawn from the head of the stream and then skipped, so
    # no evaluated sequence is ever trained on.
    stream = stream_sequences(Path(corpus), seed=args.corpus_seed, skip=0)
    held_out = [next(stream) for _ in range(args.eval_sequences)]
    training = stream_sequences(
        Path(corpus), seed=args.corpus_seed, skip=0, limit=None
    )
    for _ in range(args.eval_sequences):
        next(training)

    record = TrainingRecord()
    started = time.time()
    print(f"[train] {args.steps} steps, batch {args.batch_size}")
    for step in range(1, args.steps + 1):
        chunk = [next(training) for _ in range(args.batch_size)]
        x, y, mask = capture(pg, chunk)
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
            held = evaluate(model, pg, held_out, batch_size=args.batch_size)
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

    final = evaluate(model, pg, held_out, batch_size=args.batch_size)
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
            "corpus": str(corpus),
            "residue_band": [MIN_RESIDUES, MAX_RESIDUES],
            "draw": "block-shuffled stream, 8192-record blocks, seeded; a prefix "
            "of UniRef50 is a family rather than a sample",
            "scored_positions": "residue tokens only; ProGen3's special and "
            "terminus tokens are excluded from the objective",
            "estimand": "reads the MoE block input, predicts the MoE block "
            "output before the residual add",
            "bounded_reproduction": "a declared step budget, not ProGenMech's "
            "5M-sequence epoch; the deliverable is CLT minus PLT at equal "
            "budget, not an absolute NMSE against their published value",
            "deviations_from_released_code": [
                "decoders initialised directly, rather than by re-running "
                "kaiming_uniform_ on the encoder weight inside the decoder loop",
                "decoder norms unconstrained, matching their trained behaviour "
                "(their norm_weights/norm_grad are defined and never called)",
            ],
        },
    }
    stem = f"{args.architecture}_d{args.d_hidden}_k{args.k}_s{args.corpus_seed}"
    torch.save(
        {"config": config.record(), "state_dict": model.state_dict(), "record": payload},
        args.out / f"{stem}.pt",
    )
    write_json(args.out / f"{stem}.json", payload)
    print(f"[done] held-out NMSE sum {final['nmse_sum']:.4f}  wrote {args.out / stem}.pt")


if __name__ == "__main__":
    main()
