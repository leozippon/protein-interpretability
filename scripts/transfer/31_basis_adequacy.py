#!/usr/bin/env python3
"""Read D3.h's adequacy gate off the dictionaries and alignment artefacts on disk.

**Why this stage exists.** EXP-R2-191 fixed R2.4's gate as a number before any of
its four dictionaries existed, and EXP-R2-194 read it. Nothing about that reading
was executable: Criterion A's ratio, its layer window, Criterion B1's void and
Criterion B2's threshold were applied by hand to JSON, and B2's per-layer figures
were not per-layer at all -- they were ``d_hidden - n_dead/num_layers``, a mean
over layers, because the trainer collapsed a ``(num_layers, d_hidden)`` dead mask
into one scalar before recording it. The pre-declaration requires Criterion B to
hold "at the layers a difference is reported on", and a mean cannot answer that.
This stage recovers the per-layer reading from checkpoints that already exist,
applies the criteria through :mod:`src.transfer.basis_criteria`, and writes the
verdict as an artefact instead of as prose (EXP-R2-203).

**Two definitions of a live basis, reported side by side rather than blended.**

*From the checkpoint.* ``silent_steps`` is a persisted buffer, so every trained
dictionary already carries how long each latent had been silent when training
stopped, per layer. A latent is live by this reading when it fired within the
last ``dead_steps`` training steps -- ten thousand sequences at the campaign's
setting. This needs no GPU, no corpus and no backbone: it reads one 32x8192
buffer out of an 8.6 GB file.

*From the held-out cohort* (``--held-out``). A latent is live when it fires on the
evaluation cohort the dictionary was scored on. This is the stricter reading and
the one a feature diff would actually be taken over, because a diff is read on
data and not on a training counter. It reloads the backbone and re-draws the
cohort through ``17_train_transcoder.py``'s own functions, so it is the same
population under the same screen rather than a second draw with the same name.

Neither is a better estimate of the other. They answer different questions and
the artefact carries both.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.transfer import basis_criteria  # noqa: E402
from src.transfer.arms import REPO, corpus_location, iter_corpus_records  # noqa: E402
from src.transfer.io import write_json  # noqa: E402
from src.transfer.replaceable import joint_mode_corpus  # noqa: E402
from src.transfer.transcoders import (  # noqa: E402
    live_latents_per_layer,
    load_trained_transcoder,
)

SCHEMA_VERSION = "r2_transfer_basis_adequacy_v1"
DEFAULT_OUT = REPO / "results/transfer/basis_adequacy"


def _load_stage(filename: str) -> Any:
    """Import a stage whose module name starts with a digit."""

    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(f"_transfer_stage_{filename[:2]}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE17 = _load_stage("17_train_transcoder.py")


def _pair(argument: str, what: str) -> tuple[str, str]:
    name, _, value = argument.partition("=")
    if not name or not value:
        raise argparse.ArgumentTypeError(f"{what} must be written NAME=PATH, not {argument!r}")
    return name, value


def read_checkpoint_basis(path: Path) -> dict[str, Any]:
    """A dictionary's per-layer live basis and the run that produced it.

    Memory-mapped, because the buffer this reads is 2 MB inside an 8.6 GB file
    and materialising the rest would cost a minute per cell for nothing.
    """

    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    missing = sorted({"config", "state_dict"} - set(checkpoint))
    if missing:
        raise RuntimeError(
            f"{path} is not a checkpoint from 17_train_transcoder.py (missing {missing})"
        )
    state = checkpoint["state_dict"]
    if "silent_steps" not in state:
        raise RuntimeError(
            f"{path} carries no 'silent_steps' buffer, so its per-layer live basis "
            "cannot be recovered. Nothing here reconstructs it from the cross-layer "
            "scalar in the training record: that scalar is a sum over layers and a "
            "mean is not a per-layer count"
        )
    config = checkpoint["config"]
    record = checkpoint.get("record") or {}
    return {
        "dictionary": str(path),
        "config": config,
        "settings": record.get("settings", {}),
        "held_out_nmse_per_layer": (record.get("held_out") or {}).get("nmse_per_layer"),
        "silent_steps": state["silent_steps"],
        "live_per_layer": live_latents_per_layer(
            state["silent_steps"], int(config["dead_steps"])
        ),
    }


def held_out_census(basis: dict[str, Any], *, device: str) -> dict[str, Any]:
    """Count the live basis again, on the cohort the dictionary was scored on.

    The backbone, the band and the cohort all come from ``17_train_transcoder.py``
    -- the same functions the training run called, driven from the settings block
    that run wrote into its own checkpoint. A stage that re-derived any of them
    would be measuring a different population under the dictionary's name, which
    is the failure the near-duplicate screen exists to make visible rather than a
    detail of implementation.
    """

    settings = basis["settings"]
    required = ("joint_checkpoint", "rendering", "mode", "max_tokens", "corpus_seed",
                "steps", "batch_size", "eval_sequences")
    absent = [name for name in required if settings.get(name) in (None, "")]
    if absent:
        raise RuntimeError(
            f"{basis['dictionary']} does not declare {absent} in its settings block, "
            "so the cohort it was held out on cannot be redrawn. This stage refuses "
            "to substitute a fresh draw for it"
        )

    source = joint_mode_corpus(settings["mode"])
    low, high = STAGE17.CORPUS_BAND[source]
    corpus = corpus_location(source)
    print(f"[paths] corpus {corpus} ({source}, mode {settings['mode']})")
    handle, target, (low, high) = STAGE17.open_joint_target(
        Path(settings["joint_checkpoint"]),
        rendering=settings["rendering"],
        mode=settings["mode"],
        device=device,
        max_tokens=int(settings["max_tokens"]),
        protein_context=settings.get("protein_context"),
        band=(low, high),
    )

    def records() -> Iterator[tuple[str, str | None]]:
        return iter_corpus_records(source, min_symbols=low, max_symbols=high)

    symbol_unit = "characters" if source == "openwebtext" else "residues"
    cohort, screen, offset = STAGE17.held_out_cohort(
        records,
        corpus_seed=int(settings["corpus_seed"]),
        steps=int(settings["steps"]),
        batch_size=int(settings["batch_size"]),
        eval_sequences=int(settings["eval_sequences"]),
        symbol_unit=symbol_unit,
    )
    replacement, _, _ = load_trained_transcoder(Path(basis["dictionary"]))
    model = replacement.model.to(device=device, dtype=torch.float32)
    held = STAGE17.evaluate(
        model, handle, cohort, batch_size=int(settings["batch_size"])
    )
    del model, replacement, handle
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "backbone": target["checkpoint_facts"].get("resolved_path"),
        "weights_sha256": target["checkpoint_facts"].get("weights_sha256"),
        "corpus": str(corpus),
        "symbol_band": [low, high],
        "held_out_offset": offset,
        "near_duplicate_screen": screen,
        "nmse_per_layer": held["nmse_per_layer"],
        "nmse_sum": held["nmse_sum"],
        "live_basis": held["live_basis"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        action="append",
        default=[],
        required=True,
        metavar="LABEL=PATH.pt",
        help="a trained dictionary to read the gate on. Repeat once per cell; the "
        "gate is over all of them, because 'in all four cells' is what Criterion B "
        "was declared as and a gate that passes on three of four is a failed gate",
    )
    parser.add_argument(
        "--alignment",
        action="append",
        default=[],
        metavar="MODE=PATH.json",
        help="a 25_model_diffing_baselines.py artefact, keyed by the mode it "
        "measured. Criterion A is read per mode from it, and Criterion B1's "
        "descriptive numbers need it as their denominator. Omitted means no "
        "Criterion A is issued -- which is a stated absence, not a pass",
    )
    parser.add_argument(
        "--mode-of",
        action="append",
        default=[],
        metavar="LABEL=MODE",
        help="which alignment mode a cell is read against, when the cell label "
        "does not name it. Defaults to the mode in the dictionary's own settings",
    )
    parser.add_argument(
        "--held-out",
        action="store_true",
        help="also count the live basis on the cohort each dictionary was scored "
        "on. Loads a backbone and streams the corpus, so it needs a GPU and the "
        "staged corpora; without it the artefact carries the checkpoint reading "
        "alone and says so",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cells = [_pair(entry, "--cell") for entry in args.cell]
    alignments = {
        mode: json.loads(Path(path).read_text(encoding="utf-8"))
        for mode, path in (_pair(entry, "--alignment") for entry in args.alignment)
    }
    mode_of = dict(_pair(entry, "--mode-of") for entry in args.mode_of)

    criterion_a = {
        mode: basis_criteria.criterion_a(*basis_criteria.alignment_residuals(artefact))
        for mode, artefact in alignments.items()
    }
    for mode, reading in criterion_a.items():
        print(
            f"[criterion A] {mode}: median R {reading['median']:.3f} "
            f"IQR [{reading['iqr'][0]:.3f}, {reading['iqr'][1]:.3f}] "
            f"{reading['fraction_above_one']:.3f} above 1 -> {reading['verdict']}"
        )

    readings: list[dict[str, Any]] = []
    gate_input: list[dict[str, Any]] = []
    for label, path in cells:
        basis = read_checkpoint_basis(Path(path))
        d_model = int(basis["config"]["d_model"])
        mode = mode_of.get(label) or basis["settings"].get("mode")
        entry: dict[str, Any] = {
            "cell": label,
            "dictionary": basis["dictionary"],
            "mode": mode,
            "config": basis["config"],
            "from_silent_steps": {
                "definition": "a latent that fired within the last dead_steps "
                "training steps, read off the checkpoint's own persisted counter",
                "dead_steps": int(basis["config"]["dead_steps"]),
                "live_per_layer": basis["live_per_layer"],
                "b2": basis_criteria.criterion_b2(basis["live_per_layer"], d_model),
            },
        }
        if mode in alignments and basis["held_out_nmse_per_layer"]:
            cross, _ = basis_criteria.alignment_residuals(alignments[mode])
            entry["b1_descriptive"] = basis_criteria.criterion_b1_descriptive(
                basis["held_out_nmse_per_layer"], cross
            )
        if args.held_out:
            census = held_out_census(basis, device=args.device)
            live = census["live_basis"]["live_per_layer"]
            entry["from_held_out"] = {
                "definition": "a latent that fires at least the threshold number "
                "of times on the cohort this dictionary was scored on",
                **census,
                "b2": {
                    threshold: basis_criteria.criterion_b2(counts, d_model)
                    for threshold, counts in live.items()
                },
            }
        b2 = entry["from_silent_steps"]["b2"]
        print(
            f"[criterion B2] {label}: mean {b2['mean_live_per_layer']:.1f} "
            f"({b2['mean_reading']}), min {b2['min_live_per_layer']} at layer "
            f"{b2['argmin_layer']}, {b2['n_failing_layers']} layer(s) below "
            f"d_model -> {b2['verdict']}"
        )
        readings.append(entry)
        gate_input.append({"cell": label, "b2": b2})

    gate = basis_criteria.basis_gate(gate_input)
    print(f"[gate] {gate['verdict']}; failing cells {gate['failing_cells']}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "criterion_a": criterion_a,
        "criterion_b1": {
            "verdict": "VOID",
            "reason": "both text control cells fail it, which under the refusal "
            "condition its own pre-declaration named makes it a specification "
            "defect rather than a result. Its numbers appear per cell as "
            "b1_descriptive and no cell is judged by them",
        },
        "cells": readings,
        "criterion_b2_gate": gate,
        "definitions": {
            "silent_steps": "the checkpoint's own dead-latent counter. A latent is "
            "live when it fired within the last dead_steps TRAINING steps, which at "
            "this campaign's batch is ten thousand sequences -- so it admits a "
            "latent no evaluation cohort of a few hundred records would ever see",
            "held_out": "the census on the cohort the dictionary was scored on. "
            "Stricter, and the basis a feature diff would actually be taken over, "
            "since a diff is read on data rather than on a training counter",
            "not_interchangeable": "neither is an estimate of the other. A cell "
            "that clears the gate on one and not the other has not half-cleared it; "
            "it has cleared one question and failed another",
        },
    }
    name = "basis_adequacy" + ("_held_out" if args.held_out else "") + ".json"
    write_json(args.out / name, payload)
    print(f"[done] wrote {args.out / name}")


if __name__ == "__main__":
    main()
