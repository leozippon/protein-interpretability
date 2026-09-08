#!/usr/bin/env python3
"""Explicit classifier-to-text handoff runner. Independent of stage 35/36/48.

H200: freeze with run_transfer_h200.sh --freeze-only, then
run_external_baseline_h200.sh --stage 49_classifier_handoff.py --expect report.json.

This script accepts the wrapper's fixed --device/--out and writes report.json.
It does not register in panel_contract STAGE_CONTRACTS. No nnsight, no peft,
no Hub downloads (local_files_only). Train does not load the receiver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.classifier_handoff import (  # noqa: E402
    ALLOWED_HEADS,
    ClassifierHead,
    CONDITION,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BETAS,
    DEFAULT_BUDGET,
    DEFAULT_DTYPE,
    DEFAULT_EPOCHS,
    DEFAULT_EPS,
    DEFAULT_GRAD_CLIP,
    DEFAULT_HEADS,
    DEFAULT_LR,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_PROMPT_MODE,
    DEFAULT_SEEDS,
    DEFAULT_WEIGHT_DECAY,
    EVALUATION_CLAIM,
    PILOT_KIND,
    PILOT_NOTE,
    REPORT_SCHEMA,
    SUBSTAGES,
    bound_pretrained_cache,
    build_classifier_head,
    cell_relative_dir,
    classifier_identity,
    command_record,
    evaluate_classifier_cell,
    generate_handoff,
    interface_head_step,
    load_classifier,
    sent_class_preservation,
    split_cache_features,
    train_all_cells,
    validate_classifier_identity,
)
from src.transfer.io import write_json as write_json_file  # noqa: E402
from src.transfer.latent_bridge import (  # noqa: E402
    ALLOWED_DTYPES,
    ALLOWED_PROMPT_MODES,
    CLASS_IDS,
    checkpoint_content_digest,
    freeze_module,
    load_prepared_payload,
    load_receiver,
    parameter_fingerprint,
    portable_identity,
    software_record,
    timed_cuda,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--substage", required=True, choices=SUBSTAGES)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--classifier-dir", type=Path)
    parser.add_argument("--receiver-path", type=Path)
    parser.add_argument("--prompt-mode", choices=ALLOWED_PROMPT_MODES, default=DEFAULT_PROMPT_MODE)
    parser.add_argument("--dtype", choices=ALLOWED_DTYPES, default=DEFAULT_DTYPE)
    parser.add_argument("--heads", nargs="+", choices=ALLOWED_HEADS, default=list(DEFAULT_HEADS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-interface-records", type=int, default=32)
    parser.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--beta1", type=float, default=DEFAULT_BETAS[0])
    parser.add_argument("--beta2", type=float, default=DEFAULT_BETAS[1])
    parser.add_argument("--trainable-budget", type=int, default=DEFAULT_BUDGET)
    return parser


def _write_report(out: Path, payload: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    body = {
        "schema_version": REPORT_SCHEMA,
        "pilot": PILOT_KIND,
        "pilot_note": PILOT_NOTE,
        "condition": CONDITION,
        "evaluation_claim": EVALUATION_CLAIM,
        "first_round_claim": (
            "this run does not freeze epsilon and does not claim a confirmatory "
            "capability result"
        ),
        **payload,
    }
    write_json_file(out / "report.json", body)


def _require(path: Path | None, name: str) -> Path:
    if path is None:
        raise ValueError(f"{name} is required for this substage")
    return path


def _betas(args: argparse.Namespace) -> tuple[float, float]:
    return (float(args.beta1), float(args.beta2))


def run_interface(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "substage": "interface",
        "status": "failed",
        "gates": {},
        "used_family_holdout": False,
        "device": args.device,
        "software": software_record(),
        "command": command_record(args),
    }
    gates = report["gates"]
    try:
        if not 1 <= args.max_interface_records <= 32:
            raise ValueError("interface max-interface-records must be in 1..32")
        prepared = load_prepared_payload(_require(args.prepared, "--prepared"))
        cache_path = _require(args.cache, "--cache")
        receiver_path = _require(args.receiver_path, "--receiver-path")
        features, manifest, train_rows, _val_rows, _test_rows = bound_pretrained_cache(
            cache_path, prepared
        )
        train_feat, _val_feat, _test_feat = split_cache_features(
            features, train_rows, _val_rows, _test_rows
        )
        train = train_rows[: args.max_interface_records]
        if not train or any(row.role != "train" for row in train):
            raise ValueError("interface requires train records only")
        gates["bounded_train_only"] = True
        gates["pretrained_donor_cache"] = True
        report.update(
            n_records=len(train),
            prepared_sha256=prepared["prepared_sha256"],
            d_in=int(train_feat.shape[1]),
            cache_d=int(manifest["d"]),
        )
        seed = int(args.seeds[0])
        step_reports: dict[str, Any] = {}
        for kind in args.heads:
            head = build_classifier_head(
                kind=kind,
                d_in=int(train_feat.shape[1]),
                budget=args.trainable_budget,
                seed=seed,
            )
            step, elapsed, peak = timed_cuda(
                args.device,
                lambda current=head: interface_head_step(
                    head=current,
                    features=train_feat[: len(train)],
                    record=train[0],
                    device=args.device,
                ),
            )
            step_reports[kind] = {**step, "walltime_s": elapsed, "peak_memory_bytes": peak}
            gates[f"{kind}_finite_nonzero_grad_and_ln_frozen"] = True
        report["optimizer_check"] = step_reports
        model, tokenizer = load_receiver(
            receiver_path, device=args.device, dtype=args.dtype, prompt_mode=args.prompt_mode
        )
        freeze_module(model)
        receiver_identity = checkpoint_content_digest(receiver_path)
        before = parameter_fingerprint(model)
        texts, elapsed, peak = timed_cuda(
            args.device,
            lambda: generate_handoff(
                predicted_classes=list(CLASS_IDS),
                receiver=model,
                tokenizer=tokenizer,
                prompt_mode=args.prompt_mode,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                max_prompt_tokens=args.max_prompt_tokens,
                batch_size=1,
            ),
        )
        preservation = sent_class_preservation(texts, list(CLASS_IDS))
        after = parameter_fingerprint(model)
        if before != after:
            raise RuntimeError("receiver parameters changed during interface handoff")
        report.update(
            receiver_identity=portable_identity(receiver_identity),
            receiver_fingerprint_before=before,
            receiver_fingerprint_after=after,
            handoff_rows=preservation,
            handoff_note=(
                "seven artificial sent classes; format control only, not gold oracle "
                "and not a capability result"
            ),
            handoff_walltime_s=elapsed,
            handoff_peak_memory_bytes=peak,
        )
        gates["receiver_frozen"] = True
        gates["handoff_preserves_all_seven_sent_classes"] = bool(
            len(preservation) == 7 and all(row["preserved"] for row in preservation)
        )
        if not gates["handoff_preserves_all_seven_sent_classes"]:
            raise RuntimeError("handoff format control failed; do not enter training")
        report["status"] = "passed"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    return report


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    prepared = load_prepared_payload(_require(args.prepared, "--prepared"))
    cache_path = _require(args.cache, "--cache")
    features, manifest, train_rows, val_rows, _test_rows = bound_pretrained_cache(
        cache_path, prepared
    )
    train_feat, val_feat, _test_feat = split_cache_features(
        features, train_rows, val_rows, _test_rows
    )
    del features, _test_feat, _test_rows
    summary = train_all_cells(
        train_features=train_feat,
        train_records=train_rows,
        val_features=val_feat,
        val_records=val_rows,
        d_in=int(train_feat.shape[1]),
        budget=args.trainable_budget,
        heads=args.heads,
        seeds=args.seeds,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=_betas(args),
        eps=args.eps,
        grad_clip=args.grad_clip,
        prepared_hash=prepared["prepared_sha256"],
        cache_manifest=manifest,
        out_dir=args.out,
    )
    return {
        **summary,
        "prepared_sha256": prepared["prepared_sha256"],
        "cache_manifest_sha256": summary["cells"][0]["identity"]["cache_manifest_sha256"],
        "device": args.device,
        "software": software_record(),
        "command": command_record(args),
    }


def _load_train_cells(
    classifier_dir: Path, *, heads: list[str], seeds: list[int]
) -> list[dict[str, Any]]:
    report_path = Path(classifier_dir) / "report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != REPORT_SCHEMA or payload.get("substage") != "train":
        raise ValueError("classifier-dir report is not a current train summary")
    expected = [(kind, seed) for kind in heads for seed in seeds]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("requested head/seed matrix must be nonempty and unique")
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells or payload.get("n_cells") != len(cells):
        raise ValueError("train summary has missing cells or inconsistent n_cells")
    actual = []
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("invalid train inventory cell")
        kind, seed = cell.get("kind"), cell.get("seed")
        if kind not in ALLOWED_HEADS or type(seed) is not int or seed < 0:
            raise ValueError("invalid train inventory head or seed")
        if cell.get("dir") != cell_relative_dir(kind, seed):
            raise ValueError("train inventory directory is not canonical")
        identity = cell.get("identity")
        if not isinstance(identity, dict) or not isinstance(identity.get("train_config"), dict):
            raise ValueError("train inventory lacks a frozen training identity")
        actual.append((kind, seed))
    if actual != expected:
        raise ValueError("train inventory does not match the complete requested head/seed matrix")
    return cells


def _qualified_eval_heads(
    args: argparse.Namespace, *, prepared_hash: str, cache_manifest: dict[str, Any], d_in: int
) -> list[tuple[dict[str, Any], ClassifierHead, str]]:
    """Bind every checkpoint to the frozen inventory and live inputs before decoding."""
    classifier_dir = _require(args.classifier_dir, "--classifier-dir")
    cells = _load_train_cells(classifier_dir, heads=args.heads, seeds=args.seeds)
    qualified = []
    for cell in cells:
        kind, seed = cell["kind"], cell["seed"]
        expected_head = build_classifier_head(
            kind=kind, d_in=d_in, budget=args.trainable_budget, seed=seed
        )
        expected = classifier_identity(
            head=expected_head,
            budget=args.trainable_budget,
            seed=seed,
            prepared_hash=prepared_hash,
            cache_manifest=cache_manifest,
            train_config=cell["identity"]["train_config"],
        )
        if cell["identity"] != expected:
            raise ValueError("train inventory identity differs from the requested head and live inputs")
        checkpoint = classifier_dir / cell["dir"] / "classifier.pt"
        head, payload = load_classifier(checkpoint)
        validate_classifier_identity(payload, expected)
        with checkpoint.open("rb") as stream:
            checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        qualified.append((cell, head, checkpoint_sha))
    return qualified


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    prepared = load_prepared_payload(_require(args.prepared, "--prepared"))
    cache_path = _require(args.cache, "--cache")
    receiver_path = _require(args.receiver_path, "--receiver-path")
    features, manifest, train_rows, val_rows, test_rows = bound_pretrained_cache(
        cache_path, prepared
    )
    _train_feat, _val_feat, test_feat = split_cache_features(
        features, train_rows, val_rows, test_rows
    )
    del _train_feat, _val_feat, train_rows, val_rows
    qualified = _qualified_eval_heads(
        args, prepared_hash=prepared["prepared_sha256"],
        cache_manifest=manifest, d_in=int(test_feat.shape[1]),
    )
    receiver_digest = checkpoint_content_digest(receiver_path)
    model, tokenizer = load_receiver(
        receiver_path, device=args.device, dtype=args.dtype, prompt_mode=args.prompt_mode
    )
    freeze_module(model)
    before = parameter_fingerprint(model)
    evaluated: list[dict[str, Any]] = []
    for cell, head, checkpoint_sha in qualified:
        kind, seed, relative = cell["kind"], cell["seed"], cell["dir"]
        result = evaluate_classifier_cell(
            head=head,
            features=test_feat,
            records=test_rows,
            receiver=model,
            tokenizer=tokenizer,
            prompt_mode=args.prompt_mode,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            batch_size=args.batch_size,
        )
        out_dir = args.out / relative
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json_file(
            out_dir / "classifier_direct.json",
            {
                "source": "classifier_direct_template",
                "evaluation_status": "exploratory",
                "evaluation_claim": EVALUATION_CLAIM,
                "rows": result["direct_rows"],
            },
        )
        write_json_file(
            out_dir / "receiver_handoff.json",
            {
                "source": "receiver_handoff",
                "evaluation_status": "exploratory",
                "evaluation_claim": EVALUATION_CLAIM,
                "rows": result["handoff_rows"],
            },
        )
        evaluated.append(
            {
                "kind": kind,
                "seed": seed,
                "dir": relative,
                "classifier_identity": cell["identity"],
                "classifier_checkpoint_sha256": checkpoint_sha,
                "n": result["n"],
                "classifier_direct_metrics": result["classifier_direct_metrics"],
                "receiver_handoff_metrics": result["receiver_handoff_metrics"],
                "error_decomposition": result["error_decomposition"],
                "classifier_inference_walltime_s": result["classifier_inference_walltime_s"],
                "classifier_inference_peak_memory_bytes": result["classifier_inference_peak_memory_bytes"],
                "receiver_decode_walltime_s": result["receiver_decode_walltime_s"],
                "receiver_decode_peak_memory_bytes": result["receiver_decode_peak_memory_bytes"],
                "evaluation_status": "exploratory",
            }
        )
    after = parameter_fingerprint(model)
    if before != after:
        raise RuntimeError("receiver parameters changed during eval")
    return {
        "substage": "eval",
        "split": "test",
        "evaluation_status": "exploratory",
        "evaluation_claim": EVALUATION_CLAIM,
        "n": len(test_rows),
        "cells": evaluated,
        "n_cells": len(evaluated),
        "selection": "all trained cells once; no test-based seed or head picking",
        "prepared_sha256": prepared["prepared_sha256"],
        "receiver_identity": portable_identity(receiver_digest),
        "receiver_fingerprint_before": before,
        "receiver_fingerprint_after": after,
        "device": args.device,
        "software": software_record(),
        "command": command_record(args),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    dispatch = {
        "interface": run_interface,
        "train": run_train,
        "eval": run_eval,
    }
    report = dispatch[args.substage](args)
    report["software"] = report.get("software") or software_record()
    _write_report(args.out, report)
    return 1 if report.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
