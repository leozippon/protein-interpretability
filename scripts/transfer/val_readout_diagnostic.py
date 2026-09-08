#!/usr/bin/env python3
"""Read-only val readout diagnostic runner. Not a numbered campaign stage.

H200: freeze with run_transfer_h200.sh --freeze-only, then
run_external_baseline_h200.sh --stage val_readout_diagnostic.py --expect report.json.

Accepts the wrapper's --device/--out. Does not register in STAGE_CONTRACTS.
No training, no test scoring, no nnsight/peft/Hub downloads.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.bridge_readout_diagnostic import (  # noqa: E402
    CONDITION,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BRIDGE_LABELS,
    DEFAULT_BUDGET,
    DEFAULT_DTYPE,
    DEFAULT_HEADS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_PROMPT_MODE,
    DEFAULT_SEEDS,
    DIAGNOSTIC_SCHEMA,
    EVALUATION_CLAIM,
    INTERFACE_MECHANICAL_RECORDS,
    PILOT_KIND,
    PILOT_NOTE,
    PRODUCTION_BRIDGE_PIN,
    PRODUCTION_CLASSIFIER_PIN,
    PRODUCTION_PREPARED_SHA256,
    PRODUCTION_RECEIVER_CONTENT_DIGEST,
    REPORT_SCHEMA,
    SUBSTAGES,
    assert_parameters_unchanged,
    bound_splits,
    class_digit_layout,
    direct_classifier_metrics,
    freeze_read_only,
    free_greedy_predictions,
    load_acceptance,
    qualify_bridges,
    refuse_existing_output,
    require_checkpoint_sha256,
    require_finite_module,
    require_roles,
    require_positive_limits,
    select_interface_records,
    suffix_invariance_check,
    teacher_forced_decomposition,
    toy_off_by_one_proof,
    validate_bridge_acceptance,
    validate_classifier_acceptance,
    write_report_json,
)
from src.transfer.io import sha256_file  # noqa: E402
from src.transfer.latent_bridge import (  # noqa: E402
    ALLOWED_DTYPES,
    ALLOWED_PROMPT_MODES,
    checkpoint_content_digest,
    command_record as latent_command_record,
    freeze_module,
    load_prepared_payload,
    load_receiver,
    parameter_fingerprint,
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
    parser.add_argument("--bridge-root", type=Path)
    parser.add_argument("--classifier-dir", type=Path)
    parser.add_argument("--bridge-acceptance", type=Path)
    parser.add_argument("--classifier-acceptance", type=Path)
    parser.add_argument("--receiver-path", type=Path)
    parser.add_argument(
        "--prompt-mode", choices=ALLOWED_PROMPT_MODES, default=DEFAULT_PROMPT_MODE
    )
    parser.add_argument("--dtype", choices=ALLOWED_DTYPES, default=DEFAULT_DTYPE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-interface-records", type=int, default=32)
    return parser


def _require(path: Path | None, name: str) -> Path:
    if path is None:
        raise ValueError(f"{name} is required for this substage")
    return path


def _load_stage49() -> Any:
    script = Path(__file__).resolve().with_name("49_classifier_handoff.py")
    spec = importlib.util.spec_from_file_location("classifier_handoff_stage49_readonly", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _qualify_heads(
    *,
    classifier_dir: Path,
    classifier_acceptance: list[dict[str, Any]],
    prepared_hash: str,
    cache_manifest: dict[str, Any],
    d_in: int,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    for cell in classifier_acceptance:
        path = classifier_dir / cell["dir"] / "classifier.pt"
        require_checkpoint_sha256(path, cell["checkpoint_sha256"])
    stage49 = _load_stage49()
    args = SimpleNamespace(
        classifier_dir=classifier_dir,
        heads=list(DEFAULT_HEADS),
        seeds=list(DEFAULT_SEEDS),
        trainable_budget=DEFAULT_BUDGET,
    )
    qualified = stage49._qualified_eval_heads(
        args, prepared_hash=prepared_hash, cache_manifest=cache_manifest, d_in=d_in
    )
    if len(qualified) != len(classifier_acceptance):
        raise ValueError("qualified classifier heads do not match the sealed matrix")
    frozen: list[dict[str, Any]] = []
    for (cell, head, checkpoint_sha), seal in zip(
        qualified, classifier_acceptance, strict=True
    ):
        if (cell["kind"], cell["seed"]) != (seal["kind"], seal["seed"]):
            raise ValueError("qualified head order does not match the sealed matrix")
        if checkpoint_sha != seal["checkpoint_sha256"]:
            raise ValueError("qualified classifier sha256 does not match the seal")
        require_finite_module(head, "classifier")
        config = dict(head.config())
        fingerprint = freeze_read_only(head.to(device))
        frozen.append(
            {
                "kind": cell["kind"],
                "seed": cell["seed"],
                "dir": cell["dir"],
                "checkpoint_sha256": checkpoint_sha,
                "head": head,
                "config": config,
                "actual_parameters": int(config["actual_parameters"]),
                "fingerprint": fingerprint,
                "identity": cell["identity"],
            }
        )
    return frozen


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the complete immutable input set before any model loader."""
    require_positive_limits(
        batch_size=args.batch_size, max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    if type(args.max_interface_records) is not int or not 1 <= args.max_interface_records <= 32:
        raise ValueError("max_interface_records must be in 1..32")
    paths = {
        name: _require(getattr(args, name), "--" + name.replace("_", "-"))
        for name in (
            "prepared", "cache", "bridge_root", "classifier_dir",
            "bridge_acceptance", "classifier_acceptance", "receiver_path",
        )
    }
    input_paths = {
        name: paths[name]
        for name in ("prepared", "cache", "bridge_acceptance", "classifier_acceptance")
    }
    input_paths["cache_manifest"] = paths["cache"].with_suffix(".json")
    input_paths["classifier_train_report"] = paths["classifier_dir"] / "report.json"
    input_hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    bridge_acceptance = load_acceptance(paths["bridge_acceptance"])
    replicas = validate_bridge_acceptance(bridge_acceptance, expected_pin=PRODUCTION_BRIDGE_PIN)
    classifier_acceptance = load_acceptance(paths["classifier_acceptance"])
    cells = validate_classifier_acceptance(
        classifier_acceptance, expected_pin=PRODUCTION_CLASSIFIER_PIN
    )
    checkpoints = [
        (f"bridge_{cell['seed']}", paths["bridge_root"] / cell["label"] / "bridge.pt", cell)
        for cell in replicas
    ] + [
        (f"{cell['kind']}_{cell['seed']}", paths["classifier_dir"] / cell["dir"] / "classifier.pt", cell)
        for cell in cells
    ]
    for name, path, seal in checkpoints:
        input_paths[name] = path
        input_hashes[name] = require_checkpoint_sha256(path, seal["checkpoint_sha256"])
    prepared = load_prepared_payload(paths["prepared"])
    if prepared["prepared_sha256"] != PRODUCTION_PREPARED_SHA256:
        raise ValueError("prepared_sha256 is not the frozen production cohort")
    splits = bound_splits(prepared, paths["cache"])
    if args.substage == "run":
        require_roles(splits["val_rows"], "val", "run records")
        if len(splits["val_rows"]) != 512:
            raise ValueError("run requires the frozen 512-row val split")
    receiver_digest = checkpoint_content_digest(paths["receiver_path"])
    if receiver_digest.get("content_digest") != PRODUCTION_RECEIVER_CONTENT_DIGEST:
        raise ValueError("receiver content digest is not the frozen production Instruct checkpoint")
    return {
        "paths": paths, "input_paths": input_paths, "input_hashes": input_hashes,
        "prepared": prepared, "splits": splits, "receiver_digest": receiver_digest,
        "bridge_acceptance": bridge_acceptance, "classifier_acceptance": classifier_acceptance,
        "classifier_cells": cells,
    }


def _load_context(args: argparse.Namespace) -> dict[str, Any]:
    context = _preflight(args)
    paths, prepared, splits = context["paths"], context["prepared"], context["splits"]
    receiver, tokenizer = load_receiver(
        paths["receiver_path"], device=args.device, dtype=args.dtype, prompt_mode=args.prompt_mode
    )
    freeze_module(receiver)
    bridges = qualify_bridges(
        bridge_root=paths["bridge_root"], acceptance=context["bridge_acceptance"],
        expected_pin=PRODUCTION_BRIDGE_PIN, prepared_hash=prepared["prepared_sha256"],
        cache_manifest=splits["manifest"], receiver_digest=context["receiver_digest"],
        prompt_mode=args.prompt_mode, device=args.device,
    )
    heads = _qualify_heads(
        classifier_dir=paths["classifier_dir"], classifier_acceptance=context["classifier_cells"],
        prepared_hash=prepared["prepared_sha256"], cache_manifest=splits["manifest"],
        d_in=int(splits["train_features"].shape[1]), device=args.device,
    )
    context.update(receiver=receiver, tokenizer=tokenizer, bridges=bridges, heads=heads)
    context["watchlist"] = _module_watchlist(context)
    context["layout"] = class_digit_layout(tokenizer, tokenizer.eos_token_id)
    return context


def _module_watchlist(context: dict[str, Any]) -> list[tuple[str, Any, str]]:
    watched = [("receiver", context["receiver"], parameter_fingerprint(context["receiver"]))]
    for bridge in context["bridges"]:
        watched.append((f"bridge_{bridge['seed']}", bridge["bridge"], bridge["fingerprint"]))
    for head in context["heads"]:
        watched.append(
            (f"{head['kind']}_{head['seed']}", head["head"], head["fingerprint"])
        )
    return watched


def _verify_unchanged(context: dict[str, Any]) -> dict[str, Any]:
    after = assert_parameters_unchanged(context["watchlist"])
    fingerprints = {
        name: {"before": before, "after": after[name]}
        for name, _module, before in context["watchlist"]
    }
    input_files = {}
    for name, path in context["input_paths"].items():
        digest = sha256_file(path)
        before = context["input_hashes"][name]
        if digest != before:
            raise RuntimeError(f"input file changed: {name}")
        input_files[name] = {"before_sha256": before, "after_sha256": digest}
    receiver_after = checkpoint_content_digest(context["paths"]["receiver_path"])["content_digest"]
    receiver_before = context["receiver_digest"]["content_digest"]
    if receiver_after != receiver_before:
        raise RuntimeError("receiver content digest changed")
    return {
        "fingerprints": fingerprints, "input_files": input_files,
        "receiver_content_digests": {"before": receiver_before, "after": receiver_after},
    }


def run_interface(args: argparse.Namespace) -> dict[str, Any]:
    refuse_existing_output(args.out)
    context = _load_context(args)
    splits = context["splits"]
    selection = select_interface_records(
        splits["train_rows"],
        max_interface_records=args.max_interface_records,
        mechanical_records=INTERFACE_MECHANICAL_RECORDS,
    )
    mechanical = selection["mechanical"]
    mechanical_feat = splits["train_features"][: selection["n_mechanical"]]
    off_by_one = toy_off_by_one_proof()
    layout = context["layout"]
    suffix_reports = []
    tf_reports = []
    greedy_reports = []
    for bridge in context["bridges"]:
        tf_reports.append(
            teacher_forced_decomposition(
                bridge=bridge["bridge"],
                receiver=context["receiver"],
                tokenizer=context["tokenizer"],
                features=mechanical_feat,
                records=mechanical,
                layout=layout,
                prompt_mode=args.prompt_mode,
                device=args.device,
                batch_size=min(args.batch_size, len(mechanical)),
                max_prompt_tokens=args.max_prompt_tokens,
            )
        )
        suffix_reports.append(
            suffix_invariance_check(
                bridge=bridge["bridge"],
                receiver=context["receiver"],
                tokenizer=context["tokenizer"],
                feature_row=mechanical_feat[0],
                record=mechanical[0],
                layout=layout,
                prompt_mode=args.prompt_mode,
                device=args.device,
                max_prompt_tokens=args.max_prompt_tokens,
            )
        )
        greedy_reports.append(
            free_greedy_predictions(
                bridge=bridge["bridge"],
                receiver=context["receiver"],
                tokenizer=context["tokenizer"],
                features=mechanical_feat,
                records=mechanical,
                prompt_mode=args.prompt_mode,
                device=args.device,
                batch_size=min(args.batch_size, len(mechanical)),
                max_prompt_tokens=args.max_prompt_tokens,
                max_new_tokens=args.max_new_tokens,
            )
        )
    direct_reports = [
        direct_classifier_metrics(
            head=item["head"], features=mechanical_feat, records=mechanical,
            device=args.device, batch_size=min(args.batch_size, len(mechanical)),
        )
        for item in context["heads"]
    ]
    unchanged = _verify_unchanged(context)
    gates = {
        "all_nine_qualified": len(context["bridges"]) == 3 and len(context["heads"]) == 6,
        "all_six_heads_exercised": len(direct_reports) == 6 and all(
            report["n"] == len(mechanical) for report in direct_reports
        ),
        "finite_teacher_forced": all(
            report["legacy_native"]["token_count"] > 0 for report in tf_reports
        ),
        "causal_shift_index": all(
            report["digit_logit_index"] == report["digit_label_position"] - 1
            for report in tf_reports
        ),
        "frozen_parameters_unchanged": True,
        "greedy_without_answer_ids": all(
            not report["answer_ids_fed"] for report in greedy_reports
        ),
        "suffix_pre_digit_stable": all(
            report["pre_digit_logits_close"] for report in suffix_reports
        ),
        "off_by_one_toy": off_by_one["detected"],
        "no_optimizer": True,
    }
    return {
        "schema_version": REPORT_SCHEMA,
        "diagnostic_schema": DIAGNOSTIC_SCHEMA,
        "pilot": PILOT_KIND,
        "pilot_note": PILOT_NOTE,
        "evaluation_claim": EVALUATION_CLAIM,
        "substage": "interface",
        "status": "passed" if all(gates.values()) else "failed",
        "condition": CONDITION,
        "role": "train",
        "n": selection["n_mechanical"],
        "selection": {
            key: value
            for key, value in selection.items()
            if key not in {"selected", "mechanical"}
        },
        "n_train": splits["n_train"],
        "n_val": splits["n_val"],
        "test_records_loaded_for_binding": True,
        "test_forwarded": False,
        "gates": gates,
        "off_by_one": off_by_one,
        "layout": {
            "prefix_len": layout["prefix_len"],
            "prefix_ids": layout["prefix_ids"],
            "digit_token_by_class": {
                str(key): value for key, value in layout["digit_token_by_class"].items()
            },
        },
        "bridges": [
            {
                "seed": item["seed"],
                "label": item["label"],
                "checkpoint_sha256": item["checkpoint_sha256"],
                "actual_parameters": item["actual_parameters"],
                "config": item["config"],
                "best_val_loss": item["best_val_loss"],
                "teacher_forced": tf,
                "greedy": {"metrics": greedy["metrics"], "answer_ids_fed": greedy["answer_ids_fed"]},
                "suffix": suffix,
            }
            for item, tf, greedy, suffix in zip(
                context["bridges"], tf_reports, greedy_reports, suffix_reports, strict=True
            )
        ],
        "heads": [
            {
                "kind": item["kind"],
                "seed": item["seed"],
                "dir": item["dir"],
                "checkpoint_sha256": item["checkpoint_sha256"],
                "actual_parameters": item["actual_parameters"],
                "config": item["config"],
                "direct": direct,
            }
            for item, direct in zip(context["heads"], direct_reports, strict=True)
        ],
        **unchanged,
        "receiver_content_digest": context["receiver_digest"]["content_digest"],
        "prepared_sha256": context["prepared"]["prepared_sha256"],
        "software": software_record(),
        "command": latent_command_record(args),
        "used_family_holdout": False,
    }


def run_run(args: argparse.Namespace) -> dict[str, Any]:
    refuse_existing_output(args.out)
    context = _load_context(args)
    splits = context["splits"]
    require_roles(splits["val_rows"], "val", "run records")
    if len(splits["val_rows"]) != 512:
        raise ValueError(f"run requires the frozen 512-row val split, got {len(splits['val_rows'])}")
    layout = context["layout"]
    val_rows = splits["val_rows"]
    val_feat = splits["val_features"]
    bridge_results = []
    for item in context["bridges"]:
        def _tf(current=item) -> dict[str, Any]:
            return teacher_forced_decomposition(
                bridge=current["bridge"],
                receiver=context["receiver"],
                tokenizer=context["tokenizer"],
                features=val_feat,
                records=val_rows,
                layout=layout,
                prompt_mode=args.prompt_mode,
                device=args.device,
                batch_size=args.batch_size,
                max_prompt_tokens=args.max_prompt_tokens,
            )

        def _greedy(current=item) -> dict[str, Any]:
            return free_greedy_predictions(
                bridge=current["bridge"],
                receiver=context["receiver"],
                tokenizer=context["tokenizer"],
                features=val_feat,
                records=val_rows,
                prompt_mode=args.prompt_mode,
                device=args.device,
                batch_size=args.batch_size,
                max_prompt_tokens=args.max_prompt_tokens,
                max_new_tokens=args.max_new_tokens,
            )

        tf, tf_s, tf_peak = timed_cuda(args.device, _tf)
        greedy, greedy_s, greedy_peak = timed_cuda(args.device, _greedy)
        legacy_nll = tf["legacy_native"]["token_weighted_nll"]
        bridge_results.append(
            {
                "seed": item["seed"],
                "label": item["label"],
                "checkpoint_sha256": item["checkpoint_sha256"],
                "actual_parameters": item["actual_parameters"],
                "config": item["config"],
                "best_val_loss": item["best_val_loss"],
                "legacy_minus_seal_best_val_loss": legacy_nll - float(item["best_val_loss"]),
                "teacher_forced": tf,
                "free_greedy": {
                    "metrics": greedy["metrics"],
                    "rows": greedy["rows"],
                    "answer_ids_fed": greedy["answer_ids_fed"],
                    "walltime_s": greedy_s,
                    "peak_memory_bytes": greedy_peak,
                },
                "teacher_forced_walltime_s": tf_s,
                "teacher_forced_peak_memory_bytes": tf_peak,
            }
        )
    head_results = []
    for item in context["heads"]:
        def _direct(current=item) -> dict[str, Any]:
            return direct_classifier_metrics(
                head=current["head"],
                features=val_feat,
                records=val_rows,
                device=args.device,
                batch_size=args.batch_size,
            )

        direct, elapsed, peak = timed_cuda(args.device, _direct)
        head_results.append(
            {
                "kind": item["kind"],
                "seed": item["seed"],
                "dir": item["dir"],
                "checkpoint_sha256": item["checkpoint_sha256"],
                "actual_parameters": item["actual_parameters"],
                "config": item["config"],
                "direct": {
                    "metrics": direct["metrics"],
                    "rows": direct["rows"],
                    "source": direct["source"],
                    "receiver_handoff_ran": False,
                },
                "walltime_s": elapsed,
                "peak_memory_bytes": peak,
            }
        )
    unchanged = _verify_unchanged(context)
    return {
        "schema_version": REPORT_SCHEMA,
        "diagnostic_schema": DIAGNOSTIC_SCHEMA,
        "pilot": PILOT_KIND,
        "pilot_note": PILOT_NOTE,
        "evaluation_claim": EVALUATION_CLAIM,
        "substage": "run",
        "status": "completed",
        "condition": CONDITION,
        "role": "val",
        "n": 512,
        "n_train": splits["n_train"],
        "n_val": splits["n_val"],
        "test_records_loaded_for_binding": True,
        "test_forwarded": False,
        "seeds": list(DEFAULT_SEEDS),
        "heads": list(DEFAULT_HEADS),
        "labels": list(DEFAULT_BRIDGE_LABELS),
        "layout": {
            "prefix_len": layout["prefix_len"],
            "prefix_ids": layout["prefix_ids"],
            "digit_token_by_class": {
                str(key): value for key, value in layout["digit_token_by_class"].items()
            },
        },
        "bridges": bridge_results,
        "classifiers": head_results,
        **unchanged,
        "receiver_content_digest": context["receiver_digest"]["content_digest"],
        "prepared_sha256": context["prepared"]["prepared_sha256"],
        "software": software_record(),
        "command": latent_command_record(args),
        "timing_note": (
            "timed_cuda is reported per component and is not an isolated peak "
            "or a wall-clock speedup claim"
        ),
        "used_family_holdout": False,
        "seed_selection": "all three seeds retained; none dropped",
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out = Path(args.out)
    dispatch = {"interface": run_interface, "run": run_run}
    report = dispatch[args.substage](args)
    report["software"] = report.get("software") or software_record()
    write_report_json(args.out, report)
    return 1 if report.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
