#!/usr/bin/env python3
"""M0/M1 EC-top-class latent-bridge runner. Independent of stage 35/36.

H200: freeze with run_transfer_h200.sh --freeze-only, then
run_external_baseline_h200.sh --stage 48_latent_bridge.py --expect report.json.

This script accepts the wrapper's fixed --device/--out and writes report.json.
It does not register in panel_contract STAGE_CONTRACTS. No nnsight, no peft,
no Hub downloads (local_files_only).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.io import write_json  # noqa: E402
from src.transfer.latent_bridge import (  # noqa: E402
    ALLOWED_DONORS,
    ALLOWED_DTYPES,
    ALLOWED_NN_METRICS,
    ALLOWED_PROMPT_MODES,
    BLOCK_INDEX_SEMANTICS,
    DIRECTION_MARKER,
    DONOR_RENDERING,
    K_SOFT_DEFAULT,
    KNOWN_STAGE34_QUEUE,
    PILOT_KIND,
    PILOT_NOTE,
    POOLING,
    REPORT_SCHEMA,
    SUBSTAGES,
    all_prepared_records,
    build_bridge_for_features,
    build_random_donor,
    cache_donor_features,
    command_record,
    CLASS_IDS,
    PreparedRecord,
    bridge_identity,
    checkpoint_content_digest,
    donor_token_ids,
    evaluation_mode,
    interface_optimizer_step,
    prediction_metrics,
    prediction_row,
    seeded_initialization,
    validate_bridge_identity,
    validate_cache_binding,
    donor_layer_index,
    embedding_std,
    generate_predictions,
    kmer_features,
    load_bridge,
    load_donor_arm,
    load_feature_cache,
    load_prepared_payload,
    load_receiver,
    nearest_neighbor_predict,
    prepare_cohort,
    prepared_records,
    save_bridge,
    save_feature_cache,
    software_record,
    target_json,
    timed_cuda,
    train_bridge,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--substage", required=True, choices=SUBSTAGES)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--cohort-json", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--bridge-checkpoint", type=Path)
    parser.add_argument("--donor", choices=ALLOWED_DONORS, default="progen2-small")
    parser.add_argument("--receiver-path", type=Path)
    parser.add_argument("--prompt-mode", choices=ALLOWED_PROMPT_MODES, default="plain")
    parser.add_argument("--dtype", choices=ALLOWED_DTYPES, default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--max-residues", type=int)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-interface-records", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--trainable-budget", type=int, default=3_000_000)
    parser.add_argument("--k-soft", type=int, default=K_SOFT_DEFAULT)
    parser.add_argument("--donor-layer", type=int)
    parser.add_argument("--donor-kind", choices=("pretrained", "random"), default="pretrained")
    parser.add_argument("--random-donor-seed", type=int, default=0)
    parser.add_argument(
        "--condition",
        choices=("pretrained_donor", "random_donor", "kmer", "raw_sequence", "oracle", "nn"),
        default="pretrained_donor",
    )
    parser.add_argument("--nn-metric", choices=ALLOWED_NN_METRICS, default="cosine")
    parser.add_argument(
        "--expect-records-sha256",
        default=KNOWN_STAGE34_QUEUE["records_sha256"],
    )
    parser.add_argument(
        "--expect-cohort-sha256",
        default=KNOWN_STAGE34_QUEUE["cohort_json_sha256"],
    )
    parser.add_argument(
        "--allow-unverified-cohort",
        action="store_true",
        help="skip the known stage-34 SHA-256 check (toy cohorts / tests)",
    )
    return parser


def _write_report(out: Path, payload: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    body = {
        "schema_version": REPORT_SCHEMA,
        "pilot": PILOT_KIND,
        "pilot_note": PILOT_NOTE,
        "block_index_semantics": BLOCK_INDEX_SEMANTICS,
        "first_round_claim": (
            "this run does not freeze epsilon and does not claim a positive "
            "capability result"
        ),
        **payload,
    }
    write_json(out / "report.json", body)


def _require(path: Path | None, name: str) -> Path:
    if path is None:
        raise ValueError(f"{name} is required for this substage")
    return path


def run_prepare(args: argparse.Namespace) -> dict[str, Any]:
    records = _require(args.records, "--records")
    cohort = _require(args.cohort_json, "--cohort-json")
    expect_r = None if args.allow_unverified_cohort else args.expect_records_sha256
    expect_c = None if args.allow_unverified_cohort else args.expect_cohort_sha256
    payload = prepare_cohort(
        records,
        cohort,
        seed=args.seed,
        max_residues=args.max_residues,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        expect_records_sha256=expect_r,
        expect_cohort_sha256=expect_c,
    )
    write_json(args.out / "prepared.json", payload)
    return {
        "substage": "prepare",
        "prepared": str(args.out / "prepared.json"),
        "exclusions": payload["exclusions"],
        "before_cap": payload["before_cap"],
        "after_cap": payload["after_cap"],
        "zero_support_classes_after_cap": payload["zero_support_classes_after_cap"],
        "ec7_note": payload["ec7_note"],
        "provenance": payload["provenance"],
        "software": software_record(),
        "command": command_record(args),
    }


def run_interface(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "substage": "interface", "status": "failed", "gates": {},
        "used_family_holdout": False, "device": args.device,
        "software": software_record(), "command": command_record(args),
    }
    gates = report["gates"]
    try:
        if not 1 <= args.max_interface_records <= 32:
            raise ValueError("M0 max-interface-records must be in 1..32")
        prepared = load_prepared_payload(_require(args.prepared, "--prepared"))
        # The monolithic JSON is hashed, but only this bounded train slice is materialized.
        train = [PreparedRecord.from_dict(row) for row in prepared["records"]["train"][:args.max_interface_records]]
        if not train or any(row.role != "train" for row in train):
            raise ValueError("M0 requires train records only")
        gates["bounded_train_only"] = True
        report.update(n_records=len(train), prepared_sha256=prepared["prepared_sha256"])
        donor = load_donor_arm(args.donor, device=args.device, dtype=args.dtype)
        receiver_path = _require(args.receiver_path, "--receiver-path")
        model, tokenizer = load_receiver(
            receiver_path, device=args.device, dtype=args.dtype, prompt_mode=args.prompt_mode
        )
        report["donor_identity"] = donor.checkpoint_digest
        report["receiver_identity"] = checkpoint_content_digest(receiver_path)
        layer = donor_layer_index(donor.n_layer, args.donor_layer)
        max_residues = args.max_residues if args.max_residues is not None else prepared["max_residues"]
        audit: list[dict[str, Any]] = []
        features, elapsed, peak = timed_cuda(args.device, lambda: cache_donor_features(
            donor=donor, records=train, device=args.device, layer=layer,
            max_tokens=args.max_tokens, max_residues=max_residues,
            batch_size=args.batch_size, audit=audit,
        ))
        report.update(content_audit=audit, donor_layer=layer, cache_walltime_s=elapsed,
                      cache_peak_memory_bytes=peak, rendering=DONOR_RENDERING, pooling=POOLING)
        gates["native_marker_full_residues_content_mask"] = True
        for limits in (
            {"max_tokens": len(train[0].sequence), "max_residues": None},
            {"max_tokens": args.max_tokens, "max_residues": len(train[0].sequence) - 1},
        ):
            try:
                donor_token_ids(donor.tokenizer, train[0], **limits)
            except ValueError:
                continue
            raise RuntimeError("M0 failed to refuse declared overlength input")
        gates["overlength_refused"] = True
        bridge = build_bridge_for_features(
            features, d_recv=int(model.get_input_embeddings().weight.shape[-1]),
            k=args.k_soft, budget=args.trainable_budget,
            output_scale=embedding_std(model.get_input_embeddings()), seed=args.seed,
        )
        step, elapsed, peak = timed_cuda(args.device, lambda: interface_optimizer_step(
            bridge=bridge, donor=donor.model, receiver=model, tokenizer=tokenizer,
            features=features, record=train[0], prompt_mode=args.prompt_mode,
            max_prompt_tokens=args.max_prompt_tokens, device=args.device,
        ))
        if step["fingerprints_before"]["donor"] != donor.checkpoint_digest["parameter_fingerprint"]:
            raise RuntimeError("donor parameters changed during feature extraction")
        report.update(optimizer_check=step, bridge_config=bridge.config(),
                      step_walltime_s=elapsed, step_peak_memory_bytes=peak)
        gates["finite_bridge_gradient_optimizer_step_frozen_endpoints"] = True
        # Seven synthetic label prompts, not seven extra cohort records. No sequence is shown.
        controls = [replace(train[0], accession=f"oracle_class_{cid}", ec_class=cid,
                            class_name=json.loads(target_json(cid))["name"], target_json=target_json(cid))
                    for cid in CLASS_IDS]
        rows, elapsed, peak = timed_cuda(args.device, lambda: generate_predictions(
            bridge=None, receiver=model, tokenizer=tokenizer, features=None,
            records=controls, prompt_mode=args.prompt_mode, user_mode="oracle", device=args.device,
            max_new_tokens=args.max_new_tokens, max_prompt_tokens=args.max_prompt_tokens, batch_size=1,
        ))
        report.update(oracle_rows=rows, oracle_metrics=prediction_metrics(rows),
                      oracle_note="seven fixed legal labels; format control only, not capability",
                      oracle_walltime_s=elapsed, oracle_peak_memory_bytes=peak)
        gates["oracle_all_seven_canonical_json"] = len(rows) == 7 and all(row["joint_correct"] for row in rows)
        if not gates["oracle_all_seven_canonical_json"]:
            raise RuntimeError("oracle-label format control failed; do not enter M1")
        report["status"] = "passed"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    return report


def _cache_records(prepared: dict[str, Any]) -> list[Any]:
    return all_prepared_records(prepared)


def run_cache(args: argparse.Namespace) -> dict[str, Any]:
    prepared = load_prepared_payload(_require(args.prepared, "--prepared"))
    records = _cache_records(prepared)
    if args.donor_kind == "pretrained":
        donor = load_donor_arm(args.donor, device=args.device, dtype=args.dtype)
    else:
        pretrained = load_donor_arm(args.donor, device=args.device, dtype=args.dtype)
        donor = build_random_donor(
            Path(pretrained.checkpoint_digest["checkpoint"]),
            pretrained.tokenizer,
            name=args.donor,
            seed=args.random_donor_seed,
            device=args.device,
            dtype=args.dtype,
        )
        if donor.checkpoint_digest["parameter_fingerprint"] == pretrained.checkpoint_digest[
            "parameter_fingerprint"
        ]:
            raise RuntimeError("random donor fingerprint matches pretrained; init failed")
    layer = donor_layer_index(donor.n_layer, args.donor_layer)

    def _run() -> Any:
        return cache_donor_features(
            donor=donor,
            records=records,
            device=args.device,
            layer=layer,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            max_residues=args.max_residues if args.max_residues is not None else prepared["max_residues"],
        )

    features, elapsed, peak = timed_cuda(args.device, _run)
    cache_path = args.out / "donor_cache.npz"
    manifest = {
        "data_hash": prepared["prepared_sha256"],
        "donor_name": donor.name,
        "donor_kind": donor.kind,
        "checkpoint_digest": donor.checkpoint_digest,
        "layer": layer,
        "n_layer": donor.n_layer,
        "pooling": POOLING,
        "rendering": DONOR_RENDERING,
        "direction_marker": DIRECTION_MARKER,
        "block_index_semantics": BLOCK_INDEX_SEMANTICS,
        "max_tokens": args.max_tokens,
        "max_residues": args.max_residues if args.max_residues is not None else prepared["max_residues"],
        "dtype": args.dtype,
        "roles": {role: len(prepared_records(prepared, role)) for role in ("train", "val", "test")},
    }
    save_feature_cache(cache_path, features, records, manifest)
    return {
        "substage": "cache",
        "cache": str(cache_path),
        "n": len(records),
        "d": int(features.shape[1]),
        "layer": layer,
        "donor_kind": donor.kind,
        "checkpoint_digest": donor.checkpoint_digest,
        "walltime_s": elapsed,
        "peak_memory_bytes": peak,
        "device": args.device,
        "timing_note": "elapsed after device synchronize; not a cross-host speedup",
        "software": software_record(),
        "command": command_record(args),
    }


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    prepared = load_prepared_payload(_require(args.prepared, "--prepared"))
    cache_path = _require(args.cache, "--cache")
    receiver_path = _require(args.receiver_path, "--receiver-path")
    all_rows = all_prepared_records(prepared)
    features, manifest = load_feature_cache(cache_path, all_rows)
    validate_cache_binding(args.condition, prepared["prepared_sha256"], manifest)
    train_rows = prepared_records(prepared, "train")
    val_rows = prepared_records(prepared, "val")
    train_feat = features[: len(train_rows)]
    val_feat = features[len(train_rows) : len(train_rows) + len(val_rows)]
    model, tokenizer = load_receiver(
        receiver_path, device=args.device, dtype=args.dtype, prompt_mode=args.prompt_mode
    )
    scale = embedding_std(model.get_input_embeddings())
    bridge = build_bridge_for_features(
        train_feat,
        d_recv=int(model.get_input_embeddings().weight.shape[-1]),
        k=args.k_soft,
        budget=args.trainable_budget,
        output_scale=scale,
        seed=args.seed,
    )
    identity = bridge_identity(
        condition=args.condition, prepared_hash=prepared["prepared_sha256"], cache_manifest=manifest,
        receiver_digest=checkpoint_content_digest(receiver_path), prompt_mode=args.prompt_mode,
        bridge_config=bridge.config(), seed=args.seed,
    )

    def _run() -> dict[str, Any]:
        return train_bridge(
            bridge=bridge,
            receiver=model,
            tokenizer=tokenizer,
            features=train_feat,
            records=train_rows,
            val_features=val_feat,
            val_records=val_rows,
            prompt_mode=args.prompt_mode,
            user_mode="latent",
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            grad_clip=args.grad_clip,
            seed=args.seed,
            max_prompt_tokens=args.max_prompt_tokens,
        )

    result, elapsed, peak = timed_cuda(args.device, _run)
    ckpt = args.out / "bridge.pt"
    save_bridge(
        ckpt,
        bridge,
        extra={
            "identity": identity,
            "cache_path": str(cache_path),
            "receiver_path": str(receiver_path),
            "trainable_budget": args.trainable_budget,
        },
    )
    write_json(args.out / "train_report.json", {**result, "test_metrics_computed": False})
    return {
        "substage": "train",
        "bridge_checkpoint": str(ckpt),
        "actual_parameters": bridge.trainable_parameter_count(),
        "trainable_budget": args.trainable_budget,
        "best_val_loss": result["best_val_loss"],
        "loss_unit": result["loss_unit"],
        "identity": identity,
        "history": result["history"],
        "walltime_s": elapsed,
        "peak_memory_bytes": peak,
        "device": args.device,
        "test_touched": False,
        "software": software_record(),
        "command": command_record(args),
    }


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    user_mode = evaluation_mode(args.condition)
    prepared = load_prepared_payload(_require(args.prepared, "--prepared"))
    receiver_path = _require(args.receiver_path, "--receiver-path")
    test_rows = prepared_records(prepared, "test")
    receiver_digest = checkpoint_content_digest(receiver_path)
    if user_mode == "latent":
        cache_path = _require(args.cache, "--cache")
        all_rows = all_prepared_records(prepared)
        features, manifest = load_feature_cache(cache_path, all_rows)
        offset = len(prepared_records(prepared, "train")) + len(prepared_records(prepared, "val"))
        test_feat = features[offset : offset + len(test_rows)]
        bridge, payload = load_bridge(_require(args.bridge_checkpoint, "--bridge-checkpoint"))
        expected = bridge_identity(
            condition=args.condition, prepared_hash=prepared["prepared_sha256"], cache_manifest=manifest,
            receiver_digest=receiver_digest, prompt_mode=args.prompt_mode,
            bridge_config=bridge.config(), seed=args.seed,
        )
        validate_bridge_identity(payload, expected)
    else:
        test_feat = None
        bridge = None
        if args.cache is not None or args.bridge_checkpoint is not None:
            raise ValueError("raw_sequence eval does not consume a cache/bridge")
        manifest = {}
    model, tokenizer = load_receiver(
        receiver_path, device=args.device, dtype=args.dtype, prompt_mode=args.prompt_mode
    )
    if bridge is not None and bridge.d_recv != int(model.get_input_embeddings().weight.shape[-1]):
        raise ValueError("bridge output width does not match receiver")

    def _run() -> list[dict[str, Any]]:
        return generate_predictions(
            bridge=bridge,
            receiver=model,
            tokenizer=tokenizer,
            features=test_feat,
            records=test_rows,
            prompt_mode=args.prompt_mode,
            user_mode=user_mode,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            batch_size=args.batch_size,
        )

    rows, elapsed, peak = timed_cuda(args.device, _run)
    metrics = prediction_metrics(rows)
    write_json(args.out / "predictions.json", {"rows": rows})
    return {
        "substage": "eval",
        "split": "test",
        "condition": args.condition,
        "metrics": metrics,
        "n": len(rows),
        "walltime_s": elapsed,
        "peak_memory_bytes": peak,
        "device": args.device,
        "checkpoint_digest": manifest.get("checkpoint_digest"),
        "receiver_identity": receiver_digest,
        "prepared_sha256": prepared["prepared_sha256"],
        "software": software_record(),
        "command": command_record(args),
    }


def run_baselines(args: argparse.Namespace) -> dict[str, Any]:
    prepared = load_prepared_payload(_require(args.prepared, "--prepared"))
    train_rows = prepared_records(prepared, "train")
    test_rows = prepared_records(prepared, "test")
    if args.condition == "nn":
        gallery = kmer_features([row.sequence for row in train_rows], k=3)
        query = kmer_features([row.sequence for row in test_rows], k=3)
        predicted = nearest_neighbor_predict(
            query,
            gallery,
            [row.ec_class for row in train_rows],
            [row.accession for row in train_rows],
            metric=args.nn_metric,
        )
        rows = [prediction_row(record, target_json(pred)) for record, pred in zip(test_rows, predicted)]
        metrics = prediction_metrics(rows)
        write_json(args.out / "predictions.json", {"rows": rows, "gallery": "fit_only"})
        return {
            "substage": "baselines",
            "condition": "nn",
            "nn_metric": args.nn_metric,
            "gallery": "fit_only_3mer_8000",
            "metrics": metrics,
            "software": software_record(),
            "command": command_record(args),
        }
    if args.condition == "kmer":
        all_rows = all_prepared_records(prepared)
        feats = kmer_features([row.sequence for row in all_rows], k=3)
        cache_path = args.out / "kmer_cache.npz"
        save_feature_cache(
            cache_path,
            feats.astype("float32"),
            all_rows,
            {
                "data_hash": prepared["prepared_sha256"],
                "donor_kind": "kmer",
                "checkpoint_digest": {"content_digest": "kmer_3_raw_frequency"},
                "layer": None,
                "pooling": None,
                "rendering": "sequence_only",
            },
        )
        return {
            "substage": "baselines",
            "condition": "kmer",
            "cache": str(cache_path),
            "d": 8000,
            "note": "3-mer frequencies; train with --cache this file and --condition kmer via train substage",
            "software": software_record(),
            "command": command_record(args),
        }
    raise ValueError(
        f"baselines substage handles nn and kmer; use eval for {args.condition}"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    dispatch = {
        "prepare": run_prepare,
        "interface": run_interface,
        "cache": run_cache,
        "train": run_train,
        "eval": run_eval,
        "baselines": run_baselines,
    }
    with seeded_initialization(args.seed):
        report = dispatch[args.substage](args)
    report["software"] = report.get("software") or software_record()
    _write_report(args.out, report)
    return 1 if report.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
