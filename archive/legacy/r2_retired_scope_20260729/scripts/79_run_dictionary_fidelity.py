#!/usr/bin/env python3
"""Prepare and run the prospective P0-2b behavioral-fidelity qualification."""

from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import os
import sys
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


R2_ROOT = Path(__file__).resolve().parents[1]
if str(R2_ROOT) not in sys.path:
    sys.path.insert(0, str(R2_ROOT))

from src.models.model_loader import (  # noqa: E402
    MODEL_REGISTRY,
    load_model,
    verify_frozen_model_inference_dtype,
)
from src.revision.dictionary_controls import (  # noqa: E402
    build_windowed_transcoder,
    load_production_profile,
    load_strict_json,
)
from src.revision.dictionary_fidelity import (  # noqa: E402
    MODES,
    aggregate_variant,
    analysis_layer,
    checkpoint_state,
    cluster_bootstrap,
    fidelity_metrics,
    load_fidelity_spec,
    load_jsonl,
    per_sequence_scores,
    reconstruct_target,
    sequence_target_mask,
    source_layers_for_target,
    verify_artifact,
    verify_model_artifacts,
)
from src.revision.io import sha256_file, write_json, write_jsonl  # noqa: E402


MODELS = ("protgpt2", "zymctrl", "progen2-medium")
SEEDS = (17, 29, 43)
SPARSE_METHODS = ("topk_clt", "relu_l1_sae", "gated_sae")
METHODS = (*SPARSE_METHODS, "dense_low_rank")
EXPECTED_TERMINAL_METHODS = {
    ("protgpt2", "relu_l1_sae"),
    ("protgpt2", "gated_sae"),
    ("progen2-medium", "relu_l1_sae"),
}
AA20 = frozenset("ACDEFGHIKLMNPQRSTVWY")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    header: str | None = None
    sequence: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(sequence)
                header, sequence = line[1:].strip(), []
            elif header is None:
                raise ValueError("FASTA sequence encountered before a header")
            else:
                sequence.append(line)
    if header is not None:
        yield header, "".join(sequence)


def parse_zymctrl_record(header: str, body: str) -> dict[str, Any]:
    identifier = header.split(maxsplit=1)[0]
    fields = identifier.split("|")
    if len(fields) < 2 or not fields[1]:
        raise ValueError(f"cannot parse EC family from {identifier!r}")
    family = fields[1]
    prefix = f"{family}<sep><start>"
    if not body.startswith(prefix) or not body.endswith("<end>"):
        raise ValueError(f"invalid ZymCTRL body for {identifier!r}")
    sequence = body[len(prefix) : -len("<end>")].upper()
    if not sequence:
        raise ValueError(f"empty amino-acid sequence for {identifier!r}")
    return {
        "id": identifier,
        "source": "ZymCTRL_SwissProt_EC_P0_2b_20260727",
        "sequence": sequence,
        "split": "evaluation",
        "family": family,
        "sha256": sha256_text(sequence),
    }


def prepare_cohort(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite cohort: {output}")
    if (
        args.count < 1
        or args.min_length < 1
        or args.max_length < args.min_length
        or args.exclude_source_prefix_records < 0
    ):
        raise ValueError("invalid cohort selection bounds")
    source_sha = sha256_file(args.fasta)
    if args.source_sha256 is not None and source_sha != args.source_sha256:
        raise ValueError("source FASTA SHA-256 mismatch")
    excluded: set[str] = set()
    exclusion_descriptors: list[dict[str, Any]] = []
    for raw in args.exclude_jsonl:
        path = raw.resolve()
        rows = load_jsonl(path)
        excluded.update(str(row["sha256"]) for row in rows)
        exclusion_descriptors.append(
            {"path": str(path), "sha256": sha256_file(path), "rows": len(rows)}
        )
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    seen: set[str] = set()
    eligible = 0
    source_records = 0
    prefix_hashes: set[str] = set()
    for header, body in iter_fasta(args.fasta):
        source_records += 1
        row = parse_zymctrl_record(header, body)
        sequence_hash = row["sha256"]
        if source_records <= args.exclude_source_prefix_records:
            prefix_hashes.add(sequence_hash)
            excluded.add(sequence_hash)
        if not set(row["sequence"]) <= AA20:
            continue
        if not args.min_length <= len(row["sequence"]) <= args.max_length:
            continue
        if sequence_hash in seen:
            continue
        seen.add(sequence_hash)
        if sequence_hash in excluded:
            continue
        eligible += 1
        priority = sha256_text(f"p0_2b_20260727:{sequence_hash}")
        entry = (-int(priority, 16), priority, row)
        if len(candidates) < args.count:
            heapq.heappush(candidates, entry)
        elif int(priority, 16) < -candidates[0][0]:
            heapq.heapreplace(candidates, entry)
    if len(candidates) != args.count:
        raise ValueError(
            f"only {len(candidates)}/{args.count} untouched eligible sequences"
        )
    if source_records < args.exclude_source_prefix_records:
        raise ValueError("source FASTA is shorter than the frozen prefix exclusion")
    selected = [entry[2] for entry in candidates]
    selected.sort(
        key=lambda row: (sha256_text(f"p0_2b_20260727:{row['sha256']}"), row["id"])
    )
    if {row["sha256"] for row in selected} & excluded:
        raise AssertionError("evaluation cohort overlaps an excluded sequence")

    output.mkdir(parents=True)
    cohort_path = output / "evaluation.jsonl"
    write_jsonl(cohort_path, selected)
    write_json(
        output / "manifest.json",
        {
            "schema_version": "r2_dictionary_fidelity_cohort_v1",
            "status": "frozen_before_evaluation",
            "selection": "lowest_sha256_of_p0_2b_20260727_colon_sequence_sha256",
            "source_fasta": {
                "path": str(args.fasta.resolve()),
                "sha256": source_sha,
            },
            "excluded_manifests": exclusion_descriptors,
            "excluded_source_prefix": {
                "records": args.exclude_source_prefix_records,
                "unique_sequence_sha256": len(prefix_hashes),
                "reason": (
                    "reconstruct_and_exclude_all_ZymCTRL_source_records_that_could_"
                    "have_entered_the_20260724_transfer_gap_pilots"
                ),
            },
            "excluded_unique_sequences": len(excluded),
            "source_records": source_records,
            "eligible_unexcluded_unique_sequences": eligible,
            "minimum_sequence_length": args.min_length,
            "maximum_sequence_length": args.max_length,
            "rows": len(selected),
            "evaluation_jsonl": {
                "path": str(cohort_path),
                "sha256": sha256_file(cohort_path),
            },
            "builder_sha256": sha256_file(Path(__file__)),
        },
    )
    print(f"cohort={cohort_path}")
    print(f"cohort_sha256={sha256_file(cohort_path)}")


def prepare_mean(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite target mean: {output}")
    manifest_path = args.cache_manifest.resolve()
    manifest = load_strict_json(manifest_path)
    provenance = manifest.get("activation_provenance", {})
    if provenance.get("model_name") != args.model_name:
        raise ValueError("cache manifest model identity changed")
    layers = manifest["selected_layers"]
    if layers != list(range(len(layers))) or args.chunk_rows < 1:
        raise ValueError("cache layers or mean chunk size are invalid")
    target = analysis_layer(len(layers), args.analysis_layer_fraction)
    matching = [
        row
        for row in manifest["shards"]
        if row["split"] == "train" and row["layer"] == target
    ]
    if len(matching) != 1:
        raise ValueError("expected exactly one training target shard")
    shard = matching[0]
    target_path = (manifest_path.parent / shard["target_path"]).resolve()
    if sha256_file(target_path) != shard["target_sha256"]:
        raise ValueError("training target shard SHA-256 mismatch")
    values = np.load(target_path, mmap_mode="r", allow_pickle=False)
    if values.shape != (shard["rows"], shard["target_dim"]):
        raise ValueError("training target shard shape changed")
    total = np.zeros(values.shape[1], dtype=np.float64)
    for start in range(0, values.shape[0], args.chunk_rows):
        chunk = np.asarray(values[start : start + args.chunk_rows], dtype=np.float32)
        if not np.isfinite(chunk).all():
            raise FloatingPointError("training target shard contains non-finite values")
        total += chunk.sum(axis=0, dtype=np.float64)
    mean = (total / values.shape[0]).astype(np.float32)
    if mean.ndim != 1 or not np.isfinite(mean).all():
        raise FloatingPointError("target mean is not a finite vector")
    model_artifacts = {
        name: provenance[name]
        for name in (
            "model_config_sha256",
            "model_weights_sha256",
            "tokenizer_sha256",
        )
    }
    if any(
        not isinstance(digest, str) or len(digest) != 64
        for digest in model_artifacts.values()
    ):
        raise ValueError("cache manifest lacks frozen model-artifact digests")
    output.mkdir(parents=True)
    mean_path = output / "target_mean.npy"
    temporary = mean_path.with_name(f".{mean_path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        np.save(handle, mean, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, mean_path)
    write_json(
        output / "manifest.json",
        {
            "schema_version": "r2_dictionary_fidelity_target_mean_v1",
            "status": "complete",
            "model_name": args.model_name,
            "analysis_layer_fraction": args.analysis_layer_fraction,
            "target_layer": target,
            "rows": int(values.shape[0]),
            "dimension": int(values.shape[1]),
            "cache_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
                "content_sha256": manifest["content_sha256"],
            },
            "training_target_shard": {
                "path": str(target_path),
                "sha256": shard["target_sha256"],
            },
            "target_mean": {
                "path": str(mean_path),
                "sha256": sha256_file(mean_path),
            },
            "model_artifacts": model_artifacts,
            "builder_sha256": sha256_file(Path(__file__)),
        },
    )
    print(f"target_mean={mean_path}")
    print(f"target_mean_sha256={sha256_file(mean_path)}")


def artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def queue_for(model: str, method: str) -> int:
    if model == "protgpt2":
        return 0
    if model == "zymctrl":
        return 1 if method in {"topk_clt", "relu_l1_sae"} else 2
    if model == "progen2-medium":
        return 3
    raise ValueError(f"unknown model: {model}")


def build_spec(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite fidelity spec: {output}")
    gate_spec_path = args.gate_spec.resolve()
    gate_spec = load_strict_json(gate_spec_path)
    receipt_path = args.eligibility_receipt.resolve()
    receipt = load_strict_json(receipt_path)
    if (
        receipt.get("schema_version") != "r2_p0_2_eligibility_receipt_v1"
        or receipt.get("panel_status")
        != "one_or_more_model_method_quality_gates_failed"
        or receipt.get("spec_sha256") != sha256_file(gate_spec_path)
    ):
        raise ValueError("eligibility receipt does not bind the executed gate spec")
    profile_path = args.profile.resolve()
    profile_sha = gate_spec["profile"]["sha256"]
    profile = load_production_profile(profile_path, profile_sha)
    protocol_path = args.protocol.resolve()
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    cohort_manifest = load_strict_json(args.cohort_manifest)
    cohort_path = verify_artifact(
        cohort_manifest["evaluation_jsonl"], "evaluation cohort"
    )
    if (
        cohort_manifest.get("schema_version") != "r2_dictionary_fidelity_cohort_v1"
        or cohort_manifest.get("status") != "frozen_before_evaluation"
        or cohort_manifest.get("rows") != 240
        or cohort_manifest.get("minimum_sequence_length") != 64
        or cohort_manifest.get("maximum_sequence_length") != 246
        or cohort_manifest.get("excluded_source_prefix", {}).get("records") != 40_000
        or cohort_manifest.get("builder_sha256") != sha256_file(Path(__file__))
    ):
        raise ValueError("evaluation cohort was not frozen")
    if (
        args.analysis_layer_fraction != 0.5
        or args.minimum_ce_denominator != 0.05
        or args.minimum_kl_denominator != 0.01
        or args.bootstrap_samples != 1000
        or args.bootstrap_seed != 20260727
    ):
        raise ValueError("fidelity thresholds differ from the prospective amendment")

    means: dict[str, dict[str, Any]] = {}
    model_artifacts: dict[str, dict[str, str]] = {}
    for model in MODELS:
        path = (args.target_means_root / model / "manifest.json").resolve()
        value = load_strict_json(path)
        if (
            value.get("model_name") != model
            or value.get("analysis_layer_fraction") != args.analysis_layer_fraction
            or value.get("builder_sha256") != sha256_file(Path(__file__))
        ):
            raise ValueError(f"target mean identity changed for {model}")
        verify_artifact(value["target_mean"], f"{model} target mean")
        means[model] = {"manifest": artifact(path), "mean": value["target_mean"]}
        model_artifacts[model] = value["model_artifacts"]

    runs: list[dict[str, Any]] = []
    for entry in gate_spec["full_runs"]:
        model, method, seed = (
            entry["model_name"],
            entry["method"],
            entry["run_seed"],
        )
        if model not in MODELS or method not in METHODS or seed not in SEEDS:
            raise ValueError("executed P0-2 gate contains an unexpected run")
        runs.append({**entry, "queue": queue_for(model, method)})
    if len(runs) != 27:
        raise ValueError(f"expected 27 completed full runs, found {len(runs)}")
    run_identities = {
        (run["model_name"], run["method"], run["run_seed"]) for run in runs
    }
    if len(run_identities) != len(runs):
        raise ValueError("executed P0-2 gate contains duplicate full runs")
    terminal = [
        {
            "model_name": row["model_name"],
            "method": row["method"],
            "status": row["status"],
            "failure_reasons": row["failure_reasons"],
        }
        for row in receipt["model_method_adjudications"]
        if row["status"] == "sparsity_match_failure"
    ]
    if {(row["model_name"], row["method"]) for row in terminal} != (
        EXPECTED_TERMINAL_METHODS
    ):
        raise ValueError("terminal P0-2 method set differs from its completed receipt")

    payload = {
        "schema_version": "r2_dictionary_fidelity_spec_v1",
        "status": "frozen_before_evaluation",
        "confirmatory_scope": (
            "prospective_new_cohort_instrument_qualification_not_p0_2_regating"
        ),
        "p0_2_gate_spec": artifact(gate_spec_path),
        "p0_2_eligibility_receipt": artifact(receipt_path),
        "profile": artifact(profile_path),
        "protocol": artifact(protocol_path),
        "implementation": {
            "runner": artifact(Path(__file__)),
            "fidelity_module": artifact(
                R2_ROOT / "src" / "revision" / "dictionary_fidelity.py"
            ),
            "dictionary_module": artifact(
                R2_ROOT / "src" / "revision" / "dictionary_controls.py"
            ),
            "model_loader": artifact(R2_ROOT / "src" / "models" / "model_loader.py"),
        },
        "evaluation_cohort": {
            "manifest": artifact(args.cohort_manifest),
            "records": artifact(cohort_path),
            "rows": cohort_manifest["rows"],
        },
        "unavailable_model_methods": terminal,
        "target_means": means,
        "model_artifacts_by_model": model_artifacts,
        "model_input_format_by_model": {
            "protgpt2": "sequence",
            "zymctrl": "zymctrl_ec",
            "progen2-medium": "progen2_n_to_c",
        },
        "analysis_layer_fraction": args.analysis_layer_fraction,
        "tokenization": profile["cache_extraction"]["tokenization"],
        "scoring": {
            "target": "next_model_token_within_protein_sequence",
            "zymctrl_excludes": "EC_prefix_start_token_and_end_token",
            "other_models": "all_valid_next_token_targets",
            "intervention": "single_target_mlp_output_with_normal_downstream_recomputation",
            "modes": list(MODES),
            "minimum_mean_ablation_ce_delta_nats": args.minimum_ce_denominator,
            "minimum_mean_ablation_kl_nats": args.minimum_kl_denominator,
            "reinjection_max_absolute_ce_delta_nats": 1e-6,
            "reinjection_minimum_argmax_agreement": 1.0,
        },
        "bootstrap": {
            "cluster_unit": "sequence",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "fidelity_gate": {
            "loss_recovered_minimum_every_seed": 0.8,
            "kl_recovered_minimum_every_seed": 0.8,
            "applies_to": "revised_downstream_instrument_qualification_only",
            "does_not_modify": "original_frozen_P0_2_FVU_adjudication",
        },
        "runs": sorted(
            runs,
            key=lambda row: (
                row["queue"],
                row["model_name"],
                row["method"],
                row["run_seed"],
            ),
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    print(f"spec={output}")
    print(f"spec_sha256={sha256_file(output)}")


class TargetSplicer:
    def __init__(
        self,
        protein_model,
        dictionary,
        *,
        target_layer: int,
        activation_threshold: float,
        target_mean: torch.Tensor,
    ) -> None:
        self.protein_model = protein_model
        self.dictionary = dictionary
        self.target_layer = target_layer
        self.activation_threshold = activation_threshold
        self.target_mean = target_mean
        self.sources = source_layers_for_target(
            target_layer,
            n_layers=dictionary.n_layers,
            window=dictionary.window,
        )
        self.mode = "clean"
        self.captured: dict[int, torch.Tensor] = {}
        self.handles = []

    def __enter__(self):
        for layer in self.sources:
            mlp = self.protein_model._get_block(layer).mlp
            self.handles.append(mlp.register_forward_hook(self._hook(layer)))
        return self

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, layer: int):
        def hook(_module, inputs, output):
            if (
                len(inputs) < 1
                or not isinstance(inputs[0], torch.Tensor)
                or not isinstance(output, torch.Tensor)
            ):
                raise TypeError("MLP hook did not receive tensor input/output")
            if layer in self.captured:
                raise RuntimeError(f"duplicate MLP input capture at layer {layer}")
            self.captured[layer] = inputs[0]
            if layer != self.target_layer or self.mode in {"clean", "reinject"}:
                return output
            if self.mode == "mean_ablate":
                return self.target_mean.to(output.dtype).expand_as(output)
            if self.mode == "dictionary":
                replacement = reconstruct_target(
                    self.dictionary,
                    self.captured,
                    target_layer=self.target_layer,
                    activation_threshold=self.activation_threshold,
                )
                return replacement.to(output.dtype)
            raise ValueError(f"unknown splice mode: {self.mode}")

        return hook

    @torch.inference_mode()
    def forward(self, mode: str, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        if mode not in MODES:
            raise ValueError(f"unknown splice mode: {mode}")
        self.mode = mode
        self.captured = {}
        logits = self.protein_model.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        if set(self.captured) != set(self.sources):
            raise RuntimeError(
                "model forward did not capture the complete source window"
            )
        if not torch.isfinite(logits).all():
            raise FloatingPointError("model produced non-finite logits")
        return logits


def build_dictionary(profile: dict, result: dict, device: torch.device):
    model_name = result["model_name"]
    method = result["method"]
    geometry = profile["cache_extraction"]["model_cache_geometry"][model_name]
    selected = result["selected_validation_configuration"]
    dictionary = build_windowed_transcoder(
        method=method,
        n_layers=geometry["n_layers"],
        input_dim=geometry["input_dim"],
        target_dim=geometry["target_dim"],
        sparse_width=profile["panel"]["topk_clt"]["width"],
        dense_rank=profile["panel"]["dense_low_rank"]["rank"],
        window=profile["estimand"]["decoder_window"],
        l1_coefficient=selected["l1_coefficient"],
        gated_auxiliary_coefficient=selected["auxiliary_coefficient"],
        topk_k=profile["panel"]["topk_clt"]["k"],
    )
    return dictionary.to(device)


def tokenize(protein_model, texts: list[str], tokenization: dict[str, Any]):
    tokenizer = protein_model.tokenizer
    tokenizer.padding_side = tokenization["padding_side"]
    tokenizer.truncation_side = tokenization["truncation_side"]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    untruncated = tokenizer(
        texts,
        add_special_tokens=tokenization["add_special_tokens"],
        padding=False,
        truncation=False,
    )["input_ids"]
    if any(
        len(token_ids) > tokenization["max_model_tokens"] for token_ids in untruncated
    ):
        raise ValueError("native model input would be truncated")
    encoded = tokenizer(
        texts,
        add_special_tokens=tokenization["add_special_tokens"],
        padding=tokenization["padding"],
        truncation=tokenization["truncation"],
        max_length=tokenization["max_model_tokens"],
        return_attention_mask=True,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]


def format_fidelity_input(record: dict[str, Any], input_format: str) -> str:
    required = {"id", "source", "sequence", "split", "family", "sha256"}
    if set(record) != required or sha256_text(record["sequence"]) != record["sha256"]:
        raise ValueError("fidelity cohort record identity changed")
    if input_format == "sequence":
        return record["sequence"]
    if input_format == "zymctrl_ec":
        return f"{record['family']}<sep><start>{record['sequence']}<end>"
    if input_format == "progen2_n_to_c":
        return f"1{record['sequence']}"
    raise ValueError(f"unknown fidelity model-input format: {input_format}")


def evaluate_run(
    protein_model,
    dictionary,
    *,
    model_name: str,
    records: list[dict[str, Any]],
    input_format: str,
    tokenization: dict[str, Any],
    target_layer: int,
    activation_threshold: float,
    target_mean: torch.Tensor,
    batch_size: int,
    scoring: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    rows = {mode: [] for mode in ("reinject", "mean_ablate", "dictionary")}
    maximum_logit_delta = 0.0
    start_id = end_id = None
    if model_name == "zymctrl":
        start_id = protein_model.tokenizer.convert_tokens_to_ids("<start>")
        end_id = protein_model.tokenizer.convert_tokens_to_ids("<end>")
        if (
            start_id is None
            or end_id is None
            or start_id == end_id
            or start_id == protein_model.tokenizer.unk_token_id
            or end_id == protein_model.tokenizer.unk_token_id
        ):
            raise ValueError("ZymCTRL tokenizer lacks distinct start/end tokens")
    with TargetSplicer(
        protein_model,
        dictionary,
        target_layer=target_layer,
        activation_threshold=activation_threshold,
        target_mean=target_mean,
    ) as splicer:
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            texts = [format_fidelity_input(row, input_format) for row in batch_records]
            ids, mask = tokenize(protein_model, texts, tokenization)
            ids = ids.to(protein_model.device)
            mask = mask.to(protein_model.device)
            target_mask = sequence_target_mask(
                ids,
                mask,
                model_name=model_name,
                start_token_id=start_id,
                end_token_id=end_id,
            )
            clean = splicer.forward("clean", ids, mask)
            for mode in ("reinject", "mean_ablate", "dictionary"):
                variant = splicer.forward(mode, ids, mask)
                batch_rows = per_sequence_scores(clean, variant, ids, target_mask)
                for record, row in zip(batch_records, batch_rows):
                    rows[mode].append({"record_sha256": record["sha256"], **row})
                if mode == "reinject":
                    maximum_logit_delta = max(
                        maximum_logit_delta,
                        float((clean.float() - variant.float()).abs().max()),
                    )
                del variant
            del clean, ids, mask, target_mask

    reinjection = aggregate_variant(rows["reinject"])
    mean = aggregate_variant(rows["mean_ablate"])
    dictionary_result = aggregate_variant(rows["dictionary"])
    metrics = fidelity_metrics(
        rows["dictionary"],
        rows["mean_ablate"],
        minimum_ce_denominator=scoring["minimum_mean_ablation_ce_delta_nats"],
        minimum_kl_denominator=scoring["minimum_mean_ablation_kl_nats"],
    )
    boot = cluster_bootstrap(
        rows["dictionary"],
        rows["mean_ablate"],
        samples=bootstrap["samples"],
        seed=bootstrap["seed"],
        minimum_ce_denominator=scoring["minimum_mean_ablation_ce_delta_nats"],
        minimum_kl_denominator=scoring["minimum_mean_ablation_kl_nats"],
    )
    reinjection_pass = (
        abs(reinjection["variant_ce_nats"] - reinjection["clean_ce_nats"])
        <= scoring["reinjection_max_absolute_ce_delta_nats"]
        and reinjection["argmax_agreement"]
        >= scoring["reinjection_minimum_argmax_agreement"]
        and maximum_logit_delta == 0.0
    )
    return {
        "reinjection": reinjection,
        "mean_ablation": mean,
        "dictionary": dictionary_result,
        "fidelity": metrics,
        "cluster_bootstrap": boot,
        "reinjection_max_absolute_logit_delta": maximum_logit_delta,
        "reinjection_gate_pass": reinjection_pass,
        "sequence_rows": rows,
    }


def run_queue(args: argparse.Namespace) -> None:
    spec_path = args.spec.resolve()
    spec = load_fidelity_spec(spec_path)
    spec_sha = sha256_file(spec_path)
    for label, descriptor in spec["implementation"].items():
        verify_artifact(descriptor, f"fidelity implementation {label}")
    verify_artifact(spec["p0_2_gate_spec"], "completed P0-2 gate spec")
    verify_artifact(spec["p0_2_eligibility_receipt"], "completed P0-2 receipt")
    verify_artifact(spec["protocol"], "P0-2b protocol")
    profile_path = verify_artifact(spec["profile"], "fidelity profile")
    profile = load_production_profile(profile_path, spec["profile"]["sha256"])
    verify_artifact(spec["evaluation_cohort"]["manifest"], "evaluation cohort manifest")
    records_path = verify_artifact(
        spec["evaluation_cohort"]["records"], "evaluation records"
    )
    records = load_jsonl(records_path)
    if len(records) != spec["evaluation_cohort"]["rows"]:
        raise ValueError("evaluation cohort row count changed")
    selected = [run for run in spec["runs"] if run["queue"] == args.queue_index]
    if not selected:
        raise ValueError(f"queue {args.queue_index} has no runs")
    models = {run["model_name"] for run in selected}
    if len(models) != 1:
        raise ValueError("one queue must contain exactly one base model")
    model_name = models.pop()
    verify_model_artifacts(
        Path(MODEL_REGISTRY[model_name]["path"]),
        spec["model_artifacts_by_model"][model_name],
    )
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None or not torch.cuda.is_available():
        raise RuntimeError("fidelity evaluation requires one explicit CUDA device")
    torch.cuda.set_device(device)
    started = time.perf_counter()
    protein_model = load_model(model_name, device=str(device), dtype=torch.bfloat16)
    dtype_receipt = verify_frozen_model_inference_dtype(protein_model, "bfloat16")
    geometry = profile["cache_extraction"]["model_cache_geometry"][model_name]
    if (
        protein_model.n_layers != geometry["n_layers"]
        or protein_model.d_model != geometry["input_dim"]
        or geometry["input_dim"] != geometry["target_dim"]
    ):
        raise ValueError("loaded base-model geometry differs from the frozen profile")
    target = analysis_layer(protein_model.n_layers, spec["analysis_layer_fraction"])
    mean_manifest_path = verify_artifact(
        spec["target_means"][model_name]["manifest"],
        f"{model_name} target-mean manifest",
    )
    mean_manifest = load_strict_json(mean_manifest_path)
    if mean_manifest["target_layer"] != target:
        raise ValueError("target mean layer differs from the frozen analysis layer")
    mean_path = verify_artifact(
        spec["target_means"][model_name]["mean"], f"{model_name} target mean"
    )
    mean = torch.from_numpy(np.load(mean_path, allow_pickle=False)).to(
        device=device, dtype=torch.float32
    )
    input_format = spec["model_input_format_by_model"][model_name]
    for run in selected:
        result_root = (
            output_root / model_name / run["method"] / f"seed_{run['run_seed']}"
        )
        result_path = result_root / "results.json"
        if result_path.exists():
            existing = load_strict_json(result_path)
            if existing.get("spec_sha256") != spec_sha:
                raise ValueError(f"existing result has a different spec: {result_path}")
            print(f"verified_existing={result_path}")
            continue
        result_source = verify_artifact(
            run["result"], f"{model_name}/{run['method']}/{run['run_seed']} result"
        )
        verify_artifact(
            run["run_manifest"],
            f"{model_name}/{run['method']}/{run['run_seed']} run manifest",
        )
        result = load_strict_json(result_source)
        if (
            result.get("model_name") != model_name
            or result.get("method") != run["method"]
            or result.get("run_seed") != run["run_seed"]
            or len(result.get("candidate_validation", [])) != 1
        ):
            raise ValueError("completed dictionary result identity changed")
        dictionary = build_dictionary(profile, result, device)
        candidate_id = result["candidate_validation"][0]["candidate_id"]
        if result.get("selected_checkpoint") != run["checkpoint"]:
            raise ValueError("result and executed gate disagree on the checkpoint")
        checkpoint_path = Path(run["checkpoint"]["path"]).resolve()
        state = checkpoint_state(
            checkpoint_path,
            expected_sha256=run["checkpoint"]["sha256"],
            expected_candidate_id=candidate_id,
        )
        dictionary.load_state_dict(state["model_state_dict"])
        dictionary.eval()
        del state
        torch.cuda.reset_peak_memory_stats(device)
        run_started = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        measured = evaluate_run(
            protein_model,
            dictionary,
            model_name=model_name,
            records=records,
            input_format=input_format,
            tokenization=spec["tokenization"],
            target_layer=target,
            activation_threshold=result["selected_validation_configuration"][
                "activation_threshold"
            ],
            target_mean=mean,
            batch_size=args.batch_size,
            scoring=spec["scoring"],
            bootstrap=spec["bootstrap"],
        )
        gate = spec["fidelity_gate"]
        fidelity = measured["fidelity"]
        run_gate_pass = (
            measured["reinjection_gate_pass"]
            and fidelity["denominators_valid"]
            and fidelity["loss_recovered"] >= gate["loss_recovered_minimum_every_seed"]
            and fidelity["kl_recovered"] >= gate["kl_recovered_minimum_every_seed"]
        )
        payload = {
            "schema_version": "r2_dictionary_fidelity_result_v1",
            "status": "complete",
            "confirmatory_scope": spec["confirmatory_scope"],
            "spec_sha256": spec_sha,
            "model_name": model_name,
            "method": run["method"],
            "run_seed": run["run_seed"],
            "target_layer": target,
            "activation_threshold": result["selected_validation_configuration"][
                "activation_threshold"
            ],
            "checkpoint_sha256": run["checkpoint"]["sha256"],
            "source_result_sha256": run["result"]["sha256"],
            "source_run_manifest_sha256": run["run_manifest"]["sha256"],
            "evaluation_cohort_sha256": spec["evaluation_cohort"]["records"]["sha256"],
            "target_mean_sha256": spec["target_means"][model_name]["mean"]["sha256"],
            "run_fidelity_gate_pass": run_gate_pass,
            **measured,
            "resources": {
                "started_at_utc": started_at,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "wall_time_seconds": time.perf_counter() - run_started,
                "peak_accelerator_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "peak_accelerator_memory_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
            },
            "model_dtype_receipt": dtype_receipt,
        }
        result_root.mkdir(parents=True)
        write_json(result_path, payload)
        print(
            f"completed={model_name}/{run['method']}/{run['run_seed']} "
            f"loss_recovered={fidelity['loss_recovered']} "
            f"kl_recovered={fidelity['kl_recovered']}"
        )
        del dictionary
        gc.collect()
        torch.cuda.empty_cache()
    print(f"queue_wall_time_seconds={time.perf_counter() - started:.3f}")


def aggregate(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {output}")
    spec_path = args.spec.resolve()
    spec = load_fidelity_spec(spec_path)
    spec_sha = sha256_file(spec_path)
    results: dict[tuple[str, str, int], dict[str, Any]] = {}
    for run in spec["runs"]:
        path = (
            args.results_root
            / run["model_name"]
            / run["method"]
            / f"seed_{run['run_seed']}"
            / "results.json"
        )
        value = load_strict_json(path)
        identity = (run["model_name"], run["method"], run["run_seed"])
        if (
            value.get("spec_sha256") != spec_sha
            or (
                value.get("model_name"),
                value.get("method"),
                value.get("run_seed"),
            )
            != identity
        ):
            raise ValueError(f"result identity changed: {path}")
        results[identity] = value
    model_methods = []
    for model in MODELS:
        present_methods = sorted(
            {method for observed_model, method, _ in results if observed_model == model}
        )
        for method in present_methods:
            runs = [results[(model, method, seed)] for seed in SEEDS]
            all_seed_pass = all(run["run_fidelity_gate_pass"] for run in runs)
            model_methods.append(
                {
                    "model_name": model,
                    "method": method,
                    "all_seed_fidelity_gate_pass": all_seed_pass,
                    "revised_downstream_eligible": (
                        method in SPARSE_METHODS and all_seed_pass
                    ),
                    "runs": [
                        {
                            "run_seed": run["run_seed"],
                            "loss_recovered": run["fidelity"]["loss_recovered"],
                            "kl_recovered": run["fidelity"]["kl_recovered"],
                            "run_fidelity_gate_pass": run["run_fidelity_gate_pass"],
                            "result_sha256": sha256_file(
                                args.results_root
                                / model
                                / method
                                / f"seed_{run['run_seed']}"
                                / "results.json"
                            ),
                        }
                        for run in runs
                    ],
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        output,
        {
            "schema_version": "r2_dictionary_fidelity_panel_v1",
            "status": "complete",
            "confirmatory_scope": spec["confirmatory_scope"],
            "spec_sha256": spec_sha,
            "required_run_count": len(spec["runs"]),
            "unavailable_model_methods": spec["unavailable_model_methods"],
            "model_method_adjudications": model_methods,
            "eligible_sparse_model_methods": [
                {
                    "model_name": row["model_name"],
                    "method": row["method"],
                }
                for row in model_methods
                if row["revised_downstream_eligible"]
            ],
        },
    )
    print(f"aggregate={output}")
    print(f"aggregate_sha256={sha256_file(output)}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)

    cohort = subparsers.add_parser("prepare-cohort")
    cohort.add_argument("--fasta", type=Path, required=True)
    cohort.add_argument("--exclude-jsonl", type=Path, action="append", required=True)
    cohort.add_argument("--output-dir", type=Path, required=True)
    cohort.add_argument("--count", type=int, default=240)
    cohort.add_argument("--min-length", type=int, default=64)
    cohort.add_argument("--max-length", type=int, default=246)
    cohort.add_argument("--exclude-source-prefix-records", type=int, default=40_000)
    cohort.add_argument("--source-sha256")
    cohort.set_defaults(handler=prepare_cohort)

    mean = subparsers.add_parser("prepare-mean")
    mean.add_argument("--cache-manifest", type=Path, required=True)
    mean.add_argument("--model-name", choices=MODELS, required=True)
    mean.add_argument("--analysis-layer-fraction", type=float, default=0.5)
    mean.add_argument("--chunk-rows", type=int, default=8192)
    mean.add_argument("--output-dir", type=Path, required=True)
    mean.set_defaults(handler=prepare_mean)

    spec = subparsers.add_parser("build-spec")
    spec.add_argument("--gate-spec", type=Path, required=True)
    spec.add_argument("--eligibility-receipt", type=Path, required=True)
    spec.add_argument("--profile", type=Path, required=True)
    spec.add_argument("--protocol", type=Path, required=True)
    spec.add_argument("--cohort-manifest", type=Path, required=True)
    spec.add_argument("--target-means-root", type=Path, required=True)
    spec.add_argument("--analysis-layer-fraction", type=float, default=0.5)
    spec.add_argument("--minimum-ce-denominator", type=float, default=0.05)
    spec.add_argument("--minimum-kl-denominator", type=float, default=0.01)
    spec.add_argument("--bootstrap-samples", type=int, default=1000)
    spec.add_argument("--bootstrap-seed", type=int, default=20260727)
    spec.add_argument("--output", type=Path, required=True)
    spec.set_defaults(handler=build_spec)

    queue = subparsers.add_parser("run-queue")
    queue.add_argument("--spec", type=Path, required=True)
    queue.add_argument("--output-dir", type=Path, required=True)
    queue.add_argument("--queue-index", type=int, choices=range(4), required=True)
    queue.add_argument("--device", required=True)
    queue.add_argument("--batch-size", type=int, default=2)
    queue.set_defaults(handler=run_queue)

    panel = subparsers.add_parser("aggregate")
    panel.add_argument("--spec", type=Path, required=True)
    panel.add_argument("--results-root", type=Path, required=True)
    panel.add_argument("--output", type=Path, required=True)
    panel.set_defaults(handler=aggregate)
    return root


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
