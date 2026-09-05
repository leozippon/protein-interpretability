#!/usr/bin/env python3
"""Fold a frozen generation/control JSONL shard with offline ESMFold assets."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import structure_evidence as se  # noqa: E402
from src.transfer.io import sha256_file, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", help="Logical device within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--min-length", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--trunk-passes", type=int, default=4, help="4 = released default; explicit API num_recycles=3")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--index-only", action="store_true", help="Combine all result objects into index.jsonl without GPU/model loading")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    if not 1 <= args.min_length <= args.max_length <= 1024 or args.chunk_size < 1 or args.trunk_passes < 1:
        parser.error("invalid supported lengths, chunk size or trunk passes")

    rows = se.load_cohort(args.cohort)
    args.out.mkdir(parents=True, exist_ok=True)
    # A resumed attempt must not leave its previous completion marker visible
    # while new work is running or a changed configuration is being rejected.
    (args.out / "structure_evidence.json").unlink(missing_ok=True)
    if args.index_only:
        manifests = sorted(args.out.glob("worker-*.json"))
        if not manifests:
            raise RuntimeError("no worker manifests")
        signatures = {json.loads(path.read_text())["evaluation_signature"] for path in manifests}
        cohorts = {json.loads(path.read_text())["cohort_sha256"] for path in manifests}
        if len(signatures) != 1 or cohorts != {sha256_file(args.cohort)}:
            raise ValueError("worker manifests disagree with cohort/configuration")
        summary = se.write_index(rows, args.out, signatures.pop(), shard_index=0, num_shards=1, filename="index.jsonl")
        print(json.dumps(summary), flush=True)
        if summary["status_counts"].get("pending", 0):
            raise SystemExit("incomplete evaluation: pending rows remain")
        write_json(args.out / "structure_evidence.json", summary)
        return

    import torch
    import transformers
    from transformers import EsmForProteinFolding

    runtime = (platform.python_version(), torch.__version__, transformers.__version__)
    if runtime != ("3.11.14", "2.9.1+cu128", "4.57.3"):
        raise RuntimeError(f"Require the validated ct Python/PyTorch/Transformers runtime; found {runtime}")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("ESMFold runtime requires an assigned CUDA GPU")
    torch.cuda.set_device(torch.device(args.device))
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    properties = torch.cuda.get_device_properties(device)
    if properties.major < 8:
        raise RuntimeError("GPU requires Ampere or newer")
    config = json.loads((args.model / "config.json").read_text())
    weights = [args.model / "pytorch_model.bin"]
    if not all(path.is_file() for path in weights):
        raise FileNotFoundError("Require complete official facebook/esmfold_v1 pytorch_model.bin")
    model_files = {path.name: sha256_file(path) for path in weights + [args.model / "config.json"]}
    if model_files != se.ESMFOLD_FILE_DIGESTS:
        raise ValueError("ESMFold checkpoint/configuration differs from the pinned official revision")
    contract = {
        "schema_version": se.SCHEMA_VERSION,
        "model_id": "facebook/esmfold_v1",
        "model_revision": se.ESMFOLD_REVISION,
        "model_files_sha256": model_files,
        "code_sha256": {"module": sha256_file(Path(se.__file__)), "runner": sha256_file(Path(__file__))},
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "scipy", "safetensors")},
        "python": platform.python_version(),
        "min_length": args.min_length,
        "max_length": args.max_length,
        "chunk_size": args.chunk_size,
        "trunk_passes": args.trunk_passes,
        "num_recycles_argument": args.trunk_passes - 1,
        "published_default_trunk_passes": config["esmfold_config"]["trunk"]["max_recycles"],
        "precision": "ESM stem float16; folding trunk float32; no autocast; TF32 disabled",
        "batch_size": 1,
        "ordering": "longest sequence first, SHA256 tie-break; exact sequence hash sharding",
        "seed": args.seed,
        "never_truncate": True,
        "plddt_conversion": "Transformers 4.57.3 raw [0,1] multiplied by 100 for metrics and PDB B factors",
        "raw_outputs": "atom37 positions/masks, raw and scaled atom pLDDT, CA pLDDT, full PAE, pTM, aatype, residue index; latent embeddings/logits omitted",
        "interpretation": se.INTERPRETATION,
    }
    signature = se.digest_json(contract)
    assigned = {se.sequence_digest(row["sequence"]): row["sequence"] for row in rows if se.shard_for(row["sequence"], args.num_shards) == args.shard_index}
    manifest_path = args.out / f"worker-{args.shard_index:03d}-of-{args.num_shards:03d}.json"
    manifest = {
        "evaluation_signature": signature, "contract": contract,
        "cohort_sha256": sha256_file(args.cohort),
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "unique_sequences_assigned": len(assigned),
        "device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": properties.name, "gpu_total_memory_bytes": properties.total_memory,
        "gpu_free_memory_before_bytes": torch.cuda.mem_get_info(device)[0],
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["evaluation_signature"] != signature or previous["cohort_sha256"] != manifest["cohort_sha256"]:
            raise ValueError("refusing to resume changed evaluation/cohort into same output")
    write_json(manifest_path, manifest)
    pending = []
    for sha, sequence in sorted(assigned.items(), key=lambda item: (-len(item[1]), item[0])):
        directory = args.out / "objects" / sha
        existing = se.load_result(directory, signature, sha)
        if existing is not None and not (args.retry_failed and existing["status"] == "failed"):
            continue
        reason = se.eligibility(sequence, min_length=args.min_length, max_length=args.max_length)
        if reason:
            write_json(directory / "result.json", {
                "schema_version": se.SCHEMA_VERSION, "evaluation_signature": signature,
                "sequence_sha256": sha, "length": len(sequence),
                "status": "not_evaluable", "reason": reason, "evaluator_truncated": False,
            })
        else:
            pending.append((sha, sequence))
    se.write_index(rows, args.out, signature, shard_index=args.shard_index, num_shards=args.num_shards)
    print(json.dumps({"shard": args.shard_index, "pending_unique_sequences": len(pending), "gpu": properties.name}), flush=True)
    if pending:
        model_started = time.monotonic()
        model, loading_info = EsmForProteinFolding.from_pretrained(
            args.model, local_files_only=True, low_cpu_mem_usage=True, output_loading_info=True,
        )
        # The released folding checkpoint omits the ESM auxiliary contact head.
        # EsmModel.forward returns embeddings; only predict_contacts calls that
        # separate head, and ESMFold never calls predict_contacts.
        allowed_missing = {"esm.contact_head.regression.weight", "esm.contact_head.regression.bias"}
        if (set(loading_info["missing_keys"]) - allowed_missing
                or loading_info["unexpected_keys"] or loading_info["mismatched_keys"] or loading_info["error_msgs"]):
            raise RuntimeError(f"unqualified checkpoint load: {loading_info}")
        manifest["checkpoint_loading"] = loading_info
        manifest["missing_head_note"] = "Only auxiliary ESM contact-prediction head may be absent; unused by folding forward"
        write_json(manifest_path, manifest)
        model.esm.half()
        model.trunk.set_chunk_size(args.chunk_size)
        model.eval().to(device)
        print(json.dumps({"model_loaded_seconds": time.monotonic() - model_started}), flush=True)
        for ordinal, (sha, sequence) in enumerate(pending, 1):
            directory = args.out / "objects" / sha
            started = time.monotonic()
            result = {
                "schema_version": se.SCHEMA_VERSION, "evaluation_signature": signature,
                "sequence_sha256": sha, "length": len(sequence), "evaluator_truncated": False,
            }
            torch.cuda.reset_peak_memory_stats(device)
            output = None
            try:
                ids = torch.tensor([[se.AA_ORDER.index(aa) for aa in sequence]], device=device, dtype=torch.long)
                with torch.inference_mode():
                    output = model(ids, attention_mask=torch.ones_like(ids), num_recycles=args.trunk_passes - 1)
                arrays = se.prediction_arrays(output)
                result.update(se.summarize_arrays(arrays))
                result["files_sha256"] = se.save_prediction(directory, arrays, se.pdb_from_arrays(arrays))
                result["status"] = "ok"
            except torch.cuda.OutOfMemoryError as error:
                result.update(status="failed", reason="cuda_out_of_memory", error=str(error)[:1500])
            except ValueError as error:
                result.update(status="failed", reason="invalid_prediction", error=str(error)[:1500])
            except Exception as error:
                result.update(status="failed", reason="unexpected_runtime_error", error=f"{type(error).__name__}: {error}"[:1500])
                write_json(directory / "result.json", result)
                raise
            finally:
                del output
                gc.collect()
                torch.cuda.empty_cache()
            result["elapsed_seconds"] = time.monotonic() - started
            result["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device)
            write_json(directory / "result.json", result)
            progress = {key: result[key] for key in (
                "sequence_sha256", "length", "status", "reason", "elapsed_seconds",
                "peak_gpu_memory_bytes", "mean_ca_plddt", "ptm",
            ) if key in result}
            print(json.dumps({"ordinal": ordinal, "total": len(pending), **progress}), flush=True)
    summary = se.write_index(rows, args.out, signature, shard_index=args.shard_index, num_shards=args.num_shards)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["summary"] = summary
    write_json(manifest_path, manifest)
    write_json(args.out / "structure_evidence.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
