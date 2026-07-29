#!/usr/bin/env python3
"""Build the hash-bound executed P0-2 gate specification."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


R2_ROOT = Path(__file__).resolve().parents[1]
if str(R2_ROOT) not in sys.path:
    sys.path.insert(0, str(R2_ROOT))

from src.revision.dictionary_gate import load_strict_json, sha256_file  # noqa: E402


MODELS = ("protgpt2", "zymctrl", "progen2-medium")
METHODS = ("topk_clt", "relu_l1_sae", "gated_sae", "dense_low_rank")
SCREENED_METHODS = ("relu_l1_sae", "gated_sae")
SEEDS = (17, 29, 43)
SCREENING_SEED = 20260717
SPARSITY_MATCH_FAILURE = "sparsity_match_failure"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--mask-receipt", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--screening-root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def descriptor(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def selected_checkpoint(result: dict, *, result_path: Path) -> dict[str, str] | None:
    if result.get("status") == SPARSITY_MATCH_FAILURE:
        if "selected_checkpoint" in result:
            raise ValueError(f"{result_path}: failed screening selected a checkpoint")
        return None
    selected = result.get("selected_checkpoint")
    if not isinstance(selected, dict) or set(selected) != {"path", "sha256"}:
        raise ValueError(f"{result_path}: malformed selected checkpoint")
    checkpoint = descriptor(Path(selected["path"]))
    if checkpoint["sha256"] != selected["sha256"]:
        raise ValueError(f"{result_path}: selected checkpoint hash changed")
    return checkpoint


def run_entry(root: Path, model: str, method: str, seed: int) -> dict:
    run_root = root / model / method
    if seed != SCREENING_SEED:
        run_root /= f"seed_{seed}"
    result_path = run_root / "results.json"
    manifest_path = run_root / "run_manifest.json"
    result = load_strict_json(result_path)
    manifest = load_strict_json(manifest_path)
    identity = (model, method, seed)
    observed = (
        result.get("model_name"),
        result.get("method"),
        result.get("run_seed"),
    )
    manifest_observed = (
        manifest.get("model_name"),
        manifest.get("method"),
        manifest.get("run_seed"),
    )
    if observed != identity or manifest_observed != identity:
        raise ValueError(f"{run_root}: run identity changed")
    return {
        "model_name": model,
        "method": method,
        "run_seed": seed,
        "run_manifest": descriptor(manifest_path),
        "result": descriptor(result_path),
        "checkpoint": selected_checkpoint(result, result_path=result_path),
    }


def write_spec(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite executed gate spec: {output}")

    screening = [
        run_entry(args.screening_root, model, method, SCREENING_SEED)
        for model in MODELS
        for method in SCREENED_METHODS
    ]
    failed = {
        (entry["model_name"], entry["method"])
        for entry in screening
        if entry["checkpoint"] is None
    }
    full = [
        run_entry(args.full_root, model, method, seed)
        for model in MODELS
        for method in METHODS
        if (model, method) not in failed
        for seed in SEEDS
    ]

    spec = {
        "schema_version": "r2_p0_2_dictionary_gate_spec_v1",
        "confirmatory": True,
        "profile": descriptor(args.profile),
        "protocol": descriptor(args.protocol),
        "mask_validation_receipt": descriptor(args.mask_receipt),
        "caches": [
            {
                "model_name": model,
                "manifest": descriptor(args.cache_root / model / "manifest.json"),
            }
            for model in MODELS
        ],
        "screening_runs": screening,
        "full_runs": full,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(spec, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output


def main() -> None:
    path = write_spec(parse_args())
    print(f"spec={path}")
    print(f"spec_sha256={sha256_file(path)}")


if __name__ == "__main__":
    main()
