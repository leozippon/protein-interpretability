#!/usr/bin/env python3
"""Independent ProteinGym controller for the 13 text amino-acid string controls.

This commit implements ``census`` only. ``probe``, ``score``, and ``analyse``
are declared and refused. The controller is a dedicated door: it does not widen
``ARM_CORPUS``, ``SCOREABLE_ARMS``, or the three budget gates.

Queue runners inject ``--device`` and ``--out`` before the remaining arguments.
Census is tokenizer-only on CPU even when ``--device`` names a CUDA card. GPU
occupancy is not a scientific exclusion.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.precision_policy import TEXT_AA_FP32_V1  # noqa: E402
from src.transfer.probes import PROTEINGYM_ROOT  # noqa: E402
from src.transfer.text_aa_cohort import (  # noqa: E402
    UNIMPLEMENTED_PHASES,
    census_models,
    load_text_aa_boundary_table,
    text_aa_model_names,
)

CENSUS_ARTEFACT = "text_aa_census.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} does not exist")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _load_native_dms_extension() -> Any:
    path = REPO_ROOT / "scripts/transfer/native_dms_extension.py"
    spec = importlib.util.spec_from_file_location("native_dms_extension", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def freeze_request(
    *,
    lookup_path: Path,
    proteingym_dir: Path,
    wildtypes_path: Path,
    wildtypes_fasta_path: Path,
) -> dict[str, Any]:
    native = _load_native_dms_extension()
    lookup = _read_json(lookup_path)
    wildtypes = _read_json(wildtypes_path)
    frozen = native.freeze_lookup_cohort(
        lookup,
        proteingym_dir=proteingym_dir,
        wildtypes=wildtypes,
        wildtypes_path=wildtypes_path,
        wildtypes_fasta_path=wildtypes_fasta_path,
    )
    frozen["lookup_sha256"] = sha256_file(lookup_path)
    frozen["lookup_path"] = str(Path(lookup_path).resolve())
    frozen["proteingym_dir"] = str(Path(proteingym_dir).resolve())
    return frozen


def run_census(
    *,
    out: Path,
    lookup_path: Path,
    wildtypes_path: Path,
    wildtypes_fasta_path: Path,
    proteingym_dir: Path,
    models: Sequence[str],
    requested_device: str,
) -> dict[str, Any]:
    table = load_text_aa_boundary_table()
    names = list(models) if models else list(text_aa_model_names(table))
    request = freeze_request(
        lookup_path=lookup_path,
        proteingym_dir=proteingym_dir,
        wildtypes_path=wildtypes_path,
        wildtypes_fasta_path=wildtypes_fasta_path,
    )
    payload = census_models(names, request=request, table=table, repo_root=REPO_ROOT)
    payload["created_utc"] = _utc_now()
    payload["execution"] = {
        "phase": "census",
        "tokenizer_only": True,
        "weights_loaded": False,
        "cuda_invoked": False,
        "requested_device": str(requested_device),
        "executed_on": "cpu",
        "gpu_occupancy_is_not_a_scientific_exclusion": True,
    }
    payload["inputs"] = {
        "lookup_sha256": request["lookup_sha256"],
        "wildtypes_sha256": request["wildtypes_sha256"],
        "wildtypes_fasta_sha256": request["wildtypes_fasta_sha256"],
        "lookup_path": request["lookup_path"],
        "proteingym_dir": request["proteingym_dir"],
        "declared_assays": request["declared_assays"],
        "declared_clusters": request["declared_clusters"],
    }
    write_json(Path(out) / CENSUS_ARTEFACT, payload)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("census", *UNIMPLEMENTED_PHASES),
        help="flat positional phase so queue injection of --device/--out still parses",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cpu",
        help="accepted because campaign runners inject it; census does not use CUDA",
    )
    parser.add_argument(
        "--protocol",
        dest="protocol_id",
        default=TEXT_AA_FP32_V1,
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="repeatable; default is the full 13. A subset cannot claim common_all13",
    )
    parser.add_argument("--lookup", type=Path)
    parser.add_argument("--wildtypes", type=Path)
    parser.add_argument("--wildtypes-fasta", type=Path)
    parser.add_argument("--proteingym-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    if str(args.protocol_id) != TEXT_AA_FP32_V1:
        raise ValueError(
            f"text-AA controller accepts only {TEXT_AA_FP32_V1!r}, got {args.protocol_id!r}"
        )
    if args.phase != "census":
        raise SystemExit(
            f"{args.phase} is not implemented; this commit is census-only "
            f"(refused: {', '.join(UNIMPLEMENTED_PHASES)})"
        )
    if args.lookup is None or args.wildtypes is None or args.wildtypes_fasta is None:
        raise ValueError("census requires --lookup, --wildtypes, and --wildtypes-fasta")
    proteingym = Path(args.proteingym_dir) if args.proteingym_dir else PROTEINGYM_ROOT
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    return run_census(
        out=out,
        lookup_path=Path(args.lookup),
        wildtypes_path=Path(args.wildtypes),
        wildtypes_fasta_path=Path(args.wildtypes_fasta),
        proteingym_dir=proteingym,
        models=list(args.model),
        requested_device=str(args.device),
    )


if __name__ == "__main__":
    main()
