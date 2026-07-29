"""Build immutable P0-3/P0-4 inputs from model and CLT checkpoints.

The builder performs no feature discovery and no biological hypothesis test.
It maps model tokens back to residues, writes residue-weighted sequence means
for the atlas, and writes continuous residue-level semantic controls whose
dense directions are fit only on the discovery cohort.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import platform
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .io import sha256_file, write_json


COHORT_FIELDS = {"id", "source", "sequence", "split", "family", "sha256"}
HEX = frozenset("0123456789abcdef")
UPPERCASE_RESIDUES = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
P0_2_SEEDS = (17, 29, 43)
P0_2_SPLITS = ("train", "validation", "test")
MODEL_INFERENCE_DTYPE_VERIFICATION = (
    "all_floating_model_parameters_exactly_declared_before_first_activation"
)
ACTIVATION_FINITE_CHECK = (
    "all_required_layer_captured_activation_and_logit_tensors_before_downstream_"
    "conversion_or_use"
)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def load_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=_reject_constant,
            object_pairs_hook=_strict_object,
        )


def load_jsonl(path: Path) -> list[Any]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {path}:{line_number}")
            try:
                rows.append(
                    json.loads(
                        line,
                        parse_constant=_reject_constant,
                        object_pairs_hook=_strict_object,
                    )
                )
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid JSONL row at {path}:{line_number}") from error
    if not rows:
        raise ValueError(f"empty JSONL file: {path}")
    return rows


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= HEX:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_keys(
    value: Any,
    required: set[str],
    optional: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing or extra:
        raise ValueError(f"{label} fields differ: missing={sorted(missing)}, extra={sorted(extra)}")
    return value


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tree_digest(root: Path, paths: Sequence[Path]) -> str:
    if not paths:
        raise ValueError(f"empty artifact class under {root}")
    return _canonical_digest(
        [
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(paths)
        ]
    )


def verify_model_artifacts(root: Path, expected: Mapping[str, str]) -> dict[str, str]:
    """Verify the deployed config, weight tree, and tokenizer/support tree."""

    root = Path(root).resolve()
    config = root / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"missing model config: {config}")
    files = [path for path in root.rglob("*") if path.is_file()]
    weights = [
        path
        for path in files
        if path.suffix == ".safetensors"
        or path.name.startswith("pytorch_model")
        and path.suffix in {".bin", ".json"}
        or path.name.startswith("model.safetensors")
    ]
    support = [path for path in files if path != config and path not in weights]
    observed = {
        "model_config_sha256": sha256_file(config),
        "model_weights_sha256": _tree_digest(root, weights),
        "tokenizer_sha256": _tree_digest(root, support),
    }
    if set(expected) != set(observed):
        raise ValueError("model_artifacts must contain exactly the three required digests")
    for field, digest in observed.items():
        if _require_sha256(expected[field], field) != digest:
            raise ValueError(f"deployed model artifact mismatch: {field}")
    return observed


def load_cohort(descriptor: Mapping[str, Any], base: Path) -> dict[str, Any]:
    descriptor = _require_keys(
        descriptor,
        {"role", "cohort_id", "path", "sha256", "split"},
        set(),
        "cohort descriptor",
    )
    role = descriptor["role"]
    if role not in {"discovery", "heldout"}:
        raise ValueError("cohort role must be discovery or heldout")
    if not isinstance(descriptor["cohort_id"], str) or not descriptor["cohort_id"]:
        raise ValueError("cohort_id must be a non-empty string")
    path = _resolve(descriptor["path"], base)
    expected_hash = _require_sha256(descriptor["sha256"], f"{role} cohort hash")
    if sha256_file(path) != expected_hash:
        raise ValueError(f"{role} cohort SHA-256 mismatch")
    rows = load_jsonl(path)
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != COHORT_FIELDS:
            raise ValueError(f"invalid frozen cohort fields at {path}:{index + 1}")
        if not all(isinstance(row[field], str) and row[field] for field in COHORT_FIELDS):
            raise ValueError(f"empty cohort value at {path}:{index + 1}")
        if row["split"] != descriptor["split"]:
            raise ValueError(f"cohort split mismatch at {path}:{index + 1}")
        digest = hashlib.sha256(row["sequence"].encode("utf-8")).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"sequence hash mismatch at {path}:{index + 1}")
        if row["id"] in seen_ids or digest in seen_hashes:
            raise ValueError(f"duplicate cohort id or sequence at {path}:{index + 1}")
        if not set(row["sequence"]) <= UPPERCASE_RESIDUES:
            raise ValueError(f"residue mapping requires uppercase alphabetic sequences: {row['id']}")
        seen_ids.add(row["id"])
        seen_hashes.add(digest)
    return {
        **descriptor,
        "path": path,
        "rows": rows,
        "sequence_hashes": [row["sha256"] for row in rows],
    }


def validate_disjoint(cohorts: Mapping[str, Mapping[str, Any]]) -> None:
    if set(cohorts) != {"discovery", "heldout"}:
        raise ValueError("exactly one discovery and one heldout cohort are required")
    if cohorts["discovery"]["cohort_id"] == cohorts["heldout"]["cohort_id"]:
        raise ValueError("discovery and heldout cohort IDs must differ")
    overlap = set(cohorts["discovery"]["sequence_hashes"]) & set(
        cohorts["heldout"]["sequence_hashes"]
    )
    if overlap:
        raise ValueError(f"discovery and heldout cohorts overlap by {len(overlap)} sequences")


def format_model_input(record: Mapping[str, str], input_format: str) -> str:
    if input_format == "sequence":
        return record["sequence"]
    if input_format == "zymctrl_ec":
        return f"{record['family']}<sep><start>{record['sequence']}<end>"
    raise ValueError(f"unknown model input format: {input_format}")


def residue_token_indices(tokenizer, input_ids: torch.Tensor, sequence: str) -> np.ndarray:
    """Map every residue to exactly one token through strict decoded alignment."""

    clean_pieces: list[str] = []
    owners: list[int] = []
    for token_index, token_id in enumerate(input_ids.detach().cpu().reshape(-1).tolist()):
        piece = tokenizer.decode([token_id], skip_special_tokens=True)
        clean = "".join(character for character in piece.upper() if character.isalpha())
        clean_pieces.append(clean)
        owners.extend([token_index] * len(clean))
    decoded = "".join(clean_pieces)
    starts = []
    cursor = 0
    while True:
        start = decoded.find(sequence, cursor)
        if start < 0:
            break
        starts.append(start)
        cursor = start + 1
    if len(starts) != 1:
        raise ValueError(
            f"sequence must occur exactly once in decoded model input; found {len(starts)}"
        )
    start = starts[0]
    mapping = np.asarray(owners[start : start + len(sequence)], dtype=np.int64)
    if mapping.shape != (len(sequence),):
        raise ValueError("token-to-residue alignment did not cover every residue")
    return mapping


def encode_layer(clt, layer: int, values: torch.Tensor) -> torch.Tensor:
    """Apply the checkpoint's exact ReLU-TopK encoder at one layer."""

    preactivation = F.relu(
        torch.einsum("bsd,fd->bsf", values, clt.W_enc[layer]) + clt.b_enc[layer]
    )
    top_values, top_indices = preactivation.topk(clt.k, dim=-1)
    sparse = torch.zeros_like(preactivation)
    return sparse.scatter_(-1, top_indices, top_values)


def select_residues(rows: Sequence[Mapping[str, str]], budget: int) -> list[tuple[str, int]]:
    """Select the lowest hash-priority residue identities with bounded memory."""

    if type(budget) is not int or budget < 1:
        raise ValueError("dense_fit_residue_budget must be a positive integer")
    eligible = sum(len(row["sequence"]) for row in rows)
    if eligible < budget:
        raise ValueError(f"dense fit budget {budget} exceeds {eligible} eligible residues")
    heap: list[tuple[int, str, int]] = []
    for row in rows:
        for position in range(len(row["sequence"])):
            priority = int(
                hashlib.sha256(f"{row['sha256']}:{position}".encode("ascii")).hexdigest(),
                16,
            )
            entry = (-priority, row["sha256"], position)
            if len(heap) < budget:
                heapq.heappush(heap, entry)
            elif priority < -heap[0][0]:
                heapq.heapreplace(heap, entry)
    return sorted((digest, position) for _, digest, position in heap)


def fit_dense_directions(
    sample: np.ndarray,
    n_components: int,
    *,
    seed: int,
    oversample: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit deterministic randomized PCA directions on discovery residues only."""

    sample = np.asarray(sample, dtype=np.float64)
    if sample.ndim != 2 or not np.isfinite(sample).all():
        raise ValueError("dense fit sample must be a finite matrix")
    n_rows, width = sample.shape
    if not 1 <= n_components <= min(n_rows - 1, width):
        raise ValueError("dense control dimension exceeds discovery sample rank bound")
    if type(oversample) is not int or oversample < 0:
        raise ValueError("dense_oversample must be a non-negative integer")
    center = sample.mean(axis=0)
    centered = sample - center
    sketch_width = min(n_rows, width, n_components + oversample)
    generator = np.random.default_rng(seed)
    sketch = centered @ generator.normal(size=(width, sketch_width))
    basis, _ = np.linalg.qr(sketch, mode="reduced")
    _, singular_values, right = np.linalg.svd(basis.T @ centered, full_matrices=False)
    if len(singular_values) < n_components or singular_values[n_components - 1] <= 1e-10:
        raise ValueError("discovery sample is rank-deficient for the dense control")
    directions = right[:n_components].T
    for column in range(n_components):
        anchor = int(np.argmax(np.abs(directions[:, column])))
        if directions[anchor, column] < 0:
            directions[:, column] *= -1
    return center.astype(np.float32), directions.astype(np.float32), singular_values[:n_components]


def random_dictionary(width: int, n_features: int, seed: int) -> np.ndarray:
    if not 1 <= n_features <= width:
        raise ValueError("random dictionary dimension must be within the model width")
    values = np.random.default_rng(seed).normal(size=(width, n_features))
    values /= np.linalg.norm(values, axis=0, keepdims=True)
    return values.astype(np.float32)


def _save_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def _save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _load_annotations(
    descriptor: Mapping[str, Any],
    base: Path,
    cohort: Mapping[str, Any],
    label_keys: set[str],
) -> tuple[dict[tuple[str, int], dict[str, int]], dict[str, Any]]:
    descriptor = _require_keys(descriptor, {"path", "sha256"}, set(), "annotation descriptor")
    path = _resolve(descriptor["path"], base)
    expected_hash = _require_sha256(descriptor["sha256"], "annotation SHA-256")
    if sha256_file(path) != expected_hash:
        raise ValueError("annotation SHA-256 mismatch")
    annotations: dict[tuple[str, int], dict[str, int]] = {}
    for row in load_jsonl(path):
        if not isinstance(row, dict) or set(row) != {"sequence_sha256", "position", "labels"}:
            raise ValueError("annotation rows require sequence_sha256, position, and labels")
        digest = _require_sha256(row["sequence_sha256"], "annotation sequence hash")
        position = row["position"]
        labels = row["labels"]
        if type(position) is not int or position < 0 or not isinstance(labels, dict):
            raise ValueError("invalid annotation position or labels")
        if set(labels) != label_keys or any(
            type(value) not in {int, bool} or int(value) not in {0, 1}
            for value in labels.values()
        ):
            raise ValueError("annotation labels must be exact configured binary keys")
        key = (digest, position)
        if key in annotations:
            raise ValueError(f"duplicate annotation row: {key}")
        annotations[key] = {name: int(value) for name, value in labels.items()}
    expected = {
        (row["sha256"], position)
        for row in cohort["rows"]
        for position in range(len(row["sequence"]))
    }
    if set(annotations) != expected:
        missing = len(expected - set(annotations))
        extra = len(set(annotations) - expected)
        raise ValueError(f"annotations do not exactly cover heldout residues: missing={missing}, extra={extra}")
    return annotations, {"path": str(path), "sha256": expected_hash, "rows": len(annotations)}


def _semantic_feature_names(
    model: str, run_seed: int, layer: int, features: Sequence[int]
) -> dict[str, list[str]]:
    return {
        "sparse_aligned": [
            f"{model}:S{run_seed}:L{layer}:F{feature}" for feature in features
        ],
        "dense_matched": [
            f"{model}:S{run_seed}:L{layer}:PC{index}"
            for index in range(len(features))
        ],
        "randomized_dictionary": [
            f"{model}:S{run_seed}:L{layer}:R{index}"
            for index in range(len(features))
        ],
    }


def _load_power_plan(
    descriptor: Mapping[str, Any],
    base: Path,
    *,
    feature_names: Mapping[str, Sequence[str]],
    label_names: set[str],
) -> dict[str, Any]:
    """Verify an independent pilot plan and its exact hypothesis coverage."""

    descriptor = _require_keys(
        descriptor, {"path", "sha256"}, set(), "prospective_power_plan"
    )
    if not isinstance(descriptor["path"], str):
        raise ValueError("prospective_power_plan.path must be a string")
    path = _resolve(descriptor["path"], base)
    expected_hash = _require_sha256(
        descriptor["sha256"], "prospective_power_plan.sha256"
    )
    if not path.is_file() or path.suffix != ".json":
        raise FileNotFoundError(f"prospective power plan must be a JSON file: {path}")
    if sha256_file(path) != expected_hash:
        raise ValueError("prospective power-plan SHA-256 mismatch")
    plan = load_json(path)
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise ValueError("prospective power plan requires schema_version 1")
    source = plan.get("independent_source")
    if not isinstance(source, dict):
        raise ValueError("prospective power plan requires independent_source")
    for field in ("description", "standard_error_method"):
        if not isinstance(source.get(field), str) or not source[field].strip():
            raise ValueError(f"independent_source.{field} must be non-empty")
    for field in ("run_manifest_sha256", "cohort_sha256"):
        _require_sha256(source.get(field), f"independent_source.{field}")
    if source.get("independent_of_confirmatory_data") is not True:
        raise ValueError("prospective power source must be independent of confirmatory data")
    rows = plan.get("standard_errors_delta_mse")
    if not isinstance(rows, list) or not rows:
        raise ValueError("prospective power plan requires standard_errors_delta_mse")
    expected = {
        (representation, feature, label, blocking)
        for representation, names in feature_names.items()
        for feature in names
        for label in label_names
        for blocking in ("protein", "family")
    }
    required = {
        "representation",
        "feature",
        "label",
        "blocking",
        "standard_error_delta_mse",
    }
    observed = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError(f"invalid prospective power-plan row {index}")
        key = tuple(
            str(row[field])
            for field in ("representation", "feature", "label", "blocking")
        )
        if key in observed:
            raise ValueError(f"duplicate prospective power-plan row: {key}")
        try:
            standard_error = float(row["standard_error_delta_mse"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid prospective standard error for {key}") from error
        if not np.isfinite(standard_error) or standard_error <= 0:
            raise ValueError(f"prospective standard error must be positive for {key}")
        observed.add(key)
    if observed != expected:
        raise ValueError(
            "prospective power plan must exactly cover all generated hypotheses; "
            f"missing={len(expected - observed)}, extra={len(observed - expected)}"
        )
    return {
        "path": path,
        "sha256": expected_hash,
        "independent_source": source,
        "n_standard_errors": len(observed),
    }


def _nonconfirmatory_checkpoint(
    model: Mapping[str, Any], dictionary: Mapping[str, Any], device: str
):
    """Load the small online-CLT fixture path; never eligible for confirmation."""

    from src.training.clt_trainer import CLTForTraining, verify_checkpoint_directory

    if model["confirmatory"] or dictionary["confirmatory"]:
        raise ValueError("online clt.pt checkpoints are forbidden for confirmatory inputs")
    checkpoint = Path(dictionary["checkpoint"])
    manifest_hash = _require_sha256(
        dictionary["checkpoint_manifest_sha256"],
        f"{model['name']} nonconfirmatory checkpoint manifest hash",
    )
    manifest_path = checkpoint / "checkpoint_manifest.json"
    if sha256_file(manifest_path) != manifest_hash:
        raise ValueError(f"checkpoint manifest SHA-256 mismatch for {model['name']}")
    manifest = verify_checkpoint_directory(checkpoint)
    config = yaml.safe_load((checkpoint / "config.yaml").read_text(encoding="utf-8"))
    if config.get("model", {}).get("name") != model["name"]:
        raise ValueError(f"checkpoint model identity mismatch for {model['name']}")
    state = torch.load(checkpoint / "clt.pt", map_location=device, weights_only=True)
    shape = state["W_enc"].shape
    clt_config = config["clt"]
    if (
        len(shape) != 3
        or int(shape[1]) != int(clt_config["d_clt"])
        or int(state["global_step"].item()) != manifest["step"]
    ):
        raise ValueError(f"checkpoint tensor/config/step mismatch for {model['name']}")
    clt = CLTForTraining(
        n_layers=int(shape[0]),
        d_model=int(shape[2]),
        d_clt=int(clt_config["d_clt"]),
        k=int(clt_config["k"]),
        window=int(clt_config.get("window", 8)),
    )
    clt.load_state_dict(state)
    provenance = {
        "schema_version": "r2_nonconfirmatory_online_clt_fixture_v1",
        "confirmatory": False,
        "model_name": model["name"],
        "method": "online_clt_fixture_only",
        "run_seed": dictionary["run_seed"],
        "eligibility_receipt_sha256": None,
        "run_manifest_sha256": manifest_hash,
        "checkpoint_sha256": sha256_file(checkpoint / "clt.pt"),
        "checkpoint_step": manifest["step"],
        "candidate_id": None,
        "source_manifest_sha256_by_split": None,
        "eligible_downstream_layers": [],
        "geometry": {
            "n_layers": clt.n_layers,
            "d_model": clt.d_model,
            "d_clt": clt.d_clt,
            "k": clt.k,
            "window": clt.window,
        },
        "checkpoint_directory": str(checkpoint),
        "checkpoint_manifest_sha256": manifest_hash,
        "checkpoint_kind": manifest["kind"],
        "checkpoint_config_sha256": sha256_file(checkpoint / "config.yaml"),
        "executed_checkpoint_model": config["model"]["name"],
    }
    return clt.to(device).eval(), provenance


def _eligible_checkpoint(
    model: Mapping[str, Any], dictionary: Mapping[str, Any], device: str
):
    """Load only a P0-2-authorized exact-cache best.pt artifact."""

    if not model["confirmatory"] or not dictionary["confirmatory"]:
        raise ValueError("eligible best.pt loading requires confirmatory model descriptors")
    from .dictionary_gate import load_eligible_topk_clt

    run_hashes = {
        record["run_seed"]: record["run_manifest_sha256"]
        for record in model["dictionaries"]
    }
    checkpoint_hashes = {
        record["run_seed"]: record["checkpoint_sha256"]
        for record in model["dictionaries"]
    }
    clt, provenance = load_eligible_topk_clt(
        model["eligibility_receipt_path"],
        model["eligibility_receipt_sha256"],
        model_name=model["name"],
        run_seed=dictionary["run_seed"],
        checkpoint_path=dictionary["checkpoint"],
        expected_run_manifest_sha256_by_seed=run_hashes,
        expected_checkpoint_sha256_by_seed=checkpoint_hashes,
        expected_source_manifest_sha256_by_split=model[
            "source_manifest_sha256_by_split"
        ],
        requested_layers=model["layers"],
        map_location=device,
    )
    expected = {
        "schema_version": "r2_p0_2_eligible_topk_load_v1",
        "model_name": model["name"],
        "method": "topk_clt",
        "run_seed": dictionary["run_seed"],
        "eligibility_receipt_sha256": model["eligibility_receipt_sha256"],
        "run_manifest_sha256": dictionary["run_manifest_sha256"],
        "checkpoint_sha256": dictionary["checkpoint_sha256"],
        "source_manifest_sha256_by_split": model[
            "source_manifest_sha256_by_split"
        ],
    }
    if any(provenance.get(field) != value for field, value in expected.items()):
        raise ValueError("eligible TopK loader returned mismatched P0-2 provenance")
    if not set(model["layers"]) <= set(provenance["eligible_downstream_layers"]):
        raise ValueError("requested input-builder layers exceed the P0-2 allowlist")
    return clt.to(device).eval(), {**provenance, "confirmatory": True}


def _resource_snapshot(device: str) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
    }
    if device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info(torch.device(device))
        snapshot.update(
            {
                "accelerator": torch.cuda.get_device_name(torch.device(device)),
                "accelerator_free_bytes": int(free),
                "accelerator_total_bytes": int(total),
            }
        )
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        fields = {}
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            name, value = line.split(":", 1)
            if name in {"MemTotal", "MemAvailable"}:
                fields[name] = int(value.strip().split()[0]) * 1024
        snapshot["host_memory_bytes"] = fields
    return snapshot


def _centered_kmer(sequence: str, position: int) -> str:
    padded = "^" + sequence + "$"
    return padded[position : position + 3]


def _validate_models(
    spec: Mapping[str, Any],
    base: Path,
    *,
    confirmatory: bool,
    eligibility_receipt: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    records = spec.get("models")
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError("models must contain exactly three atlas models")
    output = []
    names: set[str] = set()
    common = {
        "name",
        "model_root",
        "model_artifacts",
        "layers",
        "input_format",
        "model_inference_dtype",
        "model_inference_dtype_verification",
        "activation_finiteness_check",
    }
    for index, record in enumerate(records):
        record = _require_keys(
            record,
            common
            | (
                {"dictionaries", "source_manifest_sha256_by_split"}
                if confirmatory
                else {"nonconfirmatory_fixture_checkpoint"}
            ),
            set(),
            f"model {index}",
        )
        name = record["name"]
        layers = record["layers"]
        if (
            not isinstance(name, str)
            or SAFE_NAME.fullmatch(name) is None
            or name in names
        ):
            raise ValueError("model names must be distinct non-empty strings")
        if not isinstance(layers, list) or not layers or len(set(layers)) != len(layers):
            raise ValueError(f"{name} layers must be unique and non-empty")
        if any(type(layer) is not int or layer < 0 for layer in layers):
            raise ValueError(f"{name} layers must be non-negative integers")
        if record["model_inference_dtype"] not in {"float16", "bfloat16", "float32"}:
            raise ValueError(f"unsupported dtype for {name}")
        if (
            record["model_inference_dtype_verification"]
            != MODEL_INFERENCE_DTYPE_VERIFICATION
            or record["activation_finiteness_check"] != ACTIVATION_FINITE_CHECK
        ):
            raise ValueError(f"frozen inference verification methods differ for {name}")
        if confirmatory and record["model_inference_dtype"] != "bfloat16":
            raise ValueError(f"confirmatory model inference must use bfloat16: {name}")
        dictionaries = []
        source_hashes = None
        if confirmatory:
            if eligibility_receipt is None:
                raise ValueError("confirmatory models require a P0-2 eligibility receipt")
            source_hashes = record["source_manifest_sha256_by_split"]
            if not isinstance(source_hashes, dict) or set(source_hashes) != set(P0_2_SPLITS):
                raise ValueError(
                    f"{name} source manifest hashes must bind train/validation/test"
                )
            source_hashes = {
                split: _require_sha256(source_hashes[split], f"{name} {split} source hash")
                for split in P0_2_SPLITS
            }
            raw_dictionaries = record["dictionaries"]
            if not isinstance(raw_dictionaries, list) or len(raw_dictionaries) != len(
                P0_2_SEEDS
            ):
                raise ValueError(f"{name} must provide exact dictionaries for seeds 17, 29 and 43")
            seeds: set[int] = set()
            checkpoints: set[str] = set()
            run_manifests: set[str] = set()
            for dictionary_index, descriptor in enumerate(raw_dictionaries):
                descriptor = _require_keys(
                    descriptor,
                    {
                        "run_seed",
                        "checkpoint",
                        "checkpoint_sha256",
                        "run_manifest_sha256",
                    },
                    set(),
                    f"{name} dictionary {dictionary_index}",
                )
                seed = descriptor["run_seed"]
                if type(seed) is not int or seed not in P0_2_SEEDS or seed in seeds:
                    raise ValueError(f"{name} dictionary seeds must be exactly 17, 29 and 43")
                checkpoint = _resolve(descriptor["checkpoint"], base)
                checkpoint_sha256 = _require_sha256(
                    descriptor["checkpoint_sha256"], f"{name} seed {seed} checkpoint hash"
                )
                run_manifest_sha256 = _require_sha256(
                    descriptor["run_manifest_sha256"],
                    f"{name} seed {seed} run-manifest hash",
                )
                if checkpoint_sha256 in checkpoints:
                    raise ValueError(f"{name} seed checkpoints must be distinct")
                if run_manifest_sha256 in run_manifests:
                    raise ValueError(f"{name} seed run manifests must be distinct")
                if not checkpoint.is_file() or sha256_file(checkpoint) != checkpoint_sha256:
                    raise ValueError(f"{name} seed {seed} exact-cache best.pt SHA-256 mismatch")
                dictionaries.append(
                    {
                        "confirmatory": True,
                        "run_seed": seed,
                        "checkpoint": checkpoint,
                        "checkpoint_sha256": checkpoint_sha256,
                        "run_manifest_sha256": run_manifest_sha256,
                    }
                )
                seeds.add(seed)
                checkpoints.add(checkpoint_sha256)
                run_manifests.add(run_manifest_sha256)
            if seeds != set(P0_2_SEEDS):
                raise ValueError(f"{name} dictionary seeds must be exactly 17, 29 and 43")
        else:
            descriptor = _require_keys(
                record["nonconfirmatory_fixture_checkpoint"],
                {"directory", "manifest_sha256", "run_seed"},
                set(),
                f"{name} nonconfirmatory fixture checkpoint",
            )
            seed = descriptor["run_seed"]
            if type(seed) is not int or seed < 0:
                raise ValueError("nonconfirmatory fixture run_seed must be a non-negative integer")
            dictionaries.append(
                {
                    "confirmatory": False,
                    "run_seed": seed,
                    "checkpoint": _resolve(descriptor["directory"], base),
                    "checkpoint_manifest_sha256": _require_sha256(
                        descriptor["manifest_sha256"],
                        f"{name} nonconfirmatory fixture manifest hash",
                    ),
                }
            )
        names.add(name)
        output.append(
            {
                **record,
                "confirmatory": confirmatory,
                "model_root": _resolve(record["model_root"], base),
                "dictionaries": sorted(dictionaries, key=lambda item: item["run_seed"]),
                "source_manifest_sha256_by_split": source_hashes,
                "eligibility_receipt_path": (
                    None if eligibility_receipt is None else eligibility_receipt["path"]
                ),
                "eligibility_receipt_sha256": (
                    None if eligibility_receipt is None else eligibility_receipt["sha256"]
                ),
            }
        )
    return output


def _validate_semantics(
    spec: Mapping[str, Any],
    models: Mapping[str, Mapping[str, Any]],
    cohorts: Mapping[str, Mapping[str, Any]],
    base: Path,
) -> list[dict[str, Any]]:
    analyses = spec.get("semantics")
    if not isinstance(analyses, list) or not analyses:
        raise ValueError("semantics must contain at least one analysis")
    names: set[str] = set()
    output = []
    for index, analysis in enumerate(analyses):
        analysis = _require_keys(
            analysis,
            {
                "name",
                "model",
                "run_seed",
                "layer",
                "features",
                "fit_role",
                "evaluation_role",
                "dense_fit_residue_budget",
                "dense_seed",
                "dense_oversample",
                "random_seed",
                "annotation",
                "labels",
                "confirmatory",
                "covariates",
                "test",
            },
            {"prospective_power_plan"},
            f"semantics analysis {index}",
        )
        name = analysis["name"]
        model_name = analysis["model"]
        run_seed = analysis["run_seed"]
        layer = analysis["layer"]
        features = analysis["features"]
        if (
            not isinstance(name, str)
            or SAFE_NAME.fullmatch(name) is None
            or name in names
        ):
            raise ValueError("semantics names must be distinct non-empty strings")
        if model_name not in models or layer not in models[model_name]["layers"]:
            raise ValueError(f"semantics analysis {name} names an unavailable model/layer")
        available_seeds = {
            dictionary["run_seed"] for dictionary in models[model_name]["dictionaries"]
        }
        if type(run_seed) is not int or run_seed not in available_seeds:
            raise ValueError(
                f"semantics analysis {name} names an unavailable dictionary run_seed"
            )
        if (
            not isinstance(features, list)
            or not features
            or len(set(features)) != len(features)
            or any(type(feature) is not int or feature < 0 for feature in features)
        ):
            raise ValueError(f"semantics features must be unique non-negative indices: {name}")
        if analysis["fit_role"] != "discovery" or analysis["evaluation_role"] != "heldout":
            raise ValueError("semantic dense controls must fit on discovery and evaluate on heldout")
        if type(analysis["confirmatory"]) is not bool:
            raise ValueError("semantics confirmatory must be boolean")
        if analysis["confirmatory"] is not models[model_name]["confirmatory"]:
            raise ValueError(
                "semantic confirmation status must match the P0-2-gated builder status"
            )
        for seed_field in ("dense_seed", "random_seed"):
            if type(analysis[seed_field]) is not int:
                raise ValueError(f"{seed_field} must be an integer: {name}")
        covariates = analysis["covariates"]
        test = analysis["test"]
        if not isinstance(covariates, dict) or set(covariates) != {
            "position_degree",
            "kmer_hash_buckets",
        }:
            raise ValueError(f"invalid frozen covariate specification: {name}")
        if any(type(covariates[field]) is not int or covariates[field] < 1 for field in covariates):
            raise ValueError(f"covariate dimensions must be positive integers: {name}")
        required_test = {
            "n_folds",
            "n_permutations",
            "n_bootstrap",
            "ridge_alpha",
            "seed",
            "fdr_alpha",
            "power",
        }
        if not isinstance(test, dict) or set(test) != required_test:
            raise ValueError(f"invalid frozen semantic test specification: {name}")
        if any(type(test[field]) is not int for field in ("n_folds", "n_permutations", "n_bootstrap", "seed")):
            raise ValueError(f"semantic test count/seeds must be integers: {name}")
        if test["n_folds"] < 2 or test["n_permutations"] < 1 or test["n_bootstrap"] < 1:
            raise ValueError(f"semantic test counts are below executable minima: {name}")
        if any(type(test[field]) not in {int, float} for field in ("ridge_alpha", "fdr_alpha", "power")):
            raise ValueError(f"semantic continuous test parameters must be numeric: {name}")
        if test["ridge_alpha"] <= 0 or not 0 < test["fdr_alpha"] < 1 or not 0 < test["power"] < 1:
            raise ValueError(f"semantic continuous test parameters are out of range: {name}")
        if analysis["confirmatory"] and (
            test["n_permutations"] < 1_000 or test["n_bootstrap"] < 1_000
        ):
            raise ValueError(f"confirmatory semantic resampling requires at least 1,000 replicates: {name}")
        labels = analysis["labels"]
        if not isinstance(labels, list) or not labels:
            raise ValueError(f"semantics labels must be non-empty: {name}")
        label_names: set[str] = set()
        annotation_keys: set[str] = set()
        for label in labels:
            label = _require_keys(
                label,
                {
                    "name",
                    "family",
                    "annotation_key",
                    "construction",
                    "negative_name",
                    "negative_seed",
                },
                set(),
                f"label in {name}",
            )
            if any(not isinstance(label[field], str) or not label[field] for field in ("name", "family", "annotation_key", "construction", "negative_name")):
                raise ValueError(f"label strings must be non-empty: {name}")
            if type(label["negative_seed"]) is not int:
                raise ValueError(f"negative label seeds must be integers: {name}")
            if (
                label["name"] == label["negative_name"]
                or label["name"] in label_names
                or label["negative_name"] in label_names
            ):
                raise ValueError(f"duplicate semantic label name: {name}")
            label_names.update((label["name"], label["negative_name"]))
            annotation_keys.add(label["annotation_key"])
        power_reference = analysis.get("prospective_power_plan")
        if analysis["confirmatory"] and power_reference is None:
            raise ValueError(
                f"confirmatory semantics requires a prospective_power_plan: {name}"
            )
        feature_names = _semantic_feature_names(model_name, run_seed, layer, features)
        power_plan = (
            None
            if power_reference is None
            else _load_power_plan(
                power_reference,
                base,
                feature_names=feature_names,
                label_names=label_names,
            )
        )
        annotations, annotation_provenance = _load_annotations(
            analysis["annotation"], base, cohorts["heldout"], annotation_keys
        )
        selected = select_residues(
            cohorts["discovery"]["rows"], analysis["dense_fit_residue_budget"]
        )
        names.add(name)
        output.append(
            {
                **analysis,
                "annotations": annotations,
                "annotation_provenance": annotation_provenance,
                "power_plan": power_plan,
                "fit_selected": selected,
                "fit_selected_set": set(selected),
                "fit_rows": {},
                "target_sparse": [],
                "target_dense": [],
                "target_random": [],
                "input_norm": [],
            }
        )
    return output


def _default_model_loader(model: Mapping[str, Any], device: str):
    from src.models.model_loader import load_model

    return load_model(
        str(model["model_root"]),
        device=device,
        dtype=getattr(torch, model["model_inference_dtype"]),
    )


@torch.no_grad()
def _extract_model(
    model: Mapping[str, Any],
    dictionary: Mapping[str, Any],
    cohorts: Mapping[str, Mapping[str, Any]],
    semantic_states: Sequence[dict[str, Any]],
    device: str,
    max_model_tokens: int,
    model_loader: Callable[[Mapping[str, Any], str], Any],
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    artifacts = verify_model_artifacts(model["model_root"], model["model_artifacts"])
    clt, dictionary_provenance = (
        _eligible_checkpoint(model, dictionary, device)
        if model["confirmatory"]
        else _nonconfirmatory_checkpoint(model, dictionary, device)
    )
    protein_model = model_loader(model, device)
    from src.models.model_loader import (
        assert_finite_captured_activations,
        verify_frozen_model_inference_dtype,
    )

    inference_provenance = verify_frozen_model_inference_dtype(
        protein_model, model["model_inference_dtype"]
    )
    if protein_model.n_layers != clt.n_layers or protein_model.d_model != clt.d_model:
        raise ValueError(f"model/CLT geometry mismatch for {model['name']}")
    if any(layer >= clt.n_layers for layer in model["layers"]):
        raise ValueError(f"requested atlas layer exceeds checkpoint depth: {model['name']}")
    if any(feature >= clt.d_clt for state in semantic_states for feature in state["features"]):
        raise ValueError(f"semantic feature exceeds dictionary width: {model['name']}")

    atlas: dict[str, dict[int, np.ndarray]] = {}
    activation_capture_count = 0
    for role in ("discovery", "heldout"):
        atlas_rows = {layer: [] for layer in model["layers"]}
        active_states = [
            state
            for state in semantic_states
            if state["model"] == model["name"]
            and state["run_seed"] == dictionary["run_seed"]
            and role in {state["fit_role"], state["evaluation_role"]}
        ]
        for record in cohorts[role]["rows"]:
            text = format_model_input(record, model["input_format"])
            encoded = protein_model.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=True,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            if input_ids.shape != attention_mask.shape or input_ids.shape[0] != 1:
                raise ValueError("single-sequence tokenization returned invalid shapes")
            if input_ids.shape[1] > max_model_tokens:
                raise ValueError(f"model token limit exceeded by cohort record {record['id']}")
            residue_tokens = residue_token_indices(
                protein_model.tokenizer, input_ids, record["sequence"]
            )
            token_index = torch.as_tensor(residue_tokens, device=device, dtype=torch.long)
            cache = protein_model.get_activations(input_ids, attention_mask)
            assert_finite_captured_activations(cache)
            activation_capture_count += 1
            required_layers = set(model["layers"]) | {
                state["layer"] for state in active_states
            }
            for layer in sorted(required_layers):
                raw = cache.clt_input[layer].float()
                sparse = encode_layer(clt, layer, raw)[0].index_select(0, token_index)
                raw_residue = raw[0].index_select(0, token_index)
                if layer in atlas_rows:
                    atlas_rows[layer].append(sparse.mean(dim=0).cpu().numpy())
                for state in active_states:
                    if state["layer"] != layer:
                        continue
                    if role == state["fit_role"]:
                        raw_cpu = raw_residue.cpu().numpy().astype(np.float32)
                        for position in range(len(record["sequence"])):
                            key = (record["sha256"], position)
                            if key in state["fit_selected_set"]:
                                state["fit_rows"][key] = raw_cpu[position]
                    else:
                        indices = torch.as_tensor(
                            state["features"], device=device, dtype=torch.long
                        )
                        raw_cpu = raw_residue.cpu().numpy().astype(np.float32)
                        state["target_sparse"].append(
                            sparse.index_select(1, indices).cpu().numpy().astype(np.float32)
                        )
                        state["target_dense"].append(
                            ((raw_cpu - state["dense_center"]) @ state["dense_directions"])
                            .astype(np.float32)
                        )
                        state["target_random"].append(
                            np.maximum(raw_cpu @ state["random_directions"], 0.0)
                            .astype(np.float32)
                        )
                        state["input_norm"].append(
                            raw_residue.norm(dim=1).cpu().numpy().astype(np.float32)
                        )
        atlas[role] = {
            layer: np.stack(rows).astype(np.float32)
            for layer, rows in atlas_rows.items()
        }
        if role == "discovery":
            for state in active_states:
                if set(state["fit_rows"]) != state["fit_selected_set"]:
                    raise ValueError(f"dense fit selection was not fully extracted: {state['name']}")
                sample = np.stack([state["fit_rows"][key] for key in state["fit_selected"]])
                center, directions, singular = fit_dense_directions(
                    sample,
                    len(state["features"]),
                    seed=state["dense_seed"],
                    oversample=state["dense_oversample"],
                )
                state["dense_center"] = center
                state["dense_directions"] = directions
                state["dense_singular_values"] = singular
                state["random_directions"] = random_dictionary(
                    clt.d_model, len(state["features"]), state["random_seed"]
                )
    if activation_capture_count == 0:
        raise RuntimeError(f"no frozen-model activations were verified for {model['name']}")
    geometry = {
        "n_layers": clt.n_layers,
        "d_model": clt.d_model,
        "d_clt": clt.d_clt,
        "k": clt.k,
        "window": clt.window,
    }
    if verify_model_artifacts(model["model_root"], model["model_artifacts"]) != artifacts:
        raise RuntimeError(f"model artifacts changed during extraction: {model['name']}")
    if model["confirmatory"]:
        if (
            sha256_file(dictionary["checkpoint"]) != dictionary["checkpoint_sha256"]
            or sha256_file(model["eligibility_receipt_path"])
            != model["eligibility_receipt_sha256"]
        ):
            raise RuntimeError(
                f"P0-2 receipt or exact-cache checkpoint changed during extraction: {model['name']}"
            )
    else:
        from src.training.clt_trainer import verify_checkpoint_directory

        verify_checkpoint_directory(dictionary["checkpoint"])
        if sha256_file(
            dictionary["checkpoint"] / "checkpoint_manifest.json"
        ) != dictionary["checkpoint_manifest_sha256"]:
            raise RuntimeError(
                f"nonconfirmatory fixture checkpoint changed during extraction: {model['name']}"
            )
    del protein_model
    del clt
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return atlas, {
        "model_root": str(model["model_root"]),
        "model_artifacts": artifacts,
        "confirmatory": model["confirmatory"],
        "run_seed": dictionary["run_seed"],
        "checkpoint": str(dictionary["checkpoint"]),
        "checkpoint_sha256": dictionary_provenance["checkpoint_sha256"],
        "run_manifest_sha256": dictionary_provenance["run_manifest_sha256"],
        "eligibility_receipt_sha256": dictionary_provenance[
            "eligibility_receipt_sha256"
        ],
        "source_manifest_sha256_by_split": dictionary_provenance[
            "source_manifest_sha256_by_split"
        ],
        "checkpoint_step": dictionary_provenance["checkpoint_step"],
        "candidate_id": dictionary_provenance["candidate_id"],
        "dictionary_geometry": geometry,
        "eligible_downstream_layers": dictionary_provenance[
            "eligible_downstream_layers"
        ],
        "p0_2_load_provenance": dictionary_provenance,
        **inference_provenance,
        "activation_finiteness_check": ACTIVATION_FINITE_CHECK,
        "activation_finiteness_verified": True,
    }


def _write_semantics(
    state: dict[str, Any],
    cohort: Mapping[str, Any],
    output: Path,
    provenance: Mapping[str, Any],
    builder_spec_sha256: str,
) -> None:
    directory = output / "semantics" / state["name"]
    directory.mkdir(parents=True)
    sparse = np.concatenate(state["target_sparse"], axis=0)
    dense = np.concatenate(state["target_dense"], axis=0)
    randomized = np.concatenate(state["target_random"], axis=0)
    input_norm = np.concatenate(state["input_norm"], axis=0)
    if sparse.shape != dense.shape or sparse.shape != randomized.shape:
        raise ValueError(f"semantic controls are not dimension matched: {state['name']}")

    protein_id = []
    family_id = []
    position = []
    kmer = []
    protein_length = []
    source = []
    biological: dict[str, list[int]] = {label["name"]: [] for label in state["labels"]}
    for record in cohort["rows"]:
        for residue in range(len(record["sequence"])):
            protein_id.append(record["id"])
            family_id.append(record["family"])
            position.append((residue + 0.5) / len(record["sequence"]))
            kmer.append(_centered_kmer(record["sequence"], residue))
            protein_length.append(len(record["sequence"]))
            source.append(record["source"])
            labels = state["annotations"][(record["sha256"], residue)]
            for label in state["labels"]:
                biological[label["name"]].append(labels[label["annotation_key"]])
    n_rows = len(protein_id)
    if sparse.shape[0] != n_rows or input_norm.shape != (n_rows,):
        raise ValueError(f"semantic residue ordering mismatch: {state['name']}")

    arrays: dict[str, np.ndarray] = {
        "protein_id": np.asarray(protein_id),
        "family_id": np.asarray(family_id),
        "normalized_position": np.asarray(position, dtype=np.float32),
        "centered_3mer": np.asarray(kmer),
        "input_norm": input_norm,
        "protein_length": np.asarray(protein_length, dtype=np.int32),
        "sequence_source": np.asarray(source),
        "sparse_activations": sparse.astype(np.float32),
        "dense_matched_directions": dense.astype(np.float32),
        "randomized_dictionary_activations": randomized.astype(np.float32),
    }
    labels_spec = []
    protein_id_array = arrays["protein_id"]
    for index, label in enumerate(state["labels"]):
        biological_key = f"biological_label_{index}"
        negative_key = f"negative_label_{index}"
        values = np.asarray(biological[label["name"]], dtype=np.int8)
        generator = np.random.default_rng(label["negative_seed"])
        negative = values.copy()
        for identifier in dict.fromkeys(protein_id):
            mask = np.flatnonzero(protein_id_array == identifier)
            negative[mask] = generator.permutation(negative[mask])
        arrays[biological_key] = values
        arrays[negative_key] = negative
        labels_spec.extend(
            [
                {
                    "name": label["name"],
                    "role": "biological",
                    "family": label["family"],
                    "construction": label["construction"],
                    "array": biological_key,
                },
                {
                    "name": label["negative_name"],
                    "role": "negative",
                    "family": label["family"],
                    "construction": (
                        "seeded within-protein permutation preserving every protein's prevalence"
                    ),
                    "seed": label["negative_seed"],
                    "array": negative_key,
                    "matched_to": label["name"],
                },
            ]
        )
    if any(not np.isfinite(value).all() for value in arrays.values() if np.issubdtype(value.dtype, np.number)):
        raise ValueError(f"non-finite semantic input: {state['name']}")

    bundle = _save_npz(directory / "continuous_activations.npz", arrays)
    center = _save_npy(directory / "dense_center.npy", state["dense_center"])
    dense_directions = _save_npy(
        directory / "dense_directions.npy", state["dense_directions"]
    )
    random_directions = _save_npy(
        directory / "random_dictionary_directions.npy", state["random_directions"]
    )
    feature_names = _semantic_feature_names(
        state["model"], state["run_seed"], state["layer"], state["features"]
    )
    dictionary_provenance = provenance["models"][state["model"]][
        str(state["run_seed"])
    ]
    power_reference = None
    if state["power_plan"] is not None:
        power_path = directory / "prospective_power_plan.json"
        shutil.copyfile(state["power_plan"]["path"], power_path)
        if sha256_file(power_path) != state["power_plan"]["sha256"]:
            raise RuntimeError("copied prospective power-plan hash mismatch")
        power_reference = {
            "path": power_path.name,
            "sha256": state["power_plan"]["sha256"],
        }
    runner_spec = {
        "schema_version": "r2_p0_4_conditional_semantics_spec_v3",
        "confirmatory": state["confirmatory"],
        "input_provenance": {
            "builder_spec_sha256": builder_spec_sha256,
            "model_seed": provenance["model_seed"],
            "model": state["model"],
            "run_seed": state["run_seed"],
            "eligibility_receipt_sha256": dictionary_provenance[
                "eligibility_receipt_sha256"
            ],
            "run_manifest_sha256": dictionary_provenance["run_manifest_sha256"],
            "checkpoint_sha256": dictionary_provenance["checkpoint_sha256"],
            "model_inference_dtype": dictionary_provenance[
                "model_inference_dtype"
            ],
            "observed_model_parameter_dtypes": dictionary_provenance[
                "observed_model_parameter_dtypes"
            ],
            "model_inference_dtype_verification": dictionary_provenance[
                "model_inference_dtype_verification"
            ],
            "model_inference_dtype_verified": dictionary_provenance[
                "model_inference_dtype_verified"
            ],
            "activation_finiteness_check": dictionary_provenance[
                "activation_finiteness_check"
            ],
            "activation_finiteness_verified": dictionary_provenance[
                "activation_finiteness_verified"
            ],
        },
        "data": {"path": bundle["path"], "sha256": bundle["sha256"]},
        "arrays": {
            "protein_id": "protein_id",
            "family_id": "family_id",
            "position": "normalized_position",
            "kmer": "centered_3mer",
            "input_norm": "input_norm",
            "protein_length": "protein_length",
            "sequence_source": "sequence_source",
        },
        "representations": [
            {
                "name": "sparse_aligned",
                "role": "sparse",
                "construction": "continuous activations of spec-frozen CLT feature identities",
                "array": "sparse_activations",
                "feature_names": feature_names["sparse_aligned"],
            },
            {
                "name": "dense_matched",
                "role": "dense",
                "construction": "randomized-PCA directions fit only on hash-selected discovery residues",
                "array": "dense_matched_directions",
                "feature_names": feature_names["dense_matched"],
            },
            {
                "name": "randomized_dictionary",
                "role": "randomized",
                "construction": "independently seeded unit-norm Gaussian dictionary with ReLU encoding",
                "seed": state["random_seed"],
                "array": "randomized_dictionary_activations",
                "feature_names": feature_names["randomized_dictionary"],
            },
        ],
        "labels": labels_spec,
        "covariates": state["covariates"],
        "test": state["test"],
    }
    if power_reference is not None:
        runner_spec["prospective_power_plan"] = power_reference
    write_json(directory / "conditional_semantics_spec.json", runner_spec)
    manifest = {
        "schema_version": "r2_p0_4_continuous_input_v3",
        "status": (
            "confirmatory_inputs_built_p0_2_eligible_not_adjudicated"
            if state["confirmatory"]
            else "nonconfirmatory_fixture_inputs_only"
        ),
        "confirmatory": state["confirmatory"],
        "analysis": state["name"],
        "builder_spec_sha256": builder_spec_sha256,
        "model": state["model"],
        "model_seed": provenance["model_seed"],
        "run_seed": state["run_seed"],
        "layer": state["layer"],
        "frozen_features": state["features"],
        "cohorts": {
            "fit": {"id": state["fit_role"], "sha256": provenance["cohorts"]["discovery"]["sha256"]},
            "evaluation": {"id": state["evaluation_role"], "sha256": provenance["cohorts"]["heldout"]["sha256"]},
        },
        "model_checkpoint": dictionary_provenance,
        "annotation": state["annotation_provenance"],
        "prospective_power_plan": (
            None
            if power_reference is None
            else {
                **power_reference,
                "independent_source": state["power_plan"]["independent_source"],
                "n_standard_errors": state["power_plan"]["n_standard_errors"],
            }
        ),
        "dense_fit": {
            "selection": "lowest SHA-256(sequence_sha256:zero_based_residue_position)",
            "residue_budget": len(state["fit_selected"]),
            "selection_sha256": _canonical_digest(state["fit_selected"]),
            "seed": state["dense_seed"],
            "oversample": state["dense_oversample"],
            "singular_values": state["dense_singular_values"].tolist(),
            "center": center,
            "directions": dense_directions,
        },
        "random_dictionary": {"seed": state["random_seed"], "directions": random_directions},
        "bundle": bundle,
        "runner_spec": {
            "path": "conditional_semantics_spec.json",
            "sha256": sha256_file(directory / "conditional_semantics_spec.json"),
        },
        "n_residues": n_rows,
        "n_features_per_representation": sparse.shape[1],
        "claim_boundary": (
            "This artifact contains inputs only. It is not evidence of residual biological "
            "association, a biological primitive, or a causal mechanism."
        ),
    }
    write_json(directory / "input_manifest.json", manifest)


def build_inputs(
    spec_path: Path,
    output_dir: Path,
    *,
    model_loader: Callable[[Mapping[str, Any], str], Any] | None = None,
) -> Path:
    """Execute one immutable input build and atomically publish its directory."""

    spec_path = Path(spec_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    spec = load_json(spec_path)
    spec = _require_keys(
        spec,
        {
            "schema_version",
            "confirmatory",
            "device",
            "max_model_tokens",
            "model_seed",
            "cohorts",
            "models",
            "semantics",
        },
        {"p0_2_eligibility_receipt"},
        "builder spec",
    )
    if spec.get("schema_version") != 3:
        raise ValueError("builder spec requires schema_version 3")
    if type(spec["confirmatory"]) is not bool:
        raise ValueError("builder confirmatory must be a JSON boolean")
    confirmatory = spec["confirmatory"]
    model_seed = spec["model_seed"]
    if type(model_seed) is not int or model_seed < 0:
        raise ValueError("builder model_seed must be a non-negative integer")
    device = spec.get("device")
    max_model_tokens = spec.get("max_model_tokens")
    if not isinstance(device, str) or not device:
        raise ValueError("builder spec requires a device string")
    if type(max_model_tokens) is not int or max_model_tokens < 2:
        raise ValueError("max_model_tokens must be an integer >= 2")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    base = spec_path.parent
    receipt_descriptor = spec.get("p0_2_eligibility_receipt")
    eligibility_receipt = None
    if confirmatory:
        receipt_descriptor = _require_keys(
            receipt_descriptor,
            {"path", "sha256"},
            set(),
            "p0_2_eligibility_receipt",
        )
        receipt_path = _resolve(receipt_descriptor["path"], base)
        receipt_sha256 = _require_sha256(
            receipt_descriptor["sha256"], "P0-2 eligibility receipt SHA-256"
        )
        if not receipt_path.is_file() or sha256_file(receipt_path) != receipt_sha256:
            raise ValueError("P0-2 eligibility receipt is missing or its SHA-256 changed")
        eligibility_receipt = {"path": receipt_path, "sha256": receipt_sha256}
        if model_loader is not None:
            raise ValueError("custom model loaders are forbidden for confirmatory input builds")
    elif receipt_descriptor is not None:
        raise ValueError("nonconfirmatory fixtures must not declare a P0-2 eligibility receipt")
    cohort_list = spec.get("cohorts")
    if not isinstance(cohort_list, list) or len(cohort_list) != 2:
        raise ValueError("cohorts must contain discovery and heldout descriptors")
    cohorts = {}
    for descriptor in cohort_list:
        cohort = load_cohort(descriptor, base)
        if cohort["role"] in cohorts:
            raise ValueError(f"duplicate cohort role: {cohort['role']}")
        cohorts[cohort["role"]] = cohort
    validate_disjoint(cohorts)
    models = _validate_models(
        spec,
        base,
        confirmatory=confirmatory,
        eligibility_receipt=eligibility_receipt,
    )
    models_by_name = {model["name"]: model for model in models}
    semantics = _validate_semantics(spec, models_by_name, cohorts, base)
    spec_hash = sha256_file(spec_path)
    staging = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"stale builder staging directory: {staging}")
    staging.mkdir(parents=True)
    loader = model_loader or _default_model_loader
    resources_before = _resource_snapshot(device)
    atlas_by_model = {}
    model_provenance = {}
    try:
        for model in models:
            states = [state for state in semantics if state["model"] == model["name"]]
            atlas_by_model[model["name"]] = {}
            model_provenance[model["name"]] = {}
            for dictionary in model["dictionaries"]:
                atlas, dictionary_provenance = _extract_model(
                    model,
                    dictionary,
                    cohorts,
                    states,
                    device,
                    max_model_tokens,
                    loader,
                )
                seed_key = str(dictionary["run_seed"])
                atlas_by_model[model["name"]][seed_key] = atlas
                model_provenance[model["name"]][seed_key] = dictionary_provenance
        provenance = {
            "cohorts": {
                role: {
                    "cohort_id": cohort["cohort_id"],
                    "path": str(cohort["path"]),
                    "sha256": cohort["sha256"],
                    "n_sequences": len(cohort["rows"]),
                }
                for role, cohort in cohorts.items()
            },
            "models": model_provenance,
            "model_seed": model_seed,
            "p0_2_eligibility_receipt": (
                None
                if eligibility_receipt is None
                else {
                    "path": str(eligibility_receipt["path"]),
                    "sha256": eligibility_receipt["sha256"],
                }
            ),
        }
        atlas_dir = staging / "atlas"
        atlas_dir.mkdir()
        for role, cohort in cohorts.items():
            records = []
            role_dir = atlas_dir / role
            role_dir.mkdir()
            for model in models:
                for dictionary in model["dictionaries"]:
                    seed_key = str(dictionary["run_seed"])
                    dictionary_provenance = model_provenance[model["name"]][seed_key]
                    for layer in model["layers"]:
                        matrix = atlas_by_model[model["name"]][seed_key][role][layer]
                        if matrix.shape != (
                            len(cohort["rows"]),
                            dictionary_provenance["dictionary_geometry"]["d_clt"],
                        ):
                            raise ValueError("atlas matrix geometry mismatch")
                        if not np.isfinite(matrix).all():
                            raise ValueError("atlas matrix contains non-finite values")
                        filename = (
                            f"{model['name']}_seed_{dictionary['run_seed']}_layer_{layer}.npy"
                        )
                        descriptor = _save_npy(role_dir / filename, matrix)
                        records.append(
                            {
                                "model": model["name"],
                                "dictionary_seed": dictionary["run_seed"],
                                "layer": str(layer),
                                "path": f"{role}/{filename}",
                                "sha256": descriptor["sha256"],
                                "n_rows": matrix.shape[0],
                                "n_features": matrix.shape[1],
                                "dictionary_provenance": dictionary_provenance,
                            }
                        )
            write_json(
                atlas_dir / f"{role}_manifest.json",
                {
                    "schema_version": "r2_p0_3_atlas_input_v3",
                    "confirmatory": confirmatory,
                    "model_seed": model_seed,
                    "cohort_id": cohort["cohort_id"],
                    "sequence_hashes": cohort["sequence_hashes"],
                    "matrices": records,
                    "aggregation": (
                        "arithmetic mean of continuous ReLU-TopK activations over residue "
                        "positions after strict token-to-residue alignment; special tokens excluded"
                    ),
                    "builder_spec_sha256": spec_hash,
                    "source_cohort": provenance["cohorts"][role],
                    "models": model_provenance,
                    "p0_2_eligibility_receipt": provenance[
                        "p0_2_eligibility_receipt"
                    ],
                    "status": (
                        "confirmatory_inputs_built_p0_2_eligible_not_adjudicated"
                        if confirmatory
                        else "nonconfirmatory_fixture_inputs_only"
                    ),
                },
            )
        for state in semantics:
            _write_semantics(
                state,
                cohorts["heldout"],
                staging,
                provenance,
                spec_hash,
            )
        if sha256_file(spec_path) != spec_hash:
            raise RuntimeError("builder spec changed during extraction")
        for role, cohort in cohorts.items():
            if sha256_file(cohort["path"]) != cohort["sha256"]:
                raise RuntimeError(f"{role} cohort changed during extraction")
        for state in semantics:
            annotation = state["annotation_provenance"]
            if sha256_file(Path(annotation["path"])) != annotation["sha256"]:
                raise RuntimeError(f"annotation changed during extraction: {state['name']}")
            power_plan = state["power_plan"]
            if power_plan is not None and sha256_file(power_plan["path"]) != power_plan["sha256"]:
                raise RuntimeError(
                    f"prospective power plan changed during extraction: {state['name']}"
                )
        if eligibility_receipt is not None and sha256_file(
            eligibility_receipt["path"]
        ) != eligibility_receipt["sha256"]:
            raise RuntimeError("P0-2 eligibility receipt changed during extraction")
        resources_after = _resource_snapshot(device)
        outputs = sorted(path for path in staging.rglob("*") if path.is_file())
        write_json(
            staging / "run_manifest.json",
            {
                "schema_version": "r2_p0_3_p0_4_input_build_v3",
                "confirmatory": confirmatory,
                "model_seed": model_seed,
                "status": (
                    "confirmatory_inputs_built_p0_2_eligible_not_adjudicated"
                    if confirmatory
                    else "nonconfirmatory_fixture_inputs_only"
                ),
                "builder_spec": str(spec_path),
                "builder_spec_sha256": spec_hash,
                "builder_source_sha256": sha256_file(Path(__file__)),
                "resources_before": resources_before,
                "resources_after": resources_after,
                "inputs": provenance,
                "outputs": [
                    {
                        "path": path.relative_to(staging).as_posix(),
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in outputs
                ],
                "claim_boundary": (
                    "This run only constructs hash-bound inputs. P0-3 and P0-4 remain failed "
                    "until real confirmatory analyses and prespecified gates are adjudicated."
                ),
            },
        )
        staging.rename(output_dir)
        return output_dir / "run_manifest.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
