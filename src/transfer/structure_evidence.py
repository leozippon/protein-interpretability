"""Resumable ESMFold measurements for the Direction-1 generation cohort.

Every source row survives evaluation. Exact duplicate sequences share a fold,
not a sampling unit. Predictor confidence is indirect structural evidence and
is never labelled an experimental fold, activity, or biological success.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .io import _atomic_write, sha256_file, write_json

SCHEMA_VERSION = "generation_structure_evidence_v1"
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
ESMFOLD_REVISION = "75a3841ee059df2bf4d56688166c8fb459ddd97a"
ESMFOLD_FILE_DIGESTS = {
    "pytorch_model.bin": "2ee07356b125d1e3e57503c204111fd7323347fc4735d41d3caac57c2a78e116",
    "config.json": "6b98125e2685fef2875499f6bd7c83968a077993ab53f99bf5581113665f7cc6",
}
INTERPRETATION = (
    "ESMFold confidence is indirect predicted structural feasibility, not "
    "experimentally verified folding or function. Low confidence may reflect "
    "disorder, predictor limitations, or sequence inadequacy. This evaluator "
    "does not identify learned rules or mechanisms."
)


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sequence_digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def load_cohort(path: Path) -> list[dict[str, Any]]:
    rows = []
    ids = set()
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError(f"line {line_number}: nonempty string id required")
        if row["id"] in ids:
            raise ValueError(f"duplicate source id: {row['id']}")
        if not isinstance(row.get("sequence"), str):
            raise ValueError(f"line {line_number}: sequence must be a string, including empty failures")
        if row.get("sequence_sha256", sequence_digest(row["sequence"])) != sequence_digest(row["sequence"]):
            raise ValueError(f"sequence digest mismatch for {row['id']}")
        ids.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValueError("cohort has no records")
    return rows


def eligibility(sequence: str, *, min_length: int, max_length: int) -> str | None:
    if not sequence:
        return "empty_sequence"
    if set(sequence) - set(AA_ORDER):
        return "noncanonical_residues"
    if len(sequence) < min_length:
        return "below_minimum_length"
    if len(sequence) > max_length:
        return "above_maximum_length"
    return None


def shard_for(sequence: str, num_shards: int) -> int:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    return int(sequence_digest(sequence), 16) % num_shards


def summarize_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """CA endpoint avoids weighting large side chains more heavily.

    Transformers 4.57.3 categorical_lddt explicitly returns values in [0, 1].
    The runner converts that fixed API scale to [0, 100] before this function;
    inferring the scale from a sample's maximum would be scientifically unsafe.
    """
    atom_plddt = np.asarray(arrays["atom_plddt_0_100"], dtype=np.float64)
    mask = np.asarray(arrays["atom37_atom_exists"])
    xyz = np.asarray(arrays["atom37_positions_angstrom"])
    pae = np.asarray(arrays["predicted_aligned_error_angstrom"])
    ptm = float(np.asarray(arrays["ptm"]).reshape(-1)[0])
    length = len(atom_plddt)
    if atom_plddt.shape != (length, 37) or mask.shape != atom_plddt.shape:
        raise ValueError("unexpected atom confidence/mask shape")
    if xyz.shape != (length, 37, 3) or pae.shape != (length, length):
        raise ValueError("unexpected coordinate/PAE shape")
    if length == 0 or not (mask[:, 1] == 1).all():
        raise ValueError("missing CA atoms")
    for name, values in arrays.items():
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite prediction: {name}")
    if not ((atom_plddt >= 0) & (atom_plddt <= 100)).all() or not 0 <= ptm <= 1:
        raise ValueError("confidence outside declared scale")
    ca_plddt = atom_plddt[:, 1]
    ca = xyz[:, 1]
    mean_ca = float(ca_plddt.mean())
    fraction = float(np.mean(ca_plddt >= 70.0))
    distances = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)
    return {
        "mean_ca_plddt": mean_ca,
        "fraction_ca_plddt_ge70": fraction,
        "ca_plddt": ca_plddt.tolist(),
        "predicted_confidence_event": mean_ca >= 70.0 and fraction >= 0.8,
        "mean_atom_plddt": float((atom_plddt * mask).sum() / mask.sum()),
        "ptm": ptm,
        "mean_pae_angstrom": float(pae.mean()),
        "radius_of_gyration_ca_angstrom": float(np.sqrt(np.mean(np.sum((ca - ca.mean(0)) ** 2, axis=-1)))),
        "mean_adjacent_ca_distance_angstrom": float(distances.mean()) if len(distances) else None,
    }


def prediction_arrays(output: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Retain scientific raw predictions, excluding latent embeddings/logits."""
    from transformers.models.esm.openfold_utils import atom14_to_atom37

    def numpy(value: Any) -> np.ndarray:
        return value.detach().float().cpu().numpy()

    raw_plddt = numpy(output["plddt"][0])
    if not ((raw_plddt >= 0) & (raw_plddt <= 1)).all():
        raise ValueError("Transformers ESMFold pLDDT API changed from [0, 1]")
    return {
        "atom_plddt_raw_0_1": raw_plddt,
        "atom_plddt_0_100": raw_plddt * 100.0,
        "ca_plddt_0_100": raw_plddt[:, 1] * 100.0,
        "atom37_atom_exists": numpy(output["atom37_atom_exists"][0]),
        "atom37_positions_angstrom": numpy(atom14_to_atom37(output["positions"][-1], output)[0]),
        "predicted_aligned_error_angstrom": numpy(output["predicted_aligned_error"][0]),
        "ptm": numpy(output["ptm"]),
        "aatype": output["aatype"][0].detach().cpu().numpy(),
        "residue_index": output["residue_index"][0].detach().cpu().numpy(),
    }


def pdb_from_arrays(arrays: Mapping[str, np.ndarray]) -> str:
    from transformers.models.esm.openfold_utils import OFProtein, to_pdb

    return to_pdb(OFProtein(
        aatype=arrays["aatype"],
        atom_positions=arrays["atom37_positions_angstrom"],
        atom_mask=arrays["atom37_atom_exists"],
        residue_index=arrays["residue_index"] + 1,
        b_factors=arrays["atom_plddt_0_100"],
    ))


def save_prediction(directory: Path, arrays: Mapping[str, np.ndarray], pdb: str) -> dict[str, str]:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    _atomic_write(directory / "prediction.npz", stream.getvalue())
    _atomic_write(directory / "prediction.pdb", pdb.encode())
    return {name: sha256_file(directory / name) for name in ("prediction.npz", "prediction.pdb")}


def load_result(directory: Path, signature: str, sequence_sha256: str) -> dict[str, Any] | None:
    path = directory / "result.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result["evaluation_signature"] != signature or result["sequence_sha256"] != sequence_sha256:
        raise ValueError(f"incompatible resumable result: {path}")
    if result["status"] == "ok":
        for name, digest in result["files_sha256"].items():
            if not (directory / name).is_file() or sha256_file(directory / name) != digest:
                raise ValueError(f"incomplete/corrupted prediction: {directory / name}")
    return result


def write_index(
    rows: Sequence[Mapping[str, Any]], out: Path, signature: str, *,
    shard_index: int, num_shards: int, filename: str | None = None,
) -> dict[str, Any]:
    """All rows, including duplicates and errors, remain in the denominator."""
    indexed = []
    for row in rows:
        if shard_for(row["sequence"], num_shards) != shard_index:
            continue
        sha = sequence_digest(row["sequence"])
        directory = out / "objects" / sha
        result = load_result(directory, signature, sha)
        item = dict(row)
        item["structure"] = result or {
            "status": "pending", "sequence_sha256": sha,
            "evaluation_signature": signature,
        }
        item["structure"]["object_directory"] = f"objects/{sha}"
        indexed.append(item)
    name = filename or f"index-{shard_index:03d}-of-{num_shards:03d}.jsonl"
    payload = "".join(json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for item in indexed)
    _atomic_write(out / name, payload.encode())
    summary = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_signature": signature,
        "rows": len(indexed),
        "status_counts": dict(Counter(item["structure"]["status"] for item in indexed)),
        "unique_sequences": len({item["structure"]["sequence_sha256"] for item in indexed}),
        "interpretation": INTERPRETATION,
        "confidence_event": "mean_CA_pLDDT >= 70 and fraction_CA_pLDDT_ge_70 >= 0.8; operational prediction only",
        "index_sha256": sha256_file(out / name),
    }
    write_json(out / name.replace(".jsonl", ".summary.json"), summary)
    return summary
