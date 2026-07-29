#!/usr/bin/env python3
"""Re-audit the canonical Swiss-Prot triplet mutual-information output.

The canonical 2026-05-13 analysis binarized the top 100 firing positions among
122,671 scored positions and compared plug-in mutual information (MI) against a
0.1-nat gate.  This script leaves that output untouched and writes a dated,
prevalence-aware reanalysis that:

* normalizes MI by H(B), the entropy of the top-position indicator;
* evaluates a global uniform-position permutation null;
* evaluates a within-protein null matched on amino acid and position; and
* applies Benjamini-Hochberg correction over all triplet x rich-label tests.

The script is deliberately specific to the canonical saved output.  It fails
fast if the reconstructed cohort, labels, top-event counts, or saved MI values
do not agree with the source files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import shlex
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
R2 = REPO / "r2_interpretability_transfer"
SCRIPT33 = R2 / "scripts/33_swissprot_triplet_annotation.py"
DEFAULT_SOURCE = R2 / "results/circuit_analysis/swissprot_triplet_annotation_20260513"
DEFAULT_OUTPUT = R2 / "results/circuit_analysis/swissprot_triplet_mi_reaudit_20260716"
DEFAULT_SWISSPROT = REPO / "data/processed/swissprot_all_max1022.pkl"
DEFAULT_PFAM = REPO / "data/interpro/pfam_residue.tsv"

AA = set("ACDEFGHIKLMNPQRSTVWY")
LABEL_KEYS = [
    "pfam_family",
    "swiss_category",
    "swiss_feature_type",
    "secondary_structure",
    "functional_label",
    "domain_label",
    "ptm_label",
    "topology_label",
    "region_label",
    "dominant_pfam",
]
EXPECTED_TRIPLETS = 38
EXPECTED_TOP_PER_TRIPLET = 100
EXPECTED_POSITIONS = 122_671
EXPECTED_COHORT_SIZE = 500
EXPECTED_HYPOTHESES = EXPECTED_TRIPLETS * len(LABEL_KEYS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--swissprot-cache", type=Path, default=DEFAULT_SWISSPROT)
    parser.add_argument("--pfam-residue", type=Path, default=DEFAULT_PFAM)
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def load_script33():
    spec = importlib.util.spec_from_file_location("triplet_annotation_33_reaudit", SCRIPT33)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT33}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Prefer stable repository-relative paths in saved metadata."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO.resolve()))
    except ValueError:
        return str(resolved)


def position_stratum(pos0: int, length: int) -> str:
    """Fine edge/position stratum used by the matched null."""
    pos1 = pos0 + 1
    if pos1 <= 2:
        return f"first_{pos1}"
    if pos1 <= 5:
        return "first_3_5"
    distance_from_end = length - pos0
    if distance_from_end <= 2:
        return f"last_{distance_from_end}"
    ventile = min(19, int(20 * (pos0 + 0.5) / max(length, 1)))
    return f"ventile_{ventile:02d}"


def binary_entropy(n: int, k: int) -> float:
    if not 0 < k < n:
        raise ValueError(f"binary entropy requires 0 < k < n, got k={k}, n={n}")
    p = k / n
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def mi_from_top_counts(
    total_counts: np.ndarray,
    top_counts: np.ndarray,
    n_positions: int,
    n_top: int,
) -> np.ndarray | float:
    """Compute I(label; top-event) in nats for one or many top-count rows."""
    top = np.asarray(top_counts, dtype=np.float64)
    one_dimensional = top.ndim == 1
    if one_dimensional:
        top = top[None, :]
    total = np.asarray(total_counts, dtype=np.float64)[None, :]
    if top.shape[1] != total.shape[1]:
        raise ValueError(f"top/total level mismatch: {top.shape} versus {total.shape}")
    if np.any(top < 0) or np.any(top > total):
        raise ValueError("invalid top counts")

    non_top = total - top
    top_term = np.zeros_like(top)
    top_mask = top > 0
    top_denominator = np.broadcast_to(total * n_top, top.shape)
    top_term[top_mask] = (top[top_mask] / n_positions) * np.log(
        (top[top_mask] * n_positions) / top_denominator[top_mask]
    )

    non_top_term = np.zeros_like(non_top)
    non_top_mask = non_top > 0
    non_top_denominator = np.broadcast_to(total * (n_positions - n_top), non_top.shape)
    non_top_term[non_top_mask] = (non_top[non_top_mask] / n_positions) * np.log(
        (non_top[non_top_mask] * n_positions) / non_top_denominator[non_top_mask]
    )
    result = top_term.sum(axis=1) + non_top_term.sum(axis=1)
    return float(result[0]) if one_dimensional else result


def count_matrix(codes: np.ndarray, samples: np.ndarray, n_levels: int) -> np.ndarray:
    n_perm, n_top = samples.shape
    sampled_codes = codes[samples]
    counts = np.zeros((n_perm, n_levels), dtype=np.int16)
    np.add.at(
        counts,
        (np.repeat(np.arange(n_perm), n_top), sampled_codes.reshape(-1)),
        1,
    )
    return counts


def bh_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=np.float64)
    if p.size == 0 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid p-values")
    order = np.argsort(p)
    q = np.ones(len(p), dtype=np.float64)
    running = 1.0
    for rank0 in range(len(p) - 1, -1, -1):
        idx = order[rank0]
        running = min(running, p[idx] * len(p) / (rank0 + 1))
        q[idx] = running
    return q.tolist()


def summarize_null(observed: float, null: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(null.mean()),
        "sd": float(null.std(ddof=1)),
        "q025": float(np.quantile(null, 0.025)),
        "q975": float(np.quantile(null, 0.975)),
        "p_upper": float((1 + np.count_nonzero(null >= observed - 1e-15)) / (len(null) + 1)),
    }


def build_universe(args: argparse.Namespace, script33):
    sequences, ann_by_acc, pfam_by_acc, cohort_meta = script33.choose_cohort(
        args.swissprot_cache,
        args.pfam_residue,
        EXPECTED_COHORT_SIZE,
        100,
        400,
        20260513,
    )
    if len(sequences) != EXPECTED_COHORT_SIZE:
        raise AssertionError(f"expected {EXPECTED_COHORT_SIZE} proteins, got {len(sequences)}")

    labels: dict[str, list[str]] = {key: [] for key in LABEL_KEYS}
    position_to_index: dict[tuple[str, int], int] = {}
    strata: list[tuple[int, str, str]] = []
    noncanonical_positions: list[list[str | int]] = []
    sequence_valid_lengths: list[int] = []

    for seq_idx, record in enumerate(sequences):
        accession = record["id"]
        sequence = record["sequence"]
        annotation = ann_by_acc.get(accession)
        if annotation is None:
            raise KeyError(f"missing annotation for {accession}")
        features = getattr(annotation, "features", [])
        intervals = pfam_by_acc.get(accession, [])
        n_valid = 0
        for pos0, aa in enumerate(sequence):
            if aa not in AA:
                noncanonical_positions.append([accession, pos0 + 1, aa])
                continue
            pos1 = pos0 + 1
            swiss = script33.feature_labels(features, pos1)
            label_row = {
                "pfam_family": script33.pfam_at(intervals, pos1),
                "dominant_pfam": record["dominant_pfam"],
                **swiss,
            }
            position_to_index[(accession, pos1)] = len(strata)
            strata.append((seq_idx, aa, position_stratum(pos0, len(sequence))))
            for key in LABEL_KEYS:
                labels[key].append(label_row[key])
            n_valid += 1
        sequence_valid_lengths.append(n_valid)

    if len(strata) != EXPECTED_POSITIONS:
        raise AssertionError(f"expected {EXPECTED_POSITIONS} positions, got {len(strata)}")
    if noncanonical_positions != [["O03848", 231, "X"]]:
        raise AssertionError(f"unexpected noncanonical positions: {noncanonical_positions}")
    return (
        sequences,
        cohort_meta,
        labels,
        position_to_index,
        strata,
        noncanonical_positions,
        sequence_valid_lengths,
    )


def run(args: argparse.Namespace) -> dict:
    if args.n_perm < 1:
        raise ValueError("--n-perm must be positive")
    require_file(SCRIPT33)
    require_file(args.swissprot_cache)
    require_file(args.pfam_residue)
    source_top = args.source_dir / "per_triplet_max_act_rich.tsv"
    source_mi = args.source_dir / "rich_label_mi.tsv"
    source_summary = args.source_dir / "summary.json"
    for path in (source_top, source_mi, source_summary):
        require_file(path)
    if args.source_dir.resolve() == args.out_dir.resolve():
        raise ValueError("refusing to overwrite the canonical source directory")

    start = time.time()
    script33 = load_script33()
    (
        sequences,
        cohort_meta,
        labels,
        position_to_index,
        strata,
        noncanonical_positions,
        sequence_valid_lengths,
    ) = build_universe(args, script33)
    n_positions = len(strata)
    n_top = EXPECTED_TOP_PER_TRIPLET
    top_entropy = binary_entropy(n_positions, n_top)

    canonical_summary = json.loads(source_summary.read_text())
    expected_summary = {
        "cohort_size": EXPECTED_COHORT_SIZE,
        "n_triplets": EXPECTED_TRIPLETS,
        "top_positions": EXPECTED_TOP_PER_TRIPLET,
        "mi_gate": 0.1,
    }
    for key, expected in expected_summary.items():
        if canonical_summary.get(key) != expected:
            raise AssertionError(
                f"canonical summary {key}={canonical_summary.get(key)!r}, expected {expected!r}"
            )

    top_rows = read_tsv(source_top)
    if len(top_rows) != EXPECTED_TRIPLETS * EXPECTED_TOP_PER_TRIPLET:
        raise AssertionError(f"unexpected top-row count: {len(top_rows)}")
    top_by_triplet: dict[str, list[int]] = defaultdict(list)
    label_mismatches: list[dict] = []
    for row in top_rows:
        key = (row["accession"], int(row["position_1based"]))
        if key not in position_to_index:
            raise KeyError(f"saved top position absent from reconstructed cohort: {key}")
        idx = position_to_index[key]
        top_by_triplet[row["triplet_id"]].append(idx)
        for label in LABEL_KEYS:
            if row[label] != labels[label][idx]:
                label_mismatches.append(
                    {
                        "triplet_id": row["triplet_id"],
                        "accession": row["accession"],
                        "position_1based": row["position_1based"],
                        "label": label,
                        "saved": row[label],
                        "reconstructed": labels[label][idx],
                    }
                )
    if label_mismatches:
        raise AssertionError(f"saved-label mismatch: {label_mismatches[:3]}")
    if len(top_by_triplet) != EXPECTED_TRIPLETS:
        raise AssertionError(f"expected {EXPECTED_TRIPLETS} triplets, got {len(top_by_triplet)}")

    saved_mi = {
        (row["triplet_id"], row["label"]): float(row["mi_nats"])
        for row in read_tsv(source_mi)
        if row["label"] in LABEL_KEYS
    }
    if len(saved_mi) != EXPECTED_HYPOTHESES:
        raise AssertionError(f"expected {EXPECTED_HYPOTHESES} saved MI rows, got {len(saved_mi)}")

    encoded: dict[str, np.ndarray] = {}
    totals: dict[str, np.ndarray] = {}
    for label in LABEL_KEYS:
        levels = sorted(set(labels[label]))
        level_to_index = {value: idx for idx, value in enumerate(levels)}
        codes = np.asarray([level_to_index[value] for value in labels[label]], dtype=np.int32)
        encoded[label] = codes
        totals[label] = np.bincount(codes, minlength=len(levels))

    stratum_lists: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    for idx, stratum in enumerate(strata):
        stratum_lists[stratum].append(idx)
    stratum_indices = {
        stratum: np.asarray(indices, dtype=np.int32)
        for stratum, indices in stratum_lists.items()
    }

    rng = np.random.default_rng(args.seed)
    results: list[dict] = []
    max_saved_mi_difference = 0.0
    for triplet_number, triplet_id in enumerate(sorted(top_by_triplet), start=1):
        observed_indices = np.asarray(top_by_triplet[triplet_id], dtype=np.int32)
        if len(observed_indices) != n_top or len(np.unique(observed_indices)) != n_top:
            raise AssertionError(f"{triplet_id}: expected {n_top} unique top positions")

        global_samples = np.stack(
            [rng.choice(n_positions, n_top, replace=False) for _ in range(args.n_perm)]
        ).astype(np.int32)

        observed_strata = Counter(strata[idx] for idx in observed_indices)
        matched_samples = np.empty((args.n_perm, n_top), dtype=np.int32)
        column = 0
        for stratum, count in sorted(observed_strata.items()):
            candidates = stratum_indices[stratum]
            if count > len(candidates):
                raise AssertionError(f"invalid matched stratum {stratum}: {count}>{len(candidates)}")
            random_ranks = rng.random((args.n_perm, len(candidates)))
            selected = np.argpartition(random_ranks, count - 1, axis=1)[:, :count]
            matched_samples[:, column : column + count] = candidates[selected]
            column += count
        if column != n_top:
            raise AssertionError(f"{triplet_id}: matched sample width {column}, expected {n_top}")

        for label in LABEL_KEYS:
            codes = encoded[label]
            total = totals[label]
            observed_counts = np.bincount(codes[observed_indices], minlength=len(total))
            observed_mi = float(mi_from_top_counts(total, observed_counts, n_positions, n_top))
            difference = abs(observed_mi - saved_mi[(triplet_id, label)])
            max_saved_mi_difference = max(max_saved_mi_difference, difference)

            global_counts = count_matrix(codes, global_samples, len(total))
            matched_counts = count_matrix(codes, matched_samples, len(total))
            global_null = np.asarray(
                mi_from_top_counts(total, global_counts, n_positions, n_top), dtype=np.float64
            )
            matched_null = np.asarray(
                mi_from_top_counts(total, matched_counts, n_positions, n_top), dtype=np.float64
            )
            global_summary = summarize_null(observed_mi, global_null)
            matched_summary = summarize_null(observed_mi, matched_null)
            matched_identifiable = label != "dominant_pfam"
            results.append(
                {
                    "triplet_id": triplet_id,
                    "label": label,
                    "n_label_levels": len(total),
                    "n_positions": n_positions,
                    "n_top": n_top,
                    "top_prevalence": n_top / n_positions,
                    "top_entropy_nats": top_entropy,
                    "mi_nats": observed_mi,
                    "normalized_mi": observed_mi / top_entropy,
                    "global_null_mean_nats": global_summary["mean"],
                    "global_null_sd_nats": global_summary["sd"],
                    "global_null_q025_nats": global_summary["q025"],
                    "global_null_q975_nats": global_summary["q975"],
                    "global_excess_normalized_mi": (
                        observed_mi - global_summary["mean"]
                    ) / top_entropy,
                    "global_perm_p": global_summary["p_upper"],
                    "matched_null_mean_nats": matched_summary["mean"],
                    "matched_null_sd_nats": matched_summary["sd"],
                    "matched_null_q025_nats": matched_summary["q025"],
                    "matched_null_q975_nats": matched_summary["q975"],
                    "matched_excess_normalized_mi": (
                        observed_mi - matched_summary["mean"]
                    ) / top_entropy,
                    "matched_perm_p": matched_summary["p_upper"] if matched_identifiable else 1.0,
                    "matched_identifiable": matched_identifiable,
                }
            )
        print(f"[{triplet_number:02d}/{EXPECTED_TRIPLETS}] {triplet_id}", flush=True)

    if max_saved_mi_difference > 1e-5:
        raise AssertionError(
            f"reconstructed MI differs from saved output by {max_saved_mi_difference:.6g}"
        )
    if len(results) != EXPECTED_HYPOTHESES:
        raise AssertionError(f"expected {EXPECTED_HYPOTHESES} result rows, got {len(results)}")

    global_q = bh_adjust([float(row["global_perm_p"]) for row in results])
    matched_q = bh_adjust([float(row["matched_perm_p"]) for row in results])
    for row, global_value, matched_value in zip(results, global_q, matched_q):
        row["global_bh_q"] = global_value
        row["matched_bh_q"] = matched_value

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_fields = [
        "triplet_id",
        "label",
        "n_label_levels",
        "n_positions",
        "n_top",
        "top_prevalence",
        "top_entropy_nats",
        "mi_nats",
        "normalized_mi",
        "global_null_mean_nats",
        "global_null_sd_nats",
        "global_null_q025_nats",
        "global_null_q975_nats",
        "global_excess_normalized_mi",
        "global_perm_p",
        "global_bh_q",
        "matched_null_mean_nats",
        "matched_null_sd_nats",
        "matched_null_q025_nats",
        "matched_null_q975_nats",
        "matched_excess_normalized_mi",
        "matched_perm_p",
        "matched_bh_q",
        "matched_identifiable",
    ]
    result_path = args.out_dir / "mi_reanalysis.tsv"
    write_tsv(result_path, results, result_fields)

    triplet_rows: list[dict] = []
    for triplet_id in sorted(top_by_triplet):
        rows = [row for row in results if row["triplet_id"] == triplet_id]
        best_raw = max(rows, key=lambda row: float(row["normalized_mi"]))
        best_global = max(rows, key=lambda row: float(row["global_excess_normalized_mi"]))
        identifiable = [row for row in rows if row["matched_identifiable"]]
        best_matched = max(
            identifiable,
            key=lambda row: float(row["matched_excess_normalized_mi"]),
        )
        triplet_rows.append(
            {
                "triplet_id": triplet_id,
                "best_raw_nmi_label": best_raw["label"],
                "best_raw_nmi": best_raw["normalized_mi"],
                "best_global_excess_label": best_global["label"],
                "best_global_excess_nmi": best_global["global_excess_normalized_mi"],
                "best_global_q": best_global["global_bh_q"],
                "best_matched_excess_label": best_matched["label"],
                "best_matched_excess_nmi": best_matched["matched_excess_normalized_mi"],
                "best_matched_q": best_matched["matched_bh_q"],
                "n_global_q_lt_0p05": sum(float(row["global_bh_q"]) < 0.05 for row in rows),
                "n_matched_q_lt_0p05": sum(
                    float(row["matched_bh_q"]) < 0.05 for row in identifiable
                ),
            }
        )
    triplet_path = args.out_dir / "triplet_summary.tsv"
    write_tsv(triplet_path, triplet_rows, list(triplet_rows[0]))

    matched_significant = sorted(
        [
            row
            for row in results
            if row["matched_identifiable"] and float(row["matched_bh_q"]) < 0.05
        ],
        key=lambda row: (
            float(row["matched_bh_q"]),
            -float(row["matched_excess_normalized_mi"]),
        ),
    )
    summary = {
        "analysis": "Swiss-Prot triplet MI prevalence-aware permutation re-audit",
        "status": "completed",
        "date": "2026-07-16",
        "exploratory": True,
        "source_output": display_path(args.source_dir),
        "n_triplets": len(top_by_triplet),
        "n_positions": n_positions,
        "n_top_per_triplet": n_top,
        "top_prevalence": n_top / n_positions,
        "top_entropy_nats_maximum_mi": top_entropy,
        "original_gate_nats": 0.1,
        "original_gate_over_theoretical_max": 0.1 / top_entropy,
        "n_permutations": args.n_perm,
        "seed": args.seed,
        "n_hypotheses": len(results),
        "n_global_bh_q_lt_0p05": sum(float(row["global_bh_q"]) < 0.05 for row in results),
        "n_matched_bh_q_lt_0p05": len(matched_significant),
        "max_matched_excess_normalized_mi": max(
            float(row["matched_excess_normalized_mi"])
            for row in results
            if row["matched_identifiable"]
        ),
        "matched_significant": [
            {
                "triplet_id": row["triplet_id"],
                "label": row["label"],
                "normalized_mi": row["normalized_mi"],
                "excess_normalized_mi": row["matched_excess_normalized_mi"],
                "p": row["matched_perm_p"],
                "q": row["matched_bh_q"],
            }
            for row in matched_significant
        ],
        "validation": {
            "cohort_size": len(sequences),
            "noncanonical_positions_excluded": noncanonical_positions,
            "saved_label_mismatches": len(label_mismatches),
            "max_abs_saved_mi_difference": max_saved_mi_difference,
            "n_unique_dominant_pfam": len(set(labels["dominant_pfam"])),
            "dominant_pfam_note": (
                "500 levels for 500 proteins; dominant-Pfam MI measures sequence selectivity, "
                "not replicated family generalization"
            ),
        },
        "nulls": {
            "global_positions": "100 positions uniformly sampled without replacement",
            "matched_positions": (
                "same protein + amino acid + exact first/last edge class or "
                "normalized-position ventile, sampled without replacement"
            ),
        },
        "runtime_seconds": time.time() - start,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    actual_command = " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv])
    manifest = {
        "command": actual_command,
        "canonical_command": (
            "source ~/miniconda3/etc/profile.d/conda.sh && conda activate ct && "
            "python r2_interpretability_transfer/scripts/49_reaudit_triplet_mi.py"
        ),
        "resolved_parameters": {
            "source_dir": display_path(args.source_dir),
            "out_dir": display_path(args.out_dir),
            "swissprot_cache": display_path(args.swissprot_cache),
            "pfam_residue": display_path(args.pfam_residue),
            "n_permutations": args.n_perm,
            "seed": args.seed,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "inputs": {
            display_path(source_top): sha256(source_top),
            display_path(source_mi): sha256(source_mi),
            display_path(source_summary): sha256(source_summary),
            display_path(SCRIPT33): sha256(SCRIPT33),
            display_path(args.pfam_residue): sha256(args.pfam_residue),
            display_path(args.swissprot_cache): sha256(args.swissprot_cache),
        },
        "source_script": {
            "path": display_path(Path(__file__)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "cohort_meta": cohort_meta,
        "sequence_valid_lengths_sha256": hashlib.sha256(
            json.dumps(sequence_valid_lengths, separators=(",", ":")).encode()
        ).hexdigest(),
        "outputs": {
            display_path(result_path): sha256(result_path),
            display_path(triplet_path): sha256(triplet_path),
            display_path(summary_path): sha256(summary_path),
        },
    }
    manifest_path = args.out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
