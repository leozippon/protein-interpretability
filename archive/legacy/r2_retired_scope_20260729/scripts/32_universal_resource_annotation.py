#!/usr/bin/env python3
"""Resource-level annotation for universal triplet top-firing positions.

This low-risk diagnostic asks whether the current top-firing positions can be
connected to staged external resources: Pfam residue intervals, AlphaFold
structures, and Swiss-Prot per-residue features.  It does not infer new
biology; it records coverage and direct overlaps so the next planning pass can
decide whether the R2 universal-primitives story is resource-ready.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def uniref_to_accession(seq_id: str) -> str:
    if seq_id.startswith("real_"):
        rest = seq_id[len("real_") :]
        return rest.split("_", 1)[0]
    if seq_id.startswith("random_"):
        seq_id = seq_id[len("random_") :]
    if seq_id.startswith("UniRef50_"):
        return seq_id[len("UniRef50_") :]
    if seq_id.startswith("UniRef90_"):
        return seq_id[len("UniRef90_") :]
    if seq_id.startswith("UniRef100_"):
        return seq_id[len("UniRef100_") :]
    return seq_id


def load_top_positions(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for row in rows:
        row["accession"] = uniref_to_accession(row["seq_id"])
        row["position_1based"] = int(row["position_0based"]) + 1
    return rows


def load_pfam_intervals(path: Path, accessions: set[str]) -> dict[str, list[tuple[int, int, str]]]:
    out: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    if not path.exists():
        return {}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            acc = row.get("uniprot", "")
            if acc not in accessions:
                continue
            try:
                start = int(row["start"])
                end = int(row["end"])
            except (KeyError, ValueError):
                continue
            out[acc].append((start, end, row.get("pfam_id", "")))
    return dict(out)


def load_alphafold_ids(path: Path) -> set[str]:
    ids = set()
    if not path.exists():
        return ids
    for pdb in path.glob("AF-*-F1-model_v*.pdb.gz"):
        parts = pdb.name.split("-")
        if len(parts) >= 2:
            ids.add(parts[1])
    return ids


def load_swissprot_features(path: Path, accessions: set[str]) -> dict[str, list[tuple[int, int, str, str, str]]]:
    if not path.exists():
        return {}
    sys.path.insert(0, str(REPO / "r1_encoder_interpretability_benchmark"))
    import pickle

    out = {}
    with path.open("rb") as f:
        proteins = pickle.load(f)
    for ann in proteins:
        acc = getattr(ann, "accession", "")
        if acc in accessions:
            out[acc] = list(getattr(ann, "features", []))
    return out


def plddt_at_position(af_dir: Path, accession: str, pos_1based: int) -> float | None:
    matches = sorted(af_dir.glob(f"AF-{accession}-F1-model_v*.pdb.gz"))
    if not matches:
        return None
    path = matches[-1]
    try:
        with gzip.open(path, "rt", errors="replace") as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                if line[12:16].strip() != "CA":
                    continue
                try:
                    resi = int(line[22:26])
                except ValueError:
                    continue
                if resi == pos_1based:
                    try:
                        return float(line[60:66])
                    except ValueError:
                        return None
    except OSError:
        return None
    return None


def interval_hits(intervals: list[tuple[int, int, str]], pos: int) -> list[str]:
    return [name for start, end, name in intervals if start <= pos <= end]


def feature_hits(features: list[tuple[int, int, str, str, str]], pos: int) -> list[dict]:
    hits = []
    for start, end, feat_type, desc, category in features:
        if int(start) <= pos <= int(end):
            hits.append(
                {
                    "feature_type": feat_type,
                    "description": desc,
                    "category": category,
                }
            )
    return hits


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-positions", type=Path, required=True)
    ap.add_argument("--pfam-residue", type=Path, default=REPO / "data" / "interpro" / "pfam_residue.tsv")
    ap.add_argument("--alphafold-dir", type=Path, default=REPO / "data" / "alphafold")
    ap.add_argument("--swissprot-cache", type=Path, default=REPO / "data" / "processed" / "swissprot_all_max1022.pkl")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    rows = load_top_positions(args.top_positions)
    accessions = {r["accession"] for r in rows}
    pfam = load_pfam_intervals(args.pfam_residue, accessions)
    af_ids = load_alphafold_ids(args.alphafold_dir)
    swiss = load_swissprot_features(args.swissprot_cache, accessions)

    annotated = []
    triplet_stats = defaultdict(lambda: Counter())
    category_counts = defaultdict(Counter)
    pfam_counts = defaultdict(Counter)
    plddt_values = defaultdict(list)

    for row in rows:
        tid = row["triplet_id"]
        acc = row["accession"]
        pos = int(row["position_1based"])
        pfam_hits = interval_hits(pfam.get(acc, []), pos)
        sp_hits = feature_hits(swiss.get(acc, []), pos)
        categories = sorted({h["category"] for h in sp_hits})
        af_available = acc in af_ids
        plddt = plddt_at_position(args.alphafold_dir, acc, pos) if af_available else None

        triplet_stats[tid]["events"] += 1
        triplet_stats[tid]["unique_accessions_marker"] += 0
        if pfam_hits:
            triplet_stats[tid]["pfam_event_hits"] += 1
            for x in pfam_hits:
                pfam_counts[tid][x] += 1
        if sp_hits:
            triplet_stats[tid]["swissprot_event_hits"] += 1
            for c in categories:
                category_counts[tid][c] += 1
        if af_available:
            triplet_stats[tid]["alphafold_event_hits"] += 1
        if plddt is not None:
            triplet_stats[tid]["alphafold_plddt_mapped"] += 1
            plddt_values[tid].append(plddt)

        out = {
            **row,
            "pfam_hits": ",".join(pfam_hits),
            "swissprot_categories": ",".join(categories),
            "swissprot_features": "|".join(
                f"{h['category']}:{h['feature_type']}:{h['description']}" for h in sp_hits
            ),
            "alphafold_available": str(bool(af_available)),
            "alphafold_ca_plddt_at_position": "" if plddt is None else f"{plddt:.2f}",
        }
        annotated.append(out)

    # Unique accession coverage by triplet.
    accessions_by_triplet = defaultdict(set)
    for row in annotated:
        accessions_by_triplet[row["triplet_id"]].add(row["accession"])

    summary_rows = []
    for tid in sorted(triplet_stats, key=lambda x: int(x[1:]) if x.startswith("T") else x):
        stats = triplet_stats[tid]
        accs = accessions_by_triplet[tid]
        pfam_accs = {a for a in accs if a in pfam}
        swiss_accs = {a for a in accs if a in swiss}
        af_accs = {a for a in accs if a in af_ids}
        vals = plddt_values[tid]
        summary_rows.append(
            {
                "triplet_id": tid,
                "top_events": stats["events"],
                "unique_accessions": len(accs),
                "pfam_event_hits": stats["pfam_event_hits"],
                "pfam_unique_accessions": len(pfam_accs),
                "top_pfams": ";".join(f"{k}:{v}" for k, v in pfam_counts[tid].most_common(5)),
                "swissprot_event_hits": stats["swissprot_event_hits"],
                "swissprot_unique_accessions": len(swiss_accs),
                "top_swissprot_categories": ";".join(f"{k}:{v}" for k, v in category_counts[tid].most_common(5)),
                "alphafold_event_hits": stats["alphafold_event_hits"],
                "alphafold_unique_accessions": len(af_accs),
                "alphafold_plddt_mapped": stats["alphafold_plddt_mapped"],
                "mean_plddt_if_mapped": "" if not vals else f"{sum(vals)/len(vals):.2f}",
            }
        )

    total_accs = accessions
    coverage = {
        "task": "R2 universal primitive resource annotation",
        "status": "completed",
        "top_position_file": str(args.top_positions),
        "n_top_events": len(rows),
        "n_triplets": len({r["triplet_id"] for r in rows}),
        "n_unique_accessions": len(total_accs),
        "resource_coverage": {
            "pfam_unique_accessions": sum(1 for a in total_accs if a in pfam),
            "swissprot_unique_accessions": sum(1 for a in total_accs if a in swiss),
            "alphafold_unique_accessions": sum(1 for a in total_accs if a in af_ids),
        },
        "resource_paths": {
            "pfam_residue": str(args.pfam_residue),
            "alphafold_dir": str(args.alphafold_dir),
            "swissprot_cache": str(args.swissprot_cache),
        },
        "interpretation": (
            "Coverage is measured over proteins that appear among the top-firing "
            "positions, not over the full UniRef50 cohort."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(
        args.out_dir / "top_position_resource_annotations.tsv",
        annotated,
        [
            "triplet_id",
            "rank",
            "seq_idx",
            "seq_id",
            "accession",
            "source",
            "position_0based",
            "position_1based",
            "aa",
            "residue_class",
            "position_bin",
            "consensus_z",
            "pfam_hits",
            "swissprot_categories",
            "swissprot_features",
            "alphafold_available",
            "alphafold_ca_plddt_at_position",
        ],
    )
    write_tsv(
        args.out_dir / "triplet_resource_summary.tsv",
        summary_rows,
        [
            "triplet_id",
            "top_events",
            "unique_accessions",
            "pfam_event_hits",
            "pfam_unique_accessions",
            "top_pfams",
            "swissprot_event_hits",
            "swissprot_unique_accessions",
            "top_swissprot_categories",
            "alphafold_event_hits",
            "alphafold_unique_accessions",
            "alphafold_plddt_mapped",
            "mean_plddt_if_mapped",
        ],
    )
    (args.out_dir / "resource_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")

    lines = [
        "# Universal Triplet Resource Annotation",
        "",
        f"- Top-firing events: {coverage['n_top_events']}",
        f"- Triplets: {coverage['n_triplets']}",
        f"- Unique accessions: {coverage['n_unique_accessions']}",
        f"- Pfam-covered accessions: {coverage['resource_coverage']['pfam_unique_accessions']}",
        f"- Swiss-Prot-covered accessions: {coverage['resource_coverage']['swissprot_unique_accessions']}",
        f"- AlphaFold-covered accessions: {coverage['resource_coverage']['alphafold_unique_accessions']}",
        "",
        "## Interpretation",
        "",
        coverage["interpretation"],
        "",
        "If coverage is near zero, the next planning pass should not assume that "
        "Pfam/Swiss-Prot/AlphaFold labels are available for the current UniRef50 "
        "top-firing set. A different sequence cohort or additional resource "
        "mapping is required.",
        "",
    ]
    (args.out_dir / "resource_annotation.md").write_text("\n".join(lines))
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
