#!/usr/bin/env python3
"""One-shot staging check for CADD/REVEL/dbNSFP indel competitors.

The current IndelMissense v1 artefact is protein-HGVS based.  dbNSFP's BGZF
variant table is coordinate-indexed, so a real match requires genomic
chrom/pos/ref/alt fields.  This script makes the staging attempt explicit and
records whether the currently staged files are sufficient for a head-to-head.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
R1 = REPO / "r1_encoder_interpretability_benchmark" / "results" / "variant_effect"


def read_header(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                line = line.lstrip("#")
            return line.split("\t")
    return []


def scan_dbnsfp(path: Path, n: int) -> dict:
    if not path.exists():
        return {"exists": False}
    opener = gzip.open if path.suffix == ".gz" else open
    header = []
    rows = 0
    indel_like = 0
    ref_i = alt_i = None
    with opener(path, "rt", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if not header:
                header = line.lstrip("#").split("\t")
                lower = [x.lower() for x in header]
                for cand in ("ref", "reference"):
                    if cand in lower:
                        ref_i = lower.index(cand)
                        break
                for cand in ("alt", "alternative"):
                    if cand in lower:
                        alt_i = lower.index(cand)
                        break
                continue
            parts = line.split("\t")
            rows += 1
            if ref_i is not None and alt_i is not None and len(parts) > max(ref_i, alt_i):
                if len(parts[ref_i]) != 1 or len(parts[alt_i]) != 1:
                    indel_like += 1
            if rows >= n:
                break
    lower = [x.lower() for x in header]
    cadd_cols = [h for h in header if "cadd" in h.lower()]
    revel_cols = [h for h in header if "revel" in h.lower()]
    return {
        "exists": True,
        "path": str(path),
        "header_columns": len(header),
        "coordinate_columns_present": all(x in lower for x in ["#chr", "chr"]) or "pos(1-based)" in lower or "pos" in lower,
        "ref_column": header[ref_i] if ref_i is not None and ref_i < len(header) else None,
        "alt_column": header[alt_i] if alt_i is not None and alt_i < len(header) else None,
        "cadd_columns": cadd_cols[:20],
        "revel_columns": revel_cols[:20],
        "scanned_rows": rows,
        "indel_like_rows_in_scan": indel_like,
    }


def inspect_records(path: Path) -> dict:
    required_genomic = {"chrom", "chr", "pos", "position", "ref", "alt"}
    counts = Counter()
    if path.suffix == ".jsonl":
        n = 0
        fields = []
        lower = set()
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not fields:
                    fields = sorted(row)
                    lower = {x.lower() for x in fields}
                n += 1
                counts[row.get("variant_class", "")] += 1
        has_coord = ({"ref", "alt"} <= lower) and bool(lower & {"chrom", "chr"}) and bool(lower & {"pos", "position"})
        return {
            "path": str(path),
            "n_records": n,
            "fields": fields,
            "variant_class_counts": dict(counts),
            "has_genomic_coordinate_fields": has_coord,
            "missing_for_dbnsfp_tabix": sorted(required_genomic - lower),
        }

    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields = reader.fieldnames or []
        lower = {x.lower() for x in fields}
        has_coord = ({"ref", "alt"} <= lower) and bool(lower & {"chrom", "chr"}) and bool(lower & {"pos", "position"})
        n = 0
        for row in reader:
            n += 1
            counts[row.get("variant_class", "")] += 1
    return {
        "path": str(path),
        "n_records": n,
        "fields": fields,
        "variant_class_counts": dict(counts),
        "has_genomic_coordinate_fields": has_coord,
        "missing_for_dbnsfp_tabix": sorted(required_genomic - lower),
    }


def write_md(path: Path, payload: dict) -> None:
    gate = payload["acceptance_gate"]
    lines = [
        "# R1 Indel Competitor Staging Attempt",
        "",
        f"- Indel records: {payload['records']['n_records']}",
        f"- Record coordinate fields present: {payload['records']['has_genomic_coordinate_fields']}",
        f"- dbNSFP exists: {payload['dbnsfp'].get('exists')}",
        f"- dbNSFP scanned rows: {payload['dbnsfp'].get('scanned_rows')}",
        f"- dbNSFP indel-like rows in scan: {payload['dbnsfp'].get('indel_like_rows_in_scan')}",
        f"- CADD columns detected: {len(payload['dbnsfp'].get('cadd_columns') or [])}",
        f"- REVEL columns detected: {len(payload['dbnsfp'].get('revel_columns') or [])}",
        f"- Gate: {gate}",
        "",
        "## Decision",
        "",
        payload["decision"],
        "",
        "## Required To Make This Comparable",
        "",
        "- Add ClinVar genomic `chrom`, `pos`, `ref`, `alt` for every IndelMissense v1 record, or rebuild the benchmark from a VCF that preserves those fields.",
        "- Then query the coordinate-indexed dbNSFP BGZF with tabix and measure CADD/REVEL coverage.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=REPO / "data" / "indelmissense" / "v1" / "records.tsv")
    ap.add_argument("--dbnsfp", type=Path, default=REPO / "external_resources" / "baselines" / "dbnsfp" / "dbNSFP5.3.1a_grch38.gz")
    ap.add_argument("--scan-lines", type=int, default=10000)
    ap.add_argument("--out-json", type=Path, default=R1 / "indel_competitor_attempt_20260512.json")
    ap.add_argument("--out-md", type=Path, default=R1 / "indel_competitor_attempt_20260512.md")
    args = ap.parse_args()

    records = inspect_records(args.records)
    dbnsfp = scan_dbnsfp(args.dbnsfp, args.scan_lines)
    actionable = records["has_genomic_coordinate_fields"] and dbnsfp.get("exists") and (
        dbnsfp.get("cadd_columns") or dbnsfp.get("revel_columns")
    )
    if actionable:
        gate = "ready_for_coordinate_match"
        decision = "The current records contain genomic coordinates and dbNSFP exposes CADD/REVEL columns; proceed to tabix matching."
    else:
        gate = "drop_head_to_head_for_current_pass"
        decision = (
            "The current IndelMissense v1 records are protein-HGVS records without "
            "chrom/pos/ref/alt, while the staged dbNSFP table is coordinate-indexed. "
            "A defensible CADD/REVEL head-to-head cannot be run for the current pass."
        )
    payload = {
        "task": "R1 B-2 indel competitor staging",
        "status": "completed",
        "records": records,
        "dbnsfp": dbnsfp,
        "acceptance_gate": gate,
        "decision": decision,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    write_md(args.out_md, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
