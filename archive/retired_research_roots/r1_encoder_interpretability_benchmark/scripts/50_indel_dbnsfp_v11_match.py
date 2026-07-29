#!/usr/bin/env python3
"""Match IndelMissense v1.1 GRCh38 coordinates to dbNSFP CADD/REVEL rows.

This script expects a tabix-indexed dbNSFP GRCh38 BGZF and uses pysam for random
access. It is intentionally strict: matches require exact GRCh38 VCF
chromosome, position, reference allele and alternate allele.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
R1 = REPO / "r1_encoder_interpretability_benchmark" / "results" / "variant_effect"
DEFAULT_RECORDS = REPO / "data" / "indelmissense" / "v1.1_coordinates" / "records.tsv"
DEFAULT_DBNSFP = REPO / "external_resources" / "baselines" / "dbnsfp" / "dbNSFP5.3.1a_grch38.gz"


def require_pysam():
    try:
        import pysam  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise SystemExit(
            "pysam is required for dbNSFP tabix access. Install pysam or run in "
            "an environment with tabix bindings available."
        ) from exc
    return pysam


def read_records(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def normalize_chrom(chrom: str) -> list[str]:
    chrom = (chrom or "").strip()
    if not chrom:
        return []
    if chrom.startswith("chr"):
        return [chrom, chrom[3:]]
    return [chrom, f"chr{chrom}"]


def header_columns(tbx: Any) -> list[str]:
    header_lines = list(tbx.header)
    if not header_lines:
        return []
    line = header_lines[-1].lstrip("#").rstrip("\n")
    return line.split("\t")


def col_index(cols: list[str], *names: str) -> int | None:
    lower = {name.lower(): i for i, name in enumerate(cols)}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def value(parts: list[str], idx: int | None) -> str:
    if idx is None or idx >= len(parts):
        return ""
    val = parts[idx]
    return "" if val in {".", "-", "NA", "na"} else val


def query_record(tbx: Any, rec: dict[str, str], idx: dict[str, int | None]) -> tuple[dict[str, str] | None, int]:
    chrom = rec.get("grch38_chromosome", "")
    pos_s = rec.get("grch38_position_vcf", "")
    ref = rec.get("grch38_reference_allele_vcf", "")
    alt = rec.get("grch38_alternate_allele_vcf", "")
    if not chrom or not pos_s or not ref or not alt:
        return None, 0
    try:
        pos = int(pos_s)
    except ValueError:
        return None, 0

    position_hits = 0
    for query_chrom in normalize_chrom(chrom):
        try:
            iterator = tbx.fetch(query_chrom, pos - 1, pos)
        except ValueError:
            continue
        for line in iterator:
            parts = line.rstrip("\n").split("\t")
            row_pos = value(parts, idx["pos"])
            row_ref = value(parts, idx["ref"])
            row_alt = value(parts, idx["alt"])
            if row_pos == pos_s:
                position_hits += 1
            if row_pos == pos_s and row_ref == ref and row_alt == alt:
                return {
                    "dbnsfp_chrom": query_chrom,
                    "dbnsfp_pos": row_pos,
                    "dbnsfp_ref": row_ref,
                    "dbnsfp_alt": row_alt,
                    "CADD_raw": value(parts, idx["cadd_raw"]),
                    "CADD_phred": value(parts, idx["cadd_phred"]),
                    "REVEL_score": value(parts, idx["revel_score"]),
                    "REVEL_rankscore": value(parts, idx["revel_rankscore"]),
                }, position_hits
    return None, position_hits


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# IndelMissense v1.1 dbNSFP Match",
        "",
        f"- Generated UTC: {payload['generated_utc']}",
        f"- Records: {payload['n_records']}",
        f"- GRCh38 VCF-coordinate records: {payload['n_grch38_vcf_records']}",
        f"- Position hits: {payload['n_position_hits']}",
        f"- Exact chrom/pos/ref/alt matches: {payload['n_exact_matches']}",
        f"- CADD_phred coverage: {payload['coverage'].get('CADD_phred', 0)}",
        f"- REVEL_score coverage: {payload['coverage'].get('REVEL_score', 0)}",
        f"- Gate: {payload['gate']}",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    ap.add_argument("--dbnsfp", type=Path, default=DEFAULT_DBNSFP)
    ap.add_argument("--out-dir", type=Path, default=R1 / "indel_dbnsfp_v11_20260515")
    ap.add_argument("--max-records", type=int, default=0)
    args = ap.parse_args()

    pysam = require_pysam()
    records = read_records(args.records)
    if args.max_records:
        records = records[: args.max_records]

    tbx = pysam.TabixFile(str(args.dbnsfp))
    cols = header_columns(tbx)
    idx = {
        "pos": col_index(cols, "pos(1-based)", "pos"),
        "ref": col_index(cols, "ref", "reference"),
        "alt": col_index(cols, "alt", "alternative"),
        "cadd_raw": col_index(cols, "CADD_raw"),
        "cadd_phred": col_index(cols, "CADD_phred"),
        "revel_score": col_index(cols, "REVEL_score"),
        "revel_rankscore": col_index(cols, "REVEL_rankscore"),
    }
    if idx["pos"] is None or idx["ref"] is None or idx["alt"] is None:
        raise SystemExit(f"Could not identify dbNSFP coordinate columns from header: {cols[:20]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = args.out_dir / "matches.tsv"
    fields = [
        "indel_id",
        "gene",
        "protein_hgvs",
        "label",
        "grch38_chromosome",
        "grch38_position_vcf",
        "grch38_reference_allele_vcf",
        "grch38_alternate_allele_vcf",
        "position_hits",
        "exact_match",
        "CADD_raw",
        "CADD_phred",
        "REVEL_score",
        "REVEL_rankscore",
    ]

    counts = Counter()
    coverage = Counter()
    with out_tsv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for rec in records:
            has_grch38 = bool(
                rec.get("grch38_chromosome")
                and rec.get("grch38_position_vcf")
                and rec.get("grch38_reference_allele_vcf")
                and rec.get("grch38_alternate_allele_vcf")
            )
            counts["records"] += 1
            if has_grch38:
                counts["grch38_vcf_records"] += 1
            match, position_hits = query_record(tbx, rec, idx) if has_grch38 else (None, 0)
            if position_hits:
                counts["position_hits"] += 1
            if match:
                counts["exact_matches"] += 1
            row = {
                "indel_id": rec.get("indel_id", ""),
                "gene": rec.get("gene", ""),
                "protein_hgvs": rec.get("protein_hgvs", ""),
                "label": rec.get("label", ""),
                "grch38_chromosome": rec.get("grch38_chromosome", ""),
                "grch38_position_vcf": rec.get("grch38_position_vcf", ""),
                "grch38_reference_allele_vcf": rec.get("grch38_reference_allele_vcf", ""),
                "grch38_alternate_allele_vcf": rec.get("grch38_alternate_allele_vcf", ""),
                "position_hits": position_hits,
                "exact_match": bool(match),
            }
            if match:
                row.update(match)
                for key in ["CADD_raw", "CADD_phred", "REVEL_score", "REVEL_rankscore"]:
                    if match.get(key):
                        coverage[key] += 1
            writer.writerow(row)

    exact = counts["exact_matches"]
    gate = "ready_for_auc" if coverage["CADD_phred"] or coverage["REVEL_score"] else "coverage_too_low_or_no_matches"
    interpretation = (
        "Exact dbNSFP matches with CADD/REVEL scores are available; compute AUCs next."
        if gate == "ready_for_auc"
        else "No usable exact CADD/REVEL coverage was detected under strict GRCh38 chrom/pos/ref/alt matching."
    )
    payload = {
        "task": "IndelMissense v1.1 dbNSFP CADD/REVEL match",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "records": str(args.records),
        "dbnsfp": str(args.dbnsfp),
        "n_records": counts["records"],
        "n_grch38_vcf_records": counts["grch38_vcf_records"],
        "n_position_hits": counts["position_hits"],
        "n_exact_matches": exact,
        "coverage": dict(coverage),
        "gate": gate,
        "interpretation": interpretation,
        "outputs": {"matches": str(out_tsv)},
    }
    write_json = args.out_dir / "summary.json"
    write_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_md(args.out_dir / "summary.md", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
