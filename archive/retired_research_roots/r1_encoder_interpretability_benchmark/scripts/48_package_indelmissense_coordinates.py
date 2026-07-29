#!/usr/bin/env python3
"""Build a coordinate-augmented IndelMissense package.

IndelMissense v1 is a protein-sequence benchmark. This script keeps the v1
record set unchanged and attaches ClinVar genomic coordinates from
variant_summary.txt.gz, usually one GRCh37 and one GRCh38 entry per record.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[2]
DEFAULT_V1 = REPO / "data" / "indelmissense" / "v1"
DEFAULT_CLINVAR = REPO / "data" / "clinvar" / "variant_summary.txt.gz"
DEFAULT_OUT = REPO / "data" / "indelmissense" / "v1.1_coordinates"
PROTEIN_HGVS_RE = re.compile(r"p\.\(?([^\)\s]+)\)?")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def clean(value: str | None) -> str:
    if value is None:
        return ""
    value = value.strip()
    if value in {"-", "-1", "na", "N/A", "not provided"}:
        return ""
    return value


def get_column(cols: dict[str, int], *names: str) -> int | None:
    for name in names:
        if name in cols:
            return cols[name]
    return None


def take(parts: list[str], idx: int | None) -> str:
    if idx is None or idx >= len(parts):
        return ""
    return clean(parts[idx])


def extract_protein_hgvs(name: str) -> str:
    match = PROTEIN_HGVS_RE.search(name)
    return match.group(1).strip() if match else ""


def unique_entries(entries: Iterable[dict]) -> list[dict]:
    seen = set()
    unique = []
    for entry in entries:
        key = (
            entry.get("assembly", ""),
            entry.get("chromosome", ""),
            entry.get("position_vcf", ""),
            entry.get("reference_allele_vcf", ""),
            entry.get("alternate_allele_vcf", ""),
            entry.get("start", ""),
            entry.get("stop", ""),
            entry.get("variation_id", ""),
            entry.get("allele_id", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return sorted(
        unique,
        key=lambda e: (
            e.get("variation_id", ""),
            e.get("assembly", ""),
            e.get("chromosome", ""),
            e.get("position_vcf", ""),
        ),
    )


def build_coordinate_maps(clinvar_path: Path, loose_targets: set[tuple[str, str]]) -> tuple[dict, dict, Counter]:
    by_full: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    by_loose: dict[tuple[str, str], list[dict]] = defaultdict(list)
    counts = Counter()

    opener = gzip.open if clinvar_path.suffix == ".gz" else open
    with opener(clinvar_path, "rt", encoding="utf-8", errors="ignore") as f:
        header = f.readline().rstrip("\n").split("\t")
        cols = {name: i for i, name in enumerate(header)}
        idx = {
            "allele_id": get_column(cols, "AlleleID", "#AlleleID"),
            "type": get_column(cols, "Type"),
            "name": get_column(cols, "Name"),
            "gene_id": get_column(cols, "GeneID"),
            "gene": get_column(cols, "GeneSymbol", "#GeneSymbol"),
            "clinical_significance": get_column(cols, "ClinicalSignificance"),
            "clin_sig_simple": get_column(cols, "ClinSigSimple"),
            "last_evaluated": get_column(cols, "LastEvaluated"),
            "review_status": get_column(cols, "ReviewStatus"),
            "rsid": get_column(cols, "RS# (dbSNP)"),
            "nsv_esv": get_column(cols, "nsv/esv (dbVar)"),
            "rcv_accession": get_column(cols, "RCVaccession"),
            "phenotype_ids": get_column(cols, "PhenotypeIDS"),
            "phenotype_list": get_column(cols, "PhenotypeList"),
            "origin": get_column(cols, "Origin"),
            "origin_simple": get_column(cols, "OriginSimple"),
            "assembly": get_column(cols, "Assembly"),
            "chromosome_accession": get_column(cols, "ChromosomeAccession"),
            "chromosome": get_column(cols, "Chromosome"),
            "start": get_column(cols, "Start"),
            "stop": get_column(cols, "Stop"),
            "reference_allele": get_column(cols, "ReferenceAllele"),
            "alternate_allele": get_column(cols, "AlternateAllele"),
            "variation_id": get_column(cols, "VariationID"),
            "position_vcf": get_column(cols, "PositionVCF"),
            "reference_allele_vcf": get_column(cols, "ReferenceAlleleVCF"),
            "alternate_allele_vcf": get_column(cols, "AlternateAlleleVCF"),
        }
        required = [idx["name"], idx["gene"], idx["clinical_significance"]]
        if any(i is None for i in required):
            raise ValueError("ClinVar variant_summary header is missing required fields")

        for line in f:
            parts = line.rstrip("\n").split("\t")
            name = take(parts, idx["name"])
            protein_hgvs = extract_protein_hgvs(name)
            if not protein_hgvs:
                continue
            gene = take(parts, idx["gene"]).upper()
            loose_key = (gene, protein_hgvs)
            if loose_key not in loose_targets:
                continue

            clinical_significance = take(parts, idx["clinical_significance"])
            entry = {
                "assembly": take(parts, idx["assembly"]),
                "chromosome_accession": take(parts, idx["chromosome_accession"]),
                "chromosome": take(parts, idx["chromosome"]),
                "start": take(parts, idx["start"]),
                "stop": take(parts, idx["stop"]),
                "reference_allele": take(parts, idx["reference_allele"]),
                "alternate_allele": take(parts, idx["alternate_allele"]),
                "position_vcf": take(parts, idx["position_vcf"]),
                "reference_allele_vcf": take(parts, idx["reference_allele_vcf"]),
                "alternate_allele_vcf": take(parts, idx["alternate_allele_vcf"]),
                "variation_id": take(parts, idx["variation_id"]),
                "allele_id": take(parts, idx["allele_id"]),
                "type": take(parts, idx["type"]),
                "name": name,
                "gene_id": take(parts, idx["gene_id"]),
                "gene": gene,
                "clinical_significance": clinical_significance,
                "clin_sig_simple": take(parts, idx["clin_sig_simple"]),
                "last_evaluated": take(parts, idx["last_evaluated"]),
                "review_status": take(parts, idx["review_status"]),
                "rsid": take(parts, idx["rsid"]),
                "nsv_esv": take(parts, idx["nsv_esv"]),
                "rcv_accession": take(parts, idx["rcv_accession"]),
                "phenotype_ids": take(parts, idx["phenotype_ids"]),
                "phenotype_list": take(parts, idx["phenotype_list"]),
                "origin": take(parts, idx["origin"]),
                "origin_simple": take(parts, idx["origin_simple"]),
            }
            full_key = (gene, protein_hgvs, clinical_significance)
            by_full[full_key].append(entry)
            by_loose[loose_key].append(entry)
            counts["matched_clinvar_rows"] += 1
            if entry["assembly"]:
                counts[f"assembly_{entry['assembly']}"] += 1

    return by_full, by_loose, counts


def pick_assembly(entries: list[dict], assembly: str) -> dict:
    candidates = [entry for entry in entries if entry.get("assembly", "").upper() == assembly.upper()]
    if not candidates:
        return {}
    return sorted(
        candidates,
        key=lambda e: (
            not bool(e.get("position_vcf")),
            e.get("chromosome", ""),
            e.get("position_vcf", ""),
            e.get("variation_id", ""),
        ),
    )[0]


def has_genomic_coordinate(entry: dict) -> bool:
    return bool(
        entry.get("assembly")
        and entry.get("chromosome")
        and (entry.get("position_vcf") or (entry.get("start") and entry.get("stop")))
    )


def has_vcf_coordinate(entry: dict) -> bool:
    return bool(
        entry.get("assembly")
        and entry.get("chromosome")
        and entry.get("position_vcf")
        and entry.get("reference_allele_vcf")
        and entry.get("alternate_allele_vcf")
    )


def flat_coordinate_fields(entry: dict, prefix: str) -> dict:
    fields = {}
    for key in [
        "chromosome",
        "chromosome_accession",
        "start",
        "stop",
        "position_vcf",
        "reference_allele_vcf",
        "alternate_allele_vcf",
        "reference_allele",
        "alternate_allele",
    ]:
        fields[f"{prefix}_{key}"] = entry.get(key, "")
    return fields


def write_readme(out_dir: Path, metadata: dict) -> None:
    coverage = metadata["coordinate_coverage"]
    clinvar_coverage = metadata["clinvar_match_coverage"]
    vcf_coverage = metadata["vcf_coordinate_coverage"]
    readme = f"""# IndelMissense v1.1 Coordinate-Augmented Benchmark

This package keeps the IndelMissense v1 protein-sequence records unchanged and
adds ClinVar genomic coordinates from `variant_summary.txt.gz`.

## Contents

- `records.jsonl`: full WT/mutant sequence records, labels and coordinate fields.
- `records.tsv`: compact tabular metadata with GRCh37/GRCh38 coordinate columns.
- `coordinates.tsv`: one row per record-coordinate entry.
- `baseline_scores.tsv`: copied v1 SAE perturbation damage score and mechanism probabilities.
- `splits.csv`: copied v1 deterministic protein-level train/validation/test split.
- `metadata.json`: source files, counts, coordinate coverage and v1 metadata.
- `LICENSE`: copied v1 CC-BY-4.0 license for this packaging layer.

## Counts

- Records: {metadata["n_records"]}
- Records with exact ClinVar variant match: {clinvar_coverage.get("any_match", 0)}
- Records with any genomic coordinate: {coverage.get("any_coordinate", 0)}
- Records with GRCh37 coordinate: {coverage.get("grch37", 0)}
- Records with GRCh38 coordinate: {coverage.get("grch38", 0)}
- Records with any VCF-style coordinate: {vcf_coverage.get("any_vcf_coordinate", 0)}
- Records with GRCh37 VCF-style coordinate: {vcf_coverage.get("grch37", 0)}
- Records with GRCh38 VCF-style coordinate: {vcf_coverage.get("grch38", 0)}
- Coordinate status counts: {metadata["coordinate_status_counts"]}

## Coordinate Semantics

The protein `start` and `end` fields are amino-acid residue positions in the
UniProt sequence. The `grch37_*`, `grch38_*` and `genomic_coordinates` fields
are ClinVar genomic coordinates. VCF-style fields (`position_vcf`,
`reference_allele_vcf`, `alternate_allele_vcf`) are the safest columns for
downstream joins to genome-indexed resources.

## Scope

This is a coordinate augmentation of v1, not a new reconstruction pass. The
record count, labels, sequences, split IDs and baseline scores remain inherited
from IndelMissense v1.
"""
    (out_dir / "README.md").write_text(readme)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-dir", type=Path, default=DEFAULT_V1)
    ap.add_argument("--clinvar", type=Path, default=DEFAULT_CLINVAR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    records = list(iter_jsonl(args.v1_dir / "records.jsonl"))
    loose_targets = {(r.get("gene", "").upper(), r.get("protein_hgvs", "")) for r in records}
    by_full, by_loose, clinvar_counts = build_coordinate_maps(args.clinvar, loose_targets)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    record_jsonl = args.out_dir / "records.jsonl"
    record_tsv = args.out_dir / "records.tsv"
    coord_tsv = args.out_dir / "coordinates.tsv"

    base_fields = [
        "indel_id",
        "gene",
        "uniprot_id",
        "protein_hgvs",
        "variant_class",
        "label",
        "clinical_significance",
        "truncating",
        "wt_len",
        "mut_len",
        "length_delta",
        "split",
    ]
    flat_fields = [
        "clinvar_variation_ids",
        "clinvar_allele_ids",
        "coordinate_status",
        "n_coordinate_entries",
    ]
    for prefix in ["grch37", "grch38"]:
        flat_fields.extend(
            [
                f"{prefix}_chromosome",
                f"{prefix}_chromosome_accession",
                f"{prefix}_start",
                f"{prefix}_stop",
                f"{prefix}_position_vcf",
                f"{prefix}_reference_allele_vcf",
                f"{prefix}_alternate_allele_vcf",
                f"{prefix}_reference_allele",
                f"{prefix}_alternate_allele",
            ]
        )
    coord_fields = [
        "indel_id",
        "gene",
        "protein_hgvs",
        "clinical_significance",
        "assembly",
        "chromosome",
        "chromosome_accession",
        "start",
        "stop",
        "position_vcf",
        "reference_allele_vcf",
        "alternate_allele_vcf",
        "reference_allele",
        "alternate_allele",
        "variation_id",
        "allele_id",
        "type",
        "rsid",
        "rcv_accession",
        "review_status",
        "phenotype_list",
        "origin_simple",
    ]

    status_counts = Counter()
    clinvar_match_counts = Counter()
    coverage = Counter()
    vcf_coverage = Counter()
    n_coordinate_entries = 0
    n_clinvar_match_entries = 0

    with (
        record_jsonl.open("w") as out_jsonl,
        record_tsv.open("w", newline="") as out_tsv,
        coord_tsv.open("w", newline="") as out_coord,
    ):
        record_writer = csv.DictWriter(out_tsv, fieldnames=base_fields + flat_fields, delimiter="\t")
        coord_writer = csv.DictWriter(out_coord, fieldnames=coord_fields, delimiter="\t")
        record_writer.writeheader()
        coord_writer.writeheader()

        for rec in records:
            gene = rec.get("gene", "").upper()
            protein_hgvs = rec.get("protein_hgvs", "")
            clinical_significance = rec.get("clinical_significance", "")
            full_key = (gene, protein_hgvs, clinical_significance)
            loose_key = (gene, protein_hgvs)
            matched_entries = unique_entries(by_full.get(full_key, []))
            if matched_entries:
                match_status = "exact_gene_protein_clinsig"
            else:
                matched_entries = unique_entries(by_loose.get(loose_key, []))
                match_status = "gene_protein_fallback" if matched_entries else "missing"

            entries = [entry for entry in matched_entries if has_genomic_coordinate(entry)]
            if entries:
                status = match_status
            elif matched_entries:
                status = f"{match_status}_no_genomic_locus"
            else:
                status = "missing"

            grch37 = pick_assembly(entries, "GRCh37")
            grch38 = pick_assembly(entries, "GRCh38")
            variation_ids = sorted({e.get("variation_id", "") for e in matched_entries if e.get("variation_id", "")})
            allele_ids = sorted({e.get("allele_id", "") for e in matched_entries if e.get("allele_id", "")})

            public = dict(rec)
            public.update(
                {
                    "coordinate_status": status,
                    "clinvar_variation_ids": variation_ids,
                    "clinvar_allele_ids": allele_ids,
                    "genomic_coordinates": entries,
                }
            )
            public.update(flat_coordinate_fields(grch37, "grch37"))
            public.update(flat_coordinate_fields(grch38, "grch38"))
            out_jsonl.write(json.dumps(public, separators=(",", ":")) + "\n")

            row = {
                "indel_id": rec.get("indel_id", ""),
                "gene": rec.get("gene", ""),
                "uniprot_id": rec.get("uniprot_id", ""),
                "protein_hgvs": protein_hgvs,
                "variant_class": rec.get("variant_class", ""),
                "label": rec.get("label", ""),
                "clinical_significance": clinical_significance,
                "truncating": rec.get("truncating", ""),
                "wt_len": len(rec.get("wt_seq", "")),
                "mut_len": len(rec.get("mut_seq", "")),
                "length_delta": rec.get("length_delta", ""),
                "split": rec.get("split", ""),
                "clinvar_variation_ids": ",".join(variation_ids),
                "clinvar_allele_ids": ",".join(allele_ids),
                "coordinate_status": status,
                "n_coordinate_entries": len(entries),
            }
            row.update(flat_coordinate_fields(grch37, "grch37"))
            row.update(flat_coordinate_fields(grch38, "grch38"))
            record_writer.writerow(row)

            for entry in entries:
                coord_writer.writerow({field: entry.get(field, rec.get(field, "")) for field in coord_fields})
                n_coordinate_entries += 1

            status_counts[status] += 1
            clinvar_match_counts[match_status] += 1
            n_clinvar_match_entries += len(matched_entries)
            if entries:
                coverage["any_coordinate"] += 1
            if grch37:
                coverage["grch37"] += 1
            if grch38:
                coverage["grch38"] += 1
            if any(has_vcf_coordinate(entry) for entry in entries):
                vcf_coverage["any_vcf_coordinate"] += 1
            if has_vcf_coordinate(grch37):
                vcf_coverage["grch37"] += 1
            if has_vcf_coordinate(grch38):
                vcf_coverage["grch38"] += 1

    for filename in ["baseline_scores.tsv", "splits.csv", "LICENSE"]:
        src = args.v1_dir / filename
        if src.exists():
            shutil.copy2(src, args.out_dir / filename)

    v1_metadata_path = args.v1_dir / "metadata.json"
    v1_metadata = json.loads(v1_metadata_path.read_text()) if v1_metadata_path.exists() else {}
    metadata = {
        "name": "IndelMissense-v1.1-coordinate-augmented",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "v1_dir": str(args.v1_dir),
        "clinvar_variant_summary": str(args.clinvar),
        "n_records": len(records),
        "n_coordinate_entries": n_coordinate_entries,
        "n_clinvar_match_entries": n_clinvar_match_entries,
        "coordinate_coverage": dict(coverage),
        "vcf_coordinate_coverage": dict(vcf_coverage),
        "clinvar_match_coverage": {
            "any_match": sum(count for key, count in clinvar_match_counts.items() if key != "missing"),
        },
        "coordinate_status_counts": dict(status_counts),
        "clinvar_match_status_counts": dict(clinvar_match_counts),
        "clinvar_scan_counts": dict(clinvar_counts),
        "record_set_note": "Same records, sequences, labels, splits and baseline IDs as IndelMissense v1.",
        "coordinate_note": "Coordinates are copied from ClinVar variant_summary rows matched by gene symbol and protein HGVS, with clinical-significance exact matching when available.",
        "v1_metadata": v1_metadata,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    write_readme(args.out_dir, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
