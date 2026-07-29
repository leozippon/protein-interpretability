#!/usr/bin/env python
"""Merge channelopathy curation-worker TSVs into consolidated research labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "r1_encoder_interpretability_benchmark" / "data" / "channelopathy"
WORK_DIR = DATA_DIR / "curation_work"

FIELDS = [
    "gene",
    "variant_protein",
    "variant_cdna",
    "condition",
    "mechanism_label",
    "drug_response_label",
    "evidence_level",
    "evidence_type",
    "source_id",
    "source_url_or_doi",
    "notes",
]
VALID_GENES = {"KCNQ1", "SCN5A", "KCNH2", "CACNA1C"}
VALID_MECHANISMS = {"LOF", "GOF", "DN", "mixed_complex", "unknown"}
VALID_LEVELS = {"high", "medium", "low"}


def norm(value: str | None) -> str:
    text = "" if value is None else str(value).strip()
    return text if text else "unknown"


def norm_mechanism(value: str | None) -> str:
    text = norm(value)
    aliases = {
        "dominant_negative": "DN",
        "dominant-negative": "DN",
        "mixed": "mixed_complex",
        "complex": "mixed_complex",
        "uncertain": "unknown",
    }
    return aliases.get(text, text)


def read_tsv(path: Path) -> tuple[list[dict], list[str]]:
    rows = []
    problems = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        missing = [field for field in FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            return [], [f"{path}: missing fields {missing}"]
        for i, row in enumerate(reader, start=2):
            rec = {field: norm(row.get(field)) for field in FIELDS}
            rec["gene"] = rec["gene"].upper()
            rec["mechanism_label"] = norm_mechanism(rec["mechanism_label"])
            rec["evidence_level"] = rec["evidence_level"].lower()
            if rec["gene"] not in VALID_GENES:
                problems.append(f"{path}:{i}: invalid gene {rec['gene']}")
            if rec["mechanism_label"] not in VALID_MECHANISMS:
                problems.append(f"{path}:{i}: invalid mechanism {rec['mechanism_label']}")
            if rec["evidence_level"] not in VALID_LEVELS:
                original = rec["evidence_level"]
                rec["evidence_level"] = "low"
                rec["evidence_type"] = f"{rec['evidence_type']};clinvar_candidate_index"
                rec["notes"] = f"{rec['notes']} | ClinVar classification summary: {original}"
            rec["_source_file"] = path.name
            rows.append(rec)
    return rows, problems


def evidence_rank(level: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(level, 0)


def deduplicate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["gene"],
            row["variant_protein"].replace(" ", ""),
            row["condition"].lower(),
            row["mechanism_label"],
        )
        grouped[key].append(row)
    merged = []
    duplicates = []
    for key, items in grouped.items():
        items = sorted(items, key=lambda r: (evidence_rank(r["evidence_level"]), r["source_id"] != "unknown"), reverse=True)
        base = dict(items[0])
        source_ids = []
        source_urls = []
        source_files = []
        notes = []
        for item in items:
            if item["source_id"] not in source_ids:
                source_ids.append(item["source_id"])
            if item["source_url_or_doi"] not in source_urls:
                source_urls.append(item["source_url_or_doi"])
            if item["_source_file"] not in source_files:
                source_files.append(item["_source_file"])
            if item["notes"] not in notes:
                notes.append(item["notes"])
        base["source_id"] = ";".join(source_ids)
        base["source_url_or_doi"] = ";".join(source_urls)
        base["notes"] = " | ".join(notes)[:1200]
        base["_source_file"] = ";".join(source_files)
        merged.append(base)
        if len(items) > 1:
            duplicates.append({"key": key, "n": len(items), "source_files": source_files})
    merged.sort(key=lambda r: (r["gene"], r["variant_protein"], r["condition"], r["mechanism_label"]))
    return merged, duplicates


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_sources(path: Path, rows: list[dict], sources: list[Path], problems: list[str], duplicates: list[dict]) -> None:
    by_gene = Counter(r["gene"] for r in rows)
    by_mech = Counter(r["mechanism_label"] for r in rows)
    by_level = Counter(r["evidence_level"] for r in rows)
    lines = ["# Channelopathy Label Sources\n"]
    lines.append("These labels are research-use curation artifacts, not clinical guidance.\n")
    lines.append("## Inputs\n")
    for src in sources:
        lines.append(f"- `{src}`")
    lines.append("\n## Counts\n")
    lines.append(f"- Consolidated rows: {len(rows)}")
    lines.append(f"- By gene: {dict(sorted(by_gene.items()))}")
    lines.append(f"- By mechanism: {dict(sorted(by_mech.items()))}")
    lines.append(f"- By evidence level: {dict(sorted(by_level.items()))}")
    lines.append(f"- Duplicate groups merged: {len(duplicates)}")
    lines.append("\n## Validation Notes\n")
    if problems:
        for problem in problems:
            lines.append(f"- WARNING: {problem}")
    else:
        lines.append("- Schema validation passed for required fields, gene set, mechanism labels, and evidence levels.")
    lines.append("\n## Curation Caveats\n")
    lines.append("- Mechanism labels should be treated as literature-backed research labels.")
    lines.append("- Rows with `unknown` mechanism are candidate/source-index rows and should not be used as positive mechanism labels.")
    lines.append("- Drug-response labels are sparse and should be analyzed separately from mechanism labels.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, default=WORK_DIR)
    ap.add_argument("--out-tsv", type=Path, default=DATA_DIR / "channelopathy_mechanism_labels.tsv")
    ap.add_argument("--out-positive-tsv", type=Path, default=DATA_DIR / "channelopathy_mechanism_positive_labels.tsv")
    ap.add_argument("--out-drug-tsv", type=Path, default=DATA_DIR / "channelopathy_drug_response_labels.tsv")
    ap.add_argument("--out-md", type=Path, default=DATA_DIR / "channelopathy_label_sources.md")
    ap.add_argument("--out-json", type=Path, default=DATA_DIR / "channelopathy_label_summary.json")
    args = ap.parse_args()

    sources = sorted(args.work_dir.glob("*_worker.tsv"))
    rows = []
    problems = []
    for src in sources:
        src_rows, src_problems = read_tsv(src)
        rows.extend(src_rows)
        problems.extend(src_problems)
    merged, duplicates = deduplicate(rows)
    write_tsv(args.out_tsv, merged)
    positive = [r for r in merged if r["mechanism_label"] != "unknown" and r["evidence_level"] in {"high", "medium"}]
    drug = [r for r in merged if r["drug_response_label"] != "unknown"]
    write_tsv(args.out_positive_tsv, positive)
    write_tsv(args.out_drug_tsv, drug)
    write_sources(args.out_md, merged, sources, problems, duplicates)
    summary = {
        "task": "R1 T1-E channelopathy label curation merge",
        "status": "completed_with_warnings" if problems else "completed",
        "inputs": [str(p) for p in sources],
        "outputs": {
            "tsv": str(args.out_tsv),
            "positive_tsv": str(args.out_positive_tsv),
            "drug_tsv": str(args.out_drug_tsv),
            "md": str(args.out_md),
        },
        "n_input_rows": len(rows),
        "n_rows": len(merged),
        "n_positive_mechanism_rows": len(positive),
        "n_drug_response_rows": len(drug),
        "counts_by_gene": dict(Counter(r["gene"] for r in merged)),
        "counts_by_mechanism": dict(Counter(r["mechanism_label"] for r in merged)),
        "counts_by_evidence_level": dict(Counter(r["evidence_level"] for r in merged)),
        "problems": problems,
        "duplicates": duplicates,
    }
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
