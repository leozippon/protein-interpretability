#!/usr/bin/env python3
"""Preflight residue-level localization inputs for encoder mechanism features."""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


REPO = Path(__file__).resolve().parents[2]
AUDIT = REPO / "r1_encoder_interpretability_benchmark" / "results" / "variant_effect" / "mechanism_feature_audit_firing_20260504.json"
SWISSPROT = REPO / "data" / "swissprot" / "uniprot_sprot.fasta.gz"
OUT_DIR = REPO / "r0_shared_interpretability_framework" / "results" / "v0_20260515" / "encoder"
EXAMPLE_RE = re.compile(r"([^:;\s]+):(\d+)([A-Z])\(([0-9.]+)\)")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def collect_accessions(rows: list[dict[str, Any]]) -> set[str]:
    accessions: set[str] = set()
    for row in rows:
        for accession, _, _, _ in EXAMPLE_RE.findall(row.get("top_firing_examples", "")):
            accessions.add(accession)
    return accessions


def load_swissprot_lengths(path: Path, wanted: set[str]) -> dict[str, int]:
    lengths: dict[str, int] = {}
    if not path.exists():
        return lengths

    current_accessions: set[str] = set()
    current_len = 0

    def flush() -> None:
        if not current_accessions:
            return
        for accession in current_accessions:
            if accession in wanted:
                lengths[accession] = current_len

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:]
                parts = header.split("|")
                current_accessions = set()
                if len(parts) >= 2:
                    current_accessions.add(parts[1])
                first = header.split()[0]
                current_accessions.add(first)
                current_len = 0
            else:
                current_len += len(line.strip())
    flush()
    return lengths


def summarize_examples(rows: list[dict[str, Any]], lengths: dict[str, int]) -> dict[str, Any]:
    parsed_by_feature: Counter[int] = Counter()
    rows_with_any = 0
    examples: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    feature_classification_counts: Counter[str] = Counter()
    manual_counts: Counter[str] = Counter()

    for row in rows:
        matches = EXAMPLE_RE.findall(row.get("top_firing_examples", ""))
        parsed_by_feature[len(matches)] += 1
        if matches:
            rows_with_any += 1
        class_counts[row.get("class", "")] += 1
        feature_classification_counts[row.get("classification", "UNKNOWN")] += 1
        manual_counts[row.get("manual_interpretation", "")] += 1
        for accession, pos_s, aa, activation_s in matches:
            pos = int(pos_s)
            length = lengths.get(accession)
            examples.append(
                {
                    "class": row.get("class", ""),
                    "layer": row.get("layer"),
                    "feature": row.get("feature"),
                    "accession": accession,
                    "pos": pos,
                    "aa": aa,
                    "activation": float(activation_s),
                    "length": length,
                    "pos_norm": pos / length if length else None,
                    "n_terminal_20": bool(length and pos <= 20),
                    "n_terminal_50": bool(length and pos <= 50),
                    "c_terminal_20": bool(length and length - pos < 20),
                    "c_terminal_50": bool(length and length - pos < 50),
                }
            )

    mapped = [item for item in examples if item["length"]]
    pos_norm = [item["pos_norm"] for item in mapped if item["pos_norm"] is not None]
    by_class: dict[str, dict[str, Any]] = {}
    for class_name in sorted({item["class"] for item in examples}):
        cls_items = [item for item in mapped if item["class"] == class_name]
        cls_pos = [item["pos_norm"] for item in cls_items if item["pos_norm"] is not None]
        by_class[class_name] = {
            "n_examples_mapped": len(cls_items),
            "mean_pos_norm": mean(cls_pos) if cls_pos else None,
            "median_pos_norm": median(cls_pos) if cls_pos else None,
            "n_terminal_20_fraction": sum(item["n_terminal_20"] for item in cls_items) / len(cls_items) if cls_items else None,
            "n_terminal_50_fraction": sum(item["n_terminal_50"] for item in cls_items) / len(cls_items) if cls_items else None,
            "c_terminal_20_fraction": sum(item["c_terminal_20"] for item in cls_items) / len(cls_items) if cls_items else None,
            "c_terminal_50_fraction": sum(item["c_terminal_50"] for item in cls_items) / len(cls_items) if cls_items else None,
        }

    return {
        "task": "encoder mechanism localization preflight",
        "input": str(AUDIT.relative_to(REPO)),
        "swissprot": str(SWISSPROT.relative_to(REPO)),
        "n_features": len(rows),
        "n_features_with_any_parsed_example": rows_with_any,
        "parsed_examples_per_feature": dict(sorted(parsed_by_feature.items())),
        "n_top_firing_examples": len(examples),
        "n_examples_length_mapped": len(mapped),
        "length_mapping_fraction": len(mapped) / len(examples) if examples else 0.0,
        "mean_pos_norm": mean(pos_norm) if pos_norm else None,
        "median_pos_norm": median(pos_norm) if pos_norm else None,
        "n_terminal_20_fraction": sum(item["n_terminal_20"] for item in mapped) / len(mapped) if mapped else None,
        "n_terminal_50_fraction": sum(item["n_terminal_50"] for item in mapped) / len(mapped) if mapped else None,
        "c_terminal_20_fraction": sum(item["c_terminal_20"] for item in mapped) / len(mapped) if mapped else None,
        "c_terminal_50_fraction": sum(item["c_terminal_50"] for item in mapped) / len(mapped) if mapped else None,
        "mechanism_class_feature_counts": dict(sorted(class_counts.items())),
        "feature_annotation_classification_counts": dict(sorted(feature_classification_counts.items())),
        "manual_interpretation_counts": dict(sorted(manual_counts.items())),
        "by_mechanism_class": by_class,
        "interpretation": (
            "This preflight verifies that mechanism-selected SAE features have "
            "residue-level top firing positions that can be length-normalized. "
            "It is an input-quality audit for later localization, occlusion and "
            "CRPI metrics, not itself a faithfulness result."
        ),
    }


def write_md(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Encoder Mechanism Localization Preflight",
        "",
        f"- Features: {payload['n_features']}",
        f"- Features with parsed firing examples: {payload['n_features_with_any_parsed_example']}",
        f"- Top firing examples: {payload['n_top_firing_examples']}",
        f"- Length-mapped examples: {payload['n_examples_length_mapped']} ({payload['length_mapping_fraction']:.4f})",
        f"- Mean normalized position: {payload['mean_pos_norm']:.4f}" if payload["mean_pos_norm"] is not None else "- Mean normalized position: NA",
        f"- Median normalized position: {payload['median_pos_norm']:.4f}" if payload["median_pos_norm"] is not None else "- Median normalized position: NA",
        f"- N-terminal <=20 aa fraction: {payload['n_terminal_20_fraction']:.4f}" if payload["n_terminal_20_fraction"] is not None else "- N-terminal <=20 aa fraction: NA",
        f"- C-terminal <=20 aa fraction: {payload['c_terminal_20_fraction']:.4f}" if payload["c_terminal_20_fraction"] is not None else "- C-terminal <=20 aa fraction: NA",
        "",
        "## By Mechanism Class",
        "",
        "| class | mapped examples | mean pos norm | median pos norm | N-term <=20 | C-term <=20 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for class_name, item in payload["by_mechanism_class"].items():
        lines.append(
            "| {class_name} | {n} | {mean_pos:.4f} | {median_pos:.4f} | {n20:.4f} | {c20:.4f} |".format(
                class_name=class_name,
                n=item["n_examples_mapped"],
                mean_pos=item["mean_pos_norm"] if item["mean_pos_norm"] is not None else float("nan"),
                median_pos=item["median_pos_norm"] if item["median_pos_norm"] is not None else float("nan"),
                n20=item["n_terminal_20_fraction"] if item["n_terminal_20_fraction"] is not None else float("nan"),
                c20=item["c_terminal_20_fraction"] if item["c_terminal_20_fraction"] is not None else float("nan"),
            )
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"]])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    payload = load_json(AUDIT)
    rows = payload.get("rows", [])
    wanted = collect_accessions(rows)
    lengths = load_swissprot_lengths(SWISSPROT, wanted)
    summary = summarize_examples(rows, lengths)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "mechanism_localization_preflight.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_md(summary, OUT_DIR / "mechanism_localization_preflight.md")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
