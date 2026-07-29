#!/usr/bin/env python3
"""Package the current reconstructable ClinVar indel benchmark artefact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
R1_RESULTS = REPO / "r1_encoder_interpretability_benchmark" / "results" / "variant_effect"


def iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def split_for_group(group: str) -> str:
    h = int(hashlib.sha1(group.encode("utf-8")).hexdigest()[:8], 16) % 100
    if h < 80:
        return "train"
    if h < 90:
        return "validation"
    return "test"


def roc_auc_binary(y_true: list[int], y_score: list[float]) -> float | None:
    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = sum(1 for y in y_true if y == 0)
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(y_score)), key=lambda i: y_score[i])
    ranks = [0.0] * len(y_score)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and y_score[order[j]] == y_score[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    rank_sum_pos = sum(r for r, y in zip(ranks, y_true) if y == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=R1_RESULTS / "indel_records_supported_20260504.jsonl")
    ap.add_argument("--predictions", type=Path, default=R1_RESULTS / "indel_mechanism_predictions_20260504.jsonl")
    ap.add_argument("--summary", type=Path, default=R1_RESULTS / "indel_mechanism_predictions_20260504_summary.json")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data" / "indelmissense" / "v1")
    args = ap.parse_args()

    records = list(iter_jsonl(args.records))
    preds = {int(row["idx"]): row for row in iter_jsonl(args.predictions)}
    summary = json.loads(args.summary.read_text()) if args.summary.exists() else {}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    record_jsonl = args.out_dir / "records.jsonl"
    records_tsv = args.out_dir / "records.tsv"
    scores_tsv = args.out_dir / "baseline_scores.tsv"
    splits_csv = args.out_dir / "splits.csv"

    labels = Counter()
    classes = Counter()
    split_counts = Counter()
    y_true = []
    y_score = []

    with record_jsonl.open("w") as out_jsonl, records_tsv.open("w", newline="") as out_tsv, scores_tsv.open("w", newline="") as out_scores, splits_csv.open("w", newline="") as out_splits:
        record_fields = [
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
        score_fields = [
            "indel_id",
            "damage_score",
            "predicted_mechanism",
            "prob_DN",
            "prob_GOF",
            "prob_LOF",
        ]
        split_fields = ["indel_id", "split", "group_key"]
        record_writer = csv.DictWriter(out_tsv, fieldnames=record_fields, delimiter="\t")
        score_writer = csv.DictWriter(out_scores, fieldnames=score_fields, delimiter="\t")
        split_writer = csv.DictWriter(out_splits, fieldnames=split_fields)
        record_writer.writeheader()
        score_writer.writeheader()
        split_writer.writeheader()

        for rec in records:
            idx = int(rec["idx"])
            group_key = rec.get("uniprot_id") or rec.get("gene") or str(idx)
            split = split_for_group(group_key)
            pred = preds.get(idx, {})
            probs = pred.get("mechanism_probs", {})
            public = {
                "indel_id": f"indel_{idx:06d}",
                "source_idx": idx,
                "gene": rec.get("gene", ""),
                "uniprot_id": rec.get("uniprot_id", ""),
                "protein_hgvs": rec.get("protein_hgvs", ""),
                "variant_class": rec.get("variant_class", ""),
                "label": rec.get("label", ""),
                "clinical_significance": rec.get("clinical_significance", ""),
                "truncating": bool(rec.get("truncating", False)),
                "wt_seq": rec.get("wt_seq", ""),
                "mut_seq": rec.get("mut_seq", ""),
                "start": rec.get("start"),
                "end": rec.get("end"),
                "inserted_sequence": rec.get("inserted_sequence", ""),
                "length_delta": rec.get("length_delta"),
                "split": split,
            }
            out_jsonl.write(json.dumps(public, separators=(",", ":")) + "\n")
            record_writer.writerow({
                "indel_id": public["indel_id"],
                "gene": public["gene"],
                "uniprot_id": public["uniprot_id"],
                "protein_hgvs": public["protein_hgvs"],
                "variant_class": public["variant_class"],
                "label": public["label"],
                "clinical_significance": public["clinical_significance"],
                "truncating": public["truncating"],
                "wt_len": len(public["wt_seq"]),
                "mut_len": len(public["mut_seq"]),
                "length_delta": public["length_delta"],
                "split": split,
            })
            score_writer.writerow({
                "indel_id": public["indel_id"],
                "damage_score": pred.get("damage_score", ""),
                "predicted_mechanism": pred.get("predicted_mechanism", ""),
                "prob_DN": probs.get("DN", ""),
                "prob_GOF": probs.get("GOF", ""),
                "prob_LOF": probs.get("LOF", ""),
            })
            split_writer.writerow({"indel_id": public["indel_id"], "split": split, "group_key": group_key})
            labels[public["label"]] += 1
            classes[public["variant_class"]] += 1
            split_counts[split] += 1
            if public["label"] in {"pathogenic", "benign"} and "damage_score" in pred:
                y_true.append(1 if public["label"] == "pathogenic" else 0)
                y_score.append(float(pred["damage_score"]))

    auc = roc_auc_binary(y_true, y_score)
    metadata = {
        "name": "IndelMissense-v1-reconstructable",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "records_source": str(args.records),
        "predictions_source": str(args.predictions),
        "summary_source": str(args.summary),
        "n_records": len(records),
        "label_counts": dict(labels),
        "variant_class_counts": dict(classes),
        "split_counts": dict(split_counts),
        "protein_group_split": "sha1(uniprot_id) 80/10/10 train/validation/test",
        "damage_auc_all_records": auc,
        "upstream_summary": summary,
        "limitations": [
            "Only binary-label, length-compatible protein-HGVS indels are included.",
            "Frameshifts requiring transcript-aware reconstruction are not included.",
            "CADD, REVEL and SpliceAI competitor scores are not bundled in this artefact.",
            "Pfam-clan splits are not included because clan mappings were not available for all proteins in this pass.",
        ],
    }
    write_json(args.out_dir / "metadata.json", metadata)

    readme = [
        "# IndelMissense v1 Reconstructable Benchmark",
        "",
        "This directory packages the current R1 protein-sequence indel benchmark artefact.",
        "It is intentionally scoped to the subset that can be reconstructed from staged ClinVar protein HGVS and UniProt sequences.",
        "",
        "## Contents",
        "",
        "- `records.jsonl`: full WT/mutant sequence records and labels.",
        "- `records.tsv`: compact tabular metadata without full sequences.",
        "- `baseline_scores.tsv`: current SAE perturbation damage score and mechanism probabilities.",
        "- `splits.csv`: deterministic protein-level train/validation/test split.",
        "- `metadata.json`: counts, source files, baseline AUC and limitations.",
        "",
        "## Counts",
        "",
        f"- Records: {len(records)}",
        f"- Labels: {dict(labels)}",
        f"- Variant classes: {dict(classes)}",
        f"- Splits: {dict(split_counts)}",
        f"- Current damage-score AUC on all records: {auc:.4f}" if auc is not None else "- Current damage-score AUC on all records: unavailable",
        "",
        "## Scope",
        "",
        "This is not the full ~80k ClinVar indel target from the final plan.",
        "The full target remains blocked by transcript-aware frameshift reconstruction and improved UniProt mapping.",
        "Use this artefact for the current manuscript's bounded indel claim: a reconstructable protein-sequence indel diagnostic benchmark.",
        "",
    ]
    (args.out_dir / "README.md").write_text("\n".join(readme))
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
