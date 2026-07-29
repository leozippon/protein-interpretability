#!/usr/bin/env python3
"""Readiness audit for OPUS_FINAL_PLAN_20260511.

This script is intentionally lightweight: it does not load models or run GPU
work. It checks whether the inputs assumed by the no-wet-lab final plan are
actually staged locally, and writes a JSON + Markdown audit that can gate the
next H200 jobs.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def exists(path: str) -> dict:
    p = REPO / path
    return {"path": path, "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0}


def glob_any(patterns: list[str]) -> list[dict]:
    out = []
    seen = set()
    for pat in patterns:
        for p in REPO.glob(pat):
            if p.is_file() and p not in seen:
                seen.add(p)
                out.append({"path": rel(p), "size_bytes": p.stat().st_size})
    return sorted(out, key=lambda x: x["path"])


def load_json(path: str) -> dict:
    p = REPO / path
    if not p.exists():
        return {}
    with p.open() as f:
        return json.load(f)


def audit_indels() -> dict:
    tsv = REPO / "r1_encoder_interpretability_benchmark/results/variant_effect/clinvar_indels.tsv"
    supported_jsonl = REPO / "r1_encoder_interpretability_benchmark/results/variant_effect/indel_records_supported_20260504.jsonl"
    predictions_jsonl = REPO / "r1_encoder_interpretability_benchmark/results/variant_effect/indel_mechanism_predictions_20260504.jsonl"
    summary = load_json("r1_encoder_interpretability_benchmark/results/variant_effect/indel_extension_summary.json")
    supported_summary = load_json("r1_encoder_interpretability_benchmark/results/variant_effect/indel_records_supported_20260504_summary.json")
    prediction_summary = load_json("r1_encoder_interpretability_benchmark/results/variant_effect/indel_mechanism_predictions_20260504_summary.json")

    counts = Counter()
    label_counts = Counter()
    supported_by_class = Counter()
    supported_binary_by_class = Counter()
    supported_all_label_by_class = Counter()
    unsupported_reasons = Counter()
    binary_rows = 0
    supported_true = 0
    supported_binary_len_ok = 0
    truncating_supported = 0
    supported_other = 0

    if tsv.exists():
        with tsv.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                cls = row.get("variant_class", "")
                label = row.get("label", "")
                counts[cls] += 1
                label_counts[label] += 1
                if label in {"pathogenic", "benign"}:
                    binary_rows += 1
                if row.get("supported") == "True":
                    supported_true += 1
                    supported_by_class[cls] += 1
                    supported_all_label_by_class[(cls, label)] += 1
                    if label == "other":
                        supported_other += 1
                    if row.get("truncating") == "True":
                        truncating_supported += 1
                    try:
                        wt_len = int(row.get("wt_length") or 0)
                        mut_len = int(row.get("mut_length") or 0)
                    except ValueError:
                        wt_len = mut_len = 0
                    if label in {"pathogenic", "benign"} and 20 <= wt_len <= 1022 and 20 <= mut_len <= 1022:
                        supported_binary_len_ok += 1
                        supported_binary_by_class[cls] += 1
                else:
                    unsupported_reasons[row.get("support_reason", "")] += 1

    current_supported_records = sum(1 for _ in supported_jsonl.open()) if supported_jsonl.exists() else 0
    current_predictions = sum(1 for _ in predictions_jsonl.open()) if predictions_jsonl.exists() else 0

    notes = []
    if supported_binary_len_ok <= current_supported_records:
        notes.append(
            "Current protein-sequence-only reconstruction already appears to cover all binary, length-compatible supported indels."
        )
    if supported_true < 50000:
        notes.append(
            "The final plan's ~80k protein-mappable target is not reachable from the current protein-HGVS reconstruction alone; reaching it requires transcript-aware frameshift handling and/or improved UniProt mapping."
        )
    if unsupported_reasons.get("frameshift_requires_transcript", 0) > 0:
        notes.append(
            "Frameshifts dominate the remaining opportunity but require transcript/coding-frame reconstruction rather than the existing protein-sequence-only scorer."
        )

    return {
        "inputs": {
            "clinvar_indels_tsv": exists("r1_encoder_interpretability_benchmark/results/variant_effect/clinvar_indels.tsv"),
            "supported_jsonl": exists("r1_encoder_interpretability_benchmark/results/variant_effect/indel_records_supported_20260504.jsonl"),
            "prediction_jsonl": exists("r1_encoder_interpretability_benchmark/results/variant_effect/indel_mechanism_predictions_20260504.jsonl"),
        },
        "previous_summaries": {
            "extension": summary,
            "supported_records": supported_summary,
            "predictions": prediction_summary,
        },
        "tsv_counts": {
            "n_rows": sum(counts.values()),
            "variant_class_counts": dict(counts),
            "label_counts": dict(label_counts),
            "binary_label_rows": binary_rows,
            "supported_true_rows": supported_true,
            "supported_by_class": dict(supported_by_class),
            "supported_binary_len_ok_rows": supported_binary_len_ok,
            "supported_binary_len_ok_by_class": dict(supported_binary_by_class),
            "supported_other_label_rows": supported_other,
            "truncating_supported_rows": truncating_supported,
            "unsupported_reasons": dict(unsupported_reasons.most_common()),
            "current_supported_jsonl_records": current_supported_records,
            "current_prediction_jsonl_records": current_predictions,
        },
        "readiness": {
            "f_t1_1_direct_full_scale_from_current_tsv": "limited",
            "reason": "Current staged TSV has many ClinVar indel rows, but only a small binary, length-compatible reconstructable subset is ready for the existing protein-sequence scorer.",
            "notes": notes,
        },
    }


def audit_competitors() -> dict:
    files = {
        "available_baseline_scores": exists("r1_encoder_interpretability_benchmark/results/variant_effect/external_baselines_available_scores_20260507.tsv"),
        "available_baseline_summary": exists("r1_encoder_interpretability_benchmark/results/variant_effect/available_baseline_summary_20260507.json"),
        "alphamissense_matched": exists("r1_encoder_interpretability_benchmark/results/variant_effect/alphamissense_matched_scores_20260504.tsv"),
    }
    local_resources = {
        "cadd": glob_any(["**/*CADD*", "**/*cadd*"]),
        "revel": glob_any(["**/*REVEL*", "**/*revel*"]),
        "spliceai": glob_any(["**/*SpliceAI*", "**/*spliceai*"]),
        "dbnsfp": glob_any(["**/*dbNSFP*", "**/*dbnsfp*"]),
        "gmvp": glob_any(["**/*gMVP*", "**/*gmvp*"]),
    }
    readiness = {
        "missense_external_baselines": "ready",
        "indel_competitor_comparison": "not_ready",
        "reason": (
            "AlphaMissense/gMVP/ESM-1v tables are staged for missense. "
            "No local CADD, REVEL, or SpliceAI score files were found for ClinVar indels."
        ),
    }
    if local_resources["cadd"] or local_resources["spliceai"]:
        readiness["indel_competitor_comparison"] = "partial"
        readiness["reason"] = "Some candidate indel competitor resources exist, but schema/coverage still need matching."
    return {"inputs": files, "local_resource_hits": local_resources, "readiness": readiness}


def audit_r2() -> dict:
    required = {
        "r2_manuscript": exists("r2_decoder_sparse_readout_audit/manuscript/main.tex"),
        "ec_labeled_swissprot": exists("data/zymctrl/ec_labeled_swissprot.fasta"),
        "zymctrl_ec_features": exists("r2_decoder_sparse_readout_audit/results/circuit_analysis/zymctrl/ec_features.pkl"),
        "direct_effect_summary": exists("r2_decoder_sparse_readout_audit/results/circuit_analysis/zymctrl/direct_effect_features_v2_summary_20260503.json"),
        "layer_quality_map": exists("r2_decoder_sparse_readout_audit/results/checkpoint_evaluation/layer_quality_map.json"),
        "cross_model_conservation": exists("r2_decoder_sparse_readout_audit/results/circuit_analysis/cross_model_conservation_v2_20260429_corrfix.json"),
        "generated_metric_triad": exists("r2_decoder_sparse_readout_audit/results/ec_metrics/generated_metric_triad_summary_20260507.json"),
        "metric_calibration": exists("r2_decoder_sparse_readout_audit/results/ec_metrics/ec_metric_calibration_summary_20260507.json"),
    }
    downstream_resources = {
        "pfam": glob_any(["external_resources/ec_metrics/pfam/*", "data/interpro/*pfam*", "data/interpro/*Pfam*"]),
        "clean": glob_any(["external_resources/ec_metrics/clean/**/*"]),
        "cb513": glob_any(["**/*CB513*", "**/*cb513*"]),
        "deeploc": glob_any(["**/*DeepLoc*", "**/*deeploc*"]),
        "fireprotdb": glob_any(["**/*FireProt*", "**/*fireprot*"]),
        "protherm": glob_any(["**/*ProTherm*", "**/*protherm*"]),
    }
    task_readiness = {
        "ec_class_prediction": "partial_ready" if required["ec_labeled_swissprot"]["exists"] else "not_ready",
        "pfam_family_prediction": "partial_ready" if downstream_resources["pfam"] else "not_ready",
        "secondary_structure_cb513": "ready" if downstream_resources["cb513"] else "missing_data",
        "subcellular_localization_deeploc": "ready" if downstream_resources["deeploc"] else "missing_data",
        "stability_fireprot_or_protherm": "ready" if downstream_resources["fireprotdb"] or downstream_resources["protherm"] else "missing_data",
    }
    n_ready = sum(v in {"ready", "partial_ready"} for v in task_readiness.values())
    return {
        "inputs": required,
        "downstream_resource_hits": downstream_resources,
        "task_readiness": task_readiness,
        "readiness": {
            "f_t2_1_five_task_eval": "not_ready" if n_ready < 5 else "ready",
            "ready_or_partial_tasks": n_ready,
            "reason": "Only EC/Pfam-style resources are locally visible; CB513, DeepLoc, and stability benchmarks are not staged locally.",
        },
    }


def audit_manuscripts() -> dict:
    paths = [
        "archive/manuscripts/nature_methods_r1_variant_perturbation/main.tex",
        "r2_decoder_sparse_readout_audit/manuscript/main.tex",
        "archive/manuscripts/nature_methods_r1_variant_perturbation/README.md",
        "r2_decoder_sparse_readout_audit/manuscript/README.md",
        "OPUS_FINAL_PLAN_20260511.md",
        "OPUS_FINAL_PLAN_RETHINK_20260511.md",
    ]
    return {p: exists(p) for p in paths}


def write_md(path: Path, data: dict) -> None:
    indel = data["r1_indel"]
    comp = data["r1_competitors"]
    r2 = data["r2"]
    with path.open("w") as f:
        f.write("# Final Plan Readiness Audit\n\n")
        f.write(f"Date: {data['date']}\n\n")
        f.write("## Overall Verdict\n\n")
        f.write("- F-T1-1 indel scale-up is **partially ready but target-size limited** with current protein-HGVS reconstruction.\n")
        f.write("- F-T1-2 indel competitor comparison is **not ready locally**; CADD/REVEL/SpliceAI indel resources were not found.\n")
        f.write("- F-T2-1 five-task CLT downstream evaluation is **not ready as written**; CB513, DeepLoc, and stability datasets are missing locally.\n")
        f.write("- F-T0 manuscript reframing and F-T1-3 AlphaMissense ensemble are safe first tasks.\n\n")

        f.write("## R1 Indel Readiness\n\n")
        c = indel["tsv_counts"]
        f.write(f"- ClinVar indel TSV rows: {c['n_rows']:,}\n")
        f.write(f"- Binary-label rows: {c['binary_label_rows']:,}\n")
        f.write(f"- Supported reconstructed rows, all labels: {c['supported_true_rows']:,}\n")
        f.write(f"- Binary + length-compatible supported rows: {c['supported_binary_len_ok_rows']:,}\n")
        f.write(f"- Existing supported JSONL records: {c['current_supported_jsonl_records']:,}\n")
        f.write(f"- Existing prediction JSONL records: {c['current_prediction_jsonl_records']:,}\n")
        f.write(f"- Existing damage AUC: {indel['previous_summaries'].get('predictions', {}).get('pathogenicity_auc_damage')}\n\n")
        f.write("Supported binary length-compatible rows by class:\n\n")
        for k, v in sorted(c["supported_binary_len_ok_by_class"].items()):
            f.write(f"- {k}: {v:,}\n")
        f.write("\nTop unsupported reasons:\n\n")
        for k, v in list(c["unsupported_reasons"].items())[:8]:
            f.write(f"- {k}: {v:,}\n")
        f.write("\nInterpretation: the existing protein-sequence-only scorer does not have a clear path to ~80k records without transcript-aware frameshift reconstruction or improved mapping.\n\n")

        f.write("## R1 Competitor Readiness\n\n")
        f.write(f"- Missense external baseline table: {comp['inputs']['available_baseline_scores']['exists']}\n")
        f.write(f"- Indel CADD hits: {len(comp['local_resource_hits']['cadd'])}\n")
        f.write(f"- Indel REVEL hits: {len(comp['local_resource_hits']['revel'])}\n")
        f.write(f"- Indel SpliceAI hits: {len(comp['local_resource_hits']['spliceai'])}\n")
        f.write(f"- dbNSFP/gMVP local hits: {len(comp['local_resource_hits']['dbnsfp'])} / {len(comp['local_resource_hits']['gmvp'])}\n\n")
        f.write("Interpretation: F-T1-2 should not start until indel-compatible competitor score files are staged or explicitly dropped.\n\n")

        f.write("## R2 Downstream Readiness\n\n")
        for task, status in r2["task_readiness"].items():
            f.write(f"- {task}: {status}\n")
        f.write("\nResource hits:\n\n")
        for name, hits in r2["downstream_resource_hits"].items():
            f.write(f"- {name}: {len(hits)} files\n")
        f.write("\nInterpretation: F-T2-1 needs resource staging or a reduced first pass over EC/Pfam only.\n\n")

        f.write("## Recommended Immediate Execution\n\n")
        f.write("1. Run F-T0 manuscript reframing and evidence-tagging.\n")
        f.write("2. Run F-T1-3 SAE x AlphaMissense ensemble locally/H200-light.\n")
        f.write("3. Add a transcript-aware/mapping plan before committing to F-T1-1 full-scale indel scoring.\n")
        f.write("4. Stage CB513, DeepLoc, and FireProtDB/ProTherm before F-T2-1, or reduce F-T2-1 to EC/Pfam pilot.\n")


def main() -> None:
    out_dir = REPO / "results" / "final_plan_readiness"
    out_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "task": "OPUS_FINAL_PLAN_20260511 readiness audit",
        "date": date,
        "manuscripts": audit_manuscripts(),
        "r1_indel": audit_indels(),
        "r1_competitors": audit_competitors(),
        "r2": audit_r2(),
    }
    json_path = out_dir / "final_plan_readiness_20260511.json"
    md_path = out_dir / "final_plan_readiness_20260511.md"
    with json_path.open("w") as f:
        json.dump(data, f, indent=2)
    write_md(md_path, data)
    print(json.dumps({"json": rel(json_path), "md": rel(md_path)}, indent=2))


if __name__ == "__main__":
    main()
