#!/usr/bin/env python3
"""Build v0 ProteinInterpret benchmark tables from existing artefacts."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "r0_shared_interpretability_framework" / "results" / "v0_20260515"
R1 = REPO / "r1_encoder_interpretability_benchmark" / "results" / "variant_effect"
R1_ANNOT = REPO / "r1_encoder_interpretability_benchmark" / "results" / "annotation_alignment"
R2 = REPO / "r2_decoder_sparse_readout_audit" / "results" / "circuit_analysis"
R2_EC = REPO / "r2_decoder_sparse_readout_audit" / "results" / "ec_metrics"
R2_STEER = REPO / "r2_decoder_sparse_readout_audit" / "results" / "steering_benchmark"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def fmt_ci(ci: Any) -> str:
    if not isinstance(ci, list) or len(ci) != 2:
        return ""
    if ci[0] is None or ci[1] is None:
        return ""
    return f"[{float(ci[0]):.4f},{float(ci[1]):.4f}]"


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def add_encoder_pathogenicity(rows: list[dict[str, Any]]) -> None:
    summary = load_json(R1 / "available_baseline_summary_20260507.json")
    for row in summary.get("pathogenicity_table", []):
        rows.append(
            {
                "track": "encoder",
                "task": "pathogenicity",
                "dataset": row.get("cohort", ""),
                "method": row.get("method", ""),
                "metric": "ROC_AUC",
                "value": f"{float(row.get('auc')):.4f}",
                "ci95": fmt_ci(row.get("ci95")),
                "n": row.get("n", ""),
                "gate": "reference_result",
                "interpretation": "predictive utility table; not an explanation-faithfulness result",
                "source": "r1_encoder_interpretability_benchmark/results/variant_effect/available_baseline_summary_20260507.json",
            }
        )

    grouped = load_json(R1 / "grouped_pathogenicity_baselines_20260511.json")
    for row in grouped.get("methods", []):
        rows.append(
            {
                "track": "encoder",
                "task": "pathogenicity_gene_grouped",
                "dataset": grouped.get("cohort", "ClinVar2000"),
                "method": row.get("method", ""),
                "metric": "ROC_AUC",
                "value": f"{float(row.get('auc')):.4f}",
                "ci95": fmt_ci(row.get("ci95_gene_bootstrap")),
                "n": row.get("n", ""),
                "gate": "gene_grouped_external_baseline",
                "interpretation": "strong scalar methods remain above SAE-family methods under gene bootstrap/grouped evaluation",
                "source": "r1_encoder_interpretability_benchmark/results/variant_effect/grouped_pathogenicity_baselines_20260511.json",
            }
        )


def add_encoder_indel(rows: list[dict[str, Any]]) -> None:
    summary = load_json(R1 / "indel_protein_baselines_20260516" / "summary.json")
    for method, metric in summary.get("metrics", {}).items():
        if "auc" not in metric:
            continue
        rows.append(
            {
                "track": "encoder",
                "task": "indel_pathogenicity",
                "dataset": "IndelMissense_v1",
                "method": method,
                "metric": "ROC_AUC",
                "value": f"{float(metric.get('auc')):.4f}",
                "ci95": fmt_ci(metric.get("ci95")),
                "n": metric.get("n", ""),
                "gate": "standalone_sae_gate_failed" if method == "sae_damage_score" else "baseline",
                "interpretation": "IndelMissense supports benchmark/resource framing; SAE alone is not strongest",
                "source": "r1_encoder_interpretability_benchmark/results/variant_effect/indel_protein_baselines_20260516/summary.json",
            }
        )

    preflight = load_json(OUT / "encoder" / "indelmissense_v11_resource_preflight.json")
    for metric, value in [
        ("grch38_vcf_coverage", preflight.get("grch38_vcf_coverage")),
        ("grch38_exact_snv_compatible_records", preflight.get("grch38_exact_snv_compatible_records")),
        (
            "grch38_exact_snv_compatible_fraction_of_vcf",
            preflight.get("grch38_exact_snv_compatible_fraction_of_vcf"),
        ),
    ]:
        if value is None:
            continue
        rows.append(
            {
                "track": "encoder",
                "task": "indel_resource_preflight",
                "dataset": "IndelMissense_v1.1_coordinates",
                "method": "VCF allele compatibility audit",
                "metric": metric,
                "value": f"{float(value):.4f}" if isinstance(value, float) else value,
                "ci95": "",
                "n": preflight.get("n_records", ""),
                "gate": "coverage_audit",
                "interpretation": preflight.get("resource_implication", ""),
                "source": "r0_shared_interpretability_framework/results/v0_20260515/encoder/indelmissense_v11_resource_preflight.json",
            }
        )


def add_encoder_proteingym(rows: list[dict[str, Any]]) -> None:
    summary = load_json(R1 / "proteingym_benchmark_sae_signed_diagnostics_20260503.json")
    rho = summary.get("diagnostics", {}).get("rho", {})
    method_names = {
        "llr": "ESM-2 LLR",
        "sae": "SAE raw disruption",
        "negated_sae": "SAE disruption sign-corrected",
        "ensemble_signed": "SAE+LLR sign-corrected",
    }
    for key, name in method_names.items():
        metric = rho.get(key, {})
        if metric.get("mean") is None:
            continue
        rows.append(
            {
                "track": "encoder",
                "task": "dms_fitness",
                "dataset": "ProteinGym_substitutions",
                "method": name,
                "metric": "mean_Spearman_rho",
                "value": f"{float(metric.get('mean')):.4f}",
                "ci95": fmt_ci(metric.get("ci95")),
                "n": metric.get("n", ""),
                "gate": "negative_for_sae_family" if key != "llr" else "reference_baseline",
                "interpretation": "SAE-family scores do not beat ESM-2 LLR on mean DMS correlation",
                "source": "r1_encoder_interpretability_benchmark/results/variant_effect/proteingym_benchmark_sae_signed_diagnostics_20260503.json",
            }
        )


def add_encoder_annotation_alignment(rows: list[dict[str, Any]]) -> None:
    summary = load_json(R1_ANNOT / "expanded_summary_firing_20260503.json")
    for layer, layer_payload in sorted(summary.get("per_layer", {}).items(), key=lambda item: int(item[0])):
        n_features = layer_payload.get("n_features_with_firing", "")
        for metric, value in [
            ("known_features_after_expansion", layer_payload.get("new_known")),
            ("useful_features_after_expansion", layer_payload.get("new_useful")),
            ("delta_known_features_after_expansion", layer_payload.get("new_known", 0) - layer_payload.get("orig_known", 0)),
            ("delta_useful_features_after_expansion", layer_payload.get("new_useful", 0) - layer_payload.get("orig_useful", 0)),
        ]:
            rows.append(
                {
                    "track": "encoder",
                    "task": "feature_annotation_coverage",
                    "dataset": "SAE_feature_annotation_alignment",
                    "method": f"ESM-2 SAE layer {layer}",
                    "metric": metric,
                    "value": value,
                    "ci95": "",
                    "n": n_features,
                    "gate": "coverage_audit",
                    "interpretation": "expanded GO/Pfam/BioLiP annotations over firing-position-enabled SAE features; not a causal-faithfulness result",
                    "source": "r1_encoder_interpretability_benchmark/results/annotation_alignment/expanded_summary_firing_20260503.json",
                }
            )

    audit = load_json(R1 / "mechanism_feature_audit_firing_20260504.json")
    audit_rows = audit.get("rows", [])
    if audit_rows:
        counts: dict[str, int] = {}
        for row in audit_rows:
            key = str(row.get("classification", "UNKNOWN"))
            counts[key] = counts.get(key, 0) + 1
        for classification in ["KNOWN", "PARTIAL", "NOVEL"]:
            rows.append(
                {
                    "track": "encoder",
                    "task": "mechanism_feature_audit",
                    "dataset": "mechanism_feature_top150",
                    "method": "annotation-selected SAE features",
                    "metric": f"n_{classification.lower()}_features",
                    "value": counts.get(classification, 0),
                    "ci95": "",
                    "n": len(audit_rows),
                    "gate": "manual_audit",
                    "interpretation": "top mechanism-classifier features mostly have weak/family-level annotations, motivating benchmark localization and faithfulness metrics",
                    "source": "r1_encoder_interpretability_benchmark/results/variant_effect/mechanism_feature_audit_firing_20260504.json",
                }
            )

    loc = load_json(OUT / "encoder" / "mechanism_localization_preflight.json")
    if loc:
        for metric in [
            "n_features_with_any_parsed_example",
            "n_top_firing_examples",
            "length_mapping_fraction",
            "mean_pos_norm",
            "median_pos_norm",
            "n_terminal_20_fraction",
            "c_terminal_20_fraction",
        ]:
            value = loc.get(metric)
            if value is None:
                continue
            rows.append(
                {
                    "track": "encoder",
                    "task": "mechanism_localization_preflight",
                    "dataset": "mechanism_feature_top150",
                    "method": "top-firing positions + SwissProt length map",
                    "metric": metric,
                    "value": f"{float(value):.4f}" if isinstance(value, float) else value,
                    "ci95": "",
                    "n": loc.get("n_features", ""),
                    "gate": "input_quality_pass",
                    "interpretation": loc.get("interpretation", ""),
                    "source": "r0_shared_interpretability_framework/results/v0_20260515/encoder/mechanism_localization_preflight.json",
                }
            )
        for class_name, item in loc.get("by_mechanism_class", {}).items():
            rows.append(
                {
                    "track": "encoder",
                    "task": "mechanism_localization_preflight",
                    "dataset": f"mechanism_feature_top150_{class_name}",
                    "method": "top-firing positions + SwissProt length map",
                    "metric": "mean_pos_norm",
                    "value": f"{float(item.get('mean_pos_norm')):.4f}" if item.get("mean_pos_norm") is not None else "",
                    "ci95": "",
                    "n": item.get("n_examples_mapped", ""),
                    "gate": "input_quality_pass",
                    "interpretation": "mechanism-class stratified position distribution for later localization/CRPI controls",
                    "source": "r0_shared_interpretability_framework/results/v0_20260515/encoder/mechanism_localization_preflight.json",
                }
            )


def add_encoder_negative_gates(rows: list[dict[str, Any]]) -> None:
    gate_files = [
        (
            "vamp_like_abundance",
            "ProteinGym_VAMP_like_proxy",
            R1 / "vampseq_abundance_proxy_20260518" / "summary.json",
        ),
        (
            "low_homology_rescue",
            "ClinVar2000_low_homology_proxy",
            R1 / "low_homology_stratification_20260518" / "summary.json",
        ),
        (
            "am_sae_disagreement_typing",
            "ClinVar2000_AM_SAE_disagreements",
            R1 / "extended_disagreement_typing_20260518" / "summary.json",
        ),
    ]
    for task, dataset, path in gate_files:
        summary = load_json(path)
        if not summary:
            continue
        value = ""
        if "n_sae_family_beats_am" in summary:
            value = f"{summary['n_sae_family_beats_am']}/{summary.get('n_assays_with_am_n_ge_30')}"
        elif "low_q1_delta_sae_llr_minus_am" in summary:
            value = f"{float(summary['low_q1_delta_sae_llr_minus_am']):.4f}"
        elif "n_significant_q05" in summary:
            value = str(summary["n_significant_q05"])
        rows.append(
            {
                "track": "encoder",
                "task": task,
                "dataset": dataset,
                "method": "SAE-family vs AlphaMissense",
                "metric": "gate_statistic",
                "value": value,
                "ci95": fmt_ci(summary.get("low_q1_delta_ci")),
                "n": summary.get("n_variants", summary.get("n_disagreements", "")),
                "gate": "FAIL" if summary.get("acceptance_pass") is False or summary.get("acceptance_pass_proxy") is False else "",
                "interpretation": summary.get("acceptance_gate", summary.get("acceptance_gate_proxy", "")),
                "source": str(path.relative_to(REPO)),
            }
        )


def build_encoder_tables() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    add_encoder_pathogenicity(rows)
    add_encoder_indel(rows)
    add_encoder_proteingym(rows)
    add_encoder_annotation_alignment(rows)
    add_encoder_negative_gates(rows)

    datasets = [
        {
            "dataset": "ClinVar2000",
            "n": 2000,
            "status": "runnable",
            "role": "missense pathogenicity and explanation audit",
            "source": "r1_encoder_interpretability_benchmark/results/variant_effect/available_baseline_summary_20260507.json",
        },
        {
            "dataset": "CancerHoldout101",
            "n": 101,
            "status": "runnable",
            "role": "small cancer variant holdout",
            "source": "r1_encoder_interpretability_benchmark/results/variant_effect/available_baseline_summary_20260507.json",
        },
        {
            "dataset": "IndelMissense_v1.1_coordinates",
            "n": 6649,
            "status": "coordinate_augmented",
            "role": "clinical protein indel benchmark with ClinVar GRCh37/GRCh38 coordinates",
            "source": "data/indelmissense/v1.1_coordinates/metadata.json",
        },
        {
            "dataset": "ProteinGym_substitutions",
            "n": 214,
            "status": "scored_subset",
            "role": "DMS fitness correlation audit",
            "source": "r1_encoder_interpretability_benchmark/results/variant_effect/proteingym_benchmark_sae_signed_diagnostics_20260503.json",
        },
        {
            "dataset": "ProteinGym_VAMP_like_proxy",
            "n": 9,
            "status": "proxy_completed",
            "role": "abundance/stability-oriented rescue audit",
            "source": "r1_encoder_interpretability_benchmark/results/variant_effect/vampseq_abundance_proxy_20260518/summary.json",
        },
        {
            "dataset": "SAE_feature_annotation_alignment",
            "n": 5,
            "status": "completed_coverage_audit",
            "role": "GO/Pfam/BioLiP annotation coverage for firing-position-enabled SAE features",
            "source": "r1_encoder_interpretability_benchmark/results/annotation_alignment/expanded_summary_firing_20260503.json",
        },
        {
            "dataset": "mechanism_feature_top150",
            "n": 150,
            "status": "completed_manual_audit",
            "role": "manual audit of top SAE features selected by mechanism classifiers",
            "source": "r1_encoder_interpretability_benchmark/results/variant_effect/mechanism_feature_audit_firing_20260504.json",
        },
    ]
    resources = [
        {
            "resource": "dbNSFP_GRCh38",
            "local_status": "missing_in_repo",
            "remote_status": "visible_on_H200_oss_pvc",
            "next_action": "run exact coverage check with tabix/pysam; expect low REVEL/dbNSFP exact coverage for IndelMissense because GRCh38 VCF alleles are non-SNV",
        },
        {
            "resource": "PrimateAI-3D",
            "local_status": "unavailable_gated",
            "remote_status": "not_used",
            "next_action": "exclude from current benchmark unless access is approved",
        },
    ]

    out_dir = OUT / "encoder"
    fields = ["track", "task", "dataset", "method", "metric", "value", "ci95", "n", "gate", "interpretation", "source"]
    write_tsv(out_dir / "method_result_table.tsv", rows, fields)
    write_tsv(out_dir / "dataset_table.tsv", datasets, ["dataset", "n", "status", "role", "source"])
    write_tsv(out_dir / "resource_readiness.tsv", resources, ["resource", "local_status", "remote_status", "next_action"])

    return {
        "n_rows": len(rows),
        "datasets": datasets,
        "resources": resources,
        "headline": "Encoder v0 is evidence-rich for predictive utility and negative gates, but still lacks systematic localization/faithfulness metrics.",
    }


def add_decoder_discovery(rows: list[dict[str, Any]]) -> None:
    null = load_json(R2 / "universal_atlas_balanced200_wide_null_control_30x_20260513.json")
    observed = null.get("observed_triplet_counts", {})
    null_runs = null.get("null_runs", [])
    for threshold, count in observed.items():
        vals = [run.get("triplet_counts", {}).get(threshold, 0) for run in null_runs]
        null_mean = sum(vals) / len(vals) if vals else ""
        null_max = max(vals) if vals else ""
        rows.append(
            {
                "track": "decoder",
                "task": "cross_model_triplet_discovery",
                "dataset": "balanced200_three_model_atlas",
                "method": "CLT triplet matching",
                "metric": f"n_triplets_abs_corr_ge_{threshold}",
                "value": count,
                "ci95": "",
                "n": null.get("input", {}).get("n_sequences", ""),
                "gate": "PASS",
                "interpretation": f"observed={count}; 30x null mean={null_mean:.3f}; null max={null_max}",
                "source": "r2_decoder_sparse_readout_audit/results/circuit_analysis/universal_atlas_balanced200_wide_null_control_30x_20260513.json",
            }
        )


def add_decoder_characterization(rows: list[dict[str, Any]]) -> None:
    synth = load_json(R2 / "triplet_synthesis_20260515_nperm2000" / "summary.json")
    if not synth:
        return
    rows.append(
        {
            "track": "decoder",
            "task": "triplet_characterization",
            "dataset": "38_universal_triplets",
            "method": "M-2 synthesis",
            "metric": "any_significant_characterization",
            "value": f"{synth.get('n_any_significant')}/{synth.get('n_triplets')}",
            "ci95": "",
            "n": synth.get("n_triplets", ""),
            "gate": "PASS_readout_not_biology",
            "interpretation": "Most conserved triplets have low-level signatures; this does not establish biological primitives.",
            "source": "r2_decoder_sparse_readout_audit/results/circuit_analysis/triplet_synthesis_20260515_nperm2000/summary.json",
        }
    )
    for test, count in sorted(synth.get("per_test_significant_counts", {}).items()):
        rows.append(
            {
                "track": "decoder",
                "task": "triplet_characterization",
                "dataset": "38_universal_triplets",
                "method": "M-2 synthesis",
                "metric": f"{test}_significant_triplets",
                "value": count,
                "ci95": "",
                "n": synth.get("n_triplets", ""),
                "gate": "diagnostic",
                "interpretation": "full multi-label signature count",
                "source": "r2_decoder_sparse_readout_audit/results/circuit_analysis/triplet_synthesis_20260515_nperm2000/summary.json",
            }
        )


def add_decoder_negative_gates(rows: list[dict[str, Any]]) -> None:
    gate_paths = [
        ("triplet_biological_annotation", "SwissProt_rich_labels", R2 / "swissprot_triplet_annotation_20260513" / "summary.json"),
        ("triplet_basis_probe", "Pfam_EC_secondary_structure", R2 / "triplet_basis_probes_20260513" / "summary.json"),
        ("clt_feature_ablation", "N_terminal_sink_triplets", R2 / "attention_sink_causal_ablation_20260517" / "summary.json"),
        ("attention_head_ablation", "N_terminal_sink_heads", R2 / "attention_head_sink_ablation_20260518" / "summary.json"),
        ("sink_set_ablation_top8", "N_terminal_sink_head_sets", R2 / "attention_sink_set_ablation_20260518" / "summary.json"),
        ("sink_set_ablation_top32", "N_terminal_sink_head_sets", R2 / "attention_sink_set_ablation_top32_20260518" / "summary.json"),
    ]
    for task, dataset, path in gate_paths:
        summary = load_json(path)
        if not summary:
            continue
        if task == "triplet_basis_probe":
            for item in summary.get("tasks", []):
                trip = item.get("triplet_basis", {})
                esm = item.get("esm2_mean_pooled", {})
                metric = "macro_f1" if "macro_f1" in trip else "r2_uniform_average"
                rows.append(
                    {
                        "track": "decoder",
                        "task": task,
                        "dataset": item.get("task", dataset),
                        "method": "38-dim triplet basis vs ESM-2 mean pooled",
                        "metric": metric,
                        "value": f"{trip.get(metric):.4f} vs {esm.get(metric):.4f}",
                        "ci95": "",
                        "n": trip.get("n", ""),
                        "gate": "FAIL",
                        "interpretation": "triplet basis does not replace dense ESM-2 embeddings",
                        "source": str(path.relative_to(REPO)),
                    }
                )
            continue
        outcome = summary.get("outcome", "")
        if "strict_pass" in summary:
            outcome = "PASS" if summary.get("strict_pass") else "FAIL"
        rows.append(
            {
                "track": "decoder",
                "task": task,
                "dataset": dataset,
                "method": summary.get("task", task),
                "metric": "benchmark_gate",
                "value": outcome or summary.get("status", ""),
                "ci95": "",
                "n": summary.get("n_rows", summary.get("n_triplets", "")),
                "gate": outcome or "FAIL",
                "interpretation": "current evidence remains readout/diagnostic rather than causal control",
                "source": str(path.relative_to(REPO)),
            }
        )


def add_decoder_diagnostics(rows: list[dict[str, Any]]) -> None:
    quality = load_json(R2 / "universal_atlas_quality_diagnostic_20260516" / "summary.json")
    for item in quality.get("atlas_rows", []):
        rows.append(
            {
                "track": "decoder",
                "task": "checkpoint_quality_diagnostic",
                "dataset": "universal_triplet_count",
                "method": item.get("name", ""),
                "metric": "n_universal_triplets",
                "value": item.get("n_triplets", item.get("n_universal_triplets", "")),
                "ci95": "",
                "n": "",
                "gate": quality.get("outcome", ""),
                "interpretation": "universal triplet count can separate mature and weak CLT checkpoints",
                "source": "r2_decoder_sparse_readout_audit/results/circuit_analysis/universal_atlas_quality_diagnostic_20260516/summary.json",
            }
        )
    attn = load_json(R2 / "attention_output_transcoder_pilot_20260514_l23" / "summary.json")
    eval_payload = attn.get("eval", {})
    if eval_payload:
        for metric in ["mean_eval_fvu", "max_attention_corr", "max_first2_delta"]:
            rows.append(
                {
                    "track": "decoder",
                    "task": "attention_output_sparse_pilot",
                    "dataset": "ZymCTRL_layer23_attention_output",
                    "method": "attention-output sparse dictionary",
                    "metric": metric,
                    "value": f"{float(eval_payload.get(metric)):.4f}",
                    "ci95": "",
                    "n": eval_payload.get("eval_sequences", ""),
                    "gate": "readout_PASS_causal_FAIL",
                    "interpretation": "attention-output sparse model is a promising readout, but ablation did not establish causality",
                    "source": "r2_decoder_sparse_readout_audit/results/circuit_analysis/attention_output_transcoder_pilot_20260514_l23/summary.json",
                }
            )


def add_decoder_biological_context(rows: list[dict[str, Any]]) -> None:
    sink = load_json(R2 / "attention_sink_biological_correlate_20260516" / "summary.json")
    if sink:
        significant_tests = sink.get("significant_tests", [])
        significant_triplets = {item.get("triplet_id") for item in significant_tests if item.get("triplet_id")}
        best_q = min((float(item.get("q_value", 1.0)) for item in significant_tests), default=None)
        for metric, value in [
            ("n_attention_sink_triplets", len(sink.get("attention_sink_triplets", []))),
            ("n_significant_position_or_kmer_tests", len(significant_tests)),
            ("n_sink_triplets_with_significant_context", len(significant_triplets)),
        ]:
            rows.append(
                {
                    "track": "decoder",
                    "task": "attention_sink_biological_context",
                    "dataset": "N_terminal_sink_triplets",
                    "method": "position/k-mer enrichment tests",
                    "metric": metric,
                    "value": value,
                    "ci95": "",
                    "n": sink.get("n_tests", ""),
                    "gate": "PASS_context_not_causality" if sink.get("acceptance_pass") else "FAIL",
                    "interpretation": sink.get("acceptance_gate", "attention-sink triplets are tested for positional and local-sequence enrichment"),
                    "source": "r2_decoder_sparse_readout_audit/results/circuit_analysis/attention_sink_biological_correlate_20260516/summary.json",
                }
            )
        if best_q is not None:
            rows.append(
                {
                    "track": "decoder",
                    "task": "attention_sink_biological_context",
                    "dataset": "N_terminal_sink_triplets",
                    "method": "position/k-mer enrichment tests",
                    "metric": "best_bh_q_value",
                    "value": f"{best_q:.4e}",
                    "ci95": "",
                    "n": sink.get("n_tests", ""),
                    "gate": "PASS_context_not_causality",
                    "interpretation": "strong N-terminal/edge-context enrichment is contextual evidence, not a causal generation claim",
                    "source": "r2_decoder_sparse_readout_audit/results/circuit_analysis/attention_sink_biological_correlate_20260516/summary.json",
                }
            )

    coverage = load_json(R2 / "universal_primitives_balanced200_resource_annotation_20260512" / "resource_coverage.json")
    for metric, value in coverage.get("resource_coverage", {}).items():
        rows.append(
            {
                "track": "decoder",
                "task": "triplet_resource_coverage",
                "dataset": "balanced200_top_firing_positions",
                "method": "Pfam/SwissProt/AlphaFold residue-resource join",
                "metric": metric,
                "value": value,
                "ci95": "",
                "n": coverage.get("n_unique_accessions", ""),
                "gate": "coverage_audit",
                "interpretation": coverage.get("interpretation", "coverage audit over top-firing positions"),
                "source": "r2_decoder_sparse_readout_audit/results/circuit_analysis/universal_primitives_balanced200_resource_annotation_20260512/resource_coverage.json",
            }
        )


def add_decoder_generation_metrics(rows: list[dict[str, Any]]) -> None:
    calibration = load_json(R2_EC / "ec_metric_calibration_summary_20260507.json")
    for item in calibration.get("rows", []):
        metric = item.get("metric", "")
        rows.append(
            {
                "track": "decoder",
                "task": "generation_metric_calibration",
                "dataset": "real_lysozyme_vs_random_uniref50",
                "method": "external EC/structure metric stack",
                "metric": f"{metric}_effect_size_d",
                "value": f"{float(item.get('effect_size_d')):.4f}",
                "ci95": "",
                "n": f"{item.get('real_n', '')}+{item.get('random_n', '')}",
                "gate": "PASS_metric_validity",
                "interpretation": "metric validity control: separates real lysozymes from length-matched random UniRef50 sequences; not a steering-success estimate",
                "source": "r2_decoder_sparse_readout_audit/results/ec_metrics/ec_metric_calibration_summary_20260507.json",
            }
        )

    triad = load_json(R2_EC / "generated_metric_triad_summary_20260507.json")
    for item in triad.get("rows", []):
        source = item.get("source", "")
        for metric in [
            "pfam_lysozyme_like_hit_rate",
            "clean_exact_3_2_1_17_rate",
            "clean_3_2_1_prefix_rate",
            "foldseek_mean_top_tm",
            "foldseek_tm_ge_0p5_rate",
            "foldseek_tm_ge_0p7_rate",
        ]:
            value = item.get(metric)
            if value is None:
                continue
            rows.append(
                {
                    "track": "decoder",
                    "task": "generation_metric_triad",
                    "dataset": f"lysozyme_generated_{source}",
                    "method": "Pfam+CLEAN+Foldseek metric triad",
                    "metric": metric,
                    "value": f"{float(value):.4f}",
                    "ci95": "",
                    "n": item.get("n_sequences", item.get("n_foldseek_pdbs", "")),
                    "gate": "filter_pass" if source == "steered_leads" else "benchmark_measurement",
                    "interpretation": "selected steered leads pass the metric triad, but all-sequence steered-vs-unsteered lift must be judged by explicit tests",
                    "source": "r2_decoder_sparse_readout_audit/results/ec_metrics/generated_metric_triad_summary_20260507.json",
                }
            )
    for test_name, test in triad.get("tests", {}).items():
        rows.append(
            {
                "track": "decoder",
                "task": "generation_steering_effect",
                "dataset": "lysozyme_steered_all_vs_unsteered",
                "method": "ZymCTRL CLT steering + external metric triad",
                "metric": f"{test_name}_diff",
                "value": f"{float(test.get('diff', 0.0)):.4f}",
                "ci95": f"p={float(test.get('fisher_greater_p', 1.0)):.4f}",
                "n": f"{test.get('a_n', '')}+{test.get('b_n', '')}",
                "gate": "FAIL" if float(test.get("fisher_greater_p", 1.0)) >= 0.05 else "PASS",
                "interpretation": "generation-wide steering lift is not statistically decisive in the current lysozyme benchmark",
                "source": "r2_decoder_sparse_readout_audit/results/ec_metrics/generated_metric_triad_summary_20260507.json",
            }
        )

    steering_paths = [
        R2_STEER / "zymctrl_v2_onmanifold_direct_20260503.json",
        R2_STEER / "zymctrl_v2_purity_sweep_20260429_lipase_kinase_ca_m5_n64.json",
    ]
    for path in steering_paths:
        summary = load_json(path)
        if not summary:
            continue
        rows.append(
            {
                "track": "decoder",
                "task": "generation_steering_effect",
                "dataset": "ZymCTRL_EC_class_purity",
                "method": f"CLT direct-effect steering m={summary.get('multiplier', '')}",
                "metric": "n_significant_positive_classes",
                "value": f"{summary.get('n_classes_significant_positive', '')}/{summary.get('n_classes', '')}",
                "ci95": "",
                "n": summary.get("n_per_condition", ""),
                "gate": "FAIL",
                "interpretation": "direct-effect CLT steering does not produce significant positive EC-class shifts under the saved benchmark",
                "source": str(path.relative_to(REPO)),
            }
        )


def build_decoder_tables() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    add_decoder_discovery(rows)
    add_decoder_characterization(rows)
    add_decoder_diagnostics(rows)
    add_decoder_biological_context(rows)
    add_decoder_generation_metrics(rows)
    add_decoder_negative_gates(rows)

    datasets = [
        {
            "dataset": "balanced200_three_model_atlas",
            "n": 200,
            "status": "runnable",
            "role": "cross-model CLT triplet discovery and null control",
            "source": "r2_decoder_sparse_readout_audit/results/circuit_analysis/universal_atlas_balanced200_wide_null_control_30x_20260513.json",
        },
        {
            "dataset": "SwissProt_rich_labels",
            "n": 500,
            "status": "completed_negative",
            "role": "biological label alignment gate for universal triplets",
            "source": "r2_decoder_sparse_readout_audit/results/circuit_analysis/swissprot_triplet_annotation_20260513/summary.json",
        },
        {
            "dataset": "ZymCTRL_layer23_attention_output",
            "n": 0,
            "status": "pilot_completed",
            "role": "attention-output sparse readout pilot",
            "source": "r2_decoder_sparse_readout_audit/results/circuit_analysis/attention_output_transcoder_pilot_20260514_l23/summary.json",
        },
        {
            "dataset": "N_terminal_sink_ablation_suite",
            "n": 0,
            "status": "completed_negative",
            "role": "CLT/head/head-set causal gates for N-terminal readouts with high unnormalized received attention",
            "source": "r2_decoder_sparse_readout_audit/results/circuit_analysis/attention_sink_set_ablation_20260518/summary.json",
        },
        {
            "dataset": "N_terminal_sink_triplets",
            "n": 4,
            "status": "completed_context_positive",
            "role": "attention-sink triplet enrichment against N-terminal and edge-context controls",
            "source": "r2_decoder_sparse_readout_audit/results/circuit_analysis/attention_sink_biological_correlate_20260516/summary.json",
        },
        {
            "dataset": "balanced200_top_firing_positions",
            "n": 500,
            "status": "completed_resource_coverage",
            "role": "resource coverage audit for top-firing positions of universal triplets",
            "source": "r2_decoder_sparse_readout_audit/results/circuit_analysis/universal_primitives_balanced200_resource_annotation_20260512/resource_coverage.json",
        },
        {
            "dataset": "real_lysozyme_vs_random_uniref50",
            "n": 200,
            "status": "completed_metric_validity_control",
            "role": "external EC/structure metric calibration for decoder generation benchmarks",
            "source": "r2_decoder_sparse_readout_audit/results/ec_metrics/ec_metric_calibration_summary_20260507.json",
        },
        {
            "dataset": "lysozyme_generated_steered_unsteered",
            "n": 410,
            "status": "completed_generation_metric_triad",
            "role": "Pfam/CLEAN/Foldseek audit of steered leads, all steered generations and unsteered baseline",
            "source": "r2_decoder_sparse_readout_audit/results/ec_metrics/generated_metric_triad_summary_20260507.json",
        },
    ]

    out_dir = OUT / "decoder"
    fields = ["track", "task", "dataset", "method", "metric", "value", "ci95", "n", "gate", "interpretation", "source"]
    write_tsv(out_dir / "method_result_table.tsv", rows, fields)
    write_tsv(out_dir / "dataset_table.tsv", datasets, ["dataset", "n", "status", "role", "source"])
    return {
        "n_rows": len(rows),
        "datasets": datasets,
        "headline": "Decoder v0 has strong conservation/diagnostic tables, calibrated generation metrics and negative causal/steering gates; broader method comparisons remain next.",
    }


def build_method_registry() -> list[dict[str, Any]]:
    rows = [
        {
            "track": "encoder",
            "method_family": "scalar baseline",
            "method": "ESM-2 LLR",
            "status": "completed",
            "benchmark_role": "pathogenicity and DMS predictive baseline",
            "current_evidence": "strong baseline; not a localization or faithfulness method",
            "next_action": "retain as prediction baseline for localization/faithfulness comparisons",
            "source": "r0_shared_interpretability_framework/results/v0_20260515/encoder/method_result_table.tsv",
        },
        {
            "track": "encoder",
            "method_family": "sparse feature",
            "method": "ESM-2 SAE features",
            "status": "completed_readout_partial",
            "benchmark_role": "annotation coverage, mechanism-feature audit and SAE-family prediction gates",
            "current_evidence": "useful coverage audits but weak standalone predictive and rescue gates",
            "next_action": "add residue localization and perturbation faithfulness metrics",
            "source": "r1_encoder_interpretability_benchmark/results/annotation_alignment/expanded_summary_firing_20260503.json",
        },
        {
            "track": "encoder",
            "method_family": "external scalar baseline",
            "method": "AlphaMissense/gMVP/ESM-1v",
            "status": "completed",
            "benchmark_role": "strong pathogenicity comparators",
            "current_evidence": "AlphaMissense and gMVP beat SAE-family stacks under gene-grouped CV",
            "next_action": "keep as external baselines, not explanation methods",
            "source": "r1_encoder_interpretability_benchmark/results/variant_effect/grouped_pathogenicity_baselines_20260511.json",
        },
        {
            "track": "encoder",
            "method_family": "attribution baseline",
            "method": "gradient/IG/occlusion",
            "status": "not_started",
            "benchmark_role": "localization and faithfulness baseline",
            "current_evidence": "required by benchmark design but not yet run",
            "next_action": "implement top-k residue localization and occlusion controls on ClinVar/ProteinGym subsets",
            "source": "docs/BENCHMARK_PROGRESS_20260515_CN.md",
        },
        {
            "track": "decoder",
            "method_family": "sparse feature",
            "method": "MLP-output CLT triplets",
            "status": "completed_readout_negative_causal",
            "benchmark_role": "cross-model conservation, characterization and causal-ablation gates",
            "current_evidence": "38 conserved triplets under 30x null; causal ablations fail",
            "next_action": "treat as readout/diagnostic, not validated control handle",
            "source": "r2_decoder_sparse_readout_audit/results/circuit_analysis/universal_atlas_balanced200_wide_null_control_30x_20260513.json",
        },
        {
            "track": "decoder",
            "method_family": "sparse feature",
            "method": "attention-output sparse dictionary",
            "status": "pilot_completed",
            "benchmark_role": "attention-sink readout and causal-ablation pilot",
            "current_evidence": "strong attention/N-terminal readout; causal gate fails",
            "next_action": "scale beyond layer-23 pilot only if causal/generation metrics are redesigned",
            "source": "r2_decoder_sparse_readout_audit/results/circuit_analysis/attention_output_transcoder_pilot_20260514_l23/summary.json",
        },
        {
            "track": "decoder",
            "method_family": "intervention baseline",
            "method": "attention-head and sink-set ablation",
            "status": "completed_negative",
            "benchmark_role": "specificity and causal-control gate for N-terminal sink readouts",
            "current_evidence": "single-head, top-8 and top-32 head-set ablations fail strict gates",
            "next_action": "keep as negative controls in decoder benchmark",
            "source": "r2_decoder_sparse_readout_audit/results/circuit_analysis/attention_sink_set_ablation_20260518/summary.json",
        },
        {
            "track": "decoder",
            "method_family": "generation metric",
            "method": "Pfam/CLEAN/Foldseek/ESMFold triad",
            "status": "completed",
            "benchmark_role": "metric validity and generation-faithfulness scoring substrate",
            "current_evidence": "real-vs-random calibration passes; steered-vs-unsteered lift fails",
            "next_action": "reuse as generation-faithfulness metrics for future interventions",
            "source": "r2_decoder_sparse_readout_audit/results/ec_metrics/generated_metric_triad_summary_20260507.json",
        },
        {
            "track": "decoder",
            "method_family": "attribution baseline",
            "method": "attention rollout / raw activation probe / model diff",
            "status": "not_started",
            "benchmark_role": "broader decoder-method comparison",
            "current_evidence": "registry placeholder only",
            "next_action": "implement after finalizing Paper A vs Paper C boundary",
            "source": "docs/BENCHMARK_PROGRESS_20260515_CN.md",
        },
    ]
    fields = [
        "track",
        "method_family",
        "method",
        "status",
        "benchmark_role",
        "current_evidence",
        "next_action",
        "source",
    ]
    write_tsv(OUT / "method_registry.tsv", rows, fields)
    return rows


def write_summary_md(encoder: dict[str, Any], decoder: dict[str, Any]) -> None:
    lines = [
        "# ProteinInterpret v0 Benchmark Status",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Encoder-Only Benchmark v0",
        "",
        encoder["headline"],
        "",
        f"- Standardized result rows: {encoder['n_rows']}",
        f"- Dataset entries: {len(encoder['datasets'])}",
        "- Immediate blocker: dbNSFP GRCh38 is visible on the H200 OSS mount, but not in the local repo and the current pod lacks tabix/pysam.",
        "- IndelMissense v1.1 preflight shows 0 GRCh38 exact SNV-compatible records, so dbNSFP/REVEL should be treated as coverage gates rather than primary indel baselines.",
        "- Next compute tasks: localization/faithfulness metrics, gradient/IG/occlusion methods, CRPI metric, and exact dbNSFP/REVEL coverage check for IndelMissense v1.1.",
        "",
        "## Decoder-Only Benchmark v0",
        "",
        decoder["headline"],
        "",
        f"- Standardized result rows: {decoder['n_rows']}",
        f"- Dataset entries: {len(decoder['datasets'])}",
        "- Generation-metric calibration and steered-vs-unsteered tests are now included; current steering/generation-faithfulness gates remain negative.",
        "- Immediate next tasks: expand method comparison beyond CLT readouts and add model-diffing/checkpoint diagnostics.",
        "",
        "## Outputs",
        "",
        "- `encoder/method_result_table.tsv`",
        "- `encoder/dataset_table.tsv`",
        "- `encoder/resource_readiness.tsv`",
        "- `decoder/method_result_table.tsv`",
        "- `decoder/dataset_table.tsv`",
        "- `method_registry.tsv`",
        "- `summary.json`",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    encoder = build_encoder_tables()
    decoder = build_decoder_tables()
    method_registry = build_method_registry()
    payload = {
        "name": "ProteinInterpret-v0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "encoder": encoder,
        "decoder": decoder,
        "method_registry_rows": len(method_registry),
    }
    write_json(OUT / "summary.json", payload)
    write_summary_md(encoder, decoder)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
