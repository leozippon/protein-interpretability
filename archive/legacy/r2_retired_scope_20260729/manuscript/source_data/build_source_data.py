#!/usr/bin/env python3
"""Build and verify the compact Paper A source-data package.

Safe evidence files are copied byte-for-byte. Sequence-bearing canonical JSON
files are represented by deterministic, explicitly labelled derived summaries
that retain only the numeric/configuration fields used by the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
R2 = REPO / "r2_interpretability_transfer"


@dataclass(frozen=True)
class CopySpec:
    source: Path
    destination: str
    role: str
    notes: str = ""


def repo(path: str) -> Path:
    return REPO / path


P07_FREEZE = repo(
    "r2_interpretability_transfer/results/npj_revision_20260717/"
    "p0_7_prospective_v2_freeze"
)
P07_RESULT = repo(
    "r2_interpretability_transfer/results/npj_revision_20260717/"
    "p0_7_prospective_v2_result"
)
P07_EXPECTED_HASHES = {
    P07_FREEZE / "frozen_spec.json":
        "6adb9377d732c0f126191767f1573a62850e08cea04e63e45090cee1c9829167",
    P07_FREEZE / "freeze_manifest.json":
        "c8ece12dfecf1eefdafb2436ad39220d0c3c5f11e2f7a331fb712e956f0774e3",
    P07_FREEZE / "execution_claim.json":
        "32e2e12156fb733414d3ee0ab556a6d131f979cd8240d03ce1161eee04ab6e93",
    P07_RESULT / "summary.json":
        "d952a75dc2785d305bd245c6e79b0e7fc94122c475af3ecadd08d3411962dbea",
    P07_RESULT / "run_manifest.json":
        "930ad2d4fbcbee63568cd52a9c4209b2270727c6b9b8fd446a2d0044450ef039",
}


COPIES = [
    # Supplementary Table 1: model and dictionary quality context.
    CopySpec(
        repo("r2_interpretability_transfer/results/checkpoint_evaluation/protgpt2_v2_quick_eval_20260420.json"),
        "table_s01_model_quality/protgpt2_v2_quick_eval_20260420.json",
        "Supplementary Table 1: ProtGPT2 legacy unmasked quick-evaluation metrics",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/checkpoint_evaluation/zymctrl_v2_quick_eval_20260420.json"),
        "table_s01_model_quality/zymctrl_v2_quick_eval_20260420.json",
        "Supplementary Table 1: ZymCTRL legacy unmasked quick-evaluation metrics",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/checkpoint_evaluation/rerun_evaluation.json"),
        "table_s01_model_quality/reference_rerun_evaluation.json",
        "Supplementary Table 1: reference rerun metrics including ProGen2-medium",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/swissprot_triplet_mi_reaudit_20260716/padding_mask_audit.json"),
        "table_s01_model_quality/padding_mask_audit.json",
        "Supplementary Table 1: exact quick-evaluation padding audit and scoped training diagnostics",
    ),
    CopySpec(
        Path("/Data/public/models_R2/ProtGPT2/config.json"),
        "table_s01_model_quality/model_configs/protgpt2_config.json",
        "Supplementary Table 1: local ProtGPT2 architecture configuration",
        "Reviewed for credentials; contains architecture/token-ID metadata only.",
    ),
    CopySpec(
        Path("/Data/public/models_R2/ZymCTRL/config.json"),
        "table_s01_model_quality/model_configs/zymctrl_config.json",
        "Supplementary Table 1: local ZymCTRL architecture configuration",
        "Reviewed for credentials; contains architecture/token-ID metadata only.",
    ),
    CopySpec(
        Path("/Data/public/models_R2/progen2-medium/config.json"),
        "table_s01_model_quality/model_configs/progen2_medium_config.json",
        "Supplementary Table 1: local ProGen2-medium architecture configuration",
        "Reviewed for credentials; contains architecture/tokenizer metadata only.",
    ),

    # Figure 2 and Supplementary Tables 2, 4 and 5.
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/universal_atlas_balanced200_wide_null_control_30x_20260513.json"),
        "figure_02_tables_s02_s04_s05/atlas_permutation_null_30x.json",
        "Figure 2a; Supplementary Table 2: observed atlas counts and 30 sequence-assignment null runs",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/universal_atlas_balanced200_wide_summary_20260512.json"),
        "figure_02_tables_s02_s04_s05/atlas_observed_summary.json",
        "Supplementary Table 2: observed balanced-200 atlas summary",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/universal_atlas_balanced200_wide_triplets_20260512.tsv"),
        "figure_02_tables_s02_s04_s05/atlas_observed_triplets.tsv",
        "Supplementary Table 2: matched three-model triplet definitions and correlations",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/triplet_characterization_20260515_nperm2000/summary.json"),
        "figure_02_tables_s02_s04_s05/triplet_characterization_summary.json",
        "Figure 2b; Supplementary Table 4: characterization run metadata and counts",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/triplet_characterization_20260515_nperm2000/triplet_characterization.tsv"),
        "figure_02_tables_s02_s04_s05/triplet_characterization.tsv",
        "Figure 2b; Supplementary Table 4: per-triplet test statistics and adjusted q-values",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/triplet_synthesis_20260515_nperm2000/summary.json"),
        "figure_02_tables_s02_s04_s05/triplet_synthesis_summary.json",
        "Supplementary Table 4: overlapping signature and cluster counts",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/triplet_synthesis_20260515_nperm2000/triplet_signatures.tsv"),
        "figure_02_tables_s02_s04_s05/triplet_signatures.tsv",
        "Figure 2b; Supplementary Table 4: cluster order and per-triplet signature labels",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/triplet_synthesis_20260515_nperm2000/cross_test_overlap.tsv"),
        "figure_02_tables_s02_s04_s05/cross_test_overlap.tsv",
        "Figure 2c; Supplementary Table 4: pairwise signature-set Jaccard values",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/triplet_synthesis_20260515_nperm2000/kmer_motifs.tsv"),
        "figure_02_tables_s02_s04_s05/kmer_motifs.tsv",
        "Supplementary Table 4: per-triplet k-mer motif synthesis",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/triplet_synthesis_20260515_nperm2000/positional_profiles.tsv"),
        "figure_02_tables_s02_s04_s05/positional_profiles.tsv",
        "Supplementary Table 4: per-triplet positional-profile synthesis",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/universal_atlas_quality_diagnostic_20260516/summary.json"),
        "figure_02_tables_s02_s04_s05/checkpoint_comparison_summary.json",
        "Figure 2d; Supplementary Table 5: mature-versus-10k checkpoint comparison",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/universal_atlas_quality_diagnostic_20260516/atlas_quality.tsv"),
        "figure_02_tables_s02_s04_s05/checkpoint_comparison.tsv",
        "Figure 2d; Supplementary Table 5: checkpoint comparison rows",
    ),

    # Figure 3 and Supplementary Table 6.
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/attention_sink_subset_20260516/summary.json"),
        "figure_03_table_s06/nterminal_subset_summary.json",
        "Figure 3; Supplementary Table 6: selected N-terminal subset metadata",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/attention_sink_subset_20260516/attention_sink_subset.tsv"),
        "figure_03_table_s06/nterminal_subset.tsv",
        "Figure 3b-c; Supplementary Table 6: position fractions, received-attention correlations and motifs",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/triplet_characterization_20260515_nperm2000/top_firing_positions.tsv"),
        "figure_03_table_s06/top_firing_positions.tsv",
        "Figure 3a,d-f; Supplementary Table 6: saved top-event positions and 3-mer contexts",
        "Contains accession/position/context only; no full protein sequences.",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/attention_sink_biological_correlate_20260516/summary.json"),
        "figure_03_table_s06/nterminal_context_summary.json",
        "Supplementary Table 6: N-terminal/context enrichment summary",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/attention_sink_biological_correlate_20260516/biological_correlates.tsv"),
        "figure_03_table_s06/nterminal_context_tests.tsv",
        "Supplementary Table 6: descriptive N-terminal/context test rows",
    ),

    # Figure 4 and Supplementary Tables 7 and 9. Steering is derived below.
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/zymctrl/direct_effect_features_v2_summary_20260503.json"),
        "figure_04_tables_s07_s09/direct_effect_candidate_summary.json",
        "Supplementary Table 7 audit: complete direct-effect candidate signs and values",
        "Required to audit the historical steering selector's negative-attribution fallback.",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/attention_sink_causal_ablation_20260517/summary.json"),
        "figure_04_tables_s07_s09/feature_patch_summary.json",
        "Figure 4b; Supplementary Table 9: CLT feature-patch gate summary",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/attention_sink_causal_ablation_20260517/condition_summary.tsv"),
        "figure_04_tables_s07_s09/feature_patch_condition_summary.tsv",
        "Figure 4b; Supplementary Table 9: all feature-patch target/control condition means and CIs",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/attention_head_sink_ablation_20260518/summary.json"),
        "figure_04_tables_s07_s09/single_head_ablation_summary.json",
        "Figure 4b; Supplementary Table 9: single-head ablation gate summary",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/attention_head_sink_ablation_20260518/condition_summary.tsv"),
        "figure_04_tables_s07_s09/single_head_condition_summary.tsv",
        "Figure 4b; Supplementary Table 9: all single-head target/control condition means and CIs",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/attention_sink_set_ablation_20260518/summary.json"),
        "figure_04_tables_s07_s09/top8_head_set_ablation_summary.json",
        "Figure 4b; Supplementary Table 9: top-8 head-set gate summary",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/attention_sink_set_ablation_20260518/condition_summary.tsv"),
        "figure_04_tables_s07_s09/top8_head_set_condition_summary.tsv",
        "Figure 4b; Supplementary Table 9: all top-8 target/control condition means and CIs",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/attention_sink_set_ablation_top32_20260518/summary.json"),
        "figure_04_tables_s07_s09/top32_head_set_ablation_summary.json",
        "Figure 4b; Supplementary Table 9: top-32 head-set gate summary",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/attention_sink_set_ablation_top32_20260518/condition_summary.tsv"),
        "figure_04_tables_s07_s09/top32_head_set_condition_summary.tsv",
        "Figure 4b; Supplementary Table 9: all top-32 target/control condition means and CIs",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/attention_output_transcoder_pilot_20260514_l23/summary.json"),
        "figure_04_tables_s07_s09/attention_output_pilot_summary.json",
        "Supplementary Table 9: attention-output sparse-pilot readout/causal summary",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/attention_output_transcoder_pilot_20260514_l23/ablation_summary.tsv"),
        "figure_04_tables_s07_s09/attention_output_pilot_ablation_summary.tsv",
        "Supplementary Table 9: attention-output sparse-pilot ablation conditions",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/evidence/recoverability_audit_20260605_1250/steering/oracle_steering.json"),
        "figure_04_tables_s07_s09/site_mismatched_direction_summary.json",
        "Supplementary Table 9 audit note: site-mismatched mean-direction intervention summary",
        "Included as negative provenance, not as a valid oracle-controllability result.",
    ),

    # Figure 5 and Supplementary Table 8. Sequence-bearing lead JSON is derived below.
    CopySpec(
        repo("r2_interpretability_transfer/results/drug_design/ec_lysozyme_esmfold_metrics.json"),
        "figure_05_table_s08/display_subset_esmfold_metrics.json",
        "Figure 5a-c: displayed lead identifiers, lengths, paths and pLDDT metrics",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/drug_design/ec_lysozyme_esmfold_metrics_v2_20260425_r2_v2_1gpu.json"),
        "figure_05_table_s08/steered_lead_esmfold_metrics.json",
        "Figure 5d: selected steered-lead ESMFold aggregate and per-structure metrics",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/drug_design/ec_lysozyme_unsteered_esmfold_metrics_v2_20260425_r2_v2_1gpu.json"),
        "figure_05_table_s08/unsteered_esmfold_metrics.json",
        "Figure 5d: evaluated unsteered ESMFold aggregate and per-structure metrics",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/ec_metrics/foldseek_generated_lysozyme_20260507.json"),
        "figure_05_table_s08/foldseek_generated_summary.json",
        "Figure 5d; Supplementary Table 8: Foldseek staged-subset results",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/ec_metrics/generated_metric_triad_summary_20260507.json"),
        "figure_05_table_s08/generated_metric_triad_summary.json",
        "Supplementary Table 8: generation-wide and selected-lead Pfam/CLEAN/Foldseek aggregates",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/ec_metrics/pfam_generated_lysozyme_20260504.json"),
        "figure_05_table_s08/pfam_generated_summary.json",
        "Supplementary Table 8: Pfam generated-sequence aggregate evidence",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/ec_metrics/clean_generated_lysozyme_20260507.json"),
        "figure_05_table_s08/clean_generated_summary.json",
        "Supplementary Table 8: CLEAN generated-sequence aggregate evidence",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/ec_metrics/ec_metric_calibration_summary_20260507.json"),
        "figure_05_table_s08/metric_calibration_summary.json",
        "Supplementary Table 8 context: real-versus-random metric-stack calibration",
    ),

    # Figure 6 and Supplementary Table 10.
    CopySpec(
        repo("r2_interpretability_transfer/evidence/recoverability_audit_20260605_1250/probes_v2/probe_results.json"),
        "figure_06_table_s10/probe_results_v2.json",
        "Figure 6a-b; Supplementary Table 10a: amended-v2 recoverability probe results",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/evidence/recoverability_audit_20260605_1250/decision_v2b/decision.json"),
        "figure_06_table_s10/decision_v2b.json",
        "Supplementary Table 10 context: final amended decision output",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/evidence/recoverability_audit_20260605_1250/expanded_dictionary_probe_summary_20260612.json"),
        "figure_06_table_s10/expanded_dictionary_probe_summary.json",
        "Figure 6c; Supplementary Table 10: machine-readable exploratory wider-dictionary probe summary",
    ),

    # Supplementary Table 11: prospectively frozen synthetic positive control.
    CopySpec(
        P07_FREEZE / "frozen_spec.json",
        "table_s11_synthetic_positive_control/frozen_spec.json",
        "Supplementary Table 11: immutable prospective synthetic-control specification",
    ),
    CopySpec(
        P07_FREEZE / "freeze_manifest.json",
        "table_s11_synthetic_positive_control/freeze_manifest.json",
        "Supplementary Table 11: pre-execution freeze and source-binding receipt",
    ),
    CopySpec(
        P07_FREEZE / "execution_claim.json",
        "table_s11_synthetic_positive_control/execution_claim.json",
        "Supplementary Table 11: exclusive no-retry execution claim",
    ),
    CopySpec(
        P07_RESULT / "summary.json",
        "table_s11_synthetic_positive_control/summary.json",
        "Supplementary Table 11: complete synthetic-control result summary",
    ),
    CopySpec(
        P07_RESULT / "run_manifest.json",
        "table_s11_synthetic_positive_control/run_manifest.json",
        "Supplementary Table 11: command, environment, source and artifact hashes",
    ),

    # Supplementary Table 3: canonical MI inputs, re-audit and quick basis probes.
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/swissprot_triplet_annotation_20260513/summary.json"),
        "table_s03_swissprot_basis/canonical_triplet_annotation_summary.json",
        "Supplementary Table 3: canonical top-100 annotation run metadata",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/swissprot_triplet_annotation_20260513/interpretation_gate.tsv"),
        "table_s03_swissprot_basis/canonical_interpretation_gate.tsv",
        "Supplementary Table 3 provenance: original invalid-gate rows retained for audit",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/swissprot_triplet_annotation_20260513/rich_label_mi.tsv"),
        "table_s03_swissprot_basis/canonical_rich_label_mi.tsv",
        "Supplementary Table 3: canonical plug-in MI inputs reproduced by the re-audit",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/swissprot_triplet_annotation_20260513/per_triplet_max_act_rich.tsv"),
        "table_s03_swissprot_basis/canonical_top_events_rich_labels.tsv",
        "Supplementary Table 3: saved top-100 positions and rich labels",
        "Contains accession/position/label rows only; no full protein sequences.",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/swissprot_triplet_mi_reaudit_20260716/summary.json"),
        "table_s03_swissprot_basis/mi_reaudit_summary.json",
        "Supplementary Table 3: normalized-MI and matched-null headline results",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/swissprot_triplet_mi_reaudit_20260716/mi_reanalysis.tsv"),
        "table_s03_swissprot_basis/mi_reaudit_all_tests.tsv",
        "Supplementary Table 3: all 380 normalized-MI/null/P/q result rows",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/swissprot_triplet_mi_reaudit_20260716/triplet_summary.tsv"),
        "table_s03_swissprot_basis/mi_reaudit_triplet_summary.tsv",
        "Supplementary Table 3: best global/matched result per triplet",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/swissprot_triplet_mi_reaudit_20260716/run_manifest.json"),
        "table_s03_swissprot_basis/mi_reaudit_run_manifest.json",
        "Supplementary Table 3: deterministic re-audit command and input/output hashes",
    ),
    CopySpec(
        repo("r2_interpretability_transfer/results/circuit_analysis/triplet_basis_probes_20260513/summary.json"),
        "table_s03_swissprot_basis/triplet_basis_probe_summary.json",
        "Supplementary Table 3: 38-dimensional triplet-basis versus ESM-2 quick probes",
    ),
]


DERIVED_STEERING_SOURCE = repo(
    "r2_interpretability_transfer/results/steering_benchmark/zymctrl_v2_onmanifold_direct_20260503.json"
)
DERIVED_LEADS_SOURCE = repo("r2_interpretability_transfer/results/drug_design/ec_lysozyme_leads_v2.json")
EXPANDED_MD = repo("r2_interpretability_transfer/docs/analysis/EXPANDED_DICTIONARY_ANALYSIS_20260612.md")
EXPANDED_JSON = repo(
    "r2_interpretability_transfer/evidence/recoverability_audit_20260605_1250/expanded_dictionary_probe_summary_20260612.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def display_source(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO.resolve()))
    except ValueError:
        return str(resolved)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def normalize_non_finite(value, location: str = "$") -> tuple[object, dict[str, str]]:
    """Replace non-finite floats with null and record every JSON path."""
    replacements: dict[str, str] = {}

    def walk(item, path: str):
        if isinstance(item, dict):
            return {key: walk(child, f"{path}.{key}") for key, child in item.items()}
        if isinstance(item, list):
            return [walk(child, f"{path}[{index}]") for index, child in enumerate(item)]
        if isinstance(item, float) and not math.isfinite(item):
            replacements[path] = "NaN" if math.isnan(item) else ("Infinity" if item > 0 else "-Infinity")
            return None
        return item

    return walk(value, location), replacements


def strict_json(path: Path):
    """Load RFC 8259 JSON, rejecting Python NaN/Infinity extensions."""
    def reject(token: str):
        raise ValueError(f"non-finite JSON constant {token} in {path}")

    return json.loads(path.read_text(), parse_constant=reject)


def derive_steering_summary() -> tuple[Path, str, str]:
    source = json.loads(DERIVED_STEERING_SOURCE.read_text())
    expected_top = {
        "model", "clt", "n_per_condition", "layers", "features_per_layer",
        "multiplier", "feature_source", "direct_effect_features", "per_class",
        "elapsed_s", "n_classes_significant_positive", "n_classes",
    }
    if set(source) != expected_top:
        raise AssertionError(f"unexpected steering top-level fields: {set(source) ^ expected_top}")
    allowed_per_class = {
        "prompt", "interventions", "mean_unsteered", "std_unsteered",
        "mean_steered", "std_steered", "statistics",
    }
    excluded = {"example_unsteered", "example_steered"}
    per_class = {}
    for name, row in source["per_class"].items():
        if set(row) != allowed_per_class | excluded:
            raise AssertionError(f"unexpected fields for steering class {name}: {set(row)}")
        per_class[name] = {key: row[key] for key in row if key in allowed_per_class}
    output = {
        "schema_version": 1,
        "derivation": "Whitelist numeric/config fields; omit generated amino-acid examples.",
        "canonical_source": display_source(DERIVED_STEERING_SOURCE),
        "canonical_source_sha256": sha256(DERIVED_STEERING_SOURCE),
        "excluded_fields_per_class": sorted(excluded),
        **{key: source[key] for key in source if key != "per_class"},
        "per_class": per_class,
    }
    destination = PACKAGE / "figure_04_tables_s07_s09/steering_numeric_summary_no_sequences.json"
    write_json(destination, output)
    return destination, "example_unsteered; example_steered", output["derivation"]


def derive_generation_counts() -> tuple[Path, str, str]:
    source = json.loads(DERIVED_LEADS_SOURCE.read_text())
    required = {"n_generated", "top_k", "leads", "all_records", "unsteered_baseline"}
    if not required.issubset(source):
        raise AssertionError(f"lead-generation JSON missing fields: {required - set(source)}")
    retained_source_fields = {"n_generated", "top_k"}
    excluded_fields = sorted(set(source) - retained_source_fields)
    sequence_bearing_fields = ["leads", "all_records", "unsteered_baseline"]
    output = {
        "schema_version": 1,
        "derivation": "Retain only counts used for Figure 5 labels; omit every sequence-bearing record.",
        "canonical_source": display_source(DERIVED_LEADS_SOURCE),
        "canonical_source_sha256": sha256(DERIVED_LEADS_SOURCE),
        "n_generated": int(source["n_generated"]),
        "n_selected_leads": len(source["leads"]),
        "n_all_records": len(source["all_records"]),
        "n_unsteered_baseline": len(source["unsteered_baseline"]),
        "top_k": int(source["top_k"]),
        "excluded_top_level_fields": excluded_fields,
        "sequence_bearing_top_level_fields": sequence_bearing_fields,
    }
    destination = PACKAGE / "figure_05_table_s08/generation_counts_no_sequences.json"
    write_json(destination, output)
    return destination, "; ".join(excluded_fields), output["derivation"]


def derive_expanded_quality() -> tuple[Path, str, str]:
    text = EXPANDED_MD.read_text()
    pattern = re.compile(
        r"^\| (ProtGPT2|ZymCTRL|ProGen2-medium) \| (step_\d+) \| "
        r"([\d.]+) \| ([\d.]+) \| ([\d.]+)% \| ([\d.]+) \| ([\d.]+) \|$",
        re.MULTILINE,
    )
    matches = pattern.findall(text)
    if len(matches) != 3:
        raise AssertionError(f"expected three expanded-quality rows, found {len(matches)}")
    expanded = json.loads(EXPANDED_JSON.read_text())
    width = int(expanded["dictionary"]["width"])
    base_width = {"ProtGPT2": 8192, "ZymCTRL": 8192, "ProGen2-medium": 4096}
    rows = []
    for model, checkpoint, fvu, dead, dead_pct, l0, loss in matches:
        fvu_value = float(fvu)
        dead_value = float(dead)
        rows.append(
            {
                "model": model,
                "checkpoint": checkpoint,
                "d_clt": width,
                "width_ratio": width / base_width[model],
                "final_training_log_fvu": fvu_value,
                "dead_fraction": dead_value,
                "dead_percent": float(dead_pct),
                "l0": float(l0),
                "loss": float(loss),
                "fvu_gate_lt_0p15": fvu_value < 0.15,
                "dead_gate_lt_0p30": dead_value < 0.30,
                "joint_quality_gate": fvu_value < 0.15 and dead_value < 0.30,
            }
        )
    destination = PACKAGE / "figure_06_table_s10/expanded_training_quality.tsv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    derivation = (
        "Parse the three final-training-quality rows from the canonical expanded-analysis Markdown; "
        "join dictionary width from the machine-readable expanded probe summary; evaluate frozen gates."
    )
    return destination, "none (numeric table extraction)", derivation


def derive_p07_positive_control_table() -> tuple[Path, str, str]:
    """Extract only the per-seed values reported in Supplementary Table 11."""
    summary_path = P07_RESULT / "summary.json"
    summary = strict_json(summary_path)
    if (
        summary.get("schema_version")
        != "r2_p0_7_prospective_positive_control_result_v2"
        or summary.get("status") != "prospective_synthetic_gate_passed"
        or summary.get("claim_scope")
        != "synthetic_pipeline_sensitivity_only_no_pretrained_causal_inference"
        or summary.get("pretrained_model_causal_inference") is not False
        or summary.get("legacy_controls_upgraded") is not False
        or summary.get("split_counts")
        != {"assessment": 48, "discovery": 48, "train": 48}
    ):
        raise AssertionError("unexpected P0-7 production result contract")
    models = summary.get("models")
    if not isinstance(models, list) or [row.get("model_seed") for row in models] != [11, 29, 47]:
        raise AssertionError("P0-7 production result must contain seeds 11, 29 and 47")

    rows = []
    for model in models:
        doses = model.get("dose_sweep")
        if not isinstance(doses, list) or [row.get("dose") for row in doses] != [0.5, 1.0, 2.0]:
            raise AssertionError("unexpected P0-7 dose inventory")
        if not all(row.get("recovery_inside_frozen_interval") is True for row in doses):
            raise AssertionError("P0-7 dose recovery gate is not complete")
        fvu = model.get("clt_training", {}).get("fvu_per_layer")
        if not isinstance(fvu, list) or len(fvu) != 2:
            raise AssertionError("unexpected P0-7 FVU inventory")
        if (
            model.get("sensitivity") != 1.0
            or model.get("specificity") != 1.0
            or model.get("false_discovery_rate") != 0.0
            or model.get("path_localized") is not True
            or model.get("negative_equivalence_passed") is not True
        ):
            raise AssertionError("P0-7 per-seed gate differs from the reported pass")
        rows.append(
            {
                "model_seed": model["model_seed"],
                "assessment_n": summary["split_counts"]["assessment"],
                "sensitivity": model["sensitivity"],
                "specificity": model["specificity"],
                "false_discovery_rate": model["false_discovery_rate"],
                "endpoint_accuracy": model["lm_training"]["assessment_endpoint_accuracy"],
                "mean_clt_fvu": model["clt_training"]["fvu_mean"],
                "layer_1_clt_fvu": fvu[1],
                "selected_ground_truth_cosine": model["selected_ground_truth_cosine"],
                "dose_0p5_effect_recovery_ratio": doses[0]["effect_recovery_ratio"],
                "dose_1_effect_recovery_ratio": doses[1]["effect_recovery_ratio"],
                "dose_2_effect_recovery_ratio": doses[2]["effect_recovery_ratio"],
                "path_localized": model["path_localized"],
                "all_negative_equivalence_passed": model["negative_equivalence_passed"],
            }
        )

    destination = PACKAGE / "table_s11_synthetic_positive_control/per_seed_results.tsv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    derivation = (
        "Deterministic extraction of the three immutable production-seed rows; "
        "the complete summary retains dose, matched-control, wrong-site and alignment results."
    )
    return destination, "none (numeric table extraction)", derivation


AA_SEQUENCE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYX.\-*\s]+$")


def sequence_like_values(path: Path) -> list[tuple[str, int]]:
    hits: list[tuple[str, int]] = []
    if path.suffix == ".json":
        value = strict_json(path)

        def walk(item, location: str = "") -> None:
            if isinstance(item, dict):
                for key, child in item.items():
                    walk(child, f"{location}.{key}" if location else key)
            elif isinstance(item, list):
                for index, child in enumerate(item):
                    walk(child, f"{location}[{index}]")
            elif isinstance(item, str):
                compact = re.sub(r"[.\-*\s]", "", item)
                if len(compact) >= 40 and AA_SEQUENCE.fullmatch(item.upper()):
                    hits.append((location, len(compact)))

        walk(value)
    elif path.suffix in {".tsv", ".csv"}:
        delimiter = "\t" if path.suffix == ".tsv" else ","
        with path.open() as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            for row_index, row in enumerate(reader):
                for column_index, value in enumerate(row):
                    compact = re.sub(r"[.\-*\s]", "", value)
                    if len(compact) >= 40 and AA_SEQUENCE.fullmatch(value.upper()):
                        hits.append((f"row{row_index + 1}:col{column_index + 1}", len(compact)))
    return hits


SECRET_PATTERNS = [
    re.compile(r"api[_-]?key\s*[:=]", re.IGNORECASE),
    re.compile(r"access[_-]?token\s*[:=]", re.IGNORECASE),
    re.compile(r"password\s*[:=]", re.IGNORECASE),
    re.compile(r"authorization\s*[:=]", re.IGNORECASE),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
]


def assert_no_secrets(path: Path) -> None:
    text = path.read_text(errors="ignore")
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            raise AssertionError(f"possible credential in {path}: {pattern.pattern}")


def manifest_row(
    destination: Path,
    sources: list[Path],
    role: str,
    copy_type: str,
    excluded_fields: str = "",
    notes: str = "",
) -> dict[str, str | int]:
    return {
        "package_path": str(destination.relative_to(PACKAGE)),
        "source_path": " | ".join(display_source(path) for path in sources),
        "role": role,
        "copy_type": copy_type,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "source_sha256": " | ".join(sha256(path) for path in sources),
        "excluded_fields": excluded_fields,
        "notes": notes,
    }


def build() -> list[dict]:
    if not (PACKAGE / "README.md").is_file():
        raise FileNotFoundError(PACKAGE / "README.md")
    for path, expected_hash in P07_EXPECTED_HASHES.items():
        if not path.is_file() or sha256(path) != expected_hash:
            raise AssertionError(f"immutable P0-7 artifact mismatch: {path}")
    rows: list[dict] = []
    expected_files = {"README.md", "build_source_data.py", "MANIFEST.tsv", "SHA256SUMS"}

    for spec in COPIES:
        if not spec.source.is_file():
            raise FileNotFoundError(spec.source)
        destination = PACKAGE / spec.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy_type = "exact_copy"
        notes = spec.notes
        if spec.source.suffix == ".json":
            value = json.loads(spec.source.read_text())
            normalized, replacements = normalize_non_finite(value)
        else:
            normalized, replacements = None, {}
        if replacements:
            if not isinstance(normalized, dict):
                raise AssertionError(f"cannot annotate non-object JSON: {spec.source}")
            normalized["_non_finite_normalization"] = {
                "status": "historical_non_finite_values_replaced_with_null",
                "source_sha256": sha256(spec.source),
                "fields": replacements,
            }
            write_json(destination, normalized)
            copy_type = "normalized_json"
            detail = f"Replaced {len(replacements)} non-finite value(s) with null; paths recorded in _non_finite_normalization."
            notes = f"{notes} {detail}".strip()
        else:
            shutil.copy2(spec.source, destination)
            if sha256(spec.source) != sha256(destination):
                raise AssertionError(f"copy hash mismatch: {spec.source} -> {destination}")
        rows.append(
            manifest_row(destination, [spec.source], spec.role, copy_type, notes=notes)
        )
        expected_files.add(spec.destination)

    steering_path, steering_excluded, steering_notes = derive_steering_summary()
    rows.append(
        manifest_row(
            steering_path,
            [DERIVED_STEERING_SOURCE],
            "Figure 4a; Supplementary Table 7: sequence-free per-class steering numeric/config summary",
            "derived_sequence_free",
            steering_excluded,
            steering_notes,
        )
    )
    expected_files.add(str(steering_path.relative_to(PACKAGE)))

    counts_path, counts_excluded, counts_notes = derive_generation_counts()
    rows.append(
        manifest_row(
            counts_path,
            [DERIVED_LEADS_SOURCE],
            "Figure 5d: generated/selected/unsteered denominators without sequence records",
            "derived_sequence_free",
            counts_excluded,
            counts_notes,
        )
    )
    expected_files.add(str(counts_path.relative_to(PACKAGE)))

    quality_path, quality_excluded, quality_notes = derive_expanded_quality()
    rows.append(
        manifest_row(
            quality_path,
            [EXPANDED_MD, EXPANDED_JSON],
            "Supplementary Table 10b: exploratory wider-checkpoint final training quality and frozen gates",
            "derived_numeric",
            quality_excluded,
            quality_notes,
        )
    )
    expected_files.add(str(quality_path.relative_to(PACKAGE)))

    p07_path, p07_excluded, p07_notes = derive_p07_positive_control_table()
    rows.append(
        manifest_row(
            p07_path,
            [P07_RESULT / "summary.json"],
            "Supplementary Table 11: per-seed prospective synthetic-control outcomes",
            "derived_numeric",
            p07_excluded,
            p07_notes,
        )
    )
    expected_files.add(str(p07_path.relative_to(PACKAGE)))

    rows.sort(key=lambda row: row["package_path"])
    manifest_path = PACKAGE / "MANIFEST.tsv"
    fields = [
        "package_path", "source_path", "role", "copy_type", "bytes", "sha256",
        "source_sha256", "excluded_fields", "notes",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        destination = PACKAGE / row["package_path"]
        if destination.stat().st_size != int(row["bytes"]):
            raise AssertionError(f"manifest byte mismatch: {destination}")
        if sha256(destination) != row["sha256"]:
            raise AssertionError(f"manifest hash mismatch: {destination}")
        hits = sequence_like_values(destination)
        if hits:
            raise AssertionError(f"sequence-like values in {destination}: {hits[:5]}")
        if "model_configs/" in row["package_path"]:
            assert_no_secrets(destination)
        if row["copy_type"] == "exact_copy":
            source = Path(row["source_path"])
            if not source.is_absolute():
                source = REPO / source
            if sha256(source) != row["sha256"]:
                raise AssertionError(f"exact-copy source mismatch: {source}")

    checksum_path = PACKAGE / "SHA256SUMS"
    checksum_targets = sorted(
        path for path in PACKAGE.rglob("*")
        if path.is_file() and path != checksum_path and "__pycache__" not in path.parts
    )
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(PACKAGE)}\n" for path in checksum_targets)
    )

    actual_files = {
        str(path.relative_to(PACKAGE))
        for path in PACKAGE.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    unexpected = actual_files - expected_files
    missing = expected_files - actual_files
    if unexpected or missing:
        raise AssertionError(f"package file-set mismatch; unexpected={unexpected}, missing={missing}")
    return rows


def verify() -> tuple[int, int]:
    manifest_path = PACKAGE / "MANIFEST.tsv"
    checksum_path = PACKAGE / "SHA256SUMS"
    if not manifest_path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError("MANIFEST.tsv and SHA256SUMS must exist")
    with manifest_path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        destination = PACKAGE / row["package_path"]
        if not destination.is_file():
            raise FileNotFoundError(destination)
        if destination.stat().st_size != int(row["bytes"]):
            raise AssertionError(f"manifest byte mismatch: {destination}")
        if sha256(destination) != row["sha256"]:
            raise AssertionError(f"manifest hash mismatch: {destination}")
        sources = row["source_path"].split(" | ")
        source_hashes = row["source_sha256"].split(" | ")
        if len(sources) != len(source_hashes):
            raise AssertionError(f"source/hash arity mismatch: {row['package_path']}")
        for source_text, expected_hash in zip(sources, source_hashes):
            source = Path(source_text)
            if not source.is_absolute():
                source = REPO / source
            if not source.is_file() or sha256(source) != expected_hash:
                raise AssertionError(f"source hash mismatch: {source}")
        if row["copy_type"] == "exact_copy" and row["sha256"] != source_hashes[0]:
            raise AssertionError(f"non-identical exact copy: {row['package_path']}")
        hits = sequence_like_values(destination)
        if hits:
            raise AssertionError(f"sequence-like values in {destination}: {hits[:5]}")

    checksum_rows = []
    for line in checksum_path.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        target = PACKAGE / relative
        if not target.is_file() or sha256(target) != expected:
            raise AssertionError(f"SHA256SUMS mismatch: {relative}")
        checksum_rows.append(relative)
    expected_manifest_files = {row["package_path"] for row in rows}
    expected_package_files = expected_manifest_files | {
        "README.md", "build_source_data.py", "MANIFEST.tsv", "SHA256SUMS",
    }
    actual_package_files = {
        str(path.relative_to(PACKAGE))
        for path in PACKAGE.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if actual_package_files != expected_package_files:
        raise AssertionError(
            "package file-set mismatch; "
            f"unexpected={actual_package_files - expected_package_files}, "
            f"missing={expected_package_files - actual_package_files}"
        )
    expected_checksum_rows = expected_package_files - {"SHA256SUMS"}
    if set(checksum_rows) != expected_checksum_rows or len(checksum_rows) != len(set(checksum_rows)):
        raise AssertionError(
            "SHA256SUMS coverage mismatch; "
            f"unexpected={set(checksum_rows) - expected_checksum_rows}, "
            f"missing={expected_checksum_rows - set(checksum_rows)}"
        )
    return len(rows), len(checksum_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        rows = build()
        print(f"built {len(rows)} evidence files in {PACKAGE}")
    manifest_count, checksum_count = verify()
    total_bytes = sum(path.stat().st_size for path in PACKAGE.rglob("*") if path.is_file())
    print(
        f"verified {manifest_count} manifest rows and {checksum_count} checksums; "
        f"package bytes={total_bytes}"
    )


if __name__ == "__main__":
    main()
