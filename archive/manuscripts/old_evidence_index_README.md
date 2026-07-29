# Manuscript Evidence Index

This directory contains the current split BioCC manuscript drafts.

## Drafts

- `nature_methods_r1_variant_perturbation/`: R1, ESM-2 SAE variant
  perturbation diagnostics.
- `nature_methods_r2_circuit_diagnostics/`: R2, cross-model CLT circuit
  diagnostics for generative protein models.
- `nature_methods_initial_draft/`: archived combined scaffold and copied
  Springer Nature template source.

## R1 Evidence Map

| Claim family | Key values in draft | Evidence files |
|---|---|---|
| SAE annotation improves after saving firing positions | Known features at F1 > 0.5: L19 381, L23 301, L27 163, L31 86, L35 45 | `Research1/results/annotation_alignment/expanded_summary_firing_20260503.json`; `Research1/results/variant_effect/mechanism_feature_audit_firing_20260504.md` |
| ClinVar and cancer pathogenicity | ClinVar2000 SAE+LLR AUC 0.9143; CancerHoldout101 SAE+LLR AUC 0.9193 | `Research1/results/variant_effect/available_baseline_summary_20260507.md`; `Research1/results/variant_effect/t1a_final_available_baselines_no_primateai_20260510.md`; `Research1/results/variant_effect/grouped_pathogenicity_baselines_20260511.md` |
| Accessible scalar baselines are stronger than SAE+LLR | ClinVar2000 AlphaMissense AUC 0.9474; gMVP AUC 0.9369; ESM-1v AUC 0.9089 | `Research1/results/variant_effect/t1a_final_available_baselines_no_primateai_20260510.md`; `Research1/results/variant_effect/grouped_pathogenicity_baselines_20260511.md` |
| AlphaMissense plus SAE ensemble gate is negative | AM+SAE stack AUC 0.8542; AM+SAE z-sum AUC 0.9210, both below AlphaMissense | `Research1/results/variant_effect/alphamissense_sae_ensemble_20260511.md` |
| LOF/GOF/DN mechanism classifier does not generalize by protein | Protein-level macro-AUC about 0.516 | `Research1/results/variant_effect/mechanism_classifier_results_t0_protein_holdout_20260429.json`; `Research1/results/variant_effect/gene_level_mechanism_20260512.md` |
| ProteinGym is negative for generic DMS fitness | ESM-2 LLR mean Spearman 0.4341; sign-corrected SAE+LLR 0.4047 | `Research1/results/variant_effect/proteingym_benchmark_sae_signed_diagnostics_20260503.json` |
| IndelMissense v1 is bounded and preliminary | 6,649 binary reconstructable rows; damage-score AUC 0.7735 | `Research1/results/variant_effect/indel_records_supported_20260504_summary.json`; `Research1/results/variant_effect/indel_mechanism_predictions_20260504_summary.json`; `data/indelmissense/v1/README.md` |

## R2 Evidence Map

| Claim family | Key values in draft | Evidence files |
|---|---|---|
| CLT checkpoint quality | ProtGPT2 mean FVU 0.2013; ZymCTRL mean FVU 0.3307 | `Research2/results/checkpoint_evaluation/protgpt2_v2_quick_eval_20260420.json`; `Research2/results/checkpoint_evaluation/zymctrl_v2_quick_eval_20260420.json` |
| TopK-aware steering is negative | 0 / 8 EC classes with significant positive steering effect | `Research2/results/diagnostics/hook_sanity_20260429.json`; `Research2/results/steering_benchmark/zymctrl_v2_onmanifold_direct_20260503.json` |
| Lysozyme generated-sequence QC is bounded, not a steering win | Pfam 0.860 steered vs 0.820 unsteered; CLEAN exact 0.775 vs 0.775 | `Research2/results/ec_metrics/generated_metric_triad_summary_20260507.md`; `Research2/results/ec_metrics/ec_metric_calibration_summary_20260507.md` |
| Three-model atlas finds conserved triplets | 38 triplets at abs(r) >= 0.90; 30 at >= 0.95; 8 at >= 0.98 | `Research2/results/circuit_analysis/universal_atlas_balanced200_wide_summary_20260512.md` |
| Atlas null is near zero | 30-replicate null mean 0.067, max 1 at abs(r) >= 0.90 | `Research2/results/circuit_analysis/universal_atlas_balanced200_wide_null_control_30x_20260513.md` |
| Conserved triplets fail named biological annotation | Swiss-Prot rich-label max best-label MI 0.0045 nats, below 0.1 gate | `Research2/results/circuit_analysis/swissprot_triplet_annotation_20260513/interpretation.md` |
| Triplet basis does not replace ESM-2 embeddings | ESM-2 dominates Pfam-family, EC-top-class and secondary-structure probes | `Research2/results/circuit_analysis/triplet_basis_probes_20260513/summary.md` |
| M-1 final characterization | n_perm=2000; 37 / 38 categorized by legacy single-category summary | `Research2/results/circuit_analysis/triplet_characterization_20260515_nperm2000/summary.md` |
| M-2 synthesis supports entangled low-level signatures | 37 / 38 any significant; 21 / 38 with >=3 tests; attention subset T011/T018/T023/T025 | `Research2/results/circuit_analysis/triplet_synthesis_20260515_nperm2000/summary.md` |

## Execution Packet

The latest Opus execution status is tracked in
`../OPUS_PLAN_EXECUTION_20260515.md`.
