# Opus 2026-05-16 Plan Execution Packet

This packet records the executed low-risk additions from
`OPUS_NEXT_20260516.md`. The experiments were run locally and on the current
1-GPU H200 hold pod `jiaotongdamoxing-zhk-zip-hold-1gpu-0513b-master-0`.

## Executive Summary

- R1-Add-1 is complete but does not support a standalone SAE indel-scorer
  headline. SAE damage alone reaches AUC 0.7735, while the protein-sequence
  baselines are stronger; however, a grouped LR combining SAE damage, ESM
  region pseudo-NLL, and cheap indel features reaches AUC 0.9108.
- R1-Add-2 is complete and negative. AM-vs-SAE disagreements do not show a
  BH-significant residue-context class where SAE systematically fixes AM.
- R1-Add-3 was not run because the required historical ClinVar VUS archives are
  not staged in the repository.
- R2-Add-1 and R2-Add-3 are complete and positive for a concrete
  attention-sink subtype: T011/T018/T023 are strong N-terminal edge /
  initiator-Met-context sinks.
- R2-Add-2 is complete in the available early-checkpoint form. Mature v2 CLTs
  recover 38 universal triplets, while early10k CLTs recover only 16, supporting
  universal-triplet count as a practical CLT checkpoint-quality diagnostic.

## R1-Add-1: Indel Protein Baselines

Script:

- `Research1/scripts/42_indel_protein_baselines.py`

Outputs:

- `Research1/results/variant_effect/indel_esm_region_scores_20260516.tsv`
- `Research1/results/variant_effect/indel_protein_baselines_20260516/feature_table.tsv`
- `Research1/results/variant_effect/indel_protein_baselines_20260516/summary.md`
- `Research1/results/variant_effect/indel_protein_baselines_20260516/summary.json`

Run details:

- The H200 ESM masked-region pseudo-NLL pass completed for all 6,649
  IndelMissense v1 records.
- Remote and local MD5 for the pulled ESM score table:
  `649154aabd62b7c38b67c46b1e30b3a2`.

Final AUCs:

| Method | AUC | 95% CI |
|---|---:|---:|
| sae_esm_cheap_grouped_lr | 0.9108 | [0.9027, 0.9185] |
| cheap_feature_grouped_lr | 0.8447 | [0.8344, 0.8545] |
| esm_region_delta_mean_nll | 0.8037 | [0.7924, 0.8148] |
| sae_damage_score | 0.7735 | [0.7616, 0.7849] |
| early_truncation_score | 0.7508 | [0.7436, 0.7580] |
| truncating_score | 0.7495 | [0.7420, 0.7570] |

Gate outcome: FAIL under Opus's stated acceptance criterion. The current SAE
damage score does not beat the stronger pure protein baselines. The useful
positive finding is that SAE features combine well with ESM and simple indel
features, but this should be framed as an interpretable benchmark plus a strong
combined baseline, not as a standalone SAE indel scorer.

## R1-Add-2: AM-vs-SAE Disagreement Typing

Script:

- `Research1/scripts/43_am_sae_disagreement_typing.py`

Outputs:

- `Research1/results/variant_effect/am_sae_disagreement_typing_20260516/typed_disagreements.tsv`
- `Research1/results/variant_effect/am_sae_disagreement_typing_20260516/context_enrichment.tsv`
- `Research1/results/variant_effect/am_sae_disagreement_typing_20260516/summary.md`
- `Research1/results/variant_effect/am_sae_disagreement_typing_20260516/summary.json`

Final result:

- Prediction rows: 1,972.
- AM/SAE opposite-direction disagreements: 473.
- ClinVar review-filtered disagreements with stars >= 2: 321.
- Review-filtered outcomes: `AM_right_SAE_wrong=283`,
  `SAE_right_AM_wrong=38`.
- Context enrichment tests: 22.
- BH-significant contexts at q < 0.05: 0.

Gate outcome: FAIL. The curated residual cases remain useful as illustrative
triage examples, but they should not be presented as a systematic context-class
advantage for SAE over AlphaMissense.

## R1-Add-3: VUS Reclassification

Status: deferred.

Reason: the repository currently lacks the historical ClinVar archive snapshots
needed to identify variants that were VUS in 2023 and reclassified in
2024-2025. This should be revisited only after the historical archives are
staged.

## R2-Add-1: Attention-Sink Subset

Script:

- `Research2/scripts/37_attention_sink_subset.py`

Outputs:

- `Research2/results/circuit_analysis/attention_sink_subset_20260516/attention_sink_subset.tsv`
- `Research2/results/circuit_analysis/attention_sink_subset_20260516/summary.md`
- `Research2/results/circuit_analysis/attention_sink_subset_20260516/summary.json`

Key finding:

- T011/T018/T023 are clear N-terminal edge attention-sink triplets.
- T011: attention r 0.921, first2/first5/Nterm20/edge5mer fractions all 1.00.
- T018: attention r 0.914, first2/first5/Nterm20/edge5mer fractions all 1.00.
- T023: attention r 0.885, first2/first5/Nterm20/edge5mer fractions all 1.00.
- T025 has significant attention association but is not an N-terminal sink; it
  is better described as a local motif / non-edge attention subtype.

## R2-Add-2: Universal Atlas Quality Diagnostic

Scripts and artifacts:

- `Research2/scripts/38_universal_atlas_quality_diagnostic.py`
- `Research2/results/circuit_analysis/cross_model_conservation_3model_balanced200_early10k_20260516.json`
- `Research2/results/circuit_analysis/universal_atlas_quality_diagnostic_20260516/summary.md`
- `Research2/results/circuit_analysis/universal_atlas_quality_diagnostic_20260516/summary.json`
- `Research2/results/circuit_analysis/universal_atlas_quality_diagnostic_20260516/atlas_quality.tsv`

Comparison:

| Atlas | Universal triplets | Mean layer CKA | Mean match abs(r) |
|---|---:|---:|---:|
| v2_reference | 38 | 0.5559 | 0.8259 |
| early10k | 16 | 0.4504 | 0.6403 |

Gate outcome: PASS for the available early-checkpoint diagnostic. The exact
old v1 ProtGPT2 checkpoint was not mounted in the current pod, so this result
should be described as "early-checkpoint quality diagnostic" rather than
"v1-vs-v2 diagnostic" unless the old v1 checkpoint is restored.

## R2-Add-3: Attention-Sink Biological Correlate

Script:

- `Research2/scripts/39_attention_sink_biological_correlate.py`

Outputs:

- `Research2/results/circuit_analysis/attention_sink_biological_correlate_20260516/biological_correlates.tsv`
- `Research2/results/circuit_analysis/attention_sink_biological_correlate_20260516/summary.md`
- `Research2/results/circuit_analysis/attention_sink_biological_correlate_20260516/summary.json`

Key finding:

- T011/T018/T023 pass strong one-sided Fisher tests for N-terminal edge
  enrichment.
- For the first-two-residue test, each of T011/T018/T023 has target fraction
  1.000 versus background fraction 0.055, approximate odds ratio 3.47e3, and
  BH q-value 2.72e-117.
- T025 does not show the same N-terminal enrichment.

Gate outcome: PASS. This supports a named primitive such as "N-terminal edge
attention-sink triplets" for R2, while avoiding a broader unsupported claim of
biological convergence.

## Recommended Framing Updates

- R1 should not claim that SAE is a stronger scalar pathogenicity or indel
  predictor. The defensible contribution is an interpretable benchmark/resource
  plus a strong combined sequence baseline.
- R1 residual cases should be framed as illustrative diagnostics, not as a
  statistically validated blind-spot taxonomy.
- R2 should promote the N-terminal attention-sink triplets to a dedicated
  Results section and can add the early-checkpoint atlas count as a concrete
  CLT quality diagnostic.
- R2 should still avoid saying that the universal triplets are broad biological
  primitives; the concrete supported primitive is narrower: N-terminal edge /
  initiator-context attention sinks.
