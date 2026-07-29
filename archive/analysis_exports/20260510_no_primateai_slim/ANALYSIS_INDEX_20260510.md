# Analysis Index: No PrimateAI-3D Pass

Date: 2026-05-10 CST

PrimateAI-3D is treated as unavailable because gated access could not be
obtained. The current analysis pass should use the accessible-baseline T1-A
table only: SAE-LR, ESM-2 LLR, SAE+LLR, AlphaMissense, gMVP, and ESM-1v.

## Start Here

- `PROJECT_STATUS.md`
- `TODO_RESULTS.md`
- `Research1/results/variant_effect/t1a_final_available_baselines_no_primateai_20260510.md`
- `Research1/results/variant_effect/available_baseline_summary_20260507.md`
- `Research1/results/variant_effect/channelopathy_concordance_20260507.md`
- `Research2/results/ec_metrics/generated_metric_triad_summary_20260507.md`
- `Research2/results/ec_metrics/ec_metric_calibration_summary_20260507.md`

## Exported Local Analysis Bundles

- Preferred slim export:
  `analysis_exports/20260510_no_primateai_slim/`
- Full remote/local export including binary intermediate artifacts:
  `analysis_exports/20260510_no_primateai/`

The slim export excludes `.pkl`, `.pt`, `.npy`, and related binary
intermediates. It keeps result summaries, result tables, JSONL outputs, and
runtime logs.

## Current Interpretation

- R1 scalar pathogenicity: AlphaMissense and gMVP outperform SAE+LLR.
- R1 mechanism: scalar external baselines are weak for LOF/GOF/DN mechanism
  separation, but SAE mechanism generalization is also weak under protein-level
  holdout.
- R1 channelopathy: completed but below target; DN variants mostly collapse to
  LOF.
- R2 steering/drug-design: current result remains no-go for a strong steering
  claim; lysozyme metric stack is calibrated, but generation-wide steering lift
  is weak.
