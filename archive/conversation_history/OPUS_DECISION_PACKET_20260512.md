# Opus Decision Packet — BioInterpretebility-CC

Date: 2026-05-12

Purpose: summarize the current R1/R2 evidence after executing the no-wet-lab
final plan, and identify which remaining TODOs require strategy-level
decisions rather than more routine execution.

## Executive Summary

The project should pause broad new experimental launches until Opus decides the
claim scope. Several gates now have clear outcomes:

- R1 AlphaMissense scalar-complementarity gate failed. SAE should not be framed
  as improving AlphaMissense pathogenicity prediction.
- R1 indel work is viable only as a bounded reconstructable protein-sequence
  benchmark for now: 6,649 records, damage AUC 0.7735. The original ~80k indel
  target requires transcript-aware reconstruction/mapping work.
- R2 CLT representation pilots are usable but narrow. CLT features are roughly
  tied with raw hidden states on lysozyme-vs-random discrimination.
- R2 cross-model conservation is mixed. The balanced-200 cohort passes the
  numerical triplet gate and a simple permutation null, but the broader
  UniRef500 pilot drops below the gate.
- R2 quality/structural diagnostic pilot is the cleanest positive diagnostic
  result so far.
- R2 steering remains negative and should stay in limitations/negative
  diagnostics.

Current H200 state: the 1-GPU pod `jiaotongdamoxing-zhk-zip-final-1gpu-0511`
is still running and reserving one H200, but no experiment process is active
after the UniRef500 run.

## R1 Evidence

### R1 scalar pathogenicity and AlphaMissense gate

Evidence:

- `Research1/results/variant_effect/grouped_pathogenicity_baselines_20260511.md`
- `Research1/results/variant_effect/alphamissense_sae_ensemble_20260511.md`

ClinVar2000, gene-grouped evaluation:

| Method | n | genes | AUC | gene-bootstrap 95% CI |
|---|---:|---:|---:|---|
| AlphaMissense | 1,972 | 1,054 | 0.9474 | [0.9368, 0.9567] |
| gMVP | 1,788 | 964 | 0.9369 | [0.9252, 0.9481] |
| ESM-1v | 1,972 | 1,054 | 0.9089 | [0.8927, 0.9234] |
| SAE-LR group-CV | 1,972 | 1,054 | 0.7559 | [0.7300, 0.7809] |
| AM+SAE stack | 1,972 | 1,054 | 0.8542 | [0.8352, 0.8719] |
| AM+SAE z-sum | 1,972 | 1,054 | 0.9210 | [0.9072, 0.9330] |

Gate outcome:

- AlphaMissense remains the strongest scalar predictor.
- SAE+AlphaMissense ensembles do not improve over AlphaMissense.
- Keep SAE as an interpretation/residual diagnostic layer, not a scalar
  pathogenicity improvement claim.

### R1 mechanism prediction

Evidence:

- `Research1/results/variant_effect/mechanism_classifier_results_t0_protein_holdout_20260429.json`
- summarized in `TODO_RESULTS.md`

Existing LOF/GOF/DN mechanism classifier:

- Variant-level CV macro-AUC:
  - SAE-only: 0.7471
  - LLR-only: 0.5054
  - SAE+LLR: 0.7488
- Protein-level CV macro-AUC:
  - SAE-only: 0.5161
  - LLR-only: 0.4642
  - SAE+LLR: 0.5164

Interpretation:

- The variant-level mechanism signal exists but does not generalize under
  protein-level holdout.
- The original robust mechanism-prediction claim should remain downgraded
  unless Opus explicitly wants another gene-level/Pfam-clan experiment.

### R1 indel benchmark

Evidence:

- `results/final_plan_readiness/final_plan_readiness_20260511.md`
- `Research1/results/variant_effect/indel_mechanism_predictions_20260504_summary.json`
- `data/indelmissense/v1/metadata.json`
- `data/indelmissense/v1/README.md`

Current supported indel scope:

- ClinVar indel TSV rows: 185,655
- Binary-label rows: 130,743
- Supported reconstructed rows, all labels: 18,897
- Binary + length-compatible supported/scored rows: 6,649
- Packaged benchmark split:
  - train: 5,356
  - validation: 653
  - test: 640
- Label counts:
  - pathogenic: 5,274
  - benign: 1,375
- Variant classes:
  - deletion: 2,530
  - insertion: 2,504
  - delins: 1,187
  - duplication: 428
- Current SAE damage-score AUC over packaged records: 0.7735

Main blockers to the ~80k target:

- missing UniProt sequence: 94,503
- frameshift requires transcript-aware reconstruction: 63,637
- unparsed HGVS: 6,811
- wildtype mismatch: 1,316

Interpretation:

- The bounded IndelMissense-v1 artefact is complete and reproducible.
- The original full-scale indel target is not merely more H200 runtime; it is a
  method/resource task.

### R1 indel competitors

Evidence:

- `results/final_plan_readiness/final_plan_readiness_20260511.md`

Readiness:

- Indel CADD hits: 0
- Indel REVEL hits: 0
- Indel SpliceAI hits: 0
- dbNSFP/gMVP local hits for indel comparison: 0 / 0

Interpretation:

- F-T1-2 should not start until compatible indel score files are staged, or
  Opus decides to drop this comparison.

## R2 Evidence

### R2 representation pilots

Evidence:

- `Research2/results/ec_metrics/clt_representation_lysozyme_probe_20260511.md`
- `Research2/results/ec_metrics/clt_representation_lysozyme_probe_protgpt2_20260511.md`
- `Research2/results/ec_metrics/clt_representation_lysozyme_probe_progen2_medium_20260511.md`

Task: real EC 3.2.1.17 lysozymes vs length-matched random UniRef50 proteins.

| Model | Raw hidden AUC | CLT AUC | Delta CLT - raw |
|---|---:|---:|---:|
| ZymCTRL | 0.9926 [0.9822, 0.9994] | 0.9934 [0.9820, 0.9996] | +0.0008 |
| ProtGPT2 | 0.9899 [0.9785, 0.9985] | 0.9932 [0.9855, 0.9985] | +0.0033 |
| ProGen2-medium | 0.9994 [0.9980, 1.0000] | 0.9982 [0.9952, 1.0000] | -0.0012 |

Interpretation:

- CLT features are usable, but not convincingly better than raw hidden states.
- This does not satisfy the final five-task representation benchmark.

### R2 five-task downstream benchmark readiness

Evidence:

- `results/final_plan_readiness/final_plan_readiness_20260511.md`

Readiness:

- EC class prediction: partial ready
- Pfam family prediction: partial ready
- CB513 secondary structure: missing
- DeepLoc subcellular localization: missing
- FireProtDB/ProTherm stability: missing

Interpretation:

- The five-task benchmark should not be launched until resources are staged or
  the task is formally reduced.

### R2 generation-quality / structural diagnostic

Evidence:

- `Research2/results/ec_metrics/quality_detection_from_existing_metrics_20260511.md`
- `Research2/results/ec_metrics/pfam_calibration_lysozyme_20260507.md`

Metric-stack quality detection:

- n=200: 100 real lysozymes + 100 random UniRef50 controls
- Logistic CV AUC: 0.9649 [0.9375, 0.9868]
- Univariate AUCs:
  - mean pLDDT: 0.8913
  - confident fraction: 0.9010
  - pTM: 0.8991
  - Foldseek top TM: 0.9571
  - sequence length: 0.5198

Pfam calibration:

- real lysozyme hit rate: 0.96
- real lysozyme-like hit rate: 0.91
- random UniRef50 hit rate: 0.45
- random lysozyme-like hit rate: 0.00

Interpretation:

- This is currently the cleanest R2 positive diagnostic result.
- It supports a generation-quality diagnostic story more strongly than
  steering or representation-superiority claims.

### R2 cross-model universal atlas

Evidence:

- `Research2/results/circuit_analysis/universal_atlas_balanced200_wide_summary_20260512.md`
- `Research2/results/circuit_analysis/universal_atlas_balanced200_wide_null_control_20260512.md`
- `Research2/results/circuit_analysis/universal_triplet_source_selectivity_20260512.md`
- `Research2/results/circuit_analysis/universal_atlas_uniref500_wide_summary_20260512.md`
- `Research2/results/circuit_analysis/universal_atlas_uniref500_wide_summary_thr0p95_20260512.md`

Balanced-200 cohort:

- 100 real lysozymes + 100 random UniRef50 controls
- top feature pairs: 300
- feature pool: 4,096
- exact three-model triplets:
  - abs(r) >= 0.90: 38
  - abs(r) >= 0.95: 30
  - abs(r) >= 0.98: 8

Permutation null on balanced-200:

| Threshold | Observed triplets | Null mean | Null max | Null values |
|---:|---:|---:|---:|---|
| 0.90 | 38 | 0.00 | 0 | [0, 0, 0] |
| 0.95 | 30 | 0.00 | 0 | [0, 0, 0] |
| 0.98 | 8 | 0.00 | 0 | [0, 0, 0] |

Source-selectivity on balanced-200:

- 38/38 triplets are weak/source-mixed for real-lysozyme vs random-UniRef50
  discrimination.
- This supports that they are not just simple lysozyme-family detectors, but
  it does not provide Pfam/GO-style biological annotation.

UniRef500 broad pilot:

- 500 UniRef50 sequences
- top feature pairs: 300
- feature pool: 4,096
- exact three-model triplets:
  - abs(r) >= 0.90: 8
  - abs(r) >= 0.95: 3

Interpretation:

- The numerical count gate is met on balanced-200 and survives a simple null.
- The signal does not generalize strongly to a broader UniRef500 sample.
- The annotation gate is not met: we do not yet have >=10 functional/Pfam/GO
  annotations for universal triplets.

### R2 steering

Evidence:

- `Research2/results/steering_benchmark/steering_negative_summary_20260511.md`
- summarized in `TODO_RESULTS.md`

Steering results:

- Full benchmark: 8 EC classes, n=200 per condition, 0/8 significant positive
  classes.
- Strength sweep: lipase +0.072 (p=0.1066), kinase +0.026 (p=0.6584),
  carbonic anhydrase -0.047 (p=0.2077), 0/3 significant positive classes.

Interpretation:

- Do not run more steering experiments under the current CLTs/protocol.
- Keep steering as a calibrated negative result.

## Remaining TODOs by Decision Category

### Needs Opus decision

1. R1 full-scale indel target:
   - Option A: invest in transcript-aware frameshift reconstruction and better
     UniProt mapping to pursue ~80k records.
   - Option B: freeze the current paper around the bounded 6,649-record
     reconstructable benchmark.

2. R1 indel competitor comparison:
   - Option A: stage CADD/REVEL/SpliceAI/dbNSFP-compatible indel scores.
   - Option B: drop the head-to-head competitor claim and position the
     artefact as a reconstructable protein-sequence benchmark.

3. R1 mechanism narrative:
   - Option A: run a new gene-level/Pfam-clan mechanism experiment anyway.
   - Option B: downgrade mechanism to interpretation/case-study only, because
     protein-level LOF/GOF/DN generalization already failed.

4. R2 cross-model atlas claim:
   - Option A: invest in a real 10k-sequence atlas plus annotation pipeline.
   - Option B: claim only partial/cohort-dependent conservation, using
     balanced-200 and UniRef500 as calibrated pilot evidence.

5. R2 five-task downstream benchmark:
   - Option A: stage CB513, DeepLoc, FireProtDB/ProTherm and run the full
     benchmark.
   - Option B: formally reduce the representation claim to EC/Pfam/resource-
     ready pilots.

6. R2 quality diagnostic scope:
   - Option A: expand quality detection from the 200-sequence calibration into
     a main result.
   - Option B: keep it as a strong pilot/supplementary diagnostic result.

7. H200 reservation:
   - Option A: keep one H200 reserved for immediate follow-up.
   - Option B: release the pod until Opus returns a decision.

### Does not need Opus decision

These are execution/bookkeeping tasks and can proceed after the strategy is
chosen:

- Update both manuscripts to match final claim scope.
- Add evidence-path references for every numeric table.
- Build a compact reproducibility/evidence index.
- Keep R2 steering as a negative diagnostic section.
- Keep R1 AlphaMissense ensemble as a negative gate.
- Package supplementary result tables from the completed pilots.

## Suggested Conservative Framing Before Opus Decides

R1:

- Main claim should be "bounded interpretable variant-effect diagnostics over
  missense plus reconstructable protein-sequence indels."
- Avoid "SAE improves AlphaMissense" and avoid robust cross-protein
  LOF/GOF/DN prediction.
- Indel statement should stay bounded to 6,649 reconstructable binary records
  unless transcript-aware reconstruction is added.

R2:

- Main claim should lean toward "sparse circuit-latent diagnostics and partial
  cross-model conservation."
- The strongest positive R2 result is quality/structural diagnostic, not
  steering.
- The atlas result is promising but currently cohort-dependent: balanced-200
  passes the count gate; UniRef500 does not.

## Current Artifacts to Review

Primary execution logs:

- `FINAL_PLAN_EXECUTION_20260511.md`
- `TODO_RESULTS.md`
- `LOG.md`
- `PROJECT_STATUS.md`

R1 key files:

- `Research1/results/variant_effect/grouped_pathogenicity_baselines_20260511.md`
- `Research1/results/variant_effect/alphamissense_sae_ensemble_20260511.md`
- `results/final_plan_readiness/final_plan_readiness_20260511.md`
- `data/indelmissense/v1/README.md`
- `data/indelmissense/v1/metadata.json`

R2 key files:

- `Research2/results/ec_metrics/quality_detection_from_existing_metrics_20260511.md`
- `Research2/results/ec_metrics/clt_representation_lysozyme_probe_20260511.md`
- `Research2/results/ec_metrics/clt_representation_lysozyme_probe_protgpt2_20260511.md`
- `Research2/results/ec_metrics/clt_representation_lysozyme_probe_progen2_medium_20260511.md`
- `Research2/results/circuit_analysis/universal_atlas_balanced200_wide_summary_20260512.md`
- `Research2/results/circuit_analysis/universal_atlas_balanced200_wide_null_control_20260512.md`
- `Research2/results/circuit_analysis/universal_triplet_source_selectivity_20260512.md`
- `Research2/results/circuit_analysis/universal_atlas_uniref500_wide_summary_20260512.md`
- `Research2/results/steering_benchmark/steering_negative_summary_20260511.md`
