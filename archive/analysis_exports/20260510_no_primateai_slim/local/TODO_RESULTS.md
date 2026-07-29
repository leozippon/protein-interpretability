# TODO Results Snapshot

Last updated: 2026-05-10 CST

This file records the completed results for all items in `TODO.md`. It is
intended as the local English reference for paper planning and follow-up
experiments. Large checkpoints remain on the H200 storage; required JSON/TSV
result artifacts have been copied back into this local repository.

## Returned Remote Artifacts

The following remote outputs were copied back locally:

- `Research1/results/variant_effect/proteingym_benchmark_sae_latest.json`
- `Research1/results/variant_effect/proteingym_benchmark_sae_diagnostics_20260429.json`
- `Research1/results/variant_effect/proteingym_benchmark_sae_signed_20260429_signed.json`
- `Research1/results/variant_effect/proteingym_benchmark_sae_signed_diagnostics_20260503.json`
- `Research2/results/checkpoint_evaluation/protgpt2_v2_quick_eval_20260420.json`
- `Research2/results/checkpoint_evaluation/zymctrl_v2_quick_eval_20260420.json`
- `Research2/results/steered_generation/zymctrl_v2_lysozyme_smoketest.json`
- `Research2/results/r2_v2_remaining_summary_20260425_r2_v2_1gpu.json`
- `Research2/results/steering_benchmark/zymctrl_v2_purity_20260425_r2_v2_1gpu.json`
- `Research2/results/steering_benchmark/zymctrl_v2_purity_sweep_20260429_lipase_kinase_ca_m5_n64.json`
- `Research2/results/drug_design/ec_lysozyme_leads_v2.json`
- `Research2/results/drug_design/ec_lysozyme_esmfold_metrics_v2_20260425_r2_v2_1gpu.json`
- `Research2/results/drug_design/ec_lysozyme_unsteered_esmfold_metrics_v2_20260425_r2_v2_1gpu.json`
- `Research2/results/causal_ablation/zymctrl_v2_lysozyme_L12_F8088_20260425_r2_v2_1gpu.json`
- `Research2/results/circuit_analysis/cross_model_conservation_v2_20260429_corrfix.json`
- `Research2/results/circuit_analysis/zymctrl/direct_effect_features_v2.pkl`
- `Research2/results/circuit_analysis/zymctrl/direct_effect_features_v2_summary_20260503.json`
- `Research1/results/variant_effect/mechanism_feature_audit_20260503.json`
- `Research1/results/variant_effect/mechanism_feature_audit_20260503.md`

Checkpoint artifacts were not copied back because they are very large. Their
canonical H200 paths are:

- ProtGPT2 v2 CLT: `/oss-pvc/zhk_zip/outputs/research2/clt_weights/protgpt2_v2/step_200000`
- ZymCTRL v2 CLT: `/oss-pvc/zhk_zip/outputs/research2/clt_weights/zymctrl_v2/step_200000`

## R1 - Variant Mechanism Prediction

### R1-A: Mechanism Labels Dataset

Status: completed.

Artifact:

- `Research1/results/variant_effect/variant_mechanisms.tsv`
- `Research1/results/variant_effect/variant_mechanisms_summary.json`

Result:

- Total ClinVar variants processed: 2,000.
- Mechanism-labeled variants: 497.
- Class counts: LOF 297, GOF 120, DN 80, unlabeled 1,503.
- Label sources: Badonyi gene-level 408, per-variant 78, gene fallback 11.

Interpretation:

The mechanism-label dataset is large enough for a first LOF/GOF/DN classifier,
but still sparse relative to the full 2,000-variant ClinVar set.

### R1-B: Mechanism Classifier

Status: completed, but downgraded after the protein-level holdout audit.

Artifact:

- `Research1/results/variant_effect/mechanism_classifier_results.json`
- `Research1/results/variant_effect/mechanism_classifier_results_t0_protein_holdout_20260429.json`

Result:

- Classes: DN, GOF, LOF.
- Per-class sample counts: DN 80, GOF 120, LOF 297.
- Variant-level CV:
  - SAE-only macro-AUC: 0.7471.
  - LLR-only macro-AUC: 0.5054.
  - SAE+LLR macro-AUC: 0.7488.
- Protein-level CV:
  - SAE-only macro-AUC: 0.5161.
  - LLR-only macro-AUC: 0.4642.
  - SAE+LLR macro-AUC: 0.5164.

Interpretation:

The earlier variant-level split leaked protein-specific signatures. Under the
protein-level split, the mechanism classifier does not generalize cleanly across
proteins and fails the TODO_NEXT threshold of macro-AUC >= 0.7. The R1
mechanism claim should be downgraded to "SAE features contain variant-level
mechanism structure" rather than "SAE robustly predicts LOF/GOF/DN across
proteins."

### R1 T1-A: Baseline Head-to-Head

Status: historical readiness audit. This early blocker was superseded by the
available-baseline pass below; PrimateAI-3D is now treated as unavailable and
T1-A is finalized over accessible baselines.

Artifacts:

- `Research1/scripts/23_baseline_headtohead.py`
- `Research1/results/variant_effect/baseline_headtohead_readiness_20260504.json`
- `Research1/results/variant_effect/baseline_headtohead_readiness_20260504.md`

Available result:

- ClinVar 2,000 pathogenicity AUC:
  - SAE-LR: 0.8782, 95% CI [0.8619, 0.8923].
  - ESM-2 LLR: 0.8822, 95% CI [0.8677, 0.8960].
  - SAE+LLR: 0.9143, 95% CI [0.9010, 0.9259].
- Cancer holdout 101 pathogenicity AUC:
  - SAE-LR: 0.9079, 95% CI [0.8471, 0.9590].
  - ESM-2 LLR: 0.8978, 95% CI [0.8273, 0.9561].
  - SAE+LLR: 0.9193, 95% CI [0.8617, 0.9646].
- Protein-level mechanism macro-AUC remains low:
  - SAE-LR: 0.5161.
  - ESM-2 LLR: 0.4642.
  - SAE+LLR: 0.5164.

Missing assets:

- AlphaMissense.
- PrimateAI-3D.
- gMVP.
- ESM-1v.

Interpretation:

This section records the initial readiness state. The final current T1-A
interpretation is in the 2026-05-07/2026-05-10 available-baseline sections:
AlphaMissense, gMVP, and ESM-1v are scored; PrimateAI-3D is excluded because
the gated dataset could not be obtained.

### R1-C: Per-Mechanism Feature Interpretation

Status: completed as a coefficient-to-annotation audit with firing-position
examples; manual biological validation is still a separate analyst task.

Artifact:

- `Research1/results/variant_effect/mechanism_classifier_results.json`
- `Research1/results/variant_effect/mechanism_feature_audit_20260503.json`
- `Research1/results/variant_effect/mechanism_feature_audit_20260503.md`
- `Research1/results/variant_effect/mechanism_feature_audit_firing_20260504.json`
- `Research1/results/variant_effect/mechanism_feature_audit_firing_20260504.md`

Result:

- `top_features_per_class` is populated for DN, GOF, and LOF.
- The audit expands this to 150 rows: 3 classes x 5 production layers x top-10
  positive coefficient features.
- Each row includes coefficient, feature kind, best annotation, Pfam/domain
  proxy, GO/functional proxy, binding proxy, and a conservative interpretation
  note.
- The firing-position audit has 150/150 rows with top residue-level firing
  examples and 142/150 rows with expanded labels.

Interpretation:

The table is now usable for figure planning and manual residue triage. It is
still not direct biological validation: the firing examples identify residues
to inspect, but the mechanism claims should remain conservative unless an
analyst confirms the biology feature by feature.

### R1-D: ProteinGym / MAVE Benchmark

Status: completed, but the SAE benchmark result is negative.

Artifacts:

- `Research1/results/variant_effect/proteingym_benchmark.json`
- `Research1/results/variant_effect/proteingym_benchmark_sae_latest.json`
- `Research1/results/variant_effect/proteingym_benchmark_sae_diagnostics_20260429.json`
- `Research1/results/variant_effect/proteingym_benchmark_sae_signed_20260429_signed.json`
- `Research1/results/variant_effect/proteingym_benchmark_sae_signed_diagnostics_20260503.json`

Result:

- ProteinGym assays scored: 217/217.
- Assays with finite SAE result: 214.
- Mean Spearman rho:
  - LLR: 0.4341.
  - SAE: -0.2314.
  - SAE+LLR z-sum ensemble: 0.1993.
  - Negated SAE diagnostic: 0.2314.
  - Sign-corrected SAE+LLR ensemble: 0.4047
    (95% bootstrap CI [0.3818, 0.4282]).
- Ensemble-minus-LLR mean delta: -0.2348, bootstrap 95% CI [-0.2593, -0.2112].
- Ensemble beats LLR on 8.4% of usable assays.
- Sign-corrected ensemble beats LLR on 33.6% of usable assays.
- SAE beats LLR on 1.4% of usable assays.
- Negated SAE beats LLR on 15.9% of usable assays.

Interpretation:

The sign correction fixes most of the cancellation in the legacy plus ensemble,
but it still does not beat LLR on average and its win rate remains below the
TODO_NEXT 40% threshold. The classifier-based ProteinGym scorer is not used as a
paper claim because the T0-D protein-level mechanism classifier failed. ProteinGym
should be framed as a negative diagnostic: the current SAE perturbation score
tracks mechanism-like disruption rather than generic DMS fitness.

### R1-E: Held-Out Cancer Validation

Status: completed.

Artifact:

- `Research1/results/variant_effect/cancer_holdout.json`

Result:

- Train set: 1,899 variants.
- Cancer holdout test set: 101 variants.
- Overall AUC:
  - SAE: 0.9079.
  - LLR: 0.8978.
  - SAE+LLR: 0.9193.
- SAE+LLR accuracy: 0.8416.
- Oncogene subset: SAE+LLR AUC 0.9464.
- Tumor-suppressor subset: SAE+LLR AUC 0.9167.

Interpretation:

This supports the R1 generalization story. The sample size is modest, but the
SAE+LLR ensemble is stronger than either component on the cancer holdout.

### R1-F: Deep-Layer Annotation Expansion

Status: completed after firing-position rerun; improved but below the strict
L35 acceptance threshold.

Artifact:

- `Research1/results/annotation_alignment/expanded_summary.json`
- `Research1/results/annotation_alignment/expanded_summary_firing_20260503.json`
- `Research1/scripts/run_t1d_annotation_firing_20260503.sh`

Result:

- `04_analyze_our_sae.py` now supports saving firing positions.
- `feature_annotation.py` now stores `firing_positions` and
  `top_firing_examples`.
- A L35 smoke test confirmed the new pkl fields are populated.
- The full T1-D rerun completed on H200 with layers 19/23/27/31/35 and 1000
  Swiss-Prot proteins.
- KNOWN feature counts after expansion:
  - L19: 146 -> 381.
  - L23: 135 -> 301.
  - L27: 72 -> 163.
  - L31: 49 -> 86.
  - L35: 32 -> 45.
- L35 improved but remains below the TODO_NEXT acceptance target of >=60 KNOWN
  features.

Interpretation:

The firing-position rerun made the expansion scientifically useful and enabled
T1-B final feature audit. It is still below the strict deep-layer acceptance
target at L35, so this should be reported as an improvement with a limitation,
not as a fully solved annotation-depth result.

### R1-G: Indel / Frameshift Extension

Status: completed as a staging/coverage analysis.

Artifacts:

- `Research1/results/variant_effect/clinvar_indels.tsv`
- `Research1/results/variant_effect/indel_extension_summary.json`

Result:

- Rows processed in the staged summary: 185,655.
- Label counts: pathogenic 125,216, benign 5,527, other 54,912.
- Supported classes:
  - delins: 2,453.
  - deletion: 9,849.
  - insertion: 5,082.
  - duplication: 1,513.
- Major unsupported class: frameshift, which requires transcript-level context.

Interpretation:

The indel extension has enough coverage for a non-missense scope claim, but it
is currently a staged capability analysis rather than a final predictive
benchmark. Frameshifts remain outside the current protein-sequence-only scorer.

### R1 T1-C: Indel / Frameshift Mechanism Transfer

Status: completed.

Artifacts:

- `Research1/scripts/25_indel_mechanism.py`
- `Research1/results/variant_effect/indel_records_supported_20260504.jsonl`
- `Research1/results/variant_effect/indel_records_supported_20260504_summary.json`
- `Research1/results/variant_effect/indel_mechanism_classifier_20260504.pkl`
- Remote log:
  `/oss-pvc/zhk_zip/biocc/Research1/logs/runtime/t1c_indel_mechanism_20260504.log`
- Remote output:
  `/oss-pvc/zhk_zip/biocc/Research1/results/variant_effect/indel_mechanism_predictions_20260504.jsonl`
- Local output:
  `Research1/results/variant_effect/indel_mechanism_predictions_20260504.jsonl`
- Local summary:
  `Research1/results/variant_effect/indel_mechanism_predictions_20260504_summary.json`
- Local log:
  `Research1/logs/runtime/t1c_indel_mechanism_20260504.log`

Result:

- Prepared 6,649 reconstructable pathogenic/benign ClinVar indel records:
  5,274 pathogenic and 1,375 benign.
- Variant classes: deletion 2,530; insertion 2,504; delins 1,187;
  duplication 428.
- Fitted a missense LOF/GOF/DN classifier cache with 497 training variants and
  45,891 non-constant SAE feature columns.
- H200 smoke test on 2 records passed.
- Full H200 run completed for 6,649/6,649 records.
- Predicted mechanisms: LOF 4,161; GOF 1,354; DN 1,134.
- SAE damage-score pathogenicity AUC: 0.7735.

Interpretation:

This is a transfer diagnostic, not yet a validated indel mechanism claim. The
local indel feature is an affected-region SAE delta heuristic, so final
interpretation should remain conservative. The 0.7735 damage AUC is useful but
below the TODO_NEXT acceptance target of >0.85.

### R1 T1-E: Channelopathy Clinical Cohort

Status: blocked by missing curated mechanism/drug-response labels.

Artifacts:

- `Research1/scripts/26_channelopathy.py`
- `Research1/results/variant_effect/channelopathy_readiness_20260504.json`
- `Research1/results/variant_effect/channelopathy_readiness_20260504.md`

Readiness result:

- Channelopathy target genes checked: KCNQ1, SCN5A, KCNH2, CACNA1C.
- Current ClinVar LLR subset contains 16 KCNQ1 rows.
- Existing R1 variant prediction table contains 0 rows for the four target
  genes.
- Existing mechanism labels contain 0 curated channelopathy mechanism rows.
- Local DMS assets exist for SCN5A and KCNH2 ProteinGym, but these are not
  clinical mechanism/drug-response labels.

Interpretation:

T1-E cannot be accepted from local data. It needs a staged ClinGen/literature
mechanism-curated channelopathy cohort, and ideally retrospective
drug-response labels, before concordance can be measured.

## R2 - Interpretable Drug Design

### R2-A: Retrain ProtGPT2 CLT v2

Status: completed.

Artifacts:

- Checkpoint: `/oss-pvc/zhk_zip/outputs/research2/clt_weights/protgpt2_v2/step_200000`
- Local quick eval: `Research2/results/checkpoint_evaluation/protgpt2_v2_quick_eval_20260420.json`

Result:

- Layers: 36.
- d_clt: 8,192.
- k: 128.
- Mean FVU: 0.2013.
- Mean L0: 128.0.
- Mean dead fraction: 0.6197.
- Mean alive fraction: 0.3803.

Interpretation:

ProtGPT2 v2 improved reconstruction quality and alive fraction relative to the
older weak checkpoint, but the dead-feature rate is still high.

### R2-B: Retrain ZymCTRL CLT v2

Status: completed.

Artifacts:

- Checkpoint: `/oss-pvc/zhk_zip/outputs/research2/clt_weights/zymctrl_v2/step_200000`
- Local quick eval: `Research2/results/checkpoint_evaluation/zymctrl_v2_quick_eval_20260420.json`

Result:

- Layers: 36.
- d_clt: 8,192.
- k: 128.
- Mean FVU: 0.3307.
- Mean L0: 128.0.
- Mean dead fraction: 0.6682.
- Mean alive fraction: 0.3318.

Interpretation:

ZymCTRL v2 is usable for follow-up experiments, but the dead-feature rate
remains substantial. This limits the strength of fine-grained steering claims.

### R2-C: Steered Generation CLI

Status: completed and smoke-tested.

Artifact:

- `Research2/results/steered_generation/zymctrl_v2_lysozyme_smoketest.json`

Result:

- Model: ZymCTRL.
- Prompt: `3.2.1.17<sep><start>`.
- Number of generated records: 20.
- One intervention set was applied.

Interpretation:

The CLI path works and can generate steered sequences, but steering efficacy
must be judged by R2-D rather than the smoke test.

### R2-D: Steering Statistical Benchmark

Status: completed; current result is negative.

Artifacts:

- `Research2/results/steering_benchmark/zymctrl_v2_purity_20260425_r2_v2_1gpu.json`
- `Research2/results/steering_benchmark/zymctrl_v2_purity_sweep_20260429_lipase_kinase_ca_m5_n64.json`

Result:

- Full benchmark:
  - 8 EC classes.
  - n = 200 per condition.
  - Layers: L3, L12, L30.
  - Features per layer: 5.
  - Multiplier: 2.5.
  - Significant positive classes: 0/8.
- Strength sweep:
  - Classes: lipase, kinase, carbonic anhydrase.
  - n = 64 per condition.
  - Multiplier: 5.0.
  - lipase: +0.072, p = 0.1066.
  - kinase: +0.026, p = 0.6584.
  - carbonic anhydrase: -0.047, p = 0.2077.
  - Significant positive classes: 0/3.

Interpretation:

The current steering protocol does not support a strong positive R2 steering
claim. Lipase shows a weak numerical trend under stronger intervention, but it
is not statistically significant.

### R2-E: Drug Design Case Study

Status: completed as a candidate-generation case study.

Artifact:

- `Research2/results/drug_design/ec_lysozyme_leads_v2.json`

Result:

- Target: EC lysozyme.
- Generated sequences: 200.
- Lead sequences retained: 10.
- Mean generated sequence length: 214.0.
- Candidate filter captured the Glu/Asp motif in 97.5% of generated records.
- Top lead has length 174 and an ED-diad span recorded as [49, 57].

Interpretation:

The case study produces plausible filtered lead sequences, but downstream
structure and steering validation are mixed. It should be presented as a
prototype candidate-generation pipeline, not as a validated therapeutic design.

### R2-F: Structural QC with ESMFold

Status: completed.

Artifacts:

- `Research2/results/drug_design/ec_lysozyme_esmfold_metrics_v2_20260425_r2_v2_1gpu.json`
- `Research2/results/drug_design/ec_lysozyme_unsteered_esmfold_metrics_v2_20260425_r2_v2_1gpu.json`

Result:

- Steered lead set:
  - Folded sequences: 10.
  - Mean pLDDT: 67.72.
  - Median pLDDT: 71.75.
  - Globally confident fraction: 0.60.
- Unsteered comparison:
  - Folded sequences: 20.
  - Mean pLDDT: 73.19.
  - Median pLDDT: 75.67.
  - Globally confident fraction: 0.70.

Interpretation:

The structural QC does not show a foldability advantage for steering. It does
show that a subset of steered leads is foldable, but unsteered sequences score
better on this small comparison.

### R2-G: Causal Feature Ablation

Status: completed; current ablation result is null.

Artifact:

- `Research2/results/causal_ablation/zymctrl_v2_lysozyme_L12_F8088_20260425_r2_v2_1gpu.json`

Result:

- Sequences analyzed: 10.
- Positions analyzed: 1,665.
- Mean delta: 0.0.
- Mean absolute delta: 0.0.
- Driving fraction: 0.0.
- Suppressive fraction: 0.0.

Interpretation:

This ablation does not currently support a causal feature claim. It likely
needs a different feature selection procedure or an intervention path that
changes logits measurably.

### R2-H: Cross-Model Circuit Conservation

Status: completed after a metric bug fix.

Artifact:

- `Research2/results/circuit_analysis/cross_model_conservation_v2_20260429_corrfix.json`

Result:

- Models: ZymCTRL v2 and ProGen2-medium.
- Sequence cohort: 10 lysozyme leads.
- Layer-level CKA:
  - Anchor L3: 0.6216.
  - Anchor L12: 0.7935.
  - Anchor L30: 0.4198.
- Mean absolute top-feature match correlations:
  - Anchor L3: 0.9838.
  - Anchor L12: 0.9856.
  - Anchor L30: 0.9830.
- Maximum absolute feature correlation after the fix: 0.9975.

Interpretation:

The corrected metric is numerically valid. The high feature-match correlations
should still be interpreted cautiously because the cohort contains only 10
sequences; CKA is the more stable summary.

### R2 T2-A: Direct-Effect Feature Selection

Status: completed.

Artifact:

- `Research2/scripts/16_direct_effect_features.py`
- `Research2/results/circuit_analysis/zymctrl/direct_effect_features_v2.pkl`
- `Research2/results/circuit_analysis/zymctrl/direct_effect_features_v2_summary_20260503.json`

Result:

- Model: ZymCTRL v2.
- CLT checkpoint: `/oss-pvc/zhk_zip/outputs/research2/clt_weights/zymctrl_v2/step_200000`.
- EC classes: 8.
- Layers: 36.
- Top features per EC/layer: 10.
- Acceptance shape: `(8, 36, 10)`.

Interpretation:

Direct-effect features replace the previous mean-activation z-score features as
the main candidates for steering. This is a feature-selection result, not yet a
positive steering result.

### R2 T2-B: TopK-Aware On-Manifold Steering

Status: completed; result is not a positive steering rescue.

Artifact:

- `Research2/src/analysis/circuit_discovery.py`
- `Research2/scripts/11_steering_benchmark.py`
- Remote smoke output:
  `/oss-pvc/zhk_zip/biocc/Research2/results/steering_benchmark/zymctrl_v2_onmanifold_direct_smoke_20260503.json`
- Remote full-run output:
  `/oss-pvc/zhk_zip/biocc/Research2/results/steering_benchmark/zymctrl_v2_onmanifold_direct_20260503.json`
- Local full-run output:
  `Research2/results/steering_benchmark/zymctrl_v2_onmanifold_direct_20260503.json`

Result:

- The hook now re-applies CLT TopK after intervention and replaces the
  CLT-explained same-layer MLP component instead of adding an off-manifold
  single-feature decoder delta.
- A lysozyme smoke benchmark completed successfully.
- The full 8-class benchmark completed with n=100 per condition, layers
  L3/L12/L30, top-5 direct-effect features per layer, and multiplier=2.5.
- Significant positive steering: 0/8 classes.
- Largest positive shifts:
  - kinase: +0.107, 95% CI [+0.000, +0.210], permutation p=0.0532.
  - lipase: +0.057, 95% CI [-0.007, +0.122], permutation p=0.0913.

Interpretation:

This confirms the new hook and feature selector produce measurable nonzero
effects, but the current heuristic purity benchmark still does not support a
positive R2 steering claim. The final R2 viability decision should wait for the
T2-C real metric triad.

### R2 T2-C: Real EC-Class Metric Triad

Status: blocked by unstaged external tools/databases.

Readiness check:

- Missing executables in the current H200 pod: `hmmscan`, `hmmsearch`,
  `diamond`, `foldseek`, and `esm-fold`.
- Present assets: `data/interpro/pfam_residue.tsv` and ESMFold model weights at
  `/oss-pvc/zhk_zip/models/esmfold_v1`.
- No CLEAN database, Pfam HMM database, Foldseek binary, or Foldseek target
  structure database was found in the checked local/remote paths.

Interpretation:

Per TODO_NEXT guidelines, no new EC metric should be claimed without a
calibration run. T2-C remains blocked until the metric tools and databases are
staged and calibrated against known positive/negative controls.

### R2 T2-D: Viability Decision Gate

Status: completed as a no-go decision for the current steering claim.

Artifact:

- `Research2/docs/EXPERIMENT_LOG.md` EXP-R2-011.

Decision:

No-go for a strong R2 steering/drug-design claim in the current round. T0-A
shows hooks can move logits, T2-A/B show measurable but non-significant effects,
and T2-C is blocked by unstaged metric tools/databases. Do not proceed to Tier 3
wet-lab or substrate-swap claims from the current evidence.

Recommended framing:

R2 should be framed as an interpretability and layer-map pipeline with a
transparent negative steering result unless calibrated T2-C metrics later show
significant positive shifts in at least 3 of 8 EC classes, or a stronger CLT is
trained and revalidated.

## Overall Takeaways

R1 has useful SAE signals, but the headline must be narrower than originally
planned. The cancer holdout supports SAE+LLR for pathogenicity, and variant-level
mechanism labels show structure, but the LOF/GOF/DN classifier fails
protein-level generalization. ProteinGym is also negative for generic DMS
fitness improvement: the sign-corrected ensemble improves over the broken legacy
ensemble but remains below LLR on average.

R2 has the infrastructure, v2 checkpoints, generation, structural QC, ablation,
and cross-model conservation artifacts completed. However, the current steering
and causal-ablation evidence is weak or negative. R2 is best framed as an
interpretable generation pipeline with clear limitations unless a new steering
protocol produces statistically significant effects.


### External Resource Supplement Update (2026-05-04 CST)

Status: partially completed.

Completed staging:

- HMMER 3.4 compiled on H200, with `hmmscan`, `hmmsearch`, and `hmmpress` in `external_resources/tools/bin`.
- Pfam-A HMM downloaded, checksum-checked, decompressed, and `hmmpress` indexed on H200.
- DIAMOND upgraded to v2.1.24 and verified on H200.
- Foldseek AVX2 binary staged and verified on H200.
- AlphaMissense hg38 and UniProt amino-acid substitution tables staged locally and on H200.
- CLEAN source staged locally and on H200.

Still blocked:

- PrimateAI-3D requires gated Hugging Face/Illumina academic-license approval.
- gMVP requires a dbNSFP acquisition path/license decision.
- ESM-1v needs a scoring backend choice before downloading large model files.
- CLEAN still needs pretrained CLEAN weights and ESM-1b weights.
- Foldseek still needs a selected target structure database.

Manifest: `external_resources/manifests/external_resource_status_20260504.md`.

### Runnable External-Resource TODO Pass (2026-05-04 CST)

Completed runnable follow-ups after resource staging:

- T1-A AlphaMissense baseline: `Research1/results/variant_effect/alphamissense_baseline_20260504.md`.
  - ClinVar2000: 1,972/2,000 matched, AUC 0.9474 [0.9377, 0.9567].
  - CancerHoldout101: 101/101 matched, AUC 0.9700 [0.9244, 0.9988].
- T2-C Pfam/HMMER partial metric: `Research2/results/ec_metrics/pfam_generated_lysozyme_20260504.md`.
  - Steered leads: 9/10 lysozyme-like Pfam hits.
  - Steered all vs unsteered: 0.860 vs 0.820 lysozyme-like hit rate, one-sided Fisher p=0.1699.

Remaining blockers are now narrower: T1-A still needs PrimateAI-3D, gMVP, and ESM-1v; T2-C still needs CLEAN weights and a Foldseek target DB.

### Available External Baseline and T2-C Supplement Pass (2026-05-07 CST)

Status: completed for all currently accessible resources. As of 2026-05-10,
PrimateAI-3D is treated as unavailable rather than pending.

Completed R1 outputs:

- `Research1/results/variant_effect/external_baselines_available_20260507.{json,md}`
- `Research1/results/variant_effect/external_baselines_available_scores_20260507.tsv`
- `Research1/results/variant_effect/available_baseline_summary_20260507.{json,md}`
- `Research1/results/variant_effect/available_baseline_sae_residual_cases_20260507.tsv`

R1 headline table:

- ClinVar2000 AUCs: SAE-LR 0.8782, ESM-2 LLR 0.8822, SAE+LLR 0.9143,
  AlphaMissense 0.9474, gMVP 0.9369, ESM-1v 0.9089.
- CancerHoldout101 AUCs: SAE-LR 0.9079, ESM-2 LLR 0.8978, SAE+LLR 0.9193,
  AlphaMissense 0.9700, gMVP 0.9400, ESM-1v 0.8552.
- Scalar external baselines remain weak for LOF/GOF/DN mechanism separation:
  macro one-vs-rest AUC is 0.4800 for AlphaMissense, 0.4856 for gMVP, and
  0.4847 for ESM-1v, versus 0.5161 for SAE-LR and 0.5164 for SAE+LLR under the
  same scalar mechanism audit.
- Spearman correlations show substantial but incomplete overlap between SAE
  and external scorers; the residual TSV is the local artifact for selecting
  SAE-specific interpretation cases.

Final PrimateAI-3D disposition (2026-05-10 CST):

- PrimateAI-3D gated access could not be obtained for this analysis round.
- T1-A should no longer be blocked on PrimateAI-3D.
- The reviewer-facing accessible-baseline table is therefore:
  SAE-LR, ESM-2 LLR, SAE+LLR, AlphaMissense, gMVP, and ESM-1v.
- The scientific conclusion is unchanged: AlphaMissense and gMVP beat SAE+LLR
  on scalar pathogenicity, while scalar external baselines remain weak for
  LOF/GOF/DN mechanism separation.

Completed R2 generated-sequence metric triad:

- `Research2/results/ec_metrics/clean_generated_lysozyme_20260507.{json,md}`
- `Research2/results/ec_metrics/foldseek_generated_lysozyme_20260507.{json,md}`
- `Research2/results/ec_metrics/generated_metric_triad_summary_20260507.{json,md}`

R2 generated-sequence metric table:

- Steered leads: Pfam lysozyme-like 0.900, CLEAN exact 3.2.1.17 0.900,
  Foldseek mean top TM 0.883, TM >= 0.7 fraction 0.900.
- Steered all vs unsteered: Pfam lysozyme-like 0.860 vs 0.820
  (one-sided Fisher p=0.170), CLEAN exact 0.775 vs 0.775, CLEAN 3.2.1.x prefix
  0.865 vs 0.875.
- Interpretation: the selected lead filter is strong, but the generation-wide
  steering lift is not significant. R2 should still be framed cautiously unless
  calibrated T2-C controls and additional EC classes change this conclusion.

Calibration work completed:

- Prepared real-vs-random calibration set:
  `Research2/results/ec_metrics/calibration_lysozyme_20260507/`
  with 100 real EC 3.2.1.17 SwissProt/ZymCTRL lysozyme records and 100
  length-matched random UniRef50 records.
- Added runnable scripts:
  `Research2/scripts/21_prepare_ec_calibration.py`,
  `Research2/scripts/22_foldseek_calibration.py`,
  `Research2/scripts/23_ec_metric_calibration_summary.py`, and
  `Research2/scripts/run_t2c_calibration_20260507.sh`.
- H200 calibration outputs:
  `Research2/results/ec_metrics/ec_metric_calibration_summary_20260507.{json,md}`.
- Real-vs-random metric separation:
  - Pfam lysozyme-like hit: 0.910 vs 0.000, Cohen's d 4.497.
  - CLEAN exact EC 3.2.1.17: 0.960 vs 0.000, Cohen's d 6.928.
  - CLEAN 3.2.1.x prefix: 0.990 vs 0.050, Cohen's d 5.549.
  - ESMFold mean pLDDT: 79.740 vs 58.478, Cohen's d 1.786.
  - ESMFold confident fraction: 0.865 vs 0.336, Cohen's d 2.063.
  - Foldseek top TM: 0.971 vs 0.653, Cohen's d 1.593.
- All calibrated metrics exceed the TODO_NEXT separation threshold on this
  lysozyme real-vs-random control. This validates the metric stack itself, but
  does not reverse the generated steered-vs-unsteered conclusion above.

### R1 T1-E Channelopathy Label Curation (2026-05-07 CST)

Status: curated labels staged; scoring/concordance completed in the next
section.

Completed artifacts:

- `Research1/data/channelopathy/curation_work/kcnq1_lqts_worker.tsv`
- `Research1/data/channelopathy/curation_work/scn5a_worker.tsv`
- `Research1/data/channelopathy/curation_work/kcnh2_cacna1c_worker.tsv`
- `Research1/data/channelopathy/curation_work/clinvar_clingen_worker.tsv`
- `Research1/data/channelopathy/channelopathy_mechanism_labels.tsv`
- `Research1/data/channelopathy/channelopathy_mechanism_positive_labels.tsv`
- `Research1/data/channelopathy/channelopathy_drug_response_labels.tsv`
- `Research1/data/channelopathy/channelopathy_label_sources.md`
- `Research1/data/channelopathy/channelopathy_label_summary.json`
- `Research1/results/variant_effect/channelopathy_readiness_20260507.{json,md}`

Summary:

- Input worker rows: 121.
- Consolidated rows after deduplication: 119.
- Positive mechanism rows: 74.
- Drug-response rows: 40.
- Positive mechanism label coverage by gene:
  - KCNQ1: 23.
  - SCN5A: 25.
  - KCNH2: 19.
  - CACNA1C: 7.
- Mechanism label counts in the full consolidated file:
  LOF 40, GOF 13, DN 13, mixed_complex 8, unknown 45.

Caveats:

- These are research-use labels, not clinical guidance.
- Unknown rows are ClinVar/ClinGen candidate/source-index rows and should not
  be treated as positive mechanism labels.
- KCNQ1 beta-blocker labels are mostly genotype/region-level, not per-variant
  randomized drug-response evidence.
- CACNA1C A39V/G490R and similar rows have direct functional evidence but
  should be treated carefully because clinical pathogenicity has controversy.

### R1 T1-E Channelopathy Mechanism Concordance (2026-05-07 CST)

Status: scoring completed; TODO acceptance target not met.

Completed artifacts:

- `Research1/scripts/31_channelopathy_concordance.py`
- `Research1/scripts/run_t1e_channelopathy_20260507.sh`
- `Research1/data/channelopathy/channelopathy_canonical_sequences.fasta`
- `Research1/results/variant_effect/channelopathy_concordance_20260507.json`
- `Research1/results/variant_effect/channelopathy_concordance_20260507.md`
- `Research1/results/variant_effect/channelopathy_concordance_20260507.predictions.tsv`
- `Research1/results/variant_effect/channelopathy_concordance_20260507.audit.tsv`
- `Research1/results/variant_effect/channelopathy_concordance_20260507.supported.tsv`
- `Research1/results/variant_effect/channelopathy_concordance_20260507.prepare.json`
- `Research1/results/variant_effect/channelopathy_concordance_20260507.signatures.pkl`

Summary:

- High-confidence curated labels: 74.
- Scoreable missense variants: 69.
- Headline evaluated LOF/GOF/DN rows: 64.
- Mixed/complex rows scored but excluded from headline accuracy: 5.
- Accuracy: 0.625.
- Macro-F1: 0.444.
- Confusion matrix, rows=true and columns=predicted over DN/GOF/LOF:
  `[[0, 2, 11], [0, 7, 4], [3, 4, 33]]`.
- By-gene accuracy: CACNA1C 0.500, KCNH2 0.842, KCNQ1 0.455,
  SCN5A 0.632.

Interpretation:

- T1-E does not pass the >=80% expert-mechanism concordance target.
- The dominant failure mode is DN channel variants being predicted as LOF.
- KCNH2 looks strong mostly because the curated subset is LOF-heavy.
- KCNQ1 is weak because many curated DN labels collapse to LOF under the
  existing R1 mechanism classifier.
- The result is still useful as a diagnostic: the current SAE mechanism head
  distinguishes many GOF/LOF cases, but it does not yet model channel-specific
  dominant-negative biology well enough for a clinical-mechanism claim.
