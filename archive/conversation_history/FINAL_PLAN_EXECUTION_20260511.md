# Final Plan Execution Log

Date: 2026-05-11

This file records execution of `OPUS_FINAL_PLAN_20260511.md` under the
no-wet-lab two-paper strategy.

## Completed Today

### P0 Readiness Audit

Added and ran:

- `ops/final_plan_readiness_audit.py`
- `results/final_plan_readiness/final_plan_readiness_20260511.json`
- `results/final_plan_readiness/final_plan_readiness_20260511.md`

Key findings:

- `F-T1-1` indel scale-up is only partially ready. The current
  protein-HGVS-only reconstruction has 185,655 ClinVar indel rows, but only
  18,897 supported reconstructed rows across all labels and only 6,649
  binary-label, length-compatible records ready for the existing scorer.
- The current 6,649-record indel prediction run already covers the binary,
  length-compatible subset. Reaching the final plan's ~80k target requires
  transcript-aware frameshift handling and/or improved UniProt mapping.
- `F-T1-2` indel competitor comparison is not ready locally: CADD, REVEL, and
  SpliceAI indel score files were not found.
- `F-T2-1` five-task R2 downstream evaluation is not ready as written:
  EC/Pfam resources are partially ready, but CB513, DeepLoc, and
  FireProtDB/ProTherm are not staged locally.

### F-T1-3 SAE x AlphaMissense Ensemble

Added and ran:

- `Research1/scripts/34_alphamissense_ensemble.py`
- `Research1/results/variant_effect/alphamissense_sae_ensemble_20260511.json`
- `Research1/results/variant_effect/alphamissense_sae_ensemble_20260511.md`
- `Research1/results/variant_effect/alphamissense_sae_ensemble_predictions_20260511.tsv`

Method:

- Used the 1,972 ClinVar2000 variants matched to AlphaMissense.
- Used gene-symbol grouped `StratifiedGroupKFold` as the current conservative
  protein/gene-level split.
- Compared AlphaMissense, SAE-LR group-CV predictions, and two SAE+AM
  ensembles.

Result:

| Method | AUC | Variant bootstrap 95% CI |
|---|---:|---|
| AlphaMissense | 0.9474 | [0.9373, 0.9568] |
| SAE-LR group-CV | 0.7559 | [0.7341, 0.7771] |
| AM + SAE stack | 0.8542 | [0.8382, 0.8701] |
| AM + SAE z-sum | 0.9210 | [0.9083, 0.9319] |

Delta versus AlphaMissense:

| Method | Delta AUC | Group bootstrap 95% CI |
|---|---:|---|
| AM + SAE stack minus AlphaMissense | -0.0932 | [-0.1098, -0.0770] |
| AM + SAE z-sum minus AlphaMissense | -0.0264 | [-0.0344, -0.0189] |
| SAE-LR group-CV minus AlphaMissense | -0.1915 | [-0.2171, -0.1660] |

Gate decision:

- The complementarity gate fails.
- Do not claim that SAE improves AlphaMissense scalar pathogenicity.
- Keep SAE as an interpretation/residual diagnostic layer, not as an
  AlphaMissense ensemble improvement.

## Updated Execution Implications

1. Start manuscript reframing now: the R1 paper should explicitly state that
   AlphaMissense remains the stronger scalar pathogenicity predictor.
2. The R1 indel story should not promise ~80k scored records unless
   transcript-aware frameshift reconstruction is implemented.
3. The R1 indel story can still be framed as a first systematic
   protein-sequence indel diagnostic on the reconstructable subset.
4. R2 should not start the five-task downstream evaluation until CB513,
   DeepLoc, and stability resources are staged, or the plan is reduced to an
   EC/Pfam pilot.
5. SAE residual case studies remain useful only as interpretability examples,
   not as evidence of scalar predictive improvement over AlphaMissense.

### F-T0 Manuscript Reframing and Evidence Tags

Updated:

- `manuscripts/nature_methods_r1_variant_perturbation/main.tex`
- `manuscripts/nature_methods_r1_variant_perturbation/README.md`
- `manuscripts/nature_methods_r2_circuit_diagnostics/main.tex`
- `manuscripts/nature_methods_r2_circuit_diagnostics/README.md`

R1 changes:

- Added evidence-path comments before major numeric claims.
- Added gMVP and ESM-1v to the accessible scalar baseline table.
- Added the negative AlphaMissense+SAE ensemble gate as an explicit result.
- Reframed indels as 6,649 binary-label reconstructable protein-sequence edits,
  not an approximately 80k full-scale claim.
- Removed wet-lab prioritization as a required next step for the current paper;
  assay work is future validation only.

R2 changes:

- Retitled and reframed from controlled generation to sparse circuit-latent
  diagnostics for protein generation.
- Updated steering results to the TopK-aware direct-effect benchmark:
  0/8 significant positive EC-class shifts.
- Replaced the old hook-failure statement with the current interpretation:
  hook sanity passes, while the original ablation path remains unsuitable for
  causal interpretation.
- Added calibrated Pfam/CLEAN/ESMFold/Foldseek lysozyme metric-triad results.
- Removed wet-lab validation as a dependency for the current article.

### R2 Resource-Ready Pilot Experiments

Submitted a new 1-GPU H200 job:

- `jiaotongdamoxing-zhk-zip-final-1gpu-0511`
- Node: `192.168.20.203`

Completed on the pod:

1. `F-T2-3` pilot quality detection from existing metrics:
   - Output:
     `Research2/results/ec_metrics/quality_detection_from_existing_metrics_20260511.{json,md}`
   - Logistic CV AUC: 0.9649 [0.9375, 0.9868].
   - Foldseek top TM univariate AUC: 0.9571 [0.9255, 0.9800].

2. `F-T2-1` resource-ready CLT representation pilot:
   - Output:
     `Research2/results/ec_metrics/clt_representation_lysozyme_probe_20260511.{json,md}`
   - Task: real EC 3.2.1.17 lysozyme vs length-matched random UniRef50
     proteins.
   - ZymCTRL raw hidden probe AUC: 0.9926 [0.9822, 0.9994].
   - ZymCTRL pooled CLT probe AUC: 0.9934 [0.9820, 0.9996].
   - Delta CLT minus raw: +0.0008.

Interpretation:

- These pilots support the R2 diagnostic direction and show that CLT features
  are usable on a narrow EC-like discrimination task.
- They do not replace the final plan's five-task downstream benchmark because
  CB513, DeepLoc and stability resources remain missing.

## Recommended Next Actions

1. `F-T0-1/F-T0-2`: reframe both manuscripts and add evidence tags.
2. Decide whether to implement transcript-aware frameshift reconstruction for
   `F-T1-1` or revise the indel headline to the current reconstructable subset.
3. Stage or drop CADD/REVEL/SpliceAI for `F-T1-2`.
4. Stage CB513, DeepLoc, and FireProtDB/ProTherm before `F-T2-1`, or formally
   reduce `F-T2-1` to an EC/Pfam pilot.

### Continued Resource-Ready Runs

Added locally:

- `Research2/scripts/26_universal_atlas_summary.py`
- `Research1/scripts/38_package_indelmissense.py`

Completed on the H200 pod:

1. Cross-model `F-T2-1` lysozyme-vs-random CLT representation probes:
   - ProtGPT2 raw hidden probe AUC: 0.9899 [0.9785, 0.9985].
   - ProtGPT2 pooled CLT probe AUC: 0.9932 [0.9855, 0.9985].
   - Delta CLT minus raw: +0.0033.
   - ProGen2-medium raw hidden probe AUC: 0.9994 [0.9980, 1.0000].
   - ProGen2-medium pooled CLT probe AUC: 0.9982 [0.9952, 1.0000].
   - Delta CLT minus raw: -0.0012.

Completed on the H200 pod:

- `F-T2-2` resource-ready three-model conservation pilot over the first 100
  calibration records.
- Outputs:
  `Research2/results/circuit_analysis/cross_model_conservation_3model_lysozyme_20260511.json`
  and
  `Research2/results/circuit_analysis/universal_atlas_pilot_summary_20260511.{json,md}`.
- Important caveat: the first 100 calibration records are all real lysozymes,
  so this is a same-family conservation pilot rather than a balanced
  real-vs-random atlas.
- Result: 9 exact three-model feature triplets at abs(r) >= 0.90 from the
  top-30 pairwise matches. The strongest triplet has min abs(r)=0.9955 across
  ProtGPT2 L30 feature 1164, ZymCTRL L30 feature 5283, and ProGen2-medium
  mapped L22 feature 643.

Running on the H200 pod:

- `F-T2-2` balanced-200 conservation rerun over interleaved 100 real lysozymes
  and 100 random UniRef50 controls.
- Changes vs pilot: `--top-feature-pairs 100`, `--feature-pool-size 2048`.
- Output targets:
  `Research2/results/circuit_analysis/cross_model_conservation_3model_balanced200_20260511.json`
  and
  `Research2/results/circuit_analysis/universal_atlas_balanced200_summary_20260511.{json,md}`.

Completed follow-ups:

- Balanced-200 top-100 rerun: 12 exact three-model triplets at abs(r) >= 0.90.
- Balanced-200 wide-match rerun (`top_feature_pairs=300`,
  `feature_pool_size=4096`): 38 exact triplets at abs(r) >= 0.90, 30 at
  abs(r) >= 0.95, and 8 at abs(r) >= 0.98.
- Permutation null control over the same balanced-200 cohort:
  3 shuffled replicates produced 0 triplets at thresholds 0.90, 0.95, and
  0.98.
- Source-selectivity annotation over the 38 triplets:
  all 38 were weak/source-mixed for real-lysozyme vs random-UniRef50
  discrimination. This means the conserved triplets are not simple lysozyme
  family detectors, but it also means the current pass has not produced
  >=10 Pfam/functional annotations.

Running follow-up:

- Broader UniRef50 conservation pilot over 500 sequences:
  `cross_model_conservation_3model_uniref500_wide_20260512`.
  This is still a pilot relative to the final 10k-sequence atlas target.

Completed locally:

- `F-T1-7` bounded IndelMissense static benchmark artefact:
  `data/indelmissense/v1/`.
- Contents: `records.jsonl`, `records.tsv`, `baseline_scores.tsv`,
  `splits.csv`, `metadata.json`, `README.md`.
- Scope: 6,649 binary-label reconstructable protein-sequence indels.
- Deterministic protein-level split: train 5,356, validation 653, test 640.
- Current damage-score AUC over all packaged records: 0.7735.

Interpretation:

- The three model-specific lysozyme probes reinforce that CLT pooled features
  are usable but essentially tied with raw hidden features on this narrow
  calibration task.
- The IndelMissense artefact completes the bounded dataset-release path for
  the current R1 manuscript scope. It does not satisfy the original ~80k
  full-scale indel target, which remains blocked by transcript-aware
  reconstruction and mapping.
