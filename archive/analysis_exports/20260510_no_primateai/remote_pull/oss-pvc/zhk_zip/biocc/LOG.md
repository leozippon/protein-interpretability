# Research Status Log

Last updated: 2026-05-04 CST

---

## H200 Single-GPU Continuation (2026-04-25)

- Per the request to leave two H200 GPUs available for other users, submitted a
  1-GPU Arena job:
  `jiaotongdamoxing-zhk-zip-r2-v2-1gpu-0425`.
- The job is running on `i-d5cvmv6heob1nidq4ujg` / `192.168.20.204` with one
  requested H200. Kubernetes active allocation after scheduling is
  `192.168.20.203: 8/8` and `192.168.20.204: 6/8`, leaving two H200 GPUs free
  on `192.168.20.204` for Arena scheduling.
- Synced the R2 remaining runner to be single-GPU-safe:
  structural QC now uses `STRUCTURAL_CUDA_DEVICE=0` by default instead of
  hard-coded device 1, and uses `STRUCTURAL_DTYPE=fp32` by default for the
  local ESMFold path.
- Entry command:
  `bash scripts/run_r2_v2_remaining_then_hold_0424.sh`. The wrapper keeps the
  pod alive with `tail -f /dev/null` after the queue exits.
- Runtime log prefix:
  `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/logs/runtime/*_20260425_r2_v2_1gpu.log`.

---

## TODO Completion Pass (2026-04-26)

- Fixed remote zero-byte scripts on OSS:
  `Research2/scripts/15_cross_model_conservation.py` and
  `Research1/scripts/20_indel_extension.py`.
- R2-H cross-model conservation was rerun successfully on the 1-GPU H200 pod.
  Latest output now points to:
  `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/results/circuit_analysis/cross_model_conservation_v2_20260426_todo_completion.json`.
- R2-A and R2-B v2 checkpoints are present at `step_200000`:
  `/oss-pvc/zhk_zip/outputs/research2/clt_weights/protgpt2_v2/step_200000`
  and `/oss-pvc/zhk_zip/outputs/research2/clt_weights/zymctrl_v2/step_200000`.
- Added `Research1/scripts/21_proteingym_sae_followup.py` to fill the
  ProteinGym SAE/ensemble fields left pending by the original LLR-only
  benchmark.
- Started the full ProteinGym SAE follow-up in the current 1-GPU H200 pod:
  PID `2805`, log
  `/oss-pvc/zhk_zip/biocc/Research1/logs/runtime/r1d_proteingym_sae_followup_20260426.log`,
  output
  `/oss-pvc/zhk_zip/biocc/Research1/results/variant_effect/proteingym_benchmark_sae_latest.json`.
  The job is resume-friendly and writes after each assay.

## Diagnostic / Strengthening Pass (2026-04-29 CST)

- R1-D ProteinGym SAE follow-up completed all 217 assays. Added
  `Research1/scripts/22_proteingym_result_diagnostics.py` and ran an
  assay-level diagnostic on the remote output:
  `/oss-pvc/zhk_zip/biocc/Research1/results/variant_effect/proteingym_benchmark_sae_diagnostics_20260429.json`.
  Usable assays: 214/217. Mean Spearman: LLR 0.4341, SAE -0.2314,
  negated-SAE 0.2314, ensemble 0.1993. The ensemble beats LLR on only
  8.4% of usable assays, so the ProteinGym issue is systematic rather than
  a small number of outliers.
- Fixed `Research2/scripts/15_cross_model_conservation.py`: feature-profile
  Pearson correlations now divide by `n` to match `np.std(ddof=0)`, preventing
  invalid correlations greater than 1.0 on small cohorts.
- Reran R2-H after the correlation fix:
  `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/results/circuit_analysis/cross_model_conservation_v2_20260429_corrfix.json`.
  The latest symlink now points to this corrected file. Max absolute feature
  correlation is 0.9975; CKA values remain [0.6216, 0.7935, 0.4198].
- Started a small R2-D steering-strength sweep on the existing 1-GPU H200 pod
  for the classes that had the largest positive but non-significant baseline
  effects (`lipase`, `kinase`, `carbonic_anh`): multiplier 5.0, top-5
  features/layer, layers L3/L12/L30, n=64 per condition. Log:
  `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/logs/runtime/r2d_steering_sweep_20260429_lipase_kinase_ca_m5_n64.log`.
  Output:
  `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/results/steering_benchmark/zymctrl_v2_purity_sweep_20260429_lipase_kinase_ca_m5_n64.json`.
  Results: lipase +0.072 (p=0.1066), kinase +0.026 (p=0.6584),
  carbonic_anh -0.047 (p=0.2077), 0/3 significant positive. Increasing
  steering strength helps lipase numerically but does not rescue the R2-D
  steering claim by itself.

## TODO_NEXT Tier-0 Execution (2026-04-29 CST)

- T0-A R2 hook sanity diagnostic completed and passed for ZymCTRL v2 and
  ProGen2-medium:
  `Research2/results/diagnostics/hook_sanity_20260429.json`.
  ZymCTRL: multiplier=10 max logit shift 12.93, multiplier=0 shift 0.885.
  ProGen2-medium: multiplier=10 shift 3.938, multiplier=0 shift 1.250.
  Conclusion: the MLP hook path can move logits; the earlier all-zero
  causal-ablation result is not explained by a universally disconnected hook.
- T0-C `ec_features.pkl` provenance audit completed and passed:
  `Research2/results/diagnostics/ec_features_provenance_20260429.json`.
  The H200 file has 8 EC classes, 36 layers, d_clt=8192 for every layer, and
  matches the ZymCTRL v2 checkpoint dimensionality. The local
  `Research2/results/circuit_analysis/zymctrl/ec_features.pkl` was refreshed
  from H200 and validated as an 8192-dimensional pickle.
- T0-D protein-level mechanism holdout completed:
  `Research1/results/variant_effect/mechanism_classifier_results_t0_protein_holdout_20260429.json`.
  The default `mechanism_classifier_results.json` now contains both
  variant-level and protein-level CV; the previous variant-CV-only result is
  backed up as `mechanism_classifier_results_variantcv_legacy_20260429.json`.
  Variant-level SAE macro-AUC is 0.7471; protein-level SAE macro-AUC is 0.5161.
  This fails the TODO_NEXT 0.7 threshold, so the R1 headline must be downgraded
  from "predicts mechanism" to "variant-level mechanism signal does not
  generalize cleanly across proteins yet."
- T0-B signed ProteinGym rerun completed on the current 1-GPU H200 pod at
  2026-05-01 21:51 CST. Final remote output:
  `/oss-pvc/zhk_zip/biocc/Research1/results/variant_effect/proteingym_benchmark_sae_signed_20260429_signed.json`.
  Local copies:
  `Research1/results/variant_effect/proteingym_benchmark_sae_signed_20260429_signed.json`,
  `Research1/results/variant_effect/proteingym_benchmark_sae_signed_diagnostics_20260503.json`,
  and `Research1/logs/runtime/t0b_proteingym_signed_20260429_signed.log`.
  Results over 214 usable assays: LLR mean rho 0.4341, raw SAE -0.2314,
  legacy plus ensemble 0.1993, sign-corrected ensemble 0.4047
  (95% bootstrap CI [0.3818, 0.4282]). The sign fix rescues most of the
  cancellation but still does not beat LLR on average; signed ensemble beats
  LLR on 33.6% of usable assays, below the TODO_NEXT 40% threshold. Because
  T0-D protein-level mechanism CV failed, the classifier-based ProteinGym score
  was not treated as a defensible paper claim.

## TODO_NEXT Tier-1 / Tier-2 Execution (2026-05-03 CST)

- Added and ran T1-B preliminary mechanism feature audit:
  `Research1/scripts/24_mechanism_feature_audit.py`.
  Outputs:
  `Research1/results/variant_effect/mechanism_feature_audit_20260503.json`
  and `Research1/results/variant_effect/mechanism_feature_audit_20260503.md`.
  The audit contains 150 rows (3 classes x 5 layers x top-10 features) with
  coefficient, feature kind, best annotation, Pfam/domain proxy,
  GO/functional proxy, binding proxy, and a conservative manual interpretation
  note. Current annotation pkls still lack `firing_positions`, so max-activating
  sequence inspection remains blocked on T1-D.
- Added and completed T2-A direct-effect feature selection:
  `Research2/scripts/16_direct_effect_features.py`.
  Remote outputs:
  `/oss-pvc/zhk_zip/biocc/Research2/results/circuit_analysis/zymctrl/direct_effect_features_v2.pkl`
  and
  `/oss-pvc/zhk_zip/biocc/Research2/results/circuit_analysis/zymctrl/direct_effect_features_v2_summary_20260503.json`.
  Acceptance shape is `(8, 36, 10)` for 8 EC classes, 36 ZymCTRL layers, and
  top-10 direct-effect features per class/layer.
- Implemented T2-B TopK-aware on-manifold steering in
  `Research2/src/analysis/circuit_discovery.py`: the hook now recomputes CLT
  pre-activations, applies the feature intervention, re-applies TopK, and
  replaces the CLT-explained same-layer MLP component instead of adding a
  single off-manifold decoder vector.
- Updated `Research2/scripts/11_steering_benchmark.py` to optionally consume
  the T2-A direct-effect pkl. A lysozyme smoke test passed and wrote:
  `/oss-pvc/zhk_zip/biocc/Research2/results/steering_benchmark/zymctrl_v2_onmanifold_direct_smoke_20260503.json`.
- Completed the full T2-B direct-effect/on-manifold steering benchmark in the
  current 1-GPU H200 pod. Log:
  `/oss-pvc/zhk_zip/biocc/Research2/logs/runtime/t2b_onmanifold_direct_steering_20260503.log`,
  output
  `/oss-pvc/zhk_zip/biocc/Research2/results/steering_benchmark/zymctrl_v2_onmanifold_direct_20260503.json`.
  Local copies were pulled to
  `Research2/logs/runtime/t2b_onmanifold_direct_steering_20260503.log` and
  `Research2/results/steering_benchmark/zymctrl_v2_onmanifold_direct_20260503.json`.
  Result: 0/8 EC classes showed significant positive steering. Largest positive
  effects were kinase +0.107 (95% CI [+0.000, +0.210], p=0.0532) and lipase
  +0.057 (95% CI [-0.007, +0.122], p=0.0913), but neither passed the
  significance criterion. The current H200 GPU is idle after completion.
- Started T1-D annotation rerun with firing positions on the current 1-GPU H200
  pod. Code changes:
  `Research1/src/analysis/feature_annotation.py` now stores optional
  `firing_positions` and `top_firing_examples`, and
  `Research1/scripts/04_analyze_our_sae.py` now supports
  `--save-firing-positions`, `--max-firing-positions-per-feature`,
  `--checkpoint-root`, and `--out-prefix`.
  Runner:
  `Research1/scripts/run_t1d_annotation_firing_20260503.sh`.
  Remote PID 8318, log:
  `/oss-pvc/zhk_zip/biocc/Research1/logs/runtime/t1d_annotation_firing_20260503.log`.
  The runner processes layers 19/23/27/31/35 with 1000 Swiss-Prot proteins,
  writes firing-enabled base annotation pkls, backs up previous no-firing pkls,
  and then runs `19_expand_annotation.py` to
  `Research1/results/annotation_alignment/expanded_summary_firing_20260503.json`.
- T1-D status update: all five firing-enabled annotation pkls completed.
  `19_expand_annotation.py` initially exposed a filtering bug
  (`Swiss-Prot universe: 0 proteins`) because the Swiss-Prot cache stores
  `ProteinAnnotation.accession`, not `uniprot_id` dict entries. Fixed
  `Research1/scripts/19_expand_annotation.py` to collect the accession universe
  directly from `firing_positions`, stopped the bad expansion process, and
  restarted only the expansion stage as PID 9635. Fixed log:
  `/oss-pvc/zhk_zip/biocc/Research1/logs/runtime/t1d_expand_annotation_fixed_20260503.log`.
  The fixed run correctly reports a 1000-protein firing-position universe.
- T1-D completed. Local summaries/logs:
  `Research1/results/annotation_alignment/expanded_summary_firing_20260503.json`
  and `Research1/logs/runtime/t1d_expand_annotation_fixed_20260503.log`.
  Expansion increased KNOWN feature counts substantially at all layers:
  L19 146→381, L23 135→301, L27 72→163, L31 49→86, L35 32→45.
  This unlocks firing-position-based feature audit, but it fails the T1-D
  acceptance threshold for L35 (target ≥60; observed 45).
- T1-B final firing-position feature audit completed locally after pulling the
  firing-enabled expanded annotation pkls from H200. Outputs:
  `Research1/results/variant_effect/mechanism_feature_audit_firing_20260504.json`
  and `Research1/results/variant_effect/mechanism_feature_audit_firing_20260504.md`.
  The audit has 150/150 rows with firing positions and top residue examples;
  142/150 rows also have expanded labels. It is figure-planning evidence, not
  manual biological validation.
- Added `Research1/scripts/25_indel_mechanism.py` for T1-C. Local preparation
  produced 6,649 reconstructable pathogenic/benign indel records at
  `Research1/results/variant_effect/indel_records_supported_20260504.jsonl`;
  the missense mechanism classifier was cached as pure numpy LR parameters at
  `Research1/results/variant_effect/indel_mechanism_classifier_20260504.pkl`.
  A 2-record H200 smoke test passed and wrote
  `/oss-pvc/zhk_zip/biocc/Research1/results/variant_effect/indel_mechanism_predictions_smoke_20260504.jsonl`.
- Completed the full T1-C indel mechanism-transfer run on the current 1-GPU H200
  pod. Log:
  `/oss-pvc/zhk_zip/biocc/Research1/logs/runtime/t1c_indel_mechanism_20260504.log`;
  output:
  `/oss-pvc/zhk_zip/biocc/Research1/results/variant_effect/indel_mechanism_predictions_20260504.jsonl`.
  Local copies were pulled to
  `Research1/logs/runtime/t1c_indel_mechanism_20260504.log`,
  `Research1/results/variant_effect/indel_mechanism_predictions_20260504.jsonl`,
  and
  `Research1/results/variant_effect/indel_mechanism_predictions_20260504_summary.json`.
  Final results: 6,649/6,649 records predicted; pathogenicity AUC from the
  indel SAE damage score is 0.7735. Predicted mechanism counts are LOF 4,161,
  GOF 1,354, and DN 1,134. The H200 GPU is idle after completion.
- T2-C readiness check: the current pod has no `hmmscan`, `hmmsearch`,
  `diamond`, `foldseek`, or `esm-fold` executables. ESMFold weights exist, and
  `data/interpro/pfam_residue.tsv` exists, but the real CLEAN/HMMER/Foldseek
  metric triad remains blocked until calibrated tools/databases are staged.
- Added and ran T1-A baseline head-to-head readiness:
  `Research1/scripts/23_baseline_headtohead.py`.
  Outputs:
  `Research1/results/variant_effect/baseline_headtohead_readiness_20260504.json`
  and `.md`. Available local results show ClinVar2000 AUCs SAE-LR 0.8782,
  ESM-2 LLR 0.8822, and SAE+LLR 0.9143; cancer-holdout AUCs SAE-LR 0.9079,
  ESM-2 LLR 0.8978, and SAE+LLR 0.9193. T1-A remains blocked because
  AlphaMissense, PrimateAI-3D, gMVP, and ESM-1v assets are not staged.
- Added and ran T1-E channelopathy readiness:
  `Research1/scripts/26_channelopathy.py`.
  Outputs:
  `Research1/results/variant_effect/channelopathy_readiness_20260504.json`
  and `.md`. Current local data have 16 KCNQ1 ClinVar LLR rows and SCN5A/KCNH2
  DMS files, but no staged ClinGen/literature mechanism-curated cohort or
  retrospective drug-response labels, so T1-E is blocked.
- Recorded T2-D R2 viability decision in `Research2/docs/EXPERIMENT_LOG.md`:
  no-go for a strong steering/drug-design claim in the current round; R2 should
  be framed as interpretability/layer-map plus negative steering unless T2-C is
  staged/calibrated and later shows positive shifts.

---

## H200 Continuation (2026-04-24)

- ZymCTRL CLT v2 resumed to `step_200000` on the H200 cluster:
  `/oss-pvc/zhk_zip/outputs/research2/clt_weights/zymctrl_v2/step_200000`.
  Quick eval confirms `d_clt=8192`, `k=128`; mean alive fraction is ~0.332,
  with deep layers still sparse.
- Fixed R2-D steering benchmark prompt handling: `ec_features.pkl` uses class
  keys such as `lysozyme`, but ZymCTRL generation must be conditioned on EC
  numbers such as `3.2.1.17<sep><start>`. The benchmark now maps class keys to
  EC prompts and cleans generated strings against the amino-acid alphabet.
- Added `Research2/scripts/run_r2_v2_remaining_0424.sh`, a restart-friendly
  H200 runner for the remaining R2 v2 TODOs:
  steering benchmark, structural QC, causal ablation, and cross-model
  conservation.
- Submitted 2-GPU Arena job
  `jiaotongdamoxing-zhk-zip-r2-v2-remaining-0424`. It was replaced with a
  hold-after-run wrapper (`run_r2_v2_remaining_then_hold_0424.sh`): after the
  remaining R2 experiments finish, the pod stays alive with `tail -f /dev/null`
  so the two H200 GPUs remain reserved for follow-up work. As of submission it
  is still `PENDING` because the scheduler reports insufficient free H200 GPUs.
- R1 local outputs currently cover mechanism dataset/classifier, cancer
  holdout, annotation expansion staging, and indel staging. ProteinGym is
  complete for the ESM-2 LLR baseline; full SAE scoring on all ProteinGym
  mutants remains a separate heavy implementation/run rather than a completed
  result.

---

## Platform Cleanup (2026-04-01)

- Removed the legacy container entrypoints and Arena submission wrappers that targeted generic job platforms and runtime downloads.
- Removed the older server-local launch helpers that hard-coded host-specific paths and conda environments.
- Canonical H200 entrypoints are now:
  - `Research1/scripts/06_submit_h200_sae.sh`
  - `Research1/scripts/05_run_h200_batched.sh`
  - `Research2/scripts/03_submit_h200_clt.sh`
  - `Research2/scripts/04_run_h200_clt.sh`
- Offline staging is now the only supported path:
  - code and outputs on OSS
  - models and FASTA on GPFS
  - no runtime model or dataset downloads inside training jobs

---

## Research 1: How Mutations Break Proteins

**Goal**: Train Sparse Autoencoders (SAEs) on ESM-2-3B to decompose protein representations into interpretable features, then use these features for mechanistic variant effect prediction. Target venue: Nature/Science/Cell.

**Approach**: Per-layer TopK SAEs on ESM-2-3B (2560-dim hidden states), trained on UniRef50 (67M sequences). Each SAE learns d_sae=16384 features with k=256 TopK sparsity. Downstream: annotate features with GO/InterPro/ClinVar, then predict variant pathogenicity by measuring feature activation shifts.

### Current Status: SAE Training (Phase 1 of 4)

**Training infrastructure**: L20 server (8x 48GB GPUs) for completed layers; H200 cluster (16x 80GB GPUs) for ongoing production runs.

#### Completed Layers

| Layer | Steps | FVU | Dead% | Config | Notes |
|:-----:|------:|:---:|------:|--------|-------|
| 3 | 500K/500K | 0.087 | 8.4% | k=128, no norm | EXP-005. Shallow layer - mostly local sequence features |
| 7 | 500K/500K | 0.167 | 2.0% | k=128, no norm | EXP-005. Higher FVU suggests deeper layers need more capacity |

#### In-Progress Layers (stopped on L20, awaiting H200)

| Layer | Steps | Last Ckpt | Config | Notes |
|:-----:|------:|----------:|--------|-------|
| 11 | ~350K/500K | step_30000 | k=128, no norm | FVU ~0.25, dead ~10%. Training stopped |
| 15 | ~30K/500K | step_30000 | k=128, no norm | Early stage. Training stopped |
| 19 | ~90K/500K | step_90000 | k=256, norm | Newer config with normalization |

#### Not Started (planned for H200)

Layers 23, 27, 31, 35 - these are the deepest layers where protein function features are expected to concentrate. Config: k=256, d_sae=16384, normalize_activations=true.

#### Experiment History (5 experiments, 4 failed)

1. **EXP-001** (k=64, BatchTopK, d=32K): Dead spiral to 67%. Root cause: k/d ratio too low under BatchTopK.
2. **EXP-002** (k=192, BatchTopK, d=32K): Dead stabilized at 80%, FVU=0.08. BatchTopK global competition identified as structural problem.
3. **EXP-003** (k=64, per-example, d=8K): FVU worse than EXP-002. d_sae and k both too small.
4. **EXP-004** (k=128, per-example, d=16K): Dead still 81%, nearly identical to EXP-002. **Root cause discovered: pre-TopK ReLU bug** blocking auxk gradients.
5. **EXP-005** (k=128, per-example, d=16K, ReLU fix + resampling): **Success.** Layer 3: FVU=0.087, dead=8.4%. Layer 7: FVU=0.167, dead=2.0%.

Key lessons: (1) Pre-TopK ReLU was the root cause of dead features across all failed experiments. (2) Feature resampling is essential as safety net. (3) Deeper layers need k=256 and activation normalization.

#### Preliminary Annotation Results (Layer 3)

- 15,594 alive features (95.2%)
- 12 features with F1 > 0.5 (mostly chain-level protein family recognition)
- Layer 3 is too shallow (8% depth) for functional annotations - real evaluation requires layer 23+

### H200 Deployment (Canonical)

Supported H200 workflow:
- `Research1/scripts/06_submit_h200_sae.sh` - submit an H200 pod with OSS and GPFS mounted
- `Research1/scripts/05_run_h200_batched.sh` - run SAE training batches from staged model/data paths

This workflow assumes:
- source code is staged under OSS
- model weights and FASTA are staged under GPFS
- logs and checkpoints are written back to OSS

### Remaining Phases

| Phase | Description | Status | Depends On |
|:-----:|-------------|--------|------------|
| 1 | SAE training (all 8 layers) | **In progress** | H200 deployment |
| 2 | Feature annotation (GO, InterPro, domain, active site) | Not started | Phase 1 |
| 3 | Variant effect prediction (ClinVar, gnomAD) | Not started | Phase 2 |
| 4 | Mechanism classification and paper writing | Not started | Phase 3 |

---

## Research 2: Curing the Incurable Through Interpretable Protein Drug Design

**Goal**: Use Cross-Layer Transcoders (CLTs) on decoder-only protein generators to trace circuits from disease targets to therapeutic sequences, enabling interpretable protein drug design. Target venue: Nature/Science/Cell.

**Approach**: Train windowed CLTs on protein generators (ProtGPT2, ZymCTRL, InstructProtein, ProGen2) to decompose MLP computation into sparse interpretable features. Then use attribution graphs to discover circuits responsible for specific protein properties (enzyme class, binding specificity, etc.) and steer generation toward therapeutic candidates.

### Current Status: Baseline CLT Training Complete, Resampling Fix Ready (Phase 1a of 5)

**Infrastructure**: L20 server for development/testing; H200 cluster for production training.

#### H200 Baseline Results (EXP-R2-001, d_clt=4096, 100K steps, no resampling)

| Model | Final FVU | Dead% | L0 | Key Issue |
|-------|:---------:|:-----:|:--:|-----------|
| ProtGPT2 | ~0.30 | **91.9%** | 64 | Catastrophic dead features (onset at step 10K) |
| ZymCTRL | ~0.35 | **43.0%** | ~62 | Moderate dead features |
| ProGen2-medium | ~0.32 | **29.2%** | ~57 | Best of three, still too many dead |

**Root cause**: No dead feature resampling. Features that fail to specialize early die permanently under TopK competition.

**Fixes implemented** (2026-04-02):
1. Dead feature resampling every 5000 steps (adapted from R1's proven pattern)
2. Checkpoint resume support (`--resume` flag)
3. Shuffled data ordering (was sequential)
4. Circuit discovery bug fixes (hardcoded n_layers, steering hook accumulation)

**Next step**: Re-run on H200 with resampling enabled. Expect dramatic dead feature reduction.

See `Research2/docs/EXPERIMENT_LOG.md` for full analysis.

#### Supported Models (All 5 Verified)

| Model | Architecture | Params | VRAM | Load | Generate | Activations |
|-------|:-----------:|-------:|-----:|:----:|:--------:|:-----------:|
| ProtGPT2 | GPT-2 | 738M | 1.6GB | PASS | PASS | PASS |
| ZymCTRL | GPT-2 | 738M | 1.5GB | PASS | PASS | PASS |
| InstructProtein | OPT | 1.3B | 2.6GB | PASS | PASS | PASS |
| ProGen2-medium | ProGen2 | 764M | 1.6GB | PASS | PASS | PASS |
| ProGen2-xlarge | ProGen2 | 6.4B | 13GB | PASS | PASS | PASS |

Architecture-specific hook strategies: GPT-2 hooks mlp directly; OPT hooks final_layer_norm + fc2; ProGen2 uses trust_remote_code with auto_map fix.

#### CLT Architecture: Windowed Decoder

Full cross-layer transcoder with n_layers has O(n_layers^2) decoder parameters - OOMs on 36-layer models. **Windowed CLT** restricts each feature to write to the next `window` layers only, reducing to O(n_layers * window).

- d_clt = 16384 (production), 2048 (test)
- k = 64 (TopK per token per layer)
- window = 8 layers
- Peak test VRAM: ~11GB (fits single L20)

#### End-to-End Pipeline Test (5 Phases, All Pass)

Tested on ProtGPT2 and ZymCTRL with d_clt=2048, 200 steps:

| Phase | Description | Status |
|:-----:|-------------|--------|
| 1 | Load model + extract activations | PASS |
| 2 | Train windowed CLT (200 steps) | PASS |
| 3 | Feature attribution (top features per position) | PASS |
| 4 | Circuit comparison (EC 3.2.1.17 vs 1.1.1.1) | PASS |
| 5 | Steered generation (amplify/suppress features) | PASS |

#### Code Structure

```
Research2/
  src/
    models/model_loader.py          # Multi-architecture model loader (5 models)
    training/clt_trainer.py         # Windowed CLT training pipeline
    analysis/circuit_discovery.py   # Attribution, comparison, steering
  scripts/
    00_test_models.py               # Verify all 5 models
    01_train_clt.py                 # CLT training launcher
    02_test_pipeline.py             # End-to-end 5-phase test
    03_submit_h200_clt.sh           # Submit an H200 pod
    04_run_h200_clt.sh              # Run offline CLT training in the pod
  configs/
    clt_training.yaml               # Base training config
    clt_training_h200.yaml          # H200-friendly overrides
```

### H200 Deployment (Canonical)

Supported H200 workflow:
- `Research2/scripts/03_submit_h200_clt.sh` - submit a 1-GPU H200 pod
- `Research2/scripts/04_run_h200_clt.sh` - run offline CLT training against staged model/data paths

Model discovery uses `R2_MODEL_BASE_DIR`, which now defaults to the shared H200 model root under GPFS.

### Remaining Phases

| Phase | Description | Status | Depends On |
|:-----:|-------------|--------|------------|
| 0 | Pipeline validation | **Complete** | - |
| 1a | Baseline CLT training (d_clt=4096, no resampling) | **Complete** — dead feature problem identified | - |
| 1b | CLT training with resampling (d_clt=4096, 100K steps) | **Ready to submit** | Resampling code done |
| 1c | Production CLT training (d_clt=16384, 100K+ steps) | Not started | Phase 1b validation |
| 2 | Feature annotation + circuit discovery | Not started | Phase 1 |
| 3 | Disease-target circuit analysis | Not started | Phase 2 |
| 4 | Steered therapeutic design + validation | Not started | Phase 3 |
| 5 | Paper writing | Not started | Phase 4 |

---

## Key Decisions and Design Rationale

| Decision | Rationale |
|----------|-----------|
| Per-example TopK over BatchTopK (R1) | BatchTopK causes catastrophic dead features (67-81%) due to global competition |
| Pre-TopK ReLU removal (R1) | ReLU before TopK blocks auxk gradient flow, making dead features unrecoverable |
| Windowed CLT over full CLT (R2) | Full cross-layer decoder has O(n^2) params; window=8 keeps it tractable on single GPU |
| CLTs not SAEs for R2 | R2 studies inter-layer computation (MLP circuits), not per-layer features. CLTs map between layers. |
| k=256 + normalization for deeper R1 layers | Deeper ESM-2 layers have higher activation variance; normalization + more features needed |
| H200-only offline staging | Training pods consume pre-staged code, model weights, and FASTA data instead of runtime downloads |

---

## Infrastructure

| Resource | Spec | Usage |
|----------|------|-------|
| L20 server | 8x NVIDIA L20 (48GB each, 384GB total) | R1 early training, R2 development |
| H200 cluster | 16x NVIDIA H200 (80GB each, 1.28TB total) | R1 full training, R2 production |
| Data | UniRef50 (~25GB FASTA), Swiss-Prot, GO, InterPro, ClinVar | Shared between R1 and R2 |
| Models | ESM-2-3B (R1), ProtGPT2/ZymCTRL/InstructProtein/ProGen2 (R2) | Downloaded via HF mirror |

## External Resource Staging Pass (2026-05-04 CST)

- Created `external_resources/manifests/external_resource_status_20260504.{json,md}` and `external_resources/setup_h200_external_env.sh`.
- Staged HMMER 3.4, DIAMOND 2.1.24, Foldseek AVX2, and Pfam-A HMM under `/oss-pvc/zhk_zip/biocc/external_resources`; Pfam was `hmmpress` indexed successfully with 27,481 models.
- Staged AlphaMissense `AlphaMissense_hg38.tsv.gz` and `AlphaMissense_aa_substitutions.tsv.gz` locally and on H200; T1-A is now blocked by calibration/scoring plus missing PrimateAI-3D, gMVP, and ESM-1v, not by AlphaMissense absence.
- Staged CLEAN source locally and on H200. CLEAN remains not runnable because the Google Drive pretrained CLEAN package and ESM-1b weights are not staged.
- Remaining external blockers are PrimateAI-3D gated access, dbNSFP/gMVP acquisition, ESM-1v model choice/download, CLEAN weights, and a bounded Foldseek target structure DB.

## Runnable External-Resource TODO Pass (2026-05-04 CST)

- Ran `Research1/scripts/27_alphamissense_baseline.py` on the staged AlphaMissense amino-acid substitution table. ClinVar2000 matched 1,972/2,000 variants with AUC 0.9474 [0.9377, 0.9567]; CancerHoldout101 matched 101/101 with AUC 0.9700 [0.9244, 0.9988]. T1-A now has AlphaMissense scored, but PrimateAI-3D, gMVP, and ESM-1v remain missing.
- Ran `Research2/scripts/17_pfam_scan_generated.py` on the R2 lysozyme generated sequences with the staged HMMER/Pfam resources. Steered leads had 9/10 lysozyme-like Pfam hits; steered_all was 172/200 (0.860), unsteered was 164/200 (0.820), one-sided Fisher p=0.1699. This validates domain-like generations but does not establish a significant steering lift.

## Available External Baselines and T2-C Supplement (2026-05-07 CST)

- Pulled H200 outputs for gMVP, ESM-1v, CLEAN, and Foldseek back to local summaries.
- Added `Research1/scripts/29_available_baseline_summary.py` and generated
  `Research1/results/variant_effect/available_baseline_summary_20260507.{json,md}`
  plus `available_baseline_sae_residual_cases_20260507.tsv`.
- R1 result: AlphaMissense and gMVP are stronger scalar pathogenicity baselines
  than SAE+LLR, while scalar external baselines remain weak for LOF/GOF/DN
  mechanism separation. PrimateAI-3D remains the only gated missing T1-A asset.
- Added `Research2/scripts/20_generated_metric_triad_summary.py` and generated
  `Research2/results/ec_metrics/generated_metric_triad_summary_20260507.{json,md}`.
- R2 result: selected steered lysozyme leads pass Pfam, CLEAN, and Foldseek,
  but steered_all does not beat unsteered on CLEAN and only weakly lifts Pfam.
- Prepared the T2-C real-vs-random calibration set locally with 100 real
  EC 3.2.1.17 lysozymes and 100 random UniRef50 sequences:
  `Research2/results/ec_metrics/calibration_lysozyme_20260507/`.
- Added the H200 calibration runner `Research2/scripts/run_t2c_calibration_20260507.sh`.
  The first structure pass exposed a transformers ESMFold fp16 `compute_tm`
  failure; switching the structure resume runner to fp32 fixed the issue.
- Completed the real-vs-random T2-C calibration and pulled the summaries back
  locally. All metrics separate real EC 3.2.1.17 lysozymes from random UniRef50
  controls with Cohen's d > 1:
  Pfam 4.497, CLEAN exact 6.928, CLEAN prefix 5.549, ESMFold mean pLDDT 1.786,
  ESMFold confident fraction 2.063, Foldseek top TM 1.593.

## R1 Channelopathy Label Curation (2026-05-07 CST)

- Curated research-use mechanism and drug-response labels for KCNQ1, SCN5A,
  KCNH2, and CACNA1C using parallel GPT-5.5 xhigh subagents, with each worker
  producing traceable TSV rows plus source notes.
- Added `Research1/scripts/30_merge_channelopathy_labels.py` to validate,
  normalize, deduplicate, and split the curation outputs.
- Consolidated output:
  `Research1/data/channelopathy/channelopathy_mechanism_labels.tsv` with 119
  rows: 74 high-confidence mechanism rows and 45 low-confidence ClinVar/ClinGen
  candidate/source-index rows.
- Positive mechanism subset:
  `Research1/data/channelopathy/channelopathy_mechanism_positive_labels.tsv`
  with 74 rows: KCNQ1 23, SCN5A 25, KCNH2 19, CACNA1C 7.
- Drug-response subset:
  `Research1/data/channelopathy/channelopathy_drug_response_labels.tsv` with
  40 rows. These are sparse research labels and should be analyzed separately.
- Updated `Research1/scripts/26_channelopathy.py` and generated
  `Research1/results/variant_effect/channelopathy_readiness_20260507.{json,md}`;
  T1-E is no longer blocked by missing curated labels, but still requires R1
  scoring and mechanism-concordance evaluation.
- Added `Research1/scripts/31_channelopathy_concordance.py` and ran the
  supported missense subset on H200 with mutation-centered ESM-2 windows for
  long channel proteins.
- T1-E scoring result:
  `Research1/results/variant_effect/channelopathy_concordance_20260507.{json,md}`.
  Of 74 high-confidence curated labels, 69 missense variants were scoreable,
  64 LOF/GOF/DN rows were evaluated, and concordance was 0.625 with macro-F1
  0.444. This misses the TODO target of >=80%; the dominant failure mode is
  DN variants being predicted as LOF.
