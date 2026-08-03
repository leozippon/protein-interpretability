# Research Status Log

Last updated: 2026-07-29 CST

Append-only. Historical entries record the paths and framings that existed when they were written and are not normalized to the current layout.

---

## Opus M-1 Triplet Characterization (2026-05-14)

- Implemented `scripts/35_triplet_characterization.py` for the final Opus-requested R2 experiment after failed biological/downstream gates.
- Ran the experiment on the current 1-GPU H200 pod over 700 sequences and 38 conserved triplets with 200 permutations per test.
- Pulled local outputs to `results/circuit_analysis/triplet_characterization_20260514/` and runtime log `logs/runtime/r2_triplet_characterization_20260514_v2.log`.
- Final result: PASS, 37 / 38 triplets categorized. Assigned categories: 21 k-mer, 14 positional, 2 high-norm, 1 unknown.
- Corrected the BPE-boundary test before the final logged run: it now tests one-sided positive token-boundary enrichment instead of two-sided boundary association. Final category counts are unchanged.
- English summary for Opus review: `archive/conversation_history/OPUS_M1_RESULTS_20260514.md`.

---

## Opus 2026-05-15 Plan Execution

- Implemented and ran M-2 synthesis: `scripts/36_triplet_synthesis.py`.
- Final local output from the M-1 `n_perm=2000` tables: `results/circuit_analysis/triplet_synthesis_20260515_nperm2000/`.
- M-2 result: 37 / 38 triplets pass at least one characterization test; 21 / 38 pass three or more tests; attention-sink subset is T011/T018/T023/T025.
- Completed M-1 final rerun on H200 with `n_perm=2000` and `--top-position-rows 100` to `results/circuit_analysis/triplet_characterization_20260515_nperm2000/`. Final result: PASS, 37 / 38 categorized by the legacy single-category summary; assigned counts are 17 positional, 17 k-mer, 3 high-norm and 1 unknown.
- Updated R2 manuscript to the cross-model statistical-conservation framing with final `n_perm=2000` synthesis counts, and compiled both R1 and R2 drafts successfully.
- Added `manuscripts/README.md` as a top-level evidence index for the main R1/R2 numeric claims.
- Added CC-BY-4.0 license notes to `data/indelmissense/v1/`.
- The temporary Hangzhou jump-host SSH outage recovered; final H200 outputs were pulled successfully.
- Released the idle Opus H200 hold pod after confirming no active experiment process and 0 MiB / 0% GPU use; no running BioCC GPU pod remains.
- Later submitted a new 1-GPU H200 hold pod at the user's request: `jiaotongdamoxing-zhk-zip-hold-1gpu-0513b-master-0`, running on `i-d5cvmv6heob1nidq4ujg` / `192.168.20.204` with 0 MiB / 0% GPU use at creation time.
- English execution packet: `archive/conversation_history/OPUS_PLAN_EXECUTION_20260515.md`.

---

## H200 Single-GPU Continuation (2026-04-25)

- Per the request to leave two H200 GPUs available for other users, submitted a 1-GPU Arena job: `jiaotongdamoxing-zhk-zip-r2-v2-1gpu-0425`.
- The job is running on `i-d5cvmv6heob1nidq4ujg` / `192.168.20.204` with one requested H200. Kubernetes active allocation after scheduling is `192.168.20.203: 8/8` and `192.168.20.204: 6/8`, leaving two H200 GPUs free on `192.168.20.204` for Arena scheduling.
- Synced the R2 remaining runner to be single-GPU-safe: structural QC now uses `STRUCTURAL_CUDA_DEVICE=0` by default instead of hard-coded device 1, and uses `STRUCTURAL_DTYPE=fp32` by default for the local ESMFold path.
- Entry command: `bash scripts/run_r2_v2_remaining_then_hold_0424.sh`. The wrapper keeps the pod alive with `tail -f /dev/null` after the queue exits.
- Runtime log prefix: `/gpfs/jiaotongdamoxing/zhk_zip/biocc/paper_r2_nature_mi/logs/runtime/*_20260425_r2_v2_1gpu.log`.

---

## TODO Completion Pass (2026-04-26)

- Fixed remote zero-byte scripts on OSS: `scripts/15_cross_model_conservation.py` and `r1_encoder_interpretability_benchmark/scripts/20_indel_extension.py`.
- R2-H cross-model conservation was rerun successfully on the 1-GPU H200 pod. Latest output now points to: `/gpfs/jiaotongdamoxing/zhk_zip/biocc/paper_r2_nature_mi/results/circuit_analysis/cross_model_conservation_v2_20260426_todo_completion.json`.
- R2-A and R2-B v2 checkpoints are present at `step_200000`: `/oss-pvc/zhk_zip/outputs/research2/clt_weights/protgpt2_v2/step_200000` and `/oss-pvc/zhk_zip/outputs/research2/clt_weights/zymctrl_v2/step_200000`.
- Added `r1_encoder_interpretability_benchmark/scripts/21_proteingym_sae_followup.py` to fill the ProteinGym SAE/ensemble fields left pending by the original LLR-only benchmark.
- Started the full ProteinGym SAE follow-up in the current 1-GPU H200 pod: PID `2805`, log `/oss-pvc/zhk_zip/biocc/benchmark_encoder/logs/runtime/r1d_proteingym_sae_followup_20260426.log`, output `/oss-pvc/zhk_zip/biocc/benchmark_encoder/results/variant_effect/proteingym_benchmark_sae_latest.json`. The job is resume-friendly and writes after each assay.

## Diagnostic / Strengthening Pass (2026-04-29 CST)

- R1-D ProteinGym SAE follow-up completed all 217 assays. Added `r1_encoder_interpretability_benchmark/scripts/22_proteingym_result_diagnostics.py` and ran an assay-level diagnostic on the remote output: `/oss-pvc/zhk_zip/biocc/benchmark_encoder/results/variant_effect/proteingym_benchmark_sae_diagnostics_20260429.json`. Usable assays: 214/217. Mean Spearman: LLR 0.4341, SAE -0.2314, negated-SAE 0.2314, ensemble 0.1993. The ensemble beats LLR on only 8.4% of usable assays, so the ProteinGym issue is systematic rather than a small number of outliers.
- Fixed `scripts/15_cross_model_conservation.py`: feature-profile Pearson correlations now divide by `n` to match `np.std(ddof=0)`, preventing invalid correlations greater than 1.0 on small cohorts.
- Reran R2-H after the correlation fix: `/gpfs/jiaotongdamoxing/zhk_zip/biocc/paper_r2_nature_mi/results/circuit_analysis/cross_model_conservation_v2_20260429_corrfix.json`. The latest symlink now points to this corrected file. Max absolute feature correlation is 0.9975; CKA values remain [0.6216, 0.7935, 0.4198].
- Started a small R2-D steering-strength sweep on the existing 1-GPU H200 pod for the classes that had the largest positive but non-significant baseline effects (`lipase`, `kinase`, `carbonic_anh`): multiplier 5.0, top-5 features/layer, layers L3/L12/L30, n=64 per condition. Log: `/gpfs/jiaotongdamoxing/zhk_zip/biocc/paper_r2_nature_mi/logs/runtime/r2d_steering_sweep_20260429_lipase_kinase_ca_m5_n64.log`. Output: `/gpfs/jiaotongdamoxing/zhk_zip/biocc/paper_r2_nature_mi/results/steering_benchmark/zymctrl_v2_purity_sweep_20260429_lipase_kinase_ca_m5_n64.json`. Results: lipase +0.072 (p=0.1066), kinase +0.026 (p=0.6584), carbonic_anh -0.047 (p=0.2077), 0/3 significant positive. Increasing steering strength helps lipase numerically but does not rescue the R2-D steering claim by itself.

## TODO_NEXT Tier-0 Execution (2026-04-29 CST)

- T0-A R2 hook sanity diagnostic completed and passed for ZymCTRL v2 and ProGen2-medium: `results/diagnostics/hook_sanity_20260429.json`. ZymCTRL: multiplier=10 max logit shift 12.93, multiplier=0 shift 0.885. ProGen2-medium: multiplier=10 shift 3.938, multiplier=0 shift 1.250. Conclusion: the MLP hook path can move logits; the earlier all-zero causal-ablation result is not explained by a universally disconnected hook.
- T0-C `ec_features.pkl` provenance audit completed and passed: `results/diagnostics/ec_features_provenance_20260429.json`. The H200 file has 8 EC classes, 36 layers, d_clt=8192 for every layer, and matches the ZymCTRL v2 checkpoint dimensionality. The local `results/circuit_analysis/zymctrl/ec_features.pkl` was refreshed from H200 and validated as an 8192-dimensional pickle.
- T0-D protein-level mechanism holdout completed: `r1_encoder_interpretability_benchmark/results/variant_effect/mechanism_classifier_results_t0_protein_holdout_20260429.json`. The default `mechanism_classifier_results.json` now contains both variant-level and protein-level CV; the previous variant-CV-only result is backed up as `mechanism_classifier_results_variantcv_legacy_20260429.json`. Variant-level SAE macro-AUC is 0.7471; protein-level SAE macro-AUC is 0.5161. This fails the TODO_NEXT 0.7 threshold, so the R1 headline must be downgraded from "predicts mechanism" to "variant-level mechanism signal does not generalize cleanly across proteins yet."
- T0-B signed ProteinGym rerun completed on the current 1-GPU H200 pod at 2026-05-01 21:51 CST. Final remote output: `/oss-pvc/zhk_zip/biocc/benchmark_encoder/results/variant_effect/proteingym_benchmark_sae_signed_20260429_signed.json`. Local copies: `r1_encoder_interpretability_benchmark/results/variant_effect/proteingym_benchmark_sae_signed_20260429_signed.json`, `r1_encoder_interpretability_benchmark/results/variant_effect/proteingym_benchmark_sae_signed_diagnostics_20260503.json`, and `r1_encoder_interpretability_benchmark/logs/runtime/t0b_proteingym_signed_20260429_signed.log`. Results over 214 usable assays: LLR mean rho 0.4341, raw SAE -0.2314, legacy plus ensemble 0.1993, sign-corrected ensemble 0.4047 (95% bootstrap CI [0.3818, 0.4282]). The sign fix rescues most of the cancellation but still does not beat LLR on average; signed ensemble beats LLR on 33.6% of usable assays, below the TODO_NEXT 40% threshold. Because T0-D protein-level mechanism CV failed, the classifier-based ProteinGym score was not treated as a defensible paper claim.

## TODO_NEXT Tier-1 / Tier-2 Execution (2026-05-03 CST)

- Added and ran T1-B preliminary mechanism feature audit: `r1_encoder_interpretability_benchmark/scripts/24_mechanism_feature_audit.py`. Outputs: `r1_encoder_interpretability_benchmark/results/variant_effect/mechanism_feature_audit_20260503.json` and `r1_encoder_interpretability_benchmark/results/variant_effect/mechanism_feature_audit_20260503.md`. The audit contains 150 rows (3 classes x 5 layers x top-10 features) with coefficient, feature kind, best annotation, Pfam/domain proxy, GO/functional proxy, binding proxy, and a conservative manual interpretation note. Current annotation pkls still lack `firing_positions`, so max-activating sequence inspection remains blocked on T1-D.
- Added and completed T2-A direct-effect feature selection: `scripts/16_direct_effect_features.py`. Remote outputs: `/oss-pvc/zhk_zip/biocc/paper_r2_nature_mi/results/circuit_analysis/zymctrl/direct_effect_features_v2.pkl` and `/oss-pvc/zhk_zip/biocc/paper_r2_nature_mi/results/circuit_analysis/zymctrl/direct_effect_features_v2_summary_20260503.json`. Acceptance shape is `(8, 36, 10)` for 8 EC classes, 36 ZymCTRL layers, and top-10 direct-effect features per class/layer.
- Implemented T2-B TopK-aware on-manifold steering in `src/analysis/circuit_discovery.py`: the hook now recomputes CLT pre-activations, applies the feature intervention, re-applies TopK, and replaces the CLT-explained same-layer MLP component instead of adding a single off-manifold decoder vector.
- Updated `scripts/11_steering_benchmark.py` to optionally consume the T2-A direct-effect pkl. A lysozyme smoke test passed and wrote: `/oss-pvc/zhk_zip/biocc/paper_r2_nature_mi/results/steering_benchmark/zymctrl_v2_onmanifold_direct_smoke_20260503.json`.
- Completed the full T2-B direct-effect/on-manifold steering benchmark in the current 1-GPU H200 pod. Log: `/oss-pvc/zhk_zip/biocc/paper_r2_nature_mi/logs/runtime/t2b_onmanifold_direct_steering_20260503.log`, output `/oss-pvc/zhk_zip/biocc/paper_r2_nature_mi/results/steering_benchmark/zymctrl_v2_onmanifold_direct_20260503.json`. Local copies were pulled to `logs/runtime/t2b_onmanifold_direct_steering_20260503.log` and `results/steering_benchmark/zymctrl_v2_onmanifold_direct_20260503.json`. Result: 0/8 EC classes showed significant positive steering. Largest positive effects were kinase +0.107 (95% CI [+0.000, +0.210], p=0.0532) and lipase +0.057 (95% CI [-0.007, +0.122], p=0.0913), but neither passed the significance criterion. The current H200 GPU is idle after completion.
- Started T1-D annotation rerun with firing positions on the current 1-GPU H200 pod. Code changes: `r1_encoder_interpretability_benchmark/src/analysis/feature_annotation.py` now stores optional `firing_positions` and `top_firing_examples`, and `r1_encoder_interpretability_benchmark/scripts/04_analyze_our_sae.py` now supports `--save-firing-positions`, `--max-firing-positions-per-feature`, `--checkpoint-root`, and `--out-prefix`. Runner: `r1_encoder_interpretability_benchmark/scripts/run_t1d_annotation_firing_20260503.sh`. Remote PID 8318, log: `/oss-pvc/zhk_zip/biocc/benchmark_encoder/logs/runtime/t1d_annotation_firing_20260503.log`. The runner processes layers 19/23/27/31/35 with 1000 Swiss-Prot proteins, writes firing-enabled base annotation pkls, backs up previous no-firing pkls, and then runs `19_expand_annotation.py` to `r1_encoder_interpretability_benchmark/results/annotation_alignment/expanded_summary_firing_20260503.json`.
- T1-D status update: all five firing-enabled annotation pkls completed. `19_expand_annotation.py` initially exposed a filtering bug (`Swiss-Prot universe: 0 proteins`) because the Swiss-Prot cache stores `ProteinAnnotation.accession`, not `uniprot_id` dict entries. Fixed `r1_encoder_interpretability_benchmark/scripts/19_expand_annotation.py` to collect the accession universe directly from `firing_positions`, stopped the bad expansion process, and restarted only the expansion stage as PID 9635. Fixed log: `/oss-pvc/zhk_zip/biocc/benchmark_encoder/logs/runtime/t1d_expand_annotation_fixed_20260503.log`. The fixed run correctly reports a 1000-protein firing-position universe.
- T1-D completed. Local summaries/logs: `r1_encoder_interpretability_benchmark/results/annotation_alignment/expanded_summary_firing_20260503.json` and `r1_encoder_interpretability_benchmark/logs/runtime/t1d_expand_annotation_fixed_20260503.log`. Expansion increased KNOWN feature counts substantially at all layers: L19 146→381, L23 135→301, L27 72→163, L31 49→86, L35 32→45. This unlocks firing-position-based feature audit, but it fails the T1-D acceptance threshold for L35 (target ≥60; observed 45).
- T1-B final firing-position feature audit completed locally after pulling the firing-enabled expanded annotation pkls from H200. Outputs: `r1_encoder_interpretability_benchmark/results/variant_effect/mechanism_feature_audit_firing_20260504.json` and `r1_encoder_interpretability_benchmark/results/variant_effect/mechanism_feature_audit_firing_20260504.md`. The audit has 150/150 rows with firing positions and top residue examples; 142/150 rows also have expanded labels. It is figure-planning evidence, not manual biological validation.
- Added `r1_encoder_interpretability_benchmark/scripts/25_indel_mechanism.py` for T1-C. Local preparation produced 6,649 reconstructable pathogenic/benign indel records at `r1_encoder_interpretability_benchmark/results/variant_effect/indel_records_supported_20260504.jsonl`; the missense mechanism classifier was cached as pure numpy LR parameters at `r1_encoder_interpretability_benchmark/results/variant_effect/indel_mechanism_classifier_20260504.pkl`. A 2-record H200 smoke test passed and wrote `/oss-pvc/zhk_zip/biocc/benchmark_encoder/results/variant_effect/indel_mechanism_predictions_smoke_20260504.jsonl`.
- Completed the full T1-C indel mechanism-transfer run on the current 1-GPU H200 pod. Log: `/oss-pvc/zhk_zip/biocc/benchmark_encoder/logs/runtime/t1c_indel_mechanism_20260504.log`; output: `/oss-pvc/zhk_zip/biocc/benchmark_encoder/results/variant_effect/indel_mechanism_predictions_20260504.jsonl`. Local copies were pulled to `r1_encoder_interpretability_benchmark/logs/runtime/t1c_indel_mechanism_20260504.log`, `r1_encoder_interpretability_benchmark/results/variant_effect/indel_mechanism_predictions_20260504.jsonl`, and `r1_encoder_interpretability_benchmark/results/variant_effect/indel_mechanism_predictions_20260504_summary.json`. Final results: 6,649/6,649 records predicted; pathogenicity AUC from the indel SAE damage score is 0.7735. Predicted mechanism counts are LOF 4,161, GOF 1,354, and DN 1,134. The H200 GPU is idle after completion.
- T2-C readiness check: the current pod has no `hmmscan`, `hmmsearch`, `diamond`, `foldseek`, or `esm-fold` executables. ESMFold weights exist, and `data/interpro/pfam_residue.tsv` exists, but the real CLEAN/HMMER/Foldseek metric triad remains blocked until calibrated tools/databases are staged.
- Added and ran T1-A baseline head-to-head readiness: `r1_encoder_interpretability_benchmark/scripts/23_baseline_headtohead.py`. Outputs: `r1_encoder_interpretability_benchmark/results/variant_effect/baseline_headtohead_readiness_20260504.json` and `.md`. Available local results show ClinVar2000 AUCs SAE-LR 0.8782, ESM-2 LLR 0.8822, and SAE+LLR 0.9143; cancer-holdout AUCs SAE-LR 0.9079, ESM-2 LLR 0.8978, and SAE+LLR 0.9193. T1-A remains blocked because AlphaMissense, PrimateAI-3D, gMVP, and ESM-1v assets are not staged.
- Added and ran T1-E channelopathy readiness: `r1_encoder_interpretability_benchmark/scripts/26_channelopathy.py`. Outputs: `r1_encoder_interpretability_benchmark/results/variant_effect/channelopathy_readiness_20260504.json` and `.md`. Current local data have 16 KCNQ1 ClinVar LLR rows and SCN5A/KCNH2 DMS files, but no staged ClinGen/literature mechanism-curated cohort or retrospective drug-response labels, so T1-E is blocked.
- Recorded T2-D R2 viability decision in `docs/EXPERIMENT_LOG.md`: no-go for a strong steering/drug-design claim in the current round; R2 should be framed as interpretability/layer-map plus negative steering unless T2-C is staged/calibrated and later shows positive shifts.

---

## H200 Continuation (2026-04-24)

- ZymCTRL CLT v2 resumed to `step_200000` on the H200 cluster: `/oss-pvc/zhk_zip/outputs/research2/clt_weights/zymctrl_v2/step_200000`. Quick eval confirms `d_clt=8192`, `k=128`; mean alive fraction is ~0.332, with deep layers still sparse.
- Fixed R2-D steering benchmark prompt handling: `ec_features.pkl` uses class keys such as `lysozyme`, but ZymCTRL generation must be conditioned on EC numbers such as `3.2.1.17<sep><start>`. The benchmark now maps class keys to EC prompts and cleans generated strings against the amino-acid alphabet.
- Added `scripts/run_r2_v2_remaining_0424.sh`, a restart-friendly H200 runner for the remaining R2 v2 TODOs: steering benchmark, structural QC, causal ablation, and cross-model conservation.
- Submitted 2-GPU Arena job `jiaotongdamoxing-zhk-zip-r2-v2-remaining-0424`. It was replaced with a hold-after-run wrapper (`run_r2_v2_remaining_then_hold_0424.sh`): after the remaining R2 experiments finish, the pod stays alive with `tail -f /dev/null` so the two H200 GPUs remain reserved for follow-up work. As of submission it is still `PENDING` because the scheduler reports insufficient free H200 GPUs.
- R1 local outputs currently cover mechanism dataset/classifier, cancer holdout, annotation expansion staging, and indel staging. ProteinGym is complete for the ESM-2 LLR baseline; full SAE scoring on all ProteinGym mutants remains a separate heavy implementation/run rather than a completed result.

---

## Platform Cleanup (2026-04-01)

- Removed the legacy container entrypoints and Arena submission wrappers that targeted generic job platforms and runtime downloads.
- Removed the older server-local launch helpers that hard-coded host-specific paths and conda environments.
- Canonical H200 entrypoints are now:
  - `r1_encoder_interpretability_benchmark/scripts/submit_h200_sae.sh`
  - `r1_encoder_interpretability_benchmark/scripts/run_h200_batched.sh`
  - `scripts/03_submit_h200_clt.sh`
  - `scripts/04_run_h200_clt.sh`
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
- `r1_encoder_interpretability_benchmark/scripts/submit_h200_sae.sh` - submit an H200 pod with OSS and GPFS mounted
- `r1_encoder_interpretability_benchmark/scripts/run_h200_batched.sh` - run SAE training batches from staged model/data paths

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

See `docs/EXPERIMENT_LOG.md` for full analysis.

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
- `scripts/03_submit_h200_clt.sh` - submit a 1-GPU H200 pod
- `scripts/04_run_h200_clt.sh` - run offline CLT training against staged model/data paths

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
| L20 server | 8x NVIDIA L20, 45 GiB reported each (~360 GiB total) | Validation, light and CPU-only workloads |
| H200 pod | 4x NVIDIA H200, 140 GiB reported each (~560 GiB total) | All full-scale campaigns: R1 training, R2 production |

Specs are driver-reported, verified 2026-07-28, not vendor nominal — the L20's nominal 48 GB and the H200's 80 GB variant figure both previously appeared here and both overstate addressable memory. Cluster-wide allocation reads 100% even when our own pod's cards are idle; confirm capacity with `nvidia-smi` inside the pod. See the project CLAUDE.md for access and the pod-name handling rule.
| Data | UniRef50 (~25GB FASTA), Swiss-Prot, GO, InterPro, ClinVar | Shared between R1 and R2 |
| Models | ESM-2-3B (R1), ProtGPT2/ZymCTRL/InstructProtein/ProGen2 (R2) | Downloaded via HF mirror |

## External Resource Staging Pass (2026-05-04 CST)

- Created `external_resources/manifests/external_resource_status_20260504.{json,md}` and `external_resources/setup_h200_external_env.sh`.
- Staged HMMER 3.4, DIAMOND 2.1.24, Foldseek AVX2, and Pfam-A HMM under `/oss-pvc/zhk_zip/biocc/external_resources`; Pfam was `hmmpress` indexed successfully with 27,481 models.
- Staged AlphaMissense `AlphaMissense_hg38.tsv.gz` and `AlphaMissense_aa_substitutions.tsv.gz` locally and on H200; T1-A is now blocked by calibration/scoring plus missing PrimateAI-3D, gMVP, and ESM-1v, not by AlphaMissense absence.
- Staged CLEAN source locally and on H200. CLEAN remains not runnable because the Google Drive pretrained CLEAN package and ESM-1b weights are not staged.
- Remaining external blockers are PrimateAI-3D gated access, dbNSFP/gMVP acquisition, ESM-1v model choice/download, CLEAN weights, and a bounded Foldseek target structure DB.

## Runnable External-Resource TODO Pass (2026-05-04 CST)

- Ran `r1_encoder_interpretability_benchmark/scripts/27_alphamissense_baseline.py` on the staged AlphaMissense amino-acid substitution table. ClinVar2000 matched 1,972/2,000 variants with AUC 0.9474 [0.9377, 0.9567]; CancerHoldout101 matched 101/101 with AUC 0.9700 [0.9244, 0.9988]. T1-A now has AlphaMissense scored, but PrimateAI-3D, gMVP, and ESM-1v remain missing.
- Ran `scripts/17_pfam_scan_generated.py` on the R2 lysozyme generated sequences with the staged HMMER/Pfam resources. Steered leads had 9/10 lysozyme-like Pfam hits; steered_all was 172/200 (0.860), unsteered was 164/200 (0.820), one-sided Fisher p=0.1699. This validates domain-like generations but does not establish a significant steering lift.

## Available External Baselines and T2-C Supplement (2026-05-07 CST)

- Pulled H200 outputs for gMVP, ESM-1v, CLEAN, and Foldseek back to local summaries.
- Added `r1_encoder_interpretability_benchmark/scripts/29_available_baseline_summary.py` and generated `r1_encoder_interpretability_benchmark/results/variant_effect/available_baseline_summary_20260507.{json,md}` plus `available_baseline_sae_residual_cases_20260507.tsv`.
- R1 result: AlphaMissense and gMVP are stronger scalar pathogenicity baselines than SAE+LLR, while scalar external baselines remain weak for LOF/GOF/DN mechanism separation. PrimateAI-3D remains the only gated missing T1-A asset.
- Added `scripts/20_generated_metric_triad_summary.py` and generated `results/ec_metrics/generated_metric_triad_summary_20260507.{json,md}`.
- R2 result: selected steered lysozyme leads pass Pfam, CLEAN, and Foldseek, but steered_all does not beat unsteered on CLEAN and only weakly lifts Pfam.
- Prepared the T2-C real-vs-random calibration set locally with 100 real EC 3.2.1.17 lysozymes and 100 random UniRef50 sequences: `results/ec_metrics/calibration_lysozyme_20260507/`.
- Added the H200 calibration runner `scripts/run_t2c_calibration_20260507.sh`. The first structure pass exposed a transformers ESMFold fp16 `compute_tm` failure; switching the structure resume runner to fp32 fixed the issue.
- Completed the real-vs-random T2-C calibration and pulled the summaries back locally. All metrics separate real EC 3.2.1.17 lysozymes from random UniRef50 controls with Cohen's d > 1: Pfam 4.497, CLEAN exact 6.928, CLEAN prefix 5.549, ESMFold mean pLDDT 1.786, ESMFold confident fraction 2.063, Foldseek top TM 1.593.

## R1 Channelopathy Label Curation (2026-05-07 CST)

- Curated research-use mechanism and drug-response labels for KCNQ1, SCN5A, KCNH2, and CACNA1C using parallel GPT-5.5 xhigh subagents, with each worker producing traceable TSV rows plus source notes.
- Added `r1_encoder_interpretability_benchmark/scripts/30_merge_channelopathy_labels.py` to validate, normalize, deduplicate, and split the curation outputs.
- Consolidated output: `r1_encoder_interpretability_benchmark/data/channelopathy/channelopathy_mechanism_labels.tsv` with 119 rows: 74 high-confidence mechanism rows and 45 low-confidence ClinVar/ClinGen candidate/source-index rows.
- Positive mechanism subset: `r1_encoder_interpretability_benchmark/data/channelopathy/channelopathy_mechanism_positive_labels.tsv` with 74 rows: KCNQ1 23, SCN5A 25, KCNH2 19, CACNA1C 7.
- Drug-response subset: `r1_encoder_interpretability_benchmark/data/channelopathy/channelopathy_drug_response_labels.tsv` with 40 rows. These are sparse research labels and should be analyzed separately.
- Updated `r1_encoder_interpretability_benchmark/scripts/26_channelopathy.py` and generated `r1_encoder_interpretability_benchmark/results/variant_effect/channelopathy_readiness_20260507.{json,md}`; T1-E is no longer blocked by missing curated labels, but still requires R1 scoring and mechanism-concordance evaluation.
- Added `r1_encoder_interpretability_benchmark/scripts/31_channelopathy_concordance.py` and ran the supported missense subset on H200 with mutation-centered ESM-2 windows for long channel proteins.
- T1-E scoring result: `r1_encoder_interpretability_benchmark/results/variant_effect/channelopathy_concordance_20260507.{json,md}`. Of 74 high-confidence curated labels, 69 missense variants were scoreable, 64 LOF/GOF/DN rows were evaluated, and concordance was 0.625 with macro-F1 0.444. This misses the TODO target of >=80%; the dominant failure mode is DN variants being predicted as LOF.

## PrimateAI-3D Final Disposition and Analysis Export (2026-05-10 CST)

- PrimateAI-3D is treated as unavailable because gated access could not be obtained for this analysis round.
- T1-A is finalized over accessible baselines only: SAE-LR, ESM-2 LLR, SAE+LLR, AlphaMissense, gMVP, and ESM-1v.
- Added `r1_encoder_interpretability_benchmark/results/variant_effect/t1a_final_available_baselines_no_primateai_20260510.{json,md}` as the final bookkeeping artifact for the no-PrimateAI analysis pass.
- Created `analysis_exports/20260510_no_primateai/` to collect necessary result summaries, tables, logs, and project docs for local analysis. Large checkpoints, model weights, and large binary perturbation/signature pickles are intentionally excluded.

## Final No-Wet-Lab Plan Execution Start (2026-05-11 CST)

- Located Opus' final no-wet-lab strategy in `archive/conversation_history/OPUS_FINAL_PLAN_20260511.md`.
- Added and ran `ops/final_plan_readiness_audit.py`. Outputs: `results/final_plan_readiness/final_plan_readiness_20260511.{json,md}`.
- Readiness result:
  - F-T1-1 indel scale-up is target-size limited. The current protein-HGVS reconstruction has 185,655 ClinVar indel rows, 18,897 supported reconstructed rows across all labels, and only 6,649 binary-label, length-compatible records ready for the existing scorer. The existing 6,649-record prediction run already covers that ready subset.
  - F-T1-2 indel competitor comparison is not ready locally: CADD, REVEL, and SpliceAI indel score files were not found.
  - F-T2-1 R2 five-task downstream evaluation is not ready as written: EC/Pfam resources are partially ready, but CB513, DeepLoc, and FireProtDB/ProTherm are not staged locally.
- Added and ran `r1_encoder_interpretability_benchmark/scripts/34_alphamissense_ensemble.py` for F-T1-3. Outputs: `r1_encoder_interpretability_benchmark/results/variant_effect/alphamissense_sae_ensemble_20260511.{json,md}` and `alphamissense_sae_ensemble_predictions_20260511.tsv`.
- F-T1-3 result under gene-grouped CV: AlphaMissense AUC 0.9474, SAE-LR group-CV AUC 0.7559, AM+SAE stack AUC 0.8542, AM+SAE z-sum AUC 0.9210. All SAE+AM deltas versus AlphaMissense are negative under group bootstrap. Gate decision: do not claim ensemble improvement over AlphaMissense; keep SAE as an interpretation/residual diagnostic layer only.
- Added `archive/conversation_history/FINAL_PLAN_EXECUTION_20260511.md` as the English execution log for this pass.
- Completed F-T0 manuscript reframing for both split drafts.
  - R1 manuscript now includes the negative AlphaMissense+SAE ensemble gate, accessible gMVP/ESM-1v baselines, evidence-path comments, and a bounded indel statement over the 6,649 binary-label reconstructable records.
  - R2 manuscript is reframed from controlled generation to sparse circuit-latent diagnostics, updates the steering table to the TopK-aware direct-effect benchmark, records hook sanity as passing, and adds the calibrated Pfam/CLEAN/ESMFold/Foldseek lysozyme metric triad.
  - Wet-lab validation is now described only as future work, not as a dependency for either current manuscript.
- Released the idle 1-GPU H200 pod at the user's request, then submitted a new 1-GPU hold job `jiaotongdamoxing-zhk-zip-final-1gpu-0511` on `192.168.20.203`.
- Ran two resource-ready R2 pilot experiments on the new 1-GPU pod:
  - F-T2-3 metric-stack quality detection: `results/ec_metrics/quality_detection_from_existing_metrics_20260511.{json,md}`. Logistic CV AUC 0.9649 [0.9375, 0.9868]; Foldseek top TM alone AUC 0.9571.
  - F-T2-1 resource-ready CLT representation pilot: `results/ec_metrics/clt_representation_lysozyme_probe_20260511.{json,md}`. ZymCTRL raw hidden AUC 0.9926 [0.9822, 0.9994], pooled CLT AUC 0.9934 [0.9820, 0.9996], delta +0.0008 on the real-lysozyme vs random-UniRef50 calibration task.
  - Interpretation: the metric stack and CLT features work on this narrow calibration task, but this does not replace the planned five-task R2 downstream benchmark.

## Continued Final-Plan Runs (2026-05-11 CST)

- Continued execution of the no-wet-lab Opus plan on the 1-GPU H200 pod `jiaotongdamoxing-zhk-zip-final-1gpu-0511`.
- Completed additional F-T2-1 cross-model lysozyme-vs-random representation probes:
  - ProtGPT2 raw hidden AUC 0.9899 [0.9785, 0.9985], pooled CLT AUC 0.9932 [0.9855, 0.9985], delta +0.0033.
  - ProGen2-medium raw hidden AUC 0.9994 [0.9980, 1.0000], pooled CLT AUC 0.9982 [0.9952, 1.0000], delta -0.0012.
- Queued and started F-T2-2 resource-ready three-model conservation pilot: ProtGPT2, ZymCTRL, and ProGen2-medium on 100 shared lysozyme/random calibration sequences. This is a pilot, not the final 10k-sequence universal atlas gate.
- The first F-T2-2 pilot completed. It used the first 100 calibration records, which were all real lysozymes; therefore it is a same-family conservation pilot, not a balanced real/random atlas. It found 9 exact three-model feature triplets at abs(r) >= 0.90 from top-30 pairwise matches.
- Started a balanced F-T2-2 follow-up over 200 interleaved calibration records (100 real lysozymes + 100 random UniRef50 controls), with top-100 pairwise matches and feature pool size 2048. Output target: `results/circuit_analysis/universal_atlas_balanced200_summary_20260511.{json,md}`.
- Added `scripts/26_universal_atlas_summary.py` to summarize pairwise cross-model feature matches and exact three-model feature triplets.
- Completed the bounded F-T1-7 local IndelMissense static benchmark artefact at `data/indelmissense/v1/`: 6,649 reconstructable binary ClinVar protein indels; train/validation/test split 5,356/653/640; current damage-score AUC 0.7735.
- No Opus decision was needed for the above. An Opus-level decision is only needed if the project re-opens the original ~80k full-scale indel target, because that requires transcript-aware frameshift reconstruction/mapping rather than simply more GPU time.

## R2 Universal Atlas Follow-up (2026-05-12 CST)

- Completed the balanced-200 F-T2-2 rerun on 100 real lysozymes plus 100 random UniRef50 controls:
  - top-100 / pool-2048: 12 exact three-model triplets at abs(r) >= 0.90.
  - top-300 / pool-4096: 38 triplets at abs(r) >= 0.90, 30 at >= 0.95, and 8 at >= 0.98.
- Added `scripts/27_universal_atlas_null_control.py`. The permutation null control produced 0 triplets in 3 shuffled replicates at thresholds 0.90, 0.95, and 0.98, while the observed counts were 38/30/8.
- Added `scripts/28_universal_triplet_source_selectivity.py`. Source-selectivity annotation found that all 38 balanced-200 wide-match triplets are weak/source-mixed for real-lysozyme vs random-UniRef50 discrimination.
- Interpretation: the conservation count gate is met in a 200-sequence pilot and survives a simple null control, but the functional/Pfam annotation gate is not yet met.
- Started broader UniRef50 pilot: `results/circuit_analysis/cross_model_conservation_3model_uniref500_wide_20260512.json` over 500 UniRef50 sequences, still below the final 10k atlas target.

## Opus Pivot Execution Pass (2026-05-12 CST)

- Added and ran `scripts/29_universal_primitive_annotation.py` for the R2 universal-primitives pivot.
  - Pilot v2 on balanced-200, top 10 triplets: `results/circuit_analysis/universal_primitives_pilot_20260512_v2/` on the H200 pod.
  - Full cheap-label pilot on 500 UniRef50 sequences and all 38 triplets: `results/circuit_analysis/universal_primitives_uniref500_20260512/`.
  - Result: all 38 triplets have some simple amino-acid / chemistry enrichment, but the best cheap-label MI is only 0.0001-0.0019 nats, far below the Opus 0.1-nat interpretability gate. Treat these as weak sequence-composition hints, not yet named biological primitives.
- Added and ran `r1_encoder_interpretability_benchmark/scripts/39_gene_level_mechanism.py`.
  - Output: `r1_encoder_interpretability_benchmark/results/variant_effect/gene_level_mechanism_20260512.{json,md}`.
  - Gene-level Pfam-family proxy holdout over 253 genes: macro-AUC 0.5665, macro-F1 0.3769; per-class AUC DN 0.5279, GOF 0.5879, LOF 0.5837.
  - Gate: `drop_mechanism_headline`. This does not support the R1 gene-level mechanism narrative.
- Added and ran `r1_encoder_interpretability_benchmark/scripts/40_indel_competitor_attempt.py`.
  - Output: `r1_encoder_interpretability_benchmark/results/variant_effect/indel_competitor_attempt_20260512.{json,md}`.
  - dbNSFP GRCh38 is staged and has CADD/REVEL columns, but IndelMissense v1 records lack chrom/pos/ref/alt and the dbNSFP first-10k row scan has zero indel-like rows.
  - Gate: `drop_head_to_head_for_current_pass`.
- H200 status after this pass: the 1-GPU pod remains reserved, but no experiment process is running and GPU memory/utilization are 0.

## Low-Risk TODO Supplements for Opus (2026-05-12 CST)

- Added and ran `scripts/32_universal_resource_annotation.py`.
  - UniRef500 top-firing set: `results/circuit_analysis/universal_primitives_resource_annotation_20260512/`. Coverage over 3,800 top events / 452 unique accessions: Pfam 0, Swiss-Prot 0, AlphaFold 1 accession.
  - Balanced-200 top-firing set: `results/circuit_analysis/universal_primitives_balanced200_resource_annotation_20260512/`. Coverage over 500 top events / 33 unique accessions: Pfam 7 accessions, Swiss-Prot 7 accessions, AlphaFold 0. Most Swiss-Prot overlaps are chain/topology, not strong functional/domain evidence.
- Created `archive/conversation_history/OPUS_LOW_RISK_RESULTS_20260512.md` as an English replanning packet. The packet recommends that Opus decide whether R2 should remain a "universal biological primitives" paper or be downgraded to conserved latent features until a resource-annotated cohort is built.

## Opus 2026-05-16 Additions (executed 2026-05-13 CST)

- Executed the low-risk additions from `archive/conversation_history/OPUS_NEXT_20260516.md` and recorded the English result packet in `archive/conversation_history/OPUS_PLAN_EXECUTION_20260516.md`.
- Added and ran `r1_encoder_interpretability_benchmark/scripts/42_indel_protein_baselines.py`.
  - H200 ESM masked-region pseudo-NLL scoring completed for all 6,649 IndelMissense v1 records.
  - Pulled required result: `r1_encoder_interpretability_benchmark/results/variant_effect/indel_esm_region_scores_20260516.tsv` with MD5 `649154aabd62b7c38b67c46b1e30b3a2`.
  - Final output: `r1_encoder_interpretability_benchmark/results/variant_effect/indel_protein_baselines_20260516/`.
  - Result: gate FAIL for a standalone SAE indel scorer. SAE damage AUC is 0.7735, while ESM region mean pseudo-NLL is 0.8037 and cheap grouped features are 0.8447. The combined SAE+ESM+cheap grouped LR reaches AUC 0.9108, so the defensible result is benchmark/resource plus a strong combined baseline.
- Added and ran `r1_encoder_interpretability_benchmark/scripts/43_am_sae_disagreement_typing.py`.
  - Output: `r1_encoder_interpretability_benchmark/results/variant_effect/am_sae_disagreement_typing_20260516/`.
  - Result: gate FAIL. Among 321 review-filtered AM/SAE opposite-direction disagreements, AM was right / SAE wrong in 283 and SAE was right / AM wrong in 38. No residue-context enrichment survived BH q < 0.05.
- Added and ran `scripts/37_attention_sink_subset.py`.
  - Output: `results/circuit_analysis/attention_sink_subset_20260516/`.
  - Result: T011/T018/T023 are concrete N-terminal edge attention-sink triplets; T025 is attention-associated but not N-terminal.
- Added and ran `scripts/39_attention_sink_biological_correlate.py`.
  - Output: `results/circuit_analysis/attention_sink_biological_correlate_20260516/`.
  - Result: gate PASS. T011/T018/T023 have first-two-residue target fraction 1.000 vs background 0.055, approximate OR 3.47e3, BH q 2.72e-117.
- Added and ran `scripts/38_universal_atlas_quality_diagnostic.py`.
  - H200 early-checkpoint cross-model artifact: `results/circuit_analysis/cross_model_conservation_3model_balanced200_early10k_20260516.json`.
  - Summary output: `results/circuit_analysis/universal_atlas_quality_diagnostic_20260516/`.
  - Result: gate PASS in the available early-checkpoint form. Mature v2 recovers 38 universal triplets, early10k recovers 16.
- R1-Add-3 VUS reclassification was deferred because historical ClinVar archive snapshots are not staged locally.
- Created `archive/conversation_history/OPUS_R1_PROBLEM_ANALYSIS_20260513.md` as an English problem-analysis packet for Opus. It argues that R1's main issue is likely claim-target mismatch rather than broken SAE training, and proposes raw-hidden vs SAE-code diagnostics to resolve the remaining ambiguity.
- Reviewed Opus's final pivot in `archive/conversation_history/OPUS_BRILLIANT_FINAL_20260517.md` and wrote `archive/conversation_history/OPUS_BRILLIANT_FINAL_REVIEW_20260513.md`. Main correction: the T011/T018/T023 `first2_fraction=1.00` result is over the saved top-100 firing rows per triplet, not all 700 cohort proteins. The review endorses the R2 attention-sink pivot but recommends a more careful causal-ablation design with matched controls and decoder-LM-aware likelihood readouts.

## R2 Attention-Sink Causal Ablation (2026-05-13 CST)

- Added and ran `scripts/40_attention_sink_causal_ablation.py` on the H200 hold pod.
- Output directory: `results/circuit_analysis/attention_sink_causal_ablation_20260517/`.
- English result packet: `archive/conversation_history/OPUS_R2_CAUSAL_ABLATION_RESULTS_20260517.md`.
- Design: 200 Swiss-Prot N-1 sequences, ProtGPT2 / ZymCTRL / ProGen2-medium, target triplets T011/T018/T023, T025 specificity control, and two same-layer random controls per target.
- Result: FAIL. Per-sequence rows: 6000. ProtGPT2 and ProGen2-medium show near-zero NLL / attention changes. ZymCTRL shows sizable shifts for T011 and T023, but in the wrong direction for a required-feature story.
- Interpretation: the attention-sink subset remains a strong correlation/characterization result, but this MLP-output CLT ablation does not support a causal attention-sink mechanism claim.

## Opus Rescue Plan Execution (2026-05-13 CST)

- Reviewed the latest rescue plans: `archive/conversation_history/OPUS_NEXT_20260518.md` and `archive/conversation_history/OPUS_R1_RESCUE_20260518.md`.
- Added, smoke-tested, and ran `scripts/41_attention_head_sink_ablation.py` on the 1-GPU H200 pod.
  - Output: `results/circuit_analysis/attention_head_sink_ablation_20260518/`.
  - Log: `logs/runtime/r2_attention_head_sink_ablation_20260518.log`.
  - Result: FAIL. The selected target heads are N-terminal sink-like by attention mass, but direct single-head ablation produces near-zero target-feature drop and far below-threshold N-terminal NLL change across ProtGPT2, ZymCTRL, and ProGen2-medium.
  - Interpretation: T011/T018/T023 remain conserved N-terminal attention-sink-associated sparse features, but the current evidence does not support a single-head causal mechanism claim.
- Added and ran `r1_encoder_interpretability_benchmark/scripts/47_extended_disagreement_typing.py`.
  - Output: `r1_encoder_interpretability_benchmark/results/variant_effect/extended_disagreement_typing_20260518/`.
  - Result: FAIL. The audit expands to 90 original + protein-context tests, but no context survives BH q < 0.05. The closest SAE-right enrichment is `mechanism_missing`, q = 0.0717.
- Added and ran `r1_encoder_interpretability_benchmark/scripts/46_vampseq_abundance_proxy.py`.
  - Output: `r1_encoder_interpretability_benchmark/results/variant_effect/vampseq_abundance_proxy_20260518/`.
  - Result: proxy FAIL. On nine staged ProteinGym / VAMP-like abundance assays, the SAE-family signal beats AlphaMissense on only 1/9 usable assays.
- Created English Opus packet: `archive/conversation_history/OPUS_RESCUE_EXECUTION_20260518.md`.
- R1-Save-1 low-MSA stratification remains unlaunched. UniRef50 FASTA and DIAMOND tarballs are staged, but there is no cached per-protein MSA-depth table / DIAMOND database; after the new R1-Save-2 and R1-Save-3 negatives, this should be an explicit Opus decision before spending more compute.
- H200 state after the run: pod `jiaotongdamoxing-zhk-zip-hold-1gpu-0513b-master-0` remains reserved but is idle with 0 MiB GPU memory and 0% utilization.

## Final R1/R2 Rescue Attempt (2026-05-13 CST)

- Created final English result packet: `archive/conversation_history/FINAL_RESCUE_RESULTS_20260518.md`.
- Created concise Opus handoff packet: `archive/conversation_history/OPUS_FINAL_RESCUE_HANDOFF_20260518.md`.
- Added and ran `r1_encoder_interpretability_benchmark/scripts/45_low_homology_stratification.py`.
  - Output: `r1_encoder_interpretability_benchmark/results/variant_effect/low_homology_stratification_20260518/`.
  - Design: UniRef50 representative cluster-size (`n=` header) as a staged low-homology proxy; covers 1,892 / 1,972 variants and 991 proteins.
  - Result: FAIL. Low-q1 AlphaMissense AUC 0.9627 vs SAE+LLR z-ensemble AUC 0.8727; delta -0.0900 with 95% group-bootstrap CI [-0.1253, -0.0568].
  - Interpretation: the low-homology rescue does not produce a defensible AlphaMissense-complementarity scope.
- Added and ran `scripts/42_attention_sink_set_ablation.py`.
  - top-8 output: `results/circuit_analysis/attention_sink_set_ablation_20260518/`.
  - top-32 output: `results/circuit_analysis/attention_sink_set_ablation_top32_20260518/`.
  - Result: FAIL in both strict and exploratory gates. top-8 sink sets have near-zero feature drops and small dNLL effects; top-32 also fails and ZymCTRL random same-layer heads produce larger NLL shifts than the sink set.
  - Interpretation: R2 should drop both single-head and distributed-head causal attention-sink mechanism claims.
- Current H200 state after final rescue runs: pod `jiaotongdamoxing-zhk-zip-hold-1gpu-0513b-master-0` remains allocated and idle with 0 MiB / 0% GPU utilization.

## Directory Organization (2026-05-14 PT)

- Added top-level `README.md` as the project entry point.
- Added top-level `docs/DOCUMENT_INDEX.md` to make the many Opus, TODO, status, and result documents navigable.
- Moved Opus packets, final rescue packets, historical analysis packets, and historical `TODO_NEXT*` files into `archive/conversation_history/`.
- Updated root-level navigation paths in `README.md`, `docs/DOCUMENT_INDEX.md`, `PROJECT_STATUS.md`, `TODO_RESULTS.md`, and `LOG.md`.
- Follow-up cleanup: moved `PROJECT_STATUS.md`, `TODO_RESULTS.md`, `LOG.md`, and `TODO.md` into `docs/`; updated root-level navigation paths.

## R2 Attention-Output Sparse Pilot (2026-05-14 PT)

- Added `scripts/43_attention_output_transcoder_pilot.py`.
- Ran a ZymCTRL layer-23 attention-output sparse pilot on the 1-GPU H200 hold pod `jiaotongdamoxing-zhk-zip-hold-1gpu-0513b-master-0`.
- Local output: `results/circuit_analysis/attention_output_transcoder_pilot_20260514_l23/`.
- Runtime log: `logs/runtime/r2_attention_output_transcoder_pilot_20260514_l23.log`.
- Result packet: `archive/conversation_history/R2_ATTENTION_OUTPUT_TRANSCODER_PILOT_20260514.md`.
- Result: readout positive but causal ablation negative. Mean eval FVU is 0.4782, best attention-received correlation is 0.9815, and best first2 delta is 8.0173. However, target feature ablation produces dNLL pos2-10 = -0.0508, while random controls are near zero or small positive. This does not rescue the causal attention-sink mechanism claim.

## IndelMissense Coordinates (2026-05-15)

- Added `r1_encoder_interpretability_benchmark/scripts/48_package_indelmissense_coordinates.py`.
- Generated `data/indelmissense/v1.1_coordinates/` from v1 records and `data/clinvar/variant_summary.txt.gz`.
- Coverage summary:
  - exact ClinVar variant match: 6,649 / 6,649;
  - any genomic coordinate: 6,642 / 6,649;
  - GRCh37 coordinate: 6,638 / 6,649;
  - GRCh38 coordinate: 6,642 / 6,649;
  - any VCF-style coordinate: 6,635 / 6,649;
  - GRCh37 VCF-style coordinate: 6,631 / 6,649;
  - GRCh38 VCF-style coordinate: 6,635 / 6,649.
- Recorded the dataset comparison and coordinate notes in `docs/INDELMISSENSE_SIMILAR_DATASETS_AND_COORDINATES_CN.md`.

## ProteinInterpret Benchmark Scaffold (2026-05-15)

- Added shared benchmark scaffold under `r0_shared_interpretability_framework/`.
- Added `r0_shared_interpretability_framework/scripts/build_v0_benchmark_tables.py`, which standardizes existing R1/R2 artefacts into encoder and decoder benchmark v0 tables.
- Generated `r0_shared_interpretability_framework/results/v0_20260515/`.
  - Encoder v0: 65 standardized result rows over ClinVar2000, CancerHoldout101, IndelMissense v1.1 coordinates, ProteinGym substitutions, VAMP-like proxy results, SAE feature annotation coverage, and the mechanism-selected top150 feature audit. Added an IndelMissense v1.1 resource preflight showing 6,635/6,649 GRCh38 VCF-style records but 0 exact SNV-compatible records, so dbNSFP/REVEL should be treated as a coverage gate rather than a primary indel baseline.
  - Decoder v0: 29 standardized result rows covering cross-model triplet discovery, M-2 characterization, checkpoint diagnostics, attention-output sparse pilot, attention-sink biological context enrichment, resource coverage, and causal-ablation negative gates.
- H200 resource check: dbNSFP GRCh38 BGZF and `.tbi` are visible on the `/oss-pvc` mount in the current 1-GPU hold pod, but the image lacks `tabix/bcftools/pysam`; pod network is unavailable for direct `pip install`. The IndelMissense v1.1 dbNSFP/REVEL exact coverage check is therefore a dependency/tooling task; preflight suggests exact coverage should be low because the benchmark is dominated by non-SNV indel/delins alleles.
- Recorded the benchmark progress in `docs/BENCHMARK_PROGRESS_20260515_CN.md`.

## ProteinInterpret Benchmark Extension (2026-05-22)

- Extended the encoder v0 table from 65 to 75 standardized rows.
  - Added `r0_shared_interpretability_framework/scripts/encoder_mechanism_localization_preflight.py`.
  - Output: `r0_shared_interpretability_framework/results/v0_20260515/encoder/mechanism_localization_preflight.{json,md}`.
  - Result: 150/150 mechanism-selected SAE features have parseable residue-level top firing examples; 750/750 examples map to Swiss-Prot sequence lengths. This validates the input substrate for future localization, occlusion and CRPI metrics, but is not itself a faithfulness result.
- Extended the decoder v0 table from 29 to 52 standardized rows.
  - Added generation-metric calibration rows from `results/ec_metrics/ec_metric_calibration_summary_20260507.json`.
  - Added Pfam/CLEAN/Foldseek generated-sequence triad rows from `results/ec_metrics/generated_metric_triad_summary_20260507.json`.
  - Added direct-effect steering gate rows from the saved ZymCTRL steering benchmark outputs.
  - Result: the external metric stack passes real-vs-random validity controls, but all-sequence steered-vs-unsteered and EC-class steering gates remain negative.
- Added `r0_shared_interpretability_framework/results/v0_20260515/method_registry.tsv` with 9 rows covering completed methods, negative controls and not-started method baselines for the two benchmarks.

---

## Repository reorganization to the three-paper structure (2026-05-28)

- Renamed `Research2/` → `` and nested the R2 manuscript at `manuscript/` (Paper A, Nature Machine Intelligence).
- Renamed `Research1/` → `r1_encoder_interpretability_benchmark/` (Paper B, encoder-only interpretability benchmark) and added `r1_encoder_interpretability_benchmark/manuscript/README.md` as a Paper B drafting stub.
- `r0_shared_interpretability_framework/` retained as the shared framework underneath Papers B/C.
- Created `archive/` and moved superseded material there: all of `conversation_history/` (rescue-phase planning), `legacy/` proposals, the combined initial-draft and standalone R1 manuscripts, the old manuscript evidence index, and stale `docs/` (pivot/review/CN notes, old TODOs).
- Updated path references across 74 live files (project roots and the moved manuscript path); LaTeX `\texttt{}` paths in `main.tex` use escaped underscores. Updated `.mutagenignore` / `.gitignore` to the new project roots (the generic `results/`, `logs/`, `wandb/` patterns already protect renamed trees from syncing to the D mirror).
- Verified: `scripts/50_make_manuscript_figures.py` regenerates the four figures into `manuscript/figures/`, and `main.tex` compiles with Tectonic (16 pp., no errors or undefined refs).
- Rewrote `README.md`, `docs/DOCUMENT_INDEX.md`, `docs/README.md`, the `PROJECT_STATUS.md` header, and `CLAUDE.md`; added `AGENTS.md`.

---

## Recoverability-audit code implemented (2026-06-04)

- Implemented the audit pipeline under `scripts/`: `recoverability_audit.py` (engine) + `44_cache_representations.py`, `45_probe_ceiling_floor.py`, `46_oracle_direction_steering.py`, `47_decision_table.py`, `48_capacity_retrain.py`. Reuses the loaders/probes in scripts 29/33/34 and the model/CLT API in `src/`.
- Verified: py_compile clean; synthetic analysis test (ceiling/floor/gap/rho + GO/NO-GO logic); real GPU smoke 44→45→47 on ZymCTRL (24 proteins) end-to-end.
- Two v1 deviations recorded in `preregistration/DECISION_LOG.md`: local checkpoints are the 100k/`d_clt=4096` reruns (not the 200k/8192 references), and script 46 uses an ESM-2 EC judge for purity (CLEAN/Pfam pluggable).

---

## Recoverability-audit pre-registration + reorg fixes (2026-06-04)

- Added the pre-registered protocol `preregistration/` (`PROTOCOL.md`, `DECISION_LOG.md`, `README.md`): the representation-recoverability audit that separates "circuit tracing / sparse dictionaries failed (tool)" from "the generators learned little (substrate)". Defines the probing tasks (T1–T6), reusable datasets/labels, the ceiling/floor/gap metrics (`C/F/Δ/ρ/φ`), and the go/no-go thresholds for the high-cost dictionary retrain. Results dir: `results/representation_audit_20260604/`.
- Fixed two side-effects of the 2026-05-28 rename:
  - **Local path roots**: 28 scripts used `REPO / "Research1"|"Research2"` (no trailing slash, so the earlier slash-based sed missed them). Updated to `REPO / "r1_encoder_interpretability_benchmark"|"the repository root"`.
  - **Remote namespace**: the earlier sed had rewritten some absolute GPFS/OSS staging paths (`…/biocc/Research2/…` → `…/biocc/paper_r2_nature_mi/…`) while leaving others, making the remote refs inconsistent. The remote storage was never renamed, so all `/biocc/<proj>` paths were restored to the consistent prior names (`Research2`, `Research1`). Before running, confirm the live checkpoint paths on the active pod (DECISION_LOG pre-run checklist).
- **Sync resurrection cleanup**: the bidirectional D↔B sync had resurrected stale, code-only copies of `Research1/`, `Research2/`, and `manuscripts/` (no `results/`; the real results live in the renamed dirs). These were duplicates of content already in the renamed dirs + `archive/`; removed on B, verified non-resurrecting. If they reappear, the D mirror still holds the pre-reorg snapshot and the sync daemon needs the stale dirs cleared / a rescan.

---

## R2 recoverability audit full H200 run (2026-06-05)

- Ran the full gated R2 recoverability pipeline on the beliefnav H200 pod: `44_cache_representations.py` -> `45_probe_ceiling_floor.py` -> `46_oracle_direction_steering.py` -> `47_decision_table.py`.
- Cohort/cache: 820 Swiss-Prot proteins, 44,626 labelled residue positions, and 48 ZymCTRL decoder-native EC sequences. ProtGPT2 v2 and ZymCTRL v2 used the 200k-step `d_clt=8192,k=128` checkpoints; ProGen2-medium used the 100k rerun checkpoint.
- Main probe outcome: ProtGPT2 and ProGen2-medium have some R_raw-readable substrate (`substrate=RICH`), but no model satisfies the frozen capacity-retrain GO conditions. ZymCTRL is mixed.
- Oracle steering outcome: ZymCTRL raw EC-direction injection passed 0/8 EC class gates; verdict `distributed_or_robust`.
- Decision: `NO-GO`; Experiment 48 was skipped by the protocol gate, so no capacity retrain launched in this run.
- Lightweight local copy written to `evidence/recoverability_audit_20260605_1250/`. Large cache arrays remain only on GPFS at `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/results/representation_audit_20260605_1250_recoverability_full/`.

---

## R2 recoverability audit v2 corrected re-analysis (2026-06-06)

- Reviewed and executed Claude's v2 analysis corrections from `preregistration/NEXT_STEPS_v2.md`, reusing the existing 2026-06-05 cache and rerunning only `45_probe_ceiling_floor.py` and `47_decision_table.py`.
- Corrections: fold-internal PCA dimensionality control (`--pca-dim 256`), `ec_topclass_stratified` as report-only EC/Pfam confound diagnostic, dropped the broken `beats_rand` richness gate, and regressed only helix+strand for `secondary_fraction`.
- Result: all three models are now classified as `substrate_rich`, but none satisfies the frozen dictionary-bottleneck GO gate. Final decision remains `NO-GO`.
- EC confound size is large: family-disjoint vs stratified EC ceiling skill is ProtGPT2 0.260 vs 0.557, ZymCTRL 0.123 vs 0.356, ProGen2-medium 0.272 vs 0.594.
- `secondary_fraction` remains numerically unstable after PCA (very negative floor R2), so it should be treated cautiously; the well-powered residue-level structural task remains near-faithful (rho 0.812-0.981).
- Lightweight v2 outputs were copied into `evidence/recoverability_audit_20260605_1250/probes_v2/` and `evidence/recoverability_audit_20260605_1250/decision_v2/`.

---

## R2 exploratory expanded-dictionary retrain started (2026-06-06)

- User requested a direct expanded-dictionary retrain despite the frozen recoverability gate returning NO-GO. This run is therefore an **exploratory override**, not a pre-registered confirmatory experiment.
- Added `scripts/run_three_model_expand_retrain_h200.sh`. The current beliefnav pod has two H200 GPUs, so the three model jobs run sequentially, each using 2-GPU DDP (`--gpus 0,1`).
- Planned models and widths:
  - ProtGPT2: `d_clt=16384`, `k=128`, 300k steps.
  - ZymCTRL: `d_clt=16384`, `k=128`, 300k steps.
  - ProGen2-medium: `d_clt=16384`, `k=128`, 300k steps.
- Rationale for `d_clt=16384`: a 32768-wide ProtGPT2/ZymCTRL dictionary would exceed H200 memory once fp32 weights, gradients and Adam states are included; 16384 is the largest conservative width expected to fit on 143GB H200s with batch size 1/GPU.
- Also fixed a monitoring bug in `src/training/clt_trainer.py`: `dead_mean` now uses the configured `dead_feature_threshold` instead of a hard-coded 10000-step threshold. Resampling behavior was already using the configured threshold.

### Update (2026-06-07)

- ProtGPT2 reached `step_10000` and saved a complete checkpoint under `/oss-pvc/zhk_zip/outputs/research2/clt_weights/r2_clt_expand_retrain_20260606_2gpu/protgpt2/step_10000/`.
- The run then stopped during the post-checkpoint DDP synchronization with a NCCL watchdog timeout. This was caused by rank 0 writing a very large checkpoint (`clt.pt` plus optimizer state, about 70GB total) while rank 1 waited longer than the default 10-minute process-group timeout.
- Updated the trainer to initialize NCCL with a 12-hour timeout and to call `dist.barrier(device_ids=[rank])`; also increased the default checkpoint interval in the expanded retrain runner from 10k to 50k steps to reduce OSS write pressure and storage growth.
- Resumed the same exploratory run from `step_10000` with remote PID `46401`. The resumed log is `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/logs/runtime/r2_clt_expand_retrain_20260606_2gpu_resume_20260607.log`. The ProtGPT2 model log had resumed training and reached at least `step_10300` at the time of the check.

### Update (2026-06-08)

- ProtGPT2 completed `300000/300000` steps and wrote the final checkpoint at `/oss-pvc/zhk_zip/outputs/research2/clt_weights/r2_clt_expand_retrain_20260606_2gpu/protgpt2/step_300000/`.
- The final step wrote the same `step_300000` checkpoint twice because the save interval landed exactly on the terminal step and the trainer also saves after the loop. This caused a long idle-looking checkpoint window, but the run eventually exited cleanly with `DONE protgpt2 status=0`.
- The queue then started ZymCTRL with the same 2-GPU expanded-dictionary configuration. At the check, ZymCTRL had entered normal training and reached at least `step 100/300000`.

### Update (2026-06-11)

- ZymCTRL completed `300000/300000` steps and wrote the final checkpoint at `/oss-pvc/zhk_zip/outputs/research2/clt_weights/r2_clt_expand_retrain_20260606_2gpu/zymctrl/step_300000/`.
- The queue then started ProGen2-medium with the same 2-GPU expanded-dictionary configuration. At the check, ProGen2-medium was running normally at about `step 209650/300000` on two H200s.
- Current checkpoint root size is about `1.2T`; ProGen2-medium checkpoints are present through `step_200000`.

### Update (2026-06-12)

- ProGen2-medium completed `300000/300000` steps and wrote the final checkpoint at `/oss-pvc/zhk_zip/outputs/research2/clt_weights/r2_clt_expand_retrain_20260606_2gpu/progen2-medium/step_300000/`.
- The three-model expanded-dictionary retrain queue finished successfully with `ALL DONE`. No CLT training process remained at the status check; the H200 GPUs were idle aside from the pre-existing lightweight retrieval service on GPU 0.
- Final checkpoint root size is about `1.3T`.

---

## R2 expanded-dictionary downstream evaluation started (2026-06-12)

- Started a full recoverability downstream evaluation using the three expanded `d_clt=16384,k=128,step_300000` CLT checkpoints from `r2_clt_expand_retrain_20260606_2gpu`.
- Pipeline: `44_cache_representations.py` -> `45_probe_ceiling_floor.py` -> `46_oracle_direction_steering.py` -> `47_decision_table.py`.
- This is an evaluation-only run over the already trained exploratory dictionaries: `RUN_48=0` so it will not launch another retraining job even if the decision table returns GO.
- Configuration: `LAYERS=all`, `RESIDUE_LAYERS=even6`, `N_BOOT=1000`, `N_GEN_46=40`, GPU1 only (`CUDA_VISIBLE_DEVICES=1`).
- Remote output root: `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/results/representation_audit_20260612_expanded_eval/`.
- Remote runtime log: `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/logs/runtime/recoverability_expanded_eval_20260612.log`.
- Remote PID: `54388`. Initial check showed script 44 running normally and computing ESM-2 embeddings for the 820-protein cohort.

### Result (2026-06-12)

- The expanded downstream evaluation completed successfully: 44/45/46/47 all returned status 0, and 48 was skipped by design (`RUN_48=0`).
- Final decision remained **NO-GO**: all three models are `substrate_rich`, but none satisfies the frozen dictionary-bottleneck GO conditions and none becomes a clean near-faithful dictionary case.
- Main positive deltas were local rather than decisive: ZymCTRL decoder-native EC improved from F=0.525 to F=0.623 (rho 0.806 -> 0.955), and ProGen2-medium improved on EC top-class (F=0.153 -> 0.228) and Pfam family (F=0.910 -> 0.931).
- ProtGPT2 did not improve on EC/Pfam (EC floor 0.262 -> 0.204; Pfam floor 0.808 -> 0.789). ProGen2/Pfam and residue-level structure are near-faithful, but this is not a causal mechanism result.
- `secondary_fraction` remains numerically unstable/report-only: expanded dictionaries reduce the extreme negative floors but still do not produce trustworthy recoverability.
- ZymCTRL oracle residual-stream steering still failed the controllability gate: 0/8 EC classes significant; verdict `distributed_or_robust`.
- Summary written locally to `docs/R2_EXPANDED_DOWNSTREAM_ANALYSIS_20260612.md`.

### Correction (2026-06-13)

- Pulled final expanded-checkpoint training-quality metrics from the remote training logs:

  | Model | Final FVU | Dead fraction | Dead% | L0 | Loss |
  |---|---:|---:|---:|---:|---:|
  | ProtGPT2 | 0.2782 | 0.334 | 33.4% | 128.0 | 7.1176 |
  | ZymCTRL | 0.3348 | 0.044 | 4.4% | 90.6 | 0.0201 |
  | ProGen2-medium | 0.3101 | 0.122 | 12.2% | 123.0 | 0.0138 |

- Re-ran the 47 decision logic on the expanded probe results with Amendment 1b enforced: `secondary_fraction` is report-only and excluded from the rich/bottleneck gate.
- Corrected output written remotely to `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/results/representation_audit_20260612_expanded_eval/decision_no_secondary/`.
- Corrected decision remains **NO-GO**, but the reason changes to `dictionary already near-faithful on rich tasks`. There are no primary-gate bottleneck tasks after excluding `secondary_fraction`; ZymCTRL and ProGen2-medium are near-faithful on their rich tasks, while ProtGPT2 is mixed.
- Local summary updated (now located at): `docs/analysis/EXPANDED_DICTIONARY_ANALYSIS_20260612.md`.

---

## Paper A Swiss-Prot MI Gate Re-audit (2026-07-16)

- Independently audited script 33 and confirmed that its 0.1-nat gate was impossible for the defined top-100 binary event. Across 122,671 positions, the firing-event entropy and hence maximum possible MI is 0.00661255 nats (the gate is 15.12x larger).
- Ran a CPU-only, 2,000-permutation exploratory reanalysis of the unchanged saved top-position output. MI was normalized by firing-event entropy and tested against both a global position null and a within-protein null matched on amino acid and fine position/edge strata; BH correction covered 380 tests.
- Result: 224/380 associations survived the global null, but 0/380 survived the covariate-matched null. Maximum matched excess normalized MI was 0.02180 (T006 `swiss_feature_type`, raw P=0.000500, q=0.09495).
- The cohort has 500 unique dominant Pfams for 500 proteins, so its `dominant_pfam` MI cannot demonstrate cross-protein family generalization.
- Outputs and full evidence boundary: `results/circuit_analysis/swissprot_triplet_mi_reaudit_20260716/`.
- A mask audit reconstructed the exact saved quick-evaluation token totals: padding contributed 298/8,342 positions (3.57%) for ProtGPT2 and 673/24,466 (2.75%) for ZymCTRL. Training and evaluation discard `attention_mask`, so pad positions enter CLT gradients, FVU and feature-firing statistics. See `padding_mask_audit.json` in the same output directory.
- Added deterministic reproduction script `scripts/49_reaudit_triplet_mi.py` (seed 20260716, 2,000 permutations). It passed `py_compile` and a full rerun, including strict cohort/label/top-event/saved-MI validation. The two numeric TSVs reproduced byte-for-byte (SHA-256 `c144bfb...41556` and `96296e...4784`); the run manifest now records exact commands and input/output hashes.

---

## Repository reorganization and Paper A npj package (2026-07-16)

- Completed a repository-wide structural/evidence audit. The live work is now documented as two active directions (Paper A decoder/readout audit; Paper B encoder benchmark), with `r0_shared_interpretability_framework/` as shared framework and future Paper C seed.
- Removed verified duplicate resurrected roots `Research1/`, `Research2/` and `manuscripts/` from both Mutagen endpoints, archived the frozen `analysis_exports/` bundles, removed the stale root `results/`, and archived both pre-pivot proposals. Canonical project roots were preserved because source code depends on their path depth.
- Added `docs/audits/REPOSITORY_AUDIT_20260716.md` and `docs/audits/REORGANIZATION_20260716.md`; Mutagen returned to `Watching for changes` with no stale roots on either endpoint.
- Selected npj Artificial Intelligence for Paper A. Downloaded and hash-checked 20 recent open-access npj articles and the official December 2024 Springer Nature template. The manuscript now has a 149-word abstract, six rebuilt figures, a standalone ten-table supplement and a 58-file source-data package.
- Final builds: main Article 18 A4 pages; supplement 11 A4 pages; no errors, unresolved references or overfull boxes. Source-data and literature checksum verification passed.
- Redacted the hard-coded Hugging Face credential from live documentation and the live download script in favour of `HF_TOKEN`. A frozen legacy file and a backup archive may retain the old credential; revoke/rotate it rather than rewriting provenance.

---

## Canonical structure standardization (2026-07-16)

- Adopted `r<research ID>_<lower_snake_case_scope>` and migrated the live roots to `r0_shared_interpretability_framework/`, `r1_encoder_interpretability_benchmark/`, and `` at unchanged depth.
- Paused Mutagen and moved all three roots on both endpoints, preserving the dirty nested B-side Git worktrees and B-only `results/`/`logs/` payloads; resumed and flushed synchronization afterward.
- Replaced root `project_records/` with `docs/`; moved R2 compact evidence and analysis into the R2 project; normalized research plans, technical notes, manuscript audit/template paths, figure names, source-data folder names, and duplicate script prefixes.
- Updated active code, build commands, ignore rules, navigation, manifests, and checksums. Historical archive/result metadata and external GPFS/OSS/H200 path names remain unchanged as execution provenance.
- Migration map and verification scope: `docs/audits/STRUCTURE_STANDARDIZATION_20260716.md`.
- Final verification passed: Mutagen `Watching`, retired roots absent on both endpoints, 119 Python files and all shell/YAML inputs parse, R0 preflights and table generation pass, six figures rebuild, source data verifies 58 rows/61 checksums, all literature/template hashes pass, and Tectonic produces the 18-page main Article plus 11-page supplement.

---

## R0 experimental-procedure and result audit (2026-07-16)

- Reconstructed all three executable R0 procedures: two descriptive encoder preflights and the v0 evidence-ledger compiler. Confirmed that R0 performs no independent model training, inference, intervention or hypothesis test.
- Reconciled all 75 encoder and 52 decoder method-result rows with the current builder and their named upstream evidence, and documented exact procedures, results, limitations, claim boundaries and audit-time artifact hashes in `r0_shared_interpretability_framework/docs/EXPERIMENTAL_PROCEDURES_AND_RESULTS_AUDIT_20260716.md`.
- Identified release-blocking discrepancies without modifying scientific outputs: malformed decoder steering statistics from source-schema drift; a stale, mathematically invalid Swiss-Prot MI gate; an orientation-dependent VAMP-like row; Foldseek omissions/non-finite output; incorrect statistical sample sizes; dataset/registry referential-integrity failures; and missing immutable build provenance.
- Disposition: the present R0 tables are an internal evidence index with known exceptions, not a publication-ready benchmark or leaderboard.

---

## R2 experimental-procedure and result audit (2026-07-16)

- Consolidated the complete R2 procedure-to-result history across baseline and v2 CLT training, the cross-model atlas/null, semantic and resource analyses, N-terminal characterization, steering, five intervention families, recoverability, wider dictionaries, and the npj evidence package.
- Added `docs/EXPERIMENTAL_PROCEDURES_AND_RESULTS_AUDIT_20260716.md` as the authoritative human-readable audit layer; linked it from R2 and root navigation. Scientific result artifacts were not modified.
- Confirmed the current source-data verifier passes 58 manifest rows and 61 checksums (1,412,592 bytes), and the existing manuscript PDFs remain 18-page main plus 11-page supplement.
- Recorded release-critical gaps: missing derived balanced-200 cohort file/hash and model/checkpoint manifests, pair/layer-specific rather than coherent atlas null permutations, unmasked padding, cohort-sensitive triplet identities, an invalid original MI gate, and exploratory recoverability/wider-dictionary limitations.
- Identified two additional steering audit defects: negative-attribution features enter six target/layer cells through a selector fallback, and the external lysozyme metric triad evaluates an older z-score/off-manifold run rather than the later direct-effect/TopK-aware benchmark.
- Disposition: R2 supports procedure-specific sparse readouts and bounded intervention negatives, not biological primitives, a causal attention sink, validated EC steering, or capacity/optimization falsification.
## H200 Remote Helper Unification (2026-07-17 CST)

- Consolidated Hangzhou access configuration under `~/hangzhou-remote/config.sh`; command, Kubernetes and pod helpers now read a single jump-host/namespace/GPFS configuration.
- Added checksum-verified, chunked and retryable GPFS transfers in `h200_push.sh` and `h200_pull.sh`, bidirectional directory merge in `h200_sync.sh`, and an end-to-end `h200_status.sh` health check.
- Verified local-to-Windows-to-master-to-GPFS upload and reverse download with exact SHA-256 equality for single- and multi-chunk files. Verified directory push/pull with a recursive content diff and confirmed both Windows tunnel tasks, all Kubernetes nodes and GPFS read/write access.
- Updated `ops/sync_required_assets_to_h200.sh` to use the GPFS transfer helper, SHA-256 directory markers and an explicitly selected pod for OSS-only work.
- Removed superseded transfer implementations, dated hold manifests and stale one-off download/experiment queue scripts that targeted deleted pods or old research paths.
- Consolidated two competing Windows reverse-tunnel tasks into one boot-time SYSTEM task, removed the obsolete user tunnel key/script and replaced a 64 MB retry log with a bounded, rotating log. Verified automatic relay reconnection and SYSTEM ownership after migration.
- Used direct root access to the Aliyun relay to remove the obsolete user reverse-tunnel authorization. Retained exactly two restricted keys with role-based comments: local access may only open `127.0.0.1:2222`, and the Windows SYSTEM tunnel may only listen on that endpoint.

---

## R2 npj major-revision execution (2026-07-17)

- Converted the independent npj assessment into a binding nine-package P0 revision plan. Paper A remains not submission-ready; infrastructure progress is not recorded as a scientific gate pass.
- Corrected manuscript/source-data claims for cohort sensitivity, the pair/layer-specific atlas null, steering sign fallback, mixed Figure 5 chronology, basis-probe folds and attention-pilot uncertainty. The strict package now verifies 65 rows, 68 checksums and 1,649,686 bytes; both LaTeX documents compile without overfull boxes or unresolved references.
- Added hash-verified historical-cohort reconstruction, mask-aware CLT training/evaluation, deterministic split/checkpoint state, positive-only steering selection, strict structural-QC manifests, independent-atlas and conditional-semantics engines, planted causal control and nested repeated recoverability infrastructure.
- Restored the omitted 12--18 May R2 chronology in the project experiment log.
- The analytic planted simulator passed synthetic plumbing checks. A later trained-GRU control was a post-hoc development smoke whose step budget was adjusted after observing an interim output; it is not a frozen confirmatory sensitivity, specificity, FDR or equivalence result. Nested recoverability likewise passed synthetic plumbing only. Neither is a pretrained-model biological or causal result.
- Its prospectively frozen unexposed replacement was then executed once and passed all specified synthetic sensitivity, specificity, FDR, localization, dose-recovery and matched-control-equivalence gates across seeds 11/29/47. Supplementary Table 11 and its source-data receipts report this bounded pipeline calibration. The comparator is fixed, not a learned biological circuit; the real held-out pretrained P0-7 gate remains open and its raw synthetic artifacts still require the licensed DOI deposit.
- Re-reviewed `~/hangzhou-remote`, verified the four-GPU H200 pod `damoxing-zhk-zipbio-master-0`, hashed the deployed datasets, froze disjoint UniRef50 and ZymCTRL train/validation/test manifests and passed a masked ZymCTRL H200 smoke test.
- The first three TopK CLT queues (seeds 17/29/43) were discarded after the uncapped dead-feature resampler replaced roughly two thirds of the dictionary at step 15,000 and seed 43 became non-finite. Seeds 17/29 were terminated despite remaining finite because they were no longer comparable.
- A bounded deterministic resampler, finite-state checks and an exact atomic checkpoint/resume contract now pass full-width stability, byte-identical resume and 37.25-GB checkpoint-publication smokes. An archive/tree-bound launcher passed the second independent audit, and replacement seeds 17/29/43 are running finite ProtGPT2 training on GPUs 0/1/2 under a fresh GPFS root.
- GPU 3 is running the first production exact activation cache (ProtGPT2) for the P0-2 alternative-dictionary panel; GPUs 0--2 are occupied by the replacement TopK queues. No P0-2 quality gate has passed.
- Found that the master helper's `/gpfs` target is ext4 and not the pod's GPFS mount, despite the updated note. Hash-verified staging therefore uses the master transfer followed by an explicit Kubernetes copy into the selected pod. This is documented as an operational discrepancy rather than silently assuming one shared path.
- Recovered the exact 81,480-byte historical balanced-200 cohort named by the May atlas provenance from archived pod storage (SHA-256 `5bc7697a83cc7461558f8b4597a3c9b4d6a151b7ec70ca22efc7282ecde4f0a6`). Its 200 ordered objects exactly match the prior reconstruction, so local order status is now `historical_exact_file_verified`.
- Located and hash-verified the exact three historical reference CLT files named by the saved atlas: ProtGPT2 `5eca3b19284dbd9b302078e3a7e34ce7a2fc78d97b1566eae927d4d1c30f1f00`, ZymCTRL `5da70c530b83a034d1fe683a72a8cc5bd7b49463d2598036cd6b5db94ca5761d` and ProGen2-medium `5e384733dc28ecad3947b65c0c8b34f058ce50a61aab67399548c2b21687b8fd`. P0-1 remains open for upstream revisions, complete raw artifacts and public deposit; local recovery is not a release.
- Closed the P0-5 analyzer provenance gap: production script 59 now requires external SHA-256 pins, revalidates the production extractor receipt and all its artifacts, consumes the frozen `feature_matches.json` membership without rematching/postselection, and publishes only complete new directories.
- Closed the P0-8 aggregation/sample-floor gap: builder/runner schema v2 keeps reconstruction and intervention evidence separate for every dictionary seed/layer and analyzes those cells without prior averaging. Production now requires at least 480 annotated rows in both the builder and real analyzer; the legacy eight-row floor is fixture-only.
- The focused P0-5/P0-8 suite passed 34 tests plus 2 subtests; the complete R2 suite passed 165 tests plus 6 subtests. Ruff, byte compilation and strict JSON parsing passed. This was CPU contract validation only; no H200 job or scientific-gate status changed.
- Recovered exact deployed-model provenance at three distinct evidence levels: ProtGPT2's complete tree verifies to Hugging Face commit `f71aa6cf063ad784ebd53881d11332fd098eaa58`; ZymCTRL's weight verifies and `3c532ef172b9cd2e95238baadf5167ebb89fbc32` is the best-supported snapshot, but strict whole-tree proof remains incomplete; the deployed ProGen2-medium tree is a hybrid that matches no single upstream commit. Machine-readable manifests preserve those boundaries. P0-1 remains open for the exact local deposit, remaining raw artifacts, tag and DOI.
- Validated the bfloat16 cache path end-to-end after two bounded preflights correctly exposed an eager-import coupling and unsupported direct bfloat16 NumPy conversion. The minimal initializer and explicit finite-checked bfloat16 -> CPU float32 -> float16-cache conversion fixed those defects. The authoritative r7 bounded preflight passed; it remains production-ineligible.
- At 2026-07-17 19:26 CST, three clean archive-bound bfloat16 TopK queues started on H200 GPUs 0--2 for seeds 17/29/43. All reported verified bfloat16 model inference and finite training through at least step 1,000 at 0.24 seconds/step and about 53.4 GiB per GPU. The measured full sequential queue estimate is 48--60 hours.
- The 19:29 CST r7 production ProtGPT2 exact-cache attempt was stopped before completion after a read-only audit found that its receipt omitted complete imported-code binding. It produced no completion receipt and is retained only under an explicit `ineligible_unbound_code_receipt` suffix.
- A replacement r8 archive verifies one exact eight-file code/config inventory before any local import and propagates its archive/manifest binding through v3 provenance, reports and receipts. Its bounded BF16-to-FP16-cache preflight passed in 12.83 seconds and remains nonconfirmatory. At 21:12:34 CST the single full-panel r3 queue started on GPU 3 as launcher PID 16502, initially running ProtGPT2 as PID 16522. The archive, content-manifest, profile, runner and launcher hashes are recorded in EXP-R2-026. A machine-readable receipt additionally binds the full Python package environment to those hashes.
- The 48--60-hour estimate applies only to the preliminary online TopK queues. The separate alternative-dictionary screening/full plan freezes 2,125 aggregate H200 GPU-hours excluding extraction/evaluation, roughly 22 idealized days at four GPUs and longer in calendar time. No held-out dictionary-quality or P0 scientific gate has passed from launch or preflight liveness.
- Closed a downstream precision mismatch: P0-3/4/5/6/8 production paths are now bfloat16-only, verify every floating parameter before first activation, check all required activations/attentions/logits for finiteness before conversion or use and propagate versioned fail-closed integrity receipts. The integrated focused suite passed 73 tests plus 2 subtests and the complete suite passed 182 tests plus 6 subtests. This was CPU contract validation; the scheduled pretrained-model runs remain outstanding.
- Added the minimal real-pretrained P0-7 adjudication layer: one shared module, one thin CLI and one example specification, with no model inference or feature rematching. It rehashes the complete positive-control/P0-2/P0-5/P0-6 chain and raw factorial evidence, retains every paired row, uses one global positive/TOST multiplicity family, requires fidelity and path localization, and publishes only after final atomic rehash. The focused suite passed 33 tests and the complete R2 suite passed 215 tests plus 6 subtests. No eligible real intervention surface exists, so P0-7 remains open.
- Rebuilt all six Paper A figures and both PDFs. The processed source package verifies 65 manifest rows, 68 checksums and 1,649,686 bytes; main and supplementary PDFs are 23 and 12 pages. Strict parsing passed for 58 current configuration/environment/source/provenance JSON files. Author, affiliation, funding, approval, tag and DOI metadata remain external blockers.
- At 22:14 CST the three canonical GPFS TopK logs remained finite through about 39.6k--39.8k steps and had complete step-35,000 checkpoints. The r8 exact cache remained alive with 173 GiB staged and no completion receipt. GPU, host-memory and GPFS headroom were safe. Superseded OSS logs were explicitly excluded from the production lineage; no H200 problem required user action.

---

## R2 H200 progress and storage cleanup (2026-07-18 CST)

- All three bfloat16 ProtGPT2 TopK seeds completed 200,000/200,000 steps with complete final checkpoint manifests, then advanced to ZymCTRL. At 12:09 CST, seeds 17/29/43 were finite at steps 23,650/24,200/23,600; no NaN, Inf, traceback or error metric row was present.
- The r8 exact-cache queue completed and receipt-verified the 221,184,000,000-byte ProtGPT2 cache at 07:54 CST, then advanced to active ZymCTRL extraction as PID 17403. The completion-receipt SHA-256 is `c5e323df9618c14db9ec4e19332dc1b162a4a026b68381b2b1fd3c7954a73225`.
- Current estimates are about 18 hours for the remaining online ZymCTRL phase, 31--43 hours for the online ZymCTRL-plus-ProGen2-medium queue, and a lower-confidence 14--22 hours for the separate exact-cache queue. GPU, host-memory and GPFS headroom remain safe; no H200 problem requires user action.
- Removed 743,059,375,694 apparent bytes (692.03 GiB) of superseded, non-resumable state after live-PID and completion-receipt guards: 630,138,917,454 bytes from GPFS and 112,920,458,240 bytes from OSS. GPFS usage moved from 18% to 17%.
- Retained compact failure/publication evidence, the explicitly suffixed r7 ineligible parent root and every active or eligible artifact. The 23,329-byte cleanup evidence archive has SHA-256 `7601f41f8a859ceb599215b29d717d1d31cad08b6d3afd7a95dcc062f45b09c1`; the path-level receipt is `evidence/h200_cleanup_20260718/cleanup_manifest.json`.
- Storage cleanup and process liveness do not pass P0-2 or any downstream biological or causal gate.

---

## R2 BF16 prerequisite completion, screening launch and storage cleanup (2026-07-20 CST)

- All nine preliminary online TopK runs (three models x seeds 17/29/43) completed step 200,000 with complete resumable final manifests and finite canonical logs by 23:58 CST on 2026-07-19.
- All three exact activation caches have independently revalidated schema-v3 completion receipts binding bfloat16 inference, finite captures, float16 storage, the exact code inventory and 641,433,600,000 payload bytes.
- Six receipt-bound validation-only ReLU/L1 and gated-SAE screenings started on four H200 GPUs at 18:10 CST. Their 45 candidates budget 125 aggregate GPU-hours, with an honest 2--4-day calendar estimate including validation and I/O. The later full panel remains 2,000 aggregate GPU-hours.
- Early health monitoring found all four workers training at approximately 54--62 GiB/GPU with no OOM, non-finite, traceback or no-space error. No H200 resource fault requires user action.
- A guarded cleanup removed 395,430,148,930 apparent bytes (368.27 GiB): nine redundant step-195,000 resumable predecessors and one obsolete 20-step fp16 smoke. Compact metadata and a checksum receipt were retained.
- All nine step-200,000 finals, all 63 prescribed 25k--175k trajectory snapshots, all exact caches and active screening artifacts remain. The path-level receipt is `evidence/h200_cleanup_20260720/cleanup_manifest.json` (SHA-256 `f1d2e4dea103f4bf39ae125c37e4307adfd5100750efd64d9375f795401321a7`).
- These are prerequisite, validation and storage-maintenance facts only. P0-2 and every downstream biological/causal gate remain open.

---

## R2 screening OOM audit, lifecycle amendment, cleanup and fresh restart (2026-07-21 CST)

- A terminal inspection corrected the status of the July 20 validation-only dictionary screenings. All four r1 runners had completed candidates `000`--`002` and then OOMed on candidate `003`. Twelve of 45 candidates had diagnostic validation products, 33 had not run, no terminal `results.json` or `run_manifest.json` existed, and the queued ProGen2-medium and ZymCTRL ReLU/L1 invocations had never started.
- Peak allocation increased by one prior candidate's model/Adam state: ProGen2-medium gated SAE increased by exactly 32,472,373,248 bytes per candidate and a 36-layer gated path by exactly 37,257,506,816 bytes. The failure was an application CUDA-object lifecycle defect. All GPUs were idle, host memory and GPFS had ample headroom, and no H200 fault required user action.
- The r1 output was renamed `p0_2_dictionary_controls_bf16_r1_ineligible_candidate_memory_accumulation_20260721` and cannot be resumed or pooled. After checkpoint-to-validation hash guards, 12 progress and 12 best payloads totalling 576,951,858,714 apparent bytes (537.328 GiB) were removed. Twenty-eight compact validation/timing/run-state files remain. The path-level cleanup receipt is `evidence/h200_cleanup_20260721/cleanup_manifest.json`, SHA-256 `069714b1edc7d961cfe3353f4e3c9249f60255dbad7c41ab5167e48ff59ef76d`.
- The minimal correction clears optimizer state and gradients, deletes candidate-local checkpoint/model objects and applies a stable post-candidate CUDA allocation guard of at most 128 MiB. A four-repetition, full-width ProtGPT2 gated-SAE preflight completed at 14:20:58 CST with identical 62,162,163,712-byte allocation peaks and exactly 67,108,864 bytes allocated after each repetition. It accessed no test data and retained only its nonconfirmatory report.
- Fresh r3 validation-only queues started at 14:22:06 CST on GPUs 0--3 as launcher PIDs 27369--27372 under `p0_2_dictionary_controls_bf16_r3/screening`. The immutable archive, deployed-tree and launcher SHA-256 values are `6f191b554de901c7be25968f2fc96d989ae7d5d0bcb2a1c9285fdd1c3b840e44`, `486ba4e9c00a7a1a88a58a31691095e72aaf45f39745ad3117b88c694de579d4` and `bc511f78f872ce70cc85ed098ec9aad15e44aa2dc1a5add3de2a52683fcef2f6`. Initial child processes were alive. The current observed estimate is 16--24 hours; the frozen conservative budget remains 125 aggregate GPU-hours.
- The complete R2 suite passed 216 tests plus 6 subtests in 35.48 seconds; Ruff check/format, byte compilation and shell syntax checks were clean. These are execution-integrity facts only. The screening remains validation-only, has zero test access and cannot pass P0-2.
- The unused repository `.venv` was also deleted, reclaiming 389,023,458 apparent bytes. It used Python 3.13, contained plotting tools but no PyTorch and was not held by a process. The canonical environment is now `source ~/miniconda3/etc/profile.d/conda.sh && conda activate ct`; historical commands remain unchanged as provenance.

---

## R2 P0-2 adjudication-contract closure (2026-07-21 CST)

- The dictionary gate now treats an exact terminal `sparsity_match_failure` as a complete calibrated negative only when the candidate grid is complete, test access is zero, no row matches the frozen L0 interval and no configuration/checkpoint is selected. It omits and forbids that model/method's three full seeds, emits `atlas_eligible=false` and remains rejected by downstream eligibility consumers. Gate SHA-256 is `2b7bcd627dd80c3d140588451cf767982bbaf21015f68a34e907ed05ab75bc6c`.
- Added one shared mask-receipt producer and thin CLI. The durable receipt at `evidence/p0_2_mask_validation_20260721/mask_validation_receipt.json` has SHA-256 `5966e274881984b2eeabeedd749d94c313fce02fb814911f0d1082ac3c3232db`. It binds the exact screening module and test file, records both required mask tests as passed and validates through the production gate consumer.
- The earlier 216-test plus 6-subtest result remains the lifecycle-only snapshot. The final integrated pre-documentation suite passed 223 tests plus 6 subtests in 36.28 seconds; focused gate/mask verification passed 16 tests. Ruff check/format, AST and shell syntax, and strict JSON checks passed.
- This closes adjudication and mask-receipt contract gaps only. It does not inspect held-out test rows, pass P0-2 or authorize full runs before terminal screening validation.

---

## R2 terminal screening, calibrated selection and full launch (2026-07-22 CST)

- The fresh r3 validation-only screen completed all 45 frozen candidates with zero test accesses. It selected ProGen2-medium gated SAE (candidate 0; L0 132.16376074074074; FVU 0.4397525937186076), ZymCTRL gated SAE (candidate 3; L0 120.02304111111111; FVU 0.5929956473037449) and ZymCTRL ReLU/L1 SAE (candidate 0; L0 131.2233588888889; FVU 0.5899043000429988). ProGen2-medium ReLU/L1 SAE, ProtGPT2 gated SAE and ProtGPT2 ReLU/L1 SAE reached terminal calibrated `sparsity_match_failure`; their three seeds each are forbidden. The terminal receipt is `evidence/p0_2_screening_20260722/terminal_receipt.json` (SHA-256 `83bf4563a27886a1e1f148eb4094f9a7b02533936a02981caac17a805de04d6c`).
- The exact 27-run receipt-authorized full panel was scheduled at about 11:46 CST across four fresh queues on GPUs 0--3 (launcher PIDs 30587--30590) under `p0_2_dictionary_controls_bf16_r3/full`, in fixed 7/7/7/6 queues. The launcher revalidates the exact cache and terminal screening lineage before launch, then is fail-closed on each completed full result/manifest/state/checkpoint and `test_evaluation_count=1`; only after those checks does it remove the redundant completed-run `progress.pt`. The initial launch receipt SHA-256 is `306d2b56bf3cf756fa5c56dd86e0f24d3555fc6c49c02d59747216e31d3104b9`. By 12:45 CST, all four first runs had exact `in_progress` states and atomic best/progress checkpoints at step 5,000; GPUs were active at 60,793--60,825 MiB, logs were clean and GPFS had 48,192,404,062,208 bytes free. The frozen budget is 1,500 aggregate GPU-hours; the conservative longest queue estimate is about 388.9 hours plus evaluation/I/O, while observed screening throughput (queue 3 is likely critical at about 4.8 days before margin) supports a provisional 5--7 calendar days.
- Receipt-preserving cleanup removed 45 progress and 42 nonselected best checkpoints (2,103,965,204,225 apparent bytes; approximately 1.9135 TiB), retaining three selected checkpoints and 108 compact files. The local cleanup receipt SHA-256 is `f646ad90abe8554c70357860285dda4051147fc0d6d6e95c0887f97fe62befe3`.
- Local verification passed 223 tests plus 6 subtests in 36.32 seconds and the focused reviewer suite passed 30 tests. P0-2 remains open pending aggregation and adjudication of all 27 full runs; the failed pairs are calibrated dictionary-quality negatives, not biological claims. No H200 issue requires user action.

---

## Transfer-gap comparison resources staged (2026-07-24 CST)

- Frozen and validated GPT-2-large, an OpenWebText screening subset, WikiText-103 v1, all 67 BLiMP configurations and CATH-S40. The deterministic OpenWebText split contains 300,000 train, 5,000 validation and 5,000 test documents with exact-text hash disjointness.
- The 92-file, 5,753,830,070-byte payload and its checksums are available on H200 GPFS at `/gpfs/jiaotongdamoxing/zhk_zip/biocc/external_resources/transfer_gap`. Every checksum passed after extraction, and the active GPU pod independently read the GPT-2 checkpoint and manifest.
- The workstation-side selection builder is `ops/prepare_openwebtext_screen_subset.py`; the frozen revisions, paths and integrity hashes are recorded in `external_resources/manifests/transfer_gap_resources_20260724.md`.
- Transfer used a 0-GPU pod and did not change the active P0-2 queues. During transfer validation, the H200 master path named `/gpfs` was found to be local `ext4`, while real GPFS is mounted only in pods. The `~/hangzhou-remote` health and transfer helpers were corrected to verify the filesystem and hand files through an explicitly selected pod.

## 2026-07-24 — R2 transfer screen executed; check.md rewritten

Ran a matched text-vs-protein interpretability transfer screen (EXP-R2-025, TG-01..TG-07) on GPT-2-large, ProtGPT2, ZymCTRL and ProGen2-medium. The screen refutes three candidate explanations for R2's negatives (too little computation; order-invariant composition; protein-specific cost of frozen attention) and identifies the two dominant causes: a variance-versus-behaviour decoupling that makes FVU-style dictionary gates unreliable and, across the three protein models, exactly inverted relative to behavioural fidelity; and an explanation channel roughly ten times narrower than text within a sequence, which makes the within-protein matched null structurally degenerate for family-level labels.

Immediate operational consequence: add a loss-recovered splice metric to the P0-2 dictionary panel before adjudicating the 27 full runs. Scripts under `scripts/transfer_gap/`, results under `results/transfer_gap_20260724/`, full writeup with derivations and eight proposed methods in the repository-root `check.md`.

## 2026-07-27 — P0-2 closed; prospective P0-2b fidelity qualification completed

- The original frozen P0-2 gate was executed unchanged after all 27 eligible full runs completed. Receipt SHA-256 is `3f472afb0171836ad7f51d5c1d9e25b1d8d4a67aae1ac1aa51744bf18494f9b3`; the panel fails, with only ProGen2-medium TopK atlas-eligible.
- `check.md` now distinguishes the single-seed TG-01--TG-07 pilots from confirmatory evidence and contains a binding prospective amendment. It no longer retroactively replaces P0-2 or states pilot hypotheses as established causes/refutations.
- P0-2b evaluated all 27 completed checkpoints on a new 240-sequence, hash-selected, prior-manifest/source-prefix-excluded cohort with native conditioning and exact reinjection. Executed spec SHA-256 is `11092e16891922380a2d224288138968e45ea982d36b46a4007f9f6b76a8d6bf`.
- No sparse model/method qualified. ProtGPT2 and ZymCTRL failed the prespecified CE-denominator guard; ProGen2-medium had valid denominators but its best TopK seed recovered only 0.368 loss and 0.262 KL, versus 0.80 required. Aggregate SHA-256 is `7f68e08775af87a171f2e4aac2a0d88cf235280280bf1bac44f76b0a2959bd07`.
- The production ProGen2 panel descriptively aligns lower FVU with better behavioural fidelity (`rho=-0.867` for loss recovered), so it does not support generalizing the pilots' inverted-ranking claim.
- All reinjection gates passed. The longest queue took 267.383 seconds; aggregate accelerator time was 0.2141 hours and peak allocation 13.264 GiB. Results add only about 6.1 MB. H200 and GPFS were healthy.
- The exact caches/checkpoints remain retained as scientific inputs. Per the frozen decision rule, P2--P8 and legacy downstream experiments remain stopped. Detailed results are in `docs/analysis/P0_2B_DICTIONARY_FIDELITY_RESULTS_20260727.md`.

## 2026-07-28 — R2 induction probe corrected for substitution tolerance; deficit survives

- Prior work found late (Pomerants et al., arXiv:2602.23179 v5) showed that in protein language models approximate-repeat detection subsumes exact-repeat detection, putting the R2 induction census's exact-repeat probe under suspicion of measuring a special case biased toward the text arms.
- A substitution-tolerant natural-repeat probe was added alongside the exact one (never replacing it) in `src/transfer/circuits.py` and `scripts/transfer/04_circuit_primitives.py`. Criterion: ungapped, exactly two occurrences, no indels, at most 50 per cent substitution (all from the prior work), with the mean BLOSUM62 score over the substituted positions required to be non-negative. Geometry held at the exact probe's values.
- The 48-protein cohort ceiling was an artefact of exactness. Full census of the 203,063 eligible EC-labelled entries: exact 48, approximate 817, a 17.0x lift.
- The five-times head-count deficit is not a probe artefact. It holds or widens under the approximate probe on every arm and both cohort sizes (ProGen2-medium 5.00 -> 6.60, ProtGPT2 3.57 -> 3.67). The ProtGPT2 QK/OV dissociation holds. The peak-sharpness result survives in sign only: the ProGen2-medium advantage over gpt2-large falls from +21 per cent to +3 to +9 per cent and can no longer carry "sharper", only "at least as sharp".
- Text has no honest BLOSUM62 analogue, so the text control is the same criterion with the similarity rule dropped -- strictly more permissive than the protein criterion. Recorded as an asymmetry, not corrected.
- Detail, null calibration, per-arm tables and the full limitation list are in `docs/EXPERIMENT_LOG.md` under EXP-R2-048.

## 2026-07-28 — R2 induction deficit restated against a scale-matched expectation: 2.3x, not 6.5x

- The 0.10 threshold is not load-bearing. On the synthetic probe the text/protein ordering holds for every raw cut in one contiguous band 0.00525-0.4196 (62 per cent of the informative grid), which contains 0.05, 0.10, 0.20 and 0.30 in its interior. The worst-text/best-protein ratio falls monotonically with the cut -- 5.51, 2.14, 1.71, 1.54 -- and reaches 1.24 under a per-arm mean-plus-3sd cut.
- Threshold-free statistics show the finding is a tail statement only. A Mann-Whitney AUC over the full head-score distributions does not separate the modalities (pooled 0.595; 4 of 12 arm pairs below 0.5; no pair stochastically dominates). Quantile dominance separates at q75-q99 and fails at the median and at the maximum. Model-level exact permutation (4 text, 3 protein, floor p=0.0286) gives complete separation on the fraction and on q99, and p=0.257 on the median.
- A within-lineage GPT-2 ladder was censused at matched settings (gpt2, gpt2-medium, gpt2-large, gpt2-xl). The induction-head fraction FALLS with scale, -0.272 log10 per decade [-0.455, -0.088], while the head count rises from 23 to 100. **A scale explanation is therefore not refuted on the text side**, and llama-3.2-3b's low value is explained by scale plus lineage rather than being anomalous.
- Restated against that ladder's scale-matched prediction, every arm falls below its prediction interval, and the modality separation is complete but smaller than previously quoted: worst text shortfall 2.40 (dialogpt-small) against best protein shortfall 5.62 (ProtGPT2), a ratio of **2.34x**, exact one-sided p=0.0286 at the attainable floor. The 6.50x mean ratio and the 2.14x range floor were the same effect measured without a scale reference.
- Corpus alone accounts for most of a modality-sized effect. Holding architecture, parameter count and tokeniser fixed: gpt2 against DialoGPT-small is 2.30x, ProGen2-base against ProGen2-medium 2.00x. Any statement of the modality effect must quote these beside it.
- The result is probe-conditional and this is a genuine negative. The scale-adjusted separation is 2.34x on the synthetic probe, 1.08x on natural exact repeats, and absent on natural approximate repeats, where DialoGPT-small has zero heads above 0.10. The programme's declared primary probe is the natural one.
- Infrastructure defect: `R2_TEXT_MODEL_BASE_DIR` pointed at a GPFS tree holding only gpt2-large while the by-name text checkpoints were staged elsewhere, so the completed nine-stage H200 campaign's convergence control silently fitted a text side of one model. Path corrected and the missing checkpoints staged.
- Detail, tables, the correct probe-cluster bootstrap design and the full limitation list are in `docs/EXPERIMENT_LOG.md` under EXP-R2-057.

## 2026-07-29 — R2: scale explains ~0% of the induction spread; the corpus confound is not addressable

- Variance decomposition on log10(induction fraction) across ten arms: scale given modality and lineage adds **+0.003** of R-squared. Size does not explain the gap. Modality given scale and lineage adds +0.220, lineage given scale and modality +0.061.
- But only 25.4 per cent of the modality indicator survives projection off scale and lineage, because the only architecture family spanning both modalities is GPT-2 -- five text arms against one protein arm (ProtGPT2) once ZymCTRL's zero is dropped. The modality increment rests on a single protein model.
- Corpus and lineage contrasts, matched on architecture and parameter count, with probe-cluster intervals: gpt2/DialoGPT-small 2.30x [2.30, 2.40]; ProGen2-medium/ProGen2-base 2.00x [1.20, 2.00]; gpt2-medium/qwen2.5-0.5b 1.93x [1.82, 2.04]; gpt2-xl/llama-3.2-3b 2.15x [1.92, 2.29]. The matched-pair modality contrast is 5.38x [5.23, 5.54] -- 2.3 to 2.8 times the largest nuisance contrast, and it varies modality, corpus and repeat prevalence together.
- **The scale analysis cannot address the corpus-repeat confound and does not claim to.** Approximate repeats occur in 32.3 per cent of text documents against 0.402 per cent of protein entries; an induction head is only useful on a corpus that repeats; and no protein decoder trained on a repeat-rich corpus exists or can. A surviving modality coefficient is equally consistent with efficient allocation against the data.
- Convergence control re-fitted with the diversified text side (14 ladder members, five text arms in the induction fits including llama-3.2-3b). The induction coefficient survives: it shrinks 5-9 per cent and all six fits still exclude zero. Overall verdict unchanged at `underpowered`, which concerns the primary pathway metric rather than induction.
- DialoGPT-small is excluded from every convergence fit by the pre-existing denominator floor -- realized information fraction -0.5345 on OpenWebText -- so part of the 2.30x text corpus contrast may be an off-distribution effect.
- Second staging defect found and repaired: `R2_TEXT_MODEL_DIR` and `R2_TEXT_MODEL_BASE_DIR` addressed different GPFS trees, which dropped gpt2-large from the first re-run. Both now resolve to the same directory.
- Detail in `docs/EXPERIMENT_LOG.md` under EXP-R2-057 and its four addenda.

## R2 Instrument-Transfer Campaign (2026-07-29 CST)

- Ran the four-stage instrument-transfer campaign over the full eleven-arm panel on four H200s through `scripts/transfer/run_transfer_h200.sh`. Stages: `cohort_power`, `pathway_budget`, `estimand_power`, `lens_family`. Runs `20260729030034_d3646e37488d` (primary) and `20260729033029_c7503e1da439` (skip-offset sensitivity). Full detail in `docs/EXPERIMENT_LOG.md` under EXP-R2-060.
- Results: `results/transfer_20260729_instrument/` and `.../transfer_20260729_instrument_skip4000/` (pulled from GPFS).
- Headline measurements, all with a matched text control under one code path:
  - Ten of eleven arms measurable; DialoGPT-small is unmeasurable on the shared text cohort at every stage.
  - Pathway balance separates the modalities without overlap: MLP-to-attention ablation-cost ratio 1.50-2.10 across six text arms, 0.55-1.12 across four protein arms, and the separation survives two non-GPT-2 text architectures.
  - Estimand power: 52 of 76 estimands are attainable on the text control gpt2-large. Every single-submodule estimand is unattainable on it, and single-MLP attainability falls monotonically with depth inside the GPT-2 ladder, so that estimand's failure is a depth property rather than a modality one.
  - Lens family: the tuned lens improves on the logit lens at every non-identity layer on all nine capability-eligible arms, protein included.
  - Skip-offset sensitivity: text arms move by at most 0.018 nats between disjoint corpus blocks; ProtGPT2 moves by 0.599 and the ProGen2 pair by 0.16-0.23, quantifying the standing cohort-selection hazard as a magnitude.
- `scripts/transfer/` changes: eleven-arm panel in controller and worker, explicit arm-modality and model-path routing, lens capability gating, held-out unigram estimator and seeded-permutation cohort draw in `01_cohort_power.py`, content-deduplicated Markov held-out block, and a corrected `03_estimand_power.py recommend` caller contract.

## R2 Documents, Logs and Results Reorganised Against the Objective (2026-07-29 CST)

- Scope: `{docs,manuscript,evidence,preregistration,literature,logs,results}` and repository-root `docs/`. No change to `src/` or `scripts/`. Detail in `docs/EXPERIMENT_LOG.md` under EXP-R2-065.
- Retired scope archived to `archive/legacy/r2_retired_scope_20260729/` with a provenance README carrying a what / why / where-it-lives-now table: the npj Artificial Intelligence manuscript package (11 MB), the six `P0_*` protocol and receipt documents, three `NPJ_*` documents, the npj manuscript assessment, the 2026-07-16 procedure-and-results audit, the recoverability preregistration, the 20-paper npj literature corpus (40 MB), twelve retired-scope result trees (127 MB, `circuit_analysis/` alone 105 MB), seven operational evidence trees, and the R1/April runtime logs.
- 103 MB of point-in-time copies of *live* transfer result trees moved from `logs/` to `archive/legacy/r2_transfer_log_snapshots_20260728/`, with a per-snapshot identity table. They are not duplicates: earlier snapshots differ from the current tree in up to 15 files each, so they hold pre-correction states.
- Kept live and stated why: `results/final_checkpoints/` (45 GB), the `transfer_*` trees including `transfer_gap_20260724/` which the audit document cites, the `evidence/p0_2*` receipt chain behind limitation L1, and the April CLT training logs.
- Sizes: `results` 47,296 MB -> 47,170 MB; `logs` 105 MB -> 2.3 MB; root `logs` 6.0 MB -> 5.7 MB.
- Documents rewritten to describe the single current objective: repository-root `README.md` (Paper A / Paper B framing, drug-design claim discipline and canonical pointers into retired trees all removed), `r2_.../README.md`, `r2_.../docs/README.md`, `r2_.../docs/RESEARCH_PLAN.md`, `r2_.../docs/methods/TRANSFER_MEASUREMENT_PROGRAMME.md`, `docs/DOCUMENT_INDEX.md`, `docs/README.md`. `docs/PROJECT_STATUS.md` gained a true current-state head; its dated snapshots are retained verbatim below it.
- Two defects found and fixed in `.gitignore`: the R2 results, logs and wandb lines still named the pre-rename root `r2_decoder_sparse_readout_audit/`, so `logs/` was no longer ignored; and a bare `DOCUMENT_INDEX.md` line in the retired-root quarantine block had been excluding the live `docs/DOCUMENT_INDEX.md` from version control since 2026-07-16.

## 2026-07-29 — Repository and remote operations audit

- Audited the live code, contracts, statistics, H200 orchestration, evidence, documentation, and storage under the Development Principles. Repairs and accepted limitations are recorded in `docs/ENGINEERING_AUDIT.md`.
- Moved frozen provenance and retired live configuration from the repository to `/Data2/lzp/bio_archive` using copy-then-verify semantics. The external `MIGRATION_SHA256SUMS` verifies 2,928 regular files; `docs/ARCHIVE.md` records the boundary.
- Preserved final checkpoint hashes and before/after remote inventories under `evidence/checkpoint_receipts_20260729/`, then removed intermediate checkpoints and optimizer-only state. The idle four-GPU pod and temporary zero-GPU transfer pod were released; the cluster reported four schedulable H200s afterward.
- Unified active runtime resource variables under `TRANSFER_*`, moved new remote outputs under the InterpretabilityTransfer GPFS root, and retained historical identifiers only in immutable records and artifact schemas.

---

## 2026-07-30 — Repository-wide audit and repair; two plan items executed

- Audited the live code in four regions under the Development Principles: the H200 controller and worker, `src/transfer`, `scripts/transfer`, and `scripts/transfer_gap`. Findings, corrections and the two overstated claims that were rejected rather than acted on are in `docs/EXPERIMENT_LOG.md` under EXP-R2-068; engineering state is in `docs/ENGINEERING_AUDIT.md`; scientific consequences are in `docs/INTERPRETABILITY_TRANSFER_AUDIT.md`.
- Found that the remote-execution layer does not propagate a remote exit status: both access helpers return 0 whatever the remote command exits with. The campaign verdict, the remote code-freeze verification and the invocation-manifest push all depended on it, and no invocation manifest had ever reached GPFS. Repaired inside the repository — the worker states its status on its last line and every remote predicate answers on stdout — and verified against the live cluster in both directions. Catalogued as limitation L20; a second host-portability defect in the generated panel contract is L21.
- Plumbed one declared corpus draw seed through every campaign stage, replacing head-of-file prefixes; measured the exposure at 342 versus 398 distinct sequences per 400-record draw on the qualifying band.
- Ran plan item B3 (explanation channel, CPU, two seeds) and plan item B6 (non-local propagation, ~1.5 H200 GPU-h, four per-arm campaigns), and ran TG-01 for the first time. Artefacts under `results/transfer_20260730/` and `results/transfer_gap_20260729_corrected/tg01/`.
- Test suite grew from 205 tests plus 9 subtests to 303 plus 34. Ruff, both generated contracts, and shell syntax checks pass.
- `EXP-R2-067` is referenced by comments in committed code but has no log entry; that work is unlogged and this session took `EXP-R2-068` rather than collide with it.

## 2026-07-30 / 07-31 — cohort-draw robustness, and the first localised method-transfer failure

- Re-ran the pathway-budget stage on corpus-wide seeded cohorts across all twelve arms; it reproduces to two decimal places and the text/protein ranges still do not overlap. The strongest part-1 result had never been re-measured since the cohort draw was fixed.
- Narrowed the far-band propagation claim from a modality claim to a lineage claim, then resolved it with a five-window equal-case campaign: the separation is real but confined to the large-effect tail (39 of 40 comparisons at threshold 0.50, 31 of 40 at 0.05).
- Withdrew **two** pre-registered gates as specification defects after showing neither is attainable on its own text control — the ≥30 far-band case gate and the top-20 Jaccard ≥ 0.8 causal-agreement gate. Appendix B rule 2 had been violated in the recording direction in both cases.
- Built the instrument audit item D2.b needed and ran it. `causal_census_agreement` refuses a census-selected sender set rather than documenting the precondition, because the circular result is indistinguishable from a real one in the output. The measurement found the programme's first **localised method-transfer failure**: the prefix-matching census predicts causal rank on the text control and carries essentially no information about it on the architecturally-matched protein arm.
- Established the protein-side cohort-sensitivity asymmetry on equal footing for the first time — between-window spread separates by modality at identical case counts, so it is no longer arguable as a sample-size artefact.
- Test suite 318 → 324 plus 34 subtests. Ruff, both generated contracts and shell syntax pass. Three operational faults of my own are recorded in `docs/EXPERIMENT_LOG.md` rather than smoothed over: a panel-scoped stage serialised onto one GPU, a chained driver whose process match caught its own launch wrapper, and a transport timeout that killed one job.

## 2026-07-31 — A withdrawn headline, a strengthened one, and the estimands repaired underneath both

- **Withdrew the far-band propagation result as a model claim.** `DISTANCE_BANDS` was a token-unit module constant compared across arms differing 4.4x in symbols per token, while the artefact recorded `symbols_per_token` and nothing consumed it. The ordering between the two *protein* arms — same content alphabet, only the tokeniser differing — reverses between a token band and a residue band, so the sign of the contrast was a free parameter of an unstated choice. EXP-R2-070's arithmetic reproduces exactly; the estimand was not identified. **D1.a reverts from answered to open** and the lineage reading is withdrawn along with the modality reading. Catalogued as L23; standing rule 26.
- **Promoted EXP-R2-071/072 to the programme's strongest part-2 result, and it now holds on ten arms.** With every head patched, the prefix-matching census ranks causal importance on text decoders and not on protein decoders — complete separation in all 40 arm × condition cells. The alternative explanation was ruled out by the control that was ordered first for exactly that purpose: `dialogpt-small`, a *text* decoder off-distribution at −4.08 nats, does not collapse. Catalogued as L22, and it earns the programme's first Part 3 opening that is traceable rather than assumed.
- **Corrected five published figures**, each re-derived from the shipped artefacts before being recorded: two remaining-head correlations that were not reproducible at all, a cross-arm effect comparison that reverses once a 32x denominator spread is undone, a reliability range computed on the sampling unit the module declares wrong, and the modality-variance-surviving statistic, which was divided by an uncentred sum of squares whose ceiling is below one. The concentration statistics are suspended pending a noise-corrected variant.
- **Answered D1.b on artefacts already on disk, at zero GPU cost**, after repairing the instrument it needed — `12_induction_robustness.py` was hard-wired to a results tree that no longer exists and raised on every path. The tail statement stands and is now located: the ordering separates at every quantile from q75 to q99 and fails at the median and in the extreme upper tail.
- **Found that the twelfth arm breaks the synthetic probe.** ProGen2-small was admitted after §4's probe table was written; on twelve arms the ordering inverts at 0.20 and 0.30 on all three probes, and at the headline threshold the natural-approximate margin is smaller than one of its 192 heads. The matched-pair ratio survives; the worst-text-above-best-protein ordering does not.
- **Repaired the estimands and statistics underneath all of it.** The far-band band and its corruption span are now declarable in content symbols and resolved per arm; stage 04 defaults to float32; the headline concentration and rank-split statistics have an implementation and tests for the first time; the bootstrap unit floor reaches the resamplers that lacked it; out-of-lineage extrapolations are flagged; and the test suite no longer writes synthetic campaign records into the operational controller-log directory, where 189 of 294 files turned out to be its own.
- **Kept the cluster busy throughout.** Three GPUs were idle on waking; a cross-lab far-band control was validated in-pod and launched within the hour, then stopped after three of five windows once llama-3.2-3b proved slower per job than the successor campaign was worth waiting for. The symbol-matched successor was validated and launched on the freed cards, and a fourth lane was added when an orphaned census released its GPU.
- Test suite 324 → 434 plus 34 subtests. Ruff, both generated contracts and shell syntax pass.

## 2026-08-01 — Two owed checks on the headline result, and the repairs that unblocked them

- Picked up the previous session's wake point, which named D2.c as next and asked for a design pass. The design pass returned **not schedulable as specified**: ProtGPT2 cannot build a PAA instance pool at all (0.349 tokens per residue against a 512-token width and an 800-residue band), ZymCTRL admits exactly one residue length, and the selector scores the nearest antecedent while the causal statistic removes every one of them — a median of 3 occurrences on text against 13–17 on protein, which would manufacture part of the modality contrast it is meant to test. Recorded in the audit document with the cost re-derived at 12–25 H200-hours against the table's ~13.
- One D2.c blocker was **cleared**, and it was the load-bearing one. The plan believed from L5 that the copy-suppression selector fails to rank causal effect even on the text control. That figure is on the *signed* effect; on the *magnitude*, which is the statistic L22 uses, the same heads on the same artefacts read +0.53 to +0.65 against gpt2-large's induction +0.428 to +0.507. L5 stands as written; the inference drawn from it does not.
- Launched **EXP-R2-077**, the cohort-draw check EXP-R2-072 records as owed, on the four arms that define L22's separation margin. Found before spending GPU time that ProtGPT2 fails 64-case parity on two of four draws tested, so its parity in EXP-R2-071/072 was itself draw-contingent and nothing said so — a new instance of the protein-side cohort sensitivity, reaching the case set rather than the estimate.
- Extended `path_patching` to the llama and qwen2 layouts, lifting an instrument limit catalogued against L22, whose every text arm was GPT-2 architecture — the configuration that retracted the QK/OV finding. Validated end to end on a real Qwen2.5-0.5B checkpoint: head-write linearity at 2.2e-06 relative error under grouped-query attention. **EXP-R2-079** is chained behind the draw check with its expectation pre-registered, because the text scale trend predicts llama-3.2-3b at ≈+0.29 — within reach of the protein maximum — so the margin has to be interpretable before it is read.
- Corrected a published number. `head_effect_reliability` paired each standard error with the wrong centre; ZymCTRL's approximate-repeat reliability is **0.170**, not the 0.008 the audit carried as "a grid that cannot be ranked at all". The exact-case range is unaffected, so nothing resting on it moves. Two further latent defects repaired: `content_low` did not survive the pool round trip, and `cluster_bootstrap` accepted a percentile its replicate count could not resolve.
- Test suite 434 → 441 plus 34 subtests. Ruff, both generated contracts and the panel contract regeneration all pass. Four H200 GPUs at 100% throughout.

## 2026-08-01 — L22 architecture-controlled and attenuation-tested; the D2.c gate decided; a campaign recovered from a transport drop

- **L22 became architecture-controlled and draw-robust.** EXP-R2-077/079 landed, taking the head-prevalence census claim to 48 arm × condition cells over seven text and five protein arms, with the two rotary/GQA text arms removing the confound that every text arm was GPT-2 architecture. Reference comparisons had to be re-pinned at master seed 20260728 first: the initial read mixed case-seed and corpus-draw variation, which is the same conflation the campaign existed to separate.
- **Tested the most serious available objection to L22 and it survived.** If per-head effects are noisier on protein, the rank correlation is attenuated hardest exactly where the hypothesis wants it — and measured reliability *is* lower on protein (ZymCTRL 0.466 against 0.745–0.865 on text). Classical disattenuation leaves the ordering intact in all 24 measured cells. Recorded as a lower bound, because only the causal side carries a measurable reliability; the selector's own precision is unmeasured, and that is the more interesting half.
- **Resolved EXP-R2-072 item (v), both halves.** The document deferred two statistics to "the noise-corrected variant now implemented in `path_patching.py`". That variant does not exist — `effect_concentration` computes a Gini on raw magnitudes and the similarly-named `head_effect_reliability` is a different statistic. One half was corrected in the morning and the other, a suspension of §(ii) resting on the same false premise, was still telling readers not to cite a table whose every number reproduces exactly.
- **Inverted the D2.c blocker-1 prescription.** The fix is the **pool width**, not the cohort band: `--width 192` with the band unchanged admits the matched pair at **zero L13 exposure**, where the band route would have cost 1.75–2.31 nats, 1.7–2.3× the incident that created L13. Found while re-measuring, and only after a verification disagreement traced to my own naive tokenisation ignoring ProtGPT2's declared `fasta_wrapped` rendering.
- **Ran a deliberate audit for document-versus-code drift after hitting two such defects by accident.** Twelve findings, ten corrected: a flag name that exists nowhere, a rule naming an empty path while guarding ~47 GB, "five structural invariants" where there are nine, a 1600-fold vocabulary ratio that is 4748. One finding was **rejected** on re-verification and one it left open was **closed** — both checkpoints are on B, and gpt2-large and ProtGPT2 are each exactly 774,030,080 parameters, so the matched-pair claim is now verified rather than asserted. L20 and L21 remain unaudited, not confirmed.
- **Recovered a campaign that had reported itself a near-total failure.** EXP-R2-080 declared 2 successes of 15; it had in fact produced **6**, and four verified results were discarded by a lane whose success test raced a still-computing worker. One transport drop at 03:17 left orphans holding GPUs, and a second defect then read "GPU occupied" — a statement about the machine — as a verdict on the draw, burning nine jobs through their candidate lists in four-minute increments without running any of them. Both repaired against the pod's own state rather than a timer, the four results recovered and digest-verified, and the nine relaunched.
- **Decided the D2.c gate, and re-specified it first.** The pre-registered criterion compared an exhaustive width-192 census against a restricted-range, unmatched-score width-512 target whose reproducibility at fixed width is ±0.12 — larger than the +0.08 width effect it was meant to detect. Withdrawn under Appendix B rule 2 on evidence measurable at width 512 alone, so the correction does not depend on the outcome. On the like-for-like replacement the gate **passes**: +0.4515 [+0.401, +0.498] exhaustive at width 192, inside gpt2-large's own induction band of +0.428 to +0.535. The width-512 fallback and its full L13 exposure are not needed.
- Four H200s and two B L20s kept at 100% throughout, including across a session suspension that all detached work survived.

## 2026-08-01 (night) — D2.c decided on ProtGPT2; the L22 statistic re-scoped; ProGen2 unblocked

- **D2.c ran and answered its question.** ProtGPT2's copy-suppression census orders
  causal importance at +0.154 to +0.244 over three matched draws, against a text
  control at +0.449 to +0.482 and against its own *induction* value of −0.226 to
  −0.006. The L22 failure does not reproduce on a second mechanism; a partial
  transfer does.
- **Three confounds eliminated before the deficit was called one**: instance count
  (the text arm reads higher at *fewer* instances), keys per instance (~5% of the
  gap, from a measured +0.0144-per-key slope), and two-sided disattenuation — the
  first time the census side's reliability could be estimated at all, at 0.996–0.999.
- **The finding that matters is the decomposition.** On both mechanisms the all-grid
  statistic reports the opposite of the stratum a census publishes: ProtGPT2's census
  top-20 retrieves more of the causally largest heads than the text control's, while
  its aggregate is far lower. On induction the bulk that drives the aggregate is at
  the noise floor on both arms.
- **§7 item 0 closes** — not on its pre-registered criterion, which it survives, but
  because its statistic is anti-correlated with what it was meant to certify.
- **A silent-wrong-answer path removed from the PAA census.** The attention tap took
  the pattern by tuple position, which is a key-value cache on ProGen2; it survived
  only because `use_cache` is off at every call site today. The tap now identifies
  the pattern by shape contract, verified bit-identical on the three arms that
  already worked and newly functional on both ProGen2 arms.
- **A second protein decoder is reachable**, at the ban depth its tokeniser needs:
  30108 instances against 2851, while the text control is unmoved, so the pair is
  matched at the relaxed depth rather than the protein arm taking an exception.

## 2026-08-02 (morning) — D2.c answered on effect size; the census's directional selectivity found to be text-only

- **K=6 moved nothing.** Three further draws per arm landed inside the existing
  ranges endpoint for endpoint, and the stratified separation widened. The claim
  that carried the burden at K=3 now carries it at K=6.
- **D2.c answered on the question its own plan row poses.** The absolute knockout
  effect is the same size on both modalities — 0.0257 against 0.0259 nats — and the
  arms separate only once each is expressed as a fraction of its own clean margin:
  0.4–2.2% on ProtGPT2 and 0.6–0.7% on ProGen2-base against 4.1–10.8% on matched
  text controls, disjoint on every draw. The two biases in the statistic both run
  toward the null.
- **The census's directional selectivity is a text-only property.** gpt2-large's
  census top is a 21-point enrichment for *promoting* heads against its own grid,
  with control heads at exactly 0.50; ProtGPT2's top is not enriched either way.
  A magnitude ranking transfers; which way the head pushes does not.
- **The mechanism label needs care on the text control too** — a census top that is
  80% promoting is ranking prediction-addressed attention, not isolating copy
  suppression.
- **A protein-family split is on the table and is explicitly not quotable yet.**
  ProGen2-base reads negative where ProtGPT2 reads positive, but the two were
  measured in different conditions; two campaigns are running to settle it rather
  than the split being reported with a caveat.
- **A noise explanation was excluded before the split was written down**, by three
  measures that use no census: ProGen2's causal ordering reproduces across draws at
  +0.527 against +0.78 on text — real but weaker, and attenuation cannot carry a
  correlation past zero.
- **L22's margin corrected onto a K-invariant statistic.** gpt2-xl reached K=7 and
  the "gap exceeds both boundary arms' draw ranges" clause fell to 2 of 4
  conditions — because an observed range grows with K and so penalises the
  better-measured arm. On a standard deviation the gap is 3.36–3.95σ in all four.
  The claim itself strengthened: gpt2-xl moved *up*, +0.437 → +0.458.
- **Cohort-draw provenance closed in three stages.** `02_pathway_budget`,
  `03_estimand_power` and `08_lens_family` accept `--cohort-draw-seed`, select
  their cohort with it, and hand-enumerated a `configuration` block listing every
  other knob while dropping this one; their results could not be traced to a draw.
  Every campaign so far used the default, so the gap was latent. A parameterised
  invariant test now covers all fourteen stages: a stage that draws a cohort must
  record which draw it used, or dump its whole namespace. Verified to fail on the
  pre-fix source and to name the offending stage. Suite: 464 passed, 4 skipped.

## 2026-08-02 (afternoon) — the protein split is an arm property, and the two D2.c statistics disagree

- **D2.c has two answers inside one modality.** Run in ProGen2's exact condition,
  ProtGPT2 moves by +0.012 and stays positive; both ProGen2 arms stay negative and
  disjoint from it. The confound recorded this morning is eliminated rather than
  argued away.
- **ProGen2-medium replicates ProGen2-base**, and the two differ only in
  pretraining corpus at identical architecture and parameter count — so the
  agreement is neither scale nor architecture.
- **The two readings of D2.c behave very differently as modality instruments.**
  Both separate text from protein — an earlier version of this entry said ranking
  did not, which was wrong. The difference is the cross-modality gap against the
  spread *inside* the protein modality: 15.5 for effect size, 0.57 for ranking,
  where the protein arms differ from each other by nearly twice what they differ
  from text. A statistic with more structure inside a modality than across it is a
  poor instrument for a modality claim, and that is L22's statistic.
- **The valence call survived being re-checked on a second ProGen2 arm.** Only
  gpt2-large's census top is directionally selective; all three protein arms sit
  within ±0.03 of their own grid once the measurable-head selection is removed.
- **The ProGen2 negative is a family property** — base −0.234, medium −0.211,
  small −0.119, three scales and both pretraining corpora.
- **A second text arm widened one result and qualified another.** gpt2-medium's
  relative effect is 13.13% at K=3, roughly double gpt2-large's, so the size
  separation is now 8× with no overlap. Its valence shift read −0.05 on one draw
  and the directional-selectivity claim was narrowed to gpt2-large on that basis;
  at K=3 it reads −0.128 and the two text arms span −0.13 to −0.22 against the
  protein arms' −0.015 to +0.038, so the claim is stated of text again. The
  narrowing was right on one draw and wrong by three, which is what K=3 was for.
- **A claim of mine was wrong and is corrected.** I recorded that the ranking
  statistic "does not partition by modality" because ProtGPT2 sits between the text
  arms and the ProGen2 arms. That does not follow, and the line came from a
  hard-coded print in a scratch script rather than a computed test; the script now
  computes it. All three statistics separate text from protein.

## 2026-08-02 (evening) — D2.c gets its cross-lab control, and a scale trend bounds the size result

- **The control D2.c was missing is in hand and confirms the modality reading.**
  Every D2.c number had come from the GPT-2 lineage or from ProtGPT2, which is
  gpt2-large's architecture, so the modality claim was equally consistent with a
  family claim. qwen2.5-0.5b — rotary/GQA, another lab, different tokeniser and
  corpus — reads text-like on all three statistics: ranking +0.486, size 12.29%,
  valence −0.18. The instrument needed no change to run on it: the attention tap
  built for ProGen2's shape contract works unaltered under grouped-query
  attention, and the zero-mask knockout check returned exactly 0.0.
- **A scale trend bounds the size result, and it is recorded rather than
  smoothed.** That statistic halves per doubling of parameters across both
  families measured, gpt2-medium 13.13% down to gpt2-xl 3.19%, while the protein
  arms fall far more slowly. It is not a head-count artefact — A4 is a maximum
  over the grid, which biases the big arms up. The like-for-like comparison is at
  matched scale, where the gap is ~12×; the whole-panel 3.6× rests on gpt2-xl and
  the trend licenses no extrapolation past 1.5B, there being no protein arm above
  774M in this panel.
- **The ranking statistic got worse as a modality instrument**, its
  gap-to-within-protein-spread ratio falling from 0.57 to 0.38 once gpt2-xl was
  added.

## 2026-08-03 — the D2.c size result is restated: the separation is the denominator

- **The absolute effect does not separate the modalities.** On ten arms, text
  spans 0.0156–0.0626 nats and protein 0.0089–0.0159, and ProtGPT2's 0.0159
  exceeds gpt2-xl's 0.0156. The two scales touch.
- **What separates is the decision margin, and it needs no interpretability
  instrument at all.** Mean clean M-gap is 0.270–0.518 on text against 0.996–3.297
  on protein — disjoint. Protein decoders are 2–12× more confident at
  prediction-addressed positions. The size result is therefore *the same
  intervention is a smaller fraction of a protein decoder's margin*, not *the
  mechanism is weaker in protein models*.
- **A4's denominator can cross zero.** dialogpt-small, off-distribution here,
  has a mean clean M-gap of −0.785, so its ratio has no defined sign. Excluded
  from the size comparison with the reason recorded, and logged as a limitation
  of the statistic rather than worked around.
- **The scale trend is within-family, not a scale law.** Monotone over four GPT-2
  points (124M 18.40% to 1558M 3.19%) with corpus and architecture fixed, but
  llama-3.2-3b at 3B reads 8.62%, far above gpt2-xl at half the size. An earlier
  entry called it a trend "across two families"; that rested on qwen alone.
- **And the denominator difference turns out to be about recurrence, not
  confidence (EXP-R2-111).** Measured at all scored positions rather than only at
  prediction-addressed ones — no census, no knockout — the margin does not
  separate the modalities at all, and on medians the protein arms are the less
  confident ones. What separates is the ratio of the recurrence-position margin
  to the all-position margin: 0.18–0.26 on four text arms against 0.90–1.78 on
  four protein arms, disjoint. Text decoders are 4–5× less confident where a
  token recurs; protein decoders are not. This is the most economical account
  available of why prediction-addressed machinery looks weak on protein arms, and
  it ties D2.c to L22: on this evidence the instruments are not failing to see a
  mechanism so much as there being much less of a task for one to solve.
- **That entry is retracted the same day (EXP-R2-112).** Its ratio compared the
  census's *filtered* PAA instances against unfiltered positions — a mismatch I
  recorded as a limitation instead of removing. Under one uniform definition of
  recurrence, every arm is *more* confident where a token recurs, so the effect is
  not about recurrence. It is about the census's instance filter, which excludes
  induction targets and requires a decoy pool and distance range: on text that
  strips the easy repeats, on protein it does not. The D2.c denominator gap is
  therefore a property of the **evaluation interface**, not of the models — which
  is one of the four places the Research Objective asks a limitation to be
  assigned, and a more useful answer than the one it replaces.
- **The alphabet interpretation is refuted by a byte-level English decoder.**
  ByGPT5 — text, 384-symbol vocabulary, 0.83 recurrent fraction — reads 1.47–1.52,
  inside the BPE text band and below every protein arm. A coarse alphabet does not
  make a text decoder behave like a protein one.
- **And the uniform statistic fails my own instrument test**, so no modality claim
  rests on it: cross-modality gap 0.064 against a within-protein spread of 11.53.
- **The census filter account is then confirmed directly** (EXP-R2-113): the margin
  at the census's instances over the margin at all recurrent positions reads
  0.143–0.199 on six text arms and 0.283–0.927 on four protein arms, disjoint.
- **The D2.c split resolves into two effects, one of them isolated.** A modality
  effect of ~0.26 is measured by the designed matched pair, which holds
  architecture, tokenisation and vocabulary fixed. A second effect of ~0.43
  separates ProtGPT2 from the ProGen2 arms with architecture, tokenisation,
  vocabulary and corpus covarying; ZymCTRL would break it and is excluded, so that
  exclusion now carries a cost it did not carry when it was decided.
- **Mode 2 restated on retrieval rather than the spread**, after llama-3.2-3b at
  K=4 took the panel's lowest spread while still retrieving at 14× chance.
- **§7 gains its first opening since item 0 closed**, traced to EXP-R2-113:
  difficulty-matched instance selection, with a stated gate and a null result
  declared a complete answer. Proposed, not implemented — the instrument is frozen
  and this changes its instance selection.
