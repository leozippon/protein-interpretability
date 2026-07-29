# Project Status

## Current state — 2026-07-29

**One objective, one live research root.**

The repository serves a single objective: compare language-generative and protein-generative models; analyse how well existing interpretability methods transfer to protein generative models; design and validate protein-adapted methods. Parts two and three are the deliverable. There is no second paper, no drug-design line and no venue commitment.

| Root | Scope | Status |
|---|---|---|
| `` | Text-to-protein interpretability transfer | **active — the only live root** |
| `archive/retired_research_roots/r0_shared_interpretability_framework/` | ProteinInterpret evaluation framework | retired 2026-07-29 |
| `archive/retired_research_roots/r1_encoder_interpretability_benchmark/` | Encoder benchmark (ESM-2 SAE + IndelMissense) | retired 2026-07-29 |

The canonical status of every scientific claim is `docs/INTERPRETABILITY_TRANSFER_AUDIT.md`, not this file. This file records where things stand operationally.

### Where the programme stands (2026-07-29)

- **Part 1 — differences.** Closed to new measurement (audit §9 Phase A). One difference survives in weakened form: fewer heads in the upper tail of the induction-score distribution on protein arms, probe-dependent, inverting at the headline threshold on the most defensible probe, with the modality increment carried by ProtGPT2 alone. Pathway budget separates text from protein with non-overlapping ranges and survives replacement of GPT-2-large by a Qwen2 and a Llama decoder; it carries two caveats and is not yet a claim.
- **Part 2 — limitations.** Nineteen catalogued (L1–L19), most demonstrated on the text control, which makes them properties of the method rather than of protein models. This is the programme's yield. The strongest single claim available is L5.
- **Part 3 — adapted methods.** Not yet earned. Two candidate directions carry standing rejections; no proposal is currently traceable to a measured limitation with adequate evidence.

Withdrawn on evidence and not to be asserted anywhere: the induction gap as a *modality* claim; the variance–behaviour dissociation (B2 returned NO, EXP-R2-062, and plan item C4 was dropped with it); the QK/OV dissociation; the relational channel; the tokenisation explanation; peak prefix-matching strength as a memorisation-free statistic; the gpt2 / dialogpt-small corpus contrast.

### Remaining work

Approximately 90–140 H200 GPU-hours, phased and costed in audit §9. Phase B (items B1, B3–B6) builds part 2 as measured evidence; Phase C (C1, C2, C3, C5) is hard-gated construction. B2 and C4 are spent and dropped respectively.

### 2026-07-29 reorganisation

- The R2 root was renamed `r2_decoder_sparse_readout_audit/` → ``.
- Retired scope — the conserved sparse-readout atlas, EC steering, enzyme design, the npj Artificial Intelligence manuscript package, the P0 protocol set, the npj literature corpus, the recoverability preregistration — is frozen at `archive/legacy/r2_retired_scope_20260729/` with a provenance README.
- Kept live because current work depends on them: `results/final_checkpoints/` (45 GB, the only local protein dictionaries, an input to plan item C1), the `transfer_*` result trees, the `evidence/p0_2*` receipt chain behind limitation L1, and the April CLT training logs.
- 103 MB of point-in-time copies of live result trees moved from `logs/` to `archive/legacy/r2_transfer_log_snapshots_20260728/`; that root's runtime logs fell from 105 MB to 2.3 MB.

---

# Superseded status snapshots

Everything below is retained verbatim as history. It describes the retired scope — two papers, a venue, a drug-design line — and its conclusions about that scope remain accurate *for that scope*. It is not a description of the current programme. Do not resurrect claims from it without checking them against `docs/INTERPRETABILITY_TRANSFER_AUDIT.md`.

Paths written before the 2026-07-29 rename and archive move are historical identifiers and are not normalized.

## Snapshot — 2026-07-27 (Paper A P0-2/P0-2b closed)

The live repository has two active research directions: Paper A under `` and Paper B under `r1_encoder_interpretability_benchmark/`. `r0_shared_interpretability_framework/` is shared framework work and seeds a future decoder benchmark. Verified duplicate legacy roots (`Research1/`, `Research2/` and `manuscripts/`) were removed from both synchronized mirrors; frozen exports and pre-pivot proposals are under `archive/`. Live research roots now follow `r<research ID>_<lower_snake_case_scope>` and remain at the same parent depth required by path-sensitive scripts.

| # | Paper | Home | Venue | Status |
|---|-------|------|-------|--------|
| A | Audit of cross-model sparse readouts and circuit diagnostics | `` | npj Artificial Intelligence | **Not submission-ready.** P0-2 and its prospective P0-2b fidelity qualification are closed with no panel/downstream sparse instrument. The proposed transfer explanation is not established, and downstream experiments are stopped pending a new protocol. |
| B | Encoder-only interpretability benchmark (ESM-2 SAE + IndelMissense v1) | `r1_encoder_interpretability_benchmark/` | Nat MI / Nat Methods stretch | Scaffolded (v0 tables in `r0_shared_interpretability_framework/`); manuscript to draft. Salvage the archived R1 draft for Dataset + Calibrated Audit sections. |
| C | Decoder-only interpretability benchmark | future | undecided | Seeded by Paper A; not started. |

### Paper A P0-2/P0-2b closure (2026-07-27)

- All 27 receipt-authorized P0-2 full runs completed. The original frozen panel gate fails; only ProGen2-medium TopK is atlas-eligible. The immutable eligibility receipt SHA-256 is `3f472afb0171836ad7f51d5c1d9e25b1d8d4a67aae1ac1aa51744bf18494f9b3`.
- The July 24 transfer-gap screen is now explicitly exploratory. Its single-seed results and implementation limitations cannot establish causes, refutations or a text-to-protein transfer mechanism.
- The prospective P0-2b amendment evaluated every completed checkpoint on a new 240-sequence cohort. Exact reinjection passed in all 27 runs, but no sparse model/method qualified: ProtGPT2 and ZymCTRL had invalid single-site CE denominators, while ProGen2-medium TopK recovered only 0.333--0.368 loss and 0.250--0.262 KL against 0.80 gates.
- Aggregate SHA-256 is `7f68e08775af87a171f2e4aac2a0d88cf235280280bf1bac44f76b0a2959bd07`; detailed interpretation and claim boundaries are in `docs/analysis/P0_2B_DICTIONARY_FIDELITY_RESULTS_20260727.md`.
- No further Paper A experiment is active. P2--P8, legacy P0-3--P0-8, steering and atlas expansion remain stopped. A future text/protein transfer study requires a new matched, denominator-valid protocol.

### Paper A evidence repair (2026-07-16)

- The original Swiss-Prot gate compared a top-100 event with 0.1 nats even though its maximum possible mutual information was 0.00661255 nats. It is invalid. EXP-R2-022 added a reproducible 2,000-permutation exploratory re-audit: 224/380 associations survived a global-position null and 0/380 a within-protein amino-acid/position-matched null after BH correction.
- Model hooks capture architecture-specific layer-normalized CLT inputs, not a raw residual stream. CLT training/evaluation also included padded positions; quality metrics are now described as unmasked diagnostics.
- The eight-class steering score is a motif/composition heuristic, and its selector introduced 19 negative-attribution interventions across six class/layer cells. It is a no-positive-evidence pilot requiring a corrected positive-only rerun, not a clean steering negative. The input-derived/output-injected direction experiment is site mismatched.
- Wider exploratory dictionaries produced mixed linear-probe recovery and none met the preregistered FVU criterion. They do not rule out capacity or optimization.
- The strongest current Paper A core is the procedure- and cohort-sensitive matching atlas, overlapping low-level associations, an N-terminal initiator-methionine subset with high unnormalized received attention, a candidate checkpoint diagnostic, and bounded negative interventions. A separate 500-sequence UniRef50 atlas recovered eight triplets at 0.90 with no exact feature-identity overlap with the canonical 38; the implemented 30-replicate null is pair/layer-specific, not a coherent model-wise permutation.

The July manuscript/package build should therefore be read as an evidence correction, not as completion of the independent assessment. P0-1 through P0-9 remain submission gates; human metadata and a DOI-backed release are additional external blockers.

### Paper A execution update (2026-07-22)

- Mask-aware training/evaluation and immutable split manifests pass local and H200 smokes. The original TopK seeds were discarded after a mass-resampling event made seed 43 non-finite; seeds 17/29 were stopped as incomparable. A capped deterministic resampler and exact atomic checkpoint/resume contract pass full-width stability/resume and checkpoint-publication tests. The replacement r6 seed 43 nevertheless failed at step 15,394. Deterministic replay isolated non-finite frozen ProtGPT2 fp16 activations in layers 24--35; CLT parameters and optimizer state remained finite, and bfloat16 inference made the same batch finite. Seeds 17/29 and the fp16 cache extraction were stopped as ineligible diagnostics. The uniform bfloat16 amendment, explicit cache-conversion path and bounded preflight now pass. The first bfloat16 exact-cache production attempt was stopped without a completion receipt when a read-only audit found incomplete imported-code binding; it remains explicitly ineligible. The r8 extractor verifies the complete first-party archive/tree inventory before local imports.
- By 2026-07-19 23:58 CST, all nine replacement online TopK runs (three models x seeds 17/29/43) had completed 200,000 steps with complete resumable final manifests and finite canonical logs. The separate r8/r3 exact-cache queue completed all three models by 02:48 CST, with schema-v3 completion receipts binding 641,433,600,000 cache-payload bytes, bfloat16 inference, finite captures, float16 storage and the exact code inventory.
- The first receipt-bound validation-only ReLU/L1 and gated-SAE screening lineage failed after 12 of 45 candidates because it retained prior candidate model/Adam state; its r1 root is permanently quarantined and cannot be resumed or pooled. Each active runner had completed candidates `000`--`002` and OOMed on candidate `003`. A candidate-local lifecycle correction and four-boundary full-width preflight passed with a stable 67,108,864-byte post-candidate allocation; four fresh r3 queues started on GPUs 0--3 at 14:22:06 CST on 2026-07-21 as launcher PIDs 27369--27372. On 2026-07-22 that r3 validation-only screen completed all 45/45 frozen candidates with zero test accesses. It selected ProGen2-medium gated SAE (candidate 0; L0 132.16376074074074; FVU 0.4397525937186076), ZymCTRL gated SAE (candidate 3; L0 120.02304111111111; FVU 0.5929956473037449) and ZymCTRL ReLU/L1 SAE (candidate 0; L0 131.2233588888889; FVU 0.5899043000429988). ProGen2-medium ReLU/L1 SAE and both ProtGPT2 sparse methods are terminal calibrated `sparsity_match_failure` negatives, so their nine seed runs are forbidden. The terminal receipt is `evidence/p0_2_screening_20260722/terminal_receipt.json` (SHA-256 `83bf4563a27886a1e1f148eb4094f9a7b02533936a02981caac17a805de04d6c`). The exact 27-run authorized full exact-cache panel was scheduled at about 11:46 CST across four fresh queues on GPUs 0--3 as launcher PIDs 30587--30590, in fixed 7/7/7/6 queues under `p0_2_dictionary_controls_bf16_r3/full`. Their initial launch receipt has SHA-256 `306d2b56bf3cf756fa5c56dd86e0f24d3555fc6c49c02d59747216e31d3104b9`. By 12:45 CST, each first run had a valid `in_progress` state and atomic step-5,000 checkpoint, with all four GPUs actively training and no queue-log error. The frozen budget is 1,500 aggregate GPU-hours; the conservative longest-queue ceiling is about 388.9 hours (16.2 days), excluding evaluation/I/O, while observed screening throughput (queue 3 is likely critical at about 4.8 days before margin) supports a provisional 5--7 calendar days. No H200 fault requires user action, and no held-out P0-2 quality gate has passed.
- Receipt-preserving cleanup removed 743,059,375,694 apparent bytes (692.03 GiB) on July 18, 395,430,148,930 bytes (368.27 GiB) on July 20 and 576,951,858,714 bytes (537.328 GiB) on July 21. The July 22 terminal selection cleanup additionally removed 45 redundant progress and 42 nonselected best checkpoints: 2,103,965,204,225 apparent bytes (approximately 1.9135 TiB), retaining three selected best checkpoints and 108 compact files. The July 21 cleanup had removed only 24 large checkpoints from the permanently ineligible OOM lineage after validation-hash guards and retained 28 compact failure-evidence files. The path-level receipts are `evidence/h200_cleanup_20260718/cleanup_manifest.json` and `evidence/h200_cleanup_20260720/cleanup_manifest.json` and `evidence/h200_cleanup_20260721/cleanup_manifest.json` and `evidence/h200_cleanup_20260722/cleanup_manifest.json` (latest SHA-256 `f646ad90abe8554c70357860285dda4051147fc0d6d6e95c0887f97fe62befe3`).
- The final P0-2 adjudication contract now recognizes an exact screening `sparsity_match_failure` only as a complete calibrated negative: it requires a complete candidate grid with zero test access and no selected configuration/checkpoint, omits and forbids that pair's three full runs, emits `atlas_eligible=false` and remains rejected downstream. A new minimal producer wrote the confirmatory mask-validation receipt at `evidence/p0_2_mask_validation_20260721/mask_validation_receipt.json` (SHA-256 `5966e274881984b2eeabeedd749d94c313fce02fb814911f0d1082ac3c3232db`), binding the exact screening module and two passing required mask tests. The lifecycle-only 216-test snapshot is preserved in the experiment log; the final integrated pre-documentation suite passed 223 tests plus 6 subtests. The terminal r3 screen now authorizes only its 27 receipt-bound runs; the separate mask receipt remains valid. Neither passes P0-2, which remains open until those runs are aggregated and adjudicated.
- P0-3/4/5/6/8 production consumers now share the same numerical estimand: bfloat16 parameters verified before first activation, finite required-layer activations/attentions/logits verified before conversion or use, and a versioned six-field integrity receipt. Legacy float16 production schemas fail closed; real pretrained execution remains pending.
- Independent-atlas, continuous-semantics, planted-control and nested-recoverability engines have strict manifests and passing synthetic plumbing tests. The trained-GRU control remains a post-hoc development smoke. Its prospective unexposed replacement passed all frozen synthetic sensitivity, specificity, FDR, path-localization, effect-recovery and control-equivalence gates across untouched seeds. This passes only the planted pipeline-sensitivity subgate; no pretrained-model causal or biological gate is thereby passed.
- The real P0-7 adjudication path is now implemented without duplicating model inference: one strict module and thin CLI consume only hash-bound P0-2, P0-5/P0-6, identity-freeze, raw intervention and validated external-score artifacts. Complete paired rows, one global positive/TOST multiplicity family, intervention fidelity, off-path localization and atomic final rehash are enforced. This is contract infrastructure only; no eligible real pretrained intervention surface or completion receipt exists.
- Frozen UniRef50 and ZymCTRL splits, deployed data hashes, code hashes, the restored May chronology and exact commands/results are recorded in `docs/EXPERIMENT_LOG.md` (EXP-R2-025).
- The exact historical balanced-200 cohort has been recovered from archived compute storage and its alternating order verified locally (file SHA-256 `5bc7697a83cc7461558f8b4597a3c9b4d6a151b7ec70ca22efc7282ecde4f0a6`). The exact three reference CLTs named by the saved atlas were also located and hash-verified. These recoveries remove the local row-order and reference-CLT-location blockers. The complete deployed ProtGPT2 tree is now verified to Hugging Face commit `f71aa6cf063ad784ebd53881d11332fd098eaa58`; ZymCTRL is best-supported at `3c532ef172b9cd2e95238baadf5167ebb89fbc32` with its weight verified but strict whole-tree proof incomplete; the deployed ProGen2-medium tree matches no single upstream commit and requires a deposited local snapshot. None of these local recoveries provides the required public release.
- P0-1 through P0-9 remain open as complete packages. New outputs enter the manuscript only after their prespecified acceptance criteria are evaluated.

See `docs/audits/REPOSITORY_AUDIT_20260716.md`, `docs/audits/REORGANIZATION_20260716.md`, `docs/audits/STRUCTURE_STANDARDIZATION_20260716.md`, and the dated Paper A result audit.

The sections below are the **rescue-phase status log (2026-05-13 → 2026-05-18)**, retained for provenance. Their bottom-line conclusions remain accurate for the scope they describe: R1's rescue claims and R2's causal/steering claims all failed their gates. That is why the steering, attention-sink and drug-design lines are retired rather than paused.

---

# Project Status — 2026-05-13 (historical)

Two independent research projects aimed at Nature/Science/Cell.

## Executive Summary

**R1 (Variant Mechanism Prediction via SAE):** The strongest defensible result is pathogenicity complementarity: annotation-selected SAE features + ESM-2 LLR reach **AUC 0.9143** on 2,000 ClinVar variants and **AUC 0.9193** on the 101-variant cancer holdout. The LOF/GOF/DN mechanism classifier does **not** generalize under protein-level CV (SAE macro-AUC **0.5161**), so the mechanism claim must be narrowed to variant-level structure and feature interpretation. ProteinGym is a negative diagnostic, and the indel transfer experiment is useful but below threshold (damage AUC **0.7735**, target >0.85). After external staging, AlphaMissense, gMVP, and ESM-1v are scored. PrimateAI-3D is treated as unavailable after gated access could not be obtained, so the T1-A baseline table is final over the accessible baselines. AlphaMissense and gMVP are stronger on raw pathogenicity (ClinVar2000 AUC **0.9474** and **0.9369**), so R1's reviewer-facing novelty cannot be framed as beating scalar pathogenicity specialists. The 2026-05-11 gene-grouped SAE x AlphaMissense ensemble gate was negative: AlphaMissense AUC 0.9474, AM+SAE stack AUC 0.8542, AM+SAE z-sum AUC 0.9210, with negative group-bootstrap deltas versus AlphaMissense. The remaining useful angle is SAE-specific interpretation and diagnostic scope, not scalar ensemble improvement over AlphaMissense.

**R2 (Interpretable Drug Design via CLT):** v2 CLTs, hook diagnostics, direct-effect feature selection, on-manifold steering, structural QC, ablation, and cross-model conservation have all been run. The current steering evidence is negative: direct-effect/on-manifold steering gives **0/8 significant positive** EC classes. T2-C generated-sequence checks now include Pfam, CLEAN, and Foldseek. The selected steered leads pass all three checks, but the generation-wide steered-vs-unsteered lift remains weak: Pfam 0.860 vs 0.820 (Fisher p=0.170), CLEAN exact 0.775 vs 0.775. The real-vs-random lysozyme calibration is complete and validates the metric stack itself: all reported Pfam/CLEAN/ESMFold/Foldseek metrics separate real lysozymes from random UniRef50 controls with Cohen's d > 1. Current decision: no-go for a strong steering/drug-design claim unless broader EC-class generation results change this conclusion.

---

## Research 1 — SAE Variant Mechanism Prediction

### Status: working pathogenicity ensemble; mechanism claim downgraded

### SAEs
- ESM-2-3B, layers {19, 23, 27, 31, 35}
- BatchTopK SAE, d_sae=16384, k=256
- Trained 500k steps on H200 (2026-04-01)

### Annotation Alignment
| Layer | KNOWN (F1>0.5) | PARTIAL (F1>0.2) | Useful (F1>0.1) |
|------:|---------------:|-----------------:|----------------:|
| 19    | 381            | 3483             | 7481            |
| 23    | 301            | 2992             | 6773            |
| 27    | 163            | 1725             | 4423            |
| 31    | 86             | 1135             | 3197            |
| 35    | 45             | 653              | 1935            |

These counts are after the firing-position rerun and expansion. L35 improved but still missed the TODO_NEXT target of >=60 KNOWN features.

### Variant Prediction AUCs (2000 ClinVar variants, 1000 path + 1000 benign)
| Method                                       | AUC    |
|----------------------------------------------|--------|
| Raw SAE perturbation (sum \|Δf\| across layers) | 0.542  |
| Functional disruption (annotated delta)      | 0.687  |
| 12-summary-stats cross-layer LR              | 0.836  |
| Full 163k-dim vectors LR (script 11)         | 0.747  |
| **Annotation-selected features LR (script 12)** | **0.878** |
| **ESM-2 masked-marginal LLR (baseline)**     | **0.882** |
| Combined 12-stat LR + LLR (EXP-010)          | 0.894  |
| **Annotation-LR + LLR (logistic, EXP-015)**  | **0.9137** |
| **Annotation-LR + LLR (simple z-sum, EXP-015)** | **0.9143** |

**Key lesson (addressing Codex critique):** Naive use of the full 163k-dim delta vectors *overfits* with only 2000 samples (AUC 0.747, worse than 12 summary stats). Annotation-guided feature selection (F1 ≥ 0.1) recovers AUC 0.878 using ~52k features, essentially matching LLR alone. Combined with LLR via logistic stacking, reaches AUC 0.9137; via simple z-sum, **AUC 0.9143** (EXP-015) — a +3.2 point improvement over LLR alone.

### What R1 provides beyond LLR
- Mechanism-aware delta signatures (per-category perturbation)
- Feature-level interpretation (which functional site is disrupted)
- Case studies (TP53, KRAS, PTEN, KCNQ1, HBB): each variant linked to specific feature disruptions

### Honest Weaknesses
1. SAE alone does **not** beat LLR on raw AUC (0.878 vs 0.882) — but ensemble does: 0.9143 vs 0.8822 alone (EXP-015)
2. Protein-level mechanism CV fails: SAE macro-AUC 0.5161
3. ProteinGym SAE+LLR does not beat LLR on average; signed ensemble win-rate is 33.6%, below the 40% threshold
4. T1-A competitor baselines are finalized over accessible methods: AlphaMissense, gMVP, and ESM-1v are available; PrimateAI-3D is excluded because the gated dataset could not be obtained
5. SAE x AlphaMissense ensemble gate failed under gene-grouped CV; do not claim scalar pathogenicity complementarity versus AlphaMissense
6. T1-E channelopathy scoring is complete, but the current classifier misses the >=80% concordance target: 64 LOF/GOF/DN rows, accuracy 0.625, macro-F1 0.444, with DN variants mostly collapsing to LOF

### Next Improvements
- Reframe the R1 manuscript around accessible baselines, indel diagnostics, and SAE interpretation; do not claim AlphaMissense ensemble improvement
- Do not launch F-T1-1 full-scale indel scoring until transcript-aware frameshift/mapping work is scoped, because the current scorer already covers the 6,649 binary length-compatible reconstructable rows
- Stage CB513, DeepLoc, and FireProtDB/ProTherm before running the R2 five-task downstream evaluation, or reduce R2 to an EC/Pfam pilot

---

## Research 2 — CLT Circuit Discovery for Drug Design

### Status: CLTs trained, architecture mapped, steering claim currently no-go

### CLTs Trained (H200, 100k steps)
| Model          | Layers | d_clt | k  | FVU  | Dead | Assessment |
|----------------|-------:|------:|---:|-----:|-----:|------------|
| ProtGPT2       | 36     | 4096  | 64 | 0.33 | 82%  | POOR       |
| ZymCTRL        | 36     | 4096  | 64 | 0.38 | 58%  | MODERATE   |
| ProGen2-medium | 27     | 4096  | 64 | 0.33 | 57%  | GOOD       |

### Layer Quality Map (usable: alive ≥ 20% AND FVU < 0.5)
| Model          | Usable / Total | Recognition | Generation | Steering |
|----------------|---------------:|------------:|-----------:|---------:|
| ProtGPT2       | 11 / 36        | L5-11       | L12-15     | **NONE** |
| ZymCTRL        | 23 / 36        | L0-11       | L12-16     | L25-30   |
| ProGen2-medium | 20 / 27        | L2-8        | L9-14      | L20-26   |

### Reconciled ZymCTRL Architecture (fixes previous L35 claim)
Previous memory: "L27-35 = enzyme-specific output, L33-35 has 14× more discriminating EC features." Actual: **L35 CLT is 90% dead** — the discrimination signal lives in raw activations the CLT cannot capture.

Reconciled usable layers for EC-conditioned circuit tracing:
- **Recognition:** L3 (effective_L2=19.06, alive=35%)
- **Universal generation:** L12 (CLT quality=0.341, alive=64%)
- **Enzyme-specific output:** L30 (effective_L2=13.34, alive=21%)

### What works
- Circuit-tracer infrastructure integrated (Anthropic stack)
- EC-conditioned feature extraction for 8 enzyme classes (ZymCTRL)
- Feature interpretation via max-activating sequences
- Hook sanity checks pass: interventions measurably change logits
- Direct-effect feature selection completed for 8 EC classes x 36 layers
- On-manifold steering implementation runs end-to-end

### Honest Weaknesses
1. **ProtGPT2 cannot support steering** — no usable deep layers
2. **High dead rates** mean steering interventions should be constrained to a small set of usable features
3. **Steering experiments are negative so far** — 0/8 significant positive EC classes under direct-effect/on-manifold steering
4. **Real EC metrics are executed for lysozyme** — Pfam, CLEAN, and Foldseek scanned the generated lysozyme sequences, and the real-vs-random calibration passed for the metric stack
5. **Drug design claim not supported** — current evidence supports a prototype pipeline, not validated therapeutic sequence design

### Next Improvements
- Extend the calibrated T2-C metric stack to additional EC classes before any renewed steering claim
- If R2 remains a steering project, train a stronger CLT only after T2-C confirms that the current metric stack is usable
- Otherwise reframe R2 as interpretability + layer-map work

---

## Cross-Project Infrastructure

- H200 server: 16× H200 available for retraining / scaling
- Local L20 server: 8× L20 for evaluation and light training
- Conda env: `~/miniconda3/envs/ct` (PyTorch 2.9.1, CUDA 12.8, circuit-tracer 0.1.0)

## Pending Before Paper Submission

### R1
- [x] Protein-level mechanism holdout
- [x] ProteinGym sign-corrected diagnostic
- [x] Firing-position annotation rerun and expansion
- [x] Indel transfer diagnostic
- [x] Finalize T1-A without PrimateAI-3D after gated access failed
- [x] Run F-T1-3 SAE x AlphaMissense ensemble gate: completed, negative
- [x] Stage curated channelopathy labels for T1-E
- [x] Score T1-E channelopathy cohort: completed, target not met

### R2
- [x] Hook sanity and ec_features provenance diagnostics
- [x] Direct-effect feature selection
- [x] TopK-aware on-manifold steering benchmark
- [x] R2 viability decision gate: current steering claim no-go
- [x] Finish T2-C lysozyme calibration run and pull summary outputs

## Recent Experiments (2026-05-07)

See `TODO_RESULTS.md`, `r1_encoder_interpretability_benchmark/docs/EXPERIMENT_LOG.md`, and `docs/EXPERIMENT_LOG.md` for the completed TODO_NEXT pass:
- T0-A/B/C/D diagnostics and downgrades
- T1-A readiness blocker, T1-B firing-aware audit, T1-C indel transfer, T1-D annotation rerun, T1-E label curation plus concordance failure analysis
- T2-A/B direct-effect/on-manifold steering, T2-C generated metric triad, T2-C calibration runner, T2-D no-go decision

## IndelMissense Coordinate Augmentation (2026-05-15)

- Added `r1_encoder_interpretability_benchmark/scripts/48_package_indelmissense_coordinates.py`.
- Generated `data/indelmissense/v1.1_coordinates/`, which keeps the v1 record set unchanged and adds ClinVar GRCh37/GRCh38 coordinates from `data/clinvar/variant_summary.txt.gz`.
- Coverage: 6,649/6,649 records have an exact ClinVar variant match; 6,642/6,649 have at least one genomic coordinate; 6,635/6,649 have VCF-style `chrom/pos/ref/alt` coordinates suitable for dbNSFP/tabix-style joins. Seven exact ClinVar matches still lack a mappable genomic locus in the staged ClinVar table.
- The previous dbNSFP blocker for v1 is now superseded for v1.1 coordinates: coordinate-based external baselines can be retried on the VCF-covered subset.
- See `docs/INDELMISSENSE_SIMILAR_DATASETS_AND_COORDINATES_CN.md`.

## Opus Pivot Status (2026-05-12)

- R1 gene-level mechanism gate is negative: `r1_encoder_interpretability_benchmark/results/variant_effect/gene_level_mechanism_20260512.{json,md}` reports macro-AUC 0.5665 under Pfam-family proxy holdout, so the gene-level mechanism headline should be dropped.
- R1 indel competitor staging is closed for this pass: `r1_encoder_interpretability_benchmark/results/variant_effect/indel_competitor_attempt_20260512.{json,md}` records that dbNSFP has CADD/REVEL columns, but IndelMissense v1 lacks genomic coordinates needed for a valid match.
- R2 universal triplet annotation has a weak first pass: `results/circuit_analysis/universal_primitives_uniref500_20260512/` annotates all 38 triplets over 500 UniRef50 sequences. Simple amino-acid labels show weak enrichment, but best MI is only 0.0001-0.0019 nats; do not claim named biological primitives yet.
- Low-risk resource annotation is now complete: `results/circuit_analysis/universal_primitives_resource_annotation_20260512/` shows that the broad UniRef500 top-firing set has almost no current Pfam / Swiss-Prot / AlphaFold coverage. The English replanning packet is `archive/conversation_history/OPUS_LOW_RISK_RESULTS_20260512.md`.
- Current H200 state: `jiaotongdamoxing-zhk-zip-final-1gpu-0511` still reserves 1 H200, but no active experiment process is running.

## Opus M-1 Characterization Status (2026-05-14)

- Implemented and ran the final R2 characterization experiment requested in `archive/conversation_history/OPUS_NEXT_20260514.md`: `scripts/35_triplet_characterization.py`.
- Output directory: `results/circuit_analysis/triplet_characterization_20260514/`.
- Cohort: 700 sequences, reusing the 500 Swiss-Prot N-1 cohort plus 200 balanced calibration / UniRef50 records.
- Result: PASS. 37 / 38 conserved triplets were categorized after BH-corrected q < 0.05 tests.
- Assigned categories: 21 k-mer, 14 positional, 2 high-norm, 1 unknown.
- The final run is `v2`: the BPE-boundary test was corrected to the intended one-sided positive boundary-enrichment statistic. This changed BPE q-values but did not change the final category counts.
- English decision packet for Opus: `archive/conversation_history/OPUS_M1_RESULTS_20260514.md`.
- Current H200 state after completion: the 1-GPU pod remains allocated but idle with no active experiment process and 0 MiB / 0% GPU use inside the pod.

## Opus 2026-05-15 Plan Execution

- Implemented M-2 synthesis: `scripts/36_triplet_synthesis.py`.
- Final M-2 output from the M-1 `n_perm=2000` tables: `results/circuit_analysis/triplet_synthesis_20260515_nperm2000/`.
- Key final M-2 result: 37 / 38 triplets have at least one significant low-level characterization test, 21 / 38 have three or more significant tests, and the attention-sink subset is T011/T018/T023/T025.
- Completed the optional final M-1 rerun on the H200 hold pod with `n_perm=2000` and `--top-position-rows 100`; output: `results/circuit_analysis/triplet_characterization_20260515_nperm2000/`.
- Final `n_perm=2000` M-1 result: PASS, 37 / 38 categorized under the legacy single-category summary; assigned counts are 17 positional, 17 k-mer, 3 high-norm and 1 unknown.
- R2 manuscript was updated to the final Opus framing with `n_perm=2000` synthesis numbers and compiles locally.
- Added `manuscripts/README.md` as a cross-manuscript evidence index for the main R1/R2 numeric claims and local result-file pointers.
- R1 manuscript still compiles locally.
- IndelMissense v1 now includes an explicit CC-BY-4.0 license note in `data/indelmissense/v1/`.
- The Opus H200 hold pod `jiaotongdamoxing-zhk-zip-opus-hold-1gpu-0513-master-0` was released after final status checks; no running BioCC GPU pod remains.
- At the user's later request, a new 1-GPU H200 hold pod was started: `jiaotongdamoxing-zhk-zip-hold-1gpu-0513b-master-0` on `i-d5cvmv6heob1nidq4ujg` / `192.168.20.204`. It is currently idle with 0 MiB / 0% GPU use and holds the last visible free H200.
- English execution packet: `archive/conversation_history/OPUS_PLAN_EXECUTION_20260515.md`.

## Opus 2026-05-16 Additions

- English execution packet: `archive/conversation_history/OPUS_PLAN_EXECUTION_20260516.md`.
- R1-Add-1 is complete: `r1_encoder_interpretability_benchmark/results/variant_effect/indel_protein_baselines_20260516/`. The H200 ESM region scoring table was pulled locally as `r1_encoder_interpretability_benchmark/results/variant_effect/indel_esm_region_scores_20260516.tsv`. Gate outcome is negative for a standalone SAE indel scorer: SAE damage AUC 0.7735, ESM region mean pseudo-NLL AUC 0.8037, cheap grouped-feature LR AUC 0.8447. A combined SAE+ESM+cheap grouped LR reaches AUC 0.9108.
- R1-Add-2 is complete: `r1_encoder_interpretability_benchmark/results/variant_effect/am_sae_disagreement_typing_20260516/`. Gate outcome is negative: no AM-vs-SAE residue-context enrichment survives BH q < 0.05.
- R1-Add-3 VUS retrospective reclassification remains blocked until historical ClinVar archive snapshots are staged.
- R2-Add-1 is complete: `results/circuit_analysis/attention_sink_subset_20260516/`. T011/T018/T023 are concrete N-terminal edge attention-sink triplets; T025 is attention-associated but not N-terminal.
- R2-Add-2 is complete in the available early-checkpoint form: `results/circuit_analysis/universal_atlas_quality_diagnostic_20260516/`. Mature v2 recovers 38 universal triplets and early10k recovers 16, supporting universal-triplet count as a CLT checkpoint-quality diagnostic.
- R2-Add-3 is complete: `results/circuit_analysis/attention_sink_biological_correlate_20260516/`. T011/T018/T023 pass the N-terminal biological-correlate gate with first-two residue fraction 1.000 vs background 0.055 and BH q 2.72e-117.
- Current recommended framing: R1 is an interpretable benchmark/resource plus a strong combined baseline, not a scalar SAE predictor paper. R2 should promote the N-terminal attention-sink subtype and the checkpoint-quality diagnostic, while avoiding broad biological-primitive claims.

## R2 Causal Ablation Update

- Added and ran the R2 attention-sink causal ablation: `scripts/40_attention_sink_causal_ablation.py`.
- Output: `results/circuit_analysis/attention_sink_causal_ablation_20260517/`.
- Result packet: `archive/conversation_history/OPUS_R2_CAUSAL_ABLATION_RESULTS_20260517.md`.
- Outcome: FAIL. The target features are active at the N-terminal edge, but the TopK-aware CLT MLP-output ablation does not produce consistent deleterious N-terminal likelihood or downstream attention-redistribution effects across ProtGPT2, ZymCTRL, and ProGen2-medium.
- R2 should therefore phrase T011/T018/T023 as conserved N-terminal attention-sink-associated sparse features, not as causal attention-sink circuits, unless a future direct attention-head intervention changes this.

## Opus 2026-05-18 Rescue Plan Execution

- English execution packet: `archive/conversation_history/OPUS_RESCUE_EXECUTION_20260518.md`.
- R2 E-1 direct attention-head sink ablation is complete: `results/circuit_analysis/attention_head_sink_ablation_20260518/`. Gate outcome is FAIL. Paired heads have high N-terminal attention mass (roughly 0.90-0.97), but target ablations produce near-zero feature drop and dNLL pos2-10 values near zero: ProGen2-medium about -0.0011, ProtGPT2 about +0.0005, and ZymCTRL about +0.0071. This does not support a Nat Methods-grade causal head mechanism.
- R1-Save-3 extended AM-vs-SAE disagreement typing is complete: `r1_encoder_interpretability_benchmark/results/variant_effect/extended_disagreement_typing_20260518/`. Gate outcome is FAIL. No original or added protein-level context survives BH q < 0.05 across 90 tests.
- R1-Save-2 bounded abundance proxy is complete: `r1_encoder_interpretability_benchmark/results/variant_effect/vampseq_abundance_proxy_20260518/`. Gate outcome is FAIL. On nine staged ProteinGym / VAMP-like abundance assays, SAE-family beats AlphaMissense on only 1/9 usable assays.
- R1-Save-1 low-MSA stratification is the only remaining unexecuted rescue experiment. It requires building or staging a per-protein MSA-depth table; current recommendation is to ask Opus before spending more compute because R1-Save-2 and R1-Save-3 have now failed.
- Current H200 state: the 1-GPU pod `jiaotongdamoxing-zhk-zip-hold-1gpu-0513b-master-0` remains allocated and idle after the R2 E-1 run.

## Final Rescue Status

- Final result packet: `archive/conversation_history/FINAL_RESCUE_RESULTS_20260518.md`.
- R1 low-homology rescue is complete: `r1_encoder_interpretability_benchmark/results/variant_effect/low_homology_stratification_20260518/`. Gate outcome is FAIL. In the lowest UniRef50 cluster-size quartile, AlphaMissense AUC is 0.9627 and SAE+LLR z-ensemble AUC is 0.8727; the SAE+LLR-minus-AM delta is -0.0900 [-0.1253, -0.0568].
- R2 distributed sink-head rescue is complete: `results/circuit_analysis/attention_sink_set_ablation_20260518/` and `results/circuit_analysis/attention_sink_set_ablation_top32_20260518/`. Both top-8 and top-32 sink-set ablations fail strict and exploratory gates. Feature drops remain near zero or negative, and random same-layer controls can be as large or larger than sink-set effects.
- Current interpretation: the original Nat-Methods-grade rescue claims are exhausted. R1 remains a resource/audit paper centered on IndelMissense v1 and the combined indel baseline. R2 remains a conserved sparse-feature readout/diagnostic paper, not a causal attention-sink mechanism paper.
