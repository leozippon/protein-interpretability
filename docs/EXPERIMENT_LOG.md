# Research 2 — CLT Experiment Log

> **Current status (2026-07-17):** this chronological log contains historical terminology and superseded hypotheses as provenance. Current claim boundaries and scientific-gate status are authoritative in `NPJ_AI_MAJOR_REVISION_EXECUTION_STATUS_20260717.md`, `NPJ_AI_MAJOR_REVISION_PLAN_20260716.md` and `PROJECT_STATUS.md`. No historical “universal”, mechanism or steering-success wording overrides those documents.

## EXP-R2-001: Baseline CLT Training on H200 (2026-04-02)

**Goal**: Train windowed CLTs on 3 protein generators to establish baselines and identify training issues before scaling to d_clt=16384.

**Config** (shared across all 3 models):
- d_clt: 4096, k: 64, window: 8
- batch_size: 2, total_steps: 100K
- lr: 3e-4, warmup: 2000, cosine anneal
- No dead feature resampling
- No activation normalization
- H200 single GPU, data from GPFS

### Results

| Model | Final FVU | Final Dead% | Final L0 | Time/step | Notes |
|-------|:---------:|:-----------:|:--------:|:---------:|-------|
| ProtGPT2 (36L, 1280d) | ~0.30 | **91.9%** | 64.0 | 0.15s | Catastrophic dead features |
| ZymCTRL (36L, 1280d) | ~0.35 | **43.0%** | ~62 | 0.20s | Moderate dead features |
| ProGen2-medium (27L, 1024d) | ~0.32 | **29.2%** | ~57 | 0.18s | Best of the three |

### Key observations

1. **ProtGPT2 dead feature crisis**: Dead fraction jumped from 0% to 28.4% at step 10050, then to 44.1% at step 10100, eventually saturating at 91.9%. This happens right after the dead-feature tracking threshold (10000 steps) kicks in, meaning features died very early in training and never recovered.

2. **Loss scale discrepancy**: ProtGPT2 raw loss values (~3-10) are orders of magnitude larger than ZymCTRL (~0.02) and ProGen2-medium (~0.01). This suggests ProtGPT2's MLP outputs have much larger norms (confirmed by pipeline test: layer 0 mlp_out norm = 826.5, layer 12 = 1704.0). Activation normalization would help.

3. **No resampling = no recovery**: Without dead feature resampling, once features die they stay dead permanently. The dead fraction monotonically increases for all 3 models.

4. **FVU plateaus**: All models plateau around FVU 0.30-0.35. With 91.9% dead features, ProtGPT2 is effectively using only ~330 of 4096 features per layer. The model lacks capacity to improve further.

### Root cause analysis

The dead feature problem has the same root cause as R1's EXP-001 through EXP-004: without resampling, features that fail to specialize early in training never get a second chance. The CLT's TopK sparsity means features must compete for activation — once a feature falls behind, it gets selected less, its gradients vanish, and it dies permanently.

**Differences from R1**: R1 also had a pre-TopK ReLU bug that blocked auxk gradients. R2's CLT uses `F.relu()` followed by `topk()`, which is the correct ordering (ReLU then TopK). The issue is purely the absence of resampling.

### Fixes implemented for next run

1. **Dead feature resampling** (`clt_trainer.py`): Added `resample_dead_features()` — every 5000 steps, reinitializes dead features from high-reconstruction-error directions. Resets encoder, decoder, bias, and optimizer state. Adapted from R1's proven pattern.

2. **Checkpoint resume** (`clt_trainer.py`, `01_train_clt.py`): Added `load_checkpoint()` and `--resume` flag for training continuation.

3. **Shuffled data ordering** (`clt_trainer.py`): Replaced sequential batch indexing with `torch.randperm` shuffled ordering, reshuffling at epoch boundaries.

4. **Config additions** (`clt_training.yaml`, `clt_training_h200.yaml`): Added `resample_every: 5000` and `dead_feature_threshold: 10000`.

5. **Circuit discovery fixes** (`circuit_discovery.py`):
   - `load_trained_clt`: Infer n_layers and d_model from state dict instead of hardcoding 36
   - `steer_generation`: Accumulate steering delta across all interventions before returning (was returning inside loop, applying only first intervention)

### Expected improvements for next run

- Dead feature fraction should drop dramatically with resampling every 5000 steps
- FVU should improve as resampled features learn useful directions
- Scale to d_clt=16384 once resampling is validated at d_clt=4096

---

## EXP-R2-002: Rerun Evaluation with Resampling (2026-04-05)

**Goal**: Evaluate the rerun checkpoints (April 3) trained with dead feature resampling, and compare with EXP-R2-001 baselines.

**Script**: `scripts/05_evaluate_checkpoints.py`

### Rerun vs Original Comparison

| Model | Original Dead% | Rerun Dead% | Original FVU | Rerun FVU | Alive Features |
|-------|:--------------:|:-----------:|:------------:|:---------:|:--------------:|
| ProtGPT2 | 91.9% | **81.9%** | ~0.30 | 0.328 | 743/4096 |
| ZymCTRL | 43.0% | **58.3%** | ~0.35 | 0.380 | 1706/4096 |
| ProGen2-medium | 29.2% | **57.4%** | ~0.32 | 0.333 | 1745/4096 |

### Per-Layer Analysis (ProtGPT2)

| Layer | FVU | Dead% | Alive | MLP Norm |
|-------|-----|-------|-------|----------|
| 0 | 0.002 | 86.8% | 670 | 457.1 |
| 6 | 0.124 | 62.1% | 1804 | 18.3 |
| 12 | 0.010 | 78.8% | 902 | 66.2 |
| 18 | 0.395 | 84.8% | 629 | 66.6 |
| 24 | 0.603 | 83.2% | 695 | 104.0 |
| 30 | 0.491 | 94.2% | 237 | 187.8 |
| 35 | 0.200 | 91.9% | 360 | 331.1 |

### Key Findings

1. **Resampling helped ProtGPT2 but not enough**: Dead% dropped from 91.9% to 81.9%, but still catastrophic. Only 743 alive features per layer on average. The core issue is ProtGPT2's enormous MLP output norms (457 at layer 0, 331 at layer 35) creating extreme loss scale that drowns out newly resampled features.

2. **ZymCTRL and ProGen2-medium dead% increased in evaluation**: The eval metric (fired on <0.1% of tokens) is stricter than training (not fired in 10K steps). Many features that "fire" during training barely activate on held-out data.

3. **FVU varies wildly by layer**: First/last layers are easy (FVU<0.01) because MLP outputs are small. Middle layers (18-24) are hardest (FVU>0.4). This matches the intuition that mid-depth computations are most complex.

4. **d_clt=4096 is too small**: With 50-80% dead features, effective dictionary is 800-1700 features per layer. Need d_clt=16384+ for meaningful coverage of the feature space.

### Next Steps

- **Activation normalization**: Critical for ProtGPT2 (MLP norms 100x larger)
- **Scale to d_clt=16384**: H200 training with larger dictionaries
- **Longer training**: 100K steps may be insufficient; try 500K as in R1

---

## EXP-R2-003: ZymCTRL Circuit Analysis (2026-04-05)

**Goal**: Interpret CLT features, compare circuits across EC numbers, validate steering.

**Script**: `scripts/06_circuit_analysis.py --model zymctrl`

### Feature Interpretation

- **Total alive features**: 65,575 across all 36 layers
- Features show clear amino acid preferences:
  - Early layers (L9): Cysteine (C), hydrophobic (L, M, W) preferences → chemical property encoding
  - Middle layers (L18): More uniform preferences (D, P, E, W) → sequence pattern encoding
  - Deep layers (L27): Charged/polar preferences (K, D, N, H) → functional site encoding

### EC Number Circuit Comparison (8 enzyme classes)

**Feature sharing across EC numbers:**
| Category | Feature Count |
|----------|:------------:|
| Specific to 1 EC | 6,214 |
| Shared by 2 ECs | 2,422 |
| Shared by 3 ECs | 1,339 |
| Shared by 4-7 ECs | 2,294 |
| Universal (all 8 ECs) | 1,365 |

**Key insight**: 1,365 features fire for ALL 8 enzyme classes — these are universal protein generation features (start signals, positional encoding, general AA preferences). 6,214 features are specific to individual EC numbers — these encode enzyme-class-specific sequence patterns.

**Top EC-specific features (by layer):**
- Lysozyme (3.2.1.17): Layer 17, Feature 2528 (activation=0.29)
- Trypsin (3.4.21.4): Layer 1, Feature 1470 (activation=0.39)
- DNA polymerase (2.7.7.7): Layer 35, Features 1497/3611/2802 (activation=1.65/1.37/1.35)
- Carbonic anhydrase (4.2.1.1): Layer 34, Feature 621 (activation=0.89)

DNA polymerase has the strongest EC-specific features, consistent with its highly specialized sequence requirements (polymerase active site motifs, processivity domains).

**Top pairwise comparison (lysozyme vs trypsin):**
- 6,073 differential features
- Strongest: Layer 32 Feature 2871 (diff=1.96), Layer 3 Feature 262 (diff=1.51)
- These features encode the distinct catalytic mechanisms (glycosidic bond hydrolysis vs peptide bond cleavage)

### Steering Experiment

- Baseline generation from EC 3.2.1.17 (lysozyme) produced valid protein sequence
- Differential feature identification between lysozyme and ADH succeeded
- Steering hooks registered but insufficient differential features passed the threshold for meaningful generation change
- **Conclusion**: Steering requires higher-quality CLTs (more alive features) and longer conditioning prompts to establish stable feature patterns before intervention

---

## EXP-R2-004: Layer-by-Layer EC Specificity Analysis (2026-04-05)

**Goal**: Quantify how enzyme class (EC number) information flows through ZymCTRL's transformer layers. Which layers encode universal protein generation rules vs. enzyme-specific sequence patterns?

**Script**: `scripts/07_layer_ec_specificity.py`

### Per-Layer Feature Statistics

| Layer Group | Mean Alive | Mean Specific | Mean Universal | Mean Specificity (CV) |
|-------------|:----------:|:-------------:|:--------------:|:--------------------:|
| Early (0-8) | 469 | 222 | 46 | 1.891 |
| Middle (9-17) | 260 | 116 | 27 | 1.669 |
| Late-mid (18-26) | 307 | 133 | 33 | 1.666 |
| Deep (27-35) | 478 | 220 | 46 | 1.855 |

### Discrimination Power (EC class separation by L2 distance)

| Layer | Total L2 | Interpretation |
|-------|:--------:|----------------|
| 35 | **564.51** | Dominant: 4× more discriminating than any other layer |
| 34 | 221.95 | Second strongest EC discrimination |
| 33 | 132.84 | Significant EC-specific computation |
| 32 | 118.56 | Emerging specialization |
| 30 | 96.50 | |
| 3 | 80.31 | Early-layer EC recognition (second peak) |
| 0-8 avg | ~45 | Initial EC prompt encoding |
| 9-17 avg | ~30 | Least discriminating (general protein rules) |

### EC Pairwise Cosine Similarity

| Layer Group | Mean Similarity | Min | Max |
|-------------|:--------------:|:---:|:---:|
| Early (0-8) | 0.721 | 0.589 | 0.873 |
| Middle (9-17) | 0.676 | 0.521 | 0.916 |
| Late-mid (18-26) | 0.561 | 0.432 | 0.811 |
| Deep (27-35) | 0.624 | 0.453 | 0.809 |

### Key Findings

1. **Layer 35 dominates EC discrimination** — with total L2=564.51, it is 2.5× more discriminating than Layer 34 and 14× more than the average layer. This is where ZymCTRL concentrates enzyme-class-specific sequence generation logic.

2. **Bimodal specificity pattern** — both early (0-8) and deep (27-35) layers have high specificity, while middle layers (9-17) are least specific. This suggests:
   - Early layers: recognize the EC number tokens and begin routing
   - Middle layers: apply universal protein generation rules (hydrophobicity patterns, secondary structure propensities, amino acid transition probabilities)
   - Deep layers: translate EC class identity into specific sequence patterns (catalytic motifs, substrate binding sites, fold-specific constraints)

3. **Cosine similarity is lowest in late-mid layers (0.561)**, not the deepest — this means feature DIRECTIONS diverge most in layers 18-26, while feature MAGNITUDES diverge most in layers 33-35. The model first builds different representations (divergent directions), then amplifies the differences (divergent magnitudes) for final output.

4. **U-shaped alive feature count** — early and deep layers have ~470 alive features, middle layers only ~260. This correlates with CLT dead features being concentrated in middle layers, consistent with R1's observation that middle-depth computations are hardest to decompose.

### Biological Interpretation

ZymCTRL's computation follows a clear three-phase architecture:
1. **EC Recognition (L0-8)**: The model reads the EC number tokens and activates enzyme-class-specific routing features. The second-highest discrimination peak at Layer 3 (L2=80.31) confirms early EC encoding.
2. **Universal Generation (L9-17)**: Middle layers apply shared protein generation rules — amino acid transition probabilities, secondary structure preferences, global sequence properties. These are the features shared across all 8 EC classes.
3. **Enzyme-Specific Output (L27-35)**: Deep layers translate enzyme class identity into specific sequence patterns. Layer 35 alone accounts for more EC discrimination than all other layers combined. This is where the model decides which catalytic residues to place, which fold topology to follow, and which substrate-binding motifs to include.

This three-phase architecture parallels the biological process of protein evolution: conserved protein backbones (shared across enzyme classes) are decorated with class-specific functional elements (active sites, cofactor binding).

### Implications for Drug Design

- **Steering should target layers 33-35**: These layers contain the EC-specific generation logic. Interventions at middle layers would disrupt universal protein rules without redirecting enzyme class.
- **Circuit tracing should focus on deep layers**: The path from EC input → L35 features → output tokens is the critical circuit for therapeutic protein design.
- **Universal features (L9-17) are "do not touch" constraints**: Perturbing these would produce invalid proteins, not redirected enzymes.

---

## EXP-R2-005: Layer Quality Map (2026-04-13)

**Question:** Codex critique flagged "high dead-feature rates" as making circuit findings preliminary. Which layers of each CLT are actually usable?

**Method:**
- Read `rerun_evaluation.json` (per-layer alive%, FVU)
- Define "usable" = alive ≥ 20% AND FVU < 0.5
- Compute quality = alive × (1 - min(FVU, 1))
- Group usable layers into contiguous regions
- Reference: `scripts/08_layer_quality_map.py`

**Results:**
| Model          | Usable | Regions            | Top quality layers |
|----------------|-------:|--------------------|---------------------|
| ProtGPT2       | 11/36  | L5-15              | L8, L7, L9, L6, L15 |
| ZymCTRL        | 23/36  | L0-16, L25-30      | L8, L7, L6, L10, L9 |
| ProGen2-medium | 20/27  | L2-14, L20-26      | L4, L5, L6, L3, L2  |

**Steering feasibility** (deep usable ≥ 1 layer):
- ProtGPT2: **CANNOT steer** — no deep usable layers
- ZymCTRL: usable deep layers L25-30
- ProGen2-medium: usable deep layers L20-26

**Honest assessment:**
- **POOR**: ProtGPT2 (CLT needs retraining with adjusted hyperparameters)
- **MODERATE**: ZymCTRL (usable but deep layers only 20-30% alive)
- **GOOD**: ProGen2-medium (contiguous usable regions, best quality)

**Output files:** `results/checkpoint_evaluation/layer_quality_map.json`

---

## EXP-R2-006: Architecture Reconciliation (2026-04-13)

**Question:** EXP-R2-004 concluded L35 dominates EC discrimination (L2=564 vs avg 40). But EXP-R2-005 shows L35 has only 9.7% alive features. Can we trust the L35 claim?

**Method:**
- Compute effective_discrimination = raw_L2 × CLT_quality (alive × (1-FVU))
- Rank only USABLE layers by effective discrimination
- Reclassify three-phase architecture based on usable layers
- Reference: `scripts/09_reconcile_architecture.py`

**Reconciled ZymCTRL three-phase architecture:**
| Phase | Old (raw) | Reconciled (usable only) | Reason |
|-------|-----------|--------------------------|--------|
| Recognition | L3 | **L3** (effL2=19.06, alive=35%) | Holds |
| Universal generation | L9-17 | **L12** (quality=0.341, alive=64%) | Sharpened |
| Enzyme-specific output | L33-35 | **L30** (effL2=13.34, alive=21%) | REVISED |

**Key correction:** L35 raw L2=564 is dominated by a tiny subset of features (<10% alive). The "L35 alone accounts for more discrimination than all other layers combined" claim from EXP-R2-004 reflects the raw model, NOT the CLT representation usable for steering.

**Revised steering recommendation:**
- Steering interventions should target **L30** (deepest usable layer with good discrimination)
- **NOT** L33-35 (dead CLT, unreliable)
- Generation control at **L12** (mid-depth, highest CLT quality)
- EC-conditioned recognition at **L3** (early, moderate alive rate)

**Impact on R2 paper:** The drug-design story must be framed around the usable layers. ProtGPT2 is excluded from the steering section. ZymCTRL circuit traces go L3 → L12 → L30. ProGen2-medium (best CLT) becomes the primary experimental target.

**Output files:** `results/circuit_analysis/zymctrl/architecture_reconciliation.json`

---

## EXP-R2-007: Hook Sanity + EC Feature Provenance (2026-04-29)

**Question:** Are the negative steering / causal-ablation results caused by broken plumbing, stale EC features, or a real lack of controllable signal?

**Method:**
- Added `scripts/diagnostics/00_hook_sanity.py`.
- Added `scripts/diagnostics/01_pkl_provenance.py`.
- The hook sanity script attaches the same MLP-output hook style used by the steering code, chooses an active CLT feature on a reference lysozyme sequence, and checks teacher-forced logit changes under multiplier 1, 10, and 0.
- The provenance script hashes and shape-checks the H200 `ec_features.pkl` against the ZymCTRL v2 CLT checkpoint.

**Results:**
| Diagnostic | Result |
|------------|--------|
| ZymCTRL hook fires | pass |
| ZymCTRL multiplier=1 identical to no hook | pass |
| ZymCTRL multiplier=10 max logit shift | 12.93 |
| ZymCTRL multiplier=0 max logit shift | 0.885 |
| ProGen2-medium multiplier=10 max logit shift | 3.938 |
| ProGen2-medium multiplier=0 max logit shift | 1.250 |
| H200 ec_features dimensionality | 36 layers, d_clt=8192 |
| ZymCTRL v2 checkpoint dimensionality | 36 layers, d_clt=8192 |

**Interpretation:** The MLP hook path can move logits in both ZymCTRL v2 and ProGen2-medium. The earlier exact-zero causal-ablation result is therefore not explained by a universally disconnected hook; the more likely issue is feature selection or the specific ablation evaluation path. The H200 `ec_features.pkl` is a v2-compatible 8192-dimensional artifact, and the local copy has been refreshed from H200.

**Output files:**
- `results/diagnostics/hook_sanity_20260429.json`
- `results/diagnostics/ec_features_provenance_20260429.json`

---

## EXP-R2-008: Direct-Effect Feature Selection (2026-05-03 CST)

**Question:** Can EC steering features be selected by direct effect on EC-conditioned likelihood rather than by mean-activation z-score?

**Method:**
- Added `scripts/16_direct_effect_features.py`.
- For each EC class, ran a teacher-forced backward pass through ZymCTRL v2.
- Ranked CLT features by `feature_activation * grad(log-likelihood) * CLT_decoder_vector`, summed across the CLT decoder window.
- Used the ZymCTRL v2 CLT checkpoint at `/oss-pvc/zhk_zip/outputs/research2/clt_weights/zymctrl_v2/step_200000`.

**Result:** Completed for 8 EC classes and all 36 layers. The main `top_indices` array has shape `(8, 36, 10)`, satisfying the T2-A acceptance shape.

**Interpretation:** This provides a stricter candidate feature set for steering: features must be active and point in a direction that changes model likelihood. It does not by itself prove steering works; it feeds T2-B.

**Output files:**
- `/oss-pvc/zhk_zip/biocc/paper_r2_nature_mi/results/circuit_analysis/zymctrl/direct_effect_features_v2.pkl`
- `/oss-pvc/zhk_zip/biocc/paper_r2_nature_mi/results/circuit_analysis/zymctrl/direct_effect_features_v2_summary_20260503.json`
- `/oss-pvc/zhk_zip/biocc/paper_r2_nature_mi/logs/runtime/t2a_direct_effect_features_20260503.log`

---

## EXP-R2-009: On-Manifold Direct-Effect Steering (2026-05-03 CST)

**Question:** Does TopK-aware CLT steering improve over the old off-manifold single-decoder-vector intervention?

**Method:**
- Updated `src/analysis/circuit_discovery.py` so steering hooks: compute CLT pre-activations, apply feature multipliers, re-apply TopK, and replace the CLT-explained same-layer MLP component.
- Updated `scripts/11_steering_benchmark.py` to accept `--direct-effect-features`.
- Passed a small lysozyme smoke test with direct-effect features.
- Started an 8-class benchmark with n=100 per condition, layers L3/L12/L30, top-5 direct-effect features per layer, multiplier=2.5.

**Results:**
| EC class | Unsteered | Steered | Delta | 95% CI | permutation p |
|----------|----------:|--------:|------:|--------|--------------:|
| lysozyme | 0.890 | 0.894 | +0.004 | [-0.040, +0.047] | 0.8678 |
| trypsin | 0.723 | 0.737 | +0.013 | [-0.047, +0.073] | 0.6743 |
| ADH | 0.650 | 0.669 | +0.019 | [-0.060, +0.098] | 0.6373 |
| catalase | 0.981 | 0.970 | -0.011 | [-0.023, +0.001] | 0.0904 |
| DNA_polymerase | 0.967 | 0.959 | -0.008 | [-0.033, +0.019] | 0.5693 |
| lipase | 0.769 | 0.826 | +0.057 | [-0.007, +0.122] | 0.0913 |
| kinase | 0.510 | 0.617 | +0.107 | [+0.000, +0.210] | 0.0532 |
| carbonic_anh | 0.900 | 0.855 | -0.045 | [-0.106, +0.014] | 0.1439 |

Significant positive steering: 0/8 classes.

**Interpretation:** TopK-aware direct-effect steering is a measurable intervention and produces modest positive shifts for kinase and lipase, but it does not clear the significance threshold on the current heuristic motif purity metric. This does not rescue the R2 steering claim yet; T2-C's real metric triad is still required before the R2 go/no-go decision.

**Output paths:**
- `/oss-pvc/zhk_zip/biocc/paper_r2_nature_mi/results/steering_benchmark/zymctrl_v2_onmanifold_direct_smoke_20260503.json`
- `/oss-pvc/zhk_zip/biocc/paper_r2_nature_mi/results/steering_benchmark/zymctrl_v2_onmanifold_direct_20260503.json`
- `/oss-pvc/zhk_zip/biocc/paper_r2_nature_mi/logs/runtime/t2b_onmanifold_direct_steering_20260503.log`

---

## EXP-R2-010: T2-C Metric-Triad Readiness Check (2026-05-04 CST)

**Question:** Can the real EC metric triad be run now for T2-C?

**Method:** Checked the current 1-GPU H200 pod and local/remote project paths for required executables and metric assets.

**Result:**
- Missing executables in the pod: `hmmscan`, `hmmsearch`, `diamond`, `foldseek`, and `esm-fold`.
- Present assets: `data/interpro/pfam_residue.tsv` and ESMFold weights at `/oss-pvc/zhk_zip/models/esmfold_v1`.
- No staged CLEAN database, Pfam HMM database, Foldseek binary, or Foldseek target structure database was found in the checked paths.

**Interpretation:** T2-C remains blocked. Per the TODO_NEXT guideline, no real EC metric should be claimed until the tools/databases are staged and a calibration run reproduces expected behavior on known positives and negatives.

---

## EXP-R2-011: T2-D Viability Decision Gate (2026-05-04 CST)

**Question:** Given T2-A/B and the T2-C readiness check, should R2 continue as a steering/drug-design claim or pivot?

**Evidence:**
- T0-A hook plumbing passed: interventions can move logits.
- T2-A direct-effect feature selection completed with shape `(8, 36, 10)`.
- T2-B TopK-aware on-manifold steering produced 0/8 significant positive EC classes on the current heuristic purity benchmark.
- The strongest trends were kinase +0.107, p=0.0532, and lipase +0.057, p=0.0913; neither passed significance.
- Structural QC did not show a foldability advantage for steered lysozyme leads over unsteered controls.
- T2-C real metric triad is blocked because CLEAN/HMMER/Foldseek tools and calibrated databases are not staged.

**Decision:** No-go for a strong R2 steering/drug-design claim in the current round. Do not proceed to Tier 3 wet-lab or substrate-swap claims from the current evidence.

**Recommended framing:** R2 can still be framed as an interpretability and layer-map pipeline with a transparent negative steering result. A steering claim should be reopened only after (1) T2-C tools/databases are staged and calibrated, and (2) the metric triad shows significant positive shifts in at least 3 of 8 EC classes, or after training a stronger CLT if the current FVU/dead-feature ceiling is judged binding.

---

## EXP-R2-012: T2-C External Resource Staging Update (2026-05-04 CST)

**Question:** Can the real EC metric triad blockers be reduced by staging public tools/databases?

**Result:** HMMER 3.4, DIAMOND 2.1.24, Foldseek AVX2, and Pfam-A HMM are now staged under `/oss-pvc/zhk_zip/biocc/external_resources`. Pfam-A was decompressed and `hmmpress` indexed successfully with 27,481 HMMs. CLEAN source is also staged.

**Remaining blocker:** T2-C is still not ready for reviewer-facing claims because CLEAN pretrained weights and ESM-1b weights are missing, no Foldseek target structure database has been selected/built, and no positive/negative calibration run has been performed. The old missing-tool blocker is resolved for HMMER/DIAMOND/Foldseek/Pfam.

**Manifest:** `external_resources/manifests/external_resource_status_20260504.md`.

## EXP-R2-013: Pfam/HMMER Generated-Sequence Scan (2026-05-04 CST)

**Question:** After staging HMMER/Pfam, do generated lysozyme sequences receive real lysozyme-like Pfam hits?

**Method:** Added `scripts/17_pfam_scan_generated.py`, ran `hmmscan` against staged Pfam-A on 10 selected steered leads, 200 steered generated sequences, and 200 unsteered prompt-baseline sequences.

**Result:** Steered leads had 9/10 lysozyme-like Pfam hits. Steered all had 172/200 lysozyme-like hits (0.860), while unsteered had 164/200 (0.820). One-sided Fisher p=0.1699 for lysozyme-like hit-rate improvement.

**Interpretation:** The EC-conditioned ZymCTRL prompt already generates many lysozyme-domain-like sequences. Pfam validates biological domain plausibility, but the measured steering lift is small and not statistically significant. CLEAN and Foldseek remain required before reopening a strong steering claim.

## EXP-R2-014: CLEAN and Foldseek Generated-Sequence Metric Triad (2026-05-07 CST)

**Question:** After staging CLEAN pretrained weights, ESM-1b, Foldseek, and a bounded PDB100 target DB, do generated lysozyme sequences pass real external EC and structure checks?

**Method:** Ran `scripts/19_clean_generated.py` for CLEAN EC top-1 prediction and `scripts/18_foldseek_generated.py` for Foldseek/PDB100 structure matching. Added `scripts/20_generated_metric_triad_summary.py` to combine the existing Pfam scan with CLEAN and Foldseek.

**Results:**
- Steered leads: Pfam lysozyme-like hit rate 0.900, CLEAN exact EC 3.2.1.17 rate 0.900, Foldseek mean top TM 0.883, TM >= 0.7 fraction 0.900.
- Steered all vs unsteered: Pfam lysozyme-like 0.860 vs 0.820 (one-sided Fisher p=0.170), CLEAN exact EC 3.2.1.17 0.775 vs 0.775, CLEAN 3.2.1.x prefix 0.865 vs 0.875.
- Foldseek was run on the existing ESMFold PDBs for steered leads and unsteered baseline structures.

**Interpretation:** The selected lead filter is biologically plausible under all three metrics, but the generation-wide steering claim is still weak. CLEAN does not show a steered_all lift over the unsteered baseline, and Pfam's lift is not statistically decisive.

**Output paths:**
- `results/ec_metrics/clean_generated_lysozyme_20260507.json`
- `results/ec_metrics/foldseek_generated_lysozyme_20260507.json`
- `results/ec_metrics/generated_metric_triad_summary_20260507.json`
- `results/ec_metrics/generated_metric_triad_summary_20260507.md`

## EXP-R2-015: T2-C Real-vs-Random Calibration Runner (2026-05-07 CST)

**Question:** Do the external EC/structure metrics distinguish known EC 3.2.1.17 lysozymes from length-matched random UniRef50 proteins before we use them as evidence on generated sequences?

**Method:** Added `scripts/21_prepare_ec_calibration.py` to prepare 100 real SwissProt/ZymCTRL lysozyme sequences and 100 random UniRef50 sequences with length 80-250 aa. Added `scripts/22_foldseek_calibration.py` and `scripts/23_ec_metric_calibration_summary.py`, then launched `scripts/run_t2c_calibration_20260507.sh` on the 1-GPU H200 pod.

**Result:** Completed on H200. The initial ESMFold structure pass failed in fp16 with a transformers `compute_tm` numerical error; rerunning the structure stage in fp32 fixed the issue and produced 200/200 PDB files.

Real-vs-random separation:

| Metric | Real mean | Random mean | Effect size d |
|---|---:|---:|---:|
| Pfam lysozyme-like hit | 0.910 | 0.000 | 4.497 |
| CLEAN exact 3.2.1.17 | 0.960 | 0.000 | 6.928 |
| CLEAN 3.2.1.x prefix | 0.990 | 0.050 | 5.549 |
| ESMFold mean pLDDT | 79.740 | 58.478 | 1.786 |
| ESMFold confident fraction | 0.865 | 0.336 | 2.063 |
| Foldseek top TM | 0.971 | 0.653 | 1.593 |

**Interpretation:** The metric stack itself is now calibrated for the lysozyme control: all metrics exceed the TODO_NEXT separation threshold. This supports using Pfam/CLEAN/ESMFold/Foldseek as filters and controls, but it does not rescue the generated steered-vs-unsteered result, which remains weak on CLEAN and non-significant on Pfam.

**Output paths:**
- `results/ec_metrics/calibration_lysozyme_20260507/`
- `results/ec_metrics/pfam_calibration_lysozyme_20260507.json`
- `results/ec_metrics/clean_calibration_lysozyme_20260507.json`
- `results/ec_metrics/calibration_real_lysozyme_esmfold_20260507.json`
- `results/ec_metrics/calibration_random_uniref50_esmfold_20260507.json`
- `results/ec_metrics/foldseek_calibration_lysozyme_20260507.json`
- `results/ec_metrics/ec_metric_calibration_summary_20260507.json`
- Remote log: `/gpfs/jiaotongdamoxing/zhk_zip/biocc/paper_r2_nature_mi/logs/runtime/t2c_calibration_20260507.log`

## EXP-R2-016: Universal Primitive Cheap-Label Annotation (2026-05-12 CST)

**Question:** Do the 38 balanced-200 cross-model universal triplets map to simple sequence-level biological labels strongly enough to justify the "named universal primitives" pivot?

**Method:** Added `scripts/29_universal_primitive_annotation.py`. The run uses the existing 38 triplets, extracts top firing positions across ProtGPT2, ZymCTRL, and ProGen2-medium CLTs, and computes enrichment plus mutual information for amino-acid identity, coarse residue chemistry, sequence source, and sequence-position bins. The main run used 500 UniRef50 sequences and the top 100 firing positions per triplet.

**Result:** All 38 triplets show some weak enrichment under simple labels (acidic/basic/polar/gly-pro/hydrophobic), but best MI is very small: min 0.000116, median 0.000688, max 0.001924 nats.

**Interpretation:** This does not meet the Opus 0.1-nat interpretability gate. The triplets remain statistically interesting, but the next experiment must add Pfam/structure/per-residue annotations before any causal intervention or named biological-primitives claim.

**Output paths:**
- `results/circuit_analysis/universal_primitives_uniref500_20260512/summary.json`
- `results/circuit_analysis/universal_primitives_uniref500_20260512/interpretation.md`
- `results/circuit_analysis/universal_primitives_uniref500_20260512/mutual_information.tsv`
- `results/circuit_analysis/universal_primitives_uniref500_20260512/simple_enrichment.tsv`
- Remote log: `logs/runtime/r2_universal_primitives_uniref500_20260512.log`

## EXP-R2-017: Universal Primitive Resource Coverage Audit (2026-05-12 CST)

**Question:** Are the top-firing positions from the universal-triplet pilot covered by staged Pfam, Swiss-Prot per-residue, or AlphaFold resources?

**Method:** Added `scripts/32_universal_resource_annotation.py`. The script maps top-firing `seq_id` values to UniProt-like accessions, checks Pfam residue interval overlap, Swiss-Prot feature overlap, and local AlphaFold structure availability.

**Result:** For the broad UniRef500 top-firing set, coverage is effectively absent: 3,800 events over 452 unique accessions have 0 Pfam-covered accessions, 0 Swiss-Prot-covered accessions, and 1 AlphaFold-covered accession. For the balanced-200 top-10-triplet pilot, 500 events over 33 unique accessions have 7 Pfam-covered and 7 Swiss-Prot-covered accessions, but most Swiss-Prot overlaps are chain/topology rather than strong functional/domain labels.

**Interpretation:** The current UniRef500 top-firing set is not resource-ready for biological naming. Before causal intervention, R2 needs either a resource-annotated cohort or additional ID mapping / resource staging.

**Output paths:**
- `results/circuit_analysis/universal_primitives_resource_annotation_20260512/`
- `results/circuit_analysis/universal_primitives_balanced200_resource_annotation_20260512/`

## EXP-R2-018: Recoverability Audit Full H200 Run (2026-06-05)

**Question:** Do failures of sparse-feature circuit tracing reflect a CLT dictionary bottleneck, or are the underlying protein generators themselves weak substrates for the target functional readouts?

**Method:** Ran the gated recoverability pipeline on the beliefnav H200 pod: `44_cache_representations.py`, `45_probe_ceiling_floor.py`, `46_oracle_direction_steering.py`, and `47_decision_table.py`. The run extracted R_raw, R_code, R_recon, ESM-2, and composition baselines for ProtGPT2 v2, ZymCTRL v2, and ProGen2-medium. The probe tasks were EC top-class, Pfam family, secondary-structure fraction, residue-level secondary structure, and decoder-native EC classification.

**Result:** The cache covered 820 Swiss-Prot proteins, 44,626 labelled residues, and 48 decoder-native EC sequences. ProtGPT2 and ProGen2-medium were classified as RICH substrates but did not satisfy the frozen retrain GO conditions; ZymCTRL was mixed. Oracle EC steering for ZymCTRL passed 0/8 class gates.

**Interpretation:** Under the pre-registered gate, this run does not justify the high-cost capacity retrain. Any further retraining should be labelled as an exploratory override rather than a protocol-confirmed next step.

**Output paths:**
- Remote results: `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/results/representation_audit_20260605_1250_recoverability_full/`
- Remote log: `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/logs/runtime/recoverability_full_20260605_1250.log`
- Local lightweight copy: `evidence/recoverability_audit_20260605_1250/`

## EXP-R2-019: Recoverability Audit v2 Corrected Re-analysis (2026-06-06)

**Question:** Does the original NO-GO remain after fixing analysis-design issues in the probe pass: high-dimensional random-projection control, EC/Pfam family confounding, and unstable secondary-fraction regression?

**Method:** Reused the EXP-R2-018 representation cache and reran `45_probe_ceiling_floor.py` plus `47_decision_table.py`. The v2 pass used fold-internal PCA (`--pca-dim 256`), added report-only `ec_topclass_stratified`, removed `R_rand` from the richness gate, and regressed helix+strand for `secondary_fraction`.

**Result:** The decision remains NO-GO. All three models are now substrate-rich, but none meets the frozen dictionary-bottleneck GO condition. The family-disjoint vs stratified EC ceiling skill gap is large: ProtGPT2 0.260/0.557, ZymCTRL 0.123/0.356, ProGen2-medium 0.272/0.594. Residue-level secondary-structure recovery is near-faithful (rho 0.812-0.981).

**Interpretation:** The corrected analysis converts the earlier muddy NO-GO into a cleaner result: these models contain readable signal, but current CLTs are not shown to discard that signal in a way that justifies the gated capacity retrain. `secondary_fraction` remains unstable and should be treated as a diagnostic rather than a primary claim driver.

**Output paths:**
- Remote probes: `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/results/representation_audit_20260605_1250_recoverability_full/probes_v2/`
- Remote decision: `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/results/representation_audit_20260605_1250_recoverability_full/decision_v2/`
- Remote log: `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/logs/runtime/recoverability_v2_20260606_0015.log`
- Local lightweight copy: `evidence/recoverability_audit_20260605_1250/`

## EXP-R2-020: Exploratory Expanded-Dictionary Retrain (2026-06-06)

**Question:** If the pre-registered recoverability gate is overridden, can wider CLT dictionaries improve feature coverage / recoverability across all three R2 decoder models?

**Method:** Started an exploratory three-model retrain on the beliefnav H200 pod. Each model uses two-GPU DDP (`--gpus 0,1`) and runs sequentially because only two H200 GPUs are available in the pod. The planned configuration is `d_clt=16384`, `k=128`, batch size 1/GPU, 300k steps, resampling every 2500 steps, and dead-feature threshold 5000. ProtGPT2 and ProGen2-medium use UniRef50; ZymCTRL uses the EC-labelled ZymCTRL FASTA.

**Status:** Started as an exploratory override after NO-GO. This is not a pre-registered confirmatory experiment unless a later analysis explicitly reclassifies it.

**Update 2026-06-07:** ProtGPT2 reached `step_10000` and saved a complete checkpoint. The first run then stopped during post-checkpoint DDP synchronization with a NCCL watchdog timeout: rank 0 spent longer than the default 10-minute timeout writing the large checkpoint to OSS while rank 1 waited. The trainer was updated to use a 12-hour NCCL timeout and `dist.barrier(device_ids=[rank])`; the expanded retrain runner now defaults to 50k-step checkpoint intervals instead of 10k. The same output directory was resumed from `step_10000` with remote PID `46401`, and ProtGPT2 had resumed normal training to at least `step_10300`.

**Update 2026-06-08:** ProtGPT2 completed `300000/300000` steps and exited with `DONE protgpt2 status=0`. Its final checkpoint is in `/oss-pvc/zhk_zip/outputs/research2/clt_weights/r2_clt_expand_retrain_20260606_2gpu/protgpt2/step_300000/`. The terminal step spent a long time in an idle-looking state because the same checkpoint was written twice: once by the 50k-step interval and once by the post-loop final save. The queue then started ZymCTRL, which had reached at least `step 100/300000` at the time of the status check.

**Update 2026-06-11:** ZymCTRL completed `300000/300000` steps and wrote its final checkpoint under `/oss-pvc/zhk_zip/outputs/research2/clt_weights/r2_clt_expand_retrain_20260606_2gpu/zymctrl/step_300000/`. The queue then started ProGen2-medium. At the check, ProGen2-medium was running normally on two H200s at about `step 209650/300000`; checkpoints existed through `step_200000`. The checkpoint root size was about `1.2T`.

**Update 2026-06-12:** ProGen2-medium completed `300000/300000` steps and wrote its final checkpoint under `/oss-pvc/zhk_zip/outputs/research2/clt_weights/r2_clt_expand_retrain_20260606_2gpu/progen2-medium/step_300000/`. The three-model queue finished successfully with `ALL DONE`. No CLT training process was running at the status check, and the checkpoint root size was about `1.3T`.

**Output paths:**
- Remote checkpoint root: `/oss-pvc/zhk_zip/outputs/research2/clt_weights/r2_clt_expand_retrain_20260606_*/`
- Remote runtime logs: `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/logs/runtime/r2_clt_expand_retrain_20260606_*/`
- Resume log: `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/logs/runtime/r2_clt_expand_retrain_20260606_2gpu_resume_20260607.log`

## EXP-R2-021: Expanded-Dictionary Recoverability Re-evaluation (2026-06-12)

**Question:** After the exploratory expanded-dictionary retrain completed for all three decoder models, do the wider CLTs improve representation recoverability or change the downstream GO/NO-GO diagnosis?

**Method:** Started the full recoverability downstream pipeline using the expanded `d_clt=16384,k=128,step_300000` checkpoints for ProtGPT2, ZymCTRL, and ProGen2-medium. The run executes `44_cache_representations.py`, `45_probe_ceiling_floor.py`, `46_oracle_direction_steering.py`, and `47_decision_table.py` with `LAYERS=all`, `RESIDUE_LAYERS=even6`, `N_BOOT=1000`, and `N_GEN_46=40`. `RUN_48=0` because this run evaluates the already trained exploratory dictionaries rather than launching another gated retrain.

**Status:** Running on GPU1 of the beliefnav H200 pod. Initial check showed script 44 successfully building the 820-protein cohort and computing ESM-2 reference embeddings.

**Result:** The run completed successfully through 44/45/46/47; 48 was skipped by design (`RUN_48=0`). The final decision remained **NO-GO**. Local improvements were observed for ZymCTRL decoder-native EC (F=0.525 -> 0.623, rho=0.806 -> 0.955) and ProGen2-medium EC/Pfam (EC F=0.153 -> 0.228; Pfam F=0.910 -> 0.931). ProtGPT2 EC/Pfam did not improve. The report-only `secondary_fraction` task remained numerically unstable, and ZymCTRL oracle steering still passed 0/8 EC gates (`distributed_or_robust`).

**Correction 2026-06-13:** Re-ran the 47 decision on the expanded probes with Amendment 1b enforced (`secondary_fraction` is report-only and excluded from the rich/bottleneck gate). Corrected output: `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/results/representation_audit_20260612_expanded_eval/decision_no_secondary/`. The corrected decision remains **NO-GO**, but the reason is now `dictionary already near-faithful on rich tasks`. There are no primary-gate bottleneck tasks. ZymCTRL and ProGen2-medium are near-faithful on their rich tasks; ProtGPT2 is mixed because EC top-class remains just below the faithful threshold (rho 0.785).

Final expanded checkpoint training-quality metrics from the remote logs:

| Model | Final FVU | Dead fraction | Dead% | L0 | Loss |
|---|---:|---:|---:|---:|---:|
| ProtGPT2 | 0.2782 | 0.334 | 33.4% | 128.0 | 7.1176 |
| ZymCTRL | 0.3348 | 0.044 | 4.4% | 90.6 | 0.0201 |
| ProGen2-medium | 0.3101 | 0.122 | 12.2% | 123.0 | 0.0138 |

**Output paths:**
- Remote results: `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/results/representation_audit_20260612_expanded_eval/`
- Remote log: `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/logs/runtime/recoverability_expanded_eval_20260612.log`
- Local summary: `docs/R2_EXPANDED_DOWNSTREAM_ANALYSIS_20260612.md`

## EXP-R2-022: Swiss-Prot Triplet MI Gate Re-audit (2026-07-16)

**Question:** Is the 0.1-nat rich-label mutual-information gate in script 33 attainable for a top-100 firing event, and do the saved Swiss-Prot top positions show annotation association after prevalence calibration and low-level matched nulls?

**Method:** CPU-only reanalysis of the unchanged `swissprot_triplet_annotation_20260513` outputs. Reconstructed the deterministic 500-protein cohort and 122,671 scored positions, reproduced all saved MI values, normalized each MI by the binary top-event entropy `H(B)`, and ran 2,000 permutations for (i) a global uniform-position null and (ii) a within-protein null matched on amino acid and fine normalized-position/edge strata. Applied BH correction across 38 triplets x 10 rich-label families (380 hypotheses). GPU and host memory were checked before and after; the run was CPU-only.

**Result:** The original gate is mathematically unreachable: with 100 positives among 122,671 positions, `H(B)=0.00661255` nats, so 0.1 nats is 15.12 times the maximum possible MI. The reconstructed labels matched all 3,800 saved top rows, and saved MI values were reproduced to maximum absolute error `4.99e-9` nats. The global null yielded 224/380 BH-significant associations, consistent with strong sequence/position selectivity. After matching protein, amino acid and position, 0/380 survived BH correction; the maximum matched excess normalized MI was 0.02180 (T006, `swiss_feature_type`; raw P=0.000500, q=0.09495). The cohort contains 500 unique dominant Pfams for 500 proteins, so `dominant_pfam` MI is sequence selectivity rather than replicated family-generalization evidence. This is an exploratory repair on saved top events, not a confirmatory replacement gate.

**Output:** `results/circuit_analysis/swissprot_triplet_mi_reaudit_20260716/`

**Padding/mask audit addendum:** Reconstructed the exact 100-sequence, batch-size-2 quick-evaluation tokenization. The saved totals include 298/8,342 (3.57%) pad positions for ProtGPT2 and 673/24,466 (2.75%) for ZymCTRL. `get_activations` never receives the tokenizer attention mask, and CLT loss/FVU and firing counts are unmasked. Training-like random-pair diagnostics had larger mean padding fractions (20.81% ProtGPT2, 20.23% ProGen2 on a 1,000-sequence eligible UniRef50 sample; 10.66% ZymCTRL on a 3,000-sequence local EC sample), but these sample estimates are not exact remote-training fractions. Details: `padding_mask_audit.json` in the output directory.

**Reproducibility update:** Added `scripts/49_reaudit_triplet_mi.py`, with canonical defaults `--seed 20260716 --n-perm 2000`. The script fails fast unless it reconstructs the expected 500-protein/122,671-position cohort, 38 x 100 unique top events, all 3,800 saved rich labels, and the saved plug-in MI values within `1e-5`; it also refuses to write into the canonical source directory. Verified with `python -m py_compile`, then ran the full command:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ct
python scripts/49_reaudit_triplet_mi.py
```

The full rerun returned the same headline counts (224/380 global BH-positive; 0/380 matched BH-positive) and reproduced both numeric tables byte-for-byte: `mi_reanalysis.tsv` SHA-256 `c144bfb16123bc774aaf5a22b291bba113ea0d5f2058a0f05e7fb11a39941556` and `triplet_summary.tsv` SHA-256 `96296e31170492f88b32ca709f54f6039a153e1a17fe66bc3827ca4d3ac94784`. `run_manifest.json` now records the exact command, resolved parameters, environment, source-script hash, input hashes and output hashes.

## EXP-R2-023: npj evidence-corrected manuscript and release package (2026-07-16)

**Question:** Can Paper A be converted to the official Springer Nature/npj Article format while ensuring that every semantic, intervention and recoverability statement respects the audited evidence?

**Method:** Selected npj Artificial Intelligence as the primary venue, staged the official December 2024 Springer Nature v3.1 package, and assembled 20 Nature-hosted open-access npj papers from the preceding three years (10 npj AI, 10 npj Systems Biology and Applications). Rewrote the title, 149-word abstract, Introduction, Results, Discussion, Methods, statistics and availability statements; moved all ten tables into a standalone supplement. Rebuilt all six figures from canonical files, including a vector workflow, all saved target/control intervention rows, all 100 N-terminal contexts and machine-readable wider-dictionary values. Created a compact source-data builder with 58 evidence files and full provenance/hashes. Added the July evidence correction to the recoverability amendment log and corrected source-code documentation for the legacy `resid_pre`/`resid_post` field names.

**Commands:**

```bash
source /Data/lzp/BioInterpretebility-CC/.venv/bin/activate
python -m py_compile scripts/50_make_manuscript_figures.py \
  scripts/49_reaudit_triplet_mi.py
python scripts/50_make_manuscript_figures.py
python manuscript/source_data/build_source_data.py --verify-only

source ~/miniconda3/etc/profile.d/conda.sh
conda activate latex
cd manuscript
tectonic --reruns 2 main.tex
tectonic supplementary_information.tex
```

**Result:** PASS for the manuscript/package build. `main.pdf` is 18 A4 pages with six figures and no main tables; `supplementary_information.pdf` is 11 A4 pages with exactly ten tables. Both compile without errors, unresolved references or overfull boxes. The source-data verifier passes 58 manifest rows and 61 checksums (1,410,189 bytes). All 20 literature PDF hashes pass. The manuscript's `sn-jnl.cls` and `sn-nature.bst` are byte-identical to the staged official package. Remaining blockers are human metadata, restoration/deposition of the exact 200-sequence cohort, immutable upstream-model revisions and a DOI-backed release.

## EXP-R2-024: Canonical directory and manuscript-package migration (2026-07-16)

**Scope:** Structural migration only; scientific result files and historical execution metadata were not rewritten.

**Changes:** Renamed the live root to ``; moved compact recoverability evidence to `evidence/`, plans and analysis notes under `docs/`, and manuscript audits under `docs/manuscript/`; renamed the official package directory to `manuscript/template/`; normalized figure and source-data directory names; and renamed the figure builder to `scripts/50_make_manuscript_figures.py` to remove a duplicate prefix.

**Verification:** Rebuild all six figures and the source-data package, verify the literature/template/source-data hashes, compile the main and supplementary manuscripts, parse all first-party Python/shell files, and confirm both Mutagen endpoints contain only the canonical live roots. External GPFS/OSS/H200 paths retain their historical namespaces until separately migrated.

## Retrospective May chronology addendum (logged 2026-07-17)

The July assessment identified a gap in this chronological log: the central 13--18 May atlas, characterization and intervention runs were present in immutable result directories but absent from the numbered narrative above. This table restores their dates without renumbering or reclassifying the historical experiments.

| Date | Historical run | Audited result and boundary | Canonical output |
|---|---|---|---|
| 2026-05-12/13 | Balanced-200 atlas and 30-replicate assignment null | 38/30/8 triplets at the three thresholds; null maximum 1. The null independently permuted each pair/layer edge and was not a coherent model-wise null. | `results/circuit_analysis/*balanced200_wide*` |
| 2026-05-12 | UniRef500 atlas sensitivity | 8 triplets at 0.90 and 3 at 0.95; no exact feature-identity overlap with the canonical 38. | `results/circuit_analysis/*uniref500_wide*` |
| 2026-05-13 | Original Swiss-Prot gate | Saved 0/38 result under the later-proven impossible 0.1-nat threshold. | `results/circuit_analysis/swissprot_triplet_annotation_20260513/` |
| 2026-05-13 | Triplet-basis probes | The 38-dimensional basis was below ESM-2 on all three quick tasks; folds were record-level, not family-aware. | `results/circuit_analysis/triplet_basis_probes_20260513/` |
| 2026-05-14/15 | Characterization and synthesis | 37/38 triplets had at least one low-level association and 21/38 had at least three; tests overlap and do not establish independent semantics. | `results/circuit_analysis/triplet_*20260515_nperm2000/` |
| 2026-05-14 | Attention-output sparse pilot | High selected readout, but the causal ablation moved opposite to the stated direction; saved uncertainty values are means, not bootstrap intervals. | `results/circuit_analysis/attention_output_transcoder_pilot_20260514_l23/` |
| 2026-05-16 | N-terminal subset/context | Three initiator-methionine readouts; 18/24 exploratory context tests. Received attention was unnormalized for eligible causal queries. | `results/circuit_analysis/attention_sink_*20260516/` |
| 2026-05-16 | One-pair checkpoint diagnostic | Mature reference recovered 38 versus 16 for the early ProtGPT2/ZymCTRL pair; one comparison is a candidate diagnostic, not validation. | `results/circuit_analysis/universal_atlas_quality_diagnostic_20260516/` |
| 2026-05-17 | CLT feature patches | All nine model--triplet gates failed for the tested sites and strengths. | `results/circuit_analysis/attention_sink_causal_ablation_20260517/` |
| 2026-05-18 | Single-head, top-8 and top-32 head-set ablations | Every strict/exploratory gate failed; these are bounded negatives, not proof of no mechanism. | `results/circuit_analysis/attention_*ablation*20260518/` |

## EXP-R2-025: npj assessment response and confirmatory infrastructure (2026-07-17)

**Question:** What revisions and new experiments are required by `docs/npj_ai_manuscript_assessment.md`, and can the submission-critical pipelines be corrected, frozen and safely started without overstating their current evidential status?

### Evidence and manuscript corrections

- Reclassified the project as major revision/not submission-ready and created `docs/NPJ_AI_MAJOR_REVISION_PLAN_20260716.md`, which itemizes all nine P0 work packages, acceptance criteria and claim limits.
- Corrected the manuscript and supplement for cohort sensitivity, no canonical feature-identity overlap in UniRef500, the incoherent historical atlas null, the steering selector's 19 negative-attribution interventions in six cells, record-level basis-probe folds, the attention-pilot uncertainty wording and Figure 5's mixed-run chronology.
- Retitled the Article around cross-model sparse readouts rather than implying stable conservation. The older lysozyme z-score/off-manifold run is now kept distinct from the 3 May direct-effect pilot.
- Packaged the complete direct-effect candidate summary and normalized all non-finite release JSON values to `null` with path/token/source-hash metadata. The strict source package verifies 59 manifest rows, 62 checksums and 1,469,342 bytes after the final claim-boundary rebuild.

### P0-1/P0-2 reproducibility and dictionary corrections

- Reconstructed the documented alternating 100-real/100-random historical cohort with full records and hashes. Its ordered-cohort SHA-256 is `07213d4a9cefbdb055206e08d3137722c446acf43b5b3342db571b977032c724`. It remains `reconstructed_unverified`: a common row permutation cannot be proven from atlas correlations, and no missing upstream revision is invented.
- Made model forwarding, CLT loss/FVU/L0/firing/dead tracking, resampling and held-out evaluation attention-mask aware. Added deterministic seeds, rank-disjoint shuffling, immutable-manifest validation and RNG/data-cursor checkpoint state. Removed unused configuration keys and the duplicate final checkpoint write.
- Froze `docs/P0_2_DICTIONARY_PROTOCOL_20260717.md` before new training. The confirmatory seeds are 17/29/43; the TopK CLT is `d_clt=8192`, `k=128`, window 8 for 200,000 steps. Alternative sparse/dense controls share the same multi-layer transcode estimand and immutable splits; their production run is fail-closed until the frozen profile validates.
- On H200, hashed the full deployed FASTAs. UniRef50 is `0eb363b2...aa2c1a`; ZymCTRL is `2a1253ce...74cda3`. Frozen UniRef50 train/validation/test files contain 300,000/5,000/5,000 unique records; the training file hash is `f4270143...2525bb`. The ZymCTRL split contains 44,000/2,500/2,500 unique proteins after removing 15,386 exact duplicates; its training hash is `f9727a37...3f755`. EC prompts are header-verified, removed from the protein sequence field and reconstructed only for model input.

### H200 execution

The updated remote health check passed. The selected pod is `damoxing-zhk-zipbio-master-0`, with four H200 GPUs (143,771 MiB each), about 1.8 TiB host memory available and GPFS/OSS mounts. A four-step ZymCTRL masked training smoke test passed and wrote hash-pinned configuration and trainer state; all GPUs returned to 0 MiB afterward.

The master's documented `/gpfs` path is actually an ext4 directory and is not the GPU pod's GPFS mount. The checksum-verified helper transfer therefore lands on the master and requires a second explicit `kubectl cp` bridge into the pod. The production P0-2 archive SHA-256 is `556f7ef8519a1d669921d195919cfa229885c664c925f0f309b9fc77bc4bc684`.

Three independent queues were launched after live GPU/memory checks:

| Seed | GPU | Pod PID | First healthy log | Queue scope |
|---:|---:|---:|---|---|
| 17 | 0 | 1302 | step 50, 0.25 s/step, 53,355 MiB | ProtGPT2 -> ZymCTRL -> ProGen2-medium |
| 29 | 1 | 1777 | step 50, 0.25 s/step, 53,355 MiB | ProtGPT2 -> ZymCTRL -> ProGen2-medium |
| 43 | 2 | 2253 | step 50, 0.25 s/step, 53,361 MiB | ProtGPT2 -> ZymCTRL -> ProGen2-medium |

The queues run under `/oss-pvc/zhk_zip/outputs/npj_revision_20260717/`; GPU 3 remains unoccupied for other confirmatory work. Launch is not a quality-gate pass. Held-out evaluation and all downstream atlas work wait for completed checkpoints.

**Superseded 2026-07-17:** all three queues in the table above were later discarded after the uncapped step-15,000 resampling event made seed 43 non-finite; seeds 17/29 were stopped as incomparable. GPU 3 is no longer free: it runs the production ProtGPT2 exact-cache job. The authoritative replacement launch is the archive-bound r6 queue recorded at the end of this log.

### P0-3/P0-4 infrastructure

Implemented frozen discovery-A/evaluation-B atlas scoring, signed and absolute analyses, greedy/Hungarian/optimal-transport/joint matching, coherent model-wise plus-one permutation nulls, stability/Jaccard/ambiguity diagnostics and variance decomposition. Implemented continuous conditional semantics with protein/family blocking, within-protein randomization, matched sparse/dense/null controls, global BH correction and protein bootstrap. A later audit corrected the initial power labeling: standard errors bootstrapped from analyzed data now yield only a retrospective detectability diagnostic, while a confirmatory run requires a complete SHA-bound independent-pilot power plan for prospective MDE reporting. Synthetic CLI tests pass; no real P0-3 or P0-4 result exists yet. See `docs/NPJ_REVISION_P0_3_P0_4_EXECUTION.md`.

### P0-7/P0-8 synthetic pipeline validation

The planted causal benchmark used five model seeds and independent 256-record discovery/evaluation cohorts. It recovered every planted path with sensitivity/specificity 1.0, FDR 0 and effect-recovery ratio 1.0; all 15 matched controls were TOST-equivalent inside the frozen +/-0.10 band. Summary SHA-256: `f37953b3...40592`. This passes the synthetic positive-control gate only; it is not evidence of causality in a pretrained protein model.

The nested recoverability synthetic run used 192 samples, 48 identity groups, five analysis seeds, two dictionary seeds, 25 outer and 100 inner group-disjoint folds, and PCA/random-projection/NMF/ICA/random-dictionary controls fit only on outer training data. Mean floor-minus-ceiling macro-F1 was -0.09587 and -0.08520 for the two code seeds; all paired group-bootstrap intervals excluded zero. Summary SHA-256: `c41d10bf...0dc24`. This validates plumbing only, not a protein recoverability gate. See `docs/P0_7_P0_8_CONFIRMATORY_PROTOCOL_20260717.md`.

### Verification and current disposition

- Full revision suite at the P0-7/P0-8 handoff: 40 tests plus 6 subtests passed; strict JSON, artifact/source hashes, `py_compile` and Ruff passed.
- Steering positive-only selection/paired-statistics tests and mask/split tests passed locally.
- Main and supplementary Tectonic builds pass without unresolved references or overfull boxes; only underfull warnings remain.
- Source-data verification passes at 59 rows/62 checksums/1,470,095 bytes after the literature/provenance README refresh.

The analytic simulator and trained control provide synthetic/development plumbing checks only; neither is a confirmatory positive-control pass. P0-1 through P0-9 are not collectively complete. The manuscript remains not submission-ready while the replicated dictionaries, real held-out atlas/semantic/counterfactual/steering/ recoverability experiments, symmetric Figure 5 replacement, final provenance map, human metadata and DOI-backed release remain pending.

## 2026-07-17 — P0-2 alternative-control production contract and CPU validation

No GPU job was launched in this step. The alternative-control estimand was fixed to the same window-8, architecture-specific CLT-input to MLP-output transcoding target as the primary CLT. The production profile now freezes 1,000,000/100,000/100,000 valid-token cache budgets, exact ZymCTRL prompt reconstruction, tokenizer behavior, float16 storage, model geometry, a validation-only 10,000-step screening stage and full 200,000-step seeds 17/29/43. Dense rank 128 is active-width matched and explicitly not raw-parameter matched.

The cache extractor uses a path-independent hash priority over sequence digest and unpadded token position, writes the exact priority order to JSONL, and preallocates one input/target array per layer and split. Frozen payload estimates are 221,184,000,000 bytes each for ProtGPT2/ZymCTRL and 199,065,600,000 bytes for the deployed 27-layer, width-1,536 ProGen2-medium (641,433,600,000 total; 769,720,320,000 bytes at the 1.2 panel gate). The exact-cache panel now includes cached TopK, ReLU/L1, gated and dense methods. The planning basis of one optimizer step/second gives 125 screening plus 2,000 full-run GPU-hours, 2,125 GPU-hours total.

Independent launch audit then identified and closed ten blockers: corrected ProGen geometry; aggregate receipt-backed panel capacity gating; train/ validation-only screening reads; inference-mode evaluation; final-step resume; runtime-independent cache identity; local deployed-artifact hash verification; a bounded P0-2-ineligible cache plus one-step full-width optimizer preflight; cumulative resource/checkpoint-I/O planning; and an exact-cache TopK comparator. Existing online TopK queues are now explicitly preliminary and non-comparative. The frozen storage plan reports 3,429,943,879,680 retained checkpoint bytes and a conservative 73,010,316,902,400-byte worst-case write volume. No exact-cache GPU job was launched during this hardening step.

Verification command:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ct
ruff check scripts/58_run_dictionary_controls.py \
  scripts/61_build_dictionary_activation_cache.py \
  src/revision/dictionary_controls.py tests/test_dictionary_controls.py \
  tests/test_dictionary_cache_builder.py tests/test_mask_contract.py \
  src/models/model_loader.py
python -m json.tool configs/p0_2_dictionary_controls_production_profile.json
PYTHONPATH=. pytest -q tests
```

Result after audit hardening: Ruff and strict JSON passed; 80 tests plus 6 subtests passed. The CPU smoke covers prompt reconstruction, two-pass token-ID identity, exact budget selection, padding exclusion, priority-aligned cache rows, free-space failure, layer-indexed hooks, resumability and best-checkpoint reload. Production cache extraction and alternative-control training remain unlaunched pending final profile/hash review and live storage/GPU checks.

## 2026-07-17 — Live H200 stability failures, bounded P0-2 preflights and amendments

### Preliminary online TopK failure and discarded queues

The original three online TopK queues used code archive SHA-256 `556f7ef8519a1d669921d195919cfa229885c664c925f0f309b9fc77bc4bc684`. At the step-15,000 resampling event, seeds 17/29/43 replaced respectively 193,287/200,333/199,016 dead features (65.5%/67.9%/67.5%) from one small activation batch. Seed 43 first logged a non-finite loss at step 15,400. That queue was stopped; seeds 17 and 29 remained finite through approximately 25,100/24,950 but were also stopped because their uncapped histories were not comparable to a corrected restart. These runs have no scientific eligibility and their logs remain only as failure diagnostics.

The amended trainer selects dead features oldest-first, caps each layer/event at 5%, fails immediately on non-finite loss or gradients, and validates finite model/optimizer state before publication. A 600-step full-width seed-43 stress smoke used resampling every 50 steps after a 100-step inactivity threshold. Nine resampling events each respected the 14,724-feature panel cap and the run remained finite. A separate resume from step 300 reproduced the uninterrupted step-600 model byte-for-byte: `clt.pt` SHA-256 `d5c5cdef06f385ca510e6c92d73860175c72671a21602d753d3a625848d67867`. All 39 optimizer tensor states and parameter groups, scheduler state and RNG/data-cursor state also matched semantically. This validates bounded stability/resume behavior only; short-run FVU/dead rates are not quality results.

An independent launch audit then found that a successful resume smoke alone was insufficient. The trainer/launchers were hardened to require fresh versioned GPFS outputs, exact resolved-config/source/world-size/step/scheduler/ cohort binding, complete file inventories and hashes, nonzero exit on arbitrary interrupt, corruption checks before retention conversion, rejection of any stale staging directory, frozen profile/config/cohort hashes, idle GPUs and a verified final checkpoint manifest. Active resumable retention is two full checkpoints plus model-only snapshots every 25,000 steps. Fourteen focused mask/checkpoint tests pass. Replacement three-seed queues remain unlaunched until the amended archive passes a second independent audit and a live full-width checkpoint-publication smoke.

### Exact-cache bounded H200 preflights

The first cache preflight failed before model loading because the runner reset CUDA peak-memory counters before initializing explicit device `cuda:3`. No cache data were written. The fix calls `torch.cuda.set_device` before resource accounting; the same correction was applied to the dictionary optimizer runner, with pre-update finite loss and non-finite-gradient failure.

The corrected ProtGPT2 cache preflight completed on GPU 3 with exactly two valid rows in each of train/validation/test. It took 29.536 s, including 17.614 s cache construction and 3.615 s complete hash validation; peak CUDA allocated/reserved memory was 1,935,082,496/2,199,912,448 bytes. The cache is explicitly P0-2-ineligible and forbidden for production reuse. Hashes:

- report: `1d1586eab9251e71cda9a22a50ee34c4b20d94eac39846368b3690d5e1731977`;
- derived profile: `44d375039011d6f121c1fe10efa5bdb938577da690e7436ceb83cf6f798b1a75`;
- cache manifest: `508bfe3e60f33acff682dee76c77222acc1642c00115e66219457adcdd49b167`;
- cache content: `f301f4c42c621102c36d45b53852763cb6ba52c4f9ff016290ab659dbba49fab`.

The full-width gated-SAE optimizer preflight then completed one update on two train rows and one inference-only validation pass, with zero test accesses. The model had 3,104,715,776 trainable parameters; the step took 0.4850 s and peak allocated/reserved memory was 62,162,163,712/62,182,653,952 bytes. Train and validation losses were finite (272.8913 and 428.4346). Report SHA-256: `5c1d0e39bc453c1fc3403f3e9eb4a1620025dd942856da8ae3e82299228d199a`. This report is also non-confirmatory and non-reusable.

After both bounded gates passed, the real ProtGPT2 activation-cache extraction was launched as pod PID 5431 on GPU 3 at `/gpfs/jiaotongdamoxing/zhk_zip/biocc/npj_revision_20260717/p0_2_exact_cache_r1/`. It uses production profile SHA-256 `dca7b8ec335e9a5814b2d58a7199d7de39d667559c6e30194478bcca5f68a9c2`, code archive SHA-256 `6e263f7937229ddc29a0eb5c15cb997a67e28761911aff3e78dafd218c16ac4d`, and the frozen 1,000,000/100,000/100,000 valid-token budgets. Launch is not a completion receipt or quality-gate pass; ZymCTRL/ProGen caches and all exact-cache training remain pending.

### P0-7/P0-8 corrected synthetic status

The trained GRU/fixed-adapter control v3 completed with summary SHA-256 `b747f8f7041a71ae21949ce7299883b6bd7eb2f3d6033ce56d3a0660d7501929` and manifest SHA-256 `1db451a83b2336ec56dde00b072b3119b5f57b52f5fca7b6226ab4a304c0f411`. It is retained only as a deliberately easy, post-hoc-developed synthetic CLT/intervention-hook smoke: the label is explicit, endpoint displacement is analytically planted, and the same evaluation cohorts were used while raising CLT training from 100 to 150 steps. It is not a frozen confirmatory positive control, does not validate pretrained-transformer path patching, and does not close P0-7.

The dimension-truthful nested-recoverability v2 synthetic run records the requested and actual dimensions without silent caps and performs paired group bootstraps against every control. Summary/manifest SHA-256 values are `eccda8a4fc2903781d6ed1063cf5d440865ba18fd817c6427084e0b939c44709` and `47f8f16cd395b4fc54bafb3058e73038e12181436cc8690dc3879ddb0c34a080`. It remains synthetic plumbing evidence only; real common-dimensional pretrained representations, quality relationships and interventions are pending.

## 2026-07-17 — P0-3/P0-4 production input-builder contract

Implemented the missing production bridge as one reusable module (`src/revision/input_builder.py`) and one thin CLI (`scripts/64_build_revision_inputs.py`). The builder verifies disjoint frozen cohort JSONLs, deployed model config/weight/tokenizer trees and complete CLT checkpoint manifests before extracting anything. It uses exact decoded token-to-residue coverage, excludes prompt/special tokens, and emits SHA-bound continuous residue-weighted atlas matrices for all requested layers.

For conditional semantics it consumes spec-frozen model/layer/feature identities and a complete binary residue-annotation JSONL. It fits matched dense randomized-PCA directions only on a hash-selected discovery-cohort sample, constructs a separately seeded random dictionary, projects held-out rows without retaining the full dense held-out matrix, generates within-protein prevalence-matched negative labels, and writes a directly executable script-53 spec plus full input provenance. The whole output tree is staged and published atomically without overwrite.

The emitted semantic spec is explicitly non-confirmatory unless the builder input supplies a SHA-bound, independently sourced prospective power plan with exact hypothesis coverage under script 53's schema. Confirmatory-data bootstrap standard errors are not reused or relabeled as prospective inputs.

CPU fixture verification exercised the complete build, six atlas matrices per cohort, the generated NPZ/spec through the existing conditional-semantics runner, deterministic residue mapping/dense fitting, and fail-closed cohort, checkpoint/model tamper detection and prospective-power-plan passthrough. Result: 4 tests passed; focused Ruff, `py_compile` and `diff --check` passed. No real checkpoint extraction or H200 job was run in this step. P0-3 and P0-4 remain failed pending passing P0-2 dictionaries, real frozen annotations/cohorts, full extraction and prespecified scientific gate adjudication.

Final integration verification after the prospective-power contract update: 97 tests plus 6 subtests passed; strict example-JSON parsing and focused Ruff, `py_compile` and repository `diff --check` also passed.

## 2026-07-17 — TopK archive binding, checkpoint-publication smoke and replacement launch

The second independent launch audit returned one final NO-GO: the queue had accepted both an archive path and its expected digest from the caller and did not prove that `PROJECT_ROOT` was extracted from that archive. The clearance uses a deliberately separate launcher (avoiding a self-referential archive digest), a hard-coded archive/deployment path, a hard-coded archive SHA-256, the archive-contained whole-tree checksum manifest, exact deployed file inventory, and a no-symlink rule. Deployment-only verification passed on the pod with:

- payload archive SHA-256 `acf6534f0ea0e1082bc1063b4450b01cba7f0357052f071650b5942d0aa7906e`;
- tree-manifest SHA-256 `9a0cf1a781d1feca1fed26cdbc9d0e22ac8e01a6292d2c874b96abc6fe3be1c4`;
- final r6 launcher SHA-256 `b190585e2b546f29fd412a9b6a22f63d80f9deccdc66ba7b8cff24d18605f21c`;
- confirmatory configuration SHA-256 `a12a210ec6b11304c4d3986570897146369f9fe5bdcb5bbdcc505b7344267fba`.

Fourteen adversarial checkpoint/mask tests passed. The independent auditor then returned GO.

The first full-width publication smoke used two steps and stopped before training because the OneCycle scheduler requires at least four steps. Its log SHA-256 is `bc6d6ab21a5f7151092b81fc5e8575f10d3bf3e1d70d942a5a5abd30841c4812`; no checkpoint was written. The fresh four-step rerun on GPU 0 atomically published and independently reverified a complete 37,251,935,365-byte resumable checkpoint. Its checkpoint-manifest SHA-256 is `9b87370996984eebe3aaf0f01b77e1e01db2c61f938a46b92bbdfea235ca0996` and log SHA-256 is `fddbd4c9eb92c7b301b871583932a7aca0f0bd0651074372a482b230fdc4cf98`. The manifest binds step 4, world size 1, trainer SHA-256 `e6551ca3caa0ba9aa2326451bcbbeba2673913d5fd1ed9505f3bc441696385a8`, and complete model/optimizer/scheduler/config/RNG-state hashes.

An initial r5 queue launch then correctly failed before GPU use because the direct smoke had created unlisted Python bytecode in that deployed code tree. No training output was created. The identical verified payload was extracted into a fresh r6 tree; the launcher disables bytecode writes and deployment verification passed again.

Replacement TopK queues launched at 2026-07-17 14:59:47 CST:

| Seed | GPU | Pod PID | Initial finite observation |
|---:|---:|---:|---|
| 17 | 0 | 7143 | step 100, loss 201.5742, FVU 0.9860, 0.24 s/step |
| 29 | 1 | 7144 | step 100, loss 127.7179, FVU 0.9857, 0.24 s/step |
| 43 | 2 | 7145 | step 100, loss 161.5224, FVU 0.9868, 0.24 s/step |

Each seed runs ProtGPT2, ZymCTRL and ProGen2-medium sequentially under the fresh GPFS root `dictionaries_topk_stable_r1`. GPU 3 continues the independent production ProtGPT2 exact-cache extraction as PID 5431. These are launch and plumbing facts only: no held-out dictionary quality gate or P0 scientific gate has passed.

## 2026-07-17 — P0-6 real generation executor contract

Implemented the missing hash-bound generation bridge for the corrected eight-class steering rerun in `src/revision/steering_execution.py` with the thin CLI `scripts/63_execute_corrected_steering.py` and deployment template `configs/p0_6_steering_executor_spec.example.json`. The executor independently anchors the script-60 freeze by its externally supplied manifest SHA-256, rechecks every frozen artifact and source hash, verifies the declared local model config/weight/tokenizer trees and complete CLT checkpoint, and rejects model/tokenizer/checkpoint identities or decoder norms that differ from the plan.

The only implemented intervention is the plan's exact `additive_decoder_direction_v1` contract: add the dose-scaled same-layer CLT decoder vector at every active token at either the architecture-specific CLT input or MLP output. Unsupported sites or semantics fail. Live hook receipts verify and retain nonzero realized displacement, invocation/token counts, vector hashes and numerical error. Python, NumPy and Torch streams are reset for every plan row, while frozen temperature/top-p/length settings and fixed `do_sample=True`, `top_k=0` settings are retained with prompt and continuation token IDs. Every planned row must validate before an atomic directory publish; errors, interrupts, existing outputs and stale partial staging all fail closed.

CPU fake-model coverage exercises prompt-only, target, disjoint random and norm-matched arms at both hook sites, deterministic paired sampling, full script-60 output compatibility, freeze tamper detection, unsupported site/semantics, overwrite refusal and cleanup after a planted mid-run failure. Focused result: 3 tests passed; Ruff and `py_compile` passed. No H200 generation, endpoint scoring or pretrained-model P0-6 analysis was run. This is execution infrastructure only, and P0-6 remains open.

## 2026-07-17 — P0-4 global multiplicity receipt contract

Added the standalone fail-closed collector `src/revision/semantic_adjudication.py`, thin CLI `scripts/67_adjudicate_conditional_semantics.py`, frozen-spec template and `docs/P0_4_JOINT_ADJUDICATION_PROTOCOL_20260717.md`. The collector requires an externally supplied hash of its exact run list; verifies every run manifest, script-53 spec, NPZ, power plan and output; reconstructs each run's full hypothesis cross product; independently detects within-protein-degenerate biological labels; and rejects missing, extra or duplicate hypotheses.

Per-run q-values are retained only as ignored provenance. One BH correction is recomputed over the complete model/layer/representation/feature/label/blocking union. Retrospective bootstrap detectability remains separately labeled from hash-bound prospective MDEs. Joint prospective MDEs use independent standard errors with conservative planning over the full global family; powered-bound adjudication refuses any incomplete plan. Twelve adversarial and decision-rule CPU tests passed, with focused Ruff and `py_compile` checks also passing. No real semantic run or scientific gate was adjudicated. A verified joint receipt and real frozen runs are both required before P0-4 can pass; P0-4 remains open.

Verification command/config: `PYTHONPATH=. pytest -q tests/test_semantic_adjudication.py`, plus focused `ruff check`, `py_compile`, strict example-JSON parsing, dataclass/table schema comparison and repository `diff --check`; result: 12 tests passed and every static check passed.

## 2026-07-17 — P0-6 endpoint scorer execution receipt

Closed the remaining analysis-side provenance gap in `src/revision/steering_protocol.py` and `scripts/60_prepare_corrected_steering.py`. A real `--stage analyze` invocation now requires `--score-receipt` plus its separately supplied byte-exact `--score-receipt-sha256`. The receipt binds the freeze ID, generation-output file SHA-256, score-file SHA-256 and frozen endpoint-specification SHA-256. It must contain exactly one complete execution for each and only each frozen endpoint.

Every validated execution must exactly reproduce the frozen scorer name/version/artifact SHA-256, calibration-cohort SHA-256, validated/primary flags, experimental unit, expected unit count and the canonical hash of the exact plan-ID or generation-set-ID coverage. Heuristic executions remain explicitly `heuristic_supporting_only` and cannot masquerade as validated. Partial, extra, duplicate, non-complete, identity-mismatched, file-mismatched or coverage-mismatched receipts fail before an analysis output directory is created. `completed_confirmatory_analysis` is assigned only after this receipt validation succeeds. The synthetic pipeline constructs its own explicit synthetic receipt and retains `synthetic_pipeline_validation_only` status.

Added the annotated exact-schema template `configs/p0_6_score_receipt.example.json` and documented the scoring boundary and revised invocation in `docs/P0_5_P0_6_CONFIRMATORY_PROTOCOL_20260717.md`. Focused adversarial coverage includes receipt/artifact tamper, generation and score mismatch, partial/extra endpoint executions, validated-scorer identity mismatch, heuristic validation masquerade, incomplete execution and incomplete coverage. Verification: `PYTHONPATH=. pytest -q tests/test_steering_protocol.py` reported 11 passed; focused Ruff and `py_compile` passed. A CPU-only full freeze-to-analysis smoke accepted the synthetic receipt while retaining the synthetic-only status. No H200 execution, endpoint scoring or scientific gate was performed; P0-6 remains open.

## 2026-07-17 — P0-8 exact-cache production input and receipt contract

Implemented the missing production bridge in `src/revision/recoverability_input_builder.py` with the thin CLI `scripts/68_build_nested_recoverability_inputs.py` and an annotated deployment template. The production path first verifies the new P0-2 panel eligibility receipt, including the exact model/method adjudication, all seeds 17/29/43, source-manifest hashes, selected-run hashes, `best.pt` hashes and eligible downstream layers. Only then does it load the authorized exact-cache TopK checkpoints through `load_eligible_topk_clt`; online `clt.pt` checkpoints and unreceipted arrays are not accepted.

The builder binds the local model config/weight/tokenizer trees, the complete P0-2 train/validation/test cohorts, a disjoint enlarged P0-8 evaluation cohort, task annotations and row order, and exact identity assignments. It rejects exact-sequence or identity-cluster overlap across P0-2 and P0-8. Targets and groups are never passed to the representation extractor. The built-in path extracts CLT input, MLP output, per-seed sparse code and per-seed reconstruction at every authorized layer. It derives reconstruction error internally and accepts intervention effect only through a separately hash-bound, complete, row-aligned execution receipt.

One atomic build publishes exactly an NPZ, a frozen script-55 runner spec and an input receipt. Script 55 now requires that production receipt and its external SHA-256 for every real run; the receipt, NPZ, runner-spec hash and all analysis arguments must agree before `confirmatory_real` can be set. Fixture receipts, arbitrary arrays and parameter substitutions fail closed.

CPU fake-model/exact-cache-adapter coverage exercised the full build and direct script-55 loading, plus arbitrary-array injection, non-eligible P0-2 receipt, checkpoint-hash substitution, identity leakage, missing intervention rows, production loader overrides, fixture-receipt misuse and runner-argument tamper. Verification: 9 focused builder/receipt tests plus the 7 existing nested-recoverability tests (2 subtests) passed; focused Ruff, `py_compile`, strict example-JSON parsing and `diff --check` passed. No real model extraction or H200 job was run. P0-8 remains open until P0-2 yields eligible exact-cache dictionaries and the enlarged frozen tasks are executed and adjudicated.

## 2026-07-17 — Exact historical balanced-200 cohort recovered

The archived path named by the saved May atlas provenance was found in the authorized H200 pod at `/oss-pvc/zhk_zip/biocc/Research2/results/ec_metrics/calibration_lysozyme_balanced200_20260511.json`. The file is 81,480 bytes with SHA-256 `5bc7697a83cc7461558f8b4597a3c9b4d6a151b7ec70ca22efc7282ecde4f0a6`. It records the expected 100 real lysozyme and 100 random UniRef50 proteins in strict alternating order. Every ordered record object exactly equals the previous reconstruction before the added cohort index, split, family and sequence-hash provenance fields. The canonical enriched ordered-record hash therefore remains `07213d4a9cefbdb055206e08d3137722c446acf43b5b3342db571b977032c724`.

The source was copied through `kubectl cp`, transferred with the refreshed SHA-verifying helper and retained under `results/npj_revision_20260716/manifests/historical_recovery/`. Script 51 now accepts only a separately pinned historical-file SHA, verifies its exact schema/source/construction and ordered record equality, and otherwise retains the old `reconstructed_unverified` status. Two mismatch/upgrade tests pass. The rebuilt local manifest has status `historical_exact_file_verified`, cohort output SHA-256 `9a90ba10143cc6fc7f097d74b49e706e4b3d9e422d0173976312ff968c902052` and run-manifest SHA-256 `b777213fa3edaacac55da80050cf999fa2ac4ad50983910cd5cb10f4a4b3dd3a`; a separate `--verify-only` run reproduced those hashes.

This resolves only the local historical row order. At this point in the chronology, P0-1 remained open because immutable historical upstream revisions, not-yet-recovered reference CLTs, complete raw generation/intervention artifacts, a tagged release and a DOI-backed licensed deposit were still missing. No historical atlas was reinterpreted and no scientific gate was upgraded.

## 2026-07-17 — Historical atlas reference CLTs recovered

The three exact reference CLT paths named by the saved May atlas output were located on the authorized pod's archived compute storage and hashed in place:

- ProtGPT2 `step_200000/clt.pt`: 12,418,875,728 bytes, SHA-256 `5eca3b19284dbd9b302078e3a7e34ce7a2fc78d97b1566eae927d4d1c30f1f00`;
- ZymCTRL `step_200000/clt.pt`: 12,418,875,728 bytes, SHA-256 `5da70c530b83a034d1fe683a72a8cc5bd7b49463d2598036cd6b5db94ca5761d`;
- ProGen2-medium `step_100000/clt.pt`: 5,412,154,231 bytes, SHA-256 `5e384733dc28ecad3947b65c0c8b34f058ce50a61aab67399548c2b21687b8fd`.

The exact retained YAML configurations were copied into `evidence/historical_reference_checkpoints_20260717/`; their remote sizes and hashes were reproduced, and `manifest.json` binds the three CLTs to the saved atlas output. No checkpoint was modified or interpreted as new evidence. This removes the local reference-CLT-location blocker only. P0-1 remains open because the files are not in a persistent licensed public deposit, immutable upstream model/tokenizer revisions were not retained, and complete raw generation/intervention artifacts, a release tag and DOI are still absent.

## 2026-07-17 — P0-3/P0-4 exact-cache eligibility integration

Replaced the production input builder's online-training-directory/`clt.pt` assumption with the public P0-2 `load_eligible_topk_clt` contract. Builder schema version 2 now requires a separately SHA-pinned P0-2 eligibility receipt, exact seed-17/29/43 `best.pt` and run-manifest hashes for every model, exact train/validation/test source-manifest maps, a frozen model seed and requested layers within the receipt allowlist. The builder loads and extracts every seed artifact separately and emits matrices keyed by model, dictionary seed and layer. Atlas and continuous-semantics manifests retain the receipt, run seed, run-manifest, checkpoint, model-artifact, source-manifest and layer-eligibility provenance. The P0-2 receipt intentionally authenticates dictionary/cache provenance rather than pretrained-model seed or deployed model-tree digests; the latter remain a separate builder-verified contract whose exact equality is required between discovery and held-out manifests.

The former tiny online checkpoint path remains only as the explicitly named `nonconfirmatory_fixture_checkpoint` under top-level `confirmatory: false`. That path cannot declare a P0-2 receipt, cannot use a production loader override, and cannot emit confirmatory atlas or semantic status. Confirmatory semantics must name an eligible run seed and continue to require an independent prospective power plan.

Script 52 now requires external SHA-256 values for both discovery and held-out builder manifests. For every confirmatory analysis it selects only matrix rows whose embedded seed equals the analysis seed, verifies discovery/held-out model seed and dictionary provenance equality, revalidates the actual P0-2 receipt and all-seed run/checkpoint/source hashes, checks the selected `best.pt`, and enforces its layer allowlist. The confirmatory panel must contain exactly one analysis for each seed 17, 29 and 43, with distinct run-manifest and checkpoint hashes for every model; relabeling the same matrices or checkpoint as another seed fails. The verified seed-artifact map is propagated into both the atlas summary and run manifest.

Focused CPU verification covered the nonconfirmatory end-to-end builder and script-53 handoff, exact public-loader argument propagation, loader-provenance tamper, fixture-to-confirmatory upgrade attempts, custom-loader bypass, manifest-hash tamper, manifest-scope upgrade, seed mislabeling and duplicate seed artifacts. Result: 16 tests passed plus 4 atlas subtests; focused Ruff, `py_compile` and strict JSON parsing passed. After integration with the concurrent steering-contract changes, the complete project suite passed with 144 tests plus 6 subtests. No H200 run or real confirmatory analysis was performed. P0-3 and P0-4 remain open pending eligible P0-2 artifacts and real execution.

## 2026-07-17 — P0-6 exact-cache execution and score-receipt binding

Replaced the real corrected-steering executor's online-training-directory assumption with the public P0-2 eligibility contract. Execution-spec schema v2 now requires a separately SHA-pinned eligibility receipt, `topk_clt`, one selected seed, that seed's exact-cache `best.pt`, complete seed-17/29/43 run-manifest and checkpoint hashes, exact train/validation/test source-manifest hashes and requested layers. The executor calls `require_eligible_model_method` before loading and loads only through `load_eligible_topk_clt`; online/preliminary `clt.pt`, an ineligible model-method pair, an unauthorized layer, seed-map mismatch or checkpoint hash drift aborts.

Generation receipt schema v2 records the full P0-2 binding and distinguishes real production completion from injected CPU test fixtures. Its public downstream verifier rechecks the immutable freeze, colocated generation output, execution specification and source hashes, then reruns the P0-2 gate and rehashes the selected `best.pt`. Script 60's real analysis stage now requires the separately pinned v2 execution receipt. The scorer receipt must bind that same receipt SHA-256 in addition to the generation, score and endpoint-spec artifacts; synthetic analysis must leave the execution-receipt binding null.

Focused CPU verification covered eligible loader propagation, fixture/real receipt separation, online `clt.pt` rejection, downstream receipt verification, selected-checkpoint drift and exact scorer-receipt binding. Result: 15 tests passed; focused Ruff and `py_compile` passed and both example JSON files parsed strictly. No H200 job or real steering run was performed. P0-6 remains open until P0-2 supplies an eligible exact-cache model-method pair and the frozen generation/scoring protocol is executed.

## 2026-07-17 — P0-2 BF16 frozen-inference stability amendment

The r6 online ProtGPT2 seed-43 failure was reproduced at the frozen-model activation boundary. With float16 inference, the batch containing UniRef50 records `B2KU69` and `A0A4V2PAV5` yielded non-finite CLT inputs and MLP targets from layers 24--35, while CLT parameters and optimizer state remained finite. A deterministic bfloat16 replay of the identical tokens made all 36 input and target layers finite; the maximum absolute target activation was 37,632, below the float16 finite limit used for cache storage.

The post-freeze correction changes only frozen-model inference precision and fail-closed numerical receipts. `configs/clt_training_confirmatory.yaml` now declares bfloat16 inference. `src/training/clt_trainer.py` verifies all floating frozen-model parameters against that declaration before training, checks every captured input/target tensor before float32 CLT conversion, and stores the dtype receipt in resumable trainer state. The production cache profile and `scripts/61_build_dictionary_activation_cache.py` now require/record bfloat16 inference while retaining float16 cache storage; every captured activation is checked before conversion/write, followed by the existing converted-array finiteness check. Versioned cache provenance, execution reports and completion receipts prevent pre-amendment outputs from being credited.

Disposition: all online r6 outputs are discarded and cannot be resumed or pooled. The active pre-amendment float16 exact-cache extraction and any partial files or receipts under its root are also ineligible; deployment must stop them, preserve failure evidence as needed, and restart both paths in fresh BF16-specific roots. No H200 process or file was changed while preparing this local amendment. P0-2 remains open until all amended runs and held-out gates complete.

The amended online configuration SHA-256 is `2f3ff5ae49553801a2ec07ac67a291716899253da04f3803accf8b09eee88adf`; the exact-cache profile and runner SHA-256 values are `35dc4923294de439a8707c6390ca85c290ecedced4d874672f2cf628e76ab217` and `e8d486d6b9dba5a83f3658383ca989ff01f057f500eb7e9924ea7cbfd6336bc6`. Fresh online and exact-cache output roots are respectively `dictionaries_topk_bf16_r2` and `p0_2_exact_cache_bf16_r2`; the launchers refuse existing outputs. Forty-seven focused cache/trainer/gate tests passed. Focused Ruff (with the trainer's pre-existing E741 exclusions), `py_compile`, strict JSON parsing and shell syntax checks passed. A concurrent full-suite snapshot had 148 passes plus 6 subtests and 11 unrelated P0-5/P0-8 integration failures; those are assigned to their owning revisions and are not counted as BF16 validation.

## 2026-07-17 — P0-7 prospective unexposed positive control

Implemented the prospective replacement for the nonconfirmatory rotated-array and exposed-`<lr0>/<lr1>` controls. The v2 benchmark uses a canonical-amino-acid causal grammar whose future W/Y endpoint depends on equality of residues at positions 8 and 32; neither a label nor a query/control token is visible at the prediction site. It trains three small seed-rotated autoregressive models and the repository ReLU--TopK CLT, restricts selection to train/discovery data, opens the assessment cohort only after selection and control matching, and reports complete 0.5/1.0/2.0 dose sweeps at the target, wrong layer, wrong position and two active orthogonal nuisance paths. Hard matching covers layer, firing frequency, mean activation, unit decoder norm, general direct-logit norm, received attention and reconstruction contribution; matched and wrong-site effects require TOST equivalence inside +/-0.50 W/Y log-odds.

Development did not relax any gate or caliper. The initial 32-coordinate/TopK-4 fixture (9101/9201/9301) failed per-seed sensitivity and FDR/alignment. A nuisance-control redesign solved the matching failure, but a 12-coordinate fixture (11101/11201/11301) still failed seed-29 sensitivity because a rare decoder row was incorrectly counted as another planted mechanism. The corrected one-mechanism-per-seed sensitivity estimand retains coordinate-level specificity/FDR. Fresh fixtures 12101/12201/12301 and, after the complete dose-site expansion, 13101/13201/13301 passed all unchanged gates. The final fixture had per-seed sensitivity/specificity/FDR 1/1/0, endpoint accuracy 1, layer-1 FVU 0.037/0.090/0.073, all negative equivalence checks and alignment 0.987/0.976/0.970. These fixtures remain development evidence only.

After 16 focused legacy-plus-new tests, Ruff, `py_compile`, strict JSON and diff checks passed, the production spec was frozen and executed exactly once on CPU:

```text
python scripts/69_run_prospective_positive_control.py freeze \
  --spec configs/p0_7_prospective_positive_control_spec.json \
  --spec-sha256 6adb9377d732c0f126191767f1573a62850e08cea04e63e45090cee1c9829167 \
  --out-dir results/npj_revision_20260717/p0_7_prospective_v2_freeze
python scripts/69_run_prospective_positive_control.py execute \
  --frozen-dir results/npj_revision_20260717/p0_7_prospective_v2_freeze \
  --freeze-manifest-sha256 c8ece12dfecf1eefdafb2436ad39220d0c3c5f11e2f7a331fb712e956f0774e3 \
  --out-dir results/npj_revision_20260717/p0_7_prospective_v2_result
```

The exclusive, unedited production outcome is `prospective_synthetic_gate_passed`: all three seeds had sensitivity/specificity/FDR 1/1/0, endpoint accuracy 1, localized paths, all negative equivalence checks and dose recovery numerically equal to 1. Mean CLT FVU was 0.017/0.297/0.065, layer-1 FVU was 0.029/0.082/0.041 and assessment alignment was 0.935/0.998/0.946. Receipts: freeze ID `cc95047bde5eb125795418a9c4253ba1194dda172c217c33cff997e3d295e8e3`, execution claim SHA-256 `32e2e12156fb733414d3ee0ab556a6d131f979cd8240d03ce1161eee04ab6e93`, summary SHA-256 `d952a75dc2785d305bd245c6e79b0e7fc94122c475af3ecadd08d3411962dbea` and run-manifest SHA-256 `930ad2d4fbcbee63568cd52a9c4209b2270727c6b9b8fd446a2d0044450ef039`.

This passes only the planted synthetic pipeline-sensitivity portion of P0-7. The fixed comparator is not a learned biological circuit, no pretrained-model causal inference follows, the legacy controls remain nonconfirmatory and the real held-out pretrained target/control intervention gate remains open. No H200 resource was used.

## 2026-07-17 — P0-5/P0-8 fail-closed analysis contracts

Closed the remaining static integrity gaps in the confirmatory P0-5 and P0-8 paths. Script 59 now requires external SHA-256 pins for the production extractor receipt, natural cohort, measurement table, discovery cohort and equivalence specification. It rehashes the complete extractor artifact inventory, consumes the exact target/control membership in `feature_matches.json`, derives the control count from that frozen contract and rejects any rematching, postselection or membership drift. New output directories publish through a private sibling stage and are removed completely on failure.

The P0-8 builder and runner now preserve reconstruction error and intervention efficacy separately for every dictionary seed and layer. The NPZ, intervention receipt/freeze, runner and input-receipt schemas were versioned accordingly; missing or additional seed/layer cells fail. Quality--probe, efficacy--probe and quality--efficacy relationships are estimated per cell, without averaging over seeds or layers first. Production builder schema v2 and the real analyzer both enforce the enlarged decoder-native minimum of 480 annotated rows; the old eight-row floor remains test-fixture-only.

Validation used the `ct` environment and no H200 resource:

```text
pytest -q tests/test_n_terminal_counterfactuals.py \
  tests/test_nested_recoverability.py \
  tests/test_recoverability_input_builder.py
34 passed, 2 subtests passed

pytest -q tests
165 passed, 6 subtests passed
```

Focused Ruff, `py_compile` and strict JSON parsing also passed. These are contract and CPU-fixture results only. No pretrained-model P0-5 measurement, production P0-8 cache or scientific gate was run or upgraded.

## EXP-R2-026: BF16 production restart and upstream-revision audit (2026-07-17 CST)

### Model-tree provenance recovery

Audited the complete deployed pretrained-model directories rather than assigning one revision from a branch name:

- ProtGPT2: all ten Hugging Face metadata files name full revision `f71aa6cf063ad784ebd53881d11332fd098eaa58`; ordinary-file Git object identities and the LFS weight SHA-256 all match the deployed bytes. The complete deployed model/tokenizer tree is therefore verified to that exact upstream commit.
- ZymCTRL: `3c532ef172b9cd2e95238baadf5167ebb89fbc32` is the best-supported snapshot and the 2,879,060,617-byte weight is cryptographically verified (SHA-256 `7497973d9b5950dee3d2a97e150fd959098981622f560aea74f1a647fad7e94a`). Strict whole-tree proof remains incomplete because independent upstream object identities were not recoverable for every ordinary file.
- ProGen2-medium: no one upstream commit matches the deployed tree. The local `config.json` combines a later `_name_or_path` with the earlier unqualified `auto_map`, a combination absent from the six-commit history. The deployed 3,059,237,976-byte weight SHA-256 is `93ed2f0077b9b6c6aff0408bc13bf374b2897dbf31d926191a51e9803ea51215`, but exact whole-tree release requires depositing the hybrid local snapshot.

Machine-readable evidence is under `evidence/upstream_model_revision_recovery_20260717/`. The ProtGPT2 manifest SHA-256 is `313a234ddbc2255439a181417443c777a43efb72e050dd5de9576112a31db01a`. P0-1 remains open: model-tree auditing is not a DOI-backed deposit and does not replace missing raw generations/interventions, release metadata or human approval.

### Exact-cache amendment preflight

Two non-production deployment attempts exposed software defects before any eligible cache was written. The r5 minimal payload imported unrelated revision modules through an eager package initializer; the initializer was reduced to a minimal package declaration. The r6 bounded preflight then showed that NumPy cannot directly serialize CPU bfloat16 tensors; the writer now performs the explicit, finite-checked bfloat16 -> CPU float32 -> declared float16-cache conversion. Both failed preflights were retained as diagnostics and are ineligible.

The r7 bounded ProtGPT2 preflight then passed on GPU 3 in 9.65 seconds, with all floating model parameters observed as bfloat16, all captured activations finite, exactly two selected rows per split and float16 cache storage. It is explicitly production-ineligible and forbidden as a production cache. Receipt hashes:

- r7 payload archive: `d7ec3acd379553958e98905b26556cb34fab2d0a89c796e0c3f6dd01214ffbc1`;
- deployed tree manifest: `80306477c8d677b329a3a18c714021b45f0808a32e59e29a0454ec8ab27dd358`;
- preflight report: `6193de30ccccf7157541303be4cde289b2749acec7ef6e3286ee657879dfd4ef`;
- cache manifest: `62945d242d81f482e5c1238fc7e9ed278e97adebc598f2b09044653855e2d222`;
- cache content: `544610bf614e3b68b3894d42bfe62b0e32d20aac9ac3b55a3e1bf153cb1a5a1a`.

All GPUs returned to 0 MiB after the bounded preflight.

### Clean BF16 H200 production launch

Resource checks immediately before launch reported four allocated H200 GPUs (143,771 MiB each), approximately 1.8 TiB available host memory and sufficient storage. No H200 resource problem required user action.

At 2026-07-17 19:26:12 CST, fresh archive-bound TopK queues started from `dictionaries_topk_bf16_r2`:

| Seed | GPU | Pod PID | Verified early state |
|---:|---:|---:|---|
| 17 | 0 | 13694 | bfloat16 verified; finite through step 1,000; 0.24 s/step |
| 29 | 1 | 13695 | bfloat16 verified; finite through step 1,000; 0.24 s/step |
| 43 | 2 | 13696 | bfloat16 verified; finite through step 1,000; 0.24 s/step |

Each GPU used approximately 53.4 GiB. The immutable online payload archive SHA-256 is `67f02eb8c09e07303b005318da3bd7146a461fec6536e95cbe40daaaabd273ea`, tree-manifest SHA-256 is `7aa9012d7acd7e6ffeb054a61a7e1ab717336484dc8c71f9b1996eed2940b905` and launcher SHA-256 is `58d1114b46c451f8263f6c5b5da11a95dc9d52f716eae0e83f19aa5f7d16f3e9`. The measured ProtGPT2 phase estimate is 13--14 hours; allowing for the two subsequent models, checkpoints and held-out evaluation, the provisional queue estimate is 48--60 hours.

All three runs atomically published and resumed from their first complete step-5,000 checkpoints while remaining finite. Each checkpoint is about 35 GiB and includes the CLT, optimizer, scheduler, exact trainer/RNG/data-cursor state, configuration and checkpoint manifest. The step-5,000 checkpoint-manifest SHA-256 values are seed 17 `8b3dd12090cc74c809e5c946ac0381e2405e08333a177677637b0a2ba14051f7`, seed 29 `2d82ff531636549fa907e2634ef77d4d6dd7e9ddf35c196acacea0a2a56f5fc9` and seed 43 `09fa7fea0c18fa859ba3702d1ea77179b859544ed992cb2a260db9c8910375a4`. These resumable partial checkpoints are operational safeguards, not final dictionary artifacts or quality-gate evidence.

The first eligible dead-feature event occurred at step 7,500. Each seed reported exactly 14,724 resampled layer-feature coordinates, 5.0% of the 294,912-coordinate panel, rather than replacing the full interim dead set. All three runs remained finite through step 7,600; the displayed interim dead fraction fell from approximately 0.94 before the event to approximately 0.89 afterward. This verifies the live cap behavior only. The high interim dead fraction is not hidden or interpreted prospectively; final held-out quality must be adjudicated against the frozen P0-2 gate after all training completes.

At 19:29 CST, production ProtGPT2 exact-cache extraction started on GPU 3 as pod PID 15141 under fresh root `p0_2_exact_cache_bf16_r2`. Watcher PID 15142 is bound to launcher SHA-256 `cfd54015fdcb1e1cfa51bfe568d9138e4987db86c8234953d7cab77f0ead7393` and will launch ZymCTRL and ProGen2-medium only after a verified ProtGPT2 completion receipt. The runner and profile SHA-256 values are respectively `e8d486d6b9dba5a83f3658383ca989ff01f057f500eb7e9924ea7cbfd6336bc6` and `35dc4923294de439a8707c6390ca85c290ecedced4d874672f2cf628e76ab217`.

Launch, liveness and preflight success do not pass P0-2. The exact cache, alternative-dictionary screening/full runs and frozen held-out quality adjudication must all complete before any downstream biological analysis can consume these artifacts.

### Exact-cache complete-code receipt amendment

A read-only post-launch audit found that the r7 completion schema pinned the runner and model loader but not every imported local revision module. PID 15141 and watcher 15142 were stopped cleanly before completion; no completion receipt existed and GPU 3 returned to 0 MiB. The incomplete root was preserved as `p0_2_exact_cache_bf16_r2_ineligible_unbound_code_receipt`, with the same explicit suffix applied to its logs. It must not be resumed, renamed back or credited toward capacity/scientific gates.

The r8 runner verifies a complete code archive, the byte-identical deployed `CODE_CONTENT_SHA256SUMS`, exact archive and deployed-tree inventories and every listed file hash before importing any local `src` module. It rejects archive links, non-regular/duplicate/escaping members and any missing, extra or modified deployed file; local bytecode writes are disabled after verification. The v3 activation-provenance, execution-report, preflight-report and completion-receipt schemas carry the archive SHA, manifest SHA and exact-inventory flag. One launcher now owns the full ProtGPT2 -> ZymCTRL -> ProGen2-medium sequence and records GPU/host state before and after each model. The expanded focused suite passed 30 tests; the complete suite passed 172 tests plus 6 subtests before the final one-line bytecode guard and model-specific launcher labels, after which the 30 focused tests, Ruff, `py_compile` and `bash -n` passed again.

The deterministic r8 package contains exactly eight manifested code/config files plus the manifest. Deployment hashes are:

- archive: `eb646dffac5b2bfe151af47e36b53ad2233c1204af69a819529897c5a257ebe9`;
- code-content manifest: `3098a61a1b1617d023fa1badd92d28727f121d7f2d1a217c0b9b20e06be5fad6`;
- production profile: `eb33d6e8fdf551b60b95238766fcf97e3e2fe5a91f0f5882dd9212d129572db2`;
- cache runner: `1e3ffdcb2b721bc0d119dc15bdf1804b3cac43bc98bf2174caacd882d59a801e`;
- full-panel launcher: `c87a45e3454b687c15853e6d90b2958cb263e7ef6c6c385b0aa70295864af5d1`.

The archive and launcher were sent through the checksum-verified master GPFS handoff and explicitly copied into pod `damoxing-zhk-zipbio-master-0`. The pod independently rechecked the archive/manifest hashes, all eight content hashes, the exact inventory and absence of links before atomically publishing the pristine `code_p0_2_exact_cache_bf16_r8` tree.

At 21:11:06 CST a bounded r8 ProtGPT2 preflight started on GPU 3 and completed at 21:11:21. Its v3 report records all floating model parameters as bfloat16, all captured CLT inputs/MLP outputs finite before storage conversion, float16 cache storage and exactly two selected valid rows in each split. It is marked `p0_2_eligible=false` and `production_cache_reuse_forbidden=true`. Wall time was 12.83 seconds; peak allocated/reserved accelerator memory was 1,935,082,496/2,199,912,448 bytes. Hashes are:

- preflight report: `3cae6fa190cd11408a1eb98c835c943fc5d806334ed56d8bc57c6d68474ed9a4`;
- cache manifest: `6fd2b5ffa00ba62b41c3a8e87a58302a26aaeca538d73fa8eeec21304c93aff9`;
- cache content: `5bc3250b378b3a5a3e7c6cc277127c90180296a3a8c3106c75484b6ee3879431`.

GPU 3 again returned to 0 MiB. At 21:12:34 CST the production r3 full-panel queue started from a fresh root as launcher PID 16502 and ProtGPT2 runner PID
16522. GPUs 0--2 remained occupied by the three finite online TopK queues; GPU 3 was free before launch, host memory had about 1.8 TiB available and GPFS had about 46 TB free. The environment receipt captured during the run binds the five immutable run hashes, CPython 3.12.7, PyTorch 2.7.1+cu128/CUDA 12.8, NumPy 2.1.2, Transformers 4.52.4, Tokenizers 0.21.1, Safetensors 0.5.3, all four H200 driver rows and the complete sorted `pip freeze`. Environment-receipt SHA-256 is `3dce112700e17ec0bda8534527f731f67bf31b0946f9408d19075759ea6e17dc`; the freeze SHA-256 is `02d5b4cfd98079188bfdf7d20382d5e5d62f6d44412e92d59129f7d45eefbafa`.

The 48--60-hour estimate above applies only to the preliminary online TopK queues. The separate frozen alternative-dictionary plan budgets 2,125 aggregate GPU-hours for screening/full controls, excluding cache extraction and evaluation (about 22 idealized days at four GPUs, longer in calendar time). Neither queue launch nor environment capture changes a scientific gate.

All three online ProtGPT2 seeds subsequently published and resumed past their atomic step-25,000 checkpoints while remaining finite through at least step 28,250/28,400/28,400. The step-25,000 checkpoint-manifest SHA-256 values are seed 17 `aab1f5215277631c56187d2d7f2eeeb2acb25b57b4ca14537d789323119ad378`, seed 29 `5519a3dc11ab2ec8ffc87653b6ebb9c73b3d7ce518b864dfe28a7c77f9e154cc` and seed 43 `485e856c6acb681bff63e0d14151bea9ba22e82202d5531b330fb906f1c0e3a8`. The live displayed dead fractions were approximately 0.68--0.70 and remain interim training diagnostics, not a held-out quality result.

### Downstream bfloat16/finiteness contract amendment

A cross-path audit found that the P0-3/4/5/6/8 production examples still declared float16 inference even though ProtGPT2 float16 had already produced non-finite late-layer activations and P0-2 had moved to bfloat16. All production and confirmatory consumers now require bfloat16, verify every floating model parameter as exactly bfloat16 before the first activation, and check every required captured tensor for finiteness before conversion or downstream use. P0-3/4/8 check all required CLT-input and MLP-output captures; P0-5 also checks attentions and baseline/intervened logits; P0-6 checks intervention-site tensors and every generation-step logit before sampling.

The standardized receipt fields are declared dtype, observed parameter dtypes, verification method, dtype-verification result, finiteness-check definition and finiteness-verification result. Exact consumers reject missing, false, stale or tampered values. Production schemas were versioned rather than silently changing semantics: P0-3/4 input products use v3, the P0-5 extractor spec uses v3 with v4 summaries/manifests, the P0-6 executor spec/receipt use v3 with a v2 summary, and the P0-8 input receipt uses v3. Explicit nonproduction fixtures retain honest float32 support but cannot enter a production consumer.

The integrated focused suite passed 73 tests plus 2 subtests; the complete R2 suite passed 182 tests plus 6 subtests. Ruff, byte compilation, strict JSON and stale-schema/float16-production scans passed. This was CPU fake-model and adversarial contract validation only; it did not run a pretrained model, alter an H200 job or pass a scientific gate.

## EXP-R2-027: Pretrained P0-7 adjudicator and revision verification (2026-07-17 CST)

**Purpose.** Complete the remaining inference-free P0-7 analysis contract and verify the revised code, processed package, figures, manuscript and live H200 lineage without crediting unrun scientific gates.

### Minimal P0-7 implementation

One module, one thin CLI and one example specification were added:

- `src/revision/causal_adjudication.py`, SHA-256 `696b7e3ace1e6eaacb9dc14691064abb77f673ad9cecc089c9d7c7c9231cfb19`;
- `scripts/71_adjudicate_pretrained_causal_interventions.py`, SHA-256 `7478a9787b60258004d479e6716a64c6756ad9e3bfa4366c64ad43d6658cc41a`;
- `configs/p0_7_pretrained_causal_adjudication_spec.example.json`, SHA-256 `271e71d35fd38340edf496577db041fbca378c4b06d95f2035b2c532abcf5915`.

The module performs no model inference and no rematching. It rehashes the complete prospective synthetic prerequisite, eligible P0-2 run manifests and checkpoints, P0-5/P0-6 receipts and their colocated artifacts, frozen disjoint cohorts/feature identities, all raw intervention and external-score rows and their model/dictionary/scorer/calibration/code bindings. It requires the exact evaluation x identity x target/control x site x strength inventory; preserves every paired row before inference; applies one global Benjamini--Hochberg family over positive and both TOST component p-values; and requires intended feature fidelity, resolved off-target/reconstruction displacement, logit and behavior results, and on-path/off-path localization. Inputs, source and outputs are rehashed again before atomic publication. Missing, stale, substituted, non-finite, prematurely averaged or inconclusive evidence cannot produce a resolved scientific status.

Focused adversarial verification passed 33 tests. The final complete R2 suite passed 215 tests plus 6 subtests in 40.20 seconds. Ruff check and format, byte compilation, CLI loading and strict example-JSON parsing passed. The focused test and protocol SHA-256 values are `ba6a67f5fb1d987a7f3a9d1855b3d550212a67f4e68605e2203faaa0c56470bc` and `73adfe7371f5e99cd930c28eeba49038e838d7fd0eaae4f617b8a06d2616e30b`. No eligible real P0-7 surface or completion receipt exists; the real gate remains open and failed by default.

### H200 read-only progress audit

The refreshed `~/hangzhou-remote` pod helper was used with explicit pod `damoxing-zhk-zipbio-master-0`. At 22:14 CST the canonical GPFS logs were finite through steps 39,550/39,700/39,750 for TopK seeds 17/29/43. The latest complete step-35,000 checkpoint-manifest SHA-256 values were `13ae8fb16dea3694951ca8e53e7eaef6044cd8e20b5378edc1208854f43a51af`, `ad3dd29150c49b6cb5bc161de7b7908defac8869d47f17edb895dfce3e2946e4` and `2369c1c288273f67af89f157bb38be4012c18f9faf0b61a5752007c592b64a9e`. Older OSS queue logs were identified as superseded artifacts outside this production lineage; their rows are not used to assess the active jobs.

The r8 ProtGPT2 exact-cache PID remained alive in GPFS-heavy extraction with 173 GiB staged and no completion receipt. GPUs 0--2 used about 54.8 GiB each, GPU 3 used about 2.8 GiB, host memory had about 1.8 TiB available and GPFS had about 46 TiB free. No H200 issue required user intervention. Process liveness, partial checkpoints and staged cache bytes do not pass P0-2.

### Manuscript and processed-package verification

`build_source_data.py --verify-only` verified 65 manifest rows, 68 checksums and 1,649,686 package bytes. The source manifest, checksum file and historical panel/table map SHA-256 values were `02a840cf0eda5c2b89fd6000b7377013d2218e53593c7385b9fcd11341356a05`, `7e099126c7991c6704c30e140c090f643c55079148fc030fc81ab5d3ea7522e8` and `86ce0d9f0bd0dbf79d8a78f270b6d7aaae1c789fd7e8030583e4e4ba574fb8d4`. All six figures regenerated. Tectonic rebuilt the 23-page main PDF and 12-page supplementary PDF with underfull-box warnings only; their SHA-256 values are `6edd0f206aceaaa32f63e5c142f063cad83133dfb90b766926d4189c4e396e7f` and `47e0933fac9452a5eae747cb72ec0f87273b6edfeef5d0c75c06a447d837ce53`. Strict parsing passed for 58 current configuration, environment, source-data and provenance JSON files.

## EXP-R2-028: H200 progress, receipt validation and bounded storage cleanup (2026-07-18 CST)

**Purpose.** Track the active revision queues from their canonical GPFS lineage, validate the first production completion receipt and relieve storage pressure without touching any active, resumable or scientifically eligible artifact.

### Verified execution progress

The refreshed `~/hangzhou-remote` helpers were used with explicit pod `damoxing-zhk-zipbio-master-0`. At 12:04 CST, all three bfloat16 ProtGPT2 online TopK runs had completed 200,000/200,000 steps and atomically published complete final checkpoints. Final checkpoint-manifest SHA-256 values were:

- seed 17: `e1e0702090b5c556d45abe736e136f5713ba55144488556e1797d3a0ba1dd6bb`;
- seed 29: `cbbd80cee07730b725e6be171569f86ceed75b713bfe1d3254b13afe025f4cbd`;
- seed 43: `22fda6fc1f60c7fd3df238aec2ba6b57750993da29c39fe70ae63eaf52b4f272`.

Each manifest was complete, resumable and step-consistent, with all five declared files present at their declared sizes. Full checkpoint payloads were not rehashed during concurrent GPFS I/O. The three sequential queues advanced to ZymCTRL and, after cleanup at 12:09 CST, remained finite at steps 23,650/24,200/23,600 for seeds 17/29/43 at approximately 0.36 seconds per step. The apparent `inf` text scan hits were only the word `inference` in the verified-bfloat16 header; there were zero NaN, Inf, non-finite, traceback or error metric rows. The latest complete step-20,000 ZymCTRL manifest hashes were, respectively, `637810eb1f2626841e222d74f7770228b6fad28940566b505cb75abc7e05e8ac`, `a80537559b7df9da4c646f36a0e12ec66f78ce7eccbeab1470c7e6989539e229` and `d1cf2cf5baa517e079026918dbf5712093f57dfdf4663f1fe5c40be083c3b564`.

The r8 exact-cache queue completed ProtGPT2 at 07:54 CST and advanced to ZymCTRL as runner PID 17403 under launcher PID 16502. The ProtGPT2 receipt is internally consistent, records `verified_complete`, bfloat16 frozen-model inference, finite captures and 221,184,000,000 cache-payload bytes. SHA-256 values are:

- completion receipt: `c5e323df9618c14db9ec4e19332dc1b162a4a026b68381b2b1fd3c7954a73225`;
- cache manifest: `d3fa68212612bb42ed9d75c0b49a606db44a9171c00b0662c05b41f48f63dd34`;
- execution report: `9ada765572ba519c28e8fed0b5f4bbd34aceb41ed642acf8b8c215ce5304af2a`;
- cache-content identity: `ddf4ea1e99d5b382124b31b80fba1bcd99c4e84a9c48b00e28827c955c4f87dc`.

The active `.zymctrl.tmp-17403` directory is owned by PID 17403 and remains protected. GPUs 0--2 used about 55.7 GiB each at 89--100% utilization; GPU 3 used about 2.8 GiB while GPFS-bound. Host available memory was about 1.8 TiB. The current ZymCTRL phase has approximately 18 hours remaining; the established online three-model window leaves roughly 31--43 hours for ZymCTRL plus ProGen2-medium. The independent exact-cache queue has a lower-confidence 14--22-hour remaining estimate because it exposes no durable row cursor. No H200 fault or resource condition requires user intervention.

### Receipt-preserving storage cleanup

Before deletion, every candidate was checked for an absent live PID and absent eligible `completion_receipt.json`. Small logs, configurations and manifests from the first three GPFS candidates were archived at `logs/cleanup_evidence_20260718/superseded_large_artifacts_small_evidence.tar.gz` (23,329 bytes; SHA-256 `7601f41f8a859ceb599215b29d717d1d31cad08b6d3afd7a95dcc062f45b09c1`). The following apparent bytes were then removed:

- 223,519,958,526 bytes: full superseded float16 `dictionaries_topk_stable_r1` root;
- 184,683,511,863 bytes: full aborted pre-amendment `p0_2_exact_cache_r1` root;
- 37,251,935,377 bytes: `clt.pt`, `optimizer.pt`, `scheduler.pt` and `trainer_state.pt` from the four-step publication smoke; its original log, manifest and configuration remain in place;
- 184,683,511,688 bytes: inactive `.protgpt2.tmp-15141` payload from the permanently ineligible r7 unbound-code attempt; the explicitly suffixed parent root and diagnostic log remain;
- 112,920,458,240 bytes: eight `clt.pt`/`optimizer.pt` files from superseded OSS stability/resume and bounded ZymCTRL smokes; their small diagnostic evidence remains in place.

Total apparent payload removed was 743,059,375,694 bytes (692.03 GiB): 630,138,917,454 GPFS bytes and 112,920,458,240 OSS bytes. Immediate GPFS usage fell from 18% to 17%; the post-cleanup snapshot had 49,802,486,218,752 bytes available. Concurrent active writes make filesystem-level deltas unsuitable as the exact deletion total, so the total above is the sum of pre-deletion object sizes. The path-level machine-readable receipt is `evidence/h200_cleanup_20260718/cleanup_manifest.json`.

Protected roots were unchanged: `dictionaries_topk_bf16_r2`, `p0_2_exact_cache_bf16_r3`, its live `.zymctrl.tmp-17403`, current r7/r8 code trees and archives, cohorts, environment receipts, launchers and canonical logs. All four launcher PIDs and runner PID 17403 remained alive after cleanup, and all active GPU allocations were unchanged. This is operational progress and storage maintenance only; it does not pass the held-out P0-2 quality gate or any downstream biological or causal gate.

## EXP-R2-029: BF16 prerequisite completion, validation-only screening launch and bounded cleanup (2026-07-20 CST)

**Purpose.** Verify the terminal state of both H200 prerequisite queues, start the next frozen P0-2 stage only from receipt-verified exact caches, and remove redundant storage without touching any final, trajectory, cache or active screening artifact.

### Completed preliminary online TopK queues

The three seed queues completed their final ProGen2-medium member and exited normally at 23:55:54, 23:58:29 and 23:47:47 CST on 2026-07-19 for seeds 17, 29 and 43, respectively. Thus all nine model/seed runs reached step 200,000. Every final manifest is schema v2, `complete=true`, `kind=resumable`, step-consistent and backed by all five declared files at the declared sizes. Canonical logs contain no NaN/Inf, non-finite, traceback, OOM or runtime-error row.

| Model | Seed 17 manifest SHA-256 | Seed 29 manifest SHA-256 | Seed 43 manifest SHA-256 |
|---|---|---|---|
| ProtGPT2 | `e1e0702090b5c556d45abe736e136f5713ba55144488556e1797d3a0ba1dd6bb` | `cbbd80cee07730b725e6be171569f86ceed75b713bfe1d3254b13afe025f4cbd` | `22fda6fc1f60c7fd3df238aec2ba6b57750993da29c39fe70ae63eaf52b4f272` |
| ZymCTRL | `8c350b463f430bac2f61df31436506656c17563f0a821a60bbc611dbefbdc745` | `eb2a473a671b72422617d04535473aed0d13a927a30aca1253159d645077b186` | `aa359f046aac19a0be3921574f47212e4341542aa194dac3a6e9b0b62199bbcf` |
| ProGen2-medium | `c55a648aaf91d32f754896f994a150214ae197e7381ed3fc664de3674220bc3b` | `0884275af565ba45720d5795a3e3c2d3544453bf62e812efa8d5cbb9a976003e` | `1fdf39a1917134f828ab0dd83af8fd216d8323e1934d4a5a686c0c58992831fb` |

These online checkpoints remain preliminary trajectory diagnostics. They do not use the exact activation cache and cannot enter the P0-2 matched-method comparison.

### Verified exact activation-cache panel

The r8/r3 queue completed ProtGPT2 at 07:54 CST and ZymCTRL at 17:19 CST on 2026-07-18, then ProGen2-medium at 02:48 CST on 2026-07-19. All three receipts are schema v3 with status `verified_complete`; they bind bfloat16 inference, finite pre-conversion captures, float16 cache storage, the exact first-party code inventory, their cache manifests/reports and content identities.

| Model | Receipt SHA-256 | Manifest SHA-256 | Report SHA-256 | Content SHA-256 | Payload bytes |
|---|---|---|---|---|---:|
| ProtGPT2 | `c5e323df9618c14db9ec4e19332dc1b162a4a026b68381b2b1fd3c7954a73225` | `d3fa68212612bb42ed9d75c0b49a606db44a9171c00b0662c05b41f48f63dd34` | `9ada765572ba519c28e8fed0b5f4bbd34aceb41ed642acf8b8c215ce5304af2a` | `ddf4ea1e99d5b382124b31b80fba1bcd99c4e84a9c48b00e28827c955c4f87dc` | 221,184,000,000 |
| ZymCTRL | `0b4c4e9040556d73dee68421cff13cc1497e4c8348e03a4c72f9eb0bfd0fb013` | `26307ed3694c884543d3033cd99e6cee52cd5441c9ce7f37f6f9a4f1a48dbc70` | `dcf542ba645a99a8bdeb31bcd977772657206c2238fc48a5f1822307941eaf9a` | `cb91a518627e516dfe7cd66dc9fbd61afcb60526b21df392f811f7b9bc1050de` | 221,184,000,000 |
| ProGen2-medium | `31b34aef4a059b061b4979a5afa5a4b07ccc0278b37ecac3e2a69d7153171a80` | `e51c9dacfc7cbb56fb8f860800a4e2315686838853e34f1905d95fa757d72ea9` | `ea6cb88b8bbadbef0803848c5229d4bfe1c3d88983363269f89de0386d8f61a5` | `fc3d0e80016a672d6e58a9f9c6f0f92cf5394c464ca4c633929c4b98880d3166` | 199,065,600,000 |

The exact panel contains 641,433,600,000 payload bytes. Full cache roots are protected from cleanup.

### Receipt-bound validation-only screening launch

The next frozen stage is exactly six validation-only invocations: each of ProtGPT2, ZymCTRL and ProGen2-medium with `relu_l1_sae` and `gated_sae`, using screening seed `20260717`. TopK and dense controls have no coefficient grid and therefore skip screening. A single thin dispatcher was added to close the launch-time completion-receipt guard:

- launcher `scripts/72_run_dictionary_screening_queue_h200.sh`, SHA-256 `caf40fb83c63f5af7ebde4f7b4b459aa76ec9ac53a648a3992d7efe4df7f5c9c`;
- immutable code archive SHA-256 `76ba445cbdb418602df3b0088ac41f88e84935e06614cff0a6ff3a3c99ca3811`;
- embedded tree-manifest SHA-256 `3e8628ef9f8e0a0141d491beeffed5b2429ed50836f1c830cc2440095667a713`;
- runner/profile SHA-256 values `99f4224f833dfc644f9adc45b2432b4c53a81b2cd857081d3386fc77ec513d8a` and `eb33d6e8fdf551b60b95238766fcf97e3e2fe5a91f0f5882dd9212d129572db2`.

The dispatcher independently checks each receipt and manifest pin, all v3 completion fields, the receipt-referenced report and manifest hashes, cache content identity, the exact deployed code tree, output nonexistence and target GPU vacancy. It accepts only a protocol-valid terminal screening status with seed `20260717`, zero test evaluations and `p0_2_eligible=false`.

Focused verification passed 39 tests in 9.31 seconds. Ruff, `bash -n` and strict JSON parsing passed. The initial transfer was hash-verified on the H200 master view; because the pod exposed a different GPFS namespace view, the two small files were copied explicitly through Kubernetes and then rehashed and deployed atomically inside the pod. This was a handled transfer-path condition, not an H200 compute fault.

At 18:10:02 CST four launchers started on the idle GPUs:

| GPU / launcher PID | Fixed queue |
|---|---|
| 0 / 24175 | ProtGPT2 gated SAE |
| 1 / 24176 | ZymCTRL gated SAE |
| 2 / 24177 | ProGen2-medium gated SAE, then ProGen2-medium ReLU/L1 SAE |
| 3 / 24178 | ProtGPT2 ReLU/L1 SAE, then ZymCTRL ReLU/L1 SAE |

During the 18:15--18:18 CST health window, all four launchers and Python workers remained alive, all four `run_state.json` files appeared, GPU memory was approximately 54--62 GiB per device and utilization reached 60%, 98%, 19% and 100% in the final sample. No traceback, OOM, NaN/Inf, exception, killed process or no-space error was present. Exact step counters become durable at the first step-1,000 checkpoint.

The six invocations comprise 45 candidate fits and 125 aggregate GPU-hours. The honest screening estimate is approximately 2--4 calendar days on four H200s including validation and I/O. The later 36-run full panel remains 2,000 aggregate GPU-hours, or 20.8 ideal four-GPU days before evaluation/IO overhead.

### Bounded post-completion storage cleanup

Immediately before deletion, the nine step-200,000 final manifests were revalidated as complete/resumable with exact declared file sizes. A compact remote receipt preserved the nine step-195,000 manifests/configurations, schedulers/trainer states, final manifests/configurations and the obsolete smoke metadata. The following inactive artifacts were then removed:

- nine redundant step-195,000 resumable predecessors: 320,926,909,341 apparent bytes;
- obsolete OSS 20-step ProtGPT2 float16 expansion smoke: 74,503,239,589 apparent bytes.

Total removal was 395,430,148,930 apparent bytes (368.27 GiB) across 58 files. All nine step-200,000 finals, all 63 model-only snapshots at steps 25,000--175,000, all three exact caches, screening outputs, code/archive, cohorts, receipts and canonical logs remain. The local path-level receipt is `evidence/h200_cleanup_20260720/cleanup_manifest.json`, SHA-256 `f1d2e4dea103f4bf39ae125c37e4307adfd5100750efd64d9375f795401321a7`. The remote TopK and smoke metadata archives have SHA-256 values `ef12accdfedc78f220aa8372eedd9a8e3757240effacea92f3a38948fc1af539` and `951a4b01e7ad5ff7ec47c15c1a7e82b0aac84ebaee2e97d548667cf4dcb2683a`; the receipt-checksum manifest has SHA-256 `f3d0de3cd097a549d8132b5ce6eec71f7113a67233de8240a274fb03c0d41ca8`.

### Disposition

The online TopK panel is preliminary diagnostic evidence, cache completion is prerequisite completion, and the running screenings are validation-only and P0-2-ineligible. No held-out P0-2 gate or downstream biological/causal gate has passed. No H200 issue currently requires user intervention.

## EXP-R2-030: Screening OOM diagnosis, ineligible-lineage cleanup and memory-safe restart (2026-07-21 CST)

**Purpose.** Inspect the terminal state of the validation-only screening queues, diagnose their failure without opening test rows, preserve compact failure evidence, correct candidate-local accelerator-memory lifecycle, and restart the unchanged frozen screening panel only after a repeated full-width preflight.

### Corrected terminal audit of the r1 screening lineage

The four launch observations in EXP-R2-029 remain accurate for their stated 18:15--18:18 CST health window. The later terminal audit found that all four r1 queues had subsequently failed. Each active invocation completed candidates `000`--`002` and raised a CUDA out-of-memory error while constructing or training candidate `003`. Thus 12 of the 45 planned candidate fits had produced validation diagnostics, 33 had not run, and the queued ProGen2-medium and ZymCTRL ReLU/L1 invocations had never started. None of the four roots contained `results.json` or `run_manifest.json`, while their `run_state.json` files still reported `in_progress` after the processes had exited.

The failed lineage retained the EXP-R2-029 archive, profile and launcher SHA-256 values `76ba445cbdb418602df3b0088ac41f88e84935e06614cff0a6ff3a3c99ca3811`, `eb33d6e8fdf551b60b95238766fcf97e3e2fe5a91f0f5882dd9212d129572db2` and `caf40fb83c63f5af7ebde4f7b4b459aa76ec9ac53a648a3992d7efe4df7f5c9c`. The exact queue-log and residual run-state hashes were:

| GPU / started invocation | Queue-log SHA-256 | `run_state.json` SHA-256 |
|---|---|---|
| 0 / ProtGPT2 gated SAE | `578b263574bb27eebc498383add90e636c1d53272d94de47e30dcad402d6a04f` | `12df705291599d8239e2da40e6d736ec7d4de4a7f8c3491589864c91235d808a` |
| 1 / ZymCTRL gated SAE | `4f8d33303eaa37e5a91517fae0ee6afe02ca5b410ee8750751cf1aa1ea1141f0` | `5ff2a617967d802ad355883f35d69118d16ec8b12504c0f24618dbb22a7e3bba` |
| 2 / ProGen2-medium gated SAE | `05fc455e76f257b763fe294ea4bf7c9a712bdc9dc83d29722c454a81eea6cc18` | `1873f168673f21dd3c43226a8dbc517bb8fd7bb78102a2fb6b9773eab0346362` |
| 3 / ProtGPT2 ReLU/L1 SAE | `4259c8fb20a69c3976bce7b0ebe42cc965e62f383f90a0f8ed80dd424f42fcdc` | `09decf059396f75b19747b95bd288f54f5ab6ef2cff7378090ac6a9c8a516149` |

### Exact memory-lifecycle diagnosis

Per-candidate peak allocation increased by one prior candidate's complete trainable-model and Adam payload. For ProGen2-medium gated SAE the first three peaks were 54,274,764,800, 86,747,138,048 and 119,219,511,296 bytes, an exact 32,472,373,248-byte increment per candidate. A 36-layer gated path reported 62,259,861,504, 99,517,368,320 and 136,775,137,280 bytes, an exact 37,257,506,816-byte increment per candidate. ReLU/L1 followed the same candidate-wise persistence pattern. Each increment equals 12 bytes per trainable parameter, localizing the failure to optimizer/model state retained between candidates in the same Python process.

At inspection, all four H200 GPUs were idle at 0 MiB and 0% utilization, host memory had approximately 1.3 TiB available and GPFS had approximately 45 TiB free. The failure was an application CUDA-object lifecycle defect, not an H200, host-memory or storage-resource fault, and no user intervention was required.

### Permanently ineligible quarantine and bounded cleanup

The old output root was renamed `p0_2_dictionary_controls_bf16_r1_ineligible_candidate_memory_accumulation_20260721`. Because the patched runner and module must be bound uniformly into a final run manifest, none of its 12 completed candidates can be resumed, pooled or credited. A fresh eligible lineage must execute all 45 candidates.

After matching every retained best-checkpoint hash to its validation result, the cleanup removed:

- 12 progress checkpoints totalling 432,716,642,868 apparent bytes;
- 12 best checkpoints totalling 144,235,215,846 apparent bytes; and
- 576,951,858,714 apparent bytes in total (537.328 GiB), corresponding to 576,952,172,544 allocated bytes.

The cleanup retained 28 compact files--12 validation results, 12 timing sidecars and four run states--totalling 1,787,747 bytes. Remote evidence is under `logs/cleanup_20260721_screening_oom_r1`; its receipt-checksum manifest SHA-256 is `6524241351dcf7c499c21cbffdd91d3f922ffe700edd365724e0647e8d19f706`. The local path-level receipt is `evidence/h200_cleanup_20260721/cleanup_manifest.json`, SHA-256 `069714b1edc7d961cfe3353f4e3c9249f60255dbad7c41ab5167e48ff59ef76d`.

### Minimal candidate-lifecycle amendment and verification

After a successful final checkpoint and peak capture, training now clears model gradients and optimizer state before deleting the optimizer. The thin runner then deletes the candidate's checkpoint/model objects, runs Python garbage collection, synchronizes the target device and empties the CUDA cache. It records post-candidate allocated/reserved memory and fails closed unless allocated memory is stable across later candidates and no greater than 128 MiB. The nonzero allowance is necessary because a deliberately strict first preflight found a stable 67,108,864-byte PyTorch allocator baseline after the first candidate; that exact-zero lineage remains a diagnostic and was not promoted.

Final local SHA-256 values are:

- dictionary module: `347a095c2e18a429e09011f84a45bd40b1cf5d46ebf38398e6e2a7afe57c6596`;
- screening/full runner: `56ca3c4d8e230ea6ef5cf36d394564f747fa2c5bcb4f9b837dd3e5825e6401b8`;
- repeated lifecycle-preflight runner: `8e5ee2ada79ad67ac869eb34bb36594e3188179bff6c6c35b404007a37813a94`.

The complete R2 suite passed 216 tests plus 6 subtests in 35.48 seconds. Ruff check and format, byte compilation and launcher shell syntax were clean.

### Repeated full-width preflight and immutable r3 restart

The r3 package binds archive SHA-256 `6f191b554de901c7be25968f2fc96d989ae7d5d0bcb2a1c9285fdd1c3b840e44`, embedded/deployed tree-manifest SHA-256 `486ba4e9c00a7a1a88a58a31691095e72aaf45f39745ad3117b88c694de579d4`, the runner/module hashes above and production-profile SHA-256 `eb33d6e8fdf551b60b95238766fcf97e3e2fe5a91f0f5882dd9212d129572db2`.

At 14:20:58 CST, the nonconfirmatory four-repetition ProtGPT2 gated-SAE preflight completed successfully. Allocated bytes before each repetition were 0, 67,108,864, 67,108,864 and 67,108,864; allocated bytes after every repetition were exactly 67,108,864. Each repetition reached the identical 62,162,163,712-byte peak allocation and 62,182,653,952-byte peak reservation. It made zero test accesses, removed its model/checkpoint payloads and retained only the report. The report and log SHA-256 values are `10d2cda01547e48a5dbaf5034e057bda48f4578f130dc64db24bb351e6bdcf33` and `927601deaa09ae544fbc6c8c4489e94fe2eb10a87c26d130fe224a499d851fd8`.

Launcher `scripts/75_run_dictionary_screening_queue_h200.sh`, SHA-256 `bc511f78f872ce70cc85ed098ec9aad15e44aa2dc1a5add3de2a52683fcef2f6`, started four fresh queues at 14:22:06 CST under `p0_2_dictionary_controls_bf16_r3/screening`:

| GPU / launcher PID | Fixed queue |
|---|---|
| 0 / 27369 | ProtGPT2 gated SAE |
| 1 / 27370 | ZymCTRL gated SAE |
| 2 / 27371 | ProGen2-medium gated SAE, then ProGen2-medium ReLU/L1 SAE |
| 3 / 27372 | ProtGPT2 ReLU/L1 SAE, then ZymCTRL ReLU/L1 SAE |

Initial child Python PIDs were alive after launch. The observation-based estimate for this fresh screening pass is 16--24 hours; the frozen conservative planning budget remains 125 aggregate GPU-hours. The queues retain seed `20260717`, validation-only selection, zero test access and `p0_2_eligible=false`.

### Local environment maintenance and disposition

The unused repository-local `.venv` occupied 389,023,458 apparent bytes. It used Python 3.13, contained plotting tools but no PyTorch, and was not held by any process. It was deleted; the canonical verification and training environment is now `source ~/miniconda3/etc/profile.d/conda.sh && conda activate ct`. Historical commands remain unchanged as provenance.

The r3 launch and preflight establish only a memory-safe validation-screening lineage. No held-out test evaluation, P0-2 gate or downstream biological or causal gate has passed. Full exact-cache runs remain blocked on six terminal screening products and their frozen selections.

### Final adjudication-contract closure

The fail-closed panel adjudicator now accepts an exact terminal `sparsity_match_failure` only as a complete calibrated negative. That state requires the full frozen candidate grid, zero held-out test access, no L0-matched row and no selected configuration or checkpoint. It emits `atlas_eligible=false`, dynamically removes that model/method's three seeds from the expected full-run identities, forbids any supplied full run for the failed pair, and remains unusable through the downstream eligibility consumer. All other malformed, incomplete or contradictory screening states still abort. The formatted gate and gate-test SHA-256 values are `2b7bcd627dd80c3d140588451cf767982bbaf21015f68a34e907ed05ab75bc6c` and `ca96c89e6627d031e82e1d0fe1346b38eae152f5a10e5a2e35332caf4766d5a6`.

A minimal shared producer and thin CLI now write the independently required mask-validation receipt:

- `src/revision/mask_validation_receipt.py`, SHA-256 `de9324909af75014a0bf76420abf5a1904047891b12c5da595dc9c96836180c0`;
- `scripts/76_write_mask_validation_receipt.py`, SHA-256 `287844fa0e88061aa167cce1fa1233428c3289cd26123f76463c9eb88d17ce68`; and
- `evidence/p0_2_mask_validation_20260721/mask_validation_receipt.json`, SHA-256 `5966e274881984b2eeabeedd749d94c313fce02fb814911f0d1082ac3c3232db`.

The durable receipt binds dictionary-module SHA-256 `347a095c2e18a429e09011f84a45bd40b1cf5d46ebf38398e6e2a7afe57c6596` and test-file SHA-256 `a1f2f61f8933c1e1a766bc5e827ba3747215c2286a2df2421e32d0e2b8fc1253`. Both exact required nodes, `test_valid_token_cache_is_padding_invariant_and_all_valid_equivalent` and `test_windowed_metrics_are_padding_invariant_end_to_end`, passed, and the production gate consumer validated the receipt.

The 216-test plus 6-subtest result above remains the lifecycle-only snapshot. After the gate and receipt closure, the final integrated pre-documentation suite passed 223 tests plus 6 subtests in 36.28 seconds; focused gate/mask verification passed 16 tests. Ruff check and format, AST syntax, shell syntax and strict JSON validation passed. These are adjudication-contract and mask prerequisite facts only: they perform no held-out test-set evaluation, do not pass P0-2 and do not authorize any full run before its terminal screening outcome is validated.

## EXP-R2-031: Terminal r3 screening, calibrated panel selection, cleanup and full launch (2026-07-22 CST)

**Purpose.** Close the receipt-bound, validation-only r3 screening phase; apply its frozen calibrated sparsity rule without accessing test rows; retain only the selected or compact evidence; and launch precisely the authorized full exact-cache panel.

### Terminal validation-only screening and panel decision

The r3 screen completed all 45/45 frozen candidates with zero test accesses. Three model/method pairs selected the following validation configurations (all have threshold `0`):

| Model / method | Candidate | L1 | Auxiliary | L0 | FVU |
|---|---:|---:|---:|---:|---:|
| ProGen2-medium gated SAE | 0 | `1e-5` | `0.1` | 132.16376074074074 | 0.4397525937186076 |
| ZymCTRL gated SAE | 3 | `3e-5` | `1` | 120.02304111111111 | 0.5929956473037449 |
| ZymCTRL ReLU/L1 SAE | 0 | `1e-5` | `0` | 131.2233588888889 | 0.5899043000429988 |

Three other pairs reached the exact, terminal calibrated-negative state `sparsity_match_failure`: ProGen2-medium ReLU/L1 SAE had zero eligible rows (closest L0 109.82516444444444); ProtGPT2 gated SAE had minimum/closest L0 1424.622846388889; and ProtGPT2 ReLU/L1 SAE had minimum/closest L0 1298.928968888889. Per the frozen gate, these are dictionary-quality negatives, not biological claims, and their three seed runs are forbidden.

The resulting full panel is exactly 27 runs: ProtGPT2 TopK and dense (three seeds each), ZymCTRL TopK/ReLU-L1/gated/dense (three each), and ProGen2-medium TopK/gated/dense (three each). The nine forbidden runs are the three seeds for each failed pair. The terminal screening receipt is `evidence/p0_2_screening_20260722/terminal_receipt.json`, SHA-256 `83bf4563a27886a1e1f148eb4094f9a7b02533936a02981caac17a805de04d6c`.

### Receipt-preserving cleanup

After terminal-state and selection guards, cleanup removed 45 redundant `progress.pt` files and 42 nonselected `best.pt` files (87 files total): 2,103,965,204,225 apparent bytes and 2,103,966,318,592 allocated bytes (approximately 1.9135 TiB). It retained the three selected `best.pt` files and 108 compact metadata/evidence files. The screening root then contained 35,673,800,024 apparent bytes and 35,674,349,568 allocated bytes. GPFS had 48,390,993,870,848 bytes free (19%).

The local receipt is `evidence/h200_cleanup_20260722/cleanup_manifest.json`, SHA-256 `f646ad90abe8554c70357860285dda4051147fc0d6d6e95c0887f97fe62befe3`; the remote receipt manifest SHA-256 is `607d7ae1420ad68a44c7a3e283f3f0c56fc55419d67fb83a99b57f57b06cef47`.

### Authorized full exact-cache execution

Full r1 queues started fresh at approximately 11:46 CST on 2026-07-22, on GPUs 0--3 as launcher PIDs 30587--30590, under `p0_2_dictionary_controls_bf16_r3/full`. Launcher `scripts/77_run_dictionary_full_queue_h200.sh` (SHA-256 `20eadefc7eb8102e892efb5592e7f71b8763ca41fd3dd052a725cf4170849d72`) uses fail-closed exact actual checkpoint hashes and fixed queues of 7/7/7/6 runs. Before launch it revalidates the exact cache and terminal screening lineage, including each retained selected checkpoint. After each runner returns, it verifies the full result, manifest, run state, actual selected checkpoint and `test_evaluation_count=1`; only then does it delete and log the redundant `progress.pt`. The initial launch receipt is `evidence/p0_2_full_launch_20260722/launch_receipt.json`, SHA-256 `306d2b56bf3cf756fa5c56dd86e0f24d3555fc6c49c02d59747216e31d3104b9`.

By 12:45 CST, all four first runs had exact `in_progress` states and had atomically published both `best.pt` and step-5,000 `progress.pt` checkpoints. The four timing sidecars reported 1,931--2,013 seconds through step 5,000. GPU memory was 60,793--60,825 MiB with active utilization; the queue logs had no traceback, OOM, non-finite or no-space signal. GPFS retained 48,192,404,062,208 bytes free at 20% use.

The terminal effective screening rates were 2.268 steps/s (ProGen2-medium gated), 2.651 (ProGen2-medium ReLU/L1), 1.865 (ProtGPT2 gated), 2.200 (ProtGPT2 ReLU/L1), 1.910 (ZymCTRL gated) and 2.189 (ZymCTRL ReLU/L1). Together with the prior online TopK rate of about 0.24 s/step, queue 3 is likely critical at about 4.8 days before margin. The frozen full budget is 1,500 aggregate GPU-hours; its conservative longest-queue ceiling is about 388.9 hours (16.2 days), excluding evaluation/I/O. The observed provisional calendar estimate is 5--7 days. The complete local suite passed 223 tests plus 6 subtests in 36.32 seconds; the focused reviewer suite passed 30 tests. No H200 issue requires action.

P0-2 remains open until all 27 full runs are aggregated and adjudicated. Neither selection, cleanup nor launch is a biological, causal or held-out quality result; failed pairs remain calibrated dictionary-quality negatives.

## 2026-07-24 — EXP-R2-025: text-vs-protein interpretability transfer screen (TG-01..TG-07)

Matched seven-experiment screen diagnosing why sparse-feature interpretability transfers poorly from text decoders to protein decoders. Arms: GPT-2-large (36L/1280d, matched text control), ProtGPT2 (36L/1280d), ZymCTRL (36L/1280d, run in its native `EC<sep><start>...<end>` format) and ProGen2-medium (27L/1536d). Scripts: `scripts/transfer_gap/tg0{1..7}_*.py`, collation `tg99_summarize.py`. Results: `results/transfer_gap_20260724/` with `SUMMARY.json`. Single L20 GPU, `conda activate ct`, one analysis seed (20260724) per configuration.

Headline results.

- **TG-01 (predictive-information budget, 400 sequences/arm).** Information resolved over a unigram baseline: GPT-2-large 7.05 bits/token; ProtGPT2 2.64 bits/token (0.89 bits/residue); ZymCTRL 1.97 bits/residue; ProGen2-medium 1.81 bits/residue. Top-1 accuracy 0.463 / 0.102 / 0.506 / 0.489. Share of that information available from the last 8 tokens: 0.744 / 0.569 / ~0.00 / 0.008. Held-out residue Markov baselines are 4.191 / 4.184 / 4.197 bits/residue at orders 0/1/2, so local protein context is worth ~0.007 bits.
- **TG-02 (far-context decomposition).** Order-dependent share of far-context information: 0.68 / 0.77 / 0.79 / 0.87. Protein decoders are *more* order-specific than the text control, refuting an order-invariant composition/homology account.
- **TG-03 (matched TopK SAE: relative depth 0.5, x8 expansion, k=32, 5M activations, 6,000 steps).** FVU 0.050 / 0.021 / 0.267 / 0.171; loss recovered 0.968 / 0.336 / 0.701 / 0.376. Across the three protein models the FVU ranking is exactly inverted relative to the loss-recovered ranking.
- **TG-04 (explanation channel).** Top-k event ceiling h(m/N): the historical 100-of-122,671 design caps at 0.006613 nats, and a 0.1-nat gate needs 2,516 events. Within-sequence label entropy over a 300-symbol window: text token 7.32 bits, residue identity 4.11, Pfam domain label 0.74, AlphaFold-derived structural attributes 3.61.
- **TG-05 (anchored partner identification, 180 AlphaFold models, protein-disjoint, 2,200 held-out anchor groups).** ProGen2-medium AUC: partner marginal 0.717, concat [h_i;h_j] 0.722, attention pattern 0.816. ZymCTRL: 0.714 / 0.717 / 0.757.
- **TG-06 (attention-pattern transplant, exact re-injection verified to 0.0000 nats).** Cost as a fraction of context-derived information: GPT-2-large 0.77, ZymCTRL 0.76, ProtGPT2 >1.0. Frozen attention is not a protein-specific liability.
- **TG-07 (training-free PCA-truncation splice).** At rank 512: variance explained 0.950 / 0.996 / 0.865 / 0.917 with loss recovered 0.975 / -0.105 / 0.793 / 0.537. Only GPT-2-large reaches 90% loss recovered anywhere in the sweep. ProtGPT2 participation ratio 1.1 (PC1 = 95.1% of variance).

Consequence for the live panel: P0-2 is currently adjudicated on frozen FVU/dead/L0/firing gates. TG-03 and TG-07 show variance criteria can rank protein dictionaries backwards, so a loss-recovered splice metric must be added before adjudication. Full analysis, derivations, diagnosis and proposals are in the repository-root `check.md`.

Limitations: small matched dictionary budget; single seed; CA-trace P-SEA secondary structure rather than DSSP; local human-biased AlphaFold subset; ProtGPT2's loss-recovered denominator on the TG-07 cohort is ~1 nat so its exact negative value is unstable; ZymCTRL TG-02/TG-06 windows exclude its EC tag.

## EXP-R2-032: P0-2 closure and prospective P0-2b behavioural-fidelity qualification (2026-07-27 CST)

**Purpose.** Close the original frozen P0-2 gate without retroactive changes, then test whether its 27 completed checkpoints preserve next-token behaviour well enough to enter the revised downstream programme proposed in `check.md`.

### Original P0-2 closure

The hash-bound gate builder and adjudicator completed successfully. The executed specification is `evidence/p0_2_adjudication_20260727/p0_2_dictionary_gate_spec.executed.json` (SHA-256 `f96aa1da3316adc064aa9cb14e03c772e8dc3b3954325ba3387a61530f8af2a1`). The receipt is `evidence/p0_2_adjudication_20260727/p0_2_eligibility_receipt.json` (SHA-256 `3f472afb0171836ad7f51d5c1d9e25b1d8d4a67aae1ac1aa51744bf18494f9b3`). Panel status is `one_or_more_model_method_quality_gates_failed`. Only ProGen2-medium TopK is atlas-eligible; P0-2 fails at panel level. The receipt contains the expected 27 full runs and three terminal validation sparsity failures (ProtGPT2 ReLU/L1 and gated, ProGen2-medium ReLU/L1).

The 27 full runs consumed 425.170 aggregate GPU-hours. Their queue logs had no traceback, OOM, non-finite or no-space event. Automatic completed-run cleanup removed 27 redundant progress checkpoints totalling 647,097,074,621 bytes and retained the 27 exact selected checkpoints.

### Prospective protocol and implementation

The new protocol is `docs/P0_2B_DICTIONARY_FIDELITY_PROTOCOL_20260727.md`. It explicitly preserves the P0-2 receipt and treats P0-2b as a new downstream qualification. The focused implementation is:

- `scripts/79_run_dictionary_fidelity.py`;
- `src/revision/dictionary_fidelity.py`; and
- `tests/test_dictionary_fidelity.py`.

The implementation covers all four dictionary methods, exact target-layer window reconstruction, native ZymCTRL and ProGen2 conditioning, sequence-target masks, loss/KL recovered, denominator guards, sequence-cluster bootstrap, exact reinjection, model/checkpoint/artifact binding and atomic results. Revision-5 archive SHA-256 is `2dcbbaf3405e92bf9db50c2c5c2ae43f789c7ee119dfa0ddd3e786da2eaf5a5e`. The runner/module/protocol SHA-256 values are `3c2734e84436fb0fe9ecf9b685e1ef3190632c534dbf657afaea0765fea6b4d8`, `c6bf134153a493b2620638a2dbbc15c00848e379cee965c4f9436c5f78aea7ed` and `a2225b40aa3326beb957d97cd12ad86d126544edac2e32f01054bf1f6f6d608f`.

Pre-outcome fail-closed checks found and corrected three cohort/setup issues: a noncanonical-residue FASTA row, insufficient untouched sequences under the initial 512-row/240-residue proposal, and a 260-token ZymCTRL prompt under a 250-residue ceiling. No dictionary forward pass accessed the prospective cohort during these checks. The final frozen cohort has 240 sequences, 64--246 residues, drawn from 248 eligible sequences after excluding 380,385 unique prior/source-prefix hashes. Its SHA-256 is `19474c1f94be9e4e3ff9b9ef30c76b5870a211b1c8dc3b7d0c0852afebc35dd3`. Tokenizer-only maximum lengths were 85/256/247 for ProtGPT2/ZymCTRL/ProGen2-medium, with no truncation.

The executed P0-2b specification SHA-256 is `11092e16891922380a2d224288138968e45ea982d36b46a4007f9f6b76a8d6bf`. A one-record integration smoke used an already-accessed P0 validation sequence, not the prospective cohort; exact reinjection had zero logit difference. Config/weight-tree/tokenizer-tree verification passed for all three deployed models.

### Command and execution

Four fixed queues ran from the immutable r5 tree on H200 GPUs 0--3:

```text
python scripts/79_run_dictionary_fidelity.py run-queue \
  --spec p0_2b_fidelity_spec.executed.json \
  --output-dir results --queue-index <0..3> --device cuda:0 --batch-size 2
```

Launcher PIDs were 42461--42464. All 27/27 result files completed in 267.383 seconds on the longest queue. The aggregate command was:

```text
python scripts/79_run_dictionary_fidelity.py aggregate \
  --spec p0_2b_fidelity_spec.executed.json \
  --results-root results --output p0_2b_fidelity_panel.json
```

### Result

All 27 exact-reinjection gates passed with zero maximum logit difference. No sparse model/method passed P0-2b:

- ProtGPT2 mean-ablation CE delta was 0.033783 nats/token and ZymCTRL's 0.015783, below the prespecified 0.05 denominator minimum. Their recovered ratios are undefined.
- ProGen2-medium denominators were valid (CE delta 0.061427; mean-ablation KL 0.083770). Dense loss/KL recovered were 0.120/0.071, 0.114/0.060 and 0.102/0.080. Gated values were 0.323/0.229, 0.305/0.234 and 0.331/0.243. TopK values were 0.368/0.262, 0.355/0.254 and 0.333/0.250.
- The best bootstrap upper limits (ProGen2-medium TopK seed 17) were 0.411 for loss recovered and 0.282 for KL recovered, far below 0.80.

Aggregate SHA-256 is `7f68e08775af87a171f2e4aac2a0d88cf235280280bf1bac44f76b0a2959bd07`. The detailed analysis is `docs/analysis/P0_2B_DICTIONARY_FIDELITY_RESULTS_20260727.md`.

On the nine ProGen2-medium runs, P0-2 test FVU and the new metrics ranked in the same direction: descriptive Spearman `rho=-0.867` for FVU versus loss recovered and `rho=-0.850` for FVU versus KL recovered (lower FVU is better). This production panel does not support generalizing TG-03/TG-07's exploratory inverted-ranking claim.

### Resources, storage and decision

Queue wall times were 138.039, 228.252, 137.093 and 267.383 seconds (0.2141 aggregate accelerator-hours). Per-run evaluation summed to 208.914 seconds. Maximum allocation was 14,242,418,176 bytes (13.264 GiB). Result JSON totalled 5,880,475 bytes; the synchronized evidence tree is 6,045,859 bytes. Post-run GPUs were idle and GPFS remained 21% used with about 44 TB available. No H200 issue required user action.

The exact caches and selected checkpoints were retained because they are hash-bound scientific inputs. P0-2b itself created no material storage pressure. Local verification passed 241 tests plus 6 subtests; the focused suite passed 18 tests, and Ruff check/format passed.

Per the frozen decision rule, do not launch P2--P8, legacy P0-3--P0-8, steering or atlas expansion. No further experiment is active.

## 2026-07-27 — EXP-R2-026: TG-08/TG-09/TG-10 follow-up to the P0-2b null result

Diagnostic round following P0-2b, which qualified no sparse model/method. Scripts `scripts/transfer_gap/tg08_budget_sweep.py`, `tg09_depth_profile.py`, `tg10_causal_headroom.py`. Results under `results/transfer_gap_20260724/tg08|tg09|tg10/`. Single seed unless stated.

**TG-10 (causal headroom of the P0-2b estimand; no dictionary involved; 120 sequences/arm, 64-246 residues, cohort-mean ablation baseline).** Mean-ablation CE delta in nats/token at relative depth 0.5, single MLP output: GPT-2-large +0.018, ProtGPT2 -0.069, ZymCTRL +0.017, ProGen2-medium +0.003. All four fail the P0-2b 0.05 nats/token denominator guard, including the matched text control that P0-2b did not run. GPT-2-large fails at every depth tested (0.012-0.021 at relative depths 0.15/0.33/0.50/0.67/0.85). Independent replication of P0-2b is close for ZymCTRL (+0.0167 vs 0.0158) but diverges for ProGen2-medium (+0.0031 vs 0.0614) and flips sign for ProtGPT2, indicating strong sensitivity to the ablation-baseline definition. Widening scope restores headroom: eight-layer window +0.479 / +0.233 / +1.198 / +0.025 and all-MLP +8.094 / -0.475 / +2.706 / +0.193 for GPT-2-large / ProtGPT2 / ZymCTRL / ProGen2-medium.

Cohort power is a second limiter. Unigram entropies recomputed on this exact cohort with each model's own tokenizer give ZymCTRL 3.057, ProGen2-medium 2.918 and ProtGPT2 8.389 nats/token. Against clean CE this leaves ZymCTRL 2.292 nats of context information (ample), ProGen2-medium 0.099 nats, and ProtGPT2 -1.728 nats, i.e. ProtGPT2 is off-distribution and performs worse than a context-free baseline on short EC-labelled enzymes. ZymCTRL's failure is therefore one of estimand scope alone, not cohort power.

The one non-artefactual signal: ablating all MLP outputs costs GPT-2-large 8.094 nats/token but ZymCTRL 2.706 and ProGen2-medium 0.193. Transcoders and CLTs decompose the MLP pathway specifically, so this is a measured mechanistic reason why MLP-transcoder circuit tracing has less to address in a protein decoder.

**TG-08 (residual-stream estimand; data and optimisation compute varied separately; dictionaries saved).** Loss recovered at 0.5M / 2M / 8M activations with steps fixed at 8,000: GPT-2-large 0.958 / 0.964 / 0.962 (saturated); ZymCTRL 0.610 / 0.716 / 0.843 (about +0.12 per 4x of data, still climbing); ProGen2-medium 0.206 / 0.249 / 0.286. Within ZymCTRL, FVU (0.500 / 0.405 / 0.261) and loss recovered are aligned, not inverted.

**TG-09 (training-free PCA-splice depth profile).** GPT-2-large loss recovered 0.834-0.966 across relative depths 0.15-0.85 with 5.4-7.0 nats of ablation headroom; ProtGPT2 about 0.35 at every depth with about 1.0 nat of headroom. Relative depth 0.5 was therefore not an unfair site for the text control.

**Retraction.** The TG-03/TG-07 "FVU ranking exactly inverted versus behavioural fidelity" claim is withdrawn. P0-2b (rho = -0.867 within ProGen2-medium, nine runs, three seeds) and TG-08 (within ZymCTRL) both show alignment. The surviving weaker claim is that FVU is not comparable across models and does not by itself certify behavioural fidelity. See check.md §0.5, §0.6 and the revised §8 plan.

## 2026-07-28 — R2 restructured around the transfer question; two pilot claims corrected

The direction was refocused on the text-to-protein interpretability transfer gap and its root causes. New package `src/transfer/` with entry points under `scripts/transfer/`; see `docs/RESEARCH_PLAN.md` and `docs/methods/TRANSFER_MEASUREMENT_PROGRAMME.md`. Compute policy: L20 for validation, H200 for full-scale campaigns via `scripts/transfer/run_transfer_h200.sh` (pod taken from `H200_POD`; pods are disposable and are never hard-coded).

Two corrections to the 2026-07-24 pilots, both selection/cohort effects that had inflated an effect in the direction of the hypothesis.

1. **TG-10 did not replicate P0-2b.** P0-2b evaluated EC-labelled Swiss-Prot; TG-10 drew plain Swiss-Prot for the unconditional arms. ProGen2-medium clean CE is about 2.76 nats/token on plain Swiss-Prot 64-246 aa versus about 1.64 on the EC-labelled cohort. Confirmed not to be a padding artefact: clean CE 2.762 / 2.764 / 2.767 at batch 1 / 4 / 8. Withdrawn: the claim that the P0-2b cohort starves ProGen2-medium (its context information on the correct cohort is 1.244 nats, ample), and the claimed ZymCTRL replication. Confirmed independently: ProtGPT2 is off-distribution on that cohort at -1.692 nats/token of context information and must be reported unmeasurable.

2. **The TG-05 relational result was a structure-selection artefact.** The pilot iterated `sorted(glob("AF-*"))`, i.e. accession order, over-sampling related proteins. Under a seeded permutation of all 23,586 AlphaFold models with a homology-disjoint split at 140 proteins: attention 0.598 linear / 0.628 MLP; partner-marginal 0.595 / 0.577; concatenated per-position 0.593 / 0.537; separation-only control 0.571 / 0.507. Attention remains best and a nonlinear probe raises it, so the gap is not a linear-probe artefact, but the margin is about 0.03-0.05 rather than the 0.10 reported. Homology splitting was not the cause (homology-disjoint 0.598 vs random 0.601); selection order was.

Also verified at validation scale by the rebuilt package: residue Markov order 0/1/2 at 4.174 / 4.159 / 4.206 bits/residue; event-selection ceiling for top-100-of-122,671 at 0.006613 nats, so the historical 0.1-nat gate was 15.1x unattainable; within-300-symbol label entropy of 7.335 bits for text tokens versus 0.733 bits for Pfam labels, with Pfam mean majority-label share 0.763 versus 0.054 for text.

Literature calibration recorded in check.md §0A: Anthropic circuit tracing reports a 0.61 replacement score and about 50% next-token agreement; ProGenMech (arXiv:2606.16044) reports about 60% likelihood recovery on ProGen3-112M and 95% recovery of zero-shot fitness Spearman, with its own steering failure. Published protein and text replacement fidelity are therefore comparable, which argues against locating the transfer gap primarily at dictionary fidelity.

Repository cleanup: eight superseded scripts moved to `archive/legacy/r2_superseded_scripts_20260728/` with a provenance README; empty `docker/` removed; `.ruff_cache/` added to both .gitignore files.

## 2026-07-28 — EXP-R2-027: estimand power and pathway budget under cohort parity

Rebuilt package `scripts/transfer/02_pathway_budget.py` and `03_estimand_power.py` on the P0-2b cohort (EC-labelled Swiss-Prot, one shared digest across all protein arms), 48 sequences x 2 seeds, 300 sequence-cluster bootstrap resamples, `cohort_mean` and `zero` ablation baselines.

Context information, nats/token: gpt2-large +3.95 on its own text cohort, ZymCTRL +2.09, ProGen2-medium +1.52, ProtGPT2 -1.31 (clean CE 8.71 against its own unigram 7.40) and therefore off-distribution and unmeasurable. This is a third independent confirmation that ProGen2-medium is not cohort-starved, retiring the 0.099-nat figure from the 2026-07-27 entry.

Single-MLP-output ablation at relative depth 0.5, cohort_mean / zero baselines: gpt2-large 0.0195 / 0.0227; ZymCTRL 0.0105 / 0.0166; ProGen2-medium 0.0637 / 0.0774; ProtGPT2 -0.0058 / -0.0164. P0-2b's training-target-mean figures for ZymCTRL (0.0158) and ProGen2-medium (0.0614) are corroborated under a different baseline; the ProtGPT2 sign disagreement with P0-2b (0.0338) is unexplained and that arm cannot be scored on this cohort.

Corrected claim. The 2026-07-27 entry stated the P0-2b estimand fails its denominator guard for every model. That is too strong. The text positive control fails at every depth under both baselines (0.0111 / 0.0225 / 0.0183 with cohort_mean; every bootstrap 95% upper bound at or below 0.0273; guard pass fraction 0.0 across all 300 resamples of both seeds) and ZymCTRL fails, but ProGen2-medium clears the guard at all three depths (0.127 / 0.064 / 0.071). The gate is therefore mis-specified relative to its own control rather than universally unattainable. Tool reports `p0_2b_estimand_attainable_on_text_control = false`, `attainable_panel_wide = false`, recommended powered estimand `attn_window4@d0.15@cohort_mean`, 32 of 48 candidate estimands panel-wide powered.

The TG-10 pilot was reproduced exactly on its own cohort (clean CE 2.8186 against 2.8186; mlp_all 0.1845 against 0.1932), confirming its arithmetic was correct and only its cohort was wrong.

First pathway-budget figures, whole-pathway ablation in nats/token: gpt2-large MLP 8.13 versus attention 5.23 (MLP-dominant); ZymCTRL MLP 2.60 versus attention 4.81 (attention-dominant); ProGen2-medium MLP 1.64 versus attention 1.73 (near parity). Direction matches the pathway hypothesis and mlp_all for gpt2-large reproduces the pilot at 8.09. This is NOT yet a modality claim: per check.md §0B.3 the panel cannot separate modality from tokenisation, and ProGen2-medium is near parity rather than attention-dominant.

Known measurement defects, being fixed: the default plug-in unigram estimator is biased low for 50k-vocabulary arms (gpt2-large 6.71 against the disjoint estimate 7.52), inflating share_of_context_information for those arms by about 20%; and two seeds are too few given the observed spread on small protein estimands (ZymCTRL depth 0.5 spans 0.0042-0.0168).

## 2026-07-28 — EXP-R2-028: ProtGPT2 input-rendering defect; shared-cohort design manufactured the modality gap

Two findings that retract earlier conclusions, including one in the production P0-2b qualification.

**1. ProtGPT2 was fed the wrong input format.** It was pretrained on FASTA-formatted UniRef50 (hard-wrapped at 60 residues, end-of-text separated) and its BPE merges were learned over that byte stream. `src/transfer/arms.py` rendered it as one unwrapped line, and the P0-2b protocol specifies "ProtGPT2 receives the raw sequence". Measured clean CE on 80 Swiss-Prot sequences of 600-2000 residues: raw 8.046, end-of-text+raw 8.090, wrapped-at-60 6.652, end-of-text+wrapped 6.623 nats/token — the correct rendering is worth 1.42 nats/token. Re-measured on 120 sequences: unigram 8.582, clean CE 6.352, context information +2.230 nats/token, realised information fraction 0.260, against the -1.31 nats and 0.130 recorded under the defect.

Retracted as artefacts: the 2026-07-27 claim that ProtGPT2 is 1.73 nats/token worse than a context-free baseline; the -1.692 "independent confirmation" (same broken renderer); ProtGPT2's exclusion from the estimand-power panel and from the convergence ladder. P0-2b's ProtGPT2 mean-ablation delta (0.033783) and its undefined-denominator verdict for that arm are affected; its ZymCTRL and ProGen2-medium arms are not, their native formats being correct. `arms.py` now carries `input_format="fasta_wrapped"` and every artefact records its rendering. ProtGPT2 remains the weakest ladder member at rif 0.260 against text 0.542-0.626, ZymCTRL 0.757, ProGen2-medium 0.526, ProGen2-small 0.347 — weak, but measurable.

**2. The shared-cohort design manufactured the modality gap.** Fitting mlp_share ~ realised_information_fraction + modality over the 8-member ladder: with one shared 64-246-residue EC cohort the protein offset is -0.933, 95% CI [-1.556, -0.310], excluding zero; giving each arm its own native cohort moves it to -0.657, 95% CI [-1.627, +0.314], including zero. Verdict `underpowered` (5 residual dof, interval half-width 0.971 against a 0.100 equivalence margin). A third fit under corrected ProtGPT2 rendering is pending.

One identified offset, multiplicity uncontrolled (1 of 18 fits): protein decoders have fewer attention heads above the induction prefix-matching threshold at matched convergence (-0.118 [-0.176, -0.060] on rif, R2 0.856; -0.110 [-0.139, -0.080] on log10 parameters, R2 0.956), but no deficit in the strength of their best induction head (ProGen2-small 83.7, ProGen2-medium 86.9 against GPT-2 83-87). ZymCTRL has none above threshold.

Ladder now 8 verified members: gpt2, gpt2-medium, gpt2-large, gpt2-xl, protgpt2, zymctrl, progen2-small, progen2-medium. Absent: progen2-large (missing), progen2-base (repo-qualified auto_map resolves modelling code from the Hub; fixable by stripping the prefix), progen2-xlarge (config publishes vocab_size_emb/vocab_size_lm_head but not vocab_size).

Operational: `results/transfer_20260728/` was wiped twice by an unidentified concurrent process; no rmtree or rm -rf exists in scripts/transfer or src/transfer. Output was restored from a backup and snapshots are now taken under logs/.

## 2026-07-28 — EXP-R2-029: circuit primitives (induction heads, DLA, activation patching)

`scripts/transfer/04_circuit_primitives.py`, all four arms, one code path, one EC-labelled cohort digest `de9ae47030`, 600-1000 residues so every arm including ProtGPT2 reaches the 128-token patching window. Validation scale.

Positive control passed: GPT-2-large top head L16H0 prefix matching 0.924 against head-average 0.041 and uniform baseline 0.011; 17 heads clear mean+3SD, all at depth fraction 0.31-0.66.

Induction census, max prefix matching (synthetic / natural repeats), heads above 0.10, and copying of the top head (mean normalised rank / diagonal fraction): gpt2-large 0.924 / 0.741, 69 heads, 0.947 / 0.731; ProtGPT2 0.550 / 0.465, 13 heads, 0.499 / 0.050; ProGen2-medium 0.937 / 0.549, 6 heads, 1.000 / 1.000; ZymCTRL 0.027 / 0.090, 0 heads, 0.026 / 0.000. The induction motif dissociates in ProtGPT2: prefix-matching heads with OV circuits at chance copying. ZymCTRL shows the inverse, copying heads with no prefix matching (L9H17 rank 0.971, prefix 0.021). The three protein arms disagree sharply, so this splits by tokenisation and conditioning rather than by modality. Copying score depends on token-set size (GPT-2 2,259 observed content tokens versus 20 for residue models); a matched-|T|=20 variant over 8 seeded subsamples is reported alongside.

Direct logit attribution, MLP magnitude share: gpt2-large 0.673, ProtGPT2 0.705, ZymCTRL 0.616, ProGen2-medium 0.691. This does NOT reproduce the ablation-based pathway reading in EXP-R2-027 (text MLP-dominant, ZymCTRL attention-dominant). The two estimands differ (ablation includes indirect effects, DLA is the direct linear path) but the pathway hypothesis is not currently supported by two independent measurements. ProGen2-medium is an extreme attribution outlier: top-1 component share 0.650, participation ratio 2.0 of 55, its logit essentially produced by mlp.26 alone.

Activation patching, fraction of one-token corruptions moving the logit difference by more than 0.25, band q-p = 1 / 5-8 / 33-64: gpt2-large 88 / 50 / 9 per cent; ProtGPT2 81 / 59 / 34; ZymCTRL 50 / 34 / 6; ProGen2-medium 84 / 69 / 50. Single-token perturbations propagate further in ProtGPT2 and ProGen2 than in text, consistent with the earlier finding that protein predictive information is non-local. Four structural invariants hold exactly as correctness proofs.

Per-head OV slicing is verified rather than assumed (relative error 0.0028-0.0038 on all arms), which matters because ProGen2 uses GPT-J-style qkv_proj with an 8-way model-parallel interleave and (query, value, key) ordering.

Repository hygiene: transfer entry points 01, 05 and 06 wrote to `results/transfer/<name>/` while 02, 03, 04, 07 and 08 wrote to `results/transfer_20260728/<name>/`; all now unified on the latter. The shared results root was repeatedly deleted by a concurrent process during this session; no rmtree or rm -rf exists in scripts/transfer or src/transfer, so the deletion came from an interactive shell command. Snapshots are taken under logs/.

## 2026-07-28 — EXP-R2-030: the apparent modality gap is absorbed by tokenisation

Convergence control re-run on the 8-rung ladder with per-arm native cohorts and the corrected ProtGPT2 FASTA rendering, float32 throughout.

Fitting mlp_share ~ realised_information_fraction + modality:

| design | protein offset | 95% CI |
|---|---|---|
| shared cohort, 5 rungs | -0.933 | [-1.556, -0.310] excludes 0 |
| native cohorts, broken rendering | -0.657 | [-1.627, +0.314] includes 0 |
| native cohorts, corrected rendering, unadjusted | -0.635 | [-1.095, -0.174] excludes 0 |
| native cohorts, corrected rendering, tokenisation-adjusted | -0.077 | [-0.449, +0.295] includes 0 |

Adding a symbol-level tokenisation indicator moves the protein coefficient from -0.635 to -0.077 and R2 from 0.719 to 0.961; the tokenisation coefficient carries the effect at -0.643 [-1.005, -0.286]. Raw group means: subword text 1.82, subword protein (ProtGPT2) 1.70, residue protein 1.09 — ProtGPT2 sits with the text models. An independent direct tokenisation control gives BPE-protein 1.704 versus residue-protein 1.093, difference +0.610, reproducing the fitted coefficient.

The rendering fix moved ProtGPT2 from rif 0.130 to 0.355, CE 3.338 to 2.582 bits/residue, and MLP share 2.951 to 1.704.

Verdict `underpowered` under a strengthened rule requiring a modality offset to survive tokenisation adjustment. Limitation: n=1 in the subword-protein cell, so ProtGPT2 alone carries the identification; a second subword protein model is the highest-value panel addition. Single seed, 4-5 residual dof, validation scale.

Blocked and unblocked: progen2-large excluded with the padded-vocabulary hazard recorded (logit_columns_used 51200 against tokenizer vocab 30) under a general rule permitting config vocab to exceed tokenizer vocab only up to 64-alignment padding; progen2-xlarge excluded because budget.arm_power and pathways.measure_pathways read config.vocab_size, which it does not define; progen2-base repaired by stripping the repo-qualified prefix from its auto_map after verifying md5 parity of configuration_progen.py, modeling_progen.py and tokenizer.json with its siblings, and it now loads offline at 27 layers, embed_dim 1536, 764.8M parameters, logits (1,10,32) — the paired architecture contrast against progen2-medium is unblocked. circuits.py lacks a fasta_wrapped branch, so ProtGPT2 currently has no induction or attribution row and the induction offset is unattributable pending that fix.

## 2026-07-28 — EXP-R2-031: lens family; output-interface rank; unigram estimator defect

**Lens family** (`scripts/transfer/08_lens_family.py`, float32, 11-point relative-depth grid, all four arms, protein arms sharing cohort digest e4cceb54803f). Positive control passed: GPT-2-large logit lens CE 10.46 -> 2.67 nats/token and KL-to-final 7.79 -> 0.0000, monotone in depth; the tuned lens strictly beats the untuned lens at every non-identity layer on all arms (mean KL reduction 1.47 nats); the float32 lens head reproduces the model's own final distribution to maximum KL 3.6e-7.

Jacobian lens, numerical rank of d logits / d h_l (same-position block only, so a lower bound) and blind-variance fraction at mid-depth: gpt2-large 1199-1278 of bound 1280, 0.064; ProtGPT2 1150-1278, 0.495; ZymCTRL 457 = V-1 exactly at every layer and probe, 0.632; ProGen2-medium 31 = V-1 exactly, 0.985. For residue-level protein decoders the output-interface rank is pinned by vocabulary rather than model width: ProGen2-medium's next-token interface is a rank-31 aperture onto a 1536-dimensional residual stream. The implementation records that for V-1 < d the blind-variance fraction is forced algebraically and is not an empirical finding; the empirical content is which directions survive. Validated against a central finite difference at 2.2e-4 to 2.5e-3 relative error, with an independent check that the deepest-layer rank is exactly d-2 for the 1280-wide arms, matching LayerNorm's two null directions. This splits by vocabulary, not modality: ProtGPT2, a protein model with a 50k subword vocabulary, has a near-full-rank interface like the text arms.

ZymCTRL's apparent coarse-to-fine residue-class structure under the untuned logit lens (class CE halving at relative depth 0.126 versus 0.370 within class, gap +0.244) collapses to +0.035 under the tuned lens, and ProGen2-medium flips sign (-0.023 untuned, +0.039 tuned). Most of the apparent structure is basis error, not model structure.

ProtGPT2 off-distribution finding withdrawn a third time, independently: context information +1.34, +2.04, +2.74 and +0.21 nats across four cohorts; the -1.73 figure did not reproduce on any. That figure survives in `budget.py`'s docstring and needs correcting.

**Unigram estimator defect** (pathway track). `budget.arm_power` computes a plug-in unigram baseline on the same tokens it scores. Held-out estimation gives gpt2-large 6.707 -> 7.459 (+0.752), ProtGPT2 7.218 -> 8.869 (+1.651), ZymCTRL 3.065 -> 3.081, ProGen2-medium 2.894 -> 2.897. Because share = dCE / (H - CE), this inflated shares by 19 per cent on gpt2-large and 74 per cent on ProtGPT2 and negligibly on the small-vocabulary arms. Corrected mlp_all share: gpt2-large 2.042 -> 1.711, ProtGPT2 1.980 -> 1.196, ZymCTRL 1.191 -> 1.185, ProGen2-medium 1.061 -> 1.060. Under the plug-in GPT-2-large and ProtGPT2 looked near-identical (2.04 vs 1.98); corrected they separate (1.71 vs 1.20) and ProtGPT2 groups with the protein arms. This threatens to reverse the tokenisation-absorption result of EXP-R2-030, which is therefore flagged do-not-quote pending a re-derivation with a held-out baseline. The held-out estimator also caught a genuine leak: Swiss-Prot and the EC corpus repeat sequences under different accessions (15 of 4000 shared), so skipping the measurement pool does not yield a disjoint corpus.

Corrected estimand attainability for mlp_single@d0.50@cohort_mean: gpt2-large 0.0225 (not powered), ZymCTRL 0.0168 (not powered), ProGen2-medium 0.0772 (powered), ProtGPT2 0.0895 (powered). The text positive control has the smallest single-MLP footprint in the panel and two of three protein arms clear the gate it cannot. Seed requirement from observed between-seed spread: gpt2-large 2, ZymCTRL 2, ProGen2-medium 19, ProtGPT2 41.

## 2026-07-28 — EXP-R2-032: induction dissociation survives; DLA/ablation divergence explained

**Circuit primitives re-run under corrected ProtGPT2 rendering.** `circuits.py` gained `fasta_wrapped` branches for `content_bounds` and `prefix_ids`, plus a rewrite of `natural_repeat_probes`: it had located records by substring search and assumed a constant character shift between repeat copies, both of which fail under wrapping (the record is not a substring of its own input, and a variable number of newlines falls within and between copies). Alignment now works in symbol space via `record_symbol_offsets`, verified symbol-by-symbol, and is a provable no-op for the unwrapped arms (ZymCTRL and ProGen2 give 8/8 probes at coverage 0.954, identical to before). ProtGPT2 natural-repeat coverage drops 0.854 to 0.712 because line breaks genuinely fragment repeat segmentation — a real cost of the correct rendering.

The synthetic probe is assembled directly in token space (`prefix + body + body`) so the second copy is bitwise the first; verified `copies identical=True` on all arms. Layout tokens (line breaks, 4.8 per cent of ProtGPT2's content stream, 3.6 per cent of gpt2-large's) are excluded from the unigram support uniformly across arms so no arm receives a perturbation class the others do not. The synthetic probe carries no line break across 128 tokens and is therefore itself off-distribution for a 60-residue-wrapped arm; for ProtGPT2 the natural-repeat number is the one to trust.

Induction census, ProtGPT2 broken versus corrected rendering: synthetic max 0.550 -> 0.533, natural max 0.465 -> 0.557, heads above 0.10 unchanged at 13/14, same depths 0.49-0.83, same three leading heads (L17H16, L21H8, L19H2), panel copy rank at matched N=20 0.328 -> 0.333. The dissociation finding survives: ProtGPT2 has prefix-matching heads whose OV circuits do not copy (top-head copy ranks 0.36/0.48/0.39). Positive control tightened: gpt2-large L16H0 0.952 against head-mean 0.0421 (22.6x) and uniform 0.0107 (89x), same 17 heads at depths 0.31-0.66. ProtGPT2 DLA moved as expected: mean target logit 8.38 -> 14.47, MLP share 0.705 -> 0.727, top-1 concentration 0.124 -> 0.151.

**DLA versus ablation divergence characterised.** The scoring-window hypothesis was tested and refuted: scoring ZymCTRL's EC tag moves attention share down (0.3814 -> 0.3628), and gpt2-large is bit-identical either way. The cause is estimand-intrinsic. Attention share by ablation versus DLA, identical arms, positions and code path: gpt2-large 0.404 / 0.325 (gap +0.079); ProtGPT2 0.474 / 0.271 (+0.203); ZymCTRL 0.506 / 0.381 (+0.125); ProGen2-medium 0.542 / 0.309 (+0.233). Ablation is a total-effect measure, DLA a direct-path measure with the layer-norm scale frozen; the gap is attention's indirect contribution, which it makes by moving information that later MLPs convert into logits. That gap is about three times larger in the residue-level protein arms than in text, and it cross-checks against the copy-rank result (GPT-2's induction heads write directly to the unembedding at copy rank 0.947-0.960 while ProtGPT2's copy at chance) and against the patching map (resid_post recovers about 1.0 while attn_out and mlp_out individually recover 0.1-0.6).

`progen2-base` added to `arms.py` PANEL with `MATCHED_DATA_CONTRAST`; verified to load offline at 27 layers, d_model 1536, with mlp and attention hooks resolving. The results root was wiped mid-run a third time; the track's end-of-run re-emit and `verify_outputs` recovered the artefacts.

## 2026-07-28 — EXP-R2-033: tokenisation conclusion retracted; data variation eliminated

Convergence control re-derived with a held-out unigram baseline (40,000 reference sequences per corpus, drawn past the measurement pool then content-deduplicated; the guard removed 1 shared record on the EC corpus) over the 9-rung ladder including progen2-base.

Plug-in bias tracks vocabulary: +0.739 nats for the four 50k text arms, +1.020 for ProtGPT2, +0.012 ZymCTRL, +0.009 ProGen2. MLP share plug-in -> held-out: gpt2 1.779->1.479, gpt2-medium 1.955->1.647, gpt2-large 1.715->1.453, gpt2-xl 1.839->1.565, ProtGPT2 1.704->1.243 (-27 per cent, largest move), ZymCTRL 1.161->1.156, ProGen2-small 1.097->1.087, ProGen2-medium 1.022->1.016.

Fits of mlp_share ~ realised_information_fraction, n=9: unadjusted protein offset -0.426 [-0.606, -0.245] excluding zero; tokenisation-adjusted protein offset -0.252 [-0.516, +0.011] and tokenisation offset -0.193 [-0.433, +0.046], both including zero. The tokenisation conclusion of EXP-R2-030 is RETRACTED: under the biased plug-in the tokenisation coefficient was -0.643 [-1.005, -0.286] excluding zero, and it was an artefact of inflating the single subword-protein rung by 27 per cent. Modality does not replace it — ProtGPT2 at 1.243 sits between text (1.45-1.65) and residue-protein (1.02-1.16), the effect splitting about 0.25/0.19 between the two indicators with neither resolvable at n=9. The descriptive tokenisation control falls from +0.610 to +0.170. Verdict remains underpowered.

Paired data contrast (progen2-base versus progen2-medium: identical 27 layers, width 1536, 764,803,616 parameters, same cohort, same tokenisation, different pretraining corpus): MLP share differs by 0.019, attention share by 0.072, realised information fraction by 0.010. Against a raw text-protein MLP-share gap of about 0.43, pure pretraining-data variation accounts for roughly one twenty-second. This eliminates incidental training-data differences as the explanation; the residual remains modality, tokenisation, or corpus class.

Meta-finding: the protein coefficient has moved four times under defensible corrections that never touched the models — shared cohort -0.933, native cohorts -0.657, corrected rendering -0.635 unadjusted / -0.077 adjusted, held-out baseline -0.426 unadjusted / -0.252 adjusted — each worth 0.2-0.9 on a quantity whose whole magnitude is about 0.4. At n=9, dof 5-6, one subword-protein rung, one seed and 18 uncorrected fits, the normalised-share estimand cannot carry the central claim in either direction, and further L20 validation will not change that. Strategic decision recorded in check.md §0B.19: shift weight to denominator-free estimands (the rank-(V-1) output aperture and the induction dissociation) and run the share estimand on H200 in the background at the derived seed requirement rather than as the critical path.

## 2026-07-28 — EXP-R2-034: ZymCTRL prompt-leak confound; session state

The probe track identified, before being interrupted, that ZymCTRL's native input format `EC<sep><start>sequence<end>` leaks family identity into any family, EC or function probe: the EC tag very nearly determines the Pfam family, so near-perfect Pfam skill read from ZymCTRL activations may be reading the prompt rather than the representation. This must be refused rather than caveated. Scope extends beyond probes: ZymCTRL's realised information fraction of 0.757-0.770, the highest in the panel, is partly its own prompt supplying the answer, which weakens the earlier observation that it argues against protein models being undertrained; and any atlas or conserved-feature result that treated ZymCTRL as an unconditional generator inherits the same problem. Remedies in order of preference: run it unconditioned for representation probing and record that it is then off its native distribution; hold the EC tag fixed within a probe fold; or report it separately from the unconditional arms and never pool them. The `probe_erasure/*.json` artefacts predate this finding and are provisional.

Session state: 10 modules under src/transfer/, 9 entry points under scripts/transfer/, an H200 launcher, and 30 result artefacts across seven stages (circuit_primitives, convergence_control, estimand_power, lens_family, pathway_budget, probe_erasure, plus the recommendation). Six result snapshots under logs/. Two agents terminated on a session limit mid-task with no work lost.

Next actions in priority order: re-run the induction and attribution axes now that circuits.py supports fasta_wrapped, taking those fits to n=9 with a subword-protein rung present; re-run the probe track with the ZymCTRL prompt leak refused; add a second subword-protein model to break the n=1 cell; launch the H200 share-estimand campaign in the background at the derived seed requirement (41 seeds ProtGPT2, 19 ProGen2-medium) rather than as the critical path; and locate whatever wipes results/transfer_20260728/, which struck three times and is not in committed code.

## 2026-07-28 — EXP-R2-035: results-wipe diagnosed; design degeneracy identified

The repeated destruction of `results/transfer_20260728/` is diagnosed. This research root is its own git repository and `/results/` is listed in its `.gitignore`, so `git clean -fdx` (or `-fdX`) executed here deletes every experiment artefact. That is why no `rmtree` or `rm -rf` appeared in committed code — it was interactive tidy-up, not a program. Warnings are now at the top of `.gitignore` and in `scripts/transfer/README.md`; snapshots are retained under `logs/results_snapshot_*`. `git clean -fd` without `-x` is safe.

Design degeneracy identified as the reason modality and tokenisation cannot be separated. The 2x2 currently holds text/subword n=4 (gpt2 through gpt2-xl), text/symbol-level EMPTY, protein/subword n=1 (ProtGPT2), protein/symbol-level n=4 (ZymCTRL, progen2-small/base/medium). One empty cell and one singleton is why the two indicators trade off at roughly 0.25/0.19 with neither resolvable. Staging a decoder-only byte- or character-level TEXT model would populate the empty cell and make the design identifiable — a larger gain than a second protein subword model, which still leaves a corner missing. Both are being sought, text byte-level first. If no genuinely decoder-only byte-level text LM can be staged, that is a permanent limit on what this design can resolve and must be reported as such.

H200 campaign deliberately held. The launcher is ready but the estimand it would power up is the one this programme has argued should not be the critical path, three fixes are still landing, and pod-side corpus mounting is unverified. Launching now would produce results superseded by the next correction, which is the pattern that produced four retractions today. It launches once the design cell is filled and the induction and probe re-runs report.

## 2026-07-28 — EXP-R2-036: design cells; ProtGPT2 is the only subword protein LM

**Text x symbol-level cell filled.** Four decoder-only byte-level text models staged and CPU-verified at exactly 1.000 characters per token: nllg/bygpt5-small-en (4 layers, 1472 wide, 73.5M), nllg/bygpt5-base-en (6, 1536, 139.2M), nllg/bygpt5-medium-en (12, 1536, 289.1M) and google/reformer-enwik8 (12, 1024, 149.2M). google/byt5-* correctly excluded as encoder-decoder. The bygpt5-large tier does not exist; correct repo ids carry an -en suffix.

**Protein x subword cell cannot be strengthened.** InstructProtein's official prompting format wraps every residue in a dedicated added token (`<protein>` + per-residue separator), measuring 0.984 residues/token and confirmed against 20 per-amino-acid entries in added_tokens.json; unprefixed letters give 1.538 chars/token, weak incidental BPE that is not the model's real usage. The RITA family has vocabulary 26 with one token per amino acid, 0.984 chars/token. Against ProtGPT2's measured 3.000 chars/token neither is close, and a broader mirror search found nothing else. ProtGPT2 appears to be the only public protein language model with genuine subword tokenisation, so that cell is permanently n=1 with available models — a structural limit of the field rather than a gap in the search, and a cap on what any version of this design can resolve.

With the empty cell filled both main effects become estimable at cell counts (4, 3-4, 1, 4); the singleton caps precision on the tokenisation coefficient.

**Architecture is now a recorded covariate.** ByGPT5 is T5-derived (relative position biases, T5-style layer norm) and Reformer uses LSH attention with reversible layers, so neither is architecturally comparable to the GPT-2-family text arms. The protein side already mixes architectures (ZymCTRL GPT-2/CTRL-style, ProGen2 GPT-J-style), so this is a difference of degree, but Reformer must not be used for pathway, attention or DLA measurements where its attention and residual structure are not commensurate with a standard decoder.

Integration note: ByGPT5's HuggingFace repos ship no modelling code and `model_type: "bygpt5"` is not built into transformers; the reference implementation is GitHub-only and pins transformers 4.43.3. Under 4.57.3 the legacy T5 attention path breaks against the Cache object with caching enabled and runs cleanly with use_cache=False. The checkpoints are being made self-contained with local auto_map entries and use_cache disabled, following the pattern already used by the ProGen2 checkpoints, with originals backed up.

## 2026-07-28 — EXP-R2-037: migration to H200; controller/worker launcher

Compute policy updated: H200 for experiments, L20 reserved for the lightest GPU and CPU work. The earlier decision to hold the campaign is superseded.

Cluster state at migration: all 16 GPUs allocated cluster-wide, but our own pod holds 4 idle H200s (143 GB each, 0 per cent utilisation), 192 cores and 2015 GB RAM, with GPFS mounted read-write and 44 TB free. GPFS is not mounted on the master, so transfers route through a pod.

Already on GPFS and requiring no transfer: uniref50 (36G), proteingym, interpro pfam_residue.tsv, the ZymCTRL EC corpus, ProtGPT2, ZymCTRL, progen2-medium, progen2-xlarge, esm2_t36_3B, esmfold_v1, and from the 24 July staging gpt2-large (3.1G), openwebtext-screen (1.3G), cath-s40 (825M), wikitext-103-v1 and blimp.

Staging in priority order: swissprot uniprot_sprot.fasta.gz (93M, needed by nearly every cohort); the four byte-level text models filling the empty design cell; gpt2/gpt2-medium/gpt2-xl; progen2-small/base/large; and data/alphafold (9.8G, 47,173 files).

Environment differs and is the main portability risk: pod runs torch 2.7.1+cu128 with transformers 4.52.4 against the L20 host's 2.9.1+cu128 and 4.57.3. Specific exposure: logits_to_keep support, attn_implementation="eager", torch_dtype versus dtype deprecation, Cache-object effects on custom modelling code. Every measurement is cross-checked between hosts against L20 reference values (context information gpt2-large +3.95, ZymCTRL +2.09, ProGen2-medium +1.52 nats/token; single-MLP ablation at relative depth 0.5 cohort_mean 0.0195 / 0.0105 / 0.0637). Systematic divergence is to be reported, not absorbed. Upgrading the pod's shared environment is out of scope.

Launcher refactored to a two-tier controller/worker architecture. Controller runs locally: require H200_POD, run the documented health check, compute a content hash over src/transfer and scripts/transfer, sync that snapshot to a versioned immutable GPFS path under Research2/transfer_package/<run-id>/, write a run manifest (code hash, per-file checksums, git revision and dirty state, UTC timestamp, arms, stages, parameters), invoke the worker by pod exec, stream logs and propagate the exit status. Worker runs in the pod: source the pod environment file, verify GPFS mount and data paths and idle GPU count, execute stages in dependency order with one arm per GPU, write every artefact atomically via temp-file rename with per-stage output checksums, log under a run-scoped directory, and skip stages whose verified output already exists unless forced. The worker never deletes or recreates the results root.

The code freeze is not bureaucracy for this programme: four conclusions have been retracted because a defensible change moved a number, so binding a run to an immutable code snapshot is what makes a result attributable at all.

## 2026-07-28 — EXP-R2-038: 2x2 design populated; capability gating enforced in code

The ByGPT5 checkpoints were made self-contained and durable: vendored configuration, modelling and tokenizer modules copied into each checkpoint directory, `auto_map` set to the bare local form with no repo qualifier, `use_cache: false` set deliberately (the legacy T5 attention path breaks against the Cache object under transformers 4.57.3 and runs cleanly without caching), and originals backed up. Verified fully offline with HF_HUB_OFFLINE=1 and plain Auto* loading, no manual imports or sys.path tricks; classes resolved from transformers_modules, confirming local resolution. A real bug was caught in the first pass: tokenizer_class is a custom ByGPT5Tokenizer, not the built-in ByT5Tokenizer, so a tokenizer module was genuinely required.

The 2x2 modality-by-tokenisation design is now populated in all four cells: text/subword gpt2 through gpt2-xl; text/symbol bygpt5-small/base/medium-en; protein/subword ProtGPT2 (n=1, irreducible — it is the only public protein LM with genuine subword tokenisation); protein/symbol ZymCTRL and progen2-base/medium. ByGPT5 measures 0.992 characters per token through the panel's own code path.

Architecture comparability is now enforced in code rather than documented in prose. `ArmSpec` carries `architecture` and a `capabilities` frozenset; `Arm.require(capability)` raises for a measurement family the arm cannot enter commensurably; `blocks()` resolves per architecture (transformer.h for gpt2/progen, decoder.block for t5_decoder) instead of duck-typing; and `mlp()` and `attention()` refuse any architecture outside the decomposable set. ByGPT5 carries budget and lens only. Verified that a call resolving on gpt2-large is refused on bygpt5-small-en.

google/reformer-enwik8 was staged but deliberately excluded from the panel: it ships no tokenizer (the checkpoint expects manual byte+2 encoding and AutoTokenizer resolves a ReformerTokenizer that fails for want of a sentencepiece vocab file). Admitting it would require a bespoke tokenizer shim in a shared module for a model whose LSH attention and reversible layers restrict it to the budget family regardless, when three ByGPT5 rungs already populate the cell. The checkpoint remains on disk.

## 2026-07-28 — EXP-R2-039: first identified finding — fewer induction heads, not weaker ones

Convergence control re-run with all 9 ladder rungs carrying induction and attribution rows (ProtGPT2 restored once circuits.py gained fasta_wrapped support). Held-out unigram baseline, float32, natural-repeat probe for every arm.

induction_natural_fraction_of_heads_above_threshold ~ realised_information_fraction, n=9: unadjusted protein offset -0.0970 [-0.1375, -0.0565]; tokenisation-adjusted -0.0997 [-0.1803, -0.0192] with tokenisation offset +0.0030 [-0.0701, +0.0762]; without ZymCTRL unadjusted -0.1173 [-0.1894, -0.0453] and adjusted -0.1215 [-0.2378, -0.0052]. All four exclude zero. The modality coefficient barely moves under adjustment and the tokenisation coefficient is indistinguishable from zero — the opposite pattern to the MLP share, where adjustment halved the modality term and tokenisation took the rest. The synthetic probe family agrees independently (-0.116 -> -0.120, tokenisation +0.004).

Substance. Strongest natural induction head and fraction of heads above threshold: gpt2-xl 92.4x / 0.0650; gpt2 89.2x / 0.1389; ProGen2-medium 89.6x / 0.0139; ProGen2-base 79.4x / 0.0069; ProGen2-small 79.7x / 0.0260; ProtGPT2 32.8x / 0.0194; ZymCTRL 14.6x / 0.0014. ProGen2-medium's best induction head is statistically indistinguishable from GPT-2-xl's while its head count above threshold is about five times lower and ProGen2-base's about nine times lower. Protein decoders are not missing induction; they have a few heads that are just as sharp and far fewer of them.

Not surviving, recorded so they are not quoted: DLA participation flips sign under adjustment (-0.063 -> +0.132, both including zero); DLA MLP magnitude share includes zero either way; induction_natural_max_prefix_matching_over_uniform is design-dependent (-43.0 unadjusted, -89.1 adjusted, -5.7 without ZymCTRL) and must not be quoted. MLP share remains underpowered (-0.4256 [-0.6061, -0.2451] unadjusted, -0.2522 [-0.5156, +0.0113] adjusted).

Controls. The ZymCTRL conditioning leak is not driving the induction result: removing that arm makes the offset larger, not smaller. The progen2-base versus progen2-medium contrast (identical to the parameter, differing only in pretraining corpus) moves induction head count by 0.0069 against a modality offset of -0.10, about 7 per cent, so incidental training-data differences remain eliminated.

Methodological notes. The instruction to use the natural probe for ProtGPT2 only was correctly overridden: mixing probe families across rungs inside one fit would make the modality coefficient partly a probe-family coefficient. Natural is used for all nine arms with synthetic fitted in parallel. ProtGPT2 yielded 14 of 16 natural probes, every other arm 16. A stale CIRCUITS_INPUT_FORMATS list in scaling.py excluded ProtGPT2 from the first re-run and the guard failed safe, emitting a recorded skip rather than a wrong number, at the cost of one pass.

Validation scale, n=9, single seed, 18 uncorrected fits. This is the first result worth powering up properly and it is denominator-free, so it is not exposed to the estimator failure mode that moved the share coefficient four times.

## 2026-07-28 — EXP-R2-040: probes and concept erasure; decodability is not reliance

`src/transfer/probes.py` and `scripts/transfer/09_probe_and_erasure.py`. Erasure by LEACE (Belrose et al., arXiv:2306.03819), the closed-form affine oblique projection, verified numerically to post-erasure cross-covariance about 1e-15 on the fit sample. Artefacts under results/transfer_20260728/probe_erasure/ including zymctrl__ec_fixed and zymctrl__ec_unconditioned variants.

ZymCTRL conditioning leak quantified. Pfam-family linear skill at mid-depth: +0.996 native (leaked), +0.583 under a fixed constant EC tag, +0.710 unconditioned; clean CE 1.15 / 2.88 / 3.04 nats respectively. The prompt is worth 0.29-0.41 of family skill and almost all of EC skill (EC class under the fixed tag is only +0.127). The EC tag is worth 1.73 nats/token of ZymCTRL's own next-residue cross-entropy. This withdraws the earlier observation that ZymCTRL's realised information fraction of 0.757-0.770, the highest in the panel, argues against protein models being undertrained: most of that figure is its own prompt.

Refusal is implemented in code rather than as a caveat: under native conditioning every sequence-level concept is refused with an error naming the two alternatives, both of which are implemented and were run. Structure probes are confirmed unaffected: three-state secondary-structure skill 0.232 native, 0.233 fixed, 0.229 unconditioned. Caveat: both alternatives are behaviourally degenerate at the erasure layer (mean ablation costs only 0.07-0.11 nats under the fixed tag and is negative unconditioned, so the denominator guard fires), so their probe numbers are sound but their erasure effect sizes have no usable scale.

Erasure verification passes on every arm: linear skill goes from positive to below chance (+0.44 to -0.24) while a variance-matched concept-agnostic control leaves it essentially untouched (+0.4410 to +0.4361), so the collapse is concept-specific.

Decodability and reliance separate cleanly. ProGen2-medium secondary structure is decodable (+0.26) and relied upon (+0.031 nats excess, CI [+0.021, +0.044]). ProtGPT2 Pfam family is highly decodable (+0.71) with negative excess (-0.179 nats) — encoded but not consulted by the next-residue computation. A high probe score on a protein language model is therefore not evidence that the model uses the concept.

Deviations, all deliberate and recorded in the artefacts: the primary control was changed to a raw orthonormal random direction because the matched whitened oblique control is pathological on ProtGPT2, costing 3.6 of the 4.2 nats that full mean ablation costs while displacing activations less than LEACE does (all three controls reported, with a control_matching block flagging any that is not cost-matched); ProtGPT2 is refused only for residue- and variant-level concepts, with sequence-level ones measured on a verified sequence-body token span; probe sets are matched on tokens rather than sequences; pfam_family is not family-disjoint because the label is the group, declared as family_disjoint false; progen2-base included as the matched-data contrast.

Limitations: LEACE guarantees linear erasure only and the post-erasure MLP probe retains skill up to +0.71, so reliance is gated on the linear probe with the MLP as a diagnostic; fitness has only 10 assay groups and its rank-1 erasure excess is underpowered; validation scale on L20, no H200 run.

## 2026-07-28 — EXP-R2-041: controller/worker launcher refactor

`scripts/transfer/run_transfer_h200.sh` is now the local controller and `scripts/transfer/h200_worker.sh` the pod-side worker, with the split documented in `scripts/transfer/README.md`.

Controller: requires H200_POD fail-fast, runs the documented health check first, computes a code freeze over src/transfer and scripts/transfer as a sha256sum-format manifest (matching the convention of 70_run_exact_cache_queue_h200.sh) and hashes that manifest into RUN_ID=<timestamp>_<hash prefix>, refuses to overwrite an existing snapshot, accepts an explicit RUN_ID only if its trailing hash matches the code on disk, pushes the snapshot and a run manifest through the documented access-layer tools, invokes the worker by pod exec and propagates its exit status. Verified end-to-end against the real tree: 23 correctly hashed files, real git revision d765d4bf761d, dirty flag true matching 105 modified files.

Worker: sources the pod environment file, verifies GPFS is writable and of type gpfs via stat -f, verifies data paths and GPU availability, then runs the nine stages in three dependency tiers. Every item writes into a mktemp directory inside its own final directory so the move is same-filesystem, then moves files into place and writes a per-item sha256 manifest under .manifests/; resume checks sha256sum -c against it. Tested in isolation: first run writes, second run skips as verified complete, and after deliberate corruption of an output the third run detects the mismatch and redoes the work, leaving no orphaned temp directories.

Three real defects surfaced by the refactor: 07_convergence_control.py's --ladder-table default points at docs/analysis/MODEL_LADDER_20260728.md, outside both frozen directories, so the controller copies that one file into the snapshot explicitly excluded from the code-hash scope; 04_circuit_primitives.py writes a combined panel_summary.json that would corrupt under per-arm parallelism, so it is dispatched as a single job; and 05_relational_channel.py is restricted to zymctrl and progen2-medium by both a modality check and its own docstring refusal.

Contract mismatch found and being corrected: the worker had invented R2_TRANSFER_* variable names, while the authoritative contract read by src/transfer is R2_REPO_ROOT, R2_MODEL_BASE_DIR, R2_TEXT_MODEL_DIR, R2_TEXT_MODEL_BASE_DIR, R2_OPENWEBTEXT_DIR, R2_SWISSPROT_FASTA, R2_ZYMCTRL_FASTA, R2_PFAM_RESIDUE_TSV, R2_PROTEINGYM_DIR and R2_ALPHAFOLD_DIR. Two preflight adjustments also requested: verify only the paths the requested stages need (h200_env.sh deliberately does not check existence, because swissprot and alphafold are still staging and a measurement needing neither must still run), and derive the GPU count from nvidia-smi rather than hard-coding it, with the 40 GB free-memory floor either lowered to something defensible or made a warning since observed validation peaks were 1.6-12.2 GiB.

## 2026-07-28 — EXP-R2-042: launcher contract reconciled and preflight scoped

The worker now reads the authoritative environment contract (R2_REPO_ROOT, R2_MODEL_BASE_DIR, R2_TEXT_MODEL_DIR, R2_TEXT_MODEL_BASE_DIR, R2_OPENWEBTEXT_DIR, R2_SWISSPROT_FASTA, R2_ZYMCTRL_FASTA, R2_PFAM_RESIDUE_TSV, R2_PROTEINGYM_DIR, R2_ALPHAFOLD_DIR), confirmed by reading the actual env_path and require_input_path call sites in arms.py, channels.py and probes.py rather than inferred. All invented R2_TRANSFER_* references removed.

Two integration defects caught while wiring it. The worker now invokes "$R2_PYTHON" rather than bare python3, because the pod has no conda environment. More importantly it exports R2_PACKAGE_ROOT to this run's snapshot directory *before* sourcing h200_env.sh: that file's own default for R2_PACKAGE_ROOT is run-id-less, so sourcing it first would have pointed PYTHONPATH at whichever snapshot happened to be newest rather than at the frozen snapshot for this run, silently defeating the code-freeze guarantee that the whole controller design exists to provide.

Preflight is now scoped per item rather than blanket: verify_item_data_paths computes exactly which variables a given stage and arm touch and checks only those. Verified in isolation — pathway_budget on gpt2-large checks only the text model and OpenWebText paths and passes with alphafold and proteingym pointed at nonexistent files, while relational_channel on zymctrl correctly fails on a missing R2_ZYMCTRL_FASTA with the variable named. For stages 07 and 09, whose real dependencies are conditional on which ladder rungs are staged and which probe concepts a run reaches, only source-confirmed paths are checked and the rest is left to require_input_path, erring toward not blocking a runnable job.

GPU count is derived from nvidia-smi at run time and each requested GPU index is validated against it, with --expected-gpu-count now optional; pods are disposable so a hard-coded count would be wrong at the next one. The free-memory floor was lowered from an unprofiled 40000 MiB to 16000, comfortably above the observed 1.6-12.2 GiB validation peaks, and split from occupancy: another process on the GPU remains a hard failure, insufficient free memory is a logged warning only.

Outstanding gate before any real campaign: the L20-versus-H200 numerical cross-check, which decides whether the pod reproduces the local measurements.

## 2026-07-28 — EXP-R2-043: H200 port validated; context-information label corrected

Package ported to the H200 pod (torch 2.7.1, transformers 4.52.4) and cross-checked against the L20 host (2.9.1, 4.57.3). Every filesystem location in arms.py, channels.py, probes.py and scaling.py now derives from a named environment variable defaulting to the local value, with existence checked at first use by require_input_path naming the variable; with no R2_* set, every constant and all 14 ladder paths are byte-identical to the pre-port literals, and with every variable pointed at a nonexistent path all six entry points raise FileNotFoundError naming the variable.

Cross-check result. Re-running the ported code on L20 reproduces the pre-port L20 reference exactly to six decimal places, so the path work is numerically inert. Host divergence is 1.4e-4 to 2.9e-3 nats with sign varying by arm and no systematic bias; cohort pool, held-out-reference and per-seed subsample digests are identical on both hosts; three repeat runs per host are bit-identical, so the difference is deterministic rather than GPU nondeterminism. Zero verdict flips across all 108 scope-by-seed comparisons, and every H200 point estimate lies inside the corresponding L20 cluster-bootstrap 95 per cent interval. Root cause proven to be bfloat16: a float32 rerun of ProGen2-medium collapses the divergence from 2.9e-3 to 2.6e-7 on clean CE and from 8.7e-2 to 9.2e-7 on the truncation curve.

Label correction. The context-information reference triple (gpt2-large +3.95, ZymCTRL +2.09, ProGen2-medium +1.52) has been repeatedly described in this log and in briefs as held-out-estimator figures. It is not: +3.95 is plug_in minus clean_CE, and the held-out value for gpt2-large is +4.71 (held-out unigram 7.4585 against plug-in 6.7067, clean CE about 2.75). The residue-level arms are barely affected since their plug-in bias is under 0.02 nats. No conclusion changes — the EXP-R2-033 fits were re-derived with the held-out estimator and reported as such — but the triple must be labelled plug-in.

Incompatibilities, none silently absorbed. logits_to_keep does not exist in 4.52.4, so the truncation curve refuses for vocabularies above 1024 and 01_cohort_power.py needs --skip-truncation for gpt2-large and ProtGPT2 on the pod; the guard was deliberately not relaxed because trimming was measured to be numerically non-inert (up to 0.25 in a logit and 0.12 nats in one token's NLL, mean about 1e-3), and each run now records logits_to_keep_used because ZymCTRL takes the trimmed path on L20 and the untrimmed one on the pod. One statistic is host-bound: ProGen2-medium's nll_reduction_shortest_to_longest_nats moves 0.6266 to 0.7293 (+16 per cent) under bfloat16 because it is a small difference between two roughly 2.9-nat endpoints; run that arm's truncation curves in float32. Activation-patching case selection is threshold-sensitive at two cases per band, observed only at smoke scale. Cleared after checking: eager attention accepted by all four checkpoints including ProGen2 remote code; torch_dtype must be retained because dtype would be swallowed as a config keyword on 4.52.4 and silently load float32; no transformers.models internals are touched; the legacy-tuple versus Cache difference is inert because the package never reads past_key_values.

Not exercised in the pod: ProtGPT2 and every AlphaFold-dependent path, that corpus still staging; scripts 03 and 05-09 imported but not run.

## 2026-07-28 — EXP-R2-044: worker dispatch split by vocabulary class

Applying the port findings exposed a dispatch defect. 01_cohort_power.py writes its combined report only after its whole per-arm loop completes, so one arm raising discards every arm already computed in that invocation. Since the worker dispatched 01 by kind — one protein job covering ProtGPT2, ZymCTRL and ProGen2-medium together — a single --skip-truncation setting could not satisfy both the large-vocabulary arm that requires it under transformers 4.52.4 and the small-vocabulary arms that must not have it because they can and should compute the truncation curve.

cohort_power is now four items rather than two: text (gpt2-large), protein_large_vocab (ProtGPT2, --skip-truncation), protein_small_vocab (ZymCTRL) and protein_progen2 (ProGen2-medium, --dtype float32). Each carries its own --cohort-name, because ProtGPT2's and ProGen2-medium's non-EC cohorts are content-identical and would otherwise collide on the same output filename under the shared default. Verified for both a full panel and a restricted arm list, the latter skipping the absent items rather than crashing or misclassifying.

Recorded honestly rather than narrowed: 01's --dtype governs model loading for the whole invocation, so ProGen2-medium's entire cohort_power measurement runs in float32, not only its truncation curve. That is documented in the worker header and the README.

README additions: a "Known host-bound quantities" section covering logits_to_keep_used and the ZymCTRL trimmed-on-L20 versus untrimmed-on-pod path difference, with the instruction that cross-host comparisons must check that field; the cross-check pass recorded (108 of 108, no verdict flips) with the remaining blocker identified as AlphaFold and ladder staging rather than preflight; and a statement that a real campaign must always go through the controller and never a manual GPFS push, citing the port agent's own unfrozen manual snapshot as the hazard the code freeze removes.

## 2026-07-28 — EXP-R2-045: aperture axis reads null; denominator-free is not enough

Rank-(V-1) output aperture added to the convergence ladder and run on 12 rungs, 4 probes per arm, mid-depth, same-position block only, finite-difference checks passing at 2e-4 to 6e-3 against a 2e-2 tolerance. Ranks reproduce the lens track exactly: gpt2 763/768, gpt2-large 1276/1280, gpt2-xl 1597/1600, ProtGPT2 1255/1280, ZymCTRL 457 = V-1, every ProGen2 rung 31 = V-1.

The blind-variance fraction was refused rather than fitted. rank_is_forced is true for ZymCTRL and all three ProGen2 rungs because V-1 < d_model, and among the unforced rungs the panel is four text arms plus ProtGPT2 — all subword, one protein point, tokenisation fully aliased. A modality contrast would therefore be fitted either on an algebraically forced quantity or on a single-protein-rung design, neither admissible. Recorded as aperture_blind_variance_fraction_not_fitted and excluded from the metric set.

The empirical quantity, gain_alignment_ratio, shows nothing: full unadjusted +0.140 [-0.659, +0.939]; full adjusted -0.423 [-1.802, +0.956] with tokenisation +0.627 [-0.626, +1.880]; without ZymCTRL unadjusted -0.449 [-1.729, +0.832] and adjusted -1.080 [-2.680, +0.519]. Every interval contains zero and the point estimate flips sign between forms. The aperture axis does not join induction as an identified finding.

Correction to the EXP-R2-033 strategy note. Shifting weight to denominator-free estimands was right but incomplete: the paired progen2-base versus progen2-medium contrast moves gain_alignment_ratio by +0.328, comparable to or larger than every fitted modality offset, so at four probes per arm the axis cannot resolve anything and its null reads as "too noisy" rather than "no effect". A denominator-free metric is not automatically a low-variance one. The usable contrast is induction, where the same data contrast moves the metric by 0.0069 against a -0.10 modality offset, about 7 per cent. That ratio, not the absence of a denominator, is what makes a metric usable.

New caveat attaching to every fit on the convergence axis: the three ByGPT5 rungs give byte-level text a realised information fraction of 0.733 / 0.757 / 0.769, far above every BPE text rung (0.588-0.663) and level with ZymCTRL's 0.770, while ProGen2 sits at 0.349-0.538. The axis is therefore sensitive to tokenisation granularity and not monotone in modality; the induction fit's tokenisation-adjusted form partly absorbs this but the axis itself is not neutral.

Capability gating fired correctly twice more. ByGPT5 is excluded from pathway and circuits fits by arm.supports rather than by hand, and the induction fit is bit-identical to the previous run as a result. lenses.lens_head resolves the final normalisation as transformer.ln_f with no t5_decoder branch, so a declared LENS_ARCHITECTURES gate records the skip rather than producing a number from a path never written for that architecture; the text-by-symbol cell is therefore filled for the budget axis but still empty for the lens axis, and one branch for the T5 decoder's final_layer_norm would close it.

Induction unchanged and still the single identified finding: unadjusted -0.0970 [-0.1375, -0.0565], tokenisation-adjusted -0.0997 [-0.1803, -0.0192], tokenisation term +0.0030 [-0.0701, +0.0762].

Operational: two OOMs on the first attempt when the lens residual cache and the ablation measurement were resident together at gpt2-xl; the aperture pass now takes its own 8-sequence, 256-token, batch-2 window with a 2 GiB cache cap and peak fell from 25.5 to 22.7 GiB. inspect_member now reads tokenizer vocabulary from tokenizer.json where present and from the tokenizer class otherwise, with the source recorded per rung. n_head is gated on the circuits capability because it validates a GPT-2 head decomposition a T5 arm need not satisfy.

## 2026-07-28 — EXP-R2-046: first production H200 campaign; induction finding strengthens

Run 20260728114209_3fcefa3c4017, code-frozen and hash-bound, executed on 4x H200 through the controller/worker path with STAGES=circuit_primitives across gpt2-large, protgpt2, zymctrl and progen2-medium. 48 natural-repeat probes per arm (16 in validation), 128 synthetic probes, 120-sequence cohort, 32 attribution sequences, 12 patch cases per band, seed 20260728.

Induction census: gpt2-large 48 probes, 720 heads, peak 95.5x uniform, 50 heads above 0.10 (fraction 0.0694), panel copy rank 0.531; ProtGPT2 46 probes, 720 heads, peak 41.9x, 14 above (0.0194), copy rank 0.327; ZymCTRL 48 probes, 720 heads, peak 18.9x, 0 above (0.0000), copy rank 0.303; ProGen2-medium 48 probes, 432 heads, peak 115.7x, 6 above (0.0139), copy rank 0.490.

ProGen2-medium's sharpest induction head is 115.7x uniform, higher than gpt2-large's 95.5x, while its fraction above threshold is 0.0139 against 0.0694 — five times fewer. At three times the probe count and on different hardware the finding is sharpened rather than merely reproduced: protein decoders have induction heads at least as sharp as a text decoder's and far fewer of them. ProtGPT2's panel-mean copy rank of 0.327 remains below the 0.5 chance level against gpt2-large's 0.531, so the OV dissociation holds at production scale.

Hard constraint discovered. The first launch requested 400 natural-repeat probes and failed in 18 seconds: only 48 of 400 tandem-repeat proteins exist in 203,063 eligible entries. The repeat cohort is drawn from the EC-labelled corpus because ZymCTRL requires EC labels and all four arms must share one cohort digest, and genuine tandem repeats occur at roughly 1 in 4,000 entries under the stated criterion. 48 is therefore a natural ceiling rather than a tunable; raising it requires a larger labelled corpus, a relaxed repeat criterion, or abandoning the shared-cohort design, each a measurement decision rather than a knob. The run was relaunched at 48 rather than loosening the criterion. This caps the achievable per-arm precision of the programme's one identified finding and is the first thing to address before it can carry a manuscript.

The failed first launch is the better evidence that the machinery works: stage gating skipped the eight unrequested stages, the per-item data-path preflight passed exactly the five variables circuit_primitives needs, the failure was loud and fast and attributed to a named item with a pod-side log path, and no manifest was written so the re-run was not blocked. The second launch, differing only in arguments, correctly re-ran rather than resuming, exercising the provenance keying end to end.

## 2026-07-28 — EXP-R2-047: prior work found late; induction probe design called into question

Pomerants, Nikankin, Reusch, Tsaban, Schueler-Furman and Belinkov, "Induction Meets Biology: Mechanisms of Repeat Detection in Protein Language Models", arXiv:2602.23179 (February 2026, v5 July 2026), was not found before the induction track was designed and run. The opening literature sweep covered protein SAEs, ProGenMech and the SAE-limitations corpus; the induction idea was carried across from Olsson et al. without a search naming that mechanism in this domain.

What it establishes: on masked transformer encoders (ESM-3-open-1.4B and ESM-C-600M) roughly 43 induction heads identified by attention-pattern analysis and clustering, concentrated in middle and later layers while relative-position heads spread across all layers; approximate-repeat detection functionally subsumes exact-repeat detection, the approximate circuit generalising to identical repeats with cross-task faithfulness above 1.0 (99 per cent plus accuracy on exact repeats, 79 per cent on approximate); amino-acid-similarity neurons encoding BLOSUM62 substitution groups and IMGT physicochemical classes, concentrated in early layers, with 3 per cent of MLP neurons per layer sufficient; and ESM-3 recruiting secondary-structure-sensitive neurons where sequence-only ESM-C does not. Stated limits: exactly two repeat occurrences, at most 50 per cent substitution, no indels, manually defined concepts.

Superseded here: the existence of induction heads in protein language models and their concentration in middle-to-late layers are prior work, so describing the head-count result as the programme's first identified finding overstated its standing. Not superseded on available evidence: their panel is masked encoders while ours is autoregressive decoders (ProtGPT2, ZymCTRL, ProGen2), disjoint model classes with causal-masked induction not the same object as bidirectional induction; they explicitly do not systematically compare against text language models, whereas the comparative census at matched convergence with tokenisation adjustment is our claim; and the QK/OV dissociation measured in ProtGPT2 (prefix-matching heads whose OV circuits copy at or below chance) is not part of their account, whose induction heads attend to aligned tokens and therefore function.

Threat to the current result. Our natural-repeat probe requires an exact internal repeat of at least 16 residues. If approximate detection is the general mechanism and exact a special case, the probe measures a special case and may undercount protein induction heads tuned for substitution tolerance; the comparison may be biased against the protein arms, since BPE text has abundant exact token repeats while biological repeats are predominantly approximate; and the 48-protein ceiling is an artefact of exactness rather than a fact about proteins. The five-times head-count deficit is therefore provisionally under threat and must be re-tested with a substitution-tolerant probe before being claimed. The peak-sharpness result (ProGen2-medium 115.7x uniform against gpt2-large 95.5x) is less exposed, concerning the strongest head rather than the count.

Action: an approximate-repeat probe using a BLOSUM62-grounded similarity criterion is being added alongside the exact probe, following the prior work's scope (two occurrences, at most 50 per cent substitution, no indels), with a matched-permissiveness text control and both probes reported so the comparison between them is itself the evidence.

Process change adopted in docs/RESEARCH_PLAN.md: a targeted literature search naming the specific mechanism, not merely the domain, is a required gate before a measurement track is designed and again before a formal campaign is launched, with the queries run and works found recorded per track.

## 2026-07-28 — EXP-R2-048: approximate-repeat probe added; the head-count deficit survives

The threat raised in EXP-R2-047 was tested, not argued. `circuits.py` now carries a `RepeatCriterion` value and a substitution-tolerant search (`find_approximate_internal_repeat`) alongside the unchanged exact one, and `04_circuit_primitives.py` runs both natural-repeat probes on every arm in the same pass. Schema bumped to `r2_transfer_circuit_primitives_v2`.

Criterion, fixed before any head count was looked at. Ungapped, exactly two occurrences, no indels, at most 50 per cent of aligned positions substituted -- all four from arXiv:2602.23179 v5. `min_unit` 16, `max_gap_ratio` 2.0 and `min_distinct` 8 held at the exact probe's values, so exactly one clause moves. The biological grounding is BLOSUM62: the mean BLOSUM62 score over the *substituted* positions of a window must be at least zero. BLOSUM62 entries are log-odds of a substitution in aligned blocks of homologous proteins against background, so zero is the log-odds neutral point and not a fitted number, and the rule reads as "these are diverged copies, not two unrelated segments that happen to share half their residues". Every exact repeat satisfies every clause, so the approximate cohort is a superset of the exact one; asserted in `tests/test_approximate_repeat_probe.py`.

Two designs were measured and rejected before this one, on the criterion's own statistics and never on head counts. Requiring every position to be identical or BLOSUM62-positive yields 0 hits in 20,000 entries. Raw identity at 50 per cent with no similarity rule yields 404 hits in 20,000 but 152 in a composition-preserving shuffle of the same 20,000 -- 38 per cent of the cohort would be chance. The substituted-position BLOSUM62 rule is what removes that: 76 real against 1 null on the same sample, so roughly 1.3 per cent of the approximate cohort is expected chance. Text: 968 real against 0 null in 3,000 documents; protein exact 1 real against 0 null; text exact 116 against 0.

Text control. There is no honest analogue of BLOSUM62 for text: it is an empirical log-odds table estimated from aligned blocks of homologous protein families, and no comparable table exists over characters or BPE pieces. The text criterion is therefore the same criterion with the similarity rule dropped and everything else matched -- ungapped, two occurrences, the same 50 per cent cap, the exact text probe's own 40-character unit and 15-symbol complexity floor. That makes the text probe strictly *more* permissive than the protein probe, which is the one direction worth accepting because it cannot be read as the text arm having been handed a stingier probe. Recorded, not corrected: it does bias the other way, since the text arm's accepted repeats contain substitutions no similarity rule vouched for.

Cohort ceiling resolved. Full census of the EC-labelled Swiss-Prot source, 203,063 eligible entries: exact 48 (0.0236 per cent), approximate 817 (0.4023 per cent). A 17.0x lift, and the exact count reproduces the 48 discovered by the EXP-R2-046 launch failure exactly. The 48-record ceiling was an artefact of demanding literal identity, as suspected. Text over 3,000 documents: exact 116, approximate 968.

Validation runs, induction section only, one L20 (cuda:2), peak 2.6 GiB. Size-matched run at 32 records per cohort; power run at exact 48 / approximate 256, which the lifted ceiling now permits. The power run's exact column reproduces the EXP-R2-046 H200 production numbers to the digit -- gpt2-large 95.5x uniform and 50 heads above 0.10, ProtGPT2 41.9x and 14, ZymCTRL 18.8x and 0, ProGen2-medium 115.7x and 6 -- and the exact cohort digests are bit-identical (`e53aec6e63cf2335` text, `6da42b8fbfc1d096` protein), so the exact baseline was not disturbed by the new code path.

Power run, threshold 0.10, deficit = gpt2-large's fraction over the arm's:

| arm | probe | probes | scored | coverage | peak x uniform | n>0.10 | fraction | deficit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt2-large | exact | 48 | 1655 | 0.943 | 95.5 | 50 | 0.0694 | 1.00 |
| gpt2-large | approximate | 254 | 2369 | 0.414 | 77.9 | 44 | 0.0611 | 1.00 |
| protgpt2 | exact | 46 | 409 | 0.710 | 41.9 | 14 | 0.0194 | 3.57 |
| protgpt2 | approximate | 207 | 669 | 0.310 | 25.8 | 12 | 0.0167 | 3.67 |
| zymctrl | exact | 48 | 1712 | 0.965 | 18.8 | 0 | 0.0000 | n/a |
| zymctrl | approximate | 256 | 7511 | 0.956 | 7.5 | 0 | 0.0000 | n/a |
| progen2-medium | exact | 48 | 1712 | 0.965 | 115.7 | 6 | 0.0139 | 5.00 |
| progen2-medium | approximate | 256 | 7511 | 0.956 | 85.2 | 4 | 0.0093 | 6.60 |

Size-matched run at 32: ProGen2-medium 5.10 -> 7.40, ProtGPT2 3.64 -> 5.29, gpt2-large peak 98.0x -> 106.3x, ProGen2-medium 109.3x -> 109.5x.

Answers. The head-count deficit does not shrink and does not reverse. It holds or widens under every arm and both cohort sizes: ProGen2-medium 5.00 -> 6.60 (7.40 at n=32), ProtGPT2 3.57 -> 3.67 (5.29 at n=32). The peak-sharpness result survives in sign only: ProGen2-medium remains the sharpest arm under the approximate probe (85.2x against gpt2-large's 77.9x, +9.4 per cent; +3.0 per cent at n=32) but the +21 per cent margin of the exact probe does not survive, so "at least as sharp" stands and "sharper" does not. The ProtGPT2 QK/OV dissociation holds: it retains 12 prefix-matching heads above 0.10 under the approximate probe with a peak of 25.8x uniform, and its top heads copy at matched normalised ranks 0.478, 0.606, 0.389, 0.361, 0.718 against a 0.5 chance level and a panel mean of 0.333, where ProGen2-medium's top heads all copy at 1.000. The OV score is computed from weights and is probe-independent; what the approximate probe tests is whether the same heads still prefix-match, and L17H16, L22H13, L19H2 and L21H8 are the top heads under both.

Limitations, all measurable and none repaired. (1) The natural-repeat alignment scores a second-copy token only where the first copy is segmented at the same symbol boundaries. Under substitution tolerance that costs the two BPE arms most of their coverage -- gpt2-large 0.943 -> 0.414, ProtGPT2 0.710 -> 0.310 -- while the residue-level arms lose nothing (0.965 -> 0.956). The surviving text query positions are therefore enriched for locally identical content, so gpt2-large's approximate column is contaminated toward the exact regime, which inflates the deficit. The controlled reading is the matched pair gpt2-large/ProtGPT2, same depth, width, vocabulary size and tokenisation, where both arms pay the same coverage cost: there the deficit is 3.57 -> 3.67, flat. (2) ProtGPT2 loses 49 of 256 records entirely (207 probes) and is the least powered arm at 669 scored positions. (3) Roughly 1.3 per cent of the protein approximate cohort is expected chance under the shuffle null; the text approximate cohort measured 0. (4) The search returns the longest gate-passing window per period, which is exact for "longest" but says nothing about records carrying several repeats. (5) Cohort sizes 48 and 256 are not size-matched by construction, because the exact criterion cannot supply 256 protein records; `probe_comparison.size_matched` records which run is which.

One implementation defect found and fixed before any result was taken. The first search selected the longest window at each period and applied the composition and similarity gates afterwards, so a chance alignment elsewhere in the record could shadow the real repeat at the same period and the approximate criterion failed to subsume the exact one on synthetic tandem repeats. The gates are now inside the length sweep. `test_approximate_subsumes_exact_on_generated_records` is the regression.

Artefacts: `results/transfer_20260728/circuit_primitives_approximate/` (matched, n=32) and `results/transfer_20260728/circuit_primitives_approximate_power/` (exact 48 / approximate 256). The production `circuit_primitives/` tree was not overwritten. Every artefact carries the criterion parameters, the cohort digest, the source and the full-corpus census.

## 2026-07-28 — EXP-R2-049: production run at the resolved cohort ceiling

Run 20260728142257_50a83df4e602 on 4x H200, 640 approximate probes per arm against the earlier ~256, both criteria in one artefact (schema r2_transfer_circuit_primitives_v2). In-run cohort census: protein approximate 817 of 203,063 eligible (0.402 per cent); text approximate 968 of 3,000 (32.3 per cent). Per-arm runtime 32-44 s, peak 21.0-24.4 GiB.

Exact criterion reproduces the earlier production run exactly: gpt2-large peak 95.5x uniform with 50 of 720 heads above 0.10 (0.0694); ProtGPT2 41.9x, 14 (0.0194); ZymCTRL 18.9x, 0; ProGen2-medium 115.7x, 6 of 432 (0.0139).

Approximate criterion at 640 probes: gpt2-large 635 probes, peak 71.8x, 40 heads (0.0556), coverage 0.421; ProtGPT2 464 probes, 26.9x, 13 (0.0181), coverage 0.327; ZymCTRL 640, 7.7x, 0, coverage 0.955; ProGen2-medium 640, 88.5x, 4 (0.0093), coverage 0.955.

Head-count deficit in the controlled matched pair: 3.57 exact, 3.08 approximate. The earlier reading of 3.67 at 207 probes was low-n noise, so EXP-R2-048's "holds and widens slightly" is corrected to "robust in direction and roughly threefold in the controlled comparison", with no settled magnitude. ProGen2-medium moves 5.00 to 6.00 (6.60 at lower n), directional only because it crosses tokenisation.

Peak-sharpness claim narrowed again. At production scale ProGen2-medium's peak exceeds gpt2-large's by 23 per cent (88.5 versus 71.8), larger than the 9.4 per cent at low n. But within the matched pair ProtGPT2's peak is 26.9 against gpt2-large's 71.8, under half. So "protein induction heads are at least as sharp" is NOT supported as a general claim and must be stated per arm or not at all: head count shows a robust protein deficit, while peak sharpness splits, with the residue-level cross-tokenisation arm above the text arm and the matched-pair BPE arm well below it.

Coverage collapse reproduces at scale: BPE arms fall to 0.421 and 0.327 under substitution tolerance while residue-level arms hold at 0.955, so cross-tokenisation contrasts remain directional and the matched pair remains the controlled reading. Newly quantified asymmetry: approximate repeats occur in 32.3 per cent of text documents against 0.402 per cent of protein entries, an eighty-fold prevalence difference that is itself a substrate difference.


## 2026-07-28 — EXP-R2-050: homology control — the induction head count is not memorisation

*(Renumbered from EXP-R2-042, which was already taken by the launcher-contract entry; the identifier collided because two agents logged independently.)*

`src/transfer/homology.py`, `scripts/transfer/10_homology_control.py`. Artefacts under `results/transfer_20260728/homology_control/` (exact criterion) and `.../homology_control/approximate_criterion/`.

Threat tested: the protein decoders were pretrained on UniRef50 and the natural-repeat cohort is Swiss-Prot, which is largely inside it, so induction-shaped attention on a natural repeat could be retrieval of a stored near-duplicate rather than a copying computation.

Search. DIAMOND v2.1.24 (tarball sha256 verified against the staged `.sha256`; binary sha256 `5d04a9ec...`), `blastp --very-sensitive --evalue 1e-3 --max-target-seqs 100` against the FULL local UniRef50 snapshot: 60,315,044 sequences / 17,282,055,793 letters indexed from 60,315,044 FASTA records, coverage 1.000000. No subset, so the "a partial database only underestimates homology" caveat does not apply here. Identity is `100 * nident / qlen` (percent of the QUERY identically matched), not `pident`. Strata fixed before any result: <30 / 30-70 / 70-95 / >=95.

Achieved bins, records (distinct sequences). Exact criterion, n=48: 0 (0), 8 (4), 11 (11), 29 (22) -- the <30 bin is empty, which is structural: Swiss-Prot entries belong to UniRef50 clusters whose representative is >=50 per cent identical, and observed minimum identity was 62.1. Approximate criterion, n=817: 4 (4), 136 (106), 207 (179), 470 (420); minimum identity 25.2, so the <30 bin exists only under the approximate probe. The cohort itself contains byte-identical duplicate records (one per protein x EC pair; one group of 7 and one of 5), so every interval resamples distinct sequences and the headline per-stratum numbers use one probe per distinct sequence; duplicate-weighted values are reported alongside.

Result. The head count -- the quantity the finding claims -- shows no homology gradient. ProGen2-medium 5/3/4/4 of 432 across the four bins with overlapping intervals; ZymCTRL 0/0/0/0 of 720; ProtGPT2 8/11/13/13 of 720, the only arm whose head-count intervals separate and its <30 bin has n=4. The synthetic probe, which appears in no corpus and cannot be memorised, recruits AS MANY OR MORE heads than any natural stratum (ProGen2-medium 6 of 432 against 3-5; ProtGPT2 13 of 720 against 8-13) and largely THE SAME heads: head-set Jaccard against the synthetic control is 1.00 for ProtGPT2's 70-95 and >=95 strata and 0.83 for ProGen2-medium's <30 stratum.

Peak prefix-matching over uniform does rise with identity (ProtGPT2 18.6 -> 33.8, ProGen2-medium 84.1 -> 112.4, ZymCTRL 2.5 -> 8.8), and under the pre-stated interval rule that is scored `consistent_with_memorisation` for ZymCTRL and ProGen2-medium and `indeterminate` for ProtGPT2 at n=817. Recorded unretuned. But mean repeat length also rises across the same bins (24.3 -> 36.6 symbols), and a longer repeat sharpens prefix matching by itself. A binless partial-Spearman analysis at each arm's strongest induction head, added AFTER the stratum table was seen and labelled post hoc, separates them: repeat length given identity +0.611 (ProtGPT2) and +0.649 (ProGen2-medium) against identity given length +0.150 and +0.179. Length is roughly four times the identity term. The residual identity association is small but NOT zero and is reported as a real memorisation-consistent component of induction STRENGTH that does not transfer to head count.

Internal falsification of the strong memorisation account: ZymCTRL was pretrained on EC-labelled Swiss-Prot, this cohort's own source, so it has the greatest memorisation opportunity of any arm, and it shows the least induction (0 of 720 heads at every stratum, pooled peak 8.1x uniform). If memorisation produced induction, ZymCTRL would be the strongest arm; it is the weakest.

Reading against the three interpretations fixed in the module docstring before the search was run: the head-count finding is consistent with a GENERAL MECHANISM and survives; the peak-strength statistic is INTERMEDIATE and mostly a repeat-length artefact, which is an independent reason not to quote it, matching EXP-R2-039's own note that it is design-dependent.

Limitations, none of them removable here. GPT-2's training corpus is not public -- OpenWebText is a reconstruction of WebText, not WebText -- so this control cannot run symmetrically on the text arm and supports no matched cross-modal claim. The local UniRef50 snapshot is newer than ProtGPT2's 2021_04 training release, which biases measured identity upward. UniRef50 is ProtGPT2's corpus but only a proxy for ProGen2 (UniRef90 + BFD) and for ZymCTRL (BRENDA/UniProt). The <30 bin is empty under the exact criterion and has n=4 under the approximate one. Validation scale on one L20 GPU, peak 2.4 GiB, single seed.

## 2026-07-28 — EXP-R2-050 addendum: coordinator reading of the homology control

Two conclusions, split by statistic, plus one correction to the brief I issued.

Correction: I told the agent a partial DIAMOND database would be conservative. That is wrong. A found hit is always real, so high-identity strata stay pure and a memorisation gradient found there is trustworthy, but a miss contaminates the low strata and pushes the reading toward "general mechanism" — anti-conservative for clearing the finding, which is the direction of interest. Moot in the event, since the search covered the full local UniRef50 snapshot (60,315,044 sequences, 17.3 billion letters, coverage 1.000000), but the reasoning is now recorded in the module docstring.

Head count supports a general mechanism. No consistent gradient across identity strata (ProGen2-medium 5/3/4/4 heads above 0.10, ZymCTRL 0/0/0/0), and the synthetic probe — constructed rather than drawn from any corpus, so unmemorisable — recruits as many or more heads than any natural stratum, and largely the same heads: head-set Jaccard against synthetic is 1.00 for ProtGPT2's 70-95 and 95+ strata and 0.83 for ProGen2-medium's sub-30 stratum. ProtGPT2's 8 to 13 does separate but that bin has n=4 and synthetic also gives 13. Internal falsification: ZymCTRL was pretrained on EC-labelled Swiss-Prot, this cohort's own source, so it has the greatest memorisation opportunity of any arm and shows the least induction of any arm.

Peak strength does not survive. The pre-stated interval rule returned consistent_with_memorisation for ZymCTRL and ProGen2-medium at n=817 (indeterminate for ProtGPT2), recorded unretuned. Mean repeat length rises across the same bins (24.3 to 36.6 symbols) and drives prefix matching by itself; a post hoc binless partial Spearman gives length given identity +0.611/+0.649 against identity given length +0.150/+0.179, so length is about four times the identity term, with a small but non-zero residual identity association. This converges with EXP-R2-049, which narrowed the same claim on entirely different grounds (ProtGPT2's peak being under half gpt2-large's inside the matched pair). Two independent lines now agree: report head count, do not report peak sharpness.

Cohort defect found and fixed: the repeat cohort contained byte-identical duplicate records, one per protein-by-EC pair in groups of 7 and 5. Treating them as independent would have narrowed every interval. Headline numbers now use one probe per distinct sequence, with duplicate-weighted values retained alongside.

## 2026-07-28 — EXP-R2-051: staging complete; code freeze found incomplete

Staging finished and verified. Swiss-Prot transferred with exact SHA-256 match. All four byte-level models transferred and load-tested in the pod under transformers 4.52.4 with HF_HUB_OFFLINE. Text ladder (gpt2, gpt2-medium, gpt2-xl) transferred with exact safetensors size match. ProGen2 ladder (small, base, large) transferred. AlphaFold recovered from an interrupted transfer: the rebuilt local tar hashed byte-identically, so a genuine resume reused 33 of 39 chunks; extracted and verified in the pod at exactly 47,173 files and 10,318,793,261 bytes on both sides, with three sampled file hashes matching and three sampled .pdb.gz files parsing cleanly (CA coordinates and pLDDT B-factors in range, no truncation). GPFS free space 44 TB throughout.

progen2-large defect repaired. Its config.json carried the same repo-qualified auto_map prefix that progen2-base had, so it failed to load offline in the pod. Prefix stripped on the L20 host with the original backed up, verified loading on both hosts (32 layers, embed_dim 2560, 2,779.4M parameters), and the repaired config pushed to GPFS. It still emits 51,200 logit columns against a 31-token tokenizer, so it remains excluded from fits under the padded-vocabulary rule — now for the documented reason rather than a load failure.

Full nine-stage campaign launched (run 20260728150714_b613d3afe620) and failed at tier 1: all four cohort_power items died in two seconds with ModuleNotFoundError: No module named 'src.revision', raised at 01_cohort_power.py's import of src.revision.io.

Root cause is a gap in the code freeze. freeze_manifest hashes and syncs only src/transfer/ and scripts/transfer/, but the transfer package imports from src/revision/ (io, statistics, dictionary_fidelity, nested_recoverability) and needs src/__init__.py. The port agent had pushed src/revision/ by hand into its own manual snapshot, which is why the manual path worked; a controller-launched run has no such copy.

The earlier circuit_primitives campaigns did not expose this because that stage does not import src.revision. They therefore succeeded by accident of stage selection, and their snapshots were also incomplete, meaning those runs are not reproducible from the frozen tree alone. The freeze exists to bind a result to code that fully determines it and had been under-delivering silently.

Requested fixes: extend the freeze scope to the modules actually imported rather than a hard-coded directory list; include them in the code hash and not only the pushed snapshot, since syncing without hashing would leave a change in src/revision/ unable to invalidate resume provenance or change the run id; and add an import preflight that imports every stage entry point inside the snapshot before scheduling any GPU work. This is the second defect in a row that only a real execution path exposed, after the missing explanation_channel case in build_command; neither bash -n nor --dry-run can catch either.

## 2026-07-28 — EXP-R2-052: freeze scope repaired; preflight false positive

The code freeze is now import-derived rather than a fixed directory list. freeze_manifest takes src/transfer and scripts/transfer as a baseline, then statically parses every file with Python's ast (no execution, since the controller host has no torch or transformers) for src.* imports and resolves the transitive closure, including every __init__.py on the path. Handling relative imports was required, since the package imports src.revision.* that way throughout; the agent's first pass handled only absolute imports and silently missed statistics.py and nested_recoverability.py, caught by checking the discovered set against the named list. The closure discovers src/__init__.py, src/revision/__init__.py, io.py, statistics.py, dictionary_fidelity.py, nested_recoverability.py and — transitively through dictionary_fidelity.py — dictionary_controls.py: five real modules from four named. Snapshot is 34 files, up from 23, and the same file list feeds both the sha256 code hash and the push, so hash and snapshot can no longer drift apart. RUN_MANIFEST.json now records frozen_scope with closure_derived rather than a fixed frozen_directories list.

An import preflight now imports each of the nine wired entry points before any GPFS or GPU work, collecting all failures rather than stopping at the first.

Run 20260728152900_02f91a55c9e7 aborted at that preflight with "03_estimand_power.py: AttributeError: 'NoneType' object has no attribute '__dict__'". This is a FALSE POSITIVE: the same file, from the same snapshot, imports cleanly under the pod's own interpreter via spec_from_file_location plus module_from_spec plus exec_module, and also imports cleanly on the L20 host; only the preflight sees a failure, and only for 03 while the other eight pass. The signature is the classic one for sys.modules[name] having been set to None, which Python's import machinery uses as a negative-cache marker, so an earlier entry point is poisoning the shared interpreter for a later one — order-dependent rather than intrinsic to 03.

Fix requested: isolate each import in its own short-lived subprocess so the check tests "does this file import in a clean interpreter" rather than "does it import after eight others in the same one", plus a self-test using a fake entry point that imports alone but fails after another, since the existing three-way isolation test (ok, missing module, syntax error) covers only intrinsic failures and could not have caught a contamination class.

The preflight nonetheless did its job in the sense that mattered: it stopped before scheduling four GPUs rather than after, which is the opposite of the previous run's failure mode.

## 2026-07-28 — EXP-R2-053: induction path patching — the mediated fraction is the same in the matched pair

**Goal**: adjudicate the programme's central mechanistic hypothesis — that attention's contribution to next-token prediction is more *indirect* in protein decoders than in text, mediated through later MLPs rather than written to the unembedding — with the instrument that separates direct from mediated effects while holding the rest of the model fixed (path patching; Wang et al., ICLR 2023; Goldowsky-Dill et al., arXiv:2304.05969).

**New code**: `src/transfer/path_patching.py`, `scripts/transfer/11_induction_path_patching.py`, schema `r2_transfer_path_patching_v1`.

**Design, fixed before results were read**. Cases are (query, key) pairs of the natural-repeat probes; the corrupted input replaces the token at the key — the token a prefix-matching head would copy — with one other token from the arm's own unigram; the metric is `logit_q[T] - logit_q[T']`; denoising, so `recovery = (L_x - L_corrupt) / (L_clean - L_corrupt)`. The sender is one head at the read-out position, patched at the input of the attention output projection. `direct` freezes every attention module above the sender's layer and every MLP from it; `via_mlp` frees the MLPs; `via_attn` frees attention; `total` frees both. Eligibility: `L_clean - L_corrupt >= 0.25` logits.

**Command** (L20 validation scale, one GPU, float32):

```
python scripts/transfer/11_induction_path_patching.py \
  --arms gpt2-large protgpt2 progen2-medium zymctrl --device cuda:4 --dtype float32 \
  --cohort-size 96 --unigram-max-tokens 1024 --repeat-cohort-size 44 \
  --natural-max-tokens 840 --max-case-tokens 840 \
  --n-cases 48 --cases-per-probe 6 --case-batch-size 4 \
  --resolve-receivers --resolve-senders 4 --bootstrap-resamples 10000 \
  --output-dir results/transfer_20260728/path_patching
```

**Structural invariants** (all four arms, raised on failure, none raised): `null_patch` 0.0 exact, `identity_patch` 0.0 exact, `freeze_only` 0.0 exact, `resid_final_at_q` 1.0 exact, `resid_final_all_positions` 1.0 (6e-8), `position_locality` 1.5e-7, `readout_matches_model` 2e-6, `head_write_linearity` 2.4e-6.

**Result — the matched pair does not support the hypothesis.** Per-sender-head mediated fraction under the exact criterion: ProtGPT2 0.559 (14 heads) against gpt2-large 0.539 (24 of 50 heads above the ratio floor), difference +0.020, bootstrap 95% CI [-0.121, +0.166], which includes zero. Under the approximate criterion the difference reverses: ProtGPT2 -0.474 against gpt2-large 0.506, CI [-1.279, -0.660]. Neither reading confirms a larger mediated share in the protein arm of the controlled pair. On the *effect* scale ProtGPT2's induction heads are stronger on both paths (direct +0.020, mediated +0.034, both CIs excluding zero), so the protein arm's heads do more, in the same proportions.

**Result — the cross-tokenisation arms disagree with each other.** ProGen2-medium's mediated fraction is 0.864 exact / 0.777 approximate against gpt2-large's 0.539 / 0.506 (CIs exclude zero), and its mediation is concentrated in the last two MLP blocks (L25-L26 of 27). ZymCTRL goes the other way (0.126 exact) on two usable heads. Both are directional only: they cross a tokenisation boundary, and ZymCTRL has no head above threshold under either criterion, so it entered on top-8 heads flagged as *not* induction heads by the panel's criterion.

**Sender-set stability**: the 2x2 of sender criterion by case criterion shows the exact/approximate difference is carried by the *case set*, not the sender set. ProtGPT2 reads 0.559 / 0.732 on exact cases under the two sender sets and -0.172 / -0.474 on approximate cases. gpt2-large is stable at 0.506-0.551 across all four cells.

**Why the approximate case set behaves differently**: mean `L_clean - L_corrupt` falls from 19.6 to 5.0 logits for ProtGPT2 and from 20.0 to 13.7 for gpt2-large, tracking the probe-coverage collapse (0.718 -> 0.299 and 0.945 -> 0.427). The matched pair is well matched on the exact case set (19.99 against 19.58 logits) and is not on the approximate one.

**ZymCTRL negative**: mean corruption effect 0.47 logits and 54% of cases eligible, against 20 logits and 100% for gpt2-large. Swapping a copied residue barely moves ZymCTRL's logit difference at all.

**Limitations**: validation scale (44-record repeat cohorts, 48 cases, one seed); cases drawn from only 8-17 distinct records per arm at `--cases-per-probe 6`; the sender node is the head at the read-out position only, so attention mediation sourced at other positions is out of scope; the ratio floor withholds 26 of 50 gpt2-large heads from the fraction mean, and the floor ladder in each artefact shows gpt2-large's fraction moving 0.234 -> 0.574 across floors while ProtGPT2's sits at 0.559 throughout. Full scale belongs on H200.

## 2026-07-28 — EXP-R2-053 addendum: coordinator reading of the path-patching result

The mediation hypothesis is not confirmed, and one of the two surviving findings is downgraded as a result.

Matched pair, mediated fraction: exact criterion gpt2-large 0.539 against ProtGPT2 0.559, difference +0.020 with CI [-0.121, +0.166] including zero; approximate criterion the difference reverses to -0.980, CI [-1.279, -0.660]. Neither shows a larger mediated share in the protein arm of the controlled pair.

What is supported instead: ProtGPT2's induction heads are stronger on both paths, direct +0.020 and mediated +0.034 with both CIs excluding zero. Its heads do more, in the same proportions. That is consistent with the original ablation observation without supporting the mechanism proposed for it.

Consequence. EXP-R2-032 described "attention's contribution is more indirect in protein decoders" as supported by three converging measurements: the ablation-minus-DLA gap, the copy-rank dissociation and the activation-patching shortfall. Path patching is the instrument built to test that claim directly, and in the controlled comparison it disagrees. The convergence was of three indirect proxies, none of which isolated the mediated path. The ablation-versus-DLA gap remains real and unexplained; the claim that the share of attention's effect running through later MLPs is larger in protein decoders does not survive.

ProGen2-medium does show the predicted pattern (0.864 against 0.539, CI excluding zero, mediation concentrated in L25-L26 of 27) but is cross-tokenisation and directional only, and ZymCTRL goes the opposite way at 0.126. Two cross-tokenisation arms disagreeing with each other is the signature of a tokenisation and coverage confound.

The exact row is the one to read: the exact-versus-approximate divergence is carried by the case set rather than the sender set (top-4 ProtGPT2 senders identical under both criteria, gpt2-large stable at 0.506-0.551 across all four cells of the 2x2 cross), and mean corruption effect falls 19.6 to 5.0 logits for ProtGPT2 and 20.0 to 13.7 for gpt2-large, tracking the coverage collapse. The matched pair is well matched on exact cases (19.99 against 19.58 logits) and is not on approximate ones.

Caveats keeping the null soft: the ratio floor withholds 26 of 50 gpt2-large heads, and across floors 0.001 to 0.05 gpt2-large moves 0.234 to 0.574 while ProtGPT2 sits at 0.559 throughout, so gpt2-large's fraction is floor-sensitive and ProtGPT2's is not; and effect-weighted aggregates do favour the protein arm (0.581 against 0.360) but weight by effect size, which is the finding rather than independent support. Per-head totals are 1-6 per cent of the clean/corrupt gap.

All eight structural invariants passed exactly, including head-write linearity at 2.4e-6. ZymCTRL is a clean negative: mean corruption effect 0.47 logits with 54 per cent of cases eligible, against 20 logits and 100 per cent for gpt2-large.

## 2026-07-28 — EXP-R2-054: full nine-stage campaign completes; text side found to be n=1

Run 20260728164511_b8642d4bf8da completed all nine stages on 4x H200 with no failures, after the dtype audit removed every unjustified --dtype override (exactly one measured exception retained: float32 for ProGen2-medium's cohort_power, whose truncation statistic is host-bound). Import preflight and the new argparse-construction preflight both passed for all nine entry points. Two defects were fixed to get here: an API drift where 07_convergence_control.py still called protein_repeat_cohort and text_repeat_cohort with the loose min_unit keyword after circuits.py replaced those with a RepeatCriterion value, repaired by passing dataclass_replace of the named exact criteria so the CLI flags keep working; and the worker forcing --dtype bfloat16 on 08_lens_family.py, whose float32 default is deliberate because lens quantities are differences between near-identical distributions. The latter surfaced as a FloatingPointError from the Jacobian finite-difference guard at relative error 1.008 against a 2e-2 tolerance, so the guard refused to emit wrong Jacobians rather than silently producing them. Neither defect was reachable by bash -n, --dry-run or the import preflight; the argparse preflight would not have caught the API drift either, which is documented in the worker.

Design asymmetry identified, and it limits the generalisation of the headline finding. The protein side spans three arms differing in corpus, tokeniser and architecture (ProtGPT2 GPT-2-architecture multi-residue BPE on UniRef50; ZymCTRL CTRL-style residue-level on an EC-labelled corpus; ProGen2-medium GPT-J-style residue-level on UniRef90 plus BFD). The text side, for every measurement needing the circuits capability, is gpt2-large alone: the other text ladder rungs are one architecture, tokeniser and corpus lineage at different scales, and the ByGPT5 rungs carry budget and lens capability only. The claim that protein decoders have about three times fewer induction heads than the matched text control is therefore equally consistent with gpt2-large having unusually many, and this is the mirror image of the n=1 subword-protein cell recorded in EXP-R2-036 — but unlike that one it is fixable, since text decoders are plentiful. Every cross-modality number now on GPFS carries this limitation.

Being repaired with models rather than caveats. dialogpt-small added: GPT-2's architecture, tokeniser and size (12 layers, width 768, 50,257 vocabulary) pretrained on conversational Reddit threads rather than WebText, paired with gpt2 as TEXT_DATA_CONTRAST — the exact text-side analogue of MATCHED_DATA_CONTRAST, so that comparing the two contrasts bounds how much of any cross-modality difference is corpus rather than modality. Verified loading with circuits capability, 12 blocks, GPT2MLP and GPT2Attention resolving. Architecturally diverse Qwen2, Qwen3 and Llama arms are in progress, with grouped-query attention and rotary position embeddings to be verified rather than assumed; circuits capability is to be withheld rather than faked if per-head OV decomposition cannot be handled without editing circuits.py.

Until the diversified text side reports, the correct statement of the headline is that ProtGPT2 has about three times fewer induction heads than gpt2-large in a pair matched on depth, width, vocabulary size and tokenisation — a claim about two models, not two modalities.

## 2026-07-28 — EXP-R2-055: text-side diversification; circuits capability withheld pending module work

Two architecturally diverse base text decoders added to PANEL: qwen2.5-0.5b (Qwen2, 24 layers, width 896, vocab 151,936) and llama-3.2-3b (Llama, 28 layers, width 3072, vocab 128,256), plus dialogpt-small (GPT-2 architecture and tokeniser, Reddit corpus) paired with gpt2 as TEXT_DATA_CONTRAST. Two labs, two byte-level BPE vocabularies 2.5-3x GPT-2's, two corpora, scale extending above gpt2-xl. TEXT_ARCHITECTURE_CONTRAST added.

Three nominated models rejected on a criterion the coordinator had missed: Qwen3-0.6B, Qwen3-1.7B and Llama-3.2-1B-Instruct are all post-trained, their own model cards recording SFT/DPO and thinking-mode chat templates, while every other panel member is a pure next-token pretrained decoder. Admitting one would confound the architecture and corpus contrast with a training-objective contrast that no ArmSpec field records, and would do so on the convergence axis' own denominator since post-training moves cross-entropy on raw web text. No Qwen3 base checkpoint is staged. Qwen2.5-3B is a base checkpoint and would give a scale-matched cross-lab pair with llama-3.2-3b.

Grouped-query attention handled and measured: replicating each key/value head's W_V across its query group reproduces the layer output at max relative error 4.67e-3 (Qwen2, 14 query heads over 2 kv) and 4.13e-3 (Llama, 24 over 8) in bfloat16 against a 5e-2 tolerance, about 1e-6 in float32. Trap recorded: Qwen2 carries a v_proj bias and dropping it gives 3.8e-1, 7.6x over tolerance, while Llama has no attention biases.

circuits capability WITHHELD for both new arms, and grouped-query attention is not the reason. circuits.py resolves ln_1, transformer, c_attn and a learned embedding table, none of which exist on a rotary decoder. Four changes dispatched to that module's owner: a q_proj/v_proj/o_proj branch with the GQA index mapping, taking d_head from config.head_dim where present rather than d_model over n_head since those diverge for Qwen3; per-architecture pre-attention norm resolution plus folding v_proj.bias; per-architecture embedding resolution; and an RMSNorm linearisation in direct logit attribution, since RMSNorm has no bias and does not mean-centre so the existing shift and centred terms are wrong rather than absent. Qwen has no BOS token while Llama auto-prepends one, which the prefix and content-bounds logic both assume away.

Consequence: the new arms strengthen budget, pathway and convergence only and do not yet answer the generalisation objection. pathway was verified end to end (clean CE 2.9108 / 3.0503 / 2.5264 nats and mlp_all deltas 7.4165 / 6.5707 / 6.8921 for gpt2-large, qwen and llama on a shared cohort). The decisive question, whether gpt2-large's induction head fraction of 0.0694 is typical of text decoders or an outlier, cannot be asked until circuits.py reaches them.

Regression clean: all 11 PANEL arms load, blocks() length matches declared n_layer everywhere, gpt2-large identity-checked unchanged, ByGPT5 still refuses sublayers, 270 tests plus 6 subtests pass. Two smaller defects recorded: 08_lens_family.py defaults --arms to sorted(PANEL) with no capability guard, already broken by ByGPT5 and now enlarged; and ArmSpec.source is the evaluation cohort source rather than the pretraining corpus, so carrying pretraining corpus as a fit covariate needs a coordinated arms.py and scaling.py change.

## 2026-07-28 — EXP-R2-056: text-side generalisation test; deficit survives, magnitude overstated

circuits.py extended to rotary decoders (q_proj/v_proj/o_proj branch with the grouped-query index mapping and head_dim taken from config where declared; per-architecture pre-attention norm; tied-embedding resolution; explicit RMSNorm linearisation). Induction census run on four text arms with identical settings.

Fraction of heads above 0.10 prefix matching, synthetic exact probe: gpt2-large 0.0972 (720 heads, 89.0x uniform), qwen2.5-0.5b 0.0714 (336, 89.3x), dialogpt-small 0.0694 (144, 66.0x), llama-3.2-3b 0.0387 (672, 86.4x); ProtGPT2 0.0181 (720, 49.9x), ProGen2-medium 0.0139 (432, 87.6x), ZymCTRL 0.0000 (720, 2.8x).

The objection was well founded: gpt2-large is the highest of the four text arms at 1.62x the mean of the other three, so any coefficient computed against it alone overstates the gap and the roughly threefold matched-pair figure must not be quoted as the effect size. But the ranges do not overlap — text 0.0387-0.0972 against protein 0.0000-0.0181 — so the worst text decoder still carries 2.14x the fraction of the best protein decoder, with mean text over mean protein at 6.50x. The deficit survives replacing gpt2-large with a Qwen2 and a Llama from different labs, corpora and architectures with vocabularies three times larger.

Two qualifications. The separation is prevalence, not capability: ProGen2-medium's best head reaches 87.6x uniform against gpt2-large's 89.0x, so "protein models lack induction heads" would be wrong. And llama-3.2-3b is both the lowest text arm and the widest and deepest, while the fraction normalises by head count but not by scale, so scale and modality are not separated in this table and the 2.14x floor is a range statement rather than a controlled one.

Retraction. The ProtGPT2 QK/OV dissociation reported in EXP-R2-032 and read as a modality finding is substantially a GPT-2-lineage property. Top-head OV copy rank: gpt2-large 0.960, dialogpt-small 0.704, qwen2.5-0.5b 0.357, llama-3.2-3b 0.445, ProtGPT2 0.361, ProGen2-medium 1.000. Qwen2 and Llama sit at or below chance alongside ProtGPT2 while ProGen2-medium copies perfectly, so "an induction head that also copies" is not a universal text property. Panel-level means still separate (text 0.53-0.60, protein 0.30-0.49) but the cross-modal reading is withdrawn. This is the second of EXP-R2-032's two findings to fall to a proper control, after the mediation mechanism in EXP-R2-053.

Verification per architecture at bfloat16 against a 5e-2 tolerance: OV rebuild 3.26e-3 gpt2-large, 3.78e-3 dialogpt-small, 4.11e-3 qwen2.5-0.5b, 5.69e-3 llama-3.2-3b; about 3e-7 in float32. Two traps behave oppositely and both are now recorded: omitting Qwen2's v_proj bias gives 0.518, ten times over tolerance and fails loudly, while applying LayerNorm's algebra to an RMSNorm decoder passes the reconstruction gate at 0.49 per cent logit error inside a 2 per cent tolerance while producing a systematically wrong attribution. A reconstruction gate is therefore not sufficient protection against the wrong norm algebra, and the explicit normalisation-form resolution is load-bearing.

circuits capability now declared for the rotary decoders in arms.py on the strength of that verification, replacing the runner's explicit recorded --grant-circuits override. Regression clean: gpt2-large reproduces to 2.4e-5, ProGen2-medium bit-identical. Bug fixed, surfaced by the large vocabularies: direct_logit_attribution materialised the entire unembedding in float32 (1.47 GB for Llama) when only one row per scored position is read.

## 2026-07-28 — EXP-R2-057: threshold robustness and scale separation; the deficit survives but shrinks to ~2.3x

New code: `src/transfer/induction_robustness.py`, `scripts/transfer/12_induction_robustness.py`, `scripts/transfer/13_induction_probe_bootstrap.py`, `tests/test_induction_robustness.py` (18 tests, all pass). New artefacts under `results/transfer_20260728/induction_robustness/` and `results/transfer_20260728/circuit_primitives_text_ladder/`.

**Harvest first.** Every one of the seven arms already carried the full sweep on disk -- `count_above_threshold` at 0.05/0.10/0.20/0.30 plus `count_above_data_driven` -- on all three probes, so nothing in tasks 1 and 2 needed a new measurement. Text arms come from `circuit_primitives_text_control`, protein arms from `circuit_primitives_approximate`; gpt2-large appears in both runs and reproduces exactly (identical counts at every cut, max absolute per-head difference 7.3e-4). The analysis re-derives every count from the stored `induction.per_head` matrices and refuses to proceed if a recomputed count disagrees with the stored one; none did, for any arm or probe.

**Threshold robustness on the synthetic probe: the ordering holds everywhere, the magnitude does not.** Fraction of heads above each cut, worst text over best protein in the last column:

| cut | gpt2-large | qwen2.5-0.5b | dialogpt-small | llama-3.2-3b | ProtGPT2 | ProGen2-medium | ZymCTRL | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.1500 | 0.1905 | 0.1389 | 0.1071 | 0.0194 | 0.0162 | 0.0000 | 5.51 |
| 0.10 | 0.0972 | 0.0714 | 0.0694 | 0.0387 | 0.0181 | 0.0139 | 0.0000 | 2.14 |
| 0.20 | 0.0486 | 0.0446 | 0.0417 | 0.0238 | 0.0139 | 0.0116 | 0.0000 | 1.71 |
| 0.30 | 0.0292 | 0.0387 | 0.0347 | 0.0179 | 0.0097 | 0.0116 | 0.0000 | 1.54 |
| mean+3sd | 0.0236 | 0.0327 | 0.0347 | 0.0223 | 0.0181 | 0.0116 | 0.0097 | 1.24 |

The modality ranges are disjoint at all five, but the ratio falls monotonically as the cut rises and the per-arm data-driven cut -- mean plus three standard deviations of that arm's own head distribution, which standardises each arm by its own dispersion -- nearly closes it at 1.24. Swept continuously rather than at four points, the ordering holds for every raw cut in one contiguous band 0.00525-0.4196, which is 62 per cent of the informative grid and contains all four fixed cuts in its interior; the largest ratio anywhere is 8.07 at a cut of 0.0205. **The 0.10 choice is not load-bearing and is not the most favourable one available.**

**Threshold-free statistics: the finding is a tail statement and only a tail statement.** A Mann-Whitney AUC over the full per-head score distributions does NOT separate the modalities: pooled 0.595 raw and 0.589 in multiples of the uniform baseline, pairwise range 0.437-0.689, and only 8 of 12 text-against-protein pairs exceed 0.5. Every dialogpt-small pair is below 0.5. No pair shows stochastic dominance; the survival curves cross in all 12. The reason is visible in the quantiles: protein heads cluster tightly at chance (q90 of 0.80-1.04x uniform) while text heads spread out in BOTH directions (q25 as low as 0.02x for dialogpt-small, q90 4.77-8.37x). Quantile dominance separates the modalities at q75 through q99 and fails at q50, q99.5 and q100 -- at the maximum ProGen2-medium's best head (87.6x uniform) beats every text arm but gpt2-large and qwen. The one-sided KS statistic, which maximises the prevalence gap over the cut instead of fixing one, is positive for all 12 pairs at D+ 0.211-0.407. Model-level exact permutation, 4 text against 3 protein, C(7,3)=35 assignments and a floor of p=0.0286: the fraction, the mean head score and q99 all give complete separation at the floor, and the **median gives p=0.257 with no separation at all**. Prevalence, not central tendency, and the AUC is the wrong instrument for it.

**Probe dependence is a genuine negative and is not resolved.** The same analysis on the two natural-repeat probes:

| probe | ordering at every fixed cut | breaks at | model-level p (fraction) | dialogpt-small vs ProtGPT2 |
|---|---|---|---:|---|
| synthetic | yes | -- | 0.0286, complete | holds |
| natural exact | no | 0.20, 0.30 | 0.0286, complete | holds |
| natural approximate | no | 0.10, 0.20, 0.30 | 0.114, not complete | **fails** |

On the approximate natural probe dialogpt-small has zero heads above 0.10 while ProtGPT2 has 0.0097, so the modality ordering inverts at the headline cut. This matters more than it looks: `circuits.py`'s own docstring says the synthetic probe is off-distribution for a protein decoder and "for a wrapped arm the natural-repeat score is the one to trust", and `scaling.py` declares `PRIMARY_INDUCTION_PROBE = "natural"` -- yet the seven-arm headline table is the synthetic one. The synthetic probe is however the only probe on which the seven arms are matched by construction (built in each arm's own token space, uniform baseline 0.0107 for every arm but ZymCTRL's 0.0098), while the natural probes draw text repeats from OpenWebText at 141 symbols and 0.94 coverage against protein repeats from Swiss-Prot at 30 symbols and 0.71-0.96 -- so the natural comparison is confounded by cohort in a way the synthetic one is not. Both readings are now on the record; the claim is probe-conditional and should be stated as such.

**Bootstrap, done with the right unit.** `13_induction_probe_bootstrap.py` recomputes the census retaining the per-probe axis, which the stored artefacts cannot support because the census averages over probes before writing, and resamples PROBES as clusters. Layers are not resampled -- a layer is a coordinate of the model. Heads are not resampled -- within one model they are the entire population, and an interval around a within-model head fraction would have no estimand. The cross-modality contrast is taken at the model level by exact permutation instead. Fractions at 0.10 with probe-cluster intervals: gpt2-large 0.0972 [0.0944, 0.1000], qwen2.5-0.5b 0.0714 [0.0714, 0.0744], dialogpt-small 0.0694 [0.0694, 0.0694], llama-3.2-3b 0.0387 [0.0372, 0.0432], ProtGPT2 0.0181 [0.0181, 0.0181], ProGen2-medium 0.0139 [0.0139, 0.0139], ZymCTRL 0.0000 [0.0000, 0.0000]. Probe-sampling noise is negligible and the worst-text interval does not touch the best-protein one. The recomputation reproduces the stored census exactly for six arms; llama-3.2-3b differs by one head of 672 (1.49e-3), bfloat16 non-determinism. Several intervals are degenerate, which is the correct answer when every head sits far from the cut, not a failure -- asserted as a test.

**Within-lineage scale ladder: a scale account is NOT refuted on the text side.** Census run on gpt2, gpt2-medium and gpt2-xl at exactly the settings the existing seven were measured at (script defaults, seed 20260728, bfloat16, `--sections induction`; text repeat cohorts reproduce to the record -- 116 exact and 968 approximate matches of 3000 eligible). Architecture, the 50257-piece BPE and the WebText corpus are held fixed across the four rungs:

| arm | parameters | n_layer | heads | heads above 0.10 | fraction |
|---|---:|---:|---:|---:|---:|
| gpt2 | 124,412,160 | 12 | 144 | 23 | 0.1597 |
| gpt2-medium | 354,749,440 | 24 | 384 | 53 | 0.1380 |
| gpt2-large | 773,891,840 | 36 | 720 | 70 | 0.0972 |
| gpt2-xl | 1,557,380,800 | 48 | 1200 | 100 | 0.0833 |

The COUNT rises from 23 to 100 while the FRACTION falls by 1.92x. Log-log slope -0.272 per decade of parameters, interval [-0.455, -0.088], excluding zero. So the fraction does fall with scale with lineage controlled, and llama-3.2-3b's low value is no longer anomalous. Every scale variable is collinear inside a lineage, so this identifies scale as a bundle and attributes it to no component.

**The deficit must therefore be restated against a scale-matched expectation, and it survives that restatement at about 2.3x.** The ladder predicts what each remaining arm should score at its own size (95% prediction interval, 2 residual dof); every one of the seven falls below its interval, so the shortfall ratio is the scale-adjusted statistic:

| arm | modality | predicted | prediction interval | observed | shortfall |
|---|---|---:|---|---:|---:|
| qwen2.5-0.5b | text | 0.1147 | [0.0780, 0.1688] | 0.0714 | 1.61 |
| llama-3.2-3b | text | 0.0690 | [0.0410, 0.1161] | 0.0387 | 1.78 |
| dialogpt-small | text | 0.1668 | [0.1054, 0.2640] | 0.0694 | 2.40 |
| ProtGPT2 | protein | 0.1016 | [0.0684, 0.1509] | 0.0181 | 5.62 |
| ProGen2-medium | protein | 0.1019 | [0.0686, 0.1513] | 0.0139 | 7.34 |
| ProGen2-base | protein | 0.1019 | [0.0686, 0.1513] | 0.0069 | 14.67 |
| ZymCTRL | protein | 0.1040 | [0.0702, 0.1539] | 0.0000 | infinite |

Complete separation on the scale-adjusted statistic: worst text 2.40 against best protein 5.62, a ratio of **2.34x**, exact one-sided p=1/35=0.0286 at the floor. That is the number the finding should be quoted at. It is not the 6.50x mean ratio and not the 2.14x range floor; those were the same effect measured without a scale reference.

**Corpus alone accounts for most of a modality-sized effect.** Two pairs holding architecture, parameter count, shape and tokeniser fixed and varying only the pretraining corpus: gpt2 0.1597 against dialogpt-small 0.0694 is **2.30x**, and ProGen2-base 0.0069 against ProGen2-medium 0.0139 is **2.00x** (ProGen2-base censused here for the first time). So a corpus change inside one modality moves the fraction by about the same factor as the entire scale-adjusted modality residual. Any statement of the modality effect that does not quote 2.30x beside it is overstating what is modality-specific.

**The n=7 and n=11 regressions, descriptive only.** `fraction ~ modality + scale`, one scale covariate at a time. Contrary to the concern that motivated this task, modality and scale are NOT collinear in this panel: correlations run -0.097 to +0.486 and every VIF is at most 1.31, so the fits are readable rather than degenerate. Seven-arm protein offsets: -0.0607 [-0.0950, -0.0265] with d_model, -0.0571 [-0.0989, -0.0152] with log10 parameters, -0.0583 [-0.1078, -0.0088] with head count, -0.0627 [-0.1142, -0.0111] with depth. Adding the four new rungs (n=11, 8 residual dof) moves them to -0.077 to -0.082 with every interval still excluding zero and every VIF at most 1.06. Every fit records `inferential = False`: the arms were selected for the contrast under test, so these intervals describe how tightly eleven points pin a line and are not confidence statements about decoders in general. The point prediction above is the stronger instrument and is the one to lead with.

**Scale-inversion checks, both verified on the synthetic probe.** The matched pair gpt2-large and ProtGPT2 have identical architecture, depth, width, head count, vocabulary and parameter count -- 773,891,840 on both, measured from the checkpoint headers -- and differ 0.0972 against 0.0181, a factor 5.38 that is not a scale difference by construction. The inverted pair dialogpt-small (124,412,160; 12 layers; 144 heads) against ProtGPT2 (6.2x larger) holds the modality ordering at 0.0694 against 0.0181 with the SMALLER model higher, which is the sign a pure scale account forbids. Both fail on the approximate natural probe, where dialogpt-small reads zero.

**Ladder extended.** `DEFAULT_LADDER` in `src/transfer/scaling.py` now carries dialogpt-small, qwen2.5-0.5b and llama-3.2-3b. qwen2.5-0.5b is then excluded by the pre-existing padded-vocabulary rule -- 151,936 logit columns against a 151,665-token tokenizer, a 271-column surplus over the 64-column allowance -- and the rule was left alone rather than widened to admit it, since moving a criterion to include a rung is the failure mode this codebase warns about in the same module. Text side of the fit therefore goes from four GPT-2 rungs to six members of which two are outside the lineage.

**Infrastructure defect found, and it invalidates one H200 result.** `R2_TEXT_MODEL_BASE_DIR` in `scripts/transfer/h200_env.sh` pointed at the `transfer_gap` tree, which holds only gpt2-large, while every by-name text checkpoint had been staged under the `models` tree beside the protein arms. The mismatch is silent by construction, because `inspect_member` is written to tolerate partial staging and records a missing rung as unavailable rather than failing. The consequence: the convergence control in the completed nine-stage H200 campaign fitted `ladder_used = [gpt2-large, protgpt2, zymctrl, progen2-small, progen2-base, progen2-medium]` -- **a text side of one model**, which is exactly the n=1 defect that campaign existed to remove. The twelve-member L20 run of the same stage is the one its published numbers came from. Path corrected; DialoGPT-small staged to GPFS and verified by SHA-256, Qwen2.5-0.5B and Llama-3.2-3B in transfer.

**Compute.** All census and bootstrap work on one L20 at bfloat16: peak 0.9 GiB (gpt2), 2.0 (gpt2-medium), 6.4 (gpt2-xl), 2.8 (progen2-base); GPU 2 held 29.2 GiB of other users' work before and 29.7 GiB after, host memory 93/503 GiB throughout. The convergence control could not run locally -- the machine was carrying seven other jobs and the largest free block was 16.4 GiB against a float32 requirement of roughly 20 GiB for gpt2-xl and 32 for llama-3.2-3b, and a first attempt died with `torch.OutOfMemoryError` at gpt2-xl after completing gpt2, gpt2-medium and gpt2-large. It moves to H200 (4 idle cards, 143,771 MiB each, 0 per cent utilisation at check time).

**Addendum — the scale law is probe-invariant, the scale-adjusted deficit is not.** The within-lineage log-log slope reproduces on all three probes: -0.272 [-0.455, -0.088] synthetic, -0.277 [-0.322, -0.232] natural exact, -0.294 [-0.640, +0.052] natural approximate. Four rungs, two residual degrees of freedom, so only the middle interval is tight, but the point estimates agree to within 8 per cent and the direction is the same. That is a real regularity: the induction-head *fraction* falls by roughly a factor 1.9 per decade of parameters within one lineage while the *count* rises.

The scale-adjusted modality separation does not survive the change of probe:

| probe | worst text shortfall | best protein shortfall | separation |
|---|---|---|---|
| synthetic | dialogpt-small 2.40 | ProtGPT2 5.62 | 2.34x, complete |
| natural exact | dialogpt-small 3.43 | ProtGPT2 3.69 | 1.08x, marginal |
| natural approximate | dialogpt-small infinite (0 heads) | ProtGPT2 5.51 | **fails** |

and the corpus contrast tracks it: gpt2 against dialogpt-small is 2.30x on the synthetic probe, 3.40x on natural exact and unbounded on natural approximate, while ProGen2-base against ProGen2-medium is 2.00x, 2.00x and 1.00x. On the approximate natural probe a pure corpus change inside text produces a larger effect than the entire modality contrast, which is the plainest possible statement of the limit on this finding.

Reading. The claim that survives is narrow and should be written narrowly: on a synthetic token-space probe that is matched across arms by construction, protein decoders carry about 2.3 times fewer induction heads than a scale-matched and lineage-corrected text expectation, against a within-text corpus effect of 2.3 times measured on the same probe. On in-distribution natural repeats the same comparison is marginal under an exact criterion and absent under an approximate one. The programme's declared primary probe is the natural one.

---

## 2026-07-28 — EXP-R2-058: text-side panel staged to GPFS; full byte-level validation of all eleven checkpoints

Pure staging task, no GPU work, no repo code changes. Closes the gap the EXP-R2-057 addendum flagged ("DialoGPT-small staged to GPFS and verified by SHA-256, Qwen2.5-0.5B and Llama-3.2-3B in transfer") and makes `models/` on GPFS carry the complete eleven-checkpoint panel the convergence-control and scale-ladder work depends on.

**Pod.** `damoxing-zhk-zipbio-master-0` (running, 4x H200, 0 MiB / 0% used at both start and end of the session) selected per the "contains `zip`, not the `0gpu` one" rule; the excluded candidate was `jiaotongdamoxing-zhk-zip-npj-revision-0gpu-0716-master-0`. GPFS free space 46,188,371,968 KiB (~44 TiB) confirmed via `df -T` inside the pod before starting, filesystem type `gpfs`.

**Transfer helper behaviour.** `h200_gpfs_push.sh` only takes a single local file; `h200_sync.sh` handles a directory by tarring the whole tree locally, pushing the tar, and extracting remotely, with no include/exclude filter. Pushed `gpt2-large` **file by file** with `h200_gpfs_push.sh` (config.json, generation_config.json, generation_config_for_text_generation.json, merges.txt, tokenizer.json, tokenizer_config.json, vocab.json, README.md, model.safetensors — 9 files, ~3.25 GB) rather than tarring, specifically to drop the local `.cache/huggingface/download/*` metadata that `h200_sync.sh` would otherwise have carried along. `Llama-3.2-3B` did not need a push at all: `models/_staging/Llama-3.2-3B.tar` (6,434,703,360 bytes) was already present, already pod-verified, and already excluded the redundant 6.4 GB `original/consolidated.00.pth` (non-HF Llama format) and `.cache/` — this is the "Llama-3.2-3B in transfer" state the prior log entry described. Extracted with `tar --strip-components=1` and immediately removed the tar plus a stale `_staging/DialoGPT-small.tar` left over from an earlier completed push.

**gpt2-large push.** All 9 files verified SHA-256 at push time by the helper itself; the 3,247,159,078-byte `model.safetensors` additionally re-hashed independently after landing (`5f47f3e1...bebce`, matches the B-side source computed separately, not reused from the push script's own report). Took about 19 minutes end to end over the relay (local → Windows → master → pod → GPFS), dominated by the 13-part chunked upload of the safetensors file; concurrent unrelated traffic from another agent's `run_transfer_h200.sh STAGES=convergence_control` job on the same relay was visible in `ps aux` during this window but caused no conflict (disjoint target paths).

**Qwen2.5-0.5B extraction.** The tar's members were rooted at `Qwen2.5-0.5B/`, so a naive `tar -C models/Qwen2.5-0.5B -xf ...` produced a nested `models/Qwen2.5-0.5B/Qwen2.5-0.5B/`; caught immediately post-extraction, fixed with `mv */* .. && rmdir`, then all 7 file sizes checked against the B-side source (`stat -c %s`, exact match) before the tar was deleted.

**Full byte-level validation, all eleven checkpoints.** For every arm: listed GPFS files, confirmed the JSON config parses, ran `AutoConfig.from_pretrained(<gpfs-path>, trust_remote_code=True for progen2)` under `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` inside the pod (transformers 4.52.4, `/opt/ac2/bin/python3`), and checked every weight file's size against the B-side original. Went beyond size for every weight file — full SHA-256 on both sides, computed independently (source hashed directly on B, destination hashed inside the pod against the real GPFS mount, not against the push script's self-reported digest) — for `gpt2`, `gpt2-medium`, `gpt2-xl`, `gpt2-large`, `DialoGPT-small`, `ProtGPT2` (`pytorch_model.bin`), `ZymCTRL` (`pytorch_model.bin`), `progen2-base`, `progen2-medium`, and all 8 files of `Llama-3.2-3B`. Every hash matched.

| arm | GPFS path | present | size/hash match | config loads |
|---|---|---|---|---|
| gpt2 | `models/gpt2` | y | y (SHA-256) | y (gpt2, n_layer=12) |
| gpt2-medium | `models/gpt2-medium` | y | y (SHA-256) | y (gpt2, n_layer=24) |
| gpt2-large | `models/gpt2-large` | y (newly pushed) | y (SHA-256, incl. independent re-hash) | y (gpt2, n_layer=36) |
| gpt2-xl | `models/gpt2-xl` | y | y (SHA-256) | y (gpt2, n_layer=48) |
| DialoGPT-small | `models/DialoGPT-small` | y | y (SHA-256) | y (gpt2, n_layer=12) |
| Qwen2.5-0.5B | `models/Qwen2.5-0.5B` | y (extracted, nesting fixed) | y (byte size, all 7 files) | y (qwen2, n_layer=24) |
| Llama-3.2-3B | `models/Llama-3.2-3B` | y (extracted from pre-staged tar) | y (SHA-256, all 8 files) | y (llama, n_layer=28) |
| ProtGPT2 | `models/ProtGPT2` | y | y (SHA-256, `pytorch_model.bin`) | y (gpt2, n_layer=36) |
| ZymCTRL | `models/ZymCTRL` | y | y (SHA-256, `pytorch_model.bin`) | y (gpt2, n_layer=36) |
| progen2-base | `models/progen2-base` | y | y (SHA-256, `model.safetensors`) | y (progen, n_layer=27) |
| progen2-medium | `models/progen2-medium` | y | y (SHA-256, `model.safetensors`); `config.json` differs from the B-side source by one trailing-newline byte (1156 vs 1155) | y (progen, n_layer=27) |

No truncation, no corruption, no failed transfer. The `progen2-medium` `config.json` discrepancy is the only byte-level mismatch found across the whole panel: content is identical (confirmed by direct diff of both files, same `auto_map`, no residual repo-prefix), the GPFS copy simply carries a trailing `\n` the B-side original lacks. Does not affect JSON parsing or `AutoConfig` loading (both confirmed above) and is not a truncation signature — recorded rather than silently dropped, per the audit principle. gpt2-large now confirmed 36 layers / 1280-dim, matching ProtGPT2's architecture exactly, which is the depth/width-matched-pair property the programme's text control requires.

**Cleanup.** Removed the temporary `_validate_configs.py` helper and both consumed staging tars from `models/_staging/`; `_staging/` itself disappeared once empty (not investigated further — empty-directory GC or another agent, harmless either way). No files were left under `_staging/`. No `src/transfer/` or `scripts/transfer/` files were touched, per the constraint that other agents are active there.

**Addendum 2 — corpus and lineage contrasts with intervals, and the variance decomposition.** Elevated from a side-note on the coordinator's instruction, because on this axis specifically the corpus is not a nuisance covariate: an induction head is only useful on a corpus that repeats, and approximate repeats occur in 32.3 per cent of text documents against 0.402 per cent of protein entries. Ratios at the 0.10 cut with 95 per cent probe-cluster intervals (10,000 resamples, probes as clusters, the two arms resampled independently because probe *i* of one arm and probe *i* of the other are different token sequences sharing only a seed):

| contrast | arms | ratio | interval | held fixed |
|---|---|---:|---|---|
| corpus, text | gpt2 / DialoGPT-small | 2.30 | [2.30, 2.40] | architecture, 124,412,160 parameters, 12x768, GPT-2 BPE |
| corpus, protein | ProGen2-medium / ProGen2-base | 2.00 | [1.20, 2.00] | architecture, 764,754,464 parameters, 27x1536, residue tokenisation |
| lineage | gpt2-medium / qwen2.5-0.5b | 1.93 | [1.82, 2.04] | 24 layers; 355M against 494M |
| lineage | gpt2-xl / llama-3.2-3b | 2.15 | [1.92, 2.29] | nearest available scale; 1.56B against 3.21B |
| modality, matched | gpt2-large / ProtGPT2 | 5.38 | [5.23, 5.54] | architecture, 773,891,840 parameters, 36x1280, 50257 vocabulary |
| modality, inverted | DialoGPT-small / ProtGPT2 | 3.85 | [3.85, 3.85] | nothing; the text arm is 6.2x SMALLER |

Every nuisance contrast lands between 1.9 and 2.3. The modality contrast is 5.38, so it is between 2.3 and 2.8 times the largest nuisance contrast measured -- but it varies modality, corpus and repeat prevalence together and cannot separate them.

Variance decomposition on log10(fraction), ten arms (ZymCTRL dropped for having no logarithm, which understates every modality share below since it is the largest deficit in the panel). R-squared: scale alone **0.156**; modality alone 0.816; lineage alone 0.669; scale+modality 0.870; scale+lineage 0.711; full 0.931. Increments: modality given scale and lineage **+0.220**; lineage given scale and modality **+0.061**; **scale given modality and lineage +0.003**.

Two readings, and the second is the one that matters. First: scale explains essentially none of the between-arm spread once modality and lineage are in the design -- three tenths of one per cent -- which is the cleanest form of the answer to the question this task was set. Second: only **25.4 per cent** of the modality indicator's variance survives being projected off scale and lineage, because modality is separated from lineage only inside a family containing both, and the only such family is GPT-2 -- five text arms against **one** protein arm, ProtGPT2, once ZymCTRL is dropped. The 0.220 increment therefore rests on a single protein model.

**And none of this addresses the corpus-repeat confound.** Modality and pretraining-corpus repeat prevalence are perfectly confounded across every arm in existence: no protein decoder trained on a repeat-rich corpus exists and none can, since a synthetic protein corpus at 32 per cent repeat prevalence is not a protein corpus. A surviving modality coefficient is therefore equally consistent with a decoder allocating capacity efficiently against data in which prefix repetition almost never predicts the continuation. The scale analysis above settles size and settles nothing else, and this sentence belongs beside every number in this entry.

**Addendum 3 — convergence control re-fitted with the diversified text side.** `DEFAULT_LADDER` extended; run 20260729001915_dbbc56d2be7a on H200 (4x H200, 0 MiB used and 0 per cent utilisation before the run, 0 MiB after; pod host 167 GiB of 2.0 TiB). Fourteen members used against the baseline's twelve, and the text side of the induction fits goes from four GPT-2 rungs to five arms including llama-3.2-3b -- a different laboratory, architecture, corpus and a vocabulary 2.5x larger. Absent: qwen2.5-0.5b and progen2-large under the padded-vocabulary rule, progen2-xlarge for publishing no `vocab_size`. **dialogpt-small is excluded from every fit by the pre-existing denominator floor**: its realized information fraction on OpenWebText is -0.5345 at 3.7801 bits per symbol, so it does not beat that cohort's unigram baseline and is off its own distribution there. That is worth carrying into the corpus contrast above -- part of the 2.30x text corpus effect may be an off-distribution effect rather than a corpus one, and if so the nuisance bound is looser than measured, not tighter.

Induction protein offset, baseline (four GPT-2 text rungs) against diversified (five text arms):

| fit | baseline | diversified |
|---|---|---|
| synthetic fraction ~ log10 parameters | -0.1038 [-0.1255, -0.0821] | -0.0980 [-0.1222, -0.0737] |
| synthetic fraction ~ realized information | -0.1161 [-0.1568, -0.0755] | -0.1059 [-0.1613, -0.0504] |
| synthetic fraction ~ clean CE per symbol | -0.1278 [-0.1891, -0.0666] | -0.1204 [-0.2058, -0.0349] |
| natural fraction ~ log10 parameters | -0.0847 [-0.1042, -0.0651] | -0.0809 [-0.1003, -0.0616] |
| natural fraction ~ realized information | -0.0970 [-0.1375, -0.0565] | -0.0885 [-0.1385, -0.0385] |
| natural fraction ~ clean CE per symbol | -0.1086 [-0.1697, -0.0475] | -0.1024 [-0.1797, -0.0251] |

**The induction coefficient survives.** It shrinks by 5 to 9 per cent and the intervals widen, but all six still exclude zero with the text population diversified. The convergence slope on log10 parameters is -0.0659 (synthetic) and -0.0618 (natural), the same sign and comparable magnitude to the within-lineage ladder's -0.272 on a log response. The overall control verdict is unchanged at `underpowered`, which is a statement about the primary metric (`mlp_share_of_context_information`, whose modality coefficient excludes zero unadjusted at [-0.595, -0.199] and includes it tokenisation-adjusted at [-0.572, +0.103]), not about induction.

Baseline preserved at `results/transfer_20260728/convergence_control_baseline_l20_gpt2lineage/`; diversified result at `results/transfer_20260728/convergence_control_diversified/` (SHA-256 `71f90ce7a87e7ed58ee856603dfe25affb997b7f69a52dab335ff9e81b201a0b`).

**Addendum 4 — a second silent-staging defect, and its repair.** The first H200 attempt (run 20260728235051_ee40aa9ef601) dropped gpt2-large: correcting `R2_TEXT_MODEL_BASE_DIR` to the `models` tree moved the by-name text lookup there, and gpt2-large lived only under `transfer_gap`, addressed through the separate `R2_TEXT_MODEL_DIR`. The two variables must agree, because `scaling.register_arm_spec` refuses a ladder declaration whose path disagrees with the frozen panel declaration -- pointing them at different trees raises rather than silently miscomputing, but it still costs a scheduled run. Both now resolve to `${R2_GPFS_ROOT}/models/gpt2-large`, with the reason recorded next to them. gpt2-large staged into that tree; the copy collided with a concurrent stage of the same directory by another agent and was consolidated by hand, and the resulting `model.safetensors` is byte-count identical to the local file at 3,247,159,078 bytes. DialoGPT-small, Qwen2.5-0.5B and Llama-3.2-3B pushed with SHA-256 verified end to end; GPFS at 12 TB of 55 TB after.

---

## 2026-07-29 — EXP-R2-059: prediction-addressed attention gate — NO-GO for the census as specified

Blocking go/no-go check on the proposed **prediction-addressed attention (PAA) / copy-suppression census**, the candidate second isomorphic mechanism. Gates A0-A6 on the text control plus a cohort-power check on the protein arms. New code only: `src/transfer/prediction_addressed.py`, `scripts/transfer/14_paa_census.py`. Everything on one L20 (`cuda:2`, `cuda:4`), roughly 1.6 GPU-hours; no H200 work scheduled. Artefacts under `results/transfer_20260728/paa_gate/` and `.../paa_gate_extended/`.

### Gate 0 — cohort power, held-out unigram, 200 sequences per arm

Native cohorts; text >= 800 characters truncated to 512 tokens, protein 200-800 residues (ZymCTRL on the EC-labelled corpus, the others on plain Swiss-Prot). Baseline is the disjoint held-out estimator over a 4,000-record reference corpus deduplicated by content, never the plug-in.

| arm | held-out unigram | plug-in | plug-in bias | clean CE | context information | verdict |
|---|---:|---:|---:|---:|---:|---|
| gpt2-large | 7.5509 | 7.2500 | +0.301 | 2.6454 | **+4.906** | PASS |
| ProtGPT2 | 8.7968 | 7.5062 | +1.291 | 4.7496 | **+4.047** | PASS |
| ZymCTRL | 2.9040 | 2.8971 | +0.007 | 1.7944 | **+1.110** | PASS |
| ProGen2-medium | 2.9018 | 2.8882 | +0.014 | 1.7258 | **+1.176** | PASS |

All four clear 0.5 nats/token. The plug-in bias reproduces the vocabulary-tracking pattern of EXP-R2-036 (large for the 50k-piece arms, negligible for the residue arms), so using the plug-in here would have inflated exactly the two arms whose relative position matters.

**The 0.099-nat ProGen2 starvation figure is a file-order selection artefact, and this is now measured directly.** `protein_cohort` takes records in FASTA file order, and the first block is atypical:

| ProGen2-medium, plain Swiss-Prot 200-800 aa | context information |
|---|---:|
| n=48, skip=0 (the historical configuration) | **+0.163** |
| n=48, skip=48 | +0.684 |
| n=48, skip=400 | +2.039 |
| n=200, skip=0 | **+1.176** |
| n=200, skip=400 (disjoint block) | **+1.150** |
| n=200, 400-1500 aa band | +1.271 |
| n=200, EC-labelled corpus | +1.693 |

At n=200 the figure replicates on a disjoint block to within 0.03 nats. The band sweep the task asked for was not needed: widening to 400-1500 moves it by +0.10, whereas taking more than the first 48 records moves it by +1.01. This is the same hazard as the structure-selection order defect recorded in check.md §0B.2, in a different measurement.

### A1 candidate pool — PASS

200 OpenWebText documents at exactly 512 tokens, 96,000 scored positions. **71,434 instances** against a 20,000 floor. Median dependency distance 33 tokens, median p(X | t_<=q) 0.254.

**Hard exclusion D1 discards 8,506 candidates, a rate of 10.6%.** Those are positions where the nearest antecedent k* is also an induction target (t_{k*-1} == t_q). Without D1 the two censuses would share a tenth of their scored positions.

### A2 matching feasibility against ProGen2-medium — PASS

Coarsened exact matching on doubling distance bins, unigram-percentile deciles, six confidence bins and six absolute bins on the corruption effect (change in log p(X) when the antecedent token is replaced by a unigram draw). 1,500 instances measured per arm.

| gates | retained | retention | projected over the 71,434-instance pool |
|---|---:|---:|---:|
| distance | 1500/1500 | 1.000 | 71,434 |
| + unigram percentile | 1325/1500 | 0.883 | 63,100 |
| + confidence | 859/1500 | 0.573 | 40,908 |
| + corruption effect | 562/1500 | **0.375** | **26,764** |

Against a 2,000 floor, so A2 passes with an order of magnitude to spare.

One instantiation hazard is worth recording because it nearly produced a false negative. The decoy rule bans a decoy token from being one of the model's top-20 candidates at q. Over a 20-symbol alphabet that bans the alphabet: ProGen2's instance yield falls to 6.9% (87,427 of 94,008 candidates lost to an empty decoy pool), and the survivors are a biased high-confidence subset (median confidence 0.955 against 0.387 at a relaxed depth of 3). It still leaves 6,581 usable instances at 200 sequences, so the gate passes, but on an 8-sequence pilot it left 37 and read as a hard failure. Both ban depths are run and reported.

### A3 class non-emptiness — PASS, and the class is deeper than the screen said

Attention knockout on A_q(X) implemented pre-softmax, so the surviving keys renormalise. Verified exact: an all-zero injected mask reproduces the clean logits to a maximum difference of **0.0**. Run in float32 — in bfloat16 the M-gap change quantises to multiples of 1/512, which is the size of the effect.

600 instances over 200 sequences, cluster bootstrap over sequences, 1,000 replicates. 56 heads tested in two batches: top-16 and ranks 17-32 by decoy-corrected `paa_specific`, an 8-head raw-antecedent-attention screen, and 16 random control heads.

Five heads have a lower bound above 0.005 logits, and every one sits in the top quarter of depth:

| head | depth | ΔM-gap [95% CI] | Δp(X) | ΔM-gap, top decile by antecedent mass | Δp(X), decile | paa rank | prefix matching |
|---|---:|---|---:|---:|---:|---:|---:|
| L34H14 | 0.97 | **+0.0288** [+0.0189, +0.0404] | +0.0043 | +0.261 | +0.039 | 29 | 0.142 |
| L32H0 | 0.91 | +0.0231 [+0.0169, +0.0302] | +0.0039 | +0.201 | +0.035 | 18 | 0.215 |
| L26H0 | 0.74 | +0.0213 [+0.0159, +0.0274] | +0.0033 | +0.166 | +0.027 | 1 | 0.331 |
| L27H11 | 0.77 | +0.0159 [+0.0104, +0.0218] | +0.0021 | +0.128 | +0.017 | 2 | 0.221 |
| L30H0 | 0.86 | +0.0106 [+0.0059, +0.0168] | +0.0020 | +0.096 | +0.018 | 26 | 0.269 |

16 control heads: mean -0.00014, largest +0.0031. So the five clear the control band by 3.4x to 9.3x. The late-layer localisation matches the copy-suppression literature's L10H7 of 12 layers.

There is also a **boosting** population reading the same antecedents in the opposite direction, led by L20H11 (ΔM-gap -0.0117, -0.112 on its decile), which is a bona fide induction head at prefix matching 0.726. Prediction-addressed attention is not predominantly suppressive.

### A4 causal magnitude — the number, read off rather than thresholded

Top head L34H14: **ΔM-gap +0.0288 logits, Δp(X) +0.0043**. Mean clean M-gap over the scored instances is 0.683, so the pooled effect is 4.2% of the margin. On the decile of instances where the head most reads the antecedent it is **+0.261 logits and +3.9 percentage points of p(X)**. Statistically unambiguous, small in absolute terms.

### A5 dissociation from induction — PASS, stop rule not triggered

Over all 720 heads: Spearman(prefix matching, paa_specific) = 0.388, partial Spearman with each head's total non-sink attention mass removed = **0.364**; top-20 Jaccard = **0.212**. Both inside the stop thresholds of 0.5 and 0.3.

But the head-level picture is less clean than the population statistic. All five causally confirmed suppressive heads carry prefix matching between 0.142 and 0.331, so none is a pure PAA head, and the strongest booster is an induction head.

### A6 query-source intervention — does not dissociate the classes

At q, before layer l, the residual stream receives alpha x ||x_q|| x (u_Y - u_X) normalised to unit direction, Y absent from the context and frequency-matched to X; alpha in {0, 0.5, 1, 2}; 64 instances; a seeded random direction of identical norm is run alongside every alpha. Manipulation check is decisive: p(X) falls 0.319 to 0.0002 and p(Y) rises 0.0004 to 0.865.

| class | median relative change in antecedent mass | random-direction control | excess over control | monotone |
|---|---:|---:|---:|---:|
| PAA top-8 | -0.818 | -0.424 | **-0.441** | 7/8 |
| induction top-8 | -0.832 | -0.616 | **-0.208** | 8/8 |

The specified prediction was that induction heads' attention onto A_q(X) would *not* fall. It falls, monotonically, in 8 of 8 heads, and by -0.21 in excess of a matched random perturbation. The difference between the classes is quantitative (roughly 2x) rather than qualitative. Without the random-direction control the raw numbers would have looked identical for both classes (-0.82 against -0.83) and the intervention would have appeared to discriminate nothing at all; adding the control is what turns it into a 2x contrast. Reported as a failure of the gate as specified.

### The decoy correction applied to the existing induction census — headline safe, body not

Same 16 synthetic exact repeat probes, copy length 64, that produce the published induction numbers, with decoys drawn from the same distance bin under the same exclusions.

| statistic | uncorrected | decoy-corrected |
|---|---:|---:|
| peak prefix matching | 0.9553 | 0.9549 |
| heads >= 0.10 | 70 | 69 |
| fraction >= 0.10 | 0.0972 | 0.0958 |
| head mean | 0.0421 | 0.0367 |
| head **median** | 0.0083 | **0.0013** |
| top-20 Jaccard against uncorrected | — | **1.000** |

**The induction headline is not materially changed.** Peak moves by 0.0004, the threshold count by one head, and the top-20 set is identical. The published prevalence statistic is safe against this correction.

The body of the distribution is a different matter: the median head's apparent prefix matching falls sixfold, and the rank correlation between corrected and uncorrected scores over all 720 heads is 0.81, not 1.0. Most of a typical head's "prefix matching" is a positional baseline. Any future statistic that reads the distribution rather than its tail — a mean, a variance, a mass-weighted share — must be decoy-corrected. The count-above-threshold statistic survives because it lives in the tail, where the decoy term is 0.0004.

### Verdict: NO-GO for the campaign as specified

The mechanism is real. Five late-layer heads in gpt2-large suppress a token they address by prediction, with tight cluster-bootstrap intervals and a clean separation from random controls. A3 and A4 do not kill the design, and Gate 0, A1, A2 and A5 all pass.

The **census** is what fails, on three counts.

1. **The cheap statistic has no validity as a screen.** Spearman between `paa_specific` rank and measured ΔM-gap over the 40 screened heads is **-0.062 (p = 0.71)**; restricted to ranks 1-32 it is +0.03. Rank 1 gives +0.021, rank 29 gives +0.029, and rank 13 is a *booster* at -0.012. A prevalence census needs a per-head statistic it can threshold cheaply in every arm. This one ranks heads no better than chance, so the census would count the wrong heads. This is the blocking finding.
2. **A6 does not establish the query source.** The mechanism's claim to be isomorphic-but-independent rests on being addressed by the *predicted* token rather than the current one, and the intervention designed to show that separates the classes only 2x, with induction heads also responding monotonically.
3. **The positive control has no dynamic range.** Five heads clear the control band out of 56 tested, against induction's 70 of 720 above threshold. A cross-modal *prevalence* comparison whose text control sits at ~5 cannot distinguish a protein arm at 0 from one at 3.

Cost consequence, stated accurately because the first estimate written here was wrong. Because the screen does not work, the census would have to be a full causal knockout over every head in every arm. Measured: 24 heads in 12 minutes and 32 heads in 16 minutes at 600 instances in float32 on one contended L20, so about 0.5 minutes per head, or 6 L20-hours for all 720 heads of one gpt2-large arm, roughly 42 L20-hours over a seven-arm panel. At an H200 that is plausibly 8-15 GPU-hours for the knockout census plus the instance pools, corruption effects and nulls. **Cost is therefore not the blocking reason** -- a full census is about thirty times the screen-and-confirm design's compute but still fits inside the 30-35 H200 GPU-hour envelope. What blocks the campaign is that the census would have no valid cheap statistic to report and no dynamic range to report it over.

What the evidence would support instead, if this line is pursued: a full causal knockout census reported as an **effect-size distribution** over heads rather than a thresholded prevalence, on a reduced arm set, re-costed from the measured 0.5-minutes-per-head figure. Its headline would be a distribution, not a count, so it does not answer the prevalence question the track was proposed to answer. That is a different experiment and must be re-gated before scheduling.

### Compute and hygiene

L20 only. `cuda:2` and `cuda:4`, bfloat16 for Gate 0 and float32 everywhere an attention weight or a logit difference is read; peak about 6.3 GiB on the census and causal stages, both cards returned to 3 MiB after. Host memory 107/503 GiB throughout. No H200 work scheduled, no existing file under `src/transfer/` or `scripts/transfer/` modified.

Per-sequence and per-instance matrices are emitted, not only pooled means: `census_matrices.npz` carries per-sequence (200, 36, 20) matrices for `paa_specific`, antecedent attention, decoy attention and non-sink mass, plus per-probe prefix matching corrected and uncorrected; `causal_matrices.npz` carries per-instance (head, instance) ΔM-gap, Δp(X) and antecedent attention mass together with the per-cluster reduction and cluster weights, so every statistic above can be re-bootstrapped without a rerun.

---

## 2026-07-29 — EXP-R2-060: instrument-transfer campaign, eleven arms, four stages

The first campaign to run the transfer measurement package over the whole eleven-arm panel under one code path. Four stages -- `cohort_power`, `pathway_budget`, `estimand_power`, `lens_family` -- through `scripts/transfer/run_transfer_h200.sh`, four H200s, plus a skip-offset sensitivity re-run of `cohort_power`. Every arm fed in its native rendering by `Cohort.input_strings`.

**Runs.** Primary `20260729030034_d3646e37488d`, results `results/transfer_20260729_instrument/` (pulled locally, 40 JSON, 8.1 MB). Sensitivity `20260729033029_c7503e1da439`, `--cohort-skip 4000`, results `results/transfer_20260729_instrument_skip4000/`. A new results root was used rather than `transfer_20260728/` so the earlier four-arm artefacts survive intact. GPUs 0 MiB / 0 % inside the pod before and after both runs; pod host 167 GiB of 2.0 TiB.

**Panel resolution verified before launch.** All eleven arms resolve to distinct, existing GPFS directories under `models/`, with declared depth/width matching the loaded config; no duplicate paths. This was checked explicitly because `R2_TEXT_MODEL_BASE_DIR` had previously narrowed the text side to one model without any downstream number looking wrong (§0T.6).

### Stage 1 — cohort power, held-out unigram, seeded draw, n=200

| arm | kind | held-out unigram | plug-in | bias | clean CE | context info | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| llama-3.2-3b | text | 7.8232 | 7.4214 | +0.402 | 2.3898 | **+5.4334** | PASS |
| gpt2-xl | text | 7.5387 | 7.2096 | +0.329 | 2.5569 | +4.9818 | PASS |
| gpt2-large | text | 7.5387 | 7.2096 | +0.329 | 2.6674 | +4.8713 | PASS |
| qwen2.5-0.5b | text | 7.7280 | 7.3222 | +0.406 | 2.9005 | +4.8274 | PASS |
| gpt2-medium | text | 7.5387 | 7.2096 | +0.329 | 2.8280 | +4.7106 | PASS |
| gpt2 | text | 7.5387 | 7.2096 | +0.329 | 3.1374 | +4.4013 | PASS |
| **dialogpt-small** | text | 7.5387 | 7.2096 | +0.329 | 11.6215 | **-4.0828** | **FAIL** |
| protgpt2 | protein | 8.9394 | 7.6548 | **+1.285** | 5.6423 | +3.2971 | PASS |
| zymctrl | protein | 2.8793 | 2.8775 | +0.002 | 0.8506 | +2.0287 | PASS |
| progen2-medium | protein | 2.8975 | 2.8934 | +0.004 | 1.7906 | +1.1070 | PASS |
| progen2-base | protein | 2.8975 | 2.8934 | +0.004 | 1.8371 | +1.0604 | PASS |

Ten of eleven measurable. **DialoGPT-small is unmeasurable on the shared text cohort** and is unmeasurable the same way at all three later stages, so it carries no instrument-transfer information at all. That is consistent with the denominator-floor exclusion recorded in EXP-R2-058 addendum 3, now measured directly rather than inferred.

The plug-in bias reproduces its vocabulary-tracking pattern exactly: +0.33 to +0.41 on the 50k-to-152k-piece text arms, **+1.28 on ProtGPT2**, +0.002 to +0.004 on the residue arms. Using the plug-in would have understated ProtGPT2 by 1.28 nats against residue arms understated by 0.004 -- a 300-fold asymmetry on exactly the cross-arm comparison this stage exists to support.

### Stage 1b — skip-offset sensitivity, and a protein-specific hazard quantified

`--cohort-skip 4000`, a disjoint corpus block, everything else identical.

| arm | skip=0 | skip=4000 | delta |
|---|---:|---:|---:|
| gpt2 | +4.4013 | +4.3986 | -0.0027 |
| gpt2-medium | +4.7106 | +4.7161 | +0.0055 |
| gpt2-large | +4.8713 | +4.8838 | +0.0125 |
| gpt2-xl | +4.9818 | +4.9893 | +0.0075 |
| dialogpt-small | -4.0828 | -4.0790 | +0.0039 |
| qwen2.5-0.5b | +4.8274 | +4.8101 | -0.0173 |
| llama-3.2-3b | +5.4334 | +5.4475 | +0.0141 |
| **protgpt2** | +3.2971 | +3.8962 | **+0.5990** |
| zymctrl | +2.0287 | +2.0714 | +0.0428 |
| **progen2-base** | +1.0604 | +1.2909 | **+0.2305** |
| **progen2-medium** | +1.1070 | +1.2674 | **+0.1604** |

Every text arm moves by at most 0.018 nats. ProtGPT2 moves by 0.599, the ProGen2 pair by 0.16 to 0.23. **Corpus-block sensitivity is roughly 10 to 35 times larger on the protein side than on the text side**, under a seeded permutation within each block, on identical code. No verdict flips, so the measurability conclusion is robust; but a protein cohort statistic carries a cohort-selection uncertainty of a few tenths of a nat that the matched text statistic does not, and that uncertainty is of the same order as several effects this programme has previously reported. This is the standing file-order hazard (§0U.4) measured as a magnitude rather than as an anecdote.

Independent corroboration of the same hazard from the run itself: the earlier `transfer_20260728` file-order cohort gave progen2-medium clean CE 2.5591 and context information +0.3779 on the same 64-246 band; the seeded draw gives clean CE 1.7906 and +1.1070. A 0.77-nat move in clean cross-entropy from cohort composition alone.

### Stage 2 — pathway budget: the clearest modality separation in the campaign

Share of the arm's own context information destroyed by ablating a whole pathway; mean over three cohort seeds, 200 sequences each.

| arm | kind | L | mlp_all | attn_all | **mlp/attn** |
|---|---|---:|---:|---:|---:|
| gpt2-xl | text | 48 | 1.6584 | 0.8250 | **2.01** |
| qwen2.5-0.5b | text | 24 | 1.5227 | 0.7263 | **2.10** |
| gpt2-medium | text | 24 | 1.5151 | 0.8074 | **1.88** |
| gpt2 | text | 12 | 1.4157 | 0.8005 | **1.77** |
| llama-3.2-3b | text | 28 | 1.3224 | 0.7841 | **1.69** |
| gpt2-large | text | 36 | 1.6514 | 1.0977 | **1.50** |
| protgpt2 | protein | 36 | 1.1977 | 1.0647 | **1.12** |
| progen2-medium | protein | 27 | 1.0512 | 1.1340 | **0.93** |
| progen2-base | protein | 27 | 1.0494 | 1.1346 | **0.92** |
| zymctrl | protein | 36 | 1.1805 | 2.1647 | **0.55** |
| dialogpt-small | text | 12 | none | none | none |

All six measurable text arms sit at 1.50-2.10; all four protein arms at 0.55-1.12. The ranges do not overlap. The separation survives replacing GPT-2-large with a Qwen2 and a Llama decoder -- two laboratories, rotary RMSNorm gated-feed-forward architectures, vocabularies 2.5x to 3x larger -- so it is not a GPT-2 idiosyncrasy, which is the objection the architecture-contrast arms exist to answer. In the exactly matched pair it is 1.50 against 1.12.

Read with care. These are ratios of pathway-ablation cost to context information, not a partition: values above 1 mean ablating the pathway costs more than all the context information the arm extracts. The statement supported is **relative pathway balance**, not "the MLP carries X% of the computation". The tokenisation adjustment that collapsed the earlier MLP-share coefficient (EXP-R2-058 addendum 3) has not been applied here and must be before this enters a claim.

### Stage 3 — estimand power: the attainability spine, led from the text control

76 estimands shared across all 11 arms; guards 0.05 nats CE delta and 0.01 nats KL, powered requires the point estimate and the bootstrap q025 of both to clear.

**Attainable on the text positive control gpt2-large: 52 of 76.** The other 24 are **mis-specified, not failed** -- a protein arm scoring below guard on any of them is not evidence about protein models or protein dictionaries.

| arm | kind | powered / 76 |
|---|---|---:|
| qwen2.5-0.5b | text | 74 |
| gpt2 | text | 68 |
| llama-3.2-3b | text | 65 |
| gpt2-medium | text | 56 |
| **gpt2-large (control)** | text | **52** |
| gpt2-xl | text | 48 |
| dialogpt-small | text | **0** (off-distribution) |
| progen2-medium | protein | 72 |
| progen2-base | protein | 70 |
| protgpt2 | protein | 66 |
| zymctrl | protein | 51 |

The whole unattainable set on the text control is `attn_single` at all five depths, `mlp_single` at all five depths, and `attn_window4` at depths 0.33 and 0.85 -- both ablation baselines each. That is, **every single-submodule estimand is unattainable on gpt2-large**, and window and global estimands are attainable.

`mlp_single@d0.50@cohort_mean`, the estimand the audit flagged, resolves here:

| arm | powered | CE delta | q025 | CE guard-pass fraction |
|---|---|---:|---:|---:|
| gpt2-large | **False** | 0.01934 | 0.01715 | **0.00** |
| gpt2-medium | False | 0.03339 | 0.03025 | 0.00 |
| gpt2-xl | False | 0.01484 | 0.01297 | 0.00 |
| gpt2 | True | 0.08364 | 0.07837 | 1.00 |
| qwen2.5-0.5b | True | 0.05406 | 0.05019 | 0.98 |
| llama-3.2-3b | True | 0.10182 | 0.09549 | 1.00 |
| protgpt2 | True | 0.11146 | 0.07832 | 1.00 |
| progen2-base | True | 0.06555 | 0.05931 | 1.00 |
| progen2-medium | True | 0.06216 | 0.05609 | 1.00 |
| zymctrl | False | 0.01523 | 0.01223 | 0.00 |

Confirms the audit finding at 0.0193 nats against its 0.0225 (different cohort draw, same conclusion) and adds the mechanism: **attainability of the single-MLP estimand falls monotonically with depth inside the GPT-2 lineage** -- gpt2 (12L) 0.0836 powered, gpt2-medium (24L) 0.0334, gpt2-large (36L) 0.0193, gpt2-xl (48L) 0.0148, all three deeper rungs unpowered. Three of four protein arms clear it. So the single-MLP estimand's failure on gpt2-large is a **depth** property of the estimand, not a modality property of the model, and any recovery gate built on it would have penalised deep text models and deep protein models alike for a reason having nothing to do with dictionaries.

Panel verdict from `recommendation.json`: 50 estimands powered panel-wide; 2 powered on the text control only (`attn_window4@d0.50`, both baselines); 2 powered on no arm (`attn_single@d0.50`, both baselines); recommended powered estimand `attn_window4@d0.15@cohort_mean`. The P0-2b estimand is recorded as "unattainable on the text positive control, so the estimand is mis-specified and the failures on zymctrl are not interpretable".

### Stage 4 — lens family: nine arms, capability-gated

qwen2.5-0.5b and llama-3.2-3b were **not** run. Their `ArmSpec` declares the `lens` capability, but `src.transfer.lenses.lens_head` resolves the final normalisation as `transformer.ln_f` and requires an `nn.LayerNorm` with a learned bias, which an RMSNorm decoder does not have. `LENS_ARCHITECTURES` is `("gpt2", "progen")`; the worker now filters against it and logs the skip with its reason rather than scheduling an arm the module cannot serve.

| arm | kind | scored | KL@d0.5 | KL@d0.9 | top-1@d0.5 | top-1@d0.9 | tuned-lens mean KL gain |
|---|---|---|---:|---:|---:|---:|---:|
| gpt2 | text | yes | 5.0596 | 2.4390 | 0.1224 | 0.4478 | 2.841 |
| gpt2-medium | text | yes | 4.0432 | 3.4152 | 0.1471 | 0.2027 | 1.595 |
| gpt2-large | text | yes | 4.7723 | 4.3630 | 0.0771 | 0.0982 | 1.402 |
| gpt2-xl | text | yes | 4.8193 | 4.6009 | 0.0931 | 0.0952 | 1.576 |
| dialogpt-small | text | **no** | — | — | — | — | — |
| protgpt2 | protein | yes | 3.3525 | 3.4651 | **0.0131** | **0.0140** | 1.738 |
| zymctrl | protein | yes | 3.0454 | 2.5074 | 0.0798 | 0.1072 | 0.682 |
| progen2-base | protein | yes | 0.8277 | 0.7893 | 0.1114 | 0.1214 | 0.379 |
| progen2-medium | protein | yes | 1.7411 | 1.9557 | 0.0746 | 0.0630 | 1.051 |

The tuned lens improves on the untuned logit lens at **every** non-identity layer on every scored arm, protein included -- the one instrument in this campaign with no transfer failure to report. ProtGPT2 is the outlier on top-1 agreement (0.013-0.014, five to ten times below every other arm), which is the multi-residue-BPE arm and is consistent with its unembedding aperture rather than with a lens defect.

The Jacobian finite-difference guard passed on all nine arms, maximum relative error 4.7e-3 (ProtGPT2) against the 2e-2 tolerance. This is the guard that fired at 1.008 in run `20260728160933_83ff09d5a909` when the worker forced `--dtype bfloat16` on stage 08; the float32 default was left untouched here and the guard is quiet. Confirms the diagnosis rather than assuming it.

### Code changes and two defects found by running

`scripts/transfer/` only; no `src/transfer/` file was modified.

1. `run_transfer_h200.sh` and `h200_worker.sh`: `KNOWN_ARMS` widened from four to the eleven staged arms.
2. `h200_worker.sh`: modality is now resolved by an explicit `arm_modality` enumeration. It previously treated `gpt2-large` as the only text arm and everything else as protein, which was correct for a four-arm panel and would have handed six text arms the Swiss-Prot corpus variable and the protein model root. `model_var_for_arm` now routes non-gpt2-large text arms to `R2_TEXT_MODEL_BASE_DIR`. `progen2-base` joins `progen2-medium` in the float32 `cohort_power` item.
3. `h200_worker.sh`: `lens_family` runs a capability-filtered arm list.
4. `08_lens_family.py`: `--arms` default changed from `sorted(PANEL)` (no capability guard) to a capability-filtered default. The worker passes the list explicitly regardless.
5. **`01_cohort_power.py` used the plug-in unigram estimator** for its Gate-0 verdict, while stages 02 and 03 default to the disjoint held-out estimator. Now takes `--unigram-estimator {disjoint,plugin}` defaulting to `disjoint`, with a held-out reference block deduplicated by content. No fallback path.
6. **`01_cohort_power.py` took the first `--n-seq` records in corpus file order.** Now draws under a seeded permutation from a pool (`--cohort-draw-seed`, `--cohort-pool-size`), with `--cohort-skip` to move the pool origin. `--cohort-draw-seed 0` selects file order as a declared choice.
7. **Defect found by running (would have failed silently before):** the Markov held-out block was offset by record count only. A record-count offset is not disjointness -- Swiss-Prot carries the same sequence under several accessions and the EC corpus under several EC tags -- and `markov_cross_entropy_bits` correctly refused the pair: `Markov train and test sets share 1 sequences`. Now deduplicated by content with the count recorded; at production size the ZymCTRL cohort drops 2 of 2000 Markov records and 2 of 4000 unigram reference records.
8. **Defect found by running:** the worker passed the whole arm list to `03_estimand_power.py recommend`, which raises unless exactly one arm is text. It now anchors on `TEXT_ARM` plus the protein arms, which is the contract `recommend()` documents. Every text arm is still measured; what is scoped is which single arm the panel verdict is anchored on.

Items 7 and 8 each cost one scheduled run and were caught by the stage scripts' own guards rather than by producing a wrong number, which is the behaviour the failure principle asks for.

### Compute

Primary campaign 03:04 to 03:25 pod-side, about 21 minutes wall clock on four H200s; sensitivity run about 7 minutes. Per-item peaks were well inside one card. L20 used for validation only (`cuda:2`, `cuda:4`), returned to 3 MiB.

---

## 2026-07-29 — EXP-R2-061: full-scope code audit of `src/transfer/`; DIAMOND repeat masking retracts the homology control

**Task.** Audit every module under `src/transfer/` against the hazards in Appendix B of `INTERPRETABILITY_TRANSFER_AUDIT.md`, fix the defects found, and add a regression test for each behavioural fix. No GPU measurement was run; one CPU verification against the UniRef50 snapshot was.

**Environment.** `conda activate ct`, `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, `PYTHONPATH=. pytest -q tests`. No H200 work. `scripts/transfer/` and `scripts/transfer_gap/` were read but not edited.

### Headline: EXP-R2-050 (homology control) is retracted

`homology.run_diamond_blastp` built its command without `--masking`, so DIAMOND 2.1's default tantan low-complexity/**repeat** masking was active — against a cohort that is *selected for internal tandem repeats*. The HSP stops at the repeat and `nident` under-counts, so `identity_over_query = nident / qlen` is not the query coverage the docstring claims.

Verified directly, not inferred. Cohort record 0 of `results/transfer_20260728/homology_control/` is byte-identical to `UniRef50_Q3E8Z8` over all 732 residues (checked against `data/uniref50/uniref50.fasta`). DIAMOND reported `pident 100`, `nident 607`, `qend 607`, `slen 732` — coverage 82.9%, binned `id70_to_95_close_homology` rather than `ge95_near_duplicate`. A verbatim member of ProtGPT2's pretraining corpus was recorded as a *diverged relative*.

Five of forty-eight exact-cohort records are affected, all in the same direction, all into the bin with the highest measured induction (`peak_over_uniform` 75.2 against 35.2 for `ge95` on ProtGPT2). The bias is the reverse of the only identity bias the module declares (docstring: "biases the measured identity *upward*") and it grows with repeat content — the one property the cohort is selected on and the one that also drives prefix matching.

**Consequence.** The stratum profile that the audit document reads as "interpretation 3: neither" is explained by the mis-binning. `EXP-R2-050 — homology control — not memorisation` must move from **stands** to **retracted pending re-run**. Cost to re-establish: `--stages search` only, no GPU.

Second, independent reason the same artefact cannot be quoted: `bootstrap_stratum` had no floor on its unit count. At four distinct sequences a percentile interval over at most 35 atoms is *pinched inward* — measured relative widths 9.4–15.0% at n=4 against 26.7–27.2% at n=420 in the same file — and two `consistent_with_memorisation` verdicts were decided by non-overlap of a four-unit interval against a four-hundred-unit one.

### Defects fixed (silent wrong answer)

| module | defect | fix |
|---|---|---|
| `homology` | DIAMOND repeat masking truncates alignments of repeat-selected queries | `--masking 0`; `truncated_alignment` detector; `assign_homology` raises on the signature |
| `homology` | percentile interval narrower at n=4 than at n=420 | `MINIMUM_BOOTSTRAP_UNITS = 8`; below it, `degenerate: true` and no interval |
| `homology` | `best_hit_spans_repeat` computed, documented as load-bearing, never used (27% of the exact cohort) | `stratum_integrity()` reports it per stratum |
| `homology` | `--max-target-seqs` saturated on 47/48 queries, unflagged | `hit_list_saturated` per record |
| `homology` | `sub_cohort` sliced 2 of 3 per-record arrays, copied the rest unsliced | `PER_RECORD_METADATA_KEYS`; any other per-record list raises |
| `homology` | `_correlation` aborts an arm's run in the regime the docstring predicts | `measured: false` with the constant covariate named |
| `circuits` | `summarise_patching` published an index into a *filtered* list as `best_layer` | layer indices tracked; `n_layers_with_a_mean` reported |
| `circuits` | `ov_copying_scores(layers=subset)` returned zero-filled rows for unscored layers; zero is a legal score | NaN-initialised, so `summarise_head_matrix` raises |
| `circuits` | signed pathway fractions divided by a cancelling denominator guarded only at 1e-9 | `SIGNED_FRACTION_MINIMUM_RATIO`; cancellation ratio and signed total published |
| `circuits` | repeat cohorts took the first N *matching* records of a family-grouped file | `seed=`/`skip=`; `sampling` metadata on every cohort |
| `probes` | `unit_states` read the post-final-norm state at the top layer while the eraser writes pre-norm | refused, with the reason |
| `probes` | no capability gate anywhere; `Arm.blocks()` carries none, so a `{budget, lens}` arm reached a full residual-stream erasure | `refusal_reason` gates on `pathway` |
| `probes` | `control_matching` divided by the mean-ablation denominator unguarded; a negative one **flips** `cost_is_a_matched_cost` | threaded `denominator_valid`; `None` when invalid |
| `probes` | `ec_units`/`pfam_units`/`fitness_units` took corpus prefixes (measured: 135 distinct groups against 231 under a permutation, EC-1 share doubled) | `record_order()`, seeded by default, mode recorded |
| `probes` | `family_disjoint=True` asserted for `fitness`, but ProteinGym ships four `BLAT_ECOLX` assays of one protein | one assay per target protein, enforced |
| `probes` | behaviour cohort was the first N held-out units — one assay, one protein | `stratified_unit_draw()` over the fold's groups |
| `probes` | `sequence_bootstrap` treated 40 single mutants of one wild type as 40 sequences | optional `groups=`; cluster unit recorded |
| `probes` | `SampleSet.cohort()` hashed *rendered* strings, so the cross-arm digest could never match | `Unit.content`; digest is arm-independent |
| `probes` | no WT/mutant consistency check on ProteinGym numbering | asserted per variant |
| `probes` | erasure hook leaked on a failed forward | `try/finally` |
| `path_patching` | probes consumed in list order; at campaign defaults only the first 16 of 32 contributed | seeded permutation; `probe_visit_order` recorded |
| `path_patching` | `sem` divided by `sqrt(n_cases)` over nested prefixes of one protein | `sem_probe_clustered` + `n_probes` |
| `path_patching` | `bootstrap_difference` sold head heterogeneity as a confidence interval | renamed `spread_*`/`separated_across_heads`; uncorrected p-value; caveat |
| `path_patching` | per-head floor guarding a sum over ~50 heads; "weighted mean" false under mixed signs | floor scaled by head count; `aggregate_is_a_weighted_mean`, cancellation ratio |
| `path_patching` | `attention_output_projection` resolved by attribute search | per-architecture dispatch + `require_supported_layout` |
| `path_patching` | `head_dim = d_model // n_head` silently truncating | `circuits.head_dim` |
| `path_patching` | locality invariant fell back to averaging over *ineligible* rows | eligible rows selected; raises when there are none |
| `path_patching` | `n_cases_checked` reported the batch size, not the eligible count | corrected |
| `arms`/`budget`/`lenses` | `sequence_target_mask` dispatches on the arm's **name**; a future EC-conditioned arm would have its prompt scored as content | `require_boundary_masking_support` |
| `arms` | no seeded-permutation cohort sampling existed at all | `selected_positions`, `sampling_record`, `seed=` on both constructors |
| `arms` | pattern-reading/overriding measurements trusted the caller to pass `eager` | `Arm.attn_implementation` read back from the built model; `require_eager_attention` |
| `scaling` | `nearest_neighbour_contrasts` lacked the capability gate `fit_modality_offset` applies | gate added |
| `channels` | `event_selection_ceiling` divided by `h(1) = 0` | guarded |
| `relational` | band lookup raised bare `StopIteration`; anchor-level SEM ignored protein clustering | guarded; `sem_protein_clustered` |

### `ArmSpec.source` — resolved by splitting, not renaming

`source` was the *evaluation cohort* corpus, but the bare name reads as provenance and every text arm carried `"openwebtext"` — true of the cohort, false of the pretraining data for six of seven text arms. Both facts are now fields: `evaluation_cohort_source` and `pretraining_corpus`, with `PRETRAINING_UNDECLARED` where no model card states one (ByGPT5). `source` survives as a read-only alias so `scripts/transfer_gap/tg_common.py:229`, which now *dispatches* on it, and nine artefact writers keep working. `MATCHED_DATA_CONTRAST` and `TEXT_DATA_CONTRAST` are checked at import against the new field.

### Tests

`tests/test_transfer_audit_invariants.py`, 60 tests, one per behavioural fix, written against the restored property rather than the implementation. Mutation-checked: reverting `summarise_patching`, the `probes` capability gate and the `assign_homology` truncation refusal fails exactly the three tests that defend them.

`ruff check src/transfer/ tests/test_transfer_audit_invariants.py` clean. `PYTHONPATH=. pytest -q tests` → **348 passed, 6 subtests** (288 before).

### Left for the owners of `scripts/`

Not edited, per the freeze. Each is a one-line change:

1. `scripts/transfer/06_explanation_channel.py:144` passes `alphafold_models(..., limit=n)` — a filename-order prefix, while `05_relational_channel.py` and `probes.py` permute. Switch to `channels.alphafold_model_sample(..., seed=)`. This is the L9 "declared source-order bias" caveat, and it is now fixable.
2. `scripts/transfer/10_homology_control.py` should call `homology.stratum_integrity()` and record `--masking 0` in the search artefact, then re-run `--stages search`.
3. `scripts/transfer/07_convergence_control.py:662` writes only `"source"` into its per-rung record; adding `evaluation_cohort_source` and `pretraining_corpus` would remove `analysis_frame`'s panel fallback.
4. Cohort constructors now accept `seed=`; passing one is what makes Appendix B rule 1 hold for a campaign. The default is unchanged so a running campaign is not moved mid-flight, but every cohort now carries `sampling.mode` and, in file order, `sampling.hazard`.
5. `scripts/transfer/04_circuit_primitives.py::grant_circuits` mutates `arm.spec` to override the capability gate. Explicit, opt-in and recorded, so accepted — but `Arm` being a mutable dataclass means the gate is advisory.

## 2026-07-29 — EXP-R2-062: TG-series rendering contamination repaired; the variance–behaviour dissociation is RETRACTED

Discharges plan item **B2**. `scripts/transfer_gap/tg_common.py::protein_input` returned the plain amino-acid sequence for ProtGPT2 against a model pretrained on end-of-text-separated, 60-column-wrapped FASTA (§0.1 of `INTERPRETABILITY_TRANSFER_AUDIT.md`). Repaired, blast radius measured, TG-03, TG-07 and TG-09 re-run under corrected rendering on seeded-permutation cohorts. Two L20 cards (`cuda:2`, `cuda:4`), ~1.8 GPU-hours, no H200 work. Artefacts under `results/transfer_gap_20260729_corrected/`; the 2026-07-24 tree is retained unmodified because it is cited. Pre-correction sources and a per-number status table: `archive/legacy/r2_transfer_gap_precorrection_20260729/`.

### The fix

`tg_common.py` is now an adapter over `src/transfer/arms.py` and implements no rendering, model loading or tokenisation of its own — the defect survived an earlier withdrawal precisely because two renderers existed and only one was fixed. It retains cohort *selection*, deliberately, because its rule now differs from `arms.protein_cohort`: seeded permutation of the complete eligible set, with `skip` partitioning that permutation rather than walking further down the file. No file under `src/transfer/` or `scripts/transfer/` was modified.

**Requested change to `src/transfer` (not made here):** `protein_cohort` and `text_cohort` should take a `seed` and permute. They document "deterministic file order", which is limitation L13 in the code that every stage calls. When they do, the selection layer in `tg_common` should be deleted rather than kept in sync.

### TG-00, new — input-contract positive controls

Both hazards reproduce as positive controls and the stage is now a prerequisite. ProtGPT2, 80 Swiss-Prot sequences 600–2000 aa, seeded cohort:

| rendering | CE nats/token |
|---|---:|
| raw single line (the defect) | 7.218 |
| end-of-text + raw | 7.318 |
| wrapped at 60 | 5.458 |
| end-of-text + wrapped (native) | 5.439 |

Rendering delta **1.779 nats/token** against the 1.42 of EXP-R2-028, which used a file-order cohort. File-order control at n=200, 200–800 aa, native rendering both sides: ProGen2-medium 1.723 file order against 1.483 permuted, **+0.240 nats**; ProtGPT2 −0.079. Consistent with §0.1's expectation that file-order exposure is minor at n≥200, and it is not zero.

### B2 verdict — the dissociation does NOT reproduce

TG-07, relative depth 0.5, rank sweep 1–512, alignment gap = variance explained − loss recovered:

| arm | gap @512 before | gap @512 corrected | loss recovered @512 before | corrected |
|---|---:|---:|---:|---:|
| gpt2-large | −0.025 | −0.024 | +0.975 | +0.974 |
| **ProtGPT2** | **+1.102** | **+0.118** | **−0.105** | **+0.879** |
| ZymCTRL | +0.072 | −0.040 | +0.793 | +0.838 |
| ProGen2-medium | +0.379 | +0.326 | +0.537 | +0.555 |

ProtGPT2 clean CE 8.654 → 4.993; mean-ablation headroom 1.010 → 4.028 nats/token. **gpt2-large is unchanged to three decimals**, which is what establishes that the movement is the rendering and not the cohort.

The recorded ordering inverts. ProtGPT2 is no longer the extreme arm; ProGen2-medium is, and ProtGPT2 now sits closer to the text control than to it. Taking the maximum gap over rank instead: gpt2-large +0.708, ProtGPT2 +1.416, ZymCTRL +0.289, ProGen2-medium +0.683 — **ZymCTRL, a protein arm, is 2.4x below the text control**. The decoupling is therefore not modality-conditional and does not motivate a protein-specific method line. **C4 (reliance-weighted dictionaries, 20–40 GPU-h) loses its stated motivation and should not be started on this basis.**

### The variance-concentration premise was mismeasured for every arm

Independent of rendering. PC1 and participation ratio over all positions are dominated by structurally special tokens; 200k activations at depth 0.5:

| arm | PC1 all | PC1 interior+residue | PR all | PR interior+residue | first-token norm |
|---|---:|---:|---:|---:|---:|
| gpt2-large | 0.809 | **0.034** | 1.53 | **253.4** | 24.8x |
| ProtGPT2 | 0.971 | **0.439** | 1.06 | **4.86** | 5.2x |
| ZymCTRL | 0.182 | **0.008** | 26.99 | **577.3** | 4.9x |
| ProGen2-medium | 0.588 | **0.520** | 2.88 | **3.69** | 9.7x |

GPT-2-large's concentration is its first token; ProtGPT2's is the FASTA newline introduced by the *correct* rendering (separator norm 2.5x, 5.2% of scored positions). The recorded contrast "ProtGPT2 0.951 against gpt2-large 0.809" compared two artefacts of different origin. On the controlled statistic the protein arms do not cluster and the highest effective dimension in the panel is a protein arm. Same hazard class as L6, in a different statistic. **New standing rule: report residual spectra on interior, alphabet-bearing positions.**

### TG-03 — L3 survives and is strengthened

| arm | FVU before → corrected | loss recovered before → corrected |
|---|---|---|
| gpt2-large | 0.0503 → 0.0531 | 0.968 → 0.967 |
| ProtGPT2 | 0.0211 → **0.0051** | 0.336 → **0.706** |
| ZymCTRL | 0.2666 → 0.2525 | 0.701 → 0.763 |
| ProGen2-medium | 0.1707 → 0.1313 | 0.376 → 0.463 |

ProtGPT2 now has the best FVU in the panel by 10x and ranks third of four on loss recovered; ZymCTRL has the worst FVU by 50x and ranks second. Spearman(FVU, loss recovered) = 0.000 corrected against +0.400 before, n=4, descriptive. **L3 — FVU does not track behavioural fidelity — is now demonstrated on clean data.**

### TG-09 — "~0.35 at every depth" retracted

| arm | d=0.15 | 0.33 | 0.50 | 0.67 | 0.85 |
|---|---:|---:|---:|---:|---:|
| ProtGPT2 before | +0.349 | +0.363 | −0.192 | −0.011 | +0.186 |
| ProtGPT2 corrected | **+0.996** | **+0.977** | +0.354 | +0.609 | +0.715 |
| gpt2-large corrected | +0.964 | +0.921 | +0.916 | +0.899 | +0.834 |

Headroom at depth 0.5: 1.010 → 4.028 nats. ProtGPT2 was recorded as depth-independent; corrected it is the *most* depth-dependent arm in the panel, range 0.64. This vindicates TG-09's own premise that one splice site cannot carry a cross-modality claim.

### Other defects found and fixed in the same class

- **TG-01** used the plug-in entropy `H(p̂)` on its own fitting sample as the context-free baseline (L12; bias is vocabulary-dependent, so it fell on exactly the two 50k-vocabulary arms). Replaced by held-out cross-entropy, plug-in retained beside it. Long-range shares gained a denominator floor: ZymCTRL's 1.0197 and −0.018 came from dividing by 0.386 nats. Truncation flagged as destroying conditioning on EC-conditioned arms (ZymCTRL read 21.3 nats at context 1).
- **TG-08** computed loss recovered with **no** denominator guard at all.
- **TG-06** monkeypatches `modeling_gpt2.eager_attention_forward`, which does not reach ProGen2. On such an arm capture returned nothing, injection was a no-op, the exactness check passed with error exactly 0.0 and the transplant cost read 0.000. Now asserts one captured pattern per layer.
- **TG-05** scored ZymCTRL **unconditioned** — it called `protein_input`, which had no ZymCTRL branch, three functions from the `load_zymctrl` docstring warning against exactly that. Now refuses, pointing at `scripts/transfer/05_relational_channel.py`. Structures were also drawn in filename order and split by prefix, making the train/test split a split by accession block.
- **TG-04** drew every unit set in source order. Now seeded — this discharges the standing caveat on the L9 bits/symbol figures (7.32 / 3.61 / 0.74), which should be re-derived before quoting.
- **TG-10** estimated its cohort-mean ablation baseline from the first 8 batches while applying it over the whole cohort.
- **TG-02** gained a shuffle that permutes only alphabet-bearing tokens: under the corrected rendering, permuting the far block also destroys FASTA line structure, so "order information" would be partly the cost of malformed format.
- **`tokenize_batch`** fell back to pad id 0 when a tokenizer declared neither pad nor end-of-text. Over ProGen2's 32-symbol vocabulary, id 0 is a real symbol. Now uses the shared implementation, which raises.

### Series fate

Repaired in place, not ported: TG-02, TG-06, TG-07, TG-08 and TG-09 have no equivalent in `src/transfer/`, so the series is not superseded as a whole. TG-01, TG-04, TG-05 and TG-10 are superseded and are marked so in-file and in `scripts/transfer_gap/README.md`; they are corrected rather than deleted because a superseded script that still runs is a script that will still be run. Porting the five unique stages into `src/transfer/` is the right follow-up and was out of scope here, that package being under concurrent audit (EXP-R2-061).

### Operational note

Both nested `.git` directories disappeared from the working tree at 03:11–03:14 during this session. Cause identified as deliberate housekeeping by a concurrent agent — they are intact at `archive/legacy/nested_git_history_20260729/`. No data was lost, but note that neither repository ever tracked `scripts/transfer_gap/`, so the pre-correction sources had **no** version-control record and the archived snapshot above is the only copy. GPUs 2 and 4 returned to 3 MiB; host memory 106/503 GiB.

## 2026-07-29 — EXP-R2-063: full-scope code audit of `scripts/transfer/`; one panel contract replaces five hand-maintained arm lists

**Task.** Audit the eleven stage scripts plus the controller and worker against Appendix B of `INTERPRETABILITY_TRANSFER_AUDIT.md`, starting from an eighteen-item defect list produced by the operator of EXP-R2-060. No GPU measurement; every check ran on CPU. `src/transfer/` was touched only where the task scoped it, and `10_homology_control.py` was not touched at all — a homology re-search was running against it throughout.

**Environment.** `conda activate ct`, `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, `PYTHONPATH=. pytest -q tests`. No H200 work.

### Headline: five hand-maintained arm lists, two of them wrong

Which arms a stage measures was decided in five places that could disagree with `src.transfer.arms.PANEL` and with each other — `KNOWN_ARMS` in the controller, a second copy in the worker, and the worker's own modality enumeration, lens-arm exclusion and relational-arm inclusion. Two disagreed with the panel:

- **`relational_channel` was narrowed by one arm.** The worker's list named `zymctrl` and `progen2-medium`. `progen2-base` is protein, residue-tokenised and carries the `relational` capability, and `05_relational_channel.py` refuses it for no reason. Its `--arm` default is a single arm, so nothing else would have caught it.
- **`homology_control` was widened by one arm.** The worker passed its own four-arm protein list to a script whose own `--arms` default names three, under a comment asserting the opposite. A campaign run and a direct run measured different panels.

Both are the L18 shape: well-formed numbers over a panel that is not the panel a reader assumes. `"relational"` was additionally found to be a capability that no script or module has ever consulted.

**The fix is one predicate.** `scripts/transfer/panel_contract.py` declares, per stage, what an arm must satisfy, and `arm_can_run(stage, arm)` answers it by composing four sources without restating any: `ArmSpec.capabilities` (what the panel *intends*), the measuring module's own architecture declaration (what it can *deliver* — `scaling.LENS_ARCHITECTURES`, `circuits._CIRCUIT_ARCHITECTURES`, `path_patching.SUPPORTED_ARCHITECTURES`), `ArmSpec.modality`/`tokenisation` where the design needs them, and an explicit exclusion list for what the first three cannot express. Every refusal names the declaration that made it, so a skip line distinguishes a panel decision from a module limitation from a staging fact.

The declared-versus-deliverable split (item 9) is therefore kept, not resolved: the rotary arms still declare `lens` and `scaling.lens_supported` still says no, and the predicate obeys the second. What changes is that exactly one thing consults both.

`panel_contract.sh` is generated from it and sourced by controller and worker, so bash carries no third copy. The worker re-derives it from the live panel (`panel_contract.py --verify`) inside the pod before any GPU is scheduled, and the test suite runs the same check, so a stale rendering cannot reach a measurement.

### `11_induction_path_patching.py` could not have run on four of eleven arms

The worker passed the whole `ARM_LIST` to stage 11. `path_patching` resolves an attention head's write by dispatching on `ArmSpec.architecture`:

- `qwen2.5-0.5b`, `llama-3.2-3b` — refused by `require_supported_layout`, after the checkpoint is loaded and the repeat cohort scanned.
- `progen2-base`, `progen2-medium` — **`ArmSpec.architecture` declares `"gpt2"`** (the field default, never set for them) while the checkpoints are GPT-J-style: verified in `models_R2/progen2-medium/modeling_progen.py`, attention holds `qkv_proj`/`out_proj` where GPT-2 holds `c_attn`/`c_proj`. So `attention_output_projection` looks for `c_proj`, does not find it, and raises.

Item 10's "harmless today" is right that nothing is *silently* wrong, and wrong that nothing follows: `_OUTPUT_PROJECTION_ATTRIBUTE["progen"]` is dead for panel arms, and stage 11 on a ProGen2 arm is a scheduled crash. The declaration was **not** corrected — see "Left alone" below. The scheduler now skips those four arms with the reason instead of losing a run to the exception.

### Guards that fired after expensive work — nine hoisted

Each of these validates a *command-line combination* and each used to raise after a checkpoint load, a corpus scan, or worse:

| script | combination | previously raised |
|---|---|---|
| `01` | `--max-len` vs `--truncation-contexts` | inside `budget.truncation_curve`, after `load_arm` and a full `arm_power` sweep |
| `01` | `--arms` vs `--kind`, `--arms` vs `--with-ec` | after a 4000-record pool, a 4000-record reference and a 2000-record Markov block had been read |
| `02` | `--seed-target-error-fraction`, `--minimum-ce-delta-nats`, `--minimum-kl-nats`, `--window` | inside `seed_requirement`/`build_scopes`, after every seed x every scope had been ablated |
| `03` | `--depths` collapsing to one estimand id | inside `build_estimands`, after `load_arm` |
| `05` | `--arm` capability/tokenisation | `require_residue_token_map`, immediately after `load_arm` |
| `05` | `--train-fraction` vs `--min-test-proteins` | `homology_disjoint_split`, after every protein had been encoded |
| `08` | `--jacobian-positions` vs `--n-seq` x (1 − `--train-fraction`) | `sample_jacobian_probes`, after both splits were cached and the tuned lens trained 2000 steps |
| `09` | `--concepts` vs `--arm` vs `--ec-conditioning` | after `load_arm`, though `refusal_reason` reads only `ArmSpec` |
| `09` | `--erasure-fraction` vs `--layer-fractions` | after `load_arm`, though it is arithmetic on the declared `n_layer` |

`08`'s and `09`'s are the load-bearing ones. `08`'s ceiling is `n_seq x (1 - train_fraction)`, duplicated as `evaluation_split_size` and held to `lenses.split_cohort` by a test over six sizes and four fractions. `09` now builds a spec-only `Arm` (model and tokenizer `None`) so that a check which was always spec-only fails loudly if it ever starts reading weights.

Measured: `08 --arms qwen2.5-0.5b` and `08 --n-seq 32` now fail in the time it takes to import torch; `05 --arm protgpt2` and `09 --arm protgpt2 --concepts ss3` likewise.

### Defaults that narrowed a panel

- **`01_cohort_power.py::default_arms` returned `["gpt2-large"]` for text.** A bare `--kind text` qualified a *one-arm* text side of an eleven-arm panel. Now derived; measured 7 text, 3 protein, 4 protein with `--with-ec`.
- **`02` and `03 measure` defaulted to `sorted(PANEL)`**, which admits the three ByGPT5 rungs with no `pathway` capability — arms that could only fail, and failed only after a checkpoint load. Now capability-filtered.
- **`03 recommend` defaulted to `sorted(PANEL)`**, which can *never* satisfy its own contract: it raises unless exactly one arm is text. That default is what lost a scheduled run of EXP-R2-060; the worker was fixed then and the script's default was not. Now control-anchored by construction.
- **`08`'s capability-filtered default was never enforced.** An explicit `--arms` bypassed it entirely, and the worker always passes one — so the EXP-R2-060 fix protected only the case that cannot occur. Now checked in `validate`.

Every stage that writes a per-arm artefact now records a `stage_contract` block naming which arms it measured, which were eligible and not asked for (this invocation narrowing the panel), and which were refused with the reason. The two are kept apart because they mean different things.

### Cohort bands differ across stages, and now say so (item 12)

Four stages of one campaign draw protein cohorts on three bands:

| stage | band | |
|---|---|---|
| `01_cohort_power` | 64–246 | defines the qualifying band |
| `02`, `03` | 64–246 | match |
| `08_lens_family` | **64–120** | narrower; the Jacobian sweep is quadratic in length |
| `07_convergence_control` | 600–2000 / 64–246 | per-rung, declared in `scaling.py` |

**This is a defect of declaration, not of value.** An arm that `cohort_power` qualifies at 64–246 has not been qualified on the population `lens_family` scores, and EXP-R2-060 measured protein cohort-block sensitivity at 0.16–0.60 nats — the same order as several reported effects. The bands are unchanged (changing one would move published lens numbers); each is now declared in the contract with its reason, written into every artefact beside the qualifying band and a `matches_qualifying_stage` flag, and held to the stage's own argparse default by a test.

`scaling.py` separately declares 100–246 residues unusable for ProtGPT2 (context information −1.21 nats). That figure predates the rendering fix and the held-out estimator; EXP-R2-060 reads +3.297 on 64–246. The declaration is stale but is in `src/transfer/` and was left.

### A stale panel verdict that verified cleanly

`03_estimand_power.py recommend` is the one item whose output is a function of *other items' outputs* rather than of its own command. Its resume key covered only its command, so setting `ARGS_ESTIMAND_POWER` re-ran `measure` for every arm and then **skipped `recommend` as complete**, leaving a `recommendation.json` derived from measure outputs that no longer existed — and passing its own checksum verification. The consumed per-arm manifests are now folded into its provenance.

### Item 17: a CPU aggregation no longer costs six GPU stages

`run_estimand_power` called `exit` when `recommend` failed. `recommend` is a CPU-only aggregation at the end of tier 2 that feeds no later stage, so its failure took tier 3 — six GPU stages — with it, and recovering them required a re-run that re-verified manifests and did nothing else. It is now a deferred failure: logged, no manifest written, campaign continues, worker exits non-zero at the end with every deferred failure named.

### Item 18: pod-name redaction made structural

The controller printed `H200_POD` in its startup banner and `tee`'d the worker's output verbatim. Both are gone: the banner line is deleted, `redact` filters every line the controller emits, and the worker's entire merged stdout/stderr passes through it before reaching either the terminal or the controller log. The guarantee no longer depends on the operator piping through `sed`. `H200_POD` is also no longer required for `--help`, which previously aborted with the very message the usage text explains.

### Item 11: per-item scale arguments, and a refused collision

`ARGS_<STAGE>` reaches every item of a stage at once. `cohort_power`'s four items differ in vocabulary regime, dtype and cohort name, so one knob is rarely right for all four, and a flag the worker already sets was silently accepted with argparse taking the last occurrence — which for `--cohort-name` collides two items' cohorts on one output path and for `--dtype` discards a measured reason. Added `ARGS_<STAGE>__<ITEM>` (`--item-args` on the worker), and a duplicate-option check that refuses. Both run in `verify_commands_buildable`, which builds every scheduled command before the import preflight.

### Other defects fixed

- **The import preflight covered nine of eleven entry points.** `10_homology_control.py` and `11_induction_path_patching.py` were scheduled without ever being import-checked — exactly the class of failure the preflight exists to stop. Now derived from the requested stages.
- **A panel-wide stage with an empty arm list** would have passed `--arms` with no value and let the entry point fall back to its own default, a different panel. Now skipped with a reason.
- **`estimand_power` without its text control** now refuses rather than anchoring the panel verdict on whatever else is present.
- **The run manifest re-read nine named `ARGS_*` variables**, so an override for either of the two stages with no named variable reached the worker unrecorded. It now records exactly what was collected.
- `06_explanation_channel.py` draws its AlphaFold structures under a seeded permutation (`channels.alphafold_model_sample`) instead of a filename-order — i.e. accession-order — prefix. This discharges the standing caveat on the L9 bits/symbol figures (7.32 / 3.61 / 0.74), which must be re-derived before they are quoted. The Pfam and text channels are *still* file-order draws and the artefact now says so per channel instead of one blanket `deterministic_file_order_no_stochastic_sampling`.
- `07_convergence_control.py` writes `evaluation_cohort_source` and `pretraining_corpus` beside the legacy `source`, removing `scaling.analysis_frame`'s panel fallback.
- The worker's `model_var_for_arm` and `corpus_vars_for_arms` are resolved from how `arms.PANEL` *builds* each path and from each arm's declared evaluation cohort, not from a `case` on the name. The mapping is invariant to the host, because every `ArmSpec.path` is constructed from one of the three environment constants.

### `scripts/transfer_gap/tg_common.py`: the workaround layer is gone

`arms.protein_cohort`/`text_cohort` already took `seed=` (added by EXP-R2-061), so the requested library change was in place; what remained was the local layer built to work around its absence. Removed: `_ELIGIBLE_CACHE`, `_eligible_swissprot`, `_eligible_ec`, `_eligible_text`, `_permutation`, and the selection bodies of `cohort_for` and `load_text`. That is Appendix B rule 12 one step earlier than rendering — a duplicated *eligibility* predicate decides which records exist before anything decides how they are drawn.

**Verified against the previous implementation before removing it**, at three (n, skip, band, with_ec) settings: the drawn *set* is identical, the eligible count is identical (357,536 plain / 57,096 EC), and the EC label pairing is preserved. Only the within-cohort *order* changes — `arms.selected_positions` returns ascending corpus order where the layer returned permutation order — so `Cohort.digest` values recorded before today will not reproduce byte for byte even though the sequences behind them are the same. No TG stage slices a cohort positionally, so no measured quantity depends on the order.

### Left alone, with reasons

1. **`ArmSpec.architecture` for the ProGen2 arms (item 10).** Correcting it to `"progen"` is *not* inert: `path_patching._OUTPUT_PROJECTION_ATTRIBUTE` maps `"progen"` to `out_proj`, so the change would move path patching on two panel arms from a loud refusal to a live code path that has never produced a validated number. That is a measurement decision, not an audit repair, and `src/transfer/` is out of scope. Recorded as an exclusion that names the defect, with a test asserting `PANEL[arm].architecture == "gpt2"` so the exclusion fails the moment the declaration is fixed.
2. **Provenance keyed on the whole code hash (item 16).** A per-stage identity — `src/**` plus that stage's entry point plus `h200_env.sh` — is expressible and my analysis says it is safe, since no numbered entry point imports another and everything the worker decides is already in the canonical command vector. It is not adopted, because I cannot exercise resume semantics end to end without a pod, and the failure mode of getting it wrong is a stale artefact that verifies cleanly against its own checksum — the same failure the `recommend` fix above was found to have. A redundant re-run is the cheaper error. The cost is reduced instead: `recommend` now keys on its inputs, so the most common re-run is no longer wasted.
3. **`10_homology_control.py`** — not touched. A DIAMOND re-search was running against it for the whole session. `homology.stratum_integrity()` and recording `--masking 0` in the search artefact (EXP-R2-061's deferred item 2) remain open. Two further defects found there and not fixed for the same reason: `--repeat-criterion` does not appear in *any* output filename, so an `approximate` run overwrites an `exact` run's `homology_assignment.json`, per-arm files and `panel_summary.json`; and the `--device` CUDA-availability check runs after the full multi-process repeat scan.
4. **`04_circuit_primitives.py::grant_circuits`** mutating `arm.spec` — accepted as EXP-R2-061 accepted it: explicit, opt-in per arm, recorded in the output. The contract therefore gates stage 04 on architecture, not on the capability.
5. **`14_paa_census.py`** is not a campaign stage and was out of the wired scope. Two findings recorded for its owner: its gate-0 table draws ZymCTRL from the EC-labelled corpus and the other protein arms from plain Swiss-Prot in the *same* table (`--protein-source` defaults to `plain`), which is the cross-arm-comparability failure `02`/`03`/`08` all avoid by using one corpus for every protein arm; and its pool `.npz` files are written non-atomically, so an interrupted run leaves a truncated pool that the `causal`/`match`/`query` stages will load without complaint.
6. **`12_induction_robustness.py::DEFAULT_SOURCES`** maps each arm name to a *different results directory*. It is a results-reading tool, not a campaign stage, but which measurement run an arm's census is read from is keyed purely by name and deserves the same treatment as the arm lists fixed here.

### Verified sound (items 4–8 of the operator's list)

- **Control-anchored `recommend`** — sound, but incomplete: the script's own default was still `sorted(PANEL)`. Fixed.
- **`08 --arms` capability filter** — sound as a default, but never enforced. Fixed.
- **`01` plug-in vs held-out estimator** — sound. Held-out is the default, there is no fallback, `held_out_cohort` deduplicates by content and raises on an empty reference, and `disjoint_unigram_cross_entropy_nats` floors both totals. One gap: the reference block is itself a contiguous file-order block, and it moves with `--cohort-skip`. In the EXP-R2-060 sensitivity run the skip=4000 *pool digest* is byte-identical to the skip=0 *reference digest*, so both moved together. Decomposed for ProtGPT2: of the +0.599-nat context-information shift, the baseline contributes **+0.041** and clean CE **−0.558**. The finding stands; the artefact now records the reference cohort's sampling mode beside it.
- **`01` seeded cohort draw** — sound, with one limitation now recorded: the *pool* the seeded subsample draws from is still the first 4000 eligible records in file order, so the draw is seeded within a head-of-file block rather than over the corpus. `protein_cohort(seed=)` would fix it and would move published numbers, so it was not changed.
- **Markov held-out disjointness** — sound; content deduplication with the count recorded.
- **Name-based modality dispatch (item 2)** — the `arm_modality` enumeration was correct, but was itself a fourth hand-written copy. Four further name-based dispatches survived in the worker (`LENS_ARMS`, `RELATIONAL_ARMS`, the cohort_power vocabulary split, `PROTEIN_ARMS` for homology), two of them wrong. All are now derived.
- **`model_var_for_arm` (item 3)** — the `R2_TEXT_MODEL_BASE_DIR` branch is present and correct for all eleven arms; now derived from the declaration.

### Tests

`tests/test_transfer_stage_contract.py`, **50 tests**, one per behavioural fix, written against the restored property. The bash-level fixes are covered by extracting the real functions from `h200_worker.sh` and `run_transfer_h200.sh` and exercising them against the real generated contract — `build_command`'s cohort_power dispatch, the item-scoped arguments, the duplicate-flag refusal and the pod redaction are all tested on the shipped code rather than on a description of it.

`ruff check scripts/transfer/ scripts/transfer_gap/tg_common.py` clean. `PYTHONPATH=. pytest -q tests` → **398 passed, 12 subtests** (348 before).

### Compute

CPU only. No GPU was allocated at any point; the L20 cards in use throughout belonged to other tracks and were untouched. `results/` was read but not written.

## 2026-07-29 — EXP-R2-064: homology / memorisation control re-run without repeat masking; head-count claim survives, peak-strength claim reverses

**Task.** Re-run the retracted EXP-R2-050 homology control after the `--masking 0` fix (EXP-R2-061, audit §0.05), verify the fix on the known case, re-derive the stratified induction comparison under corrected bins, respect the `MINIMUM_BOOTSTRAP_UNITS = 8` floor, and re-check the ZymCTRL falsification.

**Environment.** `conda activate ct`. **CPU only, no GPU at any point** — the L20 cards were 100% allocated to other tracks throughout and were untouched; `peak_gpu_bytes` is `null` in every artefact written here. DIAMOND v2.1.24 (binary sha256 unchanged) at 64 threads against the full local UniRef50 snapshot, 60,315,044 sequences / 17,282,055,793 letters, coverage 1.000000. Induction stage on CPU at `--dtype float32`, `OMP_NUM_THREADS=48`. No H200 work.

**Artefacts.** New directories; the retracted ones are untouched evidence.

- `results/transfer_20260729/homology_control_unmasked/` — exact criterion, n=48
- `results/transfer_20260729/homology_control_unmasked/approximate_criterion/` — n=817
- `results/transfer_20260729/homology_control_masked_bins_cpu_control/` — device control
- logs `logs/runtime/r2_homology_rerun_{search,induction}_{exact,approx}_20260729.log` and `logs/runtime/r2_homology_maskedbins_cpu_control_exact_20260729.log`

Both re-run cohorts reproduce the retracted cohort digests exactly (`3ea68a87…` exact, `3cb7bac9…` approximate) and the exact-criterion `cohort_query.faa` is byte-identical (md5 `e31c7445…`), so the only changed input is the search.

### 1. Record-0 verification — the fix took

Cohort record 0 is byte-identical to `UniRef50_Q3E8Z8` over all 732 residues (re-verified here directly against `data/uniref50/uniref50.fasta`). Under `--masking 0` DIAMOND now returns `qstart 1`, `qend 732`, `pident 100`, `identity_over_query 100.0`, `best_hit_looks_truncated false`, stratum `ge95_near_duplicate`. Previously 82.9% over 607 residues, binned `id70_to_95_close_homology`. Same subject accession either way.

### 2. The search was wrong well beyond the five known records

Exact cohort (n=48): 19 records changed identity, 5 moved up a bin (`id70_to_95 → ge95`: records 0, 4, 21, 32, 47) and 5 more moved `id30_to_70 → id70_to_95`. Records at 100% identity 25 → 30. Best hit spans the record's own repeat 35/48 → **46/48**. Bins (records / distinct sequences) 0/8/11/29 → **0/3/11/34** and 0/4/11/22 → **0/3/7/27**.

Approximate cohort (n=817): 352 identities changed, 279 upward; 89 records changed bin, 82 upward. Best hit spans the repeat 730 → **799** of 817. Records at 100% identity 402 → **438**.

**The `lt30_no_detectable_homology` stratum does not exist.** Its four members were pure masking artefacts:

| record | retracted identity | corrected identity | corrected bin |
|---|---|---|---|
| 168 | 27.24 | 97.93 | ge95 |
| 169 | 27.24 | **100.00** | ge95 |
| 170 | 25.25 | 90.49 | id70_to_95 |
| 171 | 26.80 | **100.00** | ge95 |

Two are verbatim members of the corpus. Both retracted `consistent_with_memorisation` verdicts were decided by this stratum's four-unit interval. Under a masking-free search the `<30` bin is empty in **both** cohorts, and the minimum identity anywhere is 49.4 (approximate) / 65.3 (exact).

### 3. Bootstrap floor, applied

Every interval below is reported with its unit count. Splitting the two defects: applying the 8-unit floor to the **retracted** bins, holding everything else fixed, already flips ZymCTRL from `consistent_with_memorisation` to `indeterminate` (its `<30` interval was 15.1% relative width at n=4 against 26.7% at n=420); ProGen2-medium's verdict survives the floor alone. So the floor kills one verdict and the corrected search is what settles the rest.

Exact cohort, corrected: only `ge95` clears the floor (26–27 units against 3 at `30-70` and 6–7 at `70-95`), so **all three exact-criterion verdicts are `indeterminate` for want of power**, with the two small bins recorded as `underpowered_strata` carrying their unit counts and no interval. This is a change of statistical status, not of evidence: the same bins under the same CPU/dtype read `indeterminate` too.

### 4. Corrected stratified induction — approximate cohort (every bin above the floor)

| arm | bin | rec | units | peak/uniform [95% CI] | heads>0.10 [95% CI] |
|---|---|---:|---:|---|---|
| protgpt2 | 30-70 | 117 | 59 | 13.50 [9.31, 19.14] | 9 [7, 14] /720 |
| protgpt2 | 70-95 | 171 | 106 | 24.41 [17.60, 33.32] | 10 [6, 12] /720 |
| protgpt2 | >=95 | 529 | 360 | 35.42 [29.54, 42.51] | 13 [13, 13] /720 |
| protgpt2 | synthetic | – | 16 | **60.82** [57.40, 64.38] | 13 /720 |
| zymctrl | 30-70 | 117 | 92 | 5.04 [4.08, 6.46] | 0 [0, 0] /720 |
| zymctrl | 70-95 | 171 | 145 | 7.07 [5.53, 9.15] | 0 [0, 0] /720 |
| zymctrl | >=95 | 529 | 472 | 9.02 [7.94, 10.18] | 0 [0, 0] /720 |
| zymctrl | synthetic | – | 16 | 3.13 [2.87, 3.42] | 0 /720 |
| progen2-medium | 30-70 | 117 | 92 | 68.69 [53.39, 89.79] | 4 [3, 6] /432 |
| progen2-medium | 70-95 | 171 | 145 | 85.56 [71.65, 107.38] | 4 [3, 4] /432 |
| progen2-medium | >=95 | 529 | 472 | 113.27 [99.37, 128.37] | 4 [4, 4] /432 |
| progen2-medium | synthetic | – | 16 | **87.85** [87.19, 88.47] | **6** /432 |

### 5. Result, split by statistic

**Head count — the quantity the finding claims — survives, with more power than before.** `head_count_strata_separated` is **false** for all three arms and the head-count intervals **overlap** for all three. Counts are flat: ProtGPT2 9/10/13 of 720, ZymCTRL 0/0/0, ProGen2-medium 4/4/4 of 432. On the exact cohort the gradient is if anything inverted (ProtGPT2 16/13/14, ProGen2-medium 6/5/6). The synthetic probe, which appears in no corpus and cannot be memorised, recruits as many heads as the near-duplicate stratum (ProtGPT2 13 = 13) or **more** (ProGen2-medium 6 > 4), and largely the same heads — head-set Jaccard against synthetic 1.00 for ProtGPT2's `ge95` and 0.67 for ProGen2-medium's. The three retracted legs of the head-count argument all reproduce.

**Peak prefix-matching strength reverses toward memorisation.** The pre-stated interval rule now returns `consistent_with_memorisation` for **all three arms**; ProtGPT2 moves from `indeterminate`. The corrected gradient is monotone in every arm. This is a genuine reversal and is reported as one. Three things bound it, none retuned: mean repeat length still rises across the corrected bins (23.2 → 30.8 → 42.0 symbols) and the binless partial Spearman still makes length about four times the identity term (ProtGPT2 length|identity +0.599 against identity|length +0.142; ProGen2-medium +0.636 against +0.144); the unmemorisable synthetic probe, with a 64-symbol repeat, out-peaks **every** natural stratum on ProtGPT2 (60.82 against 35.42) and sits inside the natural range on ProGen2-medium; and EXP-R2-050 and EXP-R2-049 had already independently ruled the peak statistic unreportable. The correct reading is that peak sharpness was always the wrong statistic, and it now reads memorisation-consistent when measured properly. ZymCTRL is the exception worth naming: its partial correlations are identity +0.154 against length −0.005, i.e. identity carries the whole association — but ZymCTRL has **zero** heads above 0.10 at every stratum, so this is a statement about sub-threshold attention, not about induction heads.

### 6. ZymCTRL internal falsification — holds, and does not depend on the strata

ZymCTRL's declared `pretraining_corpus` is `uniprot_ec_annotated`, this cohort's own source, so it has the greatest memorisation opportunity of any arm. Under corrected bins it remains the weakest arm on both statistics and in both cohorts: **0 of 720 heads above 0.10 in every stratum**, pooled peak 8.09x uniform (approximate) and 18.12x (exact), against ProtGPT2 13/720 at 30.83x and ProGen2-medium 4/432 at 97.99x. If memorisation produced induction, ZymCTRL would be the strongest arm; it is the weakest. Unchanged by the correction.

### 7. Device control — the CPU move costs nothing

Re-measuring the **retracted** bins on CPU/fp32 reproduces the GPU/bf16 artefact to 3–4 significant figures on every stratum and reproduces every head count exactly (e.g. ProtGPT2 `ge95` peak 35.159 → 35.093, heads 14 → 14; ProGen2-medium `70-95` 230.598 → 230.967, heads 6 → 6). Every difference reported above is attributable to the corrected search and the unit floor.

### 8. Code

`scripts/transfer/10_homology_control.py` only, closing the three items EXP-R2-061 left for the owners of `scripts/`: `assign_homology` is now passed `max_target_seqs` so `hit_list_saturated` is a fact rather than `None` (47/48 and 812/817 queries saturate); `stratum_integrity()` is written into the search artefact; `adjudicate()` now excludes a stratum whose bootstrap came back `degenerate` and names it under `underpowered_strata` with its unit count, rather than subscripting a `None` interval; every published interval carries `bootstrap_n_units`; and the CUDA bookkeeping is conditional on the requested device so the stage can run on CPU. `ruff` clean; `PYTHONPATH=. pytest -q tests` → **398 passed, 12 subtests**, unchanged.

### 9. What this control can and cannot now resolve

The module docstring predicted it and the corrected search confirms it: 438 of 817 approximate-cohort records (54%) are verbatim UniRef50 members and the `<30` bin is empty, so a Swiss-Prot repeat cohort cannot supply a genuine low-homology arm. The stratified contrast is weaker evidence than it looked, and the synthetic-repeat negative control — which no database can contaminate — is what carries the head-count conclusion. GPT-2's corpus is still not public, so there is still no symmetric text arm and no matched cross-modal claim.

## 2026-07-29 — EXP-R2-065: documents, logs and results reorganised against the objective; retired scope archived

**Scope.** `{docs,manuscript,evidence,preregistration,literature,logs,results}` and repository-root `docs/`. No change to `src/` or `scripts/`. No GPU. No measurement was run and no number in any document was changed; this entry records a reorganisation and three defects found while doing it.

### 1. The rule applied

The repository serves one objective — differences, transferability, adapted methods — and `docs/INTERPRETABILITY_TRANSFER_AUDIT.md` is its canonical record. Material was classified as **retired scope** if it exists only to serve the prior conserved sparse-readout / EC-steering / enzyme-design / npj-manuscript programme, and **live** if this document, its L1–L19 catalogue, or its §9 plan depends on it. Where the two overlapped the live reading won, and the reason is written into the archive README rather than inferred from the directory name.

### 2. Archived

`archive/legacy/r2_retired_scope_20260729/`, 178 MB, with a provenance README carrying a what / why / where-it-lives-now table per item:

- **docs** — the 2026-07-16 procedure-and-results audit, three `NPJ_*` documents, the npj manuscript assessment, six `P0_*` protocol and receipt documents, the panel/table provenance map, two manuscript notes, and the 2026-06-12 expanded-dictionary analysis whose interpretations had already been withdrawn.
- **results** — 127 MB in twelve trees: `circuit_analysis/` (105 MB), `npj_revision_20260716/` (12 MB), `drug_design/` (6.0 MB), `h200_results_20260404/`, `npj_revision_20260717/`, `ec_metrics/`, `steering_benchmark/`, `causal_ablation/`, `checkpoint_evaluation/`, `steered_generation/`, `diagnostics/`, and one loose queue summary.
- **manuscript** (11 MB), **literature** (40 MB), **preregistration** (48 KB) — whole trees.
- **evidence** — four `h200_cleanup_*` receipts, `execution_environment_20260717`, `upstream_model_revision_recovery_20260717`, `recoverability_audit_20260605_1250`.
- **logs** — the April v2 queue log, and the root-level R1 and pre-rename R2 runtime log directories.

`archive/legacy/r2_transfer_log_snapshots_20260728/`, 103 MB, kept separate and named for what it is: sixteen point-in-time copies of the **live** `results/transfer_20260728/` tree plus one `convergence_control` backup, taken as a safety net after `git clean` destroyed the results tree three times. Its README carries a per-snapshot identity table. They are **not** duplicates — the earliest differs from the current tree in all 15 of its files and the 04:22–10:26 snapshots in 7–14 each, so they preserve pre-correction states. No snapshot holds a path absent from the live tree.

### 3. Kept, against the instruction to archive, and why

- **`results/final_checkpoints/`, 45 GB.** Four CLT runs from 2026-04-03 (ProtGPT2, ZymCTRL, ProGen2-medium, ProGen2-xlarge, `step_100000`). Referenced today only by retired-scope scripts, so on a naive reading it is retired. It is kept because it is the only protein dictionary held locally and plan item C1 is a dictionary re-qualification. Its training logs and its manifest (`evidence/historical_reference_checkpoints_20260717/`) were kept with it.
- **`results/transfer_gap_20260724/`, 1.6 GB.** §0.1 states it is retained unmodified because the audit cites it.
- **`evidence/p0_2*`, 5.5 MB** — five trees. The provenance chain of the 27 dictionaries: screening (which fixes the three terminal `sparsity_match_failure` model/methods, hence nine intentionally absent runs), full launch, mask validation, adjudication, and the P0-2b fidelity panel.
- **`docs/analysis/P0_2B_DICTIONARY_FIDELITY_RESULTS_20260727.md`.** L1's quantitative evidence. The P0-2b *protocol* was archived because the executed spec JSON supersedes it; the *result* stays, and L1 now cites it by path.

**A correction to the framing this task was given.** `results/final_checkpoints/` does **not** hold "the 27 dictionaries". It holds four April CLT runs. The 27 P0-2 dictionaries are on GPFS under `…/npj_revision_20260717/p0_2_dictionary_controls_bf16_r3/full/<model>/<method>/seed_<n>/`; their `best.pt`, `results.json` and `run_manifest.json` paths and SHA-256 digests are recorded in `evidence/p0_2b_fidelity_20260727/p0_2b_fidelity_spec.executed.json` (27 entries). Both artefacts are retained; they are different objects and C1 should not be planned as though they were the same one.

### 4. Sizes

| Tree | Before | After |
|---|---:|---:|
| `results` | 47,296 MB | 47,170 MB |
| `logs` | 105 MB | 2.3 MB |
| repository-root `logs` | 6.0 MB | 5.7 MB |

Nothing was deleted. Every move was `mv` within `/Data` (device 2080 on both sides), so no bytes were copied and no free space was consumed. `/Data` remained at 94% used, 627 GB available.

### 5. Documents rewritten

Repository-root `README.md` (the Paper A / Paper B framing, the drug-design claim discipline and six canonical result pointers into now-archived trees are gone), `r2_.../README.md`, `r2_.../docs/README.md`, `r2_.../docs/RESEARCH_PLAN.md`, `r2_.../docs/methods/TRANSFER_MEASUREMENT_PROGRAMME.md`, `docs/DOCUMENT_INDEX.md`, `docs/README.md`. `docs/PROJECT_STATUS.md` gained a true current-state head and its dated snapshots are retained verbatim beneath a divider that says they describe a retired scope and that their paths are historical identifiers. `docs/REPOSITORY_STRUCTURE.md` gained the corrected `git clean` exposure figures and a rule that `logs/` must never hold a copy of a result tree.

The audit document was touched twice only, both additive: L1's evidence cell now carries resolvable paths, and Appendix A.3 records this experiment and where the retired scope went.

### 6. Retracted-claim sweep

Every live document was searched for the seven withdrawn or qualified claims — the induction gap as a *modality* claim, the variance–behaviour dissociation, the QK/OV dissociation, the relational channel, the tokenisation conclusion, peak prefix-matching strength as a memorisation-free statistic, and the gpt2/dialogpt-small corpus contrast — by claim phrasing and by the specific numbers each rests on (`0.951`, participation ratio, `rank-512`, `2.30x`, `0.336`, `−0.105`).

**No live document asserted any of them as true.** They survive only inside the audit document, correctly marked, and inside `EXPERIMENT_LOG.md` and `PROJECT_LOG.md` as history. Two adjacent overclaims were repaired rather than retracted:

- `docs/methods/TRANSFER_MEASUREMENT_PROGRAMME.md` said a difference between GPT-2-large and ProtGPT2 "cannot be attributed to model scale or tokeniser cardinality, only to the text-versus-protein modality change". With one protein arm in the matched pair that reads as licence for a modality claim. It now states the four-stage decomposition without the identifiability sentence and defers panel reasoning to `RESEARCH_PLAN.md`, which states the n = 1 limit explicitly.
- `docs/RESEARCH_PLAN.md` listed the gpt2 / dialogpt-small pair as the text-side corpus contrast without qualification. It now records that contrast as retracted under §5.05(a), and describes a four-arm panel no longer — the panel is eleven arms.

### 7. Three defects found while reorganising

1. **`.gitignore` and `.mutagenignore` still named the pre-rename root.** `r2_decoder_sparse_readout_audit/{results,logs,wandb}/` no longer matches anything, so `logs/` was ignored by neither tool after the rename. The generic `results/` and `logs/` rules happened to cover git and sync respectively, so nothing was exposed — but the explicit per-root lines exist precisely so a rename shows up as a stale line, and they had stopped doing that job. Both files now name the current root and add `r2_decoder_sparse_readout_audit/` to their retired-name quarantine blocks.
2. **A bare filename in a directory quarantine block.** Both ignore files carried a bare `DOCUMENT_INDEX.md` line, intended for `project_records/DOCUMENT_INDEX.md` but matching at any depth. `docs/DOCUMENT_INDEX.md` — the repository's navigation map — was therefore excluded from version control *and* from synchronization between 2026-07-16 and today. Removed from both. **The Mutagen session must be reloaded for the sync half to take effect**; until it is, `docs/DOCUMENT_INDEX.md` differs between the two mirrors and the D-side copy is the current one. Both ignore files now say that the block takes directory patterns only, and `docs/REPOSITORY_STRUCTURE.md` records the rule.
3. **`docs/INTERPRETABILITY_TRANSFER_AUDIT.md` §5.05 states two qualifications twice.** The `--cohort-skip` / seeded-pool qualification on (b) appears at both the head of §5.05 and again after (e); the lens cohort-band qualification on (e) appears both inline and immediately after. The second copy of the (b) qualification adds one fact the first lacks (`protein_cohort` now accepts a `seed`, and a corpus-wide draw would be stronger than either block). **Not repaired here** — it is the canonical document, another agent may be editing it, and merging the two would mean rewriting a finding rather than a pointer. Reported for its owner.

### 8. Verification

A citation checker resolved every backtick-quoted path in the live navigation and status documents against the repository root, the containing document's directory, and the R2 root. Every path-shaped citation in `INTERPRETABILITY_TRANSFER_AUDIT.md` still resolves: `results/transfer_20260729/`, `results/transfer_gap_20260729_corrected/`, `archive/legacy/r2_transfer_gap_precorrection_20260729/`, `src/transfer/arms.py`, `scripts/transfer/panel_contract.py`, `scripts/transfer_gap/tg_common.py`, `scripts/transfer_gap/tg00_input_contract.py`. One citation was re-pointed: `docs/analysis/P0_2B_DICTIONARY_FIDELITY_RESULTS_20260727.md` cited the P0-2b protocol, which moved; it now names both the archive location and the executed spec that supersedes it.

Broken citations that remain, deliberately: `docs/PROJECT_STATUS.md`'s dated snapshots and `docs/PROJECT_LOG.md`'s historical entries point into `results/circuit_analysis/`, `r1_encoder_interpretability_benchmark/` and `evidence/h200_cleanup_*`. Those are history and are not rewritten; both files now say so at the point where history begins.

No code was touched, so no test was run.

---

## 2026-07-29 — EXP-R2-066: `scripts/` and `src/` reduced to the objective's computed import closure; a scoring contract, a TG stage contract, and three live defects

**Task.** Refactor code under `scripts/` and `src/` so that what remains serves the three-part objective and nothing else. Scope: code only. `results/`, `docs/`, `manuscript/`, `evidence/`, `preregistration/` and `literature/` were handled concurrently by other agents under EXP-R2-065, into the same archive directory.

**Environment.** L20 host, `conda activate ct`. No GPU work: every check here is static analysis, unit tests, or a `--dry-run`.

### 1. The import closure, computed rather than judged

The boundary was not a decision about which directories looked relevant. An AST walker resolved every `src.*` import — absolute *and* relative, transitively — from the objective's entry points (`scripts/transfer/*.py`, `scripts/transfer_gap/*.py`) and from the four tests covering them.

Before this entry the closure reached, beyond `src/transfer/`:

| module | symbols actually used |
|---|---|
| `src/revision/io.py` | `sha256_file`, `write_json` |
| `src/revision/statistics.py` | `mean_interval`, `paired_group_bootstrap` |
| `src/revision/dictionary_fidelity.py` | `analysis_layer`, `source_layers_for_target`, `sequence_target_mask`, `per_sequence_scores`, `aggregate_variant` |
| `src/revision/nested_recoverability.py` | `make_group_splits` |
| `src/revision/dictionary_controls.py` | `WindowedTranscoder`, `load_strict_json` — **import-time only** |

**Twelve symbols out of ~6,600 lines.** The last row is the load-bearing observation: `dictionary_controls.py` is 2,775 lines and no function the transfer package calls touches it. It was in the closure solely because `dictionary_fidelity.py` imports it at module level, for `encode_source`, `reconstruct_target` and `load_fidelity_spec` — none of which `src/transfer` reaches.

`src/models/`, `src/training/` and `src/analysis/` were **never** in the closure.

The twelve symbols were vendored into `src/transfer/io.py`, `src/transfer/statistics.py` and `src/transfer/scoring.py`. The closure is now exactly `src/transfer/` (16 modules) plus `src/__init__.py`, which the H200 controller's own independently-implemented closure walker confirms: `--dry-run` reports `import closure added 1 file(s) outside src/transfer and scripts/transfer:
+ src/__init__.py`.

The rationale the old dependency rested on had already expired. `src/transfer/__init__.py` and `docs/RESEARCH_PLAN.md` both stated that stage 2 "reuses `src/revision/dictionary_fidelity.py` and `dictionary_controls.py` so that transfer measurements and the production P0-2b qualification share one windowed-transcoder implementation". B2 returned NO and C4 was dropped (EXP-R2-062), so there is no live measurement on either side of that sharing.

### 2. Archived

To `archive/legacy/r2_retired_scope_20260729/` (the directory EXP-R2-065 was already filling), with a per-file provenance table appended to its `README.md`: what each file is, why it was archived, and where its output still lives.

- **79 numbered `scripts/`** (`00_`–`79_`, `.py` and `.sh`), plus `scripts/diagnostics/` (3), `recoverability_audit.py`, `render_lysozyme_cartoons.py`, and two standalone H200 runners.
- **`src/models/`, `src/training/`, `src/analysis/`, `src/revision/`** — 28 modules, ~26,000 lines.
- **25 test files, 241 tests.**

Nothing was deleted. `results/final_checkpoints/` (45 GB) stays live, per EXP-R2-065's reasoning: it is an input to plan item C1.

### 3. Three live defects, found by the work rather than looked for

**(a) Two depth conversions disagreed on the ProGen2 arms.** `src/transfer/relational.py` carried `int(round(f * (n_layer - 1)))` while `dictionary_fidelity.analysis_layer` used `floor(f * (n_layer - 1) + 0.5)`. Python's `round` is round-half-to-even. At `0.25 * 26 = 6.5` — exact in binary, and exactly the 27-layer ProGen2 case — the first returns **layer 6** and the second **layer 7**.

`09_probe_and_erasure.py` reached the first (via `probes.analysis_layer_grid` → `relational.analysis_layers`); `02_pathway_budget.py`, `03_estimand_power.py` and `08_lens_family.py` reached the second. Two stages of one campaign disagreed about what relative depth 0.25 meant, on the two arms carrying the protein-side corpus contrast.

**Blast radius, measured rather than assumed.** `results/transfer_20260728/probe_erasure/progen2-medium.json` records `design.analysis_layers = [6, 13, 20]` for `layer_fractions = [0.25, 0.5, 0.75]`; under the unified conversion it would be `[7, 13, 20]`. The two other depths are unaffected on every panel arm. Critically, `design.erasure_layer = 13` — depth 0.50 — so **L16's anchor, "ProGen2-medium relies on ss3 at only +0.031 nats", is not affected**, nor is any other figure in §5.1. What moves is the depth-0.25 *probe-skill* row for `progen2-base` and `progen2-medium`, which the audit does not cite. `relational.analysis_layers` is deleted; `scoring.analysis_layer` is the one conversion.

**(b) `tg01_information_budget.py` could not run at all.** It registered `--seed` **twice** on one parser (defaults `20260724` and `20260729`). `argparse` raises `ArgumentError: conflicting option string: --seed` during parser construction, so the stage died before its first line of measurement. Verified directly: `python tg01_information_budget.py --help` raises. A whole TG stage was dead and nothing in the repository noticed, because nothing read those arguments.

**(c) Nine TG stages restated `DEFAULT_COHORT_SEED` as a literal, one of them wrong.** `tg_common.DEFAULT_COHORT_SEED = 20260729` exists, and its docstring says so, precisely "so that two stages of the same run draw from the same ordering and `skip` is a genuine partition across scripts as well as within one". `tg01`'s first `--seed` defaulted to the pre-correction `20260724`, which makes its permutation different from every other stage's and its skip-disjointness against them a fiction. All eleven stages now default to the constant.

### 4. Sanitised

**A name-dispatch rule replaced by a declared one.** `sequence_target_mask` selected its masking rule by matching the arm's *name* against the literal `"zymctrl"`; for any other EC-conditioned arm it returned the plain mask and **discarded the boundary token ids it was handed without a word**, so the EC tag, separator and terminator would be scored as cohort content. Two workarounds had accreted around that: `budget.require_boundary_masking_support` (a 30-line allow-list refusing unknown EC arms) and, in `probes.py`, synthesised fake names (`f"{arm.name}_plain"`) to steer the dispatch. Per the Repair principle, redesigned rather than extended: the rule is now a parameter, `scoring.target_rule(input_format, ec_conditioning=...)`, derived from `ArmSpec.input_format`. Both workarounds are gone and a second EC-conditioned arm is handled correctly by construction rather than refused by an allow-list. The `all_valid` rule now **raises** if handed boundary ids, because accept-and-discard is exactly how the predecessor failed.

**Duplicated declarations collapsed.**

| declaration | copies before | now |
|---|---:|---|
| `write_json` | 3 (`revision/io`, `tg_common`, plus non-atomic behaviour in the third) | `src/transfer/io.py`; `tg_common` delegates and keeps its progress line |
| depth fraction → layer | 8 (2 in `src/`, 6 in `scripts/transfer_gap/`) in two inconsistent rounding modes | `scoring.analysis_layer` |
| `<start>` / `<end>` literals | 4 (`arms` renderer, `budget`, `lenses`, `prediction_addressed`), one of which skipped the `unk_token_id` check | `arms.CONDITIONING_START/END` + `arms.conditioning_boundary_ids` |
| conditioning boundary id lookup | 3 | `arms.conditioning_boundary_ids` |
| cohort seed | 10 | `tg_common.DEFAULT_COHORT_SEED` |

The depth unification is **behaviour-preserving on every fraction any TG stage actually passes** (0.15/0.33/0.5/0.67/0.85 and 0.25/0.75), checked across all eleven panel arms before the change: no recorded TG number moves.

**Two declarations that nothing read are now checked.** `SAMPLING_MODES` is validated by `sampling_record`, and `TEXT_ARCHITECTURE_CONTRAST` — the declaration behind §5.05(d)'s "survives replacing GPT-2-large with a Qwen2 and a Llama decoder" — is checked at import for the property it asserts. That argument is what killed the QK/OV finding; leaving its supporting declaration unread was the shape of the next such failure.

Stale docstrings describing the retired scope were corrected in `pathways.py`, `probes.py`, `prediction_addressed.py`, `src/transfer/__init__.py`, `scripts/transfer/README.md`, `scripts/transfer_gap/README.md`, `run_transfer_h200.sh` and the R2 `README.md`. Historical references to `src/revision` in incident write-ups were **kept** — they are the record of why the code-freeze closure walker exists — and reworded so they cannot be read as describing a live dependency.

### 5. A contract for `scripts/transfer_gap/`

`scripts/transfer/panel_contract.py` (EXP-R2-063) covers the campaign stages. The TG series — the part of the repository with the most retractions — had no equivalent, and carried the identical defect. **The eleven TG stages draw protein cohorts on three different residue bands, none declared:** 400–1000 (TG-01, TG-02, TG-06), 120–1000 (TG-03, TG-07, TG-08, TG-09), 64–246 (TG-10). That is Appendix B rule 13, and §5.05(b) prices protein cohort-block sensitivity at 0.16–0.60 nats. TG-01 and TG-10 share no protein at all.

`scripts/transfer_gap/tg_contract.py` declares per stage the arms it measures, its residue band with the reason it differs from the reference band, and its cohort seed. `--verify` reads each entry point's `argparse` defaults back out of the source with `ast` — the technique `panel_contract.declared_arms_in_source` uses — and refuses on any disagreement. It found defects (b) and (c) on its first run. `stage_contract_record()` is the block a stage writes into its artefact, so the band travels with the number.

Verified to bite: seeding `tg09` at `20260724` and moving its band to 64–1000 produces two refusals and exit 2; reverting restores `12 TG stages agree`.

### 6. Tests

| | count |
|---|---:|
| before | **398** |
| retired with the code they tested | −241 (25 files) |
| `test_transfer_audit_invariants.py`: 1 workaround test → 8 invariant tests | +7 |
| `TEXT_ARCHITECTURE_CONTRAST` / `SAMPLING_MODES` invariants | +2 |
| `tests/test_transfer_gap_contract.py` (new) | +16 |
| **after** | **182** |

The 241 retired tests are named by group in the archive README. The one test *rewritten* rather than retired was `test_an_ec_conditioned_arm_the_masker_does_not_know_is_refused`, which asserted the **workaround** (`BOUNDARY_MASKED_ARMS` contains `"zymctrl"`). It is replaced by eight tests of the property the redesign restores — that the rule follows `input_format` and not the arm's name, that the plain rule refuses boundary ids rather than ignoring them, that one depth conversion governs every stage.

`ruff check` is clean over the whole of `` (190 errors before; 177 of them were in the archived scripts, and the 13 in `scripts/transfer_gap/` are fixed). All 29 live entry points import cleanly under the worker's preflight technique. `panel_contract.py --verify`, `tg_contract.py --verify` and the H200 controller `--dry-run` all pass.

**A failure that is not mine, recorded so the count is honest.** Between the baseline run and archiving, nine tests in `tests/test_dictionary_gate.py` began failing on `FileNotFoundError: docs/P0_2_DICTIONARY_PROTOCOL_20260717.md`, which EXP-R2-065 moved into the archive while the test still read it at line 399. Those tests are retired-scope and were archived; both halves of the break now sit in the same archive directory. Had they not been archived, the fix would be to restore the document beside them.

### 7. Findings for other owners

- **`docs/RESEARCH_PLAN.md`** says "Stage 2 has no new module: it reuses `src/revision/dictionary_fidelity.py` and `src/revision/dictionary_controls.py` … Interval estimation reuses `src/revision/statistics.py`." All three paths are now under `archive/`. `src/transfer/__init__.py` carries the corrected statement; the plan needs the same edit. It also lists six `scripts/transfer/` entry points where there are now eleven.
- **`configs/`** (18 files) is entirely retired scope — five CLT training YAMLs and thirteen npj-revision spec examples — and nothing live reads it. It is outside this task's scope so it was left in place, but six of its files are cited by frozen `evidence/` receipts, so it should be archived *with* that citation recorded, not deleted.
- **Nothing found here invalidates a number in the audit document.** Defect (a) moves only the depth-0.25 probe-skill rows for the two ProGen2 arms, which the audit does not cite; L16's +0.031 and +0.705 are at depth 0.50 and unaffected. Defect (b) means TG-01 has produced nothing since the corrections landed, and this is visible in the artefacts: `results/transfer_gap_20260729_corrected/` holds `tg00`, `tg03`, `tg07` and `tg09` only, with **no `tg01/`**. TG-01's sole outputs are the four files in the contaminated `results/transfer_gap_20260724/tg01/`. **No TG-01 number is currently quotable**: the corrected run does not exist and the 2026-07-24 one is under the §0.1 rendering retraction. TG-01 is now runnable again, at CPU-adjacent cost, and the audit's TG-01 row should say "not re-run" rather than "corrected" until it is.

---

## 2026-07-29 — Canonical interpretation corrections after evidence audit

**Scope.** Documentation only. No historical experiment entry was rewritten, no result was recomputed, and no experiment was launched. `docs/INTERPRETABILITY_TRANSFER_AUDIT.md` was corrected against the existing artifacts and this entry records the evidentiary basis.

1. **Scale-adjusted induction shortfall.** The **2.34x** shortfall against the GPT-2 scale-ladder prediction remains descriptive because the artifact reports no inferential test for that adjusted statistic. The artifact-backed **p = 0.0286** is detached from it: `results/transfer_20260728/induction_robustness/induction_robustness_synthetic_repeat.json::model_level_tests.fraction_above_threshold` defines that p-value as the one-sided exact permutation test over four text and three protein model-level `fraction_above_0.10` values, where it is the minimum attainable p under complete separation.

2. **P0-2b interpretation.** The matched-panel or cross-domain reading is inconclusive because P0-2b had no matched text dictionary, and the original ProtGPT2 and ZymCTRL recovered ratios were denominator-invalid. The ProGen2-medium result remains a valid within-arm negative under the original estimand and frozen 0.80 gate: `evidence/p0_2b_fidelity_20260727/p0_2b_fidelity_panel.json` records all nine ProGen2-medium runs as failures with valid recovered ratios, and the original EXP-R2-032 entry records best bootstrap upper bounds of 0.411 loss recovered and 0.282 KL recovered. The current attainability result is EXP-R2-060, `results/transfer_20260729_instrument/estimand_power/recommendation.json`: `mlp_single@d0.50@cohort_mean` is unattainable on gpt2-large and ZymCTRL, attainable on ProtGPT2 and both ProGen2 arms, 52 of 76 alternatives are attainable on gpt2-large, and 50 are powered panel-wide. The older 32-of-48 result is retained only as historical TR-027 / EXP-R2-027 (2026-07-28).

3. **Copy-suppression census scope.** EXP-R2-059 establishes only that `paa_specific` fails as a cheap ranking screen for copy-suppression on GPT-2-large. The selected heads and measured effects are retained in `results/transfer_20260728/paa_gate/selected_heads.json`, `results/transfer_20260728/paa_gate/causal.json` and their matrix sidecars; the EXP-R2-059 entry reports the resulting Spearman −0.062 (p = 0.71) over 40 tested heads and 5/56 heads above the control band. No protein arm was scored. This does not establish a general limitation of head-prevalence censuses, another proxy, or an exhaustive causal effect-size census.

4. **Homology and memorisation scope.** EXP-R2-064 supports no separation of head count across the measured protein homology strata and reports that the synthetic repeat probe recruits as many heads as the near-duplicate stratum or more. It does not refute a cross-modal memorisation contribution: `results/transfer_20260729/homology_control_unmasked/approximate_criterion/panel_summary.json` and the EXP-R2-064 entry show that the genuine `<30` stratum is absent, 438/817 approximate-cohort records are exact UniRef50 members, peak strength reverses toward memorisation, and GPT-2 has no symmetric corpus control.

5. **Resolved and exploratory statuses.** B2 was completed by EXP-R2-062 and returned NO, so it is not an unresolved question. The current estimand result is EXP-R2-060 rather than the historical TR-027 grid. C1 is exploratory calibration unless a threshold is frozen before protein-arm evaluation; a threshold selected after observing the text control is not itself a preregistered confirmatory threshold.

6. **Checkpoint verification wording.** EXP-R2-058's general prose and addendum 4 assert full source-to-GPFS SHA-256 verification, including Qwen2.5-0.5B, but the contemporaneous Qwen extraction record and validation table preserve only exact byte-size checks for its seven files. No retained digest manifest supplies the claimed Qwen hashes. The canonical wording is therefore evidence-based: all eleven checkpoints were staged and load-checked; ten have retained full SHA-256 verification statements tied to named weight files, while Qwen2.5-0.5B has byte-size-only retained evidence.

7. **Stable aliases.** The append-only log reused EXP-R2-025 through EXP-R2-032. Historical ids remain unchanged; the canonical audit now uses `TR-025` for EXP-R2-025 (2026-07-24), `TR-026` for EXP-R2-026 (2026-07-27), and `TR-027` through `TR-032` for the corresponding 2026-07-28 transfer entries. EXP-R2-060 through EXP-R2-066 are explicitly indexed because they carry the current instrument, retraction, qualification, homology and stage-contract evidence.

---

## 2026-07-30 — EXP-R2-068: repository-wide audit; the campaign channel could not report failure; plan items B3 and B6 run

**Note on the identifier.** `EXP-R2-067` is referenced by comments in committed code (`src/transfer/probes.py`, `scripts/transfer/08_lens_family.py`, `13_induction_probe_bootstrap.py`, `h200_worker.sh`, `tg05_relational_channel.py`) but has no entry in this log. That work is in `HEAD` and unlogged; this entry does not claim it. This session takes **068** to avoid the collision that renumbered EXP-R2-050.

**Scope.** Full audit of the live code under the five Development Principles, then the two plan items the audit's Phase B could reach. Four regions were audited in parallel by Opus sub-agents — the H200 controller and worker, `src/transfer`, `scripts/transfer`, `scripts/transfer_gap` — and every finding acted on was verified independently before any change. Two reported findings were **overstated and were corrected rather than acted on**: the plug-in measurability gate is a bad library default, not a corrupted campaign number, because `01_cohort_power.py` recomputes the verdict from the held-out estimator; and the FASTA chunk-boundary off-by-one is real but did not fire on the corpus in use, verified from the shipped artefact (`source_fasta_records == indexed_sequences == 60315044`, coverage 1.0), so EXP-R2-064 stands.

### The finding that matters most: no remote predicate could return false

Measured directly: `h200_pod_exec.sh -- bash -c "exit 7"` returns **0**, and `h200_pod_bash.sh "exit 5"` returns **0**. The access layer does not propagate a remote exit status. Three controller call sites depended on it:

1. **the worker invocation** — a worker that refused a campaign at preflight and scheduled no GPU was reported as `campaign complete`, exit 0;
2. **`verify_remote_snapshot`** — the remote half of the code-freeze guarantee could not fail, so it was never actually checked;
3. **`push_run_manifest`** — always took its "already present and verified" branch. **No invocation manifest had ever been pushed.** Every run directory on GPFS holds `INVOCATIONS=0` while the controller logged that the manifest was present and verified.

A campaign's verdict, its frozen code and its provenance record all travelled on a channel that cannot say no. Repaired inside the repository, because the access layer is external and shared: the worker states its own status on its last line (`TRANSFER_WORKER_EXIT=`, declared once in the worker and read out of that source by the controller), and every remote predicate answers on stdout through one `pod_predicate` helper that refuses an unrecognised reply. A missing sentinel is itself a failure, which covers a killed worker and a dropped tunnel.

**Verified live, in both directions, three times unplanned.** A zero-eligible-arm campaign now exits 1 and logs `worker reported exit status 1 (access layer said 0)`. During B6 the SSH tunnel dropped mid-run: the worker never reached its exit handler, no sentinel was emitted, and the controller refused to report completion (exit 90) — which is how I learned B6 had not finished rather than believing it had. ZymCTRL's two genuine stage failures were caught the same way. Catalogued as **L20**.

A second live defect surfaced with it. The panel contract's arm-to-variable map was *inferred* by comparing resolved paths, and the pod sets `TRANSFER_TEXT_MODEL_BASE_DIR="${TRANSFER_MODEL_BASE_DIR}"` because all checkpoints share one GPFS directory. The comparison aliased and six of seven text arms classified as protein-root arms **inside the pod only**, so the contract verified on B and disagreed in the pod. The worker's own re-derivation refused the campaign before any GPU was scheduled — the check earned its keep. `ArmSpec.path_variable` now declares the choice where the path is made. Catalogued as **L21**.

### Cohort draws

Nine campaign stages called `protein_cohort`/`text_cohort` without a seed, so the corpus-level pool was a head-of-file prefix and only the within-pool subsample was seeded. This is the open qualification on EXP-R2-060 §5.05(b) and EXP-R2-061's deferred item 4. One declaration (`arms.DEFAULT_CORPUS_DRAW_SEED`) now reaches every stage; `0` selects the historical draw. Measured exposure on the qualifying band, CPU only: a 400-record file-order draw holds **342 distinct sequences against 398** under a seeded draw, top EC-1 class share 0.458 against 0.370, mean length 169.4 against 186.4 residues, and the two draws share **3 of 400** records. A file-order cohort is 14.5% repeated sequences, which is the mechanism behind the +1.01 nat incident, now measured. A new `tests/test_cohort_draw_contract.py` caught one call site I had missed.

### Plan item B3 — explanation channel under seeded permutation. CPU. **Contrast survives.**

Every channel now draws under a seeded permutation of its whole corpus, and each unit list is visited in seeded order so the `--max-units` cut is not a prefix of the draw. Bits/symbol in one 300-symbol window, with grouped intervals and an independent second draw seed:

| channel | file order | seeded, 95% CI | second seed |
|---|---:|---|---:|
| text token identity | 7.32 | **7.326** [7.316, 7.336] | 7.331 |
| residue identity | 4.11 | **4.094** [4.089, 4.098] | 4.097 |
| structural attributes | 3.61 | **3.792** [3.746, 3.837] | 3.783 |
| Pfam domain label | 0.74 | **0.860** [0.837, 0.883] | 0.889 |

Drawn from all 574,627 eligible Swiss-Prot entries and all 23,586 AlphaFold models. The two channels that had been prefix draws moved **in the predicted direction** — Pfam +16%, structural +5% — because a family-grouped prefix is more uniform in its labels than the corpus and so understates a label channel's entropy. The text control moved +0.006 bits, which is what shows the effect is protein-specific. Text-to-Pfam ratio 8.5x against 9.9x before: closure stands, slightly smaller than recorded.

### TG-01 — run for the first time, closing audit §0.15

The recorded cause was stale: `--seed` is registered once and `--help` exits 0. The stage had simply never run. It has now run on all four TG arms under seeded draws consumed in seeded record order, with the held-out estimator. **The retracted 2026-07-24 figures moved a long way**: ProtGPT2's clean NLL falls 7.296 → **4.846** nats, its information gain rises 2.636 → **5.735** bits, top-1 rises 0.102 → **0.266**. ZymCTRL's long-range shares are **refused, not reported** — information range 0.202 nats is below the 0.5-nat denominator floor.

### Plan item B6 — non-local propagation at production scale. ~1.5 H200 GPU-h. **Gate met, claim not earned.**

`activation_patching` was made to run chunked forward passes (verified numerically identical to the single-batch form: 0 integer mismatches, max float delta 1e-4) so case counts could rise ~16x, and the per-band eligible fraction gained an interval that **resamples the source sequence rather than the case**, because cases are many corruptions of the same few sequences.

| arm | eligible / total | fraction | 95% interval | source sequences |
|---|---:|---:|---|---:|
| gpt2-large | 48 / 288 | 0.167 | [0.122, 0.215] | 168 |
| protgpt2 | 65 / 256 | 0.254 | [0.202, 0.307] | 162 |
| progen2-medium | 58 / 128 | 0.453 | [0.369, 0.538] | 100 |

All three clear the pre-registered ≥30 far-band cases (against 2–16) and every interval excludes zero. But **gpt2-large and ProtGPT2 overlap** on [0.202, 0.215]: at production scale the matched pair — the only modality-identifying comparison the panel has — is not separated, and what separates is ProGen2-medium from everything else, which is architecture and tokenisation as much as modality. gpt2-large was first run at 224 cases/band and returned 29 eligible, one short of the gate; rather than reinterpret a gate narrowly missed, it was re-run at 288. The two agree within interval.

Two corrections fall out. **The recorded trio 9% / 34% / 50% is not reproducible from the retained artefacts** (their far-band eligible fractions are 0.250 / 0.219 / 0.500 / 0.062; the "2–16 eligible cases" caveat matches 8 / 7 / 16 / 2 exactly). And **ZymCTRL cannot enter the estimand**: `build_patch_cases` cuts rows to the patching window and a conditioned rendering puts the `<end>` marker delimiting scored content hundreds of tokens beyond it, so no valid span exists. Reaching it costs an ~816-token window (~2.5 GPU-h) or an incommensurable short band. Recorded, not worked around — the conditioning prompt (L15) removing an arm from a window-based estimand. Two prior failures of this run were caught by the guard rather than producing a number, and its `--unigram-max-tokens` default of 256 is mutually inconsistent with its own 600–1000 residue protein band for that arm.

### Other repairs

Bootstrap unit floors unified and applied where missing; **eight `excludes_zero: true` separation verdicts in the path-patching panel summary rest on 3–6 heads and are withdrawn** (the audit's headline matched-pair interval resamples 14 and is unaffected). Three interventions that had a negative control and no positive control were given one or had the unfalsifiable guard deleted with the reason stated. The held-out unigram baseline's Laplace smoothing was found to carry a vocabulary-tracking bias — reproduced independently at **+0.303 nats at V=50257 against +0.0003 at V=32** — now measured, reported and swept rather than asserted to be conservative; the default is unchanged because no constant is uniformly better. In the TG series the retracted all-position residual spectrum is no longer the default-named field, seeded cohorts are consumed in seeded order, a hard-coded host path and a stale constant imported from a retracted run are gone, and stage eligibility moved from arm-name literals into the contract. `10_homology_control.py` now refuses to overwrite one repeat criterion's artefacts with another's.

### Verification

**303 tests plus 34 subtests** (from 205 plus 9), Ruff over `src`, `scripts` and `tests`, generated-panel verification including under aliased environment variables, the TG stage contract, shell syntax on both orchestration scripts, and live cluster checks of the worker-status and remote-predicate paths in both directions. B6's frozen snapshot was verified on GPFS with the corrected predicate.

### Left open, with reasons

**Corrected after review.** An earlier draft of this entry said B6's invocation manifest was absent. That was true of the *aborted* four-arm attempt, which launched under the pre-fix controller; it is **false of the three runs the result comes from**. Each of `20260730023608_…` (gpt2-large), `20260730022334_…` (ProtGPT2) and `20260730022555_…` (ProGen2-medium) holds exactly one manifest under `INVOCATIONS/`, each filename equals the SHA-256 of its own contents, and each run's frozen snapshot verifies against `CODE_CONTENT_SHA256SUMS` on GPFS. B6's provenance is complete; only the aborted attempt lacks a manifest, and it produced no result. The `cohort_power` text item still skips all seven text arms if one checkpoint is missing. Panel-wide stages still write fixed filenames into a shared results root, so a narrowed re-run overwrites a full-panel artefact; B6 used run-scoped roots. Four TG stages still report cross-entropy over all scored positions only, and the README now names the four that hold the alphabet-bearing accounting instead of claiming all of them do.

---

## 2026-07-30 — EXP-R2-068 addendum: external review closed; the induction census re-derived; B6's magnitudes withdrawn

An external review of the previous entry raised five findings. Four were real; the fifth proposed a remedy that is wrong for this transport, and I did not adopt it.

**The repeat cohorts were never seeded, and this one mattered most.** The seed plumbed through the campaign reached the *analysis* cohorts and not the *repeat* cohorts, so the induction census — the stage carrying the headline part-1 result — drew **32 of 817** matching proteins and **32 of 968** matching documents in corpus file order under the approximate criterion. About four per cent of the eligible population, taken from the head of a family-grouped corpus. The contract test passed throughout because it covered only `protein_cohort` and `text_cohort`.

Re-derived on all eleven arms at the whole matching population — **817 of 817** on the protein side, so there it is a census and order cannot matter, and 817 of 1022 on the text side. **The ordering now holds at the headline threshold on all three probe constructions.** The recorded inversion of the natural approximate probe at 0.10 does not reproduce: `dialogpt-small`, which read 0.0000 and produced the inversion, reads **0.0278**; the worst text arm (llama-3.2-3b, 0.0268) sits above the best protein arm (ProtGPT2, 0.0181). Matched pair 5.46x synthetic against the recorded 5.38x. The problem is reduced and not removed — natural approximate still inverts at 0.20 and 0.30, natural exact at 0.30, in every case because the worst text arm reaches exactly zero while a protein arm keeps one or two heads out of 720–1728. That is a small-count tail effect, and nothing here converts the result into a modality claim.

**B6's magnitudes are withdrawn, and the reason is a dtype.** Both sensitivity checks the review asked for were run. The draw does not drive the result: a disjoint window of the same permutation moves the far-band fraction by −0.021 / −0.020 / +0.016 with every interval overlapping. The threshold does drive it: swept from 0.05 to 1.0 the ordering **reverses** at the two most permissive cuts, and at 0.05 the text control propagates *more* than either protein arm. Then the threshold-free quantiles showed why the sweep looked odd — every far-band `|effect|` was an exact multiple of 1/16. The run was bfloat16 and the quantisation step was the size of the quantity. Re-running gpt2-large in float32: quantiles become continuous and the far-band fraction falls **0.1667 → 0.1042**, inflated by 60% relative and upward. Every bfloat16 far-band number in §5.1 is withdrawn. The gate-attainment result survives (float32 gives 30 eligible cases at 288 per band, exactly at the floor) because it does not depend on the magnitudes being resolved.

**The review's one wrong remedy.** It asked that the worker's exit sentinel be required to be the last line of the log. I implemented that, and it broke: `kubectl exec` appends `command terminated with exit code N` *after* the remote output, so on every failing run the sentinel is second-to-last, and the rule turned a correctly reported failure into "no sentinel". Reverted to uniqueness plus a numeric range, which gives the same guarantee against a stage quoting the constant without depending on what the transport appends. Recorded because the reasoning is the record: position is strictly stronger where it holds, and it does not hold here.

**Also fixed.** Two remaining file-order draws: `probes.text_units` took a `seed` and applied it only to positions, over a 150-of-396,000 prefix; and `01_cohort_power.py` restated the seed as a literal rather than importing the declaration every other stage is told to match. The eligibility sweep is now self-consistent — one float64 array behind the headline fraction, its interval and every swept row, the run's own cut always present in the ladder, one RNG stream. The eligibility floor is imported from `statistics` rather than restated beside a justification that was wrong twice. A guard I had added at argument time is replaced by a per-arm resolver: the campaign dispatches `circuit_primitives` as one process, so refusing for the group would have lost ten arms to fix one.

**Corrections to the previous entry.** B6's invocation manifests are **not** missing — that was true of the aborted four-arm attempt and false of the three runs the result came from; each holds one content-addressed manifest and each frozen snapshot verifies on GPFS. B6 cost **0.22 GPU-h**, not ~1.5. The L9 intervals are Student-t over per-unit values, not clustered. The path-patching headline resamples **38** heads (14 against 24), and it carries `excludes_zero: false`, so it was never a separation verdict and the floor had nothing to withdraw from it.

**The plan is restructured** around the three directions of the objective, with the old Phase A/B/C names kept in a mapping table so existing citations resolve. Remaining budget re-estimated at 55–95 H200 GPU-hours; about 8 have been spent, and the fall is mostly because measurements costed as campaigns turn out to be minutes once case counts are sized to the gate rather than to the panel.

**Still running at the time of writing:** the float32 re-run for ProtGPT2 and ProGen2-medium, and the D2.b causal effect-size measurement at 64 cases per arm (ProtGPT2's usable case count is the binding constraint — an instance of L14's coverage collapse, which refused 128 and 160 before this). Neither is quotable until it lands.

Tests 205 → 309 across the two commits. Ruff, both generated contracts and shell syntax pass.

**Second addendum, same day.** D2.b failed on ZymCTRL with the identical `<end>`-truncation refusal that had just been fixed in `04_circuit_primitives.py` — because the fix was written *inside that stage*, and `11_induction_path_patching.py` carries the same 256-token unigram default against the same 1000-residue protein band. That is Appendix B rule 12 charging for a second copy within hours of the first: a decision made properly one import away, made again locally. `conditioned_token_budget` and `CONDITIONING_TOKEN_SLACK` now live in `src/transfer/circuits.py` beside `fit_unigram` and `content_bounds`, both stages import them, and a test asserts neither stage redefines either or fits its unigram on the unresolved window. Worth stating plainly: my own repair had the shape of the defect it repaired.

**Third addendum, same day: two blockers, one code, one infrastructure.**

*Code.* With the shared token budget in place, D2.b measured all four arms — ZymCTRL included, at 171 s and 6.5 GiB — and then died on its last statement, writing the panel summary: `payload["structural_invariants"]["passed"]` raised `KeyError`. `path_patching.structural_invariants` had published a `passed` flag that could never be false, because any failure raises on the next line, and this session removed it as an unfalsifiable guard. That removal was right. What was missing is that nothing checked its consumers, so a library change with a correct motivation cost a full four-arm H200 measurement at the last line. The consumer now records the honest verdict — a returned record *is* the pass, because a failure raises — and a test asserts no stage reads the removed key and that the key really is gone, so the test cannot go vacuous if it returns.

*Infrastructure.* The relaunch and the two remaining float32 B6 arms both died mid-run when the jump host became unreachable (`Connection timed out during banner exchange`); the reverse SSH tunnel is down. This is outside the repository — `~/hangzhou-remote/README.md` is authoritative for recovery — and needs the tunnel task restarted on the Windows host before any further campaign can run. Both controllers refused to report completion, which is the third and fourth time this session that the worker-status work has correctly turned a dropped connection into a failure rather than a green campaign.

**Not quotable, and not recorded as results:** the float32 correction for ProtGPT2 and ProGen2-medium, and D2.b. Only gpt2-large's float32 far-band number exists. D2.b's per-arm payloads were computed but the run failed, so it carries no manifest and must be re-run rather than salvaged.

**Fourth addendum: the H200 came back, and four results landed. Two of them change conclusions.**

**B6 in float32 reverses the bfloat16 conclusion.** All three arms re-run, and every bfloat16 far-band number was inflated by ~0.06 absolute:

| arm | bfloat16 | float32 | 95% interval | eligible | clusters |
|---|---:|---:|---|---:|---:|
| gpt2-large | 0.1667 | **0.1042** | [0.0694, 0.1399] | 30/288 | 168 |
| protgpt2 | 0.2539 | **0.1953** | [0.1492, 0.2424] | 50/256 | 162 |
| progen2-medium | 0.4531 | **0.2734** | [0.2015, 0.3548] | 35/128 | 100 |

Three things the bfloat16 run got wrong. The **matched pair is separated**, not overlapping: gpt2-large's q975 0.1399 sits below ProtGPT2's q025 0.1492, and gpt2-large is disjoint from both protein arms while the two protein arms overlap each other — which is the shape a modality difference should have and the opposite of what the quantised run showed. The ordering is **threshold-invariant**: text < ProtGPT2 < ProGen2-medium at all six cuts from 0.05 to 2.0, over a fortyfold range, where the bfloat16 sweep had it reversing at the permissive end. And all three clear the ≥30 case gate. So the quantisation was manufacturing the very threshold-dependence I reported as a limitation. This is the strongest D1 result the programme has: the far-band propagation difference survives production scale, resolved arithmetic, a threshold sweep, and the matched-pair comparison that is the panel's only modality-identifying contrast. The n=1 structural limit and the corpus-repeat confound are untouched, and the sampling check still needs redoing in float32 — the disjoint-window runs were bfloat16.

**D2.b is not answerable by the instrument it was written for, and its Jaccard of 1.0 is arithmetic.** `11_induction_path_patching.py` computes a causal effect only for heads the census already selected by `prefix_matching >= threshold`. Verified directly: the per-head causal records are *exactly* the selected sender set (57 / 14 / 6 / 8 heads on gpt2-large / ProtGPT2 / ProGen2-medium / ZymCTRL). A top-20 Jaccard against that same census is therefore 1.0 by construction on every arm, and I nearly reported it as a pass. The gate needs causal effects for heads the census did **not** select, which is the exhaustive census §8 item 1 already flags as needing its own design; restated as D2.b′ at 20–30 GPU-h.

**Corpus-wide cohort power ran at two disjoint windows** (skip 0 and skip 4000), which is the measurement the EXP-R2-060 qualification asks for: the recorded 0.16–0.60 nat cohort-block sensitivity was seeded *within a head-of-file pool of 4000*, and this pair is seeded over the whole corpus. Artefacts under `results/transfer_20260730/cohort_corpuswide/`.

**Audit corrections to the third addendum, all against the artefacts.** The bfloat16 quantile triple I quoted, "(0.0625, 0.1250, 1.0000)", is wrong twice: those artefacts carry no quantiles field at all, and the three far-band medians are 0.0625 / 0.1250 / **0.0000** — over half of ProGen2-medium's far-band effects underflowed the grid to exactly zero, which is a stronger fact than the one I stated. 17 of 18 band medians across the three bfloat16 arms are exact multiples of 1/16 and the 18th is an even-count average of two grid points. "One or two heads out of 720–1728" is wrong: no panel arm has 1728 heads, the maximum is gpt2-xl's 1200, the protein totals are 432 and 720, and the retained counts are 7 and 5 at natural-exact 0.30 and 4 and 3 at natural-approximate 0.30. The claim that the higher-threshold inversions are all driven by the worst text arm reaching zero holds at natural-exact 0.30 and natural-approximate 0.20 — both rest entirely on `dialogpt-small`, the arm §5.05(a) declares unmeasurable — but **not** at natural-approximate 0.30, where llama-3.2-3b at 0.00298 sits below all three scored protein arms and the inversion survives dropping the zero arm. And `induction_seeded/panel_summary.json` covers only the seven text arms: the protein arms ran as a second invocation and the summary was never regenerated, so §4's eleven-arm table is a hand-assembly across two runs whose seeds and cohort digests match.

**One provenance limit worth stating.** The bfloat16 arm of the dtype comparison cannot be reproduced from the repository: its artefacts carry `_case_resampled_interval` but not the threshold sweep, so no committed revision produces exactly that pair of fields. The `eligible_fraction` definition is unchanged across the revisions involved, so the delta is attributable to dtype — but on a code reading plus artefact fields, not on a controlled A/B of one revision.

**Three defects fixed in the code, one of them mine twice over.** The consumer repair I made for `structural_invariants` replaced an unfalsifiable flag with `"structural_invariants" in payload` — a compile-time constant over a dict literal that always has the key, i.e. the same flag one layer out. It is gone; only the basis string remains. The eligibility sweep keyed rows by `f"{threshold:g}"`, six significant digits, so a `--patch-minimum-effect` of 0.100000001 collided with the ladder's own 0.1 and silently dropped a row — the row labelled with the run's own cut would have been a different threshold's, with every stated invariant reading as satisfied. Keys are now `repr` with a count check that raises. And the controller comment claiming uniqueness "gives the same guarantee" as last-line position was false: uniqueness is strictly weaker, the residual exposure is a worker killed before its trap while a stage has printed the prefix, and that is now recorded as accepted rather than asserted away.

**Fifth addendum: the panel grew by one arm, on evidence, and a candidate survey was recorded.**

Five ProGen2 checkpoints were staged on GPFS and only two were in the panel. Load-checked all of them on the pod before deciding anything: `progen2-small` 151.1M / 12x1024 / 16 heads / vocab 32, `progen2-medium` 764.8M / 27x1536, `progen2-large` 2779.4M / 32x2560, `progen2-xlarge` 6443.6M / 32x4096, every one loading and returning logits from a real forward pass.

**Admitted: ProGen2-small.** It gives the protein side a within-lineage scale rung it did not have — 151M against ProGen2-medium's 765M, same architecture, same residue tokeniser, same UniRef90+BFD30 mixture. That matters beyond one more arm: the text side has had a four-rung ladder since EXP-R2-057 and every scale-adjusted restatement of the head-count shortfall has assumed the text-side slope transports to protein, which nothing had tested. Declared as `PROTEIN_SCALE_CONTRAST` rather than a ladder, because two rungs give a slope and no curvature.

**Refused, with the measured reason, not deferred silently.** `progen2-large` declares `vocab_size` 51200 against a 31-token tokenizer, so 51169 of its logit rows are unreachable and every `config.vocab_size`-derived statistic — held-out unigram support, plug-in entropy, the rank-(V-1) aperture of L8 — would be computed over a mostly dead alphabet and would not be comparable with the other ProGen2 arms. It is a natural experiment on whether that aperture tracks the output matrix or the reachable symbols, and it deserves a gated design rather than a quiet admission. `progen2-xlarge` carries no `vocab_size` key at all, only `vocab_size_emb` and `vocab_size_lm_head`, and `budget.arm_power` reads `config.vocab_size` directly — so admitting it would raise inside `cohort_power`. Both are now in `panel_contract.STAGED_BUT_NOT_ADMITTED`, which is a new declaration: "staged and load-checked but not admitted, because admitting it would corrupt a statistic" is a decision, and it was previously indistinguishable from an absence.

**A bucket name that stopped being true.** `cohort_power`'s dispatch put ProGen2-small into the item named `protein_progen2_base`, whose rule is "protein, residue-level, not EC-conditioned, default dtype" and whose name described its only occupant. Renamed to `protein_default_dtype`. Three contract tests then failed on hard-coded panel sizes (`len(eligible) == 9`, a literal three-arm relational list, a four-name protein list) and one on the README's hand-maintained stage table — all of them legitimate consequences of a twelfth arm, and the README test catching its own drift is the mechanism working. The counts are now derived from the contract instead of restated, and the README table is regenerated from it.

**Candidate survey recorded as audit §9 D1.c**, covering MolCrawl protein GPT-2-large as the highest-value candidate for breaking the single-protein-arm limit, RITA, ProGen3+ProGenMech as the external baseline a D3 construction must replicate, Tranception for interface attribution, Pythia for seed calibration only, and Dayhoff for a data-mechanism study — each with the risk that has to be answered before admission. Four corpora are recorded as evidence packages with their traps: temporal extrapolation from UniParc timestamps, PDB-CATH out-of-family structure splits, independent MaveDB assays grouped by wild type, and a repeat-mechanism 2x2 which is the only design that would actually test the corpus-repeat confound §4 calls irremovable. And one caution the survey earns: predicted structures and sequence-inferred labels cannot demonstrate functional competence, which is the second half of the objective and currently has no instrument at all.

**Launched, long-running:** a twelve-arm campaign over `cohort_power` then `circuit_primitives --sections induction`, which qualifies ProGen2-small on the frozen cohort and puts it into the seeded census beside the other four protein arms, giving the protein-side scale contrast on the headline statistic. Results root `results/panel12_20260730`.

**Sixth addendum: the text-side scale slope does not transport to protein.**

The twelve-arm campaign completed cleanly — `cohort_power` then `circuit_primitives --sections induction`, 11 minutes on three H200s. Cohorts verified before reading anything: 817 of 817 matching proteins (a census, so order cannot matter) and 817 of 1022 text, with probe counts 808 / 817 and ProtGPT2 at 597 of 817, the multi-residue-BPE coverage loss L14 records.

**The measurement ProGen2-small was admitted for.** Every scale-adjusted statement this programme has made about the induction shortfall carried the GPT-2 ladder's slope onto the protein side, because the protein side had no ladder. It now has a two-rung contrast, and the slope is the same in sign and much shallower in magnitude:

| probe | text (4 rungs) | protein (2 rungs) | ratio |
|---|---:|---:|---:|
| synthetic | −0.0735 / decade | −0.0172 | 4.3x shallower |
| natural exact | −0.0572 | −0.0172 | 3.3x shallower |
| natural approximate | −0.0446 | −0.0238 | 1.9x shallower |

ProGen2-small reads 0.0260 on all three probes (5 heads of 192) against ProGen2-medium's 0.0139 / 0.0139 / 0.0093. So a larger protein decoder does put a smaller fraction of heads in the upper tail — the direction transports. The magnitude does not, and that cuts against the programme's own previous correction: the 2.34x scale-matched restatement removed gap using a slope three to four times too steep for the protein side, so **scale explains less of the shortfall than recorded, not more**. The 2.34x figure is a lower bound on the adjusted shortfall, not its value. Two rungs give a slope with no curvature and no interval, and the two rungs differ in depth and width together exactly as the text ladder's do.

Also visible at matched scale: the protein corpus contrast is small, ProGen2-base 0.0116 against ProGen2-medium 0.0139 on the synthetic probe, both 765M — consistent with the audit's existing reading that corpus is worth roughly half the modality contrast in log terms rather than all of it.

**Seventh addendum: the matched-pair separation is draw-dependent and is withdrawn.**

The float32 sampling check I had flagged as the last gap in the corrected B6 result has run, and it does not confirm the claim I made from the single window. On a disjoint window of the same permutation, gpt2-large is essentially immovable — 0.1042 [0.0694, 0.1399] becomes 0.1042 [0.0712, 0.1388] — while ProtGPT2 moves by −0.051, from 0.1953 [0.1492, 0.2424] to 0.1445 [0.1028, 0.1880]. On that window the two intervals overlap. **The matched-pair separation is withdrawn.** What survives both windows is the point ordering: gpt2-large below ProtGPT2 by 0.091 on one and 0.040 on the other.

The asymmetry is the more useful result, and it is one this programme already knows in another guise. The text arm's far-band fraction barely moves with the draw; the protein arm's moves by a quarter of its own value. That is §5.05(b)'s protein-specific cohort sensitivity — recorded there as 10–35x larger on the protein side, measured in nats of context information — reappearing in a completely unrelated statistic. The generalisation worth carrying: a protein arm's number carries a selection uncertainty its matched text control does not, so **any protein-versus-text interval comparison computed on a single cohort window overstates its own precision**, including the one I published an hour ago. ProGen2-medium's disjoint window has not been run, so the three-arm ordering's robustness is untested.

This is the third claim this session that a sensitivity check has weakened rather than confirmed — after the threshold sweep on the bfloat16 run and the dtype re-run itself. The pattern is worth naming: every one of them was a claim made from a single condition, and every one of them moved when the condition was varied.

**Running in parallel at the time of writing:** `pathway_budget` and `estimand_power` across all twelve arms under corpus-wide seeded cohorts (fanned over two GPUs — neither stage has been re-run since the draw was fixed), and float32 far-band propagation for ProGen2-small and ProGen2-base, which would extend that measurement to five protein arms and give a scale contrast on propagation as well as on head count.

**Eighth addendum: far-band propagation on five arms, and two scale effects with opposite signs.**

Float32 far-band propagation now covers five arms. All four protein arms sit above the text control by point estimate — gpt2-large 0.1042 against ProGen2-small 0.1797, ProtGPT2 0.1953, ProGen2-medium 0.2734, ProGen2-base 0.3125 — and three of the four are interval-disjoint from it on this window.

**ProGen2-small is not a gated result and is reported as underpowered.** It returns 23 eligible far-band cases against the pre-registered floor of 30, so its overlap with the text control is an absence of power, not a measured overlap. It needs roughly 176 cases per band; 128 was chosen from ProGen2-medium's eligibility rate, which is higher.

**The interesting result is a dissociation between two scale effects.** On the ProGen2-small / ProGen2-medium pair, with corpus, architecture and tokeniser held fixed, far-band propagation rises at **+0.1330 per decade** while the induction head fraction falls at **−0.0172 per decade**. A larger protein decoder in this lineage carries a single-token perturbation *further* while devoting a *smaller* fraction of heads to prefix-matching. An account in which protein decoders simply have "less induction machinery" predicts those two moving together; they move apart. Two rungs, no interval, so this is a direction and not a magnitude — but it is a direction that constrains the mechanism, and it exists only because the protein side now has a scale contrast at all.

**Every separation in that table is provisional on one cohort window**, and the previous addendum is the reason to say so rather than a formality: the disjoint-window check moved ProtGPT2 by −0.051 and dissolved its separation. Disjoint-window runs for ProtGPT2, ProGen2-medium and ProGen2-base are running on three GPUs; ProGen2-small's would need the larger case count first.

**Ninth addendum: the pathway-budget separation survives the cohort-draw fix, and reproduces to two decimal places.**

`pathway_budget` and `estimand_power` ran across all twelve arms on corpus-wide seeded cohorts, fanned over two GPUs, 21 minutes. This stage had never been re-run since the draw was fixed, so §5.05(d) — the strongest part-1 result on file — had been resting on file-order cohorts throughout.

It reproduces almost exactly. Text: qwen2.5-0.5b 2.10 → 2.070, gpt2-xl 2.01 → 2.005, gpt2-medium 1.88 → 1.836, gpt2 1.77 → 1.754, llama-3.2-3b 1.69 → 1.675, gpt2-large 1.50 → 1.491. Protein: ProtGPT2 1.12 → 1.129, ProGen2-medium 0.93 → 0.930, ProGen2-base 0.92 → 0.922, ZymCTRL 0.55 → 0.530. Every arm within 0.05 of its recorded value, and the newly admitted ProGen2-small lands inside the protein range at 1.057. The ranges still do not overlap: 1.491–2.070 against 0.530–1.129, now on six text and five protein arms. `dialogpt-small` returns 13.606 with `measurable_every_seed = false` and is excluded as the off-distribution arm §5.05(a) already retracts, not quietly dropped.

**The methodological point is the contrast between two statistics measured this same session.** The pathway ratio is essentially immovable across cohort draws; the far-band propagation fraction moved by a quarter of its own value on ProtGPT2 between two windows of one permutation. Cohort sensitivity is a property of the statistic and not of the modality alone, so neither can be inferred from the other — and a draw-robustness result for one measurement licenses nothing about another. That cuts both ways here: it strengthens §5.05(d) considerably and leaves the propagation separation exactly as provisional as the previous addendum found it.

**Tenth addendum: the far-band propagation ordering survives for one lineage and fails for the matched pair.**

The disjoint-window float32 runs landed for gpt2-large, ProtGPT2 and ProGen2-medium (`--cohort-skip 256`, everything else identical to EXP-R2-068). Reading the threshold ladder in both windows rather than the headline cut alone:

At the headline threshold 0.25 the ordering holds in both windows — gpt2-large 0.1042 → 0.1042, ProtGPT2 0.1953 → 0.1445, ProGen2-medium 0.2734 → 0.3203. At the permissive cuts 0.05 and 0.10 it does not: ProtGPT2 falls *below* the text control in window 2 at both. ProGen2-medium is above gpt2-large in both windows at all four thresholds and interval-disjoint in both at 0.25 and 0.50 — the only cell robust to draw and threshold simultaneously.

**The retained claim narrows from a modality to a lineage.** ProtGPT2's margin over its matched text control is smaller than the movement between two draws of one permutation, so the matched pair — the panel's only modality-identifying comparison — does not support the propagation contrast. What is left is that the ProGen2 lineage propagates a single-token perturbation further than the text control. That is the same conclusion §5.1 had already reached from the architecture-and-tokenisation side, now forced independently by the sampling side.

**Third appearance of the cohort-sensitivity asymmetry.** gpt2-large returns 0.1042 in both windows to four decimal places while both protein arms move by 0.03–0.15. §5.05(b) recorded it at 10–35x in nats; the withdrawn matched-pair separation recorded it a second time. It is now a property of the panel, not an incident.

**Two design faults, both fixed in the successor.** The arms carried unequal case counts (288 against 128), so an interval-overlap test was partly reading sample size. And two windows sample a draw distribution without characterising it — at K=2 a shifted arm and a noisy arm are indistinguishable. EXP-R2-070 launched: five disjoint windows (skip 0/256/512/768/1024) at 256 cases per band for every arm, across gpt2-large, gpt2-xl, ProtGPT2, ProGen2-small, ProGen2-base, ProGen2-medium and ZymCTRL, float32, four GPUs. This also gives ProGen2-small the case count its failed gate needs and a second text arm to test whether immovability is a text property or a gpt2-large property.

**Eleventh addendum: the shallower-protein-scale-slope pattern replicates on estimand power, and gains a fixed-architecture data control.**

The 12-arm `estimand_power` artefacts from the same campaign. Powered fraction of the 76 ablation estimands, by lineage: GPT-2 falls 88.2% → 73.7% → 68.4% → 63.2% across 124M/355M/774M/1.56B, monotone over four rungs and a 12x parameter range, slope −22.5 points/decade. ProGen2 falls 100.0% → 94.7% across 151M/765M, slope −7.5 points/decade. Same sign, 3.0x shallower — inside the 1.9–4.3x band the induction census gave on a completely different measurement.

The new piece is a control the census version did not have. ProGen2-base and ProGen2-medium are architecturally identical in the artefacts (d_model 1536, 27 layers, 32 symbols) and differ only in pretraining mixture; they read 92.1% and 94.7%. So a corpus difference at fixed architecture moves this statistic 2.6 points where a 5x size difference moves it 5.3, which is what makes the lineage slope readable as a size effect. ProtGPT2 is likewise confirmed architecturally identical to gpt2-large (1280 wide, 36 layers, 50257 vocab) — the matched pair really is matched.

Two honest limits recorded with it. Powered fraction measures the estimand battery's sensitivity on an arm, not the arm's mechanism, and ProGen2-small sits at 100.0%, a ceiling that bounds the protein slope's magnitude from below on its own. The modality ranges overlap heavily — text 0.0–96.1%, protein 65.8–100.0% — and separate nothing; the text range reaches zero only because dialogpt-small returns 0 powered and 0 context-valid estimands, which is the off-distribution arm §5.05(a) already retracts.

**Twelfth addendum: D2.b′'s instrument built, validated, and queued; plus an unplanned determinism check.**

*The instrument.* `select_senders` gained an `exhaustive` criterion admitting every head, and `causal_census_agreement` computes the causal-versus-census comparison D2.b asked for. That function raises on a non-exhaustive sender set rather than documenting the precondition, because D2.b's failure was invisible in its own output — a top-20 Jaccard of 1.0 is exactly what a genuine agreement would also give, so nothing in the artefact distinguished "the rankings agree" from "the rankings ranked the same six heads". `above_threshold` became each head's own score against the threshold instead of the set-level answer broadcast to every head: provably identical on both pre-existing criteria, and the only truthful answer on a set spanning the threshold, which is the field that identifies the census's misses. Primary statistic is the rank correlation over all heads, needing no cut; the top-k Jaccard is reported beside it and swept over k = 5/10/20/40. Six tests; suite at 324.

*Validation, not a result.* GPT-2, 144 heads exhaustive, 8 cases, 90 s on L20 GPU 6 (idle before and after; host 115 GiB of 503 GiB). Spearman ρ = +0.70 (p = 7e-23), top-20 Jaccard 0.538, six of the causal top-20 below the census threshold, strongest L8H9 at prefix-matching 0.018 and causal rank 8. Eight cases per condition is far too few for a per-head effect, and ranking noise depresses Jaccard by itself, so these numbers are not quotable. They establish only that the statistic can now fall below 1.0 and that census misses can appear in it — neither of which was possible before.

*Determinism, observed rather than planned.* Consolidating the float32 skip-256 artefacts turned up two ProtGPT2 files with different md5s. They are two separate executions of one seeded configuration, 55 minutes apart, and they agree exactly: same cohort digest 272a1859abc622b9, same 37 eligible far-band cases, same 0.1445 fraction, runtimes differing by 0.11 s. The pipeline is reproducible across executions at this configuration. Both retained under `results/transfer_20260730/_superseded/` with a README, per Appendix B rule 18.

*Scheduling.* The far-band five-window queue (EXP-R2-070, 31 jobs) runs four GPUs at ~100%. Sharding is per arm at the controller, not per campaign: stages 04 and 11 are panel-scoped inside the worker — they write `panel_summary.json` from the process that writes the per-arm JSON — so one controller holding seven arms serialises every arm onto GPU 0 and idles three GPUs. The first attempt did exactly that and was killed after 10 minutes; giving each shard its own results root removes the overwrite hazard that motivates the worker's rule. D2.b′ is chained behind it on a driver that waits for the queue to drain, so the GPUs do not idle between campaigns.

**Wake point for the two chained campaigns (launched 2026-07-30 15:56 +08:00).**

Both run detached on B and survive the session ending. Nothing needs doing until they land.

- **EXP-R2-070, far-band five-window queue.** 31 jobs, four GPU lanes, driver `logs/drivers/farband_queue.sh`, progress in `logs/farband_queue.log`, per-job logs `logs/farband_g<gpu>_<arm>_s<skip>.log`. Observed per-job times: ProGen2-small 7 min, text arms 35 min and up. Expected drain **≈20:30–21:00 +08:00**. Results land in the pod at `results/farband_20260730/skip<N>/<arm>/circuit_primitives/`.
- **D2.b′, exhaustive per-head causal census.** Chained behind it on `logs/drivers/d2bprime_queue.sh`, which polls for the far-band driver to exit; log `logs/d2bprime_queue.log`, per-arm `logs/d2bprime_<arm>.log`. Four arms in parallel, one GPU each, roughly 4 h. Expected completion **≈00:30–01:00 +08:00 on 2026-07-31**. Results at `results/d2bprime_20260730/<arm>/induction_path_patching/`.

**[SUPERSEDED — the far-band queue landed at 20:37 and its analysis is the thirteenth addendum below. D2.b′ was restarted at 21:01 and is expected ≈01:00–01:30 on 2026-07-31; on waking, read its four artefacts at `results/d2bprime_20260730/<arm>/induction_path_patching/`, primary statistic the rank correlation over all heads with the swept top-k Jaccard beside it. The ZymCTRL 816-token far-band job still needs a re-run after a transport failure.]**

**On waking, in order.** Confirm both drivers exited and the pod GPUs are idle (Appendix B rule 19). Pull the far-band artefacts and run the paired per-window comparison: for each arm and each threshold on the ladder, the sign of `f_arm − f_gpt2-large` across all five windows, plus the between-window spread. That is the test EXP-R2-069 could not perform at K=2 — it decides whether ProGen2-medium's margin over the text control is a shift or noise, and whether ProtGPT2's crossing is systematic. Then check ProGen2-small's far-band case count against the ≥30 gate it failed at 23, and whether ZymCTRL entered at all at the 816-token window. Only then read D2.b′: the primary is the rank correlation over all heads, with the swept top-k Jaccard beside it, and the question is whether the causal ranking surfaces heads the prefix-matching census scored below threshold.

**If a lane died.** Controllers refuse to report success on a transport failure, so a non-zero exit in `logs/farband_queue.log` means that job produced nothing and should be re-run rather than trusted. The queue is idempotent per job: re-running one `(arm, window)` pair writes only its own results root.

**Thirteenth addendum: EXP-R2-070 lands — the far-band separation is a large-effect-tail phenomenon, and its gate was never attainable.**

31 jobs, four GPU lanes, 15:56 to 20:37 (+08:00), 30 clean. Six arms × five disjoint windows × 256 cases per band, float32.

*The gate.* gpt2-large returns 26 / 24 / 26 / 28 / 34 eligible far-band cases at 256 per band — under the pre-registered ≥30 floor in four of five windows. gpt2-xl misses it once. Appendix B rule 2 makes an unattainable positive-control gate a specification defect, and this one was violated in the direction that cost a protein arm: ProGen2-small's recorded "gate FAILS at 23" was measured against a floor its own text control does not clear at the same case count. gpt2-large met it before only by running 288 cases per band and landing on exactly 30. Withdrawn as stated. ProGen2-small clears it here anyway (33–62).

*The result.* Counting every arm × control × window comparison: protein higher in 31/40 at threshold 0.05, 33/40 at 0.10, 36/40 at 0.25, 39/40 at 0.50 — with zero text-higher cells at 0.50 and one exact tie. Monotone across the ladder. So the modality separation lives in the **large-effect tail**, which is the shape §5.1 inferred from a single window and can now be stated across five draws and two controls. ProGen2-base and ProGen2-medium hold at 40 of 40; ProtGPT2 and ProGen2-small hold only at the two stricter cuts. No arm-control pair reaches p < 0.05 on window sign alone — the exact test floors at 0.0625 at K = 5 — and the 40 comparisons share windows and controls, so no binomial over them is quoted.

*The asymmetry, on equal footing at last.* Between-window sd at threshold 0.25, every arm on 256 cases: gpt2-large 0.0150, gpt2-xl 0.0153, ProtGPT2 0.0243, ProGen2-small 0.0435, ProGen2-base 0.0476, ProGen2-medium 0.0519. Non-overlapping by modality, smallest protein spread 1.59× the largest text one. The three earlier appearances were all arguable as sample-size artefacts because arms carried different case counts; this one is not.

**Three operational faults this session, all mine, all recorded rather than smoothed over.**

1. *Seven arms to one controller.* Stages 04 and 11 are panel-scoped inside the worker, so that serialised every arm onto GPU 0 with three idle — 17 hours of work instead of 4. Killed after 10 minutes and rebuilt as a per-arm sharded queue.
2. *A chain that waited on itself.* The D2.b′ driver polled `pgrep -f logs/drivers/farband_queue.sh`, which matched its own launch wrapper, because the wrapper's command line carried the whole script text. It waited on its own grandparent and the GPUs sat idle from 20:37 to 20:45. A chain must not key on a pattern its own invocation can contain.
3. *ZymCTRL's 816-token job died on a transport timeout* during the snapshot push. The controller correctly refused to report success. It measured nothing and is queued for a re-run; whether ZymCTRL can enter the far-band estimand at all is still open (L15).

**D2.b′ restarted at 21:01 with `--repeat-cohort-size 48` on all four arms.** The first launch died on ProtGPT2 at 47 of 64 approximate-repeat path cases; the entry point refuses rather than running short, which is the Failure Principle working. 48 is not a tuned value — Swiss-Prot yields exactly 48 exact-repeat matching records out of 203063 eligible (0.024%), so it is the corpus ceiling, and a request for 96 fails on that criterion. Verified on an idle L20 before spending H200 time: both criteria reach 64 cases at 48. Applied to every arm rather than to the arm that needed it, because unequal case counts across arms are precisely what made the earlier far-band interval comparison partly a reading of sample size. The three already-running arms were killed and restarted so all four share one configuration.

**Fourteenth addendum (2026-07-31): EXP-R2-071 — the prefix-matching census is a causal proxy on text and is not one on protein.**

*Configuration.* `11_induction_path_patching.py --exhaustive-senders --repeat-cohort-size 48`, one controller per arm, four arms on four H200s, 21:01 to 01:07 (+08:00), all four exit 0. Every head patched: 720 (gpt2-large), 720 (ProtGPT2), 432 (ProGen2-medium), 720 (ZymCTRL). 64 path cases, all four sender × case conditions. In-pod GPUs 0 MiB / 0 % before and after; host 171–179 GiB of 2.0 TiB. Artefacts pulled to `results/transfer_20260730/d2bprime/`.

*Result.* Spearman ρ between prefix-matching score and causal-effect magnitude, over every head, range across the four conditions: gpt2-large **+0.428 to +0.507**, ProtGPT2 **−0.155 to −0.006**, ProGen2-medium +0.041 to +0.207, ZymCTRL +0.216 to +0.271. The text control's minimum exceeds every protein arm's maximum in every condition. At the matched pair the census carries essentially no information about causal rank.

The failure mode is recall. ProGen2-medium's top-5 Jaccard is 1.000 — its six selected heads really are the causally strongest — and collapses to 0.250 by k=20. The strongest head the census *rejects* carries |effect| 0.0194 on gpt2-large against 0.0497 (ProtGPT2), 0.0290 (ProGen2-medium) and 0.3049 (ZymCTRL), where the census selects nothing at all yet the strongest causal head in the whole grid is one it rejected.

*Consequence, stated narrowly.* An unknown part of §4's head-count gap belongs to the instrument rather than the models: the selector under-recalls causally-important heads more on protein than on text. This does **not** show protein models have induction heads after all. It shows prefix-matching score fails to rank causal importance on protein arms while working as a proxy on the text control.

*Gate withdrawn.* Top-20 Jaccard ≥ 0.8 is met by no arm in any condition; the best is 0.667 and the text control never exceeds 0.429. Second gate this session unattainable on its own positive control (Appendix B rule 2). Answered on the threshold-free ρ instead.

*Limits.* Path patching on repeat probes measures "causally important for repeat prediction", not "induction head by mechanism". 64 cases per condition; ranking noise depresses Jaccard on every arm alike, which is why the text control is the reference rather than an absolute cut. ProtGPT2's negative ρ is small and marginal (p ≈ 0.04 at n = 720) and reads as *uninformative*, not anti-correlated.

**ZymCTRL still cannot enter the far-band estimand, and the ~816-token estimate was too small.** The re-run after the transport failure reached the GPU and failed on the measurement itself: `content_bounds` refuses because the row still does not contain exactly one `<end>` at 816 tokens. With `--protein-max-len 1000` plus the EC conditioning prompt a row needs comfortably more than 1000 tokens, so the §5.1 estimate understated it. Not pursued further: at that window the arm cannot join the equal-length paired comparison EXP-R2-070 runs, so the cost buys an incommensurable number. L15 stands.

**EXP-R2-072 launched 01:16 (+08:00)** — the same exhaustive census on four more arms, `results/d2bprime_20260731/`: gpt2 and gpt2-xl ask whether the text-side correlation is a text property or a gpt2-large property, ProGen2-small and ProGen2-base ask whether the protein-side near-zero holds across the ProGen2 ladder. Identical `--repeat-cohort-size 48`, so the eight arms combine without a case-count confound. Expected ≈05:00–05:30 (+08:00).

**Fifteenth addendum (2026-07-31): an exploratory pathway split of the causal effect, and why it is not reported as one.**

The EXP-R2-071 artefacts carry each head's effect decomposed into direct, MLP-mediated, attention-mediated and interaction routes, so a modality contrast in *how* a head's causal effect is routed looked available at no compute cost. It is not, and the reason is worth recording because it borders on a result already on file.

The four components sum to the total **exactly** (residual 0.0 on every arm), but they carry **opposite signs on 40–50% of heads** — 39.9% on gpt2-large, 44.2% ProtGPT2, 48.2% ZymCTRL, 49.8% ProGen2-medium. A ratio of summed magnitudes across such a decomposition is therefore not a share of anything, and it shows: ProGen2-medium's "MLP share" computes to 1.03, above the whole it is supposedly a share of. The apparent contrast it produced — an MLP-to-attention ratio of 1.50 / 3.58 / 6.10 / 21.04 across the four arms, which *orders the modalities opposite to* §5.05(d) — is an artefact of that normalisation and is **not recorded as a finding**.

**§5.05(d) is not exposed to the same hazard, and the distinction matters.** Its ratio is built from whole-pathway ablation costs in nats, which are positive on every arm (verified: `mlp_all` and `attn_all` ΔCE both positive on all four), so there is no cancellation to hide. The two quantities are also not the same thing — one is the cost of removing a pathway, the other is how a single head's effect distributes over pathways — and a future reader should not treat the second as a check on the first. Answering the routing question needs a designed measurement with a signed decomposition, not a ratio of magnitudes; not scheduled.

The one durable observation: under `--exhaustive-senders` the sender criterion no longer changes *which* heads are patched (both criteria select all of them), so the four sender × case conditions collapse to two case sets crossed with two censuses. The per-head effects are consequently identical across sender criteria, while ρ still varies — because ρ compares against the census's scores, which do differ. EXP-R2-071's stability across all four conditions is therefore stability against *which census you compare to* as well as against the case set, which is a stronger check than it first appears.

**Sixteenth addendum (2026-07-31): a transport drop, and a qualification to the concentration result.**

*Transport.* At 02:02:48 (+08:00) five in-flight exhaustive-census workers died together — gpt2-xl, ProGen2-base, gpt2-medium, and both seed-2 robustness runs — each with controller exit **90**, the sentinel-absent code: the worker never reached its exit handler, and the access layer reported 255, which is not authoritative. The two jobs not yet started exited 2 on the health check. No artefact was promoted; nothing was reported as complete. This is the second time the L20 sentinel machinery has refused a false success, and it is the reason none of this needs re-auditing rather than re-running. Five orphaned pod-side processes survived the drop still holding GPUs and were killed before relaunch. The six lost jobs are requeued as a **work-stealing queue** rather than one arm per GPU — the earlier fixed assignment left GPU 0 idle for 35 minutes after gpt2 finished in 10 while gpt2-xl ran for 45. dialogpt-small is ordered first because it is the control that decides whether EXP-R2-071 is a modality result or a distribution-mismatch one.

*Qualification, found by a free check.* The EXP-R2-071 concentration contrast — Gini 0.940–0.944 on three protein arms against 0.826–0.829 on two text arms — is computed on the `cases_exact` set. The same heads and the same run on `cases_approximate` give gpt2 0.759, gpt2-large 0.752, ProGen2-small 0.772, ProtGPT2 0.822, ProGen2-medium 0.754, ZymCTRL 0.564. **The separation largely collapses**: ProGen2-medium falls below gpt2, ProGen2-small is within 0.02 of it, and only ProtGPT2 stays clearly above. The claim narrows to "protein concentrates causal effect *on exact-repeat probes*", which is a statement about a probe regime as much as a modality, and the "two measurements, one structure" reading of its relation to §5.1's far-band shape does not survive. The audit section is amended in place. The check cost nothing and would have been easy not to run, which is exactly why Appendix B rule 21's second half exists.

---

## 2026-07-31 — EXP-R2-072: the off-distribution text control holds, so the census result is not distribution mismatch

*Configuration.* The exhaustive per-head causal census extended from four arms to nine at the identical `--exhaustive-senders --repeat-cohort-size 48`, so the arms combine without a case-count confound; repeat-cohort digests are byte-identical across arms within a modality. New arms: gpt2-medium, dialogpt-small, ProGen2-base. Second-seed runs for gpt2-large and ProtGPT2. gpt2-xl was still running at the time of writing. Artefacts pulled to `results/transfer_20260730/d2bprime_ext/` and `d2bprime_seed2/`.

**The control answers the question it was ordered first for.** `dialogpt-small` is a *text* decoder that §5.05(a) shows is off-distribution on the evaluation corpus at −4.08 nats of context information. If the protein arms' near-zero census-to-causal correlation were distribution mismatch rather than modality, it should collapse with them. It reads **+0.518 to +0.640** — inside the text range and above gpt2-large. Being off-distribution on the scoring corpus does not destroy the census's causal ordering; being a protein decoder does.

**Separation is complete on nine arms.** All-head Spearman ρ, range over the four sender × case conditions: gpt2 +0.657/+0.706, gpt2-medium +0.552/+0.643, gpt2-large +0.428/+0.507, dialogpt-small +0.518/+0.640 against zymctrl +0.216/+0.271, progen2-medium +0.041/+0.207, progen2-small +0.128/+0.194, progen2-base +0.050/+0.176, protgpt2 −0.155/−0.006. The text minimum exceeds every protein maximum in all **36** arm × condition cells. Catalogued as **L22** and promoted in §1 to the strongest directly measured part-2 result the programme has.

**The text-side scale trend runs against the finding rather than producing it.** ρ falls monotonically 0.657 → 0.575 → 0.428 across the GPT-2 ladder, so the matched pair uses the *lowest*-correlation text arm — the conservative direction. And ProGen2-small (151M, 12 layers) reads +0.128/+0.194 against gpt2 (124M, 12 layers) at +0.657/+0.706: matched depth and scale, fivefold gap.

**Three corrections to EXP-R2-071, all verified against the artefacts.**

1. *The top-32 claim does not generalise.* "Protein arms' top heads are ranked better than text's" holds only for ProtGPT2 (+0.792/+0.824). ProGen2-base and ProGen2-medium overlap the text range, ProGen2-small sits below every text arm, ZymCTRL is near zero. What survives, and is the load-bearing half, is the *remaining-head* contrast: text +0.268 to +0.477 against protein −0.223 to +0.259.
2. *Two of the four published remaining-head figures were not reproducible.* gpt2's recorded +0.128 re-derives as **+0.342**, gpt2-large's +0.275 as +0.350. The cause is Appendix B rule 12 at its sharpest: the Gini, share-of-grid, reliability and top-k/rest statistics had **no implementation in the repository** and were computed by unversioned throwaway code, so the split that produced them cannot be recovered. The error is conservative — the true contrast is larger.
3. *The cross-arm |effect| comparison reverses.* Per-head effects are normalised by each arm's own clean-minus-corrupt denominator, and those span **32x** (0.76 to 24.7 logits). ZymCTRL's headline 0.3049, quoted as 15.7x gpt2-large's, is **0.233 logits — the smallest of the nine**. Its size came from a denominator a thirtieth of the text control's, itself a consequence of clearing the eligibility floor on 24 of 64 cases. The within-arm statement survives; the cross-arm one is withdrawn. New Appendix B rule 27.

**A fourth correction, to the reliability check.** The published 0.916–0.991 reproduces from the **case**-level SEM, on a module that states in two places that the probe record is the sampling unit. On the probe-clustered SEM it is 0.834–0.976 on exact cases and as low as **0.189** (ZymCTRL) on approximate ones. The inference still holds where it matters — gpt2-large and ProtGPT2 have near-equal probe-clustered reliability (0.779, 0.783) at ρ +0.440 and −0.155, so differential attenuation does not explain the matched-pair gap — but reliability is a variance statistic dominated by the largest heads and was never capable of establishing that the median head is resolved.

**The concentration statistics are suspended, not withdrawn.** They sum |effect| over a grid whose majority is individually indistinguishable from zero, and the unresolved mass is not matched across arms with unequal eligible-case counts. Both the ZymCTRL exception and the "collapses on approximate probes" qualification are decided inside that bulk.

**The seed-2 runs are weaker than their name.** ρ moves ~0.03 against a ~0.6 gap, but the repeat-cohort digests are **byte-identical across the two seeds**: `--seed` governs case sampling and the bootstrap while the corpus draw is governed by a separate `--corpus-draw-seed`. These are case-resampling checks, not cohort-draw checks, and given Appendix B rule 22 a cohort-draw check on this statistic is still owed.

**What this does not show.** Protein decoders do not have induction heads after all. The census under-recalls causally important heads more on protein than on text, so an unknown part of §4's head-count gap belongs to the instrument. Every text arm on this panel is GPT-2 architecture — `induction_path_patching` refuses `qwen2` and `llama` because `SUPPORTED_ARCHITECTURES` has no module layout for them, a recorded instrument limit and a candidate extension.

## 2026-07-31 — EXP-R2-073: repository-wide audit; the far-band propagation result is withdrawn on a unit confound

*Scope.* Five regions audited in parallel by Opus sub-agents — path patching, circuits, the statistical machinery, the H200 orchestration and the TG series — under the five Development Principles, with every acted-on finding verified independently before any change. Both reported findings that touch a published number were re-derived by hand from the shipped artefacts before being recorded.

### The finding that matters most: the far-band band is declared in tokens

`DISTANCE_BANDS` is a module constant in **token** units with no per-arm resolution and no CLI knob, while `04_circuit_primitives.py` records `tokenisation.symbols_per_token` in the same artefact and never consumes it. Measured on the shipped five-window artefacts: **4.403** symbols/token (gpt2-large, gpt2-xl), **2.816** (ProtGPT2), **0.996** (every ProGen2 arm). The "33–64 token" far band is therefore 33–64 residues on ProGen2, ~93–180 residues on ProtGPT2 and ~145–282 characters on GPT-2; the "single-token corruption" is one residue, ~2.8 residues, or ~4.4 characters.

**The decisive check needs no cross-modality unit.** ProtGPT2 and ProGen2 are both protein decoders over residues; only the tokeniser differs. On raw bands, no interpolation, mean over five windows at threshold 0.25: at 33–64 **tokens** ProtGPT2 reads 0.157 against ProGen2-base 0.305 and ProGen2-medium 0.288 — the published ordering. At 25–45 **residues** (its own 9–16 token band) ProtGPT2 reads **0.381**, and at 48–90 residues — a *longer* content distance — **0.281**, at or above ProGen2's 33–64-residue value. Same reversal at threshold 0.50. Re-indexing all six arms by content symbols and interpolating at 24 symbols, inside every arm's measured range, gives gpt2-large 0.516, gpt2-xl 0.515, ProtGPT2 0.443, ProGen2-medium 0.407, ProGen2-base 0.383, ProGen2-small 0.236 — the ordering reverses with the text controls highest.

**Neither alignment isolates the model.** Token-matching matches positional distance and mismatches perturbation size; symbol-matching matches content distance and mismatches perturbation size the other way. There is no post-hoc re-indexing of these artefacts that recovers a model-attributable ordering.

EXP-R2-070's arithmetic is correct and reproduces exactly — I re-derived 31/33/36/39 of 40 and both 40/40 cells — but those are statements about a token band. §9.2's lineage claim is withdrawn along with the modality claim, **D1.a reverts from answered to open**, and EXP-R2-070 (iv)'s between-window spread asymmetry is retained as a within-arm quantity whose levels need re-derivation at matched content distance. Catalogued as **L23**; new Appendix B rule 26.

### A related qualification the same lens produced

The repeat criteria behind §4's "32.3% of text documents against 0.402% of protein entries" *are* declared in content symbols, which is what keeps them free of this confound — but they are not the same criterion. Text requires a 40-character unit with 15 distinct symbols scored by identity; protein a 16-residue unit with 8 distinct symbols scored by BLOSUM62 non-adverse substitution. In symbols the text criterion is 2.5x stricter; converted through each arm's symbols per token it is ~9 tokens against 16, so in tokens the protein criterion is stricter. Both are domain-motivated and the qualitative conclusion is untouched — no re-parameterisation closes two orders of magnitude — but the multiplier is not a unit-free fact.

### Campaign launched

**EXP-R2-073 cross-lab far band**, ten jobs (qwen2.5-0.5b and llama-3.2-3b × five disjoint windows) at EXP-R2-070's protocol, three GPU lanes, driver `logs/drivers/farband_crosslab.sh`. Both of EXP-R2-070's text controls are GPT-2, so the between-window spread asymmetry of (iv) rests on one lineage — this is the check that killed the QK/OV finding. Validated first with a 32-case smoke test in-pod on both arms before committing the queue. Honest limit, found from the smoke artefact: qwen2.5-0.5b sits at 4.723 symbols/token against GPT-2's 4.403, only 7% apart, so this run is a **lineage** control and not a test of the symbols-per-token mechanism. That mechanism is tested within the protein modality by the ProtGPT2-vs-ProGen2 contrast above, and directly by the symbol-matched successor.

### Operational

The pod's four GPUs were at 0%, 100%, 0%, 0% on waking — three idle since 06:39 when the requeue driver drained. GPU 1 holds an orphaned gpt2-xl exhaustive census whose controller reported exit 90 on a `tls: bad record MAC` transport drop at 06:11 while the pod-side worker kept computing; it is at 8 h and still running, and its artefact must be verified against its own manifest rather than trusted, because no controller will report it complete. gpt2-medium's artefact from the same class of drop **was** promoted with a manifest at 02:56 and is the reason its result is usable.

## 2026-07-31 — EXP-R2-074: the far-band successor at matched content distance — a decay rate, not a level

*Configuration.* Six arms, five disjoint windows of one seeded permutation on every arm (30 of 30 artefacts), 256 cases per band, float32, both the distance band **and** the corruption span declared in content symbols and resolved to per-arm token spans by containment. 30 jobs, four GPU lanes, 11:44 to ~15:00 (+08:00). The instrument was validated in-pod at 32 cases before the queue was committed. Resolved geometry, recorded in every artefact: gpt2-large / gpt2-xl 2–3 / 4–7 / 8–14 tokens at 4.56 characters per token, ProtGPT2 4–5 / 7–11 / 12–22 at 2.81 residues per token, every ProGen2 arm 10–16 / 18–32 / 34–64 at 0.996. Span 5 characters and 3 residues — one token on the coarsest arm of each modality, which is the finest matched perturbation the panel can reach. Artefacts: `results/transfer_20260731/symbol/`.

**The level reverses.** At the shortest content band the text arms are highest — gpt2-large 0.671, gpt2-xl 0.704 against ProtGPT2 0.545, ProGen2-small 0.525, ProGen2-medium 0.656, ProGen2-base 0.664. The token-band tables were reading the tokeniser.

**The decay separates completely.** Ratio of the 33–64-symbol band to the 9–16-symbol band: text 0.489 / 0.490 against protein 0.596 / 0.706 / 0.720 / 0.732 at threshold 0.25 (gap +0.105), and text 0.313 / 0.329 against protein 0.500 / 0.608 / 0.615 / 0.624 at threshold 0.50 (gap +0.172). **Protein decays more slowly in 40 of 40 paired arm × control × window comparisons at both thresholds.** It is a within-arm ratio, so the level the perturbation mismatch moves largely cancels, and **the matched pair separates on this estimand for the first time** — gpt2-large 0.489 against ProtGPT2 0.706, with ProtGPT2 inside the protein range rather than being the arm that fails, which it was under every previous version.

**A result I did not expect, and it is a check on Appendix B rule 22.** The between-window spread of the decay ratio does *not* separate by modality: text 0.039 / 0.055 against protein 0.047–0.104 at threshold 0.25, where ProGen2-medium's 0.047 sits below gpt2-xl's 0.055, and the ranges overlap at 0.50 as well. The protein-specific cohort sensitivity that appears in four other statistics is absent here. Rule 22 says cohort sensitivity belongs to the statistic rather than the modality; this is the first case where it works in the direction that helps.

**Residual confounds, recorded rather than engineered around.** A residue is not a character, so the span is matched within modality and not across it. Integer token quantisation makes the token-distance ratio 4.3x for text and 3.7x for ProGen2, biasing toward faster text decay; correcting under a power law moves gpt2-large from 0.489 to ≈0.53, still below the lowest protein arm. The 40 comparisons share windows and arms. One matched protein arm remains the structural limit.

**Operational.** Five gpt2-xl jobs lost the transport mid-run and their controllers returned exit 90 rather than reporting completion; in every case the pod-side worker finished and promoted an artefact, and each was admitted only after both its files were checksum-verified against the manifest the worker wrote beside them. That is the L20 machinery converting a dropped connection into a manual verification step instead of a silent green campaign, for the fifth through ninth time this programme. The pattern is systematic in the longest-running jobs and is worth a caller-side keepalive rather than a per-incident check.

**Also this session.** The cross-lab far-band control (EXP-R2-073) was stopped after three of its five windows: llama-3.2-3b at batch 16 was taking ~80 minutes a job and the successor campaign was the better use of the cards. That is a deliberate truncation, not a completed design — the batch size was chosen over-cautiously at 16 against 143 GiB of memory, and a re-run at batch 64 would cost a fraction of what was spent.

**Addendum (2026-07-31): EXP-R2-075 launched, and an operational fault of my own in the lane design.**

*The campaign.* The two cross-lab arms in the content-symbol geometry, five windows each, batch 64 — the check EXP-R2-074's decay-rate separation needs before it can be read as more than a GPT-2 property, since both of its text controls are GPT-2. This is the configuration that killed the QK/OV finding. Results root `farband_symbol_20260731/sym*/`.

*The fault.* Two llama controllers hung on a dead transport for 40–50 minutes each while their pod-side workers had already finished and promoted manifest-verified artefacts. The GPUs sat at 4 MiB and 0% the whole time, and the lanes could not claim their next job because the work-stealing loop blocks on the controller returning. **The queue design has no timeout on a hung controller**, so a dropped channel converts into idle GPU rather than into a failure — the opposite of the L20 repair's intent, which was to make a dropped channel *visible*. It is visible in the log; it is not visible to the scheduler. I killed both controllers; the lanes recorded exit 143 and immediately claimed the remaining jobs, and both orphaned artefacts were checksum-verified against the manifests their workers wrote before being admitted.

*What this costs and what it needs.* Roughly 1.5 GPU-hours of idle time on this run alone, on top of the five gpt2-xl drops earlier in the session. The pattern is systematic in the longest-running jobs and is not a per-incident problem: a lane should bound how long it waits on a controller whose GPU has gone idle, and the access layer should carry a keepalive. Recorded here rather than fixed in the same breath, because the fix belongs in the orchestration layer and this session has already changed it once.

**Addendum (2026-07-31): EXP-R2-075/076 — the cross-lab control passes, the off-distribution control narrows the result to the large-effect tail.**

*EXP-R2-075, cross-lab.* qwen2.5-0.5b and llama-3.2-3b at the identical content-symbol geometry, five windows, batch 64. Decay ratio 0.527 and 0.526 against gpt2-large 0.489 and gpt2-xl 0.490 — above the GPT-2 arms, below every protein arm. The pattern is not a GPT-2 idiosyncrasy. Two of ten jobs lost the transport; both artefacts were checksum-verified against their manifests before admission.

*EXP-R2-076, the text ladder and the control.* gpt2, gpt2-medium and dialogpt-small added, five windows each, all fifteen jobs exit 0. **dialogpt-small does here what it did not do on the census.** On the census it read +0.518 to +0.640 — squarely text — which is what ruled out distribution mismatch as the explanation. On the decay ratio it reads **0.611**, above ProGen2-small's 0.596, and the text and protein ranges **overlap by 0.015 at threshold 0.25**. At threshold 0.50 they still separate cleanly, worst text 0.397 below lowest protein 0.500, and protein is slower in 136 of 140 paired comparisons against 133 of 140 at 0.25.

*So the separation lives in the large-effect tail*, which is the third time §5.1 has found that shape on this estimand — first for the bfloat16 far-band sweep, then for EXP-R2-070's five-window comparison, now for the content-matched decay. It is beginning to look like a property of the measurement rather than an incident.

*The confound dialogpt-small raises, tested rather than asserted.* If the decay ratio tracks distributional fit rather than modality, it should correlate with each arm's context information. Across all ten arms Spearman is −0.62 (p = 0.054), but that is the modality split restated — protein arms have both low context information and slow decay. **Within the text arms it is −0.14 (p = 0.76), and +0.37 excluding dialogpt-small.** Context information does not predict the decay ratio among comparably-fitted models, so the confound is not established; dialogpt-small is the only badly-fitted arm and it is the one that overlaps, so it is not refuted either. Settling it needs a second off-distribution arm and the panel has none — a structural limit of the same kind as §2's.

*Also visible with seven text arms:* the text decay ratio spans 0.423 (gpt2) to 0.611 (dialogpt-small), much wider than the four-arm view suggested, and it does not order by scale — gpt2 124M reads 0.423 while gpt2-large 774M reads 0.489 and llama 3.21B reads 0.526, with gpt2-medium 355M at 0.509 out of order. The protein side does order by scale (ProGen2-small 0.596 to ProGen2-medium 0.732, with two 765M arms differing only in corpus at 0.720 and 0.732), so the extrapolation risk runs toward *small* protein arms, and the smallest is already measured.

**Wake point.** The H200 is idle and deliberately so: the next item is D2.c, the exhaustive causal census on copy suppression, which is the replication that would generalise L22 from one mechanism to two and is the gate on §8's opening 0. It needs a design pass rather than a driver — the previous PAA work returned NO-GO — and starting it unplanned at the end of a long session is how the specification defects in Appendix B rule 2 got written. The orchestration fault recorded above (a lane with no timeout on a hung controller) should be fixed before the next long campaign, since it cost ~1.5 GPU-hours of idle time on a single run.

---

## 2026-08-01 — EXP-R2-077/078/079: the two owed checks on L22, and the repairs that had to land first

Picked up the wake point above. The next item was recorded as D2.c; the design
pass it asked for is below and its verdict is **not schedulable as specified**.
What went onto the idle cards instead is the pair of checks EXP-R2-072 records as
owed against L22, the programme's strongest part-2 result — one of which needed a
code extension that also lifted a catalogued instrument limit.

### EXP-R2-077 — the cohort-draw check on the census-to-causal correlation

*Why this and not D2.c.* EXP-R2-072 (iv) states plainly that the seed-2 runs are
**not** cohort-draw checks: `--seed` governs case sampling and the bootstrap while
the corpus draw is governed by `--cohort-draw-seed`, and the repeat-cohort digests
were byte-identical across the two seeds. Appendix B rule 22 forbids inheriting a
draw-robustness result from another statistic. The check costs no new code.

*Design.* Four arms at the byte-identical EXP-R2-071/072 configuration
(`--exhaustive-senders --repeat-cohort-size 48`) under `--cohort-draw-seed 20260801`:
**gpt2-xl**, the text minimum at +0.371 whose fall would close the gap from above;
**zymctrl**, the protein maximum at +0.271 whose rise would close it from below;
and **gpt2-large / protgpt2**, the matched pair. One alternate draw rather than
two — at K = 2 a shifted arm cannot be told from a noisy one, so this is posed as
an ordering question, not a variance estimate. Launched 00:08 +08:00, four lanes,
work-stealing, per-job `timeout` as a mitigation for the hung-controller fault the
wake point flags (a mitigation, not the fix: it frees the lane and does not reap a
pod-side process).

*Stated before the run: part of this statistic cannot move.* Swiss-Prot yields
exactly **48** exact-repeat matching records of 203,063 eligible against a request
of 48, so `_select_matching` returns all of them under any seed and the protein
exact-repeat cohort is a **census**. Its identity cannot change. The check
therefore bites on the approximate conditions and on the text arms, where the
draws do differ — verified on B before launch: 48 of 137 matching (text exact) and
48 of 1022 (text approximate) at the default seed against 48 of 133 and 48 of 1021
at 20260801, all four digests distinct.

*A protein-specific fragility, found by the entry point refusing to run short.*
ProtGPT2 died at 40 seconds on draw 20260801: **58 of 64** approximate-repeat path
cases, 32 keys outside the unigram support. A CPU scan over four draws before any
further GPU time:

| draw | protgpt2 exact | protgpt2 approximate | zymctrl exact | zymctrl approximate |
|---|---|---|---|---|
| 20260728 (reference) | 64 | 64 | 64 | 64 |
| 20260801 | 64 | **58 — FAIL** | 64 | 64 |
| 20260802 | 64 | 64 | 64 | 64 |
| 20260803 | 64 | **57 — FAIL** | 64 | 64 |

The arm ran at 20260802. **The selection is on case-count parity — the same
criterion that set `--repeat-cohort-size 48` in EXP-R2-071 — and is evaluated
before any effect is measured, so it cannot select on the outcome.** The rejected
draws are recorded rather than hidden, and the failure rate is itself a result:
**ProtGPT2's 64-case parity in EXP-R2-071/072 was draw-contingent on roughly half
the draws tried, and nothing said so.** That is the protein-side cohort
sensitivity of §5.05(b) reaching the *case set* rather than the estimate, and it is
specific to the subword protein arm — ZymCTRL, residue-level, reaches 64 on every
draw tested. ZymCTRL kept draw 20260801 rather than being killed and restarted at
20260802: the check is per-arm by construction, each arm against its own reference
draw, so the two protein arms need not share one alternate permutation.

*Status at the time of writing: running, four GPUs at 100%. No result.*

### EXP-R2-078 — a repository audit under the Development Principles, and four repairs

Two Opus sub-agents audited `prediction_addressed.py`, `path_patching.py`,
`statistics.py`, `scoring.py` and `io.py`. Every finding acted on below was
re-derived independently from the code or the shipped artefacts before any change.
Two findings were rejected as suggestions rather than defects and are not acted on
here (a seed collision in the random-direction control that is not a statistical
error, and a dead branch).

**(1) A published number moves: the reliability figures paired the wrong centre
with the wrong standard error.** `head_effect_reliability` read the observed
variance from the case-weighted `mean` under *both* sampling units while varying
only the standard error, so the probe-clustered figure divided a probe-unit error
variance by a case-unit observed variance. `sender_recoveries` states the pairing
rule at the point it builds the record — `mean` weights probes by case count,
`sem_probe_clustered` describes `mean_probe_clustered`, which weights them equally
— and the function twelve lines away broke it. Recomputed over every shipped
`d2bprime*` artefact:

| arm / condition | as published | correctly paired |
|---|---:|---:|
| ZymCTRL, approximate cases, magnitude | **0.008** | **0.170** |
| ZymCTRL, approximate cases, signed | 0.189 | **0.443** |
| gpt2-xl, approximate cases, signed | 0.659 | 0.706 |
| gpt2-large, approximate cases, signed | 0.779 | 0.807 |
| every exact-case condition, ten arms | — | moves by < 0.006 |

The audit document's "**0.008** — a grid that cannot be ranked at all" is
withdrawn as stated. 0.170 is still very low, and the exact-case range is
unaffected, so **nothing resting on the exact-case reliability moves**, including
the inference that differential attenuation does not explain the matched-pair gap.
The mismatch was invisible because the test fixture omitted `mean_probe_clustered`
entirely and pinned the two mismatched values as expectations; they are now pinned
as *retracted* values, reconstructed in the test from the artefact, so a
regression restores them loudly.

**(2) The instrument limit on L22 is lifted, and validated on a real checkpoint.**
`path_patching` refused `qwen2` and `llama`, which is why every text arm carrying
L22 is GPT-2 architecture — the exact configuration that retracted the QK/OV
finding. The extension is mechanical, and the reason it is mechanical is worth
recording: the three things a rotary decoder does differently are all *downstream*
of the patch. RoPE is applied to queries and keys inside attention, and the sender
patch replaces the input to the output projection, which is the attention result.
Grouped-query attention changes which key/value head a query head reads and not
the layout of that projection's input, which is the concatenation of per-*query*-head
outputs in every case. RMSNorm is a drop-in for the final-norm pre-hook.

The trunk and final norm now come from `circuits.inner_decoder` and
`circuits.final_norm` rather than from six hard-coded `model.transformer.ln_f`
references — Appendix B rule 12, one declaration imported. Two guards asserted
`n_head * head_dim == d_model`, which is **not** the invariant the patch needs:
the slice indexes the output projection's *input*, and the two coincide only while
no panel arm declares its own `head_dim`.

*Validated end to end on Qwen2.5-0.5B, 336 exhaustive senders, 727 s on one L20.*
`structural invariants passed`: head-write linearity at **2.19e-06** median
relative error against a 0.02 tolerance — which is the GQA correctness proof,
since it checks that the per-head column slice of the projection's input matches
the per-query-head slice of `W_O` that `head_ov_weights` takes — readout-matches-model
at **1.21e-06**, position locality at 4.77e-07. The panel contract was regenerated;
`induction_path_patching` now declares all 12 arms.

**(3) `content_low` did not survive the pool round trip.** `save_pool`,
`load_pool` and `_subset` in `14_paa_census.py` each dropped it, so a reloaded pool
silently took the dataclass default of 0 and `antecedent_sets` would search from
position 1 instead of the first content token — adding ProtGPT2's newline wrapping
or ZymCTRL's `<sep>` to the key set the causal knockout removes, which is exactly
the failure that function documents itself as preventing. Latent, because the only
pool ever reloaded is the text control's, whose bound is 0 either way; live the
first time a protein pool is scored, which is D2.c. The round trip is now derived
from `dataclasses.fields` and a pool missing a field is refused rather than
defaulted.

**(4) A percentile nobody could resolve.** `cluster_bootstrap` accepted any
`alpha` against any replicate count, and the census's Bonferroni column asks for
`0.05 / (2 * n_heads)`: at the 24 heads it screens and 1000 replicates that is
sorted index **1.04**, so the "lower bound" is the second-smallest draw and moves
by 0.0028 logits between seeds — the size of the smallest effect published beside
it. At 720 heads it is index 0.035. One rule now ties alpha to replicates,
**replacing** the flat `replicates >= 100`, which is the same rule stated for one
alpha and stated wrongly: at the default alpha a hundred replicates put 2.5 draws
in the tail. The caller either buys the replicates or withholds the column with
its reason attached.

*Validation.* 441 tests plus 34 subtests pass (from 434), ruff clean, both
generated contracts verified.

### EXP-R2-078 (continued) — the D2.c design pass, and why it is not schedulable

**Blocker 1: the matched pair cannot enter the estimand.** Verified against the
tokenisers rather than inferred. ProtGPT2 renders at **0.349 tokens per residue**,
so a 512-token pool row needs ~1468 residues against a cohort band capped at 800 —
no row qualifies and `tokenised_rows` refuses the arm. ZymCTRL is worse than a band
problem: `content_bounds` requires exactly one `<end>` inside the window and the
row must also reach it, and those two conditions intersect at the single residue
length `width - 10` — measured, exactly 502 at width 512. So D2.c as specified runs
GPT-2 lineage against ProGen2 lineage **without the matched pair**, which is the
only modality-identifying comparison the panel has. A cross-lineage-only result on
the second mechanism cannot discharge a gate about the first.

**Blocker 2: the selector and the causal statistic are measured on different key
sets, alphabet-dependently.** `paa_attention_scores` scores attention onto the
*nearest* earlier occurrence; `knockout_effects` removes *every* earlier
occurrence. Counted from the shipped pools: median **3** occurrences before the
query on gpt2-large against **13–17** on ProGen2-medium. A rank correlation
between the two is therefore attenuated harder on protein by construction, in the
direction of the hypothesis. `corruption_effects` already fixed this exact error
for the matching gate and its docstring names it — "a conclusion the estimator
manufactured out of alphabet size".

**Blocker 3 is cleared, and it was the one that mattered most.** The plan had
inherited from L5 that `paa_specific` does not rank causal effect even on
gpt2-large, which would make D2.c a gate no text control can pass. **That figure
is on the signed effect; D2.c's statistic is the magnitude.** Recomputed on both
retained trees, no re-run:

| tree | heads | signed rho | magnitude rho |
|---|---:|---:|---:|
| `paa_gate` | 24 | +0.086 (p = 0.69) | **+0.608 (p = 0.0016)** |
| `paa_gate` | 16 screened | +0.265 (p = 0.32) | **+0.621 (p = 0.010)** |
| `paa_gate_extended` | 32 | −0.162 (p = 0.38) | **+0.533 (p = 0.0017)** |
| `paa_gate_extended` | 24 screened | −0.305 (p = 0.15) | **+0.648 (p = 0.0006)** |

Near zero on the signed scale in all four cells; +0.53 to +0.65 and significant in
all four on the magnitude scale — comparable to gpt2-large's induction +0.428 to
+0.507. **L5 stands exactly as written**, since it claims the signed result; what
changes is the inference the plan drew from it. Two limits belong in D2.c's
pre-registration: these heads were selected by `paa_specific` itself, so the
correlation is range-restricted and the all-grid value may move either way; and a
magnitude ranking conflates suppressive with promoting heads, so it is evidence
that the selector orders causal importance, not that it finds copy suppression.

**Cost, re-derived.** `knockout_effects` needs `ceil(N/B) * (2 + H)` forwards:
36,100 at gpt2-large's 720 heads and 800 instances. Anchored on EXP-R2-059's
measured 0.73 s per (16, 512) forward, **3–5 H200-hours for gpt2-large alone**,
23–39 for a ten-arm panel. Three avoidable costs sit inside it — the full
`[batch, token, vocab]` logit tensor materialised where one position is read, a
dense `(B, n_head, W, W)` mask per head per batch, and `use_cache` left on — and
fixing them brings the panel to **12–25 H200-hours**. The table's "~13 GPU-h" was
costed by analogy.

### EXP-R2-079 — the cross-lab control on L22, chained and pre-registered

qwen2.5-0.5b and llama-3.2-3b on the exhaustive causal census at the byte-identical
EXP-R2-071/072 configuration, so they combine with the ten arms already measured.
Chained behind EXP-R2-077 on the terminal lines of its queue logs rather than on a
process pattern — a chain that greps for its own launch command matches its own
wrapper, which idled the GPUs for eight minutes in EXP-R2-071.

**Pre-registered, because the scale trend makes one arm genuinely ambiguous.** The
text side falls about −0.26 per decade (0.657 / 0.575 / 0.428 / 0.371 at 124M /
355M / 774M / 1.56B). Extrapolating: qwen2.5-0.5b at 494M predicts **≈+0.60**,
comfortably inside the text range; llama-3.2-3b at 3.21B predicts **≈+0.29**,
which is *within reach of the protein maximum of +0.271*. So a llama value near
0.29 is what the scale trend predicts and is **not** evidence against L22; a value
near zero or negative, like ProtGPT2's, would be. Recording this before the run,
because reading a narrow margin either way afterwards is how the specification
defects in Appendix B rule 2 got written.

**Wake point.** Read `logs/draw_queue.log`, `logs/draw_protgpt2_queue.log` and
`logs/crosslab_census_queue.log` in that order; artefacts at
`results/.../d2bprime_draw_20260801/<arm>/` and `.../d2bprime_crosslab_20260801/<arm>/`,
primary statistic the all-head Spearman between prefix-matching score and causal
effect magnitude, read against the text range +0.371 to +0.706 and the protein
maximum +0.271. Then either clear D2.c's two blockers or scope its claim to the
lineages it can reach. The hung-controller fault is mitigated per-lane and still
not fixed in the orchestration layer.

**Addendum (2026-08-01): D2.c blocker 2 is closed in code, and the mismatch is bigger than the argument for it suggested.**

`paa_attention_scores` now emits `paa_specific_matched` beside `paa_specific`:
attention summed over the key set `antecedent_sets` returns — the set
`knockout_effects` actually removes — against a decoy baseline scaled to the same
number of keys, so the correction stays a positional baseline of matched size
rather than a per-key one subtracted from a sum. The sum rather than the mean
because the intervention's size scales with the total mass it removes, which is
what the causal effect responds to. Both scores are returned rather than one
replacing the other: `paa_specific` is what EXP-R2-059 published and L5/L6 quote,
and redefining it in place would make those numbers unreproducible. Two tests pin
the definitions against a real forward pass, one of them asserting that the two
collapse onto each other when the predicted token occurs exactly once before the
query — which is the case that catches a numerator changed to a sum while the
baseline stayed per-key.

*Validated on B, gpt2-large, 8 sequences, GPU 7 — a stage validation, **not** a
result and not quotable as one.* Mean key set **2.93**, consistent with the median
of 3 counted from the shipped pools. The two scores rank the 720 heads at
Spearman **+0.567** (partial, non-sink-corrected +0.553). Part of that at 8
sequences is estimation noise. What it establishes is that the two definitions are
not interchangeable **on the arm where the mismatch is smallest**, so on a protein
arm at 13–17 keys the divergence can only be larger. The choice of key set is not
a technicality for D2.c; it is the estimand.

**Addendum (2026-08-01): the last two audit findings — one closed, one accepted with its reason.**

*The instance cascade did not close, and 16.2% of a published denominator was
unaccounted.* `build_instance_pool` charged a position blocked by both the
induction rule and the distance rule to induction alone, and charged a position
where no candidate ever reached the antecedent test — nothing above
`min_confidence`, or no earlier occurrence of any top-k token — to nothing at all.
On the shipped gpt2-large pool: 96,000 scored against 71,533 + 8,506 + 425
accounted, leaving **15,536 positions in no category**. The cascade now has five
exits that sum to `positions_scored`, checked in the function rather than
documented, and it raises if they ever stop summing. The published
`induction_target_discard_rate` of 10.6% is *arithmetically correct* — its
denominator is induction-blocked plus instance-yielding positions — but that is
not the denominator "candidates" implies, so the artefact now names it and emits
the rate over all scored positions beside it.

*The two decoy corrections are not the same one, and cannot be made the same.*
`decoy_corrected_prefix_matching` claimed "the same decoy subtraction PAA uses"
and carried an inline comment asserting the two draws had to select from one key
population. They do not: `build_instance_pool` additionally bans a decoy key
holding one of the model's top-`ban` predictions at the query. **The obvious fix
is alphabet-pathological.** Over a twenty-symbol residue alphabet a top-20 ban is
the alphabet, and it is already measured emptying the decoy pool for 93.0% of
eligible positions on ProGen2-medium — the measurement `--protein-ban-depths`
exists because of. Adding the ban to the induction census would delete its protein
side to make a text-side definition match. Recorded as an accepted limitation with
the direction of the difference stated — this correction is the more conservative
one, since a decoy here may hold a plausible prediction whose attention is then
subtracted as though it were positional baseline. L6's headline rests on the tail
count and the top-20 Jaccard and is unaffected. This is the Restraint Principle
deciding a case where the Repair Principle would have cost an arm.

*Two further findings were reviewed and not acted on*, as suggestions rather than
defects: a seed collision in the random-direction control that reuses vectors
across `(begin, layer)` pairs summing alike — not a statistical error, though not
the independence the design implies — and a dead branch in the decoy window where
`window >= 1` is always true.

**Addendum (2026-08-01): the EXP-R2-079 chain became a per-GPU dispatcher, and I walked into the trap this log already documents.**

*The scheduling fault.* EXP-R2-079 was first chained to start after EXP-R2-077
drained *every* lane. That is wrong here: gpt2-xl carries 1200 heads on lane 0
against 720 on lanes 2 and 3, and the ProtGPT2 leg is a separate single job on
GPU 1, so the lanes free hours apart and an all-or-nothing chain would have left
up to three cards idle for those hours. Replaced with a dispatcher that claims
each GPU the moment its own predecessor announces it is done, read from the local
queue logs. Same class of waste as the hung-controller fault recorded on
2026-07-31, reached by scheduling rather than by transport.

*My own fault, recorded rather than smoothed over (Appendix B rule 20).* Killing
the first chain with `pkill -f logs/drivers/crosslab_census.sh` matched **the
shell running the pkill**, whose command line contained that path, and killed it —
so the heredoc writing the replacement never ran and the command returned 255.
This is the identical failure mode EXP-R2-071 recorded when a chain's `pgrep -f`
matched its own launch wrapper, met from the other direction. The rule earned
there — *a pattern that a launching shell's command line can contain will match
that shell* — is now a comment at the point in the dispatcher where the
predecessor markers are declared, and the dispatcher keys on log lines rather than
on processes throughout. Cost: nothing but the retry; the four running jobs are
separate `setsid` processes and were untouched.

**Addendum (2026-08-01, 02:02–02:20): a transport drop mid-campaign, the first EXP-R2-077 arm, and a scheduling lesson the drop taught.**

*The drop.* At **02:02:11** all four EXP-R2-077 controllers exited **90**
simultaneously — the sentinel-absent code — with the access layer reporting 255,
which L20 records as not authoritative. This is the tenth through thirteenth time
the sentinel machinery has refused to convert a dropped channel into a green
campaign, and again it was right to: **three of the four pod-side workers survived
the drop and are still computing** (gpt2-xl on GPU 0, gpt2-large on 2, zymctrl on
3, each ~1h52m in at the time of the check). Cluster health re-probed `Health=ok`
four minutes later, so the drop was the channel, not the cluster.

*ProtGPT2's leg had already finished.* Its worker promoted `protgpt2.json` and
`panel_summary.json` at 02:02, in the same minute the transport died, so the
controller could not report success and the artefact had to be admitted the way
this programme now admits every orphan: against the SHA-256 manifest the worker
wrote beside it. Both digests match — `protgpt2.json`
`e66f85e44a70f434f15d9f5c84ffcda164cdf0430bc84e8ac32723f06c86a398`,
`panel_summary.json` `7a163e028217fd593e0166b7f93f5fb0e49d8ed3055c1cc69207ff0347c31891`
— and the provenance record carries the exact command including
`--cohort-draw-seed 20260802`. Verified again after the pull. Local copy:
`results/transfer_20260801/d2bprime_draw/protgpt2_draw20260802/`.

**The first result. ProtGPT2's census-to-causal correlation under a different
corpus draw, and it moves AWAY from the text range.**

| condition | draw 20260728 | draw 20260802 | delta |
|---|---:|---:|---:|
| senders_exact / cases_exact | −0.0763 | **−0.1513** | −0.0750 |
| senders_exact / cases_approximate | −0.1554 | **−0.2259** | −0.0705 |
| senders_approximate / cases_exact | −0.0058 | **−0.0547** | −0.0489 |
| senders_approximate / cases_approximate | −0.0537 | **−0.0610** | −0.0072 |

Range −0.1554…−0.0058 → **−0.2259…−0.0547**. All four conditions move down, by
0.007 to 0.075 against a modality gap of about 0.5, and the arm stays far below
both the text minimum (+0.371) and the protein maximum (+0.271). **On this arm the
separation widens under a second draw rather than narrowing.** One arm only; the
matched-pair text control has not landed and nothing may be concluded about the
separation until it does.

*The pre-stated census property is confirmed in the artefact.* The protein
**exact**-repeat cohort digest is byte-identical across the two draws —
`3ea68a87c593c532` both times — exactly as recorded before the run, because
Swiss-Prot yields exactly 48 matching records against a request of 48 and
`_select_matching` returns all of them under any seed. The analysis cohort and the
approximate-repeat cohort both changed. So the movement above is carried by the
approximate cohort and the unigram fit, and the exact-case conditions are
*structurally* insensitive to the draw on the protein side — which is why they move
least in absolute terms among the conditions that could move at all.

**The scheduling lesson, and my second operational fault of the session.** The
per-GPU dispatcher fired both cross-lab jobs within seconds of the drop, because
the queue logs it keyed on had just written `ALL LANES DRAINED` and `exit 90`.
Both jobs refused to schedule — the health check stated no `Health=` line and the
controller declined an undecidable precondition, which is the guard working — but
the queue was consumed and the dispatcher exited. **After a drop the local queue
logs say "drained" while three cards are busy: the pod is the only authority on
what is running.** llama was relaunched on the genuinely free GPU 1 at 02:06 and
is computing; the qwen leg is now watched by a driver that polls in-pod
`nvidia-smi` for a card under 1000 MiB rather than trusting a local marker, and
that retries rather than surrendering the card when a health check refuses.

*Still owed operationally:* the three surviving workers have no controller left to
pull their artefacts. Each must be verified against its own manifest and pulled by
hand when it finishes, exactly as ProtGPT2's was.

**Addendum (2026-08-01, 02:30): the campaign is left autonomous, and the wake point is restated against what actually survived.**

Three watchers are running on B, all `setsid`, none keyed on a process pattern:

| driver | what it does | log |
|---|---|---|
| `crosslab_llama.sh` | llama-3.2-3b on GPU 1 under its own controller, which pulls on success | `logs/crosslab_llama_g1.log` |
| `crosslab_qwen_watch.sh` | claims the next card the **pod** reports under 1000 MiB, retries a refused health check rather than surrendering the card | `logs/crosslab_qwen_watch.log` |
| `draw_recover.sh` | verifies and pulls the three orphaned EXP-R2-077 runs when their workers promote, refusing anything whose digest does not match | `logs/draw_recover.log` |

All four H200s are at 100%: gpt2-xl, gpt2-large and zymctrl on the surviving
EXP-R2-077 workers, llama-3.2-3b on the fresh controller.

**Wake point, replacing the one written at 01:40.**

1. `logs/draw_recover.log` first. It says which of gpt2-large, zymctrl and gpt2-xl
   were pulled and verified, and names any that were **refused** on a digest
   mismatch. A refusal is a result about the channel, not about the arm, and the
   arm is then re-runnable from its recorded command.
2. Read each arm's `causal_census_agreement.spearman_census_vs_causal_magnitude.rho`
   over all four conditions and put it beside the reference draw in
   `results/transfer_20260730/d2bprime*/`. **ProtGPT2 is already in:** it moves from
   −0.1554…−0.0058 to −0.2259…−0.0547, away from the text range. The comparison
   that matters is gpt2-large, its matched control: if the text arm moves down by
   a comparable amount the check is uninformative, and if it holds near +0.43 to
   +0.51 the separation survives a second draw on the only modality-identifying
   pair the panel has.
3. `logs/crosslab_llama_g1.log` and the qwen log. Read those two against the
   pre-registration recorded above — qwen ≈ +0.60 expected, llama ≈ +0.29 expected
   and within reach of the protein maximum, so **a narrow llama margin is the
   prediction, not a refutation**.
4. Only then D2.c, whose two blockers and re-derived cost are in the audit
   document's own D2.c section.
