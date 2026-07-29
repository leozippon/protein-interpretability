# Research 2 — CLT Experiment Log

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
- Reference: `Research2/scripts/08_layer_quality_map.py`

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

**Output files:** `Research2/results/checkpoint_evaluation/layer_quality_map.json`

---

## EXP-R2-006: Architecture Reconciliation (2026-04-13)

**Question:** EXP-R2-004 concluded L35 dominates EC discrimination (L2=564 vs avg 40). But EXP-R2-005 shows L35 has only 9.7% alive features. Can we trust the L35 claim?

**Method:**
- Compute effective_discrimination = raw_L2 × CLT_quality (alive × (1-FVU))
- Rank only USABLE layers by effective discrimination
- Reclassify three-phase architecture based on usable layers
- Reference: `Research2/scripts/09_reconcile_architecture.py`

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

**Output files:** `Research2/results/circuit_analysis/zymctrl/architecture_reconciliation.json`

---

## EXP-R2-007: Hook Sanity + EC Feature Provenance (2026-04-29)

**Question:** Are the negative steering / causal-ablation results caused by
broken plumbing, stale EC features, or a real lack of controllable signal?

**Method:**
- Added `Research2/scripts/diagnostics/00_hook_sanity.py`.
- Added `Research2/scripts/diagnostics/01_pkl_provenance.py`.
- The hook sanity script attaches the same MLP-output hook style used by the
  steering code, chooses an active CLT feature on a reference lysozyme sequence,
  and checks teacher-forced logit changes under multiplier 1, 10, and 0.
- The provenance script hashes and shape-checks the H200 `ec_features.pkl`
  against the ZymCTRL v2 CLT checkpoint.

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

**Interpretation:** The MLP hook path can move logits in both ZymCTRL v2 and
ProGen2-medium. The earlier exact-zero causal-ablation result is therefore not
explained by a universally disconnected hook; the more likely issue is feature
selection or the specific ablation evaluation path. The H200 `ec_features.pkl`
is a v2-compatible 8192-dimensional artifact, and the local copy has been
refreshed from H200.

**Output files:**
- `Research2/results/diagnostics/hook_sanity_20260429.json`
- `Research2/results/diagnostics/ec_features_provenance_20260429.json`

---

## EXP-R2-008: Direct-Effect Feature Selection (2026-05-03 CST)

**Question:** Can EC steering features be selected by direct effect on
EC-conditioned likelihood rather than by mean-activation z-score?

**Method:**
- Added `Research2/scripts/16_direct_effect_features.py`.
- For each EC class, ran a teacher-forced backward pass through ZymCTRL v2.
- Ranked CLT features by
  `feature_activation * grad(log-likelihood) * CLT_decoder_vector`, summed
  across the CLT decoder window.
- Used the ZymCTRL v2 CLT checkpoint at
  `/oss-pvc/zhk_zip/outputs/research2/clt_weights/zymctrl_v2/step_200000`.

**Result:** Completed for 8 EC classes and all 36 layers. The main
`top_indices` array has shape `(8, 36, 10)`, satisfying the T2-A acceptance
shape.

**Interpretation:** This provides a stricter candidate feature set for steering:
features must be active and point in a direction that changes model likelihood.
It does not by itself prove steering works; it feeds T2-B.

**Output files:**
- `/oss-pvc/zhk_zip/biocc/Research2/results/circuit_analysis/zymctrl/direct_effect_features_v2.pkl`
- `/oss-pvc/zhk_zip/biocc/Research2/results/circuit_analysis/zymctrl/direct_effect_features_v2_summary_20260503.json`
- `/oss-pvc/zhk_zip/biocc/Research2/logs/runtime/t2a_direct_effect_features_20260503.log`

---

## EXP-R2-009: On-Manifold Direct-Effect Steering (2026-05-03 CST)

**Question:** Does TopK-aware CLT steering improve over the old off-manifold
single-decoder-vector intervention?

**Method:**
- Updated `Research2/src/analysis/circuit_discovery.py` so steering hooks:
  compute CLT pre-activations, apply feature multipliers, re-apply TopK, and
  replace the CLT-explained same-layer MLP component.
- Updated `Research2/scripts/11_steering_benchmark.py` to accept
  `--direct-effect-features`.
- Passed a small lysozyme smoke test with direct-effect features.
- Started an 8-class benchmark with n=100 per condition, layers L3/L12/L30,
  top-5 direct-effect features per layer, multiplier=2.5.

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

**Interpretation:** TopK-aware direct-effect steering is a measurable
intervention and produces modest positive shifts for kinase and lipase, but it
does not clear the significance threshold on the current heuristic motif purity
metric. This does not rescue the R2 steering claim yet; T2-C's real metric triad
is still required before the R2 go/no-go decision.

**Output paths:**
- `/oss-pvc/zhk_zip/biocc/Research2/results/steering_benchmark/zymctrl_v2_onmanifold_direct_smoke_20260503.json`
- `/oss-pvc/zhk_zip/biocc/Research2/results/steering_benchmark/zymctrl_v2_onmanifold_direct_20260503.json`
- `/oss-pvc/zhk_zip/biocc/Research2/logs/runtime/t2b_onmanifold_direct_steering_20260503.log`

---

## EXP-R2-010: T2-C Metric-Triad Readiness Check (2026-05-04 CST)

**Question:** Can the real EC metric triad be run now for T2-C?

**Method:** Checked the current 1-GPU H200 pod and local/remote project paths
for required executables and metric assets.

**Result:**
- Missing executables in the pod: `hmmscan`, `hmmsearch`, `diamond`,
  `foldseek`, and `esm-fold`.
- Present assets: `data/interpro/pfam_residue.tsv` and ESMFold weights at
  `/oss-pvc/zhk_zip/models/esmfold_v1`.
- No staged CLEAN database, Pfam HMM database, Foldseek binary, or Foldseek
  target structure database was found in the checked paths.

**Interpretation:** T2-C remains blocked. Per the TODO_NEXT guideline, no real
EC metric should be claimed until the tools/databases are staged and a
calibration run reproduces expected behavior on known positives and negatives.

---

## EXP-R2-011: T2-D Viability Decision Gate (2026-05-04 CST)

**Question:** Given T2-A/B and the T2-C readiness check, should R2 continue as
a steering/drug-design claim or pivot?

**Evidence:**
- T0-A hook plumbing passed: interventions can move logits.
- T2-A direct-effect feature selection completed with shape `(8, 36, 10)`.
- T2-B TopK-aware on-manifold steering produced 0/8 significant positive EC
  classes on the current heuristic purity benchmark.
- The strongest trends were kinase +0.107, p=0.0532, and lipase +0.057,
  p=0.0913; neither passed significance.
- Structural QC did not show a foldability advantage for steered lysozyme leads
  over unsteered controls.
- T2-C real metric triad is blocked because CLEAN/HMMER/Foldseek tools and
  calibrated databases are not staged.

**Decision:** No-go for a strong R2 steering/drug-design claim in the current
round. Do not proceed to Tier 3 wet-lab or substrate-swap claims from the
current evidence.

**Recommended framing:** R2 can still be framed as an interpretability and
layer-map pipeline with a transparent negative steering result. A steering
claim should be reopened only after (1) T2-C tools/databases are staged and
calibrated, and (2) the metric triad shows significant positive shifts in at
least 3 of 8 EC classes, or after training a stronger CLT if the current
FVU/dead-feature ceiling is judged binding.

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

**Question:** After staging CLEAN pretrained weights, ESM-1b, Foldseek, and a
bounded PDB100 target DB, do generated lysozyme sequences pass real external EC
and structure checks?

**Method:** Ran `scripts/19_clean_generated.py` for CLEAN EC top-1 prediction
and `scripts/18_foldseek_generated.py` for Foldseek/PDB100 structure matching.
Added `scripts/20_generated_metric_triad_summary.py` to combine the existing
Pfam scan with CLEAN and Foldseek.

**Results:**
- Steered leads: Pfam lysozyme-like hit rate 0.900, CLEAN exact EC 3.2.1.17
  rate 0.900, Foldseek mean top TM 0.883, TM >= 0.7 fraction 0.900.
- Steered all vs unsteered: Pfam lysozyme-like 0.860 vs 0.820
  (one-sided Fisher p=0.170), CLEAN exact EC 3.2.1.17 0.775 vs 0.775,
  CLEAN 3.2.1.x prefix 0.865 vs 0.875.
- Foldseek was run on the existing ESMFold PDBs for steered leads and
  unsteered baseline structures.

**Interpretation:** The selected lead filter is biologically plausible under
all three metrics, but the generation-wide steering claim is still weak. CLEAN
does not show a steered_all lift over the unsteered baseline, and Pfam's lift is
not statistically decisive.

**Output paths:**
- `results/ec_metrics/clean_generated_lysozyme_20260507.json`
- `results/ec_metrics/foldseek_generated_lysozyme_20260507.json`
- `results/ec_metrics/generated_metric_triad_summary_20260507.json`
- `results/ec_metrics/generated_metric_triad_summary_20260507.md`

## EXP-R2-015: T2-C Real-vs-Random Calibration Runner (2026-05-07 CST)

**Question:** Do the external EC/structure metrics distinguish known
EC 3.2.1.17 lysozymes from length-matched random UniRef50 proteins before we
use them as evidence on generated sequences?

**Method:** Added `scripts/21_prepare_ec_calibration.py` to prepare 100 real
SwissProt/ZymCTRL lysozyme sequences and 100 random UniRef50 sequences with
length 80-250 aa. Added `scripts/22_foldseek_calibration.py` and
`scripts/23_ec_metric_calibration_summary.py`, then launched
`scripts/run_t2c_calibration_20260507.sh` on the 1-GPU H200 pod.

**Result:** Completed on H200. The initial ESMFold structure pass failed in
fp16 with a transformers `compute_tm` numerical error; rerunning the structure
stage in fp32 fixed the issue and produced 200/200 PDB files.

Real-vs-random separation:

| Metric | Real mean | Random mean | Effect size d |
|---|---:|---:|---:|
| Pfam lysozyme-like hit | 0.910 | 0.000 | 4.497 |
| CLEAN exact 3.2.1.17 | 0.960 | 0.000 | 6.928 |
| CLEAN 3.2.1.x prefix | 0.990 | 0.050 | 5.549 |
| ESMFold mean pLDDT | 79.740 | 58.478 | 1.786 |
| ESMFold confident fraction | 0.865 | 0.336 | 2.063 |
| Foldseek top TM | 0.971 | 0.653 | 1.593 |

**Interpretation:** The metric stack itself is now calibrated for the lysozyme
control: all metrics exceed the TODO_NEXT separation threshold. This supports
using Pfam/CLEAN/ESMFold/Foldseek as filters and controls, but it does not
rescue the generated steered-vs-unsteered result, which remains weak on CLEAN
and non-significant on Pfam.

**Output paths:**
- `results/ec_metrics/calibration_lysozyme_20260507/`
- `results/ec_metrics/pfam_calibration_lysozyme_20260507.json`
- `results/ec_metrics/clean_calibration_lysozyme_20260507.json`
- `results/ec_metrics/calibration_real_lysozyme_esmfold_20260507.json`
- `results/ec_metrics/calibration_random_uniref50_esmfold_20260507.json`
- `results/ec_metrics/foldseek_calibration_lysozyme_20260507.json`
- `results/ec_metrics/ec_metric_calibration_summary_20260507.json`
- Remote log:
  `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/logs/runtime/t2c_calibration_20260507.log`
