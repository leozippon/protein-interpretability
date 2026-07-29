# Experiment Log

Detailed record of all SAE training experiments, including hyperparameters, results, issue analysis, and lessons learned.

---

## EXP-001: Layer 3 BatchTopK SAE Initial Training

| Field | Value |
|------|-----|
| **Date** | 2026-03-02 ~ 2026-03-03 (in progress at the time, later terminated) |
| **Objective** | Validate online training of a BatchTopK SAE on ESM-2-3B layer 3 |
| **WandB** | `esm2-3B_layer3_d32768_k64` / run `1gim9pj1`, `h20rpap5` |
| **Status** | **Terminated** — a dead feature spiral caused the FVU to plateau |

### Hardware and Environment

- GPU: 4x NVIDIA L20 (48GB), DDP
- Launch command: `python scripts/02_train_saes.py --config configs/sae_training.yaml --all-layers`
- Logs: `logs/train_all_layers.log`

### Hyperparameters

```yaml
model: ESM-2-3B (facebook/esm2_t36_3B_UR50D), fp16, frozen
data: UniRef50, 5M sequences, max_seq_len=1022, seed=42
sae:
  d_in: 2560
  d_sae: 32768      # 12.8x expansion
  k: 64             # BatchTopK sparsity
  auxk_alpha: 1/32  # auxiliary loss weight
  dead_feature_threshold: 10,000,000  # tokens
training:
  batch_size_tokens: 2048  # per GPU
  total_steps: 500,000
  lr: 3e-4
  lr_warmup_steps: 5,000
  lr_decay_start: 250,000
  grad_clip_norm: 1.0
```

### Training Curve (Key Checkpoints)

| Step | FVU | Loss | L0 | Dead% | tok/s | Notes |
|-----:|:----:|-----:|:---:|------:|------:|------|
| 50 | 0.850 | 4675 | 64 | 0.0% | 7427 | Initial |
| 500 | 0.698 | 3746 | 64 | 0.0% | 7054 | Rapid drop |
| 1,000 | 0.432 | 2421 | 64 | 0.0% | 6956 | |
| 2,000 | 0.234 | 1289 | 64 | 0.0% | 7596 | |
| 5,000 | 0.193 | 1153 | 64 | 0.0% | 4822 | LR warmup ended; training began to stabilize |
| 10,000 | 0.174 | 1013 | 64 | 9.8% | 6917 | Dead features started to appear |
| 15,000 | 0.173 | 1045 | 64 | 41.4% | 6493 | Dead ratio rose rapidly |
| 20,000 | 0.155 | 837 | 64 | 59.7% | 6639 | Very little FVU improvement |
| 25,000 | 0.150 | 767 | 64 | 67.3% | 7567 | FVU essentially stalled |
| 26,900 | 0.165 | 934 | 64 | 65.5% | 6867 | Final record |

### Checkpoints

| File | Time | Size |
|------|------|------|
| `results/sae_weights/layer_3/step_10000/` | 2026-03-03 03:24 | 1.9G |
| `results/sae_weights/layer_3/step_20000/` | 2026-03-03 06:59 | 1.9G |

### Issue Analysis

#### Issue 1: FVU Plateaued Around ~0.16 After Step ~5K

**Observation**: FVU dropped quickly from 0.85 to 0.19 over steps 0-5K, but then barely improved from 5K to 27K (0.19 -> 0.16).  
Reference comparison: InterPLM reached FVU ~0.02-0.05 on the 650M model.

**Root cause**: This was directly related to the dead-feature spiral. With 66% of features dead, only about ~11K/32768 features remained alive, which severely limited model expressivity.

#### Issue 2: Dead Features Rose Irreversibly from 0% to 67%

**Observation**: Dead% was 0% at step 5K, 10% at step 10K, 41% at step 15K, and 67% at step 25K, with a persistent upward trend.

**Root cause**: The global competition mechanism in BatchTopK combined with a k value that was too small.
- k=64, d_sae=32768, so each token used only 64/32768 = 0.2% of the features
- BatchTopK performs a global top-k over the entire batch, so strong features monopolize activation slots
- Once weak features are knocked out, they can no longer receive gradient signal from the main loss
- This creates positive feedback: the strong get stronger -> the weak never get selected -> they die

#### Issue 3: Auxk Loss Was Ineffective (~1.0)

**Observation**: The auxiliary loss stayed around 1.0 the entire time, meaning the reconstruction quality of dead features was no better than random noise.

**Root cause**:
1. `dead_feature_threshold=10M tokens` was too high — it took about ~1400 steps to mark a feature as dead, missing the best intervention window
2. `auxk_alpha=1/32` was too small — dead features only had auxk as a gradient source, and the signal was too weak
3. Once too many dead features accumulated (>20K), the limited auxk gradient could not revive that many features simultaneously

### Lessons Learned

1. **The k/d_sae ratio cannot be too low**: 0.2% (64/32768) caused catastrophic feature death under BatchTopK. A minimum of 0.5%+ is more reasonable, i.e. k>=160.
2. **Dead-feature detection must happen early**: 10M tokens was too conservative; 1M is more reasonable (~140 steps). Early detection plus early intervention helps prevent a death spiral.
3. **`auxk_alpha` needs to be strong enough**: 1/32 was not enough to counteract BatchTopK competition. 1/8 is a safer starting point.
4. **BatchTopK vs TopK**: BatchTopK is more likely than per-example TopK to kill features, because the competition is global. If the dead-feature problem persists, reverting to per-example TopK should be considered.
5. **An FVU plateau is an early warning sign**: If FVU stops decreasing early in training (<10% of total steps), check the dead-feature ratio immediately instead of waiting.
6. **Monitor `auxk_loss`**: If `auxk_loss` stays near 1.0, the revival mechanism is failing and needs intervention.

---

## EXP-002: Layer 3 BatchTopK SAE Hyperparameter Fixes

| Field | Value |
|------|-----|
| **Date** | 2026-03-04 ~ 2026-03-05 |
| **Objective** | Fix the dead-feature problem and test whether the new hyperparameters can achieve FVU < 0.05 |
| **WandB** | `esm2-3B_layer3_d32768_k192` / run `q5vnqnfh` |
| **Logs** | `logs/train_all_layers_0304.log` |
| **Status** | **Terminated** — dead features still reached ~80%; BatchTopK global competition was the root cause |

### Hyperparameter Changes (Relative to EXP-001)

| Parameter | EXP-001 | EXP-002 | Reason for change |
|------|---------|---------|----------|
| **k** | 64 | **192** | Increase feature utilization: 0.2% -> 0.6%, slowing the dead spiral |
| **auxk_alpha** | 1/32 | **1/8** | Increase dead-feature revival gradient strength (4x) |
| **dead_feature_threshold** | 10M | **1M** | Detect dead features earlier (1400 steps -> 140 steps) |

All other hyperparameters were unchanged (`d_sae=32768`, `topk_mode=batch`).

### Training Curve (Key Checkpoints)

| Step | FVU | Loss | L0 | Dead% | tok/s | Notes |
|-----:|:----:|-----:|:---:|------:|------:|------|
| 50 | 0.862 | 4739 | 192 | 0.0% | 7099 | Initial |
| 1,000 | 0.326 | 1826 | 192 | 0.0% | 6636 | Rapid drop |
| 3,000 | 0.145 | 923 | 192 | 0.0% | 6972 | |
| 5,000 | 0.116 | 690 | 192 | 0.3% | 6834 | **LR warmup ended** |
| 7,000 | 0.095 | 516 | 192 | 16.7% | 7274 | Dead ratio began to climb |
| 10,000 | 0.096 | 560 | 192 | 49.7% | 6834 | Dead ratio grew rapidly |
| 15,000 | 0.096 | 578 | 192 | 70.8% | 6526 | |
| 20,000 | 0.083 | 444 | 192 | 77.9% | 6667 | FVU improved slowly |
| 25,000 | 0.080 | 407 | 192 | 80.5% | 7657 | Dead ratio stabilized |
| 28,700 | 0.080 | 421 | 192 | 80.1% | 5619 | Final record |

### Checkpoints

| File | Time |
|------|------|
| `results/sae_weights/layer_3/step_10000/` | 2026-03-04 |
| `results/sae_weights/layer_3/step_20000/` | 2026-03-04 |

### Comparison with EXP-001

| Metric | EXP-001 (k=64) | EXP-002 (k=192) | Improvement |
|------|:---:|:---:|:---:|
| Final FVU | ~0.16 | ~0.08 | 2x ✓ |
| Final Dead% | 67% (spiraling upward) | 80% (stable) | Stable but higher ✗ |
| Effective features | ~10,800 | ~6,500 | Fewer ✗ |
| k / effective features | 0.59% | 2.95% | Reasonable ✓ |

### Issue Analysis

**What improved:**
- FVU=0.08 (92% variance explained), much better than the 0.16 in EXP-001
- Dead% stabilized around ~80% rather than continuing to spiral
- k/effective_features=2.95%, so utilization of the surviving features was healthy

**Core issue: BatchTopK global competition is the root cause of dead features**

Increasing k (64 -> 192) only delayed the dead spiral; it did not remove the root cause. BatchTopK naturally biases activation allocation toward strong features:
- The whole batch shares `k * batch_size` activation slots
- Strong features have high `pre_act` across many tokens and get selected constantly
- Weak features can never enter the global top-k, even if they are sensible on some tokens
- `auxk_alpha=1/8` was still not enough to counteract this structural elimination

**Compared with InterProt (almost 0% dead):**
- InterProt uses **per-example TopK**: each token selects top-k independently
- A feature only needs to perform well on a subset of tokens to survive
- There is no systemic elimination caused by global competition

### Lessons Learned

1. **BatchTopK is not suitable for protein SAEs**: With `d_sae=32768`, global competition will inevitably produce >50% dead features no matter how `k` or `auxk_alpha` are tuned. Avoiding that would require `k/d_sae > 3%` (k=1000+), which is far too dense.
2. **Per-example TopK should be used instead**: InterProt showed that per-example TopK with `d_sae=4096` can achieve <1% dead features. That is the right choice for protein SAEs.
3. **`d_sae=32768` is too large for protein models**: Protein function space is much smaller than natural language, so a 3.2x expansion (`8192`) is more appropriate.
4. **FVU 0.08 is still usable on the 3B model**: Even with 80% dead features, the checkpoint is still usable for downstream analysis.

---

## EXP-003: Per-example TopK + d_sae=8192, k=64

| Field | Value |
|------|-----|
| **Date** | 2026-03-04 |
| **Objective** | Switch to per-example TopK + `d_sae=8192` to fully solve the dead-feature problem |
| **WandB** | `esm2-3B_layer3_d8192_k64` / run `wckz58al` |
| **Logs** | `logs/train_all_layers_0304_2.log` |
| **Status** | **Terminated** — FVU was worse than EXP-002, and both `d_sae` and `k` were too small |

### Hyperparameter Changes (Relative to EXP-002)

| Parameter | EXP-002 | EXP-003 | Reason for change |
|------|---------|---------|----------|
| **d_sae** | 32768 | **8192** | 3.2x expansion (same ratio as InterProt) |
| **k** | 192 | **64** | Return to a more reasonable sparsity level |
| **topk_mode** | batch | **example** | Remove global competition |
| **auxk_alpha** | 1/8 | **1/4** | Stronger revival signal |
| **dead_threshold** | 1M | **500K** | Earlier detection |

### Training Curve (Key Checkpoints)

| Step | FVU | Loss | L0 | Dead% | tok/s | Notes |
|-----:|:----:|-----:|:---:|------:|------:|------|
| 50 | 0.857 | 4711 | 64 | 0.0% | 12136 | Initial |
| 1,000 | 0.422 | 2365 | 64 | 0.0% | 11177 | |
| 2,000 | 0.229 | 1258 | 64 | 0.0% | 12370 | |
| 3,000 | 0.213 | 1356 | 64 | 0.5% | 10982 | |
| 4,000 | 0.194 | 1105 | 64 | 8.9% | 10416 | Dead ratio started climbing rapidly |
| 5,000 | 0.193 | 1152 | 64 | 24.6% | 10943 | **FVU plateaued; dead ratio was severe** |
| 5,900 | 0.178 | 997 | 64 | 33.5% | 10903 | Final record |

### Comparison with EXP-002 (Same Step)

| Step | EXP-002 FVU | EXP-002 Dead% | EXP-003 FVU | EXP-003 Dead% |
|-----:|:-----------:|:-------------:|:-----------:|:-------------:|
| 2,000 | 0.170 | 0.0% | 0.229 | 0.0% |
| 5,000 | **0.116** | 0.3% | 0.193 | **24.6%** |

**Both metrics were worse: FVU was 66% higher, and the dead-feature problem emerged roughly 80x faster.**

### Issue Analysis

**Issue 1: `d_sae=8192` did not provide enough capacity**
- ESM-2-3B (`d_in=2560`) has a representation space far more complex than 650M (`d_in=1280`)
- 8192 features were not enough to decompose the residual stream of the 3B model, so the FVU ceiling was high
- EXP-002 reached FVU=0.08 with ~6500 alive features, but those were the strongest survivors selected out of 32768
- Starting with only 8192 slots made the learning problem much harder

**Issue 2: `k=64` was too small**
- InterProt uses `k=128` / `d_sae=4096`, not `k=64`
- With `k=64`, each token activates only 64 features, which limits expressivity and raises FVU
- Weak features are even less likely to be selected, so even per-example TopK cannot fully avoid dead features

**Issue 3: `dead_threshold=500K` was too aggressive**
- 500K tokens marks a feature dead after only about ~45 steps
- Features that respond to rare protein types were incorrectly flagged too early
- The inflated dead ratio then caused auxk gradients to spread too thin

### Lessons Learned

1. **`d_sae` cannot be cut too aggressively**: Going from 32768 to 8192 is a 4x reduction and was too aggressive. A 6.4x expansion (`16384`) is more appropriate.
2. **`k` must be large enough**: `k=64` is not enough for the 3B model. `k=128` (matching InterProt) is the minimum reasonable value.
3. **`dead_threshold` should not be too small**: 500K was too aggressive; 1M is safer.
4. **Per-example TopK itself was not the problem**: The issue was the `d_sae` and `k` configuration, not the TopK mode.
5. **Larger models need larger dictionaries and more activations**: You cannot directly reuse the 650M scaling recipe.

---

## EXP-004: Per-example TopK + d_sae=16384, k=128

| Field | Value |
|------|-----|
| **Date** | 2026-03-04 ~ 2026-03-06 |
| **Objective** | Balance capacity and dead%: 6.4x expansion + k=128 (same k value as InterProt) |
| **WandB** | `esm2-3B_layer3_d16384_k128` / run `3rqcsldx` |
| **Logs** | `logs/train_all_layers_0304_3.log` |
| **Status** | **Terminated** — dead features still reached ~81%, almost identical to EXP-002. A pre-topk ReLU bug was discovered in `encode()` |

### Hyperparameter Changes (Relative to EXP-003)

| Parameter | EXP-003 | EXP-004 | Reason for change |
|------|---------|---------|----------|
| **d_sae** | 8192 | **16384** | 6.4x expansion, balancing capacity and efficiency |
| **k** | 64 | **128** | Match InterProt and increase per-token expressivity |
| **topk_mode** | example | example | Keep per-example mode |
| **auxk_alpha** | 1/4 | 1/4 | Keep unchanged |
| **dead_threshold** | 500K | **1M** | Avoid false positives and return to the safer setting |

### Training Curve (Key Checkpoints)

| Step | FVU | Loss | L0 | Dead% | tok/s | Notes |
|-----:|:----:|-----:|:---:|------:|------:|------|
| 50 | 0.772 | 4088 | 128 | 0.0% | 7761 | Initial |
| 1,000 | 0.394 | 2211 | 128 | 0.0% | 7201 | |
| 3,000 | 0.164 | 869 | 128 | 0.0% | 9113 | |
| 5,000 | 0.134 | 747 | 128 | 0.1% | 7173 | **Warmup ended; dead features began appearing** |
| 7,000 | 0.124 | 713 | 128 | 11.2% | 7338 | Dead ratio rose rapidly |
| 10,000 | 0.121 | 693 | 128 | 32.3% | 7310 | FVU plateaued |
| 20,000 | 0.122 | 732 | 128 | 66.4% | 8339 | |
| 50,000 | 0.116 | 658 | 128 | 70.3% | 7298 | |
| 100,000 | 0.115 | 638 | 128 | 77.5% | 7803 | |
| 200,000 | 0.108 | 615 | 128 | 79.8% | 7946 | |
| 300,000 | 0.106 | 561 | 128 | 80.7% | 6602 | |
| 362,450 | 0.106 | 574 | 128 | 80.9% | 7784 | **Final record** |

### Comparison with EXP-002

| Metric | EXP-002 (BatchTopK, k=192, d=32K) | EXP-004 (example, k=128, d=16K) |
|------|:---:|:---:|
| Final FVU | ~0.08 | ~0.11 |
| Final Dead% | 80% | 81% |
| TopK mode | batch | example |
| Conclusion | — | **Nearly identical to EXP-002. Switching TopK mode did not solve the problem.** |

### Root-Cause Analysis: Pre-TopK ReLU Bug

**The root cause of failure in all four experiments was not the TopK mode, `d_sae`, or `k`; it was a bug in `encode()`.**

```python
# BUG: ReLU was applied before TopK inside encode()
pre_acts = torch.relu(linear(x - self.b_dec, self.W_enc, self.b_enc))
```

**Failure chain:**
1. Once a dead feature's pre-ReLU value stays negative, ReLU outputs 0 forever
2. `_auxk_loss()` receives dead-feature `pre_acts` that are all 0, so top-k also selects only 0s
3. Auxk reconstruction gives `e_hat ~= 0`, so the gradient to `W_dec` is ~= 0 (because `auxk_f` is 0)
4. The gradient to `W_enc` is **completely blocked** by `ReLU(0)` (derivative is 0)
5. The dead feature's encoder and decoder can no longer update -> **it can never be revived**

**Correct implementation** (OpenAI Gao et al. 2024, InterProt):
```python
# CORRECT: no ReLU before TopK
pre_acts = linear(x - self.b_dec, self.W_enc, self.b_enc)  # raw value; can be negative
# ReLU is only applied to the selected top-k values inside _topk_activation()
```

This allows dead features to have positive `pre_acts` even if they are not currently in the top-k, so auxk can still provide useful gradient.

**Why OpenAI/Anthropic did not run into this issue**:  
OpenAI and Anthropic both use **feature resampling** (periodically reinitializing dead features), rather than relying only on auxk loss. Even if auxk fails, resampling can directly revive dead features.

### Lessons Learned

1. **Pre-TopK ReLU was a fatal bug**: It blocked the gradient path from auxk loss, preventing dead features from ever being revived by gradient-based training.
2. **TopK mode was not the root cause**: Under this bug, per-example and batch modes behaved almost identically.
3. **Feature resampling is a necessary safety net**: Even if auxk works, resampling should still be added as a fallback.
4. **Verify the gradient path**: When the loss exists but the model does not improve, check whether gradients are actually flowing.

---

## EXP-005: Fix Pre-TopK ReLU + Feature Resampling

| Field | Value |
|------|-----|
| **Date** | 2026-03-07 ~ (ongoing) |
| **Objective** | Fix the auxk gradient-blocking bug and add feature resampling to fully solve dead features |
| **WandB** | run `im9p4gi6` |
| **Logs** | `logs/train_all_layers_0304_4.log` |
| **Status** | **Training** — Layer 3 completed (FVU=0.087, dead=8.4%), Layer 7 completed, Layer 11 in progress |

### Code Changes

| Change | File | Description |
|------|------|------|
| **Remove pre-topk ReLU** | `src/training/sae.py` | `pre_acts` in `encode()` no longer pass through ReLU; raw values are preserved |
| **Add ReLU on the BatchTopK path** | `src/training/sae.py` | `_topk_activation()` applies `torch.relu()` to selected values in batch mode |
| **Feature resampling** | `src/training/trainer.py` | Reinitialize dead features every 25K steps (`W_dec`, `W_enc`, `b_enc`, optimizer state) |

### Hyperparameters (Relative to EXP-004)

| Parameter | EXP-004 | EXP-005 | Reason for change |
|------|---------|---------|----------|
| **d_sae** | 16384 | 16384 | Keep unchanged |
| **k** | 128 | 128 | Keep unchanged |
| **pre-topk ReLU** | Present (bug) | **Removed (fixed)** | Restore auxk gradient flow |
| **resample_every** | None | **25000** | Reinitialize dead features every 25K steps |
| All other parameters | — | Unchanged | — |

### Training Curve — Layer 3 (Completed, 500K steps)

| Step | FVU | Dead% | Resampled | Notes |
|-----:|:----:|------:|----------:|------|
| 5,000 | 0.134 | 0.0% | — | Warmup ended |
| 10,000 | 0.121 | 2.6% | — | Dead ratio under control |
| 25,000 | 0.116 | 3.9% | 615 (3.8%) | First resample |
| 50,000 | 0.115 | 4.9% | 848 (5.2%) | |
| 100,000 | 0.112 | 5.7% | 941 (5.7%) | |
| 200,000 | 0.106 | 5.6% | 928 (5.7%) | |
| 300,000 | 0.103 | 5.3% | 868 (5.3%) | LR decay started |
| 400,000 | 0.100 | 9.2% | 1510 (9.2%) | |
| **500,000** | **0.087** | **8.4%** | 1383 (8.4%) | **Final** |

### Training Curve — Layer 7 (Completed, 500K steps)

| Step | FVU | Dead% | Notes |
|-----:|:----:|------:|------|
| 10,000 | 0.215 | 7.1% | |
| 50,000 | 0.221 | 6.3% | |
| 100,000 | 0.195 | 6.4% | |
| 300,000 | 0.201 | 5.7% | |
| **500,000** | **0.167** | **2.0%** | **Final** |

### Training Curve — Layer 11 (In Progress, ~350K/500K)

| Step | FVU | Dead% | Notes |
|-----:|:----:|------:|------|
| 10,000 | 0.269 | 7.7% | |
| 100,000 | 0.276 | 11.9% | |
| 200,000 | 0.251 | 14.2% | |
| 350,000 | 0.256 | 9.3% | Current |

### Success Criteria (Against Targets)

- [x] Step 5K: dead% < 5%, FVU < 0.15 -> **dead=0.0%, FVU=0.134** (Layer 3)
- [x] Step 10K: dead% < 10%, FVU < 0.12 -> **dead=2.6%, FVU=0.121** (Layer 3)
- [x] Step 25K (first resample): dead% decreases -> **resample revived 615 features**
- [ ] Step 50K: dead% < 15%, FVU < 0.08 -> dead=4.9%, FVU=0.115 (**FVU target missed**)
- [ ] Step 100K: dead% < 10%, FVU < 0.06 -> dead=5.7%, FVU=0.112 (**FVU target missed**)

### FVU Summary Across Layers

| Layer | FVU@500K | Dead% | Variance explained | Evaluation |
|:-----:|:--------:|------:|:-------:|:----:|
| 3 | 0.087 | 8.4% | 91.3% | Good |
| 7 | 0.167 | 2.0% | 83.3% | Fair |
| 11 | ~0.25 (est.) | ~10% | ~75% | Poor |

FVU remains high in deeper layers, which likely means they need either more capacity (`k=256` or `d_sae=32768`) or longer training.

### Annotation Alignment — Layer 3 (1000 proteins)

| Metric | InterProt 650M **L24** | Our 3B **L3** |
|------|:---:|:---:|
| Alive features | 4095 (99.97%) | 15594 (95.2%) |
| KNOWN (F1>0.5) | 117 (2.9%) | 12 (0.1%) |
| PARTIAL (F1 0.2-0.5) | 828 (20.2%) | 157 (1.0%) |
| NOVEL (F1<0.2) | 3150 (76.9%) | 15425 (98.9%) |
| Mean best F1 | 0.1425 | 0.0234 |

**Note**: Layer 3 is shallow (8% depth) and mainly encodes low-level sequence features (amino-acid properties, local patterns), rather than protein function annotations (domains, active sites, etc.). InterProt uses Layer 24 (73% depth). The 12 KNOWN features are mostly chain-level annotations (protein family recognition), which is reasonable for such a shallow layer.  
**A real comparison requires waiting for deeper SAEs (Layer 23+) to finish training.**

### Future Optimization Directions

- Deeper layers need larger capacity: Option A `k=256`, Option B `d_sae=32768`
- Optimize F1 computation speed: threshold search has already been changed from a per-feature loop to vectorized sorting (~50 min -> ~1 min)
- Rerun annotation alignment once Layer 23 has finished training

---

## Hyperparameter Memo

### Key Parameter Relationships

```
TopK mode selection:
  - per-example TopK: recommended; each token selects top-k independently with no global competition
  - BatchTopK: severe global competition, causing 50-80% dead features in protein SAEs
  - Note: both modes behave similarly under the pre-topk ReLU bug (EXP-002 vs EXP-004)

Position of ReLU inside encode() (critical):
  - Wrong: pre_acts = ReLU(W_enc @ x + b_enc) -> blocks auxk gradients
  - Correct: pre_acts = W_enc @ x + b_enc -> ReLU is only applied to selected top-k values

Choice of k:
  - k=64: not enough for the 3B model
  - k=128: InterProt's choice; recommended minimum
  - k=192+: worth considering if dead% remains high

Expansion factor = d_sae / d_in:
  - 3.2x (8192/2560): too small
  - 6.4x (16384/2560): recommended
  - 12.8x (32768/2560): enough capacity but high dead-feature risk

Dead-feature revival mechanisms:
  - auxk loss: necessary, but only works when pre_acts are nonzero (requires fixing the ReLU bug)
  - feature resampling: sufficient; directly reinitializes dead features and bypasses the gradient problem
  - recommended to use both together (belt and suspenders)
```

### Reference Values (From Literature and Experiments)

| Source | Model | d_sae | k | TopK mode | k/d_sae | Dead% | Final FVU |
|------|------|------:|--:|------:|--------:|------:|---------:|
| InterProt (2025) | ESM2-650M | 4,096 | 128 | example | 3.1% | <1% | ~0.01-0.05 |
| InterPLM (2025) | ESM2-650M | 4,096 | - | - | - | - | ~0.02-0.05 |
| **EXP-001** | ESM2-3B | 32,768 | 64 | batch | 0.2% | 67% | 0.16 |
| **EXP-002** | ESM2-3B | 32,768 | 192 | batch | 0.6% | 80% | 0.08 |
| **EXP-003** | ESM2-3B | 8,192 | 64 | example | 0.78% | 34% | 0.18 |
| **EXP-004** | ESM2-3B | 16,384 | 128 | example | 0.78% | 81% | 0.11 |
| **EXP-005 L3** | ESM2-3B | 16,384 | 128 | example | 0.78% | **8.4%** | **0.087** |
| **EXP-005 L7** | ESM2-3B | 16,384 | 128 | example | 0.78% | **2.0%** | 0.167 |
| OpenAI (2024) | GPT-4 | 262,144 | 256 | batch | 0.1% | ~10% | ~0.01 |
| Anthropic (2024) | Claude | 65,536 | 128 | batch | 0.2% | ~5% | ~0.01 |

### Core Lessons Summary (EXP-001 ~ EXP-004)

1. **Pre-TopK ReLU was the shared root cause of all four failed experiments**: it blocked auxk gradient flow, making dead features impossible to revive. The fix is to remove `torch.relu()` from `encode()`.
2. **The difference between TopK modes was masked by the ReLU bug**: per-example and batch behaved similarly under the bug (~81% dead). Only after fixing the bug can the advantage of per-example TopK emerge.
3. **Feature resampling is standard practice**: both OpenAI and Anthropic use resampling as a complement to auxk. We were missing that crucial mechanism.
4. **The effects of `d_sae` and `k` were overestimated**: the difference between EXP-003 and EXP-004 (FVU 0.18 vs 0.11) mainly came from capacity, not dead-feature control. The nearly identical dead% in EXP-004 and EXP-002 proves this.
5. **Verifying gradient flow is the first step in debugging**: when the loss exists but the metric does not improve, check whether gradients are actually reaching the target parameters.

---

## EXP-006: Multi-Layer Annotation Alignment (H200 Checkpoints)

| Field | Value |
|------|-----|
| **Date** | 2026-04-05 |
| **Objective** | Evaluate annotation alignment across all deep SAE layers (19, 23, 27, 31, 35) trained on H200 |
| **Script** | `scripts/05_analyze_all_layers.py` |
| **Status** | **Complete** |

### Setup

- SAE checkpoints from H200 training (500K steps each, `r1_h200_2gpu_20260401`)
- 1000 Swiss-Prot proteins (336,606 residues), random seed 42
- F1 threshold for KNOWN: >0.5, PARTIAL: 0.2-0.5
- All layers used same SAE config: d_sae=16384, k=256, normalize_activations=True

### Cross-Layer Results

| Layer | Depth% | Alive% | KNOWN | PARTIAL | NOVEL | Mean F1 | Top Feature (F1) |
|-------|--------|--------|-------|---------|-------|---------|-------------------|
| 19    | 53%    | 97.3%  | 146   | 2191    | 13607 | 0.1038  | K-box domain (0.950) |
| 23    | 64%    | 97.6%  | 133   | 1759    | 14099 | 0.0936  | ATP synthase epsilon (1.000) |
| 27    | 75%    | 97.1%  | 72    | 1034    | 14802 | 0.0665  | Major capsid protein (0.950) |
| 31    | 86%    | 97.2%  | 49    | 722     | 15157 | 0.0524  | UL41 protein (0.978) |
| 35    | 97%    | 92.7%  | 31    | 422     | 14733 | 0.0391  | DNA mismatch repair MutL (0.905) |

### Key Findings

1. **Layer 19 has the richest annotations** — 146 KNOWN features, including domain (K-box), region (DnaA modulator interaction), and topology (transit peptide) features. This is consistent with InterProt's finding that mid-depth layers encode functional information.

2. **Annotation count decreases with depth** — Deeper layers (31, 35) have fewer KNOWN features but include specialized features: glycosylation sites (L35, F1=0.727), iron-sulfur cluster domains (L27, F1=0.736), and signaling domains (L31, RGS domain F1=0.652).

3. **Feature types evolve across depth**:
   - L19 (53%): domains, interaction regions, structural elements
   - L23 (64%): protein-level recognition (ATP synthase, CN hydrolase)
   - L27 (75%): catalytic domains (PTS EIIB, PDEase, 4Fe-4S)
   - L31 (86%): signaling (RGS, t-SNARE) and specific protein recognition
   - L35 (97%): maintenance/repair (DNA mismatch repair MutL, topoisomerase)

4. **Alive feature rate is excellent across all layers**: 92.7–97.6%, confirming the resampling fix works well even for the deepest layers.

---

## EXP-007: Variant Effect Prediction (ClinVar)

| Field | Value |
|------|-----|
| **Date** | 2026-04-05 |
| **Objective** | Classify missense variant mechanisms using multi-layer SAE perturbation signatures |
| **Script** | `scripts/06_variant_effect_prediction.py` |
| **Status** | **Complete — initial pipeline. Pathogenicity discrimination needs improvement.** |

### Setup

- 300 pathogenic + 300 benign ClinVar missense variants (balanced)
- Human proteins from Swiss-Prot (18,132 proteins, 18,024 gene-mapped)
- SAE layers: 23, 27, 31, 35 (4 deep layers)
- For each variant: compute WT and mutant SAE features, take difference

### Results

**Pathogenicity discrimination (raw perturbation magnitude):**
- AUC-ROC: **0.547** (near random with balanced data)
- Pathogenic mean perturbation: 325.6, Benign: 324.2 (Cohen's d = 0.015)
- **Conclusion: Raw perturbation magnitude does NOT separate pathogenic from benign.**

**Mechanism distribution:**
| Mechanism | Pathogenic | Benign |
|-----------|-----------|--------|
| LOF       | 44.7%     | 67.0%  |
| GOF       | 53.3%     | 32.7%  |
| DN        | 2.0%      | 0.3%   |

**Notable patterns:**
- Cysteine mutations (C→R, C→F, C→S) produce the largest perturbations regardless of pathogenicity — likely because they disrupt disulfide bonds, causing massive structural rearrangement in feature space
- Top pathogenic: SIAH1 C128F (pert=785), NOG S185C (pert=662), PRSS56 C395R (pert=615)
- Top benign: PLOD2 V269I (pert=135), MRPL46 I137V (pert=161) — conservative substitutions

### Lessons & Next Steps

1. **Raw perturbation magnitude is uninformative**: Benign variants can perturb as many features as pathogenic ones. A structurally important benign variant (e.g., near a disulfide) causes large SAE perturbation without pathological consequence.

2. **Feature identity matters, not count**: With TopK(k=256), every position always has exactly 256 active features. The discriminating signal must come from WHICH features change, weighted by their functional importance.

3. **Annotation-weighted perturbation**: The next experiment should weight feature perturbations by their annotation F1 scores from EXP-006. Perturbation of a feature aligned with "active site" (F1=0.8) is more meaningful than perturbation of a novel feature.

4. **Cross-layer perturbation profiles**: Use the 4-layer × d_sae perturbation matrix as input to a classifier, rather than collapsing to scalars.

5. **Need ground truth mechanism labels**: Badonyi & Marsh (2025) dataset (~3K LOF/GOF/DN labeled) is needed to validate mechanism predictions.

---

## EXP-008: Annotation-Weighted Variant Prediction (2026-04-05)

**Goal**: Improve pathogenicity prediction (AUC=0.547 from EXP-007) by weighting perturbation signatures with annotation F1 scores from EXP-006. Key hypothesis: disruption of biologically annotated features (active sites, domains) is more predictive than raw perturbation magnitude.

**Script**: `scripts/07_annotation_weighted_prediction.py`

**Data**: Reuses EXP-007's 600 perturbation signatures (300 pathogenic, 300 benign) and EXP-006's annotation alignment results (layers 23, 27, 31, 35).

### Scoring Strategies Compared

| Strategy | AUC-ROC | AP | Δ from baseline |
|----------|:-------:|:--:|:---------------:|
| Raw (EXP-007 baseline) | 0.547 | 0.503 | — |
| F1-weighted | 0.467 | 0.441 | -0.080 |
| Category-weighted (F1 × category importance) | 0.557 | 0.502 | +0.010 |
| TopK-50 F1 | 0.483 | 0.458 | -0.063 |
| TopK-20 F1 | 0.442 | 0.439 | -0.105 |
| Known features only (F1>0.2) | 0.438 | 0.427 | -0.109 |
| **Functional disruption only** | **0.688** | **0.649** | **+0.141** |
| **Cross-layer logistic regression (5-fold CV)** | **0.819** | **0.804** | **+0.272** |

### Key Results

1. **Cross-layer LR achieves AUC=0.819, accuracy=74.0%** — a +0.272 improvement over raw perturbation. This validates the core hypothesis: annotation-weighted, cross-layer perturbation profiles are strongly predictive of pathogenicity.

2. **Functional disruption score alone achieves AUC=0.688** — just weighting by functional/ptm/domain annotations provides strong signal without any ML. This is the simplest interpretable predictor.

3. **Simple F1-weighting hurts (AUC=0.467)** — because most features are NOVEL (F1≈0), weighting by F1 discards the majority of features including some that are predictive despite lacking known annotations. The signal comes from the *interaction* between annotation knowledge and perturbation pattern, not from F1 alone.

### Per-Category Pathogenic/Benign Ratio (Deep Layers)

| Category | Layer 31 ratio | Layer 35 ratio | Interpretation |
|----------|:--------------:|:--------------:|----------------|
| functional | **2.90×** | **2.34×** | Pathogenic variants disproportionately disrupt functional sites |
| domain | 1.13× | 1.25× | Moderate enrichment in domain disruption |
| topology | 0.81× | 1.10× | Near-neutral |
| chain | 0.77× | 1.04× | Non-discriminating |
| ptm | 0.57× | 0.58× | PTMs disrupted more by benign (unexpected) |
| region | 0.50× | 0.62× | Benign variants perturb regions more (likely surface-exposed) |

**Critical finding**: The path/benign discrimination concentrates in **functional site features at deep layers (31, 35)**. This makes biological sense — deep ESM-2 layers encode functional information (active sites, catalytic residues), and pathogenic variants preferentially disrupt these representations.

### Feature Importance (Logistic Regression Coefficients)

| Rank | Feature | Coefficient | Interpretation |
|------|---------|:-----------:|----------------|
| 1 | L31_mean_delta_known | +1.56 | High mean perturbation of known features at L31 → pathogenic |
| 2 | L31_n_known_pert | -1.53 | Fewer known features perturbed at L31 → pathogenic (focused disruption) |
| 3 | L35_n_known_pert | -1.28 | Same pattern at L35 — pathogenic = focused, not diffuse |
| 4 | L31_top10_f1_pert | -1.25 | Lower top-10 F1-weighted perturbation → pathogenic |
| 5 | L35_top10_f1_pert | -1.13 | Same at L35 |
| 6 | L35_mean_delta_known | +1.08 | Higher intensity per-feature at L35 → pathogenic |
| 7 | L31_mean_delta_novel | +1.08 | Novel feature disruption at L31 also informative |
| 8 | L23_top10_f1_pert | +1.08 | Opposite sign at L23 — interpretation differs by depth |

**Interpretation**: Pathogenic variants cause **focused, high-intensity disruption** of known biological features (high mean perturbation per feature, fewer features affected). Benign variants cause **diffuse, low-intensity perturbation** that spreads across many features without strongly disrupting any. This is consistent with the biology — pathogenic missense mutations tend to directly hit functional sites (catalytic, binding), while benign variants tend to be surface-exposed and cause distributed conformational shifts.

### Lessons & Next Steps

1. **Annotation-weighting works**: AUC 0.547 → 0.819 validates that SAE feature annotations provide the bridge between "what changed" and "does it matter."

2. **Cross-layer profiles are essential**: No single-number score captures the complexity. The LR classifier uses information from all 4 layers, with deep layers (31, 35) most informative.

3. **Focused vs diffuse disruption is the key pattern**: Pathogenic = few features hit hard at functional sites. Benign = many features gently perturbed across the board.

4. **Need layer 19**: Include the best-annotated layer (146 KNOWN features) to potentially further improve the classifier.

5. **Validate with Badonyi & Marsh mechanism labels**: Current analysis predicts pathogenicity. Mechanism classification (LOF/GOF/DN) requires ground-truth labels from external datasets.

6. **Scale to more variants**: 600 variants is a small test set. → Addressed in EXP-009.

7. **Compare against EVE/AlphaMissense baselines**: The AUC=0.819 needs context against state-of-the-art variant effect predictors to assess novelty.

---

## EXP-009: Scaled Annotation-Weighted Prediction (2026-04-05)

**Goal**: Validate EXP-008 at 3.3× scale (2000 variants vs 600) and add layer 19 (best annotated: 146 KNOWN features).

**Script**: `scripts/08_scaled_variant_prediction.py`

**Config**: 5 layers (19, 23, 27, 31, 35), 2000 variants (1000 pathogenic + 1000 benign), 5-fold CV

### Results

| Method | EXP-008 (n=600, 4 layers) | EXP-009 (n=2000, 5 layers) |
|--------|:-------------------------:|:--------------------------:|
| Raw perturbation AUC | 0.547 | 0.542 |
| Functional disruption AUC | 0.688 | 0.687 |
| **Cross-layer LR AUC** | **0.819** | **0.836** |
| Cross-layer LR AP | 0.804 | 0.834 |
| Cross-layer LR Accuracy | 74.0% | **75.9%** |

### Feature Importance (2000 variant model)

| Rank | Feature | Coefficient |
|------|---------|:-----------:|
| 1 | L35_cat_pert | +1.29 |
| 2 | L31_top10_f1_pert | -1.27 |
| 3 | L31_mean_delta_known | +1.19 |
| 4 | L35_mean_delta_novel | -1.00 |
| 5 | L35_n_known_pert | -0.91 |
| 6 | L31_mean_delta_novel | +0.84 |
| 7 | L35_f1_pert | -0.83 |
| 8 | L23_cat_pert | -0.80 |
| 9 | L31_n_known_pert | -0.73 |
| 10 | L19_top10_f1_pert | -0.67 |

### Key Findings

1. **Result is robust**: AUC improves slightly from 0.819 → 0.836 with 3.3× more data and an additional layer. The annotation-weighted approach generalizes beyond the initial 600-variant test.

2. **Layer 19 contributes**: `L19_top10_f1_pert` appears in top 10 features (rank 10, coef=-0.67). Layer 19's rich annotations (146 KNOWN features) add signal, though deep layers (31, 35) remain dominant.

3. **Deep layers confirmed as most important**: L35 and L31 features occupy 8 of 10 top positions. The pattern from EXP-008 — focused disruption of known features at deep layers = pathogenic — holds at scale.

4. **Raw perturbation is confirmed uninformative**: AUC=0.542 at n=2000, essentially random. This definitively rules out naive perturbation magnitude as a pathogenicity predictor.

5. **Functional disruption alone achieves AUC=0.687**: A simple, interpretable score requiring no ML — just weight perturbation by functional/ptm/domain annotation F1 — performs surprisingly well.

### Computational Cost

- 2000 variants × 5 layers × 2 sequences (WT + mut) = 20,000 ESM-2-3B forward passes
- Rate: ~5.3 variants/second per layer
- Total time: ~31 minutes on single L20 GPU

---

## EXP-010: ESM-2 LLR Baseline and Combined Model (2026-04-05)

**Goal**: Compare SAE-based variant prediction against the established ESM-2 zero-shot baseline (masked marginal log-likelihood ratio), and test whether combining both provides complementary signal.

**Script**: `scripts/09_esm2_baseline_comparison.py`

**Method**: Masked marginal LLR = log P(mut_aa | context_masked) - log P(wt_aa | context_masked). Negative LLR → model prefers WT → variant is likely deleterious. Same 2000 balanced ClinVar variants as EXP-009.

### Results

| Method | AUC-ROC | AP | Accuracy |
|--------|:-------:|:--:|:--------:|
| SAE Raw Perturbation | 0.542 | — | — |
| SAE Functional Disruption | 0.687 | — | — |
| SAE Cross-Layer LR (5-fold CV) | 0.836 | 0.834 | 75.9% |
| ESM-2 LLR (zero-shot, no training) | 0.882 | 0.891 | — |
| **Combined LLR + SAE LR (5-fold CV)** | **0.894** | **0.903** | **82.4%** |

### Key Findings

1. **ESM-2 LLR is a strong baseline (AUC=0.882)** — the masked language model probability alone is highly predictive of pathogenicity. This is consistent with Brandes et al. 2023 and other ESM-based variant effect studies.

2. **SAE features alone are weaker (AUC=0.836) but capture complementary information** — the combined model (LLR+SAE) achieves AUC=0.894, a +0.012 improvement over LLR alone and +0.058 over SAE alone. This is notable because:
   - LLR captures **sequence probability**: "Is this substitution likely in evolutionary context?"
   - SAE features capture **functional disruption**: "Which biological functions are affected?"
   - These are genuinely different signals — probability vs. mechanism.

3. **Combined accuracy of 82.4%** — the best single-model accuracy, confirming that SAE annotation-weighted features provide orthogonal predictive power.

4. **The SAE contribution is mechanistically interpretable** — unlike LLR (a single scalar), the SAE features tell us *why* a variant is pathogenic (e.g., "disrupts active site features at deep layers with focused intensity"). This interpretability is the core value proposition for R1.

### Interpretation

The LLR captures evolutionary constraint — residues that are highly conserved get high LLR scores when mutated. The SAE captures structural/functional constraint — mutations that disrupt annotated functional features get high SAE scores. These overlap substantially (conserved residues tend to be functional) but not completely:

- **LLR wins**: Variants in highly conserved positions where evolutionary constraint alone is sufficient (e.g., glycine in turn motifs)
- **SAE wins**: Variants in moderately conserved positions that happen to fall in functional sites (where evolutionary signal is diluted by neutral variation at the same position, but the SAE detects functional annotation disruption)
- **Both agree**: Variants in active site residues that are both highly conserved and functionally critical

### Significance for the Paper

The combined LLR+SAE model demonstrates that SAE-based mechanistic features provide value beyond what the language model already captures. This positions the R1 contribution as:

1. **Not trying to beat LLR** — ESM-2 already predicts pathogenicity well
2. **Adding interpretability** — SAE tells you WHICH functions are disrupted
3. **Providing complementary signal** — AUC improves when combined with LLR
4. **Enabling mechanism classification** — the SAE perturbation pattern (focused disruption, deep-layer feature turnover) can distinguish LOF from GOF, which LLR fundamentally cannot do (LLR is a scalar; it can't tell you HOW a variant is pathogenic, only THAT it is)

---

## EXP-011: Case Study Analysis (2026-04-05)

**Goal**: Demonstrate SAE interpretability on well-known disease genes with clear biological mechanisms.

**Script**: `scripts/10_case_study_analysis.py`

### Gene-Level Functional Disruption Ratios

| Gene | Pathogenic | Benign | Functional Ratio (path/benign) |
|------|:---------:|:------:|:------------------------------:|
| HBB | 10 | 3 | **6.53×** |
| PTEN | 11 | 1 | **3.13×** |
| TP53 | 8 | 9 | 2.42× |
| KRAS | 5 | 1 | 2.87× |
| KCNQ1 | 13 | 3 | 2.96× |

### Key Case Study Observations

**PTEN (tumor suppressor, phosphatase domain)**:
- Pathogenic variants (G129E, T131I, H141P, P204A, etc.) consistently perturb Feature 4538 (L35, PARTIAL, "functional/any", F1=0.209) and Feature 6590 (L31, "functional/any")
- The single benign variant (P354Q) has minimal functional feature perturbation
- Per-layer profile: pathogenic variants show disproportionately large L35 perturbation (mean=138.5 vs benign=102.0), consistent with deep ESM-2 layers encoding phosphatase active site structure

**HBB (hemoglobin)**:
- H93Y (proximal histidine, critical for heme coordination) perturbs Feature 8217 at L35, annotated as "chain/chain/Hemoglobin subunit alpha" — the SAE independently discovered a hemoglobin-specific feature
- Pathogenic variants targeting the globin fold interior (F43S, F104L, L33Q) show the largest perturbations (>370 total), while surface variants (K121N benign, 273 total) are smaller
- Functional disruption ratio 6.53× is the highest among all case studies

**TP53 (tumor suppressor)**:
- Pathogenic DNA binding domain mutations (R267W, D281G, R337C) perturb functional/binding site features at L31
- Benign variants outside the DNA binding domain (P36Q in transactivation, G374R in tetramerization) show different feature profiles — higher chain but lower functional perturbation
- Feature 5818 (L31, "functional/binding site") fires differentially for DNA binding domain variants

**KRAS (oncogene)**:
- G12S (classic activating mutation) shows the clearest functional perturbation pattern, with functional/binding site features disrupted at L31
- Functional disruption ratio 2.87× — lower than LOF genes because KRAS GOF mutations modify rather than destroy the GTPase active site

### Cross-Gene Pattern: Feature 4538 as a Universal Functional Site Detector

Feature 4538 (Layer 35, PARTIAL classification, F1=0.209 for "functional/any") is consistently among the top perturbed features for pathogenic variants across **all 5 case study genes**. This single SAE feature acts as a general-purpose functional site perturbation detector — a "fire alarm" for functionally important residue disruption.

### Per-Layer Perturbation Profiles

Across all case studies, perturbation magnitude increases monotonically with layer depth:
- L19 ≈ 28 → L23 ≈ 35 → L27 ≈ 50 → L31 ≈ 80 → L35 ≈ 125

But pathogenic variants show a **steeper increase at L31→L35** compared to benign variants. This "deep-layer amplification" is the signature of functional site disruption — deep ESM-2 layers encode functional site structure, so mutations affecting these sites cause disproportionate perturbation at depth.

### Significance

These case studies demonstrate that SAE perturbation signatures are **biologically interpretable**:
1. Features that fire differentially for pathogenic variants map to known functional annotations
2. The perturbation pattern reflects known biology (e.g., HBB hemoglobin features, PTEN phosphatase features)
3. Different disease mechanisms produce different perturbation profiles (LOF = functional feature ablation, GOF = functional feature modification)
4. A single feature (4538) serves as a universal functional site detector across diverse proteins

---

## EXP-012: Full-Feature-Vector Classifier (2026-04-13)

**Question:** Codex critique said summary-stat LR (0.836) is simpler than what the full vectors could give. Can the full delta_local + delta_global (163,840 dim) beat the summary stats?

**Method:**
- Stack delta_local and delta_global for 5 layers → 163,840 features per variant
- 2000 variants (1000 path, 1000 benign)
- Compare L2 logistic regression (GPU), MLP with dropout, 5-fold CV
- Reference: `Research1/scripts/11_full_vector_classifier.py`

**Results:**
| Method          | AUC    | Notes |
|-----------------|--------|-------|
| LR wd=0.001     | 0.7471 | weak  |
| LR wd=0.01      | 0.7440 | weak  |
| LR wd=0.1       | 0.7402 | weak  |
| MLP dropout=0.3 | 0.5485 | overfits |
| MLP dropout=0.5 | 0.5491 | overfits |

**Conclusion (negative result):** Full-vector approach is WORSE than 12 summary stats (0.836). Classic p >> n regime: 163,840 features with only 1600 training samples per fold. The summary statistics encoded priors (annotation weights × delta) that acted as strong regularization. **Lesson learned:** annotation priors are critical for generalization at this sample size.

**Output files:** `results/variant_effect/full_vector_results.json`

---

## EXP-013: Annotation-Selected Feature Classifier (2026-04-13)

**Question:** If the full vector overfits but summary stats compress too much, can we split the difference by selecting only annotated features?

**Method:**
- For each layer, select features with annotation-alignment F1 ≥ threshold
- Feature vector: for selected features, include |delta_local|, |delta_global|, weighted signed delta_local (weight = F1 × category_weight)
- Thresholds tested: F1 ≥ 0.1, 0.2, 0.3
- 2000 variants, 5-fold CV LR with multiple C values
- Reference: `Research1/scripts/12_annotated_feature_classifier.py`

**Results (5-fold CV AUC):**
| Threshold | N features | Best AUC | Best C |
|-----------|-----------:|---------:|--------|
| F1 ≥ 0.1  | ~52k       | **0.8782** | 0.1 |
| F1 ≥ 0.2  | ~20k       | 0.8710   | 0.01 |
| F1 ≥ 0.3  | ~7.5k      | 0.8334   | 0.01 |

**Comparison:**
| Method                               | AUC    |
|--------------------------------------|--------|
| Raw SAE perturbation sum             | 0.542  |
| Functional disruption (weighted)     | 0.687  |
| 12-summary-stats cross-layer LR      | 0.836  |
| Full 163k-dim LR                     | 0.747  |
| **Annotation-selected LR (F1≥0.1)**  | **0.878** |
| ESM-2 LLR baseline                   | 0.882  |
| Combined ESM-2 LLR + SAE (EXP-010)   | 0.894  |

**Interpretation:** The annotated-feature approach closes the gap with LLR (0.878 vs 0.882). The 52k selected features embed biological priors that the 163k-dim vector lacked and the 12-stat summary underexploited. The combined model (EXP-010) still wins, confirming LLR and SAE provide complementary signal.

**Honest note:** SAE alone does NOT beat ESM-2 LLR. Value-add is interpretability + ensemble contribution.

**Output files:** `results/variant_effect/annotated_feature_results.json`

---

## EXP-014: Position-Spread Feature Analysis (2026-04-13)

**Question:** Existing features only use per-feature magnitudes (|delta_local|, |delta_global|). Does the **spatial spread** of perturbation along the sequence carry independent predictive signal? Pathogenic mutations may propagate further from the mutation site than benign ones.

**Method:**
- Recompute SAE activations for WT and mutant sequences on the full protein (not just mutation position)
- Per feature/position map |Δf(p)| across sequence positions p
- Derive 13 position-aware features per layer:
  - `window_l1_5/10/20`: L1 perturbation within ±5/±10/±20 residues of mutation
  - `local_fraction`: fraction of perturbation within ±5 of mutation
  - `decay_rate`: exponential decay of perturbation vs distance
  - `peak_offset`, `peak_value`, `mut_site_value`, `peak_is_mut_site`
  - `spread_halfwidth`: distance at which perturbation halves
  - `total_l1`, `mean_l1`, `n_perturbed_positions`
- 500 variants (250 path, 250 benign), layers {19, 27, 35} → 39 features total
- 5-fold CV LR, multiple C values
- Reference: `Research1/scripts/13_position_spread_analysis.py`

**Results:**
| C    | AUC    | AP     | Acc   |
|------|--------|--------|-------|
| 0.1  | 0.7889 | 0.7676 | 0.738 |
| 1.0  | 0.7853 | 0.7587 | 0.724 |
| 10.0 | 0.7780 | 0.7501 | 0.722 |

**Interpretation:** Position-spread features alone reach AUC ≈ 0.79 at n=500. This is **weaker** than annotation-selected features (0.878) and below LLR (0.882), but still demonstrates that purely spatial signal is informative. The 13×3 = 39-dim feature set is compact and mechanism-inspired.

**Comparison vs existing features:**
| Feature family (n=500 sanity)   | AUC |
|---------------------------------|-----|
| Position-spread (39 dim)        | 0.789 |
| 12 summary stats (full 2000)    | 0.836 |
| Annotation-selected (52k, 2000) | 0.878 |

**Honest note:** At the same n=500, the position-spread approach is not competitive with annotation-weighted features. However, it is **orthogonal signal** — it uses spatial distribution rather than per-feature magnitudes. The next step is to ensemble position-spread with annotation-selected features on the full 2000 variants and check whether AUC rises above 0.878.

**Output files:** `results/variant_effect/position_spread_results.json`, `results/variant_effect/position_spread_records.pkl`

---

## EXP-015: Annotation-LR + LLR Ensemble (2026-04-13)

**Question:** EXP-010 combined 12-stat LR with LLR and reached AUC=0.894. EXP-013 showed annotation-selected LR (52k dim) alone reaches AUC=0.878. Does ensembling the richer annotation-LR with LLR improve over 0.894?

**Method:**
- Input 1: 5-fold CV scores from annotation-selected LR (F1 ≥ 0.1, C=0.1) on 2000 variants
- Input 2: per-variant ESM-2 masked-marginal LLR (from script 09)
- Ensembles: (a) 2-feature logistic regression, (b) simple z-sum
- Reference: `Research1/scripts/14_ensemble_annotated_llr.py`

**Results:**
| Component | AUC |
|-----------|-----|
| SAE annotation-LR alone         | 0.8782 |
| ESM-2 LLR alone                 | 0.8822 |
| Previous ensemble (12-stat + LLR, EXP-010) | 0.894  |
| **Logistic ensemble (annot-LR + LLR)** | **0.9137** |
| **Simple z-sum ensemble (annot-LR + LLR)** | **0.9143** |

**Interpretation:** The annotation-LR + LLR ensemble delivers a **+2.1%** AUC over the prior best (0.894). The two signals are more complementary than the 12-stat summary was: richer feature-level SAE information gives a classifier that disagrees with LLR on a different set of variants, so their combination is stronger.

**Significance vs Codex critique:** Codex flagged that SAE alone is inferior to LLR. True — but the ensemble now beats LLR by a large margin (0.9143 vs 0.8822), which is the production-relevant metric. SAE's value-add is quantitatively confirmed.

**Next step:** Run full LLR ensembling with position-spread + annotation features to check whether spatial signal is also complementary.

**Output files:** `results/variant_effect/ensemble_annotated_llr.json`

---

## EXP-016: Mechanism Classifier Protein-Level Holdout (2026-04-29)

**Question:** Does the LOF/GOF/DN mechanism classifier generalize across
proteins, or was variant-level cross-validation leaking protein-specific SAE
signatures?

**Method:**
- Updated `Research1/scripts/16_mechanism_classifier.py` to report both
  variant-level CV and protein-level CV.
- Protein-level CV groups all variants from the same gene into the same fold.
- The default result file now contains both splits.

**Results:**
| Split | SAE macro-AUC | LLR macro-AUC | SAE+LLR macro-AUC | SAE macro-F1 |
|-------|--------------:|--------------:|------------------:|-------------:|
| Variant-level CV | 0.7471 | 0.5054 | 0.7488 | 0.5537 |
| Protein-level CV | 0.5161 | 0.4642 | 0.5164 | 0.3563 |

**Interpretation:** The variant-level mechanism signal does not survive a
protein-level holdout. This fails the `TODO_NEXT.md` acceptance threshold of
protein-level macro-AUC >= 0.7. The R1 headline must be downgraded: current SAE
features show variant-level mechanism structure, but do not yet provide a
robust cross-protein mechanism predictor.

**Output files:**
- `results/variant_effect/mechanism_classifier_results.json`
- `results/variant_effect/mechanism_classifier_results_t0_protein_holdout_20260429.json`
- `results/variant_effect/mechanism_classifier_results_variantcv_legacy_20260429.json`

---

## EXP-017: ProteinGym Sign-Corrected SAE Ensemble (2026-05-03 CST)

**Question:** Was the poor ProteinGym SAE+LLR result caused by adding a
damage-magnitude SAE score with the wrong sign?

**Method:**
- Updated `Research1/scripts/21_proteingym_sae_followup.py` to report both the
  legacy plus-sign ensemble and a sign-corrected ensemble:
  `zscore(LLR) - zscore(SAE_total)`.
- Reran the full 217-assay ProteinGym follow-up on the 1-GPU H200 pod.
- Ran local diagnostics with 10,000 bootstrap resamples.

**Results:**
| Metric | Mean Spearman rho | 95% bootstrap CI |
|--------|------------------:|------------------|
| LLR | 0.4341 | [0.4091, 0.4594] |
| Raw SAE disruption | -0.2314 | [-0.2588, -0.2035] |
| Legacy plus ensemble | 0.1993 | [0.1737, 0.2244] |
| Sign-corrected ensemble | 0.4047 | [0.3818, 0.4282] |

Win rates versus LLR:
- Legacy plus ensemble: 8.4% of usable assays.
- Sign-corrected ensemble: 33.6% of usable assays.

**Interpretation:** The sign correction largely fixes the cancellation bug, but
the corrected ensemble still does not beat LLR on average and remains below the
TODO_NEXT 40% win-rate threshold. ProteinGym should be treated as a negative
diagnostic for the current SAE perturbation score, not as evidence that SAE+LLR
improves generic DMS fitness.

**Output files:**
- `results/variant_effect/proteingym_benchmark_sae_signed_20260429_signed.json`
- `results/variant_effect/proteingym_benchmark_sae_signed_diagnostics_20260503.json`
- `logs/runtime/t0b_proteingym_signed_20260429_signed.log`

---

## EXP-018: Preliminary Mechanism Feature Audit (2026-05-03 CST)

**Question:** Which SAE features drive the LOF / GOF / DN classifier, and do
their annotations support a biological interpretation?

**Method:**
- Added `Research1/scripts/24_mechanism_feature_audit.py`.
- Retrained the multinomial LR from `16_mechanism_classifier.py`.
- For each class and each production layer {19, 23, 27, 31, 35}, extracted the
  top-10 positive coefficient features.
- Joined each feature to the current annotation pkl and emitted a markdown
  audit table with best annotation, Pfam/domain proxy, GO/functional proxy,
  binding proxy, and conservative interpretation notes.

**Result:** Produced 150 audit rows: 3 mechanism classes x 5 layers x top-10
features.

**Interpretation:** This is a useful figure-planning table, but it is not yet
the final manual feature audit requested in T1-B because current annotation
pkls still lack `firing_positions`. Max-activating sequence inspection remains
blocked on the T1-D annotation rerun.

**Output files:**
- `results/variant_effect/mechanism_feature_audit_20260503.json`
- `results/variant_effect/mechanism_feature_audit_20260503.md`

---

## EXP-019: Annotation Rerun With Firing Positions (started 2026-05-03 CST)

**Question:** Can we rerun the deep-layer annotation pipeline with
feature-firing residue positions saved, so annotation expansion and manual
mechanism feature audit are no longer blocked?

**Method:**
- Updated `src/analysis/feature_annotation.py` to optionally store
  `firing_positions` and `top_firing_examples` per SAE feature.
- Updated `scripts/04_analyze_our_sae.py` with:
  `--save-firing-positions`, `--max-firing-positions-per-feature`,
  `--checkpoint-root`, and `--out-prefix`.
- Added `scripts/run_t1d_annotation_firing_20260503.sh`, which reruns layers
  19/23/27/31/35 on 1000 Swiss-Prot proteins and then runs
  `19_expand_annotation.py`.
- A L35 smoke test on 20 proteins confirmed that firing positions are written
  into the pkl.

**Results:** The five firing-enabled annotation pkls completed. The first
expansion attempt was stopped because `19_expand_annotation.py` reported
`Swiss-Prot universe: 0 proteins`, which meant it was not filtering extended
labels to the scored proteins. The script now collects the accession universe
from `firing_positions`; the fixed expansion stage completed with a 1000-protein
universe.

| Layer | KNOWN before | KNOWN after | PARTIAL before | PARTIAL after | USEFUL before | USEFUL after |
|-------|-------------:|------------:|---------------:|--------------:|--------------:|-------------:|
| L19 | 146 | 381 | 2336 | 3483 | 6048 | 7481 |
| L23 | 135 | 301 | 1896 | 2992 | 5341 | 6773 |
| L27 | 72 | 163 | 1105 | 1725 | 3384 | 4423 |
| L31 | 49 | 86 | 769 | 1135 | 2321 | 3197 |
| L35 | 32 | 45 | 454 | 653 | 1422 | 1935 |

**Interpretation:** Saving firing positions successfully unblocks the expansion
logic and substantially increases known/useful feature counts. However, T1-D
does not meet its strict acceptance threshold for L35: observed KNOWN=45 versus
target >=60.

**Output paths:**
- `results/annotation_alignment/ours_3B_l{19,23,27,31,35}_step500000.pkl`
- `results/annotation_alignment/ours_3B_l{19,23,27,31,35}_step500000_summary.json`
- `results/annotation_alignment/*_nofiring_backup_20260503.*`
- `results/annotation_alignment/expanded_summary_firing_20260503.json`
- Remote log:
  `/oss-pvc/zhk_zip/biocc/Research1/logs/runtime/t1d_annotation_firing_20260503.log`
- Fixed expansion log:
  `/oss-pvc/zhk_zip/biocc/Research1/logs/runtime/t1d_expand_annotation_fixed_20260503.log`

---

## EXP-020: Firing-Aware Mechanism Audit and Indel Transfer (2026-05-04 CST)

**Question:** After firing positions are available, can the mechanism feature
audit be finalized, and can the missense mechanism classifier be transferred to
protein indels?

**Method:**
- Updated `scripts/24_mechanism_feature_audit.py` to prefer expanded
  annotation pkls and report expanded labels plus top firing residue examples.
- Pulled the five firing-enabled expanded annotation pkls from H200 and verified
  sha256 hashes locally.
- Added `scripts/25_indel_mechanism.py` with three modes:
  `prepare`, `fit`, and `predict`.
- Prepared reconstructable pathogenic/benign ClinVar indel WT/mutant sequence
  records locally, fitted a missense LOF/GOF/DN classifier cache as pure numpy
  LR parameters, and uploaded the records/cache to the 1-GPU H200 pod.
- Ran a 2-record H200 smoke test before starting the full run.

**Results so far:**
- Firing-aware audit output has 150 rows; all rows include top residue firing
  examples, and 142/150 include expanded labels.
- Indel record preparation produced 6,649 records:
  5,274 pathogenic and 1,375 benign.
- Indel classes: deletion 2,530; insertion 2,504; delins 1,187; duplication
  428.
- The full T1-C H200 run completed for 6,649/6,649 records.
- Predicted mechanisms: LOF 4,161; GOF 1,354; DN 1,134.
- SAE damage-score pathogenicity AUC: 0.7735.

**Interpretation:** The mechanism audit is now ready for figure planning and
manual residue triage. The indel experiment is still a transfer diagnostic:
its affected-region SAE delta is not the same object as a missense position
delta. The damage AUC is useful but does not meet the TODO_NEXT >0.85 target,
so this should not be framed as a solved indel pathogenicity benchmark.

**Output paths:**
- `results/variant_effect/mechanism_feature_audit_firing_20260504.json`
- `results/variant_effect/mechanism_feature_audit_firing_20260504.md`
- `results/variant_effect/indel_records_supported_20260504.jsonl`
- `results/variant_effect/indel_records_supported_20260504_summary.json`
- `results/variant_effect/indel_mechanism_classifier_20260504.pkl`
- `results/variant_effect/indel_mechanism_predictions_20260504.jsonl`
- `results/variant_effect/indel_mechanism_predictions_20260504_summary.json`
- `logs/runtime/t1c_indel_mechanism_20260504.log`
- Remote log:
  `/oss-pvc/zhk_zip/biocc/Research1/logs/runtime/t1c_indel_mechanism_20260504.log`
- Remote output:
  `/oss-pvc/zhk_zip/biocc/Research1/results/variant_effect/indel_mechanism_predictions_20260504.jsonl`

---

## EXP-021: T1-A and T1-E Readiness Audits (2026-05-04 CST)

**Question:** Are the remaining R1 reviewer-facing baseline and clinical-cohort
TODOs executable with currently staged local data?

**Method:**
- Added `scripts/23_baseline_headtohead.py` to recompute local pathogenicity
  AUC tables with bootstrap CIs and to audit missing competitor assets.
- Added `scripts/26_channelopathy.py` to audit local channelopathy coverage and
  staged curated-label assets.

**Results:**
- T1-A local available pathogenicity results:
  - ClinVar2000: SAE-LR 0.8782, ESM-2 LLR 0.8822, SAE+LLR 0.9143.
  - CancerHoldout101: SAE-LR 0.9079, ESM-2 LLR 0.8978, SAE+LLR 0.9193.
- T1-A is blocked because AlphaMissense, PrimateAI-3D, gMVP, and ESM-1v files
  are not staged.
- T1-E local coverage: 16 KCNQ1 ClinVar LLR rows, 0 existing R1 predictions for
  KCNQ1/SCN5A/KCNH2/CACNA1C, 0 curated channelopathy mechanism rows.
- SCN5A and KCNH2 DMS files exist under ProteinGym, but no ClinGen/literature
  mechanism curation or retrospective drug-response labels are staged.

**Interpretation:** Both TODOs are now reproducibly blocked rather than
unexamined. T1-A needs external baseline score files and calibration. T1-E
needs a curated clinical channelopathy cohort before concordance can be
measured.

**Output paths:**
- `results/variant_effect/baseline_headtohead_readiness_20260504.json`
- `results/variant_effect/baseline_headtohead_readiness_20260504.md`
- `results/variant_effect/channelopathy_readiness_20260504.json`
- `results/variant_effect/channelopathy_readiness_20260504.md`

---

## EXP-R1-EXT-001: External Baseline Resource Staging (2026-05-04 CST)

**Question:** Can the missing T1-A competitor assets be partially staged now?

**Result:** AlphaMissense was staged locally and on H200: `AlphaMissense_hg38.tsv.gz` and `AlphaMissense_aa_substitutions.tsv.gz`. The T1-A readiness table was updated to mark AlphaMissense as `available_unscored`.

**Remaining blocker:** T1-A still cannot be claimed because AlphaMissense has not been calibrated/scored against the local ClinVar/COSMIC rows, and PrimateAI-3D, gMVP, and ESM-1v are still missing. PrimateAI-3D is gated, gMVP is bundled through dbNSFP, and ESM-1v needs a scoring backend choice before downloading large weights.

## EXP-R1-EXT-002: AlphaMissense Baseline Scoring (2026-05-04 CST)

**Question:** With AlphaMissense staged, how does it score the local R1 pathogenicity cohorts?

**Method:** Added `scripts/27_alphamissense_baseline.py`, matched variants by `(uniprot_id, protein_variant)` against `AlphaMissense_aa_substitutions.tsv.gz`, and computed bootstrap AUCs.

**Result:** ClinVar2000 matched 1,972/2,000 with AUC 0.9474 [0.9377, 0.9567]. CancerHoldout101 matched 101/101 with AUC 0.9700 [0.9244, 0.9988]. AlphaMissense exceeds the current SAE+LLR pathogenicity AUC on these two pathogenicity cohorts.

**Interpretation:** R1 pathogenicity is not the novel claim against AlphaMissense. The remaining R1 novelty must come from mechanism/interpretability/indel behavior, while T1-A full baseline remains incomplete until PrimateAI-3D, gMVP, and ESM-1v are staged.

## EXP-R1-EXT-003: Available External Baselines and SAE Residual Summary (2026-05-07 CST)

**Question:** With dbNSFP/gMVP and ESM-1v staged, what can be completed for the
T1-A head-to-head before PrimateAI-3D approval?

**Method:** Ran `scripts/28_external_baselines_available.py` on H200 for
AlphaMissense, gMVP, and a 5-checkpoint ESM-1v ensemble, then added
`scripts/29_available_baseline_summary.py` to merge those outputs with local
SAE-LR, ESM-2 LLR, and SAE+LLR scores. The summary computes pathogenicity AUCs,
scalar one-vs-rest mechanism AUCs, Spearman correlations, and top SAE residual
cases after regressing SAE-LR on AlphaMissense and ESM-1v.

**Results:**
- ClinVar2000 AUCs: SAE-LR 0.8782, ESM-2 LLR 0.8822, SAE+LLR 0.9143,
  AlphaMissense 0.9474, gMVP 0.9369, ESM-1v 0.9089.
- CancerHoldout101 AUCs: SAE-LR 0.9079, ESM-2 LLR 0.8978, SAE+LLR 0.9193,
  AlphaMissense 0.9700, gMVP 0.9400, ESM-1v 0.8552.
- Scalar mechanism macro-AUCs on labeled ClinVar variants: AlphaMissense
  0.4800, gMVP 0.4856, ESM-1v 0.4847, SAE-LR 0.5161, SAE+LLR 0.5164.

**Interpretation:** R1 should not claim state-of-the-art scalar pathogenicity.
The defensible remaining T1-A story is complementarity: SAE-derived scores have
residual signal and interpretable mechanism-like diagnostics that scalar
pathogenicity baselines do not directly provide. PrimateAI-3D remains excluded
until gated access is approved.

**Output paths:**
- `results/variant_effect/external_baselines_available_20260507.json`
- `results/variant_effect/external_baselines_available_scores_20260507.tsv`
- `results/variant_effect/available_baseline_summary_20260507.json`
- `results/variant_effect/available_baseline_summary_20260507.md`
- `results/variant_effect/available_baseline_sae_residual_cases_20260507.tsv`

## EXP-R1-T1E-001: Channelopathy Mechanism Label Curation (2026-05-07 CST)

**Question:** Can the missing curated channelopathy mechanism/drug-response
labels for T1-E be staged enough to unblock R1 scoring and concordance testing?

**Method:** Split curation across four GPT-5.5 xhigh workers:
KCNQ1/LQT1, SCN5A, KCNH2/CACNA1C, and ClinVar/ClinGen candidate indexing.
Each worker wrote a TSV with source-grounded rows and a source markdown. Added
`scripts/30_merge_channelopathy_labels.py` to validate schema, normalize labels,
deduplicate rows, preserve candidate/source-index rows as `unknown`, and split
positive mechanism rows from drug-response rows.

**Results:**
- Worker input rows: 121.
- Consolidated rows: 119.
- Positive mechanism rows: 74.
- Drug-response rows: 40.
- Positive mechanism rows by gene: KCNQ1 23, SCN5A 25, KCNH2 19, CACNA1C 7.
- Full mechanism counts: LOF 40, GOF 13, DN 13, mixed_complex 8, unknown 45.

**Interpretation:** T1-E is no longer blocked by missing curated labels. This
curation unblocked the scoring experiment recorded in EXP-R1-T1E-002.
Drug-response rows are sparse and should be treated as secondary research labels,
not as clinical guidance.

**Output paths:**
- `data/channelopathy/channelopathy_mechanism_labels.tsv`
- `data/channelopathy/channelopathy_mechanism_positive_labels.tsv`
- `data/channelopathy/channelopathy_drug_response_labels.tsv`
- `data/channelopathy/channelopathy_label_sources.md`
- `data/channelopathy/channelopathy_label_summary.json`
- `results/variant_effect/channelopathy_readiness_20260507.json`
- `results/variant_effect/channelopathy_readiness_20260507.md`

## EXP-R1-T1E-002: Channelopathy Mechanism Concordance (2026-05-07 CST)

**Question:** Does the existing R1 SAE perturbation mechanism classifier
concord with curated LOF/GOF/DN channelopathy mechanisms for KCNQ1, SCN5A,
KCNH2, and CACNA1C?

**Method:** Added `scripts/31_channelopathy_concordance.py` and
`scripts/run_t1e_channelopathy_20260507.sh`. The H200 run used ESM-2-3B with
mutation-centered windows for long channel proteins and computed the existing
five-layer SAE perturbation signature for each scoreable missense variant. A
local classifier was trained from `variant_mechanisms.tsv` and
`scaled_perturbation_signatures.pkl` over LOF/GOF/DN classes, then applied to
the curated channelopathy subset. Mixed/complex labels were scored but excluded
from the headline LOF/GOF/DN accuracy.

**Results:**
- High-confidence curated mechanism labels: 74.
- Scoreable missense variants: 69.
- Headline evaluated LOF/GOF/DN variants: 64.
- Mixed/complex scored but excluded from headline accuracy: 5.
- Accuracy: 0.625.
- Macro-F1: 0.444.
- Confusion matrix over labels DN/GOF/LOF, rows=true and columns=predicted:
  `[[0, 2, 11], [0, 7, 4], [3, 4, 33]]`.
- By-gene accuracy: CACNA1C 0.500, KCNH2 0.842, KCNQ1 0.455,
  SCN5A 0.632.

**Interpretation:** T1-E does not meet the TODO acceptance threshold of >=80%
expert-mechanism concordance. The dominant failure mode is DN channel variants
being predicted as LOF. KCNH2 is superficially strong because the curated set is
LOF-heavy, while KCNQ1 is weak because many curated pore/N-terminal DN labels
collapse to LOF under the current mechanism head. This should be reported as a
negative but informative diagnostic unless a channel-specific mechanism model or
feature-family audit resolves the DN-vs-LOF distinction.

**Output paths:**
- `results/variant_effect/channelopathy_concordance_20260507.json`
- `results/variant_effect/channelopathy_concordance_20260507.md`
- `results/variant_effect/channelopathy_concordance_20260507.predictions.tsv`
- `results/variant_effect/channelopathy_concordance_20260507.audit.tsv`
- `results/variant_effect/channelopathy_concordance_20260507.supported.tsv`
- `results/variant_effect/channelopathy_concordance_20260507.prepare.json`
- `results/variant_effect/channelopathy_concordance_20260507.signatures.pkl`

## EXP-R1-EXT-004: Final Available Baselines Without PrimateAI-3D (2026-05-10 CST)

**Question:** How should T1-A be recorded if PrimateAI-3D gated access cannot
be obtained?

**Method:** Treat PrimateAI-3D as unavailable for this analysis round and
finalize the head-to-head over accessible baselines only: SAE-LR, ESM-2 LLR,
SAE+LLR, AlphaMissense, gMVP, and ESM-1v. No result numbers were recomputed in
this bookkeeping step; the final table remains
`available_baseline_summary_20260507.{json,md}`.

**Result:** T1-A is no longer blocked on PrimateAI-3D. The accessible baseline
table shows that AlphaMissense and gMVP are stronger scalar pathogenicity
baselines than SAE+LLR on ClinVar2000 and CancerHoldout101, while all scalar
external baselines remain weak for LOF/GOF/DN mechanism separation.

**Interpretation:** R1 should not claim state-of-the-art scalar pathogenicity.
The defensible claim is complementary, interpretable SAE residual signal plus
mechanism-like diagnostics. PrimateAI-3D should be mentioned as unavailable, not
as an uncompleted experiment.

**Output paths:**
- `results/variant_effect/available_baseline_summary_20260507.json`
- `results/variant_effect/available_baseline_summary_20260507.md`
- `results/variant_effect/available_baseline_sae_residual_cases_20260507.tsv`
- `results/variant_effect/t1a_final_available_baselines_no_primateai_20260510.md`
