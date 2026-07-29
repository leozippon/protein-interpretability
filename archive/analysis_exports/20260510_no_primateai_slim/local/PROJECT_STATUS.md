# Project Status — 2026-05-10

Two independent research projects aimed at Nature/Science/Cell.

## Executive Summary

**R1 (Variant Mechanism Prediction via SAE):** The strongest defensible result
is pathogenicity complementarity: annotation-selected SAE features + ESM-2 LLR
reach **AUC 0.9143** on 2,000 ClinVar variants and **AUC 0.9193** on the
101-variant cancer holdout. The LOF/GOF/DN mechanism classifier does **not**
generalize under protein-level CV (SAE macro-AUC **0.5161**), so the mechanism
claim must be narrowed to variant-level structure and feature interpretation.
ProteinGym is a negative diagnostic, and the indel transfer experiment is useful
but below threshold (damage AUC **0.7735**, target >0.85). After external
staging, AlphaMissense, gMVP, and ESM-1v are scored. PrimateAI-3D is treated as
unavailable after gated access could not be obtained, so the T1-A baseline table
is final over the accessible baselines. AlphaMissense and gMVP are stronger on
raw pathogenicity (ClinVar2000 AUC **0.9474** and **0.9369**), so R1's
reviewer-facing novelty cannot be framed as beating scalar pathogenicity
specialists. The remaining useful angle is SAE-specific residual interpretation
and mechanism-like diagnostics.

**R2 (Interpretable Drug Design via CLT):** v2 CLTs, hook diagnostics,
direct-effect feature selection, on-manifold steering, structural QC, ablation,
and cross-model conservation have all been run. The current steering evidence is
negative: direct-effect/on-manifold steering gives **0/8 significant positive**
EC classes. T2-C generated-sequence checks now include Pfam, CLEAN, and
Foldseek. The selected steered leads pass all three checks, but the
generation-wide steered-vs-unsteered lift remains weak: Pfam 0.860 vs 0.820
(Fisher p=0.170), CLEAN exact 0.775 vs 0.775. The real-vs-random lysozyme
calibration is complete and validates the metric stack itself: all reported
Pfam/CLEAN/ESMFold/Foldseek metrics separate real lysozymes from random UniRef50
controls with Cohen's d > 1. Current decision: no-go for a strong
steering/drug-design claim unless broader EC-class generation results change
this conclusion.

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

These counts are after the firing-position rerun and expansion. L35 improved
but still missed the TODO_NEXT target of >=60 KNOWN features.

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

**Key lesson (addressing Codex critique):** Naive use of the full 163k-dim
delta vectors *overfits* with only 2000 samples (AUC 0.747, worse than 12
summary stats). Annotation-guided feature selection (F1 ≥ 0.1) recovers
AUC 0.878 using ~52k features, essentially matching LLR alone. Combined
with LLR via logistic stacking, reaches AUC 0.9137; via simple z-sum,
**AUC 0.9143** (EXP-015) — a +3.2 point improvement over LLR alone.

### What R1 provides beyond LLR
- Mechanism-aware delta signatures (per-category perturbation)
- Feature-level interpretation (which functional site is disrupted)
- Case studies (TP53, KRAS, PTEN, KCNQ1, HBB): each variant linked to
  specific feature disruptions

### Honest Weaknesses
1. SAE alone does **not** beat LLR on raw AUC (0.878 vs 0.882) — but ensemble
   does: 0.9143 vs 0.8822 alone (EXP-015)
2. Protein-level mechanism CV fails: SAE macro-AUC 0.5161
3. ProteinGym SAE+LLR does not beat LLR on average; signed ensemble win-rate is
   33.6%, below the 40% threshold
4. T1-A competitor baselines are finalized over accessible methods:
   AlphaMissense, gMVP, and ESM-1v are available; PrimateAI-3D is excluded
   because the gated dataset could not be obtained
5. T1-E channelopathy scoring is complete, but the current classifier misses
   the >=80% concordance target: 64 LOF/GOF/DN rows, accuracy 0.625,
   macro-F1 0.444, with DN variants mostly collapsing to LOF

### Next Improvements
- Do not wait on PrimateAI-3D for the current analysis pass; use the accessible
  baseline table as the final T1-A comparison
- Diagnose whether the channelopathy DN->LOF collapse can be improved by a
  channel-specific mechanism head or feature family audit before making any
  clinical-mechanism claim
- Treat indel and ProteinGym results as diagnostics unless stronger scoring is
  validated

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
Previous memory: "L27-35 = enzyme-specific output, L33-35 has 14× more
discriminating EC features." Actual: **L35 CLT is 90% dead** — the
discrimination signal lives in raw activations the CLT cannot capture.

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
2. **High dead rates** mean steering interventions should be constrained
   to a small set of usable features
3. **Steering experiments are negative so far** — 0/8 significant positive EC
   classes under direct-effect/on-manifold steering
4. **Real EC metrics are executed for lysozyme** — Pfam, CLEAN, and Foldseek
   scanned the generated lysozyme sequences, and the real-vs-random calibration
   passed for the metric stack
5. **Drug design claim not supported** — current evidence supports a prototype
   pipeline, not validated therapeutic sequence design

### Next Improvements
- Extend the calibrated T2-C metric stack to additional EC classes before any
  renewed steering claim
- If R2 remains a steering project, train a stronger CLT only after T2-C
  confirms that the current metric stack is usable
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
- [x] Stage curated channelopathy labels for T1-E
- [x] Score T1-E channelopathy cohort: completed, target not met

### R2
- [x] Hook sanity and ec_features provenance diagnostics
- [x] Direct-effect feature selection
- [x] TopK-aware on-manifold steering benchmark
- [x] R2 viability decision gate: current steering claim no-go
- [x] Finish T2-C lysozyme calibration run and pull summary outputs

## Recent Experiments (2026-05-07)

See `TODO_RESULTS.md`, `Research1/docs/EXPERIMENT_LOG.md`, and
`Research2/docs/EXPERIMENT_LOG.md` for the completed TODO_NEXT pass:
- T0-A/B/C/D diagnostics and downgrades
- T1-A readiness blocker, T1-B firing-aware audit, T1-C indel transfer,
  T1-D annotation rerun, T1-E label curation plus concordance failure analysis
- T2-A/B direct-effect/on-manifold steering, T2-C generated metric triad,
  T2-C calibration runner, T2-D no-go decision
