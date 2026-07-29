# Encoder/Decoder Benchmark Split Review (2026-05-19)

The user proposed:

1. Submit R2 standalone to Nat MI.
2. Reframe R1 as an **Encoder-Only Benchmark** (built on ESM-2 SAE work).
3. Use R2's experience to develop a **Decoder-Only Benchmark** afterward.
4. Build a shared framework underneath both benchmarks.

This is a stronger plan than GPT's task-based split (Variant vs Generative).
It is architectural, which is cleaner methodologically. **Confirm direction**,
with five specific refinements below.

---

## 1. Why Architectural Split is Better Than Task Split

GPT's split was Variant vs Generative. The user's split is Encoder vs Decoder.
The architectural split is the right call because:

- **Interpretability methods are architecture-specific.** Gradient
  attribution differs between MLM and CLM. Activation patching has
  different semantics. SAE vs CLT live in different mathematical regimes.
- **The intervention pathways differ.** Encoder models mask-and-rescore;
  decoder models perturb-and-continue. Methods that work in one don't
  port to the other.
- **The audiences differ.** Encoder-side audience is variant-
  interpretation / clinical genomics. Decoder-side audience is protein
  design / generative biology. Two distinct reader communities.
- **The reusable scaffolding is architecture-specific.** Encoder
  benchmarks need: variant cohorts, residue-level labels, masking
  interfaces. Decoder benchmarks need: generation cohorts, sequence-
  level metrics, attention-trace interfaces.

Task-based splitting forces the same methods into both halves and
muddies the analysis. Architectural splitting keeps each benchmark
internally consistent.

---

## 2. The Three Papers (Confirmed)

| # | Paper | Source | Venue | Timeline |
|---|---|---|---|---|
| **A** | R2 standalone | Existing R2 evidence | Nat MI (35–45 %) → Cell Patterns fallback | **~2 weeks** |
| **B** | Encoder-Only Benchmark | R1 + new method comparisons | Nat MI / Nat Methods stretch | **~6–8 weeks** |
| **C** | Decoder-Only Benchmark | R2 lessons + new evaluation | Nat MI / Cell Patterns | **~6–8 weeks after B** |

Plus a shared framework deliverable (Section 5).

Realistic total project timeline: **4 months end-to-end**.

---

## 3. Refinements to the User's Plan

### Refinement 1: Sharpen the R2-standalone vs Paper C boundary

If R2 is published standalone AND used as a case study for Paper C, the
content must split cleanly to avoid self-plagiarism:

**Paper A (R2 standalone) — keeps only:**
- 38 cross-model triplet discovery + 30× null
- N-terminal attention-sink subset (T011/T018/T023)
- Calibrated negative steering as a Limitations section
- Brief mention of CLT quality diagnostic

**Paper C (Decoder Benchmark) — uses R2 as motivation, adds:**
- Multiple decoder models beyond the three
- Method comparison (CLT vs attention-output transcoder vs raw probes)
- Evaluation framework applied to all decoders
- The full causal-ablation negative-result analysis as benchmark
  baseline characterizations

If we don't draw this line now, the two papers will conflict at
submission review. **Mark the split decision before submitting A.**

### Refinement 2: IndelMissense v1 belongs in Paper B (Encoder Benchmark), not R1 standalone

The previous plan had IndelMissense v1 as part of an R1 resource paper.
Under the new architecture, the cleanest home for IndelMissense is as a
**centerpiece dataset of the Encoder-Only Benchmark**. It is encoder-
specific (ESM-2 perturbation pipeline) and serves as one of the
benchmark's evaluation tasks.

There is no separate "R1 standalone resource paper" anymore. R1
becomes Paper B. The previous R1 manuscript draft is salvaged as the
Section 3 (Dataset) + Section 4 (Calibrated Audit) of Paper B.

### Refinement 3: R2 is not literally submission-ready today; needs ~1-2 weeks of polish

The current R2 manuscript draft was reframed multiple times and has
mixed legacy content. Before submitting it standalone to Nat MI:

- Lock the thesis to discovery + diagnostic (not steering or biological-
  primitives).
- Rewrite the abstract per the conservative phrasing in the 2026-05-13
  review.
- Add evidence pointers for every numeric claim.
- Move all causal-ablation negatives into one compact Limitations section.
- Add the M-2 characterization synthesis to Results.
- Strip claims that will live in Paper C (full method comparisons).

This is ~1–2 weeks of focused manuscript work. Worth doing carefully.

### Refinement 4: Build the framework first, the benchmarks second

The strongest deliverable is not two separate benchmark papers — it is
**one shared evaluation framework** that produces two benchmark
instances:

```
ProteinInterpret/
├── framework/          # shared evaluation code
│   ├── metrics/        # CRPI, faithfulness, localization, etc.
│   ├── methods/        # method registry + interfaces
│   └── datasets/       # cohort loaders
├── encoder/            # Encoder-Only instance
│   ├── datasets: ClinVar2000, IndelMissense v1, ProteinGym, VAMP-seq
│   ├── methods: LLR, IS-Mut, gradient, IG, SAE, AnnoSaliency
│   └── tasks: pathogenicity, mechanism, indel, abundance
└── decoder/            # Decoder-Only instance
    ├── datasets: EC-conditioned, lysozyme cohort, structural cohorts
    ├── methods: CLT, attention-output, attention-rollout, model-diff
    └── tasks: generation faithfulness, sink characterization, quality
```

The framework itself becomes a citable resource. Builds the leaderboard
infrastructure once, reuses it for both encoder and decoder instances.
This is the "build a benchmark framework" instinct the user expressed —
it is correct, and it should be foregrounded as a deliverable.

### Refinement 5: Position explicitly against existing benchmarks

Encoder-Only Benchmark needs to differentiate from:
- **ProteinGym** (Notin Nature 2024): variant-effect prediction. Our
  difference: ProteinGym evaluates PREDICTION quality, not
  INTERPRETATION quality. We evaluate explanations.
- **PEER** (NeurIPS 2022): representation quality. Our difference: PEER
  doesn't evaluate residue-level interpretability.
- **InterPLM / InterProt** (Simon & Zou 2025): SAE concepts in ESM.
  Our difference: they evaluate one method (SAE); we evaluate ~8 methods
  on common axes.
- **Adams et al. NeurIPS 2024**: protein SAE evaluation. Same
  differentiator as InterPLM.

Decoder-Only Benchmark has fewer prior benchmarks — first-mover
advantage. Still cite ProteinBench (Ye 2024) and any generative
evaluations published in the last 6 months.

---

## 4. Suggested Order and Timing

### Phase 1 (Weeks 1–2): R2 standalone submission
- Reframe R2 manuscript per Refinement 3.
- Submit to Nat MI by end of Week 2.
- Parallel: scope Paper B (Encoder Benchmark) — method list, metric list,
  dataset list locked. No new compute yet.

### Phase 2 (Weeks 3–10): Encoder-Only Benchmark
- Implement shared framework scaffold (Week 3).
- Implement 6 methods: ESM-2 LLR, in-silico mutagenesis, gradient,
  integrated gradients, attention rollout, SAE perturbation, plus
  AlphaMissense as reference (Weeks 3–5).
- Implement CRPI metric and AnnoSaliency hybrid method (Weeks 5–6).
- Run benchmark on ClinVar2000, IndelMissense v1, ProteinGym subset,
  VAMP-seq cohort (Weeks 6–8).
- VUS reclassification audit (Week 7, in parallel).
- Draft manuscript (Weeks 8–10).
- Submit Paper B to Nat MI / Nat Methods stretch.

### Phase 3 (Weeks 11–18): Decoder-Only Benchmark
- Port framework to decoder side (Week 11).
- Implement decoder-specific methods: CLT, attention-output transcoder,
  attention rollout, model-diff, cross-model triplet matching (Weeks
  12–14).
- Run benchmark on ZymCTRL/ProtGPT2/ProGen2 lysozyme + extended enzymes
  (Weeks 14–16).
- Draft manuscript (Weeks 16–18). Submit Paper C.

End-state by 2026-09-15: three submissions, one open-source framework,
two benchmark instances.

---

## 5. Two Things The User Should Decide Now

### Decision 1: Order — sequential or parallel?
**Recommendation: sequential.** Submit R2 first, then develop the
encoder benchmark, then decoder. Parallel execution doubles compute and
splits attention. Sequential preserves quality.

The exception: scoping Paper B can happen in parallel with R2 polishing
in Weeks 1–2 (no compute needed for scoping).

### Decision 2: Does R2 standalone preclude Paper C?
**Recommendation: no, if the split in Refinement 1 is honored.** R2
discovers the triplets and N-terminal subset. Paper C evaluates many
methods on a broader benchmark. The papers are complementary, not
overlapping.

The risk: Nat MI editors may ask why two papers from the same group.
**Mitigation:** Paper A is a focused discovery paper; Paper C is a
methods/benchmark paper. They cite each other appropriately. This is
common in machine learning publishing.

---

## 6. Three Risks To Track

### Risk 1: R2 standalone gets rejected at Nat MI
Probability: ~55–65 %. **Mitigation:** Paper C absorbs R2's content if
A is rejected. Treat A as "first attempt; backup is to fold into C."
Submit A to Cell Patterns or Bioinformatics if Nat MI rejects.

### Risk 2: Benchmark scope explodes
Paper B + Paper C together are 12–16 weeks of work. **Mitigation:**
each instance must have a hard MVP scope: 6 methods, 4 axes, 3 datasets.
No NLA-style methods until they exist for proteins. No more than 4
encoder models and 4 decoder models.

### Risk 3: Framework engineering eats the timeline
Building a clean shared framework (registry, metric API, dataset API)
is a software-engineering project on top of the research. **Mitigation:**
keep the framework minimal. Don't build a leaderboard server in v1;
release scripts + README + Zenodo data. Polished infrastructure can
follow the first publication.

---

## 7. What the User Got Right vs What Needs Sharpening

### Got right
- Architectural split (encoder vs decoder) is better than task split.
- R2 can stand alone at Nat MI.
- Shared framework is the right deliverable.
- R2's experience seeds Paper C.

### Needs sharpening
- The R2-standalone vs Paper-C content boundary must be drawn now,
  before submitting A.
- IndelMissense v1 lives in Paper B, not as a separate R1 resource
  paper.
- R2 needs 1–2 weeks of manuscript polish before submission, not zero.
- The framework architecture should be designed upfront to serve both
  benchmarks.
- Explicit positioning vs ProteinGym / PEER / InterPLM / Adams et al.
  is required.

### Net assessment
The user's plan is the cleanest strategic direction we have arrived at
across eleven planning iterations. It preserves R2's discovery value,
finds a coherent home for R1 (in Paper B), creates a new strategic
deliverable (the framework + Paper C), and matches each component to a
realistic venue.

**Adopt with the five refinements. Execute sequentially. End-state by
mid-September: three submissions, one framework, two benchmark
instances.**

---

## 8. One-Sentence Summary

**Yes, the encoder/decoder split is correct; submit R2 standalone to Nat
MI in 2 weeks after manuscript polish; reframe R1 as Paper B (Encoder-
Only Benchmark) with IndelMissense v1 as the centerpiece dataset and
the three additions from `conversation_history/OPUS_BENCHMARK_PIVOT_REVIEW_20260519.md`
(CRPI metric, AnnoSaliency method, VUS reclassification audit); develop
Paper C (Decoder-Only Benchmark) afterward using R2 as motivating case
study; share one framework underneath both benchmarks as an open-source
deliverable.**
