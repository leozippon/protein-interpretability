# Opus Final Plan — No Wet-Lab Two-Paper Strategy (2026-05-11)

Reply to `OPUS_FINAL_PLAN_RETHINK_20260511.md`.

This is the final executable plan under the constraint that **neither current
paper depends on wet-lab validation**. Wet-lab is moved to "future work" only.
The plan answers the 10 questions in the prompt, produces the requested
deliverables (thesis statements, prioritized experiments with stop/go criteria,
drop list, manuscript outlines, risk register, compute plan), and supersedes
the Tier 3 items in `TODO_NEXT_20260511.md`.

---

## 0. Two-Paper Thesis Statements (final)

### R1 thesis
**"A unified, interpretable variant-effect framework spanning missense and
indel mutations: sparse autoencoder features on ESM-2 extend protein-language
variant interpretation beyond AlphaMissense's missense-only scope, provide
mechanistic explanations of disagreements, and reveal where evolutionary
likelihood does and does not capture functional disruption."**

Three pillars:
1. **Indel capability** — a unique computational reach AlphaMissense, gMVP,
   ESM-1v cannot match.
2. **Mechanistic interpretation** — per-residue SAE feature firing maps
   variant effects to specific functional categories.
3. **Honest complementarity** — SAE+LLR is not the best pathogenicity scorer
   but adds residual signal to AlphaMissense; published with case studies and
   limits.

Venue target: Nat Methods (realistic), with possible Nat Mach Intel companion
on the SAE feature atlas.

### R2 thesis
**"Cross-layer transcoder features form an interpretable, cross-model-
conserved sparse basis for protein generation models, useful as a
representation for downstream protein-function prediction and as a diagnostic
for generation quality, with negative steering results documenting the gap
between activation interpretability and causal control."**

Three pillars:
1. **Representation utility** — CLT features at least competitive with raw
   embeddings on standard protein tasks.
2. **Cross-model conservation** — a universal feature atlas across ProtGPT2,
   ZymCTRL, ProGen2.
3. **Diagnostic use** — CLT features detect generation quality (foldability,
   family membership), positioned as honest interpretability methodology.

Venue target: Nat Methods or Nat Mach Intel.

---

## 1. Answers to the 10 Questions

1. **Strongest no-wet-lab R1 story:** unified missense+indel variant
   interpretation, with the indel capability as the unique angle and the
   AlphaMissense-residual diagnostic as the interpretability contribution.
   The mechanism story is downgraded from "predict mechanism" to "annotate
   variant-effect features by mechanism category", which is defensible.

2. **R1 experiments that materially change the paper:**
   - Indel scale-up to full ClinVar indel cohort (currently 6,649; target
     ~80,000) — needed for the headline indel claim.
   - SAE+AlphaMissense ensemble test (does ensemble beat AlphaMissense alone)
     — needed for the complementarity claim.
   - Gene-level mechanism prediction with Pfam-clan holdout — needed to
     determine whether *any* mechanism claim survives.
   - Indel comparison vs SpliceAI/CADD/REVEL on in-frame indels — needed for
     the "extends beyond AlphaMissense" claim.
   - Bootstrap CIs and per-class numbers on all existing tables — needed for
     reviewer-defence.

3. **Indel expansion is essential and minimal-scope:** scale to all
   protein-mappable ClinVar indels (~80k records, ~3 H200-days). Do **not**
   add MAVE-style new data; that requires wet-lab. The point is to make the
   indel claim statistically robust on the dataset we already have.

4. **SAE-residual vs AlphaMissense framing:**
   - **Do** report Spearman correlation (0.70-0.80 range — partially
     orthogonal).
   - **Do** report SAE+AlphaMissense ensemble AUC and whether it beats
     AlphaMissense alone with bootstrap CI.
   - **Do** show ≥30 curated case studies where SAE features identify a
     specific functional-feature disruption that explains an AlphaMissense
     error or confirms its call.
   - **Do not** call SAE a "better predictor" — call it a complementary
     interpretable explanation layer.
   - **Do not** report ensemble numbers without per-protein holdout.

5. **Gene-level mechanism prediction:** test it. It's the only mechanism
   claim that could still survive. Decision criterion: if Pfam-clan-holdout
   gene-level macro-AUC ≥ 0.70, include as a R1 result. If < 0.60, remove
   the mechanism narrative entirely and frame R1 around indels +
   interpretation only.

6. **Strongest no-wet-lab R2 story:** "CLT features as an interpretable
   sparse basis for protein language models". Three sub-claims:
   (a) features beat raw embeddings on at least 2 of 5 standard tasks;
   (b) ≥30 features conserve across ≥3 protein generators;
   (c) CLT features detect structurally invalid generations.
   Steering goes into the limitations section as a calibrated negative.

7. **R2 experiments needed to justify the representation pivot:**
   - Downstream linear-probe evaluation on ≥5 protein-function tasks vs raw
     embedding baselines (CB513 SS, Pfam family, EC class, subcellular
     localization, stability/ProTherm).
   - Cross-model feature conservation at scale (matched-feature dictionary
     across ProtGPT2 v2 + ZymCTRL v2 + ProGen2-medium).
   - Hallucination/quality detection (CLT features → ESMFold-validated
     structural integrity).
   - Honest steering-negative summary: keep the existing data, do not run
     more steering experiments.

8. **Toxin and hallucination detection — keep one, defer one:**
   - **Hallucination/quality detection** stays in R2 as it's natural to the
     representation thesis (interpretable features detect when generations
     are broken).
   - **Toxin/dual-use detection** is deferred to a separate biosecurity
     paper. It's a sensitive topic that deserves its own focused treatment
     with security-expert co-authors. Keeping it in R2 dilutes the message
     and invites unrelated reviewer scrutiny.

9. **Drop from current TODO plan (because wet-lab or unlikely to change manuscripts):**
   - `P3-T0-C` (wet-lab partnership decision) — no longer relevant.
   - `P3-T1-E` (DN-on-KCNQ1 case-study) — without wet-lab validation,
     a single-gene rescue case study does not change the paper headline.
   - `P3-T3-A` rare-disease retrospective — partner-dependent.
   - `P3-T3-B` MAVE pilot — wet-lab.
   - `P3-T3-D` enzyme retargeting — wet-lab.
   - `P3-T2-C` toxin detection — deferred to separate paper.
   - All "Outcome B" and "Outcome C" framings in `TODO_NEXT_20260511.md`.

10. **Three-week execution plan with stop/go is in §3 below.**

---

## 2. Prioritized Experiment List

Tag scheme: `[F-tier]` `[project]` `[criticality]` `[effort]` `[file]`.
Criticality: **GATE** = must succeed for the paper to be written as
framed; **CONTRIB** = adds a major section if it succeeds; **HARDEN** =
reviewer-defence, low-risk improvement.

### F-T0 — Reframing (3 days, no compute)

**F-T0-1 — Manuscripts: lock thesis statements and remove overclaims.**
- [HARDEN] [both] [1 day]
- Files: `manuscripts/nature_methods_r1_variant_perturbation/main.tex`,
  `manuscripts/nature_methods_r2_circuit_diagnostics/main.tex`.
- Action: rewrite abstract and intro to the thesis statements in §0; move
  variant-CV mechanism, ProteinGym, channelopathy, R2 steering null into
  honest negative-diagnostics sections.
- Acceptance: every numeric claim has an evidence-file pointer (carryover
  from `P3-T0-B`); no claim contradicts §3 facts of the prompt.

**F-T0-2 — Sunset the steering manuscript framing in R2 README.**
- [HARDEN] [R2] [2 hours]
- Update `manuscripts/nature_methods_r2_circuit_diagnostics/README.md` to
  match the representation-and-diagnostics thesis.

### F-T1 — R1 experiments (≤ 2 weeks)

**F-T1-1 — Indel scale-up to all protein-mappable ClinVar indels.**
- [GATE] [R1] [~3 H200-days] [new: `Research1/scripts/32_indel_full_scale.py`]
- Inputs: `clinvar_indels.tsv` (already on H200), staged SAE checkpoints L19/23/27/31/35.
- Outputs:
  - `Research1/results/variant_effect/indel_full_scale_predictions_<date>.jsonl`
  - `..._summary.json` with: per-class AUC, protein-level CV AUC, mechanism
    distribution, bootstrap CIs.
- Acceptance: damage AUC ≥ 0.78 on protein-level CV, with per-class
  breakdown stratified by deletion / insertion / duplication / delins /
  in-frame / frameshift.
- **Stop criterion:** if protein-level damage AUC < 0.72 on the full cohort,
  the indel claim weakens to "first systematic indel evaluation; results
  motivate further work". Continue paper but downgrade the headline claim.

**F-T1-2 — Indel competitor comparison on in-frame subset.**
- [CONTRIB] [R1] [~1 H200-day] [new: `Research1/scripts/33_indel_competitors.py`]
- Inputs: F-T1-1 output, CADD scores (staged), REVEL (staged or pulled),
  SpliceAI for splice-region indels.
- Acceptance: head-to-head table on the in-frame indel subset showing AUC for
  each competitor + SAE-LR. If SAE-LR beats CADD/REVEL on this subset by
  any margin with overlapping CIs allowed, framing is "unified missense and
  indel"; otherwise framing is "first systematic indel comparison".

**F-T1-3 — SAE × AlphaMissense ensemble + per-protein CV.**
- [GATE] [R1] [~0.5 H200-day] [new: `Research1/scripts/34_alphamissense_ensemble.py`]
- Inputs: AlphaMissense matched scores (already staged), SAE-LR scores per
  variant.
- Method: stacked logistic regression and z-sum ensembles, evaluated under
  protein-level CV with bootstrap CIs.
- Acceptance: Δ AUC of ensemble vs AlphaMissense alone, with 95% CI. If
  CI lower bound > 0 by ≥0.005 AUC, include as a major claim. If CI
  contains 0, do not claim ensemble improvement; report as null and frame
  SAE as interpretation layer only.
- **Stop criterion:** keep the "complementarity" claim only if CI lower
  bound > 0.

**F-T1-4 — Gene-level mechanism prediction with Pfam-clan holdout.**
- [GATE] [R1] [~1 H200-day] [new: `Research1/scripts/35_gene_level_mechanism.py`]
- Inputs: per-protein aggregated SAE features (mean firing, top-quartile
  firing, mean perturbation) on the 255-protein cohort; mechanism labels at
  gene level (dominant class from Gerasimavicius/Badonyi).
- Holdout: 10-fold CV with Pfam-clan as the CV unit (no clan in train and
  test).
- Acceptance: macro-AUC with 95% CI.
- **Stop / go criterion:**
  - macro-AUC ≥ 0.70 → include as a R1 result section "Gene-level mechanism
    signatures from SAE features"
  - 0.60 ≤ macro-AUC < 0.70 → include only as a supplementary diagnostic
  - macro-AUC < 0.60 → drop mechanism narrative entirely; reframe R1 to
    indel + interpretation only

**F-T1-5 — SAE-residual case study curation.**
- [CONTRIB] [R1] [~3 analyst-days, 0 GPU] [new: `Research1/scripts/36_sae_residual_cases.py`]
- Inputs: variants where AlphaMissense is confidently wrong (review status ≥
  2 stars, disagrees with ClinVar) and SAE-LR is confidently right (or
  vice-versa).
- Output: 30-50 curated case studies with: variant, AM score, SAE-LR score,
  ClinVar truth, top firing SAE features, Swiss-Prot / Pfam / PDB
  annotation, manual interpretation.
- Acceptance: ≥30 cases with biological narrative; ≥3 distinct patterns
  documented (e.g., AM misses post-translational-modification site, SAE
  catches; SAE misses ancient evolutionary constraint, AM catches).

**F-T1-6 — Tighten existing tables with bootstrap CI and per-protein-holdout
pathogenicity rerun.**
- [HARDEN] [R1] [~0.5 H200-day] [edit: existing `available_baseline_summary` pipeline]
- Add: per-protein holdout pathogenicity AUC for SAE-LR, SAE+LLR, AlphaMissense,
  gMVP, ESM-1v on ClinVar2000 (currently is reported under variant pooling).
- Acceptance: protein-level holdout numbers added to the main table with the
  per-protein 95% CI.

**F-T1-7 — IndelMissense dataset release artefact.**
- [CONTRIB] [R1] [~3 days, 0 GPU] [new: `data/indelmissense/v1/`]
- Inputs: F-T1-1 output JSONL, predefined protein-level and Pfam-clan splits.
- Outputs: dataset README, splits CSV, JSONL, baseline scorer wrapper.
- Acceptance: a self-contained subdirectory + README that another lab could
  reproduce. **Drop wet-lab leaderboard scaffold** (which assumed external
  partnership) — release as a static benchmark instead.

### F-T2 — R2 experiments (≤ 2 weeks)

**F-T2-1 — CLT features as downstream representation (5 standard tasks).**
- [GATE] [R2] [~5 H200-days] [new: `Research2/scripts/22_clt_downstream_eval.py`]
- Models: ZymCTRL v2 CLT, ProGen2-medium CLT; also raw embeddings baseline
  for each.
- Tasks: CB513 secondary structure, Pfam family (1000 most common), EC class
  (CLEAN setup), DeepLoc subcellular, FireProtDB stability ΔΔG (Spearman).
- Probe: L2-regularised linear regression / logistic regression per task.
- Acceptance: CLT-feature linear probe vs raw-embedding linear probe head-to-
  head on all 5 tasks, with bootstrap CIs.
- **Stop / go criterion:**
  - CLT features beat embeddings on ≥ 2 of 5 tasks (95% CI lower bound > 0):
    "CLT features are at least competitive" claim survives.
  - 1 of 5 tasks: include as a partial result; reframe R2 thesis to
    "interpretable basis", not "competitive representation".
  - 0 of 5: cut the representation claim, focus R2 on atlas + diagnostic only.

**F-T2-2 — Cross-model universal feature atlas.**
- [CONTRIB] [R2] [~1 H200-week] [new: `Research2/scripts/23_universal_features.py`]
- Models: ProtGPT2 v2 + ZymCTRL v2 + ProGen2-medium v2 (latest CLTs).
- Method: feature-feature Pearson correlation across models on a shared
  10k-sequence UniRef50 cohort; layer-anchor matching via existing CKA
  pipeline.
- Output: matched-feature dictionary, per-feature Pfam-alignment annotation,
  top universal-feature list.
- Acceptance: ≥ 30 features matched at r > 0.9 across all three models, with
  ≥ 10 of those receiving an interpretable Pfam / GO annotation.
- **Stop criterion:** if < 10 features match at r > 0.9, the "universal
  atlas" claim is gone; reframe as "cross-model conservation is partial,
  with model-specific feature spaces dominating".

**F-T2-3 — Hallucination / structural-integrity detection.**
- [CONTRIB] [R2] [~3 H200-days] [new: `Research2/scripts/24_hallucination_detection.py`]
- Inputs: 1,000 ZymCTRL EC-conditioned generations across the 8 EC classes
  (existing T2-C cohort plus a 500-sequence supplement).
- Labels: ESMFold pLDDT > 70 globally confident, Foldseek TM > 0.5 to any
  PDB structure of the target EC.
- Classifier: linear / shallow-MLP on per-sequence aggregated CLT features
  from L3 / L12 / L30; compared against ESM-2 embedding baseline.
- Acceptance: AUC > 0.80 for structural-validity prediction from CLT
  features; identify ≥ 3 features that strongly indicate broken vs valid
  generations.
- **Stop criterion:** if AUC < 0.70, drop this section and rely on T2-1 and
  T2-2 only.

**F-T2-4 — Steering-negative summary written up.**
- [HARDEN] [R2] [~0.5 day, 0 GPU]
- Files: existing T2 result JSONs, manuscript "Limitations and Negative
  Diagnostics" section.
- Output: a calibrated negative-result write-up: hook diagnostics, direct-
  effect feature selection, on-manifold steering, real-vs-random metric
  calibration. Frame as: "circuit interpretability does not yet enable
  reliable causal steering of protein generators at FVU=0.33 dead=67%."
- Do **not** run more steering experiments.

### F-T3 — Cross-paper (1 week)

**F-T3-1 — Manuscript freeze and consistency pass.**
- [HARDEN] [both] [~3 days]
- Action: synchronize abstracts and discussion sections to the thesis
  statements; double-check that no claim relies on dropped Tier 3 work.
- Acceptance: external reviewer-style internal pass produces no
  contradiction between text and result files.

**F-T3-2 — Public reproducibility artefact.**
- [HARDEN] [both] [~2 days]
- Action: top-level README in `manuscripts/` linking to evidence JSONs and
  scripts for each main-text figure; package as supplementary information.

---

## 3. Three-Week Execution Plan With Explicit Gates

### Week 1

- **Day 1-2:** F-T0-1 / F-T0-2 manuscript reframe (analyst).
- **Day 1-3:** F-T1-1 indel scale-up (H200).
- **Day 1-5:** F-T2-1 downstream eval (H200, parallel to F-T1-1).
- **Day 3-5:** F-T1-3 ensemble + per-protein pathogenicity rerun (H200, light).

**Gates evaluated end-of-Week-1:**
- G1: F-T1-1 indel AUC ≥ 0.72 → continue indel headline. If lower, downgrade.
- G2: F-T1-3 ensemble CI lower bound > 0 → keep complementarity claim. If
  not, drop ensemble claim.

### Week 2

- F-T1-2 indel competitors on in-frame subset.
- F-T1-4 gene-level mechanism (must finish by mid-Week-2).
- F-T1-5 SAE-residual case curation (analyst, runs alongside).
- F-T2-1 finish + F-T2-2 universal atlas (H200).
- F-T2-3 hallucination detection (H200).

**Gates evaluated end-of-Week-2:**
- G3: F-T1-4 gene-level macro-AUC ≥ 0.70 → mechanism section in R1.
- G4: F-T2-1 CLT beats embeddings on ≥ 2 / 5 → representation claim survives.
- G5: F-T2-2 universal-feature count ≥ 30 → atlas section.

### Week 3

- F-T1-7 IndelMissense dataset packaging.
- F-T2-4 steering-negative write-up.
- F-T3-1 manuscript freeze.
- F-T3-2 reproducibility artefact.
- Buffer day for re-runs.

**End-of-Week-3 deliverables:** two manuscripts in compile-ready state with
all numeric claims pointing to evidence files; supplementary materials
packaged; IndelMissense dataset directory published in the repo.

---

## 4. Drop / Defer List

### Drop entirely (no longer in scope)
- `P3-T0-C` wet-lab partnership decision.
- `P3-T1-E` DN-on-KCNQ1 case study (single-gene, no wet-lab payoff).
- `P3-T1-F` indel-structural-ensemble — defer unless F-T1-1 underperforms;
  the ESMFold ensemble is expensive and may not be needed.
- `P3-T3-A` rare-disease retrospective (partner-dependent).
- `P3-T3-B` MAVE pilot (wet-lab).
- `P3-T3-C` web tool deployment — move to post-publication.
- `P3-T3-D` enzyme retargeting (wet-lab).
- `P3-T2-C` toxin / dual-use detection — split into a separate biosecurity
  paper with appropriate co-authors.

### Defer to future work (mentioned in manuscript "Future Directions" only)
- Stronger CLT retraining (JumpReLU, larger d_clt) — conditional on
  representation success.
- Wet-lab validation of indel mechanism predictions.
- Wet-lab validation of circuit-level enzyme retargeting.
- Clinical translation pilot (rare-disease retrospective).

---

## 5. Revised Manuscript Outlines

### R1 — `manuscripts/nature_methods_r1_variant_perturbation/main.tex`

1. **Abstract** (rewrite to thesis in §0).
2. **Introduction** — variant interpretation gap on indels, scope of SAE
   features as interpretable additions to scalar predictors.
3. **Results 1: Pathogenicity prediction with honest baselines.** Table of
   SAE-LR, ESM-2 LLR, SAE+LLR, AlphaMissense, gMVP, ESM-1v on ClinVar2000 +
   cancer holdout under protein-level CV (F-T1-6). Frame as "competitive but
   not best".
4. **Results 2: AlphaMissense complementarity and residual interpretation.**
   F-T1-3 ensemble; F-T1-5 case studies showing where SAE features add a
   mechanistic explanation.
5. **Results 3: Indel mechanism prediction.** F-T1-1 + F-T1-2 vs CADD /
   REVEL on the in-frame subset; per-class AUC table.
6. **Results 4: Gene-level mechanism signatures (conditional on F-T1-4
   gate).** Pfam-clan holdout, top features per class.
7. **Negative Diagnostics.** ProteinGym SAE-fitness scoring, variant-level
   protein-CV mechanism failure, channelopathy DN→LOF collapse, with the
   message: "evolutionary likelihood and functional-feature disruption are
   distinct, and SAE measures the latter".
8. **Discussion.** Where SAE features add value, where they don't, scope for
   indel-MAVE wet-lab validation as future work.
9. **Methods.**
10. **Supplementary.** IndelMissense dataset description (F-T1-7).

### R2 — `manuscripts/nature_methods_r2_circuit_diagnostics/main.tex`

1. **Abstract** (rewrite to thesis in §0).
2. **Introduction** — interpretability for protein generators; CLT
   methodology; why steering and representation are distinct goals.
3. **Results 1: CLT features as a downstream representation.** F-T2-1
   linear probes on 5 protein-function tasks vs raw embedding baselines.
4. **Results 2: Cross-model universal feature atlas.** F-T2-2 matched
   features across ProtGPT2 / ZymCTRL / ProGen2; annotated examples.
5. **Results 3: Generation-quality diagnostic.** F-T2-3 hallucination /
   structural-integrity detection from CLT features.
6. **Negative Diagnostics: Steering does not yet work.** F-T2-4 calibrated
   negative summary: hook plumbing verified; direct-effect feature
   selection; TopK-aware on-manifold steering; 0/8 EC classes significant;
   metric-stack calibration confirms the null is not a metric artefact.
7. **Discussion.** Why representation interpretability succeeds where
   causal steering fails at current CLT quality; scope for stronger CLTs
   and wet-lab retargeting as future work.
8. **Methods.**
9. **Supplementary.** Universal-feature dictionary; cross-model
   correspondence tables; per-EC steering result tables.

---

## 6. Risk Register

Each row: the claim, what could go wrong, conservative phrasing.

| Claim | Risk | Conservative Phrasing |
|---|---|---|
| R1: SAE+LLR is "competitive" on pathogenicity | AM and gMVP are reproducibly stronger | "SAE+LLR matches single-method baselines (LLR, ESM-1v) but is outperformed by AM and gMVP. Its value is interpretability, not absolute performance." |
| R1: SAE complements AlphaMissense | F-T1-3 ensemble CI may contain 0 | "We test whether SAE features carry signal orthogonal to AlphaMissense. If the ensemble does not improve over AM alone, we report this honestly and use SAE only for case-study interpretation." |
| R1: Indel mechanism prediction | F-T1-1 protein-level AUC may drop further on the full cohort | "We provide the first systematic indel-mechanism evaluation at this scale. Performance may be lower than the missense pipeline; we report per-class CIs and discuss limits." |
| R1: Gene-level mechanism signatures | F-T1-4 Pfam-clan holdout may fail (<0.6) | Drop the claim entirely. Do not soften — remove the section. |
| R1: SAE-residual case studies | Cases may not be biologically convincing | Limit to cases with explicit Swiss-Prot annotation evidence; do not include speculative interpretations. |
| R2: CLT features beat embeddings | May fail on all 5 tasks (F-T2-1 0/5 gate) | Drop the "representation" claim; reframe R2 as "atlas + diagnostic" paper. |
| R2: Universal-feature atlas | < 30 features may match at r > 0.9 | Report whatever matches; if < 10, downgrade to "partial cross-model conservation with model-specific feature spaces dominating". |
| R2: Hallucination detection | AUC may be < 0.70 | Drop the section; rely on T2-1 + T2-2. |
| R2: Steering negative is a "calibrated negative" | Reviewer may demand steering improvement attempt | We have already exhausted the principled fixes (direct-effect selection, on-manifold TopK steering, metric calibration). Further attempts require stronger CLTs, which is future work. |
| Both: protein-level vs variant-level CV | Reviewers may push back on the protein-CV requirement as "too strict" | Document Pfam-clan and protein-level splits separately; show both and let the reader judge. |

---

## 7. Compute / Resource Plan (3 weeks)

Approximate H200-GPU-day budget. 1 H200-day = 1 GPU × 24 hours.

| Week | Item | H200-days |
|---|---|---:|
| 1 | F-T1-1 indel scale-up | 3.0 |
| 1 | F-T2-1 downstream eval start | 2.0 |
| 1 | F-T1-3 ensemble + protein-CV pathogenicity | 0.5 |
| 2 | F-T1-2 indel competitors (in-frame subset) | 1.0 |
| 2 | F-T1-4 gene-level mechanism | 1.0 |
| 2 | F-T2-1 finish | 1.0 |
| 2 | F-T2-2 universal atlas | 2.0 |
| 3 | F-T2-3 hallucination detection | 2.5 |
| 3 | Buffer for re-runs / G-gate failures | 2.0 |
| **Total** | | **15.0** |

Notes:
- F-T1-5 case study, F-T1-7 dataset, F-T2-4 steering write-up, and all F-T0
  / F-T3 items are CPU/analyst work — no H200 budget.
- Hold approximately 30% of compute headroom in Week 3 to re-run any item
  whose gate fails (F-T1-1, F-T2-1, F-T2-2 are the riskiest).
- Use the existing 1-GPU H200 pod allocation pattern from previous runs;
  no new arena scheduling needed.

---

## 8. Net Effect vs Previous Plan

Previous `TODO_NEXT_20260511.md` had three tiers with wet-lab Tier 3 paths
and two ambitious "Outcome B/C" landings. This plan:

- **Deletes** Tier 3 entirely except as "Future Directions" prose.
- **Reframes** the R1 mechanism story from "predict variant mechanism" to
  "annotate variant effects by mechanism category and report gene-level
  signal" (smaller but defensible).
- **Reframes** R2 from "interpretable drug design" to "interpretable sparse
  representation for protein language models", with steering as a calibrated
  negative diagnostic rather than the headline.
- **Tightens** the experiment list to 12 numbered items with explicit
  stop/go gates at the end of each week.
- **Caps** the H200 budget at ~15 GPU-days over 3 weeks.

The expected outcome is two Nat Methods (or Nat Methods + Nat Mach Intel)
manuscripts in submission-ready state by end of Week 3, with all claims
backed by evidence files, no wet-lab dependencies, and the strongest
defensible interpretations of the data we have. This is not a Nature paper.
It is an honest, useful contribution that the field will adopt.

The path to a Nature paper from this state requires wet-lab validation. We
document that path explicitly in the "Future Work" sections of both
manuscripts so the option remains open for a follow-up project.
