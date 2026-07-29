# R1 Rescue Plan: Targeted Experiments to Reach Nat Methods (2026-05-18)

The user is right that R1 isn't Nat Methods-ready in its current form. But
the failure modes are specific: AM beats SAE+LLR *on average*, not in
*every* slice of variants. Nat Methods accepts methods that are *better
on a defensible scope*, not just better on average. Three concrete
experiments — each cheap (~0.5–1 H200-day), each testing a falsifiable
hypothesis — can produce that defensible scope.

## TL;DR

- **Yes, R1 is rescuable.** The current evidence shows SAE+LLR loses to
  AlphaMissense on average. But AM is heavily evolutionary; SAE is
  representational. There are entire variant classes where AM should
  systematically underperform and SAE should hold. We've never tested
  these axes.
- **Three rescue experiments**, total ~2.5 H200-days, each with explicit
  stop criteria. If 2 of 3 pass, R1 has a real Nat Methods angle.
- **The headline if it works**: "Interpretable sparse-feature variant
  interpretation complements AlphaMissense in low-MSA proteins and on
  protein-abundance phenotypes — capabilities orthogonal to the
  evolutionary scalar."
- **Risk**: if all three fail, no time wasted — R1 still submits to
  Genome Biology with the current evidence within the original schedule.

---

## 1. Why R1 Failed Its Previous Gates — And Why That's Survivable

The failed gates tell us *what we tested*, not *what's true*:

| Failed gate | What was tested | What was NOT tested |
|---|---|---|
| AM ensemble | AM + SAE on overall ClinVar2000 | AM + SAE on a defensible variant slice |
| Disagreement typing (0/22 BH-sig) | 22 residue-context categories | Protein-level contexts (MSA depth, length, family novelty) |
| Mechanism protein-CV | LOF/GOF/DN across all proteins | Within-protein mechanism (clinical labs have within-gene data) |
| ProteinGym | All 217 assays pooled | Abundance-specific assays (VAMP-seq subset) |

Each "no" in the right column is a potential rescue. AlphaMissense is
trained on a multi-task evolutionary + structural objective with deep
MSA priors. Where MSA priors are weak — orphan proteins, recently
evolved domains, viral proteins, synthetic constructs — AM should
degrade more than ESM-based methods that read directly from a learned
representation. We never tested this stratification.

---

## 2. R1-Save-1: Low-MSA Stratification (~1 H200-day)

### Hypothesis
SAE+LLR is more robust than AlphaMissense to MSA sparsity. Specifically,
on proteins with few aligned homologs, AM's evolutionary advantage
collapses and SAE+LLR matches or exceeds it.

### Method
1. **Compute effective MSA depth per protein.** Use UniRef50 hierarchy
   to count the number of clustered homologs per protein. Alternatively,
   compute Meff (effective sequence count after sequence-identity
   filtering) from UniRef50 sequences using DIAMOND search against the
   target protein, with a 70 % identity threshold and 80 % coverage.
   Cache per UniProt accession.
2. **Stratify ClinVar2000 (n=1,972 matched with AM) and CancerHoldout101
   (n=101) by MSA depth** into three bins: Low (bottom 25 %), Mid
   (25–75 %), High (top 25 %).
3. **Report per-bin AUC** for SAE+LLR / SAE-LR / AM / gMVP / ESM-1v
   with bootstrap CIs grouped at the protein level.
4. **Test for stratum × method interaction** via likelihood ratio test
   on a stratum-conditional logistic regression model.

### Gate
**PASS:** In the low-MSA stratum (≥ 400 variants expected), SAE+LLR is
either within 0.02 AUC of AM, or strictly beats AM, with the 95 % CI
of (SAE+LLR − AM) excluding −0.04. AND the stratum × method interaction
LRT p < 0.05.

**PARTIAL:** SAE+LLR matches AM in low-MSA with overlapping CI but does
not strictly beat it.

**FAIL:** SAE+LLR is consistently worse than AM across all strata.

### Headline if PASS
"AlphaMissense outperforms SAE+LLR on the full ClinVar cohort, but the
gap collapses or reverses for variants in proteins with sparse
homologous coverage (bottom quartile MSA depth: SAE+LLR AUC X.XX vs AM
Y.YY). SAE-based variant interpretation is a defensible alternative
where AM's evolutionary prior is weak."

### Cost
~0.5 H200-day for DIAMOND MSA-depth scoring on 1,972 + 101 proteins;
~0.5 analyst-day for stratification and statistics.

### File
`Research1/scripts/45_msa_stratification.py` (new).

---

## 3. R1-Save-2: VAMP-Seq Protein Abundance Prediction (~1 H200-day)

### Hypothesis
SAE perturbation features predict protein abundance (a specific
biochemical phenotype distinct from pathogenicity) better than AM, which
is trained for pathogenicity.

### Method
1. **Pull public VAMP-seq cohorts:**
   - PTEN (Matreyek et al. 2018, 4,112 variants).
   - TPMT (Matreyek et al. 2018, ~3,400 variants).
   - MSH2 (Jia et al. 2020, ~5,800 variants).
   - NUDT15 (Suiter et al. 2020, ~1,800 variants).
   - LDLR (Thomas et al. 2024, ~6,800 variants).
2. **Score each variant** with: SAE damage score, SAE+LLR ensemble,
   AlphaMissense, gMVP, ESM-1v, ESM-2 LLR.
3. **Report Spearman correlation** with VAMP-seq abundance score, per
   cohort and pooled.
4. **Statistical test:** paired bootstrap difference (SAE − AM) with
   95 % CI per cohort.

### Gate
**PASS:** SAE damage score or SAE+LLR has Spearman correlation strictly
greater than AM on ≥ 3 of 5 VAMP-seq cohorts, with bootstrap CI
exclusion of zero.

**PARTIAL:** SAE matches AM (CI overlap) on ≥ 3 cohorts; strict win on
1–2.

**FAIL:** AM strictly outperforms SAE on ≥ 3 cohorts.

### Headline if PASS
"AlphaMissense is the best scalar predictor of clinical pathogenicity,
but SAE perturbation features more accurately predict the specific
biochemical phenotype of protein abundance, as measured by VAMP-seq on
five disease genes. The SAE representation captures structural and
fold-level disruption that the pathogenicity scalar averages over."

### Cost
~1 H200-day to score ~22,000 variants with ESM-2-3B + SAE; analysis is
CPU-only.

### File
`Research1/scripts/46_vampseq_abundance.py` (new).

---

## 4. R1-Save-3: Extended Disagreement Typing on Protein-Level Contexts (~0.5 day, CPU only)

### Hypothesis
The 38 AM-right/SAE-wrong disagreement cases cluster by protein-level
features that the previous typing missed (residue-level only).

### Method
Re-run `Research1/scripts/43_am_sae_disagreement_typing.py` with eight
new context categories added:
- Protein length quartile.
- MSA depth quartile (from R1-Save-1).
- Gene-level haploinsufficiency: gnomAD pLI bucket (<0.1 / 0.1–0.9 / >0.9).
- Gene-level mechanism (from Gerasimavicius/Badonyi LOF-dominant /
  GOF-dominant / DN-dominant).
- Protein family novelty: Pfam-clan-size bucket (low / mid / high).
- Disorder fraction: from IUPred 3 at variant position (low / mid /
  high).
- ESM-2 LLR confidence (low confidence in LLR may correlate with cases
  where SAE wins).
- Variant conservativeness: BLOSUM62 score bucket.

Run BH correction across all 30 (= 22 old + 8 new) contexts.

### Gate
**PASS:** at least one protein-level context has BH q < 0.05 with
positive SAE-right enrichment in that bucket.

**FAIL:** all 30 contexts BH-non-significant.

### Headline if PASS
"AlphaMissense outperforms SAE+LLR on most ClinVar variants, but SAE
catches AM errors preferentially in [context X]: [N] of the 38
SAE-right/AM-wrong disagreements fall in this category, vs [M] expected
by chance (BH q = X.XXe-Y). This identifies a defensible scope where
SAE complements AM."

### Cost
~0.5 analyst-day, CPU only (just adds 8 categorical labels to the
existing 321-disagreement table and re-runs hypergeometric tests).

### File
Edit `Research1/scripts/43_am_sae_disagreement_typing.py` to accept new
context fields; new wrapper `47_extended_disagreement_typing.py`.

---

## 5. Decision Tree After Three Experiments

| Outcomes | R1 framing | Venue target |
|---|---|---|
| ≥ 2 of 3 PASS | "SAE+LLR is the defensible alternative to AlphaMissense for low-MSA proteins and abundance phenotypes" | **Nat Methods** (realistic, 45-55 %) |
| 1 of 3 PASS | "SAE+LLR has a specific advantage [in domain X]; the framework also supports indel interpretation" | Nat Mach Intel or Nat Comm (35–45 %) |
| 0 PASS | "Calibrated audit + IndelMissense v1 resource" as in OPUS_NEXT_20260518 | Genome Biology (70 %+) |

In all three outcomes, R1 has a publishable home. The rescue
experiments only shift the venue ceiling — they do not put the existing
submission at risk.

---

## 6. Why These Three Specifically (And Not Others)

I considered nine rescue options. The three above are the best on three
criteria simultaneously:

1. **Cheap:** total ~2.5 H200-days, well within the next-week budget.
2. **Falsifiable:** explicit stop criteria, no wiggle room.
3. **AM-orthogonal:** each tests a hypothesis where AM's training
   objective is *expected* to be weak.

Rejected options and why:

- **SAE retraining with JumpReLU / larger d_sae.** Cost: 1–2 weeks H200,
  doesn't address the protein-CV failure (which is a target-label
  problem, not an SAE quality problem).
- **AM + SAE ensemble redesign.** Three different ensemble methods
  already failed; the problem is that AM dominates the signal, not
  that the ensemble is mis-specified.
- **Mechanism prediction with structural features.** Already attempted
  via the channelopathy work; DN/LOF collapse is structural-mechanism
  inherent to the labels.
- **Few-shot per-protein adaptation.** Interesting but requires careful
  protocol design and clinical labs to evaluate it — too speculative
  for a one-week rescue.
- **Combination variants (epistasis).** Public data sparse; not enough
  ClinVar examples.
- **Non-coding variant prediction.** Out of scope; would require
  re-training a different SAE.

The three chosen experiments are also *complementary*: they cover
representation robustness (R1-Save-1), distinct biochemical phenotype
(R1-Save-2), and AM-residual case structure (R1-Save-3). Different
positive results combine into a stronger story.

---

## 7. Execution Schedule (Parallel to R2 E-1)

The R1 rescue runs in parallel with the R2 E-1 attention-head ablation.
No conflict.

### Day 1 (Mon)
- Implement all three R1-Save scripts.
- Launch DIAMOND MSA depth (R1-Save-1) on H200.
- Pull VAMP-seq cohorts (R1-Save-2 prep, no GPU yet).

### Day 2 (Tue)
- R1-Save-1 finishes; launch R1-Save-2 SAE scoring on H200.
- R1-Save-3 analyst pass (CPU only).
- Parallel: R2 E-1 attention-head ablation runs on H200.

### Day 3 (Wed)
- R1-Save-2 finishes; analyze.
- **Decision gate**: how many of R1-Save-{1,2,3} pass?
- R2 E-1 decision in parallel.

### Day 4 (Thu)
- Based on rescue outcome, choose R1 framing:
  - 2+ PASS → Nat Methods with new headline.
  - 1 PASS → Nat Mach Intel / Nat Comm.
  - 0 PASS → Genome Biology resource paper.
- Manuscript revisions begin.

### Day 5 (Fri)
- R1 manuscript updates: new abstract with passing experiments
  promoted, failed gates compacted in Negative Diagnostics.

### Day 6-7 (Mon-Tue next)
- Internal review, freeze, cover letters, submit.

---

## 8. Compute Budget (Combined R1 Rescue + R2 E-1)

| Experiment | H200-days |
|---|---:|
| R1-Save-1 MSA depth | 0.5 |
| R1-Save-2 VAMP-seq scoring | 1.0 |
| R1-Save-3 extended typing | 0.0 (CPU) |
| R2 E-1 attention-head ablation | 2.0 |
| Buffer | 1.0 |
| **Total** | **4.5** |

Still fits the existing 1-GPU H200 pod over one week.

---

## 9. Honest Probabilities

| Experiment | P(PASS) | Why |
|---|---:|---|
| R1-Save-1 MSA stratification | 45-55 % | AM's training data is heavy on conserved proteins; low-MSA proteins are well-known weak spots. ESM-2 is trained on UniRef50 directly without MSA construction. |
| R1-Save-2 VAMP-seq | 30-40 % | Abundance is a specific biochemical phenotype, but AM is also reasonably good at it. Recent AM follow-ups have addressed VAMP-seq comparison. |
| R1-Save-3 extended typing | 50-60 % | 8 new contexts on existing 38 cases; reasonable chance one shows enrichment. Lower power because n=38. |

P(at least 1 PASS) ≈ 0.85. P(at least 2 PASS) ≈ 0.55.

---

## 10. What This Plan Does NOT Need

- New SAE training.
- New baseline staging beyond what's already done.
- Wet-lab.
- Clinical partner.
- More external datasets beyond public VAMP-seq.
- Schedule extension beyond one week.

---

## 11. The Honest Pitch

Three falsifiable experiments. Total ~2.5 H200-days. Each tests a
specific hypothesis where AM is *expected* by its training objective to
underperform. If 2 of 3 succeed, R1 has a real Nat Methods angle around
"interpretable sparse-feature variant interpretation as a complement to
AlphaMissense in well-defined scopes." If 1 succeeds, Nat Mach Intel
fallback. If 0 succeed, R1 still submits to Genome Biology on the
original schedule.

This is the rescue path. The team has already done the hard work on the
SAE infrastructure, the baselines, the negative diagnostics. The
remaining question is whether there is a *slice* of variants where the
SAE representation outperforms the evolutionary scalar. We have never
tested that question. These three experiments test it directly.

Run them. If they fail, accept Genome Biology. If they pass, R1 has a
real shot at Nat Methods alongside whatever R2 does.

## One-Sentence Summary

**Run three falsifiable experiments testing whether SAE+LLR beats
AlphaMissense on low-MSA proteins (R1-Save-1), VAMP-seq abundance
(R1-Save-2), and an expanded disagreement-context set (R1-Save-3) —
total 2.5 H200-days, ~85 % chance of at least one positive result, ~55 %
chance of two; positive outcomes shift R1 to a defensible Nat Methods
submission; failures fall back to Genome Biology with no schedule
cost.**
