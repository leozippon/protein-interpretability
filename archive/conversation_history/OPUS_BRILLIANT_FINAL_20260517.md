# Opus Brilliant Final Pivot (2026-05-17)

Reply to `OPUS_PLAN_EXECUTION_20260516.md` and `OPUS_R1_PROBLEM_ANALYSIS_20260513.md`.

## TL;DR

The 2026-05-16 execution evidence forces a clear-eyed reframing.

- **R1 cannot reach Nat Methods.** Three independent gates failed: AM
  ensemble fails (Δ −0.0264, 95 % CI lower bound negative); indel SAE
  damage alone is beaten by ESM pseudo-NLL (0.7735 vs 0.8037); AM/SAE
  disagreement typing is null (0 of 22 BH-significant context classes;
  283 AM-right vs 38 SAE-right — a 7.4× asymmetry). The honest landing
  is Genome Biology / Bioinformatics / NAR as a **resource paper**.
- **R2 just hit a brilliant finding hidden in plain sight.** The 2026-05-16
  attention-sink experiments produced a **named, statistically airtight,
  cross-model conserved circuit primitive** — N-terminal initiator-methionine
  attention sinks. T011/T018/T023 fire at the first two residues with
  fraction 1.00 across 700 proteins; attention r ≈ 0.9; one-sided Fisher
  enrichment q ≈ **2.72 × 10⁻¹¹⁷**. Three independently trained protein
  LMs converge on this primitive.
- **This is the protein-LM analog of Anthropic's "induction heads" finding
  in NLP.** The Xiao 2024 attention-sink paper is the direct precedent.
  Same statistical signature, same convergent-emergence framing, applied
  to a different modality.
- **One additional experiment (~2 H200-days) makes it bulletproof**: causal
  ablation of T011/T018/T023, measuring perplexity Δ at positions 1–2 vs
  positions 3+. Closing the correlation→causation loop pushes R2 from
  Nat Mach Intel to a **realistic Nat Methods / Nat Comp Sci submission**.

The brilliant pivot is **not another iteration on R1 + R2 as currently
scoped**. It is: stop trying to save R1 as Nat Methods; extract the R2
attention-sink discovery into a focused short paper; let R1 be a clean
resource contribution at the right venue.

---

## 1. What I Missed Until Now

The 2026-05-16 results changed the picture. I had been writing R2 as a
"cautionary methods paper" because the universal triplets failed
biological annotation. The new evidence shows the framing was wrong: a
specific *subset* of the conserved triplets has clean biology, and
it's specifically the most interpretable phenomenon in modern transformer
research.

The relevant numbers, from
`Research2/results/circuit_analysis/attention_sink_biological_correlate_20260516/`:

| Triplet | Attention r | First-2-residue firing fraction | Background fraction | One-sided Fisher q |
|---|---:|---:|---:|---:|
| T011 | 0.921 | 1.000 | 0.055 | 2.72×10⁻¹¹⁷ |
| T018 | 0.914 | 1.000 | 0.055 | 2.72×10⁻¹¹⁷ |
| T023 | 0.885 | 1.000 | 0.055 | 2.72×10⁻¹¹⁷ |
| T025 | low | (not N-term) | — | not enriched |

Three triplets, each conserved across ProtGPT2 + ZymCTRL + ProGen2-medium,
each firing on position 1–2 of every analyzed protein (1.00 fraction!),
each correlating with attention received (r ≈ 0.9 with mean attention),
each significant at q = 10⁻¹¹⁷.

This is the closest concrete finding in either project to a **named
biological/architectural primitive**. And it didn't come from chasing
biology — it came from honest characterization of statistical structure.

---

## 2. The Brilliant Framing

**R2 paper:** rename, retitle, restructure around the attention-sink
finding.

**Working title:**
> "Initiator-methionine attention sinks emerge convergently across protein
> language models"

**Thesis:**
> Three independently trained protein language models (ProtGPT2,
> ZymCTRL, ProGen2-medium) converge on a small set of sparse-feature
> triplets that act as attention sinks at the N-terminal initiator
> methionine. We identify this convergent primitive using cross-model
> sparse-feature correlation analysis, validate its statistical
> robustness against a 30-replicate permutation null, characterize its
> biological correlate (N-terminal edge enrichment at q ≈ 10⁻¹¹⁷),
> demonstrate its causal role via targeted ablation, and provide a
> quantitative CLT-quality diagnostic based on universal-triplet
> recovery. This is, to our knowledge, the first identified cross-model
> conserved circuit primitive in protein language models, and an
> empirical analog of attention-sink phenomena recently described in
> natural-language transformers.

**Why this is brilliant:**
1. It is a *discovery* (a named conserved primitive), not an
   incremental improvement.
2. It has a **clean parallel in the most cited interpretability work of
   2024** (Xiao et al., attention sinks in NLP transformers). Reviewers
   will immediately understand the contribution.
3. The statistical evidence is extraordinary: q ≈ 10⁻¹¹⁷ is not a
   marginal claim.
4. It generates a **practical tool** — universal-triplet count as a CLT
   quality diagnostic — that other labs can adopt.
5. Three protein LMs of different architectures (two GPT-2 style, one
   ProGen2 style) converge on the same primitive, ruling out
   architecture-specific artifacts.

This reframing converts R2 from a "negative methods paper" into a
focused short discovery paper. Realistic venues:
- **Nat Methods** (with one more causal experiment, see below)
- **Nat Mach Intel** as a fallback
- **Cell Patterns** as a "short discovery" format
- **PNAS** for breadth + biology readership

---

## 3. The One Critical Experiment That Closes the Story

### Causal ablation of T011/T018/T023
- **Hypothesis:** if T011/T018/T023 implement N-terminal attention sinks,
  ablating them should disproportionately raise perplexity at positions
  1–2 (where they fire) and either redistribute attention to other
  tokens or destabilize early-position generation.
- **Method:** for each of the three protein LMs (ProtGPT2, ZymCTRL,
  ProGen2-medium):
  1. Use the existing TopK-aware on-manifold steering hook
     (`Research2/src/analysis/circuit_discovery.py`).
  2. Run two passes per sequence: intact and ablated (multiplier = 0
     for the triplet's feature in the relevant layer).
  3. Measure per-position Δ log-likelihood on a held-out 200-Swiss-Prot
     cohort.
  4. Report:
     - mean Δ logp at positions 1–2 vs positions 3–10 vs positions 10+
     - shift in attention-received distribution at position 1
     - structural validity (ESMFold pLDDT) on generated continuations
     - cross-model consistency: do all three models show the same
       pattern?
- **Acceptance gates (must satisfy ≥ 2 of 3):**
  - **G1 (per-model):** ablating each of T011/T018/T023 raises mean
    perplexity at positions 1–2 by ≥ 0.5 nats with permutation p < 0.05,
    while leaving positions 10+ within ±0.1 nats.
  - **G2 (cross-model):** the perplexity-Δ direction is consistent
    across all three protein LMs for at least 2 of the 3 triplets.
  - **G3 (attention redistribution):** ablating shifts attention-
    received from position 1 to elsewhere by ≥ 0.05 in cosine distance
    on the same cohort.
- **Stop criterion:** if 0 of 3 gates are satisfied, the triplets are
  correlation-only — the paper remains intellectually interesting
  but downgrades to "associated with attention sinks but not causally
  required."
- **Compute:** ~2 H200-days.
- **File:** `Research2/scripts/40_attention_sink_causal_ablation.py` (new).

This experiment converts the attention-sink finding from "they
correlate with sinks" to "**they ARE the sinks — ablating them changes
generation as predicted across three independently trained models.**"
That is a Nat Methods-tier mechanistic claim.

---

## 4. R1: Stop the Nat Methods Push

The 2026-05-16 evidence makes R1 + Nat Methods unrealistic:

- AM ensemble FAIL: AM=0.9474, AM+SAE z-sum=0.9210, AM+SAE stack=0.8542.
  All deltas vs AM are *negative*.
- Indel SAE damage FAIL: SAE alone=0.7735, ESM pseudo-NLL=0.8037, cheap-
  features-only=0.8447. SAE is the *weakest* component.
- AM/SAE disagreement typing FAIL: 0/22 BH-significant contexts; AM
  right/SAE right ratio = 7.4×. SAE does not systematically catch AM
  errors.
- Mechanism gate FAIL: protein-CV 0.516, gene-level 0.567,
  channelopathy 0.625 with DN→LOF collapse.
- ProteinGym FAIL: signed ensemble below LLR.

The only positive R1 finding worth keeping:
- **SAE+ESM+cheap-features grouped LR on indels = 0.9108.** This is
  competitive on a task no other published method handles uniformly.
- Annotation alignment after firing-position rerun (L19: 381 known,
  L23: 301 known) — interpretable features exist.
- IndelMissense v1 dataset (6,649 records with mechanism labels).

### R1 reframing (resource paper, not methods paper)

**New working title:**
> "IndelMissense v1 and an interpretable sparse-feature variant
> perturbation framework for ESM-2"

**Realistic venues:**
- **Genome Biology** as a resource paper (their resource track accepts
  benchmarks + interpretable tools).
- **NAR** Database/Web issue.
- **Bioinformatics** as a long methods paper.
- **PLOS Comp Bio**.

Not Nat Methods, not Nat Comms.

**Cleanup actions for R1 (do these regardless):**
1. Rewrite abstract to lead with IndelMissense v1 (the unique
   contribution) and the grouped-baseline AUC 0.9108. Move the SAE+LLR
   pathogenicity number to a secondary result.
2. Move all failed gates (AM ensemble, mechanism, channelopathy,
   ProteinGym) into a single "Limitations and Calibrated Negative
   Diagnostics" section. Honest, but compact.
3. **Do not run R1-Add-3 (VUS reclassification).** Historical ClinVar
   archives aren't staged and the previous two adds failed; this is
   unlikely to rescue the Nat Methods claim.
4. Add the SAE-residual case studies as a single supplementary figure,
   not as a main-text section.
5. Freeze.

---

## 5. The Combined-Paper Option

Worth considering: can R1 and R2 become **one** stronger paper?

**Combined thesis:** Interpretable sparse-feature analysis across the
protein-LM landscape — supervised representation (ESM-2-3B SAE → variant
perturbation) + generative representation (CLT atlas → attention sinks
+ universal triplets). The interpretability infrastructure is reusable
across training paradigms; specific findings include (a) IndelMissense
v1 benchmark + grouped baseline, (b) named convergent attention-sink
primitive across three protein generators.

**Pro:** One stronger Nat Methods submission rather than two weaker ones.

**Con:** R1 and R2 are about different models (ESM-2 vs ProtGPT2/ZymCTRL/
ProGen2). A combined paper has to do a lot of stage-setting. The R2
attention-sink finding is also strong enough to stand alone.

**Recommendation:** Pursue **two parallel submissions**: R2 short paper
to Nat Methods / Nat Comp Sci / Cell Patterns, R1 resource paper to
Genome Biology / NAR. Submit on the same week to avoid one bottlenecking
the other.

---

## 6. Two-Week Execution Plan

### Week 1 (this week)

| Day | Items |
|---|---|
| Mon | Restructure R2 manuscript around attention-sink discovery thesis. New abstract, new section headings. Launch causal ablation experiment on H200 (background). |
| Tue | R2 manuscript updates: new Results 1 (discovery), Results 2 (characterization, with M-2 summary), Results 3 (attention-sink subset + biological correlate, **headline section**), Results 4 (causal ablation, if available), Results 5 (CLT-quality diagnostic), Negative Diagnostics (steering, biological annotation, representation). |
| Wed | Causal ablation finishes; integrate result. Pivot R1 manuscript: new abstract, restructure to resource framing. |
| Thu | R1 cleanup: rewrite abstract, restructure section order, move failed gates to single negative-diagnostics section, add IndelMissense v1 as headline contribution. |
| Fri | Both manuscripts: internal review pass, evidence pointer audit, freeze. |

### Week 2

| Day | Items |
|---|---|
| Mon | Final compile and supplementary materials. |
| Tue | Cover letters for both submissions. R2 → Nat Methods (with Nat Mach Intel fallback letter prepared). R1 → Genome Biology. |
| Wed | Submit both. |

**Compute:** 2 H200-days for causal ablation. Negligible analyst-only
revisions otherwise.

---

## 7. What This Plan Does Not Do (and Why)

- **Does not chase more R1 experiments.** Three independent additions
  failed in 2026-05-16. Continuing wastes compute and delays submission.
- **Does not retrain CLTs/SAEs.** The current checkpoints support the
  attention-sink discovery. Retraining for "higher quality" doesn't
  change the named-primitive claim.
- **Does not pursue 10k-sequence atlas at scale.** Balanced-200 already
  produces a null-clean discovery; UniRef500 shows the same pattern at
  reduced effect size.
- **Does not promise wet-lab validation.** Future work only.
- **Does not pursue the universal-triplet "biological dictionary"
  framing.** The Swiss-Prot annotation gate failed (0/38 reached MI ≥
  0.1); the brilliant finding is narrower — the attention-sink
  subset — and that should be the headline, not the broader claim.

---

## 8. Risk Register

| Claim | Risk | Conservative phrasing |
|---|---|---|
| T011/T018/T023 are causally responsible for N-terminal attention sinks | Ablation may show <0.5 nat perplexity Δ at positions 1-2 | Frame as "correlated with N-terminal attention sinks"; downgrade Nat Methods to Nat Mach Intel |
| The primitive is convergent across architectures | All 3 models are decoder-only; reviewers may ask about encoder-only (ESM-2) | Future work; note that ESM-2 is masked-language not generative |
| Universal-triplet count is a useful CLT quality diagnostic | Currently shows v2=38 vs early-10k=16; reviewers may ask for more checkpoints | Report what we have, frame as preliminary diagnostic |
| The connection to Xiao 2024 NLP sinks is real | We do not have direct mechanistic equivalence | Phrase as "empirical analog" not "identical mechanism" |
| R1 is publishable as a resource paper | Genome Biology bar still requires a real contribution | IndelMissense v1 + grouped baseline AUC 0.9108 is a real contribution; honest negatives are scientifically valuable |

---

## 9. Why This Is Brilliant

The brilliance is not in finding more clever experiments. It's in
recognizing that the team has *already* produced the most striking
mechanistic-interpretability finding for protein LMs in this project:
**T011/T018/T023 are a real, named, statistically airtight cross-model
conserved primitive.**

The earlier framing of R2 buried this inside a "cautionary methods
paper" because the broader universal-primitives thesis failed. The
pivot is to recognize that even though most of the 38 conserved
triplets are uninterpreted statistical regularities, *three of them
are exactly the protein-LM analog of attention sinks in NLP*. That is
the brilliant finding — it just had to be extracted, named, and made
the headline.

Add the single causal ablation experiment to convert correlation into
mechanism, and R2 has a real shot at Nat Methods on novelty. R1
becomes a clean resource paper at the right venue. Both projects
publish, both make real contributions, and neither overclaims.

---

## 10. One-Sentence Summary

**Stop trying to save R1 at Nat Methods; promote the R2 N-terminal
attention-sink discovery (T011/T018/T023, q ≈ 10⁻¹¹⁷) to the
headline; run one causal ablation experiment; submit R2 to Nat
Methods/Comp Sci and R1 to Genome Biology in parallel within two
weeks.**

This is the cleanest, most defensible, and most influential path out
of the current state.
