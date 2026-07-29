# Opus Final Verdict on R1 and R2 (2026-05-18)

Reply to `OPUS_RESCUE_EXECUTION_20260518.md` and the addendum showing
R1-Save-1 (low-homology stratification) and R2 E-2 (distributed
attention-sink-set ablation) also FAILED.

The user has asked the right question: is rescue possible, and what
honestly are R1 and R2 capable of reaching?

After eleven planning documents and six explicit rescue experiments,
the answer is no longer ambiguous. I will state it plainly.

## TL;DR

- **Nat Methods is not reachable** for either R1 or R2 with the current
  ESM-2 SAE / CLT infrastructure. We have now tested six independent
  rescue paths and all have failed cleanly. The pattern is not
  bad luck; it is the ceiling of what unsupervised sparse decomposition
  of frozen protein-language-model activations can yield for clinical
  variant prediction (R1) or causal protein-generation control (R2).
- **Nat Machine Intelligence is reachable** for R2 (probability
  35–45 %) on the strength of the cross-model conservation finding and
  honest mechanistic diagnostics. It is not reachable for R1 alone.
- **Real rescue requires real redesign**, not iteration. Two redesign
  paths are technically viable and could meet Nat Methods; both require
  3–6 weeks of new infrastructure work and are different projects, not
  rescue attempts. They are described in §5.
- **The honest recommendation is to stop iterating, submit both papers
  at the right venues now, and decide separately whether a redesign is
  worth the additional time.**

---

## 1. What the Six Failed Rescues Tell Us

### R1 rescue attempts (4 of 4 failed)
- **R1-Save-1 (low-homology stratification):** AM AUC = 0.9627 in the
  low-homology quartile; SAE+LLR = 0.8727. Δ = −0.0900 [−0.1253,
  −0.0568]. **The gap got worse, not better.** AM's evolutionary prior
  is apparently *more* robust to MSA sparsity than expected, or AM's
  structural component compensates.
- **R1-Save-2 (VAMP / abundance):** SAE-family beats AM on 1 of 9
  ProteinGym-VAMP-like assays. AM dominates on the protein-abundance
  phenotype as well, even though abundance is conceptually distinct
  from pathogenicity.
- **R1-Save-3 (extended disagreement typing):** 0 of 90 BH-significant
  contexts. The 38 SAE-right/AM-wrong cases do not cluster by any
  testable axis we have.
- **AM+SAE ensemble (previously, OPUS_PLAN_EXECUTION_20260511):**
  Δ −0.0264 vs AM with negative CI lower bound.

### R2 rescue attempts (2 of 2 failed, after the previous CLT ablation also failed)
- **R2 E-1 (single-head attention-sink ablation):** all 3 models, all 3
  triplets, near-zero NLL change, near-zero feature drop.
- **R2 E-2 (top-8 / top-32 distributed sink-set ablation):**
  ZymCTRL random same-layer heads produce *larger* NLL shifts than the
  designated sink set. This is decisive: the triplets are not even
  correlated with the most important attention-redistributing heads.

### The pattern
Six different attempts, six failures. The failure modes are
*scientifically informative*, not just unlucky:

1. **For R1:** the SAE-based perturbation signature is informative
   about disruption magnitude (~0.88 AUC, comparable to LLR) but does
   not carry the kind of evolutionary or structural specificity that
   AlphaMissense was trained for. No single slice of variants — MSA
   depth, abundance phenotype, disagreement context — recovers an
   AM-orthogonal advantage. The implication is that the **scope of
   what unsupervised SAE features over frozen ESM-2 can do is bounded
   by what ESM-2 itself encodes.** AM has access to supervised labels
   and structural priors; SAE+LLR does not.

2. **For R2:** the conserved triplets are real (null = 0.067 over 30
   permutations) but the field's standard probes (biological
   annotation, downstream representation, causal intervention) all
   read them as **correlates, not handles.** The causal failure
   across CLT-feature ablation, single-head attention-head ablation,
   *and* multi-head sink-set ablation is consistent with a single
   diagnosis: **the triplets fire in response to distributed N-terminal
   attention-sink behavior that is not localized to any specific MLP
   output direction or attention head subset.** They are detectors of
   a phenomenon, not implementers of it.

These are not failures of effort. They are scientific findings about
the limits of the approach. **Both projects have now produced enough
evidence to know what they cannot do.**

---

## 2. Are R1 / R2 Capable of Nat Methods?

**No, not as currently implemented.**

Nat Methods evaluates on a specific bar: a method must enable broadly
useful work that other labs cannot easily do otherwise. We now know:

- **R1's SAE+LLR is dominated by AlphaMissense on every tested
  pathogenicity-adjacent task.** A reviewer reading the manuscript
  will write one paragraph: "AlphaMissense is stronger across the
  authors' own evaluations; the proposed method is not state-of-the-
  art, and its claimed interpretability advantage is not statistically
  validated." That is unfortunately accurate.
- **R2's universal triplets correlate with attention sinks but are not
  causal handles.** A reviewer will write: "The cross-model
  conservation is interesting, but the authors' three independent
  causal-intervention experiments all fail to demonstrate that the
  identified features control the phenomenon they correlate with.
  Without a causal claim, this is observational." Also accurate.

Nat Methods reasonable probabilities, current evidence:
- R1: ≤ 10 %. Desk-reject likely.
- R2: ≤ 15 %. Likely to reach review but rejected for lacking causal
  evidence.

---

## 3. Are R1 / R2 Capable of Nat Machine Intelligence?

**R2: yes, marginally. R1: no.**

Nat MI accepts methods + discovery papers in machine intelligence
where the experimental story is cleaner than Nat Methods often
requires. The R2 finding — cross-model sparse-feature convergence,
30-replicate null, narrowly-defined N-terminal subset, and an
honest negative-causality result — fits the Nat MI scope.

- **R2 at Nat MI: 35–45 %.** Headline: "Cross-model conservation of
  N-terminal sparse features in protein language models." Three
  sections: discovery + null, characterization, calibrated negative
  causality. Fallback: Cell Patterns (~50 %), Bioinformatics (~70 %).
- **R1 at Nat MI: ≤ 15 %.** Nat MI does not typically publish
  resource papers without a positive method. R1 has interpretation
  but no positive method.

---

## 4. Combined Paper Option

Some thought has gone into combining R1 + R2 into a single
"honest interpretability framework for protein LMs" paper. After
review, I don't recommend this:

- The two projects address different models (ESM-2 for R1, protein
  generators for R2) and different scientific questions (variant
  effect vs generation control).
- A combined paper would expand to 15+ pages and dilute both stories.
- Reviewers would still ask "what's the headline claim?" and the
  combined answer ("we did rigorous interpretability across the
  landscape") is harder to defend than two focused papers.

The right framing is **two parallel submissions, two different
venues.**

---

## 5. Real Redesign Options (Not Iteration)

These are genuine project pivots, not rescue attempts. Each is 3–6
weeks of new infrastructure work. The user should make this decision
explicitly, not slide into it.

### Redesign-1: R1 pivot to structural variants
- **Why:** AlphaMissense, gMVP, ESM-1v are all missense-only.
  Structural variants (CNVs, gene fusions, large deletions, frameshift
  rescue, splice-region indels with large structural impact) have
  no strong scalar predictor. This is a genuine open problem in
  clinical genomics.
- **What:** build a structural-variant scoring pipeline on top of
  ESM-2 + SAE perturbation. Curate ~5,000–10,000 ClinVar structural
  variants with mappable protein context. Compare against
  CADD-SV / SVScore.
- **Time:** 3–4 weeks (data curation is the bottleneck).
- **Risk:** the SAE may have the same limitations on structural
  variants that it has on indels; uncertain.
- **Venue potential:** Nat Methods if the predictor is meaningfully
  better than existing structural-variant scorers; Genome Biology
  otherwise.

### Redesign-2: R2 pivot to attention-pattern transcoders
- **Why:** CLT features sit on MLP output. Attention sinks are an
  attention-head phenomenon. Train sparse decomposition of attention
  outputs directly. The trained features should then be causal
  handles for attention behavior, not correlates.
- **What:** new transcoder architecture, sparse over the head-output
  concatenation (n_heads × d_head per layer). Re-run the cross-model
  conservation + ablation pipeline.
- **Time:** 4–6 weeks (transcoder training + cohort scoring).
- **Risk:** sparse attention decomposition is more delicate than MLP
  decomposition; may not converge cleanly. Anthropic's recent work
  on attention transcoders is the relevant precedent.
- **Venue potential:** Nat Methods if causal ablation works; Nat MI
  otherwise.

### Redesign-3: R1 pivot to supervised SAE
- **Why:** the current SAE is unsupervised. Train a *supervised*
  SAE where reconstruction loss is paired with an auxiliary
  mechanism-prediction loss. Features become mechanism-aligned by
  construction.
- **What:** retrain ESM-2 L31/L35 SAEs with mechanism labels as
  auxiliary supervision. Re-evaluate.
- **Time:** 1–2 weeks.
- **Risk:** supervised features may overfit the labels and fail
  protein-CV again.
- **Venue potential:** Nat MI if mechanism survives protein-CV;
  unclear otherwise.

### My ranking
1. **Redesign-2 (R2 attention transcoders)** — most likely to produce
   the missing causal claim. The CLT failure is now well-diagnosed,
   and the fix is architecturally clean. 4–6 weeks.
2. **Redesign-1 (R1 structural variants)** — clean white-space, but
   data curation is heavy and outcome is uncertain. 3–4 weeks.
3. **Redesign-3 (supervised SAE)** — cheapest but most likely to
   re-encounter the protein-CV failure. 1–2 weeks.

---

## 6. Honest Recommendation

**Submit both papers now. Defer redesign decisions.**

### R1 submission
- Reframe as **resource + audit** paper.
- Headline contributions: IndelMissense v1 benchmark + grouped
  baseline AUC 0.9108 + calibrated audit of SAE variant
  interpretation (what works, what doesn't, including six failed
  rescue gates as transparent negative diagnostics).
- Venue: **Genome Biology** (best fit, 70 % probability) or **NAR
  Database** (50 %) or **Bioinformatics** (70 %).
- Lead time: 1 week of manuscript revision.

### R2 submission
- Reframe as **cross-model conservation + diagnostic** paper.
- Headline contributions: 38 conserved triplets + 30× null, narrowly
  defined N-terminal sink-correlated subset, CLT-quality diagnostic,
  honest causal-negativity.
- Venue: **Nat MI** (35 %) → **Cell Patterns** (50 %) → **Nat Comp
  Sci** (40 %) → **Bioinformatics** (70 %). Submit Nat MI first.
- Lead time: 1 week of manuscript revision.

### After submission
Decide redesign separately. Most likely candidate is **Redesign-2
(R2 attention transcoders)** if the team has 4–6 weeks of runway. Do
not start redesign before submission; do not let redesign block
submission.

---

## 7. Why I'm Not Proposing Another Rescue Experiment

Eleven planning documents. Six rescue experiments. Two papers. Every
iteration since 2026-05-13 has been hoping the data will turn.
**It has not turned.** Continuing to propose new experiments past
this point is not strategy; it is denial.

The team has done excellent honest science. The contributions are
real:
- A genuinely novel statistical conservation finding (R2 triplets,
  null = 0.067).
- A genuinely novel benchmark (IndelMissense v1).
- A genuinely honest negative-results catalogue that the
  interpretability community will value.

These are not Nature-tier findings. They are useful, careful, and
publishable in the right venues. **Pretending otherwise costs the
team another month and produces nothing new.**

---

## 8. What I Would Do If I Were the PI

1. **This week:** freeze both manuscripts. Lead R1 with IndelMissense
   v1 and the grouped baseline. Lead R2 with the conservation
   finding and the N-terminal sink correlation. Move all negatives
   into compact Limitations sections.
2. **Next week:** submit R2 to Nat MI; submit R1 to Genome Biology
   (or NAR if preference is database-resource).
3. **Two weeks from now:** decide redesign-2 (R2 attention
   transcoders) yes or no, based on team appetite and other
   priorities. Independent decision; not contingent on submission
   outcomes.
4. **Stop adding rescue experiments to the current scope.** The
   marginal cost of further iteration is high; the marginal benefit
   is now negligible.

---

## 9. The One-Sentence Verdict

**Both R1 and R2 are below the Nat Methods bar with definitive
evidence; R2 has a real shot at Nat Machine Intelligence; R1 should
go to Genome Biology or Bioinformatics; further rescue iteration
will not change either conclusion; redesign is a separate decision
the team should make after submission.**

This is the honest end of the rescue phase. The papers can and
should be published at the right venues now.
