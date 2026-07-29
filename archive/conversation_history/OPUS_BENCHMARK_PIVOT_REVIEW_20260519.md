# Review: ProteinBench-Interpret Pivot Proposal (2026-05-19)

Examining `project_records/INTERPRETABILITY_BENCHMARK_PIVOT_FOR_OPUS.md`
against the user's four-part question: is the proposal **feasible,
complete, creative, brilliant**?

## TL;DR Verdict

| Dimension | Score | Diagnosis |
|---|---|---|
| **Feasible** | YES (with caveat on scope) | Most infrastructure exists. New method implementation is 4–6 weeks; ambitious version is 2–3 months. |
| **Complete** | NO | Three structural gaps: no positioning vs existing benchmarks, no leaderboard mechanism, no protein-specific metric or method. |
| **Creative** | PARTIAL | Reframing negative results as benchmark contributions is genuinely clever. The six-axis framework is borrowed from NLP/CV XAI, not invented. |
| **Brilliant** | NOT YET | Currently a defensible methodology paper. Needs one of three additions (§5) to clear the brilliance bar for a Nat MI or stretch Nat Methods submission. |

**Bottom line:** the pivot is the right strategic direction, but the
proposal as written would land at PLoS Comp Bio or Bioinformatics
(~70 %), not Nat MI (~20 %), not Nat Methods (≤ 10 %). Three concrete
additions described in §5 lift it to Nat MI ~35–45 %, Nat Methods
~15–20 %. Without those additions, it is not yet a brilliant pivot.

---

## 1. Feasibility Check

### What is already in place
- ESM-2-3B SAE checkpoints (L19/23/27/31/35) with firing-position
  annotation alignment.
- ClinVar2000, CancerHoldout101, IndelMissense v1, mechanism labels,
  ProteinGym SAE benchmark.
- Three protein-generator CLTs (ProtGPT2 v2, ZymCTRL v2,
  ProGen2-medium) with cross-model triplet atlas.
- AlphaMissense, gMVP, ESM-1v scored across the variant cohorts.
- 38 universal triplets, 30× null, characterization tests,
  attention-sink subset, checkpoint-quality diagnostic.
- Pfam/CLEAN/HMMER/Foldseek/ESMFold metric stack calibrated.
- Eleven previous planning/execution packets documenting what works
  and what doesn't.

**Roughly 60–70 % of a Variant-Explanation track is already runnable from
existing artefacts.**

### What is missing and how long it takes
- **Gradient attribution + integrated gradients on variants** — ~1 H200-day
  per cohort × 4 cohorts = 4 H200-days. Concrete and standard.
- **In-silico mutagenesis at scale** — ~2 H200-days.
- **Attention attribution / attention rollout** — ~1 H200-day per model.
- **NLA-style natural-language explanations** — non-trivial. Requires
  building activation→text→reconstruction infrastructure. **2–4 weeks
  of new engineering**, and probably defer to Future Work.
- **Localization metrics:** require curated functional-site annotations
  across the cohort with matched random controls. The Swiss-Prot
  per-residue annotations from R1 cover this partly; need to assemble
  the matched-control sampling pipeline. ~1 week of analyst time.
- **Faithfulness via ablation:** every method needs a corresponding
  intervention interface. SAE has it; LLR has it (mask + remask);
  gradient methods need a "remove top-k residues" interface. ~1 week.
- **Specificity controls:** matched random feature/residue samples per
  method, per protein. ~1 week.
- **Robustness across seeds and layers:** mostly reruns of existing
  pipelines with seed sweeps. ~1 week.

**Total realistic effort: 6–10 weeks of work (compute + engineering +
analysis) for a credible Variant-Explanation MVP. The Generative track
adds another 4–6 weeks.**

This is not a 2-week wrap-up. It is a **new 2–3 month project**, not a
rescue. The user should be clear-eyed about that before adopting it.

---

## 2. Completeness Check

Three structural gaps in the current proposal:

### Gap 1: No positioning vs existing protein-LM interpretability benchmarks
The proposal does not cite or differentiate from:

- **ProteinGym** (Notin et al., Nature 2024): variant-effect benchmark.
- **PEER** (Xu et al., NeurIPS 2022): protein representation evaluation
  benchmark.
- **ProteinBench** (Ye et al., 2024): protein generative model benchmark.
- **InterPLM** / **InterProt** (Simon & Zou, 2025): protein SAE
  benchmarking and feature interpretation.
- **Adams et al.** (NeurIPS 2024): "Sparse autoencoders specifying
  biological concepts in protein language models."

**Without explicit positioning, reviewers will desk-reject as
"yet another XAI benchmark."** The proposal must say what is novel
relative to each of these.

### Gap 2: No leaderboard / deployment mechanism
Successful benchmarks have submission portals, fixed splits, hash-
verified test sets, and a maintenance plan (ProteinGym, OGB, MoleculeNet
all have this). The proposal is silent on:

- Who maintains the benchmark after publication?
- How do new methods submit?
- Is there a held-out hidden test set?
- What is the citation policy?

A benchmark paper without these is just "a comparison study."

### Gap 3: No protein-specific metric or method
The six axes (predictive utility, localization, faithfulness, specificity,
robustness, actionability) are all standard XAI evaluation dimensions
imported from NLP/CV literature (e.g., Holistic Evaluation of Language
Models, BIG-Bench-Lite, IBM AI Explainability 360). The protein-domain
port is a translation, not a new idea.

**To stand out, at least one metric or method must be domain-specific
in a non-trivial way** — see §5.

---

## 3. Creativity Check

### Where it is creative
- **Reframing negative results as benchmark contributions.** The R1
  failed gates (mechanism, low-homology, abundance, disagreement
  typing) and the R2 failed gates (steering, single-head, multi-head,
  CLT-feature ablation) become data points in the benchmark, not
  scientific failures. This is genuinely clever post-hoc framing
  that preserves value from sunk effort.
- **Two-track structure.** Variant + Generative is a meaningful
  conceptual split that reflects how protein-LM interpretability is
  actually used in the field.
- **Anthropic-aligned framing.** The connection to circuit tracing,
  model diffing, and NLA gives the paper a venue-aware vocabulary
  reviewers will recognize.

### Where it is not yet creative
- **No novel evaluation axis.** Six axes is a standard XAI evaluation
  framework. The protein-specific innovation is missing.
- **No novel method.** The proposal lists existing methods. A
  benchmark + a method designed to be Pareto-optimal across the
  axes would be much stronger.
- **No theoretical contribution.** What is the *conceptual claim* about
  protein-LM interpretability that emerges from this work? Currently
  the answer is "many methods work on some axes and fail on others,"
  which is a finding, not a theory.

---

## 4. Brilliance Check

A brilliant benchmark paper has at least one of:

1. **A counterintuitive finding** that the benchmark uniquely surfaces.
2. **A method that wins** (often designed using benchmark insights).
3. **A protein-specific metric** that becomes the named contribution.
4. **A concrete clinical/industrial use case** that adopts the benchmark.
5. **A direct critique of the field** ("X% of published interpretability
   claims fail our faithfulness gate when re-evaluated").

The current proposal has **none of these** as a planned deliverable. It
plans to run the comparison and write up whatever the table shows. That
is methodology, not brilliance.

**Verdict on brilliance:** the proposal is sensible and defensible. It
is not yet a brilliant paper. It can become one with the additions in §5.

---

## 5. Three Concrete Additions To Make It Brilliant

If the user wants this pivot to be genuinely Nat MI-realistic, the
proposal needs to add at least two of the following before launching.

### Addition A: A novel protein-specific metric — "catalytic residue distance"
The standard XAI localization metric is "top-k overlap with ground-truth
important residues." That is generic. Replace it with a protein-domain
metric:

**Catalytic-Residue Pointing Index (CRPI):** for each pathogenic variant
in proteins with curated active-site / catalytic-residue annotations,
score how close the method's top-k attributed residues are to the
known catalytic dyad/triad/active-site centroid in 3D space (using
AlphaFold2 structures already in `external_resources/`).

- This is biology-aware in a non-trivial way.
- It directly captures clinical utility ("does the explanation point at
  the residue a structural biologist would care about?").
- It generates a single quantitative number reviewers can rank methods by.
- It would become the named contribution that distinguishes this from
  generic XAI benchmarks.

Implementation cost: ~1 week. Requires AF2 structures (already staged)
and Swiss-Prot active-site annotations (already staged via
`pfam_residue.tsv` and Swiss-Prot residue features).

### Addition B: A method designed to be Pareto-optimal — "Anno-Saliency"
Most XAI benchmarks compare existing methods. Stronger papers introduce
a new method that benefits from the benchmark insights. Propose:

**Anno-Saliency:** a hybrid method combining annotation-selected SAE
features (already implemented) with gradient saliency, weighted to
maximize CRPI (Addition A) while preserving ClinVar AUC.

Concretely: for each variant, compute
`score(r) = α · SAE_delta_annot(r) + (1-α) · |grad logp(mut|wt at r)|`,
where α is selected per protein by a validation step.

- Cheap to implement (~1 week).
- Could outperform individual methods on both predictive AUC and CRPI.
- Provides the "we propose a method that wins" headline that reviewers
  reward.

Implementation cost: ~1 week. Builds on existing pipelines.

### Addition C: A clinical reproducibility analysis — "VUS reclassification audit"
The field will care about benchmarks that produce statements about
clinical practice. Propose:

**VUS Reclassification Audit.** Pull ClinVar variants that were VUS as
of 2023 and reclassified to P/LP or B/LB by 2025. Score them blind with
every benchmark method. Report agreement rates with the eventual
reclassification — per method.

- ~120 reclassified VUS exist in public ClinVar archives (estimated).
- Direct clinical relevance: tells the field which interpretability
  methods would have made the right call before the reclassification.
- Provides the surprising-finding angle: probably AM dominates, but
  on the SAE-residual subset (where AM disagrees with the eventual
  reclassification), specific interpretability methods agree more.

Implementation cost: ~1 week including archive curation. **This was
deferred from R1-Save-3 because archives weren't staged. Stage them
now for the benchmark paper instead.**

### Why these three additions specifically
Each addresses one of the three structural gaps:
- A → protein-specific metric (Gap 3)
- B → novel method (Gap 3)
- C → clinical utility statement (Gap 1 differentiation from
  ProteinGym et al.)

With two of three, the benchmark becomes a real contribution. With all
three, this is a credible Nat MI submission and a stretch Nat Methods
submission.

---

## 6. Risks the Proposal Underestimates

### Risk 1: Scope explosion
The full proposal has 10+ methods × 6 axes × multiple datasets across
two tracks. That is **3+ months of work**. The proposal's MVP-1 reduces
this somewhat but still aims wide. Reviewer-acceptable scope:
**6 methods, 4 axes, 3 datasets, one track.** Generative track becomes
a follow-up paper or extended supplementary.

### Risk 2: Negative-result fatigue
The proposal explicitly says "no requirement that SAE wins." That is
intellectually honest, but reviewers reading a paper where every
method fails on every axis will conclude the field is unsalvageable —
not that the benchmark is useful. **At least one method must clearly
win on at least one axis** for the paper to read constructively.
Addition B (Anno-Saliency) addresses this by design.

### Risk 3: NLA inclusion is a trap
Natural-language autoencoders are a recent Anthropic concept that has
not been ported to proteins. Including NLA-style methods in the
benchmark forces the team to implement them well or risk a methods
critique. **Defer NLA explicitly to Future Work.** Mention briefly in
the discussion.

### Risk 4: This is a third project, not a pivot
The current state has two manuscript drafts in compile-ready form
(`manuscripts/nature_methods_r1_variant_perturbation/` and
`manuscripts/nature_methods_r2_circuit_diagnostics/`). Starting the
benchmark pivot means abandoning or delaying those submissions.
**The user should treat this as a new project decision, not as a
rescue of R1/R2.**

### Risk 5: The "audit of overclaimed field" framing is dangerous
Calling out the field as overclaiming invites pushback from the authors
of cited methods. Soften to "We provide systematic evaluation across
methods and axes." The benchmark's negative findings will speak for
themselves.

---

## 7. Recommended Path Forward

### If the user wants to proceed with the pivot

**Adopt the benchmark idea, but tighten and harden it:**

1. **Reduce MVP scope to one track first.** Variant-Explanation only.
   Generative track becomes a follow-up paper.
2. **Add the three concrete additions** (CRPI metric, Anno-Saliency
   method, VUS reclassification). Each ~1 week.
3. **Position explicitly against ProteinGym, PEER, InterPLM, Adams
   et al.** in the intro.
4. **Defer NLA-style methods to Future Work.**
5. **Build leaderboard infrastructure** even if minimal: hashed test
   set, submission email, version-locked baselines on Zenodo/HuggingFace.
6. **Target timeline:** 4–6 weeks to a draft. Nat MI submission
   plausible by 2026-07-01.

### If the user is undecided

**Submit R1 and R2 as planned now** (Genome Biology + Nat MI as in
`OPUS_FINAL_VERDICT_20260518.md`), AND start the benchmark pivot
**in parallel** as a third project with its own timeline. The benchmark
naturally cites both papers as "case-study findings that motivate this
benchmark." This is the lowest-risk path.

### If the user wants to maximize venue

**Defer R1/R2 submission by 6–8 weeks.** Spend that time on the three
additions and the benchmark MVP-1. Submit all three together: R1 to
Genome Biology, R2 to Cell Patterns, benchmark to Nat MI. Coordinated
release narrative is stronger than fragmented timing.

---

## 8. Direct Answer to the Four Questions

**Feasible?** Yes, but it is a 2–3 month project, not a 2-week pivot.

**Complete?** No. Missing: positioning vs existing benchmarks, a
leaderboard mechanism, a protein-specific metric or method.

**Creative?** Partially. The negative-result-as-benchmark-data
reframing is creative. The six-axis framework is borrowed.

**Brilliant?** Not yet. Currently a defensible methodology paper, ~70 %
PLoS Comp Bio probability. With the three additions (CRPI metric,
Anno-Saliency method, VUS reclassification audit), it lifts to
~35–45 % Nat MI and ~15–20 % stretch Nat Methods.

---

## 9. Final Recommendation

**The proposal is a sensible strategic direction but is not yet a
brilliant paper.** GPT has identified the right pivot but has not yet
designed the contributions that would make it stand out.

If the user has 6–8 weeks and wants a real Nat MI shot, **commit to
the benchmark pivot with the three concrete additions in §5**, defer
R1/R2 submission, and execute as a single coherent submission.

If the user prefers a lower-risk path, **submit R1 and R2 now** at
realistic venues (Genome Biology + Nat MI / Cell Patterns), and
treat the benchmark as a third project to develop in parallel with
careful citation of the first two papers as motivating case studies.

Either path is defensible. The wrong path is launching the benchmark
as currently scoped without the three additions — that produces a
6-week project that still lands at PLoS Comp Bio, the same venue R1
would land at as a resource paper with no extra effort.

The user's instinct to pivot R1 to a benchmark is sound. The execution
needs sharpening before it becomes truly brilliant.
