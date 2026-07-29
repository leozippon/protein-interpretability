# Next-Round TODOs (2026-04-29)

Goal: convert R1 from a Nat-Methods-grade tool into a Nature/Science contribution,
and convert R2 from a null steering pipeline into a real causal-circuit result —
or honestly retire the steering claim and pivot.

The work is grouped in tiers. Tier 0 must complete before any Tier 1+ result is
trusted, because two of the existing negative results are caused by code bugs,
not biology.

Each item is tagged: `[tier]` `[project]` `[impact]` `[effort]` `[file]`
where impact is the reason the experiment changes the paper, and effort is rough
H200 time. Acceptance criterion is what makes the item count as done.

---

## Operating Guidelines

These apply to every item below.

1. **Fail fast.** No silent fallbacks. If an SAE checkpoint or annotation pkl is
   missing, raise — do not substitute defaults.
2. **No claim before plumbing.** If a hook does not produce a measurable logit
   shift under multiplier=10, do not interpret its multiplier=0/2 results.
3. **Protein-level holdout, not variant-level.** Variant-level CV leaks per-protein
   SAE feature signatures and inflates AUC.
4. **Bootstrap CIs and per-class metrics.** Macro-AUC alone hides DN performance
   (n=80). Always report per-class CI.
5. **Each commit is one item.** Do not bundle unrelated fixes.
6. **Log every run** in `Research{1,2}/docs/EXPERIMENT_LOG.md` with: date, command,
   key hyperparameters, output path, headline number.
7. **Big artefacts stay on H200/GPFS.** Pull only the JSON/TSV summaries back to D.
8. **No new metrics without a calibration run.** Any new scorer (CLEAN, HMMER,
   Foldseek, AlphaMissense) must first reproduce a published number on a known
   cohort before being used to judge our own outputs.

---

## Tier 0 — Bug fixes (must do first; ≤2 days)

These are blockers. Until they pass, R2 causal claims and R1 ProteinGym claims
cannot be defended.

### T0-A: R2 hook plumbing diagnostic — [tier:0] [project:R2] [impact:BLOCKER] [effort:~2h] [file:`Research2/scripts/diagnostics/00_hook_sanity.py` (new)]

Symptom: the causal-ablation result has `mean |Δ logp| = 0.0` exactly, and
`logp_intact == logp_ablated` to 7+ decimals at every position of every
sequence. That is bit-identical, not a small effect — the steering hook is not
on the path that produces those logits.

Plan:
1. Add a hit counter and a print of `(input.norm(), output.norm(), feature_pre_act.max())`
   inside the hook in `Research2/src/analysis/circuit_discovery.py:222`.
2. Write a 30-line sanity script: load ZymCTRL v2 + the v2 CLT, pick any one
   feature with confirmed nonzero pre-activation on a lysozyme reference, run
   teacher-forced forward with multiplier ∈ {1.0 (no hook), 1.0 (with hook),
   10.0, 0.0}, assert that:
   - hook fires count > 0
   - multiplier=1.0 with hook → logp identical to no-hook (within FP precision)
   - multiplier=10 → max |Δ logp| > 1.0 nat at some position
   - multiplier=0 → max |Δ logp| > 0.05 nat at some position
3. If hit count is zero, the hook is on the wrong submodule. Inspect with
   `[(n,m) for n,m in pm.model.named_modules() if 'mlp' in n.lower()]` and
   re-attach. Suspects: GPT-2 fused inner forward, `torch.compile` cache,
   ProGen2 trust-remote-code wrapper.

Acceptance: sanity script prints the four assertions and they all pass on
ZymCTRL v2 and ProGen2-medium. Commit the script under
`Research2/scripts/diagnostics/`.

### T0-B: R1 ProteinGym sign + score redesign — [tier:0] [project:R1] [impact:HIGH] [effort:~2h offline + 1 H200 GPU-day] [file:`Research1/scripts/21_proteingym_sae_followup.py:341`]

The current SAE score is `Σ |Δf|·F1` — a magnitude of disruption. Larger means
more damage, so it should anti-correlate with DMS fitness. Two issues:
1. The ensemble adds `zscore(llr) + zscore(sae_total)` even though the two are
   anti-correlated by construction → cancellation. Mean ensemble ρ=+0.199 vs
   LLR alone +0.434. Fix sign: `zscore(llr) - zscore(sae_total)`.
2. Even with the sign fixed, negated-SAE alone only reaches +0.231. Replace the
   raw magnitude scorer with the trained mechanism-classifier's class log-odds:
   for each ProteinGym mutant, compute `P(LOF | sae) + P(DN | sae) - P(neutral | sae)`
   from the R1-B classifier and use that as the SAE-side score.

Acceptance:
- Diagnostic JSON reports both `ensemble_signed` (sign-fixed) and
  `ensemble_classifier` (classifier-based) means + bootstrap CI, and the
  win-rate vs LLR per assay-size bucket.
- If neither beats LLR on ≥40% of usable assays, frame ProteinGym in the paper
  as a *negative diagnostic* showing SAE measures mechanism not generic fitness.
  Do not pitch the SAE+LLR ensemble as a ProteinGym improvement claim.

### T0-C: R2 ec_features.pkl provenance audit — [tier:0] [project:R2] [impact:BLOCKER] [effort:~30m] [file:`Research2/scripts/diagnostics/01_pkl_provenance.py` (new)]

The local copy at `Research2/results/circuit_analysis/zymctrl/ec_features.pkl`
is shape `(36, 4096)` — the v1 d_clt. The runner at
`Research2/scripts/run_r2_v2_remaining_0424.sh:101` asserts d_clt=8192. Spot-
check the file used on H200: hash + shape print.

Acceptance: the H200 ec_features.pkl is confirmed to have d_clt=8192, was
recomputed from the v2 CLT after step_200000, and the local D copy is
refreshed.

### T0-D: R1 protein-level holdout split — [tier:0] [project:R1] [impact:HIGH] [effort:~3h] [file:`Research1/scripts/16_mechanism_classifier.py`]

Variant-level 10-fold CV almost certainly leaks per-protein SAE feature
signatures, since features for the same protein appear in train and test folds.
Re-run mechanism classification with proteins (not variants) as the CV unit.

Acceptance:
- `mechanism_classifier_results.json` reports both variant-level CV (existing
  metric) and protein-level CV.
- Drop in macro-AUC under protein-level holdout is reported honestly. If the
  protein-level macro-AUC is below 0.7, the headline R1 claim must be downgraded.

---

## Tier 1 — R1 to Nature-defensible rigor (≤2 weeks)

### T1-A: AlphaMissense / PrimateAI-3D / gMVP / ESM-1v head-to-head — [tier:1] [project:R1] [impact:CORE STORY] [effort:~1 GPU-day] [file:`Research1/scripts/23_baseline_headtohead.py` (new)]

Reviewers will demand this. Score the same 2000 ClinVar variants and the
101-variant cancer holdout with each existing competitor. Report:
- Per-method AUC + bootstrap CI on pathogenicity
- Per-method per-class AUC on mechanism (LOF/GOF/DN); these methods all output
  scalars, so most should be ≈0.5 on mechanism — that *is* the story
- Spearman of SAE-LR scores vs each competitor; the **orthogonal residual** is
  the defensible novelty figure

Acceptance: a single table comparing AUC across {SAE-LR, LLR, AlphaMissense,
PrimateAI-3D, gMVP, ESM-1v} on (a) pathogenicity, (b) per-mechanism class.

2026-05-07 status: completed for all currently available baselines except
PrimateAI-3D, which remains gated/pending. Outputs:
`Research1/results/variant_effect/available_baseline_summary_20260507.{json,md}`
and `Research1/results/variant_effect/available_baseline_sae_residual_cases_20260507.tsv`.

2026-05-10 status: PrimateAI-3D is treated as unavailable because the gated
dataset could not be obtained. T1-A is therefore finalized over accessible
baselines only: SAE-LR, ESM-2 LLR, SAE+LLR, AlphaMissense, gMVP, and ESM-1v.
Do not keep this TODO blocked on PrimateAI-3D for the current analysis pass.

### T1-B: Per-mechanism feature → annotation manual mapping — [tier:1] [project:R1] [impact:figure-ready] [effort:~1 day analyst time] [file:`Research1/scripts/24_mechanism_feature_audit.py` (new)]

The classifier's coefficients are recorded but not interpreted. For each class
(LOF / GOF / DN), pull the top-10 features by LR coefficient at each layer
{19, 23, 27, 31, 35}. For each feature, fetch its top-activating
Swiss-Prot/Pfam/PDB-residue annotations from the existing annotation pkl, and
inspect 5 max-activating sequences manually.

Hypotheses to test:
- LOF features fire on active-site / fold-core residues
- GOF features fire on regulatory / allosteric / interface residues
- DN features fire on oligomer-interface / dimerization-domain residues

Acceptance: a markdown table with one row per top feature: layer, feature_idx,
LR coefficient, top Pfam family of activating residues, top GO term, top
PDB-binding annotation, manual interpretation. This becomes Figure 3 of the
paper.

### T1-C: Indel / frameshift mechanism prediction — [tier:1] [project:R1] [impact:UNIQUE NOVELTY] [effort:~2 H200-days] [file:`Research1/scripts/25_indel_mechanism.py` (new)]

This is your most defensible novelty: AlphaMissense, ESM-1v, gMVP, PrimateAI-3D
**cannot score indels uniformly**. The 185k staged ClinVar indels (deletion,
insertion, duplication, delins) can be scored by SAE perturbation distance.

Plan:
1. For each indel, compute SAE features for WT and mutant sequences across all
   5 layers; concatenate into a perturbation signature.
2. Train mechanism classifier (or transfer from R1-B) on the missense subset,
   apply to indels.
3. Report: per-class AUC for pathogenicity (indels with pathogenic vs benign
   ClinVar labels), and predicted-mechanism distribution.

Acceptance: indel pathogenicity AUC > 0.85 on a held-out protein-level split,
and predicted mechanism-class distribution differs significantly between
truncating frameshifts (expect LOF-dominated) and in-frame insertions
(expect mixed).

### T1-D: Annotation pipeline `--save-firing-positions` rerun — [tier:1] [project:R1] [impact:enables R1-F] [effort:~1 GPU-day] [file:`Research1/scripts/04_analyze_our_sae.py` (likely)]

R1-F (annotation expansion) failed because cached pkls have
`n_features_with_firing = 0` for every layer. Rerun the annotation pipeline
with firing positions saved, then rerun `19_expand_annotation.py`.

Acceptance: deep-layer KNOWN feature counts at L35 rise from 31 to ≥60 (proof
that the expansion now scores).

### T1-E: Channelopathy clinical cohort — [tier:1] [project:R1] [impact:Nat Med hook] [effort:~3 days] [file:`Research1/scripts/26_channelopathy.py` (new)]

Channelopathies (KCNQ1, SCN5A, KCNH2, CACNA1C) have rich genotype-phenotype
data, mechanism-specific drug response, and a known LOF/GOF clinical
distinction. ClinVar + ClinGen variant curations + published cohort papers
(e.g., LQTS Genotyping Consortium, BrS registry) provide labels.

Plan:
1. Pull KCNQ1/SCN5A/KCNH2 pathogenic missense variants with curated mechanism
   from ClinGen and the LQTS literature.
2. Predict mechanism using R1 classifier; compare to expert curation.
3. For the predicted-LOF set vs predicted-GOF set, look up retrospective drug
   response (β-blocker for LOF LQTS, mexiletine for GOF SCN5A) from cohort
   papers.

Acceptance: ≥80% concordance between predicted mechanism and ClinGen-curated
mechanism on channelopathy cohort. If retrospective drug-response concordance
exists, this becomes a Nat Med-level companion finding.

Status 2026-05-07 CST: completed with a negative result. Curated labels were
staged and 64 LOF/GOF/DN missense rows were evaluated, but concordance was
0.625 with macro-F1 0.444, below the >=80% target. Main failure mode:
dominant-negative channel variants collapse to LOF under the current R1
mechanism classifier.

---

## Tier 2 — R2 to a real positive result (≤2 H200-weeks)

Conditional on T0-A (hook plumbing) succeeding.

### T2-A: Direct-effect feature selection — [tier:2] [project:R2] [impact:BLOCKER for steering] [effort:~1 H200-day] [file:`Research2/scripts/16_direct_effect_features.py` (new)]

Replace mean-activation z-score with attribution patching. For each candidate
feature f at layer L:
- Compute `grad(EC-class log-likelihood) · feature_activation` on a held-out
  EC-class cohort.
- Rank features by `|grad · activation|` (Anthropic-style direct-effect score).

Acceptance: top-10 direct-effect features per (EC, layer) saved as
`Research2/results/circuit_analysis/zymctrl/direct_effect_features_v2.pkl`,
with shape `(n_classes, n_layers, top_k=10)`.

### T2-B: TopK-aware on-manifold steering — [tier:2] [project:R2] [impact:CORE STORY] [effort:~3h coding + 1 H200-day] [file:`Research2/src/analysis/circuit_discovery.py:steer_generation`]

Current steering writes only the within-window-t=0 decoder column without
re-applying TopK; result is off-manifold and hurts foldability.

Replace with: perturb in CLT pre-activation space, re-apply training-time
TopK, project the *full window* decoder write into the residual stream at
layers L+0..L+window-1, and replace the CLT-explained portion of the MLP
output rather than adding to it:

```python
# pseudocode in the hook for layer L
pre_act = ReLU(W_enc[L] @ resid + b_enc[L])
pre_act[:, :, feat_idx] *= multiplier            # intervention
sparse = scatter_topk(pre_act, k=128)            # training-time gate
recon_full_window = einsum("bsf,fwd->bswd", sparse, W_dec[L])
# at the current layer, replace the CLT-explained portion of mlp_out
clt_explained = einsum("bsf,fd->bsd", sparse_unsteered, W_dec[L,:,0,:])
return mlp_out - clt_explained + recon_full_window[..., 0, :]
```

Downstream layers receive the propagated effect through the residual stream
naturally.

Acceptance: on a sanity case where the unsteered model nearly always emits a
specific token at a specific position, multiplier=0 on the relevant feature
must drop that token's logit by ≥1 nat without changing logits at distant
unrelated positions by more than 0.1 nat.

### T2-C: Real EC-class metric triad — [tier:2] [project:R2] [impact:CORE STORY] [effort:~2 days setup + 1 H200-day per benchmark] [file:`Research2/scripts/17_ec_metrics.py` (new)]

Replace the regex motif scorer with a triad:
1. **CLEAN** (Yu et al., Science 2023) for EC top-1 prediction. Report top-1
   and top-3 accuracy.
2. **HMMER hmmscan** against the target Pfam family. Report fraction with
   E-value < 1e-3.
3. **ESMFold + Foldseek TM-align** to known PDB structures. Report fraction
   with TM-score > 0.5 to reference.

Calibrate first: run all three on (a) 100 real lysozyme sequences from
UniProt, (b) 100 random UniRef50 sequences. Confirm reals score high, randoms
score low. Only then use on generated sequences.

Acceptance: calibration shows ≥0.8 separation (Cohen's d ≥ 1) between real and
random on each metric. Re-run R2-D steering benchmark with this metric on 8
EC classes; report whether ≥3 classes show significant positive shift.

2026-05-07 status: generated lysozyme sequences have now been measured by
Pfam, CLEAN, and Foldseek; see
`Research2/results/ec_metrics/generated_metric_triad_summary_20260507.{json,md}`.
The real-vs-random lysozyme calibration is complete; see
`Research2/results/ec_metrics/ec_metric_calibration_summary_20260507.{json,md}`.
All reported calibration metrics exceed Cohen's d > 1.

### T2-D: Decision gate on R2 viability — [tier:2] [project:R2] [impact:project-defining] [effort:1 hour] [file:none — write conclusions in `Research2/docs/EXPERIMENT_LOG.md`]

After T2-A/B/C complete, evaluate honestly:
- If ≥3 of 8 EC classes show significant positive steering shift on the
  metric triad: continue to Tier 3 (drug design + circuits).
- If 0–2 classes: the FVU=0.33 / dead=67% ceiling is binding. Pivot:
  (a) Train a stronger CLT (JumpReLU instead of TopK, d_clt=16384, longer
  training, deeper window) before any further steering claims, **or**
  (b) Reframe R2 as an interpretability + layer-map paper for Nat Methods,
  **not** a steering / drug-design paper. Do not publish "steering is hard"
  as a Nature paper.

Acceptance: explicit go/no-go decision recorded in EXPERIMENT_LOG with the
metric numbers that justified it.

---

## Tier 3 — Brilliant + societal-benefit plays (only if Tier 1/2 succeed)

### T3-A: Mechanism-driven therapeutic-strategy prediction — [tier:3] [project:R1] [impact:Nat Med] [effort:~1 month] [file:`Research1/scripts/30_strategy_prediction.py` (new)]

The clinical question is not "is this variant pathogenic" but "what
intervention?". Build a 4-way classifier mapping mechanism → strategy:
- LOF → gene replacement / AAV
- GOF → allele-specific siRNA / ASO
- DN → degrader / interface-disrupting therapy
- Neomorphic → mutation-specific antibody / inhibitor (e.g., KRAS G12C)

Validation: retrospective concordance with published response in disease
cohorts where mechanism-targeted therapy exists (cystic fibrosis CFTR
modulators, KRAS G12C inhibitors, SCN5A mexiletine, BRAF V600E inhibitors).

Acceptance: ≥75% concordance between predicted strategy and the
retrospectively successful intervention on a curated test set of ≥100
variant-treatment pairs from the literature.

### T3-B: Circuit-level enzyme retargeting + wet-lab — [tier:3] [project:R2] [impact:Nature] [effort:~3 months including wet-lab] [file:`Research2/scripts/31_substrate_swap.py` (new)]

Conditional on T2-D = go.

Concept: separate the catalytic-mechanism circuit (Glu/Asp dyad in glycoside
hydrolases) from the substrate-recognition circuit (binding-pocket residues)
in the CLT. Swap the recognition subgraph from β-galactosidase generations
into β-glucosidase generations via targeted feature steering.

Pipeline:
1. Identify catalytic vs recognition circuits using attribution graphs (T2-A).
2. Generate 200 swapped candidates.
3. ESMFold + Foldseek to filter folded ones.
4. Dock cellobiose vs lactose with DiffDock or AutoDock Vina.
5. Top-10 by predicted preference shift → wet-lab partner.
6. Express in *E. coli*, assay both substrates by HPLC.

Acceptance: ≥3/10 swapped variants show ≥10× shift in k_cat/K_M ratio toward
the target substrate while retaining ≥10% of original activity. **This is the
first interpretability-driven enzyme retargeting; one validated hit moves R2
from ICML to Nature.**

Pre-requisite check: identify wet-lab collaborator now. Without one, T3-B is
not viable; reduce R2 ambition to T2-D's "Nat Methods + layer map" outcome.

---

## Two-Week Schedule (rolling Gantt)

| Days | Project | Item | Owner notes |
|------|---------|------|-------------|
| 1–2  | R2 | T0-A hook diagnostic | blocker for everything R2 |
| 1    | R1 | T0-B sign fix + redesign | offline; <1 day |
| 1    | R2 | T0-C pkl audit | <1 day |
| 2–3  | R1 | T0-D protein-level CV | unblocks T1-A |
| 3–5  | R1 | T1-A baselines head-to-head | longest serial dependency |
| 4–7  | R1 | T1-B feature-annotation audit | for Figure 3 |
| 4–7  | R2 | T2-A direct-effect features | conditional on T0-A |
| 5–8  | R1 | T1-D annotation rerun | parallel |
| 7–10 | R2 | T2-B on-manifold steering | conditional on T2-A |
| 8–14 | R1 | T1-C indel benchmark | parallel |
| 10–12| R2 | T2-C metric triad | parallel with T2-B |
| 12–14| R2 | T2-D go/no-go gate | decision point |
| 14   | both | Update PROJECT_STATUS.md | reflect new headlines |

---

## Decision Points

These determine which tier the project ends at.

1. **After T0 (Day 3):** if hook still null under multiplier=10, R2 is paused
   until the architecture is rewritten — do not run T2-A/B/C against a broken
   hook.
2. **After T1-A (Day 5):** if protein-level CV drops mechanism macro-AUC below
   0.7, downgrade R1 headline from "predicts mechanism" to "stratifies a
   subset of mechanisms" and adjust paper claims.
3. **After T2-D (Day 14):** explicit R2 viability decision. If pivoting,
   archive `Research2/scripts/{11,12,14}_*.py` under `legacy/` and rewrite
   the R2 proposal to focus on layer-map + interpretability without steering
   claims.

---

## What NOT to Do

- Do not run more steering experiments before T0-A passes. Bit-identical logits
  mean the existing benchmarks are uninterpretable.
- Do not "fix" ProteinGym by re-tuning the magnitude scorer. The scientific
  story is "SAE measures mechanism, not fitness" — leaning into that is more
  defensible than chasing a marginal improvement.
- Do not add new layers to the SAE training. The current 5-layer set
  (19/23/27/31/35) is sufficient; further layers cost compute without
  unlocking new claims.
- Do not start T3-B (wet-lab) without a confirmed partner and a green T2-D.
- Do not rewrite CLT training during this round. v2 is good enough for T2;
  any retraining decision belongs after T2-D, not before.
