# Pre-registration — the uncertainty audit of context information

**Date:** 2026-08-20 **Status:** frozen before the campaign; asserts no result. **Scope:** the sampling uncertainty of the cohort-power estimand and of every quantity normalised by it. **Authority:** `docs/INTERPRETABILITY_TRANSFER_AUDIT.md` stays canonical for findings, retractions and limitations, and `docs/MEASUREMENTS.md` stays canonical for what each stage measures. This document adds only the analysis plan, the criteria that plan is permitted to decide, and the criteria it is explicitly not permitted to decide. When results exist they are admitted to the audit, not to this file.

**Where the campaign departed from this plan, recorded 2026-08-20 and extended 2026-08-21 (EXP-R2-216).** Three departures. The first two are narrowing and neither is a change made after the numbers existed; the third adds a control the plan did not name. E1 scored **thirteen** arms rather than fifteen: `bygpt5-small-en` and `bygpt5-base-en` were declared in `PANEL` but outside the campaign panel when the campaign ran, so they remain unscored on this estimand anywhere and their absence is not a pass. (They were admitted to the campaign panel on 2026-08-21, which makes them qualifiable in a future campaign and changes nothing about what E1 measured.) And of E3's two negative controls only the within-record token shuffle ran — no stage exposes a randomly-initialised matched-architecture mode, so that control is unrun rather than negative. The third departure is an addition: E3 gained a **unigram-predictor null control**, a synthetic arm whose predictive distribution is the smoothed unigram fitted on another block's held-out reference, so its true `I` is zero by construction and the criteria's error rate at zero is a count rather than an argument. It was built after the shuffle control proved unable to supply a clean zero, uses no separate code path, and changed no real-arm number. Everything else executed as frozen; the results and what they do to the criteria are in the audit's §5.06.

**One reason stated in this plan is wrong, corrected 2026-08-21 and marked here rather than rewritten in place.** The section below refuses the sign criterion because a near-zero `I` would have its sign decided by `α` and `V` — the smoothing bias. The refusal is upheld and measured, at 64 false passes in 104 readings whose true value is zero, but the cause is not smoothing: both terms of the null control carry the same `α` and the same `V` and the inflation largely cancels, while the bootstrap's own resampling of the estimated reference lifts the whole interval above the point estimate by a mean of +0.0185 nats, and that displacement is what the sign rule reads. The two confound because both grow with `V/R`. The measured account is the audit's §5.06(d) and L34; this paragraph records that the plan's reasoning, not its decision, was superseded.

## Why this is frozen before the numbers exist

Two facts make a post-hoc plan untrustworthy here, and both are on record in the code rather than inferred.

The 0.30 nats/token floor was chosen against an observed distribution. `pathways.pathway_metrics` defends it in its own docstring with the sentence "across all 3864 `pathway_metrics` records in `results/`, none has a context information between 0 and 0.30 nats" — which is a statement about where the panel happened to land, not about where the boundary between a usable and an unusable denominator lies. The same docstring records what the previous rule cost: gating the denominator on sign alone admitted a denominator of a hundredth of a nat and produced a share with a median of 3.57 and a 97.5th percentile of 85.4. The floor is a real improvement over that, and it is still a constant selected after seeing which side of it every arm fell on.

Four of the fifteen panel arms have never been through the held-out estimator at all, so the empirical claim the floor rests on covers eleven arms. `progen2-small` has only plug-in figures from the lens stage — +0.829 and +1.150 nats at 32 sequences — and the code marks those `cross_arm_comparable: False` by construction. `bygpt5-medium-en` was qualified at +2.462 nats during EXP-R2-129, but that artefact lives on GPFS and is not retained in this repository. `bygpt5-small-en` and `bygpt5-base-en` have never been scored anywhere, which the audit already states where it uses them as unqualified supporting rungs.

So the criteria below are declared **screening**, not confirmatory. A screening criterion may keep an arm out of a downstream stage and may say that a denominator is too small to divide by. It may not be cited as evidence that an arm does or does not extract information from context, and no number this campaign produces may be used to re-choose the floor it was measured against.

## The estimand

Context information is `I = H_baseline − H_model` in nats per scored token. `H_model` is the token-weighted mean next-token negative log-likelihood over the scored targets, which is the convention `scoring.per_sequence_scores` states and defends: a 2000-residue record contributes thirty times what a 64-residue one does. `H_baseline` is the cross-entropy of a held-out unigram on those same targets,

`H_baseline = −(1/N) Σ_v c_cohort(v) log q̂(v)`,  `q̂(v) = (r(v) + α) / (R + αV)`,

with `α = 1.0` (`pathways.LAPLACE_SMOOTHING`), `V = arm.model.config.vocab_size`, and `r` the token counts of a held-out reference of about 4000 records checked disjoint from the roughly 200 scored records by exact content match before use.

Two properties of that definition decide the whole analysis. First, it collapses to a mean of per-token contrasts, `I = (1/N) Σ_i log(p_model(t_i) / q̂(t_i))`: the model term and the baseline term are evaluated on the same tokens and are therefore **paired**. Second, `H_baseline` depends on the cohort through `c_cohort`, so resampling the cohort moves both terms. An analysis that holds the baseline fixed while resampling the model term is not producing an interval for the quantity it prints.

## What ships today, and why it is not an interval

`budget.arm_power` publishes `per_sequence_context_information_interval`, which self-declares `is_an_interval_for_context_information_nats: False` and `baseline_uncertainty_included: False`. Both flags are accurate and both were added deliberately. Four separate things are wrong with the object as an interval for `I`, and they do not all point the same way.

It is built on an unweighted mean over sequences of `(fixed baseline − per-sequence clean CE)`, which is a different estimand from the token-weighted headline printed beside it. It is a Student-t interval over records treated as independent, so it ignores near-duplicate structure. It omits reference uncertainty entirely, since the baseline enters as a constant shift applied to every endpoint. And by ignoring the pairing it charges the contrast with cohort-composition variance that cancels in the real estimand. The first three understate the uncertainty; the fourth overstates it. Nothing about the direction of the net error is known.

The estimand mismatch is visible on disk rather than only in argument. Across the 44 cohort-power records retained in `results/`, the headline value falls outside its own printed interval twice, both on `progen2-base`: the corpus-wide skip-0 draw reads a headline of 1.2306 against a printed interval of [1.0035, 1.2112], and the skip-4000 draw reads 1.2367 against [1.0324, 1.2286]. A point estimate outside its own interval is the cleanest possible demonstration that the two are not measuring the same thing.

## The analysis, frozen

A **group-level paired bootstrap**. The resampling unit is the near-duplicate group, never the record, for the reason L30 already records on a different cohort: 871 of 2048 held-out records kept a relative at 95% identity or above, and a record-level unit reports an interval narrower than the evidence supports.

Each iteration does the following, in this order. Resample cohort groups **once** and share that single draw across both the model term and the baseline's cohort counts, because they are paired. Independently resample reference groups. Recompute the smoothed unigram inside the iteration from the resampled reference, so that the smoothing is part of what is being resampled rather than a constant carried in from outside. Then recompute three quantities on that draw: `I` itself; the dimensionless `ρ = I / H_baseline`; and a bits-per-symbol variant whose `symbols_per_token` is recomputed from the resampled cohort rather than reused from the full one. Intervals are percentile intervals over the iterations.

**Resample indices are common across arms**, so that a cross-arm contrast is a paired contrast on one draw rather than a comparison of two independent intervals. Appendix B rule 21 exists because this repository has already had a claim decided by interval overlap and then narrowed by a matched re-run; non-overlap of independently bootstrapped intervals is not a test of difference and will not be used as one. Pairing is only defined within a cohort, and the panel does not share one: the text arms sit on the OpenWebText draw, `protgpt2` and the residue-level ProGen2 arms share one Swiss-Prot draw, and `zymctrl` sits on the EC-labelled draw. Common indices therefore apply within each of those groups, and a contrast that crosses them is reported as unpaired with that fact attached.

The interval is **refused** when the token-weighted Kish effective group count `n_eff = (Σw)² / Σw²`, with `w` the per-group scored-token counts, falls below the package-wide bootstrap-unit floor of 8 (`statistics.MINIMUM_BOOTSTRAP_UNITS`). The floor is applied to `n_eff` rather than to the raw group count, because a grouping whose mass sits in a few large groups has as many effective atoms as its Kish count says and not as many as its group count says. There is no fallback to record-level resampling. A record-level interval is narrowest exactly where group dependence is strongest, so offering it as a fallback would return the most confident-looking number in the case the refusal exists to catch.

Single-linkage connected components on 5-mer containment is the grouping relation, and on a cohort of about 200 records chaining into one giant component is a live risk rather than a theoretical one. The grouping summary already reports `largest_group_share`, and that field, together with `n_groups` and `n_eff`, travels with every interval. Group assignments are not currently persisted on the base cohort path — `arms.Cohort` carries no group field and `01_cohort_power.py` never calls the grouper — so producing them is part of the campaign rather than a read of existing artefacts.

## What this campaign decides, and what it does not

**No new threshold is justified here.** The 0.30 nats/token rule remains the operative gate this round; what changes is that it is reported with an interval instead of as a bare point comparison.

Three quantities are reported alongside it, with their status fixed in advance.

| reported quantity | status this round |
|---|---|
| `I` against the 0.30 nats/token floor, with a group-level paired interval | the operative screening gate |
| sign of `I` (interval lower bound above zero) | non-evidential diagnostic; expected to pass on every arm |
| `ρ = I / H_baseline`, with an interval | reported; no floor chosen, and choosing one is out of scope |

The sign criterion is reported and not adopted, and the reason is worth stating rather than leaving as a preference. `I` carries a smoothing bias that grows with vocabulary size and that no bootstrap can touch, because it is a property of the estimator rather than of the draw. On an arm whose true context information sits near zero, the sign of the measured `I` would be decided by `α` and `V` rather than by the model. A sign test is therefore safe exactly where it is redundant — on the seven text arms reading above +4.4 nats — and unreliable exactly where it would decide something. That is the shape Appendix B rule 2 was written about, one step removed: the criterion is attainable, but attainability is not the same as informativeness.

`ρ` is recorded as the dimensionless successor quantity because it removes the units the floor is expressed in, not because a value for it has been chosen. Any floor on `ρ` would be a new constant chosen against a new observed distribution, which is the failure this document exists to avoid repeating.

**Nats per token is not cross-arm comparable, and no interval repairs that.** A unigram over merged BPE pieces already encodes the character-level dependencies inside each piece, so per-symbol baseline entropy is far lower for a BPE arm than for a byte-level one, while per-symbol model cross-entropy is close to segmentation-invariant. `I` measured under two tokenizers is therefore two different estimands — in each case "information beyond a unigram over *this* inventory" — and their difference is partly a property of the inventories. Converting nats per token to bits per symbol fixes the units and does not fix this: the conversion is invariant, the choice of tokenizer is not. Cross-arm statements this round are restricted to arms sharing a tokenisation regime, or are made on `ρ` with the caveat carried.

## Downstream ratios

Every pathway share of the form `R_b = Δ_b / I_b` is recomputed **jointly inside each bootstrap iteration**, from that iteration's own numerator and denominator. It is never formed by dividing two separately computed intervals, which would discard the correlation between them and is not an interval for a ratio in any case.

A ratio is published only when the Fieller precondition `g = (z · SE(Î) / Î)² < 0.05` holds on the denominator. Where it fails, the ratio is refused with "denominator not identified away from zero", and the unnormalised `Δ` and `I` are reported instead so that nothing is lost but the division. This is the same disease the sign guard had, priced correctly: a denominator whose interval approaches zero produces a ratio with no upper bound, and reporting it with a wide interval is not more honest than refusing it.

## The measurement campaign

Five experiments, in this order. E1 is a prerequisite for everything else because no retained artefact holds the per-record statistics the bootstrap needs, and none holds the reference token count `R` that the smoothing bound `log(1 + αV/R)` is computed from.

**E1 — score all fifteen arms, retaining per-record sufficient statistics.** Including the four that have never been through the held-out estimator. Retain per-record scored-token counts and summed log-likelihoods, per-record target count vectors, the reference count vector with its total, and the near-duplicate group assignment for both cohort and reference. This is what turns the audit into a re-analysis of stored statistics rather than a re-run of the models.

**E2 — K ≥ 8 disjoint cohort blocks per arm.** Block-selection error is empirically larger than within-cohort sampling error and is currently unquantified. The evidence on disk is two K=2 pairs that disagree with each other about its size. The instrument pair, seeded within a head-of-file pool of 4000 records, moves ProtGPT2 by +0.599 nats between blocks and `progen2-base` by +0.231, about 22% of its own value. The corpus-wide pair, seeded over the whole corpus as Appendix B rule 1 requires, moves the same two arms by +0.012 and +0.006. Both pairs are `--cohort-skip 0` against `--cohort-skip 4000`, and both shift the scored cohort and the held-out reference together, so neither is a clean cohort effect and neither may be quoted as an interval. K ≥ 8 is what turns the between-block variance from a one-degree-of-freedom quantity into an estimate.

**E3 — negative controls with known-small true `I`.** A within-record token-shuffled cohort, which destroys context while preserving the unigram exactly, and a randomly initialised model at matched architecture. No arm has ever exercised these criteria anywhere near their boundary: the closest measurable arm sits at +1.06 and the nearest thing to a boundary case is `dialogpt-small` at −4.08, four nats on the wrong side. A gate whose behaviour near its own threshold has never been observed is a gate whose behaviour near its own threshold is unknown.

**E4 — an α-sweep, to measure the smoothing bias per arm.** The direction is settled: smoothing inflates the baseline, the inflation is bounded by `log(1 + αV/R)`, and it increases with `V`, which is why it falls unequally on a 32-symbol arm and a 50257-piece one. The magnitude is not settled. The figures quoted in `pathways.LAPLACE_SMOOTHING` — +0.224 nats at `V = 50257` and +0.0001 at `V = 32` on a 100k-token reference, and the sweep +0.224 / +0.176 / +0.194 / +0.770 across `α = 1, 0.5, 0.1, 1/V` — are attributed to matched Dirichlet simulations that no code in this repository reproduces, and an independent reproduction was not obtained. They are treated here as **unmeasured** and are to be replaced by this sweep. `pathways.smoothing_diagnostics` already computes the closed-form parts and `SMOOTHING_SWEEP` already declares the ladder; what is missing is that no retained cohort-power artefact carries either, or carries `R`.

**E5 — a leakage-removed sensitivity arm.** Re-score with every reference record that shares a 30-mer with any cohort record dropped. Exact-content disjointness is what `pathways.assert_disjoint` checks, and it does not reach a reference record that is a near-copy of a scored one. Such a record makes `q̂` fit the cohort better than a held-out unigram should, which lowers `H_baseline` and deflates `I` — unequally across arms, since the protein cohorts carry far more near-duplicate structure than the text ones.

## Limitations declared in advance

The bootstrap covers within-cohort and within-reference sampling error and nothing else. Block-selection error is estimated separately by E2 and is not folded into the interval, because a variance component estimated from K blocks does not belong inside a percentile interval computed from one.

Leakage between cohort and reference at the 30-mer scale is measured by E5 and is not removed from the headline. The headline stays on the declared reference, and the sensitivity arm sits beside it.

The interval is an interval for the **smoothed, plug-in estimand** `E[Ĥ_baseline^{α,V}] − H_model`, not for true information beyond a context-free model. The gap between them is the smoothing bias E4 measures. **No silent bias correction will be applied**: an unsmoothed held-out unigram does not exist — one unseen target makes it infinite — so the constant cannot be eliminated, only swept and reported.

The grouping relation is a proxy. Single-linkage components on 5-mer containment at a threshold of 0.5 were calibrated against DIAMOND identity at the 95% boundary, which is a different instrument from the one that decides whether two records are exchangeable for this estimand.

With K = 2 the between-block variance has one degree of freedom, which is the whole reason E2 raises K rather than reusing what is on disk.

## One consequence that does not wait for the campaign

**The ordering of `progen2-base` and `progen2-medium` by context information is not supported and must not be reported.** Four readings of the same gap exist on disk, and one of the four has the opposite sign:

| draw | `progen2-base` | `progen2-medium` | gap |
|---|---:|---:|---:|
| instrument, skip 0 | 1.0604 | 1.1070 | +0.047 |
| instrument, skip 4000 | 1.2909 | 1.2674 | **−0.024** |
| corpus-wide, skip 0 | 1.2306 | 1.3079 | +0.077 |
| corpus-wide, skip 4000 | 1.2367 | 1.3317 | +0.095 |

The two campaigns also disagree about the arms' levels by more than three times the gap being read — `progen2-base` moves 1.06 to 1.23 and `progen2-medium` 1.11 to 1.31 between them, at the same skip — and at the time of writing no reading of any of the four carried a valid interval. This is recorded now, as a retraction candidate against whatever text cites that ordering, because it follows from artefacts already on disk and does not depend on anything this campaign will produce.

It also fixes which figures are the current ones. The values most often quoted for these arms, +1.06 and +1.11, come from the instrument campaign seeded within a head-of-file pool; the corpus-wide campaign that Appendix B rule 1 asks for reads +1.23 and +1.31 on the same arms. Any statement of an arm's context information must name the draw it came from until E1 replaces both.

**How this consequence fared, recorded 2026-08-21 and not folded into the text above.** The conclusion holds and two of the three things this section says about it do not. Readings of the gap now carry intervals, so "no reading carries a valid interval" is spent — E1 gave the pair eight paired corpus-wide readings, and once the analysis stopped keying pairing on the producing invocation (L38) none of the eight straddles zero. Four readings were also not the whole record: more than twenty paired readings of the same gap sit on disk across six stages, and the plan's table happened to hold the four that came from one stage. What replaces the four-reading argument is stronger than it was, because the sign of the gap turns out to track the cohort the gap is read on rather than the two checkpoints — consistently negative on the plain Swiss-Prot held-out draw and positive on the EC-labelled one. `docs/INTERPRETABILITY_TRANSFER_AUDIT.md` §5.06(c) is the authority on that and carries the readings; the ordering remains unreportable, and the same superseded sentence is still quoted verbatim in the stage's `DECLARED_CONTRASTS` note.
