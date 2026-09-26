# The ProLLaMA transplant pilot: what it measured, and why it is follow-up work

This pilot asked which part of the continued-pretraining step between Llama-2-7B and ProLLaMA Stage 1 is enough, on its own, to carry that step's mutation-ranking gain. It ran to completion and **does not enter Paper 1**. Condition 1 is met, condition 2 fails on the effect, and the specificity control refutes the property a localisation claim would need. The outcome is a design for follow-up work, and the most useful thing the pilot produced is not a localisation but a refutation of the assumption a localisation of this kind rests on.

[D2_MECHANISTIC_INTERFACE.md](D2_MECHANISTIC_INTERFACE.md) owns the endpoint contract and the refusals every cell passed through; [D2_PROLLAMA_CAPABILITY_LOCALIZATION.md](D2_PROLLAMA_CAPABILITY_LOCALIZATION.md) owns the transplant design and its partition; [D1_MATCHED_PAIR_HANDOFF.md](D1_MATCHED_PAIR_HANDOFF.md) owns the pair. This document owns the pilot's measurements and its verdict.

## What was measured, and on what

The endpoint is the within-assay Spearman correlation between the native mutant-minus-wild-type summed target log likelihood and the measured effect, averaged within wild-type family at 50% identity and then equally over families. The support is the frozen 201-assay, 163-family, 25,728-variant anchor, reached by restricting each cell's per-variant records from the lineage's own 211-assay native support: containment is exact at assay and variant level, with every mutation digest and per-assay count matching the cohort, so the restriction needed no forward pass and the result composes directly with the Direction-1 capability reading.

Twenty-six cells ran on four cards between 11:11:38Z and 15:47:42Z on 2026-09-25, all twenty-six exiting cleanly with no failures. Scoping was the declared depth-quarter partition — attention and MLP crossed with four depth quarters, plus the embedding and the head, ten groups that differ between the checkpoints — rather than a sweep over 32 blocks. That choice is recorded rather than assumed: the readout depth reassessment names **zero** seed-consistent rows for this lineage, because its representation increment resolves at no split seed over the qualified control and the position axis is undefined for a multi-residue interface, and scoping a likelihood endpoint by where a fitted readout over representations resolves would apply a representation criterion to a quantity that has no folds, no projection and no readout class.

The scale every effect is read against is the distance between an untouched recipient and a fully reconstructed donor:

**ceiling − floor = +0.15298874 [0.121882, 0.184077]** in within-assay Spearman.

The floor reproduces the parent's admitted anchor figure of −0.0096077001 to 1.7e-18 and the ceiling ProLLaMA Stage 1's +0.1433810357 to 2.8e-17, both inside their admitted intervals, with per-tensor reconstruction bit-identical over all 291 tensors the runtime serves and the 32 stored rotary buffers skipped and byte-identical. The ceiling recovers exactly 1.0 [1.0, 1.0] of its own gap, which is the check that the scale is well formed.

## The result in full

Each cell installs one group's donor tensors into the recipient and re-measures. The contrast is against the floor; the share is that contrast over the ceiling-minus-floor gap. Intervals are 95% percentile family bootstraps at 2,000 draws, reported at the admitted seed and identical in point estimate at all three declared seeds.

| Group | Parameters | Endpoint | Contrast against the floor | Share of the gap |
| --- | ---: | ---: | --- | ---: |
| `head` | 131,072,000 | +0.006469 | +0.016077 [−0.00067, +0.03392] | 0.105 |
| `mlp_q3` | 1,082,130,432 | +0.006012 | +0.015620 [−0.00096, +0.03205] | 0.102 |
| `mlp_q1` | 1,082,130,432 | +0.001379 | +0.010987 [−0.00613, +0.02882] | 0.072 |
| `attention_q1` | 536,870,912 | +0.000495 | +0.010103 [−0.00742, +0.02736] | 0.066 |
| `mlp_q2` | 1,082,130,432 | +0.000007 | +0.009615 [−0.00611, +0.02573] | 0.063 |
| `attention_q3` | 536,870,912 | −0.002507 | +0.007100 [−0.00637, +0.02050] | 0.046 |
| `attention_q4` | 536,870,912 | −0.005465 | +0.004143 [−0.00494, +0.01354] | 0.027 |
| `embedding` | 131,072,000 | −0.005751 | +0.003857 [−0.00524, +0.01302] | 0.025 |
| `attention_q2` | 536,870,912 | −0.007115 | +0.002493 [−0.01181, +0.01730] | 0.016 |
| `mlp_q4` | 1,082,130,432 | −0.021461 | **−0.011853** [−0.03253, +0.00963] | **−0.078** |

**Not one of the ten resolves above zero**, at any of the three bootstrap seeds. The two largest, `head` and `mlp_q3`, miss by less than a thousandth of a Spearman unit at the lower endpoint. `mlp_q4` moves the endpoint the wrong way. The ten shares sum to **0.445** against the full transplant's 1.0, and by component they divide almost evenly — attention 0.156, MLP 0.159, embedding and head together 0.130 — which is the profile of a gain distributed across the checkpoint rather than one seated in any part of it.

## The two controls, and the inversion

**The null control fixes replay uncertainty at exactly zero.** Copying `attention_q1` from the recipient into itself — every step a treatment cell performs, the same reads, the same copies, the same live-tensor verification — returned a contrast of **0.000000 [0.000000, 0.000000]** and per-assay correlations **bit-identical to the floor's**. The intervention machinery contributes nothing of its own, so any nonzero treatment effect is above replay uncertainty by construction rather than by comparison. This is the one cell exempt from the byte-identity refusal, and the exemption does not weaken that refusal for any treatment cell: the refusal exists to stop a no-op being reported as a localisation, not to stop a no-op being reported as a no-op.

**The specificity control inverted the expectation, and that is the pilot's result.** The scrambled cell installs the donor's 32 `attention_q1` tensors at a seeded derangement of same-shape destinations — no fixed point, every slot verified against the donor digest of the tensor it received, permutation seed 20260925. It changes exactly as many parameters as the correctly assigned `attention_q1` cell, from the same donor, and destroys only the assignment of values to slots. It returns

**+0.032158 [+0.00442, +0.06121], a share of 0.210.**

It is the **only** non-ceiling cell in the pilot whose interval excludes zero, and it is **larger than every correctly assigned group**, including correctly assigned `attention_q1` at +0.010103 [−0.00742, +0.02736]. The anticipated asymmetry did not occur: the scramble moved the endpoint **up** rather than down, and of the ten treatment cells only `mlp_q4` opposed its direction.

So on this endpoint, at this granularity, the measured response is to installing donor-derived attention parameters **at all** — not to installing them in their correct destinations. A derangement within the quarter preserves whatever property drives the movement. That refutes assignment specificity directly, which is a stronger and more useful outcome than any comparison of magnitudes would have been, and it is visible only because the control was built as a real permutation of destinations rather than as a re-scoring of the same assignment.

## The three conditions

**Condition 1, seed stability: met.** Every treatment cell's point estimate is identical across the three declared bootstrap seeds, and each cell's resolution against the floor is the same at all three. Split-seed stability is **undefined** for this endpoint — a raw likelihood enters as one scalar column, so there is no fitted readout and no split seed — and is recorded as undefined rather than left to read as checked.

**Condition 2, the effect above replay uncertainty: not met.** The uncertainty term is satisfied, at exactly zero. The effect is not: ten of ten cells cross zero at every seed. The specificity control is reported beside it and is not a bar the effect had to clear; a treatment smaller in absolute value than the scramble's excursion would not have been disqualified by it, and none was.

**Condition 3, mechanistic localisation: not read.** The stop rule holds at the first genuine failure. The localisation profile is recorded for a later reader, and no cutoff was added to make the third condition automatic, because how much concentration counts as a mechanism is a judgement on the profile rather than a threshold belonging in a script.

## The non-claim

**This does not show that the ProLLaMA gain is unlocalisable.** It shows that on this support, at this power, **no single declared group carries a resolved share of it**, and that **the one intervention which did resolve had its assignment destroyed**. Those are statements about ten parameter groups, one endpoint, 163 resampling units and one pair of checkpoints. A gain distributed across a checkpoint is exactly what a continued-pretraining step might be expected to produce, and a distributed gain is not an absent one.

## The methodological caution, which generalises past this pair

A parameter transplant of this kind **does not test assignment specificity unless a within-group derangement is run alongside it.** The two interventions install the same parameters, from the same donor, in the same quantity, and differ only in destination; without the derangement there is nothing in the design that distinguishes "these values in their own places carry the behaviour" from "values with these statistics anywhere in this group move the endpoint". Anyone attempting localisation by transplant needs that control, and a positive result reported without it is uninterpretable — not weakly supported, but uninterpretable, because the experiment does not separate the two hypotheses. This pilot would have reported `head` and `mlp_q3` as the leading candidates and said nothing false; the derangement is what showed that the leading candidates were not the interesting fact.

## What would make the question answerable

Four routes exist, and they are not equally promising.

**A spectrum-matched random control — run, and it answers the question.** See the section below: the donor's content is the operative thing and its scale and spectrum are not, so single-group transplant is **not** closed as a localisation route and the pilot's failure is one of granularity and power. That route was the cheapest decisive measurement and it has been spent.

**A complement cell for necessity** is the obvious next step and I expect it to be uninformative here. With the gain distributed at roughly 0.1 per group, removing one group leaves nine, and each complement should sit near the ceiling with differences inside the interval width the single-group cells already failed to clear. Ten complements cost about 9.4 GPU-hours to learn that.

**Finer groups than depth quarters** run the wrong way. A quarter spans eight layers; a per-layer partition divides each effect by about eight while leaving the interval width unchanged, so groups that fail to resolve at quarter granularity will fail more clearly at layer granularity. This is not a power problem that finer cuts solve.

**More power** would need a different label source, not a different analysis. Resolving a contrast of +0.016 against a half-width of about 0.018 requires roughly four times the effective units, about 650 families against the anchor's 163, and the whole local ProteinGym panel supplies 174. It is not reachable from this cohort.

**A different endpoint** is possible in principle and constrained in practice for this lineage: its rendering is multi-residue at 1.30 to 1.79 residues per token, so no position-resolved endpoint is defined for it, which is the same property that kept it out of the depth selection.

## The spectrum-matched control: content is the operative thing, not scale

The scramble left two readings open, with opposite consequences. Either the movement was driven by a *distributional* property of the donor's parameters — their scale and spectrum, which a derangement preserves because it keeps each tensor intact and only moves it — or by their learned *content*, which a derangement also preserves. Separating them needs an intervention that keeps the distribution and destroys the content.

Four further cells ran on 2026-09-25 between 16:24:08Z and 17:22:08Z, all four exiting cleanly: a re-derived floor and ceiling, and two spectrum-matched cells. The construction takes each selected tensor's singular value decomposition from the donor, keeps the singular values exactly, and replaces **both** orthogonal factors with random ones under declared seed 20260926. The result carries the donor's Frobenius norm, operator norm, effective rank and full singular-value spectrum, and no learned direction at all. Verification is spectral rather than by digest, because a synthesised tensor has no precomputed digest to match: the realised spectra departed from the donor's by at most **1.3e-05** relative on `attention_q1` and **9.6e-06** on `mlp_q3`, against a declared tolerance of 1e-4, and every synthesised slot's digest matched neither checkpoint while every tensor outside the selection still carried the recipient's.

The floor and ceiling were re-derived in this dispatch because it carries a different code digest from the pilot's, and they reproduce the admitted anchor figures at **exactly the agreement the pilot's did** — 1.735e-18 and 2.776e-17 — so both arms' ceiling-minus-floor scale is +0.15298874 [0.121882, 0.184077] to floating point and the shares below are directly comparable across the two dispatches. That equality is what admits the comparison; without it the cross-arm reading would have been refused.

| Intervention on `attention_q1` | Content | Spectrum | Contrast against the floor | Share |
| --- | --- | --- | --- | ---: |
| correct assignment | donor's, in place | donor's | +0.010103 [−0.00742, +0.02736] | 0.066 |
| within-group derangement | donor's, moved | donor's | **+0.032158 [+0.00442, +0.06121]** | **0.210** |
| spectrum-matched | destroyed | donor's, exactly | **−0.000982 [−0.03072, +0.02903]** | **−0.006** |
| null, copied from itself | recipient's | recipient's | 0.000000 [0.000000, 0.000000] | 0.000 |

| Intervention on `mlp_q3` | Content | Spectrum | Contrast against the floor | Share |
| --- | --- | --- | --- | ---: |
| correct assignment | donor's, in place | donor's | +0.015620 [−0.00096, +0.03205] | 0.102 |
| spectrum-matched | destroyed | donor's, exactly | **+0.005487 [−0.01724, +0.02783]** | **0.036** |

**The scale-and-spectrum reading is refuted.** Both spectrum-matched cells sit at zero — −0.006 and 0.036 of the gap, each interval comfortably containing it — while both content-bearing interventions on the same groups move the endpoint away from it. Installing the donor's exact spectrum with random directions recovers nothing. So single-group transplant on this endpoint is **not** measuring a scale effect, localisation by this route is **not** closed at any granularity, and the pilot's failure is diagnosed as one of granularity and power rather than of the method. That is the more tractable of the two readings, and it is the one the measurement supports.

**What the control does not explain, and must not be read as explaining.** The derangement still recovers 0.210 against the correct assignment's 0.066 — more from moving the donor's tensors than from leaving them in place. The spectrum control rules out the scale explanation for that; it offers no account of the ordering itself. And the ordering is **not established**: the two intervals, [−0.00742, +0.02736] and [+0.00442, +0.06121], overlap over most of their length, so the difference between them is not resolved even though one resolves above zero and the other does not. With thirteen cells measured and no multiplicity correction across them, a single resolved interval is weak evidence standing alone. The derangement result is an anomaly for the next study to reproduce before anything is built on it, not a finding to carry forward.

**This does not reopen integration.** The ten single-group treatment cells still cross zero at every bootstrap seed, so condition 2 still fails and the pilot still does not enter Paper 1. A control interprets a negative; it cannot convert one. What has changed is that the negative now has a diagnosed cause — insufficient granularity and power against a distributed gain whose content does matter — rather than an open question between that and a closed method. The conditions record was not re-run against these cells and its verdict is unchanged.

## Scope, carried with every number above

Every treatment cell tests **sufficiency in this context** — against this recipient, on this support, under this scoring. **Necessity is untested**: no complement cell was in this budget, and neither claim may be read off the other. Recipient and donor differ by a whole training run, fixing a corpus, a schedule, a data order and a token budget at once, so a positive result would have localised **where that training's effect lands in the parameters, not an architectural locus**. **None of this identifies a circuit**, and a recovered share is a parameter-space quantity rather than a mechanism. Scoring ran at batch size one, where a batch-composition repeat check is the same single-row computation as the production forward, so no cell here is numerically validated on that basis.

## A defect that would have inverted the verdict

The analysis stage first classified a cell as a control **by its label**, and the two control cells are labelled `null_attention_q1` and `scrambled_attention_q1`. Under that rule both were typed as treatments. The consequence is worth stating plainly rather than leaving to inference: the null's exact zero would have entered the treatment set as a cell failing to resolve, and **the scramble's resolved +0.032158 would have entered it as a cell succeeding** — the only one. The pilot would then have reported a resolved, localised effect in `attention_q1` that does not exist, and condition 2 would have read as met on the strength of the control that refutes it. Both stages now determine a cell's kind from the `control` field in its own receipt, which the runner records per cell. The defect was found and fixed before any verdict was read, and that is the only reason the verdict can be trusted.

Two earlier defects, both found the same way — by running the interface's own refusals against real cells rather than against fixtures — are recorded in [D2_MECHANISTIC_INTERFACE.md](D2_MECHANISTIC_INTERFACE.md): a reconstruction check that demanded a live digest for tensors the serving runtime cannot hold, and so refused a floor that does reconstruct the recipient exactly; and a contrast that crashed below the eight-unit resampling floor instead of reporting its degeneracy.

## Limitations

The 56-family screen arm is a support-sensitivity arm and its numbers are screening numbers on 56 of 163 resampling units, never pooled with the full arm and never a filter on which cells were confirmed. Its ordering differs from the full arm's — `mlp_q3` at 0.224, `mlp_q2` at 0.197, `mlp_q1` at 0.172 — and its ten shares sum to 0.988 against the full arm's 0.445, which is what per-cell estimates look like when the unit count falls by two thirds and is a further reason not to read a localisation off single-group shares at either support.

Intervals are unadjusted pointwise percentile intervals over one biological cohort, conditional on these two checkpoints; they omit training-seed and cohort-selection variation and are not corrected across the ten cells, so the absence of a resolved cell is not a multiplicity-corrected null. The wild-type clustering at 50% identity does not guarantee remote-superfamily disjointness or exclude pretraining overlap. The scramble is one derangement under one declared seed on one group: it establishes that *this* permutation preserves the effect, not that every permutation would, and a second seed was not run. The spectrum-matched control is likewise one draw of random orthogonal factors under one seed, on two groups; a second draw was not run, and its near-zero readings are two cells rather than a distribution. The null control was run on `attention_q1` only, so replay uncertainty is measured for that group's intervention path and inherited by the others. `norm` and `rope` are byte-identical across the two checkpoints and were not run, because a zero contrast from an identity is not evidence that a group is insensitive.

## Artefacts and entry points

The pilot's cells ran from frozen snapshot `20260925040213_03cf80612952` under campaign `campaign_d2_pilot_anchor`, manifest sha256 `f5b7ae6e068c66f45c941d7eb6da19b78ca12ff44ac7af0de55eee423f4bc9ea`; the spectrum-matched control and its own ends ran from `20260925091824_ea392e417132` under `campaign_d2_spectrum_control`. The spectrum construction is `capability_transplant.transplant_spectrum_matched` under declared seed 20260926. The two control transplants are `capability_transplant.transplant_null` and `capability_transplant.transplant_scrambled`; the restriction and the endpoint are `scripts/transfer/analyse_d2_pilot_anchor.py`; the conditions are `scripts/transfer/assess_d2_pilot_conditions.py`, which stops at the first failure and declines condition 3's verdict. Tests are `tests/test_transplant_controls.py`, `tests/test_d2_pilot_anchor.py` and `tests/test_d2_pilot_conditions.py`.

Outputs are under ignored `logs/d2_prollama_pilot_20260925/`:

| Artefact | sha256 |
| --- | --- |
| `containment.json` | `6a93b542bc08b414d51d444696e84bd1862a3dc7267d75b6cb42a50829e10279` |
| `dispatch_declaration.json` | `5a4fadf821327f7cfd9d415f9a6ff6328151a09a359e365a47d9dde9996b611d` |
| `anchor_analysis_full.json` | `bab3be5c66fcc68dad411277d32bd0ceaef9df2ebb947261942eb025df30ee1a` |
| `anchor_analysis_screen.json` | `3f32f845b02f9fa0fd3d3a9e63238d108aa3f78012d9c4fc468e0dc226f0996b` |
| `conditions.json` | `4faa012b57215130a1a5fcd0d5fb485a72ec67e30db95b934321465dfb6485d2` |
| `anchor_analysis_spectrum.json` | `926511fa198ff990a2f97c053f866efd751e4f7a374f41292e9c02d0e93f1c05` |
