# Does frozen model output predict measured double-mutant nonadditivity beyond sequence-statistical controls?

This is the executed answer to the question [the pairwise protocol](D1_INFORMATION_HIERARCHY_PROTOCOL.md) pre-registers, on the support [the baseline readiness record](D1_PAIRWISE_BASELINE_READINESS.md) froze and with the labels [the label-instrument qualification](D1_PAIRWISE_LABEL_INSTRUMENT.md) admitted. The endpoint is the wild-type-centred cycle epsilon = y_AB − y_A − y_B + y_WT on the combined MegaScale `dG_ML` scale in kcal/mol. The primary quantity is the paired reduction in group-equal epsilon mean squared error in squared kcal/mol when a frozen model quantity is added to a matched control set, with the site pair as the weighted unit inside an equally weighted held group.

The protocol and the readiness record own the design, the grouping contract and the cohort. This document owns what the executed comparison measured.

## Frozen inputs

The roster is the complete pre-declared unconditioned Readout panel: 33 checkpoints, declared in `src/transfer/pairwise_epistasis.py` before any pairwise scoring and not a function of any Readout or pairwise outcome. ZymCTRL is excluded because it is a biologically conditioned interface on its own 40-assay support with no unconditional substitute. Eleven arms carry residue-level amino-acid tokenisation, three are byte-level and nineteen are BPE; those three strata are reported separately throughout.

| Frozen input | Digest |
| --- | --- |
| Cohort `logs/d1_pairwise_cohort_20260924/cohort.json` | `8133463ec30293013864685a426a99b69c51c1bac6774b37ebd036915670040f` |
| Grouping contract `data/pairwise_assets/grouping_contract_20260924.json` | `6a0a76a4ad1f083f7506f8fa29cb809b555fcf64ccbd722972de877cf26759c4` |
| Pairwise baseline Q `baseline_q.json` | `518986756d54bccd0ce9f1287fe3aa734e9ef5555d5cf5005518b10612d9c03b` |
| Label-free extraction plan, content digest | `e338420f5df70143ccfc8d16ec5479a67b330ec35d36ea7c5965ea31a59e179c` |
| Per-background independent-site profiles | `331a9399abd715ef14543e9d30242d34afdfda78767a6948205a73cc0ad55db4` |

The extraction plan carries sequences and cycle state indices only, and the extraction entry point refuses a plan carrying measurement fields, so the 428,241 production forward passes and 6,336 repeat-check forwards were label-blind by construction rather than by convention. Rebuilt from the cohort file, the plan reproduces the frozen workload exactly: 64 backgrounds covering 64 of the 101 natural groups, 8,192 cycles, 12,977 deduplicated sequences, 779,664 residues, lengths 37–72, 217 site pairs.

### Precision, and what the batch gate does and does not establish

The whole panel was extracted at batch size one — bfloat16 for the two ProGen3 arms, whose expert mixture reduces over the flattened batch and which have no kernel above float16, and float32 with TF32 disabled for the other 31. Batch composition is the only numerical failure the Readout expansion actually measured, at 0.0015 to 0.0152 nats and up to 0.0054 relative L2 on five arms at batch eight, and batch size one removes it for every arm.

The consequence must be stated plainly: **the 0.001-nat and 0.001-relative-L2 gates are inapplicable at this setting, not passed.** At batch size one the production forward and the repeat-check forward are the same single-row computation over the same string, so a zero difference is structural and certifies nothing about accuracy for unchecked sequences or against a higher-precision reference. What the repeat column does measure is run-to-run reproducibility of that identical computation, and it is small: at most 1.53e-5 nats on the summed residue-span log likelihood (rita-xl) and at most 8.40e-7 relative L2 on the projected four-block contrast (galactica-30b), over 33 arms and 64 backgrounds each. No higher-precision reference is available through the frozen loader for the literal-string text arms, and none exists at all for ProGen3.

Measured throughput, from a label-blind qualification on three backgrounds per arm, was 4.6 sequences per second for galactica-30b to 183 for gpt2. The full panel took about 4.7 GPU-hours of forwards across eleven H200 GPUs, against the protocol's provisional 10–25 GPU-hour planning range.

### One contaminated state class in the frozen cohort

A defect in the cohort reader was found after this panel was fitted. `pairwise_stability.build_cohort` keyed sequence states on `aa_seq` alone, and an insertion or deletion construct has its `aa_seq` truncated to the wild type's length in the pinned files, so such a construct is length-matched to the substitution states and entered the substitution support as though it were one. The reader now refuses any row whose `mut_type` folds to an `ins` or `del` prefix; on the pinned bytes that removes 53,441 of the 575,331 otherwise-admitted rows, and it carries 735 of the 115,546 length-matched states of these 64 backgrounds, 722 of them exclusively.

The reach in this cohort is small and, more usefully, of two different kinds. **5 of 12,977 sequences carry an insertion-construct row, 3 of them exclusively**, across 3 of 64 backgrounds — `1O6X.pdb`, `2AMI.pdb` and `2D1U.pdb` — touching 6 of 217 site pairs. Counting those as a single contaminated-cycle total would overstate the damage, because the two kinds are not equivalent:

| Effect | Extent |
| --- | --- |
| Cycles that **lose a required state**, because a state rests only on insertion-construct rows | **3** — one per background, all at positions (1,2) in the 1–2 stratum, and each the only cycle of its site pair, so 217 site pairs become 214 and the 1–2 stratum loses 3 of its 25 pairs |
| Cycles that **keep all four states with the target displaced**, because a retained median pooled an insertion-construct row | **127**, all in `2AMI.pdb` and all at separation ≥10, on its three site pairs (10,70), (14,63) and (14,66), displaced by exactly **0.0315 kcal/mol** through its wild-type median |

No group is lost by the honest accounting: `2AMI.pdb` keeps 127 of its 128 cycles. A displacement of 0.0315 kcal/mol on 127 cycles is small against an additive null of 1.15441 [0.92605, 1.39050] kcal²/mol², and it is common to every design and every arm, so it cannot manufacture an arm-specific increment. The structure-contact gate measured a complementary cut of the same defect: all 6 affected site pairs fall in its contact stratum.

The cohort was deliberately **not** rebuilt or re-digested, because three concurrent gates reference `8133463e…` and changing it would invalidate their declared supports. Rebuilding it under the fixed reader in a scratch location, purely to measure the support change, keeps the same 64 backgrounds and 64 groups and the same 8,192 cycles, and moves 12,977 to 12,973 sequences, 779,664 to 779,376 residues and 217 to 214 site pairs; the separation strata move from 819/2,341/5,032 to 816/2,342/5,034 cycles, and the 1–2 stratum's group coverage falls from 24 to 21. No background is dropped or added. The sensitivities actually reported below are restricted-evaluation ones, which leave the fits untouched.

## What was compared

The target is the cohort's epsilon. Primary prediction receives sequences and frozen sequence-derived features only; measured wild-type and single-mutant stabilities enter the target and the nuisance control's training stage, never a held-out group's feature scaling, penalty selection or calibration.

| Block | Width | Content |
| --- | ---: | --- |
| `ident` | 400 | directed substitution counts over the cycle's two substitutions |
| `geom` | 10 | relative positions, separation, separation stratum, length |
| `comp` | 40 | wild-type and double-mutant amino-acid composition |
| `local` | 400 | dipeptide frequency difference, double minus wild type |
| `prof` | 14 | mutation-local independent-site profile summaries and an availability indicator |
| `G` | 5 | the nonlinear-additive nuisance: calibrated cycle, additive level, three predicted states |
| `T` | 16 | per-state pooled token counts, their cycle, position shifts, segmentation changes |
| `Q` | 1 | the fitted pairwise four-state energy contrast |
| `M` | 1 | the model likelihood interaction M_AB − M_A − M_B + M_WT, nats |
| `M1` | 2 | the two constituent first-order likelihood differences |
| `R` | 1024 | the projected representation interaction contrast, four blocks of 256 |
| `R1` | 2048 | the two constituent first-order projected representation differences |

C is `ident+geom+comp+local+prof` at 864 coordinates. The control chain is C ⊂ C+G ⊂ C+G+T ⊂ C+G+T+Q, with C+G+T+M1 and C+G+T+R1 as the first-order controls over the matched baseline C+G+T. Representation pools, the 256-coordinate Gaussian projection and its seeds 20260923–20260926 are the admitted Readout declarations, unchanged.

Five outer and four inner held-group folds, outer seeds 20260923, 20260924 and 20260925, inner seed the outer seed plus 100 plus the zero-based outer-fold index. Groups are weighted equally, site pairs equally inside a group and cycles equally inside a site pair. The ridge recipe, its penalty grid {0.01, 0.1, 1, 10, 100} and its tie rule toward stronger regularisation are the admitted Readout ones. Intervals are 2,000-draw group-bootstrap percentile intervals conditional on the fitted cross-validation predictions.

Matched support forces the declared split. Additive, nonlinear-additive, tokenisation and model increments run on all 64 groups, 8,192 cycles and 217 site pairs; the Q-inclusive comparison runs on 45 groups, 5,760 cycles and 163 site pairs, and Q is never compared against anything on unmatched support.

### The conditions that had to hold, measured

The sequence/profile control C and the nonlinear-additive control C+G contain no arm-specific column, so their group-equal squared error must be identical for every arm at one seed. Across all 33 arms the spread is **0.0 exactly** at every seed, on both supports: C reads 0.62823, 0.63440 and 0.63732 kcal²/mol² and C+G reads 0.62607, 0.63273 and 0.63625. Every arm shares one row-identity digest and one fold-identity digest per support and seed.

The row-identity digest that proves two fits used the same rows is now rendered through Python `float` rather than left as a NumPy scalar. `repr(np.float64(0.5))` is `0.5` under NumPy 1.26 and `np.float64(0.5)` under NumPy 2, so a NumPy-scalar rendering hashes identical rows to different values across interpreters — which would fail spuriously on identical data and invite relaxing the very check that would otherwise catch a real mismatch. The exposure here is between interpreters inside a pod, not between the pod and the workstation: `requirements.txt` pins `numpy==1.26.4`, the staged runtime these cells used carries 1.26.4, and the pod's default interpreter, which the environment script otherwise selects, carries NumPy 2.1.2 on Python 3.12.7. The fixed rendering reproduces the NumPy 1.26 value exactly, so the recorded digests are unchanged at `11a289e319558f199d5c084be54cd074dc64217f31618737319b60fcad438e81` on the 64-group support and `7438d73004542fb90ec2c984581394ce9bed2535aa78032cea46f0c2b4fd9804` on the Q support.

Permuting the held-out group labels of one outer fold on the real cohort — 1,664 cycle labels and 2,625 absolute-stability labels — moved no held-out prediction of any of the 16 designs by any amount, at a maximum absolute change of 0.0, while the permuted target itself changed. The uncalibrated additive cycle inside the nuisance control reads its construction zero at 6.66e-16, and the independent-site double contrast reads its construction zero at 2.84e-14 on both supports; neither is replaced by a manufactured comparator, and no Spearman coefficient is computed for either.

## The control ladder

On 64 groups the zero-interaction additive null has group-equal mean squared error **1.15441 [0.92605, 1.39050] kcal²/mol²** and an undefined Spearman on all 64 groups, reported as undefined rather than as zero.

| Step | Seed 20260923 | 20260924 | 20260925 |
| --- | --- | --- | --- |
| C over the additive null | +0.52618 [+0.37549, +0.66745] | +0.52001 | +0.51709 |
| C+G over C | +0.00215 [−0.00084, +0.00535] | +0.00167 | +0.00107 |
| C+G+T over C+G, 31 arms | −0.00044 [−0.00354, +0.00249] | −0.00045 | −0.00147 |
| C+G+T over C+G, ProtGPT2 | +0.00393 [−0.00129, +0.00896] | +0.00367 | +0.00336 |

All entries are paired reductions in group-equal mean squared error in kcal²/mol². The sequence/profile control accounts for 45.6% of the additive null's squared error and reaches a within-background Spearman of +0.2869 [+0.2193, +0.3537]. Everything measured after it is two orders of magnitude smaller.

The nonlinear-additive nuisance is positive at every seed and its interval crosses zero at every seed. Its strength is bounded by how well absolute stability transfers across held-out families: over 15 outer folds the first stage reaches a group-equal held-out weighted R² with median 0.141 and range −0.172 to 0.262, and a held-out Spearman with median 0.489 and range 0.181 to 0.604, while the fitted isotonic response spans a median 2.60 and range 1.88–3.93 kcal/mol — comparable to the 2.53 and 2.44 kcal/mol the label instrument measured in sample. So the mandatory global-response control is implemented and measurable but weak on this endpoint, and a model increment reported over it is adjusted for only the part of the global response that cross-family sequence/profile prediction can reach.

The tokenisation descriptors are a near-null block for 31 arms, which is what a residue-level or fixed-segmentation interface implies. ProtGPT2 is the single exception, and it is the one arm whose native FASTA line wrapping changes segmentation between the four states.

## Adding model likelihood

Over the matched baseline C+G+T on 64 groups, the panel's paired reduction has median −0.00005 and seed-mean range −0.00325 to +0.00139 kcal²/mol². Three of 33 arms have all three per-seed intervals above zero:

| Arm | Stratum | 20260923 | 20260924 | 20260925 |
| --- | --- | --- | --- | --- |
| progen3-3b | amino acid | +0.00126 [+0.00046, +0.00214] | +0.00116 [+0.00034, +0.00204] | +0.00125 [+0.00045, +0.00213] |
| prollama | amino acid | +0.00100 [+0.00036, +0.00173] | +0.00096 [+0.00033, +0.00164] | +0.00108 [+0.00042, +0.00180] |
| protgpt2 | BPE | +0.00085 [+0.00041, +0.00130] | +0.00086 [+0.00038, +0.00133] | +0.00104 [+0.00056, +0.00153] |

Two arms have all three intervals below zero. By stratum the seed-mean medians are +0.00002 for amino-acid arms with 6 of 11 positive, +0.00009 for byte arms with 2 of 3 positive, and −0.00009 for BPE arms with 6 of 19 positive. The largest seed-mean point estimate on the panel is proteinglm-7b-clm at +0.00139, whose seed-20260923 interval includes zero.

These increments are 0.07% to 0.11% of the additive null's squared error and 0.16% to 0.24% of what C alone removes. The corresponding within-background Spearman increments are +0.0016 to +0.0052.

## Adding the representation contrast

**No arm, on either support, has all three per-seed intervals above zero for the representation interaction contrast.** Over C+G+T on 64 groups the panel median is −0.00720 kcal²/mol² and eight arms have all three intervals below zero. The stratum medians are −0.01678 for amino-acid arms with 0 of 11 positive, −0.00417 for byte arms with 0 of 3, and −0.00627 for BPE arms with 3 of 19. The most negative arms are the protein ones: progen3-3b at −0.01926 [−0.03495, −0.00418] and proteinglm-7b-clm at −0.02598 [−0.04265, −0.00962] at seed 20260923, with within-background Spearman falling by 0.062 to 0.122. The largest positive point estimate is gpt2-large at +0.00571 [−0.00132, +0.01262], which does not hold across seeds (+0.00077 at 20260924).

A 1,024-coordinate block fitted on 64 held groups is the binding constraint here. The negative increments are the signature of a compressed linear readout that the ridge penalty cannot shrink enough — 43 of 360 penalty selections sit at the grid's strongest value of 100 — not a measurement that representations carry no information about this endpoint. This is the Readout protocol's declared bound on the readout class, now visible as a cost on a 64-group panel.

## The first-order controls, which is where the model-side signal actually sits

The protocol requires comparing the interaction contrast against its constituent first-order differences, to separate an added contrast from simply supplying richer single-mutant information. That comparison changes the reading of this experiment.

The first-order likelihood differences alone, added to C+G+T with no interaction term, give increments 3 to 8 times larger than the interaction contrast, on five arms whose three per-seed intervals all exclude zero and all of which are protein arms:

| Arm | 20260923 | 20260924 | 20260925 | Distance ≥10, seed 20260923 |
| --- | --- | --- | --- | --- |
| proteinglm-7b-clm | +0.00812 [+0.00373, +0.01287] | +0.00770 | +0.00808 | +0.00989 [+0.00399, +0.01734] |
| progen2-xlarge | +0.00700 [+0.00331, +0.01115] | +0.00674 | +0.00774 | +0.00926 [+0.00431, +0.01489] |
| progen2-medium | +0.00426 [+0.00116, +0.00760] | +0.00338 | +0.00457 | +0.00655 [+0.00285, +0.01075] |
| progen3-3b | +0.00391 [+0.00164, +0.00652] | +0.00386 | +0.00378 | +0.00380 [+0.00086, +0.00715] |
| progen2-large | +0.00349 [+0.00074, +0.00647] | +0.00288 | +0.00389 | +0.00544 [+0.00209, +0.00930] |

By stratum the seed-mean medians are +0.00109 for amino-acid arms with 10 of 11 positive, −0.00012 for byte arms and −0.00026 for BPE arms. The increment is as large or larger on the distance ≥10 stratum, which is the non-overlapping-window support the protocol requires a beyond-local claim to appear on.

Once that first-order control is in the baseline, the likelihood interaction adds +0.00046 to +0.00101 kcal²/mol² on the same three arms as before — progen3-3b, prollama and protgpt2 — and nothing distinguishable from zero on the other 30. The first-order representation differences alone are negative for 15 arms and positive without interval separation for two (progen2-large at +0.00904 and progen3-3b at +0.00432 seed-mean); adding the representation interaction over them leaves a panel median of −0.00126 and no arm above zero at all three seeds.

Both counts in this section are the ones the indel defect measured above moves: excluding the 3 cycles that lose a required state reduces the first-order arms from 5 to 3 and the arms surviving their own first-order control from 3 to 1, in each case through one seed's lower bound crossing zero by less than 0.00023 kcal²/mol². The sensitivity section carries both readings and identifies the three emptied (1,2) site pairs as the cause.

## The Q-inclusive comparison, on its own 45 groups

On 45 groups, 5,760 cycles and 163 site pairs, the additive null reads 1.24959 [0.98154, 1.55394] kcal²/mol² and C+G+T reads 0.66170. Baseline Q's own increment over the matched baseline is +0.00578 [+0.00004, +0.01269] at seed 20260923, −0.00323 [−0.03625, +0.02782] at 20260924 and +0.00770 [−0.00109, +0.01741] at 20260925, with within-background Spearman moving from +0.2992 [+0.2310, +0.3696] to +0.3159 [+0.2457, +0.3884]. So the admitted pairwise sequence baseline is a usable comparator that itself adds little to this endpoint, and one of the three splits reverses its sign.

After Q, one arm of 33 has all three per-seed intervals above zero for the likelihood interaction: prollama at +0.00299 [+0.00080, +0.00575], +0.00190 [+0.00020, +0.00384] and +0.00224 [+0.00081, +0.00377]. The amino-acid stratum median is +0.00195 with 9 of 11 arms positive, against −0.00007 with 6 of 19 positive for BPE arms. For the representation contrast after Q, no arm has all three intervals above zero and the panel median is −0.00344.

## Sensitivity to the contaminated cycles

The whole panel was refitted twice from the same extraction artifacts, once with an evaluation dropping only the **3** cycles that lose a required state, and once dropping all **130** cycles the defect touches at all. The second is not the honest accounting — it also discards 127 cycles that keep every state — but it brackets the first from the pessimistic side, so the two together bound the correction.

The fits behind both readings are the published ones. Aggregating a refit's unrestricted evaluation reproduces the published panel report **byte for byte**, at SHA-256 `65b5cc3cdf654ed523d97aa6c9e2c540246f3d7f7f8e48571a39419391d43ba0`; only the evaluation set differs, so every design and every arm is still compared on identical rows and folds, and the arm-independent controls C and C+G still read a spread of 0.0 exactly across all 33 arms at every seed under both readings.

| Reading | Support, all groups | Support, Q groups |
| --- | --- | --- |
| Published | 64 groups, 8,192 cycles, 217 site pairs, 129.2 effective | 45 groups, 5,760 cycles, 163 site pairs, 96.1 effective |
| Drop the 3 lost-state cycles | 64 groups, 8,189 cycles, 214 site pairs, 127.5 effective | 45 groups, 5,757 cycles, 160 site pairs, 94.2 effective |
| Drop all 130 touched cycles | 63 groups, 8,062 cycles, 211 site pairs | 44 groups, 5,630 cycles, 157 site pairs |

Effective counts are Kish counts of the site-pair weights the fits actually apply. The honest reading loses no group; the pessimistic one loses group `nat-014`, whose only background is `2AMI.pdb`.

**The four conclusions the verdict rests on are unchanged under both readings.**

| Conclusion | Published | Drop 3 | Drop 130 |
| --- | --- | --- | --- |
| Representation contrast over C+G+T resolves above zero | no arm of 33; median −0.00720 | no arm; −0.00801 | no arm; −0.00816 |
| Representation contrast after Q resolves above zero | no arm of 33; median −0.00344 | no arm; −0.00165 | no arm; −0.00224 |
| Likelihood interaction over C+G+T resolves above zero | 3 arms: progen3-3b, prollama, protgpt2 | the same 3 | the same 3 |
| Likelihood interaction after Q resolves above zero | 1 arm: prollama | the same 1 | the same 1 |
| C over the additive null resolves above zero | all 33; +0.52109 | all 33; +0.53035 | all 33; +0.53613 |
| Nonlinear-additive nuisance over C resolves above zero | no arm; +0.00163 | no arm; +0.00198 | no arm; +0.00189 |

The three retained likelihood arms move by at most 0.00022 kcal²/mol². Under the honest reading progen3-3b goes from +0.00126/+0.00116/+0.00125 to +0.00107/+0.00093/+0.00106, prollama from +0.00100/+0.00096/+0.00108 to +0.00091/+0.00088/+0.00100, protgpt2 from +0.00085/+0.00086/+0.00104 to +0.00077/+0.00075/+0.00098, and prollama after Q from +0.00299/+0.00190/+0.00224 to +0.00285/+0.00174/+0.00216. Every interval still excludes zero at every seed.

**Two secondary counts do change, identically under both readings, and the targeted reading identifies why.**

The first-order likelihood differences alone resolve above zero at all three seeds for **3 arms rather than 5**: proteinglm-7b-clm, progen2-xlarge and progen3-3b survive; progen2-medium and progen2-large do not. The likelihood interaction over its own first-order control resolves above zero for **1 arm rather than 3**: prollama survives; progen3-3b and protgpt2 do not. Both changes appear already when only the 3 lost-state cycles are dropped, so they are caused by three groups each losing one of their equally weighted site pairs — a support change of 217 to 214 pairs and 129.2 to 127.5 effective — and not by the 0.0315 kcal/mol displacement on the 127 retained cycles.

In all four cases the loss is one seed's lower bound crossing zero by less than 0.00023 kcal²/mol², at seed 20260924 every time, while the other two seeds stay resolved and every point estimate stays positive:

| Contrast and arm | Seed 20260924, published | Seed 20260924, drop 3 |
| --- | --- | --- |
| M1 alone, progen2-medium | +0.00338 [+0.00014, +0.00675] | +0.00313 [−0.00005, +0.00659] |
| M1 alone, progen2-large | +0.00288 [+0.00012, +0.00597] | +0.00272 [−0.00005, +0.00574] |
| +M over C+G+T+M1, progen3-3b | +0.00080 [+0.00010, +0.00155] | +0.00062 [−0.00007, +0.00138] |
| +M over C+G+T+M1, protgpt2 | +0.00046 [+0.00000, +0.00089] | +0.00038 [−0.00008, +0.00081] |

ProtGPT2's published lower bound at that seed was already exactly +0.00000. So the all-three-seed criterion is brittle in this regime rather than the effects being unstable: the surviving first-order arms span +0.00349 to +0.00802 in point estimates under the targeted reading against +0.00288 to +0.00812 across all five arms under the published one, and no sign changes anywhere. Both counts are reported; the restricted ones are the conservative reading, and the statement that the reproducible model-side signal is concentrated in single-mutant likelihood scores of residue-level protein arms holds under all three.

Both sensitivities still leave the affected states in **training**, including `2AMI.pdb`'s displaced wild-type median, because the fits are deliberately untouched. A training-side effect is shared by every design and every arm on identical rows, so it cannot manufacture an arm-specific increment, but a full refit on a rebuilt cohort is deferred rather than done, and that deferral is a limitation rather than a result.

## Verdict for the pairwise rung

**The pairwise boundary is not crossed by the representation contrast on any arm, and is crossed only marginally, by one arm, for the likelihood interaction.** On the 45-group Q-inclusive support the single arm whose three per-seed intervals all exclude zero after Q is prollama, at +0.0019 to +0.0030 kcal²/mol², which is 0.15% to 0.24% of that support's additive-null squared error. With 33 arms, two supports, three split seeds and ten reported contrasts of unadjusted pointwise intervals, one arm clearing is consistent with multiplicity, and this result must not be read as an established beyond-pairwise increment.

**The weaker rung below it is crossed, and reproducibly.** Beyond additive, nonlinear-additive, sequence/profile and tokenisation controls on all 64 groups, three arms add likelihood-interaction information at every split seed (progen3-3b, prollama, protgpt2, +0.0003 to +0.0021 kcal²/mol²), and protein arms add first-order likelihood information at every split seed and on the distant stratum — five of them on the published evaluation (proteinglm-7b-clm, progen2-xlarge, progen2-medium, progen2-large, progen3-3b, +0.0001 to +0.0173 kcal²/mol²) and the three largest of those once the indel-affected cycles are excluded. The reproducible model-side signal on this endpoint is therefore concentrated in single-mutant likelihood scores of residue-level protein arms, not in a four-state interaction contrast, and that is exactly the alternative the protocol's first-order control exists to expose. That statement holds under both evaluations; the arm count behind it does not.

Every increment measured here is small against the endpoint's scale. The sequence/profile control removes 45.6% of the additive null's squared error; the largest reproducible model increment anywhere in the panel is 1.5% of what that control removes.

## The equal-capacity depth contrast on this cohort

The same measurement the folding-stability and remote-homology gates report, run here through this gate's own plan, cohort, baseline Q, profiles, nested folds, control ladder and `nested_compare` / `evaluate` pair, with only the representation block replaced: one hidden block's `mean` and `last` summaries, each projected to 256 coordinates under the declared depth-projection rule and concatenated to **512**, substituted for the admitted `projected` states. Seven arms, 45 blocks, three split seeds, 135 fits, no failures, support `all`. **Four qualifications travel with every number**: one axis; unadjusted for multiplicity; 512 coordinates against the published **1,024**, so **not comparable to the published increment**; and everything but the representation block is this gate's own.

Counts name their population: **seven arms are measured and five produce a comparison**, because ProGen3-112M and ProGen3-3B were selected at a depth the admitted extraction already hooked, which yields no selected-minus-admitted difference.

| Arm | Admitted depths | Selected | Selected mean | Best admitted | Difference | Seeds resolved |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ProGen2-large | 15, 31 | 25 | −0.000652 | −0.000648 | −0.000004 | 0/3 |
| ProGen2-medium | 13, 26 | 23 | −0.006142 | −0.002705 | −0.003438 | 1/3 |
| ProGen2-xlarge | 15, 31 | 28 | −0.005907 | +0.001998 | −0.007905 | 2/3 |
| ProGen3-112M | 4, 9 | 9 | +0.005863 | +0.005863 | — (selected is admitted) | 1/3 |
| ProGen3-3B | 11, 23 | 11 | −0.007474 | −0.001896 | — (selected is admitted) | 2/3 |
| ProteinGLM-7B-CLM | 17, 35 | 30 | −0.013578 | −0.004481 | −0.009098 | 3/3 |
| ProtGPT2 | 17, 35 | 21 | +0.002083 | −0.000942 | **+0.003026** | 0/3 |

Points are group-equal held-out squared-error reductions in kcal²/mol² over this gate's fullest control ladder; the three narrower R-contrasts are in the artefact and agree in sign.

**The depth-invariance control on this cohort is a derived one and is labelled as such.** The other two cohorts checksum their baseline predictions across an arm's depths; this gate's adapter did not, so that count was never measured here. What is recoverable from the persisted artefacts is the same property by another route: every contrast whose augmented design does not carry R must be unchanged across depths, and every depth must carry the same fold identity. Measured after the fact: **2,025 contrast comparisons at a largest absolute difference of exactly 0.0, and 135 fold-identity comparisons, with no mismatch of either kind**.

**The selected depth beats both admitted depths on one of the five comparison arms**, ProtGPT2, and falls short on four. That matches remote homology's one of five and not folding stability's four of five, so across the three cohorts the readout panel's selection transfers on one and not on the other two, and **no arm is above on all three**.

**One feature of this cohort is worth separating from the depth question.** Most increments here are **negative** at most depths: on this endpoint the 512-coordinate representation generally *worsens* group-equal held-out error rather than reducing it, at admitted and selected depths alike. That is a statement about this endpoint at this capacity — the published 1,024-coordinate result is not affected by it and is not comparable to it — and it means the differences above are mostly between quantities that sit at or below zero, which is why the resolved-seed column matters more than the ranking.

## Limitations that bound this result

- The nonlinear-additive nuisance is weak on this endpoint because absolute stability transfers poorly across held-out families: median held-out weighted R² 0.141 over 15 folds. The label instrument measured roughly half of epsilon's variance as a global response to the *measured* additive prediction; the primary task forbids measured singles as predictor inputs, so G absorbs only the part of that response reachable from cross-family sequence/profile prediction. A model increment over G is adjusted for that part and no more. The singles-assisted variant, which would supply measured singles to every comparator under a separately declared label budget, was not executed.
- The representation readout is bounded by its compression. Four pools projected to 256 coordinates each is 1,024 columns on 64 held groups, and the ridge grid runs out of shrinkage in 43 of 360 selections. A negative representation increment here bounds this compressed linear readout, not the information in the states. Full-width per-state block outputs are retained on the cluster so a different projection can be replayed without model inference.
- Power is set by 217 site pairs on the 64-group support and 163 on the Q support, not by the 8,192 and 5,760 cycle counts. No interval here treats cycles as independent. The effective site-pair count depends on which weighting it is computed under. Under the weighting these fits apply — groups equal, site pairs equal inside a group — it is **129.2 and 96.1**, which is the convention that bounds these intervals; the [readiness record](D1_PAIRWISE_BASELINE_READINESS.md) carries the authoritative statement of both conventions, which governs, and how they relate.
- 99.98% of admitted cycles sit on site pairs the source study selected as suspected interactions, so neither the frequency nor the magnitude of nonadditivity here estimates a population value over protein position pairs, and the separation strata are sequence-distance strata rather than a contact contrast.
- The cohort is conditioned on resolvability in all four states, so cycles whose double mutant falls below the assay's stability floor are absent.
- The frozen cohort was built by a reader that length-matched insertion and deletion constructs into the substitution support. The reader is fixed; the cohort was not rebuilt, because three concurrent gates reference its digest. The reach is 3 cycles that lose a required state, all at positions (1,2) and each the only cycle of its site pair, and 127 cycles at separation ≥10 in one background whose target is displaced by 0.0315 kcal/mol. Excluding them from the evaluation leaves every headline conclusion unchanged and reduces two secondary arm counts through the three emptied site pairs. The affected states remain in training under both sensitivities, so a full refit on a rebuilt cohort is owed.
- Held groups are conservative controls, not certified remote-family holdouts: homology below 30% identity or 80% mutual coverage is undetected, 179 of 478 backgrounds carry no Pfam label, and two backgrounds the catalogue labels natural are designed mini-proteins. One of those, `5UP5.pdb`, retrieves no UniRef50 hit and therefore carries the profile-availability indicator at zero rather than an imputed profile.
- Baseline Q's couplings are prior-dominated at a median 2.91 effective sequences per site, and it is admissible on 45 of 64 groups. On the other 19 the pairwise boundary stays unresolved; a model increment on the 45-group subset says nothing about them, and no weaker estimator was substituted there.
- The 0.001 batch gates are inapplicable at batch size one. Run-to-run reproducibility of the identical single-row computation is at most 1.53e-5 nats and 8.40e-7 relative L2, and that is not an accuracy statement. ProGen3 has no kernel above float16, so no full-precision reference exists for it.
- Intervals condition on the fitted cross-validation predictions, omit training and split variation, and are not adjusted for the panel, the supports, the seeds or the contrasts. Model pretraining exposure to these 64 backgrounds cannot be matched retrospectively and is not excluded.
- A positive increment bounds accessible predictive information under this fitted readout. It does not locate a pairwise mechanism, and it is not evidence of encoded biological knowledge.

## Artifacts

Per-arm extraction manifests, per-background arrays, the 33 per-arm fit records and both sets of 33 restricted-evaluation records are retained on the cluster under `results/pairwise_epistasis_20260924/`; the aggregated panel reports, the affected-cycle map, the scratch fixed-reader cohort and the local verifications are under ignored `logs/d1_pairwise_epistasis_20260924/`. Entry points are `scripts/transfer/build_pairwise_extraction_plan.py`, `build_pairwise_profile_features.py`, `extract_pairwise_epistasis.py`, `fit_pairwise_epistasis.py` and `analyse_pairwise_epistasis.py`, with the declarations, feature blocks, nuisance control and nested comparison in `src/transfer/pairwise_epistasis.py` and the conditions that must hold in `tests/test_pairwise_epistasis.py`.
