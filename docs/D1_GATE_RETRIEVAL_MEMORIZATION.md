# Do the model-side gains survive stratification that removes retrieval and memorization?

This gate is qualified on its own. It asks one question of two already-established gains: when the evaluation units are split by how much retrieved homolog support their wild type carries, does the gain concentrate where that support is dense and vanish where it is thin, or does it hold across the split? The first pattern is a retrieval explanation of the gain. The second is not.

The two gains at stake are the [first-order likelihood gain on measured double-mutant stability](D1_PAIRWISE_EPISTASIS_RESULTS.md) and the [profile-residual representation gain on mutation ranking](D1_HIERARCHY_CROSSED_CONTROLS.md). Nothing here re-opens either finding's own verdict, and a negative result at this gate would not settle any other gate in the capability map.

## How a stratum estimate is formed

Neither gain is refitted. A stratum's estimate is the same paired statistic over the same fitted held-out predictions, taken over the subset of resampling units the stratum holds, with the same folds, the same split seeds, the same weighting, the same label budget, the same 2,000-draw bootstrap and the same bootstrap seed as the study it reads. Because a stratification only removes whole resampling units, every retained unit keeps its internal weighting, so the estimate on the whole support is the unit-count-weighted mean of its strata. That identity is computed rather than assumed: the largest absolute decomposition residual over every reported stratification, arm, contrast, seed and support is 1.11 × 10⁻¹⁶, in squared kcal/mol on the stability side and in dimensionless Spearman on the ranking side.

The stability gain needed one re-run, because the admitted fit records report group-equal squared error and its paired reductions but not the per-group values a stratification consumes. `scripts/transfer/fit_retrieval_strata_pairwise.py` re-runs the admitted fit of one arm and keeps them. It is refused unless its realised row identity, fold identity, support counts, design widths, selected ridge penalties and every whole-support increment agree with that arm's admitted record; across the 33 arms the largest disagreement on any of the 42 whole-support increments each arm reports is 7.0 × 10⁻¹⁶ squared kcal/mol, which is floating-point noise, so the per-group values decompose the published number rather than a new one. The ranking gain needed no re-run at all: the crossed-control cell reports retain per-assay increments, and re-aggregating them over the same wild-type-family bootstrap reproduces every published summary point exactly.

Two supports are reported for every stability stratum, in a fixed order. The unfiltered one is the admitted cohort exactly as fitted, and it is the support the reproduction gate runs on, because that is what establishes that this is the admitted estimator. The declared insertion/deletion exclusion is then applied on top, as a filter on the evaluation of the same fitted predictions. The filter is never applied before the gate and the gate was never relaxed to accommodate it.

## The declared strata

Six stratifications were declared and frozen before any stratified estimate was computed. No band edge is a free parameter, because a free parameter on a stratum boundary is a place where an outcome could enter the design: the identity bands and the near-duplicate edge are imported from the frozen declaration in `src/transfer/homology.py`, the depth bands are decades of a coordinate both profile stores already retain, and the family bands are group size under each cohort's own frozen grouping contract.

| Stratification | Quantity | Bands |
| --- | --- | --- |
| `identity_band` | maximum percent identity over the query of any retrieved corpus hit | `<30`, `30–70`, `70–95`, `≥95`, at the frozen edges 0/30/70/95/100 |
| `near_duplicate` | the same quantity at the frozen 95% edge | present, absent |
| `depth_band` | reweighted alignment depth Neff of the fitted independent-site profile | no homolog support, `<10`, `10–100`, `100–1000`, `≥1000` |
| `coarse_depth` | the same quantity at the 100-effective-sequence decade | shallow, deep |
| `family_band` | member count of the unit's group under the cohort's frozen grouping contract | remote singleton, close multi-member |
| `source_provenance` | presence of the `Tsuboyama_2023` source token in any of the unit's assay identifiers | shared source, other source |

Stratum assignment reads retrieved-homolog statistics and frozen group membership only. Replacing every measured mutation effect in the ranking cohort with arbitrary values leaves every band of both cohorts byte-identical, which is the tested form of that claim rather than its assertion.

The source-provenance axis is a source-independence sensitivity and not a retrieval test, and it is reported as its own axis for that reason: folding it into an identity or family band would attribute a provenance effect to retrieval. It exists because the measured-stability cohort is drawn entirely from the Tsuboyama 2023 MegaScale release and 64 of the ranking anchor's 201 assays carry the same source, so the two gains are not independent biological confirmations of one another.

### Frozen inputs and their digests

| Input | Digest |
| --- | --- |
| Strata declaration, content | `cb56f5fe6fabd66176c5ca0928fcb718c27e4ff3bb2de819bd920148c2636681` |
| Strata declaration, file | `ca23bcf7905c8b29d0b8dab6826bf12f5926ceba0f84922d2efb0784a7d5ba01` |
| Indel exclusion, content | `326aaa49eb99cc06112992f5cd2796d79b425200b7b5c9561261aa7354c852d9` |
| Stability hit table, 478 queries, 618,403 hits | `1aa8ed8addd6e0f851919e451ed4c9daefefb35e79663421695029615b2e4d78` |
| Ranking hit table, 187 queries, 354,762 hits | `a024592e34f193e9b6ebae809ba805c45c668690e394d8c39a639abdbd0e8b5d` |
| UniRef50 DIAMOND index, 60,315,044 clusters | `88a6c44bda8f1d0936bd7a084dca602ec562d8f4dd49c648705b011769612d2c` |
| DIAMOND 2.1.24 binary | `5d04a9ecbbceea5d1a6496070155012c2538788def552f85e0d65095a532a5f5` |
| Grouping contract, MegaScale backgrounds | `6a0a76a4ad1f083f7506f8fa29cb809b555fcf64ccbd722972de877cf26759c4` |
| Ranking cohort | `4093ac34368cd9e7be02e5c84a22655ce78cb9bb4e4d58d626f281868dbe7992` |
| Stability cohort | `8133463ec30293013864685a426a99b69c51c1bac6774b37ebd036915670040f` |
| Label-free extraction plan, content | `e338420f5df70143ccfc8d16ec5479a67b330ec35d36ea7c5965ea31a59e179c` |

The two searches differ in one setting that bounds their top band: the stability search reported at most 10,000 hits per query and the ranking search at most 5,000, so `max_identity_over_query` is a maximum over reported hits for the 3 stability backgrounds and 53 ranking assays that reached those ceilings, not a maximum over the corpus.

### What the declared bands hold

The stability cohort's resampling unit is the held family group, one background each, with the site pair carrying the weight inside a group and the cycle never entering as a sample size. The ranking cohort's unit is the wild-type family at 50% identity and 80% coverage of the shorter sequence, with the assay carrying the weight inside a family. Two Kish effective counts are reported per stratum because they answer different questions: one is the Kish count of the weights the estimator applies, and one is the convention the pairwise readiness record uses, which measures how unevenly the fitted rows spread over the dependence blocks. The estimator-weight convention returns **129.2** effective site pairs on the whole 64-group support against its 217 nominal ones, and the cycle-share convention the readiness record quotes returns **121.6**; both are reported for every stratum so the two records reconcile, and the tables below lead with the estimator-weight one because that is the weighting the fits apply.

| Stratification | Band | Stability groups | Site pairs | Kish site pairs, estimator weights | Kish site pairs, cycle share | Anchor clusters | Assays | Kish assays, estimator weights |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| whole support | — | 64 | 217 | 129.2 | 121.6 | 163 | 201 | 179.6 |
| `near_duplicate` | no near-duplicate retrieved | 15 | 42 | 24.4 | 23.8 | 23 | 27 | 25.2 |
| `near_duplicate` | near-duplicate retrieved | 49 | 175 | 106.9 | 99.1 | 140 | 174 | 154.4 |
| `identity_band` | identity < 30% | 1 | 2 | 2.0 | 2.0 | 0 | 0 | 0.0 |
| `identity_band` | identity 30–70% | 2 | 5 | 3.2 | 3.2 | 5 | 6 | 5.6 |
| `identity_band` | identity 70–95% | 12 | 35 | 19.2 | 18.7 | 18 | 21 | 19.6 |
| `identity_band` | identity ≥ 95% | 49 | 175 | 106.9 | 99.1 | 140 | 174 | 154.4 |
| `coarse_depth` | Neff < 100 | 37 | 118 | 73.4 | 70.7 | 85 | 104 | 93.0 |
| `coarse_depth` | Neff ≥ 100 | 27 | 99 | 55.8 | 50.9 | 78 | 97 | 86.6 |
| `depth_band` | no homolog support | 1 | 2 | 2.0 | 2.0 | 0 | 0 | 0.0 |
| `depth_band` | Neff < 10 | 11 | 34 | 22.6 | 22.3 | 21 | 22 | 21.5 |
| `depth_band` | Neff 10–100 | 25 | 82 | 48.8 | 46.5 | 64 | 82 | 71.7 |
| `depth_band` | Neff 100–1000 | 21 | 75 | 43.9 | 40.0 | 65 | 78 | 70.5 |
| `depth_band` | Neff ≥ 1000 | 6 | 24 | 12.0 | 10.9 | 13 | 19 | 16.4 |
| `family_band` | group of one | 36 | 137 | 84.0 | 77.3 | 152 | 175 | 162.4 |
| `family_band` | group of two or more | 28 | 80 | 48.2 | 46.3 | 11 | 26 | 21.4 |
| `source_provenance` | other source | 0 | 0 | 0.0 | 0.0 | 99 | 134 | 115.2 |
| `source_provenance` | shared source | 64 | 217 | 129.2 | 121.6 | 64 | 67 | 65.2 |

Both cohorts are dominated by units whose wild type retrieves a near-duplicate: 49 of 64 groups and 140 of 163 clusters sit at or above 95% identity over the query. That is the central limitation of this gate and it is stated here rather than in a closing footnote. The three remote identity bands on the stability side hold 1, 2 and 12 groups and the two lowest on the ranking side hold 0 and 5 clusters, so the remote-identity end of the identity stratification cannot clear the package's eight-unit percentile-interval floor and carries no verdict in either direction. What is genuinely testable is the binary near-duplicate contrast, at 15 units on stability and 23 on ranking, and the depth and family contrasts, which are better balanced.

### The remote-identity limitation is not intrinsic, and its route is measured

The strata above cannot resolve the remote-identity end because this cohort has 1, 2 and 12 groups in its three lowest identity bands, not because remote-identity stability measurements do not exist. A concurrent acquisition has measured the alternative directly, with this gate's own search settings against the same retained UniRef50 index, so the number is comparable rather than merely encouraging: over 320 MGnify-flagged backgrounds of the Cho and Tsuboyama 2026 release sampled at seed 20260924, the identity bands read **40 / 146 / 100 / 34** across `<30` / `30–70` / `70–95` / `≥95`, all 40 in the lowest band returning no hit at all, so **186 of 320 backgrounds sit below 70% identity, a share of 58.1% with a Wilson interval of [52.7%, 63.4%]**; and under the frozen grouping contract those backgrounds form **192 groups of which 105 are entirely remote**. Against the eight-unit percentile floor, 105 remote groups is a resolvable stratum where 1, 2 and 12 are not.

That instrument was cross-checked against this one in the same pass. Searching all 478 MegaScale wild types, the 239 bare-PDB-named ones band as 2 / 3 / 77 / 157, which reproduces the 1 / 2 / 12 / 49 this gate measures on its 64-group one-background-per-group subset, while the other 239 band as 132 / 16 / 32 / 59 with 132 returning no hit.

The cost is in the labels, and it is large enough to govern how any verdict on that support must be stated. MGnify's two protease channels disagree far more than MegaScale's: over 1,413,334 admitted rows the mean offset between them is **−0.478 kcal/mol** and the root-mean-square difference **0.824 kcal/mol**, at a Pearson correlation of 0.857, against a combined-ΔG standard deviation of 1.251 kcal/mol. The measurement floor is therefore about two thirds of the endpoint's own spread, and a systematic offset makes the two channels non-interchangeable rather than merely noisy. This gate's admitted endpoint sits far below that: per-channel discordance 0.2337 [0.2209, 0.2458] kcal/mol against a shared component of 0.9013. A null on the MGnify support could therefore be a floor artifact rather than an absence of retrieval-independent signal, so a remote-stratum verdict there has to be stated against that floor and not against this cohort's.

The re-estimation is recorded here as the identified route rather than executed, because it is not a refit. This gate re-reads retained predictions; a remote stratum on that support needs a new four-state cohort, a new label-free extraction plan and a fresh label-blind forward pass of the panel over new sequences, which is a measurement campaign of the same order as the original extraction rather than a stratification of it. Executing it would need its own declaration with its own digest, its own noise floor measured from those two channels, and the frozen strata here left untouched, since they are digest-bound and other work references them.

### The indel exclusion, and the two mixed-provenance units

The frozen stability cohort was admitted before an insertion/deletion defect in the source measurement files was found. An indel construct's source sequence is truncated to the wild type's length, so such a row matches a four-state substitution cycle on sequence alone. `scripts/transfer/declare_indel_exclusion.py` names the affected cycles once from the pinned measurement files' `mut_type` column and the cohort's own retained source-row indices, reading no measured value: 130 of 8,192 cycles in 3 of 64 backgrounds, of which `2AMI.pdb` loses all 128 of its cycles because its wild-type state rests on a single insertion construct row. The filtered support is therefore 63 groups, 211 site pairs, 8,062 cycles and 119.3 effective site pairs.

A concurrent and more precise accounting of the same defect resolves those 130 cycles into two kinds: 3 lose a required state, each the only cycle of its site pair, and the other 127 keep all four states with the target displaced by 0.0315 kcal/mol through `2AMI.pdb`'s wild-type median. The exclusion declared here is the conservative drop-all reading, and its support counts agree with that accounting's drop-all figures exactly. Under either reading the whole-support first-order likelihood finding narrows from 5 arms to the same 3, which this gate reproduces independently below.

The filter removes those cycles from the evaluation of the same fitted predictions; they remain in the training rows of the cross-validated fit. It therefore bounds the contribution of indel-sourced cycles to the measured increment and not their contribution to the fitted coefficients. Rebuilding the cohort with indel rows removed before the per-state medians, which is what the cohort builder now does, would change the target values and the admitted support: that is a different cohort and was not executed here.

On the provenance axis, two of the 64 shared-source ranking clusters also hold an assay from another source. Dropping whole units and dropping only the shared-source assays give different supports and neither is obviously correct, so the choice is declared rather than left to the code. The primary rule is **dropping whole units**: the estimator weights each family's assays equally inside it, so dropping only the shared-source assays of a mixed unit changes that unit's internal weighting and makes its retained value a different statistic from the one the original fit averaged, while dropping whole units leaves every retained unit's weighting identical. Its cost is that 3 unshared assays are discarded with those two units, leaving 99 clusters and 134 assays. The declared sensitivity retains those two units on their unshared assays alone, at 101 clusters and 137 assays, and is reported beside the primary rule.

## Results

The declared bands and their support counts are final: they are properties of the two cohorts and of the frozen controls, not of any readout. The stability gain's stratified survival is final. The representation gain's stratified estimates are computed and reported here as **provisional**, because the readout-class sweep and the full-depth re-extraction that bound the representation readout are still running, and until they land a stratified representation null cannot be separated from a limit of the compressed linear readout class. No verdict about what the representations carry is drawn below.

### The stability gain: first-order likelihood differences, final

The refit reproduces the admitted fit. For each of the 33 arms the realised row identity, fold identity, support counts, design widths and selected ridge penalties are identical to that arm's admitted record, and over the 42 whole-support increments each arm reports the largest absolute disagreement anywhere in the panel is 7.0 × 10⁻¹⁶ squared kcal/mol. Five arms were dispatched twice, on different allocations, and the two dispatches agree on every per-group squared error to 1.3 × 10⁻¹⁴ squared kcal/mol.

On the unfiltered support the whole-support increment resolves above zero at all three seeds for 5 of 33 arms, below zero for 1 (Qwen2.5-0.5B-Instruct), and is unresolved for 27. The five are the arms this gate was asked to attack. On the indel-filtered support the count is 3 of 33 above zero — ProteinGLM-7B-CLM, ProGen2-xlarge and ProGen3-3B — with ProGen2-medium and ProGen2-large falling out at seed 20260924 only, by 0.00008 and 0.00015 squared kcal/mol on the lower bound, every point estimate still positive and the other two seeds still resolved. That is the conjunctive all-three-seed criterion being brittle at this support rather than the effects becoming unstable, and it matters for reading the strata below: a stratum that halves the effective units will tip the same criterion for further arms mechanically. A widened interval is an unresolved stratum, not a refuted gain.

**Unfiltered support**

| Arm | Stratum | Groups | Kish site pairs | 20260923 | 20260924 | 20260925 | Verdict |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| proteinglm-7b-clm | whole support | 64 | 129.2 | +0.00812 [+0.00373, +0.01287] | +0.00770 [+0.00341, +0.01209] | +0.00808 [+0.00376, +0.01277] | above zero, three seeds |
| proteinglm-7b-clm | no near-duplicate retrieved | 15 | 24.4 | +0.00901 [+0.00248, +0.01651] | +0.00971 [+0.00282, +0.01795] | +0.00877 [+0.00175, +0.01743] | above zero, three seeds |
| proteinglm-7b-clm | near-duplicate retrieved | 49 | 106.9 | +0.00785 [+0.00261, +0.01382] | +0.00709 [+0.00209, +0.01239] | +0.00787 [+0.00272, +0.01347] | above zero, three seeds |
| proteinglm-7b-clm | identity < 30% | 1 | 2.0 | -0.00136 | -0.00023 | -0.00340 | unresolved, thin support |
| proteinglm-7b-clm | identity 30–70% | 2 | 3.2 | +0.01584 | +0.01354 | +0.01553 | unresolved, thin support |
| proteinglm-7b-clm | identity 70–95% | 12 | 19.2 | +0.00874 [+0.00101, +0.01832] | +0.00989 [+0.00144, +0.02044] | +0.00866 [+0.00056, +0.01933] | above zero, three seeds |
| proteinglm-7b-clm | identity ≥ 95% | 49 | 106.9 | +0.00785 [+0.00261, +0.01382] | +0.00709 [+0.00209, +0.01239] | +0.00787 [+0.00272, +0.01347] | above zero, three seeds |
| proteinglm-7b-clm | Neff < 100 | 37 | 73.4 | +0.00890 [+0.00360, +0.01468] | +0.00889 [+0.00349, +0.01467] | +0.00882 [+0.00342, +0.01472] | above zero, three seeds |
| proteinglm-7b-clm | Neff ≥ 100 | 27 | 55.8 | +0.00705 [-0.00011, +0.01651] | +0.00608 [-0.00067, +0.01368] | +0.00706 [-0.00025, +0.01559] | unresolved |
| proteinglm-7b-clm | no homolog support | 1 | 2.0 | -0.00136 | -0.00023 | -0.00340 | unresolved, thin support |
| proteinglm-7b-clm | Neff < 10 | 11 | 22.6 | +0.01123 [+0.00206, +0.02077] | +0.00972 [+0.00118, +0.01812] | +0.01226 [+0.00225, +0.02366] | above zero, three seeds |
| proteinglm-7b-clm | Neff 10–100 | 25 | 48.8 | +0.00829 [+0.00181, +0.01524] | +0.00889 [+0.00200, +0.01658] | +0.00780 [+0.00163, +0.01469] | above zero, three seeds |
| proteinglm-7b-clm | Neff 100–1000 | 21 | 43.9 | +0.00861 [-0.00058, +0.01988] | +0.00724 [-0.00159, +0.01656] | +0.00903 [+0.00021, +0.01905] | unresolved |
| proteinglm-7b-clm | Neff ≥ 1000 | 6 | 12.0 | +0.00157 | +0.00201 | +0.00019 | unresolved, thin support |
| proteinglm-7b-clm | group of one | 36 | 84.0 | +0.00681 [+0.00139, +0.01240] | +0.00654 [+0.00066, +0.01254] | +0.00703 [+0.00161, +0.01255] | above zero, three seeds |
| proteinglm-7b-clm | group of two or more | 28 | 48.2 | +0.00980 [+0.00303, +0.01869] | +0.00920 [+0.00295, +0.01671] | +0.00944 [+0.00243, +0.01810] | above zero, three seeds |
| progen2-xlarge | whole support | 64 | 129.2 | +0.00700 [+0.00331, +0.01115] | +0.00674 [+0.00290, +0.01100] | +0.00774 [+0.00395, +0.01207] | above zero, three seeds |
| progen2-xlarge | no near-duplicate retrieved | 15 | 24.4 | +0.00695 [+0.00160, +0.01447] | +0.00747 [+0.00150, +0.01579] | +0.00638 [+0.00119, +0.01371] | above zero, three seeds |
| progen2-xlarge | near-duplicate retrieved | 49 | 106.9 | +0.00702 [+0.00276, +0.01203] | +0.00651 [+0.00212, +0.01128] | +0.00815 [+0.00359, +0.01338] | above zero, three seeds |
| progen2-xlarge | identity < 30% | 1 | 2.0 | -0.00087 | -0.00016 | -0.00389 | unresolved, thin support |
| progen2-xlarge | identity 30–70% | 2 | 3.2 | +0.01335 | +0.01272 | +0.01326 | unresolved, thin support |
| progen2-xlarge | identity 70–95% | 12 | 19.2 | +0.00654 [+0.00010, +0.01534] | +0.00723 [-0.00027, +0.01695] | +0.00609 [-0.00023, +0.01443] | unresolved |
| progen2-xlarge | identity ≥ 95% | 49 | 106.9 | +0.00702 [+0.00276, +0.01203] | +0.00651 [+0.00212, +0.01128] | +0.00815 [+0.00359, +0.01338] | above zero, three seeds |
| progen2-xlarge | Neff < 100 | 37 | 73.4 | +0.00644 [+0.00157, +0.01156] | +0.00644 [+0.00163, +0.01155] | +0.00719 [+0.00216, +0.01292] | above zero, three seeds |
| progen2-xlarge | Neff ≥ 100 | 27 | 55.8 | +0.00778 [+0.00239, +0.01434] | +0.00714 [+0.00155, +0.01378] | +0.00849 [+0.00268, +0.01540] | above zero, three seeds |
| progen2-xlarge | no homolog support | 1 | 2.0 | -0.00087 | -0.00016 | -0.00389 | unresolved, thin support |
| progen2-xlarge | Neff < 10 | 11 | 22.6 | +0.01033 [+0.00205, +0.01886] | +0.00900 [+0.00112, +0.01677] | +0.01217 [+0.00257, +0.02346] | above zero, three seeds |
| progen2-xlarge | Neff 10–100 | 25 | 48.8 | +0.00501 [-0.00085, +0.01120] | +0.00558 [-0.00067, +0.01221] | +0.00543 [-0.00006, +0.01150] | unresolved |
| progen2-xlarge | Neff 100–1000 | 21 | 43.9 | +0.00867 [+0.00141, +0.01715] | +0.00749 [+0.00013, +0.01589] | +0.00942 [+0.00175, +0.01815] | above zero, three seeds |
| progen2-xlarge | Neff ≥ 1000 | 6 | 12.0 | +0.00465 | +0.00588 | +0.00523 | unresolved, thin support |
| progen2-xlarge | group of one | 36 | 84.0 | +0.00572 [+0.00132, +0.01061] | +0.00522 [+0.00047, +0.01030] | +0.00608 [+0.00174, +0.01111] | above zero, three seeds |
| progen2-xlarge | group of two or more | 28 | 48.2 | +0.00866 [+0.00300, +0.01519] | +0.00868 [+0.00285, +0.01552] | +0.00986 [+0.00364, +0.01708] | above zero, three seeds |
| progen3-3b | whole support | 64 | 129.2 | +0.00391 [+0.00164, +0.00652] | +0.00386 [+0.00158, +0.00627] | +0.00378 [+0.00148, +0.00629] | above zero, three seeds |
| progen3-3b | no near-duplicate retrieved | 15 | 24.4 | +0.00573 [+0.00214, +0.00967] | +0.00683 [+0.00290, +0.01126] | +0.00639 [+0.00245, +0.01092] | above zero, three seeds |
| progen3-3b | near-duplicate retrieved | 49 | 106.9 | +0.00335 [+0.00073, +0.00647] | +0.00295 [+0.00046, +0.00572] | +0.00299 [+0.00045, +0.00597] | above zero, three seeds |
| progen3-3b | identity < 30% | 1 | 2.0 | -0.00185 | -0.00069 | -0.00284 | unresolved, thin support |
| progen3-3b | identity 30–70% | 2 | 3.2 | +0.00821 | +0.00778 | +0.00801 | unresolved, thin support |
| progen3-3b | identity 70–95% | 12 | 19.2 | +0.00595 [+0.00151, +0.01059] | +0.00730 [+0.00233, +0.01256] | +0.00688 [+0.00206, +0.01217] | above zero, three seeds |
| progen3-3b | identity ≥ 95% | 49 | 106.9 | +0.00335 [+0.00073, +0.00647] | +0.00295 [+0.00046, +0.00572] | +0.00299 [+0.00045, +0.00597] | above zero, three seeds |
| progen3-3b | Neff < 100 | 37 | 73.4 | +0.00439 [+0.00167, +0.00725] | +0.00450 [+0.00167, +0.00721] | +0.00423 [+0.00146, +0.00712] | above zero, three seeds |
| progen3-3b | Neff ≥ 100 | 27 | 55.8 | +0.00325 [-0.00018, +0.00783] | +0.00299 [-0.00046, +0.00703] | +0.00317 [-0.00060, +0.00755] | unresolved |
| progen3-3b | no homolog support | 1 | 2.0 | -0.00185 | -0.00069 | -0.00284 | unresolved, thin support |
| progen3-3b | Neff < 10 | 11 | 22.6 | +0.00692 [+0.00150, +0.01307] | +0.00659 [+0.00130, +0.01201] | +0.00714 [+0.00160, +0.01360] | above zero, three seeds |
| progen3-3b | Neff 10–100 | 25 | 48.8 | +0.00353 [+0.00030, +0.00674] | +0.00379 [+0.00029, +0.00722] | +0.00324 [+0.00013, +0.00646] | above zero, three seeds |
| progen3-3b | Neff 100–1000 | 21 | 43.9 | +0.00373 [-0.00083, +0.00969] | +0.00332 [-0.00126, +0.00863] | +0.00376 [-0.00118, +0.00937] | unresolved |
| progen3-3b | Neff ≥ 1000 | 6 | 12.0 | +0.00156 | +0.00182 | +0.00109 | unresolved, thin support |
| progen3-3b | group of one | 36 | 84.0 | +0.00415 [+0.00114, +0.00771] | +0.00409 [+0.00101, +0.00746] | +0.00425 [+0.00132, +0.00771] | above zero, three seeds |
| progen3-3b | group of two or more | 28 | 48.2 | +0.00361 [+0.00041, +0.00720] | +0.00357 [+0.00044, +0.00699] | +0.00319 [-0.00028, +0.00712] | unresolved |

**Indel-filtered support**

| Arm | Stratum | Groups | Kish site pairs | 20260923 | 20260924 | 20260925 | Verdict |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| proteinglm-7b-clm | whole support | 63 | 124.9 | +0.00817 [+0.00354, +0.01312] | +0.00776 [+0.00324, +0.01239] | +0.00813 [+0.00357, +0.01303] | above zero, three seeds |
| proteinglm-7b-clm | no near-duplicate retrieved | 15 | 24.4 | +0.00901 [+0.00248, +0.01651] | +0.00971 [+0.00282, +0.01795] | +0.00877 [+0.00175, +0.01743] | above zero, three seeds |
| proteinglm-7b-clm | near-duplicate retrieved | 48 | 102.2 | +0.00791 [+0.00263, +0.01422] | +0.00715 [+0.00195, +0.01266] | +0.00792 [+0.00271, +0.01402] | above zero, three seeds |
| proteinglm-7b-clm | identity < 30% | 1 | 2.0 | -0.00136 | -0.00023 | -0.00340 | unresolved, thin support |
| proteinglm-7b-clm | identity 30–70% | 2 | 3.2 | +0.01584 | +0.01354 | +0.01553 | unresolved, thin support |
| proteinglm-7b-clm | identity 70–95% | 12 | 19.2 | +0.00874 [+0.00101, +0.01832] | +0.00989 [+0.00144, +0.02044] | +0.00866 [+0.00056, +0.01933] | above zero, three seeds |
| proteinglm-7b-clm | identity ≥ 95% | 48 | 102.2 | +0.00791 [+0.00263, +0.01422] | +0.00715 [+0.00195, +0.01266] | +0.00792 [+0.00271, +0.01402] | above zero, three seeds |
| proteinglm-7b-clm | Neff < 100 | 37 | 72.8 | +0.00891 [+0.00361, +0.01471] | +0.00886 [+0.00347, +0.01463] | +0.00879 [+0.00342, +0.01471] | above zero, three seeds |
| proteinglm-7b-clm | Neff ≥ 100 | 26 | 52.1 | +0.00713 [-0.00024, +0.01716] | +0.00619 [-0.00080, +0.01402] | +0.00718 [-0.00018, +0.01618] | unresolved |
| proteinglm-7b-clm | no homolog support | 1 | 2.0 | -0.00136 | -0.00023 | -0.00340 | unresolved, thin support |
| proteinglm-7b-clm | Neff < 10 | 11 | 22.6 | +0.01123 [+0.00206, +0.02077] | +0.00972 [+0.00118, +0.01812] | +0.01226 [+0.00225, +0.02366] | above zero, three seeds |
| proteinglm-7b-clm | Neff 10–100 | 25 | 48.2 | +0.00830 [+0.00183, +0.01525] | +0.00885 [+0.00200, +0.01656] | +0.00775 [+0.00162, +0.01464] | above zero, three seeds |
| proteinglm-7b-clm | Neff 100–1000 | 21 | 43.2 | +0.00834 [-0.00069, +0.01973] | +0.00702 [-0.00197, +0.01623] | +0.00877 [-0.00007, +0.01878] | unresolved |
| proteinglm-7b-clm | Neff ≥ 1000 | 5 | 9.1 | +0.00201 | +0.00272 | +0.00053 | unresolved, thin support |
| proteinglm-7b-clm | group of one | 36 | 82.2 | +0.00666 [+0.00119, +0.01228] | +0.00638 [+0.00045, +0.01242] | +0.00684 [+0.00143, +0.01234] | above zero, three seeds |
| proteinglm-7b-clm | group of two or more | 27 | 45.5 | +0.01019 [+0.00323, +0.01874] | +0.00960 [+0.00343, +0.01666] | +0.00985 [+0.00275, +0.01831] | above zero, three seeds |
| progen2-xlarge | whole support | 63 | 124.9 | +0.00711 [+0.00341, +0.01099] | +0.00682 [+0.00299, +0.01077] | +0.00781 [+0.00390, +0.01184] | above zero, three seeds |
| progen2-xlarge | no near-duplicate retrieved | 15 | 24.4 | +0.00695 [+0.00160, +0.01447] | +0.00747 [+0.00150, +0.01579] | +0.00638 [+0.00119, +0.01371] | above zero, three seeds |
| progen2-xlarge | near-duplicate retrieved | 48 | 102.2 | +0.00716 [+0.00273, +0.01226] | +0.00661 [+0.00202, +0.01166] | +0.00825 [+0.00362, +0.01353] | above zero, three seeds |
| progen2-xlarge | identity < 30% | 1 | 2.0 | -0.00087 | -0.00016 | -0.00389 | unresolved, thin support |
| progen2-xlarge | identity 30–70% | 2 | 3.2 | +0.01335 | +0.01272 | +0.01326 | unresolved, thin support |
| progen2-xlarge | identity 70–95% | 12 | 19.2 | +0.00654 [+0.00010, +0.01534] | +0.00723 [-0.00027, +0.01695] | +0.00609 [-0.00023, +0.01443] | unresolved |
| progen2-xlarge | identity ≥ 95% | 48 | 102.2 | +0.00716 [+0.00273, +0.01226] | +0.00661 [+0.00202, +0.01166] | +0.00825 [+0.00362, +0.01353] | above zero, three seeds |
| progen2-xlarge | Neff < 100 | 37 | 72.8 | +0.00650 [+0.00166, +0.01168] | +0.00646 [+0.00165, +0.01158] | +0.00721 [+0.00217, +0.01295] | above zero, three seeds |
| progen2-xlarge | Neff ≥ 100 | 26 | 52.1 | +0.00798 [+0.00220, +0.01464] | +0.00732 [+0.00146, +0.01366] | +0.00866 [+0.00259, +0.01580] | above zero, three seeds |
| progen2-xlarge | no homolog support | 1 | 2.0 | -0.00087 | -0.00016 | -0.00389 | unresolved, thin support |
| progen2-xlarge | Neff < 10 | 11 | 22.6 | +0.01033 [+0.00205, +0.01886] | +0.00900 [+0.00112, +0.01677] | +0.01217 [+0.00257, +0.02346] | above zero, three seeds |
| progen2-xlarge | Neff 10–100 | 25 | 48.2 | +0.00511 [-0.00076, +0.01133] | +0.00561 [-0.00064, +0.01224] | +0.00547 [-0.00006, +0.01156] | unresolved |
| progen2-xlarge | Neff 100–1000 | 21 | 43.2 | +0.00856 [+0.00132, +0.01704] | +0.00741 [-0.00002, +0.01585] | +0.00932 [+0.00171, +0.01805] | unresolved |
| progen2-xlarge | Neff ≥ 1000 | 5 | 9.1 | +0.00556 | +0.00695 | +0.00587 | unresolved, thin support |
| progen2-xlarge | group of one | 36 | 82.2 | +0.00572 [+0.00126, +0.01067] | +0.00519 [+0.00043, +0.01026] | +0.00605 [+0.00169, +0.01111] | above zero, three seeds |
| progen2-xlarge | group of two or more | 27 | 45.5 | +0.00897 [+0.00325, +0.01542] | +0.00899 [+0.00308, +0.01554] | +0.01015 [+0.00396, +0.01748] | above zero, three seeds |
| progen3-3b | whole support | 63 | 124.9 | +0.00369 [+0.00136, +0.00630] | +0.00362 [+0.00129, +0.00613] | +0.00362 [+0.00126, +0.00621] | above zero, three seeds |
| progen3-3b | no near-duplicate retrieved | 15 | 24.4 | +0.00573 [+0.00214, +0.00967] | +0.00683 [+0.00290, +0.01126] | +0.00639 [+0.00245, +0.01092] | above zero, three seeds |
| progen3-3b | near-duplicate retrieved | 48 | 102.2 | +0.00306 [+0.00023, +0.00614] | +0.00261 [-0.00010, +0.00539] | +0.00275 [-0.00009, +0.00568] | unresolved |
| progen3-3b | identity < 30% | 1 | 2.0 | -0.00185 | -0.00069 | -0.00284 | unresolved, thin support |
| progen3-3b | identity 30–70% | 2 | 3.2 | +0.00821 | +0.00778 | +0.00801 | unresolved, thin support |
| progen3-3b | identity 70–95% | 12 | 19.2 | +0.00595 [+0.00151, +0.01059] | +0.00730 [+0.00233, +0.01256] | +0.00688 [+0.00206, +0.01217] | above zero, three seeds |
| progen3-3b | identity ≥ 95% | 48 | 102.2 | +0.00306 [+0.00023, +0.00614] | +0.00261 [-0.00010, +0.00539] | +0.00275 [-0.00009, +0.00568] | unresolved |
| progen3-3b | Neff < 100 | 37 | 72.8 | +0.00397 [+0.00122, +0.00680] | +0.00399 [+0.00121, +0.00674] | +0.00383 [+0.00112, +0.00676] | above zero, three seeds |
| progen3-3b | Neff ≥ 100 | 26 | 52.1 | +0.00330 [-0.00040, +0.00769] | +0.00308 [-0.00063, +0.00706] | +0.00331 [-0.00080, +0.00762] | unresolved |
| progen3-3b | no homolog support | 1 | 2.0 | -0.00185 | -0.00069 | -0.00284 | unresolved, thin support |
| progen3-3b | Neff < 10 | 11 | 22.6 | +0.00692 [+0.00150, +0.01307] | +0.00659 [+0.00130, +0.01201] | +0.00714 [+0.00160, +0.01360] | above zero, three seeds |
| progen3-3b | Neff 10–100 | 25 | 48.2 | +0.00290 [-0.00021, +0.00605] | +0.00304 [-0.00033, +0.00641] | +0.00265 [-0.00040, +0.00582] | unresolved |
| progen3-3b | Neff 100–1000 | 21 | 43.2 | +0.00364 [-0.00097, +0.00963] | +0.00321 [-0.00139, +0.00855] | +0.00366 [-0.00130, +0.00932] | unresolved |
| progen3-3b | Neff ≥ 1000 | 5 | 9.1 | +0.00188 | +0.00254 | +0.00184 | unresolved, thin support |
| progen3-3b | group of one | 36 | 82.2 | +0.00366 [+0.00065, +0.00724] | +0.00351 [+0.00039, +0.00690] | +0.00378 [+0.00084, +0.00720] | above zero, three seeds |
| progen3-3b | group of two or more | 27 | 45.5 | +0.00374 [+0.00041, +0.00745] | +0.00376 [+0.00045, +0.00724] | +0.00340 [-0.00028, +0.00729] | unresolved |

The pattern is flat to inverted, not concentrated in the retrieved-homolog strata. For all three arms that clear the criterion, the increment resolves above zero at all three seeds on the 15 groups with no retrieved near-duplicate as well as on the 49 with one, and its point estimate is the larger of the two on the near-duplicate-absent side. Across the panel the shallowest retrieval-depth band carries more arms above zero than the whole support does: 10 of 33 on the 11 groups at Neff < 10 against 5 of 33 on all 64, on both supports. The deep band at Neff ≥ 100 carries 1. On the frozen grouping contract the increment resolves on the 36 groups whose group holds one member, so it does not require a close relative among the cohort's own backgrounds.

| Arm | Support | Groups | 20260923 | 20260924 | 20260925 | Verdict |
| --- | --- | ---: | --- | --- | --- | --- |
| progen2-medium | unfiltered | 64 | +0.00426 [+0.00116, +0.00760] | +0.00338 [+0.00014, +0.00675] | +0.00457 [+0.00126, +0.00813] | above zero, three seeds |
| progen2-medium | indel-filtered | 63 | +0.00417 [+0.00091, +0.00744] | +0.00323 [-0.00008, +0.00653] | +0.00452 [+0.00104, +0.00800] | unresolved |
| progen2-large | unfiltered | 64 | +0.00349 [+0.00074, +0.00647] | +0.00288 [+0.00012, +0.00597] | +0.00389 [+0.00087, +0.00722] | above zero, three seeds |
| progen2-large | indel-filtered | 63 | +0.00351 [+0.00064, +0.00645] | +0.00279 [-0.00015, +0.00580] | +0.00388 [+0.00076, +0.00712] | unresolved |

**Panel pattern, unfiltered support**

| Stratum | Units | Above zero, three seeds | Below zero, three seeds | Unresolved |
| --- | ---: | ---: | ---: | ---: |
| whole support | 64 | 5 | 1 | 27 |
| `near_duplicate` / no near-duplicate retrieved | 15 | 3 | 0 | 30 |
| `near_duplicate` / near-duplicate retrieved | 49 | 3 | 1 | 29 |
| `identity_band` / identity < 30% | 1 | 0 | 0 | 33 |
| `identity_band` / identity 30–70% | 2 | 0 | 0 | 33 |
| `identity_band` / identity 70–95% | 12 | 2 | 0 | 31 |
| `identity_band` / identity ≥ 95% | 49 | 3 | 1 | 29 |
| `coarse_depth` / Neff < 100 | 37 | 3 | 0 | 30 |
| `coarse_depth` / Neff ≥ 100 | 27 | 1 | 1 | 31 |
| `depth_band` / no homolog support | 1 | 0 | 0 | 33 |
| `depth_band` / Neff < 10 | 11 | 10 | 1 | 22 |
| `depth_band` / Neff 10–100 | 25 | 3 | 0 | 30 |
| `depth_band` / Neff 100–1000 | 21 | 1 | 0 | 32 |
| `depth_band` / Neff ≥ 1000 | 6 | 0 | 0 | 33 |
| `family_band` / group of one | 36 | 4 | 0 | 29 |
| `family_band` / group of two or more | 28 | 4 | 0 | 29 |
| `source_provenance` / shared source | 64 | 5 | 1 | 27 |

**Panel pattern, indel-filtered support**

| Stratum | Units | Above zero, three seeds | Below zero, three seeds | Unresolved |
| --- | ---: | ---: | ---: | ---: |
| whole support | 63 | 3 | 0 | 30 |
| `near_duplicate` / no near-duplicate retrieved | 15 | 3 | 0 | 30 |
| `near_duplicate` / near-duplicate retrieved | 48 | 2 | 1 | 30 |
| `identity_band` / identity < 30% | 1 | 0 | 0 | 33 |
| `identity_band` / identity 30–70% | 2 | 0 | 0 | 33 |
| `identity_band` / identity 70–95% | 12 | 2 | 0 | 31 |
| `identity_band` / identity ≥ 95% | 48 | 2 | 1 | 30 |
| `coarse_depth` / Neff < 100 | 37 | 3 | 0 | 30 |
| `coarse_depth` / Neff ≥ 100 | 26 | 1 | 1 | 31 |
| `depth_band` / no homolog support | 1 | 0 | 0 | 33 |
| `depth_band` / Neff < 10 | 11 | 10 | 1 | 22 |
| `depth_band` / Neff 10–100 | 25 | 2 | 0 | 31 |
| `depth_band` / Neff 100–1000 | 21 | 0 | 0 | 33 |
| `depth_band` / Neff ≥ 1000 | 5 | 0 | 0 | 33 |
| `family_band` / group of one | 36 | 3 | 0 | 30 |
| `family_band` / group of two or more | 27 | 4 | 0 | 29 |
| `source_provenance` / shared source | 63 | 3 | 0 | 30 |

The remote end of the identity stratification answers nothing. The band below 30% identity holds one group and the 30–70% band holds two, so neither clears the eight-unit floor at any arm or seed; the deepest-support band holds six and does not either. Those three bands are unresolved for want of support in both directions.

### The mutation-ranking gain: profile-residual representation, provisional

Re-aggregating the retained per-assay increments over the same wild-type-family bootstrap reproduces every published summary point of every admitted cell exactly. The panel cells are the crossed-control campaign's own full-panel extension; where its cells and the admitted five-arm cells overlap, on 12 arm-seed cells, their summary points agree to 9.8 × 10⁻⁷ dimensionless Spearman, and the five-arm cells supply ProGen3-3B, which the extension's run does not cover. All 33 arms share one held-family map per split seed.

| Arm | Stratum | Clusters | Kish assays | 20260923 | 20260924 | 20260925 | Verdict |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| progen3-3b | whole anchor | 163 | 179.6 | +0.04104 [+0.02853, +0.05373] | +0.03937 [+0.02750, +0.05159] | +0.03666 [+0.02386, +0.04975] | above zero, three seeds |
| progen3-3b | no near-duplicate retrieved | 23 | 25.2 | +0.03231 [+0.00275, +0.06355] | +0.03007 [+0.00087, +0.05991] | +0.01910 [-0.01147, +0.04961] | unresolved |
| progen3-3b | near-duplicate retrieved | 140 | 154.4 | +0.04247 [+0.02895, +0.05770] | +0.04090 [+0.02784, +0.05493] | +0.03955 [+0.02582, +0.05428] | above zero, three seeds |
| progen3-3b | identity 30–70% | 5 | 5.6 | +0.02320 | +0.01818 | +0.02371 | unresolved, thin support |
| progen3-3b | identity 70–95% | 18 | 19.6 | +0.03484 [-0.00204, +0.06970] | +0.03337 [-0.00103, +0.06823] | +0.01782 [-0.01991, +0.05484] | unresolved |
| progen3-3b | identity ≥ 95% | 140 | 154.4 | +0.04247 [+0.02895, +0.05770] | +0.04090 [+0.02784, +0.05493] | +0.03955 [+0.02582, +0.05428] | above zero, three seeds |
| progen3-3b | Neff < 100 | 85 | 93.0 | +0.04720 [+0.02985, +0.06479] | +0.04521 [+0.02772, +0.06251] | +0.04556 [+0.02781, +0.06282] | above zero, three seeds |
| progen3-3b | Neff ≥ 100 | 78 | 86.6 | +0.03432 [+0.01600, +0.05385] | +0.03300 [+0.01679, +0.04970] | +0.02697 [+0.00824, +0.04713] | above zero, three seeds |
| progen3-3b | Neff < 10 | 21 | 21.5 | +0.07508 [+0.03839, +0.11913] | +0.06883 [+0.03489, +0.11160] | +0.05922 [+0.01816, +0.10787] | above zero, three seeds |
| progen3-3b | Neff 10–100 | 64 | 71.7 | +0.03805 [+0.01955, +0.05675] | +0.03746 [+0.01774, +0.05604] | +0.04108 [+0.02353, +0.05801] | above zero, three seeds |
| progen3-3b | Neff 100–1000 | 65 | 70.5 | +0.03800 [+0.01812, +0.06011] | +0.03603 [+0.01746, +0.05511] | +0.03000 [+0.00848, +0.05341] | above zero, three seeds |
| progen3-3b | Neff ≥ 1000 | 13 | 16.4 | +0.01596 [-0.00467, +0.04139] | +0.01788 [+0.00021, +0.03860] | +0.01181 [-0.00452, +0.02838] | unresolved |
| progen3-3b | group of one | 152 | 162.4 | +0.04388 [+0.03046, +0.05763] | +0.04232 [+0.02980, +0.05562] | +0.03893 [+0.02558, +0.05285] | above zero, three seeds |
| progen3-3b | group of two or more | 11 | 21.4 | +0.00178 [-0.00892, +0.01203] | -0.00146 [-0.01348, +0.01056] | +0.00534 [-0.00403, +0.01486] | unresolved |
| progen3-3b | other source | 99 | 115.2 | +0.02613 [+0.01763, +0.03523] | +0.02402 [+0.01575, +0.03258] | +0.02238 [+0.01411, +0.03072] | above zero, three seeds |
| progen3-3b | shared source | 64 | 65.2 | +0.06410 [+0.03457, +0.09445] | +0.06311 [+0.03475, +0.09070] | +0.05876 [+0.02834, +0.08965] | above zero, three seeds |
| progen3-3b | other source, assay-drop sensitivity | 101 | 117.8 | +0.02553 [+0.01716, +0.03439] | +0.02309 [+0.01482, +0.03143] | +0.02150 [+0.01335, +0.02984] | above zero, three seeds |
| proteinglm-7b-clm | whole anchor | 163 | 179.6 | +0.05347 [+0.04008, +0.06778] | +0.05570 [+0.04322, +0.06941] | +0.04720 [+0.03362, +0.06081] | above zero, three seeds |
| proteinglm-7b-clm | no near-duplicate retrieved | 23 | 25.2 | +0.05274 [+0.02372, +0.08237] | +0.05733 [+0.02577, +0.08894] | +0.03340 [-0.00301, +0.06710] | unresolved |
| proteinglm-7b-clm | near-duplicate retrieved | 140 | 154.4 | +0.05359 [+0.03891, +0.07049] | +0.05543 [+0.04124, +0.07110] | +0.04947 [+0.03498, +0.06526] | above zero, three seeds |
| proteinglm-7b-clm | identity 30–70% | 5 | 5.6 | +0.03587 | +0.03359 | +0.03023 | unresolved, thin support |
| proteinglm-7b-clm | identity 70–95% | 18 | 19.6 | +0.05742 [+0.01919, +0.09396] | +0.06392 [+0.02379, +0.10520] | +0.03428 [-0.01129, +0.07636] | unresolved |
| proteinglm-7b-clm | identity ≥ 95% | 140 | 154.4 | +0.05359 [+0.03891, +0.07049] | +0.05543 [+0.04124, +0.07110] | +0.04947 [+0.03498, +0.06526] | above zero, three seeds |
| proteinglm-7b-clm | Neff < 100 | 85 | 93.0 | +0.05257 [+0.03006, +0.07539] | +0.05696 [+0.03667, +0.07725] | +0.05263 [+0.03116, +0.07382] | above zero, three seeds |
| proteinglm-7b-clm | Neff ≥ 100 | 78 | 86.6 | +0.05446 [+0.03820, +0.07154] | +0.05433 [+0.03796, +0.07096] | +0.04129 [+0.02571, +0.05702] | above zero, three seeds |
| proteinglm-7b-clm | Neff < 10 | 21 | 21.5 | +0.06506 [-0.00047, +0.13994] | +0.07267 [+0.01529, +0.13941] | +0.05993 [-0.00332, +0.13067] | unresolved |
| proteinglm-7b-clm | Neff 10–100 | 64 | 71.7 | +0.04847 [+0.02936, +0.06835] | +0.05180 [+0.03385, +0.07237] | +0.05023 [+0.03039, +0.06982] | above zero, three seeds |
| proteinglm-7b-clm | Neff 100–1000 | 65 | 70.5 | +0.06239 [+0.04440, +0.08222] | +0.06156 [+0.04392, +0.07968] | +0.04599 [+0.02810, +0.06486] | above zero, three seeds |
| proteinglm-7b-clm | Neff ≥ 1000 | 13 | 16.4 | +0.01481 [-0.01761, +0.04188] | +0.01819 [-0.01776, +0.05143] | +0.01782 [-0.01134, +0.04531] | unresolved |
| proteinglm-7b-clm | group of one | 152 | 162.4 | +0.05567 [+0.04107, +0.07111] | +0.05828 [+0.04489, +0.07299] | +0.04937 [+0.03528, +0.06458] | above zero, three seeds |
| proteinglm-7b-clm | group of two or more | 11 | 21.4 | +0.02318 [+0.00943, +0.03608] | +0.02010 [+0.00474, +0.03334] | +0.01720 [-0.00059, +0.03579] | unresolved |
| proteinglm-7b-clm | other source | 99 | 115.2 | +0.02792 [+0.01573, +0.04018] | +0.02941 [+0.01777, +0.04056] | +0.02639 [+0.01505, +0.03738] | above zero, three seeds |
| proteinglm-7b-clm | shared source | 64 | 65.2 | +0.09299 [+0.06560, +0.12203] | +0.09636 [+0.07086, +0.12253] | +0.07939 [+0.05113, +0.10871] | above zero, three seeds |
| proteinglm-7b-clm | other source, assay-drop sensitivity | 101 | 117.8 | +0.02791 [+0.01593, +0.04050] | +0.02881 [+0.01742, +0.04057] | +0.02591 [+0.01481, +0.03697] | above zero, three seeds |

**Panel pattern, mutation ranking**

| Stratum | Units | Above zero, three seeds | Below zero, three seeds | Unresolved |
| --- | ---: | ---: | ---: | ---: |
| whole support | 163 | 11 | 6 | 16 |
| `near_duplicate` / no near-duplicate retrieved | 23 | 0 | 0 | 33 |
| `near_duplicate` / near-duplicate retrieved | 140 | 11 | 6 | 16 |
| `identity_band` / identity 30–70% | 5 | 0 | 0 | 33 |
| `identity_band` / identity 70–95% | 18 | 0 | 0 | 33 |
| `identity_band` / identity ≥ 95% | 140 | 11 | 6 | 16 |
| `coarse_depth` / Neff < 100 | 85 | 12 | 1 | 20 |
| `coarse_depth` / Neff ≥ 100 | 78 | 5 | 5 | 23 |
| `depth_band` / Neff < 10 | 21 | 2 | 0 | 31 |
| `depth_band` / Neff 10–100 | 64 | 7 | 1 | 25 |
| `depth_band` / Neff 100–1000 | 65 | 5 | 4 | 24 |
| `depth_band` / Neff ≥ 1000 | 13 | 1 | 3 | 29 |
| `family_band` / group of one | 152 | 11 | 5 | 17 |
| `family_band` / group of two or more | 11 | 0 | 13 | 20 |
| `source_provenance` / other source | 99 | 5 | 1 | 27 |
| `source_provenance` / shared source | 64 | 11 | 6 | 16 |

Two readings are available and neither is a verdict. On depth and on family membership the increment of the two native protein arms holds across bands, and is largest in the shallowest depth band (ProGen3-3B +0.05922 to +0.07508 on 21 clusters at Neff < 10 against +0.03666 to +0.04104 on all 163). On the 23 clusters with no retrieved near-duplicate it is unresolved for both arms: two of three seeds above zero, the third crossing, with point estimates from 52% to 103% of the whole-anchor value and the interval widening from 0.025–0.028 to 0.059–0.070. Across the panel no arm of 33 resolves on that band in either direction. So the low-identity end of this endpoint is unresolved for want of support, and the point estimates there give no sign that the increment is carried by the near-duplicate units.

### Source provenance

This axis lives on the ranking anchor and therefore carries the same provisional status as the section above. The experimental source shared with the stability cohort covers 64 of the 163 anchor resampling units. Excluding them under the declared whole-unit rule leaves 99 clusters and 134 assays, and the increment of both native protein arms stays above zero at all three seeds on that support, at +0.02238 to +0.02613 for ProGen3-3B and +0.02639 to +0.02941 for ProteinGLM-7B-CLM, against +0.03666 to +0.04104 and +0.04720 to +0.05570 on the whole anchor. It is 2.4 to 3.3 times larger on the shared-source band than on the other 99 clusters. Power did not degrade: at every seed the interval on the unshared stratum is narrower than on the whole anchor, by 0.0032 to 0.0093 dimensionless Spearman, because the excluded units carry the larger and more variable increments. The declared assay-drop sensitivity, which retains the two mixed units on their unshared assays alone at 101 clusters and 137 assays, moves every point estimate by at most 0.00093.

## Verdicts

**The first-order likelihood gain on measured stability survives these stratifications, on the support they provide, and the pattern is not a retrieval pattern.** For the three arms that clear the all-three-seed criterion on the corrected support, the increment resolves above zero at all three seeds on the 15 groups whose wild type retrieves no near-duplicate, on the 49 that do, on the 37 groups with shallow alignment depth, on the 11 groups at fewer than ten effective sequences, and on the 36 groups the frozen contract places in a group of one. On the deep-support band at 27 groups it resolves for one of the three and is unresolved for two. Across the panel the shallowest depth band carries twice as many arms above zero as the whole support, 10 of 33 against 5 of 33.

A retrieval or memorization explanation of this gain would put it in the high-identity, deep-support strata and remove it from the others. The measurement shows the opposite gradient at every axis that has the units to resolve: larger point estimates where no near-duplicate is retrieved, larger where alignment depth is shallow, and resolved on groups with no detected relative among the cohort's own backgrounds. The declared insertion/deletion exclusion moves the whole-support point estimates by at most 0.00022 squared kcal/mol and changes one stratum verdict among those three arms — ProGen3-3B's near-duplicate-**present** band becomes unresolved under the filter while its near-duplicate-absent band stays resolved, which again is the direction a retrieval explanation does not predict.

That verdict is bounded by what the support could carry. It rests on the binary near-duplicate contrast at 15 units, the depth contrast and the family contrast; the two remote identity bands hold 1 and 2 groups and the deepest-support band holds 6, so all three are unresolved in both directions and this gate has **not** tested the gain against remote homology. It is a statement about units whose wild type retrieves no close relative under the declared search settings, not about units a model never saw.

**The profile-residual representation gain is deferred, and no verdict is drawn.** Its stratified estimates are above, with intervals, resampling units and effective unit counts. What they show is that the increment of the two native protein arms resolves above zero at all three seeds on both coarse depth bands, on three of the four fine depth bands, on the 152 single-wild-type families and on the 99 clusters left after the shared experimental source is removed, the last at about half its whole-anchor magnitude; and that it is unresolved on the 23-cluster low-identity band, on the 13 deepest-support clusters and on the 11 multi-wild-type families, with the low-identity band's point estimates from 52% to 103% of the whole-anchor value. What they cannot yet distinguish is a stratified null from a limit of the compressed linear readout class at the retained depths, because the readout-class sweep and the full-depth re-extraction have not landed. Every number in that section is subject to recomputation, and nothing here licenses a claim about what the representations contain.

**The source-provenance axis is a source-independence sensitivity and carries no retrieval verdict.** Its result is that the ranking increment survives removing the experimental source shared with the stability cohort, at a smaller magnitude and with no loss of power, so the two endpoints are not reducible to one shared source — while remaining, by the shared component itself, not independent confirmations of one another.

## Bounds on these verdicts

- **Pretraining exposure cannot be matched retrospectively.** Every band here is a property of a search against one UniRef50 snapshot, not of any model's training corpus. A unit in the lowest identity band is a unit whose wild type retrieved no close relative under the declared search settings; it is not a unit a model never saw.
- **No detected alignment is not certified corpus exclusion.** One stability background returns no UniRef50 hit at an e-value of 1 × 10⁻³ and lands in the lowest identity band on that basis alone. That is absence of a detected alignment in the searched snapshot under the declared settings, and nothing stronger.
- **The frozen groups are conservative held-group controls, not certified remote-family disjointness.** Homology below 30% identity or 80% mutual coverage goes undetected on the stability side, 179 of 478 backgrounds carry no Pfam label, and the ranking side's 50%-identity single-linkage clusters are not a Pfam-family or CATH-superfamily holdout. Two backgrounds the catalogue labels natural are designed mini-proteins.
- **The high-identity dominance is the binding limitation, and it is a property of these cohorts rather than of the available data.** A stratification that retains 49 of 64 stability groups and 140 of 163 ranking clusters in the near-duplicate band has not tested either gain against remote homology. The remote-identity strata are unresolved for want of support: they are neither passed nor failed, and reporting their point estimates without their unit counts would misrepresent that. A support carrying 105 entirely remote groups has been measured with the same search settings, at a label floor about two thirds of that endpoint's own spread; the section above states both numbers and what a verdict there would have to be stated against.
- **Intervals are pointwise and unadjusted.** Every interval conditions on the fitted cross-validation predictions, omits training and split variation, and is unadjusted for the arms, the six stratifications, their bands, the reported contrasts, the two stability supports or the three split seeds. This document reports 20,097 intervals in total and 5,445 on the two primary contrasts, over 33 arms, six stratifications, 20 populated bands, three split seeds and two stability supports. At that count a handful of bands clearing zero in one band at one seed is expected under no effect, so every stratum verdict above rests on holding at all three seeds and on the whole-support finding it decomposes, never on an isolated interval.
- **The five arms carrying the stability finding are a targeted follow-up, and the panel is the generalizability check around them.** Those five were selected because they carry the finding under test, so their stratified numbers bound an established positive rather than estimating a fresh one. Running all 33 arms is what makes a stratum-restricted gain in an arm with no whole-support gain detectable at all; a five-arm design is blind to that by construction. The three split seeds are a sensitivity range on one cohort rather than independent replication, and checkpoint lineages inside the panel are correlated, so the per-band arm counts do not estimate a model-population prevalence.
- **The two gains are not independent endpoints.** The shared experimental source covers the whole stability cohort and 64 of the 163 ranking resampling units, so an agreement between them is not corroboration on independent biological material.
- **A surviving gain is still bounded by its own study's controls and readout class.** This gate removes retrieval and memorization as explanations to the extent its support allows. It does not locate a mechanism, does not identify which computation carries the information, and establishes no irreducible biological rule.
- **The batch-one precision gate is structurally vacuous.** Every arm's qualified extraction batch size is recorded, and at batch size one the reference forward and the production forward are the same computation, so no arm may be called numerically validated on that basis.

## What is retained, so a revisit is cheap

The representation section will be recomputed when the readout-class sweep and the full-depth re-extraction land, and everything that recomputation needs is retained rather than regenerated.

| Retained | Where | What it fixes |
| --- | --- | --- |
| Strata declaration, all six stratifications with per-unit band assignments and per-band support counts | `logs/d1_gate_retrieval_20260923/strata_declaration.json`, content digest `cb56f5fe…` | the stratum assignment of every group and cluster |
| Indel exclusion, per-background excluded cycle indices with their carrier rows | `logs/d1_gate_retrieval_20260923/indel_exclusion.json`, content digest `326aaa49…` | the second stability support |
| Stability refit records, 33 arms: per-group squared error and within-group Spearman for every design on both supports, per-group increments, fold membership, selected penalties, nuisance diagnostics | `logs/d1_gate_retrieval_20260923/stability/refit_<arm>.json`, and on the cluster under the panel wave's results directory | the fitted predictions' per-group decomposition, so no stability refit is needed again |
| Cross-dispatch duplicates, 5 arms | `logs/d1_gate_retrieval_20260923/stability_wave1/` | determinism of the refit across allocations |
| Crossed-control cell reports, 33 arms at three seeds | `logs/d1_gate_retrieval_20260923/readout_panel/` and `logs/d1_crossed_controls_20260923/reports/` | the per-assay increments the ranking strata re-aggregate |
| Gate report | `logs/d1_gate_retrieval_20260923/retrieval_strata_report.json` | every interval, support count and effective count in this document |

Folds and seeds are fixed by the studies this gate reads and are recorded in each artifact: five outer and four inner held-group folds at split seeds 20260923, 20260924 and 20260925, and 2,000 bootstrap draws at seed 20260923. A recomputation that keeps those and swaps the readout class only needs new per-assay increments; the stratum assignments, the unit counts and the aggregation are unchanged.

## Artifacts and entry points

The declaration, the exclusion and the gate report are under ignored `logs/d1_gate_retrieval_20260923/`; the stability refit records are retained on the cluster under the panel wave's results directory and pulled there as JSON. Entry points are `scripts/transfer/declare_retrieval_strata.py`, `declare_indel_exclusion.py`, `fit_retrieval_strata_pairwise.py` and `analyse_retrieval_strata.py`, with the band declarations, the partition check, the Kish conventions and the decomposition identity in `src/transfer/retrieval_strata.py` and the conditions that must hold in `tests/test_retrieval_strata.py`. The panel wave ran as 16 concurrent CPU lanes over four allocations, one cell per slot, with distinct manifest basenames per wave; no GPU was requested or held.
