# Crossed composition, local-pattern and profile controls: results

**Completed and admitted, 2026-09-24.** Fifteen fitting cells — five admitted Readout arms at three split seeds — completed with zero failures, and the cross-cell admission receipt passed every identity check.

On the frozen 201-assay Readout cohort, adding a frozen representation to a composition control raises held-cluster ranking by +0.106 to +0.147 Spearman for the two native protein arms. About two thirds of that increment is absorbed by mutation-local profile statistics, and what remains over that stronger control, +0.037 to +0.056, stays above zero in all three splits for ProGen3-3B and ProteinGLM; so does what remains over C+L+P and C+L+P+T, +0.043 to +0.061. The Llama-2 text parent's representation increment is resolved below zero over composition alone in all three splits and is unresolved over C+L+P and C+L+P+T; both ProLLaMA stages are unresolved over composition and positive but split-sensitive over the richer controls. Declared dipeptide and tripeptide difference features carry no transferable predictive information of their own on this cohort: adding them lowers the control's own held-cluster correlation. The tokenisation interface descriptors change nothing material.

## What this measures

The [information-hierarchy design](D1_INFORMATION_HIERARCHY_PROTOCOL.md) retains an inexpensive crossed comparison of composition, local-pattern and profile features on the existing Readout cohort as a complementary D1 test. This document reports it. The question is not how well any predictor ranks mutation effects; it is whether model-derived features still add stable predictive information once the level below has been controlled for, measured as a paired increment under wild-type-cluster-held-out splits.

Four nested control sets are built from sequence and alignment statistics alone. Each is then extended twice and separately, once by the arm's native likelihood and once by its frozen representation. The scientific object is the pattern of the representation increment across the nested controls: whether an increment that survives composition also survives composition plus declared short-range patterns and composition plus richer independent-site profile statistics.

Everything else is inherited unchanged from the admitted [Readout diagnostic](D1_READOUT_PROTOCOL.md): the frozen cohort, the five deterministic outer and four inner wild-type-cluster folds at split seeds 20260923, 20260924 and 20260925, the fixed representation-projection seeds 20260923 through 20260926, within-assay standardized-rank targets, equal cluster and within-cluster assay weights, training-only weighted feature scaling, the ridge penalty grid with its tie rule, and 2,000 paired wild-type-cluster bootstrap draws at seed 20260923. Only the already-admitted original five-arm artifacts are read; the concurrent panel expansion is a separate campaign and none of its outputs enter here.

## Control sets

| Set | Blocks | Width | What it expresses |
| --- | --- | ---: | --- |
| C | S | 444 | The admitted Readout sequence descriptors: 400 directed substitution counts, relative mutated-position mean and standard deviation, mutation count, wild-type and mutant 20-residue composition, and residue length over 1,024 |
| C+L | S, L | 1,100 | C plus 400 exact mutant-minus-wild-type dipeptide count differences and a 256-coordinate fixed Gaussian projection of the 8,000 tripeptide count differences |
| C+P | S, P | 458 | C plus 14 mutation-local profile coordinates: the admitted profile score's within-assay rank, wild-type and mutant column frequencies at the mutated positions, their column entropies in nats, mutated-column support, the per-substitution log-odds, and the wild type's supported-column fraction, log₁₀ N_eff, maximum corpus identity and mean supported-column entropy |
| C+L+P | S, L, P | 1,114 | Both of the above |
| C+L+P+T | S, L, P, T | 1,122 | C+L+P plus the 8 tokenisation interface coordinates below |

Adding the likelihood appends one column, the within-assay standardized rank of the native mutant-minus-wild-type log likelihood. Adding the representation appends the 1,024 fixed-projection coordinates of the four mutant-minus-wild-type hidden-state blocks. The widest design, C+L+P+T+M+R, has 2,147 columns before the unpenalized intercept.

C is the admitted sequence-only control unchanged, and C+P strictly contains the admitted profile-plus-sequence control's columns, so C+P+M strictly contains the admitted primary baseline B = P+M+S. The nested relation is C ⊂ C+L, C ⊂ C+P, and both ⊂ C+L+P ⊂ C+L+P+T; C+L and C+P are crossed, not ordered, which is why this is a crossed design rather than a single ladder.

## Tokenisation interface

The design requires per-state token counts, position shifts and segmentation changes to be visible, so that a representation increment cannot be attributed to segmentation without being seen. The T block carries, per variant: the wild-type and mutant pooled residue-bearing token counts over the 1,024-position budget, their difference (the token-count cycle, which is also the shift applied to every token position after the divergence), the common prefix and common suffix token fractions, the number of tokens outside that shared prefix and suffix summed over both states, and the tokens-per-residue ratio of each state. The counted tokens are exactly the mean-pooling denominators the production extraction used, and the recomputed maximum packed-token count per assay is required to equal the value the production extraction manifest declares.

A residue tokenizer has no segmentation degree of freedom: one token per residue in both states, so the token-count cycle is identically zero and the changed-token count is twice the number of tokens spanned by the first through last substitution. A multi-residue subword tokenizer does vary.

## Evidence and support

The fitting code is `scripts/transfer/analyse_crossed_controls.py` with `src/transfer/crossed_controls.py`; the campaign manifests are `scripts/transfer/campaign_crossed_controls_lane_a.tsv`, `_lane_b.tsv`, `_lane_c.tsv`, the interface check `_smoke.tsv` and the collector `_collect.tsv`. Fitting ran on six idle H200 cards across three allocations in three concurrent lanes, 2.5 to 8 minutes per cell.

The panel is 201 assays, 163 wild-type identity clusters and 25,728 variants — the admitted anchor, with cohort SHA256 `4093ac34368cd9e7be02e5c84a22655ce78cb9bb4e4d58d626f281868dbe7992`. Every metric is defined on every assay; no undefined-statistic exclusion occurs. The cross-cell admission receipt (`crossed_controls_admission.json`, SHA256 `90ef094e9e7d20279554065ee29610b858339370a7e7fe3c73ec1c605cc33c32`) binds all fifteen cell reports and verifies identical cohort bytes, assay support and counts, identical outer and inner group membership across arms at each split seed, bit-identical predictions for every model-independent control design, and identical control-set and feature-order definitions.

Four further bindings were verified inside every cell. The realised group membership is identical to the admitted Readout fits' membership at all three split seeds, with outer held-family sizes 33, 33, 33, 32 and 32. The mutation-local profile block is bound to the same retained column-frequency arrays the cohort's own profile scores were built from, checked by recomputing those scores: the maximum deviation is 2.842 × 10⁻¹⁴, floating-point rounding. The recomputed maximum packed-token count matches the production extraction manifest for every assay of every arm. And the retained mutant sequence is required to equal the wild type with its mutation string applied, for all 25,728 variants.

The controls reproduce their admitted counterparts exactly where they should:

| Split seed | admitted sequence-only | this study C | admitted profile+sequence | this study C+P | admitted B, ProteinGLM | this study C+P+M, ProteinGLM |
| --- | --- | --- | --- | --- | --- | --- |
| 20260923 | +0.30024 | +0.30024 | +0.43638 | +0.44659 | +0.48569 | +0.48816 |
| 20260924 | +0.29725 | +0.29725 | +0.43828 | +0.44590 | +0.48495 | +0.48436 |
| 20260925 | +0.30194 | +0.30194 | +0.44135 | +0.45316 | +0.48382 | +0.49233 |

C reproduces the admitted sequence-only correlation to five decimals at every split, and raw likelihood and raw profile correlations reproduce their admitted values identically. The 13 mutation-local profile coordinates add +0.0076 to +0.0118 Spearman over the admitted profile-plus-sequence control.

All tables below report dimensionless within-assay Spearman correlations or paired differences, averaged within wild-type cluster and then equally across the 163 clusters. Brackets are exploratory 95% percentile intervals from 2,000 paired cluster bootstrap draws, conditional on the fitted cross-validation predictions; the resampling unit is the wild-type family at 50% identity. Rank-MSE differences are in squared standardized-rank units.

## Control-set strength

C, C+L, C+P and C+L+P contain no model output, so their fitted predictions are bit-identical across all five arms. Their correlations, at the primary split and as a point range over the three splits:

| Control set | primary split 20260923 | point range across the three splits |
| --- | --- | --- |
| C | +0.30024 [+0.27330, +0.32841] | +0.29725 to +0.30194 |
| C+L | +0.25277 [+0.22628, +0.27928] | +0.25277 to +0.25887 |
| C+P | +0.44659 [+0.41803, +0.47326] | +0.44590 to +0.45316 |
| C+L+P | +0.42975 [+0.40267, +0.45584] | +0.42855 to +0.43269 |

C+L+P+T is arm-specific because T depends on the tokenizer, and differs only between the two strata: +0.42972 [+0.40266, +0.45585] for the two residue-tokenizer arms and +0.42968 [+0.40270, +0.45586] for the three subword arms at the primary split, spanning +0.42833 to +0.43292 over the three splits. The unfitted scalar comparators are the admitted ones unchanged:

| Model | raw M | raw P |
| --- | --- | --- |
| ProGen3-3B | +0.37866 [+0.35136, +0.40529] | +0.36066 [+0.33207, +0.38910] |
| ProteinGLM | +0.38984 [+0.36161, +0.41669] | +0.36066 [+0.33207, +0.38910] |
| Llama-2 parent | -0.00961 [-0.02917, +0.00957] | +0.36066 [+0.33207, +0.38910] |
| ProLLaMA Stage 1 | +0.14338 [+0.11456, +0.17101] | +0.36066 [+0.33207, +0.38910] |
| ProLLaMA Stage 2 | +0.15595 [+0.12674, +0.18482] | +0.36066 [+0.33207, +0.38910] |

Containment does not imply better transfer. Adding the 656-coordinate local-pattern block lowers the control's held-cluster correlation by 0.04747 from C to C+L and by 0.01684 from C+P to C+L+P at the primary split, and the sign of that change is the same at both other splits. The tokenisation block changes the control by at most 0.00007. Only the profile block improves the control, by +0.14635 from C to C+P.

## Likelihood increment

| Model | Control set | 20260923 | 20260924 | 20260925 | point range |
| --- | --- | --- | --- | --- | --- |
| ProGen3-3B | C | +0.12773 [+0.11054, +0.14546] | +0.13295 [+0.11547, +0.15020] | +0.12963 [+0.11196, +0.14734] | +0.12773 to +0.13295 |
| ProGen3-3B | C+L | +0.14719 [+0.13057, +0.16427] | +0.14830 [+0.13263, +0.16445] | +0.14623 [+0.12956, +0.16309] | +0.14623 to +0.14830 |
| ProGen3-3B | C+P | +0.03649 [+0.02820, +0.04487] | +0.03821 [+0.03025, +0.04587] | +0.03862 [+0.02813, +0.04915] | +0.03649 to +0.03862 |
| ProGen3-3B | C+L+P | +0.03948 [+0.03233, +0.04667] | +0.04035 [+0.03334, +0.04747] | +0.03952 [+0.03241, +0.04673] | +0.03948 to +0.04035 |
| ProGen3-3B | C+L+P+T | +0.03952 [+0.03249, +0.04675] | +0.04058 [+0.03376, +0.04775] | +0.03935 [+0.03230, +0.04654] | +0.03935 to +0.04058 |
| ProteinGLM | C | +0.13394 [+0.11587, +0.15266] | +0.14108 [+0.12303, +0.15863] | +0.13679 [+0.11832, +0.15513] | +0.13394 to +0.14108 |
| ProteinGLM | C+L | +0.14934 [+0.13372, +0.16608] | +0.15144 [+0.13620, +0.16780] | +0.14880 [+0.13287, +0.16520] | +0.14880 to +0.15144 |
| ProteinGLM | C+P | +0.04157 [+0.03189, +0.05045] | +0.03846 [+0.03051, +0.04641] | +0.03917 [+0.02920, +0.04927] | +0.03846 to +0.04157 |
| ProteinGLM | C+L+P | +0.03897 [+0.03191, +0.04626] | +0.04024 [+0.03298, +0.04749] | +0.03986 [+0.03217, +0.04723] | +0.03897 to +0.04024 |
| ProteinGLM | C+L+P+T | +0.03907 [+0.03209, +0.04634] | +0.04039 [+0.03312, +0.04760] | +0.03969 [+0.03195, +0.04704] | +0.03907 to +0.04039 |
| Llama-2 parent | C | -0.00049 [-0.00088, -0.00009] | -0.00002 [-0.00036, +0.00030] | -0.00013 [-0.00051, +0.00022] | -0.00049 to -0.00002 |
| Llama-2 parent | C+L | -0.00058 [-0.00094, -0.00023] | -0.00013 [-0.00047, +0.00020] | -0.00005 [-0.00033, +0.00024] | -0.00058 to -0.00005 |
| Llama-2 parent | C+P | -0.00034 [-0.00089, +0.00020] | +0.00018 [-0.00037, +0.00074] | +0.00002 [-0.00044, +0.00050] | -0.00034 to +0.00018 |
| Llama-2 parent | C+L+P | -0.00010 [-0.00047, +0.00027] | +0.00033 [-0.00003, +0.00070] | +0.00014 [-0.00025, +0.00053] | -0.00010 to +0.00033 |
| Llama-2 parent | C+L+P+T | -0.00004 [-0.00043, +0.00034] | +0.00023 [-0.00011, +0.00057] | -0.00014 [-0.00052, +0.00025] | -0.00014 to +0.00023 |
| ProLLaMA Stage 1 | C | +0.02656 [+0.01620, +0.03666] | +0.02941 [+0.01908, +0.03903] | +0.02833 [+0.01825, +0.03832] | +0.02656 to +0.02941 |
| ProLLaMA Stage 1 | C+L | +0.02787 [+0.01919, +0.03658] | +0.03083 [+0.02328, +0.03879] | +0.02969 [+0.02083, +0.03804] | +0.02787 to +0.03083 |
| ProLLaMA Stage 1 | C+P | +0.00629 [+0.00200, +0.01051] | +0.00456 [+0.00197, +0.00716] | +0.00363 [+0.00076, +0.00646] | +0.00363 to +0.00629 |
| ProLLaMA Stage 1 | C+L+P | +0.00409 [+0.00170, +0.00665] | +0.00463 [+0.00240, +0.00701] | +0.00464 [+0.00222, +0.00710] | +0.00409 to +0.00464 |
| ProLLaMA Stage 1 | C+L+P+T | +0.00400 [+0.00163, +0.00649] | +0.00454 [+0.00230, +0.00705] | +0.00446 [+0.00208, +0.00683] | +0.00400 to +0.00454 |
| ProLLaMA Stage 2 | C | +0.02825 [+0.01700, +0.03918] | +0.03082 [+0.02013, +0.04088] | +0.03015 [+0.01914, +0.04091] | +0.02825 to +0.03082 |
| ProLLaMA Stage 2 | C+L | +0.03215 [+0.02305, +0.04146] | +0.03546 [+0.02756, +0.04355] | +0.03351 [+0.02459, +0.04211] | +0.03215 to +0.03546 |
| ProLLaMA Stage 2 | C+P | +0.00616 [+0.00191, +0.01035] | +0.00612 [+0.00317, +0.00908] | +0.00509 [+0.00203, +0.00822] | +0.00509 to +0.00616 |
| ProLLaMA Stage 2 | C+L+P | +0.00552 [+0.00288, +0.00818] | +0.00582 [+0.00322, +0.00847] | +0.00624 [+0.00344, +0.00891] | +0.00552 to +0.00624 |
| ProLLaMA Stage 2 | C+L+P+T | +0.00561 [+0.00296, +0.00826] | +0.00595 [+0.00336, +0.00862] | +0.00610 [+0.00334, +0.00877] | +0.00561 to +0.00610 |

Both native protein arms' likelihood increments are above zero in every split and every control set, but adding the profile block cuts them by a factor of 3.2 to 3.5, from +0.12773 to +0.03649 (ProGen3) and from +0.13394 to +0.04157 (ProteinGLM) at the primary split. Both ProLLaMA stages keep a resolved likelihood increment over every control set: +0.027 to +0.035 over C and C+L, and +0.0036 to +0.0063 over the three profile control sets. The text parent's likelihood adds nothing at any level: every point estimate is within 0.0006 of zero, and the two intervals that exclude zero (over C and C+L at split 20260923) do so at magnitudes of 0.00049 and 0.00058.

## Representation increment

| Model | Control set | 20260923 | 20260924 | 20260925 | point range |
| --- | --- | --- | --- | --- | --- |
| ProGen3-3B | C | +0.10635 [+0.08910, +0.12391] | +0.10890 [+0.09263, +0.12488] | +0.11112 [+0.09429, +0.12727] | +0.10635 to +0.11112 |
| ProGen3-3B | C+L | +0.13577 [+0.11749, +0.15384] | +0.13540 [+0.11767, +0.15222] | +0.13460 [+0.11767, +0.15096] | +0.13460 to +0.13577 |
| ProGen3-3B | C+P | +0.04104 [+0.02853, +0.05373] | +0.03937 [+0.02750, +0.05159] | +0.03666 [+0.02386, +0.04975] | +0.03666 to +0.04104 |
| ProGen3-3B | C+L+P | +0.04341 [+0.03203, +0.05479] | +0.04388 [+0.03288, +0.05494] | +0.04341 [+0.03187, +0.05506] | +0.04341 to +0.04388 |
| ProGen3-3B | C+L+P+T | +0.04427 [+0.03283, +0.05560] | +0.04450 [+0.03353, +0.05590] | +0.04398 [+0.03267, +0.05569] | +0.04398 to +0.04450 |
| ProteinGLM | C | +0.14306 [+0.12268, +0.16302] | +0.14665 [+0.12756, +0.16548] | +0.14307 [+0.12499, +0.16172] | +0.14306 to +0.14665 |
| ProteinGLM | C+L | +0.16514 [+0.14569, +0.18528] | +0.16665 [+0.14902, +0.18576] | +0.15953 [+0.14195, +0.17744] | +0.15953 to +0.16665 |
| ProteinGLM | C+P | +0.05347 [+0.04008, +0.06778] | +0.05570 [+0.04322, +0.06941] | +0.04720 [+0.03362, +0.06081] | +0.04720 to +0.05570 |
| ProteinGLM | C+L+P | +0.05769 [+0.04518, +0.07128] | +0.05984 [+0.04729, +0.07352] | +0.05376 [+0.04163, +0.06641] | +0.05376 to +0.05984 |
| ProteinGLM | C+L+P+T | +0.05867 [+0.04608, +0.07226] | +0.06072 [+0.04823, +0.07427] | +0.05478 [+0.04256, +0.06742] | +0.05478 to +0.06072 |
| Llama-2 parent | C | -0.03348 [-0.04840, -0.01977] | -0.01798 [-0.03074, -0.00689] | -0.02357 [-0.03721, -0.01200] | -0.03348 to -0.01798 |
| Llama-2 parent | C+L | -0.02092 [-0.03406, -0.00903] | -0.00574 [-0.01917, +0.00497] | -0.01549 [-0.02740, -0.00459] | -0.02092 to -0.00574 |
| Llama-2 parent | C+P | -0.00244 [-0.01113, +0.00512] | +0.00024 [-0.00846, +0.00798] | -0.00945 [-0.01956, -0.00080] | -0.00945 to +0.00024 |
| Llama-2 parent | C+L+P | +0.00149 [-0.00617, +0.00830] | +0.00432 [-0.00248, +0.01087] | -0.00067 [-0.00831, +0.00637] | -0.00067 to +0.00432 |
| Llama-2 parent | C+L+P+T | +0.00169 [-0.00580, +0.00836] | +0.00470 [-0.00212, +0.01133] | -0.00061 [-0.00818, +0.00640] | -0.00061 to +0.00470 |
| ProLLaMA Stage 1 | C | +0.00720 [-0.01089, +0.02566] | +0.00844 [-0.00897, +0.02646] | -0.00904 [-0.02824, +0.01065] | -0.00904 to +0.00844 |
| ProLLaMA Stage 1 | C+L | +0.02340 [+0.00503, +0.04290] | +0.02232 [+0.00385, +0.04226] | +0.01363 [-0.00497, +0.03243] | +0.01363 to +0.02340 |
| ProLLaMA Stage 1 | C+P | +0.01408 [+0.00056, +0.02796] | +0.01031 [-0.00246, +0.02433] | +0.00129 [-0.01308, +0.01555] | +0.00129 to +0.01408 |
| ProLLaMA Stage 1 | C+L+P | +0.01555 [+0.00407, +0.02748] | +0.01365 [+0.00283, +0.02575] | +0.00783 [-0.00335, +0.01961] | +0.00783 to +0.01555 |
| ProLLaMA Stage 1 | C+L+P+T | +0.01577 [+0.00437, +0.02766] | +0.01394 [+0.00317, +0.02601] | +0.00814 [-0.00297, +0.02000] | +0.00814 to +0.01577 |
| ProLLaMA Stage 2 | C | +0.00552 [-0.01401, +0.02473] | +0.01576 [-0.00053, +0.03175] | +0.00583 [-0.01313, +0.02395] | +0.00552 to +0.01576 |
| ProLLaMA Stage 2 | C+L | +0.03086 [+0.01206, +0.05000] | +0.02886 [+0.01015, +0.04766] | +0.02618 [+0.00717, +0.04441] | +0.02618 to +0.03086 |
| ProLLaMA Stage 2 | C+P | +0.01446 [+0.00051, +0.02890] | +0.01366 [+0.00076, +0.02784] | +0.00413 [-0.01178, +0.01887] | +0.00413 to +0.01446 |
| ProLLaMA Stage 2 | C+L+P | +0.01615 [+0.00367, +0.02866] | +0.01582 [+0.00437, +0.02790] | +0.01113 [-0.00128, +0.02356] | +0.01113 to +0.01615 |
| ProLLaMA Stage 2 | C+L+P+T | +0.01670 [+0.00433, +0.02912] | +0.01649 [+0.00500, +0.02864] | +0.01156 [-0.00076, +0.02406] | +0.01156 to +0.01670 |

## Representation increment after the likelihood

This is the analogue of the admitted Readout study's primary contrast, which measures the representation against a control that already contains the likelihood.

| Model | Control set | 20260923 | 20260924 | 20260925 | point range |
| --- | --- | --- | --- | --- | --- |
| ProGen3-3B | C | +0.02395 [+0.01378, +0.03452] | +0.01983 [+0.01118, +0.02887] | +0.02194 [+0.01220, +0.03210] | +0.01983 to +0.02395 |
| ProGen3-3B | C+L | +0.03197 [+0.02188, +0.04267] | +0.02947 [+0.02062, +0.03916] | +0.03041 [+0.02148, +0.03966] | +0.02947 to +0.03197 |
| ProGen3-3B | C+P | +0.02122 [+0.01108, +0.03157] | +0.01775 [+0.00882, +0.02664] | +0.01392 [+0.00422, +0.02459] | +0.01392 to +0.02122 |
| ProGen3-3B | C+L+P | +0.02240 [+0.01378, +0.03154] | +0.02191 [+0.01341, +0.03094] | +0.02172 [+0.01328, +0.03056] | +0.02172 to +0.02240 |
| ProGen3-3B | C+L+P+T | +0.02266 [+0.01408, +0.03195] | +0.02196 [+0.01359, +0.03103] | +0.02196 [+0.01354, +0.03081] | +0.02196 to +0.02266 |
| ProteinGLM | C | +0.04346 [+0.02999, +0.05615] | +0.04224 [+0.03037, +0.05416] | +0.03954 [+0.02715, +0.05127] | +0.03954 to +0.04346 |
| ProteinGLM | C+L | +0.05685 [+0.04288, +0.07008] | +0.05776 [+0.04524, +0.06998] | +0.05205 [+0.03946, +0.06404] | +0.05205 to +0.05776 |
| ProteinGLM | C+P | +0.02701 [+0.01588, +0.03855] | +0.03175 [+0.02098, +0.04332] | +0.02267 [+0.01132, +0.03406] | +0.02267 to +0.03175 |
| ProteinGLM | C+L+P | +0.03491 [+0.02415, +0.04581] | +0.03670 [+0.02654, +0.04762] | +0.03022 [+0.01999, +0.04085] | +0.03022 to +0.03670 |
| ProteinGLM | C+L+P+T | +0.03518 [+0.02450, +0.04613] | +0.03698 [+0.02685, +0.04801] | +0.03112 [+0.02103, +0.04165] | +0.03112 to +0.03698 |
| Llama-2 parent | C | -0.03292 [-0.04779, -0.01927] | -0.01742 [-0.03033, -0.00631] | -0.02300 [-0.03661, -0.01123] | -0.03292 to -0.01742 |
| Llama-2 parent | C+L | -0.02024 [-0.03352, -0.00820] | -0.00534 [-0.01881, +0.00532] | -0.01521 [-0.02711, -0.00435] | -0.02024 to -0.00534 |
| Llama-2 parent | C+P | -0.00232 [-0.01100, +0.00531] | +0.00007 [-0.00853, +0.00790] | -0.00950 [-0.01974, -0.00097] | -0.00950 to +0.00007 |
| Llama-2 parent | C+L+P | +0.00136 [-0.00617, +0.00798] | +0.00405 [-0.00270, +0.01058] | -0.00086 [-0.00849, +0.00612] | -0.00086 to +0.00405 |
| Llama-2 parent | C+L+P+T | +0.00160 [-0.00588, +0.00829] | +0.00434 [-0.00238, +0.01084] | -0.00054 [-0.00812, +0.00641] | -0.00054 to +0.00434 |
| ProLLaMA Stage 1 | C | -0.00680 [-0.02315, +0.00934] | -0.00833 [-0.02390, +0.00669] | -0.01637 [-0.03232, -0.00039] | -0.01637 to -0.00680 |
| ProLLaMA Stage 1 | C+L | +0.00487 [-0.01193, +0.02247] | +0.00099 [-0.01594, +0.01838] | -0.00533 [-0.02185, +0.01137] | -0.00533 to +0.00487 |
| ProLLaMA Stage 1 | C+P | +0.00891 [-0.00466, +0.02290] | +0.00722 [-0.00559, +0.02056] | -0.00105 [-0.01524, +0.01267] | -0.00105 to +0.00891 |
| ProLLaMA Stage 1 | C+L+P | +0.01268 [+0.00184, +0.02399] | +0.01033 [-0.00031, +0.02179] | +0.00459 [-0.00645, +0.01575] | +0.00459 to +0.01268 |
| ProLLaMA Stage 1 | C+L+P+T | +0.01319 [+0.00230, +0.02439] | +0.01085 [+0.00031, +0.02227] | +0.00524 [-0.00574, +0.01636] | +0.00524 to +0.01319 |
| ProLLaMA Stage 2 | C | -0.00510 [-0.02101, +0.01035] | -0.00336 [-0.01725, +0.01019] | -0.01503 [-0.03135, +0.00015] | -0.01503 to -0.00336 |
| ProLLaMA Stage 2 | C+L | +0.00554 [-0.01087, +0.02194] | +0.00461 [-0.01112, +0.02076] | -0.00069 [-0.01680, +0.01523] | -0.00069 to +0.00554 |
| ProLLaMA Stage 2 | C+P | +0.00955 [-0.00429, +0.02369] | +0.00845 [-0.00402, +0.02220] | -0.00035 [-0.01521, +0.01325] | -0.00035 to +0.00955 |
| ProLLaMA Stage 2 | C+L+P | +0.01174 [+0.00008, +0.02359] | +0.01134 [+0.00055, +0.02271] | +0.00612 [-0.00547, +0.01799] | +0.00612 to +0.01174 |
| ProLLaMA Stage 2 | C+L+P+T | +0.01188 [+0.00016, +0.02380] | +0.01174 [+0.00106, +0.02310] | +0.00645 [-0.00517, +0.01811] | +0.00645 to +0.01188 |

The admitted study's corresponding contrast, representation over B = P+M+S, is +0.00739 to +0.01736 for ProGen3 with the 20260924 interval crossing zero, +0.01970 to +0.02255 for ProteinGLM with all three intervals above zero, −0.02064 to −0.01598 for the parent with all three below zero, and negative with at least one split below zero for both ProLLaMA stages.

Over C+P+M and C+L+P+M the two native protein arms' increments are above zero in all three splits, which is a stronger resolution than the admitted result for ProGen3. The negative increments of the other three arms largely do not persist: over C+L+P+M and C+L+P+T+M no arm has an interval below zero, the parent's three intervals all cross zero, and both ProLLaMA stages have two of three splits above zero over C+L+P+T+M. Over the smaller C+P+M the parent still has one split resolved below zero (−0.00950 at 20260925) and ProLLaMA Stage 1 one split resolved below zero over C+M (−0.01637 at 20260925). Because C+L+P+M is a weaker predictor of held-out clusters than B is, part of that movement is the baseline losing ground rather than the representation gaining it, and the two cannot be separated by this comparison.

## Rank-mean-squared-error reduction

| Model | Control set | adding M | adding R | adding R after M |
| --- | --- | --- | --- | --- |
| ProGen3-3B | C | +0.06135 [-0.00760, +0.10203] | +0.06961 [+0.04302, +0.10546] | +0.06830 [+0.01289, +0.17406] |
| ProGen3-3B | C+L | +0.10465 [+0.09099, +0.12083] | +0.07630 [+0.05137, +0.11275] | +0.03246 [+0.01402, +0.05761] |
| ProGen3-3B | C+P | +0.06087 [+0.03599, +0.10591] | +0.08329 [+0.01582, +0.21427] | +0.04732 [-0.00042, +0.14205] |
| ProGen3-3B | C+L+P | +0.04016 [+0.03276, +0.04979] | +0.04138 [+0.01902, +0.07326] | +0.02684 [+0.00985, +0.05108] |
| ProGen3-3B | C+L+P+T | +0.04014 [+0.03284, +0.04970] | +0.04169 [+0.01919, +0.07348] | +0.02684 [+0.00986, +0.05133] |
| ProteinGLM | C | +0.08197 [+0.02574, +0.11944] | +0.11383 [+0.08454, +0.15445] | +0.07502 [+0.02137, +0.16982] |
| ProteinGLM | C+L | +0.10721 [+0.09371, +0.12275] | +0.12257 [+0.09055, +0.16602] | +0.05827 [+0.03605, +0.08840] |
| ProteinGLM | C+P | +0.07145 [+0.03981, +0.12814] | +0.10156 [+0.03209, +0.23473] | +0.05214 [+0.00593, +0.13645] |
| ProteinGLM | C+L+P | +0.04061 [+0.03330, +0.04992] | +0.06194 [+0.03736, +0.09930] | +0.04322 [+0.02352, +0.07195] |
| ProteinGLM | C+L+P+T | +0.04055 [+0.03339, +0.04959] | +0.06238 [+0.03755, +0.09991] | +0.04332 [+0.02341, +0.07262] |
| Llama-2 parent | C | -0.00016 [-0.00034, -0.00003] | -0.00174 [-0.01313, +0.01003] | -0.00162 [-0.01297, +0.01005] |
| Llama-2 parent | C+L | -0.00009 [-0.00030, +0.00006] | -0.00093 [-0.01296, +0.01462] | -0.00091 [-0.01285, +0.01449] |
| Llama-2 parent | C+P | -0.00035 [-0.00106, +0.00014] | +0.04455 [-0.00893, +0.14884] | +0.04479 [-0.00897, +0.14948] |
| Llama-2 parent | C+L+P | -0.00004 [-0.00025, +0.00017] | +0.00476 [-0.00585, +0.01982] | +0.00467 [-0.00584, +0.01957] |
| Llama-2 parent | C+L+P+T | -0.00004 [-0.00025, +0.00018] | +0.00481 [-0.00598, +0.01994] | +0.00471 [-0.00598, +0.01969] |
| ProLLaMA Stage 1 | C | -0.04012 [-0.18475, +0.03256] | +0.02890 [+0.01137, +0.04885] | +0.07738 [-0.00159, +0.23578] |
| ProLLaMA Stage 1 | C+L | +0.03166 [+0.02450, +0.03887] | +0.02660 [+0.01003, +0.04880] | -0.00149 [-0.01764, +0.02052] |
| ProLLaMA Stage 1 | C+P | +0.00833 [-0.00256, +0.01686] | +0.06309 [+0.00611, +0.17408] | +0.05670 [-0.00356, +0.17366] |
| ProLLaMA Stage 1 | C+L+P | +0.00429 [+0.00243, +0.00619] | +0.02276 [+0.00692, +0.04250] | +0.02067 [+0.00540, +0.04020] |
| ProLLaMA Stage 1 | C+L+P+T | +0.00428 [+0.00241, +0.00620] | +0.02291 [+0.00697, +0.04265] | +0.02082 [+0.00554, +0.04045] |
| ProLLaMA Stage 2 | C | -0.03419 [-0.17182, +0.03537] | +0.02680 [+0.00753, +0.04911] | +0.07255 [-0.00403, +0.22190] |
| ProLLaMA Stage 2 | C+L | +0.03475 [+0.02727, +0.04217] | +0.03022 [+0.01322, +0.05201] | -0.00018 [-0.01587, +0.02076] |
| ProLLaMA Stage 2 | C+P | +0.01130 [+0.00366, +0.01867] | +0.06224 [+0.00360, +0.17440] | +0.05330 [-0.00697, +0.17242] |
| ProLLaMA Stage 2 | C+L+P | +0.00594 [+0.00368, +0.00826] | +0.02089 [+0.00264, +0.04180] | +0.01743 [-0.00038, +0.03810] |
| ProLLaMA Stage 2 | C+L+P+T | +0.00594 [+0.00371, +0.00830] | +0.02109 [+0.00280, +0.04200] | +0.01760 [-0.00026, +0.03827] |

The table gives the primary split; per-split intervals for the other two splits are retained in the cell reports. Across the three splits the representation's rank-MSE reduction over C+L+P spans +0.03038 to +0.04138 (ProGen3), +0.05243 to +0.06194 (ProteinGLM), +0.00274 to +0.00775 (parent), +0.01165 to +0.02276 (Stage 1) and +0.01411 to +0.02089 (Stage 2).

Unlike the admitted study, where every representation-augmentation rank-MSE interval crossed zero, these reductions are often resolved. Counting the three splits of each arm and control set, the representation's rank-MSE reduction lies above zero in 14 of 15 cells for ProGen3 and 15 of 15 for ProteinGLM, in 7 of 15 for ProLLaMA Stage 1 and 9 of 15 for Stage 2, and in none of the 15 for the text parent, whose intervals all cross zero. Spearman ordering and rank-MSE are separate endpoints and need not agree: both ProLLaMA stages have a rank-MSE reduction above zero over C at the primary split while their Spearman increment over C crosses zero at every split.

## The survival pattern and what it licenses

For ProGen3-3B and ProteinGLM the representation increment is above zero in all fifteen control-set × split combinations of each arm. It is +0.106 to +0.111 (ProGen3) and +0.143 to +0.147 (ProteinGLM) over composition alone, rises to +0.135 to +0.136 and +0.160 to +0.167 over C+L, falls to +0.037 to +0.041 and +0.047 to +0.056 over C+P, and is +0.043 to +0.045 and +0.054 to +0.061 over C+L+P and C+L+P+T. The rises track the control sets' own losses of held-cluster correlation rather than any change in the representation, and the likelihood increment moves the same way. For the Llama-2 text parent the increment is resolved below zero over composition in all three splits, below zero in two of three splits over C+L, below zero in one of three over C+P, and unresolved in all three splits over C+L+P and C+L+P+T. For both ProLLaMA stages it is unresolved in all three splits over composition. Over the richer controls it is positive: Stage 2's increment over C+L is above zero in all three splits (+0.02618 to +0.03086), and every other ProLLaMA cell has two splits above zero and one crossing, except Stage 1 over C+P, which has one above zero and two crossing.

For the composition rung, the verdict is that composition, mutation identity, position and length do not account for the representation information in the two native protein arms. For the two ProLLaMA stages the increment over composition is unresolved at every split, which is a failure to resolve additional information rather than evidence that composition accounts for it; for the text parent the increment is resolved below zero, so adding the representation to the composition control makes its held-cluster ranking worse. For the local-motif rung there is no verdict of the intended kind, because the declared short-range statistics failed as a control: the dipeptide and tripeptide difference block does not transfer across held-out wild-type clusters at all under this readout class, lowering the control's own correlation by 0.047 from C and by 0.017 from C+P. An increment that survives C+L therefore says nothing about surviving a stronger local-pattern control, and the honest reading is that this particular declared local statistic is not a competitor to the representation on this cohort. For the profile rung the verdict is that mutation-local profile frequencies, entropy and coverage absorb about two thirds of the representation increment in the native protein arms, and that a residual increment of +0.037 to +0.056 Spearman survives them in all three splits for those two arms and only those two.

A surviving increment is bounded by the declared controls, the experimental-label budget and the readout class stated above. It identifies accessible predictive information beyond those controls on held-out wild-type clusters. It does not locate a mechanism, does not say which computation carries the information, and does not establish an irreducible biological rule at any level of sequence organisation.

## The recomputation landed, and the verdict stays provisional

All 21 of this gate's cells on the anchor panel — seven arms at three split seeds — were refitted at the reassessment's selection, a depth selection at class `C4_full_random_feature` for every arm of this panel, on identical support, folds, seeds, weighting and label budget. **The verdict stays `provisional`.**

**And for five of its seven arms the comparator was not a published cell either.** Only ProGen3-3B and ProteinGLM-7B-CLM were compared against this gate's admitted five-arm cells; ProGen2-large, ProGen2-medium, ProGen2-xlarge, ProGen3-112M and ProtGPT2 were compared against **unadmitted extension cells**, for arms this gate has never published a result for. Taken with the capacity statement below, most of that half of the batch measured something other than the published quantity **and** compared it against something other than a published cell, and either fact alone understates it. This is also the answer to the provenance puzzle that batch reported as open — the reason a cell had up to three candidate files is that the extension was run twice and some candidates were never published at all.

**The depth arm runs at half the published capacity, which every number in it inherits.** Its representation is **512 coordinates** — the `mean` and `last` summaries at 256 each, taken at the one depth the reassessment selects for that arm (ProGen3-112M 9, ProGen3-3B 11, ProtGPT2 21, ProGen2-medium 23, ProGen2-large 25, ProGen2-xlarge 28, ProteinGLM-7B-CLM 30) — against the published **1,024** from four pooled summaries. Every cell of this batch carries `selection_kind: extraction_depth` and `is_admitted_pipeline: false` in its own record; none is at the admitted pair. So depth and capacity change together here, and neither the movement reported below nor its absence is attributable to depth alone or readable as a movement of the published increment. The capacity-matched question is asked separately.

Over the four metrics this record reports, **no metric-cell crosses from an interval including zero to one excluding it** — 0 of 84 — while the numbers themselves move substantially, up to 0.06685 in a point estimate and 0.07017 in an interval endpoint. Movement of that size therefore does not by itself move a verdict here.

That zero is a count over the declared metrics and **not over the data**, and the difference is a factor of eight: the same 21 cells carry **33 crossings across 5 distinct metrics** over their full 72-summary sets, concentrated in the representation-after-likelihood families. One arm-and-metric pair survives the all-three-seed rule over that wider set, ProtGPT2 on `increment_R_after_M_C`, which this record does not report. So this gate is **not a null under the new selection**, and it must not be cited as the control that makes another gate's changes readable; the general form of that error is catalogued as L56.

The control that does hold is a measured one: across this gate and the local-context gate, **254 likelihood-family metric-cells sat close enough to zero to cross under the interval movement their own cell showed, and none crossed**, against 144 crossings in 886 at-risk cells among quantities the selection reaches. The per-design absolute correlations are not a control at all — they sit 0.21 or more from zero and only 78 of 2,310 are at risk.

Every bound stated above continues to apply, and the intervals remain unadjusted.

## The class axis, on all fifteen admitted cells

Until 2026-09-25 this gate had no class-axis recomputation: the depth batch above touched two of its five arms, so thirteen of its fifteen admitted cells had never been recomputed on any axis. All fifteen have now been refitted at the settled readout class from each cell's own inputs — its own cohort, its own five extraction manifests, the profile store the campaign declares, and the anchor arm named explicitly — **on a card, one shard per card, because every published cell of this gate was fitted on one**, which makes this a confirmation rather than a screen. **The verdict stands.**

**Every compared quantity reproduces exactly.** 1,080 summary nodes over 15 cells and 5 arms — 600 representation, 315 likelihood, 165 other — with **no node present in one report and absent from the other**, a largest point movement of **0.0**, a largest interval-endpoint movement of **0.0**, and **zero resolved sign changes**. 55 nodes sat within 1e-3 of zero and could have crossed under any movement at all; none did, because nothing moved.

**A zero that clean is shown reachable before it is reported.** The two sides are genuinely different files: **0 of 15 pairs share a file digest**, and 70 top-level fields differ across the panel — `created_utc` (2026-09-24 against 2026-09-26 on the pod's clock), `runtime.device`, `anchor_arm`, `analysis_code_sha256` and the tokenisation-interface block. The published cells ran on cards `cuda:0` through `cuda:3` and the recomputation on `cuda:0` and `cuda:1`, so **bit-identical reproduction holds across cards**. Each published report was verified against the digest its admission binds, at roster construction and again at consolidation: 15 checked, 0 failures.

**Two provenance facts follow from the fields that differ.** The published cells record an `analysis_code_sha256` that is **not** the committed script's — the fitting ran from a working-tree version that predates the commit creating this file, so that exact version is unretrievable and only its digest survives; whatever differs between it and the committed script is numerically inert on all 1,080 nodes. And the published reports record no `anchor_arm`, which earlier forced the recomputation driver to refuse rather than guess: passing `progen3-3b` explicitly reproduces every published quantity exactly, so the default that was in force is now established by measurement rather than assumed.

## Tokenisation strata

| Model | registry tokenisation | stratum | prefix tokens | tokens per residue | token-count cycle | assays with nonzero cycle | tokens outside the shared prefix and suffix, maximum |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ProGen3-3B | residue | amino-acid | 0 | 1.0000 to 1.0000 | +0 to +0 | 0 of 201 | 458 |
| ProteinGLM | residue | amino-acid | 0 | 1.0000 to 1.0000 | +0 to +0 | 0 of 201 | 458 |
| Llama-2 parent | sentencepiece | bpe | 1 | 0.5581 to 0.7667 | −3 to +5 | 201 of 201 | 311 |
| ProLLaMA Stage 1 | sentencepiece | bpe | 1 | 0.5581 to 0.7667 | −3 to +5 | 201 of 201 | 311 |
| ProLLaMA Stage 2 | sentencepiece | bpe | 1 | 0.5581 to 0.7667 | −3 to +5 | 201 of 201 | 311 |

The two strata behave as their interfaces require. On the amino-acid stratum the pooled token count equals the residue count exactly in both states, the token-count cycle is zero for every one of the 25,728 variants, and the divergent-token span is the substituted interval doubled. On the subword stratum a single substitution changes the pooled token count by −3 to +5 tokens, every assay contains variants with a nonzero cycle, and each token covers 1.30 to 1.79 residues.

Controlling for those descriptors changes the result by almost nothing. Adding T to C+L+P moves the control's own correlation by 0.00003 to 0.00007 at the primary split and by at most 0.00023 across the three splits, and moves the representation increment by at most +0.00103, at ProteinGLM and split 20260925, from +0.05376 to +0.05478. Every arm's increment keeps its sign and its interval verdict at every split. The two arms whose representation increment survives the profile control are the two on the amino-acid stratum, where there is no segmentation degree of freedom to attribute it to. The three subword arms are the three whose increments are unresolved or negative. The byte stratum has no arm in this five-arm panel, so nothing here speaks to it.

## Full-panel extension: fitted on 2026-09-24, and bound by no admission record

After the five-arm result above was complete and admitted, the readout expansion campaign passed its own final admission, and extending this contrast to its panel was authorised. This section described the extension as prepared but never fitted until 2026-09-25, when a check of the gate's own class-axis coverage found its cells on disk. **The extension was in fact run, twice, on 2026-09-24**: 207 cell reports under `ccx_*` across four runs — 197 `complete` and 10 `smoke` interface cells — covering **99 distinct cells, all 33 arms at all three split seeds**, 189 of them fitted on CPU and 18 on a card, created between 03:00 and 10:24 UTC. Two of those runs are full panels (99 and 92 cells), which is why a later recomputation found up to three candidate files for one cell.

**Status, stated once and completely: fitted, 99 distinct cells over 33 arms at three split seeds, unadmitted, claimed by no document, and not to be cited.** No admission record binds any extension cell; the only admission this gate carries is `crossed_controls_admission.json`, which binds the fifteen five-arm cells above and nothing else. This document reports no quantity from any of them.

**They will not be admitted, and the reason is the rule this programme applies everywhere else: an admission written after the fit is not an admission but a ratification.** It would record that numbers exist, not that a support, a fold construction and a control ladder were fixed before anything was fitted — and nothing about these four runs across two days shows that they were. Retro-binding a digest to them would manufacture precisely the provenance L55 says a digest cannot supply. If a 33-arm crossed-controls panel is wanted, it is run under a declared admission; these cells are a reason to run one, not its substance. They are therefore recorded and left, neither recomputed nor bound.

One consequence has already been felt, and it is recorded where the recomputation is described: the depth-arm batch compared five of its seven crossed-control arms against these unadmitted extension cells while treating them as published, for arms this gate has never published a result for.

The rest of this section is the extension's preparation and interface qualification, which stand as written.

The extension panel is the 33 unconditioned arms of the expanded roster (SHA256 `1bf26a6ae8d71e3c0f790f12c92da7d3dd4a9add7be6f7ef807836297e12dd0c`) on the same cohort hash as the five-arm study, at the same three split seeds: 99 cells. ZymCTRL is excluded because its EC-conditioned 40-assay panel is a different support, and this entry point fits the anchor only. Each cell loads its own extraction manifest plus the ProGen3-3B anchor manifest, which is the scheduling device the readout protocol already declares, and is refused unless the realised common support is exactly that 201-assay anchor — an arm covering less of the anchor would need a separately named exact intersection, which this entry point does not fit. The five arms already reported would be refitted in that two-manifest form, so the extension table would be internally uniform and would reproduce their numbers as a check.

The interface qualification ran nine assay-limited cells, one per packing and rendering family. Seven passed end to end, with the recomputed maximum packed-token count matching the production extraction manifest for every assay and the recomputed profile scores agreeing to 1.8 × 10⁻¹⁵: ProtGPT2, ProGen2-small, ProtGPT3-1.3B, RITA-XL, Galactica-125M, InstructProtein and ProGen3-112M. Two literal-amino-acid-string text arms, GPT-2 and ByGPT5-small-en, failed after building every feature block, on one interface descriptor that asked a native-packing helper for a row prefix the text rendering does not have. That descriptor is now read off the production packing itself, as the leading and trailing marker-token counts of the packed wild-type row, which is defined for every family; the fix passes its unit tests but has not been re-run on H200.

Preparing the extension changed the entry point only. Every module that determines a fitted number — the panel loader, the crossed-control evaluation, the readout analysis, the profile channel — is byte-identical to the hashes the five-arm cell reports record, so the primary result above remains reproducible from those hashes and the extension's five refitted arms would have to return the same numbers.

The qualification already changes one thing about how the tokenisation strata must be read. Galactica-125M's registry label is `bpe`, but its protein-delimiter rendering emits exactly one token per residue with a token-count cycle of zero on every variant, so on the quantity that matters it has no segmentation degree of freedom at all. The declared label and the measured segmentation are therefore reported separately, and the measured one is what a segmentation attribution turns on. Among the nine qualified families the measured segmentation is one token per residue for ProGen2, ProGen3, ProtGPT3, RITA, InstructProtein and Galactica, and multi-residue for ProtGPT2 at 2.84 to 3.11 residues per token with cycles of −2 to +2 tokens.

Two caveats must travel with any later extension report. The five arms re-extracted for the expansion — Qwen2.5-7B, DialoGPT-small, ByGPT5-medium-en, ProtGPT3-1.3B and Galactica-125M — were re-run at batch size one after exceeding the 0.001 precision-drift gate at batch eight, and their batch-one requalification is structurally vacuous: at batch size one the production forward and the reference forward are the same computation, so the zero difference is by construction. Their extractions are admitted and internally consistent and must not be described as numerically validated. And the expansion is itself an exploratory extension whose outcomes have already been inspected, so a crossed-control reading of it cannot be presented as an outcome-blind confirmation of the five-arm result.

## Conditions the tests enforce

`tests/test_crossed_controls.py` covers the three conditions that have to hold for any of these numbers to be readable as an increment, plus the exactness and the refusals of the three new feature blocks.

Identical support and folds across every compared fit. All twenty designs of one cell are fitted on the same rows in the same order with the same split seed, and the evaluation refuses a run in which any design's realised outer or inner group membership differs from the shared map. Across cells, the collection step refuses divergent cohort bytes, assay support, assay counts, control-set definitions or group membership at a shared split seed, and requires bit-identical predictions for every model-independent control design — C, C+L, C+P and C+L+P contain no model output, so five arms fitted at one split seed must produce the same numbers from them.

No held-out group label reaching feature scaling or tuning. Replacing the measured effects of every group held out in one outer fold with arbitrary values must leave that fold's held-out predictions and its selected ridge penalty unchanged, for every design. The fold records are also required to have no family in both the held and training sets, and no inner validation family drawn from the held set.

Genuinely nested control sets. Containment is checked on the assembled design matrices' actual columns, not on their declared names: C's columns are a strict subset of C+L's and of C+P's, both are strict subsets of C+L+P's, C+L+P's are a strict subset of C+L+P+T's, neither C+L nor C+P contains the other, and every control set's columns survive into each of its additions.

The block tests check the dipeptide and tripeptide differences against a brute-force recount of both complete sequences, including overlapping substitutions that share a window; the profile block's mean per-substitution log-odds against the admitted lookup score; and the tokenisation block's prefix, suffix, cycle and changed-token arithmetic on a residue-tokenizer case and a multi-residue case. Each block refuses a length-changing variant, a wild-type residue that disagrees with the mutation string, a repeated or out-of-range position, a noncanonical residue, a mutant sequence that does not match its mutation string, a profile whose length disagrees with the wild type, a nonpositive background or pseudocount, and an empty token span. Sixteen tests pass.

## Limitations

The readout class is deliberately bounded. Both high-dimensional blocks are fixed label-independent Gaussian projections — 1,024 coordinates from the four pooled hidden-state summaries and 256 coordinates from the 8,000 tripeptide differences — and the predictor is linear ridge with the admitted penalty grid. A null or negative increment is a statement about that projected linear class at this label budget, not about the information in the uncompressed features. The tripeptide compression is the single most likely reason the local-pattern block failed as a control, and that failure cannot be distinguished here from the block genuinely carrying no transferable information.

Nested does not mean monotonically stronger, and on this cohort it is not. C+L predicts held-out clusters worse than C, and C+L+P worse than C+P, so an increment measured over the wider set is not evidence of surviving a stronger control. The verdicts above therefore rest on C+P, the strongest control set measured, and the C+L and C+L+P columns are reported for completeness rather than as the binding comparison.

The grouping unit is the existing single-linkage wild-type cluster at 50% identity with 80% coverage of the shorter sequence. It is not a certified Pfam-family or CATH-superfamily holdout and does not exclude overlap with pretraining corpora. The intervals are exploratory percentile intervals conditional on the fitted cross-validation predictions; they omit training and tuning variation and are not adjusted across the five arms, five control sets, three increment kinds, two endpoints or three split seeds. The three split seeds give a sensitivity range on one cohort, not independent replication.

The profile block reads the retained column-frequency arrays, which are normalised over each column's own weight, so per-column effective support is not recoverable from them; coverage is therefore the supported-or-unsupported indicator together with the wild-type-level alignment depth and identity the profile record retains. The four wild-type-level coordinates are constant within an assay, so they cannot change the within-assay Spearman endpoint at all; they act only on the rank-mean-squared-error endpoint and on how the penalty is spent.

The local-pattern block is a declared short-range sequence statistic over the complete sequence, not a motif model, and a dipeptide or tripeptide difference is not evidence of motif function. No label-permutation negative control was run in this campaign; the admitted Readout study's shuffled-label diagnostics, including its two positive native-support outcomes, remain the disclosed evidence on fitting artifacts, and the held-group label-corruption test here is an exactness check rather than a calibrated null. Native-support sensitivities were not refitted: every cell uses the common 201-assay anchor, so the ten extra Llama-lineage assays are outside this study. The panel is the original five admitted arms; the full-panel extension above would broaden the descriptive comparison and populate the byte stratum, and it is prepared rather than fitted.

Finally, the comparison is descriptive across unmatched checkpoints. The two arms that retain an increment over the profile control differ from the other three in architecture, tokenisation, training corpus and objective at once, so nothing here identifies which of those differences matters.
