# Does frozen model output carry information about substitution effects on domains whose relatives the corpus does not retrieve?

This is the remote-homology gate of the [Direction-1 capability map](D1_CAPABILITY_MAP_PLAN.md), qualified on its own support with its own instrument, its own controls and its own admission condition. A result at any other gate does not determine this one.

The gate exists because every cohort measured so far sits at high retrieval identity. The [folding-and-stability gate](D1_GATE_STABILITY.md) found no alignment-detectable residual family structure among its 101 one-per-group wild types at all, so its remote stratification bounds dependence on a chance-level nearest-neighbour set and certifies no remote-family disjointness. The [retrieval-and-memorization gate](D1_GATE_RETRIEVAL_MEMORIZATION.md) reaches 1, 2 and 12 groups in its three lower identity bands against an eight-unit percentile floor. Cho and Tsuboyama 2026's MGnify-derived libraries populate those bands: on the declared support below, **100 family groups are entirely remote and 66 entirely close**, both clearing that floor.

One thing has to be said before any number. This endpoint is **not** external confirmation of anything. It shares the cDNA-display proteolysis assay technology and an author with Tsuboyama 2023, which is one of the two development sources, so it reproduces the shared-source defect the catalogue records as L50 and may not be read as independent replication. What it supplies is remoteness, not independence. External confirmation is a separate gate on a separate source and separate assay, recorded in [D1_EXTERNAL_CONFIRMATION.md](D1_EXTERNAL_CONFIRMATION.md).

## The endpoint

**e = deltaG(variant) − deltaG(background)** on the staged table's combined `deltaG` column, in kcal/mol, inferred jointly from the trypsin and chymotrypsin channels. Larger deltaG is more stable, so a destabilising single substitution gives a negative e. That is the same two-estimate difference and the same sign convention the folding-and-stability gate measures on Tsuboyama 2023, so the two are commensurable quantities measured on two sources.

The admission rule is the tightened per-channel one the staged stability cohort already applies, not a new one: each of the combined, trypsin and chymotrypsin 95% interval widths inside [0, 0.5] kcal/mol, applied to the variant row **and** to its background's wild-type row, because the endpoint is their difference. Constructs are single amino-acid substitutions only, and every one is re-derived against both sequences: the substitution token's own wild-type residue, mutant residue and position must agree with the two strings, which must differ at that one position and nowhere else. Insertion and deletion constructs change the length and fail that check rather than entering a substitution support length-matched.

Aggregation is the median over admitted rows per (background, mutant sequence). On this support no state carried more than one admitted row, so the aggregation is an identity here and is declared rather than exercised.

Source: Zenodo DOI 10.5281/zenodo.19411306, `230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv`, sha256 `1607b877…42e1c02`. Endpoint digest over the declared cohort `3e7cf0ca040a94f9c25a3601e3657d50b3c865036fa0910bb2c3e6b9d79aea14`.

## The floor, on two scales, and why the difference decides whether this gate is readable

The two protease channels of this table disagree far more than they do on the admitted MegaScale endpoint, and the disagreement is systematic rather than symmetric. Over the 1,413,334 rows the frozen endpoint qualification admits, trypsin minus chymotrypsin deltaG has mean **−0.478** and root-mean-square **0.824 kcal/mol** against a combined-deltaG spread of 1.251, with Pearson r = 0.857 — a level-scale floor two thirds of the endpoint's own variation. The admitted MegaScale endpoint's per-channel discordance is 0.2337 [0.2209, 0.2458] kcal/mol against a shared component of 0.9013.

A mean offset of that size is common to a variant and to its own background, so it cancels in the difference this endpoint takes. That is a claim about the arithmetic, and it is measured rather than asserted. Both scales are reported here and every statement below names which one it rests on. Agreement is decomposed the way the folding-and-stability gate decomposes its own channels: the shared-component standard deviation is the square root of the between-channel covariance and is an **upper** bound on reproducible signal, because assay error common to both channels contributes to it; the per-channel discordance standard deviation is the standard deviation of the channel difference over the square root of two and is a **lower** bound on per-channel noise, because that common error cancels. No oracle ceiling is built from them. Weighting is nested — variants equal inside a mutated site, sites equal inside a background, backgrounds equal inside the family group — and intervals are 2,000-draw group-bootstrap percentile intervals at seed 20260923 over the 179 family groups.

| Quantity, kcal/mol unless dimensionless | Level scale, admitted rows of this support | Effect scale, the endpoint |
| --- | --- | --- |
| Channel offset, trypsin − chymotrypsin | −0.4205 | **+0.1340 [+0.0799, +0.1921]** |
| Root-mean-square channel difference | 0.7855 | 0.5347 [0.4639, 0.6267] |
| **Per-channel discordance standard deviation** | 0.4692 | **0.3660 [0.3174, 0.4268]** |
| Shared-component standard deviation | 0.7573 | 0.5731 [0.5280, 0.6157] |
| Pearson r | +0.7231 | +0.7106 [+0.6315, +0.7739] |
| Shared-to-discordance ratio | 1.614 | **1.566 [1.301, 1.847]** |
| Signal standard deviation | 0.8907 (channel mean) | 0.6456 |

The offset does cancel: −0.4205 kcal/mol on the level scale becomes +0.1340 [+0.0799, +0.1921] on the effect scale, a residual that is resolved away from zero but is a quarter of the level-scale value. With the background as the resampling unit instead of the family group the discordance is 0.3543 [0.3172, 0.3937] over 305 backgrounds, so the floor is not an artefact of the unit choice.

**The endpoint is admitted, and its headroom is about half the development endpoint's.** Agreement is bounded far from zero at both resampling units, and 179 family groups support held-group folds. But the reproducible component exceeds the per-channel discordance by a factor of **1.566 [1.301, 1.847]**, against **3.86 [3.61, 4.13]** for the admitted MegaScale endpoint on the same kind of decomposition. A model quantity has roughly half as much resolvable room here as it had there, on a signal standard deviation of 0.6456 against that endpoint's 0.9777. That is the reason the comparison below is self-calibrating rather than read directly.

Per stratum, on the effect scale:

| Stratum | Family groups | Variants | Effect sd | Per-channel discordance | Shared-to-discordance ratio |
| --- | ---: | ---: | ---: | --- | --- |
| Close | 66 | 1,402 | 0.6298 | 0.3532 [0.3050, 0.3992] | 1.745 [1.452, 2.084] |
| Remote | 100 | 2,078 | 0.5737 | 0.3757 [0.2985, 0.4764] | 1.387 [1.039, 1.835] |
| Mixed | 13 | 2,811 | 0.6998 | 0.3454 [0.2591, 0.4429] | 2.022 [1.505, 2.889] |

The floor is not materially different between close and remote — the two discordance intervals overlap heavily — but the remote stratum's shared component is smaller, so its ratio is lower and its lower bound touches 1.04. **Less is resolvable on the remote stratum than on the close one before any model is read**, which is exactly the confound the positive control exists to separate from homology dependence.

The source's own reported uncertainty on the drawn variants has median 0.1104, third quartile 0.1776 and maximum 0.4993 kcal/mol of 95% interval width, by construction of the admission rule.

## The declared support

Fixed before any predictor is fitted and before any model is scored, by four rules that read names, sequences and interval widths and never a measured effect.

A name's background is its longest proper underscore-prefix that is itself a name in the table. The token vocabulary of these libraries is not fixed — the staged table carries 31,412 distinct tokens, scramble controls beside substitutions, insertions and deletions — so a declared grammar misparses most of the file. The rule is the one the frozen support census used, and the entry point reproduces that census's own count of **6,051** MGnify-flagged backgrounds carrying a variant series and a wild-type row before drawing anything, or it exits.

The background draw is **320 of those 6,051 under seed 20260924**, sampled from the sorted name universe. It reads names only. It is also the draw whose wild types the frozen endpoint qualification already searched against UniRef50 and banded, so the identity bands reported here are the canonical ones rather than a second search: on those 320 the bands are 40 / 146 / 100 / 34 across <30% / 30–70% / 70–95% / ≥95% identity over the query, all 40 in the lowest band carrying no reported hit at all, 186 of 320 below 70% (Wilson 95% [52.7%, 63.4%]), and 192 family groups of which 105 are entirely remote.

A retained background must carry at least **8** admitted single substitutions, so that it contributes a site-resolved panel rather than a single row. **No variant cap is declared**: the largest admitted count on this support is 26, so a cap would never bind and is not invented.

Family grouping is the frozen contract's own alignment edge rule — exact Smith-Waterman with BLOSUM62 and NCBI default affine gaps, an edge at 30% identity and 80% coverage of both sequences. Alignment and exact sequence identity are the only union sources available: these names carry no UniProt accession and no Pfam label, so unlike the Domainome support there is no accession or family field to union on.

| Exclusion, from the 320 drawn backgrounds | Count |
| --- | ---: |
| Backgrounds without an admitted wild-type row | 2 |
| Non-substitution variant rows (insertion, deletion, scramble) | 15,518 |
| Substitution rows outside the admission rule | 1,557 |
| Substitution rows disagreeing with their own sequence | 1 |
| Admitted substitution rows | 6,353 |
| Backgrounds below the 8-substitution minimum | 13 |

| Frozen workload | Value |
| --- | --- |
| Backgrounds | 305 |
| Family groups | 179, 162 of them singletons, largest 91 |
| Variants | 6,291 |
| Mutated sites | 6,289 |
| Sites per background | 8–25, median 22 |
| Variants per site | 1–2, median 1 |
| Sequence length range | 60–80 residues |
| Sequences per arm | 6,596 |
| Residues per arm | 457,600 |
| Kish effective family groups | 179.0 |
| Kish effective backgrounds | 190.96 |
| Kish effective sites | 3,557.8 |

The independent unit is the family group, and there are 179 of them with exactly equal weight by construction. Weighting is groups equal, backgrounds equal inside a group, sites equal inside a background, variants equal inside a site. Every count reported for a stratum below is a group count, never a variant count.

Two properties of the support are worth naming because they shape what can be read. The site level is nearly vacuous — 6,291 variants over 6,289 sites, so a site carries one variant almost always — which means the three-level nesting reduces in practice to groups over backgrounds. And every retained background comes from the **dms7** library, so blocking the designs by assay is degenerate here: there is one assay block and nothing to block on.

Cohort digest `64182578cff3d1bbd2bf9e10aae5cc46d0319438a208035fb897667f48c1b3c7`; label-free extraction plan content digest `aa63ff6dc4b66ea51c9a495e1ca173ab22d622cbc4fe48bc8f92b6b012d05e48`; support declaration digest `b965b8ab38322008236b731d37f240a40de8964b5926607ecb0f48afec335ee5`; panel row-identity digest `1914da823cebfad608f637d1b71aac1f3685a3dbfac6576bc4a122d56d6932e1`.

### The identity strata, and what a group's stratum means

A group is **close** only if every member retrieves a relative at 70% identity or above, **remote** only if every member falls below it, and **mixed** otherwise. A mixed group enters neither stratum: the group is the unit a held-group readout uses, so a group that is partly close cannot certify either side.

| Stratum | Groups | Backgrounds | Variants | Kish effective groups | Bands present |
| --- | ---: | ---: | ---: | ---: | --- |
| Close | 66 | 69 | 1,402 | 66.0 | 945 variants at 70–95%, 457 at ≥95% |
| Remote | 100 | 103 | 2,078 | 100.0 | 1,493 variants at 30–70%, 585 with no reported hit |
| Mixed | 13 | 133 | 2,811 | 13.0 | all four bands |

Both close and remote clear the eight-unit floor. The mixed stratum is 13 groups but 133 backgrounds, because the 91-member group is mixed; it holds 45% of the variants on 7% of the units, which is why it is reported and read as neither.

Over the retained 305 backgrounds the bands are 35 / 145 / 94 / 31, and 35 carry no reported hit. Those differ from the canonical 40 / 146 / 100 / 34 over the sampled 320, and the difference is the admission rule and the 8-substitution minimum removing 15 backgrounds — not a second search. The 100 entirely-remote groups here likewise differ from the canonical 105 over all 320 for the same reason.

**What a band is and is not.** A band is the identity of the best hit in one UniRef50 snapshot under `diamond blastp --very-sensitive --masking 0 --evalue 1e-3`. No detected alignment is not absence from any model's pretraining corpus, and a remote group is a conservative held-group control rather than a certified remote-family holdout.

## The comparison is self-calibrating, and a remote result never travels without its outcome

With a floor this high a remote null is uninterpretable on its own: it is indistinguishable from an endpoint that cannot resolve the quantity anywhere. The design therefore reads the **same fit** out twice. The folds, the training groups and the held-out predictions are identical; only the rows the paired per-group error is computed over change, and the nested weighting is renormalised inside the retained rows. The close stratum is the positive control — the groups where a model quantity that depends on retrieved homology should resolve if the endpoint permits resolving anything at all.

The pair of readouts selects one of three outcomes, and one of them is always named:

- **resolves on close and on remote** — the quantity survives remote homology;
- **resolves on close only** — homology dependence;
- **resolves on neither** — the endpoint floor is too high on this support and the remote case is **unresolved, not null**.

An arm resolves on a stratum when all three per-seed intervals of its paired increment over that stratum exclude zero above it. A result inside the floor is not a negative. A remote result with a close stratum that did not fire is reported as survival with the control's own failure stated beside it, never silently promoted.

## Control and baseline qualification

A control may enter this comparison only if it is a competent held-out predictor in its own right. The rule was declared before any fit: a candidate is offered, in the declared order, to the standing set that has already qualified, and it is kept only if its paired reduction in group-equal held-out mean squared error over that standing set is positive at every one of the three prespecified split seeds 20260923, 20260924 and 20260925.

The base is the directed substitution identity (400 coordinates) and the mutated position's geometry (6). The candidates are composition (40), local chemistry in mutation-centred windows (52), the mutation-local profile (10) and its bounded restatement (10), and a nonlinear-additive global response; every block is imported from the folding-and-stability gate rather than reimplemented. Profiles come from one DIAMOND search of both nested gates' wild types against UniRef50 at `--very-sensitive --masking 0`, and **264 of the 305 backgrounds carry a profile while 41 carry the declared absence** — a zero availability indicator and no imputed profile, which is what a cohort selected for remoteness produces.

Increments are paired reductions in group-equal held-out mean squared error in kcal²/mol² over 179 family groups, with 2,000-draw group-bootstrap intervals; the rank column is the paired within-background Spearman increment over the same standing set.

| Candidate | Offered over | Increment, 20260923 / 20260924 / 20260925 | Own contribution at 20260923 | Rank increment | Verdict |
| --- | --- | --- | --- | --- | --- |
| Composition | base | −0.00286 / −0.00367 / −0.00158 | −0.00286 [−0.01098, +0.00509] | +0.0104 / +0.0126 / +0.0072 | discarded |
| Local chemistry | base | +0.00472 / +0.00377 / +0.00486 | **+0.00472 [+0.00064, +0.00853]** | +0.0778 / +0.0703 / +0.0669 | **kept** |
| Mutation-local profile | base + chemistry | +0.00268 / +0.00564 / +0.00535 | +0.00268 [−0.00100, +0.00618] | +0.0018 / +0.0053 / +0.0050 | **kept** |
| Bounded profile restatement | base + chemistry + profile | +0.00440 / +0.00183 / +0.00110 | **+0.00440 [+0.00002, +0.00908]** | +0.0081 / +0.0048 / −0.0029 | **kept** |
| Nonlinear-additive response | base + chemistry + both profiles | −0.02845 / −0.03021 / −0.04053 | −0.02845 [−0.03894, −0.01750] | −0.0508 / −0.0420 / −0.0443 | discarded |

The **qualified control set is the directed substitution identity, the mutated-position geometry, the local chemistry block and both profile blocks**, 478 coordinates. On that set the group-equal held-out mean squared error is 0.35069, 0.35179 and 0.35328 kcal²/mol² against a no-effect null of 0.55505, so the qualified controls remove **36.4% to 36.8%** of the null's squared error. Read as its own contribution rather than as a ladder step, the qualified set reduces the null's squared error by **0.20436 [0.16631, 0.24098] kcal²/mol²** at seed 20260923 over 179 resampled groups, and its within-background Spearman is **+0.3484 [+0.30963, +0.38573]**. The controls these increments will be read over are a competent held-out predictor of this endpoint by a wide margin, not a formality.

**Each candidate's own contribution carries an interval, and one kept block's contains zero.** The mutation-local profile is kept by the declared three-seed sign rule at +0.00268 [−0.00100, +0.00618] kcal²/mol², which is weaker evidence of competence than local chemistry's [+0.00064, +0.00853] or the bounded restatement's [+0.00002, +0.00908]. That distinction is carried rather than smoothed over: a control whose own contribution is indistinguishable from zero makes any increment measured over it uninformative, which is the failure the catalogue records as L48 with four instances. What licenses the comparisons is the qualified set as a whole, whose contribution is resolved far from zero.

Two candidates are discarded and the second is the more interesting. Composition fails for the reason it fails on every within-background difference endpoint: the mutant-minus-wild-type composition change is already determined by the substitution identity, and what remains is a background-level constant with no rank information inside a held-out background. The **nonlinear-additive global response is resolved below zero at −0.02845 [−0.03894, −0.01750] kcal²/mol², an order of magnitude larger than anything else on the ladder** — and on the folding-and-stability gate the same block was the strongest control it qualified, at +0.02529 [+0.01639, +0.03460]. The difference is the endpoint: that gate fits its first stage on absolute stability levels and applies the isotonic response to both states of a pair, while here the response is applied to the predicted effect itself, and on a support whose sites carry one variant each the monotone transform of a linear prediction generalises worse across held-out groups than the linear prediction did.

The arm's own tokenisation descriptors are offered to each arm's matched baseline under the same rule, because a representation difference can be nonzero purely through segmentation. An arm whose descriptors lower its own baseline is reported as such and its increments are read over the control set without them.

Because the mean-squared-error rule and the correlation rule agree on every block here — composition's rank increment is positive at all three seeds, so it enters the secondary set, and no other discarded candidate does — the secondary control set adds composition alone. Squared-error increments over the primary set and rank increments over the secondary set are the licensed cells; the two crossed cells are reported as sensitivities.

## The likelihood-level result, over all 33 arms

Every arm of the frozen roster was extracted and fitted: **33 of 33, with 0 failures and 0 no-record cells**. Run-to-run reproducibility of the identical single-row computation is **6.10 x 10^-5 nats** on the likelihood and **8.68 x 10^-7** relative L2 on the feature block at the worst of 33 arms and 305 backgrounds; at batch size one that is what the repeat column measures and the 0.001 gates remain structurally vacuous. Permuting the 818 held-out target rows of the first fold moves the held-out prediction of every one of the 12 declared designs by **0.0**, so no held-out label reaches feature scaling, penalty selection or the first-stage response.

**No arm's tokenisation descriptors qualified: 0 of 33.** Every arm's segmentation block lowers the held-out squared error of the set it augments at at least one seed, so every arm is read over the control set itself. The consequence is stronger than a matched baseline: the baseline carries no arm-specific column at all, so its group-equal held-out mean squared error is literally one number per seed -- **0.35069, 0.35179 and 0.35328 kcal2/mol2** over the primary set and 0.35212, 0.35354 and 0.35485 over the secondary set -- with a between-arm spread of **exactly 0.0** at every seed and both sets. The 33 arms are compared on one support, one partition and one baseline.

### The pooled likelihood increment

Paired reduction in group-equal held-out mean squared error in kcal2/mol2, model likelihood added to the primary control set, with 2,000-draw group-bootstrap intervals over the 179 family groups at each of the three prespecified split seeds. The last two columns are the same contrast read out on the close and on the remote groups alone, on identical held-out rows.

| Arm | Interface | Seed 20260923 | Seed 20260924 | Seed 20260925 | Verdict | Close | Remote |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bygpt5-base-en | byte | +0.00001 [-0.00004, +0.00007] | -0.00005 [-0.00015, +0.00004] | -0.00014 [-0.00030, -0.00000] | **unresolved** | unresolved | unresolved |
| bygpt5-medium-en | byte | -0.00010 [-0.00039, +0.00019] | -0.00017 [-0.00051, +0.00018] | +0.00006 [-0.00025, +0.00039] | **unresolved** | unresolved | unresolved |
| bygpt5-small-en | byte | -0.00002 [-0.00016, +0.00012] | +0.00009 [-0.00013, +0.00033] | -0.00015 [-0.00046, +0.00009] | **unresolved** | unresolved | unresolved |
| dialogpt-small | BPE | +0.00003 [-0.00017, +0.00022] | -0.00013 [-0.00040, +0.00015] | +0.00008 [-0.00020, +0.00037] | **unresolved** | unresolved | unresolved |
| galactica-1.3b | BPE | +0.00003 [-0.00010, +0.00016] | -0.00018 [-0.00037, +0.00002] | +0.00001 [-0.00020, +0.00024] | **unresolved** | unresolved | unresolved |
| galactica-125m | BPE | +0.00004 [-0.00014, +0.00021] | +0.00009 [-0.00003, +0.00023] | -0.00001 [-0.00031, +0.00025] | **unresolved** | unresolved | unresolved |
| galactica-30b | BPE | +0.00000 [-0.00033, +0.00030] | -0.00054 [-0.00170, +0.00029] | -0.00040 [-0.00130, +0.00028] | **unresolved** | unresolved | unresolved |
| galactica-6.7b | BPE | -0.00014 [-0.00046, +0.00011] | -0.00059 [-0.00159, +0.00014] | -0.00023 [-0.00100, +0.00030] | **unresolved** | unresolved | unresolved |
| gpt2 | BPE | +0.00008 [-0.00004, +0.00023] | +0.00014 [-0.00012, +0.00042] | +0.00007 [-0.00016, +0.00031] | **unresolved** | unresolved | unresolved |
| gpt2-large | BPE | -0.00000 [-0.00007, +0.00006] | -0.00032 [-0.00055, -0.00012] | -0.00013 [-0.00027, +0.00000] | **unresolved** | unresolved | unresolved |
| gpt2-medium | BPE | +0.00001 [-0.00006, +0.00008] | -0.00009 [-0.00035, +0.00011] | -0.00016 [-0.00043, +0.00007] | **unresolved** | unresolved | unresolved |
| gpt2-xl | BPE | +0.00006 [-0.00008, +0.00020] | +0.00007 [-0.00031, +0.00034] | +0.00008 [-0.00019, +0.00031] | **unresolved** | unresolved | unresolved |
| instructprotein | BPE | +0.00008 [-0.00015, +0.00032] | -0.00121 [-0.00311, -0.00003] | -0.00013 [-0.00054, +0.00025] | **unresolved** | unresolved | unresolved |
| llama-2-7b | BPE | -0.00008 [-0.00019, +0.00002] | -0.00011 [-0.00027, +0.00002] | -0.00012 [-0.00031, +0.00004] | **unresolved** | unresolved | unresolved |
| llama-3.2-3b | BPE | -0.00022 [-0.00038, -0.00008] | -0.00029 [-0.00054, -0.00009] | +0.00000 [-0.00006, +0.00006] | **unresolved** | unresolved | unresolved |
| progen2-base | residue | +0.00019 [-0.00021, +0.00056] | -0.00002 [-0.00118, +0.00079] | -0.00079 [-0.00238, +0.00037] | **unresolved** | unresolved | unresolved |
| progen2-large | residue | +0.00046 [+0.00002, +0.00084] | +0.00047 [-0.00092, +0.00144] | -0.00016 [-0.00186, +0.00104] | **unresolved** | unresolved | above zero |
| progen2-medium | residue | +0.00018 [-0.00020, +0.00054] | -0.00014 [-0.00130, +0.00071] | -0.00075 [-0.00234, +0.00037] | **unresolved** | unresolved | unresolved |
| progen2-small | residue | +0.00037 [-0.00003, +0.00079] | -0.00001 [-0.00086, +0.00082] | -0.00022 [-0.00106, +0.00057] | **unresolved** | unresolved | unresolved |
| progen2-xlarge | residue | +0.00058 [-0.00036, +0.00135] | +0.00082 [-0.00045, +0.00196] | +0.00051 [-0.00097, +0.00185] | **unresolved** | unresolved | above zero |
| progen3-112m | residue | +0.00005 [-0.00017, +0.00027] | -0.00001 [-0.00034, +0.00031] | -0.00020 [-0.00064, +0.00020] | **unresolved** | unresolved | unresolved |
| progen3-3b | residue | +0.00012 [-0.00055, +0.00067] | +0.00016 [-0.00080, +0.00094] | -0.00005 [-0.00118, +0.00083] | **unresolved** | unresolved | unresolved |
| prollama | residue | +0.00000 [-0.00005, +0.00006] | -0.00035 [-0.00068, -0.00005] | -0.00034 [-0.00073, -0.00001] | **unresolved** | unresolved | unresolved |
| prollama-stage-1 | residue | -0.00002 [-0.00011, +0.00007] | -0.00047 [-0.00074, -0.00022] | -0.00023 [-0.00043, -0.00004] | **unresolved** | below zero | unresolved |
| proteinglm-7b-clm | residue | +0.00003 [-0.00087, +0.00069] | +0.00031 [-0.00069, +0.00114] | +0.00012 [-0.00088, +0.00094] | **unresolved** | unresolved | unresolved |
| protgpt2 | BPE | +0.00015 [-0.00043, +0.00068] | +0.00040 [-0.00030, +0.00104] | +0.00001 [-0.00091, +0.00082] | **unresolved** | unresolved | unresolved |
| protgpt3-1.3b | BPE | +0.00049 [-0.00001, +0.00104] | +0.00067 [-0.00005, +0.00136] | +0.00040 [-0.00039, +0.00111] | **unresolved** | unresolved | unresolved |
| qwen2.5-0.5b | BPE | +0.00002 [-0.00024, +0.00037] | +0.00003 [-0.00019, +0.00025] | -0.00015 [-0.00045, +0.00014] | **unresolved** | unresolved | unresolved |
| qwen2.5-0.5b-instruct | BPE | +0.00002 [-0.00010, +0.00018] | -0.00002 [-0.00017, +0.00014] | -0.00013 [-0.00037, +0.00010] | **unresolved** | unresolved | unresolved |
| qwen2.5-32b | BPE | -0.00007 [-0.00013, -0.00001] | -0.00025 [-0.00046, -0.00010] | -0.00008 [-0.00019, +0.00002] | **unresolved** | unresolved | unresolved |
| qwen2.5-7b | BPE | -0.00008 [-0.00027, +0.00007] | -0.00021 [-0.00044, +0.00001] | -0.00004 [-0.00015, +0.00006] | **unresolved** | unresolved | unresolved |
| qwen3-8b-base | BPE | +0.00013 [-0.00008, +0.00036] | +0.00012 [-0.00026, +0.00048] | +0.00017 [-0.00018, +0.00052] | **unresolved** | unresolved | unresolved |
| rita-xl | residue | +0.00020 [-0.00007, +0.00044] | -0.00018 [-0.00116, +0.00051] | -0.00051 [-0.00175, +0.00032] | **unresolved** | unresolved | unresolved |

Panel median of the three-seed means **-0.000053 kcal2/mol2**. By interface: residue 11 arms at -0.000053 (5 positive, 0 above zero, 0 below), BPE 19 arms at -0.000047 (6 positive, 0 above zero, 0 below), byte 3 arms at -0.000058 (0 positive, 0 above zero, 0 below).

**Not one of the 33 arms resolves the pooled likelihood increment, in either direction.** Against a control set that already removes 36.4% to 36.8% of the no-effect null's squared error, every interval at every seed spans zero or fails to do so at only one or two of the three, and the interface medians are indistinguishable from each other and from zero: residue -0.000053, BPE -0.000047, byte -0.000058. The residue-against-text interface pattern the folding-and-stability gate found -- +0.00733 residue against +0.0000016 BPE and -0.00009 byte -- is absent here at the level of the pooled squared-error cell.

### The likelihood rank increment, the other licensed cell

Paired within-background Spearman increment over the secondary control set, which is the set the correlation rule qualified. Points only; the intervals are in the artefact.

| Arm | Interface | Seed 20260923 | Seed 20260924 | Seed 20260925 | Verdict |
| --- | --- | --- | --- | --- | --- |
| bygpt5-base-en | byte | -0.00073 | -0.00012 | +0.00084 | **unresolved** |
| bygpt5-medium-en | byte | +0.00146 | -0.00377 | +0.00057 | **unresolved** |
| bygpt5-small-en | byte | -0.00031 | -0.00068 | +0.00004 | **unresolved** |
| dialogpt-small | BPE | +0.00058 | -0.00131 | +0.00009 | **unresolved** |
| galactica-1.3b | BPE | -0.00198 | -0.00079 | -0.00081 | **unresolved** |
| galactica-125m | BPE | -0.00032 | +0.00083 | -0.00060 | **unresolved** |
| galactica-30b | BPE | +0.00038 | +0.00089 | -0.00053 | **unresolved** |
| galactica-6.7b | BPE | -0.00035 | +0.00234 | +0.00996 | **unresolved** |
| gpt2 | BPE | -0.00021 | +0.00192 | -0.00004 | **unresolved** |
| gpt2-large | BPE | -0.00083 | -0.00051 | +0.00041 | **unresolved** |
| gpt2-medium | BPE | -0.00018 | +0.00069 | +0.00150 | **unresolved** |
| gpt2-xl | BPE | +0.00155 | +0.00095 | +0.00051 | **unresolved** |
| instructprotein | BPE | -0.00048 | -0.00955 | -0.00207 | **unresolved** |
| llama-2-7b | BPE | -0.00159 | -0.00124 | -0.00097 | **unresolved** |
| llama-3.2-3b | BPE | -0.00113 | +0.00054 | -0.00039 | **unresolved** |
| progen2-base | residue | +0.00601 | +0.00317 | +0.00130 | **unresolved** |
| progen2-large | residue | +0.00342 | +0.00449 | +0.00114 | **unresolved** |
| progen2-medium | residue | +0.00425 | +0.00367 | +0.00047 | **unresolved** |
| progen2-small | residue | +0.00326 | +0.00238 | +0.00014 | **unresolved** |
| progen2-xlarge | residue | +0.00293 | +0.00388 | +0.00583 | **unresolved** |
| progen3-112m | residue | +0.00198 | +0.00190 | +0.00088 | **unresolved** |
| progen3-3b | residue | +0.00508 | +0.00514 | +0.00302 | **unresolved** |
| prollama | residue | +0.00091 | +0.00016 | +0.00027 | **unresolved** |
| prollama-stage-1 | residue | +0.00007 | +0.00049 | -0.00085 | **unresolved** |
| proteinglm-7b-clm | residue | -0.00031 | +0.00458 | +0.00445 | **unresolved** |
| protgpt2 | BPE | +0.00226 | +0.00484 | +0.00450 | **unresolved** |
| protgpt3-1.3b | BPE | +0.00314 | +0.00497 | +0.00338 | **unresolved** |
| qwen2.5-0.5b | BPE | -0.00056 | -0.00112 | -0.00099 | **unresolved** |
| qwen2.5-0.5b-instruct | BPE | -0.00111 | -0.00130 | -0.00077 | **unresolved** |
| qwen2.5-32b | BPE | -0.00019 | -0.00067 | -0.00041 | **unresolved** |
| qwen2.5-7b | BPE | +0.00241 | +0.00003 | -0.00058 | **unresolved** |
| qwen3-8b-base | BPE | -0.00102 | +0.00049 | +0.00466 | **unresolved** |
| rita-xl | residue | +0.00206 | +0.00091 | -0.00147 | **unresolved** |

Panel median **+0.000499**. By interface: residue 11 arms at +0.002797 (10 positive, 0 above zero, 0 below), BPE 19 arms at -0.000029 (9 positive, 0 above zero, 0 below), byte 3 arms at -0.000317 (0 positive, 0 above zero, 0 below).

**Here the interface ordering of the development gate survives while no arm resolves.** The residue stratum's median is +0.002797 with 10 of 11 arms positive, against -0.000029 for the 19 BPE arms and -0.000317 for the 3 byte-level arms -- the same ordering, about half the magnitude, and 0 of 33 arms with all three intervals above zero. On the unlicensed crossed cell, the rank increment over the primary set, one residue arm resolves above zero and none below.

### Which of the three outcomes the remote case falls in

**The close-identity positive control did not fire: 0 of 33 arms resolve on the 66 close groups, on either control set.** Two arms resolve on the 100 remote groups alone -- ProGen2-large at +0.00064 [+0.00022, +0.00104], +0.00092 [+0.00029, +0.00156] and +0.00100 [+0.00030, +0.00178] kcal2/mol2, and ProGen2-xlarge at +0.00098 [+0.00037, +0.00162], +0.00104 [+0.00026, +0.00183] and +0.00140 [+0.00040, +0.00251] -- and one arm, ProLLaMA Stage 1, resolves below zero on the close groups alone.

A remote-only resolution is **not** one of the three declared outcomes and is not promoted to one. The close stratum gates the reading: with a shared-to-discordance ratio of 1.566 [1.301, 1.847], a stratum on which the pipeline resolves nothing at all cannot tell an absent quantity from an unmeasurable one. So all **33 of 33 arms fall in the third outcome on both control sets: resolves on neither stratum -- the endpoint floor is too high on this support and the remote case is unresolved, not null.** The two remote resolutions are recorded with the flag that says their own positive control did not fire, at 0.18% to 0.40% of the baseline's remaining squared error, and they license nothing about remote homology.

**What that does and does not settle.** It does not show that a frozen model quantity carries no information about substitution effects on metagenome-assembled domains. It shows that this support cannot answer the question: the endpoint's own effect-scale floor leaves a reproducible component only 1.57 times the per-channel discordance, and on that floor the pipeline resolves nothing even where the corpus retrieves a near-duplicate. The remote-homology end of the retrieval axis is now populated -- 100 entirely remote family groups against the existing cohorts' 1, 2 and 12 -- and the binding limitation has moved from the absence of remote units to the resolution of the endpoint measured on them.

### The representation increment -- provisional, and subject to recomputation

**This table is provisional.** It is conditional on one compressed linear readout of four pooled block summaries at the retained depths, and the readout-class reassessment is still running. No verdict about representational content is drawn from it here, and the full-width states are retained so the revisit is a refit rather than a re-extraction.

Squared-error increment over the primary set, points only, provisional:

| Arm | Interface | Seed 20260923 | Seed 20260924 | Seed 20260925 |
| --- | --- | --- | --- | --- |
| bygpt5-base-en | byte | -0.00338 | -0.00128 | -0.00048 |
| bygpt5-medium-en | byte | -0.00393 | -0.00149 | -0.00106 |
| bygpt5-small-en | byte | -0.00215 | -0.00053 | +0.00009 |
| dialogpt-small | BPE | -0.00341 | -0.00128 | -0.00110 |
| galactica-1.3b | BPE | +0.00493 | +0.00583 | +0.00691 |
| galactica-125m | BPE | +0.00153 | +0.00410 | +0.00420 |
| galactica-30b | BPE | +0.00752 | +0.00977 | +0.01011 |
| galactica-6.7b | BPE | +0.00882 | +0.01054 | +0.01133 |
| gpt2 | BPE | -0.00346 | -0.00153 | -0.00019 |
| gpt2-large | BPE | -0.00330 | -0.00204 | -0.00142 |
| gpt2-medium | BPE | -0.00400 | -0.00136 | -0.00099 |
| gpt2-xl | BPE | -0.00393 | -0.00214 | -0.00127 |
| instructprotein | BPE | +0.00557 | +0.00607 | +0.00710 |
| llama-2-7b | BPE | -0.00323 | -0.00181 | -0.00131 |
| llama-3.2-3b | BPE | -0.00500 | -0.00278 | -0.00293 |
| progen2-base | residue | +0.00300 | +0.00469 | +0.00125 |
| progen2-large | residue | +0.01108 | +0.01383 | +0.01308 |
| progen2-medium | residue | +0.00887 | +0.00898 | +0.00655 |
| progen2-small | residue | +0.01203 | +0.01343 | +0.01397 |
| progen2-xlarge | residue | +0.00290 | +0.00166 | +0.00154 |
| progen3-112m | residue | +0.01301 | +0.01545 | +0.01498 |
| progen3-3b | residue | +0.00711 | +0.00722 | +0.00867 |
| prollama | residue | +0.00936 | +0.01202 | +0.01153 |
| prollama-stage-1 | residue | +0.01096 | +0.01306 | +0.01291 |
| proteinglm-7b-clm | residue | +0.00303 | +0.00574 | +0.00588 |
| protgpt2 | BPE | +0.01225 | +0.01460 | +0.01398 |
| protgpt3-1.3b | BPE | +0.00526 | +0.00437 | +0.00706 |
| qwen2.5-0.5b | BPE | -0.00495 | -0.00342 | -0.00424 |
| qwen2.5-0.5b-instruct | BPE | -0.00505 | -0.00347 | -0.00343 |
| qwen2.5-32b | BPE | -0.00123 | +0.00040 | +0.00019 |
| qwen2.5-7b | BPE | -0.00267 | -0.00204 | -0.00106 |
| qwen3-8b-base | BPE | -0.00240 | -0.00219 | -0.00241 |
| rita-xl | residue | +0.00846 | +0.00924 | +0.00924 |

Panel median +0.002976 kcal2/mol2. By interface: residue 11 arms at +0.008980 (11 positive, 0 above zero, 0 below), BPE 19 arms at -0.001922 (7 positive, 0 above zero, 0 below), byte 3 arms at -0.001711 (0 positive, 0 above zero, 0 below).

## Artifacts, and the measurement identity they were produced under

Each arm contributes two quantities, added separately and never only jointly: the likelihood difference M_mut - M_WT in nats, one column, and the projected representation difference R_mut - R_WT over four pooled blocks, 1,024 columns at 256 coordinates per block.

The full frozen 33-arm roster was dispatched and **no arm was selected by any outcome**; the campaign builder exits rather than write a manifest set that does not cover the roster exactly, and lane order is a declared descending parameter scale. Measurement identity is the singleton one the panel's interface forces: batch size one for every arm, float32 for 31 arms and bfloat16 for the two ProGen3 arms, whose mixture-of-experts block has no kernel above float16 and whose expert mixture reduces over the flattened batch. At batch size one the declared 0.001 precision gates are **structurally vacuous rather than passed**. Every cell named the staged `ct-20260905` interpreter -- Python 3.11.14, numpy 1.26.4, torch 2.9.1+cu128, transformers 4.57.3 -- rather than the pod image's 3.12.7, and pinned the BLAS thread count to 4 in its own manifest environment column, because a float32 matmul's reduction order depends on it and the admitted projection ran at that value. Every cell carried `--keep-full-features`, so the unprojected per-state block outputs are retained and the readout-class recomputation is a refit rather than a second forward pass.

Extraction run `20260924081450_45cbf26d2945`, four lane groups over ten cards, manifests `scripts/transfer/campaign_nested_rh_extract_w1{a,b,c,d}.tsv`, 33 of 33 cells `exited-ok`. Frozen inputs are in `results/remote_homology_20260924/`: `endpoint_qualification.json`, `endpoint_records.json`, `cohort.json`, `extraction_plan.json`, `support_declaration.json`, `controls_qualification.json` and `profile_features.npz` (sha256 `4c36531d...95e16ee6b`). The 33 per-arm fit records are `fits/fit_<arm>.json`, each carrying its complete fold map, the selected ridge penalty of every design in every fold, the declared design names with their column dimensions and the first-stage diagnostics of the nonlinear response; the panel is `panel/panel.json` with a digest of every fit it reads. Runtime logs stay under ignored `logs/d1_remote_homology_20260924/`.

**What a recomputation needs.** The projection is a declared Gaussian matrix determined by width and seed alone, and the unprojected per-state block outputs are retained on the cluster beside each arm's projected archive, 7.67 GiB over 10,065 files against 0.83 GiB projected. A different projection or a different readout head therefore needs a refit and not a forward pass. The two depths hooked are the extraction's own `[(L-1)//2, L-1]`, so a depth selection outside those two would need re-extraction, which is the limitation the catalogue records as L53 and which applies to this gate exactly as it applies to the folding-and-stability one.

## Limitations

**This endpoint is not independent of the development stability axis.** It shares the cDNA-display proteolysis assay technology and an author with Tsuboyama 2023, so agreement with the folding-and-stability gate is agreement within one assay principle and is not replication. It supplies remoteness, not independence.

**The floor is the binding limitation and it is not a caveat on the numbers.** The reproducible component exceeds the per-channel discordance by 1.566 [1.301, 1.847] against 3.86 [3.61, 4.13] on the admitted MegaScale endpoint, and on the remote stratum alone the ratio's lower bound touches 1.04. A remote increment inside that floor is unresolved and is not a null, and it is reported as one of the three declared outcomes or not at all.

**The strata are not equal in what they can resolve.** The remote stratum's shared component is smaller than the close stratum's (0.5212 [0.4605, 0.5805] against 0.6162 [0.5426, 0.6873] kcal/mol), so an outcome of "close only" is consistent with homology dependence *and* with a stratum difference in resolvable signal. Distinguishing those two is beyond what this support does.

**A band is a property of one search against one snapshot.** No detected alignment is not absence from a pretraining corpus, and the identity bands come from the frozen search rather than from a search over any arm's actual training data.

**Mixed groups hold most of the material.** 13 of 179 groups carry 2,811 of 6,291 variants, because the largest group spans the boundary. They enter neither stratum, so 45% of the panel's rows inform the pooled readout and neither stratified one.

**The support is narrow in two ways that bound generalization.** Every retained background is 60 to 80 residues from the dms7 library, so there is one assay block and no blocking to do, and nothing here describes a larger domain or another library. And the admission rule keeps only variants whose three intervals are all under 0.5 kcal/mol wide, so variants outside the assay's resolvable window are absent by construction.

**Multiplicity is declared and no interval is adjusted for it**, across 33 arms, three split seeds, two control sets, two model quantities, two metrics and three identity strata; the licensed cells alone are 33 arms times two model quantities times two metrics. Requiring all three per-seed intervals to exclude zero is a conjunction over three correlated resamplings of one support, not three independent tests, so it carries no nominal level; it is a stability requirement and not a corrected test.
