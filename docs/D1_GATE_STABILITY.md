# Does frozen model output carry information about how a single substitution changes folding stability?

This is the folding-and-stability gate of the [Direction-1 capability map](D1_CAPABILITY_MAP_PLAN.md), qualified on its own support with its own instrument, its own controls and its own admission condition. A negative result at any other gate does not determine this one, and nothing here determines any other gate.

The gate asks one question: does a frozen model quantity place resolvable information about the measured stability change of a single substitution, beyond controls that are competent held-out predictors in their own right? Composition, local chemistry, mutation-local evolutionary statistics and a nonlinear global response are competing explanations here, never mechanisms.

Two parts of the answer stand differently, and the difference is declared rather than discovered. The endpoint, the support, the control verdicts and the likelihood increments do not depend on the readout class and are stated as final. The representation increments are reported with their intervals and are **provisional**: they are conditional on a compressed linear readout of four pooled block summaries at the retained depths, and the readout-class reassessment the capability map records is still running. No verdict sentence about what the representations do or do not contain is written here, and the full-width states are retained so the revisit is a refit rather than a re-extraction.

## Why this gate is separate from the pairwise experiment

The [pairwise experiment](D1_PAIRWISE_EPISTASIS_RESULTS.md) measured single-mutant likelihood differences as a *control* inside an interaction experiment and found them to be its strongest reproducible model-side signal: five arms with all three per-seed intervals above zero, every one residue-level protein, as large or larger at sequence separation of at least 10 residues. That quantity was never qualified as a gate. It rode on 8,192 four-state cycles capped at 128 per background and confined to 217 site pairs the source study had selected for suspected coupling, and it inherited an admission decision taken for a four-estimate endpoint.

This gate replaces both. The endpoint is a two-estimate difference, so it has its own between-protease agreement, its own noise floor and its own support accounting, and none of the cycle endpoint's admission transfers to it. The support is unselected with respect to site: every position of every retained background is eligible, and the cohort is drawn from the whole admitted single-mutant panel rather than from the study's coupling tables.

## The endpoint

**ddG = y_mut − y_WT** on the combined MegaScale `dG_ML` scale in kcal/mol. `dG_ML` is an unfolding free energy, so a larger value is more stable and a destabilising substitution gives a negative ddG.

The sign convention and the arithmetic are read off the source's own `ddG_ML` column rather than assumed. On 368,567 accepted rows our construction matches the reported value with a root-mean-square residual of 0.0419 kcal/mol, against 2.2216 kcal/mol for the opposite sign, and the mean reported `ddG_ML` is −0.6577 kcal/mol, so single substitutions reduce stability on average. **This is a self-consistency check, not a validation.** `ddG_ML` is derived from the same `dG_ML` values we difference, so the agreement establishes that we compute what the source computes and says nothing about the accuracy of the measurement. The relevant accuracy statement is the noise floor below, which is five times larger.

The admission rule is the tightened per-channel one: finite numeric `dG_ML`, each of the combined, trypsin and chymotrypsin 95% confidence widths inside [0, 0.5] kcal/mol, and — declared here as part of the support — substitution constructs only. Aggregation is the median over accepted rows per exact (`WT_name`, `aa_seq`) state, with identical rows feeding both channels.

Two exclusions are part of the endpoint's definition rather than cleanup.

Censored `dG_ML` strings are counted and dropped, never coerced to a numeric endpoint: 53,235 rows read `<-1`, 43,059 read `>5` and 4,537 read `-`.

Insertion and deletion constructs carry an `aa_seq` truncated to the wild type's length in the pinned files, so they are length-matched and would enter a substitution support as though they were substitutions. Every row whose `mut_type` begins `ins` or `del` is excluded: 53,441 of the rows that pass the three width rules. Measured independently on the length-matched states of catalogue backgrounds, 1,486 of 504,298 states are carried by at least one indel row and 1,463 exclusively; 26 are order-1 states and one is a wild type, and a contaminated wild-type state would propagate into every difference taken against it in that background. The order histogram also settles a separate question: **every** state of substitution order three or higher — between 10 and 32 states at each order from 3 to 71 — is indel-carrying, so after the exclusion this dataset contains no substitution material above order two at all. That is the mechanical reason the higher-order gate found no triples here, rather than a bare absence.

Accepted rows number 521,890 of 776,298. Accepted combined values span −0.974 to +5.000 kcal/mol, trypsin −1.573 to +5.961 and chymotrypsin −1.094 to +5.514, with no accepted row at either channel's ±15.000 kcal/mol fit bound.

Endpoint digest over the whole natural panel: `f206d1e579bc192a00f99836e2152cfbbd2ac30dfc5bde54d1da27df4c62b243`. Over the frozen cohort: `d7235820456442818a40352b18bb4fc86e67e9b0769e9a0d1362f9acd9b9b324`.

## Between-protease agreement and the noise floor

Two proteases are two assay channels over the same sequences, the same library and the same stability-inference model, not independent measurements of one ground truth. Agreement is reported as the same decomposition the [label instrument](D1_PAIRWISE_LABEL_INSTRUMENT.md) uses for the cycle endpoint: the shared-component standard deviation is the square root of the between-channel covariance and is an *upper* bound on reproducible signal because shared assay error contributes to it, and the per-channel discordance standard deviation is the standard deviation of the channel difference over the square root of two and is a *lower* bound on per-channel measurement noise because error common to both channels cancels. No oracle ceiling is constructed from these.

Resampling preserves the measurement reuse that defines this endpoint: every variant of a background is differenced against that background's single wild-type measurement, so resampling whole backgrounds or whole family groups keeps that shared measurement's signed contribution to all of its variants intact, while variant-level resampling would break exactly that dependence. Weighting is nested — variants equal inside a mutated site, sites equal inside a background, backgrounds equal inside the resampling unit — and intervals are 2,000-draw group-bootstrap percentile intervals at seed 20260923.

On the natural stratum, 261,572 variants across 290 backgrounds, 101 family groups and 16,549 sites:

| Quantity, kcal/mol unless dimensionless | Family group as unit | Background as unit |
| --- | --- | --- |
| Units, Kish effective units | 101, 17.1 | 290, 272.9 |
| Pearson r | +0.9372 [+0.9292, +0.9447] | +0.9404 [+0.9340, +0.9464] |
| Within-background between-channel Spearman | +0.9125 [+0.9003, +0.9245] | — |
| ddG standard deviation, trypsin / chymotrypsin | 0.9216 / 0.9405 | — |
| Shared-component standard deviation | 0.9013 [0.8643, 0.9382] | — |
| **Per-channel discordance standard deviation** | **0.2337 [0.2209, 0.2458]** | 0.2451 [0.2337, 0.2564] |
| Root-mean-square channel difference | 0.3592 [0.3375, 0.3803] | — |
| Shared-to-discordance ratio | 3.86 [3.61, 4.13] | — |

The noise floor is **0.2337 [0.2209, 0.2458] kcal/mol** against a signal standard deviation of 0.9777 kcal/mol on the combined scale. The shared-to-discordance ratio of 3.86 [3.61, 4.13] compares with 2.78 [2.34, 3.26] for the four-estimate cycle on the same source: subtracting two estimates rather than four leaves materially less measurement noise, which is the power difference this gate was built to use.

The channels are not interchangeable in level. Trypsin ddG exceeds chymotrypsin ddG by +0.1405 [+0.1119, +0.1696] kcal/mol on average, so any squared-error statement in kcal²/mol² depends on which channel or which combination defines the target. The declared target is the combined `dG_ML` scale, whose reliability this decomposition bounds from its two inputs rather than certifying directly.

Replacing the median by the first row in stable source order — the declared single-row sensitivity — lowers agreement from r = +0.9372 to +0.9215 [+0.9067, +0.9345] and raises the discordance floor from 0.2337 to 0.2646 kcal/mol. Median aggregation therefore averages some construct-level noise; repeated rows are codon constructs, not biological replicates, and no inverse-variance weighting or confidence shrinkage is applied anywhere.

**The endpoint is admitted** on this support: agreement is bounded far from zero at both resampling units, the reproducible component exceeds the per-channel discordance by a factor of 3.86, and 101 family groups support held-group folds.

## The declared support

The support is fixed before any predictor is fitted and before any model is scored, by three rules that read no measurement: one natural-labelled background per final family group chosen by alphabetical `WT_name`; at most 256 single mutants per background drawn by a stable SHA-256 of the draw seed 20260925, the background name and the mutant sequence; and substitution constructs only. Family grouping is the frozen contract at `6a0a76a4ad1f083f7506f8fa29cb809b555fcf64ccbd722972de877cf26759c4`, used as delivered.

| Frozen workload | Value |
| --- | --- |
| Backgrounds, one per group | 101 |
| Natural family groups covered, of 101 | 101 |
| Variants | 25,856 |
| Distinct sequences after deduplication | 25,957 |
| Residues over those sequences | 1,501,137 |
| Sequence length range | 36–74 residues |
| Mutated sites | 5,664 |
| Sites per background | 34–72, median 58 |
| Variants per site | 1–14, median 4 |
| Kish effective groups | 101.0 |
| Kish effective sites | 5,448.4 |

The independent unit is the family group, and there are 101 of them with exactly equal weight by construction. The site is the unit inside a group, at a Kish effective 5,448.4 under the analysis weighting. Cohort digest `1e42c3cc1d11d479fcbcce23ffcc37b5799fcf7fc79c65780b1a521fd953e9de`; label-free extraction plan content digest `283b4503ea166a3d61052bf68818c8aef162692ed3eb436a7a9864759a48763d`.

### Exclusion accounting, and what the exclusions select for

Of the 330 natural-labelled catalogue backgrounds, 229 do not reach the cohort: 40 carry no accepted wild-type row, 3 fall below the 256-variant cap, and 186 lose to another background already representing their family group. Because every one of the 101 natural groups is covered, none of these exclusions costs a family; they select which member represents its family.

The 40 backgrounds without an accepted wild-type row are **not** an innocent exclusion, and the difference is large. The share of a background's own substitution rows that carry a censored `dG_ML` has a median of 0.598 (interquartile range 0.489 to 0.809) among those 40, against 0.047 (0.010 to 0.128) among the 101 retained — a thirteenfold difference in the central value. A domain whose wild-type free energy cannot be resolved to within 0.5 kcal/mol is, on this evidence, a domain whose measurements are largely outside the assay's resolvable range. On label-independent descriptors the difference is modest: median length 64 against 59 residues, mean hydropathy −0.311 against −0.424, charged fraction 0.270 against 0.288. The 3 below-cap backgrounds are more extreme still, with a censored-row share of 0.586 to 0.807, a median of 11 retrieved homologs and a hydropathy of +0.353.

The 186 backgrounds that lose to a group representative look like the retained ones on the measurement axis, with a censored-row share of 0.049, so the alphabetical rule does not select on resolvability. It does interact with retrieval depth: the retained 101 have a median of 74 retrieved homologs against 817.5 for those 186, which is a consequence of group size rather than of the rule — the retained set includes every singleton group, whose members are the shallowly retrieved ones, while the 186 all sit in multi-member groups with a median size of 17.

The consequence for generalization is stated once and applies to every number below. **The admitted support represents resolvable stability states.** Variants whose mutant state falls below the assay's stability floor or above its ceiling are absent by construction, the backgrounds whose wild type could not be resolved are absent as a stratum, and the retained backgrounds' own alignments are shallow. Nothing here estimates the effect of a strongly destabilising substitution, and nothing here describes a domain that the assay could not measure.

### The remote stratification, and why it is a property of the cohort design

The remote stratification is a training-set purge: for each outer fold, every training group whose wild type reaches a declared sub-contract local-alignment similarity to any held-out group is removed from that fold's training set. The held-out rows are untouched, so a purged fit is paired with an unpurged one on exactly the same evaluation rows, the same folds and the same seeds.

Two thresholds are declared, both strictly below the frozen contract's own 30% identity and 80% mutual coverage edge rule, and both are measured against a composition-preserving within-sequence shuffle null over the same 5,050 pairs, the same aligner and the same rule:

| Rule | Edges observed | Shuffle null, 5 draws | Groups touched | Mean purged groups where touched |
| --- | ---: | --- | ---: | ---: |
| 20% identity, 80% coverage | 5 | 1, 4, 3, 4, 2 | 9 | 1.11 |
| 20% identity, 50% coverage | 132 | 115, 118, 129, 112, 103 | 88 | 3.00 |

At 25% identity and 80% coverage the counts are 3 observed against a null of 0 to 4, and at 28% and 30% identity with 80% coverage both the observed and the null counts are 0 to 1.

Every observed count lies inside its null's range. **There is no alignment-detectable residual family structure among these 101 wild types to stratify on**, which is a property of the cohort design rather than a gate failure: taking one background per group from a contract that already merged everything at 30% identity and 80% coverage exhausts what pairwise local alignment can see. The purge therefore bounds dependence on nearest-detected training neighbours — a chance-level neighbour set, and at the aggressive threshold a 3% training reduction — and certifies no remote-family disjointness. Homology below these thresholds is undetected, 179 of the 478 catalogue backgrounds carry no Pfam label, and two backgrounds the catalogue labels natural are designed mini-proteins.

This limitation has an identified route rather than being intrinsic. A stability dataset on metagenome-assembled domains largely outside UniProt would populate the identity band nothing in the present panel occupies; acquiring it is another agent's work and is not scoped here. Retrieval-corpus identity band, retrieval depth and near-duplicate grouping are the declared strata of the separate retrieval-and-memorization gate and are not duplicated here.

## Control and baseline qualification

A control may enter this comparison only if it is a competent held-out predictor in its own right. The rule was declared before any fit: a candidate is offered, in the declared order, to the standing set that has already qualified, and it is kept only if its paired reduction in group-equal held-out mean squared error over that standing set is positive at every one of the three prespecified split seeds 20260923, 20260924 and 20260925. A candidate that lowers the standing set's held-out error is discarded and reported, never carried forward.

The base every candidate is qualified over is the directed substitution identity (400 coordinates) and the mutated position's geometry (6 coordinates). It carries no chemistry, no composition and no evolutionary statistic. The candidates, in order:

- **Composition** (40 coordinates): the wild-type and mutant amino-acid composition.
- **Local chemistry** (52 coordinates): for hydropathy, formal charge, side-chain volume and the Chou–Fasman helix and sheet parameters, the wild-type and mutant value at the mutated site and their difference, and at window radii 2 and 4 the window means of both states and their difference; plus the BLOSUM62 score of the substitution and, in each window, the mean BLOSUM62 score of the mutant and of the wild-type residue against the flanking wild-type residues with their difference.
- **Mutation-local profile** (10 coordinates): the independent-site profile's log frequencies and log odds at the mutated site, its column entropy and column weight, their window means, the wild type's effective depth and supported-column fraction, and an availability indicator. A background that retrieved no qualifying homolog carries zeros and a zero indicator, never an imputed profile.
- **Nonlinear-additive global response**: a first-stage ridge predictor of *absolute* stability over declared state descriptors, fitted on the outer training groups only with its penalty selected on the same inner held-group partition as the outer tuning, cross-fitted on that partition, followed by an isotonic response fitted on those cross-fitted predictions and applied to both states of every variant. The block is the calibrated difference, the uncalibrated predicted difference and the two predicted state levels. This lets a global monotone response on absolute stability account for part of the endpoint without any mutation-specific term beyond the first stage's own additive features.

### What the rule decided

The ladder ran on all 25,856 variants, 101 groups and 5,664 sites, with 99 of the 101 backgrounds carrying a profile and 2 (`5UP5.pdb`, `5UYO.pdb`, the two designed mini-proteins the catalogue labels natural) carrying the declared absence. Increments are paired reductions in group-equal held-out mean squared error in kcal²/mol², and the Spearman column is the paired within-background rank increment over the same standing set.

| Candidate | Offered over | Mean squared error increment, 20260923 / 20260924 / 20260925 | Rank increment, same order | Verdict |
| --- | --- | --- | --- | --- |
| Composition | base | +0.00007 / **−0.00185** / +0.00161 | +0.00081 / +0.00105 / +0.00044 | discarded |
| Local chemistry | base | +0.00390 / +0.00670 / +0.00524 | +0.00117 / +0.00243 / +0.00103 | **kept** |
| Mutation-local profile | base + chemistry | +0.01876 / **−1.62082** / +0.00165 | +0.04006 / +0.04804 / +0.03112 | discarded |
| Bounded profile restatement | base + chemistry | +0.02372 / **−1.33231** / +0.00438 | +0.04866 / +0.05796 / +0.04026 | discarded |
| Nonlinear-additive response | base + chemistry | +0.02529 / +0.02598 / +0.02067 | +0.03372 / +0.03261 / +0.03406 | **kept** |

The **qualified control set is the directed substitution identity, the mutated-position geometry, the local chemistry block and the nonlinear-additive global response**, 462 coordinates. On that set the group-equal held-out mean squared error is 0.65115, 0.64887 and 0.65747 kcal²/mol² at the three seeds against a no-effect null of 1.30776, so the qualified controls remove 49.7% to 50.4% of the null's squared error, and the within-background Spearman is +0.4890, +0.4914 and +0.4891. Read as its own contribution rather than as a ladder step, the qualified set reduces the no-effect null's squared error by **0.65661 [0.56792, 0.75029] kcal²/mol²** at seed 20260923 on 101 resampled groups, and its within-background Spearman interval is [+0.46641, +0.51011]: the controls these increments are read over are a competent held-out predictor of this endpoint by a wide margin, not a formality.

**Each candidate's own contribution carries an interval, and one kept block's interval contains zero.** At the first split seed, on 2,000 group-bootstrap draws over the same 101 groups, the increments are composition **+0.00007 [−0.01645, +0.01589]**, local chemistry **+0.00390 [−0.00068, +0.00864]**, the raw profile block **+0.01876 [−0.00883, +0.04521]**, its bounded restatement **+0.02372 [−0.00536, +0.05190]** and the nonlinear-additive response **+0.02529 [+0.01639, +0.03460]** kcal²/mol². Only the last has an interval excluding zero. The chemistry block is kept by the declared rule — a positive point increment at every one of the three seeds — and not by its own interval, and that distinction is carried rather than smoothed over: a control whose own contribution is indistinguishable from zero would make any increment measured over it uninformative, which is the failure mode the catalogue records three times. Composition is the clearest instance and is discarded for exactly that reason as well as for its sign. What licenses the comparisons below is therefore the qualified set as a whole, whose contribution is resolved far from zero, rather than each block separately.

Composition is discarded on a small failure — the largest degradation is 0.00185 kcal²/mol², of the same order as the increments a model quantity will produce. For a within-background difference endpoint this is the expected outcome: the mutant-minus-wild-type composition change is already determined by the substitution identity, and what remains is a background-level constant that has no rank information inside a held-out background.

The profile block's failure is different in kind and is the gate's main methodological finding. Its rank increment is resolved above zero at every seed — +0.0311 to +0.0480 raw, +0.0403 to +0.0580 bounded — while its squared-error increment is catastrophic at one seed of three, at −1.62 and −1.33 kcal²/mol², two orders of magnitude larger than anything else measured here. **The block transfers rank and does not transfer level.** A bounded restatement of the same information, added after reading the raw block's coordinate ranges and before any model quantity existed anywhere, barely changes it: the raw mutant log frequency sits at the pseudo-count floor of log(10⁻⁶) for more than half of all variants on a near-complete substitution scan against a median of 74 retrieved homologs, and the column weights span 0 to 5,986 effective sequences, so bounding every coordinate to ±6 and replacing the saturated coordinate by an unobserved-mutant indicator was the obvious repair — and it left the same seed failing by the same order of magnitude. The failure is therefore in how the fitted level extrapolates onto one particular held-out fold, not in the coordinate scale.

That leaves the two declared qualification metrics disagreeing about one block, and the gate does not resolve the disagreement by choosing a metric after seeing it. Both control sets are carried:

- The **primary set** is the one the mean-squared-error rule qualified: identity, geometry, chemistry and the nonlinear-additive response, 462 coordinates.
- The **secondary set** adds every discarded candidate whose rank increment is positive at all three seeds, with a superseded candidate judged only through its restatement: identity, geometry, composition, chemistry, the bounded profile and the nonlinear-additive response, 512 coordinates.

Each rule licenses the increments read on its own metric. Squared-error increments over the primary set and rank increments over the secondary set are the licensed cells; squared-error increments over the secondary set and rank increments over the primary set are reported beside them as sensitivities.

The arm's own tokenisation descriptors are offered to each baseline under the same rule, because a representation difference can be nonzero purely through segmentation. An arm whose descriptors lower its own baseline is reported as such and its increments are read over the baseline without them.

## What was compared

### The designs, and the baseline every arm was read over

Each arm contributes two quantities, added separately and never only jointly: the **likelihood difference** M_mut − M_WT in nats, one column, and the **projected representation difference** R_mut − R_WT over the four pooled blocks, 1,024 columns at 256 coordinates per block. Each is offered to the primary and to the secondary control set, and each arm's own tokenisation descriptors are offered to both baselines first under the same qualification rule the controls faced.

**No arm's tokenisation descriptors qualified: 0 of 33.** Every arm's segmentation block lowers the held-out squared error of the set it augments at at least one seed, so every arm is read over the control set itself. The consequence is stronger than a matched baseline: the baseline carries no arm-specific column at all, and its group-equal held-out mean squared error is therefore literally one number per seed rather than 33 nearly-equal ones — **0.65115, 0.64887 and 0.65747 kcal²/mol²** over the primary set and 0.60336, 0.72003 and 0.63271 over the secondary set, with a between-arm spread of **exactly 0.0** at every seed and both sets. The 33 arms are compared on one support, one partition and one baseline.

Measurement identity is the singleton one the panel's interface forces: batch size one for every arm, float32 for 31 arms and bfloat16 for the two ProGen3 arms, whose mixture-of-experts block has no kernel above float16. Re-running the forward pass reproduces the likelihood to **3.05 × 10⁻⁵ nats** and the feature block to a relative L2 of **7.42 × 10⁻⁷** at the worst of 33 arms and 101 backgrounds; at batch size one this records run-to-run reproducibility of the same single-row computation and **not** batch-composition invariance, so the declared 0.001 precision gate is structurally vacuous here rather than passed. On the three arms carrying the declared permutation check — ByGPT5-medium-en, GPT-2 and ProGen3-3B — permuting the held-out group labels over 5,376 target rows and 5,397 state rows moves the held-out prediction of every one of the twelve designs by **0.0**, so no held-out label reaches feature scaling, penalty selection or the first-stage response.

### The likelihood increment, the licensed cell

Paired reduction in group-equal held-out mean squared error in kcal²/mol², model likelihood added to the primary control set, with 2,000-draw group-bootstrap intervals over the 101 family groups at each of the three prespecified split seeds. The last two columns are the same contrast refitted with the declared remote purge in place, on identical held-out rows.

| Arm | Interface | Seed 20260923 | Seed 20260924 | Seed 20260925 | Verdict | 20/80 | 20/50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bygpt5-base-en | byte | -0.00007 [-0.00033, +0.00016] | +0.00005 [-0.00019, +0.00029] | -0.00022 [-0.00072, +0.00027] | **unresolved** | unresolved | unresolved |
| bygpt5-medium-en | byte | -0.00011 [-0.00033, +0.00013] | +0.00001 [-0.00027, +0.00030] | -0.00029 [-0.00065, +0.00004] | **unresolved** | unresolved | unresolved |
| bygpt5-small-en | byte | -0.00003 [-0.00011, +0.00005] | -0.00003 [-0.00009, +0.00003] | -0.00020 [-0.00044, +0.00001] | **unresolved** | unresolved | unresolved |
| dialogpt-small | BPE | -0.00012 [-0.00022, -0.00003] | -0.00018 [-0.00033, -0.00004] | -0.00002 [-0.00010, +0.00005] | **unresolved** | unresolved | unresolved |
| galactica-1.3b | BPE | +0.00028 [-0.00009, +0.00063] | +0.00037 [-0.00003, +0.00088] | +0.00076 [-0.00009, +0.00166] | **unresolved** | unresolved | unresolved |
| galactica-125m | BPE | +0.00012 [-0.00018, +0.00043] | +0.00010 [-0.00023, +0.00045] | -0.00003 [-0.00028, +0.00024] | **unresolved** | unresolved | unresolved |
| galactica-30b | BPE | +0.00080 [-0.00064, +0.00235] | +0.00102 [-0.00031, +0.00250] | +0.00098 [-0.00128, +0.00305] | **unresolved** | unresolved | unresolved |
| galactica-6.7b | BPE | +0.00077 [-0.00005, +0.00190] | +0.00076 [-0.00016, +0.00217] | +0.00083 [-0.00033, +0.00186] | **unresolved** | unresolved | unresolved |
| gpt2 | BPE | -0.00006 [-0.00014, +0.00001] | -0.00003 [-0.00013, +0.00007] | -0.00006 [-0.00022, +0.00009] | **unresolved** | unresolved | unresolved |
| gpt2-large | BPE | +0.00007 [-0.00012, +0.00025] | -0.00001 [-0.00024, +0.00024] | -0.00002 [-0.00028, +0.00024] | **unresolved** | unresolved | unresolved |
| gpt2-medium | BPE | -0.00001 [-0.00026, +0.00023] | -0.00000 [-0.00030, +0.00030] | -0.00035 [-0.00082, +0.00010] | **unresolved** | unresolved | unresolved |
| gpt2-xl | BPE | +0.00003 [-0.00031, +0.00037] | +0.00009 [-0.00030, +0.00051] | +0.00001 [-0.00048, +0.00051] | **unresolved** | unresolved | unresolved |
| instructprotein | BPE | +0.00024 [-0.00162, +0.00209] | -0.00019 [-0.00214, +0.00159] | -0.00378 [-0.00774, -0.00016] | **unresolved** | unresolved | unresolved |
| llama-2-7b | BPE | -0.00004 [-0.00021, +0.00012] | +0.00003 [-0.00013, +0.00021] | -0.00046 [-0.00089, -0.00008] | **unresolved** | unresolved | unresolved |
| llama-3.2-3b | BPE | -0.00015 [-0.00034, +0.00002] | -0.00012 [-0.00028, +0.00004] | -0.00023 [-0.00044, -0.00002] | **unresolved** | unresolved | unresolved |
| progen2-base | residue | +0.00607 [+0.00064, +0.01134] | +0.00521 [-0.00018, +0.01020] | +0.00571 [-0.00354, +0.01395] | **unresolved** | unresolved | unresolved |
| progen2-large | residue | +0.00486 [-0.00039, +0.00997] | +0.00474 [-0.00009, +0.00943] | +0.00329 [-0.00567, +0.01161] | **unresolved** | unresolved | unresolved |
| progen2-medium | residue | +0.00776 [+0.00195, +0.01350] | +0.00715 [+0.00173, +0.01271] | +0.00708 [-0.00293, +0.01642] | **unresolved** | unresolved | above zero |
| progen2-small | residue | +0.00497 [+0.00021, +0.00990] | +0.00520 [+0.00038, +0.01008] | +0.00437 [-0.00486, +0.01266] | **unresolved** | unresolved | unresolved |
| progen2-xlarge | residue | +0.00698 [+0.00162, +0.01210] | +0.00587 [+0.00028, +0.01101] | +0.00658 [-0.00229, +0.01475] | **unresolved** | unresolved | above zero |
| progen3-112m | residue | +0.00764 [+0.00335, +0.01219] | +0.00814 [+0.00365, +0.01295] | +0.00939 [+0.00226, +0.01651] | **above zero** | unresolved | above zero |
| progen3-3b | residue | +0.02260 [+0.01333, +0.03144] | +0.01307 [+0.00690, +0.01879] | +0.01335 [+0.00203, +0.02368] | **above zero** | above zero | above zero |
| prollama | residue | +0.00914 [+0.00475, +0.01432] | +0.00852 [+0.00457, +0.01297] | +0.00821 [+0.00060, +0.01550] | **above zero** | above zero | above zero |
| prollama-stage-1 | residue | +0.00793 [+0.00394, +0.01272] | +0.00721 [+0.00388, +0.01119] | +0.00699 [-0.00026, +0.01446] | **unresolved** | unresolved | above zero |
| proteinglm-7b-clm | residue | +0.01380 [+0.00655, +0.02064] | +0.01289 [+0.00572, +0.02009] | +0.01300 [+0.00108, +0.02424] | **above zero** | above zero | above zero |
| protgpt2 | BPE | +0.00403 [-0.00014, +0.00815] | +0.00324 [-0.00149, +0.00774] | -0.00002 [-0.00801, +0.00740] | **unresolved** | unresolved | unresolved |
| protgpt3-1.3b | BPE | +0.00462 [+0.00024, +0.00875] | +0.00400 [-0.00068, +0.00840] | +0.00573 [-0.00065, +0.01177] | **unresolved** | unresolved | above zero |
| qwen2.5-0.5b | BPE | +0.00003 [-0.00013, +0.00018] | -0.00006 [-0.00026, +0.00011] | +0.00004 [-0.00027, +0.00033] | **unresolved** | unresolved | unresolved |
| qwen2.5-0.5b-instruct | BPE | -0.00002 [-0.00023, +0.00017] | -0.00003 [-0.00031, +0.00020] | +0.00005 [-0.00041, +0.00045] | **unresolved** | unresolved | unresolved |
| qwen2.5-32b | BPE | -0.00005 [-0.00020, +0.00008] | +0.00001 [-0.00013, +0.00013] | -0.00005 [-0.00025, +0.00014] | **unresolved** | unresolved | unresolved |
| qwen2.5-7b | BPE | +0.00031 [-0.00015, +0.00083] | +0.00028 [-0.00020, +0.00075] | +0.00036 [-0.00028, +0.00097] | **unresolved** | unresolved | unresolved |
| qwen3-8b-base | BPE | -0.00007 [-0.00018, +0.00004] | -0.00043 [-0.00075, -0.00017] | -0.00012 [-0.00032, +0.00006] | **unresolved** | unresolved | unresolved |
| rita-xl | residue | +0.00488 [+0.00032, +0.00919] | +0.00436 [-0.00021, +0.00880] | +0.00501 [-0.00179, +0.01139] | **unresolved** | unresolved | above zero |

Panel median of the three-seed means +0.000315 kcal²/mol². By interface: residue 11 arms, median seed mean +0.00733, 11 positive, 4 above zero, 0 below zero; BPE 19 arms, median seed mean +0.00000, 10 positive, 0 above zero, 0 below zero; byte 3 arms, median seed mean -0.00009, 0 positive, 0 above zero, 0 below zero.

### The likelihood rank increment, the other licensed cell

Paired within-background Spearman increment over the **secondary** control set, which is the set the correlation rule qualified. Points only; the intervals are in the artefact.

| Arm | Interface | Seed 20260923 | Seed 20260924 | Seed 20260925 | Verdict | 20/80 | 20/50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bygpt5-base-en | byte | +0.00001 | +0.00014 | +0.00003 | **unresolved** | unresolved | unresolved |
| bygpt5-medium-en | byte | -0.00072 | +0.00016 | +0.00013 | **unresolved** | unresolved | unresolved |
| bygpt5-small-en | byte | -0.00007 | +0.00001 | -0.00013 | **unresolved** | unresolved | unresolved |
| dialogpt-small | BPE | -0.00010 | -0.00025 | -0.00009 | **unresolved** | unresolved | unresolved |
| galactica-1.3b | BPE | +0.00011 | -0.00010 | +0.00254 | **unresolved** | unresolved | unresolved |
| galactica-125m | BPE | -0.00012 | -0.00010 | -0.00009 | **unresolved** | unresolved | unresolved |
| galactica-30b | BPE | +0.00013 | -0.00017 | -0.00002 | **unresolved** | unresolved | unresolved |
| galactica-6.7b | BPE | -0.00206 | -0.00454 | +0.00002 | **unresolved** | unresolved | unresolved |
| gpt2 | BPE | +0.00001 | -0.00004 | +0.00001 | **unresolved** | unresolved | unresolved |
| gpt2-large | BPE | +0.00008 | -0.00008 | -0.00001 | **unresolved** | unresolved | unresolved |
| gpt2-medium | BPE | -0.00236 | -0.00002 | -0.00006 | **unresolved** | unresolved | unresolved |
| gpt2-xl | BPE | -0.00108 | +0.00020 | +0.00030 | **unresolved** | unresolved | unresolved |
| instructprotein | BPE | -0.00180 | +0.00014 | +0.00055 | **unresolved** | unresolved | above zero |
| llama-2-7b | BPE | -0.00009 | +0.00054 | -0.00003 | **unresolved** | unresolved | unresolved |
| llama-3.2-3b | BPE | -0.00163 | -0.00014 | +0.00224 | **unresolved** | unresolved | unresolved |
| progen2-base | residue | -0.00626 | +0.00322 | +0.00455 | **unresolved** | above zero | above zero |
| progen2-large | residue | -0.00713 | +0.00247 | +0.00406 | **unresolved** | above zero | above zero |
| progen2-medium | residue | -0.00533 | +0.00395 | +0.00562 | **unresolved** | above zero | above zero |
| progen2-small | residue | -0.00734 | +0.00247 | +0.00343 | **unresolved** | above zero | above zero |
| progen2-xlarge | residue | +0.00194 | +0.00354 | +0.00479 | **unresolved** | above zero | above zero |
| progen3-112m | residue | +0.00325 | +0.00146 | +0.00724 | **unresolved** | unresolved | unresolved |
| progen3-3b | residue | +0.00699 | +0.00768 | +0.00761 | **above zero** | above zero | above zero |
| prollama | residue | +0.00376 | +0.00562 | +0.00729 | **unresolved** | above zero | above zero |
| prollama-stage-1 | residue | +0.00453 | +0.00513 | +0.00346 | **unresolved** | above zero | above zero |
| proteinglm-7b-clm | residue | +0.00415 | +0.00234 | +0.00771 | **unresolved** | above zero | above zero |
| protgpt2 | BPE | +0.00097 | +0.00233 | +0.00294 | **unresolved** | unresolved | above zero |
| protgpt3-1.3b | BPE | -0.00701 | +0.00274 | +0.00408 | **unresolved** | unresolved | above zero |
| qwen2.5-0.5b | BPE | -0.00082 | -0.00009 | -0.00000 | **unresolved** | unresolved | unresolved |
| qwen2.5-0.5b-instruct | BPE | +0.00000 | -0.00013 | -0.00005 | **unresolved** | unresolved | unresolved |
| qwen2.5-32b | BPE | -0.00082 | -0.00005 | -0.00010 | **unresolved** | unresolved | unresolved |
| qwen2.5-7b | BPE | -0.00969 | -0.00023 | +0.00008 | **unresolved** | unresolved | unresolved |
| qwen3-8b-base | BPE | +0.00001 | -0.00021 | -0.00029 | **unresolved** | unresolved | unresolved |
| rita-xl | residue | -0.00642 | +0.00307 | +0.00429 | **unresolved** | above zero | above zero |

Panel median -0.000021. By interface: residue 11 arms, median seed mean +0.00342, 9 positive, 1 above zero, 0 below zero; BPE 19 arms, median seed mean -0.00010, 4 positive, 0 above zero, 0 below zero; byte 3 arms, median seed mean -0.00006, 1 positive, 0 above zero, 0 below zero.

Read as a sensitivity rather than as a licensed cell, the same rank increment over the primary set resolves above zero in 13 arms and the squared-error increment over the secondary set in none; the two unlicensed cells therefore bracket the licensed pair rather than contradicting it.

### The representation increment — provisional, and subject to recomputation

**These two tables are provisional.** They are conditional on one compressed linear readout of four pooled block summaries at the retained depths, and the readout-class reassessment is still running. No verdict about representational content is drawn from them here, and the full-width states are retained so the revisit is a refit rather than a re-extraction.

Squared-error increment over the primary set, points only, provisional:

| Arm | Interface | Seed 20260923 | Seed 20260924 | Seed 20260925 | Verdict | 20/80 | 20/50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bygpt5-base-en | byte | -0.00448 | -0.00443 | -0.01324 | **below zero** | unresolved | below zero |
| bygpt5-medium-en | byte | -0.00580 | -0.00683 | -0.01285 | **below zero** | unresolved | below zero |
| bygpt5-small-en | byte | -0.00887 | -0.00735 | -0.01538 | **below zero** | unresolved | below zero |
| dialogpt-small | BPE | -0.00866 | -0.00706 | -0.01460 | **below zero** | unresolved | below zero |
| galactica-1.3b | BPE | -0.00189 | +0.00008 | -0.00846 | **unresolved** | unresolved | unresolved |
| galactica-125m | BPE | -0.00006 | +0.00068 | -0.00885 | **unresolved** | unresolved | unresolved |
| galactica-30b | BPE | -0.00001 | -0.00078 | -0.00730 | **unresolved** | unresolved | unresolved |
| galactica-6.7b | BPE | -0.00268 | -0.00214 | -0.00964 | **unresolved** | unresolved | unresolved |
| gpt2 | BPE | -0.00347 | -0.00651 | -0.01158 | **unresolved** | unresolved | below zero |
| gpt2-large | BPE | -0.00645 | -0.00505 | -0.01510 | **below zero** | unresolved | below zero |
| gpt2-medium | BPE | -0.00682 | -0.00926 | -0.01566 | **below zero** | unresolved | below zero |
| gpt2-xl | BPE | -0.00627 | -0.00936 | -0.01705 | **below zero** | unresolved | below zero |
| instructprotein | BPE | +0.01204 | +0.01407 | +0.01102 | **unresolved** | unresolved | unresolved |
| llama-2-7b | BPE | -0.00692 | -0.00461 | -0.01351 | **below zero** | unresolved | below zero |
| llama-3.2-3b | BPE | -0.00569 | -0.00362 | -0.01123 | **unresolved** | unresolved | below zero |
| progen2-base | residue | +0.02808 | +0.03093 | +0.03016 | **above zero** | above zero | above zero |
| progen2-large | residue | +0.02286 | +0.03115 | +0.02529 | **above zero** | above zero | unresolved |
| progen2-medium | residue | +0.02262 | +0.02610 | +0.02171 | **above zero** | above zero | unresolved |
| progen2-small | residue | +0.01996 | +0.02828 | +0.02422 | **above zero** | above zero | unresolved |
| progen2-xlarge | residue | +0.01978 | +0.02349 | +0.01965 | **above zero** | above zero | unresolved |
| progen3-112m | residue | +0.02744 | +0.03072 | +0.02730 | **above zero** | above zero | above zero |
| progen3-3b | residue | +0.03213 | +0.03756 | +0.03484 | **above zero** | above zero | above zero |
| prollama | residue | +0.01161 | +0.01099 | +0.00674 | **unresolved** | unresolved | unresolved |
| prollama-stage-1 | residue | +0.01736 | +0.01695 | +0.01454 | **above zero** | above zero | above zero |
| proteinglm-7b-clm | residue | +0.06121 | +0.05874 | +0.05988 | **above zero** | above zero | above zero |
| protgpt2 | BPE | +0.02105 | +0.02287 | +0.01798 | **above zero** | above zero | unresolved |
| protgpt3-1.3b | BPE | +0.02609 | +0.02803 | +0.02238 | **above zero** | above zero | above zero |
| qwen2.5-0.5b | BPE | -0.00660 | -0.00334 | -0.01143 | **unresolved** | unresolved | below zero |
| qwen2.5-0.5b-instruct | BPE | -0.00552 | -0.00257 | -0.01154 | **unresolved** | unresolved | below zero |
| qwen2.5-32b | BPE | -0.00528 | -0.00496 | -0.01076 | **unresolved** | unresolved | below zero |
| qwen2.5-7b | BPE | -0.00623 | -0.00529 | -0.01084 | **below zero** | unresolved | below zero |
| qwen3-8b-base | BPE | -0.01300 | -0.01329 | -0.02198 | **below zero** | unresolved | below zero |
| rita-xl | residue | +0.01413 | +0.01337 | +0.00951 | **unresolved** | unresolved | unresolved |

Panel median -0.003423 kcal²/mol². By interface: residue 11 arms, median seed mean +0.02415, 11 positive, 9 above zero, 0 below zero; BPE 19 arms, median seed mean -0.00700, 3 positive, 2 above zero, 7 below zero; byte 3 arms, median seed mean -0.00850, 0 positive, 0 above zero, 3 below zero.

Within-background rank increment over the secondary set, points only, provisional:

| Arm | Interface | Seed 20260923 | Seed 20260924 | Seed 20260925 | Verdict | 20/80 | 20/50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bygpt5-base-en | byte | -0.01383 | -0.02512 | -0.01999 | **below zero** | below zero | below zero |
| bygpt5-medium-en | byte | -0.01618 | -0.02569 | -0.02357 | **below zero** | below zero | below zero |
| bygpt5-small-en | byte | -0.01849 | -0.02353 | -0.02490 | **below zero** | below zero | below zero |
| dialogpt-small | BPE | -0.01658 | -0.02226 | -0.01613 | **below zero** | below zero | unresolved |
| galactica-1.3b | BPE | -0.01000 | -0.01679 | -0.01908 | **below zero** | below zero | below zero |
| galactica-125m | BPE | -0.01437 | -0.02100 | -0.01958 | **below zero** | below zero | below zero |
| galactica-30b | BPE | -0.01175 | -0.02089 | -0.02289 | **below zero** | below zero | below zero |
| galactica-6.7b | BPE | -0.02580 | -0.01340 | -0.02185 | **below zero** | below zero | below zero |
| gpt2 | BPE | -0.01355 | -0.02209 | -0.01679 | **below zero** | below zero | below zero |
| gpt2-large | BPE | -0.01470 | -0.02300 | -0.02152 | **below zero** | below zero | unresolved |
| gpt2-medium | BPE | -0.01399 | -0.02117 | -0.02218 | **below zero** | below zero | unresolved |
| gpt2-xl | BPE | -0.01596 | -0.02747 | -0.01829 | **below zero** | below zero | unresolved |
| instructprotein | BPE | +0.00257 | -0.00026 | +0.00051 | **unresolved** | unresolved | unresolved |
| llama-2-7b | BPE | -0.01662 | -0.02195 | -0.01828 | **below zero** | below zero | unresolved |
| llama-3.2-3b | BPE | -0.01380 | -0.02054 | -0.01727 | **below zero** | below zero | unresolved |
| progen2-base | residue | +0.02285 | +0.01530 | +0.03485 | **above zero** | above zero | above zero |
| progen2-large | residue | +0.01929 | +0.01506 | +0.02652 | **above zero** | unresolved | above zero |
| progen2-medium | residue | +0.01813 | +0.01607 | +0.03533 | **above zero** | above zero | above zero |
| progen2-small | residue | +0.01735 | +0.01593 | +0.02800 | **above zero** | unresolved | above zero |
| progen2-xlarge | residue | +0.01002 | +0.00986 | +0.02113 | **unresolved** | unresolved | above zero |
| progen3-112m | residue | +0.01532 | +0.01075 | +0.02680 | **unresolved** | unresolved | above zero |
| progen3-3b | residue | +0.01344 | +0.02334 | +0.04309 | **above zero** | above zero | above zero |
| prollama | residue | +0.00065 | -0.00785 | -0.00108 | **unresolved** | unresolved | unresolved |
| prollama-stage-1 | residue | +0.00331 | -0.00254 | -0.00099 | **unresolved** | unresolved | unresolved |
| proteinglm-7b-clm | residue | +0.04543 | +0.03824 | +0.05856 | **above zero** | above zero | above zero |
| protgpt2 | BPE | +0.01624 | +0.01317 | +0.00959 | **unresolved** | unresolved | above zero |
| protgpt3-1.3b | BPE | +0.01144 | +0.00945 | +0.01358 | **unresolved** | unresolved | unresolved |
| qwen2.5-0.5b | BPE | -0.01552 | -0.01964 | -0.01902 | **below zero** | below zero | unresolved |
| qwen2.5-0.5b-instruct | BPE | -0.01447 | -0.01990 | -0.01793 | **below zero** | below zero | unresolved |
| qwen2.5-32b | BPE | -0.03416 | -0.02608 | -0.02577 | **below zero** | below zero | below zero |
| qwen2.5-7b | BPE | -0.02606 | -0.02197 | -0.01476 | **below zero** | below zero | below zero |
| qwen3-8b-base | BPE | -0.01843 | -0.02509 | -0.02117 | **below zero** | below zero | unresolved |
| rita-xl | residue | +0.00289 | +0.00196 | +0.02034 | **unresolved** | unresolved | above zero |

Panel median -0.017432. By interface: residue 11 arms, median seed mean +0.02029, 9 positive, 6 above zero, 0 below zero; BPE 19 arms, median seed mean -0.01832, 3 positive, 0 above zero, 16 below zero; byte 3 arms, median seed mean -0.02181, 0 positive, 0 above zero, 3 below zero.

### What the remote purge did to the comparison

The purge removes, for each outer fold, every training group whose wild type reaches the declared similarity to any held-out group, and leaves the held-out rows untouched. At 20% identity and 80% coverage it removes 0 to 2 training groups per fold; at 20% identity and 50% coverage it removes 26 to 38 of about 80, a 33% to 48% training reduction.

No arm's licensed increment is removed in magnitude by either purge. Three of the four resolved arms stay resolved at 20/80 and all four at 20/50, where **9** arms are resolved against 4 unpurged, and on the rank cell 10 and 13 against 1. That the count rises is a property of the purge and not extra evidence: removing a third of the training groups raises the baseline's own held-out error more than it raises the augmented design's, so the paired increment grows while resting on a weaker fit.

One arm changes label in the other direction. ProGen3-112M is unresolved at 20/80 with purged points of +0.00724, +0.01012 and +0.00790 kcal²/mol² against +0.00764, +0.00814 and +0.00939 unpurged, so its point estimates are unchanged and one of its three intervals widens across zero under a purge that removed 0 to 2 training groups per fold — a small perturbation of a borderline interval rather than a family-structure effect. The defensible reading is the one stated in the cohort section — the purge bounds dependence on nearest-detected training neighbours, which on this cohort is a chance-level neighbour set, and it certifies no remote-family disjointness.

## The class-axis recomputation, and why the depth sweep's boundary cells cannot be read against this gate

The readout reassessment selected a class and a depth per arm. This gate's fits were re-run from the inputs its records bind, over the **full 33-arm panel at all three split seeds — 99 cells, none missing**, and the published verdict **stands**.

| Quantity family | Maximum point change | Maximum interval-endpoint change |
| --- | ---: | ---: |
| Likelihood | 1.46e-16 | 2.08e-16 |
| Representation | 4.16e-16 | 7.91e-16 |

**Zero resolved sign changes over 2,970 compared summary nodes**, and no arm-and-metric pair changes at any seed, let alone all three. The table above is the four licensed quantities — the primary and secondary likelihood and representation increments, 396 nodes. The comparison itself covers every node of both records carrying a point and an interval: 2,970 across the 33 arms, 1,188 likelihood, 1,188 representation and 594 baseline, null and tokenisation nodes, with no node present in one record and absent from the other in either direction. On that wider set the largest movement is **3.19e-14** in a point and **8.09e-14** in an interval endpoint — two orders above the licensed four, six below the 1e-8 pipeline tolerance, and concentrated where the licensed four do not reach: the `secondary_*` increments and the remote-stratification strata, `llama-3.2-3b` at identity20/coverage80 for the point and `gpt2-large`'s secondary representation increment for the interval endpoint.

**The null is not silence, because the control could have fired.** 917 of the 2,970 compared nodes had an interval endpoint close enough to zero that the movement observed in their own cell could have carried it across — 747 likelihood, 74 representation and 96 other, of which 136 and 18 sit among the four licensed quantities — and none crossed. Read like for like on the licensed quantities across three gates — 254 on crossed controls, 136 here, 168 on external confirmation — quantities at risk did not move, which makes that a property of the measurement rather than a feature of one gate.

**And the depth sweep's boundary-lifted cells have no counterpart here.** This cohort renders **no position-resolved summaries at all**: its depth archives hold two summaries per block, mean and last, and no `mut` or `suffix`, because a position-resolved summary requires the wild type and each of its mutants to render to the same one-token-per-residue grid and this cohort does not. Any attempt to read the sweep's position-resolved findings — including its boundary-lifted cells — against this gate would be comparing against summaries these archives could not contain. The pooled half of the sweep transfers; the position-resolved half does not exist for this endpoint.

A further consequence for the depth axis: the admitted design here pools four summaries across **two** depths, so selecting one depth is not a narrower version of it but a differently constructed measurement at half the capacity. That question is asked separately as a matched-capacity contrast and is reported as a new measurement, explicitly not comparable to the published increment.

## The equal-capacity depth contrast, and what it says about the selection

This is the matched-capacity measurement that question asks for, and **four qualifications travel with every number in it**: it varies one axis, depth; it is unadjusted for multiplicity; its representation is **512 coordinates** — one block's `mean` and `last` at 256 each under the declared depth projection — against the published **1,024** pooled across two depths, so **it is not comparable to the published increment**; and it is this gate's own cohort, folds, control ladder, ridge grid and contrast form with nothing else changed. Seven arms, 45 blocks, three split seeds, 135 fits, no failures. The control that makes the comparison readable fired on every one of them: with only the representation block varying, the baseline predictions are identical across all depths of an arm at a given seed, **135 of 135**.

| Arm | Admitted depths | Selected | Selected mean | Best admitted | Difference | Highest depth measured |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ProGen2-large | 15, 31 | 25 | +0.013130 | +0.020825 | **−0.007695** | 8 at +0.025772 |
| ProGen2-medium | 13, 26 | 23 | +0.027999 | +0.027008 | +0.000991 | 20 at +0.036033 |
| ProGen2-xlarge | 15, 31 | 28 | +0.029989 | +0.014677 | **+0.015311** | 8 at +0.030091 |
| ProGen3-112M | 4, 9 | 9 | +0.020453 | +0.021997 | — (selected is admitted) | 4 at +0.021997 |
| ProGen3-3B | 11, 23 | 11 | +0.029823 | +0.036137 | — (selected is admitted) | 6 at +0.042170 |
| ProteinGLM-7B-CLM | 17, 35 | 30 | +0.052462 | +0.047272 | +0.005189 | 30 at +0.052462 |
| ProtGPT2 | 17, 35 | 21 | +0.035823 | +0.013471 | **+0.022351** | 21 at +0.035823 |

**The transfer question is answered, and the answer is mostly yes.** On the five arms whose selected depth is distinct from both admitted ones, the selected depth carries **more** than the better admitted depth on four — ProtGPT2 by +0.022351, ProGen2-xlarge by +0.015311, ProteinGLM-7B-CLM by +0.005189, ProGen2-medium by +0.000991 — and **less** on one, ProGen2-large by −0.007695. So a depth chosen on the readout panel does, on this different cohort and at equal capacity, usually carry more representation information than the depths the admitted design pools. On the remaining two arms the selection landed on a depth the admitted design already uses, which is its own kind of agreement.

**A second cohort answers the same question the other way, so the reading above is cohort-specific.** The remote-homology gate runs this identical contrast — same seven arms, same block rule, same 512 coordinates, same declared projection, its own cohort and control ladder — and there the selected depth carries **less** than the best admitted depth on four of the five arms where it is distinct, against more on four of five here. Only ProGen2-medium exceeds on both cohorts and only ProGen2-large falls short on both. What transfers from the readout panel is therefore not a property of the depth alone: it depends on the endpoint being predicted. That gate's record carries the table.

**Two qualifications on that reading, both measured here.** The selected depth is the best depth measured on only two of seven arms (ProteinGLM-7B-CLM and ProtGPT2); on four others an unselected, unadmitted block carries more, by as much as +0.012 on ProGen3-3B — so the selection identifies a *better-than-admitted* depth rather than the best one. And these are point means over three seeds without an adjustment for the 45 blocks compared, so a single arm's margin is not a resolved claim; the pattern across arms is what the measurement supports.



**Accounting.** 33 of 33 arms have a record. Two earlier attempts, `galactica-30b` and `qwen2.5-32b`, were **superseded rather than failed**: they were run against the wrong extraction run before this gate's arms were found to span two of them, and both were refit against their own. The screen ran at eight BLAS threads, which this gate's stage permits and which a single-cell replay reproduced at 1e-13 over 919 numbers.

## Verdict

**The likelihood quantity carries information about single-substitution stability beyond the qualified controls, in 4 of 33 arms.** Those four are ProGen3-3B (+0.02260 [+0.01333, +0.03144], +0.01307 [+0.00690, +0.01879], +0.01335 [+0.00203, +0.02368] kcal²/mol² at the three seeds), ProteinGLM-7B-CLM (+0.01380, +0.01289, +0.01300), ProLLaMA Stage 2 (+0.00914, +0.00852, +0.00821) and ProGen3-112M (+0.00764, +0.00814, +0.00939), every one a residue-level protein-pretrained checkpoint. **No arm of the 33 is resolved below zero.** Against a qualified control set that already removes half of the no-effect null's squared error, the resolved increments are 1.2% to 3.5% of the remaining error, so the quantity is resolvable and small.

**The interface pattern is the part that does not depend on one arm's interval.** All 11 residue-level arms have a positive three-seed mean, at a stratum median of +0.00733 kcal²/mol²; the 19 BPE arms sit at +0.0000016 with 10 of 19 positive, and the 3 byte-level arms at −0.00009 with none positive. On the rank metric over the secondary control set only ProGen3-3B resolves, and the stratum medians keep the same ordering (+0.00342 residue, −0.00010 BPE, −0.00006 byte). A protein-pretrained residue-level likelihood adds something to this endpoint that a text checkpoint reading the same sequences does not, and the gate's licensed claim is that pattern together with the four arms.

**The multiplicity is declared and no interval is adjusted for it.** The intervals condition on the fitted cross-validation predictions, omit training and split variation, and are not adjusted for 33 arms, three split seeds, two control sets, two model quantities, two metrics or two remote thresholds; the licensed cells alone are 33 arms times two model quantities, or 66, and the pattern across arms rather than any single arm's interval is the defensible reading. Requiring all three per-seed intervals to exclude zero in one direction is a conjunction over three correlated resamplings of one support, not three independent tests, so it carries no nominal level; it is a stability requirement, not a corrected test.

**No verdict about representational content is written here.** The representation increments above are reported with their points and their strata and are provisional pending the readout-class reassessment, after which they are recomputed from the retained states.

## Limitations

The support is the binding limitation and it is not a caveat on the numbers, it is part of what they mean. **The admitted support represents resolvable stability states**: variants outside the assay's resolvable window are absent by construction, the 40 backgrounds whose wild type could not be resolved are absent as a stratum with a thirteenfold higher censored-row share than the retained ones, and the retained backgrounds' own alignments are shallow. Nothing here estimates the effect of a strongly destabilising substitution.

The endpoint's agreement with the source's own `ddG_ML` column at a root-mean-square residual of 0.0419 kcal/mol is **self-consistency and not validation**: that column is derived from the same `dG_ML` values this endpoint differences, so the agreement establishes that we compute what the source computes and says nothing about measurement accuracy. The accuracy statement is the per-channel discordance floor of 0.2337 [0.2209, 0.2458] kcal/mol, five times larger.

The remote stratification **certifies no remote-family disjointness**. Every observed alignment-edge count among the 101 one-per-group wild types lies inside a composition-preserving shuffle null at every threshold tested, so there is no alignment-detectable residual family structure to stratify on; the purge bounds dependence on a chance-level nearest-neighbour set, and homology below those thresholds is undetected.

Multiplicity is unadjusted across 33 arms, three split seeds, two control sets, two model quantities, two metrics and two remote thresholds, as stated in the verdict.

The control set that licenses these increments is competent as a whole — it removes 49.7% to 50.4% of the no-effect null's squared error — but one of its two kept blocks, local chemistry, has a first-seed contribution interval of [−0.00068, +0.00864] kcal²/mol² that contains zero. It is kept by the declared three-seed sign rule, and that is weaker evidence of competence than the nonlinear-additive response's [+0.01639, +0.03460].

Two code and runtime facts travel with the panel. Galactica-30B and Qwen2.5-32B were extracted from a snapshot one revision earlier than the other 31 arms, differing only in where one plan-validation function lives; and every extraction and every fit ran under the pod's own interpreter, Python 3.12.7 with numpy 2.1.2, so the panel is internally consistent under one runtime rather than under the workstation's `ct` environment. The 0.001 batch-composition gate is structurally vacuous at batch size one, and the two ProGen3 arms are bfloat16 against float32 elsewhere, which is an interface constraint of their mixture-of-experts block rather than a choice.

The representation cells are conditional on the compressed linear readout class and are provisional; the likelihood cells are not, because a likelihood enters as one scalar column where a linear readout is close to the full function space.

## Artifacts

The frozen inputs are in `results/gate_stability_20260924/` on the cluster: `cohort.json` (`1e42c3cc…`), `extraction_plan.json` (content digest `283b4503…`), `profile_features.npz` (`c7d16cca…`) and `controls_qualification.json` (`f4ed91fa…`), with the endpoint digest over the cohort `d7235820…`. The per-arm extractions are `results/external_baseline/20260924000148_fbfa7ddc1f5f/<arm>/` for 31 arms and `…/20260923232409_9ab5f150f2ac/<arm>/` for Galactica-30B and Qwen2.5-32B; each carries one NPZ per background with the projected blocks, the likelihood, the token records and the repeat-check quantities, plus `manifest_<arm>.json` binding the checkpoint files, the code digest and the cohort.

The 33 per-arm fit records are `results/external_baseline/20260924024010_7caaa5ac6216/fit-<arm>/fit_<arm>.json`, gathered into `results/gate_stability_20260924/fits/`, and the panel is `results/gate_stability_20260924/panel/panel.json`.

**What is retained for the readout-class recomputation, and where.** Each fit record carries the complete fold map — the held-out groups of every outer fold at every split seed, in its `nuisance` block, together with the purged training groups of each remote fit — the selected ridge penalty for every design in every fold, the declared design names with their column dimensions, the three split seeds, and the first-stage diagnostics of the nonlinear-additive response. The projection is a declared Gaussian matrix determined by width and seed alone. The unprojected per-state block outputs were written by a separate retention pass into `results/external_baseline/20260924024010_7caaa5ac6216/full-<arm>/`, over the identical cohort, plan, sequences and batch-one setting, from a snapshot whose extraction differs from the primary one only by the flag that writes them. That pass was in flight when this section was first written and it is now **complete for all 33 arms**, verified in-pod on 2026-09-24: every arm's manifest reads status complete at **101 of 101 backgrounds and 25,957 sequences scored**, with 101 projected and 101 full-width archives each, the features shaped (rows, 4, hidden width) at hidden widths of 384 to 7,168. The earlier counts of 4 of 33 here and 3 of 33 in the [experiment log](EXPERIMENT_LOG.md) were both mid-flight readings of the same waves and neither was the final state. A class-level recomputation therefore needs to change the projection or the readout head and refit; it does not need a forward pass.

A **depth-level** recomputation does need one, and that is a bound on this gate rather than a scheduling note: the extraction hooked blocks ⌈(L−1)/2⌉ and L−1 only, so no other block's state exists on this cohort and a readout reassessment that selects any other depth cannot be applied here at all. Audit L53 carries the consequence and the measured cost of removing it — 13.891 GPU-hours over the 33 arms at batch size one, the wall clock this pass itself recorded — and [D1_RECOMPUTATION_PIPELINE.md](D1_RECOMPUTATION_PIPELINE.md) owns the plan.

Runtime logs stay under ignored `logs/d1_gate_stability_20260924/`, and the campaign manifests of every wave are in `scripts/transfer/campaign_gate_stability_*.tsv`.
