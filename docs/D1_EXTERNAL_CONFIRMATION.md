# Does the first-order likelihood finding hold on data used in neither development panel?

This is the external-confirmation step of the [Direction-1 capability map](D1_CAPABILITY_MAP_PLAN.md). The map's strongest reproducible model-side finding is that a residue-level protein-pretrained likelihood carries information about the effect of a single substitution beyond controls that are competent held-out predictors in their own right — four of 33 arms on the [folding-and-stability gate](D1_GATE_STABILITY.md), and a residue-against-text interface pattern that does not rest on any one arm's interval. Both development sources are shared with the readout panel: ProteinGym's 217 deep mutational scans, 64 of whose readout anchors are Tsuboyama 2023, and Tsuboyama 2023 itself. This step asks the same question on a third source with a third assay.

**The answer is yes, and the two statements that carry it do not rest on any single arm's interval.**

**The interface ordering reproduces.** All **11 of 11** residue-level arms carry a positive three-seed mean, at a stratum median of **+0.000444** squared normalised fitness, against **7 of 19** BPE arms at **−0.0000048** and **0 of 3** byte-level arms at **−0.0000114**. That is the ordering the [folding-and-stability gate](D1_GATE_STABILITY.md) found, an order of magnitude apart, on a different source, a different assay and a different measured quantity.

**On rank the split falls on the pretraining corpus rather than on the interface, which is the sharper statement.** **14 of 33 arms resolve above zero, and they are exactly the 14 pretrained on protein sequence**: the 11 residue-level arms plus InstructProtein, ProtGPT2 and ProtGPT3-1.3B, the three protein-corpus BPE checkpoints. The 19 that do not resolve are exactly the text checkpoints. Stratum medians +0.004798 residue, +0.000029 BPE, −0.000092 byte. What separates a resolving arm from a non-resolving one here is what it was pretrained on, not how it segments a sequence.

Two arms also resolve on the squared-error cell and none resolves below zero on either: ProGen2-xlarge and ProteinGLM-7B-CLM, the latter one of the four the development gate resolved.

**What agreement here does and does not mean.** The endpoint is cellular abundance by protein-fragment complementation in *Saccharomyces cerevisiae*, which is a **different measured quantity** from proteolysis-derived stability and from ProteinGym's heterogeneous assays. Agreement across two different measured quantities is stronger evidence than agreement within one, and it is not a repeat measurement of either estimand, so a disagreement here would not by itself have been a failure to replicate. Every sentence below that reports agreement says which of the two it is.

**And this gate does not speak to remote homology, by measurement rather than by judgement.** Banding all 428 retained wild types against the same UniRef50 search the profiles were built from gives 426 near duplicates, 1 at 70–95% identity and 1 with no reported hit, so **95 of the 96 family groups are entirely close and none is remote**. There is no remote stratum here to read. That measurement is what makes the two gates this cycle added **complementary by construction rather than redundant, and neither a weaker version of the other**: this gate supplies a third source and a third assay **without remoteness**, and the [remote-homology gate](D1_GATE_REMOTE_HOMOLOGY.md) supplies **remoteness without independence**, sharing its assay technology and an author with a development source. The section below gives the banding in full.

## The endpoint

**f = normalized_fitness** of one single amino-acid substitution on the Domainome 1.0 abundance-PCA scale, dimensionless, normalised by the source per domain so that the wild type reads 0 and the median nonsense variant reads about −1. Larger is greater cellular abundance, so a destabilising or abundance-lowering substitution gives a negative f.

Both normalisation anchors are measured rather than assumed: 521 of the 522 wild-type rows read exactly 0.0 and the largest absolute wild-type value in the file is 0.216, and the per-domain median nonsense value has median −1.024 with quartiles −1.267 and −0.902 over 522 domains. There is no second measured state — the wild-type reference is identically zero by the source's construction — which is what makes this endpoint a **one-estimate level** rather than the two-estimate difference the stability gate targets, and it is why the nonlinear-additive response is applied to the predicted effect itself here rather than to two measured states.

The admission rule is a detected single amino-acid substitution of a domain whose wild type is known, with a finite `normalized_fitness` and a finite `normalized_fitness_sigma`. Nonsense rows are a separate class and are counted, never coerced onto the missense endpoint. Every accepted row's position, wild-type residue, mutant residue and mutant sequence are re-derived against the domain's own wild type, so a row whose coordinates disagree with its own sequence is refused rather than scored.

| Row class, of 602,882 in the pinned file | Count |
| --- | ---: |
| Wild-type rows | 522 |
| Undetected rows carrying no counts at all | 27,624 |
| Nonsense rows | 28,721 |
| Missense rows | 546,537 |
| Missense rows without a finite endpoint | 9,851 |
| Missense rows of a domain without a wild type | 0 |
| **Accepted rows** | **536,164** |

The undetected rows are a library-coverage absence rather than a censored measurement, and the difference matters for what the support represents: this assay reports a low abundance as a low fitness, so dropping rows with no counts does not select against destabilising substitutions the way a stability assay's resolvability floor does.

Source: Zenodo DOI 10.5281/zenodo.14356805, `Supplementary_Table_2_fitness_scores_normalized_domainranks.txt.zip`, sha256 `071ce5fa…f0150a79d`, CC-BY-4.0. Endpoint digest over the declared cohort `51528adbffa0d87976d6e3c484c1f761db30ffd6790d2d91a80a993117009a7e`.

## The 28 domains that make this external, and how they were found

No identifier in either development source names Beltran or the domainome. An identifier-only screen would therefore have reported this endpoint disjoint from both and left every overlapping domain in the confirmation support. Matching by **sequence**, exactly or by containment in either direction, finds 28 of the 522: **17 match a ProteinGym target, 17 match a Tsuboyama background, and 6 match both**, which is why the union is 28 rather than 34.

Those 28 are excluded by name before anything else happens, so they are not drawn, not scored and cannot reach any number reported here. The endpoint is then qualified on **its own support**, the 494 remaining domains, and that support reproduces its declared accounting exactly: **494 domains, 413 UniProt accessions, 120 Pfam families and 541,202 substitution rows**. The entry point exits rather than proceed if any of those four counts moves, so a change in the exclusion set or in the parse fails at that step rather than moving a number in this document.

A declared alignment screen then runs on top of the exclusion, and it is a second and stricter rule rather than a repeat of the first. A retained wild type is refused if it reaches 50% identity over 80% of its own length against any of the 217 ProteinGym targets or any of the 478 Tsuboyama backgrounds — the readout cohort's own wild-type-family rule, so the screen removes a domain that would sit in a development target's family under the development panel's own definition. **63 of the 494 cross it** under `diamond blastp --very-sensitive --masking 0` (1,850 reported alignments over 494 queries against 695 subjects, 180 queries aligning at all), and an exact Smith-Waterman screen against the short Tsuboyama backgrounds finds 51 crossing the same rule. Their union leaves 428. At the frozen grouping contract's weaker 30% edge, 140 of the 494 would cross; that is reported as a declared stratum and is not applied. The largest identity over the query of any reported hit is 97.5%.

## The floor, on two scales

This endpoint has real replicate structure, and it is reported on the two scales it is measured on. Neither is combined with the other and no oracle ceiling is constructed from either.

The **endpoint-scale floor** is the source's own reported `normalized_fitness_sigma`, in dimensionless normalised-fitness units. On the clean 494-domain support it has median **0.0898**, quartiles 0.0585 and 0.1437, 95th percentile 0.3451 and maximum 3.107 over 504,819 variants. Over the 395,089 replicate-complete substitution rows of the 428 retained domains it is tighter: median 0.0818, quartiles 0.0549 and 0.1211, 95th percentile 0.2151.

The **replicate floor** is the source's own three biological replicates, in log2-enrichment units, which is the scale the replicates are measured on and not the scale of the endpoint. The between-replicate standard deviation over all three has median **0.657**, quartiles 0.389 and 1.062 and 95th percentile 1.953. Replicate 1 against replicate 2 is decomposed the way the stability gate decomposes its two protease channels — shared-component standard deviation as an **upper** bound on reproducible signal, because error common to both replicates contributes to it, and per-replicate discordance standard deviation as a **lower** bound on per-replicate noise, because that common error cancels. Weighting is nested and intervals are 2,000-draw group-bootstrap percentile intervals at seed 20260923.

| Quantity, log2 enrichment unless dimensionless | Clean support, domain as unit | Declared cohort, family group as unit |
| --- | --- | --- |
| Units, variants | 494, 457,346 | 96, 395,089 |
| Pearson r | +0.8168 [+0.8064, +0.8267] | +0.8212 [+0.8012, +0.8390] |
| Replicate offset, 1 − 2 | +0.0971 [+0.0547, +0.1420] | **+0.0159 [−0.0710, +0.1087]** |
| Root-mean-square replicate difference | 1.4127 [1.3832, 1.4414] | 1.4182 [1.3575, 1.4738] |
| Shared-component standard deviation | 2.1043 [2.0431, 2.1650] | 2.1488 [2.0302, 2.2597] |
| **Per-replicate discordance standard deviation** | **0.9965 [0.9759, 1.0167]** | **1.0028 [0.9591, 1.0418]** |
| Shared-to-discordance ratio | 2.1116 [2.0410, 2.1844] | 2.1429 [2.0065, 2.2828] |

**The endpoint is admitted** on this support: replicate agreement is bounded far from zero at both units, the reproducible component exceeds the per-replicate discordance by a factor of 2.14, and 96 family groups support held-group folds. Two features of the table are worth naming, and the first needs its unit stated to be read at all. The replicate offset on the declared cohort is **+0.0159 [−0.0710, +0.1087] log2 enrichment**, indistinguishable from zero: **the two replicates are interchangeable in level**, so nothing has to be done about an offset here, where the [remote-homology gate](D1_GATE_REMOTE_HOMOLOGY.md) has a systematic channel offset in a free-energy unit that it has to show cancels. Those two offsets are quantities in different units on different sources and are not compared as numbers anywhere in this record; what is compared is whether each vanishes.

The second is the only cross-endpoint comparison this section makes, and it is dimensionless by construction. The shared-to-discordance **ratio** of 2.14 [2.01, 2.28] is lower than the admitted MegaScale endpoint's 3.86 [3.61, 4.13] and higher than the remote-homology endpoint's 1.566 [1.301, 1.847], so this endpoint sits between them in headroom. Even that comparison is loose rather than exact: the replicate ratio is a property of raw count enrichment, while the endpoint is a per-domain normalisation of a fitted quantity, so the two are not one number on two scales and the relation between them is the source's own normalisation, which is not reconstructed here.

## The declared support

Fixed before any predictor is fitted and before any model is scored, by four rules that read no measurement. Development overlap is removed by sequence; the alignment screen above is applied; family grouping unions exact sequence identity, shared UniProt accession, shared Pfam accession and the frozen contract's own 30%/80% alignment edge; and at most **256** substitutions per domain are drawn by a stable SHA-256 of the draw seed 20260924, the domain identifier and the mutant sequence, over domains carrying at least 64 accepted rows.

Two of those union sources are declared here rather than taken from the frozen grouping contract, because the contract cannot express them for this dataset: two domains cut from one UniProt accession are one group, and two domains the source library annotates with one Pfam accession are one group. Every domain carries exactly one Pfam label, so unlike the contract's hmmscan annotation there is no unlabelled stratum — the contract had no Pfam label for 179 of its 478 backgrounds. The alignment edge rule, its scoring and its thresholds are the contract as delivered.

| Frozen workload | Value |
| --- | --- |
| Domains in the source | 522 |
| After the sequence-overlap exclusion | 494 |
| After the alignment screen | 428 |
| Domains below the 64-row minimum | 0 |
| **Domains retained** | **428** |
| Family groups | 96, 53 of them singletons, largest 49 |
| UniProt accessions, Pfam families retained | 350, 115 |
| Eligible substitutions | 434,486 |
| Variants drawn | 109,568, the cap binding on every domain |
| Mutated sites | 23,569 |
| Sites per domain | 18–93, median 56 |
| Variants per site | 1–18, median 4 |
| Sequence length range | 19–97 residues |
| Sequences per arm | 109,996 |
| Residues per arm | 6,289,818 |
| Kish effective family groups | 96.0 |
| Kish effective domains | 140.7 |
| Kish effective sites | 7,600.7 |

The independent unit is the **family group**, and there are 96 of them with exactly equal weight by construction. Every effective count in that table is the Kish count of the weights the estimator applies, at its own level, over the 109,568 drawn variants; the replicate-floor table above reports 7,785.3 effective sites instead because it is computed over a different row set, the 395,089 replicate-complete rows, and the convention travels with each number rather than between them. Inside a group each domain carries equal weight, inside a domain each mutated site, and inside a site each variant. That three-level nesting is the one substitution this step makes to the development fit's two-level weighting, which assumed one background per group: this support carries several domains per group — 428 domains over 96 groups — so a two-level weighting would weight a domain by its site count and let a long domain dominate its family. Every count reported below is a domain, family-group or effective count, never a variant count.

There is one assay here, abundance PCA in yeast, so blocking the designs by assay is degenerate: there is one block and nothing to block on.

Cohort digest `fca2a53a03e4b7cc8b69a2f2348848d59f3ee073e1c70ee0e4599a854fc474d9`; label-free extraction plan content digest `805adabe10200fdfb5992796c40d78c328a4f510da80ce56f0f663ee923578aa`; support declaration digest `5b79f1cb1bb98623c696cdbac9d3c3b6d634a5f474971b1c327a5aa02640cae9`.

## Retrieval identity: a measurement that there is no remote stratum, and what it makes the two gates to each other

The [remote-homology gate](D1_GATE_REMOTE_HOMOLOGY.md) reads its model quantity out separately on close-identity and remote-identity family groups, and gates the reading on the close stratum firing first. Whether that frame applies here is a measurement and was made rather than assumed: the same stratification was computed on this cohort's 428 retained wild types, from the same UniRef50 search the profiles were built from, banded by the same frozen declaration -- identity of the best hit as `100 * nident / qlen`.

| Retrieval band of the best hit | Domains, of 428 |
| --- | ---: |
| ≥95% identity, near duplicate | 426 |
| 70–95%, close homology | 1 |
| 30–70%, remote homology | 0 |
| <30% or no reported hit | 1 |

**95 of the 96 family groups are entirely close, 1 is mixed, and none is remote.** The three-outcome frame therefore does not apply to this gate, and the reason is the measurement above rather than a judgement that a remote stratum would be too thin: **there is no remote stratum**, so a stratified readout here would compare 95 groups against zero rather than against a few. That is a property of the material — these are human protein domains and UniRef50 retrieves a near-duplicate of essentially every one of them.

**It is also what fixes the relation between the two gates this cycle added.** They are complementary by construction and neither is a weaker version of the other. This gate supplies a third source, a third assay and a different measured quantity, with **no remoteness**: its confirmation holds where a corpus retrieves a near relative of essentially every target. The remote-homology gate supplies **remoteness without independence**: 100 entirely remote family groups against the existing cohorts' 1, 2 and 12, on an assay technology and with an author shared with a development source, so it can populate the retrieval axis and cannot confirm anything. Reading either as a substitute for the other misreads both.

What gates the reading here is the general form of the same precondition: the qualified control set must be a competent held-out predictor before an increment over it means anything. It is, by the widest margin of the three sibling gates, and the next section reports it with its own interval.

## Control and baseline qualification

A control may enter this comparison only if it is a competent held-out predictor in its own right, under the same rule the stability gate declared: a candidate is offered, in the declared order, to the standing set that has already qualified, and it is kept only if its paired reduction in group-equal held-out mean squared error over that standing set is positive at every one of the three prespecified split seeds 20260923, 20260924 and 20260925. Every feature block is imported from the stability gate rather than reimplemented, so the control side of this confirmation is the same code the development measurement fitted.

The base is the directed substitution identity (400 coordinates) and the mutated position's geometry (6). Profiles come from one DIAMOND search of both nested gates' wild types against UniRef50 at `--very-sensitive --masking 0`: **427 of the 428 domains carry a profile and 1 carries the declared absence** -- zeros and a zero availability indicator, never an imputed profile.

Increments are paired reductions in group-equal held-out mean squared error in squared normalised fitness over 96 family groups, with 2,000-draw group-bootstrap intervals; the rank column is the paired within-domain Spearman increment over the same standing set.

| Candidate | Offered over | Increment, 20260923 / 20260924 / 20260925 | Own contribution at 20260923 | Rank increment | Verdict |
| --- | --- | --- | --- | --- | --- |
| Composition (40) | base | +0.00104 / +0.00111 / +0.00106 | +0.00104 [-0.00048, +0.00253] | +0.00128 / +0.00154 / +0.00095 | **kept** |
| Local chemistry (52) | base + composition | +0.00095 / +0.00103 / +0.00101 | +0.00095 [+0.00049, +0.00140] | +0.00988 / +0.00970 / +0.01042 | **kept** |
| Mutation-local profile (10) | base + composition + chemistry | +0.00892 / +0.00934 / +0.00978 | +0.00892 [+0.00639, +0.01132] | +0.11338 / +0.11317 / +0.11477 | **kept** |
| Bounded profile restatement (10) | + the raw profile | +0.00067 / +0.00039 / +0.00027 | +0.00067 [+0.00015, +0.00131] | +0.00411 / +0.00386 / +0.00404 | **kept** |
| Nonlinear-additive response (2) | + both profiles | +0.00119 / +0.00150 / +0.00090 | +0.00119 [+0.00030, +0.00201] | +0.00605 / +0.00441 / +0.00353 | **kept** |

**Every one of the five candidates qualified, and nothing was discarded.** The qualified control set is therefore the full ladder -- identity, geometry, composition, chemistry, both profile blocks and the nonlinear-additive response, **520 coordinates** -- and the secondary control set the correlation rule would accept is identical to it, so the two licensed cells coincide here and the crossed sensitivities the stability gate reports do not arise.

On that set the group-equal held-out mean squared error is **0.09104 , 0.09053 , 0.09125** against a no-effect null of 0.19201, so the qualified controls remove **52.5% to 52.9%** of the null's squared error. Read as its own contribution rather than as a ladder step, the qualified set reduces the null's squared error by **+0.10096 [+0.09227, +0.10928]** at seed 20260923 over 96 resampled groups, and its within-domain Spearman is **+0.47684 [+0.45266, +0.50119]**. The controls these increments will be read over are a competent held-out predictor of this endpoint by a wide margin, and by a wider one than either sibling gate's: this set removes over half the null where the remote-homology gate's removes 36.4% to 36.8%.

**Each candidate's own contribution carries an interval, and one kept block's contains zero.** Composition is kept by the declared three-seed sign rule -- +0.00104, +0.00111 and +0.00106 -- while its own first-seed interval is [-0.00048, +0.00253] and spans zero; the other four resolve above zero. That distinction is carried rather than smoothed over: a control whose own contribution is indistinguishable from zero makes any increment measured over it uninformative, which the catalogue records as L48 with four instances. What licenses the comparisons is the qualified set as a whole, whose contribution is resolved far from zero.

Two of the ladder's verdicts differ from the development gate's on the same blocks, and both differences are endpoint properties rather than disagreements. The **mutation-local profile is this endpoint's strongest control** at +0.00892 [+0.00639, +0.01132] with a rank increment of +0.113 at every seed, where on the folding-and-stability gate the same block was discarded for a catastrophic squared-error failure at one seed of three: it transferred rank and not level there, and it transfers both here. And **composition qualifies here where it failed there**, because that endpoint is a within-background difference in which the composition change is already determined by the substitution identity, while this one is a level with no second state, so a background-level composition term carries information the identity block does not.

## The likelihood-level result, over all 33 arms

Every arm of the frozen roster was extracted and fitted: **33 of 33, with 0 failures and 0 no-record cells**. The independent unit is the **family group** and there are **96** of them, at a Kish effective 96.0 groups, 140.7 domains and 7,600.7 sites under the weighting the estimator applies -- groups equal, domains equal inside a group, sites equal inside a domain, variants equal inside a site. Every interval below is a 2,000-draw group-bootstrap percentile interval resampling those 96 family groups at seed 20260923; no count below is a variant count.

Run-to-run reproducibility of the identical single-row computation is **1.53 x 10^-5 nats** on the likelihood and **8.67 x 10^-7** relative L2 on the feature block at the worst of 33 arms and 428 domains; at batch size one that is what the repeat column measures and the 0.001 gates remain structurally vacuous. Permuting the **30,976** held-out target rows of the first fold moves the held-out prediction of every one of the 12 declared designs by **0.0**, so no held-out label reaches feature scaling, penalty selection or the first-stage response.

**Tokenisation descriptors qualified in 23 of 33 arms**, which is the one place this gate behaves unlike either sibling: on the folding-and-stability gate and on the remote-homology gate none of the 33 qualified, so every arm there was read over one shared baseline. Here 23 arms are read over a matched baseline that already carries their own segmentation descriptors, and the baseline's group-equal held-out mean squared error consequently varies between arms -- from 0.09039 to 0.09125 squared normalised fitness over the three seeds, a between-arm spread of **0.00086** against a no-effect null of 0.19201. Both arms that resolve below are among the 23, so their increments are read over a baseline that already accounts for how they segment the sequence.

### The likelihood increment, the licensed cell

Paired reduction in group-equal held-out mean squared error in squared normalised fitness, model likelihood added to the qualified control set, over 96 family groups at each of the three prespecified split seeds.

| Arm | Interface | Tokenisation qualified | Seed 20260923 | Seed 20260924 | Seed 20260925 | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| bygpt5-base-en | byte | yes | -0.00000 [-0.00004, +0.00003] | +0.00000 [-0.00003, +0.00003] | -0.00003 [-0.00006, -0.00000] | unresolved |
| bygpt5-medium-en | byte | yes | -0.00005 [-0.00011, -0.00001] | -0.00003 [-0.00006, -0.00000] | -0.00002 [-0.00006, +0.00000] | unresolved |
| bygpt5-small-en | byte | yes | -0.00000 [-0.00001, +0.00001] | -0.00001 [-0.00002, +0.00000] | -0.00002 [-0.00003, -0.00000] | unresolved |
| dialogpt-small | BPE | yes | -0.00002 [-0.00005, +0.00001] | -0.00000 [-0.00002, +0.00002] | -0.00003 [-0.00007, -0.00001] | unresolved |
| galactica-1.3b | BPE | yes | -0.00001 [-0.00012, +0.00006] | -0.00006 [-0.00022, +0.00005] | -0.00006 [-0.00012, -0.00000] | unresolved |
| galactica-125m | BPE | yes | +0.00006 [-0.00002, +0.00014] | +0.00003 [-0.00005, +0.00011] | +0.00001 [-0.00007, +0.00010] | unresolved |
| galactica-30b | BPE | yes | +0.00009 [-0.00007, +0.00026] | +0.00016 [-0.00006, +0.00041] | +0.00005 [-0.00010, +0.00021] | unresolved |
| galactica-6.7b | BPE | yes | +0.00005 [-0.00007, +0.00014] | +0.00009 [-0.00019, +0.00039] | -0.00005 [-0.00019, +0.00008] | unresolved |
| gpt2 | BPE | yes | +0.00000 [-0.00003, +0.00004] | -0.00001 [-0.00005, +0.00002] | -0.00001 [-0.00004, +0.00003] | unresolved |
| gpt2-large | BPE | yes | -0.00001 [-0.00002, +0.00000] | -0.00000 [-0.00001, +0.00001] | -0.00001 [-0.00002, +0.00000] | unresolved |
| gpt2-medium | BPE | yes | -0.00002 [-0.00009, +0.00005] | +0.00001 [-0.00006, +0.00007] | +0.00001 [-0.00005, +0.00007] | unresolved |
| gpt2-xl | BPE | yes | -0.00001 [-0.00002, +0.00000] | -0.00001 [-0.00003, +0.00001] | -0.00001 [-0.00002, +0.00001] | unresolved |
| instructprotein | BPE | yes | -0.00005 [-0.00030, +0.00018] | +0.00003 [-0.00026, +0.00031] | -0.00002 [-0.00029, +0.00025] | unresolved |
| llama-2-7b | BPE | no | -0.00001 [-0.00002, +0.00000] | -0.00001 [-0.00002, +0.00000] | -0.00002 [-0.00003, -0.00001] | unresolved |
| llama-3.2-3b | BPE | no | -0.00001 [-0.00003, +0.00000] | -0.00002 [-0.00004, -0.00000] | -0.00002 [-0.00004, -0.00000] | unresolved |
| progen2-base | residue | yes | +0.00058 [-0.00011, +0.00123] | +0.00058 [+0.00002, +0.00111] | +0.00069 [+0.00007, +0.00130] | unresolved |
| progen2-large | residue | yes | +0.00055 [-0.00002, +0.00113] | +0.00050 [+0.00003, +0.00099] | +0.00047 [-0.00009, +0.00104] | unresolved |
| progen2-medium | residue | yes | +0.00050 [-0.00010, +0.00112] | +0.00049 [-0.00002, +0.00100] | +0.00050 [-0.00010, +0.00109] | unresolved |
| progen2-small | residue | yes | +0.00007 [-0.00029, +0.00045] | +0.00020 [-0.00011, +0.00053] | +0.00012 [-0.00025, +0.00051] | unresolved |
| progen2-xlarge | residue | yes | +0.00163 [+0.00099, +0.00223] | +0.00145 [+0.00083, +0.00206] | +0.00154 [+0.00093, +0.00209] | **above zero** |
| progen3-112m | residue | yes | +0.00033 [-0.00020, +0.00079] | +0.00020 [-0.00038, +0.00070] | +0.00026 [-0.00019, +0.00068] | unresolved |
| progen3-3b | residue | yes | +0.00049 [-0.00004, +0.00102] | +0.00038 [-0.00013, +0.00088] | +0.00046 [-0.00011, +0.00104] | unresolved |
| prollama | residue | no | +0.00005 [-0.00046, +0.00046] | -0.00006 [-0.00062, +0.00045] | +0.00011 [-0.00042, +0.00054] | unresolved |
| prollama-stage-1 | residue | no | +0.00029 [-0.00013, +0.00071] | +0.00004 [-0.00065, +0.00065] | +0.00013 [-0.00032, +0.00058] | unresolved |
| proteinglm-7b-clm | residue | yes | +0.00143 [+0.00080, +0.00202] | +0.00100 [+0.00037, +0.00160] | +0.00114 [+0.00049, +0.00177] | **above zero** |
| protgpt2 | BPE | no | +0.00009 [-0.00042, +0.00055] | +0.00014 [-0.00030, +0.00054] | +0.00018 [-0.00024, +0.00058] | unresolved |
| protgpt3-1.3b | BPE | yes | -0.00010 [-0.00044, +0.00023] | +0.00008 [-0.00023, +0.00039] | +0.00001 [-0.00028, +0.00031] | unresolved |
| qwen2.5-0.5b | BPE | no | -0.00002 [-0.00005, +0.00000] | -0.00002 [-0.00006, +0.00000] | -0.00001 [-0.00003, +0.00000] | unresolved |
| qwen2.5-0.5b-instruct | BPE | no | -0.00002 [-0.00007, +0.00001] | -0.00001 [-0.00004, +0.00003] | +0.00000 [-0.00003, +0.00003] | unresolved |
| qwen2.5-32b | BPE | no | +0.00006 [-0.00008, +0.00021] | +0.00015 [-0.00001, +0.00039] | +0.00011 [-0.00008, +0.00037] | unresolved |
| qwen2.5-7b | BPE | no | +0.00006 [-0.00004, +0.00018] | +0.00009 [-0.00002, +0.00023] | +0.00006 [-0.00006, +0.00021] | unresolved |
| qwen3-8b-base | BPE | no | +0.00002 [-0.00003, +0.00007] | +0.00000 [-0.00004, +0.00004] | -0.00003 [-0.00008, +0.00001] | unresolved |
| rita-xl | residue | yes | +0.00026 [-0.00017, +0.00068] | +0.00025 [-0.00017, +0.00067] | +0.00031 [-0.00013, +0.00077] | unresolved |

Panel median of the three-seed means **+0.000029**. By interface: residue 11 arms at a median of +0.000444 (11 positive, 2 above zero, 0 below), BPE 19 arms at a median of -0.000005 (7 positive, 0 above zero, 0 below), byte 3 arms at a median of -0.000011 (0 positive, 0 above zero, 0 below).

**Two arms resolve above zero and none below: ProGen2-xlarge and ProteinGLM-7B-CLM, both residue-level protein-pretrained checkpoints.** ProteinGLM-7B-CLM is one of the four the [folding-and-stability gate](D1_GATE_STABILITY.md) resolved on proteolysis-derived stability, so one arm resolves on both a development endpoint and this one.

**The interface pattern is the part that does not rest on one arm's interval, and it is the confirmation.** All **11 of 11** residue-level arms have a positive three-seed mean, at a stratum median of **+0.000444**; 7 of the 19 BPE arms are positive at a median of -0.000005, and 0 of the 3 byte-level arms at -0.000011. That is the ordering the development gate found on a different source, a different assay and a different measured quantity, with the residue stratum an order of magnitude above the other two.

### The likelihood rank increment, and where the confirmation is strongest

Paired within-domain Spearman increment over the qualified control set. Points only; the intervals are in the artefact.

| Arm | Interface | Tokenisation qualified | Seed 20260923 | Seed 20260924 | Seed 20260925 | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| bygpt5-base-en | byte | yes | -0.00002 | +0.00012 | +0.00004 | unresolved |
| bygpt5-medium-en | byte | yes | -0.00024 | -0.00020 | -0.00015 | **below zero** |
| bygpt5-small-en | byte | yes | -0.00000 | -0.00016 | -0.00012 | unresolved |
| dialogpt-small | BPE | yes | -0.00015 | +0.00003 | -0.00006 | unresolved |
| galactica-1.3b | BPE | yes | -0.00002 | +0.00003 | +0.00002 | unresolved |
| galactica-125m | BPE | yes | +0.00031 | +0.00006 | +0.00022 | unresolved |
| galactica-30b | BPE | yes | +0.00012 | +0.00012 | +0.00021 | unresolved |
| galactica-6.7b | BPE | yes | +0.00018 | +0.00033 | +0.00031 | unresolved |
| gpt2 | BPE | yes | -0.00000 | -0.00011 | -0.00005 | unresolved |
| gpt2-large | BPE | yes | -0.00006 | -0.00002 | -0.00003 | unresolved |
| gpt2-medium | BPE | yes | -0.00014 | -0.00018 | -0.00006 | unresolved |
| gpt2-xl | BPE | yes | -0.00007 | -0.00013 | -0.00007 | unresolved |
| instructprotein | BPE | yes | +0.00095 | +0.00134 | +0.00127 | **above zero** |
| llama-2-7b | BPE | no | +0.00001 | +0.00003 | -0.00015 | unresolved |
| llama-3.2-3b | BPE | no | -0.00003 | -0.00007 | -0.00012 | unresolved |
| progen2-base | residue | yes | +0.00532 | +0.00585 | +0.00513 | **above zero** |
| progen2-large | residue | yes | +0.00368 | +0.00395 | +0.00386 | **above zero** |
| progen2-medium | residue | yes | +0.00522 | +0.00588 | +0.00556 | **above zero** |
| progen2-small | residue | yes | +0.00199 | +0.00218 | +0.00195 | **above zero** |
| progen2-xlarge | residue | yes | +0.00762 | +0.00797 | +0.00721 | **above zero** |
| progen3-112m | residue | yes | +0.00525 | +0.00559 | +0.00559 | **above zero** |
| progen3-3b | residue | yes | +0.00396 | +0.00435 | +0.00428 | **above zero** |
| prollama | residue | no | +0.00305 | +0.00300 | +0.00331 | **above zero** |
| prollama-stage-1 | residue | no | +0.00466 | +0.00466 | +0.00507 | **above zero** |
| proteinglm-7b-clm | residue | yes | +0.00699 | +0.00745 | +0.00639 | **above zero** |
| protgpt2 | BPE | no | +0.00239 | +0.00232 | +0.00220 | **above zero** |
| protgpt3-1.3b | BPE | yes | +0.00164 | +0.00185 | +0.00189 | **above zero** |
| qwen2.5-0.5b | BPE | no | -0.00014 | +0.00003 | -0.00014 | unresolved |
| qwen2.5-0.5b-instruct | BPE | no | -0.00007 | +0.00007 | +0.00009 | unresolved |
| qwen2.5-32b | BPE | no | +0.00027 | +0.00039 | +0.00089 | unresolved |
| qwen2.5-7b | BPE | no | +0.00023 | +0.00027 | +0.00050 | unresolved |
| qwen3-8b-base | BPE | no | +0.00014 | +0.00005 | +0.00010 | unresolved |
| rita-xl | residue | yes | +0.00322 | +0.00393 | +0.00346 | **above zero** |

Panel median **+0.000272**. By interface: residue 11 arms at a median of +0.004798 (11 positive, 11 above zero, 0 below), BPE 19 arms at a median of +0.000029 (11 positive, 3 above zero, 0 below), byte 3 arms at a median of -0.000092 (1 positive, 0 above zero, 1 below).

**On rank the confirmation is unambiguous: 14 of 33 arms resolve above zero, and every one of the 11 residue-level arms is among them.** The other three are the protein-corpus BPE arms -- InstructProtein, ProtGPT2 and ProtGPT3-1.3B -- so the 14 are exactly the arms pretrained on protein sequence, whatever their interface, and the 19 remaining are exactly the text checkpoints. Residue median +0.004798 against BPE +0.000029 and byte -0.000092. Because the qualified set and the correlation rule's secondary set are identical here, the same 14 arms resolve on the secondary cell and the two licensed cells agree rather than needing to be read separately.

### Verdict

**The first-order likelihood finding is confirmed on data used in neither development panel, and on a different measured quantity.** A frozen protein-pretrained likelihood carries information about how a single substitution changes cellular abundance beyond a control set that already removes 52.5% to 52.9% of the no-effect null's squared error: two arms resolve on the squared-error cell, 14 on the rank cell, and the residue-against-text interface ordering reproduces at every level -- 11 of 11 residue arms positive on squared error against 0 of 3 byte arms, and 11 of 11 resolved on rank.

**What that does not license.** Abundance by complementation in yeast is a different measured quantity from proteolysis-derived stability and from ProteinGym's heterogeneous assays, so this is cross-quantity corroboration and not replication of either estimand; agreement across two measured quantities is stronger evidence than agreement within one, and it is not a repeat measurement. The resolved increments are 1.1% to 1.8% of the baseline's remaining squared error, so the quantity is resolvable and small, exactly as it was on the development endpoint.

**Which of the declared outcomes obtains.** The remote-homology gate's three outcomes are conditioned on a close-identity positive control, and they do not apply here: as measured above, this support has no remote stratum -- 95 of its 96 family groups are entirely close and none is remote. The readability precondition in its general form is what gates this gate, and it fired: the qualified control set is a competent held-out predictor with its own contribution resolved at 0.10096 [0.09227, 0.10928] squared normalised fitness. So the increments above are read, rather than being an unresolved statement about the support, which is precisely what the remote-homology result was.

### The representation increment -- provisional, and subject to recomputation

**This table is provisional and carries no verdict.** Read it as a measurement of one readout at two depths, not as a statement about what these representations contain. It is conditional on one compressed linear readout of four pooled block summaries at **the two depths this extraction hooked and no others** -- the middle block and the final block, which is all the entry point computes. The readout-class reassessment has published its selection, class `C4_full_random_feature` with per-(arm, panel) depths, but the gate-level recomputation has not run.

So the reading below -- **0 of 33 arms above zero at a panel median of −0.004582** -- is **not a verdict, and specifically not a not-detected verdict**. It is what one linear readout of two pooled depths yields, on a gate whose own likelihood cell resolves two arms and 14 rank cells at the same depths, and it cannot presently be distinguished from a measurement-limited reading: L53 records exactly that hazard, that a gate fitted at the admitted depth produces a well-formed increment with no indication another depth exists.

The full-width per-state block outputs are retained, **126.24 GiB over 14,124 archives**, so a readout-class refit at the hooked depths needs no forward pass. A depth outside those two needs re-extraction, and **that every-block pass is now authorised and queued behind three running waves**, at roughly 84 calibrated GPU-hours and 2,034 GiB; the likelihood result above is what decided it. The representation reading will be recomputed from the retained states and from that pass before any verdict is written.

| Arm | Interface | Seed 20260923 | Seed 20260924 | Seed 20260925 |
| --- | --- | --- | --- | --- |
| bygpt5-base-en | byte | -0.00483 | -0.00450 | -0.00433 |
| bygpt5-medium-en | byte | -0.00364 | -0.00342 | -0.00370 |
| bygpt5-small-en | byte | -0.00376 | -0.00343 | -0.00320 |
| dialogpt-small | BPE | -0.00592 | -0.00568 | -0.00493 |
| galactica-1.3b | BPE | -0.00449 | -0.00428 | -0.00447 |
| galactica-125m | BPE | -0.00442 | -0.00495 | -0.00450 |
| galactica-30b | BPE | -0.00475 | -0.00488 | -0.00473 |
| galactica-6.7b | BPE | -0.00419 | -0.00388 | -0.00419 |
| gpt2 | BPE | -0.00474 | -0.00466 | -0.00484 |
| gpt2-large | BPE | -0.00569 | -0.00510 | -0.00542 |
| gpt2-medium | BPE | -0.00486 | -0.00544 | -0.00477 |
| gpt2-xl | BPE | -0.00582 | -0.00501 | -0.00554 |
| instructprotein | BPE | -0.00570 | -0.00556 | -0.00563 |
| llama-2-7b | BPE | -0.00503 | -0.00560 | -0.00507 |
| llama-3.2-3b | BPE | -0.00483 | -0.00515 | -0.00499 |
| progen2-base | residue | -0.00036 | -0.00057 | -0.00275 |
| progen2-large | residue | +0.00105 | +0.00027 | +0.00205 |
| progen2-medium | residue | -0.00542 | -0.00393 | -0.00434 |
| progen2-small | residue | -0.00351 | -0.00347 | -0.00357 |
| progen2-xlarge | residue | -0.00108 | -0.00104 | -0.00227 |
| progen3-112m | residue | -0.00522 | -0.00430 | -0.00575 |
| progen3-3b | residue | -0.00303 | -0.00224 | -0.00080 |
| prollama | residue | -0.00463 | -0.00372 | -0.00393 |
| prollama-stage-1 | residue | -0.00454 | -0.00473 | -0.00493 |
| proteinglm-7b-clm | residue | +0.00448 | +0.00178 | +0.00372 |
| protgpt2 | BPE | -0.00387 | -0.00306 | -0.00370 |
| protgpt3-1.3b | BPE | -0.00302 | -0.00316 | -0.00303 |
| qwen2.5-0.5b | BPE | -0.00645 | -0.00496 | -0.00512 |
| qwen2.5-0.5b-instruct | BPE | -0.00612 | -0.00488 | -0.00527 |
| qwen2.5-32b | BPE | -0.00465 | -0.00443 | -0.00466 |
| qwen2.5-7b | BPE | -0.00469 | -0.00502 | -0.00454 |
| qwen3-8b-base | BPE | -0.00473 | -0.00524 | -0.00506 |
| rita-xl | residue | -0.00180 | -0.00219 | -0.00250 |

Panel median -0.004582. By interface: residue 11 arms at a median of -0.002164 (2 positive, 0 above zero, 0 below), BPE 19 arms at a median of -0.004987 (0 positive, 0 above zero, 0 below), byte 3 arms at a median of -0.003586 (0 positive, 0 above zero, 0 below).

## Artifacts, and the measurement identity they were produced under

Each arm contributes two quantities, added separately and never only jointly: the likelihood difference M_mut - M_WT in nats, one column, and the projected representation difference R_mut - R_WT over the four pooled blocks, 1,024 columns at 256 coordinates per block. Each was offered to the qualified control set, and each arm's own tokenisation descriptors were offered to that baseline first under the same qualification rule the controls faced.

The full frozen 33-arm roster was dispatched and **no arm was selected by any outcome**; the campaign builder exits rather than write a manifest set that does not cover the roster exactly, and lane order is a declared descending parameter scale. Measurement identity is the singleton one the panel's interface forces: batch size one for every arm, float32 for 31 arms and bfloat16 for the two ProGen3 arms, whose mixture-of-experts block has no kernel above float16 and whose expert mixture reduces over the flattened batch, so ProGen3 is scored as singletons for that reason as well. At batch size one the declared 0.001 precision gates are **structurally vacuous rather than passed**. Every extraction cell named the staged `ct-20260905` interpreter -- Python 3.11.14, numpy 1.26.4, torch 2.9.1+cu128, transformers 4.57.3 -- rather than the pod image's 3.12.7, and pinned the BLAS thread count to 4 in its own manifest environment column, because a float32 matmul's reduction order depends on it and the admitted projection ran at that value. Every fit ran under the same interpreter at the same pinned thread count.

Extraction run `20260924085613_5f8d32cf0368`, four lane groups over eleven cards, manifests `scripts/transfer/campaign_nested_ec_extract_w2{a,b,c,d}.tsv`, **33 of 33 cells `exited-ok`** with 0 failures and 0 no-record across all four status files. The fits ran in-pod on CPU under snapshot `20260924211833_ef121749ee3b`, which differs from the extraction snapshot only in the state-index derivation described below.

Frozen inputs are in `results/external_confirmation_20260924/`: `endpoint_qualification.json`, `cohort.json`, `extraction_plan.json`, `support_declaration.json`, `profile_features.npz` (sha256 `9582bdba...dca8024723`), `controls_qualification.json` (`a1ac4873...86ce475dc`), `retrieval_bands.json`, and the alignment screen's own query and subject FASTA files with the DIAMOND manifest under `screen/`. The 33 per-arm fit records are `fits/fit_<arm>.json`, each carrying its complete fold map, the selected ridge penalty of every design in every fold, the declared design names with their column dimensions and the first-stage diagnostics of the nonlinear-additive response; the panel is `panel/panel.json` with a digest of every fit it reads. Runtime logs stay under ignored `logs/d1_external_confirm_20260924/`.

**One defect found and fixed between extraction and fitting.** The per-arm loader compared each unit's extracted state indices against a `state` field, which the remote-homology cohort records and this one leaves implicit -- the extraction plan derives it, the cohort does not carry it -- so all 33 fits failed immediately on a missing key. The loader now derives the index from the variant's position in its unit's own declared order, which is exactly what the plan turned into `sequences = [wild type] + variant sequences`, so both cohort schemas are read without either being re-declared. Two things follow and are stated here so a later reader does not re-litigate them. **No completed extraction was invalidated**: the repair is on the reading side, the archives were not rewritten and neither cohort digest moved. And **the archive's own position check still binds the order** the derived index depends on -- every unit's extracted `variant_positions` must equal its cohort variants' positions in the same order, or the fit refuses -- so deriving the index from the order is not a weaker check than reading a recorded field, it is the same check with the field's redundancy removed. The repair carries a test.

## Limitations

**This is cross-quantity corroboration and not replication.** Abundance by complementation in yeast is a different measured quantity from proteolysis-derived stability and from ProteinGym's heterogeneous assays. The agreement reported above is stronger evidence than agreement within one quantity would be, and it is not a repeat measurement of either estimand; a disagreement would not by itself have been a failure to replicate.

**The screen is conservative and the exclusion is not.** The 28 sequence-overlapping domains are removed from the support entirely — not partitioned, not down-weighted — so nothing reported here can rest on them. The further 66 domains the alignment screen removes are removed on a 50%/80% rule against the development sources, which is stricter than the frozen contract's own 30% edge; at that weaker threshold 140 of the 494 would cross, and homology below it is undetected. The screen's negative statement rests on exhaustive query and subject sets — every retained wild type against every development target — which is what such a statement requires.

**The support represents resolvable, well-covered domains.** A variant with no detected counts is absent, the retained domains span 19 to 97 residues, and the cap binds on every one of the 428, so the panel is 256 of a median 1,015 eligible substitutions per domain drawn label-blind. Nothing here describes a domain the library did not cover.

**One endpoint feature limits what the nonlinear response can do.** There is no second measured state, so the isotonic response is applied to the predicted effect rather than to two measured levels; that substitution is declared and is the only one made to the development recipe's first stage.

**The replicate floor and the endpoint floor are on different scales** and are not combined. The replicate decomposition is in log2 enrichment and the reported uncertainty is in normalised fitness; the relation between them is the source's own per-domain normalisation of a fitted quantity and is not reconstructed.

**This gate says nothing about remote homology.** Its support is saturated at close retrieval identity -- 426 of 428 domains band as near duplicates and 95 of 96 family groups are entirely close -- so the confirmation holds where a corpus retrieves a near relative of essentially every target, and the [remote-homology gate](D1_GATE_REMOTE_HOMOLOGY.md) is where that axis is tested. That gate's own verdict on its own support is unresolved rather than null.

**The representation cells are depth-limited as well as readout-limited.** The extraction hooked the two blocks its entry point computes, the middle and the final, and no other. A readout-class refit at those two depths needs no forward pass, but the reassessment's per-arm depth selection lies outside them for some arms, and re-extracting this cohort at further depths is costed at roughly 84 calibrated GPU-hours and 2,034 GiB. That is limitation L53, and it binds this gate as it binds three others; the likelihood cells above are unaffected, because a likelihood enters as one scalar column with no depth to select.

**Multiplicity is declared and no interval is adjusted for it**, across 33 arms, three split seeds, two model quantities and two metrics; the control sets coincide here, so the licensed cells are 33 arms times two model quantities times two metrics. Requiring all three per-seed intervals to exclude zero is a conjunction over three correlated resamplings of one support, not three independent tests, so it carries no nominal level; it is a stability requirement and not a corrected test.
