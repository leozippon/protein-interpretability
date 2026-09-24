# Does the first-order likelihood finding hold on data used in neither development panel?

This is the external-confirmation step of the [Direction-1 capability map](D1_CAPABILITY_MAP_PLAN.md). The map's strongest reproducible model-side finding is that a residue-level protein-pretrained likelihood carries information about the effect of a single substitution beyond controls that are competent held-out predictors in their own right — four of 33 arms on the [folding-and-stability gate](D1_GATE_STABILITY.md), and a residue-against-text interface pattern that does not rest on any one arm's interval. Both development sources are shared with the readout panel: ProteinGym's 217 deep mutational scans, 64 of whose readout anchors are Tsuboyama 2023, and Tsuboyama 2023 itself. This step asks the same question on a third source with a third assay.

**What agreement here would and would not mean.** The endpoint is cellular abundance by protein-fragment complementation in *Saccharomyces cerevisiae*, which is a **different measured quantity** from proteolysis-derived stability and from ProteinGym's heterogeneous assays. Agreement across two different measured quantities is stronger evidence than agreement within one, and it is not a repeat measurement of either estimand, so a disagreement here is not by itself a failure to replicate. Every sentence below that reports agreement says which of the two it is.

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

**The endpoint is admitted** on this support: replicate agreement is bounded far from zero at both units, the reproducible component exceeds the per-replicate discordance by a factor of 2.14, and 96 family groups support held-group folds. Two features of the table are worth naming. The replicate offset on the declared cohort is **+0.0159 [−0.0710, +0.1087]**, indistinguishable from zero — the two replicates are interchangeable in level, unlike the two protease channels of the [remote-homology gate](D1_GATE_REMOTE_HOMOLOGY.md), whose offset of −0.42 kcal/mol makes them non-interchangeable rather than merely noisy. And the shared-to-discordance ratio of 2.14 is lower than the admitted MegaScale endpoint's 3.86 [3.61, 4.13] and higher than the remote-homology endpoint's 1.566 [1.301, 1.847], so this endpoint sits between them in headroom. The replicate ratio is a property of raw count enrichment, while the endpoint is a per-domain normalisation of a fitted quantity, so the two are not the same number on two scales and the relation between them is the source's own normalisation and is not reconstructed here.

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
| Kish effective sites | 7,785.3 |

The independent unit is the **family group**, and there are 96 of them with exactly equal weight by construction. Inside a group each domain carries equal weight, inside a domain each mutated site, and inside a site each variant. That three-level nesting is the one substitution this step makes to the development fit's two-level weighting, which assumed one background per group: this support carries several domains per group — 428 domains over 96 groups — so a two-level weighting would weight a domain by its site count and let a long domain dominate its family. Every count reported below is a domain, family-group or effective count, never a variant count.

There is one assay here, abundance PCA in yeast, so blocking the designs by assay is degenerate: there is one block and nothing to block on.

Cohort digest `fca2a53a03e4b7cc8b69a2f2348848d59f3ee073e1c70ee0e4599a854fc474d9`; label-free extraction plan content digest `805adabe10200fdfb5992796c40d78c328a4f510da80ce56f0f663ee923578aa`; support declaration digest `5b79f1cb1bb98623c696cdbac9d3c3b6d634a5f474971b1c327a5aa02640cae9`.

## Control and baseline qualification

A control may enter this comparison only if it is a competent held-out predictor in its own right, under the same rule the stability gate declared: a candidate is offered, in the declared order, to the standing set that has already qualified, and it is kept only if its paired reduction in group-equal held-out mean squared error over that standing set is positive at every one of the three prespecified split seeds 20260923, 20260924 and 20260925. Every feature block is imported from the stability gate rather than reimplemented, so the control side of this confirmation is the same code the development measurement fitted.

The base is the directed substitution identity (400 coordinates) and the mutated position's geometry (6). The candidates are composition (40), local chemistry in mutation-centred windows (52), the mutation-local profile (10), its bounded restatement (10), and the nonlinear-additive global response. Profiles come from one DIAMOND search of both nested gates' wild types against UniRef50 at `--very-sensitive --masking 0`; a domain that retrieved no qualifying homolog carries zeros and a zero availability indicator, never an imputed profile.

The ladder's verdicts, the qualified set's own contribution over the no-effect null and each candidate's own contribution with its interval are **pending**: the ladder is running on the 109,568-variant panel as this is written and the entry point has not yet written `controls_qualification.json`. The L48 rule it applies is declared: each candidate carries its own contribution with a group-bootstrap interval and the qualified set carries its own contribution over the no-effect null with one, because a control contributing indistinguishably from zero makes any increment measured over it uninformative, and a block kept by the three-seed sign rule whose own interval contains zero is reported as exactly that rather than as a competent control.

## What is staged, and what it is waiting for

Each arm contributes two quantities, added separately and never only jointly: the likelihood difference M_mut − M_WT in nats, one column, and the projected representation difference R_mut − R_WT over the four pooled blocks, 1,024 columns at 256 coordinates per block. Each is offered to the qualified control set, and each arm's own tokenisation descriptors are offered to that baseline first under the same qualification rule the controls faced, because a representation difference can be nonzero purely through segmentation.

The full frozen 33-arm roster is dispatched and **no arm is selected by any outcome**; the campaign builder exits rather than write a manifest set that does not cover the roster exactly, and the order arms reach a card is a declared descending parameter scale, not a result. Measurement identity is the singleton one the panel's interface forces: batch size one for every arm, float32 for 31 arms and bfloat16 for the two ProGen3 arms, whose mixture-of-experts block has no kernel above float16. At batch size one the declared 0.001 precision gates are **structurally vacuous rather than passed** — the repeat forward is the identical single-row computation, so its zero is structural, and the repeat column measures run-to-run reproducibility of that computation. ProGen3 is scored as singletons for the separate reason that its expert mixture reduces over the flattened batch. Every cell names the staged `ct-20260905` interpreter — Python 3.11.14, numpy 1.26.4, torch 2.9.1+cu128, transformers 4.57.3 — rather than the pod image's, and pins the BLAS thread count to 4, because a float32 matmul's reduction order depends on it and the admitted projection ran at that value. Every cell carries `--keep-full-features`, so the unprojected per-state block outputs are retained and a readout-class recomputation is a refit rather than a second forward pass.

**The likelihood cells are not deferred and will be reported as final.** The representation cells are computed and retained but carry **no verdict**: they are conditional on the compressed linear readout of four pooled block summaries at the retained depths, and the readout-class reassessment the capability map records is still running.

This cohort is 4.2 times the residue workload of the stability gate per arm — 6,289,818 residues against 1,501,137 — so its extraction is the long pole. Extraction run `20260924085613_5f8d32cf0368`, manifests `scripts/transfer/campaign_nested_ec_extract_w2{a,b,c,d}.tsv` over eleven cards in four lane groups, dispatched onto cards as the [remote-homology gate's](D1_GATE_REMOTE_HOMOLOGY.md) own waves free them. Frozen inputs are in `results/external_confirmation_20260924/`: `endpoint_qualification.json`, `cohort.json`, `extraction_plan.json`, `support_declaration.json`, `profile_features.npz`, and the alignment screen's own query and subject FASTA files with the DIAMOND manifest under `screen/`. Runtime logs stay under ignored `logs/d1_external_confirm_20260924/`.

## Limitations

**This is cross-quantity corroboration and not replication.** Abundance by complementation in yeast is a different measured quantity from proteolysis-derived stability and from ProteinGym's heterogeneous assays. Agreement would be stronger evidence than agreement within one quantity, and a disagreement would not by itself be a failure to replicate.

**The screen is conservative and the exclusion is not.** The 28 sequence-overlapping domains are removed from the support entirely — not partitioned, not down-weighted — so nothing reported here can rest on them. The further 66 domains the alignment screen removes are removed on a 50%/80% rule against the development sources, which is stricter than the frozen contract's own 30% edge; at that weaker threshold 140 of the 494 would cross, and homology below it is undetected. The screen's negative statement rests on exhaustive query and subject sets — every retained wild type against every development target — which is what such a statement requires.

**The support represents resolvable, well-covered domains.** A variant with no detected counts is absent, the retained domains span 19 to 97 residues, and the cap binds on every one of the 428, so the panel is 256 of a median 1,015 eligible substitutions per domain drawn label-blind. Nothing here describes a domain the library did not cover.

**One endpoint feature limits what the nonlinear response can do.** There is no second measured state, so the isotonic response is applied to the predicted effect rather than to two measured levels; that substitution is declared and is the only one made to the development recipe's first stage.

**The replicate floor and the endpoint floor are on different scales** and are not combined. The replicate decomposition is in log2 enrichment and the reported uncertainty is in normalised fitness; the relation between them is the source's own per-domain normalisation of a fitted quantity and is not reconstructed.

**Multiplicity will be declared and unadjusted** across 33 arms, three split seeds, two control sets, two model quantities and two metrics. Requiring all three per-seed intervals to exclude zero is a conjunction over three correlated resamplings of one support, not three independent tests, so it carries no nominal level; it is a stability requirement and not a corrected test.
