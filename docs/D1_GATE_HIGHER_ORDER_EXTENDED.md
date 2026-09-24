# Higher-order interaction on ProteinGym: the labels exist, the noise floor still decides

The [MegaScale higher-order gate](D1_GATE_HIGHER_ORDER.md) stopped because `dataset2` contains no variant with three substitutions relative to its own background wild type, leaving a cross-background route with two independent site pairs. This document asks the same question of data that does contain measured combinatorial states, under the permission to download additional datasets. It owns the ProteinGym order census, the raw-archive provenance, the qualified endpoint's support and its measurement-noise floor. The [protocol](D1_INFORMATION_HIERARCHY_PROTOCOL.md) owns the estimand discipline; the [label instrument](D1_PAIRWISE_LABEL_INSTRUMENT.md) owns the between-channel decomposition whose construction is reused here on a different scale.

The work is measurement-only and CPU-only. No model score, likelihood or representation enters any quantity below, and no GPU was scheduled.

## Decision

**The higher-order gate is revived as a measurement and remains unresolved for the model question.** Four findings carry it, and the fourth is the one that settles it.

Measured higher-order variants exist in quantity. Across the 217 local ProteinGym substitution assays, 943,211 of 2,465,767 variants carry three or more substitutions, and 206,390 of them have their complete lower-order support measured — 35,917 at order 3, 128,301 at order 4 and 42,172 at orders 5 through 9. That is the opposite of the MegaScale finding and it is what made a second attempt worth making.

The support and the noise channel sit in different assays. Complete cubes of order 3 or above appear in 9 of 217 assays. Of those, exactly one carries a genuine replicate channel structure: `PHOT_CHLRE_Chen_2023`, whose raw file holds three independent FACS replicates and the wild-type state. Its cube support rests on **20 measured substitutions at 15 positions in one protein**. The assays with the richest position support — `HIS7_YEAST_Pokusaeva_2019` at 89 positions and `CAPSD_AAV2S_Sinai_2021` at 26 — carry no per-variant uncertainty of any kind in either the processed or the raw file, so their contrasts cannot be qualified at all.

The endpoint does not clear its floor on either assay where a floor can be measured. On `PHOT_CHLRE`, the order-3 cycle's reproducible-across-replicate component has a standard deviation of 0.0071 [0.0000, 0.0246] log10 brightness against a per-channel discordance of 0.1219 [0.1149, 0.1284], a ratio of 0.064 [0.000, 0.214]. The admitted pairwise endpoint's ratio, for comparison, is 2.78 [2.34, 3.26].

And the same holds on the best-powered support in the census, after the channel for it was acquired. The `HIS7_YEAST_Pokusaeva_2019` source deposition supplies one fitness value per *nucleotide* genotype, so synonymous genotypes of one amino-acid state are a genuine replicate channel, and it covers all 496,137 of the benchmark's variants and measures the wild-type window in all twelve segment libraries. On 2,971 order-3 cubes resting on 598 site triples across **76 positions**, at 119.6 Kish effective site triples against the admitted pairwise cohort's 121.6, the ratio is 0.276 [0.000, 0.496]. Read as generously as the construction allows — against the pooled label's own noise rather than one channel's — it is 0.391 [0.000, 0.702], and at orders 4 and 5 the reproducible component's point estimate is 0.0000 with upper bounds of 0.378 and 0.000. **Any variation seen here is a measurement limitation, not a model finding**, and no model fit was run.

The recommendation is now to stop on ProteinGym. The expansion this document originally recommended was executed rather than left as a proposal: the channel was acquired for the richest available position support and the endpoint failed on it too, on a support whose effective unit count matches the pairwise endpoint's. The remaining option and its cost are at the end.

## The order census, over all 217 assays

The census reads only the `mutant` column, so the support it reports cannot be a function of the labels that support would carry. Every one of the 2,465,767 rows parses as a colon-joined list of canonical-residue substitutions at distinct positions, no assay repeats a variant, and **no assay carries a wild-type row** — the order-0 count is zero by construction of the benchmark, which is the structural difference from MegaScale that Section [Estimand](#the-estimand-and-what-restored-the-absolute-one) turns on.

| Order | Variants | Order | Variants |
| ---: | ---: | ---: | ---: |
| 1 | 696,311 | 6 | 130,395 |
| 2 | 826,245 | 7 | 149,318 |
| 3 | 84,134 | 8 | 134,593 |
| 4 | 187,850 | 9 | 90,108 |
| 5 | 92,213 | ≥10 | 74,600 |

The tail runs to order 44. Completeness is the requirement that decides which of these carry a cycle: every non-empty proper subset of a variant's substitutions must itself be a measured state, checked level by level from the singles upwards, with the enumeration stopped at the first incomplete level. The check is exact rather than heuristic — a level of `comb(order, size)` distinct subsets cannot be complete when the assay holds fewer states of that order in total — which is what keeps an order-44 variant from demanding 1.76e13 corners.

Eleven assays carry any variant of order 3 or above, and nine carry at least one complete cube:

| Assay | Rows | Max order | Order ≥3 | Complete cubes by order | Site triples | Triples with ≥10 cubes |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `SPG1_STRSG_Wu_2016` | 149,360 | 4 | 147,193 | 3: 25,035; 4: 109,235 | 4 | 4 |
| `HIS7_YEAST_Pokusaeva_2019` | 496,137 | 28 | 494,494 | 3: 4,852; 4: 10,728; 5: 13,809; 6: 10,118; 7: 3,884; 8: 682; 9: 66 | 1,083 | 78 |
| `PHOT_CHLRE_Chen_2023` | 167,529 | 15 | 165,231 | 3: 890; 4: 2,753; 5: 5,073; 6: 5,212; 7: 2,700; 8: 587; 9: 40 | 428 | 1 |
| `F7YBW8_MESOW_Aakre_2015` | 9,192 | 4 | 8,656 | 3: 2,798; 4: 5,465 | 4 | 4 |
| `CAPSD_AAV2S_Sinai_2021` | 42,328 | 28 | 30,963 | 3: 1,954; 4: 120; 5: 1 | 558 | 28 |
| `Q8WTC7_9CNID_Somermeyer_2022` | 33,510 | 43 | 21,049 | 3: 320 | 297 | 0 |
| `GFP_AEQVI_Sarkisyan_2016` | 51,714 | 15 | 37,853 | 3: 56 | 56 | 0 |
| `Q6WV12_9MAXI_Somermeyer_2022` | 31,401 | 13 | 14,268 | 3: 10 | 10 | 0 |
| `D7PM05_CLYGR_Somermeyer_2022` | 24,515 | 23 | 13,198 | 3: 2 | 2 | 0 |
| `F7YBW8_MESOW_Ding_2023` | 7,922 | 10 | 7,756 | none | 0 | 0 |
| `GCN4_YEAST_Staller_2018` | 2,638 | 44 | 2,550 | none | 0 | 0 |

### The independent unit is not the cube, and it is not the site triple either

A cube count is not a sample size, and on this data the site-triple count is not one either. Every cube on one site triple reuses that triple's whole lower-order series, and every site triple in an assay reuses the same underlying single substitutions. Both levels have to be reported:

| Assay | Complete order-3 cubes | Site triples | Substitutions the cubes are built from | Positions |
| --- | ---: | ---: | ---: | ---: |
| `SPG1_STRSG_Wu_2016` | 25,035 | 4 | 76 | 4 |
| `F7YBW8_MESOW_Aakre_2015` | 2,798 | 4 | 37 | 4 |
| `PHOT_CHLRE_Chen_2023` | 890 | 428 | 20 | 15 |
| `CAPSD_AAV2S_Sinai_2021` | 1,954 | 558 | 166 | 26 |
| `HIS7_YEAST_Pokusaeva_2019` | 4,852 | 1,083 | 166 | 89 |
| `GFP_AEQVI_Sarkisyan_2016` | 56 | 56 | 114 | 91 |

A four-position combinatorial library has `comb(4, 3) = 4` position triples however many variants sit on them, which is why the two largest cube counts in the panel contribute four independent position combinations each. The ordering of this table is the reverse of the ordering by cube count, and it is the one that matters.

## Nothing else needed downloading, but the noise channel did

Combinatorial cubes were already local, so no combinatorial dataset was downloaded. What the processed benchmark CSVs cannot supply is any replicate channel for the contrast: one `DMS_score` per variant, no uncertainty column, no duplicated state, and — verified directly rather than assumed — no second assay overlapping a complete cube. `SPG1_STRSG_Olson_2014` covers all four `Wu_2016` positions and shares 2,167 measured states with it, but carries no variant of order 3, so 0 of 25,035 `Wu_2016` cubes are re-measured; the two `F7YBW8_MESOW` assays mutate disjoint position sets, 59/60/61/64 against 48–82, and share no state at all.

One asset was therefore retrieved: ProteinGym's own pre-preprocessing archive, which restores per-replicate columns, read counts and the wild-type rows the processed CSVs drop.

| Asset | Bytes | sha256 |
| --- | ---: | --- |
| `substitutions_raw_DMS.zip`, ProteinGym v1.3 | 112,249,365 | `6d83b16585de2b71b67ae1985193b9eec2e01804784286c515ff276b5372e412` |
| `DMS_substitutions.csv`, reference file | 208,734 | `a8f498011532a74aa9fe556a50555a75e928c5837d19c06a87592ae04049b308` |

Retrieved 2026-09-23 by HTTP GET, both status 200, from `https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/substitutions_raw_DMS.zip` and `https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv`; release record Zenodo `10.5281/zenodo.15293562`. The archive holds 213 members and 495,158,523 uncompressed bytes; 11 members were extracted. Realised disk use is **0.367 GB** measured by `du`, and **0.455 GB** for everything this gate retrieved including the two expansion depositions below. The shared volume held 435 GB free on 96% use when the work started and 364 GB free on 97% use when it finished; that movement is another campaign's concurrent 83.5 GB staging rather than these retrievals, and free space was checked before and after each one. The manifest, including per-member digests, is at ignored `data/proteingym_raw/MANIFEST.json`.

The raw/processed mapping is verified rather than assumed. For the qualified assay, all 165,427 non-wild-type states of the primary measurement block and all 2,121 of the secondary block reconcile to the processed CSV by variant identifier, with zero absent; and every corner of every admitted cube is required to exist in the processed CSV, because that is where a model-side sequence would come from. A silent raw/processed mismatch would otherwise put a floor from one variant set onto contrasts built from another.

What the raw archive actually yields differs sharply by assay, and this is the finding that decides the gate:

| Assay | Per-variant uncertainty in the raw file | Wild-type row | What that supports |
| --- | --- | --- | --- |
| `PHOT_CHLRE_Chen_2023` | `rep1`, `rep2`, `rep3`: three independent FACS replicates | yes, one per block | a **measured** floor and a between-channel decomposition |
| `GFP_AEQVI_Sarkisyan_2016` | SD over unique barcodes, with the barcode count | yes | a propagated floor only; no second channel exists |
| `SPG1_STRSG_Wu_2016`, `SPG1_STRSG_Olson_2014` | input and selected read counts | yes (`VDGV`, Hamming distance 0) | a **lower-bound** floor from counting noise only |
| `GCN4_YEAST_Staller_2018` | two normalised replicates | no | nothing: the assay carries no complete cube |
| `HIS7_YEAST_Pokusaeva_2019` | none; the file is `mutant,selection` | no | nothing from this archive — see the expansion below, which sources a channel elsewhere |
| `CAPSD_AAV2S_Sinai_2021` | none | yes | nothing from this archive, and nothing from its source deposition either |
| Somermeyer ×3 | replicate *count* only, no dispersion | yes | no floor is available |
| `F7YBW8_MESOW_Aakre_2015` | none | yes (fitness 0.0) | no floor is available |

## The estimand, and what restored the absolute one

The processed CSVs carry no wild-type row, and the protocol records the consequence: the unmeasured wild-type value is an assay constant, it enters an order-k cycle contrast with coefficient `(-1) ** k`, and it therefore cancels from within-assay ranks or centred residuals of that contrast and from nothing else. That estimand identifies neither absolute sign nor absolute magnitude and does not remove global assay nonlinearity. It is the estimand this gate would have had to declare on the processed benchmark alone, and it remains the only one available for `HIS7_YEAST_Pokusaeva_2019` and `SPG1_STRSG_Olson_2014`, whose raw files carry no wild-type row either.

The raw archive changes it for the rest. Eight of the eleven higher-order assays carry a measured wild-type state in their raw file, so the declared endpoint is the **absolute wild-type-centred order-k cycle** in the assay's own score units rather than a centred residual. Centring is then unnecessary rather than merely avoidable: the contrast's signed coefficients already sum to zero at every order from 1 upwards, so an assay-wide additive offset cancels without it.

For the qualified assay the scale is **log10 brightness**, taken as `log10(rep_i)` from the study's own per-replicate values. Two properties of that file are checked rather than assumed. `rep_i` is normalised by its measurement block's wild-type replicate mean, and `log_rep_i - log10(rep_i)` recovers that block's wild-type log brightness, which takes exactly two values across the file and is identical across a row's three channels to three decimal places; so the file holds two measurement blocks, 165,428 states and 2,122 states, each with its own wild-type row, and every admitted cube is drawn from one block. And 63 raw rows name a stop codon as `*`; a truncated chain is not a substitution state, so they are excluded and counted, in the same spirit as the MegaScale indel exclusion.

Nonadditivity on a declared transformed assay scale is not by itself a molecular interaction. The floor below is measured on the same scale as the endpoint, so the comparison is internally consistent, but neither quantity is a free energy and the log10 choice is a declared convention rather than a thermodynamic one.

**Construction and sign.** The order-k cycle is the order-k case of the same signed complete-cube contrast the MegaScale pilot declared, reused from `src/transfer/higher_order_cycle.py` rather than reimplemented: `(-1) ** (k - popcount)` per corner, so the all-mutant corner enters positively and the order-2 case is the wild-type-centred pairwise cycle. Positive is the brighter direction. Tests fix the construction against a synthetic generator with a planted third-order term, check that both a strictly additive generator and a strictly pairwise generator with couplings of any magnitude return the construction zero below 1e-12, and check that a partial cube refuses rather than reporting an additive prediction of its missing corner.

**Declaration digest.** `logs/d1_gate_higher_order_extended_20260923/support.json`, sha256 `ddc71784275e710e32849b0aff3dc7c05b86e46b410e912341571a310123e70a`, written before any replicate value was read. The noise-floor stage refuses to run unless that digest still matches and unless it re-derives the same cube count at every order.

## The measurement-noise floor

An order-k cycle sums `2 ** k` corner estimates with unit weights, so its channel noise grows with order and had to be quantified both ways. Measured: the cycle is computed separately in each of the three replicate channels and decomposed pairwise, averaging the three channel pairs. Propagated: the per-state, per-channel measurement standard deviation `sigma` is the pooled within-state variance across the three channels over exactly the corner states the cubes of that order use, and if corner errors were independent the per-channel discordance of the cycle would be `sqrt(2 ** k) * sigma`.

All quantities in log10 brightness; intervals are 95% percentile group bootstraps over **site tuples**, 2,000 resamples at seed 20260923, which is the level the estimand lives at because every cube on one site tuple reuses that tuple's lower-order series.

| Order | Cubes | Site tuples | Kish | `sigma` | Propagated floor | Measured discordance | Shared component | Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2 | 176 | 103 | 65.6 | 0.0548 | 0.1095 | 0.0832 [0.0730, 0.0922] | 0.0187 [0.0000, 0.0405] | 0.232 [0.000, 0.529] |
| 3 | 890 | 428 | 256.7 | 0.0528 | 0.1493 | 0.1219 [0.1149, 0.1284] | 0.0071 [0.0000, 0.0246] | **0.064 [0.000, 0.214]** |
| 4 | 2,753 | 1,184 | 703.4 | 0.0530 | 0.2120 | 0.1714 [0.1658, 0.1769] | 0.0192 [0.0000, 0.0315] | 0.114 [0.000, 0.188] |
| 5 | 5,073 | 2,190 | 1,338.1 | 0.0522 | 0.2952 | 0.2388 [0.2333, 0.2442] | 0.0337 [0.0158, 0.0449] | 0.140 [0.065, 0.189] |
| 6 | 5,212 | 2,491 | 1,655.4 | 0.0511 | 0.4086 | 0.3328 [0.3252, 0.3403] | 0.0288 [0.0000, 0.0692] | 0.084 [0.000, 0.210] |
| 7 | 2,700 | 1,550 | 1,141.9 | 0.0500 | 0.5652 | 0.4523 [0.4383, 0.4657] | 0.0472 [0.0197, 0.1010] | 0.112 [0.048, 0.234] |
| 8 | 587 | 415 | 339.5 | 0.0482 | 0.7717 | 0.5994 [0.5645, 0.6351] | 0.0429 [0.0000, 0.1365] | 0.082 [0.000, 0.246] |
| 9 | 40 | 36 | 33.3 | 0.0438 | 0.9916 | 0.6748 [0.5293, 0.7921] | 0.0761 [0.0000, 0.3340] | 0.118 [0.000, 0.596] |

The propagation is close and consistently conservative: measured over propagated is 0.76 to 0.82 at orders 2 through 8 and 0.68 at order 9, so corner errors are weakly positively correlated within a cube rather than independent — the three channels share block and sorting structure — and the independent-error prediction overstates the floor by about a fifth at every order. That agreement is what establishes the propagation model rather than assuming it.

### Whether the endpoint clears it

It does not, at any order. The reproducible-across-channel component is 6 to 23 per cent of the per-channel discordance, and its interval includes zero at orders 2, 3, 4, 6, 8 and 9. The shared component is an upper bound on reproducible interaction signal, because error common to all three channels contributes to the covariance, and the discordance is a lower bound on per-channel measurement noise, because error common to the channels cancels in the difference; a ratio of 0.064 between an upper bound and a lower bound supports no statement that the order-3 cycle carries reproducible interaction signal. What it does support is a bound in the other direction: on 428 site triples, the reproducible third-order component of log10 brightness has a standard deviation of at most 0.0246 log10, or 5.8 per cent in brightness, at the upper end of its interval.

The failure is a property of this assay rather than of interaction order. Order 2 fails by the same margin, on an endpoint that the MegaScale panel measures at a ratio of 2.78 [2.34, 3.26]. The reason is visible in the assay's own dynamic range: across the primary block's 20 single-substitution states the log10 brightness has a standard deviation of 0.0577, against a per-state per-channel measurement standard deviation of 0.0548. The constituent first-order effects of this combinatorial series are themselves at the replicate noise level, so the differences of differences built from them cannot be above it. This is a library chosen around near-neutral substitutions, and that choice is what the floor is reading.

### The count-derived check on the largest cube support

`SPG1_STRSG_Wu_2016` has no replicate channel, so its only noise model is multinomial counting noise from input and selected read counts. That is a **lower bound on total measurement noise**: it captures sequencing and library-sampling variance and says nothing about assay-level systematic error, batch structure or the selection's own nonlinearity. An endpoint that merely clears it is not qualified to the standard the replicate channel sets. On the selected-over-input enrichment ratio, with a zero selected count floored at one read:

| Order | Cubes | Site tuples | SD of the cycle | Counting-noise RMS | Ratio | Median \|cycle\|/SD | Fraction above 2 SD |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2,091 | 6 | 3.311 | 0.323 | 10.25 | 11.32 | 0.868 |
| 3 | 25,035 | 4 | 3.089 | 0.476 | 6.49 | 5.89 | 0.782 |
| 4 | 109,235 | 1 | 2.907 | 0.633 | 4.59 | 3.81 | 0.706 |

The cycle is six times counting noise at order 3 and 78.2 per cent of cubes exceed twice their own counting standard deviation, so this support is not counting-noise limited. It is limited three other ways instead. The unit count is 4 site triples at order 3 and 1 at order 4, below the package's 8-unit resampling floor, so no interval is reported and the frozen nested held-group design cannot be instantiated. The floor is a lower bound, so clearing it does not qualify the endpoint. And the enrichment ratio is a bounded, saturating scale on which global assay nonlinearity alone produces nonadditivity, which the protocol lists as a required alternative rather than optional cleanup.

## The expansion: a synonymous-genotype channel on the best-powered support

The two assays with the richest position support carry no per-variant uncertainty in either the processed benchmark or ProteinGym's raw bundle, so their channels were sought in the source depositions. One was found and one was not.

**`HIS7_YEAST_Pokusaeva_2019` is qualified.** The publication (PLoS Genetics `10.1371/journal.pgen.1008079`, PMID 30969963) deposits its processed data at `github.com/Lcarey/HIS3InterspeciesEpistasis`, whose `Data/synonymous_variants_rescaled_data.tab.xz` carries one scaled fitness value per *nucleotide* genotype rather than per amino-acid state: 4,337,018 rows over 969,721 amino-acid states in twelve segment libraries. Synonymous genotypes of one amino-acid state are independent library members, independently grown and independently sequenced, so they are a replicate channel in the same sense as MegaScale's two proteases and a stronger one than a counting-noise model.

Three properties are checked rather than assumed. Each segment's window is located in the benchmark's own target sequence by minimising consensus mismatch, and every placement is unique — 1 to 11 mismatches against a next-best of 13 to 24 — and the twelve located windows tile positions 6 to 211, exactly the mutated region the reference file declares. The deposition covers **496,137 of 496,137** processed variants, so the channel and the contrasts are built from one variant set. And the wild-type window is measured in all twelve segments, 105 synonymous genotypes in total, so the endpoint is the absolute wild-type-centred cycle rather than a centred residual. Excluded and counted: 42,656 non-modal-length windows and 179,562 windows naming a residue outside the canonical twenty, which the deposition writes as `_` for a stop or a gap.

Channels are two alternating-rank means over each state's synonymous genotypes, ordered by a digest of the genotype's own DNA sequence. The ordering is label-independent by construction — ordering by fitness instead would put the high measurements in one channel and manufacture a channel offset — and 350,594 processed variants carry the two or more genotypes a split requires.

| Order | Cubes | Site tuples | Kish | Positions | Propagated floor | Measured discordance | Shared component | Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2 | 869 | 281 | 122.6 | 76 | 0.1034 | 0.1177 [0.0995, 0.1385] | 0.0596 [0.0232, 0.0800] | 0.506 [0.178, 0.742] |
| 3 | 2,971 | 598 | 119.6 | 76 | 0.1391 | 0.1583 [0.1398, 0.1776] | 0.0437 [0.0000, 0.0716] | **0.276 [0.000, 0.496]** |
| 4 | 6,078 | 765 | 72.6 | 75 | 0.1885 | 0.2058 [0.1855, 0.2277] | 0.0000 [0.0000, 0.0509] | 0.000 [0.000, 0.267] |
| 5 | 7,412 | 592 | 35.1 | 67 | 0.2557 | 0.2553 [0.2334, 0.2795] | 0.0000 [0.0000, 0.0000] | 0.000 [0.000, 0.000] |

All in scaled-fitness units, where 0 is the nonsense level and 1.0 the wild type. The measured discordance again lands close to the propagated prediction, 1.00 to 1.14 times it, so the channels behave as independent replicates of the same state. At order 2 the reproducible component is resolved above zero, 0.0596 [0.0232, 0.0800], so this assay does carry a measurable pairwise interaction component; it is simply smaller than the noise it has to clear. At order 3 its lower bound reaches zero, and at orders 4 and 5 the point estimate is the construction zero.

**`CAPSD_AAV2S_Sinai_2021` is not qualified, and this is a measured refusal rather than an unchecked gap.** Its source deposition (`github.com/churchlab/Generative_AAV_design`, for bioRxiv `10.1101/2021.04.16.440236`) does carry per-replicate read counts — six plasmid and eleven virus columns — but only for the 7,446-row VAE-designed library partition. Mapped onto the benchmark's numbering over the mutated region 561–588, those rows cover **18 of 42,328** processed variants, **0 of the 1,954** complete order-3 cubes, and no wild-type row. No floor can be established for this assay, and its 558 site triples across 26 positions stay unusable.

### Reading the ratio against the label rather than against one channel

The shared-to-discordance ratio compares the reproducible component against *one channel's* noise, which is the construction the admitted label instrument uses and the one the numbers above report. The quantity a model comparison would actually be fitted to is the pooled label, whose noise is smaller: for a two-way split of all replicates the pooled standard deviation is exactly the channel standard deviation divided by sqrt(2), and for `PHOT_CHLRE`'s three channels averaged into one label it is divided by sqrt(3). Multiplying through gives the generous reading, and it changes no verdict:

| Assay | Order 2 | Order 3 | Deepest order measured |
| --- | --- | --- | --- |
| `HIS7` synonymous | 0.716 [0.252, 1.049] | 0.391 [0.000, 0.702] | 0.000 [0.000, 0.000] at order 5 |
| `PHOT_CHLRE` replicates | 0.402 [0.000, 0.915] | 0.110 [0.000, 0.371] | 0.204 [0.000, 1.032] at order 9 |

Only `HIS7`'s order-2 interval reaches 1 at all, and it reaches it at its upper end. Nothing at order 3 or above comes close on either assay.

## Verdict, and why it is not a model null

**Unresolved, not "not detected".** The two verdicts are different and must not be merged. *Added 2026-09-24:* this expansion is the second of five assessments, and the line has since closed as **measurement-limited with the limit characterised**, which is the closed vocabulary's term for this state and is still not *not detected* — see [D1_HIGHER_ORDER_QUALIFICATION_HARNESS.md](D1_HIGHER_ORDER_QUALIFICATION_HARNESS.md) and audit F29. What this expansion contributes is the **channel-coverage** failure: complete order-3 cubes exist in 9 of 217 assays and only 3 of those 9 carry a replicate or dispersion channel at all. "Not detected" would require a qualified endpoint and a model comparison that found no increment; here no model quantity was read at all, because the endpoint did not clear the floor on either assay where a floor is measurable, and the assays with the largest cube support cannot be qualified above a lower bound or at all. Nothing in this document is a statement about any checkpoint's likelihood assignments or representations.

What *is* established is a measurement result, and it is stronger than the MegaScale pilot's because the support is no longer thin. On 598 site triples across 76 positions at 119.6 Kish effective units — the pairwise endpoint's own scale of support — the reproducible third-order component of this assay's fitness is at most 0.702 of the label's own measurement noise at the upper end of its interval, and at orders 4 and 5 it is indistinguishable from zero. The order-3 cycle on these labels cannot carry a model comparison, and that is a property of the labels rather than of the models.

The 33-arm roster was not run, and no arm was inspected, scored or selected. That matters for what a later run can claim: the roster remains unconditioned on any outcome of this gate.

## Expand or stop, with the cost

**Stop on ProteinGym.** The expansion this gate first recommended was executed, not proposed: a genuine replicate channel was sourced for the richest position support in the census, and the endpoint failed on it at 119.6 Kish effective site triples — the same scale of support on which the pairwise endpoint's ratio is 2.78 [2.34, 3.26]. Two assays now carry measurable floors and both fail at every order from 3 upwards; a third with 558 site triples has been shown, by a measured check rather than an assumption, to have no channel reaching its cubes. Nothing in the remaining panel changes that: the four fluorescent-protein assays contribute 2 to 320 complete triples, and the two four-position libraries contribute four site triples each.

Do not spend the GPU hours. The 9 to 10 H200-hours across 33 arms remain the right budget for a qualified cube panel, and no such panel exists in this data.

The one acquisition that could still change the answer is a combinatorial panel whose labels are **fitted free energies with their own uncertainties**, rather than an enrichment ratio or a fluorescence ratio on a declared scale. The Escobedo, Voigt, Faure and Lehner randomized-core libraries remain the strongest candidate: Zenodo `10.5281/zenodo.16266645`, published 2025-07-24 under CC-BY-4.0, two files of 19.7 MB (`combinatorialcores_github_repo.zip`, md5 `6f0626ae43afba3b03087593197dc3a3`) and 2,226.2 MB (`Escobedo_et_al_protein_randomization_data_files.zip`, md5 `e7d6433bf7cca53071836a097ca63069`). It was assessed and not downloaded: the raw archive plus the HIS3 deposition already supplied two channels, and neither criterion that would justify 2.25 GB — the site-triple count and the presence of per-variant uncertainties on the fitted energies — is checkable from the record metadata. It is recorded as an assessed, unspent option rather than a recommendation, because the finding here is that the *endpoint* is noise-limited on every ProteinGym assay that can be qualified, and an energy scale fixes the estimand rather than the signal-to-noise ratio. A panel of that kind should be qualified on its own floor before any compute is committed to it, exactly as here.

## Limitations

- One assay carries a measured floor, and a result resting on one assay is a one-assay result. Its cube support is 20 substitutions at 15 positions in one protein, so nothing here bears on arbitrary position combinations or on other families, and the 428 site triples are subsets of those 15 positions rather than 428 independent position samples.
- The three replicate channels share library, sorting protocol, block structure and the study's own brightness normalisation. Error common to all three inflates the shared component and is invisible to the channel differences, so the ratio is an upper bound over a lower bound. Nothing here establishes accuracy against an external measurement.
- The scale is a declared convention. The endpoint is a cycle in log10 brightness for the replicate assay and in a linear enrichment ratio for the count assay; neither is an energy, and global assay nonlinearity is not removed by either. The protocol's requirement that a fitted global nonlinearity be treated as a required alternative is unmet here because no model comparison was run.
- The support is conditioned on resolvability in every corner, which is stronger at order k than at order 2 in proportion to `2 ** k`: cubes whose high-order corners fell below the assay's detection floor are absent, and that selection is against exactly the strongly interacting combinations.
- The count-derived floor covers sequencing and library sampling only. It is a lower bound on total noise, and the 6.49 ratio at order 3 is therefore not a qualification.
- Two measurement blocks exist in the qualified assay and cubes are restricted to one. The 19 variants measured in both blocks are too few to measure a between-block offset on the cube support, so the cross-block contrast is not attempted.
- `HIS7_YEAST_Pokusaeva_2019` and `SPG1_STRSG_Olson_2014` have no wild-type row in either the processed or the raw file, so only the centred-residual estimand is available for them, and they have no uncertainty channel, so it is unqualified.
- `GFP_AEQVI_Sarkisyan_2016` carries a genuine per-variant dispersion over barcodes and was not computed: 56 cubes on 56 site triples with no second channel yields only the same propagated-lower-bound comparison that the count assay already supplies on 25,035 cubes.
- `CAPSD_AAV2S_Sinai_2021`'s refusal rests on the one deposition its publication points to. A per-replicate count table for the broader capsid library may exist in the upstream studies that supplied those measurements; this gate checked the deposition named by the assay's own publication and did not pursue the upstream ones.
- The HIS3 channel split makes each channel a mean over half of a state's synonymous genotypes, and most states carry exactly two, so a channel is usually one genotype. The pooled-label reading corrects for that exactly, by sqrt(2), and is reported alongside.
- The HIS3 segment windows overlap, and a cube's corners are pooled across whichever segments measured them. A per-segment offset would not cancel from the contrast if a cube's corners came from different segment libraries; the deposition's fitness is rescaled to a common nonsense-to-wild-type scale per segment, which makes the pooling defensible but does not certify that no residual per-segment scale difference remains.
- No fold-identity or held-out-label leakage test on a fitted readout exists, because no readout was fitted. That is recorded as absent rather than satisfied. The leakage-adjacent condition that does exist is tested: the admitted support is a function of the measured state keys and never of their values, which is why the support could be declared and digested before any channel value was read.

## A note on the MegaScale zero

The MegaScale gate found zero variants of order 3 or above among admitted substitution rows and traced the apparent 2,095 higher-order rows to insertions whose `aa_seq` is truncated to wild-type length. A sibling gate has since reproduced that census independently and found that **every** state of substitution order 3 or above in those bytes is indel-carrying, across orders 3 to 71. That makes the zero mechanical rather than a bare absence: `dataset2` is a singles-and-doubles library, and everything that reads as higher order in it is a length-changing construct seen through a truncated column.

## Artifacts and entry points

`src/transfer/proteingym_higher_order.py` holds the `mutant` grammar, the completeness rule, the cube addressing, the replicate-channel split and the two-channel decomposition; the contrast and its sign convention are imported from `src/transfer/higher_order_cycle.py`. `tests/test_proteingym_higher_order.py` holds the conditions that must always hold. Outputs are under ignored `logs/d1_gate_higher_order_extended_20260923/`, with these digests:

| Artifact | sha256 |
| --- | --- |
| `count_floor.json` | `ff6aab36bd9e9e5809539ec8b8960ce406d46b1431153f8c93170dd40c944a87` |
| `his3_synonymous.json` | `6354fc6848333c71f33f8e8b9e791913d9e414f455e601a9dcc29f67f2cf766c` |
| `noise_floor.json` | `d304270798ff079d2798668a5bde584375ac6f071f2d0834ed9edbed8444e1e2` |
| `proteingym_census.json` | `d56d9ca83fa15e3f30064519a26c28e1a18999f03d716b69d5ed04662bbf5313` |
| `support.json` | `ddc71784275e710e32849b0aff3dc7c05b86e46b410e912341571a310123e70a` |

```
python scripts/transfer/inventory_proteingym_higher_order.py census \
  --out logs/d1_gate_higher_order_extended_20260923/proteingym_census.json
python scripts/transfer/qualify_proteingym_higher_order.py support \
  --out logs/d1_gate_higher_order_extended_20260923/support.json
python scripts/transfer/qualify_proteingym_higher_order.py noise-floor \
  --declaration logs/d1_gate_higher_order_extended_20260923/support.json \
  --declaration-digest ddc71784275e710e32849b0aff3dc7c05b86e46b410e912341571a310123e70a \
  --out logs/d1_gate_higher_order_extended_20260923/noise_floor.json
python scripts/transfer/qualify_proteingym_higher_order.py count-floor \
  --out logs/d1_gate_higher_order_extended_20260923/count_floor.json
python scripts/transfer/qualify_proteingym_higher_order.py synonymous \
  --out logs/d1_gate_higher_order_extended_20260923/his3_synonymous.json
```

The two source depositions retrieved for the expansion are bound, with their digests and the exact API queries, in ignored `data/higher_order_channel_sources.json`.
