# Is interaction signal enriched in structural-contact residue pairs?

This gate asks whether nonadditivity concentrates in residue pairs that touch in an experimental structure, relative to non-contacting pairs matched on the confounders that would otherwise explain the difference. It is a conditioning and explanatory test on the [flagship pairwise experiment](D1_PAIRWISE_EPISTASIS_RESULTS.md), not another capability score, and it has two parts in a fixed order:

1. **Measured enrichment.** Does the *measured* cycle epsilon concentrate at contacts? This is a biological-validity check on the [admitted label instrument](D1_PAIRWISE_LABEL_INSTRUMENT.md). If measured nonadditivity carries no contact structure, a model-side enrichment on the same endpoint could not be interpreted in either direction.
2. **Model-side enrichment.** Conditional on part 1, do the likelihood-interaction and representation-contrast predictions concentrate at contacts?

**Verdict: part 1 is not passed, so part 2 was not run.** On the declared primary contact definition, matching and weighting, the measured contact-minus-matched-control difference in mean absolute epsilon is **+0.192 kcal/mol [−0.061, +0.412]**, which does not resolve, and the global-response-adjusted component — the part of epsilon that is not a monotone response to the additive prediction — is **−0.001 kcal/mol [−0.095, +0.075]**, which bounds any pair-specific contact enrichment below about 0.08 kcal/mol, or 18% of the control arm's own mean. The structures are not what fails: the same annotation, read through the same alignment and the same chain, reproduces the established relationship between burial and single-substitution destabilization on this cohort. What fails is the contrast, on a support whose site pairs the source study selected as coupling tables, whose short-separation contacts have no control at all, and whose control arm carries a Kish effective 25.1 site pairs.

The gate owns the structural annotation, the matching and these measurements. The protocol, the cohort and the endpoint belong to the [readiness record](D1_PAIRWISE_BASELINE_READINESS.md) and the flagship document, and nothing here restates or amends them.

## What the support can and cannot be asked

Three properties of the support bound every number below, and they belong here rather than in a limitations tail.

**The site pairs were selected for suspected coupling.** 99.98% of admitted cycles sit on site pairs the source study itself chose as coupling tables, because it suspected an interaction or a structural contact there. On the 211 evaluated site pairs, 137 carry a hydrogen-bond-network table name and 80 a `dmutv5` name, six of them both. Neither the contact rate in this cohort (158 of 211 site pairs, 74.9%) nor any effect measured on it estimates a population value over arbitrary protein position pairs: the non-contact arm is not a sample of non-contacting position pairs, it is the residue of the study's own selection that turned out not to touch.

**Short-separation pairs are outside the comparison by construction.** All 25 site pairs at sequence separation 1–2 are contacts, and the cohort holds no non-contacting pair at that separation anywhere, so no control exists to match them to. Sequence separation is a distinct axis of this cohort and not a contact annotation, so it must be matched; the price is that the gate is silent on separation 1–2.

**The control arm governs the power.** 111 retained contacts face 53 controls with a Kish effective count of 25.1 across 28 contributing groups. A comparison that does not resolve here is unresolved, not an established absence, and the effective control count is reported beside every estimate for that reason.

Taken together, the licensed question is narrow: contact status among source-selected site pairs at separation ≥3, in states the assay resolves in all four corners of the cycle.

## The structures

Every one of the 64 cohort backgrounds carries a PDB-derived `WT_name`, and an experimental structure was retrieved for all 64 from `https://files.rcsb.org/download/{pdb_id}.cif.gz` on 2026-09-24, 23.6 MB in total. Each file is bound to its own SHA-256 and byte count in the annotation record. No predicted structure is used anywhere in this gate and no folding was run, so the [predicted-structure instrument's](D1_STRUCTURE_INSTRUMENT_RECONCILIATION.md) pLDDT scope does not apply here.

| Provenance | Value |
| --- | --- |
| Backgrounds requested / annotated / excluded | 64 / 64 / 0 |
| Experimental method | 53 solution NMR, 11 X-ray diffraction |
| Deposited models | 10 to 50 per NMR entry, median 20; 1 per X-ray entry |
| Wild-type identity to the matched entity | 100.0% on all 64, over the whole wild type |
| Site-pair residues observed in coordinates | both residues of all 217 site pairs |
| Backgrounds with more than one chain in the model | 1 (`1LP1.pdb`, two copies of one entity) |
| Site pairs within 5 Å of another chain | 0 of 217 |

Numbering is not assumed. The pinned wild-type sequence is aligned to the structure's `_entity_poly` sequence, each wild-type position is mapped to the entity position of its matched column, and that entity position indexes `_atom_site.label_seq_id` directly, so author numbering, insertion codes and chain renaming never enter. The map is a constant offset of 0 to 43 residues on every background, and the one-letter code at every mapped site-pair position equals the wild-type residue. A background whose identity falls below 100.0%, or whose site-pair residues are unobserved, is refused by name rather than annotated from a near match; on this cohort nothing was refused, so the excluded-background accounting is empty by measurement and not by omission.

Two facts restrict what the structures license. `5UP5.pdb` is one of the two backgrounds the grouping step identified as a designed mini-protein mislabelled as natural by a PDB-name screen. It stays in the support carrying that flag rather than being reclassified; it contributes one ineligible contact at separation 1 and one control at separation 4, and removing it moves the primary difference from +0.192 to +0.221 kcal/mol [−0.006, +0.456] and the adjusted difference from −0.001 to +0.007 [−0.085, +0.083]. And a deposited structure is one conformer, or one ensemble, of the isolated domain: contact and burial are computed over the matched chain alone, which is the object the proteolysis assay reports on, but the assay's folded state is not the deposited coordinates and no exact-sequence structural measurement of the assayed construct exists.

## The contact definition and the matching scheme

Both were declared before any endpoint was read, and the annotation that carries them is label-free by construction: the cohort reaches it through a projection that copies names, wild types and site-pair indices only, and a test enforces that the annotation is byte-identical under arbitrary changes to every measurement in the cohort file.

**Contact.** Two residues are in contact when the minimum distance between any pair of their non-hydrogen atoms is at or below **5.0 Å** in the first deposited model of the matched chain; hydrogens, waters, non-polymer heteroatoms and alternate locations after the first are dropped. Two further definitions are reported on the same support and the same matching: **CB–CB at or below 8.0 Å**, with CA standing in for glycine, which is the convention contact prediction uses and which misses a side-chain contact between distant backbones; and **contact in at least half of the deposited models**, which is the definition an NMR ensemble supports and which a single-model entry answers trivially.

**Relative solvent accessibility.** Shrake–Rupley accessible area over the isolated matched chain of the first model, 256 golden-spiral points per atom, probe radius 1.4 Å, divided by the Tien et al. 2013 residue maximum.

**Matching.** Coarsened exact matching, with cells on the three confounders the contrast must balance:

| Axis | Cells | Why it is matched |
| --- | --- | --- |
| Sequence separation | 3–9, 10–19, ≥20 residues | A distinct axis of this cohort, not a contact annotation; an unmatched test would rediscover it |
| Relative accessibility | pair mean at or below 0.2847, above it | Burial is what makes a pair contact at all, and it also scales substitution effects |
| Wild-type residue class | no hydrophobic residue, at least one | The pair's residue identities carry their own effect scale |

The accessibility boundary is the median pair-mean accessibility over the 217 annotated site pairs, a covariate marginal no endpoint enters. Within the eligible cells, a cell holding only one arm is pruned and counted. Each retained contact carries weight one and each cell's controls carry that cell's contact count spread equally among them, so the estimand is the average over retained contacts.

Achieved balance on the primary definition and weighting, as standardized mean differences against the pooled standard deviation of the eligible sample:

| Covariate | Difference before | SMD before | Difference after | SMD after |
| --- | ---: | ---: | ---: | ---: |
| Separation, residues | −6.96 | −0.480 | +1.22 | +0.084 |
| log10 separation | −0.230 | −0.661 | −0.016 | −0.045 |
| Accessibility, pair mean | −0.058 | −0.421 | −0.015 | −0.109 |
| Accessibility, pair minimum | −0.046 | −0.304 | +0.005 | +0.033 |
| Accessibility, pair maximum | −0.069 | −0.441 | −0.035 | −0.220 |
| Hydrophobic residues in the pair | −0.163 | −0.225 | +0.039 | +0.054 |
| Charged residues in the pair | −0.025 | −0.030 | −0.108 | −0.131 |
| Kyte–Doolittle mean | −0.356 | −0.147 | −0.065 | −0.027 |
| Background length, residues | −0.135 | −0.014 | −1.80 | −0.182 |

Matching removes most of the separation and burial imbalance, the two largest, and leaves pair-maximum accessibility at −0.220 and background length at −0.182, both residual within-cell imbalances the coarsening does not reach. Charged-residue count, hydropathy and length are balance-only covariates rather than matching cells; adding cells for them at 53 controls would prune more contacts than it would balance. The table is computed for the primary weighting. The group-equal variant re-flattens each arm's group weight after the cell weights are formed, so its cell shares are no longer exactly matched, which is one reason it is a sensitivity and not a second primary.

**Intervals.** 2,000-draw percentile intervals from a two-stage bootstrap at seed 20260923: held groups are drawn with replacement, then the site pairs inside each drawn group. Both arms of a drawn group travel together and the matching weights are re-derived inside every draw, so the control arm follows the contacts the draw contains rather than carrying weights fitted on the observed sample. The cycle count never enters as a sample size. The outer unit clears the repository's eight-unit floor at 63 groups; the inner stage carries no floor, because a background legitimately holds one site pair.

## The support, and one declared filter

`build_cohort` keys a state on `aa_seq` alone, so an insertion or deletion construct whose `aa_seq` is truncated to the wild type's length enters the substitution support as though it were a substitution. Five such states reach the frozen cohort, across `1O6X.pdb`, `2AMI.pdb` and `2D1U.pdb`, carrying 130 of 8,192 cycles (1.59%). They are not valid measurements of a substitution effect, and they are excluded here as a declared filter on cycles; the frozen cohort digest `8133463e…` is untouched, because three gates declare it as their support. The exclusion removes six site pairs, **all six from the contact stratum** — contamination sitting wholly in one arm is exactly the shape that biases an enrichment comparison — and `2AMI.pdb` leaves entirely because its own wild-type state is indel-derived. The filtered support is the primary one; the unfiltered support is reported as a sensitivity, where the primary difference reads +0.176 kcal/mol [−0.085, +0.405] instead of +0.192.

| Support | Value |
| --- | --- |
| Backgrounds and held groups | 63 (one background per group) |
| Site pairs annotated / evaluated | 217 / 211 |
| Cycles | 8,062 |
| Contacts / non-contacts, primary definition | 158 / 53 |
| Retained after matching | 111 contacts, 53 controls |
| Pruned: ineligible separation / contact-only cells | 22 / 25 |
| Kish effective site pairs, contact / control arm | 111.0 / 25.1 |
| Groups contributing, contact / control arm | 54 / 28 |
| Cycles behind the two arms | 4,812 / 1,695 |

## Part 1: what the measured endpoint does

Each site pair contributes one value, cycles weighted equally inside it. The endpoint is the cohort's own epsilon in kcal/mol; the adjusted endpoint subtracts a twenty-quantile-bin response in the measured additive prediction, fitted in sample over the retained cycles exactly as the label-instrument qualification did, with a fitted response spanning 2.41 kcal/mol.

| Endpoint | Contact | Matched control | Difference [95%] |
| --- | ---: | ---: | --- |
| Mean absolute epsilon | +0.933 | +0.741 | **+0.192 [−0.061, +0.412]** |
| Mean epsilon | +0.843 | +0.651 | +0.193 [−0.104, +0.451] |
| Mean absolute adjusted epsilon | +0.408 | +0.409 | **−0.001 [−0.095, +0.075]** |
| Mean adjusted epsilon | +0.021 | −0.086 | +0.106 [−0.136, +0.257] |
| Mean absolute half channel difference | +0.158 | +0.126 | +0.031 [−0.007, +0.064] |

All entries are kcal/mol at the site-pair level, on the primary definition and weighting, over 111 contacts and 53 controls at a Kish effective control count of 25.1. Under group-equal weighting the four interaction endpoints read +0.165 [−0.046, +0.344], +0.124 [−0.113, +0.327], +0.029 [−0.062, +0.088] and +0.056 [−0.106, +0.188], at a Kish effective control count of 37.0.

The last row is a pure-noise negative control on this exact support: the half-difference between the trypsin-channel and the chymotrypsin-channel epsilon of the same cycle, rebuilt from the pinned per-channel columns under the cohort's own admission rule, so every quantity in it is label-instrument discordance rather than interaction. Its per-cycle standard deviation here is 0.201 kcal/mol, against 0.284 kcal/mol for the whole unsampled panel, and its mean absolute value is 0.172 kcal/mol. That the same estimator returns **+0.031 kcal/mol [−0.007, +0.064]** on it is the floor the interaction estimates must be read against: a contact-minus-control difference of that size is reachable from channel discordance alone, and the adjusted interaction difference of −0.001 kcal/mol sits inside that band.

The definition sensitivities move the point estimate without changing the reading. The CB definition gives +0.064 [−0.150, +0.330] on mean absolute epsilon and −0.060 [−0.131, +0.070] adjusted. The ensemble-majority definition gives +0.250 [−0.011, +0.443] and −0.012 [−0.119, +0.066]; its group-equal variant, +0.208 [+0.017, +0.373], is the one interval among the 44 reported on this support whose interval excludes zero, which is what 44 unadjusted pointwise intervals produce and is not read as a result.

### By experimental method, and which contact definition the NMR majority deserves

53 of the 64 backgrounds are solution NMR, where the first deposited model carries no special status and the ensemble spread is conformational uncertainty rather than noise. A contact reproduced across an ensemble is a stronger annotation than one read from a single arbitrarily indexed model, so the two definitions are measured against each other per method rather than pooled behind the first model.

| Stratum | Site pairs | Groups | Contacts, first model | Contacts, majority of models | Site pairs changing status |
| --- | ---: | ---: | ---: | ---: | ---: |
| Solution NMR | 169 | 52 | 120 | 122 | 12 (5 first-model only, 7 majority only) |
| X-ray diffraction | 42 | 11 | 38 | 38 | 0 (one model per entry) |
| Pooled | 211 | 63 | 158 | 160 | 12 |

The two definitions disagree on 12 of 169 NMR site pairs, 7.1%, and the median ensemble fraction of first-model contacts is 1.00 — most first-model contacts hold in every deposited model. They also do not disagree on the reading:

| Stratum and definition | Mean absolute epsilon | Adjusted | Contacts / controls (Kish effective controls) |
| --- | --- | --- | --- |
| NMR, first model | +0.179 [−0.118, +0.426] | −0.007 [−0.086, +0.075] | 75 / 49 (26.7) |
| NMR, majority of models | +0.215 [−0.069, +0.443] | −0.002 [−0.106, +0.066] | 100 / 47 (22.5) |
| X-ray, either definition | +0.335 [−0.591, +1.068] | −0.057 [−0.277, +0.227] | 16 / 3 (2.7) |

All entries are kcal/mol at the site-pair level under the primary weighting. On the NMR stratum the majority-of-models definition is the better instrument and is the one to read there; it moves the point estimate by +0.036 kcal/mol on the raw endpoint and by +0.005 on the adjusted one, and both remain unresolved and flat respectively. The X-ray stratum cannot answer the question at all: its matched control arm is three site pairs at a Kish effective 2.7, so its interval spans 1.66 kcal/mol and is reported only to show that the pooled estimate is not carried by those eleven backgrounds. The first-model choice is therefore not what produces the negative.

### The structures work, which is what makes the null readable

A flat contact contrast has at least four candidate causes: the structures, the contact definition, the matching, or the endpoint. The structure instrument is qualified here by a route independent of the contact contrast. Relative accessibility at a site-pair position is read through the same alignment, the same chain and the same calculation as the contact calls, while the mean measured single-substitution ddG at that position is a label the annotation never saw.

| Positive control | Value |
| --- | --- |
| Site-pair positions carrying measured singles | 324, median 16 singles per position |
| Within-background rank correlation, accessibility against mean ddG | **+0.285 [+0.130, +0.434]** over 43 backgrounds |
| Mean ddG, buried positions (accessibility ≤ 0.2847) | −1.554 kcal/mol over 161 positions |
| Mean ddG, exposed positions | −1.052 kcal/mol over 163 positions |

Positive `dG_ML` is more stable, so a destabilizing substitution carries negative ddG. Buried positions are destabilized by 0.502 kcal/mol more than exposed ones, and the background-equal rank correlation between accessibility and mean ddG is bounded away from zero, resampling backgrounds. The numbering, the chain choice and the accessibility calculation therefore carry real information about this assay, and the flat contact result is a property of the contrast and its support rather than of the structures.

## Verdict

**The gate returns a negative at part 1, and part 2 was not run.**

Licensed: on the declared primary definition, matching and weighting, and among source-selected site pairs at separation ≥3 in resolvable states, the contact-versus-matched-control difference in measured nonadditivity **does not resolve** on the raw endpoint, at +0.192 kcal/mol [−0.061, +0.412], and is **bounded near zero** on the pair-specific endpoint that removes the global response, at −0.001 kcal/mol [−0.095, +0.075] against a noise-endpoint difference of +0.031 [−0.007, +0.064] on the same support.

Licensed: what raw contact-versus-control difference there is behaves like the global stability response rather than pair-specific coupling. Contacts carry larger single-substitution effects — buried positions are 0.502 kcal/mol more destabilized than exposed ones — the endpoint carries roughly half its variance as a monotone response to the additive prediction, and removing that response removes the whole difference.

Licensed: the structure instrument is not what fails, by the burial positive control.

Not licensed: reading this as evidence that model interaction predictions are unrelated to contacts, or that contacts carry no coupling. The control arm's Kish effective count is 25.1 and the raw endpoint's interval half-width is about 0.24 kcal/mol, so a real effect of that size would go undetected; 25 of the 158 contacts sit in cells no control occupies and a further 22 at separations where no control exists. The correct statement is that the endpoint admitted for the flagship experiment does not resolve a contact contrast on the support the flagship runs on, so a model-side contact enrichment measured on it would not have been interpretable — which is why part 2 was not run rather than run and reported as uninformative.

## Limitations

- **Site-pair selection dominates**, as stated above: 99.98% of admitted cycles sit on source-selected coupling tables, and neither prevalence nor effect size transfers to arbitrary position pairs in either direction.
- Power is set by 53 non-contact site pairs at a Kish effective count of 25.1 across 28 groups, not by 8,062 cycles. No interval here treats cycles as independent.
- Separation 1–2 has no control anywhere in this cohort, so 22 contacts are outside the comparison by construction on the filtered support, and 25 further contacts fall in cells no control occupies. The estimand is the average over the 111 retained contacts, not over all contacts.
- The adjusted endpoint conditions on the measured additive prediction, which is partly a consequence of contact: burying two interacting residues raises both single effects. It is therefore a lower bound on a contact effect while the raw endpoint is an upper bound, and both fail to resolve.
- The coarsening leaves within-cell imbalance — pair-maximum accessibility at −0.220 SMD and background length at −0.182 SMD after weighting — and charged-residue count, hydropathy and length are balance-only rather than matched.
- A deposited structure is one conformer or one ensemble of the isolated domain, and 53 of 64 backgrounds are solution NMR. The X-ray stratum is too small to answer the question on its own: three matched control site pairs at a Kish effective 2.7.
- The contact definition is a distance threshold on heavy atoms. It does not distinguish a hydrogen bond, a salt bridge or a packing contact, and it does not measure coupling: two residues can touch without their substitutions interacting thermodynamically, which is one mechanism by which a contact stratum can carry no extra nonadditivity.
- The indel filter drops 130 cycles whose states alias an insertion or deletion construct rather than correcting them, because the frozen cohort's medians are not recomputed here.
- `5UP5.pdb` remains flagged as a designed mini-protein inside the natural stratum; it is not reclassified, and the estimate without it is reported.
- The primary support carries 44 intervals — three contact definitions by five endpoints by two weightings, plus the method strata and the flagged-background sensitivity — and they are pointwise and unadjusted for that multiplicity. One excludes zero. Each conditions on the observed site-pair statistics.

## Artifacts and entry points

`scripts/transfer/build_contact_annotation.py` writes the label-free annotation and `scripts/transfer/measure_contact_epsilon_enrichment.py` the measurement; the contact definition, the accessibility calculation, the matching and the two-stage bootstrap are in `src/transfer/contact_enrichment.py`, and the conditions that must hold — the contact threshold on synthetic geometry, the achieved balance, the refusal of a misaligned numbering, and the invariance of both the annotation and the matching to every measurement — are in `tests/test_contact_enrichment.py`. Raw outputs, the 64 structure files and their digests are under ignored `logs/d1_gate_structure_contact_20260924/`: the annotation is `contact_annotation.json`, SHA-256 `3d07129b9d40c7e64f6099c16101240a2baa58abdd7ef3a5798f20cf5e49508c`, bound to cohort `8133463e…`; the measurement is `epsilon_enrichment.json`, SHA-256 `b5c050c09b11fb422abc0a0eb0f41d1f387aa5cffd09d78363b409d58b7342ab`, bound to that annotation and to both pinned MegaScale parquet digests. Both are written through the shared atomic writer that refuses non-finite values. The whole gate is CPU work: 17 s for the annotation over 64 structures and about 29 minutes single-process for the measurement, whose cost is 88 two-stage bootstraps of 2,000 draws each over the two supports, two background bootstraps for the positive control, and one pass over the pinned parquet files. No GPU was requested and no pod was held.
