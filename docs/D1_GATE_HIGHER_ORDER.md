# Higher-order interaction: a bounded pilot, and why it stops at the label instrument

This gate asks whether measured higher-order dependence — an interaction that a strictly pairwise model with position-independent couplings cannot express — is available on the pinned MegaScale `dataset2` bytes at a scale that could carry a model comparison. It is qualified on its own: the four-state pairwise interaction contrast returned a null for the representation on all 33 arms of the [pairwise experiment](D1_PAIRWISE_EPISTASIS_RESULTS.md), and nothing in that outcome was treated as predicting this one, because a third-order cycle is not nested inside a four-state contrast. The pilot was designed to stop cheaply if the support turned out thin, and that is what happened.

The work is measurement-only and CPU-only. No model score, likelihood or representation enters any quantity below, and no GPU was scheduled. The [label instrument](D1_PAIRWISE_LABEL_INSTRUMENT.md) owns the admitted per-channel rule and the between-channel decomposition, the [baseline readiness record](D1_PAIRWISE_BASELINE_READINESS.md) owns the grouping contract, and the [protocol](D1_INFORMATION_HIERARCHY_PROTOCOL.md) owns the estimand discipline. This document owns what the availability inventory counted and what the noise floor measured.

## Decision

**The higher-order gate is unresolvable on this data.** Two findings carry the decision, and either alone would be sufficient. *Added 2026-09-24:* this pilot is the first of five assessments, and the line has since closed as **measurement-limited with the limit characterised** rather than as not detected — see [D1_HIGHER_ORDER_QUALIFICATION_HARNESS.md](D1_HIGHER_ORDER_QUALIFICATION_HARNESS.md) and audit F29. What this pilot contributes to that verdict is the **order** failure: these bytes contain no substitution variant of order three or above at all.

The support is two independent units. `dataset2` contains no variant with three or more substitutions relative to its own background wild type, so the only third-order measurement in these bytes is a cross-background one, and after the admitted rule that route yields 60 complete eight-state cycles resting on **2 distinct site pairs in 2 family groups**, against the 217 site pairs and 121.6 Kish effective site pairs the pairwise cohort carries. The frozen five-outer, four-inner held-group fold design cannot be instantiated on two groups at all.

The endpoint does not clear its own measurement-noise floor. On those 60 cycles the third-order cycle's reproducible-across-channel component has a standard deviation of 0.464 kcal/mol against a per-channel discordance standard deviation of 0.436 kcal/mol, a ratio of 1.06, where the pairwise endpoint's ratio is 2.78 [2.34, 3.26]. The measured discordance sits inside the interval the label instrument's own order-2 measurement propagates to order three, 0.403 [0.359, 0.451] kcal/mol. **Any variation seen here is a measurement limitation, not a model finding**, and no model fit was run.

The recommendation is to stop. Expansion is not available from this dataset at any cost, because the limit is the absence of measured combinatorial states rather than a shortage of compute.

## What was available, for both candidate endpoints

The inventory reads only the *keys* of the admitted measured states, never their values, so the support it reports cannot be a function of the labels that support would carry. It runs the label instrument's tightened rule — finite numeric `dG_ML` and a confidence width in [0, 0.5] kcal/mol on the combined estimate and on each protease channel separately — which accepts 575,331 of 776,298 rows, and then restricts to substitution-labelled rows, leaving 521,890. Median aggregation per exact (`WT_name`, `aa_seq`) state is the instrument's.

### Triple mutants: none exist

Among admitted substitution rows the census reads 2,557 wild-type states, 402,962 single substitutions, 115,984 double substitutions and **zero variants of order three or above**. The `mut_type` vocabulary confirms it independently: every one of the 776,298 rows is `wt`, a colon-joined pair of at most two uppercase substitutions, an insertion or a deletion.

A naive count of `aa_seq` differences says otherwise, and the difference is a trap worth recording. Before the substitution restriction, 2,095 rows appear to carry three or more substitutions relative to their catalogue wild type. Every one is a `mut_type` insertion. A length-changing construct's `aa_seq` is truncated to the wild type's length, so an insertion frame-shifts the residues after its position and reads as a run of substitutions — `insG5` in a 72-residue background reads as five of them, and `insG1` reads as exactly one. The third-order endpoint inside a single background therefore has no support, and the apparent support is an artifact of the column.

### Background dependence: two site pairs

The available route is a mutation pair whose complete four-state cycle is measured in two different wild-type backgrounds. When the two backgrounds differ at exactly one site that the pair does not touch, all eight corners of a third-order cube are measured, four in each background, and the contrast is an exact third-order cycle rather than a repeat.

| Wild types differ at | Contrasts | Mutation pairs | Site pairs | Family groups | Kish site pairs | Contrast order |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 site | 60 | 56 | 2 | 2 | 1.97 | 3 |
| ≤ 2 sites | 82 | 74 | 3 | 2 | 2.30 | 3–4 |
| ≤ 3 sites | 271 | 260 | 5 | 3 | 2.09 | 3–5 |
| ≤ 5 sites | 274 | 260 | 5 | 3 | 2.14 | 3–6 |
| ≤ 10 sites | 581 | 552 | 7 | 5 | 3.13 | 3–12 |

Site-pair counts are the independent units, because every mutation pair on one site pair shares both single-mutant series in both backgrounds. The wider cuts buy contrasts rather than units, and they buy them at the cost of the estimand: 307 of the 581 contrasts at the widest cut rest on backgrounds differing at 10 to 12 positions, where the measured difference of two epsilons is a sum of orders 10 through 12 and no longer a third-order quantity.

The exact third-order support is four background pairs on two site pairs, in two family groups both already represented in the frozen pairwise cohort:

| Reference | Variant | Site pair | Background site | Cycles | Group | Length |
| --- | --- | --- | ---: | ---: | --- | ---: |
| `1E0L.pdb` | `2MWA.pdb` | 9–23 | 19 | 20 | nat-007 | 37 |
| `1E0L.pdb` | `2MWB.pdb` | 9–23 | 30 | 14 | nat-007 | 37 |
| `1W4H.pdb` | `2BTH.pdb` | 26–36 | 40 | 18 | nat-030 | 44 |
| `1W4H.pdb` | `2WXC.pdb` | 26–36 | 16 | 8 | nat-030 | 44 |

All 60 cycles fall in the ≥10-residue separation stratum. Both site pairs are source-selected coupling tables, so the same scope restriction the label instrument records applies here and nothing generalises to arbitrary position pairs.

Other routes were checked and give nothing. Mutation pairs recurring across backgrounds of unequal length cannot be addressed by position at all and are excluded. Lifting the distance cut entirely admits 1,577 contrasts on 13 site pairs across 25 backgrounds, but 996 of them rest on equal-length backgrounds differing at 19 to 49 of their 37 to 72 positions. Two unrelated domains sharing a residue pair at the same index is index coincidence rather than a background contrast, so admitting those would add 6 site pairs by abandoning the estimand.

## The chosen endpoint, declared before any fit

The endpoint is the **exact third-order cycle on the background-distance-1 support**, 60 cycles on 2 site pairs in 2 family groups. It was chosen on those counts alone, which were produced and written before the noise floor was measured.

**Construction.** Index the cube by `(A, B, C)` where A and B are the mutation pair's two substitutions ordered by sequence position and C is the single substitution separating the reference wild type from the variant wild type. All eight corners are measured states: the four with C off in the reference background, the four with C on in the variant background. The contrast is

    omega = y_ABC − y_AB − y_AC − y_BC + y_A + y_B + y_C − y_WT

which is the order-3 case of one signed complete-cube contrast whose order-2 case is the pairwise cycle epsilon the instrument admitted. Equivalently `omega = epsilon(variant) − epsilon(reference)`, which is the identity that makes a recurring mutation pair a third-order measurement.

**Sign convention.** The reference background is the alphabetically smaller `WT_name`, a label-independent choice that coincides with the documented parent for every name-derived parent/child pair because a parent's name is a prefix of its child's. Positive `dG_ML` is the more stable direction, established in the instrument from the source `ddG_ML` column at a root-mean-square residual of 0.018 kcal/mol, so `epsilon > 0` places a double substitution above its additive prediction and `omega > 0` places the pair's interaction epsilon higher in the variant background than in the reference. Swapping the two backgrounds flips the sign exactly.

Two properties bound what the construction can be blamed for. A per-background additive offset cancels exactly, because the four reference corners carry signs −1, +1, +1, −1 and the four variant corners +1, −1, −1, +1, so a background-specific scale shift in `dG_ML` cannot manufacture omega. And a strictly pairwise generator with position-independent couplings of any magnitude returns omega at its construction zero, below 1e-12 kcal/mol for values of order 1 kcal/mol, which is floating-point cancellation and is measured rather than assumed.

**Declaration digest.** `logs/d1_gate_higher_order_20260923/inventory.json`, sha256 `3b0b883eaacdafcbf15bcce6a953380ac112038207fd05490d1a253991298b26`, generated 2026-09-24T05:05:13 UTC, with the noise floor measured at 05:07:19 UTC. The noise-floor stage refuses to run unless that digest still matches and unless it re-derives the same contrast count at every cut.

## The measurement-noise floor

An order-3 cycle subtracts eight estimates where an order-2 cycle subtracts four, so its channel noise is larger and had to be quantified. The two constituent epsilons share no measurement — the reference corners and the variant corners are disjoint states — so their channel-difference variances add and the order-3 discordance standard deviation is sqrt(2) times the order-2 one. Applied to the instrument's own measurement on 92 source clusters and 25.1 Kish effective clusters:

| Instrument stratum | Order-2 discordance SD | Propagated order-3 floor |
| --- | --- | --- |
| all | 0.2842 [0.2574, 0.3120] | **0.4019 [0.3639, 0.4412]** |
| ≥10 residues | 0.2849 [0.2537, 0.3186] | **0.4029 [0.3587, 0.4505]** |

All entries kcal/mol; the ≥10 row is the stratum all 60 cycles fall in. This is the only channel-noise estimate on this endpoint that carries a usable interval, and it is inherited by propagation rather than measured on the pilot's own two units.

Measured directly on the 60 cycles, with the instrument's own estimators:

| Quantity, kcal/mol | Trypsin | Chymotrypsin |
| --- | ---: | ---: |
| Mean omega | −0.041 | −0.151 |
| SD of omega | 0.694 | 0.574 |

| Decomposition | Point estimate |
| --- | ---: |
| Between-channel Pearson r | +0.540 |
| Shared-component SD, kcal/mol | 0.464 |
| Per-channel discordance SD, kcal/mol | 0.436 |
| Shared-to-discordance ratio | 1.06 |
| Channel offset, trypsin minus chymotrypsin, kcal/mol | +0.109 |

No interval accompanies any of these. Two site pairs and two family groups are below the declared floor of eight resampling units: with n units the probability that a bootstrap draw is one unit repeated is n^(1−n), which is 0.5 at n = 2, so the percentile endpoints would be individual unit means rather than a sampling interval. Reporting them as an interval would misdescribe the support, so the output withholds them and names the reason.

The floor is if anything understated on this particular support. The constituent order-2 discordance measured on these same site pairs is 0.561 kcal/mol in the reference backgrounds and 0.596 in the variant backgrounds, roughly twice the panel-wide 0.284, so these two site pairs are among the noisiest in the panel rather than the cleanest. Propagating from those local values by the same square-root rule predicts 0.818 kcal/mol, well above the 0.436 measured, which means the two constituent channel differences are positively correlated rather than independent: some channel error is common to the reference and variant background measurements, and the square-root propagation is conservative here.

### Whether the endpoint clears it

It does not. The reproducible component, 0.464 kcal/mol, exceeds the per-channel discordance, 0.436 kcal/mol, by a factor of 1.06. The shared component is an upper bound on reproducible interaction signal because shared assay error contributes to the between-channel covariance, and the discordance is a lower bound on per-channel measurement noise because error common to both channels cancels in the difference. A ratio of 1.06 between an upper bound and a lower bound supports no statement that omega carries reproducible interaction signal at all. For comparison, the same ratio on the pairwise endpoint is 2.78 [2.34, 3.26], and on its global-response-adjusted scale 2.06 [1.77, 2.36].

Widening the relatedness cut raises the ratio to 1.20 at ≤3 sites and 1.42 at ≤10 sites, still far below the pairwise endpoint's, and at those cuts the contrast is no longer third order. The trend is what a mixture of a genuine higher-order component with a growing number of lower-order terms would produce, and it is not evidence for the third-order quantity.

## Power, stated explicitly

The pilot's unresolved result is not a null. Two facts bound what could have been detected.

The design cannot be instantiated. The frozen contract is five outer and four inner held-group folds on three prespecified split seeds with equal group weighting. Two family groups admit at most two outer folds of one group each and no inner loop for penalty selection, against a matched baseline block of 864 coordinates. There is no fold assignment on this support under which the composed comparison exists.

Even ignoring that, the interval width is an order of magnitude above the effects this endpoint family produces. Group-bootstrap half-widths scale roughly as the inverse square root of the effective unit count, so moving from 121.6 to 1.97 Kish effective site pairs inflates them by a factor of about 7.9. The pairwise experiment's reproducible increments carried half-widths of 0.0008 to 0.0046 kcal²/mol², which inflate to roughly 0.007 to 0.036 kcal²/mol². The largest reproducible model increment measured anywhere on the pairwise endpoint was +0.0021 kcal²/mol² for the likelihood interaction and +0.0081 for the first-order likelihood block. The smallest increment this support could separate from zero therefore exceeds the largest effect the analogous endpoint has produced. This is an order-of-magnitude scaling argument on the reported intervals, not a formal power calculation, and it points the same way as the floor.

## Verdict and recommendation

**Stop.** The higher-order gate is reported as unresolvable on this data rather than as a null. The endpoint exists, its construction is exact and its sign convention is fixed and tested, but its support is two independent units and its reproducible component does not separate from the two protease channels' own disagreement.

No expansion is recommended and none would help. The constraint is the absence of measured combinatorial states: `dataset2` contains no triple substitution, and the cross-background route is exhausted at 60 cycles by the dataset itself, not by the admission rule. Relaxing the admission rule to the combined-width-only version raises the raw count but not the unit count, and it reintroduces the fit-bound channel values the instrument excluded for cause. Spending compute here would buy a fit whose interval cannot exclude the effects it is looking for.

If a measured combinatorial panel of the pairwise cohort's scale ever became available — order 64 family groups with complete eight-state cubes — the model-side cost would be about twice the pairwise panel's, since a cube carries eight states where a cycle carries four: roughly 9 to 10 GPU-hours of forwards across 33 arms on H200 at batch size one, plus the same CPU fitting budget. That is a cost for labels that do not exist, and it is recorded so the number is not rediscovered rather than as a proposal.

## A defect found on the way, in the pairwise cohort

The `aa_seq` truncation described above is not only a census trap. The pairwise cohort builder keys measured states on `aa_seq` alone and does not consult `mut_type`, so a length-changing construct whose truncated `aa_seq` happens to match the wild type's length enters the substitution support as if it were a substitution state.

On the admitted, length-matched support, 1,486 of 504,298 states are carried by at least one insertion or deletion row. Of these, 1,463 are carried only by indel rows and 23 pool an indel row with genuine substitution rows under one median; 1 is a wild-type state, 26 are order 1 and 25 are order 2, which is the set that can enter a four-state cycle.

In the frozen cohort (`logs/d1_pairwise_cohort_20260924/cohort.json`, sha256 `8133463e…`), 5 of 12,977 sequences rest on such a state, in 3 of the 64 backgrounds (`1O6X.pdb`, `2AMI.pdb`, `2D1U.pdb`), and they enter **130 of 8,192 cycles**, or 1.59%. Three are order-2 states carried *only* by an `insA2` insertion row, so those double-mutant corners are entirely an insertion construct's stability. One is `1O6X.pdb`'s order-1 state pooled by median over `V1G`, `insG1` and `insG2`. One is `2AMI.pdb`'s wild-type state, a median over 9 rows one of which is `insG1`, and a wild type enters every cycle of its background with weight +1.

The impact on the pairwise result is small — 1.59% of cycles in 3 of 64 equally weighted groups, with medians that are robust where several rows contribute — and it is reported rather than repaired, because `src/transfer/pairwise_stability.py` and the frozen cohort are not this gate's to write. The fix is one clause in the admission rule: exclude rows whose `mut_type` begins `ins` or `del`. The `mut_type` vocabulary is exactly `wt`, colon-joined uppercase substitutions, `ins<residue><position>` and `del<residue><position>`, so the lowercase prefix identifies every length-changing row without a value-dependent filter.

This gate applies that exclusion to its own support, and the exclusion is not cosmetic here: it removes the entire 1–2 separation stratum from the background-dependence inventory, six contrasts on the `1UBQ.pdb` point-mutant backgrounds whose site pair 1–2 rested on exactly this conflation at sequence position 1.

## Limitations

- Two site pairs and two family groups. Every quantity measured on the exact third-order support is a point estimate with no interval, and the two site pairs are not a sample of anything.
- Both site pairs are source-selected coupling tables, and both lie in the ≥10-residue separation stratum, which on this panel is a sequence-distance stratum and not a contact contrast. Nothing here bears on arbitrary position pairs or on local site pairs.
- The floor is propagated from the instrument's order-2 measurement under the assumption that the two constituent cycles' channel errors are independent. On this support they are positively correlated, so the propagated floor is conservative; the measured order-3 discordance of 0.436 kcal/mol is the number the verdict rests on, and it has no interval.
- Between-channel agreement measures reproducibility across two channels sharing sequences, library, counts and the proteolysis-model assumptions. Shared error inflates the shared component and is invisible to the channel difference, so the 1.06 ratio is an upper bound over a lower bound. Nothing here establishes accuracy against an external thermodynamic measurement.
- The support is conditioned on resolvability in all eight states under the tightened per-channel rule, so cubes whose triple-substituted corner falls below the assay's stability floor are absent. This selection is stronger than the pairwise cohort's, which needs four resolvable states rather than eight.
- The cross-background construction assumes the two backgrounds' `dG_ML` values are on one scale up to a per-background additive offset. The offset cancels exactly; a per-background multiplicative scale difference would not, and nothing in the retained provenance certifies its absence.
- No model was scored, so this document makes no statement about any model's likelihood assignments or representations on higher-order dependence. The gate is unresolved, not negative.

## Artifacts and entry points

`scripts/transfer/inventory_higher_order_support.py` runs both stages; `src/transfer/higher_order_cycle.py` holds the cube contrast, the sign convention and the support enumeration; `tests/test_higher_order_cycle.py` holds the conditions that must always hold.

Those conditions are the ones this pilot can carry. The tests fix the eight-corner construction and its sign convention against a synthetic generator with a planted third-order term, check that both a strictly additive and a strictly pairwise generator return the construction zero, check that the support enumeration is handed measured state keys and never a value, and check that the noise-floor stage refuses to run without a declaration and that the support it re-derives matches the declared inventory at every cut. A fold-identity test and a held-out-label-leakage test on a fitted readout are not instantiable, because no readout was fitted; that is recorded as absent rather than as satisfied.

Outputs are the ignored `logs/d1_gate_higher_order_20260923/inventory.json` (sha256 `3b0b883eaacdafcbf15bcce6a953380ac112038207fd05490d1a253991298b26`) and `noise_floor.json` (sha256 `93d3885fe5c5831dbfd100eed5616629d7f79f01f8e0fc22f78572db76186a8e`).

```
python scripts/transfer/inventory_higher_order_support.py inventory \
  --out logs/d1_gate_higher_order_20260923/inventory.json
python scripts/transfer/inventory_higher_order_support.py noise-floor \
  --inventory logs/d1_gate_higher_order_20260923/inventory.json \
  --declaration-digest 3b0b883eaacdafcbf15bcce6a953380ac112038207fd05490d1a253991298b26 \
  --out logs/d1_gate_higher_order_20260923/noise_floor.json
```
