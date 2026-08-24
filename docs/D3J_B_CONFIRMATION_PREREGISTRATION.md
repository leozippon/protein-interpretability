# Pre-registration — two-sided alphabet-chemistry confirmation

**Date:** 2026-08-24 **Status:** frozen before any D3.j-B result **Scope:** D3.j-B axis construction and two independent confirmation draws. **Authority:** `docs/INTERPRETABILITY_TRANSFER_AUDIT.md` remains canonical for findings, limitations, and claim status. This document fixes the design and stopping rules; it asserts no result.

## Question

Does replacing the input embedding row of one amino acid with another damage a protein decoder according to declared physicochemical similarity after the strongest implemented fragment-statistics explanation is made to predict the opposite ordering?

D3.j-A cannot answer this question two-sidedly. Its admission axis is a 3-mer context-profile cosine, while its ceiling is the fragment conditional's own substitution damage; the two quantities were empirically almost uncorrelated. D3.j-B therefore defines admission and the matching ceiling from the same fragment-damage quantity. Old D3.j-A artefacts are not reinterpreted.

A pass establishes only checkpoint likelihood sensitivity aligned with the declared chemical descriptors. It does not establish that the model uses chemistry in a downstream task, that the sensitivity is a mechanism, or that the model contains biological knowledge.

## Frozen panel and data

The protein arms are `progen2-small`, `progen2-base`, `progen2-medium`, and `zymctrl`. Each arm uses its native tokenizer and rendering. `zymctrl` remains EC-conditioned, so any result on that arm is conditional on the supplied EC label.

The attainability control is `bygpt5-medium-en`. Two independently seeded OpenWebText controls are run, one for each confirmation index. A protein confirmation is unreadable unless its corresponding text control passes.

For each protein arm, the construction cohort contains 4,096 records in the arm's declared 64–246-residue evaluation population. Construction freezes two further 4,096-record confirmation slots under the same corpus-permutation seed at:

- confirmation 1: `construction_skip + 4,096`;
- confirmation 2: `construction_skip + 8,192`.

The stage must verify all three cohorts pairwise. Exact sequence overlap or a shared single-linkage component under 5-mer containment of at least 0.5 returns `VOID`; no replacement seed or offset is tried in this campaign. Construction records each slot's seed, skip, cohort digest, provenance digest, records, and content hashes. Confirmation must reproduce its frozen slot exactly and cannot override the seed or skip.

## Frozen axis and intervention

The chemistry axis is Euclidean distance over z-scored hydropathy, formal charge, side-chain volume, and Grantham polarity. It reads no sequence corpus.

The opposing axis is the UniRef50 order-7 fragment conditional's substitution damage on the construction cohort. For an unordered amino-acid pair, the two directed damages are combined by their arithmetic mean. Both directions must be measurable without smoothing; otherwise that pair is excluded before the contradiction set is formed. The construction artefact freezes the axis matrix, observed mask, pair membership, source digests, tokenizer and rendering identity, and the tercile/quartile/quintile cuts.

The model intervention replaces every input-embedding row read of residue `r` with residue `s`, leaving the output head and all other parameters unchanged. Damage is scored only at the target after a read of `r`; targets equal to `r` or `s` are excluded. Identity replacement, invariant checks, BLOSUM62, and at least eight norm-matched random replacement directions remain mandatory controls.

The tercile cut is the headline. Quartile and quintile are sensitivity readings from the same measured pairs. Every readable cut needs at least eight unordered pairs in both contradiction quadrants.

## Construction and confirmation

Construction is axis-only. It may load the checkpoint to obtain the native tokenizer and rendering, but it does not measure model substitution damage. Its only successful terminal state is `AXIS_CONSTRUCTED` after:

1. arm single-symbol coverage clears 0.99;
2. the declared cut has at least eight pairs per quadrant;
3. the matching fragment conditional predicts the distributional side, with its contrast strictly below zero;
4. construction and both frozen confirmation cohorts are pairwise exact- and near-duplicate-disjoint.

Each confirmation reuses the frozen axis and pair membership without recomputing quantiles. It recomputes the matching fragment damage on the independent evaluation cohort. If the matching ceiling no longer predicts the distributional side, the cell returns `VOID`; the pair set is not changed.

Confirmation 1 and confirmation 2 are separate evidential draws. They are not pooled to rescue a failed or ambiguous draw. A campaign-level positive requires the same positive verdict on both.

## Estimand and uncertainty

For each directed pair, damage is the token-weighted increase in held-out next-token negative log-likelihood. The primary contrast is

`Delta = mean damage(chemically dissimilar / fragment-similar) - mean damage(chemically similar / fragment-dissimilar)`.

The model and matching fragment ceiling retain per-record numerator and count sufficient statistics. Each of 2,000 bootstrap iterations resamples calibrated near-duplicate sequence groups and substituted-symbol groups. The model and ceiling share both draws, so model-minus-ceiling remains paired. The bootstrap reports:

- model `Delta` and its 95% interval;
- matching-ceiling `Delta` and its 95% interval;
- model-minus-ceiling difference and its 95% interval.

Both the sequence-group and substituted-symbol Kish effective counts must be at least eight. A failed unit floor returns `CROSSED_INTERVAL_REFUSED`; there is no token-, position-, or record-level fallback.

## Frozen verdict rules

A confirmation reads `CHEMISTRY` only if all clauses hold:

1. the model point estimate has `Delta > 0`;
2. the paired model-minus-ceiling 95% interval is entirely above zero;
3. model `Delta` is at least twice the positive part of the matching ceiling's `Delta`;
4. model `Delta` exceeds the 95th percentile of the norm-matched random-direction control;
5. all construction, coverage, attainability, digest, independence, and effective-unit gates pass.

`RECOMBINATION` requires the crossed model-Delta interval to lie entirely below zero. A positive point that fails the ceiling is `INSIDE_CEILING`. Any interval that does not separate the accounts is `UNDECIDED`. A refused interval or failed construction invariant is `VOID`, not a negative result.

The arm-level campaign statement requires both confirmations to return `CHEMISTRY`. One positive and one non-positive confirmation is reported as draw dependence. No result is selected by pooling arms, cuts, fragment orders, or seeds.

## Ceiling and sensitivity ladder

Order 7 is the matching admission rung and the only rung that decides D3.j-B. Orders 1–7 are still reported on the fixed pair set. Order 1 must give exactly zero damage and acts as an indexing check. BLOSUM62 remains a second evolutionary-statistics diagnostic, not part of the chemistry axis.

The result is closed as recombination if it lies inside the implemented ceiling. A positive D3.j-B result would remain a candidate requiring a separate causal-use experiment and independent biological validation.

## Execution and outputs

The campaign uses one immutable pinned commit and exact expected JSON basenames. Every accepted JSON must be nonempty, parseable, and accompanied by an atomic SHA-256 sidecar. Pre- and post-run GPU and host-memory snapshots and runtime logs are mandatory. Construction, text controls, and confirmations use separate labels and output directories.

The available four-card allocation is used only for independent cells. The intended schedule is construction and text-control gates first, followed by the four protein confirmations in parallel for index 1, then the four in parallel for index 2. Any failed prerequisite stops its dependent cells rather than replacing them with extra seeds. Runtime is bounded at 24 hours, with the campaign stopped and reported honestly if the two confirmations cannot finish within the allocation.
