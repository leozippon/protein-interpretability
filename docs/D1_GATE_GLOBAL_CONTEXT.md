# Is the residual model-side likelihood information long-range additive sequence context?

This is one independently qualified gate on the flagship pairwise result in [D1_PAIRWISE_EPISTASIS_RESULTS.md](D1_PAIRWISE_EPISTASIS_RESULTS.md). It is qualified on its own terms, and a negative here settles nothing at any other gate.

The question is narrow. The flagship fit found its reproducible model-side signal in the *first-order* likelihood differences of residue-level protein arms — five arms whose three per-seed intervals all exclude zero, at +0.0003 to +0.0173 kcal²/mol² over the matched baseline C+G+T — and found that increment as large or larger on the distance ≥10 separation stratum. It also found that the four-state representation contrast resolves above zero for no arm of 33. A per-position effect that depends on distant sequence content, with no interaction term anywhere, would produce exactly that pattern, and no control in the declared ladder represents it: the sequence/profile control C carries mutation identities, positions, whole-sequence composition, a whole-sequence dipeptide difference and mutation-local profile summaries, and the nuisance G applies one global monotone response to an additive prediction. So this gate asks whether the residual model-side information is longer-range but still primarily additive sequence context. If it is, that is a real and valuable negative about interaction knowledge, and nothing here strains against it.

Separation on this cohort is a sequence-distance definition and not a structural-contact annotation. 99.98% of admitted cycles sit on site pairs the source study selected as suspected interactions, so the ≥10 stratum separates distant pairs from adjacent ones and does not separate contacting pairs from non-contacting ones.

## The declared control block, frozen before any gate fit

Two blocks are declared in `src/transfer/global_context.py`. The content digest of that declaration — the scales, the radii, both coordinate lists and the additivity statement — is

**`b3687416edb17d64108705738398c3d260d52e3ea91f3fd4e723e36f349788e6`**

frozen at 2026-09-24T05:15Z, before the panel was fitted, and written into every fit record this gate produces. `tests/test_global_context_gate.py` fails if the digest in this document is not the digest the code implements.

Both blocks are built from a per-site vector evaluated at each of the cycle's two mutated sites and then summed. A per-site vector is a function of the wild-type background, one site index and that one site's own substituted residue. The residue-level scales are the declared three-property basis already used elsewhere in this repository — Kyte-Doolittle hydropathy, formal side-chain charge at pH 7, side-chain van der Waals volume in cubic angstroms — imported rather than re-tabulated.

| Coordinate group | Width | Content, per mutated site |
| --- | ---: | --- |
| Inside-window scale means | 9 | mean of each scale over the residues within 1, 2 and 5 residues of the site, excluding the site |
| Inside-window substitution products | 9 | the site's own change in each scale, multiplied by that scale's inside-window mean |
| Local composition | 20 | count of each residue within 5 residues of the site, over the background length |
| Distant scale means | 15 | mean of each scale over the whole background excluding the site, and over the residues more than 2, 5, 10 and 20 residues away |
| Distant substitution products | 15 | the site's own change in each scale, multiplied by that scale's distant mean |
| Distant composition | 20 | count of each residue more than 5 residues away, over the background length |
| Outside-set sizes | 4 | size of each distant set over the background length |
| Terminus distances | 3 | residues to the N terminus, to the C terminus, and to the nearer of the two |

The bounded-receptive-field comparator **W** is the first three groups, 38 coordinates, and reads only residues at sequence distance at most 5 from its own mutated site. The global-context block **X** is all eight groups, 95 coordinates, and reads the whole wild-type background. W's coordinates are the leading coordinates of X, so the column space of W is contained in that of X and the two context steps of the ladder are nested.

The window ladder runs to the full background: the distant means include the rung over every residue of the background except the site itself, and the four complement rungs summarise content beyond 2, 5, 10 and 20 residues against cohort lengths of 37 to 72 residues. Amino-acid composition is reported on both sides of the 5-residue boundary; whole-sequence composition is already in C, so what these coordinates add is the position-referenced split of it.

### Why each block is additive by construction

Three properties, and the first is what makes the gate meaningful at all.

The per-site vector cannot read the other mutated site. `site_context_features` receives one site and that site's substituted residue, and it reads context off the **wild-type background**, never a mutant state. A context read off the double mutant would carry the other site's substitution and would be a cross-site term; reading the wild type removes that possibility rather than bounding it.

The block is the plain sum of the two per-site vectors. So under the linear readout every design here uses, a fitted coefficient vector applied to either block equals a sum of two per-site terms with no product, no difference and no other function of both sites, at any range. A long-range additive mechanism is expressible; a pairwise interaction term is not.

Both properties are enforced rather than asserted. The tests check that the block equals the exact sum of its two per-site vectors, that changing the partner site's substitution moves the block by exactly the partner's own per-site difference — including when the partner sits one residue away, inside every declared window — that recombining two site pairs into two different pairs leaves the sum of their blocks unchanged, and that altering background content beyond 5 residues from both sites moves X while leaving W identical.

## What is estimated, on the flagship fit's own support

Nothing is re-extracted and no model inference runs. `scripts/transfer/fit_global_context_gate.py` refits over the retained label-free extraction artefacts, assembling its rows and every other block through the admitted fit script's own `build_panel` rather than a second implementation, and refusing any run whose row-identity or fold-identity digest differs from the admitted per-arm fit it must compose with. Weighting, folds, the ridge recipe with its penalty grid and tie rule, the nonlinear-additive nuisance, the group-equal squared error and the group bootstrap are imported from `src/transfer/pairwise_epistasis.py`.

Six control sets, each extending the flagship's matched baseline C+G+T: `CGT`, `CGTW`, `CGTX`, and the same three with the two constituent first-order likelihood differences added (`CGT_M1`, `CGTW_M1`, `CGTX_M1`). The likelihood interaction contrast M is added over each, giving twelve designs and twelve declared contrasts in three families — what each context block adds on its own, the first-order likelihood gain over each context baseline, and the interaction gain over each context baseline with and without the first-order control in it. The projected representation contrast resolved above zero for no arm of 33 in the flagship fit and is outside this gate's scope.

A diagnostic on a different estimand accompanies these: the held-group weighted R² with which each context baseline reconstructs the model's own first-order likelihood differences. That is model-score approximation, not phenotype prediction, and it is reported as such.

Five outer and four inner held-group folds, outer seeds 20260923, 20260924 and 20260925, inner seed the outer seed plus 100 plus the outer-fold index — the flagship's. Groups weighted equally, site pairs equally inside a group, cycles equally inside a site pair. Feature centring and scaling fitted on training rows inside the ridge recipe; no held-out group label reaches feature scaling, penalty selection or the nuisance calibration, which the tests check by permuting one held fold's cycle and absolute-stability labels and requiring every held-out prediction of every design to be unmoved.

The independent unit is the site pair, never the cycle count: 217 site pairs on the 64-group support and 163 on the 45-group support where baseline Q is admissible. Under the weighting these fits use — each group equal, each site pair equal inside a group — the Kish effective counts are **129.2** and **96.1** site pairs. The readiness record's 121.6 and 88.4 are the same statistic under a cycle-share weighting, which is not the weighting any fit here or in the flagship uses; both conventions are computed and pinned by the tests, and they differ by about 6%.

That composability is measured rather than asserted. Across every arm and every seed, the four designs this gate shares with the admitted pairwise fit — C+G+T, C+G+T with the likelihood interaction, C+G+T with the first-order differences, and that one with the interaction added — reproduce the admitted group-equal squared error to at most **6.7e-16 kcal²/mol²**, and every arm reproduces the admitted row-identity and fold-identity digests exactly on both unfiltered supports. The fit refuses to proceed otherwise.

One detail of that check is worth recording, because it cost a run. The admitted fit's row-identity digest was computed over a string embedding the numpy scalar repr, and numpy 2.1 renders a float64 as `np.float64(x)` where numpy 1.26 renders `x`. The same 8,192 rows therefore hash to `11a289e3…` under numpy 1.26, which is where the admitted digests were produced, and to `8f278b21…` under the numpy 2.1.2 the pods now carry — both measured. This gate renders the target as a Python float, which reproduces the admitted digest in both environments. The pairwise-epistasis record has since made the same change in `src/transfer/pairwise_epistasis.py:row_identity`; a later run of this gate should import that function instead of keeping the expression local, which is not done retrospectively here because the 33 records already bind the digest of the script that produced them.

### The insertion-construct exclusion, applied as a declared filter

An insertion or deletion construct's sequence is truncated to the wild type's length in the pinned MegaScale bytes, so such a row is length-matched to the substitution states and entered the frozen cohort as though it were a substitution. The cohort is not re-frozen here, because several gates reference its digest as their declared support; the exclusion is declared over it instead, by `scripts/transfer/declare_pairwise_indel_exclusion.py`, and applied as a third support beside the two unfiltered ones.

Measured on this support: 5 of 12,977 states carry at least one insertion-construct row, in `1O6X.pdb`, `2AMI.pdb` and `2D1U.pdb`. Three of the five carry nothing else and are absent once the rule is applied; the other two keep the median of their remaining rows, which moves by 0.0107 and 0.0315 kcal/mol. The three dropped cycles are each the only cycle of a site pair at positions 1 and 2 — an inserted residue at the first or second position read as a substitution at both — so the 64-group support goes from 8,192 cycles and 217 site pairs to 8,189 and 214, with the 1–2 stratum losing 3 of its 25 site pairs, and the 45-group support from 5,760 and 163 to 5,757 and 160. No group is lost. The second corrected state is `2AMI.pdb`'s wild type, so all 127 cycles of that background keep a target moved by at most 0.0315 kcal/mol, all of them at separation ≥10. Kish effective site pairs under the fitted weighting move from 129.2 to 127.5 on the 64-group support and from 96.1 to 94.2 on the 45-group one.

This sensitivity is a refit: the rows change, so its row identity and its increments are not the admitted fit's, and it is reported beside the unfiltered numbers rather than in place of them. It is a different operation from the restriction the pairwise-epistasis record reports, which drops the three affected backgrounds from the evaluation of the untouched admitted fits and therefore runs on 63 and 44 groups; this one corrects the two contaminated medians, drops only the three contaminated cycles, and refits on 64 and 45 groups.

## What the two context blocks add on their own

Neither block carries a model quantity, so these steps differ between arms only through the 16 tokenisation descriptors the flagship's matched baseline contains, which are arm-specific by construction. Entries are the median arm's value with its interval, and the across-arm spread of the point estimates is given beside it. All entries are paired reductions in group-equal epsilon mean squared error in squared kcal/mol on the 64-group support, with 2,000-draw group-bootstrap percentile intervals over the 64 held groups.

| Step | 20260923 | 20260924 | 20260925 | across-arm spread |
| --- | --- | --- | --- | --- |
| W over C+G+T | −0.00269 [−0.00607, +0.00093] | −0.01033 [−0.01745, −0.00401] | −0.00261 [−0.00688, +0.00167] | 2.4e−4 to 2.0e−3 |
| X over C+G+T | +0.00434 [−0.00391, +0.01268] | +0.00322 [−0.00540, +0.01200] | −0.00051 [−0.01106, +0.00989] | 4.5e−4 to 6.4e−4 |
| X over C+G+T+W | +0.00703 [−0.00095, +0.01492] | +0.01354 [+0.00084, +0.02654] | +0.00211 [−0.00799, +0.01124] | 3.0e−4 to 1.6e−3 |

No arm of the 33 has all three per-seed intervals on one side of zero for any of these three steps. The across-arm spread of the underlying designs is larger than the spread of the increments — C+G+T itself varies by 0.00546 kcal²/mol² across arms at one seed, and C+G+T+X by 0.00532 — because the tokenisation block's contribution is arm-specific and mostly cancels in the difference.

The bounded block lowers the baseline's accuracy at all three seeds and resolves below zero at one of them: 38 added coordinates that read only a ±5-residue window cost this predictor accuracy on 64 held groups rather than buying any. The global-context block is positive at two of three seeds and negative at the third, and its interval crosses zero at every seed. Over the bounded block it is positive at all three seeds and resolves above zero at one.

By separation stratum at seed 20260923, X over C+G+T reads +0.00536 [−0.01108, +0.02348] on 24 groups and 25 site pairs at separation 1–2, −0.00668 [−0.01832, +0.00346] on 36 groups and 61 site pairs at 3–9, and +0.00838 [−0.00215, +0.02000] on 53 groups and 131 site pairs at ≥10. The largest point estimate is on the distant stratum, which is where a long-range additive mechanism should show, and no stratum's interval excludes zero.

**This bounds the gate's own strength and must be read before its verdict.** The control this gate declares is measurably informative only at the same order of magnitude as the model increments it is meant to absorb, +0.003 to +0.004 kcal²/mol² against an additive null of 1.15441, and at 64 held groups and a Kish effective 129.2 site pairs its own contribution is not resolved above zero at all three seeds. A first-order gain that survives it therefore survives a control of demonstrated but unresolved strength. The ridge penalty selected for the 980-column C+G+T+X design is 10.0 in every outer fold at every seed, one step below the grid's strongest value, so the design is not running out of shrinkage in the way the flagship's 1,024-column representation block did.

A consequence for reading the bounded comparator's rows below: because W lowers its own baseline, an increment measured over C+G+T+W is measured over a weaker baseline and is inflated accordingly, most visibly at seed 20260924 where W costs 0.01046. Every bounded-comparator number below is reported beside W's own effect for that reason.

## How much of the first-order likelihood gain survives

The flagship's five resolved arms, before and after the global-context block, on the 64-group support. Entries are paired reductions in group-equal epsilon mean squared error in squared kcal/mol; the first row of each arm reproduces the admitted pairwise fit's own published value.

| Arm | Baseline | 20260923 | 20260924 | 20260925 | Distance ≥10, 20260923 |
| --- | --- | --- | --- | --- | --- |
| proteinglm-7b-clm | C+G+T | +0.00812 [+0.00373, +0.01287] | +0.00770 | +0.00808 | +0.00989 [+0.00399, +0.01734] |
| | C+G+T+X | +0.00792 [+0.00365, +0.01264] | +0.00765 | +0.00810 | +0.00933 |
| progen2-xlarge | C+G+T | +0.00700 [+0.00331, +0.01115] | +0.00674 | +0.00774 | +0.00926 |
| | C+G+T+X | +0.00699 [+0.00336, +0.01104] | +0.00693 | +0.00776 | +0.00896 |
| progen2-medium | C+G+T | +0.00426 [+0.00116, +0.00760] | +0.00338 | +0.00457 | +0.00655 |
| | C+G+T+X | +0.00415 [+0.00110, +0.00751] | +0.00335 | +0.00462 | +0.00635 |
| progen3-3b | C+G+T | +0.00391 [+0.00164, +0.00652] | +0.00386 | +0.00378 | +0.00380 |
| | C+G+T+X | +0.00376 [+0.00149, +0.00637] | +0.00392 | +0.00395 | +0.00352 |
| progen2-large | C+G+T | +0.00349 [+0.00074, +0.00647] | +0.00288 | +0.00389 | +0.00544 |
| | C+G+T+X | +0.00349 [+0.00066, +0.00656] | +0.00294 | +0.00408 | +0.00542 |

Across the five arms the seed-mean gain after the block is 0.99 to 1.02 times the gain before it, and on the distant stratum 0.96 to 1.01 times. All five keep three per-seed intervals above zero after the block, and all five keep them above zero on the distance ≥10 stratum.

At panel level the count is unchanged. Five arms of 33 resolve above zero at all three seeds over C+G+T and the same five over C+G+T+X — progen2-large, progen2-medium, progen2-xlarge, progen3-3b and proteinglm-7b-clm. One arm resolves below zero before the block and none after. On the distance ≥10 stratum seven arms resolve above zero before and six after, the loss being rita-xl. The tokenisation-stratum medians of the seed-mean increment move from +0.00109 to +0.00093 for the eleven amino-acid arms, with 10 of 11 positive in both cases, from −0.00012 to −0.00006 for the three byte arms and from −0.00026 to −0.00024 for the nineteen BPE arms.

The bounded comparator behaves the same way — the five arms stay resolved over C+G+T+W, and protgpt2 joins them — but that sixth arm is an artefact of the comparator weakening its own baseline: W costs 0.01033 kcal²/mol² at seed 20260924, and every increment measured over it is measured over a worse predictor.

**So a bounded-receptive-field predictor and a full-context one absorb the same share of the first-order gain, which is between nothing and about four percent.** The information is not local; it is also not the long-range additive content this block represents.

## The likelihood interaction after the same block

The three arms the flagship reports as resolved for the likelihood interaction over its own first-order control stay resolved, at values that move in the fourth decimal place.

| Arm | Baseline | 20260923 | 20260924 | 20260925 |
| --- | --- | --- | --- | --- |
| prollama | C+G+T+M1 | +0.00095 [+0.00032, +0.00164] | +0.00090 [+0.00031, +0.00154] | +0.00101 [+0.00040, +0.00170] |
| | C+G+T+X+M1 | +0.00103 [+0.00039, +0.00173] | +0.00095 [+0.00035, +0.00160] | +0.00110 [+0.00047, +0.00182] |
| progen3-3b | C+G+T+M1 | +0.00092 [+0.00024, +0.00168] | +0.00080 [+0.00010, +0.00155] | +0.00090 [+0.00021, +0.00165] |
| | C+G+T+X+M1 | +0.00094 [+0.00026, +0.00171] | +0.00082 [+0.00012, +0.00158] | +0.00096 [+0.00029, +0.00171] |
| protgpt2 | C+G+T+M1 | +0.00058 [+0.00025, +0.00093] | +0.00046 [+0.00000, +0.00089] | +0.00075 [+0.00041, +0.00112] |
| | C+G+T+X+M1 | +0.00058 [+0.00023, +0.00095] | +0.00047 [+0.00004, +0.00091] | +0.00075 [+0.00039, +0.00114] |

The same three arms resolve above zero for the interaction over the context baseline without the first-order control, before and after the block. Two arms resolve below zero in both readings, galactica-1.3b and gpt2-large. So the additive global block absorbs none of this increment either — it is 0.00005 kcal²/mol² larger after the block for prollama and unchanged for the other two.

## Can the block reconstruct the model's own first-order score?

This is the model-approximation estimand and it is reported as a diagnostic, not as phenotype prediction. Predicting each constituent first-order likelihood difference from a context baseline on held groups, with the same folds and weights, gives a within-background rank correlation whose panel median is +0.1519 for C+G+T and +0.1675 for C+G+T+X; the median arm gains +0.0050 and the largest gain on any arm is +0.0305. The five first-order arms sit at +0.182 to +0.319 for C+G+T and +0.185 to +0.329 for C+G+T+X — proteinglm-7b-clm, the largest, reads +0.31895 [+0.25385, +0.38148] before and +0.32937 after.

The group-equal weighted R² of the same predictions is far below zero for those five arms, between −18.4 and −10.1, and between −0.4 and −1.0 for the text arms. A held-group prediction of a protein arm's likelihood score is therefore worse than that group's own mean in squared error while still ordering the variants inside a background at a rank correlation near +0.3: the sequence-statistical blocks recover part of the ordering and nothing of the scale, and adding the long-range additive coordinates moves the ordering by about +0.005.

## The two other supports

On the 45 groups and 163 site pairs where baseline Q is admissible, two arms resolve above zero for the first-order gain over C+G+T — progen2-xlarge and proteinglm-7b-clm — and the same two over C+G+T+X. For the interaction after the first-order control, prollama and qwen2.5-0.5b-instruct resolve above zero before the block and prollama alone after it, and dialogpt-small resolves below zero before and not after. This support carries 96.1 Kish effective site pairs against 129.2, and single-seed swings of 0.01 to 0.02 kcal²/mol² appear on it that do not appear on the full support, so its disagreements are within its own noise.

Under the declared insertion-construct exclusion, on 8,189 cycles and 214 site pairs, three arms resolve above zero for the first-order gain before the block and the same three after it — progen2-xlarge, progen3-3b and proteinglm-7b-clm — with progen2-medium and progen2-large losing their third seed's lower bound. For the interaction after the first-order control, prollama resolves above zero before and after. The exclusion moves the first-order increment by a median 0.000134 and at most 0.000779 kcal²/mol² across the 33 arms and three seeds. That loss of two arms is the same one the pairwise-epistasis record reports from its own cycle-level restriction of the untouched fits, reached here by a refit on corrected medians, so two different operations on the same contamination agree on which arms are brittle.

## Verdict

**The gate is answered, and the answer is that longer-range but additive sequence context does not explain the residual model-side likelihood information.** The declared block absorbs between nothing and about four percent of the first-order likelihood gain, on the full support and on the distance ≥10 stratum where the alternative would have had to act, and it leaves the five resolved arms resolved and the panel count at five of 33. It absorbs none of the likelihood interaction increment: the three arms that resolve above zero over their own first-order control keep doing so, at values differing in the fourth decimal place. The bounded-receptive-field comparator absorbs the same negligible share, so the information is not local either.

**What that does and does not license.** It licenses the narrow statement that this model-side signal is not in the linear span of the declared long-range additive representation — three physicochemical scales and one composition split, summed over the two mutated sites, at window radii 1, 2 and 5 residues, the full background and the complements beyond 2, 5, 10 and 20 residues. It does not license calling the signal pairwise: a first-order likelihood difference is a single-mutant quantity, and the two other gates in this map ask the questions this one cannot.

**And the negative is bounded by the control's own strength.** The block's contribution to the endpoint is not resolved above zero at all three seeds on its own, and its contribution to reconstructing the model's first-order score is about +0.005 rank correlation. A control that weak cannot exclude the alternative it represents with much force, and the honest reading is that the alternative was implemented, measured and found to absorb nothing, rather than that long-range additivity has been ruled out. A stronger version of this gate would need a representation of distant content richer than three scales, a nonlinear per-position response, or more than 129 effective site pairs.

## What is retained, and what this gate does not measure

This gate is declared over model likelihood only. `GATE_ADDITIONS` contains the likelihood interaction and nothing else, no design here reads the projected representation blocks, and the representation contrast is therefore not measured rather than measured and reported. A likelihood enters as a scalar column, so a linear readout of it spans nearly the whole function of that scalar, which is why these contrasts are reported as final while a representation-level verdict would not be.

For a later representation-side revisit, everything needed is retained and nothing has to be re-extracted. The flagship's full-width per-state block outputs sit beside the projected ones under `results/pairwise_epistasis_20260924/extraction/<arm>/`, so a different projection or a wider readout can be replayed without model inference. This gate's 33 per-arm records under `results/gate_global_context_20260924/fits/` carry, per support and per seed, the held-group membership of every outer fold, the selected ridge penalty of every design and every reconstruction fit, the design widths, the nuisance diagnostics and the declaration digest; the panel report and a local copy of the records are under ignored `logs/d1_gate_global_context_20260924/`. The blocks X and W are a function of the extraction plan alone, so they can be rebuilt from `src/transfer/global_context.py` without reading any measurement.

## Execution

Fits ran in-pod on CPU lanes with no model loaded, no forward pass and no GPU: 33 arms across four allocations at eight concurrent processes and eight threads each, from the frozen working-tree snapshot `20260923231105_9094cd33e85a`, with the lane assignment in `scripts/transfer/campaign_d1_gate_global_context.tsv` and a host resource receipt per lane beside the outputs. All 33 cells exited 0 with `# FAILURES 0` in all four lanes. Wall clock ran from about 17 minutes for the first lane's eight arms to about three hours for the lane whose pod carried another campaign's load average of 345 to 427 on 192 cores; no card was claimed and no idle threshold was touched. Aggregation ran on Compute in the `ct` environment. The panel's own conditions are in the report: one row identity and one fold identity per support and seed across all 33 arms, the shared designs reproducing the admitted fit to 7.8e-16 kcal²/mol², the first-order increment reproducing the admitted value to 2.4e-16, and the nuisance control re-deriving the admitted fit's median held-out weighted R² of 0.141 over 15 folds with an isotonic response spanning a median 2.60 kcal/mol.

## Limitations that bound this gate

- The blocks are additive *and* linearly read. A long-range mechanism that is additive across sites but nonlinear in its own context — a saturating dependence on distant hydrophobic content, say — is only partly representable here. The nuisance G supplies one global monotone response to an additive prediction; it does not supply a per-position nonlinear response to context. A gain that survives X therefore survives a linear long-range additive representation, not every additive one.
- Distant sequence content enters through three physicochemical projections and one composition split at 5 residues. A dependence on distant content orthogonal to hydropathy, charge, volume and that composition split is not represented, and the three scales were declared before this gate rather than chosen to maximise absorption.
- Reading context off the wild-type background is what makes the blocks additive, and it is also what they cannot express: a mechanism in which the two substitutions jointly change the context each other sees is excluded by construction. That exclusion is the point of the gate, not an oversight, but it means a surviving gain is not thereby localised to anything.
- The comparator's receptive field is bounded in its own coordinates only. Every design here extends C+G+T, and C already carries whole-sequence composition and a whole-sequence dipeptide difference, so "bounded receptive field" describes the added block and not the whole predictor.
- Power is set by 217 site pairs and a Kish effective 129.2 on the 64-group support, and by 163 and 96.1 on the 45-group one, never by the 8,192 and 5,760 cycle counts. Intervals are group-bootstrap percentile intervals conditional on the fitted cross-validation predictions; they omit training and split variation and are not adjusted for 33 arms, three supports, three seeds and twelve contrasts of unadjusted pointwise intervals.
- The reconstruction diagnostic is on the model-approximation estimand and carries its own readout bound: an 885-to-980-column ridge fitted on about 51 training groups. A negative weighted R² there bounds that readout's cross-family transfer, and says nothing about whether a different predictor could reproduce the score.
- Separation is sequence distance. 99.98% of admitted cycles sit on site pairs the source study selected as suspected interactions, so the ≥10 stratum is not a non-contact stratum and none of these numbers estimates a population value over protein position pairs.
- No inference runs here, so every precision caveat of the retained artefacts carries over unchanged, including that the 0.001-nat and 0.001-relative-L2 batch gates are inapplicable rather than passed at batch size one for all 20 batch-one arms of the panel. No arm is numerically validated on that basis.
- A gain that survives this gate is not evidence of interaction knowledge. The first-order likelihood difference is a single-mutant quantity; surviving an additive long-range control means the information is not in this declared additive representation, not that it is pairwise. A gain that is absorbed is the more definite result, and it bounds the model-side signal to what long-range additive sequence context already supplies.
