# Readout-class sweep on the frozen readout states

The [readout panel](D1_READOUT_RESULTS.md) reported a panel-wide near-null on its primary endpoint: across 34 arms, 31 place no resolved additional mutation-effect information in their frozen pooled states beyond the profile and sequence descriptors, and the largest resolved anchor increment is +0.02255 in Spearman. That null is bounded by the readout class, not by the representation. This experiment varies the readout class on the same frozen states, holding the cohort, the assay supports, the panels, the split seeds, the fold maps, the resampling unit and the resampling contract fixed, and asks whether the near-null survives.

The retained representation vectors permit that replay without any model inference. This is a refitting workload over existing arrays, and it runs on CPU.

## What the retained artifacts support, and what they do not

Each retained array holds the four mutant-minus-wild-type block summaries at full hidden width, 384 to 7,168 coordinates per block, together with the wild-type state, the native likelihood, the measured effects, the profile scores and the ordered mutation identifiers. Two parts of a fuller sweep are therefore not available and are not replaced by a proxy:

- **A per-layer depth sweep cannot be fitted.** The extraction hooked exactly two blocks per arm, the block after half the stack and the last block, recorded in each manifest as its two zero-based indices. No other layer's state was retained, so depth can be resolved at two points and no more. This experiment reports those two points, crossed with the two pooling rules, and states the resolution as two depths rather than a sweep.
- **Position-resolved features cannot be fitted.** Only the target-residue-token mean and the last residue-token state were retained; no per-position state exists in the artifacts. A position-resolved readout would require fresh model forward passes over the whole cohort, which this execution does not perform. The mean-versus-last-token contrast below is a two-point pooling contrast, not position resolution.

Both bounds are properties of the admitted extraction, not of this analysis, and neither is removed by the classes below.

## Declared class set

Declared and frozen in `src/transfer/readout_class_sweep.py` and in the campaign manifests before any fit was inspected. Every class fits three predictors on identical rows: the matched supervised baseline B, the representation alone R, and B+R. B is the 446 admitted baseline features — the within-assay standardized ranks of the native likelihood M and the external profile score P, and the 444 sequence descriptors. R is the representation at the class's own resolution.

| Class | Head | Representation coordinates | Penalty grid | Capacity selected in inner folds |
| --- | --- | --- | --- | --- |
| C0 | ridge | 1,024 fixed Gaussian projection | 5-point admitted grid, 1e-2 to 1e2 | penalty |
| C1 | ridge | 1,024 fixed Gaussian projection | 9-point grid, 1e-2 to 1e6 | penalty |
| C2 | ridge | uncompressed, 4 x hidden width, 1,536 to 28,672 | 9-point grid | penalty |
| C3 | ridge on linear block plus random features | 1,024 projection, mapped from the same 1,024 | 9-point grid | penalty and bandwidth |
| C4 | ridge on linear block plus random features | 1,024 projection, mapped from the uncompressed 4 x hidden | 9-point grid | penalty and bandwidth |
| D | ridge | one depth-and-pooling block alone, hidden width | 9-point grid | penalty |

C0 is the admitted class and its admitted grid. It exists to prove the pipeline is matched: every cell refits it and compares the result against its own admitted report — the assay support, the outer and inner cluster fold membership, the selected penalties, the per-assay raw likelihood and profile correlations, and the per-variant held-out predictions of B, R, B+R and both label-permutation controls. A cell whose support, folds or selected penalties differ fails immediately; a cell records its worst absolute prediction deviation, and the panel gate refuses the whole panel if any cell exceeds 1e-8, the tolerance the admitted final admission receipt used for the same comparison.

C1 to C4 are the {linear, random-feature} x {compressed, full-width} factorial. The penalty grid is extended for them because a design of up to 28,672 standardized columns needs a penalty range the five-point grid cannot express; its upper end, 1e6, already exceeds the largest attainable eigenvalue of the weighted correlation matrix by a factor above 30, so the grid brackets the near-constant predictor. The random-feature head is the smallest defensible nonlinear class: 2,048 random Fourier features approximating a radial-basis kernel, with each input group standardized on the fitting rows alone and divided by the square root of its own width so that the baseline and representation groups contribute comparably to the kernel argument, and with one bandwidth multiplier from {0.5, 1, 2, 4} selected jointly with the penalty inside inner folds. The nonlinear designs retain the compressed coordinates in their linear block, so each contains the compressed linear class of the same predictors; neither contains the full-width linear class, and the four classes are reported separately rather than as one nested sequence.

The D classes resolve the two extraction depths and the two pooling rules separately: `middle_mean`, `middle_last`, `final_mean`, `final_last`, each fitted alone and alongside B. They run at the primary split seed, matching the convention the admitted study uses for its sensitivities.

## What is held fixed

Everything the admitted study fixed, imported from its own modules rather than restated: the cohort and its digest, the assay support of each of the 17 panels, the three split seeds 20260923, 20260924 and 20260925, the five outer and four inner cluster folds from the same seeded deterministic splitter with the inner seed equal to the outer seed plus 100 plus the zero-based outer-fold index, the within-assay standardized average-rank targets, equal weight per wild-type cluster and per assay within a cluster and per variant within an assay, training-only weighted feature means and scales, the unpenalized intercept, the weighted mean-squared-error objective with the squared coefficient norm, ties broken toward the stronger penalty, 2,000 paired bootstrap draws at seed 20260923 and alpha 0.05 over the unit *wild-type family at 50% identity*, and the fixed within-assay label permutation at seed 20269923 used for fitting and inner tuning and evaluated against the untouched held-out effects.

The only change to the linear algebra is the solver: one Cholesky factorization per penalty in place of one symmetric eigendecomposition for the whole grid, because on this allocation a 16,384-column eigendecomposition costs about seventy Cholesky factorizations of the same matrix while the full-width designs reach 29,118 columns. The objective, weighting, standardization and returned prediction are unchanged, the equivalence is asserted by test to 1e-8 absolute and relative, and on real designs it is demonstrated by the C0 reproduction in every cell.

## Endpoints

Per arm, per panel, per split seed and per class, on the same paired assays and the same resampling unit as the admitted report:

- the primary increment, the cluster-equal mean within-assay Spearman difference between B+R and B in that class;
- the representation-over-likelihood contrast, R minus the raw native likelihood in that class;
- the paired rank-MSE reduction, MSE(B) minus MSE(B+R) in that class;
- the fitted Spearman of B, R and B+R, and the label-permutation controls at the primary seed.

Strata are never pooled. `literal_text_AA` carries the falsification control: in the admitted panel that stratum shows the representation-over-likelihood contrast resolved positive in all 42 cells, from +0.06975 to +0.17027, while the increment over the matched baseline is resolved *below* zero in all 42. If a stronger class produces increments on that stratum too, the increments are not protein-specific and are reported as such.

## Execution

132 cells, one per admitted (arm, panel, split seed) fit cell, dispatched as six CPU lanes of the in-pod campaign queue over the shared filesystem where the retained arrays live. No cell loads a checkpoint or runs a forward pass. Cells are ordered longest-first by a declared cost model over the fitted design dimensions and the panel row counts; that model reads the retained hidden widths only and no measured effect or fitted outcome.

## Results

Pending. This section is written only after the campaign completes and its cells pass their own C0 reproduction checks.

## Limitations carried from the admitted panel

Batch-one requalification is structurally vacuous for all 20 batch-one arms of the 34, not only the five re-extracted ones: at batch size one the check's reference forward and the production forward are the same single-row computation, so the exactly-zero drift is true by construction. Those arms are admitted and internally consistent, not numerically validated. ProGen3 is singleton-only, with measured batch dependence of 0.02464 nats per scored token at batch size two and up to 0.07179 across sixteen equal-length rows, and no kernel above float16; both ProGen3 arms ran at batch size one. Wild-type clustering at 50% identity does not guarantee remote-superfamily disjointness or exclude pretraining overlap. The three split seeds are robustness checks on one biological cohort, not independent replications. Checkpoint lineages inside the panel are correlated, so counts of resolved intervals do not estimate a model-population prevalence. Intervals condition on the fitted cross-validation predictions, omit training, tuning and split variation, and are not adjusted across arms, classes, endpoints or seeds.
