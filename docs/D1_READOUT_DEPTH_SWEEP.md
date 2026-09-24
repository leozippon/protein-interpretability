# Depth-resolved sweep of the frozen readout

The [readout panel](D1_READOUT_RESULTS.md) reported a panel-wide near-null on its primary endpoint: across 34 arms, 31 place no resolved additional mutation-effect information in their frozen pooled states beyond the profile and sequence descriptors, and the largest resolved anchor increment is +0.02255 in Spearman. The [class sweep](D1_READOUT_CLASS_SWEEP.md) varies the fitted function on those same retained states and records the one bound it cannot pass: the admitted extraction hooked exactly two blocks per arm and retained only the target-residue-token mean and the last residue-token state, so depth is resolved at two points and position-resolved features do not exist.

This experiment lifts that bound by re-extracting. Capturing every transformer block's hidden state costs the same single forward pass per sequence as capturing two; the additional cost is storage and fitting, not inference. The question is whether any depth places resolved information beyond the matched supervised baseline where the two pooled depths did not, and if so where in depth — which is also the quantity a mechanism study needs.

## What is declared, and when

The roster, the depth grid, the projection seeds, the fitted designs, the reproduction controls and the campaign manifests were all frozen before any fit was inspected. Their digests are recorded in the [experiment log](EXPERIMENT_LOG.md) alongside the frozen snapshot identifier.

**The roster selects nothing.** It is every arm of the admitted expansion roster — all 34 extraction arms, on all 17 admitted panels, at every split seed each panel declares, for 54 fitted (arm, panel) cells covering the same 132 admitted (arm, panel, seed) fit cells. Choosing arms by their admitted readout outcome would bias a depth sweep toward or away from finding information, so no arm is chosen at all.

**The depth grid is every block.** For an arm with *L* transformer blocks the grid is the zero-based block indices 0 through *L*−1, from 4 blocks for ByGPT5-small-en to 64 for Qwen2.5-32B, 1,004 blocks over the 34 arms. The two admitted indices, ⌈(*L*−1)/2⌉ rounded down and *L*−1, are members of that grid, which is what makes the reproduction control below possible.

**Position resolution is an interface property, not a choice.** A mutant-minus-wild-type difference at one grid position compares the same residue index in both renderings only when the rendering assigns exactly one residue-bearing token per residue. That holds for 19 of the 34 arms — the five ProGen2 sizes, both ProGen3 sizes, ProteinGLM-7B-CLM, ProtGPT3-1.3B, RITA-XL, the four Galactica sizes, InstructProtein, the three byte-level ByGPT5 arms and ZymCTRL — and fails for the 15 arms whose tokens carry several residues: the Llama-2-7B parent, both ProLLaMA stages, ProtGPT2's FASTA wrapping, and the eleven byte-pair literal-text controls. The 15 excluded arms keep the full depth grid under the two pooled rules; they receive no position-resolved substitute, because none is defined. The rule is checked mechanically on the first eligible assay before extraction and then enforced on every row.

## What is captured

One forward pass per sequence, with the same cohort, native rendering, scoring span, marker exclusions, precision setting, batch size and batch composition as the admitted extraction. At every block:

| Summary | Content | Arms |
| --- | --- | --- |
| `mean` | Arithmetic mean over the target's residue-bearing tokens | 34 |
| `last` | The last residue-bearing token's state | 34 |
| `mut` | Mean mutant-minus-wild-type difference over the substituted residues' own tokens | 19 |
| `suffix` | That mean over every residue token after the last substituted one | 19 |

`mean` and `last` are the admitted pooling rules, retained as absolute values so the mutant-minus-wild-type difference is formed exactly as the admitted artifacts form it. `mut` and `suffix` are retained as differences, because the wild-type reference depends on which residues a given mutant substitutes. The complementary prefix difference — every residue token before the first substituted one — is provably zero for a causal model, since no earlier position attends to a later one, and is retained as a per-variant per-block maximum absolute value rather than as a feature. A variant whose last substitution is at the final residue has no suffix; its `suffix` is the zero vector and such variants are counted.

Storage is written uncompressed with one array per depth, so a fit streams one depth at a time rather than holding a whole stack. Each per-assay archive also carries the native mutant-minus-wild-type likelihood, the wild-type state and likelihood, the ordered mutation identifiers, the measured effects, the profile scores and the batch repeat-check quantities, under one bound measurement identity.

## What is held fixed

Everything the admitted study and the class sweep fix, imported from the admitted modules rather than restated: the cohort and its digest, the assay support of each of the 17 panels, the three split seeds 20260923, 20260924 and 20260925, the five outer and four inner wild-type-cluster folds from the same seeded deterministic splitter with the inner seed equal to the outer seed plus 100 plus the zero-based outer-fold index, the within-assay standardized average-rank targets, equal weight per wild-type cluster and per assay within a cluster and per variant within an assay, training-only weighted feature means and scales, the unpenalized intercept, the five-point ridge grid with ties broken toward the stronger penalty, and 2,000 paired bootstrap draws at seed 20260923 and alpha 0.05 over the unit *wild-type family at 50% identity*. The fitting procedure is the admitted `nested_predict`, called unchanged.

The matched supervised baseline B is built from the **admitted** likelihood, profile score and sequence descriptors, so it is byte-identical to the baseline the admitted panel reports and every depth increment is measured against exactly that baseline. The re-extracted likelihood is compared against the admitted one and gated at the admitted 0.001-nat limit.

## The fitted designs

Each design is the representation alone, R, and the baseline plus the representation, B+R, against the shared baseline B. Every projection is Gaussian with entries of standard deviation 1/√*d*, seeded by a declared rule that depends only on the base seed, the depth index and the summary index — never on a label, an observed vector or a fitted outcome.

| Axis | Representation | Coordinates | Base seed | Seeds fitted |
| --- | --- | ---: | ---: | --- |
| `admitted` | The admitted four blocks at the two admitted depths | 1,024 | 20260923 | all declared |
| `depth`*d* | `mean` and `last` at depth *d* | 512 | 20261123 | all declared |
| `wide`*d* | `mean` and `last` at depth *d* | 1,024 | 20271123 | primary only |
| `union` | `mean` and `last` at every depth | 64·*L* | 20281123 | primary only |
| `pos`*d* | `mut` and `suffix` at depth *d* | 512 | 20261123 | all declared |
| `full`*d* | all four summaries at depth *d* | 1,024 | 20261123 | primary only |

`depth` is the primary axis and carries the same per-block compression as the admitted study, so the only thing changing along it is depth. `wide` repeats it at the admitted total representation width, so an unresolved increment at one depth cannot be read as a consequence of halving the coordinate budget. `union` supplies every depth at once, so information spread across depths is not missed by resolving one depth at a time. `pos` and `full` add the position-resolved summaries where the rendering defines them. The fixed within-assay label permutation at seed 20269923 runs at the primary seed for the baseline and the admitted class, as the admitted study runs it.

## The reproduction control

Nothing new is reported by a cell until it has recovered the admitted numbers twice.

The **pipeline-identity** control refits the admitted class from the admitted retained states through this experiment's own code, at the primary split seed, and requires the held-out predictions of B, R and B+R to agree with the admitted report to at most 1e-8 absolute — the tolerance the admitted final admission receipt used for the same comparison. It also requires identical support, identical outer and inner cluster fold membership and identical selected ridge penalties. A cell that fails it raises rather than continuing.

The **extraction-agreement** control compares the re-extracted pooled summaries at the two admitted depths against the admitted retained arrays, as a per-variant relative L2 over the concatenated four blocks in the same form as the admitted precision gate, and refits the admitted class from the re-extracted states at every declared seed, binding it to the same admitted report by the same equalities. A deviation here is a statement about the reproducibility of the forward pass, not about the fitting code, and the two controls are reported separately so the distinction survives.

## Results

Pending. This section is written only after the campaign completes and its cells pass both reproduction controls.

## Limitations carried and added

A null at every depth remains bounded by the readout class, the pooling and position rules, the label budget and the cohort, and does not establish that biological information is absent. The class here is one fixed compressed linear family; the class sweep varies that axis separately, and the two axes compose but are not fitted jointly. The position-resolved summaries exist for 19 arms and cannot be defined for the other 15, so a comparison across that boundary confounds interface with position resolution. The label budget is 163 wild-type clusters on the anchor panel and 30 on the EC-conditioned panel, whose intervals are correspondingly about twice as wide. Wild-type clustering at 50% identity does not guarantee remote-superfamily disjointness or exclude pretraining overlap. The three split seeds are robustness checks on one biological cohort, not independent replications, and the minimum-to-maximum point ranges reported with them are not uncertainty intervals. Checkpoint lineages inside the panel are correlated, so counts of resolved intervals do not estimate a model-population prevalence. Intervals condition on the fitted cross-validation predictions, omit training, tuning and split variation, and are not adjusted across arms, depths, endpoints or seeds.

Batch-one requalification is structurally vacuous for every batch-one arm, here as in the admitted panel: at batch size one the repeat check's reference forward and the production forward are the same single-row computation, so an exactly-zero drift is true by construction and certifies nothing about numerical accuracy. Those arms are internally consistent, not numerically validated. ProGen3 is singleton-only — its expert mixture reduces over the whole flattened batch and it has no kernel above float16, with measured batch dependence of 0.02464 nats per scored token at batch size two and up to 0.07179 across sixteen equal-length rows — so no full-precision reference exists for either ProGen3 arm and both run at batch size one, as they did in the admitted panel.
