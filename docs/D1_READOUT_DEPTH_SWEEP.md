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

## What the controls measured, and the two repairs they forced

Extraction agreement is exact, on the whole panel and not only on a sample. Over all 54 cells and all 132 (arm, panel, seed) rows the re-extracted pooled summaries at the two admitted depths reproduce the admitted retained arrays at maximum relative L2 **0.0**, maximum absolute difference **0.0**, exactly equal on every variant, with maximum absolute native-likelihood difference **0.0 nats**, against gates of 0.001 relative L2 and 0.001 nats. The re-extraction is therefore a strict superset of the admitted extraction rather than an approximation of it, and the causal prefix difference is 0.0 at every block of every position-resolved arm.

Fit-level reproduction passes at the tolerance the admitted admission receipt used. The worst held-out prediction deviation over the 132 rows is **5.123e-13** against 1e-8, and the pipeline-identity control, which refits the admitted class from the admitted retained states at the primary seed, records a worst deviation of **4.539e-13** over its 54 rows. It got there the hard way: two cells raised it first, and the repair below is why the rest pass.

The same binding shows at the endpoint rather than only at the predictions, which is what a reader can check against the [readout panel](D1_READOUT_RESULTS.md) directly. At the primary split seed the refitted admitted class returns +0.019700 [+0.009420, +0.029708] for `proteinglm-7b-clm` on the anchor panel against the panel's reported +0.01970 [+0.00942, +0.02971], and +0.013413 for `progen2-base` against its reported +0.01341; the fixed label permutation returns −0.024801 and −0.019776 for `gpt2-xl` against the panel's reported −0.02480 and −0.01978. Every digit the panel prints is recovered.

On `proteinglm-7b-clm` / `EC_conditioned40` the control recovered the admitted predictions to 1.268e-14 on the baseline, 7.550e-15 on the representation and 1.865e-14 on baseline-plus-representation; on `qwen2.5-32b` / `native_qwen2.5-32b` it raised at 3.483e-08 after 61 minutes of fitting, as it is required to. The cause is that a float32 matrix product is not invariant to how its rows are blocked, and this experiment formed the admitted design over a whole panel at once while the admitted analysis forms it one assay at a time. Measured on that panel's admitted arrays — 212 assays, 27,071 variants, hidden width 5,120 — the whole-panel product departs from the per-assay product by up to 7.629e-06 in absolute value against a largest absolute entry of 723.245, at every BLAS thread count of 1, 4 and 16, while the per-assay product is identical at all three. The repair is to block the rows by assay as the admitted analysis blocks them and to pin the BLAS thread count, so the reduction order is the admitted one rather than the host's.

The exposure follows the panel, not the arm: `qwen2.5-32b` passed on the 201-assay anchor panel at 3.249e-13 and raised on its own 212-assay native panel, and `gpt2-xl` did the same, passing the anchor panel at 3.526e-13 and raising its native panel at 5.368e-08. Because the 15 native panels are the largest supports in the roster, the remaining fitting was re-dispatched under the repaired construction rather than left to raise cell by cell: the 13 cells already complete are kept, since each is bound to its admitted report inside the declared 1e-8, and the 39 cells not yet complete — 10 then running, 27 pending and the two that raised — were stopped and refitted. The two constructions therefore appear in one panel; both are certified against the admitted report at the declared tolerance, and the measurement on the class sweep shows that a perturbation three orders of magnitude larger moves no reported correlation.

The second repair is to the extraction receipt. Two of 6,834 written archives, one assay each on `instructprotein` and `galactica-1.3b` at about 101 MB, carry a recorded SHA-256 that does not bind the bytes now on disk, and the fit stage refused both panels rather than reading them. The archives themselves are intact: the recorded and on-disk lengths are equal to the byte, all 84 ZIP members pass their stored CRC32 checks, the stored identity, assay name and mutation digest bind the manifest, all 128 variants are present and every array is finite, the digest is stable across repeated reads, and each assay was extracted exactly once. Because the member CRC32 values are written from the extractor's own in-memory arrays, those checks passing establishes that the array bytes on disk are the ones the extractor produced; what drifted is the container digest, which was read back from the shared filesystem immediately after the rename without the write having settled. The writer now flushes and fsyncs before the rename. The recorded digest is not rewritten to match the file, because a receipt that is adjusted to its subject certifies nothing; the two arms were re-extracted instead, on the first cards to fall idle, and their republished manifests now bind every archive — 0 of 201 mismatched on each. Both panels then fitted normally.

## Results

The panel artifact is `results/external_baseline/d1_readout_depth_panel_20260925/readout_depth_panel.json` on the shared filesystem, SHA256 `076b93525abbb126e6f5a828ad758c0b369eea198dda45f0af57fa916d197939`, built from the 54 cell reports by `scripts/transfer/summarise_depth_panel.py`. It carries the per-block fitted estimates with their intervals, resampling unit and draw count, the falsification tally, the seed-consistent blocks, the per-arm breadth reading, the ceilings and each cell's own digest, so every figure below can be checked without re-fitting. The per-cell reports hold the fits themselves: one cell's primary seed carries 707 named summaries and its three seeds 1,885, each with point, interval, unit, unit count, draw count and alpha, under `seeds` → `summaries`. The cross-cell aggregates exist only in the panel artifact, because no per-cell file can hold a count over cells.

All 54 cells completed and every one passes both reproduction controls. The panel covers the 34 admitted arms on the 17 admitted panels at every declared seed, 132 (arm, panel, seed) fit rows, and 1,004 transformer blocks. Every quantity below is a cluster-equal mean within-assay Spearman difference against the matched admitted baseline, on the unit *wild-type family at 50% identity* — 163 families on the anchor panel and 30 on the EC-conditioned panel — with 2,000 paired bootstrap draws at seed 20260923 and alpha 0.05. Strata are never pooled, and no interval is adjusted across arms, depths, axes or seeds.

### The falsification control holds at every depth

No literal-text arm places resolved additional information in any single block, on any axis. Aggregated over every panel and every axis the stratum carries **1,638 depth cells on 14 arms and 12 panels, of which 0 resolve above zero, 1,573 resolve below and 65 are unresolved**, and its largest point estimate anywhere is +0.000771911134033733, at block 0 of ByGPT5-medium-en, itself unresolved. On the anchor panel alone its 13 arms give 1,098 depth cells with 0 above zero and 1,055 below. The `wide`, `pos` and `full` axes add 81, 66 and 15 further cells with 0 resolved above zero. Each of the ten native literal-text panels repeats it on its own support: GPT-2 0 of 12, GPT-2-medium 0 of 24, GPT-2-large 0 of 36, GPT-2-XL 0 of 48, Qwen2.5-0.5B 0 of 24, Qwen2.5-0.5B-Instruct 0 of 24, Qwen2.5-7B 0 of 28, Qwen2.5-32B 0 of 64, Qwen3-8B-base 0 of 36, Llama-3.2-3B 0 of 28, DialoGPT-small 0 of 12. Across roughly 1,600 literal-text depth cells the count of resolved positive increments is zero. The gains reported below are therefore protein-specific: a different depth does not recover them on literal text, so they cannot be an artifact of the readout reaching more of a shared string-level signal.

The same asymmetry appears in the `union` axis, which supplies every depth at once. At the primary seed it resolves **above** zero for five cells, all protein interfaces — ProGen2-base +0.015760, ProGen2-small +0.014159, ProGen3-3B +0.015882 and ProteinGLM-7B-CLM +0.025387 on the anchor panel, and ProteinGLM-7B-CLM +0.034350 on the EC panel — and **below** zero for every literal-text arm on both its supports, from −0.010688 to −0.041688, for the Llama-2-7B parent on all three of its panels (−0.023933, −0.028417, −0.067585), for both ProLLaMA stages on the EC panel (−0.080010, −0.079516) and for ZymCTRL (−0.032995). Supplying more depth to an arm that carries nothing makes it worse, which is what adding uninformative columns to a well-supervised baseline does.

### Where the information sits, when it is there

Counting resolved intervals over 1,004 blocks without adjustment would invite a tail artifact, so the tables report only blocks resolved above zero at **every** declared split seed of their panel. Ranges are the minimum-to-maximum point estimate over those seeds, not uncertainty intervals. The two informative axes are reported separately, because they are different designs: `depth` is the two pooled rules at one block, and `pos` is the mutation-localized and suffix summaries at one block, defined only for the 19 arms whose rendering assigns one residue-bearing token per residue.

The pooled `depth` axis gives 23 such blocks, on 8 arms:

| Arm | Panel | Block | Relative depth | Increment across seeds |
| --- | --- | ---: | ---: | --- |
| ProteinGLM-7B-CLM | EC_conditioned40 | 34 of 36 | 0.97 | +0.042542 to +0.053010 |
| ProteinGLM-7B-CLM | EC_conditioned40 | 33 of 36 | 0.94 | +0.037016 to +0.043653 |
| ProteinGLM-7B-CLM | EC_conditioned40 | 35 of 36 | 1.00 | +0.030321 to +0.035961 |
| ProGen3-3B | EC_conditioned40 | 23 of 24 | 1.00 | +0.016251 to +0.018326 |
| ProteinGLM-7B-CLM | anchor201 | 30 of 36 | 0.86 | +0.013499 to +0.016616 |
| ProteinGLM-7B-CLM | anchor201 | 33 of 36 | 0.94 | +0.012705 to +0.018599 |
| ProteinGLM-7B-CLM | anchor201 | 11 of 36 | 0.31 | +0.012088 to +0.016246 |
| ProteinGLM-7B-CLM | anchor201 | 12 of 36 | 0.34 | +0.011899 to +0.019062 |
| ProteinGLM-7B-CLM | anchor201 | 32 of 36 | 0.91 | +0.011563 to +0.015172 |
| ProGen3-112M | anchor201 | 9 of 10 | 1.00 | +0.011416 to +0.013702 |
| ProteinGLM-7B-CLM | anchor201 | 35 of 36 | 1.00 | +0.011073 to +0.016260 |
| ProtGPT2 | anchor201 | 21 of 36 | 0.60 | +0.010775 to +0.016836 |
| ProGen2-large | anchor201 | 25 of 32 | 0.81 | +0.010746 to +0.012079 |
| ProGen2-medium | anchor201 | 23 of 27 | 0.88 | +0.010690 to +0.017632 |
| ProGen3-3B | anchor201 | 11 of 24 | 0.48 | +0.010343 to +0.016814 |
| ProGen2-xlarge | anchor201 | 28 of 32 | 0.90 | +0.009763 to +0.012978 |
| ProGen2-medium | anchor201 | 24 of 27 | 0.92 | +0.009331 to +0.014571 |
| ProtGPT2 | native_protgpt2 | 6 blocks, 18 to 25 | 0.51 to 0.71 | +0.009067 to +0.012232, one seed only |

The position-resolved `pos` axis gives 16, on 6 arms, and reaches higher on the anchor panel than the pooled axis does:

| Arm | Panel | Block | Relative depth | Increment across seeds |
| --- | --- | ---: | ---: | --- |
| ProteinGLM-7B-CLM | EC_conditioned40 | 34 of 36 | 0.97 | +0.034720 to +0.040477 |
| ProteinGLM-7B-CLM | EC_conditioned40 | 35 of 36 | 1.00 | +0.031473 to +0.037639 |
| ProGen3-3B | EC_conditioned40 | 22 of 24 | 0.96 | +0.028442 to +0.032694 |
| ProteinGLM-7B-CLM | anchor201 | 33 of 36 | 0.94 | +0.019177 to +0.022215 |
| ProGen3-3B | EC_conditioned40 | 23 of 24 | 1.00 | +0.015988 to +0.017836 |
| ProteinGLM-7B-CLM | anchor201 | 34 of 36 | 0.97 | +0.015951 to +0.019604 |
| RITA-XL | anchor201 | 23 of 24 | 1.00 | +0.015119 to +0.019877 |
| ProGen3-3B | anchor201 | 22 of 24 | 0.96 | +0.014607 to +0.024514 |
| ProteinGLM-7B-CLM | anchor201 | 35 of 36 | 1.00 | +0.013130 to +0.018203 |
| ProGen3-3B | anchor201 | 23 of 24 | 1.00 | +0.012918 to +0.017575 |
| ProtGPT3-1.3B | anchor201 | 0 of 17 | 0.00 | +0.011911 to +0.014963 |
| ProGen2-large | anchor201 | 28 of 32 | 0.90 | +0.010852 to +0.016373 |
| ProGen2-xlarge | anchor201 | 28 of 32 | 0.90 | +0.010782 to +0.018143 |
| ProteinGLM-7B-CLM | anchor201 | 30 of 36 | 0.86 | +0.010107 to +0.011859 |
| ProGen2-xlarge | anchor201 | 25 of 32 | 0.81 | +0.009961 to +0.014246 |
| ProGen2-medium | anchor201 | 0 of 27 | 0.00 | +0.006983 to +0.010573 |

Nine arms appear across the two tables — ProGen2-large, ProGen2-medium, ProGen2-xlarge, ProGen3-112M, ProGen3-3B, ProteinGLM-7B-CLM, ProtGPT2, ProtGPT3-1.3B and RITA-XL — and every one is a protein interface. No literal-text arm appears in either table, and neither does ZymCTRL, the Llama-2-7B parent or either ProLLaMA stage. The blocks cluster in the upper stack: 26 of the 39 entries sit at relative depth 0.81 or deeper, with three low-depth exceptions, ProteinGLM's blocks 11 and 12 of 36, ProtGPT3-1.3B's block 0 of 17 and ProGen2-medium's block 0 of 27.

The two widened designs add nothing. The `wide` axis, which repeats the pooled design at the admitted total width of 1,024 coordinates instead of 512, and the `full` axis, which supplies all four summaries at one block, each yield **zero** blocks resolved above zero at every declared seed. Doubling the per-block coordinate budget does not convert an unresolved block into a resolved one.

### Per-arm depth selection

This is the table downstream work needs. For each arm and panel where a block resolves above zero at every seed on the pooled axis, it gives the admitted two-depth increment for comparison and states whether the admitted extraction had already resolved that cell.

| Arm | Panel | Blocks (admitted pair) | Admitted two-depth increment | Selected block | Increment there | Reading |
| --- | --- | --- | --- | ---: | --- | --- |
| ProGen2-large | anchor201 | 32 (15, 31) | +0.002565 to +0.009063, 0 of 3 resolved | 25 | +0.010746 to +0.012079 | boundary lifted |
| ProGen2-medium | anchor201 | 27 (13, 26) | +0.002743 to +0.009809, 0 of 3 | 23 | +0.010690 to +0.017632 | boundary lifted |
| ProGen2-xlarge | anchor201 | 32 (15, 31) | +0.000746 to +0.006475, 0 of 3 | 28 | +0.009763 to +0.012978 | boundary lifted |
| ProGen3-112M | anchor201 | 10 (4, 9) | +0.002612 to +0.006659, 0 of 3 | 9 | +0.011416 to +0.013702 | boundary lifted |
| ProGen3-3B | EC_conditioned40 | 24 (11, 23) | +0.003071 to +0.017990, 0 of 3 | 23 | +0.016251 to +0.018326 | boundary lifted |
| ProtGPT2 | anchor201 | 36 (17, 35) | +0.001548 to +0.009622, 0 of 3 | 21 | +0.010775 to +0.016836 | boundary lifted |
| ProtGPT2 | native_protgpt2 | 36 (17, 35) | +0.003693, 0 of 1 | 21 | +0.012232 | boundary lifted |
| ProGen3-3B | anchor201 | 24 (11, 23) | +0.007388 to +0.017362, 2 of 3 | 11 | +0.010343 to +0.016814 | already resolved |
| ProteinGLM-7B-CLM | anchor201 | 36 (17, 35) | +0.019700 to +0.022549, 3 of 3 | 30 | +0.013499 to +0.016616 | already resolved, and lower |
| ProteinGLM-7B-CLM | EC_conditioned40 | 36 (17, 35) | +0.045421 to +0.051388, 3 of 3 | 34 | +0.042542 to +0.053010 | already resolved, no gain |

Seven of the ten entries lift a boundary: the arm resolved nothing at the admitted pair at any seed and resolves at a single block at every seed. **Five** of those seven selected blocks are not members of the admitted pair. The other two are, and they say something different: ProGen3-112M's block 9 of 10 and ProGen3-3B's block 23 of 24 on the EC panel are both the final block, which the admitted extraction did hook. For those two arms what bounded the result was not the depth but the pooling across depths — the admitted design dilutes 1,024 coordinates over two depths and two rules, while the depth axis spends 512 on one block, and the narrower, better-targeted design resolves where the diluted one does not. The three remaining entries were already resolved by the admitted extraction, and for those the selection does **not** improve on it — for ProteinGLM on the anchor panel the best single block is *lower* than the admitted pooled pair, +0.013499 to +0.016616 against +0.019700 to +0.022549, and on the EC panel block 34's +0.042542 to +0.053010 straddles the admitted +0.045421 to +0.051388, exceeding it at one seed and falling below it at the other two.
### What this changes, and what it does not

The admitted two-depth extraction was a partial measurement boundary, and the part it bounded was **breadth, not magnitude**. Breadth: the admitted panel resolves a positive increment for 3 arms of 34; depth resolution finds seed-consistent resolved blocks for 9, and for 7 arm-and-panel cells it converts a cell that resolved nothing at the admitted pair at any seed into one that resolves at a single other block at every seed. Every one of those arms is a protein interface and six of the seven selected blocks are not members of the admitted pair. Magnitude: the ceiling does not move. The largest seed-consistent increment anywhere on the 163-family anchor panel is +0.024514, on ProGen3-3B's position-resolved block 22 of 24, against the admitted panel maximum of +0.02255; the largest on the pooled axis is +0.019062, below that maximum; and on the 30-family EC panel the best block straddles the admitted pair rather than beating it, +0.042542 to +0.053010 against +0.045421 to +0.051388, higher at one seed and lower at the other two. For the two cells the admitted extraction had already resolved, the best single block is no better than the admitted pooled pair and on the anchor panel it is worse.

So the panel-wide near-null survives as a statement about how much information the frozen states place beyond profile, likelihood and sequence descriptors, and it does not survive as a statement about how few arms place any. A representation-ceiling reading is strengthened on magnitude: 1,004 blocks, four summaries, two pooling rules and a design that supplies every depth at once do not lift the anchor-panel increment beyond the +0.02 to +0.025 band the admitted two depths already reached. The competing reading, that the admitted extraction hooked the wrong blocks, is correct for which arms resolve and wrong for how much they resolve. Anyone reading the EC panel's +0.053010 as a fivefold gain over the anchor-panel resolutions should note that it rests on 30 wild-type families rather than 163, that its interval is correspondingly about twice as wide, and that the admitted two-depth fit on that same panel already reached +0.045421 to +0.051388 — so it is the same quantity measured on a smaller support, not new information found at a new depth.

None of this licenses a claim that the protein arms' resolved increments are large enough to matter biologically. On the anchor panel they are +0.009 to +0.020 Spearman against a matched baseline that already reaches +0.436 to +0.489 there, so the representation adds on the order of two to four percent of the baseline's own held-cluster correlation.

## Limitations carried and added

The designs along the `depth`, `wide`, `union`, `pos` and `full` axes are formed over a whole panel at once, which is the right construction for them because they have no admitted counterpart to reproduce. Their float32 products therefore depend on the host's BLAS blocking in the same way the admitted design did, at a relative scale of about 1e-08 on the design coordinates; the campaign fixes the thread count per cell, so a cell is reproducible on its own terms, and the measurement above shows that a perturbation three orders of magnitude larger moves no reported correlation. A refit of one of these axes on a different host or thread budget is nonetheless not guaranteed bit-identical.

A null at every depth remains bounded by the readout class, the pooling and position rules, the label budget and the cohort, and does not establish that biological information is absent. The class here is one fixed compressed linear family; the class sweep varies that axis separately, and the two axes compose but are not fitted jointly. The position-resolved summaries exist for 19 arms and cannot be defined for the other 15, so a comparison across that boundary confounds interface with position resolution. The label budget is 163 wild-type clusters on the anchor panel and 30 on the EC-conditioned panel, whose intervals are correspondingly about twice as wide. Wild-type clustering at 50% identity does not guarantee remote-superfamily disjointness or exclude pretraining overlap. The three split seeds are robustness checks on one biological cohort, not independent replications, and the minimum-to-maximum point ranges reported with them are not uncertainty intervals. Checkpoint lineages inside the panel are correlated, so counts of resolved intervals do not estimate a model-population prevalence. Intervals condition on the fitted cross-validation predictions, omit training, tuning and split variation, and are not adjusted across arms, depths, endpoints or seeds.

Batch-one requalification is structurally vacuous for every batch-one arm, here as in the admitted panel: at batch size one the repeat check's reference forward and the production forward are the same single-row computation, so an exactly-zero drift is true by construction and certifies nothing about numerical accuracy. Those arms are internally consistent, not numerically validated. ProGen3 is singleton-only — its expert mixture reduces over the whole flattened batch and it has no kernel above float16, with measured batch dependence of 0.02464 nats per scored token at batch size two and up to 0.07179 across sixteen equal-length rows — so no full-precision reference exists for either ProGen3 arm and both run at batch size one, as they did in the admitted panel.
