# Held groups, frozen cohort and pairwise sequence baseline for measured double-mutant stability

This document records the two blocking preparation steps for the D1 measured double-mutant nonadditivity experiment that `D1_INFORMATION_HIERARCHY_PROTOCOL.md` pre-registers: a validated family-grouping contract, and the pairwise sequence baseline Q together with its independent-site control. Both steps are label-independent and model-independent. Nothing here establishes a scientific result about any model, and no phenotype, DMS score or model output was read at any point in either step.

## The family-grouping contract

`WT_cluster` construction was not recovered from local provenance, so the source label cannot stand in for a grouping contract. The contract was therefore written and frozen before any group was computed: `data/pairwise_assets/grouping_contract_20260923.json`, sha256 `9e02b4b4fa5010901444480527a6d3b3499a9fdec128ce233eb0a393d841097b`, frozen 2026-09-24T01:05:11Z. The freeze covers the alignment algorithm, its substitution matrix and gap costs, the identity and coverage definitions, the edge rule, the Pfam threshold and the handling of missing Pfam labels.

The background universe is the 478 MegaScale backgrounds in the existing disjointness catalogue, 330 natural-labelled and 148 design-labelled. Every catalogue sequence was checked against the unique `mut_type == 'wt'` amino-acid sequence of the same `WT_name` in the pinned dataset2 parquet files, with zero mismatches, and every background carries exactly one `WT_cluster` value. One parquet background, `1A0N.pdb`, lies outside the catalogue and carries no wild-type row at all, so it cannot contribute a wild-type-centered cycle under any rule.

Groups are connected components over six declared merge sources: exact sequence identity, the frozen alignment edge, equal source `WT_cluster`, the documented parent relationship read from the name (so `1BK2.pdb_L5S` stays joined to `1BK2.pdb`), shared Pfam accession, and — for design-labelled backgrounds only — the catalogue topology or run-family label that covers shared ancestry across design rounds.

The alignment is exact Smith-Waterman with affine gaps over all 114,003 distinct pairs: BLOSUM62 half-bit scores, a gap of length k costing 11 + k, the optimum taken at the highest-scoring match cell with ties resolved in row-major order, percent identity over alignment columns including gap columns, and coverage measured as the aligned span over each sequence's own length. An edge requires at least 30.0% identity and at least 80.0% coverage of both sequences. The batched implementation that carries path statistics through the recurrence was checked against an independent per-pair traceback on 369 pairs — 250 drawn at a fixed seed plus every same-parent pair — with zero disagreements on score, alignment columns, identical pairs and both coverages. That check runs inside the group build and is fatal on any mismatch.

Pfam labels come from HMMER 3.4 `hmmscan` at the curated gathering thresholds against the pressed `Pfam-A.hmm`. 299 of 478 backgrounds carry at least one domain, across 122 distinct accessions; the remaining 179 carry none. A background with no label contributes no Pfam edge and is not merged with any other unlabelled background. That is an annotation gap on 32–74 residue domains, not evidence that those backgrounds are distinct families.

### What running the frozen contract revealed, and the one amendment

The frozen contract produced 94 groups: 93 natural-only and one 196-member component holding all 148 design-labelled backgrounds together with 48 natural-labelled ones. Thirty-six cross-label alignment edges had fused them, collapsing well-separated Pfam families — protein-G/B domains (PF01378), homeodomains (PF00046), ribosomal L9 (PF01281), PF11621, PF11623 and PF25787 — through the design set acting as a transitive hub, because de novo mini-proteins share generic secondary-structure topologies and chain to one another at this threshold.

That outcome contradicts the protocol's requirement that source-defined design backgrounds form a separate stratum rather than independent biological families. The contract's own `mixed` clause, which retained such a component as merged, was the defect. It was amended to restrict the alignment edge to within-stratum pairs and to report every cross-label edge instead of merging it: `data/pairwise_assets/grouping_contract_20260924.json`, sha256 `6a0a76a4ad1f083f7506f8fa29cb809b555fcf64ccbd722972de877cf26759c4`, 2026-09-24T01:23:52Z. The amendment is a function of sequence, name and annotation only. Both group counts are reported here, and the superseded run is retained under `logs/d1_pairwise_grouping_20260923/`.

The edge rule is far from chance-level in aggregate. A composition-preserving within-sequence shuffle over the same 114,003 pairs and the same rule accepted 21–34 edges per draw across five fixed-seed draws (natural–natural 2–5, design–design 12–26, cross-label 2–6), against 3,061 observed edges (1,835 natural–natural, 1,190 design–design, 36 cross-label). Cross-label edges are thus enriched roughly sixfold to eighteenfold over the shuffle null and are not simply noise; what they are not is evidence of a shared biological family between two backgrounds joined only through a chain of generic topologies.

Two of the ten flagged natural-labelled backgrounds are themselves designed mini-proteins that the four-character PDB-name screen mislabelled, which the protocol anticipated. `5UYO.pdb` aligns at 100.0% identity to a declared HEEH design and `5UP5.pdb` at up to 55.6% identity to EHEE and EEHEE designs; neither carries a Pfam label, and `5UP5.pdb` returned zero UniRef50 hits at an e-value of 1e-3, which is independent label-free corroboration. The amended contract keeps them in the natural stratum carrying that flag rather than silently reclassifying them. The other eight flagged backgrounds each carry a single edge at 30.0–38.5% identity to one design series and each carry a distinct, well-supported Pfam label.

### The readiness gate

The final group count after the union is **102**: 101 natural groups and 1 design group. Group sizes run from 1 to 148 members, with 69 singletons; the largest natural group holds 61 members and the single design group holds all 148 design-labelled backgrounds. Under the superseded contract the count was 94.

All 148 design-labelled backgrounds falling into one group is the honest consequence of the stratum rule plus within-design chaining. The design stratum therefore supports one background and cannot carry held-group inference on its own, which matches the label-instrument qualification's independent finding that four topology labels are not four independent design families.

These are conservative held-group controls, not proof of remote-family disjointness. Homology below 30% identity or below 80% mutual coverage is not detected, so two groups may still share a remote ancestor. Pfam annotation is incomplete on 179 of 478 backgrounds. `WT_cluster` provenance remains unrecovered, so that source contributes merges without certifying anything. The design stratum's ancestry is read from catalogue labels, not from a recovered design-pipeline record.

## The frozen cohort

The admission rule is the tightened one the label-instrument qualification admitted, not the protocol's provisional rule: a contributing row needs finite numeric `dG_ML` and a confidence width in [0, 0.5] kcal/mol on the combined estimate **and on each protease channel separately**, `deltaG_t_95CI` and `deltaG_c_95CI`. Each of the three widths is checked against its own upper-minus-lower bound before use, and a missing channel column is fatal. On the pinned files this accepts 575,331 of 776,298 rows. Cycles are built from exact four-state sequence matches with the median over accepted rows per `(WT_name, aa_seq)` state, every contributing row retained with its own interval, and no inverse-variance weighting anywhere.

The frozen cohort takes one background per final group by alphabetical `WT_name` order — label-independent — and up to 128 eligible double substitutions per background by the builder's stable-hash draw over the double-mutant sequence, which inspects no measurement magnitude. All four matching sequences are retained for every sampled cycle.

| Frozen workload | Value |
| --- | --- |
| Backgrounds, one per group | 64 |
| Natural groups covered, of 101 | 64 |
| Cycles | 8,192 |
| Distinct sequences after inference deduplication | 12,977 |
| Residues over those sequences | 779,664 |
| Sequence length range | 37–72 residues |
| Eligible cycles per chosen background | 129 to 3,799, median 566 |

The earlier feasibility draw of 17,822 sequences and 1,060,345 residues does not carry over: it used a different sampler, a looser admission rule and a provisional 86-source-cluster panel.

| Separation stratum | Cycles | Groups | Distinct site pairs |
| --- | --- | --- | --- |
| 1–2 | 819 | 24 | 25 |
| 3–9 | 2,341 | 36 | 61 |
| ≥10 | 5,032 | 53 | 131 |

The distant stratum covers 53 of the 64 cohort groups, so the protocol's requirement that any beyond-local increment appear on non-overlapping-window support is testable with the majority of the held groups represented.

Site-pair support is the binding power constraint. The cohort carries 217 distinct background/site-pair combinations, a median of 3 site pairs per background (range 1 to 14, with 17 backgrounds carrying a single site pair) and 1 to 128 cycles per site pair (median 27). Taking the site pair as the independent unit, the Kish effective count is far below the 8,192 cycle count under either weighting convention, and no interval on this cohort may be computed as though cycles were independent. That is a substantial improvement on the roughly 25 effective units the label-instrument qualification measured on the full unsampled panel, and it is a direct consequence of the 128-cycle cap and one-background-per-group rule equalising the weighting.

### Which effective site-pair count governs interval power

This is the authoritative statement for the pairwise cohort, and every other record points here rather than restating it. An effective count is meaningless without the weighting it was computed under, and this cohort admits two.

| Convention | 64-group support | 45-group Q-inclusive support |
| --- | ---: | ---: |
| **Estimator weights — governing.** Each group equal, each site pair equal inside a group, each cycle equal inside a site pair, which is what `readout_analysis.row_weights` realises and what every fit on this cohort applies | **129.2** | **96.1** |
| Cycle share. A site pair weighted by its share of its group's cycles | 121.6 | 88.4 |

**The estimator-weight pair governs**, because it is the Kish count of the weights the fits actually apply, and therefore the one that bounds their intervals. The cycle-share pair answers a different question — how unevenly the fitted rows spread over the dependence blocks — and is retained because it is informative about the cohort's balance and because this record quoted it first. Both are correct arithmetic under their own convention, both are computed and pinned by the tests, and they differ by about 6%: a reader who compares one of them against the other across two of our records would wrongly conclude that one is an error. The governing figures are the ones the [flagship fit](D1_PAIRWISE_EPISTASIS_RESULTS.md), the [global-context gate](D1_GATE_GLOBAL_CONTEXT.md), the [retrieval gate](D1_GATE_RETRIEVAL_MEMORIZATION.md) and audit register F20 lead with; the retrieval gate reports both side by side per stratum, which is how the two reconcile at band level.

Exclusions are fully accounted in the cohort file: 167 natural backgrounds fall below the 128-cycle cap, 43 carry no accepted wild-type row, and 56 lose to another background already representing their group. The largest excluded eligible count is 126 cycles, just under the cap, so the protocol's separately declared smaller-background sensitivity has real support to draw on. The 148 design-labelled backgrounds are not eligible for this natural primary panel and are counted as such.

## The pairwise sequence baseline

### Staging and asset verification

The frozen plmc binary, upstream commit `18c9e55e3bd2f14f4968be19a807b401996c929a`, sha256 `34043ab57ae859601d5a82dabe1ca70b79cb383ac9f7723bfb23e5a605e95772`, is compatible with the remote pods and has been staged there. The binary's highest symbol-version requirements are `GLIBC_2.34`, `GOMP_4.0` and `OMP_1.0`; the pod provides glibc 2.39 and a libgomp carrying GOMP up to 5.1.2, and every `ldd` entry resolves. The staged copy hashes correctly on GPFS; the transport does not preserve the executable bit, which had to be restored before the binary would run. Running the same synthetic three-site alignment with the same explicit unit weights on Compute and inside the pod produced a **byte-identical** parameter file, sha256 `c61529034cc34f6f5441194bbf50480b463b2911f00e52550b45e472100f72f2`. Nothing was installed in any pod.

The documented annotation assets were re-verified read-only and every one matches the Compute copy byte-for-byte: `diamond` 28,295,160 bytes sha256 `5d04a9ec…`, `hmmscan` 1,004,288 bytes sha256 `8b2950b3…`, `Pfam-A.hmm.h3i` 2,720,749 bytes sha256 `0a90fab0…`, `reference_corpus_metadata.json` 708 bytes sha256 `f9ccefae…`, `Pfam-A.hmm` 2,067,046,650 bytes sha256 `a78a42d6…`, and the UniRef50 DIAMOND database 25,345,024,406 bytes sha256 `88a6c44b…`. Because the assets are identical, the searches and annotations run on Compute are equivalent to reusing the staged remote copies, and all substantive work here stayed CPU-only and off the GPU lanes.

The pinned parameter reader's coupling axis orientation was verified against real plmc output rather than assumed. A two-site alignment with disjoint site alphabets and a one-way pairing gives J[A][D] = +4.87 and J[D][A] = 0.0000, so the first residue axis belongs to the lower-index position; the earlier synthetic check could not have detected a transpose, because the four-state contrast is invariant under one. The four-state contrast computed from sequence scores equals the contrast computed from the four coupling entries directly.

### Homolog support

The retained disjointness hit table is a retrieval certificate only: it caps at 100 hits per query and keeps no aligned strings. A fresh search with the same command builder, field list and parser retrieved the aligned rows: DIAMOND 2.1.24 `blastp`, `--very-sensitive`, `--masking 0`, e-value 1e-3, `--max-target-seqs 10000`, against the 60,315,044-cluster / 17,282,055,793-letter UniRef50 index, for all 478 backgrounds. It returned 618,403 hit lines in about eight minutes.

Retrieval is deep for some backgrounds and very shallow for many. Across the 330 natural backgrounds, 328 returned at least one hit, with a median of 334 hits and a 10th percentile of 8; 13 reached the 10,000-hit ceiling. On the 64 frozen cohort backgrounds specifically the median is 98 hits, the 25th percentile is 22, twelve backgrounds returned fewer than 20 hits, one (`5UP5.pdb`) returned none, and three reached the ceiling. Alignment rows were reconstructed with the existing coordinate reconstruction, validated against the existing profile estimator's own column frequencies, and reweighted at the existing 80% identity threshold.

### Fitting and regularization selection

Each background gets one plmc pseudolikelihood fit and one independently fitted independent-site model on identical homolog support. Homolog rows split 80/20 by a stable hash of each aligned row's own residues, so the split is independent of hit ordering, puts identical rows on the same side, and cannot be influenced by any measurement. The coupling L2 penalty is selected from {0.16, 0.8, 4.0, 16.0, 80.0} by held-out homolog sequence prediction alone, with the field penalty held at the upstream example's 0.01 and a 200-iteration cap; the selected value is then refit on the full support.

The selection criterion is the mean gap-ignoring per-site conditional log likelihood of held-out rows — exactly the functional plmc optimises under `-g`, evaluated out of sample. That identity was verified numerically rather than asserted: reconstructing plmc's own weighted objective from the serialized parameters on a training alignment reproduced plmc's reported `-loglk` of 2881.8 to within 0.012, a relative agreement of 4e-6 consistent with float32 parameter serialization. The independent-site control is scored on the same positions with the same convention, so the two models are compared on identical support.

Forty-five of the 64 cohort backgrounds carry a fitted pairwise model. Nineteen do not: eighteen fall below the declared support floor of 20 training rows or 5 held-out rows once the 80% coverage filter is applied, and `5UP5.pdb` returns no UniRef50 hit at all — consistent with its being one of the two mislabelled designed mini-proteins the grouping step identified. The support floors are declared, not tuned: a background below them is reported as unsupported rather than fitted with a number that cannot be defended.

| Fitted-background support (45 backgrounds) | min | Q1 | median | Q3 | max |
| --- | --- | --- | --- | --- | --- |
| Reconstructed alignment rows | 29 | 72 | 357 | 829 | 9,931 |
| Training rows | 24 | 55 | 283 | 667 | 7,909 |
| Held-out rows | 5 | 15 | 69 | 157 | 2,219 |
| Reweighted depth at 30% identity (Neff) | 5.9 | 46.0 | 152.9 | 395.5 | 5,985.6 |
| Effective depth per site (Neff / length) | 0.107 | 0.845 | 2.91 | 6.95 | 92.1 |

The selected coupling penalty is 0.16 on 4 backgrounds, 0.8 on 16, 4.0 on 17, 16.0 on 8 and 80.0 on none. Four selections sit at the grid's weakest-regularization edge and none at the strongest, so the grid brackets the held-out optimum for 41 of 45 backgrounds and is open only on the weak side for the other four.

The pairwise model predicts held-out homolog sequences better than its own independent-site control on **all 45 fitted backgrounds**, by +0.0389 to +1.0106 nats per scored site with a median of +0.3879. That is the label-independent check the protocol requires, and it is the only sense in which this baseline is established: passing it makes Q a usable sequence comparator, not a biologically correct model of these families.

Convergence is reported honestly and then bounded by a sensitivity run. Under the 200-iteration budget, L-BFGS reported minimization success on 17 of the 45 final refits and on 86 of the 225 sweep fits; the rest stopped at the cap. Because every regularization value is fitted under the same budget, the selection itself is a matched comparison, but a truncated optimizer is a weaker claim to a regularized maximum-a-posteriori estimate, so the budget was tested directly. A nine-background subset spanning the whole depth range, from 5.9 to 5,985.6 effective sequences, was refitted at a tenfold larger cap of 2,000 iterations. The selected coupling penalty is **identical on all nine**, and the held-out gain moves by at most 0.00122 nats per site, while the larger budget does reach the optimizer's own criterion on 8 of 9 final refits against 4 of 9 at the smaller cap, and on 40 of 45 sweep fits.

| Background | Neff | Selected penalty, 200 / 2,000 | Held-out gain, 200 / 2,000 | Converged, 200 / 2,000 |
| --- | --- | --- | --- | --- |
| 1LP1.pdb | 5.9 | 0.16 / 0.16 | +1.0106 / +1.0106 | yes / yes |
| 2MXD.pdb | 28.5 | 0.8 / 0.8 | +0.5650 / +0.5650 | yes / yes |
| 2K5P.pdb | 46.0 | 4.0 / 4.0 | +0.1559 / +0.1559 | yes / yes |
| 6ACV.pdb | 95.3 | 0.8 / 0.8 | +0.2991 / +0.2990 | no / yes |
| 1UZC.pdb | 152.9 | 0.8 / 0.8 | +0.5318 / +0.5318 | no / yes |
| 1I2T.pdb | 282.1 | 0.8 / 0.8 | +0.3906 / +0.3905 | no / yes |
| 3L1X.pdb | 479.8 | 4.0 / 4.0 | +0.3155 / +0.3143 | no / yes |
| 2K1B.pdb | 1,233.2 | 4.0 / 4.0 | +0.3929 / +0.3922 | no / yes |
| 2LGW.pdb | 5,985.6 | 16.0 / 16.0 | +0.2573 / +0.2573 | no / no |

The unreported convergence flag at 200 iterations therefore reflects a strict optimizer criterion on a fit whose objective and held-out prediction have effectively plateaued, not a materially unfinished estimate. Nothing in the selection or the admissibility verdict depends on the iteration cap; the delivered parameters are the 200-iteration ones, and this sensitivity is the ground for keeping them.

Depth remains the binding weakness. A pairwise model on a 37–72 residue domain carries between roughly 260,000 and 1,020,000 coupling parameters, against a median reweighted depth of 152.9 effective sequences and a median of 2.91 effective sequences per site. The couplings are therefore prior-dominated for most backgrounds, which is exactly why the regularization is selected out of sample rather than fixed. The support cannot be deepened locally: the UniRef50 source FASTA is not retained on disk — only its DIAMOND index — so no HMM-based iterative search can be run against that corpus offline, and the one protein FASTA that is available, Swiss-Prot, is about two orders of magnitude smaller than UniRef50 and would give shallower alignments rather than deeper ones. Three backgrounds reached the 10,000-hit retrieval ceiling and are right-censored in depth.

### The four-state energy contrast interface

An independent-site log profile has a zero double contrast by construction for aligned substitutions, and the implementation measures that zero rather than replacing it. Over every cycle of every fitted background the maximum absolute independent-site double contrast is 2.8e-14, which is floating-point cancellation and not a signal. The fitted pairwise contrast on the same cycles is genuinely nonzero, at 2.3e-5 to 5.27 in absolute unnormalized log-weight units with a median of 0.187 and a 90th percentile of 0.771. No Spearman coefficient and no pairwise comparator is manufactured for the independent-site contrast; its constituent-site features and a nonlinear calibration are the relevant stronger nuisance controls, and the label-instrument qualification has already made that nonlinear-additive control mandatory rather than optional.

## Verdict and limitations

**The family grouping is ready.** The readiness gate reads 102 final groups, 101 of them natural, built from a contract frozen before computation with an exactly verified aligner, one declared and dated amendment, and a shuffle null showing the edge rule is far from chance-level. They are conservative held-group controls, not certified remote-family holdouts.

**The frozen cohort is ready, with the site pair as its independent unit.** 64 backgrounds cover 64 of the 101 natural groups with 8,192 cycles over 12,977 deduplicated sequences and 779,664 residues, on the tightened per-channel admission rule. Power is set by 217 site pairs and, under the governing estimator weighting, 129.2 Kish effective site pairs, not by the cycle count.

**Baseline Q is admissible on 45 of the 64 groups and not on the other 19.** On the 45 it passes its declared label-independent check on every background. The 19 without it are not a fixable gap with the corpora on hand.

Because the protocol requires identical support for every incremental comparison, the consequence is a split evaluation that must be fixed before any model is scored:

| Comparison | Support |
| --- | --- |
| Additive null, sequence/profile baseline C, nonlinear-additive nuisance G, model likelihood and representation increments | all 64 groups, 8,192 cycles, 217 site pairs, 129.2 Kish effective site pairs under the governing convention and 121.6 under cycle share |
| The same set with pairwise baseline Q added | 45 groups, 5,760 cycles, 9,143 deduplicated sequences, 550,964 residues, 163 site pairs, 96.1 Kish effective site pairs under the governing convention and 88.4 under cycle share |

Per-stratum group coverage on the Q-inclusive support is 19 groups and 20 site pairs at separation 1–2, 26 groups and 49 site pairs at 3–9, and 37 groups and 94 site pairs at ≥10. The distant stratum, where the protocol requires any beyond-local increment to appear, therefore rests on 37 groups and 94 site pairs once Q is in the comparison.

The protocol's fallback applies to the 19 unsupported groups rather than to the experiment as a whole: on those groups the claim narrows to the additive and nonlinear-additive controls and the pairwise boundary stays unresolved. It would be wrong to read a model increment over Q on the 45-group subset as a boundary claim about the other 19, and equally wrong to substitute a weaker estimator there. The retired bias-corrected mutual-information/APC channel, which failed its DMS diagnostic on 0 of 22 eligible assays, is not used, renamed or resurrected anywhere in this work.

Limitations beyond those already stated in place:

- Homology below 30% identity or 80% mutual coverage is undetected, so two held groups may still share a remote ancestor; 179 of 478 backgrounds carry no Pfam label, and `WT_cluster` provenance was never recovered.
- All 148 design-labelled backgrounds form a single group. The design stratum supports one background and no held-group inference.
- Two backgrounds the catalogue labels natural are designed mini-proteins. They stay in the natural stratum carrying that flag; any analysis that treats the natural stratum as biological must exclude or re-stratify them explicitly.
- Coupling depth is thin: a median of 2.91 effective sequences per site against hundreds of thousands of coupling parameters, with the couplings consequently prior-dominated and the support not deepenable from the corpora retained on this host.
- The cohort is conditioned on resolvability in all four states, so cycles whose double mutant falls below the assay's stability floor are absent, and 99.98% of admitted cycles sit on site pairs the source study itself selected as suspected interactions. Sequence separation on this panel is not a contact contrast.
- Passing the held-out-sequence check establishes a usable sequence comparator only. It says nothing about whether the fitted couplings correspond to physical interactions, and a model increment over Q remains bounded by the declared controls, the label budget and the readout class.
- The frozen cohort and the fitted parameters are retained under ignored `logs/`; they are inputs to the pending experiment, not results.
