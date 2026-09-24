# Recomputing the representation cells when the readout reassessment lands

Every representation-level verdict in the [capability map](D1_CAPABILITY_MAP_PLAN.md) is provisional until the [readout-class sweep](D1_READOUT_CLASS_SWEEP.md) and the [depth sweep](D1_READOUT_DEPTH_SWEEP.md) complete, after which the representation analyses at every gate are revisited and, where necessary, recomputed. This document owns that recomputation: the driver that refits a published cell at a selected readout class and depth, the inventory of what each gate retained for it, and the contract each gate's report is checked against. It owns no measurement. Every quantity, interval and verdict belongs to the gate record it is read from, and each gate record stays authoritative for its own numbers.

**Nothing has been recomputed.** The reassessment's results do not exist yet, so no selection exists to recompute at, and the driver refuses to run without one. What is built and verified is the machinery, the refusals and the retention accounting, so that the recomputation is a dispatch rather than a re-derivation when the selection arrives.

## The two selection kinds, and why they decide everything

A reassessment can return two different kinds of answer, and they have different retention requirements. Conflating them is the mistake this pipeline is built to prevent.

A **readout-class** selection is a different function of the four pooled summaries at the two blocks the admitted extraction hooked — a wider projection, the uncompressed features, a nonlinear head, or one depth-and-pooling block alone. Every gate that retained its full-width block outputs can serve it, with no forward pass anywhere.

An **extraction-depth** selection names a block other than those two, or a position-resolved summary. No retained array holds one. A gate can serve it only if a depth-resolved extraction exists **on that gate's own cohort**, and the depth campaign re-extracts the Readout cohort's 17 panels rather than any other cohort. For a gate fitted elsewhere, a depth selection is a fresh measurement campaign and not a refit, whatever that gate retained.

## What each gate retained, and what it can serve

The declaration is frozen in `src/transfer/gate_retention.py` at content digest `932f07fbcaa28d817a0c1257e0a891273dc3609b5a4b92b54dbbcad14899d9be`, and `scripts/transfer/inventory_gate_retention.py` probes it against a given repository root and, in a pod, against the allocation's shared filesystem. A location in a store the invocation was given no root for is reported unreachable from that host rather than absent, because the substantive arrays live on the allocation and calling them missing from the workstation would bury the ones that really are. All 30 repository-store locations the declaration names are present on Compute; the cluster-store locations are unreachable from it by construction.

Seventeen entries cover the twelve mechanism gates, the two activities that are not gates, and the three upstream studies whose retained arrays two of the gates read instead of their own extraction. Eight entries carry representation cells, 825 in total.

| Entry | Representation cells | Resampling unit | A different class | A different depth |
| --- | ---: | --- | --- | --- |
| Local chemistry and local context | 99 | wild-type family at 50% identity | from the admitted Readout extractions' full-width states | from the depth campaign, same cohort and arms |
| Folding and stability | 99 | MegaScale natural family group | from its own full-width retention pass, verified complete | **requires a fresh extraction** |
| Residue interactions | 99 | held family group | from its own full-width per-state block outputs | **requires a fresh extraction** |
| Remote homology | 99 | family group | from its own full-width states, 33 of 33 cells complete | **requires a fresh extraction** |
| External confirmation | 99 | family group | from its own full-width states, once its extraction completes | **requires a fresh extraction** |
| Evolution and adaptation | 99 | wild-type family at 50% identity | by re-aggregation, after the crossed-control cells | as for a class |
| Readout panel, upstream | 132 | wild-type family at 50% identity | yes | yes |
| Crossed controls, upstream | 99 | wild-type family at 50% identity | yes | yes |
| Higher-order dependence, both records | 0 | — | no cell exists | no cell exists |
| 3D structural constraints | 0 | — | no cell exists | no cell exists |
| Generation and control | 0 | — | no cell exists | no cell exists |
| Global-but-additive context | 0 | — | no cell exists | no cell exists |
| D2 ProLLaMA transplant | 0 published | — | archives retained, no cell declared | no cell exists |
| Conformational dynamics, molecular recognition, cellular regulation | 0 | — | not launched | not launched |

*No cell exists* is not a blocked recomputation and is never reported as one. The two higher-order records stopped at endpoint qualification and read no model quantity; the structure gate's part 2 was not run because part 1 is not passed; the generation gate's endpoints are properties of emitted sequences; the global-context gate is declared over model likelihood only, which is why its contrasts are final rather than provisional. Each carries its own reason in the declaration.

### Two findings the inventory produced

**Four gates cannot serve a depth selection from anything they retained.** Each hooked the block after half the stack and the last block on its own cohort and wrote no other block's state: folding and stability on 101 MegaScale family groups, residue interactions on 64 held groups of double-mutant cycles, remote homology on 179 MGnify-derived family groups, and external confirmation on 96 family groups over 428 Domainome domains. The last two were added after this inventory was first written and are the reason it now covers seventeen entries rather than fifteen — an inventory that claims to cover every gate has to be extended when a gate arrives, or its central verdict is quietly wrong. All four ran the same extractor, `extract_stability_singles.py`, so the repeatable depth option already reaches every one of their cohorts and only their manifests are missing; what a depth-resolved recomputation needs from them is a fresh forward pass, not new code. Their class-level recomputation is unaffected, and all four retained their full-width states.

**The stability gate's full-width retention pass is complete, which neither of the two accounts of it said.** The gate record stated 4 of 33 arms complete with 8 in progress; the experiment log's entry for the same pass stated 3 of 33 with 5 in progress. Both were mid-flight readings of the same waves and neither was the final state. Verified in-pod on 2026-09-24, every one of the 33 arms reads status complete at **101 of 101 backgrounds and 25,957 sequences scored**, with 101 projected and 101 full-width archives each and features shaped (rows, 4, hidden width) at hidden widths of 384 to 7,168. The gate record is corrected; the log entry is left as the chronology it is. So the stability gate's class-level recomputation has no outstanding retention prerequisite, and what remains for it is the selection and its adapter.

**A prerequisite of this pipeline's own that the same probe found, and has since fixed: the admitted admission receipt was not staged on the allocation.** The pipeline-identity tolerance is read from `results/transfer/readout_20260923/final_admission.json`, and `results/` is not tracked, so that receipt existed on the workstation and not in a pod — where the retained arrays are and where a full-cell replay has to run. The driver refuses rather than defaulting, so the recomputation could not have run in-pod at all. The receipt is now staged at the same path and digest-verified against the workstation copy, `79f6d2a02d52e14b79b5b041eba541d652c34fb573039c35b29c0ef8c7b4d9a3`, and reads `baseline_tolerance.atol = 1e-08` in the pod.

**A full audit of what else the driver and the gates read found that one file and nothing more.** Every other artefact is on the allocation, some at paths this document first guessed wrong, which the audit corrected: the anchor cohort is at `data/context_mutation_rescue/cohort.json` at the bound digest `4093ac34…` rather than only in the workstation's `logs/`; the depth campaign's archives are under `results/external_baseline/<wave>/depth_extract_<arm>/` with the sharded arms merged under `depth_merge_<arm>/`, 104 directories, not under a `results/transfer/readout_depth_20260923/` path; the pairwise extraction plan is at `data/pairwise_epistasis/extraction_plan.json`, content digest `e338420f…`; and the admitted analysis reports the replay is read against are in-pod as 18 `readout_analysis_*` directories. Present and complete: 33 stability `full-` and 33 `fit-` directories, 28 and 73 local-context cell directories, 33 stability fit records, 33 pairwise extraction and 33 pairwise fit directories, and the global-context fits. The `logs/` tree is absent from the allocation by design and holds no input the driver needs. With the receipt staged, the transcribed `PIPELINE_TOLERANCE = 1e-8` in the depth sweep's analysis stage became the one safe line it was waiting to be, and that stage now reads the tolerance from the receipt through the same function this driver uses, with `--admission-receipt` for a run whose code is a frozen snapshot and a refusal rather than a default when the receipt cannot be read.

## The targeted three-depth extraction, planned and undispatched

The two depth-blocked gates matter more than an inconvenience, because the depth sweep resolves depth on its own cohort and finds information there: three residue-level protein arms resolve above zero at depths the admitted extraction never hooked, at the same block at all three split seeds, while the literal-text falsification control holds at every depth with 0 of 336 cells resolved above zero and 314 below. Until a depth selection can be applied to the stability and residue-interaction cohorts, a *not detected* reading on their representation halves cannot be distinguished from a measurement-limited one. Audit L53 carries that as an open limitation.

What removes it is not a full depth sweep of those cohorts but a targeted set: each arm's **selected** depth plus the two admitted depths retained alongside, so the new cells are comparable with the published ones on the same arrays. An arm whose selected depth already coincides with an admitted one hooks two rather than three. `src/transfer/targeted_depth.py` holds the plan at content digest `b85de9a89f08448fd69a1b616676acbd9ff60e7acfdd0d651da2dcfcc78cd01b` and `scripts/transfer/build_targeted_depth_campaign.py` writes it.

**The cost is measured, not estimated.** Capturing a third block is read from the same forward pass, so the GPU cost of a three-depth extraction is the wall clock the completed two-depth extraction of that cohort already recorded, arm by arm, from its own manifests. Only the retained storage grows, and it grows exactly in proportion to the depths hooked.

| Cohort | Arms | GPU-hours, measured | Retained state, 2 depths | Retained state, 3 depths | Longest arm |
| --- | ---: | ---: | ---: | ---: | --- |
| 101 MegaScale groups, 25,957 states per arm | 33 | 13.891 | 29.75 GiB | 44.63 GiB | galactica-30b, 1.559 h |
| 64 cycle groups, 12,977 states per arm | 33 | 4.596 | 14.88 GiB | 22.31 GiB | galactica-30b, 0.792 h |
| Both | 66 cells | **18.487** | 44.63 GiB | **66.95 GiB** | — |

The storage model is validated against what the completed pass actually wrote: over the 33 arms of the stability cohort it gives 29.75 GiB at two depths against a measured 29.79 GiB, and the worst single arm differs by 0.002 GiB.

**The lanes are queueable.** Both extractors now accept a repeatable `--extra-block-index`, through one shared rule — `readout_extraction.hooked_block_indices` — which replaced the `[(len(blocks) - 1) // 2, len(blocks) - 1]` expression each of them carried. Four properties hold and are tested against both real extractors rather than a stand-in. With the option absent the hooked set is bit-identically what it was, for every block count in the panel and across the whole plausible range, which is what 33 arms of completed extraction and every published cell rest on. The admitted pair stays at feature positions 0 to 3 and extras are appended in ascending order after it, so an archive with extra depths still presents the admitted four blocks where a reader of the admitted layout expects them. The hooked set is recorded in the receipt twice over, as `block_indices` and as a `feature_blocks` name per block and pooling rule, so a later reader can tell which blocks an archive holds rather than inferring it from a count. And an index outside the stack is refused rather than clamped, because clamping would hook a block the caller did not ask for and record it as the one it did. The projected archive is untouched: the projection is still formed over the admitted four blocks alone under seeds 20260923–20260926, so a published cell reads a byte-identical array whether or not extra depths were hooked, and extras are retained at full width only — which is what a depth-resolved refit reads anyway, and inventing projection seeds for them would declare a contract nothing has declared.

The builder still reads each extractor's source before marking a campaign queueable, because that marking has to follow the code rather than the date it was written. **The one remaining blocking input is the per-arm depth selection** — or the depth-agnostic variant below, which needs none.

### Every block, for the same GPU time

The GPU cost is the forward pass and does not follow the number of hooked blocks, so hooking **every** block of both cohorts costs the same 18.487 GPU-hours as hooking three. Only the storage differs, and the difference is what the decision turns on.

| Variant | Depths hooked over the panel | Stability | Cycle | Two-cohort total | Resolves any later depth |
| --- | ---: | ---: | ---: | ---: | --- |
| Every block | 871 | 480.14 GiB | 240.04 GiB | **720.18 GiB** | yes |
| Every second block | 437 | 240.47 GiB | 120.22 GiB | 360.68 GiB | no |
| Every fourth block | 220 | 120.52 GiB | 60.25 GiB | 180.77 GiB | no |
| Three depths, selected plus the two admitted | ≤ 99 | 44.63 GiB | 22.31 GiB | 66.95 GiB | no |
| The two admitted depths, as today | 66 | 29.75 GiB | 14.88 GiB | 44.63 GiB | no |

The other two depth-blocked gates were not in this costing when it was first made, and they change its scale rather than its conclusion. Per cohort at every block: stability 480.14 GiB, cycle 240.04, remote homology 122.01, Domainome **2,034.65** — **2,876.84 GiB over the four**, which is 7.7% of free space. The Domainome cohort alone is three times the two originally costed, because it scores 109,996 sequences per arm against the stability cohort's 25,957. GPU cost per cohort is the wall clock its own extraction recorded: 13.891, 4.596 and a measured 5.028 GPU-hours for the three complete ones, against a scaled 58.865 for the Domainome one whose extraction is in flight — **82.4 GPU-hours over the four**.

### A state-count estimate understates by 1.42, measured

This is the calibration any later estimate on these cohorts should carry, and it rests on one cohort. Scaling the stability cohort's measured total by the ratio of sequences scored predicts **3.530 GPU-hours** for the remote-homology cohort against the **5.028 measured** — the state-count ratio understates by a factor of **1.42**, because a longer target costs more per forward pass than a shorter one and a count of states does not see length. Carried through, Domainome's nominal **58.865** is nearer **83.9 GPU-hours** and a four-cohort sweep nearer **107.4**. The factor is reported beside the plain scaling rather than folded into it: it comes from a single cohort, and a second measured cohort would either confirm it or move it. Any future estimate built on state counts alone should carry both the factor and that caveat.

### Scope: three cohorts now, Domainome deferred

The sweep runs on the three cheap cohorts — folding stability at 13.891, residue interactions at 4.596 and remote homology at a measured 5.028 GPU-hours, **23.5 GPU-hours and about 842 GiB together** — and holds the Domainome one.

The reason is proportion, not parsimony. Domainome alone is 2,034.65 GiB and, on the calibrated figure, about 84 GPU-hours: more than four times the other three combined. And its confirmation claim does not yet exist even at the likelihood level — 9 of 33 arms extracted when last probed, and the sibling remote-homology gate's own likelihood half came back unresolved on both strata. If Domainome's likelihood cell also fails to resolve, a full-depth representation pass there would buy the recomputation of cells nobody would read. Deferring costs nothing in total work, because the GPU cost is per-pass whenever the pass runs; what it costs is time already being spent on that cohort's likelihood extraction. It is revisited when that lands.

One depth across both cohorts costs 22.32 GiB, and the 33 arms carry 871 blocks between them. The footprint is concentrated: `galactica-30b` at 48 blocks and width 7,168 takes 99.81 GiB and `qwen2.5-32b` at 64 blocks and width 5,120 takes 95.05 GiB, so those two arms are 27% of the full sweep.

**The allocation has 37,190.7 GiB free of 55,578.0 GiB, at 34% used**, measured on the store the archives would be written to, where `results/` already holds 1.2 TiB. A 720.18 GiB full sweep is **1.9% of the free space**.

**So a depth-agnostic pass is affordable and a subset is not worth its saving.** Every subset reintroduces exactly the conditionality the pass exists to remove: a depth the reassessment selects that falls on an unhooked block leaves the gate where L53 found it, and the depth sweep has already shown once that the admitted depths were the wrong place to look — +0.053010 [+0.029159, +0.077924] at block 34 of 36 against a near-null at the admitted pair. Halving the storage to buy back a fraction of that risk is the wrong trade at 1.9% utilisation. The recommendation is the full sweep, which also removes the per-arm selection from the critical path: it is currently the single blocking input for the whole recomputation, and a depth-agnostic pass does not need it.

Two bounds on that recommendation. The storage figure is for the full-width archives, and the depth sweep's own convention of one array per depth is what keeps a later fit able to stream one depth at a time rather than holding a stack; a fit over a 720 GiB corpus of states is a scheduling problem of its own, separate from the extraction. And 720.18 GiB is a measured arithmetic consequence of the retained archive shapes, validated at two depths against what the completed pass actually wrote, but it is not itself a measurement of a run that has happened.

`build_targeted_depth_campaign.py manifests --every-block` writes that variant and takes no selection; `--selection` writes the three-depth one. Neither dispatches.

**The archive layout is load-bearing, not incidental.** Beyond the admitted pair each hooked block is written as its own array, `features_d<index>`, holding that block's mean and last summaries. `np.load` on an npz reads a named array whole, so a fit over a 720 GiB corpus of states can stream one depth at a time only if each depth is its own array; a single `features` array per background would make the corpus a scheduling crisis instead. The default — no extra depths — writes exactly the one array it always wrote, byte for byte, because 33 arms of completed extraction and every published cell read that key. Anyone writing a fit over these archives should read one `features_d<index>` at a time and take the hooked order from the receipt's `block_indices` and `feature_blocks`.

### The dispatch decision, and what it is waiting on

The full every-block sweep is the decided variant. Its campaigns are written and frozen-ready as `scripts/transfer/campaign_depthall_gate_*.tsv`, one lane per cohort for `gpt2-large`, and `scripts/transfer/campaign_depthall_wave_*.tsv`, 32 lanes per cohort for the rest, so the gate arm is never extracted twice. The manifest reaches a pod only inside a code snapshot, so a dispatch is one `--freeze-only` plus one queue launch.

**One gate stood before the wave, and it is the property no test in this repository can otherwise reach.** `extract_batch`'s hooking loop needs a loaded checkpoint, so the order in which it registers hooks and concatenates their captures is preserved by construction rather than by measurement — and if that order moved, every archive in the sweep would be silently wrong while looking well formed. The gate is one arm extracted with `--every-block`, whose feature positions 0 to 3 must come out byte-identical to that arm's already-retained archive: a real end-to-end check on a loaded checkpoint against a known-good artefact.

**It passed.** `gpt2-large` on the stability cohort, every one of its 36 blocks hooked, 101 of 101 backgrounds and 25,957 sequences, cell `exited-ok` in **872.0 s against the 829.2 s** the same arm's two-depth extraction recorded — a 5% overhead for eighteen times the blocks, which is the pooling and the writing rather than the inference, and is the invariance claim measured instead of argued. Each archive holds 36 arrays of shape (rows, 2, 1280). Compared background by background against the retained two-depth archives, the admitted four block summaries are **byte-identical on all 101, at a maximum absolute difference of 0.0**. Every repeat check in the run read `repeat_delta_M_nats=0.0` with a relative L2 of about 4.0e-07 to 4.3e-07.

`scripts/transfer/verify_depth_archive_identity.py` is that comparison as a reusable stage, and it is itself tested on a faithful archive and on the failure that matters — the two admitted blocks swapped, so every number is present and every one is in the wrong place, which it must fail and does.

**All three cohorts are gated and all three pass.** Each was gated on its own cohort rather than on a sibling's result, because the hooking rule is shared but the cohort, the rendering and the archive layout are not.

| Gate | Extractor | Backgrounds | Wall clock against the two-depth run | Maximum absolute difference |
| --- | --- | ---: | --- | ---: |
| Folding and stability | `extract_stability_singles.py` | **101 / 101** | 872.0 s against 829.2 | **0.0** |
| Residue interactions | `extract_pairwise_epistasis.py` | **64 / 64** | 480 s against 144 | **0.0** |
| Remote homology | `extract_stability_singles.py` | **305 / 305** | 360 s against a 548 s per-arm mean | **0.0** |

470 backgrounds compared across three cohorts and two extractors, every one byte-identical at the admitted positions. The overhead is largest on the shortest run, which is what it should be if it is writing rather than inference.

### The defect the gate run caught, and where it actually lives

**The campaign queue derives a cell's output directory from the run id and the lane label alone**, and both cohorts' gate manifests labelled their cell `depth3-gpt2-large`. The second cell found the first's manifest at that path and was reported **skipped-complete** — writing nothing while presenting a clean status file. On the 32-arm wave that would have produced no archives at all for a whole cohort, under a tally showing every cell accounted for: a silent success, which is the failure mode this repository's principles put above all others. No reasoning about the hooking rule would have found it, because it was not in the hooking rule.

Every lane label now carries its cohort's own tag — `depth3-fs-`, `depth3-ri-`, `depth3-rh-`, `depth3-ec-` — the tags are asserted distinct by test, and the manifests were regenerated and re-frozen. The stability gate's verdict is unaffected: it ran first, under the old label, and its archives are the ones verified.

The hazard is catalogued as audit **L54**, scoped method and open, so it does not survive only as a paragraph here: the next person to meet it will meet it behind a clean tally, as this did.

**Whether the same collision is reachable elsewhere: it was reachable only from here, and the queue already enforces what it can.** `h200_campaign_queue.sh` refuses a manifest that names one label twice, with exactly the right diagnosis — *two cells would share one results directory; a resume key has no condition axis* — and exits 2. That check is central and correct, and it cannot see this collision, because the two cells were in **two manifests parsed by two launches**. Of the other campaign builders, `build_class_sweep_campaign.py` puts the arm, the panel and the seed in every label; `build_depth_campaign.py` spans one cohort, so it has no cross-cohort axis to collide on; and `build_nested_gate_campaign.py` takes a required `--gate` from a closed set and prefixes every label with it, which is why the remote-homology and Domainome waves sit in `rh-<arm>` and `ec-<arm>` directories. This builder was the only one that spanned more than one cohort without putting the cohort in the label, and the defect was its own.

**The residual gap is at the queue, and it is not another uniqueness assertion.** The skip is keyed on an expected artefact *existing* at the derived path, not on that artefact's recorded identity matching the cell about to run, so any two launches whose labels coincide — one builder or two, one session or months apart — silently resume each other's output. The smallest complete fix is for the skip to require that the artefact's own `identity` block names the arm and the plan digest of the cell it is about to skip; every extractor already writes one, so the check has its material. That change is **not made here**: `h200_campaign_queue.sh` is shared infrastructure whose success and failure paths are recorded as exercised in a real pod, three campaigns are running through it as this is written, and tightening skip semantics mid-flight could turn another agent's legitimate resume into a refusal. It belongs to whoever owns that layer, with its own validation pass, and is recorded here so the judgement is stated rather than implied by its absence.

So the sweep's gate is resolved and the wave is held only by the ordering below.

## The driver

`src/transfer/recomputation.py` with `scripts/transfer/recompute_representation_cells.py`. Three subcommands, in the order they are used.

`plan` joins a cell roster with the reassessment's per-arm selection and reports which cells can run and which cannot, before any array is read. `replay` refits one cell at the **admitted** selection and is the pipeline-identity control; it is the only subcommand that runs before the reassessment lands. `run` is the recomputation itself.

A cell is a gate, an arm, a panel, a split seed, a selection and the path to the published cell report. The support is not an input: it is read from the published report, so a cell is fitted on the rows its own published increment was fitted on rather than on a separately declared support that could differ. The fitting procedure is the gate's own entry point, called unchanged — for the Readout family, `readout_analysis.evaluate_readouts` — so the folds, the targets, the equal cluster and assay and variant weights, the penalty grid, the tie rule, the label-permutation control and the resampling contract are the published cell's and not this driver's. What the driver contributes is the representation block at the selected class and depth, the identity refusal, and the comparison.

Each recomputed cell reports every published metric of its gate with the recomputed point and interval beside the published point and interval, the resampling unit and draw count both were formed under, the change in the point estimate, and whether the resolved sign changed. A metric whose resampling unit differs between the two reports is marked not comparable with the reason, rather than differenced.

### The refusal conditions

The driver refuses rather than proceeding when any of the following holds. Each is a case in which the recomputed number would not be a recomputation of the published one.

- **Support.** The ordered assay support differs from the published cell's. A different set is refused with the absent and added assays named; the same set in a different order is refused separately, because the fitted rows are concatenated in assay order.
- **Label budget and support counts.** The unit count or the variant count differs.
- **Split seed and folds.** The split seed differs, or any outer fold's held-unit membership differs, or any inner fold's membership differs. The seed alone does not certify the partition: a different unit set under the same seed gives a different partition, so membership is compared directly.
- **Resampling contract.** The resampling unit label, the draw count or the resampling seed differs.
- **Reduction order.** The process has no discoverable BLAS thread pool, so the admitted float32 reduction order cannot be pinned.
- **Depth without a depth-resolved extraction.** The selection names a block or a position-resolved summary the retained states do not hold, or omits the coordinate count or projection seed its axis requires.
- **Pipeline identity.** At the admitted selection, the refitted held-out predictions depart from the published ones by more than the tolerance the admitted admission receipt declares. That tolerance is read from `results/transfer/readout_20260923/final_admission.json` — `baseline_tolerance.atol`, 1e-8 — and a missing or non-positive value in that receipt is itself a refusal. A replay that invented its own tolerance would certify nothing.
- **No selection.** `run` and `plan` refuse without the reassessment's own declaration of class and depth per arm.
- **No adapter, or no cells.** A gate with no representation cell, a gate whose retention cannot serve the selection kind, a gate with no implemented adapter, or a cell whose published report is absent, is refused at plan time with what is missing named.
- **A retained archive that is not there.** An archive the extraction manifest lists and the filesystem does not hold is reported as a refusal naming the file, because that is a retention finding rather than a fault in the loader.

At a selection other than the admitted one the prediction departures are reported and **not** gated: at a new class or depth the held-out predictions are supposed to move, and gating them there would refuse the recomputation for doing its job. What must not move is the support, the folds, the seeds, the weighting and the label budget, and those are gated at every selection.

### Thread count and per-assay blocking

Two properties of the admitted compressed block are part of pipeline identity, and both are delegated to the modules that own them rather than restated: the BLAS thread count is pinned inside `readout_class_sweep.projected_block`, and the rows are blocked by assay through `readout_depth.admitted_design`. The driver adds a measurement rather than a second implementation: `product_blocking_departure` reports how far the whole-panel product sits from the per-assay one on the arrays at hand, with the largest absolute entry it is read against and the BLAS pools it was measured under, so the constraint appears as a number in the cell report.

Measured on one retained extraction this host holds — three assays and nine variants of `proteinglm-7b-clm` at hidden width 4,096 — the driver's per-assay product is bit-identical to an independently assembled per-assay product, the per-assay product is identical at 4 and at 48 BLAS threads at a maximum absolute difference of 0.0, the whole-panel product moves between those two thread counts by up to 5.722 × 10⁻⁶, and the whole-panel product departs from the per-assay one by up to 6.199 × 10⁻⁶ against a largest absolute entry of 18.236. Both departures sit at the same relative scale as the 7.629 × 10⁻⁶ against 723.245 the depth sweep measured on its 212-assay panel, which is what makes the pin and the blocking necessary rather than decorative.

Every identity digest renders each label as a Python value before hashing. `repr(np.float64(0.5))` is `0.5` under NumPy 1.26 and `np.float64(0.5)` under NumPy 2, and `json.dumps` refuses a NumPy integer outright, so a fold label or an assay identifier arriving as a NumPy scalar would either hash differently across interpreters or fail to serialise — the same defect that cost the global-context gate a run on its row-identity digest.

### One real replay, and what it did not reach

The pipeline-identity control was replayed against the artefacts Compute holds. The admitted 201-assay `proteinglm-7b-clm` cell at split seed 20260923 reads as 201 assays, 163 wild-type clusters, 25,728 variants, 2,000 draws at seed 20260923 over the unit *wild-type family at 50% identity*, at identity digest `f3881aa9cd448f5b43dce6561af86613d20804436bc6cc1e6bee59eba39d1864`; substituting the same arm's cell at split seed 20260924 is refused with the split seed, the outer fold membership and the inner fold membership all named. The admitted extraction's own archives are not staged on this host, so the loader refuses the replay by naming the missing archive: a full-cell replay runs in a pod, where those arrays live.

The fit half was therefore verified end to end on a synthetic cell instead — 10 assays in 10 clusters, 80 variants, hidden width 24 — where the admitted-selection replay recovers every design's held-out predictions at a maximum absolute departure of 0.0 and every published increment at a point change of 0.0 with identical intervals, and a switch to the uncompressed class keeps the identity, moves the primary increment by +0.0381 in Spearman, and is correctly not gated.

## The unified gate contract

`src/transfer/gate_contract.py`, content digest `bfe873ba3382c1728842dc08afa26d276ab5084ee5b49022040311595bbc2533`, with `scripts/transfer/check_gate_contract.py`. The contract is the six steps the capability map already declares — endpoint qualification, control qualification, native likelihood behaviour, representation readout, held-family and remote generalization, and causal validation where justified — written in a form a reader or a test can check a gate against. **No gate is refactored onto it.** Each gate keeps its own modules, scripts and record, and what the contract holds is a read-only transcription of the elements each record already carries.

Four rules carry the substance, and each exists because a gate record shows why. A control must carry its own contribution in one of three declared forms. A representation element may not be reported as settled while the reassessment is open. A likelihood element is not deferred, because a likelihood enters as one scalar column and a linear readout of it is close to the full function space over that feature. And a step not reached is named — *unresolved*, *not detected*, *not applicable* and *absent* are four states, never merged — with its reason.

**The three qualification forms are what the first pass of this check got wrong, and the correction is in the contract rather than in the gate.** The form every gate with a fitted readout uses is an **interval**: the control's paired contribution to the baseline it augments, with its resampling interval, so that a control indistinguishable from zero is visible rather than implied — the stability gate keeps a block whose own contribution is +0.00390 [−0.00068, +0.00864] kcal²/mol² and records that this makes *beyond controls* weaker than the phrase suggests. The generation-and-control gate cannot use that form and does not need it, because its controls are generators rather than feature blocks: what has to be established is that each emits the statistics it declares and nothing more, which is a property of every row rather than of an average over them. So **exactness** — a decidable property checked on all 14,467 pairs, as for the corpus fragment's 0 containment and 0 length failures — and **order_statistic** — a signed position on a declared axis, as for the order-4 Markov cohort's −0.0007 nats per residue against a length-matched corpus reference — are declared forms of their own, each carrying what its qualification establishes in place of an interval. Forcing those into interval form would be a weaker claim about a stronger fact, so the contract accepts them and the gate record now states for each control why its form is the tight one. An interval-form contribution that carries no interval and declares no other form is still a divergence.

What makes the check read-only and still meaningful is the transcription check: every quantity a declaration names must occur verbatim in the record it points at. All 78 declared quantities do.

**All fifteen retention entries are declared, and all fifteen conform.** The nine gates with records of their own, the three upstream studies whose retained arrays four of the gates read — the readout panel, the crossed-control study and the D2 ProLLaMA transplant — and the three unlaunched gates, which point at the capability map that states their admission conditions and declare every step absent with its reason, so that an unlaunched gate is a recorded state rather than an omission. The D2 transplant is the one entry whose causal step is reported rather than absent: a parameter transplant is an intervention on the checkpoint, and it localises which group of parameters carries a measured likelihood difference without identifying any computation inside the model.

## What remains before the recomputation can run

1. **The selection.** The reassessment must declare, per arm, the selected class and the selected depth. Nothing runs without it, and nothing should: a recomputation at a default selection would report an increment no reassessment chose.
2. **Five pending adapters, four of them loaders.** The driver loads and refits the Readout family; the crossed-control, local-context, stability and pairwise adapters are declared pending with what each needs, and the plan refuses those gates by name until they exist. Each needs a retained artefact schema that is only readable on the allocation: the retained column-frequency arrays behind the mutation-local profile block for the two anchor gates, the per-background archive layout of the stability gate's full-width retention pass, and the per-background array layout of the pairwise extraction. Writing them against an unread layout would be writing them blind. The fifth, the evolution-and-adaptation gate's, is not a loader at all: that gate refits nothing, so its adapter re-aggregates the crossed-control cells' recomputed per-assay increments over the frozen strata and runs after them.
3. **The cell roster.** One record per published cell — gate, arm, panel, split seed, published report, cohort and extraction manifest — assembled from each gate's own admission receipts.
4. **A full-cell pipeline-identity replay in a pod**, on one real admitted cell, before any recomputed number is read. The receipt it needs is now staged.
5. **The 32-arm wave**, held until the Domainome extraction completes so the bulk of the GPU time stays on the endpoint already in progress. Its gate has passed, its manifests are written and its snapshot is frozen, so the dispatch is one queue launch. Domainome stood at 9 of 33 cells complete, 6 running and 18 pending across four lane groups when last probed.
6. **Domainome's likelihood cell**, before its depth pass is reconsidered. Its manifests are deliberately not written: a manifest on disk here is one somebody decided to run.

## Limitations

The driver enforces identity of support, folds, seeds, weighting, label budget and resampling contract; it does not and cannot enforce that a recomputed increment means the same thing as the published one, which is a matter of the readout class and belongs to each gate's own reading. A recomputed increment is as bounded as the published one it replaces: by the declared controls, the label budget, the cohort, and now by the selected class and depth rather than the admitted ones.

The pipeline-identity tolerance of 1e-8 absolute is the tolerance the admitted admission receipt used for the same comparison. It bounds the refit's agreement with the published predictions and certifies nothing about numerical accuracy against a higher-precision reference. The per-assay blocking and the pinned thread count make a cell reproducible on its own terms; a refit on a different host or thread budget is not guaranteed bit-identical, and the class sweep's own measurement — that a perturbation three orders of magnitude larger moves no reported correlation — is what bounds the consequence rather than a proof of invariance.

The retention declaration is a transcription of what each gate record states it retained. A location that is declared and, when probed on the allocation, absent is a finding the probe produces; a location that is present is present, and the probe does not certify that its contents are the arrays the published fit read. That check belongs to the loader, which verifies the cohort digest, the manifest identity, each archive's checksum and metadata, the exact mutation order, the measured effects and the profile scores before a single row is fitted.

The contract's element declarations are read-only and digest-bound to the records they transcribe, so they cannot drift unnoticed; they are not a second source for any quantity, and a reader wanting a gate's numbers reads that gate's record.

## Evidence and entry points

`src/transfer/gate_retention.py`, `src/transfer/recomputation.py`, `src/transfer/gate_contract.py` and `src/transfer/targeted_depth.py`, with `scripts/transfer/inventory_gate_retention.py`, `scripts/transfer/recompute_representation_cells.py`, `scripts/transfer/check_gate_contract.py` and `scripts/transfer/build_targeted_depth_campaign.py`. Two duplications were collapsed while this was built: the fold-identity digest, which `fit_pairwise_epistasis.py` records and `fit_global_context_gate.py` checks against it, now comes from one `pairwise_epistasis.fold_identity`, and the same gate's local copy of the row-identity expression now imports `pairwise_epistasis.row_identity`, which is what the pairwise record asked a later run of it to do. Both digests are preserved exactly, so every value already recorded under them still verifies. One verification is recorded here rather than left to the suite that would otherwise carry it. The depth sweep's analysis stage now reads its pipeline-identity tolerance from the admitted receipt, and the suite that exercises that stage end to end takes tens of minutes on a contended host — measured at a load average of 54.78 on 192 cores while five agents were working, where this driver's own suite went from 45 s to 371 s across runs. So the two properties the substitution could have broken are asserted directly on the real stage instead: that `fit_cell` still accepts exactly the arguments its existing callers pass with the receipt optional, and that the tolerance comes from the receipt and equals the 1e-8 it declares, with no transcribed copy surviving in the stage. The conditions that must hold are in `tests/test_recomputation_pipeline.py`, 61 tests, covering every axis of identity drift as a separate refusal, the thread-pool and per-assay-blocking properties on synthetic and on real retained arrays, the depth refusals, the tolerance's provenance, the plan's four refusal classes, the NumPy-scalar rendering hazard, the contract's four rules as negative paths, and the admitted-selection replay end to end. Reports are written under ignored `logs/d1_recomputation_20260924/`.

```
python scripts/transfer/inventory_gate_retention.py \
  --out logs/d1_recomputation_20260924/gate_retention.json
python scripts/transfer/check_gate_contract.py --gate local_context \
  --out logs/d1_recomputation_20260924/gate_contract.json
python scripts/transfer/build_targeted_depth_campaign.py estimate \
  --out logs/d1_recomputation_20260924/targeted_depth_estimate.json

# the depth sweep, once a card is idle: freeze once, gate, then the wave
export H200_POD=<running-pod-name>
eval "$(scripts/transfer/run_transfer_h200.sh --freeze-only)"
~/hangzhou-compute/ssh_tunnel/h200_pod_exec.sh -- bash -lc "
  setsid nohup bash '${SNAPSHOT_DIR}/scripts/transfer/h200_campaign_queue.sh' \
    --manifest '${SNAPSHOT_DIR}/scripts/transfer/campaign_depthall_gate_folding_stability.tsv' \
    --snapshot depth3='${SNAPSHOT_DIR}' > /dev/null 2>&1 < /dev/null & disown"
# verify feature positions 0-3 byte-identical to the retained archive, then the
# wave manifests from the same snapshot
python scripts/transfer/recompute_representation_cells.py replay \
  --arm <arm> --panel <panel> --split-seed 20260923 \
  --cohort logs/d1_readout_20260923/cohort.json \
  --manifest <extraction>/manifest_<arm>.json --published <admitted analysis>.json \
  --out logs/d1_recomputation_20260924/replay_<arm>.json
```
