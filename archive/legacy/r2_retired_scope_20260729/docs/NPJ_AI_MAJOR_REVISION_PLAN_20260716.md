# npj Artificial Intelligence major-revision plan

**Date frozen:** 2026-07-16  
**Binding review:** `npj_ai_manuscript_assessment.md`  
**Evidence audit:** `EXPERIMENTAL_PROCEDURES_AND_RESULTS_AUDIT_20260716.md`  
**Disposition:** major revision before submission; not submission-ready

Live implementation/execution state is tracked separately in
`NPJ_AI_MAJOR_REVISION_EXECUTION_STATUS_20260717.md`; this file remains the
binding frozen task and acceptance plan.

## 1. Decision and scope

The July manuscript is an evidence-corrected internal audit, not a completed
response to the independent assessment. The assessment requires nine
priority-zero (P0) packages. None passed at plan freeze. P0-1 and P0-9 have
partial infrastructure; P0-2 through P0-8 require new experiments.

The computational route is the binding minimum. Wet-lab validation would
strengthen biological claims but is not treated as a computational task that
can be completed autonomously. Author metadata, coauthor approval, a release
tag and DOI minting also require human or external authority.

Until every relevant gate passes, the safe scientific summary is:

> One fixed, cohort-sensitive procedure identified 38 recurrent sparse-readout
> triplets across three protein generators relative to an implemented
> pair/layer-specific reassignment null. A separate UniRef50 cohort recovered
> fewer triplets and no canonical feature identities. Existing semantic tests
> are exploratory; N-terminal readouts have high unnormalized received
> attention; ablation negatives are bounded to tested paths; and the
> eight-class steering run is a no-positive-evidence pilot with a feature-sign
> selection defect.

## 2. Current evidence and resource state

### Completed and retained

- The invalid 0.1-nat Swiss-Prot gate was removed and replaced by a
  deterministic exploratory 2,000-permutation re-audit.
- CLT hook terminology, padding exposure, heuristic steering endpoints,
  site-mismatched intervention and wider-dictionary limits are disclosed.
- The official Springer Nature package, six historical figures, ten
  supplementary tables and processed source-data checksums build.
- Existing feature/head intervention rows are reported rather than selected.

### Submission-critical deficits

- The exact central cohort ordering and the three reference CLT files named by
  the saved atlas are now locally recovered and hash-verified; immutable
  historical upstream revisions and a persistent licensed deposit remain
  missing.
- Dictionary training and evaluation are unmasked and single-seed.
- Discovery and assessment overlap; the atlas null is not coherent model-wise.
- Continuous conditional semantics and N-terminal counterfactuals are absent.
- The steering selector includes 19 negative-attribution interventions across
  six class/layer cells.
- The causal pipeline has no realistic positive-control mechanism or
  equivalence-bounded negative.
- Recoverability is non-nested, single-seed and based on only 48
  decoder-native generated examples.
- Historical Figure 5 mixes runs and asymmetric selected samples.

### Compute snapshot

The Hangzhou access note is `/home/lzp/hangzhou-remote/README.md`; the former
`H200_BIOCC_STATUS.md` was removed. A live read-only query on 2026-07-16 found
both H200 nodes fully requested (8/8 plus 8/8), one unrelated pending GPU pod,
and no running BioCC/`zhk-zip` pod. Local L20 GPUs 4--7 were idle, but full
8,192/16,384-wide Adam CLT retraining is H200-class. Capacity must be rechecked
before each remote launch.

### 2026-07-17 execution update

The user-provided pod `damoxing-zhk-zipbio-master-0` exposes four H200s. The
refreshed helpers now live under `~/hangzhou-remote/ssh_tunnel/` and pass their
reported health checks. A separate cross-view probe nevertheless showed the
master path mounted from `/dev/vda3` as ext4 and the same probe path absent in
the pod's GPFS view. The helper's `GPFS=read-write` result tests only the
master-local directory, not master--pod visibility. Versioned payloads
therefore require an explicit, hash-verified `kubectl cp` bridge after the
documented master transfer. The configuration file is mode 600, but the helper
still supplies its gateway credential as a Plink/PowerShell command argument;
that is a hardening item, and no credential value is retained in project
artifacts. Neither discrepancy blocks the current runs, so no user-side H200
intervention is presently required.

P0-2 now has a frozen protocol (`P0_2_DICTIONARY_PROTOCOL_20260717.md`), strict
split tooling and a successful four-step mask-aware ZymCTRL H200 smoke run.
P0-3/P0-4 have validated infrastructure and an execution contract
(`NPJ_REVISION_P0_3_P0_4_EXECUTION.md`); P0-7/P0-8 have a separate confirmatory
protocol (`P0_7_P0_8_CONFIRMATORY_PROTOCOL_20260717.md`). These are
infrastructure milestones only: no corresponding P0 scientific gate is marked
passed.

The exact historical balanced-200 JSON named by the May atlas provenance was
subsequently recovered from archived pod storage. Its file SHA-256 is
`5bc7697a83cc7461558f8b4597a3c9b4d6a151b7ec70ca22efc7282ecde4f0a6`;
all 200 ordered record objects exactly equal the prior reconstruction before
added provenance fields, retaining ordered-record SHA-256
`07213d4a9cefbdb055206e08d3137722c446acf43b5b3342db571b977032c724`.
The three exact historical reference CLT files named by the saved atlas output
were also located on archived compute storage and hash-verified: ProtGPT2
`5eca3b19284dbd9b302078e3a7e34ce7a2fc78d97b1566eae927d4d1c30f1f00`,
ZymCTRL
`5da70c530b83a034d1fe683a72a8cc5bd7b49463d2598036cd6b5db94ca5761d`
and ProGen2-medium
`5e384733dc28ecad3947b65c0c8b34f058ce50a61aab67399548c2b21687b8fd`.
Their retained YAML configurations and recovery manifest are under
`evidence/historical_reference_checkpoints_20260717/`. P0-1 therefore no
longer has a local cohort-order or reference-CLT-location blocker. It remains
open because complete upstream identity is model-specific: the deployed
ProtGPT2 model/tokenizer tree is verified to commit
`f71aa6cf063ad784ebd53881d11332fd098eaa58`; ZymCTRL is best-supported at
`3c532ef172b9cd2e95238baadf5167ebb89fbc32` with its weight verified but
strict whole-tree proof incomplete; and the deployed ProGen2-medium tree is a
hybrid that matches no single upstream commit and therefore requires a
deposited local snapshot. Complete raw artifacts, a tagged release and a
DOI-backed licensed deposit are also still absent. Local recovery is not
archival release.

## 3. Minimal implementation contract

The revision must not clone the legacy pipeline into another large tree.
Historical scripts/results remain provenance. Foundational behavior is fixed
in existing model/trainer/evaluator files, while new confirmatory work shares:

1. one immutable cohort schema:
   `{id, source, sequence, split, family, sha256}`;
2. one strict run manifest containing ordered-cohort, source, model,
   tokenizer, checkpoint, config, code, tool and database hashes;
3. one strict JSON writer that rejects non-finite values;
4. cached activations reused by matcher/null sensitivity analyses; and
5. fail-fast gates with no silent fallback from a missing positive candidate
   to an opposite-sign intervention.

Large outputs remain under `results/` or `logs/` so synchronization ignores
them. Every run records GPU/host memory before and after execution.

## 4. Itemized revision tasks

### P0-1 — Exact reproducibility

**State at freeze:** partial; gate failed.

Tasks:

1. Reconstruct the documented alternating order of the retained 100 real and
   100 random records; save IDs, sources, sequences, ordered SHA-256 and source
   hash. A common row permutation leaves correlations unchanged, so atlas
   equality alone cannot prove historical order; label the reconstruction as
   validated only after rerunning the canonical atlas and resolving independent
   provenance evidence.
2. Hash local and H200 model weights, configs and tokenizer files; record that
   historical Hugging Face commit IDs were not retained rather than inventing
   them.
3. Hash all reference/wider CLTs and executed configs; retain training and
   evaluation cohort manifests.
4. Archive full generation JSONL, seeds, sampler settings, intervention rows,
   raw intervention outputs and tool/database versions.
5. Replace non-finite JSON constants with `null` in release copies, attach
   explicit JSON-path/status metadata and retain hashes of historical inputs.
6. Add a machine-readable panel/table provenance map. The current historical
   audit is `PANEL_TABLE_PROVENANCE_MAP_20260717.json`; its explicit missing
   fields must be replaced by content-addressed confirmatory manifests before
   this task can pass.
7. After human approval, tag the exact code and mint DOI-backed code,
   checkpoint and data deposits.

Acceptance criterion: an independent party can regenerate every main figure
and table from immutable deposited inputs. A processed checksum package alone
does not pass.

### P0-2 — Mask-aware, replicated, quality-gated dictionaries

**State at freeze:** not run; gate failed.

Tasks:

1. Return both `input_ids` and `attention_mask` from batching.
2. Pass the mask into each frozen model forward.
3. Exclude invalid token rows from MSE, target variance, FVU, L0, firing/dead
   tracking, resampling probabilities and held-out diagnostics.
4. Add synthetic tests proving that changing padded token values does not
   change loss, FVU, firing or resampling, plus an all-valid equivalence test.
5. Remove the unused `init_dec_norm_scale` and `dec_norm_weight` YAML keys
   unless an explicitly tested implementation is added.
6. Freeze disjoint train/validation/test manifests before training.
7. Train at least three (target five) seeds for every model and dictionary
   family.
8. Compare TopK CLT with at least two sparse alternatives (JumpReLU or
   conventional SAE and gated SAE) plus a matched dense low-rank control.
9. Report seed/layer FVU, dead fraction, firing-frequency and decoder-norm
   distributions and reconstruction-error quantiles on the held-out test set.
10. Freeze the quality gate before atlas construction; do not build a
    biological atlas from a failed dictionary.

Acceptance criterion: zero padding contamination, immutable splits, seed-level
held-out quality estimates and passing prespecified quality gates for every
dictionary used downstream.

### P0-3 — Independent atlas replication

**State at freeze:** not run; gate failed.

Tasks:

1. Discover feature pools and alignments on cohort A.
2. Freeze feature identities, layers and matches, then score signed
   correlations on independent cohort B.
3. Report both positive-only and absolute-correlation analyses; do not equate
   anticorrelation with the same concept without justification.
4. Draw one permutation per model/replicate and reuse it across all pairs and
   layers; run at least 1,000 replicates and use plus-one empirical p-values.
5. Compare greedy, Hungarian, optimal-transport and joint three-way matching.
6. Sweep feature-pool size, layer mapping and correlation threshold using
   prespecified grids.
7. Report discovery count, held-out signed correlation, identity Jaccard,
   match ambiguity/confidence and null distributions.
8. Estimate variance attributable to cohort, model seed, dictionary seed and
   matcher.

Acceptance criterion: correspondences retain meaningful held-out correlations
and quantified stability across cohorts and methods. Otherwise report
procedure dependence as the result.

### P0-4 — Continuous conditional semantics

**State at freeze:** top-event repair only; gate failed.

Tasks:

1. Save continuous activation values rather than only 100 selected events.
2. Prespecify label families, k-mer model, position basis, input-norm, protein
   length and sequence-source covariates.
3. Use protein-blocked and family-blocked folds and within-protein conditional
   randomization.
4. Match label prevalence and reuse identical folds/dimensionality controls
   for sparse, dense and randomized representations.
5. Test incremental held-out prediction from the biological label after
   conditioning on low-level covariates.
6. Correct across features, layers and labels; bootstrap at protein level.
7. Include negative labels/random dictionaries and a prospective power or
   smallest-effect analysis.

Acceptance criterion: a replicated residual biological association or a
powered prespecified bound showing the residual is small. The current 0/380
top-event result cannot pass this gate.

### P0-5 — N-terminal counterfactuals

**State at freeze:** descriptive only; gate failed.

Tasks:

1. Normalize each key's received attention by its number of eligible causal
   queries.
2. Match proteins by length and normalized position and control features by
   firing frequency and CLT-input norm.
3. Evaluate held-out natural `MXX`, `M->A`, internal `MXX` insertion and
   artificial internal-truncation conditions.
4. Separate BOS-token, initiator-methionine and generic early-position effects.
5. Measure the sparse feature before any attention intervention and test
   attention-mediated effects conditional on position.
6. Report paired per-protein effects and uncertainty.

Acceptance criterion: clear separation, or demonstrated nonseparation, of
methionine, position and received-attention contributions. Do not use
“attention sink” without passing evidence.

### P0-6 — Corrected eight-class steering

**State at freeze:** defective historical pilot; gate failed.

Tasks:

1. Rank/select positive direct effects on an independent held-out sequence set.
2. Require the requested number of positive features or fail the cell; never
   substitute negative-attribution rows.
3. Package the full attribution table and selection decision for every cell.
4. Use paired random streams where scientifically appropriate and preserve
   every sequence, prompt, token seed and sampler setting.
5. Include prompt-only, random-feature and norm-matched-feature arms.
6. Run a prespecified multiplier/site dose grid and correct for eight classes
   and all searched cells.
7. Use validated generation-wide class endpoints; Pfam/CLEAN and symmetric
   structure samples are supporting, correlated computational endpoints rather
   than functional activity.
8. Freeze equivalence regions before generation.

Acceptance criterion: a clean positive or an equivalence-bounded negative for
all eight classes. The historical pilot remains provenance only.

### P0-7 — Causal redesign and planted positive control

**State at freeze:** bounded historical negatives only; gate failed.

Tasks:

1. Split feature/head discovery from intervention evaluation.
2. Match controls on layer, firing frequency, mean activation, decoder norm,
   direct logit effect, received-attention mass and reconstruction
   contribution.
3. Sweep feature, layer, intervention site, strength and model.
4. Record intended feature change, off-target sparse-code changes,
   reconstruction displacement, logit displacement and sequence behavior.
5. Add path patching or causal mediation for a validated property.
6. Freeze smallest effects of scientific interest and use TOST, ROPE or
   confidence intervals against equivalence bands.
7. Build the assessment's planted protein grammar with known motif, position,
   length, family, long-range and N-terminal variables and a known causal
   adapter/direction.
8. Measure recovery sensitivity, specificity, false-discovery rate, path
   localization and intervention-effect recovery across model seeds.

Acceptance criterion: the same pipeline detects and controls the planted
mechanism at realistic effect size. Only then can target-feature nulls support
causal bounds.

**2026-07-17 execution note:** the one-time prospectively frozen synthetic
subgate passed across model seeds 11/29/47. This establishes sensitivity only
for the fixed planted comparator and effect size. The held-out pretrained-model
target/control executor, measurements and equivalence adjudication remain open;
historical target nulls are not upgraded.

### P0-8 — Nested repeated recoverability

**State at freeze:** exploratory; gate failed.

Tasks:

1. Store identical folds for CLT input, code, reconstruction and every
   baseline.
2. Select layer only inside the outer training fold; assess it on the untouched
   outer test fold.
3. Repeat analysis seeds and dictionary seeds.
4. Report paired group-bootstrap intervals for `F-C` as well as ratios.
5. Increase the decoder-native cohort and use identity-aware splits.
6. Add matched-dimensional PCA, random projection, NMF/ICA and random
   dictionary controls.
7. Add local tasks for contacts, motifs, active sites or structural
   neighborhoods.
8. Relate reconstruction error to recoverability and intervention efficacy.

Acceptance criterion: paired intervals identify which signals are reliably
retained and how retention depends on dictionary quality.

### P0-9 — Figures, chronology and provenance

**State at freeze:** partial infrastructure; gate failed.

Tasks:

1. Replace Figure 5 with one immutable, equal-arm, identically selected run
   using one ESMFold setup and one Foldseek database revision; show full
   distributions and uncertainty. Move post-selected structures to the
   Supplement unless selection was prospective.
2. Rebuild Figure 1 around formal recurrence (`R`), residual semantics (`S`),
   causal computation (`C`) and external utility (`U`), each with independent
   data and a gate.
3. Replace Figure 2 with cohort/seed/matcher/layer/pool/threshold stability.
4. Show conditional semantic effects, N-terminal counterfactuals,
   intervention-fidelity/dose/equivalence panels, corrected generation-wide
   steering, quality--recoverability relationships and checkpoint trajectories
   only after their canonical outputs exist.
5. Map every panel/table to one experiment, command, version set and hashes.
6. Correct all historical chronology and never combine runs silently.

Acceptance criterion: every display is regenerated from immutable canonical
outputs and has a complete provenance row.

## 5. Model-panel and positioning requirement

ProtGPT2 and ZymCTRL share a GPT-2-family architecture; ProGen2-medium supplies
the principal architectural contrast, and CLT input sites are not homologous.
The revision must either add an encoder-style PLM plus another independent
autoregressive family, or use controlled small models in the planted benchmark
to separate:

`architecture x corpus x tokenizer x dictionary seed`.

Until then, use “three-model recurrence”, never “architecture-independent
conservation”. Reposition the work as an evidence-calibrated evaluation
framework only after the framework's prospective gates have been executed.
The completed synthetic calibration varies seeds and rotations within one
architecture, corpus construction and token vocabulary; it therefore does not
satisfy this factorial decomposition requirement.

## 6. Immediate no-GPU correction register

The following corrections are required regardless of future experimental
outcomes:

- [x] change project status from “revision complete” to “major revision”; 
- [x] disclose cohort sensitivity and the pair/layer-specific null;
- [x] remove “reproducible” from the current atlas claim;
- [x] disclose the steering selector's 19 negative-attribution interventions;
- [x] distinguish the older lysozyme intervention from the direct-effect pilot;
- [x] disclose the current Figure 5 mixed-run provenance and chronology;
- [x] correct record-level basis-probe folds and attention-pilot bootstrap text;
- [x] package the full direct-effect candidate table;
- [x] create strict JSON release copies with explicit non-finite-field metadata;
- [x] recover and validate the exact ordered historical cohort locally;
- [x] add the omitted May experiment chronology;
- [x] complete the mask-aware code/tests and freeze the confirmatory protocol.

Checkboxes record artifact work only; they do not imply that any P0 scientific
gate has passed.

## 7. Final submission gate

Submission remains blocked until:

1. P0-1 through P0-9 meet their acceptance criteria;
2. the positive-control benchmark establishes pipeline sensitivity;
3. the model-panel limitation is resolved or the scope is narrowed explicitly;
4. new experiments are logged in the project and repository chronologies;
5. all figures, tables, text and source data agree with immutable manifests;
6. verified author, affiliation, funding, compute and contribution metadata
   replace placeholders; and
7. coauthors approve the final tagged, DOI-backed release.
