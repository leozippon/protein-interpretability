# P0-7/P0-8 confirmatory protocol

**Frozen:** 2026-07-17 CST  
**Binding assessment:** `npj_ai_manuscript_assessment.md`  
**Parent plan:** `NPJ_AI_MAJOR_REVISION_PLAN_20260716.md`  
**Status:** prospective synthetic sensitivity gate passed once; real
pretrained-model gates remain unrun and failed by default

## Scope and claim boundary

This protocol implements the two methodological controls required by assessment
packages P0-7 and P0-8. It does not reinterpret the historical intervention or
recoverability results. A passing synthetic control establishes that the
revised analysis can recover a planted mechanism at the tested effect size. It
does not establish that a sparse feature in ProtGPT2, ZymCTRL or ProGen2 is
causal. Likewise, a successful synthetic nested-CV test validates fold and
estimator plumbing, not recoverability in the protein-model cohorts.

## P0-7: planted causal control and equivalence analysis

### Frozen design

`scripts/54_run_planted_causal_control.py` generates two immutable, disjoint
cohorts from the same synthetic protein grammar:

- a discovery cohort used for candidate ranking only;
- an evaluation cohort used for effect, mediation and equivalence estimates
  only.

Every record stores the complete sequence, split, source, family, motif,
motif-position class, length, long-range dependency, controlled N-terminal
residue and sequence SHA-256. The grammar varies all variables requested by the
assessment: motif identity, motif position, length, family, long-range state
and N-terminal state.

For each model seed, a different orthogonal rotation hides one known causal
sparse direction at a different feature/layer identity. The direction has a
known mediated effect on a paired binary sequence endpoint. All candidate
directions have unit decoder norm, comparable general logit displacement and
the same feature-profile schema. Candidate discovery uses one-sided direct
effects with Benjamini--Hochberg correction. The evaluation set reports:

1. intended feature change;
2. off-target sparse-code displacement;
3. reconstruction displacement;
4. target-logit displacement;
5. paired sequence-behaviour change;
6. known mediator change and indirect effect;
7. recovery sensitivity, specificity, false-discovery rate, path localization
   and intervention-effect recovery across model seeds; and
8. TOST results for same-layer controls matched on firing frequency, mean
   activation, decoder norm, general direct-logit-effect norm, received
   attention mass and reconstruction contribution.

The equivalence band is frozen before execution. The default matched-control
band is `[-0.10, +0.10]` target-logit units. TOST uses alpha 0.05, hence the
corresponding 90% interval, while the report also includes a 95% effect
interval. Failure to reject either nonequivalence null is reported as
inconclusive, never as equivalence.

### Synthetic positive-control gate

The synthetic control passes only if all conditions hold:

- sensitivity at least 0.80;
- specificity at least 0.95;
- false-discovery rate at most 0.10;
- every seeded planted path is localized while all matched controls meet their
  frozen equivalence test; and
- every seeded intervention-effect recovery ratio lies in `[0.80, 1.20]`.

These thresholds apply only to the planted benchmark. Protein-model target
features require a separately frozen smallest effect of scientific interest,
held-out discovery/evaluation sequences, matched controls and dose/site/model
sweeps. Historical nonsignificant results cannot be upgraded to equivalent
effects retrospectively.

### Canonical command and artifacts

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ct
python r2_interpretability_transfer/scripts/54_run_planted_causal_control.py \
  --model-seeds 11 29 47 71 89 --per-split 256
```

The output directory contains strict `summary.json`, `cohort.jsonl`, complete
candidate and intervention JSONL rows, and a run manifest with command,
versions, source hashes and artifact hashes. JSON rejects NaN and infinity.

### Trained autoregressive development smoke (non-confirmatory amendment)

The v1 control above rotates simulated feature arrays; it validates analysis
plumbing but does not satisfy the assessment's request to train a language
model and sparse dictionary. `scripts/62_run_trained_positive_control.py`
therefore adds a separate, bounded training-path smoke with three immutable
splits:

- a training split optimizes each small causal recurrent language model and
  the repository's `CLTForTraining` ReLU--TopK dictionary;
- a discovery split alone selects a dictionary feature using its direct
  endpoint effect and its long-range-variable association after adjustment for
  motif, motif position, length, k-mer count, family and N-terminal state; and
- an evaluation split alone scores cross-seed activation alignment, endpoint
  accuracy, dose recovery, wrong-layer/wrong-position localization and matched
  dictionary controls.

Every input is an autoregressive conditional protein record that exposes the
long-range state as an explicit `<lr0>` or `<lr1>` token, followed by a family
token, amino-acid sequence and terminal query. The two-layer model has a
seed-specific orthogonal hidden-space rotation. A fixed sparse adapter copies
that exposed variable into layer 1 and has exclusive readout access to the two
endpoint tokens. All other language-model parameters are trained by next-token
loss. This makes the endpoint and planted intervention effect intentionally
easy and analytically prescribed; it tests training, dictionary and
intervention wiring, not whether a language model learned a latent biological
mechanism.

The implementation records development thresholds of at least three unique
model seeds, sensitivity at least 0.80, specificity at least 0.95,
false-discovery rate at most 0.10, endpoint accuracy at least 0.80, activation
Spearman correlation at least 0.80, dose-effect recovery in `[0.80, 1.20]`, CLT
FVU at most 0.50, and TOST intervals inside `[-0.50, +0.50]` log-odds units.
These are not a frozen confirmatory gate: the same evaluation cohort was
inspected after 100 CLT steps and reused when training was increased to 150
steps. The reported feature controls are the nearest available candidates
subject to ground-truth orthogonality, but their matching distances span
2.82--4.14; they are not well-matched controls, so their equivalence results do
not validate the control design.

Dictionary recovery means positive cosine at least 0.70 to the known rotated
adapter direction; this direction-level rule allows a TopK dictionary to split
one mechanism across redundant coordinates without choosing a post-hoc feature
identity. State tensors, vocabulary, cohorts, source files and artifacts all
receive SHA-256 digests; the development run also stores one model-plus-CLT
checkpoint per seed under `checkpoints/`.

The bounded CPU development command is:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ct
python r2_interpretability_transfer/scripts/62_run_trained_positive_control.py \
  --model-seeds 11 29 47 --per-split 48 --clt-steps 150 --device cpu
```

`--device cuda` runs the same code path when a bounded accelerator run is
explicitly scheduled. This post-hoc smoke does not pass P0-7, validate a
mechanism in ProtGPT2, ZymCTRL or ProGen2, or upgrade any pretrained-model
negative to equivalence. A prospectively frozen benchmark with an unexposed
latent dependency, untouched evaluation cohort and genuinely matched controls
is still required.

### Prospective unexposed long-range benchmark v2

The remaining synthetic sensitivity control is implemented by
`scripts/69_run_prospective_positive_control.py` and
`src/revision/prospective_positive_control.py`. It is deliberately separate
from both earlier controls. In particular, it does not relabel the exposed
`<lr0>`/`<lr1>` GRU smoke as confirmatory and does not upgrade any legacy
pretrained-model null.

The v2 grammar uses only BOS, EOS and the 20 canonical amino acids as
model-visible tokens. In each sequence, residues A/C at zero-based positions 8
and 32 define a genuinely distal equality relation; neither anchor has marginal
information about the relation. The relation determines the *future* canonical
endpoint residue (W for unequal, Y for equal), which is causally masked at the
endpoint-prediction site. There is no relation, query, family, control or
endpoint-label token. Each split independently balances target state, both
anchor marginals, motif, motif-position class, length, family and N-terminal
state. Two additional balanced equality relations at positions 10/34 and 12/36
provide active, same-layer, target-endpoint-orthogonal control mechanisms.

For each of model seeds 11, 29 and 47, a two-layer causal autoregressive model
trains its content parameters by next-token loss. A seed-specific orthogonal
rotation hides a fixed sparse carrier/adapter path that becomes available only
after the distal anchors. The target path controls W-versus-Y log odds; the two
nuisance paths have equal-strength general-logit readouts but exactly zero
W-versus-Y effect. This fixed planted comparator makes causal truth known; it is
not represented as an emergent biological circuit. The repository's
`CLTForTraining` ReLU--TopK implementation is then trained on the training split
only. The minimal production geometry is 10 coordinates with TopK 3, matching
the three planted sparse relations without the redundant 32-coordinate
development geometry; frozen carrier and adapter gains are 0.75 and 1.50.

Discovery alone fits the prespecified conditional activation model, adjusting
for motif, motif position, length, k-mer count, family, N-terminal state, both
target-anchor states and both nuisance relations. Positive coordinates require
all of: a positive conditional coefficient, BH-adjusted one-sided
`q < 0.05`, and direct endpoint effect at least the value implied by the frozen
0.70 known-direction cosine. Decoder rows that never activate are recorded but
cannot be ground-truth recovered features. Sensitivity is defined for the one
planted mechanism in each model seed (detected/not detected); specificity and
false-discovery rate remain coordinate-level. This avoids incorrectly treating
a TopK split into redundant aligned coordinates as several independent planted
mechanisms. Cross-model alignment uses the sum of discovery-selected sparse
activations weighted by their frozen known-direction cosine, rather than
post-hoc selection of the most correlated coordinate.

Only after model fitting, CLT fitting, selection and control construction are
complete is the assessment split opened. It reports endpoint accuracy, CLT FVU,
intended-direction and off-target displacement, complete doses 0.5/1.0/2.0,
wrong-layer and wrong-position arms, downstream W/Y log-odds and probability
changes, effect recovery and cross-seed activation alignment. The two nuisance
controls use orthogonal unit directions and the same discovery activation gate
as the target. They are matched at the frozen layer/site on firing frequency,
mean activation, unit decoder norm, general direct-logit-effect norm, received
attention mass and reconstruction contribution. No nearest-neighbour fallback
exists. Both component-wise hard calipers and a maximum standardized distance
of 2.5 must pass. The absolute firing-frequency and attention-mass calipers are
0.20 and `1e-12`; the absolute log-ratio calipers for mean activation, unit
decoder norm, direct-logit-effect norm and reconstruction contribution are
1.00, 0.35, 0.70 and 1.00. Execution fails rather than widening a caliper.

The negative smallest effect of scientific interest is fixed at 0.50 W/Y
log-odds units. Wrong-layer, wrong-position and both nuisance-direction effects
must each pass paired TOST at alpha 0.05 inside `[-0.50,+0.50]`. Failure of
either one-sided test is inconclusive, not equivalence. Every model seed must
individually attain sensitivity at least 0.80, specificity at least 0.95, FDR
at most 0.10, endpoint accuracy at least 0.80, mean and layer-1 FVU at most
0.50, path localization, all negative equivalence tests and dose recovery in
`[0.80,1.20]`. Every seed pair must have confound-adjusted activation Spearman
correlation at least 0.80. Aggregate means are descriptive and cannot rescue a
failing seed.

#### Development audit before the immutable run

The hard gates and calipers were not relaxed during development:

- The first 32-coordinate/TopK-4 fixture (split seeds 9101/9201/9301) failed as
  intended: seed sensitivities were 0.40/0.33/0.33, FDR was
  0.33/0.75/0.00 and two alignment pairs were below 0.80, although path,
  effect-recovery and negative-equivalence checks passed. Learned dictionary
  coordinates could not supply genuinely matched orthogonal controls; close
  anti-direction coordinates were rejected rather than admitted.
- Adding two planted orthogonal nuisance mechanisms supplied valid controls,
  but a 12-coordinate/TopK-3 fixture (11101/11201/11301) still failed because
  seed 29 had sensitivity 0.50. Discovery showed that the extra nominal
  positive was a 2/48-firing decoder row with a negative semantic coefficient,
  not a second planted mechanism. The other two seeds had sensitivity 1.0 and
  every endpoint, FVU, localization, equivalence and alignment gate passed.
- After correcting the estimand to one planted mechanism per seed and reducing
  redundant detector geometry, a wholly new fixture
  (12101/12201/12301) passed every unchanged per-seed gate: sensitivity,
  specificity and FDR were 1.0/1.0/0.0 for all seeds; endpoint accuracy was
  1.0; layer-1 FVU was 0.039/0.176/0.071; all path and negative-equivalence
  checks passed; all dose-recovery ratios were numerically 1.0; and pairwise
  alignment correlations were 0.979/0.989/0.982. This is development
  evidence for the detector, not the production result.
- The final source expanded wrong-layer, wrong-position and both nuisance
  controls across all three doses. A further untouched fixture
  (13101/13201/13301) passed the complete surface: every seed again had
  sensitivity/specificity/FDR 1.0/1.0/0.0 and endpoint accuracy 1.0; layer-1
  FVU was 0.037/0.090/0.073; all 18 wrong-site tests and all 18 nuisance-control
  dose tests were equivalent; and pairwise alignment was 0.987/0.976/0.970.
  This is the development qualification for the exact pre-freeze source, not a
  production result.

The production specification is
`configs/p0_7_prospective_positive_control_spec.json`, with distinct frozen
split seeds 4101/4201/4301. The CLI exposes only two stages and no scientific
parameter overrides. `freeze` requires the exact spec SHA-256, publishes the
cohort and source-bound freeze atomically and refuses overwrite. `execute`
requires the exact freeze-manifest SHA-256, verifies every artifact and source,
and creates an exclusive execution claim *before* training. A claim forbids
retry or retuning even when execution fails; results publish atomically only
after a complete run. The canonical forms are:

```bash
python r2_interpretability_transfer/scripts/69_run_prospective_positive_control.py freeze \
  --spec r2_interpretability_transfer/configs/p0_7_prospective_positive_control_spec.json \
  --spec-sha256 <exact-sha256> --out-dir <new-freeze-directory>
python r2_interpretability_transfer/scripts/69_run_prospective_positive_control.py execute \
  --frozen-dir <new-freeze-directory> \
  --freeze-manifest-sha256 <exact-sha256> --out-dir <new-result-directory>
```

#### Immutable production disposition

The exact production spec was frozen and executed once on CPU on 2026-07-17;
there was no retry and no source or parameter change between freeze and
execution. The immutable receipts are:

- spec SHA-256:
  `6adb9377d732c0f126191767f1573a62850e08cea04e63e45090cee1c9829167`;
- freeze ID:
  `cc95047bde5eb125795418a9c4253ba1194dda172c217c33cff997e3d295e8e3`;
- freeze-manifest SHA-256:
  `c8ece12dfecf1eefdafb2436ad39220d0c3c5f11e2f7a331fb712e956f0774e3`;
- exclusive execution-claim SHA-256:
  `32e2e12156fb733414d3ee0ab556a6d131f979cd8240d03ce1161eee04ab6e93`;
- summary SHA-256:
  `d952a75dc2785d305bd245c6e79b0e7fc94122c475af3ecadd08d3411962dbea`;
- run-manifest SHA-256:
  `930ad2d4fbcbee63568cd52a9c4209b2270727c6b9b8fd446a2d0044450ef039`.

The unedited outcome was `prospective_synthetic_gate_passed`. Every model seed
had sensitivity 1.0, specificity 1.0, FDR 0, endpoint accuracy 1.0, a localized
path, and equivalence for all six wrong-site dose arms and both matched controls
at all three doses. Selected known-direction cosines were 1.000/0.993/1.000;
mean CLT FVU was 0.017/0.297/0.065 and layer-1 FVU was
0.029/0.082/0.041. All dose effect-recovery ratios were numerically 1.0.
Assessment-set cross-model alignment was 0.935/0.998/0.946. All matched-control
calipers passed without fallback; standardized distances were 2.119--2.123
against the frozen maximum 2.5.

A passing production synthetic result establishes only sensitivity of this
trained synthetic CLT/intervention pipeline at the planted effect size. It is
not evidence of causality in any pretrained protein model, does not establish a
biological primitive and does not by itself close the real held-out
pretrained-model intervention portion of P0-7.

### Held-out pretrained-model adjudication contract

The remaining real-model gate is implemented as a separate, fail-closed
adjudication layer in `src/revision/causal_adjudication.py`, with the canonical
specification template
`configs/p0_7_pretrained_causal_adjudication_spec.example.json` and CLI
`scripts/71_adjudicate_pretrained_causal_interventions.py`. Its schemas are
`r2_p0_7_pretrained_causal_adjudication_spec_v1`,
`r2_p0_7_pretrained_identity_freeze_receipt_v1`,
`r2_p0_7_pretrained_intervention_evaluation_receipt_v1`,
`r2_p0_7_pretrained_causal_adjudication_summary_v1` and
`r2_p0_7_pretrained_causal_adjudication_receipt_v1`.

Adjudication is permitted only after all of the following inputs are frozen and
byte-addressed:

1. the immutable prospective-positive-control production manifest and result,
   whose hashes, pass status and synthetic-only claim scope must validate;
2. a complete P0-2 eligibility receipt authorizing every requested
   model/dictionary-seed/layer cell, with the receipt rehashed and its exact
   run-manifest, checkpoint and source-cohort digest inventories matched; the
   rehashed run manifests transitively bind the validated model revision plus
   config, weight-tree and tokenizer hashes;
3. the exact production P0-5 extraction manifest and P0-6 execution receipt
   that bind the upstream measurements used in the causal evidence;
4. one identity-freeze receipt, created before evaluation, that binds the
   discovery and held-out evaluation cohorts, their protein IDs and exact
   sequence SHA-256 values; the cohorts must have no overlap by either ID or
   sequence hash;
5. target and matched-control feature identities frozen before evaluation,
   including layer, firing frequency, mean activation, decoder norm, direct
   logit effect, received-attention mass and reconstruction contribution; and
6. the exact model, dictionary seed, layer, target/control feature, intervention
   site and strength grid, endpoint definitions, path-localization contrasts,
   BH family, fixed TOST margins, executable source and every input artifact.

The validator accepts no discovery-time rematching, nearest-neighbour fallback,
added or substituted feature, widened caliper, changed endpoint, reduced grid or
unresolved digest. A stale receipt, an ineligible P0-2 cell, a positive-control
development run, a positive-control result outside its declared synthetic-only
scope, or a mismatch between any declared digest and the bytes on disk fails
before analysis.

The evaluation artifact contains one raw row for every frozen evaluation ID by
model, dictionary seed, layer, feature role/identity, site and strength. Each
row retains `evaluation_id`, `sequence_sha256` and `identity_set_id`. The
intervention artifact reports finite intended-feature change, off-target
sparse-code change, reconstruction displacement and downstream logit
displacement; the combined evaluation receipt binds a separate external-score
artifact containing the finite validated sequence endpoint and path-patching or
causal-mediation endpoint on the identical row keys. Missing,
additional or duplicate cells,
non-finite endpoint values, identity substitutions and a row-order or artifact
hash mismatch fail. Cell means, model means and other pre-aggregated rows cannot
replace this raw factorial evidence.

Inference is paired on the frozen evaluation identity. Every target/control and
path-localization contrast uses the same evaluation rows in both arms. The
prespecified positive p-value and both one-sided TOST component p-values for
every measure and factorial cell form one complete Benjamini--Hochberg family.
Equivalence requires both adjusted TOST components to pass. Correction is never
split by endpoint, test type or cell after inspecting results. Matched controls
and frozen wrong-site target arms use paired TOST with the endpoint-specific
margins and alpha stored in the specification. Margins cannot be supplied or
changed at adjudication time. Results remain labelled at the exact
model/dictionary-seed/layer/feature/site/strength cell; averaging over any of
those axes before paired inference is forbidden, and an aggregate result cannot
rescue a failed or inconclusive required cell.

Resolution also requires intervention fidelity rather than a behavior-only
effect. At the frozen on-path site, intended-feature change must be positive;
the lower-is-better off-target-code and reconstruction endpoints must not be
inconclusive; and logit, behavior and path endpoints must resolve under their
prespecified tests. Every off-path behavior and logit contrast must be
equivalent. A localized path with inconclusive intended-feature fidelity, or a
behavior change without the required logit/path and off-target evidence, remains
inconclusive.

A localized positive additionally requires positive intended-feature change,
resolved lower-is-better off-target and reconstruction displacement, positive
on-path logit and validated behaviour effects, positive on-path mediation, and
equivalent off-path behaviour, logit and mediation effects. An
equivalence-bounded negative requires verified intervention fidelity while the
prespecified logit, behaviour and complete path surface are equivalent. An
inconclusive fidelity, off-target, reconstruction, logit, behaviour or path
cell prevents resolution; a favourable average cannot rescue it.

The canonical command is:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ct
python r2_interpretability_transfer/scripts/71_adjudicate_pretrained_causal_interventions.py \
  --spec /path/to/frozen_p0_7_pretrained_adjudication_spec.json \
  --spec-sha256 <exact-sha256> --out-dir <new-result-directory>
```

Publication is staged in the destination filesystem. The implementation
rehashes the specification, module and CLI source, identity freeze,
positive-control and P0-2 receipts, every colocated positive-control/P0-5/P0-6
artifact, every model/dictionary/scorer/calibration/code binding, raw
intervention evidence and all generated artifacts after analysis; only then may
it atomically publish `completion_receipt.json`. An existing destination is
never overwritten. An exception, incomplete factorial surface, failed gate or
final rehash mismatch leaves no completion receipt. The receipt records the
summary and all source/input/artifact hashes so a third party can detect any
later relocation, omission or edit by resolving and rehashing the declared
artifacts.

This contract validates and adjudicates frozen evidence; it does not manufacture
the interventions. At this freeze the prospective synthetic prerequisite has
passed, but no conforming pretrained-model specification, complete raw
evaluation surface or completion receipt has been produced. P0-7 therefore
remains open and failed by default, and historical nulls remain exploratory.

## P0-8: nested repeated recoverability

### Input contract

`scripts/55_run_nested_recoverability.py` consumes a cached NPZ. Every input
requires `y` and `groups`, where `groups` are the frozen sequence-identity or
family clusters used to prevent homology leakage. A confirmatory real input
also requires a complete aligned reconstruction-error and intervention-effect
array for every declared dictionary seed and layer; the runner fails before
analysis if either inventory is absent or incomplete. Synthetic plumbing may
omit both because it is not allowed to satisfy the real quality--recoverability
gate. NPZ arrays use:

```text
rep__<representation_name>__<layer_name>
quality__reconstruction_error__<code_seed_name>__<layer_name>
quality__intervention_effect__<code_seed_name>__<layer_name>
```

Each representation matrix must have shape `[n_samples, n_features]` and
contain only finite values. Multiple sparse-code names (for example
`code_seed_0`, `code_seed_1`, ...) are independent dictionary seeds, not
repeated analysis seeds.

For a real run, an arbitrary NPZ is not eligible.
`scripts/68_build_nested_recoverability_inputs.py` is the production bridge.
It requires the byte-exact P0-2 eligibility receipt and loads only the three
all-seed-authorized exact-cache TopK `best.pt` checkpoints through the verified
`WindowedTranscoder`-to-CLT adapter. The receipt must authorize the requested
model, source-cohort hashes and downstream layers. The builder independently
binds the local model config, weight tree and tokenizer tree; the enlarged
P0-8 cohort and task annotations in their exact row order; train, validation
and test identity assignments; every dictionary seed, run-manifest hash and
checkpoint hash; and a separately frozen intervention-evidence receipt.

Identity assignments alone are not accepted as proof of homology control. Their
descriptor must include a clustering receipt that binds the exact assignment
file and every P0-2/P0-8 source-cohort hash, plus the clustering algorithm name,
version, identity and coverage thresholds, full command, and a resolvable
executable path/SHA-256. The builder rehashes the receipt and executable before
publication.

Targets and identity groups are not passed to representation extraction. The
builder derives CLT-input, MLP-output, per-seed code and per-seed reconstruction
matrices directly from verified artifacts. Reconstruction error is derived
inside the builder separately for each declared seed/layer cell; it is never
averaged over seeds or layers before analysis. Intervention efficacy is
accepted only as the same complete seed/layer inventory from a row-order-bound
external v3 receipt. That receipt must point to a byte-exact v2 freeze manifest
created before test
evaluation. The freeze independently repeats the task/cohort/row-order,
model-artifact, dictionary-checkpoint, seed/layer inventory and
effect-definition bindings and records a resolvable producer source
path/SHA-256, command and environment. Each effect-artifact row contains an
exact `dictionary_seed -> layer -> effect` object; missing, additional or
non-finite cells fail. The builder rehashes the freeze, producer source and
effect artifact before publication; an ungrounded digest string is
insufficient. Exact
sequence overlap, identity-group overlap, test leakage, checkpoint or seed
substitution, non-eligible P0-2 receipts, absent quality vectors and dimensions
below the frozen common dimension all fail. The only published files are one
NPZ, one script-55 runner specification and one immutable input receipt.

Production builder schema v3 additionally fixes model inference to `bfloat16`.
Before the first forward, every floating model parameter must verify as exactly
`bfloat16`. Immediately after every forward and before casting, encoding,
pooling, NumPy conversion or other use, every captured CLT input and MLP output
must pass `torch.isfinite`. The v3 receipt records the declared dtype, observed
parameter dtypes, exact verification methods and successful boolean results;
script 55 rejects old receipt schemas or any missing/false verification.

Production builder schema v3 requires `task.minimum_samples`. The frozen
enlarged decoder-native floor is 480 annotated rows (eight classes times 60
rows per class); production specs declaring a smaller floor, artifacts below
their declared floor, and script-55 real runs with fewer than 480 rows all
fail. The value eight remains available only to explicitly labelled CPU test
fixtures.

### Frozen analysis

1. Create repeated, identity-group-disjoint outer folds from the analysis seed.
2. Within each outer training set, create one shared set of inner group folds.
3. Select a layer independently for every representation using only those
   shared inner folds. Within every inner training fold, apply
   `StandardScaler -> PCA(d)` before the probe, with the prespecified common
   dimension `d`; never fit the scaler or PCA on validation samples.
4. Refit the same scaler--PCA--probe pipeline on the complete outer training
   fold and predict the untouched outer test fold exactly once. Sparse code,
   CLT input, reconstruction and other dense representation arms therefore all
   enter their probes at the same exact dimension.
5. Reuse the identical outer folds for CLT input (ceiling), every dictionary
   code (floor), reconstruction and every control.
6. Derive PCA, random projection, NMF, ICA and random-dictionary controls from
   the selected ceiling layer at that same exact `d`. Every learned transform
   is fit on the outer training fold only. If `d` exceeds any representation
   width or the centered training-fold rank, the run fails rather than silently
   capping it. The primary comparison uses `d=256`. An optional complete rerun
   at TopK active width 128 is retained only as the separately labelled
   `active_width_rank_sensitivity` track; it is never described as a raw
   coordinate-width match. PCA of a sparse code tests matched-rank retained
   information and does not preserve coordinate sparsity.
   ICA uses 16 prespecified deterministic starts, records the accepted
   start and fails the run if none converges; a convergence warning is never
   accepted as a result.
7. Report the full fold manifest, selected layers, inner scores, untouched
   outer predictions and fold metrics.
8. For each analysis seed and dictionary seed, calculate paired group-bootstrap
   intervals for floor-minus-ceiling and every floor-minus-control contrast,
   with ratios where defined. Sampling retains complete identity groups and
   uses the same bootstrap rows for both representations in each contrast.
9. For every dictionary seed and quality layer, calculate separate
   group-bootstrap Spearman relationships for reconstruction error versus probe
   error, intervention efficacy versus probe error, and reconstruction error
   versus intervention efficacy. Seed/layer cells are labelled in every row and
   are not averaged before these relationships are estimated.

The infrastructure accepts global or local tasks without changing the
statistical procedure. Confirmatory inputs must include the enlarged
decoder-native cohort and prespecified contact, motif, active-site or structural
neighbourhood tasks described in the assessment. The current 48-sequence EC
cohort is insufficient and cannot enter the production code path.

### Canonical input example

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ct
python r2_interpretability_transfer/scripts/55_run_nested_recoverability.py \
  --input /path/to/immutable_representations.npz \
  --input-sha256 <sha256> \
  --input-receipt /path/to/input_receipt.json \
  --input-receipt-sha256 <sha256> \
  --ceiling clt_input \
  --floors code_seed_17 code_seed_29 code_seed_43 \
  --comparison-dimension 256 --active-width-dimension 128 \
  --analysis-seeds 101 211 307 401 503 \
  --outer-splits 5 --inner-splits 4 --n-bootstrap 1000
```

`--synthetic` runs only a CPU plumbing check. The result directory contains the
input hash, exact fold assignments, all fold rows, all untouched predictions,
paired intervals, quality relationships and source/artifact hashes. Every fold
row and paired comparison carries `dimension_track`, `track_role` and
`probe_input_dimension`; only `primary_common_dimension` is a confirmatory
estimand, while `active_width_rank_sensitivity` is descriptive sensitivity
analysis.

Script 55 sets `confirmatory_real=true` only when the supplied production
receipt hash, colocated NPZ hash, runner-spec hash and every frozen runner
argument match exactly. A fixture receipt, unreceipted NPZ or parameter change
is rejected before analysis.

## Remaining confirmatory work

Neither P0 package passes merely because these scripts and their synthetic
tests pass. Remaining work is:

- apply the P0-7 protocol to a prospectively frozen real mechanism and to the
  pretrained-model target/control interventions across feature, layer, site,
  strength and model;
- freeze endpoint-specific equivalence bands before looking at those results;
- produce multi-seed, identity-aware P0-8 caches from the mask-aware,
  quality-gated dictionaries;
- add the enlarged decoder-native and local structural tasks; and
- connect the canonical outputs to the manuscript figures and source-data
  provenance map only after their gates are evaluated.
