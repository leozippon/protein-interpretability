# P0-3/P0-4 confirmatory execution contract

**Date:** 2026-07-16  
**Binding assessment:** `docs/npj_ai_manuscript_assessment.md`  
**Revision matrix:** `docs/NPJ_AI_MAJOR_REVISION_PLAN_20260716.md`

This note defines the executable contract for P0-3 (independent atlas
replication) and P0-4 (continuous conditional semantics). It does not record a
scientific result. As of this note, only synthetic CPU tests have run; neither
gate has passed.

## Canonical production input build

`src/revision/input_builder.py` and the thin CLI
`scripts/64_build_revision_inputs.py` close the previously missing bridge from
P0-2-eligible exact-cache TopK dictionaries to the inputs consumed by scripts
52 and 53. Start from
`configs/npj_revision_input_builder_spec.example.json`, replace every
placeholder, hash the completed spec, and do not edit it after extraction
begins.

The builder fails closed unless it can verify:

1. exactly one discovery and one held-out cohort in the frozen six-field
   JSONL schema, with distinct cohort IDs and no sequence-hash overlap;
2. three deployed model roots through separate config, weight-tree and
   tokenizer/support-tree hashes;
3. one SHA-pinned P0-2 eligibility receipt plus the exact seed-17, seed-29 and
   seed-43 `best.pt`, run-manifest and checkpoint hashes for every model;
4. the receipt's exact train/validation/test source-manifest hashes and
   requested-layer allowlist through `load_eligible_topk_clt`;
5. exact decoded token-to-residue coverage for every sequence, excluding
   special/prompt tokens; and
6. a SHA-checked residue annotation JSONL that covers every held-out residue
   exactly once for every prespecified binary label.

Before the first forward pass, the builder requires declared `bfloat16`
inference and verifies that every floating model parameter is exactly
`bfloat16`. Immediately after every forward, and before any layer selection,
cast, encoding, NumPy conversion or downstream use, it rejects a non-finite
captured CLT input or MLP output. The declared dtype, observed parameter
dtypes, verification methods and successful boolean results are carried in the
v3 atlas, semantic and build provenance. A failed check publishes no output.

The production schema is version 3 and requires `confirmatory: true`, a frozen
non-negative `model_seed`, `p0_2_eligibility_receipt`, and exactly three
dictionary descriptors per model. Each descriptor names one seed in
`{17,29,43}`, its exact-cache `best.pt` SHA-256 and its P0-2 run-manifest
SHA-256. Online-training directories and `clt.pt` are not accepted in this
path. Tiny CPU fixtures use the separately named
`nonconfirmatory_fixture_checkpoint` field under top-level
`confirmatory: false`; that path cannot emit a confirmatory atlas or semantic
artifact and cannot declare a P0-2 receipt.

The P0-2 receipt authorizes the dictionary/cache side of this contract: model
name and method, all dictionary seed/run/checkpoint hashes, immutable source
splits, geometry and the eligible-layer allowlist. It does not contain a
pretrained-model seed or deployed config/weight/tokenizer-tree digests. Those
remain a separate consumer-side immutable contract: the builder verifies and
records them, and script 52 requires exact discovery/held-out agreement,
including the v3 bfloat16 and finiteness receipts. No
missing P0-2 model-seed field is inferred or synthesized.

The residue annotation schema is one strict JSON object per line:

```json
{"sequence_sha256":"<lowercase SHA-256>","position":0,"labels":{"frozen_binary_label":1}}
```

Atlas outputs are continuous ReLU--TopK activation matrices with one row per
sequence and one column per dictionary feature. A separate matrix is emitted
for every model, eligible dictionary seed and requested layer. Each entry is
the arithmetic mean over residue positions after strict token alignment, so
padding, BOS/EOS, and model prompt tokens are excluded. The emitted
discovery/held-out manifests are directly consumable by
`scripts/52_run_revision_atlas.py`.

For each semantic analysis, the spec freezes one model, eligible `run_seed`,
layer and feature list. The builder emits the corresponding continuous
held-out residue activations;
randomized-PCA dense directions fit only on a prespecified hash-selected sample
of discovery residues; an independently seeded unit-norm Gaussian dictionary;
input norms and low-level covariates; biological labels; and seeded
within-protein prevalence-matched negative labels. It writes both the NPZ
bundle and a SHA-bound `conditional_semantics_spec.json` directly consumable by
script 53. Held-out dense inputs are projected during extraction rather than
retained as a full hidden-state matrix. The generated v3 conditional-semantics
spec repeats the six dtype/finiteness fields, and script 53 rejects old
confirmatory schemas, non-bfloat16 declarations, mismatched observed dtypes or
any missing/false verification result. Legacy specs without this provenance
remain accepted only when explicitly `confirmatory: false`.

The production example is deliberately incomplete but confirmatory: every
receipt, checkpoint, run and cohort placeholder must be replaced before use.
A confirmatory semantic build must also supply a SHA-bound
`prospective_power_plan`. That independent-pilot JSON must satisfy script 53's
schema-version-1 contract and exactly cover every emitted representation,
feature, biological/negative label and protein/family blocking hypothesis with
a positive prespecified standard error for delta MSE. The builder validates the
coverage, copies the byte-identical plan into the output and passes its hash to
script 53. It never derives a prospective bound from the confirmatory cohort's
observed bootstrap standard errors.

Example input-build command:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ct
python r2_interpretability_transfer/scripts/64_build_revision_inputs.py \
  --spec r2_interpretability_transfer/configs/npj_revision_input_builder_spec.json \
  --out-dir r2_interpretability_transfer/results/npj_revision/p0_3_p0_4_inputs
```

The output directory is staged and published atomically, is never overwritten,
and includes before/after accelerator and host-memory records plus hashes for
all files. The current verification used only tiny CPU fixtures. No real model
extraction or P0-3/P0-4 scientific adjudication has yet occurred.

## P0-3: independent atlas replication

The reusable implementation is `src/revision/atlas.py`; the immutable runner
is `scripts/52_run_revision_atlas.py`. A production run must start from cached
continuous activation matrices and two cohort manifests with non-overlapping
sequence hashes. Discovery uses cohort A to choose variance-ranked feature
pools, layers, and matches. Cohort B only scores those frozen identities.

The confirmatory runner enforces:

1. externally SHA-pinned discovery and held-out builder manifests with
   confirmatory P0-2-eligible status;
2. exactly three named model matrices for the analysis's declared eligible
   seed, with identical row order inside each cohort;
3. exact agreement of discovery/held-out model seed, receipt, run-manifest,
   checkpoint, model-artifact and source-manifest provenance;
4. independent revalidation of the actual P0-2 receipt, selected `best.pt` and
   requested-layer allowlist before analysis;
5. exactly one analysis for each seed 17, 29 and 43, with distinct checkpoint
   and run-manifest hashes rather than arbitrary matrices carrying seed labels;
6. disjoint discovery/test sequence hashes and SHA-256-checked `.npy` inputs;
7. positive-only and absolute-correlation analyses, with signed correlations
   retained in every output row;
8. greedy, Hungarian, entropic optimal-transport, and joint three-way candidate
   matching;
9. at least two feature-pool sizes, thresholds, and layer maps;
10. one independently drawn row permutation per model and replicate, reused for
   all layers and model pairs;
11. at least 1,000 null replicates and plus-one empirical p-values; and
12. discovery count, held-out retention, exact-identity Jaccard, signed
   held-out correlations, ambiguity, confidence, null distributions, and
   descriptive variance attribution.

The grid may set `max_feature_pool_by_matcher`: this retains the historical
4,096-feature sensitivity run for greedy matching while bounding cubic or
transport-based matcher comparisons to a prespecified common pool. Skipped
cells are structural exclusions declared in the frozen spec, not runtime
fallbacks.

The joint matcher optimizes a three-edge triangle score over a prespecified
candidate graph and then chooses disjoint triangles greedily. It is a joint
three-way sensitivity method, not a claim of globally optimal hypergraph
matching. The optimal-transport matcher retains a soft coupling weight and
uses a discrete maximum-mass assignment for identity comparisons.

`null.selection` restricts the expensive 1,000-replicate null to a
prespecified primary matcher/mode and canonical layer/pool/threshold setting.
All matchers, both correlation modes, and every requested threshold remain in
the observed/held-out stability sweep. This keeps the primary null coherent
without multiplying it across every sensitivity-grid cell; any additional
null sensitivity settings must be declared before execution.

Variance attribution uses drop-one fixed-effect partial R-squared over cohort,
model seed, dictionary seed, matcher, correlation mode, layer map, pool size,
and threshold. A factor is explicitly marked non-estimable when only one level
is present or the design is rank-confounded. This is descriptive variance
attribution, not a random-effects population estimate.

Example command:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ct
python r2_interpretability_transfer/scripts/52_run_revision_atlas.py \
  --spec r2_interpretability_transfer/configs/npj_revision_atlas_spec.json \
  --out-dir r2_interpretability_transfer/results/npj_revision/atlas
```

Start from `configs/npj_revision_atlas_spec.example.json`, replace every input
path/hash, add one analysis per dictionary/model seed, and freeze the spec
before examining confirmatory outputs. The runner refuses to overwrite a
non-empty result directory.

## P0-4: continuous conditional semantics

The reusable implementation is `src/revision/semantics.py`; the immutable
runner is `scripts/53_run_conditional_semantics.py`. Its feature-level estimand
is the out-of-fold reduction in mean squared activation error after adding a
prespecified biological label to the low-level covariate model. It also reports
the corresponding incremental held-out R-squared:

\[
\Delta R^2 =
\frac{\mathrm{MSE}_{\mathrm{covariates}}-
      \mathrm{MSE}_{\mathrm{covariates+label}}}
     {\operatorname{Var}(a)}.
\]

The low-level design is fixed by the run spec: polynomial normalized-position
terms, stable SHA-256-hashed k-mer buckets, input norm, log protein length, and
sequence-source indicators. Ridge preprocessing is fit inside each training
fold. Each confirmatory biological hypothesis is a frozen binary indicator and
adds one fixed label column.

The confirmatory runner enforces:

1. continuous, finite activation matrices rather than selected top events;
2. identical protein-blocked and family-blocked folds for every representation
   and label;
3. dimension-matched sparse, dense, and randomized-dictionary representations;
4. one binary array per prespecified biological hypothesis, plus a negative
   label matched to its exact category-count prevalence (multi-class
   annotations must be frozen as separate binary hypotheses before testing);
5. within-protein conditional label randomization with at least 1,000
   replicates and plus-one p-values;
6. at least 1,000 protein-cluster bootstrap replicates for 95% intervals;
7. one global Benjamini--Hochberg correction across all representation,
   feature/layer, label, and blocking tests; and
8. a conservative prospective minimum-detectable mean-squared-error reduction
   derived from standard errors frozen from an independent pilot or
   training-only cohort, the full planned multiplicity and prespecified power.

The confirmatory runner requires a SHA-bound `prospective_power_plan` with a
positive independent-source standard error for every representation, feature,
label and blocking hypothesis. It refuses a confirmatory run if that plan is
missing or incomplete. The standard error estimated by bootstrapping the
analyzed proteins is also reported, but its derived detectability value is
explicitly retrospective and cannot be cited as prospective power. The
companion schema template is
`configs/npj_conditional_semantics_power_plan.example.json`.

Every representation and label records its frozen construction method;
randomized dictionaries and negative labels also require explicit seeds.

When a label is constant within every protein, the conditional randomization
is non-informative. Such rows are retained with `permutation_degenerate=true`
and cannot support a confirmatory residual-semantic conclusion.

Example command:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ct
python r2_interpretability_transfer/scripts/53_run_conditional_semantics.py \
  --spec r2_interpretability_transfer/configs/npj_conditional_semantics_spec.json \
  --out-dir r2_interpretability_transfer/results/npj_revision/conditional_semantics
```

Start from `configs/npj_conditional_semantics_spec.example.json` and
`configs/npj_conditional_semantics_power_plan.example.json`. The NPZ bundle,
independent power plan and run spec must be immutable and SHA-256 recorded
before testing.

## Output and claim boundary

Both runners write strict JSON, explicit TSVs, exact input/output hashes,
commands, parameters, and environment versions. Atlas outputs can quantify
cohort/procedure stability; they cannot establish that a match represents the
same semantic concept. Conditional-semantic outputs can quantify incremental
held-out association under the prespecified model; they cannot establish a
biological primitive or causal mechanism. P0-3 and P0-4 remain failed until
real independent cohorts, all planned seeds/controls, and prespecified
scientific acceptance thresholds have been executed and reviewed.
