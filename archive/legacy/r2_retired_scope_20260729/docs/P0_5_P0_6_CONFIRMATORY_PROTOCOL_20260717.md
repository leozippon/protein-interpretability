# P0-5/P0-6 confirmatory protocol

Date: 2026-07-17  
Scope: infrastructure freeze for the `npj Artificial Intelligence` revision

This protocol implements the preparation and analysis contracts requested in
Sections 3.6, 3.8 and 3.9 of `npj_ai_manuscript_assessment.md`. The checked-in
outputs are CPU-only synthetic validation fixtures. They do not pass either
scientific gate, do not revise a historical result and must not be cited as
pretrained-model evidence.

## P0-5: N-terminal counterfactuals

### Frozen estimands and factorial

The evaluation unit is a held-out protein with a natural canonical `MXX`
start. `build_counterfactual_variants` constructs four paired conditions from
that protein:

1. the unmodified natural `MXX` sequence;
2. `M->A`, retaining every other residue;
3. an internal insertion of the same three-residue `MXX` motif; and
4. an artificial truncation beginning at that inserted internal copy.

Each condition is measured with the native BOS token and with BOS removed.
Every target and control feature must have the complete protein x condition x
BOS factorial for both roles of every frozen target/control protein pair.
Proteins are compared exactly within protein; target features are matched only
to same-model, same-layer controls, using firing frequency and CLT-input norm.
The evaluator requires one reciprocal target/control membership per
`protein_pair_id` and fails an incomplete cell instead of imputing it.

For attention tensors indexed as `[..., query, key]`, a valid key `k` is
eligible for valid causal queries `q >= k`. The reported quantity is

```
sum_q attention[..., q, k] / number_of_eligible_valid_queries(k)
```

Padding and invalid queries/keys are excluded. Raw received mass and the
denominator are retained. Sparse-feature activation is explicitly the value
measured before any attention intervention.

The conservative attention-path perturbation starts from that same baseline
forward pass. A second forward pass masks the focal token as an attention key
for every layer. Returned attentions must confirm absolute focal-key mass at or
below `1e-6` for every strict-suffix query (`q > focal`) or extraction aborts.
The estimands are the resulting increase in strict-suffix observed-token NLL
and change in mean observed-token logit. Predictions made by the focal query,
including the immediate next token, are excluded. This is a focal-key path
perturbation, not formal feature mediation and not by itself evidence for an
attention-sink mechanism.

The primary paired contrasts are `M->A - natural`, internal motif minus natural
N-terminus, artificial start minus the identical internal motif, and native
BOS minus removed BOS. Protein bootstrap intervals use proteins, not tokens,
as the sampling unit. For each matched target, the confirmatory estimand is the
target contrast minus the mean of its matched-control contrasts on the exact
same proteins. Two-sided sign randomization, protein bootstrap intervals and
TOST are reported for every activation and normalized-attention contrast. The
same within-condition contrast is also differenced between each target protein
and its matched control protein. The focal-key NLL and observed-logit effects
are analyzed only with this target-protein-minus-matched-control-protein
difference-in-differences. Holm correction is applied jointly across all
feature-control cells and all protein-pair difference-in-differences cells,
for both separation and equivalence tests.
A within-protein conditional model reports the
association between opportunity-normalized received attention and feature
activation after normalized position, sequence length, motif/start state and
BOS are included.
That coefficient is a conditional association, not a causal mediation
estimate and not evidence for an “attention sink.”
Separately, a pair-conditioned model reports each focal-key effect conditional
on normalized focal position and log sequence length (with motif/start, BOS and
target/control role covariates). It carries the same non-mediation boundary.

Every measurement row is joined fail-closed to the frozen variant by
`protein_id`, `condition` and exact sequence SHA-256. It must preserve raw
ordered `token_ids`, their canonical-JSON SHA-256, tokenizer revision and the
focal token index. The eligible-query denominator must equal the valid token
count minus that focal index. All feature rows for a protein/condition/BOS/model
must have identical tokenization, and every target and control must cover the
identical frozen protein set. The evaluation cohort is rejected if either a
protein identifier or exact sequence overlaps the supplied discovery cohort.

### Interfaces and artifacts

The real pretrained-model measurement stage is implemented by
`scripts/65_extract_n_terminal_measurements.py`. It takes a single frozen
specification (an annotated template is
`configs/p0_5_n_terminal_extractor_spec.example.json`) and requires the exact
SHA-256 of that specification on the command line. The extractor is
fail-closed and performs no feature discovery:

- target, full protein-control-pool and discovery manifests must be mutually
  disjoint by both protein ID and exact sequence SHA-256;
- every target and candidate protein control must have a natural `MXX` start
  and an explicit source `focal_position` equal to the frozen internal
  insertion site derived at `internal_fraction=0.55`; a merely in-range or
  N-terminal position is rejected;
- one distinct control protein is assigned to each target by a global
  minimum-cost assignment, subject to frozen absolute length and normalized
  focal-position calipers; an incomplete assignment aborts;
- the discovery-only feature-profile table has exact fields `model`, `layer`,
  `feature`, `feature_role`, `firing_frequency` and `input_norm`, and its
  descriptor binds the discovery-cohort SHA-256;
- control features are same-model/same-layer candidates inside both frozen
  absolute log firing-ratio and log input-norm-ratio calipers; insufficient
  controls abort rather than widening a caliper;
- the deployed model tree and tokenizer/support tree are verified against
  frozen hashes before inference;
- inference is frozen to `bfloat16`, and every floating model parameter must be
  observed as exactly `bfloat16` before the first activation is produced;
- the dictionary is loaded only through the public P0-2
  `load_eligible_topk_clt` gate using a complete hash-verified eligibility
  receipt, exact all-seed run-manifest/checkpoint hashes, exact train/
  validation/test source-manifest hashes and a receipt-authorized layer; the
  selected checkpoint must be the exact-cache `best.pt`, while an online
  `clt.pt` or failed receipt is rejected; and
- BOS handling is frozen explicitly as either the tokenizer's verified native
  leading BOS or an explicit prepend of the tokenizer BOS to a no-special-token
  encoding. The two token lists must differ only by that leading BOS.

Every required-layer CLT input and attention tensor, plus every baseline or
focal-key-intervention logit tensor, is checked for finiteness before conversion
or downstream use. The v4 summary and extraction receipt record the declared
dtype, observed parameter dtypes, exact verification methods and successful
dtype/finiteness results. Any mismatch or non-finite tensor aborts without a
published directory.

The command-line extractor always runs in `production` mode. Production mode
forbids model or dictionary loader injection; both overrides are available only
through the Python API's explicit `test_fixture` mode, whose manifest status is
`verified_test_fixture_complete` rather than a production status. Extraction
occurs in a private staging directory and publishes the complete directory by
one atomic rename. A failure or interrupt removes the staging tree and leaves no
apparently complete destination.

Each variant/BOS input first receives one unintervened evaluation-mode pass.
The selected CLT input, attention tensor and baseline logits are captured from
that pass. The extractor applies the receipt-authorized exact-cache ReLU--TopK
encoder, measures the focal feature, averages attention over heads at the
feature layer, sums the focal key's eligible causal queries and preserves the
raw sum and denominator. It then performs the all-layer focal-key mask pass
described above. It rejects malformed attention/logit shapes, non-finite
values, causal-mask violations and any focal-key leakage. Ordered token IDs,
their canonical-JSON hash, formatted-input hash, actual focal CLT-input norm,
reciprocal pair IDs/roles, calipers, intervention verification and complete
model/P0-2 dictionary provenance are retained in the artifacts.

Example extraction (the output directory must be absent):

```bash
python scripts/65_extract_n_terminal_measurements.py \
  --spec configs/p0_5_n_terminal_extractor_protgpt2_frozen.json \
  --spec-sha256 <exact-lowercase-sha256> \
  --out-dir results/npj_revision_20260716/p05_protgpt2_extraction
```

The resulting `natural_cohort.jsonl` and `measurements.jsonl` feed the stable
analyzer without schema conversion. Real analysis requires external SHA-256
pins for the extractor receipt and every input. It rehashes the complete
extractor artifact inventory, requires a production-mode v4 receipt with
verified numerical-integrity fields, reads the exact target/control membership
from the receipt-bound `feature_matches.json`,
and derives `control_count` from that frozen contract. It never reranks controls
or selects a smaller subset:

```bash
python scripts/59_run_n_terminal_counterfactuals.py \
  --sequences results/npj_revision_20260716/p05_protgpt2_extraction/natural_cohort.jsonl \
  --sequences-sha256 <exact-sha256> \
  --measurements results/npj_revision_20260716/p05_protgpt2_extraction/measurements.jsonl \
  --measurements-sha256 <exact-sha256> \
  --extractor-receipt results/npj_revision_20260716/p05_protgpt2_extraction/run_manifest.json \
  --extractor-receipt-sha256 <exact-sha256> \
  --discovery-cohort discovery_proteins.jsonl \
  --discovery-cohort-sha256 <exact-sha256> \
  --equivalence-spec frozen_n_terminal_equivalence.json \
  --equivalence-spec-sha256 <exact-sha256> \
  --out-dir results/npj_revision_20260716/p05_protgpt2_analysis
```

The destination must not exist. Script 59 writes to a private sibling staging
directory and atomically renames it only after the analysis, manifest and final
input rehash all succeed; failures remove the staging directory.

This code path has CPU/mock contract coverage only at this revision. No
pretrained-model measurement or P0-5 scientific gate is represented by those
tests.

`scripts/59_run_n_terminal_counterfactuals.py` accepts a JSONL cohort with
`protein_id` and `sequence`, prepares all four variants, and optionally analyzes
a complete measurement table. In addition to the exact join fields above,
measurement rows contain BOS policy, model/layer/feature identity and role,
pre-intervention activation, raw received attention, eligible-query count,
normalized focal position, sequence length, firing frequency, input norm,
reciprocal `protein_pair_id`/role metadata, baseline and key-masked strict-
suffix NLL, baseline and key-masked observed-token logit means, their effects,
and the maximum verified focal-key attention leakage.

Preparation-only example:

```bash
python scripts/59_run_n_terminal_counterfactuals.py \
  --sequences heldout_mxx.jsonl \
  --sequences-sha256 <exact-sha256> \
  --out-dir results/npj_revision_20260716/n_terminal_counterfactuals_real
```

Analysis adds the extractor receipt and all externally pinned immutable inputs:

```bash
python scripts/59_run_n_terminal_counterfactuals.py \
  --sequences extraction/natural_cohort.jsonl \
  --sequences-sha256 <exact-sha256> \
  --measurements extraction/measurements.jsonl \
  --measurements-sha256 <exact-sha256> \
  --extractor-receipt extraction/run_manifest.json \
  --extractor-receipt-sha256 <exact-sha256> \
  --discovery-cohort discovery_proteins.jsonl \
  --discovery-cohort-sha256 <exact-sha256> \
  --equivalence-spec frozen_n_terminal_equivalence.json \
  --equivalence-spec-sha256 <exact-sha256> \
  --out-dir results/npj_revision_20260716/n_terminal_counterfactuals_real
```

The equivalence JSON freezes `alpha`, multiplicity
`holm_all_feature_control_and_protein_pair_did_cells`, and positive,
scientifically justified margins for `feature_activation_pre`,
`normalized_received_attention`, `suffix_nll_increase_key_masked`, and
`suffix_observed_token_logit_change_key_masked`. Those margins must be frozen
before outcome inspection; the synthetic fixture's numerical margins are test
values, not production thresholds. Every measurement row carries its immutable
lowercase `feature_match_id`; inconsistent IDs, a changed control count or
membership differing from `feature_matches.json` fail instead of triggering a
new nearest-control calculation. The output directory contains the natural
cohort, exact counterfactual sequences and hashes, normalized measurement rows,
a summary, and a manifest hashing every input, source file and artifact.

The CPU contract check is:

```bash
python scripts/59_run_n_terminal_counterfactuals.py --synthetic
```

Its canonical status string is `synthetic_pipeline_validation_only`.

## P0-6: corrected eight-class steering

### Selection and generation freeze

Feature attribution must be computed on a selection cohort whose identifier
and SHA-256 hash differ from the evaluation cohort. Every requested
class/layer/site cell must contain the requested number of strictly positive
direct effects. A deficient cell fails with no opposite-sign fallback. The
full candidate table and every selection decision are preserved.

The generation plan covers all eight prespecified EC prompts, prompt-only,
target, random-feature and decoder-norm-matched-feature arms, every frozen
site and positive dose, and the same token seeds across paired arms. Random and
norm-matched controls exclude every selected target at the same layer/site,
including targets selected for another EC class. Decoder-norm matches are
one-to-one and must pass a frozen absolute log-norm-ratio caliper; the freeze
records every pair and aggregate balance diagnostics.

Before generation, the preparation command validates and writes immutable
copies of:

- endpoint direction, primary/validated flags and a positive equivalence
  margin;
- model and tokenizer revisions;
- CLT checkpoint SHA-256;
- distinct selection/evaluation cohort SHA-256 values; and
- the code revision.

At least one primary endpoint must be independently validated. Primary and
validated fields must be JSON booleans; a primary heuristic endpoint is
rejected. Each validated endpoint freezes `experimental_unit` (`generation` or
`generation_set`), scorer name/version, the scorer artifact's path and SHA-256,
and the independent calibration cohort's path and SHA-256. Real freezes resolve
and rehash both files; digest strings without the corresponding artifacts are
ineligible. Heuristic scores may be retained as
non-claim-eligible supporting endpoints. The generation plan, endpoint
specification, provenance, analysis/multiplicity rule, feature decisions and
control decisions are separately hashed.

Each plan ID hashes the complete row (prompt, arm, interventions, dose, paired
RNG stream, sampler, executed protocol source hashes and upstream content
binding), rather than only its display coordinates. Generation-set IDs bind the exact member plan IDs. A freeze ID
then binds the complete frozen artifact set. Both stages refuse a non-empty
output directory, and analysis is prohibited from writing into the freeze.

Real preparation requires the scientific inputs plus both exact cohort
manifests:

```bash
python scripts/60_prepare_corrected_steering.py --stage freeze \
  --attributions selection_attributions.jsonl \
  --feature-pool eligible_feature_pool.jsonl \
  --endpoint-specs frozen_endpoint_specs.json \
  --provenance frozen_provenance.json \
  --selection-cohort selection_proteins.jsonl \
  --evaluation-cohort evaluation_proteins.jsonl \
  --selection-split-id selection_v1 \
  --evaluation-split-id evaluation_v1 \
  --norm-log-caliper 0.10 \
  --generation-set-size 4 \
  --out-dir results/npj_revision_20260716/corrected_steering_freeze
```

The cohort manifests must be disjoint by both protein identifier and exact
sequence, and their byte hashes must equal the corresponding provenance fields.
The provenance object requires `model_revision`, `tokenizer_revision`,
`clt_checkpoint_sha256`, `selection_cohort_sha256`,
`evaluation_cohort_sha256`, and `code_revision`. The three SHA-256 values are
validated syntactically and the cohort hashes must differ.

The executor must emit exactly one full canonical amino-acid sequence for each
`plan_id`, plus raw token IDs and their hash, stop reason and runtime metadata.
Runtime metadata binds generator revision, host/device, start time, elapsed
seconds, evaluation mode, actual hook site, multiplier semantics and paired RNG
stream. The validator rejects a non-`eval` run or a hook/RNG mismatch. Completed
records retain the prompt, arm, site, dose, intervention features, seed, sampler
settings and all raw output provenance. No lead-only or post-selected subset is
an eligible analysis input.

### Real generation executor

`scripts/63_execute_corrected_steering.py` is the fail-closed bridge from the
immutable script-60 freeze to real decoder generation. Its separately hashed
execution specification declares deployment facts (local model root,
model/tokenizer revisions, config/weight/tokenizer tree hashes, dtype and
device) and one exact P0-2 eligibility binding. The latter consists of the
externally pinned eligibility-receipt path/hash, `topk_clt`, one selected seed,
the selected exact-cache `best.pt`, all three seed-17/29/43 run-manifest and
checkpoint hashes, the train/validation/test source-manifest hashes and the
requested layer set. An annotated template is
`configs/p0_6_steering_executor_spec.example.json`.

The executor requires the externally recorded SHA-256 values of both the
script-60 freeze manifest and execution specification. It then verifies every
frozen artifact, freeze ID, current frozen-protocol source hash, deployed model
tree and current P0-2 receipt before loading. The public
`require_eligible_model_method` gate must authorize the exact model, method,
all-seed artifact map, source manifests and requested layers, after which only
`load_eligible_topk_clt` may load the selected `best.pt`. An online/preliminary
`clt.pt`, a selected-checkpoint mismatch, all-seed hash drift, an ineligible
model-method pair or a layer outside the receipt allowlist aborts. Plan rows
must match the verified exact-cache CLT geometry and decoder norms. It supports exactly the
frozen `additive_decoder_direction_v1` semantics: for each declared feature it
adds `dose * W_dec[layer][feature, 0]` at every active token at either the
architecture-specific CLT-input hook or same-layer MLP-output hook. Any other
site or multiplier semantics aborts. Every live hook verifies a nonzero
realized tensor displacement against the intended vector and records calls,
tokens, vector hashes, norms and displacement error.

The execution specification freezes model inference to `bfloat16`. Before the
first generation forward pass, every floating model parameter must be observed
as exactly `bfloat16`. Every required intervention-site activation is checked
before conversion or modification, every changed activation is checked before
use, and a model-level forward hook checks every generation-step causal-LM logit
tensor before sampling consumes it. The aggregate finiteness result becomes
true only after all generation rows and hook receipts complete.

For each class/seed pair the prompt and sampler must be identical across arms.
The executor restores and reseeds Python, NumPy and Torch streams before every
row, fixes sampling to `do_sample=True` and `top_k=0`, and applies the frozen
temperature, top-p and maximum-new-token settings. It preserves prompt token
IDs and generated-continuation token IDs separately, including their hashes
and exact scope. The final `generation_outputs.jsonl` is accepted directly by
script 60's analysis stage.

The output directory must not exist. Generation occurs in a private staging
directory; only a complete set that passes script 60's generation validator is
published by atomic rename. Errors and interrupts remove that staging tree,
while a stale staging tree from an externally killed process causes a hard
failure for explicit audit rather than an implicit resume.

```bash
python scripts/63_execute_corrected_steering.py \
  --frozen-dir results/npj_revision_20260716/corrected_steering_freeze \
  --freeze-manifest-sha256 <exact-run-manifest-sha256> \
  --execution-spec configs/p0_6_steering_executor_zymctrl_frozen.json \
  --execution-spec-sha256 <exact-execution-spec-sha256> \
  --out-dir results/npj_revision_20260716/corrected_steering_generations
```

The published v3 receipt hashes the complete generation JSONL and v2 summary, the
freeze, execution specification, deployed model artifacts, P0-2 receipt,
all-seed run/checkpoint/source bindings, selected `best.pt` and all executed
source files. Both artifacts record the declared dtype, observed parameter
dtypes, exact verification methods and successful dtype/finiteness results.
Downstream verification re-runs the P0-2 eligibility gate and
rehashes the current selected checkpoint; a later artifact or source drift is
therefore fatal. Its status certifies generation completeness only. It is not a
steering result and does not pass P0-6 before the frozen validated endpoints
are scored and the separate analysis resolves all eight classes.

Endpoint scoring is a separate, immutable execution boundary. The scoring
system must publish a JSON receipt matching
`configs/p0_6_score_receipt.example.json`; script 60 never manufactures this
receipt for a real run. The receipt binds the freeze ID, byte-exact
`generation_outputs.jsonl`, byte-exact v3 generation-execution receipt,
byte-exact score JSONL and byte-exact frozen endpoint-specification artifact.
Script 60 independently verifies that execution receipt against the same
freeze, colocated generation output, current source files and current P0-2
receipt/`best.pt` hashes before accepting scores. The score receipt contains
exactly one `complete` execution for every frozen endpoint. For each endpoint,
expected and scored coverage are the count and canonical SHA-256 of the exact
sorted `plan_id` set or `generation_set_id` set, according to the frozen
experimental unit.

An independently validated execution must reproduce the frozen validated and
primary flags, scorer name, scorer version, scorer artifact path/SHA-256,
calibration-cohort path/SHA-256 and experimental unit exactly. Script 60 rehashes
both real artifacts again when validating the score receipt, so post-freeze
artifact drift is fatal. A heuristic execution
must remain `heuristic_supporting_only`, `validated: false` and
`primary: false`; it cannot become claim-eligible through the receipt. Any
partial, duplicate, unknown or extra endpoint execution, coverage mismatch,
artifact mismatch or non-complete status aborts before an analysis directory is
created. The receipt file itself is independently pinned on the command line,
so editing the receipt and updating only its internal fields is insufficient.

Analysis is a separate invocation:

```bash
python scripts/60_prepare_corrected_steering.py --stage analyze \
  --frozen-dir results/npj_revision_20260716/corrected_steering_freeze \
  --generation-outputs results/npj_revision_20260716/corrected_steering_generations/generation_outputs.jsonl \
  --generation-execution-receipt results/npj_revision_20260716/corrected_steering_generations/execution_receipt.json \
  --generation-execution-receipt-sha256 <exact-execution-receipt-file-sha256> \
  --scores frozen_endpoint_scores.jsonl \
  --score-receipt frozen_endpoint_score_receipt.json \
  --score-receipt-sha256 <exact-score-receipt-file-sha256> \
  --out-dir results/npj_revision_20260716/corrected_steering_analysis
```

### Inference and decision rule

Every generation-level endpoint must cover every planned generation exactly
once. A generation-set endpoint must instead cover each frozen replicated set
exactly once and include the exact member plan IDs; this keeps diversity and
other set properties at their correct experimental unit. Non-finite, duplicate,
unknown or incomplete scores fail. Each intervention arm is paired with
prompt-only by class and seed or replicated-set index. The analysis reports a paired
mean, protein/generation-pair bootstrap interval, a one-sided paired
sign-randomization test, and a two-one-sided equivalence test against the
frozen smallest effect size of interest. Holm correction covers all classes,
arms, sites, doses, endpoints and specificity contrasts, including TOST.
Only validated primary endpoints are claim-eligible. A class is positive only
when the same target cell is positive versus prompt-only, random features and
norm-matched features after correction. It is an equivalence-bounded negative
only when every primary target-vs-prompt cell over the frozen grid is
equivalent; otherwise it is inconclusive.
All eight classes must resolve before the P0-6 acceptance gate can pass.
The real status `completed_confirmatory_analysis` is emitted only after the
separately pinned score receipt passes all bindings and coverage checks; its
validated summary is retained in `analysis.json`, a semantic copy is published
as `validated_score_receipt.json`, and the original input file hash is retained
in the analysis manifest.

The CPU-only end-to-end contract check is:

```bash
python scripts/60_prepare_corrected_steering.py --synthetic
```

It freezes and analyzes separate subdirectories, includes both per-generation
and replicated-set endpoints, and deliberately plants two specific positive
classes plus equivalence-bounded nulls for the others. Its
`all_eight_resolved` value tests code flow only. This path constructs an
explicit in-memory receipt with status `synthetic_fixture_complete`, and the
analysis remains `synthetic_pipeline_validation_only`; neither status is valid
for a real confirmatory analysis.

## Required work before scientific use

The checked-in code now enforces the previously listed join, cohort, matching,
specificity, freeze/analyze, raw-provenance and experimental-unit contracts.
The following empirical work remains mandatory and is intentionally not
represented by the synthetic outputs:

- preregister domain-meaningful P0-5/P0-6 equivalence margins before inspecting
  pretrained-model outcomes;
- run the P0-5 model-specific measurement extractor on disjoint held-out
  proteins and supply the exact tokenizations accepted by this analyzer;
- first obtain a complete P0-2 eligibility receipt for the intended
  model/`topk_clt` pair and its exact seed-17/29/43 caches; only then instantiate
  and hash-freeze the P0-6 deployment specification, execute the complete frozen
  generation plan, and audit the emitted hook-displacement receipts and
  paired-stream metadata;
- supply real independently calibrated endpoint artifacts and generation-wide
  or replicated-set scores with frozen, resolvable scorer/calibration paths and
  hashes, then
  externally hash-pin a complete scorer-execution receipt; and
- regenerate manuscript figures/source data only after those H200 outputs pass
  the gates.

Until then the historical steering run remains a defective pilot, the
N-terminal result remains descriptive, and neither “attention sink” nor
successful/cleanly null steering is supported.
