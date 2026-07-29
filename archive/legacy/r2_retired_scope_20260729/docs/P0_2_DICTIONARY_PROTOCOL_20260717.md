# P0-2 confirmatory dictionary protocol

**Frozen:** 2026-07-17, before new mask-aware training  
**Binding assessment:** `npj_ai_manuscript_assessment.md`  
**Status at freeze:** infrastructure validated; scientific gate not yet passed

## Cohorts

Every run consumes an immutable JSONL file with exactly
`{id, source, sequence, split, family, sha256}`. Exact duplicate sequences are
removed before splitting. The source FASTA, ordered split files, selection
script and executed configuration are SHA-256 recorded.

- UniRef50: 300,000 train, 5,000 validation and 5,000 test sequences of at most
  256 residues, selected from the documented eligible prefix and shuffled with
  split seed `20260717`.
- ZymCTRL Swiss-Prot EC cohort: 44,000 train, 2,500 validation and 2,500 test
  unique sequences of at most 256 residues, using the same split seed. The
  deployed `EC<sep><start>protein<end>` records are normalized to pure protein
  sequence in the manifest; the header-verified EC prompt is reconstructed
  only at tokenization time.
- Training sees only `train`; checkpoint selection may use `validation`; all
  reported final quality estimates use `test` once.

The UniRef50 `family` field is its UniRef50 cluster identifier. The ZymCTRL
field is the complete EC number parsed from the FASTA identifier. These fields
are provenance metadata for dictionary training; family-blocked inference is
performed in downstream semantic and recoverability protocols.

## Preliminary online TopK replication

The pre-existing online queues cover ProtGPT2, ZymCTRL and ProGen2-medium. For
each model, train
independent seeds `17`, `29` and `43` with:

- `d_clt=8192`, `k=128`, decoder window 8;
- batch size 2, 200,000 optimizer steps, Adam at `3e-4`;
- 3,000-step warmup, gradient clipping at 1.0;
- dead-feature resampling every 2,500 steps after a 5,000-step inactivity
  threshold; and
- exact exclusion of padding from model attention, reconstruction loss, FVU,
  L0, firing/dead tracking, resampling probabilities and final evaluation.

Runs are independent single-GPU replicas. A GPU is not used as a DDP replica
for a different seed. Checkpoints include optimizer, scheduler, RNG and data
cursor state so an interrupted confirmatory run cannot silently change its
sample stream.

These online-manifest replicas are preliminary quality diagnostics. They do
not use the immutable activation cache below and therefore are not an exact
matched comparison against the alternative dictionaries. They cannot satisfy
P0-2 by themselves and must not be pooled with the cached panel. The exact
comparison retrains `topk_clt` through `scripts/58_run_dictionary_controls.py`
on the same cached rows as all controls.

### Post-freeze numerical-stability amendment

The first three online queues used immutable code archive SHA-256
`556f7ef8519a1d669921d195919cfa229885c664c925f0f309b9fc77bc4bc684`.
At step 15,000, one resampling event replaced 193,287--200,333 features
(65.5--67.9% of the panel, depending on seed) from a single activation batch.
Seed 43 first logged a non-finite loss at step 15,400 and remained non-finite;
seeds 17 and 29 remained finite but were terminated near step 25,000 because
their uncapped resampling histories were not comparable to a corrected seed 43.
All three runs are discarded as preliminary failure diagnostics and cannot be
resumed or used in a scientific aggregate.

Before any restart, the stability contract was amended to select dead features
oldest-first and replace at most 5% of each layer per resampling event. It also
adds immediate non-finite loss/gradient failure, finite checkpoint-state
validation, 5,000-step integrity-checked resumable checkpoints, two-checkpoint
resume retention and model-only analysis snapshots every 25,000 steps. Active
checkpoints are written on GPFS; an existing output directory, checkpoint or
staging directory is a hard error. Arbitrary interrupt-time checkpoints are
forbidden because a forward pass can advance firing/cursor state before an
optimizer update commits.

After all nine replacement queues published complete step-200,000 manifests,
the redundant step-195,000 resumable predecessors were removed on 2026-07-20.
All nine final checkpoints and all 63 prescribed model-only trajectory
snapshots at steps 25,000--175,000 remain retained. This terminal compaction
does not change checkpoint eligibility or the analysis schedule.

The amended resampling implementation was first tested with an intentionally
aggressive 600-step, full-width seed-43 smoke run (resampling every 50 steps,
100-step inactivity threshold). All events respected the 5% cap and the run
remained finite. A separate resume from step 300 reproduced the uninterrupted
step-600 `clt.pt` byte-for-byte (SHA-256
`d5c5cdef06f385ca510e6c92d73860175c72671a21602d753d3a625848d67867`);
all optimizer tensors, parameter groups, scheduler state and RNG/data-cursor
state were semantically identical. This is a stability/resume validation, not
a dictionary-quality result and not a P0-2 gate pass.

#### Second post-freeze amendment: frozen-model inference precision

The replacement r6 online queues exposed a separate upstream numerical
failure. For the ProtGPT2 batch containing UniRef50 records `B2KU69` and
`A0A4V2PAV5`, frozen-model float16 inference produced non-finite CLT-input and
MLP-output tensors from layers 24--35. The CLT parameters and Adam state
remained finite, localizing the failure before the trainable dictionary. A
deterministic replay of the same token batch with bfloat16 frozen-model
inference made every input and target tensor finite across all 36 layers; the
largest absolute target activation was 37,632. This remains within the finite
range of the float16 cache representation.

The precision change is a narrowly scoped, failure-driven amendment. Frozen
model inference is now bfloat16 for both the preliminary online queues and the
exact-cache extractor. The online trainer verifies that every floating model
parameter is bfloat16 before the first activation, checks every captured
CLT-input and MLP-output tensor for finiteness before conversion to the
float32 CLT path, and writes the declared/observed dtype receipt into resumable
trainer state. The exact-cache runner performs the same parameter and
activation checks before any storage conversion or write. Cache storage
remains float16, and the existing post-conversion finiteness check remains
mandatory. The amended production-profile, activation-provenance, execution
report and completion-receipt schemas record bfloat16 inference, verified
parameter dtypes, the all-captured-activation check and float16 storage
separately.

All r6 online runs are discarded and must restart from empty output
directories; none may be resumed, pooled or used for P0-2. Likewise, any
float16-inference exact-cache partial output predating this amendment is
ineligible and must be discarded. The new exact-cache panel must start in a
fresh root and cannot credit an old completion receipt. No cohort, token
selection, model identity, seed, optimization schedule, estimand or quality
gate changed. This amendment is a numerical-integrity correction, not a gate
pass or permission to inspect held-out outcomes.

#### Third post-freeze amendment: complete first-party code binding

A read-only receipt audit found that the first bfloat16 production attempt
bound the runner and selected module hashes but not the complete imported local
code inventory. That r7 attempt was stopped before completion, produced no
completion receipt and is permanently ineligible. Its staging root is retained
under the explicit `ineligible_unbound_code_receipt` suffix and must not be
renamed, resumed or credited.

On 2026-07-18, storage maintenance removed only the inactive 184,683,511,688-
byte `.protgpt2.tmp-15141` payload after confirming that PID 15141 was dead and
that no completion receipt existed. The explicitly suffixed parent root and
its 175-byte diagnostic log remain retained; the log SHA-256 is
`9e4dc8161fc1346dccbe050fac7e470f84d2a856b30104ac78dad5fc8db8acf5`.
This compaction does not change the r7 attempt's permanent ineligibility.

The replacement exact-cache runner now verifies one immutable code archive and
the deployed `CODE_CONTENT_SHA256SUMS` before importing any local `src` module.
The supplied manifest path must resolve to the manifest beside the executing
project tree. The archive must contain exactly one
`r2_interpretability_transfer` root, the byte-identical manifest and exactly
the regular files listed by it. Absolute or escaping paths, duplicate members,
links, non-regular members, unlisted files, missing files and any content/hash
mismatch fail closed. The deployed tree is checked against the same exact file
inventory and byte hashes. Local bytecode writes are disabled so a verified
tree cannot acquire an unmanifested `__pycache__` between panel models.

Activation provenance, cache-execution reports, preflight reports and
completion receipts use their v3 schemas and carry the code-archive SHA-256,
code-content-manifest SHA-256 and a true exact-inventory verification flag.
Production-cache validation and receipt-gated capacity credit require the same
binding; a cache produced by another code archive cannot be silently reused.
The H200 launcher owns the complete ProtGPT2, ZymCTRL and ProGen2-medium sequence
itself, checks the four deployed archive/manifest/profile/runner hashes before
starting, verifies that GPU 3 is free before each model, and records GPU and
host-memory state both before and after each completed model. This amendment
changes provenance integrity only; cohort, estimand, token budget, storage
dtype, model identity and scientific quality gates are unchanged.

#### Fourth post-freeze amendment: cache-completion receipt launch guard

The production validation-screening dispatcher must verify independently
pinned completion-receipt and cache-manifest SHA-256 values before each
invocation. It requires receipt schema v3, status `verified_complete`, the
correct model/profile/payload, bfloat16 model inference, float16 cache storage
and a verified exact code inventory. It also rehashes the receipt-referenced
manifest and execution report and compares the cache-content identity.

The dispatcher refuses an existing output directory or occupied target GPU.
After the runner exits, it requires a protocol-valid terminal screening status,
seed `20260717`, zero test evaluations and `p0_2_eligible=false`. This amendment
changes provenance and completion eligibility only; it does not change
cohorts, model identity, cached rows, candidate grids, selection, estimands or
quality gates, and it does not pass P0-2.

#### Fifth post-freeze amendment: candidate memory lifecycle and fresh screening lineage

The first receipt-bound screening lineage completed three candidates in each
of four active runner processes, then failed on candidate four because prior
candidate model and Adam state remained live on the accelerator. Peak
allocation increased by exactly 12 bytes per trainable parameter after each
completed candidate. The resulting r1 partial outputs are permanently
ineligible: they may be retained as compact failure diagnostics but cannot be
resumed, pooled with patched candidates or credited toward a screening
selection.

This failure-driven correction changes candidate-local resource lifecycle
only. After final checkpoint and peak capture, training clears gradients and
optimizer state. The runner deletes candidate checkpoint/model objects,
collects Python garbage, synchronizes the target CUDA device, empties its
cache, and records allocated and reserved memory. A multi-candidate process
fails closed unless post-candidate allocated memory is no greater than 128 MiB
and remains exactly stable after its first observed baseline. The bound admits
the measured stable 64 MiB PyTorch allocator residue while rejecting
candidate-wise accumulation.

Before a production restart, a nonconfirmatory full-width gated preflight must
cross four independent candidate boundaries in one process and satisfy the
same memory guard with zero test access. The authoritative preflight completed
four repetitions with identical 62,162,163,712-byte peak allocation,
62,182,653,952-byte peak reservation and 67,108,864 bytes allocated after
every repetition. It retained only its ineligible report.

Every corrected production restart uses a new immutable archive and deployed
tree, a new launcher and an empty output root. The r3 archive, tree manifest,
runner, dictionary module, profile, preflight runner and launcher SHA-256
values are, respectively:

- `6f191b554de901c7be25968f2fc96d989ae7d5d0bcb2a1c9285fdd1c3b840e44`;
- `486ba4e9c00a7a1a88a58a31691095e72aaf45f39745ad3117b88c694de579d4`;
- `56ca3c4d8e230ea6ef5cf36d394564f747fa2c5bcb4f9b837dd3e5825e6401b8`;
- `347a095c2e18a429e09011f84a45bd40b1cf5d46ebf38398e6e2a7afe57c6596`;
- `eb33d6e8fdf551b60b95238766fcf97e3e2fe5a91f0f5882dd9212d129572db2`;
- `8e5ee2ada79ad67ac869eb34bb36594e3188179bff6c6c35b404007a37813a94`;
  and
- `bc511f78f872ce70cc85ed098ec9aad15e44aa2dc1a5add3de2a52683fcef2f6`.

This amendment does not change a cohort, source/cache row, model identity,
candidate grid, screening seed, step count, validation-selection rule,
estimand or scientific quality gate. Preflight success, memory stability and
process liveness do not pass P0-2 and do not authorize test access.

## Frozen quality gate

Report all seeds and layers, without selecting a favorable seed. A dictionary
may enter the confirmatory atlas only if:

1. all padding-invariance and all-valid-equivalence tests pass;
2. every checkpoint and test cohort hash agrees with its run manifest;
3. mean held-out test FVU is below 0.50 in every seed;
4. at least 75% of layers have held-out FVU below 0.50 in every seed;
5. median dead-feature fraction is below 0.50 in every seed; and
6. any layer used downstream has dead-feature fraction below 0.70 in every
   seed and is selected without using test-set biological outcomes.

Failure narrows the result to a dictionary-quality limitation. It does not
license a biological atlas from the failed model/seed.

The prospective `FVU < 0.50` rule is a minimum inclusion screen: it requires
more reconstructed than unexplained held-out variance on average in every
seed, plus the layer/dead-feature conditions above. It is not described as
high-fidelity reconstruction, and FVU remains a continuous covariate in every
downstream quality--recoverability analysis. This new masked, multi-seed rule
does not replace or retroactively relax the historical wider-dictionary
`FVU < 0.15` preregistered criterion. Those historical wider runs failed their
own criterion and remain failed; both thresholds and their distinct scopes are
reported.

## Alternative dictionaries and dense control

All methods estimate the same window-8, multi-layer transcoding objective. A
source layer encodes the architecture-specific CLT-input tensor and its code may
write to the MLP-output targets at that layer and the following seven layers,
truncated at the final layer. The cached TopK CLT, conventional ReLU/L1 SAE, gated SAE
and dense low-rank control therefore differ in bottleneck/activation rule, not
in source tensor, target tensor or decoder window.

The TopK CLT, ReLU/L1 SAE and gated SAE each have width 8,192. TopK fixes
`k=128`. Before the three confirmatory seeds, ReLU/L1 and gated coefficients and
reporting thresholds are screened once per model with seed `20260717`: each
candidate trains for 10,000 steps on the first 100,000 rows of the cache's
hash-priority order, with 1,000 warmup steps, and uses all 100,000 validation
rows. Mean validation L0 must lie within 10% of 128. Among candidates in that
interval, select the lowest validation FVU; a method with no eligible candidate
is reported as failing the sparsity-match requirement. The chosen coefficient
and threshold are frozen before full 200,000-step seeds `17`, `29` and `43` and
are not reselected per seed. Cached TopK (`k=128`) and dense rank 128 have no
regularization grid and skip screening. No screening run accesses test data.
The frozen loss grids, activation-threshold grid and
tie-breaking rule are in
`configs/p0_2_dictionary_controls_production_profile.json`.

The dense control has rank 128, matching the active TopK bottleneck width. It is
explicitly **not** matched to the raw parameter count of either 8,192-wide
sparse alternative. Report raw trainable parameters, a prespecified training
and inference FLOP proxy, wall time, and peak allocated and reserved accelerator
memory for every method; these differences remain visible in all comparisons.

The cached TopK and every control/hyperparameter candidate use the same
immutable activation cache, selected layers, split rows, padding mask and row
order. Run independent
seeds `17`, `29` and `43`, with byte-identical cache manifests across methods
within each model. Test data are evaluated once after validation-only selection.
The comparison reports test FVU, reconstruction-error quantiles, L0, firing
frequency, dead fraction and decoder norms for every seed. Alternative methods
are quality controls; they do not enter downstream biology unless they
independently pass the same frozen gate. No method is selected using atlas or
semantic outcomes.

### Frozen activation-cache sampling and storage

The alternative controls do not cache every token from the sequence cohorts.
For each model independently, cache exactly 1,000,000 train, 100,000 validation
and 100,000 test valid model-token rows. Selection assigns every valid token the
SHA-256 priority `sha256(sequence_sha256 + ":" + token_position)`, where
`token_position` is its ordinal in the unpadded model-token stream, and retains
the lowest priorities in each split. This rule is independent of input path,
manifest order and batch padding. An exact budget is mandatory; insufficient
eligible rows abort extraction. The three source manifests must remain
sequence-disjoint.

Model inputs are reconstructed from frozen cohort rows before tokenization:
ProtGPT2 and ProGen2-medium receive `sequence`; ZymCTRL receives
`family<sep><start>sequence<end>`. Tokenization uses deployed tokenizer special
tokens, right padding, truncation and a maximum of 256 model tokens. The sampled
population includes every non-padding model-input token, including any control
or tokenizer-added special tokens. The exact selected `(sequence_sha256,
token_position)` keys and their ordered selection-file hashes are archived.
The first tokenizer pass uses batches of 256 unpadded inputs; activation
extraction uses sequence batches of two, matching the primary trainer. Both
padding and truncation side are right. Per-record and aggregate token-ID
fingerprints must agree between passes before a cache can be finalized.

Extraction hashes the deployed `config.json`, canonical local weight-file
tree, and remaining tokenizer/support-file tree and compares all three with
the declared digests before model loading. Digest strings are not accepted as
verification by themselves. Screening opens, hashes and scans only train and
validation source, selection and activation members. Test metadata remains
bound into the content identity, but test file contents are not accessed until
a full run.

Caches use bfloat16 frozen-model inference and float16 stored activations, with
one preallocated input plus target array per layer and split; exact budgets
make the array shapes known before extraction. Every captured CLT-input and
MLP-output tensor is checked for finiteness before conversion or write, and
every converted selected row is checked again before publication.
This bounds the full three-split reader to 216 activation files for a 36-layer
model, below the execution environment's open-file limit. At the frozen panel
geometry, the estimated activation payload is 221,184,000,000 bytes each for
ProtGPT2 and ZymCTRL and 199,065,600,000 bytes for ProGen2-medium: 641,433,600,000
bytes total (approximately 597.4 GiB). Before the first cache, and again before
every subsequent extraction, the panel-root gate must observe free storage at
least 1.2 times the payload of all caches not yet completed. A cache is credited
only after full validation and a hash-bound completion receipt. The first gate
therefore requires 769,720,320,000 bytes; later gates deduct only
receipt-verified model payloads. The panel/per-model capacity reports, storage
dtype, tokenizer configuration, input format, selection rule and exact split
budgets are part of the cache/run provenance.

The compute-planning estimate assumes one optimizer step per second until a
measured H200 calibration is available. At that rate, one 10,000-step screening
candidate is 2.78 GPU-hours; the 15 sparse candidates are 41.67 GPU-hours per
model and 125 GPU-hours across three models. Twelve full exact-cache runs per
model (cached TopK plus three controls, each with three seeds) are 666.67
GPU-hours per model and 2,000 GPU-hours total. The aggregate optimizer-step
planning estimate is therefore 2,125 GPU-hours. It excludes cache-extraction
and evaluation time: extraction wall time, ten validation passes per screening
candidate, forty validation passes per full run, and one final test pass are
reported from observation. Resumed runs carry forward cumulative wall time and
peak allocated/reserved accelerator memory. The gated proxy includes its
auxiliary decode; TopK selection work is reported separately from matmul FLOPs.
Each progress checkpoint has an atomic identity-bound timing sidecar so resume
also carries forward checkpoint-write time; the metric excludes only the tiny
sidecar bookkeeping write itself.

Checkpoint storage is planned independently of activation-cache storage. An
active candidate may retain a float32 best model (4 bytes per parameter) plus a
resumable float32 model/Adam checkpoint (12 bytes per parameter), or 16 bytes
per trainable parameter. Retaining all planned checkpoints requires
2,139,581,276,160 bytes for screening and 1,290,362,603,520 bytes for the four
full methods, 3,429,943,879,680 bytes total. In the conservative worst case in
which every validation improves the best checkpoint, repeated checkpoint
writes total 73,010,316,902,400 bytes. Each runner invocation performs a 1.1x
free-space gate for its retained requirement plus a 12-byte-per-parameter
temporary file during atomic progress-checkpoint replacement. These figures
and the exact per-model/method parameter counts are executable profile
invariants.

Before production extraction, the cache entry point may run a bounded H200
preflight with `--preflight-only`. It takes only the first two source records
and exactly two valid token rows per split, archives a derived profile, and
reports wall time and peak accelerator memory. The runner can then consume that
miniature cache with its own `--preflight-only` mode and execute exactly one
full-width optimizer step on two valid-token rows, without writing a model
checkpoint. Running the gated method exercises the largest auxiliary graph.
Both reports and the miniature cache are marked non-confirmatory,
P0-2-ineligible, and forbidden for production reuse. Passing either preflight
calibrates safety/throughput only; it does not relax a frozen production gate.

## Canonical entry points

- Split construction: `scripts/56_prepare_dictionary_splits.py`
- Budgeted activation-cache extraction:
  `scripts/61_build_dictionary_activation_cache.py --panel-cache-root ...`
- Bounded non-confirmatory cache and dictionary-optimizer H200 preflights: each
  production entry point with `--preflight-only`
- Exact-cache TopK/control screening/full runs:
  `scripts/58_run_dictionary_controls.py --stage screening|full`
- Repeated full-width candidate-memory lifecycle preflight:
  `scripts/73_preflight_dictionary_candidate_lifecycle.py`
- Receipt-bound four-GPU validation-screening dispatcher:
  `scripts/75_run_dictionary_screening_queue_h200.sh`
- Preliminary online TopK training only:
  `scripts/57_run_masked_clt_seed_queue_h200.sh`
- Held-out evaluation: `scripts/05_evaluate_checkpoints.py --manifest ...`

The launch command, pod, GPU index, before/after resource state and output
hashes are appended to `docs/EXPERIMENT_LOG.md`. This protocol does not mark
P0-2 complete until the alternative dictionary comparison and all held-out
quality reports exist.
