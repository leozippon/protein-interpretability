# npj AI major-revision execution status

**Assessment:** `npj_ai_manuscript_assessment.md`  
**Binding task plan:** `NPJ_AI_MAJOR_REVISION_PLAN_20260716.md`  
**Status date:** 2026-07-22  
**Disposition:** active major revision; not submission-ready

This ledger distinguishes implementation, live execution and scientific gate
outcomes. Passing a unit test or launching a GPU job does not pass a P0 gate.
The detailed acceptance criteria remain those in the binding task plan.

## Current assessment

The repository is substantially stronger as an auditable experiment system,
but the assessment's scientific conclusion has not yet changed. Historical
claims are now accurately bounded, the exact atlas cohort and reference CLTs
have been recovered locally, and every new real-data pipeline is being made to
fail closed on one common dictionary-eligibility receipt. The dependency that
controls the remaining computational route is P0-2: downstream confirmatory
work cannot start until all required exact-cache dictionaries pass the frozen
held-out quality gates.

The active H200 pod has four H200 GPUs and adequate host memory. A deterministic
replay showed that frozen ProtGPT2 fp16 inference, rather than the CLT or Adam
state, produced non-finite layer-24--35 activations for one legitimate
UniRef50 batch. The panel-wide bfloat16 amendment, exact-cache conversion fix
and bounded preflight passed; the failed fp16 lineages remain ineligible. All
nine replacement online TopK runs and all three exact activation caches have
completed. The first receipt-bound sparse-screening lineage failed after 12 of
45 candidates because prior-candidate optimizer/model state accumulated on each
GPU; it is quarantined and permanently ineligible. The corrected r3 screen is
terminal at 45/45 validation candidates with zero test accesses: three sparse
pairs selected and three reached calibrated `sparsity_match_failure`. The
exact 27-run receipt-authorized full panel is now scheduled across four active
queues on GPUs 0--3. All four first runs reached step 5,000 with valid
`in_progress` states and atomic checkpoints. These are prerequisite and
execution facts, not held-out quality results. No current
compute issue requires user action.

A separate downstream audit also removed the last production float16 paths.
P0-3/4/5/6/8 now require bfloat16 model parameters before the first activation,
check every required activation/attention/logit tensor for finiteness before
conversion or use, propagate a common six-field numerical-integrity receipt and
reject legacy or tampered production schemas. This is verified contract
infrastructure, not a pretrained-model outcome.

## Priority-zero ledger

| Package | Implemented or recovered | Live/observed state | Remaining acceptance work |
|---|---|---|---|
| **P0-1 exact reproducibility** | Exact historical balanced-200 file recovered and ordered records verified; exact three atlas reference CLTs located and hash-verified; strict manifests, RFC-8259 normalization and panel/table map exist. The complete deployed ProtGPT2 tree is verified to upstream commit `f71aa6cf063ad784ebd53881d11332fd098eaa58`; model-specific manifests retain the narrower ZymCTRL and ProGen2 findings. The live exact-cache run has a separate hash-bound Python/package/GPU environment receipt. | ZymCTRL is best-supported at `3c532ef172b9cd2e95238baadf5167ebb89fbc32` with its weight verified but strict whole-tree proof incomplete. The deployed ProGen2-medium tree is a hybrid that matches no single upstream commit. Historical tool/database environments and several raw artifacts remain incomplete; recovered inputs remain on archived/local compute storage, not a public deposit. | Deposit the exact ProGen2-medium local snapshot and manifest; resolve or explicitly retain the ZymCTRL whole-tree uncertainty; archive complete generation/intervention raw artifacts and tool/database versions; replace every incomplete provenance row; obtain an approved tag and DOI-backed licensed deposit. |
| **P0-2 mask-aware dictionaries** | Mask propagation, masked objectives/metrics/resampling, deterministic split/resume contracts, padding tests, exact activation-cache/control panel and fail-closed eligibility receipt are implemented. The bfloat16 amendment verifies frozen-model parameter dtype and activation finiteness before conversion; exact cache storage remains float16 with a second conversion check. The r8 extractor additionally binds the complete imported first-party code archive and exact deployed inventory before any local import. Candidate-local optimizer/model release and a stable post-candidate CUDA-memory guard now prevent cross-candidate accumulation. The adjudicator handles an exact sparsity-match failure as a calibrated negative and consumes a hash-bound confirmatory mask receipt. | The failed fp16, unbound-code and r1 screening attempts remain ineligible. The corrected r3 validation-only screen completed 45/45 candidates with zero test accesses. It selected ProGen2-medium gated SAE, ZymCTRL gated SAE and ZymCTRL ReLU/L1 SAE; ProGen2-medium ReLU/L1 SAE and both ProtGPT2 sparse methods terminally failed the calibrated sparsity match and their nine seed runs are forbidden. Terminal receipt SHA-256: `83bf4563a27886a1e1f148eb4094f9a7b02533936a02981caac17a805de04d6c`. The exact 27-run authorized full exact-cache panel was scheduled across four fresh queues on GPUs 0--3 at about 11:46 CST on 2026-07-22. Both required mask tests passed and their durable receipt validated through the production gate. No held-out P0-2 quality gate has occurred. | Complete the 27 authorized full runs; publish their atomic checkpoints, manifests and eligibility receipts; aggregate and adjudicate the frozen layer/seed FVU, dead-fraction, firing, norm and error-quantile gates before exposing any downstream biological labels. |
| **P0-3 independent atlas** | Discovery/evaluation separation, signed and positive-only scoring, coherent model-wise nulls, plus-one p-values, four matchers, stability grids and variance decomposition are implemented. Production inputs now require P0-2-eligible seed-specific `best.pt` files. | Synthetic and contract tests only. The current pretrained panel has one frozen checkpoint per model (`model_seed=0`); dictionary seeds do not identify pretrained-model-seed variance. | Build real A/B activation matrices after P0-2; freeze analysis manifests; run dictionary seeds 17/29/43 and at least 1,000 coherent null replicates; adjudicate held-out correspondence and method/cohort stability. Obtain genuinely independent pretrained checkpoints or report model-seed variance as unidentifiable rather than imputing it from dictionary seeds. |
| **P0-4 conditional semantics** | Continuous-activation models, protein/family blocking, conditional randomization, shared controls, multiplicity, group bootstrap and independent prospective-power receipt are implemented and bound to P0-2 inputs. | Synthetic and contract tests only. | Freeze label families/covariates and an independent power plan; run on eligible A/B matrices; jointly adjudicate residual effects or prespecified small-effect bounds. |
| **P0-5 N-terminal counterfactuals** | Exact natural/M-to-A/internal insertion/truncation factorial, causal-opportunity-normalized attention, matched protein/control-feature pairs, focal-position checks, paired difference-in-differences, all-layer focal-key blockade and conditional non-mediation diagnostics are implemented. Exact-cache eligibility is mandatory. The analyzer requires external hashes, revalidates the production extractor receipt/artifacts, consumes frozen feature-pair membership without rematching and publishes atomically. | No real measurements. | Freeze real cohorts, features, matching calipers and equivalence margins after P0-2; extract all conditions on held-out proteins; run paired uncertainty and joint equivalence adjudication. Do not call the result an attention sink or formal mediation test. |
| **P0-6 corrected steering** | Positive-only selection, fail-closed cells, independent attribution freeze, prompt/random/norm controls, paired generation plan, immutable execution receipts and external score-receipt verification are implemented. Real execution is being bound to the same P0-2 `best.pt` eligibility chain. | No pretrained-model generation or endpoint score run. | Complete eligibility binding; freeze validated generation-wide scorer/calibration versions and equivalence regions; execute the eight-class dose/site panel; retain every prompt/token/sequence; score once; multiplicity-correct and adjudicate a clean positive or equivalence-bounded negative. |
| **P0-7 causal redesign/positive control** | Historical target interventions remain bounded negatives. The earlier analytic simulator and token-exposed trained GRU are explicitly nonconfirmatory. The prospective unexposed long-range planted benchmark was frozen before execution and passed its prespecified sensitivity, specificity, FDR, localization, effect-recovery and matched-control-equivalence gates across three untouched model seeds. A separate inference-free pretrained adjudicator now rehashes the complete synthetic prerequisite, P0-2 eligibility chain, P0-5/P0-6 receipts, frozen identities, raw factorial rows, validated external scores and code; it preserves all paired rows, applies one global multiplicity family and publishes atomically. | The synthetic pipeline-sensitivity subgate and adversarial adjudicator contracts passed. This is not a learned biological circuit or evidence about pretrained protein models; no eligible real intervention surface, adjudication run or completion receipt exists. | Freeze and execute the held-out target/control model/feature/layer/site/dose surface with fidelity, off-target, reconstruction, logit, behavior and path-patching/mediation endpoints, then adjudicate once against the frozen equivalence regions. |
| **P0-8 recoverability** | Nested group-disjoint folds, inner-only layer selection, repeated seeds, paired bootstrap, matched-dimensional PCA/random-projection/NMF/ICA/random-dictionary controls and an exact-cache production input receipt are implemented. Reconstruction quality and intervention efficacy remain separate for every dictionary seed/layer through analysis; production requires at least 480 annotated rows. | Dimension-truthful synthetic plumbing only; the 48-sequence legacy decoder cohort is ineligible. | Build enlarged identity-aware global and local protein tasks from P0-2-eligible seed representations; run identical folds across all bases; relate paired `F-C` intervals to seed/layer-specific dictionary quality and intervention efficacy. |
| **P0-9 figures/chronology** | Historical chronology and claim defects are corrected; Figure 1 expresses R/S/C/U; a machine-readable provenance map and strict processed package exist. Supplementary Table 11 now reports the bounded synthetic calibration, and Figure 5 is explicitly labelled mixed-run historical evidence. | Current figures represent bounded historical evidence, not replacement P0 results. The current package verifies 65 manifest rows, 68 checksums and 1,649,686 bytes. | Rebuild Figures 2--6 only from canonical P0 outputs; replace Figure 5 with one equal-arm, identically selected, single-version run; show full distributions, uncertainty, dose/equivalence/fidelity and complete panel-level provenance. |

The pretrained P0-7 adjudicator is intentionally one shared module plus a thin
CLI and example specification; it adds no second inference or feature-matching
pipeline. Final SHA-256 values are module
`696b7e3ace1e6eaacb9dc14691064abb77f673ad9cecc089c9d7c7c9231cfb19`,
CLI `7478a9787b60258004d479e6716a64c6756ad9e3bfa4366c64ad43d6658cc41a`
and example specification
`271e71d35fd38340edf496577db041fbca378c4b06d95f2035b2c532abcf5915`.
Its focused adversarial suite passed 33 tests; the complete R2 suite passed 215
tests plus 6 subtests. Ruff check and format, byte compilation, CLI loading and
strict example-JSON parsing passed. This validates the contract only; no real
P0-7 input or scientific result was fabricated.

## Active H200 execution receipt

At 2026-07-17 19:26 CST, three independent TopK queues started from the fresh
`dictionaries_topk_bf16_r2` root. Their immutable online payload archive is
SHA-256 `67f02eb8c09e07303b005318da3bd7146a461fec6536e95cbe40daaaabd273ea`,
the deployed tree-manifest SHA-256 is
`7aa9012d7acd7e6ffeb054a61a7e1ab717336484dc8c71f9b1996eed2940b905`,
and launcher SHA-256 is
`58d1114b46c451f8263f6c5b5da11a95dc9d52f716eae0e83f19aa5f7d16f3e9`.
Pod PIDs 13694/13695/13696 bind seeds 17/29/43 to GPUs 0/1/2. All three
reported verified `bfloat16` inference, published valid atomic step-25,000
checkpoints, remained finite past step 28,000 and used approximately 53.5 GiB
per GPU at 0.24--0.25 seconds per step. Interim displayed dead fractions near
0.68--0.70 are retained as diagnostics and are not a held-out gate result. The
current ProtGPT2 phase is therefore approximately 13--14 hours; the sequential
three-model queues are provisionally estimated at 48--60 hours.

That 48--60-hour estimate applies only to the preliminary online TopK queues.
The frozen alternative-dictionary plan separately budgets 2,125 aggregate H200
GPU-hours for screening and full controls, excluding cache extraction and
evaluation. Even with ideal four-GPU occupancy that proxy is about 22 days;
queueing, validation and extraction make the calendar estimate longer. The two
timelines must not be conflated.

The r7 bounded ProtGPT2 exact-cache preflight remains a valid nonconfirmatory
software check, but the subsequent r7 production attempt was stopped after a
read-only audit found that its completion schema did not bind the complete
imported code tree. It produced no completion receipt. The staging root and
logs were renamed with `ineligible_unbound_code_receipt` and are forbidden for
reuse.

The authoritative r8 exact-cache payload archive SHA-256 is
`eb646dffac5b2bfe151af47e36b53ad2233c1204af69a819529897c5a257ebe9`;
its embedded/deployed code-content-manifest SHA-256 is
`3098a61a1b1617d023fa1badd92d28727f121d7f2d1a217c0b9b20e06be5fad6`.
The production profile, runner and full-panel launcher SHA-256 values are,
respectively,
`eb33d6e8fdf551b60b95238766fcf97e3e2fe5a91f0f5882dd9212d129572db2`,
`1e3ffdcb2b721bc0d119dc15bdf1804b3cac43bc98bf2174caacd882d59a801e`
and `c87a45e3454b687c15853e6d90b2958cb263e7ef6c6c385b0aa70295864af5d1`.
The v3 bounded preflight verified the exact archive/tree inventory before local
imports, all floating ProtGPT2 parameters as bfloat16, all required captures as
finite, exactly two rows per split and float16 cache storage; it remains
production-ineligible. Its report SHA-256 is
`3cae6fa190cd11408a1eb98c835c943fc5d806334ed56d8bc57c6d68474ed9a4`.

At 21:12:34 CST, launcher PID 16502 started the single
ProtGPT2--ZymCTRL--ProGen2-medium r3 queue on GPU 3; initial ProtGPT2 runner PID
16522 uses the fresh `p0_2_exact_cache_bf16_r3` root. A separate machine-readable
environment receipt binds Python, PyTorch/CUDA, NumPy, Transformers, Tokenizers,
Safetensors, the complete `pip freeze` and all five immutable run hashes; its
SHA-256 is
`3dce112700e17ec0bda8534527f731f67bf31b0946f9408d19075759ea6e17dc`.
No scientific gate is credited from process liveness, preflight success or
projected duration.

At 22:14 CST, the canonical GPFS training logs were finite through steps
39,550/39,700/39,750 for seeds 17/29/43. Their latest complete step-35,000
checkpoint-manifest SHA-256 values were, respectively,
`13ae8fb16dea3694951ca8e53e7eaef6044cd8e20b5378edc1208854f43a51af`,
`ad3dd29150c49b6cb5bc161de7b7908defac8869d47f17edb895dfce3e2946e4`
and `2369c1c288273f67af89f157bb38be4012c18f9faf0b61a5752007c592b64a9e`.
The active log files are under the fresh
`dictionaries_topk_bf16_r2/<model>/topk_clt/seed_<seed>/logs/` roots and the
GPFS launcher-log directory. Older similarly named files under the OSS output
log directory are superseded launch artifacts outside this production lineage;
one contains non-finite rows and must not be used to assess the active seed 43.

The r8 ProtGPT2 exact-cache runner remained alive in GPFS-heavy extraction
with 173 GiB staged and no completion receipt. All four processes retained
their original PID lineage; GPU memory was approximately 54.8 GiB on GPUs
0--2 and 2.8 GiB on GPU 3, host memory had about 1.8 TiB available and GPFS had
about 46 TiB free. The disk-wait state is expected for the preallocated exact
cache and is not a resource fault. The cache and every P0-2 gate remain open.

### 2026-07-18 execution transition and storage receipt

All three online ProtGPT2 TopK seeds completed 200,000 steps and published
complete final checkpoint manifests before their queues advanced to ZymCTRL.
At 12:09 CST the ZymCTRL seeds were finite at steps
23,650/24,200/23,600. Their current phase is expected to need about 18 more
hours; the remaining ZymCTRL-plus-ProGen2-medium queue estimate is 31--43
hours. These are online diagnostics only and remain excluded from the exact-
cache P0-2 alternative-dictionary comparison.

The r8 exact-cache queue completed its ProtGPT2 member at 07:54 CST. Its
internally consistent completion receipt, cache manifest, execution report and
cache-content SHA-256 values are, respectively,
`c5e323df9618c14db9ec4e19332dc1b162a4a026b68381b2b1fd3c7954a73225`,
`d3fa68212612bb42ed9d75c0b49a606db44a9171c00b0662c05b41f48f63dd34`,
`9ada765572ba519c28e8fed0b5f4bbd34aceb41ed642acf8b8c215ce5304af2a`
and `ddf4ea1e99d5b382124b31b80fba1bcd99c4e84a9c48b00e28827c955c4f87dc`.
The receipt records 221,184,000,000 payload bytes, verified bfloat16 inference
and finite captures. ZymCTRL extraction is active as PID 17403; the separate
queue has a lower-confidence 14--22-hour remaining estimate.

Bounded storage maintenance removed 743,059,375,694 apparent bytes (692.03
GiB) of superseded or permanently ineligible payloads while retaining compact
logs/configurations/manifests and every active or eligible root. The r7
`ineligible_unbound_code_receipt` parent and diagnostic log remain, but its
inactive partial tensor payload was deleted. The path-level receipt is
`evidence/h200_cleanup_20260718/cleanup_manifest.json`; its retained 23,329-byte
evidence archive has SHA-256
`7601f41f8a859ceb599215b29d717d1d31cad08b6d3afd7a95dcc062f45b09c1`.
GPFS usage fell from 18% to 17%. Active launchers and PID 17403 survived the
cleanup, resources remained safe and no H200 issue required user action.

At the 2026-07-18 checkpoint, ProtGPT2 exact-cache completion was one
prerequisite artifact and ZymCTRL/ProGen2-medium remained outstanding. Their
subsequent completion is recorded below; neither checkpoint passes P0-2.

### 2026-07-20 prerequisite completion, screening transition and storage receipt

All nine preliminary online TopK runs completed 200,000 steps by 23:58 CST on
2026-07-19. Their complete resumable final manifests and finite-log audit are
recorded in EXP-R2-029. The exact-cache queue completed ProtGPT2, ZymCTRL and
ProGen2-medium by 02:48 CST with independently revalidated completion-receipt
SHA-256 values
`c5e323df9618c14db9ec4e19332dc1b162a4a026b68381b2b1fd3c7954a73225`,
`0b4c4e9040556d73dee68421cff13cc1497e4c8348e03a4c72f9eb0bfd0fb013`
and
`31b34aef4a059b061b4979a5afa5a4b07ccc0278b37ecac3e2a69d7153171a80`.
Together the verified caches contain 641,433,600,000 payload bytes.

At 18:10:02 CST, four fixed queues started the six frozen validation-only
ReLU/L1 and gated-SAE screenings as launcher PIDs 24175--24178 on GPUs 0--3.
The immutable archive, embedded tree-manifest and launcher SHA-256 values are
`76ba445cbdb418602df3b0088ac41f88e84935e06614cff0a6ff3a3c99ca3811`,
`3e8628ef9f8e0a0141d491beeffed5b2429ed50836f1c830cc2440095667a713`
and
`caf40fb83c63f5af7ebde4f7b4b459aa76ec9ac53a648a3992d7efe4df7f5c9c`.
The dispatcher verifies the independently pinned cache receipts before every
run and enforces seed `20260717`, zero test evaluations and
`p0_2_eligible=false`. The 45 candidate fits budget 125 aggregate GPU-hours,
approximately 2--4 calendar days on four GPUs including validation and I/O.

After final-manifest and process guards, storage maintenance removed nine
redundant step-195,000 resumable predecessors and one obsolete 20-step fp16
smoke: 395,430,148,930 apparent bytes (368.27 GiB). All nine step-200,000
finals, all 63 prescribed trajectory snapshots, all exact caches and the active
screening roots remain. The path-level receipt is
`evidence/h200_cleanup_20260720/cleanup_manifest.json`; compact remote metadata
archives have SHA-256 values
`ef12accdfedc78f220aa8372eedd9a8e3757240effacea92f3a38948fc1af539`
and
`951a4b01e7ad5ff7ec47c15c1a7e82b0aac84ebaee2e97d548667cf4dcb2683a`.
No H200 fault required user action. Preliminary TopK completion, exact-cache
completion, screening liveness and cleanup do not pass P0-2.

### 2026-07-21 screening failure, lifecycle correction and restart

A terminal audit found that all four July 20 r1 screening workers had failed
after candidates `000`--`002`, each OOMing on candidate `003`. They produced
12 validation diagnostics across the 45-candidate plan, no terminal
`results.json` or `run_manifest.json`, and left stale `in_progress` run states;
the queued ProGen2-medium and ZymCTRL ReLU/L1 invocations never started.
Candidate peak allocation increased by exactly 32,472,373,248 bytes for
ProGen2-medium gated SAE and 37,257,506,816 bytes on a 36-layer gated path,
matching 12 bytes per trainable parameter of retained optimizer/model state.
The pod, all four GPUs, host memory and GPFS were healthy, so this was not an
H200 resource fault.

The root was quarantined as
`p0_2_dictionary_controls_bf16_r1_ineligible_candidate_memory_accumulation_20260721`
and cannot be resumed or pooled. After matching retained best-checkpoint
hashes to validation results, 24 large checkpoint payloads totalling
576,951,858,714 apparent bytes (537.328 GiB) were deleted while 28 compact
validation/timing/run-state files were preserved. The path-level cleanup
receipt is `evidence/h200_cleanup_20260721/cleanup_manifest.json`, SHA-256
`069714b1edc7d961cfe3353f4e3c9249f60255dbad7c41ab5167e48ff59ef76d`.

The minimal correction clears gradients and optimizer state after final
checkpoint capture, deletes candidate-local objects and applies a fail-closed
post-candidate allocated-memory bound: no greater than 128 MiB and exactly
stable after the first baseline. The complete R2 suite passed 216 tests plus
6 subtests in 35.48 seconds; Ruff check/format, byte compilation and shell
syntax checks were clean.

The immutable r3 archive, tree-manifest, runner, dictionary module, profile,
lifecycle-preflight runner and launcher SHA-256 values are:

- `6f191b554de901c7be25968f2fc96d989ae7d5d0bcb2a1c9285fdd1c3b840e44`;
- `486ba4e9c00a7a1a88a58a31691095e72aaf45f39745ad3117b88c694de579d4`;
- `56ca3c4d8e230ea6ef5cf36d394564f747fa2c5bcb4f9b837dd3e5825e6401b8`;
- `347a095c2e18a429e09011f84a45bd40b1cf5d46ebf38398e6e2a7afe57c6596`;
- `eb33d6e8fdf551b60b95238766fcf97e3e2fe5a91f0f5882dd9212d129572db2`;
- `8e5ee2ada79ad67ac869eb34bb36594e3188179bff6c6c35b404007a37813a94`;
  and
- `bc511f78f872ce70cc85ed098ec9aad15e44aa2dc1a5add3de2a52683fcef2f6`.

At 14:20:58 CST, a four-repetition full-width gated-SAE preflight completed
with before allocations of 0/67,108,864/67,108,864/67,108,864 bytes, exactly
67,108,864 bytes after every repetition and identical peak
allocated/reserved values of 62,162,163,712/62,182,653,952 bytes. It accessed
no test rows and retained only its nonconfirmatory report, SHA-256
`10d2cda01547e48a5dbaf5034e057bda48f4578f130dc64db24bb351e6bdcf33`;
its log SHA-256 is
`927601deaa09ae544fbc6c8c4489e94fe2eb10a87c26d130fe224a499d851fd8`.

At 14:22:06 CST, launcher PIDs 27369/27370/27371/27372 started the fresh r3
queues on GPUs 0/1/2/3 under
`p0_2_dictionary_controls_bf16_r3/screening`; initial child processes were
alive. Their observed estimate is 16--24 hours, while the frozen conservative
budget remains 125 aggregate GPU-hours. The queues retain seed `20260717`,
zero test access and `p0_2_eligible=false`. The unused plotting-only,
PyTorch-free Python 3.13 `.venv` was deleted; verification and execution use
the canonical `ct` Conda environment. Neither cleanup, preflight nor relaunch
passes P0-2.

### 2026-07-21 final adjudication-contract closure

The panel gate now accepts terminal `sparsity_match_failure` only as a
complete calibrated negative with zero held-out test access, the full frozen
candidate grid, no L0-matched row and no selected configuration/checkpoint.
It emits `atlas_eligible=false`, dynamically omits and forbids that
model/method's three full seeds, and is rejected by the downstream eligibility
consumer. The formatted gate and gate-test SHA-256 values are
`2b7bcd627dd80c3d140588451cf767982bbaf21015f68a34e907ed05ab75bc6c`
and
`ca96c89e6627d031e82e1d0fe1346b38eae152f5a10e5a2e35332caf4766d5a6`.

The minimal mask-receipt producer and thin CLI SHA-256 values are
`de9324909af75014a0bf76420abf5a1904047891b12c5da595dc9c96836180c0`
and
`287844fa0e88061aa167cce1fa1233428c3289cd26123f76463c9eb88d17ce68`.
The durable receipt
`evidence/p0_2_mask_validation_20260721/mask_validation_receipt.json` has
SHA-256
`5966e274881984b2eeabeedd749d94c313fce02fb814911f0d1082ac3c3232db`.
It binds dictionary-module SHA-256
`347a095c2e18a429e09011f84a45bd40b1cf5d46ebf38398e6e2a7afe57c6596`
and test-file SHA-256
`a1f2f61f8933c1e1a766bc5e827ba3747215c2286a2df2421e32d0e2b8fc1253`,
records both exact required mask nodes as passed and validates through the
production consumer.

The 216-test plus 6-subtest result above remains the lifecycle-only snapshot.
The final integrated pre-documentation suite passed 223 tests plus 6 subtests
in 36.28 seconds; focused gate/mask verification passed 16 tests. Ruff check
and format, AST syntax, shell syntax and strict JSON validation passed. This
closes adjudication/mask prerequisites only: it performs no held-out test
evaluation, does not pass P0-2 and does not authorize a full run before its
terminal screening result validates.

### 2026-07-22 terminal screening, calibrated selection and full launch

The corrected r3 validation-only screen completed all 45 frozen candidates
with zero test accesses. Its selected configurations were ProGen2-medium gated
SAE candidate 0 (`l1=1e-5`, `aux=0.1`, L0 132.16376074074074, FVU
0.4397525937186076), ZymCTRL gated SAE candidate 3 (`l1=3e-5`, `aux=1`, L0
120.02304111111111, FVU 0.5929956473037449), and ZymCTRL ReLU/L1 SAE candidate
0 (`l1=1e-5`, `aux=0`, L0 131.2233588888889, FVU 0.5899043000429988); all use
threshold 0. ProGen2-medium ReLU/L1 SAE (zero eligible; closest L0
109.82516444444444), ProtGPT2 gated SAE (minimum/closest L0
1424.622846388889) and ProtGPT2 ReLU/L1 SAE (minimum/closest L0
1298.928968888889) are exact terminal `sparsity_match_failure` outcomes.
They are calibrated dictionary-quality negatives, not biological claims. The
terminal receipt is `evidence/p0_2_screening_20260722/terminal_receipt.json`,
SHA-256 `83bf4563a27886a1e1f148eb4094f9a7b02533936a02981caac17a805de04d6c`.

The fail-closed gate therefore authorizes exactly 27 full runs: three seeds for
ProtGPT2 TopK/dense, ZymCTRL TopK/ReLU-L1/gated/dense, and ProGen2-medium
TopK/gated/dense. The three failed pairs account for the nine forbidden seed
runs. After terminal-state and selection guards, cleanup removed 45 redundant
progress files and 42 nonselected best checkpoints, totalling
2,103,965,204,225 apparent bytes and 2,103,966,318,592 allocated bytes
(approximately 1.9135 TiB), while retaining three selected best checkpoints
and 108 compact evidence files. The local receipt
`evidence/h200_cleanup_20260722/cleanup_manifest.json` has SHA-256
`f646ad90abe8554c70357860285dda4051147fc0d6d6e95c0887f97fe62befe3`; the
remote receipt-manifest SHA-256 is
`607d7ae1420ad68a44c7a3e283f3f0c56fc55419d67fb83a99b57f57b06cef47`.

At approximately 11:46 CST, launcher PIDs 30587--30590 started the fresh full
r1 queues on GPUs 0--3 under `p0_2_dictionary_controls_bf16_r3/full`, in fixed
7/7/7/6 queues. Launcher `scripts/77_run_dictionary_full_queue_h200.sh` has
SHA-256 `20eadefc7eb8102e892efb5592e7f71b8763ca41fd3dd052a725cf4170849d72`.
Before launch it revalidates the exact cache and terminal screening lineage,
including each retained selected checkpoint. After each runner returns it
verifies the full result, manifest, run state, actual selected checkpoint and
`test_evaluation_count=1`; only then does it delete and log the redundant
`progress.pt`. The initial launch receipt
`evidence/p0_2_full_launch_20260722/launch_receipt.json` has SHA-256
`306d2b56bf3cf756fa5c56dd86e0f24d3555fc6c49c02d59747216e31d3104b9`.
By 12:45 CST, all four first runs had valid `in_progress` states and atomic
best/progress checkpoints at step 5,000. GPU memory was
60,793--60,825 MiB with active utilization, no queue-log error was present and
GPFS retained 48,192,404,062,208 bytes free.
The frozen budget is 1,500 aggregate GPU-hours; the
conservative maximum longest-queue ceiling is about 388.9 hours (16.2 days),
excluding evaluation/I/O. Terminal screening rates and the prior online TopK
rate place queue 3 at about 4.8 days before margin, supporting a provisional
5--7-calendar-day estimate. The local suite passed 223 tests plus 6 subtests
in 36.32 seconds and the focused reviewer suite passed 30 tests. No H200 issue
requires action. P0-2 remains open pending aggregation and adjudication of all
27 full runs.

## Ordered execution queue

1. Complete and validate the three immutable activation caches. **Completed
   2026-07-20.**
2. Run the frozen alternative-dictionary screenings without test-set access.
   **Completed 2026-07-22: 45/45 terminal validation candidates, zero test
   accesses.**
3. Execute all and only authorized full exact-cache model/method/seed runs and
   publish atomic `best.pt`, run-manifest and eligibility receipts. **Active:
   27 runs; the three calibrated sparse-method failures forbid nine seeds.**
4. Adjudicate P0-2 before exposing any downstream biological labels.
5. Build seed-specific P0-3/P0-4/P0-5/P0-6/P0-8 production inputs only through
   the public eligibility adapter; reject online `clt.pt` queues.
6. Freeze independent power plans, labels, matching calipers, effect margins,
   generation scorers and all analysis grids before their one-claim runs.
7. Run P0-3 and P0-4 from the same immutable A/B activation products; run P0-5
   counterfactual extraction and P0-8 representation extraction in parallel
   where resources permit.
8. Execute and score P0-6 once the positive-only plan and scorer receipt both
   validate. Run symmetric structure evaluation only on the resulting single
   experiment.
9. Retain P0-7's passed prospective synthetic receipt as pipeline calibration,
   then run the still-open held-out pretrained target dose/site surfaces and
   pass their immutable rows to the implemented adjudicator before interpreting
   target nulls as causal bounds.
10. Adjudicate without threshold changes, rebuild every affected display and
    source-data row, compile both manuscripts, and perform the final claim and
    provenance audit.
11. Request human metadata, coauthor approval, release tagging and DOI minting
    only after computational gates are resolved.

## Cross-cutting assessment items

- **Positioning:** use “three-model recurrence,” not
  “architecture-independent conservation.” The R/S/C/U benchmark is proposed
  until all prospective estimands are executed.
- **Model panel:** the pretrained panel does not factorially separate
  architecture, corpus and tokenizer. The prospective planted benchmark varies
  seeds and rotations within one synthetic architecture/corpus/vocabulary, so
  it calibrates sensitivity but does not supply the missing factorial
  decomposition. The manuscript scope therefore remains explicitly narrow.
- **Literature:** the manuscript now includes the recent formalization,
  component-semantic, sparse-identifiability, representation-uncertainty,
  controllable-design and open biological-foundation-model literature named by
  the assessment.
- **Biological claims:** Pfam, CLEAN, ESMFold and Foldseek remain correlated
  computational filters, not functional activity. Wet-lab validation is an
  optional higher-impact route and cannot be completed without external human
  coordination.
- **Negative claims:** failure to reject is never called equivalence. Clean
  negative language requires confidence intervals or TOST/ROPE against margins
  frozen before observation.
- **Numerical integrity:** every confirmatory model consumer is bfloat16-only,
  verifies observed floating parameter dtypes before its first forward and
  verifies all required captured activations, attentions and logits as finite
  before conversion, scoring or sampling. Fixture-only float32 paths cannot
  produce a production receipt.
- **Architecture:** new work stays in shared revision modules with thin CLI
  entry points. No parallel legacy pipeline or silent compatibility fallback is
  introduced.

## Submission boundary

The prospective synthetic pipeline-sensitivity subgate is passed. Submission
nevertheless remains blocked until every other applicable P0 acceptance
criterion is met, including the real held-out pretrained target/control gate,
all panels are regenerated from immutable manifests, and the external release
and author-approval steps are complete. Failed scientific gates may still support
a publishable negative-results paper, but only when the relevant measurement
system has passed its controls and the resulting effect bounds are reported
without reinterpretation.
