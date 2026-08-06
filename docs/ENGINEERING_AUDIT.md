# Engineering Audit

## Scope

The 2026-07-29 audit covered repository structure, runtime contracts, H200 orchestration, statistical helpers, stage logic, evidence, documentation, remote storage, and secret boundaries under the five Development Principles in `CLAUDE.md`.

The 2026-07-30 audit (EXP-R2-068) re-covered the live code in four regions — the H200 controller and worker, `src/transfer`, `scripts/transfer`, and `scripts/transfer_gap` — and is recorded below beside it. Its scientific consequences are in `docs/INTERPRETABILITY_TRANSFER_AUDIT.md`; this file records engineering state only.

## Repaired, 2026-07-29

- Unified the live programme under `src/transfer`, `scripts/transfer`, and one generated panel contract; retired project roots and configuration sets were moved to the external archive.
- Replaced legacy `R2_*` runtime resource names with `TRANSFER_*`, removed checkout-specific path defaults, and moved new H200 packages, results, logs, caches, and generated homology resources under the InterpretabilityTransfer GPFS root.
- Corrected ProGen architecture handling and path patching, tie-aware percentiles, parent split provenance, grouped bootstrap intervals, derived-statistic bootstrap intervals, and finite-draw guards.
- Split ProGen2-base and ProGen2-medium cohort runs so the measured float32 override applies only to medium.
- Made the H200 controller derive its repository root, freeze the transitive runtime source set, verify snapshots locally and remotely, publish invocation manifests by content hash, reject duplicate or unknown arguments, and fail campaigns that skipped required data.
- Made the homology preflight validate only immutable inputs; DIAMOND extraction, database, and scratch locations are generated outputs.
- Made the transfer-gap summarizer strict by default and added explicit partial-mode behavior.
- Preserved compact scientific and checkpoint receipts with SHA-256 verification before remote checkpoint cleanup.

## Repaired, 2026-07-30 (EXP-R2-068)

### Remote execution — the campaign channel could not report failure

The access layer returns 0 whatever the remote command exits with. Measured: `h200_pod_exec.sh -- bash -c "exit 7"` returns 0, and `h200_pod_bash.sh "exit 5"` returns 0. Three call sites depended on that status and therefore could not fail:

- the worker invocation, so a preflight refusal that scheduled no GPU reported `campaign complete`, exit 0;
- `verify_remote_snapshot`, so the remote half of the code-freeze guarantee was never checked;
- `push_run_manifest`, which always took its "already present and verified" branch — **no invocation manifest had ever been pushed**, and every run directory on GPFS held `INVOCATIONS=0`.

Repaired inside the repository, because the access layer is external and shared: the worker states its own status on its last line (`TRANSFER_WORKER_EXIT=`, declared once and read out of the worker source by the controller), and every remote predicate now answers on stdout through one `pod_predicate` helper that refuses an unrecognised reply. A missing sentinel is itself a failure, which covers a killed worker and a dropped tunnel. Verified end-to-end against the live cluster in both directions.

### Panel contract — a declaration that verified on one host and not the other

The arm-to-environment-variable map was inferred by comparing resolved paths. The pod sets `TRANSFER_TEXT_MODEL_BASE_DIR="${TRANSFER_MODEL_BASE_DIR}"` because every checkpoint sits in one GPFS directory, so the comparison aliased and six of seven text arms classified as protein-root arms *inside the pod only*. `ArmSpec.path_variable` now declares the choice where the path is made; the consistency check runs only when the constants are distinct, because that is the only situation in which there is anything to check.

### Campaign accounting

A requested stage that measured nothing, and a stage that was never dispatched, both reported success. Both are now recorded and fail the campaign. `estimand_power`'s aggregation aborted the worker silently when a `measure` output was absent; it now refuses explicitly. An EXIT trap preserves the ledger on every early-exit path. `TEXT_ARM` is validated against the contract before the freeze. Checkpoint preflight is per checkpoint rather than per models-root.

### Cohort construction

Every campaign stage now draws its corpus under a declared seed (`arms.DEFAULT_CORPUS_DRAW_SEED`, one declaration); previously the corpus-level pool was a head-of-file prefix and only the within-pool subsample was seeded. `subsample_cohort` carries its parent's sampling record. Measured on the qualifying band: a 400-record file-order draw holds 342 distinct sequences against 398 under a seeded draw, and the two draws share 3 records.

### Measurement libraries

Bootstrap unit floors unified in `statistics` and applied where they were missing; three interventions that had a negative control and no positive control were given one or had the unfalsifiable guard removed with the reason stated; a per-row linearity check replaced a batch-aggregate norm ratio; the held-out unigram baseline's smoothing bias is now measured, reported and swept rather than asserted to be conservative; a FASTA record-count off-by-one at chunk boundaries was fixed (it had not fired on the corpus in use, which was verified rather than assumed). `activation_patching` runs chunked forward passes, verified numerically identical to the single-batch form.

### TG series

The retracted all-position residual spectrum is no longer the default-named field; seeded cohorts are consumed in seeded order rather than corpus-prefix order; a hard-coded host path and a stale constant imported from a retracted run were removed; stage eligibility moved from arm-name literals into the contract; every stage writes its cohort band.

## Repaired, 2026-08-04 (EXP-R2-124)

Audit of the PAA census and the path-patching comparison, scope frozen to
`prediction_addressed.py`, `path_patching.py`, `circuits.layout_token_ids` and
their tests. Scientific consequences are in `docs/INTERPRETABILITY_TRANSFER_AUDIT.md`.

- **A guard that covered one of the two paths it had to cover.** EXP-R2-118 barred
  a *predicted* layout token from the PAA candidate loop and left the decoy
  window open. `paa_specific_matched` subtracts `decoy_mean × n_keys`, so a decoy
  is the subtrahend of every head's score — the argument the module already makes
  for excluding position 0. Measured on the repaired pools: 4.30% of ProtGPT2's
  decoy keys and 2.74% of gpt2-large's were the FASTA wrap, reaching 16.1% and
  9.2% of instances, with antecedents and predicted tokens clean at 0. Barred, and
  the excluded vocabulary is now published for both paths.
- **A repair protected by a `grep`.** The layout guard's three tests were a stub
  unit test, a source-text assertion and a cascade-closure test on a fixture with
  no layout tokens. Disabling the guard outright left the suite green. Replaced
  by a behavioural test on a fixture that carries one, verified to fail under both
  mutations.
- **A gate verdict written by one line and read by nothing.**
  `a1_candidate_pool.verdict` reached FAIL in eight of 61 retained census
  artefacts and six of those went on to produce a `causal.json` in the same
  invocation. One declaration now serves the writer and the enforcer, and the
  causal stage refuses rather than recording and continuing.
- **`_depth_controlled` verified**, against a closed-form partial Spearman and an
  independent per-layer loop on every multidraw artefact; agreement 0.00e+00 to
  5.6e-16. Two unreachable edges recorded, not repaired.

Accepted, not repaired ~~: the D2.c census-to-causal rank correlation still has no
versioned implementation, so its published values come from throwaway code and
defensible reductions of the per-sequence census matrix disagree by up to 0.05 on
ProtGPT2. Recorded as the next instrument change owed.~~ **Discharged the same
day**: the statistic is `prediction_addressed.census_causal_agreement`, emitted
into each run's own `paa_gate_report.json`, and the 720-versus-712 grid
convention was measured rather than argued (median 0.0070, max 0.0249 over 35
runs). One consequence is recorded here because it affects how a panel is read:
artefacts written **before** that function existed carry no
`census_causal_agreement` key, so a reader must recompute them from
`census_matrices.npz` and `causal.json` through the module rather than treat them
as missing. On the D2.c panel that is every arm-draw at draws 20260728 and
20260801.

## Repaired, 2026-08-06

Audit of the H200 orchestration pair, scope frozen to `run_transfer_h200.sh`,
`h200_worker.sh` and their tests. Scientific consequences: none — these are
scheduling and provenance defects, and none of them altered a measured number.

- **A repair left half-applied, in the shape it was repaired from.**
  `verify_commands_buildable` used to decide a stage's item space from a list of
  stage names, and a stage absent from that list fell through to a catch-all and
  was built with the literal item `panel` as though it were an arm. That was
  repaired to read `TRANSFER_STAGE_SCOPE`; `arms_for_item`, whose consumer is the
  data-path preflight, was left with the identical shape. A panel-wide stage
  misread as per-arm resolves model variables for an arm named `panel`, finds
  none, and passes having checked nothing. Now dispatches on the declared scope,
  refuses a stage with no declared scope, and keeps `cohort_power` as the one
  named exception because its item space is separately declared as
  `TRANSFER_COHORT_ITEM_ARMS`. Six tests, including the negative path and a
  source-level pin that the dispatch is not a stage-name list.
- **`--dry-run` described a command the real path did not send.** The controller
  built `POD_COMMAND` for the dry run and re-listed every flag inside
  `invoke_worker`; nothing held them in agreement. `invoke_worker` now runs
  `POD_COMMAND`, so the array printed is the array sent. Two tests pin it.
- **A guarantee stated more broadly than it is delivered.** The redaction comment
  claimed `redact` is "applied to everything this script emits"; it is applied to
  the log lines and to the worker stream, and the fatal `echo … >&2` diagnostics
  bypass it. The claim is corrected to what the code does and the residual
  exposure is recorded below rather than papered over.
- **A comment pointing at a function that does not exist.**
  `verify_entry_points_importable_selftest` is referenced and was never written.
  The reference is replaced by the reason no such self-test can exist: the
  contamination class it named is only constructible when two entry points share
  one interpreter, which the subprocess isolation removed.

Reported and adjudicated as **not** defects, with the evidence:

- `collect_stage_args` builds its allow-list from the whole contract rather than
  the requested `ARMS`, so an `ARGS_<STAGE>__<ARM>` for an arm this invocation
  does not run is accepted and never applied. The dangerous case — a *misspelt*
  arm — is already refused with exit 2, because the misspelling is not in the
  allow-list at all. The remaining case is a declared override for an arm that is
  simply not scheduled, which is inert.
- `git rev-parse HEAD … || echo unknown` records `unknown` in the run manifest
  rather than refusing. The authoritative provenance is `CODE_HASH`, the content
  hash of the frozen snapshot, which cannot be `unknown`; the git revision is
  supplementary and recording it as unknown is honest rather than false.

## Accepted Limitations added 2026-08-06

- Fatal `echo … >&2` diagnostics in the controller are not redacted. They report a
  missing local path, an unreadable tool or an unknown argument, so they carry a
  pod name only if an operator has named a path after a pod. Bounded by operator
  naming, not by the code.
- `tests/test_path_patching_statistics.py` skips two tests that pin published
  numbers when the host lacks `results/transfer_20260730/d2bprime/`. Those
  artefacts are outside Git by policy, so the pin is host-local by construction.
  pytest reports the skip, so it is visible rather than silent.
- The controller pulls no results. A campaign that has produced its artefact and
  one whose artefact has been read are different states and the repository has no
  representation of the difference — on 2026-08-06 that left fourteen completed
  arm-draws on GPFS, unread, four of them on the arm carrying a live claim. The
  fix is not a bigger script: it is that a reading step must be part of a
  campaign's definition of done. **Half of it is now cheap**: the agreement
  statistic lives in each run's own ~1 MB report, so
  `scripts/transfer/read_paa_panel.py` reads a whole panel without moving the
  ~20 MB of matrices per arm-draw that made pulling the default. It pools only
  artefacts sharing the declared condition and states what it dropped, and it
  takes the statistic from the module rather than computing one beside it. The
  other half — a campaign that is not "done" until it has been read — is still
  convention.

## Remote Storage

Intermediate checkpoints and optimizer-only recovery state were removed after final checkpoint hashes were preserved. The project CLT weights on OSS fell from about 2.85 TB to 90 GB; GPFS dictionary trees were reduced while retaining final weights, configurations, manifests, the 27 controls, exact cache, results, logs, and frozen code. Post-clean inventories are in `evidence/checkpoint_receipts_20260729/`. The separate `GlobalRAG` tree was not modified.

## Accepted Limitations

- GPFS snapshot publication has no distributed lease. Concurrent controllers must use distinct run IDs; a correct lock requires ownership and expiry semantics rather than a local exception.
- Snapshot publication is not atomic: an interrupted push leaves a partial directory that fails verification permanently for that run id. A fresh launch mints a new id, so the failure is survivable, but the debris is invisible.
- New code snapshots are content-addressed, but model and corpus trees are still path-identified rather than fully content-digested. Old artifacts retain the provenance they recorded at execution time.
- `cohort_power`'s text item scores all seven text arms in one process, so one missing text checkpoint skips all seven. The skip is now logged and contract-derived rather than a mid-run exception, but its granularity is the item.
- Panel-wide stages write fixed filenames into a shared results root, so a narrowed re-run overwrites a full-panel artifact. Point `GPFS_RESULTS_ROOT` at a run-scoped path for a narrowed run; the resume provenance record prevents a false skip but not an overwrite.
- The H200 cluster has no outbound network and depends on a running GPFS-mounted pod for end-to-end transfer validation. The tunnel drops on runs of tens of minutes; long panel-wide stages should be split into per-arm campaigns.
- `budget.markov_cross_entropy_bits` enforces held-out status by exact string identity. Near-clonal leakage is now measured and reported rather than removed; removing it needs alignment-based clustering, which that module has no business owning.
- Four TG stages report cross-entropy over all scored positions only. The README names the four that hold the alphabet-bearing accounting rather than claiming all of them do.
- Historical experiment records retain obsolete paths and names because rewriting execution provenance would be incorrect.

## Verification

The 2026-07-29 tree passed 205 tests plus 9 subtests. The 2026-07-30 tree passes **303 tests plus 34 subtests**, Ruff over `src`, `scripts` and `tests`, generated-panel verification (including under aliased environment variables), the TG stage contract, Shell syntax checks on both orchestration scripts, and live cluster checks of the worker-status and remote-predicate paths in both directions.

### The census entry point's selective default (2026-08-06, same audit)

`--causal-heads` defaulted to **16** and `--control-offset` to **120**, which is a
*selective* census: the causal stage scores the census's own top 16 heads, and
`census_causal_agreement` then refuses the result because a correlation over a
census-selected subset reproduces the census's own ranking (standing rule 24). An
omitted flag therefore cost a GPU and answered nothing, while writing artefacts
that look like every other run's.

The consequence was not the wasted run but the duplication it forced. Every
campaign invocation had to supply the exhaustive count by hand, and **nine driver
scripts grew the same per-arm table** — gpt2-xl 1192, gpt2-large and ProtGPT2 712,
llama-3.2-3b 664, ProGen2-base and -medium 424, gpt2-medium 376, qwen2.5-0.5b 328,
ProGen2-small 184, gpt2 and dialogpt-small 136. Every entry is
`n_layer * n_head - control_heads`, a quantity the entry point can compute from
the arm and a driver cannot check. Verified against all twelve: the derived value
reproduces every hand-maintained entry exactly, including the newly admitted
byte-level arm at 184.

Both defaults are now `None`, resolved from the grid once the arm's head count is
known, with the two blocks partitioning the grid exactly and an over-large request
refused rather than truncated. An explicit value still works and is still recorded
in `settings`, so a deliberately selective run remains available and remains
visible in the artefact. Four tests, including a source-level pin that the default
is a sentinel rather than a small literal.

*Not changed, and the reason:* `--width` defaults to 512 while the contract
declares `PAA_CENSUS_WIDTH = 192`. That default is not a silent-wrong-answer path
— at width 512 ProtGPT2 admits no cohort rows at all and the run fails its
`--min-sequences` gate explicitly — so it is a footgun rather than a defect, and
the contract already passes the right value for every campaign item.
