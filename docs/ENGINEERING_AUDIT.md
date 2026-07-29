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
