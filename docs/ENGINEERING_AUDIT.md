# Engineering Audit

## Scope

The 2026-07-29 audit covered repository structure, runtime contracts, H200 orchestration, statistical helpers, stage logic, evidence, documentation, remote storage, and secret boundaries under the five Development Principles in `CLAUDE.md`.

## Repaired

- Unified the live programme under `src/transfer`, `scripts/transfer`, and one generated panel contract; retired project roots and configuration sets were moved to the external archive.
- Replaced legacy `R2_*` runtime resource names with `TRANSFER_*`, removed checkout-specific path defaults, and moved new H200 packages, results, logs, caches, and generated homology resources under the InterpretabilityTransfer GPFS root.
- Corrected ProGen architecture handling and path patching, tie-aware percentiles, parent split provenance, grouped bootstrap intervals, derived-statistic bootstrap intervals, and finite-draw guards.
- Split ProGen2-base and ProGen2-medium cohort runs so the measured float32 override applies only to medium.
- Made the H200 controller derive its repository root, freeze the transitive runtime source set, verify snapshots locally and remotely, publish invocation manifests by content hash, reject duplicate or unknown arguments, and fail campaigns that skipped required data.
- Made the homology preflight validate only immutable inputs; DIAMOND extraction, database, and scratch locations are generated outputs.
- Made the transfer-gap summarizer strict by default and added explicit partial-mode behavior.
- Preserved compact scientific and checkpoint receipts with SHA-256 verification before remote checkpoint cleanup.

## Remote Storage

Intermediate checkpoints and optimizer-only recovery state were removed after final checkpoint hashes were preserved. The project CLT weights on OSS fell from about 2.85 TB to 90 GB; GPFS dictionary trees were reduced while retaining final weights, configurations, manifests, the 27 controls, exact cache, results, logs, and frozen code. Post-clean inventories are in `evidence/checkpoint_receipts_20260729/`. The separate `GlobalRAG` tree was not modified.

The idle four-GPU project pod and the temporary zero-GPU transfer pod were deleted. The cluster subsequently reported 12 of 16 H200s allocated, leaving four schedulable.

## Accepted Limitations

- GPFS snapshot publication has no distributed lease. Concurrent controllers must use distinct run IDs; a correct lock requires ownership and expiry semantics rather than a local exception.
- New code snapshots are content-addressed, but model and corpus trees are still path-identified rather than fully content-digested. Old artifacts retain the provenance they recorded at execution time.
- The H200 cluster has no outbound network and depends on a running GPFS-mounted pod for end-to-end transfer validation.
- Jump-host connectivity and persistence are verified, but no broad Windows filesystem deletion was attempted without a bounded ownership manifest.
- Historical experiment records retain obsolete paths and names because rewriting execution provenance would be incorrect.

## Verification

The final live tree passed 205 tests plus 9 subtests, Ruff, generated-panel verification, Shell syntax checks, JSON parsing, document identity, `git diff --check`, tracked-secret scanning, and both evidence checksum manifests.
