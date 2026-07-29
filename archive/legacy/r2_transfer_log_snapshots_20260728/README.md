# Superseded intermediate copies of live transfer results (2026-07-28)

**Archived:** 2026-07-29 (EXP-R2-065)
**Source:** `r2_interpretability_transfer/logs/`

## What this is, and what it is not

This is **not** retired scope. Every file here is a point-in-time copy of a tree
that is still live: `r2_interpretability_transfer/results/transfer_20260728/`.
The copies were taken during the 2026-07-28 campaign as a safety net after
`git clean` destroyed the results tree three times (Appendix B rule 9). They are
filed under `archive/` rather than deleted because they are result artefacts and
this project does not delete result artefacts.

They are **not evidence**: no receipt, manifest or document cites them, and they
carry no provenance of their own beyond a wall-clock directory name.

## Why they are not simple duplicates

Checked file by file against the current `results/transfer_20260728/`:

| Snapshot | Files | Byte-identical to current | Differing | Absent from current |
|---|---:|---:|---:|---:|
| `results_snapshot_035618` | 15 | 0 | 15 | 0 |
| `results_snapshot_041710` | 23 | 9 | 14 | 0 |
| `results_snapshot_042202` | 25 | 11 | 14 | 0 |
| `results_snapshot_043205` | 28 | 14 | 14 | 0 |
| `results_snapshot_044550` | 32 | 18 | 14 | 0 |
| `results_snapshot_044900` | 32 | 18 | 14 | 0 |
| `results_snapshot_092911` | 32 | 18 | 14 | 0 |
| `results_snapshot_095812` | 32 | 18 | 14 | 0 |
| `results_snapshot_101413` | 32 | 21 | 11 | 0 |
| `results_snapshot_102447` | 35 | 25 | 10 | 0 |
| `results_snapshot_102641` | 35 | 25 | 10 | 0 |
| `results_snapshot_105931` | 35 | 28 | 7 | 0 |
| `results_snapshot_111428` | 38 | 38 | 0 | 0 |
| `results_snapshot_144042` | 62 | 54 | 8 | 0 |
| `results_snapshot_172758` | 67 | 67 | 0 | 0 |
| `results_snapshot_181849` | 72 | 72 | 0 | 0 |
| `convergence_control_backup` | 11 | 11 | 0 | 0 |

No snapshot contains a file that is missing from the live tree, so nothing here
is uniquely recoverable by path. The "differing" files are earlier states that a
later re-run overwrote — mostly the stages that were re-measured after the
cohort and rendering corrections of EXP-R2-059 through EXP-R2-062. Read them as
a record of *what a number used to be*, never as a current measurement.

Total 103 MB.

## If you need one

The live tree is authoritative. A snapshot is only useful to answer "what did
stage X read before the correction on 2026-07-28?", and even then the corrected
value and its reason are in `docs/EXPERIMENT_LOG.md` and
`docs/INTERPRETABILITY_TRANSFER_AUDIT.md`, which are the citable sources.
