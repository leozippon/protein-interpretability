# Structure Standardization — 2026-07-16

## Canonical root migration

| Previous live root | Canonical root |
|--------------------|----------------|
| `ProteinInterpret/` | `r0_shared_interpretability_framework/` |
| `benchmark_encoder/` | `r1_encoder_interpretability_benchmark/` |
| `paper_r2_nature_mi/` | `` |
| `project_records/` | `docs/` |

The three research roots were moved at the same repository depth on both the B and D Mutagen endpoints. The two nested B-side Git repositories, B-only result and log payloads, and the D-side root worktree were preserved in place. Old root names are quarantined in `.mutagenignore` and `.gitignore` rather than retained as compatibility symlinks.

## Document normalization

- Research plans moved from each project root to `docs/RESEARCH_PLAN.md`.
- R1 technical and legacy notes moved under `docs/methods/` and `docs/legacy/`; its standalone InterPLM paper moved to `literature/background/`.
- R0's schema note moved from a one-file `framework/` directory to `docs/BENCHMARK_SCHEMA.md`.
- Compact R2 recoverability evidence moved to `r2.../evidence/`; the associated analysis moved to `r2.../docs/analysis/`.
- Manuscript adaptation/audit notes moved to `r2.../docs/manuscript/`; the publisher package is now under `manuscript/template/`.
- Repository status, navigation, log, and audits now live under root `docs/`.

## File/folder normalization

- Duplicate numeric script prefixes were removed: R1 H200 launchers now use descriptive names and the R2 figure builder is `50_make_manuscript_figures.py`.
- Manuscript figures use zero-padded, content-specific filenames.
- Source-data directories use zero-padded `figure_NN` / `table_sNN` names.
- Runtime log categories use research IDs instead of paper letters.

Historical result manifests, archived material, and external GPFS/OSS/H200 namespaces were deliberately not rewritten. They record where earlier runs actually executed and are not live repository aliases.

Verification covers mirror convergence, absence of retired live roots, Python and shell syntax, cross-project preflights/table generation, figure and source-data regeneration, literature/template/source-data hashes, LaTeX builds, nested Git worktrees, and live symlinks.

## Verification result

- Mutagen returned to `Watching for changes` with 112 synchronized directories and 561 files on each endpoint; retired live roots are absent on B and D.
- All 119 first-party Python files compile in memory, all first-party shell scripts pass `bash -n`, all five YAML configs parse in the `ct` environment, numeric script prefixes are unique, and the actionable stale-path scan is clean. One live vendor symlink remains valid.
- Both R0 preflights and `build_v0_benchmark_tables.py` completed with canonical source paths.
- All six renamed manuscript figures regenerated. The source-data builder verified 58 manifest rows and 61 checksums (1,412,592 package bytes).
- All 20 literature PDFs and all retained template checksums pass.
- Tectonic rebuilt the 18-page main Article and 11-page supplement without errors; only benign underfull-box warnings remain.
- The live credential-pattern scan is clean. Existing unrelated dirty worktree changes in R1 and R2 were preserved.
