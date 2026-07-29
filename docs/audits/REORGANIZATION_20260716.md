# Repository reorganization — 2026-07-16

> Snapshot note: this is the first-stage cleanup record. Its statement that the three then-current roots were preserved was superseded later the same day by `STRUCTURE_STANDARDIZATION_20260716.md`, which records the coordinated same-depth R0/R1/R2 rename.

## Resulting live structure

```text
paper_r2_nature_mi/    Paper A: decoder/generative sparse-readout audit
benchmark_encoder/     Paper B: encoder interpretation benchmark
ProteinInterpret/      shared benchmark framework; future Paper C seed
data/                  shared large datasets (B-only)
external_resources/    shared third-party tools/data (B-only)
ops/                   cross-project operational scripts
project_records/       authoritative status, audit and chronological records
logs/                  shared runtime logs (ignored)
archive/               frozen provenance and export bundles
```

Hidden operational roots (`.claude/`, `.venv/`) remain at repository root.

## Changes

| Previous path/content | Action | Current location/rationale |
|---|---|---|
| `Research1/` | removed from both mirrors after duplicate verification | canonical live tree is `benchmark_encoder/` |
| `Research2/` | removed from both mirrors after duplicate verification | canonical live tree is `paper_r2_nature_mi/` |
| `manuscripts/` | removed after duplicate/superset verification | live Paper A manuscript is project-local; historical drafts are in `archive/manuscripts/` |
| `analysis_exports/` | moved on both mirrors | `archive/analysis_exports/`; frozen bundles, not a research direction |
| root `results/variant_effect/*` | exact/stale duplicates removed | canonical Paper B results remain project-local |
| root `results/final_plan_readiness/` | moved | `archive/project_records/final_plan_readiness_20260511/` |
| flat root runtime logs | categorized in place | `logs/paper_a/`, `logs/paper_b/` and `logs/shared_ops/` |
| Paper A pre-pivot proposal | archived | `archive/legacy/PAPER_A_PRE_PIVOT_DRUG_DESIGN_PROPOSAL_20260330.md` |
| Paper B pre-pivot proposal | archived | `archive/legacy/PAPER_B_PRE_PIVOT_MECHANISM_PROPOSAL_20260330.md` |
| live proposal paths | replaced | current bounded direction briefs at each project root |

The root `results/` directory disappeared after its remaining contents were classified. Large active outputs continue to use project-local generic `results/`, `logs/` and `wandb/` names so synchronization-ignore rules remain effective.

## Paths deliberately preserved

`paper_r2_nature_mi/`, `benchmark_encoder/` and `ProteinInterpret/` were not renamed or nested. Source and operational code relies on their current depth, including `parents[2]` repository resolution, result paths, remote runner paths and Mutagen ignore rules. The legacy `paper_r2_nature_mi` name now targets npj Artificial Intelligence; its name is a compatibility constraint, not the venue.

## Synchronization safeguards

`.mutagenignore` now excludes the three pre-2026-05-28 root names and both the old/new export-bundle paths. Mutagen was flushed, paused for coordinated B/D moves, resumed and flushed again. Both endpoints were checked after each move; the session returned to `Watching for changes` without conflicts.

## Provenance policy

No historical experiment record was rewritten to make old paths appear current. New work uses canonical project roots. `archive/` remains provenance-only; the new additions are frozen snapshots or superseded documents, not dependencies for live code.
