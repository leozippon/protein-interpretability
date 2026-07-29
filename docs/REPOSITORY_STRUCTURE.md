# Repository Structure and Naming

## Canonical research-root format

Every live research or shared-framework root uses:

```text
r<research ID>_<lower_snake_case_research_content>/
```

Current assignments:

| ID | Canonical root | Status |
|----|----------------|--------|
| R2 | `` | **the only live research root** — text-to-protein interpretability transfer |
| R0 | `archive/retired_research_roots/r0_shared_interpretability_framework/` | retired 2026-07-29 |
| R1 | `archive/retired_research_roots/r1_encoder_interpretability_benchmark/` | retired 2026-07-29 |
| R3 | reserved | not assigned |

IDs are stable and are **not reused** — a retired root keeps its ID inside `archive/` so that historical references remain resolvable. A directory name describes scientific scope and must not embed a journal, paper letter, transient method version, or result claim. Live research roots remain one level below the repository root because path-sensitive scripts derive the repository root from that depth; retired roots sit two levels deeper and their scripts are not expected to run.

Operational roots are exempt from research IDs: `data/`, `external_resources/`, `ops/`, `logs/`, `docs/`, and `archive/`.

## Version control

**One git repository, at the repository root.** Until 2026-07-29 there were two nested repositories (`r1_.../.git` and `r2_.../.git`), which fragmented history and meant the root could not manage the roots it contained. Both were archived to `archive/legacy/nested_git_history_20260729/` with their histories and remote configuration intact; only the `.git` metadata moved, no working-tree file was touched. Do not create a nested repository below the root again.

The root `.gitignore` carries a warning that must not be removed: results trees are ignored, so `git clean -fdx` from the root deletes every experiment result in the repository — currently ~47 GB under R2, ~3.5 GB under `archive/retired_research_roots/` and ~230 MB under `archive/legacy/`. This has already destroyed completed artefacts three times.

The ignore file names each canonical root explicitly beside the generic `results/`, `logs/` and `wandb/` rules, so that a root rename shows up as a stale line rather than as silently un-ignored data. Retired live-root names are quarantined there to prevent accidental resurrection; that block must contain directory patterns only. A bare filename in it will match a live document at any depth — `DOCUMENT_INDEX.md` sat there from 2026-07-16 to 2026-07-29 and silently excluded `docs/DOCUMENT_INDEX.md` from version control.

## Standard research layout

Use only the folders a direction actually needs, chosen from this vocabulary. A folder that exists but is empty, or that holds material for a scope that has been retired, is a navigation defect — archive it rather than keeping it as a placeholder.

```text
rN_research_content/
├── README.md
├── configs/
├── docs/
│   ├── README.md
│   ├── RESEARCH_PLAN.md
│   ├── EXPERIMENT_LOG.md
│   ├── methods/
│   └── analysis/
├── evidence/          # compact synchronized receipts only
├── scripts/
├── src/
├── tests/
├── results/           # generated/B-only, ignored by sync and by git
└── logs/              # runtime/B-only, ignored by sync and by git
```

Additional folders are legitimate when a direction needs them — `manuscript/`, `literature/`, `preregistration/`, `external/` — but each must have a live consumer. R2 carried all three of the first until 2026-07-29, when they were archived with the scope they served.

Keep generic output names `results/`, `logs/`, and `wandb/` at any depth; sync and ignore policies depend on them. Do not put irreplaceable synchronized evidence under `results/`; use `evidence/` with a provenance manifest.

`logs/` holds runtime output and is not evidence. A log is kept only while a receipt or document cites it; otherwise it is archived or removed. Never store a copy of a result tree under `logs/` — it will be read as a measurement.

## Naming rules

- Folders use lowercase snake case, except untouched upstream vendor/package directories.
- Canonical control documents use conventional uppercase names such as `README.md`, `RESEARCH_PLAN.md`, `EXPERIMENT_LOG.md`, `MANIFEST.tsv`, and `SHA256SUMS`.
- Dated documents use `<TOPIC>_YYYYMMDD.md`; dated runs use `<experiment_slug>_YYYYMMDD[_HHMM]/`.
- Python analysis scripts use a unique `NN_action_object.py` prefix within one project. Operational entry points may use descriptive unnumbered names such as `submit_h200_sae.sh`.
- Submission figures use `figure_NN_subject.{pdf,png}` and source-data folders use zero-padded identifiers such as `table_s01_model_quality/`.

## Path and provenance rules

- New code uses repository-relative paths derived from `Path(__file__)` where practical; do not introduce old root aliases or compatibility symlinks.
- External H200/GPFS/OSS paths may retain historical names until the external store is explicitly migrated. Parameterize them and label them as external.
- Generated result metadata and frozen `archive/` snapshots retain their original execution paths. The live migration map belongs in a dated audit, not in rewritten historical artifacts.
- A root rename must be coordinated on both Mutagen endpoints because `.git/`, large results, and logs are not synchronized.
