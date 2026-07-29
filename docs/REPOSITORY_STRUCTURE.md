# Repository Structure and Naming

## Live Root

InterpretabilityTransfer has one Git repository and one live research root: the repository root (`.`). Do not create nested Git repositories or nested `r<ID>_*` live roots.

Historical R0, R1, and R2 identifiers remain valid only as provenance labels in experiment IDs and frozen records. They do not define current directory boundaries.

| Path | Role |
|---|---|
| `src/transfer/` | Importable measurement library. |
| `scripts/transfer/` | Executable validation and campaign entry points. |
| `tests/` | Contract, invariant, negative-path, and end-to-end tests. |
| `docs/` | Live research and repository documentation. |
| `evidence/` | Compact, cited, reproducible receipts. |
| `external_resources/` | Tracked metadata and setup helpers; untracked payloads. |
| `data/`, `results/`, `logs/`, `wandb/` | Host-local inputs and generated state. |

Frozen historical provenance is external at `/Data2/lzp/bio_archive`; see `docs/ARCHIVE.md`.

Create only directories with an active consumer. Archive retired scope instead of retaining empty placeholders or compatibility aliases.

## Version Control

The root `.git/` is the only active Git database. The remote repository is named InterpretabilityTransfer; repository documentation and package metadata must use that name.

`.gitignore` protects host-local datasets, results, logs, credentials, environments, model files, and large binary artifacts. Generic `*.npz` files are ignored, with one narrow exception for compact receipts under `evidence/**/*.npz`.

Ignored files are not backups. Never run `git clean -fdx` or `git clean -fdX`; use `git clean -fd` or an explicit disposable path. A new tracked artifact must be small, cited, and placed under a documented boundary.

`.mutagenignore` is tracked and generic. It contains no endpoint, username, pod, secret, or machine-specific path. Synchronization policy excludes generated and private state while keeping source, documentation, metadata, and compact evidence portable.

## Storage Boundaries

| Content | Location | Policy |
|---|---|---|
| Source, tests, configuration, and docs | repository root | tracked |
| Compact evidence and checksums | `evidence/` | tracked |
| Resource descriptions and setup helpers | `external_resources/` | tracked |
| Downloaded models and tools | host storage or ignored `external_resources/` payload directories | untracked |
| Datasets | `data/` or external storage | untracked |
| Generated results | `results/` | untracked; publish separately when required |
| Runtime logs | `logs/` | untracked |
| Credentials | `.env.local` or external mode-600 configuration | untracked |
| Frozen history | `/Data2/lzp/bio_archive` | external, checksummed, never rewritten |

Do not place irreplaceable evidence under `results/` or `logs/`. Promote a compact receipt into `evidence/` with provenance to its generating command and source result.

## Naming

- Directories use lowercase snake case except untouched upstream package names.
- Canonical control documents use conventional uppercase names such as `README.md`, `RESEARCH_PLAN.md`, `EXPERIMENT_LOG.md`, `MANIFEST.tsv`, and `SHA256SUMS`.
- Dated documents use `<TOPIC>_YYYYMMDD.md`; dated runs use `<experiment_slug>_YYYYMMDD[_HHMM]/`.
- Python stage scripts use a unique `NN_action_object.py` prefix within `scripts/transfer/`.
- Operational entry points use descriptive names such as `run_transfer_h200.sh`.
- Generated figures use `figure_NN_subject.{pdf,png}` when they are part of a manuscript package.

Names describe stable purpose, not a journal target, temporary version, host, pod, result claim, or implementation incident.

## Paths And Provenance

- Live code derives the repository root at runtime and uses repository-relative paths where practical.
- Operator documents refer to the repository root as `.` or "the repository root"; they do not embed checkout locations.
- Machine, GPFS, OSS, model, and dataset paths are passed through environment variables and resolved at runtime.
- Pod names are disposable runtime values and must never be persisted in repository files, manifests, or durable logs.
- Generated metadata and frozen archive snapshots retain original execution paths because rewriting provenance is corruption.
- External access details belong in `~/hangzhou-remote`, whose README is authoritative for current connectivity and recovery.

## Clone Validation

A source clone is structurally valid when tracked files are present, `CLAUDE.md` and `AGENTS.md` are byte-identical, the JSON resource manifest parses, `.env.local` remains ignored, compact evidence NPZ files are allowed only under `evidence/`, and `python scripts/transfer/panel_contract.py --verify` passes in the declared environment.
