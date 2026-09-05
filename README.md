# InterpretabilityTransfer

InterpretabilityTransfer compares pure-text, pure-protein, and joint language–protein generative models, audits whether interpretability methods remain faithful across them, and develops adapted methods only after a reproducible failure is identified.

## Start Here

| Document | Purpose |
|---|---|
| [`summary.md`](summary.md) | Plain-language research direction, hypotheses, progress, and current conclusions |
| [`docs/INTERPRETABILITY_TRANSFER_AUDIT.md`](docs/INTERPRETABILITY_TRANSFER_AUDIT.md) | Canonical findings, limitations, retractions, evidence boundaries, and scientific decisions |
| [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) | Executable comparison guide and stage inventory derived from the canonical plan |
| [`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md) | Estimands and metrics for the registered foundational stages |
| [`scripts/transfer/README.md`](scripts/transfer/README.md) | Local validation and H200 campaign operation |
| [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) | Append-only experiment chronology |
| [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md) | Append-only repository and operations chronology |
| [`AGENTS.md`](AGENTS.md) | Research, development, documentation, and compute rules |

The summary is the reader entry point, not a second source of scientific claims. The audit controls claim status, evidence boundaries, and the current scientific plan; the research plan translates admitted work into executable comparisons and stages.

## Development Workflow

1. Read `summary.md` to identify the research question being advanced.
2. Check the audit before reusing a result, method, statistic, or closed direction.
3. Follow the research plan's smallest identifying comparison instead of expanding every method across every model.
4. Treat `scripts/transfer/panel_contract.py` as the executable source of truth for registered arms and stages. Print and verify it with:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ct
python scripts/transfer/panel_contract.py --json
python scripts/transfer/panel_contract.py --verify
```

5. Validate changed code on B, run full campaigns on H200, and append admitted results to `docs/EXPERIMENT_LOG.md`. Update the audit before promoting a result into a claim.

## Repository Layout

| Path | Purpose | Git policy |
|---|---|---|
| `src/transfer/` | Shared measurement library | tracked |
| `scripts/transfer/` | Validation and campaign entry points | tracked |
| `tests/` | Contracts, negative paths, and end-to-end checks | tracked |
| `docs/` | Research authority, plan, logs, and technical records | tracked |
| `evidence/` | Compact cited receipts and provenance | tracked |
| `external_resources/` | Resource metadata and setup helpers | metadata tracked; payloads ignored |
| `data/`, `results/`, `logs/`, `wandb/` | Local inputs and generated state | ignored |

The repository root is the only live research root. Compact evidence stays with its receipts under `evidence/`; obsolete audits and staging reports remain recoverable from Git history. Frozen retired work is stored in a checksummed external archive and is immutable rather than a runtime dependency.

## Runtime and Storage

The validated environment and H200 access rules are maintained in `AGENTS.md`; live cluster allocation must be checked rather than copied into documentation. External resource variables are declared in `external_resources/manifests/interpretability_transfer_resources.json`.

Ignored files are not protected by Git. Never use `git clean -fdx` or `git clean -fdX`. Do not delete result trees, logs, checkpoints, datasets, or frozen provenance as though they were caches; only explicitly verified disposable paths may be removed.

## License

The project's original software is available under the [MIT license](LICENSE.md). Third-party code and manuscript support files retain their upstream terms. Model weights, databases, manuscript prose, figures and scientific data are outside the software license unless a separate release states otherwise; see [licensing scope and third-party notices](THIRD_PARTY_NOTICES.md).
