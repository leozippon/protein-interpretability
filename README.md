# InterpretabilityTransfer

InterpretabilityTransfer studies how mechanistic-interpretability methods transfer from text decoders to protein generative models.

## Research Objective

The programme follows three ordered directions:

1. **Compare model families.** Identify meaningful differences between text and protein generative models.
2. **Evaluate method transfer.** Determine how and under what conditions existing interpretability methods transfer, and separate method limitations from model or data limitations.
3. **Develop adapted methods.** Design and validate protein-specific methods only when the preceding evidence identifies a concrete failure mode.

Steps 2 and 3 are the main deliverables; step 1 provides their foundation. `docs/INTERPRETABILITY_TRANSFER_AUDIT.md` is canonical for findings, limitations, retractions, and the current scientific plan.

## Repository Layout

The repository root (`.`) is the only live research root.

| Path | Purpose | Git policy |
|---|---|---|
| `src/transfer/` | Shared measurement library | tracked |
| `scripts/transfer/` | Local validation and H200 campaign entry points | tracked |
| `tests/` | Contract and behavior tests | tracked |
| `docs/` | Scientific plan, logs, methods, analyses, and navigation | tracked |
| `evidence/` | Compact receipts and provenance artifacts | tracked |
| `external_resources/` | Metadata and setup helpers for third-party resources | metadata tracked; payloads ignored |
| `data/` | Local datasets | ignored |
| `results/` | Generated experiment outputs | ignored |
| `logs/` | Runtime output | ignored except `logs/README.md` |

Frozen historical provenance, retired R0/R1 roots, old manuscripts, and retired configurations are stored outside the repository at `/Data2/lzp/bio_archive`. See `docs/ARCHIVE.md`; nothing in that tree is a live interface.

## Campaign Contract

The current contract contains 11 active model arms and 11 active stages. `scripts/transfer/panel_contract.py` is the source of truth; generated shell declarations and operator documentation must agree with it.

Active arms: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `dialogpt-small`, `qwen2.5-0.5b`, `llama-3.2-3b`, `protgpt2`, `zymctrl`, `progen2-base`, and `progen2-medium`.

Active stages: `cohort_power`, `pathway_budget`, `estimand_power`, `circuit_primitives`, `relational_channel`, `explanation_channel`, `convergence_control`, `lens_family`, `probe_and_erasure`, `homology_control`, and `induction_path_patching`.

Validate the generated contract before scheduling:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ct
python scripts/transfer/panel_contract.py --verify
```

See `scripts/transfer/README.md` for stage eligibility, local validation, and H200 operation.

## Environment

The validated workstation uses Python 3.11. Direct Python dependencies are declared in `requirements.txt`; the correct CUDA-enabled PyTorch build must be selected for the host. Copy `.env.local.example` to the ignored `.env.local` only when local download credentials are needed.

The resource interface is recorded in `external_resources/manifests/interpretability_transfer_resources.json`. It records environment-variable contracts, not machine-specific paths, credentials, pod names, or claims about current availability.

## Documents

1. `docs/INTERPRETABILITY_TRANSFER_AUDIT.md` - canonical scientific findings, limitations, and plan.
2. `docs/RESEARCH_PLAN.md` - research scope and evidence discipline.
3. `docs/EXPERIMENT_LOG.md` - chronological experiment record.
4. `docs/PROJECT_LOG.md` - repository and operations chronology.
5. `docs/REPOSITORY_STRUCTURE.md` - repository, naming, storage, and provenance rules.
6. `docs/DOCUMENT_INDEX.md` - navigation for live and frozen material.
7. `CLAUDE.md` and `AGENTS.md` - identical agent and operator instructions.

`check.md` is frozen and superseded by the canonical audit. Do not use it as the current claim source.

## Storage Safety

Ignored data, results, logs, models, and caches are not protected by Git. Never run `git clean -fdx` or `git clean -fdX`; use `git clean -fd` or an explicit disposable path. Keep compact, cited evidence under `evidence/`, where small causal receipt matrices are explicitly permitted even though generic `*.npz` files remain ignored elsewhere.

Cluster access and recovery are documented outside the repository in `~/hangzhou-remote/README.md`. Query live status before scheduling; never persist disposable pod names in repository files or durable logs.
