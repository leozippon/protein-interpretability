# Document Index

Navigation map as of the 2026-07-29 reorganisation: one objective, one live research root, everything else frozen under `archive/`.

## Start here

| File | Use |
|------|-----|
| `docs/INTERPRETABILITY_TRANSFER_AUDIT.md` | **Canonical.** Findings, retractions (§0.05, §0.1), qualifications (§5.05), limitation catalogue L1–L19, method-family coverage, plan, and Appendix B's fourteen standing methodological rules. Where any other document disagrees with it, it wins. |
| `README.md` | Objective, research roots, directory map, where the evidence is. |
| `README.md` | Live research root: position, panel, layout, evidence pointers, how to run a campaign. |
| `CLAUDE.md` / `AGENTS.md` | Environment, hardware, conventions, operational rules. |

## Live repository-wide documents

| File | Use |
|------|-----|
| `docs/PROJECT_STATUS.md` | Current state, followed by superseded status snapshots retained as history. |
| `docs/PROJECT_LOG.md` | Chronological repository and operations log. Append-only; historical entries are not rewritten. |
| `docs/README.md` | Documentation-directory orientation. |
| `docs/REPOSITORY_STRUCTURE.md` | Canonical directory and document naming rules, version-control policy, standard layout. |
| `docs/audits/REPOSITORY_AUDIT_20260716.md` | Pre-migration structural, code, data, result and risk audit. Dated snapshot; its paths predate two renames. |
| `docs/audits/REORGANIZATION_20260716.md` | First-stage duplicate cleanup and research-direction rationale. Dated snapshot. |
| `docs/audits/STRUCTURE_STANDARDIZATION_20260716.md` | Directory migration map and verification record. Dated snapshot. |

## Live R2 documents

| File | Use |
|------|-----|
| `docs/RESEARCH_PLAN.md` | Scope, panel, measurement package, literature gate, evidence discipline, compute policy. |
| `docs/EXPERIMENT_LOG.md` | Chronological experiment record, `EXP-R2-NNN`. **Ids 026–032 are each used twice** — cite by date as well as id. |
| `docs/methods/TRANSFER_MEASUREMENT_PROGRAMME.md` | Per-stage estimand and metric definitions. Methods only. |
| `docs/analysis/P0_2B_DICTIONARY_FIDELITY_RESULTS_20260727.md` | P0-2 / P0-2b dictionary-fidelity results and claim limits — the evidence behind limitation L1, and the baseline plan item C1 re-qualifies. |
| `docs/analysis/DESIGN_CELL_MODELS_20260728.md` | Modality x tokenisation design cells; why protein x subword is permanently n = 1. |
| `docs/analysis/MODEL_LADDER_20260728.md` | Within-lineage size-ladder staging. |
| `docs/analysis/H200_STAGING_20260728.md` | GPFS staging for the instrument-transfer campaign. |
| `scripts/transfer/README.md` | How to run the campaign; the H200 controller and worker. |

## Frozen provenance

Nothing under `archive/` is authoritative. Its documents, receipts and manifests retain the paths and root names that existed when they were written; those are **not** normalized, because a receipt is a statement about a past execution.

| Path | Contents |
|------|----------|
| `archive/legacy/r2_retired_scope_20260729/` | **The retired R2 scope.** Conserved sparse-readout atlas, EC steering and enzyme-design results, the npj Artificial Intelligence manuscript package, the P0 protocol set, the procedure-and-results audit, the npj literature corpus, the recoverability preregistration, and operational receipts. Has its own README with a what/why/where-now table. |
| `archive/legacy/r2_transfer_log_snapshots_20260728/` | Point-in-time copies of *live* transfer result trees taken as a safety net during the 2026-07-28 campaign. Not evidence; not simple duplicates. Has its own README with a per-snapshot identity table. |
| `archive/legacy/r2_transfer_gap_precorrection_20260729/` | Pre-correction TG-series sources and a per-number status table (audit §0.1). |
| `archive/legacy/r2_superseded_scripts_20260728/` | Superseded R2 analysis and launcher scripts. |
| `archive/legacy/nested_git_history_20260729/` | The two nested git repositories retired on 2026-07-29, histories and remote configuration intact. |
| `archive/legacy/*.md` | Historical v1/v2 research proposals. |
| `archive/retired_research_roots/` | R0 and R1 in full, with their results trees, plus the retired R1/R2 agent definitions. |
| `archive/conversation_history/` | Rescue-phase and planning packets (`OPUS_*`, `FINAL_*`, `TODO_NEXT*`). Final decision packets: `OPUS_FINAL_VERDICT_20260518.md`, `OPUS_FINAL_RESCUE_HANDOFF_20260518.md`. |
| `archive/manuscripts/` | Old combined R1+R2 scaffold, the Springer Nature template source, the superseded standalone R1 draft, and the former combined evidence index. |
| `archive/project_records/` | Superseded planning and review records. |
| `archive/analysis_exports/` | Frozen cross-project export bundles. |

**`archive/` lives on the execution host only** — it is outside the mirror's synchronization scope, so its contents are not visible from the code mirror. Inspect it with shell commands, not with file tools.

## Superseded, at the repository root

`check.md` — frozen, no longer maintained, superseded by the audit document. Its amendment sections record how individual conclusions were reached and withdrawn; its body asserts claims that have since been retracted. Do not cite it.

## External operational files (outside the repository)

| File | Summary |
|------|---------|
| `/home/lzp/hangzhou-remote/README.md` | Cluster access and the latest verified resource note. Can become stale; query live capacity before scheduling. |
| `/home/lzp/hangzhou-remote/H200_BIOCC_STATUS.md` | H200 pod / experiment status ledger. |
| `/home/lzp/hangzhou-remote/EXTERNAL_RESOURCE_DOWNLOADS.md` | External resource download status. |
