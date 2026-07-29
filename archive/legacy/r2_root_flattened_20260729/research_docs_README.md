# R2 documentation

| File | Use |
|---|---|
| `INTERPRETABILITY_TRANSFER_AUDIT.md` | **Canonical.** Findings, retractions, the L1–L19 limitation catalogue, the plan, and Appendix B's fourteen standing methodological rules. Where any other document disagrees with it, it wins. |
| `RESEARCH_PLAN.md` | Scope, panel, measurement package, evidence discipline, compute policy. |
| `EXPERIMENT_LOG.md` | Chronological record, one `EXP-R2-NNN` per experiment. Append only; re-read the tail immediately before writing, because other agents append concurrently. |
| `methods/TRANSFER_MEASUREMENT_PROGRAMME.md` | Per-stage estimand and metric definitions. Methods only; asserts no results. |
| `analysis/P0_2B_DICTIONARY_FIDELITY_RESULTS_20260727.md` | P0-2 / P0-2b dictionary-fidelity results and claim limits — the evidence behind limitation L1 and the baseline plan item C1 re-qualifies. |
| `analysis/DESIGN_CELL_MODELS_20260728.md` | Staging report for the modality x tokenisation design cells; records why the protein x subword cell is permanently n = 1. |
| `analysis/MODEL_LADDER_20260728.md` | Staging report for the within-lineage size ladder. |
| `analysis/H200_STAGING_20260728.md` | GPFS staging report for the instrument-transfer campaign. |

## Provenance hazard

Experiment ids **EXP-R2-026 through EXP-R2-032 are each used twice** in
`EXPERIMENT_LOG.md` — once by the July 17–22 npj-revision series and once by the
July 27–28 transfer series. Cite these by date as well as by id. Ids from
EXP-R2-033 onward are unique.

## Elsewhere

- Research overview: `../README.md`.
- Repository status and navigation: `../../docs/PROJECT_STATUS.md`,
  `../../docs/DOCUMENT_INDEX.md`.
- Compact synchronized receipts: `../evidence/`. Large generated outputs stay
  under ignored `../results/`.
- Retired-scope documents (the npj revision matrix, the P0 protocol set, the
  procedure-and-results audit of the conserved-readout programme) are frozen at
  `../../archive/legacy/r2_retired_scope_20260729/docs/`.
