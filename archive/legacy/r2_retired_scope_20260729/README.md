# R2 retired scope — conserved sparse-readout audit and the npj package

**Archived:** 2026-07-29 (EXP-R2-065)
**Source root:** `r2_interpretability_transfer/` (named
`r2_decoder_sparse_readout_audit/` when most of this material was produced)

## Why

The repository serves one objective:

1. compare language-generative and protein-generative models;
2. analyse the transferability of existing interpretability methods to protein
   generative models;
3. design and validate protein-adapted interpretability methods.

Everything below served a different objective — a cross-model conserved
sparse-readout atlas, EC-conditioned steering and enzyme design, and the npj
Artificial Intelligence manuscript built on them. That programme reached its own
terminal negatives (P0-2 panel gate failed; P0-2b qualified no sparse
model/method; steering, attention-sink ablation and the drug-design line are
calibrated or bounded negatives) and is not being resumed.

The canonical live document is
`r2_interpretability_transfer/docs/INTERPRETABILITY_TRANSFER_AUDIT.md`.

## Contents

### `docs/`

| Path | What it is | Why archived | Live successor |
|---|---|---|---|
| `EXPERIMENTAL_PROCEDURES_AND_RESULTS_AUDIT_20260716.md` | Procedure-to-result audit of CLT training, atlas construction, semantic tests, steering, interventions and recoverability. | Audits the retired scope end to end. | `docs/INTERPRETABILITY_TRANSFER_AUDIT.md` |
| `npj_ai_manuscript_assessment.md` | Independent assessment of the npj manuscript. | Manuscript retired. | — |
| `NPJ_AI_MAJOR_REVISION_PLAN_20260716.md` | Revision matrix (P0-1 … P0-9) answering that assessment. | Workstream stopped. | — |
| `NPJ_AI_MAJOR_REVISION_EXECUTION_STATUS_20260717.md` | Live P0 progress ledger. | Workstream stopped. | — |
| `NPJ_REVISION_P0_3_P0_4_EXECUTION.md` | P0-3/P0-4 execution contract. | Never executed; stopped under the 2026-07-27 decision rule. | — |
| `P0_2_DICTIONARY_PROTOCOL_20260717.md` | Frozen P0-2 dictionary protocol. | Superseded as a *protocol*; its executed form survives as a receipt. | `evidence/p0_2_adjudication_20260727/p0_2_dictionary_gate_spec.executed.json` |
| `P0_2_DICTIONARY_ELIGIBILITY_RECEIPT_20260717.md` | Narrative eligibility receipt. | Same. | `evidence/p0_2_adjudication_20260727/p0_2_eligibility_receipt.json` |
| `P0_2B_DICTIONARY_FIDELITY_PROTOCOL_20260727.md` | Prospective P0-2b behavioural-qualification protocol. | Protocol markdown is redundant with the executed spec; the **result remains live evidence for L1**. | `docs/analysis/P0_2B_DICTIONARY_FIDELITY_RESULTS_20260727.md` and `evidence/p0_2b_fidelity_20260727/p0_2b_fidelity_spec.executed.json` |
| `P0_4_JOINT_ADJUDICATION_PROTOCOL_20260717.md` | Joint adjudication contract. | Never executed. | — |
| `P0_5_P0_6_CONFIRMATORY_PROTOCOL_20260717.md` | Confirmatory semantics protocol. | Never executed. | — |
| `P0_7_P0_8_CONFIRMATORY_PROTOCOL_20260717.md` | Confirmatory intervention protocol. | Never executed; contract infrastructure only, no eligible intervention surface. | — |
| `PANEL_TABLE_PROVENANCE_MAP_20260717.json` | Manuscript panel/table provenance map. | Maps manuscript objects that no longer exist in the live tree. | — |
| `manuscript/MANUSCRIPT_AUDIT_20260528.md`, `manuscript/NPJ_ADAPTATION_20260716.md` | Manuscript audit and venue-adaptation notes. | Manuscript retired. | — |
| `analysis/EXPANDED_DICTIONARY_ANALYSIS_20260612.md` | Wider-dictionary downstream analysis. | Its capacity/controllability interpretations were already withdrawn by the 2026-07-16 evidence audit; nothing live depends on it. | — |

### `results/`

| Path | What it is | Why archived |
|---|---|---|
| `circuit_analysis/` (105 MB) | Cross-model conservation atlases, the 38 conserved triplets, triplet characterisation/synthesis, N-terminal attention-sink subset, attention-sink and head ablations, Swiss-Prot MI re-audit. | The conserved-readout programme. Its terminal outcome is a bounded negative on causality and a procedure-sensitive atlas. |
| `drug_design/` (6.0 MB) | Lysozyme leads, ESMFold metrics, PyMOL renders. | Drug design is not an objective of this repository. |
| `ec_metrics/` (552 KB) | Pfam/CLEAN/Foldseek/ESMFold calibration for generated enzymes. | Supports the steering/design line only. |
| `steering_benchmark/` (116 KB), `steered_generation/` (20 KB), `causal_ablation/` (96 KB) | EC-class steering benchmark, steered generation smoke, feature-ablation runs. | Calibrated negatives for a retired line. |
| `npj_revision_20260716/` (12 MB) | Corrected steering, N-terminal counterfactuals, nested recoverability, planted causal controls, trained planted controls. | P0 revision outputs; workstream stopped. |
| `npj_revision_20260717/` (880 KB) | P0-7 prospective freeze and result. | Same. |
| `h200_results_20260404/` (2.8 MB) | Configs, logs and manifests from the April CLT training campaign. | Provenance for the April CLTs; the checkpoints themselves stay live (see "Kept" below). |
| `checkpoint_evaluation/` (48 KB), `diagnostics/` (12 KB) | Layer-quality map, quick evals, hook sanity, EC-feature provenance. | Diagnostics of the retired atlas pipeline. |
| `r2_v2_remaining_summary_20260425_r2_v2_1gpu.json` | Queue summary for the April v2 campaign. | Same. |

### `manuscript/` (11 MB)

The complete npj Artificial Intelligence package: `main.tex`,
`supplementary_information.tex`, compiled PDFs, `figures/`, `source_data/` (with
`MANIFEST.tsv` and `SHA256SUMS`), the Springer Nature `template/` and `bst/`.
Archived whole. The manuscript was never submission-ready and its claim set is
the retired one. `r2_interpretability_transfer/scripts/50_make_manuscript_figures.py`
wrote into `figures/`; that script now has no live output target.

### `preregistration/` (48 KB)

`PROTOCOL.md` (R2-RECOV-AUDIT-v1), `DECISION_LOG.md`, `NEXT_STEPS_v2.md`,
`README.md` — the frozen representation-recoverability protocol for the three
protein generators. Archived because its results tree
(`results/representation_audit_20260604/`) was never present in this repository
and its go/no-go decision governed the retired dictionary-retrain line. Its
successor discipline is the audit document's Appendix B and
`docs/RESEARCH_PLAN.md` §"Evidence discipline".

### `literature/` (40 MB)

Twenty open-access npj PDFs with `npj_recent/MANIFEST.tsv` and a synthesis,
assembled to position the npj manuscript. Archived with the manuscript. The live
literature obligation is different in kind — the audit's Appendix B rule 10
requires a *per-mechanism* search recorded in the experiment-log entry for the
track that uses it, not a venue corpus.

### `evidence/` (876 KB)

| Path | What it is | Why archived |
|---|---|---|
| `h200_cleanup_20260718/`, `_20260720/`, `_20260721/`, `_20260722/` | Path-level receipts for four receipt-preserving GPFS cleanups. | Operational receipts for a retired campaign's storage, not scientific evidence. |
| `execution_environment_20260717/` | Observed execution environment for the exact-cache run. | Same. |
| `upstream_model_revision_recovery_20260717/` | Deployed model-tree revision recovery. | Same. |
| `recoverability_audit_20260605_1250/` (772 KB) | Representation-recoverability audit outputs. | Belongs to the archived preregistration. |

**Kept live** under `r2_interpretability_transfer/evidence/`:
`p0_2_screening_20260722/`, `p0_2_full_launch_20260722/`,
`p0_2_mask_validation_20260721/`, `p0_2_adjudication_20260727/`,
`p0_2b_fidelity_20260727/` — the provenance chain of the 27 dictionaries that
plan item **C1** re-qualifies — and `historical_reference_checkpoints_20260717/`,
the manifest for the retained April CLTs.

### `logs/`

| Path | What it is |
|---|---|
| `r2/r2_v2_queue_20260420.log` | April 2026 v2 steering/design queue. |
| `root/r1_encoder_interpretability_benchmark/` | Runtime logs of the R1 encoder benchmark, retired 2026-07-29. |
| `root/r2_decoder_sparse_readout_audit/` | One April follow-up queue log filed under the pre-rename root name. |

## Kept live, and why

| Path | Size | Why it is not here |
|---|---|---|
| `r2_interpretability_transfer/results/final_checkpoints/` | 45 GB | The four April 2026 CLT checkpoints (ProtGPT2, ZymCTRL, ProGen2-medium, ProGen2-xlarge, `step_100000`) are the only protein dictionaries held locally. Plan item **C1** re-qualifies dictionaries at a powered estimand (`mlp_window8@d0.50@cohort_mean`) and needs a local dictionary to do it with. See the note on the 27 dictionaries below. |
| `r2_interpretability_transfer/results/transfer_gap_20260724/` | 1.6 GB | Cited by the audit document §0.1, which states it is "retained unmodified because it is cited here". |
| `r2_interpretability_transfer/results/transfer_2026072[89]*`, `transfer_gap_20260729_corrected/` | 48 MB | The live transfer programme's artefacts, several cited by name in the audit. |
| `r2_interpretability_transfer/evidence/p0_2*` | 5.5 MB | Evidence for limitation **L1** and the input to C1. |
| `r2_interpretability_transfer/logs/r2_clt_*_20260402*`, `logs/README.md` | 1.2 MB | The training logs of the retained April checkpoints. |

**On "the 27 dictionaries".** The 27 completed P0-2 full runs are *not* in
`results/final_checkpoints/`. They live on GPFS under
`…/npj_revision_20260717/p0_2_dictionary_controls_bf16_r3/full/<model>/<method>/seed_<n>/`;
every one of their `best.pt`, `results.json` and `run_manifest.json` paths and
SHA-256 digests is recorded in
`r2_interpretability_transfer/evidence/p0_2b_fidelity_20260727/p0_2b_fidelity_spec.executed.json`.
`results/final_checkpoints/` is a different, earlier artefact: four CLT runs from
2026-04-03. Both are retained; they are not the same thing.

## Note on paths

Documents, JSON receipts and manifests here record the paths and root names that
existed when they were executed — `Research2/`, `paper_r2_nature_mi/`,
`r2_decoder_sparse_readout_audit/`, `/gpfs/…`, `/oss-pvc/…`. **They are not
rewritten.** A receipt is a statement about a past execution; changing its paths
would falsify it. The live root is `r2_interpretability_transfer/`.

Related archives: `../r2_transfer_log_snapshots_20260728/` (superseded
intermediate copies of *live* transfer results),
`../r2_transfer_gap_precorrection_20260729/` (pre-correction TG sources),
`../r2_superseded_scripts_20260728/`.
