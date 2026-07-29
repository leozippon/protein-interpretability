# R2 Manuscript Audit - 2026-05-28

## Status

The R2 manuscript is technically compileable and now includes regenerated Figures 2--5.
It is not yet submission-complete because author metadata, Figure 1 and final
acknowledgement/funding text still need manual completion.

## Actions completed

- Synced v2 ESMFold PDB outputs from the H200 pod/GPFS into the local tree:
  - `r2_interpretability_transfer/results/drug_design/ec_lysozyme_esmfold_metrics_v2_20260425_r2_v2_1gpu_pdbs/` (10 PDBs)
  - `r2_interpretability_transfer/results/drug_design/ec_lysozyme_unsteered_esmfold_metrics_v2_20260425_r2_v2_1gpu_pdbs/` (20 PDBs)
- Updated `r2_interpretability_transfer/scripts/50_make_manuscript_figures.py` so Figure 5 uses the v2 PDB directory and reads v2 ESMFold/Foldseek aggregate values from JSON evidence files.
- Regenerated:
  - `figures/figure_02_sparse_readout_atlas.pdf/.png`
  - `figures/figure_03_nterminal_readouts.pdf/.png`
  - `figures/figure_04_negative_interventions.pdf/.png`
  - `figures/figure_05_sequence_structure_checks.pdf/.png`
- Recompiled `main.pdf` with Tectonic.
- Removed internal project wording such as "R2 standalone paper" from the Introduction.
- Replaced the internal caption phrase "M-2 synthesis" with a reader-facing description.
- Added the ESMFold citation to the manuscript text.

## Verification

- `main.pdf` was regenerated on 2026-05-29 and currently has 16 pages.
- All LaTeX `\ref{}` targets resolve at the source level.
- All included figure PDFs exist.
- Numeric spot checks matched the evidence files:
  - Universal triplets: 38 / 30 / 8 at `|r| >= 0.90 / 0.95 / 0.98`
  - 30x null means: 0.067 / 0.067 / 0.033; null max: 1
  - v2 ESMFold mean pLDDT: steered 67.7, unsteered 73.2
  - Foldseek mean top TM: steered leads 0.883, unsteered 0.938

## Remaining gaps

- Figure 1 is still a placeholder workflow schematic.
- Author list and affiliation are placeholders.
- Acknowledgements/funding/compute text is still placeholder-like.
- The expanded reference list now covers PLMs, sparse dictionaries/transcoders, circuit tracing, probing controls, attention sinks, Swiss-Prot, Pfam/HMMER, CLEAN, ESMFold and Foldseek.
- Tectonic compiles successfully but still emits layout warnings, mainly underfull boxes caused by float-heavy Extended Data pages.

## Recommendation

The manuscript is now internally more consistent, especially Figure 5 versus the v2
GPFS/PDB evidence. It is suitable for scientific review by Opus/Claude, but not
ready as a submission package until the remaining metadata and Figure 1 are
completed.

## Addendum: GPT5.5 Pro Citation Pass

Updated on 2026-05-28 after the GPT5.5 Pro manuscript revision:

- `main.tex` now contains an expanded related-work introduction and 30 citation keys.
- `references.bib` was rebuilt to include all 30 cited keys; the previous file still had only four entries.
- Several conference/workshop entries were encoded as `@misc` with `howpublished` because `sn-nature.bst` emitted BibTeX stack errors for those `@inproceedings` entries.
- `main.pdf` was recompiled successfully with Tectonic.
- Current unresolved source-level issues are limited to author metadata and the Figure 1 placeholder.

## Addendum: Writing and visualization revision (NMI level)

Updated on 2026-05-28 (writing + visualization polish pass; no claims or numbers changed):

Writing (`main.tex`):
- Rewrote the abstract with a sharper hypothesis-framed opening and grounded the
  sparse-dictionary terminology as cross-layer transcoders (CLTs), so the
  `d_CLT` symbol and the "CLT feature patch" references are now defined.
- Tightened the introduction's contribution statement and the first Results
  paragraph (added the explicit CLT definition at first use).
- Unified notation: the causal Extended Data table (Table 9) now uses
  `$\Delta$NLL` everywhere (was the inconsistent `$d$NLL` -> "dNLL"), matching
  the main text and Fig. 4; used proper minus signs and "first-2" wording.
- Fixed layout warnings: Table 9 columns are now ragged-right
  (`>{\raggedright\arraybackslash}`), removing the per-cell justification
  underfull-hbox warnings; made the long code-availability path breakable with
  `\allowbreak`. The only remaining warnings are benign "underfull vbox while
  output active" notices from one-table-per-page Extended Data floats.

Visualization (`r2_interpretability_transfer/scripts/50_make_manuscript_figures.py`):
- Upgraded the shared Nature style (Arial-first stack, axis-below light grids,
  white figure/save facecolor, regular mathtext, tidy legend spacing).
- Fig. 2b: replaced the white-asterisk significance markers (which produced a
  noisy dashed-line artefact in dense rows) with a significance-masked
  sequential colormap -- only BH-significant cells are shaded by `-log10 q`,
  non-significant cells are grey, so dominant signatures read as solid blocks.
  Updated the Fig. 2 caption accordingly.
- Added value labels and y-grids across the bar panels (Fig. 2a/2d, 3a/3b/3c,
  5d), capped 95% CIs on the steering forest plot (Fig. 4a), and value labels on
  the near-zero causal bars (Fig. 4b) so the gap to the 0.5-nat gate is legible.
- Switched panel titles from verdict phrasing ("all fail", "0/8 significant")
  to neutral descriptors, leaving the conclusions to the figure captions.
- Removed dead imports/helpers from the figure script.
- All four figures regenerated (vector PDF + 600-dpi PNG) and `main.pdf`
  recompiled with Tectonic (16 pages, no errors, no undefined refs/citations).

## Addendum: repository consistency check

Updated on 2026-05-29 after the Claude repository update was reviewed:

- Re-ran the Paper A manuscript figure generator and Tectonic build; `main.pdf`
  still compiles to 16 pages with no unresolved citations or references.
- Confirmed all four included figure PDFs exist and the source-level citation
  audit reports 30 cited keys matched by 30 bibliography entries.
- Repaired live `ProteinInterpret` and H200 queue helper paths that still
  referenced the pre-split `Research1`/`Research2` directories.
- Regenerated the `r0_shared_interpretability_framework/results/v0_20260515` benchmark tables and
  the encoder mechanism-localization preflight JSON so their recorded paths
  match `r1_encoder_interpretability_benchmark/` and `r2_interpretability_transfer/`.

## Addendum 2026-06-12 — Recoverability audit integrated

Integrated the representation-recoverability audit + capacity-falsification
(EXP-R2-019/020/021) into the manuscript:

- Abstract: added the audit + fourfold-retrain finding (negatives reflect
  substrate/objective, not dictionary capacity).
- New Results subsection "A recoverability audit attributes the negatives to the
  substrate, not the dictionary" (ceiling/floor/rho, EC family-confound, capacity
  no-go + falsification, 0/8 oracle steering).
- Discussion: added a paragraph explaining the negatives mechanistically.
- Methods: added "Representation-recoverability audit" subsection.
- Extended Data Table 10 (`tab:recoverability`): per-model x readout C/F/rho/phi
  from the corrected v2 probes (decision_v2b canonical).
- Numbers sourced from `r2_interpretability_transfer/evidence/recoverability_audit_20260605_1250/`
  (probes_v2, decision_v2b) and the expanded eval `R2_EXPANDED_DOWNSTREAM_ANALYSIS_20260612.md`.

Compiles with Tectonic: 19 pp., no errors, no undefined refs/citations.
Pending (author): Figure 1 schematic, author metadata, and an optional Figure 6
visualizing ceiling-vs-floor + the EC confound (data in probes_v2).

## Addendum 2026-06-13 — Figure 6 added

Added `figure_06_recoverability_audit` (3 panels: floor-vs-ceiling faithfulness scatter
with $y=x$; EC family-disjoint-vs-stratified confound bars; recovery $\rho$
reference-vs-expanded dictionary with faithful/bottleneck lines). Generated by
`50_make_manuscript_figures.py::figure6` from the canonical
`probes_v2/probe_results.json` (original dicts) plus the recorded expanded-eval
$\rho$ values (EXP-R2-021). Wired into the recoverability Results subsection
(`\ref{fig:recoverability}`, panels a/b/c) and the data-availability figure range.
Compiles with Tectonic: 20 pp., no errors, no undefined refs. Remaining author
TODOs: Figure 1 workflow schematic and author metadata.
