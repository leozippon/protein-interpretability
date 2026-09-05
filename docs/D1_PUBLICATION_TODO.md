# Direction 1 publication work queue

Updated: 2026-09-05. Owner: root coordinator. This file tracks execution and handoffs across context compaction; scientific findings and limitations remain canonical in [the audit](INTERPRETABILITY_TRANSFER_AUDIT.md), experiment records in [the experiment log](EXPERIMENT_LOG.md), and the public overview in [summary.md](../summary.md).

## Agreed deliverable and scope

Complete the frozen generation experiments, assess whether their behavioral evidence supports learned biological information, and deliver a polished English npj Artificial Intelligence Article with reproducible results, editable figures, source data and submission materials. Direction 1 asks **whether** knowledge is evident in generation; identifying particular biological rules and mechanisms remains Directions 2 and 3. Computational structure confidence does not establish measured folding or function. No new decoding search, model matrix or outcome-driven selection is authorized by a disappointing result.

The user requests a manuscript approaching the journal's suggested main-text upper limit, substantive sections comparable to published Articles, professional visualizations and restrained table color. Use the official common Springer Nature template and the journal's requirements; do not imitate publisher production by altering the class. The verified Article guidance is a main text typically no longer than 4,000–4,500 words excluding abstract, Methods, references and figure legends; an unreferenced abstract of about 150 words; at most 10 main figures/tables; and approximately 60 references as a guideline. Follow the user’s request to meet this guidance, with each reference supporting an actual claim. All references must exist and use consistent names, case and bibliographic formatting. Code belongs at <https://github.com/leozippon/protein-interpretability>.

The user explicitly confirmed that the manuscript must **not be made public yet**. Keep all draft LaTeX, manuscript figures and review materials under the ignored local `manuscript/` directory and deliver them in local archives. Publish only the authorized code, protocols and reproducibility documentation. The final paper must include substantive numerical tables, a detailed category-level table and varied data-backed visualizations, within the journal's display guidance; extra prose cannot substitute for visible evidence.

## Completed

- [x] Initial delegated repository review and bounded independent scientific assessments.
- [x] Official template, six npj AI comparison papers and active reporting materials downloaded with provenance.
- [x] R232 and R233 protocols, selection and inference source frozen before their respective new outcomes; immutable documents remain unchanged.
- [x] Qualified H200 runtime and ESMFold; identified and excluded the faulty GPU.
- [x] R232 pilot completed, raw predictions retrieved and both natural-control calibration gates passed.
- [x] R233 native generation completed: all 800 attempts retained, including 501 compiled outputs and 299 budget-censored continuations; raw batches and input hashes verified.
- [x] Annotation assets fully staged and verified; both CPU annotation tasks actually started.
- [x] Initial LaTeX draft compiled; corrected Likang Wu, author emails and matched public ORCIDs entered.
- [x] Frozen experiment/support commits `1b9a368` and `6c21e19` pushed to the requested repository.
- [x] All main/native structure computation finished: 2,320 main rows and 256 native rows, no technical failures; remote raw hashes verified. Local retrieval and final analysis remain below.
- [x] Requested Codex experimental context management enabled in local CLI configuration and validated; applicability to the already-running conversation is unconfirmed. The repository task queue remains the durable handoff.

## Running experiments and acceptance gates

- [x] **R232 main structure — structure_completion.** All 2,320 rows and 2,315 exact raw objects verified, retrieved and checked locally, including each PDB/NPZ digest.
- [x] **R233 structure — structure_completion.** All 256 rows/objects verified and retrieved locally; compiled and censored parents remain distinguishable.
- [x] **R232 fresh reference search — annotations_completion.** Local terminal manifest and all 16,400 IDs verified; 50,680 gapped HSPs independently reconstructed, historical reference fields preserved and fresh same-hit query/target coverage validated.
- [x] **R233 Pfam/reference annotations — annotations_completion.** All 800 input records, terminal manifests and local raw queries/tables validated independently; final artifacts are in `results/transfer/progen3_generation_evidence/annotations/`. Target coverage was not retained by this search and remains unavailable.
- [x] **Frozen primary statistical analysis — manuscript_completion.** Complete main/native metrics analyzed with the frozen estimands, weights and class/group bootstrap units. Results and remaining descriptive integration belong in the canonical scientific records and manuscript, not this task list.

## Manuscript and figures

- [x] Rewrite Introduction, Results and Discussion into a connected scientific argument, meeting the official 4,000–4,500-word guidance (current accepted text: 4,059 words); Methods remain complete and separate from this count.
- [x] Replace result-dependent placeholders only after terminal verification; finalize title and abstract around actual findings and their literature contribution.
- [x] Apply the user's explicit reading-order correction: embed each main figure/table near its first Results discussion, remove internal commit/run identifiers from reader-facing prose, and compare paragraph/display organization with the published npj AI Article “The role of large language models in emergency care: a comprehensive benchmarking study” (`s44387-026-00078-2`). Do not copy publisher branding or prose. A compiling PDF with all displays appended at the end is not accepted as the final review layout.
- [x] Integrate distributions, paired contrasts, uncertainty, family recognition, nonredundant yield and reference coverage without an unjustified common-task model ranking.
- [x] Review and refine the editable draw.io workflow; add protein-structure views if they explain results, using an explicit non-cherry-picked illustration rule and actual predicted coordinates.
- [x] Check every figure at publication size: readable labels, units, sample sizes, color accessibility, panel balance and informative legends; keep SVG/PDF and source tables.
- [x] Use restrained, consistent table shading where helpful, and inspect every PDF page for typography, floats, missing content and clipping.
- [x] Verify approximately 60 substantively relevant citations against primary sources and unify BibTeX formatting and case protection; count references actually cited in the compiled manuscript.
- [x] Complete sentence-by-sentence alignment with published npj AI Articles, including each sentence’s role, syntax, tense, voice, length and evidence strength; retain the private comparison map.
- [x] Complete figure-by-figure alignment with published npj AI displays and publication-associated Nature/Cell/Science plotting code; retain editable assets and source provenance.
- [x] Use the official default single-column layout and at most four subsections in each major section. Move the design-only population table into Methods and make the first Results table a substantive quantitative result.
- [x] Standardize affiliations, including the official form of Hong Kong, China; do not invent an HKUST department.
- [x] Emphasize Zipeng Liu's reported lead contribution across the research and writing. Other authors' roles are proposed text pending factual verification; final author approval must not be invented.
- [x] Complete factual Reporting Summary answers, cover letter and submission checklist; preserve the AI-assistance disclosure. The old linked Editorial Policy Checklist is retired.
- [x] Update the audit, summary, research plan and venue assessment to actual final status; remove obsolete suggestions from the current plan without rewriting history.

- [x] Complete the additional quantitative comparison of section proportions, display placement, size and content: 14 main/Supplementary displays and 18 comparator displays are measured. Introduction is 16.4% of the final main text; Table 2 uses 9.5-pt data text and 86.2% of the column. Every main display is within two pages of its first citation, with all 30 pages visually accepted. The official single-column class remains unchanged.

## Final acceptance and delivery

- [x] One bounded independent scientific review completed. No invalidating primary-result defect found; it identified three required corrections: report observed natural/shuffle confidence-event counts, describe the two conditional contrasts as one multiplicity family, and restore the missing PDFBench author. All three corrections are integrated. The complete-page visual check and additional quantitative display comparison both passed.
- [x] Run relevant code checks and compile the manuscript: six-script syntax/lint checks and the real 800-record annotation join pass; the 30-page main and four-page supplement compile and all 29 class-table rows match source. Recheck only artifacts changed by the additional display pass.
- [x] Assemble manuscript/source-figure package, minimum reproducibility data, source-code archive and SHA-256 delivery manifest under `results/transfer/generation_evidence/publication/20260905/`; link separate verified raw-structure archives. Exclude weights, reference databases, credentials and downloaded journal papers.
- [x] Distinguish a local review package from a public archival deposit; do not invent a DOI or imply journal submission has occurred.
- [x] Commit and push focused code/documentation changes; preserve unrelated user edits to `AGENTS.md` and `CLAUDE.md`, and preserve the user's changes in `summary.md`.
- [x] Fulfil the explicit open-source request: the target GitHub repository is **public**, with the original software under MIT. Focused changes are pushed; the bounded current-file/history publication checks passed. The current manuscript has zero tracked files and remains local. Third-party terms are retained; the software license does not cover manuscript prose, figures or scientific data.
- [x] Release only the temporary task allocation after all GPU/CPU jobs and retrievals finish; deletion completed at 2026-09-05 08:03 UTC after all four assigned GPUs were idle. The original single-GPU allocation and other users' workloads were left alone.
- [x] Deliver an honest assessment of Direction 1's contribution and npj AI readiness, plus remaining author-side tasks; completion of computation does not guarantee acceptance.

## Parallel metrics assessment

- [x] Independently assess existing Sharpe, Sharpe*, DSR, beta and alpha research and its relevance to all Direction-1 tasks. [Assessment](d1-financial-metrics-assessment.md) is complete; it recommends evaluating standardized paired effects or baseline-calibration regressions only for a named question, and reserves selection adjustment for a future fully recorded candidate search. No frozen endpoint or experiment was changed.

## Operational handoff

The original scientific run is `20260904232436_00129607e3c7` (`1b9a368`); the reference-support/sequential-schedule snapshot is `20260904234251_e2f89ea14a76` (`6c21e19`). R233 structure uses the newer schedule with the **original** inference snapshot. Remote root: `/gpfs/jiaotongdamoxing/zhk_zip/InterpretabilityTransfer`. Use its `runtimes/ct-20260905/bin/python`; the pod's default Python is unqualified.

Select allocations only in the current shell, following `AGENTS.md`; never save pod names. The temporary allocation identified by label `research-campaign=d1-generation-20260905` was released after all GPU/CPU computation and transfers completed. The original single-GPU allocation remains. No experiment is awaiting retry; the local review delivery is complete. Author approval and an actual data-access/deposit arrangement remain separate submission requirements.

Local results are under `results/transfer/generation_evidence/` and `results/transfer/progen3_generation_evidence/`; operational receipts are under ignored `logs/d1_generation_biology/` and `logs/npj_ai_review/`. The final private manuscript, editorial comparison, minimum data, committed source-code ZIP and SHA-256 manifest are under the local publication directory. The public repository contains the original software and reproducibility documentation; it contains no current manuscript files. Root alone updates this task queue.
