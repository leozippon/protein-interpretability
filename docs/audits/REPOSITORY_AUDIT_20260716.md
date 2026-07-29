# Repository audit — 2026-07-16

> Snapshot note: this audit records the paths that existed when the audit was performed. The later same-day canonical mapping is documented in `STRUCTURE_STANDARDIZATION_20260716.md`; historical size/result references below are intentionally not rewritten.

## Scope and method

The audit covered the synchronized source mirror and the B-workstation-only data/result trees. All live Markdown, text configuration, shell, Python and LaTeX sources were inventoried and searched; project status records, preregistrations, experiment logs, manuscript sources and result summaries were read directly. Large binary datasets, checkpoints and generated artifacts were audited through directory inventories, sizes, schemas, manifests, hashes, metadata and representative content rather than pretending that 306 GB of binary payload can be read as prose.

The B tree contained approximately 58,186 files and 306 GB at audit time:

| Area | Approximate size | Files | Audit treatment |
|---|---:|---:|---|
| `data/` | 255 GB | 47,494 | inventory, formats, dataset roles, schemas/manifests and representative records |
| `paper_r2_nature_mi/results/` | 45 GB | 281 | all text/JSON/TSV summaries and logs; checkpoint/binary metadata |
| `benchmark_encoder/results/` | 3.5 GB | 143 | all text/JSON/TSV summaries and status records |
| `external_resources/` | 2.4 GB | 88 | inventory, licences/readmes and staged-tool roles |
| remaining live source/docs | under 1 GB | remainder | direct source/document review |

Remote expanded CLT checkpoints are substantially larger (about 1.3 TB) and are referenced by their remote manifests/logs; they are not duplicated into the source mirror.

## Research directions

The evidence supports two active directions, not three simultaneous projects:

1. **Decoder/generative PLM audit (Paper A):** cross-model sparse-readout recurrence, low-level characterization, candidate checkpoint sensitivity, interventions and recoverability. Home: `paper_r2_nature_mi/`.
2. **Encoder interpretability benchmark (Paper B):** ESM-2 SAE interpretation, IndelMissense and calibrated audit. Home: `benchmark_encoder/`.

`ProteinInterpret/` is shared evaluation infrastructure. A decoder-only benchmark is future Paper C, seeded by Paper A, rather than a third active research tree.

## Structural findings

- The resurrected root trees `Research1/`, `Research2/` and `manuscripts/` were exact or strict-superset duplicates of canonical live/archive material. They were removed from both Mutagen endpoints after adding ignore rules to prevent resurrection.
- Active project roots must remain at their current depth: at least 36 Python sources resolve the repository through `Path(__file__).parents[2]`, and many operational scripts embed the canonical root names. A cosmetic rename would silently change data and result paths.
- The old root `results/` contained duplicate/stale Paper B outputs and one planning snapshot. Duplicates were removed and the planning snapshot archived.
- `analysis_exports/` contained frozen May/June export bundles, not live research. It was moved to `archive/analysis_exports/` and excluded from synchronization because the B copy contains large binary snapshots.
- The live Paper A and Paper B proposal files described pre-pivot drug-design and mechanism claims. They were archived and replaced by short current direction briefs.

## Paper A evidence audit

The audit found five issues that materially change the manuscript:

1. **Invalid annotation gate.** The original top-100 event has maximum possible mutual information 0.00661255 nats, so the 0.1-nat gate was unattainable. EXP-R2-022 provides a deterministic, 2,000-permutation exploratory repair.
2. **Activation-space naming.** The hooks capture a layer-normalized architecture-specific CLT input, not a raw residual stream. ProtGPT2/ZymCTRL and ProGen2-medium also expose related but non-homologous hook sites.
3. **Padding.** Training and quick evaluation omit attention masks and count padded positions. Exact quick-evaluation contamination was 3.57% for ProtGPT2 and 2.75% for ZymCTRL; quality values are unmasked diagnostics.
4. **Intervention validity.** The eight-class steering metric is a motif/composition heuristic. The later direction experiment derives a vector from CLT-input coordinates but injects it at MLP output, so it cannot support a distributed/robust-mechanism conclusion.
5. **Wider dictionaries.** None met the preregistered FVU < 0.15 requirement; recovery changes were mixed. Capacity or optimization is not falsified.

The defensible Paper A core is therefore procedure-specific recurrence, overlapping low-level associations, three N-terminal initiator-methionine readouts with high unnormalized received attention, a single-pair candidate checkpoint diagnostic and bounded negative interventions.

## Reproducibility and security findings

- The exact 200-sequence atlas cohort file named by saved results is missing, preventing exact regeneration of the headline atlas from raw sequences.
- Immutable upstream model revision identifiers were not retained.
- The live documentation and one download script contained a hard-coded Hugging Face credential. Live copies were replaced with an `HF_TOKEN` environment-variable requirement. Frozen archives/backups were not rewritten; the exposed credential must be revoked and rotated.
- Author list, affiliations, funding/compute acknowledgements and a DOI-backed data/code release remain human/submission blockers.

## Verification artifacts

- Paper A evidence repair: `paper_r2_nature_mi/results/circuit_analysis/swissprot_triplet_mi_reaudit_20260716/`
- Paper A recent literature: `paper_r2_nature_mi/literature/npj_recent/MANIFEST.tsv`
- Repository changes: `project_records/REORGANIZATION_20260716.md`
- Detailed experiment chronology: each project `docs/EXPERIMENT_LOG.md` and `project_records/LOG.md`
