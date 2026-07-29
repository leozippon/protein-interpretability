# Paper A npj Artificial Intelligence package

This is the major-revision manuscript package for **Auditing semantic and causal claims from
cross-model sparse readouts in protein language models**. It uses the official
December 2024 Springer Nature `sn-jnl` class with the `sn-nature` reference
style, following the Article structure used by npj Artificial Intelligence.

## Files

- `main.tex` / `main.pdf`: 20-page Article with six main figures and no main
  tables.
- `supplementary_information.tex` / `.pdf`: standalone supplement with exactly
  11 tables.
- `references.bib`: 46 bibliography records, of which 43 are cited in the
  Article, including recent npj AI, Nature Machine Intelligence, Nature
  Methods and biological-design anchors.
- `figures/`: generated PDF/PNG figures; Figure 1 is a conceptual vector
  schematic for a proposed R/S/C/U benchmark, not an executed empirical panel.
- `source_data/`: compact processed evidence, provenance manifest and hashes for
  empirical Figures 2–6 and Supplementary Tables 1–11. Figure 5 is represented
  by aggregate metrics and provenance, not panel-complete source data.
- `template/`: untouched official December 2024 package, extracted
  source and provenance README.
- `../docs/manuscript/NPJ_ADAPTATION_20260716.md`: official-guidance mapping,
  package checks and unresolved submission items.

The 20-paper open-access literature library and synthesis are in
`../literature/`; metadata and hashes are in `../literature/npj_recent/`.

## Evidence framing

The manuscript reports procedure- and cohort-sensitive sparse-readout recurrence, overlapping
low-level associations, N-terminal initiator-methionine readouts with high
unnormalized received attention, a candidate checkpoint-sensitivity comparison,
and bounded gate failures for specified ablation paths. The eight-class
steering result is retained only as a no-positive-evidence pilot because its
fallback selector included negative-attribution features. It does not claim a
biological-primitive dictionary, causal attention sink, successful EC-class
steering, single-feature causality or capacity falsification.

Figure 1 proposes four prospective estimands—recurrence (`R`), residual
semantics (`S`), causal computation (`C`) and downstream utility (`U`)—with
independent data and gates. The complete benchmark was not executed in this
study; historical analyses supply bounded evidence for only parts of it.

The July 2026 revision additionally:

- removes the mathematically unattainable original 0.1-nat Swiss-Prot gate and
  reports the deterministic exploratory matched-null re-audit;
- identifies the hooked tensors as architecture-specific layer-normalized CLT
  inputs and discloses unmasked padded-position training/evaluation;
- labels the eight-class score as a motif/composition heuristic;
- excludes the site-mismatched direction experiment from causal conclusions;
  and
- treats wider dictionaries as exploratory because none met the preregistered
  FVU criterion; and
- reports a one-time, prospectively frozen synthetic planted-control pass in
  three model seeds as pipeline calibration only. The fixed comparator is not a
  learned biological circuit, and the real held-out pretrained-model causal
  gate remains open.

## Build

Run figures from the repository root:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ct
python r2_interpretability_transfer/scripts/50_make_manuscript_figures.py
```

Compile from this directory:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate latex
tectonic --reruns 2 main.tex
tectonic supplementary_information.tex
```

The explicit two reruns avoid a Tectonic 0.15 convergence-check warning with
the official Nature bibliography style. Both documents compile without errors
or unresolved references; the remaining messages are underfull-box warnings.

## Submission blockers

The independent assessment classifies this as a major revision before
submission. P0-1 through P0-9 in
`../docs/NPJ_AI_MAJOR_REVISION_PLAN_20260716.md` are scientific and
reproducibility gates; the items below are additional release and human-input
blockers, not the complete blocker list.

- replace placeholder coauthors, affiliation, funding, compute and
  acknowledgements with verified human metadata;
- deposit the independently recovered historical 200-sequence atlas cohort;
  the archived source file has SHA-256
  `5bc7697a83cc7461558f8b4597a3c9b4d6a151b7ec70ca22efc7282ecde4f0a6`,
  exactly validates the alternating order, and the enriched ordered records have SHA-256
  `07213d4a9cefbdb055206e08d3137722c446acf43b5b3342db571b977032c724`
  with status `historical_exact_file_verified`;
- deposit the three exact historical reference CLTs now located and
  hash-verified on archived compute storage; their identities and retained
  configurations are bound by
  `../evidence/historical_reference_checkpoints_20260717/manifest.json`;
- record immutable upstream pretrained-model revisions;
- publish a versioned code release and DOI-backed checkpoint/source-data
  deposit; and
- obtain all coauthor approvals and complete journal declarations.

`../docs/manuscript/MANUSCRIPT_AUDIT_20260528.md` is retained as a historical
pre-npj audit. Repository-wide audits are in `../../docs/audits/`; dated
evidence-repair outputs remain under `../results/circuit_analysis/`.
