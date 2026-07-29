# Paper A source data

Compact, sequence-free processed evidence for the empirical Figures 2–6 and
Supplementary Tables 1–11 of the Paper A manuscript. Figure 1 is a conceptual
proposal for an R/S/C/U benchmark, not an executed empirical panel, and therefore
has no associated source dataset. For Figure 5, this package supplies aggregate
metrics and provenance only, not panel-complete source data. The package contains
no raw protein cohorts, generated amino-acid sequences, rendered structures,
activation caches, model weights, optimizer states or checkpoints and therefore
is not an end-to-end reproduction archive.

## Layout

- `table_s01_model_quality/` — legacy dictionary-quality summaries, padding audit
  and safe local architecture config JSONs.
- `figure_02_tables_s02_s04_s05/` — recurrent-triplet atlas/null, characterization,
  synthesis and the single checkpoint-pair comparison.
- `table_s03_swissprot_basis/` — canonical Swiss-Prot top-event evidence, the
  prevalence-aware MI re-audit and quick 38-dimensional basis probes.
- `figure_03_table_s06/` — N-terminal subset, saved top-event positions and
  contextual tests.
- `figure_04_tables_s07_s09/` — steering/intervention summaries, the complete
  direct-effect candidate-sign table and all condition summaries used by
  Figure 4 and the causal supplementary table.
- `figure_05_table_s08/` — aggregate ESMFold, Foldseek, Pfam, CLEAN and
  metric-calibration summaries plus source hashes for the mixed-run lysozyme
  analysis; this is not panel-complete Figure 5 source data.
- `figure_06_table_s10/` — amended recoverability probes, decision output and
  exploratory wider-dictionary evidence.
- `table_s11_synthetic_positive_control/` — exact prospective synthetic-control
  specification and execution receipts, the complete result summary and a
  deterministic per-seed numeric extraction. This is synthetic pipeline
  calibration only and contains no pretrained-model causal evidence.

`MANIFEST.tsv` maps every evidence file to its canonical source and manuscript
role. `SHA256SUMS` covers the manifest, this README, the build script and every
packaged evidence file.

The repository-level machine-readable historical panel/table audit is
`docs/PANEL_TABLE_PROVENANCE_MAP_20260717.json`. It records missing upstream
commands, cohorts, revisions and checkpoints explicitly and therefore does not
claim that the present processed package is an end-to-end release.

## Copy and redaction policy

Safe evidence is copied byte-for-byte and marked `exact_copy` unless a
historical JSON file contains a non-finite Python extension. In that case the
packaged RFC 8259 JSON replaces each non-finite value with `null`, records every
affected JSON path and original token in `_non_finite_normalization`, retains
the historical source hash, and is marked `normalized_json`. Two canonical
inputs contain generated protein sequences and are not copied:

1. `r2_interpretability_transfer/results/steering_benchmark/zymctrl_v2_onmanifold_direct_20260503.json`
   contains `example_unsteered` and `example_steered` sequences for every class.
   `steering_numeric_summary_no_sequences.json` retains the exact numeric,
   intervention and configuration fields and records the source hash.
2. `r2_interpretability_transfer/results/drug_design/ec_lysozyme_leads_v2.json` contains
   selected, all-record and unsteered generated sequences.
   `generation_counts_no_sequences.json` retains only the denominators used in
   Figure 5 and records the source hash.

Supplementary Table 10b is a deterministic numeric extraction from the final
training-quality table in
`r2_interpretability_transfer/docs/analysis/EXPANDED_DICTIONARY_ANALYSIS_20260612.md`,
joined to the
dictionary width in the existing machine-readable expanded-probe JSON. It is
marked `derived_numeric` rather than an exact copy.

Supplementary Table 11 is bound to the immutable one-time synthetic production
run. The packaged specification, freeze manifest, exclusive execution claim,
summary and run manifest are exact copies. `per_seed_results.tsv` is a
deterministic extraction from the complete summary. The summary and run-manifest
SHA-256 values are
`d952a75dc2785d305bd245c6e79b0e7fc94122c475af3ecadd08d3411962dbea` and
`930ad2d4fbcbee63568cd52a9c4209b2270727c6b9b8fd446a2d0044450ef039`.
The fixed planted comparator is not a learned biological circuit; these files
do not support causal inference for a pretrained protein model. The
sequence-bearing cohort, full feature-discovery and intervention JSONL files,
and three model checkpoints are bound by the run manifest but omitted here;
they must accompany the final licensed DOI-backed deposit.

The extraction logic is implemented in `build_source_data.py`; excluded fields,
source hashes and derivation notes are repeated in `MANIFEST.tsv`.

## Build and verify

From the repository root:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ct
PYTHONDONTWRITEBYTECODE=1 python \
  r2_interpretability_transfer/manuscript/source_data/build_source_data.py
PYTHONDONTWRITEBYTECODE=1 python \
  r2_interpretability_transfer/manuscript/source_data/build_source_data.py --verify-only
```

The builder fails on missing or changed canonical inputs, non-identical exact
copies, non-RFC-8259 JSON, unmanifested files, checksum mismatches, sequence-like
values in JSON or TSV cells, and credential-like content in the model config
copies.

## Scope notes

- Figure 5 structure renders are presentation images rather than JSON/TSV
  source data and are not duplicated here. Panels a--c and d currently come
  from different historical runs, and structure display and steered-lead
  selection were not blinded. Only aggregate metrics, identifiers and source
  hashes are retained as provenance pending a single-run symmetric replacement;
  the underlying generated sequences and rendered structures are absent from
  this compact package.
- The exact historical 200-sequence atlas file was recovered from archived
  compute storage with SHA-256
  `5bc7697a83cc7461558f8b4597a3c9b4d6a151b7ec70ca22efc7282ecde4f0a6`.
  It exactly validates the alternating row order; the enriched records have
  ordered-record SHA-256
  `07213d4a9cefbdb055206e08d3137722c446acf43b5b3342db571b977032c724`,
  with local status `historical_exact_file_verified`. It is sequence-bearing
  and is not included here; it must be added to the final licensed deposit.
- The three exact reference CLT files named by the saved atlas were likewise
  located and hash-verified on archived compute storage. Their SHA-256 values
  are `5eca3b19284dbd9b302078e3a7e34ce7a2fc78d97b1566eae927d4d1c30f1f00`
  (ProtGPT2),
  `5da70c530b83a034d1fe683a72a8cc5bd7b49463d2598036cd6b5db94ca5761d`
  (ZymCTRL) and
  `5e384733dc28ecad3947b65c0c8b34f058ce50a61aab67399548c2b21687b8fd`
  (ProGen2-medium). The repository recovery manifest contains their exact
  retained configurations. These multi-gigabyte files are not part of this
  compact package and still require a persistent licensed deposit.
- The canonical invalid 0.1-nat Swiss-Prot gate rows are retained only as audit
  provenance. The corrected Supplementary Table 3 uses the dated MI re-audit.
- The site-mismatched mean-direction intervention is included as provenance,
  not as a valid oracle-controllability result.
- Wider-dictionary outputs remain explicitly exploratory and did not meet the
  preregistered reconstruction-quality criterion.
