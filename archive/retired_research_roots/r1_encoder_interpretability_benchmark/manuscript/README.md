# Paper B — Encoder-Only Interpretability Benchmark (manuscript)

**Status:** to be drafted. No `main.tex` exists yet.

This directory will hold the manuscript for **Paper B**, the Encoder-Only
interpretability benchmark built on the ESM-2 SAE variant work in
`r1_encoder_interpretability_benchmark/` and standardized through the shared framework in
`r0_shared_interpretability_framework/`.

## Framing

> An encoder-only interpretability benchmark for protein language models, with
> IndelMissense v1 as a centerpiece dataset and a calibrated audit of
> sparse-feature variant interpretation against accessible scalar baselines.

The benchmark evaluates *interpretation quality* (localization, faithfulness,
annotation alignment, calibrated negatives), not raw pathogenicity prediction —
this is the differentiator from ProteinGym, PEER, InterPLM and Adams et al.
(see `r0_shared_interpretability_framework/README.md`).

Do **not** revive the superseded standalone claims (SAE+LLR beating
AlphaMissense, low-homology rescue, VAMP/abundance advantage, statistically
typed AM-vs-SAE blind spots). These all failed their gates; see
`docs/PROJECT_STATUS.md`.

## Source material to salvage

- Archived standalone R1 draft (Sections 3–4 reusable as Dataset + Calibrated
  Audit): `archive/manuscripts/nature_methods_r1_variant_perturbation/`
- Encoder benchmark scaffold + v0 tables: `r0_shared_interpretability_framework/results/v0_20260515/encoder/`
- IndelMissense v1 / v1.1 coordinates: `data/indelmissense/`
- Evidence files: `r1_encoder_interpretability_benchmark/results/variant_effect/`,
  `r1_encoder_interpretability_benchmark/results/annotation_alignment/`

## Planned additions (per adopted three-paper plan)

- CRPI metric, AnnoSaliency hybrid method, VUS reclassification audit.
- Method comparison across ~6 encoder-side interpretation methods on common axes.

Use the Springer Nature `sn-jnl` class files copied into
`r2_decoder_sparse_readout_audit/manuscript/template/` as the LaTeX starting
point when drafting.
