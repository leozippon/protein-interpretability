# R1: Encoder Interpretability Benchmark

R1 contains the encoder-only protein-language-model interpretability benchmark
and Paper B materials: ESM-2 sparse autoencoders, annotation alignment,
variant/indel evaluation, IndelMissense packaging, calibrated negative results,
and the manuscript scaffold.

## Evidence boundary

The project evaluates interpretation quality—localization, faithfulness,
annotation alignment, robustness, and calibrated negatives—not a claim that an
SAE scalar score beats specialist pathogenicity predictors. Do not revive
SAE+LLR beating AlphaMissense, low-homology rescue, VAMP/abundance advantage,
or statistically typed AlphaMissense-versus-SAE blind spots; those gates were
negative.

## Layout

- `docs/RESEARCH_PLAN.md`: current bounded research plan.
- `docs/EXPERIMENT_LOG.md`: chronological experiment record.
- `docs/methods/`: technical method notes.
- `docs/legacy/`: superseded R1-only notes retained for provenance.
- `literature/`: locally retained background papers.
- `manuscript/`: Paper B manuscript scaffold.
- `configs/`, `scripts/`, and `src/`: executable project implementation.
- `results/` and `logs/`: generated B-only artifacts covered by generic ignore
  rules.

Shared benchmark tables and schema live in
`../r0_shared_interpretability_framework/`. Repository-wide status is in
`../docs/PROJECT_STATUS.md`.
