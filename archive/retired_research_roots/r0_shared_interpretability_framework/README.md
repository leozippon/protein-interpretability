# ProteinInterpret

ProteinInterpret is the shared benchmark scaffold for the two architecture-level
interpretability benchmarks in this repository:

- **Encoder-only benchmark:** variant and indel explanations for encoder or
  masked protein language models such as ESM-2.
- **Decoder-only benchmark:** explanation, conservation, diagnostic and
  intervention tests for autoregressive protein generators such as ProtGPT2,
  ZymCTRL and ProGen2.

## Relationship to active and future work

- **Paper B (encoder benchmark)** is built from this framework's encoder
  instance plus the ESM-2 SAE variant work and IndelMissense in
  `r1_encoder_interpretability_benchmark/`. See `r1_encoder_interpretability_benchmark/manuscript/README.md`.
- **Paper C (decoder benchmark)** is the future decoder instance, seeded by the
  R2 evidence in `r2_decoder_sparse_readout_audit/` (Paper A, the standalone npj Artificial
  Intelligence audit). Paper A and Paper C must keep a clean content boundary:
  A is the focused audit; C is the broad multi-method benchmark.

The current `v0_20260515` tables are not a final leaderboard. They standardize
the existing R1/R2 artefacts into a benchmark-shaped manifest so the next
experiments can be added without rewriting the result ledger.

## Layout

- `docs/BENCHMARK_SCHEMA.md`: shared result schema, benchmark axes, and current
  framework boundary.
- `scripts/build_v0_benchmark_tables.py`: builds the current v0 tables from
  existing local result files.
- `scripts/indelmissense_v11_resource_preflight.py`: audits whether
  IndelMissense v1.1 coordinates are compatible with SNV-centric external
  resources such as dbNSFP/REVEL.
- `scripts/encoder_mechanism_localization_preflight.py`: verifies whether
  mechanism-selected SAE features have residue-level firing examples that can
  be length-normalized for later localization, occlusion and CRPI metrics.
- `results/v0_20260515/encoder/`: encoder-only benchmark status and current
  result tables.
- `results/v0_20260515/decoder/`: decoder-only benchmark status and current
  result tables.

## Current Scope

The encoder v0 table emphasizes predictive utility, feature annotation coverage
and known negative gates because those are already available. The next missing
benchmark axes are localization, faithfulness, robustness and protein-specific
metrics such as CRPI.

The decoder v0 table emphasizes procedure-specific cross-model recurrence,
characterization, candidate checkpoint sensitivity, N-terminal readouts with
high unnormalized received attention, generation-metric calibration and failed
causal/steering gates. The next missing benchmark axes are broader method
comparison and model-diffing tasks.
