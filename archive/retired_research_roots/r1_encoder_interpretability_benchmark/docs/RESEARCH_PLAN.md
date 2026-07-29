# Paper B research direction: encoder interpretation benchmark and calibrated audit

**Updated:** 2026-07-16  
**Status:** benchmark scaffold complete; manuscript drafting pending

## Research question

How should sparse-feature explanations for encoder protein language models be
evaluated for predictive utility, localization, faithfulness and robustness
without conflating interpretation quality with scalar pathogenicity accuracy?

## Scope

This direction combines ESM-2 sparse-autoencoder analyses, IndelMissense v1.1
and the encoder instance of `ProteinInterpret`. The benchmark records positive
and negative results under a common schema and emphasizes explanation quality.

## Defensible findings

- IndelMissense provides a bounded clinical protein-indel resource with
  coordinate augmentation and reproducible baseline tables.
- Annotation-selected sparse features can complement internal ESM-2 baselines
  in some cohorts, but accessible specialist predictors remain stronger for
  raw missense pathogenicity.
- Protein-level mechanism classification, low-homology rescue, abundance
  advantage and AlphaMissense-versus-SAE disagreement typing failed their
  pre-specified gates.

## Evidence boundaries

Do not claim proteome-scale LOF/GOF/dominant-negative mechanism prediction,
superiority to AlphaMissense, or clinically validated explanations. The next
research contribution is a calibrated interpretation benchmark, not a new
state-of-the-art pathogenicity score.

## Canonical materials

- Manuscript scaffold: `r1_encoder_interpretability_benchmark/manuscript/`
- Experiment log: `r1_encoder_interpretability_benchmark/docs/EXPERIMENT_LOG.md`
- Shared benchmark schema: `r0_shared_interpretability_framework/framework/README.md`
- Current project status: `docs/PROJECT_STATUS.md`

The pre-pivot mechanism-classification proposal is retained only for provenance
at `archive/legacy/PAPER_B_PRE_PIVOT_MECHANISM_PROPOSAL_20260330.md`.
