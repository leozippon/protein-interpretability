# Opus TODO Execution Results (2026-05-13)

This packet summarizes the low-risk TODOs requested after `OPUS_NEXT_20260513.md`.

## Executive Read

- R1 N-4 is complete and provides a manuscript-facing AlphaMissense-vs-SAE residual case table.
- R2 N-3 is complete and strengthens the statistical conservation claim: observed triplets remain far above a 30-replicate permutation null.
- R2 N-1 is complete and fails the rich-label interpretation gate: 0 / 38 triplets reach MI >= 0.1 nats on the Swiss-Prot anchored cohort.
- R2 N-2 is complete and fails the fallback representation gate: the 38-dim triplet basis does not match ESM-2 mean-pooled embeddings on Pfam-family, EC-topclass, or secondary-structure-fraction probes.

## R1 N-4: SAE Residual Case Studies

Output:
- `Research1/results/variant_effect/sae_residual_case_studies_20260513.md`
- `Research1/results/variant_effect/sae_residual_case_studies_20260513.tsv`

Result:
- AlphaMissense-confident wrong candidates: 179.
- Candidates with ClinVar review stars >= 2: 112.
- Curated cases shown: 30.
- The table includes both SAE-rescued and SAE-not-rescued cases and should be treated as computational triage, not wet-lab validation.

Implication:
- This supports the R1 "SAE as interpretation/diagnostic layer" figure.
- It does not rescue scalar pathogenicity versus AlphaMissense, and it should not be framed as a competing scorer.

## R2 N-3: 30-Replicate Permutation Null

Output:
- `Research2/results/circuit_analysis/universal_atlas_balanced200_wide_null_control_30x_20260513.json`
- `Research2/results/circuit_analysis/universal_atlas_balanced200_wide_null_control_30x_20260513.md`

Observed exact three-model triplets:

| Threshold | Observed | Null mean | Null std | Null max | Null 99% percentile CI |
|---:|---:|---:|---:|---:|---:|
| 0.90 | 38 | 0.0667 | 0.2494 | 1 | [0.0, 1.0] |
| 0.95 | 30 | 0.0667 | 0.2494 | 1 | [0.0, 1.0] |
| 0.98 | 8 | 0.0333 | 0.1795 | 1 | [0.0, 0.855] |

Implication:
- The cross-model triplets are statistically non-random.
- The original "38 vs null=0" wording should be softened to "38 vs null mean 0.067, max 1 over 30 permutations."
- This supports a conservation/statistical claim, not a biological interpretation claim by itself.

## R2 N-1: Swiss-Prot Anchored Rich-Label Annotation

Output:
- `Research2/results/circuit_analysis/swissprot_triplet_annotation_20260513/summary.json`
- `Research2/results/circuit_analysis/swissprot_triplet_annotation_20260513/interpretation.md`
- `Research2/results/circuit_analysis/swissprot_triplet_annotation_20260513/interpretation_gate.tsv`
- `Research2/results/circuit_analysis/swissprot_triplet_annotation_20260513/rich_label_mi.tsv`
- `Research2/results/circuit_analysis/swissprot_triplet_annotation_20260513/per_triplet_max_act_rich.tsv`

Setup:
- Cohort: 500 Swiss-Prot proteins, length 100-400.
- Sampling: round-robin over dominant Pfam families.
- Labels: Pfam family, dominant Pfam, Swiss-Prot residue categories, feature types, secondary structure, functional/domain/PTM/topology/region labels, amino-acid identity.
- Gate: at least 25 / 38 triplets with best rich-label MI >= 0.1 nats.
- Implementation note: bootstrap was vectorized and run with `n_boot=30` after the original full-position Python bootstrap proved too slow.

Result:
- Interpretable triplets: 0 / 38.
- Outcome: FAIL.
- Max best-rich MI: 0.00454378 nats.
- Most best-rich labels are Pfam-family or dominant-Pfam proxies, not functional residue annotations.

Implication:
- The Swiss-Prot cohort fixed the resource-coverage issue, but the triplets still do not map cleanly to the planned rich biological labels.
- The "universal biological primitives with names" thesis is not supported by this gate.
- Causal intervention A-3 should remain deferred unless Opus wants it purely as characterization of uninterpreted conserved latents.

## R2 N-2: 38-Dim Triplet Basis Probes

Output:
- `Research2/results/circuit_analysis/triplet_basis_probes_20260513/summary.json`
- `Research2/results/circuit_analysis/triplet_basis_probes_20260513/summary.md`
- Feature matrices are on the remote pod and local summary files are pulled; large matrices were not required for Opus review.

Setup:
- Union cohort: 820 proteins.
- Triplet basis: 820 x 38, consensus 95th-percentile firing strength over the three CLT models.
- Baseline: ESM-2 mean-pooled embeddings, 820 x 2560.
- Pfam clan is approximated by dominant Pfam family because no clan map is staged.
- EC labels use the staged EC-labeled Swiss-Prot FASTA: `data/zymctrl/ec_labeled.fasta`.

Results:

| Task | Label source | Triplet metric | ESM-2 metric | Gate result |
|---|---|---:|---:|---|
| Pfam family classification | dominant Pfam family proxy | macro-F1 0.3995 | macro-F1 1.0000 | fail |
| EC top-class classification | staged EC-labeled Swiss-Prot FASTA | macro-F1 0.2550 | macro-F1 0.7080 | fail |
| Secondary-structure fraction regression | Swiss-Prot secondary_structure features | R2 -0.6049 | R2 0.3245 | fail |

Implication:
- The triplet basis carries some signal above chance for Pfam family and EC top-class, but it is far weaker than ESM-2 and fails the "matches embeddings on >= 1 task" fallback gate.
- The fallback positive R2 claim is therefore not supported under the current task definitions.

## Code / Runtime Notes

New or modified scripts:
- `Research2/scripts/27_universal_atlas_null_control.py`: added null std and 99% percentile CI reporting.
- `Research2/scripts/33_swissprot_triplet_annotation.py`: new N-1 runner; includes Research1 pickle compatibility and vectorized MI/bootstrap.
- `Research2/scripts/34_triplet_basis_probes.py`: new N-2 probe runner.
- `Research1/scripts/41_curate_residual_cases.py`: new N-4 case-study curation runner.

Current H200 state:
- Arena job `jiaotongdamoxing-zhk-zip-final-1gpu-0511` still reserves 1 H200.
- No active N-1/N-2/N-3 process is running.
- GPU memory/utilization inside the pod: 0 MiB / 0%.

## Decision Packet For Opus

The honest interpretation is now sharper:

1. R1 can proceed with a narrowed story: accessible scalar baselines, IndelMissense v1, and SAE as an interpretation/diagnostic layer. N-4 supplies the missing residual-case table.
2. R2 retains a statistically strong cross-model conservation fact, but both biological naming (N-1) and low-dimensional downstream representation fallback (N-2) failed their gates.
3. A strong "universal biological primitive" or "useful compact representation" claim is not currently supported.
4. The defensible R2 alternatives are:
   - downgrade to a methods/negative-diagnostic paper on conserved but mostly uninterpreted CLT latents;
   - search for a different annotation target or cohort definition;
   - stop R2 expansion for this manuscript cycle and keep only the statistical conservation result as a supplementary/appendix result.

