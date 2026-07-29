# M-1 Triplet Characterization Results (2026-05-14)

This packet reports execution of the only remaining R2 experiment requested in
`OPUS_NEXT_20260514.md`.

## Executive Read

- M-1 completed successfully and passed the Opus acceptance gate.
- Cohort: 700 proteins, consisting of the 500 Swiss-Prot N-1 cohort plus 200
  balanced calibration / UniRef50 records.
- Triplets tested: 38 conserved cross-model triplets from the balanced atlas.
- Acceptance target: at least 25 / 38 triplets with one significant
  characterization test after BH correction at q < 0.05.
- Result: 37 / 38 triplets were categorized, so the gate is PASS.
- Category assignments: 21 k-mer, 14 positional, 2 high-norm, 0
  attention-sink-only, 0 BPE-boundary-only, 1 unknown.

The result supports the reframed R2 story: the conserved triplets are real
statistical objects, but most characterize as local sequence/position or
activation-scale patterns rather than named biological primitives.

## Outputs

- Script:
  `Research2/scripts/35_triplet_characterization.py`
- Main summary:
  `Research2/results/circuit_analysis/triplet_characterization_20260514/summary.md`
- Machine-readable summary:
  `Research2/results/circuit_analysis/triplet_characterization_20260514/summary.json`
- Per-triplet statistics:
  `Research2/results/circuit_analysis/triplet_characterization_20260514/triplet_characterization.tsv`
- Top firing positions:
  `Research2/results/circuit_analysis/triplet_characterization_20260514/top_firing_positions.tsv`
- Cohort copy:
  `Research2/results/circuit_analysis/triplet_characterization_20260514/cohort.json`
- Runtime log:
  `Research2/logs/runtime/r2_triplet_characterization_20260514_v2.log`

## Methods

For each of ProtGPT2, ZymCTRL, and ProGen2-medium, the script loads the model
and corresponding CLT, extracts residual activations at the triplet layer,
computes CLT feature activations, and maps token-level signals back to residue
positions. For each triplet, per-model activation vectors are z-normalized and
averaged into a consensus positional activation score.

The five characterization tests are:

1. k-mer enrichment: mutual information between top-firing positions and
   centered 3-mer / 5-mer identities under random top-position permutations.
2. Positional distribution: one-sample KS statistic for normalized
   position-in-protein against Uniform(0, 1).
3. BPE-boundary enrichment: ProtGPT2-only one-sided enrichment of top firing
   at token start/end residues.
4. Attention-sink correlation: one-sided positive Pearson correlation between
   triplet activation and mean attention received at the same layer.
5. High hidden-norm correlation: one-sided positive Pearson correlation between
   triplet activation and residual hidden-state L2 norm.

All p-values are BH-corrected across all triplet/test pairs. Assignment uses
the most significant passing category per triplet.

## Result Details

Assigned categories:

| Category | Count |
|---|---:|
| k-mer | 21 |
| positional | 14 |
| high-norm | 2 |
| unknown | 1 |

Significance by test, regardless of final assigned category:

| Test | Significant triplets, q < 0.05 |
|---|---:|
| k-mer | 25 |
| positional | 35 |
| BPE-boundary enrichment | 1 |
| attention-sink correlation | 4 |
| high hidden-norm correlation | 25 |

The only unknown triplet is T037. It has no significant k-mer, positional,
BPE-boundary, attention-sink, or high-norm association after BH correction.

The four attention-sink-significant triplets are T011, T018, T023, and T025,
but each is assigned to k-mer because k-mer is at least as significant and
appears earlier in the tie-break order. T011, T018, and T023 show very large
activation/attention correlations (r approximately 0.92, 0.91, and 0.89).

The BPE-boundary test was corrected during execution. The first run used a
two-sided boundary-association statistic, which incorrectly counted strong
depletion from token boundaries as a boundary effect. The final v2 run uses the
intended one-sided positive boundary-enrichment statistic. The final category
counts are unchanged; only BPE q-values changed. After correction, only T017
has significant positive BPE-boundary enrichment.

## Scientific Interpretation

M-1 cleanly answers the remaining R2 question from Opus. The triplets are not
uncharacterized noise: 37 / 38 have at least one measurable statistical
signature. However, those signatures are mostly local sequence context,
position-in-protein, or high hidden-state norm. This is consistent with the
current R2 reframe:

1. Cross-model CLT triplets are statistically conserved.
2. They fail rich biological annotation and downstream representation gates.
3. Their measurable structure is mostly non-biological or low-level model /
   sequence statistics.

The result does not resurrect the "universal biological primitives" thesis.
It does make the cautionary-methods paper stronger, because it shows that the
conserved latents are explainable in simpler statistical terms rather than
being mysterious biological features.

## Recommended Manuscript Use

For R2, use M-1 as the third pillar after the null-control conservation result
and the negative biological/downstream probes:

- "Of 38 statistically conserved cross-model triplets, 37 were assignable to
  simple statistical categories under pre-specified tests: 21 k-mer, 14
  positional, and 2 high hidden-norm. Only one remained uncategorized."
- "These results indicate that cross-model conservation in CLT latents can
  reflect stable sequence/model statistics without implying convergence on
  named biological primitives."

Do not use M-1 to claim biological mechanism, drug design utility, or compact
downstream representation quality.

## Current Compute State

The M-1 process has finished. The current 1-GPU H200 pod is idle after the run;
inside the pod `nvidia-smi` reports 0 MiB GPU memory and 0% GPU utilization.
