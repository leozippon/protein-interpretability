# Opus Rescue Plan Execution Results (2026-05-18 plan)

This packet records execution of the latest Opus rescue proposals:
`OPUS_NEXT_20260518.md` and `OPUS_R1_RESCUE_20260518.md`.

## Executive Summary

- **R2 E-1 direct attention-head sink ablation: FAIL.**
  The selected heads are strong N-terminal sink heads, but directly zeroing a
  single paired head did not raise N-terminal NLL by the required margin and did
  not reduce T011/T018/T023 activation. This rules out the current
  single-head causal-sink story.
- **R1-Save-3 extended disagreement typing: FAIL.**
  Adding protein-level contexts increased the audit from 22 to 90 tests, but no
  context survived BH q < 0.05. The closest SAE-right enrichment was
  `mechanism_missing` with q = 0.0717, below the reporting threshold.
- **R1-Save-2 abundance rescue proxy: FAIL.**
  On nine staged ProteinGym / VAMP-like abundance assays, AlphaMissense or LLR
  remains stronger than the SAE-family signal. SAE-family beats AM on only 1/9
  usable assays, below the >=3 assay proxy gate.
- **R1-Save-1 low-MSA stratification was not launched yet.**
  UniRef50 FASTA and DIAMOND tarballs are staged, but there is no cached MSA
  depth / DIAMOND database table. Given the new negative R1-Save-2 and
  R1-Save-3 results, this should be an explicit Opus decision before spending
  additional compute.

## R2 E-1: Direct Attention-Head Sink Ablation

Script:

- `Research2/scripts/41_attention_head_sink_ablation.py`

Outputs:

- `Research2/results/circuit_analysis/attention_head_sink_ablation_20260518/`
- `Research2/logs/runtime/r2_attention_head_sink_ablation_20260518.log`

Design:

- Models: ProtGPT2 v2, ZymCTRL v2, ProGen2-medium.
- Cohort: 200 `swissprot_n1` sequences from the triplet-characterization
  cohort.
- Targets: T011, T018, T023.
- Controls: random same-layer heads, T025 specificity, and sham.
- Intervention: zero one attention-head slice before the attention output
  projection.
- Gate: at least two models must show paired-head ablation with
  `delta NLL pos2-10 >= 0.5` and `feature drop >= 0.5`, beyond random controls.

Main target rows:

| Model | Triplet | selected head | first2 mass | corr(feature) | dNLL pos2-10 | feature drop | gate |
|---|---|---:|---:|---:|---:|---:|---|
| ProGen2-medium | T011 | L11H3 | 0.943 | 0.403 | -0.0011 | 0.0000 | fail |
| ProGen2-medium | T018 | L11H3 | 0.943 | 0.399 | -0.0011 | 0.0000 | fail |
| ProGen2-medium | T023 | L11H3 | 0.943 | 0.346 | -0.0011 | -0.0001 | fail |
| ProtGPT2 | T011 | L4H15 | 0.966 | 0.624 | 0.0005 | -0.0002 | fail |
| ProtGPT2 | T018 | L4H15 | 0.966 | 0.625 | 0.0005 | -0.0003 | fail |
| ProtGPT2 | T023 | L32H16 | 0.954 | 0.172 | 0.0004 | 0.0000 | fail |
| ZymCTRL | T011 | L23H19 | 0.896 | -0.083 | 0.0071 | 0.0000 | fail |
| ZymCTRL | T018 | L23H19 | 0.896 | 0.049 | 0.0071 | 0.0000 | fail |
| ZymCTRL | T023 | L23H19 | 0.896 | 0.145 | 0.0071 | -0.0053 | fail |

Interpretation:

- The selected heads are real N-terminal sink-like heads by attention-mass
  criteria.
- The paired CLT features are still best described as **correlates/readouts** of
  a distributed N-terminal attention-sink regime, not as single-head causal
  mechanisms.
- This result supports the conservative R2 venue path from
  `OPUS_NEXT_20260518.md`: correlation and quality-diagnostic framing, not Nat
  Methods-grade causal mechanism.

## R1-Save-3: Extended Disagreement Typing

Script:

- `Research1/scripts/47_extended_disagreement_typing.py`

Outputs:

- `Research1/results/variant_effect/extended_disagreement_typing_20260518/`

Added contexts:

- Protein length quartile.
- gnomAD pLI / LOEUF / missense z buckets.
- Existing gene-level mechanism labels.
- Pfam family-size proxy at the variant position.
- ESM2 LLR absolute-confidence bucket and directional LLR bucket.
- BLOSUM62 substitution conservativeness.

Result:

- Disagreements: 473.
- Review-filtered disagreements: 321.
- Outcome counts: AM right / SAE wrong = 283; SAE right / AM wrong = 38.
- Context tests: 90.
- BH-significant tests at q < 0.05: 0.
- Gate: FAIL.

Top near-hit:

- `SAE_right_AM_wrong` enriched for `mechanism_missing`:
  32/38 vs 162/283, odds ratio 3.98, q = 0.0717.

Unavailable Opus contexts:

- MSA depth quartile: no cached per-protein MSA depth table yet.
- IUPred disorder fraction: no staged IUPred output yet.

Interpretation:

- The expanded audit does not rescue a statistically typed AM-vs-SAE blind-spot
  claim.
- Residual cases should remain illustrative triage examples.

## R1-Save-2: VAMP / Abundance Proxy

Script:

- `Research1/scripts/46_vampseq_abundance_proxy.py`

Outputs:

- `Research1/results/variant_effect/vampseq_abundance_proxy_20260518/`

Design:

- Reused the already completed R1 ProteinGym SAE benchmark.
- Joined staged AlphaMissense amino-acid substitution scores to nine
  ProteinGym / VAMP-like abundance assays.
- Compared abundance-oriented Spearman correlations for AM, LLR, SAE damage,
  and SAE+LLR ensemble.
- This is a proxy, not a full new SAE rescoring run.

Result:

- Assays requested: 9.
- Assays with at least 30 AlphaMissense matches: 9.
- SAE-family beats AM: 1/9.
- Proxy gate: FAIL.

| Assay | AM abundance rho | LLR rho | SAE abundance rho | SAE+LLR rho | best |
|---|---:|---:|---:|---:|---|
| PTEN Matreyek 2021 | 0.477 | 0.265 | -0.000 | 0.287 | AM |
| TPMT Matreyek 2018 | 0.558 | 0.476 | 0.186 | 0.290 | AM |
| MSH2 Jia 2020 | 0.416 | 0.296 | 0.077 | 0.203 | AM |
| NUDT15 Suiter 2020 | 0.680 | 0.583 | 0.248 | 0.378 | AM |
| VKOR1 abundance | 0.483 | 0.485 | 0.321 | 0.151 | LLR |
| CYP2C9 abundance | 0.598 | 0.626 | 0.311 | 0.302 | LLR |
| GCK/HXK4 abundance | 0.401 | 0.376 | 0.127 | 0.253 | AM |
| KRAS/RASK abundance | 0.123 | 0.231 | 0.028 | 0.130 | LLR |
| SLC22A1 abundance | 0.400 | 0.599 | 0.190 | 0.395 | LLR |

Interpretation:

- The staged abundance data do not support the claim that SAE perturbation
  features outperform AM on protein abundance.
- A full GPU rerun is unlikely to change the conclusion unless Opus specifies a
  substantially different abundance phenotype or SAE score definition.

## Remaining Decision

Only R1-Save-1 remains from the R1 rescue list. It requires building or staging
a per-protein MSA-depth table from UniRef50/DIAMOND and then rerunning
stratified ClinVar/CancerHoldout analysis.

Given the new negatives:

- R2 E-1 failed the last causal-mechanism gate.
- R1-Save-2 proxy failed on abundance assays.
- R1-Save-3 failed on expanded disagreement contexts.

Recommended next decision for Opus:

1. Freeze R2 at the conserved N-terminal attention-sink-associated sparse
   feature + checkpoint-quality diagnostic framing.
2. Freeze R1 at the IndelMissense v1 resource / calibrated audit framing.
3. Run R1-Save-1 only if Opus believes low-MSA stratification is worth the
   remaining compute despite two other R1 rescue failures.
