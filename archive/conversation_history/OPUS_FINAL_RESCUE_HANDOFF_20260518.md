# Final Rescue Handoff for Opus

Prepared after the user's request to make one more serious rescue attempt for
both R1 and R2.

## Bottom Line

The final rescue attempts did **not** recover the original high-impact claims.
The projects are not scientifically worthless, but the Nature Methods-grade
stories are now exhausted.

- **R1:** no evidence that SAE+LLR has a defensible AlphaMissense-complementary
  pathogenicity scope under low-homology stratification.
- **R2:** no evidence that T011/T018/T023 identify causal attention-sink heads,
  either as single heads or as multi-head sink sets.

## R1 Final Rescue: Low-Homology Stratification

Script:

- `Research1/scripts/45_low_homology_stratification.py`

Output:

- `Research1/results/variant_effect/low_homology_stratification_20260518/`

Design:

- Used staged UniRef50 representative cluster size (`n=` header field) as a
  low-homology proxy.
- Covered 1,892 / 1,972 variants and 991 proteins.
- Tested whether SAE+LLR becomes competitive with AlphaMissense in the lowest
  homology quartile.

Result:

- **Gate: FAIL**
- Low-homology quartile:
  - AlphaMissense AUC: 0.9627 [0.9418, 0.9779]
  - SAE+LLR z-ensemble AUC: 0.8727 [0.8327, 0.9067]
  - Delta SAE+LLR minus AM: -0.0900 [-0.1253, -0.0568]
- High-homology quartile:
  - AlphaMissense AUC: 0.9387 [0.9149, 0.9570]
  - SAE+LLR z-ensemble AUC: 0.8645 [0.8322, 0.8955]
  - Delta SAE+LLR minus AM: -0.0742 [-0.1072, -0.0396]

Interpretation:

- The low-homology rescue hypothesis is not supported.
- A full DIAMOND/Meff run could refine the proxy, but the low-quartile gap is
  large enough that reversal is unlikely.
- R1 should not claim SAE+LLR competitiveness against AlphaMissense.

## R2 Final Rescue: Multi-Head Sink-Set Ablation

Script:

- `Research2/scripts/42_attention_sink_set_ablation.py`

Outputs:

- `Research2/results/circuit_analysis/attention_sink_set_ablation_20260518/`
- `Research2/results/circuit_analysis/attention_sink_set_ablation_top32_20260518/`

Design:

- Tested whether the failed single-head ablation missed a distributed
  N-terminal attention-sink mechanism.
- Ablated top N-terminal sink-head sets and compared against same-layer random
  head sets.
- Ran both top-8 and top-32 sink-head interventions.

Top-8 result:

| Model | heads | dNLL pos2-10 | random dNLL | feature drop | random drop | gate |
|---|---:|---:|---:|---:|---:|---|
| ProGen2-medium | 8 | 0.0020 | 0.0008 | -0.0012 | 0.0001 | FAIL |
| ProtGPT2 | 8 | 0.0002 | 0.0009 | -0.0005 | -0.0002 | FAIL |
| ZymCTRL | 8 | 0.0406 | -0.0219 | -0.0035 | -0.0044 | FAIL |

Top-32 result:

| Model | heads | dNLL pos2-10 | random dNLL | feature drop | random drop | gate |
|---|---:|---:|---:|---:|---:|---|
| ProGen2-medium | 32 | 0.0050 | 0.0117 | -0.0006 | 0.0068 | FAIL |
| ProtGPT2 | 32 | 0.0062 | 0.0032 | -0.0012 | -0.0000 | FAIL |
| ZymCTRL | 32 | 0.0161 | 0.1652 | -0.0105 | -0.0250 | FAIL |

Interpretation:

- The distributed-head rescue is not supported.
- Ablating many sink-like heads does not reduce T011/T018/T023 activation.
- In the top-32 ZymCTRL run, random same-layer heads affect NLL more than the
  sink set, arguing against specificity.
- R2 should drop causal attention-sink mechanism claims.

## Surviving Claims

### R1

Defensible framing:

> IndelMissense v1: an interpretable protein-indel benchmark and calibrated
> audit of sparse-feature variant interpretation.

Surviving positives:

- IndelMissense v1 resource.
- Combined SAE+ESM+cheap grouped LR AUC 0.9108 on indels.
- Honest negative audit against AlphaMissense, gMVP, abundance assays,
  mechanism labels, and low-homology stratification.

Do not claim:

- SAE+LLR beats or matches AlphaMissense.
- Low-homology rescue.
- VAMP/abundance advantage.
- Statistically typed AM-vs-SAE blind spots.

### R2

Defensible framing:

> Conserved sparse-feature readouts identify N-terminal attention-sink-like
> behavior and provide a checkpoint-quality diagnostic across protein language
> models.

Surviving positives:

- 38 cross-model conserved triplets.
- T011/T018/T023 form a strong N-terminal attention-sink-associated subset.
- Universal triplet count functions as an early-vs-mature checkpoint-quality
  diagnostic.

Do not claim:

- Single-feature causality.
- Single-head causal sink mechanism.
- Distributed-head causal sink mechanism.
- Biological primitive dictionary.

## Questions for Opus

1. Should R1 now be frozen as a resource/audit paper, with target venue Genome
   Biology / NAR Database / Bioinformatics?
2. Should R2 be frozen as a model-diagnostic/readout paper, with Nature Machine
   Intelligence only if Opus believes the diagnostic framing is strong enough?
3. Should all further rescue experiments stop, given that the final
   low-homology and multi-head causal tests both failed?

## Recommendation

Stop rescue experiments. Rewrite around the surviving claims and calibrated
negative results. Further experiments are more likely to add redundant negative
evidence than to create a new Nature-family claim.
