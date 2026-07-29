# Final Rescue Results for R1 and R2

This packet records the final rescue attempt requested after
`OPUS_RESCUE_EXECUTION_20260518.md`.

## Venue Reality

Nature Machine Intelligence is **not** an easier venue than Nature Methods.
The difference is fit:

- Nature Methods needs a broadly useful method or resource with strong
  validation and clear practical adoption.
- Nature Machine Intelligence can be a better fit for model-behavior,
  representation, and diagnostic-tool claims, but it still requires a clean ML
  insight and strong evidence.

The latest experiments do not make either paper an easy Nature-family
submission. They clarify which claims are still defensible.

## R1 Final Rescue: Low-Homology Stratification

Script:

- `Research1/scripts/45_low_homology_stratification.py`

Output:

- `Research1/results/variant_effect/low_homology_stratification_20260518/`

Design:

- Tested whether SAE+LLR becomes competitive with AlphaMissense in proteins
  with low homologous coverage.
- Used staged UniRef50 representative cluster size (`n=` header field) as a
  low-homology proxy.
- This is not full DIAMOND Meff, but it covers 1,892 / 1,972 matched variants
  and 991 proteins.

Result:

- Gate: **FAIL**.
- Low-homology quartile:
  - AlphaMissense AUC: 0.9627 [0.9418, 0.9779]
  - SAE+LLR z-ensemble AUC: 0.8727 [0.8327, 0.9067]
  - Delta SAE+LLR minus AM: -0.0900 [-0.1253, -0.0568]
- High-homology quartile:
  - AlphaMissense AUC: 0.9387 [0.9149, 0.9570]
  - SAE+LLR z-ensemble AUC: 0.8645 [0.8322, 0.8955]
  - Delta SAE+LLR minus AM: -0.0742 [-0.1072, -0.0396]

Interpretation:

- The low-homology rescue hypothesis is not supported under the staged UniRef50
  proxy.
- A full DIAMOND/Meff run could refine the depth metric, but the low-quartile
  gap is large enough that it is unlikely to reverse the conclusion.
- R1 should not claim a Nat-Methods-grade SAE+LLR applicability scope against
  AlphaMissense.

## R2 Final Rescue: Multi-Head Sink-Set Ablation

Script:

- `Research2/scripts/42_attention_sink_set_ablation.py`

Outputs:

- `Research2/results/circuit_analysis/attention_sink_set_ablation_20260518/`
- `Research2/results/circuit_analysis/attention_sink_set_ablation_top32_20260518/`

Design:

- Tested whether the failed single-head intervention missed a distributed
  N-terminal attention-sink mechanism.
- Ablated top N-terminal sink-head sets and compared against same-layer random
  head sets.
- Two runs:
  - top-8 sink heads, 3 random-set controls.
  - top-32 sink heads, 1 random-set control.

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

- The multi-head distributed-sink rescue is not supported.
- Ablating many sink-like heads still does not reduce T011/T018/T023 activation.
- In the top-32 ZymCTRL run, random same-layer heads affect NLL more than the
  sink set, arguing against specificity.
- R2 should drop causal attention-sink mechanism claims. The remaining
  defensible claim is that T011/T018/T023 are conserved N-terminal
  attention-sink-associated readouts / diagnostics.

## Final Judgment

R1 and R2 are not worthless, but the original high-impact claims are exhausted.

### R1 defensible paper

Best framing:

> IndelMissense v1: an interpretable protein-indel benchmark and calibrated
> audit of sparse-feature variant interpretation.

Core positive:

- IndelMissense v1 resource.
- Combined SAE+ESM+cheap grouped LR AUC 0.9108 on indels.
- Honest calibration showing where SAE does not beat AlphaMissense.

Do not claim:

- SAE+LLR beats AlphaMissense.
- Low-homology rescue.
- VAMP/abundance advantage.
- Statistically typed AM-vs-SAE blind spots.

### R2 defensible paper

Best framing:

> Conserved sparse-feature readouts identify N-terminal attention-sink-like
> behavior and provide a checkpoint-quality diagnostic across protein language
> models.

Core positive:

- 38 cross-model conserved triplets.
- T011/T018/T023 form a strong N-terminal attention-sink-associated subset.
- Early-vs-mature checkpoint diagnostic recovers fewer universal triplets in
  weak checkpoints.

Do not claim:

- Single-feature causality.
- Single-head or distributed-head causal sink mechanism.
- Biological primitive dictionary.

## Recommended Stop Rule

Stop rescue experiments for both papers. Additional runs are now more likely to
produce variants of the same negative conclusion than a new defensible Nature
claim.

The constructive path is to freeze the claims, rewrite the manuscripts around
the surviving contributions, and target venues by evidence fit rather than
prestige:

- R1: Genome Biology / NAR Database / Bioinformatics resource track.
- R2: Nature Machine Intelligence only if framed as a rigorous ML diagnostic
  with calibrated negative interventions; otherwise Cell Patterns / ML for
  biology venues.
