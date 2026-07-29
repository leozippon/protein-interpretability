# R2 Attention-Sink Causal Ablation Results

Executed on the 1-GPU H200 hold pod after reviewing
`OPUS_BRILLIANT_FINAL_20260517.md`.

## Summary

The causal ablation experiment completed across all three protein LMs and all
planned controls. The result is **FAIL** under the pre-specified conservative
gate.

This means the current evidence supports:

> T011/T018/T023 are cross-model conserved sparse features whose top-firing
> events are strongly concentrated at the N-terminal edge and correlated with
> attention-sink statistics.

It does **not** yet support:

> T011/T018/T023 are causally required attention-sink mechanisms.

## Files

Script:

- `Research2/scripts/40_attention_sink_causal_ablation.py`

Local outputs:

- `Research2/results/circuit_analysis/attention_sink_causal_ablation_20260517/per_sequence_metrics.tsv`
- `Research2/results/circuit_analysis/attention_sink_causal_ablation_20260517/condition_summary.tsv`
- `Research2/results/circuit_analysis/attention_sink_causal_ablation_20260517/summary.md`
- `Research2/results/circuit_analysis/attention_sink_causal_ablation_20260517/summary.json`
- `Research2/results/circuit_analysis/attention_sink_causal_ablation_20260517/{protgpt2,zymctrl,progen2-medium}_conditions.json`

Remote log:

- `/oss-pvc/zhk_zip/biocc/Research2/logs/runtime/r2_attention_sink_causal_ablation_20260517.log`

## Experiment Design

Cohort:

- 200 `swissprot_n1` sequences from the M-1 characterization cohort.
- Max sequence length: 256.

Models:

- ProtGPT2 v2 CLT, step 200000.
- ZymCTRL v2 CLT, step 200000.
- ProGen2-medium CLT, step 100000.

Target triplets:

- T011.
- T018.
- T023.

Controls:

- T025 as a non-N-terminal attention-associated specificity control.
- Two same-layer random CLT features per target triplet per model.

Intervention:

- TopK-aware CLT same-layer MLP-output patch.
- Multiplier = 0.0 for the selected CLT feature.
- This is not a direct attention-head patch; any attention change is an
  indirect downstream effect.

Readouts:

- Teacher-forced delta NLL by residue-position bins.
- Main causal readout: `delta_nll_pos2_10` and
  `specificity_pos2_10_minus_11plus`.
- Attention-received redistribution summarized as downstream attention cosine
  distance and first-two-residue attention delta.
- Feature first-two-residue activation rate.

## Main Results

Per-sequence rows: 6000.

Outcome: **FAIL**.

| Model | Triplet | n | dNLL pos2-10 | dNLL 11+ | specificity | attention cosine | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| ProGen2-medium | T011 | 200 | -0.0000 | -0.0002 | 0.0001 | 0.0000 | false |
| ProGen2-medium | T018 | 200 | 0.0002 | -0.0000 | 0.0002 | 0.0000 | false |
| ProGen2-medium | T023 | 200 | 0.0003 | 0.0001 | 0.0003 | 0.0000 | false |
| ProtGPT2 | T011 | 200 | 0.0002 | -0.0002 | 0.0004 | 0.0000 | false |
| ProtGPT2 | T018 | 200 | 0.0000 | -0.0007 | 0.0008 | 0.0000 | false |
| ProtGPT2 | T023 | 200 | 0.0001 | -0.0001 | 0.0002 | 0.0000 | false |
| ZymCTRL | T011 | 200 | -0.0488 | 0.0000 | -0.0488 | 0.0000 | false |
| ZymCTRL | T018 | 200 | 0.0030 | -0.0000 | 0.0030 | 0.0000 | false |
| ZymCTRL | T023 | 200 | -0.0405 | 0.0000 | -0.0405 | 0.0000 | false |

Feature activation sanity check:

- ProtGPT2 T011/T018/T023 first-two active rate: 1.000 / 1.000 / 1.000.
- ZymCTRL T011/T018/T023 first-two active rate: 1.000 / 1.000 / 1.000.
- ProGen2-medium T011/T018/T023 first-two active rate: 0.985 / 0.985 / 0.985.

So the null result is not because the target features were inactive at the
N-terminal edge under this cohort.

## Interpretation

The causal gate failed for two reasons:

1. ProtGPT2 and ProGen2-medium show near-zero NLL and attention changes after
   feature ablation.
2. ZymCTRL shows sizable N-terminal NLL shifts for T011 and T023, but the
   direction is negative: ablation lowers NLL rather than raising it. This is
   inconsistent with the simple "feature is required for N-terminal prediction"
   story.

The attention redistribution readout is also essentially zero across models.
This may be because the intervention patches the CLT-explained MLP component
rather than directly modifying attention heads, but under the current causal
test it does not close the correlation-to-causation loop.

## Consequence for R2

The R2 attention-sink finding remains strong as a correlation/characterization
result:

- Three cross-model conserved triplets.
- Top-firing rows fully concentrated at the N-terminal edge.
- Strong attention-received correlation in the characterization analysis.
- Very strong N-terminal enrichment against background top-firing rows.

But the manuscript should not claim that these features are causally required
attention-sink mechanisms. The safe phrasing is:

> conserved N-terminal attention-sink-associated sparse features

not:

> causal initiator-methionine attention-sink circuits.

## Recommendation for Opus

Do not use the causal ablation as a Nat Methods-upgrading result.

Possible next decisions:

1. Accept the downgrade: write R2 as a focused correlation/discovery and
   quality-diagnostic paper, likely Nat Mach Intel / Cell Patterns rather than
   Nature Methods.
2. If Opus believes causality is still worth pursuing, design a more direct
   attention-head intervention rather than CLT MLP-output patching.
3. Keep R1 as a resource/audit paper and avoid further broad rescue
   experiments unless a sharply defined diagnostic is needed.
