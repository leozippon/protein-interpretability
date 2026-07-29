# R2 Attention-Output Sparse Pilot (2026-05-14)

## Purpose

This pilot tested the proposed R2 redesign direction: move away from the
existing MLP-output CLTs and train a sparse model directly on attention-output
tensors. The goal was to check whether an attention-proximal sparse dictionary
can recover stronger N-terminal attention-sink readouts and whether those
features behave like causal handles.

## Implementation

Script:

- `Research2/scripts/43_attention_output_transcoder_pilot.py`

Local output:

- `Research2/results/circuit_analysis/attention_output_transcoder_pilot_20260514_l23/`

Remote output:

- `/oss-pvc/zhk_zip/biocc/Research2/results/circuit_analysis/attention_output_transcoder_pilot_20260514_l23/`

Runtime log:

- `Research2/logs/runtime/r2_attention_output_transcoder_pilot_20260514_l23.log`

Model and layer:

- ZymCTRL
- Attention layer 23
- Target tensor: input to `attn.c_proj`, i.e. concatenated attention-head
  outputs before the attention output projection.

Sparse model:

- `d_sae = 2048`
- `k = 32`
- 3,000 train steps
- 5,000 ZymCTRL / Swiss-Prot EC-labeled training records
- 120 evaluation sequences
- 80 ablation sequences

## Result

### Reconstruction / Readout

The pilot learned a usable attention-output sparse representation:

- Mean evaluation FVU: **0.4782**
- Best attention-received correlation: **0.9815**
- Best first-two-residue activation delta: **8.0173**
- Best feature: layer 23, feature 1671
- Evaluation alive fraction: **0.0869**

Interpretation:

- The readout gate passes: attention-output sparse features can cleanly identify
  N-terminal attention-sink-associated behavior in ZymCTRL layer 23.
- The low evaluation alive fraction shows that the dictionary is still highly
  sparse / partly dead, so a scaled version should improve the training recipe
  before making broad claims.

### Ablation

The causal result is negative:

| Condition | n | dNLL all | dNLL first2 | dNLL pos2-10 | dNLL 11+ |
|---|---:|---:|---:|---:|---:|
| target_L23 | 80 | -0.00202 | -0.35771 | -0.05081 | -0.00017 |
| random1_L23 | 80 | 0.00014 | 0.01580 | 0.00414 | -0.00002 |
| random2_L23 | 80 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| random3_L23 | 80 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

Interpretation:

- Ablating the top N-terminal attention-output features does **not** produce the
  expected N-terminal likelihood damage.
- The target ablation changes N-terminal likelihood in the opposite direction
  from the desired causal-gate behavior.
- Therefore this single-layer ZymCTRL attention-output pilot does not rescue the
  causal attention-sink mechanism claim.

## Conclusion

The redesign direction is partially validated:

- **Positive:** attention-output sparse models are much closer to the
  attention-sink phenomenon than the old MLP-output CLTs as readouts.
- **Negative:** the first causal ablation is not supportive.

Recommended next decision:

- If the goal is a stronger R2 mechanism paper, run one more bounded pilot with
  improved training quality before scaling:
  - include layers 12, 23 and 30;
  - increase training steps and/or dictionary health controls;
  - add feature-drop measurement after ablation;
  - compare delta-patch ablation against direct head-set ablation.
- If that second pilot also gives reverse or null causal effects, stop the
  attention-transcoder redesign for the current manuscript and keep R2 as a
  conserved readout / checkpoint diagnostic paper.
