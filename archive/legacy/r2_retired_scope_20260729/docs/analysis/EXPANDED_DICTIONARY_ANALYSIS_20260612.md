# R2 Expanded-Dictionary Downstream Analysis (2026-06-12)

> **Evidence-audit correction (2026-07-16).** The run outputs below are retained
> as provenance, but their capacity and controllability interpretations are
> superseded. None of the wider checkpoints met the preregistered FVU < 0.15
> quality prerequisite, and the logged quality values are not a matched
> held-out comparison. The wider run therefore does not falsify dictionary
> capacity or optimization. `R_raw` is an architecture-specific layer-normalized
> CLT input, not a raw residual stream. Script 46 derives a direction in that
> input space but injects it at MLP output and scores a motif/composition
> heuristic; its `distributed_or_robust` label is not a valid mechanistic
> conclusion. See EXP-R2-022 and `r2_interpretability_transfer/preregistration/DECISION_LOG.md`.

## Run

- Remote result root:
  `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/results/representation_audit_20260612_expanded_eval/`
- Runtime log:
  `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/logs/runtime/recoverability_expanded_eval_20260612.log`
- Expanded training-quality summary:
  `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/results/representation_audit_20260612_expanded_eval/expanded_training_quality.md`
- Corrected 47 output with `secondary_fraction` report-only:
  `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/results/representation_audit_20260612_expanded_eval/decision_no_secondary/`
- Checkpoints evaluated:
  - ProtGPT2: `/oss-pvc/zhk_zip/outputs/research2/clt_weights/r2_clt_expand_retrain_20260606_2gpu/protgpt2/step_300000`
  - ZymCTRL: `/oss-pvc/zhk_zip/outputs/research2/clt_weights/r2_clt_expand_retrain_20260606_2gpu/zymctrl/step_300000`
  - ProGen2-medium: `/oss-pvc/zhk_zip/outputs/research2/clt_weights/r2_clt_expand_retrain_20260606_2gpu/progen2-medium/step_300000`
- Pipeline: 44 cache -> 45 probes -> 46 oracle steering -> 47 decision.
- `RUN_48=0`; this run evaluates the already-trained expanded dictionaries and
  does not launch another retraining job.

## Final Decision

The expanded dictionaries do not change the frozen recoverability decision.
After applying Amendment 1b, `secondary_fraction` is report-only and is excluded
from the rich/bottleneck gate.

- Retrain decision: **NO-GO**
- Reason: dictionary already near-faithful on rich tasks.
- Controllability: ZymCTRL oracle steering remains `distributed_or_robust`.
- All three models are `substrate_rich`.
- No model has a primary-gate dictionary bottleneck after excluding
  `secondary_fraction`.
- ZymCTRL and ProGen2-medium are near-faithful on their rich tasks; ProtGPT2 is
  mixed because EC top-class is just below the faithful threshold (rho 0.785).

## Expanded Training Quality

Final checkpoint lines parsed from the remote training logs:

| Model | Checkpoint | FVU | Dead fraction | Dead % | L0 | Loss |
|---|---|---:|---:|---:|---:|---:|
| ProtGPT2 | step_300000 | 0.2782 | 0.334 | 33.4% | 128.0 | 7.1176 |
| ZymCTRL | step_300000 | 0.3348 | 0.044 | 4.4% | 90.6 | 0.0201 |
| ProGen2-medium | step_300000 | 0.3101 | 0.122 | 12.2% | 123.0 | 0.0138 |

The expanded dictionaries are not merely larger failed dictionaries. Dead-unit
rates are materially lower than the earlier 4096-d baselines, especially for
ZymCTRL and ProGen2-medium, while FVU remains in the same broad reconstruction
quality range. This strengthens the negative conclusion: the downstream NO-GO
is not explained by a trivially dead or unusable expanded dictionary.

## Probe Results

Skill is chance-corrected score. `C` is the residual-stream ceiling (`R_raw`),
`F` is the sparse-code floor (`R_code`), and `rho=F/C`.

| Model | Task | C | F | gap | rho | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| ProtGPT2 | EC top-class | 0.260 | 0.204 | 0.056 | 0.785 | Some loss vs raw; worse than v2. |
| ProtGPT2 | Pfam family | 0.939 | 0.789 | 0.151 | 0.840 | High absolute score but composition baseline is also high. |
| ProtGPT2 | secondary fraction | 0.621 | 0.003 | 0.619 | 0.004 | Still not recoverable; treat as unstable/report-only. |
| ProtGPT2 | residue SS | 0.221 | 0.224 | -0.003 | 1.000 | Near-faithful / slightly above raw. |
| ZymCTRL | EC top-class | 0.123 | 0.144 | -0.021 | 1.000 | Improved over v2 but raw substrate is weak. |
| ZymCTRL | Pfam family | 0.902 | 0.719 | 0.184 | 0.796 | Worse than v2; just below faithful threshold. |
| ZymCTRL | secondary fraction | 0.397 | -6.133 | 6.530 | 0.000 | Still unstable and not usable as primary evidence. |
| ZymCTRL | residue SS | 0.149 | 0.133 | 0.016 | 0.890 | Near-faithful. |
| ZymCTRL | decoder EC | 0.652 | 0.623 | 0.029 | 0.955 | Strongly improved; near-faithful on decoder-native EC. |
| ProGen2-medium | EC top-class | 0.272 | 0.228 | 0.044 | 0.837 | Improved; faithful by rho threshold. |
| ProGen2-medium | Pfam family | 0.952 | 0.931 | 0.021 | 0.978 | Improved; near-faithful. |
| ProGen2-medium | secondary fraction | 0.608 | -253.863 | 254.472 | 0.000 | Less bad than v2 but still numerically invalid as a claim driver. |
| ProGen2-medium | residue SS | 0.256 | 0.205 | 0.050 | 0.803 | Near-faithful, similar to v2. |

## Corrected 47 Decision

This is the Amendment 1b-compliant decision, written remotely to
`decision_no_secondary/`.

| Model | Rich tasks | Bottleneck tasks | Faithful tasks | Substrate | Dictionary | Mean gap |
|---|---|---|---|---|---|---:|
| ProtGPT2 | EC top-class, Pfam family, residue SS | - | Pfam family, residue SS | RICH | mixed | 0.068 |
| ZymCTRL | residue SS, decoder EC | - | residue SS, decoder EC | RICH | near-faithful | 0.023 |
| ProGen2-medium | EC top-class, Pfam family, residue SS | - | EC top-class, Pfam family, residue SS | RICH | near-faithful | 0.039 |

`secondary_fraction` remains reported as a diagnostic, but it is not a
decision-driver. Its floor regression is still numerically unstable and should
not be used as the sole bottleneck evidence.

## Comparison To Corrected v2

Main floor-score changes (`F_new - F_v2`):

| Model | Task | Delta F |
|---|---|---:|
| ProtGPT2 | EC top-class | -0.058 |
| ProtGPT2 | Pfam family | -0.019 |
| ProtGPT2 | secondary fraction | +3.158 |
| ProtGPT2 | residue SS | +0.007 |
| ZymCTRL | EC top-class | +0.122 |
| ZymCTRL | Pfam family | -0.049 |
| ZymCTRL | secondary fraction | +178.536 |
| ZymCTRL | residue SS | +0.004 |
| ZymCTRL | decoder EC | +0.098 |
| ProGen2-medium | EC top-class | +0.075 |
| ProGen2-medium | Pfam family | +0.021 |
| ProGen2-medium | secondary fraction | +663.602 |
| ProGen2-medium | residue SS | -0.003 |

The expanded run improves several sparse-code floors, especially ZymCTRL
decoder-native EC and ProGen2 EC/Pfam. It does not produce a publishable rescue
of the original circuit-tracing claim because:

1. The frozen decision remains NO-GO.
2. After Amendment 1b, there are no primary-gate bottleneck tasks; the previous
   `secondary_fraction` bottleneck was a report-only numerical pathology.
3. Oracle residual-stream steering still passes 0/8 EC class gates.
4. Several improvements are near-faithful recoverability rather than stronger
   mechanistic controllability or causal feature evidence.

## Interpretation

The expanded dictionaries are useful as a negative-control / rescue attempt.
They show that capacity can improve some readouts, so the old dictionaries were
not perfectly saturated. However, the improved floors mostly say that larger
CLTs can preserve more linearly decodable signal up to the raw-representation
ceiling. They do not establish monosemantic biological circuits or controllable
functional mechanisms.

For the manuscript, this supports a nuanced conclusion:

- Keep: conserved sparse-feature readouts, checkpoint-quality diagnostic,
  calibrated negative controllability, and limits of current CLT-style circuit
  tracing for these protein generators.
- Do not claim: successful rescue, causal EC steering, biological-primitive
  dictionary, or a clean dictionary-bottleneck explanation.
