# Decision table (Experiment 47)

Thresholds (frozen): margin=0.1, rho_lo=0.5, rho_hi=0.8, phi_rich=0.5, min_tasks=2.

## Per-model verdict

| Model | rich tasks | bottleneck tasks | substrate | dictionary | mean gap |
|---|---|---|---|---|---:|
| protgpt2 | ec_topclass, secondary_fraction | secondary_fraction | RICH | - | 1.727 |
| zymctrl | residue_ss | - | mixed | - | 0.016 |
| progen2-medium | ec_topclass, pfam_family | ec_topclass | RICH | - | 0.162 |

## Retrain GO/NO-GO (PROTOCOL §6.3)

**Decision: NO-GO** — no model meets the GO conditions.
No retrain.

## Controllability (oracle steering, §6.2)

- zymctrl: distributed_or_robust
