# Decision table (Experiment 47)

Thresholds (frozen): margin=0.1, rho_lo=0.5, rho_hi=0.8, phi_rich=0.5, min_tasks=2.

## Per-model verdict

| Model | rich tasks | bottleneck tasks | substrate | dictionary | mean gap |
|---|---|---|---|---|---:|
| protgpt2 | ec_topclass, pfam_family, secondary_fraction, residue_ss | secondary_fraction | RICH | - | 3.776 |
| zymctrl | secondary_fraction, residue_ss, decoder_ec | secondary_fraction | RICH | - | 185.066 |
| progen2-medium | ec_topclass, pfam_family, secondary_fraction, residue_ss | secondary_fraction | RICH | - | 918.073 |

## Retrain GO/NO-GO (PROTOCOL §6.3)

**Decision: NO-GO** — no model meets the GO conditions.
No retrain.

## Controllability (oracle steering, §6.2)

- zymctrl: distributed_or_robust
