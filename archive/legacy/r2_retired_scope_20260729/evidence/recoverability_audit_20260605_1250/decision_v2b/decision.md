# Decision table (Experiment 47)

Thresholds (frozen): margin=0.1, rho_lo=0.5, rho_hi=0.8, phi_rich=0.5, min_tasks=2.

## Per-model verdict

| Model | rich tasks | bottleneck tasks | substrate | dictionary | mean gap |
|---|---|---|---|---|---:|
| protgpt2 | ec_topclass, pfam_family, residue_ss | - | RICH | near-faithful | 0.044 |
| zymctrl | residue_ss, decoder_ec | - | RICH | near-faithful | 0.073 |
| progen2-medium | ec_topclass, pfam_family, residue_ss | - | RICH | - | 0.070 |

## Retrain GO/NO-GO (PROTOCOL §6.3)

**Decision: NO-GO** — dictionary already near-faithful on rich tasks.
No retrain.

