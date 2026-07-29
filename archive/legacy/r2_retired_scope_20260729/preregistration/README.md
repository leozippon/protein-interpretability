# preregistration/

Frozen experimental protocol for the **representation-recoverability audit** of
the three protein generators (ProtGPT2, ZymCTRL, ProGen2-medium) — the test that
separates "circuit tracing / sparse dictionaries failed (tool)" from "the models
learned little (substrate)".

- `PROTOCOL.md` — the pre-registration (R2-RECOV-AUDIT-v1): probing tasks,
  reusable datasets + labels, ceiling/floor/gap metrics, and the go/no-go
  thresholds for the high-cost dictionary retrain.
- `DECISION_LOG.md` — append-only setup confirmations and amendments.

Read `PROTOCOL.md` §6 before running anything: the decision thresholds are fixed
in advance and must not be changed in response to an observed result. Results
land in `r2_interpretability_transfer/results/representation_audit_20260604/`.
