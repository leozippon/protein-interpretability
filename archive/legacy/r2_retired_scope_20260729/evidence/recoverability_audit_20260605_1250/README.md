# R2 Recoverability Audit Results (2026-06-05)

> **Evidence-audit correction (2026-07-16).** The original files and verdict
> strings in this directory are preserved for provenance. `R_raw` is an
> architecture-specific layer-normalized CLT input rather than a raw residual
> stream. The Experiment 46 direction was derived in this input space but added
> at MLP output and used the heuristic motif/composition scorer; 0/8 does not
> support the stored `distributed_or_robust` interpretation. The corrected
> manuscript uses only the selected linear-signal retention results and treats
> the wider-checkpoint comparison as exploratory. See EXP-R2-022 and
> `r2_interpretability_transfer/preregistration/DECISION_LOG.md`.

This directory contains the lightweight local copy of the H200 run
`representation_audit_20260605_1250_recoverability_full`.

## Remote provenance

- Pod: `jiaotongdamoxing-zhk-zip-beliefnav-2gpu-0603-master-0`
- Remote output:
  `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/results/representation_audit_20260605_1250_recoverability_full`
- Remote runtime log:
  `/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research2/logs/runtime/recoverability_full_20260605_1250.log`
- Local copy time: 2026-06-05

Large cache arrays (`*.npz`, `*.npy`) were intentionally not copied locally.
Only manifests, probe results, decision files, steering results, cohort metadata,
and the runtime log were copied.

## What ran

- Experiment 44: representation cache for ProtGPT2 v2, ZymCTRL v2, and
  ProGen2-medium.
- Experiment 45: recoverability probes over EC top-class, Pfam family,
  secondary-structure fraction, residue-level secondary structure, and
  decoder-native EC classification.
- Experiment 46: ZymCTRL oracle-direction EC steering.
- Experiment 47: frozen decision-table application.
- Experiment 48 was skipped automatically because Experiment 47 returned NO-GO.

## Key run metadata

- Swiss-Prot protein cohort: 820 proteins.
- Residue-level secondary-structure labels: 44,626 labelled residues from 319
  proteins.
- Decoder-native EC cohort: 48 ZymCTRL generated sequences across 8 classes.
- ESM-2 reference: `/gpfs/jiaotongdamoxing/zhk_zip/models/esm2_t36_3B_UR50D`.
- ProtGPT2/ZymCTRL CLTs: v2 checkpoints at 200k steps, `d_clt=8192`, `k=128`.
- ProGen2-medium CLT: rerun checkpoint at 100k steps, `d_clt=4096`, `k=64`.

## Headline results

- ProtGPT2: substrate is RICH, but it did not satisfy the frozen retrain GO
  conditions.
- ZymCTRL: mixed substrate; no clear dictionary-bottleneck verdict.
- ProGen2-medium: substrate is RICH, but it did not satisfy the frozen retrain
  GO conditions.
- Oracle steering: 0/8 EC classes passed the significance gate; verdict
  `distributed_or_robust`.
- Final decision: **NO-GO** for capacity retraining under the pre-registered
  protocol.

## Corrected v2 re-analysis

Claude's 2026-06-05 review identified analysis-design issues in the original
probe pass: the random-projection gate was non-functional at high sparse-code
dimensionality, EC top-class was strongly confounded with Pfam family,
`secondary_fraction` used a near-constant turn target, and high-dimensional
ridge regression was ill-conditioned. The v2 pass reused the same cached
representations and reran only scripts 45 and 47 with:

- in-fold PCA dimensionality control (`--pca-dim 256`) for all probe inputs;
- `ec_topclass_stratified` as a report-only EC/Pfam-confound diagnostic;
- the random-projection control reported but removed from the richness gate;
- `secondary_fraction` restricted to helix and strand targets.

The v2 verdict remains **NO-GO**. All three models become `substrate_rich`, but
no model satisfies the frozen dictionary-bottleneck GO gate. The stratified EC
ceilings are much higher than family-disjoint EC ceilings, confirming the family
confound. `secondary_fraction` remains numerically unstable and should be
treated cautiously; residue-level secondary structure is the better-powered
structural probe.

## Local files

- `cache/manifest.json`: run manifest and model/checkpoint metadata.
- `cache/cohort.json`: cohort metadata and selected records.
- `probes/probe_results.md`: human-readable probe table.
- `probes/probe_results.json`: full probe results with bootstrap intervals.
- `steering/oracle_steering.json`: oracle-direction steering outcomes.
- `decision/decision.md`: human-readable decision table.
- `decision/decision.json`: machine-readable decision output.
- `recoverability_full_20260605_1250.log`: copied runtime log.
- `probes_v2/probe_results.md`: corrected v2 probe table.
- `probes_v2/probe_results.json`: full corrected v2 probe results.
- `decision_v2/decision.md`: corrected v2 decision table.
- `decision_v2/decision.json`: corrected v2 machine-readable decision output.
- `recoverability_v2_20260606_0015.log`: copied v2 runtime log.
