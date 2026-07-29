# P0-4 joint conditional-semantics adjudication

Date frozen: 2026-07-17  
Status: infrastructure validated; no real P0-4 result exists

## Purpose and boundary

`scripts/53_run_conditional_semantics.py` reports q-values within one run. A
submission analysis spans models, layers, representations, features, labels
and both blocking schemes, so those per-run q-values are not the required
multiplicity correction. `scripts/67_adjudicate_conditional_semantics.py`
verifies the complete prespecified collection and recomputes one global
Benjamini--Hochberg family from the raw permutation p-values.

A verified joint receipt is necessary, not sufficient, for P0-4. It cannot
replace real continuous-activation extraction, passing P0-2 dictionaries,
independent cohorts, frozen biological and negative labels, or the scientific
acceptance criterion. The current project has none of those real joint runs;
P0-4 remains open.

## Freeze contract

Before inspecting any joint q-value, copy
`configs/npj_conditional_semantics_joint_adjudication.example.json`, replace
every placeholder and enumerate the exact Cartesian-product hypothesis list
for every model/layer run. The frozen collector spec contains:

1. one unique run directory for each model/layer cell;
2. the externally recorded SHA-256 of its `run_manifest.json`;
3. the SHA-256 of the original script-53 spec and NPZ data bundle;
4. the independent prospective-power-plan SHA-256, or JSON `null` when no such
   plan exists; and
5. every expected `(representation, feature, label, blocking)` tuple, with no
   wildcard or inferred selection.

Hash the completed collector spec independently. Execute only with that exact
hash:

```bash
python scripts/67_adjudicate_conditional_semantics.py \
  --spec configs/npj_conditional_semantics_joint_adjudication.json \
  --spec-sha256 <frozen-collector-spec-sha256> \
  --out-dir results/npj_revision/semantics/joint_adjudication
```

The output path must not exist. Publication is staged and atomic; stale
staging paths and overwrite attempts fail.

## Fail-closed checks

The collector independently rejects:

- a changed collector spec, run manifest, script-53 spec, NPZ or manifested
  output;
- an absent, extra or duplicate hypothesis in either the frozen inventory or
  result table;
- duplicate model/layer cells, extra files or non-file entries in a run
  directory, and incomplete output manifests;
- non-confirmatory summaries/specs, non-finite statistics, invalid p-values,
  inconsistent roles, counts, folds or source-data permutation fractions;
- a biological label that is constant within every protein, even if a result
  table claims otherwise; and
- an incomplete, duplicated, non-positive, changed or confirmatory-data-derived
  prospective power plan.

The collector never uses the per-run `qvalue` column. It retains it only as
`reported_per_run_qvalue_ignored`, pools every raw permutation p-value, and
writes `joint_bh_qvalue` over the exact global family
`model x layer x representation x feature x label x blocking`.

## MDE and adjudication rules

The output keeps the concepts separate:

- `retrospective_bootstrap_detectable_delta_mse` is estimated from the analyzed
  observations and remains explicitly retrospective;
- `prospective_minimum_detectable_delta_mse` is the script-53 per-run value
  reproduced from a hash-bound independent power plan; and
- `joint_prospective_minimum_detectable_delta_mse` is recomputed from the same
  independent standard error using conservative Bonferroni planning over the
  complete global family.

Three frozen modes are available:

- `multiplicity_only` produces no scientific pass/fail decision;
- `association` requires explicit biological targets from at least two
  distinct model/layer runs, both blockings for each target, a global q-value
  at or below alpha, and an effect and bootstrap lower endpoint at or above the
  frozen minimum; and
- `powered_bound` requires the same explicit replication/blocking structure,
  complete independent power plans over the entire global family, a joint
  prospective MDE at or below the frozen bound, and a bootstrap upper endpoint
  at or below that bound.

Target selection and thresholds must be frozen in the collector spec. A
software decision applies only to that estimand and cannot establish a
biological primitive, causal mechanism or dictionary-wide semantic validity.

## Published artifacts

Successful adjudication atomically publishes:

- `joint_conditional_effects.tsv`, including ignored local q-values, global BH
  q-values and separately labeled MDE fields;
- `summary.json`, including the exact family, input provenance and any frozen
  decision-rule result; and
- `completion_receipt.json`, binding the collector spec, implementation,
  input manifests and both output hashes.

The summary and receipt report two separate states. An
`artifact_validation_status` of `verified_complete` means only that the frozen
family and published artifacts passed the software contract.
`scientific_gate_status` is copied from the frozen decision rule and is one of
`not_scientifically_adjudicated`, `passed`, or `failed`; the receipt also carries
the decision mode. A scientifically failed or unadjudicated family is therefore
never represented by a lone generic completion status.
