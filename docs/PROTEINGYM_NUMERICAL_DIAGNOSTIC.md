# Galactica-30B synthetic numerical diagnostic

**Status:** authorized 2026-09-14 as a limited synthetic recipe. Not protocol v1, not an interface gate, and not a DMS score. `docs/PROTEINGYM_EXTENSION_PROTOCOL.md` is unchanged.

## Why this exists

The frozen native ProteinGym probe for `galactica-30b` (pin `e8d9f2b429f012b50cfa3b90ab6d0604dd7367fe`, run `20260913193000_6612da48b3bb`, BF16, batch 16) failed the existing batch-versus-single check on the same `MKT` / AA20 strings: **0.026644134521484376 nats/target against a frozen ceiling of 0.02**. That was not CUDA OOM. `galactica-125m`, `galactica-1.3b`, and `galactica-6.7b` passed the same gate.

This diagnostic records how that discrepancy splits across **batch membership**, **tensor width / right-padding**, and **BF16-rounded weights executed in FP32**. It does not move the 0.02 ceiling, does not mint a probe PASS, and does not produce Spearman or MODEL−LOOKUP values.

## Fixed recipe

Load `galactica-30b` once with the native strict loader at BF16. Keep `eval` and `use_cache=False`. Do not accept LOOKUP, wild types, ProteinGym CSVs, a search grid, or a threshold flag. Official wrapper `--out` / `--device` are accepted; `--arm` defaults to `galactica-30b` and refuses every other name.

Synthetic strings are the same as the old helper: `MKT` and `AA20 = ACDEFGHIKLMNPQRSTVWY`. Natural renderings must be width 5 and 22 with 3 and 20 residue targets. Three repetitions, fixed case order, every observation kept; do not pick the closest repeat.

| Case | Sequences | Tensor width | Public scorer |
|---|---|---:|---|
| `single_short` | `MKT` | 5 | yes, natural packing |
| `single_long` | AA20 | 22 | yes, natural packing |
| `mixed` | `MKT`, AA20 | 22 | yes, public mixed batch |
| `single_short_padded` | `MKT` | 22 forced | no; there is no public forced-width API |
| `duplicate_short` | `MKT`, `MKT` | 5 | yes |
| `duplicate_short_padded` | `MKT`, `MKT` | 22 forced | no |
| `duplicate_long` | AA20, AA20 | 22 | yes |

Independent packing follows the production residue labels: right-pad with the checkpoint pad id, `labels` only on scored residue positions, other positions `-100`, `logits[p-1]` predicts `ids[p]`, independent FP32 cross-entropy with `reduction="none"`. Per-target denominators are 3 or 20 residue targets, never the padded width. Duplicate rows are contrasted separately; they are not averaged first. Same-width-22 MKT contrasts are derived from those cases: `mixed[0]` vs `single_short_padded[0]`, each `duplicate_short_padded` row vs `single_short_padded[0]`, and `mixed[0]` vs `duplicate_short_padded[0]`. The recipe stays 7×3×2; these rows are not extra forwards.

Native `forward(labels=...)` is recorded beside that independent CE. Missing loss, non-finite values, or a wrong logits shape fail the run. The 0.02 gate is not applied to these observations.

## Two phases

1. **BF16.** The loaded checkpoint, as requested.
2. **FP32 execution of BF16-rounded weights.** The caller snapshots the current matmul/TF32 policy, then the same module is converted with `model.float()`. TF32 matmul is turned off and float32 matmul precision is set to `highest` for this phase only. The snapshot is restored even if conversion raises, including a non-default prior precision such as `high`. Loader `facts` stay the original BF16 observations. A second checkpoint is not loaded and a second 30B copy is not kept.

Phase 2 is not a from-checkpoint float32 interface qualification.

Row-wise contrasts that isolate effects:

- **Width / pad:** `single_short_padded` vs `single_short`; `duplicate_short_padded` vs `duplicate_short`.
- **Batch membership at the same width:** `duplicate_short` rows vs `single_short`; `duplicate_long` rows vs `single_long`; at width 22, `mixed[0]` and `duplicate_short_padded` rows vs `single_short_padded[0]`.
- **Mixed packing:** `mixed[0]` vs `single_short` (3 targets); `mixed[1]` vs `single_long` (20 targets). This is the geometry of the failed probe comparison, recorded rather than gated.
- **Precision:** the same contrasts after the in-place FP32 promotion.

CUDA elapsed time and peak allocated/reserved bytes are captured for each completed phase, including the conversion in phase 2. CPU stubs record memory as unmeasured. A CUDA measurement or required metadata read failure fails the diagnostic; it is not reclassified as a CPU observation or silently accepted as complete. A failed phase retains available observations without inventing missing resource measurements.

## What a result is allowed to mean

The artefact `galactica_numerics_diagnostic.json` has `kind = numerical_diagnostic_not_interface_gate_or_dms_score`, `no_dms_scores = true`, and `no_interface_admission = true`. It records `code_sources` from this script's `__file__` and `galactica_fitness.__file__`, not from the request. `status` is `complete` or `failed`. Wrapper admission of the bytes is not a scientific PASS.

Each phase is registered before its first forward. Completed cases are appended as they finish. A later missing native loss keeps those earlier cases; comparisons are computed only for complete 7-case blocks, so a partial block does not raise on missing keys. Cleanup (policy restore, scorer release) exceptions are recorded, force `failed`, and still attempt to write the observations; a write failure itself raises. That file is not a passing probe. There is no retry, batch-size ladder, checkpoint change, or threshold edit.

These numbers cannot attribute the 0.02 breach to a named kernel, cannot declare BF16 packing “safe”, and cannot reopen Galactica-30B ProteinGym scoring under protocol v1. If a later decision changes that protocol, it is a separate freeze.
