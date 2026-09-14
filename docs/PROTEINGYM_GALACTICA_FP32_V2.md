# Galactica native DMS protocol v2 — direct checkpoint FP32

**Status:** independent protocol after the recorded Galactica-30B BF16 synthetic failure. Not a pre-registration from before that diagnostic. Not protocol v1. Not a diagnostic PASS, wrapper ADMITTED, or a DMS score. `docs/PROTEINGYM_EXTENSION_PROTOCOL.md` and `docs/PROTEINGYM_NUMERICAL_DIAGNOSTIC.md` are unchanged.

## Why this exists

The frozen v1 probe for `galactica-30b` (pin `e8d9f2b429f012b50cfa3b90ab6d0604dd7367fe`, run `20260913193000_6612da48b3bb`, BF16) failed the mixed-batch versus single-sequence check at **0.026644134521484376 nats/target** against the frozen 0.02 ceiling. That was not CUDA OOM. A later synthetic diagnostic reproduced that difference exactly on mixed AA20 and showed that executing the same BF16-rounded weights in FP32 shrank the original geometry to about **5.29×10⁻⁵**, while a padded-width contrast remained about **1.14×10⁻⁴**.

Those numbers justify a **separate, explicit** Galactica protocol. They do not move the v1 ceiling, do not mint a v1 PASS, and do not qualify a from-checkpoint FP32 load until this protocol's own gates pass.

## What is fixed

- Protocol id: `galactica-fp32-v2`. Default native DMS remains `native-dms-v1` (Galactica BF16, RITA FP32, v1 0.02 low-precision batch check).
- Arms: the four Galactica rungs only. RITA stays on v1 as an independent single point.
- Load: **one** `load_galactica(..., dtype="float32")` inside the shared `fp32_matmul_context`. **Both** `cuda_matmul_allow_tf32` and `cudnn_allow_tf32` are False, and float32 matmul precision is `highest`, restored on the way out. This is a from-checkpoint float32 load, not `model.float()` on BF16-rounded weights, and **not** a claim that the tensor files on disk were stored as FP32.
- Geometry gate: the diagnostic's 7 cases × 3 repeats, **all 15 comparisons**, every repeat. Ceiling **1e-4** nats per residue target. Includes `duplicate_short_padded` vs `duplicate_short`. Native-vs-independent ≤1e-4 on **all seven** cases (forced-width still has native `labels` loss). Public-vs-independent ≤1e-4 only on the five natural-width cases; forced-width cases have no public scorer API.
- After geometry: the existing author-alignment mixed public check at float32, refusals, LOOKUP freeze, FASTA/CSV/WT binding, and longest eligible batch. Scientific queue, seed `20260807 + original index`, cap 1000, N→C stratum, family bootstrap, and both baselines are the v1 authority in `docs/PROTEINGYM_EXTENSION_PROTOCOL.md`; they are not copied here.
- Output: one flat `galactica_fp32_v2_outcome.json`. Success is `probe_passed=true` and `status=passed` only after restore and owned-scorer release both succeed. Ordinary failures write the same filename with `probe_passed=false`, `status=failed`, phase/exception, and **whatever evidence was already obtained** (settings, facts, runtime, sources, fingerprints, geometry, alignment, cohort). A longest-batch failure records `failure.phase=longest_eligible_batch`. Disk write failure itself is raised, not swallowed. CLI exit 1. v1 `interface_check.json` is never written. Wrapper admission of bytes is not PASS. The 1e-4 ceiling and the geometry's 21 case observations (7×3) and 45 comparison rows (15×3) are not changed to match a run.
- Analyse: explicit `--protocol galactica-fp32-v2`, complete four-rung Galactica group only. Refuses v1 BF16 probes, diagnostic artefacts, missing policy, and a three-rung subset. Status is `completed analysis`, not the v1 “no new model fitness scores” freeze sentence.

A later CUDA OOM on the longest batch may be re-probed by hand along 16→8→4→2→1. That ladder is not an automatic retry and is not a reason to drop a failing width contrast or raise 1e-4.

## What a result is allowed to mean

All four Galactica rungs must pass this protocol before they are scored together under v2. A passing geometry gate is not a ProteinGym Spearman, not a retrieval bound, and not a parameter-count effect. The BF16 v1 30B failure remains on record.
