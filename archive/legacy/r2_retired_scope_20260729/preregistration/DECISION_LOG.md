# Decision & Amendment Log — R2-RECOV-AUDIT-v1

Append-only. Every entry is dated. Amendments to `PROTOCOL.md` after the first
probe is run must appear here with a reason that is **not** motivated by an
observed result. Pre-run setup confirmations (e.g., resolving checkpoint paths)
are also logged here.

---

## 2026-06-04 — Protocol frozen

- `PROTOCOL.md` (R2-RECOV-AUDIT-v1) frozen. Thresholds in §6 are fixed.

## Pending pre-run setup (must be completed before Experiment 44)

- [ ] **Re-confirm live CLT checkpoint paths on the active H200 pod.** The
      2026-05-28 directory rename rewrote some hard-coded absolute GPFS/OSS paths
      (`…/biocc/Research2/…` → `…/biocc/paper_r2_nature_mi/…`) that may not match
      the remote staging. Resolve the real paths for
      `protgpt2_v2/step_200000`, `zymctrl_v2/step_200000`,
      `progen2-medium/step_100000` and record them here; pass via `--model-spec`.
- [ ] Confirm ESM-2 reference: `/Data/public/esm2_t36_3B_UR50D` (3B) or
      `/Data/public/esm2_t33_650M_UR50D` (light). Record which is used.
- [ ] Confirm T6 (structure) coverage ≥ 50 proteins, else drop T6 (allowed
      without amendment per §3).
- [ ] Freeze the cohort: record cohort hash, per-task class counts, and the
      family-disjoint split assignment.

## 2026-06-04 — Code implemented and smoke-validated

Implemented the audit under `r2_interpretability_transfer/scripts/`:

- `recoverability_audit.py` — engine (representation extraction, baselines,
  grouped-CV probes with chance-corrected skill + bootstrap, ceiling/floor/gap
  metrics, decision logic). Reuses the validated loaders in scripts 29/33/34.
- `44_cache_representations.py`, `45_probe_ceiling_floor.py`,
  `46_oracle_direction_steering.py`, `47_decision_table.py`,
  `48_capacity_retrain.py`.

Verification: all six `py_compile`-clean; engine import chain OK; the
analysis math validated on synthetic data (ceiling 0.74 / floor 0.16 →
rho 0.22 → GO); a real GPU smoke of 44→45→47 on ZymCTRL (24 proteins, layers
0/5, ESM-2-650M) ran end-to-end and produced the correct degenerate NO-GO.

### v1 deviations to log before interpreting results

1. **Checkpoint paths.** The engine defaults now point ProtGPT2 and ZymCTRL to
   the manuscript v2 `step_200000` checkpoints under `/oss-pvc`. ProGen2-medium
   still uses the final-checkpoint path and must be re-confirmed on the active
   pod before the headline run.
2. **T5 cohort size.** The current local steering benchmark JSON provides 48
   decoder-native EC example sequences across the 8 classes. This is adequate
   for a pilot run but below the intended full generated-sequence cohort.
3. **Oracle-steering purity (script 46).** Purity now uses the same
   class-specific scorer as `11_steering_benchmark.py`, and the verdict gate is
   restored to the frozen `>=3/8` EC classes criterion.

## 2026-06-04 — Code-review fix (Opus)

Reviewed Codex's revisions (group-level bootstrap, fail-fast grouped CV with
`--allow-cv-fallback`, tripeptide composition baseline, residue/decoder
baselines wired, `46` using `11_steering_benchmark.ec_purity_score`, EC prompts,
200k OSS default checkpoints, extended T5 loader → 48 sequences). All verified
correct and re-smoke-tested 44→45→47 end-to-end on ZymCTRL.

One correctness fix applied:

- **T2 (`pfam_family`) CV degeneracy.** The probe grouped by Pfam family while
  the label *is* the Pfam family, so family-disjoint CV held out every test
  class (macro-F1 → 0 for all representations; pre-existing, surfaced by the new
  fail-fast). Fixed in `45_probe_ceiling_floor.py`: `pfam_family` now uses
  stratified (per-record) CV; family-disjoint grouping is kept only where family
  is a confound (`ec_topclass`, `secondary_fraction`). Verified: C rose from
  0.000 to ~0.74 on a smoke cohort.

Caveats noted (not bugs): (i) `ec_purity_score` is a heuristic motif/composition
placeholder — swap in a real Pfam/HMMER or CLEAN classifier for a publishable
§6.2 verdict; (ii) on the local L20 box the OSS 200k checkpoints are not mounted
— pass `--model-spec` with local checkpoints; (iii) the `secondary_fraction`
sparse-code (floor) ridge is high-variance when alive features ≫ SS proteins —
keep enough SS proteins in the headline cohort.

## Amendments

### 2026-06-05 — Amendment 1: v2 validity corrections (analysis only; §6 thresholds unchanged)

Per `NEXT_STEPS_v2.md`. These correct demonstrable methodological errors and a
known confound; they are justified independently of the observed result (the
likely outcome — a faithful-dictionary NO-GO — is unchanged). The frozen §6
thresholds are **not** modified. Implemented in `recoverability_audit.py` and
`45_probe_ceiling_floor.py`:

1. **PCA-in-pipeline dimensionality control.** `StandardScaler → PCA(min(--pca-dim,
   n_features, min_train−1)) → estimator`, fit inside each CV fold, applied
   uniformly to R_raw/R_code/R_recon (`--pca-dim` default 256). Fixes the
   ill-conditioned `secondary_fraction` floor (R² −1.4…−26) and matches
   ceiling/floor dimensionality. PCA dim is capped to the smallest train fold so
   it is valid for uneven grouped folds.
2. **Dropped the non-functional `beats_rand` gate.** `n_alive` ≫ `d_model` made
   the up-projecting `R_rand` information-preserving (`rand ≈ ceiling`), so the
   gate did no work and false-negatived ZymCTRL `decoder_ec`. Richness now =
   ceiling beats the composition/chance baseline by margin or non-overlapping CI;
   `R_rand` is reported at a fixed dim (256) as a diagnostic only.
3. **EC reported in two variants.** Added `ec_topclass_stratified` (stratified CV)
   alongside the family-disjoint `ec_topclass` to quantify the 99% EC↔family
   confound. The stratified variant is **report-only** (`REPORT_ONLY_TASKS`),
   excluded from the rich/bottleneck gate.
4. **Structural cleanup.** `secondary_fraction` drops the near-constant turn
   target (regresses helix+strand only); per-residue `residue_ss` is a primary
   decision task.

Verified end-to-end (44→45→47) on a ZymCTRL smoke: stratified EC ceiling (0.19) >
family-disjoint (0.11) as designed; verdict moved to the clean
`substrate_rich + dictionary_near_faithful`. Residual: at very small SS cohorts
the `secondary_fraction` floor R² can still be negative (R² is unbounded below on
a hard family-disjoint regression); on the full ≥300-protein run PCA should keep
it well above the prior −26, but if it remains negative, treat
`secondary_fraction` as report-only and rely on `residue_ss`.

Run command for the corrected re-analysis (on the existing 2026-06-05 cache):

```bash
python r2_interpretability_transfer/scripts/45_probe_ceiling_floor.py \
  --cache-dir <2026-06-05 cache> --out-dir <probes_v2> --pca-dim 256
python r2_interpretability_transfer/scripts/47_decision_table.py \
  --probes <probes_v2>/probe_results.json --out-dir <decision_v2>
```

### 2026-06-06 — Amendment 1b: secondary_fraction made report-only (pre-specified contingency)

The v2 full run (EXP-R2-019) showed the `secondary_fraction` sparse-code FLOOR
still blows up (R² = −3.15 / −184.7 / −917.5 for ProtGPT2 / ZymCTRL / ProGen2)
even with PCA — the instability is StandardScaler-on-sparse-codes, not feature
count. As pre-specified in `NEXT_STEPS_v2.md`, `secondary_fraction` is now in
`REPORT_ONLY_TASKS` (reported, excluded from the rich/bottleneck gate); the
structural claim rests on `residue_ss` (ρ 0.81–0.98). This removes a spurious
"bottleneck" (and a latent false-GO risk) and does not change the decision.

Corrected decision (47 re-run on the existing `probes_v2`):
**NO-GO — "dictionary already near-faithful on rich tasks."** ProtGPT2 and ZymCTRL
= RICH + near-faithful; ProGen2 = RICH, EC only partially recovered (ρ 0.56,
between thresholds), no bottleneck. Mean gaps now sane (0.04–0.07). Output:
`decision_v2b/`.

Optional follow-up (only if a trustworthy `secondary_fraction` ρ is wanted): in
the floor regression, replace StandardScaler with a sparse-robust scaler
(e.g. MaxAbsScaler) or drop near-zero-variance code columns before scaling, then
re-probe; otherwise residue_ss is sufficient.

### 2026-06-12 — Amendment 2: exploratory capacity retrain executed; AUDIT CLOSED

Status: **exploratory override** of the NO-GO (clearly labelled, not protocol-
confirmed). EXP-R2-020 retrained all three decoders to `d_clt=16384`, k=128,
300k steps (≈2× width for ProtGPT2/ZymCTRL, ≈4× for ProGen2 vs its 4096 base);
EXP-R2-021 re-ran 44→47 on the expanded dictionaries (`RUN_48=0`).

Outcome — the capacity hypothesis is **falsified by demonstration**:
- Decision unchanged: **NO-GO**. All three substrate-rich; none a clean
  dictionary-bottleneck.
- Modest floor gains (old dicts not fully saturated): ZymCTRL decoder-EC
  ρ 0.806→0.955; ProGen2 EC ρ 0.56→0.84, Pfam ρ→0.978. ProtGPT2 EC/Pfam slightly
  worse. → capacity improves *faithfulness of recovery* toward the (linear)
  ceiling, but
- does **not** yield a biological-primitive dictionary, and oracle steering is
  still **0/8** (`distributed_or_robust`). No rescue of interpretation/control.

This converts the earlier inference ("capacity is not the bottleneck") into a
direct demonstration. **The recoverability audit is now closed.** No further
retrains without a new, separately justified amendment — the limits are
substrate signal + reconstruction objective + distributed computation, not
dictionary capacity.

Two consolidation items remain (no new training):
1. Record the expanded checkpoints' final FVU + dead-fraction (from training
   logs / a quick eval) so the negative is airtight (genuinely-better dictionary
   precondition, PROTOCOL §6.4). Pull from the remote pod.
2. Re-run `47` on the expanded probes with `secondary_fraction` already
   report-only (Amendment 1b) for a clean verdict statement; does not change NO-GO.

Canonical results for the manuscript: `decision_v2b/` (original dictionaries) is
the headline corrected verdict; EXP-R2-021 (expanded) is the capacity-
falsification robustness check.

### 2026-07-16 — Evidence-audit correction to Amendment 2 interpretation

This entry supersedes the causal and capacity interpretations in the 2026-06-12
entry without altering its recorded outputs.

- None of the expanded checkpoints met the preregistered FVU < 0.15
  prerequisite for a genuinely better dictionary; ProtGPT2 also missed the
  dead-fraction criterion. Their quality values were final training-log values,
  not a matched held-out comparison. EXP-R2-021 therefore does **not** falsify
  dictionary capacity or optimization. It is an exploratory wider-checkpoint
  probe comparison with mixed recovery changes.
- Script 46 derives its mean-difference direction from the architecture-specific
  CLT-input tensor and adds it to the MLP output. This coordinate/site mismatch,
  together with the heuristic motif/composition score, means that the 0/8
  result does **not** establish distributed or robust causal organization.
- The original protocol's `raw residual stream` terminology is incorrect for
  the captured tensors. See EXP-R2-022 and
  `results/circuit_analysis/swissprot_triplet_mi_reaudit_20260716/EVIDENCE_REPAIR_AUDIT.md`.

The retained conclusion is narrower: the reference and exploratory wider sparse
codes preserve selected linear readouts from their own architecture-specific
CLT inputs to varying degrees. These results do not isolate model substrate,
dictionary capacity, optimization, feature geometry or intervention-site
mismatch.
