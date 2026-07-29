# Next steps — Recoverability audit v2 (corrected analysis)

**Date:** 2026-06-05 · **Owner:** Codex (execution) · **Status:** plan

## Why a v2

The 2026-06-05 headline run (`r2_interpretability_transfer/evidence/recoverability_audit_20260605_1250/`)
returned NO-GO, but the *substrate-richness* verdicts are not trustworthy because
the low ceilings are driven by **measurement design, not model weakness**:

- ESM-2-3B (gold standard) drops from macro-F1 **0.705 (stratified) → 0.445
  (family-disjoint)** on the same EC cohort. EC top-class is **99% confounded
  with Pfam family** (161/163 families → one EC class), so the audit's
  (correct) family-disjoint CV measures only the small *family-independent* EC
  signal — low for everyone.
- The `R_rand` control is **non-functional**: `n_alive` (700–5,868) ≫ `d_model`
  (1,280–1,536), so the "matched-dim" random projection is information-
  preserving and `rand ≈ ceiling` everywhere; the `beats_rand` gate does no work
  and even false-negatived ZymCTRL `decoder_ec`.
- The `secondary_fraction` floor ridge is **ill-conditioned** (R² −1.4 to −26 on
  ~thousands of sparse features vs 300 proteins), creating a spurious "bottleneck".
- `secondary_fraction`'s **turn target is near-constant** (mean 0.029); Pfam is
  **composition-saturated** (base 0.855).

The robust conclusions stand (models track ESM-2; ρ = 0.7–1.0 on real signals;
NO-GO). v2 fixes the controls so the substrate verdicts are trustworthy and the
NO-GO becomes a *clean* "dictionary is faithful" result instead of an ambiguous
"no model meets the conditions."

## Pre-registration discipline

These are **validity corrections of demonstrable methodological errors**, not
outcome-driven tuning: the §6 thresholds stay **frozen**, and the fixes are
justified independently of the result (a non-functional control, an
ill-conditioned regression, a 99% confound). Before running v2, append a dated
amendment to `DECISION_LOG.md` recording each change and this rationale.

---

## Phase 1 — corrected re-analysis on the EXISTING cache (do first; cheap)

No re-extraction. Operates on the cached `reps_*.npz`, `residue_*.npz`,
`decoder_*.npz`, `esm2.npy`, `ngram.npy`, `cohort.json` from the 2026-06-05 run
(re-stage the GPFS cache locally, or run on the pod).

### Code changes

1. **Unified PCA-in-pipeline dimensionality control** (fixes T3 floor **and**
   makes ceiling/floor dimensionality-fair; makes the broken `R_rand` gate
   unnecessary). In `recoverability_audit.cv_predict_classification` /
   `cv_predict_regression`, insert PCA between the scaler and the estimator:
   `StandardScaler → PCA(n_components=min(pca_dim, n_features, n_train-1),
   svd_solver="randomized", random_state=seed) → LogReg/Ridge`. PCA fit inside
   each CV fold (no leakage). Add a `--pca-dim` CLI arg to `45` (default **256**).
   Apply to **all** representations uniformly so C, F, and R_recon are compared
   at matched dimensionality.

2. **Drop `beats_rand` from the richness gate.** In
   `recoverability_audit.per_model_verdict`, the "rich" test becomes: ceiling
   beats the composition/chance baseline by `margin_macro_f1` **or**
   `ceiling_ci_low > baseline_ci_high`. Keep computing `R_rand` (now at fixed dim
   256) only as a **reported diagnostic**, not a gate.

3. **EC reported in two variants.** Add a task `ec_topclass_stratified` (identical
   to `ec_topclass` but with per-record/solo groups → stratified CV) alongside the
   existing family-disjoint `ec_topclass`. The family-disjoint variant drives the
   conservative verdict; the stratified one is reported to quantify the
   family confound (expected ≈ 0.56 vs 0.30 skill). Do **not** use the stratified
   variant for the dictionary-bottleneck gate (it re-introduces family leakage).

4. **Structural task cleanup.** (a) Make `residue_ss` (T4, already cached, 44,626
   residues) a **primary** decision task — it is the well-powered structural
   probe. (b) For `secondary_fraction`, drop the near-constant turn target →
   regress (helix, strand) only (`target[:, :2]`).

### Run

```bash
# from repo root, ct env; CACHE = the 2026-06-05 cache dir
python r2_interpretability_transfer/scripts/45_probe_ceiling_floor.py \
  --cache-dir $CACHE --out-dir $OUT/probes_v2 --pca-dim 256 \
  --tasks ec_topclass,ec_topclass_stratified,pfam_family,secondary_fraction,residue_ss,decoder_ec
python r2_interpretability_transfer/scripts/47_decision_table.py \
  --probes $OUT/probes_v2/probe_results.json --out-dir $OUT/decision_v2
```

Cost: CPU only; residue_ss (44k residues × layers × PCA) is the slow part
(~tens of minutes); the rest is minutes. **No GPU, no re-extraction.**

### Decision gates (frozen §6 thresholds)

- **Clean NO-GO (expected):** ρ stays 0.7–1.0 on the now-trustworthy rich tasks
  (EC family-disjoint, residue-SS, decoder-EC, Pfam) → dictionary is faithful →
  a bigger dictionary won't help. This is the publishable result.
- **GO:** some model is rich AND ρ ≤ 0.5 on ≥2 tasks → genuine dictionary
  bottleneck → proceed to the gated retrain (`48`, per §6.3/§6.4).
- Report the stratified-vs-family-disjoint EC gap as the family-confound size.

---

## Phase 2 — re-extraction (ONLY if Phase 1 ceilings can't discriminate)

If, after Phase 1, the discriminative-task ceilings are still too low/variable to
separate rich-vs-thin (e.g., EC family-disjoint skill < ~0.2 even for ESM-2),
re-run `44` (H200) with more headroom:

- **Larger cohort:** `--ec-per-class 80 --pfam-per-class 24 --ss-n 500`, and more
  residue-SS proteins.
- **Richer pooling:** in `extract_protein_matrices`, concatenate mean **and** max
  pooling over tokens (localized active-site/motif signal survives).
- **ESM-2 per-residue reference** for T4 so `phi` is defined on the residue task.

Then re-run Phase 1 (45→47). Cost: one H200 cache run (~1 GPU-day) + the cheap
re-analysis.

---

## Phase 3 — steering power (optional; only to firm up the §6.2 negative)

The 0/8 oracle-steering result is underpowered (heuristic `ec_purity_score`,
single layer L3 / α=8, unsteered purity already 0.77–0.97). To make the
controllability verdict publishable:

- Sweep `--inject-layer {3,12,30}` × `--alpha {2,4,8,16}` in `46`.
- Swap the purity judge to a **trained EC classifier** (ESM-2 + logistic
  regression on labelled enzymes, or CLEAN) instead of the motif heuristic.

Cost: generation-bound (moderate). Lower priority — steering is already framed as
a calibrated negative.

---

## Deliverables

- `DECISION_LOG.md` amendment (the validity-correction rationale).
- `…/probes_v2/` + `…/decision_v2/` and a one-paragraph EXPERIMENT_LOG entry
  (EXP-R2-019) with the corrected verdict and the EC confound size.
- If GO: the §6.4-gated retrain (`48`) on the largest-gap model.

## One-line summary

Re-analyse on the existing cache with a PCA dimensionality control, a dropped
broken `rand` gate, a de-confounded two-variant EC readout, and per-residue SS as
the primary structural task — converting the muddy NO-GO into a trustworthy one —
and only re-extract (bigger cohort + richer pooling) if the ceilings still can't
discriminate.
