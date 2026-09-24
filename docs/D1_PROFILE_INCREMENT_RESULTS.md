# Profile increments and failure conditions: results

The mutation-vector analyses identify complementary predictive information beyond the specified scalar profile score for ProGen3-3B and ProteinGLM without supplied context, and for both ProLLaMA stages under homolog context. The historical failure-condition analysis separately describes where relative MODEL−LOOKUP performance varies. These are different supports and estimands; neither identifies a parameter mechanism or controls all evolutionary information.

## Mutation-level profile increments

The H200 analysis completed on 2026-09-23. The admitted archive SHA256 is `afc3fe4fecc5c6568705442670540e34840555177f1f4fd2bc1630ebeb3edda3`; the analysis artifact is `results/transfer/d1_followup_20260923/context/20260923075017_68b04c315654/context_profile_increment/profile_increment.json`, SHA256 `bfb467db60043fd583570f941ea61c4aef1ace37be21892e312d8de648f6ed8a`. All four source-vector hashes match retained complete scoring files. Exact mutation ordering, WT/family identities, draw digests, profile scores and experimental effects reconcile across the shared assays. Per-condition assay/variant counts, held-family fold identities and metric-specific support reconcile to the saved summaries; no local model inference or statistical resampling was performed for admission.

The [protocol](D1_PROFILE_INCREMENT_PROTOCOL.md) defines the scalar profile control, within-assay rank residualization and fixed ordinary least-squares predictors. `M` denotes model mutant-minus-WT score, `P` the external profile score, and `Y` the experimental effect. Every table averages assays within their existing 50%-identity WT families, then families equally. Correlations are dimensionless. Held-family Δρ is the Spearman of the `P+M` predictor minus that of the `P` predictor; ΔMSE is `MSE(P) − MSE(P+M)` in squared standardized-rank units, so positive values mean improved prediction.

All brackets are pointwise, exploratory 95% family-bootstrap intervals using 2,000 resamples. Cross-validation intervals condition on the already fitted held-family predictions; predictors are not refitted within bootstrap draws, so training-set variation is omitted. Predictors use unlabeled within-assay rank normalization of the held-out panel, a transductive setting. These intervals are not adjusted for multiple models, conditions or endpoints. Partial rank correlation removes only a linear dependence on scalar profile ranks, and the fixed `P+M` predictor tests only the specified supervised combination. Neither analysis controls all nonlinear profile relationships or all evolutionary information.

### Support and baseline

| Model | Native eligible assays / families / variants | Exact shared assays / families / variants | Native assays excluded from intersection |
| --- | --- | --- | ---: |
| ProGen3-3B | 164 / 134 / 20,992 | 164 / 134 / 20,992 | 0 |
| ProteinGLM-7B-CLM | 164 / 134 / 20,992 | 164 / 134 / 20,992 | 0 |
| ProLLaMA Stage 1 | 195 / 158 / 24,960 | 164 / 134 / 20,992 | 31 |
| ProLLaMA Stage 2 | 195 / 158 / 24,960 | 164 / 134 / 20,992 | 31 |

Each retained assay uses its frozen 128-variant draw. All eight metrics are defined on every retained assay in every model/condition/support cell; there are no undefined-statistic exclusions. The artifact enumerates all assay IDs and intersection exclusions. Shared support fixes assay and mutation identities, but each model retains its selected contexts and native encoding. The ProteinGLM and ProGen3 native analyses are exactly identical to their shared-support analyses; their rows below therefore serve both supports. ProLLaMA native-support sensitivity is reported separately.

On shared support, the raw profile/effect correlation is +0.3728 [+0.3436, +0.4060] and the held-family `P`-only Spearman is +0.3728 [+0.3404, +0.4048], for all model/condition cells on 164 assays / 134 families. On each ProLLaMA native support, these are +0.3612 [+0.3321, +0.3892] and +0.3612 [+0.3322, +0.3883], respectively, on 195 assays / 158 families. The equal point estimates have slightly different Monte Carlo interval endpoints because the metric summaries use different fixed bootstrap seeds. The profile baselines do not change with supplied context.

### Exact shared support

All rows in the next two tables use 164 assays, 134 families and 20,992 variants. “None” means no supplied sequence context; it does not mean absence of evolutionary information in checkpoint parameters.

| Model / context | ρ(M, P) [95% interval] | ρ(M, Y) [95% interval] | Partial rank ρ controlling P [95% interval] |
| --- | --- | --- | --- |
| ProGen3-3B / None | +0.5350 [+0.5048, +0.5658] | +0.3830 [+0.3528, +0.4132] | +0.2311 [+0.2021, +0.2587] |
| ProGen3-3B / Unrelated | +0.2963 [+0.2566, +0.3370] | +0.1733 [+0.1330, +0.2104] | +0.0789 [+0.0473, +0.1112] |
| ProGen3-3B / Homolog | +0.4359 [+0.4016, +0.4695] | +0.2740 [+0.2424, +0.3049] | +0.1235 [+0.0952, +0.1501] |
| ProteinGLM-7B-CLM / None | +0.5541 [+0.5270, +0.5813] | +0.3943 [+0.3635, +0.4248] | +0.2327 [+0.2026, +0.2615] |
| ProteinGLM-7B-CLM / Unrelated | +0.5549 [+0.5281, +0.5825] | +0.3962 [+0.3660, +0.4267] | +0.2352 [+0.2052, +0.2644] |
| ProteinGLM-7B-CLM / Homolog | +0.5750 [+0.5454, +0.6034] | +0.3955 [+0.3648, +0.4275] | +0.2287 [+0.1993, +0.2570] |
| ProLLaMA Stage 1 / None | +0.3844 [+0.3510, +0.4166] | +0.1430 [+0.1123, +0.1752] | +0.0050 [-0.0252, +0.0334] |
| ProLLaMA Stage 1 / Unrelated | +0.3739 [+0.3402, +0.4065] | +0.1451 [+0.1127, +0.1784] | +0.0149 [-0.0179, +0.0444] |
| ProLLaMA Stage 1 / Homolog | +0.4751 [+0.4444, +0.5054] | +0.2821 [+0.2565, +0.3073] | +0.1154 [+0.0880, +0.1403] |
| ProLLaMA Stage 2 / None | +0.3784 [+0.3472, +0.4085] | +0.1594 [+0.1288, +0.1932] | +0.0242 [-0.0056, +0.0513] |
| ProLLaMA Stage 2 / Unrelated | +0.3226 [+0.2911, +0.3520] | +0.1245 [+0.0958, +0.1559] | +0.0124 [-0.0149, +0.0391] |
| ProLLaMA Stage 2 / Homolog | +0.4192 [+0.3898, +0.4486] | +0.2497 [+0.2230, +0.2761] | +0.1021 [+0.0764, +0.1262] |

The `P`-only held-family baseline is given above; each `P+M` predictor is refitted on the declared support. The following gains use paired assay outcomes, not differences between separately bootstrapped interval endpoints.

| Model / context | Held-family ρ(P+M, Y) [95% interval] | Held-family Δρ [95% interval] | Held-family ΔMSE [95% interval] |
| --- | --- | --- | --- |
| ProGen3-3B / None | +0.4246 [+0.3938, +0.4545] | +0.0518 [+0.0355, +0.0677] | +0.0468 [+0.0335, +0.0590] |
| ProGen3-3B / Unrelated | +0.3779 [+0.3455, +0.4090] | +0.005027 [+0.000620, +0.009597] | +0.003825 [-0.000291, +0.007759] |
| ProGen3-3B / Homolog | +0.3877 [+0.3561, +0.4171] | +0.0149 [+0.0064, +0.0228] | +0.0150 [+0.0084, +0.0216] |
| ProteinGLM-7B-CLM / None | +0.4285 [+0.3966, +0.4592] | +0.0556 [+0.0407, +0.0723] | +0.0505 [+0.0385, +0.0637] |
| ProteinGLM-7B-CLM / Unrelated | +0.4299 [+0.3981, +0.4613] | +0.0571 [+0.0420, +0.0735] | +0.0515 [+0.0393, +0.0648] |
| ProteinGLM-7B-CLM / Homolog | +0.4258 [+0.3939, +0.4572] | +0.0530 [+0.0382, +0.0679] | +0.0487 [+0.0366, +0.0600] |
| ProLLaMA Stage 1 / None | +0.3728 [+0.3396, +0.4048] | -0.000018 [-0.000045, +0.000002] | -0.000419 [-0.000524, -0.000325] |
| ProLLaMA Stage 1 / Unrelated | +0.3724 [+0.3392, +0.4043] | -0.000400 [-0.000659, -0.000130] | -0.000439 [-0.000848, -0.000050] |
| ProLLaMA Stage 1 / Homolog | +0.3873 [+0.3565, +0.4163] | +0.0144 [+0.0072, +0.0219] | +0.0139 [+0.0084, +0.0198] |
| ProLLaMA Stage 2 / None | +0.3733 [+0.3403, +0.4050] | +0.000448 [-0.000605, +0.001478] | -0.000006 [-0.001090, +0.001089] |
| ProLLaMA Stage 2 / Unrelated | +0.3726 [+0.3395, +0.4045] | -0.000242 [-0.000401, -0.000093] | -0.000338 [-0.000592, -0.000102] |
| ProLLaMA Stage 2 / Homolog | +0.3835 [+0.3517, +0.4131] | +0.0107 [+0.0051, +0.0169] | +0.0103 [+0.0055, +0.0150] |

### Additional ProLLaMA native support

These rows retain all 195 eligible assays, 158 families and 24,960 variants per stage. They are not directly paired with the smaller ProteinGLM/ProGen3 panels. The raw profile/effect and held-family `P`-only baselines on this support are given above. The full native and shared tables are both retained even when conclusions agree.

| Model / context | ρ(M, P) [95% interval] | ρ(M, Y) [95% interval] | Partial rank ρ controlling P [95% interval] |
| --- | --- | --- | --- |
| ProLLaMA Stage 1 / None | +0.3863 [+0.3556, +0.4158] | +0.1442 [+0.1164, +0.1744] | +0.0106 [-0.0158, +0.0373] |
| ProLLaMA Stage 1 / Unrelated | +0.3772 [+0.3471, +0.4072] | +0.1473 [+0.1169, +0.1791] | +0.0208 [-0.0080, +0.0497] |
| ProLLaMA Stage 1 / Homolog | +0.4604 [+0.4306, +0.4871] | +0.2777 [+0.2549, +0.3019] | +0.1197 [+0.0956, +0.1435] |
| ProLLaMA Stage 2 / None | +0.3819 [+0.3536, +0.4086] | +0.1590 [+0.1297, +0.1902] | +0.0280 [+0.0017, +0.0520] |
| ProLLaMA Stage 2 / Unrelated | +0.3294 [+0.3026, +0.3559] | +0.1235 [+0.0966, +0.1524] | +0.0128 [-0.0124, +0.0369] |
| ProLLaMA Stage 2 / Homolog | +0.4027 [+0.3746, +0.4290] | +0.2421 [+0.2164, +0.2680] | +0.1025 [+0.0801, +0.1269] |

| Model / context | Held-family ρ(P+M, Y) [95% interval] | Held-family Δρ [95% interval] | Held-family ΔMSE [95% interval] |
| --- | --- | --- | --- |
| ProLLaMA Stage 1 / None | +0.3610 [+0.3326, +0.3883] | -0.000263 [-0.000438, -0.000096] | -0.000323 [-0.000594, -0.000080] |
| ProLLaMA Stage 1 / Unrelated | +0.3613 [+0.3329, +0.3887] | +0.000013 [-0.000584, +0.000587] | -0.000250 [-0.000904, +0.000404] |
| ProLLaMA Stage 1 / Homolog | +0.3780 [+0.3514, +0.4031] | +0.0167 [+0.0096, +0.0245] | +0.0155 [+0.0100, +0.0210] |
| ProLLaMA Stage 2 / None | +0.3618 [+0.3339, +0.3890] | +0.000541 [-0.000650, +0.001725] | +0.000181 [-0.000923, +0.001319] |
| ProLLaMA Stage 2 / Unrelated | +0.3611 [+0.3327, +0.3884] | -0.000158 [-0.000324, -0.000006] | -0.000270 [-0.000508, -0.000042] |
| ProLLaMA Stage 2 / Homolog | +0.3729 [+0.3457, +0.3992] | +0.0117 [+0.0059, +0.0182] | +0.0109 [+0.0061, +0.0154] |

### Interpretation

Without supplied context, ProGen3-3B and ProteinGLM each correlate positively with the profile while also retaining positive partial rank association and improving held-family prediction on shared support. Their held-family Δρ values are +0.0518 [+0.0355, +0.0677] and +0.0556 [+0.0407, +0.0723], respectively, on 164 assays / 134 families / 20,992 variants. The corresponding ΔMSE values are positive as well. This establishes complementarity relative to this scalar control and fitted predictor; it does not show that the added signal is absent from richer profile-derived representations or identify a biological computation.

On shared support, the no-context held-family Δρ estimates are −0.000018 [−0.000045, +0.000002] for ProLLaMA Stage 1 and +0.000448 [−0.000605, +0.001478] for Stage 2, on 164 assays / 134 families / 20,992 variants. These small point estimates do not establish equivalence. Their homolog-context scores nevertheless add information to `P` even though their standalone correlation remains below the profile point estimate. On shared support, homolog-context Stage 1 has partial rank correlation +0.1154 [+0.0880, +0.1403] and held-family Δρ +0.0144 [+0.0072, +0.0219]; Stage 2 has +0.1021 [+0.0764, +0.1262] and +0.0107 [+0.0051, +0.0169], respectively, on the same 164 assays / 134 families / 20,992 variants. Positive ΔMSE and the native-support sensitivity accompany these results. Thus a standalone MODEL−LOOKUP deficit does not imply absence of complementary information under the homolog condition.

The two residual and prediction endpoints must remain distinct. For example, Stage 2’s no-context partial correlation on native support is +0.0280 [+0.0017, +0.0520], while its held-family Δρ is +0.000541 [−0.000650, +0.001725] on 195 assays / 158 families / 24,960 variants. On shared support even that partial association is unresolved. Some small negative prediction increments have intervals excluding zero, but their sizes must not be inflated into a general biological deficit. Conversely, an interval spanning zero is not evidence of equivalence or absence of all useful information. The historical failure-condition results below concern separate, larger variant draws and cannot substitute for these mutation-level tests.

## Historical failure conditions

The historical-score analysis completed on H200 on 2026-09-23. It identifies conditions associated with relative mutation-ranking performance, including a stronger ProteinGLM advantage when the external profile has fewer effective homologs and a concentration of the ProGen3-112M deficit among shorter proteins. These are descriptive associations, not explanations of parameter mechanisms or a universal separation of capabilities.

The [frozen protocol](D1_PROFILE_INCREMENT_PROTOCOL.md) defines weighting, covariates and limitations. The admitted artifact is `results/transfer/d1_followup_20260923/failure_conditions/profile_increment.json`, SHA256 `18f13fc372e4548b9969de5be4ce2abc2b44206b5bb092198db538d7c751942b`, from code freeze `20260923062123_250bf3018f43`. Its source manifest, per-assay values, all strata, regression coefficients and diagnostics permit replay. This historical-score artifact contains no mutation-level residual or cross-validation results; those are reported above from the separate retained mutation vectors.

### Overall comparison

Each value is family-equal mean MODEL−LOOKUP Spearman, a dimensionless difference, with a 95% interval from 2,000 family-bootstrap draws. Supports remain arm-specific; these are not common-support cross-model comparisons. The bootstrap seed differs from earlier summaries, so interval endpoints can differ while point estimates replay exactly.

| Model | Assays / families | MODEL−LOOKUP [95% interval] |
|---|---:|---:|
| ProGen3-112M | 217 / 174 | −0.0804 [−0.1147, −0.0481] |
| ProGen3-3B | 217 / 174 | +0.0568 [+0.0357, +0.0785] |
| ProteinGLM-7B-CLM | 201 / 163 | +0.0384 [+0.0169, +0.0576] |
| Llama-2-7B | 217 / 174 | −0.3621 [−0.3923, −0.3328] |
| ProLLaMA Stage 1 | 217 / 174 | −0.2193 [−0.2540, −0.1866] |
| ProLLaMA Stage 2 | 217 / 174 | −0.2043 [−0.2365, −0.1737] |
| ProGen2-small | 201 / 163 | −0.0554 [−0.0808, −0.0305] |
| ProGen2-base | 213 / 171 | +0.0037 [−0.0261, +0.0321] |
| ProGen2-medium | 201 / 163 | −0.0043 [−0.0300, +0.0221] |
| ProGen2-large | 201 / 163 | +0.0122 [−0.0114, +0.0357] |
| ProGen2-xlarge | 201 / 163 | +0.0039 [−0.0145, +0.0222] |
| ProtGPT2 | 214 / 172 | −0.0143 [−0.0378, +0.0090] |

### Conditions associated with relative performance

Depth strata use cut points of 29.19 and 180.83 effective homologs, fixed from the complete catalogue of 187 wild types. ProteinGLM's MODEL−LOOKUP is +0.0796 [+0.0303, +0.1278] in the low-depth stratum (58 assays / 46 families), +0.0342 [+0.0041, +0.0646] in the middle (66 / 58), and +0.0105 [−0.0153, +0.0370] in the high (77 / 59). The joint regression's depth coefficient is −0.0343 [−0.0688, −0.0048] Spearman difference per standard deviation of log10 depth, conditional on the other covariates and at mean entropy (201 assays / 163 family observations).

This is relative advantage, not poorer absolute prediction on deeper profiles. ProteinGLM's raw Spearman rises from +0.3420 [+0.2957, +0.3864] in the low-depth stratum to +0.4303 [+0.3896, +0.4706] in the high-depth stratum, while LOOKUP rises from +0.2624 [+0.2242, +0.3015] to +0.4199 [+0.3783, +0.4591], on the same respective supports. These estimates support a sparse-profile advantage on this searched corpus, not a claim that performance improves when homolog information is removed.

ProGen3-3B retains positive MODEL−LOOKUP in every depth stratum: +0.0875 [+0.0358, +0.1347] (69 assays / 55 families), +0.0483 [+0.0155, +0.0817] (69 / 61), and +0.0357 [+0.0111, +0.0609] (79 / 60). Its adjusted depth coefficient is −0.0238 [−0.0611, +0.0146] per standard deviation (217 / 174); the joint analysis does not resolve a depth dependence. Positivity of one stratum and uncertainty in another would not itself establish an interaction.

Length strata have fixed cut points at 72 and 402 residues. ProGen3-112M's MODEL−LOOKUP is −0.1762 [−0.2412, −0.1137] in the short stratum (60 assays / 60 families), −0.0654 [−0.1113, −0.0155] in the middle (80 / 59), and −0.0001 [−0.0464, +0.0450] in the long (77 / 57). Its adjusted length coefficient is +0.0719 [+0.0234, +0.1193] per standard deviation of log10 residue length (217 / 174). The depth-by-entropy interaction is +0.0405 [+0.0109, +0.0739] per product of standardized covariates on that support. Both are exploratory, unadjusted intervals; neither identifies a molecular cause. The ProLLaMA stages remain below LOOKUP across every length stratum despite their positive adjusted length coefficients.

### Limits of the available covariates

The nearest-hit identity distribution is concentrated near exact matches. Its tertile cut points are 99.7768% and 100% query-normalized identity; the middle stratum contains only four assays / three families and its interval is withheld. The highest stratum contains 142 assays / 120 families in the full-support arms. The lowest tertile must not be described as remote homology. This frozen corpus cannot supply a balanced low-identity failure analysis using tertiles; the newly selected context homologs address a different identity question.

Depth information exists, but the DIAMOND hit-list cap is saturated for 56 / 217 assay rows in the full-support arms and 53 / 201 in the shorter-window arms. Effective depth is therefore a property of retained hits with right-censoring, not the complete number of homologs. The inherited log-depth variable maps effective depth below one to zero. Entropy and depth correlate +0.6080 across the 174 full-support family observations; regression design condition numbers range from 3.85 to 3.98 across arms, and every arm retains all 2,000 bootstrap fits. These finite-design diagnostics carry no population interval. No rank deficiency was observed, but adjusting for these covariates does not remove assay composition, measurement quality, corpus coverage or other unmeasured differences.

All strata and coefficients are retained in the artifact. The selected patterns above should motivate replication rather than selection of a new confirmatory hypothesis from these same outcomes.
