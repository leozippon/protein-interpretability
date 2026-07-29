# Interpretability transfer audit: text to protein generative models

**Status:** active; canonical analysis document **Supersedes:** repository-root `check.md`, which is frozen and no longer maintained **Updated:** 2026-07-29

---

## 0. Objective

1. **Differences.** Explore the differences between language-generative and protein-generative models.
2. **Limitations.** Based on existing interpretability methods for language models, analyse the limitations of those approaches when applied to protein generative models.
3. **Adapted methods.** Propose interpretability methods suitable for protein generative models.

Parts 2 and 3 are the deliverable. Part 1 is instrumental — a difference between model families matters here only insofar as it explains why a method transfers badly.

**This document reorganises the entire experimental record against that objective.** The record was accumulated while the programme was optimising for part 1 as though it were terminal, so a substantial amount of part-2 evidence exists as an unclaimed by-product of retractions and negative results. Recovering it is the main purpose of this reorganisation.

---

## 0.05 Retraction — the homology / memorisation control (EXP-R2-061)

> **Resolved by EXP-R2-064 (2026-07-29).** The search was re-run against the full UniRef50 snapshot with `--masking 0`, CPU only. Record 0 now returns 100% identity over all 732 residues and bins `ge95_near_duplicate`, so the fix took. The outcome splits by statistic. **Head count — the quantity the finding claims — survives and is reinstated**: no stratum separation on any arm, overlapping intervals everywhere, and the unmemorisable synthetic probe recruits as many heads as the near-duplicate stratum or more. **Peak prefix-matching strength reverses**: the pre-stated rule now returns `consistent_with_memorisation` on all three arms, ProtGPT2 having moved from `indeterminate` — reported as a reversal, though the repeat-length confound and EXP-R2-049 had already put that statistic out of use. The `lt30_no_detectable_homology` stratum turned out not to exist: all four of its members were truncated alignments, two of them verbatim corpus entries at 100% identity. The ZymCTRL falsification is untouched. Artefacts under `results/transfer_20260729/`; the retracted ones are kept as the evidence for this retraction. The paragraphs below stand as the record of the defect.

**`EXP-R2-050 "the induction head count is not memorisation" moves from *stands* to *retracted pending re-run*.**

`homology.run_diamond_blastp` built its DIAMOND command **without `--masking`**, so DIAMOND 2.1's default repeat masking was active — against a cohort **selected for internal tandem repeats**. The HSP stops at the repeat, `nident` under-counts, and `identity_over_query = nident/qlen` is not the query coverage its docstring claims.

Verified against the corpus rather than inferred: cohort record 0 is **byte-identical to `UniRef50_Q3E8Z8` over all 732 residues**, yet DIAMOND reported `pident 100, nident 607, qend 607, slen 732` → 82.9% → binned `id70_to_95_close_homology`. **A verbatim member of ProtGPT2's pretraining corpus was recorded as a diverged relative.** Five of forty-eight exact-cohort records are affected, all in the same direction, and all into the bin carrying the *highest* measured induction — i.e. the error works to defeat the memorisation hypothesis it was meant to test.

Compounding it: `bootstrap_stratum` had no unit floor. At n=4 a percentile interval is *pinched inward* — 9–15% relative width against 27% at n=420 in the same file — and two `consistent_with_memorisation` verdicts were decided by non-overlap of a four-unit interval against a four-hundred-unit one.

Re-run is `--stages search` only, **no GPU**. Note that `assign_homology` now raises on a truncated alignment by design, so re-running `--stages assign` over the existing masked TSV will stop rather than silently reproduce the old bins.

**Two further published numbers are now known to rest on file-order draws** and must be re-derived before citation:

- **L9's explanation-channel figures** (7.32 / 3.61 / 0.74 bits/symbol) — the stage takes an accession-order prefix while neighbouring modules permute. The caveat previously recorded against these numbers is now actionable, and a seeded sampler exists.
- **EXP-R2-040 probe and erasure numbers** — grouping units took corpus prefixes. Measured on the full EC corpus: **135 distinct grouping units under file order against 231 under a seeded permutation** of the same 400 proteins, with the EC-1 class share more than doubled. Separately, `family_disjoint=True` was asserted for `fitness` while ProteinGym ships four `BLAT_ECOLX` assays of a single 286-residue protein, and the behaviour cohort was the first N held-out units — for fitness, **forty mutants of one protein**.

And one interval is renamed rather than deleted: the **path-patching matched-pair result** (+0.020, CI [−0.121, +0.166]) resampled an arm's induction *heads* — its entire population, selected by a threshold — while the probe records, the only real sampling unit, contributed nothing. It measures heterogeneity, not sampling error. The point estimate stands; the interval does not mean what it said.

## 0.15 TG-01 is not quotable (EXP-R2-066)

`tg01_information_budget.py` **could not run at all**: `--seed` was registered twice, so argparse raised at parser construction. Verified directly, and consistent with the corrected results tree, which holds `tg00`, `tg03`, `tg07` and `tg09` and **no `tg01/`**.

So TG-01's only outputs sit in the retracted 2026-07-24 tree, produced under the broken ProtGPT2 rendering. Its row in §0.1's blast-radius table should be read as **"not re-run"**, not "corrected". No TG-01 number — unigram 8.872, NLL 7.296, gain 2.636 bits, top-1 0.102, the truncation curve — may be cited until the stage runs.

## 0.1 CLOSED — TG-series rendering contamination repaired; the dissociation is retracted (EXP-R2-062)

**Resolved 2026-07-29.** The defect described below is fixed, its blast radius is measured, and B2 has been run. `tg_common.py` no longer renders anything: it is an adapter over `src/transfer/arms.py`, and every TG cohort is now a seeded permutation of the complete eligible set. Corrected artefacts are under `results/transfer_gap_20260729_corrected/`; the 2026-07-24 tree is retained unmodified because it is cited here; pre-correction sources and a per-number status table are frozen under `/Data2/lzp/bio_archive/legacy/r2_transfer_gap_precorrection_20260729/`.

**B2's binary gate returns NO.** The variance–behaviour dissociation does not reproduce. ProtGPT2's rank-512 loss recovered moves **−0.105 → +0.879** and its alignment gap at rank 512 falls **+1.102 → +0.118**, below ProGen2-medium's +0.326. GPT-2-large is unchanged to three decimals, which is what establishes that the movement is the rendering and not the cohort. On the maximum gap over rank, **ZymCTRL (+0.289) sits 2.4x below the text control (+0.708)**: the decoupling is not modality-conditional. **C4 (reliance-weighted dictionaries, 20–40 GPU-h) loses its stated motivation and must not be started on this basis.**

**A second, independent error in the same result.** The variance-concentration premise was mismeasured for every arm, under either rendering. `PC1` and the participation ratio over all positions are dominated by structurally special tokens. On interior, alphabet-bearing positions: gpt2-large PC1 0.034 / PR 253.4 (its all-position 0.809 / 1.53 is its first token, at 24.8x the interior residual norm); ProtGPT2 0.439 / 4.86 (its 0.971 / 1.06 is the FASTA newline the *correct* rendering introduces); ZymCTRL 0.008 / 577.3; ProGen2-medium 0.520 / 3.69. The recorded contrast "ProtGPT2 0.951 against gpt2-large 0.809" compared two artefacts of different origin. The protein arms do not cluster and the highest effective dimension in the panel is a protein arm. Same hazard class as L6.

**What survives, and is now stronger.** L3 — FVU does not track behavioural fidelity — is demonstrated on clean data: ProtGPT2 has the panel's best FVU by 10x (0.0051) and ranks third of four on loss recovered (0.706), while ZymCTRL has the worst FVU by 50x (0.2525) and ranks second (0.763). Spearman(FVU, loss recovered) = 0.000 corrected, against +0.400 before. TG-09's "~0.35 at every depth" is retracted: ProtGPT2 reads 0.996 / 0.977 / 0.354 / 0.609 / 0.715 across depths, making it the *most* depth-dependent arm rather than a flat one.

**New standing rule (Appendix B).** Report residual-stream spectra on interior, alphabet-bearing positions. A participation ratio over all positions measures the attention sink and the separator tokens.

The original description follows, unchanged, because the reasoning is the record.

---

`scripts/transfer_gap/tg_common.py::protein_input` returned the **plain sequence** for ProtGPT2. Its docstring stated the problem and overrode it:

> "ProtGPT2 was trained on FASTA-like text with newline separators; ZymCTRL and ProGen2 use residue-level vocabularies. **We use the plain sequence for all three** so the compared quantity is next-residue prediction on identical content."

The rendering fix later measured that choice at **1.42 nats/token**, moving ProtGPT2's context information from −1.31 to +2.23 (L11 below). The withdrawal that followed covered some ProtGPT2 rows but **not** TG-03, TG-06, TG-07 or TG-09.

**Consequence.** Results resting on a contaminated ProtGPT2 arm include TG-03 (FVU 0.021, loss recovered 0.336), TG-07 (PC1 0.951, participation ratio 1.1, rank-512 loss recovered −0.105) and TG-09 (~0.35 at every depth). The **variance–behaviour decoupling** result is the load-bearing casualty: it is currently recorded as solid and is the entire motivation for the reliance-weighted dictionary line.

**Action, before anything is built on it:** re-run TG-03 and TG-07 under `fasta_wrapped` on a seeded-permutation cohort (~2 GPU-h). If the dissociation does not reproduce, it is retracted and the method line built on it is dropped. Until then, treat those numbers as provisional. File-order exposure for the same runs is likely minor — the effect converges by n = 200 and these used n = 400.

*Done, EXP-R2-062, 1.8 GPU-h. The dissociation did not reproduce; see the header of this section. The file-order expectation was correct in magnitude and wrong to dismiss: at n = 200 it is worth +0.240 nats/token on ProGen2-medium and −0.079 on ProtGPT2 — small beside the 1.78-nat rendering delta, not zero, and arm-dependent in sign.*

## 1. Headline position, as of today

If the work stopped now, the defensible claims are:

**On part 1 (differences).** One measured difference survives in weakened form: on synthetic repeat probes, protein decoders have fewer heads in the *upper tail* of the induction-score distribution than text decoders, after adjusting for a real within-lineage scale effect. It is **not** a general distributional difference, it **inverts** at the headline threshold on the most defensible probe, and its modality increment is carried by a single protein model. It should not be presented as a modality finding.

**On part 2 (limitations).** This is where the programme's real yield is, and much of it was demonstrated *on the text control*, which makes it a property of the method rather than of protein models. See §4.

**On part 3 (adapted methods).** Not yet earned. Two candidate directions carry standing rejections (§6.1) and no proposal is currently traceable to a measured limitation with adequate evidence.

**The strongest directly measured part-2 result available** is narrower: `paa_specific` does not provide a valid cheap ranking screen for copy-suppression on GPT-2-large, despite the causal mechanism itself being detectable. This result does not establish that head-prevalence censuses generally have a narrow domain, nor does it rule out another screen or an exhaustive causal effect-size census. Measurability remains a candidate contributor to an apparent transfer gap, not an established general explanation.

---

## PART I — Differences between text and protein generative models

### 2. The panel

Eleven autoregressive decoders under one code path, each fed in the format it was trained on. All were staged and load-checked on GPFS in EXP-R2-058; ten checkpoints received full source-to-GPFS SHA-256 verification, while Qwen2.5-0.5B was recorded as byte-size checked only.

| arm | modality | layers x width | tokenisation | corpus / conditioning |
|---|---|---|---|---|
| gpt2 | text | 12 x 768 | BPE 50257 | WebText |
| gpt2-medium | text | 24 x 1024 | BPE 50257 | WebText |
| gpt2-large | text | 36 x 1280 | BPE 50257 | WebText |
| gpt2-xl | text | 48 x 1600 | BPE 50257 | WebText |
| dialogpt-small | text | 12 x 768 | BPE 50257 | Reddit dialogue |
| qwen2.5-0.5b | text | 24 x 896 | BPE 151936 | Qwen2.5 mixture |
| llama-3.2-3b | text | 28 x 3072 | BPE 128256 | Llama-3 web corpus |
| protgpt2 | protein | 36 x 1280 | multi-residue BPE 50257 | UniRef50, FASTA |
| zymctrl | protein | 36 x 1280 | residue 458 | EC-labelled, tag-conditioned |
| progen2-base | protein | 27 x 1536 | residue 32 | corpus contrast partner |
| progen2-medium | protein | 27 x 1536 | residue 32 | corpus contrast partner |

**Design properties that make specific contrasts identifiable:**

- `gpt2-large` / `protgpt2` — identical architecture, depth, width and vocabulary size, and identical parameter count (773,891,840). The matched modality pair.
- `gpt2` / `gpt2-medium` / `gpt2-large` / `gpt2-xl` — within-lineage scale ladder, 124M to 1558M, holding architecture, tokeniser and corpus fixed.
- `gpt2` / `dialogpt-small` — same architecture and size, different corpus. Text-side corpus contrast.
- `progen2-base` / `progen2-medium` — matched architecture and parameter count, different corpus. Protein-side corpus contrast.
- `qwen2.5-0.5b`, `llama-3.2-3b` — cross-lab lineage contrast at comparable scale.

**Known structural limit:** the only model family spanning both modalities is GPT-2, with five text arms against one protein arm. Every modality coefficient is therefore carried by ProtGPT2 alone.

### 3. What was claimed, and what survived

| claim | status | why |
|---|---|---|
| Fewer induction heads, not weaker | **survives, weakened** | tail-only; probe-dependent; n=1 protein identification |
| QK/OV dissociation is modality-specific | **retracted** | GPT-2-lineage property; Qwen 0.357 and Llama 0.445 sit at chance beside ProtGPT2 0.361, while ProGen2 copies at 1.000 |
| Attention pathway more indirect in protein | **refuted** | path patching, matched pair +0.020 CI [−0.121, +0.166] |
| Relational structure readable only from attention | **retracted** | cohort-selection artefact; margin fell ~0.10 → 0.03–0.05 under seeded permutation, ~0.03 over separation-only control |
| Tokenisation explains the modality gap | **retracted** | plug-in unigram estimator biased shares by up to +1.02 nats |
| Aperture / J-Lens axis separates modalities | **null** | data contrast exceeds modality effect; instrument too noisy to resolve anything |
| MLP-share modality coefficient | **underpowered** | collapses under tokenisation adjustment at n=9 |
| Induction head-count deficit is explained by within-protein homology | **not supported within the measured protein cohorts; cross-modal contribution unresolved** | corrected full-UniRef50 control finds overlapping head-count strata and a synthetic negative control, but no genuine low-homology stratum or symmetric GPT-2 corpus comparison |
| Induction deficit is exact-repeat artefact | **refuted** | survives approximate-repeat criterion |
| ProtGPT2 is off-distribution | **retracted (own defect)** | input-rendering bug; EOT + 60-col wrap worth 1.42 nats/token |

### 4. Final status of the induction result

**Threshold robustness (synthetic probe) — passes.** Ordering holds at 0.05, 0.10, 0.20, 0.30 and the data-driven cut, and continuously across raw cuts 0.00525–0.4196, 62% of the informative grid with all four fixed cuts interior. The 0.10 threshold was not cherry-picked.

**Distributional separation — fails.** Pooled AUC 0.595; 8 of 12 pairs above 0.5; **0 of 12** show stochastic dominance; quantile dominance fails at q50, q99.5 and q100; model-level median test p = 0.257. ProGen2-medium's best head reaches 87.6x uniform, beating every text arm but two. **The result is a tail statement, not a distributional one.**

**Probe dependence — fails on the best probe.**

| probe | ordering |
|---|---|
| synthetic | holds at every threshold |
| natural exact | breaks at 0.20, 0.30 |
| **natural approximate** | **inverts at 0.10** (dialogpt-small 0.0000, ProtGPT2 0.0097), p = 0.114 |

`circuits.py` records the natural probe as the one to trust for wrapped arms and `scaling.py` declares it primary; the headline table used the synthetic probe. The natural approximate criterion is the one Pomerants et al. showed subsumes exact detection.

**Scale — real, and does not explain the descriptive shortfall.** The within-lineage ladder falls monotonically: 0.1597, 0.1380, 0.0972, 0.0833; slope **−0.272/decade [−0.455, −0.088]** with head *count* rising 23 → 100. This explains llama-3.2-3b's low value. Restated against the ladder's scale-matched prediction, the shortfall is **2.34x descriptively**; the artifact does not report a scale-adjusted inferential test. Its reported **p = 0.0286** belongs instead to the unadjusted, one-sided exact permutation test over four text and three protein models for `fraction_above_0.10`, where complete separation reaches the minimum attainable p-value. Variance decomposition: scale adds **+0.003** given modality and lineage; modality +0.220; lineage +0.061 — but only **25.4%** of the modality indicator survives projection.

**Corpus — roughly half the modality contrast in log terms.**

| contrast | ratio |
|---|---|
| ~~gpt2 / dialogpt-small~~ | ~~2.30x [2.30, 2.40]~~ **RETRACTED — see §5.05(a); dialogpt-small is off-distribution at −4.08 nats context information and cannot anchor a contrast** |
| progen2-medium / progen2-base | 2.00x [1.20, 2.00] |
| gpt2-medium / qwen2.5-0.5b | 1.93x [1.82, 2.04] |
| gpt2-xl / llama-3.2-3b | 2.15x [1.92, 2.29] |
| **modality, matched pair** | **5.38x [5.23, 5.54]** |

**The identification limit, which no experiment can remove.** Induction heads are useful only on corpora containing repeated subsequences. Approximate repeats occur in **32.3% of text documents against 0.402% of protein entries** — an eightyfold difference. A decoder allocating fewer heads to prefix-matching on a repeat-poor corpus is allocating capacity efficiently against its data. Breaking the confound would require a protein decoder trained on a repeat-rich corpus, which cannot exist: a synthetic protein corpus with 32% repeat prevalence is not a protein corpus. **Even the terminal 2x2 training design cannot fix this**, as it matches architecture, parameters, scale and budget but explicitly not the data-generating process.

---

## PART II — Limitations of transferred interpretability methods

### 5. The catalogue

**The organising distinction:** a limitation demonstrated *on the text control* is a property of the **method**; one that appears only on protein arms is a property of the **transfer**. The programme repeatedly discovered the first kind and recorded it as an embarrassment rather than a result. The `scope` column is the spine of this section.

| # | limitation | method family affected | evidence | scope |
|---|---|---|---|---|
| L1 | Recovery gates set without checking attainability; single-MLP estimand has ~0.02 nats/token causal footprint against an 80% gate | SAE / transcoder / CLT | P0-2b (`docs/analysis/P0_2B_DICTIONARY_FIDELITY_RESULTS_20260727.md`, `evidence/p0_2b_fidelity_20260727/`, `evidence/p0_2_adjudication_20260727/`); estimand power sweep | **method** — unattainable on the text control |
| L2 | A 0.1-nat mutual-information gate applied to a design with a 0.0066-nat analytic ceiling | attribution / explanation channel | explanation-channel analysis | **method** |
| L3 | FVU is not comparable across models; ranking by FVU does not track behavioural fidelity | SAE / CLT | TG-03, TG-08 budget sweep | **method** |
| L4 | Loss-recovered denominators are arm-specific; ablation-baseline choice moves the measured denominator substantially | SAE / CLT, activation patching | denominator-guard work | **method** |
| L5 | The tested `paa_specific` proxy does not rank copy-suppression heads by measured causal effect on GPT-2-large (Spearman −0.062, p = 0.71), and the screened positive control has little dynamic range (5/56 heads vs induction 70/720) | this PAA screen for a copy-suppression prevalence census | EXP-R2-059 | **method-screen limitation on one text model** — no protein arm was scored; other screens and exhaustive causal censuses remain open |
| L6 | Positional baseline dominates the *body* of per-head attention statistics; decoy correction drops head median sixfold (0.0083 → 0.0013) while leaving the tail count intact (70 → 69, Jaccard 1.000) | attention-pattern analysis, circuit census | EXP-R2-059 | **method** |
| L7 | Attribution graphs hold attention patterns fixed and do not explain how they are computed; the price is measurable | attribution graph | TG-06 frozen attention | **method** |
| L8 | Output aperture is rank-(V−1); lens-family readouts are bounded by vocabulary size, which differs 1600-fold across the panel (32 to 151936) | logit lens, tuned lens, DLA, J-Lens | EXP-R2-031, EXP-R2-045 | **transfer** — algebraic and vocabulary-conditional, not modality-specific |
| L9 | Explanation closure: an explanation drawn from an annotation channel cannot exceed that channel's information content | natural-language autoencoder, automated interpretability | analytic | **transfer** |
| L10 | J-Space / property-conditioned Jacobian is not defined for properties that are not functionals of the next-token distribution; directions come from a probe, making the object the Jacobian of the probe | J-Lens / J-Space | analytic, three independent reviews | **method**, fatal |
| L11 | Input rendering is load-bearing and silently wrong by default; ProtGPT2 moved **1.42 nats/token** between raw sequence and EOT + 60-column FASTA wrapping | every method that scores sequences | EXP-R2-028 | **transfer** |
| L12 | Plug-in unigram estimators bias information decompositions by up to **+1.02 nats** on protein arms vs +0.009 at residue level; held-out estimation is required | information-theoretic decomposition | EXP-R2-031, EXP-R2-033 | **transfer** — bias is vocabulary-dependent |
| L13 | Iterating a biological corpus in file order manufactures effects; three separate instances, most recently ProGen2 context information moving **+1.01 nats** by reading past the first 48 records | cohort construction, all methods | EXP-R2-059, §0B.2 | **transfer** |
| L14 | Aligned-pair task construction collapses coverage on subword arms, 0.943 → 0.421 | probe design, circuit analysis | EXP-R2-048 | **transfer** |
| L15 | Conditioning leak: ZymCTRL's EC tag supplies 1.73 nats, inflating confidence-conditioned statistics | probes, erasure, any conditioned readout | EXP-R2-034 | **transfer** |
| L16 | Decodability and reliance dissociate sharply; probe accuracy does not establish that the model uses the feature | linear/MLP probe | EXP-R2-040 | **method** |
| L17 | An intervention that moves everything needs a control for moving everything; raw A6 numbers read −0.82 vs −0.83 (indistinguishable), against a matched random-direction control −0.441 vs −0.208 | activation patching, steering, path patching | EXP-R2-059 | **method** |
| L18 | An environment default silently narrowed a nine-stage campaign's text side to **one model** while every downstream number remained well-formed | any multi-arm pipeline | EXP-R2-057 | **method** |
| L19 | RMSNorm-on-LayerNorm algebra passes a reconstruction gate at 0.49% error while producing systematically wrong attribution; the loud failure (omitted bias, 0.518) is safer than the quiet one | OV/QK circuit analysis on modern architectures | EXP-R2-055 | **method** |

### 5.05 Instrument-transfer campaign, 11 arms, 4x H200 (EXP-R2-060)

Four stages across the full panel under one code path. Five results change the catalogue.

**(a) `dialogpt-small` is unmeasurable, and this retracts a corpus contrast.** Clean CE 11.62 against a held-out unigram of 7.54 — context information **−4.08 nats**. It is unmeasurable the same way at all four stages. **The gpt2 / dialogpt-small 2.30x "text-side corpus contrast" recorded in §4 is therefore not valid**, because one of its two arms is off-distribution on the evaluation corpus. The protein-side contrast (progen2-medium / progen2-base, 2.00x) stands. Corpus effects remain real but the text-side estimate must be re-derived from an in-distribution pair.

**(b) Cohort-block sensitivity is 10–35x larger on the protein side.** Between disjoint corpus blocks, under a seeded permutation within each, on identical code:

| arm | shift (nats) | | arm | shift |
|---|---:|---|---|---:|
| every text arm | ≤ **0.018** | | protgpt2 | **+0.599** |
| | | | progen2-base | +0.231 |
| | | | progen2-medium | +0.160 |
| | | | zymctrl | +0.043 |

No verdict flips, so measurability is robust — but **a protein cohort statistic carries a few-tenths-of-a-nat selection uncertainty that the matched text statistic does not**, of the same order as several effects this programme has previously reported. Corroboration from the same run: file-order versus seeded draw moved progen2-medium's clean CE by **0.77 nats** on an identical band. This is a new, quantified, protein-specific limitation (extends L13).

**Two qualifications from EXP-R2-063, neither of which retracts it.** `--cohort-skip` moves the scored cohort *and* the held-out unigram reference together — the skip=4000 pool digest is byte-identical to the skip=0 reference digest — so the shift is not a pure cohort effect on a fixed baseline. Decomposed, ProtGPT2's +0.599 is **+0.041 baseline against −0.558 clean CE**: the movement is overwhelmingly the model's, not the estimator's, so the finding stands and the magnitude is real. Separately, the "seeded" draw is seeded **within a head-of-file pool of 4000 records**, not over the corpus, so it bounds within-pool selection sensitivity rather than corpus-wide selection sensitivity. The true figure is plausibly larger, not smaller. `protein_cohort` now accepts a `seed`, so a draw over the whole corpus is available and would be a stronger measurement than either block.

**(c) The single-submodule estimand fails by DEPTH, not modality — L1 revised.** `mlp_single@d0.50@cohort_mean` across the within-lineage ladder:

| gpt2 (12L) | gpt2-medium (24L) | gpt2-large (36L) | gpt2-xl (48L) |
|---:|---:|---:|---:|
| 0.0836 **powered** | 0.0334 | 0.0193 | 0.0148 |

Attainability falls **monotonically with depth**, and three of four protein arms clear the guard (protgpt2 0.111, progen2-base 0.066, progen2-medium 0.062; zymctrl 0.015 does not). So the estimand's failure on gpt2-large is a **property of the estimand and the model's depth, not of modality**. A recovery gate built on it penalises deep text and deep protein models alike, for a reason having nothing to do with dictionaries. This strengthens L1 and sharpens it: the defect is not merely "unattainable on one control" but "attainability is depth-dependent and was never checked against depth".

**52 of 76 estimands are attainable on gpt2-large.** The unattainable set is exactly *every* single-submodule estimand at all five depths and both baselines, plus `attn_window4` at d0.33 and d0.85. Recommended estimand: `attn_window4@d0.15@cohort_mean`; 50 are powered panel-wide.

**(d) Pathway budget separates cleanly — the strongest part-1 result to date.** MLP-to-attention ablation-cost ratio, **ranges do not overlap**:

| text | ratio | protein | ratio |
|---|---:|---|---:|
| qwen2.5-0.5b | 2.10 | protgpt2 | 1.12 |
| gpt2-xl | 2.01 | progen2-medium | 0.93 |
| gpt2-medium | 1.88 | progen2-base | 0.92 |
| gpt2 | 1.77 | zymctrl | 0.55 |
| llama-3.2-3b | 1.69 | | |
| gpt2-large | 1.50 | | |

Matched pair 1.50 against 1.12. **It survives replacing GPT-2-large with a Qwen2 and a Llama decoder** — different labs, rotary/RMSNorm/gated-FFN, vocabularies 2.5–3x larger — so it is not a GPT-2 idiosyncrasy, which is what killed the QK/OV finding. **Two caveats that must travel with it:** these are ratios to context information rather than a partition (values > 1 mean ablation costs more than all extracted context information), and **the tokenisation adjustment that collapsed the earlier MLP-share coefficient has not been applied.** Not yet a claim.

**(e) The tuned lens has no transfer failure.** *Qualification (EXP-R2-063): the lens stage draws its protein cohort on residues **64–120**, while `cohort_power` qualifies arms on **64–246** and the pathway and estimand stages measure on 64–246. So the arms this row scores were qualified on a different population, and (b) above prices protein cohort-block sensitivity at 0.16–0.60 nats. The band is a compute choice — the Jacobian sweep is quadratic in length — and it is now declared in the stage's artefact beside the qualifying band rather than being recoverable only by comparing argparse defaults. Nothing here is retracted; the cross-band comparison is weaker than it reads.* It improves on the logit lens at every non-identity layer on every scored arm, protein included — the only instrument in the campaign with a clean transfer. The Jacobian finite-difference guard passed on all nine scored arms, max relative error 4.7e-3 (ProtGPT2) against a 2e-2 tolerance, confirming that the earlier 1.008 failure was the forced `--dtype bfloat16` rather than an instrument defect. ProtGPT2 is the outlier on top-1 agreement (0.013–0.014, 5–10x below every other arm), consistent with its multi-residue-BPE aperture.

Two rotary arms were **not** scored and the skip was logged with its reason: `lens_head` requires `nn.LayerNorm` at `transformer.ln_f`, which an RMSNorm decoder lacks.

**Qualification from EXP-R2-063:** the lens stage measures on residues **64–120** while `cohort_power` qualifies arms on **64–246**. The cohort that was certified measurable is not the cohort that was scored. Not retracted — the tuned-lens-beats-logit-lens result is within-arm and within-band — but **the cross-band reading is weaker than it appears**, and the band is now declared in every artefact.

**Two failures were caught by the stage scripts' own guards** rather than by producing a wrong number, and both are instructive: a Markov "held-out" block offset by *record count*, which is not disjointness because Swiss-Prot carries the same sequence under several accessions; and a control-anchored aggregation receiving all seven text arms. Both fixed.

### 5.1 Quantitative anchors for the catalogue

The figures that carry the most weight, with their provenance, so the catalogue can be cited without re-deriving them.

**L1 — the original single-submodule gate is depth-dependent and unattainable on its matched text control.** In the latest EXP-R2-060 sweep, `mlp_single@d0.50@cohort_mean` costs gpt2-large **0.0193 nats/token** (bootstrap q025 0.0171) against a 0.05-nat guard, while ProtGPT2 (0.1115), ProGen2-base (0.0656) and ProGen2-medium (0.0622) clear it and ZymCTRL (0.0152) does not; **52 of 76** alternatives are attainable on gpt2-large and 50 are powered panel-wide. This makes the matched-panel or cross-domain interpretation of P0-2b inconclusive: P0-2b had no matched text dictionary, and the original ProtGPT2 and ZymCTRL recovered ratios were denominator-invalid. It does **not** erase the valid within-arm negative for ProGen2-medium under P0-2b's original estimand and frozen 0.80 gate: all nine ProGen2-medium runs had valid denominators, and the best bootstrap upper bounds were 0.411 for loss recovered and 0.282 for KL recovered.

**L5 — no valid `paa_specific` screen for copy-suppression on GPT-2-large.** `paa_specific` rank against measured ΔM-gap: Spearman **−0.062 (p = 0.71)** over 40 heads; 5 of 56 heads clear the control band against induction's 70 of 720. This does not test another proxy or an exhaustive causal effect-size census.

**L8 — the aperture is algebraically forced.** Numerical rank of `∂logits/∂h_l` is exactly **31** for ProGen2-medium and **457** for ZymCTRL, at every layer and probe, against 1150–1278 for the 50k-vocab arms. Validated against central finite differences at 2.2e-4–2.5e-3 relative error. **Vocabulary-conditional, not modality-conditional** — ProtGPT2's interface is near-full-rank. The honest label is *protein-typical, not protein-necessary*.

**L9 — explanation closure, measured.** Bits/symbol within a protein: text token identity **7.32**, residue identity 4.11, structural-oracle attributes **3.61** at coverage 1.00, **Pfam domain label 0.74** at coverage 0.56. *Caveat: this is single-run with declared source-order bias and must be re-derived before it is quoted. As of EXP-R2-063 the structural channel is drawn under a seeded permutation (`--structure-seed`), so **3.61** is now re-derivable; the Pfam and text channels are still corpus-prefix draws and the artefact records that per channel, so **7.32** and **0.74** still carry the bias.*

**L2 — the event-selection ceiling.** `I(E;L) ≤ h(m/N)`. The gate demanded 0.1 nats from a top-100-of-122,671 design whose ceiling is **0.0066 nats** — 15.1x above attainable. The reported `0/38 FAIL` carries no biological content.

**L16 — decodability is not reliance.** ProtGPT2 encodes Pfam at **+0.705** linear skill while erasing it *helps* next-residue prediction by **−0.179 nats** [−0.245, −0.097]. ProGen2-medium relies on ss3 at only +0.031 nats.

**New — residue-level annotation is refused outright on subword protein arms.** `probe_erasure/protgpt2.json` refuses `ss3`, `burial` and `fitness`: a multi-residue BPE vocabulary has no residue-to-token map and must not be approximated by a heuristic. Combined with ProtGPT2 being the **only** public protein LM with genuine subword tokenisation (3.000 chars/token against RITA 0.984 and InstructProtein 0.984), the protein x subword cell is **permanently n = 1** for residue-level methods. This is a structural limit of the field, not of this panel.

**New — de-leaking destroys the causal denominator.** Unconditioned ZymCTRL has *negative* mean-ablation ΔCE (−0.081 to −0.142) and `denominator_valid = false` for burial, ec_class and ss3. Leak control and causal power are mutually exclusive on that arm.

**New — the frozen-attention approximation is NOT a protein-specific liability.** TG-06 measures it at **77% of context information in both** gpt2-large and ZymCTRL. Reporting this null is more informative than building an attribution graph.

**New — protein dictionaries are data-limited where text ones are saturated.** gpt2-large 0.958 → 0.962 across 16x data; ZymCTRL 0.610 → 0.716 → **0.843** and still climbing at ~+0.12 per 4x. That is convergence of the **instrument**, not of the model — a distinction this programme has been blurring.

**New — the one surviving pathway difference.** Non-local propagation under activation patching: 9% (gpt2-large) against 34% (ProtGPT2) and 50% (ProGen2-medium) in the 33–64 band. It has survived every methodological correction and is denominator-free, but rests on 2–16 eligible far-band cases. **Not yet a claim.**

### 5.2 A fifth cause the evidence demands

The four candidate causes named for this programme — insufficient pretraining, limited output interface, divergent pathways, method-intrinsic non-transferability — do not cover where several retractions actually came from. A fifth is required:

**Protein-specific measurement substrate.** Input rendering worth 1.42 nats/token (L11); conditioning-tag leakage worth 1.73 nats (L15); corpus file order worth up to 1.01 nats (L13); corpus repeat statistics at 32.3% against 0.402%. None is a property of protein *computation*; all are properties of measuring protein models. The text-derived literature treats input rendering as fixed and corpus order as irrelevant, because in text they nearly are.

Attribution summary across all catalogued limitations:

| cause | support |
|---|---|
| Insufficient pretraining / convergence | **weak for the model, clear for the dictionary** — the matched-corpus contrast moves MLP share 0.019 against a ~0.43 gap (~1/22) and induction 0.0069 against −0.10 (~7%) |
| Limited output semantic interface | **strong** — algebraic (rank 31), definitional (closure), and measured |
| Divergent computational pathways | **not supported** — four instruments disagree; one provisional survivor |
| Method-intrinsic non-transferability | **strong** — every instance demonstrated on the text control |
| Protein-specific measurement substrate | **strong** — source of four retractions |

### 6. Method-family coverage

Against the fourteen families enumerated for this programme:

| family | tested | result | failure stage |
|---|---|---|---|
| Logit / tuned lens | yes | transfers; bounded by L8 | semantics |
| J-Lens / J-Space | yes | null instrument (L10, and too noisy) | semantics |
| Linear / MLP probe | yes | transfers; L16 qualifies interpretation | semantics |
| Concept erasure | yes | transfers; better than probes for reliance | semantics |
| Activation patching | yes | transfers; L4, L17 | causal |
| Path patching | yes | transfers; refuted the mediation hypothesis | causal |
| SAE / transcoder / CLT | yes | transfers with L1, L3, L4 | instrument |
| Attribution graph | partial | L7 priced but not built for protein | causal |
| Pair / attention transcoder | no | motivation retracted (§6.1) | — |
| Dedicated feature crosscoder | no | untested | — |
| Natural-language autoencoder | analytic only | L9 bounds it | semantics |
| Contrastive / concept vectors | no | untested | — |
| Training-data attribution | partial | homology control done; full influence functions not attempted | — |
| Steering / clamping | historical | calibrated negative in prior R2 work | causal |
| *(circuit census — not on the original list)* | yes | **L5: `paa_specific` screen failure for copy-suppression on GPT-2-large** | substrate |

---

## PART III — Toward adapted methods

### 7. Standing rejections

Recorded so they are not re-proposed under new names. Both have been rejected multiple times on unchanged grounds.

- **Property-conditioned Jacobian subspace** (rejected 3x). `J = ∂logits/∂h` is defined against the model's output. Secondary structure, Pfam, EC and fitness are not functionals of the next-residue distribution, so their directions can only come from a trained probe — making the object the Jacobian of the probe. Concept erasure answers the underlying question better because it measures reliance rather than decodability (L16).
- **Cross-position residue-pair sparse features** (gated). Motivation retracted: the attention margin over per-position marginals fell to 0.03–0.05 after correcting structure-selection order, and to ~0.03 over the separation-only control. Requires the relational effect to be re-established at adequate power before any construction.

### 8. Where a proposal could be earned

No proposal is advanced here that is not traceable to a catalogued limitation. The best-supported openings, in order:

1. **From L5** — a causal-effect-based mechanism comparison over a fixed budget of exhaustively tested heads, requiring no valid cheap screen. Trades coverage for validity. Per-instance ΔM-gap matrices were retained, so its feasibility is answerable without a rerun.
2. **From L1/L3/L4** — a dictionary-evaluation protocol whose estimand is power-checked on the text control before any protein arm is scored, reporting loss recovered and KL rather than FVU.
3. **From L8** — a readout that does not route through the unembedding, since the rank-(V−1) aperture is a hard vocabulary-conditional algebraic constraint and small residue vocabularies make it especially restrictive.

Any method that is built is benchmarked against SAE, CLT, probes, dense low-rank and random controls, on fidelity, causal selectivity, cross-seed stability and generalisation to disjoint protein families.

---

## 9. The plan

Total **~90–140 H200 GPU-hours**, comfortably inside a 4-card pod over a few weeks, plus L20s for CPU-adjacent work.

### Phase A — close part 1. No new measurement. ~0 GPU-h.

Write the differences section as *instrumental*: one table of every measured text-vs-protein difference with its post-control status — aperture rank (survives, vocabulary-conditional), non-local propagation (survives, validation scale), induction prevalence (probe-conditional; the scale-matched 2.34x shortfall is descriptive and has no artifact-backed scale-adjusted p-value), corpus repeat statistics, and the dissolved effects. The former gpt2 / dialogpt-small 2.30x corpus contrast is retracted because DialoGPT-small is off-distribution. **Exit condition: no further part-1 measurement is authorised.**

### Phase B — build part 2 as measured evidence. ~35–45 H200 GPU-h.

This is the deliverable.

| item | pre-registered gate | cost |
|---|---|---|
| **B1** Input-contract certification, retro-applied to every quoted number | must reproduce the 1.42-nat rendering delta and the +1.01-nat file-order delta as positive controls | ≤10 GPU-h |
| ~~**B2**~~ TG-03/TG-07 re-run under corrected rendering + seeded cohort | binary: does the variance–behaviour dissociation reproduce? **DONE, EXP-R2-062 — NO. Retracted, and C4 with it.** | ~~2~~ 1.8 GPU-h, spent |
| **B3** Explanation-channel re-derivation under seeded permutation | the 7.32 / 3.61 / 0.74 bits/symbol contrast survives, or closure stands as definitional only | CPU |
| **B4** Causal effect-size distributions on induction, 4 arms | top-20 Jaccard ≥ 0.8 against the census, else the new statistic is less sensitive and is not adopted | ~12 GPU-h |
| **B5** Same on copy-suppression | does effect-size separate arms where prevalence cannot? Directly answers §1's measurability question | ~13 GPU-h |
| **B6** Non-local propagation at production scale | far-band cases ≥ 30 per arm (currently 2–16); 9% vs 34–50% survives with intervals excluding zero | ~5 GPU-h |

~~**B2 first** — it is 2 GPU-h and it decides whether an entire method line has a motivation.~~ **Run, EXP-R2-062: the method line does not have one.** B1 is partly discharged with it — the rendering positive control reproduces at 1.78 nats/token (larger than 1.42, because the seeded cohort also removes the file-order effect the original control carried), and the file-order control reproduces at +0.240 nats/token on ProGen2-medium at n = 200 against the +1.01 recorded at n = 48. `scripts/transfer_gap/tg00_input_contract.py` is that certification stage and should be run before any TG number is quoted.

### Phase C — part 3, construction. Hard-gated. ~50–90 H200 GPU-h.

| item | gate | cost |
|---|---|---|
| **C1 — exploratory calibration** | powered-estimand re-qualification at `mlp_window8@d0.50@cohort_mean`, plus the matched gpt2-large windowed CLT that P0-2b never had; calibrate on the text control, and freeze any resulting threshold before protein-arm evaluation to support a later confirmatory gate | 17–50 GPU-h |
| **C2** Aperture-functional attribution (§8.3) | after B1; within-arm descriptive only, no cross-arm coefficient | 3–5 GPU-h |
| **C3** Oracle-grounded explanation, normalised by measured channel capacity | after B3; attainability shown on gpt2-large first | 4–8 GPU-h |
| ~~**C4** Reliance-weighted dictionaries~~ | ~~only if B2 reproduces~~ — **B2 returned NO (EXP-R2-062); dropped** | ~~20–40 GPU-h~~ |
| **C5** DMS-grounded causal gates | only on features surviving C3; internal positive control mandatory, since **no text analogue of a DMS assay exists** and this gate cannot be attainability-checked on a text control | 10–20 GPU-h |

### 9.1 Abandoned, with reasons

1. **Expansion of the present `paa_specific` screen to further mechanisms** — closed by its GPT-2-large copy-suppression control (L5). A different screen or an exhaustive causal effect-size census requires a new, independently gated design.
2. **Normalised MLP/attention share as a modality instrument** — moved four times on a quantity of magnitude ~0.4. More seeds will not fix a fragile denominator. Keep the unnormalised per-scope ΔCE values.
3. **J-Lens as a fitted cross-arm coefficient** — keep the algebraic rank, drop the regression.
4. **Pair / residue-pair transcoders** — standing rejection, motivation retracted at ~0.03 AUC over control.
5. **Property-conditioned Jacobian in its probe-derived form** — standing rejection. §8.3 is the accepted narrow version and is labelled so it is not mistaken for a revival.
6. **The 2x2 training campaign** — the most expensive design available and it cannot fix the confound that matters: matching architecture, parameters, scale and budget does not match the data-generating process. Report as an irreducible limit.
7. **Searching for a second subword protein LM** — structurally impossible with current public models.
8. **Steering re-runs before a dense readout exists** — without one, a steering run is not a test.
9. **Further investment in the induction modality claim.**

### 9.2 One unresolved question worth 5 GPU-hours

- **Does non-local propagation survive at production scale?** The one finding that has survived every correction, and denominator-free — but far bands rest on 2–16 cases. ~5 GPU-h. If it holds it is the strongest substrate claim the programme has.

---

## Appendix A — Experiment index

Organised by the objective each experiment serves. "Stands" means the result is currently defensible; "retracted" means withdrawn on evidence; "superseded" means replaced by a better-controlled measurement.

**Stable aliases for reused transfer-series ids.** The append-only log reused EXP-R2-025 through EXP-R2-032. This audit therefore uses `TR-025` = EXP-R2-025 (2026-07-24), `TR-026` = EXP-R2-026 (2026-07-27), and `TR-027` through `TR-032` = EXP-R2-027 through EXP-R2-032 (2026-07-28). These aliases do not renumber or rewrite history.

### A.1 Part 1 — differences

| id | subject | status |
|---|---|---|
| TR-025 / EXP-R2-025 (2026-07-24; TG-01..07) | transfer screen: budget, order/composition, matched SAE, frozen attention | superseded in part; **TG-01 not re-run — no TG-01 number is currently quotable** (see below) |
| TR-026 / EXP-R2-026 (2026-07-27; TG-08..10) | dictionary budget sweep; P0-2b follow-up | stands (L3) |
| TR-027 / EXP-R2-027 (2026-07-28) | estimand power and pathway budget under cohort parity | superseded quantitatively by EXP-R2-060; historical 48-estimand result |
| TR-028 / EXP-R2-028 (2026-07-28) | ProtGPT2 input-rendering defect; shared-cohort design manufactured the gap | stands (L11) |
| TR-029 / EXP-R2-029 (2026-07-28) | circuit primitives: induction, DLA, activation patching | superseded by 057 |
| TR-030 / EXP-R2-030 (2026-07-28) | apparent modality gap absorbed by tokenisation | **retracted** by 033 |
| TR-031 / EXP-R2-031 (2026-07-28) | lens family; output-interface rank; unigram estimator defect | stands (L8, L12) |
| TR-032 / EXP-R2-032 (2026-07-28) | induction dissociation survives; DLA/ablation divergence explained | partly retracted |
| EXP-R2-033 | tokenisation conclusion retracted; data variation eliminated | stands |
| EXP-R2-034 | ZymCTRL prompt-leak confound | stands (L15) |
| EXP-R2-036 | design cells; ProtGPT2 is the only subword protein LM | stands |
| EXP-R2-038 | 2x2 design populated; capability gating enforced in code | stands |
| EXP-R2-039 | first identified finding — fewer induction heads, not weaker | superseded by 057 |
| EXP-R2-045 | aperture axis reads null | stands (L8) |
| EXP-R2-046 | first production H200 campaign | superseded |
| EXP-R2-048 | approximate-repeat probe; deficit survives | superseded by 057 |
| EXP-R2-050 | homology control — no head-count separation across measured protein homology strata | **re-run and qualified** (EXP-R2-064, §0.05); no genuine low-homology stratum or symmetric text-corpus control, so no matched cross-modal memorisation claim |
| EXP-R2-050 | homology control — peak prefix-matching strength | **reverses to memorisation-consistent** on all three arms once masking is off (EXP-R2-064); already unreportable via EXP-R2-049 |
| EXP-R2-053 | induction path patching — mediation refuted | stands |
| EXP-R2-054 | nine-stage campaign; text side found to be n=1 | stands (L18) |
| EXP-R2-055 | text-side diversification | stands (L19) |
| EXP-R2-056 | text-side generalisation; deficit survives, magnitude overstated | superseded by 057 |
| **EXP-R2-057** | **threshold robustness and scale separation** | **stands — terminal for part 1** |
| **EXP-R2-060** | eleven-arm instrument-transfer campaign | current estimand-power and pathway-budget results; qualified by EXP-R2-063 |
| **EXP-R2-062** | corrected TG rendering and seeded cohorts | B2 completed NO; variance–behaviour dissociation and C4 motivation retracted |
| **EXP-R2-064** | masking-free homology control | within-protein head-count result qualified; peak-strength result reverses toward memorisation |

### A.2 Part 2 — limitations

| id | subject | limitation |
|---|---|---|
| TR-026 / EXP-R2-026 (2026-07-27; TG-08) | dictionary budget sweep | L3 |
| TR-027 / EXP-R2-027 (2026-07-28) | historical 48-estimand power sweep | L1; superseded quantitatively by EXP-R2-060 |
| TR-028 / EXP-R2-028 (2026-07-28) | input rendering | L11 |
| TR-031 / EXP-R2-031 (2026-07-28) | aperture rank; estimator bias | L8, L12 |
| EXP-R2-040 | probes and concept erasure; decodability is not reliance | L16 |
| EXP-R2-045 | denominator-free is not enough | L4 |
| EXP-R2-048 | aligned-pair coverage collapse | L14 |
| EXP-R2-055 | normalisation-form silent failure | L19 |
| EXP-R2-057 | environment default narrowed the panel | L18 |
| **EXP-R2-059** | **`paa_specific` fails as a copy-suppression ranking screen on GPT-2-large** | **L5, L6, L13, L17; no general census failure or protein-arm result** |
| **EXP-R2-060** | **eleven-arm instrument-transfer campaign** | **current L1 estimand sweep; cohort sensitivity and qualified pathway/lens results** |
| **EXP-R2-061** | **transfer-code audit and DIAMOND masking defect** | **retracted the original homology control pending EXP-R2-064** |
| **EXP-R2-062** | **corrected rendering and TG re-run** | **B2 returned NO; strengthens L3/L11 and retracts C4's motivation** |
| **EXP-R2-064** | **masking-free homology re-run** | **qualifies the head-count result and reverses peak strength; cross-modal memorisation unresolved** |

### A.3 Infrastructure and provenance

EXP-R2-035 (results-wipe diagnosis), EXP-R2-037 (H200 migration), EXP-R2-041/042/043/044 (controller/worker refactor, contract reconciliation, port validation, dispatch split), EXP-R2-049 (cohort ceiling), EXP-R2-051/052 (staging; code-freeze scope), EXP-R2-058 (GPFS staging and load checks for all eleven checkpoints; full SHA-256 for ten, byte-size-only evidence for Qwen2.5-0.5B), EXP-R2-063 (`scripts/transfer/` audit: one panel contract replaces five hand-maintained arm lists; cohort-band and panel qualifications), EXP-R2-065 (documents, logs and results reorganised against this document's objective), EXP-R2-066 (live import closure, scoring/TG contracts, TG-01 non-runnability and corrected-stage status).

**Where the retired scope went.** The conserved sparse-readout atlas, EC steering, enzyme design, the npj manuscript package, the P0 protocol set, the npj literature corpus and the recoverability preregistration are at `/Data2/lzp/bio_archive/legacy/r2_retired_scope_20260729/`. Four things were kept live because this programme depends on them: `results/final_checkpoints/` (the only protein dictionaries held locally, an input to C1), `results/transfer_gap_20260724/` (cited in §0.1), the `evidence/p0_2*` receipt chain (the evidence behind L1 and the baseline C1 re-qualifies), and the April CLT training logs. Note that the 27 P0-2 dictionaries are **not** in `results/final_checkpoints/` — they are on GPFS, and their paths and SHA-256 digests are in `evidence/p0_2b_fidelity_20260727/p0_2b_fidelity_spec.executed.json`.

### A.4 Provenance hazard

**Experiment ids EXP-R2-025 through EXP-R2-032 are each used twice** in `EXPERIMENT_LOG.md` — once by the July 17–27 npj-revision series and once by the July 24–28 transfer series. Historical entries remain unchanged. Current transfer-series citations must use the `TR-025` through `TR-032` aliases defined above or include both date and original id. Ids from EXP-R2-033 onward are unique.

---

## Appendix B — Methodological hazards, as standing rules

Each earned by a failure in this programme.

1. **Never take the first N records of a biological corpus.** Sample under a seeded permutation and report a skip-offset sensitivity. (Three instances.)
2. **Check gate attainability on the text control before applying it to a protein arm.** A gate the positive control cannot pass is a specification defect. (Two instances.)
3. **Use held-out, never plug-in, estimators for information decompositions.**
4. **Feed every arm the format it was trained on**, and verify the rendering against the model's own likelihood.
5. **An intervention that moves everything needs a control for moving everything.**
6. **A reconstruction gate does not protect against wrong norm algebra** — RMSNorm-on-LayerNorm passes at 0.49% error while corrupting attribution.
7. **Verify that environment defaults have not narrowed the panel**; print resolved paths before every campaign.
8. **Distribution-body statistics require decoy/positional correction; tail statistics do not.**
9. **Never run `git clean` under ``** — it is its own git repository with `/results/` ignored. This destroyed all experiment results three times.
10. **Search the literature by mechanism name, not by domain**, before designing a track. (Cost: an entire induction track designed without finding prior work that had already established induction heads in protein LMs.)
11. **Compute residual-stream spectra on interior, alphabet-bearing positions.** A participation ratio or top-PC share taken over all positions measures the attention sink and the format separators, not the representation. GPT-2-large reads PC1 0.809 over all positions and **0.034** with its first token dropped (participation ratio 1.53 → 253.4); ProtGPT2 reads 0.971 and **0.439** with its FASTA newlines dropped (1.06 → 4.86). Two headline numbers that were compared to each other turned out to be artefacts of different origin. (EXP-R2-062.)
12. **A single declaration, imported, never reimplemented.** The rendering defect of §0.1 survived a withdrawal that fixed `src/transfer/arms.py` because `scripts/transfer_gap/tg_common.py` carried a second copy. Any module that decides what string a model is fed must import that decision from the panel declaration. (EXP-R2-028, EXP-R2-062.) The same holds one step earlier, for *which records exist* and *which arms a stage measures*: EXP-R2-063 found the latter maintained by hand in five places, two of which disagreed with `PANEL` — one stage's panel silently narrowed by an arm, another silently widened by one. Use `scripts/transfer/panel_contract.py::arm_can_run`.
13. **Declare a stage's cohort band against the band the arms were qualified on.** Four stages of one campaign draw protein cohorts on three different bands (64–246, 64–120, 600–2000), and the cohort `cohort_power` qualifies is not the cohort `lens_family` scores. A per-stage band is a legitimate compute choice; an undeclared one lets a PASS verdict be read as covering a population it was never measured on. (EXP-R2-063.)
14. **A scheduler must obey what a module can deliver, not what the panel intends.** `ArmSpec.capabilities` is an intent and a module's architecture declaration is a deliverable; they are allowed to disagree, and exactly one predicate should consult both. Handing a stage an arm it cannot serve does not fail cheaply — it fails after the checkpoint is on the GPU. (EXP-R2-060, EXP-R2-063.)
