# Interpretability transfer audit: text to protein generative models

**Status:** active; canonical analysis document **Supersedes:** repository-root `check.md`, which is frozen and no longer maintained **Updated:** 2026-07-31

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

> **Extended by EXP-R2-068.** A unit floor now applies to that head resampling, at the eight units `homology.py` has enforced since EXP-R2-061. The headline above is unaffected, though not for the reason first written: its two-sample bootstrap resamples **38** heads (14 against 24), 14 being only the smaller side, and it carries `excludes_zero: false` — it was never a separation verdict, so there was nothing for the floor to withdraw. The two nodes that *are* separation verdicts at that count (`exact/direct_effect`, `exact/mediated_effect`) survive the floor. But **eight further `excludes_zero: true` separation verdicts in the same artefact rest on three to six heads** and are withdrawn: ProGen2-medium's exact direct effect, mediated effect and mediated fraction (n = 6, 6, 6), its approximate mediated effect and mediated fraction (n = 4, 3), and the matched pair's *approximate* direct effect, mediated effect and mediated fraction (n = 6, 6, 5). Measured coverage of a nominal 95% interval at these counts is 0.74–0.86, not 0.95 — a simulation recorded in `statistics.py`'s own docstring and not, as yet, in any artefact or test. On re-run those rows publish a point difference and no separation verdict. `results/transfer_20260728/path_patching/panel_summary.json` is retained unmodified as the record.

## 0.15 TG-01 is not quotable (EXP-R2-066)

> **CLOSED 2026-07-30 (EXP-R2-068).** Two corrections. The stated cause was stale — `tg01_information_budget.py` registers `--seed` once and `--help` exits 0, so it was *runnable, not dead*. And it has now been **run**, on all four TG arms, under a seeded permutation of the whole corpus consumed in seeded record order, with the held-out unigram estimator. Artefacts: `results/transfer_gap_20260729_corrected/tg01/`.
>
> | arm | unigram (nats) | clean NLL | gain (bits/symbol) | top-1 | information range (nats) | long-range fraction beyond 8 |
> |---|---:|---:|---:|---:|---:|---:|
> | gpt2-large | 7.587 | 2.632 | 1.615 | 0.472 | 4.884 | 0.254 |
> | protgpt2 | 8.821 | 4.846 | 2.012 | 0.266 | 3.916 | 0.630 |
> | progen2-medium | 2.894 | 1.459 | 2.075 | 0.545 | 1.427 | 0.945 |
> | zymctrl | 2.974 | 0.891 | 3.076 | 0.727 | 0.202 | refused |
>
> **The retracted figures moved a long way, and in the direction the rendering defect predicts.** ProtGPT2's clean NLL falls **7.296 → 4.846** (−2.45 nats), its information gain rises **2.636 → 5.735 bits**, and its top-1 rises **0.102 → 0.266**. None of the 2026-07-24 TG-01 numbers should ever have been read as a property of the model. ZymCTRL's long-range shares are **refused, not reported**: its information range is 0.202 nats, below the 0.5-nat denominator floor, so the shares are not defined. That refusal is the guard working, and it means ZymCTRL contributes no long-range statistic to this stage.

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

The far-band propagation result, recorded here until 2026-07-31 as the best-founded part-1 result, is **withdrawn as a model claim**: it compared arms at a distance declared in tokens across arms differing 4.4x in symbols per token, and two *protein* arms swap order on that choice alone (§5.1, EXP-R2-073). The strongest surviving part-1 measurement is therefore §5.05(d)'s pathway budget, whose statistic is a within-arm dimensionless ratio and is not exposed to the same hazard — with the two caveats that already travel with it.

**On part 2 (limitations).** This is where the programme's real yield is, and much of it was demonstrated *on the text control*, which makes it a property of the method rather than of protein models. See §4.

**On part 3 (adapted methods).** Not yet earned. Two candidate directions carry standing rejections (§6.1) and no proposal is currently traceable to a measured limitation with adequate evidence.

**The strongest directly measured part-2 result available is now EXP-R2-071/072/077/079: the prefix-matching induction census is a causal proxy on text decoders and is not one on protein decoders.** With every head patched rather than only the census's own selections, the Spearman correlation between census score and causal-effect magnitude separates the modalities completely — the text minimum (+0.371, gpt2-xl) exceeds every protein maximum (+0.282, ZymCTRL) in all **48** arm × condition cells on **seven** text and five protein arms, the matched pair included. **Two alternative explanations are ruled out by their own controls.** *Distribution mismatch:* `dialogpt-small`, a text decoder that §5.05(a) shows is off-distribution on the evaluation corpus at −4.08 nats, does **not** collapse (+0.518 to +0.640). *Architecture:* two rotary/GQA text lineages from other labs read +0.479 to +0.575, while ProtGPT2 — which shares gpt2-large's exact architecture — reads −0.226 to −0.006, so the separation tracks modality rather than model family (EXP-R2-079). The ordering also survives an independent corpus draw on the four arms that define the gap, with the matched pair separating *further* (EXP-R2-077). The failure mode is recall in the bulk of the grid, not at its top, and the consequence for this programme is direct — an unknown part of §4's head-count gap belongs to the instrument rather than to the models. This does **not** show that protein decoders have induction heads after all, and the *margin* is not yet quoted against a spread: the largest single-cell draw-to-draw movement (0.107) approaches the closest modality gap (0.121), which EXP-R2-080 is measuring at K = 4.

The earlier holder of that position remains valid and narrower: `paa_specific` does not provide a valid cheap ranking screen for copy-suppression on GPT-2-large, despite the causal mechanism itself being detectable. Neither result establishes that head-prevalence censuses generally have a narrow domain. Measurability is now a *demonstrated* contributor to an apparent transfer gap on one mechanism, rather than a candidate one.

---

## PART I — Differences between text and protein generative models

### 2. The panel

Twelve autoregressive decoders under one code path, each fed in the format it was trained on. Eleven were staged and load-checked on GPFS in EXP-R2-058, ten of them with full source-to-GPFS SHA-256 verification while Qwen2.5-0.5B was recorded as byte-size checked only; ProGen2-small was load-checked on the pod and admitted in EXP-R2-068. `scripts/transfer/panel_contract.py` is the declaration — this table is a reading of it, not a second copy.

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
| progen2-small | protein | 12 x 1024 | residue 32 | UniRef90+BFD30; protein-side scale rung |

**Design properties that make specific contrasts identifiable:**

- `gpt2-large` / `protgpt2` — identical architecture, depth, width and vocabulary size, and identical parameter count — **774,030,080**, verified from both checkpoints (2026-08-01); the previously recorded 773,891,840 was wrong by 138,240, though the *identity* it asserts is exact. The matched modality pair.
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

> ### D1.b answered — the battery re-run on the seeded 817-of-817 census (EXP-R2-073)
>
> Those figures came from a results tree that no longer exists; `12_induction_robustness.py` was hard-wired to it and raised `FileNotFoundError` on every path, so the instrument D1.b needed had to be repaired before the question could be asked. It now takes its source as an argument. Re-run on `results/transfer_20260730/panel12/`, synthetic probe, cut 0.10, on the same eleven arms:
>
> | statistic | recorded above | re-run on the seeded census |
> |---|---|---|
> | pooled AUC | 0.595 | **0.624** |
> | pairs above 0.5 | 8 / 12 | **20 / 28** (min 0.400, median 0.581, max 0.729) |
> | stochastic dominance | 0 / 12 | **2 / 28** |
> | quantile dominance | fails at q50, q99.5, q100 | fails at the same three — **and separates at q75, q90, q95, q97.5, q99** |
> | survival grid | 62% | 61.9% |
> | model-level exact test | p = 0.0286 (fraction) | **at the floor, 1/330 = 0.00303, complete separation** on fraction, mean and q99; the median statistic does not separate |
> | ladder slope / decade | −0.272 [−0.455, −0.088] | **−0.268 [−0.515, −0.020]** |
>
> **The conclusion is unchanged in direction and less absolute in wording.** "0 of 12 show stochastic dominance" cannot be restated as "none": two of twenty-eight pairs now do. And the middle of the distribution is not as featureless as the old summary implied — the ordering separates at every quantile from q75 to q99 and fails only at the median and in the extreme upper tail, where counts are single heads. **It is still a tail statement rather than a distributional one**, and the reason is now visible: the separation lives between the 75th and 99th percentiles of the per-head score, not across the distribution.
>
> **One published figure changes materially.** "Only **25.4%** of the modality indicator survives projection" was computed by dividing by an *uncentred* sum of squares, whose ceiling is n_protein/n rather than 1 — the statistic could not reach 1 even if scale and lineage explained nothing about modality. On the centred denominator, which is one minus the R² of the modality indicator on the nuisance block, the same fit gives **36.3%**. Increments are unchanged: modality +0.270, lineage +0.053, scale +0.003.
>
> **And the 2.34x scale-adjusted shortfall is withdrawn as an interval verdict.** `lineage_scale_ladder.predict()` extrapolated the GPT-2 line onto every other arm and published `observed_below_prediction_interval` for all of them, on an interval that prices residual scatter among four GPT-2 rungs and prices nothing about a different lineage, tokeniser or corpus. `in_fitted_lineage` is now carried through and the verdict is withheld off-lineage. The point ratio stands as descriptive; the interval statement does not. Independently, the arm that anchored the old comparison was `dialogpt-small`, which §5.05(a) retracts as off-distribution.

**Probe dependence — fails on the best probe.**

| probe | ordering |
|---|---|
| synthetic | holds at every threshold |
| natural exact | breaks at 0.20, 0.30 |
| **natural approximate** | **inverts at 0.10** (dialogpt-small 0.0000, ProtGPT2 0.0097), p = 0.114 |

`circuits.py` records the natural probe as the one to trust for wrapped arms and `scaling.py` declares it primary; the headline table used the synthetic probe. The natural approximate criterion is the one Pomerants et al. showed subsumes exact detection.

> **The inversion at 0.10 was an artefact of the repeat cohort's draw, and does not reproduce (EXP-R2-068).** The census's *repeat* cohorts were never seeded: under the approximate criterion the stage took **32 of 817** matching proteins and **32 of 968** matching documents in corpus file order — a four per cent head-of-file prefix on the cohort that carries this entire result. Re-run on a seeded draw at the whole matching population (**817 of 817** protein, so on the protein side it is a census and order cannot matter; 817 of 1022 text), on all eleven arms:
>
> | probe | t = 0.05 | t = 0.10 | t = 0.20 | t = 0.30 |
> |---|---|---|---|---|
> | synthetic | holds | holds | holds | holds |
> | natural exact | holds | holds | holds | **inverts** |
> | natural approximate | holds | **holds** | **inverts** | **inverts** |
>
> At the headline threshold the ordering now holds on **all three probes**. `dialogpt-small`'s natural-approximate fraction, the 0.0000 that produced the recorded inversion, reads **0.0278** on the seeded cohort; the worst text arm (llama-3.2-3b, 0.0268) sits above the best protein arm (ProtGPT2, 0.0181). Matched pair at 0.10: **5.46x** synthetic (against the recorded 5.38x), 4.07x natural exact, 2.92x natural approximate.
>
> **The probe-dependence problem is reduced, not removed.** It moves to the permissive-tail thresholds: natural approximate inverts at 0.20 and 0.30, natural exact at 0.30. Both inversions are driven by the worst text arm reaching *exactly zero* while a protein arm retains one or two heads out of 720–1728 — a small-count tail effect, not a distributional statement, and the same reason the distributional separation fails. Nothing here converts the result into a modality claim: the structural n = 1 limit of §2 and the corpus-repeat confound of the closing paragraph are untouched. Artefacts: `results/transfer_20260730/induction_seeded/`.

> ### The twelfth arm breaks the synthetic probe, and one head decides the headline (EXP-R2-073)
>
> The table above is the **eleven**-arm panel. ProGen2-small was admitted after it was written, and it is the best protein arm on every probe. Re-derived on all twelve arms from `results/transfer_20260730/panel12/`:
>
> | probe | t = 0.05 | t = 0.10 | t = 0.20 | t = 0.30 |
> |---|---|---|---|---|
> | synthetic | holds | holds | **inverts** | **inverts** |
> | natural exact | holds | holds | **inverts** | inverts |
> | natural approximate | holds | holds | inverts | inverts |
>
> **"Synthetic holds at every threshold" is panel-dependent and no longer true.** At 0.20 the worst text arm is llama-3.2-3b at 0.0238 against ProGen2-small at 0.0260; at 0.30, 0.0179 against 0.0208. Neither inversion is the zero-arm artefact the paragraph above invokes — llama's values are non-zero, and the inverting protein arm is ProGen2-small in every cell. The recorded explanation covers the eleven-arm inversions and does not cover these.
>
> **At the headline threshold the ordering still holds on all three probes, by less than one head.** On the natural-approximate probe the margin is llama-3.2-3b **0.0268** against ProGen2-small **0.0260** — 0.0008, where one head of ProGen2-small's 192 is **0.0052**. Six heads instead of five inverts it. ProGen2-small contributes **five heads** to every probe, and the same five heads set the protein-side scale slope. The headline ordering on the probe the literature says to trust is therefore decided by a single head on a single arm.
>
> This does not overturn the result — the matched pair remains 5.46x and the protein arms sit at 0.007–0.026 against text 0.027–0.216 — but it fixes its precision. **The defensible statement is the matched-pair ratio and the panel-level pattern, not the worst-text-above-best-protein ordering**, which no longer survives its own threshold sweep and turns on one head at the cut where it does.

> **The text-side scale slope does not transport to protein, and the scale adjustment below over-corrects (EXP-R2-068).** Admitting ProGen2-small gave the protein side its first within-lineage scale contrast — 151M against ProGen2-medium's 765M, architecture, residue tokeniser and UniRef90+BFD30 mixture all held. Measured on the seeded census at threshold 0.10, protein cohorts being the whole matching population:
>
> | probe | text slope / decade (GPT-2 ladder, 4 rungs) | protein slope / decade (ProGen2, 2 rungs) | ratio |
> |---|---:|---:|---:|
> | synthetic | −0.0735 | **−0.0172** | 4.3x shallower |
> | natural exact | −0.0572 | **−0.0172** | 3.3x shallower |
> | natural approximate | −0.0446 | **−0.0238** | 1.9x shallower |
>
> ProGen2-small reads 0.0260 on all three probes (5 heads of 192) against ProGen2-medium's 0.0139 / 0.0139 / 0.0093 — so the **direction transports**: a larger protein decoder also puts a smaller fraction of heads in the upper tail. The **magnitude does not**. The 2.34x scale-matched restatement below was computed by carrying the text ladder's slope onto the protein side, which nothing had tested; a slope 3–4x shallower removes correspondingly less of the raw 5.45x matched-pair gap. **Scale therefore explains *less* of the shortfall than recorded, not more, and the 2.34x figure should be read as a lower bound on the adjusted shortfall rather than as its value.**
>
> Two rungs give a slope and no curvature, and no interval; ProGen2-small and ProGen2-medium differ in depth and width together, exactly as the text ladder's rungs do. At matched scale the protein corpus contrast is small: ProGen2-base 0.0116 against ProGen2-medium 0.0139 on the synthetic probe, both 765M. Artefacts: `results/transfer_20260730/panel12/`.
>
> **The shallower-protein-slope pattern replicates on an unrelated statistic, with a fixed-architecture data control (EXP-R2-068).** `estimand_power` asks a different question of a different measurement — what fraction of the 76 ablation estimands is *powered*, rather than what fraction of heads is in the prefix-matching tail — and the same shape appears:
>
> | lineage | rungs | powered fraction | slope |
> |---|---|---|---|
> | GPT-2 | 124M / 355M / 774M / 1.56B | 88.2% → 73.7% → 68.4% → 63.2% | **−22.5 points/decade** |
> | ProGen2 | 151M / 765M | 100.0% → 94.7% | **−7.5 points/decade** |
>
> Same sign, **3.0x shallower**, landing inside the 1.9–4.3x band the census gives. The text ladder is monotone across four rungs over a 12x parameter range, so its slope is not two-point noise. **And the protein rungs now carry a control the census version lacked:** ProGen2-base and ProGen2-medium are architecturally identical here — both 1536-wide, 27-layer, 32-symbol — so they differ only in pretraining mixture, and they read 92.1% against 94.7%. A corpus difference at fixed architecture moves this statistic by 2.6 points where a 5x size difference moves it 5.3, which is what licenses reading the lineage slope as a size effect rather than a data effect.
>
> *What this does not show.* Powered fraction is a property of the estimand battery's sensitivity on each arm, not of the arm's mechanism, and a ceiling is in play — ProGen2-small is at 100.0%, so its rung cannot move up and the protein slope is bounded below in magnitude by that alone. The modality ranges themselves **overlap heavily** (text 0.0–96.1%, protein 65.8–100.0%) and separate nothing. `dialogpt-small` reads 0 powered estimands and 0 context-valid estimands, which is §5.05(a)'s off-distribution arm again and is why the text range reaches zero.

**Scale — real, and does not explain the descriptive shortfall.** The within-lineage ladder falls monotonically: 0.1597, 0.1380, 0.0972, 0.0833; slope **−0.272/decade [−0.455, −0.088]** with head *count* rising 23 → 100. This explains llama-3.2-3b's low value. Restated against the ladder's scale-matched prediction, the shortfall is **2.34x descriptively**; the artifact does not report a scale-adjusted inferential test. Its reported **p = 0.0286** belongs instead to the unadjusted, one-sided exact permutation test over four text and three protein models for `fraction_above_0.10`, where complete separation reaches the minimum attainable p-value. Variance decomposition: scale adds **+0.003** given modality and lineage; modality +0.220; lineage +0.061 — but only **25.4%** of the modality indicator survives projection.

**Corpus — roughly half the modality contrast in log terms.**

| contrast | ratio |
|---|---|
| ~~gpt2 / dialogpt-small~~ | ~~2.30x [2.30, 2.40]~~ **RETRACTED — see §5.05(a); dialogpt-small is off-distribution at −4.08 nats context information and cannot anchor a contrast** |
| progen2-medium / progen2-base | 2.00x [1.20, 2.00] |
| gpt2-medium / qwen2.5-0.5b | 1.93x [1.82, 2.04] |
| gpt2-xl / llama-3.2-3b | 2.15x [1.92, 2.29] |
| **modality, matched pair** | **5.38x [5.23, 5.54]** |

**The identification limit, which no experiment can remove.** Induction heads are useful only on corpora containing repeated subsequences. Approximate repeats occur in **32.3% of text documents against 0.402% of protein entries** — an eightyfold difference.

> *The eightyfold figure is not unit-free, and the direction of the asymmetry flips with the unit (EXP-R2-073).* The two criteria are declared in **content symbols**, which is correct and is what keeps them free of the token confound that withdrew the far-band result — but they are not the same criterion. Text requires a repeat unit of **40 characters** with 15 distinct symbols and scores similarity by identity; protein requires **16 residues** with 8 distinct symbols and scores by BLOSUM62 non-adverse substitution. In symbols the text criterion is 2.5x the stricter; converted through each arm's measured symbols per token it is 40/4.4 ≈ **9 tokens** against 16/1.0 = **16 tokens**, so in tokens the *protein* criterion is the stricter one. Both choices are domain-motivated — a 40-character phrase and a 16-residue motif are the natural repeat units of their corpora — and the qualitative conclusion is untouched, because no plausible re-parameterisation closes two orders of magnitude. What should not be quoted as a unit-free fact is the multiplier. A decoder allocating fewer heads to prefix-matching on a repeat-poor corpus is allocating capacity efficiently against its data. Breaking the confound would require a protein decoder trained on a repeat-rich corpus, which cannot exist: a synthetic protein corpus with 32% repeat prevalence is not a protein corpus. **Even the terminal 2x2 training design cannot fix this**, as it matches architecture, parameters, scale and budget but explicitly not the data-generating process.

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
| L6 | Positional baseline dominates the *body* of per-head attention statistics; decoy correction drops head median sixfold (0.0083 → 0.0013) while leaving the tail count intact (70 → 69, Jaccard 1.000). *The correction applied here is not identical to the PAA one and cannot be made identical — see §5.1* | attention-pattern analysis, circuit census | EXP-R2-059 | **method** |
| L7 | Attribution graphs hold attention patterns fixed and do not explain how they are computed; the price is measurable | attribution graph | TG-06 frozen attention | **method** |
| L8 | Output aperture is rank-(V−1); lens-family readouts are bounded by vocabulary size, which differs ~4700-fold across the panel (32 to 151936) | logit lens, tuned lens, DLA, J-Lens | EXP-R2-031, EXP-R2-045 | **transfer** — algebraic and vocabulary-conditional, not modality-specific |
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
| L20 | **No remote predicate the campaign controller relied on could return false.** The access layer returns 0 whatever the remote command exits with — measured, `h200_pod_exec.sh -- bash -c "exit 7"` and `h200_pod_bash.sh "exit 5"` both return **0**. Three consequences, each verified on the cluster: a worker that refused a campaign at preflight and scheduled no GPU was reported as `campaign complete`, exit 0; the remote half of the code-freeze guarantee was never actually checked; and the append-only invocation manifest was **never pushed at all** — every run directory on GPFS holds `INVOCATIONS=0` while the controller logged "already present and verified". A campaign's verdict, its frozen code and its provenance record all travelled on a channel that cannot say no | any remote-execution pipeline | EXP-R2-068 | **method** — the L18 shape, one layer above the ledger built to prevent it |
| L21 | A generated contract can verify on the host that generated it and disagree on the host that runs it. The arm-to-environment-variable map was *inferred* by comparing resolved paths; the pod sets `TRANSFER_TEXT_MODEL_BASE_DIR="${TRANSFER_MODEL_BASE_DIR}"` because all checkpoints share one GPFS directory, so the comparison aliased and six of seven text arms classified as protein-root arms **on the pod only** | any host-portable declaration derived from resolved paths | EXP-R2-068 | **method** |
| L22 | **A head-prevalence census's selector ranks causal importance on text decoders and does not on protein decoders.** With every head patched, the census-to-causal rank correlation separates completely: **text minimum +0.371 above protein maximum +0.282 in all 48 arm × condition cells over seven text and five protein arms**, matched pair included, pooling every draw measured. The failure is *recall in the bulk* — over the top-32 census heads ProtGPT2 reaches +0.808, above every text arm — and it is **not** distribution mismatch (`dialogpt-small`, a text decoder at −4.08 nats of context information, reads +0.518 to +0.640) and **not** architecture: two rotary/GQA text lineages read +0.479 to +0.575 while ProtGPT2, which shares gpt2-large's exact architecture, reads −0.226 to −0.006 (EXP-R2-079). The ordering survives an independent corpus draw on the four gap-defining arms, with the matched pair moving *apart* (EXP-R2-077), and survives correction for differential measurement error in all 24 cells where reliability is measured, worst-case gap +0.095 (EXP-R2-081) — so it is **not** an artefact of protein effects being estimated less precisely. An unknown part of §4's head-count gap therefore belongs to the instrument | head-prevalence census, circuit census, any selector-then-count design | EXP-R2-071, EXP-R2-072, EXP-R2-077, EXP-R2-079, EXP-R2-081 | **transfer** — demonstrated on the matched pair, absent on the text control including its off-distribution and cross-lab arms. *Margin not yet quoted against a spread:* the largest single-cell draw-to-draw movement (0.107) approaches the closest modality gap (0.121); K = 4 in flight (EXP-R2-080) |
| L23 | **A cross-arm estimand whose unit is tokens is not a cross-arm estimand when the arms differ in symbols per token.** The far-band propagation band was declared in tokens across arms at 4.4 / 2.8 / 1.0 symbols per token; the ordering between the two *protein* arms — same content alphabet, different tokeniser — reverses between a token band and a residue band. The artefact recorded `symbols_per_token` and nothing consumed it | activation patching, any windowed or distance-based estimand | EXP-R2-073 | **transfer** — the confound is created by tokenisation, which is where the modalities differ most |

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

> **Re-measured on corpus-wide seeded cohorts across twelve arms, and it reproduces to two decimal places (EXP-R2-068).** This stage had never been re-run since the cohort draw was fixed, so the separation above rested on file-order cohorts. On a seeded draw over the whole corpus:
>
> | text arm | file order | seeded | | protein arm | file order | seeded |
> |---|---:|---:|---|---|---:|---:|
> | qwen2.5-0.5b | 2.10 | **2.070** | | protgpt2 | 1.12 | **1.129** |
> | gpt2-xl | 2.01 | **2.005** | | progen2-small | — | **1.057** |
> | gpt2-medium | 1.88 | **1.836** | | progen2-medium | 0.93 | **0.930** |
> | gpt2 | 1.77 | **1.754** | | progen2-base | 0.92 | **0.922** |
> | llama-3.2-3b | 1.69 | **1.675** | | zymctrl | 0.55 | **0.530** |
> | gpt2-large | 1.50 | **1.491** | | | | |
>
> **The ranges still do not overlap**: text 1.491–2.070 against protein 0.530–1.129, now on six text and five protein arms, with the newly admitted ProGen2-small falling inside the protein range at 1.057. `dialogpt-small` is excluded, not omitted: it returns 13.606 with `measurable_every_seed = false`, which is §5.05(a)'s off-distribution arm behaving exactly as that retraction describes.
>
> **This statistic is insensitive to the cohort draw, and that is worth contrasting with the far-band propagation fraction, which is not.** Every arm reproduces within 0.05 of its file-order value here, while ProtGPT2's far-band fraction moved by 0.051 — a quarter of its own value — between two windows of the same permutation. Cohort sensitivity is a property of the statistic, not of the modality alone, and neither can be assumed from the other. The two caveats that already travel with this row are untouched: it is a ratio to context information rather than a partition, and the tokenisation adjustment has not been applied.

Matched pair 1.50 against 1.12. **It survives replacing GPT-2-large with a Qwen2 and a Llama decoder** — different labs, rotary/RMSNorm/gated-FFN, vocabularies 2.5–3x larger — so it is not a GPT-2 idiosyncrasy, which is what killed the QK/OV finding. **Two caveats that must travel with it:** these are ratios to context information rather than a partition (values > 1 mean ablation costs more than all extracted context information), and **the tokenisation adjustment that collapsed the earlier MLP-share coefficient has not been applied.** Not yet a claim.

**(e) The tuned lens has no transfer failure.** *Qualification (EXP-R2-063): the lens stage draws its protein cohort on residues **64–120**, while `cohort_power` qualifies arms on **64–246** and the pathway and estimand stages measure on 64–246. So the arms this row scores were qualified on a different population, and (b) above prices protein cohort-block sensitivity at 0.16–0.60 nats. The band is a compute choice — the Jacobian sweep is quadratic in length — and it is now declared in the stage's artefact beside the qualifying band rather than being recoverable only by comparing argparse defaults. Nothing here is retracted; the cross-band comparison is weaker than it reads.* It improves on the logit lens at every non-identity layer on every scored arm, protein included — the only instrument in the campaign with a clean transfer. The Jacobian finite-difference guard passed on all nine scored arms, max relative error 4.7e-3 (ProtGPT2) against a 2e-2 tolerance, confirming that the earlier 1.008 failure was the forced `--dtype bfloat16` rather than an instrument defect. ProtGPT2 is the outlier on top-1 agreement (0.013–0.014, 5–10x below every other arm), consistent with its multi-residue-BPE aperture.

Two rotary arms were **not** scored and the skip was logged with its reason: `lens_head` requires `nn.LayerNorm` at `transformer.ln_f`, which an RMSNorm decoder lacks.

*(The EXP-R2-063 cohort-band qualification appeared twice in this section and read as two findings; the duplicate was removed on 2026-07-30, EXP-R2-068. It is stated once, in (e) above.)*

**Two failures were caught by the stage scripts' own guards** rather than by producing a wrong number, and both are instructive: a Markov "held-out" block offset by *record count*, which is not disjointness because Swiss-Prot carries the same sequence under several accessions; and a control-anchored aggregation receiving all seven text arms. Both fixed.

### 5.1 Quantitative anchors for the catalogue

The figures that carry the most weight, with their provenance, so the catalogue can be cited without re-deriving them.

**L1 — the original single-submodule gate is depth-dependent and unattainable on its matched text control.** In the latest EXP-R2-060 sweep, `mlp_single@d0.50@cohort_mean` costs gpt2-large **0.0193 nats/token** (bootstrap q025 0.0171) against a 0.05-nat guard, while ProtGPT2 (0.1115), ProGen2-base (0.0656) and ProGen2-medium (0.0622) clear it and ZymCTRL (0.0152) does not; **52 of 76** alternatives are attainable on gpt2-large and 50 are powered panel-wide. This makes the matched-panel or cross-domain interpretation of P0-2b inconclusive: P0-2b had no matched text dictionary, and the original ProtGPT2 and ZymCTRL recovered ratios were denominator-invalid. It does **not** erase the valid within-arm negative for ProGen2-medium under P0-2b's original estimand and frozen 0.80 gate: all nine ProGen2-medium runs had valid denominators, and the best bootstrap upper bounds were 0.411 for loss recovered and 0.282 for KL recovered.

**L5 — no valid `paa_specific` screen for copy-suppression on GPT-2-large.** `paa_specific` rank against measured ΔM-gap: Spearman **−0.062 (p = 0.71)** over 40 heads; 5 of 56 heads clear the control band against induction's 70 of 720. This does not test another proxy or an exhaustive causal effect-size census.

> **L5 is on the *signed* effect, and D2.c's statistic is the magnitude — on which the same heads and the same artefacts read +0.53 to +0.65 (EXP-R2-078, CPU, no re-run).** The distinction is not a technicality and it decides whether D2.c has a positive control at all. L5 asks whether `paa_specific` ranks heads by *suppression strength*, a signed question: knocking out a suppressive head should raise the target's logit relative to the runner-up, so its ΔM-gap should be positive. L22's statistic is a different one — whether the selector orders *causal importance*, which is `|effect|` — and that is the statistic D2.c would have to replicate. Recomputed on both retained trees:
>
> | tree | heads | signed ρ | \|effect\| ρ |
> |---|---:|---:|---:|
> | `paa_gate` | 24 (16 screened + 8 control) | +0.086 (p = 0.69) | **+0.608 (p = 0.0016)** |
> | `paa_gate` | 16 screened only | +0.265 (p = 0.32) | **+0.621 (p = 0.010)** |
> | `paa_gate_extended` | 32 (24 + 8) | −0.162 (p = 0.38) | **+0.533 (p = 0.0017)** |
> | `paa_gate_extended` | 24 screened only | −0.305 (p = 0.15) | **+0.648 (p = 0.0006)** |
>
> On the signed scale the correlation is indistinguishable from zero in all four cells; on the magnitude scale it is +0.53 to +0.65 and significant in all four — **comparable to gpt2-large's induction value of +0.428 to +0.507**, which is the text control D2.c would be judged against. **L5 stands exactly as written**: no valid *signed* screen for copy suppression exists on this arm, and that is what it claims. What changes is the inference the plan had drawn from it — that D2.c's positive control was known dead. It is not known dead, and on the evidence available it is probably attainable.
>
> **Two limits on that reading, both material — and a third found later.** These 24–32 heads were *selected by `paa_specific` itself* plus a control block, so the correlation is over a restricted range and the exhaustive-grid value could move in either direction — **it moves down, to +0.5309 exhaustive at width 512 (EXP-R2-087), and the restricted reading is additionally unstable at ±0.12 across cohorts at fixed width. The third limit: these artefacts predate `paa_specific_matched` entirely, so every number in this table is an *unmatched*-score figure, which is not the score a causal comparison may use** — the same range-restriction that made EXP-R2-071's top-32 reading disagree with its all-grid one. And a magnitude ranking conflates suppressive heads with promoting ones, so a high magnitude ρ is evidence that the selector orders causal importance, not that it finds copy suppression. Both belong in D2.c's pre-registration rather than in its discussion.

**L8 — the aperture is algebraically forced.** Numerical rank of `∂logits/∂h_l` is exactly **31** for ProGen2-medium and **457** for ZymCTRL, at every layer and probe, against 1150–1278 for the 50k-vocab arms. Validated against central finite differences at 2.2e-4–2.5e-3 relative error. **Vocabulary-conditional, not modality-conditional** — ProtGPT2's interface is near-full-rank. The honest label is *protein-typical, not protein-necessary*.

**L9 — explanation closure, measured; re-derived under seeded permutation (EXP-R2-068, plan item B3).** Bits/symbol within one 300-symbol window, every channel now drawn under a seeded permutation of its **whole** corpus and every unit list visited in seeded order, with Student-t intervals over per-unit values (not clustered):

| channel | file-order draw | seeded draw, 95% interval | second draw seed |
|---|---:|---|---:|
| text token identity | 7.32 | **7.326** [7.316, 7.336] | 7.331 |
| residue identity | 4.11 | **4.094** [4.089, 4.098] | 4.097 |
| structural-oracle attributes | 3.61 | **3.792** [3.746, 3.837] | 3.783 |
| Pfam domain label | 0.74 | **0.860** [0.837, 0.883] | 0.889 |

Drawn from all **574,627** eligible Swiss-Prot entries and all **23,586** AlphaFold models rather than from a prefix; Pfam residue coverage 0.601 over 18,251 of 20,000 sampled proteins. The fourth column is an independent draw at seed 20260801 (`results/transfer_20260730/explanation_channel_seed2/`), reported because evidence-discipline rule 4 does not admit a single-draw point estimate: every channel reproduces, and both Pfam figures sit far above the file-order 0.74. **The contrast survives and the closure argument stands.** The two channels that had been prefix draws moved **in the predicted direction and by a material amount** — Pfam **+0.12 bits (+16%)**, structural **+0.18 bits (+5%)** — because a family-grouped prefix is more uniform in its labels than the corpus, so it *understates* a label channel's entropy. The text control moved **+0.0036 bits**, confirming that shard order is not family order and that the effect is protein-specific (extends L13). The text-to-Pfam ratio is **8.5x**, against 9.9x under the biased draws: the closure gap is real and slightly smaller than recorded. Artefact: `results/transfer_20260730/explanation_channel/explanation_channel.json`.

**L6 — the two decoy corrections are not the same one, and cannot be made the same (EXP-R2-078).** `decoy_corrected_prefix_matching` claimed in its docstring to apply "the same decoy subtraction PAA uses" and carried an inline comment asserting the two draws *had* to select from one key population. They do not. `build_instance_pool` additionally bans a decoy key holding one of the model's top-`ban` predicted tokens at the query; the induction census has no such condition. **The ban cannot simply be added, because it is alphabet-pathological**: over a twenty-symbol residue alphabet a top-20 ban covers the alphabet, and it is already measured emptying the decoy pool for **93.0%** of eligible positions on ProGen2-medium — which is why `--protein-ban-depths` exists at all. **The pathology is a property of the tokeniser and not of the modality, now measured on both protein tokeniser families (EXP-R2-088, width 192): the ban empties the decoy pool for 28589 of 31115 eligible positions on ProGen2-medium (vocabulary 31, per-residue) and for 5 of 10042 on ProtGPT2 (vocabulary 50257, byte-pair — identical to gpt2-large's). Instances retained move 12-fold on the first and 1.0003-fold on the second.** So "alphabet-pathological" names the per-residue tokenisers; the matched pair is exempt, and ZymCTRL at 458 sits between. Adding it here would delete the protein side of this census in order to make a text-side definition match. Recorded as an accepted limitation: the two corrections share the position window, the self- and same-token exclusions, the predecessor rule and the sink floor, and differ in the prediction ban, and the difference makes *this* correction the more conservative one — a decoy here may hold a plausible prediction, so its attention is subtracted as though it were positional baseline. L6's headline is unaffected either way: the tail count and the top-20 Jaccard are what it rests on.

**L2 — the event-selection ceiling.** `I(E;L) ≤ h(m/N)`. The gate demanded 0.1 nats from a top-100-of-122,671 design whose ceiling is **0.0066 nats** — 15.1x above attainable. The reported `0/38 FAIL` carries no biological content.

**L16 — decodability is not reliance.** ProtGPT2 encodes Pfam at **+0.705** linear skill while erasing it *helps* next-residue prediction by **−0.179 nats** [−0.245, −0.097]. ProGen2-medium relies on ss3 at only +0.031 nats.

**New — residue-level annotation is refused outright on subword protein arms.** `probe_erasure/protgpt2.json` refuses `ss3`, `burial` and `fitness`: a multi-residue BPE vocabulary has no residue-to-token map and must not be approximated by a heuristic. Combined with ProtGPT2 being the **only** public protein LM with genuine subword tokenisation (3.000 chars/token against RITA 0.984 and InstructProtein 0.984), the protein x subword cell is **permanently n = 1** for residue-level methods. This is a structural limit of the field, not of this panel.

**New — de-leaking destroys the causal denominator.** Unconditioned ZymCTRL has *negative* mean-ablation ΔCE (−0.081 to −0.142) and `denominator_valid = false` for burial, ec_class and ss3. Leak control and causal power are mutually exclusive on that arm.

**New — the frozen-attention approximation is NOT a protein-specific liability.** TG-06 measures it at **77% of context information in both** gpt2-large and ZymCTRL. Reporting this null is more informative than building an attribution graph.

**New — protein dictionaries are data-limited where text ones are saturated.** gpt2-large 0.958 → 0.962 across 16x data; ZymCTRL 0.610 → 0.716 → **0.843** and still climbing at ~+0.12 per 4x. That is convergence of the **instrument**, not of the model — a distinction this programme has been blurring.

**New — the one surviving pathway difference. Measured at production scale (EXP-R2-068, plan item B6).** Non-local propagation under activation patching: the share of single-token corruptions whose effect at a read-out position 33–64 tokens away clears the minimum-effect floor. Cohorts are seeded draws of 256 records; the interval resamples the **source sequence**, not the case. At these counts that is 1.3–1.7 cases per sequence rather than "many of a few" — the clustering is the conservative choice, not a large correction.

| arm | eligible / total | fraction | 95% interval | source sequences |
|---|---:|---:|---|---:|
| gpt2-large | 48 / 288 | **0.167** | [0.122, 0.215] | 168 |
| protgpt2 | 65 / 256 | **0.254** | [0.202, 0.307] | 162 |
| progen2-medium | 58 / 128 | **0.453** | [0.369, 0.538] | 100 |

**The gate is met and the ordering survives, but the modality separation does not.** All three arms clear the pre-registered ≥30 far-band cases (against 2–16 before), and every interval excludes zero. ProGen2-medium is disjoint from both other arms. **gpt2-large and ProtGPT2 overlap** on [0.202, 0.215] — so at production scale the text control and the matched protein arm are *not* separated, and what separates is ProGen2-medium from everything else. That is an architecture-and-tokenisation contrast as much as a modality one, and the matched pair is the only modality-identifying comparison the panel has (§2). **Still not a claim, and now for a sharper reason than sample size.**

> **Re-run in float32; the bfloat16 numbers above are withdrawn and the conclusion reverses (EXP-R2-068).**
>
> The run had been in bfloat16, whose quantisation step (1/16) is the size of the quantity: 17 of 18 band medians across the three arms were exact multiples of 1/16, and over half of ProGen2-medium's far-band effects underflowed to exactly zero. Re-run in float32 on the same cases, same cohort digest, same seeds, same 288/256/128 cases per band:
>
> | arm | bfloat16 | **float32** | 95% interval | eligible | source sequences |
> |---|---:|---:|---|---:|---:|
> | gpt2-large | 0.1667 | **0.1042** | [0.0694, 0.1399] | 30/288 | 168 |
> | protgpt2 | 0.2539 | **0.1953** | [0.1492, 0.2424] | 50/256 | 162 |
> | progen2-medium | 0.4531 | **0.2734** | [0.2015, 0.3548] | 35/128 | 100 |
>
> **The ordering is threshold-invariant, and on this window the matched pair separates — see the withdrawal below.** gpt2-large's upper bound (0.1399) sits below ProtGPT2's lower bound (0.1492); gpt2-large is disjoint from *both* protein arms while the two protein arms overlap each other — which is the shape a modality difference should have. Text < ProtGPT2 < ProGen2-medium holds at **all six** cuts from 0.05 to 2.0, a fortyfold range. Every arm clears the ≥30 case gate. The bfloat16 run had shown the matched pair overlapping and the ordering reversing at permissive cuts: **the quantisation was manufacturing the threshold-dependence that was reported as the limitation.**
>
> **The interval separation does NOT survive a disjoint cohort window (EXP-R2-068, same session).** Re-run in float32 on a disjoint window of the same permutation (`--cohort-skip 256`): gpt2-large is essentially immovable, 0.1042 [0.0694, 0.1399] → 0.1042 [0.0712, 0.1388], while **ProtGPT2 moves −0.051**, 0.1953 [0.1492, 0.2424] → 0.1445 [0.1028, 0.1880] — and on that window the two intervals **overlap** (gpt2-large q975 0.1388 against ProtGPT2 q025 0.1028). So the matched-pair *separation* claimed above is draw-dependent and is **withdrawn**. What survives both windows is the *point ordering*: gpt2-large below ProtGPT2 in each, by 0.091 and 0.040.
>
> The asymmetry is itself the finding, and it is the one this programme already knows: the text arm's far-band fraction is insensitive to which records were drawn and the protein arm's is not. That is §5.05(b)'s protein-specific cohort sensitivity — recorded there at 10–35x larger on the protein side, in nats of context information — reappearing in an unrelated statistic. **A protein arm's number carries a selection uncertainty its matched text control does not, and any protein-vs-text interval comparison drawn from a single window overstates its own precision.** ProGen2-medium's disjoint window has not been run, so the three-arm ordering's robustness is untested.
>
> **Extended to five arms in float32 (EXP-R2-068), and a second scale effect appears with the opposite sign.**
>
> | arm | params | eligible / total | fraction | 95% interval | vs text control |
> |---|---:|---:|---:|---|---|
> | gpt2-large | 774M | 30/288 | **0.1042** | [0.0694, 0.1399] | — |
> | progen2-small | 151M | 23/128 | 0.1797 | [0.1136, 0.2540] | overlaps; **gate FAILS** at 23 < 30 cases |
> | protgpt2 | 774M | 50/256 | 0.1953 | [0.1492, 0.2424] | disjoint |
> | progen2-medium | 765M | 35/128 | 0.2734 | [0.2015, 0.3548] | disjoint |
> | progen2-base | 765M | 40/128 | 0.3125 | [0.2326, 0.3984] | disjoint |
>
> **All four protein arms sit above the text control by point estimate**, and three of the four are interval-disjoint from it on this window — but read the disjoint-window block below before using this row: a second draw removes ProtGPT2 from it, and ProGen2-medium is the only arm that survives draw and threshold together. ProGen2-small is *not* a gated result: 23 eligible far-band cases against the pre-registered floor of 30, so it is reported as underpowered rather than as an overlap, and it needs roughly 176 cases per band.
>
> **The propagation slope is positive where the head-count slope is negative.** On the same ProGen2-small / ProGen2-medium pair, with corpus, architecture and tokeniser held: far-band propagation rises **+0.1330/decade** while the induction head fraction falls **−0.0172/decade**. A larger protein decoder in this lineage propagates a single-token perturbation *further* while allocating *fewer* heads to prefix-matching. Those are opposite-signed scale effects in one lineage on two statistics that a "more induction machinery" account would move together, and neither is a modality claim.
>
> **Every interval separation in this table is provisional on one cohort window.** The disjoint-window check above moved ProtGPT2 by −0.051 and dissolved its separation from the text control; the equivalent check for ProGen2-medium has now landed.
>
> **The second window splits the two protein arms, and it splits them the same way the threshold sweep does (EXP-R2-069).** Window 2 is `--cohort-skip 256`, float32, everything else identical:
>
> | threshold | gpt2-large w1 → w2 | ProtGPT2 w1 → w2 | ProGen2-medium w1 → w2 | both protein above control? |
> |---:|---|---|---|---|
> | 0.05 | 0.5312 → 0.5972 | 0.6133 → **0.5586** | 0.6719 → 0.7031 | w1 holds, **w2 fails** |
> | 0.10 | 0.3611 → 0.4132 | 0.4297 → **0.3984** | 0.4375 → 0.5859 | w1 holds, **w2 fails** |
> | 0.25 (headline) | 0.1042 → 0.1042 | 0.1953 → **0.1445** | 0.2734 → 0.3203 | holds in both |
> | 0.50 | 0.0347 → 0.0382 | 0.0781 → **0.0664** | 0.1406 → 0.1562 | holds in both |
>
> **What survives is one arm, not the modality.** ProGen2-medium sits above the text control in *both* windows at *every* threshold on the ladder, and is interval-disjoint from it at 0.25 and 0.50 in both — the only cell of this table that is robust to draw and threshold at once. **ProtGPT2 does not survive:** it moves *down* in the second window at all four thresholds while the text control moves *up*, and at the two permissive cuts the two arms cross. Its margin over its matched control is smaller than the movement between two draws of one permutation, which is a statement about the precision of the measurement rather than about ProtGPT2.
>
> So §5.1's "all four protein arms sit above the text control" must be read as a one-window point ordering, and the retained claim narrows to: **the ProGen2 lineage propagates a single-token perturbation further than the matched text control; ProtGPT2 is not distinguishable from it.** That keeps the far-band result an architecture-and-tokenisation contrast, as the paragraph above already concluded on independent grounds, and removes the reading in which it is a text-vs-protein contrast — the matched pair is precisely the comparison that fails.
>
> **The text control is the immovable arm here, and that is the third time this asymmetry has appeared.** gpt2-large returns 0.1042 in both windows to four decimal places at the headline threshold, while both protein arms move by 0.03–0.15. §5.05(b) recorded the same asymmetry in nats of context information at 10–35× and the withdrawn matched-pair interval separation recorded it a second time. It is now a documented property of this panel rather than an incident: **a protein arm's cohort carries a selection uncertainty its matched text control does not**, on at least three unrelated statistics.
>
> ### EXP-R2-070 — five windows, equal case counts, and the separation is a large-effect-tail phenomenon
>
> Six arms, five disjoint windows of one seeded permutation, **256 cases per band on every arm**, float32, 30 of 31 jobs clean. This replaces the two-draw sensitivity check with a characterised draw distribution and removes the unequal-case-count confound.
>
> **(i) The pre-registered ≥30 far-band case gate is not attainable on the text control, and that is a specification defect.** At 256 cases per band gpt2-large returns 26 / 24 / 26 / 28 / 34 eligible far-band cases — **below the gate in four of five windows** — and gpt2-xl misses it in one. Appendix B rule 2 says a gate the positive control cannot pass is a specification defect rather than a negative result, and this one was violated in the recording direction: ProGen2-small's "gate FAILS at 23" in the table above was judged against a floor its own text control does not clear at the same case count. gpt2-large only met it before by running at 288 cases per band and landing on exactly 30. **The gate is withdrawn as stated.** ProGen2-small, at 33–62 cases here, clears it in every window anyway.
>
> **(ii) The paired per-window test, all four thresholds, both text controls.** Counting every arm × control × window comparison at each threshold:
>
> | threshold | protein higher | text higher | tied |
> |---:|---:|---:|---:|
> | 0.05 | 31/40 | 9 | 0 |
> | 0.10 | 33/40 | 7 | 0 |
> | 0.25 | 36/40 | 4 | 0 |
> | **0.50** | **39/40** | **0** | 1 |
>
> **At the strictest cut every protein arm is above every text control in every window, with no exception.** The single non-positive cell is an exact tie (ProGen2-small against gpt2-large at skip 1024). The trend across the ladder is monotone: the separation is not a property of far-band propagation in general but of its **large-effect tail**. That is the shape §5.1 already inferred from one window — "text has many small effects, ProGen2-medium has fewer and much larger ones" — now measured across five draws and two controls.
>
> **(iii) Which arms survive draw *and* threshold.** ProGen2-base and ProGen2-medium are above both text controls in all five windows at all four thresholds — 40 of 40 comparisons each. ProtGPT2 and ProGen2-small are not: they hold at 0.25 and 0.50 but cross at the permissive cuts, confirming EXP-R2-069's narrowing rather than reversing it. **The matched pair separates only in the large-effect tail**, where ProtGPT2's smallest margin is +0.0039.
>
> *What this is not.* At K = 5 the exact two-sided sign test floors at p = 0.0625, so **no single arm-control pair reaches p < 0.05 on window sign alone**; and the 40 comparisons share windows and controls, so they are not 40 independent tests and no binomial over them is quoted. The evidence is the consistency pattern and its monotone dependence on threshold, not a p-value.
>
> **(iv) The cohort-sensitivity asymmetry, fourth appearance and first one immune to sample size.** Between-window standard deviation of the fraction at threshold 0.25, every arm on 256 cases: gpt2-large **0.0150**, gpt2-xl **0.0153**; ProtGPT2 0.0243, ProGen2-small 0.0435, ProGen2-base 0.0476, ProGen2-medium **0.0519**. The two modalities do not overlap — the smallest protein spread is 1.59× the largest text one. Earlier appearances (§5.05(b) at 10–35× in nats, the withdrawn matched-pair separation, EXP-R2-069) could all be argued down as sample-size artefacts because the arms carried different case counts. **This one cannot.** A protein arm's cohort carries a selection uncertainty its matched text control does not, and it is now measured on equal footing.
>
> *One job of 31 did not run.* ZymCTRL at the ~816-token window failed on a transport timeout during the code-snapshot push, not on anything it measured. It produced nothing and is queued for a re-run; whether ZymCTRL can enter this estimand at all remains open (L15).
>
> *Two design faults in this comparison, both now fixed for the successor run.* The arms carried unequal case counts — 288 for gpt2-large against 128 for ProGen2-medium — so interval-overlap tests read tighter for one arm than the other, and an ordering conclusion drawn from overlap was partly reading sample size. And two windows sample a draw distribution rather than characterise it: with K=2 there is no way to tell a shifted arm from a noisy one. EXP-R2-070 **ran** five disjoint windows at one case count (256) across six arms; its result is the block above, and it both confirmed this narrowing and withdrew the case gate the table above judges ProGen2-small against.

> ### WITHDRAWN 2026-07-31 (EXP-R2-073) — the far-band distance band is declared in tokens, and the sign of the contrast is a free parameter of that choice
>
> **`DISTANCE_BANDS` is a module constant in token units** (`src/transfer/circuits.py:88-95` as it stood; lines 72-80 are now the comment the fix added) with no per-arm resolution and no CLI knob at the time — `DISTANCE_UNITS`, `SYMBOL_DISTANCE_BANDS`, `resolve_distance_bands` and `--patch-distance-unit` were added for the successor below, while `04_circuit_primitives.py` records `tokenisation.symbols_per_token` in the same artefact and never uses it. Measured on the shipped five-window artefacts: **4.403 symbols/token** (gpt2-large, gpt2-xl), **2.816** (ProtGPT2), **0.996** (every ProGen2 arm). So the "33–64 token" far band is 33–64 residues on ProGen2, ~93–180 residues on ProtGPT2, and ~145–282 characters on GPT-2 — and the "single-token corruption" is one residue, ~2.8 residues, or ~4.4 characters. The estimand's two length scales are in different content units on different arms.
>
> **The decisive check needs no cross-modality unit at all.** ProtGPT2 and ProGen2 are both protein decoders whose content alphabet is residues; only the tokeniser differs. Reading each arm's own six-band curve, no interpolation, mean over the five windows at threshold 0.25:
>
> | comparison | ProtGPT2 | ProGen2-base | ProGen2-medium |
> |---|---:|---:|---:|
> | at 33–64 **tokens** (the published band) | **0.157** | **0.305** | **0.288** |
> | at 25–45 **residues** (ProtGPT2's 9–16 token band) | **0.381** | — | — |
> | at 48–90 **residues** (ProtGPT2's 17–32 token band) | **0.281** | — | — |
> | at 33–64 **residues** (ProGen2's 33–64 token band) | ≈0.335 (interpolated) | 0.305 | 0.288 |
>
> At matched *token* distance ProtGPT2 sits well below both ProGen2 arms; at matched *residue* distance it sits at or above both, and it does so on raw bands — its 48–90-residue value, at a **longer** content distance, already equals ProGen2's 33–64-residue value. The same reversal holds at threshold 0.50. **Two protein arms swap order on the unit choice alone**, so "protein propagates a single-token perturbation further" is not a property of protein.
>
> Re-indexing every arm by content symbols and interpolating at **24 content symbols** — inside the measured range for all six arms — gives gpt2-large 0.516, gpt2-xl 0.515, ProtGPT2 0.443, ProGen2-medium 0.407, ProGen2-base 0.383, ProGen2-small 0.236: **the ordering reverses, with the text controls highest.**
>
> **Neither alignment isolates the model, and that is the point.** Token-matching matches positional distance and mismatches perturbation size; symbol-matching matches content distance and mismatches perturbation size the other way. The confound acts at two places with opposite signs, so there is no post-hoc re-indexing of these artefacts that recovers a model-attributable ordering.
>
> **What is withdrawn.** EXP-R2-070 (ii)'s "at the strictest cut every protein arm is above every text control in every window, with no exception" and (iii)'s 40-of-40 for ProGen2-base and ProGen2-medium are *arithmetically correct and reproduce exactly from the artefacts* — I re-derived 31/33/36/39 of 40 and the 40/40 cells — but they are statements about a token band, not about the models. §9.2's "the ProGen2 lineage propagates a single-token perturbation further than the matched text control" is withdrawn, and **D1.a's status reverts from *answered* to *open*.** EXP-R2-070 (iv)'s between-window spread asymmetry is *not* withdrawn on this ground — it is a within-arm quantity at a fixed band — but its levels are unit-dependent and the spread of a proportion depends on its level, so it needs re-derivation at matched content distance before it is quoted again.
>
> **A second, independent correction to the same block.** §5.1 describes the clustering as "1.3–1.7 cases per sequence … the clustering is the conservative choice, not a large correction." That is true of the earlier `b6_float32` run, which drew cohorts of 256 records and reached 100–168 clusters. **It is false of EXP-R2-070**, which ran at the default `--cohort-size 24`: verified in every artefact, 256 cases from **24 source sequences on every arm**, i.e. 10.7 cases per sequence and 24 bootstrap clusters. The correction is large, not small, and it is asymmetric — the clustered interval is *narrower* than the naive binomial on the text arms and up to ~1.7x wider on the protein arms. Every interval-disjointness statement in EXP-R2-069 and EXP-R2-070 therefore rests on 24 clusters, well below the count at which `statistics.py`'s own coverage simulation reaches nominal. The sign of that bias is conservative for the direction that was reported, which is why it did not surface earlier.
>
> **The cross-lab lineage control passes, and it localises the weak cell (EXP-R2-073).** Both of EXP-R2-070's text controls are GPT-2, so its pattern could have been a GPT-2 property — the check that killed the QK/OV finding. `qwen2.5-0.5b` and `llama-3.2-3b` were run at the same protocol on three of the five windows (the queue was stopped after three because llama at batch 16 was running ~80 minutes a job and the successor campaign was the better use of the cards; this is a three-window control, not a five-window one). Paired per-window comparison at the **token** band, four text controls × four protein arms × three windows:
>
> | threshold | protein higher | text higher | tied | EXP-R2-070 (2 GPT-2 controls, 5 windows) |
> |---:|---:|---:|---:|---|
> | 0.05 | 39/48 | 8 | 1 | 31/40 |
> | 0.10 | 40/48 | 6 | 2 | 33/40 |
> | 0.25 | 44/48 | 4 | 0 | 36/40 |
> | 0.50 | **44/48** | **0** | 4 | 39/40 |
>
> **The pattern is not a GPT-2 idiosyncrasy**, and the monotone dependence on threshold reproduces. **Every non-positive cell at the two strict cuts is ProtGPT2 against llama-3.2-3b** — the matched-pair protein arm against the largest text arm, where llama's mean far-band fraction (0.160) sits *above* ProtGPT2's (0.147). That is the third independent route to the same narrowing: EXP-R2-069 found it by cohort draw, §5.1 by architecture, and now the lineage control finds it too. ProtGPT2 is the arm that does not separate.
>
> **A text-side scale slope appears, which the two-arm control could not show.** Mean far-band fraction at threshold 0.25: qwen2.5-0.5b (0.5B) 0.096, gpt2-large (774M) 0.099, gpt2-xl (1.56B) 0.135, llama-3.2-3b (3.2B) 0.160. Text propagation rises with scale, as the ProGen2 pair already showed for protein (0.214 → 0.305 from 151M to 765M). And it is **not** explained by symbols per token: qwen 4.645, llama 4.773, gpt2 4.494 do not order the same way as the fractions, so tokenisation granularity is a confound on the *cross-modality* comparison without being the driver of the within-text variation.
>
> **EXP-R2-070 (iv)'s spread asymmetry survives but its margin is not what was recorded.** Between-window sd at threshold 0.25 on the three common windows: llama-3.2-3b 0.0039, gpt2-large 0.0045, qwen2.5-0.5b 0.0180, gpt2-xl 0.0193 against ProtGPT2 0.0260, ProGen2-small 0.0260, ProGen2-medium 0.0412, ProGen2-base 0.0418. The modalities still do not overlap, but at **1.35x** rather than the recorded 1.59x, and the text side now spans **5x across four arms** where two arms had suggested it was tight. At K = 3 the relative standard error of an sd is ~50%, so this neither confirms nor refutes the asymmetry; what it does refute is the strength of the claim that it "cannot be argued down". It should be re-derived at matched content distance with more windows before it is quoted again.
>
> **New standing rule (Appendix B rule 26).** Declare the unit of any distance, window or band, and resolve it per arm when the arms differ in symbols per token.
>
> The successor measurement is the same instrument with the band declared in content symbols and resolved per arm, recording both the requested symbol band and the resolved token band. Until it lands, no far-band ordering across arms of different tokenisation may be cited.

> ### EXP-R2-074 — the successor, and what survives is a decay rate rather than a level
>
> Six arms, bands and corruption span **both** declared in content symbols and resolved per arm by containment, float32, 256 cases per band, **five disjoint windows** of one seeded permutation on every arm — 30 of 30 artefacts. Resolved token geometry, which the artefact records: gpt2-large / gpt2-xl 2–3, 4–7, 8–14 tokens at 4.56 characters per token; ProtGPT2 4–5, 7–11, 12–22 at 2.81 residues per token; every ProGen2 arm 10–16, 18–32, 34–64 at 0.996. Corruption span is 5 characters (one text token) and 3 residues (one ProtGPT2 token, its floor).
>
> **The level comparison reverses, exactly as the withdrawal predicted.** At the shortest content band, 9–16 symbols, the text arms are *highest*: gpt2-large 0.671 and gpt2-xl 0.704 against ProtGPT2 0.545, ProGen2-small 0.525, ProGen2-medium 0.656, ProGen2-base 0.664. Anyone reading "protein propagates a single-token perturbation further" off the token-band tables was reading the tokeniser.
>
> **What separates is the decay with distance, and it separates completely.** Ratio of the 33–64-symbol band to the 9–16-symbol band, mean over windows, threshold 0.25:
>
> | arm | modality | 9–16 | 17–32 | 33–64 | **decay ratio** | between-window sd |
> |---|---|---:|---:|---:|---:|---:|
> | gpt2-large | text | 0.670 | 0.521 | 0.328 | **0.489** | 0.039 |
> | gpt2-xl | text | 0.703 | 0.523 | 0.345 | **0.490** | 0.055 |
> | progen2-small | protein | 0.525 | 0.420 | 0.312 | **0.596** | 0.080 |
> | protgpt2 | protein | 0.545 | 0.448 | 0.382 | **0.706** | 0.104 |
> | progen2-base | protein | 0.664 | 0.570 | 0.478 | **0.720** | 0.061 |
> | progen2-medium | protein | 0.656 | 0.573 | 0.481 | **0.732** | 0.047 |
>
> Text 0.489–0.490 against protein 0.596–0.732, a gap of **+0.105**; at threshold 0.50, text 0.313–0.329 against protein 0.500–0.624, a gap of **+0.172**. **Protein decays more slowly in 40 of 40 paired arm × control × window comparisons at both thresholds, with no exceptions.**
>
> **Three reasons this is better founded than what it replaces.** It is a *within-arm ratio*, so the arm-specific level — the quantity the perturbation-size mismatch moves — largely cancels. Both legs of the confound are declared in the same unit. And **the matched pair separates for the first time on this estimand**: gpt2-large 0.489 against ProtGPT2 0.706, with ProtGPT2 sitting *inside* the protein range rather than being the arm that fails, which it was under every previous version of this measurement.
>
> **A fourth thing, which is a genuine surprise and a check on Appendix B rule 22.** The between-window spread of the decay ratio does **not** separate by modality — text 0.039 and 0.055 against protein 0.047 to 0.104 at threshold 0.25 — ProGen2-medium's 0.047 sits *below* gpt2-xl's 0.055 — and the two ranges overlap at threshold 0.50 as well, where ProGen2-base reads 0.034 against gpt2-xl's 0.079. The protein-specific cohort sensitivity that appears in four other statistics does not appear in this one. That is rule 22 working in the direction that helps: cohort sensitivity is a property of the statistic, and this statistic happens to be a robust one.
>
> **The cross-lab control passes, and the margin narrows honestly (EXP-R2-075).** Both of the text controls above are GPT-2, which is the configuration that killed the QK/OV finding. `qwen2.5-0.5b` and `llama-3.2-3b` — rotary, RMSNorm, gated FFN, vocabularies 2.5–3x larger — were run at the identical content-symbol geometry on all five windows. Decay ratio at threshold 0.25: qwen2.5-0.5b **0.527**, llama-3.2-3b **0.526**, against gpt2-large 0.489 and gpt2-xl 0.490. Both sit above the GPT-2 arms and both stay below every protein arm. With four text controls the separation holds — **77 of 80** paired arm × control × window comparisons at threshold 0.25 and **79 of 80** at 0.50 — but the gap narrows from +0.105 to **+0.069** at 0.25 and from +0.172 to **+0.106** at 0.50, because qwen is now the worst text arm rather than gpt2-large. The three non-positive cells at 0.25 are ProGen2-small, the lowest protein arm, against qwen.
>
> **The text decay ratio does not track scale, and the protein one does.** Text reads 0.527 (494M), 0.489 (774M), 0.490 (1.56B), 0.526 (3.21B) — flat to within ±0.02 across a **6.5x** parameter range and two lineages, with no monotone order. Protein rises: ProGen2-small 0.596 (151M) to ProGen2-medium 0.732 (765M), while ProGen2-base and ProGen2-medium — both 765M, differing only in pretraining mixture — read 0.720 and 0.732, so the rise is scale rather than corpus. **This matters for what could overturn the result.** For the census ρ the text side declines with scale, so a much larger protein decoder is the risk. Here the risk runs the other way: the closest protein arm to the text band is the *smallest* one, and it is already measured. A larger protein decoder would widen this gap, not close it.
>
> **NARROWED by the off-distribution control (EXP-R2-076). At the permissive threshold the ranges overlap.** The text side was extended to seven arms — gpt2, gpt2-medium and `dialogpt-small` added to the four above — and the last of those is the control that made the census result convincing, a *text* decoder §5.05(a) shows is off-distribution at −4.08 nats. On the census it behaved like a text arm. **Here it does not.** Decay ratio at threshold 0.25:
>
> | | text arms | protein arms |
> |---|---|---|
> | @ 0.25 | gpt2 0.423, gpt2-large 0.489, gpt2-xl 0.490, gpt2-medium 0.509, llama 0.526, qwen 0.527, **dialogpt-small 0.611** | ProGen2-small 0.596, ProtGPT2 0.706, ProGen2-base 0.720, ProGen2-medium 0.732 |
> | @ 0.50 | gpt2 0.287, gpt2-large 0.313, gpt2-xl 0.329, gpt2-medium 0.348, qwen 0.363, llama 0.394, **dialogpt-small 0.397** | ProGen2-small 0.500, ProtGPT2 0.608, ProGen2-medium 0.615, ProGen2-base 0.624 |
>
> **At threshold 0.50 the ranges still separate cleanly** — the worst text arm, dialogpt-small at 0.397, sits below the lowest protein arm at 0.500, and protein is slower in **136 of 140** paired comparisons. **At threshold 0.25 they overlap by 0.015**, entirely because dialogpt-small (0.611) exceeds ProGen2-small (0.596); protein is still slower in 133 of 140. So this separation, like the far-band level result before it, **lives in the large-effect tail** — the same shape §5.1 has now found three times.
>
> **Is the decay ratio measuring distributional fit rather than modality?** That is the hypothesis dialogpt-small raises, and it is testable against the arms' own context information. Across all ten arms Spearman(context information, decay ratio) is **−0.62** (p = 0.054) — but that is the modality split restated, since protein arms have both low context information and high decay. **Within the text arms it is −0.14 (p = 0.76), and excluding dialogpt-small it is +0.37.** So context information does not predict the decay ratio among comparably-fitted models, and the confound is *not established*. Nor is it refuted: dialogpt-small is the only badly-fitted arm on the panel, and it is the one that overlaps. **A second off-distribution arm is what would settle this, and the panel does not have one** — which is a structural limit of the same kind as §2's, not an oversight.
>
> **The residual confounds, stated rather than engineered around.** A residue is not a character, so 3 residues against 5 characters is matched within modality and not across it; the ratio cancels the level but not this. Integer token quantisation makes the *token*-distance ratio 4.3x for text and 3.7x for ProGen2, which biases toward faster text decay — correcting for it under a power law moves gpt2-large from 0.489 to ≈0.53, still below the lowest protein arm, so the separation survives the correction with less room. The 40 comparisons share windows and arms and are not 32 independent tests. And this remains an architecture-and-tokenisation contrast as much as a modality one: the panel still has one matched protein arm (§2).

> Read the withdrawal above before this paragraph. What survives it: the measurement is at production scale, in resolved float32 arithmetic, with sequence-clustered intervals and equal case counts, and its *within-arm* quantities are sound. What does not: any cross-arm ordering, because the arms were compared at a distance that means something different on each of them. The limits that were already recorded stand unchanged and are now joined by a larger one — the structural n = 1 limit of §2, the corpus-repeat confound, and ZymCTRL's structural exclusion from the estimand. Artefacts: `results/transfer_20260730/b6_float32/`, `results/transfer_20260730/farband5/`.

**New — protein dictionaries are data-limited where text ones are saturated.** gpt2-large 0.958 → 0.962 across 16x data; ZymCTRL 0.610 → 0.716 → **0.843** and still climbing at ~+0.12 per 4x. That is convergence of the **instrument**, not of the model — a distinction this programme has been blurring.

**New — the one surviving pathway difference. Measured at production scale (EXP-R2-068, plan item B6).** Non-local propagation under activation patching: the share of single-token corruptions whose effect at a read-out position 33–64 tokens away clears the minimum-effect floor. Cohorts are seeded draws of 256 records; the interval resamples the **source sequence**, not the case. At these counts that is 1.3–1.7 cases per sequence rather than "many of a few" — the clustering is the conservative choice, not a large correction.

| arm | eligible / total | fraction | 95% interval | source sequences |
|---|---:|---:|---|---:|
| gpt2-large | 48 / 288 | **0.167** | [0.122, 0.215] | 168 |
| protgpt2 | 65 / 256 | **0.254** | [0.202, 0.307] | 162 |
| progen2-medium | 58 / 128 | **0.453** | [0.369, 0.538] | 100 |

**The gate is met and the ordering survives, but the modality separation does not.** All three arms clear the pre-registered ≥30 far-band cases (against 2–16 before), and every interval excludes zero. ProGen2-medium is disjoint from both other arms. **gpt2-large and ProtGPT2 overlap** on [0.202, 0.215] — so at production scale the text control and the matched protein arm are *not* separated, and what separates is ProGen2-medium from everything else. That is an architecture-and-tokenisation contrast as much as a modality one, and the matched pair is the only modality-identifying comparison the panel has (§2). **Still not a claim, and now for a sharper reason than sample size.**

> **Two sensitivity checks were then run, and the second withdraws the numbers above (EXP-R2-068).**
>
> *Sampling.* A disjoint second window of the same permutation (`--cohort-skip 256`) moves the far-band fraction by **−0.021 / −0.020 / +0.016** for gpt2-large / ProtGPT2 / ProGen2-medium, every interval overlaps its partner, and all three still clear the case gate (42 / 60 / 60). The draw is not driving the result.
>
> *Threshold.* It is not threshold-invariant, and Appendix B rule 17 is the reason to have looked. Swept over the eligibility cut:
>
> | cut | gpt2-large | ProtGPT2 | ProGen2-medium |
> |---:|---:|---:|---:|
> | 0.05 | **0.771** | 0.715 | 0.484 |
> | 0.10 | 0.455 | **0.563** | 0.477 |
> | 0.25 | 0.146 | 0.234 | **0.469** |
> | 0.50 | 0.052 | 0.078 | **0.461** |
> | 1.00 | 0.017 | 0.023 | **0.258** |
>
> **At the two most permissive cuts the ordering reverses or scrambles: at 0.05 the text control propagates *more* than either protein arm.** The recorded ordering exists only from 0.25 upward. So the honest statement is not that protein propagates further, but that the far-band effect *distributions have different shapes* — text has many small effects, ProGen2-medium has fewer and much larger ones.
>
> *And the measurement was not trustworthy at this magnitude.* The run was in **bfloat16**, and the far-band `|effect|` quantiles came back at exact multiples of 1/16 (0.0625, 0.1250, 1.0000) — the quantisation step was the size of the quantity being measured. This is the hazard `14_paa_census.py` already carries a `--census-dtype float32` default for, on the same reasoning.
>
> **Confirmed by re-running gpt2-large in float32, and the error was large.** The quantiles become continuous (0.0061 / 0.0236 / 0.0549 / 0.1420 / 0.3595 at q05 / q25 / q50 / q75 / q95) and the far-band eligible fraction falls from **0.1667 to 0.1042** — bfloat16 inflated it by 60% in relative terms, and inflated it *upward*, in the direction that makes the text control look closer to the protein arms. Every bfloat16 far-band number in this section is therefore withdrawn, including the sweep and the disjoint-window comparison, which were computed on the same quantised effects. The two protein arms are re-running.
>
> What survives independently of the magnitudes: the gate is attainable (float32 gpt2-large gives 30 eligible far-band cases at 288 per band, exactly at the pre-registered floor), the sampling insensitivity is a statement about which records were drawn rather than about effect size, and ZymCTRL's structural exclusion is a property of its rendering. **Standing rule to add: measure a far-band effect in float32.** An effect of order 0.05 logits cannot be read off a bfloat16 metric whose step is 0.0625 — the same lesson as EXP-R2-059's M-gap, relearned on a different statistic because the default was never changed for this stage.

**Two further corrections this run forces.** First, **the previously recorded trio 9% / 34% / 50% is not reproducible from the retained artefacts.** Their far-band eligible fractions are 0.250 (gpt2-large), 0.219 (ProtGPT2), 0.500 (ProGen2-medium) and 0.062 (ZymCTRL); only ProGen2-medium's matches, and the accompanying "2–16 eligible far-band cases" matches the counts 8 / 7 / 16 / 2 exactly. The figures should be read as superseded rather than confirmed. Second, **ZymCTRL cannot enter this measurement at all.** `build_patch_cases` cuts every row to the patching window, and a conditioned rendering puts the `<end>` marker that delimits scored content hundreds of tokens beyond a 128-token window, so no valid content span exists. `build_patch_cases` now refuses the arm up front from the declaration (`patch_seq_len_refusal`); when this was measured it was `content_bounds` refusing one row at a time inside the loop, after the checkpoint was already on the GPU. Reaching it needs an ~816-token window at roughly 2.5 GPU-h, or a short protein band incommensurable with the other three. This is the conditioning prompt (L15) removing an arm from a window-based estimand — recorded, not worked around. Artefacts: `results/transfer_20260730/b6_nonlocal_propagation/`.

### 5.2 A fifth cause the evidence demands

The four candidate causes named for this programme — insufficient pretraining, limited output interface, divergent pathways, method-intrinsic non-transferability — do not cover where several retractions actually came from. A fifth is required:

**Protein-specific measurement substrate.** Input rendering worth 1.42 nats/token (L11); conditioning-tag leakage worth 1.73 nats (L15); corpus file order worth up to 1.01 nats (L13); corpus repeat statistics at 32.3% against 0.402%; a distance band declared in tokens whose ordering reverses in residues (L23); and a head selector that ranks causal importance on text and not on protein (L22). None is a property of protein *computation*; all are properties of measuring protein models. The text-derived literature treats input rendering as fixed, corpus order as irrelevant, and a token as a unit of content, because in text those are nearly true.

Attribution summary across all catalogued limitations:

| cause | support |
|---|---|
| Insufficient pretraining / convergence | **weak for the model, clear for the dictionary** — the matched-corpus contrast moves MLP share 0.019 against a ~0.43 gap (~1/22) and induction 0.0069 against −0.10 (~7%) |
| Limited output semantic interface | **strong** — algebraic (rank 31), definitional (closure), and measured |
| Divergent computational pathways | **not supported** — four instruments disagree; the one provisional survivor (far-band propagation) was withdrawn on EXP-R2-073's unit confound |
| Method-intrinsic non-transferability | **strong** — every instance demonstrated on the text control |
| Protein-specific measurement substrate | **strongest** — source of six retractions, and now of the programme's best part-2 result (L22) |

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
| *(circuit census — not on the original list)* | yes | **L22: the selector ranks causal importance on text and not on protein, with the off-distribution text control ruling out distribution mismatch; L5: `paa_specific` screen failure for copy-suppression on GPT-2-large** | substrate |

---

## PART III — Toward adapted methods

### 7. Standing rejections

Recorded so they are not re-proposed under new names. Both have been rejected multiple times on unchanged grounds.

- **Property-conditioned Jacobian subspace** (rejected 3x). `J = ∂logits/∂h` is defined against the model's output. Secondary structure, Pfam, EC and fitness are not functionals of the next-residue distribution, so their directions can only come from a trained probe — making the object the Jacobian of the probe. Concept erasure answers the underlying question better because it measures reliance rather than decodability (L16).
- **Cross-position residue-pair sparse features** (gated). Motivation retracted: the attention margin over per-position marginals fell to 0.03–0.05 after correcting structure-selection order, and to ~0.03 over the separation-only control. Requires the relational effect to be re-established at adequate power before any construction.

### 8. Where a proposal could be earned

No proposal is advanced here that is not traceable to a catalogued limitation. The best-supported openings, in order:

0. **From L22 (new, EXP-R2-071/072) — a census-validity diagnostic, and it is the first opening this programme has earned rather than assumed.** The measured limitation is specific: a head-prevalence census's selector orders causal importance on text decoders and does not on protein decoders, the failure is in the bulk of the grid rather than at its top, and an off-distribution *text* control does not reproduce it. That licenses a small, general instrument — before a prevalence census is reported for any model, exhaustively patch a bounded random sample of heads and report the selector-to-causal rank correlation against the text control's value; below it, the count is not interpretable and must be published as a selector diagnostic rather than as a mechanism count. It is cheap, it is threshold-free, its attainability is demonstrated on the positive control by construction, and it is the rare case where the adapted method *is* the measurement that exposed the limitation. **Still gated**: it rests on one mechanism, so D2.c must replicate it on copy suppression first. If D2.c returns a text-like correlation on protein, this opening closes.

1. **From L5** — a causal-effect-based mechanism comparison over a fixed budget of exhaustively tested heads, requiring no valid cheap screen. Trades coverage for validity. Per-instance ΔM-gap matrices were retained, so its feasibility is answerable without a rerun. *Partly discharged for induction by EXP-R2-071/072; open for copy suppression as D2.c.*
2. **From L1/L3/L4** — a dictionary-evaluation protocol whose estimand is power-checked on the text control before any protein arm is scored, reporting loss recovered and KL rather than FVU.
3. **From L8** — a readout that does not route through the unembedding, since the rank-(V−1) aperture is a hard vocabulary-conditional algebraic constraint and small residue vocabularies make it especially restrictive.

Any method that is built is benchmarked against SAE, CLT, probes, dense low-rank and random controls, on fidelity, causal selectivity, cross-seed stability and generalisation to disjoint protein families.

---

## 9. The plan, organised by the three directions

Restructured 2026-07-30 (EXP-R2-068). The previous Phase A/B/C scheme had drifted
from the objective it served: items were numbered by the order they were thought
of, and two of them turned out to belong to different directions. This version is
organised by the three directions themselves, with one-letter identifiers, and
each direction states what it has established before what it still owes. Retired
item names are kept in §9.1 so nothing is lost.

Remaining budget **~55–95 H200 GPU-hours**. The programme has spent about 8 to
date, which is the main reason the estimate has fallen: measurements that were
costed as campaigns turned out to be minutes once the case counts were sized to
the gate rather than to the panel.

### D1 — Differences between the model families

**Established.** One difference survives as a bounded, reproducible statement: on
repeat probes drawn from the whole matching population, protein decoders put a
smaller fraction of heads in the upper tail of the prefix-matching distribution
than text decoders. As of EXP-R2-068 that ordering holds at the headline
threshold on **all three** probe constructions, matched pair 5.46x — the recorded
inversion on the best probe was an artefact of the repeat cohort's file-order
draw. It remains **not a modality claim**: five text arms against one protein arm
in the only family spanning both (§2), and an eightyfold corpus repeat-prevalence
confound that no experiment can remove.

Two further differences are open rather than closed:

| item | question | status | cost |
|---|---|---|---|
| **D1.a** | Do the far-band propagation *magnitudes* differ, once resolved? | **Reopened, then re-answered, and the answer changed (EXP-R2-073 / 074).** The token-band result is withdrawn: two protein arms swap order on the unit choice alone. Re-measured over five windows with the band *and* the corruption span declared in content symbols, the **level** contrast reverses — text is highest at short content distance — and what separates is the **decay rate**: across **seven** text controls the separation is clean at threshold 0.50 (worst text 0.397 below lowest protein 0.500, 136 of 140 paired comparisons) and **overlaps by 0.015 at 0.25**, where the off-distribution arm dialogpt-small reaches 0.611. Like the far-band level result, it lives in the large-effect tail. The matched pair separates on this estimand for the first time; the text decay ratio is flat across a 6.5x scale range where the protein one rises, so a larger protein arm would widen the gap rather than close it. Residual unit caveats in §5.1 | ~6 + ~6 GPU-h, spent |
| ~~**D1.b**~~ | Is the tail statement a *distribution* statement at full population? | **Answered (EXP-R2-073), CPU, no GPU spent.** No: it remains a tail statement, and the tail is now located — the ordering separates at every quantile from q75 to q99 and fails at the median and in the extreme upper tail. Pooled AUC 0.624, 20 of 28 pairs above 0.5, 2 of 28 showing stochastic dominance against the recorded 0 of 12. Two published figures moved with it: modality variance surviving projection 25.4% → **36.3%**, and the 2.34x scale-adjusted shortfall loses its interval verdict as an out-of-lineage extrapolation. Detail in §4 | 0 GPU-h, spent |

**Exit condition unchanged in spirit:** D1 exists to explain transfer failures, not
for its own sake. No new D1 measurement is authorised beyond these two.

### D1.c — Candidate arms and corpora, with the risk each carries

Reviewed 2026-07-30. Recorded here so the survey is not lost, and so that nothing
on it can be admitted without its stated risk being answered first. **None is in
the panel.** The panel grew by exactly one arm this session, ProGen2-small, and
only after a load check on the pod.

| candidate | what it would buy | why it is not admitted yet |
|---|---|---|
| **MolCrawl protein GPT-2-large** | The highest-value candidate: a *second* protein arm on GPT-2-large's 36x1280x20-head backbone, so the modality coefficient would stop resting on ProtGPT2 alone — the structural limit of §2 | Matched on backbone, **not** on parameter count (a small protein vocabulary shrinks the embedding and output layers); reported parameter counts disagree with the checkpoint; `bos_token_id`/`eos_token_id` disagree with the tokenizer's special symbols; no peer-reviewed validation; ProteinGym-finetuned variants must be excluded. "Both trained on UniRef50" does not fix release, filtering or compute budget. It would *reduce* the single-arm risk; it cannot on its own establish a causal modality difference |
| **RITA** (one medium checkpoint) | Tests whether the conclusions are specific to the GPT-2 / ProGen2 lineages | Untried here; admit one rung before four |
| **ProGen3-112M + ProGenMech** | The external baseline a D3 construction must replicate against, and the first sparse-MoE protein decoder in scope | Mandatory to replicate *before* D3, and must not be treated as already passing this programme's gates |
| **Tranception, retrieval off/on** | A clean D2 interface-attribution experiment: identical weights, external homology retrieval toggled | Not part of the modality measurement; separate design |
| **Pythia / PolyPythia multi-seed** | Seed variance and learning dynamics on the text side | Instrumental calibration only. It must **not** become extra text-side voting evidence — the text side already has seven arms against five protein |
| **Dayhoff UR50/UR90 variants** | Controlled data-mechanism study | Only if the evidence points at a training-data confound |
| **ProGen2-large** | Same lineage, same 31-token alphabet, a 1600x wider output head — a natural experiment on whether L8's rank-(V-1) aperture tracks the output *matrix* or the reachable *symbols* | Load-checked: 2779.4M parameters, works, but `vocab_size` 51200 against a 31-token tokenizer. Every `config.vocab_size`-derived statistic would be over a mostly dead alphabet. Deserves its own gated design; see `panel_contract.STAGED_BUT_NOT_ADMITTED` |
| **ProGen2-xlarge** | Would extend the protein ladder to 6.4B, a 43x range | Load-checked: 6443.6M parameters, forward pass returns width 32 — but the config carries **no** `vocab_size`, only `vocab_size_emb`/`vocab_size_lm_head`, and `budget.arm_power` reads `config.vocab_size`. Admissible once the panel reads a *declared* alphabet size rather than trusting a key two checkpoints in one lineage spell differently |

**Corpora, as evidence packages rather than more FASTA.** Four are worth building, in this order, and each carries a trap that has to be designed around rather than noticed afterwards. A *temporal* extrapolation set per model cutoff, from UniParc first-observation timestamps against a pinned Swiss-Prot release — but UniRef cluster IDs are rebuilt between releases and cannot be the identifier, and new appearance is not novelty without homology exclusion. A *PDB-CATH* set of experimental structures released after each cutoff, split out-of-family on CATH superfamilies with S20/S35 sensitivity beside the S40 already here, distinguishing an old sequence with a new structure from a genuinely new sequence. An *independent DMS* collection from late MaveDB assays absent from the model papers and from ProteinGym, grouped by wild type and homologous cluster — millions of mutant rows over one wild type are not independent units, which is the same error L1's cohort made at smaller scale. And a *repeat-mechanism* 2x2 (repeat present/absent x low/normal complexity) from RepeatsDB, Swiss-Prot repeat records and synthetic constructs, matched on length, taxonomy, domain architecture, compositional entropy and maximum training-set similarity — which is the only design that tests the corpus-repeat confound §4 calls irremovable, rather than restating it. MGnify is optional and cannot be a clean gold standard: it is largely predicted proteins and may overlap BFD.

**A standing caution this survey earns.** Predicted structures, pLDDT and sequence-inferred labels support stratification and covariate adjustment. They cannot demonstrate functional competence or "acquired protein knowledge", which is the second half of the programme's objective and currently has no instrument at all.

### D2 — Where the methods transfer, and whose fault it is when they do not

The deliverable. Twenty-three limitations are catalogued in §5, each scoped as
method, model, data or interface. What remains:

| item | pre-registered gate | cost |
|---|---|---|
| **D2.a** | Input-contract certification retro-applied to every quoted number; must reproduce the 1.78-nat rendering delta and the file-order delta as positive controls. TG-00 now does both and TG-01 has run, so this is partly discharged | ≤4 GPU-h |
| ~~**D2.b**~~ | Causal effect-size distributions on induction: top-20 Jaccard ≥ 0.8 against the census. **RUN, EXP-R2-068 — the gate is not answerable by this instrument.** `11_induction_path_patching.py` computes a causal effect only for heads the census already selected (verified: the per-head records are *exactly* the selected sender set, 57 / 14 / 6 / 8 heads). Three arms were selected by `prefix_matching >= threshold`; **ZymCTRL's 8 came from the `top_k_no_head_above_threshold` fallback with `n_above_threshold: 0`**, so it entered the causal stage with no head above the census cut at all, so a top-20 Jaccard against that same census is **1.0 by construction** on all four arms. The comparison is circular and its result carries no information. Answering it needs causal effects for heads the census did **not** select — the exhaustive causal census §8 item 1 already flags as requiring a new, independently gated design. Restated as D2.b′ below | 0.4 GPU-h, spent |
| ~~**D2.b′**~~ | Exhaustive per-head causal effect on induction, every head patched rather than every selected sender. **RUN (EXP-R2-071). The gate fails on every arm including the text control, and the threshold-free statistic separates the matched pair completely** | ~16 GPU-h, spent |

### EXP-R2-071 — the prefix-matching census is a causal proxy on text and is not one on protein

Four arms, every head patched (720 / 720 / 432 / 720), 64 path cases, all four sender × case conditions, 4 h wall-clock on four GPUs. This is the first measurement in the programme that can see the census's **misses**.

**(i) The all-head rank correlation separates the matched pair completely, and it is stable.** Spearman ρ between prefix-matching score and causal-effect magnitude over every head, range across the four conditions:

| arm | census selects | ρ range across 4 conditions |
|---|---:|---|
| gpt2 | 18/144 (12.50%) | **+0.657 to +0.706** |
| gpt2-large | 57/720 (7.92%) | **+0.428 to +0.507** |
| zymctrl | 0/720 (0.00%) | +0.216 to +0.271 |
| progen2-small | 5/192 (2.60%) | +0.128 to +0.194 |
| progen2-medium | 6/432 (1.39%) | +0.041 to +0.207 |
| protgpt2 | 14/720 (1.94%) | **−0.155 to −0.006** |

The text minimum (+0.428) exceeds every protein maximum (+0.271) in all 24 arm × condition cells.

> **Correction to the first reading of this table.** It was recorded here that the census "carries essentially no information about causal rank" on protein. **That is wrong.** The all-grid ρ is low on protein because the bulk of the grid is unordered, not because the top of it is: restricted to the 32 heads the census scores highest, ρ is **+0.808 on ProtGPT2** against +0.647 (gpt2) and +0.554 (gpt2-large), while over the remaining heads it collapses.
>
> **Corrected again on the nine-arm panel (EXP-R2-072), and one half of it does not generalise.** The top-32 statement was made on two text and two protein arms. Across the range of all four sender × case conditions on nine arms it reads:
>
> | arm | ρ over the top-32 census heads | ρ over the remaining heads |
> |---|---|---:|
> | gpt2 | +0.481 to +0.647 | +0.342 |
> | gpt2-medium | +0.554 to +0.679 | +0.477 |
> | gpt2-large | +0.554 to +0.670 | +0.350 |
> | dialogpt-small | +0.477 to +0.703 | +0.268 |
> | protgpt2 | **+0.792 to +0.824** | **−0.223** |
> | progen2-base | +0.584 to +0.686 | +0.094 |
> | progen2-medium | +0.551 to +0.790 | +0.088 |
> | progen2-small | +0.417 to +0.496 | +0.133 |
> | zymctrl | −0.100 to +0.225 | +0.259 |
>
> **"Protein arms' top heads are ranked better than text's" survives only for ProtGPT2.** ProGen2-base and ProGen2-medium overlap the text range; ProGen2-small sits below every text arm and ZymCTRL is near zero. The *second* half survives and is the load-bearing one: **the remaining-head correlation separates the modalities** — text +0.268 to +0.477 against protein −0.223 to +0.259 — so the census orders the bulk of a text grid and does not order the bulk of a protein grid, which is what makes a head *count* a bad summary of a protein arm.
>
> **Two of the four originally published remaining-head figures were not reproducible and are replaced above.** The recorded +0.128 for gpt2 re-derives as **+0.342** from the cited artefact, and +0.275 for gpt2-large as **+0.350**; ProtGPT2's −0.265 and ProGen2-medium's +0.089 re-derive within 0.04. The cause is Appendix B rule 12 in its sharpest form: these statistics had **no implementation in the repository** and were computed by unversioned throwaway code, so the split that produced them cannot be recovered. They now live in `path_patching.py` with tests, and the direction of the error was conservative — the true text bulk correlation is *higher*, so the contrast is larger than was reported.

**(ii) The structural difference underneath is concentration, and it is head-count free.** ~~*Suspended pending re-derivation — see EXP-R2-072 (v). … Do not cite this table or its qualification until the noise-corrected variant lands.*~~ **Un-suspended (EXP-R2-081, 2026-08-01).** The suspension named a "noise-corrected variant" that was never implemented and never will be — see the correction at (v). It has been replaced by a measurement of the quantity it was worried about. These statistics are head-count free but not **case**-count free, and they sum \|effect\| over a grid whose majority is individually indistinguishable from zero; the question was how much of that integrated mass is estimation noise, and whether its size differs across arms. It does not, on exact probes: probe-clustered reliability of the per-head \|effect\| is 0.861–0.980 on text against 0.823–0.934 on protein, so **the table below is interpretable as it stands and the ZymCTRL exception is real.** The *approximate*-probe qualification stays withdrawn, because there protein reliability falls to 0.466–0.706 and concentration falls with it, and these artefacts cannot separate the two. Every figure in the table re-derives exactly from the retained artefacts under the versioned implementation. Effect concentration across the head grid, using statistics that do not depend on how many heads an arm has:

| arm | Gini of \|effect\| | share of grid carrying half the total effect |
|---|---:|---:|
| gpt2 (text) | 0.829 | 4.9% |
| gpt2-large (text) | 0.826 | 3.2% |
| progen2-small | **0.941** | **1.6%** |
| protgpt2 | **0.940** | **1.0%** |
| progen2-medium | **0.944** | **0.7%** |
| zymctrl | 0.612 | 10.6% |

Three protein arms sit at Gini 0.940–0.944 against 0.826–0.829 for both text arms, with half the causal effect carried by 0.7–1.6% of heads against 3.2–4.9%. **Non-overlapping.** This is the same shape §5.1 reports in the distance dimension — text spreads many small effects, protein concentrates fewer large ones — now measured in the head dimension on an independent estimand. Two measurements, one structure.

> **Qualification, and it is a serious one: the contrast is specific to the exact-repeat case set.** The table above is computed on `cases_exact`. Recomputing every arm's Gini on `cases_approximate` — the same heads, the same run, the other case set — gives gpt2 0.759, gpt2-large 0.752, ProGen2-small 0.772, ProtGPT2 0.822, ProGen2-medium **0.754**, ZymCTRL 0.564. **The separation largely collapses**: ProGen2-medium falls *below* gpt2, ProGen2-small is within 0.02 of it, and only ProtGPT2 remains clearly above. So the honest statement is not "protein concentrates causal effect" but **"protein concentrates causal effect on exact-repeat probes, and the contrast does not survive the approximate-repeat probes"** — which makes it a statement about a probe regime as much as about a modality, and removes the clean "two measurements, one structure" reading of its relation to §5.1. Recorded before the wider panel lands rather than after, because it was found by a check that cost nothing and would have been easy not to run.

**ZymCTRL is the exception and runs the other way** (Gini 0.612, 10.6% of heads for half the effect — the least concentrated arm on the panel). It is also the one arm with a conditioning prompt, so this is a rendering-and-conditioning contrast rather than a modality one, and it is flagged rather than absorbed: any statement of the form "protein concentrates" is false of ZymCTRL as rendered.

**(iii) Noise is not the explanation, and this was tested rather than assumed.** Per-head effect estimates carry SEMs, so the reliability of each arm's head ranking is computable: **0.916 to 0.991** across the six arms, and the SNR of the mean effect is 1.26–6.15. Disattenuating the Pearson correlation for measurement error moves it by at most 0.04 on any arm. The low protein ρ is not attenuation from noisier estimates.

> **Corrected (EXP-R2-072): that range is computed on the sampling unit this module declares to be the wrong one.** `path_patching.py` publishes both a case-level `sem` and a `sem_probe_clustered`, and states in two places that the probe record, not the case, is the sampling unit. The 0.916–0.991 figures reproduce exactly, from the **case**-level SEM on effect *magnitude*. Recomputed by the versioned implementation on the probe-clustered SEM the same statistic reads **0.827–0.975** on the exact-repeat cases, and on the approximate ones ZymCTRL falls to **0.008** — a grid that cannot be ranked at all. **The inference (iii) draws still holds where it matters:** on approximate cases gpt2-large and ProtGPT2 have near-equal probe-clustered reliability while their ρ are far apart, so differential attenuation does not explain the matched-pair gap. But reliability is a *variance* statistic dominated by the few large heads, so a value near 1 was never capable of establishing that the median head is resolved — which is the separate problem EXP-R2-072 (v) records against the concentration statistics. Both units and both scales are now emitted, each labelled, so the choice is visible in the artefact rather than made in an analysis script.
>
> > **Corrected again (EXP-R2-078): the 0.008 is not a reliability at all, and the paired figure is 0.170.** The correction above changed the standard error and left the *centre* alone. `head_effect_reliability` read the observed variance from the case-weighted `mean` under both sampling units, so the probe-clustered figure divided a probe-unit error variance by a case-unit observed variance — two different estimators. `sender_recoveries` states the pairing rule at the point it builds the record: `mean` weights each probe by how many cases it contributed, and `sem_probe_clustered` describes `mean_probe_clustered`, which weights probes equally. Recomputed with each standard error against the centre it belongs to, on the same artefacts:
> >
> > | arm / condition | as published | correctly paired |
> > |---|---:|---:|
> > | ZymCTRL, approximate cases, magnitude | **0.008** | **0.170** |
> > | ZymCTRL, approximate cases, signed | 0.189 | **0.443** |
> > | gpt2-xl, approximate cases, signed | 0.659 | 0.706 |
> > | gpt2-large, approximate cases, signed | 0.779 | 0.807 |
> > | every exact-case condition, ten arms | — | moves by < 0.006 |
> >
> > **What this changes.** "A grid that cannot be ranked at all" is withdrawn as stated: 0.170 is very low — 83% of the head-to-head spread on that condition is estimation noise — but it is twenty times the retracted figure and it is not zero. The exact-repeat range is unaffected, so **nothing that rests on the exact-case reliability moves**, including the (iii) inference that differential attenuation does not explain the matched-pair gap. The mismatch was invisible because the test fixture omitted `mean_probe_clustered` entirely and the two retracted values were pinned as expectations. They are now pinned as *retracted* values, reconstructed in the test from the artefact, so a regression to them fails loudly.

**(iv) The census's failure mode is recall.** ProGen2-medium's top-5 Jaccard is **1.000** and collapses to 0.250 by k=20. The strongest head the census *rejects* carries \|effect\| 0.0194 on gpt2-large against 0.0497 (ProtGPT2), 0.0290 (ProGen2-medium) and 0.3049 on ZymCTRL, where the census selects nothing at all.

> **The cross-arm reading of those four numbers is withdrawn (EXP-R2-072, verified against the artefacts).** A per-head effect is normalised by that arm's own clean-minus-corrupt denominator, and the denominators span **32x** across the panel. Restated in logits:
>
> | arm | eligible | mean denominator (logits) | rejected head, normalised | rejected head, **logits** |
> |---|---:|---:|---:|---:|
> | gpt2 | 63/64 | 24.68 | 0.0337 | **0.832** |
> | gpt2-medium | 63/64 | 24.69 | 0.0158 | 0.391 |
> | gpt2-large | 63/64 | 24.10 | 0.0194 | 0.469 |
> | dialogpt-small | 36/64 | 12.41 | 0.0773 | 0.959 |
> | protgpt2 | 64/64 | 20.68 | 0.0497 | **1.027** |
> | progen2-base | 61/64 | 9.22 | 0.1179 | **1.087** |
> | progen2-small | 59/64 | 8.12 | 0.0596 | 0.484 |
> | progen2-medium | 62/64 | 9.36 | 0.0290 | 0.271 |
> | **zymctrl** | **24/64** | **0.76** | **0.3049** | **0.233** |
>
> ZymCTRL's headline 0.3049 — quoted as 15.7x gpt2-large's — is **0.233 logits, the smallest of the nine**. The sentence's force came entirely from a denominator a thirtieth the size of the text control's, itself a consequence of ZymCTRL clearing the eligibility floor on only 24 of 64 cases. **What survives is the within-arm statement**, which is the one that matters for the census: on ZymCTRL the census selects no head at all while heads with real causal effect exist, and the top-k Jaccard collapse with k is a within-arm quantity. Appendix B rule 27 now travels with any cross-arm effect quote.

**(v) What this does and does not say about §4's head-count gap.** It does **not** show that prefix matching is a broken selector on protein — at the top of the grid it is a better one there than on text. It shows that a **count** of heads above a fixed prefix-matching threshold is a poor summary of protein arms specifically, because their causal importance is concentrated in far fewer heads and the threshold sits in a region where the score no longer orders anything. A head count is a statistic about the bulk; protein causal structure is not in the bulk.

**(vi) The pre-registered gate is a specification defect, the second this session.** Top-20 Jaccard ≥ 0.8 is met by **no arm in any condition**; the best observed is 0.667 and the text control never exceeds 0.538. Appendix B rule 2 applies. Withdrawn as stated.

*Limits.* The causal effect is path patching on repeat probes, so a high-effect head is "causally important for repeat prediction", not "an induction head by mechanism". Effects are per-head means over 64 cases and ranking noise depresses Jaccard on every arm alike, which is exactly why the text control is the reference rather than an absolute cut. ProtGPT2's negative ρ values are small and marginal (p ≈ 0.04 at n = 720) and should be read as *uninformative*, not as anti-correlation. Artefacts: `results/transfer_20260730/d2bprime/`.

**D2.b′'s instrument, and why the precondition is enforced rather than documented.** `select_senders` gained an `exhaustive` criterion that admits every head, and `causal_census_agreement` computes the comparison the gate asks for. That function **raises** on a non-exhaustive sender set. The reason is that D2.b's failure is invisible in its own output: a top-20 Jaccard of 1.0 is exactly what a genuine agreement would also produce, so nothing in the artefact distinguished "the two rankings agree" from "the two rankings ranked the same six heads". A comment would not have stopped the next run from reproducing it. Two further points of the design: `above_threshold` became each head's own score against the threshold rather than the set-level answer broadcast to every head — identical on both pre-existing criteria, and the only truthful answer on a set that spans the threshold, which is precisely the field that says which causally-ranked heads the census missed. And the primary statistic is the **rank correlation over all heads**, which needs no cut at all; the top-20 Jaccard the gate was written in is reported beside it and swept over k = 5/10/20/32/40, per Appendix B rule 17.

> *Validated end-to-end on GPT-2, and the instrument is not returning the circular answer.* At a deliberately small validation size — 8 cases, 144 heads, 90 s on one L20 — the census ranking and the causal ranking give Spearman **ρ = +0.70** (p = 7×10⁻²³) and a **top-20 Jaccard of 0.538**, against the 1.0 the selective sender set returned by construction. Six of the causal top-20 are heads the census scores *below* threshold, the strongest being L8H9 at prefix-matching 0.018 but causal rank 8. **These are not results and must not be quoted as any:** 8 cases per condition is far too few for a per-head effect estimate, and ranking noise depresses Jaccard on its own. What they establish is that the gate is now answerable — the statistic can fall below 1.0, and census misses can appear in it, neither of which was possible before. The measurement itself is queued behind EXP-R2-070 on four arms.
### EXP-R2-072 — the off-distribution text control does not collapse, so the census result is not distribution mismatch

The exhaustive census was extended from four arms to nine (gpt2-xl still running), at the identical `--exhaustive-senders --repeat-cohort-size 48` configuration, so all nine combine without a case-count confound. The cohort digests are byte-identical across arms within a modality, which is what licenses pooling them.

**(i) `dialogpt-small` is the control this measurement most needed, and it answers the alternative explanation.** §5.05(a) shows dialogpt-small is off-distribution on the evaluation corpus at −4.08 nats of context information — a *text* decoder that is unmeasurable the way protein arms are often suspected of being. If the protein arms' near-zero census-to-causal correlation were distribution mismatch rather than modality, dialogpt-small should collapse with them. **It does not:** ρ **+0.518 to +0.640**, squarely inside the text range, and *above* gpt2-large. Being off-distribution on the scoring corpus does not destroy the census's causal ordering; being a protein decoder does.

**(ii) The separation is complete on ten arms.** All-head ρ between prefix-matching score and causal-effect magnitude, range across the four sender × case conditions:

| arm | modality | params | heads | census selects | ρ range |
|---|---|---:|---:|---:|---|
| gpt2 | text | 124M | 144 | 18 (12.50%) | +0.657 to +0.706 |
| gpt2-medium | text | 355M | 384 | 44 (11.46%) | +0.552 to +0.643 |
| dialogpt-small | text | 124M | 144 | 7 (4.86%) | +0.518 to +0.640 |
| gpt2-large | text | 774M | 720 | 57 (7.92%) | +0.428 to +0.507 |
| gpt2-xl | text | 1.56B | 1200 | 78 (6.50%) | **+0.371 to +0.458** |
| zymctrl | protein | 710M | 720 | 0 (0.00%) | **+0.216 to +0.271** |
| progen2-medium | protein | 765M | 432 | 6 (1.39%) | +0.041 to +0.207 |
| progen2-small | protein | 151M | 192 | 5 (2.60%) | +0.128 to +0.194 |
| progen2-base | protein | 765M | 432 | 3 (0.69%) | +0.050 to +0.176 |
| protgpt2 | protein | 774M | 720 | 14 (1.94%) | −0.155 to −0.006 |

The text minimum (+0.371) exceeds every protein maximum (+0.271) in all **40** arm × condition cells, on five text and five protein arms, two of which — gpt2-large and ProtGPT2 — are the matched pair. gpt2-xl's artefact was orphaned by a transport drop and its controller correctly refused to report it complete; it was admitted only after both files were verified against the SHA-256 manifest the worker wrote beside them.

**(iii) The text side carries a monotone scale effect over four rungs, and it runs against the finding rather than producing it.** ρ falls **0.657 → 0.575 → 0.428 → 0.371** across 124M / 355M / 774M / 1.56B, roughly −0.26 per decade. Two consequences, and the first is the one that matters. Extending the ladder puts the panel's *lowest* text correlation at gpt2-xl, so the separation is now measured against the strongest available text counter-example: the margin narrows from 0.16 to **0.10** and does not close. And scale supplies no alternative account — ProGen2-small (151M, 12 layers) reads +0.128 to +0.194 against gpt2 (124M, 12 layers) at +0.657 to +0.706, matched depth and matched scale for a fivefold gap. A pure scale story needs the protein arms on the text ladder's line; at 151M that line predicts ≈+0.65 and the arm reads +0.16. **What the ladder does establish is that this statistic is not scale-free**, so a future protein arm an order of magnitude larger than ProGen2-medium is the measurement that could still overturn it.

**(iv) A second seed moves the statistic by less than the modality gap — but it is a weaker check than its name suggests.** Re-running gpt2-large and ProtGPT2 at `--seed 20260731` gives +0.461 to +0.538 (from +0.428 to +0.507) and −0.180 to −0.055 (from −0.155 to −0.006): both shift by ~0.03 against a gap of ~0.6. **However, the repeat-cohort digests are byte-identical across the two seeds**, because `--seed` governs case sampling and the bootstrap while the corpus draw is governed by a separate `--cohort-draw-seed` (`arms.DEFAULT_CORPUS_DRAW_SEED`). So these are **case-resampling** robustness runs, not cohort-draw robustness runs. Given §5.05(b) and Appendix B rule 22 — cohort sensitivity is a property of the statistic and protein arms carry more of it — a cohort-draw check on this statistic is still owed.

**(v) The concentration statistics of EXP-R2-071 (ii) are under re-derivation and should not be quoted meanwhile.** Adding ProGen2-base at Gini 0.958 (highest on the panel) and dialogpt-small at 0.769 preserves the recorded shape on exact-repeat probes, but the statistic sums \|effect\| over the whole head grid, and more than half of each arm's heads are individually indistinguishable from zero — so the Gini is partly integrating estimation noise whose mass is not matched across arms with unequal eligible-case counts. Both the ZymCTRL exception and the "collapses on approximate probes" qualification are decided in that unresolved bulk. They are suspended, not withdrawn, ~~pending the noise-corrected variant now implemented in `path_patching.py`~~.

> **Correction and resolution (EXP-R2-081, 2026-08-01).** The struck clause was **false**: no noise-corrected concentration variant is implemented in `path_patching.py` or anywhere else in the repository. `effect_concentration` (`path_patching.py:2173`) computes `gini` on raw \|effect\| with no correction. What EXP-R2-078 implemented was `head_effect_reliability`, a *different* statistic — the signal fraction of the observed cross-head variance, not a corrected Gini. As written, this item suspended a result pending a fix that did not exist.
>
> That statistic nevertheless decides item (v), by measuring the very mass the item calls unresolved. Probe-clustered `reliability_magnitude_ranking` over the exhaustive conditions: **exact probes — text 0.861–0.980, protein 0.823–0.934; approximate probes — text 0.745–0.865, protein 0.466–0.706.**
>
> - **On exact probes the noise mass *is* matched across modalities, so concentration is interpretable there, and the ZymCTRL exception is real** — its Gini of **0.607** against **0.78–0.94** across the arms measured in the same alternate draw stands at reliability 0.823 and cannot be deflated into agreement with them. *Draw and arm set stated because they matter:* 0.607 is the exact-probe total-effect Gini on the 2026-08-01 draw; the reference draw gives **0.612**, which is the figure in the (ii) table, and across the full twelve-arm reference panel the other arms span **0.769** (dialogpt-small) to **0.958** (ProGen2-base). The exception holds on either draw and against either comparison set.
> - **On approximate probes it is not matched.** ZymCTRL reaches 0.466, i.e. over half its observed cross-head variance is estimation noise. The "collapses on approximate probes" qualification is therefore confounded with an estimation-noise collapse on the protein side and **stays withdrawn**: concentration does fall on approximate probes for every arm, but on protein the reliability falls with it and these artefacts cannot separate the two.

*Limits.* Same as EXP-R2-071: path patching on repeat probes measures causal importance for repeat prediction, not induction by mechanism; 64 cases per condition. ~~Every text arm on this panel is GPT-2 architecture, so a cross-lab text arm would strengthen (ii) — `induction_path_patching` refuses `qwen2` and `llama` because `SUPPORTED_ARCHITECTURES` has no module layout for them, which is a recorded instrument limit and a candidate extension, not an omission.~~ **Both limits are discharged below: the instrument now resolves the llama and qwen2 layouts (EXP-R2-078) and two rotary/GQA text arms have been measured (EXP-R2-079).** Artefacts: `results/transfer_20260730/d2bprime_ext/`, `results/transfer_20260730/d2bprime_seed2/`.

> **Both owed checks are now in flight (2026-08-01). Neither result is in this document yet; do not anticipate them.**
>
> **The cohort-draw check (iv) asks for — EXP-R2-077.** Four arms at the identical configuration under a different corpus draw: gpt2-xl (the text minimum, the arm whose fall would close the gap from above), ZymCTRL (the protein maximum, whose rise would close it from below), and the matched pair. One alternate draw, not two: at K = 2 a shifted arm cannot be told from a noisy one, so this is posed as an ordering question rather than a variance estimate. **Part of the statistic is immune to the draw by construction and that is worth stating**: Swiss-Prot yields exactly 48 exact-repeat matching records of 203,063 eligible against a request of 48, so the protein exact-repeat cohort is a corpus *census* and its identity cannot change under any seed. The check therefore bites on the approximate conditions and on the text arms.
>
> *A protein-specific fragility surfaced before any GPU time was spent.* ProtGPT2 reaches only **58 of 64** approximate-repeat path cases under draw 20260801 and **57** under 20260803, because a multi-residue BPE puts the repeat's key token outside the unigram support fit on the analysis cohort; ZymCTRL, residue-level, reaches 64 on every draw tested. The entry point refuses to run short, so the arm ran at 20260802 instead and the rejected draws are recorded. **This means ProtGPT2's 64-case parity in EXP-R2-071/072 was itself draw-contingent**, on roughly half the draws tried, and nothing said so — a new instance of the protein-side cohort sensitivity of §5.05(b), reaching the case set rather than the estimate.
>
> **The cross-lab control this limits paragraph asks for — EXP-R2-079.** The instrument limit is lifted: `path_patching` now resolves the llama and qwen2 layouts, validated end to end on a Qwen2.5-0.5B checkpoint at 336 exhaustive senders with head-write linearity at 2.2e-06 relative error under grouped-query attention (EXP-R2-078). qwen2.5-0.5b and llama-3.2-3b run at the identical configuration. **Pre-registered, because the scale trend makes one arm ambiguous:** extrapolating (iii)'s −0.26 per decade predicts ≈+0.60 for qwen (494M) and **≈+0.29 for llama (3.21B) — within reach of the protein maximum of +0.271**. A llama value near 0.29 is therefore what the scale trend predicts and is *not* evidence against L22; a value near zero or negative, like ProtGPT2's, would be. Written down before the run, because reading a narrow margin either way afterwards is the failure Appendix B rule 2 keeps finding.

**Both checks are in (2026-08-01). Results below; the pre-registration above is left standing so the miss is visible.**

**EXP-R2-077 — the cohort draw.** Case seed held fixed so only the corpus draw moves. All four cells per arm, reference → alternate draw:

| arm | exact/exact | exact/approx | approx/exact | approx/approx |
|---|---:|---:|---:|---:|
| gpt2-large | +0.4276 → **+0.5350** | +0.4402 → **+0.5034** | +0.4725 → **+0.5089** | +0.5071 → **+0.5217** |
| gpt2-xl | +0.4087 → **+0.4448** | +0.3711 → **+0.4028** | +0.4576 → **+0.4261** | +0.4404 → **+0.4150** |
| zymctrl | +0.2706 → **+0.2823** | +0.2197 → **+0.2468** | +0.2490 → **+0.2098** | +0.2160 → **+0.2269** |
| protgpt2 | −0.0763 → **−0.1513** | −0.1554 → **−0.2259** | −0.0058 → **−0.0547** | −0.0537 → **−0.0610** |

The ordering survives, and it survives in the informative way: **the matched pair moved in opposite directions** — gpt2-large up in all four cells, ProtGPT2 down in all four — so this is not a common-mode shift of the panel. Text minimum +0.3711 → +0.4028, protein maximum +0.2706 → +0.2823, gap **+0.1005 → +0.1205**. The pre-stated census property held in the artefact: the protein exact-repeat cohort digest is byte-identical across draws (`3ea68a87c593c532`).

**EXP-R2-079 — the cross-lab control, and a miscalibrated pre-registration.** Exhaustive over every head; all nine structural invariants pass on both arms — eight quantitative checks at ≤2.5e-06 against a 1e-03 tolerance, worst case `head_write_linearity`, plus the `freeze_only_perturbed` positive control, which is required to *move* the metric rather than to leave it unchanged.

| arm | arch | grid | exact/exact | exact/approx | approx/exact | approx/approx | pre-registered |
|---|---|---:|---:|---:|---:|---:|---:|
| llama-3.2-3b | llama | 672 | +0.5034 | +0.4794 | +0.5046 | +0.5038 | ≈ +0.29 |
| qwen2.5-0.5b | qwen2 | 336 | +0.5749 | +0.5267 | +0.5746 | +0.5555 | ≈ +0.60 |

qwen landed near prediction. **llama did not — predicted ≈+0.29 "within reach of the protein maximum", observed +0.4794 to +0.5046, high by ≈+0.19.** The scale-trend extrapolation under-predicts this lineage. The miss is in the direction that strengthens L22, which is precisely when it is tempting to leave unstated, so it is stated: the −0.26-per-decade trend of (iii) does not transfer across lineages and must not be used to pre-register a cross-lab arm again.

**L22 is now architecture-controlled.** Its 40 cells had every text arm on GPT-2 architecture, so modality was confounded with lineage — and ProtGPT2 *being* a GPT-2 made that the obvious attack. At 48 cells over twelve arms the confound is cut both ways: two text lineages from different labs, with different tokenizers, position encodings and attention layouts (llama, qwen: **+0.4794 … +0.5749**) sit above the protein ceiling, while the arm sharing ProtGPT2's exact architecture sits furthest above it. All-text minimum **+0.4028** against protein maximum **+0.2823**, separated in every cell. **The separation tracks modality, not architecture.**

*The limit that is now binding.* Largest single-cell movement between draws **0.1073**; mean absolute movement over sixteen cells **0.0398**; closest modality approach **0.1205**. One cell moved almost as far as the whole gap, so at K = 2 the gap cannot yet be quoted against a spread — only as a surviving ordering. **EXP-R2-080** (launched 2026-08-01, ≈44 GPU-h) takes the four gap-defining arms to K = 4 (K = 3 for gpt2-xl, which costs 11 GPU-h per draw — a budget asymmetry, recorded not smoothed) and gives the eight remaining arms their first alternate draw, so all 48 cells contribute.

**EXP-R2-081 — the attenuation objection, tested and survived.** The most serious objection available to L22 is that protein \|effect\| estimates are noisier, so the rank correlation is attenuated harder on protein — *toward zero, the direction of the hypothesis*. It is the failure `corruption_effects` had to fix for the matching gate ("a conclusion the estimator manufactured out of alphabet size"), and it had not been tested. **The objection has a real basis:** probe-clustered reliability of the per-head \|effect\| is 0.823–0.934 on protein against 0.861–0.980 on text for exact probes, and **0.466–0.706 against 0.745–0.865** for approximate probes — ZymCTRL's approximate conditions are over half estimation noise. **The separation survives correction in all 24 measured cells.** Under classical disattenuation r → r∕√reliability the gap widens in two conditions and narrows in the two carried by ZymCTRL's noisy approximate cells, worst case exact/approx at **+0.1560 → +0.0946**; pooled over all 24 cells, text minimum +0.4560 against protein maximum +0.3613. Correcting for the differential noise does not close the gap. *Limits:* only the causal side is disattenuated (census-score reliability is unmeasured, so these are lower bounds, and the same one-sided assumption is applied to both modalities); the formula is derived for Pearson and applied to Spearman; and the reliability used is a variance ratio for the per-head means, not literally the rank variable's. A sensitivity analysis, not an exact correction — decisive only because the conclusion does not depend on the correction's size.

| **D2.c** | Same on copy suppression — does effect size separate arms where prevalence cannot? Directly answers the measurability question §1 leaves open. **All blockers cleared and the attainability gate passed (EXP-R2-078, EXP-R2-082, EXP-R2-087): the matched pair enters at `--width 192` with the cohort band unchanged, at zero L13 exposure.** The gate was re-specified before it was read — the +0.53 to +0.65 target is restricted-range, unmatched-score and reproducible only to ±0.12 at fixed width — and passed against gpt2-large's own induction band: **+0.4515 [+0.401, +0.498]** exhaustive at width 192, inside +0.428 to +0.535. ZymCTRL is irreducibly excluded from any shared window. See the D2.c section below | **1–2 GPU-h** for the check; panel cheaper than the 12–25 previously carried, since width 192 is ~2.7x cheaper per forward |
| **D2.d** | Complete the TG campaign: six of eleven measured stages have no artefact in the corrected tree (`missing_matrix`: tg02, tg04, tg05, tg06, tg08, tg10; 18 of 35 required artefacts present), so `SUMMARY.json` can only be produced in partial mode | ~6 GPU-h |

### D2.c — the design pass, and the three things that have to be built first

> **Status (2026-08-01): all three are cleared; one attainability check stands between this and the panel.** Blocker 3 was cleared in EXP-R2-078 (the positive control exists on the magnitude scale), blocker 2 closed in code in the same pass, and blocker 1 was re-measured in EXP-R2-082 — where the prescription written below turned out to be **wrong**, and the fix is the pool width rather than the cohort band. The section is left standing with its errors struck and corrected in place, because the reasoning that produced the wrong prescription is the point. Read the blocker-1 correction block and the revised order of work at the end.

Costed at "~13 GPU-h" while it was one line in a table. A design pass
(EXP-R2-078) put that at **12–25 H200-hours** for a ten-arm panel and found that
the measurement as imagined cannot be run at all on the two arms that matter
most. Recorded here before any compute is scheduled, because the plan's own
history is that a gate written into a table and costed by analogy is how
Appendix B rule 2 keeps being violated.

**What D2.c is.** L22 says a head-prevalence census's selector orders causal
importance on text decoders and not on protein decoders, measured on *one*
mechanism. §8 item 0 is gated on replicating that on a second mechanism, copy
suppression: score every head with `paa_specific`, measure every head's causal
effect by knockout, and compare the two rankings — the same design as D2.b′, a
different selector and a different metric.

**Blocker 1 — the matched pair cannot enter the estimand.** Verified directly
against the tokenisers, not inferred:

| arm | tokens per residue | residues needed for a 512-token row | admitted at width 512, band 520–800 |
|---|---:|---:|---|
| progen2-medium | 1.001 | ~511 | yes |
| zymctrl | 1.016 | ~504 | **only at exactly 502 residues** |
| protgpt2 | 0.349 | **~1468** | **none** — but **10037 instances at width 192** (EXP-R2-088) |

ProtGPT2's multi-residue BPE means no row inside the cohort band reaches the
pool's 512-token width, so `tokenised_rows` refuses the arm outright. ZymCTRL is
worse than a band problem: its conditioned rendering is
`{ec}<sep><start>{seq}<end>`, `content_bounds` requires exactly one `<end>` inside
the window, and the two conditions intersect at the single residue length
`width − 10`. **So D2.c as specified would run on the GPT-2 lineage against the
ProGen2 lineage and would not include the matched pair** — the only
modality-identifying comparison this panel has (§2). A cross-lineage-only result
on the second mechanism cannot discharge a gate about the first. ~~Fixing it means
raising `--protein-max-len` past ~1500 for ProtGPT2, which moves the cohort band
and is an L13 exposure that must be declared and measured~~, and giving ZymCTRL a
window co-designed with its conditioning prompt — the same wall L15 put in front
of the far-band estimand.

> **Blocker 1 re-measured (EXP-R2-082, 2026-08-01, CPU on B, no GPU). The prescription above was wrong, and the matched pair is available at zero L13 exposure.** Four corrections and one replacement design.
>
> **(1) Raising `--protein-max-len` does not work, because it is not the binding parameter.** The census band is `[--census-protein-min-len, --protein-max-len]` and the *floor* binds. Admitted rows out of 400 at width 512: band `[520, 1500]` → 7; `[520, 2000]` → 17; **`[520, 4000]` → 36**, still below `--min-sequences 64`, so the arm keeps failing. Full admission needs the **floor** raised to ~1550 (`[1550, 4000]` → 400/400). The struck clause would not have fixed the arm.
>
> **(2) The refusal is width-conditional, not unconditional — and this is the fix.** In the **unchanged** band 520–800, ProtGPT2 admits **400/400 at width 128, 320–355/400 at width 192, 65/400 at width 256, and 0 at width ≥ 320**. (Two independent measurements: a direct tokenisation of the declared `fasta_wrapped` rendering and a run through `tokenised_rows` itself agree exactly at 256 and at ≥ 320 and differ by ~9% at 192, from cohort-construction filtering. Tokens per residue 0.352 against the table's band-restricted 0.351.) **Lowering the pool width admits the matched pair without touching the band at all.**
>
> **(3) ZymCTRL's "1.016 tokens per residue" is a band artefact, not a tokeniser property.** Its rendering is exactly `R + 10` tokens on 5000/5000 records — a *constant* prefix, so the "rate" is `1 + 10/R`. Reporting it as a rate disguises that the constraint is exact: `tokenised_rows` requires `total == width`, so the admissible length is the single point `R = width − 10`, which the same paragraph then states correctly. The EC prefix is 9 tokens for 212,144 of 212,165 eligible records and 10 for 21 of them (a three-digit EC field splits), giving **two disjoint points, not a range** — and the two cannot be mixed, since `tokenised_rows` rejects inconsistent content offsets.
>
> **(4) Not previously recorded: the single-length ZymCTRL window breaks `build_cohorts`.** `14_paa_census.py` draws its reference corpus from the *same* band with a skip, and no exact residue length can supply it — the largest single length holds 959 records against a request of 4000. A trial run died on exactly this at a request of 600. Eighteen further lengths are poisoned by the 10-token EC tags. So the ZymCTRL window is not merely narrow; as specified it does not run.
>
> **The replacement design — width, not band.** `--width 192` with the cohort band **unchanged** at 520–800: gpt2-large admits 400/400 (attainability on the control, rule 2), ProtGPT2 320–355/400. **L13 exposure is zero** — identical band, identical eligible stratum — so a D2.c result can actually be read against L22, which sits in the same 200–800 protein regime. Forwards are ~2.7× cheaper than at width 512. And blocker 2 nearly vanishes on the pair that matters: median key set 1 on ProtGPT2 against 2 on gpt2-large, so the residual mismatch runs **against** the hypothesis rather than for it, unlike the cross-lineage fallback's 17 against 3.
>
> **Measured, and it holds — EXP-R2-088.** ProtGPT2 builds **10037** instances at width 192 (the width-512 reading was *none*), and A2 passes for both protein arms at both ban depths against a gate of 2000: ProtGPT2 **11263** projected at ban 20 and **11545** at ban 3, ProGen2-medium **5318** and **11941**. gpt2-large's own key set is **2.86 keys per instance at width 192** against 5.52 at width 512, and the cheap selector agrees far better with the knockout-matched score in the narrow window — partial Spearman **0.580** against **0.390** — which is a second, independent reason the width route helps that the design did not claim. **The matched pair needs no declared ban-depth asymmetry**: at vocabulary 50257 the specified depth 20 costs gpt2-large and ProtGPT2 the same nothing. *One new constraint, from the pool rather than from assumption:* ProtGPT2 yields ~50 instances per sequence, so at the census default of 200 sequences it returns 10037 and would read **FAIL** against `--a1-minimum 20000`; D2.c needs a cohort request nearer 600 records, which at width 192 is cheap.
>
> *Three costs, declared not hidden.* ~~The blocker-3 positive control (+0.53 to +0.65) is a width-512 measurement and **must be re-established at width 192 before anything is applied** — if it fails there, this design is dead.~~ **Re-established and passed at width 192 (EXP-R2-087), but not against that number**, which is a restricted-range *unmatched*-score figure whose own reproducibility at fixed width is ±0.12; the exhaustive matched reading is +0.4515 [+0.401, +0.498] against gpt2-large's induction band of +0.428 to +0.535. Two of eight distance bins become unreachable at width 192, so the estimand no longer probes long-range PAA; symmetric across the pair, but it narrows the claim. And the width filter selects ProtGPT2 records on **BPE compressibility** (+1.20 sd of tokens-per-residue at width 192, raising mean cohort length 628 → 643) while rejecting no text document at any width — a selection that exists only on the protein side because only there does a token stop being a residue.
>
> **What the band route would have cost, now measured.** Moving to 1500–4000 shares **not one record** with the band L22's protein arms were measured on, shrinks the eligible stratum 12.6×, raises internal 3-mer repeat content 2.7× — the very property both mechanisms are defined on — and moves context information by **−1.75 nats at a common window, −2.31 under the gate0 formula**, against L13's catalogued 1.01. *The prescription in the struck clause would have cost 1.7–2.3× the incident that created L13.* (Validation scale, 96/1500 sequences; direction and magnitude secure, third decimal not.)
>
> **ZymCTRL remains irreducibly outside any shared window.** ProtGPT2 needs `t/r ≥ width/(width − 10) > 1` and never exceeds 0.40; `width − 10 ≥ 2.5·width` has no positive solution, and 0 of 581 records at R = 502 reach 512 ProtGPT2 tokens. **No width, at any cohort size, admits both.** ZymCTRL is therefore a separately declared per-arm configuration (`--width 348`, band 338–338, 400/400 admitted) or it is not run — and either way it cannot contribute to a statement about a common cohort.

**Blocker 2 — the census and the causal statistic are measured on different key
sets, and the difference is alphabet-size-dependent.** `paa_attention_scores`
scores attention onto `pool.antecedent`, the *nearest* earlier occurrence of the
predicted token. `knockout_effects` removes every earlier occurrence, via
`antecedent_sets`. Counted from the shipped pools, occurrences before the query:
median **3** per instance on gpt2-large against **13–17** on ProGen2-medium. The
selector therefore sees one of three keys on the text arm and one of thirteen on
the protein arm, so a rank correlation between the two is attenuated harder on
protein *by construction* — in the direction of the hypothesis. This is the same
error `corruption_effects` already fixed for the matching gate, whose docstring
says it plainly: "a conclusion the estimator manufactured out of alphabet size".

**Closed in code (EXP-R2-078), and the size of it is now measured rather than
argued.** `paa_attention_scores` emits `paa_specific_matched` beside
`paa_specific`: attention summed over the key set `antecedent_sets` returns — the
one the knockout removes — against a decoy baseline scaled to the same number of
keys, so the correction stays a positional baseline of matched size rather than a
per-key one subtracted from a sum. Both are returned, because `paa_specific` is
what EXP-R2-059 published and L5/L6 quote, and redefining it in place would make
those numbers unreproducible. *On a validation-scale run — 8 sequences, gpt2-large,
not a result and not quotable as one — the mean key set is 2.93 and the two scores
rank the 720 heads at Spearman **+0.567**.* Part of that is estimation noise at
this cohort size, but it establishes that the two definitions are not
interchangeable **on the arm where the mismatch is smallest**; on a protein arm at
13–17 keys the divergence can only be larger. D2.c must use the matched score.

**Blocker 3, now cleared — the positive control is probably attainable.** The
plan had inherited from L5 the belief that `paa_specific` does not rank causal
effect even on gpt2-large, which would make D2.c a gate no text control can pass.
That figure is on the signed effect; on the magnitude scale L22 uses, the same
heads read **+0.53 to +0.65**. See the L5 block in §5.1 for the table and for the
two limits on that reading.

**Cost, re-derived rather than carried over.** `knockout_effects` needs
`ceil(N/B) × (2 + H)` forward passes for `H` heads and `N` instances: 36,100 at
gpt2-large's 720 heads and 800 instances, against the 24 heads the gate stage runs
today. Anchored on EXP-R2-059's measured 0.73 s per (16, 512) forward, that is
**3–5 H200-hours for gpt2-large alone** and 23–39 for a ten-arm panel, with
gpt2-xl 42% of it. The stage also materialises the full `[batch, token, vocab]`
logit tensor on every pass where one position is read, allocates a dense
`(B, n_head, W, W)` knockout mask per head per batch, and leaves `use_cache` on —
`path_patching._readout` avoids the first of these and documents why. Fixing the
three brings the panel to **12–25 H200-hours**. TF32 would recover roughly 4x more
but must first pass the quantisation check that rejected bfloat16 for this metric,
which is cheap and has not been run.

**Order of work, and it is not "start the campaign".** ~~Fix blocker 2; decide and
declare the protein band and window that admit ProtGPT2 and ZymCTRL, or record
that the matched pair is unavailable and scope D2.c's claim accordingly;~~ run the
exhaustive census on gpt2-large alone and check the all-grid positive control
against the restricted-range +0.53 to +0.65; only then schedule the panel.

**Order of work as it now stands (EXP-R2-078, EXP-R2-082).** The first two items
are done: blocker 2 is closed in code, and the band question is decided — the fix
is the **pool width**, not the band, so the matched pair enters at **`--width 192`
with the cohort band unchanged** and at zero L13 exposure, while **ZymCTRL is
irreducibly excluded from any shared window** and is a separately declared per-arm
configuration or is not run. What remains, in order:

1. ~~**The attainability check, and nothing may be scheduled before it.** gpt2-large's
   exhaustive PAA census at **width 192**, read against the +0.53 to +0.65 the
   restricted-range reading gives at width 512.~~ **DONE, and the criterion was
   replaced before it was read — EXP-R2-087.** Both widths were measured
   exhaustively over all 720 heads (EXP-R2-083/084), which showed the
   pre-registered comparison cannot decide width:

   > **Why the stated criterion was withdrawn, on evidence that does not involve the
   > width-192 number.** The width-512 / restricted-24 / unmatched cell is nominally
   > *identical* between the historical run and a fresh one, and reads **+0.6078**
   > against **+0.4922** — a drift of **0.116 at fixed width**, against a width
   > effect of +0.08. A target reproducible only to ±0.12 cannot resolve a +0.08
   > difference. The census is not the culprit (Spearman +0.986 across the cohort
   > change, max |difference| 1.5e-3); the causal half contributes −0.065, the
   > cohort swap −0.020 and re-selecting the 24 heads on the new ranking −0.047.
   > **And the target is on the wrong score**: the historical artefacts are schema
   > `r2_transfer_paa_gate_v1` and contain no `paa_specific_matched` at all, so
   > +0.53–+0.65 is an *unmatched* figure while this section requires the matched
   > one. Appendix B rule 2, third instance. The correction is outcome-independent:
   > all of it is measurable at width 512 alone.

   **The replacement, and it is the one §5.1 already draws.** gpt2-large's width-192
   copy-suppression census must order causal importance about as well as
   gpt2-large's own **induction** census does — the text control L22 is measured
   against. Exhaustive, magnitude statistic, matched score, bootstrapped over the
   200 sequences with census and causal resampled jointly (alignment verified, and
   the resample regenerates every published per-head `delta_m_gap` exactly).

   | width | exhaustive ρ (matched) | 95% CI |
   |---:|---:|---|
   | 192 | **+0.4515** | [+0.401, +0.498] |
   | 512 | **+0.5309** | [+0.451, +0.547] |

   **PASS.** +0.4515 lands inside the induction band **+0.428 to +0.535** (same arm,
   same statistic, also exhaustive). The width penalty is **+0.079** (CI [−0.015,
   +0.120]) on the matched score and +0.083 (CI [+0.016, +0.118]) unmatched — the
   two agree, so width 192 costs about 0.08 and at most ~0.12. **The design is not
   dead and the width-512 fallback at band 1550–4000 is not needed**, so the full
   1.75–2.31 nat L13 exposure is not paid.

   > **And the pass is draw-robust — EXP-R2-089.** A margin of +0.0239 read off one
   > corpus draw is not a decision, given the same class of statistic was measured
   > moving 0.107 between draws on this very arm. Re-run at three further draws
   > (cohort digests all distinct, A1 instances 21415–21619): **+0.4515, +0.4669,
   > +0.5206, +0.4522** — **4 of 4 inside the band**, worst draw clearing the floor
   > by +0.0239. **The draw the gate was decided on is the lowest of the four**, so
   > the original reading was the conservative one. The statistic's own draw range
   > is 0.0691 matched and 0.0282 unmatched, against the 0.1073 the *induction*
   > reading shows on the same arm and condition — exhaustiveness buys draw
   > stability, which is a further argument for the exhaustive criterion over the
   > restricted-range one it replaced.
2. Only then the protein arms, on the declared width-192 configuration, with the
   two narrowed-estimand costs stated in the claim: two of eight distance bins
   unreachable, and a cohort selected on BPE compressibility.
3. ZymCTRL separately or not at all, and if separately then `build_cohorts` needs
   its reference band decoupled from its cohort band first — as specified the
   single-length window does not run.

### D3 — Adapted methods, hard-gated

Not earned. No proposal is advanced that is not traceable to a catalogued
limitation, and the two standing rejections of §7 are unchanged.

| item | gate | cost |
|---|---|---|
| **D3.a** | Dictionary evaluation whose estimand is power-checked on the text control first, reporting loss recovered and KL rather than FVU; threshold frozen before any protein arm is scored | 17–50 GPU-h |
| **D3.b** | Aperture-functional attribution: after D2.a, within-arm descriptive only, no cross-arm coefficient | 3–5 GPU-h |
| **D3.c** | Oracle-grounded explanation normalised by measured channel capacity: after the L9 re-derivation, attainability shown on gpt2-large first | 4–8 GPU-h |
| **D3.d** | DMS-grounded causal gates, only on features surviving D3.c. **No text analogue of a DMS assay exists**, so this gate cannot be attainability-checked on a text control and needs an internal positive control instead | 10–20 GPU-h |

### 9.0 Retired item names, so citations still resolve

The Phase A/B/C identifiers are referenced in the experiment log and in code
comments, and those records are not rewritten. The mapping:

| old | new | note |
|---|---|---|
| Phase A | D1 | closed to open-ended measurement; two bounded items remain |
| B1 | D2.a | input-contract certification |
| B2 | — | completed, returned NO (EXP-R2-062); the dissociation and C4 are retracted |
| B3 | — | completed, contrast survives (EXP-R2-068); see §5.1 L9 |
| B4 | D2.b | causal effect-size on induction |
| B5 | D2.c | same on copy suppression |
| B6 | — | completed (EXP-R2-068): gate met, magnitudes withdrawn for bfloat16 quantisation, re-run tracked as D1.a |
| C1 | D3.a | dictionary evaluation with a power-checked estimand |
| C2 | D3.b | aperture-functional attribution |
| C3 | D3.c | oracle-grounded explanation |
| C4 | — | dropped; B2 removed its motivation |
| C5 | D3.d | DMS-grounded causal gates |

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

### 9.2 ~~One unresolved question worth 5 GPU-hours~~ — answered (EXP-R2-068)

- ~~**Does non-local propagation survive at production scale?**~~ **Run at 1.5 GPU-h. It survives as an ordering and fails as a modality claim.** Every arm clears the case gate and every interval excludes zero, but the matched pair — the only modality-identifying comparison the panel has — overlaps at [0.202, 0.215]. What separates is ProGen2-medium from both other arms, which is an architecture and tokenisation contrast, and ZymCTRL cannot enter the estimand at all because its conditioning prompt puts the scored span outside the patching window. The strongest available substrate claim is therefore **not** a modality claim. Detail in §5.1.

- **Follow-on, and it narrowed further (EXP-R2-069), then resolved (EXP-R2-070), then was withdrawn (EXP-R2-073 — see §5.1).** A disjoint second window removed ProtGPT2 from the ordering entirely, leaving ProGen2-medium as the only arm above the text control in both draws at every threshold, so the surviving statement was about one lineage rather than one modality. The five-window equal-case successor confirmed that narrowing and confined the separation to the large-effect tail. **All of it is now withdrawn as a model claim**, because every one of those comparisons was made at a band declared in tokens across arms that differ 4.4x in symbols per token, and the ordering between the two protein arms reverses under a content-symbol band. The lineage reading is withdrawn along with the modality reading; what remains is a well-executed measurement of an estimand that was not identified.

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
9. **Never run `git clean` under this repository** — it is its own git repository with `/results/` ignored. This destroyed all experiment results three times.
10. **Search the literature by mechanism name, not by domain**, before designing a track. (Cost: an entire induction track designed without finding prior work that had already established induction heads in protein LMs.)
11. **Compute residual-stream spectra on interior, alphabet-bearing positions.** A participation ratio or top-PC share taken over all positions measures the attention sink and the format separators, not the representation. GPT-2-large reads PC1 0.809 over all positions and **0.034** with its first token dropped (participation ratio 1.53 → 253.4); ProtGPT2 reads 0.971 and **0.439** with its FASTA newlines dropped (1.06 → 4.86). Two headline numbers that were compared to each other turned out to be artefacts of different origin. (EXP-R2-062.)
12. **A single declaration, imported, never reimplemented.** The rendering defect of §0.1 survived a withdrawal that fixed `src/transfer/arms.py` because `scripts/transfer_gap/tg_common.py` carried a second copy. Any module that decides what string a model is fed must import that decision from the panel declaration. (EXP-R2-028, EXP-R2-062.) The same holds one step earlier, for *which records exist* and *which arms a stage measures*: EXP-R2-063 found the latter maintained by hand in five places, two of which disagreed with `PANEL` — one stage's panel silently narrowed by an arm, another silently widened by one. Use `scripts/transfer/panel_contract.py::arm_can_run`.
13. **Declare a stage's cohort band against the band the arms were qualified on.** Four stages of one campaign draw protein cohorts on three different bands (64–246, 64–120, 600–2000), and the cohort `cohort_power` qualifies is not the cohort `lens_family` scores. A per-stage band is a legitimate compute choice; an undeclared one lets a PASS verdict be read as covering a population it was never measured on. (EXP-R2-063.)
14. **A scheduler must obey what a module can deliver, not what the panel intends.** `ArmSpec.capabilities` is an intent and a module's architecture declaration is a deliverable; they are allowed to disagree, and exactly one predicate should consult both. Handing a stage an arm it cannot serve does not fail cheaply — it fails after the checkpoint is on the GPU. (EXP-R2-060, EXP-R2-063.)
15. **A remote runner's success is not the transport's exit code.** Verify the channel carries failure before trusting a green campaign: this one does not, and returned 0 for a command that exited 7. A campaign's verdict must be decidable from the campaign's own output. (EXP-R2-068, L20.)
15b. **Measure a small causal effect in float32, not bfloat16.** A far-band
patching effect of order 0.05 logits cannot be read off a bfloat16 metric whose
quantisation step is 0.0625: the eligible fraction it produced was inflated by 60%
relative, upward. `14_paa_census.py` had already learned this on the M-gap and
defaults to float32 for it; `04_circuit_primitives.py` did not, and the same
lesson cost a second measurement. (EXP-R2-068.)

16. **A declaration derived by comparing resolved paths is not host-portable.** Two environment variables that are distinct on the authoring host can alias on the running host, and then the comparison silently picks whichever branch is written first. Declare the choice where it is made. (EXP-R2-068, L21.)

17. **Prefer threshold-free statistics.** Where a threshold is unavoidable, sweep
it and show the ordering is invariant across the sweep. Two results this
programme recorded as orderings turned out to hold only above a cut: the
far-band propagation ordering reverses at the two most permissive eligibility
thresholds, and the protein/text separation in EXP-R2-070 is clean at 0.50 and
31/40 at 0.05.

18. **Never delete a result artifact; move it.** Frozen provenance is external at
`/Data2/lzp/bio_archive` and is never edited in place. A zero-byte file from a
failed pull is not an artifact, but moving it costs nothing and records that the
pull path can fail silently.

19. **Check `nvidia-smi` and `free -h` before and after every GPU run**; on H200,
run them inside the selected pod. Allocation is not utilisation, and "the GPUs
were idle" must be distinguishable from "nobody looked".

20. **Report failures rather than routing around them.** A refutation is a valid
result; state it plainly. This includes one's own operational failures — three
were recorded in EXP-R2-070's log entry rather than smoothed over, and one of
them (a panel-scoped stage serialised onto a single GPU) would otherwise have
cost thirteen hours silently.

21. **Give every arm the same number of sampling units before comparing them, and
use a scale-free statistic when the arms differ in size.** An interval-overlap
test across arms carrying 288 and 128 cases is partly a reading of sample size,
and it supported a far-band claim that a matched-count re-run then narrowed. The
same error hides inside per-arm proportions: "the top 1% of heads" is one head on
GPT-2 and seven on ProtGPT2, so concentration must be read from a Gini or a
share-of-grid rather than a top-k share. (EXP-R2-070, EXP-R2-071.)

22. **Cohort sensitivity is a property of the statistic, not of the modality.**
Measured in one session on the same arms: the pathway-budget ratio reproduced
within 0.05 on every arm across draws, while the far-band eligible fraction moved
a quarter of its own value on ProtGPT2 between two windows of one permutation. A
draw-robustness result for one measurement licenses nothing about another, in
either direction. (EXP-R2-068, EXP-R2-069.)

23. **Never normalise a signed decomposition by summed magnitudes.** The four
causal-path components sum to the total exactly and still oppose in sign on
40–50% of heads, so a "share" built from summed absolute values is not a share of
anything — ProGen2-medium's reached 1.03 of the whole it was supposedly part of,
and the modality contrast it produced was an artefact of the normalisation.
Confirm the components do not cancel before dividing by them. (EXP-R2-071.)

24. **A statistic compared against the set that selected its own inputs is
circular, and the circular answer is indistinguishable from a real one.** A
top-20 Jaccard against the prefix-matching census returned 1.0 on all four arms
because the sender set *was* the census-selected set — which is exactly what
genuine agreement would return, so nothing in the artefact revealed the defect.
Enforce the precondition in code and raise; a comment does not stop the next run.
(D2.b, repaired in `causal_census_agreement`.)

25. **Before reading a low rank correlation as absence of signal, test reliability
and concentration.** Protein arms gave an all-head census-to-causal ρ near zero,
which was first recorded here as the census carrying no information about causal
rank. That was wrong: per-head reliability was high, so it was not attenuation,
and ρ over the top 32 heads was +0.808 on ProtGPT2 — higher than either text
control then on the panel. The bulk of the grid was unordered, not the top of it,
and the finding had to be retracted and restated. (EXP-R2-071.) *The supporting
numbers in this rule were themselves corrected by EXP-R2-072/073 — see §EXP-R2-071;
the reasoning is what the rule carries, not the figures.*

26. **Declare the unit of every distance, window and band, and resolve it per arm
when the arms differ in symbols per token.** The far-band propagation estimand was
declared in tokens and compared across arms at 4.4x, 2.8x and 1.0x symbols per
token. The ordering between the two *protein* arms — same content alphabet, only
the tokeniser differing — reverses between a token band and a residue band, so the
sign of the reported contrast was a free parameter of an unstated choice. The
artefact recorded `symbols_per_token` in the same file and nothing consumed it. A
band constant with no per-arm resolution is a cross-arm claim waiting to happen.
(EXP-R2-073.)

27. **A statistic quoted across arms must state its denominator.** Per-head causal
effects are normalised by each arm's own clean-minus-corrupt metric, and those
denominators span 32x across the panel (0.76 to 24.10 logits). The same head read
15.7x the text control's on the normalised scale and 0.50x on the logit scale. A
normalised effect is a within-arm quantity unless the denominator travels with it.
(EXP-R2-072.)
