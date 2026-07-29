# Pre-registration: Representation-Recoverability Audit of Protein Generators

**Protocol ID:** R2-RECOV-AUDIT-v1
**Date frozen:** 2026-06-04
**Owner dir:** `r2_interpretability_transfer/`
**Status:** PRE-REGISTERED. Thresholds and analyses below are fixed *before* any
result is inspected. Any change after the first probe is run must be recorded as
a dated amendment in `r2_interpretability_transfer/preregistration/DECISION_LOG.md`, with
the reason, and must not be motivated by an observed outcome.

---

## 1. Motivation and the confound this resolves

R2's circuit-tracing programme produced a stack of negatives: conserved triplets
are dominated by local-context / positional / residual-norm signatures, no
Swiss-Prot biological dictionary emerged, the 38-triplet basis loses to ESM-2 on
downstream probes, and steering / causal ablations failed their gates. Two
explanations are currently **confounded** and cannot be separated by any
experiment run so far:

- **H1 — tool/dictionary failure.** The models encode rich biological structure,
  but the cross-layer transcoder (CLT) dictionaries (dead-feature fraction
  0.57–0.67, FVU 0.20–0.33; Extended Data Table 1 of the manuscript) and the
  intervention path are too lossy to expose or manipulate it.
- **H2 — substrate failure.** The models encode little beyond local
  n-gram/positional statistics, so interpretability correctly reports near
  emptiness.

There is already indirect evidence against the *strong* form of H2 (ZymCTRL
generates EC-class-appropriate enzymes that pass Pfam/CLEAN/structure filters
with Cohen's d > 1; a prior note records that the L35 EC-discrimination signal
"lives in raw activations the CLT cannot capture"). This protocol turns that
hint into a quantitative, pre-registered test.

**Primary question.** For each model, decompose the apparent failure into a
*recoverable ceiling* (what a linear probe extracts from the raw residual
stream), a *recovered floor* (what the same probe extracts from the CLT sparse
codes), and the *gap* between them. The gap localizes the failure to the tool;
the ceiling-vs-baseline contrast localizes it to the substrate.

This study does **not** revive any claim outside R2's discipline (no
single-feature causality, no causal attention-sink mechanism, no
biological-primitive dictionary, no working steering). Steering here is used only
as a *controllability measurement*, reported as calibrated.

---

## 2. Hypotheses as pre-stated, directional predictions

| ID | Prediction | Decides |
|----|-----------|---------|
| P1 | Raw-activation ceiling beats the n-gram/position baseline by a pre-set margin on ≥2 tasks for ZymCTRL and ProGen2-medium. | Substrate is non-trivial (rejects strong H2). |
| P2 | The CLT sparse-code floor recovers < half the ceiling skill (recovery ratio ρ ≤ 0.5) on ≥2 tasks. | Failure is dictionary-localized (supports H1). |
| P3 | ProtGPT2 (BPE tokenizer) shows a lower ceiling and/or larger residue-mapping degradation than the per-residue models. | Tokenization is a confound, not a model verdict. |
| P4 | An oracle probe-direction injection shifts ZymCTRL EC-class purity (permutation p < 0.05) where CLT-feature steering did not. | The model is controllable; sparse features were the wrong handle (H1), not "uncontrollable model". |

Each prediction has a pre-registered numeric rule in §6. We commit to reporting
all four outcomes regardless of direction.

---

## 3. Probing tasks (full list)

Probes are **linear** (L2-regularized logistic regression for classification,
ridge for regression) so the probe never adds capacity beyond a linear readout;
this keeps the ceiling an honest *linear-decodability* ceiling.

| Task | Type | Level | Label source (see §4) | Chance / baseline |
|------|------|-------|-----------------------|-------------------|
| T1 EC top-class (7-way) | classification | protein (mean-pooled) | EC-labeled Swiss-Prot FASTA + GO EC-xref fallback | majority / stratified shuffle |
| T2 Pfam family (top-20) | classification | protein | dominant Pfam family from InterPro residue map | majority |
| T3 Secondary-structure fractions (helix/strand/turn) | regression (3 targets) | protein | Swiss-Prot SS residue features | mean predictor (R²=0) |
| T4 Per-residue 3-state SS | classification | residue | Swiss-Prot SS residue features (helix/strand/coil) | residue majority |
| T5 Decoder-native EC class | classification | protein | the ZymCTRL EC conditioning tag of generated sequences (8 classes) | majority |
| T6 (stretch) Residue contact / local fold | classification | residue-pair | staged AlphaFold/PDB structures (`data/alphafold`, `data/pdb`) | distance-prior |

- **T1–T3** reuse the exact label loaders already implemented in
  `34_triplet_basis_probes.py` (`build_records`, `secondary_fractions`,
  `parse_ec_fasta`, GO EC-xref fallback).
- **T5 is the decisive substrate test**: it asks whether the model linearly
  encodes the very conditioning signal it demonstrably *uses* during generation.
  A model cannot conditionally generate EC-appropriate enzymes without encoding
  EC class somewhere; T5 measures where, and whether the CLT keeps it.
- **T4/T6** are the residue-level ceiling; they are the tasks most likely to
  expose dictionary information loss because the CLT operates per-token.
- **T6 is a stretch** task gated on staged-structure coverage; it is not part of
  the primary go/no-go (§6) and may be dropped without amendment if coverage
  is < 50 proteins with usable structures.

---

## 4. Reusable datasets and labels (exact paths)

All repository-relative to repo root; large trees are B-only (inspect via Bash).

| Asset | Path | Use |
|-------|------|-----|
| Swiss-Prot annotation cache | `data/processed/swissprot_all_max1022.pkl` | T1–T4 cohort + SS/Pfam labels (383 MB) |
| Pfam residue intervals | `data/interpro/pfam_residue.tsv` | T2 dominant-family labels (18 MB) |
| EC-labeled Swiss-Prot FASTA | `data/zymctrl/ec_labeled_swissprot.fasta` | T1 EC top-class (primary) |
| GO annotations + ontology | `data/go/goa_uniprot_all.gaf.gz`, `data/go/go-basic.obo` | T1 EC label fallback via EC xrefs (17 GB GAF) |
| ZymCTRL EC table | `data/zymctrl/enzyme.dat` | EC-class metadata for T5 |
| Decoder-native EC cohort | `r2_interpretability_transfer/results/steered_generation/`, `r2_interpretability_transfer/results/steering_benchmark/zymctrl_v2_onmanifold_direct_20260503.json` | T5 generated-sequence set, 8 EC classes |
| Structures (stretch) | `data/alphafold/`, `data/pdb/` | T6 contacts / DSSP residue SS |
| **ESM-2-3B (positive control)** | `/Data/public/esm2_t36_3B_UR50D` | external rich-representation reference |
| ESM-2-650M (light option) | `/Data/public/esm2_t33_650M_UR50D` | cheaper reference for iteration |

**Models and CLT checkpoints (representations under test).** Use the *same v2
reference checkpoints* as the manuscript so the audit speaks to the paper's
result, not a different dictionary:

| Model | Tokenizer | CLT checkpoint (reference) | Manuscript FVU / dead |
|-------|-----------|----------------------------|-----------------------|
| ProtGPT2 | BPE (multi-residue) | `protgpt2_v2/step_200000` | 0.2013 / 0.6197 |
| ZymCTRL | per-residue | `zymctrl_v2/step_200000` | 0.3307 / 0.6682 |
| ProGen2-medium | per-residue | `progen2-medium step_100000` (final_checkpoints) | 0.33 / 0.57 |

> ⚠ Checkpoint *paths* in existing scripts point at GPFS/OSS staging on the H200
> pod and were partially rewritten by the 2026-05-28 directory rename. **Re-confirm
> the live checkpoint paths on the active pod before running** (see
> `r2_interpretability_transfer/preregistration/DECISION_LOG.md`); pass them via
> `--model-spec`, do not trust the hard-coded defaults.

**Cohort construction (frozen).** Length filter 100–400 aa (matches existing
probes). Protein-level **family-disjoint** splits: no Pfam family spans train and
test (prevents homology leakage; stricter than the existing StratifiedKFold).
Target cohort sizes: T1 ≥ 40/class × 7; T2 12/family × 20; T3/T4 ≥ 300 proteins;
T5 = all generated sequences passing the basic validity filter. Global seed
`20260604`; 5 CV folds; 5 seed repeats for CI. Identical cohort and splits across
all representations (paired comparison).

---

## 5. Representations compared, and the ceiling/floor/gap metrics

For each model × layer, build these feature matrices on the **identical** cohort
(per-residue codes mean-pooled to protein level for protein tasks; mapped to
residues via `token_residue_spans` for residue tasks):

| Symbol | Representation | Source call | Role |
|--------|----------------|-------------|------|
| `R_raw` | raw residual stream `resid_pre[layer]` | `pm.get_activations(ids).resid_pre` | **ceiling** |
| `R_code` | CLT sparse codes `clt.encode(resid_pre)[layer]` (alive features only) | `clt.encode(...)` | **floor** |
| `R_recon` | CLT reconstruction of its target | CLT forward output | intermediate (what the dictionary can re-express) |
| `B_ngram` | k-mer composition (k≤3) + length + mean normalized position | computed | null baseline |
| `R_rand` | random projection of `R_raw` to `dim(R_code_alive)` | computed | dimensionality control |
| `ESM2` | ESM-2-3B mean-pooled `last_hidden_state` | `esm2_matrix` in script 34 | external rich reference |

**Skill** (chance-corrected) for a representation `R` on task `t`:

```
skill(R,t) = metric(R,t) − chance(t)
```
where `metric` = macro-F1 (T1,T2,T4,T5), R² (T3, with chance 0), and `chance` is
the stratified-shuffle / majority value estimated on the same folds.

**Three core quantities** (reported per model × task, with best-layer and a
layer profile):

```
Ceiling   C(t)  = skill(R_raw, t)             # best layer
Floor     F(t)  = skill(R_code, t)            # same layer as the ceiling, and best-code-layer (both reported)
Gap       Δ(t)  = C(t) − F(t)
Recovery  ρ(t)  = F(t) / C(t)        (defined only when C(t) > 0; clipped to [0,1])
```

Auxiliary, to interpret the substrate against a rich reference:

```
Reference fraction   φ(t) = C(t) / skill(ESM2, t)     # how rich is the decoder vs ESM-2
Dimensionality check                C(t) vs skill(R_rand, t)  # ceiling must beat random projection at matched dim
Reconstruction retention            skill(R_recon, t) / C(t)
```

**Uncertainty.** 95% bootstrap CIs over **proteins** (1,000 resamples) for every
skill, and across the 5 seed repeats. Comparisons (e.g., C vs baseline, F vs C)
use the paired bootstrap; significance after Benjamini–Hochberg across
(task × model × layer). Sparse-code probes use strong L2 (`C≤0.5`) on **alive
features only** and are additionally reported at matched dimension (PCA of
`R_raw` to `dim(R_code_alive)`) so a ceiling/floor gap cannot be an artifact of
feature count.

---

## 6. Decision rules and go/no-go thresholds

All thresholds are frozen here. "On ≥2 tasks" means among the primary tasks
{T1, T2, T3, T4, T5}; T6 is excluded from gating.

### 6.1 Per-model substrate verdict (H1 vs H2)

| Verdict | Condition |
|---------|-----------|
| **Substrate is rich** | `C(t)` beats `B_ngram` skill by ≥ **0.10** (macro-F1) or non-overlapping 95% CI, **and** beats `R_rand`, on **≥2** tasks; ideally `φ(t) ≥ 0.5` (decoder retains ≥50% of ESM-2's recoverable skill). |
| **Substrate is thin** | `C(t)` within the 95% CI of `B_ngram` on **all** primary tasks. → reports **H2** for that model. |
| **Dictionary is the bottleneck** | Substrate is rich **and** `ρ(t) ≤ 0.5` on ≥2 tasks. → supports **H1**. |
| **Dictionary near-faithful** | `ρ(t) ≥ 0.8` on the tasks where the substrate is rich. → failure is not the dictionary's coverage. |

### 6.2 Controllability verdict (oracle steering, Experiment 3)

Inject the T5/T1 EC probe direction into `resid_pre` at the recognition layer
(L3 family for ZymCTRL) using the **same TopK-aware path and the same 8-class
purity gate** as `11_steering_benchmark.py`, for comparability.

| Verdict | Condition |
|---------|-----------|
| **Model is controllable** | oracle-direction Δ(class purity) ≥ **0.05** with permutation **p < 0.05** on ≥3/8 classes. → prior CLT-feature steering failure is a *handle* problem (H1). |
| **Behaviour is distributed/robust** | oracle direction fails the same gate. → the steering negative reflects representation geometry / redundancy, a genuine model property, **not** merely bad features. |

### 6.3 GO / NO-GO for the high-cost retraining (Experiment 5)

The expensive step (train one stronger dictionary on the best decoder: larger
width, dead-feature resampling, longer schedule) runs **only if** the cheap
experiments justify it:

- **GO** — run the retrain — **iff** there exists ≥1 model meeting *both*:
  1. **Rich substrate** (§6.1) on ≥2 tasks, **and**
  2. **Dictionary bottleneck** `ρ(t) ≤ 0.5` on ≥2 tasks.
  Retrain **target** = the model with the largest absolute mean gap
  `mean_t Δ(t)`. Rationale: the information provably exists (`C` high) and the
  current dictionary provably misses it (`ρ` low), so a better dictionary *can*
  close the gap.

- **NO-GO (substrate thin)** — if every model is "Substrate is thin" (§6.1).
  Conclusion: the apparent circuit-tracing failure is substrate-limited (H2);
  no dictionary can help. Report this as the finding; do not spend retrain
  compute.

- **NO-GO (already faithful)** — if for the rich-substrate model(s)
  `ρ(t) ≥ 0.8` on the rich tasks. Conclusion: the dictionary already preserves
  the recoverable signal; the failure is downstream (intervention/geometry),
  and a bigger dictionary will not change the narrative.

### 6.4 Pre-registered success criterion for the retrain itself

If GO, the new dictionary is declared to **confirm H1 (tool-limited)** iff,
relative to the reference checkpoint and at matched cohort/splits:

- dead-feature fraction **< 0.30** **and** FVU **< 0.15** (a genuinely better
  dictionary), **and**
- recovery ratio `ρ(t)` rises by **≥ +0.20 absolute** on ≥2 tasks (moves toward
  the unchanged ceiling `C`), **and/or**
- the previously-failed CLT-feature steering now reaches the §6.2 controllability
  gate.

Outcomes short of this are reported as **partial / H1-architectural**: even a
well-trained sparse dictionary cannot make the information linearly monosemantic
— a substantive, publishable result about the limits of sparse dictionaries on
generative protein decoders, distinct from "the model learned nothing."

---

## 7. Confound controls and validity threats

- **Tokenization (P3).** All per-token quantities are mapped to residues with
  `token_residue_spans` / `add_token_values_to_residues`. Primary cross-model
  contrasts are also run on a **per-residue-only** subset (ZymCTRL, ProGen2);
  ProtGPT2(BPE) is reported separately to quantify the tokenizer penalty.
- **Homology leakage.** Family-disjoint CV (no Pfam family across train/test).
  T2 additionally reported with clan-level holdout if a clan map is later staged.
- **Dimensionality.** `R_rand` matched-dim control and PCA-matched ceiling so a
  high-dimensional sparse code cannot win or lose on capacity alone.
- **Layer selection.** Probe **all** layers; report the full layer profile and
  the best layer; the ceiling/floor must be compared at the *same* layer (plus a
  best-of each, both reported) to avoid layer cherry-picking.
- **Probe capacity.** Linear probes only; L2 fixed a priori (`C≤0.5` for codes,
  `C=1.0` for `R_raw`/ESM-2); no per-task tuning.
- **Multiplicity.** BH correction across (task × model × layer).
- **Blinding.** Thresholds in §6 are frozen; the analyst computes all skills and
  CIs before applying the decision table; deviations logged.

---

## 8. Implementation plan (new scripts under `r2_interpretability_transfer/scripts/`)

Continue the existing numbering; reuse the loaders cited above rather than
re-implementing.

| Script | Role | Reuses |
|--------|------|--------|
| `44_cache_representations.py` | For each model × layer × cohort, cache `R_raw`, `R_code` (alive), `R_recon`; build `B_ngram`, `R_rand`; cache `ESM2`. | `src/models/model_loader.py` (`load_model`, `get_activations.resid_pre`, spans), `src/training/clt_trainer.py` (`load_trained_clt`, `encode`), `34_triplet_basis_probes.esm2_matrix` |
| `45_probe_ceiling_floor.py` | Run T1–T5 linear probes on every representation; compute `C/F/Δ/ρ/φ`, bootstrap CIs, layer profiles; emit per-(model,task) tables. | `34_triplet_basis_probes.py` probe machinery + label loaders (`build_records`, `run_task`) |
| `46_oracle_direction_steering.py` | §6.2 oracle-direction injection vs CLT-feature steering, 8-class purity gate. | `10_steered_generation.py`, `11_steering_benchmark.py`, `diagnostics/00_hook_sanity.py` |
| `47_decision_table.py` | Apply §6 thresholds, emit the per-model verdict and the single GO/NO-GO for the retrain. | outputs of 45/46 |
| `48_capacity_retrain.py` *(only on GO)* | Train one stronger dictionary on the GO target (≥4× width, dead-feature resampling, longer schedule); re-evaluate FVU/dead and re-run 44/45 on it. | `src/training/clt_trainer.py` |

**Output location:** `r2_interpretability_transfer/results/representation_audit_20260604/`
with subdirs `cache/`, `probes/`, `steering/`, `decision/`, `retrain/`. Every
table carries the cohort hash, seed, checkpoint path, and git state.

---

## 9. Compute budget and stopping rule

- Experiments 44–47 are **cheap**: cached activations + linear probes + one
  steering sweep. Estimated ≤ 1 GPU-day total on a single L20/H200 (ESM-2-3B
  embedding of a few hundred proteins is the largest single cost; use the 650M
  reference if memory-bound).
- Experiment 48 (retrain) is the only large cost and is **gated** by §6.3.
- **Stopping rule.** Run 44→45→46→47. Inspect the §6.3 verdict. If NO-GO, stop
  and write the audit as a substrate/faithfulness finding. If GO, run 48 once on
  the single largest-gap model; do **not** iterate dictionary hyper-parameters
  hunting for features — one well-specified retrain answers the H1 question; a
  second attempt would require a new, dated amendment with its own pre-set gate.

---

## 10. What each outcome means for the paper

| §6.3 verdict | Headline the audit licenses |
|--------------|-----------------------------|
| GO → retrain confirms H1 | "Protein generators encode recoverable biological structure that current sparse dictionaries discard; a faithfulness-targeted dictionary recovers it." Constructive, method-positive. |
| GO → retrain partial | "The information exists but resists sparse monosemantic recovery even with a strong dictionary — a limit of sparse dictionaries on generative decoders." Methodological, novel. |
| NO-GO (thin) | "These generators encode little linearly-accessible biology beyond local statistics; interpretability negatives reflect the substrate, not the tool." Cautionary, well-scoped. |
| NO-GO (faithful) | "Dictionaries are faithful to a thin recoverable signal; the bottleneck is intervention/geometry (controllability), not feature coverage." |

Any of these is a coherent, defensible contribution and directly supplies the
*feasibility / faithfulness-audit* spine discussed for the flagship framing. No
outcome requires crossing R2's claim-discipline lines.
