# Transfer measurement programme: estimands and metrics

**Updated:** 2026-07-29 **Status:** methods only; asserts no results. **Scope:** this file states, per stage, the quantity estimated (the estimand) and how it is scored (the metric). Scope, panel, evidence discipline and compute policy are in `docs/RESEARCH_PLAN.md` and are **not** restated here. Findings and their status are in `docs/INTERPRETABILITY_TRANSFER_AUDIT.md`, which wins over both.

The panel is eleven arms; `scripts/transfer/panel_contract.py::arm_can_run` decides which of them a given stage may run, and refuses the rest by name and reason. A stage's cohort band is declared in its own artefact beside the band its arms were qualified on — the bands differ per stage by design (64–246, 64–120, 600–2000) and an undeclared band lets a verdict be read as covering a population it was never measured on.

---

## Prerequisite — cohort power (`01_cohort_power.py`)

**Estimand.** The context-derived information an arm commits on a frozen cohort: the arm's context-free (unigram) entropy on the cohort minus its clean next-token cross-entropy. The unigram reference is **held out**, never plug-in; a plug-in estimator biases this by up to +1.02 nats on a 50k-vocabulary arm against +0.009 at residue level (L12).

**Metric.** Context information in nats/token against a minimum-information threshold. An arm below threshold is reported **unmeasurable on that cohort**, not scored further and not reported as failing. A Markov-baseline ladder and a visible-context truncation curve are recorded as supporting diagnostics, not as the pass/fail criterion.

**Cohort construction.** A seeded permutation of the eligible set, never a head-of-file prefix (Appendix B rule 1). The stage records a skip-offset sensitivity; note that `--cohort-skip` moves the scored cohort *and* the held-out reference together, so its output is a joint shift and must be decomposed before it is read as a cohort effect.

## Stage 1 — pathway budget (`02_pathway_budget.py`)

**Estimand.** The share of an arm's context information carried by one sublayer pathway (MLP or attention; a single layer, a depth-relative window, or the whole pathway) against the residual-block ablation and against each other, swept over **relative** depth so arms of different layer count are commensurable.

**Metric.** Cross-entropy delta and KL divergence from the clean predictive distribution under ablation against a declared baseline (for example a cohort-mean patch), each required to clear a minimum-effect-size guard before being interpreted, reported as a share of the cohort's context information with sequence-cluster bootstrap intervals across independent cohort-subsample seeds.

**Two properties of the reported ratio that must travel with it.** It is a ratio to context information, not a partition — values above 1 mean the ablation costs more than all the context information extracted. And the tokenisation adjustment that collapsed the earlier MLP-share coefficient has not been applied to it.

## Prerequisite — estimand power (`03_estimand_power.py`)

**Estimand.** Which member of the same pathway x depth x width x baseline sweep has a large enough causal footprint to support a recovery-fraction gate at all. "Powered" means the whole 95% sequence-cluster interval clears both guards, not just the point estimate.

**Metric.** The same cross-entropy-delta and KL metrics as the pathway-budget stage, aggregated per arm and then across the panel into one verdict per estimand: powered on the text control; powered on every protein arm; and, when both hold, the most localised such estimand, selected as the recommended one to hang any future recovery gate on. The artefact records `attainable_on_text_control` explicitly.

**Why this is a prerequisite and not a result.** Attainability falls monotonically with depth across the within-lineage ladder, so a gate built on a single-submodule estimand penalises deep text and deep protein models alike for a reason having nothing to do with dictionaries. Checking attainability before application is Appendix B rule 2.

## Stage 1, 4 — circuit primitives (`04_circuit_primitives.py`)

**Estimand.** Three measurements per arm, each testing an assumption the text-derived circuit toolkit inherits: the prefix-matching (induction) score per head over repeat probes; direct logit attribution per head; and an activation-patching map over layer x position.

**Metric.** Per-head prefix-matching score against a threshold sweep and against a decoy/positional control — the correction matters for the *body* of the distribution and not for its tail, so a tail statement and a distributional statement are scored separately. Probes come in three constructions (synthetic, natural exact, natural approximate) and the construction is reported with the number, because the ordering is probe-dependent.

## Stage 1 — convergence control (`07_convergence_control.py`)

**Estimand.** Whether a measured text-versus-protein gap is attributable to modality rather than to pretraining convergence, model scale or tokeniser cardinality, by measuring the same quantity across a within-lineage scale ladder and across matched-corpus pairs.

**Metric.** The gap restated against the ladder's scale-matched prediction, plus a variance decomposition into scale, lineage and modality components with the fraction of the modality indicator surviving projection reported alongside.

## Stage 3 — relational channel (`05_relational_channel.py`)

**Estimand.** Whether a long-range residue-contact partner, anchored against sequence-separation-matched decoys, is identifiable from per-position hidden states at swept relative depths, or only from the raw attention pattern — the pathway a frozen-attention attribution graph would discard.

**Metric.** ROC-AUC with standard error for a linear probe and a small MLP probe per predictor arm, on a homology-disjoint train/test split built from Pfam-family and k-mer clustering, contrasted against the same measurement on a leaky random split to size the homology-leakage gap, and against a separation-only control.

**Applicability.** Requires a residue-level tokeniser with a defined residue-to-token map, which restricts the stage to ZymCTRL and ProGen2-medium. ProtGPT2's multi-residue BPE and the text arms' word-level BPE do not define one, and must not be given a heuristic approximation.

## Stage 3 — lens family (`08_lens_family.py`)

**Estimand.** Per-layer readout agreement for the logit lens, the tuned lens and the property-conditioned Jacobian, plus the numerical rank of `∂logits/∂h_l` — the output aperture, which is algebraically bounded by vocabulary size.

**Metric.** Top-1 agreement and cross-entropy per layer per lens, within-arm; rank of the Jacobian validated against central finite differences with the relative error reported against a stated tolerance. Cross-arm lens comparisons are bounded by the aperture and by the arm's cohort band, both of which are recorded in the artefact.

## Stage 3 — probes and erasure (`09_probe_and_erasure.py`)

**Estimand.** Two quantities that are routinely conflated: **decodability**, the skill of a probe trained to read a property off the residual stream; and **reliance**, the change in the model's own next-token loss when that property's subspace is erased (LEACE).

**Metric.** Probe skill with intervals over grouping units drawn under a seeded permutation of the corpus, and erasure-induced cross-entropy delta with the same unit definition. Grouping units are sampled, not taken in file order — under file order the same 400 proteins yield 135 distinct grouping units against 231 under a seeded permutation.

**Refusals are outputs.** A residue-level property on a subword protein arm is refused, not approximated.

## Stage 3 — explanation channel (`06_explanation_channel.py`)

**Estimand**, in four parts, none of which depends on any interpretability method: (A) the analytic maximum mutual information a top-k event-selection design can report at a given cohort size and realised event count — an upper bound on what any gate expressed that way could show, independent of any model; (B) the marginal entropy of curated Pfam residue labels; (C) the marginal entropy of AlphaFold-derived structural attributes (secondary-structure state, contact-number bin, pLDDT-confidence bin); (D) the within-sequence label entropy over one fixed window for text-token identity, residue identity, Pfam label and structural attributes — the decisive part, because this package's matched null permutes labels within a sequence, and a label constant within a sequence is invariant under that null and so has no power by construction rather than by result.

**Metric.** Nats and an attainable/unattainable verdict against a stated gate size for (A); bits/symbol (Shannon entropy) for (B)–(D). Each channel's sampling mode is recorded per channel, because they differ: the structural channel is drawn under a seeded permutation, the Pfam and text channels are still corpus-prefix draws and are marked as such.

## Stage 1 — homology control (`10_homology_control.py`)

**Estimand.** Whether the protein induction signal is computation or retrieval of a memorised corpus entry, by stratifying the probe cohort by sequence identity to the pretraining corpus and comparing induction across strata, against a synthetic-repeat negative control that no database can contaminate.

**Metric.** DIAMOND `blastp` identity over query, **with `--masking 0`** — the default repeat masking truncates the HSP at exactly the repeats a repeat-selected cohort is built from, and under-counts identity in the direction that defeats the memorisation hypothesis. `assign_homology` raises on a truncated alignment rather than binning it. Stratum bootstraps carry a unit floor and a `bootstrap_n_units` field; a degenerate stratum is named under `underpowered_strata` rather than being given an interval.

## Stage 4 — induction path patching (`11_induction_path_patching.py`)

**Estimand.** How much of the induction heads' logit effect is written directly to the output versus routed through later components, on matched pairs.

**Metric.** Direct-effect fraction with an interval **over probe records**, the only real sampling unit. Resampling an arm's induction heads — its entire population, selected by a threshold — measures heterogeneity, not sampling error, and produces an interval that does not mean what it appears to.

## Stage 1 — induction robustness and bootstrap (`12`, `13`)

**Estimand.** Threshold robustness of the census ordering across fixed and data-driven cuts and across the continuous cut grid; distributional separation (AUC, stochastic and quantile dominance, model-level median test); and the head fraction with a probe-cluster bootstrap interval.

**Metric.** Both stages read census artefacts already on disk; no model is loaded and no GPU is touched. The bootstrap is over probes because the census averages each head's score over probes before writing, so the stored artefacts cannot support it without re-reading per-probe records.

## Stage 1 — PAA census gate (`14_paa_census.py`)

**Estimand.** Whether a prevalence census can be run for a second mechanism (prediction-addressed attention / copy suppression) at all: whether a cheap per-head statistic exists that ranks heads by measured causal effect, and whether it has dynamic range.

**Metric.** Rank correlation between the cheap statistic and the measured per-head causal effect, with its p-value, plus the count of heads clearing a control band against the same count for induction. The gate is go/no-go and is run **on the text control first**.

---

## Where outputs land

`results/transfer_<YYYYMMDD>[_<suffix>]/<stage>/`. A campaign writes one subdirectory per stage plus a per-arm JSON inside it. Every artefact records its own provenance: estimator name, sampling mode, cluster unit, capability gate, cohort band, tolerance. If a number can be misread, the disambiguating field sits next to it.
