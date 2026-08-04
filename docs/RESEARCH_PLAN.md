# R2 research plan: text-to-protein interpretability transfer

**Updated:** 2026-08-04 **Status:** active **Subordinate to:** `docs/INTERPRETABILITY_TRANSFER_AUDIT.md`. That document holds the findings, the retractions and the costed plan. This one holds scope, panel, measurement package, discipline and compute policy. Where they disagree, the audit document wins.

## Objective

1. **Differences.** Compare language-generative and protein-generative models.
2. **Transferability.** Analyse the limitations of text-derived interpretability methods applied to protein generative models.
3. **Adapted methods.** Design and validate protein-adapted methods.

Parts 2 and 3 are the deliverable. Part 1 is instrumental: a difference between model families matters here only insofar as it explains why a method transfers badly. Part 1 is closed to new measurement — see the audit document §9 D1 (the Phase A/B/C names were retired on 2026-07-30; §9.0 holds the mapping).

## Scope

Autoregressive protein generators and matched text decoders. Encoder work is out of scope.

The prior conserved sparse-readout programme — atlas, EC steering, enzyme design, and npj manuscript — is **retired**, with its documents, results, and receipts frozen at `/Data2/lzp/bio_archive/legacy/r2_retired_scope_20260729/`. It is not an input to this programme. Two of its artefacts are exceptions and remain live because this programme depends on them:

- `evidence/p0_2_adjudication_20260727/` and `evidence/p0_2b_fidelity_20260727/` are the evidence behind limitation **L1** and the baseline plan item **C1** re-qualifies. `docs/analysis/P0_2B_DICTIONARY_FIDELITY_RESULTS_20260727.md` is their prose form.
- `results/final_checkpoints/` holds the four April 2026 CLT checkpoints, the only protein dictionaries held locally.

## Model panel

Twelve autoregressive decoders under one code path, each fed in the format it was trained on. Full table, with the identifying contrasts, in the audit document §2. `scripts/transfer/panel_contract.py` is the declaration; the lists below are a reading of it.

| modality | arms |
|---|---|
| text | gpt2, gpt2-medium, gpt2-large, gpt2-xl, dialogpt-small, qwen2.5-0.5b, llama-3.2-3b |
| protein | protgpt2, zymctrl, progen2-base, progen2-medium, progen2-small |

Design properties worth restating because they bound every inference drawn here:

- **gpt2-large / protgpt2** is the matched modality pair — identical architecture, depth, width, vocabulary size and parameter count (774,030,080, verified from both checkpoints; the 773,891,840 recorded here until 2026-08-04 was wrong by 138,240, though the identity it asserts is exact).
- **gpt2 → gpt2-medium → gpt2-large → gpt2-xl** is a within-lineage scale ladder holding architecture, tokeniser and corpus fixed.
- **progen2-base / progen2-medium** is the protein-side corpus contrast. The text-side corpus contrast (gpt2 / dialogpt-small) is **retracted**: dialogpt-small reads −4.08 nats context information on the evaluation corpus and cannot anchor a contrast (audit §5.05(a)).
- **qwen2.5-0.5b, llama-3.2-3b** are the cross-lab lineage contrast. They are what distinguishes a modality effect from a GPT-2 idiosyncrasy; that distinction is what retracted the QK/OV finding and what supports the pathway budget result.

**Structural limit, irremovable.** The only model family spanning both modalities is GPT-2, five text arms against one protein arm. Every modality coefficient is therefore carried by ProtGPT2 alone. A single-model contrast must not be presented as a modality finding.

Each arm is fed input in the format it was trained on; format selection is enforced in code by the panel declaration in `src/transfer/arms.py`. This is not cosmetic — rendering is worth 1.42–1.78 nats/token on ProtGPT2 (L11).

## Four-stage decomposition

An interpretability result depends on a chain of stages. A transfer failure can originate at any of them, and the stages fail differently, so each is measured separately rather than reported as one end-to-end score.

| stage | question | method families covered |
|---|---|---|
| **1. Substrate** | Where is the computation, and how much of it is there? | transformer-circuits framework (QK/OV), induction heads, head-prevalence census, superposition |
| **2. Instrument** | Can a sparse dictionary replace that computation faithfully? | dictionary learning, SAEs, transcoders, cross-layer transcoders, crosscoders |
| **3. Semantics** | Can the recovered features be named and validated? | logit and tuned lens, direct logit attribution, probes, concept erasure, automated interpretability |
| **4. Causal verification** | Do interventions on features control behaviour? | activation patching, path patching, attribution graphs, steering and clamping |

Stage 1 is measured first and without any dictionary, because a dictionary result cannot be interpreted until the substrate it decomposes is characterised. Stage 2 has no dedicated module. It previously reached into `src/revision/` so that transfer measurements and the retired dictionary qualification shared one windowed-transcoder implementation; that rationale expired when B2 returned NO and C4 was dropped (§0.1). An AST-resolved import closure showed the dependency was **twelve symbols across five modules, dragging 2,775 lines in solely through a module-level import for three functions this package never called**. Those twelve are now vendored into `src/transfer/{io,statistics,scoring}.py`, and the closure is exactly `src/transfer/` plus `src/__init__.py` — verified by the H200 controller's own freeze walker. `src/revision/`, `src/models/`, `src/training/` and `src/analysis/` are archived.

Method-family coverage — which families have been tested and where each one fails — is the audit document §6.

## Measurement package

Library `src/transfer/`, entry points `scripts/transfer/`:

| entry point | stage | what it establishes |
|---|---|---|
| `01_cohort_power.py` | prerequisite | per-arm context-derived information on a frozen cohort; arms below threshold are reported **unmeasurable**, not failing |
| `02_pathway_budget.py` | 1 | share of next-token computation carried by the MLP pathway versus the attention pathway |
| `03_estimand_power.py` | prerequisite | which ablation estimands have enough causal footprint to support a recovery gate at all |
| `04_circuit_primitives.py` | 1, 4 | induction/copying head census, direct logit attribution, activation-patching map |
| `05_relational_channel.py` | 3 | whether residue-pair structure is readable from per-position states or only from the attention pattern |
| `06_explanation_channel.py` | 3 | bits of explanation available per symbol from an annotation channel, and the analytic ceiling on event-selection designs |
| `07_convergence_control.py` | 1 | separates modality from convergence, scale and tokenisation |
| `08_lens_family.py` | 3 | logit lens, tuned lens, J-Lens; the output-aperture rank the lens family is bounded by |
| `09_probe_and_erasure.py` | 3 | decodability (probe skill) against reliance (LEACE concept erasure) |
| `10_homology_control.py` | 1 | whether the protein induction signal is computation or retrieval of a memorised corpus entry |
| `11_induction_path_patching.py` | 4 | how much of the induction heads' logit effect is written directly |
| `12_induction_robustness.py` | 1 | threshold robustness and scale/modality separation of the census, from artefacts on disk; no GPU |
| `13_induction_probe_bootstrap.py` | 1 | probe-cluster bootstrap of the induction head fraction |
| `14_paa_census.py` | 1 | go/no-go gate for a prediction-addressed-attention census — the second mechanism (copy suppression) |

`scripts/transfer/panel_contract.py` is the **single declaration** of which arms each stage may run and why not. `arm_can_run(stage, arm)` is the predicate; `panel_contract.sh` is its generated bash rendering, sourced by the controller and the worker, verified in preflight and by the test suite. Regenerate with `--emit` after any edit to `src/transfer/arms.py`. Never hand-write an arm list.

## Literature gate

**A targeted literature search is a required gate before a measurement track is designed, and again before a formal campaign is launched. The search must name the specific mechanism being ported, not merely the domain being ported into.**

This exists because it has already failed once. The opening sweep searched protein-language-model interpretability and found ProGenMech, InterPLM and the sparse-autoencoder limitations corpus; an induction-head track was designed and run on that basis. Searching "induction heads protein language model" would have returned Pomerants et al., arXiv:2602.23179, which establishes induction heads in protein LMs, counts them, and shows that approximate-repeat detection subsumes exact-repeat detection — a result that both anticipates part of the finding and calls the probe design into question.

Record, in the `EXP-R2-NNN` entry for the track and before it is built: the queries run, the works found, what each already establishes, and what the track adds beyond them.

## Evidence discipline

These rules exist because this project has twice set a numerical gate whose attainability was never checked: a 0.1-nat mutual-information gate against a design with a 0.0066-nat analytic ceiling (L2), and an 80%-recovery gate against an estimand whose causal footprint is of order 0.02 nats/token (L1).

1. **Attainability before application.** No gate is applied to a protein arm until it has been shown attainable on the text control under the same procedure. A gate the text control cannot pass is a specification defect.
2. **Power before scoring.** No recovery ratio is reported for an arm whose cohort context-information is below threshold, or whose denominator fails its guard. Such an arm is reported as unmeasurable, not as having failed.
3. **Denominator provenance is a first-class parameter.** Ablation-baseline choice moves a measured denominator substantially; the choice is recorded in every output and its sensitivity is reported.
4. **Seeds and intervals or nothing.** No single-seed point estimate enters a conclusion. Sequence-cluster bootstrap intervals are the default.
5. **Exploratory results stay labelled.** Pilot measurements are descriptive and motivate hypotheses; they do not establish that an assumption holds, fails, or causes an observed gap.
6. **A limitation is scoped before it is reported.** One demonstrated on the text control is a property of the **method**; one appearing only on protein arms is a property of the **transfer**. The distinction is the spine of the catalogue and is not optional.

Appendix B of the audit document holds the further standing rules — twenty-eight entries numbered to 27, because rule 15b was inserted beside 15 rather than renumbering a set that is cited by number elsewhere. Each is earned by a specific failure here. Read it before touching measurement code.

## Compute policy

Sizes below are driver-reported, not vendor nominal.

- **L20 (local, 8 x 45 GiB reported).** Validation only: small cohorts, short runs, interface and sanity checks, CPU-only stages. Check `nvidia-smi` before allocating; the machine is shared.
- **H200 (remote, 4 x 140 GiB reported in-pod).** All full-scale campaigns. Launched via `scripts/transfer/run_transfer_h200.sh`; see `scripts/transfer/README.md`.

Capacity must be confirmed with `nvidia-smi` **inside the pod**. The cluster health check reports allocation across the whole cluster and routinely reads 100% while our own cards sit idle, because every card is assigned to some pod including ours. Treating that figure as our availability would wrongly defer a campaign that could have started immediately.

## Deliverables

1. A stage-resolved characterisation of where text-to-protein interpretability transfer loses performance, with a matched text control at every stage, and with each limitation scoped as method-intrinsic or transfer-specific.
2. A reusable, gated measurement package with attainability and power checks built in.
3. Whichever constructive result the catalogued limitations earn. No proposal is advanced that is not traceable to a catalogued limitation; the three best-supported openings are in the audit document §8.

Total remaining budget, phased and costed, is the audit document §9. That document is the single figure; do not restate it here.

## Canonical materials

- Findings, retractions, catalogue, plan: `docs/INTERPRETABILITY_TRANSFER_AUDIT.md`
- Methods, per-stage estimand and metric: `docs/methods/TRANSFER_MEASUREMENT_PROGRAMME.md`
- Chronological record: `docs/EXPERIMENT_LOG.md`
- Dictionary receipts behind L1 and C1: `docs/analysis/P0_2B_DICTIONARY_FIDELITY_RESULTS_20260727.md`, `evidence/p0_2_adjudication_20260727/`, `evidence/p0_2b_fidelity_20260727/`
- Retired scope, frozen: `/Data2/lzp/bio_archive/legacy/r2_retired_scope_20260729/`
- Superseded and frozen, do not cite: repository-root `check.md`
