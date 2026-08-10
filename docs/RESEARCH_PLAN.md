# Research Plan

**Updated:** 2026-08-10 **Status:** active implementation guide **Authority:** `docs/INTERPRETABILITY_TRANSFER_AUDIT.md` defines the current scientific plan as well as findings, retractions, limitations, and claim status. This document translates its admitted directions into comparisons, execution order, and stage inventory; it introduces no independent scientific claim or admission.

## Research Design

The programme follows three directions in order:

1. **Compare model families from first principles.** Separate effects of tokenization, training data, architecture, and modality across pure-text, pure-protein, and joint language–protein generative models.
2. **Use and audit interpretability methods.** Apply each method as a controlled measurement tool and determine whether a failure belongs to the method itself or to transfer across models and modalities.
3. **Develop adapted methods.** Build a new method only after a reproducible failure has been localized, then require causal and independent-data validation before using it for biological claims.

The existing pure-text and pure-protein campaign is the inherited baseline, not the whole programme. Its frozen results are not rerun merely to expand coverage. New experiments select the smallest comparison that can distinguish a declared hypothesis; the project does not attempt a methods × models Cartesian product.

## Comparison Hierarchy

Prefer comparisons in this order because each lower row leaves more possible explanations:

| Comparison | What it helps identify | Remaining limitation | Availability |
|---|---|---|---|
| Same joint checkpoint, text mode versus protein mode | Modality under shared weights and most of the architecture | Tokenizer, prompt, output space, and training exposure may still differ | **none qualified** (EXP-R2-151) |
| Matched checkpoint lineage | When a capability or mechanism appears across training stages | A released stage usually changes a package of data and objectives, not one factor | active |
| Shape-matched pure text and protein models | Modality while holding major architectural dimensions fixed | Training data and optimization remain different | in use |
| Dense versus MoE within one modality | Whether routing or expert structure explains a measured failure | Model family and training data must still be controlled | in use |
| Cross-lab or cross-architecture replication | Whether a result generalizes beyond one lineage | It is not a clean causal attribution by itself | in use |

**The first row is currently empty, and that changes the order of work rather than the hierarchy.** Both reachable dense joint decoders were qualified and refused on opposite modes at identical architecture and scale: one carries text and an unmeasurable protein mode, the other protein and a collapsed text mode. Until a checkpoint qualifies in both modes, joint-model work proceeds on the second row — a matched lineage whose stages differ in protein adaptation — and any joint result must state which mode was measurable. Add a joint MoE only if a qualified dense comparison leaves routing or expert specialization as a live explanation; there is no such comparison yet, so a joint MoE is not scheduled. Encoder–decoder, protein-encoder-plus-LLM, or cross-attention systems enter only when a hypothesis depends on their component boundary and the intervention target has been defined for that architecture.

A joint checkpoint's qualification is a real gate and not a formality. Establish the rendering against the model's own likelihood before reading any model property, and report an arm below the context-information threshold as unmeasurable on that cohort rather than as failing.

## Execution Order

### 1. Qualify the model and interface

Strictly load the checkpoint, render each mode as trained, identify scored positions, verify output semantics, and run a fixed likelihood self-check. An unavailable or unverifiable checkpoint is reported as unavailable; it is not replaced silently.

### 2. Establish a measurable comparison

Define the unit being compared—token, character, residue, sequence, attention head, expert, or concept—and show that the target effect is attainable and statistically powered. Distances and windows must use a content unit that remains meaningful across tokenizers.

### 3. Apply the minimum informative method

Use local Lens, Probe, Erasure, or Patching measurements before a costly replacement model when they can answer the hypothesis. PAA is judged by causal top-k retrieval against chance and a depth-only baseline, not by the withdrawn all-grid rank statistic. Replacement models are judged by behavior after sequential replacement; reconstruction quality remains a diagnostic rather than proof of fidelity.

### 4. Test stability and attribution

Use negative controls, an appropriate positive control, independent seeds, and family-disjoint data. A result from a joint checkpoint does not strengthen or retract an existing pure-model claim unless the comparison directly identifies the same quantity.

### 5. Admit an adapted method or biological study

The currently motivated candidates are a concept-aligned Lens for incompatible output spaces and language-mediated causal concept alignment using real sequence–description pairs. Simple retrieval, affine, orthogonal, shuffled-pair, and rank-matched baselines come before an Adapter MLP. Biological claims additionally require external phenotype or functional evidence; linguistic plausibility and model scores are insufficient.

## Measurement Stages

The executable registered panel is declared only by `scripts/transfer/panel_contract.py`. This inventory records the purpose and status of the current entry points so code and research design remain connected; detailed metrics for the registered panel stages are in `docs/MEASUREMENTS.md`.

| Entry point | Role |
|---|---|
| `01_cohort_power.py` | Context-information qualification |
| `02_pathway_budget.py` | MLP and attention pathway contribution |
| `03_estimand_power.py` | Attainability and power qualification |
| `04_circuit_primitives.py` | Circuit census, attribution, and patching primitives |
| `05_relational_channel.py` | Relational information in states versus attention |
| `06_explanation_channel.py` | Explanation-channel capacity |
| `07_convergence_control.py` | Scale, convergence, and tokenizer controls |
| `08_lens_family.py` | Logit, tuned, and output-aperture Lens measurements |
| `09_probe_and_erasure.py` | Decodability versus behavioral reliance |
| `10_homology_control.py` | Homology and memorization controls |
| `11_induction_path_patching.py` | Causal path of induction heads |
| `12_induction_robustness.py` | Threshold and scale robustness |
| `13_induction_probe_bootstrap.py` | Probe-cluster uncertainty |
| `14_paa_census.py` | PAA selection and exhaustive causal labels |
| `15_replacement_faithfulness.py` | Behavioral and diagnostic causal checks for replacement models |
| `16_fitness_recovery.py` | Fitness baseline and recovery interface |
| `17_train_transcoder.py` | Local PLT and CLT training controls |
| `18_das_subspace.py` | Retained negative-design record; closed and not scheduled |
| `19_routing_locality.py` | Router, expert-set, and boundary diagnostics |
| `20_retrieval_bound.py` | Model fitness versus training-corpus profile retrieval |
| `21_joint_mode_qualification.py` | Joint text/protein mode qualification for a non-panel checkpoint |

Stages outside the registered panel use their dedicated launcher or a direct validated invocation as documented in `scripts/transfer/README.md`. A new joint-model adapter is admitted to the executable contract only after its interface and negative paths are tested.

## Evidence Discipline

1. Check interface validity before interpreting a number.
2. Check attainability, statistical power, and denominator validity before applying a gate.
3. Use a text positive control where a real analogue exists; otherwise define an internal positive control.
4. Match independent units, content windows, training budgets, and baseline capacity to the claim being made.
5. Record seeds and uncertainty; a pilot or single draw remains exploratory.
6. Scope each limitation as method-intrinsic or transfer-specific, then localize transfer-specific failures to model, training data or stage, modality, or evaluation interface.
7. Record every experiment, including an explicit failure or unavailable interface, in `docs/EXPERIMENT_LOG.md`.

Appendix B of the audit contains the failure-earned standing rules and must be read before changing measurement code.

## Literature and Compute

Search the current literature for the specific mechanism before designing a track and again before launching a formal campaign. Record what the new experiment distinguishes from prior work.

Use B only for interface validation and small correctness checks. Run full campaigns on H200, confirm the allocation and free memory inside the selected pod, and follow `scripts/transfer/README.md`. Do not copy temporary GPU counts, pod names, or storage paths into this plan.

## Deliverables

1. Identified differences in model foundations and causal computation, with remaining confounds stated.
2. A method-transfer map whose failures are scoped and supported by valid controls.
3. An adapted method only where the preceding evidence earns it.
4. Biological or novelty claims only where a validated mechanism predicts external evidence beyond homology retrieval.
