# Research Plan

**Updated:** 2026-08-10 **Status:** active implementation guide **Authority:** `docs/INTERPRETABILITY_TRANSFER_AUDIT.md` defines the current scientific plan as well as findings, retractions, limitations, and claim status. This document translates its admitted directions into comparisons, execution order, and stage inventory; it introduces no independent scientific claim or admission.

## Research Design

The programme follows three directions in order:

1. **Compare model families from first principles.** Separate effects of tokenization, training data, architecture, and modality across pure-text, pure-protein, and joint language–protein generative models.
2. **Use and audit interpretability methods.** Apply each method as a controlled measurement tool and determine whether a failure belongs to the method itself or to transfer across models and modalities.
3. **Develop and validate methods for biological knowledge.** Propose and validate interpretability methods that distinguish knowledge encoded by protein-generative models from retrieval and surface correlation. Require causal, retrieval-aware, and independent biological validation before using a method to formulate or test a new biological hypothesis.

The existing pure-text and pure-protein campaign is the inherited baseline, not the whole programme. Its frozen results are not rerun merely to expand coverage. New experiments select the smallest comparison that can distinguish a declared hypothesis; the project does not attempt a methods × models Cartesian product.

## Comparison Hierarchy

Prefer comparisons in this order because each lower row leaves more possible explanations:

| Comparison | What it helps identify | Remaining limitation | Availability |
|---|---|---|---|
| Same joint checkpoint, text mode versus protein mode | Modality under shared weights and most of the architecture | Tokenizer, prompt, output space, and training exposure may still differ | **ProLLaMA Stage 1 and Stage 2 qualified in both modes** (EXP-R2-152); method experiments not yet run |
| Matched checkpoint lineage | When a capability or mechanism appears across training stages | A released stage usually changes a package of data and objectives, not one factor | active |
| Shape-matched pure text and protein models | Modality while holding major architectural dimensions fixed | Training data and optimization remain different | in use |
| Dense versus MoE within one modality | Whether routing or expert structure explains a measured failure | Model family and training data must still be controlled | in use |
| Cross-lab or cross-architecture replication | Whether a result generalizes beyond one lineage | It is not a clean causal attribution by itself | in use |

**The first row now has a qualified bridge, but its controls still determine what it can identify.** Galactica and InstructProtein were refused on opposite modes at identical architecture and scale; the ProLLaMA matched lineage then supplied two checkpoints whose text and protein modes are both measurable. These checkpoints permit same-weight mode comparisons and same-input model diffing across training stages, but they do not control tokenizer, rendering, training exposure, or the biological meaning of an activation. Add a joint MoE only if the qualified dense comparison leaves routing or expert specialization as a live explanation; no joint MoE is currently scheduled. Encoder–decoder, protein-encoder-plus-LLM, or cross-attention systems enter only when a hypothesis depends on their component boundary and the intervention target has been defined for that architecture.

A joint checkpoint's qualification is a real gate and not a formality. Establish the rendering against the model's own likelihood before reading any model property, and report an arm below the context-information threshold as unmeasurable on that cohort rather than as failing.

## Execution Order

### 1. Qualify the model and interface

Strictly load the checkpoint, render each mode as trained, identify scored positions, verify output semantics, and run a fixed likelihood self-check. An unavailable or unverifiable checkpoint is reported as unavailable; it is not replaced silently.

### 2. Establish a measurable comparison

Define the unit being compared—token, character, residue, sequence, attention head, expert, or concept—and show that the target effect is attainable and statistically powered. Distances and windows must use a content unit that remains meaningful across tokenizers.

Check that the manipulation is *available* on every arm it is claimed for, not only that the arm is eligible for the stage. A conditioned arm whose scored span is delimited by start and end tokens cannot take any manipulation that truncates the sequence: truncation removes the closing boundary and destroys the definition of what is scored, so the stage refuses rather than scoring a partial span. That exclusion is structural and no budget removes it; report the arm as excluded from that manipulation and state which arms the resulting matrix therefore covers.

### 3. Apply the minimum informative method

Use local Lens, Probe, Erasure, or Patching measurements before a costly replacement model when they can answer the hypothesis. PAA is judged by causal top-k retrieval against chance and a depth-only baseline, not by the withdrawn all-grid rank statistic. Replacement models are judged by behavior after sequential replacement; reconstruction quality remains a diagnostic rather than proof of fidelity.

### 4. Test stability and attribution

Use negative controls, an appropriate positive control, independent seeds, and family-disjoint data. A result from a joint checkpoint does not strengthen or retract an existing pure-model claim unless the comparison directly identifies the same quantity.

### 4b. Prefer a training-free control to a new campaign

A literature review on 2026-08-10 reordered the programme around one observation: the cheapest decisive experiments available were controls that require no training at all — a neuron-basis baseline, a norm- and angle-matched perturbation null, a collision null for a token-space census, and a re-reading of admitted artefacts on an unnormalised scale. One of those re-readings refuted an alternative explanation of the programme's largest open question at no compute cost. Before proposing a campaign, establish that no training-free control answers the same question, and state explicitly which control was considered and why it does not suffice.

Two consequences carry into how results are reported. Report the **numerator as well as the ratio**: a recovery fraction whose denominator differs several-fold across arms hides whichever direction the reader does not check, and the unnormalised quantity is often the stronger statement. And **score a selector against the trivial baseline available from its own coordinates and against its own arm's chance rate**, never against a uniform baseline, when arms differ in alphabet size — a uniform reference does not correct for symbol collision and a token-space score is diluted by it.

### 5. Develop and validate a method for biological knowledge

The currently motivated candidates are a concept-aligned Lens and three distinct alignment experiments. D3.h is standard Model Diffing: compare matched ProLLaMA checkpoints on identical inputs, separately within each mode, and stop if simpler baselines answer the question before a Crosscoder is needed. **Run base → stage 1 before stage 1 → stage 2, on the evidence.** The two adapted stages are indistinguishable on the estimand that qualified them — their difference sits inside the cohort-draw spread in both modes — so a feature-level difference found between them would have no behavioural anchor. Base → stage 1 moves the text mode by more than twenty draw spreads and takes the protein mode from unmeasurable to measurable, and is the comparison that can establish whether the method resolves anything on this lineage at all. Its own limitation travels with it: the base model's protein mode is unmeasurable, so on that side the base is a pre-adaptation representational reference, not a behavioural control. D3.g is language-mediated concept alignment: use genuine sequence–description pairs only after sequence- or concept-level aggregation; text and protein token positions are not directly paired. D3.i is gated and unranked: it receives no implementation or compute until D3.g succeeds and both a biologically grounded protein circuit and a language-side circuit pass their own causal-faithfulness gates. Jacobians may nominate edge correspondences, but edge/path interventions must validate them in both original models; cycle consistency is necessary and not sufficient. The authoritative admission and stopping rules are in §8 of the audit.

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
3. A causally validated method that can distinguish encoded biological knowledge from retrieval or surface correlation.
4. A new biological claim only where that method predicts independent evidence beyond homology retrieval.
