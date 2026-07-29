# Independent assessment of the manuscript for *npj Artificial Intelligence*

## Executive verdict

### Conclusion 1 — Is the manuscript currently eligible for submission?

**The study is clearly within the topical scope of *npj Artificial Intelligence*, but the present manuscript is not yet scientifically ready for a competitive Article submission.**

The distinction matters:

- **Scope eligibility: yes.** The journal seeks transformative and interdisciplinary AI research, including work that bridges theoretical advances and practical scientific applications and reframes established research problems. A study auditing mechanistic-interpretability claims in biological foundation models fits that remit well. ([nature.com](https://www.nature.com/npjai/aims))
- **Article-scale workload: yes.** The journal describes an Article as a substantial, technically complex study involving several approaches. Your combination of sparse-dictionary training, cross-model matching, semantic analysis, probing, generation, feature interventions and attention interventions satisfies the workload and structural expectations of an Article. ([nature.com](https://www.nature.com/npjai/content-types))
- **Current evidential readiness: no.** The central recurrence result is demonstrably procedure- and cohort-sensitive; the dictionary-quality evidence is not mask-aware or properly replicated; the main null is not the most defensible coherent null; the eight-class steering experiment contains a feature-sign selection defect; several discovery and evaluation stages overlap; and exact reproduction is presently impossible because central cohorts, revisions and checkpoints are not fully archived.

My recommendation is therefore:

> **Do not submit the present version. Treat it as a major-revision-before-submission manuscript.**

This judgment is not based on the absence of positive steering or causal effects. A rigorous negative-results paper could be highly suitable for *npj Artificial Intelligence*. The problem is that some of the current negative findings do not yet distinguish “no mechanism” from “an insufficient or defective measurement and intervention pipeline.”

I reviewed the main manuscript and supplementary information in detail. fileciteturn0file1 fileciteturn0file0 I also treated the attached [experimental procedures and results audit](sandbox:/mnt/data/EXPERIMENTAL_PROCEDURES_AND_RESULTS_AUDIT_20260716.md) as an important provenance layer rather than simply accepting the narrative in the manuscript.

---

# 1. Literature review

## 1.1 Review scope

The literature review was conducted through **16 July 2026** and focused on five connected areas:

1. Formal standards for explainability and mechanistic interpretability.
2. Sparse representations, semantic alignment and causal localization.
3. Sparse-autoencoder interpretation of protein and genomic models.
4. Protein representation quality and out-of-distribution generalization.
5. Controllable protein generation and biological validation.

The most relevant publication families were *npj Artificial Intelligence*, *npj Systems Biology and Applications*, *npj Drug Discovery*, *Nature*, *Nature Machine Intelligence*, *Nature Methods*, *Nature Biotechnology* and *Nature Communications*.

## 1.2 Literature landscape and the evidential bar it creates

| Literature strand | Representative Nature/npj work | Main lesson for this manuscript |
|---|---|---|
| **Faithfulness, human alignment and formal correctness** | Lee et al. integrate faithfulness and human alignment into the model rather than relying only on post-hoc explanation; Haufe et al. argue that explainability claims require explicit formalization; Pesapane et al. advocate preregistered causal and invariance tests rather than explanation alone. ([nature.com](https://www.nature.com/articles/s44387-025-00023-9)) | Your recurrence–association–intervention ladder is intellectually aligned with this literature. However, the formal estimands, independent validation sets and prospective acceptance criteria remain incomplete in implementation. |
| **Robustness and identifiability of interpretations** | Esser-Skala and Fortelny show that biological interpretations can vary across repeated training and inherit knowledge biases, and that control experiments are necessary. Klindt et al. organize sparse interpretability around identifiability, disentangling by sparse coding and behavioral assessment. ([nature.com](https://www.nature.com/articles/s41540-023-00310-8)) | Cross-model recurrence is not enough. Stability across dictionary seeds, cohorts, matching methods and model checkpoints must be estimated explicitly. |
| **Semantic component analysis and causal relevance** | Concept Relevance Propagation links concepts to prediction relevance; SemanticLens maps internal components into semantic spaces and couples semantic descriptions to component-level attribution; Wu et al. combine sparse-pattern discovery with targeted perturbation that substantially impairs the corresponding behavior. ([nature.com](https://www.nature.com/articles/s42256-023-00711-8)) | Your manuscript is stronger than simple activation visualization because it attempts interventions. It is weaker than these exemplars because the intervention pipeline has not yet demonstrated sensitivity to a known positive mechanism or a robust effect beyond matched controls. |
| **Sparse interpretation of biological foundation models** | InterPLM reports thousands of ESM-2 SAE features associated with binding sites, motifs and functional domains, and demonstrates annotation recovery and targeted steering. Evo 2 includes SAE-based genomic feature exploration, releases model parameters, training data and code, and experimentally validates controlled genomic designs. ([nature.com](https://www.nature.com/articles/s41592-025-02836-7)) | Your strongest comparative advantage is skepticism: it explicitly tests whether recurrence really supports semantic or causal claims. Its disadvantage is that it currently supplies neither an independently validated semantic discovery nor a fully interpretable negative causal result. |
| **Representation quality and realistic generalization** | KinForm evaluates standard and sequence-identity-aware cross-validation and shows that conclusions change with similarity to the training distribution. METL combines biophysical simulations with experimental sequence–function data and demonstrates design from small datasets. Recent Nature Methods studies also quantify representation uncertainty and compare or consolidate protein model knowledge. ([nature.com](https://www.nature.com/articles/s41540-026-00692-5)) | Family-disjoint, identity-aware and uncertainty-aware evaluation is becoming expected. Your family-disjoint EC result is valuable, but layer selection, fold construction, seed repetition and confidence estimates need stronger implementation. |
| **Protein generation and design validation** | ProtGPT2 established natural-like de novo sequence generation; ProGen demonstrated functional sequences across families; RFdiffusion demonstrated de novo structure and function design. Johnson et al. calibrated computational metrics against experimentally measured enzyme activity and showed that no individual computational metric was universally sufficient. Closed-loop PLM/biofoundry and autonomous enzyme-engineering platforms now connect model proposals to laboratory testing. ([nature.com](https://www.nature.com/articles/s41467-022-32007-7)) | Pfam, CLEAN, pLDDT and Foldseek are useful filters, but they are not independent proof of function. Computational-only steering claims must at least use calibrated, symmetric, generation-wide endpoints; biological design claims increasingly require experimental validation. |
| **Controllable protein design as a field** | The recent *npj Drug Discovery* survey defines controllable protein generation as sampling under explicit structural, functional or higher-level constraints and identifies evaluation as a central unresolved challenge. ([nature.com](https://www.nature.com/articles/s44386-026-00054-5)) | The current motif/composition heuristic is below the evidential standard needed to claim EC control. A validated constraint-satisfaction endpoint is required. |

## 1.3 Overall synthesis of the literature

The literature has moved beyond asking whether an internal feature is easily nameable. The current evidential hierarchy is approximately:

\[
\text{detectable}
\;\rightarrow\;
\text{stable}
\;\rightarrow\;
\text{semantically specific}
\;\rightarrow\;
\text{used by the model}
\;\rightarrow\;
\text{causally manipulable}
\;\rightarrow\;
\text{externally useful}.
\]

Your manuscript correctly argues that these levels cannot substitute for one another. Figure 1 on page 3 is therefore the manuscript’s strongest conceptual contribution: it explicitly separates recurrence, association and intervention, with a separate recoverability audit. fileciteturn8file15

The difficulty is that the current implementation securely reaches only the first level:

> **Sparse activation profiles recur under one particular cohort, layer map, variance filter, greedy matcher and pair/layer-specific assignment null.**

The semantic and causal levels remain either exploratory or inconclusive for methodological reasons.

---

# 2. Horizontal comparative analysis

## 2.1 Comparison with representative Nature/npj publications

| Comparator | Strongest evidence in that work | Where the present manuscript is stronger | Where the present manuscript is weaker |
|---|---|---|---|
| **Lee et al., npj AI 2025** | Explanation components are integrated into prediction, with faithfulness and human alignment treated jointly. ([nature.com](https://www.nature.com/articles/s44387-025-00023-9)) | More explicit distinction between semantic recurrence and causal control. | Sparse features are post-hoc readouts and are not shown to participate faithfully in the corresponding model decisions. |
| **Haufe et al., npj AI 2026; Pesapane et al., npj AI 2026** | Explicit call for formal estimands, testability, prospective controls and causal/invariance trials. ([nature.com](https://www.nature.com/articles/s44387-026-00095-1)) | The manuscript operationalizes an evidence ladder and records failed gates rather than hiding them. | Several gates were amended after results; important tests remain post hoc, under-replicated or based on overlapping discovery/evaluation data. |
| **Esser-Skala & Fortelny, npj Systems Biology 2023** | Interpretation robustness assessed across repeated training and bias controls. ([nature.com](https://www.nature.com/articles/s41540-023-00310-8)) | Cross-model rather than only within-model recurrence is investigated. | Dictionary seeds are not repeated, and cohort/matcher dependence is not quantified as a distribution. |
| **Wu et al., npj AI 2025** | Perturbing an extremely sparse parameter subset materially degrades the targeted capability. ([nature.com](https://www.nature.com/articles/s44387-025-00031-9)) | Uses multiple intervention families and matched random controls. | No intervention passes its gate; pipeline sensitivity to a real positive circuit is not demonstrated. |
| **CRP and SemanticLens, Nature Machine Intelligence** | Scalable concept maps are connected to component relevance and decision behavior. ([nature.com](https://www.nature.com/articles/s42256-023-00711-8)) | More cautious about assigning human-readable labels to recurrent features. | Semantic characterization is based largely on selected top events and low-level covariates, with no scalable validated semantic atlas. |
| **InterPLM, Nature Methods 2025** | Thousands of biologically interpretable SAE features, annotation recovery and targeted sequence steering. ([nature.com](https://www.nature.com/articles/s41592-025-02836-7)) | Tests whether apparently conserved features survive stronger confounder and intervention audits. | Dictionary quality is substantially weaker; semantic positives do not survive the matched audit; steering does not produce a validated benefit. |
| **Evo 2, Nature 2026** | Large-scale open biological model, SAE exploration tools, released data/code/weights and experimentally validated controlled designs. ([nature.com](https://www.nature.com/articles/s41586-026-10176-5)) | More focused analysis of when sparse-feature claims fail. | Much smaller and older model panel, incomplete release provenance, and no experimental functional validation. |
| **Johnson et al., Nature Biotechnology 2025** | Computational scores calibrated against measured expression, folding and enzyme activity; composite metrics validated experimentally. ([nature.com](https://www.nature.com/articles/s41587-024-02214-2)) | Adds mechanistic interventions rather than only endpoint scoring. | Pfam/CLEAN/structure predictions are not calibrated against activity for the generated samples and are partly based on selected, asymmetric subsets. |
| **KinForm and METL** | Identity-aware generalization, matched baselines and experimentally relevant property prediction/design. ([nature.com](https://www.nature.com/articles/s41540-026-00692-5)) | Examines internal mechanism rather than only predictive performance. | Family leakage, layer selection and representation comparisons are not yet handled with equally strong nested, identity-aware evaluation. |

## 2.2 Distinctive contribution that should be preserved

The manuscript has four genuine strengths:

1. **It asks a neglected question.** Most sparse-interpretability studies search for compelling features; this study asks whether cross-model recurrence is actually evidentially sufficient.
2. **It exposes a mathematically invalid original gate.** The original 0.1-nat mutual-information threshold exceeded the maximum possible event entropy by approximately fifteenfold. Correcting this rather than burying it is scientifically commendable. fileciteturn8file1
3. **It reports negative interventions in full.** Figure 4 presents the target and random-control effects rather than reporting only selected maxima. fileciteturn7file16
4. **It already uses appropriately bounded language.** The current manuscript largely avoids claiming universal features, an attention sink, successful steering or absence of biological information.

Those strengths make the study worth rescuing. They do not, by themselves, make the current evidence sufficient for publication.

---

# 3. In-depth vertical analysis

## 3.1 Research question and positioning

### Strength

The question—whether conserved sparse readouts warrant semantic and causal claims—is important, general and timely. The “recurrence does not imply semantics or control” thesis is broader than protein modeling and could become a general contribution to AI interpretability.

### Limitation

The present evidence supports a narrower statement:

> A particular sparse-feature matching procedure finds recurrent activation profiles in three protein generators, but the tested semantic and intervention analyses do not validate stronger interpretations.

That is scientifically valid, but it may appear to editors as a narrow case-study audit unless the paper produces a reusable methodology, benchmark or formal evaluation framework.

### Required repositioning

The main contribution should become:

> **A prospective, evidence-calibrated benchmark for evaluating conserved sparse representations.**

Protein language models should be presented as the principal case study rather than the only reason the paper matters.

---

## 3.2 Model panel and claims of cross-architecture conservation

The study includes ProtGPT2, ZymCTRL and ProGen2-medium. However, ProtGPT2 and ZymCTRL are both GPT-2-family, 36-layer models; ProGen2-medium provides the main architectural contrast. The CLT input is also not homologous across all three models: it is a layer-normalized post-attention MLP input in ProtGPT2 and ZymCTRL, but a shared layer-normalized block input in ProGen2-medium. fileciteturn8file5

Therefore:

- “three-model recurrence” is justified;
- “architecture-independent conservation” would not be;
- model-family, tokenizer and training-data effects are presently entangled.

A stronger panel should factorially separate:

\[
\text{architecture}
\times
\text{training corpus}
\times
\text{tokenizer}
\times
\text{dictionary seed}.
\]

At minimum, add either an encoder-style PLM and another independent autoregressive family, or train multiple smaller controlled models for which architecture and training corpus can be varied independently.

---

## 3.3 Sparse dictionary quality

The reference dictionaries have approximately:

- ProtGPT2: FVU 0.201 and dead fraction 0.620;
- ZymCTRL: FVU 0.331 and dead fraction 0.668;
- ProGen2-medium: FVU approximately 0.33 and dead fraction approximately 0.57.

Moreover, ProtGPT2 and ZymCTRL quick evaluations included right-padding positions, and the wider dictionaries all failed the preregistered FVU criterion. fileciteturn8file17turn7file0

This is a major interpretive problem because every downstream negative has at least four possible explanations:

\[
\begin{aligned}
&\text{the model lacks the relevant mechanism},\\
&\text{the dictionary failed to recover it},\\
&\text{the matching method failed to align it},\\
&\text{the intervention site did not control it}.
\end{aligned}
\]

The current wider-dictionary experiment does not distinguish these explanations because its quality comparison is unmatched and the wider runs still miss the predefined quality gate.

### Required work

- Make all training and evaluation mask-aware.
- Retrain at least three, preferably five, seeds per model.
- Use immutable train/validation/test sequence manifests.
- Predefine a held-out quality gate before atlas construction.
- Report FVU by layer, dead fraction, frequency distribution, decoder-norm distribution and reconstruction-error quantiles.
- Compare TopK CLTs with at least two alternatives, such as conventional SAE/JumpReLU or gated SAE and a dense low-rank baseline.
- Do not perform the biological atlas on dictionaries that fail the quality gate.

---

## 3.4 Conserved-triplet atlas

### What is currently supported

The manuscript finds 38 exact triplets at \(|r|\geq0.90\), 30 at \(|r|\geq0.95\) and eight at \(|r|\geq0.98\), versus a maximum of one in 30 implemented null replicates. fileciteturn8file16 This is a clear and potentially valuable result.

### Principal threats

#### A. Discovery and evaluation use the same cohort

The fixed balanced 200-sequence cohort determines feature variance, pairwise matching and triplet enumeration. There is no independent cohort on which the discovered correspondences are confirmed.

#### B. Strong cohort sensitivity

The experimental audit reports that a separate 500-sequence UniRef50 cohort recovered only eight triplets at 0.90 and three at 0.95, with no exact feature-identity overlap with the canonical 38. This does not invalidate the canonical result, but it changes its interpretation from “conserved features” to “cohort-dependent recurrent readouts.” See the [audit attachment](sandbox:/mnt/data/EXPERIMENTAL_PROCEDURES_AND_RESULTS_AUDIT_20260716.md).

#### C. The null is not coherent at the model level

The audit reports that each model-pair/layer edge uses a separate random sequence reassignment. Consequently, the three edges used to form a null triangle may arise from different random assignments. That can suppress transitive null triplets more strongly than a single coherent model-wise permutation.

The defensible comparison is therefore:

> Observed triplets versus the implemented pair/layer-specific reassignment null,

not yet:

> Observed triplets versus the null distribution expected under absence of cross-model correspondence.

#### D. Greedy pairwise matching plus triangle closure is algorithm-dependent

The method first selects pairwise top-300 greedy one-to-one matches and then requires triangular closure. The result can change with:

- pair ordering;
- greedy tie resolution;
- feature-pool size;
- variance threshold;
- layer mapping;
- correlation threshold.

A global three-way alignment or hypergraph matching objective would be more defensible.

#### E. Absolute correlation needs justification

The sparse activations are nonnegative, but their across-sequence profiles can still be negatively correlated. A feature that activates when another feature is inactive is not automatically the same conserved concept. Unless essentially all retained edges are positive, matching on \(|r|\) can combine co-activation with complementary or mutually exclusive behavior.

### Required atlas experiment

For every model seed and held-out cohort:

1. Discover alignments on cohort A.
2. Freeze feature identities and matching.
3. Evaluate signed correlations on cohort B.
4. Repeat with positive-only correlation.
5. Use a coherent model-wise sequence permutation with at least 1,000 replicates.
6. Compare greedy matching, Hungarian assignment, optimal transport and a joint three-way method.
7. Report triplet count, held-out correlation, identity Jaccard and matching confidence.
8. Repeat across feature-pool sizes, layer mappings and thresholds.
9. Estimate variance attributable to cohort, model seed, dictionary seed and matching method.

Without this experiment, the 38-triplet count remains a descriptive outcome rather than a reproducible population quantity.

---

## 3.5 Semantic interpretation

Correcting the impossible MI threshold is one of the manuscript’s best decisions. The re-audit finds 224/380 significant associations under a global position null but 0/380 after a within-protein, amino-acid- and position-matched null. fileciteturn8file1

However, 0/380 should not be interpreted as evidence that the triplets contain no biological information because:

- only the top 100 events per triplet were retained;
- activation magnitude outside those events was discarded;
- the analysis was constructed after the invalid gate was discovered;
- dominant Pfam labels were closely tied to protein identity;
- power under the matched null is not established.

The current conclusion—“no corrected matched-null association in the saved top-event analysis”—is appropriately bounded.

### Stronger semantic experiment

Use continuous activation values and fit, for each feature or aligned feature set, a hierarchical conditional model such as:

\[
a_{ip}
=
f(\text{position}_{ip})
+
g(\text{k-mer}_{ip})
+
\beta_1 \Vert x_{ip}\Vert
+
\beta_2\text{length}_i
+
u_{\text{protein }i}
+
\gamma\,\text{biological label}_{ip}
+
\epsilon_{ip}.
\]

The confirmatory question should be whether \(\gamma\) improves held-out prediction after low-level covariates have been conditioned out.

Recommended controls include:

- protein-blocked and family-blocked cross-validation;
- conditional randomization within protein;
- matched label prevalence;
- identical folds and dimensionality controls for sparse and dense representations;
- prespecified label families;
- correction over features, layers and labels;
- bootstrap confidence intervals at the protein level;
- negative labels and randomized dictionaries;
- testing unique variance rather than only marginal association.

This would turn “low-level confounding is common” into a quantitative result.

---

## 3.6 N-terminal readouts and received attention

T011, T018 and T023 are convincingly N-terminal readouts: each selected the first two residues in 100 different proteins and is enriched for initiator-methionine contexts. Their unnormalized received-attention correlations are 0.921, 0.914 and 0.885. fileciteturn8file18

The limitation is fundamental: in a causal decoder, early key positions are available to more future queries than later positions. Raw received attention is therefore partly a deterministic function of position.

### Necessary controls

- Divide received attention by the number of eligible causal queries.
- Compare against proteins matched on length and normalized position.
- Match N-terminal features to control features with similar firing frequency and input norm.
- Replace the initial methionine while retaining the rest of the sequence.
- Insert the same methionine-containing motif at internal positions.
- Compare true starts with artificial sequence truncations.
- Separate BOS-token effects, initiator-methionine effects and general early-position effects.
- Evaluate whether changing the N-terminal residues changes the feature before any attention intervention.
- Test whether attention-mediated effects remain after conditioning on position.

A particularly informative counterfactual design would compare:

\[
\begin{array}{ll}
\text{natural N-terminus} & \text{MXX}\ldots\\
\text{amino-acid counterfactual} & \text{AXX}\ldots\\
\text{position counterfactual} & \ldots\text{MXX}\ldots\\
\text{truncation counterfactual} & \text{MXX}\ldots\text{ from an internal segment}.
\end{array}
\]

Only after these controls should the term “attention sink” be considered.

---

## 3.7 Feature and attention interventions

The manuscript tests CLT-component patching, single-head ablation, top-8 and top-32 head sets and an attention-output sparse pilot. No test meets the joint direction, feature-damage and matched-control gate. fileciteturn7file0

This is useful negative evidence, but it is not yet a clean causal falsification because:

- feature/head discovery and evaluation overlap;
- random controls are primarily layer-matched, not fully activation-, norm- or attention-mass-matched;
- the CLT feature is encoded at one tensor but intervention is applied to the CLT-explained MLP-output component;
- no positive-control circuit establishes that the intervention protocol could detect a mechanism of realistic size;
- only a limited set of intervention strengths and sites was tested.

### Required causal redesign

Use a held-out intervention set and report a complete dose–site response surface:

\[
\Delta y = f(\text{feature},\text{layer},\text{site},\text{strength},\text{model}).
\]

Controls should be matched on:

- layer;
- activation frequency;
- mean activation;
- decoder norm;
- direct logit effect;
- attention mass;
- reconstruction contribution.

Each intervention should report:

1. Intended feature change.
2. Off-target sparse-code changes.
3. Reconstruction displacement.
4. Downstream logit displacement.
5. Sequence-level behavioral effect.
6. Difference from matched controls.

Path patching or causal mediation should then test:

\[
\text{sparse readout}
\rightarrow
\text{head/MLP computation}
\rightarrow
\text{token logits}
\rightarrow
\text{sequence property}.
\]

---

## 3.8 Negative results require equivalence testing

Most current conclusions rely on failure to pass a positive gate. This is not equivalent to evidence that effects are negligible.

Before rerunning, define a smallest effect size of scientific interest, for example:

- \(\Delta\mathrm{NLL}\) of 0.05 or 0.1 nat over a prespecified window;
- a five-percentage-point change in validated class probability;
- a prespecified change in target-feature activation;
- a minimum odds ratio for Pfam or EC classification.

Then use:

- two one-sided equivalence tests;
- Bayesian ROPE analysis;
- or confidence intervals interpreted against the prespecified equivalence region.

A robust negative claim would read:

> Across five seeds and held-out proteins, the 95% interval excludes effects larger than the prespecified meaningful threshold.

That would be substantially stronger than \(P>0.05\).

---

## 3.9 Enzyme-class steering

The eight-class benchmark generated 100 steered and 100 unsteered sequences per class, but the endpoint is a motif/composition heuristic rather than a validated EC classifier. No class passes the positive gate. fileciteturn8file3

More importantly, the audit reports that the fallback selector included 19 negative-attribution features across six class/layer cells. Thus, some interventions nominally intended to amplify positive direct effects amplified features with the opposite estimated effect. The result is not a clean eight-class positive-direction negative.

### Mandatory rerun

- Require every selected feature to have positive held-out direct effect.
- Select features on a separate sequence set.
- Package the full attribution table.
- Use identical seeds or paired random-number streams for steered and control generations where appropriate.
- Preserve every generated sequence and all generation metadata.
- Evaluate all classes with generation-wide endpoints.
- Correct for eight classes and any multiplier/site search.
- Include a dose sweep and a prompt-only baseline.
- Add a random-feature and norm-matched feature baseline.

The current result may remain as a transparent pilot, but should not be the principal steering experiment.

---

## 3.10 External lysozyme evaluation

The generation-wide sequence endpoints show no clear advantage:

- Pfam-like hit rate: 0.860 steered versus 0.820 unsteered;
- CLEAN exact EC: 0.775 versus 0.775.

The structure comparison uses ten selected steered leads and twenty unsteered sequences, and the displayed structures are post-selected. fileciteturn8file3turn8file12

The audit additionally reports that Figure 5 combines structures and aggregate results from different historical runs. That is a provenance problem, not merely a caption issue.

### Required replacement

Figure 5 should be rebuilt from one immutable experiment with:

- equal numbers of steered and unsteered sequences;
- identical selection criteria;
- generation-wide or randomly sampled structure prediction;
- one ESMFold version and parameter set;
- one Foldseek database revision;
- no cross-run mixing;
- blinded rank selection;
- full distributions rather than selected means;
- confidence intervals and paired or randomized comparisons.

Pfam, CLEAN, ESMFold and Foldseek are also correlated computational endpoints. Fold similarity is not enzyme activity, and CLEAN/Pfam inherit database and model biases. Johnson et al. showed why computational metrics should be calibrated against measured protein activity rather than treated as interchangeable evidence. ([nature.com](https://www.nature.com/articles/s41587-024-02214-2))

A small randomized wet-lab experiment would therefore have disproportionate value. Even one enzyme family with expression, soluble fraction and activity measured for balanced steered and control sets would materially change the paper’s standing.

---

## 3.11 Representation recoverability

The recoverability analysis asks whether sparse codes preserve signals linearly decodable from their own architecture-specific CLT inputs. This is a reasonable and useful diagnostic. The strongest results are Pfam and residue-secondary-structure recovery, whereas family-disjoint EC recovery is weak for ZymCTRL and moderate for ProGen2-medium. fileciteturn7file13

The word “ceiling” should remain carefully qualified because the CLT input is:

- one selected hook point;
- architecture-specific;
- not the raw residual stream;
- not a model-wide information ceiling.

The analysis remains exploratory because:

- the preregistered five-seed repetition was not completed;
- best-layer selection and assessment are not fully nested;
- multiplicity across layers is incomplete;
- the decoder-native EC set has only 48 sequences;
- the basis-comparison implementation described in the audit does not fully match the stronger family-aware wording.

### Required redesign

- Nested cross-validation for layer selection.
- Identical folds across CLT input, sparse code, reconstruction and dense baselines.
- Repeated seeds.
- Paired bootstrap intervals for \(F-C\), not only ratios.
- Larger decoder-native cohorts.
- Identity-aware splits.
- PCA, random projection, NMF/ICA and random-dictionary controls at matched dimensions.
- Tasks involving contacts, motifs, active sites and structural neighborhoods, not only global labels.
- Analysis of whether reconstruction error predicts recoverability and intervention efficacy.

---

## 3.12 Reproducibility and release readiness

The main manuscript states that the original 200-sequence atlas cohort, CLT checkpoints and full generation collections are not yet in a persistent archive. fileciteturn8file0

This is a hard submission blocker. Nature Portfolio policy states that readers must be able to replicate and build upon published claims, and that the materials, data, code and protocols needed for those claims must be made available. The *npj Artificial Intelligence* guide also requires transparent access conditions for the minimum dataset needed to interpret, verify and extend the work. ([nature.com](https://www.nature.com/nature-portfolio/editorial-policies/reporting-standards))

Before submission, the archive needs:

- exact upstream model revisions;
- tokenizer revisions;
- checkpoint hashes;
- CLT checkpoints;
- all executed configurations;
- the exact ordered 200-sequence cohort and hash;
- discovery and held-out cohorts;
- complete generated sequence collections;
- tool and database versions;
- raw intervention outputs;
- strict JSON without non-finite values;
- commit/tag identifiers;
- a DOI-backed repository release;
- a machine-readable provenance manifest linking every figure and table to inputs and code.

A checksum manifest for processed tables is helpful, but it does not replace missing raw inputs or checkpoints.

---

## 3.13 Writing and figure quality

### Strong elements

- Figure 1 is conceptually excellent.
- Figure 4 reports the complete intervention matrix and appropriately emphasizes effect sizes.
- The manuscript repeatedly distinguishes exploratory from confirmatory evidence.
- The Discussion does not overstate the negative findings.

### Necessary changes

- **Figure 2:** add cohort, seed, matcher and threshold stability; the current bar chart visually implies more universality than has been established.
- **Figure 3:** add position-normalized attention and counterfactual controls; otherwise the attention panel may be interpreted mechanistically despite the caveat.
- **Figure 4:** add equivalence bands and intervention-fidelity panels.
- **Figure 5:** replace entirely with a single-run, symmetric, generation-wide analysis.
- **Figure 6:** show uncertainty, nested-selection performance and quality–recoverability relationships.
- Move selected attractive protein structures to the Supplement unless the selection is prospectively defined.

The manuscript has approximately the right Article-scale structure and number of main figures. The journal’s guideline is up to ten display items and roughly 4,000–4,500 main-text words, with around sixty references as a nonbinding guideline. ([nature.com](https://www.nature.com/npjai/content-types)) The current bibliography should nevertheless be updated substantially with the 2025–2026 literature reviewed above.

---

# 4. Submission-readiness scorecard

The scores below are independent expert judgments, not journal acceptance probabilities.

| Criterion | Score / 5 | Assessment |
|---|---:|---|
| Fit to *npj Artificial Intelligence* | **4.5** | Strong fit to AI interpretability, scientific AI and interdisciplinary methodological evaluation. |
| Workload and technical breadth | **4.5** | More than sufficient volume for an Article. |
| Conceptual framing | **4.0** | The recurrence–association–causality distinction is strong and timely. |
| Methodological novelty | **3.5** | Cross-model conserved CLT readouts are novel, but the matching procedure is not yet a broadly validated method. |
| Statistical rigor | **2.0** | Several tests are exploratory, under-replicated or lack correct experimental units and equivalence analysis. |
| Dictionary and representation validity | **2.0** | High dead fractions, padding exposure and unmatched quality comparisons undermine downstream inference. |
| Causal evidence | **1.5** | All tests are negative, but pipeline sensitivity and clean intervention validity have not been established. |
| Biological validation | **1.5** | Mostly computational endpoints, with selected/asymmetric structure evaluation and no activity assay. |
| Reproducibility | **1.5** | Missing central cohort identity, revisions and checkpoints are submission-critical. |
| Writing and claim calibration | **4.0** | Unusually candid and generally well bounded. |
| **Overall current submission readiness** | **2.0** | Strong research package, not yet a submission-ready Article. |

---

# 5. Conclusion 2 — What must be added?

## 5.1 Priority-zero experiments required before submission

| Priority | Required work | Minimum satisfactory outcome |
|---|---|---|
| **P0-1: Exact reproducibility** | Reconstruct and hash the canonical cohort; archive model revisions, tokenizers, CLTs, configs, raw generations and outputs; create DOI release. | A third party can regenerate every main figure and table from immutable inputs. |
| **P0-2: Mask-aware dictionaries** | Retrain and evaluate with masks, held-out sequences and multiple seeds. | Quality-gated dictionaries with seed-level FVU/dead statistics and no padding contamination. |
| **P0-3: Independent atlas replication** | Discovery/test cohorts, coherent model-wise null, at least 1,000 null replicates, multiple matchers and signed-correlation analysis. | Conserved correspondences retain meaningful held-out correlation and stability across cohorts and methods. |
| **P0-4: Semantic conditional tests** | Continuous activations, protein-blocked and family-blocked tests, conditioning on k-mer, position, norm, length and source. | Either a reproducible residual biological association or a powered bound showing that such associations are small. |
| **P0-5: N-terminal counterfactuals** | Position-normalized attention, motif and position swaps, matched controls and held-out proteins. | Clear separation—or demonstrated nonseparation—of methionine, position and attention effects. |
| **P0-6: Correct steering rerun** | Positive-only selector, independent feature selection, immutable full generations and validated generation-wide endpoints. | A clean positive or clean equivalence-bounded negative across the eight classes. |
| **P0-7: Causal intervention redesign** | Held-out discovery/evaluation, matched controls, dose/site sweeps, positive controls and equivalence thresholds. | Evidence that the intervention system can recover a known mechanism and can meaningfully bound effects for the target features. |
| **P0-8: Recoverability replication** | Nested cross-validation, repeated seeds, identical folds and matched-dimensional baselines. | Paired confidence intervals demonstrating which signals are reliably retained and under which dictionary-quality conditions. |
| **P0-9: Figure and chronology rebuild** | Recreate Figure 5 from one run; add provenance mapping and correct all historical inconsistencies. | Every displayed panel is traceable to one documented experiment and version set. |

## 5.2 The most important missing experiment: a positive control

At present, all high-level interventions fail. Without a positive control, it is impossible to know whether the pipeline is appropriately falsifying the hypothesized mechanism or is simply insensitive.

A particularly strong addition would be a **planted-ground-truth biological language-model benchmark**:

1. Construct a synthetic protein grammar with known causal variables:
   - motif identity;
   - motif position;
   - length;
   - family label;
   - long-range dependency;
   - controlled N-terminal token.
2. Train small autoregressive models with several seeds.
3. Optionally insert a known adapter or sparse direction that causally controls one property.
4. Train the same CLT/SAE pipeline.
5. Ask whether the procedure:
   - recovers the planted feature;
   - aligns it across models;
   - distinguishes semantics from correlated position/k-mer features;
   - localizes the causal path;
   - successfully steers the controlled endpoint.
6. Quantify sensitivity, specificity, false-discovery rate and intervention effect recovery.

This would transform the paper from an audit of one unsuccessful pipeline into a validated benchmark for evaluating interpretability pipelines.

---

# 6. High-novelty research directions

## 6.1 Evidence-Calibrated Mechanistic Interpretability Benchmark

The strongest new direction is to generalize the current Figure 1 into a benchmark with four formal estimands:

\[
\begin{aligned}
R &: \text{cross-model recurrence},\\
S &: \text{semantic specificity beyond confounders},\\
C &: \text{causal effect on model computation},\\
U &: \text{downstream design utility}.
\end{aligned}
\]

Each should have independent datasets, nulls, controls and acceptance thresholds. The benchmark could output an “evidence card” for every feature:

| Field | Example |
|---|---|
| Dictionary quality | FVU, dead rate, reconstruction contribution |
| Stability | seed/cohort/matcher confidence |
| Low-level confounds | position, k-mer, norm, token boundary |
| Semantic residual evidence | held-out conditional effect |
| Intervention fidelity | target and off-target sparse-code changes |
| Causal effect | equivalence-bounded behavioral effect |
| External utility | validated sequence or functional endpoint |
| Provenance | model, checkpoint, cohort and code hashes |

This is substantially more general and more likely to satisfy the journal’s “reframing research challenges” criterion. ([nature.com](https://www.nature.com/npjai/aims))

## 6.2 Subspace conservation rather than exact feature identity

Exact one-feature-to-one-feature correspondence may be too restrictive because sparse dictionaries are not identifiable at the individual-feature level. A concept may rotate, split or merge across dictionaries.

Compare:

- individual-feature matching;
- sparse subspace matching;
- canonical correlation;
- optimal transport;
- multi-set CCA;
- shared latent-factor models;
- joint dictionary learning.

A key result could be:

> Individual features are unstable, but low-dimensional subspaces carrying a property are conserved.

That would connect directly to the recent identifiability and sparse-coding synthesis. ([nature.com](https://www.nature.com/articles/s42256-026-01259-z))

## 6.3 Architecture–data–tokenizer decomposition

Train or select a panel that allows the following contrasts:

- same architecture, different corpus;
- same corpus, different architecture;
- same sequences, different tokenization;
- same setup, different random seed;
- masked versus autoregressive objective;
- different scales within one family.

A variance-components analysis could then estimate how much alignment is attributable to each factor.

## 6.4 Prospective checkpoint diagnostic

The mature-versus-10k comparison—38 versus 16 triplets—is intriguing but currently anecdotal. fileciteturn8file18

Track:

- reconstruction quality;
- dead fraction;
- CKA;
- triplet count;
- held-out semantic residual evidence;
- causal intervention sensitivity;

over many checkpoints and seeds.

Then test prospectively whether early triplet statistics predict later:

- dictionary quality;
- downstream task performance;
- calibration;
- generation quality;
- robustness.

This could become a genuine AI-training diagnostic rather than a post-hoc observation.

## 6.5 Causal pathway tracing in protein decoders

For a validated motif or property:

\[
\text{input residues}
\rightarrow
\text{sparse feature}
\rightarrow
\text{attention/MLP pathway}
\rightarrow
\text{logits}
\rightarrow
\text{generated motif or structure}.
\]

Combine:

- feature attribution;
- activation patching;
- path patching;
- sparse mediation;
- head/MLP interventions;
- controlled sequence generation.

A positive pathway would considerably strengthen the paper. A carefully bounded failure, after demonstrating positive-control sensitivity, would also be publishable.

## 6.6 Uncertainty-aware interpretability

For every semantic label or feature name, report:

- probability of feature stability;
- uncertainty across seeds;
- uncertainty across cohorts;
- matcher ambiguity;
- label uncertainty;
- causal-effect uncertainty.

This would align the study with recent emphasis on uncertainty across protein representations. ([nature.com](https://www.nature.com/articles/s41592-026-03028-7))

## 6.7 Closed-loop functional validation

A high-impact biological extension would select one narrow, experimentally tractable endpoint:

- lysozyme activity;
- fluorescence;
- thermostability;
- binding;
- soluble expression.

Use a randomized and blinded design with equal steered and control sets. Even a modest experiment could determine whether the internal intervention changes a real property, rather than another learned model’s score.

Wet-lab validation is not an absolute requirement for an *npj Artificial Intelligence* methods paper, but it becomes close to essential when the manuscript discusses protein design utility.

---

# 7. Recommended visualization work

## Revised main-figure plan

| Figure | Proposed content | Scientific purpose |
|---|---|---|
| **Figure 1 — Evidence benchmark** | Retain the recurrence–association–intervention ladder, but add formal estimands, independent datasets and acceptance gates. | Makes the general AI-method contribution explicit. |
| **Figure 2 — Stability landscape** | Heat maps or phase diagrams of triplet count and identity Jaccard across cohorts, seeds, matchers, layer maps, pool sizes and thresholds. | Shows whether conservation is a stable property or a procedural outcome. |
| **Figure 3 — Conditional semantics** | For each triplet, marginal association versus association remaining after conditioning on position, k-mer, norm, length and protein identity. | Separates nameable correlations from unique biological information. |
| **Figure 4 — N-terminal counterfactuals** | Natural starts, methionine substitutions, motif shifts and truncation controls; causal-opportunity-normalized attention. | Distinguishes motif, position and attention explanations. |
| **Figure 5 — Intervention forest and dose response** | Every target and matched control with equivalence bands, feature damage, off-target effects and multiple strengths/sites. | Converts null-significance testing into calibrated causal evidence. |
| **Figure 6 — Corrected steering** | Full generation-wide distributions for validated class scores, novelty, diversity, model likelihood and structural metrics; equal sampling. | Provides a clean steering conclusion. |
| **Figure 7 — Dictionary quality and recoverability** | Seed-level FVU/dead fraction versus recoverability and intervention effect; nested-CV uncertainty. | Tests whether failed interventions are predicted by dictionary quality. |
| **Figure 8 — Prospective training trajectory** | Checkpoint curves for CKA, triplet count, reconstruction and downstream behavior across seeds. | Supports the proposed checkpoint-diagnostic contribution. |

## Supplementary visualizations

- Alluvial diagrams showing feature alignments across matchers and cohorts.
- Signed-correlation distributions, not only \(|r|\).
- Null-distribution histograms with observed statistics.
- Protein-blocked permutation distributions.
- Feature firing-frequency and decoder-norm distributions.
- Off-target intervention matrices.
- EC-class confusion and calibration plots.
- ECDF or violin plots for all generated-sequence metrics.
- Power and equivalence plots.
- A provenance DAG connecting each displayed result to checkpoint, cohort, code commit and tool/database version.
- A claim–threat–control–status matrix.

---

# 8. Recommended revised scientific story

A stronger title would be:

> **Conservation is not mechanism: a prospective benchmark for semantic and causal validation of sparse features in protein language models**

or:

> **Evidence-calibrated auditing of conserved sparse representations in biological foundation models**

The revised paper should make three primary claims:

1. **Cross-model sparse-feature recurrence is measurable but depends on cohort, dictionary quality and matching procedure.**
2. **Most apparent semantics can be decomposed into low-level sequence, position and activation-scale contributions; residual biological semantics require conditional held-out tests.**
3. **Causal conclusions require intervention-fidelity checks, positive controls and equivalence-bounded effects; feature recurrence alone is not sufficient.**

The current N-terminal and steering results would then become case studies within a general benchmark rather than carrying the full novelty burden.

---

# 9. Practical decision paths

## Computational-only route to *npj Artificial Intelligence*

Submission becomes realistically defensible without wet-lab work if the revised study includes:

- exact public reproducibility;
- mask-aware, multi-seed quality-gated dictionaries;
- independent cohort replication;
- a coherent and adequately sampled null;
- multiple matching algorithms;
- a planted positive-control benchmark;
- corrected eight-class steering;
- held-out causal tests;
- equivalence-bounded negatives;
- nested recoverability analysis.

This route should emphasize general interpretability methodology rather than protein design efficacy.

## Higher-impact biological route

Add all computational corrections plus:

- a validated mechanistic pathway for at least one feature;
- symmetric generation-wide biological evaluation;
- one blinded functional assay.

That version could support stronger claims about AI-guided biological design and would be competitive with the evidential standards visible in recent *Nature Methods*, *Nature Biotechnology* and *Nature Communications* studies.

## Submission strategy

After the priority-zero work is complete, a pre-submission inquiry is appropriate. *npj Artificial Intelligence* explicitly invites such inquiries when suitability is uncertain. ([nature.com](https://www.nature.com/npjai/aims)) The inquiry should include:

- a one-paragraph general AI contribution;
- the benchmark design;
- the cross-cohort stability figure;
- the positive-control result;
- the corrected intervention/steering figure;
- the public release DOI.

Submitting an inquiry before those corrections would probably elicit the same concerns identified here.

---

# Final answers

## 1. Is the current manuscript eligible for submission to *npj Artificial Intelligence*?

**Topically yes; scientifically not yet.**

The workload, breadth and framing are sufficient. The present evidence is not. I would classify the manuscript as a strong, unusually candid internal audit that requires substantial confirmatory reruns before submission. The largest blockers are cohort-sensitive recurrence, the non-coherent null, dictionary-quality and padding problems, the steering sign-selection defect, non-independent discovery/evaluation, mixed-run visualization and incomplete exact reproducibility.

## 2. What must be added?

The indispensable additions are:

1. Exact DOI-backed reproduction package.
2. Mask-aware multi-seed dictionary retraining and quality gating.
3. Independent cohorts, coherent nulls and multiple alignment methods.
4. Continuous, protein-blocked conditional semantic testing.
5. Position-normalized and counterfactual N-terminal experiments.
6. Positive-control validation of the causal pipeline.
7. Corrected steering with validated, symmetric endpoints.
8. Held-out, matched, dose–response interventions with equivalence bounds.
9. Nested, repeated recoverability analysis.
10. Stability, causal-effect, full-distribution and provenance visualizations.

The most novel and strategically valuable extension is to convert the work into an **evidence-calibrated benchmark that measures when sparse-feature recurrence does—and does not—support semantic, causal and design claims**.

### Reviewed attachments

[Experimental audit](sandbox:/mnt/data/EXPERIMENTAL_PROCEDURES_AND_RESULTS_AUDIT_20260716.md) · [Main manuscript PDF](sandbox:/mnt/data/main(2).pdf) · [Main manuscript TeX](sandbox:/mnt/data/main(3).tex) · [Supplementary PDF](sandbox:/mnt/data/supplementary_information.pdf) · [Supplementary TeX](sandbox:/mnt/data/supplementary_information.tex) · [Bibliography file](sandbox:/mnt/data/main(1).bbl)
